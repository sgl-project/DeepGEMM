# Paired BF16 paged MQA producer

The SM100-only `fp4_paged_mqa_logits_bf16` API computes MXFP4 MQA scores with
BF16 head weights and BF16 output. It optionally counts every live non-NaN
output score exactly once in a 1024-bin histogram. Adding the histogram does
not change the logits. This is a separate API; existing MQA APIs and their
default FP32 scores are unchanged.

```python
schedule = deep_gemm.get_paged_mqa_logits_bf16_metadata(
    context_lens, 128, deep_gemm.get_num_sms(),
    indices=request_ids, tokens_per_request=n,
)
histogram = torch.zeros((rows, 1024), dtype=torch.int32, device="cuda")
scores = deep_gemm.fp4_paged_mqa_logits_bf16(
    (q_fp4, q_sf), kv_cache, weights, context_lens, block_table,
    schedule, max_context_len,
    indices=request_ids, histogram=histogram, tokens_per_request=n,
)
```

Input contract:

- Q is packed MXFP4 int8 `[rows, 1, 32, 64]`; Q scales are packed UE8M0
  int32 `[rows, 1, 32]`. Both are contiguous.
- KV is the existing page layout, uint8 `[pages, 128, 1, 68]`: each page
  stores all 128 packed 64-byte payloads, then all 128 four-byte scales.
- Weights are BF16 `[rows, 32]`, with contiguous heads and 16-byte row alignment.
- Causal lengths are contiguous int32 `[rows, 1]`, each in
  `[0, max_context_len]`. Pages are int32 `[rows, ceil(max_context_len / 128)]`
  or wider, with contiguous columns.
- Optional request IDs are contiguous int32 `[rows]`. Only adjacent equal
  IDs share KV loads; their page-table rows must be identical and lengths
  nondecreasing. Unequal-length runs and repeated IDs in nonadjacent runs are
  supported. Omitting IDs treats every row independently.
- `tokens_per_request` is a CPU hint in 1..6, not a grouping operation.
  Hints 1..4 select Q4 with three TMEM stages; 5/6 select Q6 with two TMEM
  stages. Hint 6 also swizzles the histogram's shared-memory addresses.
  Always use the same hint, IDs and lengths for schedule and producer.
- Schedule generation supports up to 16384 rows. Use `get_num_sms()` for
  its SM count. The returned int32 `[num_sms + 1, 2]` schedule uses 384-key
  splits and is incompatible with the generic MQA schedule.

The result is BF16 `[rows, max_context_len]`, with stride rounded up to 1536
elements. Only columns below each row's causal length contain defined scores;
consumers must mask the other columns. Every tensor is on the same CUDA device.
Tensor geometry is checked on the host; values, page IDs and within-run
relationships are the caller's responsibility.

`histogram`, when provided, is contiguous int32 `[rows, 1024]`. Counts are
**added** to its current contents. Zero it before a standalone call, or use a
selector that resets it after consuming it. The coarse-bin encoding is defined
in `deep_gemm/mqa_bf16/histogram.cuh`; bins run in descending score order,
canonicalize signed zero, and exclude NaNs. It preserves the paired SGLang
BF16 selector's encoding. No output scores are dropped or replaced by block
maxima.

For CUDA graphs, warm up the API first and capture schedule generation before
the producer. Replay then reads live lengths and request IDs. Keep shapes,
addresses and CPU hint fixed for a captured graph.

The paired SGLang selector returns the exact top-512 of these BF16 scores,
preferring lower physical slots for ties and padding with `-1`. BF16 rounding
can change selected slots relative to the existing FP32 score path.

Run the GPU test with `python -m pytest sgl_deep_gemm/tests/test_mqa_bf16.py`.
