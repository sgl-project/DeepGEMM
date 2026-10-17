
## Introduction

sgl-deep-gemm is a pypi package built from SGLang's customized branch of DeepGemm. Comparing with origina DeepGemm, it supports the following features to better support SGLang:
1. ABI support: ops are registered with `TORCH_LIBRARY` and the extension has no CPython dependency, so a single wheel runs on every Python version (it is still built against one PyTorch version).
2. pypi support: easy installation with `pip install sgl-deep-gemm`. No need to manually search for wheel links.
3. Fast iteration: add custom kernels and bump versions at no time.

## Usage
Build requirements include CUDA 12.9 or newer, a C++20 compiler with `std::format` support (for example GCC 13), and the elfutils development headers (`apt-get install libdw-dev` on Debian/Ubuntu). Initialize the CUTLASS and DeepJIT submodules with `git submodule update --init --recursive`.

To build it locally, run `bash build_sgl_deep_gemm.sh`, then pip install the wheel generated under `dist`.

K-grouped FP8 GEMM and its scale-packing helper preserve compact per-group scaling factors by default. Pass `use_padded_sf_layout=True` to both packing and GEMM when packed scales cover each group's padded K extent. The flag is appended to existing arguments, so existing positional calls retain their behavior.

SM100 Mega MoE sizes `BLOCK_M` from the expected tokens per expert, assuming every EP rank sends as many tokens as the local rank. Under attention data parallelism the ranks can be uneven, and an idle rank would then pick the smallest tile while it still computes the experts other ranks route to it. Pass `global_num_tokens` (the real token count summed over all EP ranks, identical on every rank) to `fp8_fp4_mega_moe`, `nvfp4_mega_moe`, or `bf16_mega_moe` to size tiles from the EP-wide load; pass the same value to `get_block_m_for_mega_moe` when laying out shared-expert scales. The hint only affects tile selection, and omitting it keeps the previous heuristic.

SM100 paged MQA logits can also emit a coarse score histogram for a downstream top-k (LiteTopK decode), so the top-k skips its own histogram pass over the logits. `fp8_fp4_paged_mqa_logits(..., indices=, histogram=)` takes an int32 `[rows, 1024]` tensor (contiguous, 8-byte aligned, on the logits' device) and adds to it the count of every stored (column below the row's context length) non-NaN BF16 logit of each row, in descending order (bin 0 holds the largest): FP16-RN exponent and 4 mantissa bits for `|x| < 16`, unit-width bins up to 223 (lower-inclusive for negatives), one saturating bin at each end, `-0` as `+0` (`deep_gemm/include/deep_gemm/epilogue/coarse_histogram.cuh`). Counts are added to the tensor's contents, so zero it before an independent call (for example a memset node in the same CUDA graph). Logits are bit-identical with and without the histogram. It supports MXFP8 and MXFP4 with 32 or 64 heads; other head counts and SM90 are rejected. With MXFP4 and head dim 128 the histogram variant keeps 7 KV stages instead of 8, since its 16-bit counters take that shared memory.

Run wheel validation with `bash sgl_deep_gemm/run_tests.sh`. The Mega MoE reference comparisons require DeepEP with `ElasticBuffer`; Mega Gate and Mega mHC also require TileKernels and TileLang. Install the validation extras with `pip install 'sgl-deep-gemm[dev]'` (TileLang 0.1.9 and TileKernels 1.0.0). It runs legacy, lazy-init, and compute-sanitizer checks; `--skip-sanitizer` is available for ordinary development runs. Distributed MegaMoE runs include separate Hopper numerical accuracy coverage; the sanitizer's automatic discovery covers single-process tests and reports distributed entrypoints separately.

Run `bash sgl_deep_gemm/run_tests.sh --release` for reduced attention coverage and focused `memcheck`/`synccheck`, targeting a release gate under one hour. Selected cases retain their original assertions; other ordinary and distributed tests remain unchanged. The default command keeps the full suite; the table shows release / full attention case counts.

| GPU architecture | Dense MQA | Paged MQA | Sparse MQA |
| --- | ---: | ---: | ---: |
| Hopper (SM90) | 8 / 192 | 8 / 24 | unsupported |
| Blackwell (SM100/SM103) | 36 / 2304 | 72 / 4320 | 12 / 161 |
| Blackwell (SM120) | 16 / 256 | 16 / 48 | unsupported |

Use Compute Sanitizer 2025.4.1 or newer for Python tests; older releases can retain Python tensors while collecting host backtraces and exhaust GPU memory. Set `COMPUTE_SANITIZER` to select a separately installed executable. See [NVIDIA's release notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#updates-in-2025-4-1).

To release a new set of wheels, please contact SGLang team and run the [release workflow](https://github.com/sgl-project/sglang/actions/workflows/release-whl-deepgemm.yml) under SGLang repo

For each major version release (0.X.Y -> 0.(X+1).0), a new branch should be created (release/v0.(X+1).0) for stability purpose.

For any incoming pull requests, it should be rebased upon `dev` branch. Any newly added or modified tests should be put under `sgl_deep_gemm/tests`
