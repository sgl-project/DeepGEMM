"""Bounded SM100 cleanup regression; includes endpoints beyond the output allocation.

Run with PYTORCH_NO_CUDA_MEMORY_CACHING=1 under compute-sanitizer --tool memcheck
so allocator slab padding cannot hide writes beyond the last output row.
"""

import torch

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import per_custom_dims_cast_to_fp8


def test_clean_logits_bounds():
    if torch.cuda.get_device_capability()[0] != 10:
        print('Skipping SM100 separate-cleanup regression on this architecture')
        return

    torch.manual_seed(910)
    # At least one Q block per SM selects the long dense FP8/BF16 cleanup path.
    m, n, h, d = 4 * deep_gemm.get_num_sms() + 4, 16384, 32, 128
    q = torch.randn((m, h, d), device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    kv = per_custom_dims_cast_to_fp8(
        torch.randn((n, d), device='cuda', dtype=torch.bfloat16), (0,), False)
    weights = torch.randn((m, h), device='cuda', dtype=torch.float32)
    starts = torch.zeros(m, device='cuda', dtype=torch.int32)
    ends = torch.full((m,), n, device='cuda', dtype=torch.int32)
    columns = torch.arange(n, device='cuda')

    for ks, ke in [(17001, 17003), (0, 17003), (n - 1, 2147483647),
                   (2147483647, 2147483647), (31, n - 17), (n, n)]:
        # The last row makes scalar cleanup beyond stride_logits observable to memcheck.
        starts[-1], ends[-1] = ks, ke
        out = deep_gemm.fp8_fp4_mqa_logits(
            (q, None), kv, weights, starts, ends,
            clean_logits=True, logits_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        valid = (columns >= min(ks, n)) & (columns < min(ke, n))
        assert torch.isneginf(out[-1, ~valid]).all(), (ks, ke)
        assert torch.isfinite(out[-1, valid]).all(), (ks, ke)
        if valid.any():
            # FP32 uses the independent fused-cleaning path with scheduler-clamped bounds.
            reference = deep_gemm.fp8_fp4_mqa_logits(
                (q, None), kv, weights, starts, ends,
                clean_logits=True, logits_dtype=torch.float32)
            diff = calc_diff(out[-1, valid], reference[-1, valid].bfloat16())
            assert diff < 3e-5, (ks, ke, diff)
        print(f'Cleanup endpoint case passed: [{ks}, {ke}), stride={out.stride(0)}')
    print('CLEANER ENDPOINT REGRESSION PASS')


if __name__ == '__main__':
    test_clean_logits_bounds()
