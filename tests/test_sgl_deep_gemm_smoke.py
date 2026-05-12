# Smoke test for the sgl-deep-gemm wheel + tvm-ffi wrapping of the
# w4a4-related additions (mega_moe_pre_dispatch and the FP4/MXF4 env
# surface threaded through fp8_fp4_mega_moe). Single-GPU only.

import os
import sys
import torch
import deep_gemm


def _check_public_symbols():
    needed = [
        "mega_moe_pre_dispatch",
        "fp8_fp4_mega_moe",
        "get_symm_buffer_for_mega_moe",
        "transform_weights_for_mega_moe",
        "SymmBuffer",
    ]
    missing = [n for n in needed if not hasattr(deep_gemm, n)]
    if missing:
        raise AssertionError("missing public symbols: " + str(missing))
    if not hasattr(deep_gemm._C, "mega_moe_pre_dispatch"):
        raise AssertionError("tvm-ffi binding missing for mega_moe_pre_dispatch")


def _check_pre_dispatch_smoke():
    # Tiny end-to-end pass through the tvm-ffi wrapper.
    M, P, H, K, G = 8, 16, 256, 4, 32
    x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
    topk_idx = torch.zeros(M, K, dtype=torch.int32, device="cuda")
    topk_weights = torch.randn(M, K, dtype=torch.float32, device="cuda")

    buf_x = torch.zeros(P, H, dtype=torch.float8_e4m3fn, device="cuda")
    num_groups = H // G
    buf_x_sf = torch.zeros(P, num_groups // 4, dtype=torch.int32, device="cuda")
    buf_topk_idx = torch.zeros(P, K, dtype=torch.int64, device="cuda")
    buf_topk_weights = torch.zeros(P, K, dtype=torch.float32, device="cuda")

    deep_gemm.mega_moe_pre_dispatch(
        x, topk_idx, topk_weights,
        buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights,
        num_tokens=M, group_size=G, use_fp4_acts=False,
    )
    torch.cuda.synchronize()

    # Pad rows must be (-1, 0).
    assert torch.all(buf_topk_idx[M:] == -1).item()
    assert torch.all(buf_topk_weights[M:] == 0.0).item()
    # Valid rows pass through (int32 -> int64 widening).
    assert torch.equal(buf_topk_idx[:M], topk_idx.to(torch.int64))


def main() -> int:
    _check_public_symbols()
    print("PASS public symbols")

    if not torch.cuda.is_available():
        print("SKIP runtime: no CUDA")
        return 0
    cc_major, _ = torch.cuda.get_device_capability()
    if cc_major != 10:
        print(f"SKIP runtime: kernel needs SM100, device is sm{cc_major}x")
        return 0

    _check_pre_dispatch_smoke()
    print("PASS pre_dispatch smoke")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
