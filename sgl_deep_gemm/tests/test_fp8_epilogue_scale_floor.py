"""Custom FP8 absmax floors preserve tiny values when fusing a quantizer."""

from itertools import product
import torch

import deep_gemm


def quantize_reference(x, eps):
    m, k = x.shape
    grouped = x.float().view(m, k // 32, 32)
    scale = torch.pow(
        2.0, torch.ceil(torch.log2(grouped.abs().amax(-1).clamp_min(eps) / 448.0))
    )
    q = (grouped / scale[..., None]).to(torch.float8_e4m3fn).view(m, k)
    exponents = (torch.log2(scale) + 127).to(torch.uint8)
    return q, exponents.contiguous().view(torch.int32)


def _check_fp8_epilogue_scale_floor(rows, eps, clamp):
    torch.manual_seed(42)
    # Identity-like weights make the contraction exact, including zeros,
    # tiny nonzero values and values on both sides of the scale floor.
    groups, k, n = 2, 512, 512
    x = torch.randn(rows, groups, k, device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    x[1].mul_(1e-8)
    x[2].mul_(1e-11)
    from deep_gemm.utils import per_token_cast_to_fp8

    xq, xs = per_token_cast_to_fp8(x.view(-1, k), use_ue8m0=True)
    xq, xs = xq.view(rows, groups, k), xs.view(rows, groups, -1)
    wq = torch.eye(k, device="cuda").to(torch.float8_e4m3fn).repeat(groups, 1, 1)
    ws = torch.ones(groups, n // 128, k // 128, device="cuda")
    ws[1].mul_(2.0**-20)
    bf16 = torch.empty(rows, groups, n, device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_einsum("bhr,hdr->bhd", (xq, xs), (wq, ws), bf16)
    expected = bf16.flatten(1)
    if clamp is not None:
        expected = expected.clamp(-clamp, clamp)
    rq, rs = quantize_reference(expected, eps)
    output = torch.empty_like(bf16, dtype=torch.float8_e4m3fn)
    scales = torch.empty_strided(
        (rows, groups * n // 128),
        (1, (rows + 3) // 4 * 4),
        device="cuda",
        dtype=torch.int32,
    )
    count = torch.empty(1, device="cuda", dtype=torch.uint32)
    epilogue = deep_gemm.epilogue.FP8CastWithClamp(
        scales, clamp_value=clamp, clamped_count=count, amax_eps=eps
    )
    deep_gemm.fp8_einsum("bhr,hdr->bhd", (xq, xs), (wq, ws), output, epilogue=epilogue)
    torch.testing.assert_close(
        output.flatten(1).view(torch.uint8), rq.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(scales, rs, rtol=0, atol=0)
    if eps == 1e-4 and clamp is None:
        legacy_out = torch.empty_like(output)
        legacy_sf = torch.empty_strided(
            scales.shape, scales.stride(), device="cuda", dtype=torch.int32
        )
        # The original four-field torch.ops epilogue encoding remains valid.
        torch.ops.deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            xq,
            xs,
            wq,
            ws,
            legacy_out,
            legacy_sf,
            None,
            [1, 128, 128],
            0,
            None,
            None,
            None,
        )
        torch.testing.assert_close(
            legacy_out.view(torch.uint8), output.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(legacy_sf, scales, rtol=0, atol=0)
    assert count.item() == (0 if clamp is None else (bf16.abs() > clamp).sum().item())


def _check_invalid_floor_rejected(eps):
    x = torch.zeros(1, 1, 512, device="cuda", dtype=torch.float8_e4m3fn)
    xs = torch.ones(1, 1, 4, device="cuda")
    w = torch.zeros(1, 512, 512, device="cuda", dtype=torch.float8_e4m3fn)
    ws = torch.ones(1, 4, 4, device="cuda")
    out = torch.empty_like(x)
    sf = torch.empty_strided((1, 4), (1, 4), device="cuda", dtype=torch.int32)
    try:
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (x, xs),
            (w, ws),
            out,
            epilogue=deep_gemm.epilogue.FP8CastWithClamp(sf, amax_eps=eps),
        )

    except (RuntimeError, ValueError):
        return
    raise AssertionError(f"Invalid absmax floor was accepted: {eps}")


def test_fp8_epilogue_scale_floor():
    if torch.cuda.get_device_capability()[0] != 10:
        return
    for rows, eps, clamp in product([3, 65, 513], [1e-4, 1e-10], [None, 0.5]):
        _check_fp8_epilogue_scale_floor(rows, eps, clamp)
    for eps in [0.0, -1.0, float("nan"), float("inf"), 1e-45]:
        _check_invalid_floor_rejected(eps)


def test_other_epilogue_encodings():
    if torch.cuda.get_device_capability()[0] != 10:
        return
    x = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
    weight = torch.eye(512, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    for epilogue, expected in [
        (None, x),
        (deep_gemm.epilogue.Identity(), x),
        (deep_gemm.epilogue.Alpha(0.5), x * 0.5),
    ]:
        deep_gemm.bf16_gemm_nt(x, weight, out, epilogue=epilogue)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    test_fp8_epilogue_scale_floor()
    test_other_epilogue_encodings()
    print("FP8 scale-floor and epilogue compatibility tests passed")
