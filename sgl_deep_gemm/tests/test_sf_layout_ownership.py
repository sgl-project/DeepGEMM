"""Regression test: scale-layout transforms must return owning tensors.

Across the tvm-ffi boundary, inputs are wrapped as non-owning
``torch::from_blob`` views. Fast paths that return their input unchanged used
to export that non-owning alias through DLPack, so once the caller dropped its
last reference the caching allocator could recycle the storage while the
returned tensor still pointed at it (use-after-free, surfacing as NaN weight
scales when loading FP8 checkpoints).

Reported in https://github.com/sgl-project/sglang/issues/39684

Run on one idle SM90+ GPU:

    python -m pytest sgl_deep_gemm/tests/test_sf_layout_ownership.py -v
"""

import gc

import pytest
import torch

import deep_gemm


def _assert_output_owns_storage(make_out, expected_value):
    sf = torch.full((1, 128, 40), expected_value, device="cuda", dtype=torch.float32)
    source_ptr = sf.data_ptr()
    out = make_out(sf)
    if not isinstance(out, torch.Tensor):
        out = torch.from_dlpack(out)
    torch.cuda.synchronize()

    del sf
    gc.collect()
    # Simulate the small allocations made by subsequent model loading/forward
    # work; NaN-filled buffers act as an overwrite sentinel.
    sentinels = [torch.full((1, 128, 40), float("nan"), device="cuda") for _ in range(20)]
    torch.cuda.synchronize()

    assert torch.isfinite(out).all().item(), (
        "live output aliases recycled scale storage "
        f"(source ptr reused: {any(x.data_ptr() == source_ptr for x in sentinels)})"
    )
    assert torch.allclose(out, torch.full_like(out, expected_value))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_transform_sf_into_required_layout_owns_storage():
    # (FP32, 128, 128) recipe on SM90 hits the check-only pass-through path,
    # which returns the input itself.
    _assert_output_owns_storage(
        lambda sf: deep_gemm.transform_sf_into_required_layout(
            sf, 16384, 5120, (1, 128, 128),
            num_groups=1, is_sfa=False, disable_ue8m0_cast=True,
        ),
        0.0005,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_get_mn_major_tma_aligned_tensor_owns_storage():
    # Feed a tensor that is already in MN-major TMA-aligned layout so the
    # already-aligned fast path returns the input itself.
    base = torch.full((1, 128, 40), 0.0005, device="cuda", dtype=torch.float32)
    aligned = torch.empty_strided((1, 128, 40), (128 * 40, 1, 128),
                                  device="cuda", dtype=torch.float32)
    aligned.copy_(base)

    source_ptr = aligned.data_ptr()
    out = deep_gemm.get_mn_major_tma_aligned_tensor(aligned)
    if not isinstance(out, torch.Tensor):
        out = torch.from_dlpack(out)
    torch.cuda.synchronize()

    del aligned, base
    gc.collect()
    sentinels = [torch.full((1, 128, 40), float("nan"), device="cuda") for _ in range(20)]
    torch.cuda.synchronize()

    assert torch.isfinite(out).all().item(), (
        "live output aliases recycled scale storage "
        f"(source ptr reused: {any(x.data_ptr() == source_ptr for x in sentinels)})"
    )
    assert torch.allclose(out, torch.full_like(out, 0.0005))
