import torch

import deep_gemm
from deep_gemm.testing import get_arch_major
from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8


def _quantize_rows(x: torch.Tensor):
    return per_token_cast_to_fp8(x, use_ue8m0=get_arch_major() == 10)


def _quantize_weights(x: torch.Tensor):
    if get_arch_major() == 9:
        return per_block_cast_to_fp8(x, use_ue8m0=False)
    return per_token_cast_to_fp8(x, use_ue8m0=True)


def _quantize_grouped(x: torch.Tensor, quantize):
    values, scales = zip(*(quantize(group) for group in x))
    return torch.stack(values), torch.stack(scales)


def _slice_rows(x, m: int):
    return x[0][:m], x[1][:m]


def _run_dense(a, b, m: int, n: int):
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt(_slice_rows(a, m), b, out)
    return out


def test_batch_invariant_fp8_dense():
    torch.manual_seed(0)
    max_m, n, k = 512, 3072, 2048
    a = _quantize_rows(torch.randn((max_m, k), device="cuda", dtype=torch.bfloat16))
    b = _quantize_weights(torch.randn((n, k), device="cuda", dtype=torch.bfloat16))

    old_mode = deep_gemm.get_batch_invariant()
    try:
        deep_gemm.set_batch_invariant(True)
        reference = _run_dense(a, b, max_m, n)

        # These M values cross multiple heuristic BLOCK_M/BLOCK_N choices on
        # SM90. Every row must nevertheless use the same reduction atom.
        for m in (1, 17, 65, 129, 257):
            actual = _run_dense(a, b, m, n)
            assert torch.equal(
                actual, reference[:m]
            ), f"dense FP8 output changed for {m=}"
    finally:
        deep_gemm.set_batch_invariant(old_mode)


def test_batch_invariant_fp8_m_grouped_contiguous():
    torch.manual_seed(1)
    num_groups, max_m, n, k = 4, 512, 1024, 1024
    a = _quantize_rows(torch.randn((max_m, k), device="cuda", dtype=torch.bfloat16))
    b = _quantize_grouped(
        torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16),
        _quantize_weights,
    )

    def run(m: int):
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        # Keeping all rows in one expert makes prefix comparisons direct while
        # changing the total grouped-M shape seen by the heuristic.
        grouped_layout = torch.zeros((m,), device="cuda", dtype=torch.int32)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            _slice_rows(a, m), b, out, grouped_layout
        )
        return out

    old_mode = deep_gemm.get_batch_invariant()
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        deep_gemm.set_batch_invariant(True)
        reference = run(max_m)
        for m in (128, 256, 384):
            actual = run(m)
            assert torch.equal(
                actual, reference[:m]
            ), f"contiguous grouped FP8 output changed for {m=}"
    finally:
        deep_gemm.set_batch_invariant(old_mode)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


def test_batch_invariant_fp8_m_grouped_masked():
    torch.manual_seed(2)
    num_groups, max_m, n, k = 4, 256, 1024, 1024
    a = _quantize_grouped(
        torch.randn((num_groups, max_m, k), device="cuda", dtype=torch.bfloat16),
        _quantize_rows,
    )
    b = _quantize_grouped(
        torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16),
        _quantize_weights,
    )

    def run(masked_m: torch.Tensor, expected_m: int):
        out = torch.empty((num_groups, max_m, n), device="cuda", dtype=torch.bfloat16)
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
            a, b, out, masked_m, expected_m
        )
        return out

    old_mode = deep_gemm.get_batch_invariant()
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        deep_gemm.set_batch_invariant(True)
        full_m = torch.full((num_groups,), max_m, device="cuda", dtype=torch.int32)
        reference = run(full_m, max_m)

        masked_m = torch.tensor([1, 17, 65, 129], device="cuda", dtype=torch.int32)
        for expected_m in (16, 64, 256):
            actual = run(masked_m, expected_m)
            for group, m in enumerate(masked_m.tolist()):
                assert torch.equal(actual[group, :m], reference[group, :m]), (
                    "masked grouped FP8 output changed for "
                    f"{group=}, {m=}, {expected_m=}"
                )
    finally:
        deep_gemm.set_batch_invariant(old_mode)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


if __name__ == "__main__":
    assert get_arch_major() in (9, 10)
    test_batch_invariant_fp8_dense()
    test_batch_invariant_fp8_m_grouped_contiguous()
    test_batch_invariant_fp8_m_grouped_masked()
    print("Batch-invariant FP8 tests passed")
