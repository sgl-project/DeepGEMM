import torch

from deep_gemm.utils import align, get_mk_alignment_for_contiguous_layout


def assert_psum_zero_padding(a: torch.Tensor | tuple, d: torch.Tensor, grouped_layout: torch.Tensor, dtype_label: str) -> None:
    a_data = a[0] if isinstance(a, tuple) else a
    for group_idx, current_m in enumerate(grouped_layout.cpu().tolist()):
        aligned_m = align(current_m, get_mk_alignment_for_contiguous_layout())
        if current_m < aligned_m:
            a_padding = a_data[current_m: aligned_m]
            d_padding = d[current_m: aligned_m]
            assert torch.equal(a_padding, torch.zeros_like(a_padding)), f'{group_idx=}, nonzero {dtype_label} input padding'
            assert torch.equal(d_padding, torch.zeros_like(d_padding)), f'{group_idx=}, nonzero {dtype_label} output padding'
