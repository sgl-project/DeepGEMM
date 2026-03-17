"""Layout utility wrappers.

These functions call into the tvm-ffi _C module for CUDA kernel execution
and handle output tensor allocation on the Python side.
"""
from __future__ import annotations

import torch
from .math import align, ceil_div


def _get_C():
    from .. import _C
    return _C


def get_tma_aligned_size(mn: int, element_size: int) -> int:
    return int(_get_C().get_tma_aligned_size(mn, element_size))


def get_mk_alignment_for_contiguous_layout() -> int:
    return int(_get_C().get_mk_alignment_for_contiguous_layout())

get_m_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
get_k_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout


def _pack_fp32_into_ue8m0_fallback(x: torch.Tensor) -> torch.Tensor:
    """PyTorch fallback for ue8m0 packing when the CUDA kernel cannot handle the layout."""
    assert x.dtype == torch.float and x.dim() in (2, 3)
    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    b = x.shape[0]
    ue8m0_tensor = (x.view(torch.int) >> 23).to(torch.uint8)
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)
    padded = torch.zeros((b, aligned_mn, aligned_k), device=x.device, dtype=torch.uint8)
    padded[:, :mn, :k] = ue8m0_tensor
    padded = padded.view(-1).view(dtype=torch.int).view(b, aligned_mn, aligned_k // 4)
    transposed = torch.zeros((b, aligned_k // 4, aligned_mn), device=x.device, dtype=torch.int).mT
    transposed[:, :, :] = padded
    aligned_x = transposed[:, :mn, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


try:
    _get_C().preprocess_sf  # probe availability

    def get_mn_major_tma_aligned_tensor(sf: torch.Tensor) -> torch.Tensor:
        """Transpose FP32 scaling factors into MN-major TMA-aligned layout."""
        C = _get_C()
        info = C.preprocess_sf(sf)
        dim, ng, mn_pp, sf_k_pp, tma_mn = int(info[0]), int(info[1]), int(info[2]), int(info[3]), int(info[4])

        out_numel = ng * sf_k_pp * tma_mn
        flat = torch.empty(out_numel, dtype=torch.float32, device=sf.device)

        if dim == 2:
            out = torch.as_strided(flat, (mn_pp, sf_k_pp), (1, tma_mn))
        else:
            out = torch.as_strided(flat, (ng, mn_pp, sf_k_pp), (sf_k_pp * tma_mn, 1, tma_mn))

        success = C.get_mn_major_tma_aligned_tensor(sf, out)
        if not success:
            remove_dim = False
            if sf.dim() == 2:
                sf = sf.unsqueeze(0)
                remove_dim = True
            b, m, k = sf.shape
            tma_mn_f = get_tma_aligned_size(m, sf.element_size())
            t = torch.zeros((b, k, tma_mn_f), dtype=sf.dtype, device=sf.device).mT
            t[:, :m, :k] = sf
            out = t[:, :m, :]
            if remove_dim:
                out = out.squeeze(0)
        return out

    def get_mn_major_tma_aligned_packed_ue8m0_tensor(sf: torch.Tensor) -> torch.Tensor:
        """Pack FP32 scaling factors into UE8M0 int32 in MN-major TMA-aligned layout."""
        C = _get_C()
        info = C.preprocess_sf(sf)
        dim, ng, mn_pp, sf_k_pp, tma_mn = int(info[0]), int(info[1]), int(info[2]), int(info[3]), int(info[4])
        packed_sf_k = ceil_div(sf_k_pp, 4)
        tma_mn_int = get_tma_aligned_size(mn_pp, 4)

        out_numel = ng * packed_sf_k * tma_mn_int
        flat = torch.empty(out_numel, dtype=torch.int32, device=sf.device)

        if dim == 2:
            out = torch.as_strided(flat, (mn_pp, packed_sf_k), (1, tma_mn_int))
        else:
            out = torch.as_strided(flat, (ng, mn_pp, packed_sf_k), (packed_sf_k * tma_mn_int, 1, tma_mn_int))

        success = C.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf, out)
        if not success:
            return _pack_fp32_into_ue8m0_fallback(sf)
        return out

    def get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
            sf: torch.Tensor, ks_tensor: torch.Tensor, ks: list[int]) -> torch.Tensor:
        """Pack k-grouped FP32 scaling factors into UE8M0 int32 in MN-major TMA-aligned layout."""
        C = _get_C()
        sf_k, mn = sf.shape
        num_groups = len(ks)

        packed_sf_k = sum(ceil_div(k, 512) for k in ks)
        tma_mn_int = get_tma_aligned_size(mn, 4)

        out_numel = packed_sf_k * tma_mn_int
        flat = torch.empty(out_numel, dtype=torch.int32, device=sf.device)
        # C++ output: shape (mn, packed_sf_k) with strides (1, tma_mn_int) — MN-major
        out = torch.as_strided(flat, (mn, packed_sf_k), (1, tma_mn_int))

        ks_list_tensor = torch.tensor(ks, dtype=torch.int32, device='cpu')
        C.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
            sf, ks_tensor, ks_list_tensor, num_groups, out)
        # Return transposed: (packed_sf_k, mn) for compatibility with split
        return out.T

except AttributeError:
    pass
