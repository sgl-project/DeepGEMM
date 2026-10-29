from __future__ import annotations

import atexit
import os
import torch
from glob import glob
from typing import Optional, Tuple, Union

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

_extension_paths = glob(os.path.join(os.path.dirname(__file__), '_C*.so'))
if not _extension_paths:
    raise ImportError('DeepGEMM extension is missing; build the TORCH_LIBRARY _C module first.')
torch.ops.load_library(max(_extension_paths, key=os.path.getmtime))


class _TorchOps:
    """`_C.<name>` is `torch.ops.deep_gemm.<name>.default`, bound once to skip per-call overload resolution."""

    def __getattr__(self, name: str):
        op = getattr(torch.ops.deep_gemm, name).default
        setattr(self, name, op)
        return op


_C = _TorchOps()


# Public wrappers match the sgl-deep-gemm wheel interface.
# Runtime config
# ---------------------------------------------------------------------------
set_num_sms = _C.set_num_sms
get_num_sms = _C.get_num_sms
# set_compile_mode / get_compile_mode are not exported on this branch.
set_tc_util = _C.set_tc_util
get_tc_util = _C.get_tc_util
set_pdl = _C.set_pdl
get_pdl = _C.get_pdl
use_deterministic_algorithms = _C.use_deterministic_algorithms
get_deterministic_algorithms = _C.get_deterministic_algorithms
set_ignore_compile_dims = _C.set_ignore_compile_dims
shutdown = _C.shutdown

# Epilogue classes
from . import epilogue
from .epilogue import encode as _encode_epilogue


def set_block_size_multiple_of(value: Union[int, Tuple[int, int]]):
    m, n = (value, value) if isinstance(value, int) else value
    _C.set_block_size_multiple_of(m, n)


# cuBLASLt Kernels
def cublaslt_gemm_nt(a, b, d, c=None):
    _C.cublaslt_gemm_nt(a, b, d, c)


def cublaslt_gemm_nn(a, b, d, c=None):
    _C.cublaslt_gemm_nn(a, b, d, c)


def cublaslt_gemm_tn(a, b, d, c=None):
    _C.cublaslt_gemm_tn(a, b, d, c)


def cublaslt_gemm_tt(a, b, d, c=None):
    _C.cublaslt_gemm_tt(a, b, d, c)

def _parse_tensor_or_tuple(input):
    if type(input) is tuple or type(input) is list:
        return input[0], input[1]
    elif isinstance(input, torch.Tensor):
        scale = torch.tensor([1.0], dtype=torch.float32, device=input.device)
        return input, scale

    assert False, "Expected Tensor, (Tensor, Tensor) tuple, or [Tensor, Tensor] list"

# ---------------------------------------------------------------------------
# GEMM / Attention / Einsum wrappers (handle optional params in Python)
# ---------------------------------------------------------------------------
try:
    def fp8_fp4_gemm_nt(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='', disable_ue8m0_cast=False, alpha=None, epilogue=None):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.fp8_fp4_gemm_nt(a_data, a_sf, b_data, b_sf, d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, alpha, *_encode_epilogue(epilogue))

    def fp8_fp4_gemm_nn(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='', disable_ue8m0_cast=False, alpha=None, epilogue=None):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.fp8_fp4_gemm_nn(a_data, a_sf, b_data, b_sf, d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, alpha, *_encode_epilogue(epilogue))

    def fp8_fp4_gemm_tn(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='', disable_ue8m0_cast=False, alpha=None, epilogue=None):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.fp8_fp4_gemm_tn(a_data, a_sf, b_data, b_sf, d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, alpha, *_encode_epilogue(epilogue))

    def fp8_fp4_gemm_tt(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='', disable_ue8m0_cast=False, alpha=None, epilogue=None):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.fp8_fp4_gemm_tt(a_data, a_sf, b_data, b_sf, d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, alpha, *_encode_epilogue(epilogue))

    fp4_gemm_nt = fp8_fp4_gemm_nt
    fp8_gemm_nt = fp8_fp4_gemm_nt
    fp8_gemm_nn = fp8_fp4_gemm_nn
    fp8_gemm_tn = fp8_fp4_gemm_tn
    fp8_gemm_tt = fp8_fp4_gemm_tt

    def m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, grouped_layout, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='nk', disable_ue8m0_cast=False, use_psum_layout=False, ensure_zero_padding=True, expected_m_for_psum_layout=None, epilogue=None):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.m_grouped_fp8_fp4_gemm_nt_contiguous(a_data, a_sf, b_data, b_sf, d, grouped_layout, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, use_psum_layout, ensure_zero_padding, expected_m_for_psum_layout, *_encode_epilogue(epilogue))

    def m_grouped_fp8_fp4_gemm_nn_contiguous(a, b, d, grouped_layout, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='nk', disable_ue8m0_cast=False, use_psum_layout=False, ensure_zero_padding=True, epilogue=None):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.m_grouped_fp8_fp4_gemm_nn_contiguous(a_data, a_sf, b_data, b_sf, d, grouped_layout, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, use_psum_layout, ensure_zero_padding, *_encode_epilogue(epilogue))

    m_grouped_fp4_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous
    m_grouped_fp8_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous
    m_grouped_fp8_gemm_nn_contiguous = m_grouped_fp8_fp4_gemm_nn_contiguous

    def bf16_gemm_nt(a, b, d, c=None, compiled_dims='', alpha=None, epilogue=None):
        _C.bf16_gemm_nt(a, b, d, c, compiled_dims, alpha, *_encode_epilogue(epilogue))

    def bf16_gemm_nn(a, b, d, c=None, compiled_dims='', alpha=None, epilogue=None):
        _C.bf16_gemm_nn(a, b, d, c, compiled_dims, alpha, *_encode_epilogue(epilogue))

    def bf16_gemm_tn(a, b, d, c=None, compiled_dims='', alpha=None, epilogue=None):
        _C.bf16_gemm_tn(a, b, d, c, compiled_dims, alpha, *_encode_epilogue(epilogue))

    def bf16_gemm_tt(a, b, d, c=None, compiled_dims='', alpha=None, epilogue=None):
        _C.bf16_gemm_tt(a, b, d, c, compiled_dims, alpha, *_encode_epilogue(epilogue))

    def einsum(expr, a, b, d, c=None, use_cublaslt=False, epilogue=None):
        _C.einsum(expr, a, b, d, c, use_cublaslt, *_encode_epilogue(epilogue))

    def fp8_einsum(expr, a, b, d, c=None, recipe=(1, 128, 128), epilogue=None):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        d_data, d_sf = d if isinstance(d, (tuple, list)) else (d, None)
        _C.fp8_einsum(expr, a_data, a_sf, b_data, b_sf, d_data, d_sf, c, recipe, *_encode_epilogue(epilogue))

    def fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, indices=None, histogram=None):
        (q, q_sf) = q if isinstance(q, (tuple, list)) else (q, None)
        return _C.fp8_fp4_paged_mqa_logits(q, q_sf, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, indices, histogram)

    def fp8_fp4_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k, schedule_meta=None):
        (q, q_sf), (kv_data, kv_sf) = (q if isinstance(q, (tuple, list)) else (q, None)), _parse_tensor_or_tuple(kv)
        return _C.fp8_fp4_mqa_logits(q, q_sf, kv_data, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k, schedule_meta)

    def get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices=None):
        return _C.get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices)

    def tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=None):
        _C.tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits)

    def transform_sf_into_required_layout(sf, mn, k, recipe, num_groups=None, is_sfa=None, disable_ue8m0_cast=False, psum_layout=None):
        (recipe_a, recipe_b, recipe_c) = recipe if len(recipe) == 3 else (recipe[0], recipe[1], None)
        return _C.transform_sf_into_required_layout(sf, mn, k, recipe_a, recipe_b, recipe_c, num_groups, is_sfa, disable_ue8m0_cast, psum_layout)

    get_mk_alignment_for_contiguous_layout = _C.get_mk_alignment_for_contiguous_layout

    def m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, masked_m, expected_m, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='nk', disable_ue8m0_cast=False, epilogue=None):
        (a, a_sf), (b, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        return _C.m_grouped_fp8_fp4_gemm_nt_masked(a, a_sf, b, b_sf, d, masked_m, expected_m, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, *_encode_epilogue(epilogue))

    m_grouped_fp4_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked
    m_grouped_fp8_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked
    fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked

    def k_grouped_fp8_gemm_tn_contiguous(a, b, d, ks, grouped_layout, c=None, recipe=(1, 1, 128), compiled_dims='mn', use_psum_layout=False, use_padded_sf_layout=False, epilogue=None):
        """Use compact per-group SF by default; opt in to padded SF with use_padded_sf_layout."""
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.k_grouped_fp8_gemm_tn_contiguous(a_data, a_sf, b_data, b_sf, d, ks, grouped_layout, c, recipe, compiled_dims, use_psum_layout, use_padded_sf_layout, *_encode_epilogue(epilogue))

    def k_grouped_fp8_gemm_nt_contiguous(a, b, d, ks, grouped_layout, c=None, recipe=(1, 1, 128), compiled_dims='mn', use_psum_layout=False, use_padded_sf_layout=False, epilogue=None):
        """Use compact per-group SF by default; opt in to padded SF with use_padded_sf_layout."""
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.k_grouped_fp8_gemm_nt_contiguous(a_data, a_sf, b_data, b_sf, d, ks, grouped_layout, c, recipe, compiled_dims, use_psum_layout, use_padded_sf_layout, *_encode_epilogue(epilogue))

    def m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout, compiled_dims='nk', use_psum_layout=False, ensure_zero_padding=True, expected_m_for_psum_layout=None, epilogue=None):
        _C.m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout, compiled_dims, use_psum_layout, ensure_zero_padding, expected_m_for_psum_layout, *_encode_epilogue(epilogue))

    def m_grouped_bf16_gemm_nn_contiguous(a, b, d, grouped_layout, compiled_dims='nk', use_psum_layout=False, ensure_zero_padding=True, epilogue=None):
        _C.m_grouped_bf16_gemm_nn_contiguous(a, b, d, grouped_layout, compiled_dims, use_psum_layout, ensure_zero_padding, *_encode_epilogue(epilogue))

    def m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m, compiled_dims='nk', epilogue=None):
        _C.m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m, compiled_dims, *_encode_epilogue(epilogue))

    def k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks, grouped_layout, c=None, compiled_dims='mn', use_psum_layout=False, epilogue=None):
        _C.k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks, grouped_layout, c, compiled_dims, use_psum_layout, *_encode_epilogue(epilogue))

    bf16_m_grouped_gemm_nt_masked = m_grouped_bf16_gemm_nt_masked

except AttributeError:
    pass


def cublaslt_nvfp4_gemm_nt(a, b, d, c=None):
    (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
    _C.cublaslt_nvfp4_gemm_nt(a_data, a_sf, b_data, b_sf, d, c)


def batched_syrk(a, d):
    _C.batched_syrk(a, d)


def batched_symm(a, b, d):
    _C.batched_symm(a, b, d)


def k_grouped_fp4_gemm_nt_contiguous(a, b, d, ks, grouped_layout, c=None,
                                      recipe=(1, 1, 32), compiled_dims='mn', use_psum_layout=False, epilogue=None):
    (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
    _C.k_grouped_fp4_gemm_nt_contiguous(a_data, a_sf, b_data, b_sf, d, ks, grouped_layout,
                                       c, recipe, compiled_dims, use_psum_layout, *_encode_epilogue(epilogue))


def get_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, num_heads):
    return _C.get_mqa_logits_metadata(
        cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, num_heads)


def get_sparse_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens,
                                    sparse_kv_block_indices, qk_dtype, sparse_block_kv,
                                    use_unaligned_ks=False):
    return _C.get_sparse_mqa_logits_metadata(
        cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, sparse_kv_block_indices,
        qk_dtype, sparse_block_kv, use_unaligned_ks)


def get_paged_sparse_mqa_logits_metadata(context_lens, block_table, indices, page_kv,
                                          sparse_kv_block_indices, qk_dtype, sparse_block_kv):
    return _C.get_paged_sparse_mqa_logits_metadata(
        context_lens, block_table, indices, page_kv, sparse_kv_block_indices,
        qk_dtype, sparse_block_kv)


def fp8_fp4_sparse_mqa_logits(q, kv, weights, metadata, num_max_sparse_blocks,
                               sparse_block_kv, use_unaligned_ks=False):
    (q_data, q_sf), (kv_data, kv_sf) = (q if isinstance(q, (tuple, list)) else (q, None)), _parse_tensor_or_tuple(kv)
    return _C.fp8_fp4_sparse_mqa_logits(
        q_data, q_sf, kv_data, kv_sf, weights, metadata, num_max_sparse_blocks,
        sparse_block_kv, use_unaligned_ks)


def fp8_fp4_paged_sparse_mqa_logits(q, kv_cache, weights, metadata,
                                     num_max_sparse_blocks, sparse_block_kv):
    q_data, q_sf = q if isinstance(q, (tuple, list)) else (q, None)
    return _C.fp8_fp4_paged_sparse_mqa_logits(
        q_data, q_sf, kv_cache, weights, metadata, num_max_sparse_blocks, sparse_block_kv)


def bf16_mega_gate(x, weight, num_topk, use_shared_as_routed, num_shared_experts,
                     routed_scaling_factor, ep_rank, scoring_func='sqrtsoftplus', mask=None,
                     bias=None, image_bias=None, image_token_mask=None, fix_routing_mask=None,
                     to_physical_map=None, logical_count=None, unmapped_topk_idx=None,
                     force_random=None, out=None):
    out_idx, out_weights = (None, None) if out is None else out
    result = _C.bf16_mega_gate(
        x, weight, num_topk, use_shared_as_routed, num_shared_experts,
        routed_scaling_factor, ep_rank, scoring_func, mask, bias, image_bias,
        image_token_mask, fix_routing_mask, to_physical_map, logical_count,
        unmapped_topk_idx, force_random, out_idx, out_weights)
    return (out_idx, out_weights) if out is not None else result


def mega_mhc(x, residual, shifted_prev_mix, post_mix, comb_res_mix, fn, mix_scales,
                mix_bases, hc_mult, hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps,
                num_sinkhorn_iters, rmsnorm_weight, rmsnorm_eps, rmsnorm_scale,
                new_residual, new_prev_mix, new_post_mix, new_comb_res_mix,
                y_bf16=None, y_fp8=None, y_gemm_sf=None, y_routed_sf=None,
                y_shared_sf=None, shared_sf_block_m=0):
    _C.mega_mhc(
        x, residual, shifted_prev_mix, post_mix, comb_res_mix, fn, mix_scales,
        mix_bases, hc_mult, hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps,
        num_sinkhorn_iters, rmsnorm_weight, rmsnorm_eps, rmsnorm_scale,
        new_residual, new_prev_mix, new_post_mix, new_comb_res_mix,
        y_bf16, y_fp8, y_gemm_sf, y_routed_sf, y_shared_sf, shared_sf_block_m)

# Mega kernels
from . import mega
from .mega import (
    SymmBuffer,
    transform_weights_for_mega_moe,
    transform_scales_for_mega_moe,
    fp8_fp4_mega_moe,
    nvfp4_mega_moe,
    bf16_mega_moe,
    mega_moe_pre_dispatch,
    mega_moe_pre_dispatch_sm90,
    get_block_m_for_mega_moe,
)


class SM90SymmBuffer:
    def __init__(self, group,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 use_fp8_dispatch: bool = True,
                 activation: str = 'swiglu'):
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden

        layout_args = (group.size(), num_experts,
                       num_max_tokens_per_rank, num_topk,
                       hidden, intermediate_hidden,
                       use_fp8_dispatch, activation)
        num_bytes = _C.get_symm_buffer_size_for_sm90_mega_moe(*layout_args)
        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        (x, x_sf, topk_idx, topk_weights,
         l1_acts, l1_acts_sf, l2_acts, l2_acts_sf) = _C.slice_symm_buffer_for_sm90_mega_moe(self.buffer, *layout_args)
        self.x = x
        self.x_sf = x_sf
        self.topk_idx = topk_idx
        self.topk_weights = topk_weights
        self.l1_acts = l1_acts
        self.l1_acts_sf = l1_acts_sf
        self.l2_acts = l2_acts
        self.l2_acts_sf = l2_acts_sf

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


def get_symm_buffer_for_sm90_mega_moe(group,
                                      num_experts: int,
                                      num_max_tokens_per_rank: int, num_topk: int,
                                      hidden: int, intermediate_hidden: int,
                                      use_fp8_dispatch: bool = True,
                                      activation: str = 'swiglu') -> SM90SymmBuffer:
    from .utils.math import align

    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_sm90_mega_moe())
    return SM90SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        use_fp8_dispatch, activation
    )


def get_symm_buffer_for_mega_moe(group,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 num_shared_experts: int = 0,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu'):
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:
        assert mma_type.split('x')[0] == 'fp8'
        assert num_shared_experts == 0
        return get_symm_buffer_for_sm90_mega_moe(
            group, num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            True, activation
        )
    return mega.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts=num_shared_experts,
        use_fp8_dispatch=use_fp8_dispatch,
        mma_type=mma_type,
        activation=activation
    )


def transform_weights_for_mega_moe_sm90(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    l1_fp8, l1_sf = l1_weights

    def _interleave_one(t, gran: int = 8) -> torch.Tensor:
        g, n, *rest = t.shape
        half = n // 2
        gate = t[:, :half].reshape(g, half // gran, gran, *rest)
        up = t[:, half:].reshape(g, half // gran, gran, *rest)
        return torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))

    return (_interleave_one(l1_fp8), l1_sf), l2_weights


def fp8_mega_moe(y: torch.Tensor,
                 l1_weights: Tuple[torch.Tensor, torch.Tensor],
                 l2_weights: Tuple[torch.Tensor, torch.Tensor],
                 sym_buffer: SM90SymmBuffer,
                 cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                 recipe: Tuple[int, int, int] = (128, 128, 128),
                 activation: str = 'swiglu',
                 activation_clamp: Optional[float] = None,
                 fast_math: bool = True,
                 num_tokens_bound: Optional[int] = None):
    """`num_tokens_bound`: upper bound on this call's per-rank token count on every rank (e.g. the maximum over a DP step);
    lets the launch pick its schedule by the call instead of by the buffer capacity. None = the buffer capacity, and a
    bound above it is rejected."""
    (l1_weights_data, l1_weights_sf) = l1_weights
    (l2_weights_data, l2_weights_sf) = l2_weights
    _C.fp8_mega_moe(
        y,
        l1_weights_data, l1_weights_sf,
        l2_weights_data, l2_weights_sf,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math,
        int(num_tokens_bound) if num_tokens_bound is not None else 0
    )


def mega_moe_pre_dispatch_sm90(x: torch.Tensor,
                               topk_idx: torch.Tensor,
                               topk_weights: torch.Tensor,
                               buf_x: torch.Tensor,
                               buf_x_sf: torch.Tensor,
                               buf_topk_idx: torch.Tensor,
                               buf_topk_weights: torch.Tensor,
                               num_tokens: int,
                               group_size: int = 128,
                               routed_scaling_factor: float = 1.0) -> None:
    _C.mega_moe_pre_dispatch_sm90(
        x, topk_idx, topk_weights,
        buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights,
        num_tokens, group_size, float(routed_scaling_factor),
    )

# Locality domain helpers
from . import locality_domain
from .locality_domain import (
    get_num_locality_domains,
    is_localization_available,
    localize,
    localize_rows,
    is_localized,
    destroy_localizer,
)

# Some utils
from . import testing
from . import utils
from .utils import *

# Legacy Triton kernels remain available from the source package.
try:
    from . import legacy
except Exception as error:
    print(f'Failed to load legacy DeepGEMM A100 Triton kernels: {error}')

# Initialize CPP modules
_C.init(os.path.dirname(os.path.abspath(__file__)), os.environ.get('CUDA_HOME', '/usr/local/cuda'))
# Native cuBLASLt exit callbacks can free handle storage before C++ static destructors.
# Release handles and workspaces while Python, PyTorch and CUDA are still available.
atexit.register(shutdown)

__version__ = '2.8.2'
