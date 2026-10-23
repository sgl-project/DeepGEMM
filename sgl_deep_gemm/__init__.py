from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import torch

from .cuda_helpers import find_cuda_home

# Set some default environment provided at setup
try:
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Build & load the TORCH_LIBRARY _C module
# ---------------------------------------------------------------------------
class _TorchOps:
    """`_C.<name>` is `torch.ops.deep_gemm.<name>.default`, bound once to skip per-call overload resolution."""

    def __getattr__(self, name: str):
        op = getattr(torch.ops.deep_gemm, name).default
        setattr(self, name, op)
        return op


def _load_module():
    """Load (or build then load) the compiled `_C` extension.

    Upstream ships `_C` as a plain `PYBIND11_MODULE` extension (`from ._C import ...`),
    while this fork historically built it as a TORCH_LIBRARY library reached through
    `torch.ops.load_library`. Try the pybind route first -- importing gives the
    operators as module attributes directly -- and fall back to the ops route so an
    older `_C.so` still works.
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(pkg_dir, '_C.so')

    if not os.path.exists(lib_path):
        from .build_ext import build_extension
        cuda_home = find_cuda_home()
        print(f'[DeepGEMM] Building _C module (CUDA_HOME={cuda_home})...')
        build_extension(os.path.dirname(pkg_dir), lib_path, cuda_home)
        print(f'[DeepGEMM] Built _C module: {lib_path}')

    try:
        from . import _C as module
        if hasattr(module, 'set_num_sms'):
            return module
    except ImportError:
        pass

    torch.ops.load_library(lib_path)
    return _TorchOps()

_C = _load_module()

# ---------------------------------------------------------------------------
# Capability query
# ---------------------------------------------------------------------------
def supported_arch_majors():
    """CUDA architecture majors this installed ``_C.so`` accepts, e.g. ``(9, 10, 11, 12)``.

    Reads the answer from the compiled binary rather than from the Python
    package, so a stale ``_C.so`` under a fresh Python side reports what it can
    actually do. Cheap and side-effect free: safe to call at import time, before
    any CUDA context exists.
    """
    return tuple(_C.supported_arch_majors())


_FEATURE_SUPPORT = ('unsupported', 'validated', 'unvalidated')


def arch_feature_status():
    """`{name: {'sm100': state, 'sm110': state}}` for every gated feature.

    SM110 is answered for its real arch major (11); nothing is mapped onto 10.
    """
    return {
        name: {'sm100': _FEATURE_SUPPORT[sm100], 'sm110': _FEATURE_SUPPORT[sm110]}
        for name, sm100, sm110 in _C.arch_feature_status()
    }


def supports_feature(arch, feature: str) -> bool:
    """Whether `feature` is enabled and validated for `arch` (an int or e.g. 'sm110')."""
    major = _arch_major(arch)
    status = arch_feature_status().get(feature)
    if status is None:
        return False
    if major == 10:
        return status['sm100'] == 'validated'
    if major == 11:
        return status['sm110'] == 'validated'
    return False


def _arch_major(arch) -> int:
    """Normalise an architecture to its major number.

    Accepts a major (``11``), a ``sm``-style identifier (``'sm110'``, ``'sm_110'``,
    ``90``) or a CUDA-style number (``110``). Anything ``>= 90`` is read as
    major-and-minor (``90 -> 9``, ``100 -> 10``, ``110 -> 11``); smaller values
    are already majors.
    """
    if isinstance(arch, str):
        value = arch.strip().lower()
        if value.startswith('sm'):
            value = value[2:]
        value = value.lstrip('_')
        if not value.isdigit():
            raise ValueError(f'unrecognised architecture {arch!r}; expected e.g. "sm110", "sm_110" or 11')
        number = int(value)
    elif isinstance(arch, int):
        number = arch
    else:
        raise TypeError(f'architecture must be an int or a str, got {type(arch).__name__}')
    return number // 10 if number >= 90 else number


def supports_arch(arch) -> bool:
    """Whether this build accepts ``arch``, given as an int or an ``sm``-style name.

    ``supports_arch('sm110')`` / ``supports_arch('sm_110')`` / ``supports_arch(110)``
    / ``supports_arch(11)`` all ask the same question: is SM110 a supported
    architecture major here?
    """
    return _arch_major(arch) in supported_arch_majors()

# ---------------------------------------------------------------------------
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
# Newer upstream exposes only the setter; keep the getter for callers that read it back.
get_deterministic_algorithms = getattr(_C, 'get_deterministic_algorithms', None)
set_ignore_compile_dims = _C.set_ignore_compile_dims


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
    def fp8_fp4_gemm_nt(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='', disable_ue8m0_cast=False, alpha=None):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.fp8_fp4_gemm_nt(a_data, a_sf, b_data, b_sf, d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, alpha)

    def fp8_fp4_gemm_nn(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='', disable_ue8m0_cast=False, alpha=None):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.fp8_fp4_gemm_nn(a_data, a_sf, b_data, b_sf, d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, alpha)

    def fp8_fp4_gemm_tn(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='', disable_ue8m0_cast=False, alpha=None):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.fp8_fp4_gemm_tn(a_data, a_sf, b_data, b_sf, d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, alpha)

    def fp8_fp4_gemm_tt(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='', disable_ue8m0_cast=False, alpha=None):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.fp8_fp4_gemm_tt(a_data, a_sf, b_data, b_sf, d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, alpha)

    fp4_gemm_nt = fp8_fp4_gemm_nt
    fp8_gemm_nt = fp8_fp4_gemm_nt
    fp8_gemm_nn = fp8_fp4_gemm_nn
    fp8_gemm_tn = fp8_fp4_gemm_tn
    fp8_gemm_tt = fp8_fp4_gemm_tt

    def m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, grouped_layout, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='nk', disable_ue8m0_cast=False, use_psum_layout=False, ensure_zero_padding=True, expected_m_for_psum_layout=None):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.m_grouped_fp8_fp4_gemm_nt_contiguous(a_data, a_sf, b_data, b_sf, d, grouped_layout, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, use_psum_layout, ensure_zero_padding, expected_m_for_psum_layout)

    def m_grouped_fp8_fp4_gemm_nn_contiguous(a, b, d, grouped_layout, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='nk', disable_ue8m0_cast=False, use_psum_layout=False, ensure_zero_padding=True):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.m_grouped_fp8_fp4_gemm_nn_contiguous(a_data, a_sf, b_data, b_sf, d, grouped_layout, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, use_psum_layout, ensure_zero_padding)

    m_grouped_fp4_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous
    m_grouped_fp8_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous
    m_grouped_fp8_gemm_nn_contiguous = m_grouped_fp8_fp4_gemm_nn_contiguous

    def bf16_gemm_nt(a, b, d, c=None, compiled_dims='', alpha=None):
        _C.bf16_gemm_nt(a, b, d, c, compiled_dims, alpha)

    def bf16_gemm_nn(a, b, d, c=None, compiled_dims='', alpha=None):
        _C.bf16_gemm_nn(a, b, d, c, compiled_dims, alpha)

    def bf16_gemm_tn(a, b, d, c=None, compiled_dims='', alpha=None):
        _C.bf16_gemm_tn(a, b, d, c, compiled_dims, alpha)

    def bf16_gemm_tt(a, b, d, c=None, compiled_dims='', alpha=None):
        _C.bf16_gemm_tt(a, b, d, c, compiled_dims, alpha)

    def einsum(expr, a, b, d, c=None, use_cublaslt=False):
        _C.einsum(expr, a, b, d, c, use_cublaslt)

    def fp8_einsum(expr, a, b, d, c=None, recipe=(1, 128, 128)):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        d_data, d_sf = d if isinstance(d, (tuple, list)) else (d, None)
        _C.fp8_einsum(expr, a_data, a_sf, b_data, b_sf, d_data, d_sf, c, recipe)

    def fp8_gemm_nt_skip_head_mid(a, b, d, head_splits, recipe=None, compiled_dims='nk', disable_ue8m0_cast=False):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.fp8_gemm_nt_skip_head_mid(a_data, a_sf, b_data, b_sf, d, head_splits, recipe, compiled_dims, disable_ue8m0_cast)

    def fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits=False, indices=None):
        op = getattr(_C, 'fp8_paged_mqa_logits', None)
        if op is not None:
            return op(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits, indices)
        # Newer upstream folds this into the fp4-capable spelling.
        q, q_sf = q if isinstance(q, (tuple, list)) else (q, None)
        return _C.fp8_fp4_paged_mqa_logits(q, q_sf, kv_cache, weights, context_lens, block_table,
                                            schedule_meta, max_context_len, indices)

    def fp8_mqa_logits(q, kv, weights, ks, ke, clean_logits=False, max_seqlen_k=0):
        """Score query rows against a contiguous KV span.

        This is the spelling SGLang calls. Newer upstream removed the standalone
        `fp8_mqa_logits` binding and kept only `fp8_fp4_mqa_logits`, whose `q` is a
        `(values, sf)` pair. Accept either input shape and forward accordingly, so
        the public name survives the rename.
        """
        op = getattr(_C, 'fp8_mqa_logits', None)
        if op is not None:
            (kv_data, kv_sf) = _parse_tensor_or_tuple(kv)
            return op(q, kv_data, kv_sf, weights, ks, ke, clean_logits, max_seqlen_k)
        # Newer upstream takes `q` and `kv` as `(values, sf)` pairs.
        q_pair = q if isinstance(q, (tuple, list)) else (q, None)
        kv_pair = _parse_tensor_or_tuple(kv)
        # NOTES: upstream's `clean_logits`/`logits_dtype` knobs are gone; the new
        # binding always produces the compressed, partly-uncleaned fp32 logits,
        # which is what `clean_logits=False` used to request.
        return _C.fp8_fp4_mqa_logits(q_pair, kv_pair, weights, ks, ke,
                                     max_seqlen_k, None)

    def fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits=False, logits_dtype=torch.float, indices=None):
        q_pair = q if isinstance(q, (tuple, list)) else (q, None)
        try:
            return _C.fp8_fp4_paged_mqa_logits(q_pair, kv_cache, weights, context_lens, block_table,
                                               schedule_meta, max_context_len, clean_logits, logits_dtype, indices)
        except TypeError:
            # Newer upstream dropped `clean_logits`, `logits_dtype` and the flat `q`.
            return _C.fp8_fp4_paged_mqa_logits(q_pair, kv_cache, weights, context_lens, block_table,
                                               schedule_meta, max_context_len, indices)

    def fp8_fp4_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits=False, max_seqlen_k=0, logits_dtype=torch.float, schedule_meta=None):
        q_pair = q if isinstance(q, (tuple, list)) else (q, None)
        kv_pair = _parse_tensor_or_tuple(kv)
        try:
            return _C.fp8_fp4_mqa_logits(q_pair, kv_pair, weights, cu_seq_len_k_start,
                                         cu_seq_len_k_end, clean_logits, max_seqlen_k, logits_dtype, schedule_meta)
        except TypeError:
            # Newer upstream dropped `clean_logits` and `logits_dtype`.
            return _C.fp8_fp4_mqa_logits(q_pair, kv_pair, weights, cu_seq_len_k_start,
                                         cu_seq_len_k_end, max_seqlen_k, schedule_meta)

    def get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices=None):
        return _C.get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices)

    def tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=None):
        _C.tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits)

    def transform_sf_into_required_layout(sf, mn, k, recipe, num_groups=None, is_sfa=None, disable_ue8m0_cast=False, psum_layout=None):
        (recipe_a, recipe_b, recipe_c) = recipe if len(recipe) == 3 else (recipe[0], recipe[1], None)
        return _C.transform_sf_into_required_layout(sf, mn, k, recipe_a, recipe_b, recipe_c, num_groups, is_sfa, disable_ue8m0_cast, psum_layout)

    get_mk_alignment_for_contiguous_layout = _C.get_mk_alignment_for_contiguous_layout

    def m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, masked_m, expected_m, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='nk', disable_ue8m0_cast=False):
        (a, a_sf), (b, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        return _C.m_grouped_fp8_fp4_gemm_nt_masked(a, a_sf, b, b_sf, d, masked_m, expected_m, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast)

    m_grouped_fp4_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked
    m_grouped_fp8_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked
    fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked

    def k_grouped_fp8_gemm_tn_contiguous(a, b, d, ks, grouped_layout, c=None, recipe=(1, 1, 128), compiled_dims='mn', use_psum_layout=False, use_padded_sf_layout=False):
        """Use compact per-group SF by default; opt in to padded SF with use_padded_sf_layout."""
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.k_grouped_fp8_gemm_tn_contiguous(a_data, a_sf, b_data, b_sf, d, ks, grouped_layout, c, recipe, compiled_dims, use_psum_layout, use_padded_sf_layout)

    def k_grouped_fp8_gemm_nt_contiguous(a, b, d, ks, grouped_layout, c=None, recipe=(1, 1, 128), compiled_dims='mn', use_psum_layout=False, use_padded_sf_layout=False):
        """Use compact per-group SF by default; opt in to padded SF with use_padded_sf_layout."""
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.k_grouped_fp8_gemm_nt_contiguous(a_data, a_sf, b_data, b_sf, d, ks, grouped_layout, c, recipe, compiled_dims, use_psum_layout, use_padded_sf_layout)

    def m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout, compiled_dims='nk', use_psum_layout=False, ensure_zero_padding=True, expected_m_for_psum_layout=None):
        _C.m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout, compiled_dims, use_psum_layout, ensure_zero_padding, expected_m_for_psum_layout)

    def m_grouped_bf16_gemm_nn_contiguous(a, b, d, grouped_layout, compiled_dims='nk', use_psum_layout=False, ensure_zero_padding=True):
        _C.m_grouped_bf16_gemm_nn_contiguous(a, b, d, grouped_layout, compiled_dims, use_psum_layout, ensure_zero_padding)

    def m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m, compiled_dims='nk'):
        _C.m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m, compiled_dims)

    def k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks, grouped_layout, c=None, compiled_dims='mn', use_psum_layout=False):
        _C.k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks, grouped_layout, c, compiled_dims, use_psum_layout)

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
                                      recipe=(1, 1, 32), compiled_dims='mn', use_psum_layout=False):
    (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
    _C.k_grouped_fp4_gemm_nt_contiguous(a_data, a_sf, b_data, b_sf, d, ks, grouped_layout,
                                       c, recipe, compiled_dims, use_psum_layout)


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


def get_bf16_mega_gate_config(num_tokens, hidden, num_routed_experts, num_topk):
    return dict(_C.get_bf16_mega_gate_config(num_tokens, hidden, num_routed_experts, num_topk))


def bf16_mega_gate(x, weight, num_topk, use_shared_as_routed, num_shared_experts,
                     routed_scaling_factor, ep_rank, scoring_func='identity', mask=None,
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
    fp8_fp4_mega_moe,
    bf16_mega_moe,
)

# Newer upstream dropped several Mega MoE helpers that this wheel used to re-export.
# Import them when present so older call sites keep working, and leave them unbound
# otherwise instead of failing the whole package import.
for _optional_mega_name in (
    'transform_scales_for_mega_moe',
    'nvfp4_mega_moe',
    'mega_moe_pre_dispatch',
    'mega_moe_pre_dispatch_sm90',
    'get_block_m_for_mega_moe',
):
    try:
        globals()[_optional_mega_name] = getattr(
            __import__('deep_gemm.mega', fromlist=[_optional_mega_name]), _optional_mega_name)
    except (ImportError, AttributeError):
        pass
del _optional_mega_name


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

    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())
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
                 fast_math: bool = True):
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
        fast_math
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

# Some utils
from . import testing
from . import utils
from .utils import *

# Initialize CPP modules
# Newer upstream takes only the library root; older builds also took CUDA_HOME.
try:
    _C.init(os.path.dirname(os.path.abspath(__file__)), find_cuda_home())
except TypeError:
    _C.init(os.path.dirname(os.path.abspath(__file__)))

# Newer upstream exposes the GEMM epilogue classes as the `_C.epilogue` submodule;
# older call sites reach them as `deep_gemm.epilogue.<Name>`.
try:
    from ._C import epilogue
except ImportError:
    epilogue = None

def _read_version() -> str:
    version_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VERSION')
    try:
        with open(version_file, 'r') as f:
            return f.read().strip()
    except OSError:
        return '0.0.0.dev0'

__version__ = _read_version()

# Allow `import deep_gemm.<name>` to resolve top-level public symbols, mirroring
# `from deep_gemm import <name>`. Without this, Python's import machinery only
# resolves submodules — top-level callables defined here are otherwise
# inaccessible via the dotted-import form.
import sys as _sys
import types as _types
for _name, _val in list(globals().items()):
    if _name.startswith('_') or _val is None or isinstance(_val, _types.ModuleType):
        continue
    _sys.modules.setdefault(f'{__name__}.{_name}', _val)
del _sys, _types, _name, _val
