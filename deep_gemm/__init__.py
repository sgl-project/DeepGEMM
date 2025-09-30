import os
import subprocess
import torch
import torch.utils.cpp_extension

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

from . import deep_gemm_cpp  # noqa: F401  # Registers ops into torch.ops without touching CUDA


def _find_cuda_home() -> str:
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        try:
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output(['which', 'nvcc'], stderr=devnull).decode().rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    assert cuda_home is not None
    return cuda_home


# Lazy runtime init to be fork-safe on Linux (avoid initializing CUDA before fork)
_dg_initialized = False

def _ensure_initialized() -> None:
    global _dg_initialized
    if _dg_initialized:
        return
    library_root = os.path.dirname(os.path.abspath(__file__))
    torch.ops.deep_gemm.init(library_root, _find_cuda_home())
    _dg_initialized = True


def _wrap_op(name: str):
    func = getattr(torch.ops.deep_gemm, name)
    def _fn(*args, **kwargs):
        _ensure_initialized()
        return func(*args, **kwargs)
    return _fn

set_num_sms = _wrap_op('set_num_sms')
get_num_sms = _wrap_op('get_num_sms')
set_compile_mode = _wrap_op('set_compile_mode')
get_compile_mode = _wrap_op('get_compile_mode')
set_tc_util = _wrap_op('set_tc_util')
get_tc_util = _wrap_op('get_tc_util')

fp8_gemm_nt = _wrap_op('fp8_gemm_nt')
fp8_gemm_nn = _wrap_op('fp8_gemm_nn')
fp8_gemm_tn = _wrap_op('fp8_gemm_tn')
fp8_gemm_tt = _wrap_op('fp8_gemm_tt')
m_grouped_fp8_gemm_nt_contiguous = _wrap_op('m_grouped_fp8_gemm_nt_contiguous')
m_grouped_fp8_gemm_nn_contiguous = _wrap_op('m_grouped_fp8_gemm_nn_contiguous')
# Export both canonical name and backward-compat alias
m_grouped_fp8_gemm_nt_masked = _wrap_op('m_grouped_fp8_gemm_nt_masked')
fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_gemm_nt_masked
k_grouped_fp8_gemm_nt_contiguous = _wrap_op('k_grouped_fp8_gemm_nt_contiguous')
k_grouped_fp8_gemm_tn_contiguous = _wrap_op('k_grouped_fp8_gemm_tn_contiguous')

# BF16 GEMMs
bf16_gemm_nt = _wrap_op('bf16_gemm_nt')
bf16_gemm_nn = _wrap_op('bf16_gemm_nn')
bf16_gemm_tn = _wrap_op('bf16_gemm_tn')
bf16_gemm_tt = _wrap_op('bf16_gemm_tt')
m_grouped_bf16_gemm_nt_contiguous = _wrap_op('m_grouped_bf16_gemm_nt_contiguous')
m_grouped_bf16_gemm_nt_masked = _wrap_op('m_grouped_bf16_gemm_nt_masked')

# cuBLASLt GEMMs
cublaslt_gemm_nt = _wrap_op('cublaslt_gemm_nt')
cublaslt_gemm_nn = _wrap_op('cublaslt_gemm_nn')
cublaslt_gemm_tn = _wrap_op('cublaslt_gemm_tn')
cublaslt_gemm_tt = _wrap_op('cublaslt_gemm_tt')

# Attention kernel
fp8_gemm_nt_skip_head_mid = _wrap_op('fp8_gemm_nt_skip_head_mid')
fp8_mqa_logits = _wrap_op('fp8_mqa_logits')
get_paged_mqa_logits_metadata = _wrap_op('get_paged_mqa_logits_metadata')
fp8_paged_mqa_logits = _wrap_op('fp8_paged_mqa_logits')

# Einsum kernel
einsum = _wrap_op('einsum')

# Layout kernels
transform_sf_into_required_layout = _wrap_op('transform_sf_into_required_layout')

# Utility functions
get_tma_aligned_size = _wrap_op('get_tma_aligned_size')
get_mk_alignment_for_contiguous_layout = _wrap_op('get_mk_alignment_for_contiguous_layout')
get_mn_major_tma_aligned_tensor = _wrap_op('get_mn_major_tma_aligned_tensor')
get_mn_major_tma_aligned_packed_ue8m0_tensor = _wrap_op('get_mn_major_tma_aligned_packed_ue8m0_tensor')
get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor = _wrap_op('get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor')

# Some utils
from . import testing
from . import utils
from .utils import *

def _verify_ops_loaded():
    expected_ops = [
        'init', 'set_num_sms', 'get_num_sms', 'set_tc_util', 'get_tc_util',
        'fp8_gemm_nt', 'fp8_gemm_nn', 'fp8_gemm_tn', 'fp8_gemm_tt',
        'm_grouped_fp8_gemm_nt_contiguous', 'm_grouped_fp8_gemm_nn_contiguous',
        'm_grouped_fp8_gemm_nt_masked', 'k_grouped_fp8_gemm_nt_contiguous',
        'k_grouped_fp8_gemm_tn_contiguous',
        'transform_sf_into_required_layout', 'get_tma_aligned_size',
        'get_mk_alignment_for_contiguous_layout', 'get_mn_major_tma_aligned_tensor',
        'get_mn_major_tma_aligned_packed_ue8m0_tensor',
        'get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor',
        'fp8_gemm_nt_skip_head_mid', 'fp8_mqa_logits',
        'get_paged_mqa_logits_metadata', 'fp8_paged_mqa_logits',
        'einsum',
        'cublaslt_gemm_nt', 'cublaslt_gemm_nn',
        'cublaslt_gemm_tn', 'cublaslt_gemm_tt',
    ]

    available_ops = list(torch.ops.deep_gemm.__dict__.keys())
    missing_ops = [op for op in expected_ops if op not in available_ops]

    if missing_ops:
        print(f"Warning: Missing operations: {missing_ops}")


_ensure_initialized()


if __debug__:
    _verify_ops_loaded()
