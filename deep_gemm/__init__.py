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

from . import deep_gemm_cpp  # noqa: F401


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


# Initialize C++ runtime once on import (safe for fork when avoiding torch CUDA_HOME)
torch.ops.deep_gemm.init(
    os.path.dirname(os.path.abspath(__file__)),
    _find_cuda_home(),
)

set_num_sms = torch.ops.deep_gemm.set_num_sms
get_num_sms = torch.ops.deep_gemm.get_num_sms
set_tc_util = torch.ops.deep_gemm.set_tc_util
get_tc_util = torch.ops.deep_gemm.get_tc_util

fp8_gemm_nt = torch.ops.deep_gemm.fp8_gemm_nt
fp8_gemm_nn = torch.ops.deep_gemm.fp8_gemm_nn
fp8_gemm_tn = torch.ops.deep_gemm.fp8_gemm_tn
fp8_gemm_tt = torch.ops.deep_gemm.fp8_gemm_tt
m_grouped_fp8_gemm_nt_contiguous = torch.ops.deep_gemm.m_grouped_fp8_gemm_nt_contiguous
m_grouped_fp8_gemm_nn_contiguous = torch.ops.deep_gemm.m_grouped_fp8_gemm_nn_contiguous
# Backward-compat alias
fp8_m_grouped_gemm_nt_masked = torch.ops.deep_gemm.m_grouped_fp8_gemm_nt_masked
k_grouped_fp8_gemm_tn_contiguous = torch.ops.deep_gemm.k_grouped_fp8_gemm_tn_contiguous

# BF16 GEMMs
bf16_gemm_nt = torch.ops.deep_gemm.bf16_gemm_nt
bf16_gemm_nn = torch.ops.deep_gemm.bf16_gemm_nn
bf16_gemm_tn = torch.ops.deep_gemm.bf16_gemm_tn
bf16_gemm_tt = torch.ops.deep_gemm.bf16_gemm_tt
m_grouped_bf16_gemm_nt_contiguous = torch.ops.deep_gemm.m_grouped_bf16_gemm_nt_contiguous
m_grouped_bf16_gemm_nt_masked = torch.ops.deep_gemm.m_grouped_bf16_gemm_nt_masked

# Layout kernels
transform_sf_into_required_layout = torch.ops.deep_gemm.transform_sf_into_required_layout

# Utility functions
get_tma_aligned_size = torch.ops.deep_gemm.get_tma_aligned_size
get_mk_alignment_for_contiguous_layout = torch.ops.deep_gemm.get_mk_alignment_for_contiguous_layout
get_mn_major_tma_aligned_tensor = torch.ops.deep_gemm.get_mn_major_tma_aligned_tensor
get_mn_major_tma_aligned_packed_ue8m0_tensor = torch.ops.deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor
get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor = torch.ops.deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor

# Some utils
from . import testing
from . import utils
from .utils import *

def _verify_ops_loaded():
    expected_ops = [
        'init', 'set_num_sms', 'get_num_sms', 'set_tc_util', 'get_tc_util',
        'fp8_gemm_nt', 'fp8_gemm_nn', 'fp8_gemm_tn', 'fp8_gemm_tt',
        'm_grouped_fp8_gemm_nt_contiguous', 'm_grouped_fp8_gemm_nn_contiguous',
        'm_grouped_fp8_gemm_nt_masked', 'k_grouped_fp8_gemm_tn_contiguous',
        'transform_sf_into_required_layout', 'get_tma_aligned_size',
        'get_mk_alignment_for_contiguous_layout', 'get_mn_major_tma_aligned_tensor',
        'get_mn_major_tma_aligned_packed_ue8m0_tensor',
        'get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor'
    ]
    
    available_ops = list(torch.ops.deep_gemm.__dict__.keys())
    missing_ops = [op for op in expected_ops if op not in available_ops]
    
    if missing_ops:
        print(f"Warning: Missing operations: {missing_ops}")
    else:
        print("All deep_gemm operations loaded successfully!")

if __debug__:
    _verify_ops_loaded()
