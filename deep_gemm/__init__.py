import os
import subprocess

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# Import functions from the CPP module
from deep_gemm import deep_gemm_cpp
torch.ops.deep_gemm.init(
    os.path.dirname(os.path.abspath(__file__)), # Library root directory path
    torch.utils.cpp_extension.CUDA_HOME         # CUDA home
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
fp8_m_grouped_gemm_nt_masked = torch.ops.deep_gemm.fp8_m_grouped_gemm_nt_masked
k_grouped_fp8_gemm_tn_contiguous = torch.ops.deep_gemm.k_grouped_fp8_gemm_tn_contiguous

# Layout kernels
transform_sf_into_required_layout = torch.ops.deep_gemm.transform_sf_into_required_layout

# Utility functions
get_tma_aligned_size = torch.ops.deep_gemm.get_tma_aligned_size
get_mk_alignment_for_contiguous_layout = torch.ops.deep_gemm.get_mk_alignment_for_contiguous_layout
get_mn_major_tma_aligned_tensor = torch.ops.deep_gemm.get_mn_major_tma_aligned_tensor
get_mn_major_tma_aligned_packed_ue8m0_tensor = torch.ops.deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor
get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor = torch.ops.deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor

# Some alias for legacy supports
# TODO: remove these later
fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_gemm_nt_masked
bf16_m_grouped_gemm_nt_masked = m_grouped_bf16_gemm_nt_masked

# Some utils
from . import testing
from . import utils
from .utils import *

def _verify_ops_loaded():
    expected_ops = [
        'init', 'set_num_sms', 'get_num_sms', 'set_tc_util', 'get_tc_util',
        'fp8_gemm_nt', 'fp8_gemm_nn', 'fp8_gemm_tn', 'fp8_gemm_tt',
        'm_grouped_fp8_gemm_nt_contiguous', 'm_grouped_fp8_gemm_nn_contiguous',
        'fp8_m_grouped_gemm_nt_masked', 'k_grouped_fp8_gemm_tn_contiguous',
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
