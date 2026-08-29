import os
import subprocess
import torch
import tvm_ffi
from glob import glob

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
    raise ImportError('DeepGEMM extension is missing; build the TVM-FFI _C module first.')
_C = tvm_ffi.load_module(max(_extension_paths, key=os.path.getmtime))


def _bind_exports(*names: str) -> None:
    for name in names:
        globals()[name] = getattr(_C, name)


# Configs and cuBLASLt kernels
_bind_exports(
    'set_num_sms', 'get_num_sms', 'set_tc_util', 'get_tc_util',
    'set_pdl', 'get_pdl',
    'cublaslt_gemm_nt', 'cublaslt_gemm_nn', 'cublaslt_gemm_tn', 'cublaslt_gemm_tt',
)

try:
    # DeepGEMM Kernels
    _kernel_exports = (
        # FP8 FP4 GEMMs
        'fp8_fp4_gemm_nt', 'fp8_fp4_gemm_nn',
        'fp8_fp4_gemm_tn', 'fp8_fp4_gemm_tt',
        'm_grouped_fp8_fp4_gemm_nt_contiguous',
        'm_grouped_fp8_fp4_gemm_nn_contiguous',
        'm_grouped_fp8_fp4_gemm_nt_masked',
        # FP8 GEMMs
        'fp8_gemm_nt', 'fp8_gemm_nn',
        'fp8_gemm_tn', 'fp8_gemm_tt',
        'fp8_gemm_nt_skip_head_mid',
        'm_grouped_fp8_gemm_nt_contiguous',
        'm_grouped_fp8_gemm_nn_contiguous',
        'm_grouped_fp8_gemm_nt_masked',
        'k_grouped_fp8_gemm_nt_contiguous',
        'k_grouped_fp8_gemm_tn_contiguous',
        # BF16 GEMMs
        'bf16_gemm_nt', 'bf16_gemm_nn',
        'bf16_gemm_tn', 'bf16_gemm_tt',
        'm_grouped_bf16_gemm_nt_contiguous',
        'm_grouped_bf16_gemm_nn_contiguous',
        'm_grouped_bf16_gemm_nt_masked',
        'k_grouped_bf16_gemm_tn_contiguous',
        # Einsum kernels
        'einsum',
        'fp8_einsum',
        # Attention kernels
        'fp8_fp4_mqa_logits',
        'get_paged_mqa_logits_metadata',
        'fp8_fp4_paged_mqa_logits',
        # Attention kernels (legacy)
        'fp8_mqa_logits',
        'fp8_paged_mqa_logits',
        # Hyperconnection kernels
        'tf32_hc_prenorm_gemm',
        # Layout kernels
        'transform_sf_into_required_layout',
    )
    # Bind per-name: one kernel absent from this build (e.g. plain FP8 GEMMs
    # not exported by the TVM-FFI bridge, or CUDA < 12.1) must not drop the rest.
    for _name in _kernel_exports:
        try:
            _bind_exports(_name)
        except AttributeError:
            pass

    # Sugared recipe-tuple form matching the sgl_deep_gemm wheel API
    # (the raw _C entry takes the recipe flattened into 3 ints).
    if hasattr(_C, 'transform_sf_into_required_layout'):
        def transform_sf_into_required_layout(sf, mn, k, recipe, num_groups=None,
                                              is_sfa=None, disable_ue8m0_cast=False):
            (recipe_a, recipe_b, recipe_c) = recipe if len(recipe) == 3 else (recipe[0], recipe[1], None)
            return _C.transform_sf_into_required_layout(
                sf, mn, k, recipe_a, recipe_b, recipe_c, num_groups, is_sfa, disable_ue8m0_cast)

    # Some alias for legacy supports
    # TODO: remove these later
    if 'm_grouped_fp8_gemm_nt_masked' in globals():
        fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_gemm_nt_masked
    if 'm_grouped_bf16_gemm_nt_masked' in globals():
        bf16_m_grouped_gemm_nt_masked = m_grouped_bf16_gemm_nt_masked
except AttributeError:
    # Expected behavior for CUDA runtime version before 12.1
    pass

# Mega kernels
from .mega import (
    SymmBuffer,
    get_symm_buffer_for_mega_moe,
    transform_weights_for_mega_moe,
    fp8_fp4_mega_moe,
    nvfp4_mega_moe,
    bf16_mega_moe,
    mega_moe_pre_dispatch,
)

# Some utils
from . import testing
from . import utils
from .utils import *

# Legacy Triton kernels for A100
try:
    from . import legacy
except Exception as e:
    if not (isinstance(e, ImportError) and 'PyInit__C' in str(e)):
        print(f'Failed to load legacy DeepGEMM A100 Triton kernels: {e}')

# Initialize CPP modules
def _find_cuda_home() -> str:
    # TODO: reuse PyTorch API later
    # For some PyTorch versions, the original `_find_cuda_home` will initialize CUDA, which is incompatible with process forks
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # noinspection PyBroadException
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


_C.init(
    os.path.dirname(os.path.abspath(__file__)), # Library root directory path
    _find_cuda_home()                           # CUDA home
)

__version__ = '2.6.1'
