from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

import torch
import tvm_ffi

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Set some default environment provided at setup
try:
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Build & load the tvm-ffi _C module
# ---------------------------------------------------------------------------
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


def _build_module(pkg_dir: str, cuda_home: str) -> str:
    """Build the _C shared library using tvm_ffi.cpp.build()."""
    import tvm_ffi.cpp

    root_dir = os.path.dirname(pkg_dir)
    cxx_abi = int(torch.compiled_with_cxx11_abi())

    os.environ.setdefault('TVM_FFI_CUDA_ARCH_LIST', _get_cuda_arch())

    extra_cflags = [
        '-std=c++17', '-O3', '-fPIC',
        '-Wno-psabi', '-Wno-deprecated-declarations',
        f'-D_GLIBCXX_USE_CXX11_ABI={cxx_abi}',
    ]
    if int(os.environ.get('DG_JIT_USE_RUNTIME_API', '0')):
        extra_cflags.append('-DDG_JIT_USE_RUNTIME_API')

    extra_include_paths = [
        f'{cuda_home}/include',
        os.path.join(root_dir, 'deep_gemm', 'include'),
        os.path.join(root_dir, 'third-party', 'cutlass', 'include'),
        os.path.join(root_dir, 'third-party', 'fmt', 'include'),
    ]
    cccl_path = f'{cuda_home}/include/cccl'
    if os.path.exists(cccl_path):
        extra_include_paths.append(cccl_path)

    extra_ldflags = [
        f'-L{cuda_home}/lib64',
        '-lcudart',
        '-lnvrtc',
        '-lcublasLt',
        '-lcublas',
    ]

    build_dir = os.path.join(pkg_dir, '_C_build')
    os.makedirs(build_dir, exist_ok=True)

    lib_path = tvm_ffi.cpp.build(
        name='_C',
        cpp_files=[os.path.join(root_dir, 'csrc', 'tvm_ffi_api.cpp')],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_dir,
    )
    # Copy the .so into the package directory for easy loading
    target = os.path.join(pkg_dir, '_C.so')
    shutil.copy2(lib_path, target)
    return target


def _get_cuda_arch() -> str:
    try:
        status = subprocess.run(
            args=['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True, check=True,
        )
        return status.stdout.decode('utf-8').strip().split('\n')[0]
    except Exception:
        return '9.0'


def _load_module() -> Module:
    """Load (or build then load) the compiled tvm-ffi module."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(pkg_dir, '_C.so')

    if not os.path.exists(lib_path):
        cuda_home = _find_cuda_home()
        print(f'[DeepGEMM] Building _C module with tvm-ffi (CUDA_HOME={cuda_home})...')
        lib_path = _build_module(pkg_dir, cuda_home)
        print(f'[DeepGEMM] Built _C module: {lib_path}')

    return tvm_ffi.load_module(lib_path)

_C: Module = _load_module()

# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------
set_num_sms = _C.set_num_sms
get_num_sms = _C.get_num_sms
set_compile_mode = _C.set_compile_mode
get_compile_mode = _C.get_compile_mode
set_tc_util = _C.set_tc_util
get_tc_util = _C.get_tc_util

# cuBLASLt Kernels
cublaslt_gemm_nt = _C.cublaslt_gemm_nt
cublaslt_gemm_nn = _C.cublaslt_gemm_nn
cublaslt_gemm_tn = _C.cublaslt_gemm_tn
cublaslt_gemm_tt = _C.cublaslt_gemm_tt

# ---------------------------------------------------------------------------
# GEMM / Attention / Einsum wrappers (handle optional params in Python)
# ---------------------------------------------------------------------------
def _opt_recipe(recipe):
    if recipe is None:
        return -1, -1, -1
    return recipe[0], recipe[1], recipe[2]

def _opt_recipe_ab(recipe_ab):
    if recipe_ab is None:
        return -1, -1
    return recipe_ab[0], recipe_ab[1]

_DUMMY = None

def _dummy_tensor(ref):
    global _DUMMY
    if _DUMMY is None or _DUMMY.device != ref.device:
        _DUMMY = torch.zeros(1, device=ref.device, dtype=torch.float32)
    return _DUMMY

try:
    def fp8_fp4_gemm_nt(a_data, a_sf, b_data, b_sf, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='', disable_ue8m0_cast=False):
        r = _opt_recipe(recipe); ra = _opt_recipe_ab(recipe_a); rb = _opt_recipe_ab(recipe_b)
        _C.fp8_fp4_gemm_nt(a_data, a_sf, b_data, b_sf, d, int(c is not None), c if c is not None else _dummy_tensor(d), r[0], r[1], r[2], ra[0], ra[1], rb[0], rb[1], compiled_dims, disable_ue8m0_cast)

    def fp8_fp4_gemm_nn(a_data, a_sf, b_data, b_sf, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='', disable_ue8m0_cast=False):
        r = _opt_recipe(recipe); ra = _opt_recipe_ab(recipe_a); rb = _opt_recipe_ab(recipe_b)
        _C.fp8_fp4_gemm_nn(a_data, a_sf, b_data, b_sf, d, int(c is not None), c if c is not None else _dummy_tensor(d), r[0], r[1], r[2], ra[0], ra[1], rb[0], rb[1], compiled_dims, disable_ue8m0_cast)

    def fp8_fp4_gemm_tn(a_data, a_sf, b_data, b_sf, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='', disable_ue8m0_cast=False):
        r = _opt_recipe(recipe); ra = _opt_recipe_ab(recipe_a); rb = _opt_recipe_ab(recipe_b)
        _C.fp8_fp4_gemm_tn(a_data, a_sf, b_data, b_sf, d, int(c is not None), c if c is not None else _dummy_tensor(d), r[0], r[1], r[2], ra[0], ra[1], rb[0], rb[1], compiled_dims, disable_ue8m0_cast)

    def fp8_fp4_gemm_tt(a_data, a_sf, b_data, b_sf, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='', disable_ue8m0_cast=False):
        r = _opt_recipe(recipe); ra = _opt_recipe_ab(recipe_a); rb = _opt_recipe_ab(recipe_b)
        _C.fp8_fp4_gemm_tt(a_data, a_sf, b_data, b_sf, d, int(c is not None), c if c is not None else _dummy_tensor(d), r[0], r[1], r[2], ra[0], ra[1], rb[0], rb[1], compiled_dims, disable_ue8m0_cast)

    fp8_gemm_nt = fp8_fp4_gemm_nt
    fp8_gemm_nn = fp8_fp4_gemm_nn
    fp8_gemm_tn = fp8_fp4_gemm_tn
    fp8_gemm_tt = fp8_fp4_gemm_tt

    def bf16_gemm_nt(a, b, d, c=None, compiled_dims=''):
        _C.bf16_gemm_nt(a, b, d, int(c is not None), c if c is not None else _dummy_tensor(d), compiled_dims)

    def bf16_gemm_nn(a, b, d, c=None, compiled_dims=''):
        _C.bf16_gemm_nn(a, b, d, int(c is not None), c if c is not None else _dummy_tensor(d), compiled_dims)

    def bf16_gemm_tn(a, b, d, c=None, compiled_dims=''):
        _C.bf16_gemm_tn(a, b, d, int(c is not None), c if c is not None else _dummy_tensor(d), compiled_dims)

    def bf16_gemm_tt(a, b, d, c=None, compiled_dims=''):
        _C.bf16_gemm_tt(a, b, d, int(c is not None), c if c is not None else _dummy_tensor(d), compiled_dims)

    def einsum(expr, a, b, d, c=None, use_cublaslt=False):
        _C.einsum(expr, a, b, d, int(c is not None), c if c is not None else _dummy_tensor(d), use_cublaslt)

    def fp8_einsum(expr, a_data, a_sf, b_data, b_sf, d, c=None, recipe=(1, 128, 128)):
        _C.fp8_einsum(expr, a_data, a_sf, b_data, b_sf, d, int(c is not None), c if c is not None else _dummy_tensor(d), recipe[0], recipe[1], recipe[2])

    def fp8_gemm_nt_skip_head_mid(a_data, a_sf, b_data, b_sf, d, head_splits, recipe=None, compiled_dims='', disable_ue8m0_cast=False):
        r = _opt_recipe(recipe)
        _C.fp8_gemm_nt_skip_head_mid(a_data, a_sf, b_data, b_sf, d, head_splits[0], head_splits[1], head_splits[2], r[0], r[1], r[2], compiled_dims, disable_ue8m0_cast)

    fp8_mqa_logits = _C.fp8_mqa_logits
    get_paged_mqa_logits_metadata = _C.get_paged_mqa_logits_metadata
    fp8_paged_mqa_logits = _C.fp8_paged_mqa_logits

    def tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=None):
        _C.tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits if num_splits is not None else -1)

    transform_sf_into_required_layout = _C.transform_sf_into_required_layout
    get_mk_alignment_for_contiguous_layout = _C.get_mk_alignment_for_contiguous_layout

    fp8_m_grouped_gemm_nt_masked = None
    bf16_m_grouped_gemm_nt_masked = None

except AttributeError:
    pass

# Some utils
from . import testing
from . import utils
from .utils import *

# Legacy Triton kernels for A100
try:
    from . import legacy
except Exception as e:
    print(f'Failed to load legacy DeepGEMM A100 Triton kernels: {e}')

# Initialize CPP modules
_C.init(
    os.path.dirname(os.path.abspath(__file__)),
    _find_cuda_home()
)

__version__ = '2.3.0'
