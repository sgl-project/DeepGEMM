import ast
import os
import re
import shutil
import setuptools
import subprocess
import sys
import platform
from setuptools import find_packages
from setuptools.command.build_py import build_py
from packaging.version import parse
from pathlib import Path

DG_SKIP_CUDA_BUILD = int(os.getenv('DG_SKIP_CUDA_BUILD', '0')) == 1
DG_FORCE_BUILD = int(os.getenv('DG_FORCE_BUILD', '0')) == 1
DG_USE_LOCAL_VERSION = int(os.getenv('DG_USE_LOCAL_VERSION', '1')) == 1
DG_JIT_USE_RUNTIME_API = int(os.environ.get('DG_JIT_USE_RUNTIME_API', '0')) == 1

# Distribution identity overrides (used by the sgl-deepgemm release workflow).
# These let a downstream packager publish to PyPI under a different name and
# version without touching deep_gemm/__init__.py. The on-disk import name stays
# `deep_gemm`; only the dist metadata changes.
SGL_DEEPGEMM_PACKAGE = os.getenv('SGL_DEEPGEMM_PACKAGE', '').strip()
SGL_DEEPGEMM_VERSION = os.getenv('SGL_DEEPGEMM_VERSION', '').strip()

current_dir = os.path.dirname(os.path.realpath(__file__))

third_party_include_dirs = [
    'third-party/cutlass/include/cute',
    'third-party/cutlass/include/cutlass',
]


def _find_cuda_home():
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        nvcc_path = shutil.which('nvcc')
        if nvcc_path is not None:
            cuda_home = str(Path(nvcc_path).parent.parent)
        else:
            cuda_home = '/usr/local/cuda'
            if not Path(cuda_home).exists():
                cuda_home = None
    assert cuda_home is not None, 'Cannot find CUDA_HOME'
    return cuda_home


CUDA_HOME = _find_cuda_home()


def get_package_version():
    if SGL_DEEPGEMM_VERSION:
        return SGL_DEEPGEMM_VERSION

    with open(Path(current_dir) / 'deep_gemm' / '__init__.py', 'r') as f:
        version_match = re.search(r'^__version__\s*=\s*(.*)$', f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))

    revision = ''
    if DG_USE_LOCAL_VERSION:
        try:
            cmd = ['git', 'rev-parse', '--short', 'HEAD']
            revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            revision = '+local'
    return f'{public_version}{revision}'


def _get_cuda_arch():
    try:
        status = subprocess.run(
            args=['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True, check=True,
        )
        return status.stdout.decode('utf-8').strip().split('\n')[0]
    except Exception:
        return '9.0'


def build_tvm_ffi_module(build_lib_dir):
    """Build the _C module using tvm_ffi.cpp.build()."""
    if DG_SKIP_CUDA_BUILD:
        return

    import tvm_ffi.cpp
    import torch

    cxx_abi = int(torch.compiled_with_cxx11_abi())
    arch = _get_cuda_arch()
    os.environ.setdefault('TVM_FFI_CUDA_ARCH_LIST', arch)

    extra_cflags = [
        '-std=c++17', '-O3', '-fPIC',
        '-Wno-psabi', '-Wno-deprecated-declarations',
        f'-D_GLIBCXX_USE_CXX11_ABI={cxx_abi}',
    ]
    if DG_JIT_USE_RUNTIME_API:
        extra_cflags.append('-DDG_JIT_USE_RUNTIME_API')

    import sysconfig
    torch_dir = os.path.dirname(torch.__file__)
    torch_include = os.path.join(torch_dir, 'include')
    torch_include_csrc = os.path.join(torch_include, 'torch', 'csrc', 'api', 'include')
    torch_lib = os.path.join(torch_dir, 'lib')
    python_include = sysconfig.get_path('include')

    extra_include_paths = [
        f'{CUDA_HOME}/include',
        python_include,
        torch_include,
        torch_include_csrc,
        os.path.join(current_dir, 'deep_gemm', 'include'),
        os.path.join(current_dir, 'third-party', 'cutlass', 'include'),
        os.path.join(current_dir, 'third-party', 'fmt', 'include'),
    ]
    cccl_path = f'{CUDA_HOME}/include/cccl'
    if os.path.exists(cccl_path):
        extra_include_paths.append(cccl_path)

    extra_ldflags = [
        f'-L{CUDA_HOME}/lib64',
        f'-L{torch_lib}',
        '-lcudart',
        '-lnvrtc',
        '-lcublasLt',
        '-lcublas',
        '-ltorch',
        '-ltorch_cpu',
        '-lc10',
        '-lc10_cuda',
        '-ltorch_cuda',
    ]

    output_dir = os.path.join(build_lib_dir, 'deep_gemm')
    os.makedirs(output_dir, exist_ok=True)

    lib_path = tvm_ffi.cpp.build(
        name='_C',
        cpp_files=[os.path.join(current_dir, 'csrc', 'tvm_ffi_api.cpp')],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=os.path.join(output_dir, '_C_build'),
    )

    target = os.path.join(output_dir, '_C.so')
    if os.path.exists(target):
        os.remove(target)
    shutil.copy2(lib_path, target)
    print(f'Built tvm-ffi module: {target}')


class CustomBuildPy(build_py):
    def run(self):
        self.prepare_includes()
        self.generate_default_envs()
        build_tvm_ffi_module(self.build_lib)
        build_py.run(self)

    def generate_default_envs(self):
        code = '# Pre-installed environment variables\n'
        code += 'persistent_envs = dict()\n'
        for name in ('DG_JIT_CACHE_DIR', 'DG_JIT_PRINT_COMPILER_COMMAND', 'DG_JIT_CPP_STANDARD'):
            code += f"persistent_envs['{name}'] = '{os.environ[name]}'\n" if name in os.environ else ''

        envs_dir = os.path.join(self.build_lib, 'deep_gemm')
        os.makedirs(envs_dir, exist_ok=True)
        with open(os.path.join(envs_dir, 'envs.py'), 'w') as f:
            f.write(code)

    def prepare_includes(self):
        build_include_dir = os.path.join(self.build_lib, 'deep_gemm/include')
        os.makedirs(build_include_dir, exist_ok=True)

        for d in third_party_include_dirs:
            dirname = d.split('/')[-1]
            src_dir = os.path.join(current_dir, d)
            dst_dir = os.path.join(build_include_dir, dirname)

            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir, ignore_dangling_symlinks=True)


if __name__ == '__main__':
    setuptools.setup(
        name=SGL_DEEPGEMM_PACKAGE or 'deep_gemm',
        version=get_package_version(),
        packages=find_packages('.'),
        package_data={
            'deep_gemm': [
                'include/deep_gemm/**/*',
                'include/cute/**/*',
                'include/cutlass/**/*',
            ]
        },
        ext_modules=[],
        zip_safe=False,
        cmdclass={
            'build_py': CustomBuildPy,
        },
        install_requires=[
            'apache-tvm-ffi',
        ],
    )
