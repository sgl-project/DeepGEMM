"""Build the `TORCH_LIBRARY` `_C.so` (no CPython dependency: one build serves every Python version)."""
from __future__ import annotations

import os
import shlex
import subprocess
from typing import List

import torch


def get_compile_command(root_dir: str, output_path: str, cuda_home: str) -> List[str]:
    torch_dir = os.path.dirname(torch.__file__)
    include_dirs = [
        f'{cuda_home}/include',
        os.path.join(torch_dir, 'include'),
        os.path.join(torch_dir, 'include', 'torch', 'csrc', 'api', 'include'),
        os.path.join(root_dir, 'deep_gemm', 'include'),
        os.path.join(root_dir, 'third-party', 'cutlass', 'include'),
        os.path.join(root_dir, 'third-party', 'deep_jit', 'include'),
    ]
    if os.path.exists(f'{cuda_home}/include/cccl'):
        include_dirs.append(f'{cuda_home}/include/cccl')

    cflags = [
        '-std=c++20', '-O3', '-fPIC', '-shared',
        '-Wno-psabi', '-Wno-deprecated-declarations',
        f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}',
    ]
    if int(os.environ.get('DG_JIT_USE_RUNTIME_API', '0')):
        cflags.append('-DDG_JIT_USE_RUNTIME_API')

    ldflags = [
        f'-L{cuda_home}/lib64', f'-L{os.path.join(torch_dir, "lib")}',
        '-lcudart', '-lnvrtc', '-lcublasLt', '-lcublas',
        '-ltorch', '-ltorch_cpu', '-lc10', '-lc10_cuda', '-ltorch_cuda',
    ]
    return [os.environ.get('CXX', 'c++'), *cflags,
            *(f'-I{d}' for d in include_dirs),
            os.path.join(root_dir, 'csrc', 'torch_library_api.cpp'),
            '-o', output_path, *ldflags]


def build_extension(root_dir: str, output_path: str, cuda_home: str) -> str:
    command = get_compile_command(root_dir, output_path, cuda_home)
    print(f'[DeepGEMM] {shlex.join(command)}', flush=True)
    subprocess.check_call(command)
    return output_path


if __name__ == '__main__':
    import sys
    # Run as a script, this directory is on `sys.path`, so the sibling module imports directly
    from cuda_helpers import find_cuda_home

    root_dir, output_path = sys.argv[1:]
    build_extension(root_dir, output_path, find_cuda_home())
