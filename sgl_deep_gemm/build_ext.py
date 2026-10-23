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
    # `torch/python.h` pulls in `Python.h`, so the interpreter's include dirs are
    # required even though the extension itself is a plain TORCH_LIBRARY.
    import sysconfig
    for key in ('include', 'platinclude'):
        path = sysconfig.get_paths().get(key)
        if path and path not in include_dirs:
            include_dirs.append(path)

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
        # `torch/python.h` drags in the CPython-facing torch binding, whose symbols
        # (e.g. `THPDtypeType`) live in libtorch_python rather than libtorch.
        '-ltorch_python',
        # The extension is imported as a Python module, so the loader resolves its
        # dependencies without a pre-loaded torch: record both search paths.
        f'-Wl,-rpath,{cuda_home}/lib64',
        f'-Wl,-rpath,{os.path.join(torch_dir, "lib")}',
    ]
    # Upstream renamed the TORCH_LIBRARY translation unit from
    # `torch_library_api.cpp` to `python_api.cpp`; accept either.
    for name in ('python_api.cpp', 'torch_library_api.cpp'):
        source = os.path.join(root_dir, 'csrc', name)
        if os.path.exists(source):
            break
    else:
        raise FileNotFoundError('no TORCH_LIBRARY source (python_api.cpp / torch_library_api.cpp) under csrc/')
    return [os.environ.get('CXX', 'c++'), *cflags,
            *(f'-I{d}' for d in include_dirs),
            source,
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
