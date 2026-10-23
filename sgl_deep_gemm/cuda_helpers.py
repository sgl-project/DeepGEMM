from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def find_cuda_home() -> str:
    """Locate the CUDA installation DeepGEMM will use.

    Raises ``RuntimeError`` with an actionable message rather than a bare
    ``AssertionError``: ``import deep_gemm`` calls this at module-import time
    (the kernel JIT compiles at run time, so CUDA is needed even to import),
    which makes this the very first error most users hit.
    """
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    source = 'CUDA_HOME/CUDA_PATH'
    if cuda_home is None:
        nvcc = shutil.which('nvcc')
        if nvcc is not None:
            cuda_home = str(Path(nvcc).parent.parent)
            source = 'inferred from `nvcc` on PATH'
        else:
            source = 'the /usr/local/cuda default'
            cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None

    if cuda_home is None:
        raise RuntimeError(
            'Cannot locate a CUDA installation.\n'
            '\n'
            '  CUDA_HOME / CUDA_PATH is not set, there is no `nvcc` on PATH, and\n'
            '  /usr/local/cuda does not exist.\n'
            '\n'
            'DeepGEMM needs CUDA even just to be imported, because its kernels are\n'
            'compiled by a JIT at run time.\n'
            '\n'
            'Point CUDA_HOME at a directory that contains both of:\n'
            '\n'
            '  <CUDA_HOME>/bin/nvcc                  the CUDA compiler\n'
            '  <CUDA_HOME>/include/curand_kernel.h   the CUDA device headers\n'
            '\n'
            'On Jetson these are sometimes split across two prefixes. Either point\n'
            'CUDA_HOME at a directory that holds both (a symlink farm is fine), or\n'
            'add the device-header directory to CPATH.\n'
            '\n'
            '  export CUDA_HOME=/path/to/cuda\n'
        )
    if not os.path.isdir(cuda_home):
        raise RuntimeError(
            f'CUDA_HOME resolves to {cuda_home!r} (via {source}), but that is not an existing directory.\n'
            '\n'
            'Set it to a directory that contains both <CUDA_HOME>/bin/nvcc and\n'
            '<CUDA_HOME>/include/curand_kernel.h, or unset it and put `nvcc` on PATH.\n'
        )
    return cuda_home


def cuda_diagnostics() -> dict:
    """Describe the CUDA install without raising, for `python -m deep_gemm.doctor`."""
    info = {'cuda_home': None, 'nvcc': None, 'device_headers': None, 'error': None}
    try:
        cuda_home = find_cuda_home()
    except RuntimeError as exc:
        info['error'] = str(exc).splitlines()[0]
        return info
    info['cuda_home'] = cuda_home
    nvcc = Path(cuda_home) / 'bin' / 'nvcc'
    info['nvcc'] = nvcc.as_posix() if nvcc.is_file() else None
    header = Path(cuda_home) / 'include' / 'curand_kernel.h'
    info['device_headers'] = header.as_posix() if header.is_file() else None
    return info


def get_cuda_arch(default: str = '9.0') -> str:
    try:
        status = subprocess.run(
            args=['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True,
            check=True,
        )
        return status.stdout.decode('utf-8').strip().split('\n')[0]
    except Exception:
        return default
