import os
import subprocess
import sys
import tempfile


TEST_PROGRAM = r'''
import os
from pathlib import Path
import re

import torch

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major
from deep_gemm.utils import ceil_div, per_token_cast_to_fp8


if get_arch_major() == 12:
    cache_dir = Path(os.environ['DG_JIT_CACHE_DIR']) / 'cache'

    def kernel_dirs(name):
        return set(cache_dir.glob(f'kernel.{name}.*'))

    def assert_compiled_shape(kernel_dir, expected):
        source = (kernel_dir / 'kernel.cu').read_text()
        match = re.search(
            r'sm120_fp8_fp4_gemm_1d1d_impl<\s*(\d+),\s*(\d+),\s*(\d+),',
            source,
        )
        assert match is not None
        assert tuple(map(int, match.groups())) == expected

    h, n, k = 4, 1024, 4096
    weight = torch.randn((h, n, k), device='cuda', dtype=torch.bfloat16)
    weight_fp8 = per_token_cast_to_fp8(weight.view(-1, k), use_ue8m0=True)
    weight_fp8 = (
        weight_fp8[0].view(h, n, k),
        weight_fp8[1].view(h, n, ceil_div(k, 128)),
    )

    bmm_kernels = None
    for tokens in (3, 11):
        activation = torch.randn((tokens, h, k), device='cuda', dtype=torch.bfloat16)
        reference = torch.einsum('bhr,hdr->bhd', activation, weight)
        activation_fp8 = per_token_cast_to_fp8(
            activation.view(-1, k), use_ue8m0=True
        )
        activation_fp8 = (
            activation_fp8[0].view(tokens, h, k),
            activation_fp8[1].view(tokens, h, ceil_div(k, 128)),
        )
        output = torch.empty((tokens, h, n), device='cuda', dtype=torch.bfloat16)

        deep_gemm.fp8_einsum(
            'bhr,hdr->bhd',
            activation_fp8,
            weight_fp8,
            output,
            recipe=(1, 1, 128),
        )
        assert calc_diff(output, reference) < 1e-3

        current_kernels = kernel_dirs('sm120_fp8_fp4_bmm')
        if bmm_kernels is None:
            assert len(current_kernels) == 1
            assert_compiled_shape(next(iter(current_kernels)), (1024, 0, 4096))
            bmm_kernels = current_kernels
        else:
            assert current_kernels == bmm_kernels

    matrix_weight = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    matrix_weight_fp8 = per_token_cast_to_fp8(matrix_weight, use_ue8m0=True)

    gemm_kernels = None
    for tokens in (3, 11):
        activation = torch.randn((tokens, k), device='cuda', dtype=torch.bfloat16)
        reference = activation @ matrix_weight.T
        activation_fp8 = per_token_cast_to_fp8(activation, use_ue8m0=True)
        output = torch.empty((tokens, n), device='cuda', dtype=torch.bfloat16)

        deep_gemm.fp8_fp4_gemm_nt(
            activation_fp8,
            matrix_weight_fp8,
            output,
            recipe=(1, 1, 128),
            compiled_dims='nk',
        )
        assert calc_diff(output, reference) < 1e-3

        current_kernels = kernel_dirs('sm120_fp8_fp4_gemm_1d1d')
        if gemm_kernels is None:
            assert len(current_kernels) == 1
            assert_compiled_shape(next(iter(current_kernels)), (1024, 0, 4096))
            gemm_kernels = current_kernels
        else:
            assert current_kernels == gemm_kernels
'''


def test_sm120_swap_ab_runtime_token_dim():
    with tempfile.TemporaryDirectory(prefix='deep-gemm-runtime-dim-') as cache_dir:
        env = os.environ.copy()
        env['DG_JIT_CACHE_DIR'] = cache_dir
        subprocess.run(
            [sys.executable, '-c', TEST_PROGRAM],
            check=True,
            cwd=cache_dir,
            env=env,
        )


if __name__ == '__main__':
    test_sm120_swap_ab_runtime_token_dim()
