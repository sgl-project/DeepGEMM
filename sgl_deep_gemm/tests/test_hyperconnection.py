import torch
import random
import re
import shutil
import tempfile
from pathlib import Path

import deep_gemm
from deep_gemm.testing import (
    test_filter,
    bench_kineto,
    calc_diff, count_bytes
)
from deep_gemm.utils import align
from generators import get_arch_major


@test_filter(lambda: get_arch_major() >= 9)
def test_hc_prenorm_gemm() -> None:
    # Needs TF32 precision for PyTorch GEMMs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print('Testing hyperconnection prenorm GEMM:')
    for m in (13, 137, 4096, 8192):
        for n, k in [(24, 28672), (24, 7680), (24, 7168)]:
            for num_splits in [None, 16]:
                a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
                b = torch.randn((n, k), dtype=torch.float, device='cuda')
                d = torch.empty((m, n), dtype=torch.float, device='cuda') if num_splits is None else \
                        torch.empty((num_splits, m, n), dtype=torch.float, device='cuda')
                s = torch.empty((m, ), dtype=torch.float, device='cuda') if num_splits is None else \
                        torch.empty((num_splits, m), dtype=torch.float, device='cuda')
                deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits)
                final_d = d if num_splits is None else d.sum(0)
                final_s = s if num_splits is None else s.sum(0)

                ref_d = a.float() @ b.T
                ref_s = a.float().square().sum(-1)

                diff = max(calc_diff(final_d, ref_d), calc_diff(final_s, ref_s))
                assert diff < 1e-8, f'{m=}, {n=}, {k=}, {diff:.10f}'

                t = bench_kineto(lambda: deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits), 'tf32_hc_prenorm_gemm', suppress_kineto_output=True)
                print(f' > Perf (m={m:5}, n={n:5}, k={k:5}, num_splits={(num_splits or 0):2}): '
                      f'{t * 1e6:4.0f} us | '
                      f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
                      f'{count_bytes(a, b, d, s) / 1e9 / t:4.0f} GB/s')
    print()




def _check_hc_prenorm_gemm_concurrent_streams() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cases = []
    for m in (13, 137):
        for num_splits in (None, 16):
            n, k = 24, 7168
            a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
            b = torch.randn((n, k), dtype=torch.float, device='cuda')
            d_shape = (m, n) if num_splits is None else (num_splits, m, n)
            s_shape = (m,) if num_splits is None else (num_splits, m)
            d = torch.empty(d_shape, dtype=torch.float, device='cuda')
            s = torch.empty(s_shape, dtype=torch.float, device='cuda')
            # Compile each specialization before queueing concurrent work.
            deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits)
            ref_d = a.float() @ b.T
            ref_s = a.float().square().sum(-1)
            cases.append((a, b, num_splits, d_shape, s_shape, ref_d, ref_s))

    streams = [torch.cuda.Stream() for _ in cases]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())

    outputs = []
    for _ in range(8):
        for stream, (a, b, num_splits, d_shape, s_shape, ref_d, ref_s) in zip(streams, cases):
            with torch.cuda.stream(stream):
                d = torch.empty(d_shape, dtype=torch.float, device='cuda')
                s = torch.empty(s_shape, dtype=torch.float, device='cuda')
                deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits)
                outputs.append((d, s, num_splits, ref_d, ref_s))
    torch.cuda.synchronize()

    for d, s, num_splits, ref_d, ref_s in outputs:
        final_d = d if num_splits is None else d.sum(0)
        final_s = s if num_splits is None else s.sum(0)
        diff = max(calc_diff(final_d, ref_d), calc_diff(final_s, ref_s))
        assert diff < 1e-8, f'{num_splits=}, {diff:.10f}'
    print('Concurrent HC-prenorm GEMM passed')


@test_filter(lambda: get_arch_major() == 10)
def test_hc_prenorm_gemm_nonzero_tmem_base() -> None:
    library_root = Path(deep_gemm.__file__).resolve().parent
    with tempfile.TemporaryDirectory() as directory:
        include = Path(directory) / 'include'
        shutil.copytree(library_root / 'include', include, symlinks=True)
        header = include / 'deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh'
        source = header.read_text()
        allocation = '        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);\n'
        assert source.count(allocation) == 1
        source = source.replace(allocation,
            '        cute::TMEM::Allocator1Sm().allocate(256, &reserved_tmem_base);\n' +
            allocation + '        cute::TMEM::Allocator1Sm().release_allocation_lock();\n')
        # Reserve half the TMEM in the same CTA so the kernel must use a nonzero base.
        initialization = '    // Initialize barriers\n'
        assert source.count(initialization) == 1
        source = source.replace(initialization,
            '    DG_STATIC_ASSERT(kNumTmemCols == 256, "Test requires 256 kernel columns");\n'
            '    __shared__ uint32_t reserved_tmem_base;\n\n' + initialization)
        synchronization = source.index('    __syncthreads();\n', source.index(allocation))
        synchronization += len('    __syncthreads();\n')
        source = (source[:synchronization] +
                  '    DG_DEVICE_ASSERT(*tmem_ptr_in_smem != 0);\n' + source[synchronization:])
        # Free the fixture first, leaving the kernel's original deallocation untouched.
        source, count = re.subn(
            r'        if \(warp_idx == 1\)\n(            cute::TMEM::Allocator1Sm\(\).free\([^\n]+\);)\n',
            r'        if (warp_idx == 1) {\n'
            r'            cute::TMEM::Allocator1Sm().free(reserved_tmem_base, 256);\n\1\n        }\n',
            source)
        assert count == 1
        header.write_text(source)
        torch.cuda.synchronize()
        deep_gemm._C.init(directory, '')
        try:
            _check_hc_prenorm_gemm_concurrent_streams()
        finally:
            deep_gemm._C.init(str(library_root), '')


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_hc_prenorm_gemm()
    test_hc_prenorm_gemm_nonzero_tmem_base()
