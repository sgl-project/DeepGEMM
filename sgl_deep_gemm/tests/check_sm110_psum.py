"""Run test_layout.py's k-grouped psum SF section on SM110, bypassing its guard.

`test_k_grouped_psum_sf_layout_kernels()` returns early unless
`get_arch_major() == 10`:

    > Skipped (packed k-grouped SF only supported on SM100)

With the SM110 patch, `csrc/apis/layout.hpp` *does* route arch 11 into that
packed psum path (the predicates at lines 109/113 accept 10 or 11), so the
kernel is reachable on Thor while the upstream suite refuses to exercise it.
That is exactly the "silently wrong" risk that needs its own check.

This script is the body of that test with the arch guard removed, so the same
assertions run on SM110. Output format is kept for easy diffing.
"""
import random
import sys

import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deep_gemm.testing import bench_kineto, count_bytes, get_arch_major  # noqa: E402
from deep_gemm.utils import (  # noqa: E402
    align, ceil_div, ceil_to_ue8m0,
    get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor,
)
from generators import enumerate_k_grouped_psum_sf_layout  # noqa: E402
from test_layout import get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl  # noqa: E402


def run():
    import deep_gemm
    print(f'arch_major      : {get_arch_major()}')
    print(f'deep_gemm       : {deep_gemm.__version__}')
    if hasattr(deep_gemm, 'supports_arch'):
        print(f'supports_arch(11): {deep_gemm.supports_arch(11)}')
    else:
        print('supports_arch    : not present in this install (pre-R4 build)')
    print()

    total = failed = 0
    cases = 0
    for mn, real_ks_cpu, aligned_ks_cpu, psum_layout, num_groups, gran_k, k_alignment \
            in enumerate_k_grouped_psum_sf_layout():
        cases += 1
        for compact_sf in (False, True):
            total += 1
            pack_kwargs = {} if compact_sf else dict(use_padded_sf_layout=True)
            grouped_layout = torch.tensor(psum_layout, dtype=torch.int, device='cuda')

            sf_ks = [ceil_div(k if compact_sf else align(k, k_alignment), gran_k) for k in real_ks_cpu]
            packed_sf_ks = [ceil_div(sf_k, 4) for sf_k in sf_ks]
            num_sf_rows = sum(sf_ks)
            fp32_sf = ceil_to_ue8m0(torch.rand((num_sf_rows, mn), dtype=torch.float, device='cuda') + 0.5)
            ref_packed_sf = torch.cat([
                get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(group_sf.T).T
                for group_sf in fp32_sf.split(sf_ks) if group_sf.size(0) > 0
            ])

            label = f'(num_groups={num_groups}, mn={mn}, sum_k={sum(real_ks_cpu)}, ' \
                    f'gran_k={gran_k}, k_align={k_alignment}, compact={int(compact_sf)})'
            try:
                packed_sf = get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                    fp32_sf, grouped_layout, aligned_ks_cpu, gran_k, k_alignment,
                    use_psum_layout=True, **pack_kwargs)
                if compact_sf:
                    assert torch.equal(packed_sf[:sum(packed_sf_ks)], ref_packed_sf), 'compact prefix mismatch'
                    exact = get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                        fp32_sf, grouped_layout, real_ks_cpu, gran_k, k_alignment, use_psum_layout=True)
                    assert torch.equal(exact, ref_packed_sf), 'compact exact mismatch'
                else:
                    assert torch.equal(packed_sf, ref_packed_sf), 'padded mismatch'

                unsynced = get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                    fp32_sf, grouped_layout, None, gran_k, k_alignment,
                    use_psum_layout=True, **pack_kwargs)
                assert torch.equal(unsynced[:sum(packed_sf_ks)], ref_packed_sf), 'unsynced mismatch'

                empty = get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                    fp32_sf, grouped_layout, [], gran_k, k_alignment,
                    use_psum_layout=True, **pack_kwargs)
                assert torch.equal(empty[:sum(packed_sf_ks)], ref_packed_sf), 'empty-ks mismatch'

                t = bench_kineto(
                    lambda: get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                        fp32_sf, grouped_layout, aligned_ks_cpu, gran_k, k_alignment,
                        use_psum_layout=True, **pack_kwargs),
                    'pack_fp32_into_ue8m0', suppress_kineto_output=True)
                print(f' > Perf {label}: {t * 1e6:4.0f} us | '
                      f'{count_bytes(fp32_sf, packed_sf, grouped_layout) / 1e9 / t:4.0f} GB/s')
            except AssertionError as exc:
                failed += 1
                print(f' > FAIL {label}: {exc}')
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f' > ERROR {label}: {type(exc).__name__}: {exc}')

    print()
    print(f'tuples={cases}  cases={total}  failed={failed}')
    print('VERDICT:', 'PASS' if failed == 0 else 'FAIL')
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    torch.manual_seed(1)
    random.seed(1)
    sys.exit(run())
