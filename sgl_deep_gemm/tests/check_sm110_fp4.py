"""Run test_fp8_fp4.py's FP4 k-grouped path on SM110, bypassing its guard.

The upstream test builds its option list like this (test_fp8_fp4.py:230-235):

    test_options = [(torch.float8_e4m3fn, QuantConfig(), <fp8 kernel>)]
    if arch_major == 10:
        test_options.append((torch.float4_e2m1fn_x2, QuantConfig((32, 32, True, True)),
                             deep_gemm.k_grouped_fp4_gemm_nt_contiguous))

So on SM110 only FP8 is ever enumerated, even though the patch does route FP4
through arch 11 (`csrc/utils/layout.hpp` accepts kPackedFP4 for major 11, and
`enumerate_k_grouped_contiguous` has a full FP4 branch: K-major only, gran_k of
32 with k_alignment 256/512/768). The generator does not restrict FP4 by arch at
all -- only the test body does -- so FP4 is reachable on Thor while the suite
declines to exercise it. That is the same class of gap as the psum SF section.

This script reproduces the FP4 half of that test with the guard removed, reusing
the test's own generators and assertions so the comparison is meaningful.
"""
import random
import sys

import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import deep_gemm  # noqa: E402
from deep_gemm.testing import (  # noqa: E402
    assert_bitwise_equal, calc_diff, get_arch_major,
)
from utils import (  # noqa: E402
    assert_direct_output_matches_fp32_accumulation, convert_to_fp8,
)

from generators import (  # noqa: E402
    KernelType, QuantConfig, get_ue8m0_usage,
    enumerate_k_grouped_contiguous,
    enumerate_k_grouped_contiguous_test_variants,
    generate_k_grouped_contiguous,
)

# The FP4 option the test only adds for arch_major == 10.
FP4_DTYPE = torch.float4_e2m1fn_x2
FP4_CONFIG = QuantConfig((32, 32, True, True))


def run():
    print(f'arch_major : {get_arch_major()}')
    print(f'deep_gemm  : {deep_gemm.__version__}')
    print(f'kernel     : k_grouped_fp4_gemm_nt_contiguous')
    print()

    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    gemm = deep_gemm.k_grouped_fp4_gemm_nt_contiguous
    dtype_opt = 'FP4'

    total = failed = 0
    for num_groups, m, n, major_a, major_b, real_ks_cpu, _, _, gran_k, k_alignment, \
            use_psum_layout, accumulate, out_dtype in enumerate_k_grouped_contiguous(FP4_DTYPE):
        recipe = (1, 1, gran_k)
        for test_real_ks_cpu in enumerate_k_grouped_contiguous_test_variants(real_ks_cpu):
            # `not is_fp4` is False here, matching the upstream loop bound.
            for compact_sf in (False, ):
                label = (f'(num_groups={num_groups}, m={m}, n={n}, gran_k={gran_k}, '
                         f'k_align={k_alignment}, psum={int(use_psum_layout)}, '
                         f'acc={int(accumulate)}, acc_ks={sum(test_real_ks_cpu)})')
                total += 1
                try:
                    total_k, a, b, c, d, ref_d, grouped_layout, host_ks_cpu = \
                        generate_k_grouped_contiguous(
                            num_groups, m, n, major_a, major_b, test_real_ks_cpu,
                            use_ue8m0=use_ue8m0, gran_k=gran_k,
                            quant_config=FP4_CONFIG,
                            use_psum_layout=use_psum_layout, k_alignment=k_alignment,
                            accumulate=accumulate, out_dtype=out_dtype, compact_sf=compact_sf)

                    initial_d = d.clone()
                    if not accumulate:
                        initial_d.fill_(float('nan'))
                    equivalent_d = initial_d.clone()
                    gemm(a, b, equivalent_d, host_ks_cpu, grouped_layout,
                         equivalent_d if accumulate else None,
                         recipe=recipe, use_psum_layout=use_psum_layout)

                    # FP4 vs converted FP8 must agree bitwise (upstream does this
                    # only for the accumulate case, which is what exists here).
                    if accumulate:
                        fp8_a, fp8_b = convert_to_fp8(a), convert_to_fp8(b)
                        fp8_a = (fp8_a[0].T.contiguous(), fp8_a[1])
                        fp8_b = (fp8_b[0].T.contiguous(), fp8_b[1])
                        equivalent_fp8_d = initial_d.clone()
                        deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
                            fp8_a, fp8_b, equivalent_fp8_d, host_ks_cpu, grouped_layout,
                            equivalent_fp8_d if accumulate else None,
                            recipe=recipe, use_psum_layout=use_psum_layout)
                        assert calc_diff(equivalent_d, equivalent_fp8_d) < 1e-14, \
                            f'FP4/FP8 mismatch {label}'

                    # Bitwise determinism, same 20-iteration stress as upstream.
                    host_ks_options = (host_ks_cpu, None, []) if use_psum_layout else (host_ks_cpu, )
                    for test_host_ks_cpu in host_ks_options:
                        for stress_idx in range(20):
                            d.copy_(initial_d)
                            gemm(a, b, d, test_host_ks_cpu, grouped_layout, c,
                                 recipe=recipe, use_psum_layout=use_psum_layout)
                            assert_bitwise_equal(
                                d, equivalent_d,
                                f'k-grouped self-consistency {stress_idx=} {dtype_opt} {label}')
                        if not accumulate:
                            assert_direct_output_matches_fp32_accumulation(
                                d,
                                lambda output, accumulator: gemm(
                                    a, b, output, test_host_ks_cpu, grouped_layout,
                                    accumulator, recipe=recipe,
                                    use_psum_layout=use_psum_layout),
                                f'{dtype_opt} K-grouped direct output {label}')

                    if accumulate:
                        diff = calc_diff(d, ref_d)
                        assert diff < FP4_CONFIG.max_diff(), \
                            f'{dtype_opt} {label} diff={diff:.5f} >= {FP4_CONFIG.max_diff()}'

                    print(f' > Perf {label}: ok (total_k={total_k})')
                except AssertionError as exc:
                    failed += 1
                    print(f' > FAIL {label}: {exc}')
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(f' > ERROR {label}: {type(exc).__name__}: {str(exc)[:300]}')

    print()
    print(f'cases={total}  failed={failed}')
    print('VERDICT:', 'PASS' if failed == 0 else 'FAIL')
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    sys.exit(run())