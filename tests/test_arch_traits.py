"""SM110 (Jetson Thor) is reported as its real arch major (11), never mapped onto SM100.

The first SM110 port widened every SM100 dispatch predicate to `major == 10 or
major == 11`. Later upstream work added variants gated on `== 10` alone while the
dispatch that consumed them still admitted `11`; merging that without care left
`fp8_fp4_mqa_logits` asking for a 384-token split that never got selected on Thor,
so the host asserted. `csrc/utils/arch_traits.hpp` is the single place that now
answers "does SM110 take this?", and these checks keep the two halves of that
answer -- family sharing and per-feature validation -- from silently converging.
"""

import deep_gemm
from deep_gemm.testing import get_arch_major


def test_supported_arch_majors_includes_sm110_as_11() -> None:
    majors = deep_gemm.supported_arch_majors()
    assert 11 in majors, majors
    for major in majors:
        assert major in (9, 10, 11, 12), f'unexpected arch major {major}'


def test_arch_feature_status_shape() -> None:
    status = deep_gemm.arch_feature_status()
    assert status, 'no feature states reported'
    states = {'unsupported', 'validated', 'unvalidated'}
    for name, per_arch in status.items():
        assert set(per_arch) == {'sm100', 'sm110'}, (name, per_arch)
        for arch, state in per_arch.items():
            assert state in states, (name, arch, state)
    # Shared SM100 kernel family: both targets run it.
    assert status['sm100_family_dispatch'] == {'sm100': 'validated', 'sm110': 'validated'}


def test_sm110_only_claims_reuse_not_validation() -> None:
    # Reuse of the SM100 family holds on SM110; newer variants stay unvalidated there
    # until a passing Thor run flips their entry by hand.
    status = deep_gemm.arch_feature_status()
    assert deep_gemm.supports_feature('sm110', 'sm100_family_dispatch')
    assert deep_gemm.supports_feature(11, 'sm100_family_dispatch')
    # Mega-Gate stays off on SM110: its kernel emits `redux.sync.max.f32`, which ptxas
    # rejects for sm_110f, so there is no working kernel to enable there.
    assert not deep_gemm.supports_feature('sm110', 'bf16_mega_gate')
    # Mega-MHC is the same class of untested SM100 fusion. The host must refuse it
    # on SM110 even if a caller opts in; SGLang not calling it today is not a gate.
    assert not deep_gemm.supports_feature('sm110', 'bf16_mega_mhc')
    # ... and an unknown feature is never assumed present.
    assert not deep_gemm.supports_feature('sm110', 'no_such_feature')
    # SM90 has no entry: not a member of the family.
    assert not deep_gemm.supports_feature('sm90', 'sm100_family_dispatch')


def test_sm110_keeps_the_384_wide_split_enabled() -> None:
    # Validated on Thor: runs correctly (rel. err ~9.8e-2, expected for BF16 weights)
    # and is 1.5x-2.9x faster than the 256-token path. Disabling it is NOT a safe
    # fallback -- `bf16_mqa_kernel` asserts block_kv == kBF16MQALogitsSplitKV, so any
    # MXFP* + BF16-weights + BF16-logits request would fail outright instead of
    # running slower. This test exists so that "turn the split off" cannot be used as
    # a quiet way to dodge the path.
    assert deep_gemm.supports_feature('sm110', 'bf16_mqa_wide_split')
    assert deep_gemm.supports_feature('sm100', 'bf16_mqa_wide_split')

def test_arch_normalisation_keeps_sm110_distinct() -> None:
    assert deep_gemm.supports_arch('sm110')
    assert deep_gemm.supports_arch('sm_110')
    assert deep_gemm.supports_arch(110)
    assert deep_gemm.supports_arch(11)


if __name__ == '__main__':
    test_supported_arch_majors_includes_sm110_as_11()
    test_arch_feature_status_shape()
    test_sm110_only_claims_reuse_not_validation()
    test_sm110_keeps_the_384_wide_split_enabled()
    test_arch_normalisation_keeps_sm110_distinct()
    print(f'SM110 arch contract: OK (device arch major {get_arch_major()})')
