#pragma once

#include <cstdint>
#include <iterator>

#include "arch.hpp"

// ---------------------------------------------------------------------------
// SM110 (Jetson Thor) support policy.
//
// SM110 is *always* reported as its real arch major (`11`); nothing here maps
// it onto SM100. What this file decides is, per kernel family, whether SM110
// takes the shared SM100 implementation and with which parameters:
//
//   * `uses_sm100_tuning(major)` -- the *original*, Thor-verified one-to-one
//     port: every dispatch predicate that the first SM110 adaptation widened
//     to `major == 10 or major == 11` consults this, so the two lists cannot
//     drift apart (see `deep_gemm.arch.sm110_uses_sm100_tuning`).
//   * `bf16_wide_split_enabled(major)` -- a *later, upstream-only* optimization
//     (the 384-token BF16 MQA split). It is a separate switch precisely because
//     "SM110 has the SM100 kernel family" does not imply "this newer variant
//     also routes correctly on SM110" -- an upstream refactor that widened one
//     predicate but not another left sm100_wide_split gated on `== 10` alone
//     while the dispatch that consumes it admitted `11`, asserting on Thor.
//   * `verified_feature(...)` -- per-feature validation state, so a behavior
//     can be enabled for one arch and kept off (falling back to the verified
//     path) for another without touching the shared implementation.
//
// Adding a SM110-enabled feature without a passing Thor run is exactly the
// failure mode this file exists to prevent, hence the explicit validated bit.
// ---------------------------------------------------------------------------
namespace deep_gemm::arch {

// How a feature behaves on one arch.
enum class FeatureSupport : int {
    Unsupported = 0,  // never dispatch here
    Validated = 1,    // enabled, covered by a passing test on this arch
    Unvalidated = 2,  // implementation exists but no test has run on this arch
};

struct FeatureState {
    FeatureSupport sm100;
    FeatureSupport sm110;
};

// Arch majors that share the SM100 tcgen05 kernel family and run it with the
// same tuning. SM110 is listed by its *real* major.
constexpr bool uses_sm100_tuning(const int64_t arch_major) {
    return arch_major == 10 or arch_major == 11;
}

// Features that were introduced by the SM110 fork itself and verified on Thor,
// or that upstream touched after SM110 routing existed. Keep entries aligned
// with what the tests exercise on each arch; `deep_gemm.arch.sm110_status()`
// reports the same table to Python.
inline constexpr struct {
    const char* name;
    FeatureState state;
} kFeatureStates[] = {
    // Shared SM100 GEMM / attention / layout / mega-gate dispatch.
    {"sm100_family_dispatch", {FeatureSupport::Validated, FeatureSupport::Validated}},
    // Mega-Gate: upstream changes its non-finite score checking. Thor never ran it.
    // NOTE(Thor): ptxas rejects `redux.sync.max.f32` for sm_110f
    // (deep_gemm/include/deep_gemm/ptx/utils.cuh:51 -> epilogue/sm100_mega_gate.cuh:92),
    // so this kernel cannot JIT on SM110 at all -- not "unvalidated", but unusable.
    {"bf16_mega_gate", {FeatureSupport::Validated, FeatureSupport::Unvalidated}},
    // Upstream's 384-token / three-warpgroup BF16 MQA split.
    // NOTE(Thor): validated on SM110 -- the MXFP8/MXFP4 + BF16-weights + BF16-logits
    // path runs correctly (rel. err ~9.8e-2, expected for BF16 weights) and is
    // 1.5x-2.9x faster than the 256-token path. Keeping it off is not a safe
    // fallback: `bf16_mqa_kernel` asserts block_kv == 384, so disabling the split
    // turns every such request into a hard failure instead of a slower kernel.
    {"bf16_mqa_wide_split", {FeatureSupport::Validated, FeatureSupport::Validated}},
    // Contiguous-K layout BLOCK_M search tuned for SM100.
    {"contiguous_mk_alignment_sm100", {FeatureSupport::Validated, FeatureSupport::Unvalidated}},
    // Mega-MHC: the SM100 fused kernel is dispatched on the shared family, but no Thor
    // test has run it. Refuse it until one does, instead of letting an opt-in caller
    // discover a JIT or numerical failure at request time.
    {"bf16_mega_mhc", {FeatureSupport::Validated, FeatureSupport::Unvalidated}},
    // The `BF16StochasticRounding` epilogue lowers to the PTX `cvt.rs.bf16x2.f32`
    // instruction, which ptxas accepts only for `sm_100a`. Measured on CUDA 13.0:
    // `sm_100a` builds, while `sm_100f`, `sm_110a`, `sm_110f`, `sm_120a` and `sm_90a`
    // all fail with "Feature '.rs' not supported". This is a hardware instruction
    // that does not exist on SM110 -- not a variant that merely lacks testing -- so
    // it is Unsupported rather than Unvalidated, and callers get a clear refusal
    // instead of a ptxas failure buried in a JIT traceback.
    {"bf16_stochastic_rounding", {FeatureSupport::Validated, FeatureSupport::Unsupported}},
};

inline constexpr size_t kNumFeatureStates = std::size(kFeatureStates);

inline constexpr int major_to_index(const int64_t arch_major) {
    return arch_major == 10 ? 0 : (arch_major == 11 ? 1 : -1);
}

constexpr bool verified_feature(const char* const name, const int64_t arch_major) {
    const auto index = major_to_index(arch_major);
    if (index < 0)
        return false;
    for (const auto& feature : kFeatureStates) {
        // Same comparison as std::string_view without pulling in <string_view>.
        const char* a = feature.name;
        const char* b = name;
        while (*a and *a == *b) { ++a; ++b; }
        if (*a == *b) {
            const auto support = index == 0 ? feature.state.sm100 : feature.state.sm110;
            return support == FeatureSupport::Validated;
        }
    }
    return false;
}

// Raw state of a feature on an arch, without collapsing it to a yes/no answer.
// Lets a caller distinguish "not tested yet" from "cannot exist here at all",
// which are different messages to give a user.
constexpr FeatureSupport feature_state(const char* const name, const int64_t arch_major) {
    const auto index = major_to_index(arch_major);
    if (index < 0)
        return FeatureSupport::Unsupported;
    for (const auto& feature : kFeatureStates) {
        const char* a = feature.name;
        const char* b = name;
        while (*a and *a == *b) { ++a; ++b; }
        if (*a == *b)
            return index == 0 ? feature.state.sm100 : feature.state.sm110;
    }
    return FeatureSupport::Unsupported;
}

// The 384-token BF16 MQA split is taken only where it has actually been run.
constexpr bool bf16_wide_split_enabled(const int64_t arch_major) {
    return uses_sm100_tuning(arch_major) and verified_feature("bf16_mqa_wide_split", arch_major);
}

}  // namespace deep_gemm::arch
