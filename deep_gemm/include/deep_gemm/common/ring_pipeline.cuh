#pragma once

#include <cstdint>

#include <cutlass/cutlass.h>

#include <cute/container/tuple.hpp>

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm {

// `kParityPhase`: keep `phase` as the parity of completed passes, so power-of-two rings may also advance by more than
// one pass per call
template <uint32_t kNumStages, bool kParityPhase = false>
struct RingPipeline {
    DG_STATIC_ASSERT(kNumStages > 0, "Ring pipeline must contain at least one stage");

    uint32_t stage_idx = 0;
    uint32_t phase = 0;

    // Callers advance by at most one full ring, so the phase toggles at most once.
    CUTLASS_DEVICE cute::tuple<uint32_t, uint32_t> advance(const uint32_t step = 1) {
        const uint32_t current_stage_idx = stage_idx;
        const uint32_t current_phase = phase;
        const uint32_t next_stage_idx = stage_idx + step;
        // Modulo and division by a power of two lower to bit operations.
        if constexpr ((kNumStages & (kNumStages - 1)) == 0) {
            stage_idx = next_stage_idx % kNumStages;
            phase ^= kParityPhase ? (next_stage_idx / kNumStages) & 1u : next_stage_idx / kNumStages;
        } else {
            stage_idx = next_stage_idx;
            if (stage_idx >= kNumStages) {
                stage_idx -= kNumStages;
                phase ^= 1u;
            }
        }
        return cute::tuple<uint32_t, uint32_t>(current_stage_idx, current_phase);
    }
};

} // namespace deep_gemm
