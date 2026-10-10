#pragma once

#include <deep_gemm/comm/barrier.cuh>

namespace deep_gemm::comm {

// Split barrier used to defer the cleanup wait into the next layer. The first
// phase-0 signal must be initialized to kNumRanks before the first launch.
template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads,
          uint32_t kGridSyncIndex, uint32_t kTag,
          typename sync_scope_t, typename workspace_t = layout::Workspace>
CUTLASS_DEVICE void nvlink_barrier_arrive(
    const workspace_t& workspace,
    const layout::SymBuffer<kNumRanks>& sym_buffer,
    const uint32_t& sm_idx, const uint32_t& thread_idx,
    const sync_scope_t& sync_scope,
    const bool& sync_prologue = true) {
    DG_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");

    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex>(
            workspace, sm_idx, thread_idx, sync_scope);

    if (sm_idx == 0) {
        const auto status = (*workspace.get_nvl_barrier_counter_ptr()) & 3;
        const auto signal_phase = status & 1;
        const auto signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);
        if (thread_idx < kNumRanks)
            ptx::red_add_rel_sys(
                sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
    }
}

template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads,
          uint32_t kGridSyncIndex, uint32_t kTag,
          typename sync_scope_t, typename workspace_t = layout::Workspace>
CUTLASS_DEVICE void nvlink_barrier_wait(
    const workspace_t& workspace,
    const layout::SymBuffer<kNumRanks>& sym_buffer,
    const uint32_t& sm_idx, const uint32_t& thread_idx,
    const sync_scope_t& sync_scope,
    const bool& sync_epilogue = true) {
    if (sm_idx == 0 and thread_idx == 0) {
        auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
        const auto status = (*counter_ptr) & 3;
        const auto signal_phase = status & 1;
        const auto signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

        constexpr int64_t kNumTimeoutCycles = 180ll * 2000000000ll;
        ptx::red_add(counter_ptr, 1);
        const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
        const auto start_clock = clock64();
        while (ptx::ld_acq_sys(signal_ptr) != target) {
            if (clock64() - start_clock >= kNumTimeoutCycles) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1000) && \
        !(defined(DG_NVLINK_BARRIER_VERBOSE_TIMEOUT) && DG_NVLINK_BARRIER_VERBOSE_TIMEOUT)
                DG_TRAP_ONLY_DEVICE_ASSERT(false);
#else
                printf("DeepGEMM NVLink barrier-wait timeout (180s): rank=%d, signal=%d, target=%d, phase=%d, sign=%d, tag=%d\n",
                       sym_buffer.rank_idx, ptx::ld_acq_sys(signal_ptr), target,
                       signal_phase, signal_sign, kTag);
                DG_DEVICE_ASSERT(false and "NVLink barrier timeout");
#endif
            }
        }
    }

    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex>(
            workspace, sm_idx, thread_idx, sync_scope);
}

} // namespace deep_gemm::comm
