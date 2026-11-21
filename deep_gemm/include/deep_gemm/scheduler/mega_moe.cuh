#pragma once

#include <cutlass/arch/barrier.h>
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

// TODO: add MegaMoE namespace
namespace deep_gemm::sched {

// Get minimal L1 warmup waves to ensure no L1 -> L2 deadlock
// constexpr func, no runtime overhead in kernel
CUTLASS_HOST_DEVICE constexpr
int get_num_l1_warmup_waves(
    const int& num_total_m_blocks,
    const int& num_clusters,
    const int& num_l1_n_clusters,
    const int& num_l2_n_clusters,
    const int& max_group_m = 4) {
    // The first L2 wave may touch multiple M blocks (plus up to `max_group_m - 1` swizzled peers in
    // the last expert M-group); all their L1 N tasks must be issued first.
    const int num_first_l2_wave_m_blocks =
        math::constexpr_ceil_div(num_clusters, num_l2_n_clusters) + (max_group_m - 1);
    const int num_l1_warmup_clusters_for_first_l2_wave = math::constexpr_ceil_div(
        num_first_l2_wave_m_blocks * num_l1_n_clusters, num_clusters);

    // When each M block has more L1 tasks than L2 tasks, the interleaved schedule
    // leaves `(num_l1_pair_n_blocks - num_l2_pair_n_blocks)` extra L1 tasks behind
    // per M block. To avoid deadlock, no L2 task for an M block (or its `max_group_m`
    // swizzled chunk) should be scheduled before those L1 tasks have been issued.
    const int num_interleave_cluster_diff_per_m_block =
        num_l1_n_clusters > num_l2_n_clusters ? num_l1_n_clusters - num_l2_n_clusters : 0;
    const int trailing_m_blocks = num_total_m_blocks > 0 ? num_total_m_blocks - 1 : 0;
    const int num_warmup_waves_for_interleave_schedule = math::constexpr_ceil_div(
        max_group_m * num_l1_n_clusters + trailing_m_blocks * num_interleave_cluster_diff_per_m_block,
        num_clusters) + 1;

    // TODO: may delay combine NVLink
    return cute::max(num_l1_warmup_clusters_for_first_l2_wave, num_warmup_waves_for_interleave_schedule);
}

// Host-side ring capacity helper for the MegaMoE schedule.
CUTLASS_HOST_DEVICE constexpr int get_num_max_live_pool_blocks(
    const int& num_total_m_blocks,
    const int& num_sms,
    const int& hidden,
    const int& intermediate_hidden) {
    constexpr int kMegaMoEBlockN = 128;
    constexpr int kNumCTAsPerCluster = 2;
    constexpr int kMaxSwizzleGroupM = 4;

    DG_UNIFIED_ASSERT((intermediate_hidden * 2) % (kNumCTAsPerCluster * kMegaMoEBlockN) == 0);
    DG_UNIFIED_ASSERT(hidden % (kNumCTAsPerCluster * kMegaMoEBlockN) == 0);
    const int num_clusters = num_sms / kNumCTAsPerCluster;
    const int num_l1_n_clusters = intermediate_hidden * 2 / (kNumCTAsPerCluster * kMegaMoEBlockN);
    const int num_l2_n_clusters = hidden / (kNumCTAsPerCluster * kMegaMoEBlockN);
    const int num_l1_clusters = num_total_m_blocks * num_l1_n_clusters;
    const int num_l1_waves = math::constexpr_ceil_div(num_l1_clusters, num_clusters);
    const int num_min_l1_warmup_waves = get_num_l1_warmup_waves(
        num_total_m_blocks, num_clusters, num_l1_n_clusters, num_l2_n_clusters, kMaxSwizzleGroupM);
    const int num_l1_warmup_waves = cute::min(num_min_l1_warmup_waves, num_l1_waves);
    const int num_l1_warmup_clusters = cute::min(num_l1_warmup_waves * num_clusters, num_l1_clusters);
    const int num_live_blocks_after_warmup = math::constexpr_ceil_div(num_l1_warmup_clusters, num_l1_n_clusters);

    // Conservative closed-form bound for the L2/L1 alternating tail. If L1 advances faster in
    // M-block space (B1 < B2), live blocks can grow by about total_m_blocks * (B2 - B1) / B2.
    // Add one global-wave margin plus `kMaxSwizzleGroupM - 1` for intra-expert N-outer swizzling.
    const int frontier_growth = num_l2_n_clusters > num_l1_n_clusters ?
        math::constexpr_ceil_div(
            num_total_m_blocks * (num_l2_n_clusters - num_l1_n_clusters), num_l2_n_clusters) : 0;
    const int wave_margin = math::constexpr_ceil_div(
        num_clusters, cute::min(num_l1_n_clusters, num_l2_n_clusters)) + (kMaxSwizzleGroupM - 1);
    return cute::min(num_total_m_blocks, num_live_blocks_after_warmup + frontier_growth + wave_margin);
}

// Computation phase for the current block
enum class BlockPhase : uint32_t {
    None = 0,
    Linear1 = 1,
    Linear2 = 2,
    SharedLinear1 = 3,
    SharedLinear2 = 4
};

template <bool kHasSharedExperts>
struct alignas(16) TaskInfo {
    BlockPhase block_phase;
    uint32_t local_expert_idx;
    uint16_t m_block_idx;
    uint16_t num_m_blocks;
    uint32_t n_cluster_idx;
    uint32_t pool_block_idx;
    uint32_t valid_m;
    uint32_t shape_n;
    uint32_t shape_k;

    CUTLASS_HOST_DEVICE
    TaskInfo(): TaskInfo(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0, 0) {}

    CUTLASS_HOST_DEVICE
    TaskInfo(const BlockPhase& block_phase,
             const uint32_t& local_expert_idx,
             const uint32_t& m_block_idx,
             const uint32_t& pair_n_block_idx,
             const uint32_t& pool_block_idx,
             const uint32_t& valid_m,
             const uint32_t& shape_n,
             const uint32_t& shape_k,
             const uint32_t& num_m_blocks = 1):
        block_phase(block_phase),
        local_expert_idx(local_expert_idx),
        m_block_idx(static_cast<uint16_t>(m_block_idx)),
        num_m_blocks(static_cast<uint16_t>(num_m_blocks)),
        n_cluster_idx(pair_n_block_idx),
        pool_block_idx(pool_block_idx),
        valid_m(valid_m), shape_n(shape_n), shape_k(shape_k) {}

    CUTLASS_DEVICE
    uint32_t is_valid() const {
        return (block_phase != BlockPhase::None);
    }

    CUTLASS_DEVICE
    uint32_t get_umma_aligned_valid_m() const {
        return math::align(valid_m, 16u);
    }

    CUTLASS_DEVICE uint32_t is_shared() const {
        return kHasSharedExperts ? (block_phase > BlockPhase::Linear2) : false;
    }
};

DG_STATIC_ASSERT(sizeof(sched::TaskInfo<true>) == sizeof(sched::TaskInfo<false>), "Invalid layout");

#if defined(__CUDACC__) or defined(__CLION_IDE__)

// Each finished L1 N block toggles its bit in `l2_full_mask`, so L2 K blocks can start as soon as they are fed
// NOTES: the mask parity per ring generation relies on the ring capacity, see `get_num_max_live_pool_blocks`
template <uint32_t L1_SHAPE_N, uint32_t BLOCK_N, uint32_t BLOCK_K,
          bool kUseMask = (L1_SHAPE_N / BLOCK_N <= 64)>
struct L2KBlockDependency;

template <uint32_t L1_SHAPE_N, uint32_t BLOCK_N, uint32_t BLOCK_K>
struct L2KBlockDependency<L1_SHAPE_N, BLOCK_N, BLOCK_K, true> {
    static constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
    static constexpr uint32_t kNumL1BlockNsPerL2KBlock = BLOCK_K / (BLOCK_N / 2);
    static constexpr uint64_t kFullMask = kNumL1BlockNs == 64 ? ~0ull : (1ull << kNumL1BlockNs) - 1;
    DG_STATIC_ASSERT(kNumL1BlockNs <= 64, "Too many L1 N blocks for the L2 readiness mask");
    DG_STATIC_ASSERT(BLOCK_K % (BLOCK_N / 2) == 0, "Invalid L1/L2 shape relationship");

    const uint64_t* mask_ptr;
    uint64_t expected_mask, pending_mask = kFullMask;

    CUTLASS_DEVICE L2KBlockDependency(const uint64_t* mask_ptr, const uint32_t& generation_idx):
        mask_ptr(mask_ptr), expected_mask(generation_idx & 1 ? 0ull : kFullMask) {}

    CUTLASS_DEVICE static void arrive(const uint64_t* mask_ptr, const uint32_t& n_block_idx) {
        ptx::red_xor_rel(mask_ptr, 1ull << n_block_idx);
    }

    // Re-read the mask only when the K block is not fed yet
    CUTLASS_DEVICE void wait(const uint32_t& k_block_idx) {
        const auto k_block_mask = ((1ull << kNumL1BlockNsPerL2KBlock) - 1) << (k_block_idx * kNumL1BlockNsPerL2KBlock);
        while (pending_mask & k_block_mask)
            pending_mask = ptx::ld_acq_gpu(mask_ptr) ^ expected_mask;
    }
};

// Wider intermediates cannot fit one bit per L1 N block in the readiness word.
// Reuse that word as a cumulative completion counter and wait for the entire L1
// result, as in the original ring protocol. L2 empty counts prevent producers
// from overwriting this generation before all L2 consumers have finished.
template <uint32_t L1_SHAPE_N, uint32_t BLOCK_N, uint32_t BLOCK_K>
struct L2KBlockDependency<L1_SHAPE_N, BLOCK_N, BLOCK_K, false> {
    static constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_K % (BLOCK_N / 2) == 0, "Invalid L1/L2 shape relationship");

    const uint64_t* count_ptr;
    uint64_t expected_count;
    bool ready = false;

    CUTLASS_DEVICE L2KBlockDependency(const uint64_t* count_ptr, const uint32_t& generation_idx):
        count_ptr(count_ptr), expected_count((static_cast<uint64_t>(generation_idx) + 1) * kNumL1BlockNs) {}

    CUTLASS_DEVICE static void arrive(const uint64_t* count_ptr, const uint32_t&) {
        ptx::red_add_rel(count_ptr, uint64_t{1});
    }

    CUTLASS_DEVICE void wait(const uint32_t&) {
        if (not ready) {
            while (ptx::ld_acq_gpu(count_ptr) < expected_count) {}
            ready = true;
        }
    }
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumRingBlocks, uint32_t kNumLocalityDomains,
          uint32_t kNumSharedExperts = 0,
          bool kEnableLPTSwizzle = false,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1Clusters = kNumL1BlockNs / 2,
          uint32_t kNumL2Clusters = kNumL2BlockNs / 2>
struct MegaMoEScheduler {
    static constexpr bool kHasShared = kNumSharedExperts > 0;
    static constexpr uint32_t kSwizzleGroupM = (kEnableLPTSwizzle and not kHasShared) ? 4u : 1u;
    static constexpr uint32_t SHARED_L1_SHAPE_N = L1_SHAPE_N * kNumSharedExperts;
    static constexpr uint32_t SHARED_L1_SHAPE_K = L1_SHAPE_K;
    static constexpr uint32_t SHARED_L2_SHAPE_N = L2_SHAPE_N;
    static constexpr uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts;
    using task_info_t = TaskInfo<kHasShared>;

    DG_STATIC_ASSERT(L1_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(SHARED_L1_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L2_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L1_SHAPE_K % BLOCK_K == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L2_SHAPE_K % BLOCK_K == 0, "Invalid shared shape");

    // NOTES: N block counts must be even so that 2 adjacent CTAs in a cluster
    // always land on the same m_block_idx with n_block_idx differing by 1
    DG_STATIC_ASSERT(kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumRingBlocks > 0, "Invalid ring buffer config");

    // Workspace
    const layout::Workspace& workspace;

    // Scheduler configs
    static constexpr uint32_t kNumScheduleStages = 2;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    uint32_t sched_stage_idx = 0;
    uint32_t sched_phase = 0;
    Barrier* task_info_full_barriers = nullptr;
    Barrier* task_info_empty_barriers = nullptr;
    task_info_t* task_infos = nullptr;

    // Pre-cached per-expert token counts and (when LPT is active) sorted physical expert IDs.
    // Layout: slot (i * 32 + lane_idx) holds its token count and physical local_expert_idx.
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};
    uint32_t stored_expert_id[kNumExpertsPerLane] = {};
    uint32_t num_total_m_blocks = 0;

    // Per-scheduler warmup waves; all CTA-pair schedulers together form one global wave.
    static constexpr uint32_t kNumSchedL1WavesDone = 0xffffffffu;
    uint32_t num_sched_l1_waves = 0;

    // Locality domain affinity. When a queue is drained, we try work stealing to ensure load balancing.
    // But when there are few tasks, we disable work stealing to avoid cross-die latency.
    static constexpr uint32_t kNumMinStealWaves = 2;
    DG_STATIC_ASSERT(kNumLocalityDomains == 1 or kNumLocalityDomains == kNumDeviceLocalityDomains, "Invalid locality domain count");
    DG_STATIC_ASSERT(kNumL1Clusters % kNumLocalityDomains == 0 and kNumL2Clusters % kNumLocalityDomains == 0, "Each domain must take whole clusters");
    uint32_t sm_locality_domain_idx = 0;

    CUTLASS_DEVICE explicit MegaMoEScheduler(const layout::Workspace& workspace):
        workspace(workspace) {}

    CUTLASS_DEVICE MegaMoEScheduler(const layout::Workspace& workspace,
                                    Barrier* task_info_full_barriers,
                                    Barrier* task_info_empty_barriers,
                                    task_info_t* task_infos):
        workspace(workspace),
        task_info_full_barriers(task_info_full_barriers),
        task_info_empty_barriers(task_info_empty_barriers),
        task_infos(task_infos) {}

    CUTLASS_DEVICE void advance_sched_pipeline() {
        DG_STATIC_ASSERT(kNumScheduleStages == 2, "Invalid stages");
        sched_stage_idx ^= 1;
        sched_phase ^= sched_stage_idx == 0;
    }

    CUTLASS_DEVICE bool get_next_task(task_info_t& task_info) {
        task_info_full_barriers[sched_stage_idx].wait(sched_phase);
        // Acquire loads, so the slot release cannot be performed before the reads complete
        DG_STATIC_ASSERT(sizeof(task_info_t) == 2 * sizeof(uint4), "Invalid task info size");
        const auto slot = reinterpret_cast<const uint4*>(task_infos + sched_stage_idx);
        auto dst = reinterpret_cast<uint4*>(&task_info);
        dst[0] = ptx::ld_acquire_cta_shared(slot);
        dst[1] = ptx::ld_acquire_cta_shared(slot + 1);
        advance_sched_pipeline();
        return task_info.is_valid();
    }

    CUTLASS_DEVICE void release_task_info() const {
        task_info_empty_barriers[sched_stage_idx ^ 1].arrive(0u);
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& slot_idx) const {
        uint32_t valid_value = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (slot_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, slot_idx % 32);
    }

    CUTLASS_DEVICE uint32_t get_expert_id(const uint32_t& slot_idx) const {
        if constexpr (not kEnableLPTSwizzle)
            return slot_idx;
        uint32_t valid_value = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (slot_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_expert_id[i] : valid_value;
        }
        return ptx::exchange(valid_value, slot_idx % 32);
    }

    // Get pool block offset for a given slot index from the per-lane token count array.
    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& slot_idx) const {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < slot_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    CUTLASS_DEVICE uint32_t get_num_total_pool_blocks() const {
        return get_pool_block_offset(kNumExpertsPerRank);
    }

    CUTLASS_DEVICE void fetch_expert_recv_count() {
        const uint32_t lane_idx = ptx::get_lane_idx();
        uint32_t raw_tokens[kNumExpertsPerLane] = {};
        uint32_t raw_keys[kNumExpertsPerLane] = {};
        uint32_t lane_max_nb = 0u;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + lane_idx;
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            raw_tokens[i] = static_cast<uint32_t>(value);
            stored_num_tokens_per_expert[i] = raw_tokens[i];
            stored_expert_id[i] = expert_idx;
            if constexpr (kEnableLPTSwizzle and not kHasShared and kNumExpertsPerRank <= 64u and kNumExpertsPerRank > 1u) {
                const uint32_t nb = (expert_idx < kNumExpertsPerRank) ? math::ceil_div(raw_tokens[i], BLOCK_M) : 0u;
                lane_max_nb = cute::max(lane_max_nb, nb);
                // Unique descending key: (num_m_blocks << 20) | (tokens << 6) | (63 - expert_idx)
                raw_keys[i] = (expert_idx < kNumExpertsPerRank)
                    ? ((nb << 20u) | (cute::min(raw_tokens[i], 16383u) << 6u) | (63u - expert_idx))
                    : 0u;
            }
        }
        __syncwarp();

        // Dynamic LPT expert ordering: sort slots descending by `num_m_blocks` in registers only when
        // multi-block routing skew is present (`max_m_blocks >= 3`), incurring zero overhead on decode
        // or balanced loads.
        if constexpr (kEnableLPTSwizzle and not kHasShared and kNumExpertsPerRank <= 64u and kNumExpertsPerRank > 1u) {
            const uint32_t max_m_blocks = __reduce_max_sync(0xffffffffu, lane_max_nb);
            if (max_m_blocks >= 3u) {
                uint32_t my_rank[kNumExpertsPerLane] = {};
                #pragma unroll
                for (uint32_t s = 0; s < kNumExpertsPerRank; ++ s) {
                    const uint32_t other_key = __shfl_sync(0xffffffffu, raw_keys[s >> 5u], s & 31u);
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
                        const uint32_t expert_idx = i * 32u + lane_idx;
                        if (expert_idx < kNumExpertsPerRank and other_key > raw_keys[i])
                            ++ my_rank[i];
                    }
                }
                #pragma unroll
                for (uint32_t slot = 0; slot < kNumExpertsPerRank; ++ slot) {
                    uint32_t found_e = 0u, found_t = 0u;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
                        const uint32_t expert_idx = i * 32u + lane_idx;
                        const uint32_t mask = __ballot_sync(
                            0xffffffffu, expert_idx < kNumExpertsPerRank and my_rank[i] == slot);
                        if (mask) {
                            const uint32_t owner_lane = static_cast<uint32_t>(__ffs(mask) - 1);
                            found_e = ptx::exchange(expert_idx, owner_lane);
                            found_t = ptx::exchange(raw_tokens[i], owner_lane);
                        }
                    }
                    if (lane_idx == (slot & 31u)) {
                        stored_expert_id[slot >> 5u] = found_e;
                        stored_num_tokens_per_expert[slot >> 5u] = found_t;
                    }
                }
                __syncwarp();
            }
        }

        num_total_m_blocks = get_num_total_pool_blocks();
        const uint32_t num_total_l1_tasks = num_total_m_blocks * kNumL1Clusters;
        const uint32_t num_total_l1_waves = math::ceil_div(num_total_l1_tasks, kNumSMs / 2);
        const uint32_t min_l1_warmup_waves = get_num_l1_warmup_waves(
            num_total_m_blocks, kNumSMs / 2, kNumL1Clusters, kNumL2Clusters, static_cast<int>(kSwizzleGroupM));

        // Issue every L1 task before the first L2 task when little L1 work is left after the warmup waves:
        // at most one extra L1 wave, or `kMaxDecodeL1FirstExtraWaves` with decode tiles (`BLOCK_M <= 64`).
        // The last M blocks then start their L1 earlier, and their L1 -> L2 chain sets the tail. The extra
        // waves also hold back every L2 task and its combine writes, which costs more than it saves past the
        // cap (e.g. many local experts with a few tokens each). Larger loads keep the interleaved order.
        // With a ring slot for every pool block, no L1 task waits for an L2 task to free a slot.
        constexpr uint32_t kMaxDecodeL1FirstExtraWaves = 8;
        constexpr uint32_t kMaxL1FirstExtraWaves = BLOCK_M <= 64 ? kMaxDecodeL1FirstExtraWaves : 1;
        const bool is_l1_first = num_total_m_blocks <= kNumRingBlocks and
            num_total_l1_waves <= min_l1_warmup_waves + kMaxL1FirstExtraWaves;
        num_sched_l1_waves = is_l1_first ? num_total_l1_waves : cute::min(min_l1_warmup_waves, num_total_l1_waves);
    }

    template <uint32_t kNumClusters>
    CUTLASS_DEVICE task_info_t create_task(const BlockPhase& block_phase,
                                           const uint32_t& linear_m_block,
                                           const uint32_t& local_n_cluster_idx,
                                           const uint32_t& domain_idx,
                                           const uint32_t& shape_n,
                                           const uint32_t& shape_k,
                                           uint32_t* out_chunk_end_block = nullptr) const {
        constexpr uint32_t kNumLocalClusters = kNumClusters / kNumLocalityDomains;
        const uint32_t lane_idx = ptx::get_lane_idx();
        const uint32_t base_n_cluster = domain_idx * kNumLocalClusters + local_n_cluster_idx;

        task_info_t result(block_phase, 0, 0, base_n_cluster, linear_m_block, 0, shape_n, shape_k, 1);
        uint32_t chunk_end_block = linear_m_block + 1u;
        uint32_t block_offset = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            // Reduce whether task falls in this (optionally LPT-sorted) expert slot
            const uint32_t slot_idx = i * 32 + lane_idx;
            const uint32_t num_tokens = stored_num_tokens_per_expert[i];
            const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
            const uint32_t inclusive_num_m_blocks = math::warp_inclusive_sum(num_m_blocks, lane_idx);
            const uint32_t lane_pool_block_offset = block_offset + inclusive_num_m_blocks - num_m_blocks;
            const bool is_owner = slot_idx < kNumExpertsPerRank and
                linear_m_block >= lane_pool_block_offset and linear_m_block < lane_pool_block_offset + num_m_blocks;
            const uint32_t owner_mask = __ballot_sync(0xffffffff, is_owner);

            // Exchange expert info (with locality-domain-preserving N-outer swizzle across up to `kSwizzleGroupM` M-blocks)
            if (owner_mask) {
                const uint32_t owner_lane_idx = static_cast<uint32_t>(__ffs(owner_mask) - 1);
                uint32_t owner_m_block_idx = linear_m_block - lane_pool_block_offset;
                uint32_t owner_n_cluster_idx = base_n_cluster;
                uint32_t owner_pool_block_idx = linear_m_block;
                uint32_t owner_chunk_end = linear_m_block + 1u;
                if constexpr (kSwizzleGroupM > 1u) {
                    constexpr uint32_t kGroupM = kSwizzleGroupM;
                    const uint32_t chunk_m_start = (owner_m_block_idx / kGroupM) * kGroupM;
                    const uint32_t chunk_m_size = is_owner ? cute::min(kGroupM, num_m_blocks - chunk_m_start) : 1u;
                    const uint32_t m_in_chunk = owner_m_block_idx - chunk_m_start;
                    const uint32_t task_in_chunk = m_in_chunk * kNumLocalClusters + local_n_cluster_idx;
                    owner_n_cluster_idx = domain_idx * kNumLocalClusters + (task_in_chunk / chunk_m_size);
                    owner_m_block_idx = chunk_m_start + (task_in_chunk % chunk_m_size);
                    owner_pool_block_idx = lane_pool_block_offset + owner_m_block_idx;
                    owner_chunk_end = lane_pool_block_offset + chunk_m_start + chunk_m_size;
                }
                const uint32_t owner_valid_m = is_owner ? cute::min(num_tokens - owner_m_block_idx * BLOCK_M, BLOCK_M) : 0u;
                const uint32_t phys_expert = kEnableLPTSwizzle ? stored_expert_id[i] : slot_idx;
                result.local_expert_idx = ptx::exchange(phys_expert, owner_lane_idx);
                result.m_block_idx = static_cast<uint16_t>(ptx::exchange(owner_m_block_idx, owner_lane_idx));
                result.num_m_blocks = static_cast<uint16_t>(ptx::exchange(num_m_blocks, owner_lane_idx));
                result.n_cluster_idx = ptx::exchange(owner_n_cluster_idx, owner_lane_idx);
                result.pool_block_idx = ptx::exchange(owner_pool_block_idx, owner_lane_idx);
                result.valid_m = ptx::exchange(owner_valid_m, owner_lane_idx);
                chunk_end_block = ptx::exchange(owner_chunk_end, owner_lane_idx);
            }
            block_offset += ptx::exchange(inclusive_num_m_blocks, 31);
        }
        if (out_chunk_end_block != nullptr)
            *out_chunk_end_block = chunk_end_block;
        return result;
    }

    static CUTLASS_DEVICE uint32_t get_next_local_task_idx(const uint32_t* task_count_ptr, const uint32_t& locality_domain_idx) {
        uint32_t result = 0;
        if (cute::elect_one_sync())
            result = ptx::atomic_add(task_count_ptr + locality_domain_idx, 1u);
        return ptx::exchange(result, 0);
    }

    template <uint32_t kNumClusters>
    static CUTLASS_DEVICE bool get_next_task_idx(const uint32_t* task_count_ptr, const uint32_t& locality_domain_idx, const uint32_t& num_m_blocks,
                                                 uint32_t& m_block_idx, uint32_t& local_n_cluster_idx, uint32_t& domain_idx) {
        constexpr uint32_t kNumLocalClusters = kNumClusters / kNumLocalityDomains;
        const uint32_t task_idx = get_next_local_task_idx(task_count_ptr, locality_domain_idx);
        if (task_idx >= num_m_blocks * kNumLocalClusters)
            return false;
        m_block_idx = task_idx / kNumLocalClusters;
        local_n_cluster_idx = task_idx % kNumLocalClusters;
        domain_idx = locality_domain_idx;
        return true;
    }

    template <uint32_t kNumClusters>
    CUTLASS_DEVICE bool get_next_task_idx(const uint32_t* task_count_ptr, const uint32_t& num_m_blocks,
                                          uint32_t& m_block_idx, uint32_t& local_n_cluster_idx, uint32_t& domain_idx) const {
        if (get_next_task_idx<kNumClusters>(task_count_ptr, sm_locality_domain_idx, num_m_blocks, m_block_idx, local_n_cluster_idx, domain_idx))
            return true;
        return kNumLocalityDomains > 1 and num_m_blocks * kNumClusters >= kNumMinStealWaves * (kNumSMs / 2) and
               get_next_task_idx<kNumClusters>(task_count_ptr, sm_locality_domain_idx ^ 1, num_m_blocks, m_block_idx, local_n_cluster_idx, domain_idx);
    }

    CUTLASS_DEVICE task_info_t get_next_task() {
        uint32_t m_block_idx, local_n_cluster_idx, domain_idx;
        while (true) {
            if (num_sched_l1_waves != kNumSchedL1WavesDone and num_sched_l1_waves) {
                // One local L1 task per scheduler; globally this is one CTA-pair wave.
                -- num_sched_l1_waves;

                // No more L1 tasks
                if (not get_next_task_idx<kNumL1Clusters>(workspace.get_l1_task_count_ptr(), num_total_m_blocks, m_block_idx, local_n_cluster_idx, domain_idx)) {
                    num_sched_l1_waves = kNumSchedL1WavesDone;
                    continue;
                }

                // Create task
                return create_task<kNumL1Clusters>(BlockPhase::Linear1, m_block_idx, local_n_cluster_idx, domain_idx, L1_SHAPE_N, L1_SHAPE_K);
            } else {
                if (not get_next_task_idx<kNumL2Clusters>(workspace.get_l2_task_count_ptr(), num_total_m_blocks, m_block_idx, local_n_cluster_idx, domain_idx))
                    break;

                // The next task should be L1
                if (num_sched_l1_waves != kNumSchedL1WavesDone)
                    num_sched_l1_waves = 1;

                // Create task
                uint32_t chunk_end_block = 0;
                auto task_info = create_task<kNumL2Clusters>(
                    BlockPhase::Linear2, m_block_idx, local_n_cluster_idx, domain_idx, L2_SHAPE_N, L2_SHAPE_K, &chunk_end_block);

                // Wait until all required L1 tasks are fetched from all scheduling queues
                const auto num_required_l1_tasks = chunk_end_block * (kNumL1Clusters / kNumLocalityDomains);
                #pragma unroll
                for (uint32_t i = 0; i < kNumLocalityDomains; ++ i)
                    while (ptx::ld_volatile(workspace.get_l1_task_count_ptr() + i) < num_required_l1_tasks) {}
                return task_info;
            }
        }
        return task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0);
    }

    CUTLASS_DEVICE void publish_task(const task_info_t& task_info, const uint32_t& lane_idx) {
        if (lane_idx < 2) {
            task_info_full_barriers[sched_stage_idx].arrive_and_expect_tx(sizeof(task_info_t), lane_idx);
            ptx::st_async_cluster(
                task_infos + sched_stage_idx, task_info,
                lane_idx, task_info_full_barriers[sched_stage_idx]
            );
        }
        __syncwarp();
        advance_sched_pipeline();
    }

    template <BlockPhase kBlockPhase, uint32_t kShapeN, uint32_t kShapeK>
    CUTLASS_DEVICE void shared_mainloop(const uint32_t& num_tokens, const uint32_t& lane_idx, const uint32_t* task_count_ptr) {
        constexpr uint32_t kNumNClusters = kShapeN / BLOCK_N / 2;
        constexpr uint32_t kNumLocalNClusters = kNumNClusters / kNumLocalityDomains;
        DG_STATIC_ASSERT(kNumNClusters % kNumLocalityDomains == 0, "Each domain must take whole clusters");
        const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
        uint32_t m_block_idx, local_n_cluster_idx, domain_idx;
        while (true) {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);

            // Use dynamic scheduling to reduce tailing across shared L1/L2 tile shapes.
            if (not get_next_task_idx<kNumNClusters>(task_count_ptr, num_m_blocks, m_block_idx, local_n_cluster_idx, domain_idx))
                break;
            const uint32_t n_cluster_idx = domain_idx * kNumLocalNClusters + local_n_cluster_idx;
            const uint32_t valid_m = cute::min(num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
            publish_task(task_info_t(kBlockPhase, 0, m_block_idx, n_cluster_idx, m_block_idx, valid_m, kShapeN, kShapeK, num_m_blocks), lane_idx);
        }
    }

    CUTLASS_DEVICE void mainloop(const uint32_t& num_tokens, const uint8_t* sm_locality_domains) {
        const auto lane_idx = ptx::get_lane_idx();
        sm_locality_domain_idx = kNumLocalityDomains == 1 ? 0u : sm_locality_domains[ptx::get_sm_idx()];

        // Shared L2 tasks go before the routed tasks if one wave fits the dispatch gap and the tiles are small:
        // decode tiles (`BLOCK_M <= 64`), or 128-row tiles with fewer than 100 CTA pairs (B300 / GB300 have 74-76;
        // the tail was ~1% slower there, and 1-2% faster with Vera Rubin's 106). Otherwise they go at the tail,
        // where they fill the CTA pairs that idle during the last routed M blocks' L1 -> L2 chain instead of
        // waiting for the shared L1 ahead of the routed L1 tasks.
        constexpr uint32_t kNumSharedL2Clusters = SHARED_L2_SHAPE_N / BLOCK_N / 2;
        constexpr bool kIsSharedL2EarlyTile = BLOCK_M <= 64 or (BLOCK_M < 192 and kNumSMs / 2 < 100);
        const bool is_shared_l2_early = kIsSharedL2EarlyTile and
            math::ceil_div(num_tokens, BLOCK_M) * kNumSharedL2Clusters <= kNumSMs / 2;
        if constexpr (kHasShared) {
            // Shared expert L1 tasks do not depend on dispatch.
            shared_mainloop<BlockPhase::SharedLinear1, SHARED_L1_SHAPE_N, SHARED_L1_SHAPE_K>(
                num_tokens, lane_idx, workspace.get_shared_l1_task_count_ptr());
            if (is_shared_l2_early)
                shared_mainloop<BlockPhase::SharedLinear2, SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K>(
                    num_tokens, lane_idx, workspace.get_shared_l2_task_count_ptr());
        }

        // Wait dispatch's results
        fetch_expert_recv_count();

        // Generate routed tasks. Keep the original wait -> claim -> publish ordering:
        // `get_next_task()` advances global task counters and must not run before the
        // schedule slot is released by consumers.
        task_info_t task_info;
        do {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
            task_info = get_next_task();
            if (task_info.is_valid()) publish_task(task_info, lane_idx);
        } while (task_info.is_valid());

        if constexpr (kHasShared) {
            if (not is_shared_l2_early)
                shared_mainloop<BlockPhase::SharedLinear2, SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K>(
                    num_tokens, lane_idx, workspace.get_shared_l2_task_count_ptr());
        }

        // Sentinel.
        task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
        publish_task(task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0), lane_idx);
    }
};

#endif

} // namespace deep_gemm::sched
