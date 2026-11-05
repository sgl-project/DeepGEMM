#pragma once

#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>

namespace deep_gemm::layout::mega_mhc {

#if defined(__CUDACC__)
// Exact BF16x2 widening: one shift and one mask, no PRMT
CUTLASS_DEVICE float2 bf16x2_to_float2(const nv_bfloat162& value) {
    const uint32_t bits = *reinterpret_cast<const uint32_t*>(&value);
    return {__uint_as_float(bits << 16), __uint_as_float(bits & 0xffff0000u)};
}
#endif

// Model geometry
static constexpr uint32_t kNumRoutes = 4;
static constexpr uint32_t kNumHCOutputs = kNumRoutes * (kNumRoutes + 2);

// FP8 scale-factor layout
static constexpr uint32_t kHiddenPerSF = 32;
static constexpr uint32_t kHiddenPerSFWord = 128;

// Kernel tile and pipeline geometry
static constexpr uint32_t BLOCK_M = 64;
static constexpr uint32_t BLOCK_N = kNumHCOutputs;
static constexpr uint32_t BLOCK_K = 64;
static constexpr uint32_t kSwizzleMode = 128;
static constexpr uint32_t kFnAtomK = kSwizzleMode / sizeof(float);
static constexpr uint32_t kSwizzleAlignment = 8 * kSwizzleMode;
static constexpr uint32_t kNumIOStages = 3;
static constexpr uint32_t kNumFnStages = 2;
static constexpr uint32_t kNumAStages = 4;
static constexpr uint32_t kNumAccumStages = 2;
static constexpr uint32_t kNumCoeffStages = 2;
static constexpr uint32_t kNumWarpsPerWG = 4;
static constexpr uint32_t kNumThreads = 6 * kNumWarpsPerWG * 32;

// Split-K workspace geometry
static constexpr uint32_t kNumGemmPartialElementsPerTask = BLOCK_M * kNumHCOutputs;
static constexpr uint32_t kNumSqrSumPartialElementsPerTask = BLOCK_M;
static constexpr uint32_t kDefaultNumSplits = 16;
static constexpr uint32_t kNumMaxSplits = 64;
static constexpr uint32_t kNumSplitBarriers = 2;
static constexpr uint64_t kSplitBarrierLineBytes = 128;
static constexpr uint32_t kNumMaxTokens = 1u << 20;
static constexpr uint32_t kNumMaxMBlocks = math::constexpr_ceil_div(kNumMaxTokens, BLOCK_M);
// A barrier line holds one 8-byte state; the execution epoch, the exit count and the task queue counter
// live in the unused tail of the buffer's first line, so the buffer is unchanged.
// The task queue counter relies on the per-launch memset of the active barrier lines to start at zero.
static constexpr uint64_t kSplitBarrierEpochOffset = kSplitBarrierLineBytes / 2;
static constexpr uint64_t kSplitBarrierExitCountOffset = kSplitBarrierEpochOffset + sizeof(uint64_t);
static constexpr uint64_t kSplitBarrierTaskCounterOffset = kSplitBarrierExitCountOffset + sizeof(uint64_t);
DG_STATIC_ASSERT(kSplitBarrierTaskCounterOffset + sizeof(uint32_t) <= kSplitBarrierLineBytes,
                 "Epoch, exit count and task counter must fit in the first barrier line");

// Host/device ABI payloads for the two consumer domains. Tensor maps remain
// standalone kernel parameters so TMA descriptors keep direct grid-constant access.
struct MixArgs {
    const float* scales;
    const float* bases;
    float* new_prev_mix;
    float* new_post_mix;
    float* new_comb_res_mix;
    float hc_norm_eps;
    float hc_pre_eps;
    float hc_post_scale;
    float sinkhorn_eps;
    uint32_t num_sinkhorn_iters;
};

struct NormArgs {
    uint32_t num_tokens;
    const nv_bfloat16* weight;
    const nv_bfloat16* new_residual;
    float eps;
    float scale;
    nv_bfloat16* y_bf16;
    __nv_fp8_e4m3* y_fp8;
    uint32_t* y_primary_sf;
    int64_t y_primary_sf_stride_token;
    int64_t y_primary_sf_stride_word;
    // Non-zero selects FlashInfer's 128x4 swizzle: words between 128-token tiles
    int64_t y_primary_sf_stride_mtile;
    uint32_t* y_shared_sf;
    int64_t y_shared_sf_stride_word;
};

// View over global split partial scratch.
//
// Data layout:
//   gemm_partials            [num_m_blocks, kNumSplits, BLOCK_M, kNumHCOutputs]
//   hc_norm_sqr_sum_partials [num_m_blocks, kNumSplits, BLOCK_M]
//   x1_sqr_sum_partials      [num_m_blocks, kNumSplits, BLOCK_M]
// where task_idx = m_block_idx * kNumSplits + k_split_idx.
template <uint32_t kNumSplits = kDefaultNumSplits>
struct Workspace {
    void* gmem_scratch;
    uint32_t num_m_blocks;

    DG_STATIC_ASSERT(kNumSplits > 0, "The number of splits must be positive");
    DG_STATIC_ASSERT(kNumSplits <= kNumMaxSplits, "The number of splits exceeds the supported maximum");

    CUTLASS_HOST_DEVICE
    Workspace(void* gmem_scratch, const uint32_t num_m_blocks):
        gmem_scratch(gmem_scratch), num_m_blocks(num_m_blocks) {}

    CUTLASS_HOST_DEVICE
    static constexpr uint32_t get_num_m_blocks(const uint32_t num_tokens) {
        return math::constexpr_ceil_div<uint32_t>(num_tokens, BLOCK_M);
    }

    CUTLASS_HOST_DEVICE
    static constexpr uint64_t get_num_scratch_bytes(const uint32_t num_m_blocks, const uint32_t num_splits = kNumSplits) {
        const auto num_tasks = static_cast<uint64_t>(num_m_blocks) * num_splits;
        return num_tasks * (kNumGemmPartialElementsPerTask + 2 * kNumSqrSumPartialElementsPerTask) * sizeof(float);
    }

    CUTLASS_HOST_DEVICE
    float* get_gemm_partial_ptr(const uint32_t task_idx = 0) const {
        return reinterpret_cast<float*>(gmem_scratch) + static_cast<uint64_t>(task_idx) * kNumGemmPartialElementsPerTask;
    }

    CUTLASS_HOST_DEVICE
    float* get_hc_norm_sqr_sum_partial_ptr(const uint32_t task_idx = 0) const {
        const auto base = get_gemm_partial_ptr(num_m_blocks * kNumSplits);
        return base + static_cast<uint64_t>(task_idx) * kNumSqrSumPartialElementsPerTask;
    }

    CUTLASS_HOST_DEVICE
    float* get_x1_sqr_sum_partial_ptr(const uint32_t task_idx = 0) const {
        const auto base = get_hc_norm_sqr_sum_partial_ptr(num_m_blocks * kNumSplits);
        return base + static_cast<uint64_t>(task_idx) * kNumSqrSumPartialElementsPerTask;
    }

};

struct SharedStorage {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Route-major TMA layouts keep every 64 x 64 destination contiguous.
    alignas(kSwizzleAlignment) nv_bfloat162 residual[kNumIOStages][kNumRoutes][BLOCK_M * BLOCK_K / 2];
    alignas(kSwizzleAlignment) nv_bfloat162 x[kNumIOStages][BLOCK_M * BLOCK_K / 2];

    // Each Fn TMA fills all routes for one 32-element TF32 swizzle atom.
    alignas(kSwizzleAlignment) float fn[kNumFnStages][BLOCK_K / kFnAtomK][kNumRoutes][kNumHCOutputs * kFnAtomK];

    // Coeff tensors retain their TMA-native per-token layouts.
    alignas(kSwizzleMode) float pre_coeff[kNumCoeffStages][BLOCK_M][kNumRoutes];
    alignas(kSwizzleMode) float post_coeff[kNumCoeffStages][BLOCK_M][kNumRoutes];
    alignas(kSwizzleMode) float comb_coeff[kNumCoeffStages][BLOCK_M][kNumRoutes * kNumRoutes];

    // Post workers own independent partials reduced before the Mix/Norm barrier arrivals.
    alignas(kSwizzleMode) float hc_norm_sqr_sums[2][BLOCK_M];
    alignas(kSwizzleMode) float x1_sqr_sums[2][BLOCK_M];
    // Fused Normal workers use dedicated scratch instead of aliasing Shifted statistics.
    alignas(kSwizzleMode) float normal_scratch[BLOCK_M];

    // Full/empty pairs connect static roles without CTA-wide block-boundary synchronization.
    alignas(Barrier) Barrier full_io_barriers[kNumIOStages];
    alignas(Barrier) Barrier empty_io_barriers[kNumIOStages];
    alignas(Barrier) Barrier store_ready_io_barriers[kNumIOStages];

    alignas(Barrier) Barrier full_fn_barriers[kNumFnStages];
    alignas(Barrier) Barrier empty_fn_barriers[kNumFnStages];

    alignas(Barrier) Barrier full_a_barriers[kNumAStages];
    alignas(Barrier) Barrier empty_a_barriers[kNumAStages];

    alignas(Barrier) Barrier full_accum_barriers[kNumAccumStages];
    alignas(Barrier) Barrier empty_accum_barriers[kNumAccumStages];

    alignas(Barrier) Barrier full_coeff_barriers[kNumCoeffStages];
    alignas(Barrier) Barrier empty_coeff_barriers[kNumCoeffStages];

    alignas(Barrier) Barrier full_stats_barrier;
    alignas(Barrier) Barrier empty_stats_barrier;

    // Gate each split's Mix arrival on all WG3 warps and, in Normal, the output-store worker.
    alignas(Barrier) Barrier mix_arrival_barrier;
    alignas(uint32_t) uint32_t tmem_base;
    // Shifted Norm and fused Normal workers share this mutually exclusive task queue.
    alignas(uint32_t) uint32_t next_norm_task_ticket;
};

// In-CTA Norm keeps several tasks' X1 tiles behind SharedStorage (VR oversized shared memory)
static constexpr uint32_t kNumX1Buffers = 3;

CUTLASS_HOST_DEVICE constexpr uint32_t get_x1_tiles_smem_offset() {
    return math::constexpr_align(static_cast<uint32_t>(sizeof(SharedStorage)), kSwizzleAlignment);
}

CUTLASS_HOST_DEVICE constexpr uint32_t get_max_num_k_blocks_per_task(const uint32_t& hidden, const uint32_t& num_splits) {
    return math::constexpr_ceil_div(hidden / BLOCK_K, num_splits);
}

CUTLASS_HOST_DEVICE constexpr uint32_t get_x1_tiles_smem_size(const uint32_t& max_num_k_blocks_per_task) {
    return kNumX1Buffers * max_num_k_blocks_per_task * BLOCK_M * BLOCK_K * static_cast<uint32_t>(sizeof(nv_bfloat16));
}

// The RMSNorm weight follows the X1 tiles, staged once per CTA
CUTLASS_HOST_DEVICE constexpr uint32_t get_in_cta_norm_smem_size(const uint32_t& max_num_k_blocks_per_task, const uint32_t& hidden) {
    return get_x1_tiles_smem_size(max_num_k_blocks_per_task) + hidden * static_cast<uint32_t>(sizeof(nv_bfloat16));
}

} // namespace deep_gemm::layout::mega_mhc
