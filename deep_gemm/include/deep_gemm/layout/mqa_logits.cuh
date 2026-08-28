#pragma once

#include <cuda_fp8.h>

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>

namespace deep_gemm::layout {

template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsMXSF,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumTmemStages,
          typename qk_dtype_t, typename reduce_dtype_t = float>
struct MQALogitsSharedStorage {
    static constexpr bool kIsFP4 = cute::is_same_v<qk_dtype_t, cutlass::float_e2m1_t>;
    static constexpr bool kIsMXFP8 = kIsMXSF and not kIsFP4;

    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    // Preserve the storage types used by the original FP8/MXFP4 kernels.  In
    // particular, keeping packed FP4 as bytes avoids perturbing ptxas register
    // allocation for the register-bound H=8 specialization.
    using qk_storage_dtype_t = cute::conditional_t<kIsFP4, uint8_t, __nv_fp8_e4m3>;
    using sf_dtype_t = cute::conditional_t<kIsMXSF, uint32_t, float>;

    static constexpr uint32_t kNumUTCCPAlignedElems = 128;
    static constexpr uint32_t kQKBytesPerElem = sizeof(qk_storage_dtype_t);
    static constexpr uint32_t kNumQKBytesPerToken = kIsFP4 ? (kHeadDim / 2) : kHeadDim;
    // Keep the established FP8/MXFP4 layout byte-for-byte stable. MXFP8 uses
    // the generalized alignment introduced with its support.
    static constexpr uint32_t kSwizzleAlignment = kIsMXFP8 ? 8 * kNumQKBytesPerToken
                                                           : (kIsFP4 ? 8 * kNumQKBytesPerToken : 512);
    static constexpr uint32_t kNumSFQ = math::constexpr_align(BLOCK_Q * kNumHeads, kNumUTCCPAlignedElems);
    static constexpr uint32_t kNumSFKV = math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems);
    static constexpr uint32_t kNumQBytesPerStage = BLOCK_Q * kNumHeads * kNumQKBytesPerToken;
    static constexpr uint32_t kNumKVBytesPerStage = SPLIT_KV * kNumQKBytesPerToken;
    static constexpr uint32_t kNumQElementsPerStage = kNumQBytesPerStage / kQKBytesPerElem;
    static constexpr uint32_t kNumKVElementsPerStage = kNumKVBytesPerStage / kQKBytesPerElem;
    // MX SF formats store per-block scale factors; FP8 stores one per-KV scale and no Q scale
    static constexpr uint32_t kNumScaleQ = kIsMXSF ? kNumSFQ : 1;
    static constexpr uint32_t kNumScaleKV = kIsMXSF ? kNumSFKV : SPLIT_KV;
    // Preserve the original scale-array alignment for existing formats; all
    // TMA destinations still satisfy the minimum 128-byte requirement when
    // they are used by TMA.
    static constexpr uint32_t kScaleAlignment = kIsMXFP8 ? 128 : (kIsFP4 ? 16 : 512);
    static constexpr uint32_t kWeightsTmaAlignment = 128;
    static constexpr uint32_t kNumWeightsElementsPerStage =
        math::constexpr_align(BLOCK_Q * kNumHeads * static_cast<uint32_t>(sizeof(reduce_dtype_t)), kWeightsTmaAlignment)
        / static_cast<uint32_t>(sizeof(reduce_dtype_t));

    DG_STATIC_ASSERT(kNumQBytesPerStage % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(kNumKVBytesPerStage % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(kSwizzleAlignment % 128 == 0, "TMA destination must be 128-byte aligned");
    DG_STATIC_ASSERT(kWeightsTmaAlignment % 128 == 0, "TMA destination must be 128-byte aligned");

    alignas(kSwizzleAlignment) qk_storage_dtype_t smem_q[kNumQStages][kNumQElementsPerStage];
    alignas(kSwizzleAlignment) qk_storage_dtype_t smem_kv[kNumKVStages][kNumKVElementsPerStage];
    alignas(kScaleAlignment) sf_dtype_t smem_sf_q[kNumQStages][kNumScaleQ];
    alignas(kScaleAlignment) sf_dtype_t smem_sf_kv[kNumKVStages][kNumScaleKV];
    alignas(kWeightsTmaAlignment) reduce_dtype_t smem_weights[kNumQStages][kNumWeightsElementsPerStage];
    // Barriers require 8-byte alignment, already guaranteed by the preceding TMA-aligned arrays.
    Barrier full_q_barriers[kNumQStages];
    Barrier empty_q_barriers[kNumQStages];
    Barrier full_kv_barriers[kNumKVStages];
    Barrier empty_kv_barriers[kNumKVStages];
    Barrier full_tmem_barriers[kNumTmemStages];
    Barrier empty_tmem_barriers[kNumTmemStages];
    uint32_t tmem_ptr_in_smem;
};

} // namespace deep_gemm::layout
