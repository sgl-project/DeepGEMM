#pragma once
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/mqa_bf16/packing.cuh>
#include <cute/arch/copy_sm100.hpp>
#include <cutlass/arch/barrier.h>

namespace deep_gemm::mqa_bf16::math {
using namespace deep_gemm::math;
// Block-wide prefix sum. Synchronize the block before reusing `warp_sums`.
template <uint32_t kNumThreads, typename T>
CUTLASS_DEVICE T cta_exclusive_sum(const T value, T* warp_sums, T& total) {
    constexpr uint32_t kNumWarps = kNumThreads / 32;
    DG_STATIC_ASSERT(kNumThreads >= 32 and kNumThreads % 32 == 0 and kNumWarps <= 32, "Invalid CTA scan shape");

    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const T lane_sum = warp_inclusive_sum(value, lane_idx);
    if (lane_idx == 31)
        warp_sums[warp_idx] = lane_sum;
    __syncthreads();

    const T warp_total = lane_idx < kNumWarps ? warp_sums[lane_idx] : 0;
    const T warp_sum = warp_inclusive_sum(warp_total, lane_idx);
    total = ptx::exchange(warp_sum, kNumWarps - 1);
    return lane_sum - value + ptx::exchange(warp_sum - warp_total, warp_idx);
}

template <uint32_t kNumThreads, typename T>
CUTLASS_DEVICE T cta_exclusive_sum(const T value, T* warp_sums) {
    T total;
    return cta_exclusive_sum<kNumThreads>(value, warp_sums, total);
}


} // namespace deep_gemm::mqa_bf16::math
namespace deep_gemm::mqa_bf16::ptx {
using namespace deep_gemm::ptx;
template <uint32_t kNumValues>
CUTLASS_DEVICE void tmem_load_32dp32b(const uint32_t tmem_addr, uint32_t* values) {
    DG_STATIC_ASSERT(kNumValues == 4 or kNumValues == 8 or kNumValues == 16 or
                     kNumValues == 32 or kNumValues == 64 or kNumValues == 128,
                     "Invalid TMEM load width");
    using Loader = cute::conditional_t<kNumValues == 4, cute::SM100_TMEM_LOAD_32dp32b4x,
                   cute::conditional_t<kNumValues == 8, cute::SM100_TMEM_LOAD_32dp32b8x,
                   cute::conditional_t<kNumValues == 16, cute::SM100_TMEM_LOAD_32dp32b16x,
                   cute::conditional_t<kNumValues == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                   cute::conditional_t<kNumValues == 64, cute::SM100_TMEM_LOAD_32dp32b64x,
                                                               cute::SM100_TMEM_LOAD_32dp32b128x>>>>>;
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        Loader::copy(tmem_addr, values[Is]...);
    }(cute::make_index_sequence<kNumValues>{});
}

CUTLASS_DEVICE void umma_arrive_no_elect(cutlass::arch::ClusterTransactionBarrier& barrier) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&barrier))));
}

// Store the low 16 bits of a packed pair: the `.b16` store truncates a 32-bit source register
CUTLASS_DEVICE void st_global_low_bf16(nv_bfloat16* ptr, const nv_bfloat162& value) {
    asm volatile("st.global.b16 [%0], %1;" :: "l"(ptr), "r"(*reinterpret_cast<const uint32_t*>(&value)));
}

CUTLASS_DEVICE void mbarrier_arrive_count(
    cutlass::arch::ClusterTransactionBarrier& barrier, const uint32_t count) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&barrier))), "r"(count));
}


} // namespace deep_gemm::mqa_bf16::ptx
namespace deep_gemm::mqa_bf16 {
// This producer only uses K-major operands. Offsets are logical FP4 elements.
template <typename T>
CUTLASS_DEVICE uint32_t advance_packed_umma_desc_lo(uint32_t base, uint32_t offset, uint32_t k) {
    return base + ((offset + k) * static_cast<uint32_t>(sizeof(T)) / get_smem_pack_factor<T>() >> 4u);
}
} // namespace deep_gemm::mqa_bf16
