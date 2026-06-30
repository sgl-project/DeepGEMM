#pragma once

#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) || defined(__CLION_IDE__)

#include <cute/swizzle.hpp>

#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm::sm120 {

template <int kSwizzleBytes>
using CuTeSwizzle = cute::Swizzle<__builtin_ctz(kSwizzleBytes) - 4, 4, 3>;

template <int kSwizzleBytes>
struct SwizzleContext {
    int row_base_addr;
    int row_xor_bits;

    CUTLASS_DEVICE void init(const int row, const int row_stride) {
        row_base_addr = row * row_stride;
        row_xor_bits =
            CuTeSwizzle<kSwizzleBytes>::apply(row_base_addr) ^ row_base_addr;
    }

    CUTLASS_DEVICE void* addr(char* smem_tile, const int col_byte) const {
        return smem_tile + row_base_addr + (col_byte ^ row_xor_bits);
    }
};

template <int kSwizzleBytes>
CUTLASS_DEVICE void load_a_fragment(
        uint32_t (&frag)[4], char* smem_a,
        const SwizzleContext<kSwizzleBytes>& ctx, const int lane,
        const int k_step, const int mma_k) {
    const int col = (lane >> 4) * 16 + k_step * mma_k;
    ptx::SM90_U32x4_LDSM_N::copy(
        frag[0], frag[1], frag[2], frag[3], ctx.addr(smem_a, col));
}

template <int kSwizzleBytes>
CUTLASS_DEVICE void load_b_fragment_x2(
        uint32_t (&frag)[2], char* smem_b,
        const SwizzleContext<kSwizzleBytes>& ctx, const int lane,
        const int k_step, const int mma_k) {
    const int col = ((lane >> 3) & 1) * 16 + k_step * mma_k;
    ptx::SM90_U32x2_LDSM_N::copy(frag[0], frag[1], ctx.addr(smem_b, col));
}

CUTLASS_DEVICE uint32_t load_sf(const char* smem_sf, const int idx) {
    return *reinterpret_cast<const uint32_t*>(
        smem_sf + idx * static_cast<int>(sizeof(int32_t)));
}

}  // namespace deep_gemm::sm120

#endif
