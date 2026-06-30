#pragma once

#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) || defined(__CLION_IDE__)

#include <cuda/std/cstdint>

namespace deep_gemm::mma::sm120 {

static constexpr int FP4_MMA_M = 16;
static constexpr int FP4_MMA_N = 8;
static constexpr int FP4_MMA_K = 64;

__device__ __forceinline__ uint16_t extract_sf_pair(
        const uint32_t packed, const uint32_t first_byte_idx) {
    return static_cast<uint16_t>(
        (packed >> (first_byte_idx * 8)) & 0xffff);
}

__device__ __forceinline__ void fp4_mma_block_scaled(
        float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
        const uint16_t sfa, const uint16_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {%11, %12}, {%13}, {%14, %15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(static_cast<uint32_t>(sfa)),
          "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)),
          "r"(static_cast<uint32_t>(sfb)),
          "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)));
}

}  // namespace deep_gemm::mma::sm120

#endif
