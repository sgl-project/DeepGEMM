#pragma once

#include <deep_gemm/common/sm90_mxfp4_decode.cuh>

namespace deep_gemm {

template <typename = void>
__global__ void sm90_mxfp4_mark_active_experts(const int32_t* ids, int32_t* mask,
                                              const int64_t count, const int num_experts) {
    extern __shared__ uint32_t present[];
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
        present[e] = 0;
    __syncthreads();
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        const int e = ids[i];
        if (e >= 0 and e < num_experts)
            atomicExch(present + e, 1u);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
        if (present[e])
            atomicExch(mask + e, 1);
}

template <bool kMasked>
__global__ void sm90_mxfp4_expand_tiled(const uint8_t* packed, const uint32_t* sf,
                                      uint8_t* output, const int32_t* mask, const int expert_offset,
                                      const int n, const int k) {
    if constexpr (kMasked)
        if (not mask[expert_offset + blockIdx.y])
            return;
    const int row = threadIdx.x;
    const int kc = blockIdx.x % (k / 128);
    const int nt = blockIdx.x / (k / 128);
    const int e = blockIdx.y;
    const int64_t tile = (static_cast<int64_t>(e) * (n / 256) + nt) * (k / 128) + kc;
    const uint32_t scale = sf[(static_cast<int64_t>(e) * (k / 128) + kc) * n + nt * 256 + row];
    uint8_t* dst = output + tile * 32768 + (row / 8) * 1024 + (row % 8) * 16;
    #pragma unroll
    for (int g = 0; g < 4; ++ g) {
        const uint4 word = *reinterpret_cast<const uint4*>(packed + tile * 16384 + g * 4096 + row * 16);
        const auto lut = sm90_mxfp4_e4m3_lookup((scale >> (8 * g)) & 0xffu);
        const auto a = sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(word.x, lut);
        const auto b = sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(word.y, lut);
        const auto c = sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(word.z, lut);
        const auto d = sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(word.w, lut);
        *reinterpret_cast<uint4*>(dst + (2 * g) * 128) = make_uint4(a.x, b.x, c.x, d.x);
        *reinterpret_cast<uint4*>(dst + (2 * g + 1) * 128) = make_uint4(a.y, b.y, c.y, d.y);
    }
}

} // namespace deep_gemm
