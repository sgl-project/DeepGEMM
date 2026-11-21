#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda/std/cstdint>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

// Fused BF16 → quant + topk copy + pad-fill kernel that produces the exact
// byte layout DeepGEMM's mega-MoE symmetric buffer expects in its `x`,
// `x_sf`, `topk_idx`, and `topk_weights` slots. The acts format follows the
// consuming GEMM's `MmaKind`:
//
//   - `MXFP8FP4` → FP8 (E4M3) acts + UE8M0 group scales; row stride = `hidden`.
//   - `MXFP4`    → packed FP4 (E2M1) acts + UE8M0 group scales; row stride
//                  = `hidden / 2`. Layout: byte holds 2 nibbles, low nibble =
//                  even col, high nibble = odd col, matching
//                  `deep_gemm.utils.per_token_cast_to_fp4`.
//   - `NVFP4`    → same packing as `MXFP4`, but UE4M3 SFs over 16-element
//                  groups plus one FP32 per-token outer scale in `buf_x_scales`.
//
// All paths share the SF byte layout: `byte_off = token*num_groups + group`,
// with the contiguous `(P, num_groups/4)` int32 slot storing 4 bytes per int32
// in row-major order.
//
// The FP4 quant matches `per_token_cast_to_fp4` (host helper) bytewise,
// including round-to-nearest-even at E2M1 midpoints.

// ceil_to_ue8m0(raw_scale) — matches `deep_gemm.utils.math.ceil_to_ue8m0`:
// returns the UE8M0 exponent byte (in [1, 254]) such that 2^(exp-127) is the
// smallest power of 2 >= raw_scale.
__forceinline__ __device__ uint32_t pre_dispatch_cast_to_ue8m0(float raw_scale) {
    uint32_t bits = __float_as_uint(raw_scale);
    uint32_t exp = (bits >> 23u) & 0xFFu;
    uint32_t mantissa = bits & 0x7FFFFFu;
    if (mantissa != 0u) exp += 1u;
    if (exp < 1u)   exp = 1u;
    if (exp > 254u) exp = 254u;
    return exp;
}

__forceinline__ __device__ float pre_dispatch_warp_max_nonneg(float val, uint32_t mask) {
    const uint32_t u = __float_as_uint(val);
    uint32_t m;
    asm volatile("redux.sync.max.u32 %0, %1, %2;" : "=r"(m) : "r"(u), "r"(mask));
    return __uint_as_float(m);
}

template <bool kFullWarps>
__forceinline__ __device__ float pre_dispatch_block_reduce_max(float val) {
    __shared__ float smem[32];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t num_warps = (blockDim.x + 31u) >> 5u;
    uint32_t lanes = 32u;
    uint32_t mask = 0xffffffffu;
    if constexpr (!kFullWarps) {
        lanes = min(32u, blockDim.x - (threadIdx.x & ~31u));
        mask  = lanes == 32u ? 0xffffffffu : (1u << lanes) - 1u;
    }
    val = pre_dispatch_warp_max_nonneg(val, mask);
    if (lane == 0u)
        smem[warp] = val;
    __syncthreads();
    if constexpr (kFullWarps) {
        val = (lane < num_warps) ? smem[lane] : 0.0f;
    } else {
        val = 0.0f;
        for (uint32_t w = lane; w < num_warps; w += lanes)
            val = fmaxf(val, smem[w]);
    }
    return pre_dispatch_warp_max_nonneg(val, mask);
}

template <uint32_t kGroupSize, MmaKind kMmaKind, bool kUsePDL, uint32_t kVecElems = 8>
__launch_bounds__(1024, 2)
__global__ void mega_moe_pre_dispatch_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t*       __restrict__ topk_idx,
    const float*         __restrict__ topk_weights,
    const float*         __restrict__ expert_scales,
    void*                __restrict__ buf_x,
    int32_t*             __restrict__ buf_x_sf,
    float*               __restrict__ buf_x_scales,
    int64_t*             __restrict__ buf_topk_idx,
    float*               __restrict__ buf_topk_weights,
    const uint32_t num_tokens,
    const uint32_t padded_max,
    const uint32_t hidden,
    const uint32_t num_groups,
    const uint32_t top_k) {
    static_assert(kGroupSize == 16 || kGroupSize == 32 || kGroupSize == 64 || kGroupSize == 128,
                  "kGroupSize must be 16, 32, 64, or 128");
    static_assert(kVecElems == 8 || kVecElems == 16, "kVecElems must be 8 or 16");
    constexpr bool kNvfp4 = kMmaKind == MmaKind::NVFP4;
    constexpr bool kPackedFp4 = get_element_bits(kMmaKind) == 4;
    static_assert(kMmaKind == MmaKind::MXFP8FP4 || kMmaKind == MmaKind::MXFP4 || kNvfp4,
                  "Acts must be FP8 (MXFP8FP4) or packed FP4 (MXFP4 / NVFP4)");
    static_assert(!kNvfp4 || kGroupSize == 16,
                  "NVFP4 acts use 16-element UE4M3 blocks");
    static_assert(kNvfp4 || kGroupSize >= 32,
                  "UE8M0 modes keep the original 32/64/128 group sizes");
    static_assert(kVecElems == 8 || kNvfp4,
                  "16-element per-thread vectorization is specialized for 1-group/thread NVFP4");
    static_assert(kGroupSize % kVecElems == 0, "kGroupSize must be a multiple of kVecElems");
    constexpr uint32_t kThreadsPerGroup = kGroupSize / kVecElems;

    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    if constexpr (kUsePDL) {
        cudaGridDependencySynchronize();
    }

    if (bid < num_tokens) {
        // ---- Quantize path: one CTA per valid token ----
        const uint32_t token_id = bid;
        const auto* token_in = x + static_cast<uint64_t>(token_id) * hidden;

        if constexpr (kVecElems == 16) {
            // 1-group/thread NVFP4 path: each thread loads 16 BF16 elements (2 x uint4),
            // eliminating cross-lane group amax shuffles and halving CTA thread count.
            const auto* in_u4 = reinterpret_cast<const uint4*>(token_in) + tid * 2u;
            const uint4 bits0 = in_u4[0];
            const uint4 bits1 = in_u4[1];
            const auto* p0 = reinterpret_cast<const __nv_bfloat162*>(&bits0);
            const auto* p1 = reinterpret_cast<const __nv_bfloat162*>(&bits1);

            const __nv_bfloat162 m0 = __hmax2(__habs2(p0[0]), __habs2(p0[1]));
            const __nv_bfloat162 m1 = __hmax2(__habs2(p0[2]), __habs2(p0[3]));
            const __nv_bfloat162 m2 = __hmax2(__habs2(p1[0]), __habs2(p1[1]));
            const __nv_bfloat162 m3 = __hmax2(__habs2(p1[2]), __habs2(p1[3]));
            const __nv_bfloat162 m_all = __hmax2(__hmax2(m0, m1), __hmax2(m2, m3));
            const float local_max = __bfloat162float(__hmax(m_all.x, m_all.y));

            const float row_max = fmaxf(pre_dispatch_block_reduce_max<true>(local_max), 1e-4f);
            const float pts = row_max / (448.0f * 6.0f);
            const __nv_fp8_storage_t sf_fp8 = __nv_cvt_float_to_fp8(
                local_max / (6.0f * pts), __NV_SATFINITE, __NV_E4M3);
            const __half_raw sf_hr = __nv_cvt_fp8_to_halfraw(sf_fp8, __NV_E4M3);
            const float sf_dec = __half2float(*reinterpret_cast<const __half*>(&sf_hr));
            const float inv_scale = sf_dec > 0.0f ? 1.0f / (sf_dec * pts) : 0.0f;

            if (tid == 0u)
                buf_x_scales[token_id] = pts;

            // Coalesce 4 adjacent groups' UE4M3 SF bytes into one int32_t store.
            const uint32_t sf_u32 = static_cast<uint32_t>(sf_fp8);
            const uint32_t sf01 = __byte_perm(sf_u32, __shfl_xor_sync(0xffffffffu, sf_u32, 1), 0x0040u);
            const uint32_t sf0123 = __byte_perm(sf01, __shfl_xor_sync(0xffffffffu, sf01, 2), 0x5410u);
            if ((tid & 3u) == 0u) {
                buf_x_sf[(token_id * num_groups + tid) >> 2u] = static_cast<int32_t>(sf0123);
            }

            const float2 inv2 = {inv_scale, inv_scale};
            const uint32_t lo32 = math::cvt_pack_f32x8_to_e2m1x8<true>(
                __fmul2_rn(__bfloat1622float2(p0[0]), inv2),
                __fmul2_rn(__bfloat1622float2(p0[1]), inv2),
                __fmul2_rn(__bfloat1622float2(p0[2]), inv2),
                __fmul2_rn(__bfloat1622float2(p0[3]), inv2));
            const uint32_t hi32 = math::cvt_pack_f32x8_to_e2m1x8<true>(
                __fmul2_rn(__bfloat1622float2(p1[0]), inv2),
                __fmul2_rn(__bfloat1622float2(p1[1]), inv2),
                __fmul2_rn(__bfloat1622float2(p1[2]), inv2),
                __fmul2_rn(__bfloat1622float2(p1[3]), inv2));

            auto* row_out64 = static_cast<uint64_t*>(buf_x) +
                              static_cast<uint64_t>(token_id) * (hidden / 16u);
            row_out64[tid] = (static_cast<uint64_t>(hi32) << 32u) | static_cast<uint64_t>(lo32);
        } else {
            // Coalesced 16-byte BF16 vector load (8 elements per thread).
            uint4 in_bits = reinterpret_cast<const uint4*>(token_in)[tid];
            const auto* bf16_pairs = reinterpret_cast<const __nv_bfloat162*>(&in_bits);

            float vals[kVecElems];
            const __nv_bfloat162 m01 = __hmax2(__habs2(bf16_pairs[0]), __habs2(bf16_pairs[1]));
            const __nv_bfloat162 m23 = __hmax2(__habs2(bf16_pairs[2]), __habs2(bf16_pairs[3]));
            float local_max = __bfloat162float(__hmax(__hmax2(m01, m23).x, __hmax2(m01, m23).y));
            #pragma unroll
            for (uint32_t i = 0; i < kVecElems / 2; ++i) {
                float2 fp = __bfloat1622float2(bf16_pairs[i]);
                vals[2 * i + 0] = fp.x;
                vals[2 * i + 1] = fp.y;
            }

            // Reduce absmax across the kThreadsPerGroup threads that cover one
            // group. Lanes outside the group keep their own value (different
            // group's max), so SF write below is gated to one thread per group.
            local_max = math::warp_reduce<kThreadsPerGroup, /*kIntergroupReduce=*/false>(
                local_max, math::ReduceMax<float>{});

            float inv_scale;
            uint32_t sf_byte;
            if constexpr (kNvfp4) {
                // NVFP4 per-token recipe: x ≈ e2m1 * ue4m3(sf) * pts with
                //   pts = row_absmax / (448 * 6)
                //   sf  = e4m3_rn(group_absmax / (6 * pts))   (≤ 448 by construction)
                const float row_max = fmaxf(
                    pre_dispatch_block_reduce_max<false>(local_max), 1e-4f);
                const float pts = row_max / (448.0f * 6.0f);
                const __nv_fp8_storage_t sf_fp8 = __nv_cvt_float_to_fp8(
                    local_max / (6.0f * pts), __NV_SATFINITE, __NV_E4M3);
                const __half_raw sf_hr = __nv_cvt_fp8_to_halfraw(sf_fp8, __NV_E4M3);
                const float sf_dec = __half2float(*reinterpret_cast<const __half*>(&sf_hr));
                inv_scale = sf_dec > 0.0f ? 1.0f / (sf_dec * pts) : 0.0f;
                sf_byte = static_cast<uint32_t>(sf_fp8);
                if (tid == 0u)
                    buf_x_scales[token_id] = pts;
            } else {
                // Match host `per_token_cast_to_fp4/fp8`: clamp absmax to 1e-4
                // before dividing by the dtype's max representable value.
                const float absmax = fmaxf(local_max, 1e-4f);
                constexpr float kFinfoMax = kPackedFp4 ? 6.0f : 448.0f;
                const float raw_scale = absmax / kFinfoMax;
                const uint32_t ue8m0_exp = pre_dispatch_cast_to_ue8m0(raw_scale);
                sf_byte = ue8m0_exp;
                inv_scale = __uint_as_float((254u - sf_byte) << 23u);
            }

            if constexpr (kPackedFp4) {
                const float2 inv2 = {inv_scale, inv_scale};
                const uint32_t packed = math::cvt_pack_f32x8_to_e2m1x8<true>(
                    __fmul2_rn({vals[0], vals[1]}, inv2),
                    __fmul2_rn({vals[2], vals[3]}, inv2),
                    __fmul2_rn({vals[4], vals[5]}, inv2),
                    __fmul2_rn({vals[6], vals[7]}, inv2));
                auto* row_out = static_cast<uint32_t*>(buf_x) +
                                static_cast<uint64_t>(token_id) * (hidden / 8u);
                row_out[tid] = packed;
            } else {
                uint64_t packed = 0;
                #pragma unroll
                for (uint32_t i = 0; i < kVecElems / 2; ++i) {
                    const __nv_fp8x2_storage_t fp8x2 = __nv_cvt_float2_to_fp8x2(
                        make_float2(vals[2 * i + 0] * inv_scale, vals[2 * i + 1] * inv_scale),
                        __NV_SATFINITE, __NV_E4M3);
                    packed |= static_cast<uint64_t>(fp8x2) << (16u * i);
                }
                auto* row_out = static_cast<uint64_t*>(buf_x) +
                                static_cast<uint64_t>(token_id) * (hidden / 8u);
                row_out[tid] = packed;
            }

            const uint32_t group_id = tid / kThreadsPerGroup;
            const uint32_t within_group_id = tid % kThreadsPerGroup;
            if (within_group_id == 0u && group_id < num_groups) {
                const uint32_t byte_off = token_id * num_groups + group_id;
                reinterpret_cast<uint8_t*>(buf_x_sf)[byte_off] =
                    static_cast<uint8_t>(sf_byte);
            }
        }

        // Copy this token's topk row, optionally folding a per-expert scale
        // (e.g. w2 weight_scale_2 ratios) into the weight. top_k is small
        // (≤ num_threads enforced at host); each tid<top_k thread copies one.
        if (tid < top_k) {
            const uint32_t off = token_id * top_k + tid;
            const int32_t expert = topk_idx[off];
            buf_topk_idx[off] = static_cast<int64_t>(expert);
            float w = topk_weights[off];
            if (expert_scales != nullptr && expert >= 0)
                w *= expert_scales[expert];
            buf_topk_weights[off] = w;
        }
    } else {
        // ---- Pad path: trailing CTAs fill [num_tokens, padded_max) topk
        // slots with (-1, 0.0), 4 slots per thread (vectorized int4/float4 on
        // 16B-aligned interior, scalar fallback at unaligned edges).
        const uint32_t copy_bid  = bid - num_tokens;
        const uint32_t pad_base  = num_tokens * top_k;
        const uint32_t total     = padded_max * top_k;
        const uint32_t base_slot = pad_base + (copy_bid * blockDim.x + tid) * 4u;
        const uintptr_t align_mask =
            reinterpret_cast<uintptr_t>(buf_topk_idx + base_slot) |
            reinterpret_cast<uintptr_t>(buf_topk_weights + base_slot);
        if ((align_mask & 15u) == 0u && base_slot + 3u < total) {
            reinterpret_cast<int4*>(&buf_topk_idx[base_slot])[0]      = make_int4(-1, -1, -1, -1);
            reinterpret_cast<int4*>(&buf_topk_idx[base_slot])[1]      = make_int4(-1, -1, -1, -1);
            reinterpret_cast<float4*>(&buf_topk_weights[base_slot])[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        } else {
            #pragma unroll
            for (uint32_t k = 0; k < 4u; ++k) {
                if (base_slot + k < total) {
                    buf_topk_idx[base_slot + k]     = static_cast<int64_t>(-1);
                    buf_topk_weights[base_slot + k] = 0.0f;
                }
            }
        }
    }

    if constexpr (kUsePDL) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

}  // namespace deep_gemm
