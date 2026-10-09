#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <cutlass/arch/barrier.h>

namespace deep_gemm::epilogue {

// Coarse score histogram of the paged MQA producers (FP32 `fp8_fp4_paged_mqa_logits` and BF16
// `fp4_paged_mqa_logits_bf16`): int32 [rows, 1024], bins in descending score order (bin 0 holds the largest), counting
// every live non-NaN score once. |x| < 16: FP16-RN exponent and 4 mantissa bits; 16 <= |x| < 223: unit-width bins,
// lower-inclusive for negatives; one saturating bin at each end. -0 counts as +0.
constexpr uint32_t kNumCoarseHistogramBins = 1024;

// 1024-bin coarse key of a live, non-NaN FP32 score
CUTLASS_DEVICE uint32_t coarse_histogram_bin(float score) {
    // Preserve exact FP16-RN bins below the boundary near 16.
    // Code 304 joins that boundary to 17; later bins have width one.
    const uint32_t bits = __float_as_uint(score);
    uint32_t magnitude = bits & 0x7fffffffu;
    const bool negative = (bits >> 31) && magnitude != 0;
    int code = (__half_as_ushort(__float2half_rn(score)) & 0x7fffu) >> 6;
    if (code >= 304) {
        magnitude -= negative;  // lower-inclusive negative unit bins
        const float bounded = __uint_as_float(min(magnitude, 0x435f0000u));
        code = max(304, __float2int_rd(bounded) + 288);
    }
    return negative ? 512 + code : 511 - code;
}

// The same key of a non-NaN BF16 score. For 2^-14 <= |x| < 16 (BF16 exponent 113..130) the FP16 conversion is exact,
// so the FP16 code is the BF16 exponent and top mantissa bits: (|bits| >> 3) - (112 << 4)
CUTLASS_DEVICE uint32_t coarse_histogram_bin(const __nv_bfloat16& score) {
    const uint32_t bits = __bfloat16_as_ushort(score);
    const uint32_t magnitude = bits & 0x7fffu;
    if (magnitude - 0x3880u < 0x900u)
        return (bits & 0x8000u) ? (magnitude >> 3) - 1280u : 2303u - (magnitude >> 3);
    return coarse_histogram_bin(__bfloat162float(score));
}

// Empty default policy: no storage, initialization, cleanup or runtime branch.
struct NoHistogram {
    static constexpr bool kEnabled = false;
    static constexpr uint32_t kNumSlots = 1;
};

// Adds this thread's bins to a global row. Thread t owns the bin pairs (2t, 2t + 1) + i * 2 * kMathThreads: one warp
// add covers two full 128B lines, and a packed 64-bit add is exact while every bin stays a non-negative int32 (no carry
// between the lanes), which the zero-at-entry contract guarantees. The loads are issued back to back, so a flush costs one shared-memory round trip.
template <uint32_t kBins, uint32_t kMathThreads, bool kClear>
CUTLASS_DEVICE void flush_coarse_histogram(int* local, int* global, const int& row) {
    constexpr uint32_t kPairsPerThread = kBins / (2 * kMathThreads);
    uint2 counts[kPairsPerThread];
    #pragma unroll
    for (uint32_t i = 0; i < kPairsPerThread; ++ i)
        counts[i] = *reinterpret_cast<const uint2*>(local + 2 * threadIdx.x + i * 2 * kMathThreads);
    int* row_bins = global + static_cast<uint64_t>(row) * kBins + 2 * threadIdx.x;
    #pragma unroll
    for (uint32_t i = 0; i < kPairsPerThread; ++ i) {
        if ((counts[i].x | counts[i].y) != 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(row_bins + i * 2 * kMathThreads),
                      static_cast<unsigned long long>(counts[i].x) | (static_cast<unsigned long long>(counts[i].y) << 32));
            if constexpr (kClear)
                *reinterpret_cast<uint2*>(local + 2 * threadIdx.x + i * 2 * kMathThreads) = make_uint2(0u, 0u);
        }
    }
}

// A CTA moves to its next request: publish the finished rows (one slot per token) and clear them
template <uint32_t kBins, uint32_t kMathThreads, uint32_t kSlots>
__device__ __noinline__ void switch_coarse_histogram_row(int* local, int* global, int row) {
    cutlass::arch::NamedBarrier(kMathThreads, 1).sync();
    if (row >= 0) {
        #pragma unroll
        for (uint32_t slot = 0; slot < kSlots; ++ slot)
            flush_coarse_histogram<kBins, kMathThreads, true>(local + slot * kBins, global, row + static_cast<int>(slot));
    }
    cutlass::arch::NamedBarrier(kMathThreads, 1).sync();
}

// Two scores of one column packed for the verify fast paths. Exactness needs denormals preserved (DeepGEMM JIT does not
// pass -ftz): adding +0.0 turns -0.0 into +0.0 and keeps every other sign, so a sign bit equals the mapping's `negative`.
struct CoarseHistogramPair {
    float2 value;    // canonical scores
    uint32_t half;   // FP16-RN of both, low half = value.x
    uint32_t big;    // bit 15 / 31: |half| >= 16 (unit-width bins, or inf / NaN)
    uint32_t huge;   // bit 15 / 31: |half| >= 1024 (inf / NaN included), left to the exact path

    CoarseHistogramPair() = default;

    CUTLASS_DEVICE CoarseHistogramPair(const float& a, const float& b) {
        value = __fadd2_rn(make_float2(a, b), make_float2(0.0f, 0.0f));
        const __half2 packed = __floats2half2_rn(value.x, value.y);
        half = *reinterpret_cast<const uint32_t*>(&packed);
        const uint32_t magnitude = half & 0x7fff7fffu;
        big = (magnitude + 0x34003400u) & 0x80008000u;
        huge = (magnitude + 0x1c001c00u) & 0x80008000u;
    }

    // |x| < 16: bin = code ^ (negative ? 512 : 511) on the FP16 code; byte offsets, low half = value.x
    CUTLASS_DEVICE uint32_t fine_offsets() const {
        return (((half >> 4) & 0x07fc07fcu) ^ 0x07fc07fcu) ^ (((half >> 15) & 0x00010001u) * 0xffcu);
    }

    // 16 <= |x| < 1024: the lower-inclusive unit bin is floor(x) ^ sign(x), since ~floor(x) = ceil(|x|) - 1 for x < 0;
    // floor comes from one round-down add of 1.5 * 2^23 for both scores
    CUTLASS_DEVICE void unit_offsets(uint32_t& low, uint32_t& high) const {
        uint64_t in, out;
        asm("mov.b64 %0, {%1, %2};" : "=l"(in) : "f"(value.x), "f"(value.y));
        asm("add.rm.f32x2 %0, %1, %2;" : "=l"(out) : "l"(in), "l"(0x4b4000004b400000ull));
        low = unit_offset(static_cast<uint32_t>(out));
        high = unit_offset(static_cast<uint32_t>(out >> 32));
    }

    // Both unit offsets packed like fine_offsets(); valid for halves with 16 <= |x| < 1024
    CUTLASS_DEVICE uint32_t packed_unit_offsets() const {
        uint32_t low, high;
        unit_offsets(low, high);
        return low | (high << 16);
    }

    CUTLASS_DEVICE static uint32_t unit_offset(uint32_t magic_bits) {
        const int32_t floor_x = static_cast<int32_t>(magic_bits - 0x4b400000u);
        const int32_t sign = floor_x >> 31;
        const uint32_t unit = min(max(static_cast<uint32_t>(floor_x ^ sign), 16u), 223u);
        return ((unit + 288u) << 2) ^ ((511u - static_cast<uint32_t>(sign)) << 2);
    }
};

// Counts only live non-NaN scores. Caller owns zero-at-entry/consumer-reset of global bins.
// kSlots > 1: the kSlots tokens of one request (a verify step) share every KV split, so each token keeps its own
// bins and the CTA only publishes when it moves to the next request. `current_row` is the request's first row.
template <uint32_t kBins, uint32_t kMathThreads, uint32_t kSlots = 1>
struct CoarseHistogram {
    static constexpr uint32_t kNumSlots = kSlots;
    static constexpr bool kEnabled = true;
    static_assert(kBins == 1024 and kBins % (2 * kMathThreads) == 0 and kSlots >= 1);
    int* global;
    const uint32_t* lengths;
    uint32_t logits_stride;
    const uint32_t* schedule_meta;
    uint32_t num_rows;
    int* local = nullptr;
    int current_row = -1;
    uint32_t current_length[kSlots] = {};
    uint32_t min_limit = 0;  // columns below it are live for every token of the request

    CUTLASS_DEVICE void load_lengths(uint32_t first_row) {
        min_limit = logits_stride;
        #pragma unroll
        for (uint32_t slot = 0; slot < kSlots; ++ slot) {
            current_length[slot] = lengths[first_row + slot];
            min_limit = min(min_limit, current_length[slot]);
        }
    }

    // Request ownership changes only at Q-block boundaries, never inside the KV split loop.
    CUTLASS_DEVICE void prepare(uint32_t first_row) {
        if (static_cast<int>(first_row) != current_row) {
            switch_coarse_histogram_row<kBins, kMathThreads, kSlots>(local, global, current_row);
            current_row = static_cast<int>(first_row);
            load_lengths(first_row);
        }
    }

    // Called after prepare(): a verify step's scores of one column, token t in scores[t].
    CUTLASS_DEVICE void observe_split(uint32_t column, const float (&scores)[kSlots]) {
        static_assert(kSlots > 1);
        constexpr uint32_t kPairs = (kSlots + 1) / 2;
        // The last pair of an odd slot count carries a dummy high half: it never blocks either fast path
        constexpr uint32_t kLastMask = kSlots % 2 == 0 ? 0x80008000u : 0x00008000u;
        CoarseHistogramPair pairs[kPairs] = {};
        uint32_t big_any = 0, big_all = 0x80008000u, huge_any = 0;
        #pragma unroll
        for (uint32_t pair = 0; pair < kPairs; ++ pair) {
            const bool has_high = 2 * pair + 1 < kSlots;
            pairs[pair] = CoarseHistogramPair(scores[2 * pair], has_high ? scores[2 * pair + 1] : 0.0f);
            const uint32_t mask = pair + 1 == kPairs ? kLastMask : 0x80008000u;
            big_any |= pairs[pair].big & mask;
            big_all &= pairs[pair].big | ~mask;
            huge_any |= pairs[pair].huge & mask;
        }
        auto* bytes = reinterpret_cast<uint8_t*>(local);
        // `predicated`: a token whose length the column has passed skips its add
        const auto add_pair = [&](auto predicated, uint32_t pair, uint32_t offsets) {
            if (not predicated or column < current_length[2 * pair])
                atomicAdd(reinterpret_cast<int*>(bytes + (2 * pair) * kBins * sizeof(int) + (offsets & 0xffffu)), 1);
            if (2 * pair + 1 < kSlots and (not predicated or column < current_length[2 * pair + 1]))
                atomicAdd(reinterpret_cast<int*>(bytes + (2 * pair + 1) * kBins * sizeof(int) + (offsets >> 16)), 1);
        };
        const auto add_fine = [&](auto predicated) {
            #pragma unroll
            for (uint32_t pair = 0; pair < kPairs; ++ pair)
                add_pair(predicated, pair, pairs[pair].fine_offsets());
        };
        const auto add_unit = [&](auto predicated) {
            #pragma unroll
            for (uint32_t pair = 0; pair < kPairs; ++ pair)
                add_pair(predicated, pair, pairs[pair].packed_unit_offsets());
        };
        // Mixed magnitudes: both offsets per half, selected by the half's |x| >= 16 flag
        const auto add_mixed = [&](auto predicated) {
            #pragma unroll
            for (uint32_t pair = 0; pair < kPairs; ++ pair) {
                const uint32_t unit_half = (pairs[pair].big >> 15) * 0xffffu;  // bit 15 -> 0xffff, bit 31 -> 0xffff0000
                add_pair(predicated, pair, (pairs[pair].fine_offsets() & ~unit_half) | (pairs[pair].packed_unit_offsets() & unit_half));
            }
        };
        // Paths are chosen per warp, so a lane with other scores never makes the whole warp run two paths. A column
        // past some token's length (or a huge score) leaves the unpredicated paths; only huge scores need the exact one.
        const bool exact = column >= min_limit or huge_any != 0;
        const bool fine = __all_sync(0xffffffffu, big_any == 0);
        const bool unit = __all_sync(0xffffffffu, big_all == 0x80008000u);
        const bool any_exact = __any_sync(0xffffffffu, exact);
        if (fine and not any_exact) {
            add_fine(std::false_type{});
        } else if (unit and not any_exact) {
            add_unit(std::false_type{});
        } else if (not any_exact) {
            add_mixed(std::false_type{});
        } else if (__any_sync(0xffffffffu, huge_any != 0)) {
            #pragma unroll
            for (uint32_t slot = 0; slot < kSlots; ++ slot) {
                const float score = scores[slot];
                if (column < current_length[slot] and column < logits_stride and not isnan(score))
                    atomicAdd(local + slot * kBins + coarse_histogram_bin(score), 1);
            }
        } else if (fine) {
            add_fine(std::true_type{});
        } else if (unit) {
            add_unit(std::true_type{});
        } else {
            add_mixed(std::true_type{});
        }
    }

    CUTLASS_DEVICE void initialize(int* scratch, uint32_t math_thread_idx) {
        local = scratch;
        // The CTA starts at its scheduled token, which is its first row: load the lengths off the score path
        const uint32_t first_row = schedule_meta[blockIdx.x * 2];
        if (first_row < num_rows) {
            current_row = static_cast<int>(first_row);
            load_lengths(first_row);
        }
        for (uint32_t bin = math_thread_idx; bin < kSlots * kBins; bin += kMathThreads)
            local[bin] = 0;
        cutlass::arch::NamedBarrier(kMathThreads, 1).sync();
    }

    // `token` is the row's index inside its Q block (compile-time in the caller's unrolled token loop)
    CUTLASS_DEVICE void observe(uint32_t row, uint32_t column, float score, uint32_t token) {
        const uint32_t slot = kSlots == 1 ? 0 : token;
        const int first_row = static_cast<int>(row - slot);
        // All math threads visit the same rows, including masked tail lanes.
        if (first_row != current_row) {
            switch_coarse_histogram_row<kBins, kMathThreads, kSlots>(local, global, current_row);
            current_row = first_row;
            load_lengths(row - slot);
        }
        if (column < current_length[slot] and column < logits_stride and not isnan(score))
            atomicAdd(local + slot * kBins + coarse_histogram_bin(score), 1);
    }

    // Called after the math threads' final barrier: every shared atomic is visible and the bins die with the CTA
    CUTLASS_DEVICE void finish() {
        if (current_row >= 0) {
            #pragma unroll
            for (uint32_t slot = 0; slot < kSlots; ++ slot)
                flush_coarse_histogram<kBins, kMathThreads, false>(local + slot * kBins, global, current_row + static_cast<int>(slot));
        }
    }
};

}  // namespace deep_gemm::epilogue
