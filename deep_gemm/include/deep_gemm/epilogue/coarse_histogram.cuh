#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>

namespace deep_gemm::epilogue {

// Optional coarse score histogram of the SM100 paged MQA logits (`fp8_fp4_paged_mqa_logits(..., histogram=)`):
// int32 [rows, 1024], bins in descending score order (bin 0 holds the largest), counting every stored non-NaN score
// once. |x| < 16: FP16-RN exponent and 4 mantissa bits; 16 <= |x| < 223: unit-width bins, lower-inclusive for
// negatives; one saturating bin at each end. -0 counts as +0.
constexpr uint32_t kNumCoarseHistogramBins = 1024;

// 1024-bin coarse key of a non-NaN FP32 score (the definition)
CUTLASS_DEVICE uint32_t coarse_histogram_bin(float score) {
    // Preserve exact FP16-RN bins below the boundary near 16.
    // Code 304 joins that boundary to 17; later bins have width one.
    const uint32_t bits = __float_as_uint(score);
    uint32_t magnitude = bits & 0x7fffffffu;
    const bool negative = (bits >> 31) and magnitude != 0;
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

// Empty default policy: no storage, initialization, cleanup or runtime branch
struct NoHistogram {
    static constexpr bool kEnabled = false;
    static constexpr uint32_t kMaxTaskSplits = 0xffffffffu;  // Tasks are not capped
};

// Two adjacent int32 bins (even bin in the low word) as one 64-bit addend. The add is exact while every bin stays a
// non-negative int32 (no carry between the words), which the zero-at-entry contract guarantees.
CUTLASS_DEVICE unsigned long long pack_bin_pair(const uint32_t& even, const uint32_t& odd) {
    return static_cast<unsigned long long>(even) | (static_cast<unsigned long long>(odd) << 32);
}

namespace coarse_histogram_detail {

CUTLASS_DEVICE uint32_t prmt(const uint32_t& a, const uint32_t& b, const uint32_t& selector) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(selector));
    return d;
}

CUTLASS_DEVICE uint32_t min_u16x2(const uint32_t& a, const uint32_t& b) {
    uint32_t d;
    asm("min.u16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

// Both FP32 halves rounded toward -inf after adding `magic` (one packed add)
CUTLASS_DEVICE void add_rm_f32x2(uint32_t& lo, uint32_t& hi, const uint32_t& magic) {
    asm("{\n"
        ".reg .b64 x, m;\n"
        "mov.b64 x, {%0, %1};\n"
        "mov.b64 m, {%2, %2};\n"
        "add.rm.f32x2 x, x, m;\n"
        "mov.b64 {%0, %1}, x;\n"
        "}" : "+r"(lo), "+r"(hi) : "r"(magic));
}

// Shared-memory atomic add without return (`red`); ordered with the named barriers (volatile asm keeps program order)
CUTLASS_DEVICE void red_add(const uint32_t& smem_addr, const uint32_t& value) {
    asm volatile("red.shared.add.u32 [%0], %1;" :: "r"(smem_addr), "r"(value));
}

}  // namespace coarse_histogram_detail

// Bins of two BF16 scores packed in one 32-bit register (low half: token 2k, high half: token 2k + 1), computed for
// both halves at once. Each in-range half has a 16-bit key (sign bit included) with 8 * bin = 0x47f8 - key (positive)
// or key - 0xa800 (negative):
// - fine, 2^-14 <= |x| < 16: the key is the score bits & 0xfff8;
// - unit, 16 <= |x| < inf (and -inf): the key is 8 * (2080 + u) | sign with u = min(floor(|x| - [x < 0] ulp), 223), so
//   8 * bin = 8 * (223 - u) (positive) or 8 * (800 + u) (negative).
// `fine_key` / `unit_key` return the keys XOR the negative mask, and `offsets` maps them to 8 * bin + the half's slot
// offset (`slot_offsets` holds slot offset + 0x47f8 per half). Every half stays in [0, 2^16) for in-range scores, so the
// packed 32-bit arithmetic is exact (verified over all pairs of BF16 values). NaN, +-0, +inf, |x| < 2^-14 and -16 are
// in neither range.
struct CoarseHistogramPairBins {
    // 0xffff in the halves holding a negative score
    CUTLASS_DEVICE static uint32_t negative(const uint32_t& bits) {
        return coarse_histogram_detail::prmt(bits, 0u, 0xbb99u);
    }

    // Bit 15 / 31: 0x3880 <= |bits| < 0x4180, i.e. 2^-14 <= |x| < 16 (false for NaN; no carry between the halves)
    CUTLASS_DEVICE static uint32_t fine_range(const uint32_t& bits) {
        const uint32_t magnitude = bits & 0x7fff7fffu;
        return (magnitude + 0x47804780u) & ~(magnitude + 0x3e803e80u) & 0x80008000u;
    }

    // |bits| minus one for negative halves: the next lower BF16 magnitude, so floor() gives the lower-inclusive
    // negative bins. A borrow out of the low half happens only for -0, which is in no range
    CUTLASS_DEVICE static uint32_t unit_magnitude(const uint32_t& bits, const uint32_t& negative) {
        return bits - (negative & 0x80018001u);
    }

    // Bit 15 / 31: 0x4180 <= unit magnitude < 0x7f80, i.e. 16 <= |x| < inf and -inf (false for NaN)
    CUTLASS_DEVICE static uint32_t unit_range(const uint32_t& magnitude) {
        return (magnitude + 0x3e803e80u) & ~(magnitude + 0x00800080u) & 0x80008000u;
    }

    CUTLASS_DEVICE static uint32_t fine_key(const uint32_t& bits, const uint32_t& negative) {
        return (bits & 0xfff8fff8u) ^ negative;
    }

    // u from one packed FP32 add of 1.5 * 2^23 + 2080 rounding toward -inf (exact for these integers): its low 16 bits
    // are 2080 + floor(min(unit magnitude, 223))
    CUTLASS_DEVICE static uint32_t unit_key(const uint32_t& magnitude, const uint32_t& negative) {
        using namespace coarse_histogram_detail;
        const uint32_t clamped = min_u16x2(magnitude, 0x435f435fu);
        uint32_t lo = clamped << 16, hi = clamped & 0xffff0000u;
        add_rm_f32x2(lo, hi, 0x4b400820u);
        return (prmt(lo, hi, 0x5410u) << 3) ^ (negative & 0x7fff7fffu);
    }

    CUTLASS_DEVICE static uint32_t offsets(const uint32_t& flipped_key, const uint32_t& negative,
                                           const uint32_t& slot_offsets) {
        return (negative & 0x10071007u) - flipped_key + slot_offsets;
    }
};

// Counts the stored non-NaN BF16 scores of every logits row into int32 `global[rows, 1024]` (added; the caller zeroes
// it). Shared memory keeps a window of two Q blocks of consecutive rows as 16-bit counters, four rows per 8-byte cell
// (cell = bin, row r of a group at bytes 2 * (r % 4)), groups of four rows 8 KiB apart. A CTA's tasks visit rows in
// increasing order and the Q blocks of one request alternate every chunk, so the window publishes (named barrier 1 of
// the math threads, packed 64-bit global atomics, clear) only at a task that leaves it or would take the splits counted
// since the last publish past kMaxTaskSplits (no 16-bit counter can wrap: a split adds at most one score per math
// thread to a row; the scheduler caps tasks at kMaxTaskSplits splits), and at exit.
// Per split the tokens are counted in pairs (`CoarseHistogramPairBins`) on a warp-uniform path: all scores of the warp
// fine, all unit, a mix of the two (per-half select), or else the exact scalar path (NaN, +-0, +inf, tiny scores, -16,
// and columns some token does not store).
template <uint32_t kMathThreads, uint32_t kBlockQ>
struct RowWindowHistogram {
    static constexpr bool kEnabled = true;
    static constexpr uint32_t kBins = kNumCoarseHistogramBins;
    static constexpr uint32_t kWindowRows = 2 * kBlockQ;
    static constexpr uint32_t kNumGroups = kWindowRows / 4;
    static constexpr uint32_t kGroupBytes = kBins * 8;
    static constexpr uint32_t kNumSmemBytes = kNumGroups * kGroupBytes;
    static constexpr uint32_t kMaxTaskSplits = 0xffffu / kMathThreads;
    static constexpr uint32_t kNumPairs = (kBlockQ + 1) / 2;
    DG_STATIC_ASSERT(kWindowRows % 4 == 0, "The window holds whole groups of four rows");

    int* global;
    uint32_t smem = 0;              // shared address of the window
    uint32_t window_row = 0;        // row of slot 0
    uint32_t num_used_slots = 0;    // rows [window_row, window_row + num_used_slots) hold counts
    uint32_t num_splits = 0;        // splits counted since the last publish
    uint32_t value_even = 1;        // 16-bit counter increment of the task's even tokens (odd: `value_even ^ 0x10001`)
    uint32_t slot_offsets[kNumPairs] = {};  // packed (slot byte offset + 0x47f8) of tokens 2k, 2k + 1

    // Byte offset of a slot's counter word inside the window
    CUTLASS_HOST_DEVICE static constexpr uint32_t slot_offset(const uint32_t& slot) {
        return (slot / 4) * kGroupBytes + ((slot / 2) % 2) * 4;
    }
    DG_STATIC_ASSERT(slot_offset(kWindowRows - 1) + 0x47f8u < 0x10000u, "Packed slot offsets must fit 16 bits");

    CUTLASS_DEVICE void initialize(void* scratch, const uint32_t& math_thread_idx) {
        smem = static_cast<uint32_t>(__cvta_generic_to_shared(scratch));
        auto words = static_cast<uint4*>(scratch);
        for (uint32_t i = math_thread_idx; i < kNumSmemBytes / 16; i += kMathThreads)
            words[i] = make_uint4(0u, 0u, 0u, 0u);
        cutlass::arch::NamedBarrier(kMathThreads, 1).sync();
    }

    // Adds the window's rows to the global bins (thread t: bin pairs t + i * kMathThreads of every used group). The math
    // threads are the CTA's first kMathThreads threads; reading the index here keeps it out of the main loop's registers
    template <bool kClear>
    CUTLASS_DEVICE void flush() {
        const uint32_t thread_idx = threadIdx.x;
        #pragma unroll
        for (uint32_t group = 0; group < kNumGroups; ++ group) {
            if (group * 4 >= num_used_slots)
                break;
            #pragma unroll
            for (uint32_t pair = thread_idx; pair < kBins / 2; pair += kMathThreads) {
                const uint32_t addr = smem + group * kGroupBytes + pair * 16;
                uint4 cells;
                asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(cells.x), "=r"(cells.y), "=r"(cells.z), "=r"(cells.w) : "r"(addr));
                if (cells.x == 0 and cells.y == 0 and cells.z == 0 and cells.w == 0)
                    continue;
                const uint32_t even[4] = {cells.x & 0xffffu, cells.x >> 16, cells.y & 0xffffu, cells.y >> 16};
                const uint32_t odd[4] = {cells.z & 0xffffu, cells.z >> 16, cells.w & 0xffffu, cells.w >> 16};
                #pragma unroll
                for (uint32_t j = 0; j < 4; ++ j) {
                    const uint32_t slot = group * 4 + j;
                    const unsigned long long counts = pack_bin_pair(even[j], odd[j]);
                    // Only the slots of task rows (all below the logits' row count) are ever non-zero
                    if (counts != 0) {
                        atomicAdd(reinterpret_cast<unsigned long long*>(
                                      global + static_cast<uint64_t>(window_row + slot) * kBins + 2 * pair),
                                  counts);
                    }
                }
                if constexpr (kClear)
                    asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" :: "r"(addr), "r"(0u));
            }
        }
    }

    // At every task (Q-block ownership changes only between tasks; every math thread calls it): `num_slots` = rows the
    // task touches from `first_row`, `num_task_splits` <= kMaxTaskSplits (the scheduler caps tasks). Publishing whenever
    // the splits counted since the last publish would exceed kMaxTaskSplits keeps every 16-bit counter from wrapping,
    // with no per-split check
    CUTLASS_DEVICE void prepare(const uint32_t& first_row, const uint32_t& num_slots, const uint32_t& num_task_splits) {
        if (num_used_slots > 0 and (first_row < window_row or first_row + num_slots > window_row + kWindowRows or
                                    num_splits + num_task_splits > kMaxTaskSplits)) {
            cutlass::arch::NamedBarrier(kMathThreads, 1).sync();
            flush<true>();
            cutlass::arch::NamedBarrier(kMathThreads, 1).sync();
            num_used_slots = num_splits = 0;
        }
        num_splits += num_task_splits;
        if (num_used_slots == 0)
            window_row = first_row;
        const uint32_t task_slot = first_row - window_row;
        num_used_slots = task_slot + num_slots > num_used_slots ? task_slot + num_slots : num_used_slots;
        value_even = 1u << ((task_slot % 2) * 16);
        #pragma unroll
        for (uint32_t k = 0; k < kNumPairs; ++ k)
            slot_offsets[k] = ((slot_offset(task_slot + 2 * k + 1) + 0x47f8u) << 16) +
                              slot_offset(task_slot + 2 * k) + 0x47f8u;
    }

    // Counts the two tokens of pair k at their packed byte offsets
    template <uint32_t kNumTokens>
    CUTLASS_DEVICE void count_pair(const uint32_t& k, const uint32_t& offsets) const {
        coarse_histogram_detail::red_add(smem + (offsets & 0xffffu), value_even);
        if (2 * k + 1 < kNumTokens)
            coarse_histogram_detail::red_add(smem + (offsets >> 16), value_even ^ 0x10001u);
    }

    // One split: `scores[i]` = token i's BF16 result (both halves), stored iff seq_k_start[i] <= kv_offset < seq_k_end[i]
    template <uint32_t kNumTokens>
    CUTLASS_DEVICE void observe(const nv_bfloat162 (&scores)[kNumTokens], const uint32_t& kv_offset,
                                const uint32_t* seq_k_start, const uint32_t* seq_k_end) {
        using Bins = CoarseHistogramPairBins;
        constexpr uint32_t kNumTokenPairs = (kNumTokens + 1) / 2;
        DG_STATIC_ASSERT(kNumTokens <= kBlockQ, "Too many tokens");

        // A lane takes a packed path only when every token stores its column
        bool stored = true;
        #pragma unroll
        for (uint32_t i = 0; i < kNumTokens; ++ i)
            stored = stored and seq_k_start[i] <= kv_offset and kv_offset < seq_k_end[i];

        // Pair k = tokens 2k (low half) and 2k + 1 (high half; the last token of an odd count is duplicated)
        uint32_t bits[kNumTokenPairs], negative[kNumTokenPairs], fine[kNumTokenPairs];
        uint32_t all_fine = 0x80008000u;
        #pragma unroll
        for (uint32_t k = 0; k < kNumTokenPairs; ++ k) {
            const uint32_t lo_bits = *reinterpret_cast<const uint32_t*>(&scores[2 * k]);
            bits[k] = 2 * k + 1 < kNumTokens
                ? coarse_histogram_detail::prmt(lo_bits, *reinterpret_cast<const uint32_t*>(&scores[2 * k + 1]), 0x5410u)
                : lo_bits;
            negative[k] = Bins::negative(bits[k]);
            fine[k] = Bins::fine_range(bits[k]);
            all_fine &= fine[k];
        }
        if (__all_sync(0xffffffffu, stored and all_fine == 0x80008000u)) {
            #pragma unroll
            for (uint32_t k = 0; k < kNumTokenPairs; ++ k)
                count_pair<kNumTokens>(k, Bins::offsets(Bins::fine_key(bits[k], negative[k]), negative[k],
                                                        slot_offsets[k]));
            return;
        }

        // Scores of 16 or more (e.g. GLM indexer layers past the first): unit bins, alone or mixed with fine ones
        uint32_t magnitude[kNumTokenPairs], unit[kNumTokenPairs];
        uint32_t all_unit = 0x80008000u, all_in_range = 0x80008000u;
        #pragma unroll
        for (uint32_t k = 0; k < kNumTokenPairs; ++ k) {
            magnitude[k] = Bins::unit_magnitude(bits[k], negative[k]);
            unit[k] = Bins::unit_range(magnitude[k]);
            all_unit &= unit[k];
            all_in_range &= unit[k] | fine[k];
        }
        if (__all_sync(0xffffffffu, stored and all_unit == 0x80008000u)) {
            #pragma unroll
            for (uint32_t k = 0; k < kNumTokenPairs; ++ k)
                count_pair<kNumTokens>(k, Bins::offsets(Bins::unit_key(magnitude[k], negative[k]), negative[k],
                                                        slot_offsets[k]));
        } else if (__all_sync(0xffffffffu, stored and all_in_range == 0x80008000u)) {
            #pragma unroll
            for (uint32_t k = 0; k < kNumTokenPairs; ++ k) {
                const uint32_t select_unit = coarse_histogram_detail::prmt(unit[k], 0u, 0xbb99u);
                const uint32_t key = (Bins::fine_key(bits[k], negative[k]) & ~select_unit) |
                                     (Bins::unit_key(magnitude[k], negative[k]) & select_unit);
                count_pair<kNumTokens>(k, Bins::offsets(key, negative[k], slot_offsets[k]));
            }
        } else {
            #pragma unroll
            for (uint32_t i = 0; i < kNumTokens; ++ i) {
                const uint32_t shift = 16 * (i % 2);
                const auto score = __ushort_as_bfloat16(static_cast<unsigned short>(bits[i / 2] >> shift));
                const uint32_t offset = ((slot_offsets[i / 2] >> shift) & 0xffffu) - 0x47f8u;
                if (seq_k_start[i] <= kv_offset and kv_offset < seq_k_end[i] and not __hisnan(score))
                    coarse_histogram_detail::red_add(smem + offset + coarse_histogram_bin(score) * 8,
                                                     i % 2 == 0 ? value_even : value_even ^ 0x10001u);
            }
        }
    }

    // Called after the math threads' final barrier: every shared atomic is visible and the bins die with the CTA
    CUTLASS_DEVICE void finish() {
        flush<false>();
    }
};

}  // namespace deep_gemm::epilogue
