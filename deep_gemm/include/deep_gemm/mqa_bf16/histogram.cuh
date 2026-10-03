#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>

namespace deep_gemm::mqa_bf16::epilogue {

// Empty default policy: no storage, initialization, cleanup or runtime branch.
struct NoHistogram {
    static constexpr bool kEnabled = false;
};

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

// `coarse_histogram_bin` of a non-NaN BF16 score taken as FP32. For 2^-14 <= |x| < 16 (BF16 exponent 113..130) the FP16
// conversion is exact, so the FP16 code is the BF16 exponent and top mantissa bits: (|bits| >> 3) - (112 << 4)
CUTLASS_DEVICE uint32_t coarse_histogram_bin(const __nv_bfloat16& score) {
    const uint32_t bits = __bfloat16_as_ushort(score);
    const uint32_t magnitude = bits & 0x7fffu;
    if (magnitude - 0x3880u < 0x900u)
        return (bits & 0x8000u) ? (magnitude >> 3) - 1280u : 2303u - (magnitude >> 3);
    return coarse_histogram_bin(__bfloat162float(score));
}

namespace coarse_histogram_detail {

CUTLASS_DEVICE uint32_t prmt(const uint32_t& a, const uint32_t& b, const uint32_t& selector) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(selector));
    return d;
}

// Shared-memory atomic add without return (`red`); ordered with the named barriers (volatile asm keeps program order)
CUTLASS_DEVICE void red_add(const uint32_t& smem_addr, const uint32_t& value) {
    asm volatile("red.shared.add.u32 [%0], %1;" :: "r"(smem_addr), "r"(value));
}

}  // namespace coarse_histogram_detail

// Counts the live (stored) non-NaN BF16 scores of every logits row into int32 `global[rows, 1024]` (added; the caller
// zeroes it). Shared memory keeps a window of kWindowRows consecutive rows as 16-bit counters, four rows per 8-byte cell
// (cell = bin, row r of a group at bytes 2 * (r % 4)), groups of four rows 8 KiB apart. A CTA's tasks visit rows in
// increasing order and the Q blocks of one request alternate every chunk, so the window publishes (named barrier 1 of
// the math threads, packed 64-bit global atomics, clear) only when a task leaves it, every kMaxSplits splits (no 16-bit
// counter can wrap: a split adds at most one score per math thread to a row) and at exit.
// Per split the tokens are counted in pairs: the two BF16 scores share one 32-bit register, and for 2^-14 <= |x| < 16
// (the whole range of real indexer scores) the byte offset 8 * bin + row offset of both is computed with packed integer
// operations (8 * bin = 0x47f8 - (bits & 0xfff8) for positive, (bits & 0xfff8) - 0xa800 for negative scores); a split
// where some lane of the warp holds another value (NaN, +-Inf, 0, |x| < 2^-14 or >= 16) takes the exact scalar path.
template <uint32_t kMathThreads, uint32_t kBlockQ, uint32_t kWindowRows, bool kSwizzle = false>
struct CoarseHistogram {
    static constexpr bool kEnabled = true;
    static constexpr uint32_t kBins = 1024;
    static constexpr uint32_t kNumGroups = kWindowRows / 4;
    static constexpr uint32_t kGroupBytes = kBins * 8;
    static constexpr uint32_t kNumSmemBytes = kNumGroups * kGroupBytes;
    static constexpr uint32_t kMaxSplits = 0xffffu / kMathThreads;
    static constexpr uint32_t kNumPairs = (kBlockQ + 1) / 2;
    DG_STATIC_ASSERT(kWindowRows % 4 == 0 and kWindowRows >= kBlockQ and kMaxSplits >= 1, "Invalid coarse histogram");
    DG_STATIC_ASSERT(kNumGroups * kGroupBytes <= 0x10000, "Packed row offsets must fit 16 bits");

    int* global;
    uint32_t num_rows;
    uint32_t smem = 0;              // shared address of the window
    uint32_t thread_idx = 0;
    uint32_t window_row = 0;        // row of slot 0
    uint32_t num_used_slots = 0;    // rows [window_row, window_row + num_used_slots) hold counts
    uint32_t num_splits = 0;        // splits counted since the last publish
    uint32_t task_slot = 0;         // slot of the current Q block's first token
    uint32_t value_even = 1, value_odd = 1u << 16;     // 16-bit counter increment of the task's even / odd tokens
    uint32_t pair_offsets[kNumPairs] = {};             // packed (slot byte offset + 0x47f8) of tokens 2k, 2k + 1

    CUTLASS_DEVICE static uint32_t swizzle(const uint32_t& offset) {
        return kSwizzle ? offset ^ ((offset >> 5) & 4u) : offset;
    }

    // Byte offset of a slot's counter word inside the window
    CUTLASS_DEVICE static uint32_t slot_offset(const uint32_t& slot) {
        return (slot / 4) * kGroupBytes + ((slot / 2) % 2) * 4;
    }

    CUTLASS_DEVICE void initialize(void* scratch, const uint32_t& math_thread_idx) {
        smem = static_cast<uint32_t>(__cvta_generic_to_shared(scratch));
        thread_idx = math_thread_idx;
        auto words = static_cast<uint4*>(scratch);
        for (uint32_t i = math_thread_idx; i < kNumSmemBytes / 16; i += kMathThreads)
            words[i] = make_uint4(0u, 0u, 0u, 0u);
        cutlass::arch::NamedBarrier(kMathThreads, 1).sync();
    }

    // Adds the window's rows to the global bins (thread t: bin pairs t + i * kMathThreads of every used group)
    template <bool kClear>
    CUTLASS_DEVICE void flush() {
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
                if ((cells.x | cells.y | cells.z | cells.w) == 0)
                    continue;
                if (kSwizzle and (pair & 8u)) {
                    uint32_t tmp = cells.x; cells.x = cells.y; cells.y = tmp;
                    tmp = cells.z; cells.z = cells.w; cells.w = tmp;
                }
                const uint32_t lo[4] = {cells.x & 0xffffu, cells.x >> 16, cells.y & 0xffffu, cells.y >> 16};
                const uint32_t hi[4] = {cells.z & 0xffffu, cells.z >> 16, cells.w & 0xffffu, cells.w >> 16};
                #pragma unroll
                for (uint32_t j = 0; j < 4; ++ j) {
                    const uint32_t slot = group * 4 + j;
                    if ((lo[j] | hi[j]) != 0 and slot < num_used_slots and window_row + slot < num_rows) {
                        atomicAdd(reinterpret_cast<unsigned long long*>(
                                      global + static_cast<uint64_t>(window_row + slot) * kBins + 2 * pair),
                                  static_cast<unsigned long long>(lo[j]) | (static_cast<unsigned long long>(hi[j]) << 32));
                    }
                }
                if constexpr (kClear)
                    asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" :: "r"(addr), "r"(0u));
            }
        }
    }

    // Publishes and clears the window; every math thread calls it
    CUTLASS_DEVICE void publish() {
        cutlass::arch::NamedBarrier(kMathThreads, 1).sync();
        flush<true>();
        cutlass::arch::NamedBarrier(kMathThreads, 1).sync();
        num_splits = 0;
    }

    // At every task (Q-block ownership changes only between tasks): `num_slots` = rows the task touches from `first_row`
    CUTLASS_DEVICE void prepare(const uint32_t& first_row, const uint32_t& num_slots) {
        if (num_used_slots > 0 and (first_row < window_row or first_row + num_slots > window_row + kWindowRows)) {
            publish();
            num_used_slots = 0;
        }
        if (num_used_slots == 0)
            window_row = first_row;
        task_slot = first_row - window_row;
        num_used_slots = cute::max(num_used_slots, task_slot + num_slots);
        value_even = 1u << ((task_slot % 2) * 16);
        value_odd = 0x10001u - value_even;
        #pragma unroll
        for (uint32_t k = 0; k < kNumPairs; ++ k) {
            pair_offsets[k] = ((slot_offset(task_slot + 2 * k + 1) + 0x47f8u) << 16) + slot_offset(task_slot + 2 * k) + 0x47f8u;
        }
    }

    // One split: `scores[i]` = token i's stored BF16 result (both halves), stored iff seq_k_start[i] <= kv_offset < seq_k_end[i]
    template <uint32_t kNumTokens>
    CUTLASS_DEVICE void observe(const nv_bfloat162 (&scores)[kNumTokens], const uint32_t& kv_offset,
                                const uint32_t* seq_k_start, const uint32_t* seq_k_end) {
        using namespace coarse_histogram_detail;
        constexpr uint32_t kNumTokenPairs = (kNumTokens + 1) / 2;
        DG_STATIC_ASSERT(kNumTokens <= kBlockQ, "Too many tokens");

        // A lane takes the packed path when every token's column is stored (loop invariant bounds)
        uint32_t max_start = seq_k_start[0], min_end = seq_k_end[0];
        #pragma unroll
        for (uint32_t i = 1; i < kNumTokens; ++ i)
            max_start = cute::max(max_start, seq_k_start[i]), min_end = cute::min(min_end, seq_k_end[i]);

        // Pair k = tokens 2k (low half) and 2k + 1 (high half; the last token of an odd count is duplicated)
        uint32_t bits[kNumTokenPairs];
        uint32_t in_range = 0x80008000u;
        #pragma unroll
        for (uint32_t k = 0; k < kNumTokenPairs; ++ k) {
            const uint32_t lo_bits = *reinterpret_cast<const uint32_t*>(&scores[2 * k]);
            bits[k] = 2 * k + 1 < kNumTokens
                ? prmt(lo_bits, *reinterpret_cast<const uint32_t*>(&scores[2 * k + 1]), 0x5410u) : lo_bits;
            // Bit 15 / 31: 0x3880 <= |bits| < 0x4180, i.e. 2^-14 <= |x| < 16 (false for NaN; no carry between the halves)
            const uint32_t magnitude = bits[k] & 0x7fff7fffu;
            in_range &= (magnitude + 0x47804780u) & ~(magnitude + 0x3e803e80u);
        }

        const bool fast = in_range == 0x80008000u and max_start <= kv_offset and kv_offset < min_end;
        if (__all_sync(0xffffffffu, fast)) {
            #pragma unroll
            for (uint32_t k = 0; k < kNumTokenPairs; ++ k) {
                // Per half, with x = bits & 0xfff8 and the sign mask m: (m & 0x1007) - (x ^ m) + 0x47f8 = 0x47f8 - x
                // (positive) or x - 0xa800 (negative) = 8 * bin; plus the slot's byte offset (`pair_offsets`). Both halves
                // stay in [0, 2^16) for in-range scores, so the packed 32-bit arithmetic is exact
                const uint32_t negative = prmt(bits[k], 0u, 0xbb99u);
                uint32_t offsets = (negative & 0x10071007u) - ((bits[k] & 0xfff8fff8u) ^ negative) + pair_offsets[k];
                if constexpr (kSwizzle)
                    offsets ^= (offsets >> 5) & 0x00040004u;
                red_add(smem + (offsets & 0xffffu), value_even);
                if (2 * k + 1 < kNumTokens)
                    red_add(smem + (offsets >> 16), value_odd);
            }
        } else {
            #pragma unroll
            for (uint32_t i = 0; i < kNumTokens; ++ i) {
                const auto score = __ushort_as_bfloat16(static_cast<unsigned short>(bits[i / 2] >> (16 * (i % 2))));
                if (seq_k_start[i] <= kv_offset and kv_offset < seq_k_end[i] and not __hisnan(score))
                    red_add(smem + swizzle(slot_offset(task_slot + i) + coarse_histogram_bin(score) * 8),
                            i % 2 == 0 ? value_even : value_odd);
            }
        }
    }

    // After every KV split: publish before a 16-bit counter could wrap
    CUTLASS_DEVICE void end_split() {
        if (++ num_splits == kMaxSplits)
            publish();
    }

    // Called after the math threads' final barrier: every shared atomic is visible and the bins die with the CTA
    CUTLASS_DEVICE void finish() {
        flush<false>();
    }
};

}  // namespace deep_gemm::mqa_bf16::epilogue
