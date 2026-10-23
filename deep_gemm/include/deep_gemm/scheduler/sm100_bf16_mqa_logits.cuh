#pragma once

#include <cutlass/arch/grid_dependency_control.h>

#include <deep_gemm/common/math.cuh>

// SM100 contiguous-KV scheduling via grid-stride traversal or per-SM metadata

namespace deep_gemm::sched {

// A Q block and its KV splits, shared by contiguous and paged schedulers
struct BF16MQALogitsTask {
    uint32_t q_token_base;      // First Q token and logits row
    uint32_t num_q_tokens;      // Valid tokens in the Q block (always BLOCK_Q for contiguous KV)
    uint32_t kv_token_base;     // KV base aligned to 4 tokens (SPLIT_KV for paged KV)
    uint32_t num_kv_splits;
    bool kv_shared_with_prev;   // Same KV as prior task; false for first task or contiguous KV
};

// KV range of one Q block: covers `[k_start, k_end)` of all its rows
struct BF16MQALogitsKVSpan {
    uint32_t kv_token_base;     // `min k_start` over the rows, rounded down to 4 tokens
    uint32_t num_kv_splits;     // `ceil((max k_end - kv_token_base) / SPLIT_KV)`

    CUTLASS_HOST_DEVICE BF16MQALogitsKVSpan() = default;

    CUTLASS_HOST_DEVICE BF16MQALogitsKVSpan(const uint32_t& kv_token_base, const uint32_t& num_kv_splits):
        kv_token_base(kv_token_base), num_kv_splits(num_kv_splits) {}
};

// Metadata int32 words: per-SM starts (2), split counts (1), then 8-byte-aligned spans
CUTLASS_HOST_DEVICE constexpr uint32_t get_bf16_mqa_logits_metadata_span_word_offset(const uint32_t& num_sms) {
    return (3 * num_sms + 1) / 2 * 2;
}

// NOTES: out-of-range rows are clamped to the last valid row
template <uint32_t BLOCK_Q, uint32_t SPLIT_KV>
CUTLASS_DEVICE BF16MQALogitsKVSpan get_bf16_mqa_logits_kv_span(const uint32_t& q_block_idx,
                                                      const uint32_t& num_q_tokens, const uint32_t& num_kv_tokens,
                                                      const uint32_t* cu_seq_len_k_start,
                                                      const uint32_t* cu_seq_len_k_end) {
    uint32_t start = cute::numeric_limits<uint32_t>::max();
    uint32_t end = cute::numeric_limits<uint32_t>::min();
    #pragma unroll 8
    for (uint32_t token_idx = 0; token_idx < BLOCK_Q; ++ token_idx) {
        const auto row_idx = cute::min(q_block_idx * BLOCK_Q + token_idx, num_q_tokens - 1);
        const auto k_start = cute::min(cu_seq_len_k_start[row_idx], num_kv_tokens);
        const auto k_end = cute::min(cu_seq_len_k_end[row_idx], num_kv_tokens);
        start = cute::min(start, k_start);
        end = cute::max(end, k_end);
    }
    const uint32_t kv_token_base = start / 4 * 4;
    return BF16MQALogitsKVSpan(kv_token_base, math::ceil_div(end - kv_token_base, SPLIT_KV));
}

// CTA-wide inclusive prefix sum of `values[0, num_items)` in shared memory
template <uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t kNumSMs, bool kUseSchedule = false>
struct SM100BF16MQALogitsScheduler {
    static constexpr bool kIsPaged = false;

    uint32_t num_q_blocks;
    uint32_t num_q_tokens;
    uint32_t num_kv_tokens;
    const uint32_t* cu_seq_len_k_start;
    const uint32_t* cu_seq_len_k_end;
    const BF16MQALogitsKVSpan* kv_spans = nullptr;      // Scheduled mode only

    // Q-block cursor, starting split and remaining scheduled work
    uint32_t current_q_block_idx;
    uint32_t current_split_offset = 0;
    uint32_t remaining_splits = 0;

    CUTLASS_DEVICE SM100BF16MQALogitsScheduler(const uint32_t& sm_idx,
                                           const uint32_t& num_q_tokens,
                                           const uint32_t& num_kv_tokens,
                                           const uint32_t* cu_seq_len_k_start,
                                           const uint32_t* cu_seq_len_k_end,
                                           const uint32_t* schedule_meta = nullptr):
            num_q_blocks(math::ceil_div(num_q_tokens, BLOCK_Q)),
            num_q_tokens(num_q_tokens), num_kv_tokens(num_kv_tokens),
            cu_seq_len_k_start(cu_seq_len_k_start), cu_seq_len_k_end(cu_seq_len_k_end),
            current_q_block_idx(sm_idx) {
        DG_STATIC_ASSERT(kNumSMs > 0, "Invalid SM count");
        if constexpr (kUseSchedule) {
            const auto start = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx];
            current_q_block_idx = start.x;
            current_split_offset = start.y;
            remaining_splits = schedule_meta[2 * kNumSMs + sm_idx];
            kv_spans = reinterpret_cast<const BF16MQALogitsKVSpan*>(
                schedule_meta + get_bf16_mqa_logits_metadata_span_word_offset(kNumSMs));
        }
    }

    // Emit the next Q-block task
    CUTLASS_DEVICE bool next_q_block(BF16MQALogitsTask& task) {
        return advance<false>(task, nullptr, nullptr);
    }

    // Also return clamped KV bounds per token; tail rows reuse the last valid row
    CUTLASS_DEVICE bool next_q_block(BF16MQALogitsTask& task, uint32_t* seq_k_start, uint32_t* seq_k_end) {
        return advance<true>(task, seq_k_start, seq_k_end);
    }

    template <bool kLoadSeqBounds>
    CUTLASS_DEVICE bool advance(BF16MQALogitsTask& task, uint32_t* seq_k_start, uint32_t* seq_k_end) {
        task.num_q_tokens = BLOCK_Q;
        task.kv_shared_with_prev = false;
        if constexpr (kUseSchedule) {
            while (remaining_splits > 0 and current_q_block_idx < num_q_blocks) {
                const auto span = kv_spans[current_q_block_idx];
                const auto split_offset = current_split_offset;
                current_split_offset = 0;
                if (split_offset < span.num_kv_splits) {
                    if constexpr (kLoadSeqBounds) {
                        #pragma unroll
                        for (uint32_t token_idx = 0; token_idx < BLOCK_Q; ++ token_idx) {
                            const auto row_idx = cute::min(current_q_block_idx * BLOCK_Q + token_idx, num_q_tokens - 1);
                            seq_k_start[token_idx] = cute::min(cu_seq_len_k_start[row_idx], num_kv_tokens);
                            seq_k_end[token_idx] = cute::min(cu_seq_len_k_end[row_idx], num_kv_tokens);
                        }
                    }
                    task.q_token_base = current_q_block_idx * BLOCK_Q;
                    task.kv_token_base = span.kv_token_base + split_offset * SPLIT_KV;
                    task.num_kv_splits = cute::min(span.num_kv_splits - split_offset, remaining_splits);
                    remaining_splits -= task.num_kv_splits;
                    ++ current_q_block_idx;
                    return true;
                }
                ++ current_q_block_idx;
            }
            return false;
        } else {
            if (current_q_block_idx >= num_q_blocks)
                return false;
            const auto q_block_idx = current_q_block_idx;
            current_q_block_idx += kNumSMs;
            task.q_token_base = q_block_idx * BLOCK_Q;
            if constexpr (kLoadSeqBounds) {
                // The bounds of the rows give the span, so load them only once
                uint32_t start = cute::numeric_limits<uint32_t>::max();
                uint32_t end = cute::numeric_limits<uint32_t>::min();
                #pragma unroll
                for (uint32_t token_idx = 0; token_idx < BLOCK_Q; ++ token_idx) {
                    const auto row_idx = cute::min(q_block_idx * BLOCK_Q + token_idx, num_q_tokens - 1);
                    const auto k_start = cute::min(cu_seq_len_k_start[row_idx], num_kv_tokens);
                    const auto k_end = cute::min(cu_seq_len_k_end[row_idx], num_kv_tokens);
                    seq_k_start[token_idx] = k_start;
                    seq_k_end[token_idx] = k_end;
                    start = cute::min(start, k_start);
                    end = cute::max(end, k_end);
                }
                task.kv_token_base = start / 4 * 4;
                task.num_kv_splits = math::ceil_div(end - task.kv_token_base, SPLIT_KV);
            } else {
                const auto span = get_bf16_mqa_logits_kv_span<BLOCK_Q, SPLIT_KV>(q_block_idx, num_q_tokens, num_kv_tokens,
                                                                            cu_seq_len_k_start, cu_seq_len_k_end);
                task.kv_token_base = span.kv_token_base;
                task.num_kv_splits = span.num_kv_splits;
            }
            return true;
        }
    }
};

} // namespace deep_gemm::sched
