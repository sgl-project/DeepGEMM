#pragma once

#include <deep_gemm/common/math.cuh>

namespace deep_gemm::sched {

// Minimal scheduler for SGLang's DeepSeek-V4 indexer contract:
// one query token per request, 2-D context lengths, and no varlen indices.
template <uint32_t kBlockKV, uint32_t kNumBlocksPerSplit>
struct SM120FP4PagedMQALogitsScheduler {
    const uint32_t* context_lens;
    uint32_t batch_size;

    uint32_t current_q_idx;
    uint32_t current_kv_idx;
    uint32_t end_q_idx;
    uint32_t end_kv_idx;
    uint32_t current_num_kv;

    CUTLASS_DEVICE SM120FP4PagedMQALogitsScheduler(
            const uint32_t sm_idx, const uint32_t batch_size,
            const uint32_t* context_lens, const uint32_t* schedule_meta) {
        this->context_lens = context_lens;
        this->batch_size = batch_size;

        const auto current =
            reinterpret_cast<const uint2*>(schedule_meta)[sm_idx];
        const auto end =
            reinterpret_cast<const uint2*>(schedule_meta)[sm_idx + 1];
        current_q_idx = current.x;
        current_kv_idx = current.y * kNumBlocksPerSplit;
        end_q_idx = end.x;
        end_kv_idx = end.y * kNumBlocksPerSplit;
        refresh_num_kv();
    }

    CUTLASS_DEVICE void refresh_num_kv() {
        current_num_kv = current_q_idx < batch_size
            ? math::ceil_div(context_lens[current_q_idx], kBlockKV)
            : 0;
    }

    CUTLASS_DEVICE static uint32_t atom_to_token_idx(
            const uint32_t q_idx) {
        return q_idx;
    }

    CUTLASS_DEVICE static uint32_t atom_to_block_table_row(
            const uint32_t q_idx) {
        return q_idx;
    }

    CUTLASS_DEVICE static uint32_t get_last_advance() {
        return 1;
    }

    CUTLASS_DEVICE bool exist_q_atom_idx(const uint32_t q_idx) const {
        return q_idx < end_q_idx or
               (q_idx == end_q_idx and end_kv_idx > 0);
    }

    CUTLASS_DEVICE bool fetch_next_task(
            uint32_t& q_idx, uint32_t& kv_idx, uint32_t& num_kv) {
        q_idx = current_q_idx;
        kv_idx = current_kv_idx;
        num_kv = current_num_kv;

        if (current_q_idx == end_q_idx and current_kv_idx == end_kv_idx)
            return false;

        current_kv_idx += kNumBlocksPerSplit;
        if (current_kv_idx >= current_num_kv) {
            current_kv_idx = 0;
            ++current_q_idx;
            refresh_num_kv();
        }
        return true;
    }
};

}  // namespace deep_gemm::sched
