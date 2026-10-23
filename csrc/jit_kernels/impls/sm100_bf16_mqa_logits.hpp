#pragma once

#include <format>

#include "../../runtime/runtime.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// SM100 contiguous-KV MQA logits with MX scales, BF16 weights and compressed BF16 logits: one math warpgroup
// per 128-token KV tile of a 384-token split, separate KV and SF producers, and KV stages reused across Q
// blocks that share their splits
static constexpr int kBF16MQALogitsSplitKV = 384;
static constexpr int kBF16MQALogitsNumTmemStages = 3;
static constexpr int kBF16MQALogitsNumSpecializedThreads = 128;

struct BF16MQALogitsConfig {
    bool is_fp4;
    int num_heads, head_dim, block_q, umma_n;
    int num_q_stages, num_kv_stages;
    static constexpr int split_kv = kBF16MQALogitsSplitKV;
    static constexpr int num_tmem_stages = kBF16MQALogitsNumTmemStages;
    static constexpr int num_specialized_threads = kBF16MQALogitsNumSpecializedThreads;
    static constexpr int num_math_threads = (kBF16MQALogitsSplitKV / 128) * 128;

    BF16MQALogitsConfig(const int& num_heads, const int& head_dim, const int& block_q, const at::ScalarType& qk_dtype):
            is_fp4(qk_dtype == kPackedFP4), num_heads(num_heads), head_dim(head_dim), block_q(block_q) {
        umma_n = math::align(block_q * num_heads, 8);
        num_q_stages = is_fp4 ? 2 : 3;
        num_kv_stages = is_fp4 ? 8 : 3;
    }

    const char* qk_dtype_name() const {
        return is_fp4 ? "cutlass::float_e2m1_t" : "cutlass::float_e4m3_t";
    }

    // TMA swizzle mode: one token row of Q / KV
    int swizzle_mode() const {
        return is_fp4 ? head_dim / 2 : head_dim;
    }

    // Weights rows are padded to 16 bytes for TMA
    int num_weight_elements_per_row() const {
        return static_cast<int>(math::align(num_heads * static_cast<uint32_t>(sizeof(nv_bfloat16)), 16u) / sizeof(nv_bfloat16));
    }

    // Template arguments of `layout::BF16MQALogitsSharedStorage`; the generated source asserts `sizeof` == `smem_size()`
    std::string shared_storage_args() const {
        return std::format("{}, {}, {}, {}, {}, {}, {}, {}, {}",
                           num_heads, head_dim, block_q, split_kv, umma_n, num_q_stages, num_kv_stages, num_tmem_stages,
                           qk_dtype_name());
    }

    // Mirrors `layout::BF16MQALogitsSharedStorage`
    int smem_size() const {
        constexpr uint32_t kNumBarrierBytes = 8;
        constexpr uint32_t kNumUTCCPAlignedElems = 128;
        constexpr uint32_t kTmaAlignment = 128;
        const uint32_t num_qk_bytes_per_token = is_fp4 ? head_dim / 2 : head_dim;
        const uint32_t swizzle_alignment = 8 * num_qk_bytes_per_token;
        const uint32_t num_sf_q = math::align(static_cast<uint32_t>(umma_n), kNumUTCCPAlignedElems);
        const uint32_t num_sf_kv = math::align(static_cast<uint32_t>(split_kv), kNumUTCCPAlignedElems);
        const uint32_t num_weight_bytes_per_stage = math::align(
            static_cast<uint32_t>(block_q * num_weight_elements_per_row() * sizeof(nv_bfloat16)), kTmaAlignment);

        uint32_t num_smem_bytes = 0;
        const auto add_region = [&](const uint32_t& num_region_bytes, const uint32_t& alignment) {
            num_smem_bytes = math::align(num_smem_bytes, alignment) + num_region_bytes;
        };
        add_region(num_q_stages * umma_n * num_qk_bytes_per_token, swizzle_alignment);
        add_region(num_kv_stages * split_kv * num_qk_bytes_per_token, swizzle_alignment);
        add_region(num_q_stages * num_sf_q * sizeof(uint32_t), kTmaAlignment);
        add_region(num_kv_stages * num_sf_kv * sizeof(uint32_t), kTmaAlignment);
        add_region(num_q_stages * num_weight_bytes_per_stage, kTmaAlignment);
        const uint32_t num_barriers = 3 * (num_q_stages + num_kv_stages) + 2 * num_tmem_stages;
        add_region(num_barriers * kNumBarrierBytes, kNumBarrierBytes);
        add_region(sizeof(uint32_t), alignof(uint32_t));
        const int smem_size = static_cast<int>(math::align(num_smem_bytes, swizzle_alignment));
        DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);
        return smem_size;
    }
};

static void sm100_bf16_mqa_logits(const torch::Tensor& q, const torch::Tensor& sf_q,
                                  const torch::Tensor& kv, const torch::Tensor& sf_kv,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& cu_seq_len_k_start,
                                  const torch::Tensor& cu_seq_len_k_end,
                                  const torch::Tensor& logits,
                                  const int& num_q_tokens, const int& num_kv_tokens,
                                  const int& stride_logits,
                                  const int& num_heads, const int& head_dim, const int& block_q,
                                  const at::ScalarType& qk_dtype) {
    const BF16MQALogitsConfig config(num_heads, head_dim, block_q, qk_dtype);
    const int num_sms = runtime->get_num_sms();

    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, num_q_tokens * num_heads,
                                               head_dim, block_q * num_heads,
                                               static_cast<int>(q.stride(1)),
                                               config.swizzle_mode(), 0, false, not config.is_fp4);
    const int kv_tma_tokens = config.split_kv % 256 == 0 ? 256 : 128;
    const auto tensor_map_kv = make_tma_2d_desc(kv, head_dim, num_kv_tokens,
                                                head_dim, kv_tma_tokens,
                                                static_cast<int>(kv.stride(0)),
                                                config.swizzle_mode(), 0, false, not config.is_fp4);
    const auto tensor_map_sf_kv = make_tma_2d_desc(sf_kv,
                                                   get_tma_aligned_size(num_kv_tokens, static_cast<int>(sf_kv.element_size())), 1,
                                                   kv_tma_tokens, 1, 0, 0);
    const auto tensor_map_sf_q = make_tma_2d_desc(sf_q, num_heads, num_q_tokens,
                                                  num_heads, block_q,
                                                  static_cast<int>(sf_q.stride(0)), 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, num_q_tokens,
                                                     config.num_weight_elements_per_row(), block_q,
                                                     static_cast<int>(weights.stride(0)), 0);
    const int smem_size = config.smem_size();

    // Compile
    const auto kernel = jit->compile("sm100_bf16_mqa_logits", std::format(R"(
#include <deep_gemm/impls/sm100_bf16_mqa_logits.cuh>

using namespace deep_gemm;

static_assert(sizeof(layout::BF16MQALogitsSharedStorage<
    {}
>) == {}, "Incorrect MQA logits shared-memory size");

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_mqa_logits<
        {}, {},
        false,
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {},
        {}
    >);
}};
)",
    config.shared_storage_args(), smem_size,
    num_heads, head_dim,
    block_q, config.split_kv, config.umma_n,
    config.num_q_stages, config.num_kv_stages, config.num_tmem_stages,
    num_sms,
    config.num_specialized_threads, config.num_math_threads,
    config.qk_dtype_name()));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(config.num_specialized_threads + config.num_math_threads, 1, 1),
        },
        num_q_tokens, num_kv_tokens,
        stride_logits,
        cu_seq_len_k_start.data_ptr<int>(), cu_seq_len_k_end.data_ptr<int>(),
        static_cast<int*>(nullptr),
        logits.data_ptr(),
        tensor_map_q, tensor_map_sf_q,
        tensor_map_kv, tensor_map_sf_kv,
        tensor_map_weights
    );
}

} // namespace deep_gemm
