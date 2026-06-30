#pragma once

#include "sm100_mqa_logits.hpp"

namespace deep_gemm {

static constexpr int kSM120SmemCapacity = 101376;

static void sm120_fp4_paged_mqa_logits_metadata(
        const torch::Tensor& context_lens,
        const torch::Tensor& schedule_meta,
        const int batch_size, const int num_sms) {
    constexpr int next_n = 1;
    constexpr int split_kv = 128;
    constexpr int num_threads = 256;
    const int smem_size = 2 * batch_size * static_cast<int>(sizeof(int));
    DG_HOST_ASSERT(smem_size <= kSM120SmemCapacity);

    const SM100PagedMQALogitsMetadataRuntime::Args args = {
        .next_n = next_n,
        .is_context_lens_2d = true,
        .is_varlen = false,
        .split_kv = split_kv,
        .num_sms = num_sms,
        .num_requests = batch_size,
        .num_q_tokens_total = batch_size,
        .context_lens = context_lens.data_ptr<int>(),
        .indices = nullptr,
        .schedule_meta = schedule_meta.data_ptr<int>(),
        .launch_args = LaunchArgs(1, num_threads, smem_size)
    };
    const auto code = SM100PagedMQALogitsMetadataRuntime::generate(args);
    const auto runtime =
        compiler->build("sm120_fp4_paged_mqa_logits_metadata", code);
    SM100PagedMQALogitsMetadataRuntime::launch(runtime, args);
}

class SM120FP4PagedMQALogitsRuntime final
        : public LaunchRuntime<SM120FP4PagedMQALogitsRuntime> {
public:
    struct Args {
        int batch_size;
        int block_table_stride;
        int logits_stride;

        int* context_lens;
        float* logits;
        int* block_table;
        int* schedule_meta;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_sf_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_sf_kv;
        CUtensorMap tensor_map_weights;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args&) {
        return R"(
#include <deep_gemm/impls/sm120_fp4_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm120_fp4_paged_mqa_logits);
}
)";
    }

    static void launch_impl(
            const KernelHandle& kernel, const LaunchConfigHandle& config,
            Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config,
            args.batch_size, args.logits_stride, args.block_table_stride,
            args.context_lens, args.logits, args.block_table,
            args.schedule_meta, args.tensor_map_q, args.tensor_map_sf_q,
            args.tensor_map_kv, args.tensor_map_sf_kv,
            args.tensor_map_weights
        ));
    }
};

static void sm120_fp4_paged_mqa_logits(
        const torch::Tensor& q, const torch::Tensor& sf_q,
        const torch::Tensor& kv_cache, const torch::Tensor& kv_cache_sf,
        const torch::Tensor& weights, const torch::Tensor& context_lens,
        const torch::Tensor& logits, const torch::Tensor& block_table,
        const torch::Tensor& schedule_meta, const int batch_size,
        const int num_heads, const int head_dim, const int num_kv_blocks,
        const int block_kv, const int logits_stride,
        const int block_table_stride, const int num_sms,
        const int split_kv) {
    constexpr int num_tma_threads = 128;
    constexpr int num_math_threads = 256;
    constexpr int num_q_stages = 2;
    constexpr int num_kv_stages = 3;

    DG_HOST_ASSERT(num_heads == 64 and head_dim == 128);
    DG_HOST_ASSERT(block_kv == 64 and split_kv == 128);
    DG_HOST_ASSERT(logits_stride % split_kv == 0);
    DG_HOST_ASSERT(logits.scalar_type() == torch::kFloat32);

    const auto tensor_map_q = make_tma_2d_desc(
        q, head_dim, batch_size * num_heads,
        head_dim, num_heads, static_cast<int>(q.stride(2)),
        head_dim / 2, 0, false, false);
    const auto tensor_map_sf_q = make_tma_2d_desc(
        sf_q, num_heads, batch_size, num_heads, 1,
        static_cast<int>(sf_q.stride(1)), 0);
    const auto tensor_map_weights = make_tma_2d_desc(
        weights, num_heads, batch_size, num_heads, 1,
        static_cast<int>(weights.stride(0)), 0);
    const auto tensor_map_kv = make_tma_3d_desc(
        kv_cache, head_dim, block_kv, num_kv_blocks,
        head_dim, block_kv, 1,
        static_cast<int>(kv_cache.stride(1)),
        static_cast<int>(kv_cache.stride(0)),
        head_dim / 2, 0, false, false);
    const auto tensor_map_sf_kv = make_tma_2d_desc(
        kv_cache_sf, block_kv, num_kv_blocks, block_kv, 1,
        static_cast<int>(kv_cache_sf.stride(0)), 0);

    const int swizzle_alignment = head_dim / 2 * 8;
    const int smem_q_size_per_stage = num_heads * head_dim / 2;
    const int aligned_smem_sf_q_size_per_stage = align(
        num_heads * static_cast<int>(sizeof(int)), swizzle_alignment);
    const int aligned_smem_weight_size_per_stage = align(
        num_heads * static_cast<int>(sizeof(float)), swizzle_alignment);
    const int smem_q_pipe_size =
        num_q_stages *
            (smem_q_size_per_stage +
             aligned_smem_sf_q_size_per_stage +
             aligned_smem_weight_size_per_stage) +
        align(num_q_stages * 8 * 2, swizzle_alignment);

    const int smem_kv_size_per_stage = block_kv * head_dim / 2;
    const int aligned_smem_sf_kv_size_per_stage = align(
        block_kv * static_cast<int>(sizeof(int)), swizzle_alignment);
    const int smem_kv_pipe_size =
        num_kv_stages *
            (smem_kv_size_per_stage +
             aligned_smem_sf_kv_size_per_stage) +
        align(num_kv_stages * 8 * 2, swizzle_alignment);

    const int num_groups = split_kv / block_kv;
    const int smem_size =
        smem_q_pipe_size + num_groups * smem_kv_pipe_size + 4;
    DG_HOST_ASSERT(smem_size <= kSM120SmemCapacity);

    const SM120FP4PagedMQALogitsRuntime::Args args = {
        .batch_size = batch_size,
        .block_table_stride = block_table_stride,
        .logits_stride = logits_stride,
        .context_lens = context_lens.data_ptr<int>(),
        .logits = logits.data_ptr<float>(),
        .block_table = block_table.data_ptr<int>(),
        .schedule_meta = schedule_meta.data_ptr<int>(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_sf_q = tensor_map_sf_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_sf_kv = tensor_map_sf_kv,
        .tensor_map_weights = tensor_map_weights,
        .launch_args = LaunchArgs(
            num_sms, num_tma_threads + num_math_threads, smem_size)
    };
    const auto code = SM120FP4PagedMQALogitsRuntime::generate(args);
    const auto runtime =
        compiler->build("sm120_fp4_paged_mqa_logits", code);
    SM120FP4PagedMQALogitsRuntime::launch(runtime, args);
}

}  // namespace deep_gemm
