#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/sm120.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// SM120 paged MQA logits. Follows dev's per-arch style; uses the atom scheduler
// (scheduler/sm120_paged_mqa_logits.cuh). split_kv is fixed at 128 (2 groups × 64).

// ---- metadata ----
class SM120PagedMQALogitsMetadataRuntime final: public LaunchRuntime<SM120PagedMQALogitsMetadataRuntime> {
public:
    struct Args {
        int aligned_batch_size;
        int split_kv;
        int num_sms;
        bool is_varlen;

        int batch_size;
        int next_n;
        int num_next_n_atoms;
        bool is_context_lens_2d;
        int* context_lens;
        int* indices;
        int* schedule_metadata;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/scheduler/sm120_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sched::sm120_paged_mqa_logits_metadata<
        {}, {}, {}, {}
    >);
}};
)", args.aligned_batch_size, args.split_kv, args.num_sms, args.is_varlen ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.batch_size, args.next_n, args.is_context_lens_2d,
            static_cast<uint32_t>(args.num_next_n_atoms),
            args.context_lens, args.indices, args.schedule_metadata));
    }
};

static void sm120_paged_mqa_logits_metadata(const torch::Tensor& context_lens,
                                            const torch::Tensor& schedule_metadata,
                                            const int& batch_size, const int& next_n,
                                            const int& block_kv, const int& num_sms,
                                            const bool& is_context_lens_2d,
                                            const int& num_next_n_atoms,
                                            const bool& is_varlen, const int* indices_ptr) {
    constexpr int split_kv = 128;
    constexpr int num_threads = 32;
    const int aligned_batch_size = align(batch_size, 32);
    DG_HOST_ASSERT(split_kv % block_kv == 0);

    const int num_smem_ints = is_varlen ? 3 * aligned_batch_size + 1 : aligned_batch_size;
    const int smem_size = num_smem_ints * static_cast<int>(sizeof(int));
    DG_HOST_ASSERT(smem_size <= SM120ArchSpec::smem_capacity);

    const SM120PagedMQALogitsMetadataRuntime::Args& args = {
        .aligned_batch_size = aligned_batch_size,
        .split_kv = split_kv,
        .num_sms = num_sms,
        .is_varlen = is_varlen,
        .batch_size = batch_size,
        .next_n = next_n,
        .num_next_n_atoms = num_next_n_atoms,
        .is_context_lens_2d = is_context_lens_2d,
        .context_lens = context_lens.data_ptr<int>(),
        .indices = const_cast<int*>(indices_ptr),
        .schedule_metadata = schedule_metadata.data_ptr<int>(),
        .launch_args = LaunchArgs(1, num_threads, smem_size)
    };
    const auto code = SM120PagedMQALogitsMetadataRuntime::generate(args);
    const auto runtime = compiler->build("sm120_paged_mqa_logits_metadata", code);
    SM120PagedMQALogitsMetadataRuntime::launch(runtime, args);
}

// ---- FP8 paged ----
class SM120FP8PagedMQALogitsRuntime final: public LaunchRuntime<SM120FP8PagedMQALogitsRuntime> {
public:
    struct Args {
        int batch_size; int next_n; int num_heads; int head_dim; int block_kv;
        bool is_context_lens_2d; bool is_varlen;
        int block_table_stride; int logits_stride;
        int num_q_stages; int num_kv_stages; int split_kv;
        int* context_lens; void* logits; int* block_table; int* indices; int* schedule_meta;
        CUtensorMap tensor_map_q; CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_kv_scales; CUtensorMap tensor_map_weights;
        at::ScalarType logits_dtype;
        int num_specialized_threads; int num_math_threads;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        DG_HOST_ASSERT(128 % args.num_heads == 0);
        return fmt::format(R"(
#include <deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_paged_mqa_logits<
        {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)", args.next_n, args.num_heads,
    args.head_dim, args.block_kv,
    args.is_context_lens_2d, args.is_varlen ? "true" : "false",
    args.num_q_stages, args.num_kv_stages,
    args.split_kv,
    args.num_specialized_threads, args.num_math_threads,
    to_string(args.logits_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.batch_size, args.logits_stride, args.block_table_stride,
            args.context_lens, args.logits, args.block_table, args.indices, args.schedule_meta,
            args.tensor_map_q, args.tensor_map_kv,
            args.tensor_map_kv_scales, args.tensor_map_weights));
    }
};

static void sm120_fp8_paged_mqa_logits(const torch::Tensor& q,
                                       const torch::Tensor& kv_cache,
                                       const torch::Tensor& kv_cache_scales,
                                       const torch::Tensor& weights,
                                       const torch::Tensor& context_lens,
                                       const torch::Tensor& logits,
                                       const torch::Tensor& block_table,
                                       const torch::Tensor& indices,
                                       const torch::Tensor& schedule_meta,
                                       const at::ScalarType& logits_dtype,
                                       const int& batch_size, const int& next_n,
                                       const int& num_heads, const int& head_dim,
                                       const int& num_kv_blocks, const int& block_kv,
                                       const bool& is_context_lens_2d,
                                       const bool& is_varlen,
                                       const int& logits_stride,
                                       const int& block_table_stride,
                                       const int& num_sms,
                                       const int& split_kv) {
    constexpr int num_specialized_threads = 128;
    DG_HOST_ASSERT(block_kv == 64);
    constexpr int mma_m = 64;
    const int num_math_warp_groups = split_kv / mma_m;
    const int num_math_threads = num_math_warp_groups * 4 * 32;
    constexpr int num_q_stages = 2, num_kv_stages = 3;
    DG_HOST_ASSERT(split_kv % mma_m == 0 and logits_stride % split_kv == 0);

    const int next_n_per_cta = (is_varlen or next_n >= 2) ? 2 : 1;
    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, batch_size * next_n * num_heads,
                                               head_dim, next_n_per_cta * num_heads,
                                               static_cast<int>(q.stride(2)), head_dim);
    const auto tensor_map_kv = make_tma_3d_desc(kv_cache, head_dim, block_kv, num_kv_blocks,
                                                head_dim, block_kv, 1,
                                                static_cast<int>(kv_cache.stride(1)),
                                                static_cast<int>(kv_cache.stride(0)), head_dim);
    const auto tensor_map_kv_scales = make_tma_2d_desc(kv_cache_scales, block_kv, num_kv_blocks,
                                                       block_kv, 1,
                                                       static_cast<int>(kv_cache_scales.stride(0)), 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, batch_size * next_n,
                                                     num_heads, next_n_per_cta,
                                                     static_cast<int>(weights.stride(0)), 0);

    // SM120a: per-group KV SMEM (like SM90), not flat split_kv layout
    const int swizzle_alignment = head_dim * 8;
    const int smem_q_size_per_stage = next_n_per_cta * num_heads * head_dim * static_cast<int>(q.element_size());
    const int aligned_smem_weight_size_per_stage = align(next_n_per_cta * num_heads * static_cast<int>(weights.element_size()), swizzle_alignment);
    const int smem_q_pipe_size = num_q_stages * (smem_q_size_per_stage + aligned_smem_weight_size_per_stage) + align(num_q_stages * 8 * 2, swizzle_alignment);

    const int smem_kv_size_per_stage = block_kv * head_dim * static_cast<int>(kv_cache.element_size());
    const int aligned_smem_kv_scale_size_per_stage = align(block_kv * static_cast<int>(kv_cache_scales.element_size()), swizzle_alignment);
    const int smem_kv_pipe_size = num_kv_stages * (smem_kv_size_per_stage + aligned_smem_kv_scale_size_per_stage) + align(num_kv_stages * 8 * 2, swizzle_alignment);

    const int smem_size = smem_q_pipe_size + num_math_warp_groups * smem_kv_pipe_size + 4;
    DG_HOST_ASSERT(smem_size <= SM120ArchSpec::smem_capacity);

    const SM120FP8PagedMQALogitsRuntime::Args args = {
        .batch_size = batch_size, .next_n = next_n, .num_heads = num_heads, .head_dim = head_dim, .block_kv = block_kv,
        .is_context_lens_2d = is_context_lens_2d, .is_varlen = is_varlen,
        .block_table_stride = block_table_stride, .logits_stride = logits_stride,
        .num_q_stages = num_q_stages, .num_kv_stages = num_kv_stages, .split_kv = split_kv,
        .context_lens = context_lens.data_ptr<int>(), .logits = logits.data_ptr(),
        .block_table = block_table.data_ptr<int>(),
        .indices = is_varlen ? indices.data_ptr<int>() : nullptr,
        .schedule_meta = schedule_meta.data_ptr<int>(),
        .tensor_map_q = tensor_map_q, .tensor_map_kv = tensor_map_kv,
        .tensor_map_kv_scales = tensor_map_kv_scales, .tensor_map_weights = tensor_map_weights,
        .logits_dtype = logits_dtype,
        .num_specialized_threads = num_specialized_threads, .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(num_sms, num_specialized_threads + num_math_threads, smem_size)
    };
    const auto code = SM120FP8PagedMQALogitsRuntime::generate(args);
    const auto runtime = compiler->build("sm120_fp8_paged_mqa_logits", code);
    SM120FP8PagedMQALogitsRuntime::launch(runtime, args);
}

// ---- FP4 paged ----
class SM120FP4PagedMQALogitsRuntime final: public LaunchRuntime<SM120FP4PagedMQALogitsRuntime> {
public:
    struct Args {
        int batch_size; int next_n; int num_heads; int head_dim; int block_kv;
        bool is_context_lens_2d; bool is_varlen;
        int block_table_stride; int logits_stride;
        int num_q_stages; int num_kv_stages; int split_kv;
        int* context_lens; void* logits; int* block_table; int* indices; int* schedule_meta;
        CUtensorMap tensor_map_q; CUtensorMap tensor_map_sf_q; CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_sf_kv; CUtensorMap tensor_map_weights;
        at::ScalarType logits_dtype;
        int num_tma_threads; int num_math_threads;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm120_fp4_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp4_paged_mqa_logits<
        {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)", args.next_n, args.num_heads,
    args.head_dim, args.block_kv,
    args.is_context_lens_2d, args.is_varlen ? "true" : "false",
    args.num_q_stages, args.num_kv_stages,
    args.split_kv,
    args.num_tma_threads, args.num_math_threads,
    to_string(args.logits_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.batch_size, args.logits_stride, args.block_table_stride,
            args.context_lens, args.logits, args.block_table, args.indices, args.schedule_meta,
            args.tensor_map_q, args.tensor_map_sf_q,
            args.tensor_map_kv, args.tensor_map_sf_kv, args.tensor_map_weights));
    }
};

static void sm120_fp4_paged_mqa_logits(const torch::Tensor& q,
                                       const torch::Tensor& sf_q,
                                       const torch::Tensor& kv_cache,
                                       const torch::Tensor& kv_cache_sf,
                                       const torch::Tensor& weights,
                                       const torch::Tensor& context_lens,
                                       const torch::Tensor& logits,
                                       const torch::Tensor& block_table,
                                       const torch::Tensor& indices,
                                       const torch::Tensor& schedule_meta,
                                       const at::ScalarType& logits_dtype,
                                       const int& batch_size, const int& next_n,
                                       const int& num_heads, const int& head_dim,
                                       const int& num_kv_blocks, const int& block_kv,
                                       const bool& is_context_lens_2d,
                                       const bool& is_varlen,
                                       const int& logits_stride,
                                       const int& block_table_stride,
                                       const int& num_sms,
                                       const int& split_kv) {
    constexpr int num_tma_threads = 128;
    constexpr int num_math_threads = 256;
    DG_HOST_ASSERT(split_kv == 128 and logits_stride % split_kv == 0);

    constexpr int num_q_stages = 2, num_kv_stages = 3;
    const int next_n_atom = (is_varlen or next_n >= 2) ? 2 : 1;

    DG_HOST_ASSERT(head_dim == 128);

    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, batch_size * next_n * num_heads,
                                               head_dim, next_n_atom * num_heads,
                                               static_cast<int>(q.stride(2)),
                                               head_dim / 2, 0, false, false);
    const auto tensor_map_sf_q = make_tma_2d_desc(sf_q, num_heads, batch_size * next_n,
                                                  num_heads, next_n_atom,
                                                  static_cast<int>(sf_q.stride(1)), 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, batch_size * next_n,
                                                     num_heads, next_n_atom,
                                                     static_cast<int>(weights.stride(0)), 0);
    const auto tensor_map_kv = make_tma_3d_desc(kv_cache, head_dim, block_kv, num_kv_blocks,
                                                head_dim, block_kv, 1,
                                                static_cast<int>(kv_cache.stride(1)),
                                                static_cast<int>(kv_cache.stride(0)),
                                                head_dim / 2, 0, false, false);
    const auto tensor_map_sf_kv = make_tma_2d_desc(kv_cache_sf, block_kv, num_kv_blocks,
                                                   block_kv, 1,
                                                   static_cast<int>(kv_cache_sf.stride(0)), 0);

    // SM120: per-group KV SMEM
    const int swizzle_alignment = head_dim / 2 * 8;
    const int smem_q_size_per_stage = next_n_atom * num_heads * head_dim / 2;
    const int aligned_smem_sf_q_size_per_stage = align(next_n_atom * num_heads * static_cast<int>(sizeof(int)), swizzle_alignment);
    const int aligned_smem_weight_size_per_stage = align(next_n_atom * num_heads * static_cast<int>(sizeof(float)), swizzle_alignment);
    const int smem_q_pipe_size = num_q_stages * (smem_q_size_per_stage + aligned_smem_sf_q_size_per_stage + aligned_smem_weight_size_per_stage) +
                                 align(num_q_stages * 8 * 2, swizzle_alignment);

    const int smem_kv_size_per_stage = block_kv * head_dim / 2;
    const int aligned_smem_sf_kv_size_per_stage = align(block_kv * static_cast<int>(sizeof(int)), swizzle_alignment);
    const int smem_kv_pipe_size = num_kv_stages * (smem_kv_size_per_stage + aligned_smem_sf_kv_size_per_stage) +
                                  align(num_kv_stages * 8 * 2, swizzle_alignment);

    const int num_groups = split_kv / block_kv;
    const int smem_size = smem_q_pipe_size + num_groups * smem_kv_pipe_size + 4;
    DG_HOST_ASSERT(smem_size <= SM120ArchSpec::smem_capacity);

    const SM120FP4PagedMQALogitsRuntime::Args args = {
        .batch_size = batch_size, .next_n = next_n, .num_heads = num_heads, .head_dim = head_dim, .block_kv = block_kv,
        .is_context_lens_2d = is_context_lens_2d, .is_varlen = is_varlen,
        .block_table_stride = block_table_stride, .logits_stride = logits_stride,
        .num_q_stages = num_q_stages, .num_kv_stages = num_kv_stages, .split_kv = split_kv,
        .context_lens = context_lens.data_ptr<int>(), .logits = logits.data_ptr(),
        .block_table = block_table.data_ptr<int>(),
        .indices = is_varlen ? indices.data_ptr<int>() : nullptr,
        .schedule_meta = schedule_meta.data_ptr<int>(),
        .tensor_map_q = tensor_map_q, .tensor_map_sf_q = tensor_map_sf_q,
        .tensor_map_kv = tensor_map_kv, .tensor_map_sf_kv = tensor_map_sf_kv,
        .tensor_map_weights = tensor_map_weights,
        .logits_dtype = logits_dtype,
        .num_tma_threads = num_tma_threads, .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(num_sms, num_tma_threads + num_math_threads, smem_size)
    };
    const auto code = SM120FP4PagedMQALogitsRuntime::generate(args);
    const auto runtime = compiler->build("sm120_fp4_paged_mqa_logits", code);
    SM120FP4PagedMQALogitsRuntime::launch(runtime, args);
}

} // namespace deep_gemm
