#pragma once

#include <format>

#include "../../runtime/runtime.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"
#include <deep_gemm/mqa_bf16/layout.cuh>

namespace deep_gemm::mqa_bf16 {

// A paired API: the BF16 schedule uses split 384 and must never consume the
// legacy paged API's split-256 metadata. The hint is CPU configuration; request
// runs and causal lengths remain on the device.
struct Config {
    int block_q, num_tmem_stages, num_kv_stages;
    bool histogram, swizzle;
    static constexpr int split_kv = 384;

    Config(int tokens_per_request, bool histogram):
        block_q(tokens_per_request >= 5 ? 6 : 4),
        num_tmem_stages(tokens_per_request >= 5 ? 2 : 3),
        num_kv_stages(tokens_per_request >= 5 ? (histogram ? 6 : 7) : (histogram ? 7 : 8)),
        histogram(histogram), swizzle(tokens_per_request == 6) {
        DG_HOST_ASSERT(tokens_per_request >= 1 and tokens_per_request <= 6);
    }

    int storage_size() const {
        using FP4 = cutlass::float_e2m1_t;
        if (block_q == 6 and num_kv_stages == 6)
            return sizeof(layout::MQALogitsSharedStorage<32, 128, 6, 384, 192, 2, 6, 2, FP4>);
        if (block_q == 6)
            return sizeof(layout::MQALogitsSharedStorage<32, 128, 6, 384, 192, 2, 7, 2, FP4>);
        if (num_kv_stages == 7)
            return sizeof(layout::MQALogitsSharedStorage<32, 128, 4, 384, 128, 2, 7, 3, FP4>);
        return sizeof(layout::MQALogitsSharedStorage<32, 128, 4, 384, 128, 2, 8, 3, FP4>);
    }

    int smem_size() const {
        // The histogram window holds two Q blocks of rows: 16-bit counters, 1024 bins per row
        const int bytes = storage_size() + (histogram ? 2 * block_q * 1024 * static_cast<int>(sizeof(uint16_t)) : 0);
        DG_HOST_ASSERT(bytes <= SM100ArchSpec::smem_capacity);
        return bytes;
    }
};

static void metadata(const torch::Tensor& lens, const torch::Tensor& indices,
                     const torch::Tensor& schedule, int num_sms, int tokens_per_request) {
    const int rows = lens.size(0);
    const int threads = rows <= 512 ? 256 : (rows <= 2048 ? 512 : 1024);
    const int smem = (2 * rows + threads / 32 + 1) * sizeof(int);
    DG_HOST_ASSERT(smem <= SM100ArchSpec::smem_capacity);
    const int block_q = Config(tokens_per_request, false).block_q;
    const auto kernel = jit->compile("sm100_paged_mqa_logits_bf16_metadata", std::format(R"(
#include <deep_gemm/mqa_bf16/scheduler.cuh>
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::mqa_bf16::sched::sm100_paged_mqa_logits_metadata<384, {}, {}, {}>);
}}
)", num_sms, block_q, threads));
    jit->launch(
        kernel, {
            .num_smem_bytes = smem,
            .grid_dim = dim3(1, 1, 1),
            .block_dim = dim3(threads, 1, 1),
        },
        rows, lens.data_ptr<int>(), indices.data_ptr<int>(), schedule.data_ptr<int>()
    );
}

static void paged(const Config& c, const torch::Tensor& q, const torch::Tensor& sf_q,
                  const torch::Tensor& kv, const torch::Tensor& sf_kv, const torch::Tensor& weights,
                  const torch::Tensor& lens, const torch::Tensor& table, const torch::Tensor& indices,
                  const torch::Tensor& schedule, const torch::Tensor& logits, int* histogram) {
    const int rows = q.size(0), num_sms = runtime->get_num_sms();
    const auto tensor_map_q = make_tma_2d_desc(
        q, 128, rows * 32, 128, c.block_q * 32,
        static_cast<int>(q.stride(2)), 64, 0, false, false);
    const auto tensor_map_sf_q = make_tma_2d_desc(
        sf_q, 32, rows, 32, c.block_q, static_cast<int>(sf_q.stride(1)), 0);
    const auto tensor_map_kv = make_tma_3d_desc(
        kv, 128, 128, kv.size(0), 128, 128, 1,
        static_cast<int>(kv.stride(1)), static_cast<int>(kv.stride(0)),
        64, 0, false, false);
    const auto tensor_map_sf_kv = make_tma_2d_desc(
        sf_kv, 128, sf_kv.size(0), 128, 1, static_cast<int>(sf_kv.stride(0)), 0);
    const auto tensor_map_weights = make_tma_2d_desc(
        weights, 32, rows, 32, c.block_q, static_cast<int>(weights.stride(0)), 0);
    const auto kernel = jit->compile("sm100_paged_mqa_logits_bf16", std::format(R"(
#include <deep_gemm/mqa_bf16/kernel.cuh>
using namespace deep_gemm::mqa_bf16;
static_assert(sizeof(layout::MQALogitsSharedStorage<32, 128, {}, 384, {}, 2, {}, {}, cutlass::float_e2m1_t>) == {});
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_paged_mqa_logits<
        32, 128, 128, {}, {}, 2, {}, {}, 384, {}, 128, 384, cutlass::float_e2m1_t, {}, {}
    >);
}}
)", c.block_q, c.block_q * 32, c.num_kv_stages, c.num_tmem_stages, c.storage_size(),
    c.block_q, c.block_q * 32, c.num_kv_stages, c.num_tmem_stages, c.num_kv_stages,
    c.histogram, c.swizzle));
    jit->launch(
        kernel, {
            .num_smem_bytes = c.smem_size(),
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(512, 1, 1),
        },
        rows, static_cast<int>(logits.stride(0)), static_cast<int>(table.stride(0)),
        lens.data_ptr<int>(), logits.data_ptr(), table.data_ptr<int>(),
        indices.data_ptr<int>(), schedule.data_ptr<int>(),
        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv,
        tensor_map_weights, histogram
    );
}

} // namespace deep_gemm::mqa_bf16
