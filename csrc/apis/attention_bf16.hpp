#pragma once

#include "../jit_kernels/impls/sm100_mqa_logits_bf16.hpp"

namespace deep_gemm::attention {

static void check_bf16_int_tensor(const torch::Tensor& t, const torch::Device& device) {
    DG_HOST_ASSERT(t.is_cuda() and t.device() == device);
    DG_HOST_ASSERT(t.scalar_type() == torch::kInt and t.is_contiguous());
    DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(t.data_ptr()) % 8 == 0);
}

static torch::Tensor bf16_request_indices(const torch::Tensor& lens,
                                          const std::optional<torch::Tensor>& indices) {
    if (indices.has_value()) {
        check_bf16_int_tensor(*indices, lens.device());
        DG_HOST_ASSERT(indices->dim() == 1 and indices->size(0) == lens.size(0));
        return *indices;
    }
    // Without request IDs every row is independent, regardless of the tile hint.
    return torch::arange(lens.size(0), lens.options());
}

static torch::Tensor get_paged_mqa_logits_bf16_metadata(const torch::Tensor& context_lens,
                                                       int block_kv, int num_sms,
                                                       const std::optional<torch::Tensor>& indices,
                                                       int tokens_per_request) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10 and block_kv == 128);
    DG_HOST_ASSERT(num_sms > 0 and num_sms <= runtime->get_num_sms());
    check_bf16_int_tensor(context_lens, context_lens.device());
    DG_HOST_ASSERT(context_lens.dim() == 2 and context_lens.size(1) == 1);
    auto ids = bf16_request_indices(context_lens, indices);
    auto schedule = torch::empty({num_sms + 1, 2}, context_lens.options());
    mqa_bf16::metadata(context_lens, ids, schedule, num_sms, tokens_per_request);
    return schedule;
}

static torch::Tensor fp4_paged_mqa_logits_bf16(const std::pair<torch::Tensor, torch::Tensor>& q,
                                              const torch::Tensor& fused_kv_cache,
                                              const torch::Tensor& weights,
                                              const torch::Tensor& context_lens,
                                              const torch::Tensor& block_table,
                                              const torch::Tensor& schedule_meta,
                                              int max_context_len,
                                              const std::optional<torch::Tensor>& indices,
                                              const std::optional<torch::Tensor>& histogram,
                                              int tokens_per_request) {
    const auto& [q_fp, q_sf] = q;
    const auto device = q_fp.device();
    const mqa_bf16::Config config(tokens_per_request, histogram.has_value());
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10 and q_fp.is_cuda());
    DG_HOST_ASSERT(q_fp.dim() == 4 and q_fp.size(1) == 1 and q_fp.size(2) == 32 and q_fp.size(3) == 64);
    DG_HOST_ASSERT(q_fp.scalar_type() == kPackedFP4 and q_fp.is_contiguous());
    const int rows = q_fp.size(0);
    check_bf16_int_tensor(q_sf, device);
    DG_HOST_ASSERT(q_sf.dim() == 3 and q_sf.size(0) == rows and q_sf.size(1) == 1 and q_sf.size(2) == 32);
    DG_HOST_ASSERT(weights.is_cuda() and weights.device() == device and weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(weights.dim() == 2 and weights.size(0) == rows and weights.size(1) == 32);
    DG_HOST_ASSERT(weights.stride(1) == 1 and weights.stride(0) % 8 == 0);
    DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(weights.data_ptr()) % 16 == 0);

    check_bf16_int_tensor(context_lens, device);
    DG_HOST_ASSERT(context_lens.dim() == 2 and context_lens.size(0) == rows and context_lens.size(1) == 1);
    DG_HOST_ASSERT(max_context_len >= 0);
    DG_HOST_ASSERT(block_table.is_cuda() and block_table.device() == device and block_table.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(block_table.dim() == 2 and block_table.size(0) == rows and block_table.stride(1) == 1);
    DG_HOST_ASSERT(block_table.size(1) >= math::ceil_div(max_context_len, 128));
    check_bf16_int_tensor(schedule_meta, device);
    DG_HOST_ASSERT(schedule_meta.dim() == 2 and schedule_meta.size(0) == runtime->get_num_sms() + 1 and schedule_meta.size(1) == 2);
    auto ids = bf16_request_indices(context_lens, indices);

    DG_HOST_ASSERT(fused_kv_cache.is_cuda() and fused_kv_cache.device() == device);
    DG_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kByte and fused_kv_cache.dim() == 4);
    DG_HOST_ASSERT(fused_kv_cache.size(1) == 128 and fused_kv_cache.size(2) == 1 and fused_kv_cache.size(3) == 68);
    DG_HOST_ASSERT(fused_kv_cache.stride(1) == 68 and fused_kv_cache.stride(3) == 1);
    DG_HOST_ASSERT(fused_kv_cache.stride(0) >= 128 * 68 and fused_kv_cache.stride(0) % 16 == 0);
    DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(fused_kv_cache.data_ptr()) % 16 == 0);
    const int blocks = fused_kv_cache.size(0), kv_stride = fused_kv_cache.stride(0);
    auto kv = torch::from_blob(fused_kv_cache.data_ptr(), {blocks, 128, 64}, {kv_stride, 64, 1},
                                q_fp.options());
    auto sf_kv = torch::from_blob(fused_kv_cache.data_ptr<uint8_t>() + 128 * 64,
                                   {blocks, 128}, {kv_stride / 4, 1}, q_sf.options());

    int* hist_ptr = nullptr;
    if (histogram.has_value()) {
        check_bf16_int_tensor(*histogram, device);
        DG_HOST_ASSERT(histogram->dim() == 2 and histogram->size(0) == rows and histogram->size(1) == 1024);
        hist_ptr = histogram->data_ptr<int>();
    }
    // LCM(384, 512): whole KV splits and 1024-byte aligned BF16 rows.
    const int stride = align(max_context_len, 1536);
    auto logits = torch::empty({rows, stride}, weights.options()).slice(1, 0, max_context_len);
    if (rows > 0 and max_context_len > 0)
        mqa_bf16::paged(config, q_fp, q_sf, kv, sf_kv, weights, context_lens, block_table, ids,
                         schedule_meta, logits, hist_ptr);
    return logits;
}

} // namespace deep_gemm::attention
