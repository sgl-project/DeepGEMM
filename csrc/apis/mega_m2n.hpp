#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

#include "../utils/compatibility.hpp"
#include "../utils/math.hpp"
#include "../runtime/runtime.hpp"
#include "../jit_kernels/impls/sm100_mega_moe_m2n.hpp"

namespace deep_gemm::mega_m2n {

using BufferSlice = std::function<std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(
    const torch::Tensor&)>;

static std::tuple<int64_t, int64_t, BufferSlice>
get_symm_buffer_size_for_mega_moe_m2n(
    const int& num_ag_ranks,
    const int& num_eg_ranks,
    const int& num_experts,
    const int& num_max_tokens_per_rank,
    const int& num_topk,
    const int& hidden,
    const int& intermediate_hidden,
    const bool& use_nvfp4) {
    DG_HOST_ASSERT(num_ag_ranks > 0 and num_eg_ranks > 0);
    DG_HOST_ASSERT(num_experts % num_eg_ranks == 0);
    const int block_k = use_nvfp4 ? 256 : 128;
    DG_HOST_ASSERT(hidden % block_k == 0 and intermediate_hidden % block_k == 0);

    const auto workspace = layout::WorkspaceM2N(
        nullptr,
        num_ag_ranks,
        num_eg_ranks,
        num_experts,
        num_max_tokens_per_rank,
        num_topk);

    const int packing = use_nvfp4 ? 2 : 1;
    const int gran_k = use_nvfp4 ? 16 : 32;
    const auto fp8_token_layout = layout::Data(hidden / packing);
    const auto bf16_token_layout = layout::Data(hidden * 2);
    const auto fp8_intermediate_token_layout = layout::Data(intermediate_hidden / packing);
    const auto fp8_sf_layout = layout::Data(hidden / gran_k);
    // Pad the allocation tail, not the M-major scale stride.
    const auto fp8_intermediate_sf_layout = layout::Data(
        math::align(intermediate_hidden / gran_k, 16));
    const auto input_topk_idx_layout =
        layout::Data(num_topk * sizeof(int64_t), false);
    const auto input_topk_weights_layout =
        layout::Data(num_topk * sizeof(float), false);
    const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout,
        1,
        num_max_tokens_per_rank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout,
        1,
        num_max_tokens_per_rank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout,
        1,
        num_max_tokens_per_rank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout,
        1,
        num_max_tokens_per_rank,
        input_topk_idx_buffer.get_end_ptr());

    const auto num_max_pool_tokens =
        static_cast<int>(workspace.num_max_pool_tokens);
    int num_max_padded_sf_pool_tokens = 0;
    for (const int block_m : layout::kCandidateBlockM) {
        num_max_padded_sf_pool_tokens = std::max(
            num_max_padded_sf_pool_tokens,
            layout::get_num_sf_ring_tokens(
                num_max_pool_tokens, block_m));
    }

    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout,
        1,
        num_max_pool_tokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout,
        1,
        num_max_padded_sf_pool_tokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout,
        1,
        num_max_pool_tokens,
        l1_sf_buffer.get_end_ptr());

    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout,
        1,
        num_max_pool_tokens,
        l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout,
        1,
        num_max_padded_sf_pool_tokens,
        l2_token_buffer.get_end_ptr());

    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout,
        num_topk,
        num_max_tokens_per_rank,
        l2_sf_buffer.get_end_ptr());

    DG_HOST_ASSERT(num_max_padded_sf_pool_tokens % 4 == 0);

    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto x = torch::from_blob(
            math::advance_ptr(
                buffer.data_ptr(),
                reinterpret_cast<int64_t>(input_token_buffer.base)),
            {num_max_tokens_per_rank, hidden / packing},
            torch::TensorOptions()
                .dtype(use_nvfp4 ? torch::kByte : torch::kFloat8_e4m3fn)
                .device(buffer.device()));
        auto x_sf = torch::from_blob(
            math::advance_ptr(
                buffer.data_ptr(),
                reinterpret_cast<int64_t>(input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden / (gran_k * 4)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        auto topk_idx = torch::from_blob(
            math::advance_ptr(
                buffer.data_ptr(),
                reinterpret_cast<int64_t>(input_topk_idx_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(
                buffer.data_ptr(),
                reinterpret_cast<int64_t>(input_topk_weights_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(buffer.device()));
        auto l1_acts = torch::from_blob(
            math::advance_ptr(
                buffer.data_ptr(),
                reinterpret_cast<int64_t>(l1_token_buffer.base)),
            {num_max_pool_tokens, hidden / packing},
            torch::TensorOptions()
                .dtype(use_nvfp4 ? torch::kByte : torch::kFloat8_e4m3fn)
                .device(buffer.device()));
        auto l1_acts_sf = torch::from_blob(
            math::advance_ptr(
                buffer.data_ptr(),
                reinterpret_cast<int64_t>(l1_sf_buffer.base)),
            {num_max_padded_sf_pool_tokens, hidden / (gran_k * 4)},
            {1, num_max_padded_sf_pool_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        auto l2_acts = torch::from_blob(
            math::advance_ptr(
                buffer.data_ptr(),
                reinterpret_cast<int64_t>(l2_token_buffer.base)),
            {num_max_pool_tokens, intermediate_hidden / packing},
            torch::TensorOptions()
                .dtype(use_nvfp4 ? torch::kByte : torch::kFloat8_e4m3fn)
                .device(buffer.device()));
        auto l2_acts_sf = torch::from_blob(
            math::advance_ptr(
                buffer.data_ptr(),
                reinterpret_cast<int64_t>(l2_sf_buffer.base)),
            {num_max_padded_sf_pool_tokens, intermediate_hidden / (gran_k * 4)},
            {1, num_max_padded_sf_pool_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        return std::make_tuple(
            x,
            x_sf,
            topk_idx,
            topk_weights,
            l1_acts,
            l1_acts_sf,
            l2_acts,
            l2_acts_sf);
    };

    return {
        reinterpret_cast<int64_t>(combine_token_buffer.get_end_ptr()),
        static_cast<int64_t>(num_max_pool_tokens),
        slice_input_buffers};
}

static void initialize_symm_buffer_for_mega_moe_m2n(
    const torch::Tensor& sym_buffer, const int& num_ranks) {
    constexpr auto signal_offset =
        layout::WorkspaceM2N::get_nvl_barrier_signal_offset(0);
    DG_HOST_ASSERT(sym_buffer.is_cuda());
    DG_HOST_ASSERT(sym_buffer.scalar_type() == torch::kInt8);
    DG_HOST_ASSERT(sym_buffer.is_contiguous());
    DG_HOST_ASSERT(sym_buffer.nbytes() >= signal_offset + sizeof(int));
    DG_HOST_ASSERT(num_ranks > 0);
    sym_buffer
        .narrow(0, static_cast<int64_t>(signal_offset), sizeof(int))
        .view(torch::kInt32)
        .fill_(num_ranks);
}

static void mega_moe_m2n_ag(
    const torch::Tensor& y,
    const torch::Tensor& hidden_states,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx,
    const int& num_ag_ranks,
    const int& num_eg_ranks,
    const int& num_max_tokens_per_rank,
    const int& num_experts,
    const int& num_topk,
    const int& intermediate_hidden,
    const float& output_scale,
    const int& num_sms,
    const bool& use_nvfp4,
    const std::optional<torch::Tensor>& input_scale) {
    if (use_nvfp4) {
        DG_HOST_ASSERT(input_scale.has_value());
        DG_HOST_ASSERT(input_scale->is_cuda() and input_scale->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(input_scale->numel() == 1);
    }
    const auto num_tokens = static_cast<int>(hidden_states.size(0));
    const auto hidden = static_cast<int>(hidden_states.size(1));
    DG_HOST_ASSERT(y.is_cuda() and hidden_states.is_cuda());
    DG_HOST_ASSERT(y.scalar_type() == torch::kBFloat16 and y.is_contiguous());
    DG_HOST_ASSERT(
        hidden_states.scalar_type() == torch::kBFloat16 and
        hidden_states.is_contiguous());
    DG_HOST_ASSERT(y.size(0) == num_tokens and y.size(1) == hidden);
    DG_HOST_ASSERT(
        topk_ids.scalar_type() == torch::kInt64 or
        topk_ids.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(
        topk_ids.is_cuda() and topk_ids.is_contiguous() and
        topk_ids.size(0) == num_tokens and topk_ids.size(1) == num_topk);
    DG_HOST_ASSERT(
        topk_weights.is_cuda() and topk_weights.is_contiguous() and
        topk_weights.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(
        topk_weights.size(0) == num_tokens and
        topk_weights.size(1) == num_topk);
    DG_HOST_ASSERT(
        static_cast<int>(sym_buffer_ptrs.size()) ==
        num_ag_ranks + num_eg_ranks);
    DG_HOST_ASSERT(rank_idx >= 0 and rank_idx < num_ag_ranks);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);

    const auto [num_required_bytes, num_max_pool_tokens, slice] =
        get_symm_buffer_size_for_mega_moe_m2n(
            num_ag_ranks,
            num_eg_ranks,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden, use_nvfp4);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    const auto [
        x,
        x_sf,
        buffer_topk_idx,
        buffer_topk_weights,
        l1_acts,
        l1_acts_sf,
        l2_acts,
        l2_acts_sf] = slice(sym_buffer);

    if (jit->device.get_arch_major() == 10) {
        sm100_mega_moe_m2n_ag(
            y,
            hidden_states,
            topk_ids,
            topk_weights,
            sym_buffer_ptrs,
            rank_idx,
            num_ag_ranks,
            num_eg_ranks,
            num_max_tokens_per_rank,
            num_experts,
            num_tokens,
            num_topk,
            hidden,
            intermediate_hidden,
            static_cast<int>(num_max_pool_tokens),
            static_cast<int>(l1_acts_sf.size(0)),
            output_scale,
            num_sms,
            use_nvfp4,
            input_scale.has_value() ? input_scale->const_data_ptr<float>() : nullptr);
        return;
    }
    DG_HOST_UNREACHABLE("Unsupported architecture");
}

static void mega_moe_m2n_eg(
    const torch::Tensor& weight_descs,
    const std::vector<torch::Tensor>& sym_buffers,
    const std::vector<std::vector<int64_t>>& sym_buffer_ptrs,
    const int& rank_idx,
    const int& num_ag_ranks,
    const int& num_eg_ranks,
    const int& num_max_tokens_per_rank,
    const int& num_experts,
    const int& num_topk,
    const int& hidden,
    const int& intermediate_hidden,
    const int& expected_num_tokens_per_rank,
    const int& num_experts_per_rank,
    const bool& use_fp8_weights,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const int& num_sms,
    const bool& use_nvfp4,
    const std::optional<torch::Tensor>& l1_alphas,
    const std::optional<torch::Tensor>& l2_alphas,
    const std::optional<torch::Tensor>& l2_act_scales) {
    const auto num_lanes = static_cast<int>(sym_buffers.size());
    DG_HOST_ASSERT(num_lanes >= 1 and num_lanes <= 4);
    DG_HOST_ASSERT(sym_buffer_ptrs.size() == sym_buffers.size());
    DG_HOST_ASSERT(rank_idx >= num_ag_ranks);
    DG_HOST_ASSERT(rank_idx < num_ag_ranks + num_eg_ranks);
    DG_HOST_ASSERT(num_experts_per_rank * num_eg_ranks == num_experts);
    DG_HOST_ASSERT(weight_descs.is_cuda());
    DG_HOST_ASSERT(weight_descs.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(weight_descs.is_contiguous());
    DG_HOST_ASSERT(weight_descs.dim() == 3);
    DG_HOST_ASSERT(weight_descs.size(1) == 4 and weight_descs.size(2) == 128);

    if (use_nvfp4) {
        DG_HOST_ASSERT(not use_fp8_weights);
        DG_HOST_ASSERT(l1_alphas.has_value() and l2_alphas.has_value() and l2_act_scales.has_value());
        for (const auto& scales : {*l1_alphas, *l2_alphas, *l2_act_scales}) {
            DG_HOST_ASSERT(scales.is_cuda() and scales.scalar_type() == torch::kFloat);
            DG_HOST_ASSERT(scales.is_contiguous() and scales.size(0) == weight_descs.size(0));
            DG_HOST_ASSERT(scales.size(1) == num_experts_per_rank);
        }
        DG_HOST_ASSERT(l1_alphas->dim() == 3 and l1_alphas->size(2) == 2);
        DG_HOST_ASSERT(l2_alphas->dim() == 2 and l2_act_scales->dim() == 2);
    }

    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    const auto [num_required_bytes, num_max_pool_tokens, slice] =
        get_symm_buffer_size_for_mega_moe_m2n(
            num_ag_ranks,
            num_eg_ranks,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden, use_nvfp4);

    std::vector<std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>>
        views;
    views.reserve(num_lanes);
    for (int lane = 0; lane < num_lanes; ++lane) {
        DG_HOST_ASSERT(
            sym_buffers[lane].nbytes() >=
            static_cast<size_t>(num_required_bytes));
        DG_HOST_ASSERT(
            static_cast<int>(sym_buffer_ptrs[lane].size()) ==
            num_ag_ranks + num_eg_ranks);
        views.emplace_back(slice(sym_buffers[lane]));
    }
    while (static_cast<int>(views.size()) < 4) {
        views.push_back(views.front());
    }

    if (jit->device.get_arch_major() == 10) {
        sm100_mega_moe_m2n_eg(
            std::get<4>(views[0]),
            std::get<5>(views[0]),
            std::get<6>(views[0]),
            std::get<7>(views[0]),
            std::get<4>(views[1]),
            std::get<5>(views[1]),
            std::get<6>(views[1]),
            std::get<7>(views[1]),
            std::get<4>(views[2]),
            std::get<5>(views[2]),
            std::get<6>(views[2]),
            std::get<7>(views[2]),
            std::get<4>(views[3]),
            std::get<5>(views[3]),
            std::get<6>(views[3]),
            std::get<7>(views[3]),
            weight_descs,
            static_cast<int>(weight_descs.size(0)),
            sym_buffer_ptrs[0],
            sym_buffer_ptrs[num_lanes > 1 ? 1 : 0],
            sym_buffer_ptrs[num_lanes > 2 ? 2 : 0],
            sym_buffer_ptrs[num_lanes > 3 ? 3 : 0],
            num_lanes,
            rank_idx,
            num_ag_ranks,
            num_eg_ranks,
            num_max_tokens_per_rank,
            num_experts_per_rank,
            expected_num_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            activation_clamp,
            fast_math,
            use_fp8_weights,
            num_sms,
            use_nvfp4,
            l1_alphas.has_value() ? l1_alphas->const_data_ptr<float>() : nullptr,
            l2_alphas.has_value() ? l2_alphas->const_data_ptr<float>() : nullptr,
            l2_act_scales.has_value() ? l2_act_scales->const_data_ptr<float>() : nullptr);
        return;
    }
    DG_HOST_UNREACHABLE("Unsupported architecture");
}

}  // namespace deep_gemm::mega_m2n
