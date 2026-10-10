#pragma once

#include "../apis/mega_m2n.hpp"
#include "../utils/m2n_memory.hpp"

Tuple<Tensor, Tensor> dg_allocate_m2n_buffer(int64_t num_bytes) {
    auto [buffer, metadata] = m2n_memory::allocate_buffer(num_bytes);
    return Tuple<Tensor, Tensor>(
        Tensor::FromDLPack(at::toDLPack(buffer)),
        Tensor::FromDLPack(at::toDLPack(metadata)));
}

Tensor dg_import_m2n_buffer(TensorView metadata) {
    DG_HOST_ASSERT(metadata.device().device_type == kDLCPU);
    DG_HOST_ASSERT(metadata.dtype().code == kDLInt && metadata.dtype().bits == 64 && metadata.dtype().lanes == 1);
    DG_HOST_ASSERT(metadata.ndim() == 1 && metadata.size(0) == 10);
    DG_HOST_ASSERT(!metadata.strides().data() || metadata.strides()[0] == 1);
    m2n_memory::Metadata values;
    std::memcpy(values.data(), static_cast<const char*>(metadata.data_ptr()) + metadata.byte_offset(), sizeof(values));
    auto buffer = m2n_memory::import_buffer(values);
    return Tensor::FromDLPack(at::toDLPack(buffer));
}

static std::vector<int64_t> dg_m2n_ptrs_to_vector(
    const Array<int64_t>& pointers) {
    std::vector<int64_t> result;
    result.reserve(pointers.size());
    for (auto it = pointers.begin(); it != pointers.end(); ++it) {
        result.push_back(*it);
    }
    return result;
}

Tuple<
    int64_t,
    int64_t,
    TypedFunction<Tuple<
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor>(TensorView)>>
dg_get_symm_buffer_size_for_mega_moe_m2n(
    int64_t num_ag_ranks,
    int64_t num_eg_ranks,
    int64_t num_experts,
    int64_t num_max_tokens_per_rank,
    int64_t num_topk,
    int64_t hidden,
    int64_t intermediate_hidden,
    bool use_nvfp4) {
    auto [num_bytes, num_max_pool_tokens, slice] =
        mega_m2n::get_symm_buffer_size_for_mega_moe_m2n(
            static_cast<int>(num_ag_ranks),
            static_cast<int>(num_eg_ranks),
            static_cast<int>(num_experts),
            static_cast<int>(num_max_tokens_per_rank),
            static_cast<int>(num_topk),
            static_cast<int>(hidden),
            static_cast<int>(intermediate_hidden), use_nvfp4);

    auto slice_input_buffers = [=](TensorView buffer) {
        auto [
            x,
            x_sf,
            topk_idx,
            topk_weights,
            l1_acts,
            l1_acts_sf,
            l2_acts,
            l2_acts_sf] = slice(convert_to_torch_tensor(buffer));
        return Tuple<
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor>(
            Tensor::FromDLPack(at::toDLPack(x.view(at::kChar))),
            Tensor::FromDLPack(at::toDLPack(x_sf)),
            Tensor::FromDLPack(at::toDLPack(topk_idx)),
            Tensor::FromDLPack(at::toDLPack(topk_weights)),
            Tensor::FromDLPack(at::toDLPack(l1_acts.view(at::kChar))),
            Tensor::FromDLPack(at::toDLPack(l1_acts_sf)),
            Tensor::FromDLPack(at::toDLPack(l2_acts.view(at::kChar))),
            Tensor::FromDLPack(at::toDLPack(l2_acts_sf)));
    };

    return Tuple<
        int64_t,
        int64_t,
        TypedFunction<Tuple<
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor>(TensorView)>>(
        num_bytes,
        num_max_pool_tokens,
        slice_input_buffers);
}

void dg_initialize_symm_buffer_for_mega_moe_m2n(
    TensorView sym_buffer, int64_t num_ranks) {
    mega_m2n::initialize_symm_buffer_for_mega_moe_m2n(
        convert_to_torch_tensor(sym_buffer), static_cast<int>(num_ranks));
}

void dg_mega_moe_m2n_ag(
    TensorView y,
    TensorView hidden_states,
    TensorView topk_ids,
    TensorView topk_weights,
    TensorView sym_buffer,
    Array<int64_t> sym_buffer_ptrs,
    int64_t rank_idx,
    int64_t num_ag_ranks,
    int64_t num_eg_ranks,
    int64_t num_max_tokens_per_rank,
    int64_t num_experts,
    int64_t num_topk,
    int64_t intermediate_hidden,
    double output_scale,
    int64_t num_sms,
    bool use_nvfp4,
    Optional<TensorView> input_scale) {
    mega_m2n::mega_moe_m2n_ag(
        convert_to_torch_tensor(y),
        convert_to_torch_tensor(hidden_states),
        convert_to_torch_tensor(topk_ids),
        convert_to_torch_tensor(topk_weights),
        convert_to_torch_tensor(sym_buffer),
        dg_m2n_ptrs_to_vector(sym_buffer_ptrs),
        static_cast<int>(rank_idx),
        static_cast<int>(num_ag_ranks),
        static_cast<int>(num_eg_ranks),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_experts),
        static_cast<int>(num_topk),
        static_cast<int>(intermediate_hidden),
        static_cast<float>(output_scale),
        static_cast<int>(num_sms),
        use_nvfp4,
        input_scale.has_value()
            ? std::optional<torch::Tensor>(convert_to_torch_tensor(input_scale.value()))
            : std::nullopt);
}

Tensor dg_make_m2n_weight_descs(
    TensorView l1_weights,
    TensorView l1_weights_sf,
    TensorView l2_weights,
    TensorView l2_weights_sf,
    int64_t num_experts_per_rank,
    int64_t hidden,
    int64_t intermediate_hidden,
    bool use_nvfp4) {
    auto result = make_m2n_weight_descs(
        convert_to_torch_tensor(l1_weights),
        convert_to_torch_tensor(l1_weights_sf),
        convert_to_torch_tensor(l2_weights),
        convert_to_torch_tensor(l2_weights_sf),
        static_cast<int>(num_experts_per_rank),
        static_cast<int>(hidden),
        static_cast<int>(intermediate_hidden), use_nvfp4);
    return Tensor::FromDLPack(at::toDLPack(result));
}

void dg_mega_moe_m2n_eg(
    TensorView weight_descs,
    TensorView sym_buffer0,
    TensorView sym_buffer1,
    TensorView sym_buffer2,
    TensorView sym_buffer3,
    Array<int64_t> sym_buffer_ptrs0,
    Array<int64_t> sym_buffer_ptrs1,
    Array<int64_t> sym_buffer_ptrs2,
    Array<int64_t> sym_buffer_ptrs3,
    int64_t num_lanes,
    int64_t rank_idx,
    int64_t num_ag_ranks,
    int64_t num_eg_ranks,
    int64_t num_max_tokens_per_rank,
    int64_t num_experts,
    int64_t num_topk,
    int64_t hidden,
    int64_t intermediate_hidden,
    int64_t expected_num_tokens_per_rank,
    int64_t num_experts_per_rank,
    bool use_fp8_weights,
    Optional<double> activation_clamp,
    bool fast_math,
    int64_t num_sms,
    bool use_nvfp4,
    Optional<TensorView> l1_alphas,
    Optional<TensorView> l2_alphas,
    Optional<TensorView> l2_act_scales) {
    const std::vector<torch::Tensor> sym_buffers = {
        convert_to_torch_tensor(sym_buffer0),
        convert_to_torch_tensor(sym_buffer1),
        convert_to_torch_tensor(sym_buffer2),
        convert_to_torch_tensor(sym_buffer3)};
    const std::vector<std::vector<int64_t>> pointer_lists = {
        dg_m2n_ptrs_to_vector(sym_buffer_ptrs0),
        dg_m2n_ptrs_to_vector(sym_buffer_ptrs1),
        dg_m2n_ptrs_to_vector(sym_buffer_ptrs2),
        dg_m2n_ptrs_to_vector(sym_buffer_ptrs3)};
    DG_HOST_ASSERT(num_lanes >= 1 and num_lanes <= 4);
    const std::vector<torch::Tensor> active_sym_buffers(
        sym_buffers.begin(), sym_buffers.begin() + num_lanes);
    const std::vector<std::vector<int64_t>> active_pointer_lists(
        pointer_lists.begin(), pointer_lists.begin() + num_lanes);
    const auto activation_clamp_value = activation_clamp.has_value()
        ? std::optional<float>(static_cast<float>(activation_clamp.value()))
        : std::nullopt;

    mega_m2n::mega_moe_m2n_eg(
        convert_to_torch_tensor(weight_descs),
        active_sym_buffers,
        active_pointer_lists,
        static_cast<int>(rank_idx),
        static_cast<int>(num_ag_ranks),
        static_cast<int>(num_eg_ranks),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_experts),
        static_cast<int>(num_topk),
        static_cast<int>(hidden),
        static_cast<int>(intermediate_hidden),
        static_cast<int>(expected_num_tokens_per_rank),
        static_cast<int>(num_experts_per_rank),
        use_fp8_weights,
        activation_clamp_value,
        fast_math,
        static_cast<int>(num_sms),
        use_nvfp4,
        l1_alphas.has_value()
            ? std::optional<torch::Tensor>(convert_to_torch_tensor(l1_alphas.value()))
            : std::nullopt,
        l2_alphas.has_value()
            ? std::optional<torch::Tensor>(convert_to_torch_tensor(l2_alphas.value()))
            : std::nullopt,
        l2_act_scales.has_value()
            ? std::optional<torch::Tensor>(convert_to_torch_tensor(l2_act_scales.value()))
            : std::nullopt);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    allocate_m2n_buffer, dg_allocate_m2n_buffer);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    import_m2n_buffer, dg_import_m2n_buffer);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    get_symm_buffer_size_for_mega_moe_m2n,
    dg_get_symm_buffer_size_for_mega_moe_m2n);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    initialize_symm_buffer_for_mega_moe_m2n,
    dg_initialize_symm_buffer_for_mega_moe_m2n);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_m2n_ag, dg_mega_moe_m2n_ag);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_m2n_weight_descs, dg_make_m2n_weight_descs);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_m2n_eg, dg_mega_moe_m2n_eg);
