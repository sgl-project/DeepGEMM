#pragma once

#include <format>
#include <torch/torch.h>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../heuristics/mega_moe.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// Stored row `32*i3 + 8*i1 + i2` lands at slot `32*i3 + 4*i2 + i1`: TRT's
// 8x4 -> 4x8 L2 transpose, inverted by the axis order alone.
static CUtensorMap make_trtllm_fp4_l2_tma_desc(
    const torch::Tensor& weights, const int inner_bytes, const int rows) {
    DG_HOST_ASSERT(weights.is_contiguous() and rows % 128 == 0);
    DG_HOST_ASSERT(inner_bytes % 128 == 0);
    CUtensorMap result;
    const cuuint64_t dims[4] = {static_cast<cuuint64_t>(inner_bytes), 4, 8,
                                static_cast<cuuint64_t>(rows / 32)};
    const cuuint64_t strides[3] = {8ull * inner_bytes,
                                   static_cast<cuuint64_t>(inner_bytes), 32ull * inner_bytes};
    const cuuint32_t box[4] = {128, 4, 8, 4};
    const cuuint32_t element_strides[4] = {1, 1, 1, 1};
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, weights.data_ptr(), dims,
        strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return result;
}

// TRT-LLM stores weight SFs as [row block][k chunk][128 words]. L1's 128-word UTCCP
// group is identical, so only the outer axes swap; L2's also carries TRT's 32-row
// shuffle, at `slot = a + 4*(b>>2) + 32*(b&3)`.
static CUtensorMap make_trtllm_fp4_sf_tma_desc(
    const torch::Tensor& sf, const bool& is_l1,
    const int& num_row_blocks, const int& num_k_chunks,
    const int& box_k_chunks, const int& box_row_blocks) {
    DG_HOST_ASSERT(num_row_blocks > 0 and num_k_chunks > 0);
    DG_HOST_ASSERT(box_k_chunks <= num_k_chunks and box_row_blocks <= num_row_blocks);
    CUtensorMap result;
    const auto row_block_stride = static_cast<cuuint64_t>(num_k_chunks) * 512ull;
    const cuuint32_t element_strides[5] = {1, 1, 1, 1, 1};
    if (is_l1) {
        const cuuint64_t dims[3] = {128, static_cast<cuuint64_t>(num_k_chunks),
                                    static_cast<cuuint64_t>(num_row_blocks)};
        const cuuint64_t strides[2] = {512ull, row_block_stride};
        const cuuint32_t box[3] = {128, static_cast<cuuint32_t>(box_k_chunks),
                                   static_cast<cuuint32_t>(box_row_blocks)};
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
            &result, CU_TENSOR_MAP_DATA_TYPE_UINT32, 3, sf.data_ptr(), dims,
            strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    } else {
        const cuuint64_t dims[5] = {4, 4, 8, static_cast<cuuint64_t>(num_k_chunks),
                                    static_cast<cuuint64_t>(num_row_blocks)};
        const cuuint64_t strides[4] = {128ull, 16ull, 512ull, row_block_stride};
        const cuuint32_t box[5] = {4, 4, 8, static_cast<cuuint32_t>(box_k_chunks),
                                   static_cast<cuuint32_t>(box_row_blocks)};
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
            &result, CU_TENSOR_MAP_DATA_TYPE_UINT32, 5, sf.data_ptr(), dims,
            strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }
    return result;
}

static void sm100_fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& shared_l1_acts, const torch::Tensor& shared_l1_acts_sf,
    const torch::Tensor& shared_l2_acts, const torch::Tensor& shared_l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const torch::Tensor& shared_l1_weights, const torch::Tensor& shared_l2_weights,
    const torch::Tensor& shared_l1_weights_sf, const torch::Tensor& shared_l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const float& swiglu_alpha,
    const bool& use_situ,
    const bool& fast_math,
    const bool& use_x_scales,
    const float* l1_alphas,
    const float* l2_alphas,
    const float* l2_act_scales,
    const MmaKind& mma_kind,
    const bool& use_fp8_combine,
    const bool& use_trtllm_weights = false,
    // Bit 0: read L1's weight SFs from TRT-LLM storage, bit 1: L2's.
    const int& trtllm_sf_mask = 0
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens,
        mma_kind);

    if (use_trtllm_weights)
        DG_HOST_ASSERT(mma_kind == MmaKind::NVFP4 and num_shared_experts == 0);
    if (trtllm_sf_mask != 0)
        DG_HOST_ASSERT(use_trtllm_weights and config.block_n % 128 == 0 and
                       config.block_k % (config.gran_k * 4) == 0);

    // Make tensormap
    const bool is_packed_fp4 = mma_kind == MmaKind::NVFP4 or mma_kind == MmaKind::MXFP4;
    const int kGranK = config.gran_k;
    const int elem_bits = get_element_bits(mma_kind);
    const auto to_inner = [=](const int& num_elems) { return num_elems * elem_bits / 8; };
    const auto weight_tensor = [=](const torch::Tensor& t) { return is_packed_fp4 ? t.view(torch::kUInt8) : t; };
    const auto weight_inner = [=](const int& num_elems) { return is_packed_fp4 ? num_elems / 2 : num_elems; };
    const int block_k_inner = to_inner(config.block_k);
    const int sf_smem_outer_dim = config.block_k / (kGranK * 4);
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     to_inner(hidden), config.num_ring_tokens,
                                                     block_k_inner, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_sf_ring_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_l1_weights = make_tma_2d_desc(weight_tensor(l1_weights),
                                                        weight_inner(hidden), num_experts_per_rank * intermediate_hidden * 2,
                                                        block_k_inner, config.load_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode, 0, false, not is_packed_fp4);
    const auto tensor_map_l1_weights_sf = (trtllm_sf_mask & 1) ? make_trtllm_fp4_sf_tma_desc(
        l1_weights_sf, true, num_experts_per_rank * intermediate_hidden * 2 / 128,
        hidden / (kGranK * 4), sf_smem_outer_dim, config.block_n / 128) :
        make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                           sf_smem_outer_dim);
    // NOTES: L1 output and L2 activations are essentially the same tensor.
    // Post-SwiGLU output has half the N width (`BLOCK_N / 2` per input tile),
    // so the swizzle mode is also halved (128 -> 64).
    const int l1_out_block_n_bytes = to_inner(config.block_n / 2);
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       to_inner(intermediate_hidden), config.num_ring_tokens,
                                                       l1_out_block_n_bytes, config.store_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       l1_out_block_n_bytes);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     to_inner(intermediate_hidden), config.num_ring_tokens,
                                                     block_k_inner, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_sf_ring_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_l2_weights = use_trtllm_weights ? make_trtllm_fp4_l2_tma_desc(
        l2_weights, weight_inner(intermediate_hidden), num_experts_per_rank * hidden) :
        make_tma_2d_desc(weight_tensor(l2_weights),
                                                        weight_inner(intermediate_hidden), num_experts_per_rank * hidden,
                                                        block_k_inner, config.load_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode, 0, false, not is_packed_fp4);
    const auto tensor_map_l2_weights_sf = (trtllm_sf_mask & 2) ? make_trtllm_fp4_sf_tma_desc(
        l2_weights_sf, false, num_experts_per_rank * hidden / 128,
        intermediate_hidden / (kGranK * 4), sf_smem_outer_dim, config.block_n / 128) :
        make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                           hidden, intermediate_hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                        sf_smem_outer_dim);

    const auto tensor_map_shared_l1_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_acts,
        to_inner(hidden), num_max_tokens_per_rank,
        block_k_inner, config.load_block_m,
        static_cast<int>(shared_l1_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_acts_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_acts_sf,
        static_cast<int>(shared_l1_acts_sf.size(0)), hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l1_acts_sf;
    const auto tensor_map_shared_l1_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        weight_tensor(shared_l1_weights),
        to_inner(hidden), shared_intermediate_hidden * 2,
        block_k_inner, config.load_block_n,
        static_cast<int>(shared_l1_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l1_weights;
    const auto tensor_map_shared_l1_weights_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_weights_sf,
        shared_intermediate_hidden * 2, hidden,
        config.block_n, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l1_weights_sf;
    const auto tensor_map_shared_l1_output = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        to_inner(shared_intermediate_hidden), num_max_tokens_per_rank,
        l1_out_block_n_bytes, config.store_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        l1_out_block_n_bytes) : tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        to_inner(shared_intermediate_hidden), num_max_tokens_per_rank,
        block_k_inner, config.load_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_acts_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_acts_sf,
        static_cast<int>(shared_l2_acts_sf.size(0)), shared_intermediate_hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l2_acts_sf;
    const auto tensor_map_shared_l2_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        weight_tensor(shared_l2_weights),
        to_inner(shared_intermediate_hidden), hidden,
        block_k_inner, config.load_block_n,
        static_cast<int>(shared_l2_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l2_weights;
    const auto tensor_map_shared_l2_weights_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_weights_sf,
        hidden, shared_intermediate_hidden,
        config.block_n, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l2_weights_sf;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    const auto num_sms = runtime->get_num_sms();

    // Compile
    const auto kernel = jit->compile("sm100_fp8_fp4_mega_moe", std::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {}, {},
        {},
        {},
        {}, {}, {}, {}, {}, {},
        {}, {}
    >);
}};
)", num_max_tokens_per_rank,
    hidden, intermediate_hidden,
    num_experts, num_shared_experts,
    num_topk,
    config.block_m, config.block_n, config.block_k,
    config.store_block_m,
    config.sf_block_m, config.sf_block_n,
    to_mma_kind_name(mma_kind),
    config.num_ring_tokens,
    config.num_sf_ring_tokens,
    config.num_stages,
    config.num_bytes_per_pull,
    config.num_dispatch_threads, config.num_non_epilogue_threads, config.num_epilogue_threads,
    num_sms, num_ranks,
    to_string(activation_clamp), to_string(swiglu_alpha),
    use_situ ? "true" : "false",
    fast_math ? "true" : "false",
    use_x_scales ? "true" : "false",
    (l1_alphas != nullptr) ? "true" : "false",
    (l2_alphas != nullptr) ? "true" : "false",
    (l2_act_scales != nullptr) ? "true" : "false",
    use_fp8_combine ? "true" : "false",
    to_string(l1_weights.scalar_type()),
    use_trtllm_weights ? "true" : "false",
    trtllm_sf_mask));
    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads, 1, 1),
            .cluster_dim = dim3(2, 1, 1),
        },
        y.data_ptr(),
        cumulative_local_expert_recv_stats_ptr,
        l1_alphas, l2_alphas, l2_act_scales,
        num_tokens,
        layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        tensor_map_l1_acts,
        tensor_map_l1_acts_sf,
        tensor_map_l1_weights,
        tensor_map_l1_weights_sf,
        tensor_map_l1_output,
        tensor_map_l2_acts,
        tensor_map_l2_acts_sf,
        tensor_map_l2_weights,
        tensor_map_l2_weights_sf,
        tensor_map_shared_l1_acts,
        tensor_map_shared_l1_acts_sf,
        tensor_map_shared_l1_weights,
        tensor_map_shared_l1_weights_sf,
        tensor_map_shared_l1_output,
        tensor_map_shared_l2_acts,
        tensor_map_shared_l2_acts_sf,
        tensor_map_shared_l2_weights,
        tensor_map_shared_l2_weights_sf);
}

} // namespace deep_gemm
