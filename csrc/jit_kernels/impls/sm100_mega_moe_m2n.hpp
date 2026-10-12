#pragma once

#include <torch/python.h>

#include "../../runtime/launch.hpp"
#include <format>
#include "../../utils/exception.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/mega_moe_m2n.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe_m2n.hpp"

namespace deep_gemm {

// ---------------------------------------------------------------------------
// AG kernel runtime (send + recv only; no GEMM, no cluster)
// ---------------------------------------------------------------------------
class SM100MegaMoEM2NAGRuntime final {
public:
    struct Args {
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ag_ranks, num_eg_ranks;
        int num_max_pool_tokens, num_padded_sf_pool_tokens;
        int num_dispatch_threads, num_combine_threads;
        int num_sms;
        bool topk_idx_is_64;
        bool use_nvfp4;
        const float* input_scale;

        void* y;
        const void* hidden_ptr;
        const void* topk_ids_ptr;
        const void* topk_weights_ptr;
        int num_tokens;
        float output_scale;
        layout::SymBuffer<> sym_buffer_ptrs;

        deep_jit::cuda::LaunchOptions launch_options;
    };

    static std::string generate_impl(const Args& args) {
        return std::format(R"(
#include <deep_gemm/impls/sm100_mega_moe_m2n.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_mega_moe_m2n_ag_impl<
        {},
        {}, {},
        {}, {},
        {},
        {},
        {}, {},
        {},
        {}, {},
        {}, {}
    >);
}};
)", args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.num_max_pool_tokens,
    args.num_padded_sf_pool_tokens,
    args.num_dispatch_threads, args.num_combine_threads,
    args.num_sms,
    args.num_ag_ranks, args.num_eg_ranks,
    args.topk_idx_is_64 ? "true" : "false",
    args.use_nvfp4 ? "true" : "false");
    }

    static void launch(const std::shared_ptr<deep_jit::cuda::Kernel>& kernel, const Args& args) {
        jit->launch(kernel, args.launch_options,
            args.y,
            static_cast<const nv_bfloat16*>(args.hidden_ptr),
            args.topk_ids_ptr,
            static_cast<const float*>(args.topk_weights_ptr),
            args.num_tokens,
            args.output_scale,
            args.input_scale,
            args.sym_buffer_ptrs
        );
    }
};

static std::tuple<int, int> get_ag_combine_smem_config_m2n(
    const int& hidden, const int& num_experts,
    const int& num_combine_threads) {
    constexpr int kSmemAlignment = 1024;
    const int num_combine_warps = num_combine_threads / 32;
    const int num_hidden_bytes = hidden * 2;
    // Preserve the chunk size to keep combine accumulator register use unchanged.
    const int num_chunks = (3 * num_combine_warps * num_hidden_bytes <= 160 * 1024) ? 1 : 2;
    const int chunk_bytes = num_hidden_bytes / num_chunks;
    const int smem_chunks = align(2 * num_combine_warps * chunk_bytes, kSmemAlignment);
    const int smem_expert_count = align(num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_barriers = num_combine_warps * 2 * 8;
    return {num_chunks, smem_chunks + smem_expert_count + smem_barriers};
}

static void sm100_mega_moe_m2n_ag(
    const torch::Tensor& y,
    const torch::Tensor& hidden_states,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx,
    const int& num_ag_ranks, const int& num_eg_ranks,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_max_pool_tokens, const int& num_padded_sf_pool_tokens,
    const float& output_scale,
    const int& num_sms_opt,
    const bool& use_nvfp4,
    const float* input_scale) {
    constexpr int kNumDispatchThreads = 128;
    constexpr int kNumCombineThreads = 256;
    const auto num_sms = num_sms_opt > 0
        ? std::min(num_sms_opt, jit->device.get_num_sms())
        : jit->device.get_num_sms();
    const auto [num_chunks, smem_size] = get_ag_combine_smem_config_m2n(
        hidden, num_experts, kNumCombineThreads);

    const SM100MegaMoEM2NAGRuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ag_ranks = num_ag_ranks, .num_eg_ranks = num_eg_ranks,
        .num_max_pool_tokens = num_max_pool_tokens,
        .num_padded_sf_pool_tokens = num_padded_sf_pool_tokens,
        .num_dispatch_threads = kNumDispatchThreads,
        .num_combine_threads = kNumCombineThreads,
        .num_sms = num_sms,
        .topk_idx_is_64 = topk_ids.scalar_type() == torch::kInt64,
        .use_nvfp4 = use_nvfp4,
        .input_scale = input_scale,
        .y = y.data_ptr(),
        .hidden_ptr = hidden_states.data_ptr(),
        .topk_ids_ptr = topk_ids.data_ptr(),
        .topk_weights_ptr = topk_weights.data_ptr(),
        .num_tokens = num_tokens,
        .output_scale = output_scale,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .launch_options = make_launch_options(num_sms,
                                  kNumDispatchThreads + kNumCombineThreads,
                                  smem_size, 1)
    };

    const auto code = SM100MegaMoEM2NAGRuntime::generate_impl(args);
    const auto kernel = jit->compile("sm100_mega_moe_m2n_ag", code);
    SM100MegaMoEM2NAGRuntime::launch(kernel, args);
}

// ---------------------------------------------------------------------------
// EG kernel runtime (recv + experts + send)
// ---------------------------------------------------------------------------
class SM100MegaMoEM2NEGRuntime final {
public:
    struct Args {
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ag_ranks, num_eg_ranks;
        float activation_clamp;
        bool fast_math;
        bool use_fp8_weights;
        bool use_nvfp4;
        const float* l1_alphas;
        const float* l2_alphas;
        const float* l2_act_scales;
        MegaMoEM2NConfig config;

        int num_lanes;
        layout::SymBuffer<> sym_buffer_ptrs;
        layout::SymBuffer<> sym_buffer_ptrs1;
        layout::SymBuffer<> sym_buffer_ptrs2;
        layout::SymBuffer<> sym_buffer_ptrs3;

        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l1_acts_1;
        CUtensorMap tensor_map_l1_acts_sf_1;
        CUtensorMap tensor_map_l1_output_1;
        CUtensorMap tensor_map_l2_acts_1;
        CUtensorMap tensor_map_l2_acts_sf_1;
        CUtensorMap tensor_map_l1_acts_2;
        CUtensorMap tensor_map_l1_acts_sf_2;
        CUtensorMap tensor_map_l1_output_2;
        CUtensorMap tensor_map_l2_acts_2;
        CUtensorMap tensor_map_l2_acts_sf_2;
        CUtensorMap tensor_map_l1_acts_3;
        CUtensorMap tensor_map_l1_acts_sf_3;
        CUtensorMap tensor_map_l1_output_3;
        CUtensorMap tensor_map_l2_acts_3;
        CUtensorMap tensor_map_l2_acts_sf_3;
        const void* weight_descs;       // gmem [num_layers][4] CUtensorMap
        int num_layers;

        deep_jit::cuda::LaunchOptions launch_options;
    };

    static std::string generate_impl(const Args& args) {
        return std::format(R"(
#include <deep_gemm/impls/sm100_mega_moe_m2n.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_mega_moe_m2n_eg_impl<
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {},
        {}, {}
    >);
}};
)", args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.store_block_m,
    args.config.sf_block_m, args.config.sf_block_n,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.config.num_dispatch_pull_stages,
    args.launch_options.grid_dim->x,
    args.num_ag_ranks, args.num_eg_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    args.use_fp8_weights ? "true" : "false",
    args.num_lanes,
    args.use_nvfp4 ? "true" : "false");
    }

    static void launch(const std::shared_ptr<deep_jit::cuda::Kernel>& kernel, const Args& args) {
        jit->launch(kernel, args.launch_options,
            args.sym_buffer_ptrs,
            args.sym_buffer_ptrs1,
            args.sym_buffer_ptrs2,
            args.sym_buffer_ptrs3,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l1_acts_1,
            args.tensor_map_l1_acts_sf_1,
            args.tensor_map_l1_output_1,
            args.tensor_map_l2_acts_1,
            args.tensor_map_l2_acts_sf_1,
            args.tensor_map_l1_acts_2,
            args.tensor_map_l1_acts_sf_2,
            args.tensor_map_l1_output_2,
            args.tensor_map_l2_acts_2,
            args.tensor_map_l2_acts_sf_2,
            args.tensor_map_l1_acts_3,
            args.tensor_map_l1_acts_sf_3,
            args.tensor_map_l1_output_3,
            args.tensor_map_l2_acts_3,
            args.tensor_map_l2_acts_sf_3,
            static_cast<const CUtensorMap*>(args.weight_descs),
            static_cast<uint32_t>(args.num_layers),
            args.l1_alphas,
            args.l2_alphas,
            args.l2_act_scales
        );
    }
};

// Build the 4 per-layer weight tensormaps (l1_w, l1_w_sf, l2_w, l2_w_sf) as
// a 4x128-byte CPU blob.  The weight descriptors are CONFIG-INDEPENDENT
// (128-byte K tiles, BLOCK_N = swizzle = 128 within each MMA type),
// so they can be built once at weight-registration time.
static torch::Tensor make_m2n_weight_descs(
    const torch::Tensor& l1_weights, const torch::Tensor& l1_weights_sf,
    const torch::Tensor& l2_weights, const torch::Tensor& l2_weights_sf,
    const int& num_experts_per_rank,
    const int& hidden, const int& intermediate_hidden,
    const bool& use_nvfp4) {
    constexpr int kBlockK = 128, kLoadBlockN = 128, kBlockN = 128, kSwizzleW = 128;
    const auto weight_tensor = [=](const torch::Tensor& t) { return use_nvfp4 ? t.view(torch::kUInt8) : t; };
    const int packing = use_nvfp4 ? 2 : 1;
    const int gran_k = use_nvfp4 ? 16 : 32;
    const int sf_chunks = use_nvfp4 ? 4 : 1;
    const auto tm_l1_w = make_tma_2d_desc(weight_tensor(l1_weights),
                                          hidden / packing, num_experts_per_rank * intermediate_hidden * 2,
                                          kBlockK, kLoadBlockN,
                                          static_cast<int>(l1_weights.stride(-2)),
                                          kSwizzleW);
    const auto tm_l1_w_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                             intermediate_hidden * 2, hidden,
                                             kBlockN, gran_k, num_experts_per_rank, 0, 0, false, sf_chunks);
    const auto tm_l2_w = make_tma_2d_desc(weight_tensor(l2_weights),
                                          intermediate_hidden / packing, num_experts_per_rank * hidden,
                                          kBlockK, kLoadBlockN,
                                          static_cast<int>(l2_weights.stride(-2)),
                                          kSwizzleW);
    const auto tm_l2_w_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                             hidden, intermediate_hidden,
                                             kBlockN, gran_k, num_experts_per_rank, 0, 0, false, sf_chunks);
    auto blob = torch::empty({4, 128}, torch::TensorOptions().dtype(torch::kUInt8));
    static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap must be 128B");
    std::memcpy(blob.data_ptr<uint8_t>() + 0 * 128, &tm_l1_w, 128);
    std::memcpy(blob.data_ptr<uint8_t>() + 1 * 128, &tm_l1_w_sf, 128);
    std::memcpy(blob.data_ptr<uint8_t>() + 2 * 128, &tm_l2_w, 128);
    std::memcpy(blob.data_ptr<uint8_t>() + 3 * 128, &tm_l2_w_sf, 128);
    return blob;
}

static void sm100_mega_moe_m2n_eg(
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_acts1, const torch::Tensor& l1_acts_sf1,
    const torch::Tensor& l2_acts1, const torch::Tensor& l2_acts_sf1,
    const torch::Tensor& l1_acts2, const torch::Tensor& l1_acts_sf2,
    const torch::Tensor& l2_acts2, const torch::Tensor& l2_acts_sf2,
    const torch::Tensor& l1_acts3, const torch::Tensor& l1_acts_sf3,
    const torch::Tensor& l2_acts3, const torch::Tensor& l2_acts_sf3,
    const torch::Tensor& weight_descs,    // device uint8 [num_layers, 4, 128]
    const int& num_layers,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const std::vector<int64_t>& sym_buffer_ptrs1,
    const std::vector<int64_t>& sym_buffer_ptrs2,
    const std::vector<int64_t>& sym_buffer_ptrs3,
    const int& num_lanes,
    const int& rank_idx,
    const int& num_ag_ranks, const int& num_eg_ranks,
    const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& expected_num_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const bool& use_fp8_weights,
    const int& num_sms_opt,
    const bool& use_nvfp4,
    const float* l1_alphas,
    const float* l2_alphas,
    const float* l2_act_scales
) {
    const auto num_experts = num_experts_per_rank * num_eg_ranks;
    const auto num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));

    // A sub-device grid lets two microbatch lanes co-run their fused kernels;
    // 2-CTA clusters require an even SM count.
    auto num_sms = num_sms_opt > 0
        ? std::min(num_sms_opt, jit->device.get_num_sms())
        : jit->device.get_num_sms();
    num_sms = std::max(num_sms / 2 * 2, 4);

    // Keep fixed M64 while the task scheduler interleaves L1 and L2.
    const auto config = get_mega_moe_m2n_config(
        num_ag_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, expected_num_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens, use_nvfp4);

    const int packing = use_nvfp4 ? 2 : 1;
    const int gran_k = use_nvfp4 ? 16 : 32;
    const int sf_chunks = config.block_k / (gran_k * 4);

    // Make tensormaps (same as the symmetric kernel)
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden / packing, config.num_max_pool_tokens,
                                                     config.block_k / packing, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_padded_sf_pool_tokens, hidden,
                                                        config.sf_block_m, gran_k,
                                                        1, 0, 0, false, sf_chunks);
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden / packing, config.num_max_pool_tokens,
                                                       config.block_n / (2 * packing), config.store_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       config.swizzle_acts_mode / (2 * packing));
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden / packing, config.num_max_pool_tokens,
                                                     config.block_k / packing, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_padded_sf_pool_tokens, intermediate_hidden,
                                                        config.sf_block_m, gran_k,
                                                        1, 0, 0, false, sf_chunks);

    // Lane-1 activation/output maps (weights are shared across lanes)
    const auto tensor_map_l1_acts_1 = make_tma_2d_desc(l1_acts1,
                                                       hidden / packing, config.num_max_pool_tokens,
                                                       config.block_k / packing, config.load_block_m,
                                                       static_cast<int>(l1_acts1.stride(-2)),
                                                       config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf_1 = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf1,
                                                          config.num_padded_sf_pool_tokens, hidden,
                                                          config.sf_block_m, gran_k,
                                                          1, 0, 0, false, sf_chunks);
    const auto tensor_map_l1_output_1 = make_tma_2d_desc(l2_acts1,
                                                         intermediate_hidden / packing, config.num_max_pool_tokens,
                                                         config.block_n / (2 * packing), config.store_block_m,
                                                         static_cast<int>(l2_acts1.stride(-2)),
                                                         config.swizzle_acts_mode / (2 * packing));
    const auto tensor_map_l2_acts_1 = make_tma_2d_desc(l2_acts1,
                                                       intermediate_hidden / packing, config.num_max_pool_tokens,
                                                       config.block_k / packing, config.load_block_m,
                                                       static_cast<int>(l2_acts1.stride(-2)),
                                                       config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf_1 = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf1,
                                                          config.num_padded_sf_pool_tokens, intermediate_hidden,
                                                          config.sf_block_m, gran_k,
                                                          1, 0, 0, false, sf_chunks);
    const auto tensor_map_l1_acts_2 = make_tma_2d_desc(l1_acts2,
                                                       hidden / packing, config.num_max_pool_tokens,
                                                       config.block_k / packing, config.load_block_m,
                                                       static_cast<int>(l1_acts2.stride(-2)),
                                                       config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf_2 = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf2,
                                                          config.num_padded_sf_pool_tokens, hidden,
                                                          config.sf_block_m, gran_k,
                                                          1, 0, 0, false, sf_chunks);
    const auto tensor_map_l1_output_2 = make_tma_2d_desc(l2_acts2,
                                                         intermediate_hidden / packing, config.num_max_pool_tokens,
                                                         config.block_n / (2 * packing), config.store_block_m,
                                                         static_cast<int>(l2_acts2.stride(-2)),
                                                         config.swizzle_acts_mode / (2 * packing));
    const auto tensor_map_l2_acts_2 = make_tma_2d_desc(l2_acts2,
                                                       intermediate_hidden / packing, config.num_max_pool_tokens,
                                                       config.block_k / packing, config.load_block_m,
                                                       static_cast<int>(l2_acts2.stride(-2)),
                                                       config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf_2 = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf2,
                                                          config.num_padded_sf_pool_tokens, intermediate_hidden,
                                                          config.sf_block_m, gran_k,
                                                          1, 0, 0, false, sf_chunks);
    const auto tensor_map_l1_acts_3 = make_tma_2d_desc(l1_acts3,
                                                       hidden / packing, config.num_max_pool_tokens,
                                                       config.block_k / packing, config.load_block_m,
                                                       static_cast<int>(l1_acts3.stride(-2)),
                                                       config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf_3 = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf3,
                                                          config.num_padded_sf_pool_tokens, hidden,
                                                          config.sf_block_m, gran_k,
                                                          1, 0, 0, false, sf_chunks);
    const auto tensor_map_l1_output_3 = make_tma_2d_desc(l2_acts3,
                                                         intermediate_hidden / packing, config.num_max_pool_tokens,
                                                         config.block_n / (2 * packing), config.store_block_m,
                                                         static_cast<int>(l2_acts3.stride(-2)),
                                                         config.swizzle_acts_mode / (2 * packing));
    const auto tensor_map_l2_acts_3 = make_tma_2d_desc(l2_acts3,
                                                       intermediate_hidden / packing, config.num_max_pool_tokens,
                                                       config.block_k / packing, config.load_block_m,
                                                       static_cast<int>(l2_acts3.stride(-2)),
                                                       config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf_3 = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf3,
                                                          config.num_padded_sf_pool_tokens, intermediate_hidden,
                                                          config.sf_block_m, gran_k,
                                                          1, 0, 0, false, sf_chunks);

    const SM100MegaMoEM2NEGRuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ag_ranks = num_ag_ranks, .num_eg_ranks = num_eg_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .use_fp8_weights = use_fp8_weights,
        .use_nvfp4 = use_nvfp4,
        .l1_alphas = l1_alphas,
        .l2_alphas = l2_alphas,
        .l2_act_scales = l2_act_scales,
        .config = config,
        .num_lanes = num_lanes,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .sym_buffer_ptrs1 = layout::SymBuffer<>(sym_buffer_ptrs1, rank_idx),
        .sym_buffer_ptrs2 = layout::SymBuffer<>(sym_buffer_ptrs2, rank_idx),
        .sym_buffer_ptrs3 = layout::SymBuffer<>(sym_buffer_ptrs3, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l1_acts_1 = tensor_map_l1_acts_1,
        .tensor_map_l1_acts_sf_1 = tensor_map_l1_acts_sf_1,
        .tensor_map_l1_output_1 = tensor_map_l1_output_1,
        .tensor_map_l2_acts_1 = tensor_map_l2_acts_1,
        .tensor_map_l2_acts_sf_1 = tensor_map_l2_acts_sf_1,
        .tensor_map_l1_acts_2 = tensor_map_l1_acts_2,
        .tensor_map_l1_acts_sf_2 = tensor_map_l1_acts_sf_2,
        .tensor_map_l1_output_2 = tensor_map_l1_output_2,
        .tensor_map_l2_acts_2 = tensor_map_l2_acts_2,
        .tensor_map_l2_acts_sf_2 = tensor_map_l2_acts_sf_2,
        .tensor_map_l1_acts_3 = tensor_map_l1_acts_3,
        .tensor_map_l1_acts_sf_3 = tensor_map_l1_acts_sf_3,
        .tensor_map_l1_output_3 = tensor_map_l1_output_3,
        .tensor_map_l2_acts_3 = tensor_map_l2_acts_3,
        .tensor_map_l2_acts_sf_3 = tensor_map_l2_acts_sf_3,
        .weight_descs = weight_descs.data_ptr(),
        .num_layers = num_layers,
        .launch_options = make_launch_options(num_sms,
                                  config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, 2)
    };

    const auto code = SM100MegaMoEM2NEGRuntime::generate_impl(args);
    const auto kernel = jit->compile("sm100_mega_moe_m2n_eg", code);
    SM100MegaMoEM2NEGRuntime::launch(kernel, args);
}

} // namespace deep_gemm
