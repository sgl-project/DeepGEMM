// F5: Host-side JIT runtime for the packed-FP4 MoE kernel.
//
// Mirrors `SM100FP8FP4MegaMoERuntime` but targets `sm100_fp4_mega_moe_impl`,
// passing packed-FP4 (`fp4_unpacked_smem=false`) TMA descriptors for both A
// and B operands and for L1/L2 activation pools.
//
// Gated by env var DG_MEGA_MOE_FP4=1 at the dispatch site so the default R20
// path remains byte-identical when the flag is unset.
#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/system.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

class SM100FP4MegaMoERuntime final : public LaunchRuntime<SM100FP4MegaMoERuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        MegaMoEConfig config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormap
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_l2_weights_sf;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {}
    >);
}};
)", args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.store_block_m,
    args.config.sf_block_m, args.config.sf_block_n,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.tensor_map_l2_weights_sf
        ));
    }
};

// Build the FP4 MoE TMA descriptors. All A/B/output activation maps use the
// packed-FP4 layout (`fp4_unpacked_smem=false`); weights and SFs use the same
// packed-FP4 layout as the F4 standalone GEMM.
static void sm100_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));

    // Reuse the FP8 MoE heuristic for now (block sizes are the same in element
    // units; the FP4 kernel internally halves SMEM bytes per stage).
    auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden, num_padded_sf_pool_tokens);

    // Packed-FP4 with kind::mxf4 needs load_block_m * block_k / 2 >= 1024 for
    // SMEM alignment. block_m=16 → load_block_m=8 → 8*128/2=512 < 1024 → crash.
    // Bump to block_m=32 (matching sglang PR #27's fix).
    if (config.block_m < 32) {
        config.block_m = 32;
        config.store_block_m = 16;
        config.load_block_m = 16;
        config.load_block_n = config.block_n;
    }

    // R2/R3-close: per-band BLOCK_K default, keyed on the *post-bump* block_m.
    // The 32-band (incl. the bumped 16), 64-band, and 192-band benefit from 256
    // (128B swizzle, halved K-iterations); the 96/128 bands regress, so they
    // keep 128. R3-close added the 64-band (profiler measured +2-6% at ntok
    // 256/320 across two sessions). At BLOCK_K=256 a packed-FP4 row is 128 bytes
    // = one 128B swizzle atom.
    config.block_k = (config.block_m == 32 or config.block_m == 64 or config.block_m == 192) ? 256 : 128;

    // DG_BLOCK_K (128/256) force-overrides the band default, for the in-process
    // A/B. Read per invocation (like DG_NUM_EXPERTS_PER_WAVE); 0/unset keeps the
    // band default above.
    if (const int env_block_k = get_env<int>("DG_BLOCK_K"); env_block_k != 0) {
        DG_HOST_ASSERT(env_block_k == 128 or env_block_k == 256);
        config.block_k = env_block_k;
    }
    DG_HOST_ASSERT(hidden % config.block_k == 0 and intermediate_hidden % config.block_k == 0);

    // The packed-FP4 kernel's swizzle atom is one row: BLOCK_K packed elements =
    // BLOCK_K/2 bytes. BLOCK_K=128 -> 64B swizzle; BLOCK_K=256 -> 128B swizzle.
    // Override the heuristic's value to match.
    config.swizzle_acts_mode = config.swizzle_weights_mode = (config.block_k == 256 ? 128 : 64);

    // F19: recompute pipeline for packed FP4 (0.5 bytes/elem instead of 1).
    // The FP8 heuristic sized A/B at 1 byte/elem; packed FP4 halves that,
    // allowing ~2x more pipeline stages for better TMA/MMA overlap.
    {
        constexpr int kSmemAlignment = 1024;
        constexpr int kNumEpilogueStages = 2;
        constexpr int kNumTMAStoreStages = 2;
        const int load_block_m = config.block_m / 2;
        const int num_dispatch_warps = config.num_dispatch_threads / 32;
        const int num_epilogue_warps = config.num_epilogue_threads / 32;
        const int num_epilogue_warpgroups = num_epilogue_warps / 4;

        // Dispatch region (unchanged from FP8)
        const int smem_expert_count_size = ceil_div(
            num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment) * kSmemAlignment;
        const int smem_send_buffers_size = ceil_div(
            static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
            kSmemAlignment) * kSmemAlignment;
        const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

        // C/D output region: L1 FP4 (half bytes) and L2 BF16 (unchanged)
        const int smem_cd_l1 = num_epilogue_warpgroups * config.store_block_m * (config.block_n / 2) / 2 * kNumTMAStoreStages;
        const int smem_cd_l2 = num_epilogue_warpgroups * config.store_block_m * config.block_n * static_cast<int>(sizeof(nv_bfloat16));
        const int smem_cd = std::max(smem_cd_l1, smem_cd_l2);

        // Barriers, amax, tmem pointer (unchanged)
        const int smem_barriers = (num_dispatch_warps + kNumEpilogueStages * 2 + num_epilogue_warps * 2) * 8;
        const int smem_amax_reduction = config.store_block_m * num_epilogue_warps * static_cast<int>(sizeof(float));
        const int smem_tmem_ptr = 4;

        // Per-stage: packed FP4 A + packed FP4 B + SFA + SFB + full/empty barriers.
        // One SF uint32 column covers 128 K-elements, so BLOCK_K=256 needs 2 columns.
        const int smem_sfa_per_stage = config.sf_block_m * 4 * (config.block_k / 128);
        const int smem_sfb_per_stage = config.sf_block_n * 4 * (config.block_k / 128);
        const int fp4_smem_per_stage = load_block_m * config.block_k / 2
                                     + config.block_n * config.block_k / 2
                                     + smem_sfa_per_stage + smem_sfb_per_stage + 2 * 8;

        const int smem_fixed = smem_dispatch_size + smem_cd + smem_amax_reduction + smem_barriers + smem_tmem_ptr;
        int fp4_num_stages = (SM100ArchSpec::smem_capacity - smem_fixed) / fp4_smem_per_stage;
        DG_HOST_ASSERT(fp4_num_stages >= 2);

        // R2: optional DG_NUM_STAGES clamp (stage sweep at BLOCK_K=256). Clamps
        // the auto-computed stage count down (never below 2, never above auto);
        // off/0 → byte-identical. Applied before both config fields so num_stages
        // and smem_size stay consistent (single source of truth).
        if (const int env_stages = get_env<int>("DG_NUM_STAGES"); env_stages > 0)
            fp4_num_stages = std::max(2, std::min(env_stages, fp4_num_stages));

        config.num_stages = fp4_num_stages;
        config.smem_size = smem_fixed + fp4_num_stages * fp4_smem_per_stage;
    }

    constexpr int kGranK = 32;
    // SF TMA box outer dim: one uint32 SF column per 128 K-elements. At
    // BLOCK_K=256 each TMA delivers 2 columns. box-outer=2 must not cross a
    // per-expert SF boundary, so each expert's SF-K column count (shape_k/128)
    // must be a multiple of sf_box_outer_dim.
    const int sf_box_outer_dim = config.block_k / 128;
    DG_HOST_ASSERT((hidden / (kGranK * 4)) % sf_box_outer_dim == 0 and
                   (intermediate_hidden / (kGranK * 4)) % sf_box_outer_dim == 0);
    // Activation TMA descs: packed FP4, fp4_unpacked_smem=false. l1/l2 acts
    // tensors are expected to be `torch::kFloat8_e4m3fn` reinterpret-views
    // (one FP4 byte holds 2 elements). We pass the *element* count as the
    // inner dim; make_tma_2d_desc handles the packed swizzle math.
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode, 0, false, /*fp4_unpacked_smem=*/false);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_padded_sf_pool_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false, sf_box_outer_dim);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode, 0, false, /*fp4_unpacked_smem=*/false);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false, sf_box_outer_dim);
    // L1 output -> L2 acts: packed FP4 with N halved post-SwiGLU. The output's
    // swizzle atom width is a fixed 32B (= 64 packed FP4 elems per atom): the
    // epilogue's FP4 store layout (32B rows, kFP4BankGroupBytes XOR pattern) is
    // BLOCK_K-independent. Decoupled from swizzle_acts_mode so the BLOCK_K=256
    // 128B acts swizzle does not corrupt the L2 activation layout.
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_max_pool_tokens,
                                                       config.block_n / 2, config.store_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       32, 0, false, /*fp4_unpacked_smem=*/false);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode, 0, false, /*fp4_unpacked_smem=*/false);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_padded_sf_pool_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false, sf_box_outer_dim);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode, 0, false, /*fp4_unpacked_smem=*/false);
    const auto tensor_map_l2_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                           hidden, intermediate_hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false, sf_box_outer_dim);

    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    const auto num_sms = device_runtime->get_num_sms();
    const SM100FP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_l1_weights_sf = tensor_map_l1_weights_sf,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .tensor_map_l2_weights_sf = tensor_map_l2_weights_sf,
        .launch_args = LaunchArgs(num_sms,
                                  config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, 2)
    };

    const auto code = SM100FP4MegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp4_mega_moe", code);
    SM100FP4MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
