#pragma once

#include "mega_moe.hpp"

namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) MegaMoE configuration
// ----------------------------------------------------------------------------
// SM90 differs from SM100 in:
//   - No tensor memory (TMEM): WGMMA accumulators live in registers.
//   - No FP4: weights are FP8 e4m3 with per-128 channel float scales.
//   - No 2-CTA cluster MMA: TMA multicast cluster=2 may still be used.
//   - Activation SF is float, not UE8M0 int: L1 input uses per-128 K and the
//     fused L1 epilogue writes L2 activation SF at per-64 K granularity.
// The kernel implementation is in `deep_gemm/impls/sm90_fp8_mega_moe.cuh`.
// ============================================================================

struct MegaMoESM90Config {
    int block_m, block_n, block_k;
    int cluster_size;
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;
    int swizzle_acts_mode, swizzle_weights_mode;
    int num_stages, smem_size;
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    friend std::ostream& operator << (std::ostream& os, const MegaMoESM90Config& config) {
        os << "MegaMoESM90Config("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", cluster_size=" << config.cluster_size
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads << ")";
        return os;
    }
};

static std::tuple<int, int> get_block_config_for_mega_moe_sm90(
    const int& num_ranks, const int& num_experts,
    const int& num_topk, const int& num_tokens, const int &intermediate_hidden) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    // The relaxed 2-WG threshold enables the block_m=128 / 4-WG path only
    // above a higher tokens/expert bar (instead of the original >= 64),
    // trading two extra warpgroups for fewer register spills. On H20 the
    // smaller SM count (78 vs 132 on H100/H200) makes the extra warpgroups
    // costly, so the relaxation applies in two intermediate_hidden regimes:
    //   * pro   (>= 3072): 4-WG only when expected_tokens_per_expert > 512
    //   * flash (<= 2048): 4-WG only when expected_tokens_per_expert > 576,
    //     because 2-WG + BLOCK_N=256 outperforms 4-WG in part of the flash
    //     batch range -- 4-WG is reserved for the heaviest flash batches.
    // On H200/H100 the larger SM count makes the extra warpgroups always win,
    // so the original 4-WG-first (>= 64) threshold is kept for every shape,
    // as well as for the H20 mid-range (2048 < intermediate_hidden < 3072).
    const int num_sms = device_runtime->get_num_sms();
    const bool is_h20 = num_sms <= 84;
    const bool apply_h20_pro_relaxation   = is_h20 and intermediate_hidden >= 3072;
    const bool apply_h20_flash_relaxation = is_h20 and intermediate_hidden <= 2048;
    bool auto_split_mn;
    if (apply_h20_pro_relaxation)
        auto_split_mn = expected_tokens_per_expert > 512.0f;
    else if (apply_h20_flash_relaxation)
        auto_split_mn = expected_tokens_per_expert > 576.0f;
    else
        auto_split_mn = expected_tokens_per_expert >= 64.0f;
    if (auto_split_mn)
        return {128, 512};

    const int block_m = 64;
    const int num_epilogue_warpgroups = 2;

    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );
    return {block_m, num_epilogue_warpgroups * 128};
}

static bool should_use_swap_ab_for_mega_moe_sm90(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& block_m, const int& num_epilogue_threads) {
    // swapAB is ENABLED by default (the L1 SF-pool stride bug that corrupted
    // pool blocks >= 1 was fixed: BLOCK_M -> SF_BLOCK_M in the swapAB L1 epilogue).
    // Kill-switch retained: set DG_SM90_FP8_SWAP_AB=0 to force the non-swap path.
    // swapAB composes with the fused shared expert (shared's token-axis output
    // matches swapAB's reduced M-axis), so no special-casing is needed here.
    if (get_env<int>("DG_SM90_FP8_SWAP_AB", 1) == 0)
        return false;
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    const bool decode_split_n_path =
        block_m == 64 and num_epilogue_threads == 256;
    return decode_split_n_path and num_tokens <= 128 and expected_tokens_per_expert > 0.0f;
}

static std::pair<int, int> get_pipeline_config_for_mega_moe_sm90(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const bool& use_swap_ab = false) {
    constexpr int kSmemAlignment = 1024;

    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    const int smem_cd_l1 = block_m * (block_n / 2);
    const int smem_cd_l2 = block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd_swap_l1 = use_swap_ab
        ? block_m * (block_n / 2) *
              (static_cast<int>(sizeof(float)) + static_cast<int>(sizeof(uint8_t)))
        : 0;
    const int smem_cd = align(
        std::max(std::max(smem_cd_l1, smem_cd_l2), smem_cd_swap_l1),
        kSmemAlignment);

    const int smem_sfa_per_stage = align(2 * block_m * static_cast<int>(sizeof(float)), 128);
    const int smem_sfb_per_stage = 0;
    const int smem_per_stage = block_m * block_k + block_n * block_k +
                               smem_sfa_per_stage + smem_sfb_per_stage;

    // The scheduler adds 2 task-info full/empty barrier pairs and two 32-byte
    // task-info slots (see `sm90_fp8_mega_moe.cuh`). `SM90MegaMoETaskInfo` is
    // alignas(16)/32 B while a barrier slot is only 8 B, so the kernel pads one
    // extra barrier when the preceding barrier count is odd
    // (`kTaskInfoBarrierPad = kTaskInfoBaseBarriers & 1u`). Only
    // `num_dispatch_warps` affects that parity (2*num_stages, 2*num_epilogue_warps
    // and the 4 task-info barriers are all even), so mirror the same pad here.
    const int smem_task_info_barriers = 4;  // 2 full + 2 empty
    const int smem_task_info_pad = (num_dispatch_warps & 1) * 8;
    const int smem_barriers_fixed =
        (num_dispatch_warps + 2 * num_epilogue_warps + smem_task_info_barriers) * 8 +
        smem_task_info_pad;
    const int smem_task_infos = 2 * 32;
    const int smem_barriers_per_stage = 2 * 8;
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_barriers_fixed + smem_task_infos;

    const int num_stages = (smem_capacity - smem_fixed) /
                           (smem_per_stage + smem_barriers_per_stage);
    DG_HOST_ASSERT(num_stages >= 2);
    const int smem_size = smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage);
    DG_HOST_ASSERT(smem_size <= smem_capacity);

    // Cross-check against the kernel's exact barrier/task-info layout: the
    // task-info ring (including the alignment pad) must end inside the
    // allocated dynamic shared memory.
    const int smem_task_info_end =
        smem_dispatch_size + smem_cd + num_stages * smem_per_stage +
        (num_dispatch_warps + 2 * num_stages + 2 * num_epilogue_warps +
         smem_task_info_barriers + smem_task_info_pad / 8) * 8 +
        smem_task_infos;
    DG_HOST_ASSERT(smem_task_info_end <= smem_size);
    return {num_stages, smem_size};
}

static MegaMoESM90Config get_mega_moe_config_sm90(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens) {
    const auto [block_m, num_epilogue_threads] = get_block_config_for_mega_moe_sm90(
        num_ranks, num_experts, num_topk, num_tokens, intermediate_hidden);
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    const bool auto_split_mn =
        block_m == 128 and num_epilogue_threads == 512;
    const bool decode_split_n_path =
        block_m == 64 and num_epilogue_threads == 256;
    const bool decode_use_block_n_256 =
        decode_split_n_path and
        expected_tokens_per_expert >= 0.25f and
        (2 * intermediate_hidden) % 256 == 0 and hidden % 256 == 0;
    const bool use_swap_ab = should_use_swap_ab_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        block_m, num_epilogue_threads);
    int block_n = use_swap_ab ? 256
                              : (auto_split_mn ? 256 :
                                 (decode_use_block_n_256 ? 256 : 128));
    const int block_k = 128;
    const int cluster_size = 1;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;

    // The scheduler needs a dedicated producer warp, so the non-epilogue section
    // is exactly 3 warps (TMA-A, TMA-B, producer) and dispatch is a single warp:
    // dispatch + non-epilogue = 32 + 96 = 128, a whole warpgroup that keeps the
    // math warpgroups 128-thread aligned. This is the minimal aligned topology for
    // every epilogue width:
    //   * 2-WG (epilogue=256): 32 + 96 + 256 = 384 threads, ceiling
    //     65536/384 = 170 >= 168, so the epilogue accumulators do not spill.
    //   * 4-WG (epilogue=512): 32 + 96 + 512 = 640 threads. Halving dispatch to one
    //     warp is the cost of fitting the producer warp under 128-thread alignment;
    //     a 2-dispatch-warp variant would pad to 768 threads and spill worse.
    const int num_dispatch_threads = 32;
    const int num_non_epilogue_threads = 96;
    DG_HOST_ASSERT((num_dispatch_threads + num_non_epilogue_threads) % 128 == 0);

    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
        SM90ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        use_swap_ab);

    const auto config = MegaMoESM90Config {
        block_m, block_n, block_k,
        cluster_size,
        num_max_pool_tokens, num_padded_sf_pool_tokens,
        swizzle_acts_mode, swizzle_weights_mode,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads
    };

    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "MegaMoESM90Config(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={}, swap_ab={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk,
            use_swap_ab);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
