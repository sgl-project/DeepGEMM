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
    int num_experts_per_wave;
    int num_stages, smem_size;
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    friend std::ostream& operator << (std::ostream& os, const MegaMoESM90Config& config) {
        os << "MegaMoESM90Config("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", cluster_size=" << config.cluster_size
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads << ")";
        return os;
    }
};

static std::tuple<int, int> get_block_config_for_mega_moe_sm90(
    const int& num_ranks, const int& num_experts,
    const int& num_topk, const int& num_tokens) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    const bool auto_split_mn = expected_tokens_per_expert > 64.0f;
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

// SM90 retains the original wave scheduler and its ring-capacity heuristic.
// Keep these helpers local to the Hopper path: upstream's SM100 scheduler now
// sizes live task pools directly and no longer exposes the legacy helpers.
static int get_num_wave_pool_tokens_for_mega_moe_sm90(
    const int& num_ranks, const int& num_topk, const int& num_max_tokens_per_rank,
    const int& num_experts_per_wave, const int& block_m) {
    DG_HOST_ASSERT(num_max_tokens_per_rank % block_m == 0);
    const auto num_tokens_from_all_ranks = num_max_tokens_per_rank * num_ranks;
    if (num_experts_per_wave == 1)
        return num_tokens_from_all_ranks;

    return std::min(
        num_tokens_from_all_ranks * num_experts_per_wave,
        math::align(
            num_tokens_from_all_ranks * num_topk + num_experts_per_wave * (block_m - 1),
            block_m));
}

static int get_num_experts_per_wave_for_mega_moe_sm90_legacy(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms,
    const int& num_ring_tokens, const int& num_max_tokens_per_rank, const int& num_ranks) {
    int num_max_experts_per_wave = num_experts_per_rank;
    while (num_max_experts_per_wave > 0 and
           get_num_wave_pool_tokens_for_mega_moe_sm90(
               num_ranks, num_topk, num_max_tokens_per_rank,
               num_max_experts_per_wave, block_m) > num_ring_tokens)
        --num_max_experts_per_wave;
    DG_HOST_ASSERT(num_max_experts_per_wave > 0 and "Buffer size is too small");

    constexpr int kImbalanceFactor = 2;
    const float num_expected_tokens_per_expert =
        static_cast<float>(num_tokens * num_topk) / num_experts_per_rank;
    const int num_expected_m_blocks = std::max(
        ceil_div(static_cast<int>(std::ceil(num_expected_tokens_per_expert)), block_m), 1);
    const int num_l1_n_blocks = (2 * intermediate_hidden) / block_n;
    const int num_expected_l1_blocks_per_expert = num_expected_m_blocks * num_l1_n_blocks;
    int num_min_expected_experts_to_fill_sms =
        ceil_div(kImbalanceFactor * num_sms, num_expected_l1_blocks_per_expert);

    if (num_expected_tokens_per_expert < 1)
        num_min_expected_experts_to_fill_sms = num_experts_per_rank;
    if (num_min_expected_experts_to_fill_sms >= num_max_experts_per_wave)
        return num_max_experts_per_wave;
    if (num_expected_l1_blocks_per_expert >= num_sms)
        return num_min_expected_experts_to_fill_sms;

    const int num_sweep_max_experts_per_wave = std::min(
        num_max_experts_per_wave, num_min_expected_experts_to_fill_sms * 2);
    int best_num_experts_per_wave = num_min_expected_experts_to_fill_sms;
    float best_tail_ratio = -1.0f;
    for (int num_experts_per_wave = num_min_expected_experts_to_fill_sms;
         num_experts_per_wave <= num_sweep_max_experts_per_wave;
         ++num_experts_per_wave) {
        const int remainder = num_experts_per_rank % num_experts_per_wave;
        const float tail_ratio = remainder == 0 ?
            1.0f : static_cast<float>(remainder) / num_experts_per_wave;
        if (tail_ratio > best_tail_ratio) {
            best_tail_ratio = tail_ratio;
            best_num_experts_per_wave = num_experts_per_wave;
        }
    }
    return best_num_experts_per_wave;
}

static int get_num_experts_per_wave_for_mega_moe_sm90(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms,
    const int& num_ring_tokens, const int& num_max_tokens_per_rank, const int& num_ranks) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    if (expected_tokens_per_expert < 1.0f or expected_tokens_per_expert > 4.0f)
        return num_experts_per_rank;

    if (block_m == 64 and intermediate_hidden >= 3072) {
        const int num_n_blocks_per_expert = (2 * intermediate_hidden) / block_n;
        const int single_wave_blocks =
            num_experts_per_rank * num_n_blocks_per_expert;
        if (single_wave_blocks >= 4 * num_sms)
            return num_experts_per_rank;
    }
    return get_num_experts_per_wave_for_mega_moe_sm90_legacy(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms,
        num_ring_tokens, num_max_tokens_per_rank, num_ranks);
}

static bool should_use_swap_ab_for_mega_moe_sm90(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& block_m, const int& num_epilogue_threads) {
    // swapAB is ENABLED by default (the L1 SF-pool stride bug that corrupted
    // pool blocks >= 1 was fixed: BLOCK_M -> SF_BLOCK_M in the swapAB L1 epilogue).
    // Kill-switch retained: set DG_SM90_FP8_SWAP_AB=0 to force the non-swap path.
    if (get_env<int>("DG_SM90_FP8_SWAP_AB", 1) == 0)
        return false;
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    const bool decode_split_n_path =
        block_m == 64 and num_epilogue_threads == 256;
    return decode_split_n_path and expected_tokens_per_expert < 30.0f
           and expected_tokens_per_expert > 0.0f;
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

    const int smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps) * 8;
    const int smem_barriers_per_stage = 2 * 8;
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_barriers_fixed;

    const int num_stages = (smem_capacity - smem_fixed) /
                           (smem_per_stage + smem_barriers_per_stage);
    DG_HOST_ASSERT(num_stages >= 2);
    const int smem_size = smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage);
    DG_HOST_ASSERT(smem_size <= smem_capacity);
    return {num_stages, smem_size};
}

// Activation-scale granularities (L1 input, L2 intermediate). Both SwiGLU and
// SiTU use the same fixed recipe on SM90.
static constexpr int kSM90FP4L1ActSFGranK = 128;
static constexpr int kSM90FP4L2ActSFGranK = 64;

static float get_fp4_sm90_swiglu_prefill_threshold() {
    return static_cast<float>(get_env<int>("DG_SM90_FP4_PREFILL_E", 80));
}

// ---- Analytic config cost model (SiTU) ----
// Replaces the hand-tuned swapAB/decode/prefill boundaries with a per-expert
// expected-cost comparison. Fitted 2026-07-30 on 21 same-shape A/B
// measurements (docs 15.12-15.15); all 21 classify correctly, and the
// model's two out-of-sample predictions (prefill wins inside the former
// decode band at e=220/350) were confirmed at +3.8%/+2.8% (doc 15.16).
// With X ~ Poisson(e) tokens per expert (uniform-routing approximation),
// in units of one regular-mainloop BLOCK_M row:
//   R(M) = E[ceil(X/M)]*M                    expected processed rows
//   T(M) = E[ceil(X/M)]                      per-expert B-decode passes
//   cost_decode  = R(64) + kBDecodeRows*T(64)
//   cost_prefill = (1-kPrefillRowGain)*R(128) + kBDecodeRows*T(128)
//   cost_swap    = kSwapRowCost*R(8) + kSwapExpertRows*P(X>0)
// The row/decode ratio is shape-independent to first order (both scale with
// N*K), so the constants transfer across expert shapes; refit via
// scratchpad fit_cost_model.py if the architecture changes.
static float fp4_expected_num_tiles(const float& e, const int& m) {
    // E[ceil(X/m)] via a normal approximation with continuity correction;
    // matches the offline fitting script exactly.
    const float sigma = std::sqrt(e);
    float total = 0.0f;
    for (int k = 0; ; ++ k) {
        const float z = (static_cast<float>(k * m) + 0.5f - e) / sigma;
        const float p = 0.5f * std::erfc(z * 0.70710678f);
        total += p;
        if (p < 1e-5f and static_cast<float>(k * m) > e)
            break;
    }
    return total;
}

enum class FP4SM90ConfigKind { kSwapAB, kDecode, kPrefill };

static FP4SM90ConfigKind get_fp4_sm90_situ_config_kind(
    const float& expected_tokens_per_expert) {
    constexpr float kBDecodeRows    = 6.29f;
    constexpr float kPrefillRowGain = 0.0030f;
    constexpr float kSwapRowCost    = 1.335f;
    constexpr float kSwapExpertRows = 40.6f;
    const float e = expected_tokens_per_expert;
    if (e <= 0.0f)
        return FP4SM90ConfigKind::kDecode;
    const float t64  = fp4_expected_num_tiles(e, 64);
    const float t128 = fp4_expected_num_tiles(e, 128);
    const float t8   = fp4_expected_num_tiles(e, 8);
    const float p_active = 1.0f - std::exp(-e);
    const float cost_decode  = t64 * 64.0f + kBDecodeRows * t64;
    const float cost_prefill = (1.0f - kPrefillRowGain) * t128 * 128.0f +
                               kBDecodeRows * t128;
    const float cost_swap = kSwapRowCost * t8 * 8.0f +
                            kSwapExpertRows * p_active;
    const bool swap_enabled = get_env<int>("DG_SM90_FP4_SWAP_AB", 1) != 0;
    if (swap_enabled and cost_swap <= cost_decode and cost_swap <= cost_prefill)
        return FP4SM90ConfigKind::kSwapAB;
    return cost_prefill < cost_decode ? FP4SM90ConfigKind::kPrefill
                                      : FP4SM90ConfigKind::kDecode;
}

// Whether `e` should run the prefill bundle (BLOCK_M=128 + early_b_decode +
// ss_nsplit). SiTU always uses the analytic cost model above; SwiGLU keeps its
// measured scalar boundary.
static bool is_fp4_sm90_prefill_band(const float& expected_tokens_per_expert,
                                     const bool& use_situ) {
    if (use_situ)
        return get_fp4_sm90_situ_config_kind(expected_tokens_per_expert) ==
               FP4SM90ConfigKind::kPrefill;
    return expected_tokens_per_expert >=
           get_fp4_sm90_swiglu_prefill_threshold();
}

static std::tuple<int, int> get_block_config_for_mega_moe_sm90_fp4(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens, const bool& use_situ) {
    (void)num_max_tokens_per_rank;

    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    const bool auto_split_mn =
        is_fp4_sm90_prefill_band(expected_tokens_per_expert, use_situ);
    const bool ultra_small_split_n =
        expected_tokens_per_expert > 0.0f and expected_tokens_per_expert < 0.375f;
    int block_m = auto_split_mn ? 128 : 64;
    int num_epilogue_warpgroups = (auto_split_mn or ultra_small_split_n) ? 2 : block_m / 64;
    DG_HOST_ASSERT(block_m >= 64 and block_m % 64 == 0);
    DG_HOST_ASSERT(num_epilogue_warpgroups >= 1 and
                   ((block_m / num_epilogue_warpgroups == 64) or
                    (block_m == 64 and num_epilogue_warpgroups > 1)));

    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );
    return {block_m, num_epilogue_warpgroups * 128};
}

static int get_num_experts_per_wave_for_mega_moe_sm90_fp4(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms,
    const int& num_ring_tokens, const int& num_max_tokens_per_rank, const int& num_ranks) {
    // Simplified: schedule FP4 expert waves exactly like the FP8 path. The
    // historical flash/pro wave tables (9 + 12 first-match rules) were tuned
    // point-by-point on benchmark batches and are retired.
    return get_num_experts_per_wave_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms,
        num_ring_tokens, num_max_tokens_per_rank, num_ranks);
}

static std::pair<int, int> get_pipeline_config_for_mega_moe_sm90_fp4(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const bool& use_early_b_decode = false,
    const bool& use_decode_done_mbarrier = false,
    const bool& use_swap_ab = false) {
    constexpr int kSmemAlignment = 1024;

    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int smem_cd_l1 = block_m * (block_n / 2);
    const int smem_cd_l2 = block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd_swap_l1 = use_swap_ab
        ? block_m * (block_n / 2) *
              (static_cast<int>(sizeof(float)) + static_cast<int>(sizeof(uint8_t)))
        : 0;
    const int smem_cd_base = std::max(smem_cd_l1, smem_cd_l2);
    const int smem_cd = align(std::max(smem_cd_base, smem_cd_swap_l1), kSmemAlignment);

    const bool fp4_split_n_eligible =
        block_m == 64 and num_epilogue_warpgroups > 1 and
        block_n % num_epilogue_warpgroups == 0 and
        (block_n / num_epilogue_warpgroups) >= 64;
    const int wg_l1_out_block_n = fp4_split_n_eligible
        ? (block_n / num_epilogue_warpgroups) / 2
        : 0;
    const bool split_n_shares_sf =
        fp4_split_n_eligible and wg_l1_out_block_n < kSM90FP4L2ActSFGranK;
    const int fp4_split_n_amax_scratch_slots = 32 * 2 * 2;
    const int smem_amax_scratch = split_n_shares_sf
        ? align(fp4_split_n_amax_scratch_slots * static_cast<int>(sizeof(uint32_t)),
                kSmemAlignment)
        : 0;
    const int l1_sfa_groups_per_block_k = block_k / kSM90FP4L1ActSFGranK;
    const int l2_sfa_groups_per_block_k = block_k / kSM90FP4L2ActSFGranK;
    const int smem_sfa_per_stage =
        align(std::max(l1_sfa_groups_per_block_k, l2_sfa_groups_per_block_k) *
                  block_m * static_cast<int>(sizeof(float)),
              128);
    const int smem_sfb_per_stage =
        align(block_n * static_cast<int>(sizeof(uint32_t)), 128);

    const int smem_b_decoded_per_stage = block_n * block_k;
    const int smem_b_packed_per_stage = block_n * (block_k / 2);
    const int smem_per_stage = block_m * block_k +
                               smem_b_decoded_per_stage +
                               smem_b_packed_per_stage +
                               smem_sfa_per_stage + smem_sfb_per_stage;

    const int smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps) * 8;
    const int smem_decode_full_per_stage = use_early_b_decode ? 8 : 0;
    const int smem_decode_done_per_stage =
        use_decode_done_mbarrier ? 8 : 0;
    const int smem_barriers_per_stage =
        2 * 8 + smem_decode_full_per_stage + smem_decode_done_per_stage;
    const int smem_fixed =
        smem_dispatch_size + smem_cd + smem_amax_scratch + smem_barriers_fixed;

    // No FP4 stage cap (FP8 parity): always use as many pipeline stages as
    // SMEM allows. The historical 11-rule cap table is retired.
    const int num_stages = (smem_capacity - smem_fixed) /
                           (smem_per_stage + smem_barriers_per_stage);
    DG_HOST_ASSERT(num_stages >= 2);
    return {num_stages,
            smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage)};
}

static MegaMoESM90Config get_mega_moe_config_sm90_fp4(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens,
    const bool& use_situ,
    const bool& use_early_b_decode = false,
    const bool& use_decode_done_mbarrier = false,
    const bool& use_swap_ab = false) {
    const auto [block_m, num_epilogue_threads] = get_block_config_for_mega_moe_sm90_fp4(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens,
        use_situ);
    const int block_k = 128;
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    const int block_n = 128;
    int fp4_num_epilogue_warpgroups = num_epilogue_threads / 128;
    const bool fp4_flash_shape = intermediate_hidden <= 2048;
    const bool fp4_pro_shape = intermediate_hidden >= 3072;
    const bool fp4_flash_or_pro_shape = fp4_flash_shape or fp4_pro_shape;
    // Shape bands depend only on model shape and routing density; kernel bands add tile/thread constraints.
    const bool fp4_split_n_eligible =
        block_m == 64 and block_n % 128 == 0;
    const bool fp4_split_n_shape_band =
        fp4_flash_or_pro_shape and
        expected_tokens_per_expert > 0.0f and
        not is_fp4_sm90_prefill_band(expected_tokens_per_expert, use_situ);
    if (fp4_split_n_eligible and fp4_split_n_shape_band) {
        fp4_num_epilogue_warpgroups = 2;
    }
    DG_HOST_ASSERT(fp4_num_epilogue_warpgroups >= 1);
    DG_HOST_ASSERT((block_m / fp4_num_epilogue_warpgroups == 64) or
                   (block_m == 64 and fp4_num_epilogue_warpgroups > 1 and
                    block_n % fp4_num_epilogue_warpgroups == 0 and
                    (block_n / fp4_num_epilogue_warpgroups) >= 64));
    const int fp4_num_epilogue_threads = fp4_num_epilogue_warpgroups * 128;
    const int cluster_size = 1;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 0;

    const int num_sms = device_runtime->get_num_sms();
    int num_experts_per_wave = get_num_experts_per_wave_for_mega_moe_sm90_fp4(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms,
        num_max_pool_tokens, num_max_tokens_per_rank, num_ranks);

    const bool fp4_small_block_n_kernel =
        block_m == 64 and block_n == 128;
    const bool fp4_split_n_decode_thread_kernel_band =
        fp4_small_block_n_kernel and fp4_split_n_shape_band;
    const bool fp4_2wg_decode_offload_kernel_band =
        block_m == 128 and block_n == 128 and
        fp4_num_epilogue_threads == 256 and
        is_fp4_sm90_prefill_band(expected_tokens_per_expert, use_situ);
    const bool fp4_decode_assist_thread_kernel_band =
        fp4_2wg_decode_offload_kernel_band or
        (fp4_small_block_n_kernel and
         expected_tokens_per_expert > 0.0f and expected_tokens_per_expert <= 24.0f);
    const int default_num_dispatch_threads =
        (fp4_split_n_decode_thread_kernel_band or
         fp4_decode_assist_thread_kernel_band) ? 64 : 128;
    const int num_dispatch_threads = default_num_dispatch_threads;
    DG_HOST_ASSERT(num_dispatch_threads == 64 or num_dispatch_threads == 128);
    const int default_num_non_epilogue_threads =
        fp4_split_n_decode_thread_kernel_band ? 320 :
        (fp4_decode_assist_thread_kernel_band ? 192 : 128);
    const int num_non_epilogue_threads = default_num_non_epilogue_threads;
    DG_HOST_ASSERT(num_non_epilogue_threads >= 128 and
                   num_non_epilogue_threads % 64 == 0);
    DG_HOST_ASSERT((num_dispatch_threads + num_non_epilogue_threads) % 128 == 0);

    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90_fp4(
        SM90ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k,
        num_dispatch_threads / 32, fp4_num_epilogue_threads / 32,
        use_early_b_decode, use_decode_done_mbarrier,
        use_swap_ab);

    const auto config = MegaMoESM90Config {
        block_m, block_n, block_k,
        cluster_size,
        num_max_pool_tokens, num_padded_sf_pool_tokens,
        swizzle_acts_mode, swizzle_weights_mode,
        num_experts_per_wave,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, fp4_num_epilogue_threads
    };

    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "MegaMoESM90FP4Config(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={}, use_situ={}, early_b_decode={}, decode_done_mbarrier={}, swap_ab={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk,
            use_situ, use_early_b_decode, use_decode_done_mbarrier, use_swap_ab);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

static MegaMoESM90Config get_mega_moe_config_sm90(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens) {
    const auto [block_m, num_epilogue_threads] = get_block_config_for_mega_moe_sm90(
        num_ranks, num_experts, num_topk, num_tokens);
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    const bool auto_split_mn =
        block_m == 128 and num_epilogue_threads == 512;
    const bool decode_split_n_path =
        block_m == 64 and num_epilogue_threads == 256;
    const bool decode_use_block_n_256 =
        decode_split_n_path and intermediate_hidden >= 2048 and
        expected_tokens_per_expert >= 0.25f and
        (2 * intermediate_hidden) % 256 == 0 and hidden % 256 == 0;
    const bool use_swap_ab = should_use_swap_ab_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        block_m, num_epilogue_threads);
    int block_n = use_swap_ab ? 128
                              : (auto_split_mn ? 256 :
                                 (decode_use_block_n_256 ? 256 : 128));
    const int block_k = 128;
    const int cluster_size = 1;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;

    const int num_sms = device_runtime->get_num_sms();
    const int num_experts_per_wave = get_num_experts_per_wave_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms,
        num_max_pool_tokens, num_max_tokens_per_rank, num_ranks);

    const bool reduce_decode_threads = num_epilogue_threads == 128;
    const bool decode_split_n =
        block_m == 64 and num_epilogue_threads == 256;
    const bool shrink_non_epilogue = reduce_decode_threads or decode_split_n;
    const int num_dispatch_threads =
        (num_epilogue_threads == 512 or shrink_non_epilogue) ? 64 : 128;
    const bool split_sfa_loader_warp = false;
    const int num_non_epilogue_threads =
        split_sfa_loader_warp ? 128 :
            ((num_epilogue_threads == 512 or shrink_non_epilogue) ? 64 : 128);
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
        num_experts_per_wave,
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
