#pragma once

#include "../../runtime/runtime.hpp"

#include "mega_moe.hpp"

#include <deep_gemm/layout/sm90_mega_moe.cuh>

namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) MegaMoE configuration
// ----------------------------------------------------------------------------
// SM90 differs from SM100 in:
//   - No tensor memory (TMEM): WGMMA accumulators live in registers.
//   - No FP4: weights are FP8 e4m3 with per-128 channel float scales.
//   - No 2-CTA cluster MMA: TMA multicast cluster=2 may still be used.
//   - Activation SF is float, not UE8M0 int: L1 input uses per-128 K and the
//     fused L1 epilogue writes L2 activation SF at per-128 or per-64 K granularity (the buffer's l2_act_sf_gran_k).
// The kernel implementation is in `deep_gemm/impls/sm90_fp8_mega_moe.cuh`.
// ============================================================================

struct MegaMoESM90Config {
    int block_m, block_n, block_k;
    int cluster_size;
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;
    int num_ring_tokens;
    int swizzle_acts_mode, swizzle_weights_mode;
    int num_experts_per_wave;
    int num_stages, smem_size;
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;
    bool half_l2_cd;
    bool multicast_on_b;
    // > 0: L2 tiles interleaved into the L1 phase at this lag (units of kSM90LagUnitM m-blocks, encoded lag + 1000 x group); 0: wave schedule
    int l2_lag_units;
    // L2 BF16 staging passes (0 = derived from half_l2_cd; 4 = quarter-width buffer)
    int l2_cd_passes;
    // L2 BF16 staging on the quarter-width buffer: 0 = column passes, 2 = row passes, 4 = row passes through stmatrix
    int l2_stage_mode;
    // 1: combine the tokens whose experts have all published their done flag while waiting in the pre-combine barrier (needs hidden >= 512 x topk); 0: barrier, then combine
    int early_combine;
    // pull arrivals published per gpu-scope release fence (1 = one red.release per row; N > 1 ramps 1, 2, 4, ... every 8 rows of a warp)
    int pull_publish_batch;

    friend std::ostream& operator << (std::ostream& os, const MegaMoESM90Config& config) {
        os << "MegaMoESM90Config("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", cluster_size=" << config.cluster_size
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", num_ring_tokens=" << config.num_ring_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", half_l2_cd=" << config.half_l2_cd << ", l2_cd_passes=" << config.l2_cd_passes
           << ", multicast_on_b=" << config.multicast_on_b
           << ", l2_lag_units=" << config.l2_lag_units
           << ", l2_stage_mode=" << config.l2_stage_mode
           << ", early_combine=" << config.early_combine
           << ", pull_publish_batch=" << config.pull_publish_batch << ")";
        return os;
    }
};

// L2-lag schedule selection. Decided when the symm buffer is sized, returned to Python as the encoded template value and passed
// back at every launch, so the ring capacity and the kernel schedule can never disagree: a full-pool request of at least
// kSm90AutoLagMinTokens tokens per rank gets the lag schedule on a lag-sized ring, everything else the wave schedule.
static constexpr int kSm90AutoLagMinTokens = 1024;
static constexpr double kSm90AutoLagFraction = 0.45;
static constexpr int kSm90AutoLagGroup = 4;
static constexpr int kSm90AutoLagMinUnits = 8;
static constexpr int kSm90AutoLagMaxUnits = 64;
static constexpr int kSm90LagRingMargin = 4;

static int get_sm90_auto_late_lag(const int& num_max_tokens_per_rank, const int& num_topk, const int& num_experts_per_rank) {
    const int units_per_expert = std::max(1, (num_max_tokens_per_rank * num_topk + num_experts_per_rank * 1024 - 1) /
                                             (num_experts_per_rank * 1024));
    const int units = num_experts_per_rank * units_per_expert;
    return std::clamp(static_cast<int>(kSm90AutoLagFraction * units + 0.5), kSm90AutoLagMinUnits, kSm90AutoLagMaxUnits);
}

static int get_sm90_l2_lag_units_for_buffer(const int& num_max_tokens_per_rank, const int& num_experts_per_wave_knob,
                                            const int& num_topk, const int& num_experts_per_rank) {
    if (num_experts_per_wave_knob == 0 and num_max_tokens_per_rank >= kSm90AutoLagMinTokens)
        return get_sm90_auto_late_lag(num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    return 0;
}

// Template encoding consumed by the scheduler: lag + 1000 * group
static int get_sm90_l2_lag_encoded(const int& lag) {
    return lag > 0 ? lag + 1000 * kSm90AutoLagGroup : 0;
}

static int sm90_lag_units_of(const int& encoded) { return encoded % 1000; }
static int sm90_lag_group_of(const int& encoded) { return std::max(encoded / 1000, 1); }

// Ring capacity (in pool blocks) that the lag schedule needs: L2(p) is issued at most kSM90LagUnitM * (lag + group) pool
// blocks after L1(p), plus the blocks that may be in flight on top.
static int get_sm90_lag_ring_blocks(const int& lag, const int& group) {
    return static_cast<int>(layout::kSM90LagUnitM) * (lag + std::max(group, 1)) + kSm90LagRingMargin;
}

static int get_token_alignment_sm90() {
    return layout::kSM90LCMBlockM;
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

// Decode single wave: the BLOCK_M-64 split-N topology under this many tokens per rank runs one wave of all local experts when its worst-case pool fits the ring.
static constexpr int kSm90SingleWaveMaxTokens = 256;

// Per-call wave schedule on lag-ring buffers: when the call's worst-case pool (num_ranks x per-rank token bound x topk rows plus one
// padding block per local expert) fits the ring, every pool block of the call maps to ring generation 0, so no slot is reused inside
// the call and the wave schedule cannot wait on a slot release. Without a caller bound the capacity is tested, which never fits.
static constexpr int kSm90CallWaveMaxTokens = 256;

static bool sm90_single_wave_pool_fits(const int& num_ranks, const int& num_topk, const int& num_wave_bound_tokens_per_rank,
                                       const int& num_experts_per_rank, const int& block_m, const int& num_ring_tokens) {
    return get_num_wave_pool_tokens_for_mega_moe_sm90(num_ranks, num_topk, num_wave_bound_tokens_per_rank, num_experts_per_rank, block_m) <= num_ring_tokens;
}

static bool sm90_is_decode_single_wave_call(const int& block_m, const int& num_tokens) {
    return block_m == 64 and num_tokens <= kSm90SingleWaveMaxTokens;
}

// Per-rank token count the M-keyed host rules are evaluated at: the caller's global bound when given (the same on every rank), else this rank's own count.
static int sm90_rule_tokens_per_rank(const int& num_tokens, const int& num_tokens_bound) {
    return num_tokens_bound > 0 ? num_tokens_bound : num_tokens;
}

static bool sm90_call_wave_class(const int& block_m, const int& num_tokens, const int& num_tokens_bound) {
    if (block_m == 64)
        return num_tokens <= kSm90SingleWaveMaxTokens;
    if (block_m == 128)
        return sm90_rule_tokens_per_rank(num_tokens, num_tokens_bound) <= kSm90CallWaveMaxTokens;
    return false;
}

// Publish-batch rule: calls with at least this many tokens per rank publish kSm90PullPublishBatch pull rows per release fence.
static constexpr int kSm90PullPublishMinTokens = 8192;
static constexpr int kSm90PullPublishBatch = 16;

static constexpr int kSm90PullEagerMaxTokens = 256;
static constexpr int kSm90PdlMaxTokens = 256;

static int get_num_max_pool_tokens_sm90(
    const int& num_ranks, const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_experts_per_rank) {
    return layout::get_num_max_pool_tokens_sm90(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
}

static int get_num_padded_sf_pool_tokens_sm90(const int& num_data_pool_tokens) {
    int num_padded = 0;
    for (const int& block_m: layout::kSM90CandidateBlockM)
        num_padded = std::max(num_padded, layout::get_num_sf_ring_tokens(num_data_pool_tokens, block_m));
    return num_padded;
}

static bool is_sm90_candidate_block_m(const int& block_m) {
    return std::any_of(layout::kSM90CandidateBlockM, layout::kSM90CandidateBlockM + layout::kNumSM90CandidateBlockMs,
                       [=](const auto& candidate) { return candidate == block_m; });
}

static std::tuple<int, int> get_block_config_for_mega_moe_sm90(
    const int& num_ranks, const int& num_experts,
    const int& num_topk, const int& num_tokens) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    const bool auto_split_mn = expected_tokens_per_expert >= 64.0f;
    if (auto_split_mn)   // 2 math warpgroups on the 128x256 tile (64x256 each, spill-free)
        return {128, 256};

    const int block_m = 64;
    const int num_epilogue_warpgroups = 2;

    DG_HOST_ASSERT(is_sm90_candidate_block_m(block_m));
    return {block_m, num_epilogue_warpgroups * 128};
}

static int get_num_experts_per_wave_for_mega_moe_sm90(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms,
    const int& num_ring_tokens, const int& num_max_tokens_per_rank, const int& num_ranks,
    // Per-rank token count the ring-capacity clamp is evaluated at (<= num_max_tokens_per_rank)
    const int& num_wave_bound_tokens_per_rank,
    const int& num_tokens_bound = 0) {
    // Ring mode must derive the wave size from the ring capacity first and
    // foremost: a wave whose pool exceeds the ring would reuse slots inside a
    // single wave (L1 epilogue of generation g+1 waits on L2 consumption that
    // only happens after the wave's whole L1 phase) and deadlock. The
    // occupancy-driven early returns below are only safe when the ring covers
    // the full pool.
    const int num_max_pool_tokens = get_num_max_pool_tokens_sm90(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const bool call_pool_fits_ring = num_ring_tokens < num_max_pool_tokens and
        sm90_call_wave_class(block_m, num_tokens, num_tokens_bound) and
        sm90_single_wave_pool_fits(num_ranks, num_topk, num_wave_bound_tokens_per_rank, num_experts_per_rank, block_m, num_ring_tokens);
    if (num_ring_tokens < num_max_pool_tokens and not call_pool_fits_ring)
        return get_num_experts_per_wave_for_mega_moe_sm90_legacy(
            num_experts_per_rank, num_tokens, num_topk,
            intermediate_hidden, block_m, block_n, num_sms,
            num_ring_tokens, num_wave_bound_tokens_per_rank, num_ranks);

    if (sm90_is_decode_single_wave_call(block_m, num_tokens))
        return num_experts_per_rank;

    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    if (block_m == 64 and expected_tokens_per_expert > 4.0f) {
        const int num_n_blocks_per_expert = (2 * intermediate_hidden) / block_n;
        const int wave = std::max(1, num_sms / std::max(1, num_n_blocks_per_expert));
        return std::min(wave, num_experts_per_rank);
    }
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
    const int& block_m, const int& num_epilogue_threads, const int& l2_act_sf_gran_k) {
    // swapAB is ENABLED by default (the L1 SF-pool stride bug that corrupted
    // pool blocks >= 1 was fixed: BLOCK_M -> SF_BLOCK_M in the swapAB L1 epilogue).
    // Kill-switch retained: set DG_SM90_FP8_SWAP_AB=0 to force the non-swap path.
    if (deep_jit::get_env<int>("DG_SM90_FP8_SWAP_AB", 1) == 0)
        return false;
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    const bool decode_split_n_path =
        block_m == 64 and num_epilogue_threads == 256;
    // The swapAB L1 epilogue hands post-SwiGLU values across warpgroups before quantization, which is not bitwise stable run to run,
    // so swapAB is selected only where one warpgroup owns a whole L2 activation-SF group of the tile (the kernel asserts the same).
    constexpr int kSwapABWarpgroupOutputColumns = (128 / 2) / 2;
    if (decode_split_n_path and kSwapABWarpgroupOutputColumns < l2_act_sf_gran_k)
        return false;
    return decode_split_n_path and num_tokens <= 128 and expected_tokens_per_expert > 0.0f;
}

static std::tuple<int, int, int, bool> get_block_mn_config_for_mega_moe_sm90(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden, const int& l2_act_sf_gran_k) {
    const auto [block_m, num_epilogue_threads] = get_block_config_for_mega_moe_sm90(
        num_ranks, num_experts, num_topk, num_tokens);
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    const bool split_m2 =
        block_m == 128 and num_epilogue_threads == 256;
    const bool decode_split_n_path =
        block_m == 64 and num_epilogue_threads == 256;
    // The split-N decode tile's post-SwiGLU output (BLOCK_N / 2 columns) must be exactly one L2 activation-SF group (kernel
    // static_assert): with the per-128 SF only the 256-wide tile qualifies; the 128-wide tile is reserved for the per-64 recipe.
    // The 2-CTA cluster pairs adjacent N blocks, so both N block counts must be even at the chosen
    // BLOCK_N (scheduler/mega_moe.cuh kNumL1BlockNs/kNumL2BlockNs): 256-wide needs multiples of 512.
    const bool decode_tile_n_256_fits = (2 * intermediate_hidden) % 512 == 0 and hidden % 512 == 0;
    const bool decode_use_block_n_256 = decode_split_n_path and decode_tile_n_256_fits and
        (l2_act_sf_gran_k == 128 or (intermediate_hidden >= 2048 and expected_tokens_per_expert >= 0.25f));
    DG_HOST_ASSERT((not decode_split_n_path or decode_use_block_n_256 or l2_act_sf_gran_k == 64) &&
                   "the 128-wide decode tile needs the per-64 L2 activation scale: with the per-128 scale hidden and "
                   "2 x intermediate_hidden must be multiples of 512 (256-wide tile, even N block counts)");
    const bool use_swap_ab = (not decode_use_block_n_256) and
        should_use_swap_ab_for_mega_moe_sm90(
            num_experts_per_rank, num_tokens, num_topk,
            block_m, num_epilogue_threads, l2_act_sf_gran_k);
    const int block_n = use_swap_ab ? 128
                                    : (split_m2 ? 256 :
                                       (decode_use_block_n_256 ? 256 : 128));
    return {block_m, block_n, num_epilogue_threads, use_swap_ab};
}

// Derives the ring capacity from the wave size, mirroring SM100's causality
// (E_wave is decided first, the pool capacity is derived from it) instead of
// deriving E_wave from a user-supplied capacity. 0 = full-pool buffers (the auto lag ring from kSm90AutoLagMinTokens tokens
// per rank on), -1 = auto wave size via the SM100 occupancy heuristic, N > 0 = fixed wave size for the derivation only (at call time the wave
// size is re-derived from the stored capacity, like SM100).
static std::pair<int, int> get_num_ring_tokens_for_sm90_mega_moe(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_experts_per_wave_knob, const int& l2_act_sf_gran_k) {
    DG_HOST_ASSERT(num_experts_per_wave_knob >= -1);
    const int lag = get_sm90_l2_lag_units_for_buffer(num_max_tokens_per_rank, num_experts_per_wave_knob, num_topk, num_experts_per_rank);
    const int lag_encoded = get_sm90_l2_lag_encoded(lag);
    if (num_experts_per_wave_knob == 0 and lag == 0)
        return {0, 0};
    DG_HOST_ASSERT(num_max_tokens_per_rank % get_token_alignment_sm90() == 0 and
                   "num_max_tokens_per_rank must be token-aligned before deriving a ring capacity");
    // Size with the worst-case token count: block_m is monotone in num_tokens,
    // so the sizing block_m >= any call-time block_m and the call-time capacity
    // clamp never fires for buffers derived here.
    const auto [block_m, block_n, num_epilogue_threads, use_swap_ab] = get_block_mn_config_for_mega_moe_sm90(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, l2_act_sf_gran_k);
    const int num_max_pool_tokens = get_num_max_pool_tokens_sm90(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    if (num_experts_per_wave_knob == 0) {
        const int tokens = get_sm90_lag_ring_blocks(lag, sm90_lag_group_of(lag_encoded)) * 128;
        // Clamping to the full pool leaves the encoded lag describing more ring than is allocated, which is
        // safe: a pool-sized ring never wraps, so the lag schedule has no capacity constraint to violate (and the
        // launch skips the wrap assertion for it).
        return {std::min(math::align(tokens, get_token_alignment_sm90()), num_max_pool_tokens), lag_encoded};
    }
    int num_experts_per_wave;
    if (num_experts_per_wave_knob > 0) {
        num_experts_per_wave = std::min(num_experts_per_wave_knob, num_experts_per_rank);
    } else {
        // Auto: call the SM100 occupancy heuristic directly with the full pool
        // as the capacity (the clamp loop is a no-op since
        // wave_span(E_local) <= full pool). The `_sm90` wrapper is wrong here:
        // with ring == full pool its gate takes the legacy early returns.
        num_experts_per_wave = get_num_experts_per_wave_for_mega_moe_sm90_legacy(
            num_experts_per_rank, num_max_tokens_per_rank, num_topk,
            intermediate_hidden, block_m, block_n, runtime->get_num_sms(),
            num_max_pool_tokens, num_max_tokens_per_rank, num_ranks);
        // Floor at 2: a wave-1 ring holds exactly the recv working set, leaving no
        // headroom between producer and consumer.
        num_experts_per_wave = std::min(std::max(num_experts_per_wave, 2), num_experts_per_rank);
    }
    return {std::min(math::align(get_num_wave_pool_tokens_for_mega_moe_sm90(
                         num_ranks, num_topk, num_max_tokens_per_rank, num_experts_per_wave, block_m),
                     get_token_alignment_sm90()),
                     num_max_pool_tokens),
            lag_encoded};
}

// Weight-SF staging slot of one math warpgroup (floats; kernel `kNumWeightSFFloatsPerWG`); `wg_block_n` is the warpgroup's N extent.
static int get_sm90_weight_sf_floats_per_warpgroup(const int& hidden, const int& intermediate_hidden, const int& wg_block_n) {
    const int num_sf_groups_per_wg = wg_block_n >= 128 ? wg_block_n / 128 : 1;
    return std::max(2 * (hidden / 128), num_sf_groups_per_wg * (intermediate_hidden / 128));
}

// L2 stage mode 4 parks the four unwanted rows of every stmatrix.x4 in the warpgroup's weight-SF slot, which must hold at least this many bytes (kernel static_assert).
static constexpr int kSm90StmatrixJunkSlotBytes = 256;

static std::pair<int, int> get_pipeline_config_for_mega_moe_sm90(
    const int& smem_capacity,
    const int& num_experts, const int& hidden, const int& intermediate_hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const bool& use_swap_ab = false, const bool& half_l2_cd = false,
    const int& l2_cd_passes = 0,
    const int& tile_table_entries = 0,
    const int& l2_act_sf_gran_k = 64,
    const bool& early_combine = false,
    const int& pull_publish_batch = 1) {
    constexpr int kSmemAlignment = 1024;
    const int num_l2_cd_passes = l2_cd_passes != 0 ? l2_cd_passes : (half_l2_cd ? 2 : 1);

    const int smem_expert_count_bytes = num_experts * static_cast<int>(sizeof(uint32_t));
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_send_buffers_size;

    const int smem_cd_l1 = block_m * (block_n / 2);
    const int smem_cd_l2 = block_m * (block_n / num_l2_cd_passes) * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd_swap_l1 = use_swap_ab
        ? block_m * (block_n / 2) *
              (static_cast<int>(sizeof(float)) + static_cast<int>(sizeof(uint8_t)))
        : 0;
    const int smem_cd = align(
        std::max(std::max(smem_cd_l1, smem_cd_l2), smem_cd_swap_l1),
        kSmemAlignment);

    const int smem_sfa_per_stage =
        align((l2_act_sf_gran_k == block_k ? 1 : 2) * block_m * static_cast<int>(sizeof(float)), 128);
    const int smem_sfb_per_stage = 0;
    const int smem_per_stage = block_m * block_k + block_n * block_k +
                               smem_sfa_per_stage + smem_sfb_per_stage;

    const int num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int wg_block_n = (block_m == 64 and num_epilogue_warpgroups > 1) ? block_n / num_epilogue_warpgroups :
        ((block_m == 128 and block_n == 256 and num_epilogue_warpgroups == 4) ? 128 : block_n);
    const int weight_sf_floats_per_wg = get_sm90_weight_sf_floats_per_warpgroup(hidden, intermediate_hidden, wg_block_n);
    const int smem_weight_sf = align(
        num_epilogue_warpgroups * weight_sf_floats_per_wg * static_cast<int>(sizeof(float)), 128);

    const int smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps) * 8;
    const int smem_barriers_per_stage = 2 * 8;
    const int smem_early_combine = early_combine ? num_dispatch_warps * 8 + 64 + 8 : 0;
    const int smem_tile_table = tile_table_entries * 8;
    const int smem_tile_table_region = std::max(smem_tile_table, smem_expert_count_bytes) + 8;
    const int smem_pull_pending = pull_publish_batch > 1 ?
        align(num_dispatch_warps * pull_publish_batch * static_cast<int>(sizeof(uint32_t)), 16) + 8 : 0;
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_weight_sf + smem_barriers_fixed + smem_early_combine + smem_tile_table_region + smem_pull_pending;

    const int num_stages = (smem_capacity - smem_fixed) /
                           (smem_per_stage + smem_barriers_per_stage);
    DG_HOST_ASSERT(num_stages >= 2);
    const int smem_size = smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage);
    DG_HOST_ASSERT(smem_size <= smem_capacity);
    return {num_stages, smem_size};
}

static MegaMoESM90Config get_mega_moe_config_sm90(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens,
    const int& l2_act_sf_gran_k,
    const int& num_ring_tokens = 0,
    // > 0: the caller's bound on this call's token count on EVERY rank (the same value on every rank); 0: use the capacity
    const int& num_tokens_bound = 0,
    // encoded L2-lag schedule the symm buffer was sized for (the value the sizing returned); -1 = none, i.e. the wave schedule
    const int& l2_lag_encoded_in = -1) {
    const auto [block_m, block_n, num_epilogue_threads, use_swap_ab] = get_block_mn_config_for_mega_moe_sm90(
        num_ranks, num_experts, num_experts_per_rank,
        num_tokens, num_topk, hidden, intermediate_hidden, l2_act_sf_gran_k);
    const int l2_lag_encoded = l2_lag_encoded_in >= 0 ? l2_lag_encoded_in : 0;
    const int block_k = 128;
    const int num_sms = runtime->get_num_sms();
    const bool cluster_pairing_valid =
        num_sms % 2 == 0 and
        ((2 * intermediate_hidden) / block_n) % 2 == 0 and
        (hidden / block_n) % 2 == 0 and
        (2 * intermediate_hidden) % block_n == 0 and hidden % block_n == 0;
    const int cluster_size = (block_m == 128 and block_n == 256 and cluster_pairing_valid) ? 2 : 1;
    const bool multicast_on_b = cluster_size == 2;
    const int num_max_pool_tokens = get_num_max_pool_tokens_sm90(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    // 0 means "no ring": data pools sized by the full pool (legacy behavior).
    const int num_effective_ring_tokens = num_ring_tokens == 0 ? num_max_pool_tokens : num_ring_tokens;
    DG_HOST_ASSERT(num_tokens_bound == 0 or num_tokens_bound >= num_tokens);
    // the M-keyed rules below read the bound as given, so a bound above the capacity would pick a schedule for traffic this buffer cannot receive
    DG_HOST_ASSERT(num_tokens_bound <= num_max_tokens_per_rank);
    const int call_bound = num_tokens_bound > 0 ? num_tokens_bound : num_max_tokens_per_rank;
    const int num_wave_bound_tokens_per_rank =
        std::min(num_max_tokens_per_rank, align(std::max(call_bound, 1), get_token_alignment_sm90()));
    // a qualifying call on a lag ring whose worst-case single-wave pool fits runs the wave schedule (kL2LagUnits 0) on the same ring
    const int l2_lag_units_buffer = sm90_lag_units_of(l2_lag_encoded);
    const bool call_wave_schedule = l2_lag_units_buffer > 0 and num_ring_tokens != 0 and
        sm90_call_wave_class(block_m, num_tokens, num_tokens_bound) and
        sm90_single_wave_pool_fits(num_ranks, num_topk, num_wave_bound_tokens_per_rank, num_experts_per_rank, block_m, num_ring_tokens);
    // every other call on a lag ring runs the lag of its own per-rank token bound, never above the buffer's (the ring demand grows with the lag)
    const int l2_lag_units_call = std::min(l2_lag_units_buffer,
        get_sm90_auto_late_lag(num_wave_bound_tokens_per_rank, num_topk, num_experts_per_rank));
    DG_HOST_ASSERT(get_sm90_lag_ring_blocks(l2_lag_units_call, sm90_lag_group_of(l2_lag_encoded)) <=
                   get_sm90_lag_ring_blocks(l2_lag_units_buffer, sm90_lag_group_of(l2_lag_encoded)) &&
                   "the buffer's ring must hold the call's lag schedule");
    const int l2_lag_encoded_launch = call_wave_schedule ? 0 : get_sm90_l2_lag_encoded(l2_lag_units_call);
    const int l2_lag_units = sm90_lag_units_of(l2_lag_encoded_launch);
    DG_HOST_ASSERT((l2_lag_units == 0 or num_ring_tokens != 0) && "the L2-lag schedule needs a ring-sized buffer");
    if (num_ring_tokens != 0) {
        DG_HOST_ASSERT(num_ring_tokens % get_token_alignment_sm90() == 0);
        DG_HOST_ASSERT(num_ring_tokens <= num_max_pool_tokens);
        if (l2_lag_units > 0) {
            DG_HOST_ASSERT(num_ring_tokens % block_m == 0);
            // the wrap constraint is the allocation's own block count (get_sm90_lag_ring_blocks, sized at BLOCK_M 128 for the buffer's
            // lag; the call's lag and BLOCK_M are at most those). A ring clamped to the full pool never reuses a slot, so it is exempt
            if (num_ring_tokens < num_max_pool_tokens) {
                const int group = sm90_lag_group_of(l2_lag_encoded_launch);
                DG_HOST_ASSERT(num_ring_tokens / block_m >= get_sm90_lag_ring_blocks(l2_lag_units, group) &&
                               "Lag schedule: the ring must hold the lag + group units plus the in-flight blocks");
            }
        } else {
            const int num_min_ring_tokens = get_num_wave_pool_tokens_for_mega_moe_sm90(
                num_ranks, num_topk, call_wave_schedule ? num_wave_bound_tokens_per_rank : num_max_tokens_per_rank, 1, block_m);
            DG_HOST_ASSERT(num_ring_tokens >= num_min_ring_tokens &&
                           "Ring capacity must be within [tokens from all ranks, full pool]");
        }
    }
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;

    // The wave size is re-derived from the (derived-at-allocation) capacity via
    // the occupancy heuristic, with the capacity clamp evaluated at this call's per-rank token bound (num_tokens_bound).
    const int num_experts_per_wave = l2_lag_units > 0 ? num_experts_per_rank :
        get_num_experts_per_wave_for_mega_moe_sm90(
            num_experts_per_rank, num_tokens, num_topk,
            intermediate_hidden, block_m, block_n, num_sms,
            num_effective_ring_tokens, num_max_tokens_per_rank, num_ranks,
            num_wave_bound_tokens_per_rank, num_tokens_bound);
    DG_HOST_ASSERT((not call_wave_schedule or block_m != 128 or num_experts_per_wave == num_experts_per_rank) &&
                   "the BLOCK_M-128 per-call wave schedule expects one wave of all local experts");

    const bool reduce_decode_threads = num_epilogue_threads == 128;
    const bool decode_split_n =
        block_m == 64 and num_epilogue_threads == 256;
    const bool split_m2 =
        block_m == 128 and num_epilogue_threads == 256;
    const bool shrink_non_epilogue = reduce_decode_threads or decode_split_n or split_m2;
    const int num_dispatch_threads =
        (num_epilogue_threads == 512 or shrink_non_epilogue) ? 64 : 128;
    const bool split_sfa_loader_warp = false;
    const int num_non_epilogue_threads =
        split_sfa_loader_warp ? 128 :
            ((num_epilogue_threads == 512 or shrink_non_epilogue) ? 64 : 128);
    DG_HOST_ASSERT((num_dispatch_threads + num_non_epilogue_threads) % 128 == 0);

    // Early combine (kernel kEarlyCombineMode 1) is selected from hidden >= 512 x topk (the staging bound of the original dispatch-warp
    // receiver, kept as the selection threshold)
    const int early_combine_fits_smem = hidden >= 512 * num_topk ? 1 : 0;
    const int tile_table_entries = layout::get_sm90_tile_table_entries_compact<int>(
        num_max_pool_tokens, block_m, (2 * intermediate_hidden) / block_n, hidden / block_n, num_sms);
    const bool ring_mode_pull = num_ring_tokens != 0 and num_effective_ring_tokens < num_max_pool_tokens;
    const auto pull_publish_batch_fits_ring = [&](const int& n) {
        return not ring_mode_pull or
               n * num_sms * (num_dispatch_threads / 32) < (num_effective_ring_tokens / block_m - num_experts_per_rank - 1) * block_m;
    };
    const bool pull_publish_rule_on = sm90_rule_tokens_per_rank(num_tokens, num_tokens_bound) >= kSm90PullPublishMinTokens and
        pull_publish_batch_fits_ring(kSm90PullPublishBatch);
    const int pull_publish_batch = pull_publish_rule_on ? kSm90PullPublishBatch : 1;
    // Publish-batch bound (ring mode): a dispatch warp at pool block p waits for the L1 consumers of block p - R (R = ring blocks),
    // which need every row of that block published. The kernel publishes a warp's pending rows before it blocks on a slot; this
    // bound additionally keeps the batch's unpublished span (N x 264 / block_m + one partial tail block per expert) inside the ring.
    if (pull_publish_batch > 1 and ring_mode_pull) {
        DG_HOST_ASSERT(pull_publish_batch_fits_ring(pull_publish_batch) &&
                       "publish batch too large for the ring: a warp's pending rows could be needed by the ring slot it waits for");
    }
    auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
        SM90ArchSpec::smem_capacity,
        num_experts, hidden, intermediate_hidden,
        block_m, block_n, block_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        use_swap_ab, /*half_l2_cd=*/false,
        0, tile_table_entries, l2_act_sf_gran_k, early_combine_fits_smem != 0, pull_publish_batch);

    const bool split_phase_prefill =
        block_m == 128 and block_n == 256 and hidden >= 4096;
    bool half_l2_cd = false;
    int l2_cd_passes = 0;
    int l2_stage_mode = 0;
    if (split_m2) {
        // row passes on the quarter-width buffer: stmatrix (mode 4) when the weight-SF slot holds kSm90StmatrixJunkSlotBytes, else plain stores (mode 2)
        const int weight_sf_slot_bytes =
            get_sm90_weight_sf_floats_per_warpgroup(hidden, intermediate_hidden, block_n) * static_cast<int>(sizeof(float));
        l2_stage_mode = use_swap_ab ? 0 : (weight_sf_slot_bytes >= kSm90StmatrixJunkSlotBytes ? 4 : 2);
        l2_cd_passes = l2_stage_mode != 0 ? 4 : 2;
        const auto [ns_half, sz_half] = get_pipeline_config_for_mega_moe_sm90(
            SM90ArchSpec::smem_capacity,
            num_experts, hidden, intermediate_hidden,
            block_m, block_n, block_k,
            num_dispatch_threads / 32, num_epilogue_threads / 32,
            use_swap_ab, /*half_l2_cd=*/true,
            l2_cd_passes, tile_table_entries, l2_act_sf_gran_k, early_combine_fits_smem != 0, pull_publish_batch);
        half_l2_cd = true;
        num_stages = ns_half;
        smem_size = sz_half;
    } else if (split_phase_prefill and num_stages < 3) {
        const auto [ns_half, sz_half] = get_pipeline_config_for_mega_moe_sm90(
            SM90ArchSpec::smem_capacity,
            num_experts, hidden, intermediate_hidden,
            block_m, block_n, block_k,
            num_dispatch_threads / 32, num_epilogue_threads / 32,
            use_swap_ab, /*half_l2_cd=*/true,
            0, tile_table_entries, l2_act_sf_gran_k, early_combine_fits_smem != 0, pull_publish_batch);
        if (ns_half > num_stages) {
            half_l2_cd = true;
            num_stages = ns_half;
            smem_size = sz_half;
        }
    }

    // The early-combine signal reads the k-block kNumStages ahead of the current tile, so both GEMMs need
    // strictly more k-blocks than stages (impls/sm90_fp8_mega_moe.cuh, kEarlyCombine). num_stages is only
    // final here; the smem above was sized with early combine on, so demoting now over-allocates, never under.
    const int early_combine = (early_combine_fits_smem != 0 and
                               hidden / block_k > num_stages and
                               intermediate_hidden / block_k > num_stages) ? 1 : 0;

    const auto config = MegaMoESM90Config {
        block_m, block_n, block_k,
        cluster_size,
        num_max_pool_tokens, num_padded_sf_pool_tokens,
        num_effective_ring_tokens,
        swizzle_acts_mode, swizzle_weights_mode,
        num_experts_per_wave,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
        half_l2_cd,
        multicast_on_b,
        l2_lag_encoded_launch,
        l2_cd_passes,
        l2_stage_mode,
        early_combine,
        pull_publish_batch
    };

    if (deep_jit::get_env<int>("DG_JIT_DEBUG") or deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = std::format(
            "MegaMoESM90Config(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_tokens_bound={}, num_topk={}, swap_ab={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens,
            num_tokens_bound, num_topk, use_swap_ab);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
