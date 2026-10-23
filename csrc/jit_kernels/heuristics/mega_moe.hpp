#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <format>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_jit/utils/env.hpp>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "sm100.hpp"

namespace deep_gemm {

struct MegaMoEConfig {
    // Block tiling
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int store_block_m_l1, store_block_m_l2;

    // SF block sizes (UTCCP 128-aligned)
    int sf_block_m, sf_block_n;
    // SF granularity along K
    int gran_k;

    // Ring capacity and SF ring token count
    int num_ring_tokens;
    int num_sf_ring_tokens;

    // Swizzle modes for TMA descriptors
    int swizzle_acts_mode, swizzle_weights_mode;

    // Pipeline stages and shared memory
    int num_stages, smem_size;

    // Thread layout
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    // Dispatch pull config
    int num_bytes_per_pull;

    friend std::ostream& operator << (std::ostream& os, const MegaMoEConfig& config) {
        os << "MegaMoEConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m_l1=" << config.store_block_m_l1 << ", store_block_m_l2=" << config.store_block_m_l2
           << ", sf_block_m=" << config.sf_block_m << ", sf_block_n=" << config.sf_block_n
           << ", gran_k=" << config.gran_k
           << ", num_ring_tokens=" << config.num_ring_tokens
           << ", num_sf_ring_tokens=" << config.num_sf_ring_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", num_bytes_per_pull=" << config.num_bytes_per_pull << ")";
        return os;
    }
};

// Locality domains are read per physical SM, so a grid smaller than the GPU stays correct
static int get_num_sms_for_mega_moe() {
    const auto num_sms = runtime->get_num_sms();
    DG_HOST_ASSERT(num_sms > 0 and num_sms % 2 == 0 and num_sms <= jit->device.get_num_sms());
    return num_sms;
}

static MmaKind parse_mma_kind(const std::string& mma_type_str) {
    if (mma_type_str == "bf16xbf16")
        return MmaKind::BF16;
    if (mma_type_str == "fp8xfp4" or mma_type_str == "fp8xfp8")
        return MmaKind::MXFP8FP4;
    if (mma_type_str == "mxf4xmxf4")
        return MmaKind::MXFP4;
    DG_HOST_ASSERT(mma_type_str == "nvfp4xnvfp4");
    return MmaKind::NVFP4;
}

static std::string to_mma_type_string(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::BF16:     return "bf16xbf16";
        case MmaKind::MXFP8FP4: return "fp8xfp4";
        case MmaKind::MXFP4:    return "mxf4xmxf4";
        case MmaKind::NVFP4:    return "nvfp4xnvfp4";
    }
    DG_HOST_UNREACHABLE("Unknown MMA kind");
}

static std::string to_mma_kind_name(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::BF16:     return "MmaKind::BF16";
        case MmaKind::MXFP8FP4: return "MmaKind::MXFP8FP4";
        case MmaKind::MXFP4:    return "MmaKind::MXFP4";
        case MmaKind::NVFP4:    return "MmaKind::NVFP4";
    }
    DG_HOST_UNREACHABLE("Unknown MMA kind");
}

static bool is_mma_with_sf(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::MXFP8FP4 or mma_kind == MmaKind::MXFP4 or mma_kind == MmaKind::NVFP4;
}

static bool is_mxf4_mma_kind(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::MXFP4;
}

static bool is_nvfp4_mma_kind(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::NVFP4;
}

static std::tuple<int, int, int, int, int, int> get_block_config_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens,
    const MmaKind& mma_kind,
    const std::optional<int64_t>& global_num_tokens) {
    // Experts receive tokens from every rank, so an idle or lightly loaded rank still computes remote load.
    // Without a caller hint, assume all ranks match the local count.
    DG_HOST_ASSERT(not global_num_tokens.has_value() or
                   (*global_num_tokens >= 0 and
                    *global_num_tokens <= static_cast<int64_t>(num_max_tokens_per_rank) * num_ranks));
    const float num_total_tokens = global_num_tokens.has_value() ?
        static_cast<float>(*global_num_tokens) : static_cast<float>(num_tokens) * num_ranks;

    // Expected tokens per expert, plus a one-sigma routing margin for the tile choice
    const float num_expected_tokens = num_total_tokens * num_topk / num_experts;
    const float num_covered_tokens = num_expected_tokens + std::sqrt(num_expected_tokens);

    // Every M block costs roughly 128 extra rows (its tasks reload the weight tiles), so use the fewest blocks
    // per expert and the smallest tile covering them: one block up to 192 rows, then blocks of up to 240 rows.
    // A single 240-row block per expert only pays off with many local experts (about 14 or more blocks per rank).
    int block_m = num_expected_tokens <= 10 ? 16 : 32;
    if (num_expected_tokens <= 24 and deep_jit::get_env<int>("DG_MEGA_MOE_SKEWED_ROUTING"))
        block_m = num_expected_tokens <= 16 ? 64 : 128;
    if (num_expected_tokens > 24) {
        int num_blocks = static_cast<int>(std::ceil(num_covered_tokens / 240));
        if (num_blocks == 1 and num_covered_tokens > 192 and num_experts / num_ranks < 14)
            num_blocks = 2;
        for (const int& candidate: {64, 128, 192, 240}) {
            block_m = candidate;
            if (num_blocks * candidate >= num_covered_tokens)
                break;
        }
    }

    // Must divide `WG_BLOCK_M` (= block_m / 2), which pins each arm: 240 -> 120 and
    // 128 -> 64 cannot take 48, 192 -> 96 can and halves its store iterations.
    // Use smaller, asymmetric store blocks at FP8xFP4 prefill to save shared memory for sufficient TMA pipeline depth
    int store_block_m_l1 = block_m <= 16 ? 8 : block_m <= 64 ? 16 : block_m <= 128 ? 32 : block_m <= 192 ? 48 :
                           mma_kind == MmaKind::MXFP8FP4 ? 24 : 40;
    int store_block_m_l2 = (block_m == 240 and mma_kind == MmaKind::MXFP8FP4) ? 8 : store_block_m_l1;
    int block_k = 128 * 8 / get_element_bits(mma_kind);
    int cluster_size = 2, num_epilogue_warpgroups = 2;

    // Offline tuning hook: "cluster,block_m,store_block_m,block_k,epilogue_warpgroups",
    // `block_k` in the pre-scaling 8-bit units above; `store_block_m` sets both L1 and L2.
    if (const auto override_str = deep_jit::get_env<std::string>("DG_MEGA_MOE_TILES"); not override_str.empty()) {
        int v[5];
        const int n = std::sscanf(override_str.c_str(), "%d,%d,%d,%d,%d",
                                  v + 0, v + 1, v + 2, v + 3, v + 4);
        DG_HOST_ASSERT(n == 5 and "DG_MEGA_MOE_TILES needs 5 comma-separated ints");
        cluster_size = v[0], block_m = v[1], store_block_m_l1 = store_block_m_l2 = v[2];
        block_k = v[3] * 8 / get_element_bits(mma_kind), num_epilogue_warpgroups = v[4];
    }

    // Check whether our `block_m` lies in `kCandidateBlockM`
    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );

    return {cluster_size, block_m, store_block_m_l1, store_block_m_l2, block_k, num_epilogue_warpgroups * 128};
}

static std::pair<int, int> get_pipeline_config_for_mega_moe(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_bytes_per_pull, const int& store_block_m_l1, const int& store_block_m_l2,
    const int& sf_block_m, const int& sf_block_n, const int& gran_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const MmaKind& mma_kind) {
    constexpr int kSmemAlignment = 1024;
    constexpr int kNumEpilogueStages = 2;
    constexpr int kNumTMAStoreStages = 2;
    const int elem_bits = get_element_bits(mma_kind);

    // Always multicast on A
    const int load_block_m = block_m / 2;

    // Dispatch region
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(num_bytes_per_pull), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    // C/D output region: max of L1 output staging and L2 BF16 staging.
    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int smem_cd_l1 = num_epilogue_warpgroups * store_block_m_l1 * (block_n / 2) * kNumTMAStoreStages * elem_bits / 8;
    const int smem_cd_l2 = num_epilogue_warpgroups * store_block_m_l2 * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = align(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);

    // Schedule task payloads
    constexpr int kNumScheduleStages = 2;
    const int smem_task_info = kNumScheduleStages * static_cast<int>(sizeof(sched::TaskInfo<true>));

    // Barriers (stage-independent): dispatch + tensor memory full/empty + combine (2 per epilogue warp)
    // + schedule task publish full/empty barriers.
    const int smem_barriers = (num_dispatch_warps + kNumEpilogueStages * 2 + num_epilogue_warps * 2 + kNumScheduleStages * 2) * 8;

    // Amax warp-pair reduction buffer for SwiGLU's cross-warp amax exchange.
    const int smem_amax_reduction = is_mma_with_sf(mma_kind) ?
        store_block_m_l1 * num_epilogue_warps * static_cast<int>(sizeof(float)) : 0;

    // Tensor memory pointer
    const int smem_tmem_ptr = 4;

    const int smem_sf_inv_lut = mma_kind == MmaKind::NVFP4 ?
        256 * static_cast<int>(sizeof(float)) : 0;

    // SF is aligned to UTCCP 128-element granularity
    const int smem_sfa_per_stage = is_mma_with_sf(mma_kind) ? sf_block_m * (block_k / gran_k) : 0;
    const int smem_sfb_per_stage = is_mma_with_sf(mma_kind) ? sf_block_n * (block_k / gran_k) : 0;

    // Per-stage: A tile + B tile + optional SF tiles + full/empty barriers.
    // NOTES: for `MXFP8FP4` the FP4 weights are unpacked into 8-bit containers in smem,
    // so A and B always share the same per-element footprint
    const int smem_a_size_per_stage = load_block_m * block_k * elem_bits / 8;
    const int smem_b_size_per_stage = block_n * block_k * elem_bits / 8;
    DG_HOST_ASSERT(smem_a_size_per_stage % kSmemAlignment == 0);
    DG_HOST_ASSERT(smem_b_size_per_stage % kSmemAlignment == 0);
    const int smem_stage_barriers = 2 * 8;
    const int smem_size_per_stage = smem_a_size_per_stage + smem_b_size_per_stage + smem_sfa_per_stage + smem_sfb_per_stage + smem_stage_barriers;

    // Fixed total
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_amax_reduction + smem_barriers +
        smem_task_info + smem_tmem_ptr + smem_sf_inv_lut;

    // Select maximum number of stages
    const int num_stages = (smem_capacity - smem_fixed) / smem_size_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);

    return {num_stages, smem_fixed + num_stages * smem_size_per_stage};
}

static MegaMoEConfig get_mega_moe_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_ring_tokens,
    const int& num_sf_ring_tokens,
    const MmaKind& mma_kind,
    const std::optional<int64_t>& global_num_tokens) {

    // Block config
    const auto [cluster_size, block_m, store_block_m_l1, store_block_m_l2, block_k, num_epilogue_threads] =
        get_block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens,
                                      mma_kind, global_num_tokens);
    DG_HOST_ASSERT(num_ring_tokens % block_m == 0);
    const int block_n = 128;
    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const auto [sf_block_m, sf_block_n] = is_mma_with_sf(mma_kind) ?
        SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(block_m, block_n, mma_kind) : std::pair(0, 0);
    // NOTES: FP8 activations and FP4 weights (unpacked to 8-bit in smem) both use 128B swizzle;
    // NVFP4 keeps FP4 packed, so a 128B swizzle atom covers twice as many elements
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;
    const int gran_k = is_mma_with_sf(mma_kind) ? get_sf_gran_k(mma_kind) : 32;

    // Thread layout
    const int num_dispatch_threads = 128;
    const int num_non_epilogue_threads = 128;

    // Pull: divide token bytes by 2 until <= num_max_pull_bytes
    const int num_max_pull_bytes = is_mma_with_sf(mma_kind) ? 8192 : 4096;
    int num_bytes_per_pull = hidden * get_element_bits(mma_kind) / 8;
    while (num_bytes_per_pull > num_max_pull_bytes) {
        DG_HOST_ASSERT(num_bytes_per_pull % 2 == 0);
        num_bytes_per_pull /= 2;
    }

    // Pipeline
    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe(
        SM100ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k, num_bytes_per_pull, store_block_m_l1, store_block_m_l2,
        sf_block_m, sf_block_n, gran_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        mma_kind);

    const auto config = MegaMoEConfig {
        block_m, block_n, block_k,
        load_block_m, load_block_n, store_block_m_l1, store_block_m_l2,
        sf_block_m, sf_block_n, gran_k,
        num_ring_tokens, is_mma_with_sf(mma_kind) ? num_sf_ring_tokens : 0,
        swizzle_acts_mode, swizzle_weights_mode,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
        num_bytes_per_pull
    };

    // Print configs for the first time
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = std::format(
            "MegaMoEConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, global_num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens,
            global_num_tokens.value_or(-1), num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
