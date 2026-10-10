#pragma once

#include <algorithm>
#include <unordered_set>

#include <deep_gemm/layout/mega_moe.cuh>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include <deep_jit/utils/env.hpp>
#include <format>
#include <iostream>
#include "sm100.hpp"

namespace deep_gemm {

struct MegaMoEM2NConfig {
    // Block tiling
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int store_block_m;

    // SF block sizes (UTCCP 128-aligned)
    int sf_block_m, sf_block_n;

    // Pool capacity and SF-padded token count
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;

    // Swizzle modes for TMA descriptors
    int swizzle_acts_mode, swizzle_weights_mode;

    // Number of experts to process per wave
    int num_experts_per_wave;

    // Pipeline stages and shared memory
    int num_stages, smem_size;

    // Thread layout
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;
    int num_dispatch_pull_stages;

    friend std::ostream& operator << (std::ostream& os, const MegaMoEM2NConfig& config) {
        os << "MegaMoEM2NConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m=" << config.store_block_m
           << ", sf_block_m=" << config.sf_block_m << ", sf_block_n=" << config.sf_block_n
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", num_dispatch_pull_stages=" << config.num_dispatch_pull_stages << ")";
        return os;
    }
};

static std::pair<int, int> get_pipeline_config_for_mega_moe_m2n(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k, const int& store_block_m,
    const int& sf_block_m, const int& sf_block_n,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const int& num_dispatch_pull_stages,
    const bool& use_nvfp4) {
    constexpr int kSmemAlignment = 1024;
    constexpr int kNumEpilogueStages = 2;
    constexpr int kNumTMAStoreStages = 2;
    DG_HOST_ASSERT(num_dispatch_pull_stages >= 1);

    // Always multicast on A
    const int load_block_m = block_m / 2;

    // Dispatch region
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(
            layout::Data(use_nvfp4 ? hidden / 2 : hidden), num_dispatch_warps,
            num_dispatch_pull_stages).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    // C/D output region: max of L1 FP8 (2 TMA stages, BLOCK_N/2 post-SwiGLU) and L2 BF16 (1 stage)
    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int packing = use_nvfp4 ? 2 : 1;
    const int sf_chunks = block_k / ((use_nvfp4 ? 16 : 32) * 4);
    const int smem_cd_l1 = num_epilogue_warpgroups * store_block_m * (block_n / 2) / packing * kNumTMAStoreStages;
    const int smem_cd_l2 = num_epilogue_warpgroups * store_block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = std::max(smem_cd_l1, smem_cd_l2);

    // Barriers (stage-independent): dispatch + tensor memory full/empty + combine (2 per epilogue warp)
    const int smem_barriers = (
        num_dispatch_warps * num_dispatch_pull_stages +
        kNumEpilogueStages * 2 + num_epilogue_warps * 2) * 8;

    // Amax reduction
    const int smem_amax_reduction = store_block_m * num_epilogue_warps * static_cast<int>(sizeof(float));

    // Tensor memory pointer
    const int smem_tmem_ptr = 4;

    // SF is aligned to UTCCP 128-element granularity
    const int smem_sfa_per_stage = sf_block_m * 4 * sf_chunks;
    const int smem_sfb_per_stage = sf_block_n * 4 * sf_chunks;

    // Per-stage: A tile + B tile + SFA tile + SFB tile + full/empty barriers.
    const int smem_a_per_stage = load_block_m * block_k / packing;
    const int smem_b_per_stage = block_n * block_k / packing;
    const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage + 2 * 8;

    // Fixed total
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_amax_reduction + smem_barriers + smem_tmem_ptr;

    // Select maximum num_stages
    const int num_stages = (smem_capacity - smem_fixed) / smem_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);

    return {num_stages, smem_fixed + num_stages * smem_per_stage};
}

static MegaMoEM2NConfig get_mega_moe_m2n_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens,
    const bool& use_nvfp4) {
    // M2N keeps its full-pool scheduler and activation pipeline independent
    // of the native MegaMoE ring-buffer configuration.
    const int block_m = 64;
    const int store_block_m = 32;
    const int num_epilogue_threads = 128;
    const int num_dispatch_pull_stages = 1;
    const int block_n = 128;
    const int block_k = use_nvfp4 ? 256 : 128;
    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const auto [sf_block_m, sf_block_n] = SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(block_m, block_n, MmaKind::MXFP8FP4);
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    // NOTES: FP8 activations and FP4 weights (unpacked to 8-bit in smem) both use 128B swizzle
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;

    // Waves
    const int num_experts_per_wave = num_experts_per_rank;

    // Thread layout
    const int num_dispatch_threads = 128;
    const int num_non_epilogue_threads = 128;

    // Pipeline
    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_m2n(
        SM100ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k, store_block_m,
        sf_block_m, sf_block_n,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        num_dispatch_pull_stages, use_nvfp4);

    const auto config = MegaMoEM2NConfig {
        block_m, block_n, block_k,
        load_block_m, load_block_n, store_block_m,
        sf_block_m, sf_block_n,
        num_max_pool_tokens, num_padded_sf_pool_tokens,
        swizzle_acts_mode, swizzle_weights_mode,
        num_experts_per_wave,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
        num_dispatch_pull_stages
    };

    // Print configs for the first time
    if (deep_jit::get_env<int>("DG_JIT_DEBUG") or deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = std::format(
            "MegaMoEM2NConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={}, use_nvfp4={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk, use_nvfp4);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
