#pragma once

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumBytesPerPull,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    // Stream A0.1: L1 epilogue quantizes SwiGLU output to FP4 (E2M1) + UE8M0 SF
    // instead of FP8 (E4M3) + UE8M0 SF; halves the per-row gmem footprint.
    bool kUseFp4Acts = false,
    // Stream A0.5: when set (and kUseFp4Acts also set), L1 + L2 mainloops run
    // kind::mxf4 (K=64 dense FP4) instead of kind::mxf8f6f4 (K=32 with-padding).
    bool kUseMxf4Kind = false,
    // Stream B: when set, the L2 epilogue ships FP8 E4M3 + per-(token, N=128)
    // UE8M0 SF over NVLink instead of BF16. Independent of the FP4 flags.
    bool kUseFp8Combine = false,
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_mega_moe_impl(void* y,
                            int* cumulative_local_expert_recv_stats,
                            const uint32_t num_tokens,
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;

    // Template checks
    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue and combine threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");

    // Thread indices
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights_sf);
    }

    // Workspaces
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    // Token and buffer layouts
    // Stream A0.0b: under kUseFp4Acts the symmetric `x` slot (and the L1 token
    // pool that mirrors it) holds packed E2M1 (kHidden/2 bytes/token) instead
    // of dense E4M3 (kHidden bytes/token). The SF slot is unchanged.
    constexpr uint32_t kInputTokenBytes = kUseFp4Acts ? (kHidden / 2) : kHidden;
    constexpr auto fp8_token_layout = layout::Data(kInputTokenBytes);
    constexpr auto bf16_token_layout = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    constexpr auto fp8_sf_layout = layout::Data(kHidden / 32);
    constexpr auto fp8_intermediate_sf_layout = layout::Data(kIntermediateHidden / 32);
    constexpr auto input_topk_idx_layout = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    // Registered inputs
    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxTokensPerRank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumMaxTokensPerRank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, kNumMaxTokensPerRank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, kNumMaxTokensPerRank,
        input_topk_idx_buffer.get_end_ptr());

    // SF and its buffer configs
    constexpr uint32_t kGranK = 32;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "No padding is needed for SFB");

    // UTCCP 4x32 transpose index mapping within each 128-element group
    const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
               (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    // L1 inputs
    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxPoolTokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumPaddedSFPoolTokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, kNumMaxPoolTokens,
        l1_sf_buffer.get_end_ptr());

    // L2 inputs
    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, kNumMaxPoolTokens,
        l1_topk_weights_buffer.get_end_ptr()
    );
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens,
        l2_token_buffer.get_end_ptr()
    );

    // Combine inputs.
    // Stream B: under kUseFp8Combine the slot holds FP8 E4M3 (kHidden bytes/token)
    // + a separate SF slot holding UE8M0 bytes (kHidden / kCombineGranK bytes/token).
    // Off → BF16 (kHidden*2 bytes/token), zero-sized SF slot.
    constexpr uint32_t kCombineGranK = 128;
    DG_STATIC_ASSERT(kHidden % kCombineGranK == 0, "kHidden must be a multiple of 128 for FP8 combine SF");
    constexpr auto combine_token_layout = layout::Data(
        kUseFp8Combine ? kHidden : (kHidden * 2));
    constexpr auto combine_sf_layout = layout::Data(
        kUseFp8Combine ? (kHidden / kCombineGranK) : 0,
        /*require_tma_alignment=*/false);
    const auto combine_token_buffer = layout::Buffer(
        combine_token_layout, kNumTopk, kNumMaxTokensPerRank,
        l2_sf_buffer.get_end_ptr()
    );
    const auto combine_sf_buffer = layout::Buffer(
        combine_sf_layout, kNumTopk, kNumMaxTokensPerRank,
        combine_token_buffer.get_end_ptr()
    );

    // Data types
    // NOTES: activations are FP8 (e4m3), weights are FP4 (e2m1)
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::detail::float_e2m1_unpacksmem_t;
    // Stream A0.2 / A0.0b: under kUseFp4Acts both L1 and L2 read A as E2M1.
    // Same 1-byte smem footprint as E4M3 (unpacksmem padded), so smem sizing,
    // swizzle, and umma_desc strides are unchanged; only the MMA instruction
    // descriptor's A-dtype field and the source-side TMA expect_tx differ.
    using l2_a_dtype_t = cute::conditional_t<kUseFp4Acts, b_dtype_t, a_dtype_t>;
    using l1_a_dtype_t = cute::conditional_t<kUseFp4Acts, b_dtype_t, a_dtype_t>;

    // MMA configs
    // NOTES: always swap A/B, 2-CTA MMA, and matrices are K-major
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    constexpr uint32_t UMMA_BLOCK_K = 128;
    // Stream A0.5: kind::mxf4 runs K=64 dense per MMA call (vs K=32 for
    // kind::mxf8f6f4). The number of MMA calls per UMMA_BLOCK_K halves (4 → 2).
    constexpr uint32_t UMMA_K = kUseMxf4Kind ? 64 : 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;  // Multicast on A
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Invalid block N");

    // Swizzle configs
    // Stream A0.5: under kUseMxf4Kind, A/B smem use dense FP4 (_ALIGN8B,
    // 2 nibbles/byte) instead of with-padding (_ALIGN16B, 1 byte/elem). The
    // per-128-element K-atom byte width halves (128 → 64), so swizzle mode
    // tracks it. (#347 hardcodes 128 = UMMA_BLOCK_K * 1 byte for the FP8 /
    // with-padding FP4 layouts.)
    constexpr uint32_t kSwizzleAMode = kUseMxf4Kind ? 64 : 128;
    constexpr uint32_t kSwizzleBMode = kUseMxf4Kind ? 64 : 128;
    DG_STATIC_ASSERT(sizeof(l2_a_dtype_t) == sizeof(a_dtype_t), "L2 A dtype must match A in smem footprint");
    DG_STATIC_ASSERT(sizeof(l1_a_dtype_t) == sizeof(a_dtype_t), "L1 A dtype must match A in smem footprint");
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N % kSwizzleCDMode == 0, "Invalid block N");

    // Epilogue configs
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Shared memory sizes
    // NOTES: FP8 CD output for L1 (2 TMA stages, BLOCK_N/2 post-SwiGLU), BF16 output for L2 (no TMA, a single stage)
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;
    // Stream A0.1: FP4 acts pack 2 elements/byte → L1 output row byte-stride halves.
    constexpr uint32_t L1_OUT_ROW_BYTES = kUseFp4Acts ? (L1_OUT_BLOCK_N / 2) : L1_OUT_BLOCK_N;
    constexpr uint32_t AMAX_REDUCTION_WARP_BUFFER_SIZE = STORE_BLOCK_M / 2; // float2

    struct SharedStorage {
        alignas(kSharedMemoryAlignment) uint32_t expert_token_count[kNumExperts];
        alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumDispatchWarps][kNumBytesPerPull];
        union {
            alignas(kSharedMemoryAlignment) cutlass::float_e4m3_t l1[kNumEpilogueWarpgroups][kNumTMAStoreStages][STORE_BLOCK_M * L1_OUT_BLOCK_N];
            alignas(kSharedMemoryAlignment) nv_bfloat16 l2[kNumEpilogueWarpgroups][STORE_BLOCK_M * BLOCK_N];
        } smem_d;
        // Stream A0.5: dense FP4 (mxf4) halves the per-stage A/B byte footprint.
        // a_dtype_t/b_dtype_t are 1 byte, so element count == byte count, and
        // `sizeof(SharedStorage::smem_a[0])` stays the authoritative per-stage
        // byte size for both layouts.
        alignas(kSharedMemoryAlignment) a_dtype_t smem_a[kNumStages][kUseMxf4Kind ? (LOAD_BLOCK_M * BLOCK_K / 2) : (LOAD_BLOCK_M * BLOCK_K)];
        alignas(kSharedMemoryAlignment) b_dtype_t smem_b[kNumStages][kUseMxf4Kind ? (LOAD_BLOCK_N * BLOCK_K / 2) : (LOAD_BLOCK_N * BLOCK_K)];
        uint32_t smem_sfa[kNumStages][SF_BLOCK_M * (BLOCK_K / 128)];
        uint32_t smem_sfb[kNumStages][SF_BLOCK_N * (BLOCK_K / 128)];
        float2 amax_reduction[kNumEpilogueWarps][AMAX_REDUCTION_WARP_BUFFER_SIZE];
        Barrier dispatch_barriers[kNumDispatchWarps];
        Barrier full_barriers[kNumStages];
        Barrier empty_barriers[kNumStages];
        Barrier tmem_full_barriers[kNumEpilogueStages];
        Barrier tmem_empty_barriers[kNumEpilogueStages];
        Barrier combine_barriers[kNumEpilogueWarps * 2];
        uint32_t tmem_ptr_in_smem;
    };
    constexpr uint32_t kNumReusableSmemBytes = offsetof(SharedStorage, dispatch_barriers);
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);

    // Send buffers
    constexpr auto pull_layout = layout::Data(kNumBytesPerPull);
    const auto smem_send_buffers = layout::Buffer(
        pull_layout, kNumDispatchWarps, 1,
        static_cast<void*>(shared_storage.dispatch_send_buffer));

    // Tensor memory size
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // A cluster sync is essential for 2CTA tensor memory allocation
    comm::cluster_sync_with_relaxed_arrive();

    // Initialization
    if (warp_idx == 0) {
        // Clean shared memory
        if (cute::elect_one_sync()) {
            // The bytes must be 8 bytes aligned
            ptx::st_shared_bulk(
                shared_storage.expert_token_count,
                math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment)
            );
        }
    } else if (warp_idx == 1) {
        // Init m-barriers for dispatch
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            shared_storage.dispatch_barriers[i].init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Arrive at 2 CTAs, A + B
                shared_storage.full_barriers[i].init(2 * 2);
                shared_storage.empty_barriers[i].init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                // Arrive at all CTAs
                shared_storage.tmem_full_barriers[i].init(1);
                // Arrive only at the leader CTA
                shared_storage.tmem_empty_barriers[i].init(2 * kNumEpilogueThreads);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                shared_storage.combine_barriers[i].init(1);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 3) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, &shared_storage.tmem_ptr_in_smem);
    }
    // NOTES: Using `.relaxed` is allowed here since `fence_barrier_init` is `.release.cluster`,
    // and `barrier.cluster.wait.aligned` is by default `.acquire`
    comm::cluster_sync_with_relaxed_arrive();

    // Task scheduler
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumExpertsPerWave,
        kNumSMs, kNumRanks>(workspace);

    // MMA pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM Barrier indices
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;

    // NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 3;

    // Adjust registers
    // NOTES: more experts per rank will cost more schedulers' registers
    constexpr bool kUseMoreEpilogueRegisters = kNumExpertsPerRank <= 64;
    constexpr uint32_t kNumDispatchRegisters = kUseMoreEpilogueRegisters ? 48 : 96;
    constexpr uint32_t kNumNonEpilogueRegisters = kUseMoreEpilogueRegisters ? 40 : 88;
    constexpr uint32_t kNumEpilogueRegisters = kUseMoreEpilogueRegisters ? 208 : 160;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    // Grid sync index assignments (dispatch and epilogue use separate counters to avoid conflicts)
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        // Dispatch warps
        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            // TODO: figure out better unrolling
            // Now, `unroll` is better than `unroll 8`
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                // Allocate slots for each token-topk
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // Count experts' tokens
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
           atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Get SM offset (~6.5 us)
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(shared_storage.expert_token_count[i]);
            shared_storage.expert_token_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source indices (~2 us with 512 tokens)
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

        // Grid sync
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        // Write expert count
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Barrier before pulling
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
            /* After the NVLink barrier, there is a grid sync */ true
        );

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Pull token data and SF from remote ranks into local L1 buffer
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = &shared_storage.dispatch_barriers[warp_idx];

        // Cache expert token counts in registers (same pattern as scheduler)
        scheduler.fetch_expert_recv_count();

        // Per-rank counts for current expert (re-loaded when expert changes)
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
            // Advance expert until within the range
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;

                // Update pool block offset for the new expert
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);

                // Move start and end to the next expert
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }

            // Finish all tokens
            if (current_expert_idx >= kNumExpertsPerRank)
                break;

            // Load per-rank counts when expert changes
            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    // TODO: this is not coalesced
                    stored_rank_count[i] = j < kNumRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }

            // Round-robin rank selection via iterative min-peeling
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                // Compute active count and min across all ranks
                // NOTES: reduce within each lane first, then warp-reduce once
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                // Hit in the current round
                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }

                // Move into the next round
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] -= cute::min(remaining[i], length);
            }

            // Read source token-topk index (written by remote dispatch via NVLink)
            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;

            // Hidden bytes are divided into chunks.
            // Stream A0.0b: under kUseFp4Acts the input token slot is packed E2M1
            // (kHidden/2 bytes), so chunk over the actual input byte count and
            // clamp the per-pull chunk to it (kNumBytesPerPull is sized off the
            // logical hidden dim and may exceed the FP4 token footprint).
            constexpr uint32_t kPullTotalBytes = kInputTokenBytes;
            constexpr uint32_t kPullChunkBytes = kNumBytesPerPull < kPullTotalBytes ? kNumBytesPerPull : kPullTotalBytes;
            constexpr uint32_t kNumChunks = kPullTotalBytes / kPullChunkBytes;
            DG_STATIC_ASSERT(kNumChunks * kPullChunkBytes == kPullTotalBytes, "kNumBytesPerPull must divide the input token bytes");

            // TMA load token from remote rank and store into local
            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            const auto src_base_ptr = sym_buffer.map(
                input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(), current_rank_in_expert_idx);
            const auto dst_base_ptr = l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr();
            const auto issue_and_wait_pull_store = [&](const uint32_t& i) {
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                ptx::tma_store_1d(
                    math::advance_ptr(dst_base_ptr, i * kPullChunkBytes),
                    pull_buffer.get_base_ptr(), kPullChunkBytes
                );
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
            };
            if (cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumChunks; ++ i) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        math::advance_ptr(src_base_ptr, i * kPullChunkBytes),
                        pull_mbarrier, kPullChunkBytes
                    );
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kPullChunkBytes);
                    i != (kNumChunks - 1) ? issue_and_wait_pull_store(i) : void();
                }
            }
            __syncwarp();

            // Load and store SF (overlaps with last chunk's TMA load from remote)
            constexpr uint32_t kNumSFUint32 = kHidden / 128;
            DG_STATIC_ASSERT(kNumSFUint32 > 0 and kHidden % 128 == 0, "Invalid SF");
            const auto remote_sf_ptr = sym_buffer.map(
                input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                current_rank_in_expert_idx);
            const auto local_sf_ptr = l1_sf_buffer.get_base_ptr<uint32_t>();
            const auto sf_pool_token_idx = expert_pool_block_offset * SF_BLOCK_M +
                transform_sf_token_idx(token_idx_in_expert);
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFUint32, 32u); ++ i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFUint32)
                    local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];
            }
            __syncwarp();

            // Store weights and metadata
            if (cute::elect_one_sync()) {
                // Load weights
                const auto weight = *sym_buffer.map(
                    input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                    current_rank_in_expert_idx);
                *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;

                // Write source metadata for combine write-back
                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                // Complete last chunk's store
                issue_and_wait_pull_store(kNumChunks - 1);
                ptx::red_add_rel(
                    workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
            }
            __syncwarp();
        }

        // Clean workspace for the next usage, and also do cumulative stats
        // NOTES: it is overlapped with combine reduction epilogue
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            // SM 0: clear expert send count
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
        } else {
            // Other SMs: clean blocks
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                // Read expert token count before clearing
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                // Compute expert pool block offset
                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                // Wait read count ready
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                // Clean expert token count, and add cumulative results
                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                // Clean per-rank token count
                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                // Clean L1 and L2 arrival stuffs
                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + j) = 0;
                    *workspace.get_l2_arrival_mask_ptr(expert_pool_block_offset + j) = 0;
                }
                __syncwarp();
            }
        }

        // Wait for all ranks to finish cleaning
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* Before the NVLink barrier, there is a grid sync */ true,
            /* At the end of kernel does not need to sync */ false
        );
    } else if (warp_idx == kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for tokens with SFA
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_a_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const auto shape_k = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_K : L1_SHAPE_K;
            const auto shape_sfa_k = math::ceil_div(shape_k, kGranK * 4u);

            // Compute pool block offset for this expert
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;

            // Wait the entire token arrival for linear 1
            if (block_phase == sched::BlockPhase::Linear1) {
                const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                const auto expected = scheduler.template get_valid_m<false>();
                while (ptx::ld_acq(ptr) != expected);
            } else {
                // The L1 output's block N is halved into `BLOCK_K / 2`, so we have to wait 2x L1 blocks' arrival
                // NOTES: Originally we wait blocks on-demand to overlap L1 calculation
                // with L2, but this optimization is negative when `num_experts_per_wave`
                // guarantees L1's completion when L2 starts. So we remove it.
                // In the future, if `num_experts_per_wave` is not large enough
                // due to small `num_experts_per_rank`, we may need to add it back or add a switch
                DG_STATIC_ASSERT(BLOCK_K % BLOCK_N == 0, "Invalid block sizes");
                const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);

                constexpr uint32_t kShiftOffset = (L2_SHAPE_K / BLOCK_N) * 2;
                DG_STATIC_ASSERT(kShiftOffset <= 64, "Invalid shift amount");
                constexpr uint64_t kExpectedMask = kShiftOffset == 64 ?
                    static_cast<uint64_t>(-1) : (1ull << kShiftOffset) - 1;
                while (ptx::ld_acq_gpu(ptr) != kExpectedMask);
            }

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute token offset from pool block index
                uint32_t m_idx = pool_block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfa_m_idx = pool_block_idx * SF_BLOCK_M;
                uint32_t sfa_k_idx = k_block_idx * (BLOCK_K / 128);

                // Add 2 CTA offsets for non-leader CTA
                if (not is_leader_cta)
                    m_idx += scheduler.template get_valid_m<true>() / 2;

                // TMA copy tokens and SFA, then arrive at full barrier
                if (cute::elect_one_sync()) {
                    // Stream A0.2/A0.0b/A0.5: under FP4 acts both L1 and L2 load A as
                    // packed E2M1. mxf4 uses a raw 2SM 2D TMA (dense FP4 sub-byte
                    // destination would be mis-strided by tma::copy's element math).
                    if constexpr (kUseMxf4Kind) {
                        cute::SM100_TMA_2SM_LOAD_2D::copy(
                            tensor_map_a_ptr,
                            reinterpret_cast<uint64_t*>(&shared_storage.full_barriers[stage_idx]),
                            static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                            reinterpret_cast<uint8_t*>(shared_storage.smem_a[stage_idx]),
                            k_idx, m_idx);
                    } else if constexpr (kUseFp4Acts) {
                        tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, l1_a_dtype_t>(
                            tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx],
                            reinterpret_cast<l1_a_dtype_t*>(shared_storage.smem_a[stage_idx]),
                            k_idx, m_idx, 2);
                    } else {
                        tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                            tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_a[stage_idx], k_idx, m_idx, 2);
                    }
                    tma::copy<SF_BLOCK_M, 1, 0>(
                        tensor_map_sfa_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx, 2);
                    if (is_leader_cta) {
                        // Source bytes per peer × 2 peers. For FP4 _ALIGN16B (non-mxf4)
                        // the smem is 2× the source, so use the per-stage size ×1.
                        const uint32_t expect_a_bytes = (kUseFp4Acts and not kUseMxf4Kind)
                            ? static_cast<uint32_t>(sizeof(SharedStorage::smem_a[0]))
                            : static_cast<uint32_t>(sizeof(SharedStorage::smem_a[0])) * 2;
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(expect_a_bytes + sizeof(SharedStorage::smem_sfa[0]) * 2);
                    } else {
                        shared_storage.full_barriers[stage_idx].arrive(0u);
                    }
                }
                __syncwarp();
            }
        });
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for weights with SF
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_b_ptr =
                block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights : &tensor_map_l1_weights;
            const auto tensor_map_sfb_ptr =
                block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights_sf : &tensor_map_l1_weights_sf;

            const auto shape_k = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_K : L1_SHAPE_K;
            const auto shape_n = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_N : L1_SHAPE_N;
            const auto shape_sfb_k = math::ceil_div(shape_k, kGranK * 4u);

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute weight offset
                uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                uint32_t sfb_k_idx = local_expert_idx * shape_sfb_k + k_block_idx * (BLOCK_K / 128);

                // TMA copy weights with SF
                if (cute::elect_one_sync()) {
                    // Stream A0.5: mxf4 dense FP4 B uses a raw 2SM 2D TMA (see A side).
                    if constexpr (kUseMxf4Kind) {
                        cute::SM100_TMA_2SM_LOAD_2D::copy(
                            tensor_map_b_ptr,
                            reinterpret_cast<uint64_t*>(&shared_storage.full_barriers[stage_idx]),
                            static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                            reinterpret_cast<uint8_t*>(shared_storage.smem_b[stage_idx]),
                            k_idx, n_idx);
                    } else {
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx], k_idx, n_idx, 2);
                    }
                    tma::copy<BLOCK_N, 1, 0>(
                        tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                    if (is_leader_cta) {
                        // For the FP4 unpacksmem (non-mxf4) path, smem size == source
                        // bytes summed over peers (sizeof(b_dtype_t)==1); under mxf4
                        // dense the smem halves, so ×2 to recover source-bytes-summed.
                        const uint32_t expect_b_bytes = kUseMxf4Kind
                            ? static_cast<uint32_t>(sizeof(SharedStorage::smem_b[0])) * 2
                            : static_cast<uint32_t>(sizeof(SharedStorage::smem_b[0]));
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(expect_b_bytes + sizeof(SharedStorage::smem_sfb[0]) * 2);
                    } else {
                        shared_storage.full_barriers[stage_idx].arrive(0u);
                    }
                }
                __syncwarp();
            }
        });
    } else if (warp_idx == kNumDispatchWarps + 2) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM MMA issue warp (only the leader CTA will run)
        if (is_leader_cta) {
            // Make instruction descriptor with block scaling
            // NOTES: always swap A/B
            auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                UMMA_M, UMMA_N,
                cute::UMMA::Major::K, cute::UMMA::Major::K
            >();
            // Stream A0.2/A0.5: FP4-acts idesc with an E2M1 A-dtype field. mxf4
            // must encode E2M1 as MXF4Format (dense float_e2m1_t) rather than
            // MXF8F6F4Format (unpacksmem) — wrong enum → cudaErrorIllegalInstruction.
            using mxf4_e2m1_t = cute::float_e2m1_t;
            using fp4_a_dtype_for_idesc = cute::conditional_t<kUseMxf4Kind, mxf4_e2m1_t, b_dtype_t>;
            using fp4_b_dtype_for_idesc = cute::conditional_t<kUseMxf4Kind, mxf4_e2m1_t, l1_a_dtype_t>;
            auto instr_desc_fp4 = cute::UMMA::make_instr_desc_block_scaled<
                fp4_a_dtype_for_idesc, fp4_b_dtype_for_idesc,
                float, cutlass::float_ue8m0_t,
                UMMA_M, UMMA_N,
                cute::UMMA::Major::K, cute::UMMA::Major::K
            >();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);

            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            // Stream A0.5: dense FP4 smem → pass byte-width K (UMMA_BLOCK_K/2)
            // and a uint8_t dtype so make_umma_desc's stride math matches.
            cute::UMMA::SmemDescriptor a_desc, b_desc;
            if constexpr (kUseMxf4Kind) {
                a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, UMMA_BLOCK_K / 2, kSwizzleAMode>(
                    reinterpret_cast<uint8_t*>(shared_storage.smem_a[0]), 0, 0);
                b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, UMMA_BLOCK_K / 2, kSwizzleBMode>(
                    reinterpret_cast<uint8_t*>(shared_storage.smem_b[0]), 0, 0);
            } else {
                a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, UMMA_BLOCK_K, kSwizzleAMode>(shared_storage.smem_a[0], 0, 0);
                b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, UMMA_BLOCK_K, kSwizzleBMode>(shared_storage.smem_b[0], 0, 0);
            }
            uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * sizeof(SharedStorage::smem_a[0]) / 16 : 0u;
            uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * sizeof(SharedStorage::smem_b[0]) / 16 : 0u;

            // Checks for MMA instructions
            DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            // Persistently schedule over blocks
            uint32_t current_iter_idx = 0;
            scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                         const uint32_t& local_expert_idx,
                                         const uint32_t& num_k_blocks,
                                         const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                // Dynamic update of UMMA N based on effective M
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, scheduler.template get_valid_m<true>());
                if constexpr (kUseFp4Acts)
                    mma::sm100::update_instr_desc_with_umma_n(instr_desc_fp4, scheduler.template get_valid_m<true>());

                // Wait tensor memory empty barrier arrival
                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[accum_stage_idx].wait(accum_phase ^ 1);
                ptx::tcgen05_after_thread_sync();

                // Empty barrier arrival
                auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                    auto umma_arrive = [](const uint64_t* barrier) {
                        constexpr uint16_t kCTAMask = (1 << 2) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.empty_barriers[stage_idx]));

                    // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };

                // Launch MMAs
                #pragma unroll 2
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait TMA load completion
                    shared_storage.full_barriers[stage_idx].wait(phase);
                    ptx::tcgen05_after_thread_sync();

                    const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                    const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
                    if (cute::elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t umma_k_block_idx = 0; umma_k_block_idx < BLOCK_K / UMMA_BLOCK_K; ++ umma_k_block_idx) {
                            // UTCCP copy SFA and SFB to TMEM
                            using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = shared_storage.smem_sfa[stage_idx] + umma_k_block_idx * SF_BLOCK_M + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = shared_storage.smem_sfb[stage_idx] + umma_k_block_idx * SF_BLOCK_N + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                            }

                            // Issue UMMA
                            #pragma unroll
                            for (uint32_t k = 0; k < UMMA_BLOCK_K / UMMA_K; ++ k) {
                                if constexpr (kUseMxf4Kind) {
                                    // Stream A0.5: kind::mxf4 (UMMA_K=64). SF id top-2 bits are
                                    // HALF-WORD offsets for scale_vec::2X → encode as `k * 2`.
                                    // Dense FP4 smem K-stride is byte/2, so advance on uint8_t
                                    // with k*UMMA_K/2 (and the umma_k_block byte offset halved).
                                    const auto sf_id = k * 2u;
                                    const auto runtime_instr_desc =
                                        mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc_fp4, sf_id, sf_id);
                                    a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                        cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, uint8_t>(
                                            a_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_M / 2, k * UMMA_K / 2);
                                    b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                        cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, uint8_t>(
                                            b_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_N / 2, k * UMMA_K / 2);
                                    ptx::SM100_MMA_MXF4_2x1SM_SS::fma(
                                        b_desc, a_desc, accum_stage_idx * UMMA_N,
                                        k_block_idx > 0 or umma_k_block_idx > 0 or k > 0, runtime_instr_desc,
                                        kTmemStartColOfSFB, kTmemStartColOfSFA);
                                } else {
                                    // Stream A0.0b: under kUseFp4Acts both L1 and L2 read A as
                                    // E2M1 → pick the FP4 idesc; otherwise the FP8 idesc.
                                    const auto runtime_instr_desc = kUseFp4Acts
                                        ? mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc_fp4, k, k)
                                        : mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);
                                    a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                        cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_M * sizeof(a_dtype_t), k * UMMA_K);
                                    b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                        cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_N * sizeof(b_dtype_t), k * UMMA_K);
                                    ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                        b_desc, a_desc, accum_stage_idx * UMMA_N,
                                        k_block_idx > 0 or umma_k_block_idx > 0 or k > 0, runtime_instr_desc,
                                        kTmemStartColOfSFB, kTmemStartColOfSFA);
                                }
                            }
                        }
                    }
                    __syncwarp();

                    // Commit to the mbarrier object
                    // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
                }
            });

            // To safely deconstruct barriers, we need another round of waits
            if (current_iter_idx > 0) {
                const auto accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages].wait(accum_phase_idx);
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 3) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&shared_storage.tmem_ptr_in_smem) == 0);

        // GEMM epilogue warps
        const auto epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const auto epilogue_wg_idx = epilogue_warp_idx / 4;
        const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const auto warp_idx_in_wg = epilogue_warp_idx % 4;
        DG_STATIC_ASSERT((kNumDispatchWarps + kNumMMANonEpilogueWarps) % 4 == 0 and
                         kNumEpilogueWarps % 4 == 0, "Invalid epilogue warps");

        // TODO: support effective block M
        // NOTES:
        //  - 2 warpgroups divide the whole BM into BM / 2
        //  - 4 warps divide the whole BN into BN / 4
        //  - BM / 2 is further divided into stored blocks, i.e. with `STORE_BLOCK_M` size
        //  - `STORE_BLOCK_M` in further divided into `ATOM_M`
        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t ATOM_M = 8;
        constexpr uint32_t kNumBankGroupBytes = 16u;
        constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M / ATOM_M;
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M == 0, "Invalid warpgroup block M");
        DG_STATIC_ASSERT(STORE_BLOCK_M % ATOM_M == 0, "Invalid store block M");
        DG_STATIC_ASSERT(BLOCK_N == 128, "Invalid block N");

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Persistently schedule over blocks
        uint32_t current_iter_idx = 0;
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            // Wait UMMA arrival
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
            shared_storage.tmem_full_barriers[accum_stage_idx].wait(accum_phase);
            ptx::tcgen05_after_thread_sync();

            // Compute offsets
            // NOTES: use shuffle here to let NVCC know warp divergence won't happen
            const uint32_t valid_m = ptx::exchange(scheduler.template get_valid_m<false>(), 0);
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            uint32_t m_idx = pool_block_idx * BLOCK_M;
            uint32_t n_idx = n_block_idx * BLOCK_N;

            if (block_phase == sched::BlockPhase::Linear1) {
                // Unified L1 epilogue: SwiGLU in-place using granularity 8 interleaved weights
                // With `SM100_TMEM_LOAD_16dp256b1x`, gate/up pairs are:
                float stored_cached_weight = 0;

                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // Early break if the entire store block is beyond the valid token range
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        break;
                    }

                    // Iterate all atoms in the store block
                    float2 swiglu_values[kNumAtomsPerStore * 2];
                    float2 amax_values[kNumAtomsPerStore];
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        const uint32_t j = s * kNumAtomsPerStore + i;

                        // Load weights from global into register cache per 32 tokens
                        DG_STATIC_ASSERT(32 % ATOM_M == 0, "Invalid block size");
                        if ((j * ATOM_M) % 32 == 0 and (WG_BLOCK_M % 32 == 0 or j * ATOM_M + lane_idx < WG_BLOCK_M)) {
                            stored_cached_weight = *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx)
                                .get_base_ptr<float>();
                        }

                        // Load weights from register cache
                        const float2 weights = {
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)
                        };

                        // Load from TMEM
                        uint2 raw_values[4];
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M;
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               raw_values[0].x, raw_values[0].y, raw_values[1].x, raw_values[1].y);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               raw_values[2].x, raw_values[2].y, raw_values[3].x, raw_values[3].y);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Signal tensor memory consumed on the last atom
                        if (j == WG_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        }

                        auto fp32_values = reinterpret_cast<float2*>(raw_values);
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            auto bf16_gate = __float22bfloat162_rn(fp32_values[k * 2 + 0]);
                            auto bf16_up =   __float22bfloat162_rn(fp32_values[k * 2 + 1]);

                            // Clamp
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                            }

                            // SwiGLU
                            auto gate = __bfloat1622float2(bf16_gate);
                            auto neg_gate_exp = make_float2(
                                kFastMath ? __expf(-gate.x) : expf(-gate.x),
                                kFastMath ? __expf(-gate.y) : expf(-gate.y));
                            const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                            if constexpr (kFastMath) {
                                gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                            } else {
                                gate = {gate.x / denom.x, gate.y / denom.y};
                            }
                            const auto up = __bfloat1622float2(bf16_up);
                            swiglu_values[i * 2 + k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
                        }

                        // Amax reduction
                        amax_values[i].x = math::warp_reduce<4, true>(
                            cute::max(cute::abs(swiglu_values[i * 2 + 0].x), cute::abs(swiglu_values[i * 2 + 1].x)),
                            math::ReduceMax<float>());
                        amax_values[i].y = math::warp_reduce<4, true>(
                            cute::max(cute::abs(swiglu_values[i * 2 + 0].y), cute::abs(swiglu_values[i * 2 + 1].y)),
                            math::ReduceMax<float>());
                        if (lane_idx < 4)
                            shared_storage.amax_reduction[epilogue_warp_idx][i * (ATOM_M / 2) + lane_idx] = amax_values[i];
                        __syncwarp();
                    }

                    // Wait shared memory release from previous TMA store
                    // And fence `shared_storage.amax_reduction`
                    const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                    ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Cast to FP8 E4M3 and store into shared memory
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        // Reduce amax
                        const float2 wp_amax =
                            shared_storage.amax_reduction[epilogue_warp_idx ^ 1][i * (ATOM_M / 2) + lane_idx % 4];
                        amax_values[i].x = cute::max(amax_values[i].x, wp_amax.x);
                        amax_values[i].y = cute::max(amax_values[i].y, wp_amax.y);

                        // Calculate SF (UE8M0 byte; finfo divisor differs: 1/448 for
                        // FP8 E4M3, 1/6 for FP4 E2M1).
                        float2 sf, sf_inv;
                        if constexpr (kUseFp4Acts) {
                            math::get_e2m1_sf_and_sf_inv(amax_values[i], sf, sf_inv);
                        } else {
                            math::get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv);
                        }

                        // Apply scale, cast, store into shared memory.
                        const float2 upper = __fmul2_rn(swiglu_values[i * 2 + 0], sf_inv);
                        const float2 lower = __fmul2_rn(swiglu_values[i * 2 + 1], sf_inv);
                        if constexpr (kUseFp4Acts) {
                            // FP4 epilogue: pack E2M1 nibbles into a dense canonical smem
                            // layout (TMA descriptor built with swizzle=0). Under SwapAB the
                            // tcgen05.ld lane→position map places adjacent N-cols on lanes
                            // T and T^4, so packing two values per byte needs a `shfl_xor 4`
                            // buddy; the half-warp gate (group=lane/4, group%2==0) makes each
                            // active lane write 4 bytes while its buddy half donates.
                            const float buddy_ux = __shfl_xor_sync(0xffffffffu, upper.x, 4);
                            const float buddy_uy = __shfl_xor_sync(0xffffffffu, upper.y, 4);
                            const float buddy_lx = __shfl_xor_sync(0xffffffffu, lower.x, 4);
                            const float buddy_ly = __shfl_xor_sync(0xffffffffu, lower.y, 4);

                            const uint32_t frag  = lane_idx % 4;          // row-pair index 0..3
                            const uint32_t group = lane_idx / 4;          // col-group index 0..7
                            const bool     is_active = (group % 2u) == 0u;

                            if (is_active) {
                                const uint8_t byte_ux = static_cast<uint8_t>(
                                    math::cvt_pack_f32_to_e2m1x2(upper.x, buddy_ux));
                                const uint8_t byte_uy = static_cast<uint8_t>(
                                    math::cvt_pack_f32_to_e2m1x2(upper.y, buddy_uy));
                                const uint8_t byte_lx = static_cast<uint8_t>(
                                    math::cvt_pack_f32_to_e2m1x2(lower.x, buddy_lx));
                                const uint8_t byte_ly = static_cast<uint8_t>(
                                    math::cvt_pack_f32_to_e2m1x2(lower.y, buddy_ly));

                                constexpr uint32_t kFp4WarpStripeBytes = 8;  // 16 elements / 2
                                const uint32_t byte_pos_upper = group / 2u;             // 0..3
                                const uint32_t byte_pos_lower = 4u + group / 2u;        // 4..7
                                const uint32_t row_even = i * ATOM_M + 2u * frag;
                                const uint32_t row_odd  = row_even + 1u;
                                // wg offset is folded into the `[epilogue_wg_idx]` index.
                                const auto base = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx])
                                                + warp_idx_in_wg * kFp4WarpStripeBytes;
                                auto write_byte = [&](uint32_t row, uint32_t bp, uint8_t v) {
                                    auto p = base + row * L1_OUT_ROW_BYTES + bp;
                                    asm volatile("st.shared.u8 [%0], %1;\n"
                                                 :: "l"(__cvta_generic_to_shared(p)),
                                                    "r"(static_cast<uint32_t>(v)));
                                };
                                write_byte(row_even, byte_pos_upper, byte_ux);
                                write_byte(row_odd,  byte_pos_upper, byte_uy);
                                write_byte(row_even, byte_pos_lower, byte_lx);
                                write_byte(row_odd,  byte_pos_lower, byte_ly);
                            }
                        } else {
                            const auto fp8x4_values = __nv_fp8x4_e4m3(make_float4(upper.x, upper.y, lower.x, lower.y));

                            // STSM
                            uint32_t row = lane_idx;
                            uint32_t col = warp_idx_in_wg;
                            const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx])
                                + i * ATOM_M * L1_OUT_BLOCK_N
                                + row * L1_OUT_BLOCK_N
                                // Use 64B swizzle for SwiGLU, so divided by 2
                                + (col ^ (row / 2)) * kNumBankGroupBytes;
                            ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);
                        }

                        // Store SF to `l2_sf_buffer` as UE8M0 (MN-major layout)
                        // Only one warp per pair writes (both hold the same SF after cross-warp reduce)
                        // Each lane < 4 holds SF for 2 rows (sf.x and sf.y)
                        if (warp_idx_in_wg % 2 == 0 and lane_idx < 4) {
                            const uint32_t k_idx = n_block_idx * (BLOCK_N / kGranK / 2) + warp_idx_in_wg / 2;
                            const uint32_t k_uint_idx = k_idx / 4, byte_idx = k_idx % 4;
                            const uint32_t mn_stride = kNumPaddedSFPoolTokens * sizeof(uint32_t);
                            const auto sf_base_ptr = l2_sf_buffer.get_base_ptr<uint8_t>();
                            // NOTES: consecutive tokens (t, t + 1) are in the same 32-group, so `sf_idx` differs by 4
                            // NOTES: originally there was:
                            //   - `const uint32_t token_idx_in_expert = m_block_idx * BLOCK_M + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M + lane_idx * 2
                            //   - `scheduler.get_current_pool_block_offset() * SF_BLOCK_M + transform_sf_token_idx(token_idx_in_expert)`
                            // We find out that
                            //   1. `m_block_idx * BLOCK_M` mod `BLOCK_M` is 0, and `epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M + lane_idx * 2` is always < `BLOCK_M`, so we can put `m_block_idx * BLOCK_M` outside
                            //   2. `lane_idx * 2` controls the lowest 3 bit of `token_idx_in_expert`, and `transform_sf_token_idx` is a bitwise-independent transformation if the input is less than `BLOCK_M`, so we can put `lane_idx * 2` outside
                            // This reduce the number of computation instructions.
                            const uint32_t token_base_idx = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                            __builtin_assume(token_base_idx < BLOCK_M);
                            const auto sf_pool_token_idx = scheduler.get_current_pool_block_offset() * SF_BLOCK_M
                                + m_block_idx * SF_BLOCK_M + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;
                            const auto sf_addr = k_uint_idx * mn_stride + sf_pool_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx;
                            sf_base_ptr[sf_addr] =
                                (*reinterpret_cast<const uint32_t*>(&sf.x) >> 23);
                            sf_base_ptr[sf_addr + 4 * static_cast<uint32_t>(sizeof(uint32_t))] =
                                (*reinterpret_cast<const uint32_t*>(&sf.y) >> 23);
                        }
                        __syncwarp();
                    }
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Issue TMA store after all atoms in this store block
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        // FP4 path: descriptor element type is uint8 with half the inner
                        // dim → out_n in packed bytes (L1_OUT_BLOCK_N / 2).
                        const uint32_t out_n_idx = kUseFp4Acts
                            ? (n_block_idx * (L1_OUT_BLOCK_N / 2))
                            : (n_block_idx * L1_OUT_BLOCK_N);
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx],
                            out_n_idx,
                            m_idx + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }

                // Notify L2
                // TODO: less epilogue sync scope
                ptx::tma_store_wait<0>();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    DG_STATIC_ASSERT(L2_SHAPE_K <= 64 * L1_OUT_BLOCK_N, "L2 shape K is too large");
                    ptx::red_or_rel_gpu(
                        workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                        1ull << n_block_idx
                    );
                }
                __syncwarp();
            } else {
                DG_STATIC_ASSERT(STORE_BLOCK_M % 8 == 0, "Invalid store M");
                constexpr uint32_t kNumRowsPerWarp = STORE_BLOCK_M / 8;

                // L2 BF16 epilogue: write GEMM output to remote combine buffer via NVLink
                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // Early break if the entire store block is beyond the valid token range
                    // TODO: check performance
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        break;
                    }

                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_M / ATOM_M; ++ i) {
                        // Load from TMEM using .16x256b shape to satisfy STSM layout requirements
                        // Start from lane index 0 and 16
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                        uint32_t values[ATOM_M];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Wait shared memory release from previous NVLink store
                        // NOTES: skip for the first store block since the prior full barrier already ensures completion
                        if (i == 0 and s > 0)
                            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        // Signal tensor memory consumed
                        if (s == WG_BLOCK_M / STORE_BLOCK_M - 1 and i == STORE_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        }

                        // Store into shared memory
                        // NOTES: each lane provides its own address for stmatrix; 2 warps share a BF16 swizzle atom
                        uint32_t row = lane_idx % 8;
                        uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                            (warp_idx_in_wg / 2) * STORE_BLOCK_M * kSwizzleCDMode +
                            i * ATOM_M * kSwizzleCDMode +
                            row * (kNumBankGroupBytes * 8) +
                            (col ^ row) * kNumBankGroupBytes;
                        ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                            math::cast_into_bf16_and_pack(values[0], values[1]),
                            math::cast_into_bf16_and_pack(values[2], values[3]),
                            math::cast_into_bf16_and_pack(values[4], values[5]),
                            math::cast_into_bf16_and_pack(values[6], values[7]),
                            smem_ptr
                        );
                    }

                    // Wait shared memory ready
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Write into remote buffers
                    // Each warp writes 2 rows (lane_idx/16 splits the warp into two halves, one per row)
                    const uint32_t row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M;
                    const uint32_t bank_group_idx = lane_idx % 8;

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + row_in_store;

                        // Skip padding rows beyond the actual token count for this expert
                        if (m_idx_in_block >= valid_m)
                            break;

                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx;
                        const uint32_t dst_token_idx = src_metadata.token_idx;
                        const uint32_t dst_topk_idx = src_metadata.topk_idx;

                        // Read from shared memory
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                            (lane_idx % 16 / 8) * STORE_BLOCK_M * kSwizzleCDMode +
                            row_in_store * kSwizzleCDMode +
                            (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;
                        const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr));

                        if constexpr (kUseFp8Combine) {
                            // Stream B: BF16 (in `packed`) → FP8 E4M3 + per-row UE8M0 SF.
                            // 16 lanes (lane_idx & ~15u) cover one row's BLOCK_N=128
                            // elements (= 8 BF16 each). Per-row amax via a 16-lane reduce,
                            // then quantize.
                            const auto bf_pairs = reinterpret_cast<const nv_bfloat162*>(&packed);
                            float local_amax = 0.0f;
                            #pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                const float2 vf = __bfloat1622float2(bf_pairs[q]);
                                local_amax = cute::max(local_amax, cute::abs(vf.x));
                                local_amax = cute::max(local_amax, cute::abs(vf.y));
                            }
                            // Reduce within the 16-lane group sharing this row. Use a
                            // 16-lane mask (NOT 0xffffffff): the other half-warp may have
                            // exited early on padding rows, so a full-warp shfl would hang.
                            const uint32_t row_mask = 0x0000FFFFu << (16u * (lane_idx / 16));
                            local_amax = cute::max(local_amax, __shfl_xor_sync(row_mask, local_amax, 1));
                            local_amax = cute::max(local_amax, __shfl_xor_sync(row_mask, local_amax, 2));
                            local_amax = cute::max(local_amax, __shfl_xor_sync(row_mask, local_amax, 4));
                            local_amax = cute::max(local_amax, __shfl_xor_sync(row_mask, local_amax, 8));

                            // UE8M0 SF (E4M3, finfo_max = 448).
                            const int log2_ceil = math::fast_log2_ceil(local_amax * (1.0f / 448.0f));
                            const float sf_inv = math::fast_pow2(-log2_ceil);
                            const uint8_t sf_byte = static_cast<uint8_t>(log2_ceil + 127);

                            // Scale, cast 4 BF16 pairs → 8 FP8 (= 2 fp8x4 = uint64).
                            float4 lo, hi;
                            const auto lo_pair = __bfloat1622float2(bf_pairs[0]);
                            const auto lo_pair_b = __bfloat1622float2(bf_pairs[1]);
                            const auto hi_pair = __bfloat1622float2(bf_pairs[2]);
                            const auto hi_pair_b = __bfloat1622float2(bf_pairs[3]);
                            lo.x = lo_pair.x * sf_inv;
                            lo.y = lo_pair.y * sf_inv;
                            lo.z = lo_pair_b.x * sf_inv;
                            lo.w = lo_pair_b.y * sf_inv;
                            hi.x = hi_pair.x * sf_inv;
                            hi.y = hi_pair.y * sf_inv;
                            hi.z = hi_pair_b.x * sf_inv;
                            hi.w = hi_pair_b.y * sf_inv;
                            const __nv_fp8x4_e4m3 fp8_lo(lo);
                            const __nv_fp8x4_e4m3 fp8_hi(hi);
                            const uint64_t fp8_uint64 =
                                (uint64_t(fp8_lo.__x)) |
                                (uint64_t(fp8_hi.__x) << 32);

                            // Write 8 FP8 bytes (uint64) to remote, replacing the BF16 16-byte write.
                            const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                                       .get_data_buffer(dst_token_idx);
                            const auto dst_ptr = math::advance_ptr<uint64_t>(
                                dst_token.get_base_ptr(),
                                n_idx * static_cast<uint32_t>(sizeof(uint8_t)) +
                                (lane_idx % 16) * static_cast<uint32_t>(sizeof(uint64_t)));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = fp8_uint64;

                            // 1 SF byte per row tile, written by lane 0 of the 16-lane group.
                            if ((lane_idx & 15u) == 0) {
                                const auto sf_token = combine_sf_buffer.get_rank_buffer(dst_topk_idx)
                                                                         .get_data_buffer(dst_token_idx);
                                const auto sf_ptr = math::advance_ptr<uint8_t>(
                                    sf_token.get_base_ptr(),
                                    n_block_idx * static_cast<uint32_t>(sizeof(uint8_t)));
                                *sym_buffer.map(sf_ptr, dst_rank_idx) = sf_byte;
                            }
                        } else {
                            // Default BF16 path (16 bytes/lane = 8 BF16).
                            const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                                       .get_data_buffer(dst_token_idx);
                            const auto dst_ptr = math::advance_ptr<float4>(
                                dst_token.get_base_ptr(),
                                n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) + (lane_idx % 16) * static_cast<uint32_t>(sizeof(float4)));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        }
                    }
                }

                // Ensure the next epilogue safe to use shared memory
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
        });

        // Deallocate tensor memory
        // NOTES: must be called by the same logical warp ID on both CTAs
        if (epilogue_warp_idx == 0)
            Allocator().free(0, kNumTmemCols);

        // NVLink barrier (grid sync + cross-rank signal + grid sync): ~4 us
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                             kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
            workspace, sym_buffer, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );

        // Barrier with dispatch warps, so that they can do clean workspace
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Combine: reduce top-k results and write back
        // NOTES: reuse shared memory from start up to the barriers
        // 1 token, 1 topk latency: ~3 us
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        // 3 slots of chunk is needed: 2 load stages and 1 store
        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;

        // NOTES: either 1 or 2 chunks for simplicity
        // NOTES: Restrict on both smem and register
        constexpr uint32_t kNumChunks =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= kNumReusableSmemBytes and kHidden <= 32 * kNumMaxRegistersForBuffer ? 1 : 2;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= kNumReusableSmemBytes, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements (one per lane)");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        // Verify combined shared memory budget at runtime
        DG_DEVICE_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= kNumReusableSmemBytes);

        // Per-warp buffer: 2 stage load buffers + 1 store buffer
        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer  = math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        // Per-warp barriers
        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return &shared_storage.combine_barriers[i + epilogue_warp_idx * 2];
        });

        // Iterate over all tokens
        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumEpilogueWarps) {
            // Read top-k slot indices: each lane reads one slot, then broadcast via exchange
            DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            // Stream B: FP8 path loads kNumChunkBytes / 2 per slot (FP8 = 1 byte/elem)
            // and reads a per-(slot, token, n_block) UE8M0 SF byte to dequant on the
            // fly. Output is BF16 either way → store byte count is kNumChunkBytes.
            constexpr uint32_t kNumLoadBytesPerChunk =
                kUseFp8Combine ? (kNumChunkBytes / 2) : kNumChunkBytes;
            constexpr uint32_t kNumLoadUint4PerLane =
                kUseFp8Combine ? (kNumUint4PerLane / 2) : kNumUint4PerLane;
            // Per-uint4 load: BF16 → 8 BF16 = 4 float2; FP8 → 16 FP8 = 8 float2 (dequant'd).
            constexpr uint32_t kNumF32PairsPerLoadUint4 =
                kUseFp8Combine ? 8u : 4u;
            // Per-element offset in the chunk for SF lookup.
            constexpr uint32_t kNumLoadElemsPerChunk = kHidden / kNumChunks;

            // Iterate all chunks
            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumLoadBytesPerChunk;

                // Move mask and load
                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) -> int {
                    if (mask) {
                        // Move
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;

                        // Load FP8 / BF16 chunk
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumLoadBytesPerChunk);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumLoadBytesPerChunk);
                        }
                        __syncwarp();
                        return static_cast<int>(slot_idx);
                    }
                    return -1;
                };

                // Load the first selection
                int active_slot = move_mask_and_load(load_stage_idx);

                // Accumulate all top-k contributions for this chunk in float registers
                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (active_slot >= 0) {
                    // Prefetch next top-k into the buffer while current is being accumulated.
                    int next_slot = move_mask_and_load(load_stage_idx ^ 1);

                    // Wait for current slot's load.
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);

                    if constexpr (kUseFp8Combine) {
                        // Per-slot SF base for this token.
                        const uint8_t* sf_token_ptr =
                            combine_sf_buffer.get_rank_buffer(static_cast<uint32_t>(active_slot))
                                              .get_data_buffer(token_idx).get_base_ptr<uint8_t>();
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumLoadUint4PerLane; ++ j) {
                            const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                            // 16 elements at this offset; since 16 < 128 they share one SF byte.
                            const uint32_t sf_idx =
                                (chunk * kNumLoadElemsPerChunk + (j * 32 + lane_idx) * 16) / kCombineGranK;
                            const uint8_t sf_byte = __ldg(sf_token_ptr + sf_idx);
                            const float sf = math::fast_pow2(static_cast<int>(sf_byte) - 127);
                            const uint32_t* w = reinterpret_cast<const uint32_t*>(&uint4_values);
                            #pragma unroll
                            for (uint32_t l = 0; l < 4; ++ l) {
                                // Each uint32 = 4 FP8 = 2 FP8x2 — convert via FP16 intermediate.
                                uint32_t f16_lo, f16_hi;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
                                             : "=r"(f16_lo) : "h"(uint16_t(w[l] & 0xFFFFu)));
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
                                             : "=r"(f16_hi) : "h"(uint16_t(w[l] >> 16)));
                                float vlx, vly, vhx, vhy;
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(vlx) : "h"(uint16_t(f16_lo & 0xFFFFu)));
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(vly) : "h"(uint16_t(f16_lo >> 16)));
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(vhx) : "h"(uint16_t(f16_hi & 0xFFFFu)));
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(vhy) : "h"(uint16_t(f16_hi >> 16)));
                                auto& acc_lo = reduced[j * kNumF32PairsPerLoadUint4 + l * 2 + 0];
                                auto& acc_hi = reduced[j * kNumF32PairsPerLoadUint4 + l * 2 + 1];
                                acc_lo.x = __fmaf_rn(vlx, sf, acc_lo.x);
                                acc_lo.y = __fmaf_rn(vly, sf, acc_lo.y);
                                acc_hi.x = __fmaf_rn(vhx, sf, acc_hi.x);
                                acc_hi.y = __fmaf_rn(vhy, sf, acc_hi.y);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                            const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                            const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                                ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                        }
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                    active_slot = next_slot;
                }

                // Cast & write to smem store-buffer.
                // BF16: kNumUint4PerLane stores. FP8: kNumLoadUint4PerLane * 2 stores.
                if constexpr (kUseFp8Combine) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumLoadUint4PerLane; ++ j) {
                        uint4 lo, hi;
                        auto lo_bf = reinterpret_cast<nv_bfloat162*>(&lo);
                        auto hi_bf = reinterpret_cast<nv_bfloat162*>(&hi);
                        #pragma unroll
                        for (uint32_t l = 0; l < 4; ++ l) {
                            lo_bf[l] = __float22bfloat162_rn(reduced[j * kNumF32PairsPerLoadUint4 + l]);
                            hi_bf[l] = __float22bfloat162_rn(reduced[j * kNumF32PairsPerLoadUint4 + 4 + l]);
                        }
                        if (j == 0) {
                            ptx::tma_store_wait<0>();
                            __syncwarp();
                        }
                        ptx::st_shared(combine_store_buffer + (j * 32 + lane_idx) * 2 + 0,
                                       lo.x, lo.y, lo.z, lo.w);
                        ptx::st_shared(combine_store_buffer + (j * 32 + lane_idx) * 2 + 1,
                                       hi.x, hi.y, hi.z, hi.w);
                    }
                } else {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        uint4 casted;
                        auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);
                        if (j == 0) {
                            ptx::tma_store_wait<0>();
                            __syncwarp();
                        }
                        ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                       casted.x, casted.y, casted.z, casted.w);
                    }
                }
                __syncwarp();

                // TMA store the BF16 chunk to gmem y (output byte offset tracks BF16 chunks).
                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk * kNumChunkBytes),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm
