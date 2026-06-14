import argparse
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Tuple

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto


def import_baseline():
    # Load legacy implements from third-party
    deep_ep, tilelang_ops, do_bench, is_legacy_loaded = None, None, None, False
    # noinspection PyBroadException
    try:
        import deep_ep
        import importlib.util
        from tilelang.profiler.bench import do_bench
        spec = importlib.util.spec_from_file_location(
            'tilelang_ops',
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'third-party', 'tilelang_ops', '__init__.py'))
        tilelang_ops = importlib.util.module_from_spec(spec)
        sys.modules['tilelang_ops'] = tilelang_ops
        spec.loader.exec_module(tilelang_ops)
        is_legacy_loaded = True
    except Exception as ex:
        dist_print(f'Failed to load legacy code: {ex}, skip baseline benchmarking', once_in_node=True)
        dist_print(once_in_node=True)
    return deep_ep, tilelang_ops, do_bench, is_legacy_loaded


# TODO: skip the test for SM90
# noinspection PyUnboundLocalVariable,PyShadowingNames
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    # A1.b: pin each rank to its GPU's NUMA-local CPU node so that bench timing
    # isn't perturbed by OS scheduler shuffling python processes across nodes.
    # `nvidia-smi topo -m` mapping on this box (CUDA_VISIBLE_DEVICES=4,5,6,7):
    #   local_rank 0,1 → GPU 4,5 → NUMA 3 → CPUs 72-95, 216-239
    #   local_rank 2,3 → GPU 6,7 → NUMA 5 → CPUs 120-143, 264-287
    numa_cpus = {
        0: list(range(72, 96)) + list(range(216, 240)),
        1: list(range(72, 96)) + list(range(216, 240)),
        2: list(range(120, 144)) + list(range(264, 288)),
        3: list(range(120, 144)) + list(range(264, 288)),
    }
    if local_rank in numa_cpus:
        try:
            os.sched_setaffinity(0, numa_cpus[local_rank])
        except Exception:
            pass

    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    # Settings
    # A1: bench-sweep mode -- size buffer for the MAX shape and reuse it across
    # all per-shape inner loops. Avoids NCCL re-init and symm-buffer realloc.
    sweep_shapes = []
    if args.shape_ntoks:
        sweep_shapes = [int(s) for s in args.shape_ntoks.split(',') if s.strip()]
        num_max_tokens_per_rank = max(sweep_shapes)
        num_tokens = sweep_shapes[0]  # placeholder; reassigned per shape below
    else:
        num_max_tokens_per_rank = args.num_max_tokens_per_rank
        num_tokens = max(0, args.num_max_tokens_per_rank - random.randint(0, args.num_max_removed_tokens)) \
            if args.num_tokens == 0 else args.num_tokens
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank

    # Allocate symmetric memory
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden
    )

    # Create inputs
    # noinspection PyGlobalUndefined
    def create_inputs():
        global x, topk_idx, topk_weights, l1_weights, l2_weights, transformed_l1_weights, transformed_l2_weights
        global cumulative_local_expert_recv_stats_fused
        global cumulative_local_expert_recv_stats_baseline
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        l1_weights = torch.randn(
            (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
        l2_weights = torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        cumulative_local_expert_recv_stats_fused = torch.randint(
            0, 100, (num_experts_per_rank, ), dtype=torch.int, device='cuda')
        cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_fused.clone()
        if args.masked_ratio > 0:
            rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
            topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
            topk_weights.masked_fill_(topk_idx < 0, 0)

        # Check SF requirements
        assert hidden % 128 == 0
        assert intermediate_hidden % 128 == 0
        assert l1_weights.shape[2] % 128 == 0 and l2_weights.shape[2] % 128 == 0

        # Cast inputs to FP8 (or FP4 under DG_USE_FP4_ACTS) with per-32 UE8M0 SF
        if os.environ.get('DG_USE_FP4_ACTS', '0') != '0' or os.environ.get('DG_MEGA_MOE_FP4', '0') != '0':
            x = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        else:
            x = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)

        # Cast grouped BF16 weights to FP4 with MN-major SF
        # TODO: merge with `cast_fp8_fp4_with_major`
        def cast_grouped_weights_to_fp4(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            num_groups, n, k = bf16_weights.shape
            w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
            w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
            for i in range(num_groups):
                w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
            w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
            return w, w_sf

        l1_weights = cast_grouped_weights_to_fp4(l1_weights)
        l2_weights = cast_grouped_weights_to_fp4(l2_weights)
        transformed_l1_weights, transformed_l2_weights = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

    # Get torch views of buffer (sgl-deep-gemm wraps tensors with tvm_ffi; convert via dlpack)
    def _as_torch(t):
        if isinstance(t, torch.Tensor):
            return t
        return torch.from_dlpack(t)
    _buf_x = _as_torch(buffer.x)
    _buf_x_sf = _as_torch(buffer.x_sf)
    _buf_topk_idx = _as_torch(buffer.topk_idx)
    _buf_topk_weights = _as_torch(buffer.topk_weights)

    # Run fused mega MoE
    # NOTES: copy x into buffer before each call because debug mode zeros the entire buffer
    def run_fused():
        # x[0] is per_token cast packed bytes. For FP4 it has hidden/2 bytes; buffer.x is sized for FP8 (hidden bytes).
        nbytes_x = x[0].shape[1]
        _buf_x[:num_tokens, :nbytes_x].copy_(x[0])
        nbytes_sf = x[1].shape[1]
        _buf_x_sf[:num_tokens, :nbytes_sf].copy_(x[1])
        _buf_topk_idx[:num_tokens].copy_(topk_idx)
        _buf_topk_weights[:num_tokens].copy_(topk_weights)

        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        # noinspection PyTypeChecker
        deep_gemm.fp8_fp4_mega_moe(
            y,
            transformed_l1_weights, transformed_l2_weights,
            buffer,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_fused,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math)
        )
        return y, cumulative_local_expert_recv_stats_fused

    dist_print('Config:', once_in_node=True)
    dist_print(f' > Tokens: {num_tokens}/{num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    dist_print(f' > Buffer: {buffer.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(once_in_node=True)

    # A1: Multi-shape bench sweep mode. Allocates buffer once for the max shape,
    # then iterates all shapes & reps in this single process. Pre-warms heavily
    # to ramp clocks and JIT-compile each kernel before timing it.
    if sweep_shapes:
        dist_print(f'Bench sweep: shapes={sweep_shapes}, reps_per_shape={args.bench_reps}', once_in_node=True)

        # ---- Phase 1: heavy GPU burnin to ramp clocks (no measurement) ----
        # B200 boost takes a few seconds of sustained heavy work to settle.
        burnin_size = 8192
        burnin_a = torch.randn(burnin_size, burnin_size, device='cuda', dtype=torch.bfloat16)
        burnin_b = torch.randn(burnin_size, burnin_size, device='cuda', dtype=torch.bfloat16)
        for _ in range(80):
            burnin_a @ burnin_b
        torch.cuda.synchronize()
        del burnin_a, burnin_b
        dist.barrier()
        dist_print(' > Burnin complete', once_in_node=True)

        # ---- Phase 2: per-shape JIT warmup ----
        # First call of each shape triggers JIT compile; second-onwards are cached.
        for ntok in sweep_shapes:
            num_tokens = ntok  # closure variable used by create_inputs / run_fused
            create_inputs()
            for _ in range(2):
                run_fused()
        torch.cuda.synchronize()
        dist.barrier()
        dist_print(' > JIT warmup complete', once_in_node=True)

        # ---- Phase 3: per-shape measurement ----
        safe_div = lambda a, b: float('nan') if b == 0 else a / b
        use_fp4 = os.environ.get('DG_USE_FP4_ACTS', '0') != '0' or os.environ.get('DG_MEGA_MOE_FP4', '0') != '0'
        act_bytes = 0.5 if use_fp4 else 1.0

        for ntok in sweep_shapes:
            num_tokens = ntok
            # Per-shape ramp-up across reps: do one extra throwaway rep first to
            # absorb clock ramp-down from waiting at the previous shape's barrier.
            extra_reps = 1
            # In-process interleaved wave A/B: DG_WAVE_AB="0,64" alternates the
            # DG_NUM_EXPERTS_PER_WAVE override per rep (0 = heuristic default).
            wave_ab = [int(x) for x in os.environ.get('DG_WAVE_AB', '').split(',') if x.strip()]
            # In-process interleaved BLOCK_K A/B: DG_BLOCKK_AB="128,256" alternates
            # the DG_BLOCK_K override per rep. Value 0 means "unset env" -> the
            # wrapper's per-band BLOCK_K default (R2's primary A/B is 128,0:
            # all-128 vs band-default). Same mechanism as DG_WAVE_AB; result lines
            # gain a `bk=N` tag (bk=- for the unset/band-default rep).
            blockk_ab = [int(x) for x in os.environ.get('DG_BLOCKK_AB', '').split(',') if x.strip()]
            # In-process interleaved kNumStages A/B: DG_STAGES_AB="0,4" alternates
            # the DG_NUM_STAGES clamp per rep (0 = unset -> auto stage count).
            # result lines gain an `st=N` tag.
            stages_ab = [int(x) for x in os.environ.get('DG_STAGES_AB', '').split(',') if x.strip()]
            for rep in range(args.bench_reps + extra_reps):
                cur_wave = None
                if wave_ab:
                    cur_wave = wave_ab[rep % len(wave_ab)]
                    if cur_wave > 0:
                        os.environ['DG_NUM_EXPERTS_PER_WAVE'] = str(cur_wave)
                    else:
                        os.environ.pop('DG_NUM_EXPERTS_PER_WAVE', None)
                cur_blockk = None
                if blockk_ab:
                    cur_blockk = blockk_ab[rep % len(blockk_ab)]
                    if cur_blockk > 0:
                        os.environ['DG_BLOCK_K'] = str(cur_blockk)
                    else:
                        os.environ.pop('DG_BLOCK_K', None)
                cur_stages = None
                if stages_ab:
                    cur_stages = stages_ab[rep % len(stages_ab)]
                    if cur_stages > 0:
                        os.environ['DG_NUM_STAGES'] = str(cur_stages)
                    else:
                        os.environ.pop('DG_NUM_STAGES', None)
                # Refresh inputs per rep so routing changes
                create_inputs()
                # Per-shape ramp-up: 60 untimed calls to settle clocks at this workload
                for _ in range(60):
                    run_fused()
                torch.cuda.synchronize()
                dist.barrier()

                # Timed phase
                t_fused = bench_kineto(
                    run_fused, 'mega_moe',
                    num_tests=80,
                    barrier=lambda: dist.barrier(),
                    trace_path=None)

                # Skip the throwaway rep
                if rep < extra_reps:
                    continue
                rep_idx = rep - extra_reps

                # Compute metrics
                gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
                gtk_local = gathered_topk_idx.clone()
                gtk_local[(gtk_local < rank_idx * num_experts_per_rank) | \
                          (gtk_local >= (rank_idx + 1) * num_experts_per_rank)] = -1
                num_recv_tokens = (gtk_local != -1).sum().item()
                num_touched_experts = torch.unique(gtk_local.flatten()).numel() - 1

                tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_fused)
                num_hbm_bytes = (
                    num_touched_experts * intermediate_hidden * 2 * hidden // 2 +
                    num_touched_experts * hidden * intermediate_hidden // 2 +
                    int(num_recv_tokens * hidden * act_bytes) +
                    int(num_recv_tokens * intermediate_hidden * act_bytes) +
                    int(num_recv_tokens * intermediate_hidden * act_bytes) +
                    num_recv_tokens * hidden * 2
                )
                hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_fused)
                ai = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3), num_hbm_bytes)

                for r in range(num_ranks):
                    if r == rank_idx:
                        print(f'[ntok={ntok:>4} rep={rep_idx:>2} rank={rank_idx} wave={cur_wave if cur_wave is not None else "-"} '
                              f'bk={cur_blockk if cur_blockk is not None else "-"} '
                              f'st={cur_stages if cur_stages is not None else "-"}] '
                              f'{tflops:>6.1f} TFLOPS | HBM {hbm_gbs:>6.0f} GB/s | '
                              f'AI {ai:>6.1f} | {t_fused * 1e6:>6.1f} us | '
                              f'tokens={num_recv_tokens} experts={num_touched_experts}',
                              flush=True)
                    dist.barrier()

        # Cleanup
        dist.barrier()
        buffer.destroy()
        dist.destroy_process_group()
        return

    # Only do NCU profiling
    if args.ncu_profile_only:
        create_inputs()
        dist_print(f'Run fused kernel:', once_in_node=True)

        # Warmup
        for _ in range(5):
            run_fused()
        torch.cuda.synchronize()
        dist.barrier()

        # Benchmark via kineto (records per-call kernel time)
        t_fused = bench_kineto(
            run_fused, 'mega_moe',
            barrier=lambda: dist.barrier(),
            trace_path=None)

        # Compute "active tokens" from topk_idx (local rank's received tokens proxy)
        # For ncu standalone test we use num_recv_tokens = num_tokens * num_topk / num_ranks
        # (assuming even routing - matches test's analytical formula structure)
        gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
        gathered_topk_idx_local = gathered_topk_idx.clone()
        gathered_topk_idx_local[(gathered_topk_idx_local < rank_idx * num_experts_per_rank) | \
                                (gathered_topk_idx_local >= (rank_idx + 1) * num_experts_per_rank)] = -1
        num_recv_tokens = (gathered_topk_idx_local != -1).sum().item()
        num_touched_experts = torch.unique(gathered_topk_idx_local.flatten()).numel() - 1

        # TFLOPS: 3 matmuls (L1 left, L1 right, L2), each 2 * M * N * K
        safe_div = lambda a, b: float('nan') if b == 0 else a / b
        tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_fused)

        # HBM bytes: weights (FP4 = 0.5B) + acts (FP8=1B or FP4=0.5B) + output (BF16=2B)
        use_fp4 = os.environ.get('DG_USE_FP4_ACTS', '0') != '0' or os.environ.get('DG_MEGA_MOE_FP4', '0') != '0'
        act_bytes = 0.5 if use_fp4 else 1.0
        num_hbm_bytes = (
            num_touched_experts * intermediate_hidden * 2 * hidden // 2 +    # L1 weights (FP4)
            num_touched_experts * hidden * intermediate_hidden // 2 +        # L2 weights (FP4)
            int(num_recv_tokens * hidden * act_bytes) +                      # L1 acts read
            int(num_recv_tokens * intermediate_hidden * act_bytes) +         # L1 output write (acts)
            int(num_recv_tokens * intermediate_hidden * act_bytes) +         # L2 acts read
            num_recv_tokens * hidden * 2                                      # L2 output write (BF16)
        )
        hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_fused)

        # Arithmetic intensity (FLOPs / Byte)
        ai = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3), num_hbm_bytes)

        dist_print('Performance:', once_in_node=True)
        for r in range(num_ranks):
            if r == rank_idx:
                print(f'[rank {rank_idx:2}/{num_ranks}] | '
                      f'{tflops:6.1f} TFLOPS | '
                      f'HBM {hbm_gbs:6.1f} GB/s | '
                      f'AI {ai:6.1f} FLOP/B | '
                      f'{t_fused * 1e6:6.1f} us | '
                      f'tokens={num_recv_tokens} touched_experts={num_touched_experts}',
                      flush=True)
            dist.barrier()

        # Destroy and exit
        dist.barrier()
        buffer.destroy()
        dist.destroy_process_group()
        return

    # Non-overlapped baseline: EP dispatch + GEMM + EP combine
    deep_ep, tilelang_ops, tilelang_bench, is_legacy_loaded = import_baseline()
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    ep_buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank, hidden=hidden,
        num_topk=num_topk, use_fp8_dispatch=True,
        explicitly_destroy=True,
        allow_multiple_reduction=False,
        gpu_timeout_secs=10, cpu_timeout_secs=30
    ) if is_legacy_loaded else None

    def run_baseline():
        recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
            x, topk_idx=topk_idx, topk_weights=topk_weights,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_baseline,
            num_experts=num_experts, expert_alignment=alignment,
            do_cpu_sync=False, do_handle_copy=False,
            do_expand=True, use_tma_aligned_col_major_sf=True,
        )
        n = recv_x[0].size(0)
        l1_y = torch.empty((n, intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            recv_x, l1_weights, l1_y, handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True, recipe=(1, 1, 32))
        # noinspection PyCallingNonCallable
        l1_y = tilelang_ops.swiglu_apply_weight_to_fp8(
            x=l1_y,
            topk_weights=recv_topk_weights,
            avail_tokens=handle.psum_num_recv_tokens_per_expert[-1],
            num_per_channels=32,
            use_col_major_scales=True,
            round_scale=True,
            ue8m0_scale=True,
            output_bf16=False,
            clamp_value=args.activation_clamp,
            fast_math=bool(args.fast_math)
        )
        l2_y = torch.empty((n, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            l1_y, l2_weights, l2_y, handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True, recipe=(1, 1, 32))
        return ep_buffer.combine(l2_y, handle=handle)[0], cumulative_local_expert_recv_stats_baseline

    # Check correctness (must be bitwise identical)
    num_correctness_tests = 1 if args.num_correctness_tests is None else args.num_correctness_tests
    # noinspection PyBroadException
    if is_legacy_loaded and num_correctness_tests > 0:
        dist_print('Running correctness tests:', once_in_node=True)
        for i in range(num_correctness_tests):
            create_inputs()
            for fused_result, baseline_result in zip(run_fused(), run_baseline()):
                assert torch.equal(fused_result, baseline_result)
            if (i + 1) % 100 == 0 or i == num_correctness_tests - 1:
                dist_print(f' > Correctness test #{i + 1}/{num_correctness_tests} passed', once_in_node=True)
        dist_print(once_in_node=True)
    else:
        create_inputs()

    # Count local received tokens
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) | \
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    # Benchmark
    t_fused = bench_kineto(
        run_fused, 'mega_moe',
        barrier=lambda: ep_buffer.barrier(use_comm_stream=False) if ep_buffer else dist.barrier(),
        trace_path=None if not args.dump_profile_traces else f'{args.dump_profile_traces}/mega_moe_rank{rank_idx}.json')
    t_baseline = tilelang_bench(run_baseline, _n_warmup=5, _n_repeat=1, backend='cudagraph', return_mode='median') / 1e3 if is_legacy_loaded else 0

    # TFLOPS: 3 matmuls (L1 left, L1 right, L2), each 2 * M * N * K
    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_fused)

    # HBM bytes: weights (FP4 packed = 0.5 bytes) + activations (FP8 = 1 byte) + output (BF16 = 2 bytes)
    num_touched_experts = torch.unique(gathered_topk_idx.flatten()).numel() - 1 # NOTES minus 1 to exclude "-1"
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden // 2 +   # L1 weights (FP4)
        num_touched_experts * hidden * intermediate_hidden // 2 +       # L2 weights (FP4)
        num_recv_tokens * hidden +                                      # L1 acts read (FP8)
        num_recv_tokens * intermediate_hidden +                         # L1 output write (FP8)
        num_recv_tokens * intermediate_hidden +                         # L2 acts read (FP8)
        num_recv_tokens * hidden * 2                                    # L2 output write (BF16)
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_fused)

    # NVLink bytes: dispatch pull + combine write-back
    num_nvlink_bytes = num_recv_tokens * hidden * 3
    nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, t_fused)

    # Combine reduction (serial) time approximation
    t_reduction = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12

    # Summary
    approx_factor = t_fused / (t_fused - t_reduction)
    dist_print('Performance:', once_in_node=True)
    dist_print(f' > EP: {rank_idx:2}/{num_ranks} | '
               f'{tflops:4.0f} TFLOPS | '
               f'overlap: '
               f'{tflops * approx_factor:4.0f} TFLOPS, '
               f'HBM {hbm_gbs * approx_factor:4.0f} GB/s, '
               f'NVL {nvlink_gbs * approx_factor:3.0f} GB/s | '
               f'{t_fused * 1e6:4.0f} us, '
               f'reduction: {t_reduction * 1e6:4.1f} us | '
               f'{safe_div(t_baseline, t_fused):.2f}x legacy')

    # Exit
    dist.barrier()
    buffer.destroy()
    ep_buffer.destroy() if is_legacy_loaded else None
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PyTorch symmetric memory')

    # Resource settings
    parser.add_argument('--ncu-profile-only', action='store_true', help='Only run profiling without correctness test')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--shape-ntoks', type=str, default='',
                        help='Comma-separated tokens/rank shapes to sweep in-process, e.g. "8,32,128,512"')
    parser.add_argument('--bench-reps', type=int, default=3,
                        help='Number of bench_kineto repetitions per shape (in-process)')

    # Model settings
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192, help='Number of maximum tokens per rank')
    parser.add_argument('--num-tokens', type=int, default=0, help='Number of tokens per rank (follow max minus removed if 0)')
    parser.add_argument('--num-max-removed-tokens', type=int, default=0, help='Maximum number of tokens to remove')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=3072, help='Intermediate hidden size')
    parser.add_argument('--activation-clamp', type=float, default=10, help='Clamp value for activation')
    parser.add_argument('--num-experts', type=int, default=384, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of expert selections')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    parser.add_argument('--fast-math', type=int, default=1, help='Enable fast math (0 or 1, default: 1)')

    # Test settings
    parser.add_argument('--num-correctness-tests', type=int, default=None, help='Pressure test')
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')
    parser.add_argument('--local-rank-idx', type=int, default=None, help='Run as single process with this local rank (e.g. for NCU prof)')
    args = parser.parse_args()

    # Create dump trace directories
    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    if args.local_rank_idx is not None:
        # Single-process mode: each process is launched separately (e.g. by NCU)
        test(args.local_rank_idx, args.num_processes, args)
    else:
        # Launch tests
        num_processes = args.num_processes
        torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)
