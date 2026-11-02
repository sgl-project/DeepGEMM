import argparse
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Tuple

import deep_gemm
from deep_gemm.utils import (per_token_cast_to_fp4, per_token_cast_to_fp8, per_token_cast_to_nvfp4,
                             transform_ue4m3_sf_into_required_layout)
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto


def import_baseline():
    # Load legacy implements from third-party
    deep_ep, tilelang_ops, do_bench, is_legacy_loaded = None, None, None, False
    # noinspection PyBroadException
    try:
        import deep_ep
        assert hasattr(deep_ep, 'ElasticBuffer'), 'deep_ep has no ElasticBuffer (DeepEP v2 required)'
        import importlib.util
        from tilelang.profiler.bench import do_bench
        spec = importlib.util.spec_from_file_location(
            'tilelang_ops',
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'third-party', 'tilelang_ops', '__init__.py'))
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
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    # Settings
    is_bf16xbf16 = args.mma_type == 'bf16xbf16'
    use_nvfp4 = args.mma_type == 'nvfp4xnvfp4'
    gran_k = 16 if use_nvfp4 else 32
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = max(0, args.num_max_tokens_per_rank - random.randint(0, args.num_max_removed_tokens)) \
        if args.num_tokens == 0 else args.num_tokens
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank
    norm_gain = .5 + (torch.arange(hidden, device='cuda') % 17).float() / 16 if args.post_expert_norm else None
    l1_alphas = torch.ones((num_experts_per_rank, 2), device='cuda') if args.use_x_scales else None
    l2_alphas = torch.ones(num_experts_per_rank, device='cuda') if args.use_x_scales else None

    # Allocate symmetric memory
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        mma_type=args.mma_type,
        activation=args.activation
    )

    # Cast weights into FP4
    def _cast_weights_to_fp4(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_groups, n, k = bf16_weights.shape
        w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
        w_sf = torch.empty((num_groups, n, k // gran_k), device='cuda', dtype=torch.float)
        for i in range(num_groups):
            if use_nvfp4:
                w[i], w_sf[i] = per_token_cast_to_nvfp4(bf16_weights[i], gran_k=gran_k)
            else:
                w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=gran_k)
        w_sf = transform_ue4m3_sf_into_required_layout(w_sf, n) if use_nvfp4 else \
            deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, gran_k), num_groups)
        return w, w_sf

    # Create inputs
    # noinspection PyGlobalUndefined
    def create_inputs():
        global x, topk_idx, topk_weights, l1_weights, l2_weights, transformed_l1_weights, transformed_l2_weights
        global cumulative_local_expert_recv_stats_fused
        global cumulative_local_expert_recv_stats_baseline
        x = (torch.randn((num_tokens, hidden), device='cuda', dtype=torch.bfloat16) * args.x_scale).bfloat16()
        l1_weights = torch.randn(
            (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
        l2_weights = torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        if args.normalize_topk_weights:
            topk_weights = topk_weights.softmax(dim=-1)
        cumulative_local_expert_recv_stats_fused = torch.randint(
            0, 100, (num_experts_per_rank, ), dtype=torch.int, device='cuda')
        cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_fused.clone()
        if args.masked_ratio > 0:
            rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
            topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
            topk_weights.masked_fill_(topk_idx < 0, 0)
        if args.use_x_scales:
            topk_idx = topk_idx.int()

        if not is_bf16xbf16:
            # Cast inputs to FP8/FP4 with per-32 UE8M0 SF
            assert hidden % 128 == 0 and intermediate_hidden % 128 == 0

            # Stream A0.0b: when the flag is on, the symm buffer's `x` slot is sized
            # for packed E2M1 (`hidden/2` bytes/token), so we must quantize at the
            # source to match.
            if use_nvfp4:
                if not args.use_x_scales:
                    x = per_token_cast_to_nvfp4(x, gran_k=gran_k, use_packed_ue4m3=True) if num_tokens else (
                        torch.empty((0, hidden // 2), dtype=torch.int8, device='cuda'),
                        torch.empty((0, hidden // 64), dtype=torch.int32, device='cuda'))
            elif args.mma_type == 'mxf4xmxf4':
                x = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            else:
                x = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)

            l1_weights = _cast_weights_to_fp4(l1_weights)
            l2_weights = _cast_weights_to_fp4(l2_weights)

        transformed_l1_weights, transformed_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights, args.activation, args.mma_type))
        if args.weight_layout == 'trtllm':
            from test_mega_moe_nvfp4_trtllm import l1_to_trtllm, l2_to_trtllm, sf_to_trtllm, swap_l1_sf_halves
            l1, l2 = transformed_l1_weights, transformed_l2_weights
            transformed_l1_weights = l1_to_trtllm(l1[0]), sf_to_trtllm(swap_l1_sf_halves(l1[1]), True)
            transformed_l2_weights = l2_to_trtllm(l2[0]), sf_to_trtllm(l2[1], False)
        if args.use_localization:
            assert deep_gemm.is_localization_available(), 'Weight localization unavailable'
            localize = deep_gemm.localize_rows if args.weight_layout == 'trtllm' else deep_gemm.localize
            def place(weight):
                return (localize(weight[0]), weight[1]) if isinstance(weight, tuple) else localize(weight)
            transformed_l1_weights, transformed_l2_weights = place(transformed_l1_weights), place(transformed_l2_weights)

    # Run fused mega MoE
    # NOTES: copy x into buffer before each call because debug mode zeros the entire buffer
    def run_fused():
        if args.use_x_scales:
            if num_tokens:
                deep_gemm.mega_moe_pre_dispatch(x, topk_idx, topk_weights, buffer.x, buffer.x_sf,
                    buffer.topk_idx, buffer.topk_weights, num_tokens, group_size=16,
                    mma_type=args.mma_type, buf_x_scales=buffer.x_scales)
        else:
            if is_bf16xbf16:
                buffer.x[:num_tokens].copy_(x)
            else:
                buffer.x[:num_tokens].copy_(x[0])
                buffer.x_sf[:num_tokens].copy_(x[1])
            buffer.topk_idx[:num_tokens].copy_(topk_idx)
            buffer.topk_weights[:num_tokens].copy_(topk_weights)

        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        kernel_kwargs = dict(
            y=y, l1_weights=transformed_l1_weights, l2_weights=transformed_l2_weights,
            sym_buffer=buffer,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_fused,
            activation=args.activation,
            activation_clamp=args.activation_clamp if args.activation != 'situ' else None,
            fast_math=bool(args.fast_math), weight_layout=args.weight_layout)
        if args.post_expert_norm:
            kernel_kwargs.update(post_expert_norm=True, post_expert_norm_weight=norm_gain,
                post_expert_norm_eps=args.post_expert_norm_eps,
                post_expert_norm_weight_first=args.post_expert_norm_weight_first)
        if args.use_x_scales:
            kernel_kwargs.update(use_x_scales=True, l1_alphas=l1_alphas, l2_alphas=l2_alphas)
        if not is_bf16xbf16:
            kernel_kwargs['recipe'] = (1, 1, gran_k)
        (deep_gemm.bf16_mega_moe if is_bf16xbf16 else deep_gemm.fp8_fp4_mega_moe)(**kernel_kwargs)
        return y, cumulative_local_expert_recv_stats_fused

    dist_print('Config:', once_in_node=True)
    dist_print(f' > MMA: {args.mma_type}', once_in_node=True)
    dist_print(f' > Weight layout: {args.weight_layout}, localized: {args.use_localization}', once_in_node=True)
    dist_print(f' > Post-expert norm: {args.post_expert_norm}, weight first: {args.post_expert_norm_weight_first}, '
               f'epsilon: {args.post_expert_norm_eps}', once_in_node=True)
    dist_print(f' > Tokens: {num_tokens}/{num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    dist_print(f' > Buffer: {buffer.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(once_in_node=True)

    # Only do NCU profiling
    if args.ncu_profile_only:
        create_inputs()
        dist_print(f'Run fused kernel:', once_in_node=True)
        run_fused()
        dist_print(f' > Done, exiting', once_in_node=True)

        # Destroy and exit
        dist.barrier()
        buffer.destroy()
        deep_gemm.destroy_localizer()
        dist.destroy_process_group()
        return

    # Non-overlapped baseline: EP dispatch + GEMM + EP combine
    deep_ep, tilelang_ops, tilelang_bench, is_legacy_loaded = (None, None, None, False) if use_nvfp4 else import_baseline()
    if not use_nvfp4 and not is_legacy_loaded and args.num_correctness_tests != 0:
        raise RuntimeError("Mega MoE correctness requires DeepEP ElasticBuffer and TileLang reference operators")
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    ep_buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank, hidden=hidden,
        num_topk=num_topk, use_fp8_dispatch=True,
        explicitly_destroy=True,
        allow_multiple_reduction=False,
        num_gpu_timeout_secs=10, num_cpu_timeout_secs=30
    ) if is_legacy_loaded else None

    # Baseline params differ by mma type
    run_baseline = None
    if is_legacy_loaded:
        if is_bf16xbf16:
            dispatch_kwargs = {'do_cpu_sync': False, 'do_handle_copy': False, 'do_expand': True}
            gemm_fn = deep_gemm.m_grouped_bf16_gemm_nt_contiguous
            gemm_kwargs = {'compiled_dims': '', 'use_psum_layout': True}
            swiglu_kwargs = {'round_scale': False, 'ue8m0_scale': False, 'output_bf16': True}
            get_num_tokens = lambda recv_x: recv_x.size(0)
        else:
            dispatch_kwargs = {'do_cpu_sync': False, 'do_handle_copy': False,
                               'do_expand': True, 'use_tma_aligned_col_major_sf': True}
            gemm_fn = deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous
            gemm_kwargs = {'use_psum_layout': True, 'recipe': (1, 1, 32)}
            swiglu_kwargs = {'round_scale': True, 'ue8m0_scale': True, 'output_bf16': False}
            get_num_tokens = lambda recv_x: recv_x[0].size(0)

        def run_baseline():
            # Dispatch
            recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
                x, topk_idx=topk_idx, topk_weights=topk_weights,
                cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_baseline,
                num_experts=num_experts, expert_alignment=alignment,
                **dispatch_kwargs)
            num_recv_tokens = get_num_tokens(recv_x)

            # L1 GEMM
            l1_y = torch.empty((num_recv_tokens, intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
            gemm_fn(recv_x, l1_weights, l1_y, handle.psum_num_recv_tokens_per_expert, **gemm_kwargs)

            # SwiGLU
            swiglu_result = tilelang_ops.swiglu_apply_weight_to_fp8(
                x=l1_y, topk_weights=recv_topk_weights,
                avail_tokens=handle.psum_num_recv_tokens_per_expert[-1],
                num_per_channels=32, use_col_major_scales=True,
                clamp_value=args.activation_clamp, fast_math=bool(args.fast_math),
                **swiglu_kwargs)
            l1_y = swiglu_result[-1] if is_bf16xbf16 else swiglu_result

            # L2 GEMM
            l2_y = torch.empty((num_recv_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            gemm_fn(l1_y, l2_weights, l2_y, handle.psum_num_recv_tokens_per_expert, **gemm_kwargs)

            # Combine
            return ep_buffer.combine(l2_y, handle=handle)[0], cumulative_local_expert_recv_stats_baseline

    # NVFP4 correctness is covered by the NVFP4 layout, scale and global-token tests.
    num_correctness_tests = 1 if args.num_correctness_tests is None else args.num_correctness_tests
    can_compare_bitwise = is_bf16xbf16 or args.mma_type == 'fp8xfp4'
    if use_nvfp4 and num_correctness_tests > 0:
        dist_print('NVFP4 correctness: run test_mega_moe_nvfp4_trtllm.py, '
                   'test_mega_moe_nvfp4_alphas.py and test_mega_moe_global_tokens.py', once_in_node=True)
    elif is_legacy_loaded and num_correctness_tests > 0 and not can_compare_bitwise:
        dist_print(f'Skipping bitwise correctness vs legacy baseline (FP8-acts only, '
                   f'undefined for `{args.mma_type}`)', once_in_node=True)
    # noinspection PyBroadException
    if not use_nvfp4 and is_legacy_loaded and num_correctness_tests > 0 and can_compare_bitwise:
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
        run_fused, 'sm100_fp8_fp4_mega_moe' if args.use_x_scales else 'mega_moe',
        num_tests=args.num_bench_tests,
        barrier=lambda: ep_buffer.barrier(use_comm_stream=False) if ep_buffer else dist.barrier(),
        trace_path=None if not args.dump_profile_traces else f'{args.dump_profile_traces}/mega_moe_rank{rank_idx}.json')
    t_baseline = tilelang_bench(run_baseline, _n_warmup=5, _n_repeat=1, backend='cudagraph', return_mode='median') / 1e3 if is_legacy_loaded else 0

    # TFLOPS: 3 matmuls (L1 left, L1 right, L2), each 2 * M * N * K
    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_fused)

    # HBM bytes: weights + activations + output
    num_touched_experts = torch.unique(gathered_topk_idx[gathered_topk_idx >= 0]).numel()
    act_elem_size, weight_elem_size = (2, 2) if is_bf16xbf16 else (0.5 if use_nvfp4 else 1, 0.5)
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden * weight_elem_size   # L1 weights
        + num_touched_experts * hidden * intermediate_hidden * weight_elem_size     # L2 weights
        + num_recv_tokens * hidden * act_elem_size                                  # L1 acts read
        + num_recv_tokens * intermediate_hidden * act_elem_size                     # L1 output write
        + num_recv_tokens * intermediate_hidden * act_elem_size                     # L2 acts read
        + num_recv_tokens * hidden * 2                                              # L2 output write (always BF16)
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_fused)

    # NVLink bytes: dispatch pull + combine write-back
    num_nvlink_bytes = num_recv_tokens * hidden * (act_elem_size + 2)
    nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, t_fused)

    # Combine reduction (serial) time approximation
    t_reduction = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12

    # Summary
    approx_factor = t_fused / (t_fused - t_reduction)
    legacy_comparison = f' | {safe_div(t_baseline, t_fused):.2f}x legacy' if is_legacy_loaded else ''
    dist_print('Performance:', once_in_node=True)
    dist_print(f' > EP: {rank_idx:2}/{num_ranks} | '
               f'{tflops:4.0f} TFLOPS | '
               f'overlap: '
               f'{tflops * approx_factor:4.0f} TFLOPS, '
               f'HBM {hbm_gbs * approx_factor:4.0f} GB/s, '
               f'NVL {nvlink_gbs * approx_factor:3.0f} GB/s | '
               f'{t_fused * 1e6:4.0f} us, '
               f'reduction: {t_reduction * 1e6:4.1f} us{legacy_comparison}')
    max_time = torch.tensor(t_fused, dtype=torch.float64, device='cuda')
    dist.all_reduce(max_time, op=dist.ReduceOp.MAX, group=group)
    dist_print(f' > Max-rank kernel mean: {max_time.item() * 1e6:.2f} us '
               f'({gathered_topk_idx.size(0)} global tokens)', once_in_node=True)

    # Exit
    dist.barrier()
    buffer.destroy()
    ep_buffer.destroy() if is_legacy_loaded else None
    deep_gemm.destroy_localizer()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PyTorch symmetric memory')

    # Resource settings
    parser.add_argument('--ncu-profile-only', action='store_true', help='Only run profiling without correctness test')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')

    # Model settings
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192, help='Number of maximum tokens per rank')
    parser.add_argument('--num-tokens', type=int, default=0, help='Number of tokens per rank (follow max minus removed if 0)')
    parser.add_argument('--num-max-removed-tokens', type=int, default=0, help='Maximum number of tokens to remove')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=3072, help='Intermediate hidden size')
    parser.add_argument('--activation-clamp', type=float, default=10, help='Clamp value for activation')
    parser.add_argument('--num-experts', type=int, default=384, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of expert selections')
    parser.add_argument('--x-scale', type=float, default=1.0, help='Synthetic input amplitude')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    parser.add_argument('--normalize-topk-weights', action='store_true', help='Softmax the selected routing logits')
    parser.add_argument('--use-x-scales', action='store_true', help='NVFP4 pre-dispatch with per-token outer scales and unit weight alphas')
    parser.add_argument('--fast-math', type=int, default=1, help='Enable fast math (0 or 1, default: 1)')
    parser.add_argument('--mma-type', type=str, default='fp8xfp4',
                        choices=('fp8xfp4', 'mxf4xmxf4', 'nvfp4xnvfp4', 'bf16xbf16'), help='MMA type')
    parser.add_argument('--activation', type=str, default='swiglu', choices=('swiglu', 'situ', 'swigluoai'),
                        help='Activation: swiglu, situ, or swigluoai')
    parser.add_argument('--weight-layout', choices=('megamoe', 'trtllm'), default='megamoe')
    parser.add_argument('--use-localization', action='store_true', help='Localize weights along N; retain ordinary scale placement')
    parser.add_argument('--post-expert-norm', action='store_true', help='Enable weighted post-expert RMSNorm (NVFP4)')
    parser.add_argument('--post-expert-norm-weight-first', action='store_true')
    parser.add_argument('--post-expert-norm-eps', type=float, default=1e-6)

    # Test settings
    parser.add_argument('--num-correctness-tests', type=int, default=None, help='Pressure test')
    parser.add_argument('--num-bench-tests', type=int, default=30, help='Launches per profiler warmup/active step')
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')
    parser.add_argument('--local-rank-idx', type=int, default=None, help='Run as single process with this local rank (e.g. for NCU prof)')
    args = parser.parse_args()
    if args.num_bench_tests <= 0 or (args.num_correctness_tests is not None and args.num_correctness_tests < 0):
        parser.error('Benchmark iterations must be positive and correctness iterations nonnegative')
    if args.weight_layout == 'trtllm' and args.mma_type != 'nvfp4xnvfp4':
        parser.error('TRT-LLM weight layout requires NVFP4')
    if args.use_x_scales and args.mma_type != 'nvfp4xnvfp4':
        parser.error('--use-x-scales requires NVFP4')
    if args.post_expert_norm and (args.mma_type != 'nvfp4xnvfp4' or args.activation != 'swiglu'):
        parser.error('Post-expert norm currently requires NVFP4 SwiGLU')
    if args.post_expert_norm_weight_first and not args.post_expert_norm:
        parser.error('--post-expert-norm-weight-first requires --post-expert-norm')
    if args.post_expert_norm_eps <= 0:
        parser.error('--post-expert-norm-eps must be positive')

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
