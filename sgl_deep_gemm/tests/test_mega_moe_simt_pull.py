# SIMT dispatch pull vs the TMA pull, for FP4 activations.
#
# The pull only moves token bytes, scale factors, top-k weights and x-scales into the local ring, so the
# two pulls must produce bitwise-equal outputs and expert receive counts. Checks, per MMA type and token count:
#   - `DG_MEGA_MOE_SIMT_PULL=0/1` resolves to the requested pull (read from `DG_PRINT_CONFIGS`)
#   - `y` and the receive counts are bitwise equal between the two pulls
# The default shape wraps the ring at full load (ring < routed tokens), which exercises the slot-reuse wait.
#
# Usage:
#   python3 sgl_deep_gemm/tests/test_mega_moe_simt_pull.py --num-processes 2

import argparse
import os
import re
import sys
import tempfile
import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_nvfp4, transform_ue4m3_sf_into_required_layout
from deep_gemm.utils.dist import dist_print, init_dist

_CONFIG_LINE = re.compile(r'num_tokens=(\d+),.*use_simt_pull=(true|false)\)')


def capture_simt_pull(fn):
    """Runs `fn` and returns the `use_simt_pull` values the C++ heuristic printed to fd 1."""
    sys.stdout.flush()
    saved = os.dup(1)
    with tempfile.TemporaryFile(mode='w+') as f:
        os.dup2(f.fileno(), 1)
        try:
            result = fn()
            torch.cuda.synchronize()
        finally:
            os.dup2(saved, 1)
            os.close(saved)
        f.seek(0)
        return result, {m.group(2) == 'true' for m in _CONFIG_LINE.finditer(f.read())}


def cast_weights(w: torch.Tensor, is_nvfp4: bool):
    num_groups, n, k = w.shape
    gran_k = 16 if is_nvfp4 else 32
    packed = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
    sf = torch.empty((num_groups, n, k // gran_k), device='cuda', dtype=torch.float)
    for i in range(num_groups):
        packed[i], sf[i] = per_token_cast_to_nvfp4(w[i], gran_k=gran_k) if is_nvfp4 else \
            per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=gran_k)
    sf = transform_ue4m3_sf_into_required_layout(sf, n) if is_nvfp4 else \
        deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, gran_k), num_groups)
    return packed, sf


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    os.environ['DG_PRINT_CONFIGS'] = '1'
    hidden, inter = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_local_experts = num_experts // num_ranks
    resolved = set()

    for mma_type, use_x_scales in (('nvfp4xnvfp4', False), ('nvfp4xnvfp4', True), ('mxf4xmxf4', False)):
        is_nvfp4 = mma_type == 'nvfp4xnvfp4'
        gran_k = 16 if is_nvfp4 else 32
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group, num_experts, args.num_max_tokens_per_rank, num_topk, hidden, inter, mma_type=mma_type)
        torch.manual_seed(rank_idx)
        l1_weights, l2_weights = deep_gemm.transform_weights_for_mega_moe(
            cast_weights(torch.randn((num_local_experts, inter * 2, hidden), dtype=torch.bfloat16, device='cuda'), is_nvfp4),
            cast_weights(torch.randn((num_local_experts, hidden, inter), dtype=torch.bfloat16, device='cuda'), is_nvfp4),
            'swiglu', mma_type)
        l1_alphas = torch.ones((num_local_experts, 2), device='cuda') if use_x_scales else None
        l2_alphas = torch.ones(num_local_experts, device='cuda') if use_x_scales else None

        for num_tokens in [int(t) for t in args.tokens.split(',')]:
            torch.manual_seed(1000 + rank_idx)
            x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 0.1
            scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
            topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
            if not use_x_scales:
                x_q = per_token_cast_to_nvfp4(x, gran_k=gran_k, use_packed_ue4m3=True) if is_nvfp4 else \
                    per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=gran_k, use_packed_ue8m0=True)

            def run():
                if use_x_scales:
                    deep_gemm.mega_moe_pre_dispatch(
                        x, topk_idx.int(), topk_weights, buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights,
                        num_tokens=num_tokens, group_size=gran_k, mma_type=mma_type, buf_x_scales=buffer.x_scales)
                else:
                    buffer.x[:num_tokens].copy_(x_q[0])
                    buffer.x_sf[:num_tokens].copy_(x_q[1])
                    buffer.topk_idx[:num_tokens].copy_(topk_idx)
                    buffer.topk_weights[:num_tokens].copy_(topk_weights)
                y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                stats = torch.zeros((num_local_experts, ), dtype=torch.int, device='cuda')
                deep_gemm.fp8_fp4_mega_moe(
                    y=y, l1_weights=l1_weights, l2_weights=l2_weights, sym_buffer=buffer,
                    cumulative_local_expert_recv_stats=stats, recipe=(1, 1, gran_k), activation='swiglu',
                    fast_math=bool(args.fast_math), use_x_scales=use_x_scales, l1_alphas=l1_alphas, l2_alphas=l2_alphas)
                return y, stats

            outputs = {}
            for simt_pull in (0, 1):
                os.environ['DG_MEGA_MOE_SIMT_PULL'] = str(simt_pull)
                dist.barrier()
                outputs[simt_pull], printed = capture_simt_pull(run)
                # The heuristic prints each config key once; x-scales share the key of the plain NVFP4 round
                key = (mma_type, num_tokens, simt_pull)
                assert printed == {bool(simt_pull)} or (not printed and key in resolved), \
                    f'{mma_type}: DG_MEGA_MOE_SIMT_PULL={simt_pull} resolved to use_simt_pull {printed}'
                resolved.add(key)
            del os.environ['DG_MEGA_MOE_SIMT_PULL']
            (y_tma, stats_tma), (y_simt, stats_simt) = outputs[0], outputs[1]
            assert torch.isfinite(y_tma).all(), f'{mma_type}: non-finite output'
            assert torch.equal(y_tma, y_simt), \
                f'{mma_type} (x-scales {use_x_scales}), {num_tokens} tokens, rank {rank_idx}: SIMT pull changed y, ' \
                f'max |delta| {(y_tma.float() - y_simt.float()).abs().max().item():.3e}'
            assert torch.equal(stats_tma, stats_simt), f'{mma_type}, {num_tokens} tokens: receive counts differ'
            dist_print(f' > {mma_type:12s} x-scales {int(use_x_scales)}, {num_tokens:5d} tokens/rank: '
                       f'SIMT == TMA pull, bitwise', once_in_node=True)

        dist.barrier()
        buffer.destroy()

    dist_print('OK', once_in_node=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=2)
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=4096)
    parser.add_argument('--tokens', type=str, default='4096,1000,7', help='tokens per rank, comma-separated')
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--intermediate-hidden', type=int, default=512)
    parser.add_argument('--num-experts', type=int, default=16)
    parser.add_argument('--num-topk', type=int, default=4)
    parser.add_argument('--fast-math', type=int, default=1)
    args = parser.parse_args()
    assert args.num_experts % args.num_processes == 0
    assert args.hidden % 1024 == 0, 'the SIMT pull needs FP4 tokens of a multiple of 512 bytes'
    assert max(int(t) for t in args.tokens.split(',')) <= args.num_max_tokens_per_rank
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
    sys.exit(0)
