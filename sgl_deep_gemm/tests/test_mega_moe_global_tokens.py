# `global_num_tokens`: tile selection from the EP-wide token count.
#
# Without the hint every rank sizes BLOCK_M from `local tokens x num_ranks`. Under
# attention DP an idle or lightly loaded rank then picks a tiny tile while still
# computing the experts that remote ranks route to it.
#
# Checks, for balanced / single-active-rank / uneven / all-idle loads:
#   - the hint only changes tiling: `y` and expert receive counts are bitwise equal
#   - the kernel's BLOCK_M matches `get_block_m_for_mega_moe` with the same hint,
#     and on idle ranks it differs from the unhinted choice (the hint has teeth)
#   - out-of-range hints are rejected
#
# Usage:
#   python3 sgl_deep_gemm/tests/test_mega_moe_global_tokens.py --num-processes 8
#   # MiniMax M3.1 prefill shape, one active rank holding a 16K chunk:
#   python3 sgl_deep_gemm/tests/test_mega_moe_global_tokens.py --num-processes 8 \
#       --hidden 6144 --intermediate-hidden 3072 --num-experts 128 --num-topk 4 \
#       --num-tokens 16384 --num-max-tokens-per-rank 16384 --patterns single_first --bench 30

import argparse
import os
import re
import sys
import tempfile
import torch
import torch.distributed as dist
from typing import Dict, Optional, Tuple

import deep_gemm
from deep_gemm.utils import per_token_cast_to_nvfp4, transform_ue4m3_sf_into_required_layout
from deep_gemm.utils.dist import dist_print, init_dist

GRAN_K = 16
MMA_TYPE = 'nvfp4xnvfp4'
_CONFIG_LINE = re.compile(r'num_tokens=(\d+), global_num_tokens=(-?\d+),.*: MegaMoEConfig\(block_m=(\d+)')


def cast_weights(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_groups, n, k = w.shape
    packed = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
    sf = torch.empty((num_groups, n, k // GRAN_K), device='cuda', dtype=torch.float)
    for i in range(num_groups):
        packed[i], sf[i] = per_token_cast_to_nvfp4(w[i], gran_k=GRAN_K)
    return packed, transform_ue4m3_sf_into_required_layout(sf, n)


def tokens_per_rank(pattern: str, num_ranks: int, num_tokens: int) -> list:
    if pattern == 'balanced':
        return [num_tokens] * num_ranks
    if pattern == 'single_first':
        return [num_tokens] + [0] * (num_ranks - 1)
    if pattern == 'single_last':
        return [0] * (num_ranks - 1) + [num_tokens]
    if pattern == 'uneven':
        return [num_tokens * i // max(num_ranks - 1, 1) for i in range(num_ranks)]
    if pattern == 'all_idle':
        return [0] * num_ranks
    raise ValueError(f'Unknown pattern {pattern!r}')


class ConfigCapture:
    """Collects `DG_PRINT_CONFIGS` lines, which the C++ heuristic writes to fd 1 once per key."""

    def __init__(self):
        self.block_m: Dict[Tuple[int, int], int] = {}

    def run(self, fn):
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
            for match in _CONFIG_LINE.finditer(f.read()):
                self.block_m[(int(match.group(1)), int(match.group(2)))] = int(match.group(3))
        return result


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    os.environ['DG_PRINT_CONFIGS'] = '1'
    hidden, inter = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_local_experts = num_experts // num_ranks
    num_max_tokens_per_rank = args.num_max_tokens_per_rank

    # Out-of-range hints must fail before any kernel launch.
    limit = num_ranks * num_max_tokens_per_rank
    for bad in (-1, limit + 1):
        try:
            deep_gemm.get_block_m_for_mega_moe(
                num_ranks, num_experts, num_max_tokens_per_rank, 1, num_topk, MMA_TYPE, global_num_tokens=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'global_num_tokens={bad} was accepted (limit {limit})')
    dist_print(f' > out-of-range hints rejected (valid range [0, {limit}])', once_in_node=True)

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, num_max_tokens_per_rank, num_topk, hidden, inter, mma_type=MMA_TYPE)

    torch.manual_seed(rank_idx)
    l1 = torch.randn((num_local_experts, inter * 2, hidden), dtype=torch.bfloat16, device='cuda')
    l2 = torch.randn((num_local_experts, hidden, inter), dtype=torch.bfloat16, device='cuda')
    l1_weights, l2_weights = deep_gemm.transform_weights_for_mega_moe(
        cast_weights(l1), cast_weights(l2), 'swiglu', MMA_TYPE)
    del l1, l2

    capture = ConfigCapture()
    dist_print(f'Config: {num_ranks} ranks x {num_local_experts} local experts, top-{num_topk}, '
               f'hidden={hidden}, inter={inter}, {num_max_tokens_per_rank} max tokens/rank', once_in_node=True)

    for pattern in args.patterns.split(','):
        counts = tokens_per_rank(pattern, num_ranks, args.num_tokens)
        num_tokens = counts[rank_idx]
        global_num_tokens = sum(counts)

        torch.manual_seed(1000 + rank_idx)
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 0.1
        if num_tokens > 0:
            x_packed, x_sf = per_token_cast_to_nvfp4(x, gran_k=GRAN_K, use_packed_ue4m3=True)
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        # Idle ranks launch one masked row, as SGLang does: the binding needs a non-null `y`.
        num_rows = max(num_tokens, 1)

        def run(hint: Optional[int], recv_stats: torch.Tensor) -> torch.Tensor:
            buffer.topk_idx[:num_rows].fill_(-1)
            buffer.topk_weights[:num_rows].zero_()
            if num_tokens > 0:
                buffer.x[:num_tokens].copy_(x_packed)
                buffer.x_sf[:num_tokens].copy_(x_sf)
                buffer.topk_idx[:num_tokens].copy_(topk_idx)
                buffer.topk_weights[:num_tokens].copy_(topk_weights)
            y = torch.empty((num_rows, hidden), dtype=torch.bfloat16, device='cuda')
            deep_gemm.nvfp4_mega_moe(
                y, l1_weights, l2_weights, buffer,
                cumulative_local_expert_recv_stats=recv_stats,
                fast_math=bool(args.fast_math), global_num_tokens=hint)
            return y[:num_tokens]

        def run_and_check(hint: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor, int]:
            stats = torch.zeros((num_local_experts, ), dtype=torch.int, device='cuda')
            dist.barrier()
            y = capture.run(lambda: run(hint, stats))
            dist.barrier()
            key = (num_rows, -1 if hint is None else hint)
            assert key in capture.block_m, f'[{pattern}] kernel config for {key} was not printed'
            expected = deep_gemm.get_block_m_for_mega_moe(
                num_ranks, num_experts, buffer.num_max_tokens_per_rank, num_rows, num_topk, MMA_TYPE,
                global_num_tokens=hint)
            assert capture.block_m[key] == expected, \
                f'[{pattern}] rank {rank_idx}: kernel BLOCK_M {capture.block_m[key]} != ' \
                f'get_block_m_for_mega_moe {expected} (hint={hint})'
            return y, stats, expected

        y_local, stats_local, block_m_local = run_and_check(None)
        y_global, stats_global, block_m_global = run_and_check(global_num_tokens)
        assert torch.equal(y_local, y_global), \
            f'[{pattern}] rank {rank_idx}: global_num_tokens changed the output, max |delta| ' \
            f'{(y_local.float() - y_global.float()).abs().max().item() if num_tokens else 0:.3e}'
        assert torch.equal(stats_local, stats_global), f'[{pattern}] rank {rank_idx}: receive counts differ'
        num_received = stats_global.sum().clone()
        dist.all_reduce(num_received, group=group)
        num_routed = torch.tensor(num_tokens * num_topk, dtype=torch.int, device='cuda')
        dist.all_reduce(num_routed, group=group)
        assert num_received.item() == num_routed.item(), \
            f'[{pattern}] received {num_received.item()} routed tokens, expected {num_routed.item()}'

        # Idle ranks are where the local estimate goes wrong; the hint must change their tile.
        if num_tokens == 0 and global_num_tokens >= num_ranks * num_max_tokens_per_rank // 8:
            assert block_m_local != block_m_global, \
                f'[{pattern}] idle rank {rank_idx}: BLOCK_M {block_m_local} unchanged by the hint'

        block_ms = [None] * num_ranks
        dist.all_gather_object(block_ms, (block_m_local, block_m_global), group=group)
        dist_print(f' > [{pattern:12s}] tokens/rank {counts}: bitwise equal; '
                   f'BLOCK_M local/global per rank {block_ms}', once_in_node=True)

        if args.bench and pattern in args.bench_patterns.split(','):
            for hint in (None, global_num_tokens):
                for _ in range(5):
                    run(hint, None)
            times = {None: [], global_num_tokens: []}
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(args.bench):
                # Alternate arms so clock or thermal drift affects both equally.
                for hint in (None, global_num_tokens):
                    dist.barrier()
                    torch.cuda.synchronize()
                    start.record()
                    run(hint, None)
                    end.record()
                    torch.cuda.synchronize()
                    times[hint].append(start.elapsed_time(end) * 1e3)
            medians = torch.tensor([sorted(t)[len(t) // 2] for t in times.values()], device='cuda')
            all_medians = [torch.empty_like(medians) for _ in range(num_ranks)]
            dist.all_gather(all_medians, medians, group=group)
            all_medians = torch.stack(all_medians).tolist()
            dist_print(f'   bench [{pattern}] median us per rank (local -> global):', once_in_node=True)
            for r, (t_local, t_global) in enumerate(all_medians):
                dist_print(f'     rank {r} ({counts[r]:6d} tokens, BLOCK_M {block_ms[r][0]:3d} -> '
                           f'{block_ms[r][1]:3d}): {t_local:9.1f} -> {t_global:9.1f} '
                           f'({t_global / t_local - 1:+.1%})', once_in_node=True)

    dist_print('OK', once_in_node=True)
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=2)
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=1024)
    parser.add_argument('--num-tokens', type=int, default=1024, help='tokens on a fully loaded rank')
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--intermediate-hidden', type=int, default=512)
    parser.add_argument('--num-experts', type=int, default=16)
    parser.add_argument('--num-topk', type=int, default=4)
    parser.add_argument('--fast-math', type=int, default=1)
    parser.add_argument('--patterns', type=str, default='balanced,single_first,single_last,uneven,all_idle')
    parser.add_argument('--bench', type=int, default=0, help='time each arm N times')
    parser.add_argument('--bench-patterns', type=str, default='single_first,uneven')
    args = parser.parse_args()
    assert args.num_experts % args.num_processes == 0
    assert args.num_tokens <= args.num_max_tokens_per_rank
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
    sys.exit(0)
