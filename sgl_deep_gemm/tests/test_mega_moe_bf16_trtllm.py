# Check BF16 TRT-layout bitwise equality and benchmark both layouts.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29531 \
#       python3 sgl_deep_gemm/tests/test_mega_moe_bf16_trtllm.py --bench 60

import argparse
import sys
import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils.dist import dist_print, init_dist


def to_block_major_k(w: torch.Tensor) -> torch.Tensor:
    """Canonical `[E, N, K]` -> TRT-LLM BlockMajorK, physically `[E, K/64, N, 64]`."""
    e, n, k = w.shape
    assert n % 32 == 0 and k % 64 == 0
    return w.view(e, n, k // 64, 64).permute(0, 2, 1, 3).contiguous()


L1_ROW_MAP = [((r & 1) << 4) | ((1 - ((r >> 3) & 1)) << 3) | (((r >> 4) & 1) << 2) | ((r >> 1) & 3)
              for r in range(32)]


def l1_to_trtllm(w: torch.Tensor, row_map=None) -> torch.Tensor:
    """Scatter L1 rows: s4=r0, s3=~r3, s2=r4, s[1:0]=r[2:1]."""
    row_map = L1_ROW_MAP if row_map is None else row_map
    assert sorted(row_map) == list(range(32)), 'row map is not a permutation'
    e, n, k = w.shape
    bmk = to_block_major_k(w).view(e, k // 64, n // 32, 32, 64)
    out = torch.empty_like(bmk)
    out[:, :, :, torch.as_tensor(row_map, device=w.device)] = bmk
    return out.reshape(e, n, k)


def l2_to_trtllm(w: torch.Tensor) -> torch.Tensor:
    """TRT-LLM transposes each 32-row L2 group 8x4 -> 4x8."""
    e, n, k = w.shape
    bmk = to_block_major_k(w).view(e, k // 64, n // 32, 8, 4, 64)
    return bmk.transpose(3, 4).reshape(e, n, k).contiguous()


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_ranks == 1, 'one rank stands in for one EP rank of a larger MoE'
    torch.manual_seed(0)

    num_tokens, hidden, inter = args.num_tokens, args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_global_experts = max(args.num_global_experts, num_experts)

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_global_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    # Simulate one rank's expert slice.
    topk_idx = torch.where(topk_idx < num_experts, topk_idx, torch.full_like(topk_idx, -1))
    topk_weights = topk_weights.masked_fill(topk_idx < 0, 0)
    num_pool_rows = int((topk_idx >= 0).sum())

    l1 = torch.randn((num_experts, inter * 2, hidden), dtype=torch.bfloat16, device='cuda') / hidden ** 0.5
    l2 = torch.randn((num_experts, hidden, inter), dtype=torch.bfloat16, device='cuda') / inter ** 0.5

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, args.num_max_tokens_per_rank, num_topk,
        hidden, inter, mma_type='bf16xbf16')

    mm_l1, mm_l2 = deep_gemm.transform_weights_for_mega_moe(l1, l2, 'swiglu', 'bf16xbf16')
    arms = {'megamoe': (mm_l1, mm_l2), 'trtllm': (l1_to_trtllm(mm_l1), l2_to_trtllm(mm_l2))}

    def run(weights, layout: str) -> torch.Tensor:
        buffer.x[:num_tokens].copy_(x)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.bf16_mega_moe(
            y=y, l1_weights=weights[0], l2_weights=weights[1], sym_buffer=buffer,
            activation='swiglu', fast_math=bool(args.fast_math), weight_layout=layout)
        torch.cuda.synchronize()
        return y

    dist_print(f'Config: {num_tokens} tokens, hidden={hidden}, inter={inter}, '
               f'{num_topk}/{num_global_experts} experts, {num_experts} local, '
               f'pool {num_pool_rows} rows', once_in_node=True)

    y_mm = run(arms['megamoe'], 'megamoe')
    y_trt = run(arms['trtllm'], 'trtllm')
    assert y_mm.abs().sum() > 0, 'megamoe arm produced an all-zero output'
    assert torch.equal(y_trt, y_mm), (
        'trtllm weight layout differs from megamoe layout: '
        f'max |delta| {(y_trt.float() - y_mm.float()).abs().max().item():.3e}')
    dist_print(' > trtllm weight layout == megamoe layout, bitwise', once_in_node=True)

    # A missing gate/up swap must change the output.
    bad_map = [s ^ 8 for s in L1_ROW_MAP]
    bad_l1 = l1_to_trtllm(mm_l1, bad_map)
    assert not torch.equal(run((bad_l1, arms['trtllm'][1]), 'trtllm'), y_mm), \
        'a wrong L1 row map also matched -- the bitwise check has no teeth'
    dist_print(' > row-map control: a wrong gate/up half diverges', once_in_node=True)

    if args.bench:
        order = [(name, name, w) for name, w in arms.items()]
        if args.aa:
            # A/A control for allocation-placement effects.
            order.append(('megamoe#2', 'megamoe',
                          (arms['megamoe'][0].clone(), arms['megamoe'][1].clone())))
        for label, name, weights in order:
            for _ in range(5):
                run(weights, name)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            buffer.x[:num_tokens].copy_(x)
            buffer.topk_idx[:num_tokens].copy_(topk_idx)
            buffer.topk_weights[:num_tokens].copy_(topk_weights)
            times = []
            for _ in range(args.bench):
                start.record()
                deep_gemm.bf16_mega_moe(
                    y=y, l1_weights=weights[0], l2_weights=weights[1], sym_buffer=buffer,
                    activation='swiglu', fast_math=bool(args.fast_math), weight_layout=name)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end) * 1e3)
            times.sort()
            dist_print(f' > bench {label:10s}: median {times[len(times) // 2]:.1f} us, '
                       f'min {times[0]:.1f}, p90 {times[int(len(times) * 0.9)]:.1f} (n={len(times)})',
                       once_in_node=True)

    dist_print('OK', once_in_node=True)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192)
    parser.add_argument('--num-tokens', type=int, default=2048)
    parser.add_argument('--hidden', type=int, default=4096)
    parser.add_argument('--intermediate-hidden', type=int, default=4096)
    parser.add_argument('--num-experts', type=int, default=64, help='local experts on this rank')
    parser.add_argument('--num-global-experts', type=int, default=256)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--fast-math', type=int, default=1)
    parser.add_argument('--aa', action='store_true',
                        help='also bench a second canonical clone as an A/A control')
    parser.add_argument('--bench', type=int, default=0,
                        help='time the mega kernel N times per weight layout')
    args = parser.parse_args()
    assert args.num_tokens <= args.num_max_tokens_per_rank
    torch.multiprocessing.spawn(test, args=(1, args), nprocs=1)
    sys.exit(0)
