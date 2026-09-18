# NVFP4 mega-MoE with TRT-LLM's weight storage: bitwise check and per-layout bench.
#
# MegaMoE aliases TRT-LLM's final weight buffers, so sglang can hand the kernel
# TRT-packed experts with `weight_layout="trtllm"` instead of repacking them.
# TRT's L1 shuffle swaps the two 8-row halves of every 16-row group (`row ^ 8`);
# TRT's L2 shuffle transposes each 32-row group 8x4 -> 4x8. L2 is undone by the
# load's tensor map and L1 by the SwiGLU epilogue reading the swapped register
# slots, so the two layouts must agree BITWISE -- same values, same MMA lanes,
# same accumulation order, only the load path differs.
#
# The weight SFs come from TRT's own storage too ([row block][k chunk][128
# words]) rather than a MegaMoE-layout copy, which is what lets a caller alias
# TRT's scale tensors instead of keeping a second set.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29532 \
#       python3 sgl_deep_gemm/tests/test_mega_moe_nvfp4_trtllm.py --bench 60

import argparse
import sys
import torch
import torch.distributed as dist
from typing import Tuple

import deep_gemm
from deep_gemm.utils import per_token_cast_to_nvfp4, transform_ue4m3_sf_into_required_layout
from deep_gemm.utils.dist import dist_print, init_dist

GRAN_K = 16


def cast_weights(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_groups, n, k = w.shape
    packed = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
    sf = torch.empty((num_groups, n, k // GRAN_K), device='cuda', dtype=torch.float)
    for i in range(num_groups):
        packed[i], sf[i] = per_token_cast_to_nvfp4(w[i], gran_k=GRAN_K)
    return packed, transform_ue4m3_sf_into_required_layout(sf, n)


def l1_to_trtllm(w: torch.Tensor) -> torch.Tensor:
    """Swap the two 8-row halves of every 16-row L1 group (`row ^ 8`)."""
    e, n, k = w.shape
    return w.view(e, n // 16, 2, 8, k).flip(2).reshape(e, n, k).contiguous()


def l2_to_trtllm(w: torch.Tensor) -> torch.Tensor:
    """TRT-LLM transposes each 32-row L2 group 8x4 -> 4x8."""
    e, n, k = w.shape
    return w.view(e, n // 32, 8, 4, k).transpose(2, 3).reshape(e, n, k).contiguous()


def swap_l1_sf_halves(sf: torch.Tensor) -> torch.Tensor:
    """TRT's L1 scales follow its weight rows through `reorder_w1w3_to_w3w1`.

    That is the same `row ^ 8` as the weights, which the UTCCP 4x32 transpose
    has already turned into `p ^ 32` on these packed SF words.
    """
    e, mn, k = sf.shape
    return torch.empty_like(sf).copy_(sf.reshape(e, mn // 64, 2, 32, k).flip(2).reshape(e, mn, k))


def sf_to_trtllm(sf: torch.Tensor, is_l1: bool) -> torch.Tensor:
    """MegaMoE SF words -> TRT-LLM's [row block][k chunk][128 words].

    Derived from `make_trtllm_fp4_sf_tma_desc` rather than restated: for a box
    coordinate `(i0, i1, i2)` the descriptor reads gmem word `i0 + 32*i1 + 4*i2`
    (strides 128B/16B over 4B words) and writes SMEM slot `i0 + 4*i1 + 16*i2`
    (box extents 4, 4, 8, innermost fastest). L1's box is the flat 128, so its
    slot map is the identity and only the outer axes swap.
    """
    e, mn, k_words = sf.shape
    src = sf.reshape(e, mn // 128, 128, k_words)
    if is_l1:
        gmem_of_slot = torch.arange(128, device=sf.device)
    else:
        s_idx = torch.arange(128, device=sf.device)
        gmem_of_slot = (s_idx % 4) + 32 * ((s_idx // 4) % 4) + 4 * (s_idx // 16)
    out = torch.empty((e, mn // 128, k_words, 128), dtype=sf.dtype, device=sf.device)
    # SMEM slot `s` must receive MegaMoE's row `s`, so it lands at gmem `f(s)`.
    out[:, :, :, gmem_of_slot] = src.permute(0, 1, 3, 2)
    # The kernel reads this through a tensor map, so only the byte order matters;
    # the (experts, mn, k words) shape is kept for the host's SF shape check.
    return out.reshape(e, mn, k_words)


# noinspection PyUnboundLocalVariable
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_ranks == 1, 'one rank stands in for one EP rank of a larger MoE'
    torch.manual_seed(0)

    num_tokens, hidden, inter = args.num_tokens, args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_global_experts = max(args.num_global_experts, num_experts)

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x_packed, x_sf = per_token_cast_to_nvfp4(x, gran_k=GRAN_K, use_packed_ue4m3=True)
    scores = torch.randn((num_tokens, num_global_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    # Keep this rank's expert slice, mask the rest: the pool is then the ~2T rows
    # an EP rank of `num_global_experts` sees for a T-token chunk.
    topk_idx = torch.where(topk_idx < num_experts, topk_idx, torch.full_like(topk_idx, -1)).long()
    topk_weights = topk_weights.masked_fill(topk_idx < 0, 0)
    num_pool_rows = int((topk_idx >= 0).sum())

    l1 = torch.randn((num_experts, inter * 2, hidden), dtype=torch.bfloat16, device='cuda')
    l2 = torch.randn((num_experts, hidden, inter), dtype=torch.bfloat16, device='cuda')
    l1_packed, l1_raw_sf = cast_weights(l1)
    l2_packed, l2_raw_sf = cast_weights(l2)

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, args.num_max_tokens_per_rank, num_topk,
        hidden, inter, mma_type='nvfp4xnvfp4')

    (mm_l1, mm_l1_sf), (mm_l2, mm_l2_sf) = deep_gemm.transform_weights_for_mega_moe(
        (l1_packed, l1_raw_sf), (l2_packed, l2_raw_sf), 'swiglu', 'nvfp4xnvfp4')
    # Store the same weights and scales in TRT order, including L1's row swap.
    trt_l1, trt_l2 = l1_to_trtllm(mm_l1), l2_to_trtllm(mm_l2)
    trt_l1_sf = sf_to_trtllm(swap_l1_sf_halves(mm_l1_sf), True)
    trt_l2_sf = sf_to_trtllm(mm_l2_sf, False)
    arms = {
        'megamoe': ((mm_l1, mm_l1_sf), (mm_l2, mm_l2_sf)),
        'trtllm': ((trt_l1, trt_l1_sf), (trt_l2, trt_l2_sf)),
    }

    def run(weights, layout: str) -> torch.Tensor:
        buffer.x[:num_tokens].copy_(x_packed)
        buffer.x_sf[:num_tokens].copy_(x_sf)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.fp8_fp4_mega_moe(
            y=y, l1_weights=weights[0], l2_weights=weights[1], sym_buffer=buffer,
            recipe=(1, 1, GRAN_K), activation='swiglu',
            fast_math=bool(args.fast_math), weight_layout=layout)
        torch.cuda.synchronize()
        return y

    dist_print(f'Config: {num_tokens} tokens, hidden={hidden}, inter={inter}, '
               f'{num_topk}/{num_global_experts} experts, {num_experts} local, '
               f'pool {num_pool_rows} rows', once_in_node=True)

    # Dense tensor maps must reject scale views whose strides they cannot honor.
    for layer, sf in enumerate((trt_l1_sf, trt_l2_sf)):
        strided_sf = sf[..., :1].contiguous().expand_as(sf)
        assert not strided_sf.is_contiguous()
        weights = list(arms['trtllm'])
        weights[layer] = (weights[layer][0], strided_sf)
        try:
            run(weights, 'trtllm')
        except RuntimeError as error:
            assert 'is_contiguous' in str(error), str(error)
        else:
            raise AssertionError(f'L{layer + 1} accepted non-contiguous TRT scales')

    y_mm = run(arms['megamoe'], 'megamoe')
    y_trt = run(arms['trtllm'], 'trtllm')
    assert y_mm.abs().sum() > 0, 'megamoe arm produced an all-zero output'
    assert torch.equal(y_trt, y_mm), (
        'trtllm layout differs from megamoe layout: '
        f'max |delta| {(y_trt.float() - y_mm.float()).abs().max().item():.3e}')
    dist_print(' > trtllm weights + scales == megamoe layout, bitwise', once_in_node=True)

    # Two controls, one per descriptor family. A kernel that ignored either
    # shuffle would still match above; each of these must diverge.
    assert not torch.equal(run(((mm_l1, trt_l1_sf), arms['trtllm'][1]), 'trtllm'), y_mm), \
        'unshuffled L1 weights also matched -- the weight check has no teeth'
    dist_print(' > weight control: unshuffled L1 diverges', once_in_node=True)
    assert not torch.equal(run(((trt_l1, mm_l1_sf.contiguous()), (trt_l2, mm_l2_sf.contiguous())), 'trtllm'), y_mm), \
        'incorrect SF packing also matched -- the SF check has no teeth'
    dist_print(' > sf control: incorrect SF packing read as TRT diverges', once_in_node=True)

    if args.bench:
        order = [(name, name, w) for name, w in arms.items()]
        if args.aa:
            # A/A control: a second canonical clone, allocated after the first,
            # carries whatever placement cost the second arm of any A/B pays.
            order.append(('megamoe#2', 'megamoe',
                          ((mm_l1.clone(), mm_l1_sf.clone()), (mm_l2.clone(), mm_l2_sf.clone()))))
        for label, name, weights in order:
            for _ in range(5):
                run(weights, name)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            buffer.x[:num_tokens].copy_(x_packed)
            buffer.x_sf[:num_tokens].copy_(x_sf)
            buffer.topk_idx[:num_tokens].copy_(topk_idx)
            buffer.topk_weights[:num_tokens].copy_(topk_weights)
            times = []
            for _ in range(args.bench):
                start.record()
                deep_gemm.fp8_fp4_mega_moe(
                    y=y, l1_weights=weights[0], l2_weights=weights[1], sym_buffer=buffer,
                    recipe=(1, 1, GRAN_K), activation='swiglu',
                    fast_math=bool(args.fast_math), weight_layout=name)
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
