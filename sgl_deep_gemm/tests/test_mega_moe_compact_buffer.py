# FP8xFP4 SiTU MegaMoE: symmetric buffer with an exact row count and a largest M tile.
#
# Covers `SymmBuffer(exact_num_max_tokens=True, max_block_m=N)`:
#   * the buffer keeps the given row count and is smaller than the default one;
#   * outputs are bitwise equal to the default buffer's at every call size whose tile fits;
#   * a call that would pick a larger tile, a row count that is not a multiple of 128 and a `max_block_m`
#     that is not a candidate tile are rejected.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 python3 sgl_deep_gemm/tests/test_mega_moe_compact_buffer.py

import argparse
import os
import tempfile

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist


def _cast_grouped_weights_to_fp4(weights: torch.Tensor):
    num_groups, n, k = weights.shape
    data = torch.empty((num_groups, n, k // 2), dtype=torch.int8, device="cuda")
    scale = torch.empty((num_groups, n, k // 32), dtype=torch.float, device="cuda")
    for group_idx in range(num_groups):
        data[group_idx], scale[group_idx] = per_token_cast_to_fp4(
            weights[group_idx], use_ue8m0=True, gran_k=32
        )
    scale = deep_gemm.transform_sf_into_required_layout(
        scale, n, k, (1, 32), num_groups
    )
    return data, scale



HIDDEN = 1024
INTERMEDIATE_HIDDEN = 512
NUM_EXPERTS = 16
NUM_TOPK = 4


def _weights():
    torch.manual_seed(0)
    l1_weights = torch.randn((NUM_EXPERTS, 2 * INTERMEDIATE_HIDDEN, HIDDEN), dtype=torch.bfloat16, device="cuda") / HIDDEN**0.5
    l2_weights = torch.randn((NUM_EXPERTS, HIDDEN, INTERMEDIATE_HIDDEN), dtype=torch.bfloat16, device="cuda") / INTERMEDIATE_HIDDEN**0.5
    return deep_gemm.transform_weights_for_mega_moe(
        _cast_grouped_weights_to_fp4(l1_weights), _cast_grouped_weights_to_fp4(l2_weights), activation="situ")


def _run(buffer, weights, num_tokens: int, **kwargs):
    torch.manual_seed(1000 + num_tokens)
    x = torch.randn((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
    scores = torch.randn((num_tokens, NUM_EXPERTS), dtype=torch.float, device="cuda")
    topk_weights, topk_idx = torch.topk(scores, NUM_TOPK, dim=-1, largest=True, sorted=False)
    topk_weights = torch.softmax(topk_weights, dim=-1)
    x_fp8 = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    buffer.x[:num_tokens].copy_(x_fp8[0])
    buffer.x_sf[:num_tokens].copy_(x_fp8[1])
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_weights)
    y = torch.empty((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
    deep_gemm.fp8_fp4_mega_moe(y, weights[0], weights[1], buffer, activation="situ", **kwargs)
    torch.cuda.synchronize()
    assert torch.isfinite(y).all()
    return y


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    jit_cache = tempfile.TemporaryDirectory(prefix="deep-gemm-compact-buffer-")
    os.environ["DG_JIT_CACHE_DIR"] = jit_cache.name
    _, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_ranks == 1, "the compact buffer is tested on a one-rank group"
    os.environ["DG_COMM_KERNEL_DEBUG"] = "0"

    weights = _weights()
    layout = (group, NUM_EXPERTS, 256, NUM_TOPK, HIDDEN, INTERMEDIATE_HIDDEN)
    default = deep_gemm.mega.SymmBuffer(*layout, activation="situ")
    compact = deep_gemm.mega.SymmBuffer(*layout, activation="situ", exact_num_max_tokens=True, max_block_m=64)
    assert default.num_max_tokens_per_rank > 256, "the default buffer aligns its row count up"
    assert compact.num_max_tokens_per_rank == 256 and compact.x.shape[0] == 256
    assert compact.buffer.nbytes < default.buffer.nbytes, (compact.buffer.nbytes, default.buffer.nbytes)

    # 192 tokens: 48 expected tokens per expert, the largest load that still takes a 64-row tile
    for num_tokens in (1, 16, 64, 192):
        expected = _run(default, weights, num_tokens)
        assert torch.equal(_run(compact, weights, num_tokens), expected), f"compact buffer differs at {num_tokens} tokens"
        assert torch.equal(_run(compact, weights, num_tokens), expected), f"compact buffer is not repeatable at {num_tokens} tokens"

    # 256 tokens pick a 128-row tile: fine on the default buffer, rejected on the tile-limited one
    _run(default, weights, 256)
    try:
        _run(compact, weights, 256)
    except Exception:
        pass
    else:
        raise AssertionError("a call that picks a tile above `max_block_m` must be rejected")
    assert torch.equal(_run(compact, weights, 64), _run(default, weights, 64)), "the buffer is unusable after a rejected call"

    for bad_kwargs in (dict(exact_num_max_tokens=True), dict(max_block_m=48)):
        bad_layout = (group, NUM_EXPERTS, 200 if "exact_num_max_tokens" in bad_kwargs else 256, NUM_TOPK, HIDDEN, INTERMEDIATE_HIDDEN)
        try:
            deep_gemm.mega.SymmBuffer(*bad_layout, activation="situ", **bad_kwargs)
        except Exception:
            pass
        else:
            raise AssertionError(f"{bad_kwargs} must be rejected")

    default.destroy()
    compact.destroy()
    dist_print("compact buffer: sizes, bitwise equality with the default buffer, rejections", once_in_node=True)
    dist.destroy_process_group()
    jit_cache.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=1)
    parsed_args = parser.parse_args()
    torch.multiprocessing.spawn(
        test,
        args=(parsed_args.num_processes, parsed_args),
        nprocs=parsed_args.num_processes,
    )
