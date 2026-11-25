# FP8xFP4 MegaMoE: which calls may use the row-shuffled (TRT) weight layout.

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
    jit_cache = tempfile.TemporaryDirectory(prefix="deep-gemm-row-shuffled-gates-")
    os.environ["DG_JIT_CACHE_DIR"] = jit_cache.name
    _, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_ranks == 1
    os.environ["DG_COMM_KERNEL_DEBUG"] = "0"

    weights = _weights()
    buffer = deep_gemm.mega.SymmBuffer(group, NUM_EXPERTS, 256, NUM_TOPK, HIDDEN, INTERMEDIATE_HIDDEN, activation="situ")
    expected = _run(buffer, weights, 32)
    y = torch.empty((32, HIDDEN), dtype=torch.bfloat16, device="cuda")

    for layout in ("row_shuffled", "trtllm"):
        try:
            deep_gemm.fp8_fp4_mega_moe(y, weights[0], weights[1], buffer, activation="swiglu", weight_layout=layout)
        except (ValueError, AssertionError):
            pass
        else:
            raise AssertionError(f"{layout} without SiTU must be rejected")
        try:
            deep_gemm.fp8_fp4_mega_moe(y, weights[0], weights[1], buffer, activation="situ", weight_layout=layout,
                                       shared_l1_weights=weights[0], shared_l2_weights=weights[1])
        except (ValueError, AssertionError):
            pass
        else:
            raise AssertionError(f"{layout} with shared experts must be rejected")

    # The default layout is untouched by the rejected calls
    assert torch.equal(_run(buffer, weights, 32), expected)
    buffer.destroy()
    dist_print("row-shuffled FP8xFP4 gates: rejections hold, default layout unchanged", once_in_node=True)
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
