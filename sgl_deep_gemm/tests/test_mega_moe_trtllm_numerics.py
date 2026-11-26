# FP8xFP4 SiTU MegaMoE: opt-in trtllm-gen numerics (`numerics='trtllm'`, `split4_combine_max_tokens`).

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


def _run_case(group, num_topk: int):
    num_max_tokens_per_rank = 8192
    hidden = 1024
    intermediate_hidden = 512
    num_experts = 16
    split4_limit = 84

    torch.manual_seed(num_topk)
    # The buffer takes the group's expert count, every rank holds its own share of the weights
    num_experts_per_rank = num_experts // group.size()
    l1_weights = torch.randn(
        (num_experts_per_rank, 2 * intermediate_hidden, hidden), dtype=torch.bfloat16, device="cuda"
    ) / hidden**0.5
    l2_weights = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device="cuda"
    ) / intermediate_hidden**0.5
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
        _cast_grouped_weights_to_fp4(l1_weights),
        _cast_grouped_weights_to_fp4(l2_weights),
        activation="situ",
    )
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
        activation="situ",
    )
    cumulative_recv_stats = torch.zeros((num_experts_per_rank,), dtype=torch.int, device="cuda")

    def run(num_tokens: int, **kwargs):
        torch.manual_seed(1000 + num_tokens)
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device="cuda")
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        topk_weights = torch.softmax(topk_weights, dim=-1)
        x_fp8 = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        buffer.x[:num_tokens].copy_(x_fp8[0])
        buffer.x_sf[:num_tokens].copy_(x_fp8[1])
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
        cumulative_recv_stats.zero_()
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        deep_gemm.fp8_fp4_mega_moe(
            y, transformed_l1, transformed_l2, buffer,
            cumulative_local_expert_recv_stats=cumulative_recv_stats,
            activation="situ", **kwargs,
        )
        torch.cuda.synchronize()
        assert torch.isfinite(y).all()
        return y

    for num_tokens in (1, 32, split4_limit, split4_limit + 1, 128):
        default = run(num_tokens)
        assert torch.equal(default, run(num_tokens)), "default path is not deterministic"
        assert torch.equal(default, run(num_tokens, numerics="default", split4_combine_max_tokens=0))

        chain = run(num_tokens, numerics="trtllm")
        assert torch.equal(chain, run(num_tokens, numerics="trtllm")), "trtllm numerics are not deterministic"
        diff = calc_diff(chain, default)
        assert diff < 5e-3, f"trtllm numerics diverged from the default: {diff} at {num_tokens} tokens"

        split4 = run(num_tokens, numerics="trtllm", split4_combine_max_tokens=split4_limit)
        if num_tokens > split4_limit or num_topk == 1:
            # Above the limit the chain runs; with one slot both orders are one fma and one rounding
            assert torch.equal(split4, chain), f"split-4 combine differs at {num_tokens} tokens, top-k {num_topk}"
        else:
            diff = calc_diff(split4, chain)
            assert diff < 1e-6, f"split-4 combine diverged from the chain: {diff} at {num_tokens} tokens"

    for bad_kwargs in (dict(split4_combine_max_tokens=84), dict(numerics="trtllm", activation="swiglu")):
        try:
            kwargs = dict(activation="situ")
            kwargs.update(bad_kwargs)
            y = torch.empty((1, hidden), dtype=torch.bfloat16, device="cuda")
            deep_gemm.fp8_fp4_mega_moe(y, transformed_l1, transformed_l2, buffer, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad_kwargs} must be rejected")

    buffer.destroy()


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    jit_cache = tempfile.TemporaryDirectory(prefix="deep-gemm-trtllm-numerics-")
    os.environ["DG_JIT_CACHE_DIR"] = jit_cache.name
    _, _, group = init_dist(local_rank, num_local_ranks)
    os.environ["DG_COMM_KERNEL_DEBUG"] = "0"

    for num_topk in (1, 4, 8):
        _run_case(group, num_topk)

    dist_print("trtllm numerics: default unchanged, chain / split-4 orders consistent", once_in_node=True)
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
