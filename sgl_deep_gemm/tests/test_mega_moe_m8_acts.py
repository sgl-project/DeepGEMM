# FP8xFP4 SiTU MegaMoE: 8-row activation tiles for 16-row MMA blocks.
#
# Covers:
#   experts that receive 9 .. 16 tokens (one 16-row MMA block, loaded as two 8-row tiles) give the same rows as
#   the same tokens in calls where their experts receive more or fewer tokens
#
# Checks, per shape and call size (there is no second code path to compare with, so the checks are invariants):
#   * every local expert receives exactly the tokens routed to it (`cumulative_local_expert_recv_stats`);
#   * the same call twice is bitwise equal (nothing is left behind in the workspace);
#   * a row's result does not depend on the other rows: a call on a prefix of the tokens and a call on a
#     permutation of them reproduce the rows bitwise;
#   * a call that routes nothing leaves the next call unchanged;
#   * a captured CUDA graph replays correctly with other inputs.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 python3 sgl_deep_gemm/tests/test_mega_moe_m8_acts.py

import argparse
import os
import socket
import tempfile

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist

# (experts of the group, top-k, routing pattern)
CASES = [(2, 1, 'uniform'), (4, 2, 'uniform'), (16, 8, 'uniform'), (1, 1, 'uniform')]
SIZES = (1, 9, 12, 16, 17, 24, 31, 32, 40, 64, 512)
NUM_MAX_TOKENS = 512
HIDDEN = 1024
INTERMEDIATE_HIDDEN = 512


def _cast_grouped_weights_to_fp4(weights: torch.Tensor):
    num_groups, n, k = weights.shape
    data = torch.empty((num_groups, n, k // 2), dtype=torch.int8, device="cuda")
    scale = torch.empty((num_groups, n, k // 32), dtype=torch.float, device="cuda")
    for group_idx in range(num_groups):
        data[group_idx], scale[group_idx] = per_token_cast_to_fp4(
            weights[group_idx], use_ue8m0=True, gran_k=32
        )
    scale = deep_gemm.transform_sf_into_required_layout(scale, n, k, (1, 32), num_groups)
    return data, scale


def _route(pattern: str, num_tokens: int, num_experts: int, num_topk: int, generator):
    scores = torch.rand((num_tokens, num_experts), device="cuda", generator=generator)
    if pattern == "high":
        # the last experts take almost everything
        scores[:, -max(num_topk, num_experts // 8):] += 1.0
    elif pattern == "sparse":
        # every fourth expert is loaded, the others stay empty
        scores[:, torch.arange(num_experts, device="cuda") % 4 != 3] -= 1.0
    return scores.topk(num_topk, dim=-1).indices


def _run_case(group, rank_idx: int, num_experts: int, num_topk: int, pattern: str):
    num_ranks = group.size()
    num_local_experts = num_experts // num_ranks
    first_expert = rank_idx * num_local_experts
    torch.manual_seed(1 + rank_idx + num_experts + num_topk)
    l1_weights = torch.randn(
        (num_local_experts, 2 * INTERMEDIATE_HIDDEN, HIDDEN), dtype=torch.bfloat16, device="cuda"
    ) / HIDDEN**0.5
    l2_weights = torch.randn(
        (num_local_experts, HIDDEN, INTERMEDIATE_HIDDEN), dtype=torch.bfloat16, device="cuda"
    ) / INTERMEDIATE_HIDDEN**0.5
    l1, l2 = deep_gemm.transform_weights_for_mega_moe(
        _cast_grouped_weights_to_fp4(l1_weights), _cast_grouped_weights_to_fp4(l2_weights), activation="situ",
    )
    del l1_weights, l2_weights
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, NUM_MAX_TOKENS, num_topk, HIDDEN, INTERMEDIATE_HIDDEN, activation="situ",
    )
    stats = torch.zeros((num_local_experts,), dtype=torch.int, device="cuda")
    tag = f"{num_ranks} rank(s), {num_experts} experts, top-{num_topk}, {pattern}"

    def inputs(num_tokens: int, seed: int):
        # NOTES: every rank draws the same routing, so that the ranks' expert counts agree
        g = torch.Generator(device="cuda").manual_seed(seed)
        ids = _route(pattern, num_tokens, num_experts, num_topk, g)
        weights = torch.rand((num_tokens, num_topk), device="cuda", generator=g)
        g = torch.Generator(device="cuda").manual_seed(seed * 131 + rank_idx)
        x = torch.randn((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda", generator=g)
        return x, ids, weights

    def stage(num_tokens, x, ids, weights):
        if num_tokens > 0:
            x_fp8 = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            buffer.x[:num_tokens].copy_(x_fp8[0])
            buffer.x_sf[:num_tokens].copy_(x_fp8[1])
        buffer.topk_idx[:num_tokens].copy_(ids)
        buffer.topk_weights[:num_tokens].copy_(weights)

    def call(num_tokens, y=None):
        y = torch.empty((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda") if y is None else y
        deep_gemm.fp8_fp4_mega_moe(
            y, l1, l2, buffer, cumulative_local_expert_recv_stats=stats, activation="situ",
        )
        torch.cuda.synchronize()
        return y

    def run(num_tokens, x, ids, weights, check_stats: bool = True):
        stage(num_tokens, x, ids, weights)
        stats.zero_()
        y = call(num_tokens)
        assert torch.isfinite(y.float()).all(), tag
        if check_stats and num_ranks == 1:
            # NOTES: with several ranks an expert's count sums what every rank routed to it; one rank sees its own ids
            local = ids[(ids >= first_expert) & (ids < first_expert + num_local_experts)] - first_expert
            expected = torch.bincount(local.flatten(), minlength=num_local_experts).to(stats.dtype)
            assert torch.equal(stats, expected), f"expert receive counts differ: {num_tokens} tokens, {tag}"
        return y

    def barrier():
        # Ranks of one group must issue the same sequence of calls
        if num_ranks > 1:
            dist.barrier(group)

    for num_tokens in SIZES:
        x, ids, weights = inputs(num_tokens, 1000 + num_tokens)
        ref = run(num_tokens, x, ids, weights)
        assert torch.equal(ref, run(num_tokens, x, ids, weights)), f"not reproducible: {num_tokens} tokens, {tag}"
        if num_tokens > 1:
            perm = torch.randperm(num_tokens, device="cuda", generator=torch.Generator(device="cuda").manual_seed(num_tokens))
            permuted = run(num_tokens, x[perm], ids[perm], weights[perm])
            assert torch.equal(ref[perm], permuted), f"rows depend on their position: {num_tokens} tokens, {tag}"
            half = num_tokens // 2
            prefix = run(half, x[:half], ids[:half], weights[:half])
            assert torch.equal(ref[:half], prefix), f"rows depend on the call size: {num_tokens} vs {half} tokens, {tag}"
        barrier()

    # A call that routes nothing, between two equal calls
    small = inputs(16, 7)
    ref = run(16, *small)
    nothing = torch.full((16, num_topk), -1, dtype=torch.int64, device="cuda")
    run(16, small[0], nothing, small[2], check_stats=False)
    assert int(stats.sum()) == 0 or num_ranks > 1, f"a call that routes nothing received tokens, {tag}"
    assert torch.equal(ref, run(16, *small)), f"a call that routes nothing changed the next call, {tag}"
    # A full buffer, then the small call again
    run(NUM_MAX_TOKENS, *inputs(NUM_MAX_TOKENS, 8))
    assert torch.equal(ref, run(16, *small)), f"a full call changed the next call, {tag}"
    barrier()

    # CUDA graph: the captured call replayed with other inputs
    num_tokens = 16
    y = torch.empty((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
    stage(num_tokens, *small)
    call(num_tokens, y)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deep_gemm.fp8_fp4_mega_moe(
            y, l1, l2, buffer, cumulative_local_expert_recv_stats=stats, activation="situ",
        )
    for seed in (21, 22):
        other = inputs(num_tokens, seed)
        ref = run(num_tokens, *other).clone()
        stage(num_tokens, *other)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(ref, y), f"graph replay differs, {tag}"
        barrier()

    buffer.destroy()


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    jit_cache = tempfile.TemporaryDirectory(prefix="deep-gemm-m8-acts-")
    os.environ["DG_JIT_CACHE_DIR"] = jit_cache.name
    if "MASTER_PORT" not in os.environ and num_local_ranks == 1:
        # One process: any free port, so that the test can run next to other DeepGEMM tests on the same host
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            os.environ["MASTER_PORT"] = str(s.getsockname()[1])
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    os.environ["DG_COMM_KERNEL_DEBUG"] = "0"

    for num_experts, num_topk, pattern in CASES:
        if num_experts % num_ranks == 0:
            _run_case(group, rank_idx, num_experts, num_topk, pattern)
    dist_print("8-row activation tiles for 16-row MMA blocks: invariants hold", once_in_node=True)
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
