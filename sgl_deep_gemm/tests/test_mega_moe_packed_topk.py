# FP8xFP4 MegaMoE, one-rank group: opt-in packed top-k input (`topk_format='packed'`, `topk_first_expert`).

import os
import socket
import tempfile

import torch
import torch.distributed as dist

import deep_gemm
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
    scale = deep_gemm.transform_sf_into_required_layout(scale, n, k, (1, 32), num_groups)
    return data, scale


def _run_case(group, num_topk: int, first_expert: int, numerics: dict):
    num_max_tokens = 512
    hidden = 1024
    intermediate_hidden = 512
    num_experts = 16  # of this group
    num_global_experts = 160  # ids in the packed words: this group owns [first_expert, first_expert + num_experts)
    assert first_expert + num_experts <= num_global_experts

    torch.manual_seed(num_topk + first_expert)
    l1_weights = torch.randn(
        (num_experts, 2 * intermediate_hidden, hidden), dtype=torch.bfloat16, device="cuda"
    ) / hidden**0.5
    l2_weights = torch.randn(
        (num_experts, hidden, intermediate_hidden), dtype=torch.bfloat16, device="cuda"
    ) / intermediate_hidden**0.5
    l1, l2 = deep_gemm.transform_weights_for_mega_moe(
        _cast_grouped_weights_to_fp4(l1_weights), _cast_grouped_weights_to_fp4(l2_weights), activation="situ",
    )
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, num_max_tokens, num_topk, hidden, intermediate_hidden, activation="situ",
    )
    stats = torch.zeros((num_experts,), dtype=torch.int, device="cuda")
    words = buffer.topk_idx.view(torch.int32).view(-1)[: num_max_tokens * num_topk].view(num_max_tokens, num_topk)

    def inputs(num_tokens: int, seed: int):
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda", generator=g)
        # global routing: most slots go to other groups, as in an expert-parallel deployment; every token keeps one here
        scores = torch.rand((num_tokens, num_global_experts), device="cuda", generator=g)
        scores[:, first_expert : first_expert + num_experts] += 0.35
        ids = scores.topk(num_topk, dim=-1).indices
        weights = torch.rand((num_tokens, num_topk), device="cuda", generator=g).to(torch.bfloat16)
        if num_tokens > 1 and num_topk > 1:
            ids[1, 0] = -1  # a padded slot
        return per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True), ids, weights

    def stage_x(num_tokens, x_fp8):
        buffer.x[:num_tokens].copy_(x_fp8[0])
        buffer.x_sf[:num_tokens].copy_(x_fp8[1])

    def call(num_tokens: int, y=None, **kwargs):
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") if y is None else y
        deep_gemm.fp8_fp4_mega_moe(
            y, l1, l2, buffer, cumulative_local_expert_recv_stats=stats, activation="situ", **numerics, **kwargs,
        )
        torch.cuda.synchronize()
        assert torch.isfinite(y.float()).all()
        return y

    def run_int64(num_tokens, x_fp8, ids, weights, **kwargs):
        stage_x(num_tokens, x_fp8)
        local = ids - first_expert
        local[(ids < first_expert) | (ids >= first_expert + num_experts)] = -1
        buffer.topk_idx[:num_tokens].copy_(local)
        buffer.topk_weights[:num_tokens].copy_(weights.float())
        return call(num_tokens, **kwargs)

    def stage_packed(num_tokens, x_fp8, ids, weights, fp32_weights: bool):
        stage_x(num_tokens, x_fp8)
        ids = ids.to(torch.int32)
        if fp32_weights:
            buffer.topk_weights[:num_tokens].copy_(weights.float())
            packed = (ids << 16) | (1 << 30)
        else:
            buffer.topk_weights[:num_tokens].fill_(float("nan"))  # not read
            packed = (ids << 16) | weights.contiguous().view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)
        packed[ids < 0] = -1
        words[:num_tokens].copy_(packed)

    packed_kwargs = dict(topk_format="packed", topk_first_expert=first_expert)
    for num_tokens in (1, 2, 16, 84, 85, 128, num_max_tokens):
        x_fp8, ids, weights = inputs(num_tokens, 1000 + num_tokens)
        ref = run_int64(num_tokens, x_fp8, ids, weights)
        assert ref.float().abs().sum() > 0
        assert torch.equal(ref, run_int64(num_tokens, x_fp8, ids, weights, topk_format="int64", topk_first_expert=0)), \
            "the default format changed"
        for fp32_weights in (False, True):
            stage_packed(num_tokens, x_fp8, ids, weights, fp32_weights)
            assert torch.equal(ref, call(num_tokens, **packed_kwargs)), \
                f"packed != int64: {num_tokens} tokens, top-k {num_topk}, first expert {first_expert}, fp32 weights {fp32_weights}"

    # A full int64 call overwrites every word of `topk_idx`; the packed format keeps nothing there between calls
    small = inputs(16, 7)
    ref = run_int64(16, *small)
    run_int64(num_max_tokens, *inputs(num_max_tokens, 8))
    stage_packed(16, *small, False)
    assert torch.equal(ref, call(16, **packed_kwargs)), f"packed call after a full int64 call, first expert {first_expert}"

    # CUDA graph: the captured packed call replayed with other inputs
    num_tokens = 16
    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    stage_packed(num_tokens, *small, False)
    call(num_tokens, y, **packed_kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deep_gemm.fp8_fp4_mega_moe(
            y, l1, l2, buffer, cumulative_local_expert_recv_stats=stats, activation="situ", **numerics, **packed_kwargs,
        )
    for seed in (21, 22):
        other = inputs(num_tokens, seed)
        ref = run_int64(num_tokens, *other).clone()
        stage_packed(num_tokens, *other, False)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(ref, y), f"graph replay of the packed call, first expert {first_expert}"

    for bad_kwargs in (dict(topk_first_expert=16), dict(topk_format="packed", topk_first_expert=-1),
                       dict(topk_format="packed", topk_first_expert=16384), dict(topk_format="words")):
        try:
            deep_gemm.fp8_fp4_mega_moe(y, l1, l2, buffer, activation="situ", **bad_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad_kwargs} must be rejected")

    buffer.destroy()


def test(local_rank: int, num_local_ranks: int):
    jit_cache = tempfile.TemporaryDirectory(prefix="deep-gemm-packed-topk-")
    os.environ["DG_JIT_CACHE_DIR"] = jit_cache.name
    if "MASTER_PORT" not in os.environ:
        # One process: any free port, so that the test can run next to other DeepGEMM tests on the same host
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            os.environ["MASTER_PORT"] = str(s.getsockname()[1])
    _, _, group = init_dist(local_rank, num_local_ranks)
    os.environ["DG_COMM_KERNEL_DEBUG"] = "0"

    for numerics in (dict(), dict(numerics="trtllm"), dict(numerics="trtllm", split4_combine_max_tokens=84)):
        for num_topk in (1, 8):
            for first_expert in (0, 16, 144):
                _run_case(group, num_topk, first_expert, numerics)
        dist_print(f"packed top-k == int64 top-k ({numerics or 'default numerics'})", once_in_node=True)
    dist.destroy_process_group()
    jit_cache.cleanup()


if __name__ == "__main__":
    torch.multiprocessing.spawn(test, args=(1,), nprocs=1)
