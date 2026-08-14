import argparse
import os
import re
import tempfile
from pathlib import Path

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
    scale = deep_gemm.transform_sf_into_required_layout(
        scale, n, k, (1, 32), num_groups
    )
    return data, scale


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    jit_cache = tempfile.TemporaryDirectory(prefix="deep-gemm-situ-")
    os.environ["DG_JIT_CACHE_DIR"] = jit_cache.name
    _, _, group = init_dist(local_rank, num_local_ranks)
    os.environ["DG_COMM_KERNEL_DEBUG"] = "0"

    num_tokens = 32
    num_max_tokens_per_rank = 8192
    hidden = 1024
    intermediate_hidden = 512
    num_experts = 8
    num_topk = 2

    torch.manual_seed(0)
    x = torch.full((num_tokens, hidden), 0.25, dtype=torch.bfloat16, device="cuda")
    l1_weights = torch.full(
        (num_experts, 2 * intermediate_hidden, hidden),
        0.25,
        dtype=torch.bfloat16,
        device="cuda",
    )
    l2_weights = torch.full(
        (num_experts, hidden, intermediate_hidden),
        0.25,
        dtype=torch.bfloat16,
        device="cuda",
    )
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device="cuda")
    topk_weights, topk_idx = torch.topk(
        scores, num_topk, dim=-1, largest=True, sorted=False
    )

    x_fp8 = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
        _cast_grouped_weights_to_fp4(l1_weights),
        _cast_grouped_weights_to_fp4(l2_weights),
        activation="situ",
    )
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        activation="situ",
    )
    cumulative_recv_stats = torch.zeros((num_experts,), dtype=torch.int, device="cuda")

    def run(activation: str, activation_clamp=None):
        buffer.x[:num_tokens].copy_(x_fp8[0])
        buffer.x_sf[:num_tokens].copy_(x_fp8[1])
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
        cumulative_recv_stats.zero_()
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        deep_gemm.fp8_fp4_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            buffer,
            cumulative_local_expert_recv_stats=cumulative_recv_stats,
            activation=activation,
            activation_clamp=activation_clamp,
            fast_math=True,
        )
        torch.cuda.synchronize()
        return y

    situ = run("situ")
    clamped_swiglu = run("swiglu", activation_clamp=0.03125)

    assert torch.isfinite(situ).all()
    assert torch.isfinite(clamped_swiglu).all()

    kernel_sources = [
        path.read_text()
        for path in Path(jit_cache.name).glob(
            "cache/kernel.sm100_fp8_fp4_mega_moe.*/kernel.cu"
        )
    ]
    assert len(kernel_sources) == 2
    assert any(
        re.search(
            r"cute::numeric_limits<float>::infinity\(\),\s+true,\s+true,",
            source,
        )
        for source in kernel_sources
    ), "explicit SiTU did not instantiate kUseSitu=true"
    assert any(
        re.search(r"0x1p-5f,\s+false,\s+true,", source) for source in kernel_sources
    ), "activation_clamp=0.03125 still instantiated kUseSitu=true"

    try:
        run("situ", activation_clamp=0.03125)
    except AssertionError as error:
        assert "activation_clamp" in str(error)
    else:
        raise AssertionError("SiTU must reject the unrelated activation_clamp option")

    dist_print(
        "Explicit SiTU and tightly clamped SwiGLU both launched", once_in_node=True
    )
    buffer.destroy()
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
