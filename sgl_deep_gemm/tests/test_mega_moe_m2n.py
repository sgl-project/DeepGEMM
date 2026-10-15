"""M2N correctness, without benchmarking.

Single GPU: python test_mega_moe_m2n.py
AG2/EG2: torchrun --standalone --nproc_per_node=4 test_mega_moe_m2n.py
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist

if "LOCAL_RANK" in os.environ:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

import deep_gemm
from deep_gemm import _C
from deep_gemm.mega.m2n import (
    get_symm_buffer_for_mega_moe_m2n,
    make_m2n_weight_descs,
    mega_moe_m2n_ag,
    mega_moe_m2n_eg,
)
from deep_gemm.utils import (
    cast_back_from_fp4,
    per_token_cast_to_fp4,
    per_token_cast_to_fp8,
    transform_ue4m3_sf_into_required_layout,
)

HIDDEN, INTERMEDIATE, EXPERTS, TOPK, CAPACITY = 512, 512, 8, 2, 129
CLAMP = 1.25
OUTPUT_SCALES = (1.0, 2.5)


def test_layout(capacity: int = 132, use_nvfp4: bool = False) -> None:
    ag_size, eg_size = 2, 2
    num_experts, num_topk = 8, 2
    hidden, intermediate_hidden = HIDDEN, INTERMEDIATE
    num_bytes, pool_tokens, slice_buffers = _C.get_symm_buffer_size_for_mega_moe_m2n(
        ag_size,
        eg_size,
        num_experts,
        capacity,
        num_topk,
        hidden,
        intermediate_hidden,
        use_nvfp4,
    )
    buffer = torch.zeros(int(num_bytes), dtype=torch.int8, device="cuda")
    _C.initialize_symm_buffer_for_mega_moe_m2n(buffer, ag_size + eg_size)
    views = tuple(map(torch.from_dlpack, slice_buffers(buffer)))
    x, x_sf, topk_ids, topk_weights, l1, l1_sf, l2, l2_sf = views

    assert int(pool_tokens) > 0
    element_divisor, sf_divisor = (2, 64) if use_nvfp4 else (1, 128)
    assert x.shape == (capacity, hidden // element_divisor)
    assert x_sf.shape == (capacity, hidden // sf_divisor)
    assert topk_ids.shape == (capacity, num_topk)
    assert topk_weights.shape == (capacity, num_topk)
    assert l1.shape == (int(pool_tokens), hidden // element_divisor)
    assert l2.shape == (int(pool_tokens), intermediate_hidden // element_divisor)
    assert l1_sf.shape[1] == hidden // sf_divisor
    assert l2_sf.shape[1] == intermediate_hidden // sf_divisor
    assert l1_sf.stride() == (1, l1_sf.shape[0])
    assert l2_sf.stride() == (1, l2_sf.shape[0])

    base = buffer.data_ptr()
    end = base + buffer.numel()
    assert all(base <= view.data_ptr() < end for view in views)
    assert all(view.data_ptr() % 16 == 0 for view in views)
    seeded_words = buffer.view(torch.int32)
    assert torch.count_nonzero(seeded_words).item() == 1
    assert seeded_words.max().item() == ag_size + eg_size


def _nvfp4_quantize(x):
    groups = x.float().view(x.shape[0], -1, 16)
    sf = (groups.abs().amax(dim=-1) * (1.0 / 6.0)).clamp(max=448)
    sf = sf.to(torch.float8_e4m3fn).float()
    inv_sf = torch.where(sf > 0, sf.reciprocal(), 0.0)
    scaled = groups * inv_sf.unsqueeze(-1)
    magnitude = scaled.abs()
    codes = torch.zeros_like(scaled, dtype=torch.uint8)
    # E2M1 RN rounds midpoint ties to the even code, not always downward.
    for index, boundary in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)):
        codes += (magnitude > boundary) | ((index % 2 == 1) & (magnitude == boundary))
    codes |= ((scaled < 0) & (codes != 0)).to(torch.uint8) << 3
    pairs = codes.view(x.shape[0], -1, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).view(torch.int8), sf


def _nvfp4_roundtrip(x):
    packed, sf = _nvfp4_quantize(x)
    return cast_back_from_fp4(packed, sf, gran_k=16)


def _make_nvfp4_scales(layer):
    experts = torch.arange(EXPERTS, device="cuda", dtype=torch.float32)
    input_scale = torch.tensor(2.0 ** (layer + 2), device="cuda")
    # Different layers, local experts, and gate/up columns expose index errors.
    l1_scales = torch.stack(
        (2.0 ** (3 + (experts + layer) % 3), 2.0 ** (5 + (experts + 2 * layer) % 3)),
        dim=-1,
    )
    l2_scales = 2.0 ** (4 + (experts + layer) % 3)
    l2_act_scales = 2.0 ** ((experts + layer) % 3 - 1)
    return input_scale, l1_scales, l2_scales, l2_act_scales


def _make_weights(layer: int, use_fp8_weights: bool = False, nvfp4_scales=None):
    generator = torch.Generator(device="cuda").manual_seed(100 + layer)
    quantized, reference = [], []
    for projection, (n, k) in enumerate(
        ((2 * INTERMEDIATE, HIDDEN), (HIDDEN, INTERMEDIATE))
    ):
        raw = (
            torch.randn(
                (EXPERTS * n, k),
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.125
        )
        gran_k = 32
        if nvfp4_scales is not None:
            gran_k = 16
            _, l1_scales, l2_scales, _ = nvfp4_scales
            row_scales = (
                l1_scales.repeat_interleave(INTERMEDIATE, dim=1)
                if projection == 0
                else l2_scales[:, None].expand(EXPERTS, HIDDEN)
            )
            packed, scales = _nvfp4_quantize(raw.float() * row_scales.reshape(-1, 1))
            reference.append(
                cast_back_from_fp4(packed, scales, gran_k=16).view(EXPERTS, n, k)
            )
        elif use_fp8_weights:
            blocks = raw.float().view(EXPERTS, n // 128, 128, k // 128, 128)
            scales = blocks.abs().amax(dim=(2, 4)).clamp_min(1e-4) / 448
            scales = torch.exp2(torch.ceil(torch.log2(scales)))
            packed = (blocks / scales[:, :, None, :, None]).to(torch.float8_e4m3fn)
            packed = packed.reshape(EXPERTS * n, k)
            scales = scales.repeat_interleave(128, dim=1).repeat_interleave(4, dim=2)
            reference.append(
                (
                    packed.float().view(EXPERTS, n, k // 32, 32) * scales.unsqueeze(-1)
                ).view(EXPERTS, n, k)
            )
        else:
            packed, scales = per_token_cast_to_fp4(raw, use_ue8m0=True, gran_k=32)
            reference.append(
                cast_back_from_fp4(packed, scales, gran_k=32).view(EXPERTS, n, k)
            )
        quantized.append(
            (packed.view(EXPERTS, n, -1), scales.view(EXPERTS, n, k // gran_k))
        )
    return quantized, reference


def _fp8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    packed, scales = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32)
    return (packed.float().view(x.shape[0], -1, 32) * scales.unsqueeze(-1)).view_as(x)


def _reference(
    x, ids, weights, layer_weights, output_scale, clamp=CLAMP, nvfp4_scales=None
):
    # Match the actual quantized operands, not the original BF16 weights.
    if nvfp4_scales is not None:
        input_scale, l1_scales, l2_scales, l2_act_scales = nvfp4_scales
        x = _nvfp4_roundtrip(x.float() * input_scale)
    else:
        x = _fp8_roundtrip(x.float())
    output = torch.zeros_like(x)
    for expert in range(EXPERTS):
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        gate_up = x[rows] @ layer_weights[0][expert].T
        if nvfp4_scales is not None:
            alphas = (input_scale * l1_scales[expert]).reciprocal()
            gate_up *= alphas.repeat_interleave(INTERMEDIATE)
        gate_up = gate_up.bfloat16().float()
        gate, up = gate_up.chunk(2, dim=-1)
        if clamp is not None:
            gate = gate.clamp(max=clamp)
            up = up.clamp(-clamp, clamp)
        activation = (gate / (1 + torch.exp(-gate))) * up
        route_weights = weights[rows, slots, None]
        if nvfp4_scales is not None:
            activation = _nvfp4_roundtrip(
                activation.bfloat16().float() * l2_act_scales[expert]
            )
        else:
            activation = _fp8_roundtrip(activation * route_weights)
        expert_output = activation @ layer_weights[1][expert].T
        if nvfp4_scales is not None:
            expert_output *= (l2_scales[expert] * l2_act_scales[expert]).reciprocal()
        expert_output = expert_output.bfloat16().float()
        if nvfp4_scales is not None:
            expert_output *= route_weights
        output.index_add_(0, rows, expert_output)
    return (output * output_scale).bfloat16()


def _make_input(rank, layer, lane, count, phase):
    generator = torch.Generator(device="cuda").manual_seed(
        1000 * phase + 100 * rank + 10 * layer + lane
    )
    # Mirror SGLang's non-null, unrouted sentinel for an empty A participant.
    x = torch.randn(
        (max(count, 1), HIDDEN),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    rows = torch.arange(x.shape[0], device="cuda")
    first = (rows + 3 * rank + layer + lane) % EXPERTS
    if phase == 0:
        # Fill the per-source expert slots, including the final input row.
        first.fill_(layer)
    ids = torch.stack((first, (first + EXPERTS // 2) % EXPERTS), dim=1).int()
    weights = (
        torch.rand((x.shape[0], TOPK), generator=generator, device="cuda") * 0.75
        + 0.125
    )
    ids[::3, 1] = -1
    if count == 0:
        x.zero_()
        ids.fill_(-1)
        weights.zero_()
    return x, ids, weights


def _check_output(actual, expected, ids, label):
    assert torch.isfinite(actual).all(), label
    empty_rows = (ids < 0).all(dim=1)
    assert torch.count_nonzero(actual[empty_rows]).item() == 0, label
    rms = expected.float().square().mean().sqrt().item()
    error = actual.float() - expected.float()
    if rms == 0:
        assert (
            empty_rows.all()
        ), f"{label}: active routes produced an all-zero reference"
        assert torch.count_nonzero(actual).item() == 0, label
        return
    relative_rms = error.square().mean().sqrt().item() / rms
    print(f"{label}: relative_rms={relative_rms:.6g}", flush=True)
    # Use a BF16-scale RMS error budget for the independently accumulated reference.
    assert relative_rms <= 2**-8, label


def run_distributed(use_fp8_weights: bool = False, use_nvfp4: bool = False) -> None:
    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=120),
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )
    assert dist.get_world_size() == 4, "Use torchrun --nproc_per_node=4 (AG2/EG2)"
    rank, group = dist.get_rank(), dist.group.WORLD
    # Only AG has a prior rendezvous, as with single-node A CustomAllReduceV2
    # and cross-node F. M2N must not inherit its process-global exchange index.
    from torch._C._distributed_c10d import _SymmetricMemory

    history_group = dist.new_group(ranks=[0, 1], backend="nccl")
    history_tensor = history_handle = None
    if rank < 2:
        history_tensor = _SymmetricMemory.empty_strided_p2p(
            (4096,),
            (1,),
            torch.uint8,
            torch.device("cuda", torch.cuda.current_device()),
            history_group.group_name,
        )
        history_handle = _SymmetricMemory.rendezvous(history_tensor)
    dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = False
    mma_type = "nvfp4xnvfp4" if use_nvfp4 else "fp8xfp4"
    buffers = [
        get_symm_buffer_for_mega_moe_m2n(
            group,
            ag_size=2,
            eg_size=2,
            num_experts=EXPERTS,
            num_max_tokens_per_rank=CAPACITY,
            num_topk=TOPK,
            hidden=HIDDEN,
            intermediate_hidden=INTERMEDIATE,
            mma_type=mma_type,
        )
        for _ in range(3)
    ]
    assert all(buffer.num_max_tokens_per_rank == 132 for buffer in buffers)
    nvfp4_scales = [
        _make_nvfp4_scales(layer) if use_nvfp4 else None for layer in range(2)
    ]
    layers = [
        _make_weights(layer, use_fp8_weights, nvfp4_scales[layer]) for layer in range(2)
    ]
    clamp = None if use_fp8_weights or use_nvfp4 else CLAMP
    weight_storage, descriptors = [], []
    l1_alphas = l2_alphas = l2_act_scales = None
    if rank >= 2:
        local_experts = EXPERTS // 2
        start = (rank - 2) * local_experts
        for quantized, _ in layers:
            local_weights = []
            for packed, scales in quantized:
                n = packed.shape[1]
                k = packed.shape[2] * (1 if use_fp8_weights else 2)
                packed = packed[start : start + local_experts].contiguous()
                scales = scales[start : start + local_experts].contiguous()
                if use_nvfp4:
                    scales = transform_ue4m3_sf_into_required_layout(scales, n)
                else:
                    scales = deep_gemm.transform_sf_into_required_layout(
                        scales, n, k, (1, 32), local_experts
                    )
                local_weights.append((packed, scales))
            transformed = deep_gemm.transform_weights_for_mega_moe(
                *local_weights, mma_type=mma_type
            )
            weight_storage.append(transformed)
            descriptors.append(
                make_m2n_weight_descs(
                    *transformed,
                    num_experts_per_rank=local_experts,
                    hidden=HIDDEN,
                    intermediate_hidden=INTERMEDIATE,
                    mma_type=mma_type,
                )
            )
        descriptors = torch.stack(descriptors).cuda()
        if use_nvfp4:
            local = slice(start, start + local_experts)
            l1_alphas = torch.stack(
                [(s[0] * s[1][local]).reciprocal() for s in nvfp4_scales]
            )
            l2_alphas = torch.stack(
                [(s[2][local] * s[3][local]).reciprocal() for s in nvfp4_scales]
            )
            l2_act_scales = torch.stack([s[3][local] for s in nvfp4_scales])
    streams = [torch.cuda.Stream() for _ in range(3)] if rank < 2 else []

    # Same allocations cross N1/N2/N3 phases, including partial/all-empty lanes
    # and unequal counts. Each phase is one persistent EG for two layers.
    phases = (
        ((CAPACITY,), (CAPACITY,)),
        ((17, 7, 3), (0, 11, 1)),
        ((0,), (0,)),
        ((1, 0, 33), (9, 0, 17)),
        ((16, 8), (13, 0)),
    )
    for phase, counts in enumerate(phases):
        num_lanes = len(counts[0])
        inputs, outputs = [], []
        if rank < 2:
            inputs = [
                [
                    _make_input(rank, layer, lane, count, phase)
                    for lane, count in enumerate(counts[rank])
                ]
                for layer in range(2)
            ]
            outputs = [
                [torch.full_like(data[0], float("nan")) for data in layer]
                for layer in inputs
            ]
        torch.cuda.synchronize()
        dist.barrier()

        def run():
            if rank < 2:
                for stream in streams[:num_lanes]:
                    stream.wait_stream(torch.cuda.current_stream())
                for layer in range(2):
                    for lane in range(num_lanes):
                        with torch.cuda.stream(streams[lane]):
                            mega_moe_m2n_ag(
                                outputs[layer][lane],
                                *inputs[layer][lane],
                                buffers[lane],
                                output_scale=OUTPUT_SCALES[layer],
                                num_sms=24,
                                input_scale=(
                                    nvfp4_scales[layer][0] if use_nvfp4 else None
                                ),
                            )
                for stream in streams[:num_lanes]:
                    torch.cuda.current_stream().wait_stream(stream)
            else:
                mega_moe_m2n_eg(
                    descriptors,
                    buffers[:num_lanes],
                    expected_num_tokens_per_rank=max(1, max(map(max, counts))),
                    num_experts_per_rank=EXPERTS // 2,
                    use_fp8_weights=use_fp8_weights,
                    activation_clamp=clamp,
                    fast_math=True,
                    l1_alphas=l1_alphas,
                    l2_alphas=l2_alphas,
                    l2_act_scales=l2_act_scales,
                )

        run()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for layer_outputs in outputs:
            for output in layer_outputs:
                output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        if rank < 2:
            for layer in range(2):
                for lane in range(num_lanes):
                    data = inputs[layer][lane]
                    expected = _reference(
                        *data,
                        layers[layer][1],
                        OUTPUT_SCALES[layer],
                        clamp,
                        nvfp4_scales[layer],
                    )
                    _check_output(
                        outputs[layer][lane],
                        expected,
                        data[1],
                        f"rank={rank} phase={phase} layer={layer} lane={lane}",
                    )
        dist.barrier()

    torch.cuda.synchronize()
    for buffer in buffers:
        buffer.destroy()
    history_handle = history_tensor = None
    if rank < 2:
        dist.destroy_process_group(history_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument("--fp8", action="store_true")
    quantization.add_argument("--nvfp4", action="store_true")
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--intermediate", type=int, default=INTERMEDIATE)
    args = parser.parse_args()
    HIDDEN, INTERMEDIATE = args.hidden, args.intermediate
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        for capacity in (4, 128, 132):
            test_layout(capacity, use_nvfp4=args.nvfp4)
    else:
        run_distributed(use_fp8_weights=args.fp8, use_nvfp4=args.nvfp4)
    print("OK")
