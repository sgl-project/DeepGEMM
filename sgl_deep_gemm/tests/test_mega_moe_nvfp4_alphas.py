# NVFP4 outer-scale (global scale) plumbing for the mega-MoE kernel.
#
# Four tables carry the NVFP4 dequant alphas:
#
#   x_scales      per token, written by `mega_moe_pre_dispatch` (1 / gs_x1)
#   l1_alphas     [num_LOCAL_experts, 2] = 1 / (gs_w1, gs_w3), on the L1 accum
#   l2_alphas     [num_LOCAL_experts]    = 1 / gs_w2,          on the L2 accum
#   expert_scales [num_GLOBAL_experts]   folded into the topk weight, O(1) only
#
# The fc2 alpha is deliberately NOT routed through `expert_scales`: that lands on
# the L1 output, which is then re-quantized to NVFP4 with a bare per-block E4M3
# SF, and a realistic `1 / gs_w2` (~1e-4) sinks that SF under E4M3's subnormal
# floor — `y` comes back all zeros (measured). `expert_scales` is exercised
# separately below with an O(1) ratio, which is all it can carry.
#
# Method: quantize the same BF16 weights twice — once with gs = 1 and no alphas,
# once with per-expert gs and the alphas that undo it — and compare `y`.
#   - Power-of-two gs: every scaling is exact in FP32/E4M3, so the two runs must
#     be BITWISE equal.
#   - Realistic gs (= 448*6/amax, not powers of two): the two encodings round
#     differently, so this arm is only a sanity band — measured 0.18 rel-L2 on
#     `y`, which is what two independent NVFP4 encodings cost (7.7% apart on the
#     weights alone, over two GEMMs). Its job is to show realistic gs magnitudes
#     neither saturate the E4M3 SF nor zero it, not to catch subtle bugs; the
#     pow2 arm and the roll controls are the sharp instruments here.
#   - Control: rolling either alpha table by one expert must change `y`,
#     otherwise the test would pass on a kernel that ignores the table.
#
# Run with >= 2 ranks: local- vs global-expert index confusion only shows up on
# a rank whose `expert_lo` is nonzero.
#
# Usage:
#   PYTHONPATH=/workspace/chunan/DeepGEMM CUDA_VISIBLE_DEVICES=4,5 MASTER_PORT=29511 \
#       python3 sgl_deep_gemm/tests/test_mega_moe_nvfp4_alphas.py --num-processes 2

import argparse
import sys
import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils import cast_to_ue4m3, transform_ue4m3_sf_into_required_layout
from deep_gemm.utils.math import _quantize_to_fp4_e2m1
from deep_gemm.utils.dist import dist_print, init_dist

GRAN_K = 16
E4M3_MAX = 448.0
FP4_MAX = 6.0


def cast_to_nvfp4_with_gs(w: torch.Tensor, gs) -> tuple:
    """NVFP4 with an outer scale. Dequant contract: `w ~= fp4 * sf / gs`.

    `gs` is a scalar or a per-row (m,) tensor — the L1 weight carries gs_w1 on
    its gate rows and gs_w3 on its up rows. `gs = 1` reproduces
    `per_token_cast_to_nvfp4` exactly.
    """
    m, n = w.shape
    v = w.view(m, -1, GRAN_K).float()
    gs = torch.as_tensor(gs, dtype=torch.float, device=w.device).reshape(-1, 1).expand(m, 1)
    sf = cast_to_ue4m3(v.abs().amax(dim=2) * (gs / FP4_MAX))
    scale = torch.where(sf > 0, gs / sf, torch.zeros_like(sf))
    codes = _quantize_to_fp4_e2m1(v * scale.unsqueeze(2)).view(m, n)
    codes2 = codes.view(m, n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return packed.contiguous(), sf


def cast_grouped(w: torch.Tensor, gs_rows) -> tuple:
    """`w` is (E, N, K) BF16; `gs_rows` is one per-row gs tensor per expert."""
    num_experts, n, k = w.shape
    packed = torch.empty((num_experts, n, k // 2), dtype=torch.int8, device=w.device)
    sf = torch.empty((num_experts, n, k // GRAN_K), dtype=torch.float, device=w.device)
    for e in range(num_experts):
        packed[e], sf[e] = cast_to_nvfp4_with_gs(w[e], gs_rows[e])
    return packed, transform_ue4m3_sf_into_required_layout(sf, n)


def make_gs(amax_per_expert: torch.Tensor, pow2: bool) -> torch.Tensor:
    """Realistic NVFP4 global scale, `448 * 6 / amax`, optionally snapped to a
    power of two so the two runs stay bitwise comparable.

    Same-shaped random weights have near-identical amax, so the pow2 arm would
    otherwise hand every expert the same scale and the roll-by-one control would
    be vacuous. Spread it downwards per expert — never up, `gs > 448*6/amax`
    saturates the E4M3 SF.
    """
    gs = (E4M3_MAX * FP4_MAX) / amax_per_expert.clamp_min(1e-4)
    if not pow2:
        return gs
    spread = torch.pow(2.0, -(torch.arange(gs.numel(), device=gs.device) % 4).float())
    return torch.pow(2.0, torch.floor(torch.log2(gs))) * spread


# noinspection PyUnboundLocalVariable
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(1000 + rank_idx)

    num_tokens = args.num_tokens
    hidden, inter = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_local_experts = num_experts // num_ranks
    expert_lo = rank_idx * num_local_experts

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, args.num_max_tokens_per_rank, num_topk,
        hidden, inter, mma_type='nvfp4xnvfp4')

    # `x` is damped so the L1 output amax stays under the L2 activation SF's
    # 448*6 ceiling; weights stay unit-variance so their own block SFs stay in
    # E4M3's normal range in BOTH runs (a gs=1 encoding of small weights lands on
    # E4M3 subnormals, where power-of-two rescaling is no longer exact).
    x = (torch.randn((num_tokens, hidden), device='cuda') * args.x_scale).bfloat16()
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.int()

    # Weights are local, but a GLOBAL table (`expert_scales`) has to know every
    # rank's values. Draw all experts from a shared seed and keep the local
    # slice, so the global tables agree across ranks.
    torch.manual_seed(7)
    l1_all = torch.randn((num_experts, inter * 2, hidden), dtype=torch.bfloat16, device='cuda')
    l2_all = torch.randn((num_experts, hidden, inter), dtype=torch.bfloat16, device='cuda')
    gate_amax = l1_all[:, :inter].abs().float().amax(dim=(1, 2))
    up_amax = l1_all[:, inter:].abs().float().amax(dim=(1, 2))
    l2_amax = l2_all.abs().float().amax(dim=(1, 2))
    l1_local = l1_all[expert_lo:expert_lo + num_local_experts].contiguous()
    l2_local = l2_all[expert_lo:expert_lo + num_local_experts].contiguous()
    del l1_all, l2_all

    base_w = deep_gemm.transform_weights_for_mega_moe(
        cast_grouped(l1_local, [torch.ones(inter * 2, device='cuda')] * num_local_experts),
        cast_grouped(l2_local, [torch.ones(hidden, device='cuda')] * num_local_experts),
        'swiglu', 'nvfp4xnvfp4')

    def run(weights, l1_alphas=None, l2_alphas=None, expert_scales=None, tw=None,
            l2_act_scales=None):
        deep_gemm.mega_moe_pre_dispatch(
            x, topk_idx, topk_weights if tw is None else tw,
            buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights,
            num_tokens=num_tokens, group_size=GRAN_K, mma_type='nvfp4xnvfp4',
            buf_x_scales=buffer.x_scales, expert_scales=expert_scales)
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.fp8_fp4_mega_moe(
            y=y, l1_weights=weights[0], l2_weights=weights[1], sym_buffer=buffer,
            recipe=(1, 1, GRAN_K), activation='swiglu', fast_math=bool(args.fast_math),
            use_x_scales=True, l1_alphas=l1_alphas, l2_alphas=l2_alphas,
            l2_act_scales=l2_act_scales)
        dist.barrier()
        torch.cuda.synchronize()
        return y

    y_base = run(base_w)
    dist_print(f'Config: {num_ranks} ranks x {num_local_experts} local experts, '
               f'{num_tokens} tokens, hidden={hidden}, inter={inter}, '
               f'|y_base| mean {y_base.float().abs().mean().item():.4g}', once_in_node=True)

    for pow2 in (True, False):
        gs_w1, gs_w3, gs_w2 = (make_gs(a, pow2) for a in (gate_amax, up_amax, l2_amax))
        scaled = deep_gemm.transform_weights_for_mega_moe(
            cast_grouped(l1_local, [torch.cat([gs_w1[expert_lo + e].expand(inter),
                                               gs_w3[expert_lo + e].expand(inter)])
                                    for e in range(num_local_experts)]),
            cast_grouped(l2_local, [gs_w2[expert_lo + e].expand(hidden)
                                    for e in range(num_local_experts)]),
            'swiglu', 'nvfp4xnvfp4')

        # Both LOCAL index space; `l1_alphas` columns are (gate, up).
        local = slice(expert_lo, expert_lo + num_local_experts)
        l1_alphas = torch.stack([1.0 / gs_w1[local], 1.0 / gs_w3[local]], dim=1).contiguous()
        l2_alphas = (1.0 / gs_w2[local]).contiguous()

        y_scaled = run(scaled, l1_alphas, l2_alphas)
        tag = 'pow2' if pow2 else 'realistic'
        if pow2:
            assert torch.equal(y_base, y_scaled), \
                f'[{tag}] power-of-two global scales must cancel bitwise, max |delta| = ' \
                f'{(y_base.float() - y_scaled.float()).abs().max().item():.3e}'
            dist_print(f' > [{tag}] gs_w2 = {gs_w2.tolist()} — bitwise equal', once_in_node=True)
        else:
            rel = ((y_scaled.float() - y_base.float()).norm() / y_base.float().norm()).item()
            assert rel < args.rel_tol, f'[{tag}] rel-L2 {rel:.4f} >= {args.rel_tol}'
            dist_print(f' > [{tag}] rel-L2 = {rel:.5f} (< {args.rel_tol})', once_in_node=True)

        for name, wrong in (('l1_alphas', (l1_alphas.roll(1, dims=0).contiguous(), l2_alphas)),
                            ('l2_alphas', (l1_alphas, l2_alphas.roll(1).contiguous()))):
            assert not torch.equal(y_scaled, run(scaled, *wrong)), \
                f'[{tag}] rolling `{name}` changed nothing — the kernel ignores it'
        dist_print(f' > [{tag}] roll-by-one control: both tables have teeth', once_in_node=True)

    # `l2_act_scales` (LOCAL index): pow2 fc2 input gs is cancelled by `1/gs`
    # in `l2_alphas`. Damp x so the scaled block SFs stay under E4M3's 448
    # satfinite ceiling. Exact pow2 invariance still breaks for blocks whose SF
    # lands in E4M3's SUBNORMAL range (absolute 2**-9 grid, not scale
    # invariant): a re-rounded SF moves single FP4 codes, perturbing outputs by
    # at most ~1 BF16 ulp of the tensor scale. Bound accordingly, not bitwise.
    x = (x.float() * 0.25).bfloat16()
    y_damped = run(base_w)
    gs_in = torch.pow(2.0, (torch.arange(num_local_experts, device='cuda',
                                         dtype=torch.float) % 3))
    y_gs = run(base_w, l2_alphas=(1.0 / gs_in).contiguous(),
               l2_act_scales=gs_in.contiguous())
    d = (y_gs.float() - y_damped.float()).abs()
    tol = torch.finfo(torch.bfloat16).eps * y_damped.float().abs().max()
    rel = (d.norm() / y_damped.float().norm()).item()
    assert d.max() <= tol and rel < 1e-3, \
        f'l2_act_scales cancellation off: max |d| {d.max().item():.3e} ' \
        f'(tol {tol.item():.3e}), rel-L2 {rel:.3e}'
    assert not torch.equal(y_damped, run(base_w, l2_act_scales=gs_in.contiguous())), \
        'l2_act_scales alone changed nothing -- the kernel ignores it'
    dist_print(f' > l2_act_scales: pow2 gs cancels to 1 tensor-scale ulp '
               f'(max |d| {d.max().item():.2e}, rel {rel:.1e}), teeth ok',
               once_in_node=True)

    # `expert_scales`, GLOBAL index space: folding an O(1) per-expert ratio in the
    # kernel must equal folding it into `topk_weights` on the host. Bitwise —
    # both are the same FP32 multiply, so this pins the index space exactly.
    ratios = (1.0 + 0.25 * torch.arange(num_experts, device='cuda', dtype=torch.float)
              / num_experts).contiguous()
    host_tw = topk_weights * torch.where(topk_idx >= 0, ratios[topk_idx.long().clamp_min(0)], 1.0)
    assert torch.equal(run(base_w, expert_scales=ratios), run(base_w, tw=host_tw)), \
        '`expert_scales` does not match a host-side fold — check the global expert index'
    dist_print(' > expert_scales: matches a host-side topk_weights fold', once_in_node=True)

    dist_print('OK', once_in_node=True)
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=2)
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=1024)
    parser.add_argument('--num-tokens', type=int, default=1024)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--intermediate-hidden', type=int, default=512)
    parser.add_argument('--num-experts', type=int, default=8)
    parser.add_argument('--num-topk', type=int, default=2)
    parser.add_argument('--x-scale', type=float, default=0.1)
    parser.add_argument('--fast-math', type=int, default=1)
    parser.add_argument('--rel-tol', type=float, default=0.30)
    args = parser.parse_args()
    assert args.num_experts % args.num_processes == 0
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
    sys.exit(0)
