# Bytewise + correctness probe for `deep_gemm.mega_moe_pre_dispatch`.
#
# The fused pre-dispatch kernel produces the exact byte layout DeepGEMM's
# mega-MoE symmetric `x`, `x_sf`, `topk_idx`, and `topk_weights` slots expect.
# This test verifies bit-for-bit equivalence against the in-tree host helpers
# (`per_token_cast_to_fp8`, `per_token_cast_to_fp4`) for both the FP8 and the
# packed FP4 dtype branches, plus the pad-fill correctness contract.
#
# Single-GPU; no distributed init needed.

import argparse
import sys
import torch

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.math import _quantize_to_fp4_e2m1, pack_ue4m3_to_int


def _ref_per_token_nvfp4_with_pts(x: torch.Tensor):
    m, n = x.shape
    assert n % 16 == 0
    # Compute reference scales on CPU so float32 division uses IEEE-754 RN (`div.rn.f32`),
    # matching the device kernel instead of ATen CUDA's `div.Scalar` reciprocal multiply.
    x_f = x.cpu().float()
    row_max = x_f.abs().amax(dim=1).clamp(min=1e-4)
    pts = row_max / torch.tensor(448.0 * 6.0, dtype=torch.float32)
    x_view = x_f.view(m, -1, 16)
    group_max = x_view.abs().amax(dim=2)
    sf_fp8 = (group_max / (torch.tensor(6.0, dtype=torch.float32) * pts.unsqueeze(1))).clamp(max=448.0).to(torch.float8_e4m3fn)
    sf_dec = sf_fp8.to(torch.float16).float()
    denom = sf_dec * pts.unsqueeze(1)
    inv_scale = torch.where(sf_dec > 0, torch.tensor(1.0, dtype=torch.float32) / denom, torch.zeros_like(denom))
    codes = _quantize_to_fp4_e2m1(x_view * inv_scale.unsqueeze(2)).view(m, n)
    codes2 = codes.view(m, n // 2, 2)
    packed = ((codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)).contiguous()
    return packed.to(x.device), pack_ue4m3_to_int(sf_fp8.to(x.device)), pts.to(x.device)


def _alloc_outputs(padded_max: int, hidden: int, top_k: int,
                   group_size: int, use_fp4_acts: bool):
    num_groups = hidden // group_size
    assert num_groups % 4 == 0
    if use_fp4_acts:
        buf_x = torch.empty((padded_max, hidden // 2), dtype=torch.int8, device='cuda')
    else:
        buf_x = torch.empty((padded_max, hidden), dtype=torch.float8_e4m3fn, device='cuda')
    buf_x_sf = torch.empty((padded_max, num_groups // 4), dtype=torch.int32, device='cuda')
    buf_topk_idx = torch.empty((padded_max, top_k), dtype=torch.int64, device='cuda')
    buf_topk_weights = torch.empty((padded_max, top_k), dtype=torch.float32, device='cuda')
    # Sentinel-fill so any write-correctness bug shows up as a non-zero diff.
    buf_x.fill_(0)
    buf_x_sf.fill_(0)
    buf_topk_idx.fill_(0)
    buf_topk_weights.fill_(0)
    return buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights


def _run_one(use_fp4_acts: bool, args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    M = args.num_tokens
    P = args.padded_max
    H = args.hidden
    K = args.top_k
    G = args.group_size
    assert P >= M, 'padded_max must be >= num_tokens'

    # --- Inputs (BF16 acts, int32 topk_idx, float topk_weights) ---
    x = torch.randn((M, H), dtype=torch.bfloat16, device='cuda')
    if use_fp4_acts and M > 0 and G >= 32:
        # Force scale=1 in the first group and exercise every positive/negative
        # E2M1 midpoint; random BF16 inputs alone need not hit each tie.
        x[0, :G] = 0
        x[0, :16] = torch.tensor(
            [0, .25, .75, 1.25, 1.75, 2.5, 3.5, 5, 6,
             -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5],
            dtype=x.dtype, device=x.device)
    # Use plausible expert ids in [0, num_experts) and float weights.
    num_experts = args.num_experts
    topk_idx = torch.randint(0, num_experts, (M, K), dtype=torch.int32, device='cuda')
    topk_weights = torch.randn((M, K), dtype=torch.float32, device='cuda')

    buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights = _alloc_outputs(P, H, K, G, use_fp4_acts)

    # --- Kernel under test ---
    deep_gemm.mega_moe_pre_dispatch(
        x, topk_idx, topk_weights,
        buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights,
        num_tokens=M, group_size=G,
        mma_type='mxf4xmxf4' if use_fp4_acts else 'fp8xfp4',
    )
    torch.cuda.synchronize()

    # --- Reference (host helper) ---
    if use_fp4_acts:
        ref_x, ref_sf = per_token_cast_to_fp4(
            x, use_ue8m0=True, gran_k=G, use_packed_ue8m0=True)
    else:
        ref_x, ref_sf = per_token_cast_to_fp8(
            x, use_ue8m0=True, gran_k=G, use_packed_ue8m0=True)

    # --- Bytewise compare on valid-token rows ---
    if use_fp4_acts:
        # ref_x is int8 (M, H/2); buf_x[:M] is int8 (M, H/2). Compare raw bytes.
        kernel_bytes = buf_x[:M].view(torch.uint8)
        ref_bytes = ref_x.view(torch.uint8)
    else:
        # ref_x is float8_e4m3fn (M, H); compare via uint8 view.
        kernel_bytes = buf_x[:M].view(torch.uint8)
        ref_bytes = ref_x.view(torch.uint8)
    diff_x = (kernel_bytes != ref_bytes)
    if diff_x.any().item():
        bad = diff_x.nonzero()
        first = bad[0].tolist()
        i, j = first[0], first[1]
        raise AssertionError(
            f'[{"FP4" if use_fp4_acts else "FP8"}] buf_x mismatch '
            f'at row {i}, col {j}: kernel={int(kernel_bytes[i, j])} '
            f'ref={int(ref_bytes[i, j])} (total mismatches={int(diff_x.sum())})')
    if use_fp4_acts and M > 0 and G >= 32:
        expected_midpoints = torch.tensor(
            [0x00, 0x22, 0x44, 0x66, 0x07, 0xaa, 0xcc, 0xee],
            dtype=torch.uint8, device=x.device)
        assert torch.equal(kernel_bytes[0, :8], expected_midpoints), 'FP4 ties must round to even'

    # SF byte layout: (M, num_groups/4) int32 → (M, num_groups) UE8M0 bytes.
    kernel_sf_bytes = buf_x_sf[:M].view(torch.uint8)
    ref_sf_bytes = ref_sf.view(torch.uint8)
    diff_sf = (kernel_sf_bytes != ref_sf_bytes)
    if diff_sf.any().item():
        bad = diff_sf.nonzero()
        first = bad[0].tolist()
        i, j = first[0], first[1]
        raise AssertionError(
            f'[{"FP4" if use_fp4_acts else "FP8"}] buf_x_sf mismatch '
            f'at row {i}, byte {j}: kernel={int(kernel_sf_bytes[i, j])} '
            f'ref={int(ref_sf_bytes[i, j])} (total mismatches={int(diff_sf.sum())})')

    # --- topk pass-through and pad-fill ---
    # Valid rows: int32 → int64 widening match.
    if not torch.equal(buf_topk_idx[:M], topk_idx.to(torch.int64)):
        raise AssertionError('topk_idx pass-through mismatch on valid rows')
    if not torch.equal(buf_topk_weights[:M], topk_weights):
        raise AssertionError('topk_weights pass-through mismatch on valid rows')
    # Pad rows.
    if P > M:
        if not torch.all(buf_topk_idx[M:] == -1).item():
            raise AssertionError('pad rows of buf_topk_idx must equal -1')
        if not torch.all(buf_topk_weights[M:] == 0.0).item():
            raise AssertionError('pad rows of buf_topk_weights must equal 0.0')

    print(f'  PASS  '
          f'[{"FP4" if use_fp4_acts else "FP8"}] '
          f'M={M} P={P} H={H} K={K} G={G} — bytewise equal vs host helper '
          f'+ pad-fill correct')


def _run_nvfp4(hidden: int, top_k: int, args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    M = args.num_tokens
    P = args.padded_max
    H = hidden
    K = top_k
    G = 16
    assert P >= M

    x = torch.randn((M, H), dtype=torch.bfloat16, device='cuda')
    if M > 0:
        # Exercise E2M1 midpoints and tiny negatives (-0 -> +0 canonicalization) at exact scale=1.
        x[0, :] = 0
        x[0, 16] = 448.0 * 6.0
        x[0, :16] = torch.tensor(
            [0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0,
             -0.05, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, 6.0],
            dtype=x.dtype, device=x.device)

    num_experts = args.num_experts
    topk_idx = torch.randint(-1, num_experts, (M, K), dtype=torch.int32, device='cuda')
    topk_weights = torch.randn((M, K), dtype=torch.float32, device='cuda')
    expert_scales = torch.rand((num_experts,), dtype=torch.float32, device='cuda') + 0.5

    buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights = _alloc_outputs(P, H, K, G, use_fp4_acts=True)
    buf_x_scales = torch.zeros((P,), dtype=torch.float32, device='cuda')

    deep_gemm.mega_moe_pre_dispatch(
        x, topk_idx, topk_weights,
        buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights,
        num_tokens=M, group_size=G, mma_type='nvfp4xnvfp4',
        buf_x_scales=buf_x_scales, expert_scales=expert_scales,
    )
    torch.cuda.synchronize()

    ref_x, ref_sf, ref_pts = _ref_per_token_nvfp4_with_pts(x)
    assert torch.equal(buf_x[:M].view(torch.uint8), ref_x.view(torch.uint8)), \
        f'[NVFP4] buf_x mismatch for H={H} K={K}'
    assert torch.equal(buf_x_sf[:M].view(torch.uint8), ref_sf.view(torch.uint8)), \
        f'[NVFP4] buf_x_sf mismatch for H={H} K={K}'
    assert torch.allclose(buf_x_scales[:M], ref_pts, rtol=1e-6, atol=1e-7), \
        f'[NVFP4] buf_x_scales mismatch for H={H} K={K}'
    assert torch.equal(buf_topk_idx[:M], topk_idx.to(torch.int64)), \
        f'[NVFP4] topk_idx mismatch for H={H} K={K}'

    ref_weights = torch.where(
        topk_idx >= 0,
        topk_weights * expert_scales[topk_idx.clamp(min=0)],
        topk_weights)
    assert torch.allclose(buf_topk_weights[:M], ref_weights, rtol=1e-6, atol=1e-6), \
        f'[NVFP4] topk_weights mismatch for H={H} K={K}'
    if P > M:
        assert torch.all(buf_topk_idx[M:] == -1).item(), 'pad rows of buf_topk_idx must equal -1'
        assert torch.all(buf_topk_weights[M:] == 0.0).item(), 'pad rows of buf_topk_weights must equal 0.0'

    print(f'  PASS  [NVFP4] M={M} P={P} H={H} K={K} G={G} — bytewise equal vs host helper + pad-fill correct')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-tokens', type=int, default=512)
    parser.add_argument('--padded-max', type=int, default=576)  # > num_tokens to exercise pad
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--group-size', type=int, default=32)
    parser.add_argument('--num-experts', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dtype', choices=['fp8', 'fp4', 'nvfp4', 'both', 'all'], default='all')
    args = parser.parse_args()

    if args.dtype in ('fp8', 'both', 'all'):
        _run_one(use_fp4_acts=False, args=args)
    if args.dtype in ('fp4', 'both', 'all'):
        _run_one(use_fp4_acts=True, args=args)
    if args.dtype in ('nvfp4', 'all'):
        _run_nvfp4(hidden=args.hidden, top_k=args.top_k, args=args)
        if args.hidden == 1024:
            # Also cover the 8-elem/thread fallback (non-512 hidden) and unaligned odd-top_k pad-fill.
            _run_nvfp4(hidden=1280, top_k=3, args=args)
    print('OK')


if __name__ == '__main__':
    sys.exit(main())
