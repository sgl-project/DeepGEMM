"""Fused SM90 MegaMoE APIs, derived from PR #440 and extended with MXFP4."""
import torch
import types
from typing import Optional, Tuple, Union

from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C
from .mxfp4 import (
    MXFP4ProcessedWeights as _MXFP4ProcessedWeights,
    transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90,
    _validate_processed_mxfp4_kernel_weights,
)

def _default_l2_act_sf_gran_k_sm90_fused(hidden: int, intermediate_hidden: int) -> int:
    return 128 if hidden % 512 == 0 and (2 * intermediate_hidden) % 512 == 0 else 64

class SM90FusedSymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 use_fp8_dispatch: bool = True,
                 activation: str = 'swiglu',
                 num_experts_per_wave: Optional[int] = None,
                 l2_act_sf_gran_k: Optional[int] = None):
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        # Wave-size knob; it selects the schedule the buffer is sized for (get_buffer_schedule_sm90_fused in
        # csrc/jit_kernels/heuristics/sm90_fused_mega_moe.hpp): None (0) = the default, a lag-sized ring running the L2-lag schedule
        # from 1024 tokens per rank and full-pool buffers below that; -1 = wave schedule, ring sized by the occupancy heuristic;
        # N > 0 = wave schedule, ring sized for a fixed wave of N experts
        self.num_experts_per_wave = 0 if num_experts_per_wave is None else num_experts_per_wave
        self.l2_act_sf_gran_k = l2_act_sf_gran_k if l2_act_sf_gran_k is not None else \
            _default_l2_act_sf_gran_k_sm90_fused(hidden, intermediate_hidden)

        num_bytes, num_ring_tokens, l2_lag_encoded = \
            _C.get_symm_buffer_size_for_sm90_fused_mega_moe(
                group.size(), num_experts,
                num_max_tokens_per_rank, num_topk,
                hidden, intermediate_hidden,
                use_fp8_dispatch, activation,
                self.num_experts_per_wave, self.l2_act_sf_gran_k,
            )
        # Ring capacity (0 = full pool) and encoded L2-lag schedule the buffer was sized for; passed back at every launch
        self.num_ring_tokens = num_ring_tokens
        self.l2_lag_encoded = l2_lag_encoded
        allocator = torch if group.size() == 1 else symm_mem
        self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = _C.slice_symm_buffer_for_sm90_fused_mega_moe(
            self.buffer, group.size(), num_experts, num_max_tokens_per_rank,
            num_topk, hidden, intermediate_hidden, use_fp8_dispatch, activation,
            self.num_ring_tokens, self.l2_act_sf_gran_k)

    def destroy(self):
        self.handle = None
        for name in (
            'x', 'x_sf', 'topk_idx', 'topk_weights',
            'l1_acts', 'l1_acts_sf', 'l2_acts', 'l2_acts_sf',
        ):
            setattr(self, name, None)
        self.buffer = None
        self.group = None

def fp8_mxfp4_mega_moe(y: torch.Tensor,
                       l1_weights: _MXFP4ProcessedWeights,
                       l2_weights: _MXFP4ProcessedWeights,
                       sym_buffer: SM90FusedSymmBuffer,
                       cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                       recipe: Tuple[int, int, int] = (1, 1, 32),
                       activation: str = 'swiglu',
                       activation_clamp: Optional[float] = None,
                       fast_math: bool = True,
                       *,
                       num_tokens_bound: Optional[int] = None,
                       split_k: bool = False,
                       preexpanded_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
    """SM90 fused MegaMoE with packed MXFP4 routed weights (opt-in).

    ``split_k`` is a hint that the call is a decode step of ONE global token (the same value on every rank); it selects the instance
    that splits every tile's K over two CTAs when this rank's routing allows (results stay correct for any routing). Leave it off
    otherwise: the split code costs the other small calls ~1-2 %.

    ``l1_weights`` / ``l2_weights`` are the processed triples
    ``(packed_e2m1, relative_ue8m0, weight_scale_2)`` returned by
    :func:`transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90`. The buffer must come from
    ``get_symm_buffer_for_sm90_mega_moe(..., fused=True)``; the FP8 entry point
    :func:`fp8_mega_moe` is unchanged.

    ``preexpanded_weights`` optionally supplies caller-owned FP8 tiles produced
    by :func:`expand_mxfp4_mega_moe_weights`. Refresh them on the same stream
    before each layer call. MXFP4 activation and secondary-scale rules still apply.
    """
    if not isinstance(sym_buffer, SM90FusedSymmBuffer):
        raise TypeError('fp8_mxfp4_mega_moe requires an SM90FusedSymmBuffer (fused=True)')
    _validate_processed_mxfp4_kernel_weights(l1_weights, l2_weights)
    expected_local_experts = sym_buffer.num_experts // sym_buffer.group.size()
    expected_l1_shape = (expected_local_experts, 2 * sym_buffer.intermediate_hidden, sym_buffer.hidden // 2)
    expected_l2_shape = (expected_local_experts, sym_buffer.hidden, sym_buffer.intermediate_hidden // 2)
    def logical_shape(w):   # the tiled layout [E, N/256, K/128, 16384] (layout='tiled') has the same bytes as [E, N, K/2]
        return (w.size(0), w.size(1) * 256, w.size(2) * 64) if w.dim() == 4 else tuple(w.shape)
    if (l1_weights[0].dim() == 4) != (l2_weights[0].dim() == 4):
        raise ValueError('L1 and L2 MXFP4 weights must use the same layout')
    if logical_shape(l1_weights[0]) != expected_l1_shape or logical_shape(l2_weights[0]) != expected_l2_shape:
        raise ValueError(
            f'SM90 MXFP4 weights do not match the buffer: expected {expected_l1_shape} and {expected_l2_shape}, '
            f'got {tuple(l1_weights[0].shape)} and {tuple(l2_weights[0].shape)}')
    if preexpanded_weights is not None:
        _validate_preexpanded_weights(l1_weights, l2_weights, preexpanded_weights)
        l1_weights = (preexpanded_weights[0], *l1_weights[1:])
        l2_weights = (preexpanded_weights[1], *l2_weights[1:])
    _C.sm90_fused_fp8_mxfp4_mega_moe(
        y,
        *l1_weights, *l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math,
        sym_buffer.num_ring_tokens,
        0 if num_tokens_bound is None else num_tokens_bound,
        sym_buffer.l2_lag_encoded, sym_buffer.l2_act_sf_gran_k,
        bool(split_k)
    )


def _validate_preexpanded_weights(l1_weights, l2_weights, output):
    if len(output) != 2:
        raise ValueError('Expected L1 and L2 preexpanded output buffers')
    ranges = []
    for (packed, _, _), expanded in zip((l1_weights, l2_weights), output):
        expected = (*packed.shape[:3], 32768)
        if (packed.ndim != 4 or packed.shape[-1] != 16384 or
                expanded.shape != expected or expanded.dtype != torch.float8_e4m3fn or
                expanded.device != packed.device or not expanded.is_contiguous()):
            raise ValueError(f'Expected a contiguous FP8 tiled buffer {expected} on {packed.device}')
        if expanded.data_ptr() % 16:
            raise ValueError('Preexpanded weights require 16-byte alignment')
        ranges.append((expanded.data_ptr(), expanded.data_ptr() + expanded.nbytes))
    if max(ranges[0][0], ranges[1][0]) < min(ranges[0][1], ranges[1][1]):
        raise ValueError('L1 and L2 preexpanded buffers must not overlap')
    for start, end in ranges:
        for triple in (l1_weights, l2_weights):
            for source in triple:
                if max(start, source.data_ptr()) < min(end, source.data_ptr() + source.nbytes):
                    raise ValueError('Preexpanded buffers must not overwrite source weights or scales')


def build_mxfp4_mega_moe_expert_mask(topk_ids, mask):
    """Refresh a rank-local physical-expert mask, including during Graph replay.

    The caller must union the mask across the EP group before masked expansion.
    The mask must not overlap the input IDs.
    """
    _C.build_mxfp4_mega_moe_expert_mask(topk_ids, mask)


def expand_mxfp4_mega_moe_weights(l1_weights, l2_weights, output, *, active_mask=None, expert_offset=0):
    """Expand tiled MXFP4 into caller-owned FP8 buffers on the current stream.

    Refresh both buffers for every layer invocation, including Graph replay.
    The caller must serialize users of the output storage and keep it alive.
    With a mask, only marked experts are written. It must cover the EP-wide
    physical routes for this invocation; unmarked storage remains untouched.
    The mask must not overlap either output buffer.
    """
    _validate_processed_mxfp4_kernel_weights(l1_weights, l2_weights)
    _validate_preexpanded_weights(l1_weights, l2_weights, output)
    for (packed, scale, _), expanded in zip((l1_weights, l2_weights), output):
        _C.expand_mxfp4_mega_moe_weight(packed, scale, expanded, active_mask, expert_offset)
