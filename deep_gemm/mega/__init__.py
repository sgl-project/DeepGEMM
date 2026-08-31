import os
import torch
import types
import warnings
from typing import Tuple, Optional, Union
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C


def _check_legacy_fp4_acts_env() -> None:
    """The fork's `DG_USE_FP4_ACTS` / `DG_USE_MXF4_KIND` env vars are superseded by
    `mma_type`, and are refused rather than ignored.

    They cannot be silently remapped. `DG_USE_FP4_ACTS` selected FP4 activations under
    `kind::mxf8f6f4` while leaving weights in the gran-8 *unpacked* gate/up interleave;
    the equivalent typed kinds (`mxf4xmxf4` / `nvfp4xnvfp4`) are dense `kind::mxf4` and
    need the gran-16 *packed* interleave from `_interleave_weights_packed_fp4`. Honouring
    the env var here would leave a caller who does not also pass `mma_type` to
    `transform_weights_for_mega_moe` handing packed-layout kernels unpacked weights —
    silently wrong results. Fail loudly instead.
    """
    for name in ('DG_USE_FP4_ACTS', 'DG_USE_MXF4_KIND'):
        if int(os.environ.get(name, '0')) != 0:
            raise RuntimeError(
                f'`{name}` is no longer supported; it is replaced by the symmetric '
                f"buffer's `mma_type`. Pass mma_type='mxf4xmxf4' (per-32 UE8M0 scales) "
                f"or 'nvfp4xnvfp4' (per-16 UE4M3) to get_symm_buffer_for_mega_moe(), and "
                f'the SAME mma_type to transform_weights_for_mega_moe() — the FP4 kinds '
                f'need the packed gate/up interleave, so weights transformed without it '
                f'would be laid out wrongly.'
            )


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu'):
        assert activation in ('swiglu', 'swigluoai', 'situ'), f'Unsupported activation `{activation}`'
        assert activation != 'situ' or mma_type == 'fp8xfp4', \
            '`situ` activation is supported only for `fp8xfp4` MegaMoE'
        _check_legacy_fp4_acts_env()
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation

        # Allocate a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            mma_type, activation,
            num_shared_experts
        )
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

        # Create input buffer views (as torch tensors, not tvm-ffi tensors).
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.shared_l1_acts, self.shared_l1_acts_sf,
         self.shared_l2_acts, self.shared_l2_acts_sf,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf,
         self.x_scales) = map(
            torch.from_dlpack, slice_input_buffers(self.buffer))

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 num_shared_experts: int = 0,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu') -> SymmBuffer:
    # Align token count
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    # Backward compat: derive `mma_type` from `use_fp8_dispatch` if provided
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts,
        mma_type=mma_type, activation=activation
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    # Unsqueeze for 2D
    assert t.dim() in (2, 3)
    squeeze_group_dim = t.dim() == 2
    if squeeze_group_dim:
        t = t.unsqueeze(0)

    # Transpose
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    result = torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))
    return result.squeeze(0) if squeeze_group_dim else result


def _interleave_weights_packed_fp4(t: torch.Tensor) -> torch.Tensor:
    assert t.dim() in (2, 3)
    squeeze_group_dim = t.dim() == 2
    if squeeze_group_dim:
        t = t.unsqueeze(0)

    g, n, *rest = t.shape
    half = n // 2
    assert half % 16 == 0
    gate = t[:, :half].reshape(g, half // 16, 16, *rest)
    up = t[:, half:].reshape(g, half // 16, 16, *rest)
    result = torch.cat([gate[:, :, 0::2], up[:, :, 0::2], gate[:, :, 1::2], up[:, :, 1::2]], dim=2)
    result = torch.empty_like(t).copy_(result.reshape(g, n, *rest))
    return result.squeeze(0) if squeeze_group_dim else result


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    # Unsqueeze for 2D
    assert sf.dtype == torch.int and sf.dim() in (2, 3)
    squeeze_group_dim = sf.dim() == 2
    if squeeze_group_dim:
        sf = sf.unsqueeze(0)

    # Transpose
    num_groups, mn, packed_sf_k = sf.shape
    assert mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    result = torch.empty_like(sf).copy_(result)
    return result.squeeze(0) if squeeze_group_dim else result


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu',
    mma_type: str = 'fp8xfp4'
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
           Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    assert activation in ('swiglu', 'swigluoai', 'situ'), f'Unsupported activation `{activation}`'
    assert activation != 'situ' or (
        isinstance(l1_weights, tuple) and isinstance(l2_weights, tuple)
    ), '`situ` activation is supported only for FP8xFP4 MegaMoE weights'
    if isinstance(l1_weights, tuple):
        # FP8/MXFP4/NVFP4: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP.
        interleave = _interleave_weights_packed_fp4 if mma_type in ('mxf4xmxf4', 'nvfp4xnvfp4') \
            else _interleave_weights
        l1_w = interleave(l1_weights[0])
        l1_sf = _transpose_sf_for_utccp(interleave(l1_weights[1]))
        l1_transformed = (l1_w, l1_sf)
        # L2: only transpose SF for UTCCP
        l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    else:
        # BF16: L1 interleave gate/up, L2 unchanged
        l1_transformed = _interleave_weights(l1_weights)
        l2_transformed = l2_weights
    return l1_transformed, l2_transformed



def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True,
                     use_x_scales: bool = False,
                     l1_alphas: Optional[torch.Tensor] = None,
                     l2_alphas: Optional[torch.Tensor] = None,
                     l2_act_scales: Optional[torch.Tensor] = None):
    if use_x_scales and sym_buffer.mma_type != 'nvfp4xnvfp4':
        raise ValueError(
            '`use_x_scales` is only supported for `nvfp4xnvfp4`; '
            f'got mma_type={sym_buffer.mma_type!r}'
        )

    (l1_weights_data, l1_weights_sf) = l1_weights
    (l2_weights_data, l2_weights_sf) = l2_weights
    (shared_l1_data, shared_l1_sf) = shared_l1_weights if shared_l1_weights is not None else (None, None)
    (shared_l2_data, shared_l2_sf) = shared_l2_weights if shared_l2_weights is not None else (None, None)
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights_data, l1_weights_sf,
        l2_weights_data, l2_weights_sf,
        shared_l1_data, shared_l1_sf,
        shared_l2_data, shared_l2_sf,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe, sym_buffer.mma_type,
        activation, activation_clamp,
        fast_math, use_x_scales, l1_alphas, l2_alphas, l2_act_scales
    )

def nvfp4_mega_moe(y: torch.Tensor,
                   l1_weights: Tuple[torch.Tensor, torch.Tensor],
                   l2_weights: Tuple[torch.Tensor, torch.Tensor],
                   sym_buffer: SymmBuffer,
                   shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                   shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                   cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                   activation: str = 'swiglu',
                   activation_clamp: Optional[float] = None,
                   fast_math: bool = True,
                   use_x_scales: bool = False,
                   l1_alphas: Optional[torch.Tensor] = None,
                   l2_alphas: Optional[torch.Tensor] = None,
                   l2_act_scales: Optional[torch.Tensor] = None):
    fp8_fp4_mega_moe(
        y, l1_weights, l2_weights, sym_buffer,
        shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats,
        recipe=(1, 1, 16),
        activation=activation, activation_clamp=activation_clamp,
        fast_math=fast_math, use_x_scales=use_x_scales,
        l1_alphas=l1_alphas, l2_alphas=l2_alphas, l2_act_scales=l2_act_scales
    )


def bf16_mega_moe(y: torch.Tensor,
                  l1_weights: torch.Tensor,
                  l2_weights: torch.Tensor,
                  sym_buffer: SymmBuffer,
                  shared_l1_weights: Optional[torch.Tensor] = None,
                  shared_l2_weights: Optional[torch.Tensor] = None,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True):
    _C.bf16_mega_moe(
        y,
        l1_weights,
        l2_weights,
        shared_l1_weights,
        shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        activation, activation_clamp,
        fast_math
    )


def get_block_m_for_mega_moe(num_ranks: int, num_experts: int,
                             num_max_tokens_per_rank: int, num_tokens: int,
                             num_topk: int, mma_type: str = 'fp8xfp4') -> int:
    """`BLOCK_M` the kernel will pick — callers need it to lay out shared-expert SFs."""
    return int(_C.get_block_m_for_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk, mma_type))


def mega_moe_pre_dispatch(x: torch.Tensor,
                          topk_idx: torch.Tensor,
                          topk_weights: torch.Tensor,
                          buf_x: torch.Tensor,
                          buf_x_sf: torch.Tensor,
                          buf_topk_idx: torch.Tensor,
                          buf_topk_weights: torch.Tensor,
                          num_tokens: int,
                          group_size: int = 32,
                          mma_type: str = 'fp8xfp4',
                          buf_x_scales: Optional[torch.Tensor] = None,
                          expert_scales: Optional[torch.Tensor] = None) -> None:
    _C.mega_moe_pre_dispatch(
        x, topk_idx, topk_weights,
        buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights,
        num_tokens, group_size, mma_type,
        buf_x_scales, expert_scales,
    )


def mega_moe_pre_dispatch_sm90(x: torch.Tensor,
                               topk_idx: torch.Tensor,
                               topk_weights: torch.Tensor,
                               buf_x: torch.Tensor,
                               buf_x_sf: torch.Tensor,
                               buf_topk_idx: torch.Tensor,
                               buf_topk_weights: torch.Tensor,
                               num_tokens: int,
                               group_size: int = 128,
                               routed_scaling_factor: float = 1.0) -> None:
    _C.mega_moe_pre_dispatch_sm90(
        x, topk_idx, topk_weights,
        buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights,
        num_tokens, group_size, float(routed_scaling_factor),
    )
