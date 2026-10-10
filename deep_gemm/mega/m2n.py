from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .. import _C
from ..utils.math import align


class MegaMoEM2NSymmBuffer:
    """Symmetric storage shared by AG ranks followed by EG ranks.

    All ranks construct and destroy buffers in the same order. Keep the buffer
    alive through all GPU work and graph replays that access it; CUDA IPC needs
    collective destroy, not garbage collection, while the group is alive.
    A failed construction invalidates the group and requires all participants
    to restart; partial-group recovery is not supported.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        ag_size: int,
        eg_size: int,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        mma_type: str = "fp8xfp4",
    ) -> None:
        if mma_type not in ("fp8xfp4", "nvfp4xnvfp4"):
            raise ValueError(f"Unsupported M2N MMA type: {mma_type}")
        self.mma_type = mma_type
        self.group = group
        self.ag_size = int(ag_size)
        self.eg_size = int(eg_size)
        if group.size() != self.ag_size + self.eg_size:
            raise RuntimeError(
                f"group size {group.size()} does not match "
                f"ag_size + eg_size ({self.ag_size + self.eg_size})"
            )

        self.num_experts = int(num_experts)
        self.num_topk = int(num_topk)
        self.hidden = int(hidden)
        self.intermediate_hidden = int(intermediate_hidden)
        # Keep FP32 routing-weight regions 16-byte aligned. BLOCK_M alignment
        # belongs to the expert pools, not the per-source input capacity.
        self.num_max_tokens_per_rank = align(int(num_max_tokens_per_rank), 4)

        num_bytes, num_max_pool_tokens, slice_input_buffers = (
            _C.get_symm_buffer_size_for_mega_moe_m2n(
                self.ag_size,
                self.eg_size,
                self.num_experts,
                self.num_max_tokens_per_rank,
                self.num_topk,
                self.hidden,
                self.intermediate_hidden,
                self.mma_type == "nvfp4xnvfp4",
            )
        )
        self.num_max_pool_tokens = int(num_max_pool_tokens)
        self.buffer, local_handle = map(
            torch.from_dlpack, _C.allocate_m2n_buffer(int(num_bytes))
        )
        # Exchange handles within this group, independently of other modules'
        # Torch symmetric-memory rendezvous history.
        local_handle = local_handle.to(device=self.buffer.device)
        handles = torch.empty(
            (group.size() * local_handle.numel(),),
            dtype=local_handle.dtype,
            device=self.buffer.device,
        )
        dist.all_gather_into_tensor(handles, local_handle, group=group)
        handles = handles.cpu().view(group.size(), -1)
        self._peer_buffers = {
            rank: torch.from_dlpack(_C.import_m2n_buffer(handle))
            for rank, handle in enumerate(handles)
            if rank != group.rank()
        }
        self._buffer_ptrs = [
            (
                self.buffer if rank == group.rank() else self._peer_buffers[rank]
            ).data_ptr()
            for rank in range(group.size())
        ]
        self.buffer.zero_()
        _C.initialize_symm_buffer_for_mega_moe_m2n(self.buffer, self.group.size())
        self.group.barrier().wait()
        torch.cuda.synchronize()

        (
            self.x,
            self.x_sf,
            self.topk_idx,
            self.topk_weights,
            self.l1_acts,
            self.l1_acts_sf,
            self.l2_acts,
            self.l2_acts_sf,
        ) = map(torch.from_dlpack, slice_input_buffers(self.buffer))

    @property
    def buffer_ptrs(self) -> list[int]:
        return list(self._buffer_ptrs)

    @property
    def rank(self) -> int:
        return int(self.group.rank())

    def destroy(self) -> None:
        """Collectively release peer mappings before their owning allocations."""
        if self.buffer is None:
            return
        torch.cuda.synchronize(self.buffer.device)
        self.group.barrier().wait()
        self._peer_buffers.clear()
        self.group.barrier().wait()
        self._buffer_ptrs = []
        self.x = None
        self.x_sf = None
        self.topk_idx = None
        self.topk_weights = None
        self.l1_acts = None
        self.l1_acts_sf = None
        self.l2_acts = None
        self.l2_acts_sf = None
        self.buffer = None
        self.group = None


def get_symm_buffer_for_mega_moe_m2n(
    group: dist.ProcessGroup,
    *,
    ag_size: int,
    eg_size: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    mma_type: str = "fp8xfp4",
) -> MegaMoEM2NSymmBuffer:
    return MegaMoEM2NSymmBuffer(
        group,
        ag_size=ag_size,
        eg_size=eg_size,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        mma_type=mma_type,
    )


def mega_moe_m2n_ag(
    y: torch.Tensor,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sym_buffer: MegaMoEM2NSymmBuffer,
    *,
    output_scale: float = 1.0,
    num_sms: int = 0,
    input_scale: Optional[torch.Tensor] = None,
) -> None:
    """Run fused AG quantize, dispatch, combine, and output reduction.

    NVFP4 requires a CUDA FP32 scalar ``input_scale`` multiplying the input
    before quantization; its inverse must be folded into EG ``l1_alphas``.
    """
    _C.mega_moe_m2n_ag(
        y,
        hidden_states,
        topk_ids,
        topk_weights,
        sym_buffer.buffer,
        sym_buffer.buffer_ptrs,
        sym_buffer.rank,
        sym_buffer.ag_size,
        sym_buffer.eg_size,
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        sym_buffer.intermediate_hidden,
        float(output_scale),
        int(num_sms),
        sym_buffer.mma_type == "nvfp4xnvfp4",
        input_scale,
    )


def make_m2n_weight_descs(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    *,
    num_experts_per_rank: int,
    hidden: int,
    intermediate_hidden: int,
    mma_type: str = "fp8xfp4",
) -> torch.Tensor:
    """Build one layer's config-independent ``[4, 128]`` descriptor blob."""
    return torch.from_dlpack(
        _C.make_m2n_weight_descs(
            l1_weights[0],
            l1_weights[1],
            l2_weights[0],
            l2_weights[1],
            int(num_experts_per_rank),
            int(hidden),
            int(intermediate_hidden),
            mma_type == "nvfp4xnvfp4",
        )
    )


def mega_moe_m2n_eg(
    weight_descs: torch.Tensor,
    sym_buffers: Sequence[MegaMoEM2NSymmBuffer],
    *,
    expected_num_tokens_per_rank: int,
    num_experts_per_rank: int,
    use_fp8_weights: bool = True,
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    num_sms: int = 0,
    l1_alphas: Optional[torch.Tensor] = None,
    l2_alphas: Optional[torch.Tensor] = None,
    l2_act_scales: Optional[torch.Tensor] = None,
) -> None:
    """Run persistent EG receive, expert MLP, and response for one to four lanes.

    NVFP4 requires contiguous CUDA FP32 tables: ``l1_alphas`` [layers, experts, 2]
    for gate/up, and ``l2_alphas`` / ``l2_act_scales`` [layers, experts]. Alphas
    multiply the FP32 accumulators; ``l2_act_scales`` multiplies SwiGLU before
    quantization, and its inverse must be folded into ``l2_alphas``.
    """
    if not 1 <= len(sym_buffers) <= 4:
        raise ValueError("sym_buffers must contain between one and four lanes")

    first = sym_buffers[0]
    for sym_buffer in sym_buffers[1:]:
        if (
            sym_buffer.group is not first.group
            or sym_buffer.ag_size != first.ag_size
            or sym_buffer.eg_size != first.eg_size
            or sym_buffer.mma_type != first.mma_type
            or sym_buffer.num_experts != first.num_experts
            or sym_buffer.num_topk != first.num_topk
            or sym_buffer.hidden != first.hidden
            or sym_buffer.intermediate_hidden != first.intermediate_hidden
            or sym_buffer.num_max_tokens_per_rank != first.num_max_tokens_per_rank
        ):
            raise ValueError("all M2N lanes must use the same layout and group")

    ffi_buffers = list(sym_buffers) + [first] * (4 - len(sym_buffers))
    _C.mega_moe_m2n_eg(
        weight_descs,
        *(sym_buffer.buffer for sym_buffer in ffi_buffers),
        *(sym_buffer.buffer_ptrs for sym_buffer in ffi_buffers),
        len(sym_buffers),
        first.rank,
        first.ag_size,
        first.eg_size,
        first.num_max_tokens_per_rank,
        first.num_experts,
        first.num_topk,
        first.hidden,
        first.intermediate_hidden,
        int(expected_num_tokens_per_rank),
        int(num_experts_per_rank),
        bool(use_fp8_weights),
        activation_clamp,
        bool(fast_math),
        int(num_sms),
        first.mma_type == "nvfp4xnvfp4",
        l1_alphas,
        l2_alphas,
        l2_act_scales,
    )
