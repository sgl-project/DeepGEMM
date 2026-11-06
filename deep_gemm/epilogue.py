"""GEMM epilogue classes for the `epilogue` argument of the GEMM APIs.

`alpha` and the FP8 `(d, sfd)` output pair are the shorthand for `Alpha` and unclamped `FP8CastWithClamp`.
Each class crosses the `torch.ops` boundary as `(kind, scalar, sfd, count)`; the kinds mirror
`make_epilogue_class` in `csrc/torch_library_api.cpp`.
"""
from typing import Optional, Tuple

import torch


class Epilogue:
    _kind = 0

    def _encode(self) -> Tuple[int, Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._kind, None, None, None


class Identity(Epilogue):
    _kind = 1


class Alpha(Epilogue):
    _kind = 2

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def _encode(self):
        return self._kind, self.alpha, None, None


class FP8CastWithClamp(Epilogue):
    """
    Cast BF16-rounded GEMM output to FP8, optionally clamping before scaling.
    `clamped_count` is an optional single-element CUDA uint32 tensor; the kernel writes the total asynchronously.
    """
    _kind = 3

    def __init__(self, sfd: torch.Tensor, clamp_value: Optional[float] = None,
                 clamped_count: Optional[torch.Tensor] = None) -> None:
        self.sfd, self.clamp_value, self.clamped_count = sfd, clamp_value, clamped_count

    def _encode(self):
        return self._kind, None if self.clamp_value is None else float(self.clamp_value), self.sfd, self.clamped_count


class BF16StochasticRounding(Epilogue):
    _kind = 4


def encode(epilogue: Optional[Epilogue]) -> Tuple[int, Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    return (0, None, None, None) if epilogue is None else epilogue._encode()
