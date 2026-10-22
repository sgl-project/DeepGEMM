"""GEMM epilogue classes for the `epilogue` argument of the GEMM APIs.

`alpha` and the FP8 `(d, sfd)` output pair are the shorthand for `Alpha` and unclamped `FP8CastWithClamp`.
Each class crosses the `torch.ops` boundary as `(kind, scalar, sfd, count, amax_eps)`; the kinds mirror
`make_epilogue_class` in `csrc/torch_library_api.cpp`.
"""
from typing import Optional, Tuple

import torch


class Epilogue:
    _kind = 0

    def _encode(self) -> Tuple[int, Optional[float], Optional[torch.Tensor], Optional[torch.Tensor], Optional[float]]:
        return self._kind, None, None, None, None


class Identity(Epilogue):
    _kind = 1


class Alpha(Epilogue):
    _kind = 2

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def _encode(self):
        return self._kind, self.alpha, None, None, None


class FP8CastWithClamp(Epilogue):
    """
    Cast BF16-rounded GEMM output to FP8, optionally clamping before scaling.
    `clamped_count` is an optional single-element CUDA uint32 tensor; the kernel writes the total asynchronously.
    """
    # The default floor preserves existing results; 1e-10 matches SGLang.
    _kind = 3

    def __init__(self, sfd: torch.Tensor, clamp_value: Optional[float] = None,
                 clamped_count: Optional[torch.Tensor] = None, amax_eps: float = 1e-4) -> None:
        self.sfd, self.clamp_value, self.clamped_count = sfd, clamp_value, clamped_count
        self.amax_eps = float(amax_eps)

    def _encode(self):
        return self._kind, None if self.clamp_value is None else float(self.clamp_value), self.sfd, self.clamped_count, self.amax_eps


class BF16StochasticRounding(Epilogue):
    _kind = 4


def encode(epilogue: Optional[Epilogue]) -> Tuple[int, Optional[float], Optional[torch.Tensor], Optional[torch.Tensor], Optional[float]]:
    return (0, None, None, None, None) if epilogue is None else epilogue._encode()
