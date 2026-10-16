import ctypes

import torch

from . import _C
from .utils import mlopart as _mlopart

# `_C` has no CPython dependency, so MLOPart (which needs Python IPC) is reached through these
# `ctypes` callbacks; they must stay referenced for as long as the library may call them
_UUID_SIZE = 16
_IsAvailable = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_uint8))
_CreateMemory = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_int)
_Release = ctypes.CFUNCTYPE(None)


def _uuid(ptr) -> bytes:
    return ctypes.string_at(ptr, _UUID_SIZE)


def _is_available(uuid_ptr) -> bool:
    try:
        return _mlopart.is_available(_uuid(uuid_ptr))
    except Exception as ex:
        print(f'DeepGEMM MLOPart probe failed: {ex}')
        return False


# A callback cannot raise through C++, so its exception is kept and re-raised by the allocating wrapper
_callback_error = None


def _create_memory(uuid_ptr, num_bytes, domain_idx) -> int:
    global _callback_error
    try:
        return _mlopart.create_memory(_uuid(uuid_ptr), num_bytes, domain_idx)
    except Exception as ex:
        _callback_error = ex
        return -1


def _allocate(op, *args) -> torch.Tensor:
    global _callback_error
    _callback_error = None
    try:
        return op(*args)
    except RuntimeError:
        error, _callback_error = _callback_error, None
        if error is not None:
            raise error from None
        raise


_callbacks = (_IsAvailable(_is_available), _CreateMemory(_create_memory), _Release(_mlopart.release))
_C.set_mlopart_callbacks(*(ctypes.cast(f, ctypes.c_void_p).value for f in _callbacks))

get_num_locality_domains = _C.get_num_locality_domains
get_granularity = _C.get_locality_domain_granularity
is_localization_available = _C.is_localization_available
is_localized = _C.is_localized
release_mlopart = _C.release_mlopart
get_sm_locality_domains = _C.get_sm_locality_domains
get_balanced_sm_locality_domains = _C.get_balanced_sm_locality_domains


def empty(sizes, dtype: torch.dtype, domain_idx: int) -> torch.Tensor:
    """`torch.empty(sizes, dtype=dtype)` in the locality domain `domain_idx`"""
    return _allocate(_C.locality_domain_empty, list(sizes), dtype, domain_idx)


def empty_per_domain(sizes, dtype: torch.dtype) -> torch.Tensor:
    """`torch.empty((num_domains, *sizes), dtype=dtype)`, with slice `d` in domain `d`"""
    return _allocate(_C.locality_domain_empty_per_domain, list(sizes), dtype)


def localize(t: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """
    Distribute the weights across locality domains: `(*B, N, K)` -> `(num_domains, *B, N / num_domains, K)`.
    """
    num_domains, dim = get_num_locality_domains(), dim % t.dim()
    assert t.size(dim) % num_domains == 0, f'Weights `{tuple(t.shape)}` must slice into {num_domains} parts along dim {dim}'
    localized = empty_per_domain((*t.shape[:dim], t.size(dim) // num_domains, *t.shape[dim + 1:]), t.dtype)
    localized.copy_(t.unflatten(dim, (num_domains, -1)).movedim(dim, 0))
    return localized


def destroy_localizer() -> None:
    """
    Quit the allocator backed by MLOPart. The allocated tensors stay valid.
    """
    release_mlopart()
