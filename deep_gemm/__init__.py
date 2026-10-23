import atexit
import os
import torch

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# Configs
from . import _C
from ._C import (
    shutdown,
    set_num_sms,
    get_num_sms,
    set_tc_util,
    get_tc_util,
    set_ignore_compile_dims,
    set_block_size_multiple_of,
    set_pdl,
    get_pdl,
    use_deterministic_algorithms,
)

# cuBLASLt Kernels
from ._C import (
    cublaslt_gemm_nt, cublaslt_gemm_nn,
    cublaslt_gemm_tn, cublaslt_gemm_tt,
    cublaslt_nvfp4_gemm_nt,
    batched_syrk, batched_symm,
)

# DeepGEMM Kernels
from ._C import (
    # FP8-FP4 GEMMs (accept any FP8/FP4 operand combination)
    fp8_fp4_gemm_nt, fp8_fp4_gemm_nn,
    fp8_fp4_gemm_tn, fp8_fp4_gemm_tt,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
    m_grouped_fp8_fp4_gemm_nn_contiguous,
    m_grouped_fp8_fp4_gemm_nt_masked,
    # FP8/FP4 GEMMs (alias)
    fp4_gemm_nt,
    fp8_gemm_nt, fp8_gemm_nn,
    fp8_gemm_tn, fp8_gemm_tt,
    m_grouped_fp4_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nn_contiguous,
    m_grouped_fp4_gemm_nt_masked,
    m_grouped_fp8_gemm_nt_masked,
    k_grouped_fp4_gemm_nt_contiguous,
    k_grouped_fp8_gemm_nt_contiguous,
    k_grouped_fp8_gemm_tn_contiguous,
    # BF16 GEMMs
    bf16_gemm_nt, bf16_gemm_nn,
    bf16_gemm_tn, bf16_gemm_tt,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nn_contiguous,
    m_grouped_bf16_gemm_nt_masked,
    k_grouped_bf16_gemm_tn_contiguous,
    # MegaGate kernels
    bf16_mega_gate,
    # Einsum kernels
    einsum,
    fp8_einsum,
    # Attention kernels
    get_mqa_logits_metadata,
    fp8_fp4_mqa_logits,
    get_paged_mqa_logits_metadata,
    fp8_fp4_paged_mqa_logits,
    get_sparse_mqa_logits_metadata,
    get_paged_sparse_mqa_logits_metadata,
    fp8_fp4_sparse_mqa_logits,
    fp8_fp4_paged_sparse_mqa_logits,
    # Hyperconnection kernels
    mega_mhc,
    tf32_hc_prenorm_gemm,
    # Layout kernels
    transform_sf_into_required_layout,
    # MegaMoE
    get_block_m_for_mega_moe,
)

# Mega kernels
from .mega import (
    SymmBuffer,
    get_symm_buffer_for_mega_moe,
    transform_weights_for_mega_moe,
    fp8_fp4_mega_moe,
    bf16_mega_moe,
)

# Epilogue classes
from ._C import epilogue

# Locality domain helpers
from . import locality_domain
from .locality_domain import (
    get_num_locality_domains,
    is_localization_available,
    localize,
    is_localized,
    destroy_localizer,
)

# Some utils
from . import testing
from . import utils
from .utils import *

# Legacy Triton kernels for A100
try:
    from . import legacy
except Exception as e:
    print(f'Failed to load legacy DeepGEMM A100 Triton kernels: {e}')

# Initialize CPP modules
_C.init(os.path.dirname(os.path.abspath(__file__)))
# Native cuBLASLt exit callbacks can free handle storage before C++ static destructors.
# Release handles and workspaces while Python, PyTorch and CUDA are still available.
atexit.register(shutdown)

__version__ = '2.8.2'


# ---------------------------------------------------------------------------
# Capability query
# ---------------------------------------------------------------------------
def supported_arch_majors():
    """CUDA architecture majors this installed `_C.so` accepts, e.g. `(9, 10, 11, 12)`.

    Answered by the compiled binary, so a stale `_C.so` under a fresh Python side
    reports what it can actually do. No CUDA context is created.
    """
    return tuple(_C.supported_arch_majors())


# Per-feature support state, as reported by the compiled binary
# (csrc/utils/arch_traits.hpp). 0 = unsupported, 1 = enabled and tested on that
# arch, 2 = implementation exists but no test has run on that arch.
_FEATURE_SUPPORT = ('unsupported', 'validated', 'unvalidated')


def arch_feature_status():
    """`{name: {'sm100': state, 'sm110': state}}` for every gated feature.

    SM110 is answered for its real arch major (11); nothing is mapped onto 10.
    This separates "SM110 shares the SM100 kernel family" from "this newer
    variant has actually been run on SM110".
    """
    return {
        name: {'sm100': _FEATURE_SUPPORT[sm100], 'sm110': _FEATURE_SUPPORT[sm110]}
        for name, sm100, sm110 in _C.arch_feature_status()
    }


def supports_feature(arch, feature: str) -> bool:
    """Whether `feature` is enabled and validated for `arch` (an int or e.g. 'sm110')."""
    major = _arch_major(arch)
    status = arch_feature_status().get(feature)
    if status is None:
        return False
    if major == 10:
        return status['sm100'] == 'validated'
    if major == 11:
        return status['sm110'] == 'validated'
    return False


def _arch_major(arch) -> int:
    """Normalise an architecture to its major number without mapping SM110 onto SM100."""
    if isinstance(arch, str):
        value = arch.strip().lower()
        if value.startswith('sm'):
            value = value[2:]
        value = value.lstrip('_')
        if not value.isdigit():
            raise ValueError(f'unrecognised architecture {arch!r}; expected e.g. "sm110", "sm_110" or 11')
        number = int(value)
    elif isinstance(arch, int):
        number = arch
    else:
        raise TypeError(f'architecture must be an int or a str, got {type(arch).__name__}')
    return number // 10 if number >= 90 else number


def supports_arch(arch) -> bool:
    """Whether this build accepts `arch`, given as an int or an `sm`-style name.

    `supports_arch('sm110')` / `supports_arch(110)` / `supports_arch(11)` ask the
    same question: is SM110 (arch major 11) a supported architecture here?
    """
    return _arch_major(arch) in supported_arch_majors()

