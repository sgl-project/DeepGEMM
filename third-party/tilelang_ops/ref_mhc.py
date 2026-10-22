"""Compatibility for the release workflow's Mega mHC dependency check.

Mega mHC now uses the PyTorch reference in the wheel test instead of the
optional TileKernels baseline. Keep the old readiness probe tied to that
reference so existing release workflows can still validate its imports.
"""

import importlib.util
from pathlib import Path


def has_baseline() -> bool:
    reference_path = Path(__file__).resolve().parents[2] / 'sgl_deep_gemm' / 'tests' / 'test_mega_mhc.py'
    try:
        spec = importlib.util.spec_from_file_location('_deep_gemm_mhc_reference', reference_path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        return callable(reference.mhc_reference)
    except Exception as error:
        print(f'Failed to load Mega mHC reference: {error}')
        return False
