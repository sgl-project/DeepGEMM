"""Stress SM120 BF16 stage reuse with identical inputs and fresh output buffers.

Run this without instrumentation; the existing BF16 tests provide sanitizer
coverage. Synchronizing every call can conceal scheduling-dependent failures.
"""

import unittest

import torch

import deep_gemm


class TestBF16Repeatability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
            raise unittest.SkipTest('SM120 GPU required')

    def test_nn_reuses_pipeline_stages_without_changing_output(self):
        previous = (
            deep_gemm.get_deterministic_algorithms()
            if hasattr(deep_gemm, 'get_deterministic_algorithms') else None
        )
        if previous is not None:
            deep_gemm.use_deterministic_algorithms(True)
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(20260914)
                a = torch.randn(908, 15360, device='cuda', dtype=torch.bfloat16)
                weight = torch.randn(3840, 15360, device='cuda', dtype=torch.bfloat16)
                b = weight.T
                reference = torch.empty(908, 3840, device='cuda', dtype=torch.bfloat16)
                deep_gemm.bf16_gemm_nn(a, b, reference)
                torch.cuda.synchronize()

                # Keep each batch's outputs alive and synchronize only after the
                # batch, so allocation reuse and pipelined launches remain exercised.
                for start in range(0, 1024, 32):
                    outputs = []
                    mismatches = []
                    for _ in range(32):
                        out = torch.empty_like(reference)
                        deep_gemm.bf16_gemm_nn(a, b, out)
                        outputs.append(out)
                        mismatches.append((out != reference).any())
                    changed = torch.stack(mismatches).cpu().tolist()
                    self.assertFalse(
                        any(changed),
                        f'BF16 output changed for identical inputs in calls '
                        f'{[start + i for i, value in enumerate(changed) if value]}',
                    )
        finally:
            if previous is not None:
                deep_gemm.use_deterministic_algorithms(previous)


if __name__ == '__main__':
    unittest.main()
