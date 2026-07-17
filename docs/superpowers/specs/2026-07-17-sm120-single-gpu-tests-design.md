# SM120 Single-GPU Test Selection

## Goal

Limit `sgl_deep_gemm/run_tests.sh` on SM120 GPUs, including RTX 5090, to the four single-GPU test files updated for SM120 by DeepGEMM PR #56.

## Test selection

When `nvidia-smi` reports compute capability with architecture major `12`, the runner executes only:

- `test_bf16.py`
- `test_einsum.py`
- `test_fp8_fp4.py`
- `test_attention.py`

It does not execute `test_hyperconnection.py`, `test_layout.py`, any Mega-MoE test, `test_lazy_init.py`, or `test_sanitizer.py`. Existing files excluded by this rule remain visible in the summary as skipped with an SM120-specific reason.

SM90 and SM100/SM103 test selection and behavior remain unchanged.

## Implementation

Define the four PR #56 files as an SM120-specific list. Select that list when `ARCH_MAJOR` equals `12`; otherwise use the existing single-GPU list. Add an SM120 branch to the Mega-MoE architecture dispatch before the existing Blackwell branch so those tests are recorded as skipped instead of invoked.

The existing lazy-init and sanitizer exclusions already do not invoke their files. Their skip records require no behavioral change.

## Verification

Use a shell harness with stubbed `nvidia-smi` and Python executables plus temporary test files to simulate compute capability 12.0. Assert that exactly the four intended test files are invoked and that all other test files are skipped. Repeat with compute capability 10.0 to confirm the pre-existing test selection remains intact. Run `bash -n` and `shellcheck` when available.
