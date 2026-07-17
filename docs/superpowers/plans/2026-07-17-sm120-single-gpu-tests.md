# SM120 Single-GPU Test Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `sgl_deep_gemm/run_tests.sh` execute only the four PR #56 test files on SM120 GPUs while preserving SM90 and SM100/SM103 behavior.

**Architecture:** Keep the existing runner and add an architecture-specific selection layer around its single-GPU list. Add an SM120 branch before the existing Mega-MoE Blackwell dispatch so no Mega-MoE file is invoked on RTX 5090.

**Tech Stack:** Bash, stubbed command-line test harness, Git

## Global Constraints

- SM120 is identified by `ARCH_MAJOR=12` from `nvidia-smi` compute capability.
- SM120 runs only `test_bf16.py`, `test_einsum.py`, `test_fp8_fp4.py`, and `test_attention.py`.
- SM90 and SM100/SM103 behavior remains unchanged.
- Excluded existing tests appear in the summary as skipped with an SM120-specific reason.

---

### Task 1: Add SM120 test selection

**Files:**
- Create: `sgl_deep_gemm/tests/test_run_tests_sm120.sh`
- Modify: `sgl_deep_gemm/run_tests.sh:40-174`

**Interfaces:**
- Consumes: `ARCH_MAJOR`, `TESTS_DIR`, `run_test`, and `skip_test` from `run_tests.sh`.
- Produces: architecture-specific `SINGLE_GPU_TESTS` contents and an SM120 Mega-MoE skip branch.

- [ ] **Step 1: Write the failing shell regression test**

Create a harness that copies `run_tests.sh` into a temporary fake source tree, provides fake test files, and stubs `nvidia-smi` and Python. The Python stub must return an installed-wheel path for the import guard, fail the optional `deep_ep` probe, and append every invoked `.py` filename to `CALL_LOG`.

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/../run_tests.sh"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "${TMP_DIR}"' EXIT

make_stubs() {
  local case_dir="$1"
  mkdir -p "${case_dir}/bin"
  cat >"${case_dir}/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
if [ "${1:-}" = "-L" ]; then
  echo "GPU 0: fake GPU"
else
  echo "${FAKE_ARCH}"
fi
EOF
  cat >"${case_dir}/bin/python" <<'EOF'
#!/usr/bin/env bash
if [ "${1:-}" = "-c" ]; then
  if [[ "${2:-}" == *"import deep_gemm"* ]]; then
    printf '/opt/fake-wheel/deep_gemm/__init__.py'
    exit 0
  fi
  exit 1
fi
printf '%s\n' "$1" >>"${CALL_LOG}"
EOF
  chmod +x "${case_dir}/bin/nvidia-smi" "${case_dir}/bin/python"
}

run_case() {
  local arch="$1"
  local include_mega="$2"
  local case_dir="${TMP_DIR}/${arch/./_}"
  mkdir -p "${case_dir}/sgl_deep_gemm/tests"
  cp "${RUNNER}" "${case_dir}/sgl_deep_gemm/run_tests.sh"
  touch "${case_dir}/sgl_deep_gemm/tests/"{test_bf16.py,test_einsum.py,test_fp8_fp4.py,test_hyperconnection.py,test_layout.py,test_attention.py}
  if [ "${include_mega}" -eq 1 ]; then
    touch "${case_dir}/sgl_deep_gemm/tests/"{test_mega_moe.py,test_mega_moe_l1_fp4_accuracy.py,test_mega_moe_l1_sentinel.py,test_mega_moe_pre_dispatch.py,test_mega_moe_hopper.py,test_mega_moe_pre_dispatch_sm90.py,test_lazy_init.py,test_sanitizer.py}
  fi
  make_stubs "${case_dir}"
  CALL_LOG="${case_dir}/calls" FAKE_ARCH="${arch}" \
    PATH="${case_dir}/bin:${PATH}" PYTHON="${case_dir}/bin/python" \
    bash "${case_dir}/sgl_deep_gemm/run_tests.sh" "${case_dir}" >"${case_dir}/output"
}

run_case 12.0 1
expected_sm120=$'test_bf16.py\ntest_einsum.py\ntest_fp8_fp4.py\ntest_attention.py'
actual_sm120=$(<"${TMP_DIR}/12_0/calls")
[ "${actual_sm120}" = "${expected_sm120}" ]
for test_file in test_hyperconnection.py test_layout.py \
  test_mega_moe.py test_mega_moe_l1_fp4_accuracy.py test_mega_moe_l1_sentinel.py \
  test_mega_moe_pre_dispatch.py test_mega_moe_hopper.py test_mega_moe_pre_dispatch_sm90.py; do
  grep -Fq -- "SKIP ${test_file} (not supported on SM120)" "${TMP_DIR}/12_0/output"
done

run_case 10.0 0
expected_sm100=$'test_bf16.py\ntest_einsum.py\ntest_fp8_fp4.py\ntest_hyperconnection.py\ntest_layout.py\ntest_attention.py'
actual_sm100=$(<"${TMP_DIR}/10_0/calls")
[ "${actual_sm100}" = "${expected_sm100}" ]
```

Run the harness once with `FAKE_ARCH=12.0` and assert that `CALL_LOG` contains exactly:

```text
test_bf16.py
test_einsum.py
test_fp8_fp4.py
test_attention.py
```

Also assert that the output contains SM120 skip records for `test_hyperconnection.py`, `test_layout.py`, and each present Mega-MoE file. Run it again with `FAKE_ARCH=10.0`, create only the six existing single-GPU files, and assert that all six are invoked in their original order.

- [ ] **Step 2: Run the regression test to verify it fails**

Run:

```bash
bash sgl_deep_gemm/tests/test_run_tests_sm120.sh
```

Expected: exit nonzero because the current SM120 path invokes `test_hyperconnection.py`, `test_layout.py`, and present Blackwell Mega-MoE files.

- [ ] **Step 3: Implement the minimal selection change**

In `run_tests.sh`, update the architecture comment to include SM120. Retain the current six-file list as the default and define the four-file SM120 list. When `ARCH_MAJOR` is `12`, run the SM120 list and record existing default-only files as skipped with reason `not supported on SM120`; otherwise run the unchanged default list.

```bash
DEFAULT_SINGLE_GPU_TESTS=(
  test_bf16.py
  test_einsum.py
  test_fp8_fp4.py
  test_hyperconnection.py
  test_layout.py
  test_attention.py
)
SM120_SINGLE_GPU_TESTS=(
  test_bf16.py
  test_einsum.py
  test_fp8_fp4.py
  test_attention.py
)
SM120_UNSUPPORTED_SINGLE_GPU_TESTS=(
  test_hyperconnection.py
  test_layout.py
)

if [ "${ARCH_MAJOR}" -eq 12 ]; then
  SINGLE_GPU_TESTS=("${SM120_SINGLE_GPU_TESTS[@]}")
else
  SINGLE_GPU_TESTS=("${DEFAULT_SINGLE_GPU_TESTS[@]}")
fi

for t in "${SINGLE_GPU_TESTS[@]}"; do
  if [ -f "${TESTS_DIR}/${t}" ]; then
    run_test "${t}"
  else
    skip_test "${t}" "not present in this branch"
  fi
done

if [ "${ARCH_MAJOR}" -eq 12 ]; then
  for t in "${SM120_UNSUPPORTED_SINGLE_GPU_TESTS[@]}"; do
    [ -f "${TESTS_DIR}/${t}" ] && skip_test "${t}" "not supported on SM120"
  done
fi
```

Before the existing `ARCH_MAJOR -ge 10` Mega-MoE branch, add:

```bash
elif [ "${ARCH_MAJOR}" -eq 12 ]; then
  for t in "${MEGA_MOE_ALL[@]}"; do
    [ -f "${TESTS_DIR}/${t}" ] && skip_test "${t}" "not supported on SM120"
  done
```

- [ ] **Step 4: Run focused verification**

Run:

```bash
bash sgl_deep_gemm/tests/test_run_tests_sm120.sh
bash -n sgl_deep_gemm/run_tests.sh
bash -n sgl_deep_gemm/tests/test_run_tests_sm120.sh
git diff --check
```

Expected: every command exits zero. If `shellcheck` is installed, also run it on both shell files and require zero findings.

- [ ] **Step 5: Commit the implementation**

```bash
git add sgl_deep_gemm/run_tests.sh sgl_deep_gemm/tests/test_run_tests_sm120.sh
git commit -m "test: limit SM120 single-GPU coverage"
```
