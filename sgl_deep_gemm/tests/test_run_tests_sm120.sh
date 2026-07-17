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
grep -Fq -- "SKIP test_lazy_init.py (" "${TMP_DIR}/12_0/output"
grep -Fq -- "SKIP test_sanitizer.py (" "${TMP_DIR}/12_0/output"

run_case 10.3 1
expected_sm100=$'test_bf16.py\ntest_einsum.py\ntest_fp8_fp4.py\ntest_hyperconnection.py\ntest_layout.py\ntest_attention.py\ntest_mega_moe_l1_fp4_accuracy.py\ntest_mega_moe_l1_sentinel.py\ntest_mega_moe_pre_dispatch.py'
actual_sm100=$(<"${TMP_DIR}/10_3/calls")
[ "${actual_sm100}" = "${expected_sm100}" ]

run_case 9.0 1
expected_sm90=$'test_bf16.py\ntest_einsum.py\ntest_fp8_fp4.py\ntest_hyperconnection.py\ntest_layout.py\ntest_attention.py\ntest_mega_moe_hopper.py\ntest_mega_moe_pre_dispatch_sm90.py'
actual_sm90=$(<"${TMP_DIR}/9_0/calls")
[ "${actual_sm90}" = "${expected_sm90}" ]

echo "run_tests.sh architecture selection checks passed"
