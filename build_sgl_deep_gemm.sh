#!/usr/bin/env bash
#
# Build a wheel for the `sgl-deep-gemm` distribution.
#
# Distribution name: sgl-deep-gemm. Top-level import name: `deep_gemm`
# (so existing call sites like `import deep_gemm` in sglang keep working).
#
# Build flow:
#   1. Initialises submodules (cutlass, deep_jit) — same prerequisite as `bash build.sh`.
#   2. Stages the package layout under build/deep_gemm/ with the Python
#      sub-modules pulled from the source deep_gemm/ tree (utils, testing,
#      legacy, mega).
#   3. Reads the version string from sgl_deep_gemm/VERSION.
#   4. Pre-compiles the TORCH_LIBRARY `_C.so` extension and bundles it into the wheel.
#   5. Invokes `python -m build` to produce dist/*.whl.

set -euo pipefail

# Which interpreter builds _C.so decides which torch ABI _C.so is linked
# against: torch declares `c10_cuda_check_implementation` with a different
# signature in 2.13 (no CUDAErrorLogCapture* argument) than in 2.14, so a _C.so
# built against one is not importable under the other. Never let this be an
# accident of $PATH ordering -- override PYTHON_EXE, or make sure the first
# python3 on $PATH is the environment you intend to install into.
PYTHON_EXE="${PYTHON_EXE:-$(which python3 || which python)}"
ROOT_DIR=$(realpath "$(dirname "$0")")
BUILD_DIR="${ROOT_DIR}/build"
PKG_DIR="${BUILD_DIR}/deep_gemm"
DIST_DIR="${ROOT_DIR}/dist"

cd "$ROOT_DIR"

echo "--- Build interpreter ---"
echo "PYTHON_EXE: $PYTHON_EXE"
"$PYTHON_EXE" -c '
import sys, torch
print(f"python    : {sys.version.split()[0]}")
print(f"torch     : {torch.__version__}")
print(f"torch cuda: {torch.version.cuda}")
' || { echo "Error: $PYTHON_EXE cannot import torch — it must be the environment you will install into." >&2; exit 1; }

if [[ -z "${DG_SKIP_TORCH_CHECK:-}" ]]; then
    if ! "$PYTHON_EXE" -c '
import torch, re, sys, subprocess, glob, os
# Find the signature _C.so will be linked against, so a torch/pip mismatch is
# reported here rather than as "undefined symbol" at import time.
lib = os.path.join(os.path.dirname(torch.__file__), "lib", "libc10_cuda.so")
out = subprocess.run(["nm", "-D", "--defined-only", lib],
                     capture_output=True, text=True).stdout
want_new = "CUDAErrorLogCaptureE" in out
print("torch ABI: "
      + (">=2.14 (c10_cuda_check_implementation takes CUDAErrorLogCapture*)" if want_new
         else "<2.14 (c10_cuda_check_implementation has no CUDAErrorLogCapture*)"))
'; then
        echo "Note: could not probe the torch ABI via libc10_cuda.so (nm missing?); continuing." >&2
    fi
fi

if [[ ! -f "setup.py" || ! -d "sgl_deep_gemm" || ! -d "deep_gemm" || ! -d "csrc" ]]; then
    echo "Error: Run from the DeepGEMM project root." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# CUDA prerequisites. Checked up front so a misconfigured toolkit surfaces as
# an instruction rather than as a strange `nvcc` or linker error much later.
# ---------------------------------------------------------------------------
echo "--- CUDA toolchain ---"

CUDA_HOME_RESOLVED="$("$PYTHON_EXE" - <<'PY'
import os, shutil, sys
from pathlib import Path
home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if not home:
    nvcc = shutil.which("nvcc")
    home = str(Path(nvcc).parent.parent) if nvcc else "/usr/local/cuda"
print(home)
PY
)" || { echo "Error: could not determine a CUDA home." >&2; exit 1; }

if [[ ! -d "$CUDA_HOME_RESOLVED" ]]; then
    cat >&2 <<EOF
Error: CUDA_HOME resolves to '$CUDA_HOME_RESOLVED', which is not a directory.

  export CUDA_HOME=/path/to/cuda        # must contain bin/nvcc
EOF
    exit 1
fi
echo "CUDA_HOME: $CUDA_HOME_RESOLVED"

if [[ ! -x "${CUDA_HOME_RESOLVED}/bin/nvcc" ]]; then
    cat >&2 <<EOF
Error: no executable 'nvcc' at ${CUDA_HOME_RESOLVED}/bin/nvcc.

  DeepGEMM compiles its host layer and JITs its kernels, so a full CUDA
  toolkit is required (a runtime-only install is not enough).

  export CUDA_HOME=/path/to/cuda        # the one with bin/nvcc
EOF
    exit 1
fi
echo "nvcc      : $("${CUDA_HOME_RESOLVED}/bin/nvcc" --version | tail -1)"

# SM110 is only accepted by CUDA 12.9+. Catch a toolkit that is too old here,
# where the message can still name the real problem.
if ! "${CUDA_HOME_RESOLVED}/bin/nvcc" --list-gpu-arch 2>/dev/null | grep -q 'compute_11[0-9]'; then
    cat >&2 <<EOF
Warning: ${CUDA_HOME_RESOLVED}/bin/nvcc does not list compute_110 among its
         supported architectures. Building for SM110 (Jetson Thor) needs
         CUDA 12.9 or newer; older toolkits will fail on the device code.
         Continuing anyway -- ignore this if you are targeting SM90/SM100.
EOF
fi

# The runtime JIT needs the CUDA *device* headers (curand_kernel.h etc.).
# On Jetson these are sometimes installed to a different prefix than nvcc.
if [[ ! -f "${CUDA_HOME_RESOLVED}/include/curand_kernel.h" ]]; then
    cat >&2 <<EOF
Warning: ${CUDA_HOME_RESOLVED}/include/curand_kernel.h is missing.

  The host layer will build, but the runtime JIT needs the CUDA device
  headers and will fail later with a confusing compile error. Fix one of:

    * point CUDA_HOME at a prefix that has BOTH bin/nvcc and include/
      (a symlink farm combining the two is fine), or
    * add the device-header directory to CPATH:
        export CPATH=/path/to/cuda/include:\$CPATH
EOF
fi
echo "headers   : $([[ -f "${CUDA_HOME_RESOLVED}/include/curand_kernel.h" ]] && echo present || echo MISSING)"

echo "--- Initialising submodules ---"
git submodule update --init --recursive

echo "--- Linking CUTLASS headers into deep_gemm/include ---"
ln -sfn "${ROOT_DIR}/third-party/cutlass/include/cutlass" "${ROOT_DIR}/deep_gemm/include/cutlass"
ln -sfn "${ROOT_DIR}/third-party/cutlass/include/cute" "${ROOT_DIR}/deep_gemm/include/cute"

echo "--- Preparing build directory ---"
rm -rf "$BUILD_DIR"
mkdir -p "$PKG_DIR"

cp sgl_deep_gemm/LICENSE sgl_deep_gemm/README.md sgl_deep_gemm/pyproject.toml "$BUILD_DIR/"
cp sgl_deep_gemm/__init__.py sgl_deep_gemm/cuda_helpers.py sgl_deep_gemm/build_ext.py \
   sgl_deep_gemm/doctor.py "$PKG_DIR/"

# `__init__.py` imports `.utils`, `.testing`, `.legacy`, `.mega` — pulled from
# the existing deep_gemm/ tree.
for sub in utils testing legacy mega; do
    cp -r "deep_gemm/${sub}" "$PKG_DIR/"
done

# Headers required by the runtime JIT (same set the deep_gemm wheel ships).
mkdir -p "$PKG_DIR/include"
cp -r "${ROOT_DIR}/deep_gemm/include/deep_gemm" "$PKG_DIR/include/deep_gemm"
cp -r "${ROOT_DIR}/third-party/cutlass/include/cute" "$PKG_DIR/include/cute"
cp -r "${ROOT_DIR}/third-party/cutlass/include/cutlass" "$PKG_DIR/include/cutlass"
cp -r "${ROOT_DIR}/third-party/deep_jit/include/deep_jit" "$PKG_DIR/include/deep_jit"

echo "--- Reading version from sgl_deep_gemm/VERSION ---"
if [[ ! -f "sgl_deep_gemm/VERSION" ]]; then
    echo "Error: sgl_deep_gemm/VERSION is missing — create it with the desired version (e.g. 0.0.1)." >&2
    exit 1
fi
# Strip surrounding whitespace; the file is the single source of truth.
tr -d '[:space:]' < sgl_deep_gemm/VERSION > "$PKG_DIR/VERSION"
echo "Version: $(cat "$PKG_DIR/VERSION")"

echo "--- Compiling _C.so ---"
"$PYTHON_EXE" sgl_deep_gemm/build_ext.py "$ROOT_DIR" "$PKG_DIR/_C.so"

echo "--- Installing build frontend ---"
"$PYTHON_EXE" -m pip install --quiet --upgrade build

echo "--- Building wheel ---"
mkdir -p "$DIST_DIR"
"$PYTHON_EXE" -m build --wheel "$BUILD_DIR" --outdir "$DIST_DIR"

echo "--- Done ---"
ls -lh "$DIST_DIR"/sgl_deep_gemm-*.whl 2>/dev/null || ls -lh "$DIST_DIR"/sgl-deep-gemm-*.whl 2>/dev/null || ls -lh "$DIST_DIR"/sgl_deep_gemm*.whl
