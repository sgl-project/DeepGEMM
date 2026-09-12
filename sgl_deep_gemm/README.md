
## Introduction

sgl-deep-gemm is a pypi package built from SGLang's customized branch of DeepGemm. Comparing with origina DeepGemm, it supports the following features to better support SGLang:
1. ABI support: with the help of tvm-ffi wrappers, a single wheel can run on different python versions.
2. pypi support: easy installation with `pip install sgl-deep-gemm`. No need to manually search for wheel links.
3. Fast iteration: add custom kernels and bump versions at no time.

## Usage
Build requirements include CUDA 12.9 or newer, a C++20 compiler with `std::format` support (for example GCC 13), and the elfutils development headers (`apt-get install libdw-dev` on Debian/Ubuntu). Initialize the CUTLASS and DeepJIT submodules with `git submodule update --init --recursive`.

To build it locally, run `bash build_sgl_deep_gemm.sh`, then pip install the wheel generated under `dist`.

K-grouped FP8 GEMM and its scale-packing helper preserve compact per-group scaling factors by default. Pass `use_padded_sf_layout=True` to both packing and GEMM when packed scales cover each group's padded K extent. The flag is appended to existing arguments, so existing positional calls retain their behavior.

Run wheel validation with `bash sgl_deep_gemm/run_tests.sh`. The Mega MoE reference comparisons require DeepEP with `ElasticBuffer`; Mega Gate and Mega mHC also require TileKernels and TileLang. Install the validation extras with `pip install 'sgl-deep-gemm[dev]'` (TileLang 0.1.9 and TileKernels 1.0.0). The runner enables NVML-based CUDA discovery to preserve fork compatibility while retaining TVM FFI's DLPack fast path. It runs legacy, lazy-init, and compute-sanitizer checks; `--skip-sanitizer` is available for ordinary development runs. Distributed MegaMoE runs include separate Hopper numerical accuracy coverage; the sanitizer's automatic discovery covers single-process tests and reports distributed entrypoints separately.

For the release gate, run `bash sgl_deep_gemm/run_tests.sh --release`. This opt-in profile targets a run under one hour on Hopper and Blackwell by reducing attention case counts and the set of functions run under Compute Sanitizer. Runtime still depends on the GPU, compilation cache, and installed tools; the default command retains the full suite. The runner prints the profile and each selected/available attention case count.

| GPU architecture | Dense MQA | Paged MQA | Sparse MQA |
| --- | ---: | ---: | ---: |
| Hopper (SM90) | 8 / 192 | 8 / 24 | unsupported |
| Blackwell (SM100/SM103) | 36 / 2304 | 72 / 4320 | 12 / 161 |
| Blackwell (SM120) | 16 / 256 | 16 / 48 | unsupported |

All selected tuples come from the original enumerators and run their original input generation, reference comparisons, tolerances, repeated-call assertions, metadata checks, and benchmarks. Dense selection keeps every format/output/weights dtype pair and all four compression/CP combinations, rotates head counts and dimensions, and includes one long-KV and one large-Q case per format. Paged selection uses four cases per normal/varlen, format, and dtype pair, covering all page sizes and both NextN or per-request token limits. It uses batch size 256 and includes one long-KV case per format. Sparse selection keeps 12 existing small-Q cases covering both formats, contiguous/paged KV, both sparse block sizes, aligned/unaligned starts, empty context, and a long paged context. The full matrix provides broader combinations and large-batch stress coverage.

The release profile runs both `memcheck` and `synccheck` on `test_attention.test_gemm_skip_head_mid`, plus `test_clean_logits_bounds.test_clean_logits_bounds` on SM100/SM103. Other GPU test entrypoints and distributed MegaMoE checks retain their existing coverage. The full profile retains automatic sanitizer discovery. `--skip-sanitizer` and `--skip-mega-moe` still work for development; omit them for release validation.

To run only the selected attention cases against an installed wheel, use `cd sgl_deep_gemm/tests && DG_TEST_PROFILE=release python3 test_attention.py`. The existing `DG_MQA_NUM_CASES` random cap remains available in the full profile and cannot be combined with the release profile. CPU-only selection and runner-dispatch checks run with `python3 -m unittest discover -s sgl_deep_gemm -p test_release_test_profile.py` from the repository root.

Use Compute Sanitizer 2025.4.1 or newer for Python tests; older releases can retain Python tensors while collecting host backtraces and exhaust GPU memory. Set `COMPUTE_SANITIZER` to select a separately installed executable. See [NVIDIA's release notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#updates-in-2025-4-1).

To release a new set of wheels, please contact SGLang team and run the [release workflow](https://github.com/sgl-project/sglang/actions/workflows/release-whl-deepgemm.yml) under SGLang repo

For each major version release (0.X.Y -> 0.(X+1).0), a new branch should be created (release/v0.(X+1).0) for stability purpose.

For any incoming pull requests, it should be rebased upon `dev` branch. Any newly added or modified tests should be put under `sgl_deep_gemm/tests`
