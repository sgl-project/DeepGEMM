
## Introduction

sgl-deep-gemm is a pypi package built from SGLang's customized branch of DeepGemm. Comparing with origina DeepGemm, it supports the following features to better support SGLang:
1. TVM FFI wrappers expose the kernels to Python; wheel compatibility depends on the build configuration.
2. pypi support: easy installation with `pip install sgl-deep-gemm`. No need to manually search for wheel links.
3. Fast iteration: add custom kernels and bump versions at no time.

## AFD M2N extension

`deep_gemm.mega.m2n` provides Mega MoE kernels for Attention-FFN Disaggregation (AFD). Attention-group (AG) ranks quantize and dispatch activations, then combine the expert outputs. Expert-group (EG) ranks run a persistent kernel that receives activations, computes the routed-expert MLPs, and returns the results. AG and EG can have different numbers of ranks.

### Requirements

M2N requires SM100 or SM103 GPUs with peer-accessible memory. The allocator uses CUDA FABRIC handles when available and CUDA IPC otherwise. Cross-host execution requires a shared NVLink fabric with CUDA FABRIC access; the IPC path is limited to peers on the same host. M2N does not use DeepEP or NVSHMEM.

The AG API accepts contiguous BF16 activations and produces BF16 outputs. Expert weights must use the Mega MoE layout:

| Expert weights | Buffer `mma_type` | EG `use_fp8_weights` |
| --- | --- | --- |
| FP8 | `fp8xfp4` | `True` |
| MXFP4 | `fp8xfp4` | `False` |
| NVFP4 | `nvfp4xnvfp4` | `False` |

`fp8xfp4` selects the FP8 activation-buffer format for both FP8 and MXFP4 weights. BF16 expert weights are not supported. Hidden and intermediate dimensions must be multiples of 128, or 256 for NVFP4. NVFP4 additionally requires activation scales and per-expert scaling tables; see the API docstrings in [`deep_gemm/mega/m2n.py`](../deep_gemm/mega/m2n.py).

### Build and install

Use a Linux Python environment with PyTorch and the [build requirements](#usage) installed. Build in an environment matching the target Python, platform, and PyTorch/CUDA ABI. Run from the repository root:

```bash
python3 -m pip install 'apache-tvm-ffi==0.1.11' ninja
bash build_sgl_deep_gemm.sh
python3 -m pip install dist/sgl_deep_gemm-*.whl
```

The build script initializes missing submodules and writes the wheel to `dist/`. It detects the GPU architecture automatically; set `TVM_FFI_CUDA_ARCH_LIST` when building for a different target or without a visible GPU. Set `SGL_DEEP_GEMM_VERSION` to override the wheel version without changing `sgl_deep_gemm/VERSION`. Keep the CUDA toolkit and C++ compiler available at runtime for DeepJIT compilation.

From a directory outside the source checkout, check that the installed package exposes the M2N API:

```bash
python3 -c 'from deep_gemm.mega.m2n import get_symm_buffer_for_mega_moe_m2n, make_m2n_weight_descs, mega_moe_m2n_ag, mega_moe_m2n_eg'
```

### Interfaces

Create one process group containing the AG ranks first, followed by the EG ranks. The number of experts must be divisible by the number of EG ranks.

- `get_symm_buffer_for_mega_moe_m2n` allocates a communication buffer collectively across the group.
- `make_m2n_weight_descs` builds one layer's weight descriptors from weights prepared with `deep_gemm.transform_weights_for_mega_moe`.
- `mega_moe_m2n_ag` runs quantization, dispatch, combine, and output reduction for one layer on an AG rank.
- `mega_moe_m2n_eg` runs the expert layers on an EG rank. It accepts one to four buffer lanes, all with the same layout and process group.

All ranks must construct buffers in the same order and keep them alive through kernel execution and CUDA graph replay. Call `buffer.destroy()` collectively before destroying the process group. A failed buffer construction requires restarting the whole group.

For a complete distributed example, see [`test_mega_moe_m2n.py`](tests/test_mega_moe_m2n.py), which uses two AG ranks and two EG ranks with eager execution and CUDA graph replay.

### Attribution

The M2N implementation was adapted from [FastAFD](https://github.com/hao-ai-lab/FastAFD/tree/3c7161949310b6d59d6b4cf9bf997a4935c8113b/python/minisgl/kernel/csrc/deepgemm). Files derived from that implementation include:

- [`deep_gemm/include/deep_gemm/impls/sm100_mega_moe_m2n.cuh`](../deep_gemm/include/deep_gemm/impls/sm100_mega_moe_m2n.cuh)
- [`deep_gemm/include/deep_gemm/layout/mega_moe_m2n.cuh`](../deep_gemm/include/deep_gemm/layout/mega_moe_m2n.cuh)
- [`csrc/jit_kernels/impls/sm100_mega_moe_m2n.hpp`](../csrc/jit_kernels/impls/sm100_mega_moe_m2n.hpp)

The symmetric Mega MoE implementation originates in DeepGEMM. The [packaged license](LICENSE) retains the DeepSeek and sgl-project MIT notices for the derived M2N code alongside the SGLang package license.

## Usage
Build requirements include CUDA 12.9 or newer, a C++20 compiler with `std::format` support (for example GCC 13), and the elfutils development headers (`apt-get install libdw-dev` on Debian/Ubuntu). Initialize the CUTLASS and DeepJIT submodules with `git submodule update --init --recursive`.

To build it locally, run `bash build_sgl_deep_gemm.sh`, then pip install the wheel generated under `dist`.

K-grouped FP8 GEMM and its scale-packing helper preserve compact per-group scaling factors by default. Pass `use_padded_sf_layout=True` to both packing and GEMM when packed scales cover each group's padded K extent. The flag is appended to existing arguments, so existing positional calls retain their behavior.

Run wheel validation with `bash sgl_deep_gemm/run_tests.sh`. The Mega MoE reference comparisons require DeepEP with `ElasticBuffer`; Mega Gate and Mega mHC also require TileKernels and TileLang. Install the validation extras with `pip install 'sgl-deep-gemm[dev]'` (TileLang 0.1.9 and TileKernels 1.0.0). The runner enables NVML-based CUDA discovery to preserve fork compatibility while retaining TVM FFI's DLPack fast path. It runs legacy, lazy-init, and compute-sanitizer checks; `--skip-sanitizer` is available for ordinary development runs. Distributed MegaMoE runs include separate Hopper numerical accuracy coverage; the sanitizer's automatic discovery covers single-process tests and reports distributed entrypoints separately.

Run `bash sgl_deep_gemm/run_tests.sh --release` for reduced attention coverage and focused `memcheck`/`synccheck`, targeting a release gate under one hour. Selected cases retain their original assertions; other ordinary and distributed tests remain unchanged. The default command keeps the full suite; the table shows release / full attention case counts.

| GPU architecture | Dense MQA | Paged MQA | Sparse MQA |
| --- | ---: | ---: | ---: |
| Hopper (SM90) | 8 / 192 | 8 / 24 | unsupported |
| Blackwell (SM100/SM103) | 36 / 2304 | 72 / 4320 | 12 / 161 |
| Blackwell (SM120) | 16 / 256 | 16 / 48 | unsupported |

Use Compute Sanitizer 2025.4.1 or newer for Python tests; older releases can retain Python tensors while collecting host backtraces and exhaust GPU memory. Set `COMPUTE_SANITIZER` to select a separately installed executable. See [NVIDIA's release notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#updates-in-2025-4-1).

To release a new set of wheels, please contact SGLang team and run the [release workflow](https://github.com/sgl-project/sglang/actions/workflows/release-whl-deepgemm.yml) under SGLang repo

For each major version release (0.X.Y -> 0.(X+1).0), a new branch should be created (release/v0.(X+1).0) for stability purpose.

For any incoming pull requests, it should be rebased upon `dev` branch. Any newly added or modified tests should be put under `sgl_deep_gemm/tests`
