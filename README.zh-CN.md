# DeepGEMM SM110（Jetson AGX Thor）适配

[English](README.md) | **简体中文**

本项目把 DeepGEMM v0.2.3 适配到 NVIDIA Jetson AGX Thor（SM110），供 SGLang 使用。

- 设备：NVIDIA Jetson AGX Thor，`sm_110` / compute capability 11.0
- 发行包：`sgl-deep-gemm==0.2.3+sm110`
- Python 导入名：`deep_gemm`
- 详细验证记录：[docs/sm110-thor-support.md](docs/sm110-thor-support.md)
- 上游原始 README：[README.upstream.md](README.upstream.md)

## 改了什么

上游 DeepGEMM v0.2.3 不识别架构主版本 11，所以 Thor 会在选 kernel 之前报 `Unsupported architecture`。

这个 fork 做了以下修改：

- 把 SM110 加入原有的 SM100 系列调度条件；
- 修改 10 个 host 头文件中的 45 处判断；
- 没有修改设备端 kernel、CUTLASS 和 JIT 编译逻辑；
- 增加 `deep_gemm.supports_arch('sm110')`，方便 SGLang 判断当前版本是否支持 SM110。

本次复用 DeepGEMM 已有的 SM100 kernel。完整文件清单和测试证据见[详细验证文档](docs/sm110-thor-support.md)。

## 安装

### 环境要求

- NVIDIA Jetson AGX Thor 或其他受支持的 NVIDIA GPU
- CUDA Toolkit 13.0 或更新版本，并且包含 `nvcc`
- 支持 C++20 的 GCC/G++ 13
- 与 CUDA 版本匹配的 PyTorch
- `libdw-dev`

本次实际测试环境为 CUDA 13.0、PyTorch 2.14.1+cu130、GCC 13.3 和 Python 3.12。

缺少系统依赖时安装：

```bash
sudo apt-get install -y libdw-dev
```

### 下载代码

```bash
git clone https://github.com/yeasen-z/DeepGEMM-thor-sm110.git
cd DeepGEMM-thor-sm110
```

构建脚本会自动下载所需的 `cutlass` 和 `deep_jit` 子模块。也可以在 clone 时直接加 `--recursive`：

```bash
git clone --recursive https://github.com/yeasen-z/DeepGEMM-thor-sm110.git
```

### 编译和安装

先进入准备运行 SGLang 的环境，并使用同一个 Python 编译和安装（将 `your-env` 换成你的环境名，也可以激活 venv）：

```bash
conda activate your-env
export PYTHON_EXE="$(which python)"

# CUDA 不在标准位置时再设置；能自动找到时可以不设置
# export CUDA_HOME=/usr/local/cuda

bash build_sgl_deep_gemm.sh
"$PYTHON_EXE" -m pip install --force-reinstall --no-deps dist/sgl_deep_gemm-*.whl
```

构建脚本会打印实际使用的 Python 和 PyTorch 版本，请确认它们就是目标环境中的版本。

### 验证安装

请在源码目录之外运行，避免 Python 优先导入当前源码目录：

```bash
cd /tmp
python -c "
import deep_gemm, torch
print('version:', deep_gemm.__version__)
print('device :', torch.cuda.get_device_capability())
print('SM110  :', deep_gemm.supports_arch('sm110'))
a = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
b = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
c = torch.empty(128, 128, device='cuda', dtype=torch.bfloat16)
deep_gemm.bf16_gemm_nt(a, b, c)
torch.cuda.synchronize()
print('GEMM   : OK')
"
```

Thor 上预期输出：

```text
version: 0.2.3+sm110
device : (11, 0)
SM110  : True
GEMM   : OK
```

也可以运行完整诊断：

```bash
python -m deep_gemm.doctor
```

安装的版本 `0.2.3+sm110` 满足 SGLang 的 `sgl-deep-gemm==0.2.3` 依赖。

## 配置 SGLang

```bash
export SGLANG_ENABLE_JIT_DEEPGEMM=1
```

FP8 模型可以使用 DeepGEMM 后端：

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B-FP8 \
  --fp8-gemm-backend deep_gemm \
  --dtype bfloat16
```

配套的 SGLang SM110 适配请使用 [yeasen-z/sglang-inscope](https://github.com/yeasen-z/sglang-inscope)，并按该仓库的说明安装。本仓库只包含 DeepGEMM 的修改；SGLang 侧的配套改动（包括 UE8M0 scale 判断）详见[集成说明](docs/sm110-thor-support.md#6-sglang-integration-prerequisite)。

## CUDA 查找

不一定需要手动设置 `CUDA_HOME`。DeepGEMM 会按以下顺序查找 CUDA：

1. `CUDA_HOME` 或 `CUDA_PATH`；
2. `PATH` 中的 `nvcc`；
3. `/usr/local/cuda`。

当 CUDA 安装在非标准目录、机器上有多个 CUDA 版本，或者自动查找失败时，再手动指定：

```bash
export CUDA_HOME=/path/to/cuda
```

选中的 CUDA 目录需要包含 `bin/nvcc`，并且需要有运行时 JIT 所需的 device headers。第一次使用某个 kernel 可能较慢，因为需要现场编译并缓存。默认缓存目录是 `$HOME/.dj`，需要时可以设置 `DG_JIT_CACHE_DIR`。

## 注意事项

1. **给 SGLang 使用时走上面的 wheel 构建方式。** 不要使用仓库根目录的 `pip install .`。
2. **不要同时安装 `sgl-deep-gemm` 和另一个名为 `deep_gemm` 的发行包。** 它们使用同一个 Python 包目录，可能互相覆盖。
3. 编译、安装、运行 SGLang 要使用同一个 Python 环境。编译产物与 PyTorch/CUDA 环境有关。
4. 第一次运行可能较慢，因为要编译 JIT kernel；后续会使用缓存。
5. 本 fork 不提供通用预编译二进制，请在目标机器上编译。
6. 记录中的测试只使用了一台 Jetson AGX Thor。Mega MoE/Gate/mHC，以及 dense 或 m-grouped FP4 没有纳入测试。

## 测试摘要

在 Jetson AGX Thor 测试机上：

- BF16、FP8、layout、attention、einsum 和 hyperconnection 测试均无失败；
- 未加保护条件的 k-grouped FP4：189/189 通过；
- 未加保护条件的 psum layout：224/224 通过；
- SGLang + Qwen3-8B-FP8 可以生成正确文本；
- 相同 workload 的 steady-state prefill：DeepGEMM 5,331 tok/s，Triton 3,639 tok/s，提升 1.47x。

性能对比仅覆盖 prefill：执行模型 36 层中的 24 层，共 41 个请求、1,440,227 tokens，不代表完整模型或 decode 性能。

完整测试设备、命令、限制和测量方法见[详细验证文档](docs/sm110-thor-support.md)。

## 上游项目

本 fork 基于 [sgl-project/DeepGEMM](https://github.com/sgl-project/DeepGEMM) v0.2.3。上游原始 README 保存在 [README.upstream.md](README.upstream.md)。
