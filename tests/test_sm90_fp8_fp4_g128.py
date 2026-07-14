"""SM90 FP8xFP4(g128, fp32-scale) accuracy + benchmark script.

这份脚本用于你们新的权重编码语义：

1. A 仍然是 FP8(e4m3) + per-token/per-128 scale
2. B 仍然是 packed FP4(两个 4-bit nibble 打包成一个 int8)
3. B 的 scale 改成 `group_size=128`，且 dtype 明确为 `torch.float32`

它和原始 `test_sm90_fp8_fp4.py` 的关键区别是：

- 不再走偏向 `gran_k_b=32` 的 fast path 测试逻辑
- 直接构造与你们转换脚本一致的 FP4(g128, fp32-scale) 权重
- 同时覆盖 contiguous / masked 两条常用调用路径

用法：
python /sgl-workspace/sglang/test_sm90_fp8_fp4_g128.py --repo-root /root/work/DeepGEMM

"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Callable

import torch

FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

DEFAULT_SHAPES = [(4096, 7168), (7168, 2048), (4096, 4096)]
DEFAULT_GROUPS = (8, 16, 32)
DEFAULT_M_PER_GROUP = (1, 4, 8, 16, 32, 64)

W4_DIFF_THRESHOLD = 0.015
FP8_DIFF_THRESHOLD = 0.05

deep_gemm = None
calc_diff = None
per_token_cast_to_fp8 = None

def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())

def _parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"invalid shape item: {item!r}, expected like 4096x7168")
        n_str, k_str = item.split("x", 1)
        shapes.append((int(n_str), int(k_str)))
    return shapes

def _discover_repo_root(explicit_repo_root: str | None) -> Path:
    candidates: list[Path] = []
    if explicit_repo_root:
        candidates.append(Path(explicit_repo_root).resolve())

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()

    candidates.extend(
        [
            cwd,
            script_dir,
            script_dir.parent,
            cwd.parent,
        ]
    )
    candidates.extend(script_dir.parents)
    candidates.extend(cwd.parents)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "deep_gemm").exists():
            return candidate

    raise FileNotFoundError(
        "cannot locate DeepGEMM repo root. Please pass --repo-root /path/to/DeepGEMM"
    )

def _load_deep_gemm(repo_root: Path) -> None:
    global deep_gemm, calc_diff, per_token_cast_to_fp8

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    deep_gemm = importlib.import_module("deep_gemm")
    calc_diff = importlib.import_module("deep_gemm.testing").calc_diff
    per_token_cast_to_fp8 = importlib.import_module("deep_gemm.utils.math").per_token_cast_to_fp8

def _require_sm90() -> None:
    assert torch.cuda.is_available()
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        raise RuntimeError(f"This benchmark is intended for SM90, got sm_{major}x")

def _time_cuda(fn: Callable[[], None], warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3

def _effective_bytes(
    groups: int,
    logical_m: int,
    n: int,
    k: int,
    a_gran_k: int,
    *,
    fp8_b: bool,
    b_gran_k: int,
) -> int:
    a_scale_k = (k + a_gran_k - 1) // a_gran_k
    b_scale_k = (k + b_gran_k - 1) // b_gran_k
    a_bytes = logical_m * k + logical_m * a_scale_k * 4
    b_data_bytes = groups * n * k if fp8_b else groups * n * (k // 2)
    b_scale_bytes = groups * n * b_scale_k * 4
    d_bytes = logical_m * n * 2
    return a_bytes + b_data_bytes + b_scale_bytes + d_bytes

def _build_grouped_layout(groups: int, m_per_group: int):
    m = groups * m_per_group
    group_starts = [group_id * m_per_group for group_id in range(groups)]
    group_ends = [(group_id + 1) * m_per_group for group_id in range(groups)]
    grouped_layout = torch.arange(groups, device="cuda", dtype=torch.int32).repeat_interleave(m_per_group)
    return m, group_starts, group_ends, grouped_layout

def _get_m_alignment_for_contiguous_layout() -> int:
    """DeepGEMM contiguous grouped layout 要求每个 group 的 M 段按 block 对齐。

    `deep_gemm.get_mk_alignment_for_contiguous_layout()` 的返回值在不同版本里可能是：
    - 一个 int(只代表 M 对齐）
    - 一个 tuple/list(例如 (align_m, align_k)
    """
    align = deep_gemm.get_mk_alignment_for_contiguous_layout()
    if isinstance(align, (tuple, list)):
        return int(align[0])
    return int(align)

def _align_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a

def _unpack_fp4_values(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.int8 or packed.ndim != 2:
        raise ValueError(f"expected int8 2D packed tensor, got {packed.dtype}, {tuple(packed.shape)}")

    out_dim, packed_in_dim = packed.shape
    packed_u8 = packed.view(torch.uint8)
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    table = FP4_TABLE.to(device=packed.device)
    values = torch.stack((table[low.long()], table[high.long()]), dim=-1).reshape(out_dim, packed_in_dim * 2)
    return values

def _dequant_fp4_grouped(packed: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    fp4 = _unpack_fp4_values(packed)
    out_dim, in_dim = fp4.shape
    expected_scale_shape = (out_dim, in_dim // group_size)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"scale shape mismatch: got {tuple(scale.shape)}, expected {expected_scale_shape}"
        )
    full_scale = scale.float().repeat_interleave(group_size, dim=1)
    return fp4 * full_scale

def _quantize_real_to_fp4_grouped(
    real: torch.Tensor,
    group_size: int,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if real.ndim != 2:
        raise ValueError(f"expected 2D tensor, got shape {tuple(real.shape)}")

    out_dim, in_dim = real.shape
    if in_dim % group_size != 0:
        raise ValueError(f"in_dim={in_dim} must be divisible by group_size={group_size}")
    if in_dim % 2 != 0:
        raise ValueError(f"in_dim={in_dim} must be even for nibble packing")

    num_groups = in_dim // group_size
    grouped = real.float().view(out_dim, num_groups, group_size)
    max_abs = grouped.abs().amax(dim=-1)
    scale = torch.clamp(max_abs / 6.0, min=eps).float()

    codebook = FP4_TABLE.to(device=real.device).view(1, 1, 1, 16)
    normalized = grouped / scale.unsqueeze(-1)
    dist = (normalized.unsqueeze(-1) - codebook).abs()
    nibble = dist.argmin(dim=-1).to(torch.uint8).view(out_dim, in_dim)

    low = nibble[:, 0::2]
    high = nibble[:, 1::2]
    packed_u8 = low | (high << 4)
    packed_i8 = packed_u8.view(torch.int8)
    return packed_i8, scale

def _cast_back_from_fp8_1d(x: torch.Tensor, sf: torch.Tensor, gran_k: int = 128) -> torch.Tensor:
    group_idx = torch.arange(x.size(-1), device=x.device) // gran_k
    return x.float() * sf[..., group_idx]

def _make_b_fp4_g128(real_b: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    return _quantize_real_to_fp4_grouped(real_b, group_size=group_size)

def _resolve_contiguous_kernel():
    fn = getattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma", None)
    if fn is None:
        raise RuntimeError(
            "cannot find `m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma` in deep_gemm. "
            "Please use the DeepGEMM branch/build that contains the SM90 FP8xFP4 kernels."
        )
    return fn

def _resolve_masked_kernel():
    fn = getattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_masked_sm90_fused_wgmma", None)
    if fn is None:
        raise RuntimeError(
            "cannot find `m_grouped_fp8_fp4_gemm_nt_masked_sm90_fused_wgmma` in deep_gemm. "
            "Please use the DeepGEMM branch/build that contains the SM90 FP8xFP4 kernels."
        )
    return fn

def _benchmark_case_contiguous(
    groups: int,
    m_per_group: int,
    n: int,
    k: int,
    *,
    a_gran_k: int = 128,
    b_gran_k: int = 128,
) -> dict[str, float | int]:
    if b_gran_k != 128:
        raise ValueError("this g128 script expects b_gran_k == 128")

    kernel = _resolve_contiguous_kernel()
    # NOTE: contiguous grouped layout 要求每个 group 的 token 段在 M 维按 block 对齐；
    # 否则在某些配置下可能出现错误结果（而不是直接报错）。
    align_m = _get_m_alignment_for_contiguous_layout()
    m_per_group_aligned = _align_up(m_per_group, align_m)
    m, group_starts, group_ends, grouped_layout = _build_grouped_layout(groups, m_per_group_aligned)

    # 每个 group 只有前 m_per_group 行是有效 token，后面 padding 行填 0，保证参考值与 kernel 一致。
    a_ref_src = torch.zeros((m, k), device="cuda", dtype=torch.bfloat16)
    for group_id in range(groups):
        start = group_id * m_per_group_aligned
        a_ref_src[start:start + m_per_group] = torch.randn(
            (m_per_group, k), device="cuda", dtype=torch.bfloat16
        )
    b_ref_src = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a = per_token_cast_to_fp8(a_ref_src, use_ue8m0=False, gran_k=a_gran_k)
    a_dequant = _cast_back_from_fp8_1d(a[0], a[1], gran_k=a_gran_k)

    b_fp4 = torch.empty((groups, n, k // 2), device="cuda", dtype=torch.int8)
    b_sf = torch.empty((groups, n, k // b_gran_k), device="cuda", dtype=torch.float)
    b_fp8_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_fp8_sf = torch.empty((groups, n, k // a_gran_k), device="cuda", dtype=torch.float)
    ref = torch.zeros((m, n), device="cuda", dtype=torch.bfloat16)

    for group_id in range(groups):
        packed, scale = _make_b_fp4_g128(b_ref_src[group_id], group_size=b_gran_k)
        b_fp4[group_id] = packed
        b_sf[group_id] = scale

        b_dequant = _dequant_fp4_grouped(packed, scale, group_size=b_gran_k)
        b_fp8_data[group_id], b_fp8_sf[group_id] = per_token_cast_to_fp8(
            b_dequant.to(torch.bfloat16), use_ue8m0=False, gran_k=a_gran_k
        )
        # 仅对每个 group 的有效 token 行计算参考值；padding 行保持 0。
        start = group_id * m_per_group_aligned
        end = start + m_per_group
        if m_per_group > 0:
            ref[start:end] = (a_dequant[start:end] @ b_dequant.t()).to(torch.bfloat16)

    b_w4 = (b_fp4, b_sf)
    b_fp8 = (b_fp8_data, b_fp8_sf)

    d_fp8 = torch.empty_like(ref)

    def run_fp8():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            a,
            b_fp8,
            d_fp8,
            grouped_layout,
            recipe_a=(1, a_gran_k),
            recipe_b=(1, a_gran_k),
            use_psum_layout=False,
        )

    d_w4 = torch.empty_like(ref)

    def run_w4():
        kernel(
            a,
            b_w4,
            d_w4,
            grouped_layout,
            gran_k=b_gran_k,
            compiled_dims="nk",
            use_psum_layout=False,
        )

    run_fp8()
    run_w4()

    fp8_diff = calc_diff(d_fp8, ref)
    w4_diff = calc_diff(d_w4, ref)
    fp8_elapsed = _time_cuda(run_fp8)
    w4_elapsed = _time_cuda(run_w4)

    # 带宽模型按“实际参与计算/写回的 M 行数”计：contiguous 场景下 padding 行也会发生 A/D 的读写。
    logical_m = groups * m_per_group_aligned
    w4_bytes = _effective_bytes(groups, logical_m, n, k, a_gran_k, fp8_b=False, b_gran_k=b_gran_k)
    fp8_bytes = _effective_bytes(groups, logical_m, n, k, a_gran_k, fp8_b=True, b_gran_k=a_gran_k)

    return {
        "groups": groups,
        "m_per_group": m_per_group,
        "n": n,
        "k": k,
        "w4_us": w4_elapsed * 1e6,
        "w4_gbps": w4_bytes / w4_elapsed / 1e9,
        "w4_diff": w4_diff,
        "fp8_us": fp8_elapsed * 1e6,
        "fp8_gbps": fp8_bytes / fp8_elapsed / 1e9,
        "fp8_diff": fp8_diff,
        "speedup": fp8_elapsed / w4_elapsed,
    }

def _benchmark_case_masked(
    groups: int,
    m_per_group: int,
    n: int,
    k: int,
    *,
    max_m: int = 128,
    a_gran_k: int = 128,
    b_gran_k: int = 128,
) -> dict[str, float | int]:
    if b_gran_k != 128:
        raise ValueError("this g128 script expects b_gran_k == 128")

    kernel = _resolve_masked_kernel()
    masked_m = torch.full((groups,), m_per_group, device="cuda", dtype=torch.int32)

    a_ref_src = torch.randn((groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    b_ref_src = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a_data = torch.empty((groups, max_m, k), device="cuda", dtype=torch.float8_e4m3fn)
    a_sf = torch.empty((groups, max_m, k // a_gran_k), device="cuda", dtype=torch.float)
    for group_id in range(groups):
        a_data[group_id], a_sf[group_id] = per_token_cast_to_fp8(
            a_ref_src[group_id], use_ue8m0=False, gran_k=a_gran_k
        )
    a = (a_data, a_sf)
    a_dequant = _cast_back_from_fp8_1d(a[0], a[1], gran_k=a_gran_k)

    b_fp4 = torch.empty((groups, n, k // 2), device="cuda", dtype=torch.int8)
    b_sf = torch.empty((groups, n, k // b_gran_k), device="cuda", dtype=torch.float)
    b_fp8_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_fp8_sf = torch.empty((groups, n, k // a_gran_k), device="cuda", dtype=torch.float)
    ref = torch.zeros((groups, max_m, n), device="cuda", dtype=torch.bfloat16)

    for group_id in range(groups):
        packed, scale = _make_b_fp4_g128(b_ref_src[group_id], group_size=b_gran_k)
        b_fp4[group_id] = packed
        b_sf[group_id] = scale

        b_dequant = _dequant_fp4_grouped(packed, scale, group_size=b_gran_k)
        b_fp8_data[group_id], b_fp8_sf[group_id] = per_token_cast_to_fp8(
            b_dequant.to(torch.bfloat16), use_ue8m0=False, gran_k=a_gran_k
        )

        valid_m = int(masked_m[group_id].item())
        if valid_m > 0:
            ref[group_id, :valid_m] = (a_dequant[group_id, :valid_m] @ b_dequant.t()).to(torch.bfloat16)

    b_w4 = (b_fp4, b_sf)
    b_fp8 = (b_fp8_data, b_fp8_sf)

    d_w4 = torch.empty_like(ref)

    def run_w4():
        kernel(
            a,
            b_w4,
            d_w4,
            masked_m,
            m_per_group,
            gran_k=b_gran_k,
            gran_k_a=a_gran_k,
            gran_k_b=b_gran_k,
        )

    d_fp8 = torch.empty_like(ref)

    def run_fp8():
        deep_gemm.m_grouped_fp8_gemm_nt_masked(
            a,
            b_fp8,
            d_fp8,
            masked_m,
            m_per_group,
            recipe_a=(1, a_gran_k),
            recipe_b=(1, a_gran_k),
        )

    run_w4()
    run_fp8()

    w4_diff = max(
        calc_diff(d_w4[group_id, :m_per_group], ref[group_id, :m_per_group])
        for group_id in range(groups)
    )
    fp8_diff = max(
        calc_diff(d_fp8[group_id, :m_per_group], ref[group_id, :m_per_group])
        for group_id in range(groups)
    )
    w4_elapsed = _time_cuda(run_w4)
    fp8_elapsed = _time_cuda(run_fp8)

    logical_m = groups * m_per_group
    w4_bytes = _effective_bytes(groups, logical_m, n, k, a_gran_k, fp8_b=False, b_gran_k=b_gran_k)
    fp8_bytes = _effective_bytes(groups, logical_m, n, k, a_gran_k, fp8_b=True, b_gran_k=a_gran_k)

    return {
        "groups": groups,
        "m_per_group": m_per_group,
        "n": n,
        "k": k,
        "w4_us": w4_elapsed * 1e6,
        "w4_gbps": w4_bytes / w4_elapsed / 1e9,
        "w4_diff": w4_diff,
        "fp8_us": fp8_elapsed * 1e6,
        "fp8_gbps": fp8_bytes / fp8_elapsed / 1e9,
        "fp8_diff": fp8_diff,
        "speedup": fp8_elapsed / w4_elapsed,
    }

def _print_header(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))
    print("B scale format: group_size=128, dtype=torch.float32, non-UE8M0")
    print("groups | m/group | n | k | W4 us | W4 GB/s | W4 diff | FP8 us | FP8 GB/s | FP8 diff | Speedup")
    print("-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --")

def _print_row(row: dict[str, float | int]) -> None:
    print(
        f"{row['groups']} | {row['m_per_group']} | {row['n']} | {row['k']} | "
        f"{row['w4_us']:.0f} | {row['w4_gbps']:.0f} | {row['w4_diff']:.4f} | "
        f"{row['fp8_us']:.0f} | {row['fp8_gbps']:.0f} | {row['fp8_diff']:.4f} | "
        f"{row['speedup']:.2f}x"
    )

def _run_suite(
    mode: str,
    groups_list: tuple[int, ...],
    m_per_group_list: tuple[int, ...],
    shapes: list[tuple[int, int]],
    *,
    a_gran_k: int,
    b_gran_k: int,
    max_m: int,
    fail_on_threshold: bool,
) -> None:
    if mode in ("contiguous", "both"):
        _print_header("Contiguous FP8xFP4(g128) benchmark")
        for groups in groups_list:
            for m_per_group in m_per_group_list:
                for n, k in shapes:
                    row = _benchmark_case_contiguous(
                        groups,
                        m_per_group,
                        n,
                        k,
                        a_gran_k=a_gran_k,
                        b_gran_k=b_gran_k,
                    )
                    _print_row(row)
                    if fail_on_threshold:
                        assert row["w4_diff"] < W4_DIFF_THRESHOLD, row
                        assert row["fp8_diff"] < FP8_DIFF_THRESHOLD, row

    if mode in ("masked", "both"):
        _print_header("Masked FP8xFP4(g128) benchmark")
        for groups in groups_list:
            for m_per_group in m_per_group_list:
                for n, k in shapes:
                    row = _benchmark_case_masked(
                        groups,
                        m_per_group,
                        n,
                        k,
                        max_m=max_m,
                        a_gran_k=a_gran_k,
                        b_gran_k=b_gran_k,
                    )
                    _print_row(row)
                    if fail_on_threshold:
                        assert row["w4_diff"] < W4_DIFF_THRESHOLD, row
                        assert row["fp8_diff"] < FP8_DIFF_THRESHOLD, row

def main() -> None:
    parser = argparse.ArgumentParser(description="SM90 FP8xFP4 g128 benchmark for fp32-scale weights")
    parser.add_argument("--repo-root", type=str, default=None, help="DeepGEMM 仓库根目录")
    parser.add_argument(
        "--mode",
        choices=("contiguous", "masked", "both"),
        default="both",
        help="运行 contiguous、masked 或两者都跑",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="8,16,32",
        help="要测试的 group 数，逗号分隔，如 8,16,32",
    )
    parser.add_argument(
        "--m-per-group",
        type=str,
        default="1,4,8,16,32, 64",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default="4096x7168,7168x2048,4096x4096",
        help="测试形状，格式 NxK, 逗号分隔",
    )
    parser.add_argument("--a-gran-k", type=int, default=128)
    parser.add_argument("--b-gran-k", type=int, default=128)
    parser.add_argument("--max-m", type=int, default=128, help="masked 路径下每组分配的最大 m")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-fail-on-threshold",
        action="store_true",
        help="仅打印结果，不对 diff 阈值做 assert",
    )
    args = parser.parse_args()

    if args.b_gran_k != 128:
        raise ValueError("this script is specifically for FP4 + group_size=128")

    repo_root = _discover_repo_root(args.repo_root)
    _load_deep_gemm(repo_root)

    torch.manual_seed(args.seed)
    _require_sm90()

    groups_list = _parse_csv_ints(args.groups)
    m_per_group_list = _parse_csv_ints(args.m_per_group)
    shapes = _parse_shapes(args.shapes)

    print(f"repo_root = {repo_root}")
    print("running SM90 FP8xFP4 g128 benchmark ...")
    started = time.time()

    _run_suite(
        mode=args.mode,
        groups_list=groups_list,
        m_per_group_list=m_per_group_list,
        shapes=shapes,
        a_gran_k=args.a_gran_k,
        b_gran_k=args.b_gran_k,
        max_m=args.max_m,
        fail_on_threshold=not args.no_fail_on_threshold,
    )

    print()
    print(f"done in {time.time() - started:.1f}s")

if __name__ == "__main__":
    main()
