"""SM90 FP8 block32 correctness and activation-scale contract tests."""

import os
import re
import sys
import tempfile

import torch

import deep_gemm


def _ceil_div(x, y):
    return (x + y - 1) // y


def _run(a, sa, b, sb, *, capture_config=False, **recipe):
    out = torch.empty((a.shape[0], b.shape[0]), device="cuda", dtype=torch.bfloat16)
    config_log = ""
    if capture_config:
        sys.stdout.flush()
        with tempfile.TemporaryFile(mode="w+") as trace:
            stdout_fd = os.dup(1)
            try:
                os.dup2(trace.fileno(), 1)
                deep_gemm.fp8_gemm_nt((a, sa), (b, sb), out, **recipe)
            finally:
                os.dup2(stdout_fd, 1)
                os.close(stdout_fd)
            trace.seek(0)
            config_log = trace.read()
    else:
        deep_gemm.fp8_gemm_nt((a, sa), (b, sb), out, **recipe)
    torch.cuda.synchronize()
    return out, config_log


def _reference(a, sa, b, sb):
    # Independent FP32 dequantized reference: each K32 product has its own scales.
    n, k = b.shape
    out = torch.zeros((a.shape[0], n), device="cuda", dtype=torch.float32)
    for group in range(_ceil_div(k, 32)):
        start, end = group * 32, min((group + 1) * 32, k)
        b_scale = sb[:, group].repeat_interleave(32)[:n]
        product = a[:, start:end].float() @ b[:, start:end].float().T
        out += product * sa[:, group, None] * b_scale[None, :]
    return out.to(torch.bfloat16)


def _selected_banks(config_log):
    match = re.search(r"block_m=(\d+), block_n=(\d+)", config_log)
    assert match is not None, f"SM90 did not report a selected tile: {config_log}"
    block_m, block_n = map(int, match.groups())
    # WGMMA M=64; the kernel uses a 128-row wave for larger M tiles.
    wave_m = min(block_m, 128)
    banks = 1 if block_m > wave_m else 4 if block_n <= 64 else 2
    return banks, block_m, block_n


def test_invalid_activation_granularity():
    m = n = 32
    k = 128
    a = torch.ones((m, k), device="cuda").to(torch.float8_e4m3fn)
    b = torch.ones((n, k), device="cuda").to(torch.float8_e4m3fn)
    invalid = (
        (32, 32, 32, False),
        (32, 32, 32, True),
        (128, 128, 128, False),
        (128, 128, 128, True),
        (128, 1, 128, False),
    )
    for gran_m, gran_n, gran_k, separate in invalid:
        # Allocate enough backing for every row if the guard regresses.
        backing = torch.ones((m, _ceil_div(k, gran_k)), device="cuda")
        sa = backing[:_ceil_div(m, gran_m)]
        sb = torch.ones((_ceil_div(n, gran_n), _ceil_div(k, gran_k)), device="cuda")
        recipe = (dict(recipe_a=(gran_m, gran_k), recipe_b=(gran_n, gran_k))
                  if separate else dict(recipe=(gran_m, gran_n, gran_k)))
        try:
            _run(a, sa, b, sb, **recipe)
        except RuntimeError as error:
            assert "gran_m == 1" in str(error), (recipe, str(error))
        else:
            raise AssertionError(f"SM90 accepted unsupported A-scale granularity: {recipe}")


def test_block32_correctness():
    # Cover K128-stage fragments, M/N tails, and both physical B-scale layouts.
    # The remaining shapes select all three bank schedules on H200; fallbacks
    # keep this check useful when another Hopper selects different tiles.
    cases = (
        (1, 8, 32, False, False, False),
        (17, 40, 96, True, True, False),
        (65, 72, 160, False, True, False),
        (129, 96, 224, True, False, False),
        (128, 32, 2048, False, False, True),
        (2048, 1024, 512, True, True, True),
        (2048, 2048, 512, False, False, True),
        (2048, 1024, 2048, True, True, True),
        (4096, 2048, 512, False, False, True),
    )
    observed_banks = set()
    for m, n, k, transpose_sb, separate_recipe, check_banks in cases:
        a = torch.randint(0, 4, (m, k), device="cuda").float().to(torch.float8_e4m3fn)
        b = torch.randint(0, 4, (n, k), device="cuda").float().to(torch.float8_e4m3fn)
        sa = torch.pow(2.0, torch.randint(-3, 1, (m, _ceil_div(k, 32)), device="cuda").float())
        sb = torch.pow(2.0, torch.randint(-3, 1, (_ceil_div(n, 32), _ceil_div(k, 32)), device="cuda").float())
        if transpose_sb:
            sb = sb.T.contiguous().T
        recipe = (dict(recipe_a=(1, 32), recipe_b=(32, 32))
                  if separate_recipe else dict(recipe=(1, 32, 32)))
        actual, config_log = _run(a, sa, b, sb, capture_config=check_banks, **recipe)
        expected = _reference(a, sa, b, sb)
        assert torch.equal(actual, expected), (
            f"block32 mismatch at {m=}, {n=}, {k=}, {transpose_sb=}, {separate_recipe=}; "
            f"max error={(actual.float() - expected.float()).abs().max().item()}")
        bank_note = ""
        if check_banks:
            banks, block_m, block_n = _selected_banks(config_log)
            observed_banks.add(banks)
            bank_note = f" banks={banks} tile=({block_m},{block_n})"
        print(f"PASS block32 M={m} N={n} K={k} transposed_B_scales={transpose_sb}{bank_note}", flush=True)
        if observed_banks == {1, 2, 4}:
            break
    assert observed_banks == {1, 2, 4}, observed_banks


def test_block128_control():
    m = n = 32
    k = 128
    a = torch.ones((m, k), device="cuda").to(torch.float8_e4m3fn)
    b = torch.ones((n, k), device="cuda").to(torch.float8_e4m3fn)
    sa = torch.ones((m, 1), device="cuda")
    sb = torch.ones((1, 1), device="cuda")
    for recipe in (dict(recipe=(1, 128, 128)),
                   dict(recipe_a=(1, 128), recipe_b=(128, 128))):
        actual, _ = _run(a, sa, b, sb, **recipe)
        assert torch.all(actual == 128), recipe


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        print("SKIP: SM90 GPU required")
    else:
        os.environ["DG_PRINT_CONFIGS"] = "1"
        torch.manual_seed(32)
        test_invalid_activation_granularity()
        test_block32_correctness()
        test_block128_control()
        print("PASS: SM90 block32/activation-scale cases")
