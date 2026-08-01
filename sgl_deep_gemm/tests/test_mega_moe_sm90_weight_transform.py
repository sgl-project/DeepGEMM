"""Regression test for SM90 MegaMoE FP8 weight-block sanitization."""

import argparse

import torch

import deep_gemm


def main() -> None:
    parser = argparse.ArgumentParser()
    # Keep the common test-runner interface; this test is intentionally single-rank.
    parser.add_argument("--num-processes", type=int, default=1)
    parser.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available")
        return
    if torch.cuda.get_device_capability()[0] != 9:
        print("SKIP: requires an SM90 CUDA device")
        return

    l1_weight = torch.full(
        (1, 256, 128), 7, dtype=torch.float8_e4m3fn, device="cuda"
    )
    l2_weight = torch.full(
        (1, 128, 128), 7, dtype=torch.float8_e4m3fn, device="cuda"
    )
    l1_scale = torch.full((1, 2, 1), 1.0e-20, dtype=torch.float32, device="cuda")
    l2_scale = torch.full((1, 1, 1), 1.0e-20, dtype=torch.float32, device="cuda")

    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        (l1_weight, l1_scale), (l2_weight, l2_scale)
    )
    transformed_l1_weight, transformed_l1_scale = transformed_l1
    transformed_l2_weight, transformed_l2_scale = transformed_l2

    assert torch.count_nonzero(transformed_l1_weight).item() == 0
    assert torch.count_nonzero(transformed_l2_weight).item() == 0
    assert torch.isfinite(transformed_l1_scale).all()
    assert torch.isfinite(transformed_l2_scale).all()
    assert torch.all(transformed_l1_scale >= 1.0e-12)
    assert torch.all(transformed_l2_scale >= 1.0e-12)
    # The input tensors are caller-owned and must remain unchanged.
    assert torch.count_nonzero(l1_weight).item() == l1_weight.numel()
    assert torch.count_nonzero(l2_weight).item() == l2_weight.numel()
    print("PASS: tiny SM90 MegaMoE weight blocks were sanitized")


if __name__ == "__main__":
    main()
