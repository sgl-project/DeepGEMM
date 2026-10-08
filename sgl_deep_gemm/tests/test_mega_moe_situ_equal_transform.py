import torch

# K3 config constants baked into the kernel (activation_situ_{beta,linear_beta}).
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0


def _situ_reference(gate: torch.Tensor, up: torch.Tensor):
    """Exact SiTU: beta * tanh(gate / beta) * sigmoid(gate), linear_beta * tanh(up / linear_beta)."""
    return (SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate),
            SITU_LINEAR_BETA * torch.tanh(up / SITU_LINEAR_BETA))


def _situ_equal_transform(gate: torch.Tensor, up: torch.Tensor):
    """The rewrite the kernel's SiTU path uses:
    sigmoid(g) = (1 + tanh(g/2)) / 2 and tanh(g/2) = 2T / (1 + T^2) with T = tanh(g/4)
    give 4 * tanh(g/4) * sigmoid(g) = 2T(1 + T)^2 / (1 + T^2)."""
    t = torch.tanh(gate * 0.25)
    s = 1.0 + t
    return (2.0 * t * (s * s)) / (1.0 + t * t), SITU_LINEAR_BETA * torch.tanh(up * 0.04)


def test_situ_equal_transform_accuracy():
    """The kernel's rewrite must reproduce the exact SiTU definition to fp32 rounding.

    Both SiTU branches are bounded (|gate| <= beta, |up| <= linear_beta), so an absolute
    bound is a uniform relative bound. Measured worst case is 9.6e-07 (fp32) and
    1.8e-15 (fp64); the thresholds keep ~10x headroom.
    """
    for dtype, tol in ((torch.float32, 1e-5), (torch.float64, 1e-13)):
        gate = torch.cat([
            torch.linspace(-64.0, 64.0, 200001, dtype=dtype),
            torch.linspace(-1000.0, 1000.0, 200001, dtype=dtype),
        ])
        gate_ref, up_ref = _situ_reference(gate, gate)
        gate_new, up_new = _situ_equal_transform(gate, gate)
        gate_err = (gate_new - gate_ref).abs().max().item()
        up_err = (up_new - up_ref).abs().max().item()
        assert gate_err < tol, f"SiTU gate rewrite error {gate_err} exceeds {tol} ({dtype})"
        assert up_err < tol, f"SiTU up rewrite error {up_err} exceeds {tol} ({dtype})"
        print(f" > {dtype}: gate max|d|={gate_err:.3e}  up max|d|={up_err:.3e}")


if __name__ == "__main__":
    test_situ_equal_transform_accuracy()
    print("SiTU equal-transform identity holds")
