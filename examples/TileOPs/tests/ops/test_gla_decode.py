"""Correctness tests for GLADecodeOp (NPU).

``GLADecodeOp`` (baseline ``GLARecurrentFwdOp``) is the single-token gated
linear attention decode.  The independent golden is the pure-torch CPU
recurrence (``GLADecodeWorkload.ref_program``), independent of the NPU
Vector/k-tiled algorithm.

Coverage: fp16/bf16/bf32, the manifest serving shapes (DK=DV=128), non-tile
shapes, and a custom readout scale.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.linear_attention.gla_decode import GLADecodeOp
from tileops.workloads.linear_attention import GLADecodeWorkload, gla_decode_tolerance

_F16 = torch.float16
_BF16 = torch.bfloat16
_F32 = torch.float32

# (B, H, DK, DV, dtype, scale, id)
_CASES = [
    (2, 4, 64, 64, _F16, -1.0, "smoke-d64-fp16"),
    (2, 4, 64, 64, _BF16, -1.0, "smoke-d64-bf16"),
    (1, 8, 128, 128, _BF16, -1.0, "serving-b1-h8"),
    (1, 32, 128, 128, _BF16, -1.0, "serving-b1-h32"),
    (8, 32, 128, 128, _BF16, -1.0, "serving-b8-h32"),
    (1, 2, 32, 16, _F32, -1.0, "f32-untiled"),
    (2, 3, 16, 16, _F16, -1.0, "small-k16"),
    (1, 4, 128, 64, _F16, -1.0, "rect-d128x64"),
    (2, 3, 64, 64, _BF16, 0.25, "custom-scale"),
]


def _device() -> str:
    return get_device_backend().name


def _mark(idx: int, dtype: torch.dtype, case_id: str):
    if idx == 0:
        return [pytest.mark.smoke, pytest.mark.packaging]
    if "smoke" in case_id or "serving" in case_id:
        return pytest.mark.smoke
    return pytest.mark.full


@pytest.mark.parametrize(
    "B, H, DK, DV, dtype, scale, case_id",
    [pytest.param(*c, id=c[-1], marks=_mark(i, c[4], c[-1])) for i, c in enumerate(_CASES)],
)
def test_gla_decode(B, H, DK, DV, dtype, scale, case_id) -> None:
    test = GLADecodeWorkload(B, H, DK, DV, dtype, scale)
    q, k, v, gk, state = test.gen_inputs()
    op = GLADecodeOp(B, H, DK, DV, scale=scale, dtype=dtype)
    o, ns = op(q, k, v, gk, state)
    ref_o, ref_ns = test.ref_program(q, k, v, gk, state)
    tol = gla_decode_tolerance(dtype)
    err_o = float((o.cpu().float() - ref_o).abs().max().item())
    err_ns = float((ns.cpu().float() - ref_ns).abs().max().item())
    torch.testing.assert_close(o.cpu().float(), ref_o, atol=tol["atol"], rtol=tol["rtol"])
    torch.testing.assert_close(ns.cpu().float(), ref_ns, atol=tol["atol"], rtol=tol["rtol"])
    print(f"[{case_id}] max_abs_err o={err_o:.3e} new_state={err_ns:.3e} (tol={tol['atol']:.1e})")


@pytest.mark.smoke
def test_gla_decode_default_scale() -> None:
    """A non-positive scale resolves to ``DK ** -0.5``."""
    op = GLADecodeOp(1, 2, 64, 64, scale=-1.0, dtype=torch.float16)
    assert op.kernel.scale == pytest.approx(64 ** -0.5)


@pytest.mark.smoke
def test_gla_decode_eval_roofline() -> None:
    B, H, DK, DV = 1, 8, 128, 128
    op = GLADecodeOp(B, H, DK, DV, dtype=torch.bfloat16)
    flops, nbytes = op.eval_roofline()
    assert flops == B * H * (5 * DK * DV + DK + DV)
    assert nbytes == B * H * (3 * DK + 2 * DV + 2 * DK * DV) * 2


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
