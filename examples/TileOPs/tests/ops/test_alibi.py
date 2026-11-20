"""Correctness tests for AlibiFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_special_elementwise.py`` — the
``alibi`` subset only (the shared GPU file also covers the Tensor-bound
clamp ops; only this migrated op's cases are ported).

``AlibiFwdOp`` is a generative op: it has no tensor input.  The full
``(num_heads, seq_len, seq_len)`` bias is synthesized from the
construction-time integer parameters, so the reference below is computed
directly from ``slope_h = 2**(-8*(h+1)/H)`` and ``|i - j|``.

Adaptations from GPU (TileOPs) to NPU:

- T1: tensor creation uses ``get_device_backend().name`` instead of the
  hard-coded ``"cuda"``.
- T2: the GPU ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)``
  guard is dropped (the NPU device backend resolves the target).
- T3/T4: the GPU ``tune`` parametrization and autotune try/except are
  omitted — the NPU kernel uses heuristic config selection only.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.elementwise.alibi import AlibiFwdOp
from tileops.workloads.workload_base import FixtureBase


def _device() -> str:
    return get_device_backend().name


class AlibiFixture(FixtureBase):
    PARAMS = [
        ("seq_len, num_heads, dtype", [
            pytest.param(128, 8, torch.float16, marks=pytest.mark.smoke),
            pytest.param(128, 8, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@AlibiFixture
def test_alibi(seq_len: int, num_heads: int, dtype: torch.dtype) -> None:
    op = AlibiFwdOp(seq_len=seq_len, num_heads=num_heads, dtype=dtype)
    out = op()

    # Reference: slope_h = 2^(-8*(h+1)/H), bias = -slope * |i - j|
    positions = torch.arange(seq_len, device=_device(), dtype=torch.float32)
    dist = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()
    slopes = torch.pow(
        2.0,
        -8.0 * torch.arange(1, num_heads + 1, device=_device(), dtype=torch.float32) / num_heads,
    )
    ref = (-slopes[:, None, None] * dist[None, :, :]).to(dtype)

    tol = {"atol": 1e-2, "rtol": 1e-2} if dtype == torch.float16 else {"atol": 1e-5, "rtol": 1e-5}
    torch.testing.assert_close(out, ref, **tol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
