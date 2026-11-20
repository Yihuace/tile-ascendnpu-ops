"""Performance benchmark for RopeLlama31Op (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_rope.py`` -- the Llama 3.1
(``RopeLlama31Op``) portion only (the shared GPU file also covers the GPT-NeoX
and LongRoPE variants; only this migrated op's benchmark is ported).

Workload shapes, dtypes and layout are read from the manifest
(``tileops/manifest/position_encoding.yaml``) via ``workloads_to_params`` with
``include_extra=True`` (the ``layout`` param is carried in the extra dict).

Single-input note: the op has exactly one tensor input (``x``), so the
manifest-driven ``workloads_to_params`` / ``ManifestBenchmark`` path applies
directly.  ``cos`` / ``sin`` are computed by the op and are not manifest
inputs.

The pure-PyTorch rotation is a performance baseline (not an accuracy-only
golden), so it is kept and profiled under the ``torch-ref`` tag.  Its cos/sin
tables are built outside the timed window, so only the rotation is measured.

NPU adaptations (T3/T4): the GPU bench has no ``tune`` parametrization and no
autotune try/except.  Manual tensor creation and the baseline tables use the
device backend instead of the hard-coded ``"cuda"``.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.device import get_device_backend
from tileops.ops.position_encoding.rope_llama31 import RopeLlama31Op
from tileops.workloads.position_encoding import RopeLlama31OpWorkload

_LLAMA31_OP = "RopeLlama31Op"

# Bench-local: manifest workload entries carry no ``base``; the op and the
# baseline both use the manifest signature default (``base: 10000.0``).
_BASE = 10000.0


def _rope_tables(seq_len: int, head_dim: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Half-split cos/sin tables, shape ``(seq_len, head_dim)``.

    The timed rotation cost depends only on table geometry, which Llama 3.1
    shares with neox, so the neox baseline table serves here too.  Frequency
    values only affect accuracy, not the timed rotation cost.
    """
    half = head_dim // 2
    device = get_device_backend().name
    freqs = 1.0 / (
        _BASE ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    angles = torch.outer(
        torch.arange(seq_len, device=device, dtype=torch.float32), freqs,
    )
    return (torch.cat([torch.cos(angles)] * 2, dim=-1).to(dtype),
            torch.cat([torch.sin(angles)] * 2, dim=-1).to(dtype))


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_LLAMA31_OP, include_extra=True),
)
def test_rope_llama31_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    layout = op_params["layout"]
    if layout == "1d":
        seq_len, head_dim = shape
        batch = num_heads = 1
    else:
        batch, seq_len, num_heads, head_dim = shape

    test = RopeLlama31OpWorkload(
        layout, batch, seq_len, num_heads, head_dim, dtype, base=_BASE
    )
    (x,) = test.gen_inputs()

    # Llama 3.1 manifest workloads carry no scaling params; both the op and the
    # baseline use the signature defaults (scale_factor=8.0, low=1.0, high=4.0,
    # original_max_position=8192).
    op = RopeLlama31Op(layout=layout, base=_BASE)

    # Warmup: trigger JIT compilation before timed profiling.
    op(x)
    get_device_backend().synchronize()

    params = {"shape": shape, "dtype": dtype, "layout": layout}
    bm = ManifestBenchmark(_LLAMA31_OP, op, test)
    result = bm.profile(op, x)
    BenchmarkReport.record(op, params, result, tag="tileops")

    cos, sin = _rope_tables(seq_len, head_dim, dtype)
    if layout != "1d":
        cos, sin = (t.view(1, seq_len, 1, head_dim) for t in (cos, sin))
    result_bl = bm.profile(lambda t: _rotate(t, cos, sin), x)
    BenchmarkReport.record(op, params, result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
