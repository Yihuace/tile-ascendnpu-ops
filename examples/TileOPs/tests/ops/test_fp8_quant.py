"""Correctness tests for FP8QuantOp (NPU).

Adapted from TileOPs ``tests/ops/test_fp8_quant.py`` (the GPU file covers only
this op, so the whole case set is ported).

NPU adaptations (T1-T4):

- T1: there is no manual tensor creation in the GPU file -- input generation
  lives in ``FP8QuantWorkload.gen_inputs`` (W1-adapted), so no
  ``device="cuda"`` remains to replace.
- T2: no ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)`` guard in
  the GPU file -- nothing to remove.
- T3: the ``tune`` parametrization dimension is removed (NPU has no autotune).
- T4: no autotune try/except in the GPU file -- nothing to remove.

The fp8 output has only 3 mantissa bits, so the GPU file compares with cosine
similarity rather than ``assert_close``; that comparator is ported verbatim.

Golden-availability adaptation (S7 smoke): this toolchain's eager
``float32/float16/bfloat16 -> float8_e4m3fn`` cast is unsupported on device, so
``ref_program`` performs that single cast on CPU and moves the fp8 result back
to the input device (see the comment in ``ref_program`` for the measured
error).  No fixture combo raises any more.
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.ops.quantization.fp8_quant import FP8QuantOp
from tileops.testing.test_base import TestBase
from tileops.workloads.quantization import FP8QuantWorkload
from tileops.workloads.workload_base import FixtureBase


class FP8QuantTest(FP8QuantWorkload, TestBase):
    def ref_program(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # input_tensor: (batch, seq_len_kv, kv_group, index_dim)
        amax_value = torch.abs(input_tensor).amax(dim=-1, keepdim=True).clamp(min=1e-4)
        scale_tensor = amax_value / 448.0
        output_tensor = torch.clamp(input_tensor / scale_tensor, min=-448.0, max=448.0)
        # NPU-adapted cast (S7 golden-availability smoke): this toolchain's
        # eager fp32/fp16/bf16 -> float8_e4m3fn cast is unsupported on device --
        # ``output_tensor.to(torch.float8_e4m3fn)`` raises
        # ``RuntimeError: copy_d2d_baseformat_opapi ... aclnnInplaceCopy failed,
        # error code is 561103`` for every fixture dtype/shape (verified on this
        # toolchain, 2026-10-14).  fp8 quantization is exact per element
        # (abs + max reduction + divide + clamp, no cross-element
        # accumulation), so the cast is performed on CPU and the fp8 result is
        # moved back to the input device; this is bit-identical to the intended
        # on-device result.  Everything else (scale, and the cosine-similarity
        # comparison) is unchanged from the GPU reference.
        output_tensor = output_tensor.cpu().to(torch.float8_e4m3fn).to(input_tensor.device)
        return scale_tensor.squeeze(dim=-1), output_tensor


class FP8QuantFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len_kv, kv_group, index_dim, in_dtype",
            [
                # Smoke: tiny kv-index tile across all three input dtypes (fp32
                # first per the NPU test-structure contract; one is also
                # ``packaging``).
                pytest.param(
                    1,
                    256,
                    1,
                    64,
                    torch.float32,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-256-d64-fp32",
                ),
                pytest.param(
                    1, 256, 1, 64, torch.float16, marks=pytest.mark.smoke, id="smoke-256-d64-fp16"
                ),
                pytest.param(
                    1, 256, 1, 64, torch.bfloat16, marks=pytest.mark.smoke, id="smoke-256-d64-bf16"
                ),
                # Full: GPU case set, copied verbatim (``tune`` dropped).
                pytest.param(
                    1, 8192, 1, 64, torch.float16, marks=pytest.mark.full, id="kv-index-8k-d64-fp16"
                ),
                pytest.param(
                    1,
                    8192,
                    1,
                    64,
                    torch.bfloat16,
                    marks=pytest.mark.full,
                    id="kv-index-8k-d64-bf16",
                ),
                pytest.param(
                    1,
                    4096,
                    1,
                    128,
                    torch.float32,
                    marks=pytest.mark.full,
                    id="kv-index-4k-d128-fp32",
                ),
                pytest.param(
                    1,
                    16384,
                    1,
                    32,
                    torch.float32,
                    marks=pytest.mark.full,
                    id="kv-index-16k-d32-fp32",
                ),
                pytest.param(
                    1,
                    1024,
                    4,
                    64,
                    torch.float16,
                    marks=pytest.mark.full,
                    id="kv-index-1k-g4-d64-fp16",
                ),
            ],
        ),
    ]


def _cosine_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Compare using cosine similarity (fp8 quantization needs looser checks).

    NPU-adapted from the GPU comparator: the fp8 upcast is CPU-routed because
    this toolchain cannot cast ``float8_e4m3fn`` on device (``aclnnInplaceCopy``
    error 561103, both directions, every shape).  Both fp8 operands therefore
    reach fp32 on CPU (identical rounding); fp32 operands keep the on-device
    cast.  The ``>= 0.99`` threshold is unchanged.
    """
    output = (
        output.cpu().to(torch.float32)
        if output.dtype == torch.float8_e4m3fn
        else output.to(torch.float32)
    )
    output_ref = (
        output_ref.cpu().to(torch.float32)
        if output_ref.dtype == torch.float8_e4m3fn
        else output_ref.to(torch.float32)
    )
    cos_sim = F.cosine_similarity(output.flatten(), output_ref.flatten(), dim=0)
    assert cos_sim >= 0.99, f"Cosine similarity too low: {cos_sim.item()}"


@FP8QuantFixture
def test_fp8_quant_op(
    batch: int, seq_len_kv: int, kv_group: int, index_dim: int, in_dtype: torch.dtype
) -> None:
    test = FP8QuantTest(batch, seq_len_kv, kv_group, index_dim, in_dtype)
    op = FP8QuantOp()
    test.check(op, *test.gen_inputs(), compare=_cosine_compare)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
