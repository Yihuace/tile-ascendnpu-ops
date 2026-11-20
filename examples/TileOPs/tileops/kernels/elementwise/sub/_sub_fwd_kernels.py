"""Extracted TileLang kernel implementation for SubFwdOp.

TileLang kernels: sub_fwd_kernel

The GPU ``SubFwdKernel`` inherits ``AlphaScaledBinaryKernel(BinaryKernel)`` whose
``@tilelang.jit`` builders live in ``tileops/kernels/elementwise/_builders.py``
(``make_binary_direct`` / ``make_binary_explicit`` / ``make_binary_register_copy``).
The per-op pointwise math is the closure built by
``AlphaScaledBinaryKernel._alpha_op_func``::

    y = a - alpha * b                       # combine == subtraction for SubFwd

Because that op body is built dynamically (a closure over the construction-time
``alpha`` scalar, and the ``bool`` / integer / float branches), the extraction
script cannot statically inline it (Pattern A) and falls back to Pattern B (raw
builders).  This file performs that merge by hand -- the same merge Pattern A
performs for a statically-defined ``op_func`` -- inlining the alpha-scaled
subtraction op body into the ``make_binary_register_copy`` builder to produce a
single self-contained ``sub_fwd_kernel`` factory parameterized by ``alpha``.

Design note (binary + broadcast): the GPU ``BinaryKernel`` picks
``register_copy`` when the two operands are contiguous and have the same shape,
and falls back to the stride-based ``explicit_parallel`` / ``direct`` builders
for broadcast.  Following the NPU elementwise precedent (``lerp_tensor`` /
``docs/gpu_to_npu_adaptation.md`` section 3.4 E15), the broadcast semantics are
resolved at the Op layer (``SubFwdOp`` expands both operands to
``broadcast_shapes(input, other)`` and flattens them) so this kernel is the
flat, same-shape ``register_copy`` path -- the one with the simplest NPUIR
re-implementation.  The broadcast contract (``torch.sub``) is preserved
faithfully end to end.

The factory is the GPU reference implementation (CUDA execution model).  The
NPU kernel component re-implements it for ``target="npuir"``, applying the
K1-K4 / K11 adaptations (decorator, grid/sync, ``threads``/``npt`` collapse
into ``block_size``, padding strategy).  The factory's leading parameters
``(N, dtype, alpha, output_dtype, block_size)`` are the interface the wrapper
(``sub.py``) and the NPU re-implementation share; only the trailing
NPU-specific ``num_cores`` is reserved for the re-implementation.
"""

import tilelang
import tilelang.language as T
from tvm import DataType

__all__ = ["sub_fwd_kernel"]


# Bitwise/integer dtypes take native integer arithmetic (the GPU
# ``AlphaScaledBinaryKernel`` routes them through ``BITWISE_DTYPES``); the
# trailing bool is grouped with the integers.  ``SubFwdOp`` restricts its
# element types to the floats, but the branch is kept faithful to the GPU
# builder so the reference is complete.
_INT_DTYPES = ("bool", "uint8", "int8", "int16", "int32", "int64")

# Default CUDA thread count for the reference register_copy body.  The NPU
# re-implementation replaces the whole CUDA threads x npt decomposition with a
# single ``block_size`` UB tile width (K11).
_THREADS = 128


def _narrows(dtype, out_dtype):
    """Whether a result is stored narrower than its operand."""

    return DataType(out_dtype).bits < DataType(dtype).bits


# ---------------------------------------------------------------------------
# TileLang kernel function (merged make_binary_register_copy + SubFwd alpha body).
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[2])
def sub_fwd_kernel(N, dtype, alpha=1, output_dtype=None, block_size=8192, num_cores=None):
    """Subtract (scalar-alpha) forward kernel: ``y = a - alpha * b``.

    Elementwise binary ``register_copy`` strategy (load -> compute -> store),
    with the alpha-scaled subtraction inlined.  ``alpha`` is baked into the op
    body, so every distinct ``alpha`` is its own specialization and the kernel
    surface stays scalar-free.  Float inputs route the scalar multiply through
    fp32 to dodge narrow-type literal issues for fp16 / bf16; integer / bool
    inputs keep native integer arithmetic (the GPU ``AlphaScaledBinaryKernel``
    behavior).

    Compile-time params:
        N: flattened element count (each input is a contiguous ``(N,)`` vector).
        dtype: input dtype, "float16" | "bfloat16" | "float32".
        alpha: multiplier on ``b`` (int | float, default 1); baked in.
        output_dtype: optional output dtype (defaults to ``dtype``).
        block_size: UB tile width (folds the GPU ``threads * npt`` product).
        num_cores: optional explicit kernel-count override -- reserved for the
            NPU re-implementation's persistent core split; ignored here.

    The returned ``main(a, b, y, shape)`` is invoked as ``kernel(a, b, N)``;
    ``y`` is auto-allocated (``out_idx=[2]``).
    """
    out_dtype = output_dtype or dtype

    # Inlined AlphaScaledBinaryKernel._alpha_op_func op body (SubFwd's
    # ``_combine`` is plain subtraction).
    def op_func(a, b):
        if alpha == 1:
            # Identity multiplier: skip the scalar multiply.
            return a - b
        if dtype in _INT_DTYPES:
            # Native integer arithmetic; coerce alpha into the input dtype's
            # representable range in Python (PyTorch "scalar wraps via the
            # input dtype" semantics).
            int_alpha = int(bool(alpha)) if dtype == "bool" else int(alpha)
            scaled_b = T.cast(int_alpha, a.dtype) * b
            return a - scaled_b
        scaled_b = T.cast(T.cast(alpha, "float32") * T.cast(b, "float32"), a.dtype)
        return a - scaled_b

    @T.prim_func
    def main(
        a: T.Tensor((N,), dtype),
        b: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(T.ceildiv(N, block_size), threads=_THREADS) as bx:
            base = bx * block_size
            for i in T.Parallel(block_size):
                if base + i < shape:
                    y[base + i] = op_func(a[base + i], b[base + i])

    return main
