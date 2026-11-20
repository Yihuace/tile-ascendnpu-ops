"""Extracted TileLang kernel implementation for HardswishFwdOp.

TileLang kernels: hardswish_fwd_kernel

The GPU ``HardswishFwdKernel`` inherits ``FloatUnaryKernel`` (register_copy
strategy), whose ``@tilelang.jit`` builder is ``make_unary_regcopy`` in
``tileops/kernels/elementwise/_builders.py``.  The per-op pointwise math is the
``op_func`` static method on ``HardswishFwdKernel``::

    y = x * clamp(x + 3, 0, 6) * (1 / 6)

This file inlines that ``op_func`` into the ``make_unary_regcopy`` builder to
produce a single self-contained ``hardswish_fwd_kernel`` factory (the same
merge ``extract_tl_kernel.py`` Pattern A performs).  Unlike Hardsigmoid, the
HardSwish body uses plain ``T.min`` / ``T.max`` -- it needs no NaN-propagating
helpers, so only the ``_narrows`` builder helper is carried over.

The factory is the GPU reference implementation (CUDA ``threads``/``npt``
execution model).  The NPU kernel component re-implements it for
``target="npuir"``, applying the K1-K4 / K11 adaptations (decorator, grid/sync,
``threads``/``npt`` collapse into ``block_size``, padding strategy).
"""

import functools

import tilelang
import tilelang.language as T
from tvm import DataType

__all__ = ["hardswish_fwd_kernel"]


# ---------------------------------------------------------------------------
# Builder helper (ported from tileops/kernels/elementwise/_builders.py).
# ---------------------------------------------------------------------------


def _narrows(dtype, out_dtype):
    """Whether a result is stored narrower than its operand."""

    return DataType(out_dtype).bits < DataType(dtype).bits


# ---------------------------------------------------------------------------
# TileLang kernel function (merged make_unary_regcopy + HardswishFwdKernel.op_func).
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def hardswish_fwd_kernel(N, dtype, output_dtype=None, threads=256, num_per_thread=8):
    """Hardswish forward kernel: ``y = x * clamp(x + 3, 0, 6) * (1 / 6)``.

    Elementwise unary ``register_copy`` strategy (fragment load -> compute ->
    fragment store), with the per-op math inlined.  ``threads`` / ``npt`` are
    the GPU execution parameters; the NPU component collapses
    ``threads * npt`` into a single ``block_size`` (K11).
    """

    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        # Inlined HardswishFwdKernel.op_func locals (parameter-free constants).
        three = T.cast(3.0, "float32")
        six = T.cast(6.0, "float32")
        zero = T.cast(0.0, "float32")
        one_sixth = T.cast(1.0 / 6.0, "float32")

        # A narrower result is stored from a thread's own contiguous run, in one
        # store; the fragment layout would split it into one store per vector.
        contiguous = _narrows(dtype, out_dtype) and N % block_size == 0

        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                if contiguous:
                    base = bx * block_size + T.get_thread_binding() * npt_arg
                    x_run = T.alloc_local((npt_arg,), dtype)
                    y_run = T.alloc_local((npt_arg,), out_dtype)
                    for v in T.vectorized(npt_arg):
                        x_run[v] = x[base + v]
                    for v in T.unroll(npt_arg):
                        y_run[v] = (
                            x_run[v] * T.min(T.max(x_run[v] + three, zero), six) * one_sixth
                        )
                    for v in T.vectorized(npt_arg):
                        y[base + v] = y_run[v]
                else:
                    x_reg = T.alloc_fragment((block_size,), dtype)
                    y_reg = T.alloc_fragment((block_size,), out_dtype)
                    T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        y_reg[i * npt_arg + j] = (
                            x_reg[i * npt_arg + j]
                            * T.min(T.max(x_reg[i * npt_arg + j] + three, zero), six)
                            * one_sixth
                        )
                    T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel
