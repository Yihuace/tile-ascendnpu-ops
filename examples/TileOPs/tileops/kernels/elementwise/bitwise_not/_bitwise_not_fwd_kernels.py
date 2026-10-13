"""Extracted TileLang kernel implementation for BitwiseNotFwdOp.

TileLang kernels: bitwise_not_fwd_kernel

The GPU ``BitwiseNotFwdKernel`` inherits ``UnaryKernel`` (``register_copy``
strategy; ``direct`` for a bool operand), whose ``@tilelang.jit`` builder is
``make_unary_regcopy`` / ``make_unary_direct`` in
``tileops/kernels/elementwise/_builders.py``.  The per-op pointwise math is the
dtype-branched ``op_func`` static method on ``BitwiseNotFwdKernel``::

    if   dtype == "bool":   y = (x == 0)          # logical NOT
    elif dtype == "uint8":  y = xor(x, 255)        # all-ones byte
    else:                   y = xor(x, -1)         # all-ones of the dtype

(``T.bitwise_xor`` with the all-ones constant is used because
``T.bitwise_not`` fails on vectorized ``int4`` CUDA types in the source repo.)

Because that op body has conditional/early returns, ``extract_tl_kernel.py``
cannot statically inline it (Pattern A) and falls back to Pattern B (raw
``make_unary_direct`` / ``make_unary_regcopy`` builders that call an unresolved
runtime ``op_func_for`` registry).  This file performs that merge by hand -- the
same merge Pattern A performs for a statically-defined ``op_func`` -- inlining
the dtype-branched bitwise-NOT body into the elementwise unary body to produce a
single self-contained ``bitwise_not_fwd_kernel`` factory.  The 1-D
``T.Parallel(block_size)`` loop is the ``register_copy`` elementwise map (the
``make_unary_regcopy`` ``T.Parallel(threads_arg, npt_arg)`` decomposition
collapsed into a single ``block_size`` axis, K11).

The factory is the GPU reference implementation (CUDA ``threads`` / ``npt``
execution model).  The NPU kernel component re-implements it for
``target="npuir"``, applying the K1-K4 / K11 adaptations (decorator, grid/sync,
``threads``/``npt`` collapse into ``block_size``, padding strategy).  The
factory's leading parameters ``(N, dtype, output_dtype, block_size)`` are the
interface the wrapper (``bitwise_not.py``) and the NPU re-implementation share;
only the trailing NPU-specific ``num_cores`` is reserved for the
re-implementation.
"""

import tilelang
import tilelang.language as T

__all__ = ["bitwise_not_fwd_kernel"]

# Default CUDA thread count for the reference elementwise body.  The NPU
# re-implementation replaces the whole CUDA threads x npt decomposition with a
# single ``block_size`` UB tile width (K11).
_THREADS = 128


@tilelang.jit(out_idx=[1])
def bitwise_not_fwd_kernel(N, dtype, output_dtype=None, block_size=8192, num_cores=None):
    """Bitwise NOT forward kernel: ``y = ~x`` for bool / integer inputs.

    Elementwise unary body (load -> compute -> store) with the dtype-branched
    bitwise-NOT inlined.  The ``op_func`` reproduces the GPU
    ``BitwiseNotFwdKernel.op_func`` semantics for every admitted dtype:
      - bool  -> ``x == 0``      (logical NOT),
      - uint8 -> ``xor(x, 255)`` (all-ones byte),
      - other -> ``xor(x, -1)``  (all-ones of the dtype).

    Compile-time params:
        N: flattened element count (input is a contiguous ``(N,)`` vector).
        dtype: input dtype, "bool" | "uint8" | "int8" | "int16" | "int32" | "int64".
        output_dtype: optional output dtype (defaults to ``dtype``).
        block_size: UB tile width (folds the GPU ``threads * npt`` product).
        num_cores: optional explicit kernel-count override -- reserved for the
            NPU re-implementation's persistent core split; ignored here.

    The returned ``main(x, y, shape)`` is invoked as ``kernel(x, N)``; ``y``
    is auto-allocated (``out_idx=[1]``).
    """
    out_dtype = output_dtype or dtype

    # Inlined BitwiseNotFwdKernel.op_func (dtype-branched bitwise NOT; the
    # branch is resolved at JIT trace time from the compile-time ``dtype``).
    def op_func(x):
        if dtype == "bool":
            return x == T.cast(0, "bool")
        if dtype == "uint8":
            return T.bitwise_xor(x, T.cast(255, "uint8"))
        return T.bitwise_xor(x, T.cast(-1, dtype))

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(T.ceildiv(N, block_size), threads=_THREADS) as bx:
            base = bx * block_size
            for i in T.Parallel(block_size):
                if base + i < shape:
                    y[base + i] = op_func(x[base + i])

    return main
