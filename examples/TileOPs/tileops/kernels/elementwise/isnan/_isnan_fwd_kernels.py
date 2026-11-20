"""Extracted TileLang kernel implementation for IsnanFwdOp.

TileLang kernels: isnan_fwd_kernel

The GPU ``IsnanFwdKernel`` inherits ``FloatPredicateKernel`` (``register_copy``
strategy) with ``OUTPUT_DTYPE = torch.bool`` and the per-op pointwise math::

    op_func(x) = T.isnan(T.cast(x, "float32"))

``FloatPredicateKernel`` derives from ``FloatUnaryKernel`` -> ``UnaryKernel``,
whose ``@tilelang.jit`` builder is ``make_unary_regcopy`` in
``tileops/kernels/elementwise/_builders.py``.  Because the bool output goes
through storage, ``UnaryKernel`` wraps the body with
``store_unary_bool_as_int8`` (``BOOL_STORAGE_DTYPE = "int8"``)::

    y = T.if_then_else(op_func(x), T.cast(1, "int8"), T.cast(0, "int8"))

The op's ``kernel_map`` also declares ``AlwaysFalseFwdKernel`` (a
``TorchFallbackKernel`` answering "False everywhere" for the integer / bool
inputs of the ``isnan_exact`` variant).  Only the standard float ``isnan`` path
is migrated here, so this file carries the single float-predicate kernel.

``extract_tl_kernel.py`` cannot statically inline the op body for this op
(``UnaryKernel`` resolves it at runtime through the ``register_op_func`` /
``op_func_for`` registry) and falls back to Pattern B (the raw
``make_unary_direct`` / ``make_unary_regcopy`` builders that call the unresolved
``op_func_for(op_name)``).  This file performs the merge by hand -- the same
merge Pattern A performs for a statically-defined ``op_func`` -- inlining the
``isnan`` predicate plus the bool-as-int8 store into the elementwise unary body
to produce a single self-contained ``isnan_fwd_kernel`` factory.  The 1-D
``T.Parallel(block_size)`` loop is the elementwise map (the
``make_unary_regcopy`` ``T.Parallel(threads_arg, npt_arg)`` decomposition
collapsed into a single ``block_size`` axis, K11).

The factory is the GPU reference implementation (CUDA ``threads`` / ``npt``
execution model).  The NPU kernel component re-implements it for
``target="npuir"``, applying the K1-K4 / K11 adaptations (decorator, grid/sync,
``threads``/``npt`` collapse into ``block_size``, padding strategy).  The
factory's leading parameters ``(N, dtype, output_dtype, block_size)`` are the
interface the wrapper (``isnan.py``) and the NPU re-implementation share; only
the trailing NPU-specific ``num_cores`` is reserved for the re-implementation.
"""

import tilelang
import tilelang.language as T

__all__ = ["isnan_fwd_kernel"]

# Default CUDA thread count for the reference elementwise body (the GPU
# ``_DEFAULT_THREADS``).  The NPU re-implementation replaces the whole CUDA
# ``threads`` x ``npt`` decomposition with a single ``block_size`` UB tile width
# (K11).
_THREADS = 128

# Bool storage dtype (``BOOL_STORAGE_DTYPE``): bool i1 cannot load/store GM, so
# the predicate result is written as a one-byte int8 0/1 and the caller views it
# back to bool (see ``isnan.py``).
_BOOL_STORAGE_DTYPE = "int8"


@tilelang.jit(out_idx=[1])
def isnan_fwd_kernel(N, dtype, output_dtype=None, block_size=8192, num_cores=None):
    """Isnan forward kernel: ``y = isnan(x)`` (bool output, int8 byte storage).

    Elementwise unary body (load -> compute -> store) with the float-predicate
    ``isnan`` inlined.  The body reproduces the GPU
    ``IsnanFwdKernel(op_func(x) = T.isnan(T.cast(x, "float32")))`` semantics,
    wrapped by ``store_unary_bool_as_int8`` so the bool result is written as an
    ``int8`` 1/0 byte::

        y = T.if_then_else(T.isnan(T.cast(x, "float32")), 1, 0)   # int8

    The NPU re-implementation may compute the same predicate with the
    self-compare ``x != x`` idiom (NaN is the only value unequal to itself),
    which avoids a cast and a special function.

    Compile-time params:
        N: flattened element count (input is a contiguous ``(N,)`` vector).
        dtype: input dtype, "float16" | "bfloat16" | "float32".
        output_dtype: optional output (storage) dtype; defaults to ``"int8"``
            (the bool byte storage).
        block_size: UB tile width (folds the GPU ``threads * npt`` product).
        num_cores: optional explicit kernel-count override -- reserved for the
            NPU re-implementation's persistent core split; ignored here.

    The returned ``main(x, y, shape)`` is invoked as ``kernel(x, N)``; ``y`` is
    auto-allocated (``out_idx=[1]``).
    """
    out_dtype = output_dtype or _BOOL_STORAGE_DTYPE

    # Inlined IsnanFwdKernel.op_func + store_unary_bool_as_int8: NaN detection
    # in fp32, stored as an int8 0/1 byte.
    def op_func(x):
        return T.if_then_else(
            T.isnan(T.cast(x, "float32")),
            T.cast(1, out_dtype),
            T.cast(0, out_dtype),
        )

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
