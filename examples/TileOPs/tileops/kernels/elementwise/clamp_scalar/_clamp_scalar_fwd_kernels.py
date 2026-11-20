"""Extracted TileLang kernel implementation for ClampScalarFwdOp.

TileLang kernels: clamp_scalar_fwd_kernel

The GPU ``ClampFwdKernel`` inherits ``ScalarParamUnaryKernel`` (register_copy
strategy), whose ``@tilelang.jit`` builder is ``make_unary_regcopy`` in
``tileops/kernels/elementwise/_builders.py``.  The per-op pointwise math is the
closure built by ``ClampFwdKernel._make_op_func``::

    wide = cast(x, float32)
    if min is not None: wide = nan_max(wide, cast(min, float32))
    if max is not None: wide = nan_min(wide, cast(max, float32))
    y = Cast(x.dtype, wide)                    # computed in fp32, cast back

Because that op body is built dynamically (a closure over the construction-time
``min`` / ``max`` scalars), the extraction script cannot statically inline it
(Pattern A) and falls back to Pattern B (raw builders).  This file performs that
merge by hand -- the same merge Pattern A performs for a statically-defined
``op_func`` -- inlining the clamp op body into the ``make_unary_regcopy`` builder
to produce a single self-contained ``clamp_scalar_fwd_kernel`` factory
parameterized by ``min_val`` / ``max_val``.

The NaN-propagating ``nan_min`` / ``nan_max`` helpers are ported verbatim from
``tileops/kernels/elementwise/_nan.py`` (``tvm.tirx`` -> ``tvm.tir``, the
equivalent module available under the NPU tilelang).

The factory is the GPU reference implementation (CUDA ``threads``/``npt``
execution model).  The NPU kernel component re-implements it for
``target="npuir"``, applying the K1-K4 / K11 adaptations (decorator, grid/sync,
``threads``/``npt`` collapse into ``block_size``, padding strategy).  The
factory's leading parameters ``(N, dtype, ..., min_val, max_val)`` are the
interface the wrapper (``clamp_scalar.py``) and the NPU re-implementation share;
only the trailing GPU execution parameters are adapted.
"""

import functools

import tilelang
import tilelang.language as T

# ``import tilelang`` above registers the bundled ``tvm`` package on sys.modules,
# so ``tvm.tir`` (the NPU-equivalent of the GPU's ``tvm.tirx``) is importable.
import tvm.tir as tirx
from tvm import DataType

__all__ = ["clamp_scalar_fwd_kernel", "nan_max", "nan_min"]


# ---------------------------------------------------------------------------
# NaN-propagating min/max (ported verbatim from tileops/kernels/elementwise/_nan.py;
# the GPU's ``import tvm.tirx as tirx`` becomes ``import tvm.tir as tirx``).
# ---------------------------------------------------------------------------


def bound(x, body):
    """``body(x)`` with *x* evaluated once, however often *body* mentions it."""

    if isinstance(x, (tirx.Var, tirx.FloatImm, tirx.IntImm)):
        return body(x)
    var = tirx.Var("nan_operand", x.dtype)
    return tirx.Let(var, x, body(var))


def _is_nan(x):
    return T.isnan(T.Cast("float32", x))


def _propagate_nan(a, b, plain):
    """``plain(a, b)``, and NaN where either operand is NaN."""

    def select(a, b):
        nan = T.Cast(a.dtype, T.cast(float("nan"), "float32"))
        return tirx.Select(tirx.any(_is_nan(a), _is_nan(b)), nan, plain(a, b))

    return bound(a, lambda a: bound(b, lambda b: select(a, b)))


def _nan_intrin(name, a, b):
    """A TileLang NaN-propagating builtin, named the way TileLang names its own."""

    return tirx.call_intrin(a.dtype, tirx.op.Op.get(name), a, b)


def _select(a, b, plain, intrin):
    dtype = str(a.dtype)
    if not dtype.startswith(("float", "bfloat")):
        # Integer and bool have no NaN.
        return plain(a, b)
    if dtype in ("float16", "bfloat16"):
        return _nan_intrin(intrin, a, b)
    return _propagate_nan(a, b, plain)


def nan_max(a, b):
    """``max(a, b)``, NaN where either operand is NaN."""

    return _select(a, b, T.max, "tl.max_nan")


def nan_min(a, b):
    """``min(a, b)``, NaN where either operand is NaN."""

    return _select(a, b, T.min, "tl.min_nan")


# ---------------------------------------------------------------------------
# Builder helper (ported from tileops/kernels/elementwise/_builders.py).
# ---------------------------------------------------------------------------


def _narrows(dtype, out_dtype):
    """Whether a result is stored narrower than its operand."""

    return DataType(out_dtype).bits < DataType(dtype).bits


# ---------------------------------------------------------------------------
# TileLang kernel function (merged make_unary_regcopy + ClampFwdKernel op body).
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def clamp_scalar_fwd_kernel(
    N, dtype, output_dtype=None, min_val=None, max_val=None, threads=128, num_per_thread=8
):
    """Clamp (scalar bounds) forward kernel: ``y = clamp(x, min, max)``.

    Elementwise unary ``register_copy`` strategy (fragment load -> compute ->
    fragment store), with the per-op clamp math inlined.  The bounds are baked
    into the op body, so every distinct ``(min_val, max_val)`` is its own
    specialization; a bound that is ``None`` is not applied.  The comparison
    runs in float32 and the result is cast back to the input dtype, so a half
    input keeps the precision of the comparison; NaN in the input or in a bound
    propagates NaN, as in ``torch.clamp``.

    ``threads`` / ``npt`` are the GPU execution parameters; the NPU component
    collapses ``threads * npt`` into a single ``block_size`` (K11).
    """

    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        # Inlined ClampFwdKernel._make_op_func op body: upcast to fp32, apply
        # whichever bounds are present (NaN-propagating), cast back.
        def op_func(x):
            wide = T.cast(x, "float32")
            if min_val is not None:
                wide = nan_max(wide, T.cast(min_val, "float32"))
            if max_val is not None:
                wide = nan_min(wide, T.cast(max_val, "float32"))
            return T.Cast(x.dtype, wide)

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
                        y_run[v] = op_func(x_run[v])
                    for v in T.vectorized(npt_arg):
                        y[base + v] = y_run[v]
                else:
                    x_reg = T.alloc_fragment((block_size,), dtype)
                    y_reg = T.alloc_fragment((block_size,), out_dtype)
                    T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        y_reg[i * npt_arg + j] = op_func(x_reg[i * npt_arg + j])
                    T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel
