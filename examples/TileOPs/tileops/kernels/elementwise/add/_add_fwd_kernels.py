"""Extracted TileLang kernel implementation for AddFwdOp.

TileLang kernels: add_fwd_kernel

The GPU ``AddFwdKernel`` inherits ``AlphaScaledBinaryKernel(BinaryKernel)`` whose
``@tilelang.jit`` builders live in ``tileops/kernels/elementwise/_builders.py``
(``make_binary_direct`` / ``make_binary_explicit`` / ``make_binary_register_copy``).
The per-op pointwise math is the closure built by
``AlphaScaledBinaryKernel._alpha_op_func``::

    y = a + alpha * b                       # combine == addition for AddFwd

Because that op body is built dynamically (a closure over the construction-time
``alpha`` scalar, and the ``bool`` / integer / float branches), the extraction
script cannot statically inline it (Pattern A) and falls back to Pattern B (raw
builders).  This file performs that merge by hand -- the same merge Pattern A
performs for a statically-defined ``op_func`` -- inlining the alpha-scaled
addition op body into the ``make_binary_register_copy`` builder to produce a
single self-contained ``add_fwd_kernel`` factory parameterized by ``alpha``.

Design note (binary + broadcast): the GPU ``BinaryKernel`` picks
``register_copy`` when the two operands are contiguous and have the same shape,
and falls back to the stride-based ``explicit_parallel`` / ``direct`` builders
for broadcast.  Following the NPU elementwise precedent (``lerp_tensor`` /
``docs/gpu_to_npu_adaptation.md`` §3.4 E15), the broadcast semantics are
resolved at the Op layer (``AddFwdOp`` expands both operands to
``broadcast_shapes(input, other)`` and flattens them) so this kernel is the
flat, same-shape ``register_copy`` path -- the one with the simplest NPUIR
re-implementation.  The broadcast contract (``torch.add``) is preserved
faithfully end to end.

The factory is the GPU reference implementation (CUDA ``threads``/``npt``
execution model).  The NPU kernel component re-implements it for
``target="npuir"``, applying the K1-K4 / K11 adaptations (decorator, grid/sync,
``threads``/``npt`` collapse into ``block_size``, padding strategy).  The
factory's leading parameters ``(N, dtype, alpha, output_dtype)`` are the
interface the wrapper (``add.py``) and the NPU re-implementation share; only
the trailing GPU execution parameters are adapted.
"""

import functools

import tilelang
import tilelang.language as T
from tvm import DataType

__all__ = ["add_fwd_kernel"]


# ---------------------------------------------------------------------------
# Builder helper (ported from tileops/kernels/elementwise/_builders.py).
# ---------------------------------------------------------------------------


def _narrows(dtype, out_dtype):
    """Whether a result is stored narrower than its operand."""

    return DataType(out_dtype).bits < DataType(dtype).bits


# Bitwise/integer dtypes take native integer arithmetic (the GPU
# ``AlphaScaledBinaryKernel`` routes them through ``BITWISE_DTYPES``); the
# trailing bool is grouped with the integers.  ``AddFwdOp`` restricts its
# element types to the floats, but the branch is kept faithful to the GPU
# builder so the reference is complete.
_INT_DTYPES = ("bool", "uint8", "int8", "int16", "int32", "int64")


# ---------------------------------------------------------------------------
# TileLang kernel function (merged make_binary_register_copy + AddFwd alpha body).
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def add_fwd_kernel(N, dtype, alpha=1, output_dtype=None, threads=256, num_per_thread=8):
    """Add (scalar-alpha) forward kernel: ``y = a + alpha * b``.

    Elementwise binary ``register_copy`` strategy (fragment load -> compute ->
    fragment store), with the alpha-scaled addition inlined.  ``alpha`` is baked
    into the op body, so every distinct ``alpha`` is its own specialization and
    the kernel surface stays scalar-free.  Float inputs route the scalar
    multiply through fp32 to dodge narrow-type literal issues for fp16 / bf16;
    integer / bool inputs keep native integer arithmetic (the GPU
    ``AlphaScaledBinaryKernel`` behavior).

    ``threads`` / ``npt`` are the GPU execution parameters; the NPU component
    collapses ``threads * npt`` into a single ``block_size`` (K11).
    """

    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[2])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        # Inlined AlphaScaledBinaryKernel._alpha_op_func op body (AddFwd's
        # ``_combine`` is plain addition).
        def op_func(a, b):
            if alpha == 1:
                # Identity multiplier: skip the scalar multiply.
                return a + b
            if dtype in _INT_DTYPES:
                # Native integer arithmetic; coerce alpha into the input
                # dtype's representable range in Python (PyTorch "scalar wraps
                # via the input dtype" semantics).
                int_alpha = int(bool(alpha)) if dtype == "bool" else int(alpha)
                scaled_b = T.cast(int_alpha, a.dtype) * b
                return a + scaled_b
            scaled_b = T.cast(T.cast(alpha, "float32") * T.cast(b, "float32"), a.dtype)
            return a + scaled_b

        # A narrower result is stored from a thread's own contiguous run, in one
        # store; the fragment layout would split it into one store per vector.
        contiguous = _narrows(dtype, out_dtype) and N % block_size == 0

        @T.prim_func
        def main(
            a: T.Tensor((N,), dtype),
            b: T.Tensor((N,), dtype),
            y: T.Tensor((N,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                if contiguous:
                    base = bx * block_size + T.get_thread_binding() * npt_arg
                    a_run = T.alloc_local((npt_arg,), dtype)
                    b_run = T.alloc_local((npt_arg,), dtype)
                    y_run = T.alloc_local((npt_arg,), out_dtype)
                    for v in T.vectorized(npt_arg):
                        a_run[v] = a[base + v]
                    for v in T.vectorized(npt_arg):
                        b_run[v] = b[base + v]
                    for v in T.unroll(npt_arg):
                        y_run[v] = op_func(a_run[v], b_run[v])
                    for v in T.vectorized(npt_arg):
                        y[base + v] = y_run[v]
                else:
                    a_reg = T.alloc_fragment((block_size,), dtype)
                    b_reg = T.alloc_fragment((block_size,), dtype)
                    y_reg = T.alloc_fragment((block_size,), out_dtype)
                    T.copy(a[bx * block_size : (bx + 1) * block_size], a_reg)
                    T.copy(b[bx * block_size : (bx + 1) * block_size], b_reg)
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        idx = i * npt_arg + j
                        y_reg[idx] = op_func(a_reg[idx], b_reg[idx])
                    T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel
