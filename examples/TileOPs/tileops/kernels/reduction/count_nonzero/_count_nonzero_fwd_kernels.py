"""Extracted TileLang kernel implementation.

TileLang kernels: _logical_reduce_kernel
Extracted from the GPU repo via ``extract_tl_kernel.py`` (pattern B) from repo
root: /workspace/TileOPs-gpu

Scope note (migration unit ``CountNonzeroFwdOp``): the GPU kernel source
(``tileops/kernels/reduction/logical_reduce.py``) defines two factories --
the baseline ``_logical_reduce_kernel`` (used by ``LogicalReduceKernel``) and
the N-tiled ``_logical_reduce_kernel_tiled`` (used only by the long-row tiling
fallback). This migration unit ports **only the baseline factory**; the tiled
variant is intentionally not extracted/migrated.

The factory is shared across the any/all/count_nonzero kinds via ``op_kind``;
this package drives it with ``op_kind="count_nonzero"`` (int64 output).

Helper functions referenced by the factory (``align_up``,
``_pad_value_for_op``) are not ``@tilelang.jit`` builders, so the extraction
script does not inline them; they are appended verbatim below so this module is
self-contained. ``align_up`` is copied verbatim from
``tileops/kernels/reduction/_primitives.py``; ``_pad_value_for_op`` is copied
verbatim from ``tileops/kernels/reduction/logical_reduce.py``.
"""

# --- external imports (preserved) ---
import tilelang.language as T
import functools
import tilelang

# --- module-level constants referenced by the kernels ---
DEFAULT_ALIGNMENT: int = 256


# --- helper functions (appended verbatim: referenced by the factory) ---
def align_up(n: int, alignment: int) -> int:
    """Round *n* up to the nearest multiple of *alignment*.

    Args:
        n: Value to align.
        alignment: Alignment boundary (must be positive).

    Returns:
        Smallest multiple of *alignment* that is >= *n*.

    Raises:
        ValueError: If *alignment* is not positive.
    """
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((n + alignment - 1) // alignment) * alignment


def _pad_value_for_op(op_kind: str) -> float:
    """Return the identity value for masked padding."""
    if op_kind == "all":
        return 1.0
    return 0.0


# --- TileLang kernel functions (@tilelang.jit) ---
def _logical_reduce_kernel(M: int, N: int, op_kind: str, dtype: str):
    """Build a TileLang any/all/count_nonzero kernel.

    Cast input to bool (0.0 or 1.0 in float32), then:
      - any: reduce_max over the row (1.0 if any element is non-zero)
      - all: reduce_min over the row (1.0 if all elements are non-zero)
      - count_nonzero: reduce_sum over the row (count of non-zero elements)

    Args:
        M: Number of rows (product of all leading dimensions).
        N: Original hidden dimension (last dim, before padding).
        op_kind: One of "any", "all", "count_nonzero".
        dtype: TileLang dtype string (e.g. "float16", "bfloat16", "float32").

    Returns:
        A TileLang JIT-compiled kernel factory accepting (block_m, threads).
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _needs_pad = N_padded != N
    _pad_val = _pad_value_for_op(op_kind)

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.macro
        def compute(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), "int8" if op_kind != "count_nonzero" else "int64"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                bool_vals = T.alloc_fragment((block_m, N_padded), "float32")
                result = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment(
                    (block_m,), "int8" if op_kind != "count_nonzero" else "int64"
                )

                if _needs_pad:
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.if_then_else(
                            T.And(pid_m * block_m + i < M, j < N),
                            T.cast(x[pid_m * block_m + i, j], "float32"),
                            T.cast(_pad_val, "float32"),
                        )
                else:
                    # Load via shared memory
                    T.copy(x[pid_m * block_m, 0], shared_buf)

                    # Cast to fp32
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                # Cast to bool (0.0 or 1.0)
                for i, j in T.Parallel(block_m, N_padded):
                    bool_vals[i, j] = T.if_then_else(x_f32[i, j] != 0.0, 1.0, 0.0)

                if op_kind == "any":
                    # any: result is 1 if max(bool_vals) == 1
                    T.reduce_max(bool_vals, result, dim=1)
                elif op_kind == "all":
                    # all: result is 1 if min(bool_vals) == 1
                    T.reduce_min(bool_vals, result, dim=1)
                else:
                    # count_nonzero: sum of bool values per row
                    T.reduce_sum(bool_vals, result, dim=1)

                if op_kind == "count_nonzero":
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(result[i], "int64")
                else:
                    # Cast result to int8 (bool representation: 0 or 1)
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(result[i] > 0.5, "int8")

                # Write output
                T.copy(out_local, out[pid_m * block_m])

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), "int8" if op_kind != "count_nonzero" else "int64"],  # noqa: F821
        ):
            compute(x, out)

        return main

    return _func


__all__ = ["_logical_reduce_kernel"]
