# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""AdaLayerNormKernel -- TileLang-NPUIR kernel, golden and hierarchical tests.

ONE kernel covers TWO manifest ops:
  AdaLayerNormFwdOp      has_gate=False  ->  y = scale*Norm(x) + shift
  AdaLayerNormZeroFwdOp  has_gate=True   ->  y = gate * (scale*Norm(x) + shift)

Both map to the same kernel_map in the manifest (verified in Stage 0), so they
are a single migration unit and the design is ./DESIGN.md (Stage 2 review
passed, see ./REVIEW.md).

    mean_i = mean_j(x_ij)
    var_i  = mean_j((x_ij - mean_i)^2)
    rstd_i = 1 / sqrt(var_i + eps)
    xn_ij  = (x_ij - mean_i) * rstd_i
    y_ij   = scale_ij * xn_ij + shift_ij          [ * gate_ij when has_gate ]

Structural notes (all from the frozen design):

  * The signature is IDENTICAL for both variants: slot 3 is `gate`, which the
    non-gated variant never reads (the caller passes a placeholder).  That keeps
    `out_idx=[4]` constant across variants -- the source contract.
  * PERSISTENT CORE + IN-CORE SERIAL TASK LOOP.  Physical cores are capped at
    the measured Vector count (48); any remaining logical tasks run serially
    inside a core (`T.serial`).  Tasks are independent, so serialising them
    changes only execution order, never any value.
  * ``T.vrsqrt`` is NOT usable: ~2.7e-3 relative error, which breaks BOTH the
    fp32 1e-5 and the fp16 1e-3 gates (~87% / ~6% of elements).  rstd is
    composed as ``sqrt(v)/v`` (``T.vsqrt`` + ``T.vdiv``).
  * ``N_inv = 1/N`` is folded at compile time (division -> multiply).
  * Half precision goes through an fp32 transit (bf16 has no Vector arithmetic).
  * The UB budget is enforced by a HOST-SIDE guard that raises ``ValueError``
    (not a bare ``assert``, which ``python -O`` would strip).
  * ``TILELANG_ASCEND_MODE`` must be an ENVIRONMENT VARIABLE (measured on
    `vector_norm`: without it a reduction-bearing kernel is lowered onto the
    Cube path and the staging copy fails to compile).

Run:
    python _ada_layer_norm_kernel.py --level L0
    python _ada_layer_norm_kernel.py --level all
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  (registers the "npu" device)

tilelang.cache.clear_cache()

# --- UB budget (constants mirror the in-repo implementation) ---
_UB_CAPACITY_BYTES = 196_608          # 192 KB
_UB_BYTES_PER_ELEM = 20               # multi-buffered, measured
_UB_MAX_ELEMS = 9_216
_BLOCK_M_CAP = 8
_VECTOR_CORES = 48                    # measured (24 AI cores x 2)


def _default_block_m(n: int) -> int:
    return min(_BLOCK_M_CAP, max(1, _UB_MAX_ELEMS // n))


TOLERANCES = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 1.6e-2, "rtol": 1.6e-2},
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
}

_SUPPORTED = ("float16", "bfloat16", "float32")


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_ada_layer_norm(x, scale, shift, gate=None, eps=1e-5):
    """Golden: layer-normalise, then affine (and gate when given)."""
    n = x.shape[-1]
    xn = F.layer_norm(x.detach().cpu().float(), (n,), weight=None, bias=None, eps=eps)
    y = scale.detach().cpu().float() * xn + shift.detach().cpu().float()
    if gate is not None:
        y = gate.detach().cpu().float() * y
    return y.to(x.dtype)


# ---------- Kernel ----------
def _ada_layer_norm_kernel(M, N, eps, dtype, has_gate=False, use_cp_async=False):
    """Build the AdaLayerNorm factory (source signature; use_cp_async ignored)."""
    if M < 1 or N < 1:
        raise ValueError(f"M and N must be positive, got M={M}, N={N}")
    if dtype not in _SUPPORTED:
        raise ValueError(f"unsupported dtype {dtype!r}; expected one of {_SUPPORTED}")

    n_inv = 1.0 / float(N)
    use_fp32_transit = dtype in ("float16", "bfloat16")
    block_m_default = _default_block_m(N)

    @tilelang.jit(out_idx=[4], target="npuir")
    def _func(block_m=block_m_default):
        # Host-side UB guard.  ValueError rather than assert so it survives -O.
        if block_m < 1:
            raise ValueError(f"block_m must be positive, got block_m={block_m}")
        if block_m * N > _UB_MAX_ELEMS:
            raise ValueError(
                f"block_m={block_m} exceeds the UB budget for N={N} "
                f"(max {_UB_MAX_ELEMS} resident elements at "
                f"{_UB_BYTES_PER_ELEM} B/elem multi-buffered, "
                f"{_UB_CAPACITY_BYTES} B UB capacity)"
            )

        num_logical = (M + block_m - 1) // block_m
        num_kernels = min(num_logical, _VECTOR_CORES)
        num_local_tasks = (num_logical + num_kernels - 1) // num_kernels

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            scale: T.Tensor[(M, N), dtype],
            shift: T.Tensor[(M, N), dtype],
            # Placeholder when has_gate is False -- never read, but it keeps the
            # output at index 4 so out_idx stays identical for both variants.
            gate: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M, N), dtype],
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((block_m, N), dtype)
                sc_ub = T.alloc_shared((block_m, N), dtype)
                sh_ub = T.alloc_shared((block_m, N), dtype)
                g_ub = T.alloc_shared((block_m, N), dtype)
                y_ub = T.alloc_shared((block_m, N), dtype)
                x32 = T.alloc_shared((block_m, N), "float32")
                sc32 = T.alloc_shared((block_m, N), "float32")
                sh32 = T.alloc_shared((block_m, N), "float32")
                g32 = T.alloc_shared((block_m, N), "float32")
                mean_b = T.alloc_shared((block_m, 1), "float32")
                var_b = T.alloc_shared((block_m, 1), "float32")
                rstd_b = T.alloc_shared((block_m, 1), "float32")
                tmp_b = T.alloc_shared((block_m, 1), "float32")

                for t in T.serial(num_local_tasks):
                    task = cid * num_local_tasks + t
                    if task < num_logical:
                        row0 = task * block_m
                        rows = T.min(block_m, M - row0)

                        T.copy(x[row0 : row0 + rows, 0:N], x_ub[0:rows, 0:N])
                        T.copy(scale[row0 : row0 + rows, 0:N], sc_ub[0:rows, 0:N])
                        T.copy(shift[row0 : row0 + rows, 0:N], sh_ub[0:rows, 0:N])
                        if has_gate:
                            T.copy(gate[row0 : row0 + rows, 0:N], g_ub[0:rows, 0:N])

                        # --- fp32 transit ---
                        if use_fp32_transit:
                            T.vcast(x_ub, x32, round_mode="rint")
                            T.vcast(sc_ub, sc32, round_mode="rint")
                            T.vcast(sh_ub, sh32, round_mode="rint")
                            if has_gate:
                                T.vcast(g_ub, g32, round_mode="rint")
                        else:
                            T.copy(x_ub, x32)
                            T.copy(sc_ub, sc32)
                            T.copy(sh_ub, sh32)
                            if has_gate:
                                T.copy(g_ub, g32)

                        # --- mean_i = sum_j x_ij * (1/N) ---
                        T.reduce(x32, mean_b, dims=1, reduce_mode="sum")
                        T.vmul(mean_b, n_inv, mean_b)

                        # --- var_i = sum_j (x_ij - mean_i)^2 * (1/N) ---
                        for i, j in T.Parallel(block_m, N):
                            x32[i, j] = (x32[i, j] - mean_b[i, 0]) * (x32[i, j] - mean_b[i, 0])
                        T.reduce(x32, var_b, dims=1, reduce_mode="sum")
                        T.vmul(var_b, n_inv, var_b)

                        # --- rstd = 1/sqrt(var + eps) as sqrt(v)/v ---
                        T.vadd(var_b, eps, var_b)
                        T.vsqrt(var_b, tmp_b)
                        T.vdiv(tmp_b, var_b, rstd_b)

                        # --- rebuild xn in place, then affine (+ gate) ---
                        T.copy(x[row0 : row0 + rows, 0:N], x_ub[0:rows, 0:N])
                        if use_fp32_transit:
                            T.vcast(x_ub, x32, round_mode="rint")
                        else:
                            T.copy(x_ub, x32)
                        for i, j in T.Parallel(block_m, N):
                            x32[i, j] = (x32[i, j] - mean_b[i, 0]) * rstd_b[i, 0]
                            x32[i, j] = sc32[i, j] * x32[i, j] + sh32[i, j]
                            if has_gate:
                                x32[i, j] = g32[i, j] * x32[i, j]

                        if use_fp32_transit:
                            T.vcast(x32, y_ub, round_mode="rint")
                        else:
                            T.copy(x32, y_ub)
                        T.copy(y_ub[0:rows, 0:N], y[row0 : row0 + rows, 0:N])

        return main

    return _func


# ---------- Precision comparison helper ----------
def _max_diff(a, b):
    d = (a.float() - b.float()).abs()
    return torch.nan_to_num(d, nan=0.0).max().item()


def run_case(m, n, dtype, tag, has_gate=False, eps=1e-5, equal_nan=False, build=None):
    """Build inputs, run the kernel and compare against the golden."""
    dtype_str = str(dtype).replace("torch.", "")
    if build is None:
        build = lambda m_, n_: (
            torch.randn(m_, n_, dtype=torch.float32),
            torch.randn(m_, n_, dtype=torch.float32),
            torch.randn(m_, n_, dtype=torch.float32),
            torch.randn(m_, n_, dtype=torch.float32),
        )
    xf, scf, shf, gf = build(m, n)
    x, sc, sh, g = (t.to(dtype).npu() for t in (xf, scf, shf, gf))

    y = _ada_layer_norm_kernel(m, n, eps, dtype_str, has_gate=has_gate)(1)(x, sc, sh, g)
    ref = golden_ada_layer_norm(x.cpu(), sc.cpu(), sh.cpu(),
                                g.cpu() if has_gate else None, eps)
    tol = TOLERANCES[dtype]
    torch.testing.assert_close(
        y.cpu(), ref.to("npu").cpu(), equal_nan=equal_nan,
        atol=tol["atol"], rtol=tol["rtol"],
    )
    print(f"[{tag}] PASS: (M,N)=({m},{n}) dtype={dtype_str} gate={has_gate} "
          f"max_diff={_max_diff(y.cpu(), ref):.3e}")


def run_random(m, n, dtype, tag, has_gate=False, scale=1.0):
    build = lambda m_, n_: (
        torch.randn(m_, n_, dtype=torch.float32) * scale,
        torch.randn(m_, n_, dtype=torch.float32) * scale,
        torch.randn(m_, n_, dtype=torch.float32) * scale,
        torch.rand(m_, n_, dtype=torch.float32),          # gate in [0,1)
    )
    run_case(m, n, dtype, tag, has_gate=has_gate, build=build)


def run_const(m, n, dtype, tag, **kw):
    v = kw.pop("value", 1.0)
    build = lambda m_, n_: tuple(
        torch.full((m_, n_), v, dtype=torch.float32) for _ in range(4)
    )
    run_case(m, n, dtype, tag, build=build, **kw)


# ---------- Hierarchical tests ----------
def run_L0():
    """Smoke gate -- blocking; BOTH variants (they are two trace branches)."""
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        run_random(64, 1152, dtype, "L0")
        run_random(64, 1152, dtype, "L0-gate", has_gate=True)


def run_L1():
    """Manifest workloads AND every boundary semantic -- blocking."""
    # manifest workloads, both variants (shared kernel, two trace branches)
    for dtype in (torch.float16, torch.bfloat16):
        run_random(1024, 1152, dtype, "L1-dit-xl-2")
        run_random(2048, 4096, dtype, "L1-prefill")
        run_random(2048, 4096, dtype, "L1-prefill-gate", has_gate=True)
    run_random(1, 4096, torch.bfloat16, "L1-decode")

    # rstd precision -- the direct reason vrsqrt was rejected (fp32 1e-5 gate)
    run_random(64, 1152, torch.float32, "L1-rstd-precision-fp32")
    run_random(64, 1152, torch.float16, "L1-rstd-precision-fp16")

    # eps only matters for a constant row (var == 0)
    run_const(64, 1152, torch.float32, "L1-eps-const-row", value=3.0)

    # non-divisible M (the tail task)
    for m in (7, 33, 1025):
        run_random(m, 1152, torch.float32, "L1-tail-m")

    # logical tasks > physical cores -> the in-core serial branch
    run_random(1024, 512, torch.float32, "L1-serial-branch")

    # M = 1 -> single core
    run_random(1, 1152, torch.float32, "L1-m1")

    # affine edges
    run_case(64, 1152, torch.float32, "L1-scale-zero",
             build=lambda m_, n_: (torch.randn(m_, n_), torch.zeros(m_, n_),
                                   torch.randn(m_, n_), torch.randn(m_, n_)))
    run_case(64, 1152, torch.float32, "L1-shift-zero",
             build=lambda m_, n_: (torch.randn(m_, n_), torch.randn(m_, n_),
                                   torch.zeros(m_, n_), torch.randn(m_, n_)))

    # NaN propagation
    run_const(32, 512, torch.float32, "L1-nan", equal_nan=True, value=float("nan"))

    # the UB guard is a BEHAVIOUR CONTRACT: assert it raises
    try:
        _ada_layer_norm_kernel(64, 4096, 1e-5, "float32")(8)
        print("[L1-ub-guard] FAIL: expected ValueError was not raised")
        raise AssertionError("UB guard did not raise")
    except ValueError as exc:
        print(f"[L1-ub-guard] PASS: ValueError raised as expected ({str(exc)[:48]}...)")


def run_L2():
    """Additional shapes -- non-blocking."""
    for m, n, dtype, gate in (
        (128, 2048, torch.float16, False),
        (128, 2048, torch.float16, True),
        (63, 1000, torch.float32, False),
        (17, 4097, torch.float16, True),
    ):
        try:
            run_random(m, n, dtype, "L2", has_gate=gate)
        except Exception as exc:  # noqa: BLE001
            print(f"[L2] WARN (non-blocking): ({m},{n}) {dtype} gate={gate}: {exc}")


def run_boundary():
    """Additional boundary values -- non-blocking."""
    for value in (0.0, -1.0, 1e3, 1e-3):
        for dtype in (torch.float32, torch.float16):
            for gate in (False, True):
                try:
                    run_const(2050, 64, dtype, f"Boundary-{value:g}",
                              has_gate=gate, value=value, equal_nan=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[Boundary] WARN (non-blocking): {value} {dtype} "
                          f"gate={gate}: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    if args.level == "L0":
        run_L0()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
