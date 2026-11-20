# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""FusedAddRMSNormFwdOp -- TileLang-NPUIR kernel, golden and hierarchical tests.

Covers the fused add + RMSNorm operator.  The frozen design is ./DESIGN.md
(Stage 2 review passed, see ./REVIEW.md).

    s            = x + residual
    rstd         = 1 / sqrt(mean(s^2) + eps)
    y            = s * rstd * weight
    residual_out = s

This is the first operator in this batch with TWO outputs, which changes three
things: ``out_idx`` is a list (``[3, 4]``), the golden returns a tuple, and the
blocking suite must assert BOTH outputs (checking only ``y`` would miss a broken
residual passthrough).

HARD CONSTRAINTS measured on this toolchain -- each fails SILENTLY or at compile
time:

  * ``T.vrsqrt`` is NOT usable here: it is measured at ~2.7e-3 relative error,
    which EXCEEDS the fp16 tolerance (1e-3).  The reciprocal square root is
    therefore composed as ``sqrt(v)/v`` (``T.vsqrt`` + ``T.vdiv``) -- the same
    composition the in-repo `ada_layer_norm` kernel adopted for the same reason,
    and the same one the in-repo reference (`examples/norm/.../single_n.py`)
    documents.
  * ``T.vdiv`` requires IDENTICAL operand shapes, so the sqrt result goes into
    its own ``(1, 1)`` buffer rather than a row-wide one.
  * A row-wide broadcast is NOT accepted by the npuir v-ops here; the scalar
    element is indexed instead (``rstd[0, 0]``) -- the only form that works.
  * The alignment tail must be CLEARED.  ``n_align = align_up(N, 16)`` for
    fp16/bf16, and 0 is the identity for the sum, so the tail contributes
    nothing; without the clear it would pollute ``mean(s^2)``.  Note the manifest
    workloads all have N a multiple of 16, so this path is ONLY exercised by the
    deliberately unaligned cases in L1 below.
  * ``TILELANG_ASCEND_MODE`` must be exported as an ENVIRONMENT VARIABLE.  A
    plain python name does not set it; without it a reduction-bearing kernel can
    be lowered onto the Cube path and fail to compile (measured on `vector_norm`).

Run:
    python _fused_add_rms_norm_kernel.py --level L0
    python _fused_add_rms_norm_kernel.py --level all
"""

import os

# See the module docstring: this MUST be an environment variable, not a name.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

tilelang.cache.clear_cache()

# 32-byte alignment expressed in elements (same rule as the in-repo reference).
_ALIGN_ELEMS = {"float16": 16, "bfloat16": 16, "float32": 8}

TOLERANCES = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 1.6e-2, "rtol": 1.6e-2},
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
}


def _align_up(value: int, factor: int) -> int:
    return ((value + factor - 1) // factor) * factor


def _aligned_cols(n: int, dtype: str) -> int:
    return _align_up(n, _ALIGN_ELEMS[dtype])


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_fused_add_rms_norm(x, residual, weight, eps=1e-6):
    """Golden for the four equations, returning ``(y, residual_out)``.

    Computed in fp32 and cast back once, matching the kernel's opmath domain.
    It does NOT mirror the NPU sqrt/div composition or the alignment handling,
    keeping the comparison independent.
    """
    s_f32 = x.detach().cpu().to(torch.float32) + residual.detach().cpu().to(torch.float32)
    rstd = torch.rsqrt(s_f32.square().mean(dim=-1, keepdim=True) + eps)
    y = (s_f32 * rstd * weight.detach().cpu().to(torch.float32)).to(x.dtype)
    return y, s_f32.to(x.dtype)


# ---------- Kernel ----------
def _fused_add_rms_norm_kernel(M, N, eps, dtype):
    """Build the fused add + RMSNorm factory (source signature).

    Returns a JIT callable taking ``block_m`` (kept for interface compatibility
    with the GPU factory; rows are processed one per core, so it is unused).
    """
    if M < 1 or N < 1:
        raise ValueError(f"M and N must be positive, got M={M}, N={N}")
    if dtype not in _ALIGN_ELEMS:
        raise ValueError(f"unsupported dtype {dtype!r}")

    n_align = _aligned_cols(N, dtype)
    tail_cols = n_align - N
    avg_factor = 1.0 / float(N)

    @tilelang.jit(out_idx=[3, 4], target="npuir")
    def _func(block_m=1):
        # Two branches, and the split is NOT cosmetic: for a fp32 input the
        # staging `vcast(x_tile, s_f32)` would be a SAME-DTYPE cast, which
        # quantises the value to an integer (measured on `cos`; ~2.7 absolute
        # error here).  The fp32 branch therefore computes add/mul in the dtype
        # domain and never performs a same-dtype cast.
        if dtype != "float32":

            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                residual: T.Tensor((M, N), dtype),
                weight: T.Tensor((N,), dtype),
                y: T.Tensor((M, N), dtype),
                residual_out: T.Tensor((M, N), dtype),
            ):
                with T.Kernel(M, is_npu=True) as (pid_m, _):
                    x_tile = T.alloc_ub((1, n_align), dtype)
                    r_tile = T.alloc_ub((1, n_align), dtype)
                    w_tile = T.alloc_ub((1, n_align), dtype)
                    s_tile = T.alloc_ub((1, n_align), dtype)
                    out_tile = T.alloc_ub((1, n_align), dtype)
                    s_f32 = T.alloc_ub((1, n_align), "float32")
                    sq_f32 = T.alloc_ub((1, n_align), "float32")
                    w_f32 = T.alloc_ub((1, n_align), "float32")
                    rstd = T.alloc_ub((1, 1), "float32")
                    rsqrt_tmp = T.alloc_ub((1, 1), "float32")

                    if tail_cols > 0:
                        # Zero the alignment tail: it contributes 0 to the sum of
                        # squares (0 is the identity) and is never copied out.
                        T.clear(x_tile)
                        T.clear(r_tile)
                        T.clear(w_tile)

                    T.copy(x[pid_m : pid_m + 1, 0:N], x_tile[0:1, 0:N])
                    T.copy(residual[pid_m : pid_m + 1, 0:N], r_tile[0:1, 0:N])
                    T.copy(weight[0:N], w_tile[0, 0:N])

                    # --- s = x + residual (fp32), and the residual passthrough ---
                    T.vcast(x_tile, s_f32)
                    T.vcast(r_tile, sq_f32)
                    T.vadd(s_f32, sq_f32, s_f32)
                    T.vcast(s_f32, s_tile, round_mode="rint")
                    T.copy(s_tile[0:1, 0:N], residual_out[pid_m : pid_m + 1, 0:N])

                    # --- rstd = 1/sqrt(mean(s^2) + eps) ---
                    T.vmul(s_f32, s_f32, sq_f32)              # s^2
                    T.vmul(sq_f32, avg_factor, sq_f32)        # 1/N folded in pre-reduce
                    T.reduce(
                        sq_f32,
                        rstd,
                        dims=1,
                        reduce_mode="sum",
                        size=[1, n_align],
                    )
                    T.vadd(rstd, eps, rstd)
                    # 1/sqrt(v) composed as sqrt(v)/v: raw T.vrsqrt is ~2.7e-3
                    # relative error, over the fp16 tolerance.  vdiv needs equal
                    # shapes, hence the dedicated (1, 1) buffer.
                    T.vsqrt(rstd, rsqrt_tmp)
                    T.vdiv(rsqrt_tmp, rstd, rstd)

                    # --- y = s * rstd * weight (broadcast via the scalar element) ---
                    T.vmul(s_f32, rstd[0, 0], s_f32)
                    T.vcast(s_f32, out_tile, round_mode="rint")
                    T.vcast(out_tile, s_f32)
                    T.vcast(w_tile, w_f32)
                    T.vmul(s_f32, w_f32, s_f32)
                    T.vcast(s_f32, out_tile, round_mode="rint")
                    T.copy(out_tile[0:1, 0:N], y[pid_m : pid_m + 1, 0:N])

        else:
            # fp32 path: same-dtype tiles, NO vcast staging (see the note above).
            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                residual: T.Tensor((M, N), dtype),
                weight: T.Tensor((N,), dtype),
                y: T.Tensor((M, N), dtype),
                residual_out: T.Tensor((M, N), dtype),
            ):
                with T.Kernel(M, is_npu=True) as (pid_m, _):
                    x_tile = T.alloc_ub((1, n_align), dtype)
                    r_tile = T.alloc_ub((1, n_align), dtype)
                    w_tile = T.alloc_ub((1, n_align), dtype)
                    s_tile = T.alloc_ub((1, n_align), dtype)
                    out_tile = T.alloc_ub((1, n_align), dtype)
                    sq_f32 = T.alloc_ub((1, n_align), "float32")
                    tmp_f32 = T.alloc_ub((1, n_align), "float32")
                    rstd = T.alloc_ub((1, 1), "float32")
                    rsqrt_tmp = T.alloc_ub((1, 1), "float32")

                    if tail_cols > 0:
                        T.clear(x_tile)
                        T.clear(r_tile)
                        T.clear(w_tile)

                    T.copy(x[pid_m : pid_m + 1, 0:N], x_tile[0:1, 0:N])
                    T.copy(residual[pid_m : pid_m + 1, 0:N], r_tile[0:1, 0:N])
                    T.copy(weight[0:N], w_tile[0, 0:N])

                    # s = x + residual -- dtype-domain add, no same-dtype cast
                    T.vadd(x_tile, r_tile, s_tile)
                    T.copy(s_tile[0:1, 0:N], residual_out[pid_m : pid_m + 1, 0:N])

                    # rstd = 1/sqrt(mean(s^2) + eps)
                    T.vmul(s_tile, s_tile, sq_f32)
                    T.vmul(sq_f32, avg_factor, sq_f32)
                    T.reduce(
                        sq_f32,
                        rstd,
                        dims=1,
                        reduce_mode="sum",
                        size=[1, n_align],
                    )
                    T.vadd(rstd, eps, rstd)
                    T.vsqrt(rstd, rsqrt_tmp)
                    T.vdiv(rsqrt_tmp, rstd, rstd)

                    # y = s * rstd * weight
                    T.vmul(s_tile, rstd[0, 0], tmp_f32)
                    T.vmul(tmp_f32, w_tile, out_tile)
                    T.copy(out_tile[0:1, 0:N], y[pid_m : pid_m + 1, 0:N])

        return main

    return _func


# ---------- Precision comparison helper ----------
def _max_diff(a, b):
    d = (a.float() - b.float()).abs()
    d = torch.nan_to_num(d, nan=0.0)
    return d.max().item()


def run_case(x, residual, weight, tag, eps=1e-6, equal_nan=False):
    """Run the kernel and compare BOTH outputs against the golden."""
    m, n = x.shape
    dtype = x.dtype
    dtype_str = str(dtype).replace("torch.", "")

    y, residual_out = _fused_add_rms_norm_kernel(m, n, eps, dtype_str)(1)(x, residual, weight)
    ref_y, ref_r = golden_fused_add_rms_norm(x.cpu(), residual.cpu(), weight.cpu(), eps)

    tol = TOLERANCES[dtype]
    torch.testing.assert_close(
        y.cpu(), ref_y.to("npu").cpu(), equal_nan=equal_nan,
        atol=tol["atol"], rtol=tol["rtol"],
    )
    torch.testing.assert_close(
        residual_out.cpu(), ref_r.to("npu").cpu(), equal_nan=equal_nan,
        atol=tol["atol"], rtol=tol["rtol"],
    )
    # residual_out is a passthrough of s; assert it independently of the golden
    # so a coincidental match in y cannot mask a broken passthrough.
    s_ref = (x.cpu().float() + residual.cpu().float()).to(dtype)
    torch.testing.assert_close(
        residual_out.cpu(), s_ref.to("npu").cpu(), equal_nan=equal_nan,
        atol=tol["atol"], rtol=tol["rtol"],
    )
    print(
        f"[{tag}] PASS: (M,N)=({m},{n}) dtype={dtype_str} eps={eps:g} "
        f"max_diff_y={_max_diff(y.cpu(), ref_y):.3e} "
        f"max_diff_res={_max_diff(residual_out.cpu(), ref_r):.3e}"
    )


def run_random(m, n, dtype, tag, eps=1e-6, scale=1.0):
    x = (torch.randn(m, n, dtype=torch.float32) * scale).to(dtype).npu()
    residual = (torch.randn(m, n, dtype=torch.float32) * scale).to(dtype).npu()
    weight = torch.randn(n, dtype=torch.float32).to(dtype).npu()
    run_case(x, residual, weight, tag, eps=eps)


# ---------- Hierarchical tests ----------
def run_L0():
    """Smoke gate -- blocking."""
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        run_random(64, 512, dtype, "L0")


def run_L1():
    """Manifest workloads AND every boundary semantic section 0.1 declares must
    be preserved -- blocking."""
    # the four ACTIVE manifest workloads (max N is 8192, not the 16384 the raw
    # manifest lists, so this is the real worst case for the UB budget)
    for dtype in (torch.float16, torch.bfloat16):
        run_random(2048, 4096, dtype, "L1-8b-prefill")
        run_random(2048, 8192, dtype, "L1-70b-prefill")   # 176 KB declared union
    run_random(1, 4096, torch.bfloat16, "L1-8b-decode-M1")

    # N NOT a multiple of the 16-element alignment -> the tail-clear path, which
    # NO manifest workload can reach (all of theirs are multiples of 16).
    for n in (4095, 8191, 100):
        run_random(64, n, torch.float16, "L1-unaligned-n")

    # tiny reductions
    for n in (1, 2, 16):
        run_random(64, n, torch.float32, "L1-tiny-n")

    # eps only matters when s == 0; force it.
    zeros = torch.zeros((32, 512), dtype=torch.float32)
    run_case(zeros.npu(), zeros.npu(), torch.ones(512).npu(), "L1-zero-row-eps", equal_nan=True)

    # weight edge cases: weighting by 0 should zero y while residual_out stays s
    x = torch.randn(32, 512, dtype=torch.float32).npu()
    r = torch.randn(32, 512, dtype=torch.float32).npu()
    run_case(x, r, torch.zeros(512).npu(), "L1-weight-zero", equal_nan=True)
    run_case(x, r, torch.ones(512).npu(), "L1-weight-one", equal_nan=True)

    # NaN propagation through add / square / reduce
    nan_x = torch.full((16, 512), float("nan"), dtype=torch.float32).npu()
    run_case(nan_x, torch.zeros(16, 512).npu(), torch.ones(512).npu(),
             "L1-nan", equal_nan=True)

    # rstd precision: the direct reason vrsqrt was rejected.  A row of large
    # magnitudes makes the fp16 tolerance bite.
    big = torch.full((16, 512), 100.0, dtype=torch.float16).npu()
    run_case(big, torch.zeros(16, 512, dtype=torch.float16).npu(),
             torch.ones(512, dtype=torch.float16).npu(), "L1-rstd-precision", equal_nan=True)


def run_L2():
    """Additional shapes -- non-blocking (warn on failure)."""
    for m, n, dtype in (
        (128, 2048, torch.float16),
        (512, 1024, torch.bfloat16),
        (7, 1000, torch.float32),
        (33, 4097, torch.float16),
    ):
        try:
            run_random(m, n, dtype, "L2")
        except Exception as exc:  # noqa: BLE001  (L2 is advisory)
            print(f"[L2] WARN (non-blocking): ({m},{n}) {dtype}: {exc}")


def run_boundary():
    """Additional boundary values -- non-blocking (warn on failure)."""
    cases = {
        "neg-all": lambda m, n, d: -torch.ones((m, n), dtype=d),
        "huge-static": lambda m, n, d: torch.full((m, n), 1e3, dtype=d),
        "tiny-static": lambda m, n, d: torch.full((m, n), 1e-3, dtype=d),
    }
    for name, gen in cases.items():
        for dtype in (torch.float32, torch.float16):
            try:
                x = gen(2050, 64, dtype).npu()
                run_case(x, gen(2050, 64, dtype).npu(),
                         torch.ones(64, dtype=dtype).npu(),
                         f"Boundary-{name}", equal_nan=True)
            except Exception as exc:  # noqa: BLE001  (Boundary is advisory)
                print(f"[Boundary] WARN (non-blocking): {name} {dtype}: {exc}")


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
