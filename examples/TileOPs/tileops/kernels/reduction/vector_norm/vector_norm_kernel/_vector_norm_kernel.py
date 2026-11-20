# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""VectorNormKernel -- TileLang-NPUIR kernel, golden and hierarchical tests.

Covers three manifest ops that share ONE kernel class (resolution pattern B):
L1NormFwdOp (ord=1), L2NormFwdOp (ord=2), InfNormFwdOp (ord=inf).
The frozen design is ./DESIGN.md (Stage 2 review passed, see ./REVIEW.md).

This is the first REDUCTION-family operator in this batch (the six earlier ones
were all elementwise), and that changes three things:

  1. There IS a reduction axis (dim=-1), so the reduction ORDER matters for
     floating point.  The design therefore keeps the whole reduction axis inside
     a single core -- splitting it across cores would need a second, inter-core
     reduction and change the summation order, diverging from the source
     operator.  That is a SEMANTIC decision, not a tuning choice.
  2. The binding constraint is UB CAPACITY, not bandwidth: the capacity formula is
         block_m * N * (2*elem_bytes + 4)
     (two dtype tiles plus the fp32 shadow).  For the widest workload, (64, 8192)
     in bf16, block_m=1 already costs 64 KB, so block_m is pinned to 1.
  3. `T.reduce_*` operates on FRAGMENTS, not UB, so the data path is
     GM -> UB -> fragment -> reduce -> UB -> GM.

NaN semantics differ per branch and must be verified separately:
  * l1 / l2 use reduce_sum  -> NaN propagates naturally
  * inf   uses reduce_max   -> NaN does NOT propagate; the Op layer
    (InfNormFwdOp._post_kernel) patches NaN rows, exactly as on GPU.
This suite can only reach the kernel, so the inf-NaN behaviour is RECORDED here
(not asserted) and the Op-layer patch is noted for the integration stage.

Run:
    python vector_norm_kernel.py --level L0    # blocking gate
    python vector_norm_kernel.py --level all   # L0 + L1 (blocking) + L2/Boundary (warn)
"""

import os

# Developer mode MUST be exported as an ENVIRONMENT VARIABLE here, not merely
# held in a python name.  Without it this reduction kernel is lowered onto the
# Cube path: T.alloc_shared lands in `cbuf` (= L1) and the staging copy becomes
#     error: 'hivm.hir.copy' op Unsupported copy from cbuf to zero!
# Verified on this toolchain: with TILELANG_ASCEND_MODE=Developer the identical
# kernel compiles and runs; without it bishengir-compile exits non-zero.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import torch
import tilelang
import tilelang.language as T
import torch_npu  # noqa: F401  (registers the "npu" device)

tilelang.cache.clear_cache()

_ORD = {"l1": 1, "l2": 2, "inf": float("inf")}

# Tolerances lifted verbatim from the GPU source suite.
TOLERANCES = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 1.6e-2, "rtol": 1.6e-2},
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
}

# UB budget used to pick block_m: 192 KB capacity, x1.7 auto-multi-buffer
# inflation -> ~113 KB effective (CONST-capacity-910B2C).
_UB_EFFECTIVE_KB = 113


def ub_bytes_for(block_m, n, elem_bytes):
    """Capacity formula from the wrapper docstring:
    block_m * N * (2*elem_bytes + 4) -- two dtype tiles + one fp32 shadow."""
    return block_m * n * (2 * elem_bytes + 4)


def pick_block_m(m, n, elem_bytes, candidates=(1, 2, 3, 4, 8, 16)):
    """Largest block_m from `candidates` that fits the UB budget (never exceeds m)."""
    best = 1
    for bm in candidates:
        if bm > m:
            break
        if ub_bytes_for(bm, n, elem_bytes) <= _UB_EFFECTIVE_KB * 1024:
            best = bm
    return best


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_vector_norm(x: torch.Tensor, op_kind: str) -> torch.Tensor:
    """Golden for the three norms, derived from the operator semantics only.

    Reduces in fp32 then casts back -- the same opmath domain the kernel uses.
    It does NOT mirror the NPU blocking/fragment structure, keeping the
    comparison independent.
    """
    x_cpu = x.detach().cpu()
    return torch.linalg.vector_norm(
        x_cpu.to(torch.float32), ord=_ORD[op_kind], dim=-1
    ).to(x_cpu.dtype)


# ---------- Kernel (single-tile: the whole row fits in UB) ----------
def _vector_norm_kernel(M, N, op_kind, dtype):
    """Build the single-tile vector-norm factory.

    NOTE the decorator placement: this factory returns ANOTHER function
    (``_func(block_m)``), so ``@tilelang.jit`` belongs on the INNER function.
    The elementwise family's factories return a prim_func directly and therefore
    carry the decorator on the OUTER function -- copying that shape here raises
    ``AttributeError: 'function' object has no attribute 'script'``.
    """

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):
                x_ub = T.alloc_shared((block_m, N), dtype)           # GM staging
                x_local = T.alloc_fragment((block_m, N), dtype)
                x_f32 = T.alloc_fragment((block_m, N), "float32")    # accumulation domain
                row_acc = T.alloc_fragment((block_m, 1), "float32")
                out_ub = T.alloc_shared((block_m,), dtype)

                real_m = T.min(block_m, M - pid_m * block_m)

                # Step 1: GM -> UB -> fragment; the tail block slices both sides
                # together so no out-of-bounds read and the shapes stay equal.
                T.copy(
                    x[pid_m * block_m : pid_m * block_m + real_m, 0:N],
                    x_ub[0:real_m, 0:N],
                )
                T.copy(x_ub, x_local)

                # Step 2: widen to fp32 for the accumulation
                for i, j in T.Parallel(block_m, N):
                    x_f32[i, j] = T.cast(x_local[i, j], "float32")

                # Step 3: per-row transform + reduction.  `op_kind` is a
                # Python-level value resolved at trace time, so only one branch
                # is ever emitted.
                if op_kind == "l1":
                    T.vabs(x_f32, x_f32)
                    T.reduce_sum(x_f32, row_acc, dim=1)
                elif op_kind == "l2":
                    T.vmul(x_f32, x_f32, x_f32)
                    T.reduce_sum(x_f32, row_acc, dim=1)
                    T.vsqrt(row_acc, row_acc)
                else:
                    # inf: max(|x|).  NOTE: reduce_max does not propagate NaN;
                    # NaN rows are patched at the Op layer
                    # (InfNormFwdOp._post_kernel), as in the GPU source.
                    T.vabs(x_f32, x_f32)
                    T.reduce_max(x_f32, row_acc, dim=1)

                # Step 4: cast back, (i, 0) -> (i,)
                for i in T.Parallel(block_m):
                    out_ub[i] = T.cast(row_acc[i, 0], dtype)

                # Step 5: UB -> GM
                T.copy(
                    out_ub[0:real_m],
                    y[pid_m * block_m : pid_m * block_m + real_m],
                )

        return main

    return _func


# ---------- Precision comparison helper ----------
def _max_diff(y, ref):
    d = (y.float() - ref.float()).abs()
    d = torch.nan_to_num(d, nan=0.0)
    return d.max().item()


def run_case(x, op_kind, tag, equal_nan=False):
    """Run the kernel on an NPU tensor and compare against the golden."""
    assert x.dim() == 2, "kernel-level cases are (M, N)"
    m, n = x.shape
    dtype = x.dtype
    dtype_str = str(dtype).replace("torch.", "")
    block_m = pick_block_m(m, n, torch.tensor([], dtype=dtype).element_size())
    y = _vector_norm_kernel(m, n, op_kind, dtype_str)(block_m)(x)
    ref = golden_vector_norm(x.cpu(), op_kind).to("npu")

    tol = TOLERANCES[dtype]
    torch.testing.assert_close(
        y.cpu(), ref.cpu(), equal_nan=equal_nan, atol=tol["atol"], rtol=tol["rtol"]
    )
    print(
        f"[{tag}] PASS: (M,N)=({m},{n}) dtype={dtype_str} block_m={block_m} "
        f"max_diff={_max_diff(y.cpu(), ref.cpu()):.3e}"
    )


def run_random(m, n, dtype, op_kind, tag, scale=1.0):
    x = (torch.randn(m, n, dtype=torch.float32) * scale).to(dtype).npu()
    run_case(x, op_kind, tag)


# ---------- Hierarchical tests ----------
def run_L0():
    """Smoke gate -- blocking."""
    for op_kind in ("l1", "l2", "inf"):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            run_random(32, 512, dtype, op_kind, "L0")


def run_L1():
    """Full-scale workloads AND every boundary semantic that section 0.1
    declares must be preserved -- blocking."""
    # manifest workloads, all three kinds (one kernel, three semantics)
    for op_kind in ("l1", "l2", "inf"):
        run_random(2048, 4096, torch.float16, op_kind, "L1-hidden-state")
        run_random(2048, 4096, torch.bfloat16, op_kind, "L1-hidden-state")
        # the widest row: 64 KB for block_m=1, the capacity-tight case
        run_random(64, 8192, torch.bfloat16, op_kind, "L1-long-row")

    # non-divisible M -- the tail-block path (block_m > 1 makes M%block_m != 0)
    for m in (7, 33, 1025):
        run_random(m, 512, torch.float32, "l2", "L1-tail-m")

    # tiny reductions
    run_random(64, 1, torch.float32, "l2", "L1-n1")     # N = 1
    run_random(64, 2, torch.float32, "l1", "L1-n2")     # N = 2
    run_random(1, 512, torch.float32, "inf", "L1-m1")   # M = 1

    # l1 / l2 DO propagate NaN (they go through reduce_sum)
    nan_x = torch.full((16, 64), float("nan"), dtype=torch.float32).npu()
    run_case(nan_x, "l1", "L1-nan-l1", equal_nan=True)
    run_case(nan_x, "l2", "L1-nan-l2", equal_nan=True)

    # +/-inf rows: l1 -> +inf, l2 -> +inf, inf -> +inf
    inf_x = torch.full((16, 64), float("inf"), dtype=torch.float32).npu()
    for op_kind in ("l1", "l2", "inf"):
        run_case(inf_x, op_kind, f"L1-inf-{op_kind}", equal_nan=True)

    # l2 sqrt domain: an all-zero row (sqrt(0)) and a row whose squares overflow
    zero_row = torch.zeros((16, 64), dtype=torch.float32)
    zero_row[1] = 0.0
    run_case(zero_row.npu(), "l2", "L1-l2-zero-row", equal_nan=True)

    big = torch.full((16, 64), 1e20, dtype=torch.float32)
    run_case(big.npu(), "l2", "L1-l2-overflow", equal_nan=True)
    # note: 64 * (1e20)^2 = 6.4e41 > fp32 max, so the fp32 reference also
    # overflows to +inf; the comparison is therefore inf == inf.

    # 3-D input reduced over dim=[0, 2] -> (4, 128, 512) becomes (512, 128) after
    # the host-side reshape; assert the reshaped semantic here.
    x3 = torch.randn(4, 128, 512, dtype=torch.float32)
    xr = x3.permute(1, 0, 2).reshape(128, -1).npu()          # (M=128, N=4*512)
    ref = torch.linalg.vector_norm(
        x3.permute(1, 0, 2).reshape(128, -1).to(torch.float32), ord=2, dim=-1
    )
    y = _vector_norm_kernel(128, 2048, "l2", "float32")(1)(xr)
    torch.testing.assert_close(
        y.cpu(), ref, atol=TOLERANCES[torch.float32]["atol"],
        rtol=TOLERANCES[torch.float32]["rtol"],
    )
    print("[L1-3d-dim02] PASS: (4,128,512) permuted to (128, 2048), l2")


def record_inf_nan():
    """Record -- do NOT assert -- the inf-branch NaN behaviour.

    l1/l2 propagate NaN through reduce_sum; inf uses reduce_max, which does NOT
    propagate it.  The Op layer (InfNormFwdOp._post_kernel) patches NaN rows,
    exactly as in the GPU source.  This suite only reaches the kernel, so the
    kernel-level behaviour is recorded here rather than asserted.
    """
    print("[L2] inf-branch NaN record (kernel level; NOT asserted)")
    nan_x = torch.full((16, 64), float("nan"), dtype=torch.float32).npu()
    y = _vector_norm_kernel(16, 64, "inf", "float32")(1)(nan_x).cpu()
    got_nan = bool(torch.isnan(y).any().item())
    print(f"[L2]   reduce_max on an all-NaN row -> any NaN in output: {got_nan} "
          f"(expected False; the Op-layer _post_kernel patch supplies NaN)")


def run_L2():
    """Additional shapes AND the inf-NaN record -- non-blocking."""
    for m, n, dtype in (
        (512, 2048, torch.float16),
        (1024, 1024, torch.bfloat16),
        (63, 1000, torch.float32),
        (17, 4097, torch.float16),
    ):
        for op_kind in ("l1", "l2", "inf"):
            try:
                run_random(m, n, dtype, op_kind, "L2")
            except Exception as exc:  # noqa: BLE001  (L2 is advisory)
                print(f"[L2] WARN (non-blocking): ({m},{n}) {dtype} {op_kind}: {exc}")

    record_inf_nan()


def run_boundary():
    """Additional boundary values -- non-blocking (warn on failure)."""
    cases = {
        "all-zero": lambda m, n, d: torch.zeros((m, n), dtype=d),
        "one-hot": lambda m, n, d: torch.eye(max(m, n), dtype=d)[:m, :n],
        "huge-static": lambda m, n, d: torch.full((m, n), 1e3, dtype=d),
    }
    for name, gen in cases.items():
        for dtype in (torch.float32, torch.float16):
            for op_kind in ("l1", "l2", "inf"):
                try:
                    run_case(gen(2050, 64, dtype).npu(), op_kind,
                             f"Boundary-{name}", equal_nan=True)
                except Exception as exc:  # noqa: BLE001  (Boundary is advisory)
                    print(f"[Boundary] WARN (non-blocking): {name} {dtype} "
                          f"{op_kind}: {exc}")


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
