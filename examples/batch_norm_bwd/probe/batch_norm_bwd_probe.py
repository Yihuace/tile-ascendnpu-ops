# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Feasibility probe: BatchNorm backward, ``target="npuir"`` (Ascend910B3, CANN 8.5.2).

Stage-0 pre-migration FEASIBILITY PROBE (harness). This is NOT the migrated
operator; it is a minimal but *representative* kernel that exercises the hard
capabilities a migration of ``BatchNormBwdOp`` depends on:

  * per-channel reduction over the flattened ``(N, S)`` axis of ``grad_out``
    and ``grad_out * x_hat`` (two accumulators), where ``x_hat`` derives from
    ``x, mean, rstd`` (the channel axis is the *middle* axis of the native
    ``[N, C, S]`` layout, so the reduction is a strided per-channel plane
    gather, not a trailing-axis row reduction);
  * the ``grad_x`` elementwise epilogue;
  * per-channel scalar outputs ``grad_weight`` (dgamma) and ``grad_bias``
    (dbeta).

GPU source semantics (``TileOPs-main/src/tileops/kernels/norm/batch_norm.py``,
``_batch_norm_bwd_kernel`` L1487; independent of the NPU algorithm):

    L              = N * S
    x_hat[c,i]     = (x[c,i] - mean[c]) * rstd[c]
    grad_bias[c]   = sum_i grad_out[c,i]
    grad_weight[c] = sum_i grad_out[c,i] * x_hat[c,i]
    grad_x[c,i]    = weight[c] * rstd[c] / L
                     * ( L * grad_out[c,i] - grad_bias[c] - x_hat[c,i] * grad_weight[c] )

Structure (reuses the fused_add_layer_norm NPU pattern, Developer mode):
  * ``T.Kernel(C, is_npu=True)`` — one core per channel; the whole
    ``(N, S)`` channel plane is held in UB (fits the manifest workloads), so
    the per-channel reduction is entirely *within one core*: NO cross-core
    merge, NO atomics.
  * per-channel plane gather via a native-layout slice
    ``grad_out[0:N, c, 0:S]`` -> ``(N, S)`` UB buffer;
  * two accumulators reduced with a two-step ``T.reduce`` (dims=[1] then
    dims=[0]); per-channel scalars broadcast with ``T.vbrc``;
  * fp16/bf16 go through an explicit fp32 transit chain (``T.vcast`` rint),
    fp32 is computed directly (bf16 has no Vector arithmetic).

Run: python batch_norm_bwd_probe.py
Prints per-case ``max_abs_err`` against a pure-torch CPU golden and a final
PROBE PASS/FAIL.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import traceback

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

__all__ = ["batch_norm_bwd_probe", "golden_batch_norm_bwd"]

_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")
_TOLERANCE = {
    "float16": 1e-3,
    "bfloat16": 1.6e-2,
    "float32": 1e-5,
}


# ---------------------------------------------------------------------------
# Independent pure-torch CPU golden (does NOT mirror the NPU algorithm)
# ---------------------------------------------------------------------------
def golden_batch_norm_bwd(grad_out, x, weight, mean, rstd):
    """Reference BatchNorm backward (fp32 opmath on CPU).

    Inputs ``grad_out, x``: ``(N, C, S)``; ``weight, mean, rstd``: ``(C,)``.
    Returns ``(grad_x, grad_weight, grad_bias)`` with ``grad_x`` rounded to
    the input dtype (single round-back), the channel grads in fp32.
    """
    N, C, S = grad_out.shape
    L = N * S
    go = grad_out.float()
    xf = x.float()
    w = weight.view(1, C, 1)
    m = mean.view(1, C, 1)
    r = rstd.view(1, C, 1)
    x_hat = (xf - m) * r
    grad_bias = go.sum(dim=(0, 2))                      # (C,)
    grad_weight = (go * x_hat).sum(dim=(0, 2))          # (C,)
    gb = grad_bias.view(1, C, 1)
    gw = grad_weight.view(1, C, 1)
    grad_x = (w * r / L) * (L * go - gb - x_hat * gw)   # (N, C, S)
    return grad_x.to(grad_out.dtype), grad_weight, grad_bias


# ---------------------------------------------------------------------------
# Probe kernel
# ---------------------------------------------------------------------------
def batch_norm_bwd_probe(N, C, S, dtype="float16"):
    """Build the probe kernel factory for a fixed ``(N, C, S, dtype)``.

    Returns a JIT callable ``fn()`` compiling
    ``main(grad_out, x, weight, mean, rstd)`` -> ``(grad_x, grad_weight, grad_bias)``
    (``out_idx=[5, 6, 7]``).
    """
    if N < 1 or C < 1 or S < 1:
        raise ValueError(f"N, C, S must be positive, got {N}, {C}, {S}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")

    L = N * S
    accum = "float32"
    L_inv = 1.0 / float(L)
    use_transit = dtype in ("float16", "bfloat16")

    @tilelang.jit(out_idx=[5, 6, 7], target="npuir")
    def _func():
        if use_transit:

            @T.prim_func
            def main(
                grad_out: T.Tensor((N, C, S), dtype),
                x: T.Tensor((N, C, S), dtype),
                weight: T.Tensor((C,), accum),
                mean: T.Tensor((C,), accum),
                rstd: T.Tensor((C,), accum),
                grad_x: T.Tensor((N, C, S), dtype),      # out_idx=5
                grad_weight: T.Tensor((C,), accum),      # out_idx=6
                grad_bias: T.Tensor((C,), accum),        # out_idx=7
            ):
                with T.Kernel(C, is_npu=True) as (cid, _):
                    stage = T.alloc_shared((N, S), dtype)     # fp16/bf16 transit
                    g = T.alloc_shared((N, S), accum)         # grad_out plane
                    xh = T.alloc_shared((N, S), accum)        # x_hat plane
                    prod = T.alloc_shared((N, S), accum)      # scratch
                    brc = T.alloc_shared((N, S), accum)       # broadcast scratch
                    red = T.alloc_shared((N, 1), accum)       # per-n partial
                    gb = T.alloc_shared((1, 1), accum)
                    gw = T.alloc_shared((1, 1), accum)
                    w1 = T.alloc_shared((1, 1), accum)
                    m1 = T.alloc_shared((1, 1), accum)
                    r1 = T.alloc_shared((1, 1), accum)
                    coef = T.alloc_shared((1, 1), accum)

                    # ---- per-channel parameters ----
                    T.copy(weight[cid : cid + 1], w1)
                    T.copy(mean[cid : cid + 1], m1)
                    T.copy(rstd[cid : cid + 1], r1)

                    # ---- native-layout per-channel plane gather + fp32 transit ----
                    T.copy(grad_out[0:N, cid, 0:S], stage)
                    T.vcast(stage, g, round_mode="rint")
                    T.copy(x[0:N, cid, 0:S], stage)
                    T.vcast(stage, xh, round_mode="rint")

                    # ---- x_hat = (x - mean) * rstd ----
                    T.vbrc(m1, brc)
                    T.vsub(xh, brc, xh)
                    T.vbrc(r1, brc)
                    T.vmul(xh, brc, xh)

                    # ---- grad_bias = sum(grad_out); grad_weight = sum(grad_out*x_hat) ----
                    T.reduce(g, red, dims=[1], reduce_mode="sum", clear=True)
                    T.reduce(red, gb, dims=[0], reduce_mode="sum", clear=True)
                    T.vmul(g, xh, prod)
                    T.reduce(prod, red, dims=[1], reduce_mode="sum", clear=True)
                    T.reduce(red, gw, dims=[0], reduce_mode="sum", clear=True)
                    T.copy(gb, grad_bias[cid : cid + 1])
                    T.copy(gw, grad_weight[cid : cid + 1])

                    # ---- grad_x = (w*rstd/L) * (L*g - gb - x_hat*gw) ----
                    T.vmul(w1, r1, coef)          # w * rstd
                    T.vmul(coef, L_inv, coef)     # / L
                    T.vmul(g, float(L), prod)     # L * grad_out
                    T.vbrc(gb, brc)
                    T.vsub(prod, brc, prod)       # - grad_bias
                    T.vbrc(gw, brc)
                    T.vmul(xh, brc, brc)          # x_hat * grad_weight
                    T.vsub(prod, brc, prod)       # - x_hat*grad_weight
                    T.vbrc(coef, brc)
                    T.vmul(prod, brc, prod)       # * coef

                    T.vcast(prod, stage, round_mode="rint")
                    T.copy(stage, grad_x[0:N, cid, 0:S])

        else:

            @T.prim_func
            def main(
                grad_out: T.Tensor((N, C, S), dtype),
                x: T.Tensor((N, C, S), dtype),
                weight: T.Tensor((C,), accum),
                mean: T.Tensor((C,), accum),
                rstd: T.Tensor((C,), accum),
                grad_x: T.Tensor((N, C, S), dtype),
                grad_weight: T.Tensor((C,), accum),
                grad_bias: T.Tensor((C,), accum),
            ):
                with T.Kernel(C, is_npu=True) as (cid, _):
                    g = T.alloc_shared((N, S), accum)
                    xh = T.alloc_shared((N, S), accum)
                    prod = T.alloc_shared((N, S), accum)
                    brc = T.alloc_shared((N, S), accum)
                    red = T.alloc_shared((N, 1), accum)
                    gb = T.alloc_shared((1, 1), accum)
                    gw = T.alloc_shared((1, 1), accum)
                    w1 = T.alloc_shared((1, 1), accum)
                    m1 = T.alloc_shared((1, 1), accum)
                    r1 = T.alloc_shared((1, 1), accum)
                    coef = T.alloc_shared((1, 1), accum)

                    T.copy(weight[cid : cid + 1], w1)
                    T.copy(mean[cid : cid + 1], m1)
                    T.copy(rstd[cid : cid + 1], r1)

                    T.copy(grad_out[0:N, cid, 0:S], g)
                    T.copy(x[0:N, cid, 0:S], xh)

                    T.vbrc(m1, brc)
                    T.vsub(xh, brc, xh)
                    T.vbrc(r1, brc)
                    T.vmul(xh, brc, xh)

                    T.reduce(g, red, dims=[1], reduce_mode="sum", clear=True)
                    T.reduce(red, gb, dims=[0], reduce_mode="sum", clear=True)
                    T.vmul(g, xh, prod)
                    T.reduce(prod, red, dims=[1], reduce_mode="sum", clear=True)
                    T.reduce(red, gw, dims=[0], reduce_mode="sum", clear=True)
                    T.copy(gb, grad_bias[cid : cid + 1])
                    T.copy(gw, grad_weight[cid : cid + 1])

                    T.vmul(w1, r1, coef)
                    T.vmul(coef, L_inv, coef)
                    T.vmul(g, float(L), prod)
                    T.vbrc(gb, brc)
                    T.vsub(prod, brc, prod)
                    T.vbrc(gw, brc)
                    T.vmul(xh, brc, brc)
                    T.vsub(prod, brc, prod)
                    T.vbrc(coef, brc)
                    T.vmul(prod, brc, prod)

                    T.copy(prod, grad_x[0:N, cid, 0:S])

        return main

    return _func


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_case(N, C, S, dtype_str):
    torch_dtype = getattr(torch, dtype_str)
    torch.manual_seed(0)
    grad_out = (torch.randn(N, C, S) * 0.5).to(torch_dtype)
    x = torch.randn(N, C, S).to(torch_dtype)
    weight = torch.randn(C)
    mean = torch.randn(C)
    rstd = torch.rand(C).abs() + 0.5

    kernel = batch_norm_bwd_probe(N, C, S, dtype_str)()
    gx, gw, gb = kernel(
        grad_out.npu(), x.npu(), weight.npu(), mean.npu(), rstd.npu()
    )
    gx, gw, gb = gx.cpu(), gw.cpu(), gb.cpu()

    ref_gx, ref_gw, ref_gb = golden_batch_norm_bwd(grad_out, x, weight, mean, rstd)

    err_gx = (gx.float() - ref_gx.float()).abs().max().item()
    err_gw = (gw - ref_gw).abs().max().item()
    err_gb = (gb - ref_gb).abs().max().item()
    tol = _TOLERANCE[dtype_str]

    print(
        f"[PROBE] (N={N},C={C},S={S}) dtype={dtype_str} "
        f"max_abs_err grad_x={err_gx:.3e} grad_weight={err_gw:.3e} "
        f"grad_bias={err_gb:.3e} (rtol=atol={tol:.1e})"
    )
    # Proper elementwise criterion on all three outputs.
    ok = True
    for name, out, ref in (("grad_x", gx, ref_gx),
                           ("grad_weight", gw, ref_gw),
                           ("grad_bias", gb, ref_gb)):
        try:
            torch.testing.assert_close(
                out.float(), ref.float(), rtol=tol, atol=tol,
                msg=lambda m, name=name: f"{name}: {m}",
            )
        except AssertionError as exc:
            ok = False
            print(f"  {name} FAILED: {str(exc).splitlines()[0]}")
    return ok, max(err_gx, err_gw, err_gb)


CASES = [
    (8, 64, 1024, "float16"),   # manifest resnet-stage1 (B=8,C=64,S=32x32)
    (4, 64, 256, "float16"),
    (8, 64, 256, "bfloat16"),
    (4, 64, 256, "float32"),
]


def main():
    print("device:", torch.npu.get_device_name(0))
    all_ok = True
    worst = 0.0
    failures = []
    for N, C, S, dtype_str in CASES:
        print(f"\n===== case (N={N},C={C},S={S}) {dtype_str} =====", flush=True)
        try:
            ok, err = _run_case(N, C, S, dtype_str)
            worst = max(worst, err)
            all_ok = all_ok and ok
            if not ok:
                failures.append((N, C, S, dtype_str, err))
        except Exception as e:  # noqa: BLE001
            all_ok = False
            failures.append((N, C, S, dtype_str, repr(e)))
            print(f"[PROBE] EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"\nworst max_abs_err across cases: {worst:.3e}")
    if all_ok:
        print("PROBE PASS")
    else:
        print(f"PROBE FAIL: {failures}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
