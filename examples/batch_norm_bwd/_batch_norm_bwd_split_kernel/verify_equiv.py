"""verify_equiv.py -- _batch_norm_bwd_split_kernel §1.6.1 adopted-item equivalence.

Self-contained, torch CPU + fp64 reference (no NPU / no tilelang dependency).
Stage 2 (design review, dimension 8) re-runs this file.

Op semantics (§0.1 of DESIGN.md, from the GPU source kernel
``tileops/kernels/norm/batch_norm.py:_batch_norm_bwd_split_kernel`` L1795 and the
repo reference ``workloads/norm.py`` function ``batch_norm_backward``).  Per
channel ``c`` with ``L = N*S`` and ``x_hat = (x - mean) * rstd``::

    grad_bias[c]   = sum_l grad_out[c, l]
    grad_weight[c] = sum_l grad_out[c, l] * x_hat[c, l]
    grad_x[c, l]   = weight[c] * rstd[c] / L
                     * ( L * grad_out[c, l] - grad_bias[c] - x_hat[c, l] * grad_weight[c] )

Baseline (the docstring / literal math, computed in fp64 then rounded to the input
dtype per the fp32-opmath golden contract) for one channel:

    xh    = (x - mean) * rstd
    GB    = sum(g)
    GW    = sum(g * xh)
    coef  = w * rstd / L
    gx    = coef * (L * g - GB - xh * GW)

Adopted items (§1.6.1); candidate = the NPU split design (its ``finalize`` phase
folds the coefficients and its ``apply`` phase distributes the epilogue):

  O1 division -> reciprocal multiply (L is a compile-time constant, fl(1/L))
        per = (w * rstd) * fl(1/L)        (was  coef = w * rstd / L)
     Identity up to the fp64 rounding of the constant fl(1/L); turns the one
     per-channel vdiv into a vmul.

  O2 distributed epilogue (fold L and coef into per-channel scalars)
        scale = w * rstd
        gx    = scale * g - per * GB - per * GW * xh
     (was  gx = coef * (L*g - GB - xh*GW))
     Exact algebra after distributing coef:
        coef * L * g = (w*rstd) * g = scale * g
        coef * GB    = per * GB
        coef * xh * GW = per * GW * xh
     Removes one per-element multiply.

Discrimination section (pins the semantic choice the design rejects):

  A. Rejected ``grad_weight`` via the raw-moment identity
        GW = rstd * ( sum(g * x) - mean * sum(g) )   (was  sum(g * x_hat))
     Algebraically exact but subtracts two large fp32 sums; for large-mean /
     small-variance channels it suffers catastrophic cancellation and must
     differ from the centered form (violations > 0).  The centered ``x_hat``
     product is kept.

Method: random tensors (randn) + corner values (large-mean/small-variance,
constant, zero, subnormal, +-inf, NaN, extreme weight, degenerate shapes) at each
dtype; baseline and candidate computed in fp64 and rounded to the target dtype;
ulp / violation comparison against the repo tolerances (fp32 1e-5, fp16 1e-2,
bf16 2e-2 -- ``workloads/norm.py`` function ``batch_norm_bwd_tolerance``).  Any
adopted item that fails must not be adopted.
"""

import math

import torch

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Repo tolerances (examples/TileOPs/tileops/workloads/norm.py batch_norm_bwd_tolerance).
_TOL = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (2e-2, 2e-2),
}

# (N, S) shapes: the manifest workloads (smoke-fc, resnet-stage1/2/3,
# large-spatial tile) plus degenerate / small shapes.
_SHAPES = ((32, 1), (8, 1024), (4, 1024), (4, 784), (4, 2048), (1, 1), (2, 3))


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _round_to(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Round an fp64 tensor to *dtype*, returned again as fp64 (exact value)."""
    return x.to(dtype).to(torch.float64)


def _ulp_diff(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> float:
    """Max difference between two dtype-representable fp64 tensors, in dtype ulps."""
    finfo = torch.finfo(dtype)
    if bool((torch.isfinite(a) != torch.isfinite(b)).any()):
        return float("inf")
    mask = torch.isfinite(a) & torch.isfinite(b)
    if not bool(mask.any()):
        return 0.0
    aa = a[mask]
    mag = torch.clamp(aa.abs(), min=finfo.tiny)
    exponent = torch.floor(torch.log2(mag))
    spacing = float(finfo.eps) * torch.pow(torch.tensor(2.0, dtype=torch.float64), exponent)
    return float(((aa - b[mask]).abs() / spacing).max().item())


def _violation_rate(y_ref: torch.Tensor, y_cand: torch.Tensor, dtype: torch.dtype) -> float:
    atol, rtol = _TOL[dtype]
    bad = (y_cand - y_ref).abs() > (atol + rtol * y_ref.abs())
    return float(bad.to(torch.float64).mean().item())


def _bn_bwd_channel_fp64(g, x, w, mean, rstd, mode):
    """Per-channel BN backward in fp64. Returns (gx, GB, GW); g/x shape (L,).

    mode 'baseline' = literal (coef division, product epilogue);
    mode 'candidate' = optimized (per = w*rstd*fl(1/L), distributed epilogue).
    """
    L = g.numel()
    xh = (x - mean) * rstd
    GB = g.sum()
    GW = (g * xh).sum()
    scale = w * rstd
    if mode == "baseline":
        coef = scale / float(L)
        gx = coef * (float(L) * g - GB - xh * GW)
    else:
        per = scale * float(1.0 / float(L))  # O1: fl(1/L) compile-time constant
        gx = scale * g - per * GB - per * GW * xh  # O2: distributed epilogue
    return gx, GB, GW


def _moment_gw_fp32(g, x, w, mean, rstd):
    """Rejected raw-moment grad_weight == rstd*(sum(g*x) - mean*sum(g)), in fp32."""
    g32 = g.to(torch.float32)
    x32 = x.to(torch.float32)
    m32 = torch.tensor(float(mean), dtype=torch.float32)
    r32 = torch.tensor(float(rstd), dtype=torch.float32)
    gb = g32.sum()
    gx = (g32 * x32).sum()
    return (r32 * (gx - m32 * gb)).to(torch.float64)


def _moment_channel_fp64(g, x, w, mean, rstd, dtype):
    """Epilogue built on the rejected raw-moment grad_weight (fp32), centered xh."""
    L = g.numel()
    xh = (x - mean) * rstd
    GB = g.sum()
    GW_m = _moment_gw_fp32(g, x, w, mean, rstd)
    scale = w * rstd
    coef = scale / float(L)
    return coef * (float(L) * g - GB - xh * GW_m)


def _make_moment_case(L, dtype, seed):
    """x with a large mean and small spread -> catastrophic cancellation stressor."""
    gen = torch.Generator().manual_seed(seed)
    base = 3000.0 if dtype == torch.float32 else 1000.0
    x = base + torch.randn(L, generator=gen, dtype=torch.float64)
    g = torch.randn(L, generator=gen, dtype=torch.float64) * 0.5
    x = x.to(dtype).to(torch.float64)  # dtype-representable small spread
    g = g.to(dtype).to(torch.float64)
    w = float(torch.randn(1, generator=gen, dtype=torch.float64).item())
    mean = float(x.mean().item())
    rstd = 1.0
    return g, x, w, mean, rstd


def _make_case(n, s, dtype, seed):
    """Return per-channel (g, x, w, mean, rstd) fp64 tensors holding dtype values."""
    gen = torch.Generator().manual_seed(seed)
    L = n * s
    g = torch.randn(L, generator=gen, dtype=torch.float64) * 0.5
    if L > 1:
        g[L // 2:] = torch.randn(L - L // 2, generator=gen, dtype=torch.float64) * 50.0
    x = 3.0e3 + torch.randn(L, generator=gen, dtype=torch.float64) * 0.5
    half = max(1, L // 2)
    x[half:] = torch.randn(L - half, generator=gen, dtype=torch.float64)
    if L >= 4:
        x[half // 2: half] = 2.5

    g = g.to(dtype).to(torch.float64)
    x = x.to(dtype).to(torch.float64)

    if L >= 1:
        x[0] = float("inf")
    if L >= 2:
        g[1] = float("nan")
    if L >= 3:
        x[2] = 1.0e-40 if dtype != torch.float16 else 6e-5  # subnormal

    w = float((torch.randn(1, generator=gen, dtype=torch.float64)).item())
    w = min(max(w, -65504.0 if dtype != torch.float32 else -3.0e38),
            65504.0 if dtype != torch.float32 else 3.0e38)
    mean = float(torch.randn(1, generator=gen, dtype=torch.float64).item())
    rstd = float(torch.rand(1, generator=gen, dtype=torch.float64).item()) + 0.5
    return g, x, w, mean, rstd


def _adopted_items(dtype: torch.dtype):
    results = []
    for i, (n, s) in enumerate(_SHAPES):
        g, x, w, mean, rstd = _make_case(n, s, dtype, seed=5000 + 17 * i + int(dtype == torch.float16))
        gx_base, gb_b, gw_b = _bn_bwd_channel_fp64(g, x, w, mean, rstd, "baseline")
        gx_cand, gb_c, gw_c = _bn_bwd_channel_fp64(g, x, w, mean, rstd, "candidate")
        gx_base_r = _round_to(gx_base, dtype)
        gx_cand_r = _round_to(gx_cand, dtype)
        results.append(
            {
                "item": f"O1+O2 joint  [{_dtype_name(dtype)}, N={n}, S={s}]",
                "dtype": _dtype_name(dtype),
                "shape": (n, s),
                "max_ulp": _ulp_diff(gx_base_r, gx_cand_r, dtype),
                "violation": _violation_rate(gx_base_r, gx_cand_r, dtype),
                "gb_equal": bool((_round_to(gb_b, dtype) == _round_to(gb_c, dtype)).item()),
                "gw_equal": bool((_round_to(gw_b, dtype) == _round_to(gw_c, dtype)).item()),
            }
        )
    return results


def _discr_moment(dtype: torch.dtype):
    out = []
    for i, (n, s) in enumerate(_SHAPES):
        g, x, w, mean, rstd = _make_moment_case(n * s, dtype, seed=5000 + 17 * i)
        gx_base, _, _ = _bn_bwd_channel_fp64(g, x, w, mean, rstd, "baseline")
        gx_mom = _moment_channel_fp64(g, x, w, mean, rstd, dtype)
        gx_base_r = _round_to(gx_base, dtype)
        gx_mom_r = _round_to(gx_mom, dtype)
        out.append(
            {
                "item": f"rejected moment GW vs centered [{_dtype_name(dtype)}, N={n}, S={s}]",
                "dtype": _dtype_name(dtype),
                "max_ulp": _ulp_diff(gx_base_r, gx_mom_r, dtype),
                "violation": _violation_rate(gx_base_r, gx_mom_r, dtype),
            }
        )
    return out


def main() -> int:
    print("=== _batch_norm_bwd_split_kernel §1.6.1 equivalence verification (fp64 ref, torch CPU) ===")
    print("(adopted O1 reciprocal-mul for /L ; O2 distributed epilogue)\n")

    adaptive = []
    for d in _FLOAT_DTYPES:
        adaptive.extend(_adopted_items(d))

    print(f"{'adopted item (candidate vs baseline)':<44} | {'max ulp':>9} | {'viol':>8} | verdict")
    print("-" * 88)
    all_pass = True
    for r in adaptive:
        ok = (r["violation"] == 0.0) and math.isfinite(r["max_ulp"])
        if r["dtype"] == "float32" and r["max_ulp"] > 1e6:
            ok = False
        verdict = "EQUIV_PASS" if ok else "EQUIV_FAIL"
        all_pass = all_pass and ok
        print(f"{r['item']:<44} | {r['max_ulp']:>9.2f} | {r['violation']:>8.4f} | {verdict}")

    print("\nadopted-item result table (one row per adopted item):")
    print(f"{'adopted item':<34} | {'dtype':<8} | {'max ulp':>10} | {'viol':>8} | verdict")
    print("-" * 84)
    by_dtype = {}
    for r in adaptive:
        by_dtype.setdefault(r["dtype"], []).append(r)
    for dname, rows in by_dtype.items():
        m_ulp = max(r["max_ulp"] for r in rows)
        m_vio = max(r["violation"] for r in rows)
        v = "EQUIV_PASS" if m_vio == 0.0 else "EQUIV_FAIL"
        print(f"{'O1+O2 (joint)':<34} | {dname:<8} | {m_ulp:>10.2f} | {m_vio:>8.4f} | {v}")

    print("\ndiscriminant A (rejected form): raw-moment grad_weight = rstd*(Sg.x - mean*Sg) (fp32)\n"
          "vs centered Sg.xh -- not equivalent:")
    print(f"{'case':<60} | {'viol':>8} | conclusion")
    print("-" * 84)
    discr_a = {}
    for d in _FLOAT_DTYPES:
        rows = _discr_moment(d)
        discr_a[_dtype_name(d)] = max(r["violation"] for r in rows)
        for r in rows:
            print(f"{r['item']:<60} | {r['violation']:>8.4f} | "
                  f"{'DISCRIM_OK' if r['violation'] > 0.0 else 'no-violation (this shape)'}")
    discr_a_ok = discr_a["float32"] > 0.0
    print(f"discriminant A criterion: fp32 max violation={discr_a['float32']:.4f} "
          f"(atol=rtol=1e-5 must be > 0), fp16 max={discr_a['float16']:.4f}, "
          f"bf16 max={discr_a['bfloat16']:.4f} (coarser tolerances mask the same "
          f"cancellation -- why the centered form is kept across dtypes) "
          f"-> {'DISCRIM_OK' if discr_a_ok else 'DISCRIM_FAIL'}")

    ok_all = all_pass and discr_a_ok
    print("\n" + "=" * 78)
    if ok_all:
        print("overall: adopted O1+O2 all EQUIV_PASS; raw-moment form rejected -- adoptable")
    else:
        print("overall: a failure exists -- adoption forbidden")
    print("=" * 78)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
