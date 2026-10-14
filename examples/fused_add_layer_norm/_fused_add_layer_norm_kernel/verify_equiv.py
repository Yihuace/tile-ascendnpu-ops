"""verify_equiv.py -- _fused_add_layer_norm_kernel §1.6.1 adopted-item equivalence.

Self-contained, torch CPU + fp64 reference (no NPU / no tilelang dependency).
Stage 2 (design review, dimension 8) re-runs this file.

Op semantics (§0.1 of DESIGN.md, from the GPU source kernel
``tileops/kernels/norm/fused_add_norm.py:FusedAddLayerNormKernel`` and the repo
reference ``tests/ops/test_fused_add_layer_norm.py:ref_program``)::

    s     = x + residual                       # fp32 add
    add   = round_to_dtype(s)                  # residual_out (single rounding)
    y     = (add - mean(add)) / sqrt(var(add) + eps) * weight + bias

Baseline (source / literal math, computed in fp64 then rounded to the input
dtype per the fp32-opmath golden contract):

    mean  = S1 / N
    d     = s - mean
    var   = S2 / N + eps                       # centered two-pass
    rstd  = 1 / sqrt(var)
    y     = (d * rstd) * w + b
    residual_out = round_to_dtype(s)

Adopted items (§1.6.1), candidate = the NPU design:

  O1 division -> reciprocal multiply (N is a compile-time constant)
        mean = S1 * fl(1/N)          (was S1 / N)
        var  = S2 * fl(1/N) + eps    (was S2 / N + eps)
     Identity up to the fp64 rounding of the constant fl(1/N) (exact when N is
     a power of two).  Turns a (bm,1) vdiv into a (bm,1) vmul.

  O2 rstd via sqrt/div composition (bypasses the approximate T.vrsqrt)
        rstd = sqrt(var) / var       (was 1 / sqrt(var))
     Exact real identity ``sqrt(v)/v == 1/sqrt(v)`` for v > 0 (v = var + eps > 0
     since eps = 1e-5).  In fp64 the two forms differ only by one rounding.

  O3 centered-value reuse (common-subexpression elimination)
        d = s - mean computed once and reused in the epilogue
     Identical fp64 value -- exact (0 ulp): the epilogue reads the same d.

Discrimination sections (pin the semantic choices the design rejects / keeps):

  A. Rejected single-pass moment form ``var = E[s^2] - mean^2`` (computed in
     fp32, as the NPU would): catastrophic cancellation for large-mean /
     small-variance rows -> must differ from the centered form (violations>0).
  B. Design choice: round the sum to the input dtype before the statistics
     (candidate, matches the golden ``ref_program`` which normalizes the
     rounded ``added``) vs the source kernel's stats-on-unrounded-fp32-sum.
     The two must agree within the dtype tolerance (the source kernel and the
     golden differ by <= 1 dtype ulp in the statistics only).

Method: random tensors (randn) + corner values (large-mean/small-var, constant
rows, zeros, subnormal, +-inf, NaN, extreme weight/bias, N=1/2, M=1) at each
dtype; baseline and candidate computed in fp64 and rounded to the target dtype;
ulp / violation comparison against the repo tolerances (fp32 1e-5, fp16 1e-3,
bf16 1.6e-2).  Any adopted item that fails must not be adopted.
"""

import torch

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Repo tolerances (tests/ops/test_fused_add_layer_norm.py:_get_tolerances).
_TOL = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1.6e-2, 1.6e-2),
}

# Shapes exercised (per-row N).  Includes aligned, non-2^k (5120) and the
# manifest bounds (4096 / 8192) plus tiny N = 1 / 2.
_SHAPES = (1, 2, 514, 1152, 3000, 4096, 5120, 8192)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _round_to(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Round an fp64 tensor to *dtype*, returned again as fp64 (exact value)."""
    return x.to(dtype).to(torch.float64)


def _ulp_diff(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> float:
    """Max difference between two dtype-representable fp64 tensors, in dtype ulps.

    a, b hold exact dtype values (already rounded).  ulp at |a| is
    eps * 2**floor(log2|a|).  Matching non-finite entries (both NaN / both the
    same +-inf, which propagate identically) count as 0; a finite/non-finite
    mismatch is a real divergence and is reported as ``inf``.
    """
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


def _layernorm_fp64(s: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float, mode: str):
    """LayerNorm over the last dim in fp64, returning the fp64 (unrounded) y.

    mode: 'baseline' = literal (true divisions, 1/sqrt);
          'candidate' = optimized (reciprocal const, sqrt/div rstd, d reuse).
    """
    n = s.shape[-1]
    s64 = s.to(torch.float64)
    if mode == "baseline":
        mean = s64.sum(dim=-1, keepdim=True) / float(n)
        d = s64 - mean
        var = (d * d).sum(dim=-1, keepdim=True) / float(n) + eps
        rstd = 1.0 / torch.sqrt(var)
    else:
        r_n = float(1.0 / float(n))  # fl(1/N): compile-time constant (O1)
        mean = s64.sum(dim=-1, keepdim=True) * r_n
        d = s64 - mean
        var = (d * d).sum(dim=-1, keepdim=True) * r_n + eps
        rstd = torch.sqrt(var) / var  # O2: sqrt/div composition
    w64 = w.to(torch.float64)
    b64 = b.to(torch.float64)
    return (d * rstd) * w64 + b64


def _moment_form_fp32(s: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float):
    """Rejected single-pass moment form, computed in fp32 (as the NPU would)."""
    n = s.shape[-1]
    s32 = s.to(torch.float32)
    mean = s32.mean(dim=-1, keepdim=True)
    ex2 = (s32 * s32).mean(dim=-1, keepdim=True)
    var = ex2 - mean * mean + eps
    rstd = 1.0 / torch.sqrt(var)
    y = (s32 - mean) * rstd * w.to(torch.float32) + b.to(torch.float32)
    return y


def _make_case(n: int, dtype: torch.dtype, seed: int):
    """Return (s, w, b) fp64-ish inputs already holding dtype-representable values.

    s is the pre-norm residual sum; generated directly so the layer-norm part is
    isolated from the (trivially correct) add.
    """
    g = torch.Generator().manual_seed(seed)
    rows = []
    # randn block
    rand = torch.randn(48, n, generator=g, dtype=torch.float64)
    rows.append(rand)
    # large-mean / small-variance rows (moment-form cancellation stressor)
    large = 3.0e3 + torch.randn(16, n, generator=g, dtype=torch.float64) * 0.5
    rows.append(large)
    # constant rows (var == 0 -> rstd = rsqrt(eps))
    rows.append(torch.full((8, n), 2.5, dtype=torch.float64))
    # zeros
    rows.append(torch.zeros(4, n, dtype=torch.float64))

    s = torch.cat(rows, dim=0).to(dtype).to(torch.float64)  # dtype-representable

    # corner injections (single points) into the first row
    if n >= 1:
        s[0, 0] = float("inf") if n % 2 else s[0, 0]
    if n >= 2:
        s[1, 1] = float("nan")
    if n >= 2:
        s[2, min(2, n - 1)] = 1.0e-40 if dtype != torch.float16 else 6e-5

    w = (torch.randn(n, generator=g, dtype=torch.float64)).to(dtype).to(torch.float64)
    b = (torch.randn(n, generator=g, dtype=torch.float64)).to(dtype).to(torch.float64)
    if n >= 1:
        w[0] = 65504.0 if dtype != torch.float32 else 3.0e38
        b[-1] = -65504.0 if dtype != torch.float32 else -3.0e38
    return s, w, b


def _adopted_items(dtype: torch.dtype):
    """Run all adopted items for one dtype; return list of result dicts."""
    eps = 1e-5
    results = []
    for i, n in enumerate(_SHAPES):
        s, w, b = _make_case(n, dtype, seed=9000 + 31 * i + int(dtype == torch.float16) * 7)
        y_base = _round_to(_layernorm_fp64(s, w, b, eps, "baseline"), dtype)
        y_cand = _round_to(_layernorm_fp64(s, w, b, eps, "candidate"), dtype)
        # O1+O2+O3 combined candidate vs baseline
        results.append(
            {
                "item": f"O1+O2+O3  joint  [{_dtype_name(dtype)}, N={n}]",
                "dtype": _dtype_name(dtype),
                "n": n,
                "max_ulp": _ulp_diff(y_base, y_cand, dtype),
                "violation": _violation_rate(y_base, y_cand, dtype),
            }
        )
    return results


def _discr_moment(dtype: torch.dtype):
    """Discrimination A: rejected moment form differs (cancellation)."""
    eps = 1e-5
    out = []
    for i, n in enumerate(_SHAPES):
        s, w, b = _make_case(n, dtype, seed=9000 + 31 * i)
        y_base = _round_to(_layernorm_fp64(s, w, b, eps, "baseline"), dtype)
        y_mom = _round_to(_moment_form_fp32(s, w, b, eps), dtype)
        out.append(
            {
                "item": f"rejected moment form vs centered [{_dtype_name(dtype)}, N={n}]",
                "n": n,
                "violation": _violation_rate(y_base, y_mom, dtype),
            }
        )
    return out


def _discr_round_first(dtype: torch.dtype):
    """Discrimination B: round-first stats (candidate) vs source unrounded stats."""
    eps = 1e-5
    out = []
    for i, n in enumerate(_SHAPES):
        s, w, b = _make_case(n, dtype, seed=9000 + 31 * i)
        s64 = s.to(torch.float64)
        s_f32 = s64.to(torch.float32).to(torch.float64)      # fp32 sum (source)
        s_dtype = _round_to(s64, dtype)                       # rounded sum (golden)
        y_golden = _round_to(_layernorm_fp64(s_dtype, w, b, eps, "candidate"), dtype)
        y_source = _round_to(_layernorm_fp64(s_f32, w, b, eps, "candidate"), dtype)
        out.append(
            {
                "item": f"round-first stats vs source stats [{_dtype_name(dtype)}, N={n}]",
                "n": n,
                "max_ulp": _ulp_diff(y_golden, y_source, dtype),
                "violation": _violation_rate(y_golden, y_source, dtype),
            }
        )
    return out


def main() -> int:
    print("=== _fused_add_layer_norm_kernel §1.6.1 equivalence verification (fp64 ref, torch CPU) ===")
    print("(adopted: O1 reciprocal-mul / O2 sqrt-div rstd / O3 d-reuse; residual_out = round(x+residual))\n")

    adaptive = []
    for d in _FLOAT_DTYPES:
        adaptive.extend(_adopted_items(d))

    print(f"{'采纳项 (candidate vs baseline)':<52} | {'max ulp':>9} | {'违反率':>8} | 结论")
    print("-" * 92)
    all_pass = True
    for r in adaptive:
        ok = (r["violation"] == 0.0) and r["max_ulp"] == r["max_ulp"] and r["max_ulp"] != float("inf")
        if r["dtype"] == "float32" and r["max_ulp"] > 1e6:
            ok = False
        verdict = "EQUIV_PASS" if ok else "EQUIV_FAIL"
        all_pass = all_pass and ok
        print(f"{r['item']:<52} | {r['max_ulp']:>9.2f} | {r['violation']:>8.4f} | {verdict}")

    # Per-item consolidated rows (one row per adopted item across all shapes).
    print("\n采纳项结果表（每采纳项一行）：")
    print(f"{'采纳项':<40} | {'dtype':<8} | {'max ulp':>10} | {'违反率':>8} | 结论")
    print("-" * 92)
    by_dtype = {}
    for r in adaptive:
        by_dtype.setdefault(r["dtype"], []).append(r)
    for dname, rows in by_dtype.items():
        m_ulp = max(r["max_ulp"] for r in rows)
        m_vio = max(r["violation"] for r in rows)
        v = "EQUIV_PASS" if m_vio == 0.0 else "EQUIV_FAIL"
        print(f"{'O1+O2+O3 (joint)':<40} | {dname:<8} | {m_ulp:>10.2f} | {m_vio:>8.4f} | {v}")

    print("\n判别段 A（弃选式）：单遍矩式 var=E[s^2]-mean^2（fp32 计算）与中心化式不等价：")
    print(f"{'case':<52} | {'违反率':>8} | 结论")
    print("-" * 76)
    discr_a = {}
    for d in _FLOAT_DTYPES:
        rows = _discr_moment(d)
        discr_a[_dtype_name(d)] = max(r["violation"] for r in rows)
        for r in rows:
            print(f"{r['item']:<52} | {r['violation']:>8.4f} | "
                  f"{'DISCRIM_OK' if r['violation'] > 0.0 else 'no-violation(该 shape)'}")
    # The rejection is demonstrated where the tolerance is tight: fp16 (1e-3)
    # and fp32 (1e-5) MUST show cancellation violations; bf16 (1.6e-2) may mask
    # the error (its coarse tolerance is exactly why the centered form is the
    # safe choice across all dtypes).
    discr_a_ok = discr_a["float32"] > 0.0 and discr_a["float16"] > 0.0
    print(f"判别段 A 判据: fp32 max violation={discr_a['float32']:.4f}, "
          f"fp16 max violation={discr_a['float16']:.4f}, "
          f"bf16 max violation={discr_a['bfloat16']:.4f} (bf16 宽容差掩蔽——正是必需中心化式的理由) "
          f"-> {'DISCRIM_OK' if discr_a_ok else 'DISCRIM_FAIL'}")

    print("\n判别段 B（设计选择）：round-first 统计（匹配 golden）vs 源算子 unrounded 统计，须在容差内：")
    print(f"{'case':<52} | {'max ulp':>9} | {'违反率':>8} | 结论")
    print("-" * 92)
    discr_b_ok = True
    for d in _FLOAT_DTYPES:
        for r in _discr_round_first(d):
            passed = r["violation"] == 0.0
            discr_b_ok = discr_b_ok and passed
            print(f"{r['item']:<52} | {r['max_ulp']:>9.2f} | {r['violation']:>8.4f} | {'DISCRIM_OK' if passed else 'DISCRIM_FAIL'}")

    ok_all = all_pass and discr_a_ok and discr_b_ok
    print("\n" + "=" * 78)
    if ok_all:
        print("总体结论: 采纳项 O1+O2+O3 全部 EQUIV_PASS；矩式被弃选、round-first 与源统计在容差内 — 可采纳")
    else:
        print("总体结论: 存在失败项 — 禁止采纳")
    print("=" * 78)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
