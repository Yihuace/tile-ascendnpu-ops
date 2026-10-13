"""Equivalence machine-verification for GeluAndMulFwdOp design adoptions (D-1).

Runs on torch CPU only (no NPU).  For each adopted optimization in
DESIGN.md §1.6.1 this script compares the *candidate* formula (the NPU
algorithm) against the *baseline* formula (the GPU source algorithm) and the
independent fp64 reference on random gated tensors plus IEEE corner values,
then prints an ulp / violation table.

Revision history
----------------
v2 (this file) replaces the v1 native-vector-``T.verf`` mechanism, which was
found NOT to be a real erf on this toolchain: ``T.verf`` lowers to a 4-term
truncated Maclaurin series ``erf(x) ~ (2/sqrt(pi)) (x - x^3/3 + x^5/10 -
x^7/42)`` (``tilelang-mlir-ascend/src/target/codegen_npuir_api.cc``
``VerfCodegen``), valid only for |x| <~ 1, and ``T.erf`` routes to the same
codegen.  GELU's erf argument ``gate/sqrt(2)`` is ~N(0,0.5), so ~16% of samples
have |x| > 1 and the kernel diverges there.  The candidate is now an explicit
clamped minimax polynomial erf, evaluated in fp32, valid over the whole real
line (verified by the RANGE section below).

Adopted items (see DESIGN.md §1.6.1):

  item 1  explicit clamped degree-20 minimax polynomial erf (fp32) in place of
          the source erf.  The source dispatches to the GPU degree-10 clamped
          polynomial for fp16/bf16 and to an exact erf for fp32; the NPU has
          neither (``T.verf``/``T.erf`` are the broken truncated series above),
          so one fp32-accurate polynomial serves all three dtypes.
            baseline : GPU source erf -- clamped degree-10 polynomial + keep_nan
                       for fp16/bf16, exact erf (torch.erf) for fp32
            candidate: explicit clamped degree-20 polynomial (torch emulation of
                       the DESIGN.md §0.6 R2 mechanism), fp32
            reference: erf(...) in fp64
          Both arms run the same fp32 GELU-and-mul chain; only the erf body
          differs.

  item 2  full fp32-domain chain with a single final cast to the storage dtype
          (the NPU algorithm) in place of the GPU narrow-dtype tail arithmetic.
            baseline : erf cast to dtype, then  g * 0.5 * (1 + erf) * value
                       evaluated entirely in the storage dtype
            candidate: gate*0.5*(1+erf)*value evaluated in fp32, cast once at the end
            reference: same expression in fp64, rounded once to dtype
          The candidate matches torch's fp32-opmath golden (F.gelu(gate)*value).

  counter fp32 polynomial-erf arm (the *rejected* GPU degree-10 baseline) vs the
          fp64 reference, evaluated in fp32.  This quantifies why the source
          dispatches fp32 to an exact erf instead of the deg-10 polynomial
          (DESIGN.md §1.6.1 对照项), and is also why the NPU path cannot reuse
          the deg-10 body for fp32.  Rejected-baseline control, not an adopted
          candidate; not required to pass.

RANGE verification (new in v2) asserts the erf mechanism is faithful beyond
|x| < 1, covering the exact failure the v1 ``T.verf`` design had:

  RANGE  erf(e) full-range max abs error vs fp64 over a dense grid on [-8, 8];
  RANGE  exact saturation: erf(+inf) = +1, erf(-inf) = -1, erf(-0.0) = -0.0;
  RANGE  NaN propagation: erf(NaN) = NaN;
  BOUNDARY  gelu(-inf) = NaN and gelu(+inf) = +inf, for both the source
         expression x * 0.5 * (1 + erf(x/sqrt(2))) and the torch golden, all
         three dtypes.

Output: one table row per (adopted item x dtype x shape), plus an aggregate,
plus the fp32 polynomial counter table, plus the RANGE / BOUNDARY table.  Exit
code is non-zero if any adopted candidate exceeds its tolerance or a
range/boundary check fails.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

INV_SQRT2 = 0.7071067811865476

# ---- GPU source erf (baseline / rejected counter) ------------------------
_GPU_CLAMP = 3.6
_GPU_POLY_COEFFS = (
    8.971932411193848,
    -33.1397590637207,
    53.60123062133789,
    -48.02798080444336,
    26.02545738220215,
    -8.488715171813965,
    1.7499406337738037,
    -0.09359221160411835,
    0.11330155283212662,
    0.1387363225221634,
    0.2777777910232544,
)

# ---- NPU explicit clamped minimax polynomial erf (candidate) -------------
# erf(x) ~= clip(t * P(w), -1, 1),  t = clamp(x, -A, A),  w = 1 - (t/A)^2,
# P(w) = c0 + c1 w + ... + c20 w^20  (horner, highest degree first below).
# c0 = 1/A = 0.25 is set exactly so that t*P(0) = +/-1 exactly in fp32 at
# |x| >= A: the output clip then yields exactly +/-1, which is what makes
# gelu(-inf) = (-inf)*0 = NaN reproduce.  A = 4 so erf(A) >= 1 - 1.5e-8.
# fp32 full-range max abs error of this body vs fp64 erf: ~3.2e-7.
_ERF_A = 4.0
_ERF_POLY_COEFFS = (
    # highest degree first (w^20 -> w^0)
    0.6223669518555077,
    -1.9650265648484124,
    1.7959468299705577,
    0.6293564960002123,
    -1.2602024549300905,
    -0.6436158375195619,
    0.9124701824538465,
    0.7893043491919204,
    -0.6004141464231496,
    -0.6852507940864765,
    0.6814123183594656,
    0.5075096218729144,
    -0.8645258038808001,
    0.6561585328209293,
    -0.16863897097754185,
    0.11303623533482877,
    0.0610352907567128,
    0.07873436862465412,
    0.09372210642535163,
    0.12500045811449384,
    0.25,  # = 1/A, exact saturation
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
DTYPE_NAME = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}

# Gated (M, N) shapes: the smoke row (16,128), a couple of extra widths, and a
# couple of wider rows.  (The 2048x14336 manifest workload is not enumerated --
# the equivalence is element-wise and shape-independent, and the corner values
# are overlaid on every shape.)
SHAPES = [(16, 128), (8, 256), (1, 128), (4, 64), (32, 128), (64, 512), (3, 257)]


def _tolerances(dtype: torch.dtype) -> tuple:
    # Mirrors tests/ops/test_gelu_and_mul.py::_get_tolerances.
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.float16:
        return 1e-2, 1e-2
    return 1.6e-2, 1.6e-2


def _corner_values(dtype: torch.dtype) -> torch.Tensor:
    fi = torch.finfo(dtype)
    vals = [
        0.0, -0.0, 1.0, -1.0, fi.max, -fi.max, fi.tiny, -fi.tiny,
        float("inf"), float("-inf"), float("nan"), 1.5, -2.25, 3.0, -3.0,
        0.6931, -4.5, 4.5, 1e-4, -1e-4,
    ]
    return torch.tensor(vals, dtype=dtype)


def _make_gate_value(M: int, N: int, dtype: torch.dtype, seed: int):
    torch.manual_seed(seed)
    gate = torch.randn(M, N, dtype=torch.float32).to(dtype)
    value = torch.randn(M, N, dtype=torch.float32).to(dtype)
    corners = _corner_values(dtype)
    k = min(corners.numel(), M * N)
    gate.view(-1)[:k] = corners[:k]
    value.view(-1)[:k] = corners[:k]
    return gate, value


# ---- erf bodies ----------------------------------------------------------

def _poly_erf_f32(wide: torch.Tensor) -> torch.Tensor:
    """fp32 clamped minimax polynomial erf (torch emulation of DESIGN.md §0.6 R2).

    Mirrors the NPU op chain: vclamp -> vmul/vmul/vsub (w) -> Horner ->
    vmul (t*P) -> vclamp (clip) -> NaN restore.  All arithmetic is fp32 so the
    emulation matches the kernel's rounding.
    """
    A = _ERF_A
    one = torch.tensor(1.0, dtype=torch.float32)
    neg_a = torch.tensor(-A, dtype=torch.float32)
    hi_a = torch.tensor(A, dtype=torch.float32)
    inv_a = torch.tensor(1.0 / A, dtype=torch.float32)
    t = torch.minimum(torch.maximum(wide, neg_a), hi_a)
    w = one - (t * inv_a) * (t * inv_a)
    acc = torch.full_like(t, _ERF_POLY_COEFFS[0])
    for coeff in _ERF_POLY_COEFFS[1:]:
        acc = acc * w + coeff
    e = torch.minimum(torch.maximum(t * acc, -one), one)
    return torch.where(torch.isnan(wide), wide, e)  # keep_nan


def _gpu_poly_erf_f32(wide: torch.Tensor) -> torch.Tensor:
    """fp32 GPU source clamped degree-10 polynomial erf (torch emulation of
    kernels/elementwise/_erf.py) -- baseline for narrow dtypes, rejected for
    fp32."""
    one = torch.tensor(1.0, dtype=torch.float32)
    lo = torch.tensor(-_GPU_CLAMP, dtype=torch.float32)
    hi = torch.tensor(_GPU_CLAMP, dtype=torch.float32)
    clamped = torch.minimum(torch.maximum(wide, lo), hi)
    scaled = clamped * torch.tensor(1.0 / _GPU_CLAMP, dtype=torch.float32)
    w = one - scaled * scaled
    acc = torch.full_like(wide, _GPU_POLY_COEFFS[0])
    for coeff in _GPU_POLY_COEFFS[1:]:
        acc = acc * w + coeff
    r = torch.minimum(torch.maximum(clamped * acc, -one), one)
    return torch.where(torch.isnan(wide), wide, r)  # keep_nan


def _gelu_mul_f32(gate, value, erf_fn) -> torch.Tensor:
    """gate * 0.5 * (1 + erf(gate/sqrt(2))) * value, all in fp32."""
    g = gate.float()
    v = value.float()
    e = erf_fn(g * INV_SQRT2)
    return g * 0.5 * (1.0 + e) * v


def _gelu_mul_narrow(gate, value, dtype: torch.dtype) -> torch.Tensor:
    """GPU narrow-dtype tail: exact erf in fp32 -> cast to dtype, products in dtype."""
    g32 = gate.float()
    e = torch.erf(g32 * INV_SQRT2).to(dtype)
    half = torch.tensor(0.5, dtype=dtype)
    one = torch.tensor(1.0, dtype=dtype)
    return gate * half * (one + e) * value


def _score(cand, ref, dtype, atol, rtol):
    """Return (max_ulp_in_tolerance_units, violations, nan_mismatches)."""
    assert cand.shape == ref.shape
    c64, r64 = cand.double(), ref.double()
    both_nan = torch.isnan(c64) & torch.isnan(r64)
    nan_mis = int((torch.isnan(c64) != torch.isnan(r64)).sum().item())
    band = atol + rtol * r64.abs()
    diff = (c64 - r64).abs()
    exact = both_nan | (cand == ref)
    units = torch.where(exact, torch.zeros_like(diff), diff / band)
    max_ulp = float(units.max().item()) if units.numel() else 0.0
    close = (diff <= band) | both_nan | (cand == ref)
    violations = int((~close).sum().item())
    return max_ulp, violations, nan_mis


# ---- range / boundary verification --------------------------------------

def _check_range() -> bool:
    """Assert the candidate erf is faithful over the whole real line.

    This is the machine-checkable form of the v2 correction: the v1 ``T.verf``
    mechanism was valid only for |x| <~ 1.  Checks full-range accuracy plus exact
    saturation and NaN propagation.
    """
    ok = True
    print("RANGE  explicit clamped degree-20 polynomial erf (fp32) vs fp64")
    xs = torch.linspace(-8.0, 8.0, 400001, dtype=torch.float64)
    ref = torch.erf(xs).float()
    got = _poly_erf_f32(xs.float())
    # exclude the finite grid beyond the clamp transition? none -- full range.
    max_err = float((got.double() - ref.double()).abs().max())
    sat_ok = True
    sat = _poly_erf_f32(
        torch.tensor(
            [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 4.0, -4.0],
            dtype=torch.float32,
        )
    )
    inf_p = bool(sat[0] == 1.0)
    inf_n = bool(sat[1] == -1.0)
    nan_p = bool(torch.isnan(sat[2]))
    z_ok = bool(sat[3] == 0.0 and torch.signbit(sat[4]))  # erf(-0.0) = -0.0
    a_ok = bool(sat[5] == 1.0 and sat[6] == -1.0)
    sat_ok = inf_p and inf_n and nan_p and z_ok and a_ok
    acc_ok = max_err < _tolerances(torch.float32)[0]
    ok = ok and sat_ok and acc_ok
    print(
        f"  full-range |x|<=8 max abs err (fp32 vs fp64) = {max_err:.3e}"
        f"  (< fp32 atol={_tolerances(torch.float32)[0]:g}: {acc_ok})"
    )
    print(
        f"  saturation: erf(+inf)={sat[0].item()} erf(-inf)={sat[1].item()} "
        f"erf(NaN)=NaN:{nan_p} erf(0)={sat[3].item()} erf(-0.0) signbit:{torch.signbit(sat[4]).item()} "
        f"erf(+-4)={sat[5].item()},{sat[6].item()}  ({'OK' if sat_ok else 'MISMATCH'})"
    )
    return ok


def _check_boundary_semantics() -> bool:
    """Assert DESIGN §0.1/§0.6 R2 boundary semantics: gelu(-inf)=NaN, gelu(+inf)=+inf.

    Checks both the source activation chain (``x*half*(one+erf(x/sqrt(2)))``) and
    the torch golden, for all three dtypes, using the candidate polynomial erf.
    """
    ok = True
    print("BOUNDARY  gelu(-inf)=NaN / gelu(+inf)=+inf (source expr & golden)")
    for dtype in DTYPES:
        gate = torch.tensor([float("-inf"), float("inf")], dtype=dtype)
        value = torch.ones_like(gate)
        src = _gelu_mul_f32(gate, value, _poly_erf_f32).to(dtype).float()
        gold = (F.gelu(gate.float()) * value.float()).to(dtype).float()
        neg_ok = bool(torch.isnan(src[0]) and torch.isnan(gold[0]))
        pos_ok = bool(src[1] == float("inf") and gold[1] == float("inf"))
        ok = ok and neg_ok and pos_ok
        print(
            f"  {DTYPE_NAME[dtype]:<9} source={src.tolist()} golden={gold.tolist()}"
            f"  {'OK' if neg_ok and pos_ok else 'MISMATCH'}"
        )
    return ok


def _counter_fp32_gpu_poly_vs_ref():
    """Rejected GPU baseline: fp32 degree-10 clamped polynomial erf vs fp64 ref."""
    rows = []
    atol, rtol = _tolerances(torch.float32)
    for (M, N) in SHAPES:
        gate, value = _make_gate_value(M, N, torch.float32, seed=M * 131 + N)
        g64, v64 = gate.double(), value.double()
        ref64 = (g64 * 0.5 * (1.0 + torch.erf(g64 * INV_SQRT2)) * v64).to(torch.float32)
        poly = _gelu_mul_f32(gate, value, _gpu_poly_erf_f32).to(torch.float32)
        u, v, n = _score(poly, ref64, torch.float32, atol, rtol)
        rows.append((f"{M}x{N}", u, v, n))
    return rows


def run() -> int:
    rows = []
    failed = not _check_range()
    failed = (not _check_boundary_semantics()) or failed

    for dtype, (M, N) in itertools.product(DTYPES, SHAPES):
        atol, rtol = _tolerances(dtype)
        gate, value = _make_gate_value(M, N, dtype, seed=M * 131 + N + int(dtype is torch.bfloat16))
        # Independent fp64 reference, rounded once to the storage dtype.
        g64 = gate.double()
        v64 = value.double()
        ref64 = (g64 * 0.5 * (1.0 + torch.erf(g64 * INV_SQRT2)) * v64).to(dtype)
        # torch fp32-opmath golden (matches tests/ops/test_gelu_and_mul.py).
        golden = (F.gelu(gate.float()) * value.float()).to(dtype)

        cand = _gelu_mul_f32(gate, value, _poly_erf_f32).to(dtype)

        # --- item 1: explicit degree-20 polynomial erf vs source erf.  The
        # source uses the GPU deg-10 polynomial for fp16/bf16 and an exact erf
        # for fp32 (which is not available on the NPU path). ---
        if dtype == torch.float32:
            base1 = _gelu_mul_f32(gate, value, torch.erf).to(dtype)
        else:
            base1 = _gelu_mul_f32(gate, value, _gpu_poly_erf_f32).to(dtype)
        u1, v1, n1 = _score(cand, base1, dtype, atol, rtol)
        u1r, v1r, n1r = _score(cand, ref64, dtype, atol, rtol)
        ub1, vb1, _ = _score(base1, ref64, dtype, atol, rtol)
        rows.append(dict(
            item="1 explicit-erf vs source-erf", dtype=DTYPE_NAME[dtype], hw=f"{M}x{N}",
            cand_ulp=u1, cand_viol=v1, cand_nan=n1, cand_vs64=u1r,
            base_ulp=ub1, base_viol=vb1,
            verdict="EQUIV_PASS" if (v1 == 0 and n1 == 0 and v1r == 0 and n1r == 0) else "EQUIV_FAIL",
        ))
        if v1 or n1 or v1r or n1r:
            failed = True

        # --- item 2: fp32-transit chain vs narrow-dtype chain ---
        base2 = _gelu_mul_narrow(gate, value, dtype)
        u2, v2, n2 = _score(cand, base2, dtype, atol, rtol)
        u2g, v2g, n2g = _score(cand, golden, dtype, atol, rtol)
        ub2, vb2, _ = _score(base2, ref64, dtype, atol, rtol)
        rows.append(dict(
            item="2 fp32-transit vs narrow", dtype=DTYPE_NAME[dtype], hw=f"{M}x{N}",
            cand_ulp=u2, cand_viol=v2, cand_nan=n2, cand_vs64=u2g,
            base_ulp=ub2, base_viol=vb2,
            verdict="EQUIV_PASS" if (v2 == 0 and n2 == 0 and v2g == 0 and n2g == 0) else "EQUIV_FAIL",
        ))
        if v2 or n2 or v2g or n2g:
            failed = True

    hdr = (
        f"{'item':<30}{'dtype':<10}{'MxN':<9}{'cand_ulp':>10}{'cand_viol':>10}"
        f"{'cand_nan':>9}{'cand_vs64':>11}{'base_ulp':>10}{'base_viol':>10}  verdict"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['item']:<30}{r['dtype']:<10}{r['hw']:<9}{r['cand_ulp']:>10.4f}"
            f"{r['cand_viol']:>10}{r['cand_nan']:>9}{r['cand_vs64']:>11.4f}"
            f"{r['base_ulp']:>10.4f}{r['base_viol']:>10}  {r['verdict']}"
        )
    print("-" * len(hdr))

    def _agg(prefix):
        sub = [r for r in rows if r["item"].startswith(prefix)]
        return (
            len(sub),
            max(r["cand_ulp"] for r in sub),
            sum(r["cand_viol"] for r in sub),
            sum(r["cand_nan"] for r in sub),
            max(r["cand_vs64"] for r in sub),
            max(r["base_ulp"] for r in sub),
            sum(r["base_viol"] for r in sub),
        )

    print("AGGREGATE  item | rows | cand_ulp | cand_viol | cand_nan | cand_vs64 | base_ulp | base_viol")
    print("  item1 explicit-erf :", _agg("1"))
    print("  item2 fp32-transit :", _agg("2"))
    n_pass = sum(1 for r in rows if r["verdict"] == "EQUIV_PASS")
    print(f"total rows: {len(rows)}   EQUIV_PASS: {n_pass}   not-pass: {len(rows) - n_pass}")

    # --- rejected-baseline control: fp32 GPU deg-10 polynomial erf vs fp64.
    # Not an adopted item; expected to exceed the fp32 tolerance (that is the
    # evidence behind DESIGN §1.6.1's counter-item and the v2 correction). ---
    counter = _counter_fp32_gpu_poly_vs_ref()
    chdr = f"{'counter fp32-gpu-poly':<28}{'MxN':<9}{'max_ulp':>10}{'viol':>8}{'nan':>6}"
    print(chdr)
    print("-" * len(chdr))
    for hw, u, v, n in counter:
        print(f"{'gpu-poly-erf vs fp64':<28}{hw:<9}{u:>10.4f}{v:>8}{n:>6}")
    c_max = max(u for _, u, _, _ in counter)
    c_viol = sum(v for _, _, v, _ in counter)
    print(
        f"counter fp32-gpu-poly: rows={len(counter)} max_ulp={c_max:.4f} "
        f"total_viol={c_viol} (rejected baseline; not required to pass)"
    )

    print("OVERALL:", "EQUIV_PASS" if not failed else "EQUIV_FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
