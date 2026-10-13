"""Equivalence machine-verification for GeluTanhAndMulFwdOp design adoptions (D-1).

Runs on torch CPU only (no NPU).  For each adopted optimization in
DESIGN.md §1.6.1 this script compares the *candidate* formula (the NPU
algorithm) against the *baseline* formula (the GPU source algorithm) and the
independent fp64 reference on random gated tensors plus IEEE corner values,
then prints an ulp / violation table.

Context (why an explicit tanh mechanism, not ``T.vtanh``)
---------------------------------------------------------
The GPU source ``GeluTanhAndMulFwdKernel`` uses ``T.tanh`` (faithful on CUDA)
inside ``gelu_tanh(g) = 0.5 * g * (1 + tanh(u))`` with
``u = sqrt(2/pi) (g + 0.044715 g^3)``.  On this npuir toolchain the vector
``T.vtanh`` is NOT a faithful tanh over the operator's range, in EITHER mode
(``T.tanh`` itself is only a TIR scalar expression):
  * default/Expert -> ``CodeGenTileLangNPUIRAPI::VtanhCodegen``
    (``tilelang-mlir-ascend/src/target/codegen_npuir_api.cc``): truncated
    Maclaurin series ``x - x^3/3 + 2x^5/15 - 17x^7/315``, valid only for
    ``|x| <~ 0.5`` (this is the counter body below);
  * Developer -> ``CodeGenTileLangNPUIRDEV::VtanhCodegen``
    (``codegen_npuir_dev.cc``): exp identity ``(e^{2x}-1)/(e^{2x}+1)``, exact
    for finite ``|x| <~ 44`` but overflowing to NaN at ``x >= ~44`` and
    ``+inf``.
Hardware repro: ``repro/TRAP-vtanh-truncated-taylor.py``; cf. the ``T.verf``
truncation CG-2026-0021.  For a standard-normal gate, |u| already reaches ~10
(and ``+inf`` at the boundary), outside both paths' valid ranges, so the native
op cannot be used.

Adopted items (see DESIGN.md §1.6.1):

  item 1  explicit sigmoid-based GELU-tanh in place of ``T.vtanh``, using the
          identity ``1 + tanh(u) = 2 * sigmoid(2u)`` so that
          ``gelu_tanh(g) = 0.5 g (1 + tanh(u)) = g * sigmoid(2u)`` with
          ``sigmoid(v) = 1 / (1 + exp(-v))`` evaluated explicitly in fp32
          (vcast/vmul/vadd/vexp/vdiv; reciprocal dividend = vbrc ones buffer).
            baseline : 0.5 * g * (1 + torch.tanh(u))  (the source formula, fp32)
            candidate: g * sigmoid(2u)  (fp32 explicit chain, DESIGN §0.6 R2)
            reference: the same expression in fp64
          The identity is exact; only rounding differs.  The explicit sigmoid
          saturates automatically (exp(-v) -> 0 or +inf) and propagates NaN.

  item 2  full fp32-domain chain with a single final cast to the storage dtype
          (the NPU algorithm) in place of the GPU narrow-dtype tail arithmetic.
            baseline : inner in fp32 -> T.tanh -> cast back to dtype, then
                       g * 0.5 * (1 + tanh_val) * value in the storage dtype
            candidate: gate*0.5*(1+tanh(u))*value in fp32, cast once at the end
            reference: same expression in fp64
          The candidate matches torch's fp32-opmath golden.

  counter  the *rejected* npuir ``T.vtanh`` truncated series used as the tanh
          body vs the fp64 reference.  Rejected-baseline control, not an adopted
          candidate; it is the evidence behind the item-1 mechanism (and is not
          required to pass).

RANGE verification asserts the tanh mechanism is faithful over the whole range
the operator needs (gate can be large — ``g^3`` grows fast):

  RANGE  explicit sigmoid tanh ``2*sigmoid(2x)-1`` full-range max abs error vs
         fp64 tanh over a dense grid on [-50, 50];
  RANGE  saturation: tanh(+inf)=+1, tanh(-inf)=-1, tanh(NaN)=NaN;
  RANGE  folded gelu_tanh: full-range max abs error over a dense g grid on
         [-100, 100] (the argument u grows to ~4.5e3), plus saturation
         gelu(+inf)=+inf, gelu(-inf)=NaN, gelu(NaN)=NaN;
  RANGE  the rejected npuir truncated vtanh diverges for |x| > ~0.5.
  BOUNDARY  gelu(-inf)=NaN and gelu(+inf)=+inf, for both the source expression
         and the torch golden, all three dtypes.

Output: one table row per (adopted item x dtype x shape), plus an aggregate,
plus the rejected-vtanh counter table, plus the RANGE / BOUNDARY table.  Exit
code is non-zero if any adopted candidate exceeds its tolerance or a
range/boundary check fails.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

SQ = 0.7978845608028654   # sqrt(2/pi)
C = 0.044715

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
DTYPE_NAME = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}

SHAPES = [(16, 128), (8, 256), (1, 128), (4, 64), (32, 128), (64, 512), (3, 257)]


def _tolerances(dtype: torch.dtype) -> tuple:
    # Mirrors tests/ops/test_gelu_tanh_and_mul.py::_get_tolerances.
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


# ---- tanh / gelu bodies --------------------------------------------------

def _u_f32(g: torch.Tensor) -> torch.Tensor:
    """u = sqrt(2/pi) * (g + 0.044715 g^3), in fp32 (torch emulation of the
    kernel's vmul/vadd chain; g is already fp32)."""
    return SQ * (g + C * g * g * g)


def _sigmoid_2u_f32(u: torch.Tensor) -> torch.Tensor:
    """sigmoid(2u) = 1 / (1 + exp(-2u)) via the explicit npuir chain
    (vmul by -2 -> vexp -> vadd 1 -> vdiv by a vbrc ones buffer)."""
    vi = u * (-2.0)
    ex = torch.exp(vi)
    den = ex + 1.0
    return 1.0 / den


def _cand_tanh_f32(x: torch.Tensor) -> torch.Tensor:
    """Explicit sigmoid tanh: 2*sigmoid(2x) - 1, fp32."""
    return 2.0 * _sigmoid_2u_f32(x) - 1.0


def _cand_gelu_mul_f32(gate, value) -> torch.Tensor:
    """NPU candidate: g * sigmoid(2u) * value, all fp32."""
    g = gate.float()
    v = value.float()
    return g * _sigmoid_2u_f32(_u_f32(g)) * v


def _base_gelu_mul_f32(gate, value) -> torch.Tensor:
    """Source baseline: 0.5 * g * (1 + tanh(u)) * value, fp32 (torch.tanh)."""
    g = gate.float()
    v = value.float()
    return g * 0.5 * (1.0 + torch.tanh(_u_f32(g))) * v


def _gelu_mul_narrow(gate, value, dtype: torch.dtype) -> torch.Tensor:
    """GPU narrow-dtype tail: inner+tanh in fp32 -> cast tanh to dtype, then the
    products in the storage dtype."""
    g32 = gate.float()
    inner = _u_f32(g32)
    tanh_val = torch.tanh(inner).to(dtype)
    half = torch.tensor(0.5, dtype=dtype)
    one = torch.tensor(1.0, dtype=dtype)
    return gate * half * (one + tanh_val) * value


def _npu_vtanh_poly_f32(x: torch.Tensor) -> torch.Tensor:
    """Rejected npuir ``T.vtanh`` body: truncated Maclaurin series
    x - x^3/3 + 2x^5/15 - 17x^7/315 (fp32), from VtanhCodegen."""
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    x7 = x5 * x2
    return x - x3 / 3.0 + 2.0 * x5 / 15.0 - 17.0 * x7 / 315.0


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
    """Assert the explicit tanh mechanism is faithful over the whole needed
    range (large |gate| -> large |u|), plus saturation and NaN propagation."""
    ok = True
    atol32 = _tolerances(torch.float32)[0]

    # (a) explicit sigmoid tanh vs fp64 tanh, dense grid on [-50, 50].
    print("RANGE  explicit sigmoid tanh  2*sigmoid(2x)-1 (fp32) vs fp64 tanh")
    xs = torch.linspace(-50.0, 50.0, 500001, dtype=torch.float64)
    ref = torch.tanh(xs)
    got = _cand_tanh_f32(xs.float())
    max_err = float((got.double() - ref).abs().max())
    tanh_sat = _cand_tanh_f32(
        torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0, -0.0],
                     dtype=torch.float32))
    inf_p = bool(tanh_sat[0] == 1.0)
    inf_n = bool(tanh_sat[1] == -1.0)
    nan_p = bool(torch.isnan(tanh_sat[2]))
    z_ok = bool(tanh_sat[3] == 0.0 and tanh_sat[4] == 0.0)
    sat_ok = inf_p and inf_n and nan_p and z_ok
    ok = ok and sat_ok and (max_err < atol32)
    print(f"  density[-50,50] max abs err (fp32 vs fp64) = {max_err:.3e}"
          f"  (< fp32 atol={atol32:g}: {max_err < atol32})")
    print(f"  saturation: tanh(+inf)={tanh_sat[0].item()} tanh(-inf)={tanh_sat[1].item()} "
          f"tanh(NaN)=NaN:{nan_p} tanh(0)={tanh_sat[3].item()}  "
          f"({'OK' if sat_ok else 'MISMATCH'})")
    print("  note: raw 2*sigmoid(2x)-1 maps -0.0 -> +0.0; the ADOPTED folded form")
    print("        gelu=g*sigmoid(2u) preserves the sign of -0.0 via the g factor")
    print("        (checked in the folded-gelu RANGE row below).")

    # (b) folded gelu over a wide gate grid (u grows fast: g=100 -> u~3.6e4).
    print("RANGE  folded gelu_tanh(g) = g * sigmoid(2u), dense gate grid [-100, 100]")
    g = torch.linspace(-100.0, 100.0, 400001, dtype=torch.float64)
    u = SQ * (g + C * g ** 3)
    ref_g = 0.5 * g * (1.0 + torch.tanh(u))
    got_g = _cand_gelu_mul_f32(g.float(), torch.ones_like(g.float()))
    max_err_g = float((got_g.double() - ref_g).abs().max())
    gsat = _cand_gelu_mul_f32(
        torch.tensor([float("inf"), float("-inf"), float("nan")], dtype=torch.float32),
        torch.ones(3, dtype=torch.float32))
    gp_ok = bool(gsat[0] == float("inf"))
    gn_ok = bool(torch.isnan(gsat[1]))
    gnan_ok = bool(torch.isnan(gsat[2]))
    negz = _cand_gelu_mul_f32(torch.tensor([-0.0], dtype=torch.float32),
                              torch.tensor([1.0], dtype=torch.float32))
    negz_ok = bool(negz[0] == 0.0 and torch.signbit(negz[0]))
    ok = ok and gp_ok and gn_ok and gnan_ok and negz_ok and (max_err_g < atol32)
    print(f"  max abs err (fp32 vs fp64) = {max_err_g:.3e}"
          f"  (< fp32 atol={atol32:g}: {max_err_g < atol32})")
    print(f"  saturation: gelu(+inf)={gsat[0].item()} gelu(-inf)=NaN:{gn_ok} "
          f"gelu(NaN)=NaN:{gnan_ok}  gelu(-0.0) signbit:{torch.signbit(negz[0]).item()}  "
          f"({'OK' if gp_ok and gn_ok and gnan_ok and negz_ok else 'MISMATCH'})")
    return ok


def _check_boundary_semantics() -> bool:
    """Assert DESIGN §0.1 boundary semantics: gelu(-inf)=NaN, gelu(+inf)=+inf."""
    ok = True
    print("BOUNDARY  gelu(-inf)=NaN / gelu(+inf)=+inf (source expr & golden)")
    for dtype in DTYPES:
        gate = torch.tensor([float("-inf"), float("inf")], dtype=dtype)
        value = torch.ones_like(gate)
        src = _base_gelu_mul_f32(gate, value).to(dtype).float()
        gold = (F.gelu(gate.float(), approximate="tanh") * value.float()).to(dtype).float()
        neg_ok = bool(torch.isnan(src[0]) and torch.isnan(gold[0]))
        pos_ok = bool(src[1] == float("inf") and gold[1] == float("inf"))
        ok = ok and neg_ok and pos_ok
        print(f"  {DTYPE_NAME[dtype]:<9} source={src.tolist()} golden={gold.tolist()}"
              f"  {'OK' if neg_ok and pos_ok else 'MISMATCH'}")
    return ok


def _counter_npu_vtanh_vs_ref():
    """Rejected npuir T.vtanh truncated series vs fp64 tanh (control)."""
    print("COUNTER  rejected npuir T.vtanh truncated Maclaurin vs fp64 tanh")
    xs = torch.tensor([0.5, 1.0, 2.0, 3.0, 6.0, 10.0], dtype=torch.float32)
    got = _npu_vtanh_poly_f32(xs)
    ref = torch.tanh(xs.double())
    for x, a, b in zip(xs.tolist(), got.tolist(), ref.tolist()):
        print(f"  vtanh({x:>5}) = {a:>14.4f}   tanh = {b:>10.6f}   |err| = {abs(a - b):.4f}")
    return max(abs(a - b) for a, b in zip(got.tolist(), ref.tolist()))


def run() -> int:
    rows = []
    failed = not _check_range()
    failed = (not _check_boundary_semantics()) or failed
    _counter_npu_vtanh_vs_ref()

    for dtype, (M, N) in itertools.product(DTYPES, SHAPES):
        atol, rtol = _tolerances(dtype)
        gate, value = _make_gate_value(M, N, dtype, seed=M * 131 + N + int(dtype is torch.bfloat16))
        # Independent fp64 reference, rounded once to the storage dtype.
        g64, v64 = gate.double(), value.double()
        u64 = SQ * (g64 + C * g64 ** 3)
        ref64 = (0.5 * g64 * (1.0 + torch.tanh(u64)) * v64).to(dtype)
        # torch fp32-opmath golden (matches tests/ops/test_gelu_tanh_and_mul.py).
        golden = (F.gelu(gate.float(), approximate="tanh") * value.float()).to(dtype)

        cand = _cand_gelu_mul_f32(gate, value).to(dtype)

        # --- item 1: explicit sigmoid-based GELU-tanh vs the source tanh. ---
        base1 = _base_gelu_mul_f32(gate, value).to(dtype)
        u1, v1, n1 = _score(cand, base1, dtype, atol, rtol)
        u1r, v1r, n1r = _score(cand, ref64, dtype, atol, rtol)
        ub1, vb1, _ = _score(base1, ref64, dtype, atol, rtol)
        rows.append(dict(
            item="1 explicit-sigmoid vs source-tanh", dtype=DTYPE_NAME[dtype], hw=f"{M}x{N}",
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
        f"{'item':<34}{'dtype':<10}{'MxN':<9}{'cand_ulp':>10}{'cand_viol':>10}"
        f"{'cand_nan':>9}{'cand_vs64':>11}{'base_ulp':>10}{'base_viol':>10}  verdict"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['item']:<34}{r['dtype']:<10}{r['hw']:<9}{r['cand_ulp']:>10.4f}"
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
    print("  item1 explicit-sigmoid :", _agg("1"))
    print("  item2 fp32-transit     :", _agg("2"))
    n_pass = sum(1 for r in rows if r["verdict"] == "EQUIV_PASS")
    print(f"total rows: {len(rows)}   EQUIV_PASS: {n_pass}   not-pass: {len(rows) - n_pass}")

    print("OVERALL:", "EQUIV_PASS" if not failed else "EQUIV_FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
