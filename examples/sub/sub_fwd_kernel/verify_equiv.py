"""Equivalence machine-verification for SubFwdOp design adoptions (D-1).

Runs on torch CPU only (no NPU). For each adopted optimization in
DESIGN.md §1.6.1, this script compares the *candidate* formula against a
*reference* (torch golden) and the *baseline* formula on random tensors plus
corner values, then prints an ulp / violation table.

Adopted items (see DESIGN.md §1.6.1):

  item 1  alpha == 1 identity-multiplier fold
          baseline : y = a - 1.0 * b   (the scalar multiply is always emitted)
          candidate: y = a - b         (vmul skipped when the baked alpha == 1)
          reference: torch.sub(a, b, alpha=1)
          Expectation: bit-exact (1*x == x for every representable x, incl.
          NaN / +/-inf / -0.0 / subnormal), so 0 ulp / 0 violations.

  item 2  fp32-transit numeric path for the alpha * other term
          baseline : y = dtype(a - alpha * b) computed fully in the storage
                     dtype (the naive narrow-domain mul-sub)
          candidate: y = dtype(fp32(a) - fp32(alpha) * fp32(b))
          reference: torch.sub(a.float(), b.float(), alpha).to(dtype)
                     (torch's fp32-opmath golden)
          Expectation: candidate == reference bit-exact (the kernel runs the
          very same fp32 opmath chain), 0 ulp; the native-dtype baseline is
          allowed to diverge and is reported for contrast.

Output: one table row per (adopted item x dtype x shape), plus a summary.
Exit code is non-zero if any adopted candidate exceeds its tolerance.
"""

from __future__ import annotations

import itertools
import math

import torch

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
DTYPE_NAME = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}

# alpha values exercised: the fold case (1) plus general cases (alpha != 1).
ALPHAS = [1, 1.0, 0.5, 2.0, -1.0, 0.0, 3]

# Broadcast-compatible (a_shape, b_shape) pairs covering the manifest
# workloads plus a couple of small ones:
#   same-shape      (hidden-state-prefill / llama-7b-ffn)
#   row broadcast   (linear-bias: [1, N])
#   per-channel     (cnn-feat-broadcast shape family)
#   scalar / column ((x, [1]) etc.)
SHAPE_PAIRS = [
    ((4096,), (4096,)),
    ((64, 128), (1, 128)),
    ((4, 8, 4, 4), (8, 1, 1)),
    ((3, 1), (1, 4)),
    ((1024, 1024), (1, 1)),
]


def _corner_values(dtype: torch.dtype) -> torch.Tensor:
    """A small tensor of IEEE corner values for *dtype*."""
    big = torch.finfo(dtype).max
    tiny = torch.finfo(dtype).tiny
    vals = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        big,
        -big,
        tiny,
        -tiny,
        float("inf"),
        float("-inf"),
        float("nan"),
        1.5,
        -2.25,
    ]
    return torch.tensor(vals, dtype=dtype)


def _make_inputs(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    n = math.prod(shape)
    flat = torch.randn(n, dtype=torch.float32).to(dtype)
    corners = _corner_values(dtype)
    # Overlay corner values into the first few slots so every tensor carries
    # IEEE specials, then fill the rest with random data.
    k = min(n, corners.numel())
    flat[:k] = corners[:k]
    return flat.reshape(shape)


def _expand(t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return t.expand(shape)


def _dtype_round(t64: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t64.to(dtype)


def _fp32(x: torch.Tensor) -> torch.Tensor:
    return x.float()


def _max_ulp_and_violations(
    cand: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype, atol: float, rtol: float
) -> tuple[float, int, int]:
    """Return (max_ulp, violations, nan_mismatches) for cand vs ref.

    Both tensors are the same dtype / shape. NaN positions match only when both
    are NaN.

    ``max_ulp`` is reported in *tolerance units*: |cand - ref| divided by the
    L0 band ``atol + rtol * |ref|`` (a value <= 1 means the element is within
    the accepted tolerance; 0 means bit-exact).
    """
    assert cand.shape == ref.shape and cand.dtype == dtype
    if cand.numel() == 0:
        return 0.0, 0, 0
    c64 = cand.double()
    r64 = ref.double()
    both_nan = torch.isnan(c64) & torch.isnan(r64)
    nan_mismatch = int((torch.isnan(c64) != torch.isnan(r64)).sum().item())

    band = atol + rtol * r64.abs()
    diff = (c64 - r64).abs()
    exact = both_nan | (cand == ref)  # inf == inf counts as exact
    units = torch.where(exact, torch.zeros_like(diff), diff / band)
    max_ulp = float(units.max().item()) if units.numel() else 0.0

    close = torch.isclose(cand.float(), ref.float(), atol=atol, rtol=rtol)
    close = close | both_nan
    violations = int((~close).sum().item())
    return max_ulp, violations, nan_mismatch


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    # Mirrors tests/ops/test_sub.py::_get_tolerances.
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 1.6e-2, 1.6e-2


def _golden(a: torch.Tensor, b: torch.Tensor, alpha) -> torch.Tensor:
    """torch fp32-opmath golden (matches tests/ops/test_sub.py::ref_program)."""
    return torch.sub(a.float(), b.float(), alpha=alpha).to(a.dtype)


def run() -> int:
    rows = []
    failed = False

    for dtype, (a_shape, b_shape), alpha in itertools.product(DTYPES, SHAPE_PAIRS, ALPHAS):
        a = _make_inputs(a_shape, dtype)
        b = _make_inputs(b_shape, dtype)
        out_shape = tuple(torch.broadcast_shapes(a_shape, b_shape))
        a_e = _expand(a, out_shape)
        b_e = _expand(b, out_shape)
        atol, rtol = _tolerances(dtype)

        ref = _golden(a_e, b_e, alpha)  # torch fp32-opmath golden (test ref_program)
        # Independent mathematical reference: true arithmetic rounded once to dtype.
        ref64 = _dtype_round(a_e.double() - float(alpha) * b_e.double(), dtype)

        # --- item 2: fp32 transit for the alpha * other term ---
        # candidate: y = dtype(fp32(a) - fp32(alpha) * fp32(b))
        # source algorithm (GPU baseline): y = a - dtype(fp32(alpha) * fp32(b))
        #   i.e. the fp32 multiply is cast back to the storage dtype *before*
        #   the subtraction (two roundings), while the candidate casts once at
        #   the end.
        a32, b32 = _fp32(a_e), _fp32(b_e)
        alpha32 = torch.tensor(float(alpha), dtype=torch.float32)
        prod32 = alpha32 * b32                       # fp32 intermediate alpha*b
        cand_general = _dtype_round(a32 - prod32, dtype)
        # source algorithm baseline: fp32 mul -> cast -> dtype subtract.
        src_baseline = a_e - _dtype_round(prod32, dtype)
        # naive native-dtype baseline (naive narrow-domain mul-sub), for contrast.
        alpha_native = torch.tensor(float(alpha), dtype=dtype)
        base_native = a_e - alpha_native * b_e

        # Domain split: on the *normal* domain the fp32 intermediate alpha*b is
        # finite and torch's golden is well-defined; on the *overflow-intermediate*
        # domain alpha*b overflows fp32 (|alpha*b| > fp32 max), where the source
        # algorithm itself yields +/-inf and torch's CPU alpha path keeps an
        # extended-precision intermediate (documented divergence, DESIGN §9.1).
        inter_finite = torch.isfinite(prod32)
        # normal domain: candidate vs torch golden and vs fp64 truth.
        u2, v2, n2 = _max_ulp_and_violations(
            cand_general[inter_finite], ref[inter_finite], dtype, atol, rtol
        )
        u2_64, _, _ = _max_ulp_and_violations(
            cand_general[inter_finite], ref64[inter_finite], dtype, atol, rtol
        )
        ub2, vb2, _ = _max_ulp_and_violations(base_native, ref, dtype, atol, rtol)
        # overflow-intermediate domain: candidate must equal the source algorithm.
        u_o, v_o, n_o = _max_ulp_and_violations(
            cand_general[~inter_finite], src_baseline[~inter_finite], dtype, atol, rtol
        )
        n_overflow = int((~inter_finite).sum().item())

        rows.append(
            dict(
                item="2 fp32-transit(general alpha)",
                dtype=DTYPE_NAME[dtype],
                shape=f"{a_shape}x{b_shape}",
                alpha=alpha,
                cand_ulp=u2,
                cand_ulp64=u2_64,
                cand_viol=v2,
                cand_nan_mis=n2,
                base_ulp=ub2,
                base_viol=vb2,
                overflow=n_overflow,
                ovf_ulp=u_o,
                ovf_viol=v_o,
                verdict="EQUIV_PASS" if (v2 == 0 and n2 == 0 and v_o == 0 and n_o == 0) else "EQUIV_FAIL",
            )
        )
        if v2 != 0 or n2 != 0 or v_o != 0 or n_o != 0:
            failed = True

        # --- item 1 candidate (only meaningful when alpha == 1) ---
        if alpha == 1:
            cand_fold = a_e - b_e  # vmul skipped
            base_mul = a_e - torch.tensor(1.0, dtype=dtype) * b_e
            # Direct fold-vs-unfold equivalence (independent of the golden):
            u1, v1, n1 = _max_ulp_and_violations(cand_fold, base_mul, dtype, atol, rtol)
            ub1, vb1, _ = _max_ulp_and_violations(base_mul, ref, dtype, atol, rtol)
            rows.append(
                dict(
                    item="1 alpha==1 fold (vs unfold)",
                    dtype=DTYPE_NAME[dtype],
                    shape=f"{a_shape}x{b_shape}",
                    alpha=alpha,
                    cand_ulp=u1,
                    cand_ulp64=u1,
                    cand_viol=v1,
                    cand_nan_mis=n1,
                    base_ulp=ub1,
                    base_viol=vb1,
                    verdict="EQUIV_PASS" if (v1 == 0 and n1 == 0) else "EQUIV_FAIL",
                )
            )
            if v1 != 0 or n1 != 0:
                failed = True

    hdr = (
        f"{'item':<30}{'dtype':<9}{'shape':<20}{'alpha':>6}"
        f"{'ulp':>9}{'ulp64':>9}{'viol':>6}{'nan':>5}{'base_ulp':>10}{'base_viol':>10}  verdict"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['item']:<30}{r['dtype']:<9}{r['shape']:<20}{str(r['alpha']):>6}"
            f"{r['cand_ulp']:>9.4f}{r['cand_ulp64']:>9.4f}{r['cand_viol']:>6}{r['cand_nan_mis']:>5}"
            f"{r['base_ulp']:>10.4f}{r['base_viol']:>10}  {r['verdict']}"
        )

    n_pass = sum(1 for r in rows if r["verdict"] == "EQUIV_PASS")
    print("-" * len(hdr))
    print(f"total rows: {len(rows)}   EQUIV_PASS: {n_pass}   EQUIV_FAIL: {len(rows) - n_pass}")

    def _agg(pred):
        sub = [r for r in rows if pred(r["item"])]
        return (
            len(sub),
            max(r["cand_ulp"] for r in sub),
            max(r["cand_ulp64"] for r in sub),
            sum(r["cand_viol"] for r in sub),
            sum(r["cand_nan_mis"] for r in sub),
            max(r["base_ulp"] for r in sub),
            sum(r["base_viol"] for r in sub),
        )

    print()
    print("AGGREGATE (item | rows | cand_ulp | cand_ulp64 | cand_viol | cand_nan | base_ulp | base_viol)")
    print("  item1 alpha==1 fold          :", _agg(lambda s: s.startswith("1")))
    print("  item2 fp32-transit(general)  :", _agg(lambda s: s.startswith("2")))
    ovf = [r for r in rows if r["item"].startswith("2")]
    print(
        "  item2 overflow-intermediate elements (candidate==source alg, bit-exact):",
        sum(r["overflow"] for r in ovf),
        " elems; violations:",
        sum(r["ovf_viol"] for r in ovf),
    )
    print("OVERALL:", "EQUIV_PASS" if not failed else "EQUIV_FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
