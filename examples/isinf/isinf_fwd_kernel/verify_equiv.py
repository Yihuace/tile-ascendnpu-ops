"""verify_equiv.py — IsinfFwdOp §1.6.1 adopted-item equivalence verification.

``torch.isinf`` is an EXACT predicate (bool output, no floating tolerance), so the
baseline-vs-optimized comparison is done in exact boolean arithmetic on the torch
CPU -- the "fp64 reference" degenerates to bit-exact equality for this predicate
op.  Every adopted optimization must reproduce the baseline predicate bit for bit
(0 mismatches).

Baseline (source GPU ``IsinfFwdKernel.op_func``):
    y = T.isinf(T.cast(x, "float32"))    # widen to fp32, then IEEE isinf

Adopted items (2), both computed with ORDERED comparisons only (this toolchain's
``T.vcmp "ne"`` is ordered -- no unordered mode; CG-2026-0024):

  1. Ordered equality against the two infinities (eliminates the special function)
       baseline : torch.isinf(x)
       optimized: (x == +inf) | (x == -inf)
     Identity: for IEEE-754 binary floats, an ``eq`` comparison against an
     infinity is TRUE only for that exact infinity; ordered ``eq`` is FALSE for
     NaN, so NaN answers "not infinite" -- the correct IEEE answer.  ``isinf`` is
     therefore exactly ``(x == +inf) | (x == -inf)``.  This is the shortest
     correct ordered-compare form; the alternative ``NOT(isfinite) & NOT(isnan)``
     form (``isfinite = |x| < +inf``, ``isnan = ~(x <= x)``) computes the same
     result but needs more vector ops (vabs + two compares + complement + AND) and
     the ``T.vabs`` bf16 gap (docs .../数学操作/T.vabs.md §2.2.1 bf16 = ×).

  2. Native-dtype predicate (eliminate the fp32 widening cast)
       baseline : torch.isinf(x.to(float32))   # widen then isinf
       optimized: (x == +inf) | (x == -inf)    # computed in the input dtype
     Equivalence: IEEE infinity / NaN are representation-invariant under widening
     (a +/-inf or NaN in fp16/bf16 stays the same in fp32 and vice versa), so
     widening before the predicate does not change the answer.  Checked per input
     dtype (fp16 / bf16 / fp32).

Discrimination sections (pin the semantic choices that the design rejects):

  A. Rejected ``x - x != 0`` shortcut: for NaN, ``x - x`` is NaN and ``NaN != 0``
     is TRUE under IEEE (unordered ne on the host), so that variant misclassifies
     NaN as infinite.  mismatches > 0 -> DISCARD.
  B. Alternative accepted form ``NOT(isfinite) & NOT(isnan)`` must agree exactly
     with ``(x == +inf) | (x == -inf)`` (both are among the allowed ordered
     mechanisms) -> mismatches == 0.

The on-device ordered-comparison evidence is
``examples/isnan/isnan_fwd_kernel/repro/CONST-vcmp-ne-ordered.py`` (raw vcmp mask
readout, three dtypes) and capability gap CG-2026-0024: this toolchain's
``npuir_cmp`` has no ``unordered`` mode and lowers ``"ne"`` as an ORDERED
not-equal, so ordered ``eq`` / ``le`` / ``ge`` / ``lt`` / ``gt`` are the only
usable comparison semantics (a NaN operand makes all of them false).  This CPU
harness pins the same ordered semantics in the exact boolean domain.

Method: random tensors + corner values (NaN / +-inf / +-0 / subnormal / min / max)
per dtype; baseline and optimized evaluated in torch CPU; exact boolean
comparison.  Any mismatching adopted item must not be adopted.
"""

import torch

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_INF = float("inf")


def _corner_values() -> torch.Tensor:
    """Special + boundary float64 values spanning the predicate's domains."""
    return torch.tensor(
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            0.0,
            -0.0,
            1.0,
            -1.0,
            1.1754943508222875e-38,   # fp32 smallest normal
            1.401298464324817e-45,    # fp32 smallest subnormal
            3.4028234663852886e38,    # fp32 max
            -3.4028234663852886e38,
            6.103515625e-05,          # fp16 smallest normal
            5.960464477539063e-08,    # fp16 smallest subnormal
            65504.0,                  # fp16 max
            -65504.0,
        ],
        dtype=torch.float64,
    )


def _make_input(dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Random + corner input of *dtype* (CPU)."""
    g = torch.Generator().manual_seed(seed)
    rand = (torch.randn(1 << 12, generator=g, dtype=torch.float64) * 8.0)
    # Inject a healthy NaN / inf fraction so the true branch is exercised.
    mask = torch.rand(rand.numel(), generator=g, dtype=torch.float64)
    rand[mask < 0.10] = float("nan")
    rand[(mask >= 0.10) & (mask < 0.15)] = float("inf")
    rand[(mask >= 0.15) & (mask < 0.20)] = float("-inf")
    corner = _corner_values()
    x64 = torch.cat([rand, corner], dim=0)
    return x64.to(dtype)


def _isinf_ordered_eq(x: torch.Tensor) -> torch.Tensor:
    """Adopted candidate: isinf = (x == +inf) | (x == -inf) (ordered eq)."""
    return (x == _INF) | (x == -_INF)


def _isinf_notfinite(x: torch.Tensor) -> torch.Tensor:
    """Alternative accepted form: NOT(isfinite) & NOT(isnan) (ordered compares)."""
    isfinite = torch.abs(x) < _INF          # |x| < +inf  (ordered "lt")
    isnan = ~(x <= x)                       # ordered "le" self-compare complement
    return (~isfinite) & (~isnan)


def _isinf_sub_variant(x: torch.Tensor) -> torch.Tensor:
    """Rejected shortcut: (x - x) != 0 (misclassifies NaN as infinite)."""
    return (x - x) != 0


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _item1_ordered_eq(x: torch.Tensor, dtype: torch.dtype) -> dict:
    """Item 1: isinf(x) vs (x == +inf) | (x == -inf), native dtype."""
    baseline = torch.isinf(x)
    candidate = _isinf_ordered_eq(x)
    mism = int((baseline != candidate).sum().item())
    return {
        "item": f"isinf(x) -> (x==+inf)|(x==-inf)  [ordered eq, {_dtype_name(dtype)}]",
        "dtype": _dtype_name(dtype),
        "n": int(x.numel()),
        "mismatches": mism,
        "verdict": "EQUIV_PASS" if mism == 0 else "EQUIV_FAIL",
    }


def _item2_native_dtype(x: torch.Tensor, dtype: torch.dtype) -> dict:
    """Item 2: isinf(widen(x)) vs (x==+inf)|(x==-inf) computed in native dtype."""
    baseline = torch.isinf(x.to(torch.float32))   # source: cast to fp32, then isinf
    candidate = _isinf_ordered_eq(x)              # optimized: native-dtype ordered eq
    mism = int((baseline != candidate).sum().item())
    return {
        "item": f"isinf(cast(x,f32)) -> (x==+inf)|(x==-inf)  [native {_dtype_name(dtype)}]",
        "dtype": _dtype_name(dtype),
        "n": int(x.numel()),
        "mismatches": mism,
        "verdict": "EQUIV_PASS" if mism == 0 else "EQUIV_FAIL",
    }


def _discr_sub(x: torch.Tensor, dtype: torch.dtype) -> dict:
    """Rejected (x - x) != 0 differs from isinf (fails at NaN)."""
    ref = torch.isinf(x)
    mism = int((_isinf_sub_variant(x) != ref).sum().item())
    nan_count = int(torch.isnan(x).sum().item())
    return {
        "item": f"rejected (x-x)!=0 vs isinf [{_dtype_name(dtype)}]",
        "dtype": _dtype_name(dtype),
        "nan_count": nan_count,
        "mismatches": mism,
        "verdict": "DISCRIM_OK" if mism > 0 else "DISCRIM_FAIL",
    }


def _discr_alt(x: torch.Tensor, dtype: torch.dtype) -> dict:
    """Alternative accepted form NOT(isfinite)&NOT(isnan) agrees with isinf."""
    ref = torch.isinf(x)
    mism = int((_isinf_notfinite(x) != ref).sum().item())
    return {
        "item": f"alt NOT(isfinite)&NOT(isnan) vs isinf [{_dtype_name(dtype)}]",
        "dtype": _dtype_name(dtype),
        "mismatches": mism,
        "verdict": "DISCRIM_OK" if mism == 0 else "DISCRIM_FAIL",
    }


def main() -> int:
    print("=== IsinfFwdOp §1.6.1 equivalence verification (exact bool, torch CPU) ===")
    print("(mechanism: ordered eq against +/-inf; NO vcmp 'ne', NO scalar T.isinf)\n")

    inputs = {
        d: _make_input(d, seed=300 + i) for i, d in enumerate(_FLOAT_DTYPES)
    }

    r1 = [_item1_ordered_eq(inputs[d], d) for d in _FLOAT_DTYPES]
    r2 = [_item2_native_dtype(inputs[d], d) for d in _FLOAT_DTYPES]
    sub = [_discr_sub(inputs[d], d) for d in _FLOAT_DTYPES]
    alt = [_discr_alt(inputs[d], d) for d in _FLOAT_DTYPES]

    print(f"{'item':<64} | {'n':>6} | {'mismatches':>10} | 结论")
    print("-" * 100)
    for r in r1 + r2:
        print(f"{r['item']:<64} | {r['n']:>6} | {r['mismatches']:>10} | {r['verdict']}")

    print("\n采纳项结果表：")
    print(f"{'采纳项':<58} | {'mismatches':>10} | 结论")
    print("-" * 84)
    m1 = sum(r["mismatches"] for r in r1)
    v1 = "EQUIV_PASS" if all(r["verdict"] == "EQUIV_PASS" for r in r1) else "EQUIV_FAIL"
    print(f"{'1. isinf(x) -> (x==+inf)|(x==-inf)  [fp16/bf16/fp32]':<58} | {m1:>10} | {v1}")
    m2 = sum(r["mismatches"] for r in r2)
    v2 = "EQUIV_PASS" if all(r["verdict"] == "EQUIV_PASS" for r in r2) else "EQUIV_FAIL"
    print(f"{'2. isinf(cast(x,f32)) -> (x==+inf)|(x==-inf)  [fp16/bf16/fp32]':<58} | {m2:>10} | {v2}")

    print("\n判别段 A（弃选式）：(x - x) != 0 与 isinf 不等价（NaN 处误判为 inf）：")
    print(f"{'dtype':<10} | {'NaN 数':>6} | {'mismatches':>12} | 结论")
    print("-" * 60)
    for d in sub:
        print(f"{d['dtype']:<10} | {d['nan_count']:>6} | {d['mismatches']:>12} | {d['verdict']}")

    print("\n判别段 B（备选采纳式）：NOT(isfinite)&NOT(isnan) 与 isinf 等价：")
    print(f"{'dtype':<10} | {'mismatches':>12} | 结论")
    print("-" * 46)
    for d in alt:
        print(f"{d['dtype']:<10} | {d['mismatches']:>12} | {d['verdict']}")

    adopted_pass = all(r["verdict"] == "EQUIV_PASS" for r in r1 + r2)
    discr_pass = all(d["verdict"] == "DISCRIM_OK" for d in sub + alt)
    all_pass = adopted_pass and discr_pass
    print("\n" + "=" * 78)
    if all_pass:
        print("总体结论: 全部采纳项 EQUIV_PASS；判别段确认 (x-x)!=0 被弃选、"
              "NOT(isfinite)&NOT(isnan) 等价 — 可采纳")
    else:
        print("总体结论: 存在失败项（采纳项或判别段）— 禁止采纳")
    print("=" * 78)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
