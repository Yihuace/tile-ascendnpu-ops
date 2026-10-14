"""verify_equiv.py — IsfiniteFwdOp §1.6.1 adopted-item equivalence verification.

``torch.isfinite`` is an EXACT predicate (bool output, no floating tolerance), so
the baseline-vs-optimized comparison is done in exact boolean arithmetic on the
torch CPU — the "fp64 reference" degenerates to bit-exact equality for this
predicate op.  Every adopted optimization must reproduce the baseline predicate
bit for bit (0 mismatches).

Baseline (source GPU ``IsfiniteFwdKernel.op_func``):
    y = T.isfinite(T.cast(x, "float32"))    # widen to fp32, then IEEE isfinite

Adopted items (2), both computed with ORDERED comparisons only (this toolchain's
``T.vcmp "ne"`` is ordered — no unordered mode; CG-2026-0024):

  1. Ordered magnitude bound (eliminates the special function)
       baseline : torch.isfinite(x)
       optimized: (x > -inf) & (x < +inf)
     Identity: for IEEE-754 binary floats, a value is finite iff it is strictly
     greater than -inf and strictly less than +inf.  Ordered ``gt``/``lt`` are
     FALSE when either operand is NaN, and ``+inf < +inf`` / ``-inf > -inf`` are
     FALSE, so NaN / +inf / -inf all answer "not finite" — the correct IEEE
     answer.  This is the vabs-free realization of the canonical
     ``isfinite(x) = (|x| < +inf)`` mechanism: the two-sided ordered compare is
     algebraically identical to ``|x| < +inf`` but needs no ``T.vabs`` (which is
     bf16-unsupported — docs .../数学操作/T.vabs.md §2.2.1 bf16 = ×), so it works
     uniformly on fp16 / bf16 / fp32.  ``(x > -inf) & (x < +inf)`` is exactly
     ``|x| < +inf`` because ``x > -inf`` ⟺ ``x < +inf`` holds for every
     non-|inf| magnitude and both are false together for NaN.

  2. Native-dtype predicate (eliminate the fp32 widening cast)
       baseline : torch.isfinite(x.to(float32))   # widen then isfinite
       optimized: (x > -inf) & (x < +inf)         # computed in the input dtype
     Equivalence: IEEE infinity / NaN / finiteness are representation-invariant
     under widening (a +/-inf or NaN in fp16/bf16 stays the same in fp32 and vice
     versa), so widening before the predicate does not change the answer.  Checked
     per input dtype (fp16 / bf16 / fp32).

Discrimination sections (pin the semantic choices that the design rejects/keeps):

  A. Alternative accepted form ``|x| < +inf`` (the canonical shortest mechanism)
     must agree exactly with ``isfinite`` -> mismatches == 0.  (Host ``abs`` +
     ``lt`` is exactly what ``vcmp(|x|, +inf, "lt")`` computes on NPU where
     ``T.vabs`` exists.)
  B. Rejected ``(x * x) < +inf`` shortcut: for a finite value whose square
     overflows the dtype (e.g. fp16 1000 -> 1e6 -> inf; fp32/bf16 1e20 -> inf),
     ``x * x`` becomes +inf and the test answers "not finite" -> misclassification.
     mismatches > 0 -> DISCARD.

The on-device ordered-comparison evidence is
``examples/isnan/isnan_fwd_kernel/repro/CONST-vcmp-ne-ordered.py`` (raw vcmp mask
readout, three dtypes) and capability gap CG-2026-0024: this toolchain's
``npuir_cmp`` has no ``unordered`` mode and lowers ``"ne"`` as an ORDERED
not-equal, so ordered ``eq`` / ``le`` / ``ge`` / ``lt`` / ``gt`` are the only
usable comparison semantics (a NaN operand makes all of them false).  This CPU
harness pins the same ordered semantics in the exact boolean domain.

Method: random tensors + corner values (NaN / +-inf / +-0 / subnormal / min / max
/ overflow-prone finite) per dtype; baseline and optimized evaluated in torch CPU;
exact boolean comparison.  Any mismatching adopted item must not be adopted.
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
            1000.0,                     # fp16-finite, 1000^2 = 1e6 overflows fp16
            1.1754943508222875e-38,     # fp32 smallest normal
            1.401298464324817e-45,      # fp32 smallest subnormal
            3.4028234663852886e38,      # fp32 max
            -3.4028234663852886e38,
            3.0e38,                     # bf16-finite, 3e38^2 overflows bf16
            1.0e20,                     # fp32/bf16-finite, 1e20^2 overflows both
            -1.0e20,
            6.103515625e-05,            # fp16 smallest normal
            5.960464477539063e-08,      # fp16 smallest subnormal
            65504.0,                    # fp16 max
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


def _isfinite_two_sided(x: torch.Tensor) -> torch.Tensor:
    """Adopted candidate: isfinite = (x > -inf) & (x < +inf) (ordered gt/lt)."""
    return (x > -_INF) & (x < _INF)


def _isfinite_abs_lt(x: torch.Tensor) -> torch.Tensor:
    """Canonical shortest mechanism: |x| < +inf (ordered lt on magnitude)."""
    return torch.abs(x) < _INF


def _isfinite_square_lt(x: torch.Tensor) -> torch.Tensor:
    """Rejected shortcut: (x * x) < +inf (overflows for large finite values)."""
    return (x * x) < _INF


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _item1_two_sided(x: torch.Tensor, dtype: torch.dtype) -> dict:
    """Item 1: isfinite(x) vs (x > -inf) & (x < +inf), native dtype."""
    baseline = torch.isfinite(x)
    candidate = _isfinite_two_sided(x)
    mism = int((baseline != candidate).sum().item())
    return {
        "item": f"isfinite(x) -> (x>-inf)&(x<+inf)  [ordered, {_dtype_name(dtype)}]",
        "dtype": _dtype_name(dtype),
        "n": int(x.numel()),
        "mismatches": mism,
        "verdict": "EQUIV_PASS" if mism == 0 else "EQUIV_FAIL",
    }


def _item2_native_dtype(x: torch.Tensor, dtype: torch.dtype) -> dict:
    """Item 2: isfinite(widen(x)) vs two-sided compare in native dtype."""
    baseline = torch.isfinite(x.to(torch.float32))   # source: cast to fp32, then isfinite
    candidate = _isfinite_two_sided(x)               # optimized: native-dtype ordered compare
    mism = int((baseline != candidate).sum().item())
    return {
        "item": f"isfinite(cast(x,f32)) -> (x>-inf)&(x<+inf)  [native {_dtype_name(dtype)}]",
        "dtype": _dtype_name(dtype),
        "n": int(x.numel()),
        "mismatches": mism,
        "verdict": "EQUIV_PASS" if mism == 0 else "EQUIV_FAIL",
    }


def _discr_abs_lt(x: torch.Tensor, dtype: torch.dtype) -> dict:
    """Alternative accepted form |x| < +inf agrees with isfinite."""
    ref = torch.isfinite(x)
    mism = int((_isfinite_abs_lt(x) != ref).sum().item())
    return {
        "item": f"canonical |x| < +inf vs isfinite [{_dtype_name(dtype)}]",
        "dtype": _dtype_name(dtype),
        "mismatches": mism,
        "verdict": "DISCRIM_OK" if mism == 0 else "DISCRIM_FAIL",
    }


def _discr_square_lt(x: torch.Tensor, dtype: torch.dtype) -> dict:
    """Rejected (x*x) < +inf differs from isfinite (overflow at large finite)."""
    ref = torch.isfinite(x)
    mism = int((_isfinite_square_lt(x) != ref).sum().item())
    return {
        "item": f"rejected (x*x)<+inf vs isfinite [{_dtype_name(dtype)}]",
        "dtype": _dtype_name(dtype),
        "mismatches": mism,
        "verdict": "DISCRIM_OK" if mism > 0 else "DISCRIM_FAIL",
    }


def main() -> int:
    print("=== IsfiniteFwdOp §1.6.1 equivalence verification (exact bool, torch CPU) ===")
    print("(mechanism: ordered |x| < +inf / (x>-inf)&(x<+inf); NO vcmp 'ne', NO scalar T.isfinite)\n")

    inputs = {
        d: _make_input(d, seed=400 + i) for i, d in enumerate(_FLOAT_DTYPES)
    }

    r1 = [_item1_two_sided(inputs[d], d) for d in _FLOAT_DTYPES]
    r2 = [_item2_native_dtype(inputs[d], d) for d in _FLOAT_DTYPES]
    abslt = [_discr_abs_lt(inputs[d], d) for d in _FLOAT_DTYPES]
    sq = [_discr_square_lt(inputs[d], d) for d in _FLOAT_DTYPES]

    print(f"{'item':<66} | {'n':>6} | {'mismatches':>10} | 结论")
    print("-" * 102)
    for r in r1 + r2:
        print(f"{r['item']:<66} | {r['n']:>6} | {r['mismatches']:>10} | {r['verdict']}")

    print("\n采纳项结果表：")
    print(f"{'采纳项':<60} | {'mismatches':>10} | 结论")
    print("-" * 86)
    m1 = sum(r["mismatches"] for r in r1)
    v1 = "EQUIV_PASS" if all(r["verdict"] == "EQUIV_PASS" for r in r1) else "EQUIV_FAIL"
    print(f"{'1. isfinite(x) -> (x>-inf)&(x<+inf)  [fp16/bf16/fp32]':<60} | {m1:>10} | {v1}")
    m2 = sum(r["mismatches"] for r in r2)
    v2 = "EQUIV_PASS" if all(r["verdict"] == "EQUIV_PASS" for r in r2) else "EQUIV_FAIL"
    print(f"{'2. isfinite(cast(x,f32)) -> (x>-inf)&(x<+inf)  [fp16/bf16/fp32]':<60} | {m2:>10} | {v2}")

    print("\n判别段 A（等价备选式）：|x| < +inf（vcmp 'lt' on |x|）与 isfinite 等价：")
    print(f"{'dtype':<10} | {'mismatches':>12} | 结论")
    print("-" * 46)
    for d in abslt:
        print(f"{d['dtype']:<10} | {d['mismatches']:>12} | {d['verdict']}")

    print("\n判别段 B（弃选式）：(x * x) < +inf 与 isfinite 不等价（大有限值平方溢出误判）：")
    print(f"{'dtype':<10} | {'mismatches':>12} | 结论")
    print("-" * 46)
    for d in sq:
        print(f"{d['dtype']:<10} | {d['mismatches']:>12} | {d['verdict']}")

    adopted_pass = all(r["verdict"] == "EQUIV_PASS" for r in r1 + r2)
    discr_pass = all(d["verdict"] == "DISCRIM_OK" for d in abslt + sq)
    all_pass = adopted_pass and discr_pass
    print("\n" + "=" * 78)
    if all_pass:
        print("总体结论: 全部采纳项 EQUIV_PASS；判别段确认 |x|<+inf 等价、"
              "(x*x)<+inf 被弃选 — 可采纳")
    else:
        print("总体结论: 存在失败项（采纳项或判别段）— 禁止采纳")
    print("=" * 78)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
