"""verify_equiv.py — IsnanFwdOp §1.6.1 adopted-item equivalence verification (revision v1).

``torch.isnan`` is an EXACT predicate (bool output, no floating tolerance), so the
baseline-vs-optimized comparison is done in exact boolean arithmetic on the torch
CPU -- the "fp64 reference" degenerates to bit-exact equality for this predicate
op.  Every adopted optimization must reproduce the baseline predicate bit for bit
(0 mismatches).

Baseline (source GPU ``IsnanFwdKernel.op_func``):
    y = T.isnan(T.cast(x, "float32"))     # widen to fp32, then IEEE isnan

Adopted items (2), all computed on the *target* Ascend semantics:

  1. Ordered self-compare complement NaN predicate
       baseline : torch.isnan(x)
       optimized: ~(x <= x)              # = NOT(x <= x), ordered "le" self-compare
     Identity: for IEEE-754 binary floats, ``x <= x`` is TRUE for every non-NaN
     value (including +-inf, +-0 and subnormals) and FALSE only for NaN, so
     ``isnan(x) == NOT(x <= x)``.
     NOTE (v0 -> v1): the classic ``isnan(x) == (x != x)`` identity holds only for
     an UNORDERED ``!=``.  On the target Ascend npuir toolchain ``T.vcmp "ne"`` is
     lowered as an ORDERED not-equal (``ne == lt | gt``), so a self-compare
     ``x != x`` is FALSE for NaN and does NOT detect it (measured in
     ``repro/CONST-vcmp-ne-ordered.py``).  The v0 design relied on the unordered
     assumption and failed; v1 uses the ordered ``le`` complement.

  2. Native-dtype predicate (eliminate the fp32 widening cast)
       baseline : torch.isnan(x.to(float32))   # widen then isnan
       optimized: ~(x <= x)                    # computed in the input dtype
     Equivalence: IEEE NaN is representation-invariant under widening (a NaN in
     fp16/bf16 stays NaN in fp32 and vice versa), so widening before the predicate
     does not change the answer; the ordered ``x <= x`` result is likewise
     representation-invariant.  Checked per input dtype (fp16 / bf16 / fp32).

Discrimination section (pins the *semantic* distinction that v0 got wrong):
    ordered_ne(x) = (x < x) | (x > x)     # the toolchain's ordered "ne" lowering
    asserts:  ordered_ne != isnan         (mismatches > 0, at the NaN elements)
              ~(x <= x)     == isnan      (mismatches == 0)
The authoritative on-device measurement of the ordered ``ne`` behavior and the
``NOT(x<=x)`` recipe is ``repro/CONST-vcmp-ne-ordered.py`` (three dtypes, raw
vcmp mask readout).  This CPU harness pins the same distinction in the exact
boolean domain so a wrong comparison-mode choice is caught at design time.

Method: random tensors + corner values (NaN / +-inf / +-0 / subnormal / min / max)
per dtype; baseline and optimized evaluated in torch CPU; exact boolean
comparison.  Any mismatching adopted item must not be adopted.
"""

import torch

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


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
    # A true zero and a subnormal are already reachable from the corners below.
    corner = _corner_values()
    x64 = torch.cat([rand, corner], dim=0)
    return x64.to(dtype)


def _ordered_le_isnan(x: torch.Tensor) -> torch.Tensor:
    """Target Ascend predicate: isnan = NOT (x <= x) (ordered 'le' self-compare)."""
    return ~(x <= x)


def _ordered_ne_isnan(x: torch.Tensor) -> torch.Tensor:
    """Emulate the toolchain's ORDERED 'ne' lowering: ne(x,x) = lt | gt (== false)."""
    return (x < x) | (x > x)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _item1_self_compare(x: torch.Tensor, dtype: torch.dtype) -> dict:
    """Item 1: isnan(x) vs NOT(x <= x), native dtype."""
    baseline = torch.isnan(x)
    candidate = _ordered_le_isnan(x)
    mism = int((baseline != candidate).sum().item())
    return {
        "item": f"isnan(x) -> NOT(x <= x)  [ordered 'le' self-compare, {_dtype_name(dtype)}]",
        "dtype": _dtype_name(dtype),
        "n": int(x.numel()),
        "mismatches": mism,
        "verdict": "EQUIV_PASS" if mism == 0 else "EQUIV_FAIL",
    }


def _item2_native_dtype(x: torch.Tensor, dtype: torch.dtype) -> dict:
    """Item 2: isnan(widen(x)) vs NOT(x <= x) computed in the native input dtype."""
    baseline = torch.isnan(x.to(torch.float32))   # source: cast to fp32, then isnan
    candidate = _ordered_le_isnan(x)              # optimized: native-dtype ordered le
    mism = int((baseline != candidate).sum().item())
    return {
        "item": f"isnan(cast(x,f32)) -> NOT(x <= x)  [native {_dtype_name(dtype)}]",
        "dtype": _dtype_name(dtype),
        "n": int(x.numel()),
        "mismatches": mism,
        "verdict": "EQUIV_PASS" if mism == 0 else "EQUIV_FAIL",
    }


def _discrimination(x: torch.Tensor, dtype: torch.dtype) -> dict:
    """Ordered 'ne' self-compare is NOT isnan; NOT(x<=x) IS isnan.

    Returns the ordered-ne mismatch count (>0 expected: it misses every NaN) and
    the le-complement mismatch count (0 expected).  This is the design-time guard
    against the v0 mistake (relying on an unordered 'ne').
    """
    ref = torch.isnan(x)
    ne_mism = int((_ordered_ne_isnan(x) != ref).sum().item())
    le_mism = int((_ordered_le_isnan(x) != ref).sum().item())
    nan_count = int(ref.sum().item())
    ok = (ne_mism > 0) and (le_mism == 0)
    return {
        "item": f"ordered-ne vs le-complement discrimination [{_dtype_name(dtype)}]",
        "dtype": _dtype_name(dtype),
        "nan_count": nan_count,
        "ordered_ne_mismatches": ne_mism,
        "le_complement_mismatches": le_mism,
        "verdict": "DISCRIM_OK" if ok else "DISCRIM_FAIL",
    }


def main() -> int:
    print("=== IsnanFwdOp §1.6.1 equivalence verification (exact bool, torch CPU) ===")
    print("(v1: ordered 'le' self-compare complement; v0's unordered 'ne' is overturned)\n")

    inputs = {
        d: _make_input(d, seed=200 + i) for i, d in enumerate(_FLOAT_DTYPES)
    }

    r1 = [_item1_self_compare(inputs[d], d) for d in _FLOAT_DTYPES]
    r2 = [_item2_native_dtype(inputs[d], d) for d in _FLOAT_DTYPES]
    disc = [_discrimination(inputs[d], d) for d in _FLOAT_DTYPES]

    print(f"{'item':<64} | {'n':>6} | {'mismatches':>10} | 结论")
    print("-" * 100)
    for r in r1 + r2:
        print(f"{r['item']:<64} | {r['n']:>6} | {r['mismatches']:>10} | {r['verdict']}")

    print("\n采纳项结果表：")
    print(f"{'采纳项':<58} | {'mismatches':>10} | 结论")
    print("-" * 84)
    m1 = sum(r["mismatches"] for r in r1)
    v1 = "EQUIV_PASS" if all(r["verdict"] == "EQUIV_PASS" for r in r1) else "EQUIV_FAIL"
    print(f"{'1. isnan(x) -> NOT(x <= x)  [fp16/bf16/fp32]':<58} | {m1:>10} | {v1}")
    m2 = sum(r["mismatches"] for r in r2)
    v2 = "EQUIV_PASS" if all(r["verdict"] == "EQUIV_PASS" for r in r2) else "EQUIV_FAIL"
    print(f"{'2. isnan(cast(x,f32)) -> NOT(x <= x)  [fp16/bf16/fp32]':<58} | {m2:>10} | {v2}")

    print("\n判别段（非采纳项）——有序 'ne' 自比较 ≠ isnan，'le' 取反 == isnan：")
    print(f"{'dtype':<10} | {'NaN 数':>6} | {'ordered-ne 失配':>14} | {'le-取反 失配':>12} | 结论")
    print("-" * 76)
    for d in disc:
        print(
            f"{d['dtype']:<10} | {d['nan_count']:>6} | {d['ordered_ne_mismatches']:>14} "
            f"| {d['le_complement_mismatches']:>12} | {d['verdict']}"
        )

    adopted_pass = all(r["verdict"] == "EQUIV_PASS" for r in r1 + r2)
    discrim_pass = all(d["verdict"] == "DISCRIM_OK" for d in disc)
    all_pass = adopted_pass and discrim_pass
    print("\n" + "=" * 78)
    if all_pass:
        print("总体结论: 全部采纳项 EQUIV_PASS；判别段确认 ordered-ne ≠ isnan 且 le-取反 == isnan — 可采纳")
    else:
        print("总体结论: 存在失败项（采纳项或判别段）— 禁止采纳")
    print("=" * 78)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
