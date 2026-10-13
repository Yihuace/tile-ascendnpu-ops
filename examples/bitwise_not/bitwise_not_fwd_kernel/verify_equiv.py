"""verify_equiv.py — BitwiseNotFwdOp §1.6.1 adopted-item equivalence verification.

`torch.bitwise_not` is an EXACT integer/boolean op (no floating tolerance), so the
baseline-vs-optimized comparison is done in exact integer/boolean arithmetic on the
torch CPU — "fp64 reference" degenerates to bit-exact equality for this integral
op.  Every adopted optimization must reproduce the baseline output bit for bit.

Adopted items (2):

  1. Integer complement via the dedicated unary NOT primitive:
       baseline (source GPU BitwiseNotFwdKernel.op_func):
           signed ints : y = bitwise_xor(x, -1)     ; uint8 : y = bitwise_xor(x, 255)
       optimized (NPU npuir):
           y = T.vnot(x)                            (all integer dtypes incl. uint8)
     Identity: XOR with all-ones == bitwise complement (~x) == vnot(x), for every
     fixed-width integer type.  Bit-exact (no rounding anywhere).

  2. Boolean logical-NOT computed directly in the byte-storage domain:
       baseline (source GPU bool path, byte storage):
           y = (x_byte == 0)                       # compare -> mask -> store byte
       optimized (NPU npuir):
           y = x_byte ^ 1                          # single vxor on the byte-stored bool
     Identity: for byte-stored bool (bytes in {0, 1}), (x == 0) == (x ^ 1).
     Bit-exact.

Method: random tensors + corner values (min / max / 0 / all-ones) per dtype; both
formulas evaluated with torch CPU integer semantics; exact elementwise comparison.
Any mismatching item is EQUIV_FAIL and must not be adopted.
"""

import torch


# ---------------------------------------------------------------------------
# Item 1: integer complement — xor(all-ones) vs bitwise_not
# ---------------------------------------------------------------------------
_INT_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def _all_ones_value(dtype: torch.dtype) -> int:
    """All-ones constant of *dtype* as a Python int (255 for uint8, -1 otherwise)."""
    if dtype == torch.uint8:
        return 255
    return -1


def _corner_int_values(dtype: torch.dtype) -> torch.Tensor:
    info = torch.iinfo(dtype)
    vals = [0, 1, info.min, info.max, _all_ones_value(dtype) & info.max]
    return torch.tensor(vals, dtype=dtype)


def _item1_one_dtype(dtype: torch.dtype, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    if dtype == torch.uint8:
        rand = torch.randint(0, 256, (1 << 12,), generator=g, dtype=torch.int64).to(dtype)
    else:
        info = torch.iinfo(dtype)
        rand = torch.randint(
            info.min, info.max, (1 << 12,), generator=g, dtype=torch.int64
        ).to(dtype)

    x = torch.cat([rand, _corner_int_values(dtype)], dim=0)

    # baseline: xor with the all-ones constant of the dtype (source GPU form)
    ones = torch.full_like(x, _all_ones_value(dtype))
    baseline = torch.bitwise_xor(x, ones)

    # optimized: dedicated complement primitive (NPU T.vnot)
    candidate = torch.bitwise_not(x)

    mism = int((baseline != candidate).sum().item())
    return {
        "dtype": str(dtype).split(".")[-1],
        "n": int(x.numel()),
        "mismatches": mism,
        "verdict": "EQUIV_PASS" if mism == 0 else "EQUIV_FAIL",
    }


# ---------------------------------------------------------------------------
# Item 2: bool logical-NOT in byte-storage domain — (x == 0) vs (x ^ 1)
# ---------------------------------------------------------------------------
def _item2_bool(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    x_bool = torch.randint(0, 2, (1 << 12,), generator=g, dtype=torch.int64).to(torch.bool)
    corner = torch.tensor([False, True], dtype=torch.bool)
    x_bool = torch.cat([x_bool, corner], dim=0)

    # byte-storage view (this backend stores bool one byte per element)
    x_byte = x_bool.view(torch.uint8)

    # baseline: source GPU bool path -> logical NOT of the byte-stored bool
    baseline = (x_byte == 0).to(torch.uint8)

    # optimized: single vxor with 1 on the byte-stored bool
    one = torch.ones_like(x_byte)
    candidate = torch.bitwise_xor(x_byte, one)

    mism = int((baseline != candidate).sum().item())
    return {
        "dtype": "bool",
        "n": int(x_bool.numel()),
        "mismatches": mism,
        "verdict": "EQUIV_PASS" if mism == 0 else "EQUIV_FAIL",
    }


def main() -> int:
    print("=== BitwiseNotFwdOp §1.6.1 equivalence verification (exact, torch CPU) ===\n")
    print("Item 1: integer complement  baseline=xor(x, all-ones)  optimized=bitwise_not(x)\n")
    print(f"{'dtype':<8} | {'n':>7} | {'mismatches':>10} | 结论")
    print("-" * 48)

    results = []
    for dtype in _INT_DTYPES:
        r = _item1_one_dtype(dtype, seed=1000 + _INT_DTYPES.index(dtype))
        results.append(r)
        print(f"{r['dtype']:<8} | {r['n']:>7} | {r['mismatches']:>10} | {r['verdict']}")

    print("\nItem 2: bool logical-NOT in byte domain  baseline=(x==0)  optimized=(x^1)\n")
    r2 = _item2_bool(seed=2000)
    results.append(r2)
    print(f"{r2['dtype']:<8} | {r2['n']:>7} | {r2['mismatches']:>10} | {r2['verdict']}")

    print("\n采纳项结果表：")
    print(f"{'采纳项':<54} | {'mismatches':>10} | 结论")
    print("-" * 78)
    print(f"{'1. xor(x, all-ones) -> vnot(x)  [uint8/int8/int16/int32/int64]':<54} | "
          f"{sum(x['mismatches'] for x in results[:5]):>10} | "
          f"{'EQUIV_PASS' if all(x['verdict'] == 'EQUIV_PASS' for x in results[:5]) else 'EQUIV_FAIL'}")
    print(f"{'2. (x == 0) -> (x ^ 1)  [bool byte storage]':<54} | "
          f"{r2['mismatches']:>10} | {r2['verdict']}")

    all_pass = all(r["verdict"] == "EQUIV_PASS" for r in results)
    print("\n" + "=" * 78)
    print(f"总体结论: {'全部 EQUIV_PASS — 可采纳' if all_pass else '存在 EQUIV_FAIL — 禁止采纳'}")
    print("=" * 78)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
