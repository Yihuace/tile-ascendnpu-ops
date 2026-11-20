#!/usr/bin/env python3
"""等价性机器验证（D-1）——bitwise_not_fwd_kernel 的 §1.6.1 采纳优化项。

对照对象：
  #1 整数路径算子降代换   baseline: y = x ^ (-1)      candidate: y = ~x
  #2 uint8  路径算子降代换  baseline: y = x ^ 0xFF     candidate: y = ~x & 0xFF
  #3 bool   路径域内改写    baseline: y = (x == 0)     candidate: y_byte = x ^ 1

参照域：整数/布尔位运算无舍入，本脚本用**任意精度 Python int** 逐位精确计算
（等价于 fp64 精确参照，且避免 int64 无符号边界溢出）；ulp 差 = |candidate-baseline|
（位运算下正确即恒 0）。输入覆盖随机张量 + 角点值（各 dtype min/max/0/±1/符号位边界）。

用法：python3 verify_equiv.py
"""

import random

random.seed(20261012)
N = 1 << 20  # 1M 随机样本/项

SIGNED_BITS = {"int8": 8, "int16": 16, "int32": 32, "int64": 64}


def corner_values(bits, signed):
    if signed:
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    else:
        lo, hi = 0, (1 << bits) - 1
    vals = [0, 1, -1, 2, -2, lo, hi, lo + 1, hi - 1]
    if signed:
        vals += [1 << (bits - 1), -(1 << (bits - 1)) + 1]
    return [v for v in vals if lo <= v <= hi]


def rand_bits(bits, n):
    return [random.getrandbits(bits) for _ in range(n)]


def evaluate(name, bits, signed):
    """#1/#2：baseline x^(-1)（全 1 掩码） vs candidate ~x（按位取反）。"""
    mask = (1 << bits) - 1
    vals = rand_bits(bits, N) + [v & mask for v in corner_values(bits, signed)]
    max_ulp = 0
    viol = 0
    for v in vals:
        baseline = (v ^ mask) & mask          # x ^ (-1)（bits 位全 1 掩码）
        candidate = (~v) & mask               # 按位取反
        d = abs(candidate - baseline)
        if d:
            viol += 1
            if d > max_ulp:
                max_ulp = d
    return {
        "item": name,
        "n": len(vals),
        "max_ulp": max_ulp,
        "viol_rate": viol / len(vals),
        "verdict": "EQUIV_PASS" if viol == 0 else "EQUIV_FAIL",
    }


def evaluate_bool():
    """#3：baseline (x == 0) vs candidate x ^ 1（1 字节存储 x∈{0,1}）。"""
    vals = [random.getrandbits(1) for _ in range(N)] + [0, 1, 0, 1]
    max_ulp = 0
    viol = 0
    for v in vals:
        baseline = 1 if v == 0 else 0          # x == 0（逻辑非）
        candidate = v ^ 1                      # 1 字节存储语义
        d = abs(candidate - baseline)
        if d:
            viol += 1
            if d > max_ulp:
                max_ulp = d
    return {
        "item": "bool 路径  (x==0) vs (x^1)",
        "n": len(vals),
        "max_ulp": max_ulp,
        "viol_rate": viol / len(vals),
        "verdict": "EQUIV_PASS" if viol == 0 else "EQUIV_FAIL",
    }


def main():
    rows = [
        evaluate("整数路径 int8   ~x vs x^-1", 8, True),
        evaluate("整数路径 int16  ~x vs x^-1", 16, True),
        evaluate("整数路径 int32  ~x vs x^-1", 32, True),
        evaluate("整数路径 int64  ~x vs x^-1", 64, True),
        evaluate("uint8 路径   ~x vs x^255", 8, False),
        evaluate_bool(),
    ]

    hdr = f"{'item':<32}{'samples':>10}{'max_ulp':>9}{'viol_rate':>12}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['item']:<32}{r['n']:>10}{r['max_ulp']:>9}{r['viol_rate']:>12.6f}  {r['verdict']}")

    fails = [r for r in rows if r["verdict"] != "EQUIV_PASS"]
    print()
    if fails:
        print(f"RESULT: EQUIV_FAIL ({len(fails)} item(s)) -> 禁止采纳")
        raise SystemExit(1)
    print(f"RESULT: ALL {len(rows)} ITEMS EQUIV_PASS")


if __name__ == "__main__":
    main()
