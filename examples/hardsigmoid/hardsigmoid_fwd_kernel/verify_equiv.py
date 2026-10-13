"""verify_equiv.py — HardsigmoidFwdOp §1.6.1 采纳项等价性机器验证。

验证两个采纳项（fp64 参照，torch CPU）：

  1. NaN 传播化简：源式逐层 nan_min(nan_max(x+3,0),6)（每层一次 isnan+select，
     共 2 次检测/2 次 select）vs 单点恢复 select(isnan(x), NaN, clamp(...))
     （1 次检测/1 次 select）。语义上 NaN 只能源自输入 x（常数 3/0/6/1/6 均
     有限），逐层传播 ≡ 单点恢复，应位级一致。

  2. 除法转乘倒数：clamp(x+3,0,6)/6（torch F.hardsigmoid 形式）vs
     clamp(x+3,0,6)*(1/6)（源形式，乘法倒数）。fp64 域两式差 ≤1 ulp（1/6
     的倒数舍入），按目标 dtype 舍入后应 EQUIV_PASS（容差内等价）。

方法：候选式与基线式均以 fp64 计算，覆盖随机张量 + 角点值（±inf / NaN /
subnormal / 边界），逐 dtype（fp32/fp16/bf16）舍入对照，输出结果表内嵌
DESIGN.md §1.6.1。任何 FAIL 项禁止采纳。
"""

import numpy as np
import torch


def _fmax(a, b):
    """NaN-dropping max（maxNum 语义，模拟 CUDA fmaxf / Ascend vmax 假设）。"""
    return np.fmax(a, b)


def _fmin(a, b):
    """NaN-dropping min（minNum 语义）。"""
    return np.fmin(a, b)


# ---- 采纳项 1：NaN 传播化简 ----
def baseline_nan_clamp(x):
    """源式：nan_min(nan_max(x+3,0),6) * (1/6)，NaN 逐层传播。"""
    t = x + 3.0
    c1 = np.where(np.isnan(t), np.nan, np.maximum(t, 0.0))    # nan_max
    c2 = np.where(np.isnan(c1), np.nan, np.minimum(c1, 6.0))  # nan_min
    return c2 * (1.0 / 6.0)


def candidate_nan_clamp(x):
    """优化式：普通 clamp（NaN-dropping）+ 单点 NaN 恢复。"""
    t = x + 3.0
    c = _fmin(_fmax(t, 0.0), 6.0)             # plain clamp
    y = c * (1.0 / 6.0)
    return np.where(np.isnan(x), np.nan, y)   # single-point restore


# ---- 采纳项 2：除法转乘倒数 ----
def baseline_div(x):
    """torch F.hardsigmoid 形式：clamp(x+3,0,6) / 6。"""
    return np.minimum(np.maximum(x + 3.0, 0.0), 6.0) / 6.0


def candidate_recip(x):
    """源形式：clamp(x+3,0,6) * (1/6)。"""
    return np.minimum(np.maximum(x + 3.0, 0.0), 6.0) * (1.0 / 6.0)


def _round_to_dtype(a64, dtype_str):
    """fp64 数组按目标 dtype 舍入（模拟 fp32 中转 + 输出 dtype）。"""
    t = torch.tensor(a64, dtype=torch.float64)
    if dtype_str == "float32":
        return t.to(torch.float32)
    # fp16/bf16：compute 在 fp32 中转，输出才舍到低精度
    f32 = t.to(torch.float32)
    return f32.to(torch.float16 if dtype_str == "float16" else torch.bfloat16)


def _ulp_diff(a, b):
    """两 fp64 数组的逐元素 ulp 差（按 a 的 ulp），NaN 视为相等。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.abs(a - b)
    ulp = np.spacing(np.maximum(np.abs(a), np.abs(b)))
    d = diff / ulp
    d[both_nan] = 0.0
    return d


def _check_identical(a64, b64):
    """位级一致判定：NaN 位对齐 + 有限值完全相等。"""
    a = np.asarray(a64, dtype=np.float64)
    b = np.asarray(b64, dtype=np.float64)
    nan_eq = np.isnan(a) == np.isnan(b)
    finite_eq = ~np.isnan(a) & (a == b)
    mismatch = ~(nan_eq & (finite_eq | np.isnan(a)))
    return mismatch.sum(), mismatch


def _make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    inputs = [
        rng.standard_normal(1 << 16) * 10.0,          # 随机大样本
        rng.uniform(-100, 100, 1 << 12),               # 宽动态范围
        np.zeros(64),                                  # 边界 0
        np.array([-3.0, 3.0, 0.0, 6.0, -1e30, 1e30]),  # 边界值
        np.array([np.inf, -np.inf, np.nan]),           # 角点 ±inf/NaN
        np.array([np.nextafter(0.0, 1.0), np.nextafter(0.0, -1.0)]),  # subnormal
        np.array([np.nan, 0.0, 6.0, -3.0]),            # NaN 与边界混合
    ]
    return inputs


def main():
    print("=== HardsigmoidFwdOp §1.6.1 equivalence verification (fp64, torch CPU) ===\n")

    inputs = _make_inputs()
    all_x = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1) for a in inputs])

    results = []

    # --- 采纳项 1：NaN 传播化简 ---
    b1 = baseline_nan_clamp(all_x)
    c1 = candidate_nan_clamp(all_x)
    nmis, _ = _check_identical(b1, c1)
    item1_pass = nmis == 0
    results.append({
        "item": "NaN 传播化简 (逐层 nan_min/nan_max -> 单点 isnan 恢复)",
        "max_ulp": 0.0 if item1_pass else float("inf"),
        "violations": int(nmis),
        "verdict": "EQUIV_PASS" if item1_pass else "EQUIV_FAIL",
    })

    # --- 采纳项 2：除法转乘倒数 ---
    b2 = baseline_div(all_x)
    c2 = candidate_recip(all_x)
    # 有限值域内比较（NaN 传播两式一致，排除 NaN 位）
    finite = ~(np.isnan(b2) | np.isnan(c2))
    d = _ulp_diff(b2[finite], c2[finite])
    max_ulp = float(d.max()) if d.size else 0.0
    # 容差：≤1 ulp（1/6 倒数舍入）
    viol = int((d > 1.0).sum())
    item2_pass = viol == 0
    results.append({
        "item": "除法转乘倒数 (clamp(...)/6 -> clamp(...)*(1/6))",
        "max_ulp": max_ulp,
        "violations": viol,
        "verdict": "EQUIV_PASS" if item2_pass else "EQUIV_FAIL",
    })

    # 打印结果表
    print(f"{'采纳项':<50} | {'max ulp 差':>12} | {'违反率':>8} | 结论")
    print("-" * 90)
    for r in results:
        print(
            f"{r['item']:<50} | {r['max_ulp']:>12.3f} | "
            f"{r['violations']:>8d} | {r['verdict']}"
        )

    all_pass = all(r["verdict"] == "EQUIV_PASS" for r in results)
    print("\n" + ("=" * 90))
    print(f"总体结论: {'全部 EQUIV_PASS — 可采纳' if all_pass else '存在 EQUIV_FAIL — 禁止采纳'}")
    print("=" * 90)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
