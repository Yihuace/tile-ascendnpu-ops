"""verify_equiv.py — HardswishFwdOp §1.6.1 采纳项等价性机器验证。

采纳项（1 项，fp64 参照，torch CPU）：

  1. 除法转乘倒数：
       原式（基线）  : y = x * clamp(x + 3, 0, 6) / 6        （torch F.hardswish 语义）
       优化后公式    : y = x * clamp(x + 3, 0, 6) * (1 / 6)  （源 HardswishFwdKernel.op_func 形式）
     0..6 区间的 clamp 无奇点，(x * clamp) / 6 与 (x * clamp) * (1/6) 在 fp64 域
     仅相差 1/6 的倒数舍入，两式在 fp64 下逐位对照（有限值域）应 ≤1 ulp。

说明：Hardswish 与同族 Hardsigmoid 不同——GPU 源算子 ``HardswishFwdKernel.op_func``
用普通 ``T.min`` / ``T.max``（非 ``nan_min`` / ``nan_max``），因为 hardswish 的输出含
``x * ...`` 因子，NaN/±inf 由该乘法天然传播；``clamp(x+3,0,6)`` 的 NaN 语义对结果无影响。
因此本算子**无** NaN 单点恢复优化项（对照 Hardsigmoid 的 §1.6.1 项 1），verify 脚本
只覆盖除法→乘倒数这一项。

方法：候选式与基线式均以 fp64 计算，覆盖随机张量 + 角点值（±inf / NaN / subnormal /
边界），逐 dtype（fp32/fp16/bf16）按目标舍入路径对照，输出结果表内嵌 DESIGN.md §1.6.1。
任何 EQUIV_FAIL 项禁止采纳。
"""

import numpy as np
import torch


# ---- 采纳项 1：除法转乘倒数 ----
def baseline_div(x):
    """原式（基线）：x * clamp(x+3, 0, 6) / 6。"""
    clamp = np.minimum(np.maximum(x + 3.0, 0.0), 6.0)
    return x * clamp / 6.0


def candidate_recip(x):
    """优化式：x * clamp(x+3, 0, 6) * (1/6)（源 HardswishFwdKernel.op_func 形式）。"""
    clamp = np.minimum(np.maximum(x + 3.0, 0.0), 6.0)
    return x * clamp * (1.0 / 6.0)


def _round_to_dtype(a64, dtype_str):
    """fp64 数组按目标 dtype 舍入（模拟 fp32 中转 + 输出 dtype）。"""
    t = torch.tensor(a64, dtype=torch.float64)
    if dtype_str == "float32":
        return t.to(torch.float32)
    # fp16/bf16：compute 在 fp32 中转，输出才舍到低精度
    f32 = t.to(torch.float32)
    return f32.to(torch.float16 if dtype_str == "float16" else torch.bfloat16)


def _ulp_diff(a, b):
    """两数组的逐元素 ulp 差（按两值绝对量级较大者），NaN 视为相等。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.abs(a - b)
    ulp = np.spacing(np.maximum(np.abs(a), np.abs(b)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = diff / ulp
    d[both_nan] = 0.0
    return d


def _make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    inputs = [
        rng.standard_normal(1 << 16) * 10.0,           # 随机大样本
        rng.uniform(-100, 100, 1 << 12),                # 宽动态范围
        np.zeros(64),                                   # 边界 0
        np.array([-3.0, 3.0, 0.0, 6.0, -1e30, 1e30]),   # 边界值（x+3 落 0/6）
        np.array([np.inf, -np.inf]),                    # 角点 ±inf（乘法传播）
        np.array([np.nextafter(0.0, 1.0), np.nextafter(0.0, -1.0)]),  # subnormal
        np.array([np.nan, 0.0, 6.0, -3.0]),             # NaN 与边界混合
    ]
    return inputs


def main():
    print("=== HardswishFwdOp §1.6.1 equivalence verification (fp64, torch CPU) ===\n")

    inputs = _make_inputs()
    all_x = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1) for a in inputs])

    results = []

    # --- 采纳项 1：除法转乘倒数 ---
    # fp64 参照下两式数值对照（有限值域；NaN/±inf 由 x*... 乘法天然传播，两式一致）
    b64 = baseline_div(all_x)
    c64 = candidate_recip(all_x)
    finite = ~(np.isnan(b64) | np.isnan(c64)) & np.isfinite(b64) & np.isfinite(c64)
    d = _ulp_diff(b64[finite], c64[finite])
    max_ulp = float(np.nanmax(d)) if d.size else 0.0
    viol = int(np.sum(d > 1.0))
    item_pass = viol == 0
    results.append({
        "item": "除法转乘倒数 (x*clamp(...)/6 -> x*clamp(...)*(1/6))",
        "max_ulp": max_ulp,
        "violations": viol,
        "verdict": "EQUIV_PASS" if item_pass else "EQUIV_FAIL",
    })

    # 逐 dtype 舍入路径对照（fp32 中转后按输出 dtype 舍入）
    print("逐 dtype 舍入路径对照（违反率 = ulp 差 > 1 的元素数）：")
    print(f"{'dtype':<10} | {'max ulp 差':>12} | {'违反率':>8} | 结论")
    print("-" * 60)
    for dtype_str in ("float32", "float16", "bfloat16"):
        rb = _round_to_dtype(b64, dtype_str).to(torch.float64).numpy()
        rc = _round_to_dtype(c64, dtype_str).to(torch.float64).numpy()
        fin = ~(np.isnan(rb) | np.isnan(rc)) & np.isfinite(rb) & np.isfinite(rc)
        dd = _ulp_diff(rb[fin], rc[fin])
        m = float(np.nanmax(dd)) if dd.size else 0.0
        v = int(np.sum(dd > 1.0))
        print(f"{dtype_str:<10} | {m:>12.3f} | {v:>8d} | "
              f"{'EQUIV_PASS' if v == 0 else 'EQUIV_FAIL'}")

    # 打印结果表（内嵌 DESIGN.md §1.6.1 的 canonical 表）
    print("\n采纳项结果表：")
    print(f"{'采纳项':<52} | {'max ulp 差':>12} | {'违反率':>8} | 结论")
    print("-" * 96)
    for r in results:
        print(
            f"{r['item']:<52} | {r['max_ulp']:>12.3f} | "
            f"{r['violations']:>8d} | {r['verdict']}"
        )

    all_pass = all(r["verdict"] == "EQUIV_PASS" for r in results)
    print("\n" + "=" * 96)
    print(f"总体结论: {'全部 EQUIV_PASS — 可采纳' if all_pass else '存在 EQUIV_FAIL — 禁止采纳'}")
    print("=" * 96)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
