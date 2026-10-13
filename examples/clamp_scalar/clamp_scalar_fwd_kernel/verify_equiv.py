"""verify_equiv.py — ClampScalarFwdOp §1.6.1 采纳项等价性机器验证。

迁移自 GPU ``ClampFwdKernel._make_op_func``（逐曾 nan_max/nan_min 传播 NaN），
验证 NPU 设计的三个采纳项（fp64 参照，torch CPU）：

  1. NaN 传播化简（单点恢复）：源式对每个出现的 bound 各做一次
     ``nan_max``/``nan_min``（fp32 路径下每个 = isnan(a)||isnan(b) 守卫 + Select
     回填 NaN，两 bound 共 2 次检测/2 次 Select）。NPU 设计改用普通 clamp
     （vmax/vmin/vclamp，NaN-dropping）+ 单点恢复 ``select(isnan(x), x, clamp)``
     （1 次检测/1 次 Select）。语义上 NaN 只能源自输入 x（finite bound），
     逐层传播 ≡ 单点恢复，应位级一致（NaN 位对齐）。

  2. NaN-bound 退化分支：torch.clamp(x, min=NaN) / clamp(x, max=NaN) 对整个输出
     返回 NaN（torch 2.7.1 实测）。NPU vmax/vmin 为 NaN-dropping（PL-1.11 P8），
     故编译期（trace-time）对 NaN bound 特化出「全 NaN 输出」分支。基线逐层
     nan_max/nan_min 对 NaN bound 也给全 NaN，两式一致。

  3. NaN 值来源：源式回填 canonical NaN 常数（``T.cast(float("nan"),"float32")``），
     NPU 设计直接回选输入 x（其 NaN 位模式）。两者均为 NaN，NaN 语义下相等
     （并规避 VP-2026-0142 的 NaN FloatImm cache-metadata 陷阱）。

方法：候选式与基线式均以 fp64 计算，覆盖随机张量 + 角点值（±inf / NaN /
subnormal / 边界）与 bound 组合（both / min-only / max-only / NaN bound），
逐 dtype（fp32/fp16/bf16）舍入对照，输出结果表内嵌 DESIGN.md §1.6.1。
任何 FAIL 项禁止采纳。
"""

import numpy as np
import torch


# ---- NaN-propagating max/min（源 _nan.py 语义，fp64） ----
def nan_max(a, b):
    return np.where(np.isnan(a) | np.isnan(b), np.nan, np.maximum(a, b))


def nan_min(a, b):
    return np.where(np.isnan(a) | np.isnan(b), np.nan, np.minimum(a, b))


def _clamp_dropnan(x, lo, hi):
    """普通 clamp（NaN-dropping：fmax/fmin 丢 NaN），lo/hi 可为 None。"""
    r = x
    if lo is not None:
        r = np.fmax(r, lo)
    if hi is not None:
        r = np.fmin(r, hi)
    return r


def baseline_op(x, lo, hi):
    """源式：逐 bound nan_max/nan_min（NaN 逐层传播）。"""
    r = x
    if lo is not None:
        r = nan_max(r, lo)
    if hi is not None:
        r = nan_min(r, hi)
    return r


def candidate_op(x, lo, hi):
    """NPU 设计：普通 clamp（NaN-dropping）+ 单点恢复 + NaN-bound 退化分支。"""
    def isnan(v):
        return isinstance(v, float) and np.isnan(v)

    # 2. NaN-bound 退化分支：任一 present bound 为 NaN -> 全 NaN。
    if (lo is not None and isnan(lo)) or (hi is not None and isnan(hi)):
        return np.full_like(x, np.nan, dtype=np.float64)
    # 1. 普通 clamp + 单点恢复（回选 x，保留其 NaN 位模式）。
    r = _clamp_dropnan(x, lo, hi)
    return np.where(np.isnan(x), x, r)


def _round_to_dtype(a64, dtype_str):
    """fp64 数组按目标 dtype 舍入（模拟 fp32 中转 + 输出舍入）。"""
    t = torch.tensor(a64, dtype=torch.float64)
    if dtype_str == "float32":
        return t.to(torch.float32).to(torch.float64).numpy()
    f32 = t.to(torch.float32)
    (torch.float16 if dtype_str == "float16" else torch.bfloat16)
    low = f32.to(torch.float16 if dtype_str == "float16" else torch.bfloat16)
    return low.to(torch.float64).numpy()


def _ulp_diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = np.abs(a - b)
        ulp = np.spacing(np.maximum(np.abs(a), np.abs(b)))
        d = np.where(ulp > 0, diff / ulp, 0.0)
    d[both_nan] = 0.0
    return d


def _finite_ulp_max(a, b):
    """排除 NaN 位（两式 NaN 语义另行核验）后的最大 ulp 差。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite = ~(np.isnan(a) | np.isnan(b))
    if not finite.any():
        return 0.0
    d = _ulp_diff(a[finite], b[finite])
    return float(d.max()) if d.size else 0.0


def _nan_mismatch(a, b):
    """NaN 位是否对齐（True 处两式 NaN 状态不同）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return int((np.isnan(a) != np.isnan(b)).sum())


def _make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal(1 << 16),                       # 随机标准正态（含 -0.5/0.5 跨界）
        rng.standard_normal(1 << 16) * 10.0,                # 宽范围随机
        np.zeros(64),                                       # 边界 0
        np.array([-0.5, 0.5, 0.0, -1e30, 1e30, 1e-30]),     # bound 边界值
        np.array([np.inf, -np.inf, np.nan, 0.0]),           # 角点 ±inf / NaN
        np.array([np.nextafter(0.0, 1.0), np.nextafter(0.0, -1.0)]),  # subnormal
        np.array([np.nan, 0.5, -0.5, 1e30, np.inf]),        # NaN 与角点混合
    ]


def _all_inputs():
    return np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1) for a in _make_inputs()])


def _check_bounds(name, lo, hi, x, dtype_str):
    b = baseline_op(x, lo, hi)
    c = candidate_op(x, lo, hi)
    b = _round_to_dtype(b, dtype_str)
    c = _round_to_dtype(c, dtype_str)
    nan_mis = _nan_mismatch(b, c)
    max_ulp = _finite_ulp_max(b, c)
    viol = int((_ulp_diff(b, c) > 1.0).sum())
    ok = nan_mis == 0 and viol == 0
    return {
        "name": name,
        "dtype": dtype_str,
        "nan_mismatch": nan_mis,
        "max_ulp": max_ulp,
        "violations": viol,
        "verdict": "EQUIV_PASS" if ok else "EQUIV_FAIL",
    }


def main():
    print("=== ClampScalarFwdOp §1.6.1 equivalence verification (fp64, torch CPU) ===\n")
    x = _all_inputs()

    bound_sets = [
        ("both-finite (min=-0.5, max=0.5)", -0.5, 0.5),
        ("min-only (-0.5)", -0.5, None),
        ("max-only (0.5)", None, 0.5),
        ("min=NaN (all-NaN branch)", float("nan"), 0.5),
        ("max=NaN (all-NaN branch)", -0.5, float("nan")),
    ]

    rows = []
    for name, lo, hi in bound_sets:
        for dtype_str in ("float32", "float16", "bfloat16"):
            r = _check_bounds(name, lo, hi, x, dtype_str)
            rows.append((f"{name} [{dtype_str}]", r))

    print(f"{'采纳项 / bound 组合 [dtype]':<48} | {'NaN 失配':>9} | {'max ulp':>8} | {'违反':>5} | 结论")
    print("-" * 100)
    for label, r in rows:
        print(
            f"{label:<48} | {r['nan_mismatch']:>9d} | {r['max_ulp']:>8.3f} | "
            f"{r['violations']:>5d} | {r['verdict']}"
        )

    # 采纳项 3：NaN 值来源（canonical 常数 vs 回选输入 x 的 NaN 位模式）——NaN 语义下等价。
    xb = np.array([np.nan, 0.5, -0.5], dtype=np.float64)
    base_canon = baseline_op(xb, -0.5, 0.5)         # 用 canonical NaN 常数
    cand_self = candidate_op(xb, -0.5, 0.5)         # 回选输入 x（NaN 位模式）
    item3_ok = (
        _nan_mismatch(base_canon, cand_self) == 0
        and _finite_ulp_max(base_canon, cand_self) == 0.0
    )
    print(
        f"{'NaN 值来源 (canonical vs 回选 x 的 NaN)':<48} | {_nan_mismatch(base_canon, cand_self):>9d} | "
        f"{_finite_ulp_max(base_canon, cand_self):>8.3f} | {0:>5d} | "
        f"{'EQUIV_PASS' if item3_ok else 'EQUIV_FAIL'}"
    )

    all_pass = all(r["verdict"] == "EQUIV_PASS" for _, r in rows) and item3_ok
    print("\n" + "=" * 100)
    print(f"总体结论: {'全部 EQUIV_PASS — 可采纳' if all_pass else '存在 EQUIV_FAIL — 禁止采纳'}")
    print("=" * 100)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
