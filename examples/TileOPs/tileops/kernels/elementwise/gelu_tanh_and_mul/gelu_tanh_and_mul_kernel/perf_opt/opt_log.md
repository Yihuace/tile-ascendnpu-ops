# GeluTanhAndMulFwdKernel — Stage-4 optimize log

- 算子 `GeluTanhAndMulFwdOp`；纯 Vector（AIV）；单 kernel/单 func。
- 口径 `msprof op Task Duration`；tilelang 0.1.2+417dae99 / CANN 8.5.2 / 910B3。

## 诊断
prefill fp16 `aiv_vec_ratio=0.95` → **compute-bound**（sigmoid 形式 tanh，向量打满）→ 带宽率非正确判据。
kernel **无 NaN 恢复**（sigmoid 自饱和，无 inf/inf 路径）。

## 优化点：折叠 `u(g)` 与 `-2u` 两次 vmul 为一次
baseline：`u32 = (g+cg³)` → `vmul(u32, √(2/π))` → `vmul(u32, -2)` = `-2u`（两次 vmul）。
perf_opt：`vi32 = (g+cg³) · (−2√(2/π))`（一次 vmul）。数学恒等，省 1 条向量 op。

## 回归
`perf_opt ... --level all` 全绿（含 tanh 饱和 / 大负值 / signed-zero 边界）。
验收 `tileops-report run --op GeluTanhAndMulFwdOp` status=passed。

## A/B（msprof）
| label | dtype | base µs | opt µs | Δlat | base% | opt% |
|---|---|---:|---:|---:|---:|---:|
| prefill | fp16 | 215.62 | 200.14 | −7.2% | 47.8 | 48.4 |
| prefill | bf16 | 220.22 | 204.82 | −7.0% | 49.7 | 50.4 |
| swiglu-b32 | fp16 | 5.86 | 5.70 | −2.7% | 23.6 | 22.8 |
| swiglu-b32 | bf16 | 6.02 | 5.84 | −3.0% | 24.4 | 23.7 |
| decode | bf16 | 2.18 | 2.24 | 微型豁免 | 2.5 | 2.3 |
