# HardsigmoidFwdKernel — Stage-4 optimize log

- 算子 `HardsigmoidFwdOp`；kernel `hardsigmoid_kernel/`（纯 Vector / AIV）。
- 基线 `hardsigmoid_fwd_kernel.py`（Stage 3）→ 优化版 `perf_opt/hardsigmoid_fwd_kernel.py`。
- 唯一口径：`msprof op Task Duration(us)`。
- 工具链：tilelang 0.1.2+417dae99 / CANN 8.5.2 / Ascend910B3。

## 优化点 1：去除冗余 NaN 恢复
baseline 链：`vcast → vadd(x+3) → vclamp(0,6) → vmul(1/6) → vbrc(NaN)+vcmp+vselect → vcast`。
实测本工具链 `vadd/vclamp/vmul` 已传播 NaN（NaN→NaN）→ 删除 `vbrc+vcmp+vselect`（6→3 向量 op）。

## 优化点 2：修 fp32 UB 溢出（原 baseline bug）
`_DEFAULT_BLOCK_SIZE` 标量 8192 → fp32 多缓冲需 ~224KB>192KB，`smoke-1m-float32` 编译失败（baseline 报告 partial）。
改为按 dtype：fp32=4096 / fp16=8192 / bf16=8192；`__init__` 构造时传入。

## 回归
`perf_opt/hardsigmoid_fwd_kernel.py --level all` 全绿（含 NaN→NaN / inf / 边界）。
验收 `tileops-report run --op HardsigmoidFwdOp`：status=passed，正确性 7/7。

## A/B（msprof）
| label | dtype | base µs | opt µs | Δlat | base% | opt% |
|---|---|---:|---:|---:|---:|---:|
| fc-wide | fp16 | 28.98 | 17.2 | −40.6% | 50.5 | 55.7 |
| fc-wide | bf16 | 28.88 | 16.9 | −41.5% | 57.0 | 64.6 |
| se-gate / se-gate-deep | fp16/bf16 | ~2.1–2.9 | ~2.1–2.6 | 微型豁免 | ~0.4–2.4 | ~0.4–2.1 |
