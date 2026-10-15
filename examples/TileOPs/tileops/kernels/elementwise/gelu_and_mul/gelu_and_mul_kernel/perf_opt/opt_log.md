# GeluAndMulFwdKernel — Stage-4 optimize log

- 算子 `GeluAndMulFwdOp`；纯 Vector（AIV）；单 kernel/单 func。
- 口径 `msprof op Task Duration`；tilelang 0.1.2+417dae99 / CANN 8.5.2 / Ascend910B3。

## 诊断
prefill fp16 `PipeUtilization`：**aiv_vec_ratio=0.879**、mte2=0.142、mte3=0.057 → **compute-bound**
（degree-20 Horner 多项式 erf，向量打满，DMA 仅 14%）→「带宽利用率」非正确判据。

## 优化点：去除冗余 NaN 恢复（vcmp+vselect）
baseline 在多项式链前 `vcmp(g32,g32,"ne")`、链尾 `vselect(nan_mask, g32, e32, e32)`。
实测多项式 erf 链 + `vclamp` 已传播 NaN → 删除恢复两点。

## 回归
`perf_opt ... --level all` 全绿（含 inf/nan 边界）。验收 `tileops-report run --op GeluAndMulFwdOp` status=passed。

## A/B（msprof）
| label | dtype | base µs | opt µs | Δlat | base% | opt% |
|---|---|---:|---:|---:|---:|---:|
| prefill | fp16 | 878.76 | 797.22 | −9.3% | 43.2 | 46.0 |
| prefill | bf16 | 877.64 | 800.24 | −8.8% | 44.0 | 46.6 |
| swiglu-b32 | fp16 | 15.12 | 13.92 | −7.9% | 33.6 | 35.3 |
| swiglu-b32 | bf16 | 14.94 | 13.86 | −7.2% | 34.6 | 36.1 |
| decode | bf16 | 3.96 | 3.8 | 微型豁免 | 4.8 | 4.8 |
