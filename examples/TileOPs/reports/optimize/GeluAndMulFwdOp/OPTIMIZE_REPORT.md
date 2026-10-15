# 优化报告：GeluAndMulFwdOp（Stage-4 optimize）

## 0. 身份
| 项 | 值 |
|---|---|
| 算子 | `GeluAndMulFwdOp`（`y = gelu(x[:, :N]) * x[:, N:]`）；单 kernel/单 func/纯 Vector |
| kernel | `tileops/kernels/elementwise/gelu_and_mul/gelu_and_mul_kernel/` |
| 口径 | `msprof op Task Duration(us)`；tilelang `0.1.2+417dae99` / **CANN 8.5.2** / 910B3 |

## 1. 优化方法
**去除冗余 NaN 恢复**。诊断：prefill fp16 `aiv_vec_ratio=0.879`、mte2=0.142 → **compute-bound**
（degree-20 Horner 多项式 erf 把向量单元打满），**故「带宽利用率」不是本算子的正确判据**（正确判据是向量利用率）。
baseline 在链前 `vcmp` + 链尾 `vselect` 做 NaN 恢复；实测多项式链 + `vclamp` 已传播 NaN → 删除（省 2 条向量 op）。

## 2. 优化结果（msprof）
| label | dtype | base µs | opt µs | Δlat | base% | opt% |
|---|---|---:|---:|---:|---:|---:|
| prefill | fp16 | 878.76 | **797.22** | **−9.3%** | 43.2 | 46.0 |
| prefill | bf16 | 877.64 | 800.24 | −8.8% | 44.0 | 46.6 |
| swiglu-b32 | fp16 | 15.12 | 13.92 | −7.9% | 33.6 | 35.3 |
| swiglu-b32 | bf16 | 14.94 | 13.86 | −7.2% | 34.6 | 36.1 |
| decode | bf16 | 3.96 | 3.8 | 微型豁免 | 4.8 | 4.8 |

> 延迟降 7%–9%。**带宽率仍 ~46% 属正常**：本算子 compute-bound（向量 0.88 打满），带宽率非其判据；进一步降延迟需削减 erf 多项式规模（设计层改动，超出本优化范围）。

## 3. 时间与 token
| 项 | 值 |
|---|---|
| 开始 / 结束（UTC） | 2026-10-15T14:25:02Z → 14:41:12Z |
| 墙钟时长 | **≈ 16.2 min** |
| token input/output/reasoning | 3,228 / 1,675 / 2,397 |
| token 合计 / cost | **7,300** / $0.0153 |

## 4. 产物
- `.../gelu_and_mul_kernel/perf_opt/{gelu_and_mul_fwd_kernel.py, opt_log.md, perf_records.jsonl}`
- baseline `reports/tileops/20261013_215524_720318_GeluAndMulFwdOp/`；perf_opt `reports/tileops/20261015_222824_082803_GeluAndMulFwdOp/`（passed）
