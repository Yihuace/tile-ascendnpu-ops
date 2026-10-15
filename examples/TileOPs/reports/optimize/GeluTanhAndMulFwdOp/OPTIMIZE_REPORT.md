# 优化报告：GeluTanhAndMulFwdOp（Stage-4 optimize）

## 0. 身份
| 项 | 值 |
|---|---|
| 算子 | `GeluTanhAndMulFwdOp`（`gelu_tanh(x[:,:N]) * x[:,N:]`）；单 kernel/单 func/纯 Vector |
| kernel | `.../gelu_tanh_and_mul_kernel/` |
| 口径 | `msprof op Task Duration(us)`；tilelang `0.1.2+417dae99` / **CANN 8.5.2** / 910B3 |

## 1. 优化方法
**折叠常量、减一条向量 op**。诊断：prefill `aiv_vec_ratio=0.95` → **compute-bound**
（用 sigmoid 实现 tanh：`sigma(2u)=1/(1+exp(-2u))`），**故「带宽利用率」非其判据**。
链中 `u(g)=(g+cg³)·√(2/π)` 与 `−2u` 是两次连续 `vmul`，可合并为
`−2u = (g+cg³)·(−2√(2/π))`（一次 vmul，数学恒等）。kernel 无 NaN 恢复（sigmoid 自饱和）。

## 2. 优化结果（msprof）
| label | dtype | base µs | opt µs | Δlat | base% | opt% |
|---|---|---:|---:|---:|---:|---:|
| prefill | fp16 | 215.62 | **200.14** | **−7.2%** | 47.8 | 48.4 |
| prefill | bf16 | 220.22 | 204.82 | −7.0% | 49.7 | 50.4 |
| swiglu-b32 | fp16 | 5.86 | 5.70 | −2.7% | 23.6 | 22.8 |
| swiglu-b32 | bf16 | 6.02 | 5.84 | −3.0% | 24.4 | 23.7 |
| decode | bf16 | 2.18 | 2.24 | 微型豁免 | 2.5 | 2.3 |

> 延迟降 ~7%。带宽率仍 ~48% 属正常（compute-bound，向量 0.95）。进一步收益需替换 tanh 实现（设计层）。

## 3. 时间与 token
| 项 | 值 |
|---|---|
| 开始 / 结束（UTC） | ≈2026-10-15T14:41:28Z → 14:57:08Z |
| 墙钟时长 | **≈ 15.7 min** |
| token input/output/reasoning | 2,887 / 1,718 / 3,930（+ cache 0） |
| token 合计 / cost | **8,535** / $0.0092 |

## 4. 产物
- `.../gelu_tanh_and_mul_kernel/perf_opt/{gelu_tanh_and_mul_fwd_kernel.py, opt_log.md, perf_records.jsonl}`
- baseline `reports/tileops/20261013_224655_433825_GeluTanhAndMulFwdOp/`；perf_opt `reports/tileops/20261015_224427_350020_GeluTanhAndMulFwdOp/`（passed）
