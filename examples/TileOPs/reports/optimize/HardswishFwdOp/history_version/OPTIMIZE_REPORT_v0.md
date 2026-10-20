# 优化报告：HardswishFwdOp（Stage-4 optimize）

## 0. 身份
| 项 | 值 |
|---|---|
| 算子 | `HardswishFwdOp`；单 kernel/单 func/纯 Vector（AIV） |
| kernel | `tileops/kernels/elementwise/hardswish/hardswish_kernel/` |
| 口径 | `msprof op Task Duration(us)` |
| 工具链 | tilelang `0.1.2+417dae99` / **上一代工具链**（版本见 `docs/cann9_adaptation.md` §13 两代对照）/ Ascend910B3 |

## 1. 结论（重要）：**该算子 compute-bound，无实质优化空间**
`PipeUtilization`（fc-wide fp16）：**`aiv_vec_ratio=0.939`**、mte2=0.534、mte3=0.275
→ **向量单元 94% 打满，属计算受限**（非访存受限）。

- 计算链已是最少形态：`vadd(x+3) → vclamp(0,6) → vmul(x,·) → vmul(1/6)`（+2 `vcast`），
  **无冗余 op**（本 kernel 无 NaN 恢复链），亦无 fused vmad 可再合并 → 无法再减向量工作量。
- **因此「带宽利用率 ≥60%」判据对本算子不适用**（计算受限；正确判据是向量利用率，已达 0.94）。
  其"带宽率 50–61%"是计算吃满、DMA 未饱和的自然结果，**不是性能缺陷**。

## 2. 所做改动（仅 fp32 健壮性）
- `_DEFAULT_BLOCK_SIZE` fp32 8192→4096（fp32 多缓冲 UB >192KB 编译失败；benchmark 无 fp32 用例故 baseline 未暴露）；
  `__init__` 构造时传入 block_size。
- 产物仅 `perf_opt/`；wrapper 未改（无 perf 增益，不切换）。

## 3. 优化结果（msprof，均在噪声内）
| label | dtype | base µs | opt µs | base% | opt% |
|---|---|---:|---:|---:|---:|
| fc-wide | fp16 | 20.12 | 19.88 | 54.6 | 55.2 |
| fc-wide | bf16 | 21.12 | 20.92 | 60.6 | 61.2 |
| mbv3-stage2 | fp16/bf16 | 22.9 / 24.3 | 22.7 / 23.9 | 55.0 / 60.4 | 55.6 / 61.6 |
| mbv3-stage3 | fp16/bf16 | 15.8 / 16.2 | 15.7 / 16.0 | 49.9 / 56.7 | 50.3 / 57.6 |

## 4. 时间与 token
| 项 | 值 |
|---|---|
| 开始 / 结束（UTC） | 2026-10-15T14:04:01Z → 14:24:40Z |
| 墙钟时长 | **≈ 20.6 min** |
| token input/output/reasoning | 2,083 / 2,573 / 6,912 |
| token 合计 / cost | **11,568** / $0.0181 |

（来源：`reports/optimize/HardswishFwdOp/.start_utc`；`opencode.db` 窗口求和。）

## 5. 产物
- `.../hardswish_kernel/perf_opt/opt_log.md`、`perf_records.jsonl`
- baseline 报告 `reports/tileops/20261013_150641_447235_HardswishFwdOp/`；复跑 `reports/tileops/20261015_220902_635504_HardswishFwdOp/`（passed）
