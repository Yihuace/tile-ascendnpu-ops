# 优化报告：HardswishFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `HardswishFwdOp`；单 kernel / 单 func / 纯 Vector（AIV）；语义 hardswish（原文以计算链 `vadd(x+3) → vclamp(0,6) → vmul(x,·) → vmul(1/6)` 表述） |
| kernel（基准，只读） | `tileops/kernels/elementwise/hardswish/hardswish_kernel/` |
| 调优产物 | `.../hardswish_kernel/perf_opt/`（原文仅记「产物仅 `perf_opt/`」；具体 kernel 文件名原文未记录） |
| 口径 | `msprof op Task Duration(us)` |
| **工具链版本戳** | tilelang `0.1.2+417dae99` / **CANN 8.5.2** / Ascend910B3 / 日期 `2026-10-15`（任务窗口起 14:04:01Z） |
| 达标线 | 原文按「带宽利用率 **≥60%**」判据讨论，并判定**该判据对本算子不适用**（compute-bound）；正确判据为向量利用率（已达 0.94）；AI 数值原文未记录 |
| benchmark | 原文未记录 benchmark 文件::函数与 tune/smoke 计数（原文仅给 label：`fc-wide` / `mbv3-stage2` / `mbv3-stage3`） |
| 过程日志 | `.../hardswish_kernel/perf_opt/opt_log.md`、`perf_records.jsonl` |
| 前置报告 | 本文件 v0 归档 `history_version/OPTIMIZE_REPORT_v0.md`（内容相同，仅结构未归一） |

## 1. 结论

- 一句话：**该算子 compute-bound，无实质优化空间**——`PipeUtilization`（fc-wide fp16）
  `aiv_vec_ratio = 0.939`、mte2 = 0.534、mte3 = 0.275，**向量单元 94% 打满**；计算链已是最少形态，
  无冗余 op 可删、无 fused vmad 可合并。
- **达标数（双分母）**：原文未记录 x/y 与 a/b 达标数（原报告未按双分母统计，结论为「带宽判据不适用」）；
  豁免 **0**（原文未声明豁免项；6 个 workload 时延 15.8–24.3µs，均非微型 shape）。
- **9.0 端到端复测（2026-10-18，run `20261018_133632_996783_HardswishFwdOp`）**：正确性 **7/7**、ratio **50.20–60.95%（2/6 ≥60%）**，与 8.5 的 compute-bound 结论一致；wrapper 未切换。
- 关键结果：**6 项 A/B 全部在噪声内**（原文结论），无性能收益：

  | workload | base µs | opt µs |
  |---|---:|---:|
  | fc-wide fp16 / bf16 | 20.12 / 21.12 | 19.88 / 20.92 |
  | mbv3-stage2 fp16 / bf16 | 22.9 / 24.3 | 22.7 / 23.9 |
  | mbv3-stage3 fp16 / bf16 | 15.8 / 16.2 | 15.7 / 16.0 |

- **是否建议采纳 wrapper**：❌ **不建议**（原文「无 perf 增益，不切换」；wrapper 未改）。
- 唯一落地改动是 **fp32 健壮性修复**（`_DEFAULT_BLOCK_SIZE` fp32 8192→4096 并在构造时传入），
  与性能无关。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测 | 数值 | 结论 |
|---|---|---|
| 瓶颈 pipe（fc-wide fp16） | **`aiv_vec_ratio = 0.939`**、`mte2 = 0.534`、`mte3 = 0.275` | **向量单元 94% 打满 → 计算受限**，非访存受限 |
| 计算链形态 | `vadd(x+3) → vclamp(0,6) → vmul(x,·) → vmul(1/6)`（+2 `vcast`） | 4 条计算 op 已是 hardswish 公式的最小形态；本 kernel **无 NaN 恢复链**（设计已注明），无冗余 op |
| 算子融合机会 | 无 fused vmad 可再合并 | 无法再减向量工作量 |
| 判据匹配性 | 带宽率 50–61% | 这是计算吃满、DMA 未饱和的自然结果，**不是性能缺陷** ⇒ 「带宽利用率 ≥60%」判据不适用 |
| fp32 UB 容量 | fp32 多缓冲下 UB **>192KB** | `_DEFAULT_BLOCK_SIZE = 8192` 对 fp32 会编译失败；benchmark 无 fp32 用例，故 baseline 未暴露 |

### 2.2 方法（做了什么、为什么这样做）

1. **`_DEFAULT_BLOCK_SIZE` fp32 8192 → 4096**（fp16/bf16 维持 8192）。
   - 改了什么：按 dtype 设置默认 block size，fp32 减半。
   - 为什么有效：fp32 在多缓冲下每元素占用 4B，8192 会使 UB 需求 >192KB 而编译失败；4096 落在
     192KB 预算内。
   - 正确性依据：改后复跑端到端 report `reports/tileops/20261015_220902_635504_HardswishFwdOp/`
     （**passed**）；本改动只影响 fp32 的 tiling，不改变数值链。
2. **`__init__` 构造时传入 `block_size`**：使上述默认值真正生效（否则构造期仍取旧默认）。
3. **不做性能向改动**：诊断已证计算链为最小形态且无冗余 op；wrapper 不切换（无 perf 增益）。

### 2.3 逐轮记录

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| R0 | 诊断 | `PipeUtilization` + 计算链分析 + 判据匹配性核对 | `aiv_vec_ratio = 0.939` → compute-bound；链已最小、无 NaN 恢复链、无 fused vmad | 判 ceiling，不追加结构候选（避免无效深挖） |
| R1 | fp32 健壮性 | `_DEFAULT_BLOCK_SIZE` fp32 8192→4096；`__init__` 传入 block_size | 6 项 A/B 见表 §3，**全部在噪声内**（原文结论），report passed | ✅ 采纳（唯一落地改动，非性能项） |
| R2 | wrapper 切换评估 | — | 无 perf 增益 | ❌ 否决：不切 wrapper（原文） |

## 3. 优化结果（msprof，均在噪声内）

| workload | base µs | final µs | Δ | base SOL% | final SOL% | 是否达标 |
|---|---:|---:|---:|---:|---:|---|
| fc-wide / fp16 | 20.12 | 19.88 | −1.2% | 54.6 | 55.2 | 判据失配（原文） |
| fc-wide / bf16 | 21.12 | 20.92 | −0.9% | 60.6 | 61.2 | 判据失配（原文） |
| mbv3-stage2 / fp16 | 22.9 | 22.7 | −0.9% | 55.0 | 55.6 | 判据失配（原文） |
| mbv3-stage2 / bf16 | 24.3 | 23.9 | −1.6% | 60.4 | 61.6 | 判据失配（原文） |
| mbv3-stage3 / fp16 | 15.8 | 15.7 | −0.6% | 49.9 | 50.3 | 判据失配（原文） |
| mbv3-stage3 / bf16 | 16.2 | 16.0 | −1.2% | 56.7 | 57.6 | 判据失配（原文） |

> Δ 由本表 base/final 两列计算（原文未单列 Δ，仅记「均在噪声内」）；SOL% 为原文 `base%` / `opt%` 列。
> 测量为单次 `msprof op`；原文未记录交错复测次数与噪声带的具体数值，仅以「均在噪声内」定性。

## 4. 精度

- 端到端 report 复跑：`reports/tileops/20261015_220902_635504_HardswishFwdOp/` → **passed**。
- `--level` 分层（L0/L1/L2/Boundary）逐项结果、max_diff 数值、NaN/±inf/−0/subnormal 边界用例：
  **原文未记录**（原文该阶段只记录了 report 复跑结论）。

## 5. 时间与成本

| 项 | 值 |
|---|---|
| 开始 / 结束（UTC） | 2026-10-15T14:04:01Z → 14:24:40Z |
| 墙钟时长 | **≈ 20.6 min** |
| token input/output/reasoning | 2,083 / 2,573 / 6,912 |
| token 合计 / cost | **11,568** / $0.0181 |

> 来源：`reports/optimize/HardswishFwdOp/.start_utc`；`opencode.db` 窗口求和。

## 6. 产物

| 类别 | 路径 / 状态 |
|---|---|
| final kernel | `perf_opt/` 内版本（具体文件名原文未记录） |
| 调优记录 | `.../hardswish_kernel/perf_opt/opt_log.md`、`perf_records.jsonl` |
| baseline 报告 | `reports/tileops/20261013_150641_447235_HardswishFwdOp/` |
| 复跑报告 | `reports/tileops/20261015_220902_635504_HardswishFwdOp/`（**passed**） |
| gate 4 | 原文未记录 |
| 算子状态 | 原文未记录（`hardswish_kernel/` 无 `.stage_state.json`） |
| wrapper 切换 | **未切换**（原文「wrapper 未改（无 perf 增益，不切换）」） |
| pattern-library 回写 | 原文未记录 |

## 7. 天花板 / 豁免 / 缺口

- **豁免项：无**（原文未声明豁免；6 个 workload 时延 15.8–24.3µs，不属于微型 shape）。
- **判据失配（正确判据 = 向量利用率）**：`aiv_vec_ratio = 0.939`（≈0.94）已达计算上限；
  「带宽利用率 ≥60%」对本算子不适用。
- **ceiling 证据**：`PipeUtilization`（fc-wide fp16）`vec 0.939 / mte2 0.534 / mte3 0.275`；
  计算链 4 op 为公式最小形态、无冗余 op、无 fused vmad 可用。**无 CG 编号**（原文未引用能力缺口条目）。
- **贴线风险**：原文未记录（判据不适用；`fc-wide fp16` 55.2%、`mbv3-stage3 fp16` 50.3% 等数值仅作
  「计算吃满、DMA 未饱和」的旁证，非达标依据）。
- **缺口**：fp32 路径未被 benchmark 覆盖（原文「benchmark 无 fp32 用例故 baseline 未暴露」），
  该能力缺口原文未记为 CG 条目。

## 8. 建议

1. **不切换 wrapper**（原文结论）：无 perf 增益，`perf_opt/` 版本仅作为 fp32 健壮性修复的载体保留。
2. 评估口径：本算子应使用**向量利用率**（0.94）而非带宽型 SOL；对 compute-bound 逐元素算子套用
   60% 带宽线会得到错误的「未达标」结论。
3. **benchmark 覆盖**：原文指出 benchmark 无 fp32 用例，导致 fp32 UB 溢出直到本轮才被发现；
   补 fp32 用例属本轮未执行动作（原文未记录是否已补）。
4. 未解决项：计算链无进一步压缩空间（原文已证「无冗余 op、无 fused vmad」）；
   若未来工具链提供可融合的乘加或更高吞吐的向量指令，需重开评估。
5. 跨 CANN 版本复核：本报告为 CANN 8.5.2 口径，升级后应重测 `aiv_vec_ratio` 与 6 个 workload 时延。
