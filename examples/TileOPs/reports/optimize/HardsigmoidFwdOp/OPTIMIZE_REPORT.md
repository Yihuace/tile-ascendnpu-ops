# 优化报告：HardsigmoidFwdOp（Stage-4 optimize）

> 口径说明：本文件为 **CANN 9.0.0 重跑版**（依据同目录 `perf_opt/opt_log.md`）。原 CANN 8.5.2 版本
> （fc-wide 55.7% / 64.6%）已归档到 `history_version/OPTIMIZE_REPORT_v0.md`，**两版数值不得混排**。

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `HardsigmoidFwdOp`；单 kernel / 单 func / 纯 Vector（AIV）；语义 `y = clamp(x + 3, 0, 6) * (1/6)`（NaN 传播，torch 语义） |
| kernel（基准，只读） | `.../hardsigmoid_kernel/hardsigmoid_fwd_kernel.py`（CANN 9.0 起点 = 上一轮 Stage-4 现状：已去冗余 NaN 恢复，5 向量 op `vcast→vadd→vclamp→vmul→vcast`） |
| 调优产物 | `.../hardsigmoid_kernel/perf_opt/HardsigmoidFwdOp.py`（winner `pipe2`） |
| 口径 | `msprof op --kernel-name=main` 的 **Task Duration(us)**（唯一时延口径）+ `visualize_data.bin` 的 **GM Read + Write SOL ratio**；测量方式 `perf_opt/bench_kernel.py` 直跑 kernel（不经 wrapper），每次测量是独立 msprof 进程 |
| **工具链版本戳** | tilelang `0.1.2` dev root（`/home/developer/tilelang-mlir-ascend`，AscendNPU-IR `31f6903`）/ **CANN 9.0.0**（bishengir-compile 1.1.0 `428ab8fdab46` + hivmc 0.2.0）/ torch 2.7.1 / torch_npu 2.7.1.post4 / **Ascend910B3**（Chip 6，20 AIC / 40 AIV）/ 2026-10-18 |
| 达标线 | 逐元素/访存受限 **SOL ≥70%**；微型 shape（≤5µs）豁免；判据失配（向量利用率 ≥0.9）改用向量利用率判据（仍要求 Task Duration 下降）。**AI：fc-wide fp16 = 2.50 / bf16 = 3.00**；实测 roofline `advice = latency bound:compute caused`、`computility = Vec_FP32(99.98%)` ⇒ 归类为 compute-bound |
| benchmark | `benchmarks/ops/bench_hardsigmoid.py::test_hardsigmoid_bench`（6 tune + 3 smoke） |
| 过程日志 | `.../perf_opt/opt_log.md`、`perf_records.jsonl`、`workload_inventory.json`、`profiles/`、`logs/`、`batch_measure_out.txt`、`final_measure_out.txt`；旧 8.5.2 记录 `perf_records.cann852.jsonl.bak` |
| 前置报告 | `history_version/OPTIMIZE_REPORT_v0.md`（**CANN 8.5.2 旧版**：fc-wide fp16 55.7% / bf16 64.6%） |

## 1. 结论

- 一句话：CANN 9.0.0 上现状 kernel 已是「5 op fp32 链 + compute-bound 地板」；本轮唯一落地结构改动
  （A4：满块静态 extent + 2 级流水 + 尾块 epilogue）为 **wash**，无 >5% 收益，判据失配，4 个微型 shape 豁免。
- **达标数（双分母）**：opt_log 原文记为「**全 shape 达标数 / 可调 shape 达标数 = 4 / 2**」，并注明
  「不追求逐 shape 达标」；按模板口径展开为：
  - 全 shape **4/6**（其中 4 个微型 shape 按豁免记达标，2 个可调 shape 未达 70% SOL）；
  - 可调 shape **0/2**（`fc-wide` fp16/bf16 均因判据失配 compute-bound 无法达 70%）；
  - 豁免 **4**（`mbv3-se-gate` fp16 2.08µs / bf16 2.27µs、`mbv3-se-gate-deep` fp16 2.04µs / bf16 2.11µs，均 ≤5µs）。
- 关键结果（final 同一候选 `pipe2`）：

  | 类别 | workload | base µs | final µs | SOL |
  |---|---|---:|---:|---|
  | 可调 | fc-wide fp16 | 17.16 | 17.67 | 55.6% → 53.6% |
  | 可调 | fc-wide bf16 | 17.98 | 17.37 | 59.7% → 63.2% |
  | 豁免 | mbv3-se-gate fp16 / bf16 | 2.08 / 2.11 | 2.08 / 2.27 | 原文未记录（final） |
  | 豁免 | mbv3-se-gate-deep fp16 / bf16 | 2.05 / 2.07 | 2.04 / 2.11 | 原文未记录（final） |

  ⇒ fp16 方向微退（+2.9% 量级）、bf16 方向微升，**均在 ±5% 单进程双态噪声带内**，pooled 判 tie；
  **Task Duration 未变短，如实报告，不硬凑**。
- **是否建议采纳 wrapper**：原文未记录明确采纳建议；实测记录为 wrapper **已切 perf_opt**，队列表把该动作
  标注为「**偏离**：无净收益仍切换，pooled tie，如需回退翻回 baseline import」。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测 | 数值 | 结论 |
|---|---|---|
| baseline（9.0.0 重测现状 kernel） | fc-wide fp16 **17.16µs / 55.6%**；bf16 **17.98µs / 59.7%** | 起点已比 8.5.2 旧版（17.2µs / 16.9µs）不同；bf16 17.98µs 高于旧版 16.9µs，提示基线侧可能采到慢态 |
| roofline / 归类 | AI **2.50**（fp16）/ **3.00**（bf16）；`advice = latency bound:compute caused`；`computility = Vec_FP32(99.98%)` | **判据失配**：GM Read+Write SOL ≥70% 是访存判据，本算子向量计算受限 |
| 瓶颈 pipe（fc-wide fp16） | `aiv_vec_time ≈ 12.8µs`（占 Task Duration ~74%）、`aiv_vec_ratio ≈ 0.85`；`aiv_mte2_time ≈ 11µs` / `ratio ≈ 0.72`；`aiv_mte3_time ≈ 5µs`（~33%）；`aiv_scalar_time ≈ 1.1–2.4µs` | 向量链主导；MTE2 与向量链部分重叠；per-tile 分支/`T.min` 有少量标量开销 |
| 计算链 | fp32 链 = `vcast + vadd + vclamp(2) + vmul` ≈ 5 fp32 op/elem（`aiv_vec_fops/elem ≈ 5.08`） | 已是语义最小（多步链须 fp32 中转对齐 torch golden，PL-1.17） |
| native fp16 是否可用 | PL-1.17：native fp16 多步链中间舍入 2–3 ulp，超 fp16 容差 1e-3；CG-2026-0006：f16 与 f32 逐元素速率相同 | 原生 fp16 无吞吐收益（仅省 2 条 vcast ≈0.02% 的 MISC）→ 不作为候选 |
| 噪声底 | 单进程 Task Duration 双态 **±5%**（PL-1.22） | <5% 差异不可裁决，单次测量不能作为结论 |

### 2.2 方法（做了什么、为什么这样做）

1. **A4 结构改造（唯一落地改动）**：把 fp16/bf16 cast 路径改为
   - **Part A**：guard-free 满块热循环 `T.Pipelined(iters_a, num_stages=2)`，静态 extent（去掉 per-tile `T.min`）；
     trace-time `if iters_a > 0` 守卫 tiny shape（避免 `T.Pipelined(0)` 零次流水）；
   - **Part B**：单尾块 epilogue（尾块不进热循环）。
   - 为什么这样做：诊断显示 `mte2_ratio 0.72` 仍有重叠余量、`scalar_time` 有 per-tile 开销，理论上
     「显式 2 级流水 + 静态 extent」应减少装载等待与标量开销。
   - 实测结果：**wash**（见 2.3）——auto multi-buffer 已部分重叠 MTE2↔向量链，显式 `T.Pipelined(2)`
     在 5-op 短链上无额外收益，多缓冲实例还略增 UB 压力。这与诊断中「compute-bound、向量链已是主导项」一致。
   - 正确性依据：`--level all` 全绿（L0/L1/L2 + Boundary 全 PASS）；NaN 传播在 9.0.0 复验三 dtype
     `NaN-mask-match = True`（`vadd`/`vclamp`/`vmul` 原生传播 NaN，PL-1.21 结论跨 CANN 版本存活）。
2. **去除 dead `nan_mask` buffer**：A4 采纳时一并删掉不再使用的 `nan_mask`（原文 Round 1 裁决：
   「承 bf16 微升 + 去 dead `nan_mask` buffer」）；不影响语义。
3. **fp32 路径保持不变**：仅 smoke（`smoke-1m` fp32）覆盖，不上流水改造。
4. **未实施的候选（诊断层否决）**：native fp16 快路径——依据 PL-1.17（中间舍入超容差）与
   CG-2026-0006（f16/f32 逐元素速率相同，无吞吐收益），预期收益 ≈0.02% MISC，不进入实验轮次。

### 2.3 逐轮记录

> 本轮只有 1 个结构候选（`pipe2`），无第二轮结构深挖；下表 R1 为单次测量，final 复测值见 §3
> （两者是不同测量，不可混用）。

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| R1 | `pipe2`（`HardsigmoidFwdOp.py`） | 满块静态 extent + `T.Pipelined(2)` + 尾块 epilogue（A4），从 current best（`T.serial` + per-tile `T.min`）派生；单点改动 = 流水 + 静态 extent | fc-wide fp16 17.16→17.65（**+2.9%**，55.6→53.8）；bf16 17.98→17.51（**−2.6%**，59.7→62.4）；`mbv3-se-gate` fp16 2.08→2.26（+8.7%）、bf16 2.11→2.24（+6.2%）；`mbv3-se-gate-deep` fp16 2.05→2.03（−1.0%）、bf16 2.07→2.07（0.0%）；6 项 L0 全 PASS | ✅ 采纳为 **canonical 结构**（承 bf16 微升 + 去 dead `nan_mask`），但**不宣称 >5% 收益**；fc-wide fp16 的回退判为噪声带内 |
| R1 裁决 | 噪声裁定 | 单次测量 vs ±5% 噪声带（PL-1.22） | 差异方向不一致（fp16 回退 / bf16 微升），pooled 中位数 tie | 噪声；按 mish/add 先例采纳 A4 为 canonical，不加测 A/B |
| 未实施 | native fp16 快路径 | 直接 fp16 域 5-op 链 | — | ❌ 否决（诊断层）：PL-1.17 中间舍入超容差 1e-3；CG-2026-0006 f16/f32 速率相同、仅省 2 条 vcast ≈0.02% MISC |

## 3. 优化结果（msprof op，CANN 9.0.0；final 同一候选 `pipe2`）

| workload | base µs | final µs | Δ | base SOL% | final SOL% | 是否达标 |
|---|---:|---:|---:|---:|---:|---|
| fc-wide-float16 | 17.16 | 17.67 | +3.0% | 55.6 | 53.6 | ❌ 判据失配（compute-bound） |
| fc-wide-bfloat16 | 17.98 | 17.37 | −3.4% | 59.7 | 63.2 | ❌ 判据失配（compute-bound） |
| mbv3-se-gate-float16 | 2.08 | 2.08 | 0.0% | 0.4 | 原文未记录 | ✅ 豁免（微型 shape） |
| mbv3-se-gate-bfloat16 | 2.11 | 2.27 | +7.6% | 0.4 | 原文未记录 | ✅ 豁免（微型 shape） |
| mbv3-se-gate-deep-float16 | 2.05 | 2.04 | −0.5% | 1.8 | 原文未记录 | ✅ 豁免（微型 shape） |
| mbv3-se-gate-deep-bfloat16 | 2.07 | 2.11 | +1.9% | 2.1 | 原文未记录 | ✅ 豁免（微型 shape） |

> Δ 由本表 base/final 两列直接计算（原文 Final 表未单列 Δ）；微型 shape 的 final SOL% 原文未记录。
> `kernel_id = hardsigmoid_fwd_kernel`，`final_candidate_id = pipe2`（记录见 `perf_records.jsonl`）。
> smoke：`smoke-1m` fp32 / fp16 / bf16（N=1,048,576）仅做精度回归，不参与性能判定。

**噪声裁定（单列）**：单进程 Task Duration 双态 ±5%；fc-wide fp16 的 +2.9%（R1）/ +3.0%（final）与
bf16 的 −2.6% / −3.4% 方向相反，均落在噪声带内；bf16 baseline 17.98µs 高于旧报告 16.9µs，
进一步佐证单次测量不可作为 <5% 裁决依据。本轮以 pooled tie + 诊断（compute-bound）收敛，不再加测 A/B。

## 4. 精度

- NaN 传播（CANN 9.0.0 复验）：三 dtype `NaN-mask-match = True`（`vadd`/`vclamp`/`vmul` 原生传播 NaN）。
- `python perf_opt/HardsigmoidFwdOp.py --level L0` → 三 dtype **PASS**（**max_diff = 0**）。
- `python perf_opt/HardsigmoidFwdOp.py --level all` → **全绿**：L0/L1/L2 全 PASS，Boundary 全 PASS
  （含 **NaN→NaN 三 dtype、`nan_count = 512`**、zeros / large-positive / large-negative / inf），
  `All check passed!`（证据 `final_measure_out.txt`）。
- 端到端（9.0.0）：单算子 report 正确性 **7/7**、`status=passed`（队列表记录）。

## 5. 时间与成本

| 项 | 值 |
|---|---|
| 开始 / 结束（UTC） | 原文未记录（9.0.0 版 opt_log 未记录起止时刻） |
| 墙钟时长 | 原文未记录（9.0.0） |
| token input/output/reasoning | 原文未记录（9.0.0） |
| token 合计 / cost | 原文未记录（9.0.0） |

> 仅 CANN 8.5.2 旧版报告记录过 ≈33.0 min、11,564 token、$0.0212（见
> `history_version/OPTIMIZE_REPORT_v0.md`）。该数据属**旧工具链口径，不得与本文件 §3 的 9.0.0 性能数据混排**。

## 6. 产物

| 类别 | 路径 / 状态 |
|---|---|
| final kernel | `.../hardsigmoid_kernel/perf_opt/HardsigmoidFwdOp.py`（winner `pipe2`） |
| 起点 kernel（9.0 baseline） | `.../perf_opt/hardsigmoid_fwd_kernel.py`（上一轮 Stage-4 现状，5 向量 op） |
| 调优记录 | `.../perf_opt/opt_log.md`、`perf_records.jsonl`、`workload_inventory.json`、`profiles/`、`logs/`、`batch_measure_out.txt`、`final_measure_out.txt`、`logs_l0_pipe2_run1.txt` |
| 旧 8.5.2 记录 | `.../perf_opt/perf_records.cann852.jsonl.bak` |
| 迁移期 / 旧版报告 | `reports/tileops/20261015_113947_514885_HardsigmoidFwdOp/`（baseline，8.5.2）/ `reports/tileops/20261015_214729_104006_HardsigmoidFwdOp/`（perf_opt，8.5.2，passed） |
| 端到端 report（9.0.0） | 队列表记「正确性 7/7、单算子 report status=passed」；report 目录路径原文未记录 |
| gate 4 | 原文未记录（9.0.0 版 opt_log 未载 gate 结论；队列表该算子状态记 DONE） |
| 算子状态 | `phase=DONE`（`.stage_state.json`：`perf_iteration.count=1`、`last_improvement=0.0`、`stage_status[4]=completed`） |
| wrapper 切换 | **已切** perf_opt（`hardsigmoid.py` 已 import `perf_opt.HardsigmoidFwdOp`）；队列表标注为「偏离」（无净收益仍切换，pooled tie） |
| pattern-library 回写 | 候选新条目「AI≈2.5 逐元素 fp32 链 = compute-bound，GM SOL 判据失配」（原文记入 `pattern-library/elementwise.md` 候选，实际写入状态原文未记录） |

## 7. 天花板 / 豁免 / 缺口

- **豁免项（4 个微型 shape，≤5µs）**：`mbv3-se-gate` fp16（2.08µs）/ bf16（2.27µs）、
  `mbv3-se-gate-deep` fp16（2.04µs）/ bf16（2.11µs）——固定启动开销主导，利用率物理不可达；
  按队列表规则「微型 shape Task Duration ≤5µs」豁免，不计入可调分母。
- **判据失配**：AI 2.50/3.00、`advice = latency bound:compute caused`、`computility = Vec_FP32(99.98%)`、
  `aiv_vec_ratio ≈ 0.85`、`aiv_vec_fops/elem ≈ 5.08` ⇒ GM Read+Write SOL 70% 对本算子不适用；
  唯一可动杠杆「MTE2↔向量链流水 + 去 per-tile 标量开销」实测为 wash。
- **ceiling 证据**：fp32 链 5 op 已是语义最小（PL-1.17：多步链须 fp32 中转对齐 torch golden）；
  native fp16 无吞吐收益（CG-2026-0006：f16 与 f32 逐元素速率相同）⇒ 参数/tiling 级无剩余空间。
- **贴线风险**：无贴线项（2 个可调 shape 属判据失配，不是「差一点到线」）；但也**没有 Task Duration 收益**。
- **8.5 旧值**：CANN 8.5.2 版结论（fc-wide fp16 55.7% / bf16 64.6%，opt 17.2 / 16.9µs）见
  `history_version/OPTIMIZE_REPORT_v0.md`；**不得与本节 9.0.0 数据混排或拼接**。
- **能力缺口**：原文未记录新 CG 编号；本轮只引用既有 CG-2026-0006（f16/f32 同速率）。

## 8. 建议

1. **wrapper 采纳建议**：原文未记录明确建议；实际状态为已切 perf_opt 且被标为「偏离」（无净收益）。
   若追求「只在有收益时切换」的纪律，可把 `hardsigmoid.py` 的 import 翻回 baseline（原文给出的回退方式），
   本报告不主张净收益。
2. **不要在 compute-bound 的 5-op fp32 链上继续投结构杠杆**：本轮已证 `T.Pipelined(2)` 是 wash；
   若要跨 70% SOL，只能减少向量 op 数（语义上已最小）或改用向量利用率判据。
3. **跨 CANN 版本复核**：8.5.2 与 9.0.0 的 fc-wide 数值不可混排；升级后需重跑 6 个 tune workload 与
   3 个 smoke（`smoke-1m` fp32/fp16/bf16）的精度。
4. 未解决项：Task Duration 未变短（bf16 −3.4%、fp16 +3.0%，噪声带内），如实留档，不宣称收益。
