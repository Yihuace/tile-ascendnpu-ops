# FusedAddLayerNormFwdOp — Stage 4 第二轮结构深挖（R7）数据包

> 供主会话合并进 `examples/TileOPs/reports/optimize/FusedAddLayerNormFwdOp/OPTIMIZE_REPORT.md`。
> 口径：`msprof op` Task Duration(us)（唯一时延口径），`--launch-count=20 --warm-up=5`，取 20 次中位数。
> 工具链：tilelang 0.1.2 dev root（`/mnt/workspace/tilelang-mlir-ascend`）+ CANN 9.0.0 + Ascend910B3（20 AICore × 2 AIV = 40 向量核）/ 2026-10-20。
> 数据来源：`perf_opt/perf_records.jsonl`（round 7）、`perf_opt/profiles/`、`perf_opt/logs/r7/`。本文数字与 jsonl 一致，未做任何手工修饰。

## 0. 身份与产物

| 项 | 值 |
|---|---|
| 算子 / kernel | `FusedAddLayerNormFwdOp` / `_fused_add_layer_norm_kernel`（单 kernel、单 `@T.prim_func` 家族、纯 Vector / Developer 模式） |
| 只读基线 | `_fused_add_layer_norm_kernel.py`（Stage-3，wrapper 现接此文件） |
| 本轮产物 | `perf_opt/FusedAddLayerNormFwdOp.py` |
| 产物 sha256 | `6b051de2958b086cbb6c5f23cb57681911af9ae3a327564b61adca96867528be` |
| 本轮起点 | R6 final（hoist-wb + loads-first，sha256 `d86b7d1d…`） |
| 轮次 | R7（1 轮结构深挖，5 个实验分支 + 1 次合并） |
| 实测墙钟 | 2026-10-20 19:30 → 20:30（含 3 次 L0/L1 回归、~35 次 msprof 采集） |

## 1. 诊断证据（profile 数值 + 归因）

### 1.1 起点 profile（8b-prefill fp16，current best = R6 final）

raw：`perf_opt/profiles/r7diag-cur-8b-fp16/`（`--aic-metrics=Default,PipeUtilization,Roofline`）

| 指标 | 数值 |
|---|---|
| Task Duration(us) median / min / max | 62.69 / 62.26 / 63.08 |
| Block Dim / Mix Block Dim | 40 / NA |
| aiv_time | 60.0µs |
| aiv_vec_time / ratio | 45.35µs / **0.756** |
| aiv_mte2_time / ratio | 20.5µs / 0.336 |
| aiv_mte3_time / ratio | 10.3µs / 0.170 |
| aiv_scalar_time / scalar_wait | 9.4µs / 49.5µs |

归因（三条，均可复核）：

1. **瓶颈已从装载转到向量链**：vec 0.756 忙、MTE2 0.336 ⇒ R5 的「MTE2/VEC 串行」已解决；剩余 gap 只有 `aiv_time 60.0` → `Task Duration 62.7` 的 2.7µs（launch + 尾块），核间 imbalance 1.6%（1024 logical / 40 核）。
2. **无 AIV 重复执行**：PipeUtilization 每 block 仅 1 条 `vector0` 子块、Mix Block Dim = NA ⇒ 不命中 PL-1.13「双 AIV 重复执行」，`num_kernels = 40` 是真满载（1 wave）。因此本轮**前置否决**了 subid 分片候选（同类 batch_norm_bwd 曾靠它拿 −28~−46%，形态不适用于本 kernel）。
3. **两处实现级缺陷**（由逐 dtype 路径读代码 + 流量对账发现）：
   - `block_m==1`（70b，N=8192）仍在用单 `stage` 四角色复用链（R6 因 UB 差 256 bits 放弃 loads-first），而该分支本来就有第二个同形 dtype 平面 `wstage`（权重上转后即空闲）；
   - fp32 直通路径（13b，N=5120）把 `T.copy(weight)/T.copy(bias)` 留在 persistent serial 循环内——R3 的 A10 外提只覆盖了两条 transit 分支。按流量算：每 task payload 4×N×4B，因子流 2×N×4B ⇒ **GM 流量 +50%**。

### 1.2 终点 profile

| workload | Task Duration(us) | vec ratio | mte2 ratio | aiv_time | vec_time | mte2_time | mte3_time |
|---|---:|---:|---:|---:|---:|---:|---:|
| 70b-prefill fp16（final） | 117.09 | 0.702 | 0.351 | 113.5 | 79.6 | 39.9 | 19.1 |
| 13b-prefill fp32（final） | 78.64 | 0.599 | 0.467 | 74.7 | 44.6 | 34.9 | 24.4 |

raw：`perf_opt/profiles/r7diag-final-llama-70b-prefill-float16/`、`perf_opt/profiles/r7diag-final-llama-13b-prefill-float32/`。
13b 的 pipe 画像已从算力侧（vec 主导）转为访存侧（mte2 0.467，约 2.1 TB/s 有效带宽），说明 op2 的 GM 流量削减是有效杠杆。

## 2. 方法与逐轮记录

### 2.1 分支（每个分支只改一个主要优化点，均从 R6 final 派生）

| opt_id | 分支文件 | 改动（最小 delta） | 结果 | 处置 |
|---|---|---|---|---|
| op1 `pinput_stage` | `FusedAddLayerNormFwdOp_r7_loadsfirst_bm1.py` | `block_m==1` 分支：x→`wstage`、residual→`stage` 两条 MTE2 先发，再两次 vcast（UB 零增量） | 70b prefill fp16 141.14→116.98µs（**−17.1%**）、bf16 144.78→120.53（**−16.8%**） | ✅ 采纳 |
| op2 `fp32hoist` | `FusedAddLayerNormFwdOp_r7_fp32hoist.py` | fp32 路径：weight/bias `T.copy` 移出 serial 循环（新增 `b1` 常驻） | 13b fp32 108.54→79.82µs（**−26.6%**，两 rep −26.4~−27.3%） | ✅ 采纳 |
| op3 `pipe2` | `FusedAddLayerNormFwdOp_r7_pipe2.py` | 两处 `T.serial` → `T.Pipelined(num_local_tasks, num_stages=2)` | 8b fp16 62.51/62.84 vs 62.86/62.66；bf16 64.72/64.53 vs 64.81/64.82（方向不一致，**tie**） | ❌ config_no_gain |
| op4 `storecast` | `FusedAddLayerNormFwdOp_r7_storecast.py` | bm>=2 分支输出 round-back 折叠进跨 dtype `T.copy` | 8b fp16 +5.1%、bf16 +5.0%（稳定同向回退） | ❌ invalid |
| op5 `hoistbrc` | `FusedAddLayerNormFwdOp_r7_hoistbrc.py` | `vbrc(w1)/vbrc(b1)` 提到循环外（`(block_m,N)` 平面），去掉 2/12 全平面 pass | 8b fp16 63.21/62.92 vs 63.07/62.50；bf16 64.64/64.73 vs 64.54/65.29（**tie**） | ❌ config_no_gain |
| merged | `FusedAddLayerNormFwdOp_r7_merged.py` → `FusedAddLayerNormFwdOp.py` | op1 + op2 合入同一 `@T.prim_func` 家族（不同编译期分支，互不干扰） | 全 7 tune workload 无回退（8b tie、70b −17%、13b −27%） | ✅ final |

测量纪律：每个分支先 L0（`--level L0`，13/13 PASS）再 `msprof op`；分支间同会话、同 workload、**A/B/A/B 交错**（`run_experiments.py --order workload --reps 2`），平区（<3%）用配对方向一致性裁决；每分支测量后向 `perf_records.jsonl` 追加一行（append-only，`round=7`）。

### 2.2 关键否证记录（避免下轮重复）

- **「vec_time ∝ pass 数」被本轮实测否证**：op5 去掉 2/12 个全平面 pass，Task Duration 不动（4 个 (workload, rep) 组合方向不一致，最大 −0.7%）。⇒ 8b 的剩余时间是每 task 依赖链延迟（2 次 row-reduce 串行 + 约 16 个链接），不是 pass 吞吐。
- **task 级软件流水无空间**：op3 tie ⇒ auto-multi-buffer 已覆盖 task 间重叠。
- **跨 dtype store 折叠不划算**：op4 需内部 staging 平面 + 额外 MTE，+5% 稳定回退。
- **未启动**：① 双 reduce 合并为一次 `reduce_sum`（语义不可能：variance 需先有 mean；`T.reduce` 无多输出，见 `docs/Tilelang.language/规约操作/T.reduce.md` §2.2.2 单维不同条款；单遍方差 R5 已 tie 且有数值风险）；② N 方向分核（需第二 kernel + GM workspace，破坏 wrapper 单 kernel 契约；40 AIV 已满载）；③ Cube GEMV 归约（Developer 模式 persistent + `T.gemm` 运行期崩溃，CG-2026-0010；需 Expert 重写，未实测，属未实证假设）；④ subid 分片（前置否决，见 §1.1-2）。

## 3. 逐 workload 结果（baseline / final / Δ，与 perf_records.jsonl 一致）

final 记录：`round=7`、`candidate_id=r7final`、`phase=final`，7 条来自同一候选（`FusedAddLayerNormFwdOp.py`，sha256 `6b051de2…`）；baseline 为同会话相邻交错的 Stage-3 基线重测。

| workload | base µs | final µs | Δ | base SOL% | final SOL% | 判据/线 | 达标 |
|---|---:|---:|---:|---:|---:|---|---|
| llama-13b-prefill-float32 | 106.520 | 79.010 | −25.83% | 44.49 | 60.66 | 访存 70% | 否 |
| llama-70b-decode-bfloat16 | 4.620 | 4.210 | −8.87% | 0.81 | 0.92 | 微型豁免 | 豁免 |
| llama-70b-prefill-bfloat16 | 169.340 | 120.400 | −28.90% | 32.86 | 40.56 | 算力 60% | 否 |
| llama-70b-prefill-float16 | 165.310 | 116.790 | −29.35% | 29.18 | 35.27 | 算力 60% | 否 |
| llama-8b-decode-bfloat16 | 4.220 | 4.190 | −0.71% | 0.49 | 0.53 | 微型豁免 | 豁免 |
| llama-8b-prefill-bfloat16 | 84.530 | 64.430 | −23.78% | 31.16 | 37.88 | 算力 60% | 否 |
| llama-8b-prefill-float16 | 82.140 | 62.650 | −23.73% | 27.44 | 33.05 | 算力 60% | 否 |

### 3.1 相对本轮起点（R6 best）的净增量

| workload | R6 best µs | R7 final µs | Δ vs R6 |
|---|---:|---:|---:|
| llama-13b-prefill-float32 | 108.702 | 79.010 | **−27.31%** |
| llama-70b-prefill-float16 | 140.873 | 116.790 | **−17.10%** |
| llama-70b-prefill-bfloat16 | 144.733 | 120.400 | **−16.81%** |
| llama-8b-prefill-float16 | 63.251 | 62.650 | −0.95%（噪声内） |
| llama-8b-prefill-bfloat16 | 64.551 | 64.430 | −0.19%（噪声内） |
| llama-8b-decode-bfloat16 | 4.340 | 4.190 | −3.46%（豁免） |
| llama-70b-decode-bfloat16 | 4.820 | 4.210 | −12.66%（豁免） |

**对本轮目标的回答**：目标「prefill 在 current best 上再降 10–20%」——70b（双 dtype）−16.8~−17.1%、13b −27.3% 达成；**8b 未达成（tie）**，并附结构上限证据（§2.2 三条否证 + vec 0.756 / mte2 0.336 / 核间 imbalance 1.6%）。

## 4. 精度

- `python perf_opt/FusedAddLayerNormFwdOp.py --level all` → **`All check passed!`**（日志：`perf_opt/logs/r7/final_level_all.log`）。
- L0 13/13 PASS、L1 全过；L2/Boundary 为 WARN-only（`Boundary-extreme-affine` = weight/bias ±65504 的 fp16 溢出，R6 同源，非本轮引入）。
- 数值契约未改：round-first stats（先 `round_dtype(s)` 再进 fp32 统计）+ fp32 opmath + 单次 round-back；无算子签名 / 输出语义变化。
- 本轮合并版 L0 max_diff：bf16 3.125e-02、fp16 4.000e+00（仅 extreme-affine 溢出例，WARN）、fp32 2.861e-06。
- smoke 3 例（64×4096 fp32/fp16/bf16）仅精度回归：PASS。

## 5. 产物清单与哈希

| 文件 | 说明 | sha256 |
|---|---|---|
| `perf_opt/FusedAddLayerNormFwdOp.py` | **最终最佳**（R7 merged，签名 drop-in 不变） | `6b051de2958b086cbb6c5f23cb57681911af9ae3a327564b61adca96867528be` |
| `perf_opt/FusedAddLayerNormFwdOp_r7_loadsfirst_bm1.py` | op1 分支（per-input staging） | 见目录内文件 |
| `perf_opt/FusedAddLayerNormFwdOp_r7_fp32hoist.py` | op2 分支（fp32 affine 外提） | 同上 |
| `perf_opt/FusedAddLayerNormFwdOp_r7_pipe2.py` | op3 分支（tie，证据） | 同上 |
| `perf_opt/FusedAddLayerNormFwdOp_r7_storecast.py` | op4 分支（回退，证据） | 同上 |
| `perf_opt/FusedAddLayerNormFwdOp_r7_hoistbrc.py` | op5 分支（tie，证据） | 同上 |
| `perf_opt/FusedAddLayerNormFwdOp_r7_merged.py` | 合并候选（= final 的源文件） | 同上 |
| `perf_opt/opt_log.md` | 调优日志（含 R7 章节、候选对比表、Final 表、复盘） | — |
| `perf_opt/perf_records.jsonl` | 结构化记录（111 行；R7 新增 7 条 baseline + 32 条 candidate + 7 条 final = 46 条） | — |
| `perf_opt/workload_inventory.json` | 7 tune + 3 smoke，全部 `winner_merged` | — |
| `perf_opt/profiles/r7diag-*`、`profiles/{candidate,baseline}-r7-*` | 46 个 A/B 采集目录 + 3 个诊断 profile 目录（共 49） | — |
| `perf_opt/logs/r7/` | 本轮全部 stdout/stderr（L0、A/B 批跑、诊断 profile） | — |

簿记说明（须在合并后的报告里保留）：R6 的 7 条 `phase=final` 记录在 R7 收束时降级为 `phase=merged`（仅 phase 标签；round / candidate_id / duration_us / artifact_sha256 / msprof_raw_path 未改动），以满足「每个 tune workload 恰有一条属于当前最终候选的 final 记录」；降级前快照在 `/tmp/opencode/perf_records_before_r7_phasefix.jsonl`。处理方式与此前 R4→R6 的降级一致。

## 6. 门禁结果

- `python3 .agents/tools/statectl.py gate 4 --dir examples/TileOPs/tileops/kernels/norm/fused_add_layer_norm/fused_add_layer_norm_kernel`：见阶段返回的 `gate4` 行（只读执行）。
- 自检项：final 文件存在且签名 drop-in；L0/L1 通过；7 个 tune workload 均有 valid `msprof op` 记录（final 恰一条/ workload，同一候选）；`workload_inventory.json` 分类完整；`opt_log.md` 含候选 vs current best 对比表与 Final 表；无占位符。

## 7. 未竟项 / 天花板证据

1. **8b prefill 停在本设计族上限**（62.65/64.43µs，SOL 33.05/37.88%）：证据 = ① op5 去 2/12 全平面 pass 后 Task Duration 不动；② op3 task 级流水 tie；③ op4 round-back 折叠回退；④ vec 0.756 / mte2 0.336 / aiv_time 60.0 vs duration 62.7（核间与 launch 仅 4%）。剩余为每 task 依赖链延迟（2 次 row-reduce + 16 链接）。
2. **0/7 达 SOL 线**：prefill 需 60%（现 33–40.6%）、13b fp32 需 70%（现 60.66%）。要跨线需要换结构的量级收益（N 方向两段式分核，或把 row-reduce 搬到 Cube 的 Expert 重构），本轮预算内未立项，均已记录机制与阻塞点（CG-2026-0010）。
3. **不产出 `perf_feedback.md` / 不发 `[DESIGN_LIMIT]`**：`_shared/standards/perf-feedback.md` §1 双门槛未同时满足——归因是本设计族内的实现级修复（本轮仍拿到 −24~−29% 的基线相对收益），且不存在「被冻结但有 >2x 实测证据」的备选结构族。
4. **decode 两例**（4.19/4.21µs）属微型 shape，按既定判据豁免，未投入。
