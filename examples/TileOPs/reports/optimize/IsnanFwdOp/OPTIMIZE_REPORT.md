# 优化报告：IsnanFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `IsnanFwdOp`；单 kernel / 单 func / 纯 Vector（AIV），`y = isnan(x)`，bool 输出经 int8 字节存储 |
| kernel（基准，只读） | `tileops/kernels/elementwise/isnan/isnan_kernel/` |
| 调优产物 | `.../isnan_kernel/perf_opt/IsnanFwdOp.py`（`TUNED_BLOCK_SIZE = 8192`） |
| 口径 | `msprof op Task Duration(us)`（唯一时延口径）+ `visualize_data.bin` GM Read+Write SOL ratio；数据来自 `perf_opt/perf_records.jsonl` 的 round 5 final 与 round 6 核数扫描 |
| **工具链版本戳** | tilelang 0.1.2 dev root（AscendNPU-IR 31f6903）/ CANN 9.0.0（bishengir 1.1.0 + hivmc 0.2.0）/ Ascend910B3（20 AIC / 40 AIV）/ 2026-10-17（round 1–5）、2026-10-18（round 6 二次深挖） |
| 达标线 | 访存受限 **≥70%**（AI 与归类依据原文未记录；opt_log 记 AI 1.5–2.3、平衡点 5.18 ⇒ 访存受限） |
| benchmark | `benchmarks/ops/bench_isnan.py::test_isnan_bench`（7 tune + 3 smoke：`smoke-1m`） |
| 过程日志 | `.../isnan_kernel/perf_opt/opt_log.md`、`perf_records.jsonl`、`logs/`、`candidates/` |
| 前置报告 | 本文件 v0 已归档 `history_version/OPTIMIZE_REPORT_v0.md` |

## 1. 结论

- 一句话：**结构移植（满块静态 extent + 尾块 epilogue + `T.Pipelined(2)` + 3-pass vselect 链）拿到 **−4.08~−6.16%** 的确定性收益，但 7 个 tune case 仅 1 个过 70% 线；未达线归因为 MTE2 混合 R/W 带宽地板（结构上限，非 tiling 可解），不建议翻转 wrapper。**
- **达标数（双分母）**：可调 shape（7 tune）**1/7 ≥70%**（16m fp32 78.45%）；全 shape（含 3 个 `smoke-1m`，仅精度不参与性能判定）性能判定口径同为 **1/7**；豁免 **3**（`smoke-1m` fp32/fp16/bf16，依据：label 含 smoke、仅做精度回归，原文精度节明示不参与性能判定）。
- 关键收益（vs 基线，Task Duration）：16m-half bf16 **−6.16%**、16m bf16 **−5.62%**、16m-half fp16 **−4.08%**、256m bf16 **−1.01%**；256m fp16 **+0.55%**（噪声内）。
- 未达标分布：16m/256m fp16 55–65%，bf16 55–62%（原文结论节）。
- 结构：满块静态 extent + 尾块 epilogue + `T.Pipelined(2)` + grid-stride 40 核 + 3-pass vselect 链 @8192。
- **是否建议采纳 wrapper**：**否**。相对基线整体收益仅 0~−6.2%，且根因是硬件带宽地板（非实现可解）；perf_opt 版本与 baseline 在 fp16/256m 上打平甚至 +0.55%。是否采纳由 conductor 在回归通过后按「收益是否值得维护双版本」权衡（原文『二次深挖』采纳建议）。
- **9.0 端到端复测（2026-10-18，基线 wrapper，run `20261018_152024_774098_IsnanFwdOp`）**：正确性 **9/9**、report `passed`；5 个 perf shape **1/5 ≥70%**（16m fp32 79.33%），ratio 范围 **52.91–79.33%**，与 Stage-4 直跑结论一致；维持不采纳 perf_opt。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测 | 数值 | 结论 |
|---|---|---|
| 算力/访存归类 | 原文未记录（opt_log：AI 1.5–2.3、平衡点 5.18） | 访存受限，达标线取 70% |
| 瓶颈 pipe | fp16/fp32 `aiv_mte2_ratio` **0.81–0.90** ≫ `aiv_vec_ratio` **0.25–0.39** | **MTE2 数据搬运是瓶颈**，3-pass 向量链已完全隐藏于搬运窗口下，不是 compute/latency-bound |
| 减 pass 的真实收益 | 4→3-pass 使 `vec_time` 降 22%，但 Task Duration 仅 −3.4% | **pass 数不是杠杆**（`mte2_time` 不变）；bf16 是唯一例外（baseline 向量受限，故 3-pass 收益最大） |
| 有效带宽 | fp32(4:1) 1.70 TB/s、fp16(2:1) 1.63→1.24 TB/s(256m)、bf16 1.45→1.18 TB/s | 已贴近设备混合 R/W 地板；输入全读 + 输出 int8 1B 已是不可约流量下限 |
| 固定/流水开销 | 16m fp16 下限 `≈ mte2_time + 启动 ≈ 21.2 + 5.2 = 26.4µs`，当前 30.46µs | 理论余量 ≤14%，且需 MTE2 利用率 0.81→1.0（更深流水 UB 超限不可达）——属流水填充/排空开销 |
| 核数是否已最优 | 40 核 29.811µs（SOL ratio 63.997）vs 20 核 49.231µs（SOL ratio 38.829，+65%） | 搬运并行主导（减半核数 ≈ 减半有效带宽），40 核已最优 |
| 判据是否失配 | roofline `advice = latency bound:pipeline caused`，但 pipe 占比显示 MTE2 近饱和 | **是**：advice 会把「MTE2 混合 R/W 带宽地板」误读成「pipeline 延迟可优化」，须以 PipeUtilization 的 pipe 占比为准 |
| 噪声底 | ±3%（round 6 复测 29.811µs 与 r5 final 30.841µs 落在该噪声带内） | 单批 <3% 不可分辨；`smoke-1m` 4.4–5.6µs 由固定开销主导 |
| UB 容量边界 | block 16384 已超限；8192 为可行上限 | block 方向无余量，只能改 pass 数与映射 |

### 2.2 方法（做了什么、为什么这样做）

仅在 `perf_opt/` 内改动（基准与 wrapper 未动）：

1. **满块静态 extent + 尾块 epilogue**
   - 改了什么：热循环的 iteration extent 用编译期已知的满块数，尾块在热循环外单独处理，去掉循环内 `T.min` 与动态 extent。
   - 为什么有效：每 tile 少一组标量比较/裁剪，循环体更利于向量化；这是后续所有改动的基础结构。
   - 正确性依据：满块 + 尾块覆盖全部元素，与基线逐元素语义一致；`--level all` max_diff = 0。

2. **`T.Pipelined(num_local_tasks, num_stages=2)`**
   - 改了什么：tile 循环改为 2-stage 软件流水。
   - 为什么有效：用下一 tile 的 MTE2 搬运覆盖当前 tile 的 VEC 链，压缩流水填充/排空空隙（该算子的下限正是「`mte2_time` + 启动」）。
   - 正确性依据：不改变访问语义与写出顺序；round 1 合入后各 workload 普遍小幅下降（opt_log Round 1：−0.95~−3.91%，其中 16m-half fp16 +0.10% 为噪声）。

3. **A7：`vcmp(le) → vselect(mask, 0.0, 1.0, f16) → vcast(f16→i8)`，4 条向量 pass → 3 条**
   - 改了什么：用 `vselect` 一步产出 f16 的 0/1，替掉原来 `vcast(bool→f16)` + `vsub(1−m)` 两条 pass。
   - 为什么有效：向量链少一条 op，`vec_time` −22%；在 bf16 上（baseline 向量受限、`vec_t > mte2_t`）直接兑现为 −5.62~−6.16%，在 fp16/fp32 上因 MTE2 已饱和只兑现 −3.4%。
   - 正确性依据：语义 `mask=1`（非 NaN）→ 0.0、`mask=0`（NaN）→ 1.0，与原 4-pass 链等价；`--level all` max_diff = 0（NaN/±inf/−0 全覆盖）。
   - 约束依据：`CG-2026-0027`（无 `i1→i8` 直接 vcast、无 bool/int1 的 vnot、`T.vselect` 的 Out 不支持 int8），因此 3-pass 已是本工具链下限。

4. **`TUNED_BLOCK_SIZE = 8192` 提为模块级常量**
   - 改了什么：block 大小固定 8192 并作为模块级常量导出。
   - 为什么有效/依据：block 16384 已 UB 超限（round 3 采集失败：6 个 UB buffer × 16384、2-stage 后超 192KB），8192 是可行上限；常量形式便于 wrapper 成对引用。

### 2.3 逐轮记录

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| R1 | `pipe2`（结构移植） | 满块静态 extent + 尾块 epilogue + `T.Pipelined(2)` + grid-stride 40 核 | opt_log R1：−0.95~−3.91%（16m-half fp16 +0.10%，噪声） | ✅ 采纳为基础结构 |
| R2 | `select`（A7） | `vselect` 替 `vcast+vsub`，4 pass → 3 pass | `vec_time` −22%；Task Duration −3.4%；bf16 −5.62~−6.16% | ✅ 采纳（final 候选 `select-pipe2-bs8192`） |
| R3 | `block 16384` | block_size 8192 → 16384 | 采集失败：UB 超限（opt_log R3：6 buffer × 16384 × 2-stage > 192KB） | ❌ 否决：8192 为可行上限 |
| R5 | final 复测 | baseline/final 逐 workload 相邻交错、各 1 次独立 msprof | 7 个 tune workload 全量数据（见 §3） | ✅ 认定为 final |
| R6-OP1 | 核数扫描 40 / 20 | `num_cores` 40 → 20 | 40 核 29.811µs（SOL ratio 63.997）vs 20 核 49.231µs（SOL ratio 38.829，+65%） | ❌ 否决：40 核已最优 |
| R6-OP2 | block 12288 | block 8192 → 12288 | UB 不可行（单实例 96KB@8192→144KB@12288、1.7× multibuffer ≈245KB > 192KB，opt_log §6.2）；且 MTE2-bound 非 UB-bound，无收益 | ❌ 否决 |
| R6-OP3 | int8 直写 GM / 合并 vst | 省 1–2 条向量 pass | vec 已隐藏（0.28–0.39），不降 `mte2_time` | ❌ 否决：无收益（vec 非瓶颈） |
| R6-OP4 | 判据复核 | Task Duration 下限估算 | 非 compute-bound ⇒「3-pass × 向量吞吐」下限不适用，改按 MTE2 混合 R/W 地板 | ✅ 采纳（归因修正，无代码改动） |

**归因演进（保留被修正的旧结论）**：上一轮曾把 SOL 归因为「3-pass ISA 上限（CG-2026-0027）+ roofline `latency bound:pipeline caused`」；round 6 从 PipeUtilization 读 pipe 占比后修正为「**MTE2 混合 R/W 带宽地板**，3-pass 链已隐藏其下」——旧归因不精确，但结论（不可达 70%）不变。

## 3. 优化结果（msprof 对比表）

| workload | base µs | final µs | Δ | base SOL% | final SOL% | ≥70% |
|---|---:|---:|---:|---:|---:|:--:|
| 16m-bfloat16 | 36.861 | 34.791 | -5.62% | 51.13 | 55.26 | ❌ |
| 16m-float16 | 31.131 | 30.841 | -0.93% | 60.83 | 62.95 | ❌ |
| 16m-float32 | 49.831 | 49.301 | -1.06% | 77.34 | 78.45 | ✅ |
| 16m-half-bfloat16 | 37.031 | 34.751 | -6.16% | 51.32 | 55.35 | ❌ |
| 16m-half-float16 | 30.181 | 30.111 | -0.23% | 64.73 | 64.69 | ❌ |
| 256m-bfloat16 | 688.644 | 681.684 | -1.01% | 60.91 | 61.99 | ❌ |
| 256m-float16 | 644.903 | 648.443 | +0.55% | 65.49 | 64.99 | ❌ |

smoke：3 个 `smoke-1m`（label 含 smoke）仅做精度回归 PASS，不参与性能判定。

**测量口径与噪声**：round 5 baseline/final 相邻交错、各 1 次独立 `msprof op`；round 6 核数扫描同为相邻交错各 1 次。跨进程双态噪声 ±3%，`256m fp16 +0.55%` 与 `16m-half fp16 −0.23%` 落在噪声带内，不计为回退/收益。

## 4. 精度

- `python perf_opt/IsnanFwdOp.py --level all` → **`All check passed!`**，max_diff = 0（NaN/±inf/−0 全覆盖）。
- 3-pass vselect 链与原 4-pass 链在 L0 上逐位一致（opt_log §精度回归）。
- 3 个 `smoke-1m`（label 含 smoke）：仅精度回归，PASS，不参与性能判定。
- 关键边界：NaN（判 True）、±inf（判 False）、−0 全覆盖，max_diff = 0 即逐位一致。

## 5. 时间与成本

| 项 | 值 |
|---|---|
| 开始 / 结束（+08:00） | 2026-10-17 19:48 → 21:05 |
| 墙钟时长 | ≈ 77 min |
| token input/output/reasoning | 33,636 / 26,374 / 16,305 |
| cache-read / cost | 34,575,360 / $0.1344 |
| round 6 二次深挖（2026-10-18） | 起止时间、墙钟、token 与 cost 原文未记录 |

## 6. 产物

| 类别 | 路径 / 状态 |
|---|---|
| final kernel | `.../isnan_kernel/perf_opt/IsnanFwdOp.py`（sha256 见 round 5 final 行） |
| 调优记录 | `perf_opt/opt_log.md`、`Optimize.md`、`perf_records.jsonl`、`logs/`、`candidates/` |
| 候选文件 | `candidates/isnan_fwd_kernel_pipe2.py`、`candidates/isnan_fwd_kernel_select.py` |
| 能力缺口 | `.agents/evolution/capability-gaps.md` **CG-2026-0027** |
| gate 4 | `pass` |
| 算子状态 | `phase=DONE` |
| wrapper | 未翻转（原文『二次深挖』采纳建议：不建议采纳） |
| 端到端 report | **`reports/tileops/20261018_152024_774098_IsnanFwdOp/`（status=passed，9/9，1/5 ≥70%）**；原文未记录 |
| pattern-library 回写 | PL-1.23（CG-2026-0027 性能影响评估修正为「MTE2 地板钉死 SOL，3-pass 已隐藏」） |

## 7. 天花板 / 豁免 / 缺口

- **豁免项**：3 个 `smoke-1m`（fp32/fp16/bf16）——label 含 smoke、仅做精度回归，不参与性能判定，不计入达标分母。
- **ceiling 证据（round 6 复核后）**：
  - 判据复核：fp16/fp32 `aiv_mte2_ratio` 0.81–0.90 ≫ `aiv_vec_ratio` 0.25–0.39 ⇒ 真瓶颈是 **MTE2 混合 R/W 带宽地板**，「3-pass × 向量吞吐」下限不适用。
  - 流量不可约：输入全读 + 输出 int8 1B；有效带宽已随 R:W 与 N 退化到 fp32 1.70 / fp16 1.63→1.24 / bf16 1.45→1.18 TB/s。
  - 下限估算：16m fp16 `≥ mte2_time + 启动 ≈ 21.2 + 5.2 = 26.4µs`，当前 30.46µs，余量 ≤14%（需 MTE2 0.81→1.0，UB 超限不可达）。
  - 核数扫描（`msprof op`，16m fp16，相邻交错）：

    | num_cores | Task Duration(us) | SOL ratio | 判定 |
    |---:|---:|---:|---|
    | 40（current best） | **29.811** | 63.997 | 已最优 |
    | 20 | 49.231 | 38.829 | 回退 +65% |

  - 结构边界：block 16384 已 UB 超限；核数 40 已最优（20 核 +65%）。
  - 工具链约束：`CG-2026-0027`（无 `i1→i8` 直连 / 无 bool vnot / `vselect` Out 不支持 int8）使 predicate 链最少 3 条 pass，且该 3-pass 已非瓶颈。
- **贴线风险**：无 <3pp 贴线项；最高非达标项为 256m fp16 **64.99**、16m-half fp16 **64.69**（SOL%，见 §3），均在线下且受 ±3% 双态噪声支配，已如实标注、不做 shape 特判。
- **缺口**：唯一能力缺口为 `CG-2026-0027`（predicate 族 i1→i8 直连 / int8 vselect），性能影响已修正为「MTE2 地板钉死 SOL，3-pass 已隐藏」。
- **`[DESIGN_LIMIT]`**：不触发（无 >2× 结构加速、无设计假设矛盾），不产出 `perf_feedback.md`。

## 8. 建议

1. **不建议采纳 `perf_opt/IsnanFwdOp.py`**（原文『二次深挖』采纳建议）：整体收益 0~−6.2%，根因是硬件带宽地板而非实现可解；fp16/256m 上打平甚至 +0.55%，维护双版本不划算。
2. 若后续做工具链升级，需复核两项：设备混合 R/W 有效带宽（1.70 / 1.63→1.24 / 1.45→1.18 TB/s）与 `CG-2026-0027` 是否补齐（补齐后可复核 1–2 pass 压缩，但按本报告归因预期 SOL 几乎不动）。
3. **未解决项与原因**：16m/256m 的 fp16/bf16 55–65% gap 为 MTE2 混合 R/W 带宽地板，参数/tiling/核数均不可解；`smoke-1m` 4.4–5.6µs 属固定开销主导，按豁免处理。
