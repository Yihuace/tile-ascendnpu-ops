# Stage 4 调优日志 — `_adaptive_max_pool2d_kernel`（AdaptiveMaxPool2dFwdOp）

- task: `adaptive_max_pool2d-_adaptive_max_pool2d_kernel`（`scenario=optimize`，`mode=full`）
- 基准 kernel（只读）: `../_adaptive_max_pool2d_kernel.py`（Stage 3，精度通过）
- 最终产物: `perf_opt/_adaptive_max_pool2d_kernel.py`
- 唯一时延口径: **`msprof op` Task Duration(us)**（median / 20 launch，`--launch-count=20 --warm-up=5`）
- 环境: `source /workspace/env.sh`；Ascend910B3（`aicore_num=20` → `vector_cores=40`）；Developer 模式
  - tilelang `0.1.2+489385e9b98806365b585b6e1f984a5dae337154`（repo commit `489385e`）+ CANN 9.0.0（`bishengir-compile` 来自 CANN 9.0.0）
  - **注意**：本工具链与同族前作 `adaptive_avg_pool2d`（tilelang `8b16bcc…` / commit `417dae9`）**不同**；前作全部结论按 INDEX §2 版本戳规则视为**待重验**，本任务全部重新实测（见 §3）。
- 性能目标（DESIGN.md §11）: 等效 70% 口径（地板利用率 ≥70%）
  - `global-1x1` **322 µs → ≤ 8 µs**（以步骤 0 标定地板为准，70% 校准线 = 地板 ÷ 0.7）
  - `spp-6x6` / `nondiv-7x7` / `smoke`：best-effort
- 预算: `max_rounds=10`、`budget.max_stage4_experiments=40`，噪声阈值 3%
- 采集协议: 直接 kernel 入口（不经 Stage 5 wrapper）——
  `msprof op --kernel-name=main --output=… --launch-count=20 --warm-up=5 --dump=off --aic-metrics=… python bench.py --impl {impl} --case {workload_id} --iters 25`；
  分支执行走 `run_experiments.py`（T-4 批处理，逐测量追加 `perf_records.jsonl`）

---

## 1. Benchmark workload 清单（Phase 1）

展开入口 `benchmarks/ops/bench_pool.py::test_adaptive_max_pool2d_bench`（manifest-driven，op `AdaptiveMaxPool2dFwdOp`，
`tileops/manifest/pool.yaml`）；pytest id = `<label>-<dtype>`。完整清单见 `workload_inventory.json`。

| workload_id | shape | output_size | dtype | 实际 kernel 路径 | kind |
|---|---|---|---|---|---|
| `smoke-1x8x4x4-2x2-float16` | [1,8,4,4] | (2,2) | float16 | general | **smoke**（manifest 首条，label 独立含 `smoke` 词） |
| `global-1x1-float16` | [8,2048,7,7] | (1,1) | float16 | **C4 全局** | tune（优先） |
| `spp-6x6-float16` | [2,128,56,56] | (6,6) | float16 | general | tune |
| `nondiv-7x7-bfloat16` | [2,64,55,57] | (7,7) | bfloat16 | general | tune |

- 三条结构路径（恒等 / C4 全局 / 一般网格）分派**保持在 factory 层**，本任务未改动分层。
- **skipped**：无。
- **smoke 排除理由**：label 独立含 `smoke` → 按 `profile-collection.md §1` 归 smoke；只做精度回归，不采性能基线、不选 winner、不参与性能门禁。

### 调优顺序（依实测瓶颈排序）

1. `global-1x1-float16` —— baseline **320.83 µs**，是其余 workload 的 16 倍量级，profile 直指 C4 路径结构性缺陷（§3）→ 首位。
2. `spp-6x6-float16` —— baseline 20.37 µs，general 路径、vec-bound（vec 12.24 / aiv 18.45），但 `BlockDim=24`（40 核中 16 核空闲）→ 次位。
3. `nondiv-7x7-bfloat16` —— baseline 18.32 µs，general 路径 `BlockDim=14`（26 核空闲）→ 第三。

---

## 2. 步骤 0：copy-floor 标定（强制前置）

标定 kernel 见 `calib_bench.py`（纯拷贝、同栅格 / 同 DMA 粒度、1 元素落盘保活），
驱动 `calib_run.py`，结构化数据落 `copy_floor_calibration.jsonl`（**不写入 `perf_records.jsonl`**）。

| 标定 kernel | 栅格 / DMA 粒度 | CH | Task(us) | aiv(us) | mte2(us) | GM_to_UB(KB/block) | 70% 线 = 地板/0.7 |
|---|---|---:|---:|---:|---:|---:|---:|
| `global`（[8,2048,7,7] fp16，整平面连续搬运） | 40 block × CH×49 fp16，7 轮 | 64 | **6.85** | 4.89 | 2.13 | 36.8 | 9.79 µs |
| `global` 同上 | 同上，4 轮 | 128 | **5.09** | 2.96 | 1.21 | 36.8 | 7.27 µs |
| `global` 同上 | 同上，13 轮 | 32 | **10.99** | 9.19 | 3.85 | 40.6 | 15.70 µs |
| `general`（spp-6x6 栅格，逐 bin 行搬运） | 40 block × 行宽 56，2 波 | 32 | **9.58** | 4.68 | 3.77 | 40.0 | 13.69 µs |
| `general` 同上 | 24 block，1 波 | 64 | **8.87** | 7.44 | 6.71 | 80.0 | 12.67 µs |

**地板结论**

1. **global 类地板强依赖「每核串行轮数」而非带宽**：同一份 1.6 MB 流量，13 轮 10.99 µs → 7 轮 6.85 µs → 4 轮 5.09 µs。
   拷贝阶段 mte2 仅 3.85 / 2.13 / 1.21 µs，其余全部是**每轮 DMA→落盘 的串行延迟链**（`CONST-aiv-time-decomposition`：`Task ≈ max(mte2,vec)+scalar`）。
   即 global 路径的地板 = **轮数 × 每轮固定开销（≈0.7-0.9 µs/轮）**。
2. **一般网格地板 ≈ 8.9-9.6 µs**，同样由「波数 × 行搬运轮数」决定（spp 每单元 10 行 × 2 波）。
   该地板远高于任务书给的「≈4 µs」，说明引用值（借自同族 avg）未包含本栅格的轮数代价。
3. **70% 校准线（本轮标定为准）**：
   - global 类：地板 5.09 µs（CH=128 拷贝）→ 线 **7.27 µs**；地板 6.85 µs（CH=64）→ 线 **9.79 µs**（12.67 µs / 13.69 µs for general）
   - 任务书「≤8 µs」对应地板 ≈5.6 µs，与本轮 CH=128 标定地板 5.09 µs / CH=64 6.85 µs 同量级，**口径一致**。
4. 标定同时确认：纯拷贝下 `GM_to_UB` ≈ 输入实流量（无扇出放大）——**排除**了 baseline 的 336 KB/block 扇出是「带宽」问题（见 §3）。

> **标定修正留痕（append 说明）**：本标定首轮曾用 `tile[0:real_c,0:1,0:1]`（stride=plane 的列切片）落盘，
> 触发 `(M,1)` 列切片写放大（`UB_to_GM_datas` 52 KB/block，Task 93.95 µs），**该形态已被判定为无效标定并删除**，
> 改用「1 元素落盘保活」形态（同族 `probe_c4 copy` 已证方法）。修正后写盘 `UB_to_GM` ≈ 0。

---

## 3. 现象分析与根因（Phase 2 前置调查）

### 3.1 P0 —— PL-1.13（双 AIV subid 分片）在本工具链 **不适用于纯 Vector kernel**（实测证伪）

任务书要求「先验证 PL-1.13」。标定阶段的 subid 分片 A/B 曾给出「-20% ~ -41%」的假收益：

| 标定 | 默认 | subid 分片 | GM_to_UB(KB/block) |
|---|---:|---:|---:|
| global CH=32 | 11.22 µs | 6.66 µs | 40.6 → **18.8（≈半量）** |
| global CH=64 | 6.75 µs | 4.64 µs | 36.8 → **18.4（半量）** |

GM_to_UB 精确减半 ⇒ **分片版本只处理了一半逻辑单元**（不是并行加速）。定向探针 `probe_subid.py`（每 (cid,subid) 写 `1+subid` 到独立槽位）给出决定性证据：

```
nblk=4 out = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]   # 偶槽 1.0（subid=0），奇槽从不被写
nblk=6 out = [1.0, 0.0, ... 同形]
```

**结论：`T.Kernel(n, is_npu=True)` 的纯 Vector kernel 只有 1 个 Vector 子块，`subid` 恒为 0**；PL-1.13 的
「双 AIV 重复执行 + subid 分片」形态属 Mix kernel（`T.Scope("Vector")`）场景，**对本算子不适用**。
据此已从交付版本中移除全部分片脚手架（并在模块 docstring 留痕）。
corresponding 的假收益（-41%）作废，global 地板以**未分片**标定为准。

### 3.2 C4 全局路径根因

baseline `global-1x1` profile（median/20）：

| 指标 | 值 | 说明 |
|---|---:|---|
| Task Duration | 320.83 µs | baseline |
| aiv_time | 275.42 µs | 核忙时 |
| aiv_vec_time | 110.10 µs | 向量引擎忙 |
| aiv_mte2_time | 105.94 µs | GM→UB 跨度 |
| aiv_mte3_time | 0.87 µs | UB→GM |
| GM_to_UB_datas | **336 KB/block** | 真实输入 ≈40 KB/block → **~8.4x 扇出** |

**根因（两条，均已实测）**

1. **最内轴窄归约标量化 + 布局传播摧毁 DMA**：baseline 把整平面声明为 `(CH,H,W)` 并链式 `reduce_max(dim=2)`
   （W=7，非 8 倍数）再 `reduce_max(dim=1)`（H=7）。7 宽非向量整数倍的归约被逐元素标量化，
   且归约/比较消费者把 `T.copy` 的 UB 目标推成非常规布局 → `GM_to_UB` 36.75 → 336 KB（`TRAP-copy-layout-propagation-dma-amplification`
   同形，本工具链重现）。
2. **NaN 追踪链 UB 成本**：baseline 的 `vcmp(平面,平面) → vcast → reduce_min` 三件套（外加 fp32 平面中转）
   在 BishengHIR 归约 lowering 上按 **96-128 B / 平面元素** 记账，直接把可用 CH 压到 32。

### 3.3 一般网格路径根因

baseline `spp`：`BlockDim=24`（40 核中 16 核空闲），`aiv=18.45 / vec=12.24 / mte2=4.59 / mte3=1.03`
—— aiv ≈ vec+mte2+mte3 ⇒ **三段完全串行**；vec 主导（窗口 `vmax` + NaN `vmul` 每窗口元素 2 条向量 op）。
`nondiv`：`BlockDim=14`（26 核空闲），同形。

### 3.4 设计估算 vs 实测偏差（D-2 回填）

DESIGN.md §1.6.0「设计期估算（D-2 roofline）」：

| workload | DESIGN 估算下界 | baseline 实测 Task | 偏差 | 失准项归因 |
|---|---:|---:|---:|---|
| `global-1x1` | ≈1.3 µs | 320.83 µs | **247x** | ① 流量项把「HBM 混合地板」当成可达时延下界，**漏掉 kernel 固定开销与串行轮数**（本轮标定：同流量纯拷贝地板 5.09-10.99 µs 即已 4-8x 于估算下界）；② 未计「最内轴窄归约标量化 + 布局传播扇出 8.4x」 |
| `spp-6x6` | ≈1.3 µs | 20.37 µs | **15.7x** | 发射项按 ~0.5 µs/op 外推被设计文档自标「未文档化假设」；实测 vec 忙时 12.24 µs 已 9.4x 于估算下界 |
| `nondiv-7x7` | ≈0.7 µs | 18.32 µs | **26x** | 同上；且 `BlockDim=14`（26/40 核空闲） |

结论：DESIGN 的估算行对**纯 Vector、窄归约、窗口选择类**域外推失准（固定开销/串行轮数与归约 lowering 成本均未标定）。
该偏差行是 `[DESIGN_LIMIT]` 归因的量化证据（§8 判定为**不触发**）。

---

## 4. Iteration Log

### Round 1（base = Stage 3 baseline）

#### Diagnostic Context

- kernel_id: `…/_adaptive_max_pool2d_kernel.py::_adaptive_max_pool2d_kernel::main`
- current_best: `../_adaptive_max_pool2d_kernel.py`
- Profile Facts（baseline）：见 §3.2 / §3.3

#### Current Phenomena

- P1（global-1x1）：`GM_to_UB` 336 KB/block（8.4x 扇出）+ vec 110 µs（窄归约标量化）→ C4 路径结构性缺陷。
- P2（nondiv）：`BlockDim=14` < 40 → 26 核空闲。
- P3（spp）：`BlockDim=24` < 40 → 16 核空闲；vec 占 aiv 66%。

#### Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| G1 | P1 | C4 路径改**扁平空间轴**：`x` 声明 `(N,C,1,H*W)`，两级窄归约 → 单次宽归约 | §3.2；同族 `PL-1.23`（待重验，本任务重测） | 重写 `is_global` 分支 | Task / GM_to_UB / max_diff | **improved（采纳）** |
| G2 | P1 | C4 NaN 追踪改 `isnan(reduce_sum(x*x))`（**原位**平方） | §3.2 之 2；UB 成本 96-128 → 32 B/元素 | `vmul(flat,flat,flat)` + `reduce_sum` + `vcmp(sum,sum,"eq")` | Task / UB / max_diff | **improved（采纳）** |
| G3 | P1 | C4 UB 预算按**实测归约成本模型**（fp16 路径 32 B/元素）反推 CH | ch=128 触发 UB overflow（实测 201216 B） | `_UB_REDUCE_B_PER_ELEM` | 编译通过 + Task | **improved（采纳）** |
| G4 | P2+P3 | general 路径**单波 CH 分派**（最小可行 CH） | `units = N·out_H·ceil(C/CH)` 决定波数与最长块 | 新 `_pick_ch_general` | Task / BlockDim | **improved（采纳）** |
| G5 | P1 | C4 `T.Pipelined(2/3)` 于 unit 循环 | 想重叠 DMA 与归约 | 循环构造替换 | Task | **config_no_gain** |
| G6 | P1+P3 | PL-1.13 subid 分片 | 任务书要求先验证 | uid 按 subid 分片 | Task / GM_to_UB / L0 | **blocked（L0-3 失败 + 探针证伪，见 §3.1）** |

#### Experiment Branches

| branch | base | opt_id | major_change | L0 | Task(us) | status |
|---|---|---|---|---|---:|---|
| `_round1_gflat32` | baseline | G1 | 扁平 + **fp32** 域归约，CH=32 | pass | global **36.771** | improved |
| `_round1_gf32` | baseline | G1+G2+G3 | 扁平 + 原位平方 NaN，CH=32 | pass | global **19.300** | improved |
| `_round1_gf64` | baseline | 同上 CH=64 | pass | global **19.160** | **winner** |
| `_round1_gf96` | baseline | 同上 CH=96 | pass | global 21.020 | config_no_gain |
| `_round1_gf120` | baseline | 同上 CH=120 | pass | global 21.070 | config_no_gain |
| `_round1_gf64_pipe2` | `gf64` | G5 | `Pipelined(2)` | pass | global 18.950 | config_no_gain（-1.1%，噪声内） |
| `_round1_gf64_pipe3` | `gf64` | G5 | `Pipelined(3)` | pass | global 18.960 | config_no_gain |
| `_round1_gen32` | baseline | G4 | general CH=32 固定 | pass | spp 29.471（+44%）/ nondiv 15.450 | 一方改善一方回退 |
| `_round1_gen48` | baseline | G4 | general CH=48 固定 | pass | spp 18.970 / nondiv 17.070 | 一方改善一方回退 |

#### 候选 vs current best 对比表（B2 结构化回流，数据源 `perf_records.jsonl` + raw profile）

| kernel_id | workload_id | candidate | Task(us) | vs baseline | aiv(us) | vec(us) | mte2(us) | mte3(us) | GM_to_UB(KB/blk) | BlockDim | L0 | 判定 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| main | global-1x1-float16 | **baseline** | 320.826 | 0.0% | 275.42 | 110.098 | 105.944 | 0.873 | 336.0 | 40 | pass | 待优化 |
| main | global-1x1-float16 | gflat32（fp32 归约） | 36.771 | **-88.5%** | 34.546 | 16.337 | 9.268 | 1.655 | 52.0 | 40 | pass | improved |
| main | global-1x1-float16 | **gf64** | **19.160** | **-94.0%** | 15.318 | 5.054 | 9.096 | 0.786 | 48.0 | 40 | pass | **winner** |
| main | global-1x1-float16 | gf32 | 19.300 | -94.0% | 17.080 | 6.081 | 9.965 | 1.692 | 52.0 | 40 | pass | tie→rollback |
| main | global-1x1-float16 | gf96 | 21.020 | -93.4% | 16.017 | 4.881 | 9.776 | 0.788 | 48.0 | 40 | pass | rollback |
| main | global-1x1-float16 | gf120 | 21.070 | -93.4% | 16.038 | 6.017 | 9.611 | 0.762 | 46.0 | 40 | pass | rollback |
| main | global-1x1-float16 | gf64_pipe2 | 18.950 | -94.1% | 15.333 | 5.054 | 9.105 | 0.791 | 48.0 | 40 | pass | rollback（-1.1%，噪声内） |
| main | spp-6x6-float16 | **baseline** | 20.370 | 0.0% | 18.446 | 12.239 | 4.594 | 1.026 | 80.0 | 24 | pass | 待优化 |
| main | spp-6x6-float16 | gen32 | 29.471 | +44.7% | — | — | — | — | — | 40 | pass | rollback（两波） |
| main | spp-6x6-float16 | gen48 | **18.970** | **-6.9%** | 17.041 | 11.343 | 3.933 | 0.994 | 60.0 | 36 | pass | **winner** |
| main | nondiv-7x7-bfloat16 | **baseline** | 18.320 | 0.0% | 16.880 | 11.607 | 4.363 | 0.742 | 72.0 | 14 | pass | 待优化 |
| main | nondiv-7x7-bfloat16 | gen32 | **15.450** | **-15.7%** | 13.787 | 10.125 | 2.286 | 0.697 | 36.0 | 28 | pass | **winner** |
| main | nondiv-7x7-bfloat16 | gen48 | 17.070 | -6.8% | — | — | — | — | — | 28 | pass | rollback（块失衡 48/16） |

#### 当前 Kernel 全部 Tune Workload 合并检查（`phase=merged`，round 1）

| merged_candidate | workload | baseline_us | merged_us | precision | status |
|---|---|---:|---:|---|---|
| `_round1_merged.py` | global-1x1-float16 | 320.826 | 19.270 | bit-exact (0.0) | pass |
| `_round1_merged.py` | spp-6x6-float16 | 20.370 | 20.420 | bit-exact (0.0) | pass（-0.2%，噪声内） |
| `_round1_merged.py` | nondiv-7x7-bfloat16 | 18.320 | 15.450 | bit-exact (0.0) | pass（-15.7%） |

smoke 精度（合并版）：`smoke-1x8x4x4-2x2-float16` L0 pass（max_diff 0.0）——只列精度，不填时延。

#### Iteration Winner

- winner（global-1x1）: G1+G2+G3 → 扁平 + 原位平方 NaN + CH=64 → **19.16 µs（-94.0%）**
- winner（nondiv）: G4 → 常量 CH=32 → **15.45 µs**
- winner（spp）: G4 → 常量 CH=48 → **18.97 µs**（但 CH 需按 shape 分派，见 Round 2）
- new_current_best: `_round1_merged.py`（CH_GLOBAL=64, CH_GENERAL=64 + 常量分派）

### Round 2（base = round 1 merged）

#### Diagnostic Context

- current_best: `_round1_merged.py`（global 19.27 / spp 20.42 / nondiv 15.45）
- 剩余现象：spp 需要 CH=48、nondiv 需要 CH=32 —— round 1 的两条常量分支各自只对一半 workload 有效，
  说明 CH 的判据不是「C 大小」而是**「最小可行 CH 使 `units ≤ 物理核数`」**（同时最小化最长块）。

#### Current Phenomena

- P4：`_pick_ch_general` 原实现（「折半且仍单波才折半」）对 spp 给出 64（24 units，16 核空闲），对 nondiv 给出 32。
  正确判据应是「**取最小可行 CH**」：spp → CH=48（36 units，最长块 48）；nondiv → CH=32（28 units，最长块 32）。

#### Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| G7 | P4 | `_pick_ch_general` 改为「最小可行 CH 扫描」 | round 1 gen32/gen48 两个数据点的交叉 | `for ch in 8..base: if units<=cores: return ch` | Task / BlockDim | **improved（采纳）** |

#### 候选 vs current best 对比表（B2 结构化回流）

| kernel_id | workload_id | candidate | Task(us) | vs round1 merged | aiv(us) | vec(us) | mte2(us) | mte3(us) | GM_to_UB(KB/blk) | BlockDim | L0 | 判定 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| main | global-1x1-float16 | **round1_merged（best）** | 19.270 | 0.0% | 15.401 | 5.054 | 9.154 | 0.786 | 48.0 | 40 | pass | 保持 |
| main | global-1x1-float16 | **round2_merged** | **19.010** | **-1.3%** | 15.226 | 5.054 | 9.115 | 0.790 | 48.0 | 40 | pass | **winner（噪声内保持）** |
| main | spp-6x6-float16 | **round1_merged（best）** | 20.420 | 0.0% | 18.468 | 12.239 | 4.578 | 1.058 | 80.0 | 24 | pass | 待优化 |
| main | spp-6x6-float16 | **round2_merged** | **19.020** | **-6.9%** | 16.954 | 11.343 | 3.883 | 1.084 | 60.0 | 36 | pass | **winner** |
| main | nondiv-7x7-bfloat16 | **round1_merged（best）** | 15.450 | 0.0% | 13.787 | 10.125 | 2.286 | 0.697 | 36.0 | 28 | pass | 保持 |
| main | nondiv-7x7-bfloat16 | **round2_merged** | 15.540 | -0.6% | 13.845 | 10.125 | 2.279 | 0.660 | 36.0 | 28 | pass | **winner（噪声内保持）** |

smoke 精度（round 2）：L0 pass（max_diff 0.0）。

#### Iteration Winner

- winner: G7（spp -6.9%，其余不回退）；new_current_best = `_round2_merged.py`

### Round 3（base = round 2 merged）

#### Diagnostic Context

- current_best: `_round2_merged.py`（global 19.01 / spp 19.02 / nondiv 15.54）
- 剩余现象：global 的 `mte2 = 9.10 µs`（1.92 MB → **211 GB/s**）远低于同粒度拷贝标定的 690 GB/s；
  spp 的 `aiv ≈ vec + mte2 + mte3`（三段串行）。

#### Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| G8 | global mte2 | C4 **DMA 大块 + 归约分块**：一次搬 CH×SUB 通道，归约仍在 CH 通道静态子块上做 | 想同时拿到「大 DMA 粒度」与「小归约 UB」 | `C4_DMA_SUB` + 静态子块 reduce | Task / GM_to_UB | **config_no_gain** |
| G9 | spp 三段串行 | general unit 循环 `T.Pipelined(2/3)` | 期望重叠 DMA 与向量 | 循环构造替换 | Task | **config_no_gain** |

#### 候选 vs current best 对比表（B2 结构化回流）

| kernel_id | workload_id | candidate | Task(us) | vs round2 merged | aiv(us) | vec(us) | mte2(us) | mte3(us) | GM_to_UB(KB/blk) | BlockDim | L0 | 判定 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| main | global-1x1-float16 | **round2_merged（best）** | 19.010 | 0.0% | 15.226 | 5.054 | 9.115 | 0.790 | 48.0 | 40 | pass | 保持 |
| main | global-1x1-float16 | c4dma_48x2 | 21.270 | **+11.9%** | 16.260 | 5.039 | 9.789 | 0.783 | 48.0 | 40 | pass | **rollback** |
| main | global-1x1-float16 | c4dma_56x2 | 19.940 | **+4.9%** | 16.997 | 5.808 | 10.132 | 0.755 | 56.0 | 40 | pass | **rollback** |
| main | spp-6x6-float16 | **round2_merged（best）** | 19.020 | 0.0% | 16.954 | 11.343 | 3.883 | 1.084 | 60.0 | 36 | pass | 保持 |
| main | spp-6x6-float16 | genpipe2 | 18.970 | -0.3% | — | — | — | — | — | 36 | pass | **rollback（噪声内）** |
| main | spp-6x6-float16 | genpipe3 | 18.860 | -0.8% | 16.982 | 11.343 | 3.911 | 1.044 | 60.0 | 36 | pass | **rollback（噪声内）** |

smoke 精度（round 3）：L0 pass（max_diff 0.0）。

#### Iteration Winner

- winner: **无**（G8 均回退；G9 全部落在 3% 噪声内）
- new_current_best: 不变（`_round2_merged.py`）
- **中止判定：plateau** —— round 2 取得 6.9% 有效提升后，round 3 无任何 valid 分支超过 3% 噪声阈值。
- **参照结构 diff 检查（T-1 morph ladder，plateau 前强制）**：检索 `pattern-library/cases.md` 参考实现集与
  `examples/TileOPs` 同族（pool 族已有 7 个迁移/调优档案），最接近的形态参照为
  **PL-1.23（扁平空间轴归约）+ PL-1.22（max NaN 追踪）+ PL-1.21（纯 Vector persistent 分核）+ PL-1.4（容量/单波驱动 CH）**。
  逐项结构 diff（本 kernel vs 参照）：
  - grid 组织：一维 persistent，一致；
  - 数据流分相位：单相位，一致；
  - 片上驻留策略：全 UB，一致；
  - **向量轴布局**：参照 PL-1.22 用「核内二轴转置把 C 变最内轴」；本 kernel 实测（§3.2）该路线在当前工具链对
    resnet-global 尺度被降级为 UB→GM→UB 往返（同族前作亦在同族工具链上得到同一结论），故改用「保持原生内存序 +
    扁平空间轴 + 最内 H*W 宽归约」——**该差异已在 Round 1 以 G1 实验分支验证（320.83 → 19.16 µs）**；
  - **NaN 追踪**：参照 PL-1.22 的合法式（`vcmp eq` 掩码 + `reduce_min`）在本 kernel 的 C4 路径上 UB 成本 96-128 B/元素，
    本 kernel 改用 `reduce_sum(x*x)` 的 `isnan` 等价形（数学等价：`x*x` 仅 NaN 处为 NaN 且其余 ≥0 ⇒ 求和不会因 `+inf-inf`
    假阳性）——**该差异已在 Round 1 以 G2 验证（36.77 → 19.16 µs）**。
  - 差异点即全部候选结构变异，均已实验覆盖；**无进一步可移植的结构差异** → 允许进入终止。
- **实验计数**：round 1 = 9 分支、round 2 = 1 分支（合并复测）、round 3 = 4 分支，合计 14 个实验分支（含标定探针另计），
  在 `budget.max_stage4_experiments=40` 内。

---

## 5. Autotune Log

未使用 TileLang autotune API：全部候选都是**结构性**变更（布局/归约形态/NaN 追踪式/分核判据），参数面只有
`CH_GLOBAL`（32/64/96/120）、`C4_DMA_SUB`（1/2）、`CH_GENERAL`（32/48/64）与 pipeline 阶数（2/3），
已用等价手动粗搜 + 邻域检查覆盖：

- search space：`CH_GLOBAL ∈ {32,64,96,120}`（128 触发 UB overflow：实测需 201216 B > 196608 B）；
  `C4_DMA_SUB ∈ {1,2}`；`CH_GENERAL ∈ {32,48,64}`；`num_stages ∈ {2,3}`
- best config：`CH_GLOBAL=64`、`C4_DMA_SUB=1`、`CH_GENERAL=64`（由 `_pick_ch_general` 派生实际块：
  spp→48、nondiv→32、smoke→8）
- 邻域检查：`CH_GLOBAL` winner 64 的两侧 32/96/120 均已测；`CH_GENERAL` winner 的上限 32/48/64 均已测
- winner 的 `msprof op` 独立复测：`phase=merged` round2 → global 19.010 / spp 19.020 / nondiv 15.540；
  `phase=final` round3 → **global 18.860 / spp 18.890 / nondiv 15.460**（复测差 ≤0.8%，噪声内一致）
- 编译失败配置数：2（`CH_GLOBAL=128`、`C4_DMA_SUB=2 且 CH=64` 的 UB overflow），其余配置 L0 全过

---

## 6. 设计层天花板判定（`[DESIGN_LIMIT]`，结论：**不产出 `perf_feedback.md`**）

对照 `_shared/standards/perf-feedback.md` §1 的双条件：

| 条件 | 判定 | 依据 |
|---|---|---|
| ① 天花板由**算法/设计层**决定（非 tiling/参数可解） | **不满足** | global 当前 18.86 µs 与同栅格纯拷贝标定地板 5.09–6.85 µs 的差距（≈2.8–3.7x）可归因于**工具链 lowering 与 DMA 调度**（BishengHIR 归约 lowering 按 32 B/平面元素记账、减小 CH 上限；reduce 与 DMA 不重叠，mte2 实测 211 GB/s vs 拷贝标定 690 GB/s），且已在本 Stage 用**布局/归约形态/分块**取得 17.0x 改进；剩余缺口无算法级障碍 |
| ② 结构性加速估计 > 2x 或实测与设计假设直接矛盾 | 部分成立 | 与 DESIGN §1.6.0 的「估算下界 ≈1.3 µs」**直接矛盾**（实测纯拷贝地板即 5.09–10.99 µs）；但该矛盾已在 Stage 4 内被消化（G1/G2 换布局与 NaN 追踪式），不构成「不可达」天花板 |

→ **双条件未同时满足 ⇒ 禁止产出 `perf_feedback.md`**（按 §1 末段）。
设计假设被实测推翻的部分以**附录级事实**记录在本日志 §3.4 与 pattern-library 回写中；
若 conductor 认为 DESIGN §1.6.0 的 roofline 口径需要正式修订，请按 `perf-feedback.md` §3 路由表决策
（本 agent 不回退 Stage 1）。

### 6.1 达标的诚实说明（目标未达成）

- `global-1x1`：**18.86 µs**，目标 **≤ 8 µs 未达成**。8 µs 对应本轮标定的拷贝地板 5.09–6.85 µs（70% 线 7.27–9.79 µs），
  即目标要求 kernel 逼近纯拷贝极限；实测与地板的 2.8–3.7x 缺口由「归约 lowering 的 UB 成本（限制 CH ≤ 64 可用档）
  - reduce 与 DMA 不重叠」构成，均属工具链层。
- `spp-6x6` / `nondiv-7x7`：best-effort 未达 70% 线（18.89 / 15.46 vs 本轮标定 general 地板 8.87–9.58 µs → 线 12.67–13.69 µs），
  距线 1.1–1.5x。
- **报告 `Ratio` 如实记录、不作达标判据**（池化族 compute-roofline 上限 ≈9–15%，见同族前作 `opt_log.md §12`；
  本轮未采官方 report，故不填 Ratio 值）。

---

## 7. Final Summary

### Final Performance Test Data

`phase=final`，round 3，同一最终候选 `_adaptive_max_pool2d_kernel`（下表仅含 tune workload；smoke/skip 见下节）。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/pool/adaptive_max_pool2d/adaptive_max_pool2d_kernel/_adaptive_max_pool2d_kernel.py::_adaptive_max_pool2d_kernel::main | global-1x1-float16 | 320.826 | 18.860 | _adaptive_max_pool2d_kernel |
| tileops/kernels/pool/adaptive_max_pool2d/adaptive_max_pool2d_kernel/_adaptive_max_pool2d_kernel.py::_adaptive_max_pool2d_kernel::main | spp-6x6-float16 | 20.370 | 18.890 | _adaptive_max_pool2d_kernel |
| tileops/kernels/pool/adaptive_max_pool2d/adaptive_max_pool2d_kernel/_adaptive_max_pool2d_kernel.py::_adaptive_max_pool2d_kernel::main | nondiv-7x7-bfloat16 | 18.320 | 15.460 | _adaptive_max_pool2d_kernel |

提升：`global-1x1-float16` **-94.1%（17.0x）**；`spp-6x6-float16` **-7.3%**；`nondiv-7x7-bfloat16` **-15.6%**。

- 最终文件：`perf_opt/_adaptive_max_pool2d_kernel.py` —— 与全部 `phase=final` 记录的 `artifact_path`（相对
  `perf_opt/`）/`artifact_sha256` 一致（`perf_records.jsonl` 中 final 行 `artifact_path = "_adaptive_max_pool2d_kernel.py"`，
  按 gate 4 的口径规则相对 `perf_records.jsonl` 所在目录解析）。
- 全部 final 记录均在同一最终文件上重新采集（`profiles/final/round3/_adaptive_max_pool2d_kernel_*/`）。
- 逐 workload 归因（msprof 证据）：

| workload | baseline | final | 归因（profile 证据） |
|---|---:|---:|---|
| global-1x1 | 320.826 µs（aiv 275.42 / vec 110.10 / mte2 105.94 / GM_to_UB 336 KB/blk） | 18.860 µs（aiv 15.18 / vec 5.05 / mte2 9.10 / GM_to_UB 48 KB/blk） | 扁平空间轴消除窄归约标量化与布局传播扇出（GM_to_UB 336→48 KB，-86%）；原位平方-sum NaN 追踪把归约 UB 成本 96-128→32 B/元素，解除 CH 上限至 64；vec 110.10→5.05 µs（-95%） |
| spp-6x6 | 20.370（aiv 18.45 / vec 12.24 / mte2 4.59 / BD 24） | 18.890（aiv 16.98 / vec 11.34 / mte2 3.91 / BD 36） | 单波 CH 分派 64→48：逻辑单元 24→36，占用核数 24→36，mte2 4.59→3.91 µs |
| nondiv-7x7 | 18.320（aiv 16.88 / vec 11.61 / mte2 4.36 / BD 14） | 15.460（aiv 13.82 / vec 10.13 / mte2 2.30 / BD 28） | 单波 CH 分派 64→32：逻辑单元 14→28，占用核数 14→28，mte2 4.36→2.30 µs |

### 校准线对账（§2 标定地板 ÷ 0.7）

| 类别 | 标定地板(us) | 70% 校准线(us) | final(us) | 距线 |
|---|---:|---:|---:|---:|
| global（CH=64 拷贝） | 6.85 | 9.79 | 18.86 | 1.93x |
| global（CH=128 拷贝，本算子不可达档：归约 UB 限制） | 5.09 | 7.27 | 18.86 | 2.59x |
| general（spp 栅格 CH=32 拷贝） | 9.58 | 13.69 | 18.89 | 1.38x |
| general（CH=64 拷贝） | 8.87 | 12.67 | 15.46 | 1.22x |

### smoke / skip

| workload_id | kind | 最终处理 | 精度 |
|---|---|---|---|
| `smoke-1x8x4x4-2x2-float16` | smoke | 每次合并后及最终仅做精度回归，无性能记录 | **pass**（`precision_pass: true`，max_diff 0.000e+00） |
| — | skipped | 无 | — |

### 最终产物自检

- `python perf_opt/_adaptive_max_pool2d_kernel.py --level all`：**L0 ALL PASS（8 shapes + 3D + NaN/Inf + 契约守卫）/
  L1 ALL PASS / L2 全 PASS / Boundary 全 PASS**，`Summary max_diff per dtype: bfloat16=0.000e+00, float16=0.000e+00`
  （**逐位精确**，atol=rtol=0）。
- 三条结构路径（恒等 / C4 全局 / 一般网格）分派保持在 factory 层；三条路径在 L0/L1/L2 中均有独立触发用例并通过。
- 导出（`__all__`）：`ASCEND_MODE` / `_adaptive_max_pool2d_kernel` / `adaptive_bin` / `max_adaptive_bin_extent` /
  `golden_adaptive_max_pool2d` / `CH_IDENTITY` / `CH_GLOBAL` / `CH_GLOBAL_F32` / `CH_GENERAL`；
  factory 签名与 baseline 完全一致（`(n,c_in,h_in,w_in,out_h,out_w,dtype="float16")` → `build(block_m=None)` → `main(x)->out`，`out_idx=[1]`）。
- **perf_records.jsonl 字段更正（如实披露，append-only 例外）**：3 条 `phase=final` 记录初次写入时
  `artifact_path` 误写为 `perf_opt/_adaptive_max_pool2d_kernel.py`，而 gate 4（`gate_lint.py`
  `S4-PERF-RECORDS-RECON`）按「相对 `perf_records.jsonl` 所在目录」解析该字段，会得到
  `perf_opt/perf_opt/…` 而判定 final 未测最终文件。本次仅就地更正该**字段文本**为
  `_adaptive_max_pool2d_kernel.py`（解析后即 `perf_opt/_adaptive_max_pool2d_kernel.py`），
  **未改动任何测量值（`duration_us` / `msprof_raw_path` / `artifact_sha256` / `timestamp` 均保持原样）**，
  且不涉及任何新增/删除记录。除此之外 `perf_records.jsonl` 全程只追加。

### 已知残留（如实披露）

- `probe_ub.py` 的 lvl3/lvl4 探针存在**死代码消除**导致的 UB 读数低估（`reduce_min` 结果未被消费），
  该探针结论已被 `probe_ub2.py`（无死代码的精确镜像）取代；本日志只引用 `probe_ub2.py` 的数据。
  对最终产物**无影响**（最终 kernel 的 UB 模型由 `probe_ub2` + 真实编译边界实测确定）。
- 交付文件中保留了 3 个**未使用**的 `_UB_*` / `CH_GLOBAL_F32` 常量语义分支（bf16 走 fp32 归约域），
  均为活跃路径（bf16 全局池化用例 `L1/bf16/global` 覆盖），非残留物。

### 总体结论与中止原因

- `global-1x1` 取得 **17.0x** 结构性命中（C4 路径从「两级窄归约 + 平面自比较 NaN 掩码」改为
  「扁平空间轴单次宽归约 + 原位平方-sum NaN 追踪 + CH=64」）；general 路径以单波 CH 分派取得
  spp -7.3% / nondiv -15.6%。
- **中止原因：`plateau`**（round 2 结构提升后，round 3 无 >3% 的有效提升；主要结构候选已充分验证并完成参照结构 diff 检查）。
- **性能目标未完全达成，如实报告**（见 §6.1）；剩余缺口属工具链 lowering / DMA 调度层，不触发 `[DESIGN_LIMIT]`。

---

## 8. Skill Retrospective

### 8.1 skill 流程问题

| # | 问题 | 证据 | 建议 |
|---|---|---|---|
| R1 | **预注入的 PL-1.13 在本算子上不适用，且其「伪收益」极具迷惑性**：任务书要求「先验证 PL-1.13」，标定阶段的 subid 分片给出 -20~-41% 的漂亮数字，但 `GM_to_UB` 同步减半暴露了它是「只做了一半工作」。skill Phase 0 的「首轮必查项」未要求对**跨核/跨子块类模式**做「工作量守恒核验」（GM 流量、指令数、输出覆盖率任一）。 | §3.1 标定表（36.8→18.4 KB）+ `probe_subid.py` | R 类：Phase 0 必查项增加「分片/并行类模式必须先做工作量守恒核验（流量/覆盖）再谈收益」 |
| R2 | **编译器 UB 报错无法直接得到「成本模型」**：`probe_ub.py` 的累积式 level 探针被**死代码消除**欺骗（未消费的 `reduce_min` 被删），给出 32 B/元素 的错误模型。修正为「精确镜像 + 全部结果被消费」后才得到真实的 128 B/元素（bool 掩码链）/ 32 B/元素（原位平方）。skill 未把「UB/容量探针必须无死代码」写成纪律。 | §7「已知残留」；`probe_ub.py` vs `probe_ub2.py` 的读数差（201216 B @ ch=128 vs 401792 B @ ch=64） | R 类：`bottleneck-patterns.md` 增「容量/UB 探针的 DCE 免疫规则」 |
| R3 | **`msprof op` 的 `Task Duration` 无法区分「计算」与「串行延迟链」**：global 地板的真正驱动是「每核串行轮数」而非带宽（同流量 13 轮 10.99 µs vs 4 轮 5.09 µs）。skill 的 copy-floor 方法学建议增加「**轮数自变量扫描**」（同流量、不同轮数）以分离固定开销与流量。 | §2 标定表 | R 类：`PL-1.6-copy-floor` 增「轮数扫描」维度 |

### 8.2 价值点（vp_type + 证据三件套）

| id | vp_type | 价值点 | evidence | repro | toolchain_stamp | 回写位置 |
|---|---|---|---|---|---|---|
| VP-1 | **D** | **PL-1.13 不适用于纯 Vector npuir kernel**：`T.Kernel(n, is_npu=True)` 只有 1 个 Vector 子块，`subid` 恒为 0；按 subid 分片会**静默丢弃一半逻辑单元**（`GM_to_UB` 精确减半是判据）。适用域仅限 Mix（`T.Scope("Vector")`）kernel | §3.1、`probe_subid.py`（奇槽从不被写）、标定表 GM_to_UB 半量 | `repro/PL-1.13-vector-kernel-subid-zero.py` | tilelang 0.1.2+489385e9（489385e）+ CANN 9.0.0 + Ascend910B3 | `pattern-library/attention.md` PL-1.13（status 收窄：加「纯 Vector 不适用」适用域约束）+ `cases.md` |
| VP-2 | **D** | **C4 全局池化的 NaN 追踪等代价形**：`isnan(reduce_sum(x*x))`（**原位**平方）与 `vcmp eq 掩码 + reduce_min` 数学等价，但后者的 BishengHIR 归约 UB 成本 96–128 B/平面元素（限定 CH≤32），前者 32 B/元素（解锁 CH≤120）。`x*x` 仅 NaN 处为 NaN 且其余 ≥0 ⇒ 求和不会因 `+inf-inf` 假阳性（这是不能直接 `reduce_sum(x)` 的原因） | §3.2、§4 round1（36.77 → 19.16 µs）、`probe_ub2.py`（128 vs 32 B/元素） | `repro/PL-1.22-max-nan-squaresum.py` | 同上 | `pattern-library/layout.md` PL-1.22（update：追加 C4/整平面域的等代价形态）+ `constants.md`（归约 UB 单价） |
| VP-3 | **D** | **BishengHIR 归约 lowering 的 UB 单价**：`reduce` 类 op 的 UB 需求 ≈ 输入 tile 元素数 × 单价，单价取决归约链形态：单 `reduce_max`(fp16) ≈ <24 B/元素；`+vcmp→vcast(bool 掩码)→reduce_min` ≈ 128 B/元素；`+原位平方→reduce_sum` ≈ 32 B/元素；多一个**独立平面缓冲**（非原位）使单价翻倍（64→32 的半量关系）。归一化后与 `ch` 无关（严格线性，无固定项） | §3.2、`probe_ub2.py` 全量扫描 | `repro/TRAP-reduce-ub-per-element.py` | 同上 | `pattern-library/constants.md`（新条目 `CONST-reduce-ub-per-element`） |
| VP-4 | **P** | **单波即最小可行 CH**（general 路径 CH 判据，慢→快的判据修正）：`units = N·out_H·ceil(C/CH)`；**取满足 `units ≤ 物理核数` 的最小 CH**（= 最多核 × 最短最长块），而非「C 大小」或「尽量折半」。spp CH 64→48（24→36 units，20.42→18.97 µs）；nondiv CH 64→32（14→28 units，18.32→15.45 µs）；反例 CH=32 on spp（48 units → 两波）29.47 µs（+44%） | §4 round1/round2、`perf_records.jsonl` gen32/gen48、round2_merged | `repro/PL-1.4-min-single-wave-ch.py` | 同上 | `pattern-library/layout.md` PL-1.4（update：第六数据点，判据由「折半且单波」升级为「最小可行 CH」） |
| VP-5 | **D** | **global 类拷贝地板由「每核串行轮数」主导**：同 1.6 MB 流量，13 轮 10.99 µs / 7 轮 6.85 µs / 4 轮 5.09 µs；逐轮固定开销 ≈0.7–0.9 µs（DMA→落盘串行链），此时 mte2 仅 1.2–3.9 µs。地板 ≠ 带宽/流量，而是轮数 × 固定开销 | §2 标定表 | `repro/CONST-copy-floor-rounds.py` | 同上 | `pattern-library/elementwise.md` PL-1.6（update：轮数扫描维度）+ `constants.md` |
| VP-6 | **C** | 案例索引：`CASE-pool-adaptivemaxpool2d-migration` 的 Stage 4 续档（C4 扁平宽归约 + 原位平方-sum NaN 追踪 + 单波 CH 判据；PL-1.13 在纯 Vector 上不适用） | 本文件全文 + `perf_records.jsonl` | — | 同上 | `pattern-library/cases.md`（更新该 case 的 Stage 4 段） |
| BP_001 | **R** | 建议 skill Phase 0「首轮必查项」增加：**分片/并行类模式的「工作量守恒核验」**——采纳任何「分片/双核/拆分」类收益前，必须核验 GM 流量、指令数或输出覆盖率之一未按比例下降（防止「只做一半工作」被记成加速）。 | §8.1 R1 | — | — | proposal（不自动改 skill） |
| BP_002 | **R** | 建议 skill 增加纪律：**UB/容量类探针必须「无死代码」**（所有中间结果须被最终输出消费），否则 DCE 会给出错误的成本模型。 | §8.1 R2 | — | — | proposal（不自动改 skill） |
| BP_003 | **R** | 建议 `PL-1.6-copy-floor` 方法学增加「**轮数自变量扫描**」（同流量、不同 `num_local`），用于把「每轮固定开销」与「流量代价」分离。 | §8.1 R3 | — | — | proposal（不自动改 skill） |

> **本任务内回写（ED-C）**：VP-1/2/3/4/5 已按 Phase 4 §6 回写 pattern-library（含 `repro/` 最小代码，见 §9）；
> VP-6 更新 `cases.md`；终态 evolver 请跳过已回写条目。

---

## 9. 回写 pattern-library（本任务内，ED-C）

| 条目 | 文件 | repro | 状态 |
|---|---|---|---|
| `PL-1.13-aiv-dup-subid-split`（update：适用域收窄「纯 Vector kernel 不适用，subid 恒 0」） | `pattern-library/attention.md` | `repro/PL-1.13-vector-kernel-subid-zero.py` | 已回写 |
| `PL-1.22-maxreduce-nan-tracking`（update：追加 C4/整平面域的 `reduce_sum(x*x)` 等代价形态） | `pattern-library/layout.md` | `repro/PL-1.22-max-nan-squaresum.py` | 已回写 |
| `CONST-reduce-ub-per-element`（新增） | `pattern-library/constants.md` | `repro/TRAP-reduce-ub-per-element.py` | 已回写 |
| `PL-1.4-tiling-heuristic`（update：第六数据点，「最小可行 CH」判据取代「折半且单波」） | `pattern-library/layout.md` | `repro/PL-1.4-min-single-wave-ch.py` | 已回写 |
| `PL-1.6-copy-floor`（update：global 类地板由每核串行轮数主导 + 轮数扫描法） | `pattern-library/elementwise.md` | `repro/CONST-copy-floor-rounds.py` | 已回写 |
| `CASE-pool-adaptivemaxpool2d-migration`（更新 Stage 4 段） | `pattern-library/cases.md` | — | 已回写 |

## 10. Compiler Capability Gap

按 AGENTS.md「Compiler Capability Gap Rule」记录到 `.agents/evolution/capability-gaps.md`：
**BishengHIR 的归约 lowering 按输入 tile 元素数 × ~32–128 B 线性占用 UB（与实际 dtype 缓冲清单无关），
显著限制整平面归约路径的 CH 上限**；且同一 kernel 内 `reduce` 与 GM→UB `T.copy` 无法重叠
（实测 mte2 211 GB/s vs 同粒度纯拷贝 690 GB/s）。二者共同构成 global 类池化逼近拷贝地板的障碍。
（`T.vcmp("ne")` 的 ordered-NaN 陷阱与 `T.vcast` 同 dtype 退化属既有条目的复现，不新增。）
