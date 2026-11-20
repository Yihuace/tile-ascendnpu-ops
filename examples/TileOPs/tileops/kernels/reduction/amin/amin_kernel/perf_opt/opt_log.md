# Stage 4 调优日志 — AminFwdOp（reduction / amin）

- project: TileOPs（`examples/TileOPs`）；operator: **AminFwdOp**（manifest PascalCase 键）
- 算子目录: `examples/TileOPs/tileops/kernels/reduction/amin/amin_kernel/`
- 定稿产物（2026-10-26 R9 起）: `perf_opt/AminFwdOp.py`（**gate 契约文件名 = operator_name**，resident 路径 + canonical 单入口，顶层 re-export tiled 工厂）+ `perf_opt/_simple_reduce_kernel_tiled.py`（streaming/tiled 路径，保持原名）。`perf_opt/_simple_reduce_kernel.py` 保留为 R8 定稿的存档副本（kernel 内容与 AminFwdOp.py 逐字节相同，仅 docstring 不同），并作为 R9 A/B 的对照臂
- mode: `full`（scenario=optimize）；**2026-10-19 追加 Round 8 定向二轮**；**2026-10-26 追加 Round 9 bf16 归约路径重构**（三条路线 A/B/C 全部实测证伪，见下）
- 性能口径: **`msprof op` Task Duration(us)**（唯一 kernel 时延口径）；Roofline Ratio = `GM/L2 GM Read + Write` 条目 ratio（`parse_bin_file`），与 `tileops.reporting` 的 `Ratio(%)` 同源同口径（峰值参考 1.75 TB/s）
- 目标: 每个非 smoke case 的 Roofline Ratio ≥ 60%；未达标须给出原因
- 测量方式: 调优口径 = `_measure.py`（直接构建目标 kernel、按 wrapper 实际选择的 `(block_m, tile_n)` 采集 `msprof op`，`--launch-count=10 --warm-up=5 --aic-metrics=Roofline`）；**Round 8 候选裁决 = `.agents/tools/ab_test.py` 交错 A/B（每对交替先后顺序，`--launch-count=10 --warm-up=5`，6 对）**；生产口径 = `python -m tileops.reporting.cli run --op AminFwdOp --prof-mode msprof`（每 case 单发）
- 工具链戳: **R9 实测重取（2026-10-26）: tilelang `0.1.2+da4f6a41`（dev root `build/`，repo commit `da4f6a4`）+ CANN **9.0.0**（`/home/developer/Ascend/cann-9.0.0`）+ 设备 `Ascend910_9362`（cube 20 / vector **40**，`torch_npu.npu.get_device_properties` 实查）+ torch 2.7.1+cpu / torch_npu 2.7.1.post4**。⚠ R1–R8 行内标注的“tilelang 0.1.4 + CANN 8.5.2”为**过期戳**（与当前工具链不符），凡依赖该戳的结论按 `kb_stale_check` 纪律视为待重验——R9 已对其中「bf16 同域 reduce_min 不支持」一条重验并**证伪更正**（见 Round 9）。知识预注入 `PL-1.12-task-pipeline-depth2`（origin=attention/Expert）在本 op 的向量归约上**未复现收益**（见 Round 3/6）；`kb_stale_check` 报 `stale_count=133`（均为 attention 族，本轮未引用其余 pattern-library 条目）

## Workload Inventory（benchmark 展开，详见 `workload_inventory.json`）

benchmark 入口：`examples/TileOPs/benchmarks/ops/bench_amin.py::test_amin_bench`（`workloads_to_params("AminFwdOp", include_extra=True)`）。smoke 被 `-m "not smoke"` 排除。

| workload_id | dtype | shape | dim | 路径/配置（wrapper `ReduceKernel`） | kind |
|---|---|---|---|---|---|
| smoke-amin-float16 / bfloat16 / float32 | fp16/bf16/fp32 | (32,256) | -1 | resident bm=8 | smoke（仅精度回归） |
| hidden-state-reduce-float16 | fp16 | (2048,4096) | -1 | `_simple_reduce_kernel`，wrapper bm=4 | tune |
| hidden-state-reduce-bfloat16 | bf16 | (2048,4096) | -1 | `_simple_reduce_kernel`，wrapper bm=4（bf16 走 fp32 vcast 工作缓冲） | tune |
| 3d-multidim-reduce-float16 | fp16 | (4,128,4096) | [0,2] | op `permute(1,0,2).contiguous()` → (M=128,N=16384)；`_simple_reduce_kernel`，wrapper bm=1（核内 promoted→2） | tune |
| long-seq-reduce-bfloat16 | bf16 | (64,32768) | -1 | `_simple_reduce_kernel_tiled`，wrapper tile_n=16384（核内 clamped→8192） | tune |

> 说明：`3d-multidim-reduce` 的 `permute+contiguous` 是 op 层的一次设备内拷贝（不计入 kernel-only 口径）；kernel 侧 M=128、N=16384、流量 4.19 MB。`long-seq` 集成态（wrapper tile_n=16384）在基准 kernel 上**直接崩溃**（见下）。

## Performance Test Data — Baseline（round 0, phase=baseline）

4 次独立运行取均值（噪声对策 T-2）；pipe 指标为 profile 各 block 均值。

| workload_id | kernel | Task Duration(us) mean（4 run） | Ratio(%) mean | Block Dim | aiv_vec_ratio | aiv_scalar_ratio | aiv_mte2_ratio | aiv_mte3_ratio | profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| hidden-state-reduce-float16 | baseline resident bm=4 | **14.655** (14.12/16.26/13.84/14.40) | **65.7** | 40 | 0.266 | 0.293 | 0.599 | **0.586** | profiles/REC_base_hs_fp16_{1..4} |
| hidden-state-reduce-bfloat16 | baseline resident bm=4 | **19.465** (19.80/19.40/19.34/19.32) | **49.3** | 40 | 0.478 | 0.254 | 0.336 | 0.203 | profiles/REC_base_hs_bf16_{1..4} |
| 3d-multidim-reduce-float16 | baseline resident bm=1→2 | **7.335** (7.32/7.48/7.22/7.32) | **32.7** | 40 | 0.163 | 0.179 | 0.361 | **0.564** | profiles/REC_base_3d_fp16_{1..4} |
| long-seq-reduce-bfloat16 | baseline tiled bm=1 tn=8192† | **8.995** (9.24/9.00/8.74/9.00) | **26.7** | 40 | 0.368 | 0.194 | 0.391 | 0.091 | profiles/REC_base_lseq_bf16_{1..4} |

† **集成基线对 long-seq 直接失败**（wrapper 传 `tile_n=16384`，baseline `assert 2*tile_n*B(bf16=6) <= 184KB` 抛 `AssertionError`，见 `integration_log.md`）。为给出可对账的 baseline 记录，此处用**同一 baseline kernel 在预算合法 tile_n=8192** 下实测（即 baseline 结构在合法配置下的能力）。集成态（16384）= crash。

> **现象（Round 0 诊断）**：三例 resident baseline 的 `aiv_mte3_ratio` 高达 0.20–0.59 —— 每次 persistent 任务迭代都单独发一次 UB→GM 写出（`T.copy(out_ub, out)`），单次写出无论 payload 大小都耗 ~0.4–0.9 µs MTE3 busy，成为小 workload 的主导项（尤其 3d/bf16）；同时 `mte2_ratio` 仅 0.34–0.60，DMA 远未饱和（基线网格虽已 40 核，但逐任务写出把 DMA 挤掉）。bf16 的 `vec_ratio=0.478` 说明 vcast+fp32 reduce 是另一瓶颈。

### 设计估算 vs 实测偏差（D-2 回填）

`DESIGN.md §1.6.0` 设计期 roofline 估算下界（per-core 30.2 GB/s × 40 核 = 1.2 TB/s 外推档）：

| workload | 设计估算下界(µs) | baseline 实测(µs) | 偏差 | 失准项 |
|---|---:|---:|--:|---|
| hidden-state fp16 | ≈14 | 14.655 | 1.05× | 无（估算准） |
| hidden-state bf16 | ≈14–52 | 19.465 | 落在区间 | 发射项串行上界 |
| 3d-multidim fp16 | ≈3.5 | 7.335 | 2.10× | 发射/写出项（MTE3 每任务写出）被低估 |
| long-seq bf16 | ≈6.4（tiled 单元估算） | 8.995 | 1.41× | 发射项串行上界未达 |

> 偏差均 ≤2.1×，未触发 `[DESIGN_LIMIT]` 的「系统性偏差 >2×」归因门槛（§1.6.0-R2/R3 要求的回填对账完成）。

## Iteration Log

### Round 1 — 结构优化：批量化写出 + 连续分片 + reduce 目标切片（S2–S4），tiled S0 崩溃修复

- **base**：baseline（round 0）。
- **现象**：三例 resident baseline 由 MTE3（每任务 UB→GM 写出）主导（0.20–0.59），DMA 未饱和。
- **候选优化点（从同一 base 派生多个分支，每分支只改一个主优化点）**：
  - **S3 批量化输出写出**：每核把各任务的行极值暂存在扁平 UB buffer，**每核只发 1 次** UB→GM 拷贝（原为每任务 1 次）。
  - **S4 reduce 目标切片**：连续任务的行极值直接落在同一个 `(L*bm,1)` buffer 的连续切片，形状收缩每核 1 次。
  - **S2 连续、静态均衡的每核行区间**（S3 的前提：批量化要求每核输出行连续）。
  - **S1 网格规模**（Round 4 裁定，见下）。
  - （tiled 路径）**S0 tile_n 包络 clamp**（修复 long-seq 集成崩溃）+ 批量化写出 + 首 tile 提升。

- **实验结果**（Task Duration / Ratio；L0 全部 PASS）：

| 分支 | hs fp16 | hs bf16 | 3d fp16 | lseq bf16 | L0 |
|---|---:|---:|---:|---:|---|
| baseline（round 0 current best） | 14.66 / 65.7% | 19.47 / 49.3% | 7.34 / 32.7% | crash / 26.7%† | PASS |
| r1_contig（S2+S3+S4，guard-40） | 14.36 / 66.8% | 16.96 / 56.5% | 5.42 / 44.2% | — | PASS |
| r1 deliverable（S1 整除-32 + S2+S3+S4） | 13.86–14.82 / 64.7–69.2% | 17.32 / 55.4% | 5.18–5.26 / 45.6–46.3% | — | PASS |
| t1_clamp（tiled S0 only） | — | — | — | 8.84 / 27.1% | PASS |
| t1_full（tiled S0+S1+S2+S3，整除-32） | — | — | — | 9.00 / 26.6% | PASS |

† long-seq 集成态 crash；26.7% 为合法 tile_n=8192 的 baseline 记录。

- **winner**：resident S2+S3+S4 合并（MTE3 0.586→0.017、mte2 0.599→0.883）；tiled S0 clamp 修复崩溃。首轮以 S1 整除-32 起步（沿用同族 amax 选择），Round 4 改判为 full-40。

### Round 2 — block_m 精搜（候选 vs current best）

- **base**：Round 1 合并版（current best）。
- **候选优化点**：resident `block_m` ∈ {1,2,4,8} 精搜（bf16/fp16/3d），单点分支。
- **结果**（Task Duration / Ratio）：

| 候选（vs current best） | workload | Task Duration(us) | Ratio(%) | L0 | 结论 |
|---|---|---:|---:|---|---|
| bm=1 / bm=2 / bm=4 | hs bf16 | 17.44 / 17.04 / 16.80–17.04 | 55.0 / 56.3 / 57.1 | PASS | **bm=4 最优**（wrapper 默认） |
| bm=2 / bm=4 / bm=8 | hs fp16 | 14.16 / 13.02–13.30 / 13.70 | 67.7 / 72.1–73.6 / 70.0 | PASS | **bm=4 最优**（wrapper 默认） |
| bm=3 | 3d fp16 | 6.00 | 39.9 | PASS | 回退（>2） |
| bm=1→2（promotion）| 3d fp16 | 5.28 | 45.4 | PASS | **bm→2 最优**（已由核内 promotion 取到） |

- **结论**：`block_m` 已被 wrapper/kernel 取到实测最优（fp16/bf16 bm=4、3d bm=2）。`config_no_gain` 范围 = bm 精搜。

### Round 3 — bf16 结构性候选（候选 vs current best）

- **base**：Round 2 current best（整除-32 grid）。
- **现象**：bf16 resident `aiv_vec_time` ≈ 9.4 µs（含整宽 fp32 vcast）≈ `aiv_mte2_time`≈10.5 µs，两引擎**零重叠**（aiv_time≈15 µs）。
- **候选优化点**：
  - **bf16 同域 reduce_min**（去 vcast/work buffer）——若工具链支持可砍掉一次全宽转换。
  - **copy 融合 cast**（GM bf16 → UB fp32 直接跨 dtype `T.copy`，去独立 vcast pass）。
  - **任务循环软件流水**（`T.Pipelined(num_stages=2)`）恢复 DMA/向量重叠。

| 候选（vs current best） | workload | Task Duration(us) | Ratio(%) | aiv_vec/mte2 ratio | L0 | 结论 |
|---|---|---:|---:|---|---|---|
| current best（bm=4） | hs bf16 | 16.80–17.04 | 56.3–57.1 | 0.478 / 0.336 | PASS | — |
| bf16 同域 reduce_min | hs bf16 | — | — | — | — | **compile fail**（`TVMError: Check failed: buffer_shape0==buffer_shape1` → bf16 同域 reduce_min 本工具链**不支持**，DESIGN §9.1 约束复验成立） |
| copy 融合 cast（r3_copycast） | hs bf16 | 17.74 | 54.0 | — | PASS | **no gain**（跨 dtype copy 未省下向量工作量） |
| 任务循环 pipe2（r3_pipe2） | hs bf16 | 16.90 | 56.7 | — | PASS | **no gain**（≤3%，与本工具链 amax 同族结论一致；`PL-1.12` 未复现） |

- **结论**：bf16 的 vcast 在本工具链是**不可去**的（API 层面 `reduce_min` 不支持同域 bf16），流水/融合均无增益。current best 不变。`family_no_gain` 范围 = bf16 去 vcast / 融合 cast / 任务流水。

### Round 4 — 网格规模 A/B（exact-divisor-32 vs full-40），4 次交错（T-2 / BP_run_state_bimodality）

- **base**：Round 3 current best。
- **候选优化点**：persistent 网格从「≤40 的最大整除因子」（本机 fp16/bf16 hs 得 32 核、3d 得 32 核）改为**满 40 核 + 连续均衡分片（带 guard）**。单点分支，交错 A/B/A/B×4。

| 候选（vs current best） | workload | Task Duration mean(us)（4 run） | Ratio(%) mean | Block Dim | L0 | 结论 |
|---|---|---:|---:|---:|---|---|
| exact-32（current best） | hs fp16 | 13.815 (13.94/13.72/13.90/13.70) | 69.4 | 32 | PASS | — |
| **full-40（guard）** | hs fp16 | **13.430** (13.30/13.46/13.48/13.48) | **71.4** | 40 | PASS | **winner，+2.8%**（4/4 run 更快） |
| exact-32（current best） | hs bf16 | 17.645 (17.54/17.92/17.62/17.50) | 54.6 | 32 | PASS | — |
| **full-40（guard）** | hs bf16 | **16.385** (16.56/16.02/16.54/16.42) | **58.5** | 40 | PASS | **winner，+7.2%**（4/4 run 更快，非噪声） |
| exact-32（current best） | 3d fp16 | 5.305 (5.48/5.18/5.28/5.28) | 45.2 | 32 | PASS | — |
| full-40（guard） | 3d fp16 | 5.340 (5.30/5.34/5.34/5.38) | 44.9 | 40 | PASS | tie（≤3%） |

- **结论**：**full-40 网格**在 hs fp16/bf16 上明确胜出（bf16 高达 +7.2%），3d 打平 → 采纳为 current best 并核内合并。**证伪更正记录**：同族 `amax` perf_opt 曾把「整除-32 vs full-40」判为平区；本 op 在同工具链/同设备上以 4 次交错 A/B 得到 bf16 一致 +7.2%，说明该结论**不是算子无关的**（bf16 的向量瓶颈对核数更敏感），按 op 单独实测裁定。

### Round 5 — tiled 网格 A/B + 定稿复测（候选 vs current best）

- **base**：Round 4 current best。
- **候选优化点**：tiled 路径同样从整除-32 改满-40（带 guard）。交错 3 次。

| 候选（vs current best） | workload | Task Duration mean(us) | Ratio(%) mean | L0 | 结论 |
|---|---|---:|---:|---|---|
| exact-32（current best） | lseq bf16 | 9.08 (9.10/9.00/9.14) | 26.4 | PASS | — |
| **full-40（guard）** | lseq bf16 | **8.79** (8.90/8.94/8.54) | **27.3** | PASS | **winner，+3.2%**（3/3 run 更快） |

- 定稿 resident 复测（full-40）：hs fp16 13.30/14.20/13.48（70.2%）、hs bf16 16.72/16.44/16.56（57.9%）、3d 5.72/5.32/5.42（43.7%）。
- **winner**：tiled full-40 并入 current best；resident full-40 已合并。

### Round 6 — bf16 满网格下的流水/参数再验证（候选 vs current best）

- **base**：Round 5 current best（full-40）。
- **候选**：① 满网格 + 任务循环 pipe2；② 满网格 + bf16 bm=2。

| 候选（vs current best） | workload | Task Duration(us) | Ratio(%) | L0 | 结论 |
|---|---|---:|---:|---|---|
| current best（full-40, bm=4） | hs bf16 | 16.44–16.72 | 57.3–58.3 | PASS | — |
| full-40 + pipe2 | hs bf16 | 17.16–17.80 | 53.9–55.9 | PASS | **no gain**（回退） |
| full-40 + bm=2 | hs bf16 | 16.46–19.24（高方差） | 49.8–58.2 | PASS | **no gain**（>bm=4） |

- **结论**：bf16 在本设计族内进入 plateau；无剩余 valid 结构候选。

### Round 8 — 定向二轮：搬同族 amax 二轮手段（S1′ 网格 / S6 静态 extent）+ 结构再验证

- **触发**：生产路径 hidden-state bf16 = 58.60%（距 60% 差 1.4pp）；同族 AmaxFwdOp 二轮用「满 40 核网格 + 静态行块 extent」把同形状 bf16 做到暖机中位 60.3–60.6%。
- **base**：Round 5/6 current best（`perf_opt/_simple_reduce_kernel.py`，sha256 `7f5b01d4…`）。**先逐条核对 amax 手段是否已在 amin**：
  - **S1′ 满 40 核 + 均衡分片：已存在**。本 op 的 `_choose_grid` 自 Round 4 起即为 `nk = min(num_logical, 40)` + `cid*nl//nk` 均衡区间（Round 4 交错 A/B 已 4/4 胜 exact-32，bf16 +7.2%）。本轮无需再搬，仅复测确认生产路径 Block Dim=40。
  - **S6 静态行块 extent：未采用，本轮搬入分支验证**（`M % bm == 0` → `T.copy(x[off:off+bm,:], x_ub)`）。
- **现象（base profile，hs bf16，40 核）**：`aiv_time` mean 14.74us / max 16.05us；`aiv_vec_time` mean 7.50us（fp16 仅 2.77us）；`aiv_scalar_time` mean 4.39us；`aiv_mte2_time` mean 10.50us、`aiv_mte2_ratio` 0.713；`aiv_mte3_ratio` 0.012。→ bf16 的 **vcast+fp32 reduce 向量链比 fp16 多 4.73us**（占 kernel 时长 ~28%），是 hidden bf16 与 60% 之间的主项；DMA 已接近饱和（mte2 0.71，max 11.5us），scalar 次之。

- **测量协议修正（本轮最重要的方法学发现）**：首次用 `_measure.py` 顺序 A/B（每对固定先 base 后候选）×5 得到「base 5/5 胜、候选 +5.55%」，反序（候选先）×4 又得到「候选 4/4 胜、−7.19%」；同一 kernel 连续两次测量（A/A）第二次稳定慢 1.7–2.8%。→ **msprof 单发存在 per-process 位置/run-state 双态偏置（约 1.1–1.4us，6–8%）**，顺序 A/B 不可用于 <5% 候选。改用 `ab_test.py`：每对**交替先后顺序**（pair 0 A→B，pair 1 B→A…）、6 对、`launch-count=10`。修正后同一比较由「5.55% 回退」变为 tie。
- **交叉校验（同状态 amin vs amax）**：`ab_test` 6 对，amin base 16.44us vs amax final 16.47us，`verdict=tie`（pair diff 0.25/0.02/−0.09/−0.07/0.08/−0.03）。→ 两算子在**同状态**下完全等价；生产报告里 58.6% vs 60.6% 的差是 run-state，不是 amin 代码更慢。

- **候选优化点（均从同一 base 派生，每分支只改一个主点；L0 全 PASS；`ab_test` 6 对，除非另注）**

  | 分支 | 主改动 | 文件 | Task Duration 中位(us) | vs base | 配对胜/负 | verdict | 结论 |
  |---|---|---|---:|---:|---:|---|---|
  | base（current best） | — | `_simple_reduce_kernel.py` | 16.53 | — | — | — | — |
  | R8-S6 | 静态行块 extent（M%bm==0） | `_simple_reduce_kernel_round8_s6.py` | 16.46 | **−0.42%** | 4/6 | tie | **不采纳**（<3% 噪声） |
  | R8-GM32 | bf16 去 `x_ub`+`vcast`，GM 跨 dtype 直拷 fp32 work | `_simple_reduce_kernel_round8_gm32.py` | 16.445 | **−0.51%** | 4/6 | tie | **不采纳** |
  | R8-DYNLOOP | `T.serial(end-start)` 动态 trip count，去每任务 guard | `_simple_reduce_kernel_round8_dynloop.py` | 16.545 | +0.30% | 3/6 | tie | **不采纳** |
  | R8-BM5 | bf16 `block_m=5`（4 对） | 直接以 bm=5 调用 | 16.975 | **+2.60%** | 0/4 | tie（方向一致回退） | **不采纳** |
  | R8-BM6 | bf16 `block_m=6` | — | — | — | — | **compile fail** | UB overflow（需 1574656 bits > 1572864 bits，auto multi-buffer） |

  > S6 在 amin 上是 tie 而非 amax 的 −3.5%：结构上 min 与 max 的 DMA 形态完全相同，差异只能来自测量状态；amax 的 S6 −3.5% 是 **22 个非配对样本**，本轮在 amin 上用 6 对交替顺序复测未复现。amax 的 +5.7%（网格）结论不受影响（那是 5/5 配对胜且幅度大于位置偏置）。

- **候选 vs current best（B2 结构化对比表，hs bf16）**

  | 候选 | Task Duration(us) | Ratio(%) | aiv_vec/mte2 ratio | L0 | 结论 |
  |---|---|---:|---|---|---|
  | current best（full-40 + 动态 extent） | **16.53** | **58.0**（该状态换算） | 0.51 / 0.71 | PASS | base |
  | R8-S6（静态 extent） | 16.46 | 58.3 | 0.51 / 0.71 | PASS | tie，不采纳 |
  | R8-GM32（去 vcast） | 16.445 | 58.4 | 0.51 / 0.71 | PASS | tie，不采纳 |
  | R8-DYNLOOP | 16.545 | 58.0 | 0.51 / 0.71 | PASS | tie，不采纳 |
  | R8-BM5 | 16.975 | 56.5 | — | PASS | 回退，不采纳 |

- **合并与非回退检查**：本轮**无 winner**，`perf_opt/_simple_reduce_kernel.py` 与 `_simple_reduce_kernel_tiled.py` 均**未改动**（sha256 仍为 `7f5b01d4…` / `380b3a89…`）。两个文件各自 `--level all`（L0/L1/L2/Boundary）**全过**、无 WARN（`logs/round8/final_resident_all.log`、`final_tiled_all.log`）。hs fp16 / 3d / lseq 未做任何结构探索（用户指定不投预算），生产复测无回退。

- **生产路径复测（`tileops.reporting.cli`，3 次单发，`logs/round8/prod_run{1,2,3}.log`）**：

  | run | hs fp16 | hs bf16 | 3d fp16 | lseq bf16 |
  |---|---:|---:|---:|---:|
  | 1 | 13.06 / 73.41% | 16.72 / 57.34% | 5.50 / 43.58% | 8.68 / 27.61% |
  | 2 | 13.50 / 71.01% | **15.88 / 60.37%** | 5.62 / 42.65% | 8.92 / 26.87% |
  | 3 | 13.30 / 72.08% | 16.86 / 56.86% | 5.42 / 44.22% | 8.88 / 26.99% |
  | 中位 | **72.08%** | **57.34%** | **43.58%** | **26.99%** |

  bf16 单发分布 56.86–60.37%，与同族 amax 的二轮分布（57.41–60.99%）同形态：**定稿已跨在 60% 线上，单发结果由 run-state 决定**。

- **winner**：无。current best 保持不变；Round 8 的全部候选文件保留为实验证据（`perf_opt/_simple_reduce_kernel_round8_*.py`），`perf_records.jsonl` 追加 5 条 `round=8, phase=candidate` 记录（含 current-best 复测）。

### Round 9 — bf16 归约路径重构（三条路线 A/B/C + 两条结构补测，全部证伪；无 winner）

- **触发**：生产路径 hidden-state bf16 = 58.60%（距 60% 差 1.4 pp）。根因假设：bf16 无原生 min 归约 → 整块 `vcast` 升 fp32 再求 min，多出一次全宽提升。精确台账：本实现 bf16 每元素片上流量 **10 B**（vcast 读 2 + 写 4，fp32 reduce 读 4），fp16 只需 **4 B**（读 2 + reduce 读 2）；即提升链净增 6 B/元素、相对 fp16 多 2.5× 向量流量。目标：找到 2 B/元素的 bf16 归约域。
- **base**：R8 定稿 `perf_opt/_simple_reduce_kernel.py`（sha256 `7f5b01d4…`）。R9 未改动任何 kernel 代码，先做**能力探测（minimal repro）**再做端到端 A/B。
- **测量协议**：`.agents/tools/ab_test.py` 交错 A/B（每对交替先后顺序、6 对偶数对、`--launch-count=10 --warm-up=5`、噪声阈值 3%），与 R8 同协议；单次 msprof 的 per-process 位置偏置（~6%）不足以裁决本轮的差值。
- **工具链重取**：CANN 9.0.0 / tilelang `0.1.2+da4f6a4` / 910_9362（见文首工具链戳）。R3 的「bf16 同域 `reduce_min` 编译失败」结论绑定的是旧构建，按证伪协议**必须重验**。

#### R9 现象（base profile，hs bf16，40 核，取自 R8 profile 数据 + 本轮复测）

| 指标 | hs fp16 | hs bf16 | 差值 |
|---|---:|---:|---:|
| `aiv_time` | ~13.4us | ~17.0us | +3.6us |
| `aiv_vec_time` | 2.77us | 7.50us | **+4.73us** |
| `aiv_mte2_time`（DMA） | ~10.5us（ratio 0.71） | ~10.5us | 0 |

- **关键判读**：`aiv_time ≈ aiv_mte2_time + aiv_vec_time`（fp16 13.4 ≈ 10.5+2.77；bf16 17.0 ≈ 10.5+7.5）→ 本 kernel 的 DMA 与向量链**几乎零重叠**。fp16 在 UB 余量充足（2×64KB）时同样零重叠，说明**不是 UB 容量/多缓冲问题**，而是本编程模式（Developer + auto sync）不产生跨任务 DMA/vector 重叠。故 bf16 与 60% 的全部差距 = 多出来的 4.73us 向量时间。

#### 路线 A — bf16 原生 `T.vmin` / 同域 `T.reduce_min`

**能力探测（minimal repro，`/tmp/opencode/amin_probe/probe_*.py`）**：

| 探测 | 结果 |
|---|---|
| `T.vmin(a_bf16, b_bf16, dst_bf16)` elementwise | **编译通过 + 逐元素 bit-exact**（文档 `比较操作/T.vmin.md` §2.2.1 标注 bf16 = ×，**与实测矛盾**，属 DOC-GAP） |
| `T.reduce_min(bf16 (bm,N) → bf16 (bm,1))** | **编译通过**（R3 的 compile-fail 结论**证伪**）；随机数据 bit-exact；但**不能写 dst 切片**：写`m_all[s*bm:(s+1)*bm,0:1]`（行切片）→`VcastCodegen` shape check 失败；写 `m_all[0:bm,s:s+1]`（列切片）→ **编译通过但结果静默错误** |
| IR 反查（`TILELANG_DUMP_IR=TRUE`） | bf16`reduce_min` → `memref.copy`+ **`vcast(bf16→f32)` + `vreduce<min>(f32)` + `vcast(f32→bf16)`**；bf16`vmin`→ **2×`vcast(bf16→f32)` + `vmin(f32)` + `vcast(→bf16)`**。**hivm 层根本没有 bf16 min 指令**——编译器自己做的就是 fp32 提升 |

- **候选 A1（唯一可编译落地的形态）**：bf16 同域 `reduce_min` + 每任务 `(bm,1)` 临时 dst + 逐元素搬入 `m_all`（绕开切片 bug）。

| 候选 vs current best | workload | Task Duration(us) | vs base | 配对胜/负 | L0 / bit-exact | verdict |
|---|---|---:|---:|---:|---|---|
| base = `_simple_reduce_kernel.py` | hs bf16 | 16.905（6 次中位） | — | — | PASS | — |
| **A1** native bf16 reduce（tmp dst） | hs bf16 | **18.200** | **+7.66%** | 0/6 | PASS / bit-exact | **A_better（候选更慢）**，p=0.0312 |

- **A 结论**：`T.vmin`/`T.reduce_min` 的 bf16 形态在 hivm 层被无条件提升为 fp32，**本实现已有的显式 `vcast` + fp32 work buffer 就是最小形态**；让编译器自己插提升反而多一次「每任务 f32→bf16 回写」+ 切片绕行开销，慢 7.66%。**路线 A 既不省字节也不省时间**。
- **A 附带的语义边界（所有 dtype 一致，非回归）**：硬件 `vreduce<min>` 对混合 ±0 返回 **-0.0**（torch.amin 返回首个操作数），NaN 归一化为 `0x7fff`（fp16/bf16）/ `0x7fffffff`（fp32），而 torch 返回 `0x7fc0`/`0x7fc00000`。实测 fp16/bf16/fp32 **三者行为完全相同**（既有 fp16 fp32 路径自 Stage 4 起即如此），数值等价（`±0` 相等、NaN 判等），L0/L1 的 `assert_close`/`equal_nan=True` 全过。

#### 路线 B — 位模式整数比较（IEEE 单调映射 + int16 归约）

- **映射**：`key = u ^ ((u >> 15) & 0x7FFF)`（算术右移：负数为 0xFFFF），`reduce_min(key)` 后同式自逆还原。精确无损、2 B/元素。
- **能力探测**：

| 探测 | 结果 |
|---|---|
| int16 `reduce_min`（含 dst 行切片 + L≥2） | 编译通过、bit-exact ✓ |
| `vshr`(int16, (1,) 移位张量) | ✓ 且为**算术右移**（负数得 0xFFFF） |
| `vand` / `vxor` | 仅接受**同形或广播到的 tensor**；Python 标量与 `(1,)` 张量形态均 codegen 失败（`memref.memref_as_ptr`）→ 必须额外分配一个 (bm,N) 的 `0x7FFF` 常量 buffer |
| `T.view(x_gm_bf16, dtype="int16")`（GM 位重解释） | **codegen 失败**（`memref.memref_as_ptr`） |
| `T.view(x_ub_bf16, dtype="int16")`（UB 位重解释） | **codegen 失败**（`InternalError: GetVarValue(op->buffer_var.get())`） |

- **候选 B1（等宽 I/O 微基准：int16 输入 2 B/元素，与 bf16 臂同字节）**：`copy → vshr → vand → vxor → reduce_min(i16)`，IR 反查确认全部为**原生 i16 指令**（`vshr round:true` / `vand` / `vxor` / `vreduce<min>` on i16，无任何 cast）。

| 候选 vs current best | workload | Task Duration(us) | vs base | 配对胜/负 | L0 | verdict |
|---|---|---:|---:|---:|---|---|
| base（bf16: vcast + fp32 reduce） | hs 形状 | 17.040 | — | — | PASS | — |
| **B1** int16 位域（3 变换 op + i16 reduce） | hs 形状 | **39.195** | **+130%** | 0/6 | PASS | **A_better（候选慢 2.3×）**，p=0.0312 |

- **B 结论**：i16 指令在 910_9362 上**虽原生但吞吐远低于 fp32 向量链**（本 kernel 形态下 3 变换 + i16 reduce 的向量时间 ≈ 28.7us，是 fp32 路径 6.4us 的 4.5×），叠加 3 次全宽 pass 的字节量（12 B/元素 vs vcast 的 6 B/元素）。**路线 B 结构性亏损**；且 bf16↔int16 的位重解释在本工具链不可用（GM/UB 两处 `T.view` 均 codegen 失败），落地还需改 kernel 对外签名（输入 int16 / 输出 int16 再 `.view()` 回来），超出「只切 import」的 wrapper 约束。

#### 路线 C — 绕开提升 bug 的 pair-min 折叠 / 任务流水

| 候选 | 主改动 | Task Duration(us) | vs base | 配对胜/负 | 结果 |
|---|---|---:|---:|---:|---|
| C1 `T.Pipelined(L, num_stages=2)` | 任务循环软件流水 | 16.975 | −1.05% | 3/6（方向不一致） | **tie**；**IR 反查 stages=1/2 的 `memref<4x4096xbf16>` 分配数完全相同 → 未产生任何双缓冲**，该 API 在本 loop 形态上是 no-op |
| C2 bm=5 + GM 跨 dtype 直拷 fp32（去 `x_ub`，80KB 让双缓冲能装下） | 减小 UB 足迹以恢复重叠 | 17.845 | **+4.48%** | 0/6 | **回退**（`A_better`），p=0.0312 |
| C3 bm=6 / bm=8 + GM 直拷 fp32 | 同上，进一步减任务数 | — | — | — | **编译失败**（`bishengir-compile` rc=1，auto multi-buffer UB overflow） |
| C0 瓦片内双两两 `T.vmin` 折叠（本路线原始设想） | bf16 pair-min 后再升 fp32 | — | — | — | **结构性阻断**：需对 UB sub-buffer 切片做向量 op → `TRAP-vecop-slice-noop-and-nan-branch`(1)「向量 op 的 sub-buffer 切片被静默忽略，退化为恒等」；bf16 的列切片 reduce 更会**静默产出错误值**（上文 A 表）。按证伪协议不实现（已知静默错误形态，做了也无判别力） |

- **C 结论**：折叠方向被切片陷阱阻断；流水方向被 `T.Pipelined` 的 no-op 阻断；减少 UB 足迹能装下双缓冲但**没有重叠可恢复**（fp16 对照组在 UB 充足时同样零重叠，见现象表）。

#### 候选 vs current best（B2 结构化对比表，hs bf16）

| 候选 | Task Duration(us) | Ratio(%)（1.8 TB/s 口径） | aiv_vec / mte2 ratio | L0 | 结论 |
|---|---:|---:|---|---|---|
| **current best（R8 定稿）** | **16.905** | 55.6 | 0.51 / 0.71 | PASS | base |
| A1 native bf16 reduce（tmp dst） | 18.200 | 51.6 | — | PASS | 回退，不采纳 |
| B1 int16 位域（等宽微基准） | 39.195 | — | — | PASS | 回退 2.3×，不采纳 |
| C1 `T.Pipelined(2)` | 16.975 | 55.4 | 同 base | PASS | tie，不采纳 |
| C2 bm=5 + GM→f32 | 17.845 | 52.7 | — | PASS | 回退，不采纳 |
| C3 bm=6/8 + GM→f32 | — | — | — | — | 编译失败 |

#### 合并与非回退检查

- **本轮无 winner** → kernel 代码零改动。`perf_opt/AminFwdOp.py` = R8 定稿内容（仅 docstring 增补 R9 结论），`perf_opt/_simple_reduce_kernel_tiled.py` **未改动**（sha256 仍 `380b3a89…`）。
- 两文件各自 `--level all`（L0/L1/L2/Boundary）**全过**、无 WARN（`logs/round9/`）：`AminFwdOp.py` L0 15 项 / L1 10 项 / L2 4 项 / Boundary 5 项；`_simple_reduce_kernel_tiled.py` 同样全过（含 (5,50000)、(64,40000)、(2,262144) 等边界）。
- smoke（32,256 fp16/bf16/fp32）精度通过（含在 `--level all` 的 L0-1a/1b/1c）。
- 生产路径（`python -m tileops.reporting.cli run --op AminFwdOp --prof-mode msprof`）correctness **14/14**，benchmark 4/4 出数。

#### 本轮证据索引（可复核）

| 证据 | 路径 |
|---|---|
| 路线 A/B/C1/C2 的交错 A/B 原始 profile + 逐次 log + 裁决 JSON | `ab_r9_bf16/`、`ab_r9_dtype/`、`ab_r9_pipe2/`、`ab_r9_bm5/`（各含 `pair{1..6}{A,B}/`、`logs/`、`result.json`） |
| 路线 A 候选（原生 bf16 reduce + tmp dst） | `_simple_reduce_kernel_round9_bf16tmp.py`；`_simple_reduce_kernel_round9_native_bf16.py`（同域无临时 dst 版，L0-3 起 `VcastCodegen` shape check 失败） |
| 路线 B 等宽 I/O 微基准（bf16 臂 vs int16 臂） | `_r9_probe_dtype.py` + `_arm2.py` |
| 路线 C1 流水探针（含完全静态 exact 变体） | `_r9_probe_pipe.py`、`_r9_probe_pipe3.py` + `_arm_pipe2.sh` / `_arm_pipe2_nk32.sh` |
| 路线 C2 bm/UB 足迹探针（bm=5/6/8） | `_r9_probe_bm.py` + `_arm_bm5.sh` |
| 定稿最终精度回归（`--level all`） | `logs/round9/final_AminFwdOp_all.log`（L0 15 / L1 10 / L2 4 / Boundary 5，全 PASS）、`logs/round9/final_tiled_all.log` |
| 定稿 final 采集 raw msprof | `profiles/FIN2_{hs_fp16,hs_bf16,3d_fp16,lseq_bf16}_{1,2,3}/msprof_output` |
| 官方单算子 report | `examples/TileOPs/reports/tileops/20261026_112451_770524_AminFwdOp/report.md` |
| 最小 repro（能力探测原始脚本，仓外） | `/tmp/opencode/amin_probe/probe_vmin_bf16.py`（vmin bf16 bit-exact）、`probe_reduce4.py`（int16/bf16 reduce）、`dump_vmin.py` / `dump_ir.py` / `dump_pipe3.py`（IR 反查）、`probe_view_shared.py`（`T.view` codegen 失败）、`probe_zero.py`（±0 / NaN 边界，fp16/bf16/fp32 三者一致） |

#### 能力缺口（按 AGENTS.md「Compiler Capability Gap Rule」记录）

| 缺口 | 层 | 证据 |
|---|---|---|
| bf16 无原生 `vmin`/`vreduce<min>`（编译器无条件提升 fp32） | BishengIR / hivm | `TILELANG_DUMP_IR` 反查（本文件上文）；`docs/Tilelang.language/比较操作/T.vmin.md` §2.2.1 亦标 bf16 = ×，但**实测 `T.vmin` bf16 可编译且 bit-exact**（文档与实现不一致，DOC-GAP） |
| `T.view` 位重解释在 GM/UB 两处均 codegen 失败 | Frontend API → TileLangIR | `probe_reduce2.py` / `probe_view_shared.py` 的 InternalError |
| `T.Pipelined(num_stages>=2)` 对本 loop 形态不产生双缓冲 | TileLangIR pass | stages=1/2 的 IR 分配数相同 |
| int16 向量 op 原生但吞吐极低（≈fp32 链的 1/4.5） | 硬件/后端 | B1 实测 39.195 vs 17.040 us |

## Autotune Log

未使用 autotune 框架；`block_m` / 网格规模 / tiled `tile_n` 均按「wrapper 选择 + 核内受控 promotion + 手动粗搜」执行，并经 interleaved 多次复测裁决。搜索空间：网格 ∈ {整除 ≤40, 满 40}；`block_m` ∈ {1,2,4,8}；tiled `tile_n` ∈ {4096, 8192, 16384}。最优：**满 40 核 + bm=4（3d bm→2）+ tile_n clamp 8192**。

## Final Summary

- 定稿：`perf_opt/AminFwdOp.py`（gate 契约文件名 = operator_name；candidate `final`；content = R8 winner，顶层 re-export streaming 工厂）、`perf_opt/_simple_reduce_kernel_tiled.py`（streaming 路径，保持原名）。两文件各自 `--level all`（L0/L1/L2/Boundary）**全过**，无 WARN。
- 最终四例均由 `perf_opt/AminFwdOp.py` 采集（`phase=final`，candidate=`final`，3 run 取中位），与 `perf_records.jsonl` 的 `artifact_sha256`（`b3f7a691…`）一致；L0 逐例 `max_diff=0`。R8 的 4 条旧 `phase=final` 记录按 gate 4「每 workload 恰一条 final」要求**降级为 `phase=merged`** 并加 `note`（append-only 纪律的受控例外，已在此处与 Round 9 记明）。
- wrapper 集成路径：`amin.py` 的 baseline/perf_opt 双 import 切换块**恢复并指向 perf_opt**（`perf_opt.AminFwdOp` + `perf_opt._simple_reduce_kernel_tiled`）；4 个 tune workload 全部 `torch.amin` 语义匹配，long-seq 经 wrapper（tile_n=16384 → 核内 clamp 8192）可运行。
- 全 tune workload 非回退检查：hs fp16/bf16/3d 均较 baseline 提升；long-seq 从 crash → 可运行且与合法 baseline 持平；smoke 精度通过。
- 达标结论：**1/4 达标（hs fp16 66.7%，生产口径）**；hs bf16 **55.3%**（生产单发 16.86us）、3d 43.0%、lseq 24.8%。hs bf16 与 60% 的差距经 R9 三条路线实测证明**不可由归约域重构消除**（bf16 无原生 min 指令，编译器对 `vmin`/`reduce_min` 一律提升 fp32）。
- 中止原因：`blocked` —— R9 的 3 条路线 + 2 条结构补测全部被 **API/编译器硬约束**关闭（hivm 无 bf16 min；`T.view` 位重解释不可用；i16 向量吞吐 1/4.5；sub-buffer 切片静默错误；`T.Pipelined` no-op），family_no_gain 范围 = bf16 归约域重构族。
- 口径提示：本轮 msprof 的峰值参考为 **1.8 TB/s**（`report.md` 的 `Bandwidth (TB/s) = 1.8000`），R8 及更早报告用 1.75 TB/s；**同一 kernel 的 Ratio 会因该常数不同而差 ~1.4 pp**，跨轮比较请以 Task Duration(us) 为准。R8 的「58.60%」= 16.72us @1.75 口径；本轮 16.86–16.96us @1.8 口径 = 55.3%。

### Final Performance Test Data

Deliverable `perf_opt/AminFwdOp.py`（candidate `v3_s7_reverify`），每个 tune workload **恰一条** `phase=final` 记录（3 run 中取中位；raw profile 见 `perf_records.jsonl` 的 `msprof_raw_path`）。下表 `baseline_us` = round-0 最接近 4 run 均值的 `phase=baseline` 记录；两列均取自 `perf_records.jsonl`，`kernel_id` 为 append-only 历史沿用的稳定 workload 键（artifact 由 `artifact_path` 对账）。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/reduction/amin/amin_kernel/perf_opt/_simple_reduce_kernel.py::_simple_reduce_kernel::main | hidden-state-reduce-float16 | 14.400 | 13.660 | v3_s7_reverify |
| examples/TileOPs/tileops/kernels/reduction/amin/amin_kernel/perf_opt/_simple_reduce_kernel.py::_simple_reduce_kernel::main | hidden-state-reduce-bfloat16 | 19.400 | 17.440 | v3_s7_reverify |
| examples/TileOPs/tileops/kernels/reduction/amin/amin_kernel/perf_opt/_simple_reduce_kernel.py::_simple_reduce_kernel::main | 3d-multidim-reduce-float16 | 7.320 | 5.400 | v3_s7_reverify |
| examples/TileOPs/tileops/kernels/reduction/amin/amin_kernel/perf_opt/_simple_reduce_kernel.py::_simple_reduce_kernel_tiled::main | long-seq-reduce-bfloat16 | 9.000 | 6.540 | v3_s7_reverify |

> **v3_s7_reverify 接管说明（candidate `v3_s7_reverify`，2026-10-30 12:40–12:46 同会话 3 次 op 级 report 中位）**：`final_us` 来自同一候选在同一次复核会话内的 3 次 `tileops.reporting.cli run --prof-mode msprof` 中位（`reports/tileops/20261030_124010`/`124301`/`124551_AminFwdOp`），与 `perf_records.jsonl` 的 4 条 `phase=final`（`candidate_id=v3_s7_reverify`）一一对应；上一轮 `v3_smalltask_s7` 的 4 条 `phase=final` 已按 signal-registry §5 降级为 `phase=merged`（数值原样保留 + note）。
>
> **v3_20261101_smalltask 原始接管说明（candidate `v3_smalltask_s7`）**：上表已按本轮定稿重测更新（同一最终候选、同会话 2–7 次 `msprof op` 中位；`perf_records.jsonl` 中 r8/r9 的旧 `phase=final` 行按 signal-registry §5「二次进入 Phase 3」条款降级为 `phase=merged` 并加 `note`，数值原样保留）。本轮**只改 streaming（tiled）模块**：long-seq 的 2 行流水段（S7）；resident 模块 `AminFwdOp.py` 的 sha256 与任务开始时的备份**逐字节相同**（`b3f7a6914d8f6a96f57cdefebd3ceced29de8e3dec6dc2688b78b50eb262dbae`），因此 hs/3d 的差异只可能是会话态（见文末「本轮会话态」）。

**多 run 口径（R9 复测，3 run，`_measure.py`）**：hs fp16 13.32/13.68/13.74（中位 **13.68**，Ratio 69.98/68.13/67.84）、hs bf16 16.96/16.94/17.42（中位 **16.96**，54.96/55.02/53.51）、3d fp16 5.30/5.58/5.30（中位 **5.30**，43.97/41.76/43.97）、lseq bf16 9.16/9.00/8.94（中位 **9.00**，25.44/25.89/26.07）。

**生产口径（`tileops.reporting.cli`，`reports/tileops/20261026_112451_770524_AminFwdOp/`，单发）**：hs fp16 **13.98us / 66.67%**、hs bf16 **16.86us / 55.28%**、3d fp16 **5.42us / 42.99%**、lseq bf16 **9.40us / 24.79%**；correctness 14/14。

> long-seq 集成基线为 crash（wrapper tile_n=16384 被 baseline kernel 拒绝）；上表 9.000 = 同一 baseline kernel 在合法 tile_n=8192 下实测（可对账记录见 `perf_records.jsonl` phase=baseline）。

### 逐 case 结果与未达标原因

| case | dtype | 基线 Ratio | 优化后 Ratio | ≥60% | 未达标原因（通俗） |
|---|---|--:|--:|:-:|---|
| hidden-state-reduce | fp16 | 65.7% | **66.7%**（生产单发；3 run 中位 68.13%） | ✅ | — |
| hidden-state-reduce | bf16 | 49.3% | **55.3%**（生产单发 16.86us；3 run 54.96/55.02/53.51） | ❌ | **量化结构原因（R9 已定案）**：hivm 层**没有 bf16 min 指令**——`T.reduce_min(bf16)` 与 `T.vmin(bf16,bf16)` 都被编译器降成 `vcast(bf16→f32) + f32 min + vcast(→bf16)`（IR 反查），所以本实现的「显式 vcast + fp32 work buffer」已是**最小形态**：每元素片上 10 B（vcast 读 2 写 4 + fp32 reduce 读 4），fp16 只要 4 B。实测该差额 = `aiv_vec_time` 7.50 vs 2.77us（**+4.73us**），而本编程模式下 DMA 与向量链**零重叠**（`aiv_time ≈ mte2 + vec`，fp16 对照组同样零重叠），故 4.73us 全额计入 kernel 时长。R9 逐条实测的三条替代域：int16 位模式（原生 i16，但慢 2.3×）、瓦片内 pair-min 折叠（sub-buffer 切片被静默忽略/静默错值）、`T.Pipelined` 流水（no-op，IR 无双缓冲）——**全部证伪**，无可达优化点。 |
| 3d-multidim-reduce | fp16 | 32.7% | **43.0%** | ❌ | 这个 case 只有 4 MB 数据，理想时间仅约 2.4 µs；而每次 kernel 启动/收尾本身就有约 1.5 µs 固定开销，占了理想时间的 30%+。同时它只有 128 行，40 个向量核按行切分后每核只有 2–3 行、DMA 与计算无法重叠（同上：本模式零重叠），核利用率上不去（UB 上限把每块行数卡在 2 行）。 |
| long-seq-reduce | bf16 | 失败→26.7% | **24.8%** | ❌ | 与 3d 同因（4 MB 小体量 + 固定启动开销），且行数只有 64、每核 2 行、单行需按 8192 列分 4 段串行规约；叠加 bf16 的 fp32 提升成本。该 case 的首要价值是「从集成态崩溃修复为可运行」，性能仍受小体量与核粒度限制。 |

### 参照结构 diff 检查（T-1，plateau 前强制）

- **同族先例**：`tileops/kernels/reduction/amax/amax_kernel/perf_opt/`（同结构 amax，已调优）→ 逐结构 diff 得到两个差异点：①「网格规模」（amax 二轮已改 full-40；本 op Round 4/5 已实测并采纳 full-40，无差异）；②「S6 静态行块 extent」（amax 二轮新增；本 op Round 8 搬入后 `ab_test` 6 对 tie，**未复现** amax 的 −3.5%，不采纳）。另做了同状态交叉校验：amin base 16.44us vs amax final 16.47us（tie），排除「amin 代码本身更慢」的可能。
- **pattern-library `PL-1.20-nsplit-perrow-partial`**（elementwise.md，「小 M 大 N 2.2–2.6×」）→ 与本 op 3d/lseq（小 M 大 N）形态相近，差异点 = **N 维跨核切分 + 跨核 partial 合并**。**未采纳为 Stage 4 候选**的理由（实测/文档条款）：reduce 族行归约的当前设计是**单核按行全归约**（无需 workspace），N-split 需跨核 partial 合并——要么第二次 kernel launch（+~1.5 µs 固定开销 ≈ 4 MB 体量的理想时间 2.4 µs 之 60%，净收益 <2×），要么核内 grid-sync（本设计族/模式不支持，属设计层改动）。故判定其为**设计层结构而非可解参数**，但**估不到 >2× 可达收益**（达不到 `[DESIGN_LIMIT]` 的第二条件）→ **不产出** `perf_feedback.md`。hidden bf16 同理：去 vcast 的极限收益 ≈ 16.5→13.5us = 1.22×，也达不到 >2× 门槛。

## Skill Retrospective

- **流程问题 1（多 kernel 工件的单文件对账假设）**：`signal-registry.md §5` 与 gate 4 假定一个算子只有**一个** `perf_opt/{op}.py`。TileOPs 集成算子的 Stage 4 共享一个 `perf_opt/` 目录却可含多个函数文件（本 op：resident + tiled，wrapper 双 import），本日志按 amax 先例把 `_simple_reduce_kernel.py` 做成 canonical 单入口（顶层 re-export tiled 工厂），并把 long-seq 的 `kernel_id` 记为 `perf_opt/_simple_reduce_kernel.py::_simple_reduce_kernel_tiled::main` 以通过单值对账。建议标准显式支持 `kernel_ids` 集合或 `kernel_id` 粒度对账。
- **流程问题 2（baseline 不可运行的 workload 无有效 baseline 记录）**：集成态 long-seq 直接 crash，本日志以「同一 baseline kernel + 预算合法 tile_n」补一条可对账 baseline 并在 reason 注明；建议标准明确此类情形的 baseline 口径。
- **证伪更正记录（对照 T-2）**：同族 amax 的 perf_opt 曾把「整除-32 vs 满-40 网格」判为平区（并把一次疑似收益记为 run-state 双态噪声）。本 op 在同工具链/设备上以 4 次交错 A/B 得到 bf16 一致 +7.2%（16.39 vs 17.65 µs，4/4 run 更快）→ 说明该网格结论**算子相关**：bf16 的向量瓶颈使满核收益更明显，不能跨算子照搬，须逐 op 实测。
- **流程问题 3（顺序 A/B 的位置偏置，R8 实测）**：`msprof op` 单发存在 per-process 位置/run-state 双态。同一 kernel 连续两次测量第二次稳定慢 1.7–2.8%；固定「先 A 后 B」的 5 对得到 A 5/5 胜，反序 4 对得到 B 4/4 胜——**臂效应被位置效应完全掩盖**。`ab_test.py` 的交替顺序 + 偶数对（本 op 用 6 对）可抵消；但仍建议工具在结果里附「首/次位分别的中位」，或在 `--pairs` 为奇数时提示合并中位有偏。
- **证伪更正记录 2（S6 静态 extent 跨算子可迁移性）**：amax 二轮报告 S6 为 −3.5%（16.40→15.83us），但那是 **22 个非配对样本**；本 op 用 6 对交替顺序复测为 tie（16.46 vs 16.53us，4/6 胜，−0.42%）。两者 DMA 形态完全相同（min vs max 只改 reduce op），说明该 −3.5% 更可能是 run-state 漂移而非 S6 本身。**S6 在本工具链的 persistent 行归约上无可测收益**。

- **证伪更正记录 3（R9 重验 R3 的「bf16 同域 reduce_min 不支持」结论）**：R3 记录的失败指纹是 `TVMError: Check failed: buffer_shape0[i] == buffer_shape1[i]`，本日志当时归因为「bf16 同域 reduce 不支持」。R9 在 CANN 9.0.0 / tilelang `0.1.2+da4f6a4` 上**重验并证伪该归因**：① bf16 `reduce_min` **可以编译**、随机数据 **bit-exact**；② 那个 shape 检查的真实来源是 **`VcastCodegen`**（bf16 reduce 会被自动插 `vcast`），只在 **reduce 目标为切片** 时触发——行切片报同一个错，列切片**编译通过但静默错值**；③ 更关键的是：bf16 reduce 的真实语义 = `vcast(bf16→f32) + vreduce<min>(f32) + vcast(→bf16)`（IR 反查），即**「不支持」的实质是「不原生」**，本实现显式 vcast 的写法反而是最优写法（比编译器自动插提升快 7.66%）。**教训：编译器约束类结论必须绑定工具链戳 + 必须用 IR 反查确认「不支持 / 自动提升 / 静默错值」三者的区别**，否则会像本条一样把「自动提升」误记为「不支持」，从而错误地把一个**已经最优**的实现当成待优化的瓶颈。
- **流程问题 4（证伪协议的反向失效：把「已最优」误判为「待优化」）**：R8 的终态「bf16 的 vcast 税是设计族天花板、差 1.4pp」本身是对的，但归因文字（「API 层 bf16 同域归约缺失」）让人以为存在一条「原生 bf16 归约」的可达路径。R9 花了 3 条路线才确认那条路径不存在。建议 pattern-library 的 traps 条目补一条「bf16 归约在本工具链恒等价于 fp32 提升，显式 vcast 优于隐式」的反向条目（R9 已回写，见下）。
- **流程问题 5（`ab_test` 与 `msprof` 的目录权限/环境变量前置条件）**：`ab_test.py` 用 `os.makedirs` 建 profile 目录，默认 umask 0002 → 775，`msprof op` 直接拒绝（"is writable by any other users or group users"）并静默不产 CSV（脚本报 `no OpBasicInfo rows`）；另外 msprof 只接受**可执行文件**作为首个 token，`ENV=1 python ...` 形式会被当成不存在的程序。对策：跑 `ab_test` 前 `umask 077`，需要环境变量时用一个小 `.sh` 包一层（本目录 `_arm_pipe*.sh`）。建议 `ab_test.py` 内部 `os.makedirs(..., mode=0o700)` + `os.chmod`。

### 价值点提案（供 `tilelang-skill-evolver` 蒸馏）

- **D / vp_type=D（实测数据）——「bf16 同域 reduce_min 在 CANN 8.5.2 仍不支持」**：机制 = `T.reduce_min` 对 bf16 src/dst 触发 `TVMError: Check failed: buffer_shape0[i]==buffer_shape1[i]`，必须 `vcast` 到 fp32。证据：`_variants_r2.py::r2_bf16_direct` 编译失败（Round 3）。repro：`_variants_r2.py`。toolchain_stamp: tilelang 0.1.2 dev-build + CANN 8.5.2 + 910B3。回写位置：本日志 + `_variants_r2.py`。
- **D / vp_type=D（实测数据）——「persistent reduce 网格规模结论算子相关（同族 amax 整除-32 vs amin 满-40）」**：机制 = 满 40 核对 bf16（向量瓶颈）一致 +7.2%（4 次交错 A/B），对 fp16 +2.8%，对 3d 平区；同一 amax 结构曾判平区。证据：Round 4 表。repro：`_variants_r1.py::r1_contig`（对照 deliverable 的整除网格）。toolchain_stamp 同上。回写位置：本日志 + `_variants_r1.py`。
- **P / vp_type=P（优化模式，正向）——「reduce 类逐任务写出 → 每核一次批量化写出（同族第二证）」**：机制 = persistent reduce 核每任务迭代单独发一次 UB→GM 写出时，MTE3 单次固定开销随任务数线性累积；把各任务结果暂存扁平 UB 并在核尾每核一次写出，`aiv_mte3_ratio` 由 0.586→0.017（fp16 hs）、0.564→0.078（3d）。证据：Round 0/1 表。repro：`_variants_r1.py::_make_resident`（before = baseline 逐任务写 / after = out_acc 批量写）。toolchain_stamp 同上。回写位置：本日志 + `_variants_r1.py`（第二独立证据，升 verified）。
- **P / vp_type=P（优化模式，正向）——「tiled 路径 tile_n 包络 clamp 消除 wrapper/kernel 预算口径不一致」**：机制 = wrapper 按 bf16=2B/elem 给出 tile_n=16384，迁移 tiled kernel 按 B=6（x_ub 2B + fp32 work 4B）预算断言拒绝 → 崩溃；在 **kernel 内**把 tile_n clamp 到 `2*tile_n*B <= 96KiB`（bf16 → 8192）使两侧口径一致并消除大 staging 的 auto-multi-buffer 竞态。证据：long-seq 由 crash → 8.99 µs；`profiles/R1_clamp_lseq_bf16`。repro：`_variants_r1_tiled.py::t1_clamp`（S0 前后）。toolchain_stamp 同上。回写位置：本日志 + `_variants_r1_tiled.py`。
- **R / vp_type=R（流程规则）——多函数 TileOPs 算子 perf_records 的单文件对账假设**：见「流程问题 1」。
- **D / vp_type=D（实测数据）——「S6 静态行块 extent 在 910B3/CANN 8.5.2 的 persistent 行归约上无收益」**：机制 = `M % bm == 0` 时用编译期常量 extent 取代 `real_m = T.min(bm, M-off_m)`，在 amax 二轮曾报 −3.5%（22 个非配对样本）；本 op 以 6 对交替顺序 A/B 复测为 tie（16.46 vs 16.53us，−0.42%，4/6），GM 直拷版 16.445（−0.51%）、动态 trip count 16.545（+0.30%）同为 tie。证据：Round 8 表 + `logs/round8/ab_test_*.json`。repro：`_simple_reduce_kernel_round8_s6.py`（before/after diff 仅 `static_m` 分支）。toolchain_stamp: tilelang 0.1.4 dev-build (ab62f46) + CANN 8.5.2 + 910B3。回写位置：本日志 + 分支文件。
- **P / vp_type=P（方法/反模式）——「msprof 单发顺序 A/B 的位置偏置可完全反转结论」**：机制 = 同一 kernel 连续两次测量第二次稳定慢 1.7–2.8%；固定顺序 A→B 的 5 对得 A 5/5 胜（+5.55%），反序 4 对得 B 4/4 胜（−7.19%）。对策 = `ab_test.py` 交替顺序 + 偶数对（6 对），并分别报告首/次位中位。证据：Round 8「测量协议修正」段 + `profiles/ab_round8_bf16/`（顺序，偏置）vs `profiles/ab_round8_s6_v2/`（交替，tie）。repro：`_arm.py` + `.agents/tools/ab_test.py --pairs 6`。toolchain_stamp 同上。回写位置：本日志 + `_arm.py`。

- **D / vp_type=D（实测数据，**新增 R9**）——「hivm 层无 bf16 min：`reduce_min(bf16)` 与 `vmin(bf16,bf16)` 都被无条件降为 vcast+f32 op+vcast」**：机制 = IR 反查（`TILELANG_DUMP_IR=TRUE`）；后果 = bf16 归约的每元素片上流量恒为 10 B（vcast 2R+4W + f32 reduce 4R），显式 vcast 写法比让编译器自动插提升**快 7.66%**（18.20 vs 16.905us，6/6 交错对，p=0.0312）。证据：Round 9 路线 A 表 + 「能力探测」表。repro：`/tmp/opencode/amin_probe/probe_reduce4.py`（bit-exact 探测）、`dump_vmin.py`（IR）；本目录 `_simple_reduce_kernel_round9_native_bf16.py`（A1 候选）。toolchain_stamp: tilelang `0.1.2+da4f6a41` + CANN 9.0.0 + Ascend910_9362。回写位置：本日志 + 候选文件。
- **D / vp_type=D（实测数据，新增 R9）——「int16 位域 min 在 910_9362 上原生但吞吐仅 fp32 链的 ~1/4.5」**：机制 = i16 的 `vshr/vand/vxor/vreduce<min>` 全部原生（IR 无 cast），但等宽 I/O 微基准 39.195us vs bf16 臂 17.040us（+130%，6/6 对）。→ 任何「用整数位模式绕开 fp32 提升」的设想在本硬件上结构性亏损（3 次全宽 pass = 12 B/元素 vs vcast 的 6 B/元素，且 i16 单 op 吞吐低）。证据：Round 9 路线 B 表。repro：本目录 `_r9_probe_dtype.py`（dtype=int16 臂）+ `_arm2.py` + `ab_r9_dtype/`。toolchain_stamp 同上。回写位置：本日志 + probe。
- **D / vp_type=D（实测数据，新增 R9）——「`T.Pipelined(num_stages>=2)` 在本 loop 形态上是 no-op」**：机制 = stages=1 与 stages=2 的 IR 里 `memref<4x4096xbf16>` 分配个数完全相同（无第二份 buffer），端到端 16.975 vs 17.155us（tie，3/6，方向不一致）。推论 = 本模式（Developer + auto sync）下 persistent 归约**没有跨任务 DMA/vector 重叠**，`aiv_time ≈ mte2_time + vec_time` 在 fp16/bf16 上同时成立；「减 UB 足迹就能重叠」的假设随之证伪（bm=5 去 x_ub 后 UB 80KB 可双缓冲，实测仍 +4.48% 回退）。证据：Round 9 路线 C 表 + 现象表。repro：本目录 `_r9_probe_pipe.py` + `_arm_pipe2.sh` + `ab_r9_pipe2/`；IR 对比 `dump_pipe.py`。toolchain_stamp 同上。回写位置：本日志 + probe。
- **CG / vp_type=D（能力缺口，新增 R9，已按 AGENTS.md 记入 `.agents/evolution/capability-gaps.md`）——`T.view` 位重解释不可用**：GM 侧 `T.view(x_bf16, dtype="int16")` → `memref.memref_as_ptr` codegen 失败；UB 侧 `T.view(alloc_shared_bf16, dtype="int16")` → `InternalError: GetVarValue(op->buffer_var.get())`。堵死一切「在片上按位重解释」的实现路径。证据：`probe_reduce2.py` / `probe_view_shared.py`。repro：`/tmp/opencode/amin_probe/probe_view_shared.py`（自带断言形态：期望 codegen 失败）。toolchain_stamp 同上。
- **R / vp_type=R（流程规则，新增 R9）——`ab_test.py` 需自建 0700 profile 目录**：见「流程问题 5」；umask 0002 环境下 `os.makedirs` 产出 775 会让 `msprof op` 静默不产 CSV，表现为 `no OpBasicInfo rows`（易被误读为 kernel 崩溃）。
- **R / vp_type=R（流程规则，新增 R9）——gate 4 的「每 workload 恰一条 final」与 append-only 的冲突需要明确处置**：换定稿文件名后必须重采 final，而旧 final 行不能删除、又不能留成第二条 final（`S4-PERF-RECORDS-RECON` 会判「必须恰有一条」）。本轮按 conductor 指令把旧 final **降级为 `phase=merged` + note**，并在本日志记明；建议 `signal-registry.md §5` 显式规定该降级操作是 append-only 的合法例外（否则执行者会卡在两条规则之间）。

## Issues（记录，不阻塞）

- 集成基线 long-seq 崩溃（wrapper `_tile_n_for_block_m` 用 `_primitives.compute_tile_n`（bf16 2B/元素）给出 tile_n=16384，而迁移 tiled kernel 按 B=6 预算）——本 Stage 4 在 **perf_opt tiled kernel 内**以 S0 clamp 使两侧口径一致，case 恢复可运行；未改 wrapper（conductor 采纳动作）。
- bf16 hidden-state 未稳定达 60%：3 次生产单发为 57.34 / 60.37 / 56.86%（中位 57.34%，最好 60.37%）。API 层 bf16 同域归约缺失（`T.reduce` 文档 DataType 表 bf16 为 ×）导致 4.73us 的 vcast+fp32 reduce 税（占 kernel 28%），属设计族内实际天花板；去 vcast 的极限收益约 1.22×，未达 `[DESIGN_LIMIT]` 的 >2× 门槛，故不产出 `perf_feedback.md`。
- Round 8 的 4 个候选（S6 / GM32 / DYNLOOP / BM5）全部 tie 或回退，未改动定稿；实验分支保留在 `perf_opt/_simple_reduce_kernel_round8_*.py`。
- **R9（2026-10-26）**：hidden-state bf16 仍未达 60%（生产单发 16.86us / 55.28% @1.8 TB/s 口径）。三条路线 + 两条结构补测全部实测证伪（bf16 无原生 min、i16 位域慢 2.3×、切片折叠静默错值、`T.Pipelined` no-op、去 x_ub 减 UB 回退 4.48%），**无剩余可达优化点**；benchmark 侧唯一可达的 1–2 µs 在「DMA 与向量链重叠」，而该能力在本编程模式不可用（fp16 对照组同样零重叠）。→ 这不是参数级不足，但**可达加速比 ≈ 17.0→13.4us = 1.27× < 2×**，不满足 `perf_feedback.md` §1 第二条件，故**不产出** `perf_feedback.md`、不发 `[DESIGN_LIMIT]`（与 R8 判定一致，R9 为其补充了 IR 级证据）。
- **R9 口径漂移记录**：本轮 msprof 峰值参考 1.8 TB/s（R8 及更早为 1.75），同一 kernel 的 Ratio 相差 ~1.4pp；`perf_records.jsonl` 只记 Task Duration(us)，不受影响。
- **R3/R8 的历史归因已被 R9 更正**：见 Skill Retrospective「证伪更正记录 3」——「bf16 同域 reduce_min 不支持」改为「不原生（自动提升 fp32），显式 vcast 已最优」；`_variants_r2.py` 作为历史候选保留不改。

---

# v3_20261101_smalltask -- 小任务 case（long-seq / 3d）专项提速（AminFwdOp）

> 与同族 `AmaxFwdOp` 同一轮工作、同一命名空间（`_run_batch.py` / `cases_v3*.txt` / `_variants_s4.py` / `_variants_s5.py` 均由 amax 侧复制，`op_kind="amin"` 走 `T.reduce_min`）。环境：独占 Chip0、`TILEOPS_MSPROF_TMPDIR=/tmp/opencode/msprof_amx/tmp`、每次 msprof `env -u TILEOPS_MSPROF_OUTPUT_DIR`。备份：`history_version/20261101_smalltask/`。

## Performance Test Data（baseline，Round 0，本会话）

| workload | shape / dtype | 配置 | Task Duration(us) | Ratio(%) | L0 |
|---|---|---|---:|---:|:-:|
| hidden-state-reduce | 2048×4096 fp16 | bm=4 | 16.260 | 56.15 | PASS |
| hidden-state-reduce | 2048×4096 bf16 | bm=4 | 16.740 | 54.89 | PASS |
| long-seq-reduce | 64×32768 bf16 | bm=1, tile_n=16384→clamp 8192 | **9.040** | **24.74** | PASS |
| 3d-multidim-reduce | 128×16384 fp16 | bm=1→promote 2 | **5.380** | **41.32** | PASS |

## Round 1/2 -- 与 amax 同源的候选（结论与证据）

| 分支 | 结果 | 结论 |
|---|---|---|
| 段数 4→2（bm=2, tn=16384）/ 4→1（bm=1, tn=32768） | 编译失败 `VcastCodegen` 子区域形状校验（chunked vcast 不可用） | 与 amax 同一编译器限制（`_variants_s4.py` 为 op 无关文件，同一份代码测得） |
| 3d 任务数（bm=4 / `_res_rows` / tiled 1 段） | 5.26–5.66 µs 平区（amax 侧交错 A/B 3 reps）；amin 侧 `_res_rows` 结构同构 | **不采纳**（平区内，无跨噪声增益；resident 路径因此一行未改） |
| 手工 depth-2 ping-pong（S5） | `507034 vector core timeout` | blocked-runtime，不再重试 |

## Round 4 -- 采纳：2 行流水段（S7）合入 streaming 模块

`perf_opt/_simple_reduce_kernel_tiled.py` 与 amax 侧**同构改动**（`_stage_fits` + factory 期 `promote_bm` + `(bm, tile_n)` 泛化），差别只有一处：**保留 amin 的 NaN 语义合并链**——每段 `reduce_min(clear=True)` 进 `tile_acc`，再用逐元素 `T.min` 并入 `acc`（`reduce_min(clear=False)` 在 fp32 跨 `T.serial` 段会丢 NaN，L0-11 覆盖）。该合并是逐行的，天然可泛化到 bm 行。

合入前后交错 A/B（**3/3 配对全胜**，同会话）：

| 算子 | 旧（bm=1） | 新（bm=2） | Δ | L0 |
|---|---:|---:|---:|:-:|
| AminFwdOp long-seq | 9.12 / 9.20 / 9.16（中位 **9.16**，24.53%） | 6.50 / 6.42 / 6.54（中位 **6.50**，34.99%） | **−29.0%** | PASS |

非回退检查：
- `_simple_reduce_kernel_tiled.py --level all` → **L0 / L1 / L2 / Boundary 全过（max_diff=0.00e+00）**，含 L0-11 NaN 一致性、尾块与非整除形态（`logs/v3/tiled_level_all.log`）。
- `AminFwdOp.py --level all` 全过（resident 路径 sha256 `b3f7a691…` 与备份逐字节相同）。
- hs / 3d 为同码路，本会话实测 hs fp16 13.10–16.48（双态）、hs bf16 16.64–16.86、3d 5.32–5.48。

## Final Summary（v3_20261101_smalltask）

- 定稿：`perf_opt/AminFwdOp.py`（sha256 `b3f7a691…`，**未改**）+ `perf_opt/_simple_reduce_kernel_tiled.py`（**S7 2 行流水段**，本轮唯一改动）。
- **long-seq-reduce bf16：9.16 → 6.50 µs（−29.0%），Ratio 24.5% → 35.0%**（同会话 3 对交错 A/B，3/3 配对胜）；同 workload 全部同会话样本中位 6.56 µs / 34.78%。
- 3d-multidim-reduce fp16：不改（目标 50% 不可达；同一「launch 地板 2.80 µs + DMA 4 MB@1.96 TB/s = 4.84 µs > 4.66 µs」论证适用于本 kernel，见 amax opt_log 的 3d 天花板表）。
- hidden-state fp16/bf16：不改（同码路）。
- 两模块 `--level all` 全过；`perf_records.jsonl` 的 4 条 `phase=final` 同属候选 `v3_smalltask_s7`。
- 中止原因：`success`（long-seq 目标 40% 未达，报实际 35.0%）。

### 本轮会话态（`BP_run_state_bimodality`）

同一 `AminFwdOp.py`（sha `b3f7a691…`）本会话内 hs fp16 实测 16.26（Round 0）→ 14.62 → 16.42 → 13.10（追加 3 次，67.5–69.2%），漂移 ±20%，与批量报告里的 67% 属同一快态。所有裁决只用同会话交错 A/B。

## Skill Retrospective（v3 追加，因冻结暂缓入队）

- **P（优化模式，正向）——「流式归约的多行流水段」**：把 streaming 归约的 stage 从 1 行扩到 bm 行（`(bm, tile_n)`），在 UB 允许时把每核 `T.reduce_*` 次数按 bm 整除、把 staging DMA 从 16 KB 抬到 32 KB（每核 MTE2 38.6 → ~52 GB/s）。long-seq bf16 9.08/9.16 → 6.46/6.50 µs（−29%）。**关键约束**：多行 bf16 段的双缓冲包络会超 UB 并表现为 `507034 vector core timeout`（不是编译错误），必须在 `promote_bm` 时关 `npuir.enable_auto_multi_buffer`。repro：`_variants_s4.py::_tiled_rows` 的 before/after 即 `_variants_v3_oldtiled.py` vs `_simple_reduce_kernel_tiled.py`（本轮 S7）。toolchain_stamp: tilelang dev-build + CANN 9.0.0 + Ascend910_9362。
- **D（实测数据）——「chunked `T.vcast` 子区域被 codegen 形状校验拒绝」**：`T.vcast(x_ub[0:bm, 0:chw], x_work)` 当 `chw < tile_n` 时 `CodeGenTileLangNPUIRDEV::VcastCodegen` 报 `Check failed: buffer_shape0[i]==buffer_shape1[i]`；因此 bf16 stage 只能整宽 cast，`tile_n` 不能靠"cast 分块 + 整宽 reduce"翻倍。这条直接封死了 long-seq 的「4 段 → 2 段」（目标 40% 的实际拦路项）。repro：`_probe_*` 见 `logs/v3/v3_r1_ls_bm2_tn16k.log`。toolchain_stamp 同上。
- **D（实测数据）——「单 kernel launch 地板 2.80 µs（40 块）」**：空转（每核写 1 元素）kernel 的 `msprof op` Task Duration 2.80 µs；4 MB 纯搬运 4.84 µs（聚合 1.96 TB/s）。两者之和构成 4 MB 级 case 的硬下界，解释 3d/long-seq 的 40%/50% 门槛为何是**口径上限**而非调优不足。repro：`_variants_s4.py::_probe_floor` / `_probe_dma`。toolchain_stamp 同上。
- **R（流程规则）——「同会话交错 A/B 必须是唯一裁决口径」**：本任务同一 sha 的 kernel 在 2 小时内出现 13.18 / 14.54 / 16.28 / 14.58 µs 四档（±18%），任何跨时段的"前后"对比都会得出错误结论；`<5%` 平区裁决必须按 T-2 做同会话多配对。

## Round 5 -- 生产 CLI 复测（官方口径，`msprof`）

`python -m tileops.reporting.cli run --op AminFwdOp --prof-mode msprof` → `reports/tileops/20261030_115346_097576_AminFwdOp/`，status=**passed**。

| case | dtype | v2 报告 | v3 本次 CLI | Latency(us) | 说明 |
|---|---|--:|--:|--:|---|
| hidden-state-reduce | fp16 | 66.96 | 68.34 | 13.860 | 代码未改（sha 相同），会话抖动 |
| hidden-state-reduce | bf16 | 53.32 | 50.01 | 18.640 | 代码未改，慢态单发 |
| long-seq-reduce | bf16 | 25.61 | **35.74** | **6.520** | **S7 改动** |
| 3d-multidim-reduce | fp16 | 43.80 | 41.32 | 5.360 | 代码未改 |
| 平均 | — | 47.43 | **48.85** | — | +1.42 pp |

回归命令与日志（本轮全部重跑）：
- `python perf_opt/AminFwdOp.py --level all` → 全过（`logs/v3/resident_level_all.log`）
- `python perf_opt/_simple_reduce_kernel_tiled.py --level all` → 全过（`logs/v3/tiled_level_all.log`）

> 冻结自证：本任务**未回写** `pattern-library/`、`.agents/evolution/queue.md`、`stats.md`（任务约定覆盖 skill 的 ED-C 回写义务）；新价值点**因冻结暂缓入队**，证据只落在本目录。`python3 .agents/tools/frozen_check.py` → PASS。

## Round 6 -- S7 复核（先证伪再采纳）：与 AmaxFwdOp 同源的证据链重建

**触发**：上游复核指出"工作区 sha256 与 HEAD 逐字节相同 ⇒ 可能没改过"，且 `perf_records.jsonl` 似乎未追加。逐条核实如下（结论与同族 amax 一致，详见 `../amax/amax_kernel/perf_opt/opt_log.md` 的 Round 6）。

### 6.1 定位：S7 在，且"sha == HEAD"是备份提交晚于编辑造成的

| 事实 | 证据 |
|---|---|
| 本轮起点 HEAD（`abab30a`）不含 S7 | `git show abab30a:.../perf_opt/_simple_reduce_kernel_tiled.py` = `380b3a89…`，与 `perf_opt/_variants_v3_oldtiled.py` **逐字节相同** |
| 含 S7 的工作区文件产生于本轮 | mtime **2026-10-30 11:19:31**；`_stage_fits` L~166、`promote_bm` L220、`bm` 提升 L240 |
| 该编辑在 **11:46:12** 的备份提交 `e8d3157` 入库 | `git show e8d3157:...` = `59eb5363…`（= 工作区），提交时间晚于 mtime |
| 4 条 `phase=final` 已在 **11:49:06** 追加 | 99 行中的末 4 行（`candidate_id=v3_smalltask_s7`） |

⇒ **「工作区 sha == HEAD」不能作为"没改"的证据**；S7 真实落地。生效路径与 amax 同构：wrapper 传 `block_m=1, tile_n=16384` → `_clamp_tile_n` 用 `_B_ELEM['bfloat16']=6` 收紧到 **8192** → `_stage_fits(2,8192,bf16)=114688 ≤ 190 KiB` → `bm=2`。

### 6.2 同会话交错 A/B（7 对，逐对交替顺序）+ 3d 对照

`_measure.py`（`msprof op`，launch-count=10 / warm-up=5）：

| 算子 | 旧（bm=1，7 次） | 中位 | 新（bm=2，S7，7 次） | 中位 | Δ | 配对胜 |
|---|---|---:|---|---:|---:|---:|
| AminFwdOp | 9.16 / 9.34 / 9.22 / 9.30 / 9.38 / 9.18 / 9.34（24.5%） | **9.30** | 6.62 / 6.62 / 6.48 / 6.66 / 6.50 / 6.68 / 6.60（34.5%） | **6.62** | **−28.8%** | **7/7**（p=0.0078） |

3d 对照（resident 模块，S7 未触及，两臂同一文件）：amax 同源控制实测 Δ −1.2%（2/5 胜，p=0.81，±3% 抖动带）⇒ 3d 未被牵连。机制隔离（amax 侧 PROBE `_probe_s7isol_bm1_nomb.py`）：单独关 `enable_auto_multi_buffer` 只有 −0.6%（9.34→9.28），**收益来自 2 行 stage 本身**。

> 否证「run-state 双态」假设：交替执行的 14 次测量 7/7 同向、每对 Δ 恒定 2.6–2.8 µs，会话态漂移解释不了。

### 6.3 3 次 op 级报告复测（最终达标判定数字）

`python -m tileops.reporting.cli run --op AminFwdOp --prof-mode msprof` × 3（`reports/tileops/20261030_124010 / 124301 / 124551_AminFwdOp`，均 status=passed、正确性 14/14）：

| case | dtype | run1 | run2 | run3 | **中位 Task Duration(µs)** | **中位 Ratio** |
|---|---|---:|---:|---:|---:|---:|
| hidden-state-reduce | fp16 | 13.44 | 13.66 | 13.90 | **13.66** | 66.01% |
| hidden-state-reduce | bf16 | 17.44 | 18.54 | 17.30 | **17.44** | 54.44% |
| **long-seq-reduce** | bf16 | 6.50 | 6.54 | 6.58 | **6.54** | **35.31%** |
| 3d-multidim-reduce | fp16 | 5.40 | 5.38 | 5.42 | **5.40** | 42.37% |
| **平均** | — | — | — | — | — | **49.53%** |

**与上一轮声明的差异（如实更正）**：long-seq 由"35.74% / 6.52 µs"更正为 **35.31% / 6.54 µs**；算子平均由"48.85%"更正为 **49.53%**（本轮为 3 次复测中位，上一轮为单次 CLI 单发）。机制结论不变（S7 −28.8%）。

### 6.4 回归

- `python perf_opt/AminFwdOp.py --level all` → **All check passed**（`logs/v3_s7verify/level_all_AminFwdOp.log`）
- `python perf_opt/_simple_reduce_kernel_tiled.py --level all` → **All check passed**（含 NaN 语义 L0-11 与 `(4,13000) bf16` S7+尾块）（`logs/v3_s7verify/level_all_amin_tiled.log`）
- 3 次 op 级报告 status=passed、正确性 14/14。

## Final Summary（v3_s7_reverify）

- 定稿文件未变：`perf_opt/AminFwdOp.py`（sha256 `b3f7a691…`，未改）+ `perf_opt/_simple_reduce_kernel_tiled.py`（sha256 `59eb5363…`，S7）。本轮**只重做证据链与数字复核，未改任何 kernel 代码**。
- long-seq-reduce bf16：7 对交错 A/B **9.30 → 6.62 µs（−28.8%，7/7，p=0.0078）**；3 次 op 级复测中位 **6.54 µs / 35.31%**。
- 算子平均 Ratio **49.53%**；4 例中 0 例达 60%（hs fp16 66.01%、hs bf16 54.44%、3d 42.37%、long-seq 35.31%）。
- 中止原因：`success`（复核范围内结论闭合）。
