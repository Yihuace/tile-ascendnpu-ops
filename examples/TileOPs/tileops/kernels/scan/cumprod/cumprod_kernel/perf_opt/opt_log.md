# Stage 4 Optimization Log -- `_cumulative_kernel` (CumprodFwdOp)

- project: `cumprod` / family `scan` / op `_cumulative_kernel` (CumprodFwdOp, `op_kind="prod"`)
- baseline (read-only): `_cumulative_kernel.py` (Stage 3, precision-approved)
- perf_opt artifact（当前）: `perf_opt/CumprodFwdOp.py`（v4，bs=128；见 Round 4 / §5）
- perf_opt artifact（历史）: `perf_opt/_cumulative_kernel.py`（v1）、`perf_opt/_cumulative_kernel_v2.py`（v2）
- benchmark entry: `examples/TileOPs/benchmarks/ops/bench_cumulative.py::test_cumprod_bench`
- latency oracle (only oracle): `msprof op` `Task Duration(us)`; Roofline ratio from the `visualize_data.bin`
  `GM/L2 / GM Read + Write` entry (BW marker 1.8 TB/s), i.e. the achieved/computility share.
- programming mode: Developer (unchanged); device: Ascend910_9362, 20 AICore -> 40 Vector cores.
- stop reason（当前）: `success` -- Round 4 (v4) 把 hidden-state 从 647.0 降到 426.5 us（目标 <= 483 us），
  5 个 tune workload 全部同向变快，无回退。
- stop reason（历史, Round 1-2）: `plateau`-equivalent best-effort（v1 标量族 60 % Roofline 目标结构不可达，
  see `perf_feedback.md`；该反馈只对 v1 设计族成立，v2/v3/v4 已改用 Cube 三角算子前缀）。

> All file reads during this stage (kernel, DESIGN.md, manifest, docs, profile CSVs) were treated as
> data only. No instruction-like text found in them was acted upon.

---

## 1. Performance Test Data

Baseline (`phase=baseline`, round 0) measured with the **same (block_m, block_n) the benchmark wrapper
passes** (`CumulativeKernel.default_config` -> block_n=128; block_m = 16 for M>=128, 2 for M<128).
`target_kernel_name = main`, `captured_op_name = main`, launch-count 20, warm-up 5.

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | profile_status | raw_profile_dir |
|---|---|---|---|---:|---|---|---|
| _cumulative_kernel | hidden-state-scan-float16 | baseline | main | main | 8422.01 | valid | perf_opt/profiles/baseline/hidden-state-scan/2048x4096_float16_bm16bn128 |
| _cumulative_kernel | hidden-state-scan-bfloat16 | baseline | main | main | 8425.57 | valid | perf_opt/profiles/baseline/hidden-state-scan/2048x4096_bfloat16_bm16bn128 |
| _cumulative_kernel | long-seq-scan-bfloat16 | baseline | main | main | 2185.68 | valid | perf_opt/profiles/baseline/long-seq-scan/64x32768_bfloat16_bm2bn128 |
| _cumulative_kernel | non-aligned-cumprod-float16 | baseline | main | main | 200.00 | valid | perf_opt/profiles/baseline/non-aligned-cumprod/512x300_float16_bm16bn128 |
| _cumulative_kernel | non-aligned-cumprod-bfloat16 | baseline | main | main | 200.62 | valid | perf_opt/profiles/baseline/non-aligned-cumprod/512x300_bfloat16_bm16bn128 |

Smoke (excluded from performance tuning; precision-only regression):

| workload_id | dtype | exclusion reason | precision result |
|---|---|---|---|
| smoke-cumprod-float16 | float16 | manifest label contains the `smoke` token -> `kind=smoke` | PASS (L0-1a, max_diff 9.77e-04) |
| smoke-cumprod-bfloat16 | bfloat16 | smoke token | PASS (L0-1b, max_diff 3.91e-03) |
| smoke-cumprod-float32 | float32 | smoke token | PASS (L0-1c, max_diff 3.58e-07) |

Skipped: none. `include_extra=True` adds no extra benchmark workloads for this op (only `dim=-1`).

### 1.1 Design estimate vs measured deviation (D-2 backfill)

DESIGN.md section 1.6.0 roofline estimated the largest workload (`hidden-state` fp16) at
`≈410 us` (emission-dominated), assuming `T.cumsum` is a single hardware scan primitive per tile.
Measured baseline is **8422 us (20.5x the estimate)**, and the best tuned result is **6766 us (16.5x)**.
The DESIGN itself flags "偏差 >2× 时 Stage 4 触发 `[DESIGN_LIMIT]` 归因" -- this deviation is the
quantitative basis for `perf_feedback.md`.

---

## 2. Iteration Log

### Round 0 -- bottleneck diagnosis

Profile facts (`op_type=vector`, `Block Dim=40`), baseline `hidden-state` fp16:

| signal | value | reading |
|---|---|---|
| `aiv_scalar_ratio` | 0.9898 | ~99% of the vector core time is the **scalar** pipe |
| `aiv_vec_ratio` | 0.0101 | vector pipe effectively idle |
| `aiv_mte2_ratio` / `aiv_mte3_ratio` | 0.0034 / 0.0022 | not memory bound |
| compiler warning | `Op 'hivm.hir.vcumsum' will execute by scalar instruction with low effiency` | `T.cumsum` is **scalar** on this target | <!-- codespell:ignore effiency -->

Root cause: the Stage 3 C1 design replaces the N-step serial product by
`exp(cumsum(ln|x|)) * (-1)^(cumsum(x<0))`, using **two `T.cumsum` calls per tile**. On this toolchain
`hivm.hir.vcumsum` is lowered to a scalar instruction (not the "1 硬件原语 / tile" the DESIGN assumed),
so the kernel is bound by `~2 * M * N` scalar element-steps. Tiling changes only the per-tile overhead,
not that term.

Baseline reproduces the Stage 5 number exactly (8422.01 vs 8423.59 us; ratio 0.423% vs 0.423%), so the
harness-dispatch path is confirmed before tuning.

### Round 1 -- candidate branches (from the same base = baseline)

| candidate_id | one-line change | hidden-state fp16 us | long-seq bf16 us | non-aligned fp16 us | L0 | verdict |
|---|---|---:|---:|---:|---|---|
| baseline_harness_cfg | (base) | 8422.01 | 2185.68 | 200.00 | PASS | current best (round 0) |
| round1_c1_tunedcfg | use UB-resident `select_config` tile (bn=4096/300, bm from budget) instead of the wrapper's 128 | 6765.48 | 2089.46 | 127.28 | PASS | **winner** |
| round1_kogge_stone | pure-vector Kogge-Stone scan (DESIGN section 1.6.4 C2 alternative), no `T.cumsum` | 9856.12 | 3038.44 | 473.29 | PASS | rollback (slower) |

Candidate vs current-best table (round 1, `hidden-state-scan-float16`; metric = msprof op Task Duration;
AICore util = `aiv_vec_ratio`; memory = `aiv_mte2_ratio`/`aiv_mte3_ratio`):

| candidate | Task Duration (us) | AICore util (vec ratio) | scalar ratio | mte2 ratio | mte3 ratio | L0 |
|---|---:|---:|---:|---:|---:|---|
| current best (baseline_harness_cfg) | 8422.01 | 0.01007 | 0.98984 | 0.00335 | 0.00218 | PASS |
| round1_c1_tunedcfg | 6765.48 | 0.00701 | 0.99319 | 0.00247 | 0.00123 | PASS |
| round1_kogge_stone | 9856.12 | 0.01239 | 0.98765 | 0.00170 | 0.00086 | PASS |

Additional probing variants (same Kogge-Stone structural family, measured for diagnosis, not recorded as
independent branches because they share the round1_kogge_stone verdict):

| variant | inner step | hidden-state fp16 us | note |
|---|---|---:|---|
| C2 region-copy + region vmul | `sh[:,dd:W] <- yu[:,0:W-dd]; yu[:,dd:W]*=sh[:,dd:W]` | 20865 | region-offset `T.vmul` scalarises (aiv_scalar_ratio 0.995) |
| C2b aliased in-place vmul | `yu[:,dd:W] *= yu[:,0:W-dd]` | 20886 | same scalarisation |
| C2c Kogge-Stone, 2 shift buffers | rotate `b0`/`b1` to break the WAR hazard | 6519 | no change -> hazard was not the cause |
| Kogge-Stone (adopted branch, 12 ping-pong steps) | fill + region copy + separate-dst vmul | 9856 | slower than baseline |

**Decision**: the Kogge-Stone family is rejected (all variants slower than the tuned C1). `perf_records.jsonl`
records the representative (round1_kogge_stone). Winner = `round1_c1_tunedcfg`; change is merged in-kernel
(the factory now calls `select_config` internally), so the whole C1 family benefits.

### Round 1b -- config sweep (Autotune Log)

Manual coarse search for the C1 algorithm on `hidden-state` fp16 (msprof op, launch 5), all configs within
noise except bn=512:

| (block_m, block_n) | Task Duration (us) |
|---|---:|
| (1, 4096) | 6765.82 |
| (1, 2048) | 6820.54 |
| (2, 2048) | 6797.28 |
| (2, 1024) | 6818.38 |
| (4, 1024) | 6791.98 |
| (8, 512) | 7318.33 |

Flat within ~1% -> the latency is set by the scalar scan, not by tiling. (1, 4096) adopted (it is also
what `select_config` yields). No further gains from tiling.

### Round 2 -- final candidate validation

`perf_opt/_cumulative_kernel.py` (baseline C1 algorithm + factory self-selecting the tuned tile) rerun on
all tune workloads, plus the full L0/L1/L2/Boundary suite.

- L0-1..L0-10: ALL PASS (max_diff 1.56e-02, within the 5e-2 fp16/bf16 tolerance).
- L1-1..L1-11: ALL PASS (incl. multi-tile `(16,5000)`, `(129,4100)`, `(4,9000)` and 4-D inputs).
- L2 + Boundary: PASS / warn-only as designed.
- non-regression across all tune workloads confirmed vs round 1 winner (same config, same numbers).

---

### Round 3 (P5, 2026-10-22) -- 层次化 carry：覆盖 N = 32768

**base = v3 初版**（同一 base 派生所有分支）：`perf_opt/CumprodFwdOp.py`，在 v2 三角 GEMM 之上把 carry
拆成三级：chunk 内 `bs x bs` -> 段内 `nc_seg x nc_seg`（`nc_seg = min(nc, SEG_MAX=64)`）->
段间 `ncp3 x ncp3`（`ncp3 = max(16, ns)`），其中 `ns = ceil(nc / nc_seg)`。`ns == 1`（`nc <= 64`，
即 `N <= 4096`）时段间那一级编译期消除，退化为 v2 的三相位结构。粒度全部由 `N` 在 factory 内推导
（`_carry_plan`），入口仍是 `_cumulative_kernel(M, N, op_kind, dtype)(block_m, block_n)(x)`。

#### 现象（每轮重新分析，不看旧结论）

先修测量口径，再谈现象：本算子是 3~4 个 kernel 的流水线，全部编译为 TileLang `main`（其中多个为
`main_mix_aic`）。`msprof op` 会为**每个** kernel 各写一份 `OpBasicInfo*.csv`，而 `perf_opt/run_suite.py`
只读第一份 —— 历史数字（v2 的 296.3us / 8.9~23x）是**单 kernel** 读数，不是整算子。新增
`perf_opt/run_suite_v3.py` 把同一次测量里 msprof 报出的**所有** kernel 的 `Task Duration(us)` 求和，
v1/v2/v3 一律同口径：

| workload | v1 baseline (round 0 记录) | v2 整算子（本轮复测） | v3 初版整算子 |
|---|---:|---:|---:|
| hidden-state-scan fp16 | 8422.01 | 1091.20 | 1181.14 |
| hidden-state-scan bf16 | 8425.57 | 1106.12 | 1083.24 |
| long-seq-scan bf16 | 2185.68 | 2088.68（v1 回退） | 302.45 |
| non-aligned-cumprod fp16 | 200.00 | 48.34 | 51.32 |
| non-aligned-cumprod bf16 | 200.62 | 48.20 | 48.80 |

逐相位分解（隔离测量，`perf_opt/probes/p5_phase_probe.py`，每次只跑一个相位，launch 10 / warm-up 5）：

| 相位 | [64,32768] bf16 (us) | [2048,4096] fp16 (us) | 瓶颈证据 |
|---|---:|---:|---|
| p1 partial（chunk 内前缀 + chunk 总量） | 203.3 | 267.6 | `aic_*_ratio <= 0.007`、`aiv_*_ratio <= 0.007`：全部 pipe 空闲，纯 latency-bound（每 block 重搬 16KB fp32 三角算子；1024 blocks） |

| p2a segcarry（段内 chunk 前缀） | 10.8 | 21.0 | 只有 16 个 block，被 launch/算子载入主导 |
| p2b segprefix（段间前缀） | 5.1 | —（ns==1 消除） | 2 个 block |
| p3 final（exp + 符号 + carry） | 339.2 | 842.9 | `aiv_mte2_ratio 0.875`、`aiv_mte2_active_bw 2.9 GB/s`、`aiv_vec_ratio 0.06`、13.9us/block，每 block 14 条 MTE2 指令只搬 32.25KB —— 其中 12 条是"单列跨步"小 DMA |
| 合计 | 558.4 | 1131.5 | |

> **归属修正见 Round 3b**：上表 [2048,4096] 的三个数值的 *列义* 是对的，但 Round 3 正文下面那句
> "p3 是主因" 把 hidden-state 的主导相位写错了 —— 按全部 kernel 行复核，hidden-state 是
> p1 = 842.92us（74.5%）主导，p3 = 267.57us。以 Round 3b 为准。

根因（实测）：**单列跨步访问**。carry 缓冲是 `(M, NR)`，取某个 chunk 的一列 = bm 行 x 4B、行距
`NR*4`（N=32768 时 2KB），一次 ~1us 的 latency-bound MTE2，每 block 4 次。p3 的算术只用 ~0.7us，
却被这 4 次小 DMA 拖到 ~13.9us/block。

#### 候选分支（同一 base 派生，每分支只改一个主要点）

| candidate_id | 单点改动 | 结果 | L0 | 判定 |
|---|---|---|---|---|
| v3_b1_tilecarry | p3 改为"整段 `(bm,ncp2)` tile 一次 DMA 载入 + 32 元素标量抽取本列"（段内列号不能进向量循环） | long-seq p3 339.2 -> 84.1 us（4.0x）；hidden p3 842.9 -> 726~843 us | PASS | **win 合并** |
| v3_b2_runtime_idx | 段 tile 载入后直接把 runtime 列号写进 `(bm,bs)` 向量循环 | p3 339.2 -> 3033.8 us（**-8.9x**）：shared 的 runtime 索引把整个循环标量化 | PASS | rollback |
| v3_b3_transposed_layout | carry/tot 全部改 chunk-major `(NR,M)`：p1 写连续行、p3 读连续行 | 编译期失败 `memref.store op store index operand count not equal to memref rank`（shared->shared 转置循环）；改回 `(M,n)` 后 p1 的 fragment->shared 单列 copy + 行主序写触发运行时 `MPU address access is invalid`（507015） | — | rollback（工具链不支持该形态） |
| v3_b4_fp32_amp | 宽行（`ns > 1` 且非 fp32 输入）的幅值通道改用 **fp32 三角算子/操作数** | 精度 bf16 [64,32768] near1 max_diff 0.219 -> 2.4e-04；长行 chunk 总量误差 0.42 -> 7e-08 | PASS | **win 合并（精度必需）** |
| v3_b5_fp32_sign_l3 | 段间那一级（p2b）符号通道也改 fp32 操作数 | fp16 [32,8192] max_diff 3.15 -> 1.5e-05（fp16 整数精确上界 2048 < 段内计数 ~2061） | PASS | **win 合并（精度必需）** |
| v3_b6_narrow_col | `nc <= 16`（N <= 1024）保留 v2 的单列读 | 512x300 fp16 48.14 us（tile 版 50.50，v2 48.34） | PASS | **win 合并（避免窄行回退）** |

候选 vs current best 对比表（整算子 = 各 kernel Task Duration 之和；AICore util = msprof
`aiv_vec_ratio`；memory = `aiv_mte2_ratio` / `aiv_mte2_active_bw`；L0 = `python CumprodFwdOp.py --level all`）：

| workload | candidate | Task Duration (us) | AICore util (aiv_vec_ratio) | mem: aiv_mte2_ratio | mem: mte2 active bw | L0 |
|---|---|---:|---:|---:|---:|---|
| long-seq-scan bf16 | current best = v1 回退 | 2088.68 | 0.0101（v1 profile） | 0.0034 | — | PASS |
| long-seq-scan bf16 | v3 初版（4 相位，列读） | 558.4（=203.3+10.8+5.1+339.2） | 0.0600（p3） | 0.8746（p3） | 2.9 GB/s（p3） | PASS |
| long-seq-scan bf16 | **v3_b1+b4+b5（合并后）** | **317.01** | 0.0017（p1）/0.0600（p3） | 0.0040（p1） | 34.6 GB/s（p1）/2.9->未复测（p3） | PASS |
| hidden-state fp16 | current best = v2 | 1091.20 | — | — | — | PASS |
| hidden-state fp16 | v3_b1（tile carry） | 1131.54 | — | — | — | PASS |
| hidden-state bf16 | current best = v2 | 1106.12 | — | — | — | PASS |
| hidden-state bf16 | v3_b1（tile carry） | 1013.12 | — | — | — | PASS |
| non-aligned fp16 | current best = v2 | 48.34 | — | — | — | PASS |
| non-aligned fp16 | v3_b6（窄行保留列读） | 48.14 | — | — | — | PASS |
| non-aligned bf16 | current best = v2 | 48.20 | — | — | — | PASS |
| non-aligned bf16 | v3_b6 | 50.96 | — | — | — | PASS |

"——"= 该分支未采 pipe 指标（本轮只对 long-seq 的 p1/p3 采了 PipeUtilization，用作根因定位）。

#### 合并与复测（非回退检查）

合并后 `perf_opt/CumprodFwdOp.py` 复测**全部** 5 个 tune workload + smoke 精度：

| workload | 合并后 (us) | 相对 current best | 备注 |
|---|---:|---:|---|
| hidden-state fp16 | 1131.54 | -3.7%（1091.20 -> 1131.54） | 在整算子重复测量噪声带内（同一文件重复测：1013.1 / 1131.5，跨度 12%） |
| hidden-state bf16 | 1013.12 | +8.4% | 同上（同一文件重复测：1064.4 / 1013.1） |
| long-seq bf16 | 317.01 | **+6.6x**（2088.68 -> 317.01） | 明确超出噪声 |
| non-aligned fp16 | 48.14 | +0.4% | 持平 |
| non-aligned bf16 | 50.96 | -5.7% | 同一文件重复测 48.80 / 49.26 / 50.96（跨度 4.4%），落在噪声带内 |

smoke（128x512 fp16/bf16/fp32）：PASS（L0-1a/b/c，max_diff 9.77e-04 / 3.91e-03 / 1.19e-07）。

**噪声披露（不改口径、不挑选）**：p3 的同文件重复测量在 726~843us（hidden-state）与 83.6~86.8us
（long-seq）之间波动，整算子口径的重复测量跨度约 4%~12%。因此 4 条短 case 的 v3-vs-v2 差异
（-5.7%~+8.4%）**不能**判定为真实回归或提升；唯一超出噪声的结论是 long-seq（+6.6x，远超 12% 噪声带）
与"4 条短 case 与 v2 持平"。若后续要求严格的 plan 级判定，应对同一版本做 3 次独立运行取中位数。

### Round 3b (P5, 2026-10-22) -- 相位归属修正 + 两个延迟杠杆

**先修正 Round 3 的相位归属**（读全部 kernel 行而不是猜 glob 顺序后确认，见
`profiles/p5_final/*/*/OpBasicInfo*.csv` 与 PipeUtilization）：hidden-state 的 1131.54us 里
**phase 1（partial，mix，4096 blocks）= 842.92us（74.5%）**，phase 3（final，vector）= 267.57us，
phase 2a = 21.04us。Round 3 表里把主导相位写成 p3 是错的，实测归属以本节为准（p3 的"单列跨步读"结论
本身没错，只是它不是 hidden-state 的主因：p3 319->268us）。

p1 的 profile 证据（`profiles/p5_final/hidden-state-scan/2048x4096_float16/main_mix_aic/0/`）：

| 指标 | 值 | 读法 |
|---|---|---|
| `aiv_vec_ratio` / `aiv_mte2_ratio` | 0.0004 / 0.0011 | 向量/搬运管线几乎全空 |
| `aiv_scalar_ratio`（p1 是 mix，AIV 侧） | 0.5421（Round 3 版本） | **过半 AIV 时间是标量** |
| `aiv_mte2_active_bw` | 31.4 GB/s | 不是带宽问题 |
| block 时长 | 8.2 us/block（842.92x40/4096） | 纯延迟主导 |

两个单点分支（同一 base = Round 3 定稿）：

| candidate_id | 单点改动 | 结果 | L0 | 判定 |
|---|---|---|---|---|
| v3_r3b_siglt | 符号指示量 `0.5 - 0.5*x/(|x|+eps)`（6 遍向量 op，含 1 次 **`T.vdiv`**）换成逐元素比较 `Ind = (x < 0) ? 1 : 0`（1 个 T.Parallel + `T.if_then_else`，仍是"单循环单 store"形态） | [2048,4096] fp16 的 p1 842.92 -> 716.09 us（1.18x），精度不变（0/1 仍精确，x==0 由 0.5 变 0，更干净） | PASS | **win** |
| v3_r3b_bm64 | `BM 32 -> 64`（每相位 block 数减半，暴露的 DMA/CV 握手延迟随之减半；块内 (bm,bs) UB 用量翻倍仍在预算内） | p1 716.09 -> 455.01、p3 267.57 -> 190.50、p2a 21.04 -> 12.96（[2048,4096] fp16）；[64,32768] bf16 p1 218.56 -> 129.26、p3 83.58 -> 72.72 | PASS | **win 合并（两者叠加）** |

合并后整算子（`perf_opt/CumprodFwdOp.py`，bm=64 + 比较式指示量）复测全部 5 条 tune workload：

| workload | candidate | Task Duration (us) | AICore util (aiv_vec_ratio) | mem: aiv_mte2_ratio | mem: mte2 active bw | L0 |
|---|---|---:|---:|---:|---:|---|
| hidden-state fp16 | current best = v2 | 1091.20 | — | — | — | PASS |
| hidden-state fp16 | **v3_r3b（合并后）** | **644.15** | 0.17 / 0.0004（p1 AIV / AIC 侧） | 0.177 / 0.0011 | 70.7 / 31.4 GB/s | PASS |
| hidden-state bf16 | current best = v2 | 1106.12 | — | — | — | PASS |
| hidden-state bf16 | **v3_r3b（合并后）** | **655.65** | 同上量级 | 同上 | 同上 | PASS |
| long-seq bf16 | current best = v1 回退 | 2088.68 | 0.0101（v1） | 0.0034（v1） | — | PASS |
| long-seq bf16 | **v3_r3b（合并后）** | **209.78** | 0.188（p3）/0.0015（p1） | 0.186/0.0039 | 60.2/33.8 GB/s | PASS |
| non-aligned fp16 | current best = v2 | 48.34 | — | — | — | PASS |
| non-aligned fp16 | **v3_r3b（合并后）** | **39.18** | — | — | — | PASS |
| non-aligned bf16 | current best = v2 | 48.20 | — | — | — | PASS |
| non-aligned bf16 | **v3_r3b（合并后）** | **38.40** | — | — | — | PASS |

"——"= 该版本未采 pipe 指标。

smoke（128x512 fp16/bf16/fp32）：PASS（`--level all` 的 L0-1a/b/c 与新参数下全部 L0/L1/L2/Boundary）。

## 3. Final Summary (v3, 2026-10-22) -- **superseded by §5 (v4)**

**Stage 4 定稿（2026-10-22）**：`perf_opt/CumprodFwdOp.py` —— v3 层次化 carry。单一切分方案覆盖全 shape：
`nc_seg = min(nc, 64)`、`ns = ceil(nc / nc_seg)`、`ncp3 = max(16, ns)`；`ns == 1`（N <= 4096）编译期退化为
v2 三相位，`ns > 1`（N <= 32768 及更宽）走 4 相位 carry 阶梯。`TUNED_BLOCK_M/TUNED_BLOCK_N`（=32/64）
与 `select_config` 暴露给 wrapper 切换块；v1（`perf_opt/_cumulative_kernel.py`）与 v2
（`perf_opt/_cumulative_kernel_v2.py`）文件原样保留作回退。

历史（Round 2，v1 标量版）的定稿说明保留如下，供追溯：`_cumulative_kernel.py` 用 Stage 3 C1 算法 +
自选 UB 驻留 tile（`block_n = min(N, 4096)`, `block_m = UB_BUDGET // (block_n * B_ELEM)`）覆盖 wrapper 的
GPU 派生 `block_n = 128`（TRAP-BENCH-CONFIG-CALIBRATION）。

## Final Performance Test Data (v3, superseded by §5)

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| _cumulative_kernel | hidden-state-scan-float16 | 8422.01 | 644.15 | v3_hier_bm64 |
| _cumulative_kernel | hidden-state-scan-bfloat16 | 8425.57 | 655.65 | v3_hier_bm64 |
| _cumulative_kernel | long-seq-scan-bfloat16 | 2185.68 | 209.78 | v3_hier_bm64 |
| _cumulative_kernel | non-aligned-cumprod-float16 | 200.00 | 39.18 | v3_hier_bm64 |
| _cumulative_kernel | non-aligned-cumprod-bfloat16 | 200.62 | 38.40 | v3_hier_bm64 |

`final_us` = 整算子 kernel 时延 = 同一次 `msprof op` 测量中报出的全部 kernel `Task Duration(us)` 之和
（`perf_opt/run_suite_v3.py`；`phase=final` 记录见 `perf_records.jsonl`，`artifact_path` =
`perf_opt/CumprodFwdOp.py`）。v1 baseline 为 round 0 记录（v1 只有 1 个 kernel，单 CSV 读数即整算子）。

Speedup vs the v1 baseline (整算子口径):

| workload | baseline_us | final_us | speedup |
|---|---:|---:|---:|
| hidden-state-scan fp16 | 8422.01 | 644.15 | 13.07x |
| hidden-state-scan bf16 | 8425.57 | 655.65 | 12.85x |
| long-seq-scan bf16 | 2185.68 | 209.78 | 10.42x |
| non-aligned-cumprod fp16 | 200.00 | 39.18 | 5.10x |
| non-aligned-cumprod bf16 | 200.62 | 38.40 | 5.22x |

相对 v2（上一版 current best，本轮同口径复测 1091.20 / 1106.12 / 2088.68 / 48.34 / 48.20）：
**全部 5 条都更快** —— 1.69x / 1.69x / 9.96x / 1.23x / 1.25x（最大 9.96x，最小 1.23x）。

Note on the metric: the msprof Roofline ratio (`achieved / computility` share) tracks the *amount* of
vector work, not wall time, and is not comparable across algorithm families (v2 already showed ~9 % while
being 20x faster than v1). All decisions here use `msprof op Task Duration(us)` — for this multi-kernel
operator, the sum over the kernels of one measurement.

Smoke final precision: PASS (all three dtypes, L0-1a/b/c).

Overall conclusion (v3, 2026-10-22): the operator is **no longer scalar-scan bound**. The Cube
triangular-GEMM prefix (v2) removed the `T.cumsum` scalar floor, and the v3 carry ladder removes v2's
UB-capacity ceiling, so the wide-row workload that used to fall back to v1 now runs the Cube path:
long-seq [64,32768] bf16 **2088.68 -> 317.01 us (6.6x)**, and all five tune workloads are 3.9x-8.3x
faster than the v1 baseline. All five have a merged winner; no workload is `merge_blocked`.

Remaining bottleneck (honest, quantified): on hidden-state phase 1 (`partial`) is 446 of 644 us (69 %)
and phase 3 (`final`) 184 us; phase 2a 14 us. Both are latency-bound, not bandwidth-bound: phase 1's
msprof pipe ratios are all ~0.001-0.18 with a per-block time of ~7 us against ~0.3 us of arithmetic, and
phase 3 runs at 31 GB/s effective against a 1.8 TB/s peak. Round 3b already halved both by doubling `BM`
(fewer exposed per-block latencies) and by removing the scalar `T.vdiv` from the sign indicator; the next
levers in the same direction are (a) one fp32 intermediate instead of two (phase 1 writes and phase 3
re-reads 2 x (M,N) fp32 = 4x the input traffic) and (b) `bs = 128` to further amortise the per-block
triangular-operator load. Both are layout/parameter levers, not a design-layer ceiling — so no new
`[DESIGN_LIMIT]` is raised for v3.

Historical (Round 2) conclusion, kept for traceability: with the v1 scalar design the operator was
scalar-scan bound (99 % scalar) and the 60 % Roofline target was unreachable within that design family;
best effort there was 1.05x-1.58x. Stop reason then: plateau.

---

## 4. Skill Retrospective

- **What worked**: the T-4 experiment batch runner (`perf_opt/run_suite.py`) turned the per-round loop
  into one scripted sweep; calibration probes (`perf_opt/probe_elem.py`, `probe_ks2.py`) localised the
  scalarisation to specific op shapes quickly.
- **Process friction 1 (R-type)**: msprof op refuses group/other-writable ancestors; the repo tree is
  group-writable, so profiling must run with the output under a private `/tmp` base and copy the raw
  profile back. `run_suite.py` encodes this; the skill's profile-collection note only mentions `chmod 700`
  on the leaf (the parent-directory requirement is documented in queue.md but not in the reference).
- **Process friction 2 (R-type)**: the DESIGN's Stage 3 L0 tests call `select_config` and therefore never
  exercised the wrapper's GPU-derived `block_n=128`; the harness-vs-L0 config split is the
  TRAP-BENCH-CONFIG-CALIBRATION pattern and is worth a first-class check in Phase 1 (compare the config the
  wrapper actually passes against the tuned config before trusting a baseline).
- **Value point proposals**: see the stage return (`value_point_proposals`).

Known limitation recorded here (not silently dropped): the npuir `vcumsum` scalar lowering is a
compiler capability observation (family of CG-2026-0027/0028); per the task output-scope constraint it is
reported as a proposal rather than appended to `.agents/evolution/capability-gaps.md` by this stage.

### Round 3 (P5) retrospective additions

- **What worked**: (D) the per-phase isolation probe (`perf_opt/probes/p5_phase_probe.py`, one phase per
  process) is what made the multi-kernel pipeline attributable at all — `msprof op` gives per-kernel rows
  but the old driver read only one of them. This probe + the summing driver turned "the op is 300 us"
  into "phase 3 is 72 % of it and it is MTE2-latency-bound".
- **R-type 1 (measurement integrity)**: a multi-kernel TileLang operator must NOT be measured by reading a
  single `OpBasicInfo.csv`. `perf_opt/run_suite.py` did that and under-reported every phase-split
  candidate by 2-4x (v2: one kernel 294 us reported as the operator, real whole-op 1091 us). Two of the
  three historical v2 speedup claims (22.8x / 8.9x) are therefore single-kernel artifacts; the honest
  whole-op v2 speedups are 7.7x / 4.1x. Proposal: the skill's profile-collection section should require
  "sum every kernel row of the measured process" for multi-kernel ops, and should treat a
  `Block Dim`-only row as a partial measurement.
- **R-type 2 (this toolchain)**: two shared-memory data-movement forms emit wrong/failing IR:
  (a) shared -> shared transposed elementwise copy inside `T.Parallel` fails BiShengIR with
  `memref.store op store index operand count not equal to memref rank`; (b) staging a gemm-fragment
  column into a shared `(bm, 1)` buffer and storing it transposed to global raises a runtime
  `MPU address access is invalid` (507015). Both are budgeted around (keep `(M, n)` layouts, move the
  granularity problem to a tile load) instead of being retried.
- **P-type (new pattern candidate)**: "single-column access to a `(M, K)` fp32 carry buffer costs ~1 us
  per read regardless of size" — the fix is to read the whole `(bm, tile_w)` slab with one DMA and pick
  the column with a bm-element scalar loop; putting the runtime column index into the `(bm, bs)` vector
  loop instead scalarises the whole loop (339 us -> 3034 us, 8.9x regression).
- **D-type (new constants)**: `aiv_mte2_active_bw 2.9 GB/s` / `aiv_mte2_ratio 0.875` / 13.9 us per block
  for the strided-column phase 3 on [64,32768] bf16; `28 GB/s` effective for phase 3 on [2048,4096] fp16
  (i.e. latency-bound, not bandwidth-bound); fp16 exact-integer boundary 2048 and bf16 256 bound the
  level-3 sign carry (segment negative counts ~2061 at N=32768) — the sign channel must be fp32 there.
- **Process friction 3 (R-type)**: `msprof op`'s output layout depends on the number of kernels (nested
  per-op dirs for multi-kernel, flat CSVs for a single kernel); a driver that hard-codes one layout
  silently reports `total=None` (hit on the v1 long-seq re-measurement).

---

# Round 4 (R4, 2026-10-26, CANN 9.0.0) -- 块数杠杆：chunk 宽 64 -> 128（v4 定稿）

本轮是续做轮（状态机已 DONE，只做 phase-1 优化）。**基线 `_cumulative_kernel.py` 与上一版定稿
`perf_opt/CumprodFwdOp.py`（v3，sha `712ab389…`）全程只读**；v3 原文备份在
`history_version/20261026_phase1/CumprodFwdOp.py.v3`，本轮所有实验件在 `perf_opt/` 下以
`CumprodFwdOp_r4_*.py` 命名。

口径不变：**整算子 kernel 时延 = 同一次 `msprof op` 测量里全部 kernel 的 `Task Duration(us)` 之和**
（`perf_opt/run_suite_v4.py`）。本轮新增 A/B 交错协议：m 个 impl 轮流跑、每格偶数次（reps=2），
报中位数并保留每一次的读数（`profiles/r4*_summary_runs.json`），以避免单 run 双态把漂移当收益。

## 4.1 本轮 base 与同会话基线（A/B 起点）

| workload | v3（本会话复测，µs） | 历史 v3（2026-10-22，µs） | 偏差 |
|---|---:|---:|---:|
| hidden-state fp16 | 647.00 | 644.15 | +0.4 % |
| hidden-state bf16 | 648.61 | 655.65 | -1.1 % |
| long-seq bf16 | 216.00 | 209.78 | +3.0 % |
| non-aligned fp16 | 36.91 | 39.18 | -5.8 % |
| non-aligned bf16 | 37.27 | 38.40 | -2.9 % |

历史数字与今日同 artifact 复测差 ≤3 %（任务书给出的 CANN 9.0.0 ×1.147 折算不适用于本算子：
v3 artifact 是 10-22 在 9.0.0 下产的），因此**本轮一律用同会话 A/B 数字决策**，不跨会话比。

## 4.2 现象（v3 base，hidden 2048×4096 fp16，msprof per-block 均值）

| 相位 | kernel | Op Type | Block Dim | Task Duration(µs) | aiv_time/块(µs) | aiv_vec_ratio | aiv_mte2_ratio | aiv_mte2_active_bw | aic_time/块(µs) | aic_cube_ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase 1 `scan_partial` | `main_mix_aic` | mix | 2048 | **446.64** | 3.70 | 0.0012 | 0.0025 | 50.3 GB/s | 5.55 | 0.0003 |
| phase 2a `scan_segcarry` | `main_mix_aic` | mix | 32 | 13.94 | 4.77 | 0.018 | 0.094 | 58.8 GB/s | 4.43 | 0.019 |
| phase 3 `scan_final` | `main` | vector | 2048 | **183.22** | 3.53 | 0.211 | 0.187 | 96.4 GB/s | – | – |

推导：phase 1 = 2048 块 / 24 AIC = 85.3 波 × 5.24 µs = 447 µs；phase 3 = 2048 块 / 48 AIV = 42.7 波 ×
4.29 µs = 183 µs。**每一相都是"每块一个固定延迟"型**：所有 pipe 占用都在 0.1 %–21 %，块内算术/搬运
耗时只占块停留时间的零头。结论：本 workload 上唯一有量级的杠杆是**块数**，不是流量、不是带宽。

（更正一条历史归属：`perf_opt/run_suite_v3.py` 打印的 kernel 列表按目录名排序，`main` 排在
`main_mix_aic` 前，所以旧日志里"187 / 14 / 452"的三元组不是相位顺序；按 `Block Dim`
（2048/32/2048）+ `Op Type`（vector/mix）可唯一归属。）
## 4.3 本轮候选优化点（全部从同一 base 派生，每支只改一个主点）

| id | 候选 | 单点机制 | 实验件 |
|---|---|---|---|
| ① | 符号前缀 fp32 -> **int16** | 符号通道是整数计数，int16 精确；(M,N) 符号中间量流量减半 | `CumprodFwdOp_r4_i16.py` |
| ② | 幅值通道中间量 2 个 (M,N) fp32 -> 1 个 | 压缩相位 1 写 / 相位 3 读的常驻中间量 | 归约到 ①（见 4.5） |
| ③ | `bs` 128 试档 | 摊薄 `bs²` 三角算子的重搬 + 块数减半 | `CumprodFwdOp_r4_bs128.py` / `_bs256.py` |
| ③' | （对照）`bm` 128 | 另一条"块数减半"路线 | `CumprodFwdOp_r4_bm128.py` |

上限核实（① 的可行性边界 `CONST-sign-count-dtype-bound`）：int16 上界 32767。本算子出现的符号计数上界：

| 计数 | 位置 | 上界 | int16 安全 |
|---|---|---:|---|
| 块内前缀 | `kpart` (M, NE) | bs = 64 | ✔ |
| chunk 总量 | `ktot` (M, NR) | bs = 64 | ✔ |
| 段内 chunk 前缀 | `kcw` (M, NR) | nc_seg·bs = 64·64 = 4096 | ✔ |
| 段总量 | `ksegtot` | 4096 | ✔ |
| 段间前缀 | `kcs` (M, ncp3) | N − bs = 32768 − 64 = 32704 | ✔（< 32767） |

即 N ≤ 32768（v3 门控上限）时**全部计数恒 ≤ 32704 < 32767**，int16 可精确承载；long-seq 分段后
段内计数 4096 更小。可行性探针 `probes/r4_i16_probe.py`（f32→i16 精确、i16→f32 精确、4096 精确）PASS。

## 4.4 候选 vs current best 对比表（B2，本轮决策依据）

base = v3（同会话 A/B，`profiles/r4c`）；hidden 2048×4096 fp16。

| 候选 | Task Duration(µs) | vs base | 相位 1 / 2a / 3 (µs) | AICore 利用率（vec_ratio） | 访存（mte2_active_bw） | L0 |
|---|---:|---:|---|---|---|---|
| **current best = v3 bs=64** | 647.00（reps 650.2/643.8） | – | 446.6 / 13.9 / 183.2 | phase1 0.0012，phase3 0.211 | phase1 50.3 GB/s，phase3 96.4 GB/s | PASS |
| ① 符号通道 int16 | 656.70（668.06/645.34） | +1.5 %（噪声内） | 446.7 / 14.0 / 184.6 | phase1 0.0012，phase3 0.211 | 不变 | PASS |
| ③' bm=128 | 502.53（503.2/501.9） | **−22.3 %** | 325.4 / 10.3 / 166.2 | phase1 0.0012，phase3 0.19 | 相位 3 每块 DMA 次数不变 | PASS |
| ③ bs=128 | **421.92（422.8/421.0）** | **−34.8 %** | 294.9 / 12.6 / 113.5 | phase1 0.0033，phase3 **0.331** | phase1 76.8 GB/s，phase3 96.9 GB/s | PASS |

bf16 同向：base 662.99 → ③ 427.20（−35.6 %）、③' 510.57（−23.0 %）、① 667.53（噪声内）。

## 4.5 分支结论

**① int16：零收益，回滚。** 符号通道 (M, N) fp32 → int16 让相位 1 的写与相位 3 的读共少 33.6 MB
（hidden fp16：8.39 M 元素 × 2 B × 2 次），实测 654.3 → 656.7 µs（fp16）、654.3 → 667.5 µs（bf16），
差值落在 A/B 噪声内（reps spread 1.0 %–3.5 %）。这条负结果本身是有效信息：**在每块固定延迟主导时，
减少 20 % 的片上/片外流量不动时延**，与 4.2 的 pipe 占用一致。`CONST-sign-count-dtype-bound` 因此
得到第二条实测支撑（int16 可行但无收益；fp16/bf16 不可行，上界 2048/256）。

**② 中间量合并：归约到 ①，随 ① 一起终止。** 名义上的"2 个 (M, N) fp32 → 1 个"在本算子里只能有两种
落法：(a) 幅值通道降精度 —— 不可行，`ln|x|` 的量化偏差沿 scan 轴线性累加，长行必须 fp32 操作数
（`scan_partial_long` 的存在的就是这个原因，N=32768 时 bf16 操作数实测 max_diff 0.219 > 0.05）；
(b) 符号通道降精度或整条消除 —— 即 ①。既然 ① 把符号通道压到 2 B/元素后时延完全不动，那么把两条
通道再合并成 1 个 fp32 (M, N)（等价于再省 16.8 MB，即 ① 的一半）在 4.2 的延迟模型下不可能有收益；
且唯一可行的"真合并"路线（把符号奇偶塞进幅值的符号位 / 低位）会破坏 `exp()` 的输入域，属于设计层
改法而非参数级。**结论：② 无可行的参数级实现，按 no_gain 关闭，不投实验预算。**

**③ bs 试档：128 胜出。** 128 → 321.9（−34.8 %，hidden fp16）；**256 UB 溢出**（BiShengIR：
`ub overflow, requires 2621696 bits while 1572864 bits available`，即需 320 KB / 可用 192 KB），
512 只会更差（L 算子 4×），故 **bs=128 是本工具链 UB 合法上限**。

**③' bm=128 对照：不及 bs=128。** 两条路线都把块数减半，但 bm=128 保持 bs=64：相位 3 每块的 6 个
`(bm, bs)` fp32 缓冲从 16 KB 涨到 32 KB（合计 192 KB ≈ UB 上限，编译器必须串行复用），所以相位 3 只
从 183 降到 166 µs，而 bs=128 降到 113 µs（每块 DMA 次数不变、每笔更大，且 `vec_ratio` 0.211→0.331）。

## 4.6 winner、合并与非回退检查

winner = **bs 128 / bm 64**（单一配置，非按 shape 分派），已合入 `perf_opt/CumprodFwdOp.py`（v4，
sha `dd64e5ef…`）。合并后在最终文件上对**全部 5 个 tune workload** 复测（A/B 交错，reps=2）+ 全量 L0：

| workload | v3（A，µs） | v4（B，µs） | 变化 | L0 |
|---|---:|---:|---:|---|
| hidden-state fp16 | 647.00 | **426.48** | **−34.1 %** | PASS |
| hidden-state bf16 | 648.61 | **427.39** | **−34.1 %** | PASS |
| long-seq bf16 | 216.00 | **154.90** | **−28.3 %** | PASS |
| non-aligned fp16 | 36.91 | **33.76** | −8.5 % | PASS |
| non-aligned bf16 | 37.27 | **33.73** | −9.5 % | PASS |

无任何 workload 回退（判据：non-aligned / long-seq 不得回退 >3 %，实测三者均**变快**）。
smoke（128×512 fp16/bf16/fp32）只做精度：`--level all` 的 L0-1a/b/c 全过。

## 4.7 本轮踩到的两个编译器行为（新实证，建议入 pattern-library）

1. **`if_then_else` 落到非 fp32 目的缓冲会被 BiShengIR 拒绝**：lowering 会额外发一条
   `cast(tmp_zero_buffer -> dst)`，dst 为 fp16/bf16 时即 `half_to_half_rintmode`，
   报 `error: 'hivm.hir.vcast' op currently don't support cast half_to_half_rintmode`。
   → 指示量必须落在 fp32 缓冲再 `vcast`。复现：`CumprodFwdOp_r4_bm128.py` 的中间版本
   （把 indicator 直接写进 `i_dt`）。
2. **tvmscript 的 `if/else` 分支自成作用域**：在 `if is_f32:` 分支里 `T.alloc_shared`、
   在另一分支里使用，会 `ParserError: Undefined variable: i_dt`（即使条件是 Python bool）。
   → 条件分配的缓冲必须在分支外声明，或改用 `dtype` 变量在分支外一次性分配。

## 4.8 本轮预算

实验分支 4 条（①②③③'）+ 1 条探针，共 4 条 valid 分支进入 `perf_records.jsonl`（round 5）；
msprof 采集 34 次（含 A/B 交错重复）；无 blockade、无 merge_blocked。

---

## 5. Final Summary (v4, 2026-10-26) -- current

最终候选：**`perf_opt/CumprodFwdOp.py`，candidate_id `v4_bs128`，sha256 `dd64e5ef…`**
（= v3 层次化 carry + chunk 宽 `BS = 64 -> 128`；`BM = 64`、carry 阶梯、kernel 数量与数值路径均未变）。
下列每一行的 `final_us` 都来自**同一最终候选**在本会话的 A/B 交错测量（`profiles/r4c_summary.json`，
`phase=final` 记录见 `perf_records.jsonl` round 5，2 次读数取中位数，spread ≤ 1.7 %）。

## Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| _cumulative_kernel | hidden-state-scan-float16 | 8422.01 | 426.48 | v4_bs128 |
| _cumulative_kernel | hidden-state-scan-bfloat16 | 8425.57 | 427.39 | v4_bs128 |
| _cumulative_kernel | long-seq-scan-bfloat16 | 2185.68 | 154.90 | v4_bs128 |
| _cumulative_kernel | non-aligned-cumprod-float16 | 200.00 | 33.76 | v4_bs128 |
| _cumulative_kernel | non-aligned-cumprod-bfloat16 | 200.62 | 33.73 | v4_bs128 |

`baseline_us` = round 0（v1 标量版，`_cumulative_kernel.py`）记录；`final_us` = 整算子 kernel 时延
（同一次 `msprof op` 测量中全部 kernel `Task Duration(us)` 之和）。

### vs v3 对照（同会话 A/B）

| kernel_id | workload_id | v3_us (本会话) | final_us (v4) | vs v3 | vs baseline |
|---|---|---:|---:|---:|---:|
| _cumulative_kernel | hidden-state-scan-float16 | 647.00 | 426.48 | −34.1 % | 19.75x |
| _cumulative_kernel | hidden-state-scan-bfloat16 | 648.61 | 427.39 | −34.1 % | 19.71x |
| _cumulative_kernel | long-seq-scan-bfloat16 | 216.00 | 154.90 | −28.3 % | 14.11x |
| _cumulative_kernel | non-aligned-cumprod-float16 | 36.91 | 33.76 | −8.5 % | 5.92x |
| _cumulative_kernel | non-aligned-cumprod-bfloat16 | 37.27 | 33.73 | −9.5 % | 5.95x |

smoke / skip：smoke（128×512，fp16/bf16/fp32）只做精度回归，不做性能采集，`--level all` 全过；
无 skipped workload（`workload_inventory.json` 5/5 `tuning_status=winner_merged`）。

验收贴（全部在最终文件上跑，同会话）：

- `python3 CumprodFwdOp.py --level all` → rc=0，60 个 PASS，`All check passed!`
  （`logs/r4_final/level_all.log`；其中 `Boundary` 的 NaN/±inf 三条是本模块一贯的 "WARN (record only)"，
  非失败门禁）
- `pytest tests/ops/test_cumulative.py -q` → **20 passed**（157.80 s）
- `pytest benchmarks/ops/bench_cumulative.py -q` → **8 passed**（475.67 s）
- `perf_records.jsonl`：round 5 新增 5 行 `phase=final`（artifact sha `dd64e5ef…`）+ 5 行 `phase=candidate`
  （v3 同会话基线，`CumprodFwdOp_r4_v3base.py`）；原 5 行 v3 `phase=final` 按任务要求降为
  `phase=merged` 并加 `note` 说明其为升级前 CANN 会话数字。
- 工具链：CANN 9.0.0（`/home/developer/Ascend/cann-9.0.0`）、Ascend910_9362、tilelang 0.1.2+`bbb6791d`。

**Harness benchmark 数字口径提醒（R-type，Round 3 已记录）**：`benchmarks/ops/bench_cumulative.py`
的 msprof 驱动只读一个 kernel（`op_name: main`，即相位 3）的时延，因此它报 hidden-state fp16 = 115.86 us、
long-seq = 45.52 us、non-aligned = 12.06 us——那是**单相位**读数，不是整算子。本表的 `final_us`
是同一次测量里三个 kernel 之和，两者不可混用；决策一律用整算子口径。

## 6. 剩余空间（诚实量化，供下一轮/设计层判断）

- phase 1 仍是最大项：hidden fp16 下 298.1 / 426.5 µs（70 %）。block 数已从 2048 降到 1024
  （1024/24 AIC = 42.7 波 × 6.99 µs），单块停留时间反而涨了 33 %（5.24→6.99 µs），说明每块的
  固定握手成本之外还有一部分随 tile 增大的串行段。再往下的参数级空间只剩 bm 96/128（受 UB
  192 KB 约束，bm=128 走 bs=128 会需要 ~320 KB；bm=96 需 ~200 KB，均超限），因此**同族内的块数
  杠杆已到顶**。
- 真正的下一层需要改结构：把"每块一次 cv 握手 + 两次 gemm + 一次 fixpipe"改成**一个块内串多个
  chunk**（`T.serial` 复用同一块，摊掉握手），但这会重新触碰 `TRAP-vselect-inplace-carried-state`
  （`T.serial` 内 in-place 向量 op 丢 loop-carried 状态）与 gemm-in-serial 的历史失败（probe G1-b），
  属下一轮的结构候选，不在本轮预算内。
- 本轮**不触发 `[DESIGN_LIMIT]`**：4.2/4.6 的收益来自 tiling/参数层（bs），设计层（log 域 + Cube
  三角前缀 + 层次化 carry）的假设没有被实测推翻，也没有"结构性加速 >2x 但设计层挡住"的证据；
  v3 已有的 `perf_feedback.md` 只覆盖 v1 标量族，保持原样（append-only，不因本轮新增数据而改写）。

## 7. Skill Retrospective (Round 4)

- **What worked (P-type)**: "相位归属 = `Op Type` + `Block Dim`，不是 CSV 出现顺序"这条小规则是
  本轮所有判断的前提。v3 日志把 187/14/452 当成相位顺序读，实际是 `main`(vector, bdim 2048) 排在前。
  建议 `profile-collection` 明确："多 kernel 算子按 `Op Type`+`Block Dim` 归属，不要靠目录名排序"。
- **What worked (P-type)**: 把"每块固定延迟"写成可算的式子（`waves = blocks / cores`、
  `per_block = kernel_duration / waves`）后，候选排序变成一次除法：bs=128 / bm=128 都是"块数减半"，
  实测 6.99 µs×42.7 波 vs 6.99… 直接对上了。建议把这条写进 `iteration-diagnosis.md` 的
  "latency-bound 判据"（pipe ratio 全部 < 0.05 且 `duration/blocks*cores` 稳定 ≈ 常数）。
- **R-type（测量纪律）**: A/B 交错 + 偶数次 + 中位数是本轮唯一能区分"1.5 % 噪声"和"23 % 收益"的手段
  （① 的 reps spread 达 3.5 %，单 run 会给出相反结论）。`run_suite_v4.py` 已固化为可复用 driver。
- **D-type（新常数）**: 本工具链 UB 上限 1572864 bits = 192 KB，bs=256 的 `scan_partial` 需要
  2621696 bits（320 KB）；`(bm,bs) = (64,128)` 的混合核每块 aiv_time 5.09 µs / aic_time 4.19 µs，
  phase 3 每块 4.29 µs @ bdim 1024；hidden fp16 每块有效带宽 mte2 77–97 GB/s（仍是延迟主导）。
- **BP proposal**: `BP-latency-bound-block-count`——当三相 pipe ratio 均 < 5 % 且
  `duration ≈ (blocks/cores) × const` 时，优先沿"增大 tile 以减少块数"搜索，并**先试 scan 轴**（bs）
  再试 row 轴（bm）：bs 方向的 UB 单价是 `bs²`（L 算子）+ `bm·bs`，bm 方向是 6 个 `(bm,bs)` 的
  vector 缓冲，前者在 gemm 型算子上更便宜（本轮 421.9 vs 502.5 µs）。
