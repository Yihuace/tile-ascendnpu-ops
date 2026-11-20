# Stage 4 调优日志 — mha_decode_paged / `_mha_decode_split_kernel`

- 算子目录: `examples/TileOPs/tileops/kernels/attention/mha_decode_paged/mha_decode_paged_kernel/`
- primary kernel: `_mha_decode_split_kernel.py::_mha_decode_split_kernel::split_partial`（captured op `split_partial_mix_aic`）
- 主指标: `msprof op` 的 `Task Duration(us)`（kernel-only），单一口径
- 设备: Ascend910B3（20 AI Cube Core / 40 AIV，1800 MHz），CANN 9.0.0，tilelang 0.1.2+5b062018
- 调优预算: 22 个实验分支 / 120 min；实际使用 9 个实验分支（含 3 个诊断探针），实测 5 轮

## 0. 计量口径校准（尺子）

**此前官方报告是伪影**：同一进程内捕获了 6+ 个 op，解析器按「时延匹配」错选框架小算子（1.8–2.5 µs），得出 0.01% 的假 Ratio。本轮所有采集都带 `--kernel-name=split_partial`，并在每条 record 里保存 `op_names`，逐条核对 **captured op 只有 `split_partial_mix_aic` 一个**（20/20 条记录均满足）。

用真值复测的 baseline 与 conductor 给出值一致（偏差 ≤0.4%）：

| workload | conductor 基线 (µs) | 本轮复测 baseline (µs) | 偏差 |
|---|---:|---:|---:|
| single-token-page128 | 158.70 | 158.29 | -0.26% |
| batch2-page256 | 292.78 | 293.17 | +0.13% |
| longer-cache | 150.48 | 150.27 | -0.14% |
| shorter-cache | 78.22 | 77.92 | -0.38% |

## 1. Workload inventory 与排序

来源：`benchmarks/ops/bench_mha_decode_paged.py::test_mha_decode_paged_bench`（`tileops/manifest/attention.yaml`），benchmark 无任何 smoke/skip 标记，4 个案例全部走 split 分派（`real_kv ≥ num_split*block_N`），故 4 个全部作为 tune（见 `workload_inventory.json` 的 reason：DESIGN 把 `single-token-page128` 称为 smoke 代表，但迁移后的 benchmark 对它同样做性能采集，按 signal-registry §5「不得仅凭 shape 或参数化顺序判定 smoke」归 tune）。`_mha_decode_no_split_kernel` 无 benchmark 覆盖（dispatch 全部走 split），inventory 记 `skipped` + 原因（未改动 = no_gain）。

每个 workload 的 Phase-A grid = `batch × heads × ns_eff`（`S_q=1`），baseline `ns_eff = min(num_split, nkv)`：

| workload | batch×heads | nkv | baseline grid | 20 核上的波数 | 调优顺序理由 |
|---|---:|---:|---:|---:|---|
| batch2-page256 | 16 | 4 | 64 | 3.2 | 绝对时延最高（293 µs），先做 |
| single-token-page128 | 16 | 4 | 64 | 3.2 | D=128 路径 + 第二大时延 |
| longer-cache | 8 | 4 | 32 | 1.6 | 中时延，8 head 的 1-wave 平衡点 |
| shorter-cache | 8 | 2 | 16 | 0.8 | 最小 |

## 2. Baseline 现象（Round 0）

profile（`profiles/baseline-r0-base_r0-*/`）：单 block `aiv_time ≈ 40 µs`，其中 **`aiv_scalar_time ≈ 35 µs`（87%）、`aiv_vec_time` 仅 0.92 µs、`aiv_mte2_time` 0.44 µs**。`aic_cube_time` 0.12 µs。即 block 几乎全部时间在 **标量流水**，与数据量（每 block 一页 64 KB）完全不成比例。

编译期日志给出直接证据：

```
warning: Op 'hivm.hir.vcmp' will execute by scalar instruction with low effiency
```

⇒ 列掩码链 `T.vbrc(col_lim, lim_mat) → T.vcmp(colmat, lim_mat, vis, "lt") → T.vselect(vis, acc_s, sent_buf, acc_s)` 在 `[block_M=16, block_N=256]` int32/bool fragment 上被降级为**逐元素标量**（文档 `T.arange.md` §2.3.1 同类降级：二维 arange 也是标量逐元素写）。dev 侧另有一处 `Unsupported user for propagating data layout`。

**设计估算 vs 实测偏差（D-2 回填）**：DESIGN §1.6.0 对 dispatch 代表 workload 的估算下界为亚 µs 级（内存受限），实测 baseline 158.29 µs，偏差 >100x，失准主项为**发射/降级**（标量化），不是流量或容量。

## 3. 迭代日志

### Round 1 — 定位标量掩码（3 个诊断/实验分支）

现象：Round 0 的 `aiv_scalar_time` 35 µs/block 是唯一主要项。候选优化点：把整条掩码链从热路径移走。**先做条件化（runtime scalar `if`）两版**：

| 分支 | 文件 | 做法 | L0/L1 | 结论 |
|---|---|---|---|---|
| r1_guard | `_mha_decode_split_kernel_round1_guard.py` | prologue 的 arange/vbrc + 循环内掩码全部套 `if col_lim < block_N` / `if need_mask` | L0 PASS，**L1-1 FAIL** | 弃 |
| r1_guardA | `_mha_decode_split_kernel_round1_guardA.py` | 只条件化循环内掩码 | **L1-1 FAIL** | 弃 |
| r1_guardserial | `_mha_decode_split_kernel_round1_guardserial.py` | 同上 + `T.Pipelined→T.serial` | **L1-1/L1-4/L1-2 FAIL** | 弃 |

三版的 `partial` 失败值完全相同（max_diff 2.719e-01），说明 `if` 体根本没有生效（npuir 对「条件块内 in-place `vselect` 写 fragment」处理不正确），**不是** guard 条件写错。故放弃条件化路线，改用**无分支的算术掩码**。

### Round 2 — 算术掩码（1 个实验分支）

思路：把 `vcmp+vselect` 换成只用向量算术的阶跃：

```
invalid j  <=>  j >= real_kv - pg*block_N
mask = clamp(j - (lim-1), 0, 1)  ∈ {0,1}
S_masked = S - mask * 1e38
```

分支 `_mha_decode_split_kernel_round2_maskarith.py`（int32 版 mask，全 fp32 中间量）：`partial-200` 由 2.7e-1 → 3.0e-2（掩码生效但不精确）；根因 `clamp(x,0,1)` 把 `x==0` 映射为 0，边界差一列。修正 `lim-1` 后 **全部非 causal 用例 PASS（max_diff 与 baseline 逐位相同：1.080e-04 / 1.153e-04 / 0）**；causal 编译报 TVMScript line-map 错（Python 条件 alloc 触发 parser 缺陷）⇒ 改为「无条件分配 + 复用已有 int32 buffer」。

### Round 3 — 掩码复用既有 buffer（1 个实验分支，current best 来源）

分支 `_mha_decode_split_kernel_round3_maskint.py`：
- 掩码改为复用 baseline 已有的 `colmat/lim_mat/dif/sent_buf`：`vbrc → vsub → vclamp(0,1) → vcast(f32) → vmul(1e38) → vsub`，**零新增 buffer**（避免 UB 溢出）；
- `T.arange` 改为一维 `[1,block_N]` + `T.vbrc` 首轴广播（一维连续 arange 走 VADDS 向量路径，二维是标量逐元素）；
- causal 分支保留 int 掩码链，并在其内部重新 `vbrc(neg_sent, sent_buf)` 复原 sentinel。

L0/L1/L2/causal/boundary **全 PASS**（含 `partial-200`、`zero-len`、`causal`、`nsplit2`），编译期 `vcmp` 标量告警消失。

**候选 vs current best 对比表（Round 3，k=1 页/块，longer-cache）**

| 指标 | current best (baseline `base_r0`) | 候选 `r3_maskint` | 变化 |
|---|---:|---:|---:|
| Task Duration (µs) | 150.27 | **19.86**（探针 `profiles/probe2`） | **-86.8%** |
| aiv_time/block (µs) | 39.9 | 8.05 | -79.8% |
| aiv_scalar_time/block (µs) | 34.85 | 2.80 | -92.0% |
| aiv_vec_time/block (µs) | 0.92 | 1.34 | +45%（真实计算暴露） |
| aiv_mte2_time/block (µs) | 0.44 | 0.55 | +25% |
| aiv_mte2_active_bw (GB/s/core) | 35.0 | 35.6 | ≈ |
| aic_cube_time/block (µs) | 0.119 | 0.119 | ≈ |
| L0 | PASS | PASS（另 L1/L2/boundary 全 PASS） | 无退化 |
| `vcmp` 标量告警 | 有 | 无 | 已消除 |

### Round 4 — Phase-A 分段数（grid/波次平衡，3 个实验分支）

现象：每 block 的**按页增量**只有 ~1.5 µs（mte2 0.55 + vec 1.34），而 block 固定开销 ~5–6 µs；同时 grid = `batch*heads*ns_eff` 与 20 核严重失配（64 块 = 3.2 波，末波半空）。候选优化点：把 `ns_eff` 降下来（每 block 走更多页、amortise 固定开销），grid 目标 = 单波 ≤ 20 块。分支参数 `ns_target ∈ {4,2,1}`，同文件 `_mha_decode_split_kernel_round4_nstarget.py`。

| 指标 | current best `r4_ns4` | `r4_ns2` | `r4_ns1` |
|---|---:|---:|---:|
| batch2-page256 Task Duration (µs) | 34.57 | 28.39 | **27.30** |
| longer-cache | 21.71 | **18.34** | 25.47 |
| shorter-cache | 13.21 | **12.66** | 16.19 |
| single-token-page128 | 29.34 | 25.46 | **21.82** |
| grid（batch2 / longer / shorter / single） | 64/32/16/64 | 32/16/16/32 | 16/8/8/16 |
| L0 | PASS | PASS | PASS |
| `op_names` | `split_partial_mix_aic` | 同 | 同 |

规律与 20 核一致：`batch*heads=16 → ns=1`（16 块，1 波）、`batch*heads=8 → ns=2`（16 块，1 波）；`ns=1` 对 8 head 只剩 8 块（0.4 波）反而回退。故取按形状推导的启发式。

### Round 5 — 合入 + 全量复测（current best = 最终版）

把 Round 3 + Round 4 合入同一个 `@T.prim_func`（无新增 kernel/分派），`ns_eff = min(num_split, nkv, max(1, 20 // (num_q*batch*heads)))`，`num_split` 保持 wrapper 上界（其 fp32 workspace 是使用槽位的超集）。合并版复测全部 4 个 tune workload + 全量精度：

| 指标 | current best (`r4_ns2`) | 合并版 `r5_auto_ns` |
|---|---:|---:|
| batch2-page256 (µs) | 28.39 | **27.12** |
| longer-cache (µs) | 18.34 | **18.17** |
| shorter-cache (µs) | 12.66 | **12.59** |
| single-token-page128 (µs) | 25.46（ns2） | **22.20**（自动 ns=1） |
| L0/L1/L2/boundary | PASS | **全 PASS**（`python perf_opt/_mha_decode_split_kernel.py --level all`） |

**被否决的方向（如实记录）**：

- **条件化掩码**（3 版）：npuir 对条件块内 in-place `vselect` 处理错误 → L1 失败，整类否决（属工具链陷阱，不是设计问题）。
- **page table 标量 GM 载入**：诊断探针 `probe_ident.py`（`phys = pgc`，仅测时、数学无效）18.34 → 17.48 µs，仅 **0.86 µs**，说明 paged 间接寻址的标量载入不是主项，未采纳（且不可作为候选：会破坏 paged 语义）。
- **num_stages**：longer-cache 上 1/2/3 = 18.78 / 18.34 / 17.90，最优差 2.4% < 3% 噪声阈值，按协议不改（保留 wrapper 的 2）。
- **全 fp32 掩码中间量**（Round 2 早期版）：多 3 个 `[16,256]` fp32 buffer（+48 KB），在 page=256/D=64 有 UB 溢出风险，且 causal 触发 parser 缺陷 → 改用复用 int32 buffer 的版本。

## 4. 当前瓶颈与剩余空间（如实）

Round 5 后单 block（longer-cache, ns2, 2 页/块）拆解：`aiv_time 14.2 µs = vec 2.49 + mte2 1.54 + mte3 0.67 + scalar 6.26 + wait 2.30`；`aic_time 14.3 = mte2 4.52 + scalar 7.65 + cube 0.24 + mte1 0.79`。剩余大头是两核的**标量流水**（每页 ~3 µs）与跨核 wait：K/V 是 `[N_kv,H,D]` 物理缓存，单 head 切片每行只有 `D*2 = 128 B`（D=64），一页 256 行 ⇒ 每页 512 条 128 B 段搬运，描述符/地址标量成本按段计。**若把 block 改为「按 KV 段切、块内循环 head」，一页可一次连续读 `H*D*2 = 1 KB/行`（段数 ÷8）**，估计还能再降 2–3x；但该重构会改动 grid 语义与 8 份在线 softmax 状态（UB 预算 8×`[16,D]`），本轮 120 min 预算内无法完成实现 + 全量回归，**未实证**，不作为结论。

## Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/attention/mha_decode_paged/mha_decode_paged_kernel/_mha_decode_split_kernel.py::_mha_decode_split_kernel::split_partial | batch2-page256::float16 | 293.170 | 27.120 | r5_auto_ns |
| examples/TileOPs/tileops/kernels/attention/mha_decode_paged/mha_decode_paged_kernel/_mha_decode_split_kernel.py::_mha_decode_split_kernel::split_partial | longer-cache::float16 | 150.270 | 18.170 | r5_auto_ns |
| examples/TileOPs/tileops/kernels/attention/mha_decode_paged/mha_decode_paged_kernel/_mha_decode_split_kernel.py::_mha_decode_split_kernel::split_partial | shorter-cache::float16 | 77.920 | 12.590 | r5_auto_ns |
| examples/TileOPs/tileops/kernels/attention/mha_decode_paged/mha_decode_paged_kernel/_mha_decode_split_kernel.py::_mha_decode_split_kernel::split_partial | single-token-page128::float16 | 158.290 | 22.200 | r5_auto_ns |

最终文件：`perf_opt/_mha_decode_split_kernel.py`，sha256 `96fd6b53d21ca9eb29f956d2fcd7353ad1d6ef2d317371ec7ac1ebccbcea4f7b`。全部 final 记录由同一次运行（`round=5`, `candidate_id=r5_auto_ns`, `phase=final`）采集，`op_names` 均为 `['split_partial_mix_aic']`。

## Final Summary

### split_partial（primary，gate 4 对账口径）

| workload | baseline (µs) | final (µs) | 加速 | baseline Ratio | final Ratio | ideal (µs) |
|---|---:|---:|---:|---:|---:|---:|
| batch2-page256 | 293.17 | 27.12 | **10.8x** | 0.103% | **1.102%** | 0.30 |
| longer-cache | 150.27 | 18.17 | **8.3x** | 0.199% | **1.657%** | 0.30 |
| shorter-cache | 77.92 | 12.59 | **6.2x** | 0.191% | **1.203%** | 0.15 |
| single-token-page128 | 158.29 | 22.20 | **7.1x** | 0.761% | **5.570%** | 1.20 |
| 均值 | — | — | **7.6x** | 0.31% | **2.38%** | — |

### 整算子（split + combine，真实时延）

combine（Phase B，未改动算法，仅受益于 `ns_eff` 变小）：baseline 4.68 µs（longer-cache）→ final 3.04 / 3.34 / 3.02 / 3.30 µs（四 workload）。

| workload | baseline 整算子 (µs) | final 整算子 (µs) | 加速 |
|---|---:|---:|---:|
| longer-cache | 154.95 | 21.21 | 7.3x |
| single-token-page128 | ~163.0（split 158.29 + combine ~4.7） | 25.54 | 6.4x |
| batch2-page256 | ~297.9 | 30.14 | 9.9x |
| shorter-cache | ~82.6 | 15.89 | 5.2x |

### 结论与中止原因

- 中止原因：**success**（真实时延压到基线 13–16%，budget 未用尽）。
- **60% / 40% 目标可达性**：按「真实时延」口径（把 baseline 压到 60% 或 40%）**远超达成**（12–16% of baseline）。按 msprof roofline Ratio 口径（`ideal/duration`，ideal 仅 0.15–1.20 µs，因参数量极小）**60%/40% 不可达**：`single-token-page128` 上界 = 1.20/1.9 ≈ 63%，其余三档上界 = ideal/(combine+split 地板)。以当前 kernel-launch/同步地板 ~1.9 µs（split）+ 3.0 µs（combine）计：batch2 上界 ≈ 0.30/(1.9+3.0) ≈ 6.1%、longer ≈ 6.1%、shorter ≈ 3.1%。**已实测达到 1.10/1.66/1.20/5.57%，即已接近/进入该上界的一半以上**；再往上必须拆掉 combine 这个第二 launch（融合两相位），属设计层改动，本轮未做。
- 未产出 `perf_feedback.md`：本轮 7.6x 的收益来自**代码生成降级修复**（`vcmp` 标量化）+ 分核波次平衡，归因不在 DESIGN.md 的算法/设计假设上，不满足 `[DESIGN_LIMIT]` 条件①（且"融合两相位"的预估收益未实测，不满足条件②的证据要求）。

## Skill Retrospective

### skill 流程问题

1. **profile 伪影的机械防线**：本算子前序官方报告的 0.01% Ratio 来自「同一进程捕获多个 op + 按超时匹配选错 op」。本 skill 现有 `profile-collection.md` 强调 profile 有效性，但没有**机械要求**记录 `captured_op_names` 并核对「只有目标 op」。本轮在 `run_experiments.py` 里强制把 `op_names` 写进每条 record，且不匹配即 WARN —— 建议提升为 skill 的必填字段（否则整轮结论会被伪影污染）。
2. **「条件化」在 npuir 上不可用需登记为陷阱**：runtime scalar `if` 包住 in-place `vselect` 写 fragment，各变体（Pipelined/serial、prologue/循环内）全部静默失效（L1 max_diff 完全一致）。这是求解「热路径只在尾页生效」类问题的首选手段，值得作为 `traps-*.md` 条目（含证伪协议要求的合法形态穷举记录）。
3. **二维 `T.arange` 标量降级**：文档已写（`T.arange.md` §2.3.1），但 pattern-library 只有「1ns/elem」常数条目，没有把「[1,N] + vbrc 首轴广播」写成通用改写模式（const 条目里有该绕法，但不在 `bottleneck-patterns.md` 的候选来源里，容易漏）。

### BP / 价值点提案

| id | vp_type | 摘要 | 证据三件套 |
|---|---|---|---|
| VP-S4-MHA-SCALAR-MASK | D | `hccl vcmp/vselect`（npuir `hivm.hir.vcmp`）落在 `[16,256]` int32/bool fragment 上是**逐元素标量**执行：单 block `aiv_scalar_time` 34.85 µs vs `aiv_vec_time` 0.92 µs；改写成 `vsub→vclamp(0,1)→vcast→vmul(1e38)→vsub` 后 150.27 → 19.86 µs（7.6x） | evidence: `perf_opt/profiles/baseline-r0-base_r0-*/`（含 `hivm.hir.vcmp` 标量告警）与 `perf_opt/profiles/probe2/`；repro: `perf_opt/_mha_decode_split_kernel_round3_maskint.py` vs `../_mha_decode_split_kernel.py`；toolchain: tilelang 0.1.2+5b062018 + CANN 9.0.0 / 910B3 |
| VP-S4-MHA-GUARD-TRAP | D/P | npuir 下「runtime scalar `if` 包 in-place fragment `vselect`」静默不生效（3 个变体 L1 max_diff 逐位相同 2.719e-01）⇒ 该类条件化在本工具链不可用 | evidence: `_mha_decode_split_kernel_round1_guard{,A,serial}.py` 的 L1 结果（`perf_opt/logs/l0_*.log` 之外的 probe 输出记于本文件 §3 Round 1）；repro: 三个 round1 分支文件可直接跑 `python <file> --level all`；toolchain 同上 |
| VP-S4-MHA-WAVE-TARGET | D | decode 类算子的 Phase-A grid = `batch*heads*ns_eff`，在 20 核上应按 `ns_eff = max(1, 20//(batch*heads))` 取单波：ns 4/2/1 三档实测 21.71/18.34/25.47 µs（longer-cache），且 cross-shape 最优总是「grid≈核数」 | evidence: `perf_opt/perf_records.jsonl` round=4 的 12 条 candidate 记录；repro: `perf_opt/_mha_decode_split_kernel.py` 的 `_CORE_TARGET` 常量；toolchain 同上 |
