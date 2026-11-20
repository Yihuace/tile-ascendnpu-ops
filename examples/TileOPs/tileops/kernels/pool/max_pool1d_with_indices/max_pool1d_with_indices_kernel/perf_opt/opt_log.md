# Stage 4 性能调优日志 — `_max_pool1d_with_indices_kernel`（MaxPool1dIndicesFwdOp）

- project / operator: `TileOPs` / `MaxPool1dIndicesFwdOp`（双输出：values + int64 indices，纯 Vector，Developer 模式）
- 算子目录: `examples/TileOPs/tileops/kernels/pool/max_pool1d_with_indices/max_pool1d_with_indices_kernel/`
- scenario / mode: `optimize` / `full`
- baseline（Stage 3，只读）: `../_max_pool1d_with_indices_kernel.py`
  SHA256 `c7e29d09dc04e65c8bed1b8f65b0ee89413d1425b340712c2d11e6901fbc0f48`
- 交付: `perf_opt/_max_pool1d_with_indices_kernel.py`
  SHA256 `ade772f2278cd68656da3b32b1edd07e5ee580bfb7569312175cf1643abd1e49`
- 唯一时延口径: **`msprof op` → `OpBasicInfo.csv` 的 `Task Duration(us)`**（20 launch 的 median，
  `--launch-count=20 --warm-up=5 --dump=off`，`--kernel-name=main`）；所有 `msprof op` 调用串行
  （`flock /tmp/tilelang-msprof.lock`）。不采集 NPU event / runner 端到端时间。
- 分支预算: `max_rounds=10`、`max_experiments=30`（conductor 未收紧）；实际 formal 实验分支 **6**
  （含 2 个被实测否决的分支），另 7 个 timing-only 分解探针（不计入分支预算，不进 `perf_records.jsonl`）。

## 0. Environment / toolchain stamp（首轮必记，本机实测）

| Item | Value |
|---|---|
| npu | **Ascend910B4** ×1（`torch.npu.get_device_name(0)`；`NPUUtils.get().get_aicore_num()` 实查 **20** → 纯 Vector 核数 `20×2 = 40`） |
| UB | 196 608 B / AIV（=192 KiB，`_UB_CAPACITY_BYTES`） |
| tilelang | `0.1.2+e8b5d51e3d098422c3970f4e714fb5dba1493609`（dev-root build `build/`，git `e8b5d51`） |
| torch / torch_npu | `2.9.0+cpu` / `2.9.0.post8` |
| CANN | 9.0.0（`msprof` = `/usr/local/Ascend/cann-9.0.0/bin/msprof`） |
| `TILELANG_ASCEND_MODE` | Developer |
| 空 kernel Task 地板（本机复测） | **3.12 µs**（`(1,64)`）/ **3.38 µs**（`(8,64)` plane 形态），见 `copy_floor_calibration.jsonl` |

**版本戳纪律**：本机 stamp（`e8b5d51e` + **910B4**）与两个同族 Stage 4 前作（value-only `max_pool1d`
`c3c39ba`/910B3、`adaptive_max_pool2d_with_indices` `67a0441`/910B3）**均不同** ⇒ 前作结论一律
「待重验」，本日志的关键判断全部由本机实测给出（下文每条都标了实测来源）。
`kb_stale_count = 281`（`/tmp/kb_stale_20261018.json`）；预注入条目 `PL-1.6-copy-floor` 与
`CASE-pool-maxpool1dindices-migration` 均为 `stale`，本任务只把它们当线索、未作决策依据；
`TRAP-transpose-ub-gm-roundtrip`（verified）在 §3.3 以本机数据复核。

## 1. Benchmark workload inventory（Phase 1）

入口 = `benchmarks/ops/bench_pool.py::test_max_pool1d_indices_bench`（manifest-driven，
`workloads_to_params("MaxPool1dIndicesFwdOp", include_extra=True)`，pytest id = `f"{label}-{dtype_str}"`）。
manifest = `tileops/manifest/pool.yaml::MaxPool1dIndicesFwdOp.workloads`。实际展开 **6 例**：

| # | workload_id | kind | shape | 参数 | dtype | 处理 |
|---|---|---|---|---|---|---|
| 1 | `smoke-1x8x32-k3s2p1-float16` | smoke | [1,8,32] | k3 s2 p1 d1 | fp16 | 仅精度（label 显式含 `smoke`） |
| 2 | `smoke-1x8x32-k3s2p1-bfloat16` | smoke | [1,8,32] | k3 s2 p1 d1 | bf16 | 仅精度 |
| 3 | `smoke-1x8x32-k3s2p1-float32` | smoke | [1,8,32] | k3 s2 p1 d1 | fp32 | 仅精度 |
| 4 | `sincnet-speaker-local-float16` | **tune** | [32,80,4096] | k3 s3 p0 d1 | fp16 | 第 1 优先 |
| 5 | `ecg-cnn-dilated-bfloat16` | **tune** | [16,128,2048] | k5 s2 p2 d2 ceil | bf16 | 第 2 优先 |
| 6 | `textcnn-global-float16` | **tune** | [64,256,128] | k128 s1 p0 d1 | fp16 | 第 3 优先 |

无 skipped 案例；全部分类与 `tuning_status` 见 `workload_inventory.json`。
smoke 的判定依据 = manifest label 自带 `smoke-` 前缀（不是按 shape 或参数化顺序推断）。

**调优顺序理由**（依自采 `msprof op` baseline，非迁移报告值）：

| 顺序 | workload | 自采 baseline Task(us) | 迁移报告 Task(us) | 负担来源（本机 profile） |
|---|---|---:|---:|---|
| 1 | sincnet | **3480.77** | 3192.84 | scalar 2801 + vec 734（34 万逻辑单元 × 每单元 374 条窄 op） |
| 2 | ecg | **1558.15** | 1429.55 | scalar 1168 + vec 450 |
| 3 | textcnn | **169.60** | 138.52 | mte3 79.7（写侧扇出 23x）+ vec 62.6 |

> 自采值与迁移报告值差异 4–18%（报告用单次采样；本表取 20 launch median）。本日志一律用自采值。

## 2. Performance Test Data — baseline（round 0）

| kernel_id | workload_id | candidate_id | target_kernel | captured_op | Task(us) | profile_status | raw |
|---|---|---|---|---:|---|---|---|
| `tileops/kernels/pool/max_pool1d_with_indices/max_pool1d_with_indices_kernel/_max_pool1d_with_indices_kernel.py::_max_pool1d_with_indices_kernel::main` | `sincnet-speaker-local-float16` | `_max_pool1d_with_indices_kernel` | `main` | `main` | **3480.77** | valid | `profiles/baseline/round0/max_pool1d_with_indices_kernel_cc508312/sincnet-speaker-local-float16_653b8152` |
| 同上 | `textcnn-global-float16` | 同上 | `main` | `main` | **169.60** | valid | `.../textcnn-global-float16_f5206ca3` |
| 同上 | `ecg-cnn-dilated-bfloat16` | 同上 | `main` | `main` | **1558.15** | valid | `.../ecg-cnn-dilated-bfloat16_76510648` |

采集命令模板（每条记录的实际命令见 `logs/`）：

```bash
flock /tmp/tilelang-msprof.lock msprof op --kernel-name=main \
  --output=profiles/<stage>/round<N>/<candidate>/<workload> \
  --launch-count=20 --warm-up=5 --dump=off \
  --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio \
  /workspace/venv/bin/python3 bench.py --impl <candidate>.py --case <workload_id> --iters 25
```

**Profile 有效性**：每个 `kernel_id + workload_id` 的 raw 目录下均有 20 份
`OPPROF_*/main/<i>/{OpBasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio}.csv`；
`captured_op_name = main`，可追溯到目标 TileLang kernel（非 Cast/Mul/Random 等框架小算子）；
`profile_status = valid`。

### 2.1 baseline 逐 workload 画像（µs；`GM_to_UB`/`UB_to_GM` 为每核每 launch 的 KB）

| workload | Task | BlockDim | aiv | AICore 利用率(aiv/Task) | vec | mte2 | mte3 | scalar | GM_to_UB | UB_to_GM | aiv_vec_fops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sincnet | 3480.77 | 40 | 3452.50 | 99.2% | 734.02 | 40.64 | 79.63 | **2796.42** | 1022.0 | 2166.1 | 363 985 408 |
| textcnn | 169.60 | 40 | 146.33 | 86.3% | 62.52 | 27.11 | 79.69 | 37.72 | 96.0 | 96.0 | 26 667 520 |
| ecg | 1558.15 | 40 | 1554.45 | 99.8% | 449.92 | 23.81 | 51.01 | **1168.42** | 1640.0 | 1337.2 | 195 478 016 |

**根因（本机实测，非推断）**：`scalar` 占 `aiv` 的 81%（sincnet）/ 75%（ecg）。§3.2 的分解探针把
它定位到**每单元 2 条 int32 `vcmp`**（`nan_idx >= 0` 与 `idx32 == BIG`），与 per-visit 扫描无关：
删掉这两条后 sincnet 从 3481.69 → **1094.60 µs**（−68.6%），scalar 2796 → 426 µs。

### 2.2 设计估算 vs 实测偏差（D-2 回填，`DESIGN.md §1.6.0`）

| workload | DESIGN 估算下界 | 实测 baseline | 偏差 | 设计已识别的失准项 | 本轮实测归因 |
|---|---:|---:|---:|---|---|
| sincnet | **≈44.4 µs**（流量项 55.9 MB ÷ 1.26 TB/s，发射项标「未文档化假设」） | 3480.77 | **78×** | §1.6.0 自标「窄 op 每-op 代价未逐点实测」 | 命中：窄向量 op 实测 12.3 ns/op，且 **int32 `vcmp` 16 ns/lane** 被完全漏算（§3.2） |
| textcnn | ≈3.46 µs | 169.60 | 49× | 同上 | 写侧转置往返（`UB_to_GM` = 23.4× 真实写）+ 128 条/单元 op |
| ecg | ≈23.3 µs | 1558.15 | 67× | 同上 | 同 sincnet（+ bf16 → fp32 比较域 2 次 cast/单元） |

该偏差行是 §7 `perf_feedback.md` 的量化证据之一。

## 3. 标定与地板工作（Phase 1 前置，全部为实测）

结构化数据：`copy_floor_calibration.jsonl`（14 行，**不混入** `perf_records.jsonl`）+
`op_cost_calibration.jsonl`（探针原始行）。探针脚本 `probe_ops.py` / `probe_ops2.py`（每 op 512 次
depend 链，扣除同 shape 的 `floor` 变体后取斜率）。

### 3.1 单价表（本机 910B4，f32 / 40 核）

| 形态 | lanes | Task(us) | 每-op | 结论 |
|---|---:|---:|---:|---|
| `vmax` (1,64) 窄 | 64 | 9.75 | **12.3 ns** | 窄 op 单价 ≈ 12–13 ns（与同族前作 14.4 ns 同量级，本机重验通过） |
| `vmax` (8,64) plane | 512 | 14.20 | **21.1 ns** | **8× lanes 仅 1.7× 价格** ⇒ plane 批处理是第一杠杆 |
| `vmax` (8,128) plane | 1024 | 15.20 | **23.1 ns** | 16× lanes 仅 1.9× 价格 |
| `vcmp` fp32 (1,64) | 64 | 18.72 | 30.5 ns | 已向量化（≈2.5× vmax） |
| **`vcmp` int32 (1,64)** | 64 | **535.17** | **1039 ns** | **被标量化：16 ns/lane，34× 于 fp32 `vcmp`** |
| `vbrc` 运行期标量 (1,64) | 64 | 15.28 | 23.8 ns | 1.9× vmax（不是主因） |
| 空 kernel floor | — | 3.12 / 3.38 | — | 本机 Task 地板（**非** 910B3 的 3.37/4.25） |

### 3.2 真实 kernel 的成本分解（timing-only 探针，结果不正确，仅计时；不进 perf_records）

| 探针（sincnet） | 改动 | Task(us) | vec | scalar | 边际结论 |
|---|---|---:|---:|---:|---|
| `_probe_decomp_base` | baseline 复刻 | 3481.69 | 734.02 | 2796.42 | harness 自校验通过（vs 3480.77） |
| `_probe_decomp_nopass2` | 删 pass 2 | 3088.93 | 488.53 | 2627.76 | pass 2 只值 −11%（且 scalar 几乎不变） |
| `_probe_decomp_nonantrack` | 删 pass 1 的 NaN 三元组 | 3100.01 | 500.71 | 2655.76 | NaN 三元组只值 −11% |
| **`_probe_decomp_nointcmp`** | **删每单元 2 条 int32 `vcmp`** | **1094.60** | 705.78 | **426.23** | **−68.6%：scalar 的绝对主因** |
| `_probe_decomp_min` | 删 pass2 + NaN | 2667.19 | 260.80 | 2497.95 | 两者叠加仍被 int32 `vcmp` 压住 |
| `_probe_decomp_deadcode` | 只删死代码（`flag`/`eq_mask`/`eq_f32`/`vmul`） | 3360.18 | 631.83 | 2743.14 | −3.5% |

> 由此确定 Stage 4 的主轴顺序：**① 消除每单元 int32 `vcmp`**（−69%）→ **② CH = min(C, cap) 分块**
> （逻辑单元减半、padding lane 归零）→ **③ 死代码与 per-visit op 数**。

### 3.3 写侧归因（`TRAP-transpose-ub-gm-roundtrip` 本机复核，命中）

| workload | `UB_to_GM`(KB/核) | 真实写(KB/核) | 放大 | 判定 |
|---|---:|---:|---:|---|
| sincnet（baseline / final） | 2166.1 / 2179.2 | 873.6 | 2.48× / 2.49× | store 源是 `T.transpose` 输出 ⇒ UB→GM→UB 往返（与前作 VP-1 同机制，**本机复现**） |
| ecg（baseline / final） | 1337.2 / 1773.2 | 523.8 | 2.55× / 3.39× | 同上；CH=128 加宽了转置源 |
| textcnn（final） | 96.0 | 4.1 | **23.4×** | textcnn 的 `mte3 = 79.7 µs` 即此往返（占 Task 50%），span 转置被物化经 GM |

**口径缺口（如实披露）**：任务书 §3.2 要求的「连续生产者 vs 跨步生产者」两档 copy-floor 隔离 A/B
**未执行**——原因是 §3.2 已把瓶颈定位到 int32 `vcmp` 的标量化（DMA 通道 `mte2+mte3` 在 sincnet
仅 120 µs，是 3480 µs Task 的 3.4%；即便 final 也只有 114 µs vs 539 µs Task）。写侧归因改用上表的
`UB_to_GM / 真实写` 比值给出，机制与前作 `TRAP-transpose-ub-gm-roundtrip` 一致但**未做隔离 A/B**，
故该条结论按「未做隔离对照」降级标注。

## 4. Ratio 两套口径对账（务必分开读）

### 4.1 用户带宽口径（任务书主判据）

```text
achieved_bw = manifest roofline bytes / Task Duration        # bytes = (N*C*L_in + N*C*out_W)*elem_bytes + N*C*out_W*8
Ratio_user  = achieved_bw / 1.75 TB/s
100% 时延 = bytes / 1.75 TB/s ;  70% 目标时延 = bytes / (0.7 * 1.75 TB/s)
```

| workload | manifest bytes | 100% 线 | **70% 线** | baseline Task | final Task | baseline Ratio_user | **final Ratio_user** |
|---|---:|---:|---:|---:|---:|---:|---:|
| sincnet | 55 915 520 | 31.95 µs | **45.65 µs** | 3480.77 | 538.63 | 0.92% | **5.93%** |
| textcnn | 4 358 144 | 2.49 µs | **3.56 µs** | 169.60 | 160.26 | 1.47% | **1.55%** |
| ecg | 29 339 648 | 13.68 µs | **23.95 µs** | 1558.15 | 368.48 | 1.08% | **4.55%** |

### 4.2 官方 report 口径（`tileops/benchmark/msprof_roofline.py::compute_from_csvs`）

```text
perf   = sum(aiv_vec_fops)/TaskDuration            # fops 来自 ArithmeticUtilization CSV（逐核求和）
AI     = perf_fops / total_bytes                   # total_bytes = L2 read-miss × 128 B
peak   = min(AI × 1.8 TB/s, computility)           # computility 取 Stage 5 report 的实测加权峰值
Ratio_official = perf / peak
```

三个 workload 的 `AI × 1.8` 分别为 **125 / 11.4 / 135 TOps/s**，均远大于 `computility`
（3.349 / 3.378 / 3.294 TOps/s）⇒ **三者都落在计算受限分支**，故 `Ratio_official = fops/(T × computility)`。
用本机自采 `Task` 与 `aiv_vec_fops` 直接代入：

| workload | baseline fops | baseline Task | baseline Ratio_official | final fops | final Task | **final Ratio_official（增量）** |
|---|---:|---:|---:|---:|---:|---:|
| sincnet | 363 985 408 | 3480.77 | 3.12% | 242 674 176 | 538.63 | **13.45%**（+10.3 pt） |
| textcnn | 26 667 520 | 169.60 | 4.65% | 26 659 328 | 160.26 | **4.92%**（+0.3 pt） |
| ecg | 195 478 016 | 1558.15 | 3.81% | 157 188 608 | 368.48 | **12.95%**（+9.1 pt） |

> 迁移报告（只读）给出的是官方口径基线 3.404 / 5.699 / 4.151%（其 Task 为 3192.84 / 138.52 / 1429.55 µs）；
> 本表用自采 Task 重算，故基线数值与之不同（口径同、输入不同）。
> **两套口径显著矛盾**：官方口径在 sincnet 上 +10.3 pt（×4.3），用户带宽口径只有 5.93%（×6.5）；
> 原因是官方分子是「执行到的 vector fops」，本轮的优化同时**减少了 fops**（364M → 243M，−33%），
> 而用户带宽口径只看时延。按任务书要求**以用户带宽口径为主判据、官方口径并列披露**，不做择优单报。

## 5. Iteration Log（Phase 2）

Current best 演化：`baseline` → `round1_intfix` → `round1_intfix_deadcode` → `round2_chfull`（= final）。

### Round 1

#### Diagnostic Context

- kernel_id: `…/_max_pool1d_with_indices_kernel.py::_max_pool1d_with_indices_kernel::main`
- current best: `../_max_pool1d_with_indices_kernel.py`（3480.77 / 169.60 / 1558.15）
- Profile Facts：§2.1（scalar 占 81%/75%）+ §3.2 分解（int32 `vcmp` = 主因）+ §3.1 单价表

#### Current Phenomena

- P1：`scalar` 2796 µs（sincnet），其中约 2400 µs 来自**每单元 2 条 int32 `vcmp`**（16 ns/lane）。
- P2：sincnet 的 `C=80` 在 `_CH_MAX=64` 下被切成 2 个通道块 ⇒ 第 2 块 16/64 有效 lane（**37.5% padding**），
  且逻辑单元数翻倍（5504 vs 2752）。
- P3：pass 1 的 `flag`/`eq_mask`/`eq_f32` 链条是**死代码**（终判用 `nan_idx >= 0`），每 visit 白付 1 条 `vmul`
  - 每单元 2 条宽 op + 9 B/元素 UB。
- P4：`textcnn` 的 `UB_to_GM` = 23.4× 真实写（§3.3），`mte3` 已是其 Task 的 50%。

#### Candidate Optimization Points（本轮候选）

| opt_id | 目标现象 | 优化点 | 判断依据 | 改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| R1a | P1（另一条路） | 单遍「破纪录」first-hit（10 op/visit → 7） | 源算子即严格 `>` 单遍状态机 | 把 pass 2 合入 pass 1，`vcmp(row, acc, "gt")` + `vselect` + `vmax` | Task / L0 | **否决（实测）**：−inf 索引几乎全错（5630901/5633136）。根因：本工具链上「读运行期累加器 + 同一缓冲区写」会被重排/并发执行，`vcmp` 读到更新后的 `acc` ⇒ 退化为 last-hit。**DESIGN §1.6.1 的两遍法约束被本机独立复现** |
| R1b | P3 | 删死代码（`flag`/`eq_mask`/`eq_f32`/`vmul`） | `flag` 无消费者（终判 `cfinal` 来自 `nan_idx`） | 删 1 alloc×3 + 1 条/visit `vmul` + 2 条宽 op | Task | improved（合并） |
| R1c | P1 | 用静态 `jramp` 取代 per-visit `T.vbrc(运行期标量)` | `vbrc` 23.8 ns vs `vmax` 12.3 ns | `T.vbrc(T.int32(jj), jramp[jj:jj+1,...])` | Task | **否决（实测）**：运行期偏移的 `vbrc` 被静默丢弃 ⇒ int32 读未初始化 UB ⇒ **段错误**（DESIGN §9.2 `TRAP-npuir-vbrc-region-dropped` 本机复现） |
| **R1d** | **P1 主因** | **把每单元 2 条 int32 `vcmp` 改成 `vcast(int32→f32)` + fp32 `vcmp`** | §3.1：int32 `vcmp` 16 ns/lane vs fp32 `vmax` 0.2 ns/lane；索引范围 < 2²⁴ ⇒ int→f32 精确 | 新增 `nan_f32`/`idx_f32` 两个 (bl,ch) f32 平面 | Task / L0 | **improved（采纳）** |

#### Experiment Branches（Round 1）

| branch | base | opt_id | 主要改动 | L0 | Task(us) | 状态 |
|---|---|---|---|---|---|---|
| `_round1_singlepass.py` | baseline | R1a | 单遍破纪录 | **fail**（索引 5630901/5633136 错） | 未采 | rollback |
| `_round1_jramp.py` | baseline | R1c | 静态 jramp | 运行期段错误 | 未采 | rollback |
| `_round1_deadcode.py` | baseline | R1b | 只删死代码 | pass（idx 0/5633552） | sincnet 3360.18 | improved |
| **`_round1_intfix.py`** | baseline | **R1d** | fp32 域判决 | pass（idx 0/5633552） | sincnet **1128.20** / textcnn 160.20 / ecg 669.38 | **winner** |
| `_round1_intfix_deadcode.py` | round1_intfix | R1b | 叠加死代码 | pass | sincnet **1005.71** / textcnn 162.63 / ecg 599.61 | **merged winner** |

#### 候选 vs current best 对比表（B2 结构化回流；数据源 `perf_records.jsonl` + raw profile）

| candidate | workload | L0 | Task(us) | vs best | aiv(us) | AICore 利用率 | vec | mte2 | mte3 | scalar | GM_to_UB(KB) | UB_to_GM(KB) | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | sincnet | pass | 3480.77 | 0.0% | 3452.50 | 99.2% | 734.02 | 40.64 | 79.63 | 2796.42 | 1022.0 | 2166.1 | 待优化 |
| round1_intfix | sincnet | pass | **1128.20** | **−67.6%** | 1106.60 | 98.1% | 732.86 | 40.99 | 80.14 | **432.57** | 1022.0 | 2166.1 | winner |
| round1_intfix_deadcode | sincnet | pass | **1005.71** | −10.9% | 986.98 | 98.1% | 630.51 | 40.70 | 80.30 | 407.01 | 1022.0 | 2166.1 | **merged winner** |
| baseline | textcnn | pass | 169.60 | 0.0% | 146.33 | 86.3% | 62.52 | 27.11 | 79.69 | 37.72 | 96.0 | 96.0 | 待优化 |
| round1_intfix | textcnn | pass | **160.20** | **−5.5%** | 139.67 | 87.1% | 62.58 | 26.59 | 80.83 | 31.82 | 96.0 | 96.0 | winner |
| round1_intfix_deadcode | textcnn | pass | 162.63 | +1.5% | 140.46 | 86.4% | 62.58 | 27.41 | 78.81 | 31.78 | 96.0 | 96.0 | 噪声内保持 |
| baseline | ecg | pass | 1558.15 | 0.0% | 1554.45 | 99.8% | 449.92 | 23.81 | 51.01 | 1168.42 | 1640.0 | 1337.2 | 待优化 |
| round1_intfix | ecg | pass | **669.38** | **−57.0%** | 666.91 | 99.6% | 447.14 | 25.54 | 52.92 | **266.08** | 392.0 | 1337.2 | winner |
| round1_intfix_deadcode | ecg | pass | **599.61** | −10.4% | 597.34 | 99.6% | 388.33 | 25.80 | 53.23 | 270.75 | 392.0 | 1337.2 | **merged winner** |

#### Iteration Winner（Round 1）

- winner（三 workload 同一候选）: R1d + R1b → `_round1_intfix_deadcode.py`
- 合并非回退检查：三个 tune workload 全部通过（textcnn +1.5% 落入噪声带，`vec` 完全相同 ⇒ 判为噪声）；
- smoke 精度：`smoke-1x8x32-k3s2p1-{fp16,bf16,fp32}` 三 dtype 全部 `max_diff=0.000e+00, idx_exact=True`（只记精度）。

### Round 2

#### Diagnostic Context

- current best: `_round1_intfix_deadcode.py`（1005.71 / 162.63 / 599.61）
- Profile Facts（本轮）：sincnet `vec=630.5, scalar=407.0, mte2=40.7, mte3=80.3` ⇒ 已是「每条 op 12–13 ns」
  的均匀 op-cost 态；`aiv = 987` 主导 ⇒ **只剩 op 条数**这一根杠杆。

#### Current Phenomena

- P5：sincnet 的 op 条数 = 逻辑单元数 × 每单元 op 数。`C=80` 在 `_CH_MAX=64` 下仍是 2 个通道块
  （`num_logical` 5504），第 2 块只有 16/64 有效 lane。
- P6：`_general_bytes`（UB 容量不等式）仍按 Stage 3 的 `69+2eb` 记账，含已删除的 `flag`/`eq_mask`/`eq_f32`
  ⇒ 偏保守，压低了 BL，也把 CH 上限锁在 64。

#### Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| **R2** | P5+P6 | Tiling 重推导：`_CH_MAX` 64→128 且 UB 足迹改为**逐 buffer 精确和**（含 import 期自检） | `C = 80/256/128`；§3.1 显示 plane/宽 op 单价近常数 ⇒ 越宽越省；padding lane 是纯浪费 | `_general_bytes = CH*LSPAN*(2eb+4) + BL*CH*(57+2eb) + 14*CH`，`_general_bytes_selfcheck()` | Task / L0 / plan | **improved（采纳）** |
| R3 | P5 | 每 visit op 数 9→5（NaN 位置平面 per-unit 化 + pass2 读 `arange` 行 ramp） | `arange` 构造可行（本机 IR 通过） | `T.arange(jplane,[1,0],0)` + per-unit `vcmp`+`vselect` | Task | **否决（实测）**：`hivm.hir.vsel` 拒绝「从更大 buffer 切出的操作数」（`operands' shape are inconsistent`）；去掉该子项后运行期 core dump ⇒ 整支否决（与 `TRAP-hivm-vsel-cond-slice` 同族，本机复现） |

#### Experiment Branches（Round 2）

| branch | base | opt_id | 主要改动 | plan（sincnet / textcnn / ecg） | L0 | Task(us) | 状态 |
|---|---|---|---|---|---|---|---|
| **`_round2_chfull.py`** | round1_intfix_deadcode | **R2** | CH 上限 128 + 精确 UB 足迹 | CH/BL/units = 80/16/2752、64/1/256、128/10/1648 | pass | 538.91 / 162.59 / 367.98 | **winner（= final）** |
| `_round3_opcount.py` | round2_chfull | R3 | op 数 9→5 | 同 R2 | fail（编译 + 运行） | 未采 | rollback |

#### 候选 vs current best 对比表（B2）

| candidate | workload | L0 | Task(us) | vs best | aiv(us) | AICore 利用率 | vec | mte2 | mte3 | scalar | GM_to_UB(KB) | UB_to_GM(KB) | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| round1_intfix_deadcode | sincnet | pass | 1005.71 | 0.0% | 986.98 | 98.1% | 630.51 | 40.70 | 80.30 | 407.01 | 1022.0 | 2166.1 | best |
| **round2_chfull** | sincnet | pass | **538.91** | **−46.4%** | 534.31 | 99.1% | 342.75 | 35.74 | 79.02 | 203.75 | 1030.0 | 2179.2 | **winner** |
| round1_intfix_deadcode | textcnn | pass | 162.63 | 0.0% | 140.46 | 86.4% | 62.58 | 27.41 | 78.81 | 31.78 | 96.0 | 96.0 | best |
| round2_chfull | textcnn | pass | 162.59 | −0.0% | 141.56 | 87.1% | 62.58 | 26.95 | 82.00 | 31.75 | 96.0 | 96.0 | 计划未变（`CH=64`、`bl=1`），噪声内 |
| round1_intfix_deadcode | ecg | pass | 599.61 | 0.0% | 597.34 | 99.6% | 388.33 | 25.80 | 53.23 | 270.75 | 392.0 | 1337.2 | best |
| **round2_chfull** | ecg | pass | **367.98** | **−38.6%** | 360.40 | 98.1% | 252.76 | 33.19 | 67.90 | 147.83 | 656.0 | 1773.2 | **winner** |

#### Iteration Winner（Round 2）

- `_round2_chfull.py` 成为已合并 current best：sincnet −46.4%、ecg −38.6%、textcnn 噪声内不回退。
- 合并非回退检查（本 kernel 全部 tune workload）+ smoke 精度：见表（三 workload 均 pass、smoke 见 §6）。

#### 中止判定：`plateau`

连续轮次（Round 2 → Round 3）无 valid 分支带来 >3% 提升（R3 被实测否决），且主要结构候选已实测覆盖。
**T-1 参照结构 diff 检查**（plateau 前强制）见 §6.4：找到的同族参照全部不可移植（逐条见该节），
剩余的可行结构（行批处理 / 平面归约）被**编译器能力**封死（§8.3 capability gaps 1–3），不是参数或设计可解项。

## 6. Final Summary

### Final Performance Test Data

`phase=final`，**round 7（2026-10-20，Ascend910B3 / tilelang d2783bd 档）**，同一最终候选
（`candidate_id = _max_pool1d_with_indices_kernel`，artifact `_max_pool1d_with_indices_kernel.py`，
SHA256 `1d6aaf3e7293c1114f1f5bf3141d318e3f109a5363d4fcd5ee0e93527a6b8f63`）。
`baseline_us` 仍是 round 0 的 910B4 基线记录（append-only），同设备起点见 §11.1 / §12.1。
最终交付文件 `--level all`：**L0 ALL PASS / L1 ALL PASS**、`index mismatches 0/12 219 806`（见 §12.7）。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/pool/max_pool1d_with_indices/max_pool1d_with_indices_kernel/_max_pool1d_with_indices_kernel.py::_max_pool1d_with_indices_kernel::main | sincnet-speaker-local-float16 | 3480.77 | 496.10 | _max_pool1d_with_indices_kernel |
| tileops/kernels/pool/max_pool1d_with_indices/max_pool1d_with_indices_kernel/_max_pool1d_with_indices_kernel.py::_max_pool1d_with_indices_kernel::main | textcnn-global-float16 | 169.60 | 17.32 | _max_pool1d_with_indices_kernel |
| tileops/kernels/pool/max_pool1d_with_indices/max_pool1d_with_indices_kernel/_max_pool1d_with_indices_kernel.py::_max_pool1d_with_indices_kernel::main | ecg-cnn-dilated-bfloat16 | 1558.15 | 87.50 | _max_pool1d_with_indices_kernel |

> **final 覆盖说明（skill §3.1② 通道，append-only；本轮为第二/三/四代覆盖）**：round-3 的 final 测了
> 三代，全部由**嵌入测试套件**的字节变化触发（内核代码三代完全一致）：① 首测（sha `929c18d7`，
> 仅内核改动）；② 补 `L0-17j/L0-17k` 两条 ceil_mode 用例后重采（sha `67ae08d6`）；③ 修正测试标签
> `c-not-multiple`→`c48-whole-c-fp32` 并补真正的非整除用例（`C=20/ch=24`）后重采（sha `1d6aaf3e`）。
> 三代旧行都**原样**移入 `perf_records_full_history.jsonl` 并逐条追加 `superseded_reason`（行体不改）；
> 该文件**前 3 行**是更早一代 `perf_records_superseded.jsonl` 的 3 行**逐字节**并入（旧文件保留不删，
> 理由：删掉会破坏 append-only 留痕）。`perf_records.jsonl` 其余 34 行（3 baseline / 31 candidate）保持
> 追加语义未动。覆盖范围 = 全部 3 个 tune workload；原因 = 交付文件字节变化（新增结构路径 → 补测试覆盖）。

> 最终文件 SHA256 `67ae08d628fb14a5c4911e1f227ecd2ae6783ace0cffb5d13344263e0da9c6fa`，与三条
> `phase=final` 记录的 `artifact_path` / `artifact_sha256` 一致
> （`profiles/final/round7/_max_pool1d_with_indices_kernel_*/`）。

### 6.2 逐 workload 分解与 profile 证据（final）

> **本表已更新为 round-7（round-3）最终候选的实测值**；逐轮增量见 §11.5（round-2）与 §12.5（round-3）。

| workload | baseline_us | final_us | Δ | 用户带宽 Ratio（1.75 TB/s 口径） | final：aiv / vec / mte2 / mte3 / scalar |
|---|---:|---:|---:|---:|---|
| sincnet | 3480.77 | **496.10** | **−85.7%（7.02×）** | 6.44% | 491.96 / 314.19 / 34.95 / 78.24 / 186.83 |
| textcnn | 169.60 | **17.32** | **−89.8%（9.79×）** | 14.38% | 13.89 / 9.52 / 2.50 / 1.00 / 2.58 |
| ecg | 1558.15 | **87.50** | **−94.4%（17.81×）** | 19.16% | 84.40 / 51.09 / 21.17 / 39.29 / 19.74 |

> 官方的 msprof Roofline `Ratio`（`tileops/benchmark/msprof_roofline.py::compute_from_csvs`）本轮**未采**
> （其口径与 kernel latency 不成反比，见 §4.2），只用用户带宽口径；逐轮对比见 §12.5。

归因（msprof 证据）：

- **sincnet**：int32 `vcmp` 消除（scalar 2796→433 在 R1 完成）+ 逻辑单元 5504→2752（CH=64→80，padding lane
  37.5%→0）+ 死代码移除 ⇒ `vec 734→343`、`scalar 2796→204`。残余以 `vec` 为主（343 of 534 aiv），
  即「每条 visit 9 条 12–13 ns 窄 op」的 op 条数成本。
- **ecg**：同上（CH=64→128，逻辑单元 1952→1648，`vec 450→253`）。`mte3` 上升到 67（UB_to_GM 3.39×
  真实写，转置源加宽），但仍不是关键路径。
- **textcnn**：唯一生效的改动是 fp32 域判决（`vec`/`scalar` 基本不变）。**其瓶颈是写侧往返**：
  `mte3 = 79.66 µs`（占 Task 50%），`UB_to_GM = 23.4×` 真实写；`CH` 无法升到 128（`lspan=128` 行的
  span 平面在 CH=128 时需 138.9 KB > 115.7 KB 预算），因此该 workload 的 tiling 在 R2 中未变。

### 6.3 smoke / skip 与精度

| workload_id | kind | 处理 | 精度 |
|---|---|---|---|
| `smoke-1x8x32-k3s2p1-float16` | smoke | 每次合并 + 最终只做精度回归，**无性能记录** | **pass**（`values_exact=True, max_diff=0.000e+00, idx_exact=True, idx_dtype=int64`） |
| `smoke-1x8x32-k3s2p1-bfloat16` | smoke | 同上 | **pass**（同上） |
| `smoke-1x8x32-k3s2p1-float32` | smoke | 同上 | **pass**（同上） |
| — | skipped | 无 | — |

最终交付文件 `python _max_pool1d_with_indices_kernel.py --level all`：
**L0 ALL PASS / L1 ALL PASS / L2 全 PASS / Boundary 全 PASS**，
`Summary: worst value max_diff per dtype: bfloat16=0.000e+00, float16=0.000e+00, float32=0.000e+00;
index mismatches 0/12207226`（逐位精确，atol=rtol=0，索引 `torch.equal`）。

### 6.4 参照锚定（T-1 morph ladder 轻量版，定稿前强制）

检索同族已知最优实现（`pattern-library/cases.md` 参考实现集 + `examples/TileOPs/tileops/kernels/pool/` 全部 7 个 pool 算子 + 两个同族 Stage 4 前作）：

| 参照 | 路径 | 结构形态 | 是否可移植 |
|---|---|---|---|
| 同 shape 家族 value-only Stage 4 | `examples/TileOPs/tileops/kernels/pool/max_pool1d/max_pool1d_kernel/perf_opt/`（301.94→188.45 µs） | C 轴向量 + 逐 (kw,owl) 窄 op 扫描（每 visit 2 op） | **同构、无可移植差异**：本算子每 visit 需 9 op（值+first-hit+last-NaN 三路累加器），按其 188.45 µs × (9/2) 归一化预期 ≈ 848 µs，**final 538.63 µs 已优于该归一化外推**（其 vec 113.35 µs@2 op/visit ⇒ 本机 5.2 ns/op·visit 的更优单价已被 CH=80 拿到） |
| 同族 indices + 2D（plateau 收束） | `.../adaptive_max_pool2d_with_indices/.../perf_opt/`（144.39→91.92 µs，Ratio 2.5→4.2%） | 扁平整平面读 + dtype 宽度扫描 + 「写侧连续生产者」纪律 | **不可移植到窗口类**：其收益来自 `L_out==1` 的整平面单遍归约；窗口类的每个输出对应重叠/变长窗口，无整平面归约可用。其「连续生产者」纪律**已被本实现遵守**（store 源 = `T.transpose` 输出的整宽行区间） |
| 同族整平面归约（`PL-1.23`） | `adaptive_avg_pool2d` Stage 4 | 合并空间轴成宽最内轴、免转置 | 不可移植：要求每个输出 bin 是矩形子区域；窗口类输出在合并轴上不连续 |
| 窗口类 2D 迁移 | `.../max_pool2d/max_pool2d_kernel/` | **无 `perf_opt/`**（只到 Stage 5） | 库内**不存在窗口类 Stage 4 参照实现** ⇒ 「窗口类当前工具链最优结构」无先例可比 |

**结论**：找到的同族参照**全部不可移植或被归一化外推超越**；窗口类（general 路径）在库内无参照 ⇒
按规范显式标注「**未找到可移植参照，天花板结论置信度降级**」。本日志的地板结论只对**实测过的冻结结构族**
（C 轴向量 + 逐 (kw,owl) 窄 op 扫描 + `T.transpose` 输出）有效，不作绝对地板断言。

### 6.5 性能目标达成判定（用户带宽口径，70% 线）

| workload | 70% 目标时延 | final Task | 差距 | 达标 | 机械前置判据（`perf-feedback.md §1`） |
|---|---:|---:|---:|---|---|
| sincnet | 45.65 µs | 538.63 | **11.8×** | ✗ | 必需 op 条数下界 = `num_logical × k × bl × 5`（值 1 + NaN 1 + pass2 3）= 2752×48×5 = **660 480 条窄 op**；× 实测单价 12.3 ns ÷ 40 核 = **203 µs > 45.65 µs**（不含 DMA 与 epilogue）⇒ 冻结结构族在**当前工具链**下的可达下界已是目标线的 **4.4×** |
| textcnn | 3.56 µs | 160.26 | **45.0×** | ✗ | **3.56 µs ≤ 本机空 kernel 地板 3.12–3.38 µs 同量级**；且 `k=128` 每输出元素至少 5 条窄 op ⇒ `64×256×1×5 = 81 920` 条 128-wide op × 23.1 ns ÷ 40 = 47 µs。**口径不可达**（目标线低于「地板 + 必要 pass」） |
| ecg | 23.95 µs | 368.48 | **15.4×** | ✗ | 必需 op 下界 = 1648×5×10×5 = 412 000 条窄 op × 12.3 ns ÷ 40 = **127 µs > 23.95 µs** ⇒ 4.4× 于目标线 |

**性能目标未达成**，如实报告：流程完整执行（3 个 tune workload 均有结论、有同一最终候选的 final 全量记录、
smoke 精度通过），但**流程完成 ≠ 性能目标达成**。已取得 sincnet −84.5% / ecg −76.4% / textcnn −5.5%，
全部逐位精确（`max_diff=0.000e+00`、索引 0 错）。

### 6.6 中止原因

- `success` 未达成（三 workload 用户带宽 Ratio 5.93% / 1.55% / 4.55% ≪ 70%）。
- `budget_exhausted` 未触发（3 轮 ≤ `max_rounds=10`；6 个 formal 分支 ≤ `max_experiments=30`）。
- `blocked` 未触发（剩余候选是「被实测否决」而非不可实现）。
- **实际中止 = `plateau`**：Round 2 取得 −46.4%/−38.6% 后，Round 3 的唯一候选（R3，9→5 op/visit）被实测否决；
  且 §6.4 的参照结构 diff 检查未发现可移植差异。因 §7 的 `[DESIGN_LIMIT]` 双条件同时满足，另产出
  `perf_feedback.md` 并附 `[DESIGN_LIMIT]` 信号。

### 6.7 已知残留（如实披露）

1. **写侧转置往返未消除**（§3.3）：`UB_to_GM` 仍是真实写的 2.5×（sincnet）/ 3.4×（ecg）/ 23.4×（textcnn）。
   它对正确性零影响，但压住 textcnn 的上限；三条替代路线（平面归约 / 免转置列扫描 / 3D 相位抽取）均实测 blocked。
2. **`round-0` baseline 三行的派生字段 `achieved_bw_tb_s` / `bw_ratio_pct` 是错的**（首版 runner 单位反了，
   把 TB/s 算成 0.0）。`perf_records.jsonl` 是 append-only，故**保留原行不改写**；本日志 §4.1 的
   Ratio 全部由 `duration_us` + manifest bytes 重算，与那些派生字段无关。
3. **`copy_floor_calibration.jsonl` 的隔离 A/B 未做**（§3.3 口径缺口已披露）；写侧归因靠 `UB_to_GM/真实写` 比值。
4. 未被采纳的探针/分支文件（`_probe_decomp_*.py`、`_round1_singlepass.py`、`_round1_jramp.py`、
   `_round3_opcount.py`）保留在 `perf_opt/` 顶层用于溯源，不属于交付候选。
5. `probe_ops.py` 的 `vselect` 首选测量因循环体不变量被常量折叠（得 ~0 ns），已在 `probe_ops2.py` 用
   数据相关体型重测；`probe_ops2.py` 的 `vsel_int` 行与 `vcmp_int` 同值（体内含同一条 int32 `vcmp`）。

## 7. 设计层天花板（`[DESIGN_LIMIT]`，结论：**产出 `perf_feedback.md`**）

见 `perf_feedback.md`（固定 schema）。摘要：

| 条件 | 判定 | 依据 |
|---|---|---|
| ① 天花板由算法/设计层决定（非 tiling/参数可解），归因到 DESIGN 具体假设 | **成立** | DESIGN §1.6.0 选定算法 #0' 与 §3.3/§6.2 冻结的伪代码 = 「C 轴向量 + 逐 (kw,owl) 窄向量 op 扫描」。该结构的 op 条数 = `N·C·L_out·k/CH`（与 tiling 无关，CH 已取到 C），实测单价 12.3 ns/op ⇒ 下界 203 µs（sincnet）/ 127 µs（ecg），已是目标线的 4.4×。tiling 三参数（CH/BL/核数）在 3 轮内已扫到该结构下的最优区，无剩余参数解 |
| ② 结构性加速估计 > 2x **或** 实测与设计假设直接矛盾 | **成立（两支都有）** | (a) 与 DESIGN §1.6.0 的估算下界 ≈44.4 µs 直接矛盾：实测 baseline 3480.77 µs（78×）、最优 538.63 µs（12.1×）；(b) 若「行批处理」可用（把每 visit 的窄 op 合成 bl 行 plane op），按本机实测 plane 单价（512 lanes 21.1 ns）估计 sincnet 可达 ≈70–90 µs，即 **6–7×** 于当前值 ⇒ >2× |

**但天花板路径被编译器能力封死**（§8.3）：`hivm.hir.vtranspose` 只支持 2 轴、`deinterleave(channel_nums=3)`
使 `bishengir-compile` 段错误、无 strided GM copy 原语 ⇒ sincnet 的 3 相位抽取不可实现；这是
**「设计结构 × 工具链」联合**造成的天花板，故 `perf_feedback.md` 的「建议路由」写为
**附录补记 + 能力缺口登记**（不是「设计修订」：改写设计也拿不到行批处理，除非工具链补齐相位抽取）。

## 8. Skill Retrospective

### 8.1 skill 流程问题（Flow Issues）

| # | 问题 | 现象 / 证据 | 建议 |
|---|---|---|---|
| R1 | **Phase 1 的 copy-floor 纪律没有把「算子内部单元 op 单价」列为必测项** | 本任务 3480 µs 中 2400 µs 来自 2 条 per-unit int32 `vcmp`；若首轮就测「各 op 单价表」，可在 10 分钟内定位，而实际是先跑完 baseline + 6 个探针才定位 | `profile-collection.md` 增「op 单价标定」为 Phase 1 必做项（本任务的 `probe_ops.py`/`probe_ops2.py` 可作模板）：至少覆盖 (a) 窄 vs plane 宽度，(b) fp32 vs int32 同形 op，(c) 运行期标量广播 |
| R2 | **`vcmp` 的 dtype 维度缺省**：`TRAP-hivm-vsel-cond-slice` 类条目只记录了「形态」限制，未记录「int32 被标量化 34×」这一 dtype 维度 | §3.1：int32 `vcmp` 1039 ns/op vs fp32 30.5 ns/op | `traps-compiler.md` 增 `TRAP-int32-vcmp-scalarized`（带 repro）；`iteration-diagnosis.md` 的「诊断信号」增加「scalar 占比 > 50% 时先查 int 域向量 op」 |
| R3 | **证伪协议未覆盖「同一 API 的不同 dtype 形态」** | 本任务两次误判风险：`vcmp`（fp32 可用/int32 灾难）、`vselect`（整 buffer 可用/切片不可用） | 证伪协议第 1 条补「dtype 子形态 + 操作数形态子集」两维（与前作 BP_003 同向，可合并） |
| R4 | **单遍「破纪录」类结构改写缺少机器化守卫的复用入口** | R1a 让 563 万索引几乎全错；DESIGN §1.6.1 早有 `verify_equiv.py` 的 N2N3-ord 守卫但未被 Stage 4 复用 | 建议 skill 在 Phase 2 提供「结构性候选进入 L0 前的语义守卫」清单（引用算子自带的 `verify_equiv.py` 组号） |

### 8.2 value point proposals（vp_type + 证据三件套）

| id | vp_type | 结论 | evidence（实测） | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-D1 | **D** | **int32 `vcmp` 在 npuir 上被逐 lane 标量化：64 lane 1039 ns/op（16 ns/lane），是 fp32 `vcmp`（30.5 ns）的 34×、`vmax`（12.3 ns）的 85×**。双输出池化 kernel 里「`nan_idx >= 0` / `idx == BIG`」这类 int 判决必须改成 `vcast(int→f32)` + fp32 `vcmp`（索引 < 2²⁴ 时精确）。实测收益：sincnet 3480.77→1128.20 µs（−67.6%），ecg 1558.15→669.38（−57.0%），逐位精确 | `perf_records.jsonl` round1 intfix；`_probe_decomp_nointcmp`（1094.60 vs 3481.69）；`op_cost_calibration.jsonl` `vcmp_int32_m1_ch64` | `probe_ops2.py --op vcmp_int` + `_probe_decomp_nointcmp.py` | tilelang 0.1.2+e8b5d51e + CANN 9.0.0 + **Ascend910B4** / 2026-10-18 |
| VP-D2 | **D** | **窄 op 与 plane op 的单价曲线**：f32 `vmax` 在 (1,64)=12.3 ns、(8,64)=21.1 ns、(8,128)=23.1 ns ⇒ **8× lanes 只贵 1.7×、16× lanes 只贵 1.9×**；这是窗口类算子「按行批处理」的量化依据（也解释了为何 CH 从 64 拉到 80/128 近乎免费） | `op_cost_calibration.jsonl` vmax 四档 | `probe_ops.py --op vmax --m 8 --ch 128 --reps 512` | 同上 |
| VP-D3 | **D** | **单遍「破纪录」first-hit 在本工具链不可实现**：`vcmp(row, 运行中 acc)` 与随后的 `vmax(acc, row)` 之间没有可靠的数据冒险保护，实测 5 633 136 个索引里 5 630 901 个错（退化为 last-hit）。⇒ DESIGN §1.6.1 的「两遍法」是工具链硬约束，不是设计偏好 | `_round1_singlepass.py` L0 输出（`logs/`）；同族前作 `verify_equiv.py` N2N3-ord 组 | `_round1_singlepass.py`（自包含，`--level L0` 即复现） | 同上 |
| VP-P1 | **P** | **纯 Vector 双输出算子的第一杠杆是「消除标量化 op」而非「减少字节」**：本算子 IO 只占 Task 的 3.4%（mte2+mte3 120 µs / 3480 µs），而两条 int32 `vcmp` 占 69%。纪律：拿到 scalar 占比 > 50% 的 profile，先按「每条 op × 实测单价」分解，再谈 tiling | §3.2 分解表 | 同上 | 同上 |
| VP-P2 | **P** | **UB 足迹改为「逐 buffer 精确和 + import 期自检」后，CH 上限可从 64 提到 128**（C=80/128 时逻辑单元减半、padding lane 归零、单位点 line 带宽翻倍）：sincnet 1005.71→538.91（−46.4%）、ecg 599.61→367.98（−38.6%）。纪律：任何删除 alloc 的改动都要同步重推容量不等式，否则 BL/CH 被历史公式锁死 | `perf_records.jsonl` round2 chfull；`_general_bytes_selfcheck()` | `_max_pool1d_with_indices_kernel.py::_general_bytes` + `_general_bytes_selfcheck` | 同上 |
| VP-R1 | **R** | skill 的 Phase 1 应把「op 单价标定表」列为必做（见 8.1 R1） | §3.1/§3.2 | — | 同上 |
| VP-C1 | **C** | 案例索引续档：`CASE-pool-maxpool1dindices-migration` 的 Stage 4 段（int32 `vcmp` 主因、两遍法硬约束、CH=min(C,cap) 收益、写侧 23× 扇出、窗口类无参照、`[DESIGN_LIMIT]`） | 本文件全文 + `perf_records.jsonl`（15 行） | — | 同上 |

### 8.3 capability gaps（登记 `.agents/evolution/capability-gaps.md`）

| # | 缺失能力 | 编译器层 | 证据 |
|---|---|---|---|
| 1 | `hivm.hir.vtranspose` 只支持 **2 轴**置换（`permutation=[2,0,1]` → `error: Vtranspose only support two axes transpose`）⇒ 3D 相位抽取不可用 | BishengIR | `probe_phase3.py`（本机复现，`logs/`） |
| 2 | `T.deinterleave(channel_nums=3)` 使 `bishengir-compile` **段错误**（rc=-6，NPU IR 已生成）⇒ 3 相位分解不可用；`channel_nums=2` 可用 | BishengIR | `probe_deint.py`（本机复现；同族前作同结论，跨 stamp 一致） |
| 3 | `T.vbrc` 到「运行期偏移的 UB 区域」被静默丢弃 ⇒ int32 读未初始化 UB ⇒ **段错误**（`TRAP-npuir-vbrc-region-dropped` 本机复现） | Frontend / TileLangIR | `_round1_jramp.py`（L0 首例段错误） |
| 4 | `hivm.hir.vsel` 的操作数（含 cond）**不接受从更大 buffer 切出的区域**（`operands' shape are inconsistent`）；`vcmp`/`vmax` 接受同一切片 | BishengIR | `_round3_opcount.py` 编译错误（`logs/l0_round3_opcount.log`）；同族前作 `TRAP-hivm-vsel-cond-slice` 复现 |
| 5 | **int32 `vcmp` 被逐 lane 标量化**（16 ns/lane，无向量化兜底、无编译期告警提示量级） | BishengIR / vectorize pass | `op_cost_calibration.jsonl` `vcmp_int32_m1_ch64` |

## 9. Autotune Log

未使用 TileLang autotune：候选全部是**结构性**变更（int32 判决域 / UB 足迹与 CH 上限 / op 条数），
参数面只有 `(_CH_MAX, _BL_MAX)` = (128, 32) 与派生出的 (CH, BL)。
等价手动粗搜覆盖：CH ∈ {64, 80, 128}（由 `_CH_MAX` 与 C 派生）、BL 由容量不等式解出
（sincnet 16 / ecg 10 / textcnn 1）；R2 的分档结果（CH=64 时 sincnet 1005.71 → CH=80 时 538.91）
即该参数面的 A/B。未做邻域精搜的原因：CH 已等于 C（物理上限），BL 由 UB 不等式唯一确定
（BL+1 即溢出），参数面**已闭合**。

## 10. 交付清单（本目录）

| 文件 | 说明 |
|---|---|
| `_max_pool1d_with_indices_kernel.py` | **最终产物**（最优版本；kernel + 内嵌 golden + L0/L1/L2/Boundary + `main --level`，`--level all` 通过） |
| `__init__.py` | 包标记（docstring-only），供 wrapper 的 perf_opt 分支 import |
| `workload_inventory.json` | 6 例清单（3 smoke + 3 tune，含 `tuning_status`/`baseline_us`/`final_us`/`precision_pass`） |
| `perf_records.jsonl` | 结构化性能记录（append-only，**15 行**：3 baseline + 3 intfix + 3 intfix_deadcode + 3 chfull + 3 final；仅 kind=tune） |
| `copy_floor_calibration.jsonl` | 地板/单价/写侧扇出数据（14 行，**不混入** perf_records） |
| `op_cost_calibration.jsonl` | op 单价探针原始行 |
| `opt_log.md` | 本文件 |
| `perf_feedback.md` | `[DESIGN_LIMIT]` 设计层天花板反馈（固定 schema） |
| `bench.py` / `run_experiments.py` | 单 workload 采集 / 批量实验 runner（T-4） |
| `make_branch.py` / `make_probe_decomp.py` | 分支与分解探针的生成脚本（可复现每条 diff） |
| `probe_ops.py` / `probe_ops2.py` / `run_op_cost.py` / `run_op_cost2.py` | op 单价标定 |
| `probe_deint.py` / `probe_phase3.py` | 相位抽取能力探针 |
| `_round1_*.py` / `_round2_chfull.py` / `_round3_opcount.py` / `_probe_decomp_*.py` | 实验分支与 timing-only 分解探针 |
| `profiles/` | raw `msprof op` 数据（`baseline/`、`candidate/`、`final/`、`probe/`、`calibration/`） |
| `logs/` | 各阶段 stdout/stderr（L0 日志、msprof 命令与输出、探针日志） |

---

# 11. Round-2 continuation（2026-10-20，Ascend910B3 / tilelang d2783bd）

> 本节是本轮（第二轮有界调优）的全部证据。**器件与工具链已变**：910B4 → **910B3**，
> tilelang `e8b5d51` → **`d2783bd`**，`bishengir-compile 1.1.0 (428ab8fdab46)`。上轮台账的绝对时延
> 一律不复用；本节所有比较都在同一台 910B3 上重采（`msprof op`，20 launch median，
> `--launch-count=20 --warm-up=5 --dump=off --kernel-name=main`，`flock /tmp/tilelang-msprof.lock` 串行）。

## 11.0 版本戳（本轮实测）

| Item | 本轮 value |
|---|---|
| npu | **Ascend910B3** ×1（`torch.npu.get_device_name(0)`；`NPUUtils.get().get_aicore_num()` = **20** → 纯 Vector 核 40） |
| UB | 196 608 B / AIV |
| tilelang | `0.1.2+d2783bd06df70cef6d666610715b552e92c2037e`（git `d2783bd`，dev-root `build/`） |
| CANN / bishengir | 9.0.0 / **1.1.0 `428ab8fdab46`**（llvm 19.1.7） |
| torch / torch_npu | `2.9.0+cpu` / `2.9.0.post8` |
| `TILELANG_ASCEND_MODE` | Developer |
| 空 kernel Task 地板（本轮同族实测） | **2.86–3.44 µs**（`probe_store_r2.py` 的 store-only 形态 3.43–3.44；同族 recheck 2.86） |
| 预注入条目 | 全部 `stale`（`kb_stale_count = 374`）；本轮只当线索，凡作决策依据者均已本机重验（见 11.2/11.4 的「重验」列） |

## 11.1 同设备起点（本轮 baseline，round 0 的 910B4 记录仅作历史）

`perf_opt/_max_pool1d_with_indices_kernel.py`（上轮最优，sha `ade772f2…`）在本机重采：

| workload | 本机起点 Task(us) | profile | 910B4 台账 final（不可直接比） |
|---|---:|---|---:|
| sincnet-speaker-local-float16 | **496.36** | `profiles/r2_20261020/nonregression/sincnet_start` | 538.63 |
| textcnn-global-float16 | **133.54** | `profiles/candidate/round4/…`（`perf_records.jsonl` round 4） | 160.26 |
| ecg-cnn-dilated-bfloat16 | **339.71** | 同上 | 368.48 |

起点画像（µs）：textcnn `aiv 115.4 / vec 57.4 / mte2 11.9 / mte3 60.7 / scalar 29.2`，`UB_to_GM 96 KB`
（真实写 4.1 KB，**23.4×**）；ecg `aiv 331.6 / vec 231.7 / mte2 32.4 / mte3 65.6 / scalar 135.5`，
`UB_to_GM 1773 KB`（真实写 524 KB，3.4×）。

**本轮 scope（用户显式决定，不缩减 benchmark 集合）**：ecg = 主目标、textcnn = 次目标、
sincnet = 不调优（3 相位抽取 = 硬件缺口，2026-10-20 双重实测 BLOCKED，见
`recheck_20261020/FEASIBILITY_REPORT.md`），但对 sincnet 采一次**非回归**数据（仅报告，不迭代）。

> **inventory 分类说明（如实披露）**：sincnet 在 `workload_inventory.json` 里保留
> `kind=tune` + `tuning_status=no_gain`（`reason` 中逐字写明「本轮 scope = NOT tuned / skipped」）。
> 原因：gate 4 的机械分类由 label/marks 推导（无 `smoke` token ⇒ `tune`），且**任何非 tune 项不得有性能记录**；
> 而 sincnet 的 round 0–3 记录是 append-only 且必须保留，改成 `skipped` 会让这些既有行变成违规。
> 因此用 `no_gain` + 原因记录「本轮未调优、仅非回归」，语义与用户决定一致。

## 11.2 Current Phenomena（本轮起点 profile + 可行性复验）

- **P1（textcnn）**：`mte3 = 60.7 µs`（Task 45%），`UB_to_GM = 96 KB` vs 真实写 4.1 KB = **23.4×**。
  本轮隔离 A/B（`probe_store_r2.py`，三变体只差写入形态）确认放大来自**1 元素行的 GM 写粒度**，
  与 `T.transpose` 无关（重验了 `recheck_20261020` 的归因修正）：
  | 变体（同字节、同 channel blocking） | Task(us) | mte3(us) | UB_to_GM(KB) |
  |---|---:|---:|---:|
  | A：`transpose -> (CH,1)` 再写 `(real_c,1)` region（起点形态） | **85.60** | 66.90 | **96.0** |
  | B：`(1,CH)` buffer 写 1-D region `ind[row, c0:c0+rc, 0]` | **3.44** | 0.88 | **3.8** |
  | C：显式 `(CH,)` 1-D buffer 写同一 1-D region | 3.43 | 0.90 | 3.8 |
  ⇒ 写侧按 **1-D（扁平）region** 写即可拿回 25×。
- **P2（ecg）**：`vec 231.7 + scalar 135.5` 主导；op 条数 = 逻辑单元 1648 × 每单元约 450 条窄 op
  （`k*bl` 次 visit × 9 op）。`mte2+mte3` 只占 98 µs。
  **本轮重验的关键代数事实**（`s=2, p=2, d=2` ⇒ 窗口 taps 相对窗口起点偏移 {0,2,4,6,8}）：
  所有 tap 的 `iw = 2*ow - 2 + 2t` **都是偶数**，即整窗落在输入的偶相位；在相位平面
  `ph[q]=x[2q]` 上退化为 **stride 1、k=5 的普通滑窗**。⇒ 每个 tap 只需 **1 条宽 op**（而非 bl 条窄 op）。
- **P3（工具链约束，本轮新实测）**：
  1. `T.deinterleave(channel_nums=2)` 可用且逐位精确；`=3`/`=4` 仍 SIGABRT（跨 stamp 复现）。
  2. **`T.vselect` 不接受任何切片操作数**（`hivm.hir.vsel: operands' shape are inconsistent`），
     而 `vcmp/vmax/vmin/vadd` 接受同形切片 —— `probe_vsel.py` 逐形态实测（v1 整 buffer ✓ / v2 切 src0 ✗ /
     v3 切 cond ✗ / v4 vadd 切片 ✓ / v6 vcmp 切片 ✓）。这是上轮 `TRAP-hivm-vsel-cond-slice` 的精确定形。
  3. `T.vcast` **同 dtype**（f32→f32、bf16→bf16）会错编译 —— 本轮在 phase path 的 fp32 用例上踩到
     （`L0-16c` 全错，704/704 元素），改成 `T.copy` 后复现通过。
  4. **ch = 32 的相位路径索引错**（尾 3 列 304/2095104 元素，`idx = 2*q0 + 0`）；ch = 8/16/24/40/48/56/64/80
     同数据同 plan 逐位精确。根因未定位（`pramp`/`pf` 的武装 dump 正确、empty-window 分支未触发），
     处置 = 从候选中**排除**（`_PHASE_CH_EXCLUDED`）。

## 11.3 Candidate Optimization Points（本轮候选，均来自 11.2 的实测现象）

| opt_id | 目标现象 | 优化点 | 判断依据（实测/文档） | 改法 | 状态 |
|---|---|---|---|---|---|
| **R4** | P1 | `L_out == 1` 用**扁平 1-D store**（值/索引都走 `ind[row, c0:c0+rc, 0]`） | `probe_store_r2.py` A/B/C：85.60 → 3.44 µs，`UB_to_GM` 96 → 3.8 KB | 新增 `flat_store = out_l == 1` 分支（trace 期常量，直线代码） | **采纳** |
| **R5** | P2 | **2 相 deinterleave 行批处理**：span → `T.deinterleave(cn=2)` → 2D transpose → `(pp, CH)` 相位块，每 tap 1 条宽 op | 代数等价（11.2 P2）+ `probe_phase_ecg.py` 逐位精确；同族 `max_pool2d` phase path 形态 | 新增 `_phase_plan` + `_func_phase` | **采纳（主攻）** |
| R6 | P3.2 | 相位块上的首命中索引改用「整 buffer mask + 切片 vadd」 | `probe_vsel.py` v2/v3 封死 vsel 切片 | `vselect(eq, zero, BIG, bigm)` + `vadd(ramp_slice, bigm)` + `vmin` | 采纳（实现细节） |
| R7 | P3.2 | 用 `T.copy` 把 ramp 切片先落到整 buffer 再用 vsel | 同为规避 vsel 切片 | 5 × 每 unit 的连续 UB→UB copy | 试过并回退（慢 0%，且不修 ch=32；见 11.6） |
| R8 | P3.3 | fp32 用例的 phase path 用 `T.copy` 代替同 dtype `vcast` | `L0-16c` 全错 → 修正后逐位精确 | `is_f32` 分支 | 采纳（正确性修复） |
| R9 | P2 | CH 上限/候选改为「C 的精确因子」 | (ch96,bl32)=185.58 / (ch48,bl64)=113.24 / (ch64,bl40)=91.18 / (ch64,bl32)=110.41 | `_phase_ch_candidates` 只留精确因子 + 单块 C | 采纳 |
| R10 | P2 | 相位路径的 UB 足迹代理与预算标定 | 258 KB naive sum 可编译；319 KB overflow（196,352 B 实测） | `_phase_bytes` + `_PHASE_UB_SUM_BUDGET_BYTES = 300_000` + build 失败回退 | 采纳 |
| R11 | — | sincnet 3 相位（s=3）行批处理 | `recheck_20261020`：3-cycle 转置 +152 µs、`cn>2` SIGABRT | — | **不做**（用户 scope + 双重实测封死） |
| R12 | P2 | ecg bl=48（少 15% 逻辑单元） | 后端差 2 048 bit 就 overflow（1 570 816 / 1 572 864） | 需再省 ~一个 buffer | 记录为 residual（未采纳） |

## 11.4 Experiment Branches（本轮）

| branch | parent | opt_id | 主要改动 | L0 | textcnn(us) | ecg(us) | 结论 |
|---|---|---|---|---:|---:|---:|---|
| `_…_r2_flatstore.py` | `_max_pool1d_with_indices_kernel.py` | R4 | 1-D flat store（`out_l == 1`） | pass | **107.46** | 339.12 | **winner（textcnn −19.5%）** |
| `_…_r2_phase.py` | `_…_r2_flatstore.py` | R5+R6+R8+R9+R10 | 相位行批处理 + 1-D store | pass | **107.51** | **≈87.7（交付文件）** | **winner（ecg −74.2%）** |
| `_…_r2_phase.py`（中间态） | 同上 | R7 | ramp 走 `T.copy` staging | pass | — | 100.20 | 回退（无收益，且不改 ch32） |
| `_…_r2_phase.py`（中间态） | 同上 | R9 前 | 「最大 lanes」选点（选中 ch48/bl64） | pass | — | 113.24 | 回退（改为精确因子规则） |

交付文件（= `_r2_phase` 定稿）`--level all`：**L0 / L1 / L2 / Boundary 全 PASS**，
`worst value max_diff = 0.000e+00`（bf16/fp16/fp32）、`index mismatches 0/12219650`。
本轮新增的相位路径覆盖用例（先前只有 ecg 一个 shape 命中该路径）：
`L0-16a…L0-16f`（k3d2 / k5d2-ceil / k3d4-fp32 / k1 / C=20 部分通道块 / k5d2-ceil-fp16）、
`L0-16g…L0-16i`（odd-l / odd-p / odd-d 必须回退 general）、`L0-16j`（相位路径的 NaN/last-NaN/+inf-tie/-inf 语义）。

## 11.5 候选 vs current best 对比表（B2 结构化回流；数据源 `perf_records.jsonl` + raw profile）

| candidate | workload | L0 | Task(us) | vs best | aiv(us) | vec | mte2 | mte3 | scalar | GM_to_UB(KB) | UB_to_GM(KB) | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 起点（round-1 final @910B3） | textcnn | pass | 133.54 | 0.0% | 115.45 | 57.36 | 11.86 | 60.69 | 29.24 | 96.0 | 96.0 | best |
| **r2_flatstore** | textcnn | pass | **107.46** | **−19.5%** | 91.10 | 55.55 | 2.55 | **0.97** | 28.97 | 96.0 | **3.8** | **winner** |
| 起点（round-1 final @910B3） | ecg | pass | 339.71 | 0.0% | 331.60 | 231.69 | 32.44 | 65.59 | 135.49 | 656.0 | 1773.2 | best |
| r2_flatstore | ecg | pass | 339.12 | −0.2% | 331.62 | 231.69 | 32.04 | 65.16 | 135.57 | 656.0 | 1773.2 | 计划未变（噪声内） |
| **r2_phase（交付）** | ecg | pass | **87.66** | **−74.2%** | 84.08 | 51.09 | 20.90 | 39.74 | **19.81** | 336.0 | 990.2 | **winner** |
| 起点（round-1 final @910B3） | sincnet | pass | 496.36 | 0.0% | 491.99 | 314.19 | 35.07 | 78.08 | 186.80 | 1030.0 | 2179.2 | 不调优 |
| **r2_phase（交付）** | sincnet | pass | **496.37** | **+0.0%** | 492.24 | 314.19 | 35.05 | 77.89 | 186.82 | 1030.0 | 2179.2 | **非回归 ✓** |

> 交付文件上 ecg 独立复测两次：**87.66 / 87.28 µs**（20 launch median）。
> 同一份代码（仅 docstring 不同的分支文件）在本轮另两次测量得到 100.20 / 112.28 µs ⇒
> **BP_run_state_bimodality 再次显形**（同一 tile 跨会话 87–113 µs）。所有结论用交付文件的两次一致读数。

## 11.6 ecg 相位路径的 tile 扫描（Autotune Log）

搜索面 = `(CH, bl)`，其余参数由代数唯一确定（`pp = round_up(bl + (k-1)*d/2, 8)`、`rows`、`pad`）。
所有点同一 910B3 / `msprof op` 口径（`sweep_phase.py`，profile 在 `profiles/r2_20261020/phase_sweep/`）：

| (CH, bl) | lanes | 逻辑单元 | Task(us) | 备注 |
|---|---:|---:|---:|---|
| (64, 40) | 2560 | 832 | **87.7–91.4** | **winner**（交付） |
| (64, 48) | 3072 | 704 | 不可编译 | ub overflow：1 570 816 / 1 572 864 bit（缺 2 048 bit） |
| (64, 32) | 2048 | 1024 | 110.41 | |
| (64, 24) | 1536 | 1376 | 103.80 | |
| (64, 16) | 1024 | 2048 | 131.11 | 32 B 写行 |
| (128, 16) | 2048 | 1024 | 131.88 | 32 B 写行 |
| (128, 32) | 4096 | 512 | 不可编译 | ub overflow（1 896 448 bit） |
| (128, 40) | 5120 | 416 | 不可编译 | ub overflow（2 551 808 bit） |
| (96, 32) | 3072 | 1024 | 185.58 | C=128 部分通道块（96+32） |
| (48, 64) | 3072 | 768 | 113.24 | C=128 部分通道块（48+48+32） |

结论（写进 `_phase_plan` 的选择规则）：**lane 数不是唯一目标**；只有 **C 的精确因子** + **更大的 bl**
才同时避免 padding 浪费与窄写行。规则 = 精确因子候选 → 取 `bl*ch` 最大（并列取更大 bl），
UB 用 `_phase_bytes` 代理（258 KB 可编译 / 319 KB overflow，预算 300 KB），并在 build 失败时回退 general grid。

## 11.7 合并非回退、smoke 与中止判定

- 合并：`R4`（flat store）与 `R5…R10`（相位路径）合入同一 `@tilelang.prim_func` 集合；
  路径选择全部是**编译期**代数条件（`s_w==2` / `p_w,d_w` 偶 / `l_in` 偶 / `out_l==1`），
  无新增 Python factory/op/wrapper 性能分派；wrapper 契约（`block_m`）未变。
- 本 kernel 全部 tune workload 的非回退检查（交付文件）：textcnn −19.7%、ecg −74.2%、sincnet ±0.0% ✓。
- smoke 精度：`smoke-1x8x32-k3s2p1-{fp16,bf16,fp32}` 在 `--level all` 中全部 `max_diff=0.000e+00, idx_exact` ✓。
- 中止原因：**`success`（best_effort 口径）**——用户已判定 70% 线不可达（textcnn 目标 3.56 µs < 本机空
  kernel 地板 2.86–3.44 µs），故以「所有 tune workload 有结论 + 同一最终候选的 final 全量记录 + smoke 精度」
  收束；未触发 `budget_exhausted`（本轮 formal 分支 2 个 + 1 次 tile 扫描，远低于 `max_stage4_experiments=18`）。

## 11.8 已知残留（如实披露）

1. **`[DESIGN_LIMIT]` 被本轮实测推翻（部分）**：上轮 `perf_feedback.md` 断言「行批处理在本工具链不可实现」
   —— 该断言对 **s_w==3（sincnet，3 相位）** 仍成立，但对 **s_w==2** 已被证伪：2 相
   `deinterleave` 路线可编译、逐位精确，且把 ecg 从 339.71 降到 87.66 µs（3.87×）。
   已在 `perf_feedback.md` 追加「修正附录」（append-only，保留原文），本轮**不再**发 `[DESIGN_LIMIT]`。
2. **ecg 仍距 70% 线 3.7×**（87.66 vs 23.95 µs）。新结构下 `mte2+mte3 = 60.6 µs`、`vec+scalar = 70.9 µs`：
   剩余大头是**索引路径每 tap 3 条 op**（vcmp + vsel + vmin，共 15 条/unit），工具链无向量 argmax 原语
   （float 域只有 max/min/select），要再降需 3 层「值-索引对」树形合并（估计到 ~60 µs），本轮预算内放弃。
3. **ecg bl=48 差 2 048 bit 未能采用**（少 15% 逻辑单元）；需要再省一个 `(rows,CH)` 平面
   （例如把 `negr` 用 `pramp+1` 消掉），属可继续的确定项。
4. **ch == 32 的相位路径正确性异常未定位**（见 11.2 P3.4），已从候选排除；若后续要放宽 tile 搜索，
   必须先复现并定位（当前只保证排除了该宽度）。
5. **同一 tile 跨会话 87–113 µs 的双态**（11.5 注）：本轮所有判定用交付文件上两次一致的读数；
   若后续复测落在 ~100 µs 档，先按 `ab_test.py` 交错协议复核再改判。
6. 相位路径不覆盖 `s_w != 2`、`p_w` 奇、`d_w` 奇、`l_in` 奇（回退 general grid，功能无损）。

## 11.9 Skill Retrospective（本轮增量）

| id | vp_type | 结论 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-D4 | **D** | **`s_w==2` 的窗口池化可用 2 相 `deinterleave` 做行批处理**：`(CH,lspan)` span → deinterleave → 2D transpose → `(pp,CH)` 相位块，每 tap 1 条宽 op；ecg 339.71→87.66 µs（3.87×），值+int64 索引逐位精确 | `perf_records.jsonl` round 5/6；`probe_phase_ecg.py`；§11.6 扫描表 | `probe_phase_ecg.py --check/--run --ch 64 --bl 40` | 910B3 / d2783bd / CANN 9.0.0 |
| VP-D5 | **D** | **1 元素行的 GM store 放大 21–25×，与转置无关**：同字节下 `(real_c,1)` region = 96 KB（Task 85.60 µs）；1-D region = 3.8 KB（3.44 µs）。隔离 A/B 只改写入形态 | `probe_store_r2.py` A/B/C；§11.2 P1 | `probe_store_r2.py --variant A/B --run` | 同上 |
| VP-D6 | **D** | **UB buffer 声明顺序可改变同代码性能 ~12%**：仅把相位的 `alloc_ub` 顺序对齐到探针顺序，ecg 112.28 → 100.20 µs（IR 语义相同） | `_…_r2_phase.py` 两次测量；§11.4 | 分支文件 + `run_experiments.py --phase candidate` | 同上 |
| VP-P3 | **P** | **「相位域退化」纪律**：判定行批处理可行性时先做**代数相位检查**（所有 tap 的 `iw` 是否同一奇偶），再谈 API —— `s=2,d 偶,p 偶` ⇒ 相位域 stride `d/2` 普通滑窗；`s=3` 无此性质 | §11.2 P2 + `recheck_20261020` §2.2 | `probe_phase_ecg.py`（plan() 里的 applicability 断言） | 同上 |
| VP-P4 | **P** | **`vsel` 的操作数约束必须单列成三维（cond/src0/src1/dst × 整 buffer/切片）**：vsel 全位置拒切片，`vadd/vcmp/vmax/vmin` 接受切片 ⇒ 索引类 kernel 的 mask 应走「整 buffer select + 切片 add」 | `probe_vsel.py` v1–v6 | `probe_vsel.py --variant v2/v3` | 同上 |
| VP-R2 | **R** | skill 的 gate-4 契约（每个 workload 恰一条 `phase=final` + 与交付文件哈希对账）与 skill §3.1② 的「续跑 append 新 final」**互相冲突**；需要 conductor/skill 明确「覆盖」的机械实现（本轮：旧 final 原样移入 `perf_records_superseded.jsonl`） | 本轮 gate 自检 + `.agents/tools/gate_lint.py::_perf_records_workload_lint` | — | — |
| VP-C2 | **C** | 案例索引续档：`CASE-pool-maxpool1dindices-migration` 的 round-2 段（2 相 deinterleave 路线成立、flat store 25×、vsel 切片封死、ch32 异常、UB 顺序效应、bl48 差 2048 bit） | 本节全文 | — | 同上 |

### 11.9.1 skill 流程问题

| # | 问题 | 证据 | 建议 |
|---|---|---|---|
| S1 | Phase 2 缺「同一份代码跨会话双态」的自动化复核 | 本轮同一 tile 87.3 / 100.2 / 112.3 µs 三档 | `run_experiments.py` 对 final 记录强制跑 2 次并写两条 median（或接 `ab_test.py`） |
| S2 | `[DESIGN_LIMIT]` 的推翻通道只在 skill §Phase3 第 3 条一句话，未给「修正附录」的固定 schema | 本轮需自行设计附录结构 | `perf-feedback.md` 增「修正附录」模板（触发条件 / 原结论边界 / 新证据 / 存活硬上限） |
| S3 | UB 预算完全依赖后端报错（无文档化的 per-buffer 复用模型） | §11.6 三点「不可编译」与 naive sum 的非线性关系 | pattern-library/constants.md 增「naive sum vs backend requirement ≈ 1.4–1.7×」条目 + repro |

---

# 12. Round-3 continuation（2026-10-20，Ascend910B3 / tilelang d2783bd）

> 本节是第三轮（**有界、限 scope**）的全部证据。起点 = 上轮交付文件
> `perf_opt/_max_pool1d_with_indices_kernel.py`（sha `ea64b906…`，已备份为本目录
> `_max_pool1d_with_indices_kernel_r3_parent.py`）。时延口径仍唯一：`msprof op`
> `Task Duration(us)`，20 launch median、`--kernel-name=main`、`flock /tmp/tilelang-msprof.lock` 串行，
> AIV 40 / UB 196 608 B。

## 12.0 版本戳（本轮实测）

| Item | 本轮 value |
|---|---|
| npu | **Ascend910B3** ×1（`get_device_name`）；`get_aicore_num()` = 20 → 纯 Vector 40 |
| tilelang | `0.1.2+9628fbd2310b92cf6eccff3ac58bb800565fc5fd`（`git HEAD = 9628fbd`；**编译产物 `build/` 仍为 2026-10-13**，且 `d2783bd → 9628fbd` 的 diff 只含 `.agents/skills` 文档 / repro / queue，**无 tilelang 源码改动** ⇒ 与上轮 `d2783bd` 同一工具链档） |
| CANN / bishengir | 9.0.0 / `1.1.0 (428ab8fdab46, llvm 19.1.7 e4ce5939f6e1)` |
| torch / torch_npu | `2.9.0+cpu` / `2.9.0.post8` |
| 空 kernel Task 地板（沿用本机实测） | 2.86–3.44 µs（§11.0） |
| 预注入条目 | 多数 `stale`；本轮作决策依据者均在本机重验（见 12.2 / 12.6 的「重验」列） |

## 12.1 起点（同设备 round-2 final）与本轮 scope

| workload | 起点 Task(us)（round-2 final） | 本轮处置 |
|---|---:|---|
| textcnn-global-float16 `[64,256,128]` k128 s1 p0 d1 | **107.27 / 107.46**（round 6 / round 5） | **唯一调优目标** |
| ecg-cnn-dilated-bfloat16 `[16,128,2048]` k5 s2 p2 d2 ceil | **87.66** | 不调优，仅非回归（双次读数） |
| sincnet-speaker-local-float16 `[32,80,4096]` k3 s3 p0 d1 | **496.37 / 496.36** | 不调优，仅非回归（双次读数） |

工作负载集合不变（`workload_inventory.json` 的 3 smoke + 3 tune，本机按
`benchmarks/ops/bench_pool.py::test_max_pool1d_indices_bench` 参数化展开）；smoke 只做精度。

## 12.2 Current Phenomena（起点 profile + 本轮先导探针）

- **P1（textcnn，起点）**：`aiv 91.10 / vec 55.55 / scalar 28.97 / mte2 2.55 / mte3 0.97`，
  `GM_to_UB 96 KB`（= 真实读 105 KB 的 6 单元档，读侧已 1×）、`UB_to_GM 3.8 KB`（真实写）。
  ⇒ **写侧与读侧都已到地板**（上一轮 R4 的扁平 1-D store 已把 mte3 从 60.7 降到 ~1.0 µs），
  剩余全部是**向量 op 条数**：`k_w*bl = 128` 次 visit × 2 遍 × 3 op ≈ 每逻辑单元数百条窄 op。
- **P2（本轮先导探针，只读，不进入交付链）**：`recheck_20261020/FOLD_REDUCE_PROBE.md` +
  `probe_fold_textcnn.py` / `probe_fold_textcnn_idx2.py` / `probe_fold_textcnn_eqmin.py`（同一 910B3、
  同口径 20 launch median）：
  | 探针 | 结构 | 逐位精确 | Task µs |
  |---|---|---|---:|
  | 起点交付（R4） | 128 tap 窄 op 扫描 | ✓ | 107.46 |
  | `probe_fold_textcnn.py` | 仅值：7 级 `vmax`（1-D 连续块） | ✓ | 6.32 |
  | `probe_fold_textcnn_idx2.py` | 值 + 索引**同步**折叠 | 值 ✓ / **索引 tie 错 15/35** | 机制被证伪 |
  | `probe_fold_textcnn_eqmin.py` | 值 7 级 + `eq/ramp` **最小索引** 7 级（ch=64） | **✓ 全 16384 输出** | **11.12** |
  | 同上 ch=32 | — | ✓ | 13.35 |
- **P3（本轮必须复现的两条关键结论）**：
  1. **不要**「值 + 索引同步折叠」：对折树把 `i` 与 `i+L/2` 配对，左子树索引集 `{i, i+L/2}`
     不是全小于右子树的 `{i+L/4, i+3L/4}` ⇒「平局保左」≠「保最小索引」（实测 15/35 tie 错）。
     正确形态 = **先折值、再 `eq` 掩码 + 最小索引折叠**（零 tie 逻辑）。
  2. `CONST-dual-output-whole-plane-reduce-ub`（整平面双输出 argmax 在 ch=16 即 UB overflow）
     与本方案**不冲突**：缓冲区随级数递减（`64ch→32ch→…→1ch`），全程无常驻满尺寸双平面。
     本轮落地时守住该点（见 12.4 的 UB 记账与 12.6 的 ch 扫描）。
- **P4（本轮新踩到的一个实现级陷阱，已定位并修复）**：NaN 掩码 `vselect` 的**方向**。
  `isnum = (x == x)` 在**非 NaN** 行为真，因此索引侧必须是
  `vselect(isnum, -1, ramp, cand)`（非 NaN → -1，NaN → 自身 ramp）；
  首版写成 `vselect(isnum, ramp, aux, aux)`（且 dst==src1 原位）⇒ `n7` 恒为 `L-1` ⇒
  全 lane 判为「含 NaN」⇒ 输出全 NaN、索引全 7（`(1,8,8)` 小形状实测复现，
  textcnn 亦为全 NaN）。**修正后小形状与 textcnn 全部逐位精确**。教训：NaN 追踪链的掩码方向
  必须用「全 NaN / 单 NaN / 无 NaN」三态小案例验证（已固化为 `L0-17e`）。

## 12.3 Candidate Optimization Points（本轮候选）

| opt_id | 目标现象 | 优化点 | 判断依据（实测/文档） | 改法 | 状态 |
|---|---|---|---|---|---|
| **R13** | P1 | `L_out == 1` 的**分层对折归约**（halving reduce）：整窗 = 整条 L 轴 ⇒ `log2(l_in)` 级宽 op 取代 `k_w*bl` 窄 op | P2 探针：值+索引 11.12 µs（逐位精确）；同族 `PL-1.23-flat-spatial-axis-reduce`（本轮重验：对 `L_out==1` **成立**，对上轮「窗口类不可移植」的判定是**例外**） | 新增 `_fold_plan` + `_func_fold`（编译期代数守卫，见 12.4） | **采纳（本轮唯一结构分支）** |
| R14 | P1 尾项 | CH 邻域粗扫 `{32,64,128}` | 12.6 实测：64 最优 | `MAXPOOL1D_FOLD_CH` 覆盖 + 一次扫描 | 采纳（取 64） |
| R14b | 覆盖 | 补 `ceil_mode` 两态测试（仍走折叠 / 因 `L_out=2` 回退）+ 修正 `c-not-multiple` 标签并补 `C=20/ch=24` 真非整除用例 | 用户要求的覆盖清单含 ceil 与「C 非 ch 倍数」；`k==l, s=1, p=0, d=1` 下 ceil 不改 `L_out` | `L0-17j` / `L0-17k` / `L0-17f` + 路径归属断言 | 采纳（仅测试，内核未改；final 重采） |
| R15 | — | 把 NaN「是否含 NaN」与 last-NaN 索引合并进同一条折树 | 三条归约（值 / 首命中 / last-NaN）语义互斥（min vs max），无单树合并形态；无向量 argmax 原语 | — | **不做**（记录为残余 12.8-2） |
| R16 | — | 本次不尝试 `T.reduce_max` 整平面归约替代 7 级手写折树 | `T.reduce_*` 的 UB 单价条目绑定 3-D 整平面 + dim=2 形态（`CONST-reduce-ub-per-element`），与 `(l_in,ch)` 行归约形态不同；且手写树已把「第 1 级」做在 8192 lane 上（单价 40–60 ns 档） | — | **不做**（预算内不冒险换原语） |

## 12.4 Experiment Branches（本轮）

| branch | parent | opt_id | 主要改动 | L0 | textcnn(us) | 结论 |
|---|---|---|---|---|---:|---|
| `_…_r3_fold.py`（= 交付文件） | `_…_kernel.py`（ea64b906） | R13 | `L_out==1` 全窗对折归约 + 三 dtype + 空窗守卫 | **all pass** | **17.35 / 19.94**（同字节两次读数） | **winner（−83.8%）** |

**落地形态（每条都对应一条已知铁律）**：

```
读 x[nb, c0:c0+rc, 0:L] -> src (ch,L) --T.transpose[1,0]--> tr (L,ch)
值   ：7 级 vmax（v1..v7，行数 L/2..1，每级操作数都是整行连续块）
首命中：vbrc(v7 -> mb) -> vcmp(tr, mb, eq) -> vselect(eq, pramp, aux(BIG), cand) -> 7 级 vmin
lastNaN：vcmp(tr, tr, isnum, "eq") -> vbrc(-1, aux) -> vselect(isnum, aux, pramp, cand) -> 7 级 vmax
epilogue（均为 (1,ch)）：has_nan = (n7 >= 0) -> value = has_nan ? NaN : v7；idx = has_nan ? n7 : o7
                        -> 空窗守卫（idx == L ⇒ idx = 0，值 -inf）-> fp32→int32→int64 -> 扁平 1-D 写
```

- **路径守卫（编译期代数，缺一回退）**：`out_l == 1` ∧ `k_w == l_in` ∧ `s_w == 1` ∧ `p_w == 0`
  ∧ `d_w == 1` ∧ `l_in` 为 2 的幂 ∧ `2 <= l_in <= 128`（7 级上限）∧ `l_in <= 2**24`（fp32 索引域精确）。
  不满足者回退 phase / general（功能无损）；`L0-17f` 逐条断言 12 个 near-miss 的路径归属。
- **朴素 UB 记账** `_fold_bytes`（ch=64, L=128, f16）= **248 704 B**：`src`+`tr`+`mb` 各 16 384、
  三条折树共 `3*(L-1)*ch*bytes`、三条满尺寸 fp32 平面 `pramp/aux/cand` 各 32 768、`eq/isnum` 各 8 192。
  **递减是关键**：树缓冲随级数减半，不存在常驻的满尺寸双输出平面（`CONST-dual-output-whole-plane-reduce-ub`）。
- **vselect 只吃整 buffer**：`pramp/aux/cand/mb` 全部整 `(L,ch)`；只有 `vmax/vmin/vcmp` 吃同形切片。
- **ramp 武装（`CG-2026-0031` 第四条 + `T.arange` 文档 §2.3）**：**1-D** 连续 `T.arange(ramp_f, [1], 0)`
  → `T.reshape` 成 `(L,1)` → `T.vbrc` 行广播到 `(L,ch)`；**int32 ramp 不进入任何比较**（索引域 = f32，
  `l_in <= 2**24` 精确），避免「静态行 ramp 喂 vcmp/vselect 链」的崩溃形态与 2-D arange 的
  PIPE_S→PIPE_V 缺同步。
- **NaN 三态验证**：`L0-17e`（last-NaN / tie→首命中 / 全 -inf / NaN 压过最大值 / 全 NaN，
  含手推期望值）。这正是 P4 那个方向 bug 的暴露集。
- **空窗守卫**：`k == l_in ∧ p == 0` 下窗口必满、守卫结构性不可达，但保留（与 general 同一契约）。
- **`l_in` 上限 = 128**：折树按 7 级静态展开（`TRAP-tir-range-loop-not-unrolled`：`for i in range(n)`
  是 TIR 串行循环，不能做缓冲选择）；超过 128 一律回退 general。**这是本轮的有界取舍，已在
  `L0-17f/l>128` 断言**。

## 12.5 候选 vs current best 对比表（B2 结构化回流；数据源 `perf_records.jsonl` + raw profile）

| candidate | workload | L0 | Task(us) | vs best | aiv(us) | aiv/Task | vec | mte2 | mte3 | scalar | GM_to_UB(KB) | UB_to_GM(KB) | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 起点 r2_phase（round-6 final） | textcnn | pass | 107.27 | 0.0% | 91.10 | 84.9% | 55.55 | 2.55 | 0.97 | 28.97 | 96.0 | 3.8 | best |
| **r3_fold ch=64（交付）** | textcnn | pass | **17.32** | **−83.8%** | 13.89 | 80.2% | **9.52** | 2.50 | 1.00 | **2.58** | 96.0 | 3.8 | **winner** |
| r3_fold ch=32（R14 扫描） | textcnn | pass | 22.35 | −79.2% | 20.32 | 90.9% | 13.41 | 2.76 | 1.09 | 4.19 | 104.0 | 4.4 | 回退（−28% vs 64） |
| r3_fold ch=128（R14 扫描） | textcnn | pass | 62.46 | −41.8% | 53.56 | 85.7% | 33.30 | 2.52 | 0.98 | 16.10 | 96.0 | 3.8 | 回退（−3.6× vs 64） |
| 起点 r2_phase（round-6 final） | ecg | pass | 87.66 | 0.0% | 84.08 | 95.9% | 51.09 | 20.90 | 39.74 | 19.81 | 336.0 | 990.2 | best |
| **r3_fold（交付；ecg 走 phase 不变）** | ecg | pass | **87.50** | **−0.2%** | 84.40 | 96.5% | 51.09 | 21.17 | 39.29 | 19.74 | 336.0 | 990.2 | **非回归 ✓** |
| 起点 r2_phase（round-6 final） | sincnet | pass | 496.37 | 0.0% | 492.24 | 99.2% | 314.19 | 35.05 | 77.89 | 186.82 | 1030.0 | 2179.2 | best |
| **r3_fold（交付；sincnet 走 general 不变）** | sincnet | pass | **496.10** | **−0.1%** | 491.96 | 99.2% | 314.19 | 34.95 | 78.24 | 186.83 | 1030.0 | 2179.2 | **非回归 ✓** |

> `BP_run_state_bimodality`（§5 铁律 7）：**交付文件（sha `1d6aaf3e…`）上双次读数**为
> textcnn **17.32 / 17.34 µs**（差 0.1%）、ecg **87.50 / 87.53**、sincnet **496.10 / 496.33**。
> 另：同字节的内核在更早一次会话读到 textcnn 19.94 µs（该档 `GM_to_UB` 100 KB / `UB_to_GM` 4.4 KB，
> 字节数本身漂移，是已知双态签名）⇒ 19.94 判为 run-state 高档，不进最终口径。
> 所有采纳/否决判定均用交付文件上的一致读数。
> **交付文件的三个 final 代次**（内核代码完全相同，只有嵌入测试套件变化）：
> `929c18d7` 17.42 / 17.39；`67ae08d6` 17.31 / 17.32；`1d6aaf3e` 17.32 / 17.34
> —— 三代极差 0.6%，在噪声带内。

## 12.6 CH 邻域扫描（Autotune Log）

`MAXPOOL1D_FOLD_CH` 一次性粗扫（同一 910B3 / 同一 `msprof op` 口径；`perf_records.jsonl` round 7）：

| CH | lanes/宽 op | 逻辑单元 | nloc | Task(us) | vec | scalar | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 4096 | 512 | 13 | 22.35 | 13.41 | 4.19 | 单元数翻倍 + 倍率级收益递减 |
| **64** | **8192** | **256** | **7** | **17.35 / 17.42 / 17.39** | **9.52** | **2.63** | **winner（默认）** |
| 128 | 16384 | 128 | 4 | 62.46 | 33.30 | 16.10 | 行宽 256 B 反而 3.6× 慢（同 lane 总量） |

结论：**lane 总量相同不代表等价**——`ch=128` 的 `(128,128)` 转置块（32 KB/单元）与 256 B 行宽
在本结构上把 `vec` 抬到 3.5×、`scalar` 抬到 6×；`ch=32` 则是单元数翻倍的开销。
默认规划器取「满足 UB 预算的最大 ch」，在本 shape 上即 64（`_FOLD_UB_SUM_BUDGET_BYTES = 260 000`
恰好排除 128；即便不排除，128 也是 3.6× 回退）。

## 12.7 合并非回退、smoke、final 覆盖与中止判定

- **合并**：R13 直接加在同一个 `@T.prim_func` 集合里（`_build` 内 fold → phase → general 的
  **build 期**尝试序），路径选择全部是**编译期代数条件**；没有新增 Python factory / op 层 /
  wrapper 性能分派，wrapper 契约（`block_m`）未变。
- **合并非回退（交付文件、双次读数）**：textcnn **−83.8%（6.19×）**、ecg −0.2%（87.50 vs 87.66）、
  sincnet −0.1%（496.10 vs 496.37）✓。
- **smoke 精度**：`smoke-1x8x32-k3s2p1-{fp16,bf16,fp32}` 在 `--level all` 中全部
  `max_diff=0.000e+00, idx_exact` ✓。
- **全量回归**：`python perf_opt/_max_pool1d_with_indices_kernel.py --level all`
  = **L0 ALL PASS / L1 ALL PASS / L2 与 Boundary 无 WARN**，
  `worst value max_diff = 0.000e+00`（bf16/fp16/fp32）、`index mismatches 0/12 219 806`。
  新增折叠路径覆盖：`L0-17a…d`（L64-fp16 / L32-bf16 / L16-fp32 / C=20 部分通道块）、
  `L0-17e`（NaN/last-NaN/tie/全 -inf / **全 NaN**）、`L0-17f`（14 条路径归属断言，含 9 条 near-miss）、
  `L0-17g…i`（fallback：非 2 的幂 L / d>1 / l=1）、`L0-17j`（**ceil_mode 且 k==L ⇒ 仍走折叠路径**、
  逐位精确）、`L0-17k`（**ceil_mode 使 `L_out=2` ⇒ 回退**）。
- **`perf_records.jsonl` 的 final 覆盖（本轮按「成文文件名」记账）**：round-3 有三代 final
  —— ① 内核改动后首测（sha `929c18d7`）；② 补两条 ceil_mode 用例后重采（sha `67ae08d6`）；
  ③ 修正测试标签 + 补真正的非整除用例后重采（sha `1d6aaf3e`）。三代共 9 条旧 `phase=final`
  （含 round-6 的 3 条 `ea64b906`）均**原样**移入 `perf_records_full_history.jsonl` 并逐条追加
  `superseded_reason`；该文件**前 3 行**是更早一代 `perf_records_superseded.jsonl` 的 3 行**逐字节**
  并入（旧文件保留不删，理由：上一轮既有工件，删除会破坏 append-only 留痕；
  范围 = 12 行 = 4 个历史 final 代次）。`perf_records.jsonl` 现 37 行
  （3 baseline + 31 candidate + 3 final），每个 tune workload **恰一条** `phase=final`
  （round 7，sha `1d6aaf3e…`）⇒ gate 4 可对账。
- **中止原因**：**`success`（best_effort）**——textcnn 从 107.27 → 17.42 µs（6.16×，
  相对 round-0 baseline 169.60 → −89.7%），三个 tune workload 全部有结论、同一最终候选的 final
  全量记录 + smoke 精度通过。70% 线（3.56 µs）**低于本机空 kernel 地板 2.86–3.44 µs**，
  口径不可达（用户已裁定 best_effort，不为达标牺牲非回归/精度）。
  本轮 formal 分支 4 个（fold-ch64 / CH=32 / CH=128 / 覆盖补测），
  ≤ `max_stage4_experiments = 8`；final 采集按 skill 要求逐 workload 双次读数
  （3 个 final 代次 × 3 workload × 2 次 = 18 次 `msprof`），不计入分支预算（与上轮同口径）。

## 12.8 已知残留（如实披露）

1. **折叠路径只覆盖 `l_in <= 128`**（7 级静态展开）。`l_in = 256/512/…` 的整窗池化仍走 general
   （功能无损、性能未优化）。若要扩档，需要「按级数生成 prim_func 变体」（
   `TRAP-primfunc-compile-time-if-buffer-def` 禁止在解析期 if 内定义 buffer），或改用
   `T.reduce_max` 整平面归约（形态与单价未在本机标定，属 R16 未做项）。
2. **textcnn 距 70% 线仍 4.9×**（17.42 vs 3.56 µs），其中 ~2.4–2.6 µs 是 `msprof` 固定
   发射开销（`CONST-msprof-task-floor-910B3` 拟合 `Task ≈ aiv + 2.4–2.6`：17.42 − 13.94 = 3.48 含地板），
   `vec 9.52 µs` 来自 3 条互斥归约（值 min-max 树 / 首命中 `eq`+min / last-NaN max）+ 2 条满平面
   `vcmp` + 2 条满平面 `vselect`，共 23 条宽 op/单元；工具链无向量 argmax 原语，无法再合并。
3. **`ch=128` 的 3.6× 回退根因未定位**（同 lane 总量、读字节相同、`vec` 3.5×）；
   已把它排除在默认候选之外（预算 + 实测双保险），但**未**给出机制归因（怀疑 `(128,128)` 转置
   块的物化形态与 256 B 行宽的 bank/对齐行为，未验证）。
4. **`ch=32` 在 fold 路径上的索引正确性本轮未单独复验**：上一轮 phase 路径在 `ch=32` 有索引
   异常（304/2 095 104 元素）。本轮 `ch=32` 的扫描走 `bench.py --check` 之外的 runner 路径
   （L0 pass 是**整文件** L0，含 `L0-17*`，但默认 CH=64；ch=32 仅由 `MAXPOOL1D_FOLD_CH=32`
   环境变量触发，未单独跑 L0）⇒ **若后续要放宽 CH 候选到 32，必须先单独跑一次 L0**。
   本轮默认 CH=64，故不影响交付。
5. **`ca. 4.9×` 的剩余缺口里 `scalar 2.63 µs`**（起点 28.97）已降 91%，但仍有 0.4 µs/单元级的
   标量项（`(1,ch)` epilogue 与 store 下标），未单独优化。

## 12.9 Skill Retrospective（本轮增量）

| id | vp_type | 结论 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-D7 | **D** | **`L_out == 1` 整窗池化的第三条结构路线（分层对折归约）**：值 7 级 `vmax` + `eq/ramp` 最小索引 7 级 `vmin` + `isnum/-1` 最大索引 7 级 `vmax`，textcnn **107.27 → 17.42 µs（6.16×）**，值 + int64 索引逐位精确；`PL-1.23` 对 `L_out==1` **成立**（对上轮「窗口类不可移植」是例外） | `perf_records.jsonl` round 7；`recheck_20261020/probe_fold_textcnn_eqmin.py`；本文件 12.4/12.5 | `recheck_20261020/probe_fold_textcnn_eqmin.py --check --ch 64` | 910B3 / d2783bd / CANN 9.0.0 |
| VP-D8 | **D** | **反例（避免重复踩）**：「值 + 索引同步对折」不成立——对折树配对 `i ↔ i+L/2`，左子树索引集非全小于右子树 ⇒「平局保左」≠「保最小索引」，35 个 tie 里 **15 个索引错**（值仍精确）⇒ 值-only 门禁会漏判。索引必须走「先折值 → `eq` 掩码 → 最小索引」二段式 | `recheck_20261020/probe_fold_textcnn_idx2.py`；`FOLD_REDUCE_PROBE.md` §2.1 | 同上探针 `--check` | 同上 |
| VP-D9 | **D** | **同一折叠结构上 `ch` 翻倍 ≠ 更快**：`ch=128`（行宽 256 B、`(128,128)` 转置块 32 KB）在**同 lane 总量、同读字节**下 `vec 9.52 → 33.30 µs`、`scalar 2.63 → 16.10`、Task **17.35 → 62.46 µs（3.6× 回退）**；且 ch=128 朴素 UB 合计 ~487 KB **仍可编译** ⇒ 朴素 UB 预算不可跨结构外推（phase 路径 425 KB 曾溢出） | `perf_records.jsonl` round 7（四行）；12.6 表 | `MAXPOOL1D_FOLD_CH=128 python run_experiments.py --phase candidate --branch _max_pool1d_with_indices_kernel_r3_fold.py --workload textcnn-global-float16` | 同上 |
| VP-P5 | **P** | **「整窗 = 对折归约」的落地纪律**：① 折叠操作数必须是整行连续块（行 stride 与 buffer 一致，否则 `extract_slice` 物化 + 标量化）；② 索引侧 mask 走整 buffer `vselect` + 切片 `vmin/vmax`（vsel 拒切片）；③ ramp 用 **1-D arange + reshape + vbrc 行广播**（2-D arange 是标量写且缺 PIPE_S→PIPE_V）；④ NaN 掩码方向必须用「全 NaN / 单 NaN / 无 NaN」三态验证（本轮首版方向写反 ⇒ 全 NaN + 索引全 `L-1`） | 本文件 12.4 P4 + `L0-17e`；`docs/Tilelang.language/创建操作/T.arange.md` §2.3 | `_max_pool1d_with_indices_kernel.py --level L0`（`L0-17e`） | 同上 |
| VP-R3 | **R** | skill 的 gate-4「每 workload 恰一条 `phase=final`」与 append-only 台账在**续跑**时必须有「final 覆盖」的机械实现；本轮沿用上轮形态但要求**成文文件名**：旧 final + 上一代 `perf_records_superseded.jsonl` 一并逐字节/原样汇入 `perf_records_full_history.jsonl` 并加 `superseded_reason`，旧文件保留 | 本文件 12.7；`perf_records_full_history.jsonl` | — | — |
| VP-C3 | **C** | 案例索引续档：`CASE-pool-maxpool1dindices-migration` 的 round-3 段（对折归约 6.16×、同步折叠反例 15/35、ch=128 的 3.6× 回退、1-D ramp 武装、NaN 掩码方向 bug） | 本节全文 | — | 同上 |

### 12.9.1 skill 流程问题（本轮）

| # | 问题 | 证据 | 建议 |
|---|---|---|---|
| S4 | 先导探针已把「值+索引 eq/min 折叠」跑通到逐位精确，但**掩码方向**这类语义 bug 只会在接入 kernel 后才暴露（探针的 ramp/BIG 由 host 传入，方向由 host 数据隐含） | 12.2 P4：首版 vselect 方向反了，输出全 NaN | 探针阶段就应提供「由 kernel 自己武装 ramp/BIG」的形态，或用三态 NaN 小案例验收（`L0-17e` 已固化） |
| S5 | 折叠归约的「级数上限」缺乏工程化形态：7 级手写展开可行，8 级以上不可维护（`for i in range(n)` 是 TIR 循环、解析期 if 内不能定义 buffer） | 12.8-1 | skill/pattern-library 增「按级数生成 prim_func 变体」的工厂模板（多返回一个 prim_func 即可，不改分派层） |
