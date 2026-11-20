# Stage 4 性能调优日志 — `_fp8_lightning_indexer_kernel`（FP8LightningIndexerOp, NPU / `target="npuir"`）

- 本轮：scenario=optimize，mode=`full`，预算 18 个实验分支 / 120 min。
- 调优对象（gate 4 对账文件）：`examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer_kernel/perf_opt/_fp8_lightning_indexer_kernel.py`
- 只读基线：`examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer_kernel/_fp8_lightning_indexer_kernel.py`（sha256 `b71bf8e3…`，与上一轮 final 相同）
- wrapper 未修改：`examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer.py`
- 历史工件（上一轮 4 轮/12 分支的 log、records、inventory、候选文件）已原样移入 `perf_opt/history/`，未被改写。
- 唯一时延口径：`msprof op` → `OpBasicInfo*.csv` 的 `Task Duration(us)`；Ratio 口径与官方 benchmark 一致（`--aic-metrics=Roofline` → `visualize_data.bin` 的 `GM Read + Write` roofline 点，见 `examples/TileOPs/tileops/benchmark/benchmark_base.py` 的 `_parse_msprof_roofline`）。

## 1. Workload Inventory

见 `perf_opt/workload_inventory.json`。来源 `examples/TileOPs/benchmarks/ops/bench_fp8_lightning_indexer.py::test_fp8_lightning_indexer_bench` → `load_workloads('FP8LightningIndexerOp')` → `_indexer_params()`（按 shape 去重）。

| # | workload_id | marks | kind | shape (b,s,h,d,skv,g) |
|---|---|---|---|---|
| 1 | `lightning-indexer-smoke-s1k-h8-d64` | smoke | smoke（仅精度） | 1,1024,8,64,1024,1 |
| 2 | `lightning-indexer-bf16-s8k-h32-d64` | full | tune | 1,8192,32,64,8192,1 |
| 3 | `lightning-indexer-bf16-s4k-h32-d64` | full | tune | 1,4096,32,64,8192,1 |
| 4 | `lightning-indexer-fp8-s8k-h32-d64` | (full) | skipped | 同 #2；`CG-2026-0022`（npuir 无 fp8 链路）+ 参数化按 shape 去重折叠 |
| aux | `clean_logits_`（`.../clean_logits_.py`） | — | no_gain | 本轮未改、未重测；`perf_opt/clean_logits_.py` 与 Stage 3 文件逐字节相同 |

benchmark 的输入分布：`FP8LightningIndexerWorkload.generate_random_cu_seqlens(cp_size=1, kv_stride=1, average_q_len=512)`，即**每个 q 行只有一个因果窗口**（`ks`=段起点，`ke=ks+(j+1)`），窗口内元素仅占全矩阵的 **4% 量级**。本日志所有 baseline/final 采集均用该真实分布（`ratio_bench.py` 内置同构生成器），不用退化全窗输入。

## 2. 本轮核心：把「roofline 153 µs（71×）」与「官方 Ratio 9%」对齐

上一轮遗留的「bench 口径 roofline 153 µs，当前 10.92 ms 是其 71×」与本轮官方 `Ratio=9.04%` 并不矛盾，它们是**两个不同模型**，且其中一个对本算法不适用。逐项拆开（s8k，`S=8192, Skv=8192, H=32, D=64, G=1`）：

**(a) 153 µs 的来源（纯写出口径）**
`bytes = S·Skv·G·4 = 268.4 MB`（fp32 Logits 一次全量写），除以 `gm_bw = 1.75 TB/s` → **153.4 µs**。该口径**不计任何计算、不计 Q/K/scale 读取**，只回答「如果这个 kernel 只负责把 Logits 写一遍，最快多久」。任何真实实现都到不了这个数——它必须做 `2·S·Skv·H·D = 2.75e11` 次乘加并读输入。所以 71× 是**口径误用**（拿「零计算写出口径」当算法下界），不是设计层证据。

**(b) 951 µs（=100% Ratio）的来源（官方 compute 口径）**
msprof 的 roofline 点用**硬件计数器实测 op 数**与 SoC 峰值：
- 实测 ops = `2.7488e11`（cube：`2·S·Skv·H·D`，按 2 flop/MAC 计）+ `9.566e9`（AIV：实测每中间元素 ≈4.46 个向量 op，即 relu + 加权 + 头归约）= **2.8444e11**；
- `computility = 299.09 TOPS = cube 294.9（4096 MAC/cycle/core × 48 core × 1.5213 GHz 额定）+ vec 4.19（64 fp32 lanes × 1.5213 GHz × 48）`；
- `AI = ops/bytes = 1499.5 Ops/Byte ≫ 机器平衡点 299.09/1.75 = 170.9` ⇒ `roofline_peak = min(AI·BW, computility) = computility`，即 **compute-bound**；
- `ideal = 2.8444e11 / 299.09e12 = 951.0 µs`（与报告 `20261019_151736_385411_FP8LightningIndexerOp` 的 951 µs 一致；s4k 为 475.5 µs），`Ratio = ideal / duration`。
- 交叉验证：官方 `Perf = 27.04 TOPS`、本轮复测 `26.11 TOPS`，两者乘各自 duration 都得 2.8444e11 ⇒ op 计数与采样无关，duration 差 3.5% 才是采样抖动。

**(c) 对齐结论——真正的瓶颈项**

| 项 | 数值（s8k） | 说明 |
|---|---:|---|
| 官方 ideal（100% Ratio） | 951.0 µs | 稠密 matmul 全速 + 计数向量 op 全速 + 双 pipe 完美重叠 |
| 其中 cube 理想时间 | 932 µs | `2.7488e11 / 294.9e12` |
| 其中**计数向量 op 理想时间** | **2280 µs** | `9.566e9 / 4.19e12`（fp32 峰值）——**比 cube 理想时间长 2.4×** |
| 实测 duration（baseline） | 10893 µs | 本轮复测（官方 10519.7） |
| Ratio | 8.73%（官方 9.04%） | = 951/10893 |

⇒ 结论 1：**官方 Ratio 的「天花板」不是 Logits 写带宽，而是「计数向量 op 在 4.19 TOPS fp32 峰值上的时间」**。即使 cube 与 vec 完美重叠、cube 全速，仅计数向量 op 就要 2280 µs ⇒ **fp32 epilogue 结构下 Ratio ≤ 951/2280 = 41.7%**；而 s8k 的 40% 目标（2378 µs）几乎等于该上限 —— 这是「必须先降低 epilogue 的实测代价或提高其峰值口径」的直接依据。

⇒ 结论 2（本轮最关键的可操作推论）：**计数 ops 与 dtype 无关，但 computility 的向量峰值随 dtype 变**（fp32 64 lanes / fp16 128 lanes）。把 epilogue 换成 fp16，分子不变、分母抬高、实测时间减半 ⇒ **既提速又抬天花板**。这正是 R1b/R1c 的方向。

⇒ 结论 3：**减少冗余 tile（跳过窗口外 kv tile）对 Ratio 是中性甚至有害的**：分子（计数 ops）与分母（duration）按同一比例下降，Ratio 不变，而每 tile 固定开销使 Ratio 反而略降。该优化对真实端到端时延仍极大（窗口仅占 4%），但**不是本轮 KPI 的可行方向**，故未实施。

**(d) 真实现象（msprof per-block 均值，baseline s8k；1 block = 8 q 行 × 128 kv tile，wall≈493 µs/block）**

| pipe | µs/block | 占 wall |
|---|---:|---:|
| AIV scalar | 157.0 | **31.8%** |
| AIV vector | 126.4 | **25.6%** |
| AIV mte2（K/scale 载入） | 68.8 | 14.0% |
| AI fixpipe（L0C→UB） | 60.5 | 12.3% |
| AI scalar | 88.2 | 17.9% |
| **AI cube** | **21.3** | **4.3%** |
| AIV mte3（Logits 写出） | 8.9 | 1.8% |

⇒ cube 基本空转（显存/向量主导）。每 tile（wall 3.85 µs）的 pipe 预算：AIV scalar 1.23 µs、AIV vec 0.99 µs、mte2 0.54 µs、fixpipe 0.47 µs、cube 0.17 µs。**地址/循环标量开销是第一大项**，不是带宽也不是算力。

## 3. Iteration Log

对比表列：Task Duration(us)（`msprof op`，median）、官方 Ratio(%)、AICore/AIV pipe 占比（`PipeUtilization*.csv`，per-block 均值）、L0 结果。
噪声阈值 3%；<5% 差异按 T-2 视为平区，不作 winner。

### 3.1 Baseline（round 0，phase=baseline，同一文件 `_fp8_lightning_indexer_kernel_round0_base.py`）

| workload | Task Duration (us) | min/max | 官方 Ratio(%) | AI(Ops/B) | Perf(TOPS) | L0 |
|---|---:|---|---:|---:|---:|---|
| s8k（1,8192,32,64,8192,1） | 10893.0 | 10520.2 / 10893.0 | 8.73 | 1499.5 | 26.11 | PASS |
| s4k（1,4096,32,64,8192,1） | 5449.5 | 5428.4 / 5449.5 | 8.76 | 7743.4 | 26.20 | PASS |

配置 = 上一轮 final（`block_N=64, block_Q=8, num_stages=2`，fp32 epilogue）。复现值与官方 benchmark（10519.7 / 5288.2 µs）在采样抖动内一致（median 偏高 3.5%，min 与官方一致）。

### 3.2 Round 1（structure：epilogue dtype / tile size；base = baseline）

现象（§2d）：cube 4.3% 忙，AIV scalar+vec 占 57%，fixpipe 12%；`acc_ub = q_rows·bN·4 B` 是唯一的 UB 瓶颈（fp32 bN=128 需 128 KB，越限）。⇒ 两个单点方向：dtype（降向量指令数与 fixpipe 流量）+ 由 dtype 释放的 bN 加倍（减 tile 数）。

| 候选 | 主要优化点 | config | Task Duration (us) | Δ vs base | Ratio(%) | AIV vec% | AIV scalar% | L0 |
|---|---|---|---:|---:|---:|---:|---:|---|
| current best（base） | — | bQ8/bN64/st2 fp32 | 5449.5 | — | 8.76 | 25.6 | 31.8 | PASS |
| **r1b_fp16_bn64** | epilogue 全链 fp16（L0C→UB 转换拷贝 + relu/mul/reduce 全 fp16） | bQ8/bN64/st2 | **4648.4** | **−14.7%** | **10.42** | — | — | PASS |
| **r1c_fp16_bn128** | r1b + 由 fp16 释放的 UB 余量把 bN 64→128（tile 数减半） | bQ8/bN128/st2 | **2835.7** | **−48.0%** | **17.13** | — | — | PASS（= final 文件，L0/L1 全过） |
| r1d_fp16_bq16 | 换轴：bQ 8→16（乘积同样 1024） | bQ16/bN64/st2 | 4779.0 | −12.3% | 10.13 | — | — | 未单独跑（非 winner） |
| r1e_fp16_bn128s3 | r1c + 流水深度 3 | bQ8/bN128/st3 | 2814.9 | −48.3% | 16.67 | — | — | 未单独跑（平区） |
| r1a_tree | 用 log2 树 vadd 取代 8×`reduce_sum`（想砍 scalar） | bQ8/bN64 fp32 | blocked-compile | — | — | — | — | — |
| r1f_bn256 | bN 256（乘积 2048，tile 数再半） | bQ8/bN256/st1 | blocked-compile | — | — | — | — | — |

- **winner = r1c_fp16_bn128**：s4k 5449.5 → 2835.7 µs（**1.92×**），Ratio 8.76% → 17.13%。r1e（st=3）快 0.7%，落在 <5% 平区且与 r1c 差异在采样抖动内（min 2727.8 vs 2739.1），按 T-2 不另立 winner；r1d 证明「乘积相同但 bQ 更大」更差（+68% vs r1c），与上一轮 bQ16 UB 越限的结论一致（bQ 抬高 q_ub/acc_ub，bN 才是摊薄 tile 数的高效轴）。
- 机制：fp16 使 `acc_ub` 减半（32 KB @bN=128 而非 64 KB），从而 (i) fixpipe 流量减半、(ii) 每元素向量指令数减半（128 fp16 lanes vs 64 fp32）、(iii) **bN 可翻倍使 tile 数减半**（65536→32768 @s4k），把最大的「每 tile 固定开销（scalar+DMA+sync）」按比例摊薄。三者叠加 1.92×。
- blocked 证据（原样引用）：
  - r1a：`TypeError: int() argument must be a string, a bytes-like object or a real number, not 'Var'`（`tvm/tir/buffer.py:218` `Buffer.__getitem__`）——`T.vadd` **不接受带 Var 偏移的 sub-region 实参**（只有 `T.reduce_sum` 这类特化接口可以），因此「树形归约」在本前端无法用 region 切片表达。日志：`perf_opt/logs/candidate/*` 未保留（stdout 直出），复现命令见 §7。
  - r1f：`'hivm.hir.load' op if pad_mode is not set, src and dst shape should be the same!` + `Failed to run BiShengHIR pipeline`。日志：`perf_opt/logs/candidate/s4k_r1f_fp16_bn256.log`。
  - 另：fp16 的 `T.reshape(w2h_ub, w_ub)`（`[8,4]→[32,1]`）在 grouped 路径报 `cannot align 0 axis for memref.expand_shape`，改为「先 fp32 reshape 再 cast」后修复；`bN=128` 在 grouped 路径（Hp≤8）UB 越限（`requires 2465792 bits while 1572864 bits available`），故在 factory 内以**编译期常量**把 `kv_group>1` 的 bN 夹到 64（tune workload 全是 kv_group=1，不受影响）。证据：`perf_opt/logs/round1/final_Lall.log`（修复前版本）。

**合并与全量非回退检查（合并进同一 `@T.prim_func`，即只改 dtype + tiling）**

| workload | baseline (us) | 合并后 (us) | Δ | 官方 Ratio | L0 |
|---|---:|---:|---:|---:|---|
| s8k | 10893.0 | 5482.1 | −49.7% | 8.73 → **17.17** | PASS |
| s4k | 5449.5 | 2832.3 | −48.0% | 8.76 → **16.57** | PASS |

smoke（`lightning-indexer-s1k-h8-d64`）精度 PASS（`logs/round1/final_Lall.log` 的 L0-smoke-no-scale / L0-smoke-scale，`corr_diff 1.16e-07`）。合并版即 final，`current best` = `perf_opt/_fp8_lightning_indexer_kernel.py`（sha256 `f47a1ded…`）。

### 3.3 Round 2（收束，无新候选）

预算与本轮时间窗（120 min）用尽：本 round 1/1 轮完成 4 个 valid 分支 + 2 个 blocked-compile + 1 个未单独验证配置点，winner 已合并并在两个 tune workload 上同一最终候选复测。剩余候选方向（见 §5）均需改流水结构，不属于本轮可完成范围，如实记录。

## 4. Autotune Log

未启用 autotune 框架。搜索空间为 `(block_Q, block_N) × num_stages × epilogue dtype`，可行性可解析枚举：
- `bQ·Hp·bN·4 B ≤ L0C 128 KB` ⇒ `bQ·bN ≤ 1024`（Hp=32）；UB 约束同理（fp32/fp16 两档分别枚举）；
- 本轮实测覆盖 `(8,64)fp32`、`(8,64)fp16`、`(8,128)fp16`、`(16,64)fp16`、`(8,128/3)fp16`；`(8,256)fp16` 编译失败；`(16,128)`、`(4,256)` 因乘积不增（tile 数不变）或 UB 越限不列入。

## 5. 量化上界与 40% 可达性（交付要求项）

以 s8k 为例（`tile = bQ·bN = 1024` 已被 L0C 128 KB 定死；tile 数 `= S·Skv/1024 = 65536`，48 核 ⇒ 1365 tile/core）：

| 项 | per-tile | 换算到 s8k duration | 对应 Ratio |
|---|---:|---:|---:|
| cube 理想（4.19e6 flop @ 6.23e12 flop/s/core） | 0.67 µs | 919 µs | 103% |
| 向量理想（fp16，3 遍 × 32768 元素 @ 2.30e11/s/core） | 0.43 µs | 582 µs | 163% |
| **串行 pipe 下界（cube+vec 不重叠）** | **1.10 µs** | **1502 µs** | **63%** |
| 40% 目标所需 | 1.74 µs | 2378 µs | 40% |
| 实测 final | 4.01 µs | 5482 µs | 17.17% |

- 剩余 2.9 µs/tile 的构成（§2d 同口径）：AIV scalar 地址/循环、AIV vec 之外的搬运与同步、fixpipe、mte2 时延未被完全隐藏。
- **40% 需要的 1.74 µs/tile = 串行 pipe 理想（1.10 µs）的 63%**，即要求「每 tile 标量/同步开销 ≤ 0.6 µs」。本轮把该开销从 3.85 µs 降到 4.01→（相对 ideal）仍差 3.6×，且已用尽 L0C 限定下所有可行 tile 组合。
- **结论：40% 在本轮预算内不可达，实测 17.17%（1.92×）**。但按其工作计数下界（1502 µs < 2378 µs）**不能断言其在结构上不可能**——达不到的原因是每 tile 的标量/DMA/同步开销（Developer 模式自动同步 + 逐 tile 的 cube↔vector 交接），属实现/流水层；要兑现它需要「一次交接处理多个 kv tile」或持久化 grid 等流水重构，而 L0C 128 KB 已把 `bQ·bN` 夹在 1024（bN>128 编译越限、Hp 分块 = 上一轮 S3 实测回退 50–107%）。故本轮**不产出 `perf_feedback.md`、不回传 `[DESIGN_LIMIT]`**：`_shared/standards/perf-feedback.md` §1 条件①（设计层归因）不成立，工作计数下界低于目标时间即证伪了「设计层天花板」这一断言。

## 6. 工件、契约与偏差

- **语义未变**：本 kernel 仍写「块对齐超集」，窗口由 `clean_logits_` 覆盖；`E1`（scale 在头归约之外）与 `E3`（`has_scale=False` 时整支省略）语义不变。
- **唯一数值变化**：epilogue 计算精度 fp32 → fp16。实测 `corr_diff` 最坏 1.67e-07（L0/L1/L2/boundary 全部 ≤ 1.7e-07，门限 1e-3），`max_abs` 增至 0.31（`L1-bf16-s8k`），来源是 fp16 舍入，不影响 `corr_diff` 判据；`Logits` 输出仍是 fp32。
- **新增编译期约束**：`kv_group > 1` 时 bN 夹到 64（factory 内 Python 常量分支，无运行时分派）；grouped 路径的 `block_Q` 上限（wrapper K10）继续生效；`L2-g4` 仍为 WARN（与本轮前逐字节相同的既有行为，advisory 非阻塞）。
- **tuned 默认值暴露**：`TUNED_BLOCK_N=128`、`TUNED_BLOCK_Q=8`、`TUNED_NUM_STAGES=2`（factory 与 `run_case` 默认同步），供 wrapper 切换块成对引用；wrapper 未改。
- **记录文件**：`perf_opt/perf_records.jsonl`（6 行：baseline×2 + candidate×2 + final×2，字段按 `_shared/standards/signal-registry.md` §5）。本轮早期采集脚本写入的原始行（含一次被 shell 超时中断产生的重复 s4k final 行、两行候选 id 被 shell 引号破坏的配置点）**原样留档**在 `perf_opt/history/perf_records_round1_raw_appended.jsonl`，未删改；gate 对账用的 6 行由同一批测量值重建元数据（`artifact_path`/`artifact_sha256`/`l0_pass`），**duration 数值与留档完全一致**（2832.3 vs 2830.0/2832.3，同一候选的三次采样）。r1d/r1e 两个配置点未单独跑 L0，故不进入 records（仅在 §3.2 表中报告，且均非 winner）。
- 未修改 `{op}.py`（Stage 3 基线）、`DESIGN.md`、`REVIEW.md`、`tests/`、`benchmarks/`、`manifest/`、wrapper。

## Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer_kernel/_fp8_lightning_indexer_kernel.py::_fp8_lightning_indexer_kernel::_fp8_lightning_indexer_main | lightning-indexer-bf16-s8k-h32-d64 | 10893.0 | 5482.1 | perf_opt_final |
| examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer_kernel/_fp8_lightning_indexer_kernel.py::_fp8_lightning_indexer_kernel::_fp8_lightning_indexer_main | lightning-indexer-bf16-s4k-h32-d64 | 5449.5 | 2832.3 | perf_opt_final |

- final 候选文件 = `perf_opt/_fp8_lightning_indexer_kernel.py`，sha256 `f47a1ded6c71cf35e5747f6f1351e87bd9e05266703b2e430af2d63be4a2bf63`；两条 final 记录来自**同一文件、同一配置**（`block_N=128, block_Q=8, num_stages=2`，fp16 epilogue），launch-count=3（s8k 5465.4–5494.3，s4k 2802.4–2870.1）。
- raw profile：`perf_opt/profiles/final/s8k/perf_opt_final/OPPROF_20261101115408_IPRNMAGSRJHRHAMC`、`perf_opt/profiles/final/s4k/perf_opt_final/OPPROF_20261101115619_OGEVRXIUKKRWRLIV`（含 `visualize_data.bin`、`OpBasicInfo*.csv`、`PipeUtilization*.csv`）。
- 提升：s8k **1.99×**（Ratio 8.73→17.17）、s4k **1.92×**（Ratio 8.76→16.57）；对官方口径（9.04/8.99%）同比例换算后 40% 目标完成度 ≈ 43%。
- smoke：`lightning-indexer-smoke-s1k-h8-d64` 精度 PASS（不采性能）；`lightning-indexer-fp8-s8k-h32-d64` skipped（`CG-2026-0022` + shape 去重，未尝试真实 fp8、未伪造）。
- 最终精度回归：`python perf_opt/_fp8_lightning_indexer_kernel.py --level all`（`perf_opt/logs/round1/final_Lall.log`）→ **L0 3/3 PASS、L1 4/4 PASS、L2 2 PASS + `L2-g4` WARN（advisory 非阻塞、既有行为）、boundary 3/3 PASS、`All check passed!`（rc=0）**。

## 7. Skill Retrospective

### 7.1 流程问题

- **R｜Ratio 类 KPI 的任务必须在 Phase 0 先做「指标口径拆解」**：本轮花费约 25 min 才确认 `Ratio = ideal(ops/peak)/duration`、且 `ops` 是硬件计数（dtype 无关）而 `peak` 随 dtype 变。若一开始就知道，R1b（fp16）会被更早排为第一候选。建议 skill Phase 1 对「官方 Ratio / roofline 型目标」增加一步：解析 `visualize_data.bin` 的 `GM Read + Write` 点，打印 `AI / perf / computility / ideal` 四项并写出 `Ratio = ideal/duration` 的恒等式。
- **R｜「跳过冗余工作」对 Ratio 型 KPI 是中性**：本算子 96% 的 kv tile 在窗口外（可跳过且真实时延可降 ~20×），但分子分母同降 ⇒ Ratio 不变。建议 skill 的 bottleneck-patterns 明确「比值型指标（Ratio/利用率）不奖励减少冗余工作量，只奖励提高单位工作量效率」，避免下一轮把预算花在 skip 上。
- **R｜测量脚本必须支持「一次采集同时给出 duration + roofline 拆分」**：本轮自建 `perf_opt/ratio_bench.py`（`--aic-metrics=Roofline` → `OpBasicInfo` + bin + CSV 聚合）后才具备决策能力；此前只有 `tune_runner.py` 的 duration。建议 skill 的 profile-collection 提供该模板。
- **R｜shell 引号曾破坏 records 的 `candidate_id`**：批量循环里用 `--config "$c" --candidate-id $cid` 的字符串切分把 JSON 混入 candidate_id。建议 runner 只接受 `--config-json` 文件或单键参数，避免 shell 切分。

### 7.2 价值点提案（vp_type + 证据三件套）

1. **D｜官方 Ratio 的恒等式与「fp32 epilogue 的 41.7% 天花板」**（vp_type=D）
   - 内容：`Ratio(%) = ideal/duration`，`ideal = total_ops/computility`，`total_ops = 2·S·Skv·H·D + 计数向量 op`，`computility = cube 294.9 + vec(64 fp32 lanes/1.5213 GHz/48 core)` TOPS；本算子计数向量 op 的理想时间（2280 µs，s8k）> cube 理想（932 µs）⇒ fp32 epilogue 下 Ratio ≤ 41.7%，而 fp16 epilogue 把 vec 峰值翻倍、计数不变 ⇒ 上限抬到 ~83%。
   - evidence：§2 三张表；baseline bin（`profiles/baseline/s8k/.../visualize_data.bin`）ratio 8.7305、AI 1499.5、perf 26.11、computility 299.094；final 302.663（fp16 抬高 +3.6 TOPS）。
   - repro：`perf_opt/ratio_bench.py`（同参数重采）+ `perf_opt/analyze_profile.py`。
   - toolchain_stamp：tilelang repo HEAD 2026-11-01 / CANN 9.0.0 / Ascend910B3（1800 MHz 实测）/ torch_npu 2.7.1.post4。
2. **P｜epilogue dtype 决定「向量 op 计数 / 向量峰值」之比，是 Ratio 型 KPI 的一等杠杆**（vp_type=P）
   - 内容：把 L0C→UB 拷贝直接落到 fp16（`T.copy(acc_fp32, acc_fp16_ub)`）、relu/加权/头归约全 fp16，再 cast 回 fp32 输出：向量指令数减半、fixpipe 流量减半、UB 余量翻倍以支持更大的 bN。本算子实测 s4k 5449.5→4648.4（−14.7%），叠加 bN 64→128 后 2835.7（−48.0%）。fp32 时 bN=128 因 `acc_ub` 128 KB 越限不可行。
   - evidence：§3.2 表；`corr_diff` ≤1.7e-07（fp32 为 1.66e-07，几乎无损）。
   - repro：`perf_opt/_fp8_lightning_indexer_kernel_round1_fp16.py` vs `perf_opt/_fp8_lightning_indexer_kernel_round0_base.py`（完整 before/after）。
   - toolchain_stamp：同上。
3. **D｜「乘积固定、换轴」在本族上的实测排序**（vp_type=D）
   - 内容：`bQ·bN=1024` 时 `(8,128) fp16 = 2835.7 µs`、`(16,64) fp16 = 4779.0 µs`、`(8,128)/st3 = 2814.9 µs`、`(8,256) 编译失败`。⇒ 摊薄 tile 数应优先抬高 **bN**（K 侧 tile 在 UB 中复用 8 q 行 × 32 head），抬 bQ 会同步放大 q_ub 与 acc_ub 并降低 K 复用收益。
   - evidence：§3.2 / §4 表。
   - repro：`perf_opt/ratio_bench.py --config '{"block_N":128,"block_Q":8,"num_stages":2}'`。
   - toolchain_stamp：同上。
4. **P｜`T.vadd` 不接受带 Var 偏移的 sub-region**（vp_type=P，反向模式）
   - 内容：想用 log2 树 vadd 取代 `8×T.reduce_sum` 来砍 AIV scalar 开销时，`T.vadd(buf[q*Hp:q*Hp+half, :], ...)` 在 parser 的 evaluator 阶段直接 `TypeError`（`Buffer.__getitem__` 对 Var 取 int）。只有 `T.reduce_sum` 这类特化接口接受 region。⇒ 该前端下「树形归约」只能用整 buffer 表达，而本布局的行块是 strided，故不可行。
   - evidence：r1a compile trace（§3.2 blocked 证据原文）。
   - repro：`perf_opt/_fp8_lightning_indexer_kernel_round1_tree.py`（保留失败形态）。
   - toolchain_stamp：同上。

### 7.3 未解决 / 留给后续

- **每 tile 固定开销 2.9 µs（占 4.01 µs 的 72%）**是达到 40% 的唯一剩余障碍；可行方向：一次 cube↔vector 交接覆盖多个 kv tile（L0C 128 KB 已顶满，需降低 Hp/减少 q_rows 或落 L1）、持久化 grid 摊薄 block 级开销、或让 AIV 侧的地址计算改为整块连续（消除 8 次 per-q `reduce_sum`/`copy` 的 scalar 开销——需前端支持 region vadd 或 3D reduce）。
- `L2-g4`（kv_group=4, heads=32）仍为 advisory WARN（UB 越限），与本轮前一致；wrapper 的 `block_Q` 上限必须继续生效。
- aux `clean_logits_` 本轮未动（`no_gain`），其窗口外写 `-inf` 的代价（s4k 60.2 µs / s8k 138.1 µs，上一轮已优化 921×/802×）仍是端到端的一部分。
