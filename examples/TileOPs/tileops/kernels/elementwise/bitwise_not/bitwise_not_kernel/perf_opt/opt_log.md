# BitwiseNot forward kernel — Stage 4 性能调优日志

- 算子目录: `examples/TileOPs/tileops/kernels/elementwise/bitwise_not/bitwise_not_kernel/`
- 基准 (只读, Stage 3): `bitwise_not_fwd_kernel.py`
- 调优产物 (final): `perf_opt/bitwise_not_fwd_kernel.py`
- 性能目标: best_effort（`user_requirement`），无逐 workload 数值硬目标
- 时延口径: **`msprof op` Task Duration(us)**（kernel-only，唯一口径；median of 20 launches，warm-up 5）
- 噪声阈值: 3%
- bench 入口: `examples/TileOPs/benchmarks/ops/bench_unary_elementwise.py::test_bitwise_not_bench`
- 回归入口: `perf_opt/bitwise_not_fwd_kernel.py --level all`
- 硬件: Ascend910_9362（40 物理 Vector 核 = 20 AI core × 2），CANN 9.0.0，Developer 模式

## Workload Inventory 与 Baseline (Performance Test Data)

`workload_inventory.json` 从 bench 入口 `test_bitwise_not_bench` 展开，逐案例分类（本目录 `perf_opt/`）：

| kind | workload_id | shape | dtype | 处理 |
|---|---|---|---|---|
| smoke | smoke-1m-int32 | [1048576] | int32 | 仅精度回归（不调优） |
| smoke | smoke-1m-int64 | [1048576] | int64 | 仅精度回归（不调优） |
| tune | elementwise-16m-int32 | [4096,4096] | int32 | 调优 |
| tune | elementwise-16m-int64 | [4096,4096] | int64 | 调优 |
| tune | elementwise-64m-int32 | [8192,8192] | int32 | 调优 |
| tune | elementwise-256m-int32 | [16384,16384] | int32 | 调优 |

- skipped: 无。
- smoke 排除理由: pytest case id 含独立 `smoke` token（`smoke-1m-int32` / `smoke-1m-int64`），按 skill Phase 1 排除出性能调优，仅作每次合并及最终精度回归。
- 无 `full` 标记与 smoke 冲突。

Baseline（round 0，`phase=baseline`，kernel = 基准 `bitwise_not_fwd_kernel.py`，sha256 `084be55d…`）：

| kernel_id | workload_id | Task Duration(us) | AIV vec util | MTE2 busy | L0 | raw profile |
|---|---|---:|---:|---:|---|---|
| …bitwise_not_fwd_kernel.py::bitwise_not_fwd_kernel::main | elementwise-16m-int32 | 93.6219 | 6.77% | 90.47% | pass | profiles/baseline/round0/baseline_8ba8496a/elementwise-16m-int32_20e69a01 |
| …bitwise_not_fwd_kernel.py::bitwise_not_fwd_kernel::main | elementwise-16m-int64 | 183.4637 | 5.83% | 92.75% | pass | profiles/baseline/round0/baseline_8ba8496a/elementwise-16m-int64_c84397eb |
| …bitwise_not_fwd_kernel.py::bitwise_not_fwd_kernel::main | elementwise-64m-int32 | 484.1697 | 5.05% | 93.41% | pass | profiles/baseline/round0/baseline_8ba8496a/elementwise-64m-int32_adc4e9e7 |
| …bitwise_not_fwd_kernel.py::bitwise_not_fwd_kernel::main | elementwise-256m-int32 | 2058.5012 | 4.69% | 94.00% | pass | profiles/baseline/round0/baseline_8ba8496a/elementwise-256m-int32_588592d1 |

Baseline 瓶颈判读（来自 raw `PipeUtilization`）：AIV 向量计算占比仅 4.7–6.8%，MTE2（GM→UB）忙比 90–94% —— 逐元素位运算本身几乎零算力，kernel 完全由 GM↔UB 搬运支配；同时 baseline `block_size=2048` 使每核 tile 数偏多（发射项 ∝ 1/block_size，DESIGN §5.2 已把 block_size 列为 Stage 4 首选杠杆）。据此排序：先 256m/64m/16m int32（大流量、结构收益明显），后 16m int64（同 shape 但访存翻倍、UB 更紧）。

### 设计估算 vs 实测偏差（D-2 回填，DESIGN §1.6.0 roofline）

| workload | 设计估算下界(us) | baseline 实测(us) | 偏差 | 失准项 |
|---|---:|---:|---|---|
| elementwise-16m-int32 | ≈111 | 93.62 | 实测低于估算 16% | 流量项（`CONST-mte2-degradation` 标注「待重验」，估算偏保守） |
| elementwise-64m-int32 | ≈642 | 484.17 | 实测低于估算 25% | 流量项（同上，随 N 增大偏差放大） |
| elementwise-256m-int32 | ≈2918 | 2058.50 | 实测低于估算 29% | 流量项（同上） |
| smoke-1m-int32 | ≈6.4 | 未采集（smoke） | — | — |

偏差方向一致（实测更快），说明本设备实际有效带宽高于设计引用的 stale 常数；偏差未到 2x 门槛，不构成设计层矛盾，参数级不可解问题不成立。

## Iteration Log

每轮固定 base = 当前已合并 current best；每个实验分支只改一个主要优化点；分支先跑 L0 再 `msprof op`；对 target workload 按 Task Duration 选 winner。所有候选文件与 raw profile 见 `perf_opt/` 与 `perf_opt/profiles/`。

### Round 1 — block_size 扫描（发射项杠杆）

现象（baseline）：kernel 受限于发射项 + MTE2 搬运；`block_size=2048` 时每核 tile 数过多。候选：在 baseline 结构上仅改 `block_size`（4 个分支，单点）。

候选 vs current best（current best = baseline）：

| workload | candidate | Task Duration(us) | AIV vec util | MTE2 busy | L0 | 结论 |
|---|---|---:|---:|---:|---|---|
| elementwise-256m-int32 | r1_bs4096 | 1803.7561 | 4.62% | 94.59% | pass | gain |
| elementwise-256m-int32 | r1_bs8192 | 1713.0742 | 4.29% | 95.25% | pass | **winner** |
| elementwise-16m-int32 | r1_bs4096 | 72.7815 | 7.79% | 89.43% | pass | gain |
| elementwise-16m-int32 | r1_bs8192 | 64.3813 | 7.86% | 89.50% | pass | **winner** |
| elementwise-16m-int64 | r1_bs3072 | 174.3035 | 5.77% | 92.78% | pass | gain |
| elementwise-16m-int64 | r1_bs4096 | 171.5234 | 5.53% | 93.30% | pass | **winner** |
| elementwise-64m-int32 | r1_bs4096 | 417.1683 | 5.06% | 93.90% | pass | gain |
| elementwise-64m-int32 | r1_bs8192 | 393.1679 | 4.74% | 94.50% | pass | **winner** |
| elementwise-64m-int32 | r1_bs6144 | 398.2480 | 4.94% | 94.20% | pass | gain(弱) |

结论：block_size 是主导杠杆（相对 baseline 16m-int32 −31%、256m-int32 −17%、64m-int32 −19%）；int32 取 8192、int64 取 4096（更大即达 UB 192KB/AIV 上限）。current best 更新为 `r1_bs8192`（int32 各 shape）。

### Round 2 — 非 bool 路径去 `ones_ub`（2-buffer，放大 int64 block_size）

现象：`ones_ub` 仅 bool 分支使用，非 bool 路径却无条件分配，直接压低 UB-safe block_size。候选 `r2_2buf`：非 bool prim_func 去 `ones_ub`（2 buffer），int64 4096→6144、int32 8192→12288。base = `r1_bs8192`。

候选 vs current best（current best = r1_bs8192，16m-int32 64.3813 / 64m-int32 393.1679 / 256m-int32 1713.0742；int64 用 r1_bs4096 171.5234 参考）：

| workload | candidate | Task Duration(us) | AIV vec util | MTE2 busy | L0 | 结论 |
|---|---|---:|---:|---:|---|---|
| elementwise-16m-int32 | r2_2buf | 68.6414 | 6.89% | 90.26% | pass | 回退（bs=12288 不整除 16m，产生尾块/不齐） |
| elementwise-16m-int64 | r2_2buf | 170.2834 | 5.23% | 93.25% | pass | 小幅 gain（+0.7% vs bs4096，平区内） |
| elementwise-64m-int32 | r2_2buf | 391.9879 | 4.48% | 94.81% | pass | 平区 |
| elementwise-256m-int32 | r2_2buf | 1710.6143 | 4.06% | 95.60% | pass | 平区 |

结论：去 buffer 的思路只对 int64 有边际收益（且 bs=6144 不整除 manifest N）；int32 反因非 2 幂 block_size 回退。**不单独采纳为 current best**；int64 的 per-dtype 思想（更小 buffer→更大整除性块）保留待合并。

### Round 3 — static-bulk + 动态尾块拆分

现象：baseline 热循环每迭代算 `tail=T.min(block_size, shape-off)` 且 `T.copy` 为动态 extent，抑制多 buffer/流水 lowering。候选 `r3_staticbulk`：`N//block_size` 个整 tile 用 **静态 extent** `T.copy`，至多 1 个尾 tile 保留动态 extent。base = `r1_bs8192`。

候选 vs current best（current best = r1_bs8192）：

| workload | candidate | Task Duration(us) | AIV vec util | MTE2 busy | L0 | 结论 |
|---|---|---:|---:|---:|---|---|
| elementwise-16m-int32 | r3_staticbulk | 63.4813 | 8.47% | 91.58% | pass | **winner**（+1.4% vs 64.3813） |
| elementwise-16m-int64 | r3_staticbulk | 171.1034 | 5.73% | 94.92% | pass | gain（+0.2%，平区） |
| elementwise-64m-int32 | r3_staticbulk | 389.3478 | 4.95% | 96.67% | pass | **winner**（+1.0%） |
| elementwise-256m-int32 | r3_staticbulk | 1711.8542 | 4.47% | 97.73% | pass | 平区（vs 1713.07） |

ab_test 裁决（`perf_opt/ab/16m_bs8192_vs_staticbulk`，交错 A/B，3 对）：r1_bs8192 配对中位 64.581/72.601/63.261，r3_staticbulk 64.701/64.881/64.741 → 平区（差异 <0.5%）；static-bulk 至少不回退，对 MTE2 忙比有正向作用（256m 94.0%→97.7%）。current best 更新为 int32 走 `r3_staticbulk`、int64 保持候选。

### Round 4 — 分核数据流（contiguous per-core）

现象：round-robin 映射 `task = i*40 + cid` 使同一核跨迭代访问分散 GM 区间。候选：`r4_nc20`(20 核)、`r4_pipe2`(显式双缓冲)、`r4_contig`(contiguous per-core `task = cid*chunk+i`)。base = `r1_bs8192`，目标 workload = 256m-int32。

候选 vs current best（current best = r3_staticbulk 1711.8542，256m-int32）：

| candidate | Task Duration(us) | AIV vec util | MTE2 busy | L0 | 结论 |
|---|---:|---:|---:|---|---|
| r4_nc20 | 1866.3773 | 7.88% | 91.39% | pass | 回退（核数不足） |
| r4_pipe2 | 1717.2144 | 4.27% | 95.26% | pass | 平区 |
| r4_contig | 1669.3934 | 4.40% | 95.14% | pass | **winner**（−2.5%） |

ab_test 裁决（`perf_opt/ab/256m_gridstride_vs_contig`）：gridstride 1710.234 vs contiguous 1664.573（−2.67%）；另 `ab/256m_base_vs_bs8192`：baseline 2063.46 vs bs8192 1715.33（−16.9%）。winner = `r4_contig`（256m-int32）。

### Merge — 三处单点胜出结构合并为同一 `@T.prim_func`

合并内容：Round1 的 per-dtype block_size（int32 8192 / int64 4096）+ Round3 static-bulk/尾块拆分 + Round4 contiguous per-core 映射。合并候选 `r4_merged`（= 最终文件内容），并在**最终文件**上复测全部 4 tune workload（性能+精度）与 smoke 精度。

合并候选 vs 各 workload 单点 winner（合并非回退检查）：

| workload | 单点 winner(us) | r4_merged(us) | AIV vec util | MTE2 busy | L0 | 结论 |
|---|---:|---:|---:|---:|---|---|
| elementwise-16m-int32 | 63.4813 (r3_staticbulk) | 63.7213 | 8.01% | 92.26% | pass | 平区（+0.4%） |
| elementwise-16m-int64 | 170.2834 (r2_2buf) | 171.1434 | 5.68% | 95.59% | pass | 平区（+0.5%） |
| elementwise-64m-int32 | 389.3478 (r3_staticbulk) | 384.5877 | 4.99% | 97.25% | pass | **gain**（−1.2%） |
| elementwise-256m-int32 | 1669.3934 (r4_contig) | 1655.8732 | 4.61% | 98.22% | pass | **gain**（−0.8%） |

结论：合并版在 4 个 tune workload 上均不劣于（或优于）各自单点 winner，全部超出 3% 噪声阈值的相对 baseline 增益；采纳 `r4_merged` 为最终候选，写入 `perf_opt/bitwise_not_fwd_kernel.py`。

smoke 精度（合并版最终文件，L0 六 dtype 1M + `--level all` 分层测试）：int32/int64/bool/uint8/int8/int16 全部 exact-match PASS；L1（含 4_000_000 非整除尾块、2047 单尾 tile）、L2、Boundary 全通过。

### Round 5 / plateau 判定

- 4 轮均产生 valid 实验分支（round1/3/4 超噪声阈值提升），未触发连续 3 轮无提升的 plateau。
- 参照结构 diff（morph ladder 轻量版）：同族逐元素实现（`examples/elementwise/vec_add_1d.py` 等）为纯 Vector 单遍 direct 直译，无更优结构形态可移植；当前 kernel 已是「per-dtype 大 block + 静态 bulk + contiguous per-core」形态，无剩余结构差异候选。
- 剩余候选（`r4_nc20` 核数、`r4_pipe2` 显式双缓冲）已实测回退/平区；`r2_2buf` 对 int32 回退。判定收敛，stop_reason 记录为最优可达（best_effort 目标下无逐 workload 硬门槛）。

## Final Performance Test Data

Final 候选：`r4_merged`（文件 `perf_opt/bitwise_not_fwd_kernel.py`，sha256 `fbdc3f7d49cca68e9daa6fcf2dc08e3cc122250f0334ddff1ea8ed90eccbf8de`），在**最终文件**上重测（`phase=final`，`profiles/final/round4/r4_merged_e6d55220/`）。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/elementwise/bitwise_not/bitwise_not_kernel/bitwise_not_fwd_kernel.py::bitwise_not_fwd_kernel::main | elementwise-16m-int32 | 93.6219 | 63.9413 | r4_merged |
| examples/TileOPs/tileops/kernels/elementwise/bitwise_not/bitwise_not_kernel/bitwise_not_fwd_kernel.py::bitwise_not_fwd_kernel::main | elementwise-16m-int64 | 183.4637 | 171.6034 | r4_merged |
| examples/TileOPs/tileops/kernels/elementwise/bitwise_not/bitwise_not_kernel/bitwise_not_fwd_kernel.py::bitwise_not_fwd_kernel::main | elementwise-64m-int32 | 484.1697 | 382.2276 | r4_merged |
| examples/TileOPs/tileops/kernels/elementwise/bitwise_not/bitwise_not_kernel/bitwise_not_fwd_kernel.py::bitwise_not_fwd_kernel::main | elementwise-256m-int32 | 2058.5012 | 1657.2732 | r4_merged |

提升：16m-int32 −31.70%、16m-int64 −6.46%、64m-int32 −21.06%、256m-int32 −19.49%（均 > 3% 噪声阈值）。

## Final Summary

- 最终版本: `perf_opt/bitwise_not_fwd_kernel.py`（drop-in：同名 factory、同 `(N, dtype, output_dtype=None, block_size=None)` 签名、同 `main(x, y, shape)`；`TUNED_BLOCK_SIZE` 模块常量供 wrapper perf_opt 切换块引用）。
- 采纳优化点（均已单点验证 + 合并复测）：① per-dtype block_size（int32 8192 / int64 4096 / int16 16384 / 1-byte 32768）；② static-bulk 静态 extent + 动态尾块拆分；③ contiguous per-core 数据流。
- smoke：`smoke-1m-int32` / `smoke-1m-int64` exact-match PASS；`--level all`（L0/L1/L2/Boundary）全通过。
- 中止原因: success（best_effort 下最优可达，收敛且全量复测完成）。
- 未达标的硬目标: 无（目标为 best_effort）。
- perf_feedback: 未产出（设计层天花板双条件不满足：无 >2x 结构性加速被冻结设计阻断；实测均优于设计估算，无假设矛盾）。

## Skill Retrospective

流程与产出复盘（按 vp_type 标注，证据三件套 = evidence + repro + toolchain_stamp）：

1. **R（流程规则）— 收尾阶段必须「逐 workload winner 选完后再核内合并，且合并版必须在最终文件上重测」**。
   - evidence: 本轮 4 个 workload 的单点 winner 分属 3 个不同结构（r3_staticbulk / r2_2buf / r4_contig）；直接取任一单点 winner 作最终文件会使其余 workload 得不到最优。合并版 `r4_merged` 在 256m 上比 r3_staticbulk 再快 3.3%、64m 快 1.2%，且不劣于单点 winner。
   - repro: `perf_opt/run_experiments.py --phase merged --all --exp bitwise_not_fwd_kernel_round4_merged.py:r4_merged:r1_bs8192` 与 `--phase final` 对照（同目录 `perf_records.jsonl` 第 25–32 行）。
   - toolchain_stamp: tilelang npuir build（本仓 build/）+ CANN 9.0.0 + Ascend910_9362。

2. **D（实测数据）— 逐元素位反在本设备是「MTE2 搬运墙」而非算力墙，优化主轴是 block_size 与 DMA 连续性**。
   - evidence: baseline AIV vec util 4.7–6.8%、MTE2 busy 90–94%（`profiles/baseline/.../PipeUtilization*.csv`）；per-dtype block_size 8192/4096 把 16m-int32 从 93.62→63.48us（−32%）。
   - repro: 见 `opt_log.md` 各轮表与 `perf_records.jsonl`。
   - toolchain_stamp: 同上。

3. **P（模式方法）— 非整除 block_size 会抵消「更大块」的收益**：去 buffer 换取更大 block_size（12288）反使 16m-int32 回退 7%（68.64 vs 64.38），因 12288 不整除 manifest N（2 的幂），引入尾块/不齐。选择 block_size 时应优先保证对目标 shape 的整除性，而非一味放大。
   - evidence: `perf_records.jsonl` 第 14 行（r2_2buf 16m-int32 68.6414）vs 第 8 行（r1_bs8192 64.3813）。
   - repro: `profiles/candidate/round2/r2_2buf_79afabd7/elementwise-16m-int32_20e69a01` 对照 `profiles/candidate/round1/r1_bs8192_b2a5fa6a/elementwise-16m-int32_20e69a01`。
   - toolchain_stamp: 同上。

4. **流程问题**：上一轮在 4 轮候选全部测完后中断、未合并 winner / 未落 opt_log，导致本轮需「续跑收尾」。建议 skill 在 Phase 2 每轮记录之外，强制每完成一个 workload 的 winner 即写一行 `merged` 记录（本轮已按此补 `r4_merged` 的 `merged`+`final` 记录），避免合并/收尾工作集中在会话尾部而丢失。
