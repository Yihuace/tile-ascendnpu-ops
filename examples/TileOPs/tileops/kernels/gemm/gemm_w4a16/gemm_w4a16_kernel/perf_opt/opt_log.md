# GemmW4A16Op Stage 4 调优日志（`perf_opt/opt_log.md`）

> 算子：`GemmW4A16Op`（TileOPs 集成算子，MixCV：Cube + Vector 反量化 GEMM，W4A16，decode m=1）
> 项目：`examples/TileOPs/tileops/kernels/gemm/gemm_w4a16/gemm_w4a16_kernel/`
> 基线（Stage 3，只读）：`_gemm_w4a16_kernel.py`（factory 默认 `block_M=16, block_N=64, block_K=64, num_stages=2`）
> 定稿：`perf_opt/_gemm_w4a16_kernel.py`（结构不变，默认 tiling 改为 `block_K=128`，见 `TUNED_DEFAULT_CONFIG`）
> 性能口径：**唯一口径 = `msprof op` 的 `Task Duration(us)`**（kernel-only）；`Ratio/Mem/Compute/BW Util` 由同一次 capture 反解（`summarize.py`，公式见下）
> 工具链：tilelang 0.1.2+（dev root build）/ CANN 8.5.2 / Ascend910B3（`NPUUtils.get().get_aicore_num()=20`）/ 2026-10-16

## 0. 口径与工件

| 指标 | 定义 | 来源 |
|------|------|------|
| Task Duration (us) | kernel-only 时延，`--launch-count=20 --warm-up=5` 的中位数 | `msprof op`，`OpBasicInfo.csv` |
| Ratio (%) | msprof `GM/L2 → GM Read + Write` roofline 的 `ratio` × 100 | `visualize_data.bin` |
| Mem Roofline (%) | `perf / (AI × BW)` × 100（msprof 内存项） | 同上 |
| Compute Roofline (%) | `perf / computility` × 100（msprof 算力项） | 同上 |
| Achieved BW Util (%) | `declared_bytes / duration / 1.8 TB/s` × 100，declared_bytes = manifest roofline 公式（read+write） | `benchmark_base._build_result` 口径 |

工件：`harness.py`（msprof 驱动）、`prof_entry.py`（被测进程）、`run_experiments.py`（T-4 批量 runner）、`summarize.py`（4 指标）、`diagnose.py`（pipe 级诊断）、`gen_branch.py` + `make_final.py`（分支/定稿生成）、`profiles/`（raw）、`logs/`（stdout/stderr）、`perf_records.jsonl`（append-only）。

## 1. Performance Test Data

### 1.1 workload 清单（自 `benchmarks/ops/bench_gemm_w4a16.py` 展开）

| workload | kind | (m,n,k) | dtype | 说明 |
|---|---|---|---|---|
| compile-smoke-square-64x64x128 | smoke | 64,64,128 | float16 | 仅精度（L0-1） |
| compile-smoke-rect-128x256x256 | smoke | 128,256,256 | float16 | 仅精度（L0-2） |
| decode-l2-resident-ish | **tune** | 1,8192,8192 | float16 | L2 驻留边界 |
| decode-hbm-streaming-threshold | **tune** | 1,8192,16384 | float16 | HBM streaming 阈值 |
| decode-non-power2-low-cta | **tune** | 1,7168,20480 | float16 | 非 2 幂 N + 低 CTA |
| decode-long-k-pressure | **tune** | 1,8192,81920 | float16 | 长 K 压力（K/128=640 组） |

无 skip 项（manifest 6/6 全部可运行）。smoke 只做精度回归，不做性能采集；n=1 类（无）不存在。

### 1.2 baseline（Stage 3 默认 config = 16,64,64,2；`perf_records.jsonl` round 0）

| workload | Task Duration(us) | Ratio% | Mem% | Compute% | BW Util% | raw |
|---|---:|---:|---:|---:|---:|---|
| decode-l2-resident-ish | 5807.06 | 0.5678 | 0.0890 | 0.5679 | 0.3464 | `profiles/baseline/baseline/decode-l2-resident-ish/` |
| decode-hbm-streaming-threshold | 11219.79 | 0.5878 | 0.0922 | 0.5878 | 0.3585 | `profiles/baseline/baseline/decode-hbm-streaming-threshold/` |
| decode-non-power2-low-cta | 11994.23 | 0.6014 | 0.0943 | 0.6014 | 0.3668 | `profiles/baseline/baseline/decode-non-power2-low-cta/` |
| decode-long-k-pressure | 55224.97 | 0.5971 | 0.0941 | 0.5971 | 0.3641 | `profiles/baseline/baseline/decode-long-k-pressure/` |

**排序理由**（Phase 1：按实测性能负担排序，先大后小）：`decode-long-k-pressure` 绝对时延最大（55.2 ms，K 迭代 1280 次）且与 `decode-hbm` 同为 N=8192，先取其定 tiling 方向；`decode-l2-resident-ish` 形状最小、迭代最快，作为**筛选探针**（每个候选先在此测得方向，再回填长 K）；`decode-non-power2-low-cta`（N=7168，112 块）留作最终验证时的低 CTA 代表。

### 1.3 基线与初始瓶颈画像（`diagnose.py`，per block 中位数）

| candidate | blocks | AIV wall(µs) | AIV util% | vec(µs) | scalar(µs) | mte2(µs) | AIC util% | cube(µs) | L2 read-miss(MB)/launch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (16,64,64,2) | 128 | 828.8 | **111.9** | 632.3 | 179.2 | 94.1 | 10.6 | 4.20 | 9.05 |
| baseline `/decode-long-k` | 128 | 7890.1 | 112.6 | 6322.5 | 1782.9 | 560.1 | 10.6 | 41.96 | 90.52 |

**结论（首轮现象）**：AIV 侧 pipe（vec+scalar+mte2+mte3）合计 ≈112% of wall ⇒ **Vector 单元完全饱和**；AIC 侧全部 pipe 合计仅 ≈10.6% of wall，`cube` 仅 4.2 µs/828 µs ⇒ **Cube 99.5% 空闲**。`vector0`/`vector1` 两个子块的 vec/scalar/mte2 时间**逐位相同** ⇒ Developer MixCV 的 Vector 程序在双 AIV 上重复执行（PL-1.13 判据）。因此瓶颈是「反量化 Vector 链的执行时间」，不是 Cube、不是 GM 带宽（L2 read-miss 仅 9.05 MB/launch = declared 36.2 MB 的 25%，对应 DRAM 地板 ≈5 µs）。

### 1.4 设计估算 vs 实测偏差（D-2 回填）

| 项 | DESIGN 估算（§1.6.0 D-2） | 实测 baseline | 偏差 |
|---|---|---:|---:|
| `decode-long-k-pressure` 流量地板 | ≈287 µs（361.9 MB @1.26 TB/s；乐观 209 µs） | 55224.97 µs | **×192** |
| `decode-l2-resident-ish` 等比折算 | ≈28.7 µs | 5807.06 µs | **×202** |
| 发射项 | 「未实测假设」（引 `PL-1.6-copy-floor` 认为反量化链可被 MTE2 窗口吸收） | Vector 链占 wall 76%（vec 632.3/828.8 µs），MTE2 仅 11% | 假设被推翻 |
| 算力项 | `block_M=16` 填充下 ≈72–107 µs（300/200 TFLOPS） | cube 4.2 µs/block × 6.4 波 = 26.9 µs | 同量级（非主导项，符合） |

失准项**明确归因**：DESIGN §1.6.0 D-2 把反量化链列为「发射项未实测 + 估算依据」，认为流水化后向量 pass 可被 MTE2 窗口吸收（反证引 `PL-1.6-copy-floor`）。实测该反证不适用于本形态：本形态每个 k-tile 有 **16 个宽向量 op 串行依赖**（依赖链而非独立 pass 链），AIV 吞吐被链路长度占据，且双 AIV 重复执行使有效吞吐再折半。此为 §1 的 `[DESIGN_LIMIT]` 依据（见 `perf_feedback.md`）。

## 2. Iteration Log

每轮固定 base = 上一轮 current best；分支从同一 base 派生、每个分支只改一个主要优化点；L0 精度回归通过后才采性能；每次测量后向 `perf_records.jsonl` 追加一行（append-only）。

### Round 1 — tiling 参数轴（8 分支，筛选 workload = `decode-l2-resident-ish`）

**现象**：AIV 饱和 + AIC 空闲 ⇒ 先问「是否存在把单位元素开销摊薄的 tiling 参数」。`block_K=64` 时 packed tile 的行宽仅 32 B，向量 op 的行粒度偏窄；`block_K=128` 让一个 k-tile 恰好覆盖一个 affine group（`group_size=128`），行宽翻倍且组下标 `g` 退化为 `kk`（省去每迭代整除）。

**本轮候选 vs current best（current best = baseline 16,64,64,2 / 5807.06 µs）**

| candidate | bm,bn,bk,ns | Task Duration(µs) | Δ | Ratio% | AIV util% | AIC util% | vec(µs) | scalar(µs) | mte2(µs) | L2read MB | L0 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| baseline (best) | 16,64,64,2 | 5807.06 | — | 0.5678 | 111.9 | 10.6 | 632.3 | 179.2 | 94.1 | 9.05 | PASS | current best |
| r1_bk128_ns2 | 16,64,128,2 | **4132.70** | **−28.8%** | 0.7185 | 110.4 | 9.4 | 492.3 | 95.7 | 47.7 | 9.05 | PASS | ✅ winner |
| r1_bk128_ns3 | 16,64,128,3 | 4141.38 | −28.7% | 0.7174 | 110.3 | 9.6 | 492.3 | 96.2 | 49.9 | 9.05 | PASS | tie（stage 无效） |
| r1_bk128_ns4 | 16,64,128,4 | 4136.49 | −28.8% | 0.7177 | 110.3 | 9.4 | 492.3 | 95.7 | 49.1 | 9.05 | PASS | tie（stage 无效） |
| r1_bk64_ns3 | 16,64,64,3 | 5820.55 | +0.2% | 0.5665 | 112.0 | 12.2 | 632.3 | 179.9 | 94.8 | 9.05 | PASS | config_no_gain（stage 轴无效） |
| r1_bn96_bk64_ns2 | 16,96,64,2 | 6125.15 | +5.5% | 0.5422 | 109.4 | 8.1 | 928.2 | 242.0 | 146.2 | 9.05 | PASS | ✗ 回退 |
| r1_bn128_bk64_ns2 | 16,128,64,2 | 6420.22 | +10.6% | 0.5132 | 108.8 | 6.8 | 1234.9 | 309.7 | 174.7 | 9.05 | PASS | ✗ 回退（波量化：64 块/20 核=3.2 波→4 波） |
| r1_bn128_bk128_ns2 | 16,128,128,2 | — | — | — | — | — | — | — | — | — | n/a | ✗ 编译失败（ub overflow，需 360 KiB > 192 KiB） |
| r1_bm32_bk128_ns2 | 32,64,128,2 | 4144.49 | −28.6% | **0.8905** | 110.6 | 10.2 | 491.8 | 96.6 | 50.9 | 9.05 | PASS | 时长 tie、Ratio 更高（见 §4 注） |

**config_no_gain 范围**：`num_stages ∈ {3,4}`（对 64 与 128 两种 block_K 均无增益，Δ<0.3%）⇒ stage 轴整类无效，后续轮次固定 `num_stages=2`。
**family_no_gain 范围**：`block_N > 64`（96/128/208 均回退：波量化 + UB 压力导致 per-block 超线性代价）⇒ block_N 轴收敛于 64。
**valid 分支**：`r1_bk128_ns2`。

**本轮合并**：winner 改动＝factory 默认 `block_K 64→128`（单点，无结构变更），直接合入同一 `@T.prim_func`；合并后复测本 kernel 全部 tune workload（见 Round 3 结尾的全量复测）与 smoke 精度（L0 7/7 PASS）。

### Round 2 — 削减反量化链 op 数（2 分支 + 2 复测）

**现象**：Round 1 后 AIV 仍 110% 饱和（vec 492 µs / wall 589 µs），瓶颈定义不变但绝对量已下降；链路长度 = 每 packed 字节 11 op（cast-only recipe，DESIGN §1.6.2）。DESIGN §9.1.1 末条「工具链修复后位运算路径可作为 Stage 4 候选」给出方向：用 `T.npuir_and` / `T.npuir_shr`（全形 int16 操作数）把 11 op 压到 6–8 op（`lo = b & 15`、`hi = (b>>4) & 15`；因 npuir 把 uint8 lower 成 i8 而 256 ≡ 0 (mod 16)，低 4 位无损）。

**本轮候选 vs current best（current best = r1 winner 16,64,128,2 / 4132.70 µs）**

| candidate | module/factory | Task Duration(µs) | Δ | Ratio% | AIV util% | AIC util% | vec(µs) | scalar(µs) | mte2(µs) | L0 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| r1_bk128_ns2 (best) | baseline structure | 4132.70 | — | 0.7185 | 110.4 | 9.4 | 492.3 | 95.7 | 47.7 | PASS | current best |
| r2_vand_bk128 | `_exp_vand.py::k_vand`（8 op 解包） | 4942.11 | **+19.6%** | 0.6092 | 105.1 | 8.4 | 527.8 | 144.0 | 54.2 | PASS | ✗ 回退 |
| r2_vand_bk128_k4 | 同上 /`decode-long-k` | 47572.48 | +19.3% | 0.6328 | 105.4 | 7.9 | 5277.2 | 1432.8 | 319.7 | PASS | ✗ 回退（同向） |
| r2_vshr_bk128 | `_exp_vshr.py::k_vshr`（6 op 全位运算） | — | — | — | — | — | — | — | — | **FAIL** | ✗ blocked（aivector **timeout 507014**） |
| r2_vshr_bk128_k4 | 同上 | — | — | — | — | — | — | — | — | n/a | ✗ 编译/运行失败同因 |

**config_no_gain / blocked 说明**：`k_vshr` 的 L0 在 568 s 后以 device 侧 `aicore/aivector timeout (507014)` 终止（`logs/l0/_exp_vshr.log`）——位运算 `int16 vshr + vand` 链在本工具链上**编译通过但运行期挂死**，与 CG-2026-0025（f16 `npuir_floordiv` timeout）同族；已登记 capability gap（见 §5）。`k_vand` 可运行但比 cast-only recipe 慢 19.6%（多一个 int16 mask 缓冲的读 + i16→f16 cast），故整类「位运算/更少 op」路径在本工具链被判**无效**（不是「没试」）。

**本轮 winner**：无（best 保持 `r1_bk128_ns2`）。

### Round 3 — 反量化域/缓冲布局重排（2 分支 + 3 复测，含归因）

**现象**：Round 2 的 `k_vand` 同时改了「解包」与「反量化域」两个点（违反单点原则），需拆分归因；同时把反量化从 `[block_N, block_K]` 域下移到 packed 域可让 `vbrc/sub/mul` 的元素数减半（21→19 单位）。

**本轮候选 vs current best（current best = r1 winner / case1 4132.70 µs、case4 39896.34 µs）**

| candidate | 单点改动 | wid | Task Duration(µs) | Δ | Ratio% | AIV util% | vec(µs) | scalar(µs) | L0 | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| r1_bk128_ns2 (best) | — | l2-resident | 4132.70 | — | 0.7185 | 110.4 | 492.3 | 95.7 | PASS | current best |
| r1_bk128_ns2 (best) | — | long-k | 39896.34 | — | 0.7441 | 110.8 | 4923.0 | 950.6 | PASS | current best |
| r3_v2_bk128 | packed 域反量化（vbrc/sub/mul 减半宽） | l2-resident | 4883.40 | **+18.2%** | 0.6338 | 106.6 | 534.8 | 145.1 | PASS | ✗ 回退 |
| r3_v2_bk128_k4 | 同上 | long-k | 47442.02 | +18.9% | 0.6520 | 106.8 | 5347.6 | 1441.1 | PASS | ✗ 回退（同向） |

**归因结论**：Round 2 的回退**来自 `v2`（反量化域下移）而非 `vand`**——减少「元素数」但增加「指令数」（6 条半宽 vs 4 条全宽）在本工具链上是净亏；`k_vand` 单点本身近似中性（4942 vs 4883，Δ≈1%，噪声内）。⇒ 该工具链上**衡量向量链代价的是指令条数（行×op），不是元素数**；任何"更细粒度更多条"的改写都会变慢。

**本轮 winner**：无（best 保持 `r1_bk128_ns2`）。

### 中止判定

- **plateau**：Round 2、3 共 4 个 valid 分支全部为负（≥18% 回退），且 Round 1 已把 `block_K / block_N / num_stages / block_M` 四个参数轴穷举（best 对邻域不敏感：bk∈{64,128} 已覆盖合法上限；ns 无效；bn>64 回退；bm=16 最小合法值）。
- **参照结构 diff 检查（T-1，plateau 前置）**：同族参照 = `examples/TileOPs/tileops/kernels/moe/moe_grouped_gemm_nopad/`（Expert NT GEMM，**无** INT4 反量化前置链，不可移植：本算子的 Vector 链在参照中不存在）、`examples/mixcv/mixcv_mixkernel.py`（Expert 双 Scope + subid 分片，见下）、`pattern-library/cases.md::CASE-ssd-chunkscan-migration`（MixCV Expert，PL-1.13 subid 分片 −34%）。**唯一识别出的结构差异 = 双 AIV 重复执行**（Round 4 探针已试并被实现层否定，见下）与「Expert 双 Scope + GM 中转」（Design §1.6.3 备选 L2'，其代价是 +2.68 GB GM 往返 = 8.4× 流量，DESIGN 已量化淘汰）。
- **blocked / 结构探测（Round 4 探针）**：按 PL-1.13 判据尝试 Developer 侧 subid 分片（`k_split`：每个 AIV 处理 tile 的一半 k 域，`interleave` 写入 `W_sh[:, subid*half:...]` 的 partial dst）。L0 精度 **FAIL（98.4% NaN）**——partial-dst `interleave` 的**动态**起始偏移未被后端正确落到目标子区域（同时 `T.copy` 的 region 语义/动态 subview 属已知陷阱面：`TRAP-T-copy-region-semantics`、`TRAP-UB-dynsubview-dominance`）。⇒ Developer 侧 AIV 分片无可用落地形态；Expert 双 Scope 形态不在本任务授权范围内（用户指定 Developer）。探针记录：`_exp_split.py`、`logs/l0/_exp_split.log`（正式日志见 `logs/`；探针未产生性能记录）。

**stop_reason = plateau**（3 轮、10 个 valid 分支、2 类结构性方向被否 + 1 类结构探测 blocked；主要参数轴已穷举）。

## 3. Final Summary

### 3.1 逐 workload 对账（baseline → final，同一最终候选 `final_bk128`，`phase=final`，launch-count=20）

| kernel_id + workload_id | baseline(µs) | final(µs) | 提升 | 基线 Ratio% | 最终 Ratio% | raw（同最终候选） |
|---|---:|---:|---:|---:|---:|---|
| …::main + decode-l2-resident-ish | 5807.06 | **4132.50** | **1.405×** | 0.5678 | 0.7185 | `profiles/final/final_bk128_rerun/decode-l2-resident-ish/` |
| …::main + decode-hbm-streaming-threshold | 11219.79 | **8087.28** | **1.387×** | 0.5878 | 0.7342 | `profiles/final/final_bk128_rerun/decode-hbm-streaming-threshold/` |
| …::main + decode-non-power2-low-cta | 11994.23 | **8644.35** | **1.388×** | 0.6014 | 0.7513 | `profiles/final/final_bk128_rerun/decode-non-power2-low-cta/` |
| …::main + decode-long-k-pressure | 55224.97 | **39879.00** | **1.385×** | 0.5971 | 0.7445 | `profiles/final/final_bk128_rerun/decode-long-k-pressure/` |

### Final Performance Test Data

（gate 对账表：`kernel_id` / `workload_id` / baseline 时延 / final 时延 / 最终候选；数值与 `perf_records.jsonl` 的 `phase=final` 同一候选记录逐项一致，容差 1%）

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/gemm/gemm_w4a16/gemm_w4a16_kernel/_gemm_w4a16_kernel.py::_gemm_w4a16_kernel::main | decode-l2-resident-ish | 5807.06 | 4132.50 | final_bk128_rerun |
| tileops/kernels/gemm/gemm_w4a16/gemm_w4a16_kernel/_gemm_w4a16_kernel.py::_gemm_w4a16_kernel::main | decode-hbm-streaming-threshold | 11219.79 | 8087.28 | final_bk128_rerun |
| tileops/kernels/gemm/gemm_w4a16/gemm_w4a16_kernel/_gemm_w4a16_kernel.py::_gemm_w4a16_kernel::main | decode-non-power2-low-cta | 11994.23 | 8644.35 | final_bk128_rerun |
| tileops/kernels/gemm/gemm_w4a16/gemm_w4a16_kernel/_gemm_w4a16_kernel.py::_gemm_w4a16_kernel::main | decode-long-k-pressure | 55224.97 | 39879.00 | final_bk128_rerun |

### 3.2 最终 4 指标（新口径）

| case | dtype | Ratio (%) | Mem Roofline (%) | Compute Roofline (%) | Achieved BW Util (%) | ≥60%? |
|---|---|---:|---:|---:|---:|---|
| decode-l2-resident-ish | float16 | 0.7185 | 0.1252 | 0.7185 | 0.4868 | ✗ |
| decode-hbm-streaming-threshold | float16 | 0.7342 | 0.1279 | 0.7342 | 0.4974 | ✗ |
| decode-non-power2-low-cta | float16 | 0.7513 | 0.1309 | 0.7513 | 0.5089 | ✗ |
| decode-long-k-pressure | float16 | 0.7445 | 0.1310 | 0.7445 | 0.5042 | ✗ |

未达标原因是结构性的（见 §4 与 `perf_feedback.md`），不是参数可解。

### 3.3 精度

| 项 | 结果 |
|---|---|
| L0（7 例，final 文件） | **7/7 PASS**，worst max_diff 1.250e-01（容差 atol 7e-2 / rtol 5e-2） |
| L1（16 例） | **16/16 PASS** |
| L2（3 例，非阻塞） | 3/3 PASS |
| Boundary（9 例，非阻塞） | **8/9 PASS**，B-3（退化 1×1 输出 tile）为 baseline 已记录的 device 限制，非本轮引入 |
| smoke 性能目标 | 不适用（smoke 只做精度） |

### 3.4 结论

- 最终候选：`perf_opt/_gemm_w4a16_kernel.py`，`TUNED_DEFAULT_CONFIG = {block_m:16, block_n:64, block_k:128, num_stages:2}`，结构 = Stage 3 baseline（未改一行 kernel 体）。
- 全部 tune workload 同向 **−27.8% ~ −28.8%**（1.385–1.405×），smoke 精度保持；`final_bk128` / `final_bk128_rerun` 两次独立 20-launch 采集复现性 ≤0.3%（见 §3.5）。
- 记录规范化：`perf_records.jsonl` 的 `phase` 字段有两处机械修正——① 实验轮的 `probe2/r1sweep/r2/r3` 归一为 schema 允许值 `candidate`（只改枚举标签，测量数值/工件哈希/时间戳未动）；② 首轮 final 记录（4 行）因采集后 `TUNED_DEFAULT_CONFIG` 键名调整导致 `artifact_sha256` 与冻结终稿不符，按门禁要求以 `final_bk128_rerun` 在同一冻结文件上重测替换（原文件全文留档 `logs/perf_records_archive_20261016T1100Z.jsonl`）。全部 26 行原始记录可从该留档回溯。
- `perf_records.jsonl`：基线与最终记录均来自各自同一候选；最终 4 行 `phase=final`、`candidate_id=final_bk128`。
- 中止原因：**plateau**。目标（每 case Ratio ≥ 60%）**未达成**，判定为**设计层天花板**，产出 `perf_feedback.md` + `[DESIGN_LIMIT]`（非阻塞）。

### 3.5 复现性与记录规范化

| 项 | 值 |
|---|---|
| 冻结终稿 sha256（= final 记录 artifact_sha256） | `215e268d1d0ec437…`（`perf_opt/_gemm_w4a16_kernel.py`） |
| 两次独立 20-launch 采集差异（case1/2/3/4） | 0.29% / 0.10% / 0.01% / 0.02%（`final_bk128` vs `final_bk128_rerun`） |
| 记录留档 | `logs/perf_records_archive_20261016T1100Z.jsonl`（26 行原始记录，改前全文） |
| 修正 1 | `phase` 枚举：`probe2`(1) / `r1sweep`(7) / `r2`(2) / `r3`(4) → `candidate`（仅枚举标签，未动任何测量字段） |
| 修正 2 | 首次 final 的 4 行（artifact_sha256 `424d41a42664…`）在采集后文件被重新生成 ⇒ 哈希失配，按门禁「final 必须恰有一条且哈希一致」在同一冻结文件上重测为 `final_bk128_rerun` |

## 4. 未达标原因（供优化报告直接使用）

1. **这个算子把 95% 以上的时间花在"把权重从 4 bit 解包/反量化成 fp16"上，而这件事只能由向量单元做。** 实测（case1 baseline）：AIV 侧流水（向量+标量+搬运）合计是墙钟时间的 112%（满载），而 Cube 侧只用了 10.6%，其中矩阵乘本身仅 4.2 µs / 828 µs。也就是说矩阵单元几乎全程空闲，时间全在"准备工作"上。
2. **msprof 的 Ratio 是"实测算力 ÷ 平台算力峰值"，而这个平台的算力峰值是按矩阵单元定的（≈298 TOps/s），向量单元的峰值只有 ≈3.0 TOps/s（约 1%）。** 本算子每个输出元素平均要做 ~5.5 个向量元素操作（11 个 op/packed 字节），这些操作即使把向量单元跑到 100%，也要 ≈2525 µs——此时 Ratio 上限只有 ≈1.3%。换句话说：**只要反量化在向量单元上，Ratio 就不可能接近 60%，与 tiling/流水/分核无关。**
3. **把反量化搬到矩阵单元在当前契约下不可行。** 矩阵单元只做 fp16×fp16 矩阵乘；activation 契约固定是 fp16，因此不存在"用 int8 矩阵乘顺带完成解包"的路径（DESIGN §1.6.0 候选 3 已按语义违反淘汰）。唯一能显著抬高 Ratio 的做法是把权重先物化到 GM 再普通 GEMM，但那会让访存从 0.5 B/元素涨到 2.5 B/元素（+2.68 GB，Design §1.6.0 候选 1 量化淘汰）。
4. **另外两个理论上的结构杠杆都被实测否掉**：(a) 每个 block 的向量程序在两个 AIV 上**重复执行**（profile 判据：两子块时间逐位相同），若能按 subid 分片理论上可接近 2× —— 但 Developer 侧"部分写入 GEMM 操作数"的形态实测精度失败（NaN），无法落地；(b) 更长行宽（block_K>128）会打破"一个 k-tile 一个 affine group"的不变式，且 UB 容量不允许。
5. **Ratio 指标本身对本形态有系统性偏差**：它按"矩阵单元峰值"衡量全部算力，而本算子的算力有 ~3.6× 花在矩阵单元上（实测 cube_fops）……实际上更极端：实测向量 op 数约为矩阵乘 op 数的 3.6 倍，却被计入同一个 298 TOps/s 的分母。因此本算子的 Ratio 天花板 ≈1.3%，60% 目标在当前硬件/工具链/契约下不可达。

## 5. Skill Retrospective

### 5.1 skill 流程问题与建议

| # | 现象 | 建议（proposal） |
|---|---|---|
| 1 | T-4 批量 runner 有效（单轮 4–5 个实验一次跑完、自动追加 records），但**单次 msprof op 实验成本 ≈6–7 min**（含编译），30 个实验 ≈3.5 h；对「先筛选后验证」的双层策略友好，对「每轮全量 workload 复测」代价高 | `tilelang-op-optimize` 的 T-4 模板建议增加「筛选 workload + 回填 workload」两段式字段（本轮已人工实现：`decode-l2-resident-ish` 筛选、`decode-long-k-pressure` 回填） |
| 2 | Round 2 的 `k_vand` 分支**一次改了两个点**（解包 + 反量化域），导致归因需要 Round 3 补一轮 | 无（skill 已明文要求单点；本轮为执行失误，已在 Round 3 补归因） |
| 3 | 后台 `nohup ... &` 在本工具环境下会被 shell 超时连带杀死 / 残留进程并发占用 NPU（实测一次并发导致 msprof 挂起） | 建议 skill 的 profile-collection 增加「NPU 串行纪律」提示：同一时刻只允许一个 msprof 作业；长批次用前台调用 + 大 timeout，而不是后台 + 轮询 |
| 4 | 位运算解包（DESIGN §9.1.1 明确登记的「工具链修复后可作为 Stage 4 候选」）在本工具链上：`vand` 可运行但中性偏负，`vshr` 编译通过而**运行期 aivector timeout** | 该负结论已回写 pattern-library（traps-compiler）+ capability-gaps；建议 DESIGN 期不再把该路径列为"待工具链修复的收益项"，除非工具链版本戳变化 |

### 5.2 价值点提案（带 vp_type 与证据三件套）

| id | vp_type | 内容 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-S4-GEMMW4A16-1 | **P**（优化模式，正向） | W4A16 融合反量化 GEMM 的 tiling 主杠杆是 `block_K`：让一个 k-tile 恰好覆盖一个 affine group（`block_K = group_size`），使向量行宽翻倍、组下标退化为常量、每迭代 mte2 事务减半 —— 实测 −28% （case1: vec 632→492 µs、scalar 179→96 µs、mte2 94→48 µs） | `perf_records.jsonl` r0 baseline / r1_bk128_ns2；`diagnose.py` 表；`profiles/{baseline,r1sweep}/` | `repro/BP-vector-chain-instruction-bound.py`（本任务回写） | tilelang 0.1.2+dev root / CANN 8.5.2 / 910B3 / 2026-10-16 |
| VP-S4-GEMMW4A16-2 | **P**（优化模式，反向/否决） | 「更少 op 的位运算解包」（`vand`/`vshr` 全形 int16 缓冲，8/6 op vs cast-only 11 op）在本工具链上**不产生收益**：`vand` 版本 +19.6% 回退（多一条 mask 读 + i16→f16 cast），`vshr` 版本运行期 `aivector timeout (507014)` | `perf_records.jsonl` r2_vand_bk128；`logs/l0/_exp_vshr.log`；`_exp_vand.py` / `_exp_vshr.py` | `repro/BP-vector-chain-instruction-bound.py`（同文件 SLOW/FAST 对照 §2；位运算否决见 CG-2026-0036） | 同上 |
| VP-S4-GEMMW4A16-3 | **P**（反向） | 「反量化域下移到 packed 域（`vbrc/sub/mul` 元素数减半）」在本工具链上是**净亏**（+18%）：衡量向量链代价的是**指令条数**（行×op）而非元素数；6 条半宽 op 慢于 4 条全宽 op | `perf_records.jsonl` r3_v2_bk128 / r3_v2_bk128_k4；`diagnose.py`（vec 534.8 vs 492.3 µs） | `repro/BP-vector-chain-instruction-bound.py`（同文件 §1 见解摘要） | 同上 |
| VP-S4-GEMMW4A16-4 | **D**（实测数据） | m=1 decode 形态 W4A16 的**真实瓶颈标定**：AIV 满载（pipe 合计 112% wall）、AIC 9–11%；L2 read-miss 9.05 MB/launch（case1）= declared 36.2 MB 的 25% ⇒ DRAM 地板 ≈5 µs，而 kernel 实测 4144 µs（快 800×）⇒ 该算子**与带宽无关**，纯向量执行受限；msprof Vector roofline 峰值 3.04 TOps/s（MISC 权重 42.25 ppc × 1.8 GHz × 40 AIV）仅为 Cube 峰值（294.9 TOps/s）的 1% | `diagnose.py` 表；`profiles/baseline/*`、`profiles/final/final_bk128/*` | `repro/BP-vector-chain-instruction-bound.py`（口径复算骨架；数据见 `diagnose.py` 输出） | 同上 |
| VP-S4-GEMMW4A16-5 | **C**（案例索引） | GemmW4A16Op（TileOPs 集成，MixCV Developer）Stage 4 档案：3 轮 10 分支、唯一胜出 = `block_K=128`（1.385–1.401×）；`Ratio` 天花板 ≈1.3%（结构性，[DESIGN_LIMIT]）；参考骨架 = 本目录 `opt_log.md` + `perf_feedback.md` | 本目录全部工件 | `pattern-library/INDEX.md` + `pattern-library/cases.md` 的 `CASE-gemm-w4a16-stage4` 条目（本任务回写） | 同上 |

回写位置（ED-C，本任务已执行）：① `references/bottleneck-patterns.md` → `BP_ub_traffic_floor` 两条 update〔2026-10-16〕（"向量链代价 ∝ 指令条数（行×op）" + "算力 roofline 的单元不对称"）；② `pattern-library/cases.md` + `INDEX.md` → 新增 `CASE-gemm-w4a16-stage4`；③ `pattern-library/repro/BP-vector-chain-instruction-bound.py`（delta 骨架，py_compile 自检通过）。**BP_ 提案**：`BP_vector_chain_instruction_bound`（向量链代价 ∝ 指令条数×行数，而非元素数；推论：优先"更宽的行"而非"更少的元素"，且任何"拆细成更多条"的重写均回退）——建议 evolver 据此在 bottleneck-patterns.md 建独立 BP 条目。

### 5.3 capability-gap 登记（AGENTS.md 强制）

| gap | 层 | 现象 | 处置 |
|---|---|---|---|
| int16 向量位运算链不可生产使用 | BishengIR / 后端 codegen | `T.npuir_shr(int16, buf4, round=False)` + `T.npuir_and` 链：编译通过，运行期 device 侧 `aicore/aivector timeout (507014)`（case L0-？，568 s）；`T.npuir_and` 单独可运行但相对 cast-only recipe 无收益 | 追加至 `.agents/evolution/capability-gaps.md` → **CG-2026-0036**（与 CG-2026-0019/0025 同族；occurrences=1） |
| Developer MixCV 的 Vector 程序在双 AIV 上重复执行，且无 subid 分片的可用落地形态 | Frontend/IR（CV 拆分） | profile 判据：`vector0`/`vector1` 的 vec/scalar/mte2 时间逐位相同 ⇒ 有效向量吞吐折半；Developer 侧「partial-dst interleave 写入 GEMM 操作数子区域（动态偏移）」精度失败（98.4% NaN） | 追加至 `.agents/evolution/capability-gaps.md` → **CG-2026-0037**（Developer CV 拆分不暴露 subid 划分；occurrences=1；PL-1.13 仅在 Expert 双 Scope 下验证过） |
