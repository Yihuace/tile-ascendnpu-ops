# Stage 4 调优日志 — _argreduce_kernel（argmax / ArgmaxFwdOp）

- project: argmax ; operator: `_argreduce_kernel`；kernel_id: `_argreduce_kernel::main`（工厂按 (M,N,dtype) 分派：resident 链式 / narrow deinterleave / tiled online / **C6 N-split（round 3 翻转后新增）**）
- 任务：harness 迁移 Stage 4（mode=full），best_effort（无逐 workload 数值目标，以实测最优为结论）。本会话为**续跑**（session 1 完成至 round 2 合并时中断；round 0–2 数据来自 perf_records.jsonl 与 logs/，round 3 起为本会话实测）
- 工具链戳：tilelang 0.1.2（dev root build /home/tilelang/l00970450/upload/tilelang-mlir-ascend/build，2026-09-24）+ CANN 8.5.0 + Ascend910B2C（24 AIC × 2 = **48 Vector 核**，实查）+ torch 2.x (npu)
- 性能口径：**`msprof op` Task Duration(us)**（唯一 kernel 时延口径）；launch-count=20 / warm-up=5，取 20 次 captured launch 的均值。**C6 两段 kernel 的 workload 时延 = argreduce_partial + argreduce_merge 两个 Task Duration 之和**（同一 runner 进程、分别按 `--kernel-name` 捕获；prof_nsplit.sh）
- benchmark workload 来源：`examples/TileOPs/benchmarks/ops/bench_argmax.py::test_argmax_bench`（`workloads_to_params("ArgmaxFwdOp", include_extra=True)` 展开 manifest 8 案例：smoke×3 + tune×5）。ArgmaxFwdOp 类仍接 Stage-0 GPU kernel（NPU 不可跑），按调度指示仅提取 workload 参数集，全部时延直接对 Stage 3 standalone kernel 采集。
- 知识预注入消费：VP-2026-0044（row-reduction 基准形态——round 0 对照确认 baseline 偏离该形态实测水平，round 1–2 已对齐）；TRAP-DEVMODE-PERSIST-GEMM（含 gemm 不适用）；pattern-library 引用条目核对 status=verified（kb_stale_check 的 127 条 stale 为 attention 族，本轮未引用）。**本轮实测推翻了本任务 session 1 的一处归因**（详见 Iteration 4 证伪更正记录）。

## Workload Inventory（benchmark 展开，详见 workload_inventory.json）

| workload_id | kernel (M,N) dtype | 路径/配置（Stage 3 ladder） | kind |
|---|---|---|---|
| smoke-argmax-float16 / bfloat16 / float32 | (32,256) ×3 dtype | resident bm=8 | smoke（精度回归用，不采性能） |
| lm-head-argmax-float16 | (4,102400) fp16 | tiled bm=1 tn=5120（20 tile 零尾） | tune |
| lm-head-argmax-bfloat16 | (4,102400) bf16 | tiled bm=1 tn=4096（25 tile 零尾） | tune |
| hidden-state-argmax-float16 | (2048,4096) fp16 | resident bm=2（1024 block → persistent 48×22） | tune |
| hidden-state-argmax-bfloat16 | (2048,4096) bf16 | resident bm=1（2048 block → persistent 48×43） | tune |
| 3d-non-last-axis-argmax-float16 | (524288,4) fp16（dim=0 单维路径） | resident bm=1024（session-1 后为 narrow G=256/bw=4） | tune |

## Performance Test Data — Baseline（round 0, phase=baseline）

| workload_id | candidate | Task Duration(us) mean±stdev | Block Dim | aiv vec/scalar/mte2 ratio | aiv_mte2_bw | GM 流量（per-core×48 核对账） | profile |
|---|---|---:|---:|---|---:|---|---|
| lm-head-argmax-float16 | baseline | **25.89** ±0.16 | 4 | 0.396 / 0.329 / 0.270 | 28.3 GB/s | 200KB×4=800KB ✓ 精确 | profiles/stage4/baseline/lm-head-argmax-float16 |
| lm-head-argmax-bfloat16 | baseline | **30.61** ±0.20 | 4 | 0.488 / 0.258 / 0.255 | 25.2 GB/s | 200KB×4=800KB ✓ | profiles/stage4/baseline/lm-head-argmax-bfloat16 |
| hidden-state-argmax-float16 | baseline | **222.37** ±0.10 | 48 | **0.082 / 0.948** / 0.038 | 40.5 GB/s | 341.3KB×48=16MB ✓ | profiles/stage4/baseline/hidden-state-argmax-float16 |
| hidden-state-argmax-bfloat16 | baseline | **242.10** ±0.18 | 48 | **0.113 / 0.837** / 0.057 | 23.9 GB/s | 341.3KB×48=16MB ✓ | profiles/stage4/baseline/hidden-state-argmax-bfloat16 |
| 3d-non-last-axis-argmax-float16 | baseline | **1632.02** ±4.57 | 48 | 0.465 / 0.336 / 0.264 | 3.18 GB/s | **读侧 16× 膨胀（64MB vs 真实 4MB）**；写 4MB ✓ | profiles/stage4/baseline/3d-non-last-axis-argmax-float16 |

### 设计估算 vs 实测偏差行（D-2 回填，对照 DESIGN.md §1.6.0 roofline）

| workload | 设计估算下界 | 实测 baseline | 偏差 | 失准项 |
|---|---:|---:|---:|---|
| lm-head fp16 | 30–60µs | 25.89µs | 0.43–0.86×（优于） | 无（tn=5120 优于前次 tn=4096 校准值） |
| hidden-state fp16 | 47–74µs | 222.37µs | **3.0–4.7×** | **发射项**：融合候选循环未向量化（标量占比 0.95） |
| hidden-state bf16 | 55–100µs | 242.10µs | **2.4–4.4×** | 同上（标量 0.84） |
| 3d fp16 | ≈19µs | 1632.02µs | **≈86×** | **流量项 + 发射项双失准**：(bm,4) 窄行 2D copy 读侧 16× 膨胀 + 标量化 |

### Baseline 现象结论（Phase 1 排序依据）

1. **hidden-state（fp16/bf16）：融合候选循环标量化**（scalar 0.95/0.84）。session-1 归因嫌疑二选一：(a) `(bm,1)` 列广播操作数 `row_ext[i,0]`；(b) `T.cast(j,"float32")` 的 varange 生成失败。**Round 4 实测证伪更正：两者皆非——真触发器是 2D `T.Parallel(bm,N)` 融合循环本身在 bm≥2 时标量化**（见 Iteration 4 证伪记录）。
2. **3d：(bm,4) 窄行 2D copy 读侧 16× GM 膨胀**（每行 8B 按 128B 粒度取数）+ 同款标量化 → 1632µs（DESIGN §9.2-R10 风险兑现）。
3. **lm-head：4/48 核利用**（tiled bm=1 被 C-5 multitile 陷阱强制），单 kernel 25.9µs 已贴 4 核 MTE2 聚合带宽（800KB/30.9GB/s ≈ 25.9µs）——结构上无参数可解，DESIGN §1.6.4 C6 N-split 实验裁决为必做项。
4. 调优顺序：hidden-state fp16 → hidden-state bf16 → 3d → lm-head fp16/bf16。

## Iteration Log

### Iteration 1（session 1，round 1）：resident 链式替代融合式

- 现象：baseline hidden-state 标量化（上表）。
- 优化点：融合 `T.Parallel+if_then_else` 候选循环 → **链式** `vbrc(ext)→vcmp("eq")→vselect(idx_j, sent_v)→reduce_min`（DESIGN §1.6.3-#5 回退形态）；ext_brc 对全部 bm 物化（绕开 (bm,1) 操作数）；idx_j（T.arange）与 sent_v（标量 vbrc）hoist 到任务循环外；计算直读 x_ub（vcmp-on-shared，T.vselect.md §2.4 先例）。
- 分支：`_argreduce_kernel_round1_a_chain.py`（r1a_chain）+ 两个 config 变体（r1a_bm1 / r1a_bm256）。
- 候选 vs current best（baseline）对比表：

| workload | baseline (us) | r1a_chain (us) | vec/scalar（r1a） | L0 | 结果 |
|---|---:|---:|---|---|---|
| hidden-state fp16 | 222.367 | **47.127** | 0.33/0.60 | PASS | improved（4.72×） |
| hidden-state bf16 | 242.104 | **54.402** | 0.45/0.27 | PASS | improved（4.44×） |
| 3d fp16 | 1632.018 | 1486.838 | — | PASS | no_gain（窄行 copy 膨胀未解） |
| lm-head fp16 | 25.890 | 25.937 | — | PASS | tie（tiled 路径未改动） |
| lm-head bf16 | 30.610 | 30.766 | — | PASS | tie |
| hidden-state fp16 @bm=1（r1a_bm1） | 222.367 | 48.979 | — | PASS | config_no_gain（bm=2 更优） |
| 3d @bm=256（r1a_bm256） | 1632.018 | 1469.307 | — | PASS | config_no_gain |

- winner：r1a_chain（hidden-state 双 dtype 大幅提升；3d/lm-head 待后续轮）。

### Iteration 2（session 1，round 2）：3d 窄行 deinterleave 宽视图

- 现象：3d 的 (bm,4) 窄行 2D copy 读侧 16× 膨胀（GM_to_UB 64MB vs 真实 4MB）；probe_gran_W{4,8,16,32} copy-only 微基准定标行宽-效率曲线（W=4 → 9.74µs、W=8 → 6.28、W=16 → 5.56、W=32 → 4.52µs，W≥32 达渐近线）。
- 优化点：**narrow 宽视图 + deinterleave 相位分解**——kernel 参数声明为宽行视图 `(M//G, N*G)`（同一连续内存，narrow-path flat-view 先例）；block 按 (bw, N*G) 宽行 copy（消除读膨胀）；`T.deinterleave(channel_nums=2)` 两级树拆出 N=4 个相位 (bw,G)（channel_nums≥3 多 dst 形态触发 bishengir SIGABRT——probe_deint_bisect 实证，c2 形态已验证）；相位内 vmax/vmin 树 + vselect 首现。
- 分支：`_argreduce_kernel_round2_a_deint.py`（r2a_deint，G=256/bw=4/bm=1024）。
- 候选 vs current best（r1a_chain）对比表：

| workload | r1a (us) | r2a_deint (us) | L0 | 结果 |
|---|---:|---:|---|---|
| 3d fp16 | 1486.838 | 170.334（首测，同 sha 同配置）→ **12.381**（r2a2 复测） | PASS | improved（**131.8×** vs baseline；首测 170.33 为异常捕获，复测及 r3f/final 三次复证 12.4–12.5µs） |
| hidden-state fp16 | 47.127 | 47.112 | PASS | tie（resident 未改动） |
| hidden-state bf16 | 54.402 | 54.689 | PASS | tie |
| lm-head fp16 | 25.937 | 25.898 | PASS | tie |
| lm-head bf16 | 30.766 | 30.691 | PASS | tie |

- 合并检查：r2a2 全部 5 个 tune workload 复测（perf_records 行 15–19，phase=merged）+ L0 19 cases + edge 全过 → **r2a_deint 成为已合并 current best**。smoke 精度：L0 内 (32,256)×3 dtype PASS。

### Iteration 3（本会话，round 3）：C6 N-split 实验裁决（DESIGN §1.6.3-#4 必做项）⭐

#### Diagnostic Context（base = r2a_deint merged）

- lm-head fp16 25.898µs（Block Dim 4，mte2 28.3GB/s 聚合 = 4 核带宽地板）；bf16 30.691µs。理论：4 核 MTE2 聚合 ~30GB/s → 800KB ≈ 26µs，**实测已贴地板**——单 kernel 结构内无参数余量（bm=1 被 C-5 钉死、分核数上界 = M = 4）。
- 前次任务锚点（logs/final_round9_*）：partial ≈15.7µs + merge ≈1.7µs（tn≤256 参数化，pad 契约）。

#### 实验分支（均从 r2a_deint 派生，改 C6 结构一点）

| branch | 结构/配置 | lm-head fp16 partial+merge (us) | L0/tie 门 | 结果 |
|---|---|---:|---|---|
| r3a_chain_tn1280 | §1.6.4 规格链式 partial + (M,nchunk) 列写 + dims=1 merge；tn=1280/nchunk=80 | 25.118+1.462=**26.580** | 24/24 PASS | no_gain（劣于单 kernel） |
| r3b_chain_tn256 | 同上；tn=256/nchunk=400（前次锚点参数） | 19.653+1.633=**21.286** | PASS | no_gain（mte3 0.78——写放大） |
| r3c_chain_tn2048 | 同上；tn=2048/nchunk=50 | 29.612+1.893=**31.505** | PASS | no_gain（scalar 0.78） |
| r3d_fused_tn1280 | fused-ite 候选 + 扁平连续 ws 写（修 arange 物化 + 42× 写放大）；merge 改 dims=0 | 18.363+8.774=**27.137** | PASS | no_gain（dims=0 merge 灾难） |
| r3d_fused_tn256 | 同上；tn=256 | 18.005+33.405=**51.410** | PASS | no_gain（dims=0 @ (400,4) = 33.4µs） |
| r3e_perrow_tn1280 | **per-row (1,tn) partial**（全部 tiled 已验证形态；修 (M,tn) 融合 ite 标量化）+ T.transpose merge（回 dims=1） | 11.938+5.559=**17.497** | PASS | improved（过 0.8× 阈值） |
| r3e_perrow_tn2048 | 同上；tn=2048/nchunk=50 | 11.321+3.772=**15.093** | PASS | improved |
| **r3e_perrow_tn2560** | 同上；tn=2560/nchunk=40（最大单波 nchunk≤48） | **8.577+3.346=11.923** | PASS | **improved（0.461×，winner）** |
| r3e_perrow_tn5120 | 同上；tn=5120/nchunk=20 | 11.338+2.713=**14.051** | PASS | config_no_gain（20 核利用不足） |
| r3e_perrow_tn2560（bf16） | bf16 侧 winner 配置 | 8.913+2.917=**11.830** | PASS | improved（0.386×） |
| r3e_perrow_tn2048（bf16） | bf16 对照 | 12.052+3.262=15.314 | PASS | config_no_gain |

#### 根因诊断（r3a→r3e 的机制链，msprof 实测 + copy-only 对照更正）

1. **(M,tn) 2D 融合 ite 标量化（TRAP-parallel2d-bm-ge2-scalarize）**：r3d 的 `T.Parallel(M,tn) if_then_else` 候选循环在 M=4 下整体标量化（scalar 0.94，与 hidden-state baseline 同陷阱）——partial 地板的主因。**修复 = per-row (1,tn) 展开链**（`for r in range(M)`，全部 bm=1 已验证形态）。〔更正记录：本轮初判曾归因「(M,tn) 2D 跨步列块读 ~1GB/s」——copy-only 对照 A/B（pattern-library repro）实测两读形态 0.98× 无差异；gm_to_ub_bw 0.8–1.11 是标量化计算拉长总时长的**稀释口径**（同 profile 的 mte2_active_bw 一直有 24.9GB/s、mte2_ratio 仅 0.05–0.15）。判搬运瓶颈应用 mte2_ratio×aiv_time 与 *_active_bw——见 CONST-gm-to-ub-bw-dilution。〕
2. **(M,1) 列切片 ws 写 42× 写放大**：r3b tn=256 实测 UB_to_GM 400KB（真实 9.6KB）——每散布元素拉 128B 行。**修复 = 扁平 chunk-major 连续写**（chunk c 的 (M,) 结果落 flat `[c*M,(c+1)*M)`）。
3. **dims=0 长条形 merge 串行化**：r3d merge (nchunk,4) 逐列串行 walk（(400,4)=33.4µs、(80,4)=8.8µs）。**修复 = merge 内 T.transpose((nchunk,M)→(M,nchunk)) 回 dims=1 链**（r3a merge 1.46µs 形态；PL-1.1 transpose 家族）。
4. **T.arange (M,tn) 物化标量执行 ~1ns/elem**（r3a 链式，见 Iteration 4 的量化与绕法）。

#### 裁决（DESIGN §1.6.4 ③ 判定阈值：C6 端到端 < 0.8× 主选 → 翻转）

- fp16：**11.923µs / 25.898µs = 0.461×**；bf16：**11.830µs / 30.691µs = 0.386×** —— **均 « 0.8× → 该 shape 类分派翻转**（结构性加速 2.17×/2.59×）。
- tn 规则固化（r3e 扫描）：**最大单波 nchunk ≤ 48**（tn 升序扫描 256 倍数，首个除数命中）→ fp16/bf16 (4,102400) 均取 tn=2560/nchunk=40。
- 分派谓词（§1.6.4 ①）：`M ≤ 16 且 M < 48 且 N ≥ 32768 且 N%256==0` + tn 可导出 + merge 工作集 M×nchunk ≤ 6144。

#### 合并（r3f_nsplit_merged = r2a + C6 分派 + per-row partial + transpose merge）

| workload | current best r2a (us) | r3f merged (us) | precision | 状态 |
|---|---:|---:|---|---|
| lm-head fp16 | 25.898 | **11.994**（8.655+3.339） | L0/L1 PASS | pass（-53.7%） |
| lm-head bf16 | 30.691 | **11.851**（8.946+2.905） | PASS | pass（-61.4%） |
| hidden-state fp16 | 47.112 | 46.968 | PASS | pass（噪声内） |
| hidden-state bf16 | 54.689 | 54.347 | PASS | pass（噪声内） |
| 3d fp16 | 12.381 | 12.434 | PASS | pass（噪声内） |

- 全量 L0（19 cases + edge）+ L1（9）+ L2 对照：与 r2a 的 WARN 集完全一致（(1,1)/(2,1) 既有 record-only，非合并引入）→ **r3f 成为已合并 current best**。C6 精度专项门：跨 chunk tie / 相邻 chunk tie / 全 ±inf / M=1 / randn×3 × 2 dtype × 2 kind = **24/24 PASS**（round8-op9 4/4 mismatch 教训的针对性覆盖）。

### Iteration 4（本会话，round 4）：hidden-state 余量——semi-fused 证伪 + arange 标量地板修复

#### 现象（base = r3f）：hidden-state fp16 46.97µs（scalar 0.60）、bf16 54.35µs；per-block 2.14µs 中链式 6 op 已向量速率，但 hoisted `T.arange(idx_j (bm,N))` 物化疑为标量（Iteration 3 r3a 系列已实证 (M,tn) arange ~1ns/elem 标量地板）。

#### 分支 1：r4a_semifused（证伪更正轮）⭐

- 优化点：链式候选 → tiled 路径已验证的 **semi-fused 形态**（vbrc(ext_brc) + `T.Parallel(bm,N) if_then_else(x==ext_brc, j, BIG)`，x_work fragment staging）——同时消除 arange/sent_v/vcmp/vselect。
- 首版在 L2 (1,2)（bm=1, N=2）引入 mismatch（r3f 链式通过）→ 加 **N≥256 形态守卫**（`_SEMIFUSED_MIN_N`，链式回退；与 DESIGN §3.3 bm 形态分派同类的 codegen 安全分派）后 L0/L1/L2 与 r3f 完全一致。
- 实测（守卫版，hidden-state N=4096 走 semi-fused）：

| workload | r3f (us) | r4a (us) | vec/scalar（r4a） | L0 | 结果 |
|---|---:|---:|---|---|---|
| hidden-state fp16 | 46.968 | **222.093** | 0.082/**0.948** | PASS | **family_no_gain** |
| hidden-state bf16 | 54.347 | **242.446** | 0.113/**0.838** | PASS | **family_no_gain** |

- **证伪更正记录（canonical §2-3）**：session-1 对 baseline 标量化的归因嫌疑 (a)（(bm,1) 列广播操作数）与 (b)（varange 生成失败）**均被证伪**——r4a 已物化 ext_brc（修 (a)）、x_work staging + tiled 同款 `T.cast(j)`（tiled bm=1 向量化正常，否证 (b)），仍 0.948 标量、222µs ≈ baseline。**真触发器 = 2D `T.Parallel(bm,N)` 融合 if_then_else 循环在 bm≥2 时整体标量化**（bm=1 正常；三组操作数变体对照：baseline m[i,0] 0.95 / r4a ext_brc+x_work 0.948）。链式（显式向量 op）是 bm≥2 唯一向量化形态。误判根因 = 当时未做操作数变体对照；新数据见上表（pattern-library TRAP-parallel2d-bm-ge2-scalarize 已登记）。

#### 分支 2：r4b_arange1n（winner）

- 优化点（单点）：链式 hoist 的 `T.arange(idx_j (bm,N))` → **`T.arange(idx_row (1,N))` + 首轴 vbrc 广播 `(1,N)→(bm,N)`**（文档合法形态 src (1,N,K)→(M,N,K)）；bm=1 时同形 vbrc 空广播维触发 MLIR verify fail（`'hivm.hir.vbrc' op have empty broadcast dims array`）→ bm=1 直接 arange idx_j（该形态本就是 (1,N)）。
- 候选 vs current best（r3f）对比表：

| workload | r3f (us) | r4b (us) | vec/scalar（r4b） | L0 | 结果 |
|---|---:|---:|---|---|---|
| hidden-state fp16 | 46.968 | **38.406** | 0.427/0.487 | PASS | **improved（-18.2%）** |
| hidden-state bf16 | 54.347 | 54.561 | 0.455/0.268 | PASS | tie（bm=1 路径与 r3f 逐字节相同，+0.4% 为噪声） |

- 节省量 8.56µs 与 arange 标量模型精确吻合（(2,4096)=8192 elem × ~1.05ns）。
- 合并检查（r4b 全量复测）：

| workload | r3f (us) | r4b (us) | precision | 状态 |
|---|---:|---:|---|---|
| lm-head fp16 | 11.994 | 12.006（8.439+3.567） | PASS | pass（+0.1% 噪声） |
| lm-head bf16 | 11.851 | 11.873（8.826+3.047） | PASS | pass（+0.2% 噪声） |
| hidden-state fp16 | 46.968 | **38.406** | PASS | pass（-18.2%） |
| hidden-state bf16 | 54.347 | 54.561 | PASS | pass（+0.4% 噪声） |
| 3d fp16 | 12.434 | 12.673 | PASS | pass（+1.9% 噪声，< 3% 阈值；narrow 路径代码未改动） |

- 全量 L0/L1/L2 与 r3f 一致 → **r4b_arange1n 成为已合并 current best**（即最终版）。

### 3d 收束评估（Iter 2 后复核，无新分支）

3d 12.4–12.5µs 已**低于 DESIGN §1.6.0 roofline 估算（≈19µs，发射项模型）**；GM 读侧在 probe_gran 渐近线上（W≥32 → 4.5µs/4MB，narrow 路径宽行 2KB ≫ 渐近阈值）；写侧 4MB 连续。剩余 12.4µs vs 纯 copy 地板 ~9µs 的差距为 deinterleave/相位树/epilogue 的固定开销。(bw,G) 微调（如 G=128/bw=8，同 bm=1024）预期 <1µs——不设分支，3d 以 winner_merged 收束（131× vs baseline）。

## Final Performance Test Data

同一最终候选 r4b_arange1n，最终文件 `perf_opt/_argreduce_kernel.py`（sha256 7277fa06b47e66622f2ebc8f5368780ec802a8bb1166ddacd3eae4141f21251d）；lm-head 两行的 final_us 为 C6 两段 kernel（argreduce_partial + argreduce_merge）Task Duration 之和。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| _argreduce_kernel::main | lm-head-argmax-float16 | 25.890 | 11.775 | r4b_arange1n |
| _argreduce_kernel::main | lm-head-argmax-bfloat16 | 30.610 | 11.748 | r4b_arange1n |
| _argreduce_kernel::main | hidden-state-argmax-float16 | 222.367 | 38.218 | r4b_arange1n |
| _argreduce_kernel::main | hidden-state-argmax-bfloat16 | 242.104 | 54.829 | r4b_arange1n |
| _argreduce_kernel::main | 3d-non-last-axis-argmax-float16 | 1632.018 | 12.523 | r4b_arange1n |

提升与证据：

- lm-head fp16：**2.20×（-54.5%）**，partial 8.429 + merge 3.346（profiles/stage4_final/lm-head-argmax-float16__final_tn2560_{argreduce_partial,argreduce_merge}）
- lm-head bf16：**2.60×（-61.6%）**，partial 8.849 + merge 2.899（profiles/stage4_final/lm-head-argmax-bfloat16__final_tn2560_{argreduce_partial,argreduce_merge}）
- hidden-state fp16：**5.82×（-82.8%）**（profiles/stage4_final/hidden-state-argmax-float16__final）
- hidden-state bf16：**4.42×（-77.4%）**（profiles/stage4_final/hidden-state-argmax-bfloat16__final）
- 3d fp16：**130.3×（-99.2%）**（profiles/stage4_final/3d-non-last-axis-argmax-float16__final）

- smoke 最终精度：L0 19 cases + edge 全 PASS（含 smoke (32,256)×3 dtype；`logs/stage4_final/final_alllevels.log`）；workload_inventory `precision_pass: true` ×3。
- L1（9 cases）PASS；L2 仅既有 record-only WARN（(1,1)/(2,1) codegen 边界，r2a 对照一致，非本轮引入；NaN 哨兵行为与 DESIGN §9.2-R1 文档一致）。
- **几何平均提升（5 workload）= 11.5×**。全部 tune workload `tuning_status=winner_merged`。

## Final Summary

- 最终候选：`perf_opt/_argreduce_kernel.py`（= r2a_deint 链式/窄行 + r3e/r3f C6 N-split 分派 + r4b (1,N)-arange；工厂契约 `_argreduce_kernel(M,N,op_kind,dtype)(block_m)(x)` 不变，C6 的双发射与 workspace 分配封装在工厂组合层内——前次 round9 已证明该形态 harness bench + golden PASS 可行）。
- 关键有效优化点（按收益排序）：① 3d 窄行宽视图 + deinterleave 相位分解（131×）；② resident 链式替代 2D 融合循环（4.4–4.7×，含 round 4 证伪更正的机制归因）；③ C6 N-split per-row partial + transpose merge（lm-head 2.2×/2.6×，DESIGN §1.6.4 预注册裁决翻转）；④ (1,N)-arange + 首轴 vbrc（hidden-state fp16 再 -18.2%）。〔机制更正：C6 partial 的地板主因是 (M,tn) 2D 融合 ite 标量化（同②陷阱），非跨步读——copy-only 对照 0.98×，见 CONST-gm-to-ub-bw-dilution〕
- 中止原因：**success**（best_effort——全部 5 个 tune workload 有调优结论、最终全量记录完成、主要结构候选充分验证：C6 三件套裁决执行并翻转、semi-fused 族带机制证据关闭、3d 低于设计 roofline 收束；4 轮 / 16 个实验分支，预算 10 轮 / 30 分支内）。剩余微候选（lm-head partial 内 x_ub→w_r staging 剥离 ~5-10%、merge transpose 融合）记入遗留。
- 遗留问题：(1) (1,1)/(2,1) N=1 resident codegen 边界（L2 record-only，r2a 起既有）；(2) hidden-state bf16 54.8µs——bm=1 被 UB 预算钉死（bf16 14B/elem），semi-fused 族已证伪，链式为该形态族最优；(3) C6 分派的 DESIGN §1.4/§3.3 正式化需按 §1.6.4 ③ 走设计修订路由（见 perf_feedback.md）。
- [DESIGN_LIMIT]：**触发**（双门槛同时满足：设计层归因 = §1.6.3-#4 单 kernel 主选 + C-5 陷阱 → 4/48 核；结构性加速 = 2.20×/2.60× > 2×）→ `perf_opt/perf_feedback.md` 已产出，随返回附 `[DESIGN_LIMIT]` 信号（非阻塞；调优闭环已收束最优版本）。

## Skill Retrospective

- 流程问题：无阻塞级问题。本会话（续跑）受益于 session 1 的 runner 预置（run_bench `--mode nsplit` 接口在 round 3 直接可用）；中断续跑的工件边界（perf_records append-only + inventory）工作良好。
- 证伪更正：session-1 的标量化归因（嫌疑 (a)/(b)）被 round 4 的操作数变体对照证伪——**教训：归因「二选一」式嫌疑列表应在当轮就用变体对照消歧，而非留给后续轮**（已按 canonical §2-3 留痕：误判根因 + 合法形态 + 新数据；pattern-library TRAP-parallel2d-bm-ge2-scalarize 登记）。
- value point proposals（vp_type + 证据三件套，供 evolver 蒸馏；D/C 类已任务内回写 pattern-library）：
  - **D：T.arange (M,W) 物化标量执行 ~1ns/elem**（(1,N)+首轴 vbrc 绕法，(2,4096) 实测 -18.2%）→ constants.md CONST-arange-scalar-materialize；repro repro_arange_scalar.py（骨架形态：核内重复 arange 放大探针会误编译，勿复用该探针形态）。
  - **D（误读更正→方法论）**：gm_to_ub_bw 总时长稀释口径误判「跨步列块读慢」，copy-only 对照 0.98× 证伪 → constants.md CONST-gm-to-ub-bw-dilution（判搬运瓶颈用 mte2_ratio×aiv_time / *_active_bw）；repro repro_strided_colblock_read.py（阴性对照断言）。
  - **D：(M,1) 列切片 UB→GM 写 42× 放大**（128B 行/散布元素；400KB vs 9.6KB 实测）→ elementwise.md PL-1.19；repro 同上文件。
  - **D：dims=0 长条形 (nchunk,4) reduce 逐列串行**（(400,4)=33.4µs vs transpose+dims=1 3.3µs）→ constants.md CONST-reduce-dims0-skinny；repro repro_dims0_skinny.py。
  - **P：C6 N-split per-row partial + 扁平 chunk-major ws + transpose merge**（小 M 大 N 负载 2.2–2.6×；tn 规则 = 最大单波 nchunk≤cores）→ elementwise.md PL-1.20；repro repro_nsplit_pattern.py（delta 骨架）。
  - **C：CASE-argmax-argreduce-stage4**（本档案：C6 预注册裁决翻转 + 证伪更正 + 续跑工件边界）→ cases.md。
  - **R（proposal，不改流程文档）**：实验裁决三件套（§1.6.4 形态）+ core-split-strategy §2.4 的「备选胜出→采纳」路径在本次任务闭环良好，建议 designer 模板把「小 M 大 N 负载的 N-split 谓词」预置进 §5.5 分核三要素表（当前标准只覆盖 M 向分核）。
- pattern-library 回写：constants.md（+3 条）、elementwise.md（+2 条）、traps-compiler.md（+2 条）、cases.md（+1 条）、INDEX.md 路由行同步；repro 6 件（ED-B 断言形态）；`kb_lint.py` 校验通过。
- 工具链戳（全部新条目）：tilelang 0.1.2（dev root build 2026-09-24）+ CANN 8.5.0 + Ascend910B2C；origin_task: argmax-_argreduce_kernel-stage4-20260928。

---

## 采纳期修正附录（2026-09-28，Stage 5 采纳再验证，conductor）

Stage 4 终版（sha256 `7277fa06…`）在 TileOPs 采纳（raw-N 契约）再验证中发现 Stage 4 精度门未覆盖的窄 N 域缺陷，就地修正后现版本 sha256 `644c03e1…`：

1. **窄 N 派发守卫**（`_select_config`）：`N < TILE_ALIGNMENT(256)` 且不满足窄路径 → resident `block_m=1`。证据：链式 resident bm>1 与 tiled（两版 prim_func）在 N<256 域存在与 dtype/数值相关的静默真值损坏（(8192,2) tiled fp16 10-23/8192、(2048,16) tiled 输出垃圾索引 ~-1.08e9、(2048,30) tiled bf16 412/2048、链式 (2048,16) bm=128 261/2048 等）；resident bm=1 经 324 组合电池（N×M×dtype×seed）验证 0 失配。守卫对已调优工作负载（nsplit/链式宽 N/narrow N=4）零影响。
2. **narrow 排除 float32**（`_select_narrow`）：deinterleave 形态 fp32@N=4 损坏（(2048,4) bm=512 483/2048）；fp16/bf16 bm=1024 干净。fp32 落入守卫路径（resident bm=1，电池含 (2048,4) fp32 验证干净）。
3. **msprof 归因锚点**（工厂 `_func.msprof_kernel_name`）：nsplit→`argreduce_partial`（主 kernel），其余→`main`；经 wrapper/Op 上浮至 tier-2 解析，用于过滤 Op 层 movedim/contiguous 转置与多 launch 兄弟 kernel。

**未根因修复**（绕过式守卫）：窄 N 链式/tiled 缺陷疑为向量器小 N 行处理 + vselect carried state（DESIGN 9.1 C-1/C-8 同族）；N=1 所有路径编译失败（vbrc 空 broadcast 维度，baseline 持有 i1 规避）；宽 N 链式存在数据相关偶发错值观察（(8192,300) bm=16 seed 0 下 2/8192、seed 42 下 0）。建议独立 kernel 任务根因分析并把上述形状纳入回归电池。

验收：report 20260928_080104_950007_ArgmaxFwdOp status=passed（42/42，5 case 全 valid，0 warnings）；3d 5145→12.38µs、hidden fp16 222→38.24µs、lm-head partial 锚 25.74→8.50µs（kernel-sum ≈11.8µs）。
