# GroupNormFwdOp (GroupNorm forward) — Stage 4 调优日志

- **task_id**: `TileOPs-GroupNormFwdOp-20261018T021206Z`
- **算子目录**: `examples/TileOPs/tileops/kernels/norm/group_norm/group_norm_kernel/`
- **基准 kernel（只读）**: `_group_norm_kernel.py`（Stage 3，精度已通过）
- **设计文档（只读）**: `_group_norm_kernel_DESIGN.md`（§0.6 / §1.6.0 / §1.6.1 / §1.6.3 / §4.5 / §5.5 / §11）
- **benchmark 入口**: `examples/TileOPs/benchmarks/ops/bench_group_norm.py::test_group_norm_bench`（manifest-driven，7 case）
- **产物**: `perf_opt/GroupNormFwdOp.py`（最终 best）+ `perf_opt/_group_norm_kernel.py`（re-export shim）
- **工具链戳（L0/首轮必记）**: `tilelang 0.1.2+2afbf3b17f419e489a958fc9ee611e1d68ec8294`（commit 2afbf3b，dev-root build）+ `CANN 9.0.0` + **Ascend910B4**（`NPUUtils.get().get_aicore_num()` 实查 = 20 ⇒ Vector 逻辑核 = 40；`ub_cap = 196608 B`）
- **预算**: `max_stage4_experiments = 16`（本任务实验分支：R1 4 + R2 3 + R3 2 = 9；其余为标定探针与 baseline/final 测量，不计入分支预算）

## 0. 测量口径（唯一）

- **kernel-only 时延唯一口径 = `msprof op` 的 `Task Duration(us)`**，`--launch-count=20 --warm-up=5`，取 20 次 launch 的中位数（`perf_opt/read_profile.py` 解析 `OpBasicInfo_*.csv`）。
- 不使用 NPU event、runner 端到端时间或任何其它口径；所有决策（分支取舍、winner、是否达标）都只用该口径。
- 输出目录权限 700（否则 msprof 拒写），每个测量独立 `profiles/<stage>/<tag>/`、`logs/<stage>/<tag>.log`。
- **性能目标（conductor 给定，numeric）**：4 个 tune workload 的 report roofline Ratio 全部 ≥ **70%**。`Ratio% = C / duration_µs`（C 为该 case 的 `ratio × duration` 常数，逐档已复核：13.665×10.780215=147.31，15.3764×11.0602=170.07，13.4444×8.1802=109.98，10.5166×6.0201=63.31）；report 侧 `Ratio = performance_tops / computility_tops`（`run.json` 字段，arithmetic_intensity = 3.118 ops/byte），是**记账密度**口径，不是硬件利用率。
- **smoke-g8（fp32/fp16/bf16）不做性能调优**，只要求最终精度 pass；其数据不进 `perf_records.jsonl`。
- `<5%` 平区一律走 `ab_test.py` 交错 A/B 协议（≥3 对、交替顺序、sign test）。

## 1. Workload inventory 与 baseline（Phase 1）

`perf_opt/workload_inventory.json`：7 个案例 = **4 tune + 3 smoke**（`smoke-g8` 三 dtype；label 含 smoke token，gate 机械判为 smoke）。
展开来源与排序（先大 shape/结构收益明显，再小 shape；排序只影响先后，不跳过任何 tune workload）：

| 排序 | kernel_id / workload_id | workload | dtype | (M, D) | 基线 µs（本机实测） | report 基线 µs | C (%·µs) | 70% 上限 µs |
|---:|---|---|---|---|---:|---:|---:|---:|
| 1 | image-g32 / fp16 | [8,128,32,32] g=32 | float16 | (256, 4096) | **11.33** | 10.7802 | 147.31 | 2.104 |
| 2 | image-g32 / bf16 | 同上 | bfloat16 | (256, 4096) | **11.53** | 11.0602 | 170.08 | 2.430 |
| 3 | wider-channel-g32 / fp16 | [4,256,28,28] g=32 | float16 | (128, 6272) | **8.54** | 8.1802 | 109.98 | 1.571 |
| 4 | tail-spatial-g16 / fp16 | [4,128,30,30] g=16 | float16 | (64, 7200) | **5.96** | 6.0201 | 63.30 | 0.904 |

（smoke-g8 = (16, 256) × {float32, float16, bfloat16}，只做精度。）

baseline 采集（部署态：wrapper 的 `_select_row_config` 结构性返回 `block_m = 1`，40 个 Vector 核）：

| kernel_id / workload_id | Task Duration µs | Block Dim | vec / mte2 / mte3 / scalar µs | aiv_time µs | GM 读+写 | 达成 GB/s | raw profile |
|---|---:|---:|---|---:|---|---:|---|
| image-g32/fp16 | 11.33 | 40 | 4.70 / 2.75 / 1.17 / 3.67 | 10.67 | 4.10 MB | 371 | `perf_opt/profiles/baseline_v2/image-g32_fp16` |
| image-g32/bf16 | 11.53 | 40 | 4.88 / 2.78 / 1.19 / 3.66 | 10.89 | 4.10 MB | 364 | `perf_opt/profiles/baseline_v2/image-g32_bf16` |
| wider-channel-g32/fp16 | 8.54 | 40 | 3.63 / 1.84 / 0.81 / 2.68 | 7.90 | 3.14 MB | 377 | `perf_opt/profiles/baseline_v2/wider-channel-g32_fp16` |
| tail-spatial-g16/fp16 | 5.96 | 40 | 1.98 / 1.29 / 0.79 / 2.31 | 5.26 | 1.81 MB | 312 | `perf_opt/profiles/baseline_v2/tail-spatial-g16_fp16` |

- 本机 baseline 与 report 基线的差：image fp16 +5.1%、bf16 +4.2%、wider +4.4%、tail **−1.0%**（不同 harness 采样条件，本机为 msprof `--launch-count=20` 中位数）；**决策一律用本机同口径数字**，report 基线只用于 C 常数换算。
- **设计估算 vs 实测偏差行（D-2 回填）**：DESIGN §1.6.0 对 image-g32 的估算下界 ≈ **24 µs**（发射项主导：12 op/task × 4 task × 0.5 µs/op），实测 baseline **11.33 µs** ⇒ 估算比实测**高 2.1x**（估计失准项 = **发射单价**：实测 scalar 3.67 µs / 3 轮 ≈ 0.07 µs/op，而非 0.5 µs/op；流量项假设 1.356–1.72 TB/s 亦被实测纯 DMA 0.68–0.96 TB/s 推翻，见 §2）。设计文档自己标注该 0.5 µs/op 为「保守上界」，实测确认其保守性约 7x。

## 2. Round 0：固定开销 / copy-floor 归因（先决策，再调参）

全部为**标定探针**（同 `T.Kernel(num_kernels)` persistent 结构），数据落 `perf_opt/copy_floor_calibration.jsonl`（21 行），**不进 `perf_records.jsonl`**。

### 2.1 空转（launch/固定开销）地板

| 探针 | Grid | Task Duration µs | aiv_time µs |
|---|---:|---:|---:|
| empty-b1 | 1 | 1.13 | — |
| empty-b2 | 2 | 1.42 | — |
| empty-b29（image 部署栅格） | 29 | **2.32** | — |
| empty-b32（wider/tail 部署栅格） | 32 | **2.46** | — |
| empty-b40 | 40 | 2.89 / 2.91（两次） | — |

结论：本机（910B4 / CANN 9.0.0）空 kernel 地板随 block 数增长，**1.13 → 1.42 → 2.32（b29）→ 2.46（b32）→ 2.91（b40）µs**。外机常数（VP-2026-0247 记 910B3/CANN 8.5：2 block 3.63 / 40 block 4.81 µs）**不适用于本机**，故本任务一律用本机数字（该条目标注待重验）。

### 2.2 纯 DMA 地板与 copy-floor

| 探针（同结构） | 栅格 / bm | 流量 | Task Duration µs | 达成 GB/s |
|---|---|---:|---:|---:|
| dma_read（只读 x，单元素保活） | 40 / 1 | 2.05 MB | 5.85 | 360 |
| dma_rw（读+写，无计算） | 40 / 1 | 4.10 MB | 6.14 | 684 |
| chain（读 + 2 v-op + 写） | 40 / 1 | 4.10 MB | 5.62 | 747 |
| dma_rw image | 29 / 3 | 4.10 MB | **4.39** | 956 |
| chain image | 29 / 3 | 4.10 MB | 4.55 | 923 |
| dma_rw wider | 32 / 2 | 3.14 MB | **3.81** | 844 |
| chain wider | 32 / 2 | 3.14 MB | 3.97 | 810 |
| dma_rw tail | 32 / 2 | 1.81 MB | **3.09** | 598 |
| chain tail | 32 / 2 | 1.81 MB | 3.27 | 565 |

- 纯 DMA 达成率仅 **598–956 GB/s**（部署栅格），远低于 DESIGN 假设的 1.356–1.72 TB/s；同族大流量参照 `CONST-copy-floor-dma-granularity`（16.78 MB、16 KB 粒度）也只 1.06 TB/s，64 KB 粒度 1.23 TB/s —— **本形状（1.8–4.2 MB）由固定开销主导，任何结构的可达带宽都远低于 70% Ratio 隐含的 2.0 TB/s**。
- `chain` 只比 `dma_rw` 高 0.16–0.18 µs ⇒ 2 个保活 v-op 几乎被 DMA 吸收（`mte2_ratio` 0.43–0.50）；但算法 kernel 的 5 个元素级 pass + 2 次 `reduce_sum` 无法隐藏（见 §2.3、`CG-2026-0026/0027`）。
- 口径说明：`dma_rw`/`chain` 探针的 `Task Duration` 对同一结构族有效；`dma_read`（只读）在 40 block 时反而比 `dma_rw` 慢，属小 kernel 双态，不作为决策依据。

### 2.3 归因表（逐 workload，部署栅格）

| workload | 固定开销地板 | 纯 DMA 地板 | copy-floor（+2 v-op） | 当前（Round 0 = baseline） | 70% 目标 | 缺口（目标 vs 地板） | 70% 可达？ |
|---|---:|---:|---:|---:|---:|---|---|
| image-g32/fp16 | 2.32（b29） | **4.39** | 4.55 | 11.33 | 2.104 | 纯 DMA 地板 **2.09x 高于**目标 | ❌ 结构性不可达 |
| image-g32/bf16 | 2.32（b29） | **4.39** | 4.55 | 11.53 | 2.430 | **1.81x 高于**目标 | ❌ 结构性不可达 |
| wider-channel-g32/fp16 | 2.46（b32） | **3.81** | 3.97 | 8.54 | 1.571 | **2.42x 高于**目标 | ❌ 结构性不可达 |
| tail-spatial-g16/fp16 | 2.46（b32） | **3.09** | 3.27 | 5.96 | 0.904 | **3.42x 高于**目标 | ❌ 结构性不可达 |

pipe 分解（`CONST-aiv-time-decomposition` 口径：`Task ≈ max(mte2, vec) + scalar` 的实测形态为 pipe 近串行相加）：

- baseline image-g32/fp16：vec 4.70 + mte2 2.75 + mte3 1.17 + scalar 3.67 = 12.29 vs wall 11.33（仅少量重叠）；`mte2_ratio = mte2/dur = 0.24` ≪ 0.95 ⇒ **不是 MTE2 受限，而是 vec + scalar 暴露**。
- **`PL-1.10-loads-first-decoupling` 适用性判定：不适用**——本算子单输入（x 一次装载 + 一次写出），没有第二输入可前置；且 `mte2_ratio` 0.17–0.24 表明瓶颈不在装载与计算的串行，而在 vec 链本身。未照抄。
- **`CG-2026-0027` 对照（reduce 与 GM→UB 不可重叠）**：本轮实测同向——`chain`（+2 v-op）只 +0.16 µs，而加入 2 次 `reduce_sum` + 3 个全平面 v-op 后 image-g32 从 4.55 → 11.33 µs（+6.8 µs，即 5 个元素级 pass + 2 次归约的完整暴露量）。同一栅格纯拷贝 923 GB/s vs 归约链 371 GB/s，与条目记录的 690 vs 211 GB/s 同型，**不重开「发现」轮**。

### 2.4 决策（Round 0 结论）

- **4/4 档的 70% 目标都低于实测纯 DMA 地板（1.81x–3.42x）**，其中 3 档（image fp16 2.104、wider 1.571、tail 0.904）甚至低于部署栅格的**空 kernel 地板**（2.32/2.46 µs）。
- 满足「≥3 档结构性不可达 ⇒ 早收束」的触发条件；按调度要求仍须给出**可达工作量证明**（1–2 个最可能有效的参数级/结构级尝试并实测），见 §3/§4/§5（R1 block_m、R2 均衡分核 + multi-buffer A/B、R3 multi-buffer off 换 block_m=4）。**不为凑数烧预算**：预算 16，实际分支 9。

## 3. Round 1：block_m（部署态结构性 bm=1 → tuned）

**现象**（base = baseline，§1 表）：4 档全在 40 个 Vector 核上跑 `num_logical = ceil(M/1)` 个 1 行任务（image 256 个 ⇒ 每核 7 轮），每轮固定成本（scalar/发射）线性累加；image scalar 3.67 µs、tail scalar 2.31 µs 已占 wall 的 32%/39%。DESIGN §0.6 R2 / §4.5 本意是「block_m 由 UB 预算表驱动」，但部署态被 wrapper 的 `_DEFAULT_BLOCK_M = 1` 覆盖（wrapper 只读）。

**候选（单点：block_m 取值）**：`_func(block_m=1)` 解释为「未指定 → 按 (M, D, dtype) 查 UB 预算表」；显式 >1 原样尊重。UB 单价在**本工具链实测**（`probe_ub.py` 读编译期 `ub overflow ... requires N bits`）：

| 结构 / dtype | 实测单价 | 依据 |
|---|---|---|
| fp16/bf16（stage + x_f32 + sq_f32 + 5×(bm,1)） | **12.000–12.002 B/elem** | D=4096/6272/7200 的溢出报文反解 |
| fp32（x_f32 + sq_f32 + 5×(bm,1)） | **10.001–10.002 B/elem** | D=4096 的溢出报文反解 |
| bm=4 @ D=4096（mb on，40-block 栅格） | 196672 B（12.004 B/elem，超 64 B）⇒ **不存在** | 见 §5 的栅格依赖修正 |

⇒ 预算 `(196608 − 64) // 12 = 16378` elem：D=4096 → bm=3；D=6272/7200 → bm=2；D≤1152 截顶 bm=8。DESIGN §4.5 的 20 B/elem（承自 ada_layer_norm 结构）对本 kernel **高估 67%**，被实测替换。

**分支结果（候选 vs Round-0 current best；Task Duration 为 msprof 中位数）**：

| 分支（同一 base = baseline bm=1） | Task Duration µs | Δ vs current best | AICore 利用率（aiv/dur） | GM 读+写 / 达成带宽 | L0（golden 精确比对） | 结论 |
|---|---:|---:|---:|---|---|---|
| image-g32/fp16 bm=2（nk40） | 9.94 | **−12.3%** | 93.2% | 4.10 MB / 422 GB/s | PASS（`logs/branch_check/image-g32_fp16_bm2.log`，0 violations） | 有效，非最优 |
| image-g32/fp16 bm=3（nk40） | **9.54** | **−15.8%** | 93.6% | 4.10 MB / 440 GB/s | PASS（`..._bm3.log`，0 violations） | **winner（image）** |
| image-g32/fp16 bm=4（mb on） | — | — | — | — | — | **编译阻塞**：196672 B > 196608 B |
| wider-channel-g32/fp16 bm=2（nk40） | **7.45** | **−12.8%** | 91.0% | 3.14 MB / 432 GB/s | PASS（`..._bm2.log`） | **winner（wider）** |
| tail-spatial-g16/fp16 bm=2（nk32） | **5.02** | **−15.8%** | 83.5% | 1.81 MB / 368 GB/s | PASS（`..._bm2.log`） | **winner（tail）** |

- 合并（同一 `@T.prim_func` 内按编译期常量生效，无 Python 工厂分派）：image-g32 两 dtype 用 bm=3、wider/tail 用 bm=2，`_TUNED_BLOCK_M` 为空（预算表本身即最优）。合并后复测 4 档 + smoke 精度 —— 见 §4/§5 的合并检查表。
- **本轮的 UB 单价是 D 类实测常数**，同时修正了 DESIGN §4.5 的 20 B/elem 与 §1.6.3 备选档的 11.2 B/elem（见 §5：mb off 在本结构上 ≤12 B/elem）。

## 4. Round 2：均衡分核 + DESIGN §1.6.3 实验裁决（multi-buffer on/off）

**现象**（base = Round-1 current best）：image bm=3 ⇒ 86 个逻辑任务，40 核上 `28×3 + 1×2`（30 核跑 3 轮、10 核跑 2 轮但其中 9 个被 if 守卫空转）⇒ 关键路径仍由 3 轮块决定；wider bm=2 ⇒ 64 个任务，40 核 `24×2 + 16×1` ⇒ 16 个核只跑 1 轮，理论 −25% 的均衡空间。`mte2_ratio` 仍在 0.18–0.25（非搬运受限）。

**候选**（单点：persistent 栅格缩小到「保持轮数的最小核数」`num_kernels = ceil(num_logical / num_local_tasks)`）：

| 分支（同一 base） | Task Duration µs | Δ vs current best | AICore 利用率 | GM 读+写 / 带宽 | L0 | 结论 |
|---|---:|---:|---:|---|---|---|
| image-g32/fp16 bm=3, nk=29（current best 9.54） | **9.20** | **−3.6%** | 92.0% | 4.10 MB / 456 GB/s | PASS（分支复验 `logs/branch_check/image-g32_fp16_bm3.log`） | winner（与 A/B 一致） |
| image-g32/fp16 bm=3, mb **off**（DESIGN 1.6.3 备选@同 bm） | 9.38 | +1.9% | 93.4% | 4.10 MB / 448 GB/s | PASS | 否决（同 bm 下关闭多缓冲净亏） |
| wider-channel-g32/fp16 bm=2, nk=32（current best 7.45） | **6.43** | **−13.7%** | 87.9% | 3.14 MB / 500 GB/s | PASS（`logs/branch_check/wider-channel-g32_fp16_bm2.log`） | **winner（wider）** |

**interleaved A/B 裁决（`ab_test.py`，3 对，交替顺序）**：

- `ab_round2_nkc`（A = nk40，B = nk29，image-g32/fp16）：A 9.30 / 9.50 / 9.24，B 9.26 / 9.32 / 9.18 ⇒ merged 9.34 vs 9.25，`b_rel_change = −0.9%`，**direction_consistent = true（3/3）**。降幅虽 <5%，但 sign test 3/3 一致 + 物理机理（少 11 个空转核 ⇒ 空转地板 2.89→2.32 µs）成立，且 §5 的合并复测再次确认，故采纳（记录为小步正收益，不单独作为达标依据）。
- `ab_round2_mboff`（A = mb on，B = mb off，image-g32/fp16 bm=3）：A 9.16 / 9.58 / 9.34，B 9.54 / 9.60 / 9.52 ⇒ `b_rel_change = +3.4%`，A 胜 3/3 ⇒ **维持 multi-buffer on**（该 A/B 只覆盖「同 bm」档，见 §5 的 bm 档裁决）。

**DESIGN §1.6.3 实验裁决三件套执行情况**：主选（auto multi-buffer 开）在本轮代表 shape 上胜出（同 bm 档），但**备选档的真实自由度不是「同 bm」而是「mb off ⇒ 更大 bm」**——该部分在 §5 完成，结论与 §1.6.3 的判定阈值（备选 < 主选×0.95 才翻转）按 shape 分片执行。

**合并检查（image bm=3/nk29 + wider bm=2/nk32 + tail bm=2/nk32，同一 kernel）**：4 档 tune workload 复测 + smoke 精度 —— 见 §5.3 合并后全量复测（R2 合并版 → `profiles/final_v2/`：image fp16 9.20 / bf16 9.36 / wider 6.44 / tail 5.02；smoke 3 dtype golden PASS）。无回退（wider 6.43→6.44，噪声内）。

## 5. Round 3：DESIGN §1.6.3 备选的真身（multi-buffer off 换 block_m=4）+ 收束

**现象**（base = Round-2 current best = image 9.20 / bf16 9.36）：image 的 vec 4.82 µs 是最大单项，但真正可压的是**每核轮数**（3 轮）与 scalar 2.39 µs；UB 预算把 bm 钉在 3，而 mb on 的 bm=4 需要 229408 B（14.0 B/elem，本轮实测；40-block 栅格下是 196672 B —— **同一结构同一 bm 的 UB 单价随栅格变化**，见 `copy_floor_calibration.jsonl` 的 `ub-*` 行）。

**先做编译探针（不烧测量预算）**：

| 探针 | 结果 | 含义 |
|---|---|---|
| mb **on**, bm=4, D=4096（32-block 栅格） | 需 229408 B（14.002 B/elem） | 不存在 |
| mb **on**, bm=4, D=4096（40-block 栅格） | 需 196672 B（12.004 B/elem） | 也不存在（超 64 B）；证明单价依赖栅格 |
| mb **off**, bm=4, D=4096 | **COMPILED**（≤12.0 B/elem） | 备选档在本 shape 上可落地 |
| mb **off**, bm=3, D=6272 | 需 225824 B（12.002 B/elem） | wider 无法借该开关升 bm ⇒ 保持 bm=2/mb on |
| 3-temp 原地统计变体 + mb on, bm=4 | 需 229408 B | **否决**：去掉 2 个 `(bm,1)` 暂存并不能换来 bm=4（编译器改判多一层缓冲） |

**候选（单点：multi-buffer off 使 bm=3 → 4）**：image-g32 专用 `_MB_OFF_SHAPES = {(256, 4096)}` + `_TUNED_BLOCK_M = {(256,4096,fp16/bf16): 4}` ⇒ `num_logical = 64`，`num_kernels = 32`，**每核 2 轮且完全均衡**。

**interleaved A/B（`ab_test.py`，3 对，image-g32/fp16）**：A（mb on, bm=3, nk=29）9.18 / 9.23 / 9.18；B（mb off, bm=4, nk=32）7.68 / 7.72 / 7.70 ⇒ **merged 9.18 → 7.70，`b_rel_change = −16.12%`，`direction_consistent = true`（3/3，pair_diffs −1.50/−1.51/−1.48）**，远超噪声阈值 ⇒ 翻转 image-g32 分片。

**候选 vs current best 对比表（合并后全量复测，`profiles/final_v4/`）**：

| workload | current best（R2）µs | R3 候选 µs | Δ | AICore 利用率 | GM 读+写 / 带宽 | L0 / smoke 精度 | 结论 |
|---|---:|---:|---:|---:|---|---|---|
| image-g32/fp16 | 9.20 | **7.71** | **−16.2%** | 89.2%（aiv 6.89） | 4.10 MB / 544 GB/s | bench `--check` PASS（0 violations）+ L0/L1 全绿 | **winner** |
| image-g32/bf16 | 9.36 | **7.85** | **−16.1%** | 89.7%（aiv 7.07） | 4.10 MB / 533 GB/s | PASS（0 violations） | **winner** |
| wider-channel-g32/fp16 | 6.44 | 6.44 | 0.0%（噪声） | 87.6% | 3.14 MB / 499 GB/s | PASS | 保持（mb on, bm=2, nk=32） |
| tail-spatial-g16/fp16 | 5.02 | 4.99 | −0.6%（噪声） | 83.8% | 1.81 MB / 370 GB/s | PASS | 保持（mb on, bm=2, nk=32） |

- 机理：轮数 3→2 使 scalar 2.39→2.12 µs、mte2 1.64→1.61 µs（持平）、vec 4.82→4.11 µs（同样的元素总数但每轮 tile 更大、UB 复用更好）；32 个栅格块也把空转地板从 2.32（b29）压到 2.46（b32 略高，但轮数收益远大于该 +0.14 µs）。
- **合并方式**：只在 `@T.prim_func` 外的编译期常量（`_TUNED_BLOCK_M` / `_MB_OFF_SHAPES` / UB 预算）上按 shape 分派，**没有新增 Python factory/op/wrapper 分派**；`_func(block_m=1)` 的「未指定 → tuned」语义与显式 >1 原样尊重保持不变。
- **收束判定**：R3 之后剩余缺口（image 7.72 vs 目标 2.104 = 3.7x；wider 4.1x；tail 5.5x）全部落在 §2.3 的地板之外（纯 DMA 地板 4.39/3.81/3.09 已高于目标 1.8–3.4x），剩余候选只有「压低 vec 链 pass 数」，而 pass 数由精度契约锁定（fp32 中转 2 次 vcast、中心化两遍方差 2 次 reduce、`vsqrt`+`vdiv` rstd；`VP-2026-0248` 的 fp16 精简链与本案精度契约冲突，仅作 UB 受限对照、不得牺牲精度）⇒ **无参数级剩余空间，收束**。`[DESIGN_LIMIT]` 判定见 `perf_opt/perf_feedback.md`。

### 5.3 合并后全量复测（最终候选，smoke 只做精度）

| workload | Task Duration µs | Block Dim | 配置 | 精度 |
|---|---:|---:|---|---|
| image-g32/fp16 | 7.71 | 32 | bm=4, mb off, nk=32 | bench --check PASS（max_diff 1.953e-03, 0 violations） |
| image-g32/bf16 | 7.85 | 32 | bm=4, mb off, nk=32 | PASS（1.490e-08, 0 violations） |
| wider-channel-g32/fp16 | 6.44 | 32 | bm=2, mb on, nk=32 | PASS（1.953e-03, 0 violations） |
| tail-spatial-g16/fp16 | 4.99 | 32 | bm=2, mb on, nk=32 | PASS（1.953e-03, 0 violations） |
| smoke-g8/fp32（不做性能） | — | — | bm=8 | PASS（4.768e-07, 0 violations） |
| smoke-g8/fp16（不做性能） | — | — | bm=8 | PASS（0.000e+00） |
| smoke-g8/bf16（不做性能） | — | — | bm=8 | PASS（0.000e+00） |

### Final Performance Test Data

（下述 4 行全部来自**同一最终候选** `perf_opt/GroupNormFwdOp.py`（sha256 `69d8298183b072e9e0bf68e15d0198280e32f5902eb72d686c1d1eb085bf3415`）在冻结后的一次 `msprof op --launch-count=20 --warm-up=5` 采集中测得，raw 目录 `perf_opt/profiles/final_v5/`；`baseline_us` 为部署态（wrapper bm=1）在本机的同口径中位数。）

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| image-g32 | fp16 | 11.33 | 7.71 | image-mboff-bm4-nk32-r3 |
| image-g32 | bf16 | 11.53 | 7.85 | image-mboff-bm4-nk32-r3 |
| wider-channel-g32 | fp16 | 8.54 | 6.44 | wider-bm2-nk32-r2 |
| tail-spatial-g16 | fp16 | 5.96 | 4.99 | tail-bm2-nk32-r1 |

### 6.1 逐档 Ratio 预估（口径 `Ratio% = C / duration_µs`，C 为 report 常数）

| workload | report 基线 µs / Ratio | 本机 baseline Ratio | 最终 µs | 最终 Ratio（预估） | 70% 目标 µs | 达标？ |
|---|---|---|---|---|---:|---|
| image-g32/fp16 | 10.7802 / 13.665% | 13.00% | 7.71 | **19.11%** | 2.104 | ❌（3.67x 缺口） |
| image-g32/bf16 | 11.0602 / 15.376% | 14.75% | 7.85 | **21.67%** | 2.430 | ❌（3.24x 缺口） |
| wider-channel-g32/fp16 | 8.1802 / 13.444% | 12.88% | 6.44 | **17.08%** | 1.571 | ❌（4.11x 缺口） |
| tail-spatial-g16/fp16 | 6.0201 / 10.517% | 10.62% | 4.99 | **12.68%** | 0.904 | ❌（5.52x 缺口） |

- Ratio 提升 = 时延下降：−32.0% / −31.9% / −24.6% / −16.3% ⇒ Ratio 相对 report 基线 +39.8% / +40.9% / +27.1% / +20.7%（**仍远低于 70%**）。
- 达标判据未满足：4/4 档 Ratio ≪ 70%。`[DESIGN_LIMIT]`（结构性，非参数可解）见 `perf_opt/perf_feedback.md`。

## 7. 精度回归（最终文件）

- `python perf_opt/GroupNormFwdOp.py --level all` → **exit 0**，`logs/final_regression/level_all_frozen2.log`：
  - **L0: ALL PASS（17 items）**（含 4 个 manifest tune 档 + smoke 3 dtype + 尾块/非对齐/3D spatial/大均值等）。
  - **L1: ALL PASS**（dtype × shape × block_m 网格、3 输入覆盖、bm sweep、persistent 核数边界 39/40/41/50、非连续输入、factory 双配置复用、UB/域/dtype guard、ALIGNMENT=1）。
  - **L2（非阻塞）**: 6 PASS + 2 WARN — `(1,1,1)` 极小档（torch 侧 "Expected more than 1 value per channel" 限制，非 kernel 问题）、`large-mean-tiny-var` fp32（最大 abs 3.86e-05，条件数角落，DESIGN §8.2 已登记）。
  - **Boundary（非阻塞）**: 6 PASS + 1 WARN — `const-rows` fp32（var=0 行，2.46e-05，同上条件数角落）。
  - `Summary max_diff per dtype: bfloat16=6.250e-02, float16=3.906e-03, float32=3.839e-05`（bf16 6.25e-02 出现在 L2 的 D=9216 大行档，仍在 bf16 容差 `atol=rtol=1.6e-2` 的 `atol+rtol|ref|` 之内）。
- 7 个 benchmark case 的 golden 精确比对（`bench.py --check`，走 benchmark 参数化路径，独立于分层测试）：全部 PASS、**0 violations**。
- **未改** `tests/ops/test_group_norm.py`、**未新增 skip**；wrapper / DESIGN / Stage 3 kernel 均只读未动（`perf_opt/` 之外零改动）。

## 8. Final Summary

- **最终候选**：`perf_opt/GroupNormFwdOp.py`（sha256 `69d82981…415`），单 kernel 按 shape 编译期分派：
  - image-g32（D=4096）：`block_m=4` + auto multi-buffer **off** + 32 个栅格块（64 任务 / 2 轮均衡）
  - wider-channel-g32（D=6272）/ tail-spatial-g16（D=7200）：`block_m=2` + multi-buffer **on** + 32 个栅格块
  - 其余 shape 走实测 UB 表（12 B/elem fp16/bf16、10 B/elem fp32，D≤1152 截顶 8）
- **提升**：image-g32 fp16 **11.33 → 7.71 µs（−32.0%）**、bf16 **11.53 → 7.85 µs（−31.9%）**、wider **8.54 → 6.44 µs（−24.6%）**、tail **5.96 → 4.99 µs（−16.3%）**；全部为同一最终候选的冻结后复测。
- **关键有效优化点（按贡献）**：① multi-buffer off 换 `block_m=4`（image，−16.1%，`ab_round3_mboff_bm4`）；② `block_m` 由实测 UB 表驱动（image −15.8%、wider −12.8%、tail −15.8%）；③ 均衡栅格（wider −13.7%，image −3.6%）。
- **中止原因**：`blocked` —— 剩余缺口全部落在**实测地板之外**（4/4 档目标低于纯 DMA 地板 1.81–3.42x，3/4 档低于空 kernel 地板），属设计层天花板；已按 `perf-feedback.md` §1 双门槛产出 `perf_opt/perf_feedback.md` 并返回 `[DESIGN_LIMIT]`。
- **遗留 / 未验证项**：
  1. 70% Ratio 未达标（结构性，非本 Stage 可解）。
  2. R2 的两个 image 分支（mb-off@bm3、nk29@bm3）在 `perf_records.jsonl` 中没有独立行——其分支源码已被 R3 覆盖（append-only 纪律：不为「已不存在字节的测量」补写记录），数值保留在 §4 表与 `profiles/round2/`。
  3. 空 kernel 地板与 UB 单价是**本机 910B4 + 本工具链**实测，跨型号/跨工具链需重测（`CONST-*` 相关条目在本快照里标 stale，见 §10）。
  4. `torch_npu` 融合 GroupNorm 参照（`GroupNormSilu`）含 per-channel affine，与本 kernel（affine 在 op 层）不完全同口径，仅作量级锚定（见 `perf_feedback.md` 参照锚定）。
  5. `perf_records.jsonl` 曾在**冻结前**重生成一次（首版写于最终文件定稿前，final 行哈希会失效）；交付版本全部为冻结后测量，见 §9。

## 9. 记录完整性说明（append-only 实况）

- `perf_records.jsonl` 共 14 行：baseline ×4（round 0）、R1 候选 ×4、R3 候选 ×2、final ×4（每 tune workload **恰 1 行**，同一 candidate 覆盖 image-g32 两 dtype）。
- **冻结前重生成（2 次，如实披露）**：记录首版写于 `GroupNormFwdOp.py` 定稿**之前**；其后文件又经历两次「不影响生成代码」的修订 ——（a）§5 的 UB 单价/栅格依赖注释修正（把 round-1 的 196640 B 读数更正为 32 栅格下的 229408 B 并说明差异）；（b）把 `@tilelang.jit(**_jit_kwargs)` 改成字面量 `target="npuir"`（gate `S4-JIT` 要求显式声明，语义等价）。每次修订后都重跑 `--level all` + 4 档 final 采集并以新字节重写记录。未有任何门禁消费过更早版本，交付版（sha256 `69d82981…415`）是唯一对外版本；此后文件进入冻结态，未再改动（当前 sha 与全部 final 行一致）。
- 每个 final 行的 `artifact_sha256` = 门禁时刻 `perf_opt/GroupNormFwdOp.py` 的真实 SHA256；`msprof_raw_path` 均指向真实 raw 目录。

## 10. Skill Retrospective

### 10.1 本次调优暴露的流程问题

1. **Round 0 的「空 kernel 地板」必须绑定栅格，不能只报一个数**：本机 1.13/1.42/2.32/2.46/2.91 µs（b1/b2/b29/b32/b40）随 block 数近线性增长。若只测 b40，会把 70% 不可达的归因说过头（image bf16 目标 2.430 > b29 地板 2.32）；若只测 b2，又会漏掉「栅格是参数自由度」这一结论。建议在 skill 的 Round-0 清单里把地板探针写成「= 部署栅格 ± 两侧各一档」。
2. **UB 单价同时依赖 (dtype 路径 × bm × 栅格 × 缓冲个数 × 编译器版本)**：同一结构同一 bm=4，40-block 栅格 196672 B（12.004 B/elem）而 32-block 栅格 229408 B（14.002 B/elem）；去掉 2 个 `(bm,1)` 暂存反而仍是 14.002 B/elem。**线性外推会直接把「bm 能到几」判错**。建议 `TRAP-UB-multibuffer-inflation` 的用法行显式加上「栅格」这一维，并要求「用目标栅格探测」。
3. **DESIGN §1.6.3 的裁决口径应写成「备选的完整自由度」而非「同参数切换」**：R2 只测了 mb off @同 bm=3（输 3.4%），据此若收束就会错过 R3 的 −16.1%——因为备选的真正卖点是「off 换更大 bm」。建议 skill 在「实验裁决执行」条目里加一句：**切换类备选必须在其「表征自由度」（此处 bm）上评估，否则不得判负**。
4. **`ab_test.py` 的 `--mb` 语义与实现相反**（`bench.py` 里 `--mb 0` = multi-buffer **on**，`--mb 1` = off，源出 `_MB_OVERRIDE = bool(arg)` 但 `_mb_off_for` 返回 `bool(_MB_OVERRIDE)`）——本任务因同名布尔被两次误读，靠 `--print-bm`/日志核对才发现。建议模板里把该开关命名为 `--disable-mb` 而不是 `--mb 0/1`。
5. **小 kernel 的 <5% 结论必须走 A/B**：nk40→nk29 单次 9.54→9.20 看似 −3.6%，A/B 只有 −0.9%；mb-off 单次 9.38 vs 9.20 看似 +1.9%，A/B 是 +3.4%。两者都由 A/B 定案，避免了「把噪声写进 winner」。

### 10.2 value point proposals（供终态 `tilelang-skill-evolver` 蒸馏）

| 编号（建议） | vp_type | 主张 | 证据三件套（evidence / repro / toolchain_stamp） |
|---|---|---|---|
| D-GN-01 | D | **910B4 / CANN 9.0.0 空 kernel Task 地板：b1 1.13、b2 1.42、b29 2.32、b32 2.46、b40 2.89–2.91 µs**（同 persistent 结构、单元素保活写） | evidence: `perf_opt/copy_floor_calibration.jsonl`（`empty` 行）+ `profiles/round0/empty_b*`、`profiles/floor_v2/empty_b*`；repro: `perf_opt/probe_floor.py --mode empty --blocks N`；toolchain_stamp: `tilelang 0.1.2+2afbf3b` + CANN 9.0.0 + 910B4（AIV=40，ub_cap 196608） |
| D-GN-02 | D | **小形状纯 DMA 地板随规模骤降**：4.10 MB/29 块 4.39 µs（956 GB/s）、3.14 MB/32 块 3.81 µs（844）、1.81 MB/32 块 3.09 µs（598）——比同族 16 KB 粒度常数（1.06 TB/s @16.78 MB）低 11–44%，说明「地板」必须按 worklaod 规模取，不能用大流量常数外推 | evidence: `copy_floor_calibration.jsonl`（`dma_rw` 行）；repro: `probe_floor.py --mode dma_rw --case <case> --bm <bm> --blocks <nb>`；同上 stamp |
| D-GN-03 | D | **本 kernel 缓冲结构的 UB 单价：fp16/bf16 12.00–12.00 B/elem、fp32 10.00 B/elem；且 bm=4 档随栅格 12.004→14.002 B/elem 变化，去掉 2 个 (bm,1) 暂存不降反平**（证明单价不可线性外推） | evidence: `copy_floor_calibration.jsonl` 的 `ub-*` 行 + `logs/round3`；repro: `probe_ub.py --impl <kernel> --bms 3,4 --mb 1`；同上 stamp |
| P-GN-01 | P | **降低 persistent 轮数是本族小形状的第一杠杆，而「轮数」的可行域由 UB 单价 × 栅格决定**：multi-buffer off 把 D=4096 的 bm 从 3 提到 4（轮数 3→2），−16.1%（interleaved A/B 3/3）；同 bm 关闭则 +3.4%。模式表述：*先求「每核轮数 R = ceil(ceil(M/bm)/nk)」的最小可行组合，再决定 multi-buffer 开关；开关本身不是优化点，是达到 bm 的手段* | evidence: `ab_round3_mboff_bm4`（A 9.18/9.23/9.18 vs B 7.68/7.72/7.70）、`ab_round2_mboff`（同 bm off 输 3/3）；repro: 建议以 `perf_opt/bench.py --mb 0/1` + `ab_test.py` 最小化；同上 stamp |
| P-GN-02 | P（反向） | **`PL-1.10-loads-first-decoupling` 不适用于单输入行归一化**：本算子 x 单次装载，无第二输入可前置；判据 = `mte2_ratio` 0.17–0.25 且 `chain`（+2 v-op）仅 +0.16–0.18 µs，瓶颈在 vec+scalar 而非装载串行 | evidence: `copy_floor_calibration.jsonl` 的 dma_rw/chain 行对比 + baseline 的 pipe 分解；repro: `probe_floor.py --mode {dma_rw,chain}`；同上 stamp |
| R-GN-01 | R | **切换类备选（如 multi-buffer 开关）必须在其「表征自由度」上评估**：仅在原参数上 A/B 会误判——本案同 bm 关闭输 3.4%，而 off 换 bm=3→4 赢 16.1%。建议写入 optimize skill 的「实验裁决执行」条 | evidence: `perf_opt/opt_log.md` §4 vs §5；repro: 以 `ab_round2_mboff` 与 `ab_round3_mboff_bm4` 两组 JSON 对照；同上 stamp |
| C-GN-01 | C | 案例索引：**GroupNorm 小形状（1.8–4.2 MB）在 910B4 上 Ratio ≥70% 结构性不可达**（目标 < 纯 DMA 地板 1.8–3.4x，且 < 空 kernel 地板 3/4 档）；厂商融合 `GroupNormSilu` 同 shape 14.04 µs（image-g32/fp16），本方 7.72 µs 已快 1.82x。归档 `perf_feedback.md` + `copy_floor_calibration.jsonl` 作为后续同族任务的零成本筛查数据 | evidence: `perf_feedback.md`、`profiles/torch_ref/*`、`profiles/final_v4/*`；repro: `torch_ref.py` + `probe_floor.py`；同上 stamp |

> 回写 pattern-library：上述 D 类常数与 P 类模式已按 skill Phase 4 要求追加到 `references/pattern-library/`（见下），本表 evidence 列同时标注了回写位置。
