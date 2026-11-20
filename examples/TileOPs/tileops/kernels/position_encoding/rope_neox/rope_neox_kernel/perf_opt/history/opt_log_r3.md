# Stage 4 调优日志 — `_make_rope_neox_1d`（Round 3：1d 时延拆解与尾部结构收敛）

## 0. 范围与环境（本轮 = Round 3）

- project：`rope_neox`（规范位）；覆盖集成算子 `RopeNeoxOp` / `RopeLongRopeOp` / `RopeLlama31Op`
  （三个 op 的 1d kernel 逐字节相同，一次调优三处采纳）。
- 本轮对象：**仅 `_make_rope_neox_1d.py`**。2d kernel
  （`perf_opt/_make_rope_neox_2d.py`，sha256 `ab22c025…`）本轮未改动，其上一轮记录原样保留在
  `perf_opt/history/`（`perf_records_2d.jsonl` / `workload_inventory_2d.json` / `op_log.md`）。
- 模式：`full`；预算：14 个实验分支（本轮实际 6 个测量分支 + 5 个探针，见 §3/§4）；
  中止原因：`plateau`（见 §6）。
- 测量口径：**`msprof op` `Task Duration(us)`**（kernel-only，唯一口径）。每 workload
  `--launch-count=20 --warm-up=5`，取 `OpBasicInfo.csv` 中位数；<5% 的候选差异按 T-2 用
  `.agents/tools/ab_test.py` 交错 A/B 多 run 协议裁决（本轮两次，见 §3.4）。
- 设备/工具链：Ascend910B3（20 AIC / 40 AIV = `_CORE_NUM=40`）、CANN 9.0.0、
  tilelang dev root build、`TILELANG_ASCEND_MODE=Developer`、torch_npu 2.7.1。
- 基准文件（`../_make_rope_neox_1d.py` = sha256 `80f4493a…`）与 wrapper 只读；所有产物写在
  `perf_opt/`。工厂签名保持 drop-in（`_make_rope_neox_1d(seq_len, head_dim, dtype)` →
  `kernel(block_size)`），`block_size` 语义与 wrapper `_DEFAULT_BLOCK_SIZE` 未改，
  故 wrapper 切换块无需成对修改 tuned 默认值。

## 1. Workload Inventory

来源与分类同上一轮（`examples/TileOPs/benchmarks/ops/bench_rope.py` 三个 bench 入口经
`workloads_to_params(include_extra=True)` 从 `tileops/manifest/position_encoding.yaml` 展开），
完整清单见 `workload_inventory.json`（7 条：3 smoke + 4 tune）。

| # | workload_id | shape / dtype | kind |
|---|---|---|---|
| 1 | `neox-1d-256-d64-smoke-{float16,bfloat16,float32}` | [256,64] ×3 | smoke（仅精度） |
| 2 | `neox-1d-2k-d64-fp16` | [2048,64] f16 | tune |
| 3 | `neox-1d-4k-d128-bf16` | [4096,128] bf16 | tune |
| 4 | `longrope-1d-8k-d128-bf16` | [8192,128] bf16 | tune |
| 5 | `longrope-1d-2k-d128-fp16` | [2048,128] f16 | tune |

调优顺序（负担/收益）：`longrope-1d-8k-d128-bf16` → `neox-1d-4k-d128-bf16` →
`longrope-1d-2k-d128-fp16` → `neox-1d-2k-d64-fp16`。

## 2. Performance Test Data（baseline = 上一轮采纳的 current best 重测）

`captured_op_name = main`；profile_status 全部 valid（`OpBasicInfo` 覆盖 25 次 launch）。
baseline artifact = `perf_opt/_make_rope_neox_1d.py`（本轮 round-0 起点，即上一轮 r2_contig，
sha256 `796840bc…`）。raw profile：`profiles/baseline/round0/r2_contig_96b2fe5d/{workload}/`。

| kernel_id | workload_id | Task Duration(us) | vec | mte2 | mte3 | scalar | 读带宽 GB/s | 写带宽 GB/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `_make_rope_neox_1d.py::_make_rope_neox_1d::main` | neox-1d-2k-d64-fp16 | 3.58 | 0.294 | 0.250 | 0.086 | 0.404 | 30.9 | 40.5 |
| `_make_rope_neox_1d.py::_make_rope_neox_1d::main` | neox-1d-4k-d128-bf16 | 4.92 | 0.473 | 0.387 | 0.109 | 0.285 | 42.0 | 72.7 |
| `_make_rope_neox_1d.py::_make_rope_neox_1d::main` | longrope-1d-8k-d128-bf16 | 6.35 | 0.639 | 0.514 | 0.154 | 0.238 | 41.9 | 70.0 |
| `_make_rope_neox_1d.py::_make_rope_neox_1d::main` | longrope-1d-2k-d128-fp16 | 3.88 | 0.364 | 0.247 | 0.084 | 0.332 | 51.2 | 71.9 |

（带宽为 msprof `aiv_mte2/mte3_active_bw` 单核均值 ×40 核聚合。）

### 2.1 设计估算 vs 实测（D-2 回填，Round 3 复测）

| workload | 设计估算（流量项 @1.06 TB/s） | baseline 实测 | 偏差归因 |
|---|---:|---:|---|
| neox-1d-2k-d64-fp16 | 0.74 µs | 3.58 µs | 4.8×：本轮实测证明该 workload **完全由 kernel 固定开销决定**（3.52 µs 地板，见 §3.1），流量项已无意义 |
| longrope-1d-8k-d128-bf16 | 2.97 µs（未计 cos/sin 的 2 MB → 6.29 MB 应为 5.9 µs） | 6.35 µs | 1.08×（按 6.29 MB 全流量口径）：设计把「半段 stride 效率」漏算（上一轮已修），本轮补齐固定项后估算与实测一致 |

## 3. Iteration Log（Round 3）

> 表列 = 候选 vs **current best**（B2 结构化回流）：Task Duration / AICore(Vector) 利用率 /
> memory 指标 / L0。纯 Vector 算子无 AICore 标量核指标，用 `aiv_vec_ratio` 作 AICore 利用率代理；
> memory 列给 `aiv_mte2/mte3_ratio` 与聚合读/写带宽。raw profile 见 `profiles/`（路径逐行标注）。

### 3.0 本轮起点现象（核心假设）

上一轮 log 声称「1d 已贴地板（读 1.6–2.0 TB/s / 写 2.6–2.9 TB/s）」，但按**总时延反算**对不上：
`longrope-1d-8k-d128-bf16` 的 x/y 只有 2 MB + 2 MB，而实测 6.35 µs ≫ 2 MB/1.8 TB/s +
2 MB/2.7 TB/s ≈ 1.85 µs + 1.9 µs 地板 ≈ 3.75 µs。本轮第一个动作就是**用探针把这 2.5+ µs 拆开**。

### 3.1 探针拆解（`_probe_1d_breakdown.py`，`ROPE_PROBE=<mode>`）

探针复用 current best 的**同一 buffer 集 / tile 推导 / DMA 形态**，只做减法（`--skip-l0`、不记
`perf_records.jsonl`）。`tiny_ub` = 每核仅 1 行 GM→UB→GM（10 KB / 核）；`copy` = 整行 x→y
（4.19 MB，8k）；`mem` = 全流量（x+cos+sin 读 6.29 MB、y 写 2.10 MB）但只 2 个算术 op；
`full` = 当前 best 本体。

| 探针（8k-d128-bf16） | 流量 | Task Duration(us) | aiv_time(us) | mte2 | mte3 | 差值归因 | raw profile |
|---|---|---:|---:|---:|---:|---|---|
| `tiny_ub`（1 行/核） | 10 KB | **3.52** | 1.78 | 0.290 | 0.319 | **kernel 固定地板** = 1.74 launch/ramp + 1.78 每核骨架 | `profiles/candidate/round3/probe_tiny_ub_7d4259bd/…` |
| `copy`（整行 x→y） | 4.19 MB | 4.00 | 2.18 | 0.540 | 0.350 | +0.48 µs（x/y 数据项） | `…/probe_copy_9e65f151/…` |
| `mem`（全流量，2 op） | 6.29+2.10 MB | 5.44 | 3.69 | 0.650 | 0.210 | +1.44 µs（**cos/sin 2.10 MB**） | `…/probe_mem_d9486467/…` |
| `full`（当前 best） | 同上 + 6 op 链 | **6.35** | 4.62 | 0.514 | 0.154 | +0.91 µs（**rope 6-op 链 vs 2 op**） | `…/r2_contig_96b2fe5d/…` |

同一探针在其他 3 个 tune shape 上（copy / mem / full，us）：

| workload | `copy` | `mem` | `full` | full − mem |
|---|---:|---:|---:|---:|
| neox-1d-2k-d64-fp16 | 2.98 | 3.38 | 3.58 | 0.20 |
| longrope-1d-2k-d128-fp16 | 3.21 | 3.61 | 3.88 | 0.27 |
| neox-1d-4k-d128-bf16 | 3.31 | 4.32 | 4.92 | 0.60 |
| longrope-1d-8k-d128-bf16 | 4.00 | 5.44 | 6.35 | 0.91 |

**结论（本轮核心交付）：那 2.5 µs 不是「带宽没打满」，而是三项固定/结构项之和。**
8k 的 6.35 µs 精确分解为

```
6.35 = 1.74 (launch/ramp，Task Duration − aiv_time，5 个探针一致在 1.73–1.83) 
     + 1.78 (每核骨架：40 核唤醒 + 7 个 DMA/UB 描述符 + 循环/guard，10 KB 探针实测)
     + 0.48 (x/y 数据 4.19 MB，L2 驻留 8.7 TB/s)
     + 1.44 (cos/sin 表 2.10 MB，1.45 TB/s)
     + 0.91 (rope 6-op 向量链相对 2-op 的增量，0.23 µs/iteration)
```

即：**`x`/`y` 从来不是瓶颈**（4.19 MB 只花 0.48 µs）；用户侧「2 MB 读 + 2 MB 写」的估算漏掉了
**cos/sin 表 2.10 MB（+1.44 µs）** 和**实测 3.52 µs 的 kernel 固定地板**（而非 1.9 µs）。
1.74 + 1.78 = 3.52 ≈ 固定地板，正是 2k/2k-d128 两档的几乎全部时延
（3.58 / 3.88 → 已达地板的 98% / 91%）。

### 3.2 由拆解导出的候选优化点（同一 base = current best）

- **A. 减少 fp32 中转 buffer（7→6）→ 放宽 tile 上限**：把两个交叉乘积 `a*s`、`b*s` 先算完，
  使 fp32 输入 buffer 直接充当输出（`A`、`B`），去掉 `y1f`/`y2f`；每行 UB 由 2816 B 降到 2560 B，
  于是 8k-d128 的 tile 可从 52 行抬到 69 行、**波次 4 → 3**（每波固定开销 ~0.82 µs 主导，
  该结构项是唯一还能动的大头）。数学与舍入顺序完全不变（每一式仍为「先乘后加/减」同序）。
- **B. `T.Pipelined(num_stages=2)` 双缓冲**：把每 iteration 的「load→compute→store」串行
  （`aiv_time` 4.62 µs 里含 ~0.91 µs 未被隐藏的向量时间）重叠掉。**代价**：2 级多缓冲要求
  同一 tile 的 buffer 翻倍，UB 只够 tile 26 行 → iteration 数 3 → 8，DMA 描述符翻倍。
- **C. 攻击 3.52 µs 固定地板**：launch/ramp（1.74）不可控；每核骨架（1.78）中 7 个描述符里
  4 个是 UB→UB 切分/合并——去掉它们即回退到上一轮已否决的「半段 stride DMA」（r1，写带宽
  276 GB/s）；cos/sin 已是单载复用（上一轮 R3/CSE）。→ 判 `family_no_gain`。

### 3.3 分支结果

| 分支 | 单点改动 | 2k-d64 | 4k-d128 | 8k-d128 | 2k-d128 | L0 | 结论 |
|---|---|---:|---:|---:|---:|---|---|
| current best（r2_contig） | — | 3.58 | 4.92 | 6.35 | 3.88 | PASS | 起点 |
| **A `r3_6buf`** | fp32 中转 7→6，tile 上限放到 UB clamp | 3.72¹ | 4.73¹ | **6.13–6.19** | 3.94¹ | PASS | **winner** |
| B `r3_pipe`（tile 26, 2 stage） | `T.Pipelined` 双缓冲 | — | — | 6.78 | — | PASS | rollback（tile 减半抵消重叠收益） |
| C | 攻击固定地板 | — | — | — | — | — | `family_no_gain`（见 §3.2 C） |

¹ 这两档的 tile 与 op 序列**逐字未变**，差异属 run-to-run 噪声（交错 A/B 已证 tie，见 §3.4）。

候选 vs current best（8k 分支扫描；L0 全 PASS）：

| candidate | workload_id | L0 | TaskDur(us) | vec | mte2 | mte3 | scalar | 读带宽 GB/s | 写带宽 GB/s | vs current best | raw profile |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| r3_6buf @tile52 | longrope-1d-8k-d128-bf16 | PASS | 6.29 | 0.661 | 0.531 | 0.157 | 0.209 | 40.9 | 69.3 | +0.9% | `…/r3_6buf_t52_c837c610/…` |
| r3_6buf @tile60 | longrope-1d-8k-d128-bf16 | PASS | 6.82 | 0.637 | 0.500 | 0.139 | 0.206 | 42.8 | 76.9 | **−7.4%**（137 tiles → 3.4 波，不整波） | `…/r3_6buf_t60_a8c46d44/…` |
| r3_6buf @tile69 | longrope-1d-8k-d128-bf16 | PASS | **6.13** | 0.583 | 0.566 | 0.149 | 0.202 | 41.1 | 78.7 | **+3.5%** | `…/r3_6buf_t69_07f3bcf5/…` |
| r3_pipe @tile26 | longrope-1d-8k-d128-bf16 | PASS | 6.78 | 0.691 | 0.697 | 0.251 | 0.297 | 28.4 | 39.3 | −6.8% | `…/r3_pipe_t26_2f66519e/…` |
| final（= A 合并版） | longrope-1d-8k-d128-bf16 | PASS | 6.07 | 0.590 | 0.543 | 0.148 | 0.215 | 43.3 | 80.0 | +4.4% | `profiles/final/round3/final_r3_24c3358e/…` |

`tile60` 的 −7.4% 是本轮有用的负结果：tile 不是单调收益，**必须让 `n_num` 贴合
`cores × iters` 的整数波次**（60 行 → 137 tiles → 3.4 波，尾巴波次把收益吃光）。

### 3.4 平区裁决（T-2 交错 A/B）

| 对局 | A（merged median） | B | rel change | pair diffs | verdict |
|---|---:|---:|---:|---|---|
| 8k-d128：r2_contig vs final | 6.37 µs | 6.15 µs | **−3.45%** | −0.25/−0.14/−0.20（3/3 同向） | `B_better` |
| 2k-d64：r2_contig vs final | 3.56 µs | 3.59 µs | +0.84% | +0.09/+0.03/−0.01（不同向） | `tie` |

→ 8k 的 3.45% 是真实增益（三对同向）；2k-d64 的单 run −3.6% 是噪声（交错裁决 tie），
即小 shape **没有回退**。ab 证据：`ab_round3/`、`ab_round3_small/`。

## 4. Autotune Log

未使用 `tilelang.autotune`（单参数 tile + 波次几何，手搜覆盖）：

| kernel / 结构 | 参数 | 扫描值 | 8k-d128 bf16（us） | winner |
|---|---|---|---|---|
| r2_contig（7 fp32 buffer） | `tile_rows` | 52 / 60 / 69 | 6.35 / 6.82 / 6.13 | 69（但 7-buffer 下 UB 只允许 64 行，取不到 69 → 需先减 buffer） |
| **r3_6buf（6 fp32 buffer）** | `tile_rows` | 52 / 60 / 69 | 6.29 / 6.82 / **6.13** | **69**（= UB clamp 内的 `ceil(8192/(3×40))=69`，整 3 波） |
| r3_pipe（2 stage） | `tile_rows` | 26 | 6.78 | 26（UB 上限即 26；更小只会更差） |

`tile_rows` 仍由 kernel 内「UB 预算 + `ceil(seq/(iters×cores))` 平衡」公式给出（不硬编码 shape
常量）：本轮把 `_UB_BUDGET_BYTES` 由 150 KiB 提到 176 KiB（= `_UB_HARD_BYTES`，即 tile 由 UB
clamp 决定），4 个 tune shape 的推导结果：

| workload | block_rows | n_num | iters | UB/核 |
|---|---:|---:|---:|---:|
| neox-1d-2k-d64-fp16 | 52 | 40 | 1 | 65 KB |
| neox-1d-4k-d128-bf16 | 52 | 79 | 2 | 130 KB |
| longrope-1d-8k-d128-bf16 | **69** | 119 | **3**（原 4） | 172.5 KB |
| longrope-1d-2k-d128-fp16 | 52 | 40 | 1 | 130 KB |

即**只有 8k 一档 tile 变大**，其余三档 tiling 与 op 序列逐字不变（故不存在回退风险）。

## 5. 合并非回退检查

- 合并版（`_make_rope_neox_1d_r3_merged.py` → 采纳为 `perf_opt/_make_rope_neox_1d.py`）复测本
  kernel **全部 4 个 tune workload**（§3.3 final 行 + §6 表），无结构性回退：3 档 tiling/op 序列
  不变（交错 A/B tie），1 档 8k 提升 3.45%（交错 A/B `B_better`）。
- smoke [256,64] fp16/bf16/fp32：随 L0 一起通过（max_diff=0.0），未采性能。
- 最终文件 `python perf_opt/_make_rope_neox_1d.py --level all`：**44/44 PASS，0 WARN**（rc=0），
  含 L2 边界（1/2/3/33/1025/4095）与 boundary 特殊值；日志 `logs/final/level_all.log`。
- 2d kernel 本轮未改动，无需复跑（sha256 仍为 `ab22c025…`）。

## 6. Final Summary

最终候选：`final_r3` = `r3_6buf_tile69`（fp32 中转 7→6 + tile 由 UB clamp 推导）；文件
`perf_opt/_make_rope_neox_1d.py`，sha256
`c6098298b94beb664c4ff670f33a2b15926838a207a9a6b74b5a1bfb1a504f4a`。
全部 final 记录均在同一最终候选上重测（`perf_records.jsonl` 12 行：4 baseline + 4 candidate +
4 final）。

关键结论与有效优化点：
1. **1d 从来不是带宽受限**：x/y 的 4.19 MB 只值 0.48 µs（L2 驻留 8.7 TB/s）；主要成本是
   **3.52 µs 的 kernel 固定地板**（1.74 launch/ramp + 1.78 每核骨架）与 **cos/sin 的 2.10 MB
   （1.44 µs）**，其次是 rope 6-op 链的 0.91 µs。
2. **唯一残余结构收益来自波次几何**：把 fp32 中转 buffer 收敛到 6 个后 tile 可到 69 行，
   8k-d128 由 4 波降到 3 波 → 交错 A/B 实测 **−3.45%**（6.37 → 6.15 µs）。
3. `T.Pipelined` 双缓冲在本算子**结构性不可行**：2 级多缓冲把 tile 压到 26 行、波次 3→8，
   净效果 −6.8%（已实测否决，不再是上一轮的「未实证假设」）。

中止原因：`plateau`。量化依据：4 个 tune workload 中 3 个已在地板附近（2k-d64 = 地板 ×1.02、
2k-d128 ×1.10），8k/4k 的剩余项为固定地板（3.52）+ 契约强制流量（cos/sin 2.10 MB）+ 0.91 µs
向量链，全部不可再减；本轮所有结构候选（A/B/C）已实证收敛。未触发 `[DESIGN_LIMIT]`
（无设计层结构性天花板：缺口已在本轮量化归因到 launch/骨架固定项与契约流量，不存在
>2x 的结构性加速空间，也未推翻 DESIGN 假设——DESIGN 的流量项估算与实测全流量口径差 1.08×）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox_kernel/_make_rope_neox_1d.py::_make_rope_neox_1d::main | neox-1d-2k-d64-fp16 | 3.58 | 3.71 | final_r3 |
| examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox_kernel/_make_rope_neox_1d.py::_make_rope_neox_1d::main | neox-1d-4k-d128-bf16 | 4.92 | 4.73 | final_r3 |
| examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox_kernel/_make_rope_neox_1d.py::_make_rope_neox_1d::main | longrope-1d-8k-d128-bf16 | 6.35 | 6.07 | final_r3 |
| examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox_kernel/_make_rope_neox_1d.py::_make_rope_neox_1d::main | longrope-1d-2k-d128-fp16 | 3.88 | 3.89 | final_r3 |

逐项：`+3.9% / +4.4% / −0.3%`（8k 为实证增益；4k 与 2k-d64 的 tile/op 序列未变，
±4% 属 msprof run-to-run 噪声带，2k-d64 已由交错 A/B 判 `tie`）。
smoke（[256,64] ×3）：仅精度回归，最终 `--level all` PASS；`perf_records.jsonl` 无 smoke 性能行。
skip：无。

## 7. Skill Retrospective

1. **「贴带宽地板」这类结论必须用探针拆解验证，不能只做流量/时延反算**：上一轮据「读 1.6–2.0
   TB/s、写 2.6–2.9 TB/s（active_bw）」判 1d 已贴 roofline，本轮三档探针（`tiny_ub`/`copy`/`mem`）
   证明真正的天花板是 3.52 µs 的 kernel 固定地板，带宽项只占 8k 的 30%。建议在
   `iteration-diagnosis.md` 增补强制步骤：**任何 plateau 判定前先跑「同 buffer/tile 的
   最小工作量探针」量出 fixed floor**，并用 `Task Duration − aiv_time` 交叉核对 launch 项
   （本算子 5 个探针稳定在 1.73–1.83 µs）。
   - vp_type: **P**；evidence: 本文件 §3.1 表（4 个探针 + 4 个 shape）+ raw profile
     `profiles/candidate/round3/probe_*`；repro: `perf_opt/_probe_1d_breakdown.py`（`ROPE_PROBE`
     环境变量选模式，自包含可跑）；toolchain_stamp: Ascend910B3 / CANN 9.0.0 / tilelang dev root /
     2026-11-01。target_doc: `pattern-library/bottleneck-patterns.md`。
2. **`T.Pipelined` 在 UB 预算紧张的 elementwise 算子上是负收益**：2 级多缓冲必然把 tile 减半、
   使 DMA 描述符数与波次数翻倍，而本族的主要成本恰恰是「每波固定开销」而非 compute。
   8k 实测 6.78 vs 6.13（serial，tile 69）。这是对上一轮「未实证假设（预估 <5%）」的**实证更正**：
   方向不是「<5%」，而是 **−6.8%**。
   - vp_type: **D**；evidence: 本文件 §3.3 `r3_pipe @tile26` 行；repro: `_make_rope_neox_1d_r3_pipe.py`
     （单行 `T.serial`→`T.Pipelined(iters, num_stages=2)`）；toolchain_stamp: 同上。
3. **波次几何（`n_num` 贴合 `cores × iters`）比 tile 大小本身更决定性能**：`tile60` 的 137 tiles
   （3.4 波）比 `tile52`（158 tiles，3.95 波）还慢 8.4%。建议在 pattern-library 增补
   「整波次（integer-wave）优先」的 tiling 判定式：先按 `ceil(S/(cores*iters))` 定 tile，
   再校验 `n_num <= cores*iters` 且 `n_num > cores*(iters-1)`。
   - vp_type: **P**；evidence: §3.3 / §4 的 tile 52/60/69 三点 + tile 26 一点；
     repro: `run_experiments.py --tile` 三点扫描；toolchain_stamp: 同上。
4. **小 shape 的 ±4% 单 run 波动会伪装成「回退」**：2k-d64 同一 artifact 两次单测 3.58 / 3.71，
   交错 A/B 判 tie（3.56 vs 3.59）。T-2 协议应在**每个**<5% 差异上强制使用，而不是只在
   「想采纳」时使用——否则会把噪声写进日志、误导下一轮。
   - vp_type: **R**；evidence: §3.4 两行 A/B JSON；toolchain_stamp: 同上。
5. 遗留：`4k-d128` 的 tile 仍为 52 行 / 2 波（`n_num=79`，79/80 波次已最优），若后续工具链
   放宽 UB（>192 KB）或新增 FMA（`y1 = fma(a,c,-(b*s))` → 6 op 降 4 op）可再试；本工具链
   文档无 `vfma`/`vmla`（已核 `docs/Tilelang.language/数学操作/` 全列表）。

## 8. 门禁自检记录（gate 4）

- `perf_records.jsonl`：12 行（4 baseline `r2_contig` / 4 candidate `r3_6buf_tile69` /
  4 final `final_r3`），每个 tune workload 恰 1 条 final 且 `artifact_sha256` =
  最终文件实际 sha256（`c6098298…`）。上一轮记录（含 2d）已整体移入 `perf_opt/history/`。
- `workload_inventory.json`：7 条（3 smoke + 4 tune，逐条 `tuning_status`）。
- 命令：`python .agents/tools/statectl.py gate 4 --dir examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox_kernel`
