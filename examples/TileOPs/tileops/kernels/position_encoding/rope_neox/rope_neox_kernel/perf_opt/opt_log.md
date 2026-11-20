# Stage 4 调优日志 — `_make_rope_neox_1d`（Round 4：4k/8k 残余 3–5% 空间受控实验）

## 0. 范围与环境（本轮 = Round 4）

- project：`rope_neox`（规范位）；覆盖集成算子 `RopeNeoxOp` / `RopeLongRopeOp` / `RopeLlama31Op`
  （三个 op 的 1d kernel 逐字节相同，一次调优三处采纳）。
- 本轮目标（conductor 指定的受控实验）：在 **r3 current best**（sha256 `c6098298…`）之上，
  验证 `neox-1d-4k-d128-bf16` / `longrope-1d-8k-d128-bf16` 的剩余 3–5% 空间；2k 档只做回归确认。
- 结果：**本轮无采纳**（`no_gain`）。r3 逐字节保留为最终版本（sha256 未变），
  两个有效候选的 5-run 交错 A/B 均判 `tie`（4k 甚至略慢），见 §3.5。
- 模式：`full`；预算 `max_stage4_experiments=6` 个实验分支。本轮实际：baseline 重测 1 +
  已测量候选 2（`r4_prefetch` / `r4_fuseup`）+ 编译期否决候选 1（`r4_vcastslice`，
  未产生性能测量）= **3 个测量分支**，未超预算。
- 测量口径：**`msprof op` `Task Duration(us)`**（kernel-only，唯一口径）。
  `--launch-count=20 --warm-up=5`，取 `OpBasicInfo.csv` 中位数；<5% 的候选差异按 T-2 用
  `.agents/tools/ab_test.py` 交错 A/B 多 run 协议裁决（本轮 3 组，见 §3.5）。
- 设备/工具链：Ascend910B3（20 AIC / 40 AIV = `_CORE_NUM=40`）、CANN 9.0.0、
  tilelang dev root build、`TILELANG_ASCEND_MODE=Developer`、torch_npu 2.7.1。
- 基准文件（`../_make_rope_neox_1d.py`，sha256 `80f4493a…`）与 wrapper 只读；2d kernel
  （`perf_opt/_make_rope_neox_2d.py`，sha256 `ab22c025…`）本轮**未改动**。
- 轮次归档（按 r3 先例）：上一轮的 12 行 `perf_records.jsonl` → `history/perf_records_r3.jsonl`；
  上一轮日志 → `history/opt_log_r3.md`（含 Round 0–3 全部拆解、探针与 A/B 证据）。
  本文件只记录 Round 4，并保留 r0–r3 的结论摘要（§1.1）。

### 1.1 继承结论（Round 0–3，详见 `history/opt_log_r3.md`）

1. 1d 不是带宽受限：8k 的 `x`/`y` 4.19 MB 只值 0.48 µs（L2 8.7 TB/s）；主要成本是
   **实测 3.52 µs 的 kernel 固定地板**（1.74 launch/ramp + 1.78 每核骨架）+ **cos/sin 表
   2.10 MB ≈ 1.44 µs** + rope 6-op 链 0.91 µs。
2. r3 的唯一结构收益：fp32 中转 buffer 7→6 → 8k tile 52→69 行、波次 4→3（A/B −3.45%）。
3. 已实证否决：`T.Pipelined(num_stages=2)` −6.8%（多缓冲把 tile 压到 26 行、波次 3→8）；
   `tile=60` −7.4%（137 tiles = 3.4 波，未整波）；行 padding 修 bank group 冲突使 vec 翻倍
   （VP-2026-0149）。
4. 波次几何已到整数最优：8k 119/120、4k 79/80、2k 40/40；再降一档需 tile ≥ 103 行
   ≈ 263.7 KB > 176 KB UB 硬限 → 本工具链不可行。
5. FMA 能力缺口：`docs/Tilelang.language/数学操作/` 全目录（20 个 API）无 `vfma` / `vmla`，
   6-op 链（4 乘 + 2 加减）无法再化简。

## 2. Workload Inventory

来源同前（`examples/TileOPs/benchmarks/ops/bench_rope.py` 三个 bench 入口经
`workloads_to_params(include_extra=True)` 从 `tileops/manifest/position_encoding.yaml` 展开）；
完整清单见 `workload_inventory.json`（7 条：3 smoke + 4 tune）。

| # | workload_id | shape / dtype | kind | 本轮处理 |
|---|---|---|---|---|
| 1 | `neox-1d-256-d64-smoke-{float16,bfloat16,float32}` | [256,64] ×3 | smoke | 仅精度（`--level all` PASS） |
| 2 | `neox-1d-2k-d64-fp16` | [2048,64] f16 | tune | 回归确认 tie（§3.5） |
| 3 | `neox-1d-4k-d128-bf16` | [4096,128] bf16 | tune | **主判**：2 候选 + A/B |
| 4 | `longrope-1d-8k-d128-bf16` | [8192,128] bf16 | tune | **主判**：2 候选 + A/B |
| 5 | `longrope-1d-2k-d128-fp16` | [2048,128] f16 | tune | 回归确认（与 2k-d64 同一 `iters==1` 编译分支） |

## 3. Iteration Log（Round 4）

> 表列 = 候选 vs **current best**（B2 结构化回流）：Task Duration / AICore(Vector) 利用率
> （`aiv_vec_ratio` 作纯 Vector 算子的 AICore 利用率代理）/ memory 指标
> （`aiv_mte2/mte3_ratio`、`mte2_wait_ratio`）/ L0。raw profile 逐行标注。

### 3.0 本轮起点现象（来自 r3 final profile，本机 raw OPPROF 复核）

| workload | TaskDur | aiv_time | vec | mte2 | mte3 | scalar | mte2_wait | mte3_wait | vec cflt | scalar_vec_stall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4k-d128 | 4.73 | 3.03 | 1.50 | 1.20 | 0.34 | 0.79 | 0.69 | 0.68 | 0.195 | 0.00 |
| 8k-d128 | 6.07 | 4.27 | 2.47 | 2.32 | 0.62 | 0.92 | 0.77 | 0.78 | 0.130 | **1.18** |

由此形成的两个假设（本轮分别验证）：

- **H1（overlap）**：4k 的 `aiv_time` 3.03 而最长 pipe `vec` 只有 1.50，且 `mte2_wait` 0.69 /
  `mte3_wait` 0.68 → 每 tile 的 `load → split → cast → 6-op → cast → merge → store` 被串行化，
  DMA 未被隐藏。→ 分支 A。
- **H2（op 数 / 标量等待）**：8k 的 `aiv_scalar_vector_stall_time` 1.18 µs（占 aiv_time 28%）
  → scalar 在等 vector 完成，每 tile 约 16 条 vector 指令的**指令数**本身在关键路径上。
  → 分支 B（融合 cast 与 UB→UB 拷贝）。

### 3.1 Baseline 重测（round-4 起点 = r3，sha256 `c6098298…`）

raw profile：`profiles/baseline/round4/r3_base_r4_8ee06f17/{workload}/`。
本 session 比 r3 采集时（4.73 / 6.07）慢约 1–2%，故本轮全部比较都以**同 session 的 baseline**
为锚（这也是 §3.5 用交错 A/B 而非单 run 的原因）。

| kernel_id | workload_id | TaskDur(us) | vec | mte2 | mte3 | scalar |
|---|---|---:|---:|---:|---:|---:|
| `…_make_rope_neox_1d.py::_make_rope_neox_1d::main` | neox-1d-2k-d64-fp16 | 3.61 | 0.290 | 0.242 | 0.084 | 0.419 |
| 同上 | neox-1d-4k-d128-bf16 | 4.89 | 0.486 | 0.377 | 0.112 | 0.277 |
| 同上 | longrope-1d-8k-d128-bf16 | 6.41 | 0.560 | 0.532 | 0.141 | 0.226 |
| 同上 | longrope-1d-2k-d128-fp16 | 3.86 | 0.356 | 0.245 | 0.088 | 0.347 |

### 3.2 分支 A `r4_prefetch`：轮转软件流水（无额外 UB）

**单点改动**：把每个 tile 的 GM→UB staging 装载提前到**上一 tile 的 vector 尾段**——
`x_stage` / `c_ub` / `s_ub` 在 up-cast 之后即死亡，因此可原地预取下一 tile，**不需要任何额外
UB**（这是与已否决的 `T.Pipelined` 的本质区别：后者复制全部 buffer，tile 26 行、波次 3→8）。
`iters == 1` 的 2k 两档走编译期 `else` 分支，循环体与 r3 逐字相同（结构性零回退风险）。
文件：`_make_rope_neox_1d_r4_prefetch.py`。

| 项目 | r3 current best | r4_prefetch | 判读 |
|---|---:|---:|---|
| 4k TaskDur（单 run 粗筛） | 4.89 | **4.76**（+2.7%） | 看似有增益 → 进 A/B |
| 8k TaskDur（单 run 粗筛） | 6.41 | **6.30**（+1.7%） | 看似有增益 → 进 A/B |

但同批 raw profile 显示**收益来源不成立**：

| metric | 4k r3 | 4k prefetch | 8k r3 | 8k prefetch |
|---|---:|---:|---:|---:|
| `aiv_time` | 3.161 | 3.158（不变） | 4.351 | **4.557（更差）** |
| `aiv_vec_time` | 1.503 | 1.592（更差） | 2.466 | **2.703（更差）** |
| `aiv_mte2_time` | 1.205 | 1.079（更好） | 2.358 | 1.999（更好） |
| `aiv_mte2_wait_ratio` | 0.674 | 0.514 | 0.754 | 0.638 |
| `aiv_scalar_vector_stall_time` | 0.000 | 0.000 | 1.214 | **0.013** |
| `aiv_scalar_mte2_stall_time` | 0.000 | 0.000 | 0.000 | **0.821（新出现）** |

即：预取确实把 MTE2 的**等待**搬走了（8k `mte2_wait` 0.754→0.638、`scalar_vector_stall`
1.214→0.013），但代价是 `vec_time` 上升（+0.24 µs）并新增了 `scalar_mte2_stall` 0.82 µs，
**`aiv_time` 净变差**。→ 结构性判读：本 kernel 的 pipe 之间**本来就已经重叠**，
旋转只把等待从一条 pipe 挪到另一条；单 run 的 +1.7/+2.7% 落在 ±4% 噪声带内（§3.5 A/B 证实）。

raw profile：`profiles/candidate/round4/r4_prefetch_02d29f7e/{workload}/`。

### 3.3 分支 B `r4_fuseup`：UB→UB split 与 up-cast 融合为跨 dtype `T.copy`

**单点改动**：按 `docs/Tilelang.language/内存操作/T.copy.md` §2.3.4，Developer 模式的跨 dtype
`T.copy` 由编译器插入 `hivm::VCastOp(RoundMode=RINT)`，与显式 `T.vcast(..., "rint")` 等价，
因此把 `copy(split) + vcast(up)` 两指令融合为一指令（每 tile 少 2 条 vector 指令）。
文件：`_make_rope_neox_1d_r4_fuseup.py`（L0 9/9 PASS，`max_diff=0`）。

结果**明确负收益**，且暴露了跨 dtype copy 的真实代价：

| metric | 8k r3 | 8k fuseup | 4k r3 | 4k fuseup |
|---|---:|---:|---:|---:|
| TaskDur（单 run） | 6.41 | **7.29（−13.7%）** | 4.89 | **5.10（−4.3%）** |
| `aiv_time` | 4.351 | **5.645** | 3.161 | （同向变差） |
| `aiv_vec_time` | 2.466 | 2.667（更多） | 1.503 | — |
| `aiv_mte2_wait_ratio` | 0.754 | **0.850** | 0.674 | — |
| `aiv_vec_total_cflt_ratio` | 0.128 | **0.199** | 0.188 | — |

指令数减少反而 `vec_time` 上升 0.20 µs、`mte2_wait` 上升 → 跨 dtype UB→UB copy 的单指令
开销明显高于「同 dtype copy + vcast」两条指令之和（疑为其走 VCast 通用路径并额外占用 UB 端口）。
**结论：本工具链上「融合 cast 拷贝」是反向优化，勿再尝试**（附带 repro 见 §7）。

raw profile：`profiles/candidate/round4/r4_fuseup_a7ffb445/{workload}/`。

### 3.4 分支 C `r4_vcastslice`（**编译期否决**，无性能测量）

**单点改动尝试**：把 down-cast 直接写到 `o_stage[0:block_rows, 0:half]` 静态切片
（省掉 2 条 UB→UB merge 拷贝）。结果编译期报错：

```
src/target/codegen_npuir_dev.cc:175
TVMError: Check failed: (*as_const_int(buffer_shape0[i]) == *as_const_int(buffer_shape1[i])) is false
```

→ 复证 **R-8**：v-prefix op 的目标必须是**整 buffer**，静态切片同样非法（不只是动态切片）。
该方向（省 merge 拷贝）在本工具链**结构性不可行**，非参数问题。

### 3.5 平区裁决（T-2 交错 A/B，`ab_test.py`，launch-count=20 / warm-up=5）

对 `r4_prefetch`（唯一单 run 显示增益的候选）执行 5-run 交错 A/B；2k 只做 3-pair 回归确认。

| workload | A（r3，merged median） | B（r4_prefetch） | rel change（B vs A） | pair diffs (B−A, us) | p | 裁决 |
|---|---:|---:|---:|---|---:|---|
| neox-1d-4k-d128-bf16 | **4.73** | 4.86 | **+2.75%（B 更慢）** | −0.08 / +0.14 / +0.25 / +0.27 / −0.12 | 1.00 | **tie**（方向不一致） |
| longrope-1d-8k-d128-bf16 | 6.28 | 6.24 | −0.64% | −0.08 / +0.13 / −0.02 / +0.08 / −0.09 | 1.00 | **tie** |
| neox-1d-2k-d64-fp16（3 pair） | 3.63 | 3.57 | −1.65% | +0.01 / −0.06 / +0.03 | 1.00 | **tie** |

证据：`logs/ab_round4/4k/`、`logs/ab_round4/8k/`、`logs/ab_round4/2k/`（含每 pair 的 raw OPPROF 与 log）。

**判定**：4k 的 A/B 里 B **更慢** 2.75%（3/5 对同向变慢），8k 为 −0.64% 且方向不一致 →
两个主判 workload 均无真实增益；2k tie（无回退）。→ `r4_prefetch` 不采纳。

### 3.6 候选 vs current best 汇总表（B2）

| candidate | workload_id | L0 | TaskDur(us) | AICore 利用率 `vec_ratio` | mte2_ratio | mte3_ratio | mte2_wait | vs current best | raw profile |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| r3 current best（baseline 重测） | neox-1d-4k-d128-bf16 | PASS | 4.89 | 0.486 | 0.377 | 0.112 | 0.674 | 起点 | `profiles/baseline/round4/r3_base_r4_8ee06f17/…` |
| r4_prefetch | neox-1d-4k-d128-bf16 | PASS | 4.76 | 0.529 | 0.329 | 0.115 | 0.514 | +2.7% 单 run → A/B **tie**（实为更慢） | `profiles/candidate/round4/r4_prefetch_02d29f7e/…` |
| r4_fuseup | neox-1d-4k-d128-bf16 | PASS | 5.10 | 0.415 | 0.300 | 0.104 | 0.850 | **−4.3%**（rollback） | `profiles/candidate/round4/r4_fuseup_a7ffb445/…` |
| r3 current best（baseline 重测） | longrope-1d-8k-d128-bf16 | PASS | 6.41 | 0.560 | 0.532 | 0.141 | 0.754 | 起点 | `profiles/baseline/round4/r3_base_r4_8ee06f17/…` |
| r4_prefetch | longrope-1d-8k-d128-bf16 | PASS | 6.30 | 0.591 | 0.445 | 0.137 | 0.638 | +1.7% 单 run → A/B **tie** | `profiles/candidate/round4/r4_prefetch_02d29f7e/…` |
| r4_fuseup | longrope-1d-8k-d128-bf16 | PASS | 7.29 | 0.482 | 0.334 | 0.109 | 0.850 | **−13.7%**（rollback） | `profiles/candidate/round4/r4_fuseup_a7ffb445/…` |
| r4_vcastslice | （不适用） | — | — | — | — | — | — | 编译期否决（R-8 复证） | — |
| **final（= r3，保留）** | 4 个 tune workload 全量 | PASS | 见 §6 | — | — | — | — | 未变 | `profiles/final/round4/final_r3_r4_850d5ad8/…` |

## 4. Autotune Log

未使用 `tilelang.autotune`（本轮不动 tile / block_size：波次几何已在 r3 收敛到整数最优，
且 4k 的 tile 由 `ceil(4096/(2×40))=52` 唯一确定、8k 由 UB clamp 的 69 行唯一确定）。

## 5. 合并非回退检查

本轮**无采纳**，因此不存在合并动作；为满足「改动后必须确认 2k 不回归」，做了两项确认：

1. **结构性**：`r4_prefetch` 的 2k 两档（`iters == 1`）走编译期 `else` 分支，循环体与 r3
   逐字相同 → 不可能有结构性差异；2k-d64 的 3-pair 交错 A/B 实测 **tie**（3.63 vs 3.57）。
2. **最终文件全量复测**：`final_r3_r4` 在 4 个 tune workload 上重采（§6），同一 artifact
   （sha256 `c6098298…`）与 baseline 重测（同为该文件）逐档差异 +0.0% / +2.2% / +1.4% / −2.1%，
   均在 msprof run-to-run 噪声带内，无回退。
3. 精度：`perf_opt/_make_rope_neox_1d.py --level all` → **44/44 PASS，0 test WARN**（rc=0，
   `logs/round4/final_1d_level_all.log`）；`perf_opt/_make_rope_neox_2d.py --level all` →
   **46/46 PASS**（rc=0，`logs/round4/final_2d_level_all.log`，2d 文件 sha256 仍为 `ab22c025…`，
   本轮未改动）。smoke [256,64] fp16/bf16/fp32 随 L0 通过（`max_diff=0.0`），未采性能。

## 6. Final Summary

最终候选：**`final_r3_r4` = r3（不变）**，文件 `perf_opt/_make_rope_neox_1d.py`，
sha256 `c6098298b94beb664c4ff670f33a2b15926838a207a9a6b74b5a1bfb1a504f4a`。
本轮 3 个候选（2 个已测量 + 1 个编译期否决）均未带来超过噪声阈值的增益 →
`no_gain`，r3 保留为 winner。全部 final 记录均在**同一最终候选**上重测
（`perf_records.jsonl` 12 行：4 baseline + 4 candidate（2 候选 × 2 主判 workload）+ 4 final）。

**本轮新结论（三条，均为实测）**：
1. **pipe 已重叠，旋转预取无收益**：`r4_prefetch` 把 `mte2_wait` 0.754→0.638、
   `scalar_vector_stall` 1.214→0.013，但 `vec_time` +0.24 µs、新增 `scalar_mte2_stall` 0.82 µs
   → `aiv_time` 4.35→4.56（更差），4k 的 5-run A/B 甚至更慢 2.75%。**`BP_pipeline_overlap`
   在本结构上再次被证实（第三次）**：不是「没做重叠」，而是「重叠已饱和、等待只是在 pipe 间搬家」。
2. **融合 cast 拷贝是反向优化**：跨 dtype `T.copy` 替代 `copy + vcast`（少 2 条 vector 指令）
   → 8k −13.7%、4k −4.3%，`vec_time` 与 `mte2_wait` 同时上升。**本工具链不要再走这条路**。
3. **v-op 目标必须是整 buffer（R-8 复证）**：静态切片（`o_stage[:, 0:half]`）作为 vcast 目标
   在 codegen 处 shape check 失败 → 「省 merge 拷贝」结构性不可行。

残余空间的量化边界（用于回答「3–5% 在哪」）：本 session 4k = 4.73（A 臂）= 1.7 launch/ramp
+ 3.0 aiv；8k = 6.28 = 1.7 + 4.6 aiv。aiv 内 ~1.8–2.0 µs 是每核骨架（r3 的 `tiny_ub`
10 KB 探针实测），每 iteration 只剩 ~0.56 µs（4k）/ ~0.73 µs（8k）可动；本轮所有「动
iteration 内部」的手段（重叠、指令融合、省拷贝）要么被硬件/编译器代价吃掉，要么被 R-8 挡住。
未触发 `[DESIGN_LIMIT]`：不存在 >2× 的结构性加速空间，也未推翻 DESIGN 的任一假设
（设计流量项与实测全流量口径一致）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox_kernel/_make_rope_neox_1d.py::_make_rope_neox_1d::main | neox-1d-2k-d64-fp16 | 3.61 | 3.61 | final_r3_r4 |
| examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox_kernel/_make_rope_neox_1d.py::_make_rope_neox_1d::main | neox-1d-4k-d128-bf16 | 4.89 | 4.78 | final_r3_r4 |
| examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox_kernel/_make_rope_neox_1d.py::_make_rope_neox_1d::main | longrope-1d-8k-d128-bf16 | 6.41 | 6.32 | final_r3_r4 |
| examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox_kernel/_make_rope_neox_1d.py::_make_rope_neox_1d::main | longrope-1d-2k-d128-fp16 | 3.86 | 3.94 | final_r3_r4 |

逐项：`0.0% / +2.2% / +1.4% / −2.1%`。**注意**：baseline 与 final 是**同一个文件**
（sha256 相同）的两次独立采集，故这些差异是 msprof run-to-run 噪声的实测幅度
（同 session 同 artifact 的 4k 单 run 曾录到 4.78 / 4.89；交错 A/B 的 A 臂中位数为 4.73）。
本轮不存在真值意义上的「提升/回退」。smoke（[256,64] ×3）：仅精度回归，`--level all` PASS；
`perf_records.jsonl` 无 smoke 性能行。skip：无。

## 7. Skill Retrospective

1. **「pipe 未重叠」这个诊断信号不能只看 `aiv_time ≫ max(pipe)`**：4k 的 3.03 vs 1.50 看起来
   像 1.5 µs 的重叠缺口，但旋转预取实测证明缺口主要是**每核骨架（~1.8–2.0 µs）**，不是重叠不足。
   建议在 `iteration-diagnosis.md` 增补判据：先用「同 buffer/tile 的最小工作量探针」量出骨架，
   再算 `(aiv_time − 骨架) / iters` 才是可动的 per-iteration 工作量（本轮 0.56–0.73 µs）。
   - vp_type: **P**；evidence: §3.2 两个 shape 的 `aiv_time`/`vec_time`/stall 对照表 +
     `logs/ab_round4/` 三组 A/B；repro: `_make_rope_neox_1d_r4_prefetch.py`（相对 r3 的单点 diff）+
     `ab_test.py` 命令见 §3.5；toolchain_stamp: Ascend910B3 / CANN 9.0.0 / tilelang dev root /
     2026-11-04。target_doc: `pattern-library/bottleneck-patterns.md`（BP_pipeline_overlap 补第三证）。
2. **跨 dtype `T.copy`（融合 cast）在 npuir Developer 模式的实测代价远高于 copy+vcast**：
   少 2 条指令反而 8k −13.7%。文档只说「自动插入 VCastOp」，没有代价信息；这条实测应进 traps。
   - vp_type: **D**；evidence: §3.3 表（8k 7.29 vs 6.41、`vec_time` 2.47→2.67、
     `mte2_wait` 0.754→0.850）；repro: `_make_rope_neox_1d_r4_fuseup.py`（单点 diff，
     L0 9/9 PASS，`_bench_kernel.py` 直接可测）；toolchain_stamp: 同上。
3. **R-8 的边界要写清楚是「整 buffer，静态切片也不行」**：vcast 目标写
   `o_stage[0:rows, 0:half]` 在 codegen 报 shape check 失败。建议把 R-8 从「动态切片非法」
   扩写为「任何切片非法」。
   - vp_type: **D**；evidence: §3.4 编译错误原文；repro: 三行 diff（见 §3.4 描述，
     复现命令 `python _make_rope_neox_1d_r4_<variant>.py --level L0`）；toolchain_stamp: 同上。
4. **单 run 的 2–3% 「增益」在同一 artifact 上会自己出现**：本轮 `r4_prefetch` 单 run
   +2.7%/+1.7%，交错 A/B 后 4k 反而更慢 2.75%。**只在想采纳时才做 A/B 是不够的**——
   任何 <5% 的单 run 差异都必须先 A/B 再用作决策依据（与 r3 的 retrospective 第 4 条同源，
   本轮再次复现，建议升格为 R 类硬规则）。
   - vp_type: **R**；evidence: §3.2 单 run 数字 vs §3.5 A/B JSON；toolchain_stamp: 同上。
5. **遗留**：4k 的 `tile=52 / 79 tiles / 2 波` 与 8k 的 `69 / 119 / 3 波` 均为整数波次最优；
   若后续工具链支持 FMA（`vfma`/`vmla`）可把 6-op 链降到 4-op（预计 vec 时间 −33%，
   按 §7.1 的骨架模型对总时延影响 ≤4%），或 UB > 192 KB 时 4k 可尝试 1 波（103 行 / 40 tiles）。
   两者当前均**不可行/不适用**，已在 §1.1 记录。

## 8. 门禁自检记录（gate 4）

- `perf_records.jsonl`：12 行（4 baseline `r3_base_r4` / 2 `r4_prefetch` / 2 `r4_fuseup` /
  4 final `final_r3_r4`），每个 tune workload 恰 1 条 final 且 `artifact_sha256` = 最终文件
  实际 sha256（`c6098298…`）；上一轮 12 行已归档 `history/perf_records_r3.jsonl`。
- `workload_inventory.json`：7 条（3 smoke + 4 tune，逐条 `tuning_status`）——8k 保持
  `winner_merged`，其余 3 档 `no_gain`（reason 已补本轮 A/B 证据）。
- 命令：`python .agents/tools/statectl.py gate 4 --dir examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox_kernel`
