# `_engram_decode_kernel` Stage 4 调优日志 — 第 2 轮会话（optimize / mode=full）

- task_id: `TileOPs-_engram_decode_kernel-20261030T121905Z`
- scenario / mode: `optimize` / `full`
- 算子目录: `examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/`
- 基准 kernel（只读，未修改）: `_engram_decode_kernel.py`（sha256 `252c6a80…c7cac4`）
- **本会话 current best 起点**（第 1 轮 winner `r1_op2f`）: `perf_opt/_engram_decode_kernel_r2baseline.py`（sha256 `d30e11a9…d7a234`，与第 1 轮 `perf_opt/_engram_decode_kernel.py` 逐字节相同，仅改名留档）
- 本会话最优版本: `perf_opt/_engram_decode_kernel.py`（sha256 `3835dee0…a08f05`，候选 `r2h1_shift2d`）
- 唯一时延口径: `msprof op` → `OpBasicInfo*.csv` 的 `Task Duration(us)`（median of 20 launches, warm-up 5）
- 唯一优化目标口径: **官方 workload 的 msprof roofline `Ratio(%)`** = `benchmark_base._parse_roofline` 取 `title=="GM/L2"` 且 `bw_name` 含 `GM Read + Write` 的 `ratio*100`
- 性能目标: 官方 workload 平均 Ratio ≥ 60%（用户要求）；噪声阈值 3%；`max_stage4_experiments=36`；`max_rounds=10`
- 环境: Ascend910B3 × 1 die（实查 AIV 40 核 / AI Core 20 核）/ CANN 9.0.0 / tilelang `0.1.2+76e4550 (commit 013dbbf)` / USE_NPUIR=ON

### 0. 历史工件处置（本会话按 conductor 指令重建）

| 处置 | 说明 |
|---|---|
| `perf_opt/history/perf_records_round1.jsonl`、`perf_opt/history/opt_log_round1.md` | 第 1 轮的 `perf_records.jsonl` 与 `opt_log.md` **原样移动**到 `history/`（内容零改动）。原因：gate 4 `S4-PERF-RECORDS-RECON` 要求每个 tune workload 恰有一条 final 记录且 final 指向最终文件当前哈希，第 1 轮 final（`r1_op2f`）会与本轮 final 冲突。 |
| `perf_opt/perf_records.jsonl` | 本会话重建，**baseline = 第 1 轮 winner**（`_engram_decode_kernel_r2baseline.py`, sha `d30e11a9…`），append-only。 |
| `perf_opt/workload_inventory.json` | 重建（4 workload：1 smoke + 3 tune）。 |
| `perf_opt/bench_driver.py`、`run_prof.sh`、`parse_prof.py`、`probe/`、`profiles/`、`logs/` | 重建的采集脚手架（`parse_prof.py` 复刻 `_parse_roofline` 的 GM/L2 + `GM Read + Write` 取数，用于逐次采集自核算 Ratio）。 |

**口径校验**：本套管线在 current best 上复现官方报告（`reports/tileops/20261016_210815_236774_EngramDecodeOp/`）的 Ratio，偏差 <2%：

| workload | 官方 latency_us / Ratio | 本管线 latency_us / Ratio |
|---|---:|---:|
| decode-smoke-b1-dmem256-d256-fp16 | 21.70 / 0.7173% | 20.700 / 0.7480% |
| decode-b1-dmem512-d256-fp16 | 31.48 / 0.9728% | 30.181 / 1.0123% |
| decode-b4-dmem1024-d512-fp16 | 78.80 / 3.6182% | 78.402 / 3.6366% |
| decode-b8-dmem512-d256-bf16 | 31.38 / 4.7397% | 31.311 / 4.7505% |

（官方数略高 1–4%，差异来自官方走 benchmark wrapper 的 padding/host 侧路径，属同口径可接受偏差。）

---

## 1. Performance Test Data

### 1.1 workload inventory 与分类（Phase 1）

benchmark 入口 `examples/TileOPs/benchmarks/ops/bench_engram_decode.py::test_engram_decode_bench` 由 `_engram_decode_params()` 直读 `tileops/manifest/sequence_modeling.yaml::EngramDecodeOp.workloads`，展开 4 个参数化案例（8 张量输入，不能用 `workloads_to_params`）：

| kernel_id | workload_id | marks | kind | 说明 |
|---|---|---|---|---|
| `..._engram_decode_kernel.py::_engram_decode_kernel::main` | decode-smoke-b1-dmem256-d256-fp16 | `smoke` | **smoke** | label 含独立 `smoke` 词 → 仅精度回归，不采性能、不选 winner、不写性能记录 |
| 同上 | decode-b1-dmem512-d256-fp16 | `full` | **tune** | 单 block（grid=batch=1），测核内带宽天花板 |
| 同上 | decode-b4-dmem1024-d512-fp16 | `full` | **tune** | 关注 shape，绝对耗时最大 |
| 同上 | decode-b8-dmem512-d256-bf16 | `full` | **tune** | 唯一 bf16 路径 |

无 skipped 案例。`target_kernel_name = main`（`msprof op --kernel-name=main`，`captured_op_name` 实查为 `main`、`Op Type=vector`）。

### 1.2 tune workload baseline（current best = 第 1 轮 winner，`d30e11a9…`，launch-count=20）

| kernel_id | workload_id | candidate_id | target / captured | task_duration_us | Ratio(%) | profile_status | raw_profile_dir |
|---|---|---|---:|---:|---|---|
| `…::_engram_decode_kernel::main` | decode-b1-dmem512-d256-fp16 | r1_op2f | main / main | 30.181 | 1.0123 | valid | `profiles/baseline/decode-b1-dmem512-d256-fp16_baseline/OPPROF_20261030205222_UUNSEPQZFYZJKBYW` |
| 同上 | decode-b4-dmem1024-d512-fp16 | r1_op2f | main / main | 78.402 | 3.6366 | valid | `profiles/baseline/decode-b4-dmem1024-d512-fp16_baseline/OPPROF_20261030205345_DTATLHOHQHMBKYPM` |
| 同上 | decode-b8-dmem512-d256-bf16 | r1_op2f | main / main | 31.311 | 4.7505 | valid | `profiles/baseline/decode-b8-dmem512-d256-bf16_baseline/OPPROF_20261030205507_XBVTOGIPQCAPSOXM` |

命令模板（逐 workload 串行，`--aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio,Roofline`——**必须含 `Roofline`**，否则不产出 `visualize_data.bin`，见 `queue.md::VP-2026-0081`）：

```bash
msprof op --kernel-name=main --output=profiles/baseline/<wl>_baseline --launch-count=20 --warm-up=5 \
  --aic-metrics=...,Roofline python bench_driver.py --module _engram_decode_kernel_r2baseline.py --case <wl> --iters 30
```

### 1.3 smoke 排除

`decode-smoke-b1-dmem256-d256-fp16` 归 smoke：不采性能 baseline、不选 winner、不参加性能回退门禁；仅在最终做精度回归（§5.3，L0 通过，`y_maxdiff=2.441e-04`）。

### 1.4 调优顺序与理由

1. **decode-b4-dmem1024-d512-fp16**（关注 shape、绝对耗时 78.4 µs 最大、单 block 工作量最大）→ 结构性收益最高，先诊断。
2. **decode-b1-dmem512-d256-fp16**（单 block，迭代快，用于候选分支快速排序）。
3. **decode-b8-dmem512-d256-bf16**（唯一 bf16，验证 bf16 路径不回退）。

---

## 2. Iteration Log

### Round 2 — 现象、H1 反解与候选

#### 2.1 Diagnostic Context（base = `d30e11a9…`，最新 profile 见 §1.2）

| workload | Block Dim | Task Duration(us) | `aiv_time` | `aiv_vec_time`/ratio | `aiv_mte2_time`/ratio/active_bw | `aiv_scalar_time`/ratio | `read_main_memory_datas` | `aiv_vec_fops` |
|---|---:|---:|---:|---|---|---|---:|---:|
| b1 dmem512 d256 | 1 | 30.181 | 29.638 | 17.619 / 0.594 | 20.432 / 0.689 / 24.62 GB/s | 19.746 / 0.668 | 527.75 KB | 827 616 |
| b4 dmem1024 d512 | 4 | 78.402 | 77.634 | 56.436 / 0.727 | 55.212 / 0.711 / 36.05 GB/s | 40.665 / 0.524 | 2087.25 KB | 3 227 712 |
| b8 dmem512 d256 | 8 | 31.311 | 30.454 | 17.624 / 0.579 | 19.421 / 0.636 / 26.05 GB/s | 19.804 / 0.651 | 530.75 KB | 828 128 |

**P4（新现象）**：三条 pipe 同时 ~60–70% 占用（`RoofLine Summary Report: latency bound: pipeline caused`）→ 不是单 pipe 饱和，而是 **pipe 间重叠不足 + 每 op 组固定开销**。

**P5（H1 直接证据，最重要）**：b1（1 block）与 b8（8 block）**每 block 工作量相同**（d_mem=512, d=256），实测 `Task Duration` 30.18 vs 31.31 µs —— **时间几乎不随 block 数变化** → 时间 = 单 block 工作量时间，block 之间并行良好、无跨核争用，但也**没有额外核可用**。核利用率 = batch/40：b1 **2.5%**、b4 **10%**、b8 **20%**。

#### 2.2 H1 反解：实测常数（probe 均为 `msprof op` 口径，`probe/h1_probe.py`）

| 探针 | 配置 | Task Duration | 反解常数 |
|---|---|---:|---|
| `floor_nblk1_nch1` | 1 block，512 B GM→UB→GM | **1.89 µs** | **kernel launch/teardown 地板 ≈1.9 µs**（`aiv_time`=1.36 µs） |
| `floor_nblk40_nch1` | 40 block，同工作量 | 3.36 µs | 40 block 空载 ≈3.4 µs |
| `conv_c4096_x64` | 1 block，64 次 8 KB（fp16→fp32）共 512 KB | 13.97 µs | 每次 DMA 固定开销 **≈0.16 µs**（3 点拟合） |
| `conv_c8192_x32` | 32 次 16 KB，共 512 KB | 9.38 µs | 同上（64 次 →32 次，省 4.6 µs） |
| `conv_c16384_x16` | 16 次 32 KB，共 512 KB | 6.65 µs | 同上（→16 次，再省 2.7 µs） |
| `read_nblk1_nch8` | 1 block，8×32 KB 流式读 | 4.91 µs | **per-core 流式 GM 读 ≈68 GB/s**（`mte2_active_bw`） |
| `convert_nblk1_nch8` | 同形态、fp16→fp32 | 3.61 µs | 62 GB/s；**dtype 转换不额外收费** |

> 未文档化假设：由 3 点拟合外推「L2 命中时 per-core 有效读带宽 ≈285 GB/s」，用于 §4 的绝对上界估计，不作为当前结构可实现值。

**Ablation（单变量，真实 kernel 内做，遵循第 1 轮经验）**：b1 全 kernel 30.18 µs / **GEMV-only 26.12 µs**（删尾部）→ 尾部 4.06 µs；b4 全 78.40 / **GEMV-only 74.07 µs** → 尾部 ≈4.3 µs（且尾部可与 GEMV 重叠）。→ **GEMV 占 80–94% 时间**，与第 1 轮结论一致。

**H1 结论（量化）**：
1. b1 的 Ratio 恒等于「per-core 有效 GM 带宽 / 芯片峰值带宽」：527.75 KB / 28.41 µs = **19.0 GB/s** → 19.0/1750 = **1.09%**，实测 1.075% ✓（`msprof` roofline `bw`=1.75 TB/s）。**单核最多能做到 ~68 GB/s（实测流式）→ b1 Ratio 上限 ≈3.9%**。
2. b4 在 roofline 上是**算力受限**（AI=5.886 > computility/bw = 4.528/1.75 = 2.59）→ Ratio ≈ 已用核数/40 × 单核向量效率。4/40 = 10% 为核数比例上限，实测 3.87% = 39% 占比。
3. b8 同为算力受限，8/40 = 20% 比例上限，实测 5.21% = 26% 占比。
4. **H1 成立并是主导天花板**，但它的修法（split-d / split-K）需要跨核归约，见 §2.3。

#### 2.3 跨核归约可行性核查（决定 H1 能否落地）

| 手段 | 核查结论 | 依据 |
|---|---|---|
| `T.sync_block_set/wait` | **不可用于跨 AI Core barrier**：文档明确定义为「Block 内部的同步」，示例是同一 AI Core 内 Cube↔Vector 子块（FFTS lock），且 flag id 预算有限 | `docs/Tilelang.language/同步管道操作/T.sync_block_wait.md` §1/§2.4；`CONST-flag-id-budget`（待重验） |
| GM workspace 跨核暂存 | **前端无可用入口**：IR 里确有 `#hacc.arg_type<workspace>` + `WorkspaceArgIdx`，但只能由 launcher 自动注入，Python 侧无 `T.alloc_*` 绑定；`TILELANG_ASCEND_WORKSPACE_SIZE` 仅改 size | `docs/Tilelang算子调试指南.md` §3.2.1（IR 形态）；`tilelang/jit/jit_npu.py:1449-1510,1593-1606`；**标注为未文档化假设（无前端 API）** |
| `T.atomic_add` 到 GM | 原子加**可用**，但**没有跨 block 完成栅栏**，无法在同一 launch 内安全读取累加结果 | `docs/Tilelang.language/原子操作/T.atomic_add.md` §2（shape 无限制、Ascend 支持 fp16/fp32） |
| 用输出张量当 scratch + 双 barrier | 需要 barrier 才有意义 → 回到第 1 条不可用 | — |

→ **d 维切分 / K 维切分（split-N / split-K）在「不改 op/wrapper 契约」前提下不可实现**（唯一缺失能力 = 跨 AI Core 归约/栅栏）。按调度 prompt 要求，本轮只做**可行性论证 + 收益估计**（§4.3），不实施。

#### 2.4 Candidate Optimization Points（round 2）

| opt_id | 目标现象 | 优化点 | 判断依据 | 状态 |
|---|---|---|---|---|
| R2-A | P4：MTE2 每 op 组 0.16 µs 固定开销（64 次 8 KB DMA） | slab 暂存：`KS=32` 一次搬 16 KB，内层 `T.serial(2)` 复用（BK=16 计算块不变） | §2.2 三点拟合 | **no_gain/回退**（b1 31.42、b4 107.16） |
| R2-B | 同上（更激进：去掉 32 次 32 B 的 `e_n` 小 DMA） | 直接用 `e_local[k0*BK+kk]` 做行广播，省掉每轮两次小 copy | 32 次小 DMA ≈5 µs | **invalid**（结果错误 `y_maxdiff=1.49`：affine 索引广播被错降） |
| R2-C | UB 容量（BK=16 受 UB 限制） | `BK=32` + 释放 UB（删死代码 `e_local`、复用 4 个 `(1,d)` 归约缓冲） | UB 峰值 = 2×(32,d) fp32 + multi-buffer | **blocked**（BiShengIR `ub overflow, requires 1606656 bits while 1572864 bits available`） |
| R2-D/E | 向量效率 ~50% | fp16 暂存 + fp16 vmul + **fp16→fp32 混合归约**（指令数减半） | 期望 vec_time 腰斩 | **blocked**（编译期 `memref.expand_shape ... found 1, expected 2`；且最小探针实测 fp16→fp32 混合 reduce **静默算错**，fp16→fp16 对但精度仅 ~1%，bf16 无 reduce 支持） |
| R2-G | 恢复 R2-A 丢掉的 MTE2/VEC 重叠 | slab 手动展开 2 个计算块（去掉内层 serial 循环） | slab 在 b4 把 mte2 55→32 µs，说明搬运确实变快 | **no_gain**（b1 31.13、b4 97.50、b8 32.54） |
| R2-H1 | 尾部 ~4 µs 里 11–19 行 state-shift 的逐行小 DMA | `new_conv_state` shift 改为 **2 次 2-D 区域整块搬运**（`(max_conv_len-1, d_padded)`） | 每行 2 次小 DMA × 11/19/17 行 ≈ 0.16 µs/次 | **improved（胜出）** |
| R2-H2 | 同类：合并 `conv_w` 3 行小 copy + 删死代码 `e_local` | 在 H1 上加 conv_w 整块 copy | 同类优化点延伸 | config_no_gain（与 H1 打平，<0.2%） |

#### 2.5 本轮「候选 vs current best」对比表（B2 结构化回流）

base = current best `r1_op2f`（`d30e11a9…`）。所有分支从同一 base 派生，每分支只改一个主要优化点。筛选采集 `launch-count=10`，baseline/final `launch-count=20`。

| branch (candidate_id) | opt_id | 主要改动 | L0 | Task Duration(us) b1 / b4 / b8 | AICore 利用率（b1 vec/mte2/scalar ratio） | memory 指标（per-block GM→UB） | 结论 |
|---|---|---|---:|---|---|---:|---|
| （base）`r1_op2f` | — | 第 1 轮 K 分块 vreduce，BK=16 | pass | 30.181 / 78.402 / 31.311 | 0.594 / 0.689 / 0.668 | 527.5 KB | 起点 |
| `r2a2_slab32` | R2-A | slab KS=32 + 内层 serial | pass | **31.421 / 107.162** / — | 0.610 / 0.531 / 0.729 | 527.5 KB（mte2_time 55→32 @b4） | 回退（重叠丢失） |
| `r2b_elocal` | R2-B | `e_local[k0*BK+kk]` 广播 | **fail** | 结果错误（未采性能） | — | — | invalid |
| `r2c_bk32` | R2-C | BK=32 + 释 UB | — | 编译失败（UB overflow） | — | — | blocked |
| `r2d_h16` / `r2e_h16only` | R2-D/E | fp16 暂存 + 混合 reduce | — | 编译失败（expand_shape） | — | — | blocked |
| `r2g_slab32unroll` | R2-G | slab 手动展开 | pass | **31.131 / 97.502 / 32.541** | — | mte2 变快但总时间变差 | 回退 |
| **`r2h1_shift2d`** | **R2-H1** | **state-shift 整块 2-D copy** | pass | **28.551 / 73.561 / 28.631** | **0.630 / 0.679 / 0.705** | 527.5 KB（不变） | **improved（胜出）** |
| `r2h2_shift2d_cw` | R2-H2 | H1 + conv_w 整块 copy + 删 `e_local` | pass | 28.581 / 73.631 / 28.661 | — | 527.5 KB | config_no_gain（<0.2%） |

**R2-A/R2-G 的机制归因（重要，反直觉）**：slab 确实把 b4 的 `aiv_mte2_time` 从 55.2 → 31.9 µs（`mte2_active_bw` 36→63 GB/s），但 `Task Duration` 反而 78.4 → 107.2 µs。原因：当前 kernel 的 78.4 µs ≪ mte2 55.2 + vec 56.4（=111.6），说明 MTE2 与 VEC **已经在重叠**；而放大暂存块把 UB 占用从 `2×(16,d)` 推到 `2×(32,d)`，**挤掉了 auto multi-buffer**（`--enable-auto-multi-buffer=true` 需为被消费 buffer 再留一份），重叠消失、退化为两 pipe 串行。→ **UB=192 KB 下 BK=16 是本结构的 overlap 最优点**，与第 1 轮 BK 扫描结论一致。

#### 2.6 核内合并检查（winner = R2-H1）

R2-H1 是无条件结构性替换（无 shape/dtype 分支），直接合入 `perf_opt/_engram_decode_kernel.py`，并在合并版上复测本 kernel 全部 tune workload：

| merged_candidate | workload_id | current_best_us | merged_us | Δ | 精度 | status |
|---|---|---:|---:|---:|---|---|
| `perf_opt/_engram_decode_kernel.py`（= r2h1） | decode-b1-dmem512-d256-fp16 | 30.181 | **28.411** | −5.86% | L0 pass | pass |
| 同上 | decode-b4-dmem1024-d512-fp16 | 78.402 | **73.591** | −6.14% | L0 pass | pass |
| 同上 | decode-b8-dmem512-d256-bf16 | 31.311 | **28.571** | −8.75% | L0 pass | pass |
| 同上 | decode-smoke-b1-dmem256-d256-fp16（smoke，仅精度） | — | 不采集 | — | L0 pass（`y_maxdiff=2.441e-04`） | pass |

无回退 → 更新 current best。`state-shift` 的**逐位一致**由 L1 `_check_state_shift` 保证（`max_err=0.0e+00`）——这正是 H1 改动的关键正确性约束，实测通过。

#### 2.7 Iteration Winner

- winner: `perf_opt/_engram_decode_kernel.py`（来源 `_engram_decode_kernel_r2h_shift2d.py`，candidate_id `r2h1_shift2d`）
- 三个 tune workload 全部优于 base（−5.9% / −6.1% / −8.8%），全部超过 3% 噪声阈值；H2 与 H1 差异 <0.2%（噪声内）→ 取改动更小的 H1。
- 未采纳：R2-A/R2-G（MTE2 变快但丢失 overlap，整体回退）、R2-B（错误）、R2-C/D/E（编译/工具链阻塞）、R2-H2（打平）。

---

## 3. Autotune Log

未使用 autotune 框架。本轮的参数面只有 `BK` 与 `KS`：

| 结构 | 搜索空间 | 实测 best | 备注 |
|---|---|---|---|
| BK（第 1 轮已扫 + 本轮复核） | {8,16,32,64} | **BK=16** | BK=32/64 在 d=512 触发 UB overflow（本轮 R2-C 复核：`requires 1606656 bits while 1572864 bits available`）；BK=16 是唯一同时满足两个 d 档与 multi-buffer 重叠的取值 |
| KS（slab 搬运块，R2-A/R2-G） | {16(基线), 32} | **16（即不用 slab）** | 32 使 UB 翻倍 → 丢失 MTE2/VEC overlap |
| state-shift 搬运形态（R2-H1/H2） | {逐行 2 copy, 整块 2 copy, 整块+conv_w 合并} | **整块 2 copy** | 见 §2.5 |

winner（H1）的 `msprof op` 复测：`launch-count=10` 28.551 / 73.561 / 28.631 与 `launch-count=20` final 28.411 / 73.591 / 28.571，偏差 ≤0.5%，落在噪声内。

---

## 4. 设计估算 vs 实测、可行 Ratio 天花板与 `[DESIGN_LIMIT]` 判定

### 4.1 D-2 回填（设计估算 vs 实测偏差）

`DESIGN.md` §1.6.0 的 D-2 估算：W3（b4,d_mem1024,d512）下界 ≈25 µs、W1 ≈12 µs。

| workload | 设计估算下界 | 第 1 轮 baseline | 本轮 final | 与估算偏差 |
|---|---:|---:|---:|---:|
| b1 dmem512 d256 | ≈12 µs | 5988.77 µs | 28.41 µs | **2.4×** |
| b4 dmem1024 d512 | ≈25 µs | 23668.15 µs | 73.59 µs | **2.9×** |

失准项归因（更新第 1 轮结论）：D-2 未计入 **(a) 单核 GM 带宽上限（实测 62–68 GB/s，非芯片 1.75 TB/s）**、**(b) 每 DMA 指令 0.16 µs 固定开销（64–128 次/block）**、**(c) 单核向量效率 ~47–59 Gflop/s（峰值 111 Gflop/s）**。三者都是「单核」量级，与 DESIGN §5.5 已显式记录的「B≤8 时仅 8/20 核工作」一致。

### 4.2 逐 workload 可行 Ratio 天花板（H1 实测反解）

「本会话实测最好 Ratio」= final（`3835dee0…`）。上界分三档给出，全部由本轮实测常数推导：

| workload | 当前 Ratio（base） | 本会话实测最好 Ratio（final） | ① 结构冻结·完美 pipe 重叠上界 | ② 结构冻结·绝对上界（100% 单核效率、零 DMA 冲突） | ③ 假设有跨核归约的重设计上界（不可实现，仅估计） |
|---|---:|---:|---|---|---|
| b1 dmem512 d256 | 1.0123% | **1.0753%** | **1.56%**（= ideal 0.306 µs ÷ max(vec 17.57, mte2 18.94, scalar 19.66 µs)） | **≈3.9%**（527 KB ÷ 实测单核 68 GB/s = 7.9 µs）…上限区间 3.9–16%（若 L2 命中按 ~285 GB/s 外推） | **≈14%**（40 block，DMA 0.3 µs + launch 1.9 µs ≈ 2.2 µs） |
| b4 dmem1024 d512 | 3.6366% | **3.8742%** | **5.20%**（max(vec 54.78, mte2 52.12, scalar 40.28) = 54.78 µs） | **≈10%**（核数比例 4/40，roofline 算力受限） | **≈53%**（32 block：DMA 1.6 µs + 算力 3.6 µs + launch） |
| b8 dmem512 d256 | 4.7505% | **5.2055%** | **7.53%**（max(vec 17.57, mte2 15.86, scalar 19.74) = 19.74 µs） | **≈20%**（核数比例 8/40） | **≈50–57%**（32 block） |
| **tune 平均** | **3.1331%** | **3.3850%** | **≈4.76%** | **≈11.3%** | **≈39–41%** |

推导要点：
- ①「完美重叠上界」= 用同一 final profile 的三条 pipe 时间取 max（三 pipe 完全重叠的理论极限），是**当前结构族不改代码形态的硬上限**。
- ②「绝对上界」= 只用 launch 地板（1.89 µs）+ per-core 带宽/向量峰值 + 核数比例，**忽略 DMA 指令开销与重叠损失**，因此不可达到，只作为反证 60% 不可能的量级依据。
- ③「重设计上界」= 假设 split-d（每 batch 8 个 block，共 32 block）+ 跨核标量归约**免费**，用实测 per-core 常数外推。**该能力在前端不存在（§2.3），不可实施**。

### 4.3 「平均 Ratio ≥ 60% 是否可达」裁决

**不可达。** 三条独立证据：

1. **b1 单核带宽硬约束（决定性）**：b1 的 roofline ideal 时间只有 **0.306 µs**，而实测 **kernel launch 地板 = 1.89 µs**（1 block、512 B 搬运的空 kernel）。仅此一条就把 b1 的 Ratio 压在 **≤16.2%**（0.306/1.89），与代码好坏无关。要拿到 60% 平均，b1 需 ≤0.51 µs —— **低于本设备的空 kernel 地板**，物理上不可达。
2. **核数比例约束**：grid = batch（DESIGN §5.5 的分核方案），b4/b8 在 roofline 上是算力受限，Ratio ≈ 核数比例 × 单核向量效率 = (4/40, 8/40) × ~50% → 5%/10% 量级。本轮已把单核向量效率从 ~42% 提到 ~53%（b4 counted fops 58.9 Gflop/s ÷ 111 Gflop/s），剩余空间受 UB=192 KB 限制（放大 tile → 丢 multi-buffer 重叠，R2-A/R2-G 实测回退）。
3. **多核重设计也不够**：即使实现当前工具链不支持的跨核归约、把 b4/b8 打满 32 block（③ 档，b4≈53%、b8≈50–57%），b1 仍被 1.89 µs 地板钉在 ~14%，**平均上界 ≈39–41%**，仍到不了 60%。

→ 在不改 op 契约前提下的**可达到上界 = ①档 ≈4.8%（tune 平均）**；本会话已达 3.39%，即已取到该上界的 **71%**。

### 4.4 `[DESIGN_LIMIT]` 判定（`_shared/standards/perf-feedback.md` §1，两项须同时满足）

1. **①「性能天花板由算法/设计层决定」→ 不满足（部分成立但归因不成立）**：低核利用这一事实在 `DESIGN.md` §5.5 已被**显式记录并主动接受**（「B ≤ 8 时仅 8/20 核工作 … 固有特征（低并行度、延迟导向）」），且 split-N/split-K 已被设计主动列为「**Stage 4 结构级备选**」。本轮执行了该备选：结论是**能力缺口**（前端无跨 AI Core 归约/栅栏），不是设计选型错误。真正的天花板来源是硬件/工具链（launch 地板 1.89 µs、单核 68 GB/s、UB 192 KB、fp16 reduce 混合 dtype 不可用），不归因于 DESIGN 的具体假设。
2. **②「结构性加速估计 > 2x」→ 形式上成立但不产生设计层可执行动作**：③ 档外推确有 4.5–14× 量级，但其唯一实现路径（跨核归约）**当前前端不可得**，因此「改设计」也无法落地，收益不是设计修订能兑现的（属 `capability-gaps` 而非 DESIGN 假设错误）。

→ **不产出 `perf_opt/perf_feedback.md`；不返回 `[DESIGN_LIMIT]`。** 上述能力缺口以 vp 形式记入 §6（交任务终态 evolver 蒸馏进 `.agents/evolution/capability-gaps.md`），并按用户要求把「60% 不可达 + 可达到上界」写入本日志供 conductor 在最终报告中如实披露。

参照锚定（T-1 要求）：检索了同族/同类实现——`examples/engram/engram_decode.py`（源 GPU 形态，同 grid=(B,)）、`examples/norm/rms_norm.py`（单遍行归约）、`examples/gemv/example_gemv.py`（Cube GEMV，第 1 轮已实测因 CG-2026-0022 的 MIX 形态不稳定而弃用）。**未找到**任何「单 block 内可跨核归约」或「batch=1 时用多核拆 GEMV」的仓内先例 → 结论置信度按标准降级标注，但 §4.3 的 1.89 µs 地板反证不依赖任何先例。

---

## 5. Final Summary

### 5.1 相对 baseline 的改进（`msprof op Task Duration`，launch-count=20）

| workload_id | baseline_us（r1_op2f） | final_us（r2h1_shift2d） | 降幅 | baseline Ratio | final Ratio | Ratio 提升 |
|---|---:|---:|---:|---:|---:|---:|
| decode-b1-dmem512-d256-fp16 | 30.181 | **28.411** | −5.86% | 1.0123% | **1.0753%** | +6.2% |
| decode-b4-dmem1024-d512-fp16 | 78.402 | **73.591** | −6.14% | 3.6366% | **3.8742%** | +6.5% |
| decode-b8-dmem512-d256-bf16 | 31.311 | **28.571** | −8.75% | 4.7505% | **5.2055%** | +9.6% |
| **tune 平均** | — | — | — | **3.1331%** | **3.3850%** | **+8.0%** |

（相对**迁移期原始 baseline** 的累计提升：b1 5988.77→28.41 µs = 210.8×；b4 23668.15→73.59 µs = 321.6×；b8 5988.63→28.57 µs = 209.6×。）

## Final Performance Test Data

`kernel_id = examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/_engram_decode_kernel.py::_engram_decode_kernel::main`；`final_candidate_id = r2h1_shift2d`；final 时延在最终文件 `perf_opt/_engram_decode_kernel.py`（sha256 `3835dee05f105bdc7f6d94eded0623fa812cfe3d960d34244c67434ce6a08f05`）上以 launch-count=20、warm-up=5 重新采集，与 `perf_records.jsonl` 的 `phase=final` 行逐项一致。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/_engram_decode_kernel.py::_engram_decode_kernel::main | decode-b1-dmem512-d256-fp16 | 30.181 | 28.411 | r2h1_shift2d |
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/_engram_decode_kernel.py::_engram_decode_kernel::main | decode-b4-dmem1024-d512-fp16 | 78.402 | 73.591 | r2h1_shift2d |
| examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/_engram_decode_kernel.py::_engram_decode_kernel::main | decode-b8-dmem512-d256-bf16 | 31.311 | 28.571 | r2h1_shift2d |

### 5.2 profile 证据（final，`profiles/final/`）

| workload | raw_profile_dir | Task Duration | Block Dim | `aiv_vec_time`/ratio | `aiv_mte2_time`/ratio/active_bw | `aiv_scalar_time`/ratio | roofline（bw / AI / computility / ratio） |
|---|---|---:|---:|---|---|---|---|
| b1 dmem512 d256 | `profiles/final/decode-b1-dmem512-d256-fp16_final/OPPROF_20261030214801_LZPKRQENJAELEZEN` | 28.411 | 1 | 17.566 / 0.630 | 18.936 / 0.679 / 26.57 GB/s | 19.662 / 0.705 | 1.75 TB/s / 1.548 / 4.454 TOPS / 1.0753% |
| b4 dmem1024 d512 | `profiles/final/decode-b4-dmem1024-d512-fp16_final/OPPROF_20261030214933_POPUDVADLLJDCPMT` | 73.591 | 4 | 54.783 / 0.752 | 52.121 / 0.716 / 38.19 GB/s | 40.281 / 0.553 | 1.75 TB/s / 5.886 / 4.528 TOPS / 3.8742% |
| b8 dmem512 d256 | `profiles/final/decode-b8-dmem512-d256-bf16_final/OPPROF_20261030215107_PXWTAIJZVPESCMCN` | 28.571 | 8 | 17.567 / 0.633 | 15.864 / 0.570 / 31.89 GB/s | 19.736 / 0.711 | 1.75 TB/s / 10.871 / 4.454 TOPS / 5.2055% |

（对照 baseline：b1 vec/mte2/scalar = 0.594/0.689/0.668、mte2_time 20.432；b8 mte2_time 19.421；b4 vec 56.436 / mte2 55.212。）

### 5.3 smoke / skip 排除与精度

| workload | kind | 处理 | 最终精度 |
|---|---|---|---|
| decode-smoke-b1-dmem256-d256-fp16 | smoke | 仅精度回归（不采性能、不选 winner、不写性能记录、不进门禁） | **pass**：`y_maxdiff=2.441e-04, st_maxdiff=1.953e-03`（与 baseline 逐位一致） |

无 skipped 案例。

### 5.4 最终精度回归（`python perf_opt/_engram_decode_kernel.py --level all`，日志 `logs/final/L_all.log`）

- **L0（阻塞）PASS**：4/4 manifest 用例与 baseline **逐位一致**（`2.441e-04 / 1.221e-04 / 9.766e-04 / 7.812e-03`）
- **L1（阻塞）PASS**：`L1-multistep`(17 步) / `L1-bf16` / **`state-shift bit-exact (max_err=0.0e+00)`** / `gate-corner finite` / `L1-dil3` / `L1-dil5` / `L1-seed5` 全过 —— 其中 state-shift 逐位一致正是本轮 H1 改动的关键验证
- L2（告警不阻塞）：`L2-fp32` 告警（**继承 baseline 的既有 fp32 GM 输入不可编译限制**，BiShengIR `Failed to run BiShengHIR pipeline`，非本轮回退）；`L2-seed101/202` 通过
- Boundary（告警不阻塞）：`B-minL` / `B-w2` / `B-w1` / `B-b1` 全过

### 5.5 总体结论与中止原因

- 本轮把 `new_conv_state` shift 从「逐行 2 次小 DMA」改为「整块 2-D 两次搬运」，三个 tune workload 分别再降 **5.86% / 6.14% / 8.75%**，Ratio 平均 3.13% → **3.39%**；精度逐位不变。
- **性能目标（平均 Ratio ≥ 60%）不可达**，见 §4.3：b1 的 roofline ideal 0.306 µs 低于本设备 1.89 µs 空 kernel 地板，单这条即封死；即使加上当前工具链不支持的跨核归约重设计，平均上界也只有 ~40%。
- 中止原因：**blocked**（剩余候选方向已被实测/文档封死）：
  1. GEMV 单核结构：`BK>16` 在 d=512 触发 UB overflow，`slab` 放大暂存块会挤掉 auto multi-buffer 导致 MTE2/VEC 失去重叠（两次实测回退）；
  2. 向量半精度加速：fp16→fp32 混合 `reduce_sum` 编译报 `memref.expand_shape` 且最小探针实测**静默算错**，bf16 无 `reduce_sum` 支持（文档 §2.2.1）→ 该路线在当前工具链不可用；
  3. 多核切分：跨 AI Core 归约/栅栏在前端不存在（§2.3），超出 optimize 场景边界。
- 遗留（非阻塞，交后续）：b1 的 `aiv_scalar_time`(19.66 µs) 已成为最长 pipe，主要来自 64 次 8 KB `T.copy` + 64 次 vmul/vreduce 的逐 op 组标量开销；要再降需 (a) 跨核归约能力 或 (b) 修好 fp16 混合 dtype reduce 的 codegen。

---

## 6. Skill Retrospective

### 6.1 Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Optimize-flow | **「核利用率不足」类假设缺一条「空 kernel launch 地板」测法**：判断 Ratio 天花板是「单核算不动」还是「多核没用到」时，必须先测 launch 地板，否则会误判可优化空间。本轮 `probe/h1_probe.py --mode floor` 一次即得 1.89 µs，直接给出 b1 ≤16% 的结论 | §2.2、§4.3 | `references/iteration-diagnosis.md` 增补：「当 grid < 物理核数且 Task Duration < 50 µs 时，先测空 kernel launch 地板（1 block 512 B copy），用 `roofline_ideal / floor` 给出单 workload Ratio 硬上限，再决定是否投入 split 改造」 | R |
| Optimize-flow | **「减少 DMA 指令数」这条常规优化在本工具链上会反噬**：放大暂存块 → UB 翻倍 → auto multi-buffer 被挤掉 → MTE2/VEC 失去重叠，总时间反而变差 25–37%。skill 未给出「先确认 pipe 是否已重叠」的前置判据 | §2.5 R2-A/R2-G（b4 mte2 55→32 µs 但总 78→107 µs） | `references/bottleneck-patterns.md` 新增反例条目：「`aiv_mte2_time + aiv_vec_time > Task Duration` 表示两 pipe 已在重叠；此时单独降低 mte2 不会降低 Task Duration，反而可能因 UB 压力丢失 multi-buffer」 | P |
| Optimize-flow | **混合 dtype 归约的「静默算错」风险**未在陷阱库中登记：fp16 src → fp32 dst 的 `T.reduce_sum` **编译通过、无告警、结果错 3 个数量级**；fp16→fp16 才正确 | `/tmp/opencode/t_mixred.py`（fp16→fp32 输出 abs-sum 3.73e6 vs fp16→fp16 886）；`_engram_decode_kernel_r2e_h16only.py` 编译报 `memref.expand_shape ... found 1, expected 2` | `pattern-library/traps-runtime.md` 新增条目（D 类，附 repro）：`T.reduce_sum` 的 src/dst dtype 必须一致（文档 §2.2.1 未写明但实际如此），混合 dtype 会静默产出错误值 | D |
| Optimize-flow | **affine 索引做行广播会被错误降级**：`prod[kk,j] = w[kk,j] * e_local[k0*BK+kk]` 编译通过但结果错（`y_maxdiff=1.49`）；只有「每轮现 copy 成 `(BK,)` 小 buffer + `e_n[kk]`」才正确 | §2.5 R2-B；`_engram_decode_kernel_r2b_elocal.py` | `pattern-library/traps-compiler.md` 新增条目（含 repro）：共享内存 buffer 的行广播索引必须是「平行维自身」的纯函数，含串行循环变量的 affine 组合索引会静默取错地址 | D |

### 6.2 Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **Ascend910B3 单 AIV core 的 kernel launch 地板 = 1.89 µs（1 block，512 B copy；`aiv_time` 1.36 µs）**；40 block 空载 3.36 µs。任何 `roofline_ideal_time < 2 µs` 的 workload，其 roofline Ratio 上限 < 16%，与实现质量无关 | D | §2.2 `profiles/probe/floor_nblk1_nch1`、`floor_nblk40_nch1` | `perf_opt/probe/h1_probe.py --mode floor --nblk 1` | tilelang 0.1.2+76e4550 / CANN 9.0.0 / Ascend910B3 / 2026-10-30 | `pattern-library/constants.md`（新增 `CONST-launch-floor-aiv-910B3`） |
| **Ascend910B3 单 AIV core MTE2 常数**：流式 GM 读 68 GB/s（fp16→fp32 转换 62 GB/s，转换不额外收费）；**每次 DMA 指令固定开销 ≈0.16 µs**（512 KB 用 64/32/16 次搬运 = 13.97/9.38/6.65 µs 三点拟合）→ tile 小于 ~16 KB 时 DMA 时间由指令数而非带宽决定 | D | §2.2 `profiles/probe/{read,convert,conv_c*}` | `perf_opt/probe/h1_probe.py --mode convert --chunk-elem {4096,8192,16384}` | 同上 | `pattern-library/constants.md` + `elementwise.md`（copy-floor 曲线补 910B3 单核点） |
| **MTE2 与 VEC 已在重叠时，单独降低 MTE2 不降低 Task Duration**：b4 `mte2_time 55.2 + vec_time 56.4 = 111.6 ≫ Task 78.4` 表示重叠率 ~30%；slab 把 mte2 降到 31.9 却把 Task 推到 107.2，因为 UB 翻倍挤掉了 `--enable-auto-multi-buffer` 的双缓冲。判据：`mte2_time + vec_time > Task Duration` 时优先降低「更长的那条 pipe」或减小 UB，而非放大 tile | P | §2.5 R2-A/R2-G；`perf_opt/profiles/round2/r2a2_*`、`r2g_*` | `_engram_decode_kernel_r2a2_slab32.py` / `_r2g_slab32unroll.py`（慢）vs `_engram_decode_kernel.py`（快） | 同上 | `pattern-library/bottleneck` 反例 + `elementwise.md` |
| **`T.reduce_sum` 混合 dtype（fp16 src → fp32 dst）编译通过但静默算错**；`bf16` 源不支持（文档 §2.2.1）。半精度向量 GEMV 路线因此在本工具链不可用 | D | `/tmp/opencode/t_mixred.py`；`_engram_decode_kernel_r2e_h16only.py` 的 `memref.expand_shape` 编译错 | `perf_opt/probe/h1_probe.py` 同族探针 + 上述最小 16 行复现 | 同上 | `pattern-library/traps-runtime.md` |
| **能力缺口（compiler）**：Developer 模式缺「跨 AI Core 归约 / barrier」前端能力（`sync_block_set/wait` 仅同核 Cube↔Vector；workspace 仅 launcher 自动注入、无 `T.alloc_*` 绑定；`atomic_add` 无完成栅栏）→ `grid=batch` 的小 batch decode 类算子无法 split-N/split-K，Ratio 被核数比例钉死（本次 b1 上限 16%、b4 10%、b8 20%） | C | §2.3 的文档条款 + §4.2 三档上界 | 无（缺能力，无法构造 repro） | 同上 | `.agents/evolution/capability-gaps.md`（新增，`occurrences=1`；同族 decode 算子会重复命中） |

### 6.3 Transferable Lessons

- **先测「空 kernel 地板」再谈 Ratio 天花板**：`roofline_ideal / launch_floor` 是不依赖任何实现选择的硬上限，一步就能判定「目标是否可能」。
- **`mte2_time + vec_time > Task Duration` 是「已重叠」的信号**：此时降低单条 pipe 的搬运量不会降低总时间，改大 tile 反而会因 UB 压力丢失 multi-buffer（本轮实测回退 25–37%）。
- **小 DMA 指令的固定开销（0.16 µs/次）常大于其数据量成本**：合并小 copy（本轮 state-shift 逐行 → 整块）是低风险稳定收益（−5.9%~−8.8%）。
- **npuir 下「编译通过」不等于「语义正确」**：混合 dtype reduce、affine 索引广播两处都静默算错。任何候选必须用 `bench_driver.py --check` 先过单例 golden 再花 80 s 上 msprof。
- **micro-benchmark 探针仍是双刃剑**：本轮 `floor` 探针第一次写成 stride-32 gather store，得到 33 µs 的荒谬值（MTE3 1 GB/s）；改成连续 `T.copy` 后同为 1.89 µs。**探针必须先用「同形态连续搬运」校正，否则会得出反直觉结论**（延续第 1 轮「探针可靠性低于全 kernel」的经验）。

### 6.4 本轮未执行/未回写项

- **pattern-library / capability-gaps 未直接回写**：按本 Agent 只读约束（只写 `perf_opt/`），条目内容与 repro 位置已完整记录于 §6.1/§6.2，交任务终态 `tilelang-skill-evolver` 蒸馏合入。
- **跨核归约路线未实施**：§2.3 已给文档条款；按调度 prompt 要求只做可行性与量化收益估计，不改 op/wrapper 契约。
- **未产出 `perf_opt/perf_feedback.md`**：§4.4 两项触发条件不满足。

---

## 7. 工件清单

| 文件 | 说明 |
|---|---|
| `perf_opt/_engram_decode_kernel.py` | **最优版本**（drop-in，同工厂签名；`TUNED_BK = 16`，`TUNED_KS` 未引入——slab 未采纳），sha256 `3835dee0…a08f05` |
| `perf_opt/_engram_decode_kernel_r2baseline.py` | 本会话 baseline 留档 = 第 1 轮 winner（sha256 `d30e11a9…d7a234`，逐字节相同） |
| `perf_opt/_engram_decode_kernel_r2h_shift2d.py` | winner 候选（H1，性能采集所用文件，sha256 `45b43931…b9fffb`；最终文件仅改 docstring） |
| `perf_opt/_engram_decode_kernel_r2a_slab32.py`、`_r2a2_slab32.py`、`_r2g_slab32unroll.py`、`_r2h2_shift2d_cw.py` | 有效但未采纳的候选（slab 早期不可解析形态 / slab / slab 展开 / H2） |
| `perf_opt/_engram_decode_kernel_r2b_elocal.py`、`_r2c_bk32.py`、`_r2d_h16.py`、`_r2e_h16only.py` | 失败候选留档（结果错误 / UB overflow / 编译错误） |
| `perf_opt/opt_log.md` | 本文件 |
| `perf_opt/perf_records.jsonl` | 17 行（baseline 3 + candidate 11 + final 3），append-only |
| `perf_opt/workload_inventory.json` | 4 workload（1 smoke + 3 tune） |
| `perf_opt/history/opt_log_round1.md`、`history/perf_records_round1.jsonl` | 第 1 轮原始工件（原样移动，零改动） |
| `perf_opt/bench_driver.py` | 单 workload 启动器（`--module` 选文件，`--check` 单例 golden） |
| `perf_opt/run_prof.sh` + `perf_opt/parse_prof.py` | `msprof op` 采集 + OpBasicInfo/PipeUtilization/Memory/ArithmeticUtilization + `visualize_data.bin` roofline 解析（复刻 `_parse_roofline`） |
| `perf_opt/probe/h1_probe.py` | H1 探针（launch 地板 / 流式读 / 转换读 / 分块 copy） |
| `perf_opt/probe/ablate_gemvonly.py`、`ablate_nogemm.py` | GEMV / 尾部 ablation（仅测量用，结果不正确） |
| `perf_opt/profiles/{baseline,round2,final,probe}/` | raw `msprof op` 数据 |
| `perf_opt/logs/final/L_all.log` | 最终 `--level all` 回归日志 |
