# `_make_leaky_relu_kernel` — Stage 4 性能调优日志

- 算子目录: `examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu_kernel/`
- 基准 (只读, Stage 3): `_make_leaky_relu_kernel.py`（工厂 `_make_leaky_relu_kernel(N, dtype, negative_slope, output_dtype=None, block_size=2048)`；默认路径 `max(x, s·x)` 2-op 链 + 动态尾块标量循环；wrapper 实际传入 `block_size = _DEFAULT_BLOCK_SIZE[dtype]` = fp32 1024 / fp16·bf16 2048）
- 调优产物 (final): `perf_opt/_make_leaky_relu_kernel.py`（`candidate_id = final_merged`，sha256 `1475bf2d7a227e5d06ff40a0488729b99783675ce610ef4cb2e6a274da7b37e7`）
- 性能目标: **best_effort**（无逐 workload 数值硬门槛）
- 时延口径: **`msprof op` Task Duration(us)**（kernel-only，唯一口径；median of 20 launches，warm-up 5）
- 噪声阈值: 3%；<5% 差异须走 `ab_test.py` 交错多 run 协议
- bench 入口: `examples/TileOPs/benchmarks/ops/bench_elementwise_manifest.py::test_leaky_relu_manifest_bench`（`workloads_to_params("LeakyReluFwdOp", include_extra=True)`）
- 回归入口: `perf_opt/_make_leaky_relu_kernel.py --level all`（L0/L1 阻塞，L2/Boundary 告警不阻塞）
- 硬件/工具链: Ascend910B3（实查 20 AI core × 2 = **40 物理 Vector 核**），CANN 9.0.0，Developer 模式，tilelang 0.1.2+16d599a32c5e494412aafb238ca4a3f954fbb2d2
- `mode=full`；`budget.max_stage4_experiments` 未设置（默认 `max_rounds=10` / `max_experiments=30`）
- 设备 UB 上限实测（BiShengIR 报错原文）：`ub overflow, requires ... while 1572864 bits available` → **196608 B = 192 KB**

## Workload Inventory 与 Baseline (Performance Test Data)

`workload_inventory.json` 从 bench 入口 `test_leaky_relu_manifest_bench` 展开（manifest `elementwise.yaml::LeakyReluFwdOp` 的 4 组 workload × dtype），逐案例分类：

| kind | workload_id | shape | N | dtype | 处理 |
|---|---|---|---|---|---|
| smoke | smoke-1m-float32 | [1048576] | 1,048,576 | float32 | 仅精度回归（不调优） |
| smoke | smoke-1m-float16 | [1048576] | 1,048,576 | float16 | 仅精度回归（不调优） |
| smoke | smoke-1m-bfloat16 | [1048576] | 1,048,576 | bfloat16 | 仅精度回归（不调优） |
| tune | gan-feat-float16 | [16,256,64,64] | 16,777,216 | float16 | 调优 |
| tune | gan-feat-bfloat16 | [16,256,64,64] | 16,777,216 | bfloat16 | 调优 |
| tune | gan-feat-deep-float16 | [16,512,32,32] | 8,388,608 | float16 | 调优 |
| tune | gan-feat-deep-bfloat16 | [16,512,32,32] | 8,388,608 | bfloat16 | 调优 |
| tune | elementwise-16m-float16 | [4096,4096] | 16,777,216 | float16 | 调优 |
| tune | elementwise-16m-bfloat16 | [4096,4096] | 16,777,216 | bfloat16 | 调优 |
| tune | elementwise-16m-float32 | [4096,4096] | 16,777,216 | float32 | 调优 |

- skipped: 无。
- smoke 排除理由: pytest case id 含独立 `smoke` token（`smoke-1m-*`），按 skill Phase 1 排除出性能调优，仅作每次合并及最终精度回归。
- 全部 10 个案例实际调用同一个 Stage 3 kernel（`_make_leaky_relu_kernel`，prim_func `main`，`target_kernel_name="main"`），无多 kernel 分派；benchmark 的 `op_params` 为空 dict（manifest workload 未定义额外参数）→ 全部案例 `negative_slope = 0.01`（默认路径 `use_max=True`）。
- 排序理由：先大流量的 16m 档（gan-feat / elementwise-16m 的 f16·bf16，baseline 75 µs 量级），再 8.39M 的 gan-feat-deep 档（baseline 39 µs），最后 16m fp32（同 shape 但访存翻倍，baseline 217.7 µs，实测负担最重）。排序只影响先后，7 个 tune workload 全部完成。

Baseline（round 0，`phase=baseline`，kernel = 基准 `../_make_leaky_relu_kernel.py`，wrapper 同款 per-dtype block_size）：

| kernel_id | workload_id | Task Duration(us) | effective R+W (TB/s) | AIV vec | MTE2 | MTE3 | Scalar | L0 | raw profile |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| …leaky_relu_kernel/_make_leaky_relu_kernel.py::_make_leaky_relu_kernel::main | gan-feat-float16 | 75.62 | 0.89 | 0.354 | 0.875 | 0.411 | 0.103 | pass | `profiles/baseline/round0/baseline_8ba8496a/gan-feat-float16_ae91d9f4` |
| …::main | gan-feat-bfloat16 | 75.69 | 0.89 | 0.393 | 0.876 | 0.411 | 0.102 | pass | 同目录 |
| …::main | gan-feat-deep-float16 | 38.75 | 0.87 | 0.352 | 0.865 | 0.408 | 0.109 | pass | 同目录 |
| …::main | gan-feat-deep-bfloat16 | 39.23 | 0.86 | 0.386 | 0.862 | 0.404 | 0.112 | pass | 同目录 |
| …::main | elementwise-16m-float16 | 75.72 | 0.89 | 0.354 | 0.875 | 0.410 | 0.103 | pass | 同目录 |
| …::main | elementwise-16m-bfloat16 | 75.33 | 0.89 | 0.392 | 0.876 | 0.411 | 0.102 | pass | 同目录 |
| …::main | elementwise-16m-float32 | 217.72 | 0.62 | 0.090 | 0.601 | 0.281 | 0.052 | pass | 同目录 |

### Baseline 瓶颈判读（来自 raw `PipeUtilization`）

**GM 搬运受限，且有效带宽极低**：16m fp16 的 AIV 侧 `mte2_time = 63.32 µs` 搬运 33.55 MB 读 → **12.34 GB/s/core（40 核聚合 ≈ 0.49 TB/s）**，`mte3_time = 29.68 µs` 搬 33.55 MB 写 → 26.3 GB/s/core（聚合 ≈ 1.05 TB/s）。`vec_ratio` 仅 0.354、`scalar` 0.10。管道忙时之和 63.3+29.7+25.6+7.8 ≈ 126 µs > wall 75.7 µs → DMA 与 VEC 部分重叠，但 DMA 整体占主导。

**根因 = 单 tile 仅 2-4 KB**：fp16 `block_size=2048` → 每 tile 4 KB；40 核每核跑 205 个 tile，小突发 DMA 使读带宽退化到峰值的 ~1/4。对照同机 `CONST-elu-910B3-bw-floor`（16m fp16 bs=2048 → mte2 12.35 GB/s/core、有效 0.64 TB/s），本算子 baseline 与 elu 完全同档，**确认瓶颈为搬运而非向量链**。

**fp32 更极端**：`vec_ratio` 仅 0.090，`mte2×mte3` 占比 0.88，是纯 DMA 墙。

结论：调优主轴 = **搬运效率（block_size / tile 结构 / DMA 流水）**，而非逐 op 消融。leaky_relu 默认路径只有 2 op（`vmul`+`vmax`），向量相在整条链上占比极低，**没有超越函数可消融**（DESIGN §1.6.0 候选 #6 明确不适用），与同族 elu（4–6 op）相比更纯粹是 DMA 受限。

### 设计估算 vs 实测偏差（D-2 回填，DESIGN §1.6.0 roofline）

| workload | 设计估算下界(us) | baseline 实测(us) | 偏差 | 失准项 |
|---|---:|---:|---|---|
| gan-feat fp16 (16m) | ≈ 410 | 75.62 | 实测快 **5.4x** | 发射项：`CONST-vector-launch-overhead`（0.5µs/op）绑 910B2C 且 stale，本机 2-op 短链的发射被 tile 内重叠完全掩盖，估算严重上偏 |
| elementwise-16m fp16 | ≈ 410 | 75.72 | 实测快 **5.4x** | 同上 |
| elementwise-16m fp32 | ≈ 410 | 217.72 | 实测快 **1.9x** | 同上（fp32 流量项 107 µs 已占实测下界一半） |

偏差方向一致且 ≥1.9x → 设计把「发射项绝对主导」的模型在本设备上明显上偏；但**算法层结论（无更优算法替代、`max` 形式 2-op 最短、`block_size` 是第一杠杆、fp32 中转不可省）被本轮实测完全验证**，且 `block_size` 摊薄发射/提高 DMA 效率的判据被实测放大。**不构成设计层天花板**（无 `[DESIGN_LIMIT]`，见文末）。

## 诊断探针（诊断性，**不计入 gate、不写 perf_records**）

探针工具：`perf_opt/probe_sweep.py`（任意 impl × workload × block_size 的 `msprof op` 采集 + PipeUtilization 解析）、`perf_opt/_bench_kernel.py`（按 wrapper 语义解析 impl 的 `TUNED_BLOCK_SIZE` / `_DEFAULT_BLOCK_SIZE` 表传 `block_size`）。

### P1 — baseline block_size 扫描（16m，结构不变）

| dtype / path | bs=1024 | 2048 | 4096 | 6144 | 8192 | 10240 | 12288 | 13312 | 14336 | 16384 | 20480 | 24576 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| float16 (max 路径) | — | 75.62* | 45.26 | — | 31.73 | 29.14 | **27.96** | 28.95 | UB overflow | — | — | — |
| float32 (max 路径) | 217.72* | — | 78.44 | 63.55 | 56.86 | 56.52 | 54.90 | 54.59 | 54.29 | **52.45** | 52.07 | 51.09 |

（* 为 wrapper 默认 bs 的 baseline 实测值。）伴随指标（16m fp16）：`mte2_time` 63.3→17.4 µs、`mte2_active_bw` 12.34→45.1 GB/s/core、`vec_ratio` 0.354→0.826、有效带宽 0.89→2.40 TB/s。**结论：`block_size` 是数量级级的第一杠杆，bs=12288（fp16）即把 DMA 推至同机搬运地板（`CONST-elu-910B3-bw-floor` 记 copy floor ≈ 2.49 TB/s，本算子达 2.40 TB/s）；fp32 在 bs≥16384 后进入 <3% 的平区（2.56–2.63 TB/s）。**

### P2 — fallback（non-default slope）路径 UB 天花板

默认 `max` 路径会 DCE 掉 `z`/`cond`（`vmul`/`vmax` 不用它们），fallback 的 `vcmp`+`vselect` 则让二者 live，UB 预算更紧。实测编译天花板（`ub overflow` 判据）：

| 路径 | dtype | ok / overflow 分界 |
|---|---|---|
| max（默认，`0<s≤1`） | fp16 | 13312 ok / 14336 overflow |
| max（默认） | fp32 | 24576 ok / 28672 overflow |
| fallback（`vcmp/vselect`，动态尾块） | fp16 | 10240 ok / 11264 overflow |
| fallback（动态尾块） | fp32 | 15360 ok / 16384 overflow |
| fallback（静态整块，Round 2 结构） | fp32 | 6144 起采用保守值 |

→ 这解释了 `r1_bs` 首次以 fp32=16384 跑 `--level L0` 时 `L0-slope2.0` 触发的 `ub overflow, requires 1589504 bits while 1572864 available`：**cap 来自 fallback 而非默认路径**。对策 = 工厂内仅对 fallback 编译期收窄 tile（`_WHERE_BLOCK_SIZE`，Python 常量，无运行时分支、无新分派面）。

### P3 — 静态整块 / 动态尾块拆分（`PL-1.34`）

`block_size` 为编译期常量，`N // bs` 个整 tile 可用静态 extent 的 `T.copy`，仅余数 tile 保留动态 `T.min`。

| dtype | 结构 | bs | TaskDur(us) | 有效 R+W (TB/s) |
|---|---|---:|---:|---:|
| fp32 | 动态尾块（r1_bs） | 16384 | 51.18 | 2.62 |
| fp32 | **静态整块（r2_static）** | 8192 | 48.69 | 2.76 |
| fp32 | **静态整块** | 12288 | **48.39** | 2.77 |
| fp32 | 静态整块 | 14336 | UB overflow | — |
| fp16 | 动态尾块（r1_bs） | 12288 | 27.96 | 2.40 |
| fp16 | 静态整块 | 8192 | 27.06 | 2.48 |
| fp16 | 静态整块 | 10240 | 47.50 | 1.41 |
| fp16 | 静态整块 | 12288 | 39.49 | 1.70 |

**结论**：静态整块对 **fp32** 有净益（同 bs 下墙时更低），但把 fp32 的 UB 天花板压到 12288；对 **fp16** 在 bs=8192 仅有噪声级收益（27.06 vs 27.96，且同代码重复测量 f16 存在 ~11% 跨 run 双态），bs≥10240 明显退化——与 `PL-1.34-staticbulk-unlocks-overlap` 的「仅 DMA 受限时有净益、f16 大 bs 反例」一致。故 **fp32 走静态整块、fp16/bf16 保留动态尾块**。

## Iteration Log

每轮固定 base = 当前已合并 current best；每个实验分支只改一个主要优化点；分支先跑 L0 再 `msprof op`。**实验批处理 runner**：`perf_opt/run_experiments.py`（自 skill 模板适配，`KERNEL_ID = …/leaky_relu_kernel/_make_leaky_relu_kernel.py::_make_leaky_relu_kernel::main`；`--workload/--all`；逐测量追加 `perf_records.jsonl` 并打印「候选 vs current best」表）。

### Round 1 — per-dtype block_size（base = Stage 3 baseline）

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| r1_bs | 单 tile 2-4 KB，MTE2 仅 12.34 GB/s/core | per-dtype `block_size` | P1 扫描 + `CONST-elu-910B3-bw-floor` | 新增 `TUNED_BLOCK_SIZE`{f32 16384, f16 12288, bf16 12288}；工厂 `block_size=None` 时取该表；对 fallback 路径按 `_WHERE_BLOCK_SIZE` 收窄（P2） | TaskDur / mte2_bw / L0 | **improved（主杠杆）** |

**候选 vs current best（current best = baseline）**

| branch | workload_id | L0 | TaskDur(us) | best(us) | Δ | vec | mte2 | mte3 | mte2_time(us) | mte2_bw(GB/s/core) | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| r1_bs | gan-feat-float16 | pass | 28.02 | 75.62 | **+62.9%** | 0.828 | 0.729 | 0.368 | 17.5 | 44.7 | improved |
| r1_bs | gan-feat-bfloat16 | pass | 28.85 | 75.69 | **+61.9%** | 0.886 | 0.694 | 0.340 | — | — | improved |
| r1_bs | gan-feat-deep-float16 | pass | 15.29 | 38.75 | **+60.5%** | 0.772 | 0.685 | 0.338 | — | — | improved |
| r1_bs | gan-feat-deep-bfloat16 | pass | 15.33 | 39.23 | **+60.9%** | 0.828 | 0.661 | 0.318 | — | — | improved |
| r1_bs | elementwise-16m-float16 | pass | 28.43 | 75.72 | **+62.5%** | 0.821 | 0.734 | 0.364 | 17.5 | 44.7 | improved |
| r1_bs | elementwise-16m-bfloat16 | pass | 27.77 | 75.33 | **+63.1%** | 0.891 | 0.693 | 0.343 | — | — | improved |
| r1_bs | elementwise-16m-float32 | pass | 51.18 | 217.72 | **+76.5%** | 0.238 | 0.469 | 0.270 | 22.5 | 69.6 | improved |

- **机制**：`block_size` 2048→12288 把 GM 读突发从 4 KB 提到 24 KB，`mte2_active_bw` 12.34→44.7 GB/s/core（3.6x），DMA 相从 63.3 → 17.5 µs；整条链转为 DMA 与 VEC 充分重叠。
- **本轮 winner**：`r1_bs`（per-dtype block_size）。current best 更新为 r1_bs。全部 7 个 tune workload 均通过（本轮即完成该 kernel 非回退检查的基）。

### Round 2 — 静态整块 / 动态尾块拆分（base = r1_bs）

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| r2_static | fp32 三管近零重叠（mte2_time 22.5 + mte3_time 12.9 ≈ wall 51） | fp32 路径静态整块 + 单动态尾块 | `PL-1.34-staticbulk-unlocks-overlap` + P3 探针 | fp32 分支拆成 `task<n_full`（静态 extent `T.copy(x[off:off+bs], x_ub)`）与 `task==n_full`（动态尾块）；fp16/bf16 分支保持动态尾块 | TaskDur / ab_test / L0 | **improved（fp32）** |
| r2b_static16 | 同上的 fp16 变体 | fp16 路径静态整块 | `PL-1.34` 反例核对 | fp16 分支同样拆分 | TaskDur | config_no_gain（探针即否决，未升为正式分支） |

**候选 vs current best（current best = r1_bs）**

| branch | workload_id | L0 | TaskDur(us) | best(us) | Δ | vec | mte2 | mte3 | mte2_time(us) | mte3_time(us) | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| r2_static | elementwise-16m-float32 | pass | 47.77 | 51.18 | **+6.7%** | 0.266 | 0.905 | 0.550 | 39.0 | 23.7 | improved |
| r2_static | gan-feat-float16 | pass | 28.49 | 28.02 | −1.7% | 0.814 | 0.731 | 0.362 | — | — | tie（同代码路径，噪声） |
| r2_static | gan-feat-bfloat16 | pass | 28.30 | 28.85 | +1.9% | 0.880 | 0.697 | 0.339 | — | — | tie（同代码路径） |
| r2_static | gan-feat-deep-float16 | pass | 15.43 | 15.29 | −0.9% | 0.777 | 0.686 | 0.342 | — | — | tie |
| r2_static | gan-feat-deep-bfloat16 | pass | 15.77 | 15.33 | −2.9% | 0.832 | 0.649 | 0.318 | — | — | tie |
| r2_static | elementwise-16m-float16 | pass | 25.50 | 28.43 | +10.3% | 0.841 | 0.729 | 0.379 | — | — | tie（同代码路径；f16 跨 run 双态） |
| r2_static | elementwise-16m-bfloat16 | pass | 28.21 | 27.77 | −1.6% | 0.890 | 0.691 | 0.342 | — | — | tie（同代码路径） |

- **r2_static 相对 r1_bs 只改了 fp32 分支的循环结构**；上表 f16/bf16 行的散差（−2.9% ~ +10.3%）全部来自同代码路径的跨 run 双态（对照：gan-feat-f16 与 elementwise-16m-f16 是同一 N/同一代码，实测 28.49 vs 25.50），确认落在噪声区、**不构成回退**。
- **fp32 差异 >5% 需协议裁决** → `ab_test.py` 交错 5 对（A=r1_bs 动态 bs16384，B=r2_static 静态 bs12288）：

```json
{"a_merged_median_us": 52.68, "b_merged_median_us": 47.75, "b_rel_change": -0.0936,
 "pair_diffs_us": [-5.02, -3.64, -3.73, -6.07, -3.74], "sign_test_p": 0.0625,
 "direction_consistent": true, "noise_threshold": 0.03, "verdict": "B_better"}
```

- **r2b（fp16 静态整块）探针即否决**：bs=8192 27.06（vs 动态 27.96，噪声内）、bs=10240 47.50、bs=12288 39.49（显著退化）→ 未升为正式分支，记 `config_no_gain`（**不否定静态整块整类**，仅限 fp16 大 bs 语境，与 `PL-1.34` 反例一致）。
- **本轮 winner**：`r2_static`（fp32 静态整块）。合并版 = fp32 静态整块 + fp16/bf16 动态尾块 + per-dtype block_size，作为 current best；`perf_opt/_make_leaky_relu_kernel.py` 即该合并版。

### 收敛判定

- Round 1 为主杠杆（+60~77%），Round 2 为 fp32 结构精修（+6.7% 协议确认）。
- 收敛依据：① 有效 R+W 已达 **2.7 TB/s（fp32）/ 2.4 TB/s（fp16）**，贴/超同机 `CONST-elu-910B3-bw-floor` 的搬运地板（2.49–2.67 TB/s）；② 默认路径已是**最短闭式 2-op**（`vmul`+`vmax`），DESIGN §1.6.0 已证无更短等价式，且无超越函数可消融；③ 参照结构 diff（morph ladder 轻量版）：检索同族 `elu`/`bitwise_not` 已优化版 + `PL-1.22/1.23/1.28/1.33/1.34` —— 唯一未采纳的结构差异（fp16 静态整块、contiguous 分核映射）均与本算子实测相符地无益（P3 / `PL-1.6` grid-stride 结论）；④ 剩余候选全为 blocked/no_gain。
- 中止条件：**success**（best_effort，搬运地板达成；所有 tune workload 有结论且全量复测完成，smoke 精度通过）。

### Autotune Log

未使用 autotune。本算子为 DMA 受限，瓶颈为**连续参数**（block_size 的带宽曲线）而非离散参数搜索空间；P1 的显式一维扫描 + 编译天花板二分已给出完整曲线，autotune 无额外搜索维度可加。

## Final Summary

- 最终候选: `perf_opt/_make_leaky_relu_kernel.py`（`candidate_id = final_merged`，即 r2_static 合并版；工厂签名与基准一致，drop-in 替换）。
- 采纳优化点（2 项）：
  1. **per-dtype `block_size`**：fp32 12288 / fp16 12288 / bf16 12288（模块常量 `TUNED_BLOCK_SIZE`，供 wrapper 切换块引用），fallback 路径按 `_WHERE_BLOCK_SIZE`{f32 6144, f16 10240, bf16 10240} 编译期收窄。
  2. **fp32 路径静态整块 + 单动态尾块**（`PL-1.34`）：解锁 MTE2/MTE3/VME 重叠；fp16/bf16 保留动态尾块。

逐 `kernel_id + workload_id` 对账（**baseline 与 final 均来自 `perf_records.jsonl`，final 全部来自同一最终候选 `final_merged`**）：

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu_kernel/_make_leaky_relu_kernel.py::_make_leaky_relu_kernel::main | gan-feat-float16 | 75.62 | 27.82 | final_merged |
| examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu_kernel/_make_leaky_relu_kernel.py::_make_leaky_relu_kernel::main | gan-feat-bfloat16 | 75.69 | 28.14 | final_merged |
| examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu_kernel/_make_leaky_relu_kernel.py::_make_leaky_relu_kernel::main | gan-feat-deep-float16 | 38.75 | 15.41 | final_merged |
| examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu_kernel/_make_leaky_relu_kernel.py::_make_leaky_relu_kernel::main | gan-feat-deep-bfloat16 | 39.23 | 15.04 | final_merged |
| examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu_kernel/_make_leaky_relu_kernel.py::_make_leaky_relu_kernel::main | elementwise-16m-float16 | 75.72 | 28.57 | final_merged |
| examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu_kernel/_make_leaky_relu_kernel.py::_make_leaky_relu_kernel::main | elementwise-16m-bfloat16 | 75.33 | 28.00 | final_merged |
| examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu_kernel/_make_leaky_relu_kernel.py::_make_leaky_relu_kernel::main | elementwise-16m-float32 | 217.72 | 49.96 | final_merged |

逐 workload 提升与有效带宽（R+W 聚合 = `2·N·elem_bytes / TaskDur`）：gan-feat-f16 **+63.2%**（0.89→2.41 TB/s）、gan-feat-bf16 **+62.8%**（0.89→2.38）、gan-feat-deep-f16 **+60.2%**（0.87→2.18）、gan-feat-deep-bf16 **+61.7%**（0.86→2.23）、elementwise-16m-f16 **+62.3%**（0.89→2.35）、elementwise-16m-bf16 **+62.8%**（0.89→2.40）、elementwise-16m-f32 **+77.1%**（0.62→2.69）。全部 L0 pass；final raw profile 位于 `profiles/final/round3/final_merged_*/<workload_id>_*`。

- smoke 排除（仅精度，已跑最终回归）：smoke-1m-{float32,float16,bfloat16}。
- 最终精度回归（`perf_opt/_make_leaky_relu_kernel.py --level all`）：**L0 / L1 / L2 / Boundary 全部 PASS，`max_diff = 0.000000e+00`**；覆盖 dtype × shape × 尾块 × 非默认 slope（0.0 / 0.5 / 2.0 / 100.0）× inf/NaN/次正规 的完整组合。
- 总体结论: 默认路径（`negative_slope=0.01`）全部 tune workload 提升 **60.2%–77.1%**，达到同机逐元素搬运地板上限；精度零误差。
- 中止原因: **success**（best_effort）。

## 设计层天花板判定（`[DESIGN_LIMIT]`）

对照 `_shared/standards/perf-feedback.md` §1 两项触发条件：

1. **是否由算法/设计层决定（非 tiling/参数可解）？** —— **否**。本轮全部收益来自 `block_size`（tiling 参数）与循环结构（静态/动态尾块），二者均是参数/tiling 层杠杆，非算法/设计层；DESIGN §1.6.0 的算法结论（无更优等价式、`max` 2-op 最短、fp32 中转必需）被实测**正面验证**。
2. **结构性加速估计 > 2x 或实测与设计假设矛盾？** —— 实测与「发射项主导」的**估算数字**相矛盾（上偏 1.9–5.4x），但这属于 D-2 roofline 常数（`CONST-vector-launch-overhead` 绑 910B2C 且 stale）失准，**不构成设计假设被推翻**：设计的算法选择与瓶颈方向（访存/发射受限、`block_size` 第一杠杆）均成立，且已回填为偏差行。

→ **两项未同时满足 ⇒ 不产出 `perf_feedback.md`，无 `[DESIGN_LIMIT]` 信号**（参数级结论留在迭代内；D-2 偏差已按 `hardware-cost-model.md` §3 回填本日志）。

## Skill Retrospective

- **流程有效**：`perf-feedback` 触发门槛（两项同时满足）成功避免了把「参数/tiling 可解 + roofline 常数 stale 导致的估算偏差」误报为设计层天花板。
- **可复用的顺序**：同族 `CONST-elu-910B3-bw-floor`（block_size↔带宽曲线）直接给出第一杠杆的量级，把探针预算从「重新标定地板」压缩到「确认 + 二分编译天花板」，本轮探针数（~13 次 msprof）远低于计算受限算子。
- **新发现（本任务）**：`max`/`min` 形式编译期分工的**边界**——`max(x,s·x)` 对 `s≤1` 精确、`min(x,s·x)` 对 `s>1` 精确（本轮数值验证），但二者都**不能**覆盖 `s==0` 的 `x=-inf` 角点（golden = NaN）；且 default-path 的 DCE 会让 `z`/`cond` 消失、fallback 路径才受 UB 天花板约束——**调参时必须按「最坏编译分支」定 cap**，否则默认路径的 bs 会被 fallback 暗中卡住（本轮 `r1_bs` fp32=16384 首次 L0 失败即此）。
- **BP proposal / value point（供终态 evolver 蒸馏）**：

  | id | vp_type | 摘要 | evidence | repro | toolchain_stamp |
  |---|---|---|---|---|---|
  | VP-LR-1 | D | `max`/`min` 编译期分工的精确域：`max(x,s·x)` ≡ golden 之 `s≤1`（含 `s<0`；`s=0` 仅 `x=-inf` 1 例不符），`min(x,s·x)` ≡ golden 之 `s>1` | `_make_leaky_relu_kernel.py` L0/L1 全 PASS（slope 0.0/0.5/2.0/100.0，max_diff=0）+ 离线 20 项数值表 | repro-missing（待补：最小 CPU 数值对照脚本） | tilelang 0.1.2+16d599a / CANN 9.0.0 / 910B3 |
  | VP-LR-2 | D | 910B3 上 leaky_relu（2-op DMA 受限）的 `block_size` 带宽曲线与地板：fp16 bs12288 → 27.96 µs（2.40 TB/s）、fp32 bs16384 → 52.45 µs（2.56 TB/s）；fallback（vcmp/vselect）UB 天花板 fp16 10240 / fp32 15360 | `perf_opt/probe_sweep.py` + 本日志 P1/P2 表 | repro-missing（待补：probe_sweep 最小化） | 同上 |
  | VP-LR-3 | P | 静态整块/动态尾块拆分在 **纯 DMA 受限** 逐元素算子上对 **fp32** 有净益（+9.4% 协议确认），对 **fp16 大 bs** 退化（`PL-1.34` 反例再证）；且 fallback 编译分支需独立 UB cap，否则默认路径 bs 被 cap | ab_test JSON（52.68→47.75，5/5 一致）；P3 表 | 分支文件 `_make_leaky_relu_kernel_round2_static.py` / `round2b_static16.py`（完整 before/after） | 同上 |

- **skill 流程问题**：无阻塞性缺陷。建议（proposal，不自动改）：`run_experiments.py` 模板的 L0 前置在「候选只改了某 dtype 路径」时仍会跑全 dtype 的 `run_L0`，容易暴露非目标分支的编译 cap（本轮 fp32 bs 被 fp32-fallback 卡住即由此发现）——这是**特性而非缺陷**，建议在模板注释中显式说明「L0 会覆盖非目标 dtype/slope 的编译分支，候选默认值必须对全部分支可编译」。
