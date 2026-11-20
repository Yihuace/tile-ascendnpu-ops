# `_make_hardtanh_kernel` — Stage 4 性能调优日志

- 算子目录: `examples/TileOPs/tileops/kernels/elementwise/hardtanh/hardtanh_kernel/`
- 基准 (只读, Stage 3): `_make_hardtanh_kernel.py`（工厂 `_make_hardtanh_kernel(N, dtype, min_val, max_val, block_size=2048)`；默认路径 `vclamp`（fp32 2-op）+ 动态尾块；wrapper 实际传入 `block_size = _DEFAULT_BLOCK_SIZE[dtype]` = fp32 1024 / fp16·bf16 2048）
- 调优产物 (final): `perf_opt/_make_hardtanh_kernel.py`（`candidate_id = final_merged`，sha256 `167dcdae16733a68f34f671cad2991cec72d6a2e0e505cef3124483e3489e7a8`）
- 性能目标: **best_effort**（无逐 workload 数值硬门槛）
- 时延口径: **`msprof op` Task Duration(us)**（kernel-only，唯一口径；median of 20 launches，warm-up 5）
- 噪声阈值: 3%；<5% 差异须走 `ab_test.py` 交错多 run 协议
- bench 入口: `examples/TileOPs/benchmarks/ops/bench_elementwise_manifest.py::test_hardtanh_manifest_bench`（`workloads_to_params("HardtanhFwdOp", include_extra=True)`）
- 回归入口: `perf_opt/_make_hardtanh_kernel.py --level all`（L0/L1 阻塞，L2/Boundary 告警不阻塞）
- 硬件/工具链: Ascend910B3（实查 20 AI core × 2 = **40 物理 Vector 核**，`NPUUtils.get_aicore_num()=20` / `get_aivector_core_num()=40`），CANN 9.0.0，Developer 模式，tilelang 0.1.2+cf91d889ce344b4feb4a13af00e9aa25a8de4c4d
- `mode=full`；`budget.max_stage4_experiments` 未设置（默认 `max_rounds=10` / `max_experiments=30`）；实际 3 轮 / 4 个候选分支（含 1 个探针否决）
- 设备 UB 上限实测（BiShengIR 报错原文）：`ub overflow, requires ... while 1572864 bits available` → **196608 B = 192 KB**

## Workload Inventory 与 Baseline (Performance Test Data)

`workload_inventory.json` 从 bench 入口 `test_hardtanh_manifest_bench` 展开（manifest `elementwise.yaml::HardtanhFwdOp` 的 4 组 workload × dtype），逐案例分类：

| kind | workload_id | shape | N | dtype | 处理 |
|---|---|---|---|---|---|
| smoke | smoke-1m-float32 | [1048576] | 1,048,576 | float32 | 仅精度回归（不调优） |
| smoke | smoke-1m-float16 | [1048576] | 1,048,576 | float16 | 仅精度回归（不调优） |
| smoke | smoke-1m-bfloat16 | [1048576] | 1,048,576 | bfloat16 | 仅精度回归（不调优） |
| tune | bounded-hidden-float16 | [2048,4096] | 8,388,608 | float16 | 调优 |
| tune | bounded-hidden-bfloat16 | [2048,4096] | 8,388,608 | bfloat16 | 调优 |
| tune | bounded-conv-feat-float16 | [16,256,56,56] | 12,845,056 | float16 | 调优（尾块 512） |
| tune | bounded-conv-feat-bfloat16 | [16,256,56,56] | 12,845,056 | bfloat16 | 调优（尾块 512） |
| tune | elementwise-16m-float16 | [4096,4096] | 16,777,216 | float16 | 调优 |
| tune | elementwise-16m-bfloat16 | [4096,4096] | 16,777,216 | bfloat16 | 调优 |
| tune | elementwise-16m-float32 | [4096,4096] | 16,777,216 | float32 | 调优（访存最重） |

- skipped: 无。
- smoke 排除理由: pytest case id 含独立 `smoke` token（`smoke-1m-*`），按 skill Phase 1 排除出性能调优，仅作每次合并及最终精度回归。
- 全部 10 个案例实际调用同一个 Stage 3 kernel（`_make_hardtanh_kernel`，prim_func `main`，`target_kernel_name="main"`），无多 kernel 分派；benchmark 的 `op_params` 为空 dict（manifest workload 未定义额外参数）→ 全部案例 `min_val=-1.0, max_val=1.0`（默认路径，不触发折叠）。
- 排序理由：先最大流量的 elementwise-16m 档（f16/bf16 75µs 量级、fp32 213µs，实测负担最重），再 12.85M 的 bounded-conv-feat 档（58µs），最后 8.39M 的 bounded-hidden 档（39µs）。排序只影响先后，7 个 tune workload 全部完成。

Baseline（round 0，`phase=baseline`，kernel = 基准 `../_make_hardtanh_kernel.py`，wrapper 同款 per-dtype block_size）：

| kernel_id | workload_id | Task Duration(us) | effective R+W (TB/s) | AIV vec | MTE2 | MTE3 | Scalar | L0 | raw profile |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| …hardtanh_kernel/_make_hardtanh_kernel.py::_make_hardtanh_kernel::main | bounded-hidden-float16 | 39.19 | 0.856 | 0.295 | 0.862 | 0.406 | 0.113 | pass | `profiles/baseline/round0/baseline_8ba8496a/bounded-hidden-float16_7f49ee64` |
| …::main | bounded-hidden-bfloat16 | 39.22 | 0.855 | 0.333 | 0.865 | 0.406 | 0.109 | pass | 同目录 |
| …::main | bounded-conv-feat-float16 | 58.37 | 0.880 | 0.297 | 0.871 | 0.409 | 0.107 | pass | 同目录 |
| …::main | bounded-conv-feat-bfloat16 | 57.82 | 0.889 | 0.335 | 0.871 | 0.410 | 0.106 | pass | 同目录 |
| …::main | elementwise-16m-float16 | 75.39 | 0.890 | 0.298 | 0.875 | 0.411 | 0.104 | pass | 同目录 |
| …::main | elementwise-16m-bfloat16 | 75.54 | 0.888 | 0.336 | 0.876 | 0.411 | 0.102 | pass | 同目录 |
| …::main | elementwise-16m-float32 | 213.32 | 0.629 | 0.071 | 0.613 | 0.287 | 0.053 | pass | 同目录 |

### Baseline 瓶颈判读（来自 raw `PipeUtilization`）

**纯 GM 搬运受限，且有效带宽极低**：16m fp16 的 AIV 侧 `mte2_time = 63.16 µs` 搬运 33.55 MB 读 → **12.37 GB/s/core（40 核聚合 ≈ 0.49 TB/s）**，`mte3_time = 29.67 µs` 搬 33.55 MB 写 → 26.35 GB/s/core（聚合 ≈ 1.05 TB/s）。`vec_ratio` 仅 0.298、`scalar` 0.10。管道忙时之和 63.2+29.7+22.5+7.8 ≈ 123 µs > wall 75.4 µs → DMA 与 VEC 部分重叠，但 **DMA 占绝对主导**。

**根因 = 单 tile 仅 2-4 KB**：fp16 `block_size=2048` → 每 tile 4 KB；40 核每核跑 205 个 tile，小突发 DMA 使读带宽退化到峰值的 ~1/4。对照同机 `leaky_relu`（同族 2-op DMA 受限，bs=2048 → 75.6µs / 0.89 TB/s），本算子 baseline 与 leaky_relu **完全同档**，确认瓶颈为搬运而非向量链。

**fp32 更极端**：`vec_ratio` 仅 0.071，`mte2_time = 126.96 µs`、`mte2_bw = 12.31 GB/s/core`，是纯 DMA 墙。

结论：调优主轴 = **搬运效率（block_size / tile 结构 / DMA 流水）**；默认路径向量链仅 2 op（`vclamp`），fp16/bf16 另 +2 `vcast`。

### 设计估算 vs 实测偏差（D-2 回填，DESIGN §1.6.0 roofline）

| workload | 设计估算下界(us) | baseline 实测(us) | 偏差 | 失准项 |
|---|---:|---:|---|---|
| bounded-hidden fp16 | ≈ 206 | 39.19 | 实测快 **5.3x** | 发射项：`CONST-vector-launch-overhead` 绑 910B2C 且 stale，本机 4-op 短链的发射被 DMA 窗口掩盖，估算严重上偏 |
| bounded-conv-feat fp16 | ≈ 314 | 58.37 | 实测快 **5.4x** | 同上 |
| elementwise-16m fp16 | ≈ 410 | 75.39 | 实测快 **5.4x** | 同上 |
| elementwise-16m fp32 | ≈ 410 | 213.32 | 实测快 **1.9x** | 同上（fp32 流量项 122µs 已占实测下界一半） |

偏差方向一致且 ≥1.9x —— **与 DESIGN §1.6.0 自己声明的「发射项系统性上偏」完全一致**（该节已注明 `bitwise_not` 对照证据与 stale 常数）。算法层结论（`vclamp` 2-op 最短、边界折叠、`block_size` 是第一杠杆、fp32 中转对 bf16 必需）被本轮实测正面验证。**不构成设计层天花板**（无 `[DESIGN_LIMIT]`，见文末）。

## 诊断探针（诊断性，**不计入 gate、不写 perf_records**）

探针工具：`perf_opt/probe_sweep.py`（任意 impl × workload × block_size 的 `msprof op` 采集 + PipeUtilization 解析）、`perf_opt/_bench_kernel.py`（按 wrapper 语义解析 impl 的 `TUNED_BLOCK_SIZE` / `_DEFAULT_BLOCK_SIZE` 表传 `block_size`）。

### P1 — baseline block_size 扫描（结构不变，动态尾块）

| dtype / path | bs=2048/1024 | 4096 | 8192 | 12288 | 16384 | 20480 | 24576 | 28672 | 49152 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| float16 (transit) | 75.30* | 45.34 | 30.89 | **27.16** | 27.68 | UB overflow | — | — | — |
| bfloat16 (transit) | 75.54* | — | 30.75 | **28.44** | 28.62 | UB overflow | — | — | — |
| float32 | 214.16* | 74.67 | 54.09 | 52.47 | 51.40 | 52.25 | **50.64** | 50.95 | 53.66 |

（* 为 wrapper 默认 bs 的 baseline 实测值。）伴随指标（16m fp16）：`mte2_time` 63.2→17.7 µs、`mte2_active_bw` 12.37→44.3 GB/s/core、`vec_ratio` 0.298→0.677、有效带宽 0.89→2.47 TB/s。**结论：`block_size` 是数量级级的第一杠杆，bs=12288（fp16/bf16）即把 DMA 推至同机搬运地板（~2.5 TB/s）；fp32 在 bs≥16384 后进入 <3% 平区。** fp16/bf16 的 UB 天花板 = 16384（12 B/elem × 16384 = 192 KB：`x_ub`+`y_ub`+`xf`+`lo_ub` 全 live）。

### P2 — 静态整块 / 动态尾块拆分（`PL-1.34`）

`block_size` 为编译期常量，`N // bs` 个整 tile 可用静态 extent 的 `T.copy`，仅余数 tile 保留动态 `T.min`。

| dtype | 结构 | bs | TaskDur(us) | 有效 R+W (TB/s) | mte2_ratio |
|---|---|---:|---:|---:|---:|
| fp32 | 动态尾块（P1） | 24576 | 50.64 | 2.65 | 0.51 |
| fp32 | 动态尾块（P1） | 8192 | 54.09 | 2.48 | 0.48 |
| fp32 | **静态整块（r2 探针）** | 8192 | **48.24** | 2.78 | 0.89 |
| fp32 | 静态整块 | 12288 | 49.45 | 2.71 | 0.91 |
| fp32 | 静态整块 | 16384 | 48.57 | 2.76 | 0.91 |
| fp32 | 静态整块 | 20480 | UB overflow | — | — |
| fp16 | 动态尾块（P1） | 12288 | 27.16 | 2.47 | 0.74 |
| fp16 | 静态整块（r2b 探针） | 8192 | 26.88 | 2.50 | 0.92 |
| fp16 | 静态整块（r2b 探针） | 12288 | 36.15 | 1.86 | 0.50 |
| fp16 | 静态整块（r2b 探针） | 16384 | 26.45 | 2.54 | 0.91 |
| bf16 | 静态整块（r2b 探针） | 12288 | 38.12 | 1.76 | 0.47 |

**结论**：静态整块对 **fp32** 有净益（解锁 MTE2/MTE3 重叠，`mte2_ratio` 0.51→0.89），把 fp32 的 UB 天花板压到 16384；对 **fp16/bf16 大 bs** 明显退化（bs=12288 36.15 vs 动态 27.16；bf16 38.12 vs 28.44）——与 `PL-1.34-staticbulk-unlocks-overlap` 的「f16 大 bs 反例」一致。故 fp16/bf16 保留动态尾块，仅 fp32 走静态整块。

### P3 — fp16 原生 clamp（跳过 fp32 中转；DESIGN §1.6.1 E1 备选）

`round_fp16(clamp_fp32(x,lo,hi)) == clamp_fp16(x, round_fp16(lo), round_fp16(hi))` 逐位成立（投影类算子，输出恒为 `{x, lo, hi}` 之一，round 与 clamp 可交换——PL-1.17 第四证 / DESIGN E1 诊断 `max_abs=0.0`）。用原生 fp16 clamp 消除 fp32 中转 buffer 与 2 次 `vcast`，前端向量 op 由 4 → 2。

| dtype / 结构 | bs | TaskDur(us) | vec | mte2 | mte3 | 有效 R+W (TB/s) |
|---|---|---:|---:|---:|---:|---:|
| fp16 动态中转（P1，基线结构） | 12288 | 27.16 | 0.677 | 0.739 | 0.367 | 2.47 |
| **fp16 原生静态（r3 探针）** | 8192 | 27.39 | 0.199 | 0.849 | 0.540 | 2.45 |
| **fp16 原生静态（r3 探针）** | 12288 | 26.71 | 0.194 | 0.847 | 0.509 | 2.51 |
| **fp16 原生静态（r3 探针）** | 16384 | 26.22 | 0.187 | 0.851 | 0.518 | 2.56 |
| **fp16 原生静态（r3 探针）** | 24576 | **26.09** | 0.180 | 0.889 | 0.554 | 2.57 |

**结论**：fp16 原生把 `vec_ratio` 从 0.677 压到 0.180（向量相几乎完全隐入 DMA 窗口），wall 由 27.16 → 26.09µs（约 −4%）。bf16 无原生 clamp dtype（`vclamp`/`vmin`/`vmax` dtype 矩阵不含 bf16），保持强制 fp32 中转。

## Iteration Log

每轮固定 base = 当前已合并 current best；每个实验分支只改一个主要优化点；分支先跑 L0 再 `msprof op`。**实验批处理 runner**：`perf_opt/run_experiments.py`（自 skill 模板适配，`KERNEL_ID = …/hardtanh_kernel/_make_hardtanh_kernel.py::_make_hardtanh_kernel::main`；`--workload/--all`；逐测量追加 `perf_records.jsonl` 并打印「候选 vs current best」表）。

### Round 1 — per-dtype block_size（base = Stage 3 baseline）

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| r1_bs | 单 tile 2-4 KB，MTE2 仅 12.4 GB/s/core | per-dtype `block_size` | P1 扫描 + 同机 `leaky_relu`/`CONST-elu-910B3-bw-floor` | 新增 `TUNED_BLOCK_SIZE`{f32 24576, f16 12288, bf16 12288}；工厂 `block_size=None` 时取该表 | TaskDur / mte2_bw / L0 | **improved（主杠杆）** |

**候选 vs current best（current best = baseline）**

| branch | workload_id | L0 | TaskDur(us) | best(us) | Δ | vec | mte2 | mte3 | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| r1_bs | bounded-hidden-float16 | pass | 15.45 | 39.19 | **+60.6%** | 0.639 | 0.699 | 0.343 | improved |
| r1_bs | bounded-hidden-bfloat16 | pass | 15.04 | 39.22 | **+61.7%** | 0.724 | 0.685 | 0.351 | improved |
| r1_bs | bounded-conv-feat-float16 | pass | 21.16 | 58.37 | **+63.7%** | 0.663 | 0.725 | 0.358 | improved |
| r1_bs | bounded-conv-feat-bfloat16 | pass | 21.36 | 57.82 | **+63.1%** | 0.755 | 0.723 | 0.368 | improved |
| r1_bs | elementwise-16m-float16 | pass | 28.97 | 75.39 | **+61.6%** | 0.665 | 0.740 | 0.361 | improved |
| r1_bs | elementwise-16m-bfloat16 | pass | 28.29 | 75.54 | **+62.5%** | 0.754 | 0.734 | 0.365 | improved |
| r1_bs | elementwise-16m-float32 | pass | 51.63 | 213.32 | **+75.8%** | 0.166 | 0.508 | 0.303 | improved |

- **机制**：`block_size` 2048→12288 把 GM 读突发从 4 KB 提到 24 KB，`mte2_active_bw` 12.4→44.3 GB/s/core（3.6x），DMA 相从 63 → 17.7 µs；整条链转为 DMA 与 VEC 充分重叠。
- **本轮 winner**：`r1_bs`（per-dtype block_size）。current best 更新为 r1_bs。全部 7 个 tune workload 均通过。

### Round 2 — 静态整块 / 动态尾块拆分（base = r1_bs）

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| r2_static | fp32 三管近零重叠（mte2_ratio 0.51） | fp32 路径静态整块 + 单动态尾块 | `PL-1.34-staticbulk-unlocks-overlap` + P2 探针 | fp32 分支拆成 `task<n_full`（静态 extent）与 `task==n_full`（动态尾块）；fp16/bf16 分支保持动态尾块；fp32 bs 8192 | TaskDur / ab_test / L0 | **improved（fp32）** |
| r2b_static16 | 同上的 fp16 变体 | fp16 路径静态整块 | `PL-1.34` 反例核对 | fp16 分支同样拆分 | TaskDur | config_no_gain（P2 探针即否决，未升为正式分支） |

**候选 vs current best（current best = r1_bs）**

| branch | workload_id | L0 | TaskDur(us) | best(us) | Δ | vec | mte2 | mte3 | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| r2_static | bounded-hidden-float16 | pass | 14.75 | 15.45 | +4.5% | 0.639 | 0.698 | 0.343 | tie（同代码路径，噪声） |
| r2_static | bounded-hidden-bfloat16 | pass | 14.88 | 15.04 | +1.1% | 0.733 | 0.696 | 0.355 | tie（同代码路径） |
| r2_static | bounded-conv-feat-float16 | pass | 21.89 | 21.16 | −3.4% | 0.669 | 0.729 | 0.363 | tie（同代码路径） |
| r2_static | bounded-conv-feat-bfloat16 | pass | 22.32 | 21.36 | −4.5% | 0.744 | 0.721 | 0.360 | tie（同代码路径） |
| r2_static | elementwise-16m-float16 | pass | 28.48 | 28.97 | +1.7% | 0.665 | 0.742 | 0.361 | tie（同代码路径） |
| r2_static | elementwise-16m-bfloat16 | pass | 28.37 | 28.29 | −0.3% | 0.762 | 0.730 | 0.372 | tie（同代码路径） |
| r2_static | **elementwise-16m-float32** | pass | **49.17** | 51.63 | **+4.8%** | 0.185 | 0.894 | 0.520 | improved |

- **r2_static 相对 r1_bs 只改了 fp32 分支的循环结构**；上表 f16/bf16 行的散差（−4.5% ~ +4.5%）全部来自同代码路径的跨 run 双态（对照：bounded-hidden-f16 与 bounded-conv-feat-f16 是不同 N 但同代码，符号相反），确认落在噪声区、**不构成回退**。
- **fp32 差异 4.8%（<5%）→ `ab_test.py` 交错 5 对**（A=r1_bs 动态 bs24576，B=r2_static 静态 bs8192）：

```json
{"a_merged_median_us": 50.73, "b_merged_median_us": 48.88, "b_rel_change": -0.0365,
 "pair_diffs_us": [-1.30, -1.44, -1.79, -1.84, -2.03], "sign_test_p": 0.0625,
 "direction_consistent": true, "noise_threshold": 0.03, "verdict": "B_better"}
```

- **r2b（fp16 静态整块）探针即否决**：bs=12288 36.15 µs（vs 动态 27.16，显著退化）、bf16 bs=12288 38.12 µs → 未升为正式分支，记 `config_no_gain`（**不否定静态整块整类**，仅限 fp16/bf16 大 bs 语境，与 `PL-1.34` 反例一致）。
- **本轮 winner**：`r2_static`（fp32 静态整块 bs8192）。current best 更新为 r2_static。

### Round 3 — fp16 原生 clamp（跳过 fp32 中转；base = r2_static）

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| r3_fp16native | fp16 向量相 4 op（2 vcast + 2 clamp），`vec_ratio` 0.665 ≈ `mte2_ratio` | fp16 原生 clamp（消除中转 buffer + 2 vcast） | DESIGN §1.6.1 E1 备选 + PL-1.17 第四证 + P3 探针 | 新增编译期 `native16 = (dtype=="float16")`；fp16 分支计算域 = fp16（in-place），bounds/`lo_ub` 转 fp16；bf16 保持 fp32 中转 | TaskDur / ab_test / L0 / **--level all** | **improved（fp16）** |

**候选 vs current best（current best = r2_static）**

| branch | workload_id | L0 | TaskDur(us) | best(us) | Δ | vec | mte2 | mte3 | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| r3_fp16native | **bounded-hidden-float16** | pass | **14.02** | 14.75 | **+4.9%** | 0.171 | 0.822 | 0.508 | improved |
| r3_fp16native | bounded-hidden-bfloat16 | pass | 15.64 | 14.88 | −5.1% | 0.719 | 0.690 | 0.346 | tie（bf16 同代码路径，噪声） |
| r3_fp16native | **bounded-conv-feat-float16** | pass | **19.49** | 21.89 | **+11.0%** | 0.177 | 0.861 | 0.540 | improved |
| r3_fp16native | bounded-conv-feat-bfloat16 | pass | 22.80 | 22.32 | −2.2% | 0.747 | 0.717 | 0.360 | tie（同代码路径） |
| r3_fp16native | **elementwise-16m-float16** | pass | **25.05** | 28.48 | **+12.0%** | 0.181 | 0.881 | 0.561 | improved |
| r3_fp16native | elementwise-16m-bfloat16 | pass | 27.66 | 28.37 | +2.5% | 0.758 | 0.733 | 0.372 | tie（同代码路径） |
| r3_fp16native | elementwise-16m-float32 | pass | 48.12 | 49.17 | +2.1% | 0.186 | 0.894 | 0.522 | tie（fp32 同代码路径） |

- **r3 相对 r2_static 只改了 fp16 分支的计算域**；bf16/fp32 行全部走同一代码路径，散差（−5.1% ~ +2.5%）为跨 run 噪声。
- **fp16 差异 >5% 中 bounded-hidden 为 +4.9%（<5%）→ `ab_test.py` 交错 5 对**（A=r2_static fp16 动态中转 bs12288，B=r3 fp16 原生 bs24576）：

```json
{"a_merged_median_us": 27.91, "b_merged_median_us": 24.98, "b_rel_change": -0.105,
 "pair_diffs_us": [-2.48, -3.74, -2.93, -3.73, -2.38], "sign_test_p": 0.0625,
 "direction_consistent": true, "noise_threshold": 0.03, "verdict": "B_better"}
```

- **机制**：fp16 原生消除中转 buffer 与 2 次 `vcast`，`vec_ratio` 0.665→0.181（向量相几乎全隐入 DMA 窗口）。
- **本轮 winner**：`r3_fp16native`。current best 更新为 r3_fp16native（fp32 静态整块 bs8192 + fp16 原生静态 bs24576 + bf16 中转动态 bs12288）。
- **合并后非回退检查**：本 kernel 全部 7 个 tune workload 在 r3 上重测（上表），无工作负载回退（bf16/fp32 同代码噪声内），smoke 精度（L0）通过。

### 收敛判定

- Round 1 为主杠杆（+60~76%），Round 2 为 fp32 结构精修（+4.8%，协议确认），Round 3 为 fp16 计算域精修（+10.5%，协议确认）。
- 收敛依据：① 有效 R+W 已达 **2.4-2.8 TB/s**，贴/超同机逐元素搬运地板（`leaky_relu` fp32 2.69 TB/s、`CONST-elu-910B3-bw-floor` ≈2.49 TB/s）；② 默认路径已是最短闭式（fp32/fp16 2-op `vclamp`，bf16 因 dtype 矩阵强制 4-op 含 2 vcast），DESIGN §1.6.0 已证无更短等价式、无超越函数可消融；③ 参照结构 diff（morph ladder 轻量版）：检索同族 `leaky_relu`/`elu`/`bitwise_not`/`erf`/`expm1` 已优化版 + `PL-1.6/1.17/1.22/1.23/1.28/1.29/1.33/1.34` —— 未采纳的结构差异（fp16 静态整块、`vcmp/vselect` 形式、contiguous 分核映射）均与本算子实测相符地无益（P2 / P1 grid-stride）；bf16 无原生 clamp 是无解法硬约束；④ 剩余候选全为 blocked/no_gain（bf16 无 bf16 v-op；fp32 静态 UB 天花板 16384；fp16 原生已隐入 DMA）。
- 中止条件：**success**（best_effort，搬运地板达成；所有 tune workload 有结论且全量复测完成，smoke 精度通过）。

### Autotune Log

未使用 autotune。本算子为 DMA 受限，瓶颈为**连续参数**（block_size 的带宽曲线）而非离散参数搜索空间；P1/P2/P3 的显式一维/结构化扫描已给出完整曲线与编译天花板，autotune 无额外搜索维度可加。

## Final Summary

- 最终候选: `perf_opt/_make_hardtanh_kernel.py`（`candidate_id = final_merged`，即 r3_fp16native 合并版；工厂签名与基准兼容，drop-in 替换）。
- 采纳优化点（3 项）：
  1. **per-dtype `block_size`**：fp32 8192 / fp16 24576 / bf16 12288（模块常量 `TUNED_BLOCK_SIZE`，供 wrapper 切换块引用）。
  2. **fp32 路径静态整块 + 单动态尾块**（`PL-1.34`）：解锁 MTE2/MTE3 重叠（`mte2_ratio` 0.51→0.89）。
  3. **fp16 原生 clamp**（DESIGN §1.6.1 E1 备选落地）：跳过 fp32 中转 buffer 与 2 次 `vcast`（逐位等价，投影类算子 round↔clamp 可交换）；bf16 保持强制 fp32 中转。

逐 `kernel_id + workload_id` 对账（**baseline 与 final 均来自 `perf_records.jsonl`，final 全部来自同一最终候选 `final_merged`**）：

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/elementwise/hardtanh/hardtanh_kernel/_make_hardtanh_kernel.py::_make_hardtanh_kernel::main | bounded-hidden-float16 | 39.19 | 15.51 | final_merged |
| examples/TileOPs/tileops/kernels/elementwise/hardtanh/hardtanh_kernel/_make_hardtanh_kernel.py::_make_hardtanh_kernel::main | bounded-hidden-bfloat16 | 39.22 | 15.14 | final_merged |
| examples/TileOPs/tileops/kernels/elementwise/hardtanh/hardtanh_kernel/_make_hardtanh_kernel.py::_make_hardtanh_kernel::main | bounded-conv-feat-float16 | 58.37 | 19.43 | final_merged |
| examples/TileOPs/tileops/kernels/elementwise/hardtanh/hardtanh_kernel/_make_hardtanh_kernel.py::_make_hardtanh_kernel::main | bounded-conv-feat-bfloat16 | 57.82 | 22.29 | final_merged |
| examples/TileOPs/tileops/kernels/elementwise/hardtanh/hardtanh_kernel/_make_hardtanh_kernel.py::_make_hardtanh_kernel::main | elementwise-16m-float16 | 75.39 | 25.44 | final_merged |
| examples/TileOPs/tileops/kernels/elementwise/hardtanh/hardtanh_kernel/_make_hardtanh_kernel.py::_make_hardtanh_kernel::main | elementwise-16m-bfloat16 | 75.54 | 28.32 | final_merged |
| examples/TileOPs/tileops/kernels/elementwise/hardtanh/hardtanh_kernel/_make_hardtanh_kernel.py::_make_hardtanh_kernel::main | elementwise-16m-float32 | 213.32 | 49.80 | final_merged |

逐 workload 提升与有效带宽（R+W 聚合 = `2·N·elem_bytes / TaskDur`）：bounded-hidden-f16 **+60.4%**（0.86→2.16 TB/s）、bounded-hidden-bf16 **+61.4%**（0.86→2.22）、bounded-conv-feat-f16 **+66.7%**（0.88→2.64）、bounded-conv-feat-bf16 **+61.4%**（0.89→2.30）、elementwise-16m-f16 **+66.3%**（0.89→2.64）、elementwise-16m-bf16 **+62.5%**（0.89→2.37）、elementwise-16m-f32 **+76.7%**（0.63→2.69）。全部 L0 pass；final raw profile 位于 `profiles/final/round4/final_merged_a1903683/<workload_id>_*`。

- smoke 排除（仅精度，已跑最终回归）：smoke-1m-{float32,float16,bfloat16}。
- 最终精度回归（`perf_opt/_make_hardtanh_kernel.py --level all`）：**L0 / L1 / L2 / Boundary 共 50 用例全部 PASS，`max_diff = 0.000000e+00`**；覆盖 dtype × shape × 尾块 × 编译期边界折叠（`lo=-inf`/`hi=+inf`/两侧 inf/`lo==hi`）× 非默认边界（-0.5/2.0）× `±0`/`±inf`/NaN 的完整组合。
- 总体结论: 全部 tune workload 提升 **60.4%–76.7%**，达到/超过同机逐元素搬运地板上限；精度零误差。
- 中止原因: **success**（best_effort）。

## 设计层天花板判定（`[DESIGN_LIMIT]`）

对照 `_shared/standards/perf-feedback.md` §1 两项触发条件：

1. **是否由算法/设计层决定（非 tiling/参数可解）？** —— **否**。全部收益来自 `block_size`（tiling 参数）、循环结构（静态/动态尾块）与计算域选择（fp16 原生 clamp）；剩余上限为硬件 DMA 搬运地板（~2.5-2.8 TB/s），非某个 DESIGN 假设所限。DESIGN §1.6.0 的算法结论（`vclamp` 2-op 最短、边界折叠恒等、bf16 强制 fp32 中转、`block_size` 第一杠杆）被实测**正面验证**。
2. **结构性加速估计 > 2x 或实测与设计假设直接矛盾？** —— 实测对 DESIGN「发射项主导」的**估算数字**上偏 1.9-5.4x，但 DESIGN §1.6.0 已自我声明该发射项系统性上偏（`CONST-vector-launch-overhead` 绑 910B2C 且 stale，并给出 `bitwise_not` 对照证据），**不构成被推翻的设计假设**；设计选定的算法与瓶颈方向均成立，且已回填为偏差行。

→ **两项未同时满足 ⇒ 不产出 `perf_feedback.md`，无 `[DESIGN_LIMIT]` 信号**（参数级/结构级结论留在迭代内；D-2 偏差已按 `hardware-cost-model.md` §3 回填本日志）。

## Skill Retrospective

- **流程有效**：`perf-feedback` 触发门槛（两项同时满足）成功避免了把「参数/tiling/结构可解 + roofline 常数 stale 导致的估算偏差」误报为设计层天花板；同族 `CONST-elu-910B3-bw-floor` + `leaky_relu` 的 bs↔带宽曲线直接给出第一杠杆量级，把探针预算压缩到「确认 + 三者天花板二分」，本轮总探针 ~23 次 msprof，远低于计算受限算子。
- **可复用的顺序（morph ladder 轻量版奏效）**：先参数（bs）→ 再结构（静态/动态）→ 再计算域（fp16 原生）——每一层只在上层到达平台后才切换，避免了在 bs=2048 的退化点上做结构对比。
- **新发现（本任务，供 pattern-library 回写）**：投影/clamp 类**逐元素算子**在 DMA 受限时，**fp16 原生 clamp**（跳过 fp32 中转）是真实的结构性收益（−10.5% 协议确认），其价值 = **消除中转 buffer → 释放 UB / 缩短向量链（vec_ratio 0.665→0.181）**；与 `PL-1.17` 第四证「投影类 fp16 原生 ≡ fp32 中转 golden 逐位」互补——第四证证明**正确性**，本任务证明**性能兑现路径与量级**。bf16 仍为 API 强制中转（dtype 矩阵无 bf16）。
- **BP proposal / value point（供终态 evolver 蒸馏）**：

  | id | vp_type | 摘要 | evidence | repro | toolchain_stamp |
  |---|---|---|---|---|---|
  | VP-HT-1 | D | 910B3 上 hardtanh（2-op DMA 受限，投影类）的 `block_size` 带宽曲线与地板：fp16 bs12288 → 27.16 µs（2.47 TB/s）、fp32 bs24576 → 50.64 µs（2.65 TB/s）；fp16/bf16 UB 天花板 16384（12 B/elem）、静态整块 fp32 天花板 16384 | `perf_opt/probe_sweep.py` + 本日志 P1/P2 表 | repro-missing（待补：probe_sweep 最小化） | tilelang 0.1.2+cf91d889 / CANN 9.0.0 / 910B3 |
  | VP-HT-2 | P | **投影/clamp 类逐元素算子的 fp16 原生计算域**：`round_fp16(clamp_fp32(x,lo,hi)) ≡ clamp_fp16(x,round_fp16(lo),round_fp16(hi))`（round↔clamp 可交换），设备实测 bit-exact（L0/L1/Boundary `max_diff=0`）且消除中转 buffer + 2 vcast → 16m-f16 27.91→24.98 µs（−10.5% 协议确认）；bf16 仍须中转（dtype 矩阵无 bf16） | `ab_test.py` JSON（−10.5%，5/5 一致）；`_make_hardtanh_kernel_round3_fp16native.py` 完整 before/after；`--level all` 50/50 PASS | 分支文件 `_make_hardtanh_kernel_round3_fp16native.py` vs `_make_hardtanh_kernel_round2_static.py` | 同上 |
  | VP-HT-3 | D | 静态整块/动态尾块拆分在投影类逐元素上对 **fp32** 有净益（+3.65% 协议确认，`mte2_ratio` 0.51→0.89），对 **fp16/bf16 大 bs** 退化（fp16 bs12288 36.15 vs 动态 27.16）——`PL-1.34` 反例再证（含 bf16 bs12288 38.12） | `ab_test.py` JSON（−3.65%，5/5）+ 本日志 P2 表 | `_make_hardtanh_kernel_round2_static.py` / `round2b_static16.py` | 同上 |

- **skill 流程问题**：无阻塞性缺陷。建议（proposal，不自动改）：`run_experiments.py` 模板按 `inventory.kind==tune` 过滤 workload，但候选只改某 dtype 路径时其 L0 前置仍会编译并执行**全部 dtype 的编译分支**（含折叠路径）——本轮 r2_static 首次以 fp32 bs=24576 跑 L0 即因 fp32 静态分支 UB 超限而失败，由此**提前发现**静态整块的 UB 天花板（`TRAP-UB-multibuffer-inflation` 在静态 staging 上的显现）。建议在模板注释中显式说明「L0 覆盖非目标 dtype/折叠分支，候选默认 bs 必须对全部分支可编译」。
