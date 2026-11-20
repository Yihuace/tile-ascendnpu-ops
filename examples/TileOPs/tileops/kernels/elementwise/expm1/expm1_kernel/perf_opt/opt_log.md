# expm1_fwd_kernel — Stage 4 性能调优日志

- 算子目录: `examples/TileOPs/tileops/kernels/elementwise/expm1/expm1_kernel/`
- 基准 (只读, Stage 3): `expm1_fwd_kernel.py`（工厂 `expm1_fwd_kernel(N, dtype, output_dtype=None, block_size=_BLOCK_SIZE)`，`_BLOCK_SIZE=2048`；分段 expm1 = |x|<0.5·ln2 的 6 项 Taylor + `vexp(x)-1`，16-op fp32 / 18-op fp16·bf16 v-prefix 链，`vcmp`+`vselect` 选路）
- 调优产物 (final): `perf_opt/expm1_fwd_kernel.py`（candidate_id `r1_deg2mm`，sha256 `f25684748c72…`）
- 性能目标: **best_effort**（`user_requirement`），无逐 workload 数值硬门槛
- 时延口径: **`msprof op` Task Duration(us)**（kernel-only，唯一口径；median of 20 launches，warm-up 5）
- 噪声阈值: 3%；<5% 差异按 `ab_test.py` 交错多 run 协议（本轮无需触发——所有采纳增益 ≫5%，平区项按机制证据判 tie）
- bench 入口: `examples/TileOPs/benchmarks/ops/bench_unary_elementwise.py::test_expm1_bench`（`workloads_to_params("Expm1FwdOp", include_extra=True)`）
- 回归入口: `perf_opt/expm1_fwd_kernel.py --level all`（L0/L1 阻塞，L2/Boundary 告警不阻塞）
- 硬件: Ascend910_9362（40 物理 Vector 核 = 20 AI core × 2），CANN 9.0.0，Developer 模式，tilelang 0.1.2+a2357253662babf317ada194080078f928d155b3
- `mode=full`；`budget.max_stage4_experiments` 未设置（默认 `max_rounds=10` / `max_experiments=30`）

## Workload Inventory 与 Baseline (Performance Test Data)

`workload_inventory.json` 从 bench 入口 `test_expm1_bench` 展开（manifest `elementwise.yaml::Expm1FwdOp` 的 4 组 workload × dtype），逐案例分类：

| kind | workload_id | shape | dtype | 处理 |
|---|---|---|---|---|
| smoke | smoke-1m-float32 | [1048576] | float32 | 仅精度回归（不调优） |
| smoke | smoke-1m-float16 | [1048576] | float16 | 仅精度回归（不调优） |
| smoke | smoke-1m-bfloat16 | [1048576] | bfloat16 | 仅精度回归（不调优） |
| tune | elementwise-16m-float16 | [4096,4096] | float16 | 调优 |
| tune | elementwise-16m-bfloat16 | [4096,4096] | bfloat16 | 调优 |
| tune | elementwise-16m-float32 | [4096,4096] | float32 | 调优 |
| tune | elementwise-64m-float16 | [8192,8192] | float16 | 调优 |
| tune | elementwise-64m-bfloat16 | [8192,8192] | bfloat16 | 调优 |
| tune | elementwise-256m-float16 | [16384,16384] | float16 | 调优 |
| tune | elementwise-256m-bfloat16 | [16384,16384] | bfloat16 | 调优 |

- skipped: 无。
- smoke 排除理由: pytest case id 含独立 `smoke` token（`smoke-1m-*`），按 skill Phase 1 排除出性能调优，仅作每次合并及最终精度回归。
- 全部 10 个案例实际调用同一个 Stage 3 kernel（`expm1_fwd_kernel`，prim_func `main`，`target_kernel_name="main"`），无多 kernel 分派。

Baseline（round 0，`phase=baseline`，kernel = 基准 `expm1_fwd_kernel.py`，工厂默认 `block_size=2048`）：

| kernel_id | workload_id | Task Duration(us) | AIV vec util | MTE2 busy | MTE3 busy | Scalar busy | L0 | raw profile |
|---|---|---:|---:|---:|---:|---:|---|---|
| …expm1_fwd_kernel.py::expm1_fwd_kernel::main | elementwise-16m-float16 | 194.2700 | 0.672 | 0.328 | 0.155 | 0.107 | pass | profiles/baseline/round0/baseline_*/elementwise-16m-float16_* |
| …expm1_fwd_kernel.py::expm1_fwd_kernel::main | elementwise-16m-bfloat16 | 194.5500 | 0.686 | 0.328 | 0.155 | 0.107 | pass | profiles/baseline/round0/… |
| …expm1_fwd_kernel.py::expm1_fwd_kernel::main | elementwise-16m-float32 | 221.6700 | 0.500 | 0.302 | 0.144 | 0.078 | pass | profiles/baseline/round0/… |
| …expm1_fwd_kernel.py::expm1_fwd_kernel::main | elementwise-64m-float16 | 770.4600 | 0.673 | 0.330 | 0.156 | 0.105 | pass | profiles/baseline/round0/… |
| …expm1_fwd_kernel.py::expm1_fwd_kernel::main | elementwise-64m-bfloat16 | 767.6100 | 0.688 | 0.329 | 0.156 | 0.105 | pass | profiles/baseline/round0/… |
| …expm1_fwd_kernel.py::expm1_fwd_kernel::main | elementwise-256m-float16 | 3072.7101 | 0.673 | 0.331 | 0.159 | 0.106 | pass | profiles/baseline/round0/… |
| …expm1_fwd_kernel.py::expm1_fwd_kernel::main | elementwise-256m-bfloat16 | 3072.6101 | 0.687 | 0.331 | 0.159 | 0.106 | pass | profiles/baseline/round0/… |

### Baseline 瓶颈判读（来自 raw `PipeUtilization`）

**纯 Vector 计算/发射受限**，非搬运墙：AIV 向量管线忙比 0.50–0.69，MTE2（GM→UB）仅 0.30–0.33，MTE3 0.14–0.16，Scalar 0.08–0.11。有效带宽对照：256m fp16 声明流量 1.074 GB / 3072.71µs = **0.35 TB/s**，仅为本机核→GM ≈1.06 TB/s 的 33% —— 距搬运地板尚远，瓶颈在 **16-op v-prefix 链的向量执行吞吐**。据此排序：先大流量 dtype（256m/64m f16、bf16），再 16m f32（同 shape 但访存翻倍、vec util 最低），与实测负担一致。

### 设计估算 vs 实测偏差（D-2 回填，DESIGN §1.6.0 roofline）

| workload | 设计估算下界(us) | baseline 实测(us) | 偏差 | 失准项 |
|---|---:|---:|---|---|
| elementwise-16m-fp32 | ≈1638 | 221.67 | 实测快 **7.4x** | 发射项：`CONST-vector-launch-overhead`（0.5µs/op）绑 910B2C 且 stale，本设备 910_9362 实测普通 v-op ≈ 3–8µs/16m，严重上偏 |
| elementwise-16m-fp16 | ≈1843 | 194.27 | 实测快 **9.5x** | 发射项（同上） |
| elementwise-256m-fp16 | ≈29491 | 3072.71 | 实测快 **9.6x** | 发射项（同上） |

偏差方向一致且远超 2x → 设计把「发射项」绝对主导的模型在本设备上严重上偏；但**算法层结论（必须分段消抵消、必须自建逼近链）被本轮实测完全验证**，仅 op 数与 `block_size` 参数可优化——**不构成设计层天花板**（无 `[DESIGN_LIMIT]`，见文末）。

## Iteration Log

每轮固定 base = 当前已合并 current best；每个实验分支只改一个主要优化点；分支先跑 L0 再 `msprof op`；对 target workload 按 Task Duration 选 winner。诊断性 op 消融由 `perf_opt/probe_chain.py` 承担，结构候选由 `perf_opt/probe_struct.py` 承担（均**不计入 gate、不写 perf_records**，仅用于定位主导 op 与筛选结构）。

**实验批处理 runner**: `perf_opt/run_experiments.py`（自 skill 模板 `run_experiments_template.py` 适配，`KERNEL_ID = …/expm1_kernel/expm1_fwd_kernel.py::expm1_fwd_kernel::main`；`--workload/--all`；逐测量追加 `perf_records.jsonl` 并打印「候选 vs current best」表）。设备独立性探针 `perf_opt/_bench_kernel.py`、精度探针 `perf_opt/_check_acc.py`。

### 现象诊断（Round 1 前置）：逐 op 代价分解

同结构消融探针（`probe_chain.py`，16m fp16，bs=2048，msprof op，raw `profiles/probe/{full,no_select,short_horner,no_exp,exp_only,copy_only}`）：

| 变体 | Task Duration(us) | 归因 |
|---|---:|---|
| full（16-op 分段链） | 194.63 | — |
| no_select（`vselect`→`vadd`） | 131.49 | **`vselect` 边际 ≈ 63µs（≈8× 普通 op）** |
| short_horner（Horner 10→2，少 8 op） | 132.19 | 普通 v-op ≈ **7.8µs/op** |
| no_exp（`vexp`→`vmul`） | 200.27 | `vexp` 边际 ≤0（与 MTE2 重叠，**免费**） |
| exp_only（仅大分支 `vexp`+`vsub`） | 75.43 | ≈ 搬运地板 |
| copy_only（仅搬入/搬出+cast） | 74.91 | 搬运地板 |

**结论**：两条主导成本 —— ① `vselect`（≈63µs，与 `PL-1.29-vselect-cost` 的 6.5–8× 一致）；② 小 `|x|` Taylor 的 10 个 Horner op（≈78µs）。`vexp` 大分支被 MTE2 完全隐藏。候选优化点：

- **C1**：降低小分支多项式阶数（`T` 相应收紧）；
- **C2**：`block_size` 扫描（摊薄每 tile 固定开销，DESIGN §5.2 已列为 Stage 4 首选杠杆）；
- **C3**：换掉 `vselect` 的算术 clamp-mask 选路；
- **C4**：`tanh` 单式逼近（`expm1(x)=tanh(x/2)(e^x+1)`，falsify）；
- **C5**：结构对照（contiguous 分片 / num_cores / 静态整块拷贝）。

### Round 1 — 小分支降阶 + per-dtype block_size（采纳，C1+C2）

**现象**（current best = baseline）：纯 Vector 受限，16-op 链中 `vselect`(≈63µs) 与 10-op Horner(≈78µs) 主导；`block_size` 仅 2048（保守），per-tile 固定开销未摊薄。

**候选 vs current best（current best = baseline，`phase=candidate`，全部 msprof op Task Duration(us)，16m fp16，探针筛选 bs 值）**：见下 block_size / 结构两表。正式记录见 `perf_records.jsonl`（`phase=candidate`，candidate_id `r1_deg2mm`）。

**结构筛选（`probe_struct.py`，16m fp16，bs=2048；数学难度递增的候选）**：

| 候选 | 小分支 | 选路 | TaskDur(us) | L0/精度 | 结论 |
|---|---|---|---:|---|---|
| base | Taylor deg5, T=0.3466 | vcmp+vselect | 201.37 | pass | base |
| deg3_sel | Taylor deg3, T=0.11 | vcmp+vselect | 167.75 | pass（fp32 rel 1.30e-6） | valid |
| deg5_blend | Taylor deg5 | 算术 clamp-mask | 177.16 | **invalid（+inf→NaN）** | blocked（`inf-inf` 破 ±inf） |
| deg3_blendc | Taylor deg3 | 算术 clamp-mask+夹逼 | 153.08 | pass | valid（bs2048 快，但 bs 放大后反被 vselect 反超） |
| deg2mm_sel | **minimax deg2, T=0.05** | vcmp+vselect | 约 86（bs7168） | pass（fp32 rel 2.15e-6） | **winner** |
| tanh_id | `tanh(x/2)(e^x+1)` | 无分支 | 108.25 | **invalid（小 x rel=1.0、corner rel=0.19、+inf→NaN）** | blocked（设备 `vtanh`=(e^{2y}-1)/(e^{2y}+1) 小参数抵消，见下） |

- **C4 falsify（`tanh` 单式）**：`T.vtanh` Developer 模式实现为 `(exp(2y)-1)/(exp(2y)+1)`（`docs/Tilelang.language/数学操作/T.vtanh.md` §3），小参数 `y=x/2` 处 `exp(2y)-1` 灾难性抵消 → `L0-small-x-rel` 相对误差 **1.0**、corner `1e-7` rel **0.19**；且 `+inf` 需 `x/2` 夹逼但与 `e^x` 相乘产生 `nan`。**结论：单式 tanh 逼近对非奇函数 expm1 不可用**（与 erf 的奇函数场景本质不同——见 `PL-1.30` 适用边界「不适用于非奇函数」）。
- **C3 merge 失败机理**：算术 clamp-mask 混合在 `|x|` 大时 `small`(Taylor) 发散至 ±inf，`inf-inf` 产生 NaN，破坏 `±inf` 语义；加 `vclamp(g,±0.5)` 后可修复（`deg3_blendc` pass），但**混合本身 7 个普通 op 在 bs≥6144 时比 vselect 更贵**（bs7168：blendc 120.1 vs sel 98.7）——**本设备在本结构下 `vselect` 是更便宜的选路，非首要消除目标**（修正 `PL-1.29` 的泛化：该 6.5× 代价在**短链 + 大 bs** 下不再等价于「应优先消除」）。
- **C1 精度预算（minimax 定阶）**：小分支误差 `≈T^{K+1}/(K+2)!` 与大分支抵消误差 `≈1.18e-7·e^T/(e^T-1)` 两项平衡：
  - Taylor deg2 平衡于 T≈0.042 → max ≈3e-6；
  - **minimax deg2（3 系数）平衡于 T=0.05 → max ≈2.4e-6**（系数 `[0.9999999955, 0.5000625040, 0.1666845248]`，CPU minimax 求解）；
  - Taylor deg3/T=0.11 → 1.3e-6（7 op，多 2 op）；**选取 minimax deg2（5 op）以换取 ~13% 速度**，余量 4.5×（与同族 erf Stage 4 采纳的 3.4× 同量级）。T 再小则大分支误差上升、再大则小分支误差上升，均劣化。

**block_size 扫描（Autotune Log，C2；探针 bs 扫描，16m fp16 / fp32）**：

| 结构 / dtype | 候选 bs | TaskDur(us) | 备注 |
|---|---|---|---|
| baseline, f16 | 2048 / 4096 / 6144 / ≥8192 | 194.63 / 146.04 / 119.47 / UB overflow | 单调下降，6144 为可行上限 |
| baseline, f32 | 2048 / 4096 / 6144 / 8192 / 9216 / 10240 | 222.71 / 170.37 / 131.25 / 139.97 / 140.72 / UB overflow | 6144 最优 |
| deg2mm_sel, f16 | 7168 / 7424 / 8192 | 85.68 / 85.71 / UB overflow | 7168 起进入平台，选 7168（UB 余量安全） |
| deg2mm_sel, f32 | 7168 / 8192 / 9216 / 10240 | 119.5 / 103.31 / (≈116) / UB overflow | **8192 最优** |

**winner**: C1（minimax deg2 小分支，T=0.05）+ C2（per-dtype `block_size` = **f32 8192 / f16 7168 / bf16 7168**），落盘 `expm1_fwd_kernel_round1_deg2mm.py`，并升格为 `perf_opt/expm1_fwd_kernel.py`。

**候选 vs current best（current best = baseline，全部 msprof op Task Duration(us)，`perf_records.jsonl` `phase=candidate`）**：

| 候选 | workload | TaskDur(us) | AIV vec | MTE2 | vs best | L0 | 结论 |
|---|---|---:|---:|---:|---:|---|---|
| baseline | 16m-f16 | 194.27 | 0.672 | 0.328 | 0.0% | pass | base |
| baseline | 16m-bf16 | 194.55 | 0.686 | 0.328 | 0.0% | pass | base |
| baseline | 16m-f32 | 221.67 | 0.500 | 0.302 | 0.0% | pass | base |
| baseline | 64m-f16 | 770.46 | 0.673 | 0.330 | 0.0% | pass | base |
| baseline | 64m-bf16 | 767.61 | 0.688 | 0.329 | 0.0% | pass | base |
| baseline | 256m-f16 | 3072.71 | 0.673 | 0.331 | 0.0% | pass | base |
| baseline | 256m-bf16 | 3072.61 | 0.687 | 0.331 | 0.0% | pass | base |
| r1_deg2mm | 16m-f16 | 86.28 | 0.828 | 0.262 | **+55.6%** | pass | **winner** |
| r1_deg2mm | 16m-bf16 | 86.18 | 0.858 | 0.262 | **+55.7%** | pass | winner |
| r1_deg2mm | 16m-f32 | 103.36 | 0.601 | 0.226 | **+53.4%** | pass | winner |
| r1_deg2mm | 64m-f16 | 339.59 | 0.831 | 0.274 | **+55.9%** | pass | winner |
| r1_deg2mm | 64m-bf16 | 338.43 | 0.858 | 0.272 | **+55.9%** | pass | winner |
| r1_deg2mm | 256m-f16 | 1387.07 | 0.809 | 0.295 | **+54.9%** | pass | winner |
| r1_deg2mm | 256m-bf16 | 1391.49 | 0.831 | 0.298 | **+54.7%** | pass | winner |

### Round 2 — 结构对照（未采纳 → plateau）

**现象**（current best = r1_deg2mm）：kernel 转 vector-bound（16m f16 vec 0.83），但同结构 floor（`copy_only` bs7168）仅 **34.4µs** —— 说明仍是 op 主导，尚有理论空间；按 pattern-library 对照同族先例逐个验证结构差异。

**候选 vs current best（current best = r1_deg2mm，探针，16m f16 bs7168）**：

| 候选 | TaskDur(us) | vs best | 结论 |
|---|---:|---:|---|
| r1_deg2mm（stride，cores=40） | 85.56 | 0.0% | base |
| contig 分片（每核连续 tile 段） | 85.89 | -0.4% | **tie（<3%）** —— 同族 lerp `PL-1.6` 的 grid-stride 更优结论复现 |
| cores=20 | 166.86 | -95% | 并发不足 |
| cores=60 | 113.52 | -33% | 派发开销过高 |
| cores=80 | 87.06 | -1.8% | 无益（>40 物理核） |
| deg2mm_nosel（诊断：去 `vselect`+`vabs`+`vcmp`） | 62.76 | +26.6% | 诊断：选路区 ≈23µs，但无更便宜等价机制 |

**结论**：C5 全部 tie/无益；`num_cores=40` 已是甜点（40 物理核）。选路区 23µs 为结构性下限（算术混合更贵、tanh 单式不可用、vmin/vmax 无可用的单调选择关系），Horner 已降至精度允许的最小阶（deg2 minimax）。**参照结构 diff 检查（T-1 morph ladder）**：同族已知最优实现（erf perf_opt 的 per-dtype bs + 无掩码奇函数、bitwise_not perf_opt 的 static-bulk/contiguous、lerp_tensor 的 loads-first/contiguous）的差异点——① per-dtype bs 已采纳；② 无掩码奇形式**不适用于非奇函数 expm1**（已 falsify）；③ static-bulk 拷贝/thunk 派生于搬运墙优化，本 kernel 为 op-bound（vec 0.83、MTE2 0.26），且同族 erf R2 的 static 变体在 bs 放大后触发 UB overflow，**无适用性**；④ contiguous 已测 tie。**未发现剩余可移植结构差异**。

**中止条件**: `success`（best_effort：以通过核内合并与全量非回退检查的实测最优结果作为结论；全部 tune workload 有调优结论 + 最终全量记录 + smoke 精度通过）。

## Autotune Log

未使用 TileLang autotune API；`block_size`（及 `num_cores`）以探针手动粗搜 + 邻域精搜完成，search space 与结果见 Round 1/2 两表。winner `block_size` 已用 `msprof op` 复测（`phase=candidate`→`phase=final` 一致：16m f16 86.28→86.23，波动 <0.1%）。`num_cores` 曲线（{20:166.9, 40:85.6, 60:113.5, 80:87.1}）确认 40 为甜点，未在 final 中引入参数化。

## Final Summary

**最终候选**: `perf_opt/expm1_fwd_kernel.py`（candidate_id `r1_deg2mm`，sha256 `f25684748c72…`）= Round 1 winner；Round 2 无改动合入。
**`phase=final` 记录**: 全部 7 个 tune workload 均在同一最终候选上重新测得（`perf_records.jsonl`，`phase=final`，同一 `candidate_id` `r1_deg2mm`，`artifact_path` = `expm1_fwd_kernel.py`，sha256 与文件一致）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/elementwise/expm1/expm1_kernel/expm1_fwd_kernel.py::expm1_fwd_kernel::main | elementwise-16m-float16 | 194.2700 | 86.2300 | r1_deg2mm |
| examples/TileOPs/tileops/kernels/elementwise/expm1/expm1_kernel/expm1_fwd_kernel.py::expm1_fwd_kernel::main | elementwise-16m-bfloat16 | 194.5500 | 86.1400 | r1_deg2mm |
| examples/TileOPs/tileops/kernels/elementwise/expm1/expm1_kernel/expm1_fwd_kernel.py::expm1_fwd_kernel::main | elementwise-16m-float32 | 221.6700 | 103.2000 | r1_deg2mm |
| examples/TileOPs/tileops/kernels/elementwise/expm1/expm1_kernel/expm1_fwd_kernel.py::expm1_fwd_kernel::main | elementwise-64m-float16 | 770.4600 | 339.7500 | r1_deg2mm |
| examples/TileOPs/tileops/kernels/elementwise/expm1/expm1_kernel/expm1_fwd_kernel.py::expm1_fwd_kernel::main | elementwise-64m-bfloat16 | 767.6100 | 337.8600 | r1_deg2mm |
| examples/TileOPs/tileops/kernels/elementwise/expm1/expm1_kernel/expm1_fwd_kernel.py::expm1_fwd_kernel::main | elementwise-256m-float16 | 3072.7101 | 1389.1600 | r1_deg2mm |
| examples/TileOPs/tileops/kernels/elementwise/expm1/expm1_kernel/expm1_fwd_kernel.py::expm1_fwd_kernel::main | elementwise-256m-bfloat16 | 3072.6101 | 1391.8100 | r1_deg2mm |

逐 workload 提升：16m-f16 **-55.6%（2.25×）**、16m-bf16 **-55.7%（2.26×）**、16m-f32 **-53.4%（2.15×）**、64m-f16 **-55.9%（2.27×）**、64m-bf16 **-56.0%（2.27×）**、256m-f16 **-54.8%（2.21×）**、256m-bf16 **-54.7%（2.21×）**。profile 证据：`profiles/final/round1/r1_deg2mm_*/…`；最终 vec 利用率 0.81–0.86（f16/bf16）/ 0.60（f32）。

### smoke 与 skipped 排除

- smoke（smoke-1m-f32/f16/bf16）：不做性能调优；最终精度回归 **pass**（`perf_opt/expm1_fwd_kernel.py --level all`：`L0-smoke` fp32 `max_rel=2.216e-06`、fp16 `9.009e-04`、bf16 `0.0`；L0/L1/L2/Boundary 全层级 pass）。fp16 的 `max_rel≈9e-4` 是 fp16 输出域 1-ulp 舍入在 `|g|~1e-3` 小值处的表示效应（`atol=1e-3` 覆盖；fp32 预舍入误差 2.2e-6 ≪ 0.5 fp16 ulp 2.4e-4，故 fp16 结果与 golden 至多差 1 ulp，**必然通过**）。bf16 逐位一致。
- skipped：无。
- 失败/阻塞分支（不写 perf_records）：`tanh_id`（精度 invalid）、`deg5_blend`（±inf invalid）、`deg2_blendc`（bs2048 下 fp16 `max_rel=8.98e-4` 逼近 rtol，未采纳）。

### 总体结论与遗留

- 本算子从「16-op 分段链的 Vector 发射受限」改为「12/14-op（minimax deg2 小分支 + vselect）链 + per-dtype 大 block_size」，全部 7 个 tune workload 取得 **2.15×–2.27×** 加速（几何平均 ≈2.23×）；精度余量 4.5×（fp32 `max_rel=2.2e-6` vs rtol 1e-5），与同族 erf Stage 4 采纳的 3.4× 同量级（best_effort 下取速度更优解）。
- **遗留/权衡**：若后续要求 ≥10× 精度余量，可切 `deg3 minimax, T=0.13`（7 op，fp32 余量 10.3×，速度约为本版的 0.86×，实测 16m-f16 ≈98.7µs）；该权衡已实测记录。选路区（`abs+vcmp+vselect` ≈23µs）为本结构族实测下限，本轮未找到更便宜且保持 IEEE `±inf/NaN` 语义的等价机制。
- **设计层天花板判定**：**不构成**（无 `[DESIGN_LIMIT]`，不产出 `perf_feedback.md`）。触发条件①（设计层归因）不满足——性能不足仅源于 op 数与 `block_size` 参数，`DESIGN.md` 已把 `block_size` 列为 Stage 4 首选杠杆、且算法族（分段 expm1）被实测验证为正确选择；触发条件②（结构性加速 >2× 或实测与设计假设直接矛盾）虽满足（设计估算 vs 实测 ≥7.4×、且设计对「小分支需 deg5」的定阶被 minimax deg2 推翻），但**仅条件②满足、条件①不满足时禁止产出**（两条件须同时满足）。2.2× 加速已在同算法族内实现，无需设计修订。

## Skill Retrospective

### 流程问题

1. **探针脚本的 `for ... in range(...)` 陷阱复现**：在 `@T.prim_func` 内用 Python `for` 做编译期展开会被 TVMScript 解析器转成 TIR 循环（erf 复盘已记，本条为同族复现）——本任务改用**编译期 `if deg >= k:` 守卫手写展开**（deg 为 Python 常量），成功。建议在 traps-compiler.md `TRAP-tvm-parser-rules` 条目补记「编译期展开须手写或用 Python 常量 `if` 守卫，**不可用 `for range`**」的第二证据。
2. **精度探针的 fp16/bf16 输入量化陷阱**：`_check_acc.py` 初版对 fp16/bf16 用 fp32 randn 作 golden 输入、却把 `.to(fp16)` 后的张量喂给 kernel，导致 5–10 fp16 ulp 的伪误差（误判 deg3_sel/deg5_blend fp16 FAIL）。**修正**：先在目标 dtype 量化输入、kernel 与 golden 共用同一量化张量（与真实 harness `_make_input` 一致）。建议：`profile-collection.md` 或 `_check_acc` 模板显式写明「golden 与 kernel 输入必须同源同 dtype」。
3. **`max_rel` 对 fp16 小值过于敏感、与实际门禁不一致**：fp16 输出在 `|g|~1e-3` 处 1-ulp 舍入即产生 `max_rel≈9e-4`，但实际 `assert_close(rtol=1e-3, atol=1e-3)` 由 atol 覆盖、必然通过。建议精度探针**同时报告 `assert_close` 判定与「是否仅 1-ulp 差异」**，避免用裸 `max_rel` 误杀合法候选。
4. **算术选路（clamp-mask blend）的逆序结论**：`vselect` 在 bs=2048 的短链下确实比 7-op 算术混合贵（167.75 vs 153.08），但在 **bs≥6144** 下反被反超（98.7 vs 124.5）——「`vselect` 6.5× 代价应优先消除」不可无条件泛化，**结论必须绑定 block_size/链长**。

### 建议修改（BP proposal）

- `BP_expm1_minimax_degree`（P 类，vp_type=P）：**「分段逼近的小分支阶数应按 minimax 误差预算定阶，而非沿用 Taylor」**——同一 `T` 下 minimax 比 Taylor 低约 2–4× 误差；expm1 实证：Taylor deg3/T=0.11（7 op，1.3e-6）可被 **minimax deg2/T=0.05（5 op，2.2e-6）** 替代，省 2 op 且余量仍 4.5×。定阶准则 = 平衡「小分支截断 `T^{K+1}/(K+2)!`」与「大分支抵消 `ε_exp·e^T/(e^T-1)`」。
- `BP_vselect-cost-context`（D 类，vp_type=D）：**`vselect` 相对代价依赖 block_size/链长**——16m fp16 实测：bs2048 下 `vselect` ≈63µs（≈8× 普通 op），但 bs7168 下其相对优势使 7-op 算术混合反更贵（deg3：sel 98.7 vs blend 124.5）；同结构 nosel 与 sel 之差 = 选路区 ≈23µs（bs7168）。

### 价值点（按 vp_type，含证据三件套）

| vp_type | 条目 | evidence（实测） | repro | toolchain_stamp |
|---|---|---|---|---|
| D | expm1 16-op 链逐 op 代价：`vselect`≈63µs（bs2048）/选路区≈23µs（bs7168）；普通 v-op≈7.8µs；`vexp` 与 MTE2 重叠≈免费；copy floor bs2048 74.9µs → bs7168 34.4µs | `perf_opt/probe_chain.py`（194.63/131.49/132.19/200.27/75.43/74.91µs）+ `probe_struct.py` deg2mm sel/nosel（85.56/62.76µs），raw `profiles/probe/*` | repro/PL-1.29-vselect-cost.py（复用设备单 op 代价，已跑通）；本任务新数据 append `elementwise.md` | tilelang 0.1.2+a2357253662 / CANN 9.0.0 / Ascend910_9362 |
| P | minimax 定阶：分段逼近小分支用 minimax 而非 Taylor，平衡两分支误差预算；expm1：minimax deg2/T=0.05（5 op，2.2e-6）替代 Taylor deg3（7 op，1.3e-6） | 16m-f16 98.7→86.2µs（+13%）；`_check_acc.py` fp32 `max_rel=2.15e-6`；CPU minimax 求解脚本 | 待回填 `repro/`（next same-family task；见本表 evidence） | 同上 |
| P | 非奇函数（expm1 等）**不可**用 `tanh(x/2)(e^x+1)` 单式逼近：设备 `vtanh`=(e^{2y}-1)/(e^{2y}+1) 小参数抵消，`L0-small-x-rel` rel=1.0 | `_check_acc.py --mode tanh_id`（small-x rel=1.0、corner rel=0.19、+inf→NaN）；16m-f16 bs2048 108.25µs | 复用 `repro/PL-1.30-tanh-odd-poly-erf.py` 反例形态（本任务判为不适用） | 同上 |
| C | 案例：expm1 Stage 4「Taylor deg5+bs2048 → minimax deg2+per-dtype bs」2.15–2.27× 调优档 | `perf_opt/opt_log.md` + `perf_records.jsonl`（7/7 tune workload） | — | 同上 |

回写位置（Phase 4 ED-C，已执行）：`pattern-library/elementwise.md` —— ① `PL-1.29-vselect-cost` 追加「block_size/链长语境修正」；② `PL-1.30-tanh-odd-poly-erf` 追加「非奇函数不可用」反向边界；③ 新增 `PL-1.32-minimax-degree-piecewise`（P，含 front-matter + repro）。`pattern-library/constants.md` —— `CONST-vector-op-cost-910_9362` 追加 bs7168 语境代价表 + copy floor 随 bs 下降数据。`pattern-library/INDEX.md` 登记 PL-1.32。repro：`repro/PL-1.32-minimax-degree-piecewise.py`（CPU，实跑 REPRO PASS）。`kb_lint.py`：0 failures。

## 工件清单

- `perf_opt/expm1_fwd_kernel.py`（final，drop-in；导出 `TUNED_BLOCK_SIZE = {"float32": 8192, "float16": 7168, "bfloat16": 7168}` 供 wrapper 切换块成对引用；对齐 `_T=0.05` 与 `_COEF` 模块常量）
- `perf_opt/expm1_fwd_kernel_round1_deg2mm.py`（Round 1 winner 分支，= final）
- `perf_opt/probe_chain.py`（逐 op 消融）、`perf_opt/probe_struct.py`（结构候选）
- `perf_opt/_bench_kernel.py`、`perf_opt/_check_acc.py`、`perf_opt/run_experiments.py`（runner）
- `perf_opt/opt_log.md`、`perf_opt/perf_records.jsonl`、`perf_opt/workload_inventory.json`
- `perf_opt/profiles/`（raw msprof：`baseline/candidate/final`）、`perf_opt/logs/`（stdout/stderr）

> 说明：诊断性探针的 raw msprof 目录（原 `profiles/probe/*`，约 1.4GB）在数字提取入本表后已清理以控制体积；探针脚本 `probe_chain.py` / `probe_struct.py` 保留，可一键复跑。gate 依赖的 `baseline/candidate/final` raw 目录全部保留，`perf_records.jsonl` 的 `msprof_raw_path` 全部有效。
