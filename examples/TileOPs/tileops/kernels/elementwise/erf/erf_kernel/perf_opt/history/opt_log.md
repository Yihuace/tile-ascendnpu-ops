# Erf forward kernel — Stage 4 性能调优日志

- 算子目录: `examples/TileOPs/tileops/kernels/elementwise/erf/erf_kernel/`
- 基准 (只读, Stage 3): `erf_fwd_kernel.py`（sha256 `d0a1f7…`），工厂 `erf_fwd_kernel(N, dtype, output_dtype=None, block_size=_BLOCK_SIZE)`，A&S 7.1.26 自建逼近（21-op fp32 / 23-op fp16·bf16 v-prefix 链）
- 调优产物 (final): `perf_opt/erf_fwd_kernel.py`（sha256 `1222a61516a25e7a22f09868fbc3401c4a0f7a5be85b59d970c701d2552a5c79`）
- 性能目标: best_effort（`user_requirement`），无逐 workload 数值硬目标
- 时延口径: **`msprof op` Task Duration(us)**（kernel-only，唯一口径；median of 20 launches，warm-up 5）
- 噪声阈值: 3%
- bench 入口: `examples/TileOPs/benchmarks/ops/bench_unary_elementwise.py::test_erf_bench`（`workloads_to_params("ErfFwdOp", include_extra=True)`）
- 回归入口: `perf_opt/erf_fwd_kernel.py --level all`
- 硬件: Ascend910_9362（40 物理 Vector 核 = 20 AI core × 2），CANN 9.0.0，Developer 模式，tilelang 0.1.2+c3a94f1d561cd941ca19577d5ef74bd635bd292a
- `mode=full`；`budget.max_stage4_experiments` 未设置（默认 `max_rounds=10` / `max_experiments=30`）

## Workload Inventory 与 Baseline (Performance Test Data)

`workload_inventory.json` 从 bench 入口 `test_erf_bench` 展开（manifest `elementwise.yaml::ErfFwdOp` 的 4 组 workload × dtype），逐案例分类：

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
- smoke 排除理由: pytest case id 含独立 `smoke` token（`smoke-1m-*`），按 skill Phase 1 排除出性能调优，仅作每次合并及最终精度回归。无 `full` 标记与 smoke 冲突。
- 全部 10 个案例实际调用同一个 Stage 3 kernel（`erf_fwd_kernel`，prim_func `main`，`target_kernel_name="main"`），无多 kernel 分派。

Baseline（round 0，`phase=baseline`，kernel = 基准 `erf_fwd_kernel.py`，工厂默认 `block_size=2048`）：

| kernel_id | workload_id | Task Duration(us) | AIV vec util | MTE2 busy | MTE3 busy | Scalar busy | L0 | raw profile |
|---|---|---:|---:|---:|---:|---:|---|---|
| …erf_fwd_kernel.py::erf_fwd_kernel::main | elementwise-16m-float16 | 256.4451 | 78.72% | 24.62% | 12.01% | 11.04% | pass | profiles/baseline/round0/baseline_*/elementwise-16m-float16_* |
| …erf_fwd_kernel.py::erf_fwd_kernel::main | elementwise-16m-bfloat16 | 256.7151 | 79.81% | 24.68% | 12.00% | 11.05% | pass | profiles/baseline/round0/… |
| …erf_fwd_kernel.py::erf_fwd_kernel::main | elementwise-16m-float32 | 308.0962 | 62.61% | 21.66% | 10.62% | 7.93% | pass | profiles/baseline/round0/… |
| …erf_fwd_kernel.py::erf_fwd_kernel::main | elementwise-64m-float16 | 1015.6003 | 78.97% | 24.69% | 12.18% | 10.85% | pass | profiles/baseline/round0/… |
| …erf_fwd_kernel.py::erf_fwd_kernel::main | elementwise-64m-bfloat16 | 1016.8503 | 80.07% | 24.76% | 12.19% | 10.85% | pass | profiles/baseline/round0/… |
| …erf_fwd_kernel.py::erf_fwd_kernel::main | elementwise-256m-float16 | 4058.6312 | 78.87% | 24.82% | 12.34% | 10.93% | pass | profiles/baseline/round0/… |
| …erf_fwd_kernel.py::erf_fwd_kernel::main | elementwise-256m-bfloat16 | 4060.5111 | 80.04% | 24.82% | 12.33% | 10.93% | pass | profiles/baseline/round0/… |

### Baseline 瓶颈判读（来自 raw `PipeUtilization`）

**纯 Vector 计算/发射受限**，非 MTE2 搬运墙：AIV 向量管线忙比 62.6–80.1%，MTE2（GM→UB）仅 21.7–24.8%，MTE3 10.6–12.3%，Scalar 7.9–11.0%（各 pipe 忙比之和 >100% 说明存在跨 pipe 重叠）。有效带宽对照：256m fp16 声明流量 1.074 GB / 4058.63µs = **0.26 TB/s**，仅为单 die HBM 理论 1.64 TB/s 的 16%、核→GM 1.06 TB/s 的 25% —— 距搬运地板尚远，瓶颈在**21/23-op v-prefix 链的向量执行吞吐**（每元素 21–23 个 v-op）。据此排序：先大流量 dtype（256m/64m f16、bf16），再 16m f32（同 shape 但访存翻倍、vec util 最低），与实测负担一致。

### 设计估算 vs 实测偏差（D-2 回填，DESIGN §1.6.0 roofline）

| workload | 设计估算下界(us) | baseline 实测(us) | 偏差 | 失准项 |
|---|---:|---:|---|---|
| elementwise-16m-fp32 | ≈2150 | 308.10 | 实测快 **7.0x** | 发射项：`CONST-vector-launch-overhead`（0.5µs/op）绑 910B2C，本设备 910_9362 实测 per-op ≈ 0.055µs（≈9x 上偏） |
| elementwise-16m-fp16 | ≈2355 | 256.45 | 实测快 **9.2x** | 发射项（同上） |
| elementwise-256m-fp16 | ≈37683 | 4058.63 | 实测快 **9.3x** | 发射项（同上） |
| smoke-1m-fp32 | ≈134 | ≈31.8（集成 report） | 实测快 **4.2x** | 发射项（同上） |

偏差方向一致且远超 2x → 「发射项（`CONST-vector-launch-overhead`）在本设备严重上偏、且设计把发射项误判为绝对主导」构成 **实测与设计假设直接矛盾**，已随 `[DESIGN_LIMIT]` 反馈登记（见 `perf_feedback.md`）。

## Iteration Log

每轮固定 base = 当前已合并 current best；每个实验分支只改一个主要优化点；分支先跑 L0 再 `msprof op`；对 target workload 按 Task Duration 选 winner。所有候选文件与 raw profile 见 `perf_opt/` 与 `perf_opt/profiles/`。

**实验批处理 runner**: 首轮生成 `perf_opt/run_experiments.py`（自 skill 模板 `run_experiments_template.py` 适配），每轮的分支执行（L0 → msprof → `perf_records.jsonl` 追加）由其批量完成；诊断性 op 消融由 `perf_opt/probe_chain.py` / `probe_tanh.py` 承担（不计入 gate，不写 perf_records）。

### 现象诊断（Round 1 前置）：逐 op 代价分解

在 baseline 上做同结构消融探针（`probe_chain.py`，16m fp16，msprof op，raw `profiles/probe/{full,no_exp,no_rec,short_horner,no_sign,only_copy}`）：

| 变体 | Task Duration(us) | 归因 |
|---|---:|---|
| full（A&S 链） | 256.30 | — |
| no_exp（`vexp`→`vmul`，少 1 op） | 251.66 | `vexp` ≈ 13.8µs/16m |
| no_rec（`vrec`→`vmul`，少 1 op） | 242.60 | `vrec` ≈ 22.9µs/16m |
| short_horner（Horner 9→1，少 8 op） | 183.08 | 普通 op ≈ **9.15µs/op**（每 tile ≈73 cycle） |
| no_sign（去 `vselect`+`vmul`） | 197.16 | **`vselect` ≈ 59.1µs（≈6.5x 普通 op）** |
| only_copy（仅搬入/搬出+cast） | 76.60 | 搬运地板 |

**结论**：链上最贵单 op 是 `vselect`（bool 掩码选择）——A&S 的符号处理（`vcmp`+`vselect`+`vmul`）是**设计层可消除的结构性成本**。由此生成的候选优化点：
- **C1（主）**：换用 `erf(x) ≈ tanh(x·Q(x²))` 奇多项式（`tanh(odd)=odd`）——天然消掉 `vabs`/`vcmp`/`vselect`/`vrec`/`vexp`，op 数 21→12；
- **C2**：block_size 扫描（发射/摊薄）；
- **C3**：A&S + 算术符号（`x·vrec(|x|+eps)`）替换 `vselect`（保留 A&S 精度）；
- **C4**：静态整块 + 动态尾块 T.copy（对照 bitwise_not r3）；
- **C5**：连续核内分片（对照 bitwise_not r4）。

### Round 1 — 奇 tanh-多项式逼近 + block_size（采纳，C1+C2）

**现象**（current best = baseline）：纯 Vector 受限，21/23-op 链的 op 数与 `vselect` 代价主导（见上表）。候选 C1 的数学基础：`atanh(erf(x))` 是光滑奇函数，用 **P(x)=x·Q(x²)（Q 4 次，5 系数）** 拟合即可在 fp32 门禁内（离线 minimax + 单调性约束求解；纯奇异多项式在 4 系数时误差 2.88e-6，6 系数降到 6.2e-8 —— 系数个数是精度关键，见 `Skill Retrospective`）。

**候选 vs current best（current best = baseline，全部 msprof op Task Duration(us)）**：

| 候选 | workload | TaskDur(us) | AIV vec | MTE2 | vs best | L0 | 结论 |
|---|---|---:|---:|---:|---:|---|---|
| baseline | 16m-f16 | 256.4451 | 0.787 | 0.246 | 0.0% | pass | base |
| r1_tanh（C1，bs2048） | 16m-f16 | 167.39 | 0.942 | 0.383 | +34.7% | pass | valid |
| r1_tanh bs4096 | 16m-f16 | 143.41 | 0.961 | 0.240 | +44.1% | pass | valid |
| r1_tanh bs6144 | 16m-f16 | 137.16 | 0.969 | 0.183 | +46.5% | pass | valid |
| r1_tanh bs8192 | 16m-f16 | 131.12 | 0.972 | 0.151 | +48.9% | pass | valid |
| r1_tanh bs10240 | 16m-f16 | 124.89→117.03 | 0.972 | 0.150 | +54.4% | pass | **winner** |
| r1_short_horner（对照） | 16m-f16 | 183.08 | — | — | +28.6% | 非门禁探针 | 诊断 |

**block_size 扫描（Autotune Log，C2；`run_experiments.py --block_size`）**：

| dtype / workload | 候选 bs | TaskDur(us) | 备注 |
|---|---|---:|---|
| f16 16m | 6144 / 8192 / 10240 | 457.77→(见下) | 单调下降 |
| f16 256m | 6144 | 1990.26 | — |
| f16 256m | 8192 | 1894.50 | — |
| f16 256m | **10240** | **1822.01** | winner（11264 UB overflow） |
| f32 16m | 6144 | 119.84 | — |
| f32 16m | 8192 | 114.47 | — |
| f32 16m | 9216 | 113.30 | — |
| f32 16m | 9728 | 112.44 | 与 9216 差 <1%（噪声内），取保守 9216 |
| f32 16m | 10240 | UB overflow（`1769472 bits > 1572864 bits`） | invalid |
| bf16 16m | 8192 / 9216 / 10240 | 133.42 / 120.89 / 119.25 | 10240 best |

**winner**: C1（tanh 奇多项式）+ per-dtype `block_size` = **f32 9216 / f16 10240 / bf16 10240**，落盘 `erf_fwd_kernel_round1_tanh.py`（后续升格为 `perf_opt/erf_fwd_kernel.py`）。合并后复测本 kernel 全部 tune workload + smoke 精度：全过（见 Final Summary）。

**精度回归（C1 的代价）**：设备实测 fp32 `max_abs` 5.90e-7 → 2.98e-6（门禁 1e-5，余量 3.4x）；fp16 4.88e-4、bf16 1.95e-3 与 baseline **逐值相同**（受输出域舍入支配，1 ulp）；L0/L1/L2/Boundary 全过，±inf/NaN 与 `torch.erf` 一致。**关键修正**：设备 `vtanh` 经 `exp` 实现，参数超 fp32 exp 域即返回 NaN（x=8 实测 NaN），故链尾加 `T.vmin(P, 10)` 单 op 夹逼（`vmin` 传播 NaN，`inf/nan` 语义保持）；夹逼仅在 erf 饱和区（P≥10 首次出现在 x≈3.9）触发，误差 <1e-8。

### Round 2 — 结构性对照（C4/C5，未采纳）

**现象**（current best = r1_tanh）：kernel 已基本 Vector 饱和（vec 0.95–0.98），MTE2 仅 0.15–0.27，剩余空间很小；按 pattern-library 对照同族先例（bitwise_not perf_opt 的 static-bulk + contiguous per-core、lerp_tensor 的 contiguous chunk）逐个验证结构差异。

**候选 vs current best**：

| 候选 | workload | TaskDur(us) | AIV vec | MTE2 | vs best | L0 | 结论 |
|---|---|---:|---:|---:|---:|---|---|
| r1_tanh（best） | 16m-f16 | 117.05 | 0.972 | 0.148 | 0.0% | pass | base |
| round2_static（静态整块+动态尾块） | 16m-f16 | — | — | — | — | **FAIL** | **blocked（invalid）**：bs=10240 下双分支代码路径使 auto-multi-buffer UB 膨胀，`ub overflow, requires 1769472 bits while 1572864 available`（编译失败，不写 perf_records） |
| round2_contig（连续核内分片） | 16m-f16 | 117.03 | 0.972 | 0.151 | +0.0% | pass | tie（<3%） |
| round2_contig | 256m-f16 | 1822.30 | 0.984 | 0.209 | -0.0% | pass | tie |

**结论**：C4 blocked（UB overflow，且即便修复也非本 kernel 的主杠杆——bitwise_not 的 static-bulk 收益来自 DMA 窗口，本 kernel 是 Vector 受限）；C5 tie。r1_tanh 保持 current best。**参照结构 diff 检查（T-1 morph ladder）**：同族已知最优实现（bitwise_not r3/r4、lerp_tensor r3）的全部结构差异（static-bulk / contiguous / per-dtype block_size）已逐一实验验证，block_size 已采纳，其余两项无效——**未发现剩余可移植结构差异**。

> **记录更正说明（append-only 留痕）**：`perf_records.jsonl` 中存在一行 `round2_static / elementwise-256m-float16 / 2265.6854`（`l0_pass=true`，由 `--skip-l0` 触发，未跑回归）。该行**不是有效性主张**：该分支在 16m-f16 的 L0 回归中因 UB overflow 编译失败（见上表 FAIL 行），故其 `l0_pass=true` 不具权威性、且 2265.69µs 是编译/回退路径下的非代表性数值。按 append-only 纪律不删改该行；此注记即为其更正，winner 判定与 Final Summary 不使用该行（round2_static 未参与任何 winner 选择）。

### Round 3 — 收敛判定

r1_tanh 起，连续 2 轮（round1 的 bs 微调之外、round2 的结构对照）无 >3% 提升；Round 1 的主结构候选（tanh 族）已充分验证（含 k=4/k=6 精度-速度权衡），参照结构差异已消除。核内合并与全量复测已完成（见 Final Summary），无 `merge_blocked`。进入 Phase 3 收束。

**中止条件**: `success`（best_effort：以通过核内合并与全量非回退检查的实测最优结果作为结论；全部 tune workload 有调优结论 + 最终全量记录 + smoke 精度通过）。

## Final Summary

**最终候选**: `perf_opt/erf_fwd_kernel.py`（candidate_id `r2_merged`，sha256 `1222a6…`）= Round 1 winner（奇 tanh-多项式 + per-dtype block_size），Round 2 无改动合入。
**`phase=final` 记录**: 全部 7 个 tune workload 均在同一最终候选上重新测得（`perf_records.jsonl`，`phase=final`，同一 `candidate_id`）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/elementwise/erf/erf_kernel/erf_fwd_kernel.py::erf_fwd_kernel::main | elementwise-16m-float16 | 256.4451 | 117.0523 | r2_merged |
| examples/TileOPs/tileops/kernels/elementwise/erf/erf_kernel/erf_fwd_kernel.py::erf_fwd_kernel::main | elementwise-16m-bfloat16 | 256.7151 | 119.2524 | r2_merged |
| examples/TileOPs/tileops/kernels/elementwise/erf/erf_kernel/erf_fwd_kernel.py::erf_fwd_kernel::main | elementwise-16m-float32 | 308.0962 | 112.0822 | r2_merged |
| examples/TileOPs/tileops/kernels/elementwise/erf/erf_kernel/erf_fwd_kernel.py::erf_fwd_kernel::main | elementwise-64m-float16 | 1015.6003 | 457.7691 | r2_merged |
| examples/TileOPs/tileops/kernels/elementwise/erf/erf_kernel/erf_fwd_kernel.py::erf_fwd_kernel::main | elementwise-64m-bfloat16 | 1016.8503 | 466.6893 | r2_merged |
| examples/TileOPs/tileops/kernels/elementwise/erf/erf_kernel/erf_fwd_kernel.py::erf_fwd_kernel::main | elementwise-256m-float16 | 4058.6312 | 1822.1465 | r2_merged |
| examples/TileOPs/tileops/kernels/elementwise/erf/erf_kernel/erf_fwd_kernel.py::erf_fwd_kernel::main | elementwise-256m-bfloat16 | 4060.5111 | 1857.8271 | r2_merged |

逐 workload 提升：16m-f16 **-54.4%**、16m-bf16 **-53.5%**、16m-f32 **-63.6%**、64m-f16 **-54.9%**、64m-bf16 **-54.1%**、256m-f16 **-55.1%**、256m-bf16 **-54.2%**（几何平均 ≈ -54.5%，加速 2.15x–2.75x）。profile 证据：`profiles/final/round2/r2_merged_*/…`；最终 vec 利用率 0.962–0.985（Vector 饱和）。

### smoke 与 skipped 排除

- smoke（smoke-1m-f32/f16/bf16）：不做性能调优；最终精度回归 **pass**（`perf_opt/erf_fwd_kernel.py --level all` 的 `L0-smoke` 三项：fp32 `max_diff=2.980e-06`、fp16 `4.883e-04`、bf16 `1.953e-03`，均在门禁 1e-5 / 1e-3 / 1.6e-2 内）。附带观察：bs 调优后 smoke-1m-f16 实测 11.88µs（baseline 18.98µs）——smoke 不参与性能门禁，仅记录。
- skipped：无。

### 总体结论与遗留

- 本算子从「21/23-op A&S 链的 Vector 发射受限」改为「12/14-op 奇 tanh-多项式链的 Vector 饱和」，全部 7 个 tune workload 取得 2.15x–2.75x 加速，且 fp16/bf16 精度与 baseline 一致、fp32 余量 3.4x（仍通过门禁）。
- 遗留：fp32 精度余量由 16.9x 降至 3.4x（确定性误差上界 2.98e-6）。若后续要求 ≥10x 余量，可切 `k=6` 变体（16 op，fp32 余量 >160x，仍为 baseline 的 ~2.0x）；该权衡已实测记录，本任务按 best_effort 取速度更优的 `k=4`。
- `[DESIGN_LIMIT]` 已产出：`perf_feedback.md`（设计 §1.6.0 对 tanh 型候选 7 的淘汰依据仅在其 2 系数形态下成立；建议路由 = 附录补记）。

## Skill Retrospective

### 流程问题

1. **诊断性探针缺少独立的输出目录约定，导致 msprof 结果被同名目录跨 run 污染**：首轮 smoke-1m 曾读到 68.5µs（真值 11.9µs），根因是复用同一 `profiles/probe/tanh_4_bs10240_float16` 目录（同目录内并存 16m 与 1M 两批 `OpBasicInfo.csv`）。`run_experiments.py` 的 `safe_component(opt_id)+safe_component(workload_id)` 目录命名是正确的防呆，但**手写探针命令绕过了它**。建议：把「探针也必须用含 workload 哈希的唯一目录」写入 profile-collection.md §3 的日志要求，或提供 `--tag` 强制唯一化。
2. **`for j in range(...)` 在 `@T.prim_func` 内被 TVM script 解析器当作 `T.serial`**（报 `serial() takes from 1 to 2 positional arguments but 3 were given`）：编译期多项式 unrolling 必须**手写展开**（或用编译期 Python `if k >= j:` 逐条展开，本任务采用后者）。建议在 traps-compiler.md 的 `TRAP-tvm-parser-rules` 条目补一行：prim_func 内的 Python `for range` 会被强制转 TIR 循环，**编译期展开须手写**。
3. **`T.fill(buf, python_float)` 非法**（`value must be a tir.PrimExpr or tir.BufferLoad`），须 `T.cast(v, dtype)`；而 `T.vadd(buf, python_float, buf)` / `T.vmul(buf, python_float, buf)` 的标量形式合法——同一文件内两种标量用法不一致，易踩。

### 建议修改（BP proposal）

- `BP_erf_tanh_odd_poly`（P 类，vp_type=P）：**「同一函数族的淘汰依据须做系数预算扫描」**——设计/调优淘汰一个逼近函数族前，必须给出该族在给定系数预算下的**最小可达误差**，不能以单一低阶形态的误差定论；erf 实证：`tanh(αx+βx³)`（2 系数）误差 ~1e-3 → `tanh(x·Q(x²))`（5 系数）2.9e-6、（7 系数）6.2e-8（3 个数量级差异）。已回写 pattern-library（见下）。
- `BP_vselect-cost`（D 类，vp_type=D）：910_9362 上 `vselect`（bool 掩码选择）单 op 代价 ≈6.5x 普通 v-op（16m fp16 实测 59.1µs vs 9.15µs），是「用掩码重构奇/偶函数」类算子的首选消除对象。

### 价值点（按 vp_type，含证据三件套）

| vp_type | 条目 | evidence（实测） | repro | toolchain_stamp |
|---|---|---|---|---|
| D | `vselect` 单 op 代价 ≈6.5x 普通 op；普通 op ≈9.15µs/16m-f16；`vrec`≈22.9µs；`vexp`≈13.8µs | `probe_chain.py` 消融，`profiles/probe/{full,no_exp,no_rec,short_horner,no_sign,only_copy}`（256.30/251.66/242.60/183.08/197.16/76.60µs） | repro/PL-1.29-vselect-cost.py（本任务回写，跑通） | tilelang 0.1.2+c3a94f1d / CANN 9.0.0 / Ascend910_9362 |
| D | 设备 `vtanh` 参数 > fp32 exp 域即 NaN（`tanh(inf)=nan`），须前置夹逼；`vmin` 传播 NaN | `_check_acc.py --impl probe_tanh.py --degree 4`（无夹逼：`x=8`、`x=inf` → NaN；加 `vmin(P,10)` 后 `inf/nan` 逐值等于 torch.erf） | repro/TRAP-vtanh-large-arg-nan.py（本任务回写，跑通） | 同上 |
| P | 奇 tanh-多项式逼近：`erf(x)≈tanh(x·Q(x²))`，`tanh(odd)=odd` 消除 sign/abs/`vselect`；5 系数达 fp32 门禁 3.4x 余量，7 系数 >160x | 12/14-op 链 vs A&S 21/23-op：16m-f16 117.05 vs 256.45µs（2.19x）；fp32 `max_abs` 2.98e-6（k=4）/6.2e-8（k=6） | repro/PL-1.30-tanh-odd-poly-erf.py（本任务回写，跑通） | 同上 |
| C | 案例：erf Stage 4「A&S → 奇 tanh-多项式」2.2x 调优档 | `perf_opt/opt_log.md` + `perf_records.jsonl`（7/7 tune workload）+ `perf_feedback.md` | — | 同上 |

回写位置：`pattern-library/elementwise.md`（PL-1.29 / PL-1.30）、`pattern-library/traps-runtime.md`（TRAP-vtanh-large-arg-nan）、`pattern-library/constants.md`（单 op 代价表），repro 落 `pattern-library/repro/`。

## 工件清单

- `perf_opt/erf_fwd_kernel.py`（final，drop-in；导出 `TUNED_BLOCK_SIZE = {"float32": 9216, "float16": 10240, "bfloat16": 10240}` 供 wrapper 切换块成对引用）
- `perf_opt/erf_fwd_kernel_round1_tanh.py`（Round 1 winner 分支）、`erf_fwd_kernel_round2_static.py`（blocked）、`erf_fwd_kernel_round2_contig.py`（tie）
- `perf_opt/probe_chain.py`、`perf_opt/probe_tanh.py`（诊断探针）、`perf_opt/_bench_kernel.py`、`perf_opt/_check_acc.py`、`perf_opt/run_experiments.py`（runner）
- `perf_opt/opt_log.md`、`perf_opt/perf_records.jsonl`、`perf_opt/workload_inventory.json`、`perf_opt/progress_report.md`
- `perf_opt/perf_feedback.md`（`[DESIGN_LIMIT]`，建议路由 = 附录补记）
- `perf_opt/profiles/`（raw msprof）、`perf_opt/logs/`（stdout/stderr）

---

## Ratio 对照（框架 roofline 口径，baseline vs final）

本文档上方的时延对照表用 `msprof op Task Duration`；本节用**框架报告 `report.html` 的 `Ratio(%)` 口径**，两者互补（时延是原始测量，Ratio 是框架的 roofline 评分）。

口径：`Ratio(%) = 100 × Perf ÷ min(AI × BW上限, Computility上限)`，其中 `AI = total_fops / total_bytes`（搬运 1 字节可做多少次运算）、`Perf = msprof 计数 FLOPs ÷ Task Duration`。三列 `AI / Perf / Computility` 一并列出以便逐行复核（本节全部行满足 `Ratio = 100 × Perf ÷ Computility`，即对本算子 `Computility` 是 `min()` 的取值项）。

数据来源（同一报告流水线、同一 bench 入口 `bench_unary_elementwise.py::test_erf_bench`）：

- baseline 表：`examples/TileOPs/reports/tileops/20261012_222359_253405_ErfFwdOp`（Stage 3 基准，A&S 7.1.26 链，21-op fp32 / 23-op fp16·bf16）
- final 表：`examples/TileOPs/reports/tileops/20261014_182649_897387_ErfFwdOp`（Stage 4 调优产物，奇 tanh-多项式链，12-op fp32 / 14-op fp16·bf16）

> **Ratio 口径（2026-10-15 订正）**：分母取 **GM 带宽上限**（B4 Roofline 模型理论 GM 带宽 **0.8 TB/s**，msprof 自带）与 `Computility` 的 **较小者**，即 `Ratio(%) = 100 × Perf ÷ min(AI × 0.8, Computility)`。**下表 `Ratio (%)` 列即按此口径重算**；`report 原值` 列保留框架 report 输出的数字以追溯口径差异——本设备上该路径已退化为 profiler 内置模型值（对本算子逐行等于 `Perf ÷ Computility`），**两者在 BW 轴取值的行上不一致，验收须以本表重算列为准**。

### 表 1 — 优化前（baseline）

| Label | DType | Shape | Latency (us) | **Ratio (%)** | AI (Ops/Byte) | Perf (TOps/s) | Computility (TOps/s) | report 原值 | 取值项 |
|---|---|---|---:|---:|---:|---:|---:|---:|:--:|
| smoke-1m | float32 | [1048576] | 31.84 | **19.25** | 5.961 | 0.7853 | 4.0795 | 19.25 | COMP |
| smoke-1m | float16 | [1048576] | 18.64 | **33.99** | 12.381 | 1.3930 | 4.0978 | 33.99 | COMP |
| smoke-1m | bfloat16 | [1048576] | 18.60 | **35.40** | 12.881 | 1.4523 | 4.1025 | 35.40 | COMP |
| elementwise-16m | float16 | [4096,4096] | 257.27 | **39.05** | 12.324 | 1.6074 | 4.1158 | 39.05 | COMP |
| elementwise-16m | bfloat16 | [4096,4096] | 256.95 | **40.65** | 12.824 | 1.6747 | 4.1199 | 40.65 | COMP |
| elementwise-16m | float32 | [4096,4096] | 465.33 | **20.97** | 5.515 | 0.8576 | 4.0890 | 20.97 | COMP |
| elementwise-64m | float16 | [8192,8192] | 1015.48 | **39.56** | 9.107 | 1.6285 | 4.1167 | 39.56 | COMP |
| elementwise-64m | bfloat16 | [8192,8192] | 1017.16 | **41.06** | 9.473 | 1.6918 | 4.1207 | 41.06 | COMP |
| elementwise-256m | float16 | [16384,16384] | 4074.42 | **39.43** | 6.714 | 1.6234 | 4.1169 | 39.43 | COMP |
| elementwise-256m | bfloat16 | [16384,16384] | 4066.30 | **41.08** | 6.985 | 1.6927 | 4.1210 | 41.08 | COMP |

（baseline 全部行均落在计算轴，故与 report 原值一致。）

### 表 2 — 优化后（final）

| Label | DType | Shape | Latency (us) | **Ratio (%)** | AI (Ops/Byte) | Perf (TOps/s) | Computility (TOps/s) | report 原值 | 取值项 |
|---|---|---|---:|---:|---:|---:|---:|---:|:--:|
| smoke-1m | float32 | [1048576] | 10.04 | **52.22** | 4.514 | 1.8860 | 4.6021 | 40.98 | **BW** |
| smoke-1m | float16 | [1048576] | 10.72 | **40.66** | 9.566 | 1.8715 | 4.6029 | 40.66 | COMP |
| smoke-1m | bfloat16 | [1048576] | 10.84 | **42.32** | 10.069 | 1.9480 | 4.6031 | 42.32 | COMP |
| elementwise-16m | float16 | [4096,4096] | 107.44 | **64.54** | 9.513 | 2.9710 | 4.6034 | 64.54 | COMP |
| elementwise-16m | bfloat16 | [4096,4096] | 109.44 | **66.69** | 10.013 | 3.0700 | 4.6037 | 66.69 | COMP |
| elementwise-16m | float32 | [4096,4096] | 102.02 | **82.23** | 4.507 | 2.9645 | 4.6026 | 64.41 | **BW** |
| elementwise-64m | float16 | [8192,8192] | 419.90 | **66.03** | 7.647 | 3.0398 | 4.6035 | 66.03 | COMP |
| elementwise-64m | bfloat16 | [8192,8192] | 427.98 | **68.19** | 7.946 | 3.1392 | 4.6037 | 68.19 | COMP |
| elementwise-256m | float16 | [16384,16384] | 1670.46 | **72.82** | 5.247 | 3.0563 | 4.6035 | 66.39 | **BW** |
| elementwise-256m | bfloat16 | [16384,16384] | 1703.32 | **71.27** | 5.534 | 3.1549 | 4.6037 | 68.53 | **BW** |

**7/7 tune workload 全部 ≥60%**（64.54 / 66.69 / 82.23 / 66.03 / 68.19 / 72.82 / 71.27）；3 个 smoke-1m 仅作正确性，不参与性能判定。

### 读表注意（4 条）

1. **两表分母不同，跨表不可直接相除**：`Computility` 由 ≈4.09–4.12（baseline 表）变为 ≈4.60（final 表），而 `Ratio ∝ 1/Computility`。分母无关的可比量是 `Perf`：final/baseline = **1.83–1.88x**（16m/64m/256m）、**3.46x**（16m fp32）、**1.34x**（smoke-1m，未做专门调优）。
2. **性能门槛的适用范围（60%）**：7 个 `tune` workload 全部达标（**64.41–68.53%**，最小余量 16m fp32 的 +4.41pt ≈ 相对 6.8%）；3 个 `smoke-1m` 按 Phase 1 的 smoke token 排除出性能调优，**仅作正确性回归**，其 Ratio（40.66–42.32%）不参与性能判定。注：本文档 header 的 `性能目标: best_effort` 为调优期口径，迁移验收另有本条的 60% 门槛。
3. **不得与 declared-traffic 口径混用**：`achieved_bw_source=declared_traffic/latency` 的 run（如 `20261013_230643_120627_ErfFwdOp`）会**覆写同一个 `Ratio(%)` 键**，其数值语义为 `achieved_BW ÷ 1.64`（实测可逐行复现：0.7899→48.16、0.5733→34.96、1.1954→72.89），与本节的 roofline 口径不可比。
4. **分母漂移风险（待收敛）**：`Computility` 在 2026-10-12 / 10-13 / 10-14 三批 run 中为 **4.08 / 4.22 / 4.60**（跨度 +12.7%）。最小余量用例 16m fp32 的分母再涨 6.8% 即跌破 60%。**验收须在同一分母下重采**；本条同时是「10-14 15:00 起全算子退化到 profiler fallback」这一口径回归的量化证据。
