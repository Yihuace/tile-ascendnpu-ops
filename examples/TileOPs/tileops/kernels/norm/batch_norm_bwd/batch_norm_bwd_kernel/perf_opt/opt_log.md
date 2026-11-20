# BatchNormBwdOp Stage 4 性能调优日志（opt_log）

- 算子：`BatchNormBwdOp`（batch norm 反向：grad_input / grad_weight / grad_bias）
- 目录：`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd_kernel/`
- benchmark：`examples/TileOPs/benchmarks/ops/bench_batch_norm_bwd.py::test_batch_norm_bwd_bench`
- 环境：Ascend910B3（`get_aicore_num()=20`，即 20 AIC / 40 AIV）、CANN 9.0.0、tilelang 0.1.2（dev-root）、`msprof op` 唯一时延口径
- 调度模式：`full`（首次 Stage 4，目录下无历史 perf_opt）

## 结论速览

- R2（首轮 Stage 4）唯一有效优化点：**subid 分片（PL-1.13-aiv-dup-subid-split）**——修正 `T.Kernel(n, is_npu=True) + T.Scope("Vector")` 的 Vector 程序在每 block 双 AIV 上重复执行（Mix Block Dim = 2×Block Dim）的产能浪费。
- R3（结构级二轮深挖）三个有效优化点：① wide 的 **x_hat 消去 + 逐通道常数代数折叠**（`gw = rstd·Σ(g·x) − (mean·rstd)·Σg`，`grad_x = A·g + B·x + C`）；② wide/split 的 **per-input staging + loads-first 解耦**（单 staging 复用链使 MTE2 与 VEC 串行）；③ split apply 的 **epilogue 折叠**（`scale·g + c3 − center·x`，每 tile vbrc 5→3）。
- R3 同会话配对（vs R2 best）：**−12.3% ~ −19.8%** Task Duration，全部 6 个 tune workload 都为净收益（>3% 噪声阈）。
- 全链路（Stage-3 baseline → R3 final）：**−38.6% ~ −56.6%**（逐 workload 见 Final Performance Test Data）。
- 精度：wide/split/general 三 kernel L0/L1/L2/boundary 全 PASS，smoke 三 dtype 全 PASS，无回归。
- SOL（算力受限 60% 目标）：11.4% ~ 33.3%，**未达标**（soft 目标，缺口归因见 §6）。


---

## 1. Workload Inventory 与 kernel 分派

从 benchmark 展开（`bench_batch_norm_bwd.py::test_batch_norm_bwd_bench`）共 8 案例（smoke-fc 3 dtype + resnet-stage1/2/3 + large-spatial）。经 `select_bwd_kernel`（`batch_norm_bwd.py` 的 `_holder`/`_block_launch` 纯 Python 逻辑）逐案例核实实际命中的 Stage-3 kernel：

| workload_id | shape (N,C,S) | dtype | kind | kernel_id |
|---|---|---|---|---|
| smoke-fc-{fp32,fp16,bf16} | 32,64,1 | 3 dtype | smoke | `_batch_norm_bwd_wide_kernel`（thread→wide） |
| resnet-stage1-{float16,bfloat16} | 8,64,1024 | fp16/bf16 | tune | `_batch_norm_bwd_wide_kernel`（block→wide，held） |
| resnet-stage2-float16 | 4,128,1024 | fp16 | tune | `_batch_norm_bwd_wide_kernel`（block→wide，held） |
| resnet-stage3-{float16,float32} | 4,256,784 | fp16/fp32 | tune | `_batch_norm_bwd_wide_kernel`（block→wide，held） |
| large-spatial-float16 | 4,128,262144 | fp16 | tune | `_batch_norm_bwd_split_kernel`（split→split，splits=4） |

**关键发现**：`_batch_norm_bwd_kernel`（通用 held 单扫描 kernel）**未被任何 benchmark workload 命中**（所有案例经 `_block_launch` 命中 wide，或经 `c<1024 且 L>=65536` 命中 split）→ 无 tune workload，不产生性能记录（`no_tunable_workload`），但仍产出 subid-split 版本（L0 通过）以保证 drop-in 完整。 该 kernel 在 workload_inventory.json 中按 gate 期望分类为 `skipped`（marks=`skip`，非 benchmark 路径：所有 manifest 案例经 `select_bwd_kernel` 路由到 wide/split，通用 tiled kernel 不在 benchmark 选择范围内）。

调优排序：large-spatial（5365µs，绝对负担最大）与 resnet-stage3（C=256，负担次大）优先；其余 held 小 shape 随后。全部 6 个 tune workload 均有完整 baseline/candidate/final 记录。

---

## 2. Performance Test Data（baseline，round 0）

| kernel_id | workload_id | candidate_id | target_kernel | task_duration_us | SOL ratio(%) | AI | profile_status |
|---|---|---|---|---:|---:|---:|---|
| _batch_norm_bwd_wide_kernel | resnet-stage1-float16 | baseline_wide | main_mix_aic | 21.080 | 17.995 | 7.00 | valid |
| _batch_norm_bwd_wide_kernel | resnet-stage1-bfloat16 | baseline_wide | main_mix_aic | 20.400 | 19.698 | 7.50 | valid |
| _batch_norm_bwd_wide_kernel | resnet-stage2-float16 | baseline_wide | main_mix_aic | 25.351 | 16.117 | 7.25 | valid |
| _batch_norm_bwd_wide_kernel | resnet-stage3-float16 | baseline_wide | main_mix_aic | 42.241 | 15.281 | 9.36 | valid |
| _batch_norm_bwd_wide_kernel | resnet-stage3-float32 | baseline_wide | main_mix_aic | 43.181 | 15.866 | 3.96 | valid |
| _batch_norm_bwd_split_kernel | large-spatial-float16 | baseline_split | stats/finalize/apply_mix_aic | 5365.027 | 28.164 | 3.85 | valid |

smoke（smoke-fc 三 dtype）：仅精度回归，不采性能；skip：`_batch_norm_bwd_kernel`（无命中 workload）。

### 设计估算 vs 实测偏差（D-2 回填）

DESIGN.md §1.6.0 设计期估算（引用 CONST-mte2-degradation / CONST-vector-launch-overhead，2026-09-07 工具链，已被 kb_stale_check 标 stale）：resnet-stage1 ≈ 20µs（发射项 bound）、large-spatial ≈ 4.26ms（流量项）。实测（CANN 9.0）：resnet-stage1 21.08µs（估算偏差 <10%，基本吻合）、large-spatial 5365µs（比 4.26ms 估算高 +26%——失准项为 split 三段结构的 apply 重读流量与发射，非 pure 流量口径）。结构性结论（held 3-pass vs tiled/split 5-pass）不依赖精确常数，仍成立。

---

## 3. Iteration 1 Diagnosis（唯一有效轮）

### Diagnostic Context

- kernel_id: `_batch_norm_bwd_wide_kernel` / `_batch_norm_bwd_split_kernel`
- current_best: baseline（Stage-3 三 kernel）

#### Profile Facts（baseline，resnet-stage1-float16，msprof op）

- Task Duration: 21.08µs
- Block Dim / Mix Block Dim: **40 / 80**（Mix = 2×Block Dim）
- ArithmeticUtilization: `cube0` 全 0（无 Cube）；`vector0` 与 `vector1` 子块的 `aiv_vec_ratio` **完全相同（0.230873）**、`aiv_vec_fops` 相同（229376）
- PipeUtilization: `aiv_vec_ratio=0.231`、`aiv_scalar_ratio≈0.19`、`aiv_mte2_ratio≈0.08`、`aiv_mte3_ratio≈0.05`、`aiv_time≈10.5µs`（占 21µs 的 50%）
- roofline `GM Read+Write` ratio=0.183（18.0%），computility=`Vec_FP32(65%) + Vec_MISC(35%)`

#### 现象

- P1（AIV 重复执行）：`vector0`/`vector1` 子块指标完全相同而非各半——命中 pattern-library **PL-1.13-aiv-dup-subid-split**（`T.Kernel(n)+T.Scope("Vector")` 程序默认在每 block 双 AIV 重复执行，Mix Block Dim = 2×Block Dim）。本设计 §5.5 以 `num_kernels = min(C, 40)`（"AI cores ×2"）试图用满 40 向量核，但实际效果是 40 block × 双 AIV 冗余 = 产能 2x 浪费（而非 40 个独立向量核）。
- P2（向量利用率低）：aiv_vec_ratio 仅 23%，50% 时间 AIV idle（双 wave 调度 + 重复执行）。
- P3（Vec_MISC 占 35%）：vbrc 广播 + vcast 中转占向量工作的 35%（fp32 中转 + 逐通道标量广播的固有代价）。

#### 候选优化点

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 |
|---|---|---|---|---|---|
| OP1 | P1 AIV 重复执行 | subid 分片（PL-1.13） | PL-1.13 判据（两 AIV 子块指标相同）+ 本 profile 实测 | `num_kernels=min(C, aicore_num)` + `c = i*(num_kernels*2) + cid*2 + subid` | Task Duration / Mix Block Dim / L0 |

### Experiment Branches（从 baseline 派生，单点改动）

| branch | base | opt_id | major_change | correctness | task_duration_us (resnet-stage1-fp16) | result |
|---|---|---|---|---:|---:|---|
| _batch_norm_bwd_wide_kernel_round1_op1.py | baseline | OP1 | 分核 40→20 + subid 通道分片 | L0 PASS | 13.260 | improved (−37.1%) |
| _batch_norm_bwd_split_kernel_round1_op1.py | baseline | OP1 | 三段（stats/finalize/apply）均 subid 分片 | L0 PASS | 2905.108 (large-spatial) | improved (−45.9%) |

### 候选 vs current best 对比表（B2）

wide kernel（Task Duration(us) / SOL ratio(%) / L0）：

| workload_id | baseline_us | candidate_us | baseline_ratio% | candidate_ratio% | L0 | 提升 |
|---|---|---:|---:|---:|---:|---:|---:|
| resnet-stage1-float16 | 21.080 | 13.260 | 17.995 | 14.345 | PASS | −37.1% |
| resnet-stage1-bfloat16 | 20.400 | 14.650 | 19.698 | 14.043 | PASS | −28.2% |
| resnet-stage2-float16 | 25.351 | 16.300 | 16.117 | 12.715 | PASS | −35.7% |
| resnet-stage3-float16 | 42.241 | 23.420 | 15.281 | 14.033 | PASS | −44.6% |
| resnet-stage3-float32 | 43.181 | 24.180 | 15.866 | 14.086 | PASS | −44.0% |

split kernel：

| workload_id | baseline_us | candidate_us | baseline_ratio% | candidate_ratio% | L0 | 提升 |
|---|---|---:|---:|---:|---:|---:|---:|
| large-spatial-float16 | 5365.027 | 2905.108 | 28.164 | 25.394 | PASS | −45.9% |

> 注：subid 分片后 `ratio_pct` 下降（17.99→14.35）是**预期且正确**的——msprof roofline 的 `performance = counted_flops / duration`，重复执行期 counted_flops 被双计（两 AIV 各算一遍），分片后仅计一次，故 ratio 减半而 Task Duration 减半；AI 同理由 7.0→3.5。该 metric 在重复执行下被高估，不用于跨 subid 前后对比，仅 Task Duration 是唯一裁决口径。

### Iteration Winner

- winner: `_batch_norm_bwd_wide_kernel_round1_op1.py`（wide）、`_batch_norm_bwd_split_kernel_round1_op1.py`（split）
- reason: 全部 tune workload 的 Task Duration 显著下降（−28~−46%），L0 全 PASS，无回退。
- new_current_best: 两 subid-split 候选（合并入 `perf_opt/{wide,split}.py`）

### 后续轮次判断（plateau 判定依据）

Round 1 之后，剩余现象为：wide `aiv_scalar_ratio 23~33%`（persistent 循环 + strided copy 地址 + reduce）、`Vec_MISC 35%`（vbrc/vcast，fp32 中转 + 逐通道标量广播固有）、split apply 4 vbrc。逐一评估：

- 消除 fp32 中转 → **不可行**（正确性硬约束：bf16 无 Vector 算术（docs 列为 ×）、fp16 须对齐 fp32 golden；RETROSPECTIVE §1.6.1 已机器验证）。
- 消除 vbrc 广播 → **不可行**（逐通道标量广播是算法固有；PL-1.15 已证隐式列广播 +12~16% 税，vbrc 是更优形态）。
- 降低 scalar 开销 → 无独立结构性抓手（地址计算/reduce 与"一 AIV 一通道"结构绑定），无 >噪声阈值的可验证收益路径。
- 归约走 Cube（ones-vector GEMM）→ 形态为 M=1/K=L/N=1 的极瘦 GEMM，Cube 无优势且需 GM→L1→L0C 往返，无实测先例支撑 >2x，判为未实证假设，不启动。

即：主要结构候选（subid 分片）已充分验证并合并；其余方向为正确性硬约束或无结构性抓手 → 符合 plateau（连续无 >3% 提升的结构候选）+ 主要方向已验证的收束条件。不触发 `[DESIGN_LIMIT]`（见 §6）。

---

## 4. R3 深挖（结构级二轮，2026-10-20）

> 起点：R2 final（`perf_opt/BatchNormBwdOp.py`，subid 分片版）。目标：current best 再降 10–20% Task Duration。口径：`msprof op` Task Duration(us)（`launch-count=20`，取 20 次 median）。

### 4.0 起点复核（先重测，不沿用记忆值）

| 候选 | resnet-stage1-fp16 | stage1-bf16 | stage2-fp16 | stage3-fp16 | stage3-fp32 | large-spatial |
|---|---:|---:|---:|---:|---:|---:|
| Stage-3 baseline（重测 round 0） | 21.190 | 20.500 | 25.300 | 41.040 | 43.180 | 5373.030 |
| R2 final = R3 current best（重测） | 13.270 | 13.800 | 16.190 | 23.330 | 24.820 | 2906.750 |

两次重测与 R2 记录（21.08/20.40/25.35/42.24/43.18/5365.03 与 13.46/13.64/15.81/23.53/24.34/2903.33）偏差 ≤3%，起点可信。**同时证伪了调度 prompt 里的一个假设**：`_plan_bS(8, 1024, 22)` 实际给出 `bS = min(S, align_down(184320//(8·22), 8)) = min(1024, 1040) = 1024`，即 resnet-stage1 早已走 **single-scan（held）单扫描路径**，不存在「bS=928 → 2 tile → 5 遍流量」；该候选方向（降 pe 到 ≤20 以挤进 single-scan）不成立，未做无谓实验。

### 4.1 R3 Diagnosis（profile 事实）

wide（resnet-stage1-fp16，current best）：Task 13.27µs；`aiv_time=9.98`（占 75%）；`vec_time=4.85`（ratio 0.373）、`mte2_time=1.9`、`mte3=0.8`、`scalar_ratio≈0.27`；**Σpipe ≈ aiv_time**（`sc_wait≈6.0` 为纯等待）→ 单 staging 复用链把 MTE2→VEC 串成一条链（PL-1.10 形态）。每元素向量 op ≈ 17（2 vcast + 5 vbrc + 7 arith + 2 reduce + 1 vcast）；`aiv_time` 之外的 3.3µs 为发射/尾波开销（26%）。

split（large-spatial-fp16，current best）：stats 1285 / finalize 10.0 / apply 1612µs。stats `vec 741 + mte2 394 + scalar 417 ≈ 1552 > aiv 1274`（约 22% 重叠）；apply `vec 898 + mte2 496 + mte3 161 ≈ 1555 ≈ aiv 1601`（**零重叠**），每元素 12 个向量 op（2 vcast + 5 vbrc + 4 arith + 1 vcast）。有效带宽：apply 读 536MB/496µs = 1.08 TB/s，stats 1.36 TB/s——均低于设备混合流地板（≈1.7 TB/s），故 apply 是「串行」而非「带宽饱和」。

按序评估后**未启动**的方向与理由：wave 量化（wide 每 block 的 2 个 subid 分属不同通道，跨 AIV 协作归约需第二 kernel/跨核合并，5-pass 反噬 > 均衡收益 20%→100% 仅在 C≪40 时成立，本算子 C=64/128/256 已达 80%/80%/91%）；归约改走 Cube（上一轮已否决，仍无实测先例）；消除 fp32 中转（bf16 无 Vector 算术 + fp16 对齐 fp32 golden，正确性硬约束）。

### 4.2 实验分支（每分支只改一个主要优化点，均从同一 current best 派生）

| branch | base | 主要改动（opt_id） | L0 | 目标 workload |
|---|---|---|---|---|
| `_batch_norm_bwd_wide_kernel_r3_w1.py` | R2 wide | W1：消去逐元素 `x_hat`，`gw = rstd·Σ(g·x) − (mean·rstd)·Σg`，`grad_x = A·g + B·x + C`（17→13 op/elem） | PASS | 5 个 wide workload |
| `_batch_norm_bwd_split_kernel_r3_s1.py` | R2 split | S1：apply 段 per-input staging（loads-first，MTE2 与 VEC 解耦） | PASS | large-spatial |
| `_batch_norm_bwd_wide_kernel_r3_w2.py` | W1 | W2：wide per-input staging + `brc` 由 `prod` 复用（保持 UB 不变，single-scan 不退化） | PASS | 5 个 wide workload |
| `_batch_norm_bwd_split_kernel_r3_s2.py` | S1 | S2：apply epilogue 折叠（`scale·g + c3 − center·x`，每 tile 5→3 vbrc）+ apply 独立 tile 宽度 | PASS | large-spatial |
| `_batch_norm_bwd_split_kernel_r3_s3.py` | S2 | S3：stats 段 per-input staging + `brc` 由 `prod` 复用 | PASS | large-spatial |
| `_batch_norm_bwd_split_kernel_r3_s4.py` | S2 | S4：apply tile 循环 `T.Pipelined(num_stages=2)` | PASS | large-spatial |

### 4.3 候选 vs current best 对比表（B2，Task Duration(us) / roofline ratio(%) / L0）

wide（同一 base = R2 wide；batch A/B 分别为两次会话，故同时给出同会话比值）：

| workload_id | R2 best_us | W1_us | W2_us | ratio%（R2→W2） | L0 |
|---|---:|---:|---:|---:|---|
| resnet-stage1-float16 | 13.270 | 12.010 | 12.120 | 14.63→12.11 | PASS |
| resnet-stage1-bfloat16 | 13.800 | 13.900 | 12.400 | 14.71→13.13 | PASS |
| resnet-stage2-float16 | 16.190 | 15.430 | 14.610 | 12.48→10.51 | PASS |
| resnet-stage3-float16 | 23.330 | 22.490 | 19.170 | 13.84→12.19 | PASS |
| resnet-stage3-float32 | 24.820 | 22.530 | 20.280 | 13.46→16.37 | PASS |

> W1 已在 4/5 workload 上有效（−3.6%~−9.5%），stage1-bf16 落回噪声（+0.7%）；W2 在同会话再降，且把 bf16 拉回 −10%；AICore 侧 `aiv_time` 从 9.98→8.07µs、`vec_time` 4.85→3.77µs。roofline ratio 的绝对下降含 subid 口径效应（见 §3 注）+ op 数下降，跨轮只以 Task Duration 裁决。

split（同一 base = R2 split）：

| workload_id | R2 best_us | S1_us | S2_us | S3_us | S4_us（弃） | L0 |
|---|---:|---:|---:|---:|---:|---|
| large-spatial-float16 | 2906.750 | 2715.840 | 2528.010 | **2333.080** | 2528.350 | PASS |

> S1（apply loads-first）−6.6%；S2（apply epilogue 折叠 + apply 独立 tile 宽）再 −6.9%（同时把 S1 因统一 `pe+2` 造成的 stats +3.3% 回退修掉，stats 回到 1284µs）；S3（stats loads-first）再 −7.7%（stats 1284→1091µs）；**S4（apply `T.Pipelined(2)`）为 tie（+0.01%，远小于噪声阈）→ 判 config_no_gain，不采纳**。stage 归因（S3）：stats 1091（vec 731 / mte2 271 / scalar 0.370）、finalize 10.2、apply 1233（vec 724 / mte2 314 / mte3 179）。

### 4.3b AICore / memory 指标对比（msprof PipeUtilization，vector0 子块 median）

| workload / stage | 候选 | Task(µs) | aiv_time | vec_time | mte2_time | mte3_time | aiv_scalar_ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| resnet-stage1-fp16（wide） | R2 best | 13.270 | 9.977 | 4.848 | 2.264 | 0.810 | 0.258 |
| resnet-stage1-fp16（wide） | **W2** | 12.120 | 8.073 | 3.768 | 2.049 | 0.815 | 0.233 |
| large-spatial stats | R2 best | 1285.090 | 1273.985 | 740.849 | 394.132 | 1.767 | 0.327 |
| large-spatial stats | **S3** | 1091.360 | 1083.231 | 730.878 | 271.439 | 1.890 | 0.370 |
| large-spatial apply | R2 best | 1611.630 | 1601.601 | 898.394 | 496.398 | 161.332 | 0.056 |
| large-spatial apply | **S3** | 1232.980 | 1226.933 | 723.824 | 313.931 | 178.990 | 0.051 |

> 读法：wide 的变化来自 vec_time（op 数 17→13）与 mte2 不再与 VEC 串行；split 的两段 MTE2 时间各降 31%/37%（loads-first），apply 的 vec_time 同时降 19%（epilogue 折叠）。`aiv_time` 与 Task Duration 的差额（wide ≈3.5µs、split ≤7µs）是发射/尾段开销。

### 4.4 合并非回退检查与 winner

- wide winner = **W2**：合入 `perf_opt/_batch_norm_bwd_wide_kernel.py`；本 kernel 5 个 tune workload 全部不回退（同会话配对见 4.5）。
- split winner = **S3**（S4 弃）：合入 `perf_opt/_batch_norm_bwd_split_kernel.py`；large-spatial 不回退。
- 三工厂合并入 `perf_opt/BatchNormBwdOp.py`（split 侧沿用 host-only 更名 `_plan_tiling_split` / `_UB_SAFE_BYTES_SPLIT`，编译产物不变）；general kernel 无 benchmark 命中，保持 R2 subid 版（L0/L1/L2 全 PASS）。
- smoke 精度（smoke-fc 32×64×1，三 dtype）：`BatchNormBwdOp.py --level all` 中 L0-1/2/3 全 PASS（max_abs_err ≤ 1.9e-6），L1(14 grid+契约)/L2(4 例)/boundary 全 PASS。

### 4.5 逐 workload Final Summary（R3 final vs R2 best，**同 NPU 会话配对**）

> 会话内状态漂移在 wide 小 shape 上可达 ~7%（同一二进制两次独立运行 12.12 / 13.02µs），故下表用 `run_r3_ab.py` 在同一次 NPU 会话内交替测 R2 best 与 R3 final，比值状态干净；绝对时延取 final 侧记录（= `perf_records.jsonl` 的 phase=final 行）。

| workload_id | R2 best_us | R3 final_us | Δ% | 判定 |
|---|---:|---:|---:|---|
| resnet-stage1-float16 | 14.360 | 12.600 | −12.3% | winner_merged |
| resnet-stage1-bfloat16 | 13.780 | 11.480 | −16.7% | winner_merged |
| resnet-stage2-float16 | 16.840 | 14.320 | −15.0% | winner_merged |
| resnet-stage3-float16 | 23.630 | 20.690 | −12.4% | winner_merged |
| resnet-stage3-float32 | 24.130 | 20.880 | −13.5% | winner_merged |
| large-spatial-float16 | 2911.340 | 2333.520 | −19.8% | winner_merged |

目标（−10~−20%）达成；stop_reason = success。

### 4.6 R3 复盘与天花板证据

- **已用尽的结构杠杆**：wide 17→13 op/elem 且 `aiv_time ≈ Σpipe`（无重叠余量）；split stats 9 op/elem（fp32 中转 2 vcast + x_hat 4 op 不可消——`finalize` 签名不含 `mean` 且 wrapper 只读，无法把 `mean` 折叠到标量段）、apply 10 op/elem。
- **apply 的重叠仍未取得**：S3 后 apply `aiv 1227 ≈ vec 724 + mte2 314 + mte3 179`（零重叠）。`T.Pipelined(2)`（S4）无收益；把 3 个 vbrc 提到 tile 循环外需 3×(2,3400) fp32 = 81.6KB 常驻，超 UB 余量（当前 nominal 136KB/192KB，含 multi-buffer 膨胀后无空间）→ 判结构性受阻并留作未竟项。
- **wide 的剩余两处非向量时间**：`scalar_ratio 0.23–0.31`（逐通道 3 次 1 元素 GM→UB 取参 + 2 次 1 元素 UB→GM 回写 + strided plane 寻址 + 2 个 reduce 前缀）与 3.3µs 发射/尾波（26%）。批量取参需「UB→标量」读入形式，当前文档只允许 kernel 作用域标量（`T.vmul`/`T.vadd` §2.3 限制 2），无合法形态 → 未实证假设，未启动。
- **为何不是 [DESIGN_LIMIT]**：本轮已把「结构性」部分（op 数、流水解耦、tile 宽度）在现有设计族内消到地板，剩余缺口（SOL 11.4–33.3% vs 60%）由正确性硬约束（fp32 中转）+ 算法固有（vbrc 逐通道广播、3/5 遍流量）决定，属已知上轮归因；不存在被冻结的、可 >2x 的备选结构（Cube 归约仍为无实测先例的未实证假设）→ 不满足 perf-feedback.md §1 双门槛，**不产出 perf_feedback.md**。

---

## Final Performance Test Data

> 全部 final 时延来自**同一最终候选** `r3_final_w2s3`——合并后的 `perf_opt/BatchNormBwdOp.py`（三工厂：wide = R3 W2、split = R3 S3、general = R2 subid 版）。测量方式：`run_r3_ab.py` 在同一次 NPU 会话内交替测 R2 best 与最终候选后，对最终候选逐 workload 取 `msprof op` median-of-20 的 Task Duration（`launch-count=20`），每 workload 恰一条 `phase=final` 记录（见 `perf_records.jsonl`）。baseline 列为同批次（同日、相隔 20 分钟内）重测的 Stage-3 baseline，与 round-0 baseline（21.08/20.40/25.35/42.24/43.18/5365.03）偏差 ≤3.5%。

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---:|---:|---:|---|
| _batch_norm_bwd_wide_kernel | resnet-stage1-float16 | 20.520 | 12.600 | r3_final_w2s3 |
| _batch_norm_bwd_wide_kernel | resnet-stage1-bfloat16 | 21.160 | 11.480 | r3_final_w2s3 |
| _batch_norm_bwd_wide_kernel | resnet-stage2-float16 | 25.440 | 14.320 | r3_final_w2s3 |
| _batch_norm_bwd_wide_kernel | resnet-stage3-float16 | 41.260 | 20.690 | r3_final_w2s3 |
| _batch_norm_bwd_wide_kernel | resnet-stage3-float32 | 41.680 | 20.880 | r3_final_w2s3 |
| _batch_norm_bwd_split_kernel | large-spatial-float16 | 5375.400 | 2333.520 | r3_final_w2s3 |

逐 workload 提升（baseline → final）：−38.6% / −45.7% / −43.7% / −49.9% / −49.9% / −56.6%；相对 R2 best 的同会话配对提升（§4.5）：−12.3% / −16.7% / −15.0% / −12.4% / −13.5% / −19.8%。roofline `GM Read+Write` ratio：11.4% ~ 33.3%（wide 10.4–16.4%，split apply 33.3%）。

final 记录 artifact sha256：
- `BatchNormBwdOp.py` = `37099e2678f17ee65db331315d2175206a1a36e0409f5863f36562b2c396c37c`（final 记录 `artifact_path` 指向此文件，hash 与记录逐条一致）
- 分 kernel 文件：`_batch_norm_bwd_wide_kernel.py` = `b0b1a1ab35758a765d88d9472cc8f1e4566534e1ca44f773274224ae91c38aa1`、`_batch_norm_bwd_split_kernel.py` = `e284801e632ab9c3e0c5c81790c12bfdb2bfd05a39fb7035e2d625934b72f90b`、`_batch_norm_bwd_kernel.py`（general，subid-split，无 benchmark 命中）= `ffb95322d988ede3efe1f127b35d7b24e97f668c5b070a48b16665cf9a085aa3`

smoke 最终精度（smoke-fc [32,64,1] 三 dtype，wide kernel）：float32 PASS / float16 PASS / bfloat16 PASS（max_abs_err ≤ 1.9e-6，`logs/r3/merged_level_all.log`）。全量回归 `python perf_opt/BatchNormBwdOp.py --level all`：L0 14/14、L1 14 grid + 4 契约、L2 4、boundary 全 PASS。skip：`_batch_norm_bwd_kernel` 无 benchmark workload（非 benchmark 路径，见 §1 与 workload_inventory.json）。

---

## 5. Final Summary

- 每个 kernel 的最终候选：wide = **R3 W2**（x_hat 消去 + 常数折叠 + per-input staging），split = **R3 S3**（stats/apply per-input staging + apply epilogue 折叠 + apply 独立 tile 宽），general = R2 subid 版（无 benchmark 命中，仅 L0/L1/L2 验证，无性能记录）。三工厂合并为单一模块 `perf_opt/BatchNormBwdOp.py`（final 记录 artifact_path 指向该文件）。
- baseline→final 全 workload Task Duration 提升 **−38.6% ~ −56.6%**；其中 R3 单轮贡献 **−12.3% ~ −19.8%**（同会话配对，vs R2 best），R2 单轮贡献 −28.2% ~ −45.9%（subid 分片）。
- 精度无回归（L0 14/14、L1 14 grid + 4 契约、L2 4、boundary 全 PASS + smoke 三 dtype 全 PASS）。
- 中止原因：**success**——本轮目标（current best 再降 10–20%）达成（−12.3~−19.8%）；剩余方向为正确性硬约束（fp32 中转）、wrapper 签名约束（`finalize` 无 `mean`）或 UB 容量约束（apply vbrc hoist 需 81.6KB 常驻），见 §4.6。
- SOL（算力受限）未达 60%，缺口归因见 §6。

---

## 6. SOL 判据与缺口

- **判据选择**：本算子为 **Vector-only**（三 kernel 均单 `T.Scope("Vector")`，无 `T.gemm`/L1/L0C，msprof `aic_cube_ratio=0`）→ 非 Cube 主导混合类，**不适用 Mix ≥40% 口径**，按**算力受限 60%** 判。
- **达标结果**：SOL 11.4~33.3%（R3 final：wide 10.4~16.4%、split apply 33.3%），**未达 60%**。
- **缺口归因（roofline + pipe 证据）**：
  1. **fp32 中转（正确性硬约束）**：fp16/bf16 输入在 UB 中转 fp32 计算（bf16 无 Vector 算术、fp16 对齐 fp32 golden，DESIGN §1.6.1 机器验证），每元素 2 次 vcast + fp32 算术 = 数据宽度翻倍、ALU 吞吐减半。
  2. **Vec_MISC 35%**：vbrc 广播（逐通道 mean/rstd/scale/shift/xc，5 次/通道）+ vcast 占向量工作 35%（roofline computility_name 实测 `Vec_FP32(65%)+Vec_MISC(35%)`），即便 fp32 全速率也封顶 ~65% SOL。
  3. **标量 + 内存**：R3 final wide `aiv_scalar_ratio 23~31%`（逐通道 5 次 1 元素 copy + strided gather 地址 + 两步 reduce）、`aiv_mte2 17~27%`、split `aiv_mte2/mte3 25%/15%`（5-pass 流量）——R3 之后 `aiv_time ≈ Σpipe`（无重叠余量），向量引擎占空比 31~67%。
- **硬底线**：Task Duration 全部变短（−38.6~−56.6%），满足；60% SOL 属软目标，如实报告未达标，不硬凑。
- **不触发 [DESIGN_LIMIT]**：触发条件②（结构性加速估计 >2x 或实测与设计假设直接矛盾）不满足——fp32 中转/vbrc/内存是正确性或算法固有约束，不存在被设计冻结而可 >2x 加速的备选结构（subid 分片这一结构性修复已在 Stage 4 内完成）；条件①的"设计层天花板"亦不构成"存在更优算法族被冻结"（归约走 Cube 的极瘦 GEMM 无实测先例，标为未实证假设）。

---

## 7. Skill Retrospective

### 结论

一次有效轮次即达收束：subid 分片（PL-1.13）在本算子族（Vector-only 逐通道归约 persistent 分核）复现并验证（−28~−46%），证明该模式从 ssd_chunk_scan（MixCV Expert）可迁移到纯 Vector 逐通道归约族。skill 流程未见阻塞；测量 harness 的 opt_id 冲突 bug（详见下）已当场修正。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Runner | **实验批处理 runner 的 `opt_id` 必须全局唯一**：两实验同 `opt_id`（wide/split 均用 "baseline"/"final_subidsplit"）时 `modules[opt_id]` 被后写入覆盖，导致用 split 文件测 wide workload（AttributeError）或 final 记录候选 id 冲突（违反"同一 kernel 全部 final 用同一 candidate_id"）。本次首次 final 跑产生 1 条重复 final 行（已按"恰一条 phase=final"契约删除重复） | run_experiments.py 两次报 `no attribute _batch_norm_bwd_wide_kernel`；perf_records.jsonl 出现 large-spatial 两条 phase=final | T-4 模板注：opt_id 须含 kernel 前缀（`{opt}_{kernel}`）防多 kernel 冲突 | R |
| Pattern 迁移 | **PL-1.13 subid 分片适用于纯 Vector 逐通道归约族（非仅 MixCV）**：本算子（无 Cube、无 ws、无 flag 握手）的 persistent 分核同样命中"双 AIV 重复执行"（Mix Block Dim=80=2×40），判据"两 AIV 子块指标相同"成立；迁移形态从"lt-tile 蛇形"简化为"通道索引 +subid 线性偏移"（无需 ws 正交约束） | baseline profile vector0/vector1 指标完全相同 + Mix Block Dim 2× | PL-1.13 适用范围扩展到 Vector-only 逐通道归约 persistent；简化分片形态（通道级 +subid）作为更简先例 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **Vector-only 逐通道归约 persistent 分核的双 AIV 重复执行 + subid 通道分片**：`T.Kernel(min(C,40))` 的 `T.Scope("Vector")` 程序在每 block 双 AIV 重复执行（Mix Block Dim=80），通道循环 `c = i*num_kernels + cid` 未按 subid 区分；改 `num_kernels=min(C,20)` + `c = i*(num_kernels*2) + cid*2 + subid` 后全域 −28~−46%（msprof Task Duration） | P | baseline/candidate msprof profile（vector0/vector1 指标相同 + Mix Block Dim 2×）+ perf_records.jsonl round0/1 | 见本目录 `_batch_norm_bwd_wide_kernel.py`（subid 分片 delta） | pattern-library/attention.md PL-1.13 update / repro/PL-1.13 |
| **subid 分片使 msprof roofline ratio 减半（counted_flops 去双计）**：重复执行期 roofline `performance=counted_flops/duration` 把两 AIV 的重复 flops 各计一遍，分片后 counted_flops 减半 → ratio/AI 减半而 Task Duration 减半。跨 subid 前后对比只用 Task Duration，勿用 ratio | D | baseline ratio 17.99%/AI 7.0 → candidate ratio 14.35%/AI 3.5（同 workload，duration 21.08→13.26） | perf_records.jsonl + raw roofline visualize_data.bin | tilelang 0.1.2 dev-root + CANN 9.0.0 + Ascend910B3 / 2026-10-18 | pattern-library/constants.md（roofline 口径注） |
| **实验批处理 runner opt_id 须含 kernel 前缀防多 kernel 冲突** | R | 首次 final 跑 AttributeError + 重复 final 行 | run_experiments.py 修正 diff | 同任务 | tilelang-op-optimize autotune.md T-4 模板 |

### 复用教训（写给同族后续）

- **Vector-only 逐通道归约算子（BN 反向、逐通道 softmax 归约等）在 `T.Kernel(n)+T.Scope("Vector")` 下默认双 AIV 重复执行**——设计期"num_kernels = AI cores ×2"的假设会引入 2x 产能浪费而非 2x 吞吐；Stage 4 首查 `Mix Block Dim == 2×Block Dim` 且两 AIV 子块指标相同，命中即做 subid 通道分片。
- **fp32 中转（bf16 无 Vector 算术 + fp16 对齐 golden）是逐通道归约的 SOL 天花板主因**——SOL 封顶 ~65%（Vec_FP32 占 computility 65%），再叠加 vbrc 广播 + 内存，实测 14~25%。此类算子调优目标应锚定 Task Duration 而非 60% SOL。

---

## 8. R3 Skill Retrospective（结构级二轮）

### 结论

R3 用一个“先 profile 证伪假设、再单点改结构”的循环拿到 −12.3~−19.8%：**先证伪调度 prompt 的 single-scan 假设**（`_plan_bS` 实际给 bS=1024，held 路径早已生效），把预算全部投到实测瓶颈（`aiv_time ≈ Σpipe` → 单 staging 串行；每元素 12–17 个向量 op）。三个有效单点改动（常数代数折叠 / loads-first 解耦 / epilogue 折叠）都可跨 kernel 复用。

### Skill Flow Issues（R3）

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| 度量纪律 | **同一二进制跨进程双态在 10–20µs 级 kernel 上达 ~7%**（wide stage1：12.12 / 13.02µs 两次独立运行，各自分布很窄 min–max ≤5%），比 R3 目标增益（10–20%）同阶 → 单跑判定会误判 | `profiles/candidate-r3-r3_w2_loadsfirst-resnet-stage1-float16`（12.12，min 11.74/max 12.90）vs `profiles/final-r3-r3_final_w2s3-resnet-stage1-float16`（13.02，min 12.42/max 13.38），同一 `BatchNormBwdOp.py` | `ab_test.py` 的触发条件从「<5% 差异」上调为「<5% 差异 **或** 目标增益 <2× 观测到的跨进程双态幅度」；T-4 runner 应内置「同会话交替测 A/B」开关（本次临时加 `run_r3_ab.py` 才拿到干净比值） | R |
| 工具链陷阱 | **`hivm.hir.vsub` 在 (1,1) 标量平面上下沉为 `ins(f32, f32)` 后 verifier 报 `operand at index 0 is vector-only`**（编译失败）；而同形 `vmul`/`vadd` 与 `vadd` 别名形态可编译。绕法：标量平面只用 `vmul`/`vadd`（减法改为「乘 −1 再加」） | `perf_opt/probe_scalar_plane.py`：`vsub11: FAIL` / `vmul11/vadd11/vadd11_noalias/vadd_full: COMPILE OK`（CANN 9.0.0 + tilelang 0.1.2 dev-root / 910B3 / 2026-10-20） | pattern-library traps-compiler.md 新增 TRAP-（标量平面 vsub 非法；含 probe 形态） | D |
| 代数化简 | **逐通道归约算子的 `x_hat` 可从逐元素链消去**：`Σ g·x̂ = rstd·Σ(g·x) − (mean·rstd)·Σg` 只需 1 个逐元素 mul；`grad_x` 同步折叠为 `A·g + B·x + C`。前提是 kernel 内能拿到 `mean`（wide 满足；split 因 `finalize` 签名无 `mean` 不可用——wrapper 只读） | wide pass1 9→5 op/elem：stage1-fp16 13.27→12.01µs（−9.5%）、stage3-fp32 24.82→22.53（−9.2%）；L0 14/14（含 S=1048576 长归约，折叠未损精度） | pattern-library elementwise.md 新增「逐通道归约的 x_hat 消去 + 仿射折叠」条目 | P |
| 流水解耦 | **单 staging 缓冲复用链在 split stats/apply 上同样成立**（PL-1.10 的第三、四证）：apply `mte2_time` 496→314µs、stats 394→271µs，二者各 −6.6%/−7.7%；`brc` 可由结果缓冲复用（先当广播 scratch、后当结果），UB 不增 | `perf_records.jsonl` r3_s1_loadsfirst / r3_s2_applyfold / r3_s3_statsloadsfirst 行 + `profiles/*/stats|apply` | PL-1.10 适用范围补「split 类三段 kernel 的 stats/apply；brc→prod 复用保持 UB 中性」 | P |

### Value Point Proposals（R3）

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| (1,1) 标量平面 `vsub` 非法（下沉成 f32 标量后 verifier 拒绝）→ 标量代数只用 vmul/vadd | D | `probe_scalar_plane.py` 五形态编译矩阵 | `perf_opt/probe_scalar_plane.py` | tilelang 0.1.2 dev-root + CANN 9.0.0 + 910B3 / 2026-10-20 | pattern-library/traps-compiler.md |
| 逐通道归约 `x_hat` 消去 + `grad_x` 仿射折叠（17→13 op/elem） | P | 上表 wide 5 workload、L0 14/14 | `_batch_norm_bwd_wide_kernel_r3_w2.py`（vs `_batch_norm_bwd_wide_kernel_round1_op1.py` 的 delta） | 同上 | pattern-library/elementwise.md |
| 小 shape 跨进程双态 ~7% ⇒ A/B 必须同会话交替 | R | 12.12 vs 13.02 同文件 | `run_r3_ab.py` | 同上 | tilelang-op-optimize references/autotune.md T-4 / ab_test.py |
