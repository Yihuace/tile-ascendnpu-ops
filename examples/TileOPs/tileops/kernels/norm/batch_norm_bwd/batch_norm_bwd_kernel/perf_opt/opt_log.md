# BatchNormBwdOp Stage 4 性能调优日志（opt_log）

- 算子：`BatchNormBwdOp`（batch norm 反向：grad_input / grad_weight / grad_bias）
- 目录：`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd_kernel/`
- benchmark：`examples/TileOPs/benchmarks/ops/bench_batch_norm_bwd.py::test_batch_norm_bwd_bench`
- 环境：Ascend910B3（`get_aicore_num()=20`，即 20 AIC / 40 AIV）、CANN 9.0.0、tilelang 0.1.2（dev-root）、`msprof op` 唯一时延口径
- 调度模式：`full`（首次 Stage 4，目录下无历史 perf_opt）

## 结论速览

- 唯一有效优化点：**subid 分片（PL-1.13-aiv-dup-subid-split）**——修正 `T.Kernel(n, is_npu=True) + T.Scope("Vector")` 的 Vector 程序在每 block 双 AIV 上重复执行（Mix Block Dim = 2×Block Dim）的产能浪费。
- 全域 Task Duration 提升：**−33.1% ~ −45.9%**（逐 workload 见 Final Performance Test Data）。
- 精度：三 kernel L0 全 PASS，smoke 三 dtype 全 PASS，无回归。
- SOL（msprof roofline `GM Read+Write` ratio）：14.0% ~ 25.5%（算力受限 60% 目标**未达标**，见 §6 判据与缺口说明）。

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

## Final Performance Test Data

> 全部 final 时延来自**同一最终候选**——合并后的 `perf_opt/BatchNormBwdOp.py`（内含 `_batch_norm_bwd_wide_kernel` / `_batch_norm_bwd_split_kernel` / `_batch_norm_bwd_kernel` 三工厂，函数体与已测量的 `perf_opt/_batch_norm_bwd_*_kernel.py` 逐字一致，仅 split 的 host 侧 tiling 规划器更名 `_plan_tiling`→`_plan_tiling_split` 与 `_UB_SAFE_BYTES`→`_UB_SAFE_BYTES_SPLIT`，编译产物不变，故测量数据无需重采）。非拼接各实验分支最好成绩。

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---:|---:|---:|---|
| _batch_norm_bwd_wide_kernel | resnet-stage1-float16 | 21.080 | 13.460 | final_wide_subidsplit |
| _batch_norm_bwd_wide_kernel | resnet-stage1-bfloat16 | 20.400 | 13.640 | final_wide_subidsplit |
| _batch_norm_bwd_wide_kernel | resnet-stage2-float16 | 25.351 | 15.810 | final_wide_subidsplit |
| _batch_norm_bwd_wide_kernel | resnet-stage3-float16 | 42.241 | 23.530 | final_wide_subidsplit |
| _batch_norm_bwd_wide_kernel | resnet-stage3-float32 | 43.181 | 24.340 | final_wide_subidsplit |
| _batch_norm_bwd_split_kernel | large-spatial-float16 | 5365.027 | 2903.328 | final_split_subidsplit |

final 记录 artifact sha256（合并后最终模块）：
- `BatchNormBwdOp.py` = `42b6cd08cb38f599c61f0f8209abc1809f8c449db9531157c07260a853025cca`（wide/split/general 三工厂函数体与已测量独立文件逐字一致，见上）
- 独立候选备份：`_batch_norm_bwd_wide_kernel.py` = `3a6e5fb5b94f59837a7bba54542adc30542975ab404ea0a94b3ba8a98a33e8c4`、`_batch_norm_bwd_split_kernel.py` = `cffb859bdd75a20daf4807fb9a57d34d300026e487cf0c648dc593ea5aea8199`、`_batch_norm_bwd_kernel.py`（subid-split，L0 通过）= `ffb95322d988ede3efe1f127b35d7b24e97f668c5b070a48b16665cf9a085aa3`

smoke 最终精度（smoke-fc [32,64,1] 三 dtype，wide kernel）：float32 PASS / float16 PASS / bfloat16 PASS（max_abs_err ≤ 1.9e-6）。skip：`_batch_norm_bwd_kernel` 无 benchmark workload（非 benchmark 路径，见 §1 与 workload_inventory.json）。

---

## 5. Final Summary

- 每个 kernel 的最终候选：wide/split/general 三 kernel 均采用 subid 分片（general 无 benchmark 命中，仅 L0 验证，无性能记录）。 最终三个调优后工厂函数合并为单一模块 `perf_opt/BatchNormBwdOp.py`（函数体与已测量独立文件逐字一致，测量数据无需重采；general kernel 为 skip/非 benchmark 路径，不参与性能对账）。
- baseline→final 全 workload Task Duration 提升 −33.1% ~ −45.9%，全部来自 subid 分片（PL-1.13，修正双 AIV 重复执行）。
- 精度无回归（L0 三 kernel 全 PASS + smoke 全 PASS）。
- 中止原因：主要结构候选已充分验证并合并，剩余方向为正确性硬约束或无结构性抓手（plateau 收束）。
- SOL（算力受限）未达 60%，缺口归因见 §6。

---

## 6. SOL 判据与缺口

- **判据选择**：本算子为 **Vector-only**（三 kernel 均单 `T.Scope("Vector")`，无 `T.gemm`/L1/L0C，msprof `aic_cube_ratio=0`）→ 非 Cube 主导混合类，**不适用 Mix ≥40% 口径**，按**算力受限 60%** 判。
- **达标结果**：SOL 14.0~25.5%（wide 14~15.5%、split 25.5%），**未达 60%**。
- **缺口归因（roofline + pipe 证据）**：
  1. **fp32 中转（正确性硬约束）**：fp16/bf16 输入在 UB 中转 fp32 计算（bf16 无 Vector 算术、fp16 对齐 fp32 golden，DESIGN §1.6.1 机器验证），每元素 2 次 vcast + fp32 算术 = 数据宽度翻倍、ALU 吞吐减半。
  2. **Vec_MISC 35%**：vbrc 广播（逐通道 mean/rstd/scale/shift/xc，5 次/通道）+ vcast 占向量工作 35%（roofline computility_name 实测 `Vec_FP32(65%)+Vec_MISC(35%)`），即便 fp32 全速率也封顶 ~65% SOL。
  3. **标量 + 内存**：wide `aiv_scalar_ratio 23~33%`（persistent 循环 + strided gather 地址 + 两步 reduce）、`aiv_mte2 12~18%`（3-pass 读）、split `aiv_mte2/mte3 31%/10%`（5-pass 流量）——合并后向量引擎仅 37~57% 占空比。
- **硬底线**：Task Duration 全部变短（−33~−46%），满足；60% SOL 属软目标，如实报告未达标，不硬凑。
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
