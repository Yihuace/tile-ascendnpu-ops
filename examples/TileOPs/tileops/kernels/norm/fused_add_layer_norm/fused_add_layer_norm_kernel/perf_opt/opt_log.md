# FusedAddLayerNormFwdOp — Stage 4 调优日志（三轮深挖）

> CANN 9.0.0 / tilelang 0.1.2 dev root（`/mnt/workspace/tilelang-mlir-ascend`）/ Ascend910B3（20 AICore × 2 AIV = 40 向量核）/ 2026-10-20。
> 口径：`msprof op` Task Duration(us)（唯一时延口径）+ GM Read+Write SOL ratio。
> 分类：prefill AI=3.0~3.8 > 平衡点 2.6 ⇒ **算力受限，线 60%**；13b fp32 AI=1.02 ⇒ **访存受限，线 70%**；decode（4.0–4.2µs）属微型 shape，性能豁免。

## 0. 对象与入口

- `FusedAddLayerNormFwdOp`；kernel `_fused_add_layer_norm_kernel(M,N,eps,dtype)`（factory → `_func(block_m)`）。
- 生产基线 = Stage-3 kernel `_fused_add_layer_norm_kernel.py`（wrapper 现接此文件，本轮只读）。
- 产物 `perf_opt/FusedAddLayerNormFwdOp.py`；benchmark `bench_fused_add_layer_norm.py` 展开 7 tune + 3 smoke。
- 轮次谱系：R2 `block_m` → R3 `hoist-wb` → R5 `loadsfirst` → R6 合并 final → **R7 本轮（op1 per-input staging + op2 fp32 affine hoist）**。

## 1. 上一轮（R2–R6）结论回顾

- R2：`block_m=1` 慢 25%（N=4096），默认 `_default_block_m`（4096→2、8192/5120→1）正确。
- R3：weight/bias 循环不变量外提（A10）——70b prefill −14.0~−14.5%、8b prefill −6.4~−7.1%。
- R5：`loadsfirst`（fp16/bf16 两输入前置装载 + `block_m` 编译期分支）——8b prefill 再 −18%；`singlepass`（`E[x²]−E[x]²`）tie，弃。
- R6：合并 `hoist-wb + loadsfirst` 为 final（基线 82.452/84.012/164.233/168.283/107.642/4.010/4.460µs）。
- R6 遗留：70b prefill 只拿到 −14%（`block_m==1` 因 UB 差 256 bits 无法走 loads-first）；13b fp32 全程未改动；8b vec 链已到 62–64µs。

## 2. R5（历史）诊断与候选表

| 观测（8b-prefill fp16 baseline） | 数值 | 结论 |
|---|---|---|
| aiv_vec / aiv_mte2 / aiv_mte3 | 45.28µs（0.605）/ 22.16µs（0.296）/ 9.40µs（0.126） | MTE2/VEC 交替串行（单 `stage` 四角色复用） |
| 修复 | 双输入前置装载（copy-cast，PL-1.10） | 8b prefill −18% |

## 3. R6 最终表（历史，已被 R7 取代）

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| _fused_add_layer_norm_kernel | llama-13b-prefill-float32 | 107.642 | 108.702 | loadsfirst |
| _fused_add_layer_norm_kernel | llama-70b-decode-bfloat16 | 4.460 | 4.820 | loadsfirst |
| _fused_add_layer_norm_kernel | llama-70b-prefill-bfloat16 | 168.283 | 144.733 | loadsfirst |
| _fused_add_layer_norm_kernel | llama-70b-prefill-float16 | 164.233 | 140.873 | loadsfirst |
| _fused_add_layer_norm_kernel | llama-8b-decode-bfloat16 | 4.010 | 4.340 | loadsfirst |
| _fused_add_layer_norm_kernel | llama-8b-prefill-bfloat16 | 84.012 | 64.551 | loadsfirst |
| _fused_add_layer_norm_kernel | llama-8b-prefill-float16 | 82.452 | 63.251 | loadsfirst |

> 簿记：R6 的 7 条 `phase=final` 记录在 R7 收束时降级为 `phase=merged`（仅 phase 标签，round / candidate_id / duration_us / artifact_sha256 / msprof_raw_path 均未改动），原因同 R6 对 R4 的处理——gate 4 要求每个 tune workload 恰有一条属于当前最终候选的 final 记录。备份：`/tmp/opencode/perf_records_before_r7_phasefix.jsonl`。

## 4. R7 深挖

### 4.1 诊断（本轮实测，current best = R6 final）

`msprof op --aic-metrics=Default,PipeUtilization,Roofline`，`launch-count=20`，raw：`profiles/r7diag-cur-8b-fp16/`。

| 指标（8b-prefill fp16，current best） | 数值 |
|---|---|
| Task Duration(us)（median of 20） | 62.69（min 62.26 / max 63.08） |
| Block Dim / Mix Block Dim | 40 / NA |
| aiv_time / aiv_vec_time / aiv_vec_ratio | 60.0µs / 45.35µs / **0.756** |
| aiv_mte2_time / ratio | 20.5µs / 0.336 |
| aiv_mte3_time / ratio | 10.3µs / 0.170 |
| aiv_scalar_time / aiv_scalar_wait_time | 9.4µs / 49.5µs |

当前现象：

- **P1（向量链主导）**：vec 0.756 忙、MTE2 0.336、MTE3 0.170 ⇒ 已由 R5 的「装载串行」转为**向量链 + 每 task 依赖延迟**主导；`aiv_time 60.0` 与 `Task Duration 62.69` 只差 2.7µs（launch/尾），无核间不均（1024 logical / 40 核 = 25.6，imbalance 1.6%）。
- **P2（无双 AIV 冗余）**：PipeUtilization 每 block 只有 1 条 `vector0` 子块、Mix Block Dim = NA ⇒ 不命中 PL-1.13「双 AIV 重复执行」；`num_kernels = AI cores × 2 = 40` 是本设备 AIV 满载（1 wave），不是 2× 浪费。**否决「subid 分片」候选**（前置排查，未做实验分支）。
- **P3（70b 的 block_m==1 仍走单 staging 链）**：R6 的 UB 差 256 bits 约束仍在——但 `block_m==1` 时 resident `stage` 本身就是 `(1, N)`，`wstage` 与它同形冗余。
- **P4（13b fp32 的 weight/bias 仍在循环内重读）**：fp32 路径每个 serial task 重新 `T.copy(weight)`/`T.copy(bias)`（2×N×4B），而 payload 只有 4×N×4B ⇒ **GM 流量 +50%**（漏做了 R3 的 A10 外提，只覆盖了 transit 路径）。

理论估算（13b fp32，P4）：payload 168MB，加 weight/bias 后 252MB；79µs 下 168MB ⇒ 2.13 TB/s，已贴 fabric 上限，故 P4 的收益上限 ≈ 33% 流量。

### 4.2 本轮候选优化点

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| op1 `pinput_stage` | P3：`block_m==1` 的 MTE2→VEC→MTE2→VEC 串行 | per-input staging（复用 `wstage` 作第二 dtype 平面） | PL-1.10 第二形态；`block_m==1` 时两平面同形，UB 零增量 | x→`wstage`、residual→`stage` 两条 MTE2 先发，再两次 vcast | Task Duration / L0 | improved |
| op2 `fp32hoist` | P4：fp32 每 task 重读 weight/bias | affine 因子外提 + 增 `b1` | R3 A10 未覆盖 fp32 路径 | 循环前 `T.copy` w/b 一次，循环内只用 w1/b1 | Task Duration / L0 | improved |
| op3 `pipe2` | P1：task 间是否还有 MTE/VEC 未重叠 | `T.serial` → `T.Pipelined(num_local_tasks, num_stages=2)` | roofline advice = latency bound:pipeline caused | 两处 loop 换 Pipelined | Task Duration | config_no_gain（tie） |
| op4 `storecast` | P1：输出 round-back 多一遍 vcast | 用跨 dtype `T.copy(x_f32 → output)` 折叠 round | TRAP-C12（copy 静默跨 dtype） | bm>=2 分支删 `vcast`；bm==1 保留（staging 会 UB 溢出） | Task Duration / L0 | invalid（性能回退） |
| op5 `hoistbrc` | P1：affine 的 2 次 `vbrc` 是全平面 pass | `vbrc(w1)/vbrc(b1)` 提到循环外（`(block_m,N)` fp32 平面） | 12 pass 中可去 2 pass | 新增 `wbrc`/`bbrc` 两平面，循环内 vmul/vadd | Task Duration / L0 | config_no_gain（tie） |

前置否决（未起实验分支，证据见 4.1）：subid 分片（P2，无 AIV 重复执行）。

### 4.3 候选 vs current best 对比表（R7，同会话、同 workload、A/B/A/B 交错）

AICore 列：`vec/mte2` 为 PipeUtilization 实测 ratio；未采 PipeUtilization 的分支标 `n/p`（Task Duration 是唯一裁决口径）。

| candidate | workload | Task Duration(us) | vs current best | SOL% | AICore vec/mte2 | L0 |
|---|---|---:|---:|---:|---|---|
| current best（R6 final） | 70b-prefill-fp16 | 141.36 / 141.14 | — | 29.20 / 29.29 | n/p | PASS |
| **op1 pinput_stage** | 70b-prefill-fp16 | **117.32 / 116.98** | **−17.0%** | 35.02 / 35.20 | n/p | PASS |
| current best | 70b-prefill-bf16 | 144.92 / 144.78 | — | 33.25 / 33.41 | n/p | PASS |
| **op1 pinput_stage** | 70b-prefill-bf16 | **120.53 / 120.54** | **−16.8%** | 40.27 / 40.20 | n/p | PASS |
| current best | 13b-prefill-fp32 | 108.00 / 108.54 | — | 44.80 / 46.10 | n/p | PASS |
| **op2 fp32hoist** | 13b-prefill-fp32 | **78.82 / 79.82** | **−26.6%** | 61.34 / 61.34 | n/p | PASS |
| current best | 8b-prefill-fp16 | 62.51 / 62.84 | — | 33.14 / 32.81 | 0.756 / 0.336 | PASS |
| op3 pipe2 | 8b-prefill-fp16 | 62.86 / 62.66 | +0.3% / −0.3%（tie） | 33.04 / 33.00 | n/p | PASS |
| op4 storecast | 8b-prefill-fp16 | 65.67 / 65.77 | +5.1%（回退） | 31.51 / 31.42 | n/p | PASS |
| current best | 8b-prefill-bf16 | 64.72 / 64.53 | — | 37.76 / 37.77 | n/p | PASS |
| op3 pipe2 | 8b-prefill-bf16 | 64.81 / 64.82 | +0.1%（tie） | 37.59 / 37.63 | n/p | PASS |
| op4 storecast | 8b-prefill-bf16 | 67.95 / 67.94 | +5.0%（回退） | 35.65 / 35.72 | n/p | PASS |
| merged（op1+op2） | 8b-prefill-fp16 | 63.21 / 62.92 | +0.2% / 0.0%（tie） | 32.89 / 33.01 | n/p | PASS |
| op5 hoistbrc | 8b-prefill-fp16 | 63.07 / 62.50 | −0.2% / −0.7%（tie） | 33.22 / 33.50 | n/p | PASS |
| merged（op1+op2） | 8b-prefill-bf16 | 64.64 / 64.73 | −0.1% / +0.2%（tie） | 37.51 / 37.62 | n/p | PASS |
| op5 hoistbrc | 8b-prefill-bf16 | 64.54 / 65.29 | −0.3% / +0.9%（tie） | 38.00 / 37.21 | n/p | PASS |

判定（噪声阈值 3%，平区用交错多 run 方向一致性）：

- **op1 胜出**：70b prefill 双 dtype 一致 −17%（两 rep 同向，远超阈值）→ 合入。
- **op2 胜出**：13b fp32 −26.6~−27.3%（两 rep 同向）→ 合入。
- **op3 tie**：4 个 (workload, rep) 组合方向不一致（fp16 rep1 +0.3% / rep2 −0.3%；bf16 两 rep 均 +0.1%）⇒ `config_no_gain`，不采纳（serial 循环已被 auto-multi-buffer 覆盖，task 级软件流水无额外空间）。
- **op4 回退**：+5.0~5.1% 稳定同向 ⇒ `invalid`（跨 dtype `T.copy` 需内部 staging 平面 + 额外 MTE，1 个 vcast 换 1 次 staging 不划算）。
- **op5 tie**：去 2/12 全平面 pass 后 Task Duration 不动（4 个组合两 rep 方向不一致，最大 −0.7%）⇒ `config_no_gain`。该结果同时是 **P1 归因证据**：vec 时间不是 pass 数线性驱动，而是每 task 的依赖链延迟 + reduce 串行。
- **merged 无回退**：8b prefill 双 dtype 与 current best 差 ≤0.2%（tie），70b/13b 承接 op1/op2 收益。

### 4.4 当前 kernel 全部 tune workload 合并检查（merged = op1+op2）

| merged_candidate | kernel_id | workload_id | current_best_us | merged_us | precision | status |
|---|---|---|---:|---:|---|---|
| FusedAddLayerNormFwdOp_r7_merged.py | _fused_add_layer_norm_kernel | llama-8b-prefill-float16 | 62.51 | 62.92 | PASS | pass（tie） |
| FusedAddLayerNormFwdOp_r7_merged.py | _fused_add_layer_norm_kernel | llama-8b-prefill-bfloat16 | 64.53 | 64.73 | PASS | pass（tie） |
| FusedAddLayerNormFwdOp_r7_merged.py | _fused_add_layer_norm_kernel | llama-70b-prefill-float16 | 141.14 | 116.79 | PASS | pass（−17.3%，见 4.3/§5） |
| FusedAddLayerNormFwdOp_r7_merged.py | _fused_add_layer_norm_kernel | llama-70b-prefill-bfloat16 | 144.78 | 120.40 | PASS | pass（−16.8%） |
| FusedAddLayerNormFwdOp_r7_merged.py | _fused_add_layer_norm_kernel | llama-13b-prefill-float32 | 108.54 | 79.01 | PASS | pass（−27.2%） |
| FusedAddLayerNormFwdOp_r7_merged.py | _fused_add_layer_norm_kernel | llama-8b-decode-bfloat16 | 4.34（R6） | 4.19 | PASS | pass（微型豁免） |
| FusedAddLayerNormFwdOp_r7_merged.py | _fused_add_layer_norm_kernel | llama-70b-decode-bfloat16 | 4.82（R6） | 4.21 | PASS | pass（微型豁免） |

smoke（3 例，仅精度）：`--level all` 全过（见 §7）。

### 4.5 Iteration Winner

- winner：`FusedAddLayerNormFwdOp.py` = R7 merged（op1 + op2），sha256 `6b051de2958b086cbb6c5f23cb57681911af9ae3a327564b61adca96867528be`。
- reason：同一 kernel 同一 `@T.prim_func` 家族内两处编译期分支（`block_m==1` per-input staging；fp32 affine 外提），全部 7 tune workload 无可信回退（8b 双 dtype tie，70b/13b 显著下降），smoke 精度通过。
- new_current_best：`perf_opt/FusedAddLayerNormFwdOp.py`（R6 final 已被取代，其 final 记录降级为 merged）。
- rollback_branches：`FusedAddLayerNormFwdOp_r7_pipe2.py`（tie）、`FusedAddLayerNormFwdOp_r7_storecast.py`（回退）、`FusedAddLayerNormFwdOp_r7_hoistbrc.py`（tie）。三文件保留在 `perf_opt/` 作证据，不参与 wrapper 采纳。

### 4.6 R7 收束判定（plateau / ceiling 证据）

- 8b prefill：三方向（task 级 `T.Pipelined` tie、去 2/12 全平面 pass tie、round-back 折叠回退 +5%）全部被实测否决 ⇒ 「8b 已到本设计族结构上限」的**本轮证据**：vec 0.756 / mte2 0.336 / aiv_time 60.0 vs Task Duration 62.7（核间与 launch 只占 4%），剩余时间是每 task 依赖链（2 次 row-reduce 串行 +16 个链接）延迟，非 tiling/参数可解。
- 未启动的方向与理由：① 双 reduce 合并为单次 `reduce_sum`——语义不可能（variance 需先有 mean，`T.reduce` 无多输出，`docs/Tilelang.language/规约操作/T.reduce.md` §2.2.2 只允许单维不同）；单遍方差已在 R5 tie 且数值风险，弃。② N 方向分核（partial-sum + finalize）——需第二个 kernel + GM workspace，改变 wrapper 单 kernel 契约，且 40 AIV 已满载、收益需 > 额外 pass 成本，本轮预算内不立项。③ Cube GEMV 归约——Developer 模式 persistent + `T.gemm` 混排运行期崩溃（CG-2026-0010），需 Expert 重写，未实测，标为「未实证假设」。
- **不触发 `[DESIGN_LIMIT]`**：①归因落在本设计族的向量链延迟，但 ②「结构性加速估计 > 2x 或实测与 DESIGN 假设直接矛盾」不满足——本轮实测（8b 62.65 vs 基线 82.14、70b 116.79 vs 165.31、13b 79.01 vs 106.52）显示设计族本身仍可通过实现级修复拿到 −24~−29%，且无被冻结的备选结构有 >2x 实测证据。故不产出 `perf_feedback.md`。

## 5. R8 深挖：行批处理（row batching）与 UB 预算重标定

> 工具链：`tilelang 0.1.2` dev root（HEAD `9309edb`，包戳 `3c4e9cf9`，本地 `jit_npu.py` CANN9 shim 未变）/ CANN 9.0.0 / 910B3 / 2026-10-21。
> **模式前提（本轮踩坑）**：`tileops_env.sh` 默认导出 `TILELANG_ASCEND_MODE=Expert`，而本 kernel 是 Developer 模式 kernel；
> 未显式 `export TILELANG_ASCEND_MODE=Developer` 时 L0 会 13/13 全挂并以 `'hivm.hir.store' op only support store ub to gm currently!` 失败
> （易误判为 kernel 回归）。本轮所有 NPU 命令均显式置 Developer。

### 5.1 起点与假设

- current best = R7 final（sha256 `6b051de2…`），8b-prefill fp16 62.65µs。
- 假设（来自 `probe_cube_gemv` 同批实测）：行批处理摊薄 row-reduce——`bm=4,N=4096` reduce **0.228 vs `bm=2` 0.319 µs/row（−29%）**；
  reduce 占 R7 每 tile 2.41µs 的 26%（单次）。按 8b 2048 行 / 40 核折算，理论窗口 5–10%。

### 5.2 UB 预算模型重标定（本轮首个可复用结论）

R7 的 `_UB_MAX_ELEMS=9216`（20 B/elem 包络）把 N=4096 的 `block_m` 钉在 2。本轮换成**逐 dtype 平面清单模型**
`_ub_footprint_bytes(block_m, N, dtype)`：`fp16/bf16 且 bm≥2 = 10 B/elem + 8N`；`fp16/bf16 且 bm==1 = 20N`；`fp32 = 8 B/elem + 8N`；
auto-multi-buffer 打开时 bm≥2 路径再加 **+6 B/elem**。以后端容量原文标定：

| 配置 | 模型预测 | 后端自报 | 结论 |
|---|---:|---:|---|
| fp16 N=4096 bm=3，auto-mb ON | 229,376 B | **229,408 B**（`requires 1835264 bits`） | 模型误差 32 B |
| fp16 N=4096 bm=3，auto-mb OFF | 155,648 B | **204,832 B**（`requires 1638656 bits`） | OFF 只去掉 dtype 平面双缓冲；仍多出 49,184 B = copy-cast 的 (bm,N) staging 平面（+4 B/elem，模型未含此项，故 OFF 分支的模型只用于判方向、不用于合法化 shape） |
| fp16 N=4096 bm=4，auto-mb OFF | 196,608 B | **262,176 B**（`requires 2097408 bits`） | 同上，+65,568 B |
| fp16 N=4096 bm=2，auto-mb ON（R7 默认） | 163,840 B | 编译通过 | ≤196,608 B |
| fp16 N=8192 bm=1（R7 默认） | 163,840 B | 编译通过 | 再加 1 个 (1,N) fp32 平面即被拒（R7 记录） |

### 5.3 本轮候选优化点

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| op0 `bm3`/`bm4` | 8b reduce 占 tile 26%（单次 0.637µs / 2.41µs），bm=4 每行 reduce −29% | 行批处理 `block_m=3/4`（N=4096） | 探针同批 µs/row 曲线 | 预算模型 + tuned 表，kernel body 不变 | 编译 / UB 账 | **UB-blocked**（5.3.1） |
| op1 `fp32_bm2` | 13b fp32 每 tile 仅 1 行（bm=1），2048 logical / 40 核 = 52 task/core，MTE 每 tile 重新起停 | fp32 路径 `block_m=1→2`（N=5120） | 新模型 184,320 B ≤ 196,608 B，且 bm=2 摊薄 reduce | `_TUNED_BLOCK_M = {("fp32", 5120): 2}`（仅默认参数） | Task Duration / L0 | **improved（−13.1~−13.7%）** |
| op2 `mb0` | bm≥3 的 96,768 B 膨胀来自 auto-multi-buffer | 关闭 `npuir.enable_auto_multi_buffer` | 膨胀项实测 = x_f32 + stage 双缓冲 | `pass_configs={"npuir.enable_auto_multi_buffer": False}` | 编译 | **不可行**（只省 24,576 B；bm=3 仍需 204,832 B > 196,608 B，且 flag 全局生效会波及 70b/13b 路径） |

#### 5.3.1 行批处理为何判 UB-blocked（不靠删功能硬塞）

- bm=3（auto-mb ON）需 **229,408 B** vs 容量 **196,608 B** ⇒ 缺口 **32,800 B**；auto-mb OFF 仍需 **204,832 B**（缺口 8,224 B）；
  bm=4（OFF）需 262,176 B（缺口 65,568 B）。
- bm=3 的**逻辑下界**已 `10 B/elem × 3 × 4096 + 32,768 = 155,648 B`；后端的 +6 B/elem 双缓冲与结构无关
  （`probe_cube_gemv` §3 亦显示 bm≥8 被 UB 限）。
- 「stats 批处理 + epilogue 分块」（方向 1）不降低驻留下界：stats pass 本身要 4 行同时驻留 fp32（`x_f32`+`sq_f32` = 8 B/elem），
  bm_stats=4 时逻辑下界 `10×4×4096 + 32,768 = 196,608 B` **恰等于容量**（R7 已证等价于拒绝），叠加 +6 B/elem 为 294,912 B。
- 两项"节食"各有已实测代价：① 去 `stage` 平面走跨 dtype store（R7 op4 **+5.0%**）；② 串行化 affine 平面（weight/bias 共用一个
  (1,N) fp32 平面，每 tile 重载 = +16 KB GM 读 + 额外同步）。叠加 ≥5%，而行批处理理论上限仅 5–10% ⇒ 净收益不可信，不硬塞。
- 70b（N=8192）bm=2 需 `16 B/elem × 2 × 8192 + 65,536 = 327,680 B`（缺口 131,072 B），明确不做。

### 5.4 候选 vs current best 对比表（R8，同会话、同 workload、交错 A/B/A/B）

| candidate | workload | Task Duration(us) | vs current best | SOL% | AICore vec/mte2 | L0 |
|---|---|---:|---:|---:|---|---|
| current best（R7 final，sha `6b051de2`） | 13b-prefill-fp32 | 78.74 / 78.40 | — | 60.40 / 60.25 | 0.599 / 0.467（R7 画像） | PASS |
| **op1 fp32_bm2（bm=2）** | 13b-prefill-fp32 | **68.39 / 67.67** | **−13.1% / −13.7%** | **70.07 / 68.53** | n/p | PASS |
| op0 bm3（auto-mb ON） | 8b-prefill-fp16 | 编译失败 | — | — | — | n/a（229,408 B > 196,608 B） |
| op0 bm3（auto-mb OFF） | 8b-prefill-fp16 | 编译失败 | — | — | — | n/a（204,832 B > 196,608 B） |
| op0 bm4（auto-mb OFF） | 8b-prefill-fp16 | 编译失败 | — | — | — | n/a（262,176 B > 196,608 B） |

判定（噪声阈值 3%）：op1 两 rep 同向 −13.1%/−13.7%（≫3%），SOL 由 60.2–60.4% 抬到 68.5–70.1% ⇒ **胜出，合入**。
op0/op2 为 blocked 分支（无 valid 时延，不进 perf_records）。

### 5.5 当前 kernel 全部 tune workload 合并检查（merged = R7 + op1）

| merged_candidate | kernel_id | workload_id | current_best_us | merged_us | precision | status |
|---|---|---|---:|---:|---|---|
| FusedAddLayerNormFwdOp.py（R8） | _fused_add_layer_norm_kernel | llama-8b-prefill-float16 | 62.65 | 62.94 | PASS | pass（+0.5%，噪声内） |
| FusedAddLayerNormFwdOp.py（R8） | _fused_add_layer_norm_kernel | llama-8b-prefill-bfloat16 | 64.43 | 64.48 | PASS | pass（+0.1%，噪声内） |
| FusedAddLayerNormFwdOp.py（R8） | _fused_add_layer_norm_kernel | llama-70b-prefill-float16 | 116.79 | 117.10 | PASS | pass（+0.3%，噪声内） |
| FusedAddLayerNormFwdOp.py（R8） | _fused_add_layer_norm_kernel | llama-70b-prefill-bfloat16 | 120.40 | 120.44 | PASS | pass（0.0%） |
| FusedAddLayerNormFwdOp.py（R8） | _fused_add_layer_norm_kernel | llama-13b-prefill-float32 | 79.01 | **67.28** | PASS | pass（**−14.8%**） |
| FusedAddLayerNormFwdOp.py（R8） | _fused_add_layer_norm_kernel | llama-8b-decode-bfloat16 | 4.19 | 4.16 | PASS | pass（微型豁免） |
| FusedAddLayerNormFwdOp.py（R8） | _fused_add_layer_norm_kernel | llama-70b-decode-bfloat16 | 4.21 | 4.17 | PASS | pass（微型豁免） |

smoke（3 例，仅精度）：`--level all` 全过（见「精度回归」）。

### 5.6 R8 Iteration Winner

- winner：`perf_opt/FusedAddLayerNormFwdOp.py` = R7 final + op1（fp32 N=5120 默认 `block_m=2`）+ 预算模型重标定，sha256 `91121fa3…`。
- reason：唯一 valid 分支 op1 在 13b 上 −13.1~−13.7%（2 rep 同向）、7/7 tune workload 无可信回退、smoke 精度通过；
  8b/70b 行批处理已用后端容量原文判 blocked（5.3.1）。
- rollback_branches：`FusedAddLayerNormFwdOp_r8_bm3_mb1_blocked.py`、`FusedAddLayerNormFwdOp_r8_bm3_mb0.py`、`FusedAddLayerNormFwdOp_r8_bm4_mb0.py`
  （编译即 UB 拒绝）、`FusedAddLayerNormFwdOp_r8_fp32bm2.py`（op1 原件，已合入 final）。均保留作证据，不参与 wrapper 采纳。

### 5.7 R8 未竟项 / 下一步（明确以 UB 账为界）

- 8b/70b 行批处理在本工具链 + 本结构族下一条可行路径是**换 residency 结构**（例如把 d² 的 N 方向分块规约，使 `sq_f32` 从 (bm,N) 降为 (bm,N/2)），
  可把 bm=3 的 bf16 路径压到 ~140 KB；但需要 ≥2 处新 pass 且数值上要复核（分块 sum 顺序变化），本轮预算内未启动。
- 13b 若还能再降，需要的是 GM 带宽侧（已 70.8% SOL），而非 tiling。

## 6. R9 深挖（2026-10-22）：N 方向 d² 分块 → bm=3/4

### 6.1 一句话结论

**未达成，保留 R8（sha256 `91121fa3…`）**。目标（用 N 方向 d² 分块腾 UB 让 `block_m=3/4` 在 8b prefill 落地）
卡在**后端 UB 字节墙**与**本工具链 v-op 视图语义**两处，不是参数可解：
bm=3 实测需 **204,832 B（auto-mb OFF）/ 229,408 B（auto-mb ON）**，可用 **196,608 B**，缺口 **8,224 B / 32,800 B**；
唯一能补这 8,224 B 的两条路（关 auto-mb、把 d² 平面降为 `(bm, N/2)`）分别被 **+28.5% 回退** 与 **编译器形状检查** 否决（实测见 6.3/6.4）。

### 6.2 先量化收益：bm 曲线探针（不改 kernel）

`probe_r9_ub/run_bm_curve.sh`（R8 源码原样，`bench_shape.py` 直跑），取 **M=2048 / N=2048 / fp16** ——
选它是因该 shape 下 `bm=3/4` 仍在 UB 内，能在真实 kernel 上读出**行批处理的收益上限**（`profiles/probe-bmcurve-*`）：

| block_m | Task Duration(µs) | vs bm=2 | 说明 |
|---:|---:|---:|---|
| 2 | 44.400 | — | 1024 logical / 40 核 = 25.6 task/core |
| 3 | 37.850 | **−14.8%** | 683 / 40 = 17.1 task/core |
| 4 | 33.840 | **−23.8%** | 512 / 40 = 12.8 task/core |

- 按 `T = tasks·(a + b·bm)` 拟合（N=2048）：每 task 固定开销 **a≈0.90 µs**、每行成本 **b≈0.42 µs**；
  换到 N=4096（每行数据翻倍，b≈0.8）预测 bm=3 为 **−7~9%**，叠加 bm=3 的 wave 不齐（683 = 40×17+3，1/18 ≈ 5.6%）约 **−5~8%**。
  即：**行批处理确实是真杠杆（探针上限 15–24%），值得为它做 UB 账**；本轮失败点是容量，不是收益。
- 该探针只用于收益标定，N=2048 不在 tune 集合中，不进 `perf_records.jsonl`（原始数据 `probe_r9_ub/profiles/`、`probe_r9_ub/logs/`）。

### 6.3 UB 字节账（本轮：backend `requires ... bits` 原文，capacity 196,608 B）

`probe_r9_ub/probe_r9_build.py` 绕过 host guard 直接问后端（`FusedAddLayerNormFwdOp_r9_stage2.py` = op2 分支）：

| 布局 | bm | auto-mb | 后端自报 | 字节 | 判定 |
|---|---:|---|---:|---:|---|
| R8 copy-cast（现行） | 3 | ON | 1,835,264 bits | **229,408** | 缺口 32,800 |
| R8 copy-cast | 3 | OFF | 1,638,656 bits | **204,832** | 缺口 8,224 |
| op2 显式 staging（2 个 dtype 平面，去掉 copy-cast 隐式 scratch） | 3 | ON | 1,835,264 bits | **229,408** | 与 R8 完全相同 |
| op2 显式 staging | 3 | OFF | 1,638,656 bits | **204,832** | 与 R8 完全相同 |
| R8 copy-cast | 2 | ON | — | 163,840 | 编译通过（现行默认） |

- **关键否定证据**：显式 staging 并**没有**消掉那 4 B/elem 的隐式 scratch ——
  名义 14.67 B/elem（180,224 B）实测 16.67 B/elem（204,832 B）；即「自己拥有 copy-cast 的落地平面」这条 R8 §5.7 设想的省字节路径**实测无效**。
- 由此 bm=3 的**地板**就是 `stage(2) + x_f32(4) + sq_f32(4) + w/b(2.67) + 隐式 scratch(4) ≈ 16.67 B/elem = 204,832 B`；
  要到 196,608 B 必须再降 4 B/elem（= 一个 (bm,N) fp32 平面或等价物）。

### 6.4 为什么 "sq_f32 降为 (bm,N/2)" 在本工具链不可表达（实测）

`probe_r9_ub/colslice_probe2.py`、`colslice_probe3.py`（每个 primitive 单独编译 + 数值校验）：

| primitive | 形态 | 结果 |
|---|---|---|
| `vmul` | 列半切片 src + 全宽 `(bm,N/2)` dst | **OK**（数值正确 `max_diff<2e-3`） |
| `reduce_sum` | `(bm,N/2)` fp32 平面 → `(bm,1)`，两次 + `vadd` 合并 | **OK**（`max_diff 4.9e-04`） |
| `vbrc` | `(1,N/2)` fp32 → `(bm,N/2)` fp32 | **OK** |
| `vadd/vmul` | **列半切片做 in-place dst**（`x[0:bm,0:H] += sh`） | **编译失败** `Check failed: buffer_shape0[i] == buffer_shape1[i]`（codegen 用整 buffer 形状） |
| `vadd/vmul` | **行切片做 dst**（`x[r:r+1,0:N] *= w1`，即 per-row affine，免 `vbrc`） | **编译失败** 同上（`rowslice_probe.py`） |
| `vcast` | 列半切片 src → `(bm,N/2)` dst | **编译失败** 同上 |
| `vbrc` | dtype src → fp32 dst（省掉 w/b 的 fp32 平面） | **编译失败** `'hivm.hir.vbrc' op requires the same element type for all operands` |

→ 一般规则（本轮 8 种形态枚举得出）：**v-op 的 dst 必须是整 buffer**（不能是 region），src 可以是 region；
`vcast` 更严，连 src 的 region 形状都不认；`vbrc` 要求同 dtype。

推论（字节账的硬约束）：
- 残差加法必须有**两个全宽 fp32 平面**（`x_f32` + `sq_f32`）：`vcast` 不能把 dtype 平面写进半宽 fp32，
  而 `vadd` 不能 in-place 写列半切片。所以 `sq` 一旦降为 `(bm,N/2)`，第二步 `s = x + res` 就无合法落地面；
- d²/affine **读**列半切片是合法的（`vmul`/`reduce` 已验证），但 affine 写回必须落到全宽平面 →
  要么再要一个 `(bm,N/2)` fp32 平面（+2 B/elem，等于没省），要么 in-place 列 dst（不允许）；
- w/b 压成 dtype 也不可行（`vbrc` 不支持跨 dtype），只能换成「dtype 平面 + 共用 `(1,N)` fp32 scratch」，
  净省 8,192 B ⇒ 196,640 B，比容量**多 256 bits**，仍被拒。

### 6.5 候选优化点与分支

| opt_id | 目标现象 | 优化点 | 结果 | 证据 |
|---|---|---|---|---|
| op1 `sqhalf` | d² 平面 (bm,N) 占 49 KB，是 bm=3 的主要缺口 | d²/affine 沿 N 分两半用 `(bm,N/2)` | **blocked（不可表达）** | 6.4 primitive 表（`vcast`/in-place 列 dst 编译失败） |
| op2 `stage2` | copy-cast 隐式 scratch（4 B/elem） | 显式 2 个 dtype 平面承担落地 | **blocked（字节不变）** | 6.3 两行 229,408/204,832 与 R8 全同 |
| op3 `mb0` | auto-mb 使 dtype 平面双缓冲（+2 B/elem） | 全局关 `npuir.enable_auto_multi_buffer` | **invalid（13b +28.5%）** | 6.6 A/B 表 |
| op4 `wbdtype` | w/b 两个 `(1,N)` fp32 = 32,768 B | w/b 留 dtype，`vbrc` 内联 cast | **blocked（编译拒绝）** | `vbrc requires the same element type` |
| op5 `two-pass-half` | 两遍半宽最省 UB（bm=4 可容） | pass1 半宽算 stats，pass2 重读 `residual_out` 算 y | **纸面否决（未起分支）** | 流量 8 → 12 B/elem（**+50%**）；向量 pass 13 → ~16 全宽当量（**+45%**），全宽当量账见 6.4 推导；对 AI 3.0 的算力受限 shape 净收益不可信 |
| op6 `rowaffine` | affine 的 2 个 `vbrc` + 2 个全宽 arith（4 pass） | 逐行 affine（`(1,N)` 行切片直接乘加，免 `vbrc`，理论 2 全宽当量） | **blocked（不可表达）** | 行切片做 dst 编译失败（6.4 表第 5 行，`rowslice_probe.py`） |

### 6.6 候选 vs current best 对比表（R9，同会话、逐 workload 交错，msprof launch-count=20）

AICore 列见 R7/R8 画像（本轮未重采 PipeUtilization，Task Duration 是唯一裁决口径）。

| candidate | workload | Task Duration(us) | vs current best | SOL% | L0 |
|---|---|---:|---:|---:|---|
| current best（R8 `r9cur`） | llama-8b-prefill-float16 | **62.87** | — | 32.76 | PASS |
| op3 mb0 | llama-8b-prefill-float16 | 62.90 | +0.05%（tie） | 33.30 | PASS |
| current best | llama-8b-prefill-bfloat16 | **64.42** | — | 38.01 | PASS |
| op3 mb0 | llama-8b-prefill-bfloat16 | 65.06 | +1.0%（tie） | 37.52 | PASS |
| current best | llama-70b-prefill-float16 | **116.71** | — | 35.32 | PASS |
| op3 mb0 | llama-70b-prefill-float16 | 116.94 | +0.2%（tie） | 35.25 | PASS |
| current best | llama-70b-prefill-bfloat16 | **120.72** | — | 40.17 | PASS |
| op3 mb0 | llama-70b-prefill-bfloat16 | 120.40 | −0.3%（tie） | 40.26 | PASS |
| current best | llama-13b-prefill-float32 | **67.44** | — | 71.97 | PASS |
| op3 mb0 | llama-13b-prefill-float32 | **86.65** | **+28.5%（回退）** | 55.16 | PASS |
| current best | llama-8b-decode-bfloat16 | 4.37 | — | 0.49 | PASS |
| op3 mb0 | llama-8b-decode-bfloat16 | 4.20 | −3.9%（微型豁免） | 0.52 | PASS |
| current best | llama-70b-decode-bfloat16 | 4.34 | — | 0.90 | PASS |
| op3 mb0 | llama-70b-decode-bfloat16 | 4.18 | −3.7%（微型豁免） | 0.84 | PASS |

判定：
- **op3 mb0 ⇒ invalid**：13b fp32 `+28.5%`（67.44 → 86.65 µs，SOL 71.97% → 55.16%），远超 3% 噪声阈值，
  且该 flag 是 kernel 级全局（`_JIT_KWARGS`），不能按 shape 分派 ⇒ 不可采纳。
  这同时是**新的 D 类实测数据**：本 kernel fp32 N=5120/bm=2 路径的 `auto-multi-buffer` 值 **28.5%**（R8 当时只是"不想动全局"，本轮有了量化代价）。
- **8b/70b prefill 在 mb0 下与 R8 打平（±1%）** ⇒ 关 auto-mb 对 half 路径本身无害，但它只省 24,576 B，
  仍不足以让 bm=3 落地（204,832 > 196,608），收益为零、代价确定 ⇒ 放弃。
- op1/op2/op4 为编译/字节 blocked，op5 为纸面否决，均无 valid 时延，不进 `perf_records.jsonl`；
  已进记录的 14 行（round 9：`r9cur` ×7、`r9mb0` ×7，`phase=candidate`）为本轮全部有效测量。

### 6.7 R9 Winner / rollback

- **winner：无**。current best 保持 `perf_opt/FusedAddLayerNormFwdOp.py`（R8 merged，sha256 `91121fa392fcbf2db896394eb3e7c3654966e6ca1cde9bff250602a5745d3a45`）。
- rollback/保留证据：`FusedAddLayerNormFwdOp_r9_mboff.py`（op3 原件，已测）、`FusedAddLayerNormFwdOp_r9_stage2.py`（op2 原件，bm=3 被 UB 拒）；op1/op4/op5 无文件（不可表达/未起分支）。
- 最终交付态未改一行代码 ⇒ R8 的 7 条 `phase=final`（`r8final`）继续有效；本轮同会话对照 `r9cur` 7 点复现 R8 数值（62.87/64.42/116.71/120.72/67.44/4.37/4.34 vs R8 final 62.94/64.48/117.10/120.44/67.28/4.16/4.17，全部在 ±3% 内）。

### 6.8 R9 未竟项：跨过这道墙需要什么（给设计层/编译器层的明确清单）

1. **bm=3 的 8,224 B 缺口**只能从「少一个全宽 fp32 平面」来。要么 v-op 支持
   ①`vcast` 的区域形状语义（列半切片 → 半宽 dst），要么 ②列半切片可做 in-place dst，
   要么 ③允许 `(bm,N)` fp32 平面按 `(bm,N/2)` 逻辑视图复用。任一条成立即可让 `sq` 降半宽（省 24,576 B，bm=3 = 180,256 B）。
2. 若 ③ 不可得，退路是**把 w/b 压到 dtype + `vbrc` 支持 cast**（省 16,384 B ⇒ bm=3 = 188,448 B，落在容量内 8 KB 余量），
   当前被 `vbrc requires the same element type for all operands` 挡住。
3. bm=4（探针 −23.8% 的那一档）在 N=4096 需要 ≤ 12 B/elem，必须同时做到 (1)+(2)，属**设计级新立项**，
   不是本轮参数可解。
4. 能力缺口已按 AGENTS.md 记入 `.agents/evolution/capability-gaps.md`（CG 条目：v-op 区域形状语义 + `vbrc` 跨 dtype）。

## Final Performance Test Data

> **R9 收束说明（2026-10-22）**：R9 未产出 winner（N 方向 d² 分块在本工具链不可表达，bm=3 需 204,832 B > 196,608 B，
> 详见 §6）。交付态**未改一行**，仍为 R8 merged（sha256 `91121fa3…`），故下表 7 条 `phase=final`（`r8final`）继续有效；
> R9 同会话对照 `r9cur`（同一 artifact、`phase=candidate`、round 9）复测 7/7 workload 与下表偏差 ≤3%，作为"未回退"证据。

官方最终测量：`perf_opt/FusedAddLayerNormFwdOp.py`（R8，sha256 `91121fa3…`）。
测量方式：`run_experiments.py --experiments experiments_r8_final.json --all-tune --order workload --round 8 --phase final`，同一次 NPU 会话内逐 workload 相邻交错「Stage-3 基线 → 最终候选」，每点 `msprof op --launch-count=20 --warm-up=5`，取 20 次 Task Duration 中位数。

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| _fused_add_layer_norm_kernel | llama-13b-prefill-float32 | 107.360 | 67.280 | r8final |
| _fused_add_layer_norm_kernel | llama-70b-decode-bfloat16 | 4.650 | 4.170 | r8final |
| _fused_add_layer_norm_kernel | llama-70b-prefill-bfloat16 | 168.760 | 120.440 | r8final |
| _fused_add_layer_norm_kernel | llama-70b-prefill-float16 | 165.370 | 117.100 | r8final |
| _fused_add_layer_norm_kernel | llama-8b-decode-bfloat16 | 4.240 | 4.160 | r8final |
| _fused_add_layer_norm_kernel | llama-8b-prefill-bfloat16 | 84.260 | 64.480 | r8final |
| _fused_add_layer_norm_kernel | llama-8b-prefill-float16 | 82.360 | 62.940 | r8final |

> 簿记：R7 的 7 条 `phase=final` 在 R8 收束时降级为 `phase=merged`（仅 phase 标签，round / candidate_id / duration_us / artifact_sha256 / msprof_raw_path 均未改动），
> 与 R6→R7 处理一致。降级前快照 `/tmp/opencode/perf_records_before_r8_final.jsonl`（115 行）。

### SOL 对账（R8）

| workload | base µs | final µs | Δ vs base | vs R7 best | base SOL% | final SOL% | 判据/线 | 是否达标 |
|---|---|---:|---:|---:|---:|---:|---|---|
| llama-13b-prefill-float32 | 107.360 | 67.280 | −37.33% | **−14.85%** | 44.65 | **70.82** | 访存 70% | **是（首次达标）** |
| llama-70b-decode-bfloat16 | 4.650 | 4.170 | −10.32% | −0.95% | 0.82 | 0.87 | 微型豁免 | 豁免 |
| llama-70b-prefill-bfloat16 | 168.760 | 120.440 | −28.63% | +0.03% | 32.70 | 40.43 | 算力 60% | 否 |
| llama-70b-prefill-float16 | 165.370 | 117.100 | −29.19% | +0.27% | 29.08 | 35.01 | 算力 60% | 否 |
| llama-8b-decode-bfloat16 | 4.240 | 4.160 | −1.89% | −0.72% | 0.50 | 0.52 | 微型豁免 | 豁免 |
| llama-8b-prefill-bfloat16 | 84.260 | 64.480 | −23.47% | +0.08% | 31.09 | 37.88 | 算力 60% | 否 |
| llama-8b-prefill-float16 | 82.360 | 62.940 | −23.58% | +0.46% | 27.17 | 32.99 | 算力 60% | 否 |

**逐 workload 结论（如实）**：

- 本轮目标（8b prefill 再降 5–10%）**未达成**：8b 双 dtype 与 R7 差 +0.1~+0.5%（噪声内），行批处理被后端 UB 拒绝（5.3.1 有精确字节账）。
- 收获（13b）：`block_m=2` 让 13b fp32 到 **67.28µs / SOL 70.82%**，首次跨过访存 70% 线；相对 R7 best −14.85%，相对 Stage-3 −37.33%。
- 硬底线（Task Duration 必须变短）：**7/7 不劣于 R7 best**（4 个持平在噪声内、1 个显著下降、2 个 decode 微型豁免且变短）。
- 达标数：全 shape **1/7**（13b；decode 豁免不计）；可调 prefill shape **1/3**（13b 达标，8b/70b 仍在算力线 60% 之下）。
- 13b 的 pipe 画像仍是访存侧（R7 记录 vec 0.599 / mte2 0.467，贴 2.1 TB/s）；bm=2 后 SOL 70.8% 说明已基本吃满 fabric。

## 精度回归

`python perf_opt/FusedAddLayerNormFwdOp.py --level all` → **`All check passed!`**（L0 13/13、L1 全过；L2/Boundary WARN-only，`Boundary-extreme-affine` 为 weight/bias=±65504 的 fp16 溢出 WARN，与 R6 同源、非本轮引入）。

- 数值契约未改：仍是 round-first stats（先 `round_dtype(s)` 再进 fp32 统计），fp32 opmath、单次 round-back。
- L0 max_diff（R8 合并版，`logs/r8/level_all_r8final.log`）：bf16 3.125e-02、fp16 1.000e+00（仅 extreme-affine 的 ±65504 溢出例触发 2/16384 WARN，非阻塞）、fp32 2.861e-06；与 R7 同量级。
- R8 op1 的正确性依据：`block_m` 只是行分块参数，所有 v-op 沿 dim=1 规约 / 行广播，行间完全独立；`bm=1→2` 不改变任何计算式与 round 顺序（round-first stats + fp32 opmath + 单次 round-back 未动）。
- L1 契约守卫同步更新：UB 守卫改为按 `_ub_footprint_bytes` 判（`bm=3/N=5120/fp32`、`bm=3/N=4096/fp16`、`bm=2/N=8192/fp16` 三例必须被拒），`_default_block_m` 走 `_TUNED_BLOCK_M` + Stage-3 回退表，未 tuned 的 shape 默认值不变。

## Skill Retrospective

R7 新增（其余保留 R6 结论）：

- **D（实测数据）**：`block_m==1`（N=8192）的 loads-first 不必新增 fp32 平面——`block_m==1` 时 resident `stage` 与 `wstage` 同形，复用 `wstage` 作第二 dtype 平面即可让两条 MTE2 前置，**UB 零增量**，70b prefill −16.8~−17.0%。
  - 证据：`perf_records.jsonl` round 7（r7cur vs r7op1_pinput_stage，两 rep）、`profiles/candidate-r7-*`、`profiles/r7diag-final-llama-70b-prefill-float16/`。
- **D（实测数据）**：fp32 直通路径的 weight/bias 若留在 persistent 循环内，每个 task 多 2×N×4B 的 GM 读；N=5120、bm=1 时 payload 与因子流同量级（+50% 流量），外提后 13b fp32 108.5→79.0µs（−27%），pipe 画像 vec 0.60 / mte2 0.47。
  - 证据：`perf_records.jsonl` round 7（r7cur vs r7op2_fp32hoist）、`profiles/r7diag-final-llama-13b-prefill-float32/`。
- **R（流程规则）**：修「R3 已落地的优化点」时要逐 dtype 路径核对覆盖范围——A10（affine 外提）当时只改了两条 transit 分支，fp32 分支漏改，导致 13b fp32 三个轮次（R3/R5/R6）一直在多读 33% GM 流量而未被发现；本轮由「逐分支读代码 + 按 dtype 路径列流量账」发现。
  - 证据：R3/R5/R6 记录中 13b fp32 恒为 107.6~108.7µs（三轮未动），R7 外提后 79.0µs。
- **P（模式方法，反例）**：**向量链长度不是 pass 数的线性函数**——去掉 2/12 个全平面 pass（`vbrc` 外提）Task Duration 不动（4 个组合、2 rep 方向不一致）；同族预测式「vec_time ∝ pass 数」在依赖链主导时失效，诊断必须以 `aiv_vec_ratio` + 每 task 依赖链长度为准。
  - 证据：`perf_records.jsonl` round 7 r7op5_hoistbrc vs r7merged；`profiles/r7diag-cur-8b-fp16/`（vec 0.756、aiv_time 60.0 vs Task Duration 62.7）。
- **P（模式方法，新增形态）**：`T.Pipelined(num_stages=2)` 在「persistent 串行 task 循环 + 循环体内含单比较 if 守卫」形态下**可编译、精度通过**（isnan 先例 + 本轮两路径），但本算子 **tie**——auto-multi-buffer 已把该项收益吃掉；该形态对 CG-2026-0001(a) 的旧结论构成新证据（旧戳 2026-09-04 / CANN 26.0.rc1），需按版本戳重验后才可更新该条目。
  - 证据：`perf_records.jsonl` round 7 r7op3_pipe2（4 组合）+ `logs/r7/l0_FusedAddLayerNormFwdOp_r7_pipe2.log`。
- 证据三件套落点：perf_records.jsonl（D）、profiles/r7diag-*（D）、`FusedAddLayerNormFwdOp_r7_*.py`（P 的最小 delta：op1 = 两个 MTE2 前置 + `wstage` 复用；op2 = 两次 `T.copy` 移出循环）。

R8 新增（其余保留 R6/R7 结论）：

- **D（实测数据）**：本 kernel 的 UB 驻留成本不是「20 B/elem × 元素数」，而是逐平面清单：`fp16/bf16 bm≥2 = 10 B/elem + 8N`，
  且后端 auto-multi-buffer 再叠 **+6 B/elem**（= `x_f32` + `stage` 各一份双缓冲）。三点标定：bm=3/N=4096 后端要 229,408 B（模型 229,376 B，误差 32 B）、
  关掉 auto-mb 后 204,832 B、bm=4 关掉后 262,176 B；容量 196,608 B。
  - 证据：`perf_opt/probe_r8_ub/compile_probe.py` + `logs/r8/`（后端 `requires ... bits` 原文）、`FusedAddLayerNormFwdOp_r8_bm3_mb1_blocked.py`。
- **D（实测数据）**：fp32 路径 N=5120 从 `bm=1` 抬到 `bm=2`（新模型判定 184,320 B ≤ 196,608 B，可行）→ 13b prefill **78.74/78.40 → 68.39/67.67µs（−13.1%/−13.7%）**，
  SOL 60.4% → 68.5~70.1%，**首次跨过访存 70% 线**（最终 67.28µs / 70.82%）。行批处理在访存受限的 shape 上主要省的是每 tile 的 MTE 起停与 reduce 摊薄。
  - 证据：`perf_records.jsonl` round 8（r8cur vs r8op1_fp32bm2，2 rep）+ `profiles/candidate-r8-*`、`profiles/final-r8-r8final-*`。
- **P（模式方法，反例）**：**行批处理（降低 reduce µs/row）不是普适杠杆**——它在探针（单核、纯 reduce）上 −29%/行，但落到整 kernel 时被 UB 容量挡死：
  bm=3 需 229,408 B（auto-mb ON）/204,832 B（OFF）vs 196,608 B。把「探针上的 µs/row 收益」外推到整 kernel 前，必须先算**整 kernel 的驻留账**（含后端双缓冲膨胀项），
  否则会得到「理论 5–10%、实际不可编译」的空跑结论。
  - 证据：5.2/5.3.1 的四个容量数据点 + `probe_cube_gemv` §3 的 bm≥8 UB 限（同一机制）。
- **R（流程规则）**：跑本 kernel 的任何 NPU 命令（L0/msprof/runner）必须显式 `export TILELANG_ASCEND_MODE=Developer`——
  `tileops_env.sh` 导出 `Expert`，而 `run_experiments.py` / kernel 自身都用 `os.environ.setdefault`，不会覆盖已存在的 Expert，
  结果是未改动的 current best 也 13/13 L0 全挂（`'hivm.hir.store' op only support store ub to gm currently!`），极易被误判成 kernel 回归而回滚有效改动。
  - 证据：`TILELANG_ASCEND_MODE=Expert` 下 `FusedAddLayerNormFwdOp.py --level L0` 13 失败 vs Developer 下全过（`logs/l0_FusedAddLayerNormFwdOp.log` 与 `logs/r8/level_all_r8final.log` 对照）。
- **C（复现资产）**：`perf_opt/probe_r8_ub/compile_probe.py`（一条命令给出「模型预测 footprint vs 后端自报 requires」），可作为后续 UB 预算类候选的**先验闸门**，
  避免把编译期必然失败的 shape 带进 msprof 轮次。
- 证据三件套落点：perf_records.jsonl round 8（D）、`profiles/r8-*`（D）、`FusedAddLayerNormFwdOp_r8_fp32bm2.py`（P 的最小 delta：仅 `_TUNED_BLOCK_M` 一行 + 预算模型函数）。

R9 新增（其余保留 R6/R7/R8 结论）：

- **D（实测数据）**：行批处理的收益上限在**真实 kernel**上是 15–24%（不是 3–5%）：R8 源码原样、M=2048/N=2048/fp16，
  `bm=2/3/4 = 44.400 / 37.850 / 33.840 µs`（−14.8% / −23.8%）。拟合出每 task 固定开销 `a≈0.90 µs`、每行 `b≈0.42 µs`，
  可直接外推任意 (bm, N) 的收益，用于判定"值不值得为它做 UB 账"。
  - 证据：`probe_r9_ub/run_bm_curve.sh` + `probe_r9_ub/profiles/probe-bmcurve-N2048-float16-bm{2,3,4}/`、`probe_r9_ub/logs/logs_bmcurve_*`。
- **D（实测数据）**：本 kernel 的 `npuir.enable_auto_multi_buffer` 在 **fp32 N=5120 bm=2** 路径上值 **28.5%**
  （67.44 → 86.65 µs、SOL 71.97 → 55.16%）；对 half 路径（8b/70b prefill）**≤±1%**。即 R8 的"不动全局 flag"结论本轮有了量化代价，
  该 flag 无法按 shape 分派（`_JIT_KWARGS` 是模块级），故永久不可用于"只给 8b 省字节"。
  - 证据：`perf_records.jsonl` round 9（`r9cur` vs `r9mb0`，7 workload 交错）+ `profiles/candidate-r9-*`。
- **D（实测数据）**：**显式 staging 不能替代 copy-cast 的隐式 scratch**——把 GM→fp32 的 copy-cast 改成
  「GM→dtype staging + 显式 vcast」并多分配一个 `(bm,N)` dtype 平面后，后端字节数**一分未变**
  （bm=3：229,408 B / mb OFF 204,832 B，与 R8 完全相同）。R8 §5.7 的"自己拥有落地平面可省 4 B/elem"假设被实测证伪。
  - 证据：`probe_r9_ub/probe_r9_build.py FusedAddLayerNormFwdOp_r9_stage2.py 2048 4096 float16 3 {0,1}`。
- **P（模式方法，反例）**：**"把 scratch 平面沿 N 减半"不是一个纯布局改动**——TileLang-NPUIR 的 v-op 形状推导按**整 buffer**
  取 shape（`buffer_shape0[i] == buffer_shape1[i]`），因此 ①`vcast` 不能把列半切片写成半宽 dst，
  ②`vadd/vmul` 不能 in-place 写列半切片；只有「列半切片做 src + 全宽平面做 dst」与「整半宽平面 reduce」合法。
  结论：只要残差加法需要第二个全宽 fp32 平面，`sq` 就压不下来。做此类字节优化前，**先用 4 个 primitive 微探针确认视图语义**，
  否则字节账会算出一个写不出来的布局（本轮省了 ≥2 小时空跑）。
  - 证据：`probe_r9_ub/colslice_probe2.py`（mul/reduce/vbrc OK；vcast 半宽 FAIL）、`colslice_probe3.py`（in-place 列 dst FAIL，半宽 reduce 数值 OK）。
- **R（流程规则）**：判定"某 shape 装不下 UB"之前，先做**收益标定探针**（在一个装得下的相邻 shape 上量 bm 曲线），
  再决定是否为它付字节改造的代价——本轮若先做 6.2 的曲线探针，可以在 20 分钟内得出"缺口 8,224 B、收益 5–8%"，避免先写两版 kernel。
  - 证据：6.2 表（3 个数据点）+ 6.3 表（4 个后端字节数）。
- **C（复现资产）**：`probe_r9_ub/bench_shape.py` + `run_bm_curve.sh`（任意 M/N/dtype/bm 的 msprof 单点），
  与 `probe_r8_ub/compile_probe.py`（字节）配对，构成"先算收益、再算字节"的前置闸门。
- 证据三件套落点：perf_records.jsonl round 9（D）、`profiles/candidate-r9-*`（D）、`probe_r9_ub/*`（P/C 的最小可复现脚本）。
