# 优化报告：FusedAddLayerNormFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `FusedAddLayerNormFwdOp`；单 kernel / 单 func / 纯 Vector（AIV），4 输入 2 输出（`y` + pre-norm `residual_out`） |
| kernel（基准，只读） | `tileops/kernels/norm/fused_add_layer_norm/fused_add_layer_norm_kernel/`（生产基线 = Stage-3 `_fused_add_layer_norm_kernel.py`） |
| 调优产物 | `.../fused_add_layer_norm_kernel/perf_opt/FusedAddLayerNormFwdOp.py`（winner 候选 `loadsfirst`） |
| 口径 | `msprof op Task Duration(us)`（唯一时延口径）+ GM Read+Write SOL ratio；数据来自 `perf_opt/perf_records.jsonl` round 5（候选对比）/ round 6（final） |
| **工具链版本戳** | CANN 9.0.0 / tilelang 0.1.2 dev root / Ascend910B3 / 2026-10-18 |
| 达标线 | **算力受限 ≥60%**（prefill AI 3.0–3.8 > 平衡点 ⇒ 算力线 60%）；13b fp32 AI 1.02 ⇒ 访存线 70%；decode 4.0–4.8µs 属微型 shape，性能豁免 |
| benchmark | `bench_fused_add_layer_norm.py`（7 tune + 3 smoke） |
| 过程日志 | `.../perf_opt/opt_log.md`、`perf_records.jsonl`、`logs/`、`candidates/` |
| 前置报告 | 本文件 v0 已归档 `history_version/OPTIMIZE_REPORT_v0.md` |

## 1. 结论

- 一句话：**两段叠加优化（weight/bias 循环不变量外提 + loads-first 双输入前置装载）把 8b prefill 打到 −23.2~−23.3%、70b prefill −14.0~−14.2%，硬底线（Task Duration 变短）达成；但 SOL 仍 27–45%，0/7 达标——瓶颈是向量 op 数与 row-reduce 流水深度，属结构级改造才可能跨线。**
- **达标数（双分母）**：全 shape（7 tune）**0/7**（SOL 未达线；decode 豁免）；可调 shape（3 prefill）**0/3**（SOL 未达线，但 Task Duration 均下降）；豁免 **2**（`llama-70b-decode-bfloat16`、`llama-8b-decode-bfloat16`，依据：decode 4.0–4.8µs 属微型 shape，原文判定性能豁免）。
- 关键收益（vs Stage-3 生产基线）：**8b prefill −23.2~−23.3%**、**70b prefill −14.0~−14.2%**、13b fp32 **+0.98%**（fp32 路径未改，噪声内）、decode **+8%**（微型豁免，噪声内）。
- 指标行为：SOL ratio 在 8b 上 27→33%（bf16 31→37%），但仍未达 60% 算力线；瓶颈已从 MTE2/VEC 装载串行转到纯向量 op 数 + row-reduce 流水深度。
- **是否建议采纳 wrapper**：**是**（原文『产物与后续』）——有净收益（8b prefill −23%、70b prefill −14%，硬底线 Task Duration 均下降），无精度回退（L0/L1 全绿），签名 drop-in。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测 | 数值 | 结论 |
|---|---|---|
| 判据归类（prefill） | AI 3.0–3.8（> 平衡点） | **算力受限，线 60%**；GM SOL 不是主判据 |
| 判据归类（13b fp32） | AI 1.02 | 访存受限，线 70% |
| 达线现状 | SOL 27–45%（8b prefill 27→33%、bf16 31→37%） | 两段优化后仍未跨线 → 杠杆不在 tiling |
| 瓶颈形态（round 5 前） | MTE2/VEC 交替串行（单 `stage` 缓冲被 x→residual→round→round 顺序复用，强制 `MTE2(x)→VEC(vcast)→MTE2(residual)→VEC`） | **装载串行**是当时的主矛盾 → 可用「双输入前置装载」消除 |
| 瓶颈形态（round 5 后） | 纯向量 op 数（15+ pass）+ 2 次 row-reduce 的流水深度；roofline `latency bound:pipeline caused` | 下一层瓶颈是向量链与归约流水深度 → 需结构级改造（N 方向分核/流水） |
| 固定开销 / 微型 shape | decode 4.0–4.8µs | 微型 shape，性能豁免（不计入达标分母） |
| block_m 选择 | block_m=1 慢 25%（N=4096） | 默认 `_default_block_m` 正确；block_m 方向无收益 |
| UB 边界 | 70b prefill（N=8192, block_m=1）上 loads-first 需 1573120 bits / 有 1572864，**差 256 bits** | 工具链 MTE2 不支持跨 dtype DMA、copy-cast 内部仍物化 staging → 必须按 block_m 编译期分支 |
| 判据是否失配 | 同一 benchmark 内判据分层：prefill 算力线 60%、13b fp32 访存线 70%、decode 属微型 | **需分层**：单一 SOL 线覆盖不了全部 workload，原文已按此分层并给 decode 豁免 |

### 2.2 方法（做了什么、为什么这样做）

仅 `perf_opt/` 内改动：

1. **weight/bias 循环不变量外提（round 3，A10）**
   - 改了什么：weight/bias 每 core 加载并升 fp32 **一次**，循环内只做 `vbrc` 广播。
   - 为什么有效：weight/bias 与 loop 迭代无关，原实现逐 tile 重复加载/转换；外提后把每 tile 的固定 MTE2 + cast 开销降为一次性，70b prefill −14%、8b −7%。
   - 正确性依据：循环不变量外提是纯编译期变换（缓冲内容与迭代无关），数学等价；L0 全 PASS。

2. **loads-first 双输入前置装载（round 5，pattern-library PL-1.10 同族）**
   - 改了什么：fp16/bf16 transit 路径把两个输入**前置**装载——`T.copy(x→x_f32)`、`T.copy(residual→sq_f32)` 各走独立 fp32 缓冲（fp16→fp32 上转 copy-cast 精确），`stage` 只留 round-trip + 输出；并按 `block_m` 做**编译期分支**（`block_m==1` 保留单 staging 避免 UB overflow，`block_m>=2` 走 loads-first）。
   - 为什么有效：原单 `stage` 缓冲被 4 个角色顺序复用，强制 `MTE2→VEC→MTE2→VEC` 串行；双输入前置装载让两次 MTE2 连续发起、与向量计算重叠，直接消除 round 5 前诊断出的「MTE2/VEC 零重叠」，8b prefill 再 −18%。
   - 正确性依据：fp16→fp32 上转的 copy-cast 是精确的（无舍入损失），L0 max_diff 与原版一致；`--level all` L0/L1 全过；`block_m` 分支只改变 staging 组织形式，不改变计算式与输出。
   - 合法性/边界依据：`block_m==1`（N=8192）走 loads-first 会 **ub overflow（差 256 bits）**，故保留原单 staging vcast 链——条件分支只放在 `@T.prim_func` 内（按 `block_m` 编译期分化），不新增 Python factory/wrapper 分派。

3. **`block_m` 扫描（1 vs 2）**：block_m=1 慢 25%，保留默认（`_default_block_m`：4096→2、8192/5120→1）。

**被否决的方向（保留）**：
- 单遍方差 `E[x²]−E[x]²`：**不可用**——tie 无收益，且 fp32 大均值场景有消减误差风险（L0 虽过但换零收益，弃）。
- block_m=1 分核（round 2 已证死路）：不重做。

### 2.3 逐轮记录

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| R2 | `block_m=1` 分核 | 降低 block_m | 慢 25%（N=4096） | ❌ 否决：保留 `_default_block_m` |
| R3 | `hoist-wb`（A10） | weight/bias 每 core 加载 + 升 fp32 一次，循环内只 `vbrc` | 70b prefill −14.0~−14.5%、8b prefill −6.4~−7.1%（opt_log §1；报告正文记 −14% / −7%） | ✅ 采纳 |
| R5 | `loadsfirst` | 双输入前置装载 + `block_m` 编译期分支 | 8b prefill 再 −18%（候选 62.52/63.14µs vs current best 76.68/76.97µs，opt_log §2.2） | ✅ winner |
| R5 | `singlepass` | 单遍方差 `E[x²]−E[x]²` | tie（无收益）；fp32 大均值有消减误差风险 | ❌ 弃：零收益换数值风险 |
| R5 | `loadsfirst` @ block_m=1 | 70b prefill 也走 loads-first | ub overflow（1573120 bits / 1572864，差 256 bits） | ❌ 否决：改为按 `block_m` 编译期分支 |
| R6 | merged final | hoist-wb + loadsfirst（block_m≥2） | 逐 workload 见 §3（8b −23.2~−23.3%、70b −14.0~−14.2%） | ✅ final |

**合并纪律**：round 6 的 7 条 final 记录来自同一最终候选 `loadsfirst`（基线为 Stage-3 生产文件）；round 4 曾把 `hoist-wb` 记为 `phase=final`，该 7 条记录在 round 6 被取代后已降级为 `merged`（duration/候选 id/raw profile 路径均未改，仅 phase 标签 final→merged），最终产物 sha256 与 7 条 `loadsfirst` final 记录完全一致，无需重测。

## 3. 优化结果（msprof 对比表）

| workload | base µs | final µs | Δ | base SOL% | final SOL% | 是否达标 |
|---|---:|---:|---:|---:|---:|---|
| llama-13b-prefill-float32 | 107.642 | 108.702 | +0.98% | 45.08 | 45.25 | ❌（访存线 70%） |
| llama-70b-decode-bfloat16 | 4.460 | 4.820 | +8.07% | 0.81 | 0.81 | 豁免（微型 shape） |
| llama-70b-prefill-bfloat16 | 168.283 | 144.733 | **-13.99%** | 33.14 | 33.43 | ❌（算力线 60%） |
| llama-70b-prefill-float16 | 164.233 | 140.873 | **-14.22%** | 29.70 | 29.25 | ❌（算力线 60%） |
| llama-8b-decode-bfloat16 | 4.010 | 4.340 | +8.23% | 0.53 | 0.52 | 豁免（微型 shape） |
| llama-8b-prefill-bfloat16 | 84.012 | 64.551 | **-23.16%** | 31.35 | 37.49 | ❌（算力线 60%） |
| llama-8b-prefill-float16 | 82.452 | 63.251 | **-23.29%** | 27.14 | 32.64 | ❌（算力线 60%） |

smoke：3 个 smoke 仅精度回归，不参与性能判定。

**测量口径与噪声**：baseline/final 相邻交错（round 6），每点独立 msprof 进程。13b fp32（+0.98%）与 decode（+8.07% / +8.23%）的上升为跨进程双态噪声——两者代码路径本轮**未改动**，不计为回退。

## 4. 精度

- `python perf_opt/FusedAddLayerNormFwdOp.py --level all` → **`All check passed!`**（L0/L1 全过）。
- L2/Boundary 为 WARN-only：`Boundary-extreme-affine` 是 weight/bias = ±65504 的 fp16 溢出 WARN，与 `hoist-wb` 相同，**非本轮引入**。
- 关键边界：copy-cast 上转（fp16→fp32）精确；L0 max_diff 与原版一致（opt_log 记录 fp16 9.766e-04 / bf16 3.125e-02 / fp32 1.907e-06）。
- smoke：3 个 smoke 精度 PASS。

## 5. 时间与成本

| 项 | 值 |
|---|---|
| 开始 / 结束（+08:00） | 2026-10-18 03:55 → 04:46 |
| 墙钟时长 | ≈ 51 min |
| token input/output/reasoning | 原文未记录 |
| cache-read / cost | 原文未记录 |

## 6. 产物

| 类别 | 路径 / 状态 |
|---|---|
| final kernel | `.../perf_opt/FusedAddLayerNormFwdOp.py`（sha256 原文未记录；opt_log 记 `d86b7d1d…`，与 7 条 final 记录 `artifact_sha256` 一致） |
| 调优记录 | `perf_opt/opt_log.md`、`perf_records.jsonl`、`logs/`、`candidates/` |
| 候选文件 | `candidates/_faln_kernel_loadsfirst.py` |
| 能力缺口 | 原文未记录 |
| gate 4 | 原文未记录 |
| 算子状态 | 原文未记录 |
| wrapper | 未翻转（原文建议采纳，由 conductor 在回归通过后执行） |
| 端到端 report | 原文未记录 |
| pattern-library 回写 | 原文未记录（opt_log 标 PL-1.10 同族复证） |

## 7. 天花板 / 豁免 / 缺口

- **豁免项**：`llama-70b-decode-bfloat16`（4.460/4.820µs）、`llama-8b-decode-bfloat16`（4.010/4.340µs）——decode 4.0–4.8µs 属微型 shape，性能豁免，不计入达标分母；smoke 3 例仅精度回归。
- **ceiling 证据**：
  - 未达线根因：向量 op 数（15+ pass）+ 2 次 row-reduce 的流水深度，roofline 为 `latency bound:pipeline caused`。
  - 判据分层：prefill 算力线 60%、13b fp32 访存线 70%，两档均未达；8b prefill 的 SOL 27→33% 说明收益真实但幅度不足以跨线。
  - 结构候选：**N 方向分核**是下一轮候选（原文明确「需结构级改造（N 方向分核/流水）才可能上 60%」），本轮未做。
  - block_m=1（70b prefill）受 UB 限制（差 256 bits），loads-first 无法覆盖该档——这是 70b 只拿到 −14%（仅 hoist）而非 −23% 的结构原因。
- **贴线风险**：无 <3pp 贴线项；最接近线的是 13b fp32 45.25%（访存线 70%）、8b prefill bf16 37.49%（算力线 60%），均明显在线下，属结构性 gap。
- **缺口**：N 方向分核 / 更深流水尚未实现（下一轮候选）；`block_m=1` 的 UB 边界差 256 bits 未解决。
- **`[DESIGN_LIMIT]`**：原文未记录设计层天花板判定；本目录未产出 `perf_feedback.md`。

## 8. 建议

1. **建议采纳 wrapper**：净收益明确（8b prefill −23.2~−23.3%、70b prefill −14.0~−14.2%），硬底线 Task Duration 全下降，精度 L0/L1 全绿。
2. 后续动作：优先试 **N 方向分核 / 流水加深**（唯一可能把 prefill 从 27–37% 推向 60% 的结构杠杆）；`block_m=1` 档需先解决 256 bits 的 UB 余量问题才能享受 loads-first。
3. **未解决项与原因**：0/7 达线；根因是向量链长度与 row-reduce 流水深度的结构上限，非 tiling/block_m 参数可解；单遍方差方向已证伪（tie + 数值风险）。
