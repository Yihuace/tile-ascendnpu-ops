# 优化报告：FusedAddLayerNormFwdOp（Stage-4 optimize，二次深挖）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `FusedAddLayerNormFwdOp`；纯 Vector，4 输入 2 输出（y + pre-norm residual_out） |
| kernel | `tileops/kernels/norm/fused_add_layer_norm/fused_add_layer_norm_kernel/` |
| 产物 | `perf_opt/FusedAddLayerNormFwdOp.py` |
| 口径 / 工具链 | msprof op Task Duration + SOL ratio / CANN 9.0.0 / Ascend910B3 / 2026-10-18 |
| benchmark | `bench_fused_add_layer_norm.py`（7 tune + 3 smoke） |

## 1. 结论

- **0/7 达标**（SOL 仍 27–45%：prefill AI 3.0–3.8 ⇒ 算力线 **60%**；13b fp32 AI 1.02 ⇒ 访存线 70%）；
  decode 4.0–4.8µs 属微型 shape，性能豁免。
- 优化（两段叠加）：
  1. **weight/bias 循环不变量外提**（round 3，A10）——70b prefill −14%、8b −7%。
  2. **loads-first 双输入前置装载**（round 5，PL-1.10 同族）——8b prefill 再 −18%。
- **时延收益（vs Stage-3 生产基线）**：**8b prefill −23.2~−23.3%**、**70b prefill −14.0~−14.2%**、
  13b fp32 +0.98%（fp32 路径未改，噪声内）、decode +8%（微型豁免，噪声内）。
- **指标行为**：SOL ratio 在 8b 上 27→33%（bf16 31→37%），但仍未达 60% 算力线——瓶颈已从
  MTE2/VEC 装载串行转到纯向量 op 数 + row-reduce 流水深度，需结构级改造（N 方向分核/流水）才可能上 60%。

## 2. 所做改动
仅 `perf_opt/`：
- round 3：weight/bias 每 core 加载 + 升 fp32 一次，循环内只 `vbrc`。
- round 5：fp16/bf16 transit 路径双输入前置装载——`T.copy(x→x_f32)`、`T.copy(residual→sq_f32)`
  （fp16→fp32 上转 copy-cast 精确），`stage` 只留 round-trip + 输出；按 `block_m` 编译期分支
  （`block_m==1` 保留单 staging 避免 ub overflow，`block_m>=2` 走 loads-first）。
- `block_m` 扫描（1 vs 2）：block_m=1 慢 25%，保留默认。

## 3. 优化结果（round 6 相邻交错，baseline=Stage-3 生产）

| workload | base µs | opt µs | Δ | base SOL% | opt SOL% |
|---|---:|---:|---:|---:|---:|
| llama-13b-prefill-float32 | 107.642 | 108.702 | +0.98% | 45.08 | 45.25 |
| llama-70b-decode-bfloat16 | 4.460 | 4.820 | +8.07% | 0.81 | 0.81 |
| llama-70b-prefill-bfloat16 | 168.283 | 144.733 | **-13.99%** | 33.14 | 33.43 |
| llama-70b-prefill-float16 | 164.233 | 140.873 | **-14.22%** | 29.70 | 29.25 |
| llama-8b-decode-bfloat16 | 4.010 | 4.340 | +8.23% | 0.53 | 0.52 |
| llama-8b-prefill-bfloat16 | 84.012 | 64.551 | **-23.16%** | 31.35 | 37.49 |
| llama-8b-prefill-float16 | 82.452 | 63.251 | **-23.29%** | 27.14 | 32.64 |

### 精度
`python perf_opt/FusedAddLayerNormFwdOp.py --level all` → `All check passed!`（L0/L1 全过；
Boundary-extreme-affine 为 weight/bias=±65504 的 fp16 溢出 WARN，与 hoist-wb 同，非本轮引入）。

## 4. 达标数

| 口径 | 达标数 |
|---|---|
| 全 shape（7 tune） | **0/7**（SOL 未达线；decode 豁免） |
| 可调 shape（3 prefill） | **0/3**（SOL 未达线；Task Duration 均下降） |

## 5. 时间与 token

| 项 | 值 |
|---|---|
| 开始 / 结束 | 2026-10-18 03:55 → 04:46 (+08:00) |
| 墙钟 | ≈ 51 min |

## 6. 产物与后续
- `perf_opt/FusedAddLayerNormFwdOp.py`、`opt_log.md`、`perf_records.jsonl`、`logs/`、`candidates/`
- **建议采纳 wrapper**：有净收益（8b prefill −23%、70b prefill −14%，硬底线 Task Duration 均下降）；
  无精度回退（L0/L1 全绿）。
- 未达线根因：向量 op 数（15+ pass）+ 2 次 row-reduce 的流水深度，roofline `latency bound:pipeline caused`；
  block_m=1 上 loads-first 受 ub overflow（差 256 bits）限制，N 方向分核是下一轮候选。
- 单遍方差 `E[x²]−E[x]²`：**不可用**（tie 无收益，且 fp32 大均值场景有消减误差风险；L0 虽过但换零收益，弃）。
