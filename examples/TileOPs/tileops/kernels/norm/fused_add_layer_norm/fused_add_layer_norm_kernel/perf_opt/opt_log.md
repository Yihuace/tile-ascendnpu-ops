# FusedAddLayerNormFwdOp — Stage 4 调优日志

> CANN 9.0.0 / tilelang 0.1.2 dev root / Ascend910B3 / 2026-10-17。
> 口径：msprof op Task Duration(us) + GM Read+Write SOL ratio。
> 分类：prefill AI=3.0~3.8 > 平衡点 2.6 ⇒ **算力受限，线 60%**；13b fp32 AI=1.02 ⇒ 访存受限，线 70%。
> decode（4.1–4.7µs）属微型 shape，性能豁免。

## 0. 对象与入口
- `FusedAddLayerNormFwdOp`；kernel `_fused_add_layer_norm_kernel(M,N,eps,dtype)`（两段式：factory → `_func(block_m)`）；
- 产物 `perf_opt/FusedAddLayerNormFwdOp.py`；benchmark `bench_fused_add_layer_norm.py`（7 tune + 3 smoke）。

## 1. Baseline（round 1）
prefill：8b fp16 82.4µs/27.4%、8b bf16 84.4/30.9%、70b fp16 164.9/29.5%、70b bf16 169.6/33.1%、
13b fp32 108.2/44.4%；decode：4.2–4.6µs（0.5–0.9%，微型豁免）。
roofline advice 全部 = `latency bound:pipeline caused`。

## 2. 迭代日志（候选 vs current best，Task Duration）

### Round 2 — block_m 扫描
`block_m=1` 明显更差：8b fp16 82.3→119.8µs、bf16 84.4→123.4µs（−25%）。
默认 `_default_block_m`（N=4096→2、N=8192/5120→1）是正确的，保留。

### Round 3 — 循环不变量外提（A10）
weight/bias 原实现**每个 tile** 重新 `T.copy + vcast`；改为每 core 只加载/升精度一次，循环内只 `vbrc`。

| workload | baseline µs | hoist µs | Δ |
|---|---:|---:|---:|
| 70b-prefill bf16 | 168.75 | **144.96** | **−14.1%** |
| 70b-prefill fp16 | 165.31 | **141.05** | **−14.7%** |
| 8b-prefill bf16 | 84.49 | 79.24 | −6.2% |
| 8b-prefill fp16 | 82.28 | 76.87 | −6.6% |

**指标说明**：SOL ratio 基本不变（27→27、31→30.8、33→33.4）——Ratio 的分子含 `total_fops`，
去掉指令后 fops 与 duration 同降；**Task Duration −6~−15% 是真实收益**（硬底线满足），
但 Ratio 要上 60% 必须提升流水效率，不能只删指令。

### 合并与 winner
- winner：**weight/bias 循环不变量外提**（其余结构不变）；合并进 `perf_opt/FusedAddLayerNormFwdOp.py`；
- 未做：单遍方差（精度风险）、`T.Pipelined`（UB 预算：N=4096 已 ~120KB、N=8192 ~176KB，2-stage 会超 192KB）。

## Final Performance Test Data

官方最终测量：`perf_opt/FusedAddLayerNormFwdOp.py`（round 4，baseline/final 相邻交错）。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| _fused_add_layer_norm_kernel | llama-13b-prefill-float32 | 107.562 | 106.622 | hoist-wb |
| _fused_add_layer_norm_kernel | llama-70b-decode-bfloat16 | 4.620 | 4.660 | hoist-wb |
| _fused_add_layer_norm_kernel | llama-70b-prefill-bfloat16 | 168.653 | 145.113 | hoist-wb |
| _fused_add_layer_norm_kernel | llama-70b-prefill-float16 | 164.883 | 141.033 | hoist-wb |
| _fused_add_layer_norm_kernel | llama-8b-decode-bfloat16 | 4.250 | 4.120 | hoist-wb |
| _fused_add_layer_norm_kernel | llama-8b-prefill-bfloat16 | 84.642 | 79.252 | hoist-wb |
| _fused_add_layer_norm_kernel | llama-8b-prefill-float16 | 82.312 | 76.492 | hoist-wb |

### SOL 对账

| workload | base µs | opt µs | Δ | base SOL% | opt SOL% |
|---|---:|---:|---:|---:|---:|
| llama-13b-prefill-float32 | 107.562 | 106.622 | -0.87% | 43.80 | 45.53 |
| llama-70b-decode-bfloat16 | 4.620 | 4.660 | +0.87% | 0.81 | 0.78 |
| llama-70b-prefill-bfloat16 | 168.653 | 145.113 | -13.96% | 33.05 | 33.40 |
| llama-70b-prefill-float16 | 164.883 | 141.033 | -14.46% | 29.45 | 29.19 |
| llama-8b-decode-bfloat16 | 4.250 | 4.120 | -3.06% | 0.52 | 0.56 |
| llama-8b-prefill-bfloat16 | 84.642 | 79.252 | -6.37% | 30.97 | 30.77 |
| llama-8b-prefill-float16 | 82.312 | 76.492 | -7.07% | 27.53 | 27.07 |

**结论（如实）**：
- **0/7 达标**：prefill SOL 27–45%（算力受限线 60%、访存线 70%）；decode 微型豁免；
- 时延收益：70b prefill **−14.0~−14.5%**、8b prefill −6.4~−7.1%、decode −3.1%、13b −0.9%；
- 仍差 15–30 个点：瓶颈是 `latency bound:pipeline caused`（15+ 条向量 pass + 两次 reduce），
  下一轮应做**单遍方差 + 减少中转 cast + UB 瘦身后的 Pipelined**，属结构性改造，不是参数可解。

## 精度回归
`python perf_opt/FusedAddLayerNormFwdOp.py --level all` → **All check passed!**（L0/L1 全过）。

## Skill Retrospective
- **D**：norm 类去掉循环不变量的 GM 拷贝+vcast 可得 −14% 时延；但 SOL ratio 因 fops 同降而不动
  —— 报告必须同时给 Duration 与 SOL，不能只看 Ratio。
- **R**：block_m 扫描不能凭“更小更灵活”直觉，N=4096 时 block_m=1 反而 −25%。
- 证据：`logs/faln_baseline_runner2.log`、`logs/final_level_all.log`、`candidates/_faln_kernel_hoist.py`。
