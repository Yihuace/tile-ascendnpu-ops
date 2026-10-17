# 优化报告：FusedAddLayerNormFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `FusedAddLayerNormFwdOp`；纯 Vector，4 输入 2 输出（y + pre-norm residual_out） |
| kernel | `tileops/kernels/norm/fused_add_layer_norm/fused_add_layer_norm_kernel/` |
| 产物 | `perf_opt/FusedAddLayerNormFwdOp.py` |
| 口径 / 工具链 | msprof op Task Duration + SOL ratio / CANN 9.0.0 / Ascend910B3 / 2026-10-17 |
| benchmark | `bench_fused_add_layer_norm.py`（7 tune + 3 smoke） |

## 1. 结论

- **0/7 达标**：prefill SOL 27–45%（prefill AI 3.0–3.8 ⇒ 算力受限线 **60%**；13b fp32 AI 1.02 ⇒ 访存线 70%）；
  decode 4.1–4.7µs 属微型 shape，性能豁免。
- 优化：**weight/bias 循环不变量外提**（原实现每 tile 重做 `T.copy + vcast`）。
- 时延收益：70b prefill **−14.0~−14.5%**、8b prefill −6.4~−7.1%、8b decode −3.1%、13b fp32 −0.9%。
- **指标行为**：SOL ratio 基本不动——Ratio 分子含 `total_fops`，删指令后 fops 与 duration 同降。
  所以本报告以 Task Duration 为主证据；比值要上 60% 需提升流水效率，不是继续删指令。

## 2. 所做改动
仅 `perf_opt/`：weight/bias 每 core 加载 + 升 fp32 一次，循环内只 `vbrc`；其余结构不变。
`block_m` 扫描（1 vs 2）：block_m=1 慢 25%，保留默认。

## 3. 优化结果（round 4 相邻交错）

| workload | base µs | opt µs | Δ | base SOL% | opt SOL% |
|---|---:|---:|---:|---:|---:|
| llama-13b-prefill-float32 | 107.562 | 106.622 | -0.87% | 43.80 | 45.53 |
| llama-70b-decode-bfloat16 | 4.620 | 4.660 | +0.87% | 0.81 | 0.78 |
| llama-70b-prefill-bfloat16 | 168.653 | 145.113 | -13.96% | 33.05 | 33.40 |
| llama-70b-prefill-float16 | 164.883 | 141.033 | -14.46% | 29.45 | 29.19 |
| llama-8b-decode-bfloat16 | 4.250 | 4.120 | -3.06% | 0.52 | 0.56 |
| llama-8b-prefill-bfloat16 | 84.642 | 79.252 | -6.37% | 30.97 | 30.77 |
| llama-8b-prefill-float16 | 82.312 | 76.492 | -7.07% | 27.53 | 27.07 |

### 精度
`python perf_opt/FusedAddLayerNormFwdOp.py --level all` → `All check passed!`。

## 4. 时间与 token

| 项 | 值 |
|---|---|
| 开始 / 结束 | 2026-10-17 21:40 → 22:15 (+08:00) |
| 墙钟 | ≈ 35 min |
| tokens (in/out/reasoning) | 19,224 / 18,131 / 13,582 |
| cache-read / cost | 25,606,912 / $0.0987 |

## 5. 产物与后续
- `perf_opt/FusedAddLayerNormFwdOp.py`、`opt_log.md`、`Optimize.md`、`perf_records.jsonl`、`logs/`、`candidates/`
- 未达线的根因：15+ 条向量 pass + 2 次 row-reduce，roofline advice = `latency bound:pipeline caused`
- 下一轮：单遍方差（`E[x²]−E[x]²`，需 L0/L1 精度验证）+ 减少 fp32 中转 + UB 瘦身后的 `T.Pipelined`
- gate 4 pass；state `phase=DONE`
