# AddFwdOp — Stage 4 调优日志

> 工具链版本戳：CANN **9.0.0**（bishengir-compile 1.1.0 / hivmc 0.2.0）/ tilelang 0.1.2 dev root
> / Ascend910B3（Chip 6，20 AIC / 40 AIV）/ 2026-10-17。
> 口径：`msprof op` Task Duration(us) + `visualize_data.bin` 的 **GM Read + Write SOL ratio**（§3.1）。
> 测量方式：**直跑 kernel 文件**（`perf_opt/bench_kernel.py`），不经 wrapper。
> 达标线：逐元素/访存受限 **≥ 70%**；微型 shape（实测 2–4 µs）豁免。

## 0. 对象与入口

- kernel：`tileops/kernels/elementwise/add/add_kernel/add_fwd_kernel.py`
- 分派：`AddFwdOp.forward → AddFwdKernel → add_fwd_kernel(N, dtype, alpha, block_size)`
- benchmark：`benchmarks/ops/bench_add.py::test_add_manifest_bench`
- workload：4 label × 3 dtype = 12 case（11 tune + 1 smoke）

## 1. Baseline（9.0，standalone msprof）

| workload | µs | SOL% | AI | advice | 判定 |
|---|---:|---:|---:|---|---|
| hidden-state-prefill fp32 | 46.86 | 81.83 | 0.1 | memory bound | ✅ |
| hidden-state-prefill bf16 | 25.80 | 74.32 | 1.0 | latency bound:memory | ✅ |
| cnn-feat-broadcast fp32 | 72.08 | 81.46 | 0.1 | memory bound | ✅ |
| cnn-feat-broadcast fp16 | 37.14 | 79.05 | 0.8 | latency bound:memory | ✅ |
| cnn-feat-broadcast bf16 | 37.22 | 78.88 | 1.0 | latency bound:memory | ✅ |
| llama-7b-ffn fp32 | 62.74 | 82.13 | 0.1 | memory bound | ✅ |
| llama-7b-ffn fp16 | 33.64 | 76.59 | 0.8 | latency bound:memory | ✅ |
| llama-7b-ffn bf16 | 34.80 | 74.04 | 1.0 | latency bound:memory | ✅ |
| linear-bias fp32 | 25.04 | 76.57 | 0.1 | latency bound:memory | ✅ |
| **linear-bias fp16** | 14.18 | **67.61** | 0.8 | latency bound:memory | ❌ 差 2.4 点 |
| **linear-bias bf16** | 13.70 | **69.98** | 1.0 | latency bound:memory | ❌ 差 0.02 点 |

证据：`perf_opt/logs/baseline.tsv`、`perf_opt/logs/baseline_roofline.json`、`perf_opt/profiles/base_*`。

## 2. 诊断

- 全部 case 的 AI ≈ 0.1–1.0 op/byte → **访存受限**，达标线 70%；
- 9/11 已过线（74–82%），**只有 linear-bias 的 fp16/bf16 未过**：
  - linear-bias 是最小 shape（N=4.19M），时延 13.7–14.2 µs，固定启动开销占比最高；
  - bf16 69.98% 已在噪声内，fp16 67.61% 差约 0.5 µs；
- 同态重复实测（`perf_opt/logs/noise.tsv`）用于确定噪声底；<5% 的候选差分必须走 `ab_test.py`。

## 3. Round 1 候选计划

| ID | 改动 | 预期 | 风险 |
|---|---|---|---|
| A1 | `block_size` 8192 → 12288/16384（仅 fp16/bf16，UB 允许时） | 减少 tile 数与循环开销 | 非整除产生尾块 |
| A2 | 去掉独立 `y_ub`，复用 `a_ub` 作为输出缓冲 | 省 UB，允许更大 block；少一趟 UB 写 | 需确认 a_ub 生命周期 |
| A3 | 双缓冲 / `T.Pipelined` 让 MTE2 与 VEC 重叠 | 提升带宽利用率 | 改动大，先看 A1/A2 |

合并纪律：候选只写 `perf_opt/`，wrapper 不动；winner 合并后复测全部 11 tune + smoke 精度。
