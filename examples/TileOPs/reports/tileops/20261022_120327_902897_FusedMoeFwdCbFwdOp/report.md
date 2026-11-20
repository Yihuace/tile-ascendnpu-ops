# TileOPs Evaluation Report: FusedMoeFwdCbFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-22T12:24:25.089379+08:00 |
| evaluation_scope | FusedMoeFwdCbFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261022_120327_902897_FusedMoeFwdCbFwdOp |
| git_commit | 147b71bcf2f0d4b613eaf815fc0cd45dc22b4048 |
| correctness_target | tests/ops/test_moe_fused_moe_cb.py |
| benchmark_target | benchmarks/ops/bench_moe_fused_moe_cb.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910_9362 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
| driver | 25.5.5 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.1.10+ubuntu.20.4.cann900 |
| python | 3.11.4 |
| os | Linux-5.10.0-182.0.0.95.r2220_156.hce2.aarch64-aarch64-with-glibc2.31 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 8 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| FusedMoeFwdCbFwdOp | 100.0% (8/8) | N/A | 3 | 44.72% – 53.74% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### FusedMoeFwdCbFwdOp

| Label | Latency (us) | Ratio (%) | Mem Roofline Ratio (%) | Compute Roofline Ratio (%) | Achieved BW Util (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|
| kimi-k2-decode | 13672.6465 | 45.98 | 45.98 | 37.73 | 91.68 | {"dtype_str": "bfloat16", "ffn_size": 2048, "hidden_size": 7168, "num_experts": 256, "num_tokens": 512, "renormalize": true, "routed_scaling_factor": 2.827, "scoring_func": "sigmoid", "top_k": 8} | bfloat16 | msprof | main_mix_aic | 1.8000 |
| kimi-k2-medium | 20977.9414 | 44.72 | 44.72 | 34.46 | 59.88 | {"dtype_str": "bfloat16", "ffn_size": 2048, "hidden_size": 7168, "num_experts": 256, "num_tokens": 2048, "renormalize": true, "routed_scaling_factor": 2.827, "scoring_func": "sigmoid", "top_k": 8} | bfloat16 | msprof | main_mix_aic | 1.8000 |
| kimi-k2-prefill | 26514.0508 | 53.74 | 53.74 | 40.27 | 47.50 | {"dtype_str": "bfloat16", "ffn_size": 2048, "hidden_size": 7168, "num_experts": 256, "num_tokens": 4096, "renormalize": true, "routed_scaling_factor": 2.827, "scoring_func": "sigmoid", "top_k": 8} | bfloat16 | msprof | main_mix_aic | 1.8000 |
