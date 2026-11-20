# TileOPs Evaluation Report: FusedMoeFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-22T12:01:42.341800+08:00 |
| evaluation_scope | FusedMoeFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261022_113704_755433_FusedMoeFwdOp |
| git_commit | 147b71bcf2f0d4b613eaf815fc0cd45dc22b4048 |
| correctness_target | tests/ops/test_moe_fused_moe.py |
| benchmark_target | benchmarks/ops/bench_moe_fused_moe.py |

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
| 100.0% | 1 | 12 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| FusedMoeFwdOp | 100.0% (12/12) | N/A | 4 | 51.70% – 60.28% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### FusedMoeFwdOp

| Label | Latency (us) | Ratio (%) | Mem Roofline Ratio (%) | Compute Roofline Ratio (%) | Achieved BW Util (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|
| qwen3-235b-decode | 7874.6621 | 55.72 | 55.72 | 44.84 | 79.65 | {"dtype_str": "bfloat16", "ffn_size": 2048, "hidden_size": 7168, "num_experts": 128, "num_tokens": 512, "renormalize": false, "routed_scaling_factor": 1.0, "scoring_func": "softmax", "top_k": 8} | bfloat16 | msprof | main_mix_aic | 1.8000 |
| qwen3-235b-prefill | 21492.0293 | 56.40 | 56.40 | 40.82 | 29.45 | {"dtype_str": "bfloat16", "ffn_size": 2048, "hidden_size": 7168, "num_experts": 128, "num_tokens": 4096, "renormalize": false, "routed_scaling_factor": 1.0, "scoring_func": "softmax", "top_k": 8} | bfloat16 | msprof | main_mix_aic | 1.8000 |
| deepseek-v3-decode | 14227.7549 | 60.28 | 60.28 | 49.64 | 88.11 | {"dtype_str": "bfloat16", "ffn_size": 2048, "hidden_size": 7168, "num_experts": 256, "num_tokens": 512, "renormalize": false, "routed_scaling_factor": 1.0, "scoring_func": "sigmoid", "top_k": 8} | bfloat16 | msprof | main_mix_aic | 1.8000 |
| deepseek-v3-prefill | 25929.4414 | 51.70 | 51.70 | 38.51 | 48.57 | {"dtype_str": "bfloat16", "ffn_size": 2048, "hidden_size": 7168, "num_experts": 256, "num_tokens": 4096, "renormalize": false, "routed_scaling_factor": 1.0, "scoring_func": "sigmoid", "top_k": 8} | bfloat16 | msprof | main_mix_aic | 1.8000 |
