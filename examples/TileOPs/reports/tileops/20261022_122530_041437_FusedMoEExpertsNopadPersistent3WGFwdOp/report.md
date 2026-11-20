# TileOPs Evaluation Report: FusedMoEExpertsNopadPersistent3WGFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-22T12:48:00.929662+08:00 |
| evaluation_scope | FusedMoEExpertsNopadPersistent3WGFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261022_122530_041437_FusedMoEExpertsNopadPersistent3WGFwdOp |
| git_commit | 147b71bcf2f0d4b613eaf815fc0cd45dc22b4048 |
| correctness_target | tests/ops/test_fused_moe_experts.py |
| benchmark_target | benchmarks/ops/bench_fused_moe_experts.py |

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
| 100.0% | 1 | 20 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| FusedMoEExpertsNopadPersistent3WGFwdOp | 100.0% (20/20) | N/A | 4 | 54.19% – 59.84% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### FusedMoEExpertsNopadPersistent3WGFwdOp

| Label | Latency (us) | Ratio (%) | Mem Roofline Ratio (%) | Compute Roofline Ratio (%) | Achieved BW Util (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|
| qwen3-235b-decode | 7849.5029 | 55.91 | 55.91 | 44.99 | 79.90 | {"device": "npu", "dtype_str": "bfloat16", "ffn_size": 2048, "hidden_size": 7168, "num_experts": 128, "num_tokens": 512, "top_k": 8} | bfloat16 | msprof | main_mix_aic | 1.8000 |
| qwen3-235b-prefill | 21428.1914 | 56.15 | 56.15 | 40.55 | 29.53 | {"device": "npu", "dtype_str": "bfloat16", "ffn_size": 2048, "hidden_size": 7168, "num_experts": 128, "num_tokens": 4096, "top_k": 8} | bfloat16 | msprof | main_mix_aic | 1.8000 |
| deepseek-v3-decode | 14331.9736 | 59.84 | 59.84 | 49.28 | 87.46 | {"device": "npu", "dtype_str": "bfloat16", "ffn_size": 2048, "hidden_size": 7168, "num_experts": 256, "num_tokens": 512, "top_k": 8} | bfloat16 | msprof | main_mix_aic | 1.8000 |
| deepseek-v3-prefill | 26215.3340 | 54.19 | 54.19 | 40.52 | 48.03 | {"device": "npu", "dtype_str": "bfloat16", "ffn_size": 2048, "hidden_size": 7168, "num_experts": 256, "num_tokens": 4096, "top_k": 8} | bfloat16 | msprof | main_mix_aic | 1.8000 |
