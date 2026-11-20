# TileOPs Evaluation Report: FP8QuantOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-21T09:54:40.152163+00:00 |
| evaluation_scope | FP8QuantOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261021_095319_419340_FP8QuantOp |
| git_commit | 19315978eaa21e1f926cd483d4e6f3236a33115b |
| correctness_target | tests/ops/test_fp8_quant.py |
| benchmark_target | benchmarks/ops/bench_fp8_quant.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B2 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
| driver | 26.1.1 |
| pytorch | 2.9.0 |
| pytorch_npu | 2.9.0.post8 |
| python | 3.10.12 |
| os | Linux-5.10.0-216.0.0.115.oe2203sp4.aarch64-aarch64-with-glibc2.35 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 8 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| FP8QuantOp | 100.0% (8/8) | 2.60e+01 | 3 | 28.69% – 29.04% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### FP8QuantOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| kv-index-8k-d64-fp16 | 8.4400 | 29.04 | {"batch": 1, "index_dim": 64, "kv_group": 1, "seq_len_kv": 8192} | float16 | msprof | _fp8_quant_fwd_main | 1.7500 |
| kv-index-8k-d64-bf16 | 8.5200 | 28.76 | {"batch": 1, "index_dim": 64, "kv_group": 1, "seq_len_kv": 8192} | bfloat16 | msprof | _fp8_quant_fwd_main | 1.7500 |
| kv-index-4k-d128-fp32 | 8.1800 | 28.69 | {"batch": 1, "index_dim": 128, "kv_group": 1, "seq_len_kv": 4096} | float32 | msprof | _fp8_quant_fwd_main | 1.7500 |
