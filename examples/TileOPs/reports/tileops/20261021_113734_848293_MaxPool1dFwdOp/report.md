# TileOPs Evaluation Report: MaxPool1dFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-21T11:51:22.660805+00:00 |
| evaluation_scope | MaxPool1dFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261021_113734_848293_MaxPool1dFwdOp |
| git_commit | f947fb18c3022f05bca04af74d2f8222eb3980e7 |
| correctness_target | tests/ops/test_pool.py |
| benchmark_target | benchmarks/ops/bench_pool.py |

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
| 100.0% | 1 | 75 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| MaxPool1dFwdOp | 100.0% (47/47) | 0.00e+00 | 3 | 16.68% – 27.82% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### MaxPool1dFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| sincnet-speaker-local | 62.5400 | 16.68 | [32, 80, 4096] | float16 | msprof | main | 1.7500 |
| textcnn-global | 9.2000 | 26.05 | [64, 256, 128] | float16 | msprof | main | 1.7500 |
| ecg-cnn-dilated | 37.4200 | 27.82 | [16, 128, 2048] | bfloat16 | msprof | main | 1.7500 |
