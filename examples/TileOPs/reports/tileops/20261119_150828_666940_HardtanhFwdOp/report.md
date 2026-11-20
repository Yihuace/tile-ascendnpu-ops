# TileOPs Evaluation Report: HardtanhFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-19T15:21:23.713796+08:00 |
| evaluation_scope | HardtanhFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261119_150828_666940_HardtanhFwdOp |
| git_commit | 2cc7f79fec3fc13dfaad26f211575613f82c605b |
| correctness_target | tests/ops/test_hardtanh.py |
| benchmark_target | benchmarks/ops/bench_hardtanh.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 9.3.0-beta.1 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.1.10+ubuntu.20.4.cann900 |
| python | 3.11.4 |
| os | Linux-5.10.0-182.0.0.95.r2673_211.hce2.aarch64-aarch64-with-glibc2.31 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 8 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| HardtanhFwdOp | 100.0% (8/8) | 0.00e+00 | 5 | 61.22% – 81.45% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### HardtanhFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| mlp-hidden | 15.3200 | 62.58 | [2048, 4096] | float16 | msprof | main | 1.7500 |
| mlp-hidden | 15.6600 | 61.22 | [2048, 4096] | bfloat16 | msprof | main | 1.7500 |
| elementwise-16m | 28.3800 | 67.56 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| elementwise-16m | 27.7400 | 69.12 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| elementwise-16m | 47.0800 | 81.45 | [4096, 4096] | float32 | msprof | main | 1.7500 |
