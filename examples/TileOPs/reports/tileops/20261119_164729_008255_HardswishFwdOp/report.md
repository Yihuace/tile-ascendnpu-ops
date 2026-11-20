# TileOPs Evaluation Report: HardswishFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-19T17:01:28.040954+08:00 |
| evaluation_scope | HardswishFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261119_164729_008255_HardswishFwdOp |
| git_commit | e5dc75beb234c82d3292e900b8793b6e7fe1cc41 |
| correctness_target | tests/ops/test_hardswish.py |
| benchmark_target | benchmarks/ops/bench_hardswish.py |

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
| 100.0% | 1 | 7 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| HardswishFwdOp | 100.0% (7/7) | 7.80e-04 | 6 | 55.44% – 64.82% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### HardswishFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| mbv3-stage2 | 19.8800 | 63.28 | [32, 96, 56, 56] | float16 | msprof | main | 1.7500 |
| mbv3-stage2 | 22.6800 | 64.82 | [32, 96, 56, 56] | bfloat16 | msprof | main | 1.7500 |
| mbv3-stage3 | 14.2000 | 55.44 | [32, 240, 28, 28] | float16 | msprof | main | 1.7500 |
| mbv3-stage3 | 15.4800 | 59.36 | [32, 240, 28, 28] | bfloat16 | msprof | main | 1.7500 |
| fc-wide | 17.8200 | 61.47 | [2048, 4096] | float16 | msprof | main | 1.7500 |
| fc-wide | 20.1400 | 63.56 | [2048, 4096] | bfloat16 | msprof | main | 1.7500 |
