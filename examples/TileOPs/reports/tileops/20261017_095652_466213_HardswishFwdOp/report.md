# TileOPs Evaluation Report: HardswishFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-17T10:12:02.704103+08:00 |
| evaluation_scope | HardswishFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261017_095652_466213_HardswishFwdOp |
| git_commit | 7ca0bb7e33b332f00e67c1ee6e2daec64c096949 |
| correctness_target | tests/ops/test_hardswish.py |
| benchmark_target | benchmarks/ops/bench_hardswish.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.2 (/home/developer/tilelang-mlir-ascend) |
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
| HardswishFwdOp | 100.0% (7/7) | 9.54e-08 | 6 | 50.40% – 61.66% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### HardswishFwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| mbv3-stage2 | 22.9405 | 54.97 | 3.0079 | ? | [32, 96, 56, 56] | float16 | msprof | main | 1.7500 | latency bound:compute caused |
| mbv3-stage2 | 23.8405 | 61.66 | 3.5079 | ? | [32, 96, 56, 56] | bfloat16 | msprof | main | 1.7500 | latency bound:compute caused |
| mbv3-stage3 | 15.6403 | 50.40 | 3.0080 | ? | [32, 240, 28, 28] | float16 | msprof | main | 1.7500 | latency bound:compute caused |
| mbv3-stage3 | 16.4403 | 55.89 | 3.5080 | ? | [32, 240, 28, 28] | bfloat16 | msprof | main | 1.7500 | latency bound:compute caused |
| fc-wide | 20.2804 | 54.14 | 3.0079 | ? | [2048, 4096] | float16 | msprof | main | 1.7500 | latency bound:compute caused |
| fc-wide | 21.1804 | 60.44 | 3.5079 | ? | [2048, 4096] | bfloat16 | msprof | main | 1.7500 | latency bound:compute caused |
