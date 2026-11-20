# TileOPs Evaluation Report: AmaxFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-26T11:39:59.504368+08:00 |
| evaluation_scope | AmaxFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261026_113750_230484_AmaxFwdOp |
| git_commit | da4f6a41cbe98706e9bf9c6d0547f459b043e01c |
| correctness_target | tests/ops/test_amax.py |
| benchmark_target | benchmarks/ops/bench_amax.py |

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
| 100.0% | 1 | 14 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| AmaxFwdOp | 100.0% (14/14) | 0.00e+00 | 4 | 24.95% – 71.81% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### AmaxFwdOp

| Label | Latency (us) | Ratio (%) | Mem Roofline Ratio (%) | Compute Roofline Ratio (%) | Achieved BW Util (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|
| hidden-state-reduce | 12.9800 | 71.81 | 71.81 | 8.38 | 71.83 | [2048, 4096] | float16 | msprof | main | 1.8000 |
| hidden-state-reduce | 16.5400 | 56.35 | 56.35 | 24.92 | 56.37 | [2048, 4096] | bfloat16 | msprof | main | 1.8000 |
| long-seq-reduce | 9.3400 | 24.95 | 24.95 | 10.96 | 24.95 | [64, 32768] | bfloat16 | msprof | N/A | 1.8000 |
| 3d-multidim-reduce | 5.4800 | 42.52 | 42.52 | 4.84 | 42.52 | [4, 128, 4096] | float16 | msprof | main | 1.8000 |
