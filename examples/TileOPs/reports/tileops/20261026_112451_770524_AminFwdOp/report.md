# TileOPs Evaluation Report: AminFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-26T11:27:14.222316+08:00 |
| evaluation_scope | AminFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261026_112451_770524_AminFwdOp |
| git_commit | da4f6a41cbe98706e9bf9c6d0547f459b043e01c |
| correctness_target | tests/ops/test_amin.py |
| benchmark_target | benchmarks/ops/bench_amin.py |

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
| AminFwdOp | 100.0% (14/14) | 0.00e+00 | 4 | 24.79% – 66.67% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### AminFwdOp

| Label | Latency (us) | Ratio (%) | Mem Roofline Ratio (%) | Compute Roofline Ratio (%) | Achieved BW Util (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|
| hidden-state-reduce | 13.9800 | 66.67 | 66.67 | 7.78 | 66.69 | [2048, 4096] | float16 | msprof | main | 1.8000 |
| hidden-state-reduce | 16.8600 | 55.28 | 55.28 | 24.45 | 55.30 | [2048, 4096] | bfloat16 | msprof | main | 1.8000 |
| long-seq-reduce | 9.4000 | 24.79 | 24.79 | 10.90 | 24.79 | [64, 32768] | bfloat16 | msprof | N/A | 1.8000 |
| 3d-multidim-reduce | 5.4200 | 42.99 | 42.99 | 4.89 | 42.99 | [4, 128, 4096] | float16 | msprof | main | 1.8000 |
