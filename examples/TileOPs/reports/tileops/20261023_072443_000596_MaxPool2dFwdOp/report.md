# TileOPs Evaluation Report: MaxPool2dFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-23T07:35:38.517331+00:00 |
| evaluation_scope | MaxPool2dFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261023_072443_000596_MaxPool2dFwdOp |
| git_commit | 16f1ee4578a66e9123bb32783956e272df17ecad |
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
| MaxPool2dFwdOp | 100.0% (47/47) | 0.00e+00 | 9 | 14.71% – 31.61% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### MaxPool2dFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| resnet-stem | 122.9600 | 27.98 | [32, 64, 112, 112] | float16 | msprof | main | 1.7500 |
| resnet-stem | 110.9600 | 31.61 | [32, 64, 112, 112] | bfloat16 | msprof | main | 1.7500 |
| resnet-stem | 147.9400 | 27.85 | [32, 64, 112, 112] | float32 | msprof | main | 1.7500 |
| vgg-block | 18.1800 | 29.59 | [16, 128, 56, 56] | float16 | msprof | main | 1.7500 |
| vgg-block | 37.8400 | 25.60 | [16, 128, 56, 56] | bfloat16 | msprof | main | 1.7500 |
| vgg-block | 41.5200 | 24.96 | [16, 128, 56, 56] | float32 | msprof | main | 1.7500 |
| alexnet-ceil | 40.7800 | 14.71 | [32, 64, 55, 55] | float16 | msprof | main | 1.7500 |
| alexnet-ceil | 42.4000 | 27.28 | [32, 64, 55, 55] | bfloat16 | msprof | main | 1.7500 |
| alexnet-ceil | 66.2800 | 17.86 | [32, 64, 55, 55] | float32 | msprof | main | 1.7500 |
