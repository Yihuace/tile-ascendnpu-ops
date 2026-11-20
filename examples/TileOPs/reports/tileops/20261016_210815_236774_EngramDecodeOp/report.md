# TileOPs Evaluation Report: EngramDecodeOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-16T21:13:55.896645+08:00 |
| evaluation_scope | EngramDecodeOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261016_210815_236774_EngramDecodeOp |
| git_commit | 76e45509964c313ceca43930cceff43c7820551c |
| correctness_target | tests/ops/test_engram_decode.py |
| benchmark_target | benchmarks/ops/bench_engram_decode.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
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
| 100.0% | 1 | 5 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| EngramDecodeOp | 100.0% (5/5) | 8.77e-03 | 4 | 0.72% – 4.74% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### EngramDecodeOp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| decode-smoke-b1-dmem256-d256-fp16 | 21.7000 | 0.72 | {"batch": 1, "conv_kernel_size": 4, "d": 256, "d_mem": 256, "dilation": 3, "max_conv_len": 9} | float16 | msprof | main | 1.7500 |
| decode-b1-dmem512-d256-fp16 | 31.4800 | 0.97 | {"batch": 1, "conv_kernel_size": 4, "d": 256, "d_mem": 512, "dilation": 3, "max_conv_len": 12} | float16 | msprof | main | 1.7500 |
| decode-b4-dmem1024-d512-fp16 | 78.8000 | 3.62 | {"batch": 4, "conv_kernel_size": 4, "d": 512, "d_mem": 1024, "dilation": 5, "max_conv_len": 20} | float16 | msprof | main | 1.7500 |
| decode-b8-dmem512-d256-bf16 | 31.3800 | 4.74 | {"batch": 8, "conv_kernel_size": 4, "d": 256, "d_mem": 512, "dilation": 3, "max_conv_len": 18} | bfloat16 | msprof | main | 1.7500 |
