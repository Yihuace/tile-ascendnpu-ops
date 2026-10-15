# TileOPs Evaluation Report: GLADecodeOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T14:17:23.058691+08:00 |
| evaluation_scope | GLADecodeOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_141409_254847_GLADecodeOp |
| git_commit | 4ce834cf08cca952aa99da76086b6334427a739f |
| correctness_target | tests/ops/test_gla_decode.py |
| benchmark_target | benchmarks/ops/bench_gla_decode.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 8.5.2 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| python | 3.11.4 |
| os | Linux-5.10.0-182.0.0.95.r2673_211.hce2.aarch64-aarch64-with-glibc2.31 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 11 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| GLADecodeOp | 100.0% (11/11) | N/A | 5 | 2.47% – 14.12% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GLADecodeOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| serving-b1-h8 | 9.1200 | 2.47 | {"B": 1, "DK": 128, "DV": 128, "H": 8, "scale": 0.08838834764831845} | bfloat16 | msprof | _gla_decode_transit_main | 1.7500 |
| serving-b1-h32 | 10.2800 | 8.78 | {"B": 1, "DK": 128, "DV": 128, "H": 32, "scale": 0.08838834764831845} | bfloat16 | msprof | _gla_decode_transit_main | 1.7500 |
| serving-b1-h64 | 18.1400 | 9.95 | {"B": 1, "DK": 128, "DV": 128, "H": 64, "scale": 0.08838834764831845} | bfloat16 | msprof | _gla_decode_transit_main | 1.7500 |
| serving-b8-h32 | 55.7200 | 12.95 | {"B": 8, "DK": 128, "DV": 128, "H": 32, "scale": 0.08838834764831845} | bfloat16 | msprof | _gla_decode_transit_main | 1.7500 |
| serving-b8-h64 | 102.2400 | 14.12 | {"B": 8, "DK": 128, "DV": 128, "H": 64, "scale": 0.08838834764831845} | bfloat16 | msprof | _gla_decode_transit_main | 1.7500 |
