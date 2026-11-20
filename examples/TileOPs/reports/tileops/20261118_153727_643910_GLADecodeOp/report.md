# TileOPs Evaluation Report: GLADecodeOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-18T15:40:15.214289+08:00 |
| evaluation_scope | GLADecodeOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261118_153727_643910_GLADecodeOp |
| correctness_target | tests/ops/test_gla_decode.py |
| benchmark_target | benchmarks/ops/bench_gla_decode.py |

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
| 100.0% | 1 | 12 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| GLADecodeOp | 100.0% (12/12) | N/A | 5 | 4.35% – 46.77% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GLADecodeOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| serving-b1-h8 | 3.5000 | 4.35 | {"B": 1, "DK": 128, "DV": 128, "H": 8, "scale": 0.08838834764831845} | bfloat16 | msprof | _gla_decode_algo1_1h_main | 1.7500 |
| serving-b1-h32 | 5.2400 | 11.61 | {"B": 1, "DK": 128, "DV": 128, "H": 32, "scale": 0.08838834764831845} | bfloat16 | msprof | _gla_decode_algo1_1h_main | 1.7500 |
| serving-b1-h64 | 5.9600 | 20.42 | {"B": 1, "DK": 128, "DV": 128, "H": 64, "scale": 0.08838834764831845} | bfloat16 | msprof | _gla_decode_algo1_1h_main | 1.7500 |
| serving-b8-h32 | 13.4600 | 36.17 | {"B": 8, "DK": 128, "DV": 128, "H": 32, "scale": 0.08838834764831845} | bfloat16 | msprof | _gla_decode_algo1_1h_main | 1.7500 |
| serving-b8-h64 | 20.8200 | 46.77 | {"B": 8, "DK": 128, "DV": 128, "H": 64, "scale": 0.08838834764831845} | bfloat16 | msprof | _gla_decode_algo1_1h_main | 1.7500 |
