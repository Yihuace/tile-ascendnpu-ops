# TileOPs Evaluation Report: GLADecodeOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-23T18:10:11.357599+08:00 |
| evaluation_scope | GLADecodeOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261023_180626_074474_GLADecodeOp |
| git_commit | 60cfa4e4e811e9deaa895ae56181d9bc5be488c4 |
| correctness_target | tests/ops/test_gla_decode.py |
| benchmark_target | benchmarks/ops/bench_gla_decode.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.2+60cfa4e4e811e9deaa895ae56181d9bc5be488c4 (/mnt/workspace/tilelang-mlir-ascend@9309edb) |
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
| GLADecodeOp | 100.0% (11/11) | N/A | 5 | 3.17% – 19.94% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GLADecodeOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| serving-b1-h8 | 6.8400 | 3.17 | 3.6385 | ? | {"B": 1, "DK": 128, "DV": 128, "H": 8, "scale": 0.08838834764831845} | bfloat16 | msprof | _gla_decode_transit_main | 1.7500 | latency bound:pipeline caused |
| serving-b1-h32 | 8.6800 | 9.98 | 3.6398 | ? | {"B": 1, "DK": 128, "DV": 128, "H": 32, "scale": 0.08838834764831845} | bfloat16 | msprof | _gla_decode_transit_main | 1.7500 | latency bound:pipeline caused |
| serving-b1-h64 | 12.5400 | 13.83 | 3.6410 | ? | {"B": 1, "DK": 128, "DV": 128, "H": 64, "scale": 0.08838834764831845} | bfloat16 | msprof | _gla_decode_transit_main | 1.7500 | latency bound:pipeline caused |
| serving-b8-h32 | 37.7200 | 18.37 | 3.6385 | ? | {"B": 8, "DK": 128, "DV": 128, "H": 32, "scale": 0.08838834764831845} | bfloat16 | msprof | _gla_decode_transit_main | 1.7500 | latency bound:pipeline caused |
| serving-b8-h64 | 69.4800 | 19.94 | 3.6384 | ? | {"B": 8, "DK": 128, "DV": 128, "H": 64, "scale": 0.08838834764831845} | bfloat16 | msprof | _gla_decode_transit_main | 1.7500 | latency bound:pipeline caused |
