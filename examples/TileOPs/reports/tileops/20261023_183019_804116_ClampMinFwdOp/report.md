# TileOPs Evaluation Report: ClampMinFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-23T18:40:52.724085+08:00 |
| evaluation_scope | ClampMinFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261023_183019_804116_ClampMinFwdOp |
| git_commit | 60cfa4e4e811e9deaa895ae56181d9bc5be488c4 |
| correctness_target | tests/ops/test_clamp_min.py |
| benchmark_target | benchmarks/ops/bench_clamp_min.py |

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
| 100.0% | 1 | 14 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| ClampMinFwdOp | 100.0% (14/14) | 0.00e+00 | 7 | 70.08% – 77.31% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### ClampMinFwdOp

| Label | Latency (us) | Ratio (%) | AI (Ops/Byte) | Bound | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) | Advice |
|---|---:|---:|---:|:---:|---|---|---|---|---:|---|
| 16M-min-only | 27.3600 | 70.08 | 0.5002 | ? | [4096, 4096] | float16 | msprof | main | 1.7500 | latency bound:memory caused |
| 16M-min-only | 26.8800 | 71.33 | 1.5001 | ? | [4096, 4096] | bfloat16 | msprof | main | 1.7500 | latency bound:memory caused |
| 16M-min-only | 49.6000 | 77.31 | 0.2501 | ? | [4096, 4096] | float32 | msprof | main | 1.7500 | latency bound:memory caused |
| 64M-min-only | 126.3000 | 74.88 | 0.4055 | ? | [8192, 8192] | float16 | msprof | main | 1.7500 | latency bound:memory caused |
| 64M-min-only | 125.8400 | 75.87 | 1.2050 | ? | [8192, 8192] | bfloat16 | msprof | main | 1.7500 | latency bound:memory caused |
| 256M-min-only | 760.7400 | 73.08 | 0.2759 | ? | [16384, 16384] | float16 | msprof | main | 1.7500 | latency bound:memory caused |
| 256M-min-only | 760.7600 | 73.03 | 0.8283 | ? | [16384, 16384] | bfloat16 | msprof | main | 1.7500 | latency bound:memory caused |
