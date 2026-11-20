# TileOPs Evaluation Report: GemmOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-21T17:22:34.099638+08:00 |
| evaluation_scope | GemmOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261021_171401_517624_GemmOp |
| git_commit | 3d212b67b10f1c528ecaf74abc45b1bbf2423c47 |
| correctness_target | tests/ops/test_gemm.py |
| benchmark_target | benchmarks/ops/bench_gemm.py |

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
| 100.0% | 1 | 17 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| GemmOp | 100.0% (17/17) | 7.73e-02 | 14 | 31.72% – 74.12% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### GemmOp

| Label | Latency (us) | Ratio (%) | Mem Roofline Ratio (%) | Compute Roofline Ratio (%) | Achieved BW Util (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|
| square-1k-nn | 19.4396 | 40.86 | 11.99 | 40.86 | 17.98 | {"k": 1024, "m": 1024, "n": 1024, "trans_a": false, "trans_b": false} | float16 | msprof | main | 1.8000 |
| square-1k-nn | 18.9396 | 41.94 | 12.30 | 41.94 | 18.45 | {"k": 1024, "m": 1024, "n": 1024, "trans_a": false, "trans_b": false} | bfloat16 | msprof | main | 1.8000 |
| ds-v3-decode-gate-up | 56.2389 | 31.72 | 31.72 | 25.49 | 32.26 | {"k": 7168, "m": 128, "n": 2112, "trans_a": false, "trans_b": true} | bfloat16 | msprof | main | 1.8000 |
| ds-v3-decode-down | 37.9392 | 43.76 | 43.76 | 36.64 | 46.45 | {"k": 2048, "m": 128, "n": 7168, "trans_a": false, "trans_b": true} | bfloat16 | msprof | main | 1.8000 |
| ds-v3-prefill-gate-up | 718.4257 | 63.86 | 7.09 | 63.86 | 8.22 | {"k": 7168, "m": 4096, "n": 2112, "trans_a": false, "trans_b": true} | bfloat16 | msprof | main | 1.8000 |
| ds-v3-prefill-down | 642.5071 | 69.24 | 4.35 | 69.24 | 9.07 | {"k": 2048, "m": 4096, "n": 7168, "trans_a": false, "trans_b": true} | bfloat16 | msprof | main | 1.8000 |
| ds-v3-prefill-attn-proj | 1200.3359 | 74.12 | 7.42 | 74.12 | 6.99 | {"k": 7168, "m": 4096, "n": 4096, "trans_a": false, "trans_b": true} | float16 | msprof | main | 1.8000 |
| ds-v3-prefill-attn-proj | 1200.5560 | 74.11 | 6.69 | 74.11 | 6.99 | {"k": 7168, "m": 4096, "n": 4096, "trans_a": false, "trans_b": true} | bfloat16 | msprof | main | 1.8000 |
| k-dominant-7168x16384 | 5346.3130 | 66.57 | 33.13 | 66.57 | 4.45 | {"k": 16384, "m": 4096, "n": 7168, "trans_a": false, "trans_b": true} | bfloat16 | msprof | main | 1.8000 |
| wide-n-24576 | 1800.2040 | 63.54 | 13.17 | 63.54 | 8.93 | {"k": 1536, "m": 4096, "n": 24576, "trans_a": false, "trans_b": true} | bfloat16 | msprof | main | 1.8000 |
| mid-m16-attn | 61.1588 | 53.55 | 53.55 | 5.68 | 53.67 | {"k": 7168, "m": 16, "n": 4096, "trans_a": false, "trans_b": true} | bfloat16 | msprof | main | 1.8000 |
| mid-m32-attn | 60.2588 | 54.56 | 54.56 | 11.53 | 54.80 | {"k": 7168, "m": 32, "n": 4096, "trans_a": false, "trans_b": true} | bfloat16 | msprof | main | 1.8000 |
| mid-m64-down | 33.4793 | 49.16 | 49.16 | 20.76 | 50.68 | {"k": 2048, "m": 64, "n": 7168, "trans_a": false, "trans_b": true} | bfloat16 | msprof | main | 1.8000 |
| mid-m96-gate-up | 49.9990 | 35.17 | 35.17 | 21.50 | 35.62 | {"k": 7168, "m": 96, "n": 2112, "trans_a": false, "trans_b": true} | bfloat16 | msprof | main | 1.8000 |
