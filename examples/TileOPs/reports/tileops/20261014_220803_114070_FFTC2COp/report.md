# TileOPs Evaluation Report: FFTC2COp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-14T22:11:09.641647+08:00 |
| evaluation_scope | FFTC2COp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261014_220803_114070_FFTC2COp |
| git_commit | a2357253662babf317ada194080078f928d155b3 |
| correctness_target | tests/ops/test_fft.py |
| benchmark_target | benchmarks/ops/bench_fft.py |

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
| 77.8% | 1 | 9 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| FFTC2COp | 77.8% (7/9) | 1.42e-05 | 3 | 0.00% – 0.08% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### FFTC2COp

| Label | Latency (us) | Ratio (%) [=实测÷单die HBM理论1.64] | Shape / Parameters | DType | Mode | Kernel | AchievedBW (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| fft-64-c64-smoke | 7.1800 | 0.01 | batch_shape=[]; shape=[64] | complex64 | msprof | main | 1.7500 |
| fft-4k-c64-unbatched | 717.8000 | 0.00 | batch_shape=[]; shape=[4096] | complex64 | msprof | main | 1.7500 |
| fft-4k-c64-b64 | 1496.6801 | 0.08 | batch_shape=[64]; shape=[64, 4096] | complex64 | msprof | main | 1.7500 |

## Warnings

- FFTC2COp/FFTC2COp-2399f4b212b8: missing tileops candidate record
- FFTC2COp/FFTC2COp-45b7f253137e: missing tileops candidate record
- FFTC2COp/FFTC2COp-ae4a94f2c683: missing tileops candidate record
