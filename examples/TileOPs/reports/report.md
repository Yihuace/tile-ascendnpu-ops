# TileOPs Evaluation Report: all

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-15T07:25:49.199844+00:00 |
| evaluation_scope | all |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261015_070129_353134_all |
| git_commit | 981a6ba9e2b15cf4b8a795fafc43cb5626820aab |
| correctness_target | tests/ops |
| benchmark_target | benchmarks/ops |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B2C x 16 |
| cpu | x86_64 |
| cann | 8.5.0 |
| driver | 26.0.rc1 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1 |
| tilelang | 0.1.2+ubuntu.22.4.npuir |
| python | 3.11.15 |
| os | Linux-5.15.0-191-generic-x86_64-with-glibc2.35 |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 7 | 112 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| AdaLayerNormFwdOp | 100.0% (22/22) | 1.13e-02 | 5 | 0.44% – 68.17% |
| ArgmaxFwdOp | 100.0% (42/42) | 0.00e+00 | 5 | 4.00% – 40.56% |
| LerpTensorFwdOp | 100.0% (8/8) | 5.21e-08 | 7 | 65.95% – 78.00% |
| LogSumExpFwdOp | 100.0% (23/23) | 3.67e-08 | 6 | 3.21% – 36.70% |
| MishFwdOp | 100.0% (7/7) | 2.45e-05 | 6 | 55.26% – 67.45% |
| MultiHeadAttentionFwdOp | 100.0% (6/6) | 8.54e-04 | 8 | 14.36% – 23.58% |
| SSDChunkScanFwdOp | 100.0% (4/4) | 2.19e-04 | 11 | 0.14% – 21.11% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### AdaLayerNormFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| dit-xl-2 | 8.8200 | 44.23 | {"m": 1024, "n": 1152} | float16 | msprof | main | 1.8000 |
| dit-xl-2 | 8.8200 | 44.23 | {"m": 1024, "n": 1152} | bfloat16 | msprof | main | 1.8000 |
| llama-3.1-8b-prefill | 41.2800 | 67.74 | {"m": 2048, "n": 4096} | float16 | msprof | main | 1.8000 |
| llama-3.1-8b-prefill | 41.0200 | 68.17 | {"m": 2048, "n": 4096} | bfloat16 | msprof | main | 1.8000 |
| llama-3.1-8b-decode | 3.1800 | 0.44 | {"m": 1, "n": 4096} | bfloat16 | msprof | main | 1.8000 |

### ArgmaxFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| lm-head-argmax | 11.3800 | 4.00 | [4, 102400] | float16 | msprof | argreduce_nsplit_fused | 1.8000 |
| lm-head-argmax | 11.3000 | 4.40 | [4, 102400] | bfloat16 | msprof | argreduce_nsplit_fused | 1.8000 |
| hidden-state-argmax | 22.9800 | 40.56 | [2048, 4096] | float16 | msprof | main | 1.8000 |
| hidden-state-argmax | 44.2800 | 21.65 | [2048, 4096] | bfloat16 | msprof | main | 1.8000 |
| 3d-non-last-axis-argmax | 12.7400 | 22.18 | [4, 128, 4096] | float16 | msprof | main | 1.8000 |

### LerpTensorFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| elementwise-16m | 73.5000 | 76.09 | [4096, 4096] | float16 | msprof | main | 1.8000 |
| elementwise-16m | 71.7000 | 78.00 | [4096, 4096] | bfloat16 | msprof | main | 1.8000 |
| elementwise-16m | 160.9000 | 76.37 | [4096, 4096] | float32 | msprof | main | 1.8000 |
| elementwise-64m | 395.2800 | 69.04 | [8192, 8192] | float16 | msprof | main | 1.8000 |
| elementwise-64m | 390.7000 | 69.27 | [8192, 8192] | bfloat16 | msprof | main | 1.8000 |
| elementwise-256m | 1700.2200 | 68.50 | [16384, 16384] | float16 | msprof | main | 1.8000 |
| elementwise-256m | 1766.5200 | 65.95 | [16384, 16384] | bfloat16 | msprof | main | 1.8000 |

### LogSumExpFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| attn-weights-4k | 16.4000 | 28.42 | [32, 32, 4096] | float16 | msprof | main | 1.8000 |
| attn-weights-4k | 17.3200 | 27.00 | [32, 32, 4096] | bfloat16 | msprof | main | 1.8000 |
| attn-weights-32k | 101.6000 | 36.70 | [32, 32, 32768] | bfloat16 | msprof | main | 1.8000 |
| lm-head-logits | 14.1800 | 3.21 | [4, 102400] | float16 | msprof | main | 1.8000 |
| lm-head-logits | 14.1000 | 3.23 | [4, 102400] | bfloat16 | msprof | main | 1.8000 |
| 3d-multidim-reduce | 10.3000 | 22.62 | [4, 128, 4096] | float16 | msprof | main | 1.8000 |

### MishFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| yolo-p3 | 75.6200 | 62.89 | [16, 256, 80, 80] | float16 | msprof | main | 1.8000 |
| yolo-p3 | 77.5400 | 67.45 | [16, 256, 80, 80] | bfloat16 | msprof | main | 1.8000 |
| yolo-p4 | 40.4600 | 58.77 | [16, 512, 40, 40] | float16 | msprof | main | 1.8000 |
| yolo-p4 | 41.1600 | 63.53 | [16, 512, 40, 40] | bfloat16 | msprof | main | 1.8000 |
| fc-wide | 27.5400 | 55.26 | [2048, 4096] | float16 | msprof | main | 1.8000 |
| fc-wide | 29.3600 | 57.00 | [2048, 4096] | bfloat16 | msprof | main | 1.8000 |

### MultiHeadAttentionFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| llama-3.1-8b-short | 285.8400 | 14.72 | {"batch": 4, "causal": true, "dim": 128, "heads": 32, "seq_len": 512} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-8b-short | 289.1200 | 14.57 | {"batch": 4, "causal": true, "dim": 128, "heads": 32, "seq_len": 512} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-8b-long | 975.8600 | 23.58 | {"batch": 2, "causal": true, "dim": 128, "heads": 32, "seq_len": 2048} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-8b-long | 1033.9200 | 22.28 | {"batch": 2, "causal": true, "dim": 128, "heads": 32, "seq_len": 2048} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-70b-short | 292.3200 | 14.40 | {"batch": 2, "causal": true, "dim": 128, "heads": 64, "seq_len": 512} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-70b-short | 293.4400 | 14.36 | {"batch": 2, "causal": true, "dim": 128, "heads": 64, "seq_len": 512} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-70b-long | 998.4600 | 23.05 | {"batch": 1, "causal": true, "dim": 128, "heads": 64, "seq_len": 2048} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-70b-long | 1020.5800 | 22.57 | {"batch": 1, "causal": true, "dim": 128, "heads": 64, "seq_len": 2048} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |

### SSDChunkScanFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| b1-c2-L64-h4-p64-n32-fp16 | 30.5200 | 0.14 | {"batch": 1, "chunk_len": 64, "d_head": 64, "d_state": 32, "n_groups": 1, "n_heads": 4, "num_chunks": 2} | float16 | msprof | main | 1.8000 |
| b2-c4-L64-h8-p64-n64-fp16 | 33.5000 | 1.38 | {"batch": 2, "chunk_len": 64, "d_head": 64, "d_state": 64, "n_groups": 2, "n_heads": 8, "num_chunks": 4} | float16 | msprof | main | 1.8000 |
| b1-c2-L128-h4-p128-n32-bf16 | 35.6600 | 0.28 | {"batch": 1, "chunk_len": 128, "d_head": 128, "d_state": 32, "n_groups": 1, "n_heads": 4, "num_chunks": 2} | bfloat16 | msprof | main | 1.8000 |
| b2-c2-L64-h4-p64-n32-bf16 | 31.9200 | 0.30 | {"batch": 2, "chunk_len": 64, "d_head": 64, "d_state": 32, "n_groups": 2, "n_heads": 4, "num_chunks": 2} | bfloat16 | msprof | main | 1.8000 |
| latency-130m-4k | 112.7000 | 9.33 | {"batch": 1, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 24, "num_chunks": 16} | float16 | msprof | main | 1.8000 |
| serving-130m-4k | 831.1600 | 12.52 | {"batch": 8, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 24, "num_chunks": 16} | float16 | msprof | main | 1.8000 |
| longctx-130m-32k | 2942.8198 | 20.06 | {"batch": 4, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 24, "num_chunks": 128} | float16 | msprof | main | 1.8000 |
| latency-2p7b-4k | 307.2600 | 11.41 | {"batch": 1, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 80, "num_chunks": 16} | float16 | msprof | main | 1.8000 |
| serving-2p7b-4k | 1183.8000 | 17.04 | {"batch": 4, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 80, "num_chunks": 16} | float16 | msprof | main | 1.8000 |
| longctx-2p7b-32k | 4671.5601 | 21.11 | {"batch": 2, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 80, "num_chunks": 128} | float16 | msprof | main | 1.8000 |
| throughput-2p7b-2k | 585.6800 | 12.38 | {"batch": 4, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 80, "num_chunks": 8} | float16 | msprof | main | 1.8000 |
