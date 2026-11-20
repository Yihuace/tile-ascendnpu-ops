# NPU Benchmark Report
Generated: 2026-10-17 17:48:15

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## RopeNeoxOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | bandwidth_tbs | tflops | layout | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neox-1d-256-d64-smoke | 3.0600 | 1.3798 | 4.2895 | 1.7500 | 3.7061 | 0.0511 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_4f04z962 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.06, 'max_latency_us': 3.06, 'min_latency_us': 3.06, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | float16 | [256, 64] |
| neox-1d-256-d64-smoke | 3.1000 | 1.4283 | 4.5813 | 1.7500 | 3.7480 | 0.0535 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_2q8pzhlp | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.1, 'max_latency_us': 3.1, 'min_latency_us': 3.1, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | bfloat16 | [256, 64] |
| neox-1d-256-d64-smoke | 3.1200 | 2.0185 | 0.9991 | 1.7500 | 3.7592 | 0.0353 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_vaicjjae | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.12, 'max_latency_us': 3.12, 'min_latency_us': 3.12, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 1024} | float32 | [256, 64] |
| neox-1d-2k-d64-fp16 | 3.5200 | 8.3470 | 2.0986 | 1.7500 | 3.8461 | 0.3065 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_6l8jbafj | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.52, 'max_latency_us': 3.52, 'min_latency_us': 3.52, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | float16 | [2048, 64] |
| neox-1d-4k-d128-bf16 | 4.8200 | 24.8640 | 1.7044 | 1.7500 | 4.1238 | 0.7416 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_ik79mcoe | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 4.82, 'max_latency_us': 4.82, 'min_latency_us': 4.82, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | bfloat16 | [4096, 128] |
| neox-2d-b2-s2k-h32-d128-fp16 | 45.4800 | 48.9109 | 2.5156 | 1.7500 | 3.8242 | 1.8705 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_5bjazq4r | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 45.48, 'max_latency_us': 45.48, 'min_latency_us': 45.48, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 2d | {'block_size': 2048} | float16 | [2, 2048, 32, 128] |

### torch-ref

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | bandwidth_tbs | tflops | layout | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neox-1d-256-d64-smoke | 373.7220 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0003 | 0.0002 | 1d | {'block_size': 2048} | float16 | [256, 64] |
| neox-1d-256-d64-smoke | 354.2000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0003 | 0.0002 | 1d | {'block_size': 2048} | bfloat16 | [256, 64] |
| neox-1d-256-d64-smoke | 369.0020 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0005 | 0.0002 | 1d | {'block_size': 1024} | float32 | [256, 64] |
| neox-1d-2k-d64-fp16 | 394.2000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0020 | 0.0013 | 1d | {'block_size': 2048} | float16 | [2048, 64] |
| neox-1d-4k-d128-bf16 | 380.2760 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0083 | 0.0055 | 1d | {'block_size': 2048} | bfloat16 | [4096, 128] |
| neox-2d-b2-s2k-h32-d128-fp16 | 372.8940 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.1814 | 0.1800 | 2d | {'block_size': 2048} | float16 | [2, 2048, 32, 128] |

## RopeLongRopeOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | bandwidth_tbs | tflops | layout | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longrope-1d-256-d64-smoke | 3.1200 | 1.3533 | 4.1582 | 1.7500 | 3.7061 | 0.0502 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_dpkv5_64 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.12, 'max_latency_us': 3.12, 'min_latency_us': 3.12, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | float16 | [256, 64] |
| longrope-1d-256-d64-smoke | 3.0600 | 1.4470 | 4.5813 | 1.7500 | 3.7480 | 0.0542 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_nyazmtfh | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.06, 'max_latency_us': 3.06, 'min_latency_us': 3.06, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | bfloat16 | [256, 64] |
| longrope-1d-256-d64-smoke | 3.0000 | 2.1016 | 0.9980 | 1.7500 | 3.7592 | 0.0367 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_p0t7eq2c | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.0, 'max_latency_us': 3.0, 'min_latency_us': 3.0, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 1024} | float32 | [256, 64] |
| longrope-1d-8k-d128-bf16 | 6.1600 | 38.9094 | 1.7039 | 1.7500 | 4.1250 | 1.1602 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_nv5it2rn | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 6.16, 'max_latency_us': 6.16, 'min_latency_us': 6.16, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | bfloat16 | [8192, 128] |
| longrope-2d-b1-s8k-h32-d128-fp16 | 81.1800 | 55.0122 | 2.6250 | 1.7500 | 4.0063 | 2.2039 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_tun2wc4d | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 81.18, 'max_latency_us': 81.18, 'min_latency_us': 81.18, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 2d | {'block_size': 2048} | float16 | [1, 8192, 32, 128] |
| longrope-1d-2k-d128-fp16 | 4.1400 | 14.4749 | 1.6001 | 1.7500 | 4.0873 | 0.4053 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_p18e12zn | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 4.14, 'max_latency_us': 4.14, 'min_latency_us': 4.14, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | float16 | [2048, 128] |

### torch-ref

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | bandwidth_tbs | tflops | layout | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longrope-1d-256-d64-smoke | 365.9740 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0003 | 0.0002 | 1d | {'block_size': 2048} | float16 | [256, 64] |
| longrope-1d-256-d64-smoke | 364.0020 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0003 | 0.0002 | 1d | {'block_size': 2048} | bfloat16 | [256, 64] |
| longrope-1d-256-d64-smoke | 365.5380 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0005 | 0.0002 | 1d | {'block_size': 1024} | float32 | [256, 64] |
| longrope-1d-8k-d128-bf16 | 372.4920 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0169 | 0.0113 | 1d | {'block_size': 2048} | bfloat16 | [8192, 128] |
| longrope-2d-b1-s8k-h32-d128-fp16 | 521.9000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.2612 | 0.2572 | 2d | {'block_size': 2048} | float16 | [1, 8192, 32, 128] |
| longrope-1d-2k-d128-fp16 | 370.6140 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0042 | 0.0028 | 1d | {'block_size': 2048} | float16 | [2048, 128] |

## RopeLlama31Op

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | bandwidth_tbs | tflops | layout | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama31-1d-256-d64-smoke | 3.1200 | 1.3533 | 4.2448 | 1.7500 | 3.7061 | 0.0502 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_8qo4t7tt | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.12, 'max_latency_us': 3.12, 'min_latency_us': 3.12, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | float16 | [256, 64] |
| llama31-1d-256-d64-smoke | 3.1400 | 1.4101 | 4.4862 | 1.7500 | 3.7480 | 0.0529 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_023x_h1r | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.14, 'max_latency_us': 3.14, 'min_latency_us': 3.14, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | bfloat16 | [256, 64] |
| llama31-1d-256-d64-smoke | 3.0200 | 2.0805 | 1.0015 | 1.7500 | 3.7592 | 0.0365 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_vy5hay2i | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.02, 'max_latency_us': 3.02, 'min_latency_us': 3.02, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 1024} | float32 | [256, 64] |
| llama31-1d-8k-d128-bf16 | 6.2600 | 38.2878 | 1.7039 | 1.7500 | 4.1250 | 1.1416 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_wx9tnlu2 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 6.26, 'max_latency_us': 6.26, 'min_latency_us': 6.26, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | bfloat16 | [8192, 128] |
| llama31-2d-b1-s8k-h32-d128-fp16 | 81.5400 | 54.7693 | 2.6250 | 1.7500 | 4.0063 | 2.1942 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_5ykkkaki | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 81.540001, 'max_latency_us': 81.540001, 'min_latency_us': 81.540001, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 2d | {'block_size': 2048} | float16 | [1, 8192, 32, 128] |
| llama31-1d-2k-d128-fp16 | 4.2200 | 14.2005 | 1.6001 | 1.7500 | 4.0873 | 0.3976 | /home/developer/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261017_172054_008103_RopeNeoxOp/artifacts/msprof/main_yl42vua5 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 4.22, 'max_latency_us': 4.22, 'min_latency_us': 4.22, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | float16 | [2048, 128] |

### torch-ref

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | bandwidth_tbs | tflops | layout | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama31-1d-256-d64-smoke | 371.8020 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0003 | 0.0002 | 1d | {'block_size': 2048} | float16 | [256, 64] |
| llama31-1d-256-d64-smoke | 365.6300 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0003 | 0.0002 | 1d | {'block_size': 2048} | bfloat16 | [256, 64] |
| llama31-1d-256-d64-smoke | 341.5220 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0006 | 0.0002 | 1d | {'block_size': 1024} | float32 | [256, 64] |
| llama31-1d-8k-d128-bf16 | 336.1120 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0187 | 0.0125 | 1d | {'block_size': 2048} | bfloat16 | [8192, 128] |
| llama31-2d-b1-s8k-h32-d128-fp16 | 518.2560 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.2630 | 0.2590 | 2d | {'block_size': 2048} | float16 | [1, 8192, 32, 128] |
| llama31-1d-2k-d128-fp16 | 357.7500 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0044 | 0.0029 | 1d | {'block_size': 2048} | float16 | [2048, 128] |
