# NPU Benchmark Report
Generated: 2026-10-25 23:56:59

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## RopeLongRopeOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | bandwidth_tbs | tflops | layout | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longrope-1d-256-d64-smoke | 3.2400 | 1.3032 | 4.2596 | 1.7500 | 3.7061 | 0.0483 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261025_234635_085088_RopeLongRopeOp/artifacts/msprof/main_d1h66xf9 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.24, 'max_latency_us': 3.24, 'min_latency_us': 3.24, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | float16 | [256, 64] |
| longrope-1d-256-d64-smoke | 3.1400 | 1.4101 | 4.5651 | 1.7500 | 3.7480 | 0.0529 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261025_234635_085088_RopeLongRopeOp/artifacts/msprof/main_e638two0 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.14, 'max_latency_us': 3.14, 'min_latency_us': 3.14, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | bfloat16 | [256, 64] |
| longrope-1d-256-d64-smoke | 3.0000 | 2.0992 | 0.9991 | 1.7500 | 3.7592 | 0.0367 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261025_234635_085088_RopeLongRopeOp/artifacts/msprof/main_h_tq95sn | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.0, 'max_latency_us': 3.0, 'min_latency_us': 3.0, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 1024} | float32 | [256, 64] |
| longrope-1d-8k-d128-bf16 | 6.3000 | 38.0447 | 1.7039 | 1.7500 | 4.1250 | 1.1344 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261025_234635_085088_RopeLongRopeOp/artifacts/msprof/main_1llxh1an | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 6.3, 'max_latency_us': 6.3, 'min_latency_us': 6.3, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | bfloat16 | [8192, 128] |
| longrope-2d-b1-s8k-h32-d128-fp16 | 81.5800 | 54.7424 | 2.6250 | 1.7500 | 4.0063 | 2.1931 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261025_234635_085088_RopeLongRopeOp/artifacts/msprof/main_1b_pdktn | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 81.580002, 'max_latency_us': 81.580002, 'min_latency_us': 81.580002, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 2d | {'block_size': 2048} | float16 | [1, 8192, 32, 128] |
| longrope-1d-2k-d128-fp16 | 3.9200 | 15.2872 | 1.6001 | 1.7500 | 4.0873 | 0.4281 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261025_234635_085088_RopeLongRopeOp/artifacts/msprof/main_5_aoxcph | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.92, 'max_latency_us': 3.92, 'min_latency_us': 3.92, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | float16 | [2048, 128] |

### torch-ref

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | bandwidth_tbs | tflops | layout | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longrope-1d-256-d64-smoke | 366.4240 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0003 | 0.0002 | 1d | {'block_size': 2048} | float16 | [256, 64] |
| longrope-1d-256-d64-smoke | 380.1480 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0003 | 0.0002 | 1d | {'block_size': 2048} | bfloat16 | [256, 64] |
| longrope-1d-256-d64-smoke | 360.7720 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0005 | 0.0002 | 1d | {'block_size': 1024} | float32 | [256, 64] |
| longrope-1d-8k-d128-bf16 | 368.1440 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0171 | 0.0114 | 1d | {'block_size': 2048} | bfloat16 | [8192, 128] |
| longrope-2d-b1-s8k-h32-d128-fp16 | 519.9880 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.2622 | 0.2581 | 2d | {'block_size': 2048} | float16 | [1, 8192, 32, 128] |
| longrope-1d-2k-d128-fp16 | 381.8280 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0041 | 0.0027 | 1d | {'block_size': 2048} | float16 | [2048, 128] |
