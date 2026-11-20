# NPU Benchmark Report
Generated: 2026-10-26 00:15:46

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## RopeLlama31Op

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | bandwidth_tbs | tflops | layout | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama31-1d-256-d64-smoke | 3.1400 | 1.3447 | 4.3661 | 1.7500 | 3.7061 | 0.0498 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261026_000429_461405_RopeLlama31Op/artifacts/msprof/main_ge8j7u4g | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.14, 'max_latency_us': 3.14, 'min_latency_us': 3.14, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | float16 | [256, 64] |
| llama31-1d-256-d64-smoke | 3.1000 | 1.4283 | 4.4862 | 1.7500 | 3.7480 | 0.0535 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261026_000429_461405_RopeLlama31Op/artifacts/msprof/main_jnle_lxl | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.1, 'max_latency_us': 3.1, 'min_latency_us': 3.1, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | bfloat16 | [256, 64] |
| llama31-1d-256-d64-smoke | 2.9800 | 2.1084 | 1.0015 | 1.7500 | 3.7592 | 0.0370 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261026_000429_461405_RopeLlama31Op/artifacts/msprof/main_3jjno_j2 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2.98, 'max_latency_us': 2.98, 'min_latency_us': 2.98, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 1024} | float32 | [256, 64] |
| llama31-1d-8k-d128-bf16 | 6.2800 | 38.1659 | 1.7039 | 1.7500 | 4.1250 | 1.1380 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261026_000429_461405_RopeLlama31Op/artifacts/msprof/main_hdkfyt6e | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 6.28, 'max_latency_us': 6.28, 'min_latency_us': 6.28, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | bfloat16 | [8192, 128] |
| llama31-2d-b1-s8k-h32-d128-fp16 | 81.5200 | 54.7827 | 2.6250 | 1.7500 | 4.0063 | 2.1947 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261026_000429_461405_RopeLlama31Op/artifacts/msprof/main_lxao0k0v | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 81.519997, 'max_latency_us': 81.519997, 'min_latency_us': 81.519997, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 2d | {'block_size': 2048} | float16 | [1, 8192, 32, 128] |
| llama31-1d-2k-d128-fp16 | 3.9800 | 15.0568 | 1.6001 | 1.7500 | 4.0873 | 0.4216 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261026_000429_461405_RopeLlama31Op/artifacts/msprof/main_j8g58ev0 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.98, 'max_latency_us': 3.98, 'min_latency_us': 3.98, 'op_name': 'main', 'sample_count': 1} | msprof | N/A | N/A | 1d | {'block_size': 2048} | float16 | [2048, 128] |

### torch-ref

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | bandwidth_tbs | tflops | layout | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama31-1d-256-d64-smoke | 370.9980 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0003 | 0.0002 | 1d | {'block_size': 2048} | float16 | [256, 64] |
| llama31-1d-256-d64-smoke | 486.8000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0002 | 0.0001 | 1d | {'block_size': 2048} | bfloat16 | [256, 64] |
| llama31-1d-256-d64-smoke | 362.3780 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0005 | 0.0002 | 1d | {'block_size': 1024} | float32 | [256, 64] |
| llama31-1d-8k-d128-bf16 | 357.1400 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0176 | 0.0117 | 1d | {'block_size': 2048} | bfloat16 | [8192, 128] |
| llama31-2d-b1-s8k-h32-d128-fp16 | 547.5160 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.2490 | 0.2451 | 2d | {'block_size': 2048} | float16 | [1, 8192, 32, 128] |
| llama31-1d-2k-d128-fp16 | 361.6900 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | events | 0.0043 | 0.0029 | 1d | {'block_size': 2048} | float16 | [2048, 128] |
