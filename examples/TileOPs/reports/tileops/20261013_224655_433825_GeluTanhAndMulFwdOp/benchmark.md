# NPU Benchmark Report
Generated: 2026-10-13 22:58:50

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## GeluTanhAndMulFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | M | N | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke | 2.6800 | 3.7645 | 26.9767 | 1.7500 | 4.4152 | 0.1662 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261013_224655_433825_GeluTanhAndMulFwdOp/artifacts/msprof/main_qh7rnx04 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2.68, 'max_latency_us': 2.68, 'min_latency_us': 2.68, 'op_name': 'main', 'sample_count': 1} | msprof | 16 | 128 | float32 |
| smoke | 2.4400 | 5.0273 | 128.9697 | 1.7500 | 4.4410 | 0.2233 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261013_224655_433825_GeluTanhAndMulFwdOp/artifacts/msprof/main_npxktye8 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2.44, 'max_latency_us': 2.44, 'min_latency_us': 2.44, 'op_name': 'main', 'sample_count': 1} | msprof | 16 | 128 | bfloat16 |
| llama-8b-ffn-prefill | 215.6200 | 47.8345 | 3.8985 | 1.7500 | 4.4390 | 2.1234 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261013_224655_433825_GeluTanhAndMulFwdOp/artifacts/msprof/main_x6jghp52 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 215.619995, 'max_latency_us': 215.619995, 'min_latency_us': 215.619995, 'op_name': 'main', 'sample_count': 1} | msprof | 2048 | 14336 | float16 |
| llama-8b-ffn-prefill | 220.2200 | 49.7286 | 4.0696 | 1.7500 | 4.4488 | 2.2123 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261013_224655_433825_GeluTanhAndMulFwdOp/artifacts/msprof/main_9kfgtef5 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 220.220001, 'max_latency_us': 220.220001, 'min_latency_us': 220.220001, 'op_name': 'main', 'sample_count': 1} | msprof | 2048 | 14336 | bfloat16 |
| llama-8b-ffn-decode | 2.1800 | 2.4618 | 4.1470 | 1.7500 | 4.4410 | 0.1093 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261013_224655_433825_GeluTanhAndMulFwdOp/artifacts/msprof/main_rjgeafy1 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2.18, 'max_latency_us': 2.18, 'min_latency_us': 2.18, 'op_name': 'main', 'sample_count': 1} | msprof | 1 | 14336 | bfloat16 |
| qwen3-8b-swiglu-b32 | 5.8600 | 23.5912 | 3.8997 | 1.7500 | 4.4373 | 1.0468 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261013_224655_433825_GeluTanhAndMulFwdOp/artifacts/msprof/main_xet5hbrh | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 5.86, 'max_latency_us': 5.86, 'min_latency_us': 5.86, 'op_name': 'main', 'sample_count': 1} | msprof | 32 | 12288 | float16 |
| qwen3-8b-swiglu-b32 | 6.0200 | 24.3817 | 4.1497 | 1.7500 | 4.4472 | 1.0843 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261013_224655_433825_GeluTanhAndMulFwdOp/artifacts/msprof/main_e9vxdqwi | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 6.02, 'max_latency_us': 6.02, 'min_latency_us': 6.02, 'op_name': 'main', 'sample_count': 1} | msprof | 32 | 12288 | bfloat16 |
