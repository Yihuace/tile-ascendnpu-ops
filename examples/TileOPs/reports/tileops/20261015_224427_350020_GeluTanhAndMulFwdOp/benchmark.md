# NPU Benchmark Report
Generated: 2026-10-15 22:56:54

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
| smoke | 2.4800 | 4.6416 | 27.9690 | 1.7500 | 4.0120 | 0.1862 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_224427_350020_GeluTanhAndMulFwdOp/artifacts/msprof/main_gk84npg9 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2.48, 'max_latency_us': 2.48, 'min_latency_us': 2.48, 'op_name': 'main', 'sample_count': 1} | msprof | 16 | 128 | float32 |
| smoke | 2.3800 | 4.8553 | 121.2121 | 1.7500 | 4.4308 | 0.2151 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_224427_350020_GeluTanhAndMulFwdOp/artifacts/msprof/main_y_nkuopi | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2.38, 'max_latency_us': 2.38, 'min_latency_us': 2.38, 'op_name': 'main', 'sample_count': 1} | msprof | 16 | 128 | bfloat16 |
| llama-8b-ffn-prefill | 200.1400 | 48.3507 | 3.6485 | 1.7500 | 4.4278 | 2.1409 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_224427_350020_GeluTanhAndMulFwdOp/artifacts/msprof/main_u8victoo | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 200.139999, 'max_latency_us': 200.139999, 'min_latency_us': 200.139999, 'op_name': 'main', 'sample_count': 1} | msprof | 2048 | 14336 | float16 |
| llama-8b-ffn-prefill | 204.8200 | 50.3567 | 3.8985 | 1.7500 | 4.4390 | 2.2353 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_224427_350020_GeluTanhAndMulFwdOp/artifacts/msprof/main_i6wbgk3r | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 204.820007, 'max_latency_us': 204.820007, 'min_latency_us': 204.820007, 'op_name': 'main', 'sample_count': 1} | msprof | 2048 | 14336 | bfloat16 |
| llama-8b-ffn-decode | 2.2400 | 2.2569 | 3.8976 | 1.7500 | 4.4308 | 0.1000 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_224427_350020_GeluTanhAndMulFwdOp/artifacts/msprof/main_9iu4qcri | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2.24, 'max_latency_us': 2.24, 'min_latency_us': 2.24, 'op_name': 'main', 'sample_count': 1} | msprof | 1 | 14336 | bfloat16 |
| qwen3-8b-swiglu-b32 | 5.7000 | 22.7563 | 3.6498 | 1.7500 | 4.4260 | 1.0072 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_224427_350020_GeluTanhAndMulFwdOp/artifacts/msprof/main_v0e3z9oc | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 5.7, 'max_latency_us': 5.7, 'min_latency_us': 5.7, 'op_name': 'main', 'sample_count': 1} | msprof | 32 | 12288 | float16 |
| qwen3-8b-swiglu-b32 | 5.8400 | 23.6720 | 3.8997 | 1.7500 | 4.4373 | 1.0504 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_224427_350020_GeluTanhAndMulFwdOp/artifacts/msprof/main_7fumq4sy | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 5.84, 'max_latency_us': 5.84, 'min_latency_us': 5.84, 'op_name': 'main', 'sample_count': 1} | msprof | 32 | 12288 | bfloat16 |
