# NPU Benchmark Report
Generated: 2026-10-17 01:03:27

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## EngramGateConvFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | M | d | seq_len | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fwd-smoke-b1-s16-d256-fp16 | 4.4600 | 0.9915 | 6.7743 | 1.7500 | 4.4314 | 0.0439 | /mnt/workspace/Agentkj/examples/TileOPs/reports/tileops/20261017_005814_531237_EngramGateConvFwdOp/artifacts/msprof/main_kun5wp8e | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 4.46, 'max_latency_us': 4.46, 'min_latency_us': 4.46, 'op_name': 'main', 'sample_count': 1} | msprof | 1 | 256 | 16 | float16 |
| fwd-b1-s32-d256-fp16 | 4.9600 | 1.7832 | 7.4866 | 1.7500 | 4.4314 | 0.0790 | /mnt/workspace/Agentkj/examples/TileOPs/reports/tileops/20261017_005814_531237_EngramGateConvFwdOp/artifacts/msprof/main_5nqe3jcq | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 4.96, 'max_latency_us': 4.96, 'min_latency_us': 4.96, 'op_name': 'main', 'sample_count': 1} | msprof | 1 | 256 | 32 | float16 |
| fwd-b2-s64-d512-fp16 | 7.4600 | 9.1570 | 7.6591 | 1.7500 | 4.4791 | 0.4101 | /mnt/workspace/Agentkj/examples/TileOPs/reports/tileops/20261017_005814_531237_EngramGateConvFwdOp/artifacts/msprof/main_cr1l6ixt | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 7.46, 'max_latency_us': 7.46, 'min_latency_us': 7.46, 'op_name': 'main', 'sample_count': 1} | msprof | 2 | 512 | 64 | float16 |
| fwd-b1-s128-d256-bf16 | 6.3800 | 5.7680 | 8.1742 | 1.7500 | 4.4383 | 0.2560 | /mnt/workspace/Agentkj/examples/TileOPs/reports/tileops/20261017_005814_531237_EngramGateConvFwdOp/artifacts/msprof/main_cww5wkhw | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 6.38, 'max_latency_us': 6.38, 'min_latency_us': 6.38, 'op_name': 'main', 'sample_count': 1} | msprof | 1 | 256 | 128 | bfloat16 |
