# NPU Benchmark Report
Generated: 2026-10-17 04:42:26

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## EngramGateConvBwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | M | d | seq_len | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bwd-smoke-b1-s16-d256-fp16 | 51.9000 | 0.0604 | 17.7049 | 1.7500 | 4.4119 | 0.0027 | /mnt/workspace/Agentkj/examples/TileOPs/reports/tileops/20261017_043414_656448_EngramGateConvBwdOp/artifacts/msprof/k1_z5yc_enq | {'captured_op_count': 1, 'captured_op_names': ['k1'], 'kernel_name_attempts': ['k1'], 'kernel_name_requested': 'k1', 'kernel_name_resolved': 'k1', 'latency_us': 51.900002, 'max_latency_us': 51.900002, 'min_latency_us': 51.900002, 'op_name': 'k1', 'sample_count': 1} | msprof | 1 | 256 | 16 | float16 |
| bwd-b1-s32-d256-fp16 | 50.8000 | 0.1234 | 23.2258 | 1.7500 | 4.4119 | 0.0054 | /mnt/workspace/Agentkj/examples/TileOPs/reports/tileops/20261017_043414_656448_EngramGateConvBwdOp/artifacts/msprof/k1_rqmy92rj | {'captured_op_count': 1, 'captured_op_names': ['k1'], 'kernel_name_attempts': ['k1'], 'kernel_name_requested': 'k1', 'kernel_name_resolved': 'k1', 'latency_us': 50.799999, 'max_latency_us': 50.799999, 'min_latency_us': 50.799999, 'op_name': 'k1', 'sample_count': 1} | msprof | 1 | 256 | 32 | float16 |
| bwd-b2-s64-d512-fp16 | 59.3200 | 0.8601 | 32.0578 | 1.7500 | 4.4555 | 0.0383 | /mnt/workspace/Agentkj/examples/TileOPs/reports/tileops/20261017_043414_656448_EngramGateConvBwdOp/artifacts/msprof/k1_pzyir2z5 | {'captured_op_count': 1, 'captured_op_names': ['k1'], 'kernel_name_attempts': ['k1'], 'kernel_name_requested': 'k1', 'kernel_name_resolved': 'k1', 'latency_us': 59.32, 'max_latency_us': 59.32, 'min_latency_us': 59.32, 'op_name': 'k1', 'sample_count': 1} | msprof | 2 | 512 | 64 | float16 |
| bwd-b1-s128-d256-bf16 | 225.0800 | 0.5920 | 171.1079 | 1.7500 | 4.5696 | 0.0271 | /mnt/workspace/Agentkj/examples/TileOPs/reports/tileops/20261017_043414_656448_EngramGateConvBwdOp/artifacts/msprof/k1_lxlhlscu | {'captured_op_count': 1, 'captured_op_names': ['k1'], 'kernel_name_attempts': ['k1'], 'kernel_name_requested': 'k1', 'kernel_name_resolved': 'k1', 'latency_us': 225.080002, 'max_latency_us': 225.080002, 'min_latency_us': 225.080002, 'op_name': 'k1', 'sample_count': 1} | msprof | 1 | 256 | 128 | bfloat16 |
