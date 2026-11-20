# NPU Benchmark Report
Generated: 2026-10-16 21:13:50

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## EngramDecodeOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | batch | conv_kernel_size | d | d_mem | dilation | max_conv_len | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode-smoke-b1-dmem256-d256-fp16 | 21.7000 | 0.7173 | 1.5422 | 1.7500 | 4.4464 | 0.0194 | /mnt/workspace/Agentkj/examples/TileOPs/reports/tileops/20261016_210815_236774_EngramDecodeOp/artifacts/msprof/main_rlitnl4i | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 21.700001, 'max_latency_us': 21.700001, 'min_latency_us': 21.700001, 'op_name': 'main', 'sample_count': 1} | msprof | 1 | 4 | 256 | 256 | 3 | 9 | float16 |
| decode-b1-dmem512-d256-fp16 | 31.4800 | 0.9728 | 1.5442 | 1.7500 | 4.4542 | 0.0263 | /mnt/workspace/Agentkj/examples/TileOPs/reports/tileops/20261016_210815_236774_EngramDecodeOp/artifacts/msprof/main_tdffmfml | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 31.48, 'max_latency_us': 31.48, 'min_latency_us': 31.48, 'op_name': 'main', 'sample_count': 1} | msprof | 1 | 4 | 256 | 512 | 3 | 12 | float16 |
| decode-b4-dmem1024-d512-fp16 | 78.8000 | 3.6182 | 5.8804 | 1.7500 | 4.5283 | 0.1638 | /mnt/workspace/Agentkj/examples/TileOPs/reports/tileops/20261016_210815_236774_EngramDecodeOp/artifacts/msprof/main_ezcgbwgd | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 78.800003, 'max_latency_us': 78.800003, 'min_latency_us': 78.800003, 'op_name': 'main', 'sample_count': 1} | msprof | 4 | 4 | 512 | 1024 | 5 | 20 | float16 |
| decode-b8-dmem512-d256-bf16 | 31.3800 | 4.7397 | 10.8712 | 1.7500 | 4.4543 | 0.2111 | /mnt/workspace/Agentkj/examples/TileOPs/reports/tileops/20261016_210815_236774_EngramDecodeOp/artifacts/msprof/main_4xo05l75 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 31.379999, 'max_latency_us': 31.379999, 'min_latency_us': 31.379999, 'op_name': 'main', 'sample_count': 1} | msprof | 8 | 4 | 256 | 512 | 3 | 18 | bfloat16 |
