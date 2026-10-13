# NPU Benchmark Report
Generated: 2026-10-13 22:07:19

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## GeluAndMulFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | M | N | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke | 3.9200 | 10.4705 | 112.8372 | 1.7500 | 4.5394 | 0.4753 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261013_215524_720318_GeluAndMulFwdOp/artifacts/msprof/main_81spsvhn | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.92, 'max_latency_us': 3.92, 'min_latency_us': 3.92, 'op_name': 'main', 'sample_count': 1} | msprof | 16 | 128 | float32 |
| smoke | 4.0800 | 10.5828 | 464.3636 | 1.7500 | 4.5428 | 0.4808 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261013_215524_720318_GeluAndMulFwdOp/artifacts/msprof/main_puhueoxs | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 4.08, 'max_latency_us': 4.08, 'min_latency_us': 4.08, 'op_name': 'main', 'sample_count': 1} | msprof | 16 | 128 | bfloat16 |
| llama-8b-ffn-prefill | 878.7600 | 43.2093 | 14.6992 | 1.7500 | 4.5464 | 1.9645 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261013_215524_720318_GeluAndMulFwdOp/artifacts/msprof/main_7ppxsxrm | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 878.76001, 'max_latency_us': 878.76001, 'min_latency_us': 878.76001, 'op_name': 'main', 'sample_count': 1} | msprof | 2048 | 14336 | float16 |
| llama-8b-ffn-prefill | 877.6400 | 43.9904 | 14.6870 | 1.7500 | 4.5474 | 2.0004 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261013_215524_720318_GeluAndMulFwdOp/artifacts/msprof/main_h7hs_aus | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 877.640015, 'max_latency_us': 877.640015, 'min_latency_us': 877.640015, 'op_name': 'main', 'sample_count': 1} | msprof | 2048 | 14336 | bfloat16 |
| llama-8b-ffn-decode | 3.9600 | 4.7703 | 14.9315 | 1.7500 | 4.5428 | 0.2167 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261013_215524_720318_GeluAndMulFwdOp/artifacts/msprof/main_e4hjzd0x | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.96, 'max_latency_us': 3.96, 'min_latency_us': 3.96, 'op_name': 'main', 'sample_count': 1} | msprof | 1 | 14336 | bfloat16 |
| qwen3-8b-swiglu-b32 | 15.1200 | 33.6405 | 14.6996 | 1.7500 | 4.5459 | 1.5293 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261013_215524_720318_GeluAndMulFwdOp/artifacts/msprof/main_zaomle35 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 15.12, 'max_latency_us': 15.12, 'min_latency_us': 15.12, 'op_name': 'main', 'sample_count': 1} | msprof | 32 | 12288 | float16 |
| qwen3-8b-swiglu-b32 | 14.9400 | 34.6170 | 14.9496 | 1.7500 | 4.5469 | 1.5740 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261013_215524_720318_GeluAndMulFwdOp/artifacts/msprof/main_yaoa2wtz | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 14.94, 'max_latency_us': 14.94, 'min_latency_us': 14.94, 'op_name': 'main', 'sample_count': 1} | msprof | 32 | 12288 | bfloat16 |
