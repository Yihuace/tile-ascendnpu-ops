# NPU Benchmark Report
Generated: 2026-10-15 22:40:55

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
| smoke | 3.9400 | 10.5922 | 111.8140 | 1.7500 | 4.4240 | 0.4686 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_222824_082803_GeluAndMulFwdOp/artifacts/msprof/main_vir9ed1_ | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.94, 'max_latency_us': 3.94, 'min_latency_us': 3.94, 'op_name': 'main', 'sample_count': 1} | msprof | 16 | 128 | float32 |
| smoke | 4.0200 | 10.3814 | 448.7273 | 1.7500 | 4.5418 | 0.4715 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_222824_082803_GeluAndMulFwdOp/artifacts/msprof/main_oiq589mj | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 4.02, 'max_latency_us': 4.02, 'min_latency_us': 4.02, 'op_name': 'main', 'sample_count': 1} | msprof | 16 | 128 | bfloat16 |
| llama-8b-ffn-prefill | 797.2200 | 46.0053 | 14.1953 | 1.7500 | 4.5455 | 2.0912 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_222824_082803_GeluAndMulFwdOp/artifacts/msprof/main_j9brcjwm | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 797.219971, 'max_latency_us': 797.219971, 'min_latency_us': 797.219971, 'op_name': 'main', 'sample_count': 1} | msprof | 2048 | 14336 | float16 |
| llama-8b-ffn-prefill | 800.2400 | 46.6279 | 14.1919 | 1.7500 | 4.5465 | 2.1199 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_222824_082803_GeluAndMulFwdOp/artifacts/msprof/main_mzhz4a3x | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 800.23999, 'max_latency_us': 800.23999, 'min_latency_us': 800.23999, 'op_name': 'main', 'sample_count': 1} | msprof | 2048 | 14336 | bfloat16 |
| llama-8b-ffn-decode | 3.8000 | 4.8048 | 14.4287 | 1.7500 | 4.5418 | 0.2182 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_222824_082803_GeluAndMulFwdOp/artifacts/msprof/main_kyctclx5 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 3.8, 'max_latency_us': 3.8, 'min_latency_us': 3.8, 'op_name': 'main', 'sample_count': 1} | msprof | 1 | 14336 | bfloat16 |
| qwen3-8b-swiglu-b32 | 13.9200 | 35.2953 | 14.1958 | 1.7500 | 4.5450 | 1.6042 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_222824_082803_GeluAndMulFwdOp/artifacts/msprof/main_vbtjcmdm | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 13.92, 'max_latency_us': 13.92, 'min_latency_us': 13.92, 'op_name': 'main', 'sample_count': 1} | msprof | 32 | 12288 | float16 |
| qwen3-8b-swiglu-b32 | 13.8600 | 36.0638 | 14.4458 | 1.7500 | 4.5460 | 1.6395 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_222824_082803_GeluAndMulFwdOp/artifacts/msprof/main_oev2qknj | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 13.86, 'max_latency_us': 13.86, 'min_latency_us': 13.86, 'op_name': 'main', 'sample_count': 1} | msprof | 32 | 12288 | bfloat16 |
