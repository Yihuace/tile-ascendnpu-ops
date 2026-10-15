# NPU Benchmark Report
Generated: 2026-10-15 21:46:45

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## HardsigmoidFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | op_params | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-1m | 4.9000 | 24.4581 | 2.5023 | 1.7500 | 4.6035 | 1.0710 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_213240_149053_HardsigmoidFwdOp/artifacts/msprof/main_bugrgxr4 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 4.9, 'max_latency_us': 4.9, 'min_latency_us': 4.9, 'op_name': 'main', 'sample_count': 1} | msprof | {} | {'block_size': 8192} | float16 | [1048576] |
| smoke-1m | 4.8600 | 28.1390 | 3.0023 | 1.7500 | 4.6043 | 1.2956 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_213240_149053_HardsigmoidFwdOp/artifacts/msprof/main_m6vs894d | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 4.86, 'max_latency_us': 4.86, 'min_latency_us': 4.86, 'op_name': 'main', 'sample_count': 1} | msprof | {} | {'block_size': 8192} | bfloat16 | [1048576] |
| mbv3-se-gate | 2.4800 | 0.3628 | 2.6098 | 1.7500 | 4.5937 | 0.0166 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_213240_149053_HardsigmoidFwdOp/artifacts/msprof/main_1yvi92df | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2.48, 'max_latency_us': 2.48, 'min_latency_us': 2.48, 'op_name': 'main', 'sample_count': 1} | msprof | {} | {'block_size': 8192} | float16 | [32, 240, 1, 1] |
| mbv3-se-gate | 2.3600 | 0.4543 | 3.1301 | 1.7500 | 4.5961 | 0.0209 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_213240_149053_HardsigmoidFwdOp/artifacts/msprof/main_cwz798jv | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2.36, 'max_latency_us': 2.36, 'min_latency_us': 2.36, 'op_name': 'main', 'sample_count': 1} | msprof | {} | {'block_size': 8192} | bfloat16 | [32, 240, 1, 1] |
| mbv3-se-gate-deep | 2.0800 | 1.7201 | 2.6584 | 1.7500 | 4.5937 | 0.0790 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_213240_149053_HardsigmoidFwdOp/artifacts/msprof/main_k879q0ok | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2.08, 'max_latency_us': 2.08, 'min_latency_us': 2.08, 'op_name': 'main', 'sample_count': 1} | msprof | {} | {'block_size': 8192} | float16 | [32, 960, 1, 1] |
| mbv3-se-gate-deep | 2.1000 | 2.0423 | 3.1884 | 1.7500 | 4.5961 | 0.0939 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_213240_149053_HardsigmoidFwdOp/artifacts/msprof/main_9eowskzr | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2.1, 'max_latency_us': 2.1, 'min_latency_us': 2.1, 'op_name': 'main', 'sample_count': 1} | msprof | {} | {'block_size': 8192} | bfloat16 | [32, 960, 1, 1] |
| fc-wide | 15.9200 | 60.2202 | 2.5003 | 1.7500 | 4.6074 | 2.6349 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_213240_149053_HardsigmoidFwdOp/artifacts/msprof/main_i0weqjfo | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 15.92, 'max_latency_us': 15.92, 'min_latency_us': 15.92, 'op_name': 'main', 'sample_count': 1} | msprof | {} | {'block_size': 8192} | float16 | [2048, 4096] |
| fc-wide | 16.7000 | 65.4185 | 3.0003 | 1.7500 | 4.6075 | 3.0142 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_213240_149053_HardsigmoidFwdOp/artifacts/msprof/main_0ckedoy8 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 16.700001, 'max_latency_us': 16.700001, 'min_latency_us': 16.700001, 'op_name': 'main', 'sample_count': 1} | msprof | {} | {'block_size': 8192} | bfloat16 | [2048, 4096] |
