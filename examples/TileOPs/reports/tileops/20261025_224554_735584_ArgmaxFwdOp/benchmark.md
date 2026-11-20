# NPU Benchmark Report
Generated: 2026-10-25 22:50:16

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## ArgmaxFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | extra | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-argmax | 2.5800 | 0.4371 | 2.7955 | 1.7500 | 4.1881 | 0.0183 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261025_224554_735584_ArgmaxFwdOp/artifacts/msprof/main_hh2zavft | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 2.58, 'max_latency_us': 2.58, 'min_latency_us': 2.58, 'op_name': 'main', 'sample_count': 1} | msprof | {'dim': -1} | {'block_m': 8} | float16 | [32, 256] |
| smoke-argmax | 1.9600 | 0.7993 | 3.3939 | 1.7500 | 3.6603 | 0.0293 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261025_224554_735584_ArgmaxFwdOp/artifacts/msprof/main_0ma_5_oy | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 1.96, 'max_latency_us': 1.96, 'min_latency_us': 1.96, 'op_name': 'main', 'sample_count': 1} | msprof | {'dim': -1} | {'block_m': 8} | bfloat16 | [32, 256] |
| smoke-argmax | 1.8600 | 1.0224 | 1.4769 | 1.7500 | 3.5389 | 0.0264 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261025_224554_735584_ArgmaxFwdOp/artifacts/msprof/main_4gowqvu6 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 1.86, 'max_latency_us': 1.86, 'min_latency_us': 1.86, 'op_name': 'main', 'sample_count': 1} | msprof | {'dim': -1} | {'block_m': 8} | float32 | [32, 256] |
| hidden-state-argmax | 30.8200 | 31.1066 | 2.2182 | 1.7500 | 5.3060 | 1.2075 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261025_224554_735584_ArgmaxFwdOp/artifacts/msprof/main_sstpagw9 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 30.82, 'max_latency_us': 30.82, 'min_latency_us': 30.82, 'op_name': 'main', 'sample_count': 1} | msprof | {'dim': -1} | {'block_m': 2} | float16 | [2048, 4096] |
| hidden-state-argmax | 52.7200 | 21.7872 | 2.8402 | 1.7500 | 4.1486 | 0.9039 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261025_224554_735584_ArgmaxFwdOp/artifacts/msprof/main_aqgyhg7p | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 52.720001, 'max_latency_us': 52.720001, 'min_latency_us': 52.720001, 'op_name': 'main', 'sample_count': 1} | msprof | {'dim': -1} | {'block_m': 1} | bfloat16 | [2048, 4096] |
| 3d-non-last-axis-argmax | 13.7200 | 24.5967 | 2.9474 | 1.7500 | 3.6634 | 0.9011 | /home/developer/tilelang-ascend-npu-ops/tilelang-ascend-npu-ops/examples/TileOPs/reports/tileops/20261025_224554_735584_ArgmaxFwdOp/artifacts/msprof/main_wj9hs1y_ | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 13.72, 'max_latency_us': 13.72, 'min_latency_us': 13.72, 'op_name': 'main', 'sample_count': 1} | msprof | {'dim': 0} | {'block_m': 1024} | float16 | [4, 128, 4096] |
