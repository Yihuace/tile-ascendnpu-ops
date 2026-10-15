# NPU Benchmark Report
Generated: 2026-10-15 14:17:17

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## GLADecodeOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | B | DK | DV | H | scale | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-b2-h4-d64 | 3.4000 | 1.7320 | 3.7965 | 1.7500 | 4.2994 | 0.0745 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_141409_254847_GLADecodeOp/artifacts/msprof/_gla_decode_transit_main_iva12eb2 | {'captured_op_count': 1, 'captured_op_names': ['_gla_decode_transit_main'], 'kernel_name_attempts': ['_gla_decode_transit_main'], 'kernel_name_requested': '_gla_decode_transit_main', 'kernel_name_resolved': '_gla_decode_transit_main', 'latency_us': 3.4, 'max_latency_us': 3.4, 'min_latency_us': 3.4, 'op_name': '_gla_decode_transit_main', 'sample_count': 1} | msprof | 2 | 64 | 64 | 4 | 0.125 | {'k_tile': 32} | bfloat16 |
| serving-b1-h8 | 9.1200 | 2.4732 | 3.7313 | 1.7500 | 4.3810 | 0.1084 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_141409_254847_GLADecodeOp/artifacts/msprof/_gla_decode_transit_main_rkalpqtb | {'captured_op_count': 1, 'captured_op_names': ['_gla_decode_transit_main'], 'kernel_name_attempts': ['_gla_decode_transit_main'], 'kernel_name_requested': '_gla_decode_transit_main', 'kernel_name_resolved': '_gla_decode_transit_main', 'latency_us': 9.12, 'max_latency_us': 9.12, 'min_latency_us': 9.12, 'op_name': '_gla_decode_transit_main', 'sample_count': 1} | msprof | 1 | 128 | 128 | 8 | 0.08838834764831845 | {'k_tile': 16} | bfloat16 |
| serving-b1-h32 | 10.2800 | 8.7765 | 3.7326 | 1.7500 | 4.3810 | 0.3845 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_141409_254847_GLADecodeOp/artifacts/msprof/_gla_decode_transit_main_646q_zdt | {'captured_op_count': 1, 'captured_op_names': ['_gla_decode_transit_main'], 'kernel_name_attempts': ['_gla_decode_transit_main'], 'kernel_name_requested': '_gla_decode_transit_main', 'kernel_name_resolved': '_gla_decode_transit_main', 'latency_us': 10.28, 'max_latency_us': 10.28, 'min_latency_us': 10.28, 'op_name': '_gla_decode_transit_main', 'sample_count': 1} | msprof | 1 | 128 | 128 | 32 | 0.08838834764831845 | {'k_tile': 16} | bfloat16 |
| serving-b1-h64 | 18.1400 | 9.9473 | 3.7328 | 1.7500 | 4.3810 | 0.4358 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_141409_254847_GLADecodeOp/artifacts/msprof/_gla_decode_transit_main_f_1wtsve | {'captured_op_count': 1, 'captured_op_names': ['_gla_decode_transit_main'], 'kernel_name_attempts': ['_gla_decode_transit_main'], 'kernel_name_requested': '_gla_decode_transit_main', 'kernel_name_resolved': '_gla_decode_transit_main', 'latency_us': 18.139999, 'max_latency_us': 18.139999, 'min_latency_us': 18.139999, 'op_name': '_gla_decode_transit_main', 'sample_count': 1} | msprof | 1 | 128 | 128 | 64 | 0.08838834764831845 | {'k_tile': 16} | bfloat16 |
| serving-b8-h32 | 55.7200 | 12.9537 | 3.7330 | 1.7500 | 4.3810 | 0.5675 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_141409_254847_GLADecodeOp/artifacts/msprof/_gla_decode_transit_main_7cayu4v0 | {'captured_op_count': 1, 'captured_op_names': ['_gla_decode_transit_main'], 'kernel_name_attempts': ['_gla_decode_transit_main'], 'kernel_name_requested': '_gla_decode_transit_main', 'kernel_name_resolved': '_gla_decode_transit_main', 'latency_us': 55.720001, 'max_latency_us': 55.720001, 'min_latency_us': 55.720001, 'op_name': '_gla_decode_transit_main', 'sample_count': 1} | msprof | 8 | 128 | 128 | 32 | 0.08838834764831845 | {'k_tile': 16} | bfloat16 |
| serving-b8-h64 | 102.2400 | 14.1192 | 3.7330 | 1.7500 | 4.3810 | 0.6186 | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_141409_254847_GLADecodeOp/artifacts/msprof/_gla_decode_transit_main_40erb07x | {'captured_op_count': 1, 'captured_op_names': ['_gla_decode_transit_main'], 'kernel_name_attempts': ['_gla_decode_transit_main'], 'kernel_name_requested': '_gla_decode_transit_main', 'kernel_name_resolved': '_gla_decode_transit_main', 'latency_us': 102.239998, 'max_latency_us': 102.239998, 'min_latency_us': 102.239998, 'op_name': '_gla_decode_transit_main', 'sample_count': 1} | msprof | 8 | 128 | 128 | 64 | 0.08838834764831845 | {'k_tile': 16} | bfloat16 |
