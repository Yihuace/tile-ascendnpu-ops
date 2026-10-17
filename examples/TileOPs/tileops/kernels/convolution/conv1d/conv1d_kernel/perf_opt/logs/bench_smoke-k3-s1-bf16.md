# NPU Benchmark Report
Generated: 2026-10-17 15:39:03

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## Conv1dFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOPs/s) | Perf(TOps/s) | advice | artifact_dir | msprof | prof_mode | roofline_levels | bias | c_in | c_in_g | c_out | dilation | groups | input_shape | kernel_size | l_in | n | padding | stride | use_bias | weight_shape | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-k3-s1 | 136.6027 | 0.0390 | 5378.2607 | 1.7500 | 297.2160 | 0.1159 | latency bound:memory caused | /mnt/workspace/npuir-migration/examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/perf_opt/profiles/smoke-k3-s1-bf16/_conv1d_main_mix_aic_vredv9tz | {'captured_op_count': 1, 'captured_op_names': ['_conv1d_main_mix_aic'], 'kernel_name_attempts': ['_conv1d_main'], 'kernel_name_requested': '_conv1d_main', 'kernel_name_resolved': '_conv1d_main', 'latency_us': 136.602737, 'max_latency_us': 136.602737, 'min_latency_us': 136.602737, 'op_name': '_conv1d_main_mix_aic', 'sample_count': 1} | msprof | [{'bw': 16.763805389404297, 'bw_name': 'L1 Read + Write', 'ratio': 0.00039, 'title': 'Memory Unit(Cube)'}, {'bw': 4.190951347351074, 'bw_name': 'Read from L0C', 'ratio': 0.00039, 'title': 'Memory Unit(Cube)'}, {'bw': 8.381902694702148, 'bw_name': 'Read from L1', 'ratio': 0.00039, 'title': 'Memory Unit(Cube)'}, {'bw': 8.381902694702148, 'bw_name': 'Write to L1', 'ratio': 0.00039, 'title': 'Memory Unit(Cube)'}, {'bw': 4.162109375, 'bw_name': 'FIXP', 'ratio': 0.00039, 'title': 'Pipeline(Cube)'}, {'bw': 6.328125, 'bw_name': 'MTE1', 'ratio': 0.000403, 'title': 'Pipeline(Cube)'}, {'bw': 6.642578125, 'bw_name': 'MTE2', 'ratio': 0.00039, 'title': 'Pipeline(Cube)'}, {'bw': 3.8951170444488525, 'bw_name': 'MTE3', 'ratio': 0.000779, 'title': 'Pipeline(Cube)'}, {'bw': 3.8951170444488525, 'bw_name': 'L1 to GM', 'ratio': 0.00039, 'title': 'Memory Transfer(Cube)'}, {'bw': 8.381902694702148, 'bw_name': 'Read from UB', 'ratio': 0.000333, 'title': 'Memory Unit(Vector)'}, {'bw': 33.527610778808594, 'bw_name': 'UB Read + Write', 'ratio': 0.000554, 'title': 'Memory Unit(Vector)'}, {'bw': 33.527610778808594, 'bw_name': 'Vector Read UB', 'ratio': 0.000333, 'title': 'Memory Unit(Vector)'}, {'bw': 33.527610778808594, 'bw_name': 'Vector Write UB', 'ratio': 0.000333, 'title': 'Memory Unit(Vector)'}, {'bw': 8.381902694702148, 'bw_name': 'Write to UB', 'ratio': 0.000644, 'title': 'Memory Unit(Vector)'}, {'bw': 8.596094131469727, 'bw_name': 'MTE2', 'ratio': 0.000628, 'title': 'Pipeline(Vector)'}, {'bw': 7.296875, 'bw_name': 'MTE3', 'ratio': 0.000333, 'title': 'Pipeline(Vector)'}, {'bw': 8.596094131469727, 'bw_name': 'GM to UB', 'ratio': 0.000628, 'title': 'Memory Transfer(Vector)'}, {'bw': 7.296875, 'bw_name': 'UB to GM', 'ratio': 0.000333, 'title': 'Memory Transfer(Vector)'}, {'bw': 1.75, 'bw_name': 'GM Read + Write', 'ratio': 0.00039, 'title': 'GM/L2'}, {'bw': 7.78000020980835, 'bw_name': 'L2 Read + Write', 'ratio': 0.002158, 'title': 'GM/L2'}] | None | 8 | 8 | 16 | 1 | 1 | [1, 8, 128] | 3 | 128 | 1 | 0 | 1 | False | [16, 8, 3] | {'block_k': 128, 'block_m': 64, 'block_n': 128, 'num_stages': 1} | bfloat16 |
