# NPU Benchmark Report
Generated: 2026-10-17 15:41:08

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
| encodec-init | 268.1454 | 0.4322 | 1216.5009 | 1.7500 | 297.2160 | 1.2845 | latency bound:memory caused | /mnt/workspace/npuir-migration/examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/perf_opt/profiles/encodec-init-bf16/_conv1d_main_mix_aic_gj0oc4vm | {'captured_op_count': 1, 'captured_op_names': ['_conv1d_main_mix_aic'], 'kernel_name_attempts': ['_conv1d_main'], 'kernel_name_requested': '_conv1d_main', 'kernel_name_resolved': '_conv1d_main', 'latency_us': 268.145355, 'max_latency_us': 268.145355, 'min_latency_us': 268.145355, 'op_name': '_conv1d_main_mix_aic', 'sample_count': 1} | msprof | [{'bw': 16.763805389404297, 'bw_name': 'L1 Read + Write', 'ratio': 0.004351, 'title': 'Memory Unit(Cube)'}, {'bw': 4.190951347351074, 'bw_name': 'Read from L0C', 'ratio': 0.004351, 'title': 'Memory Unit(Cube)'}, {'bw': 8.381902694702148, 'bw_name': 'Read from L1', 'ratio': 0.004351, 'title': 'Memory Unit(Cube)'}, {'bw': 8.381902694702148, 'bw_name': 'Write to L1', 'ratio': 0.004351, 'title': 'Memory Unit(Cube)'}, {'bw': 4.162109375, 'bw_name': 'FIXP', 'ratio': 0.004351, 'title': 'Pipeline(Cube)'}, {'bw': 6.328125, 'bw_name': 'MTE1', 'ratio': 0.004351, 'title': 'Pipeline(Cube)'}, {'bw': 3.8951170444488525, 'bw_name': 'MTE3', 'ratio': 0.005126, 'title': 'Pipeline(Cube)'}, {'bw': 3.8951170444488525, 'bw_name': 'L1 to GM', 'ratio': 0.007176, 'title': 'Memory Transfer(Cube)'}, {'bw': 8.381902694702148, 'bw_name': 'Read from UB', 'ratio': 0.001365, 'title': 'Memory Unit(Vector)'}, {'bw': 33.527610778808594, 'bw_name': 'UB Read + Write', 'ratio': 0.001193, 'title': 'Memory Unit(Vector)'}, {'bw': 33.527610778808594, 'bw_name': 'Vector Read UB', 'ratio': 0.000601, 'title': 'Memory Unit(Vector)'}, {'bw': 33.527610778808594, 'bw_name': 'Vector Write UB', 'ratio': 0.000601, 'title': 'Memory Unit(Vector)'}, {'bw': 8.381902694702148, 'bw_name': 'Write to UB', 'ratio': 0.000927, 'title': 'Memory Unit(Vector)'}, {'bw': 8.596094131469727, 'bw_name': 'MTE2', 'ratio': 0.000904, 'title': 'Pipeline(Vector)'}, {'bw': 7.296875, 'bw_name': 'MTE3', 'ratio': 0.001567, 'title': 'Pipeline(Vector)'}, {'bw': 8.596094131469727, 'bw_name': 'GM to UB', 'ratio': 0.000904, 'title': 'Memory Transfer(Vector)'}, {'bw': 7.296875, 'bw_name': 'UB to GM', 'ratio': 0.001567, 'title': 'Memory Transfer(Vector)'}, {'bw': 1.75, 'bw_name': 'GM Read + Write', 'ratio': 0.004322, 'title': 'GM/L2'}, {'bw': 7.78000020980835, 'bw_name': 'L2 Read + Write', 'ratio': 0.01397, 'title': 'GM/L2'}] | None | 1 | 1 | 32 | 1 | 1 | [1, 1, 24000] | 7 | 24000 | 1 | 0 | 1 | False | [32, 1, 7] | {'block_k': 128, 'block_m': 64, 'block_n': 128, 'num_stages': 1} | bfloat16 |
