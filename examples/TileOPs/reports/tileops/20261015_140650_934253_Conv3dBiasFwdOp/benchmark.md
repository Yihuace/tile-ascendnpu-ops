# NPU Benchmark Report
Generated: 2026-10-15 14:09:00

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## Conv3dBiasFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | artifact_dir | bandwidth_tbs | msprof | prof_mode | tflops | @py_assert1 | @py_assert2 | c_in | c_in_g | c_out | d_in | dilation | groups | h_in | input_shape | kd | kh | kw | n | padding | stride | use_bias | w_in | weight_shape | config | dtype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-k3-s1-bias | 173.5600 | N/A | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_140650_934253_Conv3dBiasFwdOp/artifacts/msprof/_conv3d_bias_main_mix_aic_4707k18b | 0.0006 | {'captured_op_count': 1, 'captured_op_names': ['_conv3d_bias_main_mix_aic'], 'kernel_name_attempts': ['_conv3d_bias_main'], 'kernel_name_requested': '_conv3d_bias_main', 'kernel_name_resolved': '_conv3d_bias_main', 'latency_us': 173.559998, 'max_latency_us': 173.559998, 'min_latency_us': 173.559998, 'op_name': '_conv3d_bias_main_mix_aic', 'sample_count': 1} | msprof | 0.0689 | None | None | 8 | 8 | 16 | 8 | 1 | 1 | 16 | [1, 8, 8, 16, 16] | 3 | 3 | 3 | 1 | 1 | 1 | True | 16 | [16, 8, 3, 3, 3] | {'block_k': 128, 'block_m': 64, 'block_n': 128, 'num_stages': 1} | bfloat16 |
| r3d-stem-bias | 6403.6001 | N/A | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_140650_934253_Conv3dBiasFwdOp/artifacts/msprof/_conv3d_bias_main_mix_aic_w3ohreks | 0.0042 | {'captured_op_count': 1, 'captured_op_names': ['_conv3d_bias_main_mix_aic'], 'kernel_name_attempts': ['_conv3d_bias_main'], 'kernel_name_requested': '_conv3d_bias_main', 'kernel_name_resolved': '_conv3d_bias_main', 'latency_us': 6403.600098, 'max_latency_us': 6403.600098, 'min_latency_us': 6403.600098, 'op_name': '_conv3d_bias_main_mix_aic', 'sample_count': 1} | msprof | 0.3097 | None | None | 3 | 3 | 64 | 16 | 1 | 1 | 112 | [1, 3, 16, 112, 112] | 3 | 3 | 3 | 1 | 1 | 1 | True | 112 | [64, 3, 3, 3, 3] | {'block_k': 128, 'block_m': 64, 'block_n': 128, 'num_stages': 1} | float16 |
| video-down-bias | 85.0800 | N/A | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_140650_934253_Conv3dBiasFwdOp/artifacts/msprof/_conv3d_bias_main_mix_aic_z3mww9v6 | 0.0524 | {'captured_op_count': 1, 'captured_op_names': ['_conv3d_bias_main_mix_aic'], 'kernel_name_attempts': ['_conv3d_bias_main'], 'kernel_name_requested': '_conv3d_bias_main', 'kernel_name_resolved': '_conv3d_bias_main', 'latency_us': 85.080002, 'max_latency_us': 85.080002, 'min_latency_us': 85.080002, 'op_name': '_conv3d_bias_main_mix_aic', 'sample_count': 1} | msprof | 14.5976 | None | None | 64 | 64 | 128 | 8 | 1 | 1 | 56 | [1, 64, 8, 56, 56] | 3 | 3 | 3 | 1 | 1 | 2 | True | 56 | [128, 64, 3, 3, 3] | {'block_k': 128, 'block_m': 64, 'block_n': 128, 'num_stages': 1} | float16 |
| unet-encoder-bias | 6414.4600 | N/A | /mnt/workspace/npuir-migration/examples/TileOPs/reports/tileops/20261015_140650_934253_Conv3dBiasFwdOp/artifacts/msprof/_conv3d_bias_main_mix_aic_oo4lwh_t | 0.0039 | {'captured_op_count': 1, 'captured_op_names': ['_conv3d_bias_main_mix_aic'], 'kernel_name_attempts': ['_conv3d_bias_main'], 'kernel_name_requested': '_conv3d_bias_main', 'kernel_name_resolved': '_conv3d_bias_main', 'latency_us': 6414.459961, 'max_latency_us': 6414.459961, 'min_latency_us': 6414.459961, 'op_name': '_conv3d_bias_main_mix_aic', 'sample_count': 1} | msprof | 2.1682 | None | None | 32 | 32 | 64 | 32 | 1 | 1 | 64 | [1, 32, 32, 64, 64] | 3 | 3 | 3 | 1 | 1 | 1 | True | 64 | [64, 32, 3, 3, 3] | {'block_k': 128, 'block_m': 64, 'block_n': 128, 'num_stages': 1} | bfloat16 |
