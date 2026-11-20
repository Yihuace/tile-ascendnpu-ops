# fused_add_rmsnorm

[中文](README.md)

A fused scaled-add and RMSNorm operator implemented for TileLang NPU.

```text
x = x1 * scale + x2
rstd = rsqrt(mean(x * x, dim=gamma_suffix_axes, keepdim=True) + eps)
y = x * rstd * gamma
```

Inputs may have any number of dimensions. `gamma.shape` determines the trailing
dimensions normalized by RMSNorm, following the AscendC partitioning convention:

```text
M = product of the input dimensions before the gamma suffix
N = product of all gamma dimensions
```

## File layout

```text
__init__.py               Public Python API
fused_add_rmsnorm.py      Operator dispatch and runnable example
selector.py               Automatic kernel and tiling selection
shape_utils.py            Multidimensional shape validation and 2-D flattening
kernels/
  single_n.py             Single-row full-load path
  merge_n.py              Small-N full-load path and FP32 two-row fallback
  multi_n.py              Aligned FP16 full-load path
  split_d.py              Large-N streaming split-D path
  split_d_row_group.py    FP16/BF16 unaligned wide-row grouped path
  normal.py               Generic full-load fallback path
  row_one.py              Single-row scheduling shared by normal and multi-N
```

## API

```python
from examples.norm.fused_add_rms_norm import fused_add_rmsnorm, fused_add_rmsnorm_full

y, x_out = fused_add_rmsnorm(x1, x2, gamma, eps=1e-6, scale=1.0)
y, x_out, rstd = fused_add_rmsnorm_full(x1, x2, gamma, eps=1e-6, scale=1.0)
```

`fused_add_rmsnorm` returns the public outputs `y` and `x_out`.
`fused_add_rmsnorm_full` additionally returns `rstd` for correctness validation
and debugging.

All inputs must be on an NPU and use the same `float16`, `bfloat16`, or
`float32` dtype. `x1` and `x2` must have identical shapes, and `gamma.shape`
must match a suffix of the input shape.

## Validation results

The public automatic-selection path passed 600 correctness cases: 200 cases
each for FP16, BF16, and FP32. All 460 cases in the full performance suite also
passed correctness checks and `msprof op` profiling. The arithmetic mean of the
per-case `torch_npu / TileLang` device-time ratios was **1.180x**; the arithmetic
mean of `TileLang / torch_npu` was 0.868x.

## Performance examples

The following representative inputs were measured on Ascend 910B. Device time
is the Task Duration reported by `msprof op`. Pipeline utilization is the
arithmetic mean across all active AIV rows.

| M | N | Dtype | TileLang (us) | torch_npu (us) | torch/TL | VEC | MTE2 | MTE3 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 4096 | BF16 | 4.800 | 6.200 | 1.292x | 33.7% | 29.6% | 19.6% |
| 256 | 4096 | BF16 | 11.680 | 14.741 | 1.262x | 50.8% | 34.9% | 37.3% |
| 512 | 4096 | FP16 | 12.961 | 13.941 | 1.076x | 49.4% | 34.0% | 30.2% |
| 2048 | 8192 | BF16 | 67.483 | 100.264 | 1.486x | 86.5% | 35.0% | 35.3% |
| 12288 | 1024 | FP16 | 41.042 | 51.762 | 1.261x | 83.7% | 80.9% | 66.6% |
| 12288 | 1024 | BF16 | 57.982 | 67.763 | 1.169x | 90.7% | 44.4% | 44.1% |
| 12288 | 4096 | BF16 | 309.912 | 373.135 | 1.204x | 69.2% | 49.9% | 37.8% |
| 8192 | 16384 | FP16 | 825.913 | 976.639 | 1.183x | 42.3% | 63.4% | 33.9% |
| 8192 | 32768 | FP16 | 1773.331 | 2575.223 | 1.452x | 38.7% | 68.9% | 34.7% |
| 12288 | 32768 | FP32 | 5597.724 | 7039.521 | 1.257x | 19.4% | 80.0% | 31.2% |

## Run the example

Run the default FP16 example from this directory:

```bash
python fused_add_rmsnorm.py
```

Use command-line options to select the shape, dtype, epsilon, and residual
scaling factor:

```bash
python fused_add_rmsnorm.py \
  --m 128 --n 4096 --dtype bfloat16 --eps 1e-6 --scale 0.5
```

The example imports `torch_npu`, compares all three TileLang outputs with a
PyTorch FP32 reference, and prints the automatically selected implementation
and tiling. A working TileLang Ascend environment and an available NPU are
required.
