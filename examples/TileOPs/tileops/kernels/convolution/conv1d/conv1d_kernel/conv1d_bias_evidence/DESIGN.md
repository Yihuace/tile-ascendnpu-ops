# Conv1dBiasFwdOp — Design

## Scope

`Conv1dBiasFwdOp` is the **bias-required** variant of `Conv1dFwdOp`:

- same computation as `torch.nn.functional.conv1d`;
- `bias` is a mandatory positional input of shape `[C_out]`;
- `bias=None` is rejected;
- the kernel always dispatches its `has_bias=True` variant.

This mirrors the GPU baseline, which registers `conv1d` (bias optional) and
`conv1d-bias` as two separate manifest entries.

## Interface

```text
input   [N, C_in, L_in]        float16 | bfloat16
weight  [C_out, C_in/groups, k] same as input
bias    [C_out]                 same as input (required)
output  [N, C_out, L_out]       same as input

params: stride, padding, dilation, groups
```

`forward()` validates that `bias` is a device tensor, has the same dtype and
device as `input`, and has shape `[C_out]`; otherwise it raises `ValueError`.
After validation it delegates to `Conv1dFwdOp.forward(input, weight, bias)`.

## Kernel reuse

No new TileLang kernel is introduced. Every migrated Conv1d kernel already
implements `has_bias=True`:

- `Conv1dPointwiseKernel`
- `Conv1dKernel`
- `Conv1dUnitStrideKernel`
- `DepthwiseConv1dKernel`
- `GroupConv1dKernel`

`_select_kernel_key` is inherited unchanged, so the public dispatch keys and
the manifest `kernel_map` stay identical to `Conv1dFwdOp`.

The CANN 9.0 compiler-safe fallback introduced with `Conv1dFwdOp` (all public
conv1d keys execute on `Conv1dVectorKernel`) also applies here, which avoids
the `bishengir-compile` SIGSEGV tracked as CG-2026-0003.

## Files

| Layer | File |
|---|---|
| Op | `tileops/ops/convolution.py` (`Conv1dBiasFwdOp`) |
| Workload | `tileops/workloads/convolution.py` (`Conv1dBiasWorkload`) |
| Manifest | `tileops/manifest/convolution.yaml` (`Conv1dBiasFwdOp`) |
| Test | `tests/ops/test_conv1d_bias.py` |
| Benchmark | `benchmarks/ops/bench_conv1d_bias.py` |

## Roofline

Inherited from `Conv1dFwdOp.eval_roofline()`; because the bias is always
present, `has_bias=True` and the bias FLOP/byte terms are always counted.
