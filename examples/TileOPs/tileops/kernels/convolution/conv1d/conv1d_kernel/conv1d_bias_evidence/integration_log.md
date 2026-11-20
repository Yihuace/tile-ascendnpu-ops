# Conv1dBiasFwdOp — Integration Log

## Environment

```text
CANN        9.0.0
driver      25.5.0
NPU         Ascend910B3
Python      3.11.4
torch       2.7.1+cpu
torch_npu   2.7.1.post4
tilelang    0.1.2 (source tree, commit 9309edb)
branch      op/convolution/conv1d-bias-guoyihua
base        integration/npu-ops
```

## Changes

- `tileops/ops/convolution.py`: add `Conv1dBiasFwdOp` (thin subclass of
  `Conv1dFwdOp`, mandatory-bias contract).
- `tileops/workloads/convolution.py`: add `Conv1dBiasWorkload` (always-bias
  workload, random non-zero bias).
- `tileops/manifest/convolution.yaml`: add the `Conv1dBiasFwdOp` entry.
- `tests/ops/test_conv1d_bias.py`: correctness suite.
- `benchmarks/ops/bench_conv1d_bias.py`: manifest-driven benchmark.
- `conv1d_bias_evidence/`: design / review / retrospective / this log.

## Correctness

```bash
cd examples/TileOPs
PYTHONPATH=$PWD:$PYTHONPATH python3 -m pytest tests/ops/test_conv1d_bias.py -q
```

Result:

```text
28 passed, 2 warnings in 104.13s
```

No `bishengir-compile` crash (`err code: -11`) occurred.

## Report

Generated with:

```bash
TILELANG_ASCEND_MODE=Expert \
  python3 -m tileops.reporting.cli run --op Conv1dBiasFwdOp --prof-mode msprof
```

The generated `run.json` / `report.md` are under:

```text
examples/TileOPs/reports/tileops/<timestamp>_Conv1dBiasFwdOp/
```

## Notes

- The op reuses the Conv1d kernel package; no new TileLang kernel was added.
- The CANN 9.0 compiler-safe Vector fallback from `Conv1dFwdOp` is inherited.
