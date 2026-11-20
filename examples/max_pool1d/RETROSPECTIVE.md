# RETROSPECTIVE — max_pool1d (migration-harness)

> Op-level retrospective log (canonical schema: `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`).
> Append-only; one section per Stage execution.

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | Stage 0 scaffolded 1D bench wraps scalar workload extras with `tuple(...)`, breaking every MaxPool1dFwdOp benchmark case; benchmark stage is out of the Stage 5 edit scope, so the defect can only be recorded, never repaired within Stage 5. | `examples/TileOPs/benchmarks/ops/bench_pool.py:161`; failure message in `reports/tileops/20261014_085626_616055_MaxPool1dFwdOp/pytest/benchmark.xml`; run.json `benchmark_tests.failed=6`. | In `add-npu-op` bench template, coerce scalar extras (`(v,) if isinstance(v,int) else tuple(v)`) before wrapping so 1D ops with scalar workload extras do not abort. | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| Pre-armed CPU golden fallback for `dilation != 1` in `MaxPool1dTest.ref_program` keeps max_pool1d 1D correctness bit-exact (`max_abs_err=0.0`) on NPU, no `AclNN_*_Error`; the max_pool2d lesson generalizes to the 1D op (pure selection, `atol=rtol=0` preserved). | D | `tests/ops/test_pool.py` `MaxPool1dTest.ref_program` (lines 525-541); `reports/tileops/20261014_085626_616055_MaxPool1dFwdOp/run.json` `operators[0]` (cases=34, passed=34, `max_abs_err=0.0`). | `cd examples/TileOPs && python -m tileops.reporting.cli run --op MaxPool1dFwdOp --prof-mode msprof`; the `ecg-cnn-dilated` `dilation=2` case exercises the fallback. | tilelang 0.1.2+ddf4255 (commit ddf4255a4ae5214c27f9d731ce65c9107b748388), Ascend910B3, CANN 9.0.0, torch 2.9.0+cpu / torch_npu 2.9.0.post8 | pattern-library |
| `tuple(op_params["kernel_size"])` in a scaffolded 1D bench aborts with `TypeError: 'int' object is not iterable` when workload `include_extra` extras are scalars -> all benchmark cases fail before profiling (correctness unaffected). | C | `benchmarks/ops/bench_pool.py:161`; `tileops/workloads/pool.py` `MaxPool1dWorkload` (scalar extras); `reports/tileops/20261014_085626_616055_MaxPool1dFwdOp/run.json` `benchmark_tests.failed=6`. | `cd examples/TileOPs && python -m pytest benchmarks/ops/bench_pool.py -k test_max_pool1d_bench -q` | tilelang 0.1.2+ddf4255, Ascend910B3, CANN 9.0.0 | bottleneck-patterns |

### Transferable Lessons（可迁移教训）

- A correct kernel can still produce a `partial` report when the Stage 0-scaffolded bench routes scalar workload extras through `tuple(...)`; distinguish "benchmark stage invalid" (record only) from correctness, which is the migration success criterion.
- The max_pool2d CPU-golden fallback pattern (`dilation != 1` -> `F.max_pool1d(input.cpu(), **kwargs).to(input.device)`) is bit-exact for windowed max pooling (pure selection) and satisfies `atol=rtol=0`; it is pre-armed for max_pool1d and needs no Stage 5 change.
- Always resolve the `--meta` path as absolute (or repo-root-relative to the repo root): `integrate_kernel.py` resolves it against the CWD (`examples/TileOPs`), not the repo root.
