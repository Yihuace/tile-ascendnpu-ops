# Integration Log -- MaxPool2dFwdOp

- meta: `examples/TileOPs/tileops/kernels/pool/max_pool2d/.migration_meta.json`
- op_slug / family: `max_pool2d` / `pool`
- manifest op_name: `MaxPool2dFwdOp`
- integrated: `_max_pool2d_kernel` -> `max_pool2d_kernel/_max_pool2d_kernel.py`
- design_docs: `_max_pool2d_kernel` -> `max_pool2d_kernel/_max_pool2d_kernel_DESIGN.md`
- programming mode (from Stage 3 product `ASCEND_MODE`): `Developer`
- attempts: 2 (attempt 1: read-only root-cause diagnosis, no in-scope file modified, `[INTEGRATE_FAIL]`;
  attempt 2: in-scope test-reference fix under explicit conductor authorization, `INTEGRATE_COMPLETED`)
- wrapper: `tileops/kernels/pool/max_pool2d/max_pool2d.py` (baseline/perf_opt switch block installed;
  baseline active; `MaxPool2dKernel.ascend_mode = "Developer"`, unchanged this round)

## Integrated files

- `tileops/kernels/pool/max_pool2d/max_pool2d_kernel/_max_pool2d_kernel.py`
- `tileops/kernels/pool/max_pool2d/max_pool2d_kernel/_max_pool2d_kernel_DESIGN.md`
- `tileops/kernels/pool/max_pool2d/max_pool2d_kernel/__init__.py` (aggregation re-export: `_max_pool2d_kernel`, `pool_output_dim`)
- `tileops/kernels/pool/max_pool2d/max_pool2d.py` (wrapper, import block rewritten by the integration script)

Integrity: the integrated kernel and design copies are byte-identical to the frozen conductor products
(`examples/max_pool2d/_max_pool2d_kernel/{_max_pool2d_kernel.py,DESIGN.md}`, verified with `diff -q` again
this round). No integration-script run was needed in attempt 2 -- only the scaffold test case changed.

## Changes made this attempt (attempt 2)

- Modified (authorized in-scope scaffold test, max_pool2d section only):
  `examples/TileOPs/tests/ops/test_pool.py`, class `MaxPool2dTest.ref_program` (lines ~285-308).
  - md5 before: `4364b26dbe7e7fe99de08e158d79db59`
  - md5 after:  `f6a1dcbb358011e29663a86e908ed655`
  - backup: `tileops/kernels/pool/max_pool2d/max_pool2d_kernel/history_version/test_pool.py.pre_attempt2.bak`
- Diff summary: extracted the `F.max_pool2d(...)` kwargs into a dict and, when `dilation != 1`
  (int or tuple), compute the golden on CPU and move it back to the input device:
  `F.max_pool2d(input.cpu(), **kwargs).to(input.device)`. When `dilation == 1` the original
  on-device golden is used unchanged. Rationale: torch_npu routes `F.max_pool2d` to
  `aclnnMaxPool2dWithMask`, whose golden only supports `dilation == 1`; windowed max pooling is a
  pure selection (no accumulation) so the CPU golden is bit-identical to the NPU output. Comparison
  stays exact (`atol=rtol=0`). Coverage preserved: the `dilation=(2,1)` full case still runs and
  still reaches the TileOPs op + kernel.
- NOT touched: adaptive `*Fixture`/`*Test`/cases, file-header docstring, imports,
  `tileops/manifest/pool.yaml`, `tileops/workloads/pool.py`, `benchmarks/ops/bench_pool.py`,
  the adaptive integration/kernel packages, and the `MaxPool2dFwdOp` / `MaxPool2dKernel` /
  integrated-kernel implementation or criteria.

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261014_072101_015168_MaxPool2dFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261014_072101_015168_MaxPool2dFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261014_072101_015168_MaxPool2dFwdOp/report.html`
- run.json status: **`passed`** (CLI exit 0)
- correctness: **pass, 36/36** (`summary.correctness_passed=true`, `correctness_failed=0`); all 36
  file-level cases pass (MaxPool2dFwdOp + the shared adaptive siblings).
  - Per-operator rollup (`run.json.operators[0]`, `MaxPool2dFwdOp`): cases 22, passed 22, failed 0,
    `pass_rate=1.0`, `max_abs_err = 0.00e+00` (bit-exact).
  - The previously failing `test_max_pool2d[full-default-stride-dilation-fp16]` (dilation=(2,1)) now
    passes.
- benchmark: **pass** (`benchmark_requested=true`, `benchmark_passed=true`, `benchmark_tests=24`,
  `benchmark_failed=0`; `case_count=18` msprof cases, `invalid_profile_count=0`).
  - `MaxPool2dFwdOp` msprof cases (all `profiler_valid=true`), ratio = roofline utilization %:
    | Label | Shape | K/S/P/D | ceil | dtype | Latency (us) | Ratio (%) |
    |---|---|---|---|---|---:|---:|
    | resnet-stem | [32,64,112,112] | 3/2/1/1 | False | fp16 | 1390.3 | 13.63 |
    | resnet-stem | [32,64,112,112] | 3/2/1/1 | False | bf16 | 1391.4 | 13.72 |
    | resnet-stem | [32,64,112,112] | 3/2/1/1 | False | fp32 | 2646.5 | 12.68 |
    | vgg-block | [16,128,56,56] | 2/2/0/1 | False | fp16 | 113.6 | 14.25 |
    | vgg-block | [16,128,56,56] | 2/2/0/1 | False | bf16 | 116.4 | 14.21 |
    | vgg-block | [16,128,56,56] | 2/2/0/1 | False | fp32 | 190.9 | 12.64 |
    | alexnet-ceil | [32,64,55,55] | 3/2/0/1 | True | fp16 | 201.8 | 13.55 |
    | alexnet-ceil | [32,64,55,55] | 3/2/0/1 | True | bf16 | 202.1 | 13.68 |
    | alexnet-ceil | [32,64,55,55] | 3/2/0/1 | True | fp32 | 365.6 | 12.11 |
  - Per-operator rollup performance: 9 cases, ratio range 12.11% – 14.25%; bandwidth 1.75 TB/s.
- `warnings: []` in run.json.

## 调试历史

- attempt 1: symptom `test_max_pool2d[full-default-stride-dilation-fp16]` FAILED; classification =
  **测试/用例自身问题（参考实现/环境）**; diagnosed (no modification) that the exception originates in
  `ref_program` (`F.max_pool2d` on the NPU tensor), executed before the op; torch_npu's
  `aclnnMaxPool2dWithMask` golden supports only `dilation == 1`. Returned `[INTEGRATE_FAIL]` because
  `tests/` was then out of scope. No `history_version/` backup taken (no file modified).
- attempt 2 (this round, conductor-authorized in-scope test-reference fix):
  - backup taken (`history_version/test_pool.py.pre_attempt2.bak`, md5 `4364b26d...`).
  - fix applied: CPU golden fallback for `dilation != 1` in `MaxPool2dTest.ref_program`
    (reference path only).
  - fast smoke on the single previously-failing case (`-q`): 1 passed (2.88 s).
  - full single-op report re-run: `status=passed`, correctness 36/36, benchmark 24/24 tests, 18/18
    msprof cases valid. `INTEGRATE_COMPLETED`.

## Issues (recorded, not repaired)

1. **RESOLVED this round** -- the attempt-1 blocker (`full-default-stride-dilation-fp16` golden
   unsupported on the NPU reference backend). Fixed by the authorized CPU-golden fallback (option (a)
   suggested in attempt 1); coverage retained.
2. `--op MaxPool2dFwdOp` runs the whole shared `tests/ops/test_pool.py` / `benchmarks/ops/bench_pool.py`
   file (no per-op parameterized filter). The per-operator rollup (`run.json.operators[0]`) is the
   authoritative per-op view: `MaxPool2dFwdOp` correctness 22/22, performance 9 cases; the
   file-level `summary.correctness_tests=36` counts the shared adaptive siblings too.
3. No integration-script run was needed in attempt 2 (kernel/design copies already byte-identical and
   verified again with `diff -q`); no `[error]`/`[warn]`.
