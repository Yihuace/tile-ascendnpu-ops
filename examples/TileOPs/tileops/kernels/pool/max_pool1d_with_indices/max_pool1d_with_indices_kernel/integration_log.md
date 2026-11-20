# Integration Log -- MaxPool1dIndicesFwdOp

- meta: `examples/TileOPs/tileops/kernels/pool/max_pool1d_with_indices/.migration_meta.json`
- op_slug / family: `max_pool1d_with_indices` / `pool`
- manifest op_name: `MaxPool1dIndicesFwdOp`
- integrated: `_max_pool1d_with_indices_kernel` -> `max_pool1d_with_indices_kernel/_max_pool1d_with_indices_kernel.py`
- design_docs: `_max_pool1d_with_indices_kernel` -> `max_pool1d_with_indices_kernel/_max_pool1d_with_indices_kernel_DESIGN.md` (present, copied)
- programming mode (from Stage 3 product `ASCEND_MODE` = `"Developer"`): `Developer`
- attempts: 1 (deterministic integration + full single-op report; one controlled reference-path repair -- see 调试历史)
- wrapper: `tileops/kernels/pool/max_pool1d_with_indices/max_pool1d_with_indices.py`
  (baseline/perf_opt switch block installed; baseline active; `_MaxPool1dKernelBase.ascend_mode = "Developer"`)

## Integrated files

- `tileops/kernels/pool/max_pool1d_with_indices/max_pool1d_with_indices_kernel/_max_pool1d_with_indices_kernel.py`
- `tileops/kernels/pool/max_pool1d_with_indices/max_pool1d_with_indices_kernel/_max_pool1d_with_indices_kernel_DESIGN.md`
- `tileops/kernels/pool/max_pool1d_with_indices/max_pool1d_with_indices_kernel/__init__.py` (aggregation re-export: `_max_pool1d_with_indices_kernel`, `pool_output_dim`)
- `tileops/kernels/pool/max_pool1d_with_indices/max_pool1d_with_indices.py` (wrapper, import block rewritten by the integration script)

Integrity: the integrated kernel and design copies are byte-identical to the frozen conductor products
(`examples/max_pool1d_with_indices/_max_pool1d_with_indices_kernel/{_max_pool1d_with_indices_kernel.py,DESIGN.md}`,
verified with `diff -q`). The integration script was run **exactly once** (its aggregate-`__init__.py`
generation is non-idempotent on re-run) and the resulting `__init__.py` + relative imports were verified by
hand. Integration script `[json]` summary: `ascend_mode="Developer"`, `wrapper_rewritten=true`,
`wrapper_mode_updated=false` (already `Developer`), `smoke="pass"`, reexported
`["_max_pool1d_with_indices_kernel", "pool_output_dim"]`, no `[error]`, no `[warn]`.

## 验证结果

- report_run_json (authoritative, acceptance): `examples/TileOPs/reports/tileops/20261014_122557_915676_MaxPool1dIndicesFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261014_122557_915676_MaxPool1dIndicesFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261014_122557_915676_MaxPool1dIndicesFwdOp/report.html`
- run.json status: **`passed`** (CLI exit code 0)
- correctness: **pass** -- `summary.correctness_passed=true`, `correctness_failed=0`.
  - Per-operator rollup (`run.json.operators[0]` = `MaxPool1dIndicesFwdOp`): cases **47**, passed **47**,
    failed **0**, `pass_rate=1.0`, `avg_max_abs_err = 0.0`, `max_abs_err = 0.0` (bit-exact value output,
    `atol=rtol=0`; int64 indices exact).
  - File-level `summary.correctness_tests=75` counts the shared adaptive/max_pool2d/max_pool1d siblings in
    the same `tests/ops/test_pool.py`; all 75 passed. The authoritative per-op view is the rollup above.
  - The `dilation=2` case (`full-default-stride-dilation-fp16`) exercised the scaffold's pre-armed CPU golden
    fallback; **no** `AclNN_*_Error` occurred.
- benchmark: **pass** -- `summary.benchmark_requested=true`, `benchmark_passed=true`,
  `benchmark_tests=36`, `benchmark_failed=0`; `summary.msprof_case_count=24`,
  `invalid_profile_count=0`, `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`.
  - `operators[0].performance_cases=3` (non-smoke only; the reporting layer's `_is_smoke_record` filters
    smoke-labelled records from the per-op performance rollup by design), `ratio_range={min:3.4036, max:5.6990}`.
  - 6 `MaxPool1dIndicesFwdOp` bench records (from `benchmark.json.records`, `result.latency_us` / `result["Ratio(%)"]`):

    | label | dtype | Task Duration (us) | Roofline Ratio (%) |
    |---|---|---|---|
    | smoke-1x8x32-k3s2p1 | float16 | 10.8402 | 0.1121 |
    | smoke-1x8x32-k3s2p1 | bfloat16 | 11.0002 | 0.1105 |
    | smoke-1x8x32-k3s2p1 | float32 | 10.9202 | 0.1049 |
    | sincnet-speaker-local | float16 | 3192.8440 | 3.4036 |
    | textcnn-global | float16 | 138.5228 | 5.6990 |
    | ecg-cnn-dilated | bfloat16 | 1429.5486 | 4.1509 |

  - The reported success criterion for this task is correctness (which passed); the benchmark also passed,
    so the report is a clean `passed` (no `partial`).

## 调试历史

- attempt 1: deterministic integration succeeded on the first run; the first full single-op report
  (`reports/tileops/20261014_122210_526209_MaxPool1dIndicesFwdOp/run.json`) returned
  `status=failed`, `correctness_passed=false` (7/75 file-level failures; per-op 40/47).
  - **Symptom**: `RuntimeError: The size of tensor a (16) must match the size of tensor b (64) at
    non-singleton dimension 3` in `tileops/testing/test_base.py:record_check_result`, i.e. while comparing
    the op output tuple against `ref_program`'s tuple.
  - **Classification**: 「NPU reference 后端能力缺口」(authorized per scheduler context). The stack lands in
    `ref_program`/`record_check_result` **before** the op result is judged; the corrupt object is produced by
    `torch.nn.functional.max_pool1d` on NPU.
  - **Localisation / evidence** (direct probe on device):
    `F.max_pool1d(x_npu, ..., return_indices=True)` returns, for the values, the correct
    `(N, C, L_out)` tensor, but for the indices a **corrupt `torch.int8` `(N, C, kernel_size, L_pad)`**
    internal mask buffer (e.g. `(1, 8, 3, 64)`) instead of the required `int64` `(N, C, L_out)`.
    This affects **every** dilation: `dilation != 1` additionally raises `AclNN_Parameter_Error(EZ1001)`
    (the scaffold's pre-armed CPU fallback already handled that sub-case), while `dilation == 1` runs but
    returns the corrupt indices. Independent check: the migrated op is **bit-exact** against a CPU golden
    (`val_max_diff = 0.000e+00`, `torch.equal(idx_cpu, idx_ref) == True`) for all six parametrized cases,
    i.e. the kernel is correct; only the NPU reference is unusable.
  - **Fix (reference path only; test file backed up)**: in `tests/ops/test_pool.py` (the
    `max_pool1d_with_indices` appended section only):
    1. `MaxPool1dIndicesTest.ref_program`: always compute the golden on CPU
       (`F.max_pool1d(input.cpu(), **kwargs)`) and move both outputs back to the input device, replacing the
       conditional (`dilation != 1`) fallback. Pure-selection semantics -> bit-identical values and indices,
       `atol=rtol=0` preserved.
    2. `test_max_pool1d_indices_nan_window`: its inline reference `F.max_pool1d(x.float(), ...,
       return_indices=True)` (NPU) had the same corrupt-index defect; compute the golden on
       `x.cpu().float()` and move it back.
    3. Same test, the value-at-NaN assertion used `torch.testing.assert_close(..., atol=0, rtol=0)` with the
       default `equal_nan=False`, which is **unsatisfiable** for the all-NaN slice it selects (the sibling
       `test_adaptive_max_pool2d_indices_nan_window` deliberately avoids comparing NaN values). Added
       `equal_nan=True` -- the same NaN-aware exact-comparison convention the repo's own
       `tileops/testing/test_base.allclose_compare` uses. `atol=rtol=0` and the strict int64 index
       comparison are unchanged; no case removed, no tolerance relaxed, no skip/xfail.
    4. Cases, assertions and tolerances otherwise untouched. Backup:
       `tileops/kernels/pool/max_pool1d_with_indices/max_pool1d_with_indices_kernel/history_version/test_pool_s5_attempt1.bak`
       (sha256 `719315747b3c0210d3c8e896d0542070bae8cc9df1919d23b2150c498e9bbdc5`, byte-identical to the
       pre-edit `tests/ops/test_pool.py`). No op/kernel/wrapper/manifest/workload/benchmark file was changed.
  - **Result**: re-ran the full single-op report -> `status=passed`, correctness 47/47 (per-op), benchmark
    36/36 pass.
- No `history_version/` backup of the integration kernel package was needed (the integrated kernel/wrapper
  were never modified after integration).

## Issues

1. **NPU reference-backend gap (recorded)**: `torch.nn.functional.max_pool1d(..., return_indices=True)` is
   not usable as an NPU golden -- `dilation != 1` raises `AclNN_Parameter_Error(EZ1001)`; for every dilation
   it returns a corrupt `int8` `(N, C, kernel_size, L_pad)` indices buffer instead of `int64` `(N, C, L_out)`.
   Handled inside the authorized controlled loop by routing the reference to a CPU golden (pure-selection op).
   The Stage 0 scaffold's fallback only covered `dilation != 1`; it must also cover `dilation == 1`.
2. **Scaffold test-authoring defect (recorded)**: `test_max_pool1d_indices_nan_window` compared its all-NaN
   value slice with `assert_close` under the default `equal_nan=False`, which can never pass regardless of
   backend. Corrected in-place to `equal_nan=True` (repo-standard exact NaN-aware comparison); flagged here
   for the Stage 0 scaffold owner so future NaN-window tests pass `equal_nan=True` (or avoid comparing NaN
   values, as the adaptive sibling does).
3. `--op MaxPool1dIndicesFwdOp` runs the whole shared `tests/ops/test_pool.py` / `benchmarks/ops/bench_pool.py`
   file (no per-op parameter filter). The per-operator rollup (`run.json.operators[0]`) is the authoritative
   per-op view; file-level `summary.correctness_tests=75` counts the shared siblings too.
4. No integration-script `[error]`/`[warn]`; `DESIGN.md` was co-located with the Stage 3 product, so
   `_max_pool1d_with_indices_kernel_DESIGN.md` is present (no `missing` design doc).
