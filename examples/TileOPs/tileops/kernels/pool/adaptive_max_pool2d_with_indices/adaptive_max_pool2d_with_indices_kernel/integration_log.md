# Integration Log -- AdaptiveMaxPool2dIndicesFwdOp

- meta: `examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d_with_indices/.migration_meta.json`
- op_slug / family: `adaptive_max_pool2d_with_indices` / `pool`
- manifest op_name: `AdaptiveMaxPool2dIndicesFwdOp`
- integrated: `_adaptive_max_pool2d_with_indices_kernel` -> `adaptive_max_pool2d_with_indices_kernel/_adaptive_max_pool2d_with_indices_kernel.py`
- design_docs: `_adaptive_max_pool2d_with_indices_kernel` -> `adaptive_max_pool2d_with_indices_kernel/_adaptive_max_pool2d_with_indices_kernel_DESIGN.md`
- programming mode (from Stage 3 product `ASCEND_MODE`): `Developer`
- attempts: 1
- wrapper: `tileops/kernels/pool/adaptive_max_pool2d_with_indices/adaptive_max_pool2d_with_indices.py` (baseline/perf_opt switch block installed; baseline active; `AdaptivePool2dKernelBase.ascend_mode = "Developer"`, `AdaptiveMaxPool2dWithIndicesKernel.ascend_mode = "Developer"`)

## Integrated files

- `tileops/kernels/pool/adaptive_max_pool2d_with_indices/adaptive_max_pool2d_with_indices_kernel/_adaptive_max_pool2d_with_indices_kernel.py`
- `tileops/kernels/pool/adaptive_max_pool2d_with_indices/adaptive_max_pool2d_with_indices_kernel/_adaptive_max_pool2d_with_indices_kernel_DESIGN.md`
- `tileops/kernels/pool/adaptive_max_pool2d_with_indices/adaptive_max_pool2d_with_indices_kernel/__init__.py` (aggregation re-export: kernel fn + `adaptive_bin` + `max_adaptive_bin_extent`)
- `tileops/kernels/pool/adaptive_max_pool2d_with_indices/adaptive_max_pool2d_with_indices.py` (wrapper, rewritten by the integration script)

Integrity: the integrated kernel and design copies are byte-identical to the frozen conductor products
(`examples/adaptive_max_pool2d_with_indices/_adaptive_max_pool2d_with_indices_kernel/{_adaptive_max_pool2d_with_indices_kernel.py,DESIGN.md}`,
verified with `diff -q` -> identical). The sibling `adaptive_max_pool2d/` (plain) and
`adaptive_avg_pool2d/` packages were left untouched (append-only family rule; `find ... -newermt
"2026-10-14 00:00"` returns no files), as were `tileops/manifest/pool.yaml`, `tileops/ops/pool/`,
`tileops/workloads/pool.py`, `tests/ops/test_pool.py` and `benchmarks/ops/bench_pool.py`.

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261014_035238_349246_AdaptiveMaxPool2dIndicesFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261014_035238_349246_AdaptiveMaxPool2dIndicesFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261014_035238_349246_AdaptiveMaxPool2dIndicesFwdOp/report.html`
- benchmark_md: `examples/TileOPs/reports/tileops/20261014_035238_349246_AdaptiveMaxPool2dIndicesFwdOp/benchmark.md`
- run.json status: `passed` (CLI exit 0; warnings: [])
- correctness: **pass, 17/17** cases in `tests/ops/test_pool.py` (0 failed/errors/skipped). Per-operator rollup
  (`operators[0]`, the authoritative verdict for this op): **`AdaptiveMaxPool2dIndicesFwdOp` 7/7, pass_rate 1.0,
  avg_max_abs_err 0.00e+00** (max selects an input element -> bit-exact; int64 indices compared with
  `torch.equal` exact oracle). Cases attributed to this op:
  - `test_adaptive_max_pool2d_indices[smoke-spp-6x6-fp16]`
  - `test_adaptive_max_pool2d_indices[smoke-asymmetric-bf16]`
  - `test_adaptive_max_pool2d_indices[full-nondiv-fp16]`
  - `test_adaptive_max_pool2d_indices[full-partial-none-fp16]`
  - `test_adaptive_max_pool2d_indices_accepts_chw` (CHW path)
  - `test_adaptive_max_pool2d_indices_nan_window` (NaN window -> NaN value + last-NaN index)
  - `test_adaptive_pool2d_accepts_chw` (shared avg/max/indices CHW path)
  (The other 10 cases are the sibling `AdaptiveAvgPool2dFwdOp` (6) and `AdaptiveMaxPool2dFwdOp` (4) in the
  shared family file; all pass, not this op's rollup.)
- benchmark: **pass (valid)** -- 12/12 benchmark tests passed, 9 msprof cases, 0 profiler fallbacks,
  0 kernel-filter fallbacks, 0 invalid profiles (`invalid_profile_count=0`). Our operator's 4 bench cases
  (smoke + 3 profiled shapes) all passed:

| Label | Shape | DType | Latency (us) | Ratio (%) | Bandwidth (TB/s) | Computility (TOps/s) | msprof valid |
|---|---|---:|---:|---:|---:|---:|---|
| global-1x1 | [8, 2048, 7, 7] out=(1,1) | float16 | 151.6030 | 2.54 | 1.7500 | 3.4695 | true |
| spp-6x6 | [2, 128, 56, 56] out=(6,6) | float16 | 85.3617 | 4.57 | 1.7500 | 3.5859 | true |
| nondiv-7x7 | [2, 64, 55, 57] out=(7,7) | bfloat16 | 81.6016 | 2.56 | 1.7500 | 3.5774 | true |

(smoke-1x8x4x4-2x2 latency 5.9201 us, recorded in benchmark.json but excluded from the report overview.)

## Stage 5 gate self-check (no re-run needed; report ran last)

- `S5-ASCEND-MODE`: kernel module literal `ASCEND_MODE = "Developer"` matches the wrapper's direct-`Kernel`
  class `AdaptivePool2dKernelBase.ascend_mode = "Developer"` (and the concrete `AdaptiveMaxPool2dWithIndicesKernel`).
- `S5-REPORT-STALE`: `run.json` mtime (1791950269) is newer than the wrapper / package `__init__.py` /
  integrated kernel `.py` / aggregation `__init__.py` (1791949936 / 1791939554 / 1791949811 / 1791949936);
  no post-report edits were made to those files.

## 调试历史

none -- integration smoke passed on the first run and the single-op report returned `status=passed`
(correctness 17/17 file-wide, 7/7 for this op, benchmark valid). No file in the allowed edit scope was
modified after integration; the script was run exactly once.

## Issues (recorded, not repaired)

1. `examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py` is not idempotent w.r.t. the
   generated aggregation `__init__.py` (same defect already filed under the `adaptive_avg_pool2d` and
   `adaptive_max_pool2d` Stage 5 retrospectives): once the wrapper is rewritten, a second run's
   `parse_wrapper_imports()` returns no names, so `gen_init_py()` regenerates `__init__.py` from the
   integrated-file keys only and drops the `adaptive_bin` / `max_adaptive_bin_extent` re-exports the wrapper
   still imports. Avoided here by running the script **exactly once** with the environment sourced; the
   wrapper's original import list (`_adaptive_max_pool2d_with_indices_kernel`, `adaptive_bin`,
   `max_adaptive_bin_extent`) was therefore read correctly and the `__init__.py` re-exports all three.
2. Running `integrate_kernel.py` requires the reproduction environment (`source /workspace/env.sh`): the
   smoke test executes with `cwd=examples/TileOPs` and `sys.executable` and needs `tilelang` importable.
   The `--meta` argument is resolved relative to the current directory, so from inside `examples/TileOPs`
   it must be `tileops/kernels/pool/adaptive_max_pool2d_with_indices/.migration_meta.json` (the
   repo-root-relative `examples/TileOPs/...` form fails with `FileNotFoundError`).
3. The report's `--op AdaptiveMaxPool2dIndicesFwdOp` runs the whole `tests/ops/test_pool.py` (and
   `benchmarks/ops/bench_pool.py`) file rather than filtering to per-op test cases, so the sibling
   `AdaptiveAvgPool2dFwdOp` / `AdaptiveMaxPool2dFwdOp` cases also execute and are reported (17 correctness
   cases = 6 avg + 4 max + 7 indices incl. shared CHW). This is expected for the shared pool test file; the
   authoritative per-op verdict is the `operators` rollup, where this op is 7/7.
