# Integration Log -- AdaptiveMaxPool2dFwdOp

- meta: `examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d/.migration_meta.json`
- op_slug / family: `adaptive_max_pool2d` / `pool`
- manifest op_name: `AdaptiveMaxPool2dFwdOp`
- integrated: `_adaptive_max_pool2d_kernel` -> `adaptive_max_pool2d_kernel/_adaptive_max_pool2d_kernel.py`
- design_docs: `_adaptive_max_pool2d_kernel` -> `adaptive_max_pool2d_kernel/_adaptive_max_pool2d_kernel_DESIGN.md`
- programming mode (from Stage 3 product `ASCEND_MODE`): `Developer`
- attempts: 1
- wrapper: `tileops/kernels/pool/adaptive_max_pool2d/adaptive_max_pool2d.py` (baseline/perf_opt switch block installed; baseline active; `AdaptivePool2dKernelBase.ascend_mode = "Developer"`, `AdaptiveMaxPool2dKernel.ascend_mode = "Developer"`)

## Integrated files

- `tileops/kernels/pool/adaptive_max_pool2d/adaptive_max_pool2d_kernel/_adaptive_max_pool2d_kernel.py`
- `tileops/kernels/pool/adaptive_max_pool2d/adaptive_max_pool2d_kernel/_adaptive_max_pool2d_kernel_DESIGN.md`
- `tileops/kernels/pool/adaptive_max_pool2d/adaptive_max_pool2d_kernel/__init__.py` (aggregation re-export)
- `tileops/kernels/pool/adaptive_max_pool2d/adaptive_max_pool2d.py` (wrapper, rewritten by the integration script)

Integrity: the integrated kernel and design copies are byte-identical to the frozen conductor products
(`examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/{_adaptive_max_pool2d_kernel.py,DESIGN.md}`,
verified with `diff -q`). The sibling `adaptive_avg_pool2d` package, `tileops/manifest/pool.yaml`,
`tileops/ops/pool/`, `tileops/workloads/pool.py`, `tests/ops/test_pool.py` and `benchmarks/ops/bench_pool.py`
were left untouched (append-only family rule; confirmed `adaptive_avg_pool2d_kernel/` intact).

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261013_112226_906143_AdaptiveMaxPool2dFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261013_112226_906143_AdaptiveMaxPool2dFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261013_112226_906143_AdaptiveMaxPool2dFwdOp/report.html`
- benchmark_md: `examples/TileOPs/reports/tileops/20261013_112226_906143_AdaptiveMaxPool2dFwdOp/benchmark.md`
- run.json status: `passed` (CLI exit 0; warnings: [])
- correctness: **pass, 11/11** cases in `tests/ops/test_pool.py` (0 failed/errors/skipped); 5/5 for `AdaptiveMaxPool2dFwdOp` (4 max cases + shared `test_adaptive_pool2d_accepts_chw`), max abs error **0.00e+00** (max selects an input element -> bit-exact):
  - `test_adaptive_max_pool2d[smoke-spp-6x6-fp16]`
  - `test_adaptive_max_pool2d[smoke-asymmetric-bf16]`
  - `test_adaptive_max_pool2d[full-nondiv-fp16]`
  - `test_adaptive_max_pool2d[full-partial-none-fp16]`
  - `test_adaptive_pool2d_accepts_chw` (shared avg/max CHW path)
- benchmark: **pass (valid)** -- 8/8 benchmark tests passed, 6 msprof cases, 0 profiler fallbacks, 0 kernel-filter fallbacks, 0 invalid profiles.

| Label | Shape | DType | Latency (us) | Ratio (%) | Bandwidth (TB/s) | Computility (TOps/s) | msprof valid |
|---|---|---:|---:|---:|---:|---:|---|
| global-1x1 | [8, 2048, 7, 7] out=(1,1) | float16 | 322.1064 | 5.38 | 1.7500 | 3.4243 | true |
| spp-6x6 | [2, 128, 56, 56] out=(6,6) | float16 | 20.4204 | 7.77 | 1.7500 | 3.7076 | true |
| nondiv-7x7 | [2, 64, 55, 57] out=(7,7) | bfloat16 | 17.9804 | 4.69 | 1.7500 | 3.6953 | true |

(smoke-1x8x4x4-2x2 latency 4.0201 us, recorded in benchmark.json but excluded from the report overview.)

## Stage 5 gate self-check (no re-run needed; report ran last)

- `S5-ASCEND-MODE`: kernel module literal `ASCEND_MODE = "Developer"` matches the wrapper's direct-`Kernel`
  class `AdaptivePool2dKernelBase.ascend_mode = "Developer"` (and the concrete `AdaptiveMaxPool2dKernel`).
- `S5-REPORT-STALE`: `run.json` mtime (1791890743) is newer than the wrapper / package `__init__.py` /
  kernel `.py` (1791890523 / 1791890523 / 1791889969); no post-report edits were made to those files.

## 调试历史

none -- integration smoke passed on the first run and the single-op report returned `status=passed`
(correctness 11/11, benchmark valid). No file in the allowed edit scope was modified after integration.

## Issues (recorded, not repaired)

1. `examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py` is not idempotent w.r.t. the
   generated aggregation `__init__.py` (same defect already filed under the `adaptive_avg_pool2d`
   Stage 5 retrospective): once the wrapper is rewritten, a second run's `parse_wrapper_imports()`
   returns no names, so `gen_init_py()` regenerates `__init__.py` from the integrated-file keys only and
   drops the `adaptive_bin` / `max_adaptive_bin_extent` re-exports the wrapper still imports. Avoided here
   by running the script **exactly once** with the environment sourced; the wrapper's original import list
   (`_adaptive_max_pool2d_kernel`, `adaptive_bin`, `max_adaptive_bin_extent`) was therefore read correctly
   and the `__init__.py` re-exports all three.
2. Running `integrate_kernel.py` requires the reproduction environment (`source /workspace/env.sh`):
   the smoke test executes with `cwd=examples/TileOPs` and `sys.executable` (the venv python), and needs
   `tilelang` importable. The `--meta` argument is resolved relative to the current directory, so from
   inside `examples/TileOPs` it must be `tileops/kernels/pool/adaptive_max_pool2d/.migration_meta.json`
   (the repo-root-relative `examples/TileOPs/...` form fails with `FileNotFoundError`).
3. The report's `--op AdaptiveMaxPool2dFwdOp` runs the whole `tests/ops/test_pool.py` (and
   `benchmarks/ops/bench_pool.py`) file rather than filtering to per-op test cases, so the sibling
   `AdaptiveAvgPool2dFwdOp` cases also execute and are reported (11 correctness cases = 6 avg + 4 max +
   1 shared CHW). This is expected for the shared pool test file, but means a sibling-op regression would
   surface in this op's report -- not a max-op defect.
