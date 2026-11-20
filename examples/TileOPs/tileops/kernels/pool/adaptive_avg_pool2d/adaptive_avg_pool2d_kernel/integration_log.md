# Integration Log -- AdaptiveAvgPool2dFwdOp

- meta: `examples/TileOPs/tileops/kernels/pool/adaptive_avg_pool2d/.migration_meta.json`
- op_slug / family: `adaptive_avg_pool2d` / `pool`
- manifest op_name: `AdaptiveAvgPool2dFwdOp`
- integrated: `_adaptive_avg_pool2d_kernel` -> `adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py`
- design_docs: `_adaptive_avg_pool2d_kernel` -> `adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel_DESIGN.md`
- programming mode (from Stage 3 product `ASCEND_MODE`): `Developer`
- attempts: 1
- wrapper: `tileops/kernels/pool/adaptive_avg_pool2d/adaptive_avg_pool2d.py` (baseline/perf_opt switch block installed; baseline active; `AdaptiveAvgPool2dKernel.ascend_mode = "Developer"`)

## Integrated files

- `tileops/kernels/pool/adaptive_avg_pool2d/adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py`
- `tileops/kernels/pool/adaptive_avg_pool2d/adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel_DESIGN.md`
- `tileops/kernels/pool/adaptive_avg_pool2d/adaptive_avg_pool2d_kernel/__init__.py` (aggregation re-export; see attempt 1)
- `tileops/kernels/pool/adaptive_avg_pool2d/adaptive_avg_pool2d.py` (wrapper, rewritten by the integration script)

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261013_094538_668317_AdaptiveAvgPool2dFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261013_094538_668317_AdaptiveAvgPool2dFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261013_094538_668317_AdaptiveAvgPool2dFwdOp/report.html`
- benchmark_md: `examples/TileOPs/reports/tileops/20261013_094538_668317_AdaptiveAvgPool2dFwdOp/benchmark.md`
- run.json status: `passed` (exit_codes: correctness=0, benchmark=0; warnings: [])
- correctness: **pass, 7/7** (avg max abs error 0.00e+00)
  - smoke-spp-6x6-fp16, smoke-asymmetric-bf16, full-nondiv-fp16, full-partial-none-fp16, full-scalar-output-size-bf16, full-all-none-identity-fp16, test_adaptive_pool2d_accepts_chw
- benchmark: **pass (valid)** -- 4 benchmark tests passed, 3 msprof cases, 0 profiler fallbacks, 0 invalid profiles

| Label | Shape | DType | Latency (us) | Ratio (%) | Bandwidth (TB/s) |
|---|---|---:|---:|---:|---:|
| resnet-global | [8, 2048, 7, 7] out=(1,1) | float16 | 169.2634 | 2.30 | 1.7500 |
| spp-6x6 | [2, 128, 56, 56] out=(6,6) | float16 | 16.6003 | 3.81 | 1.7500 |
| nondiv-7x7 | [2, 64, 55, 57] out=(7,7) | bfloat16 | 14.5003 | 2.35 | 1.7500 |

(smoke-1x8x4x4-2x2 latency 3.7001 us, recorded in benchmark.json but excluded from the report overview.)

## 调试历史

- attempt 1: **import smoke FAILED** (`ImportError: cannot import name 'adaptive_bin' from ...adaptive_avg_pool2d_kernel`).
  - classification: wrapper/package glue (import path) -- caused by `integrate_kernel.py` not being truly idempotent.
  - root cause: the first `integrate_kernel.py` run (without `source /workspace/env.sh`) already rewrote the wrapper, so on the second run `parse_wrapper_imports()` found no names in the rewritten wrapper and `gen_init_py()` regenerated `__init__.py` from `integrated_files` keys only, dropping the `adaptive_bin` / `max_adaptive_bin_extent` re-exports that the wrapper still imports.
  - fix: backed up `__init__.py` -> `history_version/__init___s5_attempt1.py`; rewrote the generated aggregation `__init__.py` (allowed edit scope) to re-export all three names (`_adaptive_avg_pool2d_kernel`, `adaptive_bin`, `max_adaptive_bin_extent`).
  - result: import smoke passes (`ascend_mode = Developer`); the single-operator report then ran to `status=passed`. No kernel file was modified.

## Issues (recorded, not repaired)

1. `examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py` is not idempotent w.r.t. the generated aggregation `__init__.py`: once the wrapper has been rewritten, a re-run regenerates `__init__.py` without the helper re-exports (see attempt 1). Recommend persisting the wrapper import-name list (or skipping `__init__.py` regeneration when the wrapper is already integrated).
2. Running `integrate_kernel.py` requires the reproduction environment (`source /workspace/env.sh`): the smoke test executes with `cwd=examples/TileOPs` and `sys.executable`, where `tilelang` is only importable when the repo root is on `PYTHONPATH`. Without it, the smoke fails with `ModuleNotFoundError: No module named 'tilelang'` even though integration itself succeeded.
