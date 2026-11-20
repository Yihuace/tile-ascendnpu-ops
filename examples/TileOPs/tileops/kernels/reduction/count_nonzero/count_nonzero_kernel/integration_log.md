# Integration Log -- CountNonzeroFwdOp

- meta: examples/TileOPs/tileops/kernels/reduction/count_nonzero/.migration_meta.json
- op_slug / family: count_nonzero / reduction
- integrated:
  - `_logical_reduce_kernel` -> `tileops/kernels/reduction/count_nonzero/count_nonzero_kernel/_logical_reduce_kernel.py` (md5 `64a051c3f6fe49f8f53a4e1b9cc8beb0` matches the Stage 3 product)
- design_docs:
  - `_logical_reduce_kernel` -> `tileops/kernels/reduction/count_nonzero/count_nonzero_kernel/_logical_reduce_kernel_DESIGN.md`
    (source: `examples/count_nonzero/_logical_reduce_kernel/DESIGN.md`; none missing)
- wrapper: `tileops/kernels/reduction/count_nonzero/count_nonzero.py`
  - baseline import active (`from .count_nonzero_kernel import _logical_reduce_kernel`);
    perf_opt import commented out as a Stage 4 placeholder
  - wrapper `default_config` re-wired to the kernel's tiling single source of truth
    `_select_config(self.M, self.N, dtype_str)["block_m"]` (DESIGN.md sections 5.2 / 5.6)
- wrapper ascend_mode: `Developer` (synced from Stage 3 `ASCEND_MODE` by integrate_kernel.py)
- aggregation init: `count_nonzero_kernel/__init__.py` re-exports `_logical_reduce_kernel`
- attempts: 1

## Integration script run

- command: `python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/reduction/count_nonzero/.migration_meta.json`
  (from `examples/TileOPs/`, `PYTHONPATH=/workspace/tilelang-mlir-ascend`)
- result: `[ok] integration complete`; smoke `smoke-ok ['_logical_reduce_kernel']`;
  ascend_mode `Developer`; `design_docs` mapped for `_logical_reduce_kernel`.
- note: the first invocation failed its in-script import smoke with
  `ModuleNotFoundError: No module named 'tilelang'` — the repo root is not on the
  default site path; re-running with `PYTHONPATH=/workspace/tilelang-mlir-ascend`
  made the smoke pass. This is an environment/PYTHONPATH issue, not an integration defect.

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261016_015623_851978_CountNonzeroFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261016_015623_851978_CountNonzeroFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261016_015623_851978_CountNonzeroFwdOp/report.html`
- report_status: **passed** (`status=passed`)
- correctness: **pass** — 37/37 cases (`summary.correctness_passed=true`, `pass_rate=1.0`, 0 failed)
- benchmark: **pass** — `summary.benchmark_passed=true`, 7/7 benchmark tests passed;
  4/4 report performance cases have valid msprof profiles
  (`invalid_profile_count=0`, `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`)

### Benchmark msprof values per workload (latency_us = msprof Task Duration)

| label | shape | dtype | dim | config | latency_us | roofline util | bandwidth |
|---|---|---|---|---|---|---|---|
| sparsity-hidden | [2048, 4096] | float16 | -1 | block_m=2 | 22.7405 | 42.16% | 1.75 TB/s |
| sparsity-hidden | [2048, 4096] | bfloat16 | -1 | block_m=2 | 29.2006 | 32.83% | 1.75 TB/s |
| sparsity-seq | [32, 16384] | float16 | -1 | block_m=1 (path C) | 4.5001 | 13.32% | 1.75 TB/s |
| 3d-multidim-reduce | [4, 128, 4096] | float16 | [0, 2] | block_m=1 (path C) | 9.6602 | 24.81% | 1.75 TB/s |

> The three `smoke-count-nonzero` rows are recorded in `benchmark.json` (label contains
> "smoke", so the reporting layer intentionally keeps them out of `report.md` /
> `performance.cases`). This is a reporting-layer convention, not a profile failure.

## 调试历史

### attempt 1 — wrapper `default_config` re-wiring (preempted; no report failure)

- **症状（预防性修复，非 report 失败）**: the wrapper still carried the GPU-era 48 KiB
  SMEM heuristic in `default_config`. For the manifest workload `(2048, 4096)` it selects
  `block_m=8` -> `8 x 4096 = 32768` elements/block > the measured UB hard limit of
  24576 elements (DESIGN.md section 4.5, 8 B/element) -> compile-time `ub overflow`.
  It also cannot express the `N > E_CAP` column-tiled path.
- **分类**: wrapper 胶水错误 (glue config; identical defect class to
  `CASE-argmax-argreduce-integration` defect ②).
- **修复**: wrapper glue only — `count_nonzero.py` `default_config` now delegates to
  `_select_config(self.M, self.N, self.dtype_to_str(self._kernel_dtype))["block_m"]`
  (added `from .count_nonzero_kernel._logical_reduce_kernel import _select_config`),
  and the class docstring's stale "keep padded width within the single-tile column
  budget" note was updated to reflect `N <= 2**24` with automatic column tiling
  (DESIGN.md section 5.6). The kernel file, tests, benchmark, manifest, workloads, and
  the `_logical_reduce_kernel_DESIGN.md` snapshot were NOT touched.
  Backup: `count_nonzero_kernel/history_version/count_nonzero_s5_attempt1.py`.
- **结果**: single report run -> `status=passed`, correctness 37/37, benchmark 7/7
  (4 valid msprof cases).
- **config cross-check** (matches DESIGN.md section 5.2 frozen table): `(32,256)` -> 8,
  `(2048,4096)` -> 2, `(256,4096)` -> 2, `(128,300)` -> 8, `(32,16384)` -> path C bm=1,
  `(128,16384)` -> path C bm=1, bool input -> float32 storage bm=8.

## issues

- none blocking. Notes:
  - The wrapper still imports `device_smem_budget` / `DEFAULT_ALIGNMENT` / `align_up` from
    `_primitives` and computes `self._smem_budget` / `self._elem_bytes` though the new
    `default_config` no longer uses them (left untouched after the passing run to keep the
    verified artifact stable; harmless unused imports).
  - `report.md`'s `Kernel` column shows `main` for the column-tiled path even though the
    factory sets `msprof_kernel_name` (`count_nonzero_tiled`); same cosmetic reporting-layer
    behaviour seen for `ArgminFwdOp` — not an integration defect.
  - No NPU reference backend gaps hit: `count_nonzero`'s golden is the CPU
    `torch.count_nonzero`; all reference paths are pure-selection and ran without
    `AclNN_*_Error`.
