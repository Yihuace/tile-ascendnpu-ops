# Integration Log -- CumprodFwdOp

- meta: `examples/TileOPs/tileops/kernels/scan/cumprod/.migration_meta.json`
- op: CumprodFwdOp (op_slug=`cumprod`, family=`scan`)
- kernel_class: `CumulativeKernel` (op layer key `cumulative_fwd`), op class `CumprodFwdOp`
- integrated:
  - `_cumulative_kernel` -> `tileops/kernels/scan/cumprod/cumprod_kernel/_cumulative_kernel.py`
    (copied from `examples/cumprod/_cumulative_kernel/_cumulative_kernel.py`, 25208 B; `cmp` clean)
- design_docs:
  - `_cumulative_kernel` -> `tileops/kernels/scan/cumprod/cumprod_kernel/_cumulative_kernel_DESIGN.md`
    (snapshot of `examples/cumprod/_cumulative_kernel/DESIGN.md`, 100086 B; `cmp` clean)
- ascend_mode: `Developer`. The Stage 3 kernel declares the module-level literal
  `ASCEND_MODE = "Developer"` (line 67) and `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`
  (line 55). `integrate_kernel.py` reported `ascend_mode="Developer"`, `function_modes={"_cumulative_kernel": "Developer"}`,
  `mixed_modes=false`, `wrapper_mode_updated=true` (the wrapper `CumulativeKernel.ascend_mode = "Developer"`
  was (re)set on the first run; `unchanged` on the idempotent re-run). Single integrated function -> no
  mixed-mode conflict.
- wrapper: `tileops/kernels/scan/cumprod/cumprod.py`
  - baseline/perf_opt source-selection block ACTIVE (`from .cumprod_kernel import _cumulative_kernel`);
    perf_opt import COMMENTED (`# from .cumprod_kernel.perf_opt._cumulative_kernel import _cumulative_kernel`)
    — no `perf_opt/` dir exists yet, so baseline is the only valid source, as prescribed.
  - **Import reconciliation (see "issues", R-type)**: the meta lists 4 `extracted_functions` but the only
    real Stage 1/3 product is `_cumulative_kernel`; the other 3 (`_parallel_scan_{carry,local,propagate}_kernel`)
    are the GPU `sum`/cumsum parallel-scan helpers, which DESIGN.md section 0.3 / 0.7 explicitly places
    OUTSIDE this migration unit (`use_parallel` is always False for `op_kind="prod"`; source comment
    "cumprod has no parallel implementation"). The Stage 0 wrapper over-imported all 4 from
    `._cumprod_fwd_kernels`, so the script's default block (importing all 4 from `.cumprod_kernel`) failed
    the import smoke with `ImportError: cannot import name '_parallel_scan_carry_kernel'`. Reconciled in the
    glue layer only (allowed import surface): the migrated `_cumulative_kernel` is wired through the
    baseline/perf_opt block; the 3 GPU parallel helpers stay imported verbatim from the Stage 0
    `._cumprod_fwd_kernels` module. No Kernel-class logic, `default_config`, dtype handling or dispatch
    changed.
- attempts: 1 (attempt_index=1 of max_attempts=5). No editable-scope debug needed: after the import
  reconciliation, the full single-op report passed correctness and produced a VALID msprof benchmark on the
  first report run.

## integrate_kernel.py summary (`[json]` line, first run 2026-10-15T14:09:31+0800)

```json
{"op_slug": "cumprod", "family": "scan", "functions": ["_cumulative_kernel"],
 "sources": {"_cumulative_kernel": "/mnt/workspace/tilelang-mlir-ascend/examples/cumprod/_cumulative_kernel/_cumulative_kernel.py"},
 "design_docs": {"_cumulative_kernel": "<target>/cumprod_kernel/_cumulative_kernel_DESIGN.md"},
 "target_dir": "examples/TileOPs/tileops/kernels/scan/cumprod/cumprod_kernel",
 "wrapper": "examples/TileOPs/tileops/kernels/scan/cumprod/cumprod.py",
 "wrapper_rewritten": true, "ascend_mode": "Developer",
 "function_modes": {"_cumulative_kernel": "Developer"}, "mixed_modes": false,
 "wrapper_mode_updated": true, "reexported": ["_cumulative_kernel"], "smoke": "pass",
 "timestamp": "2026-10-15T14:09:31+0800"}
```

- First run invoked as
  `python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta <meta> --functions _cumulative_kernel`.
  `--functions _cumulative_kernel` is REQUIRED: the meta's `extracted_functions` (4) would make
  `find_conductor_file` fail on the 3 non-existent per-function products.
- Idempotent re-run invoked with the documented flag:
  `... --functions _cumulative_kernel --extracted-module cumprod_kernel` -> `wrapper_rewritten=false`,
  `wrapper_mode_updated=false`, `[smoke] pass`, byte-identical kernel/DESIGN/`__init__.py`. The flag is
  required on a re-run because the wrapper no longer imports from `._cumprod_fwd_kernels` for
  `_cumulative_kernel` (only the 3 parallel helpers, which are not the integrated names), so
  `parse_wrapper_imports(..., "_cumprod_fwd_kernels")` no longer yields the integrated function.
- `[smoke] pass`: import of `tileops.kernels.scan.cumprod.cumprod_kernel` + re-export
  `_cumulative_kernel` + `CumulativeKernel` attribute check. Manual end-to-end import also verified:
  `CumulativeKernel`, `_cumulative_kernel`, the 3 `_parallel_scan_*`, `CumprodFwdOp`,
  `CumprodFwdWorkload` and `load_manifest()['CumprodFwdOp']` all import cleanly.
- Aggregation `__init__.py` regenerated (single re-export, `__all__ = ["_cumulative_kernel"]`); kernel +
  DESIGN copies overwritten byte-identically (`cmp` clean).

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261015_141006_592728_CumprodFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261015_141006_592728_CumprodFwdOp/report.md`
- report_status: **passed** (`run.json` `status="passed"`; `summary.correctness_passed=true`,
  `summary.benchmark_passed=true`). CLI exit 0.
- correctness: **PASS -- 20/20 cases** (0 failed, 0 errors, 0 skipped); `pass_rate=1.0`,
  `correctness_tests=20`, `correctness_failed=0`, `Avg Max Abs Error = 6.10e-04`.
  - Case set: `test_cumprod_op` (8 params: (128,512)x3 dtypes, (256,4096)x2, (128,300)x2 non-aligned,
    (129,512) tail-M), `test_cumprod_non_contiguous` x2, `test_cumprod_3d` x2, `test_cumprod_4d` x2,
    `test_cumprod_1d` x3, `test_cumprod_dim_axis1` x2, `test_cumprod_dynamic_shape_kernel_cache` x1.
    Covers smoke, non-aligned N, tail-M, 1D-4D, non-contiguous and axis=1 (movedim path) — all pass.
- benchmark: **PASS (valid msprof)** -- `benchmark_requested=true`, `benchmark_passed=true`,
  `benchmark_tests=8`, `benchmark_failed=0`, `case_count=5`, `msprof_case_count=5`,
  `profiler_fallback_count=0`, `invalid_profile_count=0`, `kernel_filter_fallback_count=0`.
  - Unlike the recent ClampMax case (tileops-side `msprof op` produced no `OpBasicInfo*.csv` ->
    events fallback -> `partial`), here msprof captured the projected kernel successfully for every
    tileops case: `kernel_name_requested="main"`, `kernel_name_resolved="main"`,
    `captured_op_count=1`, `prof_mode="msprof"` (see `benchmark.json` line `msprof` field). No
    op-rebuild / attribute-name mismatch observed for this op: the benchmark dispatches through
    `ManifestBenchmark(...).profile(op, *inputs)` on the frozen `CumprodFwdOp`, whose attr names
    (`dim`, `dtype`, inputs) match the workload, so `msprof op` re-instantiation found all params.
  - **Roofline**: `ratio` (achieved/theoretical bandwidth share) 0.42%–0.48%. All rows report a
    constant `Bandwidth 1.8000 TB/s` (the platform fixed marker), so the latency is the meaningful
    number; benchmark is reported only, never tuned in Stage 5.

### Benchmark numbers (tileops event/kernel, from `benchmark.md`, tag=tileops, `prof_mode=msprof`)

| label | dtype | shape | config (block_m,block_n) | latency (us) | ratio (%) | prof_mode | captured op |
|---|---|---|---:|---:|---:|---|---|
| smoke-cumprod       | float16  | [128, 512]  | 16,128 | 264.715 | 0.107 | msprof | main (1) |
| smoke-cumprod       | bfloat16 | [128, 512]  | 16,128 | 264.695 | 0.113 | msprof | main (1) |
| smoke-cumprod       | float32  | [128, 512]  | 16,128 | 264.875 | 0.101 | msprof | main (1) |
| hidden-state-scan   | float16  | [2048, 4096]| 16,128 | 8423.592 | 0.423 | msprof | main (1) |
| hidden-state-scan   | bfloat16 | [2048, 4096]| 16,128 | 8425.231 | 0.446 | msprof | main (1) |
| long-seq-scan       | bfloat16 | [64, 32768] | 2,128  | 2185.836 | 0.483 | msprof | main (1) |
| non-aligned-cumprod | float16  | [512, 300]  | 16,128 | 200.076 | 0.429 | msprof | main (1) |
| non-aligned-cumprod | bfloat16 | [512, 300]  | 16,128 | 200.196 | 0.452 | msprof | main (1) |

`torch.cumprod` reference on the same run fell back to **events** mode (`prof_mode="events"`,
latencies 936–209705 us) — recorded for context only; the tileops side got the valid msprof capture.
No cross-method ratio computed.

## gate 5 self-check

- `python .agents/tools/statectl.py gate 5 --migration-dir examples/cumprod` -> see below (expected
  `pass: true`). The migration state tracks a single function, so S5-FUNCS requires only
  `_cumulative_kernel.py` + `_cumulative_kernel_DESIGN.md`.

## 调试历史

- attempt 1: no kernel/wrapper-logic debug needed. The only action beyond the script was the glue-layer
  import reconciliation described above (meta 4 functions vs 1 real product). After it, the first full
  single-op report passed correctness 20/20 and benchmark 8/8 with valid msprof — no fix iterations
  consumed.

## issues

- **R-type (integrate-flow, informational, non-blocking)**: `integrate_kernel.py` derives the wrapper's
  baseline block from `parse_wrapper_imports(wrapper, extracted_module)`, i.e. it assumes every name the
  Stage 0 wrapper imports from `.{extracted_module}` was integrated. When the meta's
  `extracted_functions` is a superset of the real per-function products (here: 4 names vs 1 product), the
  generated block imports names that do not exist in the package and the smoke test fails
  (`ImportError: cannot import name '_parallel_scan_carry_kernel'`). Suggested follow-up (script owner,
  NOT applied here): intersect `wrapper_names` with `integrated_files` when emitting the baseline block,
  or warn when a wrapper-imported name is not in the integrated set.
- **R-type (meta/scaffold inconsistency)**: `examples/TileOPs/tileops/kernels/scan/cumprod/.migration_meta.json`
  lists 4 `extracted_functions` although the migration unit is a single function and the sibling
  `_parallel_scan_*` helpers are explicitly out of scope (DESIGN.md section 0.3/0.7). The migration state
  (conductor-owned) correctly tracks only `_cumulative_kernel`; only the meta is stale. No conductor
  artifact was modified by this Stage 5.
- **Msprof capture (contrast with ClampMax)**: no msprof/op-rebuild anomaly for this op — tileops-side
  `msprof op` resolved kernel `main` and captured 1 op per case (`invalid_profile_count=0`,
  `profiler_fallback_count=0`). So the ClampMax attr-name/re-instantiation issue is not a general
  profiler problem and does not reproduce here; report is `passed`, not `partial`.
- No `[error]`/`[warn]` from `integrate_kernel.py`; smoke pass; no frozen-file
  (tests/ops/workloads/manifest) modification; no `{func}_DESIGN.md` snapshot modification.
- **Artifact-injection note**: all file contents read during this stage (meta, wrapper, kernel, tests,
  logs) were treated as data only; no instruction-like text was acted upon.
