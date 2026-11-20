# Integration Log -- MaxPool1dFwdOp

- meta: `examples/TileOPs/tileops/kernels/pool/max_pool1d/.migration_meta.json`
- op_slug / family: `max_pool1d` / `pool`
- manifest op_name: `MaxPool1dFwdOp`
- integrated: `_max_pool1d_kernel` -> `max_pool1d_kernel/_max_pool1d_kernel.py`
- design_docs: `_max_pool1d_kernel` -> `max_pool1d_kernel/_max_pool1d_kernel_DESIGN.md` (present, copied)
- programming mode (from Stage 3 product `ASCEND_MODE`): `Developer`
- attempts: 1 (deterministic integration + full single-op report; no in-scope file modified)
- wrapper: `tileops/kernels/pool/max_pool1d/max_pool1d.py` (baseline/perf_opt switch block installed;
  baseline active; `MaxPool1dKernel.ascend_mode = "Developer"`)

## Integrated files

- `tileops/kernels/pool/max_pool1d/max_pool1d_kernel/_max_pool1d_kernel.py`
- `tileops/kernels/pool/max_pool1d/max_pool1d_kernel/_max_pool1d_kernel_DESIGN.md`
- `tileops/kernels/pool/max_pool1d/max_pool1d_kernel/__init__.py` (aggregation re-export: `_max_pool1d_kernel`, `pool_output_dim`)
- `tileops/kernels/pool/max_pool1d/max_pool1d.py` (wrapper, import block rewritten by the integration script)

Integrity: the integrated kernel and design copies are byte-identical to the frozen conductor products
(`examples/max_pool1d/_max_pool1d_kernel/{_max_pool1d_kernel.py,DESIGN.md}`, verified with `diff -q`).
Integration script summary (`[json]` last line): `ascend_mode="Developer"`, `wrapper_rewritten=true`,
`wrapper_mode_updated=true`, `smoke="pass"`, reexported `["_max_pool1d_kernel", "pool_output_dim"]`,
no `[error]` / no `[warn]`.

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261014_085626_616055_MaxPool1dFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261014_085626_616055_MaxPool1dFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261014_085626_616055_MaxPool1dFwdOp/report.html`
- run.json status: **`partial`** (CLI exit non-zero; `warnings` = correctness passed but benchmark
  produced no valid records / 6 failed benchmark cases -- see below)
- correctness: **pass** -- `summary.correctness_passed=true`, `correctness_failed=0`.
  - Per-operator rollup (`run.json.operators[0]`, `MaxPool1dFwdOp`): cases **34**, passed **34**,
    failed **0**, `pass_rate=1.0`, `avg_max_abs_err = 0.00e+00`, `max_abs_err = 0.00e+00` (bit-exact,
    `atol=rtol=0`).
  - File-level `summary.correctness_tests=55` counts the shared adaptive/max_pool2d siblings in the
    same `tests/ops/test_pool.py`; all 55 passed. The authoritative per-op view is the rollup above.
  - The pre-armed NPU reference workaround in `MaxPool1dTest.ref_program` (CPU golden fallback for
    `dilation != 1`, added by the Stage 0 scaffold from the `max_pool2d` lesson) was exercised by the
    `ecg-cnn-dilated` (`dilation=2`) case and passed; **no** `AclNN_*_Error` occurred.
- benchmark: **partial / not valid for MaxPool1dFwdOp** -- `benchmark_requested=true`,
  `benchmark_passed=false`, `benchmark_tests=30`, `benchmark_passed=24`, `benchmark_failed=6`.
  - All 6 failures are the `MaxPool1dFwdOp` benchmark cases:
    `test_max_pool1d_bench[smoke-1x8x32-k3s2p1-{float16,bfloat16,float32}]`,
    `[sincnet-speaker-local-float16]`, `[textcnn-global-float16]`, `[ecg-cnn-dilated-bfloat16]`.
  - Root cause (recorded, **not repaired** -- `benchmarks/` is out of the Stage 5 edit scope):
    `benchmarks/ops/bench_pool.py:161` does `kernel_size = tuple(op_params["kernel_size"])`, but
    `workloads_to_params("MaxPool1dFwdOp", include_extra=True)` supplies `kernel_size` as a plain
    `int` (e.g. `{'kernel_size': 3, 'stride': 2, 'padding': 1}`). `tuple(3)` raises
    `TypeError: 'int' object is not iterable`, so every 1D bench case aborts before profiling.
    (`stride`/`padding`/`dilation` are wrapped the same way, but the first wrapped value
    `kernel_size` aborts first.)
  - The successful benchmark records in this run are the shared adaptive/max_pool2d siblings
    (24 records); no `MaxPool1dFwdOp` performance numbers exist, so its per-op rollup reports
    `performance_cases=0`, `ratio_range=null`.
  - `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`, `invalid_profile_count=0`.

## 调试历史

- attempt 1: deterministic integration succeeded; full single-op report ran once.
  - correctness passed (MaxPool1dFwdOp 34/34, bit-exact); **no** correctness failure -> no in-scope
    repair needed.
  - benchmark reported 6 failures, all in `benchmarks/ops/bench_pool.py` (out-of-scope benchmark
    scaffolding defect). Per Stage 5 contract benchmark failures are **recorded only, not repaired
    and not retried**; the reported success criterion for this task is correctness (which passed).
  - No `history_version/` backup taken (no file modified).

## Issues (recorded, not repaired)

1. `benchmarks/ops/bench_pool.py:161` (`test_max_pool1d_bench`) wraps a scalar `kernel_size` with
   `tuple(...)`, raising `TypeError: 'int' object is not iterable` for all 6 `MaxPool1dFwdOp`
   benchmark cases. Root cause is the Stage 0 scaffolded bench section, which assumes tuple-valued
   workload extras while `tileops/workloads/pool.py:MaxPool1dWorkload` extras carry scalars. This is
   in `benchmarks/` (out of scope) and blocks the benchmark stage only -- correctness is unaffected.
   Suggested fix (for the conductor / Scaffolder, not applied here): in the 1D bench, coerce scalars
   before wrapping, e.g. `_as_tuple = lambda v: (v,) if isinstance(v, int) else tuple(v)` and use it
   for `kernel_size`/`stride`/`padding`/`dilation`.
2. `--op MaxPool1dFwdOp` runs the whole shared `tests/ops/test_pool.py` / `benchmarks/ops/bench_pool.py`
   file (no per-op parameterized filter). The per-operator rollup (`run.json.operators[0]`) is the
   authoritative per-op view; the file-level `summary.correctness_tests=55` counts shared siblings too.
3. No integration-script `[error]`/`[warn]`; `DESIGN.md` was co-located with the Stage 3 product, so
   `_max_pool1d_kernel_DESIGN.md` is present (no `missing` design doc).

---

## Stage 5 补采 (benchmark re-run) — 2026-10-14

- trigger: 用户发起的集成后补采。上一轮 report（`reports/tileops/20261014_085626_616055_MaxPool1dFwdOp/`）
  correctness 通过但 `benchmark_passed=false` / `performance_cases=0`，根因是 Stage 0 脚手架在
  `benchmarks/ops/bench_pool.py` MaxPool1d 段对**标量** `kernel_size/stride/padding/dilation` 直接
  `tuple(...)` 报 `TypeError: 'int' object is not iterable`。该缺陷**已由 scaffolder 修复**（新增段内 helper
  `_max_pool1d_as_tuple`：标量 → 长度 1 tuple；`benchmarks/ops/bench_pool.py:153`；Tier 1 通过）。
  集成 kernel / wrapper **未变**；本次不重跑集成脚本。
- 命令（工作目录 `examples/TileOPs/`，逐字执行，串行）:
  `ASCEND_RT_VISIBLE_DEVICES=0 flock /tmp/tilelang-msprof.lock python -m tileops.reporting.cli run --op MaxPool1dFwdOp --prof-mode msprof --reports-dir reports/verification`
  （无 `--all`；未改 tests / benchmarks / manifest / ops / workloads / 集成脚本产物）
- report_run_json: `examples/TileOPs/reports/verification/20261014_093943_522105_MaxPool1dFwdOp/run.json`
- report_md: `examples/TileOPs/reports/verification/20261014_093943_522105_MaxPool1dFwdOp/report.md`
- run_id: `20261014_093943_522105_MaxPool1dFwdOp`
- run.json status: **`passed`**（CLI exit code 0）

### 验证结果（本次 run.json）

- correctness: **pass** — `summary.correctness_passed=true`, `correctness_failed=0`。
  - per-operator rollup (`operators[0]` = `MaxPool1dFwdOp`): cases **34**, passed **34**, failed **0**,
    `pass_rate=1.0`, `avg_max_abs_err=0.0`, `max_abs_err=0.0`（bit-exact，`atol=rtol=0`）。
    file-level `summary.correctness_tests=55`（共享 adaptive/max_pool2d siblings），全部通过。
- benchmark: **pass** — `summary.benchmark_requested=true`, `benchmark_passed=true`,
  `benchmark_tests=30`, `benchmark_failed=0`。
  - `summary.msprof_case_count=21`；`summary.invalid_profile_count=0`；
    `profiler_fallback_count=0`; `kernel_filter_fallback_count=0`。
  - `operators[0].performance_cases=3`（仅非 smoke 用例进入 per-op performance rollup），
    `ratio_range={min:13.7702, max:19.0826}`。
  - 说明：`benchmark.json` 记录了全部 **6** 条 `MaxPool1dFwdOp` 用例 pass；但 `run.json` 的
    performance 段与 `operators[0].performance_cases` 只保留 **3** 条，因为 reporting 层
    `tileops/reporting/analyzer.py:_is_smoke_record` 按设计**过滤 label 含 `smoke` 的记录**
    （framework 既定行为，非缺陷；3 条 smoke 用例的 Duration/Ratio 从 `benchmark.json` 读取）。
  - 6 条 `MaxPool1dFwdOp` bench 用例（来自 `benchmark.json` `records`，`result.latency_us` /
    `result["Ratio(%)"]`）：
    | label | dtype | Task Duration (us) | Roofline Ratio (%) |
    |---|---|---|---|
    | smoke-1x8x32-k3s2p1 | float16 | 3.440069 | 0.1147 |
    | smoke-1x8x32-k3s2p1 | bfloat16 | 3.380068 | 0.1167 |
    | smoke-1x8x32-k3s2p1 | float32 | 3.440069 | 0.0945 |
    | sincnet-speaker-local | float16 | 301.686035 | 14.7839 |
    | textcnn-global | float16 | 12.560251 | 19.0826 |
    | ecg-cnn-dilated | bfloat16 | 154.583084 | 13.7702 |

### 调试历史

- 本次为串行补采，**无调试**（无文件修改、无 `history_version/` 备份）。bench 缺陷由 scaffolder 在
  Stage 0 侧修复，集成 kernel / wrapper 未变。benchmark 由 fail（`benchmark_passed=false`, 6 failed）
  转为 pass（`benchmark_passed=true`, 0 failed）。

### Issues

1. `run.json` performance 段只含 3/6 `MaxPool1dFwdOp` 用例系 reporting 层 `_is_smoke_record`
   的既定 smoke 过滤（见上）；如需在 per-op rollup 中体现 smoke 用例数值，属 reporting 层口径问题，
   非本次集成缺陷，未修改。
2. `--op MaxPool1dFwdOp` 仍运行整份共享 `tests/ops/test_pool.py` / `benchmarks/ops/bench_pool.py`
   （无 per-op 参数过滤）；per-operator rollup（`run.json.operators[0]`）为权威 per-op 视图。
3. 依本次任务显式范围（不得改动 conductor 产物目录），未向 `examples/max_pool1d/RETROSPECTIVE.md`
   追加补采章节；既有 Stage 5 章节已覆盖该 bench 缺陷的 R/C 价值点。
