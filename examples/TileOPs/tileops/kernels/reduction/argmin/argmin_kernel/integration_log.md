# Integration Log -- ArgminFwdOp

- meta: examples/TileOPs/tileops/kernels/reduction/argmin/.migration_meta.json
- op_slug / family: argmin / reduction
- integrated:
  - `_argreduce_kernel` -> `tileops/kernels/reduction/argmin/argmin_kernel/_argreduce_kernel.py`
- design_docs:
  - `_argreduce_kernel` -> `tileops/kernels/reduction/argmin/argmin_kernel/_argreduce_kernel_DESIGN.md` (source: `examples/argmin/_argreduce_kernel/DESIGN.md`; none missing)
- wrapper: `tileops/kernels/reduction/argmin/argmin.py` (baseline import active; perf_opt import commented)
- wrapper ascend_mode: Developer (synced from Stage 3 `ASCEND_MODE`; unchanged)
- aggregation init: `tileops/kernels/reduction/argmin/argmin_kernel/__init__.py` re-exports `_argreduce_kernel`
- attempts: 1 (one controlled debug attempt)

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261014_160246_645105_ArgminFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261014_160246_645105_ArgminFwdOp/report.md`
- report_html: `examples/TileOPs/reports/tileops/20261014_160246_645105_ArgminFwdOp/report.html`
- correctness: **pass** — 42/42 cases (`summary.correctness_passed=true`, `pass_rate=1.0`, 0 failed)
- benchmark: **pass** — `summary.benchmark_passed=true`, 9/9 benchmark tests passed; 6/6 report performance cases have valid msprof profiles (`invalid_profile_count=0`, `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`)
- report_status: passed

### Benchmark msprof values per workload (latency_us = msprof Task Duration)

| label | shape | dtype | config | latency_us | msprof valid |
|---|---|---|---|---|---|
| smoke-argmin | [32, 256] | float16 | block_m=8 | 1.8000 | yes (in benchmark.json; label-filtered from report) |
| smoke-argmin | [32, 256] | bfloat16 | block_m=8 | 1.8600 | yes (in benchmark.json; label-filtered from report) |
| smoke-argmin | [32, 256] | float32 | block_m=8 | 1.8800 | yes (in benchmark.json; label-filtered from report) |
| lm-head-argmin | [4, 102400] | float16 | block_m=1 (tiled) | 26.0805 | yes |
| lm-head-argmin | [4, 102400] | bfloat16 | block_m=1 (tiled) | 31.6806 | yes |
| hidden-state-argmin | [2048, 4096] | float16 | block_m=2 | 31.1606 | yes |
| hidden-state-argmin | [2048, 4096] | bfloat16 | block_m=1 | 54.1011 | yes |
| non-aligned-argmin | [256, 300] | float16 | block_m=8 | 3.1001 | yes |
| non-aligned-argmin | [256, 300] | bfloat16 | block_m=8 | 3.3201 | yes |

> The 3 `smoke-argmin` rows are present in `benchmark.json` (valid msprof records) but
> are intentionally excluded from `run.json` `performance.cases` / `report.md` by
> `tileops/reporting/analyzer.py:_is_smoke_record` (label containing "smoke" never
> leaks into performance reports). Their values are quoted here from `benchmark.json`.
> This is a reporting-layer convention, not a profile failure.

## 调试历史

### attempt 1 — BiShengIR `ub overflow` on N_padded=4096 (fp16/bf16)

- **症状**: first report run `reports/tileops/20261014_160039_216067_ArgminFwdOp`
  → `status=failed`, correctness 40/42; 2 failed cases both `test_argmin_op[256-4096-*]`:
  `RuntimeError: NPU IR opt failed ... ub overflow, requires 2359808 bits while 1572864 bits available`
  (fp16), `... 2361856 bits ...` (bf16). The two failing cases 100% coincide with N_padded=4096.
- **分类**: 编译/运行失败 — BiShengIR PlanMemory UB overflow (compile-time), not a precision issue.
- **定位**: kernel-side `_select_config` (validated by Stage 3 L0-4/L0-5/L1) selects
  `block_m=2` (fp16) / `1` (bf16) for N_padded=4096, but the wrapper's inherited GPU-era
  `default_config` (48 KiB SMEM heuristic `SHARED_MEMORY_BUDGET_BYTES // (N_padded*elem)`)
  selected `block_m=4` and passed it into the factory as an override → `(block_m,N_padded)`
  fragments overflow UB. Same root cause also produced a hard
  `ValueError` ("A single row ... exceeds the 49152-byte budget") for lm-head `[4,102400]`
  (N_padded=102400), because `49152 // 204800 == 0`. This is exactly
  `CASE-argmax-argreduce-integration` defect ② (knowledge pre-injection).
- **修复**: wrapper glue only — `argmin.py` `default_config` now delegates to the kernel's
  tiling single source of truth `_select_config(self.M, self.N, self.dtype_str)["block_m"]`
  (added `from .argmin_kernel._argreduce_kernel import _select_config`). The kernel file and
  the tests/manifest/op were NOT touched. Backup:
  `argmin_kernel/history_version/argmin_s5_attempt1.py`.
- **结果**: re-run report → `status=passed`, correctness 42/42, benchmark 9/9 (6 valid msprof cases).
- **验证（要求）**: non-256-divisible N covered and passed — `non-aligned [256,300]` (N_padded=512),
  plus op-level `(128,300)`, `(129,512)`, `dim=None (10,30)`. Declared width = N_padded (VP-2026-0128
  contract) held: no "row 0 correct / rows>=1 wrong" symptom appeared.

## issues

- none blocking. Notes:
  - The wrapper still imports `SHARED_MEMORY_BUDGET_BYTES` from `_primitives` though the new
    `default_config` no longer uses it (left untouched after the passing run to keep the verified
    artifact byte-stable; harmless unused import).
  - Benchmark `Kernel` column reports `main` even for the tiled path (factory sets
    `msprof_kernel_name="argreduce_tiled"`, but the profiler captured `main`) — report-layer
    cosmetic, not an integration defect.
