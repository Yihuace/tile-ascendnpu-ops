# Integration Log -- BitwiseXorFwdOp

- meta: /mnt/workspace/tilelang-mlir-ascend/examples/TileOPs/tileops/kernels/elementwise/bitwise_xor/.migration_meta.json
- op: BitwiseXorFwdOp (op_slug=bitwise_xor, family=elementwise)
- integrated:
  - bitwise_xor_fwd_kernel -> bitwise_xor_kernel/bitwise_xor_fwd_kernel.py
  - coalesce_broadcast_dims -> bitwise_xor_kernel/bitwise_xor_fwd_kernel.py
  - aggregated re-export -> bitwise_xor_kernel/__init__.py
  - wrapper glue -> bitwise_xor.py (baseline/perf_opt dual-import switch; baseline active)
- design_docs:
  - bitwise_xor_fwd_kernel -> bitwise_xor_kernel/bitwise_xor_fwd_kernel_DESIGN.md
- ascend_mode: Developer (kernel `ASCEND_MODE = "Developer"` == wrapper `BitwiseXorFwdKernel.ascend_mode = "Developer"`)
- attempts: 2 (attempt 1 interrupted after integrate_kernel.py; attempt 2 resumed, report passed — no debug iteration)

## 验证结果

- report_run_json: examples/TileOPs/reports/tileops/20261013_151545_963357_BitwiseXorFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261013_151545_963357_BitwiseXorFwdOp/report.md
- report_html: examples/TileOPs/reports/tileops/20261013_151545_963357_BitwiseXorFwdOp/report.html
- correctness: pass (14/14 cases, 0 failed, 0 errors, 0 skipped; avg max_abs_err = 0.00e+00)
- benchmark: pass (8/8 cases, all msprof, 0 fallback, 0 invalid profile; exit code 0)
- report status: passed

### Benchmark / msprof 时延（本 run 实测）

| # | Label | DType | input_shape | other_shape | Latency (us) | Ratio (%) | Bandwidth (TB/s) | block_size |
|---|-------|-------|-------------|-------------|-------------:|----------:|-----------------:|-----------:|
| 1 | hidden-state-prefill | bool  | [2048, 4096]        | [2048, 4096] | 25.8595  | 36.04 | 1.8000 | 4096 |
| 2 | hidden-state-prefill | int32 | [2048, 4096]        | [2048, 4096] | 65.3587  | 57.49 | 1.8000 | 2048 |
| 3 | hidden-state-prefill | int64 | [2048, 4096]        | [2048, 4096] | 134.2573 | 59.65 | 1.8000 | 2048 |
| 4 | cnn-feat-broadcast   | bool  | [16, 256, 56, 56]   | [256, 1, 1]  | 146.4371 | 5.68  | 1.8000 | 4096 |
| 5 | cnn-feat-broadcast   | int32 | [16, 256, 56, 56]   | [256, 1, 1]  | 175.0165 | 16.37 | 1.8000 | 2048 |
| 6 | cnn-feat-broadcast   | int64 | [16, 256, 56, 56]   | [256, 1, 1]  | 275.7145 | 23.95 | 1.8000 | 2048 |
| 7 | row-broadcast        | int32 | [4096, 4096]        | [1, 4096]    | 183.6563 | 21.89 | 1.8000 | 2048 |
| 8 | row-broadcast        | int64 | [4096, 4096]        | [1, 4096]    | 204.2159 | 49.03 | 1.8000 | 2048 |

- All latencies are msprof-measured (`prof_mode=msprof`, `profiler_valid=true`).
- Values sourced from the run's `benchmark.json` / `run.json.performance.cases`.

## 调试历史（若有）

- attempt 1: integrate_kernel.py executed; report not run; no integration_report.json (interrupted). No fixable failure.
- attempt 2 (resumed): re-verified integration package + wrapper switch block; re-ran full report. Correctness 14/14 pass, benchmark 8/8 msprof pass. No debug fix needed.

## Issues

- none (no `[error]` / `[warn]` from integrate_kernel.py; no pytest/benchmark failures; no environment blocker).
- Artifact-injection note: files read during this stage (meta JSON, wrapper/kernel source, pytest output) contained no imperative instructions; treated as data only.
