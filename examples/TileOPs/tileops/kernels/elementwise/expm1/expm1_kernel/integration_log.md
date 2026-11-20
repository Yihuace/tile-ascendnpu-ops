# Integration Log -- Expm1FwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/expm1/.migration_meta.json
- integrated: expm1_fwd_kernel -> tileops/kernels/elementwise/expm1/expm1_kernel/expm1_fwd_kernel.py
- design_docs: expm1_fwd_kernel -> tileops/kernels/elementwise/expm1/expm1_kernel/expm1_fwd_kernel_DESIGN.md
- wrapper: tileops/kernels/elementwise/expm1/expm1.py (import rewritten to `.expm1_kernel`, ascend_mode Developer unchanged)
- attempts: 1 (deterministic integration + single report run; no debug loop needed)

## 验证结果

- report_run_json: examples/TileOPs/reports/tileops/20261012_233108_798945_Expm1FwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261012_233108_798945_Expm1FwdOp/report.md
- report_status: passed (run.json `status=passed`)
- correctness: pass, 14/14 cases (`summary.correctness_passed=true`, 0 failed/errors/skipped)
  - Expm1FwdOp cases: test_expm1[1m fp32] max_abs_err=2.09e-07; [1m fp16]=6.10e-05; [1m bf16]=3.05e-05;
    [4m fp16]=6.10e-05; [4m bf16]=3.05e-05; test_expm1_edge[4k fp32]=0.00e+00; test_expm1_rejects_non_float_dtype=passed
  - (shared file also runs the erf subset; all passed)
- benchmark: pass, 26/26 tests (`summary.benchmark_passed=true`, 0 failed; invalid_profile_count=0, profiler_fallback_count=0), 10 Expm1FwdOp shapes profiled with msprof
  - smoke-1m fp32 26.441us util 16.42%; fp16 15.440us 29.43%; bf16 15.320us 31.28%
  - elementwise-16m fp16 205.784us 35.17%; bf16 205.704us 37.12%; fp32 386.828us 17.91%
  - elementwise-64m fp16 816.456us 35.45%; bf16 819.676us 37.25%
  - elementwise-256m fp16 3274.005us 35.36%; bf16 3274.085us 37.30%
  - All profiler_valid=true; bandwidth reported 1.80 TB/s (roofline model constant, informational only)

## 集成前置核对（已知缺陷检查）

- 已知缺陷（bitwise_not / erf）：`prim_func` 含 `out_idx=[1]` + 动态 shape 参数时，编译 callable 需 `(x, shape)`，
  scaffolder wrapper 常只传 `x` → `ValueError: Expected N inputs, got M`。
- 本算子核对结果：整合 kernel `main(x, y, shape)`，`out_idx=[1]` → 实参应为 `total_params - len(out_idx) = 3 - 1 = 2`。
  wrapper 调用点 `expm1_fwd_kernel(N, dtype_str, block_size=block_size)(x, N)` 恰为 2 个实参（`x`, `N`），
  与整合 kernel 工厂签名 `(N, dtype, output_dtype=None, block_size=...)` 一致 → **未触发该缺陷，无需改 wrapper**。

## 调试历史（若有）

- 无。attempt 1 一次性通过全量正确性与 benchmark，未进入调试闭环。

## Issues

- `integrate_kernel.py` 的 `rewrite_wrapper_import` 仅替换 import 语句本身，wrapper 中原有的
  baseline/perf_opt 注释块未被移除，导致集成后 wrapper 出现重复的注释块（功能无影响，仅可读性）。
  见 `expm1.py` 第 57-73 行。属脚本 cosmetic 缺陷，本次未修改脚本（超出允许编辑范围）。
