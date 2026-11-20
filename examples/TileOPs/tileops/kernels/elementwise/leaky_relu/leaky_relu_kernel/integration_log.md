# Integration Log -- LeakyReluFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/leaky_relu/.migration_meta.json
- integrated: _make_leaky_relu_kernel -> tileops/kernels/elementwise/leaky_relu/leaky_relu_kernel/_make_leaky_relu_kernel.py
- design_docs: _make_leaky_relu_kernel -> tileops/kernels/elementwise/leaky_relu/leaky_relu_kernel/_make_leaky_relu_kernel_DESIGN.md
- wrapper: tileops/kernels/elementwise/leaky_relu/leaky_relu.py (import rewritten from `._leaky_relu_fwd_kernels` to `.leaky_relu_kernel`, ascend_mode Developer unchanged)
- attempts: 1 (deterministic integration + single report run; no debug loop needed)

## 集成前置核对（已知缺陷检查）

- 已知缺陷（bitwise_not / erf 等）：`prim_func` 含 `out_idx=[1]` + 动态 shape 参数时，编译 callable 需 `(x, shape)`，
  scaffolder wrapper 常只传 `x` → `ValueError: Expected N inputs, got M`。
- 本算子核对结果：整合 kernel `main(x, y, shape: T.int32)`，`out_idx=[1]` → 实参应为 `total_params - len(out_idx) = 3 - 1 = 2`。
  wrapper 调用点 `_make_leaky_relu_kernel(N, dtype_str, negative_slope, block_size=block_size)(x, N)` 恰为 2 个实参
  (`x`, `N`)，与整合 kernel 工厂签名 `(N, dtype, negative_slope, output_dtype=None, block_size=...)` 一致
  → **未触发该缺陷，未修改 wrapper 调用点**。
- 编程模式核对：整合产物 `ASCEND_MODE = "Developer"`（字面量声明 + `os.environ.setdefault("TILELANG_ASCEND_MODE","Developer")`），
  单一模式、无混用；集成脚本同步到 `LeakyReluFwdKernel.ascend_mode` 时报告 `unchanged`（wrapper 原已为 Developer）。

## 验证结果

- report_run_json: examples/TileOPs/reports/tileops/20261013_075024_365167_LeakyReluFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261013_075024_365167_LeakyReluFwdOp/report.md
- report_status: passed (run.json `status=passed`; `summary.correctness_passed=true`, `benchmark_passed=true`)
- correctness: pass, 16/16 cases in the shared correctness file (`correctness_failed=0`, 0 errors/skipped)
  - LeakyReluFwdOp cases (10, all passed, `max_abs_err=0.00e+00`): test_leaky_relu[1m fp32/fp16/bf16],
    test_leaky_relu[4m fp16/bf16], test_leaky_relu_edge[4k], test_leaky_relu_rejects_non_float_dtype,
    test_leaky_relu_eval_roofline — bit-exact against `F.leaky_relu(x.float(), 0.01).to(x.dtype)`.
  - (the same pytest file also runs the EluFwdOp subset; all passed and are outside this op's scope)
- benchmark: pass, 20/20 tests (`summary.benchmark_passed=true`, 0 failed; `invalid_profile_count=0`,
  `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`), 10 LeakyReluFwdOp shapes profiled with msprof:
  - smoke-1m fp32 17.000 us (util 13.71%); fp16 7.300 us (15.96%); bf16 7.460 us (15.62%)
  - gan-feat [16,256,64,64] fp16 77.062 us (24.19%); bf16 76.362 us (24.41%)
  - gan-feat-deep [16,512,32,32] fp16 40.001 us (23.30%); bf16 40.261 us (23.15%)
  - elementwise-16m [4096,4096] fp16 77.082 us (24.18%); bf16 76.422 us (24.39%); fp32 223.724 us (17.71%)
  - All `profiler_valid=true`; bandwidth 1.80 TB/s is the roofline-model constant (informational only).

## 调试历史（若有）

- 无。attempt 1 一次性通过全量正确性与 benchmark，未进入调试闭环。

## Issues

- `integrate_kernel.py` 的 `rewrite_wrapper_import` 仅替换 import 语句本身，wrapper 中原有的
  baseline/perf_opt 注释块未被移除，导致集成后 wrapper 出现重复的注释块（功能无影响，仅可读性）。
  与既有同类算子（`expm1.py` 第 57-73 行）表现一致。属脚本 cosmetic 缺陷，本次未修改脚本（超出允许编辑范围）。
- `run.json` `warnings` 含 20 条 `missing tileops candidate record`（Elu/Leaky 各 10），为 benchmark
  候选记录缺失告警，非正确性/性能失败；`benchmark_passed=true`，仅记录不修复。
