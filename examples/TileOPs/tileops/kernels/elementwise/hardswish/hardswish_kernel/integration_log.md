# Integration Log -- HardswishFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/hardswish/.migration_meta.json
- integrated: hardswish_fwd_kernel -> hardswish_kernel/hardswish_fwd_kernel.py
- design_docs: hardswish_fwd_kernel -> hardswish_kernel/hardswish_fwd_kernel_DESIGN.md
- attempts: 1 (Stage 5 wrapper glue adjustment; no correctness debug-loop retries)

## 集成摘要

- 集成脚本: `python3 .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/hardswish/.migration_meta.json`（从 `examples/TileOPs/` 运行，幂等；末行 `[json]` 摘要显示 import 已改写、`ascend_mode=Developer`、smoke 通过）。
- 聚合 `__init__.py`: re-export `hardswish_fwd_kernel`（baseline 模块）。
- wrapper import: 已改写为 baseline/perf_opt 双 import 切换块；baseline（`from .hardswish_kernel import hardswish_fwd_kernel`）默认激活，perf_opt 行（`.hardswish_kernel.perf_opt.hardswish_fwd_kernel`）注释占位。
- ascend_mode: Developer（集成脚本从 Stage 3 kernel 的 `ASCEND_MODE` 读取；wrapper 原声明一致，未变更）。

## 验证结果

- report_run_id: 20261013_150641_447235_HardswishFwdOp
- report_run_json: examples/TileOPs/reports/tileops/20261013_150641_447235_HardswishFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261013_150641_447235_HardswishFwdOp/report.md
- report_html: examples/TileOPs/reports/tileops/20261013_150641_447235_HardswishFwdOp/report.html
- report_status: passed
- correctness: pass（7/7 用例；`summary.correctness_passed=true`，failed=0）
- benchmark: passed（9/9 用例；`summary.benchmark_passed=true`，failed=0；msprof_case_count=6，profiler_fallback_count=0，kernel_filter_fallback_count=0，invalid_profile_count=0）
- 性能数值（来自本次 run.json / benchmark.json，未另造数据；Ratio = roofline 利用率）：

| Label | Latency (us) | Ratio (%) | Shape | DType |
|---|---:|---:|---|---|
| mbv3-stage2 | 22.94 | 54.97 | [32, 96, 56, 56] | float16 |
| mbv3-stage2 | 24.34 | 60.40 | [32, 96, 56, 56] | bfloat16 |
| mbv3-stage3 | 15.80 | 49.89 | [32, 240, 28, 28] | float16 |
| mbv3-stage3 | 16.20 | 56.72 | [32, 240, 28, 28] | bfloat16 |
| fc-wide | 20.12 | 54.58 | [2048, 4096] | float16 |
| fc-wide | 21.12 | 60.61 | [2048, 4096] | bfloat16 |

- 正确性 per-shape max_abs_err（run.json `operators[0].max_abs_err=4.77e-07`，`avg_max_abs_err=9.54e-08`）：fp32 @1048576 = 4.77e-07；fp16/bf16 及其余用例 = 0.00e+00。

## 调试历史（若有）

- glue-0（集成胶水修正，非正确性失败）：集成后核对 DESIGN §5.2 发现 wrapper `default_config` 的 `_DEFAULT_BLOCK_SIZE` 仍为 GPU `threads×npt` 折叠值（256×8=2048），而 DESIGN §5.2 明确 NPU `block_size=8192`（同族 mish / hardsigmoid Stage 4 实测最优、32B 对齐、UB 预算 128–160KB < 192KB）。按集成规范（wrapper config 胶水层允许修改）将 `_DEFAULT_BLOCK_SIZE` 由标量 2048 改为按 dtype 的 8192（三 dtype 均 8192，套用 sibling `clamp_scalar` 的 dict 写法），并更新 `default_config` 取 `_DEFAULT_BLOCK_SIZE[self.dtype_str]`、docstring 示例与注释引用 DESIGN §5.2。备份：`history_version/hardswish_s5_glue0.py`。修正后 import 冒烟通过，report 全量正确性通过、benchmark 有效（run.json 各 case `config.block_size=8192` 已生效）。无正确性调试闭环重试。

## integrate_kernel.py run (2026-10-13T15:06:01+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: rewritten
- ascend_mode: Developer
- functions: hardswish_fwd_kernel
- smoke: pass

## integrate_kernel.py run (2026-10-13T15:22:15+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: already integrated
- ascend_mode: Developer
- functions: hardswish_fwd_kernel
- smoke: pass
