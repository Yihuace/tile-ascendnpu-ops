# Integration Log -- ClampScalarFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/clamp_scalar/.migration_meta.json
- integrated: clamp_scalar_fwd_kernel -> clamp_scalar_kernel/clamp_scalar_fwd_kernel.py
- design_docs: clamp_scalar_fwd_kernel -> clamp_scalar_kernel/clamp_scalar_fwd_kernel_DESIGN.md
- attempts: 1 (Stage 5 wrapper glue adjustment; no correctness debug-loop retries)

## 集成摘要

- 集成脚本: `python3 .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/clamp_scalar/.migration_meta.json`（从 `examples/TileOPs/` 运行，幂等）。
- 聚合 `__init__.py`: re-export `clamp_scalar_fwd_kernel`（baseline 模块）。
- wrapper import: 已改写为 baseline/perf_opt 双 import 切换块；baseline（`from .clamp_scalar_kernel import clamp_scalar_fwd_kernel`）默认激活，perf_opt 行注释占位。
- ascend_mode: Developer（集成脚本从 Stage 3 kernel 的 `ASCEND_MODE` 读取；wrapper 原声明一致，未变更）。

## 验证结果

- report_run_id: 20261013_141507_226988_ClampScalarFwdOp
- report_run_json: examples/TileOPs/reports/tileops/20261013_141507_226988_ClampScalarFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261013_141507_226988_ClampScalarFwdOp/report.md
- report_html: examples/TileOPs/reports/tileops/20261013_141507_226988_ClampScalarFwdOp/report.html
- report_status: passed
- correctness: pass（11/11 用例；`summary.correctness_passed=true`，failed=0）
- benchmark: passed（14/14 用例；msprof_case_count=11，profiler_fallback_count=0，kernel_filter_fallback_count=0，invalid_profile_count=0）
- 性能数值（来自本次 run.json / benchmark.json，未另造数据；Ratio = roofline 利用率）：

| Label | Latency (us) | Ratio (%) | Shape | DType |
|---|---:|---:|---|---|
| 16M | 42.68 | 44.93 | [4096, 4096] | float16 |
| 16M | 43.00 | 50.87 | [4096, 4096] | bfloat16 |
| 16M | 55.98 | 68.50 | [4096, 4096] | float32 |
| 256M | 878.06 | 63.32 | [16384, 16384] | float16 |
| 256M | 874.26 | 63.60 | [16384, 16384] | bfloat16 |
| 16M-min-only | 39.84 | 48.13 | [4096, 4096] | float16 |
| 16M-min-only | 39.62 | 48.39 | [4096, 4096] | bfloat16 |
| 16M-min-only | 54.86 | 69.90 | [4096, 4096] | float32 |
| 16M-max-only | 39.38 | 48.69 | [4096, 4096] | float16 |
| 16M-max-only | 39.00 | 49.16 | [4096, 4096] | bfloat16 |
| 16M-max-only | 53.40 | 71.81 | [4096, 4096] | float32 |

- 正确性 per-shape max_abs_err：全部用例 `0.00e+00`（run.json `operators[0].max_abs_err=0.0`，`avg_max_abs_err=0.0`）。

## 调试历史（若有）

- glue-0（集成胶水修正，非正确性失败）：集成后核对 DESIGN §5.2 发现 wrapper `default_config` 的 `_DEFAULT_BLOCK_SIZE` 仍为 GPU `threads×npt` 折叠值（fp32=1024 / fp16=2048 / bf16=2048），而 DESIGN §5.2 明确 NPU `block_size=8192`（同族 hardsigmoid/mish 实测最优、32B 对齐、UB ~136–168KB < 192KB）。按集成规范（wrapper config 胶水层允许修改）将 `_DEFAULT_BLOCK_SIZE` 三 dtype 均改为 8192，并更新 docstring/注释引用 DESIGN §5.2。备份：`history_version/clamp_scalar_s5_glue0.py`。修正后 import 冒烟通过，report 全量正确性通过、benchmark 有效（config 中 `block_size: 8192` 已生效）。无正确性调试闭环重试。

## integrate_kernel.py run (2026-10-13T14:14:07+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: rewritten
- ascend_mode: Developer
- functions: clamp_scalar_fwd_kernel
- smoke: pass
