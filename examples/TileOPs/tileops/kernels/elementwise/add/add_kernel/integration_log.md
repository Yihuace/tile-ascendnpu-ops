# Integration Log -- AddFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/add/.migration_meta.json
- integrated: add_fwd_kernel -> add_kernel/add_fwd_kernel.py
- design_docs: add_fwd_kernel -> add_kernel/add_fwd_kernel_DESIGN.md
- attempts: 1 (no correctness debug-loop retries; no manual wrapper glue fix needed)

## 集成摘要

- 集成脚本: `python3 .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/add/.migration_meta.json`（从 `examples/TileOPs/` 运行，幂等）。
- 聚合 `__init__.py`: re-export `add_fwd_kernel`（baseline 模块）。
- wrapper import: 已改写为 baseline/perf_opt 双 import 切换块；baseline（`from .add_kernel import add_fwd_kernel`）默认激活，perf_opt 行注释占位。
- ascend_mode: Developer（集成脚本从 Stage 3 kernel 的 `ASCEND_MODE` 读取；wrapper 原声明一致，未变更）。
- wrapper 胶水核对（无需人工修复）：Stage-3 factory 签名为 `add_fwd_kernel(N, dtype, alpha=..., block_size=8192, num_cores=None) -> main(a, b, N)`，`out_idx=[2]`。wrapper `__init__` 调用 `add_fwd_kernel(self.N_total, self.dtype_str, alpha=self.alpha)`，`forward` 调用 `add_fwd_kernel(self.N_total, self.dtype_str, alpha=self.alpha, block_size=self.config["block_size"])(a, b, self.N_total)`，`default_config` 返回 `{"block_size": 8192}`（DESIGN §5.2）——**与 Stage-3 factory 完全匹配，无 threads/npt 残留，无需人工胶水修复**。

## 验证结果

- report_run_id: 20261013_162043_457669_AddFwdOp
- report_run_json: examples/TileOPs/reports/tileops/20261013_162043_457669_AddFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261013_162043_457669_AddFwdOp/report.md
- report_html: examples/TileOPs/reports/tileops/20261013_162043_457669_AddFwdOp/report.html
- report_status: passed
- correctness: pass（24/24 用例；`summary.correctness_passed=true`，failed=0）
- benchmark: passed（11/11 用例；msprof_case_count=11，profiler_fallback_count=0，kernel_filter_fallback_count=0，invalid_profile_count=0）
- 性能数值（来自本次 run.json / benchmark.json，未另造数据；Ratio = roofline 利用率）：

| Label | Latency (us) | Ratio (%) | Shape (input/other) | DType |
|---|---:|---:|---|---|
| hidden-state-prefill | 48.40 | 79.23 | [2048,4096]/[2048,4096] | float32 |
| hidden-state-prefill | 24.98 | 76.76 | [2048,4096]/[2048,4096] | bfloat16 |
| cnn-feat-broadcast | 74.86 | 78.44 | [16,256,56,56]/[256,1,1] | float32 |
| cnn-feat-broadcast | 37.54 | 78.21 | [16,256,56,56]/[256,1,1] | float16 |
| cnn-feat-broadcast | 38.92 | 75.44 | [16,256,56,56]/[256,1,1] | bfloat16 |
| llama-7b-ffn | 62.72 | 82.16 | [1024,11008]/[1024,11008] | float32 |
| llama-7b-ffn | 34.10 | 75.56 | [1024,11008]/[1024,11008] | float16 |
| llama-7b-ffn | 33.58 | 76.73 | [1024,11008]/[1024,11008] | bfloat16 |
| linear-bias | 24.40 | 78.58 | [1024,4096]/[1,4096] | float32 |
| linear-bias | 13.02 | 73.63 | [1024,4096]/[1,4096] | float16 |
| linear-bias | 14.50 | 66.12 | [1024,4096]/[1,4096] | bfloat16 |

- 正确性 max_abs_err：全部用例 `0.00e+00`（run.json `summary.avg_max_abs_err=0.0`，per-case `max_abs_err=0.00e+00`）。
- 说明：benchmark 表中带宽/核数取自 `benchmark/msprof_roofline.py`；设备为 Ascend910B3（未登记于 `SOC_CONFIG`，走 `DEFAULT_CONFIG` 兜底 num_cores=48 / gm_bw=1.8），Ratio/BW 属兜底口径。

## 调试历史（若有）

- 无正确性调试闭环。集成脚本一次通过（smoke=pass），单算子 report 一次通过（status=passed）。

## issues

- cosmetic（非阻断）：`add.py` 由 Stage-0 scaffold 模板自带的 baseline/perf_opt 注释块（baseline 行原指向 `._add_fwd_kernels`）未被集成脚本识别为“已集成”，脚本只替换了 baseline import 行，导致生成块与模板残留注释块叠加，wrapper 顶部出现重复的 selection 注释块。功能无影响（active import 唯一且正确），未手工修改以保持集成脚本输出保真；建议 scaffold 模板不预置该块或集成脚本按整块替换。

## integrate_kernel.py run (2026-10-13T16:19:42+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: rewritten
- ascend_mode: Developer
- functions: add_fwd_kernel
- smoke: pass
