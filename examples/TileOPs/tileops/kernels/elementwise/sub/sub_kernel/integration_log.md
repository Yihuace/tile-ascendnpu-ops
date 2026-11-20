# Integration Log -- SubFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/sub/.migration_meta.json
- integrated: sub_fwd_kernel -> sub_kernel/sub_fwd_kernel.py
- design_docs: sub_fwd_kernel -> sub_kernel/sub_fwd_kernel_DESIGN.md
- attempts: 1 (no correctness debug-loop retries; no manual wrapper glue fix needed)

## 集成摘要

- 集成脚本: `python3 .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/sub/.migration_meta.json`（从 `examples/TileOPs/` 运行，幂等；二次运行输出 `already integrated`）。
- 聚合 `__init__.py`: re-export `sub_fwd_kernel`（baseline 模块）。
- wrapper import: 已改写为 baseline/perf_opt 双 import 切换块；baseline（`from .sub_kernel import sub_fwd_kernel`）默认激活，perf_opt 行注释占位。
- ascend_mode: Developer（集成脚本从 Stage 3 kernel 的 `ASCEND_MODE = "Developer"` 读取；wrapper 原声明一致，未变更）。
- wrapper 胶水核对（无需人工修复）：Stage-3 factory 签名为 `sub_fwd_kernel(N, dtype, alpha=1, output_dtype=None, block_size=8192, num_cores=None) -> main(a, b, y, shape)`，`out_idx=[2]`（y 自动分配，经 `out_idx` 定位）。wrapper `__init__` 调用 `sub_fwd_kernel(self.N_total, self.dtype_str, alpha=self.alpha)`，`forward` 调用 `sub_fwd_kernel(self.N_total, self.dtype_str, alpha=self.alpha, block_size=self.config["block_size"])(a, b, self.N_total)`，`default_config` 返回 `{"block_size": 8192}`（DESIGN §5.2）——**与 Stage-3 factory 完全匹配，无 threads/npt 残留，无需人工胶水修复**。`sub.py` 由 scaffold 模板生成时不含预置 baseline/perf_opt 注释块，故集成脚本改写后无重复块（与 add.py 的 cosmetic 残留不同）。

## 验证结果

- report_run_id: 20261013_170400_275674_SubFwdOp
- report_run_json: examples/TileOPs/reports/tileops/20261013_170400_275674_SubFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261013_170400_275674_SubFwdOp/report.md
- report_html: examples/TileOPs/reports/tileops/20261013_170400_275674_SubFwdOp/report.html
- report_status: passed
- correctness: pass（24/24 用例；`summary.correctness_passed=true`，failed=0，errors=0，skipped=0）
- benchmark: passed（15/15 用例；msprof_case_count=15，profiler_fallback_count=0，kernel_filter_fallback_count=0，invalid_profile_count=0）
- 正确性 max_abs_err：17 个数值用例全部 `0.00e+00`（`properties.max_abs_err=0.00e+00`）；其余 7 个为非数值用例（`test_sub_rejects_non_float_dtype` ×3、`test_sub_kernel_rejects_non_float_dtype`、`test_sub_rejects_dtype_mismatch`、`test_sub_broadcast_output_shape`、`test_sub_eval_roofline`），无误差指标。
- 性能数值（来自本次 run.json / benchmark.json，未另造数据；Ratio = roofline 利用率）：

| Label | Latency (us) | Ratio (%) | Shape (input/other) | DType |
|---|---:|---:|---|---|
| hidden-state-prefill | 47.44 | 80.83 | [2048,4096]/[2048,4096] | float32 |
| hidden-state-prefill | 25.74 | 74.49 | [2048,4096]/[2048,4096] | bfloat16 |
| cnn-feat-broadcast | 71.00 | 82.70 | [16,256,56,56]/[256,1,1] | float32 |
| cnn-feat-broadcast | 39.00 | 75.28 | [16,256,56,56]/[256,1,1] | float16 |
| cnn-feat-broadcast | 37.76 | 77.75 | [16,256,56,56]/[256,1,1] | bfloat16 |
| llama-7b-ffn | 64.20 | 80.27 | [1024,11008]/[1024,11008] | float32 |
| llama-7b-ffn | 33.10 | 77.84 | [1024,11008]/[1024,11008] | float16 |
| llama-7b-ffn | 32.52 | 79.23 | [1024,11008]/[1024,11008] | bfloat16 |
| linear-bias | 28.02 | 68.43 | [1024,4096]/[1,4096] | float32 |
| linear-bias | 13.72 | 69.88 | [1024,4096]/[1,4096] | float16 |
| linear-bias | 13.82 | 69.37 | [1024,4096]/[1,4096] | bfloat16 |
| llama-7b-hidden | 13.70 | 69.98 | [1024,4096]/[1024,4096] | float16 |
| phi-2-ffn | 31.16 | 76.92 | [1024,10240]/[1024,10240] | float16 |
| phi-2-ffn-bias | 31.60 | 75.85 | [1024,10240]/[1,10240] | float16 |
| llama-7b-ffn-bias | 33.12 | 77.79 | [1024,11008]/[1,11008] | float16 |

- 说明：benchmark 表中带宽/核数取自 `benchmark/msprof_roofline.py`；设备为 Ascend910B3（未登记于 `SOC_CONFIG`，走 `DEFAULT_CONFIG` 兜底 num_cores=48 / gm_bw=1.8），Ratio/BW 属兜底口径。

## 调试历史（若有）

- 无正确性调试闭环。集成脚本一次通过（smoke=pass），单算子 report 一次通过（status=passed）。

## issues

- none

## integrate_kernel.py run (2026-10-13T17:03:28+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: rewritten
- ascend_mode: Developer
- functions: sub_fwd_kernel
- smoke: pass

## integrate_kernel.py run (2026-10-13T17:27:31+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: already integrated
- ascend_mode: Developer
- functions: sub_fwd_kernel
- smoke: pass
