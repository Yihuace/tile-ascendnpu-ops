# Integration Log -- IsinfFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/isinf/.migration_meta.json
- integrated: isinf_fwd_kernel -> isinf_kernel/isinf_fwd_kernel.py
- design_docs: isinf_fwd_kernel -> isinf_kernel/isinf_fwd_kernel_DESIGN.md
- attempts: 1（集成脚本一次通过 + 全量 report 一次通过；无需调试闭环，无需胶水修复）

## 集成摘要

- 集成脚本: `python3 .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/isinf/.migration_meta.json`（从 `examples/TileOPs/` 运行）。末行 `[json]` 摘要：import 改写成功、`ascend_mode=Developer`、smoke=pass、`wrapper_rewritten=true`。
- 聚合 `__init__.py`: re-export `isinf_fwd_kernel`（baseline 模块 `isinf_fwd_kernel.py`）。
- wrapper import: 已改写为 baseline/perf_opt 双 import 切换块；baseline（`from .isinf_kernel import isinf_fwd_kernel`）默认激活，perf_opt 行注释占位。
- ascend_mode: Developer（集成脚本从 Stage 3 kernel 的 `ASCEND_MODE = "Developer"` 读取；wrapper 原声明一致，未变更）。
- 设计文档快照: `isinf_fwd_kernel_DESIGN.md`（源自 `examples/isinf/isinf_fwd_kernel/DESIGN.md`，与集成 kernel 同目录，未修改）。
- import 冒烟: `from tileops.kernels.elementwise.isinf import IsinfFwdKernel` 通过。
- **bool 输出胶水（无需修复）**: wrapper `forward` 已是 `y = isinf_fwd_kernel(self.N_total, self.dtype_str)(x, self.N_total); return y.view(torch.bool)`，与参考算子 `isnan`/`bitwise_not` 一致。kernel 以 int8 字节落盘（`_BOOL_STORAGE_DTYPE = "int8"`），host `y.view(torch.bool)` 为纯元数据重解释，bit-exact；Op `output_dtype = torch.bool`。本次未修改 wrapper/kernel/任何交付件。

## 验证结果

- report_run_id: 20261014_101823_741698_IsinfFwdOp
- report_run_json: examples/TileOPs/reports/tileops/20261014_101823_741698_IsinfFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261014_101823_741698_IsinfFwdOp/report.md
- report_html: examples/TileOPs/reports/tileops/20261014_101823_741698_IsinfFwdOp/report.html
- report_status: **passed**
- correctness: **pass**（`summary.correctness_passed=true`，9/9 通过，0 失败；5 个 `test_isinf_op[...]` dtype/shape 用例 + `test_isinf_output_dtype_and_values` + `test_isinf_rejects_non_float_dtype` + `test_isinf_op_rejects_non_float_dtype` + `test_isinf_eval_roofline`）
- benchmark: **pass**（`summary.benchmark_passed=true`，10/10 通过，0 失败；带 profile 的 msprof case_count=5）
- msprof_case_count / profiler_fallback_count / kernel_filter_fallback_count / invalid_profile_count: **5 / 0 / 0 / 0**
- max_abs_err: 0.0（`operators[0].max_abs_err=0.0`，全部正确性用例精确相等 atol=rtol=0；bool 谓词为精确布尔逻辑）

### benchmark / msprof 用例构成（10 collected vs 5 profiled 的逐例说明）

- benchmark 阶段收集并执行 **10** 个 manifest workload，全部通过（`benchmark_tests=10`）：
  `smoke-1m`(float32/float16/bfloat16) + `16m`(float16/bfloat16/float32) + `256m`(float16/bfloat16) + `16m-half`(float16/bfloat16)。
- 性能分析（`tileops/reporting/analyzer.py::_analyze_performance`）**排除 label 含 smoke 的 3 个 `smoke-1m` 记录**（`_is_smoke_record`），并对相同 `(operator, case_id)` 去重：`16m-half` 的 (4096,4096) float16/bfloat16 与 `16m` 同 shape+dtype → 归并。
- 故 `performance.cases=5`（3×`16m` 三种 dtype + 2×`256m` 两种 dtype），即 `msprof_case_count=5`；`profiler_fallback_count=0`（无 msprof→events 回退）、`kernel_filter_fallback_count=0`（`--kernel-name main` 全部命中）、`invalid_profile_count=0`。

### per-shape perf（来自 run.json performance.cases，msprof，profiler_valid=true）

| label | shape | dtype | latency_us | roofline_util% | bandwidth_tb_s | kernel |
|-------|-------|-------|-----------|----------------|----------------|--------|
| 16m | (4096,4096) | float16 | 37.90 | 50.59 | 1.75 | main |
| 16m | (4096,4096) | bfloat16 | 58.36 | 41.21 | 1.75 | main |
| 16m | (4096,4096) | float32 | 50.74 | 75.58 | 1.75 | main |
| 256m | (16384,16384) | float16 | 685.62 | 61.54 | 1.75 | main |
| 256m | (16384,16384) | bfloat16 | 902.24 | 46.75 | 1.75 | main |

（未进性能表的用例：`smoke-1m` 3 例被 smoke 过滤排除；`16m-half` 2 例与 `16m` 同 shape+dtype 去重。benchmark.md Latency 与非去重记录一致。）

## 调试历史（无）

本次无正确性/编译失败，未进入调试闭环，未修改任何交付件（kernel / wrapper / DESIGN 快照均未改）。

## Stage 5 集成三护栏（R4，任务书 §2.1）

- 护栏 1（wrapper_rewritten）: `true` ✅（见文末 `integrate_kernel.py run` 块）
- 护栏 2（幂等重跑关键字）: **未命中**（本次为首次集成，`grep` 幂等关键字退出码非零、无命中）✅
- 护栏 3（生效 kernel `__module__` 指向 baseline）: `tileops.kernels.elementwise.isinf.isinf_kernel.isinf_fwd_kernel` ✅（指向 baseline 模块 `isinf_kernel`，非 perf_opt 旧谱系，也非提取参考 `_isinf_fwd_kernels`）
  - 说明：生效函数经 wrapper 模块取用（`importlib.import_module('tileops.kernels.elementwise.isinf.isinf').isinf_fwd_kernel`）；算子包 `tileops.kernels.elementwise.isinf` 的 `__init__.py` 仅 re-export `IsinfFwdKernel`，与参考算子 `isnan` 同形，故 §2.1 字面命令 `w.isinf_fwd_kernel` 在包级不可直接取到，判读按 §2.1「必须与护栏 3 联合判定」取生效 kernel 的真实来源模块。

## integrate_kernel.py run (2026-10-14T10:17:54+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: rewritten
- ascend_mode: Developer
- functions: isinf_fwd_kernel
- smoke: pass
