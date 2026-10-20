# Integration Log -- IsfiniteFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/isfinite/.migration_meta.json
- integrated: isfinite_fwd_kernel -> isfinite_kernel/isfinite_fwd_kernel.py
- design_docs: isfinite_fwd_kernel -> isfinite_kernel/isfinite_fwd_kernel_DESIGN.md
- attempts: 1（集成脚本一次通过 + 全量 report 一次通过；无需调试闭环，无需胶水修复）

## 集成摘要

- 集成脚本: `python3 .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/isfinite/.migration_meta.json`（从 `examples/TileOPs/` 运行）。末行 `[json]` 摘要：import 改写成功、`ascend_mode=Developer`、smoke=pass、`wrapper_rewritten=true`、`wrapper_mode_updated=false`（wrapper 原声明 `Developer` 与产物一致，未变更）。
- 聚合 `__init__.py`: re-export `isfinite_fwd_kernel`（baseline 模块 `isfinite_fwd_kernel.py`）。
- wrapper import: 已改写为 baseline/perf_opt 双 import 切换块；baseline（`from .isfinite_kernel import isfinite_fwd_kernel`）默认激活，perf_opt 行注释占位。
- ascend_mode: Developer（集成脚本从 Stage 3 kernel 的 `ASCEND_MODE = "Developer"` 读取；与 wrapper 原声明一致，未变更）。
- 设计文档快照: `isfinite_fwd_kernel_DESIGN.md`（源自 `examples/TileOPs/tileops/kernels/elementwise/isfinite/isfinite_kernel/isfinite_fwd_kernel_DESIGN.md`，与集成 kernel 同目录，未修改）。
- import 冒烟: `from tileops.kernels.elementwise.isfinite import IsfiniteFwdKernel` 通过；`isfinite_fwd_kernel.__module__` = `tileops.kernels.elementwise.isfinite.isfinite_kernel.isfinite_fwd_kernel`（Baseline 已生效）。
- **bool 输出胶水（无需修复）**: wrapper `forward` 已是 `y = isfinite_fwd_kernel(self.N_total, self.dtype_str)(x, self.N_total); return y.view(torch.bool)`，与参考算子 `isnan`/`isinf`/`bitwise_not` 一致。kernel 以 int8 字节落盘（`_BOOL_STORAGE_DTYPE = "int8"`），host `y.view(torch.bool)` 为纯元数据重解释，bit-exact；Op `output_dtype = torch.bool`。本次未修改 wrapper / kernel / 任何交付件。

## 验证结果

- report_run_id: 20261014_105757_589000_IsfiniteFwdOp
- report_run_json: examples/TileOPs/reports/tileops/20261014_105757_589000_IsfiniteFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261014_105757_589000_IsfiniteFwdOp/report.md
- report_html: examples/TileOPs/reports/tileops/20261014_105757_589000_IsfiniteFwdOp/report.html
- report_status: **passed**
- correctness: **pass**（`summary.correctness_passed=true`，9/9 通过，0 失败；3 个 `test_isfinite_op[1048576-dtype0..2]` + 2 个 `test_isfinite_op[4000000-dtype3..4]` + `test_isfinite_output_dtype_and_values` + `test_isfinite_rejects_non_float_dtype` + `test_isfinite_op_rejects_non_float_dtype` + `test_isfinite_eval_roofline`）
- benchmark: **pass**（`summary.benchmark_passed=true`，10/10 通过，0 失败；带 profile 的 msprof case_count=5）
- msprof_case_count / profiler_fallback_count / kernel_filter_fallback_count / invalid_profile_count: **5 / 0 / 0 / 0**
- max_abs_err: 0.0（`operators[0].max_abs_err=0.0`，全部正确性用例精确相等 atol=rtol=0；bool 谓词为精确布尔逻辑，位精确属构造必然）

### benchmark / msprof 用例构成（10 collected vs 5 profiled 的逐例说明）

- benchmark 阶段收集并执行 **10** 个 manifest workload，全部通过（`benchmark_tests=10`）：
  `smoke-1m`(float32/float16/bfloat16) + `16m`(float16/bfloat16/float32) + `256m`(float16/bfloat16) + `16m-half`(float16/bfloat16)。
- 性能分析（`tileops/reporting/analyzer.py::_analyze_performance`）**排除 label 含 smoke 的 3 个 `smoke-1m` 记录**（`_is_smoke_record`），并对相同 `(operator, case_id)` 去重：`16m-half` 的 (4096,4096) float16/bfloat16 与 `16m` 同 shape+dtype → 归并。
- 故 `performance.cases=5`（3×`16m` 三种 dtype + 2×`256m` 两种 dtype），即 `msprof_case_count=5`；`profiler_fallback_count=0`（无 msprof→events 回退）、`kernel_filter_fallback_count=0`（`--kernel-name main` 全部命中，`kernel_name_resolved=main`）、`invalid_profile_count=0`。

### per-shape perf（来自 run.json performance.cases，msprof，profiler_valid=true）

| label | shape | dtype | latency_us | roofline_util% | bandwidth_tb_s | kernel |
|-------|-------|-------|-----------|----------------|----------------|--------|
| 16m | (4096,4096) | float16 | 37.82 | 50.70 | 1.75 | main |
| 16m | (4096,4096) | bfloat16 | 58.98 | 40.78 | 1.75 | main |
| 16m | (4096,4096) | float32 | 53.22 | 72.06 | 1.75 | main |
| 256m | (16384,16384) | float16 | 683.48 | 61.73 | 1.75 | main |
| 256m | (16384,16384) | bfloat16 | 916.78 | 46.00 | 1.75 | main |

（未进性能表的用例：`smoke-1m` 3 例被 smoke 过滤排除；`16m-half` 2 例与 `16m` 同 shape+dtype 去重。benchmark.md Latency 与非去重记录一致。）

### 同类横向对照（已迁移 elementwise 谓词算子，来自各自 run.json 16m float32）

| 算子 | 16m (4096,4096) float32 latency_us | roofline_util% | bandwidth_tb_s |
|------|-----------------------------------|----------------|----------------|
| isfinite | 53.22 | 72.06 | 1.75 |
| isinf | 50.74 | 75.58 | 1.75 |
| isnan | 51.54 | 74.40 | 1.75 |

三者同为 FloatPredicateKernel 系谓词、同带宽上限（1.75 TB/s），isfinite 落于 +-inf 双侧比较 + bool 掩码合并路径，延迟与 isinf/isnan 同量级（差异在噪声范围），无反常信号。

## 调试历史（无）

本次无正确性 / 编译失败，未进入调试闭环，未修改任何交付件（kernel / wrapper / DESIGN 快照均未改）。

## Stage 5 集成三护栏（R4，任务书 §2.1）

- 护栏 1（wrapper_rewritten）: `true` ✅（`grep -n "wrapper_rewritten" .../integration_log.md` → `5:- wrapper_rewritten: true`，退出码 0）
- 护栏 2（幂等重跑关键字）: **未命中** ✅（本次为首次集成，`wrapper_rewrite_action: rewritten`；对 `{op_slug}_kernel/integration_log.md` 执行脚本幂等关键字的 `grep`，退出码 1、无命中）
- 护栏 3（生效 kernel `__module__` 指向 baseline）: `tileops.kernels.elementwise.isfinite.isfinite_kernel.isfinite_fwd_kernel` ✅（指向 baseline 模块 `isfinite_kernel`，非 perf_opt 旧谱系，也非提取参考 `_isfinite_fwd_kernels`）
  - 说明：§2.1 字面命令 `w.isfinite_fwd_kernel`（`import_module('tileops.kernels.elementwise.isfinite')`）在包级抛 `AttributeError`——该算子包 `__init__.py` 仅 re-export `IsfiniteFwdKernel`，与参考算子 `isnan`/`isinf` 同形。故按 §2.1「必须与护栏 3 联合判定」取生效 kernel 的真实来源模块（经 wrapper 模块 `tileops.kernels.elementwise.isfinite.isfinite` 取用 `isfinite_fwd_kernel`）。

## integrate_kernel.py run (2026-10-14T10:57:26+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: rewritten
- ascend_mode: Developer
- functions: isfinite_fwd_kernel
- smoke: pass
