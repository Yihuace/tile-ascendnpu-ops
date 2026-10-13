# Integration Log -- BitwiseNotFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/bitwise_not/.migration_meta.json
- integrated: bitwise_not_fwd_kernel -> bitwise_not_kernel/bitwise_not_fwd_kernel.py
- design_docs: bitwise_not_fwd_kernel -> bitwise_not_kernel/bitwise_not_fwd_kernel_DESIGN.md
- attempts: 2 (attempt 1: Stage 5 wrapper dtype glue + reported Stage 0 scaffold interface defect;
  attempt 2: re-run after conductor Stage-0 scaffold correction added the `BitwiseNotTest.__init__`
  adapter -- full correctness + benchmark now pass, no kernel/wrapper change needed)

## 集成摘要

- 集成脚本: `python3 .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/bitwise_not/.migration_meta.json`（从 `examples/TileOPs/` 运行）。末行 `[json]` 摘要：import 改写成功、`ascend_mode=Developer`、smoke=pass。二次运行幂等（`[wrapper] ... already integrated`，日志出现 `already integrated`），kernel 与 DESIGN 整文件覆盖一致。
- 聚合 `__init__.py`: re-export `bitwise_not_fwd_kernel`（baseline 模块 `bitwise_not_fwd_kernel.py`）。
- wrapper import: 已改写为 baseline/perf_opt 双 import 切换块；baseline（`from .bitwise_not_kernel import bitwise_not_fwd_kernel`）默认激活，perf_opt 行注释占位。
- ascend_mode: Developer（集成脚本从 Stage 3 kernel 的 `ASCEND_MODE = "Developer"` 读取；wrapper 原声明一致，未变更）。
- import 冒烟: `python -c "import tileops; from tileops.kernels.elementwise.bitwise_not import BitwiseNotFwdKernel"` 通过；`bitwise_not_fwd_kernel.__module__` = `tileops.kernels.elementwise.bitwise_not.bitwise_not_kernel.bitwise_not_fwd_kernel`（Baseline 已生效）。

## 验证结果（attempt 2，终态）

- report_run_id: 20261013_181824_602023_BitwiseNotFwdOp
- report_run_json: examples/TileOPs/reports/tileops/20261013_181824_602023_BitwiseNotFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261013_181824_602023_BitwiseNotFwdOp/report.md
- report_html: examples/TileOPs/reports/tileops/20261013_181824_602023_BitwiseNotFwdOp/report.html
- report_status: **passed**
- correctness: **pass**（`summary.correctness_passed=true`，11/11 通过，0 失败；含 8 个 `test_bitwise_not_op[...]` dtype 用例 + 3 个负例/roofline 用例）
- benchmark: **pass**（`summary.benchmark_passed=true`，13/13 通过，0 失败；带 profile 的 case_count=7）
- msprof_case_count / profiler_fallback_count / kernel_filter_fallback_count / invalid_profile_count: **7 / 0 / 0 / 0**
- max_abs_err: 0.0（`operators[0].max_abs_err=0.0`，全部正确性用例精确相等 atol=rtol=0）

### per-shape perf（来自 run.json performance.cases，msprof，profiler_valid=true）

| label | shape | dtype | block_size | latency_us | roofline_util% | bandwidth_tb_s |
|-------|-------|-------|-----------|-----------|----------------|----------------|
| 16m | (4096,4096) | int32 | 8192 | 50.06 | 76.60 | 1.75 |
| 16m | (4096,4096) | int64 | 8192 | 129.70 | 72.43 | 1.75 |
| 256m | (16384,16384) | int32 | 8192 | 1610.94 | 72.50 | 1.75 |
| 16m-int-small | (4096,4096) | bool | 8192 | 29.50 | 32.50 | 1.75 |
| 16m-int-small | (4096,4096) | uint8 | 8192 | 23.64 | 40.55 | 1.75 |
| 16m-int-small | (4096,4096) | int8 | 8192 | 23.00 | 41.68 | 1.75 |
| 16m-int-small | (4096,4096) | int16 | 8192 | 29.52 | 64.95 | 1.75 |

- report_status=failed 的历史（attempt 1）见下「调试历史」。attempt 2 未修改 kernel / wrapper / tests。

## 调试历史

- attempt 1（wrapper dtype 胶水，非正确性失败修复）：集成后对全部 6 个整型/bool dtype 直接调用 `BitwiseNotFwdOp` 验证，发现 byte-storage 胶水缺失：Stage 3 kernel 对 bool（i1 不可 GM load/store）与 uint8（运行时无输出 dtype 映射）统一以 **int8 字节存储** 读写，而原 wrapper 的 `forward` 直接把原始 dtype 张量传给 factory：
  - bool 输入：factory 声明 int8 张量 → 输出 int8，`assert_close` 因 dtype 不匹配失败（`out_dtype=torch.int8` vs ref `torch.bool`）。
  - uint8 输入：uint8 值被当作 int8 位型且输出为 int8 → 数值与 dtype 双重不符（`allclose=False`）。
  按集成规范（wrapper 胶水层允许 dtype 转换）在 `bitwise_not.py::forward` 增加元数据视图往返：`x.contiguous().view(torch.int8)` 进入（bool/uint8），`y.view(torch.bool)` / `y.view(torch.uint8)` 返回；整型直通。`.view` 为纯元数据重解释，字节内容不变，bit-exact。备份：`history_version/bitwise_not_s5_glue0.py`。修正后对 6 dtype 直接 op 调用全部 `torch.testing.assert_close(atol=0, rtol=0)` 通过（int32/bool/uint8/int8/int16/int64，out_dtype 全部回原 dtype）。wrapper/Op dtype 校验已覆盖 6 个整型 dtype，测试容差为精确相等（atol=rtol=0）。
- 阻塞根因（**不可修文件**，Stage 0 脚手架接口不一致，未消耗修复 attempt）：
  `tests/ops/test_bitwise_not.py::test_bitwise_not_op` 以 `BitwiseNotTest(n_total, dtype)`（`n_total: int`）构造，而 `BitwiseNotTest` 未定义 `__init__`，实际继承 `tileops/workloads/elementwise.py::BitwiseNotWorkload.__init__(self, shape: tuple, dtype)`，其 `self.shape = tuple(shape)` 对整型 `1048576` 执行 `tuple(1048576)` → `TypeError: 'int' object is not iterable`（`tileops/workloads/elementwise.py:58`）。
  `tests/` 与 `tileops/workloads/` 均在 Stage 5 禁止修改范围（integrator 规格 §核心原则 2 / §约束 4），故如实报告不改。
  对比：sibling `tests/ops/test_hardswish.py::HardswishTest` 与 `tests/ops/test_clamp_scalar.py::ClampScalarTest` 都在测试子类里加了 `__init__` 适配器 `super().__init__((n_total,), dtype)`，bitwise_not 的脚手架漏掉了该适配器。
  **建议（交回 conductor）**：在 `tests/ops/test_bitwise_not.py` 给 `BitwiseNotTest` 增加 `__init__(self, n_total, dtype)`（内部 `super().__init__((n_total,), dtype); self.n_total = n_total`），或让 `BitwiseNotWorkload` 兼容 `int`（`shape`/`n_total` 二选一），与 hardswish / clamp_scalar 先例对齐。bench 侧 `BitwiseNotWorkload(shape, dtype)` 以 manifest `input_shape`（tuple）调用，故 workload 的 shape 形参是 bench 契约所必需 —— 不一致点在测试侧。
- 说明：以上正确性失败与 wrapper/kernel 无关；直接 op 级验证（绕过 pytest fixture）已证明 6 dtype bit-exact。因 report 的 `summary.correctness_passed=false`，终态为 `[INTEGRATE_FAIL]`。
- attempt 2（无修复，仅重跑；conductor 已在 Stage 0 脚手架修正 `tests/ops/test_bitwise_not.py`，为 `BitwiseNotTest` 增加 `__init__(self, n_total, dtype): super().__init__((n_total,), dtype)` 适配器，落回禁止修改文件的既有缺陷由 conductor 处理）：重新运行幂等集成脚本（`already integrated`，`ascend_mode=Developer`，smoke=pass）→ import 冒烟通过 → 完整单算子 report。**正确性 11/11、benchmark 13/13、status=passed**，msprof_case_count=7、fallback/invalid 全 0，max_abs_err=0.0。attempt-1 的 wrapper byte-storage 胶水（bool/uint8 → int8 元数据视图）保持不变，即为终态。

## integrate_kernel.py run (2026-10-13T17:58:20+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: rewritten
- ascend_mode: Developer
- functions: bitwise_not_fwd_kernel
- smoke: pass

## integrate_kernel.py run (2026-10-13T18:15:58+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: already integrated
- ascend_mode: Developer
- functions: bitwise_not_fwd_kernel
- smoke: pass

## integrate_kernel.py run (2026-10-13T18:17:57+0800)

- wrapper_rewritten: true
- wrapper_rewrite_action: already integrated
- ascend_mode: Developer
- functions: bitwise_not_fwd_kernel
- smoke: pass
