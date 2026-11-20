# Integration Log -- ErfFwdOp

- meta: `examples/TileOPs/tileops/kernels/elementwise/erf/.migration_meta.json`
- integrated: `erf_fwd_kernel` -> `tileops/kernels/elementwise/erf/erf_kernel/erf_fwd_kernel.py`
- design_docs: `erf_fwd_kernel` -> `tileops/kernels/elementwise/erf/erf_kernel/erf_fwd_kernel_DESIGN.md`
- ascend_mode: `Developer` (read from Stage 3 product `ASCEND_MODE`; wrapper `Kernel.ascend_mode` synced, unchanged)
- attempts: 1

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261012_222359_253405_ErfFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261012_222359_253405_ErfFwdOp/report.md`
- report_status: passed
- correctness: pass (7/7 cases; `summary.correctness_passed=true`, avg max abs error 4.33e-05)
- benchmark: pass (16/16 cases in `bench_unary_elementwise.py`; 10/10 ErfFwdOp msprof records `profiler_valid=true`; `profiler_fallback_count=0`, `invalid_profile_count=0`)

### benchmark 数值（来自本次 run.json `performance.cases`，仅 ErfFwdOp）

| label | dtype | shape | latency (us) | roofline (%) | bandwidth (TB/s) |
|---|---|---|---:|---:|---:|
| smoke-1m | float32 | [1048576] | 31.8406 | 19.25 | 1.8000 |
| smoke-1m | float16 | [1048576] | 18.6404 | 33.99 | 1.8000 |
| smoke-1m | bfloat16 | [1048576] | 18.6004 | 35.40 | 1.8000 |
| elementwise-16m | float16 | [4096,4096] | 257.2651 | 39.05 | 1.8000 |
| elementwise-16m | bfloat16 | [4096,4096] | 256.9451 | 40.65 | 1.8000 |
| elementwise-16m | float32 | [4096,4096] | 465.3293 | 20.97 | 1.8000 |
| elementwise-64m | float16 | [8192,8192] | 1015.4803 | 39.56 | 1.8000 |
| elementwise-64m | bfloat16 | [8192,8192] | 1017.1603 | 41.06 | 1.8000 |
| elementwise-256m | float16 | [16384,16384] | 4074.4214 | 39.43 | 1.8000 |
| elementwise-256m | bfloat16 | [16384,16384] | 4066.3015 | 41.08 | 1.8000 |

> benchmark 只报告不修复；数值与 roofline 利用率见上表。

## 调试历史

- attempt 1:
  - 症状：集成后全量正确性 7 个用例全部失败，`ValueError: Expected 2 inputs, got 1`（`pytest/correctness.xml`；报错于 `_erf_fwd_wrapped` 调用 `erf_fwd_kernel(N, dtype_str, block_size=block_size)(x)`）。
  - 分类：wrapper 胶水错误（集成 kernel 的 `main(x, y, shape)` 新增动态 `shape` 参数，`out_idx=[1]` 下编译可调用签名实为 `(x, shape)`，wrapper 只传了 `x`）。
  - 定位：与既有 `bitwise_not` 集成同源缺陷；对照 `tileops/kernels/elementwise/bitwise_not/bitwise_not.py` 的 `kernel(x, N)` 调用点确认修复方向（无需 IR dump）。
  - 修复（仅 wrapper 胶水层 `tileops/kernels/elementwise/erf/erf.py`，备份 `erf_kernel/history_version/erf_wrapper_s5_attempt1.py`）：调用 kernel 时补传动态 `shape=N` -> `erf_fwd_kernel(N, dtype_str, block_size=block_size)(x, N)`。kernel 文件未改。
  - 结果：重新运行完整单算子 report `status=passed`，correctness 7/7、benchmark 16/16（ErfFwdOp 10/10 msprof 有效）。

## issues

- report `warnings` 含 16 条 `missing tileops candidate record`（BitwiseNotFwdOp 6 + ErfFwdOp 10，每个 benchmark case 一条）。benchmark 用例仍 16/16 passed、msprof profile 全部 `profiler_valid=true`；该 warning 属 report/record 层提示（`tileops/reporting/analyzer.py:46`），未影响正确性与 benchmark 结论，仅记录不修复（bitwise_not 集成日志已登记同类现象）。
- `integrate_kernel.py` 对**已预置**的 wrapper "Kernel source selection" 表头未做幂等合并，导致 `erf.py` 头部出现重复的表头注释块（两段相同的 `# Kernel source selection ...`）。纯注释重复，不影响 import 与运行（第二次运行即检测到 `from .erf_kernel import` 而不再改写）。记录为流程缺陷，未修改（见 RETROSPECTIVE Integrate-flow 提案）。
- 无 DESIGN.md 缺失（`[warn] no DESIGN.md` 未出现）。
