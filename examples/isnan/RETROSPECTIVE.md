# RETROSPECTIVE -- isnan (IsnanFwdOp)

> 追加式日志：每个 Stage 执行追加一个章节，不覆盖历史。schema 见 `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`。函数级复盘见 `isnan_fwd_kernel/RETROSPECTIVE.md`。

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-14

### 结论

`INTEGRATE_COMPLETED`：集成脚本一次通过（`wrapper_rewritten=true`，`ascend_mode=Developer`，smoke=pass），单算子 report 一次通过——正确性 9/9（`max_abs_err=0.0`，atol=rtol=0）、benchmark 10/10、`status=passed`，`msprof_case_count=5`、`profiler_fallback_count=0`、`invalid_profile_count=0`。无需调试闭环、无需 wrapper 胶水修复。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | report 的 `benchmark_tests`（pytest 收集数）与 `msprof_case_count`（性能分析用例数）可以不等，易被误读为「profile 缺失」：analyzer 先按 label 含 `smoke` 过滤（3 个 `smoke-1m` 被排除），再按 `(operator, case_id)` 去重（`16m-half` 与 `16m` 同 shape+dtype 归并） | `tileops/reporting/analyzer.py#L38-L40`（`_is_smoke_record`）、`#L43`（`grouped[(operator, case_id)]`）、`#L118-L122`（summary）；本次 run.json：`benchmark_tests=10` vs `msprof_case_count=5` | 在迁移任务书 §2.1/D9 的 run.json 取数说明增补「`benchmark_tests` 与 `msprof_case_count` 可因 smoke 过滤 + case_id 去重而不等，属预期」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| bool 谓词算子（`isnan`）的 int8 字节存储 + host `view(torch.bool)` 胶水在 Stage 0 wrapper 已就位时，Stage 5 **零胶水修复**即可通过：DESIGN 已冻结字节存储 → wrapper `forward` 返回 `y.view(torch.bool)`、Op `output_dtype=torch.bool`（对比 bitwise_not 需 Stage 5 补 bool/uint8 字节往返胶水） | P | `tileops/kernels/elementwise/isnan/isnan.py#L155-L156`（forward 的 `y.view(torch.bool)`）；`tileops/ops/elementwise/isnan.py#L72`（`output_dtype=torch.bool`）；本次 report `test_isnan_output_dtype_and_values` 通过 | `python -m tileops.reporting.cli run --op IsnanFwdOp --prof-mode msprof --kernel-name main`（从 `examples/TileOPs/` 运行）：正确性 9/9、max_abs_err=0.0、benchmark 10/10 | tilelang build root `/home/developer/tilelang-mlir-ascend/build`；Ascend910B3 x1；CANN 8.5.2；driver 25.5.0；pytorch_npu 2.7.1.post4 | pattern-library |

### Transferable Lessons（可迁移教训）

- 谓词/bool 输出类算子在 **Stage 0 wrapper 阶段**就把「int8 字节存储 → host `view(torch.bool)`」胶水写进 `forward`（并在 Op 里设 `output_dtype=torch.bool`）时，Stage 5 集成无需改 wrapper；若 wrapper 直接把原 dtype 张量传入以 int8 落盘的 factory，Stage 5 才会被迫在胶水层补 view 往返（bitwise_not 先例）。把字节胶水前移到 Stage 0 可让 Stage 5 一次通过。
- report 的 `benchmark_tests` 与 `msprof_case_count` **可以不等**：analyzer 按 label 含 `smoke` 过滤、再按 `(operator, case_id)`（shape+dtype）去重，同 shape 不同 label 会归并；不等不等于 profile 丢失。取数一律用本次时间戳目录的 `run.json`，不用 `latest.json`。
- `--kernel-name main` 对本算子全部命中（`kernel_filter_fallback_count=0`）；若出现 `kernel_name_resolved=None` 才需加 `kernel_name_attempts` 排查。
