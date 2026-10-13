# RETROSPECTIVE — SubFwdOp（`sub`，op-level，migration-harness）

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | 同族已集成算子（add）的 wrapper 可直接作为 Stage 5 胶水基线：`sub.py` 的 `__init__/forward/default_config` 与 Stage-3 factory `sub_fwd_kernel(N, dtype, alpha=1, output_dtype=None, block_size=8192, num_cores=None)` + `out_idx=[2]` 完全匹配，集成一次通过、无需人工胶水修复 | sub.py:104-146；sub_kernel/integration_log.md（attempts: 1，no manual wrapper glue fix） | 集成 skill 文档在「同族已集成」时提示「以同族 wrapper 为胶水模板，逐点核对 factory 签名 / out_idx / config 键」 | R |
| Integrate-flow | `sub.py`（scaffold 生成）未预置 baseline/perf_opt 注释块，故 `integrate_kernel.py` 改写后无重复 selection 注释块——反证 add.py 的 cosmetic 重复块源于 scaffold 模板预置，而非集成脚本行为 | sub.py:45-59（单一 selection 块）；add_kernel/integration_log.md#issues（重复块 cosmetic） | scaffold 模板不应预置 baseline/perf_opt 注释块（交由 integrate_kernel.py 生成），可消除 add 类残留 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 同族 `AlphaScaledBinaryKernel`（add→sub）集成零手工修复：Stage-0 scaffold wrapper 胶水（`alpha=self.alpha` 透传 + `block_size` 注入 + `forward(...)(a,b,N)` + `out_idx=[2]`）与 Stage-3 factory 直接对齐，smoke 一次通过 | P | tileops/kernels/elementwise/sub/sub.py:104-146；tileops/kernels/elementwise/sub/sub_kernel/integration_log.md | `python3 .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/sub/.migration_meta.json` | tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B3 / torch 2.7.1 / 2026-10-13 | pattern-library/cases.md |
| SubFwdOp 集成期实测：correctness 24/24（17 数值用例 max_abs_err=0.00e+00；7 非数值用例无误差指标）、benchmark 15/15 全 msprof 有效（fallback=0/invalid=0），roofline 利用率 68.4–82.7% | D | reports/tileops/20261013_170400_275674_SubFwdOp/run.json（summary.correctness_passed=true, msprof_case_count=15, profiler_fallback_count=0, invalid_profile_count=0） | `python -m tileops.reporting.cli run --op SubFwdOp --prof-mode msprof --kernel-name main`（从 examples/TileOPs/） | tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B3 / torch 2.7.1 / 2026-10-13 | pattern-library/cases.md |

### Transferable Lessons（可迁移教训）

- 迁移同族 `AlphaScaledBinaryKernel` 算子（add/sub/…）进入 TileOPs 集成期：直接复用已集成同族算子（add）的 wrapper 作为胶水模板，逐点核对 `__init__` 的 factory 调用、`forward` 的 `(a, b, N)` 调用与 `out_idx=[2]`、`default_config` 的 `block_size` 键即可零手工修复通过；不要凭算子名猜测模式，`ascend_mode` 由 `integrate_kernel.py` 从 Stage-3 kernel 的 `ASCEND_MODE` 读取（本函数为 Developer）。
- harness 集成期报告只需 `python -m tileops.reporting.cli run --op {PascalCaseOpName} --prof-mode msprof --kernel-name main`（从 `examples/TileOPs/` 运行，串行无 `-n`）；结论必须来自本次时间戳目录的 `run.json`，`status=passed` 即 correctness+benchmark 均通过。
