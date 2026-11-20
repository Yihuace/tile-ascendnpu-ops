# Integration Log -- DropoutOp

- meta: examples/TileOPs/tileops/kernels/regularization/dropout/.migration_meta.json
- op_slug / family: dropout / regularization
- integrated: `_make_dropout_kernel` -> `tileops/kernels/regularization/dropout/dropout_kernel/_make_dropout_kernel.py`
- design_docs: `_make_dropout_kernel` -> `tileops/kernels/regularization/dropout/dropout_kernel/_make_dropout_kernel_DESIGN.md`
- wrapper: `tileops/kernels/regularization/dropout/dropout.py` (import rewritten to baseline/perf_opt selection block; `ascend_mode = "Developer"` unchanged)
- ascend_mode: Developer (read from Stage 3 product `ASCEND_MODE = "Developer"`; synced to wrapper, no change needed)
- attempts: 1 (no debug loop required)

## 集成核对（Stage 5 pre-check）

- 调用约定核对：Stage 3 工厂 `_make_dropout_kernel(N, dtype, p, seed)` 返回「接受单一 `block_size` 的可调用」；wrapper `DropoutKernel.__init__` 调用 `_make_dropout_kernel(N_total, dtype_str, p, seed)`、`forward` 调 `self.kernel(block_size)(x)`（K11）——**实参个数/顺序一致，无需 wrapper 胶水改动**。
- RNG 缺口 CG-2026-0021：kernel 用核内 counter-based hash 替代 cuRAND；正确性判据为统计等价（3σ）+ 确定性重放 + 缩放契约 + 边界（非 GPU mask 逐位一致），与 `tests/ops/test_dropout.py` 一致。
- 陷阱核对：VP-2026-0133（非输出标量参数）/ VP-2026-0189（输出约定）——本 kernel `main(x, y)` + `out_idx=[1]`，无运行时 shape 标量，与 wrapper 契约吻合。

## 验证结果

- report_run_json: examples/TileOPs/reports/tileops/20261014_110038_288463_DropoutOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261014_110038_288463_DropoutOp/report.md
- report_html: examples/TileOPs/reports/tileops/20261014_110038_288463_DropoutOp/report.html
- run_id: 20261014_110038_288463_DropoutOp
- report_status: passed
- correctness: pass (28/28 cases, `summary.correctness_passed=true`, `correctness_failed=0`；0 skipped/errors)
- benchmark: pass (`benchmark_passed=true`, 6/6 workloads, `msprof_case_count=6`, `invalid_profile_count=0`, `profiler_fallback_count=0`)

### benchmark 数值（来自 run.json / benchmark.md；单位 us）

| workload | dtype | shape | block_size | tileops | torch baseline |
|---|---|---|---|---|---|
| smoke-dropout-1m | float32 | [1024,1024] | 1024 | 23.12 | 129.17 |
| smoke-dropout-1m | float16 | [1024,1024] | 2048 | 19.64 | 115.68 |
| smoke-dropout-1m | bfloat16 | [1024,1024] | 2048 | 20.16 | 135.10 |
| tokens-1k-hidden-4k-fp16 | float16 | [1024,4096] | 2048 | 70.80 | 122.23 |
| tokens-1k-hidden-10k-bf16 | bfloat16 | [1024,10240] | 2048 | 172.62 | 187.40 |
| tokens-1k-hidden-4k-fp32 | float32 | [1024,4096] | 1024 | 83.84 | 125.29 |

> 说明：tileops 各档均优于 torch `F.dropout` 基线；带宽利用率受软件 PRNG（~19–21 op/元素，替代源硬件 RNG）限制，符合 DESIGN §1.6.0 的固有落差预期。性能仅记录，不修复。

## 调试历史

- 无（attempt 1 一次通过：集成脚本 smoke pass，report correctness 28/28 + benchmark 6/6，`status=passed`）。

## issues

- report.md 出现 6 条 `missing tileops candidate record` 警告（对应 6 个 tileops 性能记录未匹配到 candidate）；`status=passed`、`benchmark_passed=true`、`invalid_profile_count=0`，属报告层候选记录匹配提示，不影响正确性/benchmark 结论，仅记录不修复。
- `integrate_kernel.py` 重写 wrapper import 时，因 wrapper 脚手架已存在同款 baseline/perf_opt 注释块，生成块与既有块**重复**（wrapper L58–L89 出现两段相同注释 + 两行 perf_opt 注释，仅 1 行 baseline import 生效）。功能无害（幂等 import 冒烟通过），记为 Integrate-flow 流程缺陷（见 RETROSPECTIVE）。
