# RETROSPECTIVE — AddFwdOp（`add`，migration-harness）

## Stage 5 (integrate) Retrospective — first_integrate — 2026-10-13

集成结论：`INTEGRATE_COMPLETED`，attempts=1。集成脚本一次通过（smoke=pass），单算子 report 一次通过（`status=passed`，correctness 24/24、benchmark 11/11、msprof_case_count=11 / fallback 0 / invalid 0）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | Stage-0 scaffold 生成的 wrapper 模板**预置**了 baseline/perf_opt 注释块（baseline 行原指向 `._add_fwd_kernels`），但 `rewrite_wrapper_import` 的替换正则只匹配**那一行 import**（`from .{extracted_module} import ...`），不识别模板自带的整块；替换后生成块与模板残留注释块叠加，wrapper 顶部出现重复的 selection 注释块。功能无影响（active import 唯一且正确），但模板保真被破坏 | `tileops/kernels/elementwise/add/add.py` L45-74（重复块）；`.agents/skills/add-npu-op/scripts/integrate_kernel.py` L310-334（`rewrite_wrapper_import`）、L319（already-integrated 探测正则）；对照 `clamp_scalar.py` L40-54 为单块（其 Stage-0 模板未预置该块） | scaffold wrapper 模板不预置 baseline/perf_opt 块，或 `rewrite_wrapper_import` 按整块（含标题注释）替换而非单行替换 | R |
| Integrate-flow | 集成脚本幂等检查 `re.search(r"from \.add_kernel(?:\.\w+)?\s+import", text)` 只认**未注释**的激活 import；首跑命中 `rewritten`（正常），但「already integrated」护栏 grep 在首跑恒为空，易被误读为护栏失败 | `integration_log.md` `- wrapper_rewrite_action: rewritten`；`.agents/skills/add-npu-op/scripts/integrate_kernel.py` L319,L358-363 | 在 integrate_kernel.py usage 注明「idempotency 护栏须在二次运行后采样」，或日志同时给出探测命中位置 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| npuir 二元 elementwise `add` 端到端集成实测（Developer 模式，block_size=8192）：correctness 24/24、max_abs_err=0；benchmark 11/11 有效 profile（fallback 0/invalid 0），Ratio 66.12%–82.16%，latency 13.02–74.86us | D | `reports/tileops/20261013_162043_457669_AddFwdOp/run.json`（`summary.correctness_tests=24`/`benchmark_tests=11`/`avg_max_abs_err=0.0`/`msprof_case_count=11`）；`benchmark.md` | `python -m tileops.reporting.cli run --op AddFwdOp --prof-mode msprof --kernel-name main` | tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b + CANN 8.5.2 + Ascend910B3 / torch_npu 2.7.1.post4 / 2026-10-13 | pattern-library/elementwise.md |

### Transferable Lessons（可迁移教训）

- harness 集成期 wrapper 胶水**通常无需人工修复**：Stage-0 模板已按 K9/K11 去除 `threads/npt`、`default_config` 返回 NPU `block_size`，Stage-3 factory 签名 `(N, dtype, alpha=..., block_size=...)` 与 wrapper `forward` 调用一致；集成前先核对 factory 签名 vs wrapper 调用，只有真正失配（如 `out_idx`、`block_size` 传参形态）才动 wrapper。
- 集成后必须用护栏 3（`wrapper 模块.add_fwd_kernel.__module__`）确认生效的是 **baseline** 而非 perf_opt 旧谱系；护栏 2（`already integrated`）首跑不出现属正常，二次运行才出现——不得单凭该 grep 空/命中判失败。
- 集成脚本产物可能带 cosmetic 冗余（见 Skill Flow Issues 首行）；禁止为消除外观冗余而手工改 wrapper，以免掩盖工具链缺陷——应记入复盘交 evolver。
