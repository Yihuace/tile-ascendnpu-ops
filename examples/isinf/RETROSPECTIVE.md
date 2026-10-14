# RETROSPECTIVE.md — isinf（op 级，migration-harness）

> 追加式日志：每次 Stage 执行追加一个章节，不覆盖历史。schema 见 `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`。Stage 5 集成复盘为 **op 级单份**（集成问题不归属单个函数）。

## Stage 5 (integrate) Retrospective — IsinfFwdOp / attempt 1 — 2026-10-14

### 结论
`IsinfFwdOp`（`torch.isinf`，bool 输出）集成一次通过：`integrate_kernel.py` 改写 wrapper import 为 baseline/perf_opt 切换块（baseline 激活，`ascend_mode=Developer`），import smoke pass；全量单算子 report `status=passed`（9/9 正确性，10/10 benchmark；`msprof_case_count=5`、`profiler_fallback_count=0`、`invalid_profile_count=0`；`max_abs_err=0.0`）。**零胶水、零调试闭环**——bool 输出字节存储 + host `view(torch.bool)` 已由 Stage 0 wrapper 备好，无需修改任何交付件。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | 任务书 §2.1 护栏 3 字面命令 `w.<func>.__module__`（`w=importlib.import_module('tileops.kernels.<family>.<op_slug>')`）在本仓算子包布局下**不可直接执行**：`{op_slug}/__init__.py` 只 re-export Kernel 类（`IsinfFwdKernel`），属函数 `isinf_fwd_kernel` 只绑定在 wrapper 子模块（`{op_slug}.{op_slug}`）上 | `TileOPs迁移任务书.md:121`；`tileops/kernels/elementwise/isinf/__init__.py`；与参考 `isnan` 同形（`isnan/__init__.py`） | §2.1 护栏 3 命令改为经 wrapper 子模块取函数，或明确要求算子包 `__init__.py` 一并 re-export 属函数 | R |
| Integrate-flow | §2.1 护栏路径 `tileops/kernels/<family>/<op_slug>/integration_log.md` 与实际写入位置不一致：`integrate_kernel.py` 的 `target_dir` 为 `.../{op_slug}_kernel/`，日志落在 `{op_slug}_kernel/integration_log.md` | `TileOPs迁移任务书.md:119-120`；`integrate_kernel.py:494`；实际 `isinf_kernel/integration_log.md` | 统一 §2.1 grep 路径为 `{op_slug}_kernel/integration_log.md`（或 integrator 输出契约同步） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **bool 谓词类算子 Stage 5 零胶水配方**：Stage 0 wrapper 已内置「int8 byte storage + host `view(torch.bool)`」胶水（`forward` 直接 `y = f(N, dtype)(x, N); return y.view(torch.bool)`）；`integrate_kernel.py` 仅改写 import 为 baseline/perf_opt 块，`ascend_mode` 同步后 forward 直接生效（isinf 与 isnan 同形），集成期无需改 wrapper/kernel | P | `tileops/kernels/elementwise/isinf/isinf.py`（forward L149/L150）；`isinf_kernel/integration_log.md`；`-` DESIGN 快照 | `python3 .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/isinf/.migration_meta.json` + `python -m tileops.reporting.cli run --op IsinfFwdOp --prof-mode msprof --kernel-name main` | tilelang 0.1.2+417dae99 + Ascend 910B3 + CANN 8.5.2（npu-smi 25.5.0） | pattern-library/elementwise.md（谓词/bool 输出族：Stage 5 集成零胶水） |

### Transferable Lessons（可迁移教训）

- **bool / 谓词类算子（isnan / isinf / bitwise_not）的 int8 byte storage + host `view(torch.bool)` 胶水在 Stage 0 wrapper 已备**：Stage 5 集成通常**零胶水**，只要 kernel 仍以 int8 字节落盘（`_BOOL_STORAGE_DTYPE`）且 wrapper `forward` 末尾有 `view(torch.bool)` 即生效；只有 report 报出 bool 输出 dtype 不匹配时才是 wrapper 胶水问题（仅允许改 wrapper）。
- **Stage 5 生效 kernel 来源验证**（§2.1 护栏 3）：用 wrapper 子模块取属函数再查 `__module__`（`importlib.import_module('tileops.kernels.<family>.<op_slug>.<op_slug>').<func>.__module__`），须指向 baseline `{op_slug}_kernel`，既非 `perf_opt` 旧谱系、也非提取参考模块 `_{op_slug}_fwd_kernels`；算子包级（`...{op_slug}`）取不到属函数。
- **10 collected vs 5 msprof 的构成规律**：benchmark 收集 manifest 全部 workload（含 3 个 `smoke-1m`），性能分析排除 label 含 smoke 的记录并按 `(operator, case_id)` 去重（`16m-half` 与 `16m` 同 shape+dtype 归并），故 `performance.cases` 小于 `benchmark_tests` 属常态，非缺陷。
