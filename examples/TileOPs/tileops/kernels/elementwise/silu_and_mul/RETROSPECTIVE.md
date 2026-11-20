# RETROSPECTIVE — silu_and_mul (op 级)

> migration-harness 复盘日志。Stage 1/2/3 的逐函数复盘位于 `examples/silu_and_mul/{func}/RETROSPECTIVE.md`；
> 本文件记录 Stage 5（op 级集成）复盘。追加式日志，不覆盖历史章节。

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` 把 wrapper 的 extracted-module import 机械改写为 `{op_slug}_kernel` import 时，**不校验 name 是否真的被集成包导出**：`gen_init_py` 对缺失 name 只打 `# [warn] ... not found in integrated modules`，却仍把它放进 `__all__`；`gen_kernel_source_block`（baseline import）与 `smoke_test` 的 `wrapper_names` 也照带该 name → 必然 `ImportError` / smoke assert 失败。 | `integrate_kernel.py` L269–L282（gen_init_py）、L323–L364（gen_kernel_source_block）、L493–L495（wrapper_names）；本任务产物 `silu_and_mul_kernel/__init__.py` 的 warn 行 + wrapper L61；运行输出 `[error] import smoke test FAILED`。 | `integrate_kernel.py`：在 `gen_init_py` / `gen_kernel_source_block` / smoke 名单生成前，用 `defining_module()` 过滤 `wrapper_names`，缺失者仅 warn 并从 baseline import、`__all__`、smoke 名单中剔除。 | R |
| Integrate-flow | smoke 的 `wrapper_names` 取自**改写前** wrapper（`parse_wrapper_imports`），与 `__init__.py` 生成逻辑（同一 names 经 `defining_module` 过滤）语义不自洽：前者不做过滤，后者做过滤但只 warn。 | `integrate_kernel.py` L493–L495 vs L539–L543；本任务首次运行 smoke 名单 = `["_make_fused_gated_explicit","activation_func"]`。 | 同上；或让 smoke 名单回落到集成包 `__init__.__all__`。 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 迁移集成 canonical 修复：NPU kernel 产品**不保留** GPU-only helper（如 `activation_func`）——wrapper baseline 只 import `{op_slug}_kernel` 的 kernel factory，helper 从冻结的 Stage-0 契约模块 `_{op_slug}_fwd_kernels.py` 以模块对象导入（`from . import _..._fwd_kernels as _stage0_contract; activation_func = _stage0_contract.activation_func`），既保 `_effective_op_func` 契约又不破坏脚本幂等 | P | `tileops/kernels/elementwise/silu_and_mul/silu_and_mul.py` L61–L75（修复后）；同族先例 `tileops/kernels/elementwise/gelu_and_mul/gelu_and_mul.py` L63–L77；DESIGN.md §0.6 R3（op_func 内联） | `cd examples/TileOPs && python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/silu_and_mul/.migration_meta.json`（修复前 smoke FAIL，修复后 `[smoke] smoke-ok`） | tilelang 0.1.2+8875071d41a4587c21b6c1be0eb075d090b5bbf4 / Ascend910_9362 (910B2C 族) / CANN 9.0.0 / torch 2.7.1+cpu / 2026-10-16 | pattern-library/elementwise.md（migration-harness 集成段）或 Integrate-flow 文档 |
| integration smoke 覆盖整个 `tileops.kernels.{family}` 包的 `__init__` 传递导入链 —— 同族任一同批集成算子 wrapper 断裂都会让本算子 smoke 失败；定位时须先看 traceback 的**首个失败模块**是否本算子 | P | 本任务首次运行 traceback 首错为 `gelu_and_mul.py` L63（同族，另一 writer 10:16:11 集成）；修复该 writer 后首错转为 `silu_and_mul.py` L61 | 两次运行记录见 integration_log.md 调试历史 | tilelang 0.1.2+8875071d41a4587c21b6c1be0eb075d090b5bbf4 / 2026-10-16 | tilelang-op-develop / integrator SKILL（集成烟测定位段） |
| migration-harness elementwise 融合门控族（`silu_and_mul` / `gelu_and_mul`，同 `_make_fused_gated_explicit` family）wrapper-glue 缺陷案例索引：同一 S3 wrapper 模板 + 同一 NPU redesign 导致同一 `activation_func` 缺失 | C | `examples/silu_and_mul/_make_fused_gated_explicit/_make_fused_gated_explicit.py`；`examples/TileOPs/tileops/kernels/elementwise/gelu_and_mul/gelu_and_mul.py` | 见上两行 repro | tilelang 0.1.2+8875071d41a4587c21b6c1be0eb075d090b5bbf4 / 2026-10-16 | tilelang-skill-evolution 案例索引（elementwise 融合门控族） |

### Transferable Lessons（可迁移教训）

- **迁移后的 NPU kernel 产品通常不保留 GPU-only 的 `op_func` 辅助**（silu 数学被内联为 v-prefix 链，`op_func` 仅作形态标识并 `del`）。因此 wrapper 的 baseline import **不得**从 `{op_slug}_kernel` import 该 helper；只 import kernel factory，helper 从冻结的 Stage-0 契约模块（`_{op_slug}_fwd_kernels.py`）以模块对象方式导入。
- 若 `integrate_kernel.py` 报 `[error] import smoke test FAILED`，**先读 traceback 首个失败模块**：`tileops.kernels.{family}.__init__` 会传递导入同族所有算子，同族任一算子 wrapper 断裂都会污染本算子 smoke；不要误判为本算子 kernel 问题。
