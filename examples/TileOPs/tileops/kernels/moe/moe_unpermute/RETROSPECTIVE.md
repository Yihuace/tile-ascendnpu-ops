# RETROSPECTIVE — moe_unpermute (op-level)

> Stage 5 (integrate) 复盘为 op 级单份（集成问题不归属单个函数）。
> Stage 1/2/3 的逐函数复盘见 `examples/moe_unpermute/_make_unpermute_kernel/RETROSPECTIVE.md`。

## Stage 5 (integrate) Retrospective — integrate_only (report deferred by conductor) — 2026-10-16

Scope: `integrate_kernel.py` deterministic integration + import smoke only; the
single-op `tileops.reporting.cli run` correctness/benchmark report was **not**
run in this invocation (explicit conductor instruction), so no debug closed-loop
occurred (attempts=0). Statements below are static-source analysis where noted.

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | GPU-ported wrapper call convention not reconciled with the Stage 3 kernel's factory topology. The GPU-scaffold wrapper treats the factory as two-level (`self._unpermute_fn = _make_unpermute_kernel(...)` then `fn = self._unpermute_fn()` with no args), but the Stage 3 kernel puts `@tilelang.jit(...)` on the shape-parameterised **outer** factory (one-level: `kernel = _make_unpermute_kernel(...)`; `kernel(tensors...)`). `integrate_kernel.py` copies the kernel and rewrites the import but does not reconcile this call shape. (Static analysis; the deferred report run is expected to surface it at runtime.) | `examples/TileOPs/tileops/kernels/moe/moe_unpermute/moe_unpermute.py:136,196`; `examples/moe_unpermute/_make_unpermute_kernel/_make_unpermute_kernel.py:147,293-300`; `tilelang/jit/__init__.py:126-173`; `tilelang/jit/kernel.py:177-194` | In `integrate_kernel.py` (or the add-npu-op integrator doc) add a pre-report check: compare the wrapper's factory invocation arity against the integrated kernel's factory signature/topology and flag (or auto-fix) a mismatched call convention. | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| none | - | - | - | - | - |

### Transferable Lessons（可迁移教训）

- 集成 harness 算子时，先比对 **wrapper 的工厂调用拓扑** 与 **Stage 3 kernel 的工厂拓扑**：GPU 抽取件常见「外层普通工厂 + 内层 `@tilelang.jit` 0 参工厂」两层形态（wrapper 存工厂后调 `self._fn()`），而 NPU Stage 3 重实现常把 `@tilelang.jit` 直接放到带 shape 参数的**外层**工厂上（一层形态，`self._fn` 本身即已编译 kernel）。两者不匹配时 wrapper 的多余 `()` 会在 forward 期报错——这是 import smoke **测不出**的（smoke 只验导入导出符号），须等 report 正确性运行暴露。
- `integrate_kernel.py` 只覆盖导入改写与 `__init__.py` 聚合，**不删除**被替换的旧 GPU 抽取文件（本例 `_moe_unpermute_fwd_kernels.py` 仍留在包内），集成后包内会同时存在 Stage-3 产物与失效的 GPU 脚手架文件；审查包内容时勿把后者当作有效实现。
- 集成脚本幂等：二次运行报 `wrapper_rewritten=false` / `wrapper_mode_updated=false` 且 smoke 仍 pass，可安全重跑。
