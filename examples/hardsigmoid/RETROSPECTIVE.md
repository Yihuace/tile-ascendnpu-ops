# RETROSPECTIVE.md — hardsigmoid（op 级，Stage 5 integrate）

> 追加式日志：每次 Stage 执行追加一个章节，不覆盖历史。schema 见 `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`。
> 本文件为 migration-harness 场景 Stage 5 的 op 级集成复盘（单份，集成问题不归属单个函数）；Stage 1/2/3 的逐函数复盘见 `examples/hardsigmoid/hardsigmoid_fwd_kernel/RETROSPECTIVE.md`。

## Stage 5 (integrate) Retrospective — first_integration（migration-harness）— 2026-10-12

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | 脚手架 wrapper 的 `forward()` 沿用 GPU `threads/npt` 调用约定（`self.kernel(threads, npt)` → `self._compiled_fn(x)`），而 `integrate_kernel.py` 仅改写 import 块 + 同步 `ascend_mode`，不覆盖 `forward()`/`default_config`；NPU factory（`kernel(x, N)`）被以 `(threads, npt)` 调用 → 5 个 op 用例报 `AttributeError: 'int' object has no attribute 'device'`，集成者须手工适配 | fix 前 `tileops/kernels/elementwise/hardsigmoid/hardsigmoid.py` `forward()`（`self.kernel(threads, npt)` 行）；`integrate_kernel.py` `rewrite_wrapper_import`（仅改 import）+ `sync_wrapper_ascend_mode`（仅改 ascend_mode） | add-npu-op skill 增补「脚手架 wrapper `forward()` 须适配 NPU factory 调用约定 `kernel(x, N)`」，并在 `smoke_test` 增加构造实例跑一次 forward 的调用约定 smoke（当前 smoke 只查 import + 类存在，不触达 forward） | R |
| Integrate-flow | 脚手架 wrapper 的 `default_config` 沿用 GPU 派生的 `block_size`（fp32 512 / fp16-bf16 1024），与 NPU kernel 设计值 8192（DESIGN.md §5.2）不符；512/1024 正确性可用（UB 预算充足、任意 block_size 均正确），但非设计值、性能次优 | fix 前 `default_config` 返回 `_BLOCK_SIZE_FP32=512`/`_BLOCK_SIZE_NON_FP32=1024`；`hardsigmoid_fwd_kernel_DESIGN.md` §5.2 `block_size=8192` | 脚手架生成器（或 `integrate_kernel.py`）在 NPU 集成时以 kernel 签名 `block_size` 默认值（或 DESIGN §5.2）覆盖 wrapper `default_config`，勿沿用 GPU `threads*npt` | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| NPU elementwise 激活 wrapper 调用约定：`@tilelang.jit` factory `kernel(N, dtype, block_size=...)` 返回编译产物、以 `kernel(x, N)` 调用（`out_idx=[1]` 时 y 自动分配、第二实参为 shape int），无 GPU `threads/npt` 分解 | P | `tileops/kernels/elementwise/mish/mish.py` `_mish_fwd_wrapped`（`mish_fwd_kernel(N, dtype, block_size=...)(x, N)`）；本任务 fix 后 `forward()` | `python -m tileops.reporting.cli run --op HardsigmoidFwdOp --prof-mode msprof`（fix 前 `AttributeError`，fix 后 correctness 7/7 + benchmark 9/9） | tilelang 0.1.2 dev-root（`/home/developer/tilelang-mlir-ascend/build`）+ Ascend（report env: Ascend910B3 x 1，CANN 8.5.2） | pattern-library/elementwise.md（或 add-npu-op skill 集成章节） |
| `integrate_kernel.py` 仅改写 wrapper import 块 + `ascend_mode`，不覆盖 `forward()` 调用约定与 `default_config` 的 block_size（脚手架 GPU 残留须集成者手工适配，smoke 不触达 forward 无法拦截） | R | `integrate_kernel.py` `rewrite_wrapper_import`/`sync_wrapper_ascend_mode` 范围；本任务 fix 前 `forward()`/`default_config` | 对任意新迁移 elementwise op 跑 `integrate_kernel.py` 后直接跑 report（脚手架 `forward()` 保留 GPU `threads/npt` 约定即复现 `AttributeError: 'int' object has no attribute 'device'`） | tilelang 0.1.2 dev-root + Ascend（report env: Ascend910B3 x 1，CANN 8.5.2） | `.agents/skills/add-npu-op`（集成流程）/ `integrate_kernel.py` |

### Transferable Lessons（可迁移教训）

- 迁移 elementwise 激活到 NPU 后集成 TileOPs 时，wrapper `forward()` 必须适配 NPU factory 调用约定：`@tilelang.jit` 工厂 `kernel(N, dtype, block_size=...)` 返回编译产物，以 `kernel(x, N)` 调用（`out_idx=[1]` 时 y 自动分配、第二实参为 shape int）；脚手架若保留 GPU `threads/npt` 分解（`self.kernel(threads, npt)` → `_compiled_fn(x)`），会因把 int（`threads`）当 tensor 传而报 `AttributeError: 'int' object has no attribute 'device'`。
- wrapper `default_config` 的 `block_size` 应取 NPU 设计值（DESIGN §5.2，本算子 8192），勿沿用 GPU `threads*npt` 派生的 512/1024；任意 block_size 正确性均可（UB 预算充足），但非设计值性能次优。
- `integrate_kernel.py` 只做 import 块改写 + `ascend_mode` 同步，不覆盖 `forward()`/`default_config`；集成后须检查 wrapper 是否仍含 GPU 调用约定残留（`self.kernel(threads, npt)` 形态），必要时手工适配（本算子已适配为 `kernel(x, N)`）。
