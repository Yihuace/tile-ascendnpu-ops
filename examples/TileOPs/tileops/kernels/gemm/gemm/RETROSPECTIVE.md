# RETROSPECTIVE — GemmOp (op_slug=`gemm`, family=`gemm`)

> Append-only. Stage 5 integration retrospective is op-level (integration issues do not
> belong to a single function). Per-function Stage 1/3 retrospectives live under
> `examples/gemm/_gemm_kernel/RETROSPECTIVE.md` and `examples/gemm/_gemv_kernel/RETROSPECTIVE.md`.

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-15

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `S5-ASCEND-MODE` in `.agents/tools/gate_lint.py::integration_lint` requires **exactly one** direct `Kernel` subclass in the wrapper (`len(classes) != 1` -> fail). A migration unit that owns two kernel functions dispatched by one Op (GEMM + GEMV) naturally ports two `Kernel` classes (Stage 0 scaffold `gemm.py`: `GemmKernel`, `GemvKernel`), so the gate rejects a structurally valid wrapper. Worked around with a shared private base `_GemmFamilyKernelBase(Kernel)` carrying `ascend_mode="Developer"`. | `.agents/tools/gate_lint.py#L1475-L1507`; fixture `.agents/tools/tests/test_statectl.py#L1395-L1401` (two functions, one `MopKernel`); gate output before fix (S5-ASCEND-MODE failure) | Relax `integration_lint` to accept N direct `Kernel` subclasses when each declares a mode equal to the single integrated source mode; or document the shared-base workaround in the add-npu-op skill. | R |
| Integrate-flow | Stage 0 scaffold + `.migration_meta.json` carry the **GPU nested-factory** call convention into `forward`/`default_config`, but the Stage 3 npuir factory is **flat** and `@tilelang.jit`-decorated. Every migration needs a wrapper-glue mapping; the skill does not state it explicitly. | `examples/TileOPs/tileops/kernels/gemm/gemm/gemm.py#GemmKernel.forward` (pre-fix `factory(...)(**config)(a,b)`); first report failure `TypeError: JitKernel_NPU.__call__() got an unexpected keyword argument 'block_m'`; same finding in `tileops/kernels/bmm/bmm/bmm_kernel/integration_log.md` | Add to add-npu-op skill: "integrated npuir factory is flat `factory(shape..., block_M=, block_N=, block_K=, num_stages=, dtype=)(a,b)`; the wrapper call-site must be remapped from the GPU nested shape." | R |
| Integrate-flow | GPU `GemvKernel.default_config` keys (`block_n`, `reduce_threads`) have no npuir counterpart (`block_M`, `block_K`); also the GPU callable is `(x, W)` while the npuir kernel is `(W, x)`. Silent unless the GEMV path is exercised. | `gemm.py#GemvKernel.default_config` + `#GemvKernel.forward` (pre-fix `self.kernel(block_n, reduce_threads, num_stages)(a,b)`, second report error `ValueError: Expected 2 inputs, got 3`) | Document the GEMV operand reversal + key remap in the skill's GPU->NPU table. | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| Wrapper glue mapping for a flat `@tilelang.jit(target="npuir")` factory (GEMM/GEMV): map lower-case config keys to factory kwargs and pass block sizes at the **factory** call, not the returned callable; store the factory handle in `__init__` and defer compilation to `Kernel.__call__`. | P | `examples/TileOPs/tileops/kernels/gemm/gemm/gemm.py#GemmKernel.forward`; report `20261015_164502_081440_GemmOp/run.json` correctness 17/17 | `ASCEND_RT_VISIBLE_DEVICES=1 python3 -m tileops.reporting.cli run --op GemmOp --prof-mode msprof` (from `examples/TileOPs/`) | tilelang 0.1.1.10+ubuntu.20.4.cann900; CANN 9.0.0; driver 25.5.5; Ascend910_9362; commit a919404a | pattern-library / add-npu-op SKILL |
| GEMV npuir kernel is `y = W @ x`, signature `(W[R,K], x[K])`; the Op dispatches `kernel(x_1d, W)`, so `GemvKernel.forward` must reverse operands (`_gemv_kernel(...)(b, a)`). | P | `gemm.py#GemvKernel.forward`; `tests/ops/test_gemm.py#test_gemv_boundary_lhs_row/rhs_col` all pass | as above (boundary cases 3000x1024, 1024x3000, 3001x1024) | same | pattern-library |
| Cube-op msprof capture (kernel `main`) works for the flat-factory GEMM/GEMV kernels with `msprof op` — no events fallback needed (contrast ClampMax). Evidence: 14/14 tileops cases `prof_mode=msprof`, `invalid_profile_count=0`. | D | `reports/tileops/20261015_164502_081440_GemmOp/run.json` `summary.msprof_case_count=14`, `profiler_fallback_count=0`; `report.md` every row `Mode=msprof`, `Kernel=main` | as above | same | bottleneck-patterns |

### Transferable Lessons（可迁移教训）

- 集成 Stage 3 npuir kernel 时，先确认工厂是 **扁平签名且已 `@tilelang.jit`**：tiling 参数（`block_M/block_N/block_K/num_stages`）在**工厂调用**时传入，返回的才是 `(a, b) -> c` 可调用对象；Stage 0 wrapper 的 GPU 嵌套形态 `factory(shape)(tiling)(a,b)` 必须改成扁平形态，否则报 `unexpected keyword argument 'block_m'`。
- `Kernel.__init__` 里**不要**直接调用扁平工厂（会立即编译且绕过 `Kernel.__call__` 的 `scoped_ascend_mode`）；应存工厂句柄（`self.kernel = _xxx_kernel`），把编译留给 `forward`。
- npuir GEMV 是 `y = W @ x`，与 GPU `(x, W)` 的调用约定**相反**；wrapper 胶水层需反转入参（`(b, a)`），并重映射 config 键（`block_n->block_M`, `reduce_threads->block_K`）。
- 若一个 Op 合法拥有多个 `Kernel` 子类，而 Stage 5 机械门禁 `S5-ASCEND-MODE` 要求 wrapper 恰好一个直接 `Kernel` 子类：引入共享私有基类（`class _XBase(Kernel)` 声明 `ascend_mode`），令各具体 kernel 继承它——门禁只统计直接继承 `Kernel` 的类，功能不变且两 kernel 仍可通过 `Kernel.__call__` 作用域编译。
- 任何 wrapper/集成 kernel 文件在 report 之后被修改，都会触发 `S5-REPORT-STALE`（报告早于集成文件）；修改顺序必须是「改文件 -> 重跑 report -> 写 integration_report.json」，报告 mtime 必须最新。
