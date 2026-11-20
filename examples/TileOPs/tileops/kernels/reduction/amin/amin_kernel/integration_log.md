# Integration Log -- AminFwdOp

- meta: examples/TileOPs/tileops/kernels/reduction/amin/.migration_meta.json
- integrated (op_slug=amin, family=reduction, kernel_class=ReduceKernel, op_class=AminFwdOp):
  - `_simple_reduce_kernel` -> `tileops/kernels/reduction/amin/amin_kernel/_simple_reduce_kernel.py`
  - `_simple_reduce_kernel_tiled` -> `tileops/kernels/reduction/amin/amin_kernel/_simple_reduce_kernel_tiled.py`
  - aggregation `tileops/kernels/reduction/amin/amin_kernel/__init__.py` (re-exports the two builders)
  - wrapper `tileops/kernels/reduction/amin/amin.py` (baseline import active -> `.amin_kernel`; `ascend_mode = "Developer"`; custom_op names namespaced to `npub::amin_reduce_*`)
- design_docs:
  - `_simple_reduce_kernel` -> `tileops/kernels/reduction/amin/amin_kernel/_simple_reduce_kernel_DESIGN.md`
  - `_simple_reduce_kernel_tiled` -> `tileops/kernels/reduction/amin/amin_kernel/_simple_reduce_kernel_tiled_DESIGN.md`
- ascend_mode: Developer (from Stage 3 kernel `ASCEND_MODE = "Developer"`; `function_modes` both Developer, `mixed_modes=false`; wrapper `ascend_mode` unchanged/consistent)
- attempts: 1

## 集成步骤

1. `integrate_kernel.py --meta ...` 幂等运行成功（`[ok] integration complete`，smoke=pass）。脚本从 Stage 3 kernel 读取的 `ASCEND_MODE=Developer` 与 wrapper `Kernel.ascend_mode` 一致，无需改写。
2. custom_op 命名空间化：wrapper 4 个 `torch.library.custom_op` 均已按 op_slug 唯一化
   （`npub::amin_reduce_simple_fwd` / `npub::amin_reduce_simple_tiled_fwd` / `npub::amin_reduce_welford_fwd` / `npub::amin_reduce_welford_tiled_fwd`），
   与同族 `amax.py` 的 `npub::amax_reduce_*` 并存无冲突。验证：
   `import tileops.kernels.reduction.{amax,amin}` 后两模块的 custom_op 名集合互不相交，`ReduceKernel` 类对象互异
   （此前未命名空间化时 amin 会劫持 amax 的 dispatch）。
3. 单算子 report（`--prof-mode msprof`）全量运行：正确性 14/14 通过后自动跑 benchmark。

## 验证结果

- report_run_json: examples/TileOPs/reports/tileops/20261015_140605_045273_AminFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261015_140605_045273_AminFwdOp/report.md
- report_status: partial（correctness 通过、benchmark 已运行；1 个 benchmark 用例失败 → CLI 退出码非零）
- correctness: pass（14/14，`summary.correctness_passed=true`，max_abs_err=0.00e+00）
- benchmark: partial（7 用例，6 通过，1 失败）——见下 performance 段
- 退出码：correctness=0，benchmark=1（由 long-seq-reduce-bfloat16 失败引起）

### workload 明细（msprof）

| label | dtype | shape | latency_us | Ratio(%) | Bandwidth(TB/s) | 状态 |
|---|---|---|---|---|---|---|
| smoke-amin | float16 | [32, 256] | 1.58 | 0.6018 | 1.75 | pass |
| smoke-amin | bfloat16 | [32, 256] | 2.00 | 0.4827 | 1.75 | pass |
| smoke-amin | float32 | [32, 256] | 1.50 | 1.2581 | 1.75 | pass |
| hidden-state-reduce | float16 | [2048, 4096] | 15.00 | 63.9137 | 1.75 | pass |
| hidden-state-reduce | bfloat16 | [2048, 4096] | 19.56 | 49.0136 | 1.75 | pass |
| 3d-multidim-reduce | float16 | [4, 128, 4096] | 7.76 | 30.8868 | 1.75 | pass |
| long-seq-reduce | bfloat16 | (large N seq) | - | - | - | **failed** |

- long-seq-reduce 失败原因（**已知 benchmark-only，不修**）：
  `AssertionError: tile_n=16384 x B=6 exceeds the x2 UB envelope; lower tile_n`。
  根因：tiled 路径 `tile_n` 预算口径在 wrapper 与迁移 kernel 间不一致——wrapper 的
  `_primitives.compute_tile_n` 按 `element_size`（bf16=2B）估值，而迁移 tiled kernel 按 B=6
  （x_ub 2B + fp32 work 4B）预算并在 `2*tile_n*b<=184KB` 断言；N 很大时 wrapper 选出的
  `tile_n=16384` 被 kernel 断言拒绝（amax 的 `long-seq-reduce` 同症状，属系统性问题）。
  按 Stage 5 策略：benchmark 失败只记录不修复。

## 调试历史

- attempt 1: 无 correctness 失败，未进入调试闭环。`integrate_kernel.py` 首跑即通过（幂等复用既有集成状态），
  report correctness 14/14 通过。唯一 benchmark 失败为已知 wrapper/kernel `tile_n` 预算口径不一致（不修）。

## issues

- 见上「long-seq-reduce」：wrapper/kernel `tile_n` 预算口径不一致（与 amax 同根因，交回 conductor 决策，不属本 Stage 可修改范围）。
- 无测试/用例/manifest 层问题；无前置集成脚本报错；无设计层错误。
