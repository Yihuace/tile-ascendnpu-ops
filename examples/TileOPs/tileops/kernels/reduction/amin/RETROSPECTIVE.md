# AminFwdOp Migration Retrospective (op-level)

> op_slug: amin | migration unit: `_simple_reduce_kernel`, `_simple_reduce_kernel_tiled` (Vector reduction 归约)
> 本文件为 op 级追加式复盘；函数级复盘见 `examples/amin/{func}/RETROSPECTIVE.md`。

## Stage 5 (integrate) Retrospective — first_integrate — 2026-10-15

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | 迁移脚手架未对**共用同一个 Kernel 类**的同族 wrapper 做 custom_op 名去重：amax/amin 复用 `ReduceKernel`，各自注册同名 `npub::reduce_simple_fwd` 等，import 顺序决定 dispatch 归属（本次为第 2 次独立出现，amax Stage 5 已立项） | `examples/TileOPs/tileops/kernels/reduction/amin/amin.py`（原 4 处 `npub::reduce_*`，本次改为 `npub::amin_reduce_*`）；`.../reduction/amax/amax.py`（已改为 `npub::amax_reduce_*`）；`examples/amax/RETROSPECTIVE.md` Stage 5 R 项 | 在 `add-npu-op` 迁移 skill 文档化「同族共用 Kernel 类的 wrapper 的 `torch.library.custom_op` 名须按 op_slug 命名空间化」（argmax/logsumexp 已为范例），或脚手架生成时自动加 `<slug>_` 前缀 | R |
| Integrate-flow | tiled 路径 `tile_n` 预算口径在 wrapper 与迁移 kernel 间不一致：wrapper `_primitives.compute_tile_n` 按 element_size（bf16=2B）估值，迁移 tiled kernel 按 B=6（x_ub 2B + fp32 work 4B）预算并断言 `2*tile_n*b<=184KB` → 大 N 时 wrapper 选出的 `tile_n=16384` 被 kernel 断言拒绝（本次为第 2 次独立出现，amax 同症状） | `.../amin/amin_kernel/_simple_reduce_kernel_tiled.py`（B=6 预算 + 断言）；`.../amin/amin.py::_tile_n_for_block_m`；`.../reports/tileops/20261015_140605_045273_AminFwdOp/benchmark.json`（long-seq-reduce bf16 报 `tile_n=16384 x B=6 exceeds the x2 UB envelope`）；`examples/amax/RETROSPECTIVE.md` Stage 5 D 项 | 在 `add-npu-op` 集成核对清单加入「大 N tiled 路径 wrapper 的 tile_n 必须走迁移 kernel 自己的 `select_tile_n`，不得沿用共享 `_primitives.compute_tile_n`」 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 同族算子（共用 Kernel 类）每个 wrapper 的 `torch.library.custom_op` 名必须按 op_slug 唯一化——amin 侧独立确认：命名空间化后 `npub::amin_reduce_*` 与 `npub::amax_reduce_*` 并存无冲突（两模块 custom_op 名集合互不相交、`ReduceKernel` 类对象互异）；未命名空间化时 amin 会劫持 amax 的 dispatch | P | `examples/TileOPs/tileops/kernels/reduction/amin/amin.py`（4 处 `npub::amin_reduce_*`）；验证脚本见 `.../amin/amin_kernel/integration_log.md` 集成步骤 2；`examples/TileOPs/tileops/kernels/reduction/{amax,amin}` | `python -c "import tileops.kernels.reduction.amax.amax as A, tileops.kernels.reduction.amin.amin as B; ..."`（打印两模块 custom_op 名集合，验证不相交）；或 `python -m pytest tests/ops/test_amax.py -q`（amin 未命名空间化时 amax 用例被 amin 签名劫持） | tilelang 0.1.4+ubuntu.20.4.cann852 dev root build + CANN 8.5.2 + Ascend910B3 + torch 2.7.1+cpu, 2026-10-15, repo commit 2238099 | `add-npu-op` 迁移 skill（custom_op 命名空间化）/ `pattern-library/traps-*.md` |
| tiled `tile_n` 预算口径 wrapper vs kernel 不一致触发 kernel 断言——amin 独立复现：N 很大 bf16 long-seq-reduce 报 `tile_n=16384 x B=6 exceeds the x2 UB envelope`（同 amax），证明该口径不一致是**系统性**缺陷而非单算子偶发；wrapper 选 16384 而 kernel 自带 `select_tile_n`（B=6）会选 8192 | D | `examples/TileOPs/reports/tileops/20261015_140605_045273_AminFwdOp/run.json`（benchmark_tests 1 failed + benchmark.md）；`.../amin/amin_kernel/_simple_reduce_kernel_tiled.py`（B=6 预算/断言） | `python -m tileops.reporting.cli run --op AminFwdOp --prof-mode msprof`（long-seq-reduce bf16 用例 error） | 同上 | `pattern-library/bottleneck-patterns` 或 `add-npu-op` 集成核对清单 |

### Transferable Lessons（可迁移教训）

- **同族算子共用同一个 Kernel 基类时，每个 wrapper 的 `torch.library.custom_op` 名必须按 `op_slug` 唯一化**：同名 `npub::...` 注册按 import 顺序相互覆盖，症状是「运行 A 算子测试却报 B 算子 kernel 的签名错误」。验证方法是 import 两个 wrapper 后比较 custom_op 名集合是否相交。
- **大 N 走 tiled 路径时，wrapper 的 `tile_n` 必须由被迁移 tiled kernel 自己的预算函数（如 `select_tile_n`）决定**，不能沿用共享的 `_primitives.compute_tile_n`：后者按 `element_size` 估值，而迁移 kernel 的 UB 预算含 fp32 工作缓冲（bf16 实际 6B/元素），会在 N 很大时选出被 kernel 断言拒绝的 `tile_n`。这是同族通病（amax、amin 均复现）。
- **`status=partial` 不等于集成失败**：本任务 correctness 14/14 通过、benchmark 仅 1 个 tiled 用例因已知 `tile_n` 口径失败；按 Stage 5 策略 benchmark 失败只记录不修复，故判定 `INTEGRATE_COMPLETED`（附 partial 告警），并把 `tile_n` 口径不一致作为 issue 交回 conductor。
- **集成脚本幂等性可安全复用既有集成状态**：`integrate_kernel.py` 重跑会整文件覆盖 kernel/DESIGN 快照，但保留已改写（含 custom_op 命名空间化）的 wrapper 与 `ascend_mode`；因此 custom_op 改名不会被幂等重跑回退。
