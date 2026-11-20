# AmaxFwdOp Migration Retrospective (op-level)

> op_slug: amax | migration unit: `_simple_reduce_kernel`, `_simple_reduce_kernel_tiled`
> 本文件为 op 级追加式复盘；函数级复盘见 `examples/amax/{func}/RETROSPECTIVE.md`。

## Stage 5 (integrate) Retrospective — first_integrate — 2026-10-15

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py::parse_wrapper_imports` 遍历整个 wrapper AST，把**同模块的全部** `from ._{slug}_fwd_kernels import ...` 名字（含 `try/except` 的**可选** Welford 导入，属 std/var、非本迁移单元）都收集起来；`rewrite_wrapper_import` 又把它们写进必需的 baseline `from .{slug}_kernel import (...)`，而聚合 `__init__.py` 只 re-export 已迁移的两个 builder → 集成后 `ImportError`。随后 smoke 又断言「所有收集名均为包导出」→ 二次误报 | 首跑集成输出 `ImportError: cannot import name '_welford_reduce_kernel' from ...amax_kernel`；`examples/TileOPs/tileops/kernels/reduction/amax/amax.py`（集成生成的 baseline 块）；`examples/TileOPs/tileops/kernels/reduction/amax/amax_kernel/__init__.py`；`.agents/skills/add-npu-op/scripts/integrate_kernel.py` L208–217（parse）/L367–391（rewrite）/L394–418（smoke） | 在 `integrate_kernel.py` 中：parse/emit 只保留 `integrated_files` 里实际定义的名称（或仅替换非 `try` 的必需 import）；smoke 只断言已集成名，不校验可选导入名 | R |
| Integrate-flow | `gen_init_py` 的 `all_names.append(name)` 在「未找到定义」分支外无条件执行，导致生成的 `amax_kernel/__init__.py` 的 `__all__` 含两个从未导入的 Welford 名（本次无害，但使 `__all__` 与实际定义不一致，`from ... import *` 会失败） | `examples/TileOPs/tileops/kernels/reduction/amax/amax_kernel/__init__.py`（`__all__` 列 4 名、仅导入 2 名）；`integrate_kernel.py` L280–282 | `gen_init_py` 仅在成功 re-export 时把名字加入 `all_names` | R |
| Integrate-flow | 迁移脚手架/脚本未对**共用一个 Kernel 类**的同族 wrapper 做 custom_op 名去重：amax、amin 复用 `ReduceKernel`，各自注册同名 `npub::reduce_simple_fwd` 等，import 顺序决定 dispatch 归属 | `examples/TileOPs/tileops/kernels/reduction/amax/amax.py`（原 L96 等 4 处）；`.../amin/amin.py` L83/100/118/147；`.../reduction/__init__.py`（先 amax 后 amin）；污染证据见 correctness.log 调用栈落到 `amin/amin.py:92` | 在 `add-npu-op` 迁移 skill 文档化「同族 wrapper 的 `torch.library.custom_op` 名须按 `op_slug` 命名空间化（argmax=`npub::argreduce_fwd`、logsumexp=`npub::logsumexp_fwd` 已为例）」，或由脚手架生成时自动加 `<slug>_` 前缀 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 同族算子（共用 Kernel 类）每个 wrapper 的 `torch.library.custom_op` 名必须按 op_slug 唯一化：amax/amin 同时注册 `npub::reduce_simple_fwd`，先 import 的 amax 被后 import 的 amin 劫持 dispatch，症状是 amax 测试抛出与之无关的 `TypeError: _func() missing 1 required positional argument: 'threads'`（amin 的未迁移 kernel 签名） | P | `examples/TileOPs/tileops/kernels/reduction/amax/amax.py`（改名 `npub::amax_reduce_*` 后 14/14 通过）；`.../amax/amax_kernel/integration_log.md` 调试历史 attempt 2；correctness.log 调用栈 `amin/amin.py:92` | `python -m pytest tests/ops/test_amax.py -q`（改名前后：0/14 → 14/14）；或 `python -c "import tileops.kernels.reduction"` 后 amax 用例报告 amin 签名错误 | tilelang 0.1.4+ubuntu.20.4.cann852 dev root build + CANN 8.5.2 + Ascend910B3 + torch 2.7.1+cpu, 2026-10-15, repo commit 2238099 | `tilelang-op-optimize/references/pattern-library/traps-*.md` 或 `add-npu-op` 迁移 skill（custom_op 命名空间化） |
| tiled 路径 `tile_n` 预算口径在 wrapper 与迁移 kernel 间不一致即触发 kernel 断言：wrapper `ReduceKernel._tile_n_for_block_m` 用共享 `_primitives.compute_tile_n`（按 `element_size`，bf16=2B），迁移 tiled kernel 用 UB 预算（bf16 B=6=2B x_ub+4B fp32 work）并在 `2*tile_n*b<=184KB` 断言；N=32768 bf16 时 wrapper 选 `tile_n=16384` → 断言失败（而 kernel 自己的 `select_tile_n` 会选 8192） | D | `examples/TileOPs/tileops/kernels/reduction/amax/amax_kernel/_simple_reduce_kernel_tiled.py` L94（`select_tile_n`，B=6）/L154（断言）；`.../amax/amax.py::_tile_n_for_block_m`（`_primitives`）；`.../reports/tileops/20261015_135122_905437_AmaxFwdOp/benchmark.json`（`long-seq-reduce` bf16 报 `tile_n=16384 x B=6 exceeds the x2 UB envelope`） | `python -m tileops.reporting.cli run --op AmaxFwdOp --prof-mode msprof`（long-seq-reduce (64,32768) bf16 用例 error） | 同上 | `pattern-library/bottleneck-patterns` 或 `add-npu-op` 集成核对清单（wrapper tile_n 须走迁移 kernel 的 `select_tile_n`） |

### Transferable Lessons（可迁移教训）

- **集成含「可选/旁路导入」的 wrapper 时，集成脚本会把同底层模块的全部导入名当作必须导出**：修法是把 baseline 必需导入收敛到本迁移单元的函数，其余（如 std/var 的 Welford builder）保持 `try/except` 可选，并让其从集成包（而非被替换的参考模块）解析；否则脚本首跑即 `ImportError` + smoke `missing exports` 误报。
- **同族算子共用同一个 Kernel 基类时，每个 wrapper 的 `torch.library.custom_op` 名必须按 `op_slug` 唯一化**；否则同名的 `npub::...` 注册按 import 顺序相互覆盖，表现为「运行 A 算子测试却报 B 算子 kernel 的签名错误」（本任务 amax 测试报 amin 未迁移 kernel 缺 `threads`）。排查时看 pytest traceback 的**调用文件**是否落在别的算子 wrapper。
- **大 N 走 tiled 路径时，wrapper 的 `tile_n` 必须由被迁移 tiled kernel 自己的预算函数（如 `select_tile_n`）决定，不能沿用共享 `_primitives.compute_tile_n`**：后者按 `element_size`（bf16=2B）估值，而迁移 kernel 的 UB 预算含 fp32 工作缓冲（bf16 实际 6B/元素），会在 N>32512 时选出被 kernel 断言拒绝的 tile_n。
- **`status=partial` 不等于集成失败**：本任务 correctness 14/14 通过、benchmark 4 个 workload 中 3 个有效，仅 tiled 路径 benchmark 失败；按 Stage 5 策略 benchmark 失败只记录不修复，故判定 `INTEGRATE_COMPLETED`（附 partial 告警）并把 tiled/wrapper 口径不一致作为 issue 交回 conductor。
