## Stage 5 (integrate) Retrospective — attempt 5.1 — 2026-10-16

Task: migration-harness, project=`any`, op=`AnyFwdOp`（唯一提取函数 `_logical_reduce_kernel`；GPU TileOPs
`AnyFwdOp` / `torch.any` 语义）。op 级集成复盘（单份）。
工具链戳：tilelang `0.1.2+7fe5a890cbba3784f27ea742af219984eda202ba`（dev root build）+ CANN 9.0.0 +
torch 2.9.0 + torch_npu 2.9.0.post8 + Ascend910B3 x1。

结论：`INTEGRATE_COMPLETED`（attempt 1 一次通过；correctness 42/42，`avg max abs error = 0.00e+00`；
benchmark 6/6，3 个 msprof 用例 `profiler_valid=true`）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py::rewrite_wrapper_import` 只替换 wrapper 内 `from .{extracted_module} import ...` 这一行，再注入自带表头的 baseline/perf_opt 块；当 Stage 0 scaffolder **已预置**整段选择块模板（本任务 `any.py` 如此，晚于 `count_nonzero`/`all` 的纯 import 模板）时，重写会产生**重复表头 + 重复 perf_opt 占位注释**（功能无害：仍只有一行激活的 baseline import） | 备份 `examples/TileOPs/tileops/kernels/reduction/any/any_kernel/history_version/any_wrapper_s5_preseed_dup.py`（改前）；`all/` 集成为单表头、`any/` 改前为双表头；改后 `any.py` 与 `all.py` 结构一致 | 让 `rewrite_wrapper_import` 先探测并吞掉 scaffolder 预置的选择块（或让 scaffolder 不再预置表头、统一由 integrator 生成），使脚本对预置 wrapper 幂等且无重复 | R |
| Info-source | 集成脚本的 import 冒烟以 `sys.executable` 运行；默认 `python3`/`python` 缺 `torch` / `typing_extensions`，且未设 `PYTHONPATH` 时 `tilelang` 不可导入，脚本以 `[error] import smoke test FAILED` 退出——易被误判为 kernel/集成缺陷 | 本任务改前若用系统解释器即报 `ModuleNotFoundError`；`all_kernel/integration_log.md` 也记录了同类现象；改用 `/workspace/venv/bin/python` + `PYTHONPATH=/workspace/tilelang-mlir-ascend` 后 `smoke-ok` | 在 `add-npu-op` skill 的集成步骤里固化「解释器 = venv 全依赖 + `PYTHONPATH` 含 repo root」的前置，或让脚本把该失败归类为环境错误 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| TileOPs-harness 集成的**解释器/路径前置**是硬门槛：`integrate_kernel.py` 冒烟与 `tileops.reporting.cli` 均须以 `/workspace/venv/bin/python`（3.10.12，含 torch/torch_npu）+ `PYTHONPATH=<repo-root>`（提供 dev-build `tilelang`）在 `examples/TileOPs/` 下运行；缺一即 `ModuleNotFoundError: No module named 'tilelang'` 假失败（与 kernel 质量无关） | D | `examples/TileOPs/tileops/kernels/reduction/any/any_kernel/integration_log.md#集成步骤`；`examples/TileOPs/tileops/kernels/reduction/all/all_kernel/integration_log.md` | `cd examples/TileOPs && PYTHONPATH=/workspace/tilelang-mlir-ascend /workspace/venv/bin/python -m tileops.reporting.cli run --op AnyFwdOp --prof-mode msprof` | tilelang 0.1.2+7fe5a890 + CANN 9.0.0 + Ascend910B3 x1 | `.agents/skills/add-npu-op/SKILL.md`（集成/报告运行前置）|
| 预置了 baseline/perf_opt 选择块的 wrapper 会让 `integrate_kernel.py` 的 `rewrite_wrapper_import` 产生重复表头（非幂等）——scaffolder 模板演进（`all`/`count_nonzero` 纯 import vs `any` 预置块）是该回归的触发条件 | R | 备份 `.../any_kernel/history_version/any_wrapper_s5_preseed_dup.py`；对照 `tileops/kernels/reduction/all/all.py`（单表头） | `diff` 集成前后 `any.py`（或重跑 `integrate_kernel.py` 观察第二次是否再注入） | 同上 | `.agents/skills/add-npu-op/scripts/integrate_kernel.py`（`rewrite_wrapper_import` 幂等性）|

### Transferable Lessons（可迁移教训）

- TileOPs-harness 集成与 report **必须在 `examples/TileOPs/` 下、用全依赖解释器
  （`/workspace/venv/bin/python`）、并把 repo root 加进 `PYTHONPATH`** 运行；这三条缺一都会以
  `ModuleNotFoundError: No module named 'tilelang'`（冒烟假失败）或 `typing_extensions` 缺失报错出现，
  **属环境/路径问题，不要据此改 kernel**。
- 迁移集成的 kernel 侧 UB 预算回落（`_resolve_block_m` / `_select_config` 硬上限）可让 GPU 期
  SMEM 启发式 `default_config` 零改动地跑通 report（本任务 `(32,4096)` 给 `block_m=8` 未越 UB），
  因此**集成期通常无需改 wrapper config**；仅在实测编译期 UB overflow 时才进受控闭环。
- `integrate_kernel.py` 的 `rewrite_wrapper_import` 对「scaffolder 已预置选择块模板的 wrapper」不幂等，
  会留下重复表头（功能无害）；集成后应目视核对 wrapper 只保留一段选择块，必要时备份后做外观清理
  （备份进 `{op}_kernel/history_version/`，不计为调试 attempt）。
- 谓词类 `any`（bool 输出 + 逻辑 OR，无浮点累加）在 harness 正确性判据下应恒为**逐位精确**
  （`avg max abs error = 0.00e+00`）；report 若出现非零 abs error 即判定为异常而非容差问题。
