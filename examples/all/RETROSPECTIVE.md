# RETROSPECTIVE — AllFwdOp (op_slug=all)

> Op-level retrospective for the migration-harness pipeline. Stage 1/2/3 per-function
> retrospectives live under `examples/all/_logical_reduce_kernel/RETROSPECTIVE.md`.

## Stage 5 (integrate) Retrospective — first_impl — 2026-10-16

> 结论 = **`INTEGRATE_COMPLETED`**（attempt 1，无调试闭环）。集成脚本一次通过 import 冒烟；
> 单算子 TileOPs report `status=passed`：正确性 42/42 全过、benchmark 6/6 全过、
> 3 个 msprof 用例 `profiler_valid=true`（`invalid_profile_count=0`）。
> 唯一失败是首跑集成脚本冒烟期的 **`ModuleNotFoundError: No module named 'tilelang'`**（`PYTHONPATH` 未含仓库根），
> 设 `PYTHONPATH=/workspace/tilelang-mlir-ascend` 后重跑即通过，属环境路径而非代码缺陷，未计入 attempt。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` 的 import 冒烟（`smoke_test()`，`subprocess.run([sys.executable, "-c", code], cwd=tileops_root)`）**不设置 `PYTHONPATH`**：tilelang 由仓库根 dev-root 提供（未安装进 site-packages），从 `examples/TileOPs/`（tileops_root）运行时会 `ModuleNotFoundError: No module named 'tilelang'` 并被误报为 `[error] import smoke test FAILED`；先跑失败后才发现需补 `PYTHONPATH=<repo root>` 重跑 | 首跑日志：`File ".../tileops/kernels/reduction/_primitives.py", line 16, in <module> import tilelang.language as T` → `ModuleNotFoundError: No module named 'tilelang'` → `[error] import smoke test FAILED`；`integrate_kernel.py#smoke_test`（未传 env）；补 `PYTHONPATH=/workspace/tilelang-mlir-ascend` 后 `[smoke] smoke-ok ['_logical_reduce_kernel']` | `integrate_kernel.py` 的 `smoke_test()` 用 `env={**os.environ, "PYTHONPATH": f"{find_tileops_root().parents[?]}{os.pathsep}{...}"}` 或在文档/agent prompt 明示「集成与 report 均在 `examples/TileOPs/` 下、`PYTHONPATH` 须含仓库根」 | R |
| Integrate-flow | 集成脚本把 conductor 产物**按原名**拷入 `{op_slug}_kernel/`（本函数名 `_logical_reduce_kernel.py`），而 Stage 0 脚手架的 extracted 文件名为 `_all_fwd_kernels.py`；wrapper import 改写为 `.all_kernel` 后，旧 `all/_all_fwd_kernels.py` 成为**孤儿文件**（不再被 import），脚本不清理也不提示 | `all_kernel/_logical_reduce_kernel.py`（集成产物）vs 残留 `all/_all_fwd_kernels.py`；`all.py` 内 `from .all_kernel import _logical_reduce_kernel`（原 `from ._all_fwd_kernels import ...`） | `integrate_kernel.py` 在改写 wrapper 后打印一条 `[warn] stale extracted module <path> no longer imported`（或说明「脚手架 extracted 文件可保留为参考」），避免集成后目录出现同名语义的两个 kernel 源 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| AllFwdOp（bool 输出 / raw `(M,N)` / `_kernel_handles_padding=True` / `_empty_dim_policy="noop"`）端到端集成**一次通过**：TileOPs report `status=passed`，正确性 42/42（含 bool/int32/int64/complex64/complex128/非整除 N/尾 M/1D-4D/多 dim/keepdim），benchmark 6/6、3 个 msprof 用例全部 `profiler_valid=true`，roofline 利用率 10.48%–43.98%、带宽 1.75 TB/s——证明 baseline 工厂 + per-op 包名空间 `npub::all_logical_reduce_fwd` 与 count_nonzero 的 `npub::logical_reduce_fwd` 无冲突 | D | `reports/tileops/20261016_033406_263435_AllFwdOp/run.json`（`summary.correctness_passed=true`/`benchmark_passed=true`/`invalid_profile_count=0`）；`report.md` 明细表 | `cd examples/TileOPs && PYTHONPATH=/workspace/tilelang-mlir-ascend python -m tileops.reporting.cli run --op AllFwdOp --prof-mode msprof` | tilelang `1bc8dbc3a1c80345a12522d815a2268e555270fa`（dev root）+ CANN 9.0.0 + Ascend910B3（torch 2.9.0 / torch_npu 2.9.0.post8） | `pattern-library/`（迁移集成通过形态；及 `npub::` per-op 命名空间防冲突实践） |
| TileOPs report / 集成冒烟**必须在 `examples/TileOPs/` 下且 `PYTHONPATH` 含仓库根**：该 venv 的 tileops 为 editable 指向 `examples/TileOPs`，tilelang 未安装（由仓库根 dev-root 提供），从仓库根直接跑模块命令或因缺 `PYTHONPATH` 跑子进程冒烟都会 `ModuleNotFoundError: No module named 'tilelang'` | D | 首跑 `integrate_kernel.py` 冒烟 `ModuleNotFoundError`；补 `PYTHONPATH=/workspace/tilelang-mlir-ascend` 后 `[smoke] smoke-ok`，report CLI exit 0 | （反例）`cd examples/TileOPs && /workspace/venv/bin/python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/reduction/all/.migration_meta.json`（无 PYTHONPATH → 冒烟失败） | 同上 | `add-npu-op` skill / integrator agent（运行前置：cwd 与 PYTHONPATH） |

### Transferable Lessons（可迁移教训）

- **集成与 report 的运行前置**：cwd = `examples/TileOPs/`，且 `PYTHONPATH` 含仓库根
  （`/workspace/tilelang-mlir-ascend`，提供 `tilelang` dev-root 包）。缺 `PYTHONPATH` 时集成脚本的
  import 冒烟会以 `ModuleNotFoundError: No module named 'tilelang'` 报 `[error] import smoke test FAILED`
  ——这是环境路径问题，不是 kernel/集成缺陷，补 `PYTHONPATH` 重跑即可，**不要**据此改 kernel。
- **集成不改 kernel 代码**：`integrate_kernel.py` 按原名整文件覆盖拷贝 conductor 产物到
  `{op_slug}_kernel/`，并把 wrapper 改写为 baseline/perf_opt 双 import 块（baseline 默认激活）。
  脚本不触碰共享文件（`ops/reduction/__init__.py`、`kernels/reduction/__init__.py`、manifest、workloads、
  `_primitives.py`）——这些由 Stage 0 脚手架预先写好本算子条目，集成期无需再改。
- **脚手架残留文件属正常**：Stage 0 的 extracted 文件（如 `all/_all_fwd_kernels.py`）在集成后不再被 import
  （wrapper 改为 `from .all_kernel import ...`），成为孤儿参考文件，**不要**手工删改或把 wrapper 指回它。
- **GPU 期 wrapper 启发式配置可在集成期直接通过**：`LogicalReduceKernel.default_config` 仍是 GPU 期
  SMEM 启发式（`(256,4096)` 下给 `block_m` 偏大），kernel 侧 `_resolve_block_m` 按 UB 元素预算自动回落，
  因此集成无需改 wrapper config（本算子 32k 用例实际采用 `block_m=1`、3D 用例 `block_m=2`）。
- **bool 输出算子的正确性判据**：`run.json` 的 `avg max abs error = 0.00e+00` 与测试内 `torch.equal`
  精确比较一致（logical AND 无浮点误差），集成期看到非零 abs error 才是异常。
