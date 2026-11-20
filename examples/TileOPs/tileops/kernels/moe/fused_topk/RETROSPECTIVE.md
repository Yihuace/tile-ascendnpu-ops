# RETROSPECTIVE — fused_topk (migration-harness, op level)

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` echoes the **last line** of the smoke subprocess output as `[smoke] ...`; when torch emits a benign trailing warning (`Permission mismatch: The owner of {path} does not match.`), that warning is printed instead of the `smoke-ok` marker, which can read as a smoke failure even though `returncode==0` and `"smoke": "pass"`. | integration run output (`[smoke]   warnings.warn(f"Permission mismatch: ...")` vs `[json] {... "smoke": "pass"}`); `integrate_kernel.py` `smoke_test()` returns `(proc.returncode==0, stdout+stderr)` and `main()` prints `out.splitlines()[-1]`. | In `integrate_kernel.py::main`, echo the line containing `smoke-ok` (or the return-code status) instead of `out.splitlines()[-1]`. | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| none | - | - | - | - | - |

### Transferable Lessons（可迁移教训）

- migration-harness Stage 5 的 import smoke 会执行被集成 kernel 模块的**顶层导入**（本算子：`import torch_npu`、`import tilelang`），因此 smoke 通过的前提是 NPU 运行时/CANN 环境**可导入**，不仅是设备可见；聚合 `__init__.py` 在 import 期即加载各 kernel 模块。
- `integrate_kernel.py` 自动从 Stage 3 产物的模块级 `ASCEND_MODE` 字面量同步 wrapper 的 `Kernel.ascend_mode`（本例 `ASCEND_MODE = "Developer"` → `FusedTopKKernel.ascend_mode = "Developer"`），无需手工填写模式；重复运行幂等（第二次报 `already integrated` / `unchanged`）。
