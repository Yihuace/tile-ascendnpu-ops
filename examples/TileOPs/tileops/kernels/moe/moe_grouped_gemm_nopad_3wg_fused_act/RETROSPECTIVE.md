# RETROSPECTIVE.md — moe_grouped_gemm_nopad_3wg_fused_act

> Op-level retrospective (migration-harness). Stage 1/2/3 per-function retrospectives live under
> `_fused_act_kernel/RETROSPECTIVE.md`; this file holds the op-level Stage 5 integration retrospective.

## Stage 5 (integrate) Retrospective — migration-harness — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` 改写 wrapper import 时**不校验** `parse_wrapper_imports` 得到的名字是否存在于集成包；当 Stage-3 产物导出名与 Stage-0 scaffold 占位名不符（本例 `_fused_act_kernel` vs `_gemm_gate_up`/`_act_packed`）时，生成指向不存在名字的 import，且 `__init__.py` 写 `# [warn] ... not found in integrated modules` | 首次运行 stdout `[json] ... "reexported": ["_fused_act_kernel"]`；生成包 `__init__.py` 含 `# [warn] _fused_act_kernel: not found in integrated modules`；`integration_log.md` §Wrapper import 改写 | `integrate_kernel.py` 在 `rewrite_wrapper_import` 前用 `defining_module` 校验名字覆盖度，任一缺失即 `[error]` 提示改 wrapper（禁止生成坏 import） | R |
| Integrate-flow | 脚本**非幂等**：wrapper 已集成后重跑，`main()` 用 `wrapper_names = parse_wrapper_imports(...) or list(functions)`，此时前者为空 → 回落到 `meta.extracted_functions`（陈旧值）重建 `__init__.py` 为坏导出并令 smoke ImportError 失败 | `integrate_kernel.py` `main()` 的 `wrapper_names` 回退分支；`meta.extracted_functions == ["_fused_act_kernel"]`；`integration_log.md` §issues #2 | 把已解析的 `wrapper_names`（或集成模块顶层导出名）持久化/复用；重跑不得回退到 stale `functions` | R |
| Integrate-flow | 脚本自带 import smoke 仅用 `hasattr` 判定导出，存在**同名子模块假阳性**：`from .pkg import name`（name == 子模块 stem）在名字缺失时回落到导入子模块，`hasattr` 为 True 但对象是 module | 首次（坏 import）运行脚本仍报 `smoke: pass`；人工复现 `type(w._fused_act_kernel) == module` 且调用抛 `TypeError: 'module' object is not callable`；`integration_log.md` §import smoke | smoke 对每个导出补 `callable()` 断言（工厂必须可调用），而非仅 `hasattr` | R |
| Develop-flow | Stage-3 交付未同步适配 wrapper：产物把 scaffold 假定的单工厂拆成两 kernel（`_gemm_gate_up`+`_act_packed`），但 wrapper 仍按单工厂 `_fused_act_kernel(...)` 调用（"K3：NPU kernel component rewrites the factory" 假设未兑现） | wrapper scaffold docstring "Pre-Stage-3 note"；产物 `_fused_act_kernel.py` 顶层 def `_gemm_gate_up`(L231)/`_act_packed`(L322)，无 `_fused_act_kernel` | migration-harness 在 Stage 3 交付校验增加"wrapper 调用点 ↔ 产物导出名"一致性项；或要求产物保留 scaffold 工厂名 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 集成期坏 import 修复法：把 wrapper 的 import glue 改为产物**真实导出名**后重跑 `integrate_kernel.py` 一次，脚本即据该 import 同时产出 canonical baseline/perf_opt 块与正确 `__init__.py`（`[warn]` 消失）——仅改 wrapper，不动脚本、不动 `__init__.py` | R | `integration_log.md` §Wrapper import 改写（"最终"块）；stdout `[json] reexported: ["_gemm_gate_up","_act_packed"]` | `python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta <meta>`（wrapper 的 import 名已改为真实导出名后） | tilelang dev root（build/）+ CANN 9.0.0 / Ascend910B2C / 2026-10-16（继承 Stage-1/3 同机标注） | `tilelang-op-integrate` / add-npu-op skill（integrate_kernel.py 缺陷目录） |
| import smoke 假阳性实证：**factory 名 == 模块 stem** 时 `from .pkg import name` 回落为导入子模块，`hasattr` 判真但对象不可调用，集成"通过"实为坏接线 | D | `integration_log.md` §import smoke；`moe_grouped_gemm_nopad_3wg_fused_act_kernel/__init__.py`（坏版本无 import 语句、仅 `[warn]`） | `python -c "import tileops.kernels.moe.moe_grouped_gemm_nopad_3wg_fused_act.moe_grouped_gemm_nopad_3wg_fused_act as w; print(type(w._fused_act_kernel))"`（坏 import 版本 → `<class 'module'>`） | tilelang dev root（build/）+ CANN 9.0.0 / Ascend910B2C / 2026-10-16 | traps-runtime / integrate skill smoke 规范 |
| 多 kernel 产物的 host 侧调度表必须补齐到 kernel 的**静态 `max_tiles`**（补 `(-1, 0)` sentinel），只发 live `total` 条会导致静态形状张量传参失败；本任务 wrapper `_build_tile_schedule` 与 Stage-3 `build_schedule` 在 6 个 test/bench shape 上逐元素一致 | P | `integration_log.md` §补充正确性证据；`_fused_act_kernel.py` `build_schedule`(L198) | `python -c "from ..._fused_act_kernel import build_schedule,max_tiles_of; from ..._3wg_fused_act import MoeGroupedGemmPersistent3WGFusedActKernel as K; ..."`（对比 ids/rows/total） | tilelang dev root（build/）+ CANN 9.0.0 / Ascend910B2C / 2026-10-16 | pattern-library/migration.md |

### Transferable Lessons（可迁移教训）

- 集成多 kernel 产物时，**不要**只信 `integrate_kernel.py` 的 `smoke: pass`：`hasattr` 会在"工厂名 == 模块 stem"时把子模块误判为导出。必须确认导入对象 **可调用**（工厂），并对集成包 `__init__.py` 逐个核对导出名与产物顶层 def 名一致。
- 若 Stage-3 产物导出名与 wrapper scaffold 的占位工厂名不同，**只改 wrapper 的 import glue 为真实导出名**，再重跑一次 `integrate_kernel.py`；不要改脚本，也不要手改生成的 `__init__.py`（脚本会据 wrapper import 重新生成）。
- harness 集成算子时，wrapper 的 `forward` 通常是 scaffold 态（按单工厂占位）。集成前先核对产物实际导出与调用签名；把 `forward` 按产物已验证的 launch 顺序（如 `run_case`）改写并逐步对齐参数顺序（含 out_idx 决定的返回值）。
- 拆分型（GEMM + Vector 双核）算子在 TileOPs 下每次 `forward` 起两个 device kernel，msprof 无 filter 时解析不到单算子 → 集成后须为 Kernel 类补 `msprof_kernel_name`（需实测具体 device kernel 名，勿猜写）。
- host 侧构造的 kernel 输入表（如调度表）必须与 kernel 的**编译期静态形状**一致；有 sentinel 语义的表要补满到 `max_tiles`，否则静态张量传参直接失败。
