# RETROSPECTIVE -- moe_permute_nopad (migration-harness)

> op 级复盘（Stage 5 integrate 为 op 级单份；逐函数 Stage 1/2/3 复盘见
> `_make_scan_kernel_nopad/RETROSPECTIVE.md` 与 `_make_gather_kernel_nopad/RETROSPECTIVE.md`）。

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` 的 `gen_init_py` / `gen_kernel_source_block` 直接用 wrapper 的 import 名集合（`parse_wrapper_imports`）生成集成包 `__init__.py` 与 baseline import block，**不过滤集成包内不存在的导出名**（未迁移的 GPU-only EP 变体 `_make_scan_kernel_nopad_ep`），生成坏 import 导致 smoke `ImportError` | `integration_log.md#已知缺陷复现--修复integrate_kernelpy`；生成物 `moe_permute_nopad_kernel/__init__.py` L9 的 `# [warn] ... not found in integrated modules` 仍把该名留在 `__all__`；wrapper `moe_permute_nopad.py` baseline block 导入该名 | 在 `.agents/skills/add-npu-op/scripts/integrate_kernel.py` 的 `gen_init_py`/`gen_kernel_source_block`（或 `main` 调 `gen_init_py` 前）过滤 `defining_module(name, integrated_files) is None` 的名；选择：跳过并从 `__all__` 移除，或显式绑定到 Stage-0 模块 | R |
| Integrate-flow | 因 `__init__.py` 每次运行都被无条件重写，上述缺陷使脚本对本算子**不幂等**：重跑会把缺失名胶水改回坏 import 并再次 smoke 失败 | `integration_log.md#已知缺陷复现--修复integrate_kernelpy`（Note 段）；脚本 `main` L539-546 无条件 `init_path.write_text(init_text)` | 同上一行的过滤逻辑落地后自然修复（重跑不再生成缺失名） | R |
| Info-source | `.migration_meta.json` 的 `extracted_functions` 声明 3 个函数（含 `_make_scan_kernel_nopad_ep`），但 conductor 产物只有 2 个函数目录；脚本默认按 meta 全量集成会 `[error] no conductor product` | `.migration_meta.json` L14-18 与 `examples/moe_permute_nopad/`（仅 2 个子目录）；`integration_log.md#集成命令` | 让 scaffolder 在元数据里区分「本任务迁移函数」与「wrapper 引用但未迁移函数」（如新增 `migrated_functions` 键），避免集成时需人工 `--functions` 覆盖 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| integrate_kernel.py import 改写不过滤集成包不存在的导出名 → 生成坏 import（`ImportError: cannot import name ... from ..._kernel`） | R | `integration_log.md#已知缺陷复现--修复integrate_kernelpy`；生成物 `moe_permute_nopad_kernel/__init__.py`、wrapper `moe_permute_nopad.py` | `cd examples/TileOPs && python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/moe/moe_permute_nopad/.migration_meta.json --functions _make_gather_kernel_nopad _make_scan_kernel_nopad`（首次运行 smoke 必失败） | tilelang 0.1.2+8875071d41a4587c21b6c1be0eb075d090b5bbf4 (repo HEAD 8875071d)；Ascend910B2C；torch 2.7.1+cpu / torch_npu 2.7.1.post4；npu-smi 25.5.5 | `.agents/skills/add-npu-op/scripts/integrate_kernel.py` / 待定 |
| 集成子集时，缺失导出名的胶水修法：把缺失名绑定到 Stage-0 模块对象（而非删除），保留 wrapper 原有代码路径可导入 | P | `integration_log.md#s5-glue-fix-final-baseline-active-perf_opt-lines-unchangedcomment-only`；smoke 输出 `_make_scan_kernel_nopad_ep.__module__ = ..._moe_permute_nopad_fwd_kernels` | 修 `moe_permute_nopad_kernel/__init__.py` + wrapper baseline block 后重跑 smoke（exit 0） | 同上 | pattern-library / 待定 |

### Transferable Lessons（可迁移教训）

- 当 wrapper 引用的 kernel 工厂只迁移了一部分（例如只迁移主路径、EP/变体路径留在 Stage-0）时，`integrate_kernel.py` 生成的集成包 `__init__.py` 与 wrapper baseline import block 会把**未迁移名**一起写进去 → smoke `ImportError`。修法：仅在本算子 import 胶水层把缺失名绑定到 Stage-0 模块对象（`from .<stage0_module> import <missing>`），不要改脚本。
- `integrate_kernel.py` 每次运行都无条件重写集成包 `__init__.py`，因此对该文件的任何手工胶水修复都不幂等；重跑前需确认是否会引入坏 import（本任务已把该缺陷报给 conductor 统一处理）。
- 运行集成脚本前先核对 `.migration_meta.json` 的 `extracted_functions` 与 conductor 产物目录是否一致；不一致时用 `--functions` 显式列出实际迁移函数，避免 `[error] no conductor product`。

## Stage 5 (integrate) Retrospective — attempt 2 re-integration (EP variant) — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `rewrite_wrapper_import` 用 `re.search(r"from \.moe_permute_nopad_kernel(\.\w+)?\s+import", text)` 短路（wrapper 已含该 import 即返回 `already integrated`），因此**增量重集成一个新函数**时脚本不会把新工厂名补进 wrapper 的 baseline import block；本次须人工补 `_make_scan_kernel_nopad_ep` + 其 `VECTOR_CORES` | `integration_log.md#另一处脚本行为本次观察到`；脚本 `integrate_kernel.py` L376-377；本次脚本输出 `wrapper_rewritten=false` | 在 `rewrite_wrapper_import` 中：即使已集成，也比对 `integrated_files` 与 wrapper 现有 import 名，把缺失的新函数名补进 baseline block（而非整体短路） | R |
| Integrate-flow | attempt 1 记录的「import 改写不过滤集成包不存在的导出名」缺陷本次**未复现**：3 函数全部迁移后生成物干净，`__init__.py` 正确重导出 3 名，上一 attempt 的 Stage-0 fallback 胶水被覆盖且不再需要。缺陷对「存在未迁移名」的情形仍潜伏 | `integration_log.md#已知缺陷复现integrate_kernelpy` 与 `#集成包-__init__py-改写`；本次 `moe_permute_nopad_kernel/__init__.py`（无 `# [warn]`、无 Stage-0 绑定） | 同 attempt 1 建议（脚本侧过滤 `defining_module(...) is None` 的名）；本条仅更新「触发条件已消失、缺陷仍在」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| NPU EP scan 变体接线契约：工厂 `_make_scan_kernel_nopad_ep(numel, e_global, e_local, top_k)`，run 可调用 `run(flat_ids, expert_map, bh, true_offsets, true_sizes, efo, slot_base, run_buf, fwd_idx, permuted_idx)`（`expert_map` 为第 2 位置参数）；wrapper 须用 **EP 模块自身的** `VECTOR_CORES` 定 scratch NB，并传 `expert_map.flatten().contiguous()` | P | `integration_log.md#ep-路径重接本次任务核心`；`examples/moe_permute_nopad/_make_scan_kernel_nopad_ep/_make_scan_kernel_nopad_ep.py#L341-368` | smoke 断言 `pkg._make_scan_kernel_nopad_ep.__module__` 指向集成包 + `inspect.signature` = `(numel, e_global, e_local, top_k)`（见 `integration_log.md#smoke-结果`） | tilelang 0.1.2+8875071d41a4587c21b6c1be0eb075d090b5bbf4 (repo HEAD 8875071d)；Ascend910B2C；torch 2.7.1 / torch_npu 2.7.1.post4；npu-smi 25.5.5 | pattern-library / 待定 |
| `integrate_kernel.py` 增量重集成不重写 wrapper import block（guard 短路）→ 新函数名须人工补进 baseline block | R | `integration_log.md#另一处脚本行为本次观察到`；脚本 `integrate_kernel.py` L376-377 | 重跑 `--functions _make_gather_kernel_nopad _make_scan_kernel_nopad _make_scan_kernel_nopad_ep`，观察 `[wrapper] ... already integrated` 且 wrapper 文件 diff 为空 | 同上 | `.agents/skills/add-npu-op/scripts/integrate_kernel.py` / 待定 |

### Transferable Lessons（可迁移教训）

- 把 wrapper 引用的**变体/EP 工厂**接入集成包时，`integrate_kernel.py` 对已集成过的 wrapper 会短路（`already integrated`），**不会**把新工厂名补进 baseline import block；必须在 wrapper 胶水层人工加「新工厂名 + 该工厂所属模块自己的 `VECTOR_CORES`」，并同步更新 factory 分派与 forward 调用。
- 同一算子不同 scan 变体（非 EP / EP）是**独立模块、各自持有 `VECTOR_CORES`**：wrapper 计算 scratch `NB = min(VECTOR_CORES, numel)` 时，必须按当前走的分支绑定对应模块的常量，不能复用另一个变体的常量（即使当前数值恰好相同）。
- 「集成了一个变体函数」不等于「wrapper 路径已通」：工厂签名与 run 可调用的参数表都可能与主路径不同（EP 版 run 多一个 `expert_map` 第 2 位置参数），须逐一对照已修好的兄弟路径的真实调法，而非照搬 GPU 参考签名。

