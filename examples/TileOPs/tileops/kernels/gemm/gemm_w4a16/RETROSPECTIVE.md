# gemm_w4a16 复盘（op 级追加式日志）

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-15

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` 对「已集成 wrapper」重复运行**非幂等**：wrapper 已改写后不再含 `from .{extracted_module} import ...`，`parse_wrapper_imports()` 返回空 → `wrapper_names` 回退为 `meta.extracted_functions`（仅函数名）；据此重生成的聚合 `{op_slug}_kernel/__init__.py` 只导出工厂函数，丢掉 wrapper baseline import 块仍请求的非函数符号（本任务常量 `GROUP_SIZE`），import 冒烟报 `ImportError: cannot import name 'GROUP_SIZE'`。首次集成因 wrapper 尚从被抽取模块 import 而偶然把常量一并导出，掩盖了缺陷 | `examples/TileOPs/tileops/kernels/gemm/gemm_w4a16/gemm_w4a16_kernel/integration_log.md#调试历史`（attempt 1）；脚本 `.agents/skills/add-npu-op/scripts/integrate_kernel.py:493-495`（回退分支）、`:269-282`（`gen_init_py` 仅遍历 `names`） | 脚本生成 `__init__` 时，导出集合应取「`parse_wrapper_imports(wrapper, extracted_module)` ∪ wrapper 当前 baseline 块列出的符号 ∪ `extracted_functions`」的并集（`defining_module()` 已能定位常量）；并在 smoke 失败时给出定位指引 | R |
| Integrate-flow | 集成脚本的冒烟失败只打印 Python traceback + `[error] import smoke test FAILED`，未指出「聚合 `__init__` 导出集与 wrapper import 不匹配」这一常见根因，定位靠人工读脚本 | 同上 | `integrate_kernel.py` smoke 失败分支增补一行诊断：比对 wrapper baseline import 符号集与 `__init__.__all__`，打印差集 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 已集成 wrapper 的 baseline import 块中，非函数符号（常量/类）必须从 **per-function 模块**直取（`from .{op}_kernel.{func} import CONST`），不能依赖聚合 `__init__`——后者按 `extracted_functions` 重生成，只保证函数符号 | P | `examples/TileOPs/tileops/kernels/gemm/gemm_w4a16/gemm_w4a16.py:50-60`（baseline 拆两行后 smoke pass）；`integration_log.md#调试历史` attempt 1 | 重跑 `python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta <meta>` 两次，第二次即触发；改 wrapper 后 `smoke: pass` | tilelang 0.1.4+ubuntu.20.4.cann852 / CANN 8.5.2 / Ascend910B3 / 2026-10-15 | `.agents/skills/add-npu-op/scripts/integrate_kernel.py`（脚本修复）或 add-npu-op skill 的 wrapper 胶水指引 |
| harness 迁移中「共享测试文件被多个算子共用」会让单算子 correctness 门禁被**范围外算子**污染：report 对整个 `source.test` 文件跑 pytest，未集成算子（如 GemmOp 用 CUDA 风格 `compile_flags=`）的失败会连带本算子 verdict 失败 | C | `integration_log.md`（上一轮 #2，18 例 `TypeError: jit() got an unexpected keyword argument 'compile_flags'`）；本轮已按算子拆分 `tests/ops/test_gemm_w4a16.py` 后 6/6 通过 | none（前一轮现象，本轮修复已消除） | tilelang 0.1.4 / CANN 8.5.2 / Ascend910B3 / 2026-10-15 | `tileops.manifest` 迁移规范：每算子独立 `source.test` / `source.bench`；或 report 增按用例筛选能力 |
| `import tilelang` 后同进程对 NPU 张量做位运算（`torch.bitwise_left_shift`，如 INT4 打包）会触发 TBE/GE 惰性 op-JIT 初始化失败（`GEInitialize failed`）；把该计算改到 **CPU** 即可规避 | D | `integration_log.md`（上一轮 #1，含最小复现 `q[:,1::4] << 4` 失败 / 去 `import tilelang` 成功）；本轮 `tileops/workloads/gemm.py::quantize_weight_int4` 改 CPU 后未复发 | `import torch, tilelang, torch_npu; q=torch.randint(0,16,(64,128),device="npu",dtype=torch.uint8); q[:,1::2] << 4` | tilelang 0.1.4 / CANN 8.5.2 / Ascend910B3 / 2026-10-15 | `.agents/evolution/capability-gaps.md`（运行期初始化冲突）或 harness workload 生成规范：张量打包/位运算留在 CPU |

### Transferable Lessons（可迁移教训）

- 集成脚本可以安全重复运行，但**重复运行会重生成聚合 `__init__.py`**：跑完后务必确认 wrapper 的 baseline import 块里每个符号都能从聚合包导出；非函数符号（如 `GROUP_SIZE`）请在 wrapper 里改从 per-function 模块直取。
- harness 迁移的 correctness 门禁以 `manifest.source.test` 为单元：只要该文件还混入其他未集成算子，本算子 verdict 就会被连带判失败——迁移前先把测试/基准按算子拆干净。
- workload 的输入生成（打包/位运算/整数运算）尽量留在 CPU；在 `import tilelang` 的进程里对 NPU 张量调用惰性 JIT 的逐元素 op 会命中 GE 初始化冲突。
- benchmark 的 `Ratio < 1%` 这类数值只记录不修复；msprof 在超大 K（本任务 K=81920, 55.1ms）下依然能产出有效 profile，未超时。
