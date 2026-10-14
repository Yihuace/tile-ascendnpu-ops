# RETROSPECTIVE — FusedAddLayerNormFwdOp（`fused_add_layer_norm`，op-level，migration-harness）

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | Stage 0 scaffold 生成的 wrapper `forward` 沿用 GPU 调用形态 `factory(M,N,eps,dtype)(x,residual,weight,bias)`，与 NPU factory 契约 `factory(M,N,eps,dtype) -> _func(block_m=default)` 不匹配（少一层 `block_m` 选择调用），集成首跑运行期即 `TypeError: _func() takes from 0 to 1 positional arguments but 4 were given`；import smoke（只 import）无法拦截此类胶水错误 | fused_add_layer_norm_kernel/history_version/fused_add_layer_norm_s5_attempt1.py（修复前）；integration_log.md#调试历史 | add-npu-op 集成 skill 应在「冒烟」阶段增加一次小 shape 端到端 `Op.forward()` 调用（而非仅 import），或在 scaffold wrapper 生成时按 NPU factory 契约（`_func(block_m)` 选择层）预置 `forward` | R |
| Integrate-flow | 任务书 §2.1 护栏 2/3 的 grep/`__module__` 路径按 `tileops/kernels/<family>/<op_slug>/integration_log.md` 书写，但 harness 下 integrate_kernel.py 实际写入 `{op_slug}_kernel/integration_log.md`；且护栏 3 的 `w.<func>` 在包 `__init__` 上不可达（包 `__init__` 仅 re-export Kernel 类，与 ada 同构） | tileops/kernels/norm/fused_add_layer_norm/fused_add_layer_norm_kernel/integration_log.md；tileops/kernels/norm/fused_add_layer_norm/__init__.py | 任务书 §2.1 护栏路径更新为 `<op_slug>_kernel/integration_log.md`，护栏 3 改为在 wrapper 模块上取 `<func>.__module__` | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 多函数/harness NPU wrapper 的 factory 调用层数陷阱（GPU `_func(block_m, threads)` → NPU `_func(block_m=default)`）：`forward` 必须 `factory(...)()(tensors)`，仅改 wrapper 胶水一行即修复；同族 ada wrapper 的 `factory(...)(block_m)(tensors)` 是该契约的正确参照 | P | tileops/kernels/norm/fused_add_layer_norm/fused_add_layer_norm.py:151-154（修复后）；history_version/fused_add_layer_norm_s5_attempt1.py；tileops/kernels/norm/ada_layer_norm/ada_layer_norm.py:104-146 | `python -c "from tileops.ops.norm.fused_add_layer_norm import FusedAddLayerNormFwdOp"` 后小 shape `op(x,r,w,b)`；再 `python -m tileops.reporting.cli run --op FusedAddLayerNormFwdOp --prof-mode msprof --kernel-name main` | tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b（2026-10-10 build）+ CANN 8.5.2 + Ascend910B3 + torch 2.7.1+cpu / torch_npu 2.7.1.post4 | pattern-library/cases.md |
| FusedAddLayerNormFwdOp 集成期实测：correctness 14/14（max_abs_err=1.56e-02 @ bf16 2048x4096，容差 1.6e-2）、benchmark 10/10 全 msprof 有效（msprof_case_count=7，fallback=0/invalid=0），双输出（output/residual_out）契约经 Op 层 reshape 正确透传；roofline 利用率 0.58–45.08% | D | reports/tileops/20261014_115534_389335_FusedAddLayerNormFwdOp/run.json（summary.correctness_passed=true, benchmark_passed=true, msprof_case_count=7, profiler_fallback_count=0, invalid_profile_count=0） | `python -m tileops.reporting.cli run --op FusedAddLayerNormFwdOp --prof-mode msprof --kernel-name main`（从 examples/TileOPs/） | tilelang 0.1.2+417dae99（2026-10-10 build）+ CANN 8.5.2 + Ascend910B3 / torch 2.7.1 / 2026-10-14 | pattern-library/cases.md |

### Transferable Lessons（可迁移教训）

- harness 迁移 NPU wrapper 时，务必先核对 Stage-3 kernel 的 factory 契约调用层数：GPU 常见 `factory(M,N,eps,dtype)(block_m, threads)(tensors)`，NPU 重设计后多为 `factory(M,N,eps,dtype)(block_m=default)(tensors)`。若 wrapper `forward` 直接把张量传给 factory 返回值（漏掉 block_m 选择层），会在运行期报 `TypeError: _func() takes from 0 to 1 positional arguments but N were given`；修复只需在 factory 后补一对空括号 `()(...)`（使用工厂默认 block_m，K11 不烘焙执行参数）。同族 ada wrapper 是可复用的参照形态。
- import smoke 只验证可导入，不能发现 wrapper `forward` 的调用胶水错误；集成期应补一次小 shape（如 M=8,N=64）端到端 `Op.forward()` 调用作为胶水冒烟，再跑完整单算子 report。
- harness 单算子集成验收：`python -m tileops.reporting.cli run --op {PascalCaseOpName} --prof-mode msprof --kernel-name main`（从 `examples/TileOPs/` 运行，串行无 `-n`）；结论必须来自本次时间戳目录的 `run.json`，`status=passed` 即 correctness+benchmark 均通过且 msprof 有效。
