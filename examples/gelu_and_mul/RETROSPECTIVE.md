# RETROSPECTIVE — GeluAndMulFwdOp（`gelu_and_mul`，migration-harness，op 级）

> Op 级单份集成复盘（集成问题不归属单个函数）。函数级 Stage 1/2/3 复盘见
> `gelu_and_mul_fwd_kernel/RETROSPECTIVE.md`。

## Stage 5 (integrate) Retrospective — attempt 2 — 2026-10-13

> 出口：**`INTEGRATE_COMPLETED`**。attempt 1（`[INTEGRATE_FAIL]`，正确性 10/11）根因不在
> kernel，而在 Stage-0 冻结测试的参考实现落在 NPU 设备侧；conductor 于 Stage 0 修正测试参考
> （改为 CPU + 显式 `torch.erf` 的精确 GELU）后本 attempt 重集成、重跑 report，全绿。
> 调试闭环消耗 attempt 数：0（两次都无允许范围内的 kernel/wrapper 修复可做）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | §2.1 护栏 3 的字面命令 `python -c "... importlib.import_module('tileops.kernels.<family>.<op_slug>'); print(w.<func>.__module__)"` 假设被检函数是 op_slug 模块的直接属性；当 wrapper 经 `_{func}_kernels.py` / `{op_slug}_kernel` 子包 re-export 时该属性不存在，命令直接抛 `AttributeError`（本次实测），无法给出基线判定 | `TileOPs迁移任务书.md` §2.1（L121）；本次实跑：`tileops.kernels.elementwise.gelu_and_mul` 无 `gelu_and_mul_fwd_kernel` 属性（`Did you mean: 'gelu_and_mul_kernel'`）；改查 wrapper 模块得基线 `...gelu_and_mul_kernel.gelu_and_mul_fwd_kernel`（非 perf_opt） | 任务书 §2.1 护栏 3 命令改为经 wrapper：`w=importlib.import_module('tileops.kernels.<family>.<op_slug>.<op_slug>'); print(w.<func>.__module__)`，或注明「若 op_slug 模块无该属性则查 wrapper 模块」 | R |
| Info-source | Stage-0 冻结测试的 GELU 参考在 **NPU 设备**上调用 `F.gelu`，而本 torch_npu 构建 `F.gelu(approximate="none")` 实为 **tanh 近似**；该参考偏差（~1.2e-3）超过 fp32 容差（1e-5），使唯一 fp32 用例对**正确 kernel** 必然失败，白耗一次 conductor 级重调度（attempt 1 → 2） | `examples/TileOPs/tests/ops/test_gelu_and_mul.py::GeluAndMulTest.ref_program`（修正前，设备侧 `F.gelu`）；`integration_log.md` 根因分析表（kernel vs 精确 GELU 4.9e-7；设备 `F.gelu(approximate="none")` vs CPU tanh 2.38e-7） | Stage-0 测试脚手架生成 erf-GELU 类算子参考时，默认在 **host（CPU）** 用显式 `torch.erf` 计算后 cast 回设备，不得依赖设备侧 `F.gelu` 的 `approximate` 语义 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **本机 torch_npu `F.gelu(approximate="none")` 在 NPU 设备上返回 tanh 近似（非精确 erf）**：设备 `F.gelu(none)` 与 CPU `F.gelu(tanh)` 差 2.38e-7，与精确 erf-GELU 差 ~1.2e-3；故 erf 语义的 GELU 测试参考**不能**在设备上求 | D | `examples/TileOPs/tests/ops/test_gelu_and_mul.py`（修正前 ref_program）；`integration_log.md`「根因分析」表 | 设备上 `F.gelu(x, approximate="none")` vs CPU 显式 erf 对比 | torch_npu 2.7.1.post4 + CANN 8.5.2 + Ascend910B3 | pattern-library/traps-runtime.md（数值口径类） |
| **erf-GELU 类迁移的测试参考形态：host CPU + 显式 `0.5*g*(1+torch.erf(g/√2))` 再 `to(device)`**；对精确 GELU 参考与 kernel 比对可达 fp32 1e-5（本次 fp32 `max_abs_err=1.43e-06`） | P | 修正后 `test_gelu_and_mul.py::ref_program`；本次 `run.json` fp32 用例 `max_abs_err=1.43e-06` | `python -m tileops.reporting.cli run --op GeluAndMulFwdOp --prof-mode msprof` | torch_npu 2.7.1.post4 + CANN 8.5.2 + Ascend910B3 | pattern-library/elementwise.md |
| **集成幂等重跑的无害信号：`grep "already integrated"` 命中 ≠ 失败**：脚本同时写 `wrapper_rewrite_action`，须与护栏 3（`__module__` 指向 baseline）联合判定；本算子重跑 `already integrated` + 模块指向 `...gelu_and_mul_kernel.gelu_and_mul_fwd_kernel`，属正常 | P | `integration_log.md` 护栏段（attempt 2）；`integrate_kernel.py` `[json]` `wrapper_rewrite_action=already integrated` | 重跑 `integrate_kernel.py --meta ...` 后核对护栏 1/3 | tilelang dev-root + CANN 8.5.2 | pattern-library/integration.md |

### Transferable Lessons（可迁移教训）

- **erf 语义的 GELU 参考必须在 host 上算**：本机 torch_npu 的设备侧 `F.gelu(approximate="none")`
  实际是 tanh 近似（与精确 erf 差 ~1.2e-3）。任何「exact GELU / erf-GELU」测试若在 NPU 上调用
  `F.gelu`，会因参考自身偏差让正确 kernel 在 fp32 容差下失败——参考应写
  `0.5*g*(1+torch.erf(g*(0.5**0.5)))` 于 CPU 求值后 cast 回设备。
- **`[INTEGRATE_FAIL]` 也可能源自冻结测试参考而非 kernel**：本次 kernel 对精确 GELU 达 4.9e-7，
  却被设备侧参考判 fail。integrator 应做定量三路归因（kernel vs 精确参考 / kernel vs 设备参考 /
  设备参考 vs 精确参考），确认根因不在允许修改范围后如实上报、不消耗 attempt、不改冻结文件。
- **§2.1 护栏 3 要落在 wrapper 模块**：当算子包用 `{op_slug}_kernel` 子包 re-export kernel 函数时，
  `tileops.kernels.<family>.<op_slug>` 上不一定有 `{func}` 属性；应 import
  `tileops.kernels.<family>.<op_slug>.<op_slug>`（wrapper）再读 `w.{func}.__module__`，确认指向
  baseline 而非 `perf_opt` 旧谱系。
- **集成脚本幂等**：重跑 `integrate_kernel.py` 覆盖 kernel/`DESIGN.md` 并生成聚合 `__init__.py`；
  `already integrated` 是常态信号，须与护栏 1（`wrapper_rewritten=true`）和护栏 3（baseline 模组）
  联合判定。
