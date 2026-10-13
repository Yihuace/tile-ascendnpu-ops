# RETROSPECTIVE — GeluTanhAndMulFwdOp（`gelu_tanh_and_mul`，migration-harness，op 级）

> Op 级单份集成复盘（集成问题不归属单个函数）。函数级 Stage 1/2/3 复盘见
> `gelu_tanh_and_mul_fwd_kernel/RETROSPECTIVE.md`。

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-13

> 出口：**`INTEGRATE_COMPLETED`**。首次集成即全绿：集成脚本 `[wrapper] rewritten`、smoke pass、
> `ascend_mode=Developer`；单算子 report `status=passed`，正确性 11/11、benchmark 7/7，
> msprof_case_count=5、fallback/invalid 全 0。调试闭环消耗 attempt 数：0（无允许范围内的修复可做）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | §2.1 护栏 3 的字面命令 `python -c "... importlib.import_module('tileops.kernels.<family>.<op_slug>'); print(w.<func>.__module__)"` 假设被检函数是 op_slug 模块的直接属性；本算子 wrapper 经 `{op_slug}_kernel` 子包 re-export `gelu_tanh_and_mul_fwd_kernel`，`tileops.kernels.elementwise.gelu_tanh_and_mul` 模块上无该属性，字面命令抛 `AttributeError`，无法给出基线判定 | `TileOPs迁移任务书.md` §2.1（L121）；本次实跑：改查 wrapper 模块 `tileops.kernels.elementwise.gelu_tanh_and_mul.gelu_tanh_and_mul` 得基线 `...gelu_tanh_and_mul_kernel.gelu_tanh_and_mul_fwd_kernel`（非 perf_opt），见 `gelu_tanh_and_mul_kernel/integration_log.md` 护栏段 | 任务书 §2.1 护栏 3 命令改为经 wrapper：`w=importlib.import_module('tileops.kernels.<family>.<op_slug>.<op_slug>'); print(w.<func>.__module__)`，或注明「若 op_slug 模块无该属性则查 wrapper 模块」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| none | - | - | - | - | - |

### Transferable Lessons（可迁移教训）

- **§2.1 护栏 3 要落在 wrapper 模块**：当算子包用 `{op_slug}_kernel` 子包 re-export kernel 函数时，
  `tileops.kernels.<family>.<op_slug>` 上不一定有 `{func}` 属性；应 import
  `tileops.kernels.<family>.<op_slug>.<op_slug>`（wrapper）再读 `w.{func}.__module__`，确认指向
  baseline 而非 `perf_opt` 旧谱系。
