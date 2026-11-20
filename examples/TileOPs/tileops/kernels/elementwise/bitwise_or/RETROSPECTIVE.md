## Stage 5 (integrate) Retrospective — first_design (migration-harness) — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| none | none | none | none | - |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 同族 Stage 5 集成零调试：`bitwise_or` 的 Stage 0 wrapper 已预置 npuir 1-byte 存储适配器（`_kernel_dtype_str`/`_as_kernel_input`/`_restore_public_dtype`，uint8/bool→int8 承载 + 公开 dtype 还原），首跑即 14/14 correctness + 9/9 benchmark 全过、`status=passed`、attempts=1；对照 `bitwise_and` 曾耗 2 attempt 在 Stage 5 才发现并补上同一适配器——同族上游 Stage 5 教训回灌到下游 Stage 0 脚手架，可使下游 Stage 5 无需任何胶水修复 | P | `bitwise_or.py` L83–L127（适配器）/ L202–L234（kernel_dtype_str + forward 承载/还原）；`integration_log.md#验证结果`；`examples/TileOPs/reports/verification/20261013_085747_281389_BitwiseOrFwdOp/run.json`（summary.correctness_passed=true, benchmark_tests=9） | `python3 .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/bitwise_or/.migration_meta.json` 后 `python3 -m tileops.reporting.cli run --op BitwiseOrFwdOp --prof-mode msprof` | tilelang 0.1.1.10+ubuntu.20.4.cann900 / git 417dae9908ad / CANN 9.0.0 / Ascend910_9362 x1 / torch_npu 2.7.1.post4 / 2026-10-13 | pattern-library/traps-runtime.md（elementwise 1-byte 存储适配器，候选）；add-npu-op Stage 0 脚手架说明 |
| plain `uint8` 直接输入路径同样命中 CG-2026-0018（非仅 bool 存储路径）：`bitwise_or` 的 manifest/test 声明 `uint8` 为公开 dtype（test `test_bitwise_or_op[4096-dtype2]`），Op 非 bool 分支直接传 uint8，须在 wrapper 边界承载为 int8 并还原公开 dtype 才成立；实测该 param PASS（max_abs_err=0） | D | `bitwise_or.py` L193/L233–234；`tests/ops/test_bitwise_or.py` L61–66；`run.json` correctness case `test_bitwise_or_op[4096-dtype2]`（outcome=passed, max_abs_err=0.00e+00） | 同上（run.json 中 dtype2 对应 torch.uint8 param） | 同上 | pattern-library/traps-runtime.md（能力缺口 dtype 须按 manifest 全集推广，候选） |

### Transferable Lessons（可迁移教训）

- **同族算子的 Stage 5 胶水教训应回灌到下游 Stage 0 wrapper 脚手架**：`bitwise_and` 在 Stage 5 才发现 npuir 无法承载 1-byte `uint8`/`bool` GM 张量（CG-2026-0018），为此补了两个适配器（承载体 int8 + 还原公开 dtype）。`bitwise_or` 的 Stage 0 wrapper 已预置同款适配器，其 Stage 5 首跑即 14/14 + 9/9 全过、零调试。**上游 Stage 5 的失败修复，应在同族下游算子的 Stage 0 预置，而不是等下游 Stage 5 重复踩坑。**
- **能力缺口 dtype 的等价替代须覆盖 manifest 声明的全部路径**：CG-2026-0018 不只影响 `torch.bool` 存储路径，也影响 manifest 显式声明的 plain `uint8` 直接输入（`BitwiseOrFwdOp` 的 `test_bitwise_or_op[4096-dtype2]`）。wrapper 的 `_as_kernel_input` / `_restore_public_dtype` 必须对二者都生效（`view` 位保持，bit-exact），否则该 param 会以 int8 结果对 uint8 参考断言失败。
