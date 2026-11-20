# group_norm 任务复盘（Retrospective）

> 本文件按 `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md` 追加式维护；每个 Stage 执行追加一个章节。
> 本章节为 **Stage 5（集成）的 op 级复盘**（集成问题不归属单个函数）；函数级复盘见 `_group_norm_kernel/RETROSPECTIVE.md`。

## Stage 5 (integrate) Retrospective — attempt_2 — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | Stage 5 默认编辑范围（kernel 集成目录 + wrapper 胶水）**不含 `ops/` 层**，但当失败根因落在 op 封装层的数值路径（本例 per-channel affine 的 dtype）时，闭环内没有任何可改文件能消除失败；agent 只能在 attempt 1 报告根因并交回，需 conductor 单独授权才能修。契约文本未给出"根因在 ops/ 且为用户显式授权的单点修复"这一合法路径，也未在授权后给出更新后的交付/记录要求 | `integration_log.md`（attempt 1「Root cause is out of the Stage 5 edit scope」/ attempt 2「user-authorized single-site ops fix」）；`tileops/ops/norm/group_norm.py` fp32-affine 单点 diff | 在 Stage 5 agent 契约的「编辑范围」与「调试闭环」段落补一条：当根因经证据定位在 ops 层数值路径、且 conductor 显式授权单点修复时，允许在**备份 + `integration_log.md` 记录 + 单点最小 diff**约束下实施，并在交付中标注该授权来源 | R |
| Integrate-flow | Stage 3 的 L0 门禁对 fp16 用了**单位 affine**（kernel 的 `_affine_for` 对 fp16 返回 `"unit"`），因此「kernel 舍回 dtype + op 层 dtype affine 二次舍入」这个 trap 在 Stage 3 从未成为阻塞门禁，直到 Stage 5 用**真实 op 调用路径**（per-channel fp16 affine）跑全量 report 才暴露。说明 kernel 级单元测试与集成级 op 路径的 affine 覆盖存在系统性缝隙 | `_group_norm_kernel/RETROSPECTIVE.md` Stage 3 价值点行（`TRAP-fp16-op-affine-double-rounding.py`，8 seeds 314 violation，kernel 级 unit affine 0/6）；`integration_log.md` attempt 1 第 4 点；本目录 attempt 2 | 在 `tilelang-op-develop`（或 design 验证方案）加一行「op-path affine 覆盖」：fp16/bf16 的 L0/L1 门禁若用单位仿射，须额外补一个**真实 per-channel affine** 的 op 级用例（或在 Stage 5 明确标为该 trap 的必查项） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| fp16 GroupNorm「kernel 舍回 dtype + op 层 dtype affine」二次舍入在**集成 report 层**复现并确证：集成 kernel 的 fp16 归一化输出与 golden 归一化张量**逐位一致**（`k_norm16 != g_norm16 : 0/4096`），7 个 fp16 用例的超差（每例 1–3 元素，max ~1.0–1.7e-3 vs 门禁 1e-3）**全部**归因于 op 层 fp16 affine 的双重舍入；改 fp32 算 affine 单次 `.to(dtype)` 舍回后 19/0（0 failed）。这是 Stage 3 CPU repro 的端到端（NPU + report）实机确认 | D | `examples/TileOPs/reports/tileops/20261015_162108_863948_GroupNormFwdOp/run.json`（attempt 1：correctness_passed=false，12/7/2）与 `.../20261016_003328_665600_GroupNormFwdOp/run.json`（attempt 2：correctness_passed=true，19/0/2）；`integration_log.md` attempt 1 定位段 3（0/4096 bit-exact）与 attempt 2 修复段；修复单点 `tileops/ops/norm/group_norm.py` fp32-affine | `cd examples/TileOPs && source /workspace/env.sh && python -m tileops.reporting.cli run --op GroupNormFwdOp --prof-mode msprof`（对比改动前后同 21 用例的 fp16 结果）；CPU 侧最小复现见 `examples/group_norm/_group_norm_kernel/repro/TRAP-fp16-op-affine-double-rounding.py` | tilelang 0.1.2+9307929320d + CANN 9.0.0 + Ascend910B3 x1 / 2026-10-16；torch 2.9.0 + pytorch_npu 2.9.0.post8 | pattern-library `traps-runtime.md`（测试设计族，与 Stage 3 的 `TRAP-fp16-op-affine-double-rounding` 条目同源，本条补 NPU 实机/report 级证据） |
| 集成期「kernel 无缺陷」的**证伪式定位手法**：先用 golden 的去-affine 归一化张量与 kernel 输出做**逐位**比较（`k_norm16 != g_norm16 : 0/4096`）排除 kernel，再把残差归到 op 层 affine 的 fp32 vs dtype 算术（`ideal = normalized.fp16 → fp32 affine → 单次舍回` 给出 `bad=0`），使"改哪个文件"有可执行证据而非猜测 | P | `integration_log.md` attempt 1 定位段 2–3；attempt 2 结果段（kernel 未改而全绿） | `cd examples/TileOPs && source /workspace/env.sh && python -m tileops.reporting.cli run --op GroupNormFwdOp --prof-mode msprof`（修复前后对比）+ 逐位比较脚本思路见 `integration_log.md` | tilelang 0.1.2+9307929320d + CANN 9.0.0 + Ascend910B3 x1 / 2026-10-16 | pattern-library `references/migration-analysis.md`（集成/调试模式：分层定位——kernel 逐位排除 → ops 层算术归因） |

### Transferable Lessons（可迁移教训）

- **迁移算子的 fp16/bf16 精度失败，先分层定位再修**：把「归一化张量」这一中间量在 kernel 输出与 golden 之间做**逐位**比较（`!= : 0/N` 即 kernel 无缺陷），残差归到 op 层 affine 的 dtype；op 层 affine 应在 **fp32** 计算并**单次** `.to(dtype)` 舍回，否则 fp16 的乘-加双重舍入会在 `n·w+b≈0` 的元素上以 ulp(|n·w|) 计误差、击穿 atol=rtol=1e-3（本例 21 用例中 7 例、每例仅 1–3 元素超差）。
- **kernel 级 L0/L1 门禁用单位 affine 会漏掉 op 路径的 affine trap**：凡 wrapper/op 把 affine 挪到 kernel 之外施加的算子（GroupNorm 的 per-channel `(C,)` affine 不匹配 kernel 的 row-broadcast `(D,)`），Stage 3 的 unit-affine 门禁对 fp16 是"假绿"；交付前须用**真实 per-channel affine** 的 op 级用例（或集成 report）覆盖一次。
- **Stage 5 的默认编辑范围不含 `ops/`**：若证据把根因定位到 op 封装层的数值路径，闭环内无合法可改文件——应报告 + 交回 conductor 授权，不得为了通过测试擅自扩大范围；获得显式授权后也只做**单点最小 diff**（本例仅 1 个表达式），备份进 `history_version/` 并在 `integration_log.md` 记录授权来源与前后结果。
