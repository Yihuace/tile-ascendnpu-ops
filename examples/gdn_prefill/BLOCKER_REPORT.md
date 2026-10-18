# GatedDeltaNetPrefillFwdOp — R8 阻塞归档报告

> 交付名 `GatedDeltaNetPrefillFwdOp`（基线 `GDNFwdOp`）；family `linear_attention`；op_slug `gdn_prefill`。
> 结论：**最后一个算子未交付（18/19）**。函数 2/6 `prefill_blocksolve_A_bthd` 在 Stage 3 被
> CANN 9.0.0 的 `bishengir-compile` 缺陷 **CG-2026-0003** 阻断；按任务书 **R8「不可行即登记并停下，
> 不得改语义绕过」** 停止并归档，未做算法/语义绕过。
> 归档时间：2026-10-17。

---

## 1. 身份与范围

| 项 | 值 |
|---|---|
| 算子 | `GatedDeltaNetPrefillFwdOp`（基线 `GDNFwdOp`，GPU kernel `kernels/linear_attention/gdn/prefill.py:29`） |
| GPU 源仓 | `/mnt/workspace/TileOPs-main`（基线 `1238a9f`） |
| 当前函数 | 2/6 `prefill_blocksolve_A_bthd`（源 `gdn/prefill_prepare.py` `_prefill_blocksolve_A_bthd_tl` L158 / wrapper L654） |
| 函数目录 | `examples/gdn_prefill/prefill_blocksolve_A_bthd/` |
| 迁移模式 | `migration` + `harness`，Stage 4 跳过 |
| 状态 | 函数级 `phase=FAILED`、`failure_reason=BLOCKED_ENVIRONMENT`；op 级 `phase=FAILED` |
| 工具链 | tilelang `0.1.2`（dev root，自建 AscendNPU-IR `31f6903`）/ torch `2.7.1+cpu` / torch_npu `2.7.1.post4` / **CANN `9.0.0`**（`bishengir 1.1.0 / 428ab8fdab46` + `hivmc 0.2.0`）/ Ascend910B3（可见设备 6）/ Python 3.11.4 |

范围沿用算子级 `examples/gdn_prefill/DESIGN.md` §1：等长 prefill、`chunk_size=64`、`dim_k ∈ {64,128}`、
`use_gate=True`、`l2norm/beta_sigmoid/allow_neg_eigval=False`、支持序列最后一个不完整 chunk。

---

## 2. 迁移流程证据

### 2.1 状态机（全部经 `statectl`）

```
statectl init --dir examples/gdn_prefill/prefill_blocksolve_A_bthd \
  --scenario migration --migration-mode harness        # stage_plan=[1,2,3]
statectl start 1 → complete 1                          # DESIGN gate pass
statectl fail 2 --reason design_revision (×2)          # 两轮检视不通过 → retry_count=2
statectl start 3 → fail 3 --fail-type runtime          # 编译失败 1/5
statectl fail 3 --blocked-code BLOCKED_ENVIRONMENT     # R8 终止（不耗 Stage 3 重试预算）
statectl migration func ... prefill_blocksolve_A_bthd --status failed
statectl migration phase ... --phase FAILED --reason "BLOCKED_ENVIRONMENT: CG-2026-0003 ..."
```

时间线（`statectl timeline-summary`）：`first_ts=2026-10-17T07:04:09Z`，`last_ts=2026-10-17T09:28:00Z`，
`dispatches=7`，`elapsed_s≈8631`；Stage 1 attempts 3（首版 + 2 次修订）、Stage 2 attempts 3（2 次不通过 + 1 次通过）、
Stage 3 attempts 1（+ 1 次 R8 终止记录）。

### 2.2 逐阶段工件与门禁

| Stage | 工件 | 门禁 |
|---|---|---|
| 1 设计 | `DESIGN.md`（1525 行，v2）；修订备份 `history_version/design_v0.md`、`design_v1.md`；`verify_equiv.py`；6 个设计探针；`history_version/design_schedule_sim.py` | `gate 1`：`pass=true, failures=[]`（三次均通过） |
| 2 检视 | `REVIEW.md`（v2 结论 **`结论: 通过`**；前两版备份 `history_version/review_v1.md`） | `gate 2`：`pass=true, failures=[]`；迁移 9 维度齐全 |
| 3 开发 | `prefill_blocksolve_A_bthd.py`（802 行，kernel + golden + L0/L1/L2/Boundary + L0-h dump） | AST `gate 3`：`pass=true`；**L0/L1 未全过 → 不满足 Stage 3 出口** |

关键设计事实（检视通过版本）：
- 算法：`A = (I+S_sys)^{-1} = (I-N)^{-1} = Π_{i=0..5}(I+N^{2^i})`（`N=-S_sys`，log-doubling，12 个链 gemm + Gram）；
- 单 `T.Scope("Cube")` + 单 `T.Scope("Vector")`，2 slot × 6 flag 双向握手；
- 尾 chunk 确定性归零（R3-a 块首清零 + R3-b 每任务 β 归零）；
- 落盘 `fixpipe→y_blk→UB→strided region`，exactly-once 调度经 2048 例仿真。

设计探针 / 机检（检视方独立复跑）：
- `verify_equiv.py`：A1–A7 全 `EQUIV_PASS`，worst viol 0；符号裁决 `2.776e-17`（`I+S`）vs `4.361e-1`（`I-S`）；
- `design_schedule_sim.py`：17 个 in-scope 形状 + `T=1..2048` 穷举，落盘 exactly-once、6 flag `#set==#wait`、无死锁；
- `design_calc_check.py`：`r3_metrics` / `l0c_budget` pass（`ub_budget`/`core_split` 为解析器 skip，文档内有人工核算表）。

---

## 3. 正确性事实（Stage 3 实跑，未掺性能）

命令（工作目录 `examples/gdn_prefill/prefill_blocksolve_A_bthd`）：
`python3 prefill_blocksolve_A_bthd.py --level L0`；归档复跑日志 `logs/L0_archive.log`。

| 用例 | shape | 结果 |
|---|---|---|
| L0-a | S=64,H=1,dk=64,fp16 | **PASS** `max_abs_err=3.773e-05` / rel `4.512e-04`，`nan=0` |
| L0-b | S=64,H=1,dk=128,fp16 | **PASS** `5.278e-05` / `4.488e-04` |
| L0-c | S=64,H=1,dk=64,bf16 | **PASS** `2.317e-04` / `3.618e-03`（门限 4e-2） |
| L0-d | S=100（尾 chunk 36 行） | **PASS** `2.780e-05`；padding 28 行 `== e_i` 精确（err 0.0），`nan=0` |
| L0-e | S=64,H=2 | **编译失败**（`err code -11`，bishengir SIGSEGV） |
| L0-f* | β 含 0 行 + g 跨度 −40 | PASS `3.400e-05` |
| L0-g* | S=64,H=16,dk=128（行距 2048B） | PASS `3.025e-05` |
| L0-h* / L0-h-tail* | dump `N/gc/gr/β/bn`（valid=64/36） | PASS `n_err 6.616e-09 / 6.297e-09`；越界行全 0；`N` 严格上三角恒 0 |
| L0-i* | S=1600（T=25,K=20,NPER=2） | **编译失败** |
| L1-a* / L1-b* | S=1024,H=4 / S=3000,H=16，dk=128 | **编译失败** |
| Boundary* | S=56/64/65，H=2 | **编译失败** |
| L2* | S=4096，kh∈{1,16}，dk∈{64,128}，fp16/bf16 | **编译失败** |

> `*` 为 Stage 3 developer 运行记录（同一脚本、同一工具链）；无 `*` 为归档时的独立复跑（日志落盘）。
> 归档复跑在 L0-e 处被编译器 SIGSEGV 中止（exit=1），与 developer 记录一致。

- golden 独立性：torch fp64 `torch.linalg.solve(I+S_sys)`；与源分块前代算法的 fp64 实现交叉校验残差 **`3.469e-18`**。
- 判定：**算法与实现侧在可编译形状上数值正确**；但 L1/L2/Boundary 无法编译，不能给 `[PRECISION_PASS]`。

---

## 4. 性能事实

**未测**。Stage 4 在 harness 迁移中跳过；Stage 5 未启动，无 `tileops-report` run、无 msprof 数据。
不提供任何估算或照搬数字。

---

## 5. 未决项与风险

### 5.1 阻塞根因（capability gap，已登记）

`.agents/evolution/capability-gaps.md` **CG-2026-0003**（`occurrences 3`，`status: recurring`）：
- `bishengir-compile` 的 `ConvertHFusionToHIVM` 对 buffer 形态 `hivm.hir.mmadL1` 取 `op.getResult(0)`
  （0 result）越界 → SIGSEGV / 挂死；CANN 发布分支（v1.1.0-post2 / v1.2.0）未回合主线修复。
- 本任务实测：同一份源码（13×`mmadL1` + 3×`fixpipe`）按 `(total_tokens, head)` 分档，
  **6 个形状可编译且数值正确，11 个形状必崩**（`S=64,H=2`、`S=1600`、`S=1024,H=4`、`S=3000,H=16`、`S=4096,H=16`、`S=56/65,H=2`）。
- IR 级二分：删 1 个 `nd2nz`（R3-a）→ 通过；删 6 级 doubling 链 → 通过；给 Vector 段加 2 个 UB buffer → 原本通过的形状转崩。
  ⇒ 触发阈值是 **IR 规模/形态**，无稳定 kernel 级绕法；通过形状亦存在被静默误编译的风险（fp64 对拍是唯一护栏）。
- 回归看门狗：`repro/TRAP-hfusion-mmadl1-segv.py`（good shape + blocked shape 双建；末行 verdict）。

### 5.2 未决项

1. **函数 2 未 done** → op 无法进入 Stage 5；`GatedDeltaNetPrefillFwdOp` 维持 **18/19 未交付**。
2. 函数 4 `correct_initial_states`、5 `fused_gdr_h`、6 `fused_gdr_fwd` 仍 `pending`（函数 1/3 已 done）。
3. 通过编译的 6 个形状也需在工具链修复后重跑全量，确认无静默误编译。
4. `head=16` strided 落盘时延、`dim_k=128` 6 级 fp32 长链、flag 计数语义仍是文档级/局部实测，工具链修复后须由 L0/L1 覆盖。

### 5.3 恢复入口（工具链修复后）

```bash
cd examples/gdn_prefill/prefill_blocksolve_A_bthd
python3 prefill_blocksolve_A_bthd.py --level all      # 无需改代码
python3 .agents/tools/statectl.py start 3 --dir examples/gdn_prefill/prefill_blocksolve_A_bthd
python3 .agents/tools/statectl.py complete 3 --dir examples/gdn_prefill/prefill_blocksolve_A_bthd
python3 .agents/tools/statectl.py migration phase --dir \
  examples/TileOPs/tileops/kernels/linear_attention/gdn_prefill --phase DEV_LOOP   # 需按终态规则重开
```

可行工具链路径（来自 gap 条目 `workaround`）：
1. 取得与 Oct 树同代的 `hivmc`（原 CANN 8.5.2 内那份，本容器不可得）；
2. 自建 `v1.1.0-post2` 并回合主线修复（删除/保护 `getResult(0)` walk）后重编 tilelang；
3. 等 CANN 修版工具链。

---

### 5.4 2026-10-18 拆分/分组/核数判定性探针

- 3 级变体使 `S=1024,H=4,dk=128` **编译并运行成功**（原版 6 级必崩）；但 `S=64,H=2,dk=64`
  在 6/3/1 级、`group=1/2`、`cores=1/默认` 下均失败（SIGSEGV 或挂死）。
- 结论：IR 规模不是唯一变量；小 shape 存在独立于链长/分组/核数的 codegen 触发点，
  **拆 kernel 只能解锁大 shape，不能收口 GDN**。要 19/19 需设计级替代（纯 Vector 前处理 /
  换 tile 布局 / 等编译器修复）。
- 证据与复现：`probe/experiments/18_level_split/PROBE_20261018.md`。

## 6. 复核入口

```bash
# 状态
python3 .agents/tools/statectl.py show --dir examples/gdn_prefill/prefill_blocksolve_A_bthd
python3 .agents/tools/statectl.py migration show --dir examples/TileOPs/tileops/kernels/linear_attention/gdn_prefill
python3 .agents/tools/statectl.py timeline-summary --dir examples/gdn_prefill/prefill_blocksolve_A_bthd

# 设计/检视机检
python3 .agents/tools/statectl.py gate 1 --dir examples/gdn_prefill/prefill_blocksolve_A_bthd
python3 .agents/tools/statectl.py gate 2 --dir examples/gdn_prefill/prefill_blocksolve_A_bthd
python3 .agents/tools/design_calc_check.py --design examples/gdn_prefill/prefill_blocksolve_A_bthd/DESIGN.md
python3 examples/gdn_prefill/prefill_blocksolve_A_bthd/verify_equiv.py
python3 examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_schedule_sim.py

# 复现阻塞
python3 examples/gdn_prefill/prefill_blocksolve_A_bthd/repro/TRAP-hfusion-mmadl1-segv.py
# 归档证据
cat examples/gdn_prefill/prefill_blocksolve_A_bthd/logs/L0_archive.log
```

### 工件清单

| 类别 | 路径 |
|---|---|
| 设计 | `examples/gdn_prefill/prefill_blocksolve_A_bthd/DESIGN.md` |
| 检视 | `examples/gdn_prefill/prefill_blocksolve_A_bthd/REVIEW.md`（`结论: 通过`） |
| 实现 | `examples/gdn_prefill/prefill_blocksolve_A_bthd/prefill_blocksolve_A_bthd.py` |
| 等价性 | `.../verify_equiv.py` |
| 调度仿真 | `.../history_version/design_schedule_sim.py` |
| 设计探针 | `.../history_version/design_probe_*.py` |
| 复现看门狗 | `.../repro/TRAP-hfusion-mmadl1-segv.py` |
| 复跑日志 | `.../logs/L0_archive.log` |
| 复盘 | `.../RETROSPECTIVE.md` |
| gap | `.agents/evolution/capability-gaps.md` CG-2026-0003 |
| 算子级设计 | `examples/gdn_prefill/DESIGN.md`（rev.2） |
