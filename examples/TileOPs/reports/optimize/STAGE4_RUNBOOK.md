# Stage-4 夜间批量执行 Runbook（plan B：主会话编排 + 真派 optimizer）

> 背景：conductor 作为 subagent 派发会被 depth 限制降级为“自己干活”；TUI 切 primary 失败且不再尝试重启。
> 本 runbook 由主会话（build agent）机械执行，**执行手一律是 `@tilelang-op-optimizer` 子会话**（干净上下文），
> 规则与 `tilelang-op-conductor` + `conductor-scenarios/optimize.md` 等价。
> 队列与进度唯一事实源：`examples/TileOPs/reports/optimize/STAGE4_QUEUE.md`。

## 0.0 模型口径（用户指令 2026-10-18）

所有子 agent（conductor / optimizer / evolver / designer / developer / reviewer / integrator / scaffolder）的 `model` 统一为 **`deepseek/deepseek-flash`**（原为 `deepseek/deepseek-v4-pro`，已批量改于 `.opencode/agents/*.md`）。
- 生效边界：新派发的子会话；已在运行中的子会话不受影响（定义在启动时加载）。
- 影响：长任务（Cube/多 kernel）上下文更短、推理更弱 → 派发 prompt 需**更结构化**（明确步骤、失败路由、必须自查 gate 4），止损线保持不变。

## 0. 角色与职责

| 角色 | 职责 | 禁区 |
|---|---|---|
| 主会话（编排） | statectl 状态机、门禁、豁免判定、精度回归 gate、wrapper 翻转、report 验收、调度 optimizer/evolver、写报告 | 不改 kernel / 不测性能（不亲自做优化） |
| `@tilelang-op-optimizer` | 诊断、候选实现、msprof 采数、A/B、写 `perf_opt/{Op}.py` + `opt_log.md` + `perf_records.jsonl` + `OPTIMIZE_REPORT.md` | 不读写 `.stage_state.json`；不改基准 `{op}.py`；不碰 NPU 之外的 owner 文件 |
| `@tilelang-skill-evolver` | 每 3 个算子批量蒸馏一次（避免 25min/次拖慢夜间吞吐），末轮必跑 | 不改 agent/skill 规则文件（Tier 2 入队） |

## 1. 单算子 SOP（严格顺序）

1. **体检**：`statectl verify --dir <op_dir>`；再 `statectl show`。
2. **入场**：
   - 无 `.stage_state.json`：`statectl init --dir <op_dir> --project <p> --op <Op> --scenario optimize --kernel-path <kernel> [--requirement "<口径>"]` → `statectl start 4`
   - `phase=DONE`：`statectl start 4 --extend`（重开 Stage 4）
   - `phase=TUNING`：直接续（不 fail、不计重试）
3. **预注入**：`python3 .agents/tools/kb_search.py "<op family dtype 症状>" --json --top 5` + `kb_stale_check.py --json`；**必查 `CONST-capacity-910B2C`**（UB=192KB / 20 AIC / K-tile 容量启发式 150KB 偏保守）。
4. **派发 optimizer**（prompt 六要素）：算子目录与 benchmark 入口、性能目标与达标线、豁免/止损口径、上机纪律、交付物清单、返回格式（含候选 vs best 对比表 + perf_records 行数）。
   - **必加一条**：返回前自己跑 `statectl gate 4 --dir <op_dir>`（只读，允许）直到 `pass: true`。已两次踩坑（FALN / BatchNormBwd）：final 记录未降级、Final 表列数不对、多 kernel 的 `artifact_path` 与状态 `kernel_opt_py_path` 不一致——全部可由自查提前发现。
   - **多 kernel 算子**：`kernel_opt_py_path` 必须是单一文件（gate 硬约束），三变体应合并为 `perf_opt/{Op}.py` 单一模块，final 记录统一指向它。
5. **回收判定**（我执行，不委托）：
   - `statectl gate 4 --dir <op_dir>` 必须 `pass:true`
   - 精度回归：`python perf_opt/{Op}.py --level all`，**L0/L1 失败 = 阻塞**
   - 有净收益 → 翻转 wrapper 切换块 → `python -m tileops.reporting.cli run --op <Op> --prof-mode msprof` 验收（正确性必须全过）
   - 无净收益/天花板 → **不翻 wrapper**，留档 ceiling 证据
   - `statectl complete 4`；`statectl timeline-summary` 存档
6. **报告**：确保 `examples/TileOPs/reports/optimize/{Op}/OPTIMIZE_REPORT.md` 存在且含结论/对比表/精度/产物/时间。
7. **进度**：更新 `STAGE4_QUEUE.md` 表格状态 + 进度日志，然后立即进入下一算子（不空转、不询问用户）。

## 2. 失败路由（夜间自动执行，不阻塞）

| 现象 | 动作 | 上限 |
|---|---|---|
| 编译/运行失败 | 带 `last_failure_summary` 重派 optimizer（同 op，新会话） | 2 次 |
| 精度回归 L0/L1 失败 | 重派 optimizer `mode=precision_fix`（只修回归，不重走轮次） | 2 次 |
| 无净收益 / ceiling 已举证 | 收束：`complete 4` + 留档 + 下一算子 | — |
| 单算子超时（Vector >60min / Cube >180min） | 取当前最优收束，报告写未竟项 | — |
| 门禁失败（gate 4 failures 非空） | `statectl fail 4` → `statectl start 4` → 带 gate 错误重派 | 2 次 |
| NPU 锁等待超时 | 等 60s 重试一次；仍失败则记录并跳到下一算子，末尾补跑 | — |
| 子会话无回执/崩溃 | 记录进度，重派一次；再失败则该 op 标记 `BLOCKED_ENV` 并继续 | 1 次 |

## 3. 豁免与止损（用户口径，硬约束）

1. **微型 shape**：Task Duration ≤5µs 或固定启动开销 ≥50% → 豁免，报告单列。
2. **结构上限已举证**：advice=`latency bound:pipeline caused` + 明确 CG/API 缺口 + block/UB 到顶 → 豁免。
3. **判据失配**：`aiv_vec_ratio ≥0.9`（或 computility=Vec_FP32 打满）→ 改用向量利用率判据，但仍要求 Task Duration 下降。
4. 达标率分母只含可调 shape，同时报「全 shape / 可调 shape」两个数。
5. 单算子 ≤2 轮结构深挖、单 shape ≤2 次；连续两轮 Task Duration 不降即判 ceiling。
6. 硬底线：Task Duration 必须变短；不达标不硬凑、不伪达标。

## 4. 夜间自主规则

- 不向用户提问；遇到本应询问的决策，取本 runbook 的默认动作并在 `STAGE4_QUEUE.md` 记录「决策点 + 理由」。
- 每个算子完成后立即派下一个，**不等待用户确认**。
- 蒸馏每 3 个算子一批；全部完成后跑最终一批 + 汇总报告。
- 任何一个算子连续两次异常即跳过（记录原因），不让单点卡住整条队列。

## 5. 完成后交付

- 更新后的 `STAGE4_QUEUE.md`（含每算子结果与豁免明细）
- 每个算子的 `OPTIMIZE_REPORT.md`
- 全批汇总：达标统计（全 shape / 可调 shape）、收益榜、ceiling 清单、能力缺口增量、待批 Tier 2 提案
