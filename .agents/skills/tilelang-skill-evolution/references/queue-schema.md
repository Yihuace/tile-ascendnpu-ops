# 提案队列字段规范（.agents/evolution/queue.md）

> 本文件是 queue.md 条目的 canonical schema。evolver 写入 queue 时严格遵守；`queue.md` 头部只保留字段速查，以本文件为准。

## 1. 条目结构

每条提案为一个 `## VP-{YYYY}-{NNNN}` 二级标题 + YAML 风格字段块：

```markdown
## VP-2026-0001
- type: P
- title: 窗口算子 strided load 的 sW≠1 跨步系数阻碍向量化，换轴收益 5.6x
- evidence:
  - examples/{project}/{op}/perf_opt/opt_log.md#round-3
  - examples/{project}/{op}/perf_opt/profiles/v3a/
- repro: python examples/{project}/{op}/perf_opt/{op}.py --level L0
- toolchain_stamp: tilelang@2026-08-28 build, Ascend910B2C, CANN 8.x
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 条目 BP-STRIDE-01：
    | 现象 | 动作 | 判据 |
    |---|---|---|
    | msprof 热点段标量占比>50% 且累加维存在 sW≠1 跨步 | 强制换向量化轴评估（对照 pattern-library §1） | 换轴实测收益 5.6x（见证据） |
- status: pending
- confirmations: 1/2
- created_by: task {task_id} 2026-09-01
- decided_by: -
- decided_note: -
```

## 2. 字段规范

> **证据语义（ED-A，2026-09-10 起）**：`evidence` 是 **provenance**（来源任务与出处，**允许失效**——过程文件不随交付件上库，记录出处即可，不做存在性核验）；`repro` 是**可执行验证**（指向知识域内自包含脚本，须存在且可执行——条目的"怎么验证"）。两者的分离使提案不因工作区清理而死链。

| 字段 | 必填 | 规范 |
|------|------|------|
| `proposal_id` | ✅ | `VP-{YYYY}-{NNNN}`，evolver 按年递增分配，不复用已决 id |
| `type` | ✅ | `D / P / R / C`（D/C 通常不入队直接合入；入队的 D = 三件套不全的降级条目） |
| `title` | ✅ | 一句话，含可检索关键词（算子类别 / API 名 / 现象） |
| `evidence` | ✅ | provenance 列表：来源 task_id + 出处描述（如 `task xxx 的 opt_log round 3` / 路径 `#` 定位——路径允许失效）；Tier 1 第二次确认时**追加**新任务的证据，不覆盖 |
| `repro` | ⭕ | 指向知识域自包含脚本路径（`pattern-library/repro/<条目ID>.py`，须存在且可执行，`repro_runner.py` 批量重验）；无法当场最小化的写 `repro-missing`（待同族任务补，ED-E）；不可复现的 P 类写 `none` 并在 title 标注。**不再接受任务工作区文件路径作为复现手段**（provenance 字段除外） |
| `toolchain_stamp` | ✅ | `tilelang build/commit + 设备 + CANN 版本`；无法确定时写 `版本戳缺失（来源任务 {task_id}）` |
| `target_doc` | ✅ | 目标文件仓库相对路径；R 类必须指向具体文件（禁止"待定"）；pattern-library 相关目标指向 `pattern-library/` 主题文件（不再指向旧单文件） |
| `delta` | ✅ | **结构化 diff 提案**：五种动作之一（add/update/consolidate/negate/deprecate）+ 完整条目正文（Tier 1）或 old/new 文本 + 定位锚文本（Tier 2，锚文本须为目标文件中真实存在的原文片段） |
| `status` | ✅ | `pending / verified / merged / rejected / expired / conflict` |
| `confirmations` | ✅ | `{n}/2`；仅 Tier 1 有意义（Tier 2 恒 `-/-`）；两次证据须来自**不同任务**。**强证据快速通道（E-3）**：P 类提案已有完整三件套（provenance + repro + stamp）且在**另一上下文复现成功**（`repro_runner.py` 重跑该 repro 通过 / `ab_test.py` 复现测量关系 / opbench 回放命中）→ 计为 2/2——不降低证据标准，只拓宽证据形态（人工审批权不变，仍适用 Tier 2） |
| `created_by` | ✅ | `task {task_id} {日期}`；后续确认追加 `confirmed_by: task {task_id} {日期}`（快速通道确认标注 `confirmed_by: repro-runner {repro路径} {日期}`） |
| `decided_by` | ⭕ | 终态必填：`evolver`（Tier 1 合入/过期）或 `human`（Tier 2 批准/否决、conflict 裁决） |
| `decided_note` | ⭕ | rejected 必填原因；merged 记录合入锚点；conflict 记录双方证据摘要 |

## 3. Tier 2（R 类）diff 提案的额外要求

R 类提案的 `delta` 必须达到"apply 模式可直接执行"的深度：

```markdown
- delta: |
    动作: update（新增小节）
    目标: .agents/skills/tilelang-op-develop/SKILL.md
    定位锚: "### Phase 5：三态判定与返回"（在该节之前插入）
    new 文本: |
      ### Phase 4.5：重试前检索陷阱表
      …（完整新文本）
    动机: Stage 3 retry_impl 平均 2.3 次/任务，其中 60% 命中 pattern-library §2 已有陷阱条目（evidence）
```

- `定位锚` 必须是目标文件中的**唯一原文片段**；apply 时找不到锚 → 条目转 `conflict`，不强行套用。
- `new 文本` / `old→new` 必须完整可粘贴，禁止"此处略"。
- 一条提案只改一个文件的一处；多处修改拆成多条。

## 4. 归档规则

- `## Pending` 区只保留 `pending` / `verified` / `conflict` 活跃条目。
- `merged` / `rejected` / `expired` 移入 `## Decided（merged / rejected / expired / conflict 归档）` 区，保留全部字段。
- 归档条目不删除、不编辑正文（仅状态与 decided_* 字段更新）。
