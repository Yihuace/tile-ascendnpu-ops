# RETROSPECTIVE.md 章节规范（canonical schema）

> 本文件是各 Stage Subagent 写入算子目录 `RETROSPECTIVE.md` 的 canonical 规范，也是 evolver 解析该文件的依据。各 stage skill（design / review / develop）与 integrator agent 内嵌了同款最小模板——以本文件为准。

## 1. 文件位置与生命周期

| 场景 | 路径 |
|------|------|
| new_op / migration-plain / optimize | `examples/{project}/{op}/RETROSPECTIVE.md` |
| migration-harness（Stage 1/2/3，逐函数） | `examples/{op_slug}/{func}/RETROSPECTIVE.md`（每函数一份） |
| migration-harness（Stage 5，op 级） | `examples/{op_slug}/RETROSPECTIVE.md`（集成复盘为 op 级单份——集成问题不归属单个函数） |

- **追加式日志**：每个 Stage 执行（含修订/重试的终态返回）追加一个章节，不覆盖历史章节。写入前先 Read 既有内容，整文件 Write 回（追加新章节）。
- **Stage 4 例外**：调优复盘不写本文件，写 `perf_opt/opt_log.md` 的 `## Skill Retrospective` 章节（既有机制不变）。
- **消费者**：conductor（session 教训搬运，只读「Transferable Lessons」小节）与 `tilelang-skill-evolver`（终态蒸馏，读全文）。同任务其他 Stage **不读**本文件（避免串味）。

## 2. 写入时机

| Stage | 写入时机 | 说明 |
|-------|---------|------|
| 1 design | 每次 DESIGN.md 产出后（first_design 与 revision 均写） | revision 模式的复盘价值最高（设计判断被推翻的过程） |
| 2 review | REVIEW.md 结论给出后（通过与不通过均写） | 不通过的价值已随 REVIEW.md 流转，复盘重点是**检视维度自身**的缺口 |
| 3 develop | 返回 `[PRECISION_PASS]` 或 `[DESIGN_ERROR]` 前 | PASS 复盘覆盖整个 attempt 链（含中间失败中发现的 API 行为实证）；`[PRECISION_FAIL]` / 运行失败不写（失败摘要已由 conductor 路由） |
| 5 integrate | 返回 `INTEGRATE_COMPLETED` 或 `[DESIGN_ERROR]` 前 | `[INTEGRATE_FAIL]` 不写（将被重调度） |

## 3. 章节模板

```markdown
## Stage {N} ({stage_name}) Retrospective — {mode|attempt} — {YYYY-MM-DD}

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| none | none | none | none | - |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| none | - | - | - | - | - |

### Transferable Lessons（可迁移教训）

- none
```

字段规范：

| 字段 | 规范 |
|------|------|
| `area`（流程问题表） | `Design-flow / Review-flow / Develop-flow / Integrate-flow / Info-source / Constraint-doc / logging / other` |
| `issue` | 本次暴露的 skill/agent 流程问题；无则 `none` |
| `evidence` | 本任务工件内的证据定位（文件#章节/行）；**不得为空**（除 none 行） |
| `suggested_doc_change` | 建议改哪个文件、怎么改（一句话） |
| `vp_type` | `D / P / R / C`（判定口径：有数字的是 D，有方法的是 P，改流程的是 R，可参考的是 C；详见 tilelang-skill-evolution skill references/distillation-rules.md §3） |
| `title`（价值点表） | 一句话，含可检索关键词 |
| `evidence`（价值点表） | 溯源路径（`文件#定位`） |
| `repro` | 复现命令或复现条件；不可复现写 `none` |
| `toolchain_stamp` | tilelang build/commit + 设备 + CANN 版本；无法确定如实标注 |
| `target_doc` | 建议的目标文件路径（不确定可写 `pattern-library / bottleneck-patterns / 待定`） |
| Transferable Lessons | 一条一行，写给**本迁移任务后续函数**的实施者；内容须自包含（不依赖本函数上下文可理解）；无则 `none` |

## 4. 质量红线

1. **无则如实写 none**——禁止为了"有产出"硬凑价值点；单任务偶然现象必须绑定本任务证据，禁止写成通用规则。
2. **D 类（实测/实证）必须带三件套**（evidence + repro + toolchain_stamp）——缺失时 evolver 会降级处理，但源头写全可减少一轮往返。
3. **负面发现优先**——"此路不通 + 原因 + 证据"（如 API 实际行为与文档不符、某形态 mis-compile）是密度最高的价值点。
4. **Transferable Lessons 不写敏感于本函数的细节**（如"本函数 UB 预算算错"）——写跨函数仍成立的教训（如"vbrc 标量→shared 不可用，用 T.copy 替代"）。
