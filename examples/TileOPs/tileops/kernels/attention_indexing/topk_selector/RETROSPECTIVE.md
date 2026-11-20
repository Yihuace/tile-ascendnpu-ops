# RETROSPECTIVE.md — `topk_selector` / `_topk_selector_kernel`

> 追加式日志:每次 Stage 执行追加一个章节,不覆盖历史章节。
> **本算子的 Stage 3 出口为「阻塞」,不产出 `[PRECISION_PASS]`,不得按"已交付"计。**

---

## Stage 1 (design) Retrospective — first_design — 2026-10-16

本单元把既有的非 skill 格式设计(421 行)整合为 skill 要求的章节结构,并**沿用**其全部实测基线(不重复造矩阵)。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | **skill 的 9 维度设计与门禁都假设"最终会有实现"**。对**已知阻塞**的算子,若干维度(内存预算、Tiling、伪代码、分核方案)在实现不存在时**只能填"未定"或被 `skip`**。流程未规定"阻塞算子如何产出合规设计" | 本任务:门禁三项 `skip`(`ub_budget`/`l0c_budget`/`core_split`),其中两项的成因是**设计主动声明"不做核算/无法给出"**;§3.3 明确**不提供可运行伪代码** | 为**阻塞型算子**增加一条明确分支:允许 §3.3/§4.3/§5.3 以「**主动声明不核算 + 给出否定性约束**」代替具体数值,并要求把"为何不能给"写成正文;门禁在这类文档上应把 `skip` 的成因回显出来 | R |
| Design-flow | **阻塞证据的 repro 缺少"忠实性"要求** —— 本任务的首次 repro 用**弱体**(单条 `vcmp`)作循环体,结果**未复现阻塞**,一度得出"阻塞已消失"的错误结论 | 本任务:`--body trivial` → 两个 wall 都"编过并运行";换成 `--body full`(完整压缩链)→ 两个 wall 都 `DiagnosticError`。**差异完全来自循环体是否忠实** | 在 `tilelang-op-develop`/`design-review` 的 repro 要求里加一条:**阻塞类 repro 必须论证其"约简的忠实性"** —— 即被约简掉的成分对结论无影响;无法论证时,须保留**对照体**(强/弱两版)并同时跑两者 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **"循环 × 压缩链"才爆,单条 `vcmp` 不爆**:`topk_selector` 的循环阻塞**依赖循环体内容** —— 体换成单条 `vcmp` 即编过;体为完整压缩链(`vcmp → cumsum → vselect → scatter`)时,静态展开 48 轮与 `T.serial(4)×T.serial(12)` **均在编译期报 `DiagnosticError`** | D | `repro/TRAP-blocker-loop-structure.py` 的 `--body full/trivial` 对照(同一脚本、同一 wall,只有体不同) | `env -u PYTHONPATH <venv>/bin/python -u examples/topk_selector/_topk_selector_kernel/repro/TRAP-blocker-loop-structure.py --wall w1 --body full`(再跑 `--body trivial` 对照) | tilelang `0.1.2+npuir+7fc4a3c8` / Ascend910B2C / CANN 9.0.0 | `pattern-library/traps-compiler.md`(循环 × 压缩链) |
| **阻塞类 repro 必须自带"忠实性论证 + 对照体"**,否则"未复现"会被误读为"阻塞已消失" | P | 本任务首次 repro(弱体)给出 `NOT BLOCKED`,结论错误;加入 `--body trivial` 对照后才看清差异来源 | 同上(两个 body 各跑一次,对照即得) | 同上 | `tilelang-op-develop` 的 repro 要求 |

### Transferable Lessons（可迁移教训）

- **"未复现"不等于"已消失"——先问约简是否忠实**:本任务的首次 repro 用单条 `vcmp` 当循环体,两个 wall 都"编过并运行",一度得出"阻塞消失"的结论。换成完整压缩链后**双双复现**。⇒ **把复杂体约简成"最小可编译单元"时,很可能顺手把被测的性质也约简掉了**;凡阻塞类 repro,**必须论证被约简掉的成分对结论无影响**,否则要保留强/弱两版对照。
- **同一现象的"症状"要按本轮实测精确化,不沿用前序记录的措辞**:本设计原先写的是 `ABORT(ERR99999)` 与 core dump(前序会话记录);本轮亲自复现的症状是**编译期 `DiagnosticError`**。三者同属"编译期失败",但**只有 `DiagnosticError` 是本轮实测的** —— 应并列呈现并标注来源,不能让前序措辞冒充本轮证据。
- **阻塞型算子的设计文档**允许**主动声明"不核算"**,但要给出**否定性约束**才有价值:本任务 §4.3 不做容量核算,却给出"单行 fp32 整维驻留需 256 KB > 192 KB ⇒ 否定整维驻留思路"这一条**有信息量的否定**。**"不知道"要变成"知道某条路走不通"。**
- **`对于 blocked 算子,"设计通过"与"可实现"必须解耦**:本设计的 Stage 2 结论是 `通过`,但 §11 同时声明"**不得按已交付计**、Stage 3 出口为阻塞"。**两个结论并列并不矛盾** —— 前者评"文档是否成立",后者报"工程是否可交付"。
