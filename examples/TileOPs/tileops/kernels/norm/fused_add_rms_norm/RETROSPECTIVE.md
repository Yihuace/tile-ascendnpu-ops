# RETROSPECTIVE.md — `fused_add_rms_norm`(op 级)

> 追加式日志。逐函数的 Stage 1/2/3 复盘见
> `examples/fused_add_rms_norm/fused_add_rms_norm_kernel/RETROSPECTIVE.md`
> (同一份亦复制到 `examples/fused_add_rms_norm/_fused_add_rms_norm_kernel/RETROSPECTIVE.md`)。

---

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-16

- **verdict**:`INTEGRATE_COMPLETED`
- **report_status**:**`passed`** ⭐ —— **本批第一个非 `partial` 的算子**
- **attempts_used**:1(集成侧);**上游 Stage 3 另有 1 次精度回退**
- **正确性**:**12 / 12**;benchmark 失败数:**0**

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | **集成脚本的设计快照查找目录与产物目录绑定**,目录一旦改名,`design_docs` **静默变 `null`** 而集成仍报 `[ok]` —— 本批**第二次**踩到(先在 `vector_norm`,这里又) | 本任务首次集成:`"design_docs": {"_fused_add_rms_norm_kernel": null}`;把 `DESIGN.md` 复制到函数名目录后重跑 → 非空 | 同 `vector_norm` 的提案:**集成脚本的 ok 判定应把"设计快照缺失"报为 warning 或非零 `design_missing`**。该提案现**已触发 2 次** | R |
| Integrate-flow | **`report_status` 的两态语义未被 harness 明文区分**:`passed` 与 `partial` 的差别是"benchmark 有无失败用例",而**失败用例的成因(采集侧能力边界 vs 算子缺陷)不在状态里**。逐元素族 5 个算子都是 `partial`,容易被误读为"这 5 个质量不如 norm 族" | 本任务:`status=passed`、`benchmark_failed=0`;逐元素族 `partial`、`benchmark_failed=2`(均为 268M 采集侧失败) | 在 harness 的 report 判读段写明:**`partial` 只表示"benchmark 阶段有失败用例",须结合失败用例的标签与报错判定成因**;并建议把"采采集侧/规模类失败"从 `partial` 中区分出来 | R |

### Transferable Lessons（可迁移教训）

- **`report_status=passed` 与 `partial` 的差别只是"benchmark 有无失败用例",不直接反映算子质量**:`fused_add_rms_norm` 是 `passed`,但它在 Stage 3 经历过**一次真实的精度回退**;而逐元素族 5 个算子虽是 `partial`,其失败**全是采集侧能力边界**。⇒ **不能按状态给算子排质量**。
- **`design_docs=null` 这类"静默降级"必须逐字段读集成 JSON 才能发现**:本批第二次踩到。集成报 `[ok]` 只说明"kernel 已就位",**不保证设计快照随包**。
- **集成侧测试缺口与设计 L1 不重合,是本批的结构性现象**(已在 `ceil`/`floor`/`exp`/`cos`/`vector_norm` 上各登记一次):本算子的缺口尤具体 —— manifest 的 `N` **全是 16 的倍数**,故**对齐尾部清零路径在集成侧从未被覆盖**,而它在设计 L1 里是通过"故意的非对齐 N"覆盖的。⇒ **设计 L1 覆盖 ≠ 集成侧覆盖**,两者必须分开登记。
