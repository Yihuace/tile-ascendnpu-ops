# RETROSPECTIVE.md — `gqa_fwd` / `_gqa_prefill_paged_with_kv_cache_fwd_kernel`

> 追加式日志:每次 Stage 执行追加一个章节,不覆盖历史章节。
> **本算子的 Stage 3 出口为「阻塞」,不产出 `[PRECISION_PASS]`,不得按"已交付"计。**

---

## Stage 1 (design) Retrospective — first_design — 2026-10-16

本单元把既有非 skill 格式设计整合为 skill 章节结构,并**沿用**其关键发现(不重造)。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | **"阻塞根因的归属"缺少强制区分**:同一个报错可以归给"仓库缺实现"(⇒ 改仓库)或"工具链版本不匹配"(⇒ 对齐版本),**两者的解锁动作完全不同**,而流程未要求把"发射侧/消费侧"分别取证 | 本任务:报错 `Dialect '' not found for custom op 'syn_instr_mode'` —— 若只看到"找不到",最自然的结论是"仓库没实现这个 op";实际是**仓库发了、wheel 不认** | 在 `tilelang-design-review` 的阻塞类核对项里加一条:**凡报"XX 找不到/不支持",须分别取"发射侧"与"消费侧"的证据**,并明确归属(缺实现 / 版本落差 / 配置);**归属不明时不得下结论** | R |
| Design-flow | **skill 生成的"同类实现引用"是很好的锚点**:本算子最终能定出"扩展而非重写"这一正确路线,靠的是**先查到仓内已有 1269 行的 dense 实现** | 本任务:§3.4 的引用表把 `multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`(1269 行)列为**扩展基座**,从而把工作量从"重写 flash-attention"降到"加分页寻址" | 无(该字段已存在且有效);**建议在 skill 说明里点明"先在仓内找同族既有实现"是降本手段** | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **`syn_instr_mode` 的阻塞是"仓内 codegen 与已装 wheel 的 dialect 版本落差"**:发射侧在 `src/target/codegen_npuir.cc:1053`(**仓库源码确有该行**),消费侧是已装 wheel 的 NPU IR 解析器**不认该形式** | D | `repro/TRAP-cann-version-dialect-skew.py` 的实跑输出(两侧并列);`codegen_npuir.cc:1053`;wheel 的 `libtilelang*.so` | `env -u PYTHONPATH <venv>/bin/python -u examples/gqa_fwd/_gqa_prefill_paged_with_kv_cache_fwd_kernel/repro/TRAP-cann-version-dialect-skew.py` | tilelang `0.1.2+npuir+7fc4a3c8` / Ascend910B2C / **CANN 9.0.0** | `pattern-library/traps-compiler.md`(版本落差类)+ `tilelang-npuir-overview` |
| **阻塞证据可用"仓库自带测试的既有失败",其可信度高于自造探针** | P | 本任务用 `tests/ops/test_multi_head_attention.py`(**4 failed, 2 passed**)作证据 —— 该失败**先于**本次迁移存在,**无法被"特意构造成失败"** | 同上 repro(它包装了该测试,不自己造 kernel) | 同上 | `tilelang-op-develop` 的阻塞证据要求 |

### Transferable Lessons（可迁移教训）

- **"找不到某个 op"的根因可能是"版本落差"而非"没实现"——必须两侧分别取证**:本任务的报错 `Dialect '' not found for custom op 'syn_instr_mode'` 最自然的读法是"仓库没实现";实际是**仓库的 codegen 会发射它,而已装 wheel 的解析器不认**。⇒ **凡报"XX 找不到/不支持",要分别取"谁发的"与"谁不认"两侧证据**;归属判错会让解锁动作南辕北辙(改仓库 vs 对齐版本)。
- **两轮 grep 才能纠正一次误判**:本设计曾记载"`syn_instr_mode` 在仓库源码**零命中**、只在 wheel 的 `.so` 里"。本轮 grep 发现**仓库确有 1 处命中**(发射点)。⇒ **"零命中"这类强断言必须重跑一次再写进文档**;它对结论的影响不是细节,而是**解锁路径的定性**。
- **先查仓内同族既有实现,再决定"重写还是扩展"**:本仓已有 1269 行的 dense GQA prefill 且**工厂签名与本算子一致** ⇒ 本算子的实质新增只是"分页寻址"。**若不先查,会把工作量估成"重写一个 flash-attention"。**
- **用"仓库自带测试的既有失败"作阻塞证据比自造探针更硬**:自造探针有"被构造成失败"的嫌疑,**既有失败**没有。本任务因此把证据做成**包装该测试**,而不是自己写 kernel。
- **`skip` 的成因不止一种**:本算子门禁的 `l0c_budget` 为 `skip`(`no affirmative Cube-path API`),但**本算子确有 Cube 路径** —— 成因是**环境阻塞导致工具看不到可编译的 Cube 路径**。这与另两项"设计主动声明不核算"的 `skip` **成因不同**,须分开判读。
