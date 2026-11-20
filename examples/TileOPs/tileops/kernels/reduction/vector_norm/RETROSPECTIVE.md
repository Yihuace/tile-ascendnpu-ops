# RETROSPECTIVE.md — `vector_norm`(op 级,覆盖 L1/L2/Inf 三个 Op)

> 追加式日志。逐函数的 Stage 1/2/3 复盘见
> `examples/vector_norm/vector_norm_kernel/RETROSPECTIVE.md`(同一份亦复制到
> `examples/vector_norm/_vector_norm_kernel/RETROSPECTIVE.md`,与集成产物的函数名目录对齐)。

---

## Stage 5 (integrate) Retrospective — attempt 1 → 3 — 2026-10-15

- **verdict**:`INTEGRATE_COMPLETED`(benchmark 无失败)
- **attempts_used**:3(**均为集成脚本层面的布局/模块匹配问题**,非 kernel 问题)
- **正确性**:`L1NormFwdOp` **28/28**、`L2NormFwdOp` **28/28**;`InfNormFwdOp` report 运行中
- **benchmark**:`benchmark_failed = 0`(归约族的 manifest 无 268M 量级 workload ⇒ 无那两条采集侧失败)

### attempt 链(如实记录)

| attempt | 结果 | 根因 |
|---|---|---|
| 1 | `[error] no conductor product for '_vector_norm_kernel'` | **产物目录命名不符流程约定**:集成按 `examples/{op_slug}/{func}/{func}.py` 查找,须用**抽取出的函数名**(`_vector_norm_kernel`)而非类名(`vector_norm_kernel`) |
| 2 | `[error] wrapper … has no 'from ._inf_norm_fwd_kernels import'` | meta 的 `extracted_file` 是**抽取时的 GPU 参考文件名**(`_inf_norm_fwd_kernels.py`),而既有 wrapper 导入的是 `from ._vector_norm_kernel._vector_norm_kernel import` |
| 3a | `[error] wrapper … has no 'from ._vector_norm_kernel import'` | `--extracted-module _vector_norm_kernel` 仍不匹配 —— wrapper 用的是**两级点号模块** |
| **3b** | ✅ `[ok] integration complete` | `--extracted-module _vector_norm_kernel._vector_norm_kernel`(完整点号路径)匹配成功 |

**随后补做一步**:首次成功时 `design_docs` 为 `null`(设计快照在 `vector_norm_kernel/` 而集成按函数名目录 `_vector_norm_kernel/` 找)⇒ 把 `DESIGN.md`/`REVIEW.md`/`RETROSPECTIVE.md` 复制到 `_vector_norm_kernel/` 后**重跑集成**,`design_docs` 正确带上。

### 集成后的仓库状态(须如实登记)

- **wrapper 已重写**:`from .vector_norm_kernel import _vector_norm_kernel` ⇒ 指向**本 Stage 3 产物**(hash `9d23b65…`)
- **新增集成包**:`tileops/kernels/reduction/vector_norm/vector_norm_kernel/`(含 `_vector_norm_kernel.py` + `__init__.py` + `_vector_norm_kernel_DESIGN.md`)
- ⚠️ **残留一个孤儿目录**:`tileops/kernels/reduction/vector_norm/_vector_norm_kernel/`(手工版,hash `eb93ac1…`)。它**已无引用**(仅其自身 `__init__.py` 自引用),但**未删除**(避免误伤未知引用)。**须由 conductor 决定是否清理**。

### 未交付项(偏差声明)

`_vector_norm_kernel_tiled`(GPU 侧的分块变体)**未纳入本次交付**。理由:manifest 的 4 个 workload 在 `block_m=1` 下均可容纳(逐条核算见 DESIGN.md §4.3/§5.3),该分支在本任务范围内**永不被触发**;交付无法验证的代码路径违背"不得把未验证当已验证"的纪律。该偏差已同时记入 `.migration_meta.json` 的 `extracted_functions_deferred` 字段。既有 NPU 侧实现仍保留分块版。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | **产物目录命名约定未文档化**:集成脚本按 `examples/{op_slug}/{func}/{func}.py` 查找,但 skill 从未明写"目录名必须是**抽取出的函数名**"。本任务按类名建目录(`vector_norm_kernel/`)直接被拒 | 本任务 attempt 1:`no conductor product for '_vector_norm_kernel'`(脚本还提示了 `--func-dir` 兜底) | 在 `tilelang-op-develop` 的交付件落盘规范里明写:**目录名 = 抽取出的函数名**,并给 pattern B(一个 kernel 多 Op)的示例 | R |
| Integrate-flow | **`--extracted-module` 对"两级点号模块"的匹配不足**:脚本按 `from .{stem} import` 匹配 wrapper,而本仓存在 `from .A.B import` 形态 ⇒ 单级 stem 必失败,且**报错信息给出的候选值(`_vector_norm_kernel`)也是错的**,需自行猜出完整点号路径 | 本任务 attempt 3a 失败后,试出 `_vector_norm_kernel._vector_norm_kernel` 才通过 | 修 `integrate_kernel.py`:匹配 `from .{stem}` 时允许 stem 含点号,或**直接从 wrapper 已有的导入行反解 stem** 并提示给用户 | R |
| Integrate-flow | **设计快照的查找目录与产物目录绑定**,产物目录一旦改名,`design_docs` 静默变成 `null`(集成仍报 `[ok]`,不报错) | 本任务首次成功时 `"design_docs": {"_vector_norm_kernel": null}` —— **集成判据没把"设计快照缺失"当问题** | 在集成脚本的 ok 判定里增加:**`design_docs` 有任一项为 `null` 时输出 warning 或非零 `design_missing`**;或在 harness 的 DoD 里列明"设计快照必须随包" | R |

### Transferable Lessons（可迁移教训）

- **交付件的目录名必须是"抽取出的函数名",不是 kernel 类名**:`examples/{op_slug}/{func}/{func}.py`。本任务按类名建了 `vector_norm_kernel/` 目录,集成直接找不到产物。**pattern B(一个 kernel 类覆盖多个 Op)时尤其容易踩** —— 因为类名与函数名不同且都能"看起来像"目录名。
- **`--extracted-module` 要填 wrapper 里 `from .` 之后的**完整**模块路径**,包括所有点号层级:本仓是 `_vector_norm_kernel._vector_norm_kernel`(两级)。脚本给出的候选值只有单级,不足。
- **集成的 `[ok]` 不等于产物齐全**:本任务首次成功时 `design_docs` 为 `null` 却仍报 `[ok]`。**读集成 JSON 的每个字段**,不要只看最后那行 `[ok] integration complete`。
- **`benchmark_failed = 0` 是可能的**:只需 manifest 不含 268M 量级 workload。归约族 `vector_norm` 的最大 workload 是 `(2048,4096)` ⇒ 那两条采集侧失败**不出现**。⇒ 进一步印证 E/R 类结论:**那 2 条失败是 workload 规模问题,不是算子问题**。
- **集成后要检查"是否留下无引用的旧实现目录"**:本任务留下 `_vector_norm_kernel/`(手工版)。**未擅自删除**(避免误伤未知引用),但**必须登记**由上层决定。

---

## L1 优化复盘（2026-11-04）

- **本仓配置与本地开发树不同**：此仓 `pick_block_m` 取 `block_m=3`（本地树为 4），
  同一改动 ratio 差 6+ pp（46.74% vs 52~67%）⇒ **调优数字不能跨环境搬运**，必须在目标仓配置下重采。
- **「能到 60」的那条路在本仓不合法**：fp16 域归约（同 dtype 累加 + `reduce_abssum`）在本仓配置下
  58.62%（+12.5 pp），但 **1/2048 行相对误差 0.1227% > 0.1% 容差**（绝对差 4.0）；
  bf16 因容差更宽（1.6e-2）而通过。
  ⇒ 「访存分支里减少向量工作是纯收益」成立，但**归约域的精度代价必须按 dtype 单独验**，
  且要先跑 `--check` 再谈收益。
- **精度安全的杠杆收益很小**：块连续分区 + 批量回写只值 +0.65 pp（fp16）/ +0.21 pp（bf16）。
  L1 在本仓 46% 的根因不是指令，而是 `block_m=3` 下每 tile 固定开销摊薄不足 —— 想上 60
  必须同时解决「更大 tile」与「累加精度」两件事。
- **测速必须与 `--check` 同跑**：fp16 域快路径单独测是 58.62%，加上 `--check` 才发现精度不达标。
## 契约变更复盘：bf16 上采到 fp16（2026-11-06）

- **性能数字必须成对披露**：bf16 上采后，上报口径从 51–52% 跳到 61–66%（因为 harness 只统计被测
  kernel），但含转换的端到端等效 ratio 只有 **≈25%**（上采 33.55 MB ≈ 22.5 µs，是 kernel 的 1.6 倍）。
  ⇒ 任何「把工作挪到被测区间之外」的优化，**必须同时给出端到端口径**，否则就是指标失真。
- **契约变更的适用边界**：只有当 dtype 转换放在调用方、且转换成本可被业务接受时才成立；
  在本例里端到端变慢 2.6 倍，所以它只是「口径例外」而不是真实优化。
- **本机无 `tvm` 时的验证手法**：`tileops/ops/reduction/__init__.py` 与 `kernels/reduction/__init__.py`
  在模块顶层 eager import 了依赖 tvm 的算子，导致 `vector_norm` 无法单独导入。
  验证时可临时把这两个 `__init__` 换成容错导入版（try/except 逐项导入），**验完必须还原且不提交**；
  真正的 CI 有 tvm，不需要该补丁。
