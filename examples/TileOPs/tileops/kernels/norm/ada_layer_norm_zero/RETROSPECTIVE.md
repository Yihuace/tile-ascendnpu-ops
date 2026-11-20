# RETROSPECTIVE.md — `ada_layer_norm`(op 级,覆盖 `AdaLayerNormFwdOp` 与 `AdaLayerNormZeroFwdOp`)

> 追加式日志。逐函数的 Stage 1/2/3 复盘见
> `examples/ada_layer_norm/ada_layer_norm_kernel/RETROSPECTIVE.md`
> (同一份亦复制到 `examples/ada_layer_norm/_ada_layer_norm_kernel/`)。

---

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-16

- **verdict**:`INTEGRATE_COMPLETED`
- **report_status**:两 Op 均 **`passed`**;正确性 **22/22**(`AdaLayerNormFwdOp`)、**14/14**(`AdaLayerNormZeroFwdOp`);`benchmark_failed = 0`
- **attempts_used**:1(集成侧);**上游 Stage 3 零回退**

### ⚠️ 本轮的**首要发现**:wrapper 的生效源是 `perf_opt`,不是本次集成的 baseline

| 项 | 事实 |
|---|---|
| wrapper 的实际状态 | `perf_opt` 导出块**生效**;baseline 导出块**被注释** |
| 集成是否翻转了开关 | **否**(`wrapper_rewritten: false`) |
| report 实际测的是谁 | **`perf_opt`** —— 因此 **22/22 与 14/14 不能当作"baseline 已在集成侧被验证"** |
| baseline 由谁验证 | **Stage 3 自带分层套件**(L0 6/6、L1 20/20、L2 4/4、Boundary 16/16,零失败) |
| 为何不翻转 | wrapper 自述策略:"perf_opt … becomes active once tuned drop-in kernels … land … **and pass their L0/L1 regression**; the baseline (Stage 3) source is active otherwise"。`perf_opt/_ada_layer_norm_kernel.py` + `perf_opt/opt_log.md` **已存在且已激活** ⇒ 满足该策略。**翻转会把已调优变体降级**,不属本次集成该做的改动 |

⇒ **本算子是本批唯一"集成侧 report 与本次交付的 Stage 3 产物走的不是同一条代码路径"的算子**,该口径限定已在 `integration_log.md` 与 `integration_report.json` 中显式登记。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | **当集成侧已存在 `perf_opt/` 且其导出块处于激活态时,流程没有规定"report 测的是谁"** —— 集成脚本照常报 `[ok]`、report 照常出 `passed`,但**被验证的代码路径并不是本次集成的产物**。若不复核 wrapper 的开关状态,会把 `perf_opt` 的成绩记到 baseline 头上 | 本任务:wrapper 的 baseline 块被注释、`perf_opt` 块生效;集成 JSON 的 `wrapper_rewritten: false` 是**唯一线索**,而它不显式说明"生效源是哪个" | 在 `tilelang-op-integrator` 的验证段增加**一步**:集成后须**判定 wrapper 的生效源**(baseline 还是 perf_opt),并在 `integration_log.md` 写明"**本次 report 验证的是哪个源**";若为 perf_opt,须同时说明 baseline 的验证来源 | R |
| Integrate-flow | **集成的 `[ok]` 不区分"我集成的产物是否被实际使用"** | 本任务:`[ok] integration complete` + `smoke: pass` 同时成立,而**生效源不是本次产物** | 在集成 JSON 里增加一个显式字段,如 `"active_source": "perf_opt" | "baseline"` | R |
| Report-flow | **report 目录按时间排序取"最新"可能取到昨天的旧 run** —— 本任务 `ls -dt` 一度取到 `20261015_124806_589781_AdaLayerNormZeroFwdOp/`(**昨天**),其 `14/14` 与本轮**数值巧合相同**,极易误用 | 本任务:首次查询时拿到的是昨天的 run;今天的是 `20261016_064033_284305_...`(本轮采集事故后重跑才产出) | 在收敛 report 结论时**必须用日期前缀过滤**(如 `reports/tileops/20261016_*_<Op>/`),不要只看"最新目录" | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **wrapper 的"双 import 切换块"会使集成侧 report 的验证对象与本次集成的产物脱钩**,`integration JSON` 的 `wrapper_rewritten` 是唯一线索但不显式说明生效源 | D | 本任务 `ada_layer_norm.py` 的切换块原文(注释:"Exactly one source block below is active; toggle by swapping the comment")+ 集成 JSON `"wrapper_rewritten": false` + `design_docs` 判断链 | 读该 wrapper 的导入块,对照集成 JSON 的 `wrapper_rewritten` 字段 | tilelang `0.1.2+npuir+7fc4a3c8` / CANN 9.0.0 | `tilelang-op-integrator` SKILL 的验证段 |

### Transferable Lessons（可迁移教训）

- **集成的 `[ok]` 不等于"你集成的产物被用上了"**:本算子 wrapper 的生效源是 `perf_opt`,而本次集成的是 baseline。**集成后必须复核 wrapper 的生效源**,并把"report 验证的是哪个源"写进集成日志 —— 否则会把别的实现的成绩记到自己的交付件上。
- **"最新目录"不等于"本轮的 run"**:`ls -dt` 取到昨天的目录时,其数值可能与今天**巧合相同**,从而被误认为本轮结果。⇒ **收敛 report 结论必须按日期前缀过滤**。
- **采集事故要按"未执行"处理,不能按"失败"**:本任务 `AdaLayerNormZeroFwdOp` 的首次采集日志为空、无目录 —— 那是**从未启动**(随会话中断一起被杀),不是运行失败。**重跑后才算数**,且须在日志里留下这条记录。
- **`perf_opt` 的存在会改变集成的"验收对象"**:当仓内既有 `perf_opt` 且其块被激活时,正确做法是**遵从 wrapper 自述的策略**(已调优 + 已过回归 ⇒ 保持激活),**不是**为了"让 report 测我的产物"而翻转它。→ 代价是**必须在文档里写清验收口径**,否则结论会被读者误读。
