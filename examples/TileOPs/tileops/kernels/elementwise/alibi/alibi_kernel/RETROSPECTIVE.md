## Stage 1 (design) Retrospective — first_design — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Constraint-doc | `tilelang-op-design` 迁移流程未提示「生成式/零输入算子」这一类别：模板/Phase M0 的三问与 golden 契约默认「有输入张量」，本算子需额外处理「无输入 → 无 flatten、无 tail shape、无读」的语义 | DESIGN.md §0.2/§1.2；`op_requirements.input_spec.shape="无（generative）"` | 在 migration-analysis.md 增加「生成式（zero-input）算子」小节：I/O 契约仅输出、golden 由构造参数合成、roofline 为 write-only | R |
| Info-source | 轻量调研算子仍需先做**设备可行性探针**（本算子核心分歧不是公式而是「索引生成 API 是否可用」）；skill 的 Phase 2 探针预算（≤2）未提示「API 存在性探针」与「性能常数探针」的区别 | DESIGN.md §3.4（13 行 API 表，其中 4 行为设备实测否决） | 在 algorithm-research.md/Phase 2 明确：涉及**索引/gather/arange/广播**等待验证原语时，API 存在性探针优先于性能常数探针 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `@tilelang.jit(out_idx=[...])` + `T.npuir_arange` 组合运行期设备崩溃（`MTE DDR out of range` 507057），显式输出参数可绕；生成式算子须改用显式输出 | D | DESIGN.md §0.6 R5 / §3.4；`.agents/evolution/capability-gaps.md` CG-2026-0016 | 同 kernel 体分别用 `out_idx=[0]`（崩）与显式输出（bit-exact）在 Ascend910_9362 实跑 | tilelang 0.1.2 dev root build / CANN 9.0.0 / Ascend910_9362 / 2026-10-13 | pattern-library/elementwise.md（PL 新增）+ capability-gaps CG-2026-0016 |
| `T.npuir_arange` 作用于单行 UB buffer（shape `(1,N)`）静默产出错误尾元素；索引生成 tile 首维须 ≥2 | D | DESIGN.md §3.4 P3/P4 / §5.3；(1,64) 尾 3 元素读 1004.0 vs 期望 11.5/11.75/12.0；CG-2026-0017 | `design_probe_alibi.py`（`(1,512)` FAIL vs `(2,512)` exact）；`test_arange.py` `(32,32)` PASS | tilelang 0.1.2 dev root build / CANN 9.0.0 / Ascend910_9362 / 2026-10-13 | pattern-library/elementwise.md + CG-2026-0017 |
| Developer 模式向量 `T.vexp2` 在 (8,64) fp32 tile 部分元素 `inf`；标量 `T.exp2` 可用（1 ulp vs `torch.pow`） | D | DESIGN.md §3.4 P1/P2；CG-2026-0018 | `design_probe_alibi.py` P2（6/512 inf）；P1（标量 exp2 max_abs 2.98e-8） | tilelang 0.1.2 dev root build / CANN 9.0.0 / Ascend910_9362 / 2026-10-13 | pattern-library/elementwise.md + traps-runtime.md + CG-2026-0018 |
| 生成式 write-only 算子的距离矩阵可「结构化分离 + 跨 head 复用（CSE）」把索引开销摊薄 H×；int16 arange ramp + vcast + vsub + vabs 精确生成 `\|i-j\|` | P | DESIGN.md §0.6 R1 / §1.6.1 #1 / §1.6.0 R3；`verify_equiv.py` 31/31 EQUIV_PASS；设备 P3/P4 0 违反 | `verify_equiv.py`；`design_probe_alibi.py` P3/P4 | tilelang 0.1.2 dev root build / CANN 9.0.0 / Ascend910_9362 / 2026-10-13 | pattern-library/elementwise.md（新条目）+ algorithm-candidates.md |
| 开发期 `tilelang.cache` 命中旧 kernel 会污染探针结论（同一 hash `d588f599` 复用于不同 kernel 体；`clear_cache()` 不足以隔离） | D | 设计期观察：probe1/2/5/7-dev 共用同一 `main.so` hash；`(4,128)` tile 在含缓存复用时报 FAIL、`TILELANG_DISABLE_CACHE=1` 独立进程报 exact=True | `TILELANG_DISABLE_CACHE=1` + 独立进程重跑同 tile | tilelang 0.1.2 dev root build / Ascend910_9362 / 2026-10-13 | pattern-library/traps-compiler.md |

### Transferable Lessons（可迁移教训）

- 生成式（零输入）算子**没有** `shape: T.int32` 尾守卫/`T.min` tail 需求（S 编译期常量）；别照搬有输入 elementwise 的运行时 shape 形态。
- 只要 kernel 内用到 `T.npuir_arange`，就**不要用 `out_idx` 自动输出**——改显式输出参数（wrapper 侧 `torch.empty` + 传入）；这是硬约束而非风格。
- `T.npuir_arange` 的 dst tile 首维必须 ≥2；单行 `(1,N)` 会静默产出垃圾尾元素（不报错）。
- 需要「每行/每列递增索引」时，用 `T.npuir_arange(dst,[1,0],off)`（行）/ `[0,1]`（列）生成 2D 坐标矩阵，再 `vsub`/`vabs`/`vmul`；不要指望向量整数 div/mod（不存在）。
- `vabs` 不支持 int16；索引在 int16 ramp 后须 `vcast` 升到 fp32 再做 `vabs`/算术。
- `vmul(A, scalar, C)` 的标量操作数只接受 Python 常量或 `tir.expr.Var`（如 `slp = -T.exp2(e)` 的 LetStmt 结果）；把 `Mul`/`Call` 表达式**内联**传参会报 `has no attribute buffer`。`vexp2` 不可靠时用标量 `T.exp2`。
- 3-D v-op 广播（`[bh,1,1]*[1,bi,bj]`）不支持——多维 tile 用 2-D（`[R,C]`）并让多出的轴走 `T.serial`。
- 设计期探针**必须**用 `TILELANG_DISABLE_CACHE=1` + 独立进程，否则 `tilelang` 缓存复用会给出与 kernel 体无关的错误结论。

## Stage 2 (review) Retrospective — review — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 维度 8「弃选论证查证（强制）」列举了 `docs/`、`examples/`、`testing/`、pattern-library，但未点名「前端 API 导出清单」（`tilelang/language/__init__.py` + `customize_npuir.py` 的 `npuir_*`）是「某向量原语不存在」类负向断言的必查位点；本次靠 grep `customize_npuir.py` 才发现 `npuir_floordiv`/`npuir_div` 已导出 | REVIEW.md 维度 8；`tilelang/language/customize_npuir.py:425`（`npuir_floordiv`）、`__init__.py:98–99`（`npuir_floordiv as vfloordiv`） | 在 tilelang-design-review SKILL Phase 1 第 5 步「弃选论证查证」补充：负向断言须对照 `tilelang/language/__init__.py` 全量导出 + `tilelang/language/customize_npuir.py` 的 `npuir_*` 定义 | R |
| Info-source | `design_calc_check.py` 对非标准列名的 §4 缓冲表全部 `skip`（ub/l0c/core_split 皆跳过），机械复核对纯 Vector 算子几乎无拦截力，全靠人工复算 | 机械复核 JSON（ub_budget/l0c_budget/core_split 三 skip） | 在 design_calc_check.py 增设对 §4.3 通用缓冲表（Buffer 名/Shape/dtype/存储层级）的解析，或在 SKILL 提示「skip 项须人工复算并在 REVIEW 注明口径」 | P |
| Review-flow | 本仓存在 `testing/npuir/parallel_ops/` 整套 Developer 模式并行算子测试（12 个）却未进入设计/检视的「同类实现基线」视线；设计 §3.5.3 只列了 `creation_ops/test_arange.py` | `testing/npuir/parallel_ops/`；DESIGN.md §3.5.3 | 在 tilelang-op-design 的 migration-analysis.md / 调研检索清单中把 `testing/npuir/` 全部子目录（parallel_ops/creation_ops/…）列为同类实现必查位点 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 迁移设计的「NPU 无某向量原语」负向断言复核须三查：① `tilelang/language/` 导出清单（`npuir_*`）② `testing/npuir/*/` 在仓测试 ③ `docs/Tilelang.language/` 官方示例——本次 alibi 的「无向量整数 div/mod」被 `T.vdiv`(int64) + `npuir_floordiv`/`vfloordiv` + `test_floordiv_floormod_parallel.py`（Developer `//`,`%` int64 精确）推翻 | P | REVIEW.md 维度 8/问题 1；`docs/Tilelang.language/数学操作/T.vdiv.md` §2.2.1；`tilelang/language/customize_npuir.py:158,425`；`testing/npuir/parallel_ops/test_floordiv_floormod_parallel.py`（commit f987008） | `python3 -m pytest testing/npuir/parallel_ops/test_floordiv_floormod_parallel.py`（设备）；或 grep `npuir_floordiv` | tilelang 0.1.2 dev root build / Ascend910_9362 | tilelang-design-review SKILL 维度 8「调研结论正确性」④ + tilelang-op-design references/algorithm-research.md 负向断言纪律 |
| 「`out_idx`+arange 设备崩溃（CG-2026-0016）」不可泛化为「`T.Parallel` 写 UB 不可用」——`docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.4 官方示例（`@tilelang.jit(target="npuir")`+`T.Parallel`+`alloc_shared`）与 `testing/npuir/parallel_ops/` 整套 Developer 测试证明其可用 | P | REVIEW.md 维度 0/8 与问题 2；`docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.3–2.4；`testing/npuir/parallel_ops/*` | `python3 -m pytest testing/npuir/parallel_ops/test_cast_parallel.py`（设备） | tilelang 0.1.2 dev root build / Ascend910_9362 | pattern-library/traps-runtime.md（新增「负向断言过度泛化」陷阱）+ tilelang-design-review SKILL 弃选论证查证 |

### Transferable Lessons（可迁移教训）

- 复核迁移设计的「NPU 无能力/不可用」类负向断言时，必查三处：`tilelang/language/customize_npuir.py` 的 `npuir_*` 定义与 `tilelang/language/__init__.py` 导出、`testing/npuir/*/` 在仓测试、`docs/Tilelang.language/` 官方示例；只查 `docs/` 目录名会漏掉「API 已实现但文档缺页」的原语（如 `npuir_floordiv`）。
- 生成式算子的 `out_idx`+`T.npuir_arange` 运行期崩溃（CG-2026-0016）是**该组合**的缺口，不得泛化为「`T.Parallel` 写 UB 不可用」或「向量整数除法不可用」——根因不同，泛化会错误否决在仓已验证的候选。
- 本仓 `testing/npuir/parallel_ops/` 是判断「某标量计算形态（div/mod、cast、min/max、一元数学）是否可 Developer 化」的一手证据位点，任何涉及索引/标量计算的迁移设计都应先扫该目录。

## Stage 1 (design) Retrospective — revision 1 — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Constraint-doc | 迁移设计的「API 能力裁决」不应止步于「能否运行」，还须量化「能运行 vs 更快替代」的成本差；reviewer 的「弃选论证查证」只覆盖负向断言，未要求「正向可行候选的相对性能量化」 | rev1 DESIGN.md §1.6.0 R3/Q4（基线实测 39.03ms vs arange 0.66ms，59.1×）；本轮才补上设备性能对照 | 在 tilelang-op-design / design-review 增加：对「可实现但被淘汰」的候选，须给出**同规模设备实测或结构化成本量化**（op 数/访存/时间），不得只写「劣」 | R |
| Info-source | v0 的 §4.5 UB 预算用解析式（18 B/elem × 1.12×）估出「16384 超限」，与设备实测（12 B/elem、16384 恰贴 192KB）矛盾——解析式未考虑 BishengIR 的**生命期复用**分配 | rev1 DESIGN.md §4.5（`ub overflow, requires 2359296 bits while 1572864`）；`design_probe_alibi_v1.py` UB 天花板探针 | 在 tilelang-op-design 提示：纯 Vector 多 mid-buffer kernel 的 UB 预算应按「峰值生命期集合」估，而非全部 buffer 求和；条件允许时用编译探针实测天花板 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 源 GPU flat→div/mod 坐标生成在 NPU **可实现**（Developer `T.Parallel`+UB+int64 `//`,`%`；向量 `vdiv`/`vfloordiv` int64），但**逐元素标量**执行，S=2048/H=32/fp16 实测 39.03ms vs arange+V-shared 0.66ms（59.1×）——坐标方案之争是**性能裁决** | D | rev1 DESIGN.md §0.5/§1.6.0 R3/Q4；`design_probe_alibi_v1.py` Q1/Q4 | `TILELANG_DISABLE_CACHE=1 python3 history_version/design_probe_alibi_v1.py`（Q1 基线全 PASS、Q4 计时） | tilelang 0.1.2+87a3307b / CANN 9.0.0 / Ascend910_9362 / 2026-10-13 | pattern-library/elementwise.md + algorithm-candidates.md（generative 坐标生成条目） |
| `T.Parallel` 写 UB（Developer）**可用**（`T.Parallel.md` §2.3–2.4 + `parallel_ops` 12 测试 + 设备 Q1）；但 `T.Parallel` 内 `T.if_then_else(cond_on_loopvar)` 与 `T.fabs(局部变量)` **编译期 segfault**（`T.Parallel.md` §2.2.1 有条款）；等价 abs 用 `T.max(d,-d)` / inline `T.fabs` 可用 | D | rev1 DESIGN.md §3.4 bisect（V0–V9）；`design_probe_alibi_v1.py`；CG-2026-0016 仅为 out_idx+arange 组合缺口 | `pytest testing/npuir/parallel_ops/test_floordiv_floormod_parallel.py`（12 passed）；bisect V4/V7 segfault vs V6/V6b/V8 通过 | tilelang 0.1.2+87a3307b / CANN 9.0.0 / Ascend910_9362 / 2026-10-13 | pattern-library traps-runtime.md（「负向断言过度泛化」+「T.Parallel 条件式限制」）+ traps-compiler.md |
| 纯 Vector 多 mid-buffer kernel 的 UB 实测峰值 = 「生命期峰值集合」（本 kernel 12 B/elem），UB 硬上限 192KB（1572864 bits）——解析式（全 buffer 求和 ×1.12）会**高估**需求、误判 block 上限 | D | rev1 DESIGN.md §4.5；`design_probe_alibi_v1.py` UB 探针（8192/12288/16384 OK；24576 报 `requires 2359296 bits while 1572864`） | `TILELANG_DISABLE_CACHE=1 python3 /tmp/opencode/p_ub.py`（会话内；结论已入 §4.5） | tilelang 0.1.2+87a3307b / CANN 9.0.0 / Ascend910_9362 / 2026-10-13 | pattern-library/constants.md CONST-capacity + elementwise.md |

### Transferable Lessons（可迁移教训）

- 迁移设计做「坐标/索引生成」等结构性决策时，**先按设备能力三查**（`tilelang/language/` 导出 + `testing/npuir/*/` 在仓测试 + `docs/` 官方示例）确认候选**可获得**，再用**同规模设备实测**比较「可获得候选」的相对成本；「不可实现」与「可实现但慢 N×」是两种完全不同的裁决依据，混用会触发负向断言复核 fail。
- `T.Parallel` 的 `if_then_else` / `fabs(局部变量)` 崩溃是**有文档条款**的已知限制（`T.Parallel.md` §2.2.1「不支持直接使用循环变量的条件语句」），不是「T.Parallel 不可用」——汇报时必须区分「整条 API 不可用」与「某写法受限」。
- 纯 Vector 多 mid-buffer 的 UB 预算：先按**生命期峰值**估，再以 BishengIR 报文的 `requires N bits` 实测校准；解析式全求和会高估（本 kernel 18 B/elem 估 vs 12 B/elem 实测）。
- 设备探针在本轮的价值集中在「把两个被推翻的负向断言换成可量化事实」：既证明基线可实现（Q1 全 PASS），也给出 59.1× 的性能差（Q4）——负向断言被推翻不等于结论翻转，**关键是补齐量化依据**。

### rev2（机械门禁修正，无技术改动）

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Process-rule | DESIGN.md 正文引用 `docs/T.vdiv.md`（省略层级的「半具体」路径）——`gate_lint._ref_path_failures` 只跳过 `docs/...`（含 `...`）与 glob/模板变量，此 token 被判为「具体的仓库相对路径」并做存在性检查，实际不存在→S1-REF-PATHS fail | rev1 DESIGN.md L992/L1053；真实路径为 `docs/Tilelang.language/数学操作/T.vdiv.md`；修正后 `gate_lint.design_lint(..., migration=True)` failures=0 | 在 tilelang-op-design / design-review 提示：正文引用仓库文档路径时，要么写**完整真实路径**（逐级 `ls` 核对），要么统一用门禁可识别的省略式 `docs/.../xxx.md`；「写了目录层级但不完整」的中间形态最易被机械门禁误判 | R |

- rev2 仅替换 2 处 `docs/T.vdiv.md` → `docs/Tilelang.language/数学操作/T.vdiv.md`（diff 确认无其他改动）；`verify_equiv.py` 重跑 46/46 `EQUIV_PASS`，技术结论不变。

## Stage 2 (review) Retrospective — review round 2 (v1/rev2) — 2026-10-13

> 本轮对象：rev1/rev2 DESIGN.md（v0 两条负向断言已修正）。检视流程自身复盘如下。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 维度 8「弃选论证前提核对」虽点名了 `T.transpose.md` 的「dtype 矩阵」，但上一轮 review 只核对了**机制类别**（核内 UB 级 vs GM 级）就判 pass，未**逐格核对 dtype 值**；本轮独立打开 `T.transpose.md` §2.2.1 才发现文档记 bf16 √ 而设计记 bf16 ×（负向断言与文档矛盾，命中同一 fail 规则） | REVIEW.md 维度 8/问题 1；`docs/Tilelang.language/创建操作/T.transpose.md` L26/L28；DESIGN.md §1.6.3 候选 #2 | 在 tilelang-design-review Phase 1 第 5 步「弃选论证查证」明确：被引 API 文档的 **dtype 矩阵须逐格核对**（对每个被断言「不支持」的 dtype，回文档矩阵取该列真值），不得只核对机制类别 | R |
| Info-source | 「`T.Parallel` 能力 bisect V0–V9」与「UB 天花板探针（8192/12288/16384/24576）」在 DESIGN 中被归于 `history_version/design_probe_alibi_v1.py`，但该脚本只实现 Q1–Q4，无落盘工件可复跑；reviewer 只能以算术自洽（12 B/elem × 16384 = 1572864 bits、×24576 = 2359296 bits 与报错文本一致）间接佐证 | DESIGN.md §3.4 bisect 段 / §4.5；`history_version/design_probe_alibi_v1.py`（仅 Q1–Q4） | 在 tilelang-op-design 探针纪律中要求：凡在 DESIGN 正文以「设备实测」呈现的探针结论，须有对应**落盘脚本 + 运行命令**；否则须就地标注「会话内实测（未落盘）」 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 迁移设计的负向 dtype 断言须**逐格回文档矩阵核对**：`T.transpose` 文档记 bf16 √（int8/int16/int32/uint64/int64/fp16/fp32/bf16 均 √），而设计以「bf16 ×」为弃选前提——「已亲自核对该文档」的自述不能替代逐列取值 | P | REVIEW.md 维度 8/问题 1；`docs/Tilelang.language/创建操作/T.transpose.md` L26/L28（bf16 列 = √）；DESIGN.md §1.6.3 候选 #2 | `sed -n '26p;28p' docs/Tilelang.language/创建操作/T.transpose.md` | — | tilelang-design-review SKILL 维度 8「向量化轴与布局决策完整性 / 弃选论证前提核对」 |

### Transferable Lessons（可迁移教训）

- 「声称已亲自核对」不等于核对正确：本任务设计三处出现「已亲自核对 X.md」的自述（div/mod 一处、T.transpose 一处），其中 T.transpose 处与文档**相反**。reviewer 对被引文档必须**以自己读到的真值为准**，自述只作线索。
- 修正一轮负向断言后，须**全量重扫**同类断言，而非只修被点名处：v0 有 3 处「不支持」类负向（div/mod、T.Parallel、T.transpose），rev1 只修了前两处，第三处（T.transpose bf16）遗留并再次触发 fail。凡「不支持/不可用/无先例」类断言，应按「三查 + 被引文档逐格核对」一次性过一遍。
- 设备结论的可复现性与结论正确性同等重要：bisect / UB 天花板这类支撑 §4 操作界的探针若无落盘脚本，reviewer 只能做算术间接佐证，无法机械复跑；设计应「结论 + 脚本 + 命令」三件齐。

## Stage 1 (design) Retrospective — revision 2 — 2026-10-13

> 本轮对象：v1/rev2 DESIGN.md（Stage 2 round 2 判不通过，唯一阻塞项 = §1.6.3 候选 #2 的 `T.transpose` dtype 负向前提与文档矛盾）。本轮为最小改动修订。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Constraint-doc | skill 的「弃选论证证据规则」要求负向断言「附已亲自核对的 docs 路径并引用具体限制条款」，但未要求**逐格取 dtype 真值**；本轮阻塞项正是「设计自述已亲自核对 T.transpose.md，却与 §2.2.1 dtype 矩阵相反」——自述核对 ≠ 取值核对 | REVIEW.md 问题 1；`docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1 L26/L28（bf16 = √）；DESIGN.md §1.6.3 候选 #2 | 在 tilelang-op-design Phase 2 §3「弃选论证证据规则」显式加一句：凡断言「某 dtype 不支持」，须回文档矩阵逐列取值并原样引用（不得只写「已核对」） | R |
| Info-source | 长寿命 DESIGN 的**会话内未落盘探针**无法在后续轮次复现：本轮尝试重建 §3.4 的 `T.Parallel` bisect（V0–V9），其中 V0–V7 与记录一致、但重建的 V8/V9（inline `fabs`）在本工具链下 segfault，与 DESIGN 记的「✓」不符——原始变体代码未留存，无法判定孰对 | 本轮实跑 `history_version/design_probe_alibi_v1_ext.py`（临时，已删）：`[Q5 V8] rc=-11 / [Q5 V9] rc=-11`；DESIGN §3.4 记 V8/V9 ✓ | 在 tilelang-op-design 探针纪律中要求：以「设备实测」呈现的**每条**结论（含 bisect 逐变体）须保留脚本原文；不可复现时应标「会话内实测（未落盘）」且**不得**被下游当作操作界依据 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 迁移设计的 API dtype 负向断言须**逐格回文档矩阵取值**：「`T.transpose` bf16 ×」与 `docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1（bf16 = √）相反，且设计自述「已亲自核对」——自述不能替代实读 | P | REVIEW.md 问题 1（round 2）；`docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1 L26/L28；DESIGN.md §1.6.3 候选 #2 | `sed -n '26p;28p' docs/Tilelang.language/创建操作/T.transpose.md` | — | tilelang-design-review SKILL 维度 8 + tilelang-op-design references/algorithm-research.md 负向断言纪律 |
| §4.5 UB 天花板已落盘复现：ALiBi 距离链 buffer 布局（R/C int16 + Rf/Cf/O fp32 + O16 dtype）在 `bi·bj ∈ {8192,12288,16384}` compile+run OK，`24576` 报 `ub overflow, requires 2359296 bits while 1572864 bits available!` → 峰值 12 B/elem、硬限 192KB/AIV、天花板 16384 elems | D | DESIGN.md §4.5/§3.4；`history_version/design_probe_alibi_ub_ceiling.py` | `TILELANG_DISABLE_CACHE=1 python3 history_version/design_probe_alibi_ub_ceiling.py`（8192/12288/16384 OK；24576 FAIL） | tilelang 0.1.2+87a3307b2a83d60ee8af09086ff280a8bb6c26c5 / CANN 9.0.0 / Ascend910_9362 / 2026-10-13 | pattern-library/constants.md CONST-capacity + elementwise.md |

### Transferable Lessons（可迁移教训）

- 「已亲自核对 X.md」是**线索不是证据**：本任务两次因该自述踩坑（v0 的 div/mod 泛化、v1 的 T.transpose dtype）。对「某 dtype/模式不支持」类断言，正确做法是打开文档矩阵**取那一列的真值**并原样引用区间（如 `T.transpose`: int8/int16/int32/uint64/int64/fp16/fp32/bf16 √，uint8/uint16/uint32/bool ×）。
- 生成式（零张量输入）算子的核内布局候选里**不存在「输入侧 permute」**：任何 host 重排只可能发生在输出侧且通常净亏（输出 layout 为契约）；写候选矩阵时须先对齐「无输入」这一语义前提。
- 会话内未落盘探针（尤其 bisect 这类含「崩溃/通过」二值结论、变体写法敏感的实验）在后续轮次**不可复现**：重建的等价变体可能落在 pass/fail 边界的另一侧。凡此类结论，落盘脚本须保留**逐变体原文**，否则只能作「会话内未落盘」标注、不得支撑操作界（§4/§5）。

## Stage 2 (review) Retrospective — review round 3 (rev2) — 2026-10-13

> 本轮对象：rev2 DESIGN.md（上一轮唯一阻塞项 = §1.6.3 候选 #2 的 `T.transpose` dtype 负向前提，已修正）。**本轮结论：通过**（0 阻塞）。检视流程自身复盘如下。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 机械复核端口对**纯 Vector 生成式算子**仍近乎全 skip：`design_calc_check.py` 的 ub_budget / l0c_budget / core_split 三项（§4.3 通用 Buffer 表与 §5.5 分核三要素的列名与解析器不匹配）本轮全部 skip，UB 预算与分核只能人工复算（12 B/elem × 8192 = 96 KB；逻辑 2/512/2048 vs 物理 40），机械拦截力为零 | 机械复核 JSON（三 skip）；DESIGN.md §4.3/§4.5/§5.5 | 在 `design_calc_check.py` 扩展解析 §4.3 通用 Buffer 表（Buffer 名/Shape/dtype/存储层级）与 §5.5 分核三要素（逻辑核数/物理核数/分核方案），或要求 SKILL 明确「skip 项须人工复算并注明口径」（round 2 已提，本轮再次命中，说明非偶发） | R |
| Info-source | rev2 §11.1 声称「全文 dtype 断言逐格核对」——reviewer 无机械手段验证该自述，只能逐份打开 6 个被引文档的 §2.2.1 dtype 矩阵**重建**核对（本轮实做 `T.transpose`/`T.vabs`/`T.vdiv`/`T.vcast`/`T.vmul`/`T.arange` 六份，全部一致）。仓库缺「设计 dtype 断言 ↔ 文档矩阵」的机械比对器 | DESIGN.md §11.1 自查清单；6 份 `docs/Tilelang.language/**/*.md` §2.2.1 | 增设轻量机械检查：解析设计中「某 op 支持/不支持某 dtype」的断言句与 doc §2.2.1 矩阵，自动比对逐格真值（可仿 `gate_lint` 的 ref-path 规则） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 迁移设计的 dtype 断言可复用 rev2 §11.1「逐条列 doc 路径 + dtype 列真值」的自查清单形态作为 reviewer 复核清单：本轮独立逐格重建 6 份文档矩阵（`T.transpose` bf16 √ / `T.vabs` int16 × / `T.vdiv` int64 √ / `T.vcast` i16→f32 rint / `T.vmul` 标量广播 2-D 规则 / `T.arange` int16 √），与设计断言全部一致，复核成本显著低于无清单时 | P | REVIEW.md 维度 8 / §11.1 独立验证表；`docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1 等 6 份 | `sed -n '26p;28p' docs/Tilelang.language/创建操作/T.transpose.md`（余同法逐文档 §2.2.1） | tilelang 0.1.2+87a3307b / CANN 9.0.0 / Ascend910_9362 / 2026-10-13 | tilelang-design-review SKILL 维度 8 + tilelang-op-design references/algorithm-research.md 负向断言纪律 |

### Transferable Lessons（可迁移教训）

- 迁移设计把每条 dtype 断言与「文档路径 + dtype 列真值」并列成自查清单（rev2 §11.1 形态）能显著降低 reviewer 复核成本、并避免上一轮「自述已核对但取值相反」的失效模式；建议后续迁移设计普遍采用该形态。
- 设备实测类结论（`T.Parallel` bisect、UB 天花板、Q4 性能比）在 review 上下文通常不可复跑：reviewer 的正确做法是核对**落盘脚本存在性 + 报错/数值的算术自洽 + 显式「未落盘」标注**，既不得把未落盘结论当作操作界（§4/§5）依据，也不应仅因其未落盘而否决有独立佐证的候选。
- 生成式/多 mid-buffer 纯 Vector 算子的 UB 预算与分核在现有机械门禁下均为 skip——reviewer 必须人工复算并把口径写进 REVIEW.md「机械复核」段的 skip 说明，避免「skip 即通过」的误读。

## Stage 3 (develop) Retrospective — first_impl (attempt 1) — 2026-10-13

> 结果：`[PRECISION_PASS]`。L0 12/12、L1 10/10、L2 5/5、Boundary 6/6 全通过（`--level all`），一次编译通过、零精度失败。实现按 DESIGN.md §3.3 伪代码 + §5/§6 落地，未偏离设计结论。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | `tilelang-op-develop` 的 `templates/op_template.py` 把 `@tilelang.jit(out_idx=[-1], target="npuir")` 作为「canonical kernel 示例」，而 `out_idx` + `T.npuir_arange` 正是设备实测会运行期崩溃（`MTE DDR out of range` 507057，CG-2026-0016）的组合；生成式/arange 类算子照抄模板即踩坑，模板缺一句「arange / 生成式算子须改显式输出参数 + host 侧 `torch.empty`」 | `templates/op_template.py` L20；本算子 `_make_alibi_kernel.py` L144（无 `out_idx`）；DESIGN.md §0.6 R5 / §3.5.1 | 在 `op_template.py` 的 `out_idx` 示例旁加注释：使用 `T.npuir_arange`（或生成式零输入）的 kernel 禁用 `out_idx`，改显式输出参数（引 CG-2026-0016） | R |
| Info-source | develop skill 的环境前提只说「tilelang 与 torch_npu 可正常导入」，未要求比对**实际工具链 stamp 与设计文档绑定的 stamp**（DESIGN.md R-17 明确设计结论绑 `0.1.2+87a3307b`）。本次实跑环境为 `0.1.2+d822cb9b71766e0914a939cea85d68284cf580e1`（不同 build hash），若设计结论对该工具链敏感将静默失真；本次结论恰好全部复现（见价值点 1），属运气与设计稳健性共同作用 | `_make_alibi_kernel.py` 运行日志（loading tilelang libs from dev root）；DESIGN.md §9.1 R-17 / §3.4 工具链戳 | 在 tilelang-op-develop SKILL「环境前提」增加一步：记录实际 tilelang build/commit，与 DESIGN.md 绑定的 stamp 不一致时显式提示「设计设备结论待重验」（对齐 R-17） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 生成式 ALiBi arange-ramp + head 维 D 共享 kernel 在**当前工具链**端到端复现设计期结论：全任务矩阵（S∈{64,128,200,255,256,512,1000,1024,2048,4096}、H∈{1,4,5,6,8,16,32}、fp16/bf16/fp32）0 失败；fp32 max_abs 3.8e-6~1.2e-4（在 `atol+rtol·\|ref\|` 内）、fp16 bit-exact 至 3.9e-3、bf16 ≤8.0（1 ulp）；对角 `i=j` 精确 0；S=4096（`\|i-j\|>2048`）int16 ramp 精确 | `_make_alibi_kernel.py`（run_L0/L1/L2/Boundary 输出）；DESIGN.md §3.4 Q1/Q2 | `TILELANG_DISABLE_CACHE=1 python examples/alibi/_make_alibi_kernel/_make_alibi_kernel.py --level all` | tilelang 0.1.2+d822cb9b / CANN 9.0.0 / Ascend910_9362 / 2026-10-13（设计戳为 87a3307b，不同 build，结论仍复现） | pattern-library/elementwise.md + capability-gaps CG-2026-0016/0017/0018（复现确认） |
| 编译期 S 的纯 Vector / 生成式 tile 划分用「**因子搜索** `bj\|S`、`bi=max(2, budget//bj)` 且 `bi\|S`」可在**任意 S**（含 1000/255/200 等非 2 幂）上实现逐 tile 整除，从而完全免除 `T.min` 尾块守卫；比「bi=budget//bj 后补尾守卫」更简单且写出连续（bj=S 时整行 copy） | `_make_alibi_kernel.py` L115-140（`_select_blocks`）；run `L0-nondiv-1000` max_diff=0.0、`L2 S=200` 3.8e-6、`L2 S=255` 0.0；DESIGN.md §5.4 主策略 | `TILELANG_DISABLE_CACHE=1 python examples/alibi/_make_alibi_kernel/_make_alibi_kernel.py --level all`（观察 S=1000/200/255 用 `bi=None` 自动因子划分且全过） | tilelang 0.1.2+d822cb9b / CANN 9.0.0 / Ascend910_9362 / 2026-10-13 | pattern-library/elementwise.md + tilelang-op-design references/algorithm-research.md（tiling 选择纪律） |

### Transferable Lessons（可迁移教训）

- 生成式（零张量输入）NPU kernel 一律用**显式输出参数** + host 侧 `torch.empty(...).npu()` 分配并传入；只要 kernel 内用了 `T.npuir_arange` 就**绝不能**用 `@tilelang.jit(out_idx=[...])`（设备运行期崩溃 `MTE DDR out of range` 507057，非编译期报错）。
- `T.npuir_arange(dst, strides, offset)` 的 dst tile **首维必须 ≥2**；`(1, N)` 单行 tile 会静默产出垃圾尾元素（不报错），`bi=max(2, ...)` 是硬约束。
- 编译期约束的 S 用**因子搜索**选 tile：`bj` 取 ≤S 的最大因子（优先 =S）、`bi` 取 budget//bj 内最大且整除 S 的因子并强制 ≥2；这样每 tile 恰好整除，无需 `T.min` 尾块裁剪。仅当 S 为超大素数等无合适因子时才退化为 `T.min` 守卫。
- 与串行循环轴无关的中间量（如 ALiBi 的 `|i-j|` 距离矩阵与 head 无关）应**在循环外算一次、循环内复用**（CSE），这是本类生成式算子发射成本的主要杠杆。
- 每轴一次的**标量** `T.exp2` 是可靠的（1 ulp vs `torch.pow`），可作 `T.vmul` 的标量操作数；不要用向量 `T.vexp2`（本设备部分元素 `inf`，CG-2026-0018）。
- 开发/验证期统一用 `TILELANG_DISABLE_CACHE=1` + 独立进程跑，避免 `~/.tilelang/cache` 旧 kernel 复用污染「同形不同 kernel 体」的结论（R-13）。
- 实现环境与设计文档的工具链 stamp 可能不同（本任务设计绑 `87a3307b`、实跑 `d822cb9b`）；即便「可正常导入」也应在已知会设备结论敏感时先复跑一个最小端到端用例再全量展开。

