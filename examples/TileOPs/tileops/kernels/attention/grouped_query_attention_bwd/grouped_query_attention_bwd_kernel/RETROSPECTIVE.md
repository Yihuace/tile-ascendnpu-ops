## Stage 1 (design) Retrospective — first_design — 2026-10-15

> task: `grouped_query_attention_bwd-_gqa_bwd_kernel`（migration-harness 逐函数，函数 3/3 = 主函数）
> skill: `tilelang-op-design`（Phase M0 → R → M1 → 1 → R → 2）

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 迁移类「多步融合 MixCV」算子（flash-bwd：gemm×5 + 逐元素 + 跨核原子）在 `examples/` 的最佳先例是 `examples/deepseek_v32/sparse_mla_bwd.py`，但该文件不在任何 skill 的引用列表/mapping 中；本次靠手工 `grep -rln "T.gemm" examples/` + `grep -rln "atomic_add"` 才定位。设计 skill 的「同类 examples 检索」步骤缺少**按 API 名反查 NPU 先例**的机械入口 | `grep -rln "T.gemm" examples/` → 37 文件；最终选定先例 `examples/deepseek_v32/sparse_mla_bwd.py`（未在 skill 清单） | `tilelang-op-design` 的 examples 检索补「按 API 名 grep（`T.gemm`/`atomic_add`/`vexp2`）+ `is_npu=True` 过滤」捷径 | R |
| Info-source | 迁移决策的「归约结构」选择（哪个输出走原子、哪个走核内寄存器）对同族算子是**通用设计岔路**，但 skill 的 migration-analysis.md 未给出判据；本次自行推导「原子量 ∝ 另一维权块数、且写序确定性」的量化比较（§1.6.0 R3/R4） | `DESIGN.md#§1.6.0 R3`（#1 vs #3 的原子量比 `2*groups*BM/BN`） | migration-analysis.md 补「跨核归约结构选择：把网格维放在『求和集合最小』的那一维」通则 | P |
| Tooling | `verify_equiv.py` 的角点用例若按「dtype 极值直接相乘」构造，会因 128 项点积在 fp32 溢出而使**参照本身**变 inf/nan，产生假失败（首次运行 6/10 失败）；且合成 `lse` 若不满足 `lse >= max(score*scale)`，`exp2` 会溢出（与真实语义不符） | `verify_equiv.py` v1 运行输出 `wide_dynamic_range/dtype_extremes/subnormal_neg_shift → FAIL`；修正为「按真实语义导出 lse（行 logsumexp）+ dtype 极值四次根缩放」后 10/10 PASS | D-1 脚本模板补两条构造纪律：① `lse` 必须按 log2 域行 logsumexp 生成；② dtype 极值需缩放到点积可表示 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **GQA 反向的 NPU 归约结构通则**：flash-bwd 的 dq（跨 kv block）与 dk/dv（跨 group head）两处归约中，把**网格首维取 KV head、group 在核内 Python 展开**可同时（a）使 dk/dv 完全核内累加（零原子 + 确定性写序）、（b）K/V 块核内被 groups 个 head 复用（读量 ÷groups）、（c）把 dq 原子量压到 `B*H*S*D*(S/BM)`（比取 q_head 网格的 dk/dv 原子量小 `2*groups*BM/BN` 倍）。代价是非持久网格多波次（本算子 512/20 = 25.6 波，+1.6%） | P | `examples/grouped_query_attention_bwd/_gqa_bwd_kernel/DESIGN.md#§0.6 R1/R3, §1.6.0 R3/R4` | L0 对照：`n_kv=1` 与 `n_kv>1` 两配置的 dq 逐元素对拍 | tilelang 0.1.2 dev root / CANN 9.0.0 / Ascend910B3（实查 20 AIC） | `tilelang-op-design/references/migration-analysis.md` + `pattern-library/attention.md` |
| **Developer 模式含 gemm 的算子无法用持久化分核时的兜底口径**：`TRAP-DEVMODE-PERSIST-GEMM` 禁 persistent+`T.serial` 多任务+`T.gemm` 后，`core-split-strategy.md` §1 要素③「极大规模 -> 核内串行」路径对该类算子**永不可用**；须走「非持久多波次 + 显式记录残留不均 + causal 逆序调度（重负载块取小 cid）」兜底。建议标准 §1 要素③ 显式补该分支 | R | `DESIGN.md#§5.5③`；`core-split-strategy.md` §1 要素③（未覆盖 gemm 禁用 persistent 的分支） | — | tilelang 0.1.2 dev root / CANN 9.0.0 / Ascend910B3 | `core-split-strategy.md` §1 要素③ |

### Transferable Lessons（可迁移教训）

- **flash-attention 反向迁移的最强 NPU 先例是 `examples/deepseek_v32/sparse_mla_bwd.py`**（Developer MixCV，`T.gemm`×5 的 `b_transpose`/`a_transpose` 组合、`vexp2`、`ds = P⊙(dp−Delta)·sm_scale`、`T.atomic_addx4`、`T.Pipelined`），比按关键词检索到的「attention」先例更贴近；后续 flash-bwd 类迁移应优先查它。
- **`T.atomic_add` 在 NPU 上可用且 fp32 支持**（`docs/Tilelang.language/原子操作/T.atomic_add.md` §2.2.1，降级 `hivm::StoreOp`）；但 `size=` 形态覆盖**连续**区域——BSHD 固定 head 的跨步目标必须退化为 `T.serial` 行循环 + `size=[D]`（先例 `examples/indexer/indexer_bwd.py` L109–111）。若误写整块 `size=[BN,D]` 会**静默写错**。
- **`make_dq_layout` / `make_swizzled_layout` 与 `T.GemmWarpPolicy` / `threads=` / `compile_flags` / `pass_configs` 是 GPU 专属**，迁移时逐项舍弃并给依据（前者唯一目的是 Ampere 标量 `atomicAdd` 向量化；NPU 无此约束）。
- **L0C 容量对「多累加器共存」的算子是主约束**（`CONST-cube-gemm-l0c-traffic-floor`：32768 元素 / 128KB）：flash-bwd 同时驻留 `dk_f+dv_f`（程序级累加器）+ 瞬态 gemm 输出，必须先按容量反推 `BM/BN`（本算子 D=128 时 `BN` 必须从 64 降到 32 才留 25% 余量）。
- **因果三角负载不均**是多波次非持久网格下的临界路径主导项（比核数整除余数 1.6% 大得多）；缓解手段是把重负载（大 `jb`）映射到小 `cid` 优先调度——只改任务顺序、不改结果，是零成本优化。
- **D-1 等价性脚本的合成数据必须遵守算子语义约束**：`lse` 必须是行 logsumexp（否则 `exp2` 溢出，测试自身失真）；「dtype 极值」类角点须缩放到点积可表示（否则参照溢出产生假失败）；判定量只应看「候选 vs 基线」，与 fp64 的偏差仅作信息量。

---

## Stage 2 (design review) Retrospective — first_review — 2026-10-15

> task: `grouped_query_attention_bwd-_gqa_bwd_kernel`（migration-harness，函数 3/3）
> skill: `tilelang-design-review`；结论 **不通过**（1 阻塞 + 5 建议）

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Tooling | `design_calc_check.py` 的 `core_split` 对**一维展平 grid**（`nblk = B·H_kv·ceil(S/BM)`）的设计产生**假阳性 fail**：checker 用 2D `ceil(M/bm)×ceil(N/bn)` 启发式，`M` 误取自 `S=128`、`bn` 误取 `BN=64`，`declared` 又被 `逻辑核数[^\d]*(\d+)` 误匹配成 `0`（文档从未声明 0）。本次耗时自行逐 workload 复算（4/256/512/128/256 全对）才排除误报。skill 规定「`core_split` fail → 维度 3 fail」，会直接误伤正确的设计 | `python3 design_calc_check.py --design ...` → `core_split: fail, recomputed ceil(128/64) x ceil(1/64) = 2 vs declared 0`；文档 §5.5 逐 workload 声明的核数与独立复算一致 | `design_calc_check.py` 的 core_split 增加「一维展平 grid（`nblk`/`num_logical_kernels`）分支」：解析 `nblk = ...` 表达式并校验，无法解析时降级为 `skip` 而非 fail；或在 `tilelang-design-review` Phase 1 步 7 注明「core_split fail 须先判定是否 1D-grid 解析假阳性再定 fail」 | D |
| Tooling | `design_calc_check.py` 的 `ub_budget` 对 §4.5 表**整体 skip**：表首列 cell 为反引号包名（`` `k_s` ``），`re.match(r"^[a-zA-Z_]", c)` 判定为 header 行而跳过，导致「无可解析缓冲表」。若解析成功，本可机械暴露本设计的「小计 108544 vs 行加和 98560」不一致（VP：机械拦截位失效） | 输出 `ub_budget: skip, no per-level buffer tables parsed`；人工加和 98560 B ≠ 声明 108544 B | 解析器允许 cell 以反引号/加粗包裹的 buffer 名（strip ``` ` ```/`**` 后再判断），并在 `合计/小计` 行做 declared-vs-recompute 比对 | D |
| Process | 维度 0 迁移核对清单未单列「**条件分支 / 编译期开关的语义保持**」检查项：本次「源码 `if is_causal:` 守卫的掩码在 §3.3 被判为无条件执行」是维度 0 最重的语义偏差，但按现有表需自己从 §0.1/§8.1/源码三处交叉才定位；表内只有笼统的「语义理解正确性/一致性」 | 源码 `gqa_bwd.py` L590 `if is_causal:`（掩码条件化）vs `DESIGN.md` §3.3 L576-579（无条件掩码）；NPU 测试 `test_grouped_query_attention_bwd.py` smoke 用例 `causal=False` | `tilelang-design-review` 维度 0 表新增检查项「条件语义保持」：迁移时源码的每个 `if <编译期开关>:` 分支（`is_causal`/`return_half`/`inplace` 等）须在 §3.3/§6 有对应条件化实现，缺失即 fail | P |
| Process | 「D-1 脚本与实现常数保真」无检查项：`verify_equiv.py` 用 `-inf` sentinel、实现规格用 `-3.0e38`，二者数学等价但机器验证未覆盖实际常数。现清单只查「脚本结果与内嵌表一致」，不查「脚本是否复刻了实现的常数/分支」 | `verify_equiv.py` L112/L185 `float("-inf")` vs `DESIGN.md` §3.2 L483 `NEG=-3.0e38` | 维度 8 机械复核补一条：确认 verify_equiv 的候选式与 §3.2/§3.3 的常数、分支结构一致（不一致则要求设计补测或显式声明等价依据） | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **一维展平 grid 的 core_split 机械核对**：`nblk = B·H_kv·ceil(S/BM)`（及一切 `T.Kernel(nblk)` 展平设计）不能用 2D `ceil(M/bm)×ceil(N/bn)` 复算；须按展平维定义独立校验，且 checker 无法解析时应 `skip` 而非 `fail` | D | 本检视 `core_split` 假阳性（`declared 0` 误匹配）+ §5.5 逐 workload 复算一致 | `.agents/tools/design_calc_check.py --design examples/grouped_query_attention_bwd/_gqa_bwd_kernel/DESIGN.md` | tilelang 0.1.2 dev root / CANN 9.0.0 | `.agents/tools/design_calc_check.py` + `tilelang-design-review` Phase 1 步 7 |
| **UB 预算表的反引号/加粗解析**：`§4.5` 类表格首列常为 `` `k_s` `` 形式，机械解析须先剥除 markdown 装饰，否则整表 skip、能力拦截位失效 | D | 本检视 `ub_budget: skip`；人工加和 98560 ≠ 声明 108544 | 同上 | tilelang 0.1.2 dev root / CANN 9.0.0 | `.agents/tools/design_calc_check.py` |
| **迁移条件分支语义核对项**：`is_causal` 等编译期开关的源码守卫（`if is_causal:`）在迁移伪代码中易被漏成无条件执行，须列入维度 0 强制核对 | P | 源码 L590 vs §3.3 L576-579；测试 smoke `causal=False` | — | — | `tilelang-design-review` 维度 0 检查表 |

### Transferable Lessons（可迁移教训）

- **迁移核对必须回到源码的 `if <编译期开关>:` 守卫**：`is_causal` 的掩码守卫是本次唯一阻塞项；§0.1 描述正确、golden 正确，但 §3.3 漏守卫——「语义文字对」不等于「实现规格对」，三者（§0.1 / §3.3 / golden）需同时交叉核对。
- **机械 fail 先判定是否为「公式不适用」的解析假阳性**：1D 展平 grid、反引号包名等文档形态会令启发式 checker 误判；`design_calc_check` 为 advisory，须人工复算后再定维度 fail/pass。
- **负向候选的量化倍数要独立重推**：`#3 原子量 16x` 实为 `2·BM/BN = 4x`（漏消 `H_kv/H=1/groups`）；方向性结论对但数字错，说明「复算比值」须逐因子对消、不能只信文档结论。
- **D-1 脚本要复刻实现的常数与分支**：若脚本用 `-inf` 而实现用 `-3e38`，机器验证就未覆盖真实运行路径；等价性声明须与脚本可核范围对齐（或显式说明常数无关性）。

---

## Stage 1 (design) Retrospective — revision v1 — 2026-10-16

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Process | revision 模式下「机械复核（gate 1）通过」与「Stage 2 语义检视通过」是两套判据：v0 的 gate 1 已 pass，但阻塞级的「`is_causal` 掩码未条件化」gate 1 完全不检（`S1-*` 规则只做存在性/文档级检查）。设计者若以 gate 1 pass 自证会漏掉语义分支问题 | `statectl gate 1` 对 v0 返回 `pass: true`，而 REVIEW 判不通过 | 明确 gate 1 仅覆盖「文档结构/存在性」，语义级条件分支核对属 Stage 2 维度 0；Stage 1 revision 须逐条读 REVIEW 而非只看 gate | P |
| Process | 「跨章节一致性」在 revision 中易漏：v0 已在 §0.1/§8.1 正确条件化 `is_causal`，仅 §3.3/§6.2 漏守卫——单点错误散布于多处引用同一语义的章节 | v0 §0.1（正确）/§3.3（错误）不一致 | revision 修复语义类问题时，用 `grep` 全文档定位该语义的**全部**引用点（`is_causal`/掩码/loop_st）逐处同步 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **revision 修复的「语义散发点」清单法**：语义级修复（条件掩码、常数、倍数）须以关键词全文档 grep 定位全部引用点再逐处同步，避免 §0.1/§8.1 对而 §3.3/§6.2 错的跨章节不一致 | R | v0 `is_causal` 散发于 §0.1/§1.4/§3.1/§3.2/§3.3/§6.2/§8.2/§9.2；v1 全部同步 | — | — | `tilelang-op-design` Phase 8 / `tilelang-design-review` |

### Transferable Lessons（可迁移教训）

- **编译期开关的守卫必须与 `loop_st` 同处条件化**：`loop_st = f(...) if is_causal else 0` 与掩码 pass 是同一 `is_causal` 语义的两半，二者必须同时包进 `if is_causal:`——只条件化循环下界而漏掩码，会让 `is_causal=False` 静默算错。
- **主选 API 的「先例引用」要逐行核对 API 名**：`sparse_mla_bwd` 反向核用的是 `T.vexp2(src,dst,tmp)` 而非 `T.exp2`；迁移设计引用先例时须打开该行确认，避免「张冠李戴」。
- **知识库告警与 reviewer 建议冲突时以证据链更强者为准并显式登记**：本次 reviewer 建议主选 `T.vexp2`（文档 + 先例），而 KB `CG-2026-0018`/`VP-2026-0172` 记录了向量 `T.vexp2` 的跨设备 `inf` 告警——处置：**采 vexp2 为主选（文档/先例支撑）**，同时把 KB 告警登记为 Stage 3 首测项并给出回退链（`T.vexp`/标量 `T.exp2`），而非二选一。

---

## Stage 2 (design review) Retrospective — re_review (v1) — 2026-10-16

> task: `grouped_query_attention_bwd-_gqa_bwd_kernel`（migration-harness，函数 3/3）
> skill: `tilelang-design-review`；结论 **通过**（0 阻塞 + 4 建议级残余项）
> 复盘对象为**检视流程自身**（设计问题已随 REVIEW.md 流转，不重复抄入）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Tooling | `design_calc_check.py` 的 `core_split` 对**一维展平 grid** 的假阳性在**同任务第 2 次运行再次复现**（已达 recurring 阈值），仍须人工逐 workload 复算才排除。skill Phase 1 步 7 的硬规则「`core_split` fail → 维度 3 fail」若字面执行会误伤正确设计 | 重跑输出仍为 `core_split: fail, recomputed ceil(128/64) x ceil(1/64) = 2 vs declared 0`；§5.5 逐 workload 复算 4/256/512/128/256 全对 | （强化既有 VP）checker 增 1D 展平 grid 分支或降级 `skip`；skill 明确「core_split fail 须先判定 1D-grid 解析假阳性再定 fail」 | D |
| Tooling | 机械 `ub_budget` 对 §4.5 **再次整体 skip**（表首列反引号包名被当 header）——即便 v1 的小计已订正，机械拦截位对「反引号包名列」的设计仍失效 | 重跑 `ub_budget: skip`；人工加和 106752 B（含 tmp）/ 98560 B（不含）与文中声明一致 | （强化既有 VP）解析器 strip 反引号/加粗后再判 header | D |
| Process | skill 未规定「**结论: 通过** 时残余**建议级**问题的承载位置」：模板只给了「不通过」的「检视问题列表」，而 gate 又禁止「通过」报告出现该章节；本次把 4 项非阻塞残余放入自拟的「检视附注」章节 | 本 REVIEW.md「检视附注：非阻塞残余项（建议级，不影响结论）」；gate 校验「通过时无问题列表」= 0 命中 | `tilelang-design-review` 模板补「通过时残余建议级项」章节（命名非『检视问题列表』），避免 reviewer 自行发明或误删 | P |
| Process | 门禁「无占位符」自检陷阱：reviewer 在『通过』报告里若**引用**被禁字面量（占位符标记）做「已检查」说明，会自我触发 `placeholder_found` 失败 | 本报告初稿维度 7 一句「grep … 无命中」曾含被禁字面量，已改写为「全文扫描无命中」 | skill/gate 说明补一句：自检结论勿逐字引用被禁 token；或 gate 改用「非行内代码/非否定语境」判定 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **1D 展平 grid 的 core_split 假阳性（occurrences=2，recurring）**：同一一维 `T.Kernel(nblk)` 设计在两次 Stage 2 均触发 `core_split: fail`（`declared 0` 误匹配），提示该 VP 已达「跨任务 recurring」标准，应产出 checker 修补提案 | D | 本任务 Stage 2 首检 + 重检两次同一假阳性；§5.5 复算一致 | `design_calc_check.py --design …/_gqa_bwd_kernel/DESIGN.md` | tilelang 0.1.2 dev root / CANN 9.0.0 | `.agents/tools/design_calc_check.py` |
| **「通过」报告的残余建议级项承载章节**：补入 REVIEW.md 模板，明确命名（如「检视附注」）与「不得命名为检视问题列表」，统一各 reviewer 的处理 | P | 本报告自拟「检视附注」章节；gate 禁止通过报告出现「检视问题列表」 | — | — | `tilelang-design-review` REVIEW.md 模板 |
| **『通过』报告自检禁用 token 陷阱**：自检说明不得逐字引用占位符/待办标记，否则 gate `placeholder_found` 自伤 | P | 本报告初稿一句含被禁字面量，改写后 0 命中 | — | — | `tilelang-design-review` 输出规则 / gate 说明 |

### Transferable Lessons（可迁移教训）

- **重检必须逐条回源码/文档复核修复项，不轻信 revision 说明表**：v1 的修订说明逐条列出修复，但结论只能由 reviewer 亲自回源码（`is_causal` 守卫 L580/L590）、回文档（`T.vexp2.md` 签名、`T.gemm.md` 无 policy）、重跑机器验证（verify_equiv 10/10 全 EQUIV_PASS）后给出；本轮 6 项修复项均亲核通过。
- **复核修复项须连同「上一轮建议的固有边界」一起看**：上一轮问题 6（细节引用）只列了 §0.1/§0.5/§11/§1.6.0 几处，未列 `indexer_bwd.py` 行号——故 v1「已修问题 6」不等于「全文引用行号都对」；重检主动扩展扫描（发现 `indexer_bwd` 行号与源码 `loop_st` 行号仍为旧值），并判为**非阻塞建议级**（证据形态真实、结论不受影响）。
- **机械 fail 一律先判「公式是否适用」再定维度**：`core_split` 对 1D 展平 grid 不适用（假阳性），`ub_budget` 因反引号包名 skip；二者均须人工复算后落 pass/warn，不能按模板字面 fail。
- **等价性机器验证须与实现常数/分支逐位一致**：v1 已把 `verify_equiv` 的 sentinel 改为实现的 `-3.0e38`（含 `assert`），重跑与内嵌表逐行一致——reviewer 复核时应同时确认「常数一致」与「脚本分支结构未偏离实现」，而非仅看是否全 PASS。

---

## Stage 3 (develop) Retrospective — retry_impl (attempt 2) — 2026-10-16

入口：`retry_impl`，attempt_index=2。上次调度被中断，交接态为 `python _gqa_bwd_kernel.py --level L0`
→ **SIGSEGV exit 139**（`--help` 正常 → 崩溃发生在首次编译/lowering）。本轮先备份
`history_version/_gqa_bwd_kernel_impl_s3_attempt1.py` 再定位，最终 L0/L1/L2/Boundary 全绿
（37 项 PASS，exit 0），`gate 3` = `pass: true`。

### 根因与修复（4 个独立缺陷）

| # | 缺陷 | 现象 | 根因 | 修复 |
|---|------|------|------|------|
| 1 | `T.if_then_else(<运行时标量比较>, …)` | **SIGSEGV exit 139**（无任何诊断） | 因果掩码写成 `T.if_then_else(kv0+ii <= q0+jj, x, NEG)`；`kv0/q0` 为 block/loop 级运行期标量 → 命中 `TRAP-if_then_else-runtime-scalar-cmp`（`examples/deepseek_dsa_decode/_sparse_mla_kernel/repro/`）。**用移除掩码的变体做二分即从 SIGSEGV 变为可诊断报错**，定位唯一触发行 | 换成矩阵-矩阵向量形态：`T.arange` 建行列索引 → `dif = colmat - rowmat` → `T.vbrc(kv0-q0, lim_mat)` → `T.vcmp(dif, lim_mat, vis, "ge")` → `T.vselect(vis, qkt_ub, neg_buf, qkt_ub)` |
| 2 | 掩码机器放在 `alloc_shared` 上 | **Vector core execution timed out（507034）**，kernel 挂死 | `T.arange/vsub/vbrc/vcmp/vselect` 的 int32/bool/f32 缓冲若用 `T.alloc_shared` 会挂死；同代码换成 `alloc_fragment` 立即跑通（`mv_frag` vs `mv_nosel` 变体对照，且移除 `vselect` 仍挂 → 罪魁是 `vbrc+vcmp`，非 `vselect`） | 掩码机器全部改 `T.alloc_fragment`（与已验证的 `_mha_decode_split_kernel` 一致） |
| 3 | 1-D `[BN]` shared 行向量 + 列读 | 编译失败：`'memref.expand_shape' op has source rank 3 and result rank 2`（kernel.npuir:80:9） | `T.copy(Lse[bz,h,q0:q0+BN], s1d)` + `s1d[j]` 生成的 `tensor.expand_shape [BN]->[1,BN]` 被 BiShengIR `propagate-reshape` **反向折叠到基础 3-D GM memref**（`memref<1x8x128>` → `memref<1x1>`） | 行缓冲改 2-D `[1,BN]` + 2-D 源切片 `Lse[bz,h:h+1,q0:q0+BN]`（`mv_lse2d` 变体：causal=False 直接编译通过） |
| 4 | **golden 的 `torch.unsqueeze(v, 0)`** | kernel 与 golden 的 dq/dk 差 ~100% 参考幅值 | softmax 概率是 `p[q, kv]`（query=行），per-query 的 `lse`/`delta` 必须沿 **dim 0** 广播即 `unsqueeze(v, 1)`；`unsqueeze(v, 0)` 产出 `[1,S]`，torch 按最后一维对齐 → 实际对齐到 **KV** 轴。仅在 `S_q == S_kv` 时静默；且算子自测的 `lse` 沿 S 恒定（`6.0+0.05*randn(B,H,1)` 展开）使 lse 半边**精确抵消**，只有 `delta=rowsum(o*do)` 暴露它 | golden 改为 `torch.unsqueeze(..., 1)`；独立 SDPA 反传对拍从 2.0e+02 降到 1.4e-06 |

> 干扰项排除记录：一度把缺陷 4 误判为「`delta_s[0, jj]` 沿错轴广播」，因为对拍用的是**有缺陷的 golden**。
> 差分实验推翻了该结论：把 delta 行改回索引式 `dp_ub[ii,jj] -= delta_s[0,jj]`（Parallel）与 v-op 形态
> 结果**逐位相同**（max_diff 均 2.179e-03），证明索引式 `[0, jj]` 本身正确。教训见下。

### 4 项残余风险验证（DESIGN/REVIEW 交接项）

| 项 | 结论 | 证据 |
|----|------|------|
| ① `dq` 行段原子并发（`n_kv=1` vs `>1`） | **通过** | L1 `nkv1`（S=64，单 KV 块，无跨块原子）与 `nkv4`（S=256，跨块原子）分别与单块 golden 逐元素对拍通过；Boundary 同题两跑 spread `dq ≤ 3.8e-06 / 2.3e+01`（并发 fp32 原子顺序不同 → 位不相同但量级为累加噪声；位相同不是合理契约） |
| ② `T.Pipelined` 运行期下界 `loop_st` | **通过** | causal 路径 `loop_st = (kv0//BN)*1` 为运行期值（非 0 起始），全部 causal 用例编译并通过；非 causal 走 `loop_st=0`。DESIGN §9.4 的回退（0 起始 + 全程掩码）**无需启用** |
| ③ 向量 `T.vexp2`（`P` 的 isfinite） | **通过（未命中 KB 告警）** | `T.vexp2(qkt_ub, qkt_ub, tmp)` 保留；若 P 出现 `inf` 则 dq/dk/dv 必然污染，而全部 37 例的最大误差均在 fp16/bf16 操作数量化底噪（~半 ulp）内，故 P 有界 (`lse=rowmax` 的 Boundary-special 例亦然，输入 ×4、x≤0) → 无需回退 `T.vexp` |
| ④ bf16 路径 Cube 操作数 dtype | **通过** | `p16/ds16` 随 `dtype` 参数取 bf16，与 `k_s/v_s/q_s/do_s` 一致；bf16 全部用例通过（4 个 manifest workload + groups/边界），误差 ~1.7e-3 相对张量尺度 ≈ bf16 半 ulp |

### 精度门槛修订（**已记录的偏离**，非实现缺陷）

DESIGN §8.2 的 fp16 `(atol=1e-3, rtol=1e-4)` / bf16 `(5e-3, 1e-3)` **低于该设计自身算法的量化底噪**：
与 GPU 源算子一致，`P`/`ds` 被显式 `T.vcast` 到存储 dtype 后再做 `dv/dk/dq` 的 GEMM（源
`gqa_bwd.py` L581/L604 的 `qkt_cast`/`dst_cast`）。round-to-nearest 的算子树化贡献 ≤ 半个 ulp
的相对误差（fp16 2.4e-4，bf16 2.0e-3），任何累加顺序都无法消除；实测相对张量尺度误差
fp16 ≈2.4e-4、bf16 ≈1.7e-3。故 `_TOLERANCE` 改为 **scale-aware**：有效绝对下限
`max(floor, scale*max|ref|)`，fp16 = `(rtol 1e-3, scale 1e-3)`、bf16 = `(rtol 1e-2, scale 1e-2)`，
对实测底噪留 4–6× 余量；真实逻辑错误仍是 O(1) 相对偏差，必被拦截。
反证实验：把 `P`/`ds` 改回 fp32 作 GEMM A 操作数（`mv_f32` 探针）→ 编译通过但**静默错误**
（dq max_diff 3.8e+02 vs 参考 1.25e+01），故存储 dtype 折算必须保留。
该偏离属「等价替换 + 记录理由」，同时作为 [[DESIGN_ERROR]] 候选上报（§8.2 阈值表内部不自洽）。
**注**：源算子 `p16/ds16` 折算语义未改动，仅测试判据修订。

### Skill Flow Issues

| 类别 | 现象 | 证据 | 建议 |
|------|------|------|------|
| Debug 流程 | `tilelang-debug-helper` 的「最小复现缩减」在**纯 Vector 小内核**上失效：无 `T.gemm` 的 Developer 模式小内核即使含常量乘法也返回**全零**（`test_brc` form g），据此得出的广播结论全属假阳性 | `test_brc.py ... form a/b/e/f/g/h/i` 全部 max_diff≈2e+02 且 out 全 0；加入 `T.gemm` 后才复现真实现象 | helper/skill 补一条：npuir Developer 模式的最小复现必须**保留至少一个 Cube op**（MixCV），否则降级路径不同、结论不可信 |
| 定位手法 | 缺陷 4（golden 错轴）被「对拍基准本身有缺陷」掩盖，代价是数轮无效 kernel 改动 | `golden_check.py`：golden vs SDPA 2.0e+02；修正 unsqueeze 后 1.4e-06 | 建议在 `tilelang-op-develop` 的首测流程中加一条硬规则：**任何一个「kernel 与 golden 差 O(1)」的判定，必须先跑「golden vs 独立参考（SDPA/autograd）」，再动 kernel** |
| 定位手法 | `assert_close` 失败消息里逐张量明细被 `str(e).splitlines()[1]` 取空，报告显示 `dq:` 空行，无法判断是哪个张量 | 首次 L0 失败输出 `dq:\ndk:\ndv:` 全空 | 模板改为直接输出各张量 `max_diff/ref_scale/atol_eff`（已在本产物落地） |
| 门禁 | `T.vcmp` 掩码方案触发编译告警 `Op 'hivm.hir.vcmp' will execute by scalar instruction with low effiency`（功能正确，仅性能） | 每个 causal 变体编译日志 | 作为 Stage 4 性能调优的已知起点登记 |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **golden 的 `unsqueeze(v,0)` 把 per-query 量对齐到 KV 轴，且被「沿 S 恒定的 lse」静默掩盖**：显式 attention 反向 golden 高频陷阱；必须用随 S 变化的量（delta）或独立 SDPA 反传才能暴露 | P | 本任务：golden vs SDPA 2.0e+02 → 修正后 1.4e-06；kernel 与修正后 golden 全绿（37 例） | `examples/grouped_query_attention_bwd/_gqa_bwd_kernel/repro/TRAP-golden-unsqueeze-axis.py`（CPU，秒级，PASS） | tilelang 0.1.2+7b8d1ac9 / CANN 9.0.0 / Ascend910B3 / torch 2.7.1 | `.agents/skills/tilelang-op-develop`（golden 编写规则）+ pattern-library traps-numerics |
| **掩码/索引向量机器在 `alloc_shared` 上会挂死 vector core（rc=507034 timeout），`alloc_fragment` 正常** | D | 同 kernel 仅改掩码缓冲作用域：shared → timeout（`mv_nosel`/`mv_frag` 对照；移除 `vselect` 仍挂 → 触发点是 `vbrc+vcmp`）；fragment → 通过 | —（探针未转正：挂死型探针会污染 NPU 状态，未做有界复现） | 同上 | pattern-library traps-runtime；`tilelang-vector-skill`/`tilelang-mixcv-skill` 的缓冲作用域选型 |
| **1-D `[BN]` shared 行向量 + 列读触发 BiShengIR `propagate-reshape` 把 `expand_shape` 折回基础 3-D GM memref**（rank 3 → rank 2 编译错） | D | kernel.npuir:80:9 报错；改 2-D `[1,BN]` + 2-D 源切片后编译通过 | —（同上，已在 RETROSPECTIVE 记录 IR 证据） | 同上 | pattern-library traps-compiler；`docs/Tilelang算子调试指南.md` |
| **npuir Developer 模式最小复现必须保留 Cube op**：无 `T.gemm` 的纯 Vector 小内核返回全零，导致假阳性结论 | P | `test_brc` 全 9 个 form 均 out≡0；加 `T.gemm` 后才可信 | — | 同上 | `tilelang-debug-helper` 最小复现规则 |
| **「kernel 与 golden 差 O(1)」时先验证 golden**，而非先改 kernel | P | 本任务因先改 kernel 浪费数轮；golden vs SDPA 一步定位 | — | 同上 | `tilelang-op-develop` 首测流程 |

### Transferable Lessons（可迁移教训）

- **对拍基准必须先自证**：kernel 与 golden 出现 O(1) 差异时，**先**用独立参考（SDPA/autograd）验证 golden，**再**动 kernel。本轮反了顺序，代价是数轮无效改动。
- **恒定量会掩盖轴错误**：测试输入若让某个向量沿被测轴恒定（本任务 lse 沿 S 恒定），任何「轴对齐」错误都会精确抵消。写 golden/测试时必须让**每个参与广播的量都沿两个轴变化**，否则该量不构成任何验证力。
- **`unsqueeze` 的广播轴要按 `[行,列]` 语义显式确认**：`p[q, kv]` 下 per-query 量必须 `unsqueeze(1)`；`S_q == S_kv` 会让错轴版本静默通过形状检查。
- **最小复现的形态决定结论可信度**：npuir Developer 模式存在多套降级路径，纯 Vector 小内核与 MixCV 内核行为不同（前者可返回全零）。复现必须与真实算子同形态（含 Cube op），否则「定位」到的可能是 harness 假象。
- **SIGSEGV 无诊断时用「逐块移除 + 二分」定位**：把可疑语法结构整块替换后若崩溃变为可诊断编译错，即锁定该结构为触发点（本轮据此从 SIGSEGV 定位到 `T.if_then_else` 运行期标量比较）。
- **精度阈值须与算法的量化底噪自洽**：先算出「算子链上最粗的量化点」（此处 P/ds 落 fp16/bf16 = 半 ulp），再定阈值；否则阈值不可达，会诱使实现者去做无意义的「提精度」甚至误改语义。
