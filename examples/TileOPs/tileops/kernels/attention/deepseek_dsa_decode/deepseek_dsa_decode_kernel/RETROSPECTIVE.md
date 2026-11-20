# RETROSPECTIVE — `_sparse_mla_kernel` (deepseek_dsa_decode, migration-harness)

## Stage 1 (design) Retrospective — first_design — 2026-10-15

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 机械门禁 `S1-PLACEHOLDER`（`TEMPLATE_BRACE_RE`）把含 CJK 的 LaTeX/集合记号（如 `Σ_{j : idx_j 有效}`）判为未替换模板变量——合法数学式必须改写为无花括号散文形式才能过门禁 | `examples/deepseek_dsa_decode/_sparse_mla_kernel/DESIGN.md` §0.1（首版 `{j : idx_j 有效}` 被判 fail）；`.agents/tools/gate_lint.py` L40/L216 | 在 `tilelang-op-design` SKILL.md 或 `design-template.md` 加一条「数学记号避免在花括号内混入 CJK（改写成散文或加 `,、∈` 分隔符）」的写作提示 | R |
| Design-flow | 机械门禁 `S1-EQUIV-EXEC` 对 §1.6.1 区域全文扫描字面量 `EQUIV_FAIL`——即便它只出现在**被否决候选**的记录行（采纳项全 PASS），也判 fail；设计文档被迫回避该字面量，削弱「否决证据留痕」的清晰度 | `examples/deepseek_dsa_decode/_sparse_mla_kernel/DESIGN.md` §1.6.1 被否决候选表（首版含 `EQUIV_FAIL` 被判 fail，改为「未通过」后通过）；`.agents/tools/gate_lint.py` L327-332 | `gate_lint.py` 的 `EQUIV_FAIL` 检查应限定在**采纳项结果表**内（或允许被否决行用带标注的形式），使否决项可如实留痕 | R |
| Info-source | `constants.md` 缺 Ascend910B3 的核→GM 读带宽条目；`CONST-elu-910B3-bw-floor` 是逐元素聚合 R+W 口径、`CONST-ascend910-9362-profile` 属 910_9362，设计期 roofline 只能给宽区间 | `examples/deepseek_dsa_decode/_sparse_mla_kernel/DESIGN.md` §1.6.0 R4 定量 roofline；`.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md` | 建议以设计期探针（D-3）实测 910B3 核→GM/L2 读带宽并回写 `constants.md`（新 D 类条目） | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| softmax scale 折入 Q（`(QKᵀ)·s → (Q·s)Kᵀ`）在 fp16 通过、**bf16 失败**（违反率 4.55%）——bf16 8 位尾数下 `Q·s`（s≈0.042）舍入经长内积放大，超容差；大 D（512）注意力**不得**用此优化 | D | `examples/deepseek_dsa_decode/_sparse_mla_kernel/DESIGN.md` §1.6.1 OPT-X 行 + `verify_equiv.py` `[OPT-X ... REJECTED]` 臂 | `python3 examples/deepseek_dsa_decode/_sparse_mla_kernel/verify_equiv.py`（torch CPU；bf16 臂 max_abs 9.766e-02 / viol 4.550e-02） | torch CPU（dtype 级结论，非设备绑定）；仓库 tilelang 0.1.2+00aeb862 | `algorithm-candidates.md` ALG-attention / `layout.md` |
| 大 D 稀疏 MLA（D=512）在 NPU 上：源 Hopper `H_per_block=128`（KV 只读 1 遍）**因 L0C=128KB 不可行**（`acc_o[128,512]f32=256KB`），head 分块引入 **KV 重读倍数 = H/block_heads** 成为 roofline 主导项；`block_heads`（32/64）是把「KV 重读」换成「UB/L0C 容量压力」的旋钮 | P | `examples/deepseek_dsa_decode/_sparse_mla_kernel/DESIGN.md` §1.6.0 R3/R4（候选 0 容量淘汰、候选 2/2′ 对比）+ §1.6.3 实验裁决三件套 | 设计期分析（无设备实测）；常数见 `CONST-capacity-910B2C`（L0C 128KB） | tilelang 0.1.2+00aeb862 / Ascend910B3 / 2026-10-15 | `pattern-library/attention.md`（新增「大 D 稀疏 MLA head 分块 ↔ KV 重读」条目） |
| Developer 模式「persistent 分核 + `T.gemm` + v-prefix」不可用（CG-2026-0010）在本任务**独立第二识别** → `recurring`：极大规模（逻辑核 4096 ≫ 物理核 20）无法走标准核内 `T.serial` persistent，被迫非持久 + 波次残余（≈0.5%） | D | `examples/deepseek_dsa_decode/_sparse_mla_kernel/DESIGN.md` §5.5 + §9.4；`.agents/evolution/capability-gaps.md` CG-2026-0010（occurrences 1→2, status recurring）+ `traps-runtime.md` TRAP-DEVMODE-PERSIST-GEMM | `python repro/TRAP-DEVMODE-PERSIST-GEMM.py`（pattern-library）；本任务实态：mainstream 逻辑核 4096 | tilelang 0.1.2+1990aa9fe4 / 910B2C（首证）；本任务 0.1.2+00aeb862 / Ascend910B3 | `capability-gaps.md` CG-2026-0010（已更新 proposal） |

### Transferable Lessons（可迁移教训）

- **bf16 精度契约下，任何「把标量折进输入张量」的优化（scale 折入 Q/K）都必须先机器验证**：bf16 只有 8 位尾数，折入的张量值域变化会经内积放大，fp16 通过不代表 bf16 通过（本例 fp16 PASS / bf16 违反率 4.55%）。
- 稀疏 MLA 的 KV gather 是**行连续**（每行 = Dqk×dtype 字节）→ 用串行 `T.copy` 即可，**不触发** `TRAP-C10`（跨步 gather 缺失）；只有**元素级跨步** gather 才受该陷阱约束。
- 大 D 注意力在 NPU 上的**首要设计约束是 L0C = 128KB**：`block_heads · (D + block_N) · 4 ≤ 128KB` 直接封顶 `block_heads`，由此产生的 **KV 重读倍数（H/block_heads）** 是 roofline 主导项——应优先评估「acc_o 驻 UB（`alloc_shared`）以放大 block_heads」的备选（本例 BH=64）。
- Developer 模式写 npuir 的注意力/稀疏算子：**不要用 persistent 核内多任务循环**（`TRAP-DEVMODE-PERSIST-GEMM`）；逻辑核远大于物理核时用非持久一维 grid + 硬件波次调度，并把波次残余如实记录。
- 迁移设计文档写数学式时避免 `{...}` 内含中文（门禁 `S1-PLACEHOLDER` 会误判）；记录被否决候选时避免字面量 `EQUIV_FAIL`（门禁 `S1-EQUIV-EXEC` 全文扫描）。

## Stage 2 (design-review) Retrospective — first_review — 2026-10-15

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | `design_calc_check.py` 的 `core_split` 检查对「逻辑核数由派生表达式给出」的设计解析失败（本设计 `num_logical = B·S·(H//block_heads)`），回退为 `declared logical cores = 1` 并 skip——检视侧只能人工复核 `§5.5`，机械拦截位在此类设计上失效 | `python3 .agents/tools/design_calc_check.py --design examples/deepseek_dsa_decode/_sparse_mla_kernel/DESIGN.md` → `core_split: skip, "insufficient parsed dims {M:32,N:64,bn:64}; declared logical cores = 1"`（实际 §5.5 num_logical=4096） | `design_calc_check.py` 的 `core_split` 解析器应支持 `num_logical = <expr>`（含 `//`、乘积、`T.Kernel(expr, is_npu=True)`）的符号/常量折叠，或将无法解析时的 `declared logical cores` 明确报为 UNKNOWN 而非 1 | R |
| Review-flow | 维度 8「调研结论正确性」要求独立复算 roofline，但 `constants.md` 无 Ascend910B3 核→GM/L2 读带宽条目（`CONST-elu-910B3-bw-floor` 为逐元素聚合口径、`CONST-ascend910-9362-profile` 属 910_9362），检视只能核对区间算术自洽（4.0–9.4ms），无法核验常数本身 | `examples/deepseek_dsa_decode/_sparse_mla_kernel/DESIGN.md` §1.6.0 R4；`.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md` | 同 Stage 1 的 D 类：以设计期/Stage 4 探针实测 910B3 核→GM/L2 读带宽并回写 `constants.md`；检视维度 8 可增加「roofline 依赖的常数是否均为 verified 且设备匹配」的显式核对项 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 迁移维度 0 的源码核对必须覆盖**由 head_kv 派生的 grid 尺寸量**：`head_kv = heads // kv_group`（≠ `H_kv`）→ `replicate_h = head_kv // 64`（>64 时）→ `h_per_block`；本任务 `heads=128, kv_group=1` 得 `replicate_h=2, h_per_block=64`，而设计因把 `head_kv` 误作 `H_kv` 断言 `replicate_h=1`/`全头单 block`，错误值传播至 §0.4/§0.5/§0.6/§1.4/§1.6.0 | P | `tilelang-design-review` 维度 0 检视：源 `deepseek_dsa_decode.py` L78/111/130-137/204-210 vs `DESIGN.md` §0.3 L101/§0.6 R4 L199/§1.4 L294/§1.6.0 L367 | 源码静态核对（无需设备） | tilelang 0.1.2+00aeb862 / 2026-10-15 | `.agents/skills/tilelang-design-review/SKILL.md` 维度 0「算法解读完整性」检查项：增补「grid 维度派生量（replicate_h/h_per_block/h0/head 复制）须按派生源核对，警惕 H_kv/head_kv 混淆」 |
| 维度（API 可行性/维度 8）须逐 op 核对**正向** dtype 论断，不止核对负向断言：本设计称「`T.vexp` 覆盖 fp16/fp32/bf16」，而 `docs/Tilelang.language/数学操作/T.vexp.md` §2.2.1 与 `T.vexp2.md` 均对 bf16 标 ×；同类「文档矩阵 ≠ 设备实现」分歧见 `TRAP-bf16-vop-doc-matrix-divergence` | P | `tilelang-design-review` 检视：`DESIGN.md` §1.6.1 OPT-1 L433/§9 R-13 L1112 vs `docs/Tilelang.language/数学操作/T.vexp.md` §2.2.1/`T.vexp2.md`；`pattern-library/traps-compiler.md` TRAP-bf16-vop-doc-matrix-divergence | 文档静态核对 | 仓库 doc（无版本绑定） | `.agents/skills/tilelang-design-review/SKILL.md` 维度 1/维度 8「替代 API 可行性」检查项：增补「正向 dtype/代价论断同样须逐 op 对照 `docs/…/*.md` dtype 矩阵，分歧处引用 TRAP-bf16-vop-doc-matrix-divergence 并要求设备 micro-probe」 |

### Transferable Lessons（可检视流程可迁移教训）

- **迁移检视的“最高收益动作”是亲自重算源算子里的派生量**：本设计中 `heads=128 / kv_group=1` 是 manifest 明文，但由 `head_kv = heads // kv_group` 派生的 `replicate_h`（=2）被误算成 1——只读 §0 自述会漏检，必须打开源码逐条复算 `L78/L111/L130-137/L204-210`。已作为 Value Point 落 `tilelang-design-review` 维度 0。
- **“同一错误值在多节重复出现”不是一致性，而是同一个错误读数被复制**：§0.3/§0.6/§1.4/§1.6.0/§3.5.2 五处一致地写 `replicate_h=1 / 256KB`，一致性检查全过——检视不能以“多节互洽”代替“与源码互洽”。
- **通过与否不能看任务叙述的语气**：调度 prompt 把多数核对项写成“请确认……是否可行”，易诱导放水；本检视凭源码证据判维度 0 fail。**正向 API 论断与负向论断同权核对**（本设计 `T.vexp` bf16 正向论断即与文档矛盾）。
- **机械复核只覆盖可机检子集**：`verify_equiv.py` 重跑与内嵌表逐项一致（EQUIV_PASS）**不等于**设计整体通过——等价性、调研算术均与结论独立；`design_calc_check` 的 `ub_budget`/`core_split` 双 skip 必须由人工补齐（本设计 UB 小计与所列行算术不符即在此人工段发现）。

## Stage 1 (design) Retrospective — revision 1 — 2026-10-15

> 触发：Stage 2 检视不通过（1 阻塞 + 4 建议），按 REVIEW.md 修订 `DESIGN.md` + `verify_equiv.py` docstring。

### 修订摘要与根因

| # | 检视问题 | 根因 | 修订动作 |
|---|---|---|---|
| 阻塞 1 | `replicate_h` 误读（源 head 复制维） | 把 `head_kv = heads // kv_group`（=128）误当作 `H_kv`（=kv_group=1），断言 `replicate_h=1`；遗漏源优化 O17 | §0.3 改为逐源码行号代入参数推导（`head_kv=128>64 ⇒ replicate_h=2, h_per_block=64`，源 grid x=`S·replicate_h=2048`）；§0.4 补 O17、§0.5 补四态处置（重新设计→R4）、§0.6 R4 承接；重算 §1.4/§1.6.0 R3/R4/§3.5.2（源寄存器态 `2×[64,256]f32=128KB`；源 KV 遍历 2 遍/5.12GB） |
| 建议 2 | `T.vexp` bf16 正向论断与文档矛盾 | 设计期未逐 op 读 dtype 矩阵（`T.vexp.md`/`T.vexp2.md` §2.2.1 均标 bf16 ✗） | §1.6.1 OPT-1/§0.5 O10/§3.5.2/§9 R-13/L0-2 措辞更正为「exp 在 fp32 域执行故不触发缺口」；引用 `TRAP-bf16-vop-doc-matrix-divergence` |
| 建议 3 | §4.5 UB 小计算术三处不自洽 + 膨胀系数取低档 | 小计列与所列行/注释不一致；只取 `×1.10–1.12` 未列 `~1.7x` 档 | §4.5 统一为逐行求和 173,568 B≈169.5KB；显式取两档（`~1.7x`→288KB>192KB ⇒ 削减方案改为必需，削减态≈41.5KB）；§1.6.3 BH=64 补「膨胀 >1.26x 则不可行」阈值分支 |
| 建议 4 | §1.6.3 未枚举 softmax 归约 lane 映射 | 只标「水平归约」，未把垂直扫描列为并列候选 | §1.6.3 补 lane 映射候选表（L1 水平 vs L2 垂直扫描）+ 量化弃选理由（发射 ×64 + 列主序 repack） |
| 建议 5 | `verify_equiv.py` docstring 把 OPT-2 误标为「fold scale into Q」 | 文案错位（代码标志/结果正确） | docstring OPT-2 改为「重结合 + `m·s` CSE 提升」；`Q~=Q·s` 归入被否决段 OPT-X |

### Retrospective corrections（修正上文历史条目）

- 上文 Stage 1 first_design 的 Value Point「大 D 稀疏 MLA…源 Hopper `H_per_block=128`（KV 只读 1 遍）…`acc_o[128,512]f32=256KB`」**数值有误**，应更正为：源 `replicate_h=2`（每 s 2×64 头 block）、KV 读 2 遍、源寄存器态 `2×[64,256]f32=128KB`（+`acc_s`16KB≈144KB）。**结论（源全头累加在 L0C 不可行、须 head 分块、KV 重读是 roofline 主导）不变**。

### Transferable Lessons（可迁移教训，本次新增）

- **迁移解读里凡出现「由某个 host 常量派生、又用于 grid/循环上界」的量，必须写明代数并代入本任务实参**：`head_kv=heads//kv_group` 与 `H_kv=kv_group` 形近义异，一字之差使 `replicate_h` 从 2 变 1、源 grid 尺寸与累加器容量全部错 2×。写设计时应把派生链写成 `L78 → L111 → L130-137 → L204-210` 的可复算形式，而非只给结论。
- **「遗漏源优化手段」常与「误读派生量」同源**：head 复制（O17）既是 grid 维度来源、又是 register-pressure 优化；一旦误读就同时丢失「一个维度」与「一项优化意图」→ §0.4/§0.5/§0.6 连锁失效。补优化清单时应回看每个派生量的**动机**。
- **UB「小计」列必须由所列行机械求和（含小 buffer），膨胀系数须把常数条目的最高档（`~1.7x`）与低档（`×1.10–1.12`）都列出并给出取档依据**；只取低档会把「不可编译」误判为「贴限」。

## Stage 2 (design-review) Retrospective — re-review (revision 1) — 2026-10-15

> 触发：修订版 DESIGN.md 重检（上一轮 1 阻塞 + 4 建议）。本轮结论 **通过**（9 维度全 pass，0 阻塞 / 0 建议）。
> 复盘对象为**检视流程自身**；设计内容问题已随 REVIEW.md 流转，不在此重复。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 维度 1/2 无显式检查项要求核对「**Cube 累加器放置形态**（L0C `alloc_fragment` vs UB `alloc_shared`）+ 每迭代 Vector rescale + `T.gemm` 回累加」这一组合是否有本仓先例；本轮须人工检索多个先例才发现本设计主选（L0C fragment + `T.vmul(acc_o,alpha,acc_o)` + `T.gemm(...,initC=False)`）有直接先例，而最常见的同族先例 `sparse_mla_fwd.py` 反而用 UB shared——若只查被引先例会误判 API 可行性 | 设计主选：`DESIGN.md` §3.3（`acc_o=T.alloc_fragment` L0C）+ §4.3；直接先例：`examples/deepseek_v4/example_sparse_mla_fwd_kernel.py` L95/L135/L139/L141；对照形态：`examples/deepseek_v32/sparse_mla_fwd.py` L81（`acc_o=alloc_shared` UB） | `tilelang-design-review` SKILL 维度 1/2 增补「Cube 累加器放置形态（L0C fragment 与 UB shared 二形态）在带 per-iteration Vector rescale 的算法中须逐形态找 `examples/` 先例佐证，不得以单一先例推断」 | P |
| Review-flow | 维度 7「内部一致」未把「§4.5 结论性强制项（削减方案为必需）↔ §3.3 伪代码 buffer 声明」纳入交叉核对；本轮发现 §3.3 仍物化 `alpha_mat`/`l_mat`（与 §4.5「不物化」结论措辞张力），仅因 §9 R-2/R-3 已承接而未升级为阻塞——此类「结论节 vs 伪代码节」的不一致易被逐节自洽掩盖 | `examples/deepseek_dsa_decode/_sparse_mla_kernel/DESIGN.md` §3.3 L727-728（物化 `alpha_mat`/`l_mat`）vs §4.5 L907-916（削减方案为必需）+ §9 R-2/R-3 | `tilelang-design-review` SKILL 维度 7 增补「§4.5/§1.6.x 的结论性强制项须与 §3.3 buffer 声明逐项交叉核对」 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 复核「上一轮阻塞项是否真正修复」不能只看被点名的节：本例 `replicate_h` 更正后必须回源码复核**整条派生链** `L78 head_kv → L111 padded_h → L130-137 replicate_h/h_per_block → L164-168 grid x → L204-210 s_i/h0` 每一步均与新读数自洽，并逐处重算全部派生数字（§1.4/§1.6.0 R3/R4/§3.5.2 的 256KB→128KB、1 遍→2 遍、5.12GB） | P | 本轮检视：源码 `deepseek_dsa_decode.py` L78/111/130-137/164-168/204-210 vs `DESIGN.md` §0.3/§0.4 O17/§0.5/§0.6 R4/R6/§1.4/§1.6.0/§3.5.2 | 源码静态核对（无需设备） | tilelang 0.1.2+00aeb862 / Ascend910B3 / 2026-10-15 | `.agents/skills/tilelang-design-review/SKILL.md` 维度 0「设计与决策一致性」检查项：增补「修订后重检须沿派生链逐级复核并清点全部派生数字处」 |
| 迁移检视「无新语义偏差」的确认动作：对修订中被改动的每个派生量，须回源码核对**其上游输入与下游用途**是否同步更新（本例 O17 既是 grid 维度来源又是 register-pressure 优化，误读会同时丢失维度与优化意图）；另须做一次全文检索确认旧错误读数无残留（仅允许出现在历史修订说明的勘误引用中） | P | 本轮检视：`grep` 确认 `replicate_h=1`/`256KB`/`67 KB` 仅存于 §修订说明；O17 处置链 §0.4→§0.5→§0.6 R4 完整 | `grep -n "replicate_h=1\|256KB\|67 KB" DESIGN.md` | tilelang 0.1.2+00aeb862 / 2026-10-15 | `.agents/skills/tilelang-design-review/SKILL.md` 维度 0 |

### Transferable Lessons（可检视流程可迁移教训）

- **重检修订版时，「修复效果」= 数字更新贯穿全部派生处 + 源码派生链逐级自洽**，而非被点名节的文字已改。逐处清点派生数字（本例 5 节）比只读修订说明更可靠；同时全文检索旧错误读数，只允许其出现在勘误引用中。
- **正向可行性与负向否决同权、且都要落到本仓先例**：MixCV 中「Cube 累加器放 L0C fragment 还是 UB shared」直接决定能否每迭代 Vector rescale——必须找带该组合的先例（本例 `example_sparse_mla_fwd_kernel.py`），不能凭「同族先例 `sparse_mla_fwd.py` 能跑」推断（它用 UB）。
- **「结论节」的强制项要回落到「伪代码节」核对**：§4.5 判定「不物化削减方案为必需」时，§3.3 若仍声明物化 buffer 即存在实施歧义——检视须把结论性强制项与 `T.alloc_*` 声明逐项对齐，而非按节独立判自洽。

## Stage 3 (develop) Retrospective — first_impl (attempt 1) — 2026-10-15

> 触发：首次实现（`first_impl`），终态 `[PRECISION_PASS]`。L0 5/5、L1 5/5 pass；
> L2 4 pass + 1 warn（BH=64 不可编译）；Boundary 4 pass + 4 契约拒绝 pass。
> 本复盘覆盖整条 attempt 链（含 2 次编译失败的实证）。
> 工具链戳：tilelang 0.1.2+00aeb862 / Ascend910B3（aicore=20, aiv=40）/ torch_npu 2.7.1.post4 / 2026-10-15。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | `tilelang-op-develop` Phase 2/5 的「常见错误」表没有「MLIR verifier / BiShengIR pass 报错 → 回到**谓词掩码的构造形态**」这一诊断入口；本轮算子掩码（列谓词）连续 2 次编译失败，第 3 次才定型。错误报文（`hivm.hir.vcmp` verifier、`insert_slice size mismatch`、`memref.expand_shape`）完全不指向「写法选择」这一根因 | `examples/deepseek_dsa_decode/_sparse_mla_kernel/_sparse_mla_kernel.py` 掩码段落注释 + `repro/TRAP-if_then_else-runtime-scalar-cmp.py`、`repro/TRAP-parallel-GM-row-into-2D-fragment.py` | `tilelang-op-develop` Phase 2 增补「谓词掩码构造 checklist：`T.vbrc`（标量/行/列广播）+ `T.vcmp`（张量-张量）+ `T.vselect`；禁用 `T.if_then_else` 带 runtime scalar；`T.Parallel` 体内禁止裸 GM 行赋值到 2D fragment」 | P |
| Develop-flow | 无「探针 kernel 必须 ≥2 block」的提示：`T.Kernel(1)` 会把块索引常量折叠，使 runtime-scalar 代码路径**完全不被覆盖**——本轮 5 个隔离探针因此连续漏报（见 Value Point 1 的分支矩阵） | `repro/TRAP-if_then_else-runtime-scalar-cmp.py`（`nblk=1→OK` / `nblk=4→FAIL` 实测）；同源证据：`/tmp/opencode/variant_g.log` | `tilelang-op-develop` Phase 2 增补「最小复现探针必须 `T.Kernel(≥2)`，否则块索引被折叠、runtime 标量路径不触发」 | P |
| Develop-flow | 分层测试模板（`mha_decode_paged` 先例）只有单一容差槽位；当 design 的 blocking 容差（1e-2 项目标准）与集成 harness 断言容差（manifest `atol=3e-4/rtol=1e-5`）不一致时，实施者须自行设计「strict/loose 双报告」才不会把「harness 将失败」掩盖成简单 PASS | `DESIGN.md` §8.2 R-7 vs `_sparse_mla_kernel.py::_compare/_run_case` 的 `strict_ok`/`loose_ok`；实测 `L0/L1` 输出行 | `tilelang-op-develop` 模板的 `run_case` 增补「design blocking 容差 与 集成 harness 容差 双轨记录（strict 仅记录不阻塞）」槽位，并在 Phase 5 要求交付时披露两者差值 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| npuir Developer 下 `T.if_then_else(<int load> <= <runtime scalar>, a, b)` 位于 `T.Parallel` 内会被 mis-lower：标量被 `arith.trunci(i32->i1)` + `vbrc` 成 i1 张量后送 `hivm.hir.vcmp`（verifier 报 "operand at idx 0 and 1 should have element type ..."）。触发条件含「`T.Kernel(>1)` 使标量不可常量折叠」——`T.Kernel(1)` 探针**看不到**该 bug。绕法：`T.vbrc`(标量→张量) + `T.vcmp`(张量-张量) + `T.vselect` | D | `_sparse_mla_kernel.py` 掩码段落（首次失败 run1）；IR 证据 `arith.trunci(%161):(i32)->i1` → `vbrc` → `vcmp(tensor<32x64xi32>, tensor<32x64xi1>)` | `python examples/deepseek_dsa_decode/_sparse_mla_kernel/repro/TRAP-if_then_else-runtime-scalar-cmp.py`（断言编译失败，exit 0） | tilelang 0.1.2+00aeb862 / Ascend910B3 / torch_npu 2.7.1.post4 / 2026-10-15 | `pattern-library/traps-compiler.md`（新增条目）/ `.agents/skills/tilelang-vector-skill/SKILL.md` |
| npuir Developer 下 `T.Parallel(M,N)` 体内**纯赋值** `frag[i,j] = GMRow[j]` 会把 GM 行读成 (1,N) 并 `tensor.insert_slice` 到 (M,N) → "expected type to be 'tensor<32x64xi32>' or a rank-reduced version. (size mismatch)"。同一行**放在 `T.if_then_else` 表达式里**却能正确广播（同为 mis-lowering 的伴随现象）。绕法：`T.copy(row, row_frag(1,N), size=[1,N])` + `T.vbrc((1,N)→(M,N))` | D | `_sparse_mla_kernel.py` 第 2 次编译失败；IR 证据 `%103 = tensor.insert_slice((1x64xi32, 32x64xi32)) static_sizes=[32,64]` | `python examples/deepseek_dsa_decode/_sparse_mla_kernel/repro/TRAP-parallel-GM-row-into-2D-fragment.py`（断言编译失败，exit 0） | tilelang 0.1.2+00aeb862 / Ascend910B3 / 2026-10-15 | `pattern-library/traps-compiler.md`（新增条目） |
| **（正向）Developer `T.gemm(src1, src2, dst, initC, size=[a,b,c])` 的前缀列语义实测成立**：`dst[M,c] = src1[M,b] @ src2[b,:c]` 只读 `src2`（行跨度 > c）的前 c 列——`(32,64)@(64,576)→(32,512)` max_diff=3.906e-3。**这直接关闭 `DESIGN.md` §9.1 R-9 的未实测项**，使 PV 可复用单次 576 宽 gather，不必付 R-9 备选的 `+89%` gather 代价 | D | `_sparse_mla_kernel.py` PV 行（`size=[block_heads, block_N, D]`，实跑通过）；上游 Expert 先例 `examples/deepseek_v32/sparse_mla_fwd_dynamic_shape.py` CubePVMatmul | `python examples/deepseek_dsa_decode/_sparse_mla_kernel/repro/API-gemm-prefix-columns-developer.py`（`max_diff=0.00390625<1e-2`，exit 0） | tilelang 0.1.2+00aeb862 / Ascend910B3 / 2026-10-15 | `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1（`size` 说明补「src2 允许比 c 宽，取前缀列」）|
| 列谓词掩码（同一列所有行同判据）的可落地构造链：`T.copy`(行→(1,N)) → `T.vbrc`(标量 lim→(1,N)) → `T.vcmp`((1,N) 张量-张量) → `T.vselect`(→ (1,N) 惩罚值) → `T.vbrc`((1,N)→(M,N) **列广播**) → `T.vadd`(累加进 L0C 分数)。四步缺一即退化为前两条 TRAP；`-1e38` 哨兵使 `exp()` 精确为 0，归约自然忽略该列 | P | `_sparse_mla_kernel.py` 掩码段（`idx_row/lim_row/vis_row/zero_row/sent_row/pen_row/pen_mat`）；文档依据 `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2 例 2（`(1,N)→(M,N)`） | `python examples/deepseek_dsa_decode/_sparse_mla_kernel/repro/PATTERN-column-mask-vbrc-broadcast.py`（逐位相等断言，exit 0） | tilelang 0.1.2+00aeb862 / Ascend910B3 / 2026-10-15 | `pattern-library/attention.md`（掩码子模式）/ `.agents/skills/tilelang-vector-skill/SKILL.md` |
| **（负向实测）`block_heads=64` 备选不可编译**：BiShengIR 报 `ub overflow, requires 2115584 bits while 1572864 bits available`（UB 需 258KB > 192KB）；叠加 L0C 侧 `64*(512+64)*4=147456B > 131072B`（>128KB）。故 `DESIGN.md` §5.2/§1.6.3 的 BH=64 备选在 910B3 上**双向越限**，BH≤56 的约束是硬约束而非保守估计 | D | `--level all` 的 `[L2-5] WARN: ... ub overflow, requires 2115584 bits while 1572864 bits available`；`DESIGN.md` §5.2（`BH ≤ 56`）/§1.6.3（BH=64 阈值分支） | `python examples/deepseek_dsa_decode/_sparse_mla_kernel/_sparse_mla_kernel.py --level all`（L2-5 用例 `block_heads=64`） | tilelang 0.1.2+00aeb862 / Ascend910B3 / 2026-10-15 | `pattern-library/attention.md` / `constants.md`（`CONST-capacity-910B2C` 增补本条实测） |
| **（负向实测）非默认 `block_N=128` 不可编译**：同型 UB overflow（264KB > 192KB）。`block_N` 在本算子不是自由旋钮，64 是为 L1 双 stage + UB 余量共同选定的唯一实测可用值（`block_N=32` 可用，见 `[boundary-bn32]` PASS） | D | `_sparse_mla_kernel.py` run_boundary 注释（bn128 探针）；`[boundary-bn32] ... out=PASS` | `python examples/deepseek_dsa_decode/_sparse_mla_kernel/_sparse_mla_kernel.py --level all`（Boundary 段） | tilelang 0.1.2+00aeb862 / Ascend910B3 / 2026-10-15 | `pattern-library/attention.md` |
| **（集成风险，D 类）manifest 断言 `atol=3e-4 / rtol=1e-5` 对 fp16 输出在 O(0.5) 量级不可达**：实测 max_diff = 4.883e-04 = 2^-11 = **1 个 fp16 ulp @|o|≈0.5**（`L0-1/L0-4/L0-5/L1-1/L1-3/L1-5` 均为该值；bf16 为 2 ulp）。差异来源是 fp16 末次舍入 + 求和分组不同（golden 单次全 topk einsum vs kernel 32 块 online），**非算法语义偏差**（`L0-3` 主流 shape 输出量级小，strict 亦 PASS：2.441e-04）。`DESIGN.md` §8.2 R-7 的预测被实测坐实；Stage 5 集成前需就 harness 容差达成结论 | D | `--level all` 的 `strict(3e-4/1e-5)=note` 行 vs `[L0-3] strict=PASS`；`DESIGN.md` §8.2 R-7 / §9.1 R-7 | `python examples/deepseek_dsa_decode/_sparse_mla_kernel/_sparse_mla_kernel.py --level L0`（`_compare` 同时报 loose/strict） | tilelang 0.1.2+00aeb862 / Ascend910B3 / 2026-10-15 | `docs/开发指南.md`（容差章节）/ 集成 harness 说明 |
| Developer CV-split 为本 kernel 自动生成 **217344 B/block 的 workspace**（`main_infer_workspace_shape_function()`），`DESIGN.md` §4 内存规划未计入该 GM 往返；同时 `T.vcmp` 在 i32 上被降级为标量指令（编译期 warning "will execute by scalar instruction with low effiency"，命中 §9.1 R-5 的预判） | D | `--level all` 日志 `Workspace size: main_infer_workspace_shape_function() = 217344 bytes/block`；编译期 warning `loc(...:88:13): warning: Op 'hivm.hir.vcmp' will execute by scalar instruction with low effiency` | `python examples/deepseek_dsa_decode/_sparse_mla_kernel/_sparse_mla_kernel.py --level L0`（stderr 可见） | tilelang 0.1.2+00aeb862 / Ascend910B3 / 2026-10-15 | `pattern-library/attention.md`（Stage 4 线索）/ `constants.md` |

### Transferable Lessons（可迁移教训，本迁移任务后续函数实施者）

- **谓词掩码一律走 `T.vbrc` + `T.vcmp` + `T.vselect`，绝不用带 runtime 标量的 `T.if_then_else`，也不在 `T.Parallel` 里裸赋值 GM 行到 2D fragment**：这两种写法在 npuir Developer 下分别 mis-lower 成 i1-vcmp 与 `insert_slice` 尺寸错配，报错文本（MLIR verifier / BiShengIR pass 名）完全不提示「换个写法」，会白耗编译轮次。列谓词（同列同判据）用 `T.vbrc((1,N)→(M,N))` 广播，行谓词用 `(M,1)→(M,N)`。
- **任何涉及 block 索引派生标量的最小复现，探针都要用 `T.Kernel(≥2)`**：`T.Kernel(1)` 会把块索引常量折叠，runtime 标量路径根本不生成，探针会给出「编译通过」的假阴性（本轮 5 个隔离探针全中此坑，最终靠 `nblk=4` 才复现）。
- **迁移前先把「同族先例里没出现的那个 API 形态」单独探针实测，不要先按备选方案降级实现**：本轮 `T.gemm(size=[M,b,c])` 前缀列（R-9 未实测项）实测成立，直接省掉备选的 +89% gather 代价；反向亦然——BH=64/BN=128 两个「设计里的备选/旋钮」实测双向越限，若先按备选实现会白写一遍。
- **交付前必须把 harness 断言的容差与本设计的 blocking 容差分别实跑并同时披露**：fp16 输出在 O(1) 量级的 1 ulp（≈4.9e-4）会稳定超过 `atol=3e-4`，这是 dtype 表示精度而非实现缺陷；只报单一「PASS」会把集成期失败埋到下游。
- **`T.vbrc` 的广播能力要以文档 §2.2.2 为准并实测**：`(1,N)→(M,N)`、`(M,1)→(M,N)` 均可用（rank 相同 + 非 1 维必须相等）；标量→任意形状也可用。`(BM,1)→(BM,1)` 之类空广播会被拒。
