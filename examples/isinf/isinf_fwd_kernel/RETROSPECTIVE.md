# RETROSPECTIVE.md — isinf_fwd_kernel

> 追加式日志：每次 Stage 执行追加一个章节，不覆盖历史。schema 见 `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`。

## Stage 1 (design) Retrospective — first_design（migration-harness）— 2026-10-14

### 结论
`IsinfFwdOp`（`torch.isinf`，bool 输出）迁移设计完成。核心机制：**原生 dtype 双定向有序 `eq` 取并**——`isinf(x) = (x == +inf) ∨ (x == −inf)`（`T.vcmp(x, ±inf, mask, "eq")` + fp16 域 `vadd` 合并），**不使用**标量 `T.isinf`、**不做** fp32 宽化、**不使用** `vcmp "ne"`。等价性由 `verify_equiv.py` 机器验证（2 项 EQUIV_PASS + 判别段 A/B）。gate 1 一次通过。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | **谓词族内部机制不同构**：`isnan` 用「自比较取反」（`NOT(x<=x)`），`isinf` 用「双定向 `eq` 取并」（无自比较可判 ±inf）；若把 isnan 的「取反」套路照搬到 isinf 会得到无意义/更长的式（`NOT(isfinite)&NOT(isnan)` 需 vabs + 更 op）。两式都受同一工具链约束（有序 `vcmp`）但结构不同 | DESIGN.md §0.6 R2、§1.6.0 R1 候选表、§1.6.1；CG-2026-0024 | 在 pattern-library/elementwise.md 的谓词族配方增补「isinf 专用式」（定向 `eq` 取并），与「isnan 自比较取反」并列，避免互相仿写 | P |
| Constraint-doc | **`T.infinity` bf16 不支持**（`创建操作/T.infinity.md` §2.2.1 bf16 ×）；谓词/常量比较类算子若需 ±inf 常量且输入含 bf16，无文档化配方。本设计用 `T.cast(T.infinity("float32"), "bfloat16")` 精确窄化（**未文档化假设**，先例仅 `T.fill` 接受 Cast 标量） | DESIGN.md §3.3/§3.4/§9.1；`docs/Tilelang.language/创建操作/T.infinity.md` §2.2.1 | `T.infinity` 文档补「bf16（及窄精度）常量构造」示例或明确回退（fp32 常量 + 宽化比较）；标注为能力缺口候选 | P |
| Constraint-doc | **bool 掩码「或」合并无直接 API**：`vnot`/`vor`/`vmax` 的 DataType 表均不含 `bool/int1`，`vselect` 不含 bool 操作数，`vcast` 无 `i1→i8` 直连；「两 bool 掩码合并 + 落字节」只能经 fp16 域 `vcast + vadd`（或 vselect 于 f16） | DESIGN.md §0.6 R3、§3.2、§9.1；`docs/Tilelang.language/逻辑操作/T.vor.md` §2.2.1、`/比较操作/T.vmax.md` §2.2.1、`/逻辑操作/T.vnot.md` §2.2.1、`/条件操作/T.vselect.md` §2.2.1、`/数据类型转换操作/T.vcast.md` §2.2.1 | 在 `T.vcmp.md`（或逻辑/条件文档）补「bool 谓词输出的合并与落盘配方」：`vcmp → vcast(bool→f16) → vadd（互斥合并）→ vcast(f16→i8)` + host `view(bool)` | P |
| Info-source | pattern-library `CONST-vcmp-ne-nan-unordered` 已标 `overturned`（`INDEX.md:116`），但诸多 elementwise 先例的注释仍以「ne 无序」为前提表述；新谓词算子调研时须先读索引状态 | DESIGN.md §1.6.0/R4、§9.1；`.agents/evolution/capability-gaps.md` CG-2026-0024 | 升级后的模式库条目保留 overturned 标注即可（本任务无新增，记录以强化复验习惯） | C |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **isinf 的等价化简式（精确机器验证）**：`T.isinf(T.cast(x,"float32"))` → 原生 dtype 双定向有序 `eq` 取并 `(x==+inf)∨(x==−inf)`——(1) IEEE 有序 `eq` 对 NaN 恒假、仅对确切 ±inf 为真；(2) 宽化不改变 inf/NaN 性故可省 fp32 中转；fp16/bf16/fp32 各 4111 元素（含 NaN/±inf/±0/次正规/极值角点）mismatches=0，两项 EQUIV_PASS；判别段 A 证明减法变体 `(x−x)!=0` 在 NaN 处失配（=NaN 数）；判别段 B 证明 `NOT(isfinite)&NOT(isnan)` 等价的备选式 | D | DESIGN.md §1.6.1（优化结论 + 结果表）、§1.6.0 R1 候选 1；`verify_equiv.py` | `python examples/isinf/isinf_fwd_kernel/verify_equiv.py` | tilelang 0.1.2 dev-root（417dae99）+ Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/elementwise.md（谓词族先例：isinf 定向 `eq` 取并 vs isnan 自比较取反） |
| **bool 谓词多掩码合并/落盘配方**：`vcmp` 掩码（bool，不可写 GM）→ `vcast`（`i1→f16`）→ 在 fp16 域 `vadd` 合并（互斥掩码 → 0/1，精确）→ `vcast`（`f16→i8`）→ host `view(torch.bool)`；因 `vor/vmax/vnot` 不支持 `bool/int1`、`vselect` 不支持 bool 操作数、`vcast` 无 `i1→i8` 直连，fp16 中转是文档化可行路径 | P | DESIGN.md §0.6 R3、§3.2、§9.1；`docs/Tilelang.language/逻辑操作/T.vor.md` §2.2.1、`/比较操作/T.vmax.md` §2.2.1、`/逻辑操作/T.vnot.md` §2.2.1、`/条件操作/T.vselect.md` §2.2.1、`/数据类型转换操作/T.vcast.md` §2.2.1 | none（机制判定；行为由 §8.1 golden `torch.isinf` 精确比对覆盖） | tilelang 0.1.2 dev-root + Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.2 | docs/Tilelang.language/比较操作/T.vcmp.md（或逻辑文档） |
| **`T.infinity` bf16 缺口与常量构造绕法**：`T.infinity("bfloat16")` 文档标 ×；bf16 无限常量由 `T.cast(T.infinity("float32"), "bfloat16")` 精确窄化构造（bf16 保留 fp32 8 位指数域，inf 窄化无损），`T.fill` 接受该 Cast 标量（先例 `T.fill(one_ub, T.cast(1.0,"float16"))`）；**未文档化假设**，Stage 3 首验，失败回退 fp32 宽化比较 | P | DESIGN.md §3.3/§3.4/§9.1；`docs/Tilelang.language/创建操作/T.infinity.md` §2.2.1（bf16 ×）；`examples/isnan/isnan_fwd_kernel/repro/CONST-vcmp-ne-ordered.py`（`T.fill` + Cast 先例） | none（设计假设；待 Stage 3 落地验证回填） | tilelang 0.1.2 dev-root + Ascend 910B2C + CANN 8.5.2 | docs/Tilelang.language/创建操作/T.infinity.md（补 bf16 构造） |

### Transferable Lessons（可迁移教训）

- **谓词族内部按「判定目标」再分式，不可整体照搬**：`isnan`（判自比较为假）用 `NOT(x<=x)`；`isinf`（判等值于 ±inf，无自比较可判）用双定向有序 `eq` 取并 `(x==+inf)∨(x==−inf)`。两者共享同一约束（本工具链 `vcmp` 为有序、无 `unordered`），但**结构不同、不可互写**。`isfinite` 可复用 isinf 的双向 `eq` 之否定或 `|x|<+inf`（注意 vabs bf16 ×）。
- **本工具链 `vcmp` 为有序**（`ne ≡ lt∨gt`，NaN 自比较恒 false，无 `unordered` 模式；CG-2026-0024）：任何「靠无序语义」的谓词（`ne` 自比较、`!=` 判 NaN）在本后端都不成立；逆运算/取反必须显式构造（定向 `eq`、`le` 取反、`isfinite` 取反）。
- **bool 掩码不能直接做逻辑/选择/字节落盘**：`vor/vmax/vnot` 不含 `bool/int1`、`vselect` 不含 bool 操作数、`vcast` 无 `i1→i8` 直连——一律先 `vcast` 到 f16，在 fp16 域用算术（`vadd`/`vsub(1−m)`）或 `vselect`，再 `vcast` 到 i8；host `view(torch.bool)`。
- **常量类谓词先查 `T.infinity` 的 dtype 表**：bf16 ×（fp16/fp32 √）。bf16 常量可由 fp32 inf 精确窄化（bf16 指数域与 fp32 同宽）；此路径为未文档化假设，须在 Stage 3 首验。
- **减法「造 NaN」判 inf 是错误捷径**：`(x − x) != 0` 在 NaN 处也成立（`NaN != 0` 为真），把 NaN 误判为 inf；正确式只在 ±inf 处为真（定向 `eq`），设计期 `verify_equiv.py` 判别段可零成本拦截。
