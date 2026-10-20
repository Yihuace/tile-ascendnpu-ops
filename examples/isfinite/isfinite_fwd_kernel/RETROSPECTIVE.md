# RETROSPECTIVE.md — isfinite_fwd_kernel

> 追加式日志：每次 Stage 执行追加一个章节，不覆盖历史。schema 见 `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`。

## Stage 1 (design) Retrospective — first_design（migration-harness）— 2026-10-14

### 结论

`IsfiniteFwdOp`（`torch.isfinite`，bool 输出）迁移设计完成。核心机制：**规范最短式 `isfinite(x) = (|x| < +inf)`**（有序 `vcmp(|x|, +inf, "lt")`）；因 `T.vabs` 的 DataType 表**不支持 bf16**（`docs/Tilelang.language/数学操作/T.vabs.md` §2.2.1 bf16 ×）而 bf16 为必需 workload，采用其**代数等价的免 vabs 双侧有序式** `(x > -inf) ∧ (x < +inf)`（`T.vcmp(x, ±inf, "gt"/"lt")` + fp16 域 `vmul` 取与合并），**不使用**标量 `T.isfinite`、**不做** fp32 宽化、**不使用** `vcmp "ne"`。等价性由 `verify_equiv.py` 机器验证（2 项 EQUIV_PASS + 判别段 A/B）。gate 1 一次通过。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Constraint-doc | **规范最短机制 `vcmp(|x|,+inf,"lt")` 依赖 `T.vabs`，而 `T.vabs` bf16 ×**：谓词/取绝对值类算子若需 `|x|` 且输入含 bf16，无文档化配方；本设计退到代数等价的免 vabs 双侧有序式 `(x>-inf)∧(x<+inf)` | DESIGN.md §0.6 R2、§1.6.0 R1 候选 1/2、§3.3；`docs/Tilelang.language/数学操作/T.vabs.md` §2.2.1（bf16 ×） | 在 `docs/Tilelang.language/数学操作/T.vabs.md` 补「bf16 无 vabs 时的等价替代配方」（双侧有序/NOT 组合），或标注能力缺口候选 | P |
| Info-source | **谓词族三式同源不同式，不可互写**：`isnan` 用自比较取反 `NOT(x<=x)`；`isinf` 用双定向 `eq` 取并 `(x==+inf)∨(x==-inf)`；`isfinite` 用双侧有序取与 `(x>-inf)∧(x<+inf)`（等价于 `|x|<+inf`）。三者共享「有序 `vcmp`、无 `unordered`」约束（CG-2026-0024），但结构不同 | DESIGN.md §0.6 R2、§1.6.0 R1、§1.6.1；`examples/TileOPs/tileops/kernels/elementwise/isnan/isnan_kernel/isnan_fwd_kernel_DESIGN.md`、`examples/TileOPs/tileops/kernels/elementwise/isinf/isinf_kernel/isinf_fwd_kernel_DESIGN.md` | 在 `pattern-library/elementwise.md` 的谓词族配方增补「isfinite 专用式」（双侧有序取与），与 isnan 自比较取反、isinf 定向 `eq` 取并并列 | P |
| Constraint-doc | **bool（i1）掩码的「与」合并无直接 API**：`vand`/`vnot` 的 DataType 表均不含 `bool/int1`，`vselect` 不含 bool 操作数，`vcast` 无 `i1→i8` 直连；「两 bool 掩码与合并 + 落字节」只能经 fp16 域 `vcast + vmul`（0/1 相乘 = 与） | DESIGN.md §0.6 R3、§3.2、§9.1；`docs/Tilelang.language/逻辑操作/T.vand.md` §2.2.1、`/逻辑操作/T.vnot.md` §2.2.1、`/条件操作/T.vselect.md` §2.2.1、`/数据类型转换操作/T.vcast.md` §2.2.1 | 在 `T.vcmp.md`（或逻辑/条件文档）补「bool 谓词输出的合并与落盘配方」：`vcmp → vcast(bool→f16) → vmul（与）/ vadd（或，互斥时）→ vcast(f16→i8)` + host `view(bool)` | P |
| Info-source | pattern-library `CONST-vcmp-ne-nan-unordered` 已标 `overturned`（`INDEX.md:116`），谓词族先例注释仍多以「ne 无序」为前提；新谓词算子调研须先读索引状态 | DESIGN.md §1.6.0/R4、§9.1；`.agents/evolution/capability-gaps.md` CG-2026-0024 | 无需新增（记录以强化先读索引状态 + 本工具链实跑复核的习惯） | C |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **isfinite 的等价化简式（精确机器验证）**：`T.isfinite(T.cast(x,"float32"))` → 原生 dtype 双侧有序取与 `(x>-inf)∧(x<+inf)`（等价于规范最短式 `|x| < +inf`）——(1) IEEE 有序比较下有限值两侧同真、±inf 一侧为假、NaN 两侧皆假；(2) 宽化不改变有限性故可省 fp32 中转；(3) `|x|<+inf` 与双侧式等价（判别段 A）；fp16/bf16/fp32 各 4115 元素（含 NaN/±inf/±0/次正规/极值/溢出敏感值角点）mismatches=0，两项 EQUIV_PASS；判别段 B 证明平方式 `(x*x)<+inf` 在大有限值处溢出误判（fp16 3 / bf16 3 / fp32 5 失配） | D | DESIGN.md §1.6.1（优化结论 + 结果表）、§1.6.0 R1 候选 2；`examples/isfinite/isfinite_fwd_kernel/verify_equiv.py` | `python examples/isfinite/isfinite_fwd_kernel/verify_equiv.py` | tilelang 0.1.2 dev-root（417dae99）+ Ascend NPU（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/elementwise.md（谓词族先例：isfinite 双侧有序取与） |
| **`T.vabs` bf16 缺口与 `|x|` 的免 vabs 替代**：`T.vabs` 文档标 bf16 ×；`isfinite` 等需 `|x|` 的逐元素谓词若输入含 bf16，不能直接用规范最短式 `vcmp(|x|,+inf,"lt")`；替代为代数等价的双侧有序式 `(x>-inf)∧(x<+inf)`（原生 dtype，全 dtype 统一，无需 dtype 分支） | P | DESIGN.md §0.6 R2、§1.6.0 R1 候选 1/2；`docs/Tilelang.language/数学操作/T.vabs.md` §2.2.1（bf16 ×） | none（机制判定；行为由 §8.1 golden `torch.isfinite` 精确比对覆盖） | tilelang 0.1.2 dev-root + Ascend NPU（实查 20 AI Core）+ CANN 8.5.2 | docs/Tilelang.language/数学操作/T.vabs.md（补 bf16 替代配方） |
| **`T.infinity` bf16 缺口与常量构造绕法**：`T.infinity("bfloat16")` 文档标 ×；bf16 无限常量由 `T.cast(T.infinity("float32"), "bfloat16")` 精确窄化构造（bf16 保留 fp32 8 位指数域，inf 窄化无损），`T.fill` 接受该 Cast 标量（先例 `T.fill(one_ub, T.cast(1.0,"float16"))`）；**未文档化假设**，Stage 3 首验，失败回退 fp32 宽化比较 | P | DESIGN.md §3.3/§3.4/§9.1；`docs/Tilelang.language/创建操作/T.infinity.md` §2.2.1（bf16 ×）；`examples/isnan/isnan_fwd_kernel/repro/CONST-vcmp-ne-ordered.py`（`T.fill` + Cast 先例） | none（设计假设；待 Stage 3 落地验证回填） | tilelang 0.1.2 dev-root + Ascend NPU + CANN 8.5.2 | docs/Tilelang.language/创建操作/T.infinity.md（补 bf16 构造） |

### Transferable Lessons（可迁移教训）

- **谓词族按「判定目标」再分式，不可整体照搬**：`isnan`（判自比较为假）用 `NOT(x<=x)`；`isinf`（判等值于 ±inf）用双定向有序 `eq` 取并 `(x==+inf)∨(x==-inf)`；`isfinite`（判幅值小于 +inf）用双侧有序取与 `(x>-inf)∧(x<+inf)`（等价于 `|x|<+inf`）。三者共享同一工具链约束（`vcmp` 为有序、无 `unordered`；CG-2026-0024），但**结构不同、不可互写**。
- **`T.vabs` 不支持 bf16**：任何想用 `|x|` 的逐元素 NPU 设计，在必需 bf16 时须改用代数等价、免 vabs 的形式（如 `|x| < +inf` → `(x>-inf)∧(x<+inf)`）；不要为单个 dtype 引入 dtype 分支或不必要的 fp32 宽化。
- **bool 掩码的合并/落盘统一走 fp16 域**：`vcmp` 掩码（bool，不可写 GM）→ `vcast`（`i1→f16`）→ fp16 域 `vmul`（与）或 `vadd`（或，互斥时）→ `vcast`（`f16→i8`）→ host `view(torch.bool)`；因 `vand/vnot` 不支持 `bool/int1`、`vselect` 不支持 bool 操作数、`vcast` 无 `i1→i8` 直连。
- **本工具链 `vcmp` 为有序**（`ne ≡ lt∨gt`，NaN 自比较恒 false，无 `unordered` 模式）：任何「靠无序语义」的谓词（`ne` 自比较、`!=` 判 NaN）在本后端都不成立；有限性/无穷/NaN 判定一律用定向有序比较（`lt`/`gt`/`eq`/`le` 及其显式组合）。

## Stage 3 (develop) Retrospective — first_impl（migration-harness）— 2026-10-14

### 结论

`isfinite_fwd_kernel`（`y = isfinite(x)`，bool 输出）NPU 实现完成，**Developer** 模式，一次通过。实现严格按 DESIGN §0.6 R2/R3：`T.vcmp(x_ub, neginf_ub, m_lo, "gt")` + `T.vcmp(x_ub, posinf_ub, m_hi, "lt")`（原生 dtype 有序）→ `vcast(i1→f16)` ×2 → `T.vmul(f_lo, f_hi, f_lo)`（0/1 相乘 = 逻辑与）→ `vcast(f16→i8)` → int8 字节 GM，host `view(torch.bool)`。两次编译均 `AscendNPU IR compile success`，`--level all` 全层 max_diff=0.00e+00（精确）。两处 Stage 1 标注的「未文档化假设」均实证成立：**bf16 常量 `T.cast(T.infinity("float32"), "bfloat16")` + `T.fill` 接受 Cast 标量**，且 fp32 路径 `block_size=8192` 实测正常编译运行（未触发 UB 溢出）。判定 `[PRECISION_PASS]`。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Constraint-doc | Stage 1 对 `block_size=8192`（fp32）过度保守：预测峰值 ~156KB、×1.7 悲观膨胀 265KB 超限，故 DESIGN §5.2 取 4096；但 harness 接口签名（与 GPU 抽取值/同族 isinf 一致）为默认 8192，实测 fp32 8192 八 buffer 结构编译运行均正常 | `isfinite_fwd_kernel.py:93`（默认 8192）、`--level L0` fp32 8192 PASS、L0 bf16 8192 PASS；对比 `examples/TileOPs/tileops/kernels/elementwise/isinf/isinf_kernel/isinf_fwd_kernel.py` 同 8-buffer 结构默认 8192 | 在 `constants.md` / `T.alloc_ub` 相关条目补「显式 `alloc_ub` 结构实测 ≈ 手工核算（非 ×1.7 auto-multibuffer 膨胀），UB 预算应按实际 alloc 结构核」——避免后续算子按悲观膨胀过度调小 block_size | P |
| Info-source | DESIGN §5.2 的 `block_size=4096` 与 harness 强制接口签名 `block_size=8192` 冲突；本实现按 task 强制签名取 8192（DESIGN 伪代码默认 4096 未照抄），功能与精度不受影响 | `isfinite_fwd_kernel.py:93`；DESIGN.md §3.3/§5.2 | 设计阶段 block_size 默认值应与 harness wrapper 接口签名对齐（K11 wrapper rule），避免「设计默认 ≠ 接口默认」 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **bf16 `±inf` 常量构造绕法实证**：`T.cast(T.infinity("float32"), "bfloat16")`（及取负）经 `T.fill` 填入 UB 后用于 `vcmp`，bf16 路径精确（`--level L0/L1` bf16 max_diff=0）；`T.infinity` bf16 × 缺口由此绕开，无需 fp32 宽化 | D | `isfinite_fwd_kernel.py:167-172`、bf16 L0/L1 PASS=0；DESIGN §3.3/§9.1 标「未文档化假设」 | `python examples/TileOPs/tileops/kernels/elementwise/isfinite/isfinite_kernel/isfinite_fwd_kernel.py --level L0`（bf16 用例） | tilelang 0.1.2 dev-root（417dae99）+ Ascend NPU（实查 20 AI Core）+ CANN 8.5.2 | docs/Tilelang.language/创建操作/T.infinity.md（补 bf16 由 fp32 inf 精确窄化的构造） |
| **fp16 域 `vmul` 做 bool 与合并的落盘链**：两 `vcmp` bool 掩码 → `vcast(bool→f16)`（0.0/1.0）→ `vmul`（0/1 相乘 = 逻辑与，精确）→ `vcast(f16→i8)` → int8 字节；三 dtype 全层 max_diff=0 | D | `isfinite_fwd_kernel.py:203-210`；`--level all` 全 PASS=0 | `python examples/TileOPs/tileops/kernels/elementwise/isfinite/isfinite_kernel/isfinite_fwd_kernel.py --level all` | tilelang 0.1.2 dev-root + Ascend NPU + CANN 8.5.2 | docs/Tilelang.language/比较操作/T.vcmp.md（补 bool 谓词合并落盘配方：与用 vmul、或用 vadd） |

### Transferable Lessons（可迁移教训）

- **bf16 无限常量**：本工具链 `T.infinity("bfloat16")` 不可用；bf16 `±inf` 由 `T.cast(T.infinity("float32"), "bfloat16")`（取负同理）构造，`T.fill` 接受该 Cast 标量——D 类实测成立，可复用于任何需 bf16 `±inf` 常量的逐元素谓词算子（isnan/isinf/isfinite 同族）。
- **bool 谓词「与」合并用 `vmul`、「或」合并用 `vadd`**：两掩码先各 `vcast(i1→f16)` 成 0.0/1.0，再在 fp16 域用 `vmul`（与）或 `vadd`（或；互斥掩码时和 ∈ {0,1} 精确）合并，最后 `vcast(f16→i8)` 落 0/1 字节 + host `view(torch.bool)`。
- **UB 预算按显式 alloc 结构核算**：显式 `T.alloc_ub` 结构的实测占用 ≈ 手工逐 buffer 求和；DESIGN 期不应按 auto-multibuffer 的 ×1.7 悲观膨胀过度调小 `block_size`（本算子 fp32 `block_size=8192` 8 buffer 实测编译运行正常，DESIGN §5.2 的 4096 属过度保守）。
