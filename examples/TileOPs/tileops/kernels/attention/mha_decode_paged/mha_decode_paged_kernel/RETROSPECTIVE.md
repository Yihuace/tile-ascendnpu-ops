## Stage 1 (design) Retrospective — first_design — 2026-10-15

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | `tilelang-op-design` 模板 §5 Tiling 未把「分核策略三要素」列为子节标题，本设计自行新增 §5.5；`_shared/standards/core-split-strategy.md §1` 要求门禁机械核对存在性 | DESIGN.md#§5.5 | 在 design-template.md §5 增加固定子节 `### 5.5 分核策略（物理核数适配）` 骨架 | R |
| Info-source | MixCV 算子在 source 已是「Developer 平坦 TileLang（T.gemm+T.Parallel，无 Scope）」，但 skill 双轨（Developer/Expert）未给出「源为 Developer 形态的 attention 迁移」的 default 判定指引，需自行对照 `flash_attn_npuir_dev.py` 与 `TRAP-DEVMODE-PERSIST-GEMM` | DESIGN.md#§2、#§0.6-R1 | 在 tilelang-mixcv-skill 增「源为 Developer 平坦形态 → 优先非持久 Developer MixCV，除非 native CPU flags 触发」指引 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| decode MHA（S_q=1）两侧 GEMV 在 Cube 分形 M/N≥16 下必有一维 pad 16×；memory-bound 下选 Cube 而非纯 Vector 外层积（后者发射项 O(block_N)·D/8 op 爆炸） | P | DESIGN.md#§1.6.0-R3/R4、#§0.6-R2 | `none`（设计期结构论证） | tilelang dev root build 2026-10-15 / Ascend910_9362(aicore=20,aivector=40) / CANN 9.0.0 | algorithm-candidates.md ALG-gemv / ALG-attention |
| 等价性机器验证（D-1）：fold-scale-into-Q + 自然 exp 域 + 重缩放因子重结合三项在 fp16 mixed-precision 下违反率 0（max_abs 9.77e-4，随 OPT-1 单列 max_abs 9.77e-4 / OPT-2 精确 0 / OPT-3 3.05e-5） | D | DESIGN.md#§1.6.1 结果表；verify_equiv.py | `python examples/mha_decode_paged/_mha_decode_no_split_kernel/verify_equiv.py` | torch 2.9 CPU（无 NPU 依赖） | 待定 |
| npuir 核内 GM 标量索引（用运行时循环变量寻址 `block_table[bz,k]`/`real_seqlen_kv[bz]`）为未文档化能力——paged/动态 KV 类算子（decode/gather）的设计期不确定项，绕法 = UB 行缓存 + 局部标量读 | C | DESIGN.md#§9.1-R-2 | `none`（待 Stage 3 首验） | tilelang dev root build 2026-10-15 / Ascend910_9362 / CANN 9.0.0 | capability-gaps.md（待确证后登记）/ traps-runtime.md |

### Transferable Lessons（可迁移教训）

- **paged KV 装载必须用 slice 形态 `T.copy`**（`K[off*bn:(off+1)*bn, h, 0:dim] → l1[0:bn,0:dim]`），禁止 `load_nd2nz`/base+size 形态（对「行 stride=H·D、列 D 连续」的跨步区域静默平坦误读）——见 `TRAP-load-nd2nz-strided`。
- **decode（S_q=1）的 M 维退化**：Cube 两 GEMM 的分形 M/N≥16 使 query 维必被 pad 16×；用 block_M=16 + 行级隔离（per-row softmax/写回）+ Q_shared 清零处理，勿尝试用 head 维打包（各 head 的 K/V 不同，不可合并）。
- **动态 KV 长度下把循环界静态化**：用编译期 `ceil(N_kv_phys/page)` 作界 + OOB 掩码（有限大数哨兵，非 -inf），规避动态 trip-count 限制并保留 `T.Pipelined`；掩码必须用 `-1e38`/`vselect` 而非 `-inf`（`-inf-(-inf)=NaN`）。
- **Developer 模式 MixCV 不得引入 persistent 网格 + 核内 `T.serial` 多任务 + gemm**（`TRAP-DEVMODE-PERSIST-GEMM` 运行时崩溃）；非持久每核一任务规避。
- **`T.reduce(clear=False)` 在未初始化 buffer 上是静默错误形**——一律 `clear=True`。

## Stage 2 (review) Retrospective — first_review — 2026-10-15

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 维度 0/维度 5 检查项未覆盖「源**动态 trip count → 静态界 + 掩码**」这一迁移形态的专属陷阱：源因动态界永不迭代全掩码块，静态化后会**新引入**全掩码块，而设计常沿用 causal 场景的「哨兵 + `==-inf` 守卫」组合（二者不同域） | REVIEW.md#问题1；DESIGN.md#§0.6-R4/§6.4/§9.1-R-5 | 在 `tilelang-design-review` 维度 0 增检查项「动态界静态化后，源未出现的边界块（全掩码/空块）是否被新方案的守卫正确覆盖」，并在 `tilelang-op-design` migration-analysis.md 增同款避坑条目 | R |
| Review-flow | 维度 8 等价性的**机器验证臂有效性**无检查项——D-1 脚本可能出现「if/else 两分支同值」的同义反复臂仍标 EQUIV_PASS | REVIEW.md#问题4；verify_equiv.py#L76-81 | 在 `tilelang-design-review` 维度 8「机械复核」增一句：核对每个 D-1 臂的 baseline/candidate 是否确为不同表达式（否则该臂结论以人工推演替代） | R |
| Info-source | `PL-1.11-causal-mask-scalartrap` 的 `−1e38` 哨兵 pattern 为 **causal**（每行 ≥1 可见列）实测，未标注其「不适用于存在全掩码行的非 causal OOB 场景」，易被迁移设计跨场景套用 | REVIEW.md#问题1；pattern-library/attention.md#PL-1.11 bullet 126 | 在 attention.md PL-1.11 增一句适用边界：`−1e38` 哨兵 + 该守卫组合不覆盖全掩码行（行 max 会退化为哨兵，需额外 P/rs 置零） | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 动态 KV 界静态化（`ceildiv(N_kv_phys/page)`）会**新引入尾部全掩码块**；`−1e38` 哨兵 + `m==-inf` 守卫不同域 → 守卫失效（`m_cur=−1e38≠−inf`），`rs=exp(m_prev−m_cur)` 溢出/P 退化为 1；须让守卫与哨兵同域且同时对 rs/P/s_sum 置零，或对无有效列块直接置零 | P | REVIEW.md#问题1；DESIGN.md#§0.6-R4/§6.4/§9.1-R-5 | `none`（设计期静态推演；Stage 3 应以 real_kv<N_kv / real_kv=0 复现） | tilelang dev root build 2026-10-15 / Ascend910_9362 / CANN 9.0.0 | attention.md（mask/OOB 族）/ algorithm-candidates.md ALG-attention |
| online softmax 状态更新的两种等价书写（「先存 m_prev 再算 m_cur」vs「先算 m_cur 再存」）对**初始化缓冲**要求不同：前者须 init `m_cur`、后者须 init `m_prev`；伪代码若 init 与 copy 顺序错配，首迭代 `rs` 取未初始化值 | P | REVIEW.md#问题2；DESIGN.md#§3.3 L531/542 vs §1.6.1 L347 | `none`（设计期静态推演） | tilelang dev root build 2026-10-15 / Ascend910_9362 / CANN 9.0.0 | algorithm-candidates.md ALG-softmax / traps-runtime.md |
| 实测复核：`verify_equiv.py` D-1 重跑与 §1.6.1 内嵌表逐格一致（OPT-1 9.766e-4 / OPT-2 0.0 / OPT-3 3.052e-5 / adopted 9.766e-4，违反率 0），`design_calc_check.py` ub/l1/l0c/r3 全 pass、core_split skip（人工复核 num_logical≤20 实查成立） | D | REVIEW.md#机械复核 | `python3 examples/mha_decode_paged/_mha_decode_no_split_kernel/verify_equiv.py`；`python3 .agents/tools/design_calc_check.py --design <DESIGN.md>` | tilelang dev root build 2026-10-15 / torch 2.9 CPU / Ascend910_9362(aicore=20,aivector=40) / CANN 9.0.0 | 待定 |

### Transferable Lessons（可迁移教训）

- **动态 trip count 改静态界后，必须枚举「源从不迭代、静态界会新到达」的边界块**（全掩码块 / 空块），并确认守卫能覆盖——源正确 ≠ 静态化后仍正确。
- **掩码哨兵与稳定化守卫必须同域**：用有限大数哨兵（如 `−1e38`）时，若守卫仍写 `== -inf` 则永不触发；且行 max 会退化为哨兵，令 `P=exp(0)=1` 而非 0，必须同时对 rs/P/s_sum 设零。
- **online softmax 的「旧 max 保存」与「初始化」缓冲必须一致**：`copy(m_cur, m_prev)`-then-`reduce` 形态须 init `m_cur`；`reduce`-then-`copy` 形态须 init `m_prev`。init 错位会让首迭代读到未初始化值。
- **分区/掩码类 pattern（如 causal `−1e38` 哨兵）有其适用边界**（每行 ≥1 可见列），跨场景（OOB 全掩码行）套用前须核对边界是否仍成立。

## Stage 1 (design) Retrospective — revision 1 — 2026-10-15

> 触发：Stage 2 检视不通过（2 阻塞 + 3 建议）。本版从 REVIEW.md 与 v0 差异蒸馏「哪类设计判断被推翻、正确依据是什么」。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 迁移设计在「源码语义正确」的默认假设下推进，未把「动态界→静态界后**新到达**的边界块」当作独立设计对象；REVIEW 阻塞1 即此盲区 | REVIEW.md#问题1；DESIGN v0 §0.6-R4/§6.4 | `tilelang-op-design` references/migration-analysis.md 增「静态化改变迭代域 ⇒ 必须枚举新增块并逐个给出守卫」条目 | R |
| Design-flow | 掩码哨兵与稳定化守卫的**同域**约束未在 design-template 的 §6.4/边界语义处显式提醒；v0 出现 `-1e38` 哨兵 + `==-inf` 守卫的跨域组合 | REVIEW.md#问题1；DESIGN v0 §6.4 | design-template.md §6.4 骨架增一行「哨兵域 == 守卫域（同域判定）」 | R |
| Design-flow | 伪代码的 online 状态「初始化缓冲」与「copy 顺序」未绑定为不变量，v0 出现 init `m_prev` 却先 `copy(m_cur,m_prev)` 的错配 | REVIEW.md#问题2；DESIGN v0 §3.3 L531/542 | design-template.md §3.3 备注增「被读缓冲先 init；先算新 max 再存旧 max」示范 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 静态界新引入的**尾部全掩码块**的稳定化：`rs` 必须取 **1（恒等）** 而非 0——尾部块之后不再有有效块，`rs=0` 会把已累积 `logsum/acc_o` 清零（错误归零）；守卫须与哨兵**同域**（`all_masked = m_cur ≤ -1e30` vs 哨兵 `-1e38`），并对 `m_cur` 取安全值使 `P=exp(-1e38)=0` | P | REVIEW.md#问题1；DESIGN v1 §0.6-R4/§6.4；verify_equiv.py MASK-CHECK | `python examples/mha_decode_paged/_mha_decode_no_split_kernel/verify_equiv.py` | tilelang dev root build 2026-10-15 / torch CPU / Ascend910_9362(aicore=20,aivector=40) / CANN 9.0.0 | attention.md（mask/OOB 族）/ ALG-softmax |
| 机器实证：静态界+同域恒等守卫 vs 源动态界，在 `real_kv∈{0,3,8,9,17,24,32}` 上 max_abs=0.000e+00、无 NaN/inf、`real_kv=0` 输出 0；**反证** v0（`-1e38`+`==-inf`）在 `real_kv=3/8/17` 得 NaN、`real_kv=0` 得非零（4.087） | D | DESIGN v1 §1.6.1 注；verify_equiv.py `MASK-CHECK` / `_online_track` | `python examples/mha_decode_paged/_mha_decode_no_split_kernel/verify_equiv.py` | torch 2.9 CPU | 待定 |
| D-1 等价臂**有效性**判据：臂的 baseline/candidate 必须是不同表达式；同值 `if/else` 为同义反复。OPT-2 应在 **raw 域**对照 `exp2(m_p·s−m_c·s)` vs `exp2((m_p−m_c)·s)`（fp32 因子级 max_abs 2.384e-07 证非空；fp16 输出级归零） | D | REVIEW.md#问题4；verify_equiv.py `opt2_factor_check` | `python examples/mha_decode_paged/_mha_decode_no_split_kernel/verify_equiv.py` | torch 2.9 CPU | tilelang-design-review 维度 8 机械复核条目 |

### Transferable Lessons（可迁移教训）

- **静态化改变迭代域**：把源动态 trip 界换成编译期静态界时，必须枚举「源从不迭代、静态界会到达」的块（全掩码/空块），逐个确认守卫覆盖且与哨兵同域；否则「源正确」不代表新方案正确。
- **全掩码块的正确稳定化是恒等（`rs=1, P=0`）而非清零**：掩码块只在序列尾部（有效列是前缀），`rs=0` 会波及已累积状态；`rs=1` 才既零贡献又保状态。
- **online softmax 的 init/copy 顺序是不变量**：`m_prev` init `-inf` ⇒ 循环内先算 `m_cur`、末尾才 `copy(m_cur,m_prev)`；若写成先 copy 后算，则须 init `m_cur`。init 与 copy 顺序错配会读未初始化值。
- **D-1 臂需自检非空**：若两个分支逐字同值（如 `m_p*1.0 - m_c*1.0`），该臂为同义反复，须改造成真实对照（含因子级量化指标），否则机器验证对该臂失效。

## Stage 2 (review) Retrospective — re-review (design v1) — 2026-10-15

> 触发：设计修订 v1 后重检通过（0 阻塞 + 2 建议）。本版蒸馏「重检相较于首检，检视流程自身暴露了什么问题」。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 首检的**修改建议本身未被检视流程验证**：首检问题1 建议「同时置 `rs=0`」，但尾部全掩码块的 `rs=0` 会清零已累积 `logsum/acc_o`（错误归零）——修订版正确改用恒等 `rs=1` 并论证不采纳该措辞。reviewer 提出的修复方向须自身独立推演正确性，不能只要求「守卫与哨兵同域」而不核验修复后的语义 | REVIEW.md v0 #问题1 建议1；DESIGN v1 §0.0 未采纳项 / §0.6-R4 | 在 `tilelang-design-review` 维度 0/8 增一句：reviewer 给出的修改建议涉及状态更新取值时，须在报告内给出该取值的独立推演依据（而非仅指向「同域」）；重检时须复核「宣称不采纳的建议」其反驳是否成立 | R |
| Review-flow | 维度表未覆盖「伪代码**复用既有 buffer** 承接一个 src/dst 须同 shape 的算子」这一形态：`T.vcast(acc_o[16,dim], acc_s_cast[16,page_size])` 在 `dim≠page_size` 时违反 `T.vcast.md §2.2.2`，但 §3.3 骨架首检/重检的固定维度清单均未拦截 | DESIGN v1 §3.3 epilogue；`docs/Tilelang.language/数据类型转换操作/T.vcast.md §2.2.2`；对照 `flash_attn_npuir_dev.py:84` 的专用 `O_cast` | 在 `tilelang-design-review` 维度 1（API 可行性）增检查项「伪代码复用 buffer 时，一次复用是否满足该 op 文档的 src/dst shape 约束（cast/select 等逐元素 op 通常要求同 shape）」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 尾部全掩码块的稳定化取值：`rs` 必须为 **1（恒等）**，`rs=0` 会清零已累积 `logsum/acc_o`（因超界块恒为尾部、其后无有效块）；「同域守卫」只解决守卫触发问题、不决定取值，取值须单独推演 | P | DESIGN v1 §0.6-R4/§6.4；REVIEW v1 复核结论；verify_equiv.py `MASK-CHECK` | `python examples/mha_decode_paged/_mha_decode_no_split_kernel/verify_equiv.py` | tilelang dev root build 2026-10-15 / torch CPU / Ascend910_9362(aicore=20,aivector=40) / CANN 9.0.0 | attention.md（mask/OOB 族）/ algorithm-candidates.md ALG-softmax |
| 重检实测：DESIGN v1 的 §1.6.1 内嵌表与 `verify_equiv.py` 重跑逐格一致（OPT-1 9.766e-4 / OPT-2 0.0 / OPT-3 3.052e-5 / adopted 9.766e-4，违反率 0），`MASK-CHECK`（real_kv∈{0,3,8,9,17,24,32}）max_abs=0、无 NaN/inf、real_kv=0 输出 0；`design_calc_check.py` ub/l1/l0c/r3 全 pass、core_split skip（人工复核 num_logical≤20 实查成立） | D | REVIEW.md（v1）#机械复核 | `python3 examples/mha_decode_paged/_mha_decode_no_split_kernel/verify_equiv.py`；`python3 .agents/tools/design_calc_check.py --design <DESIGN.md>` | tilelang dev root build 2026-10-15 / torch 2.9 CPU / Ascend910_9362(aicore=20,aivector=40) / CANN 9.0.0 | 待定 |

### Transferable Lessons（可迁移教训）

- **reviewer 的修复建议也要被独立推演**：首检「哨兵/守卫必须同域」方向正确，但附带「对 rs 置零」的取值错误——同域只保证守卫触发，取值需按算子语义（尾部块、已累积状态）单独论证。重检须逐条复核「设计声称不采纳 REVIEW 建议」的反驳是否成立。
- **伪代码 buffer 复用须核 shape 约束**：`T.vcast`/`T.vselect` 等逐元素 op 文档要求 src/dst 同 shape；复用跨用途 buffer（如 `acc_s_cast` 兼作输出 cast 目的）在 `dim≠page_size` 的分档 workload 上会破shape，应显式分配专用缓冲。
- **重检仍须实跑机械复核 + 实查物理核数**：不得因「上一轮已核」而省略——DESIGN 修订可能同步改动内嵌表/脚本；本轮独立复跑 `NPUUtils.get().get_aicore_num()=20 / get_aivector_core_num()=40` 与 §5.5 一致。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-15

> 触发：首次实现（Developer MixCV）通过全部门禁（L0 5/5、L1 4/4、L2 5/5、Boundary 4/4）。本版蒸馏实现期从设备实测得到的 API 事实、调试手法与设计承接缺口。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | DESIGN 沿用源 GPU `pass_configs={TL_ENABLE_FAST_MATH: True}`，但 npuir 工具链的 `tilelang.PassConfigKey` **无该成员** → 首个 L0 直接 `AttributeError: TL_ENABLE_FAST_MATH`。迁移类设计继承源 config 时应核对 NPU 侧枚举存在性 | `_mha_decode_no_split_kernel.py` 初版；`dir(tilelang.PassConfigKey)`（69 项，无 FAST_MATH） | 在 `tilelang-op-design` migration-analysis.md / design-template §4.7 增「源 `pass_configs`/`compile_flags` 逐项核对 npuir `PassConfigKey` 是否存在，缺失项显式标注为无 NPU 语义并移除」 | R |
| Develop-flow | 设计期把「核内 GM 标量索引（`block_table[bz,k]`）」标为**未文档化能力**（R-2）并给退路，但未给出**首验探针**的最小形态；实现期需自行构造最小 probe 才能确认（结果是可用） | DESIGN §9.1-R-2 / §0.6-R3；本轮 `probe_paged.py` STAGE 1 | 在 `tilelang-op-develop` Phase 2 增「DESIGN 标注为未文档化/风险项的能力，先用 ≤30 行最小 probe 单独确认后再写入正式 kernel」 | R |
| Develop-flow | 崩溃面（compiler segfault）无内置隔离流程：`T.vcmp(tensor, runtime_scalar)` 令 npuir 前端 **SEGFAULT（无 stderr 诊断）**，只能靠二分 probe 定位；设计/检视文档的负向断言需附 `repro` 才能被后续任务快速复现 | 本轮 `/tmp/opencode/probe_mask.py` STAGE 34（segfault，exit 139）→ STAGE 35 变体 `bcast/pen/submant` 全通过 | 在 tilelang-debug-helper 增「compiler segfault 的 stage-wise 二分 probe 模板」（逐 op 累加 + 每个变量一个 stage id） | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `T.vcmp(A, scalar_runtime, cond, "<cmp>")`（A 为 tensor/fragment、scalar 为 **BufferLoad/循环变量表达式**）令 npuir 编译器 **SEGFAULT（exit 139，无诊断）**；绕法 = 先把 runtime scalar 经 `T.vbrc(lim, mat)` 广播成同形矩阵，再做**矩阵-矩阵** `T.vcmp`（bit-exact）。注意与既有「vcmp 标量化**性能**陷阱」不同：本条是**编译崩溃** | D | `examples/mha_decode_paged/_mha_decode_no_split_kernel/repro/TRAP-vcmp-scalar-operand.py`；本轮 probe 二分（STAGE 31/33 pass → STAGE 34 segfault；`bcast/pen/submant` 三变体 pass） | `python examples/mha_decode_paged/_mha_decode_no_split_kernel/repro/TRAP-vcmp-scalar-operand.py` | tilelang 0.1.2+180c5d20623b1f9171484e2f5bca0e489c978daf (dev root build) / torch 2.7.1 + torch_npu 2.7.1.post4 / Ascend910_9362 / CANN 9.0.0 | traps-compiler.md（TRAP-vcmp-scalar-operand） |
| **paged 动态页号装载可用（R-2 正向结论）**：`phys = block_table[bz,k]`（`k` 为 `T.Pipelined` 循环变量）作为 slice-form `T.copy` 的 GM 行偏移，在 Developer 模式 **`num_stages=1` 与 `2` 均编译+运行正确**（非恒等页表逐位一致）——DESIGN §9.1-R-2 的「未文档化能力 + 退路（bt_ub/num_stages=1）」**无需启用退路** | D | `examples/mha_decode_paged/_mha_decode_no_split_kernel/repro/TRAP-paged-dynamic-index-sliceload.py`；本轮 probe STAGE 1 + L0-5（shuffle 页表）PASS | `python examples/mha_decode_paged/_mha_decode_no_split_kernel/repro/TRAP-paged-dynamic-index-sliceload.py` | tilelang 0.1.2+180c5d20623b1f9171484e2f5bca0e489c978daf (dev root build) / Ascend910_9362 / CANN 9.0.0 | traps-runtime.md（TRAP-paged-dynamic-index-sliceload）/ attention.md（paged 寻址）/ capability-gaps.md（R-2 不登记为缺口） |
| 静态界 + 同域恒等守卫（`all_masked = m_cur ≤ -1e30` → `m_cur_safe=0 / rs=1 / P=exp(-1e38)=0 / s_sum=0`）**设备端实跑通过**：`real_kv=200`（page=256，块 0 部分掩码 + 块 1 全掩码）max_diff 1.28e-4 且输出非 0；`real_kv=0`（全块掩码）max_diff **0.000**、输出精确 0、无 NaN/inf；`real_kv=300`（末块部分掩码）1.26e-4 | D | `_mha_decode_no_split_kernel.py` L1-1/L1-2/L1-3；L0-1..5 max_diff 5.4e-5~1.2e-4 | `python examples/mha_decode_paged/_mha_decode_no_split_kernel/_mha_decode_no_split_kernel.py --level all` | tilelang 0.1.2+... (dev root build) / Ascend910_9362 / CANN 9.0.0 | attention.md（mask/OOB 族，Stage 1/2 VP 的设备端第二证） |
| REVIEW 建议 ①（epilogue `T.vcast` 复用 `acc_s_cast` 在 `dim≠page_size` 时破 shape）在实现中**必须**用专用 `O_cast [block_M, dim]`；同时 `T.vcast` 的 src/dst **形状必须一致**（本 kernel 三处：P 量化 `[16,page]`、输出 cast `[16,dim]`），复用跨用途 buffer 会在 manifest 的 D=64/page=256 档直接踩中 | D | `_mha_decode_no_split_kernel.py`（`O_cast` 专用缓冲）；`docs/Tilelang.language/数据类型转换操作/T.vcast.md §2.2.2` | `python examples/mha_decode_paged/_mha_decode_no_split_kernel/_mha_decode_no_split_kernel.py --level all`（L0-2/L0-3/L0-4 为 D=64/page=256 档） | tilelang 0.1.2+... (dev root build) / Ascend910_9362 / CANN 9.0.0 | traps-compiler.md / attention.md（Stage 2 VP 的 Stage 3 落地印证） |
| 调试手法：**aicore 异常 `error code 507015` + `When the D-cache reads and writes data to the UB, ... non-zero value`（UB 总线错）未必是 UB 越界**——本轮根因是测试 harness 把 **CPU 端 `block_table`** 传给了 NPU kernel（`torch.arange(...)` 未带 `device="npu"`），非法 GM 地址触发 MTE/D-cache 错。定位法：kernel 本体用独立 probe（同 shape/同输入）单独跑通后，差异只剩 harness ⇒ 先查输入张量 device/dtype/contiguity | P | 本轮 `probe_exact.py`（同 kernel 同 shape 设备端 RAN OK）vs `_run_case`（崩溃）；修 `_gen_inputs` 加 `device="npu"` 后 L0 5/5 | `none`（任务内定位，教训可直接迁移） | tilelang 0.1.2+... (dev root build) / Ascend910_9362 / CANN 9.0.0 | Tilelang算子调试指南.md / traps-runtime.md |

### Transferable Lessons（可迁移教训）

- **`T.vcmp` 的第二操作数若是「运行时标量」（BufferLoad / 循环变量表达式）会让 npuir 编译器直接段错误**：先 `T.vbrc(scalar, mat)` 广播为同形矩阵，再做矩阵-矩阵 `T.vcmp`。同形常量 buffer（如 `thr_vec`）或 Python 字面量无此问题（字面量在 trace 期被折叠）。
- **迁移时源侧 `pass_configs` 不能照抄**：`TL_ENABLE_FAST_MATH` 在 npuir `PassConfigKey` 中不存在，照搬即 `AttributeError`；迁移前用 `dir(tilelang.PassConfigKey)` 核对每项。
- **paged / 动态索引类算子的 GM 标量索引在 Developer 模式可用**：`phys = block_table[bz,k]` 作 slice-form `T.copy` 的 GM 行偏移，`T.Pipelined` `num_stages=1/2` 均正确；无需先上 UB 行缓存退路。仍必须用 **slice 形态** `T.copy`（与 `TRAP-load-nd2nz-strided` 一致）。
- **aicore `507015` / UB 总线错先怀疑输入在 CPU**：kernel 本体若能在独立 probe 中同 shape 跑通，则崩溃面在 harness——检查所有输入张量的 `device="npu"`、contiguity 与 dtype（本轮 CPU `block_table` 即此型）。
- **静态界 + 同域恒等守卫已获设备端第二证**：`real_kv=0` 输出精确 0、`real_kv<page` 尾整块全掩码无污染；`rs` 取恒等 1（非 0）是尾部块唯一正确取值（Stage 1/2 VP 的设计推演在 Stage 3 实跑成立）。
- **`T.vcast` src/dst 形状必须一致**：epilogue 输出 cast 与 P 量化 cast 的目标形状不同时（`[16,dim]` vs `[16,page]`），不能复用同一 buffer；用专用 `O_cast`。

