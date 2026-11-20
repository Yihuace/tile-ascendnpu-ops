# RETROSPECTIVE.md — isnan_fwd_kernel

> 追加式日志：每次 Stage 执行追加一个章节，不覆盖历史。schema 见 `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`。

## Stage 1 (design) Retrospective — first_design（migration-harness）— 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | **float 谓词族**（isnan/isinf/isfinite）与 **float 算术激活族**（hardsigmoid/hardswish/mish）迁移套路不同：谓词族输出是**精确 bool**、**无浮点算术、无 fp32 宽化中转、无浮点容差**；若照抄激活族的「fp32 中转 + 浮点容差」模板会给谓词强加无意义的 `vcast` 宽化与容差 | DESIGN.md §0.1（谓词无宽化、输出 bool）、§1.6.1（谓词精确等价）、§11.3（float dtype 契约）、§8.2（atol=rtol=0） | migration-analysis.md / algorithm-research.md 的 elementwise 分族增补「谓词族」：直接原生 dtype 比较 + 字节 bool 输出，**不复用**算术族的 fp32 中转模式 | P |
| Constraint-doc | **bool 谓词输出的字节落盘**在 NPU 上需绕行：`vcmp` 掩码 dtype 为 bool 且不可写回 GM；`vcast` 表**无 `i1→i8` 直连**（仅 `i1→f16/f32`），`vselect` 表**不支持 int8/uint8 操作数**——故「掩码→0/1 字节」只能走 `vcast` 两步（`i1→f16→i8`）或 `vselect(f16)+vcast`。文档未给出这条组合配方 | DESIGN.md §0.6 R3、§3.2 步骤 2/3、§9.1；`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1；`docs/Tilelang.language/条件操作/T.vselect.md` §2.2.1 | 在 `docs/Tilelang.language/比较操作/T.vcmp.md` 或 vcast 文档补「bool 谓词输出落盘配方」示例：`vcmp → vcast(bool→f16) → vcast(f16→i8)` + host `view(bool)` | P |
| Tooling | `constants.md` `CONST-aicore-910B2C` 记 24 核，本环境 `NPUUtils.get().get_aicore_num()` 实查返回 20（hardsigmoid/hardswish/bitwise_not/isnan 四任务复现）；纯 Vector ×2 = 40 | DESIGN.md §5.5（实查 20 ×2 = 40） | `CONST-aicore-910B2C` 增补「本工作区设备实查 20（四任务复现）」稳定注释或拆分设备档位 | C |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| NaN 谓词的等价化简（精确机器验证）：`T.isnan(T.cast(x,"float32"))` → 原生 dtype 自比较 `vcmp(x,x,"ne")`——(1) `isnan(x) ⟺ (x != x)`（IEEE 唯一不等自身），(2) 宽化不改变 NaN 性故可省 fp32 中转；fp16/bf16/fp32 各 4097+ 元素（含 NaN/±inf/±0/次正规/极值角点）mismatches=0，三项 EQUIV_PASS | D | DESIGN.md §1.6.1（优化结论 + 结果表）；`verify_equiv.py` | `python examples/isnan/isnan_fwd_kernel/verify_equiv.py` | tilelang 0.1.2 dev-root（/home/developer/tilelang-mlir-ascend/build）+ Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/elementwise.md（谓词族先例，与 fp32 中转算术族并列） |
| bool 谓词输出的字节落盘配方：`vcmp` 掩码（bool，不可写 GM）→ `vcast`（`i1→f16`）→ `vcast`（`f16→i8`）→ host `view(torch.bool)`；因 `vcast` 无 `i1→i8` 直连、`vselect` 不支持 int8/uint8 操作数，两步 cast 是文档化可行路径；与 bitwise_not 的 int8 字节存储（CG-2026-0019）同源 | P | DESIGN.md §0.6 R3、§3.2、§9.1；`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1；`docs/Tilelang.language/条件操作/T.vselect.md` §2.2.1 | none（机制判定；行为由 §8.1 golden `torch.isnan` 精确比对覆盖） | tilelang 0.1.2 dev-root + Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.2 | docs/Tilelang.language/比较操作/T.vcmp.md（或 vcast 文档） |
| float 谓词族迁移**无需** fp32 宽化中转与浮点容差：谓词 = 原生 dtype 比较，输出精确 bool（atol=rtol=0）——与 float 算术激活族（hardsigmoid/hardswish 的 fp32 中转）是关键分族依据；`isnan/isinf/isfinite` 同族同构 | C | DESIGN.md §0.1/§1.6.1/§11.3；`docs/Tilelang.language/比较操作/T.vcmp.md` §2.2.1 | none（设计判定；行为由 §8.1 golden 覆盖） | tilelang 0.1.2 dev-root + Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/elementwise.md |

### Transferable Lessons（可迁移教训）

- 迁移 elementwise 前先按 **dtype 域 / 输出类型分族**：float 算术激活族（fp16/bf16/fp32 → fp32 中转 + NaN 链 + 浮点容差）；float 谓词族（`isnan/isinf/isfinite` → **原生 dtype 比较 + 精确 bool 输出、无宽化、无浮点容差**）；整数/布尔族（原生位运算 + 字节存储）。三类模板不可互相仿写。
- NPU 上 **NaN 检测用 `vcmp(x,x,"ne")` 自比较**（IEEE 无序：NaN≠NaN→true、±inf==±inf→false），**不要**用标量 `T.isnan`（无向量掩码形式），也不要用 `x-x != 0`（±inf 会被误判为 NaN）。自比较错误率 0（`verify_equiv.py` 三 dtype 实证，见 pattern-library `CONST-vcmp-ne-nan-unordered`）。
- `vcmp` 的 bool 掩码**不可写回 GM**；把它转成 0/1 字节输出必须经 `vcast`：`i1→f16`（得 1.0/0.0）再 `f16→i8`（得 1/0）。`vcast` 无 `i1→i8` 直连、`vselect` 不支持 int8/uint8 操作数——不要试图一步 select/cast 出 int8。
- bool 输出（谓词类算子）以 **int8 字节存储**承载并 host `view(torch.bool)`（bool i1 不可 load/store GM；CG-2026-0019：npuir auto output dtype 映射缺 uint8/bool 键）。这与 bitwise_not 的字节存储策略同源。
- 谓词族的分核持久化结构（`T.Kernel(num_kernels, is_npu=True)` + `T.serial(num_local_tasks)` + `T.min` 尾块 guard + `NPUUtils.get().get_aicore_num()*2`）与算术族完全同构，可直接复用；区别只在 compute（`vcmp`+`vcast` 而非 fp32 中转链）。

## Stage 3 (develop) Retrospective — first_impl（migration-harness）— 2026-10-14

### 结论
`[DESIGN_ERROR]`：冻结设计 §0.6 R2 / §9.1 的核心前提（Ascend `vcmp(x,x,"ne")` 自比较 = IEEE **无序**、NaN≠NaN→true）在本工具链上**不成立**。实测 `"ne"` 为**有序**不等（`ne ≡ lt∨gt`），NaN 自比较 = false，故设计指定实现无法检出 NaN（L0 在 NaN 注入位精确失配 12.5%）。修正需改 R2 算法（换比较模式 + 取反），属 Stage 1 设计修订，不在 Stage 3 范围。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | pattern-library `CONST-vcmp-ne-nan-unordered`（INDEX.md:116，status=**verified**，2026-10-12 CANN 8.5.0 首验）在本工具链（tilelang 0.1.2 dev-root commit 417dae99 + 910B2C + CANN 8.5.2）已被**推翻**：`vcmp "ne"` 自比较 NaN→false（有序），非无序 | `repro/CONST-vcmp-ne-ordered.py`（直接读出 mask：ne(NaN)=0、eq/le/ge(NaN)=0、le/ge(非NaN)=1，三 dtype）；DESIGN.md §0.6 R2/§9.1 | 工具链版本升级后必须 `kb_stale_check.py` 复验并**刷新该条目版本戳/状态为 overturned**；该条目是设计 R2 的唯一依据，陈旧即导致整个 isnan 设计失败 | D |
| Info-source | 旧 repro `CONST-vcmp-ne-nan-unordered.py` 在本工具链仍**通过**，但属**非判别性**假通过：其 hardsigmoid 链（vadd→vclamp→vmul）本身传播 NaN，输出 NaN 与 vcmp mask 是否命中无关；断言未隔离 vcmp 输出 | `python .agents/.../repro/CONST-vcmp-ne-nan-unordered.py` → EXIT=0（假通过）；对照直接 mask-readout 探针 `repro/CONST-vcmp-ne-ordered.py` | repro 规范（ED-B）增补「孤立被测原语」红线：验证某 API 语义时，输出必须**只**经该 API 的返回值决定，禁止经由会独立传播同一现象的算术链，否则断言不可判别 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **Ascend `vcmp "ne"` 是有序不等（非 IEEE 无序）**：`ne(x,x)` 对 NaN 返回 false（`ne ≡ lt∨gt`），**不可**用作 NaN 自比较检测；旧常数 `CONST-vcmp-ne-nan-unordered` 在本工具链被推翻（tilelang 0.1.2 commit 417dae99 + CANN 8.5.2）；同一工具链下旧 repro 仍假通过（非判别性） | D | `repro/CONST-vcmp-ne-ordered.py`（三 dtype 直接 mask readout：ne(NaN)=0）；`isnan_fwd_kernel.py --level all` L0 失配 12.5%（恰为 NaN 注入位） | `python examples/isnan/isnan_fwd_kernel/repro/CONST-vcmp-ne-ordered.py` | tilelang 0.1.2 dev-root（commit 417dae9908ad2ea5ffb74294a7075b6cb64b4f2b, /home/developer/tilelang-mlir-ascend/build）+ Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/INDEX.md（`CONST-vcmp-ne-nan-unordered` → overturned）+ traps-runtime.md |
| **Ascend NaN 自比较检测可用 `NOT (x <= x)`**：`vcmp(x,x,"le")`（或 `"ge"`）自比较对**所有非 NaN**（含 ±inf/±0/次正规）为 true、仅 NaN 为 false，取反即 isnan；配方 `vcmp "le" → vcast(i1→f16) → vsub(1.0, ·) → vcast(f16→i8)` + host `view(bool)`，fp32/fp16/bf16 精确（atol=rtol=0） | P | `repro/CONST-vcmp-ne-ordered.py`（`isnan_via_le` 三 dtype 与 `torch.isnan` 全等） | `python examples/isnan/isnan_fwd_kernel/repro/CONST-vcmp-ne-ordered.py` | tilelang 0.1.2 dev-root + Ascend 910B2C + CANN 8.5.2 | pattern-library/elementwise.md（谓词族 NaN 检测配方）+ docs/Tilelang.language/比较操作/T.vcmp.md |

### Transferable Lessons（可迁移教训）

- **不要依据 pattern-library 常数直接冻结依赖其语义的算法**：`CONST-vcmp-ne-nan-unordered` 标 verified 但版本戳绑 CANN 8.5.0；本工具链（CANN 8.5.2）行为相反。设计引用 `vcmp` 语义类常数时，Stage 1 必须按当前工具链**实跑**复核，不得只在文档层引用。
- **NaN 检测在 Ascend 上用 `vcmp(x,x,"le")`（或 `"ge"`）自比较取反，而非 `"ne"`**：本工具链 `"ne"` = 有序不等（NaN→false，不可用）；`"le"/"ge"` 自比较对非 NaN 恒 true、仅 NaN false，取反即精确 isnan（三 dtype 实证）。
- **repro 必须孤立被测原语（判别性）**：旧 vcmp repro 因测试链自身传播 NaN 而假通过；验证某 API 语义时输出须**只**经该 API 返回值决定，禁止经由会独立复现同一现象的算子链。
- **工具链升级后先跑 `kb_stale_check.py` / repro 复验再复用旧常数**：本条 D 类发现即一次工具链漂移导致的常数推翻，若不经实跑复核将直接污染下游设计。

## Stage 1 (design) Retrospective — revision v1（migration-harness，revision 触发=Stage 3 `[DESIGN_ERROR]`）— 2026-10-14

### 结论
v0 已冻结的 §0.6 R2 / §9.1 前提（Ascend `vcmp(x,x,"ne")` 自比较 = IEEE **无序**）被本工具链实测推翻（`ne` 为**有序**不等，NaN 自比较=false，检不出 NaN）。修订：NaN 谓词改为 **`isnan(x) = NOT (x <= x)`**（`vcmp(x,x,"le")` 有序自比较 + fp16 域 `vsub(1.0, ·)` 取反），§0.6 R2/R3、§1.6.0、§1.6.1、§1.6.2、§3.1–§3.4、§4.3–§4.5、§8.3、§9.1–§9.3、§11 同步更新，并新增 §0.0「相对上一版的关键调整 + 为何不会再犯同一错误」。`verify_equiv.py` 重跑并新增「有序 `ne` ≠ isnan / `le`-取反 == isnan」判别段。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | pattern-library `CONST-vcmp-ne-nan-unordered`（INDEX.md:116，`status: verified`，版本戳 CANN 8.5.0）在本工具链（CANN 8.5.2）被推翻：`vcmp "ne"` 自比较 NaN→false（有序）。设计曾**唯一**依赖该条目，陈旧即导致整个 isnan 设计失败 | DESIGN.md §0.0/§9.1；`repro/CONST-vcmp-ne-ordered.py` | 升级工具链后强制 `kb_stale_check.py` 复验并刷新该条目状态/版本戳为 overturned（由 conductor/evolver 归档，设计方不直接改 pattern-library） | D |
| Design-flow | revision 模式的「为何不会再犯同一错误」需制度化：比较语义类结论**不得只引文档/常数**，须对每种目标 dtype 用**孤立原语**探针实跑确认（NaN/±inf/±0/次正规边界） | DESIGN.md §0.0（三条防复发）+ §1.6.1 判别段 | `tilelang-op-design` / `algorithm-research.md` 增补「后端比较/谓词语义类结论：设计冻结前实跑判别性探针」红线 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **NaN 谓词等价性验证须含「比较模式」判别段**：不仅验证 `isnan(x) ⟺ NOT(x<=x)` 数学恒等，还要断言「有序 `ne` 自比较 ≠ isnan」（NaN 处失配>0）与「`le`-取反 == isnan」（失配=0），使「比较模式选错」在设计期被机器拦截（v0 正是漏了此判别，拖到 Stage 3 才暴露） | P | DESIGN.md §1.6.1（判别段结果表）+ §0.0；`verify_equiv.py`（`_discrimination`） | `python examples/isnan/isnan_fwd_kernel/verify_equiv.py`（三 dtype：ordered-ne 失配=NaN 数、le-取反失配=0） | tilelang 0.1.2 dev-root + Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.2 | tilelang-op-design Phase 2 第 5 步（等价性机器验证）——增补「语义敏感比较类算子的判别段要求」 |
| **`vcmp "le"` 自比较取反 = 精确 isnan 且可落字节**：配方 `vcmp "le" → vcast(i1→f16) → vsub(1.0, ·) → vcast(f16→i8)` + host `view(bool)`，fp16/bf16/fp32 全精确；是 v0 `ne` 路线失败后的替代主选（同一常数推翻的正确 fallback） | D | DESIGN.md §0.6 R2/R3、§3.2、§4.3–§4.5；`repro/CONST-vcmp-ne-ordered.py`（`isnan_via_le` 三 dtype 全等） | `python examples/isnan/isnan_fwd_kernel/repro/CONST-vcmp-ne-ordered.py` | tilelang 0.1.2 dev-root + Ascend 910B2C + CANN 8.5.2 | pattern-library/elementwise.md（谓词族 NaN 检测配方）+ docs/Tilelang.language/比较操作/T.vcmp.md |

### Transferable Lessons（可迁移教训）

- revision 模式的核心价值：**被推翻的是设计判断的依据链，不是单行代码**——本次改的是「比较语义来源」（改用本工具链实跑，而非 pattern-library 常数）+「谓词算法」（`ne`→`le` 取反），§0.0 必须写清「为何不会再犯」而不只是「改了什么」。
- **等价性机器验证要区分「数学恒等」与「后端下降语义」**：`isnan(x) ⟺ (x!=x)` 在 IEEE 数学上成立，但在**有序** `ne` 下降的后端上不成立；`isnan(x) ⟺ NOT(x<=x)` 对有序/无序 `<=`（有序比较）都成立于非 NaN 常真。设计期加判别段可零成本拦截此类后端语义分歧。
- **取反不能想当然**：`vnot` 不支持 `bool/int1`（`T.vnot.md` §2.2.1）、`vselect` 不支持 int8/uint8（`T.vselect.md` §2.2.1）——掩码取反须在 fp16 域用 `vsub(1.0, ·)`（0.0/1.0 精确），或先转 f16 再取反；**不要**假定存在向量 `not` 直通字节。
- 源 GPU 的 `NeFwdKernel` docstring（CUDA `__hne` 有序、`!=` 无序）是**源后端**语义，**不可外推**到 Ascend；跨后端迁移时比较语义一律重新实测。

## Stage 3 (develop) Retrospective — first_impl（migration-harness，revision v1 重实现）— 2026-10-14

### 结论
`[PRECISION_PASS]`：按 DESIGN v1 的 `isnan(x) = NOT (x <= x)` 方案重写 `isnan_fwd_kernel.py`（`T.vcmp(x_ub,x_ub,mask_ub,"le")` → `vcast(i1→f16)` → `T.vsub(1.0, m)` → `vcast(f16→i8)`），全程无 `vcmp "ne"`。`python isnan_fwd_kernel.py --level all` exit 0，31 项全过（L0 3 dtype + 3 负向、L1 13、L2 9、Boundary 3），三 dtype 精确 `max_diff=0.00e+00`，含判别用例 all-NaN→all-True、±inf→all-False。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 谓词类算子缺少强制的「错误比较模式判别用例」模板：v0 的 `vcmp "ne"` 错误仅在 Stage 3 因 L0 NaN 位失配暴露；本次 v1 能一次通过，直接得益于 DESIGN §8.3 已内置 all-NaN→all-True / ±inf→all-False 判别对 | `isnan_fwd_kernel.py` `run_boundary()`（all-NaN / ±inf 断言）；对比 v0 `[DESIGN_ERROR]`（L0 NaN 注入位失配 12.5%） | `tilelang-op-develop` skill 的测试模板对「谓词/比较类算子」增补必填判别用例（用错误比较模式必然失败的最小断言），而非仅依赖随机特殊值抽样 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **isnan 字节落盘配方端到端实测通过**：`vcmp("le") → vcast(i1→f16) → vsub(1.0,·) → vcast(f16→i8)` + host `view(torch.bool)`，fp16/bf16/fp32 精确（`max_diff=0`），含 all-NaN/all-±inf 判别；证实 v1 设计 R2/R3 在真实 NPU 可落地 | D | `isnan_fwd_kernel.py`（kernel + `--level all` 输出）；DESIGN.md §0.6 R2/R3 | `cd examples/isnan/isnan_fwd_kernel && python isnan_fwd_kernel.py --level all` | tilelang 0.1.2 dev-root（commit 417dae9908ad2ea5ffb74294a7075b6cb64b4f2b）+ Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/elementwise.md（谓词族 NaN 检测配方）+ docs/Tilelang.language/比较操作/T.vcmp.md |

### Transferable Lessons（可迁移教训）

- 谓词族（`isnan/isinf/isfinite`）在 Ascend 上落 bool 的通用配方：`T.vcmp` 有序自比较取反（`isnan=NOT(x<=x)` / `isinf` 可用 `le`/`ge` 对 ±inf 的等价式）→ `vcast(i1→f16)` → `T.vsub(1.0,·)` 取反 → `vcast(f16→i8)` → host `view(torch.bool)`；本工具链实测三 dtype 精确（0.0/1.0 在 fp16 精确可表示）。
- 谓词类算子 L0/Boundary **必须**放「用错误比较模式必然失败」的判别用例（如 all-NaN→all-True、±inf→all-False）——随机特殊值抽样会因错误模式在非 NaN 位恰好也返回 0 而假通过。
- 同一 `vcmp` 语义在本工具链已实测为有序：`"ne"` 不可用于 NaN 自比较；新谓词实现一律用 `le`/`ge`/`eq` 自比较取反，并在 L0 覆盖 all-NaN 与 all-±inf 两组反向用例。
