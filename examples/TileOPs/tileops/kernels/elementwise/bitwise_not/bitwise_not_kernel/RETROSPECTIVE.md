# bitwise_not_fwd_kernel — Stage 复盘（RETROSPECTIVE）

## Stage 1 (design) Retrospective — first_design — 2026-10-12

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 任务预注入的 stale 清单（`PL-1.6-copy-floor`/`PL-1.10`/`PL-1.14`/`PL-1.17`）与 `constants.md` 的 `CONST-mte2-degradation` 同源同数据、却分属「stale 禁用」与「可用常数」两栏，设计期需自行判断引用边界 | DESIGN.md#1.6.0 设计期估算行；任务 prompt「必读共享标准」与「知识预注入」两节 | 在 `pattern-library/INDEX.md` 或 `constants.md` 顶部标注 `CONST-mte2-degradation` 与 `PL-1.6-copy-floor` 的同源关系与当前可引用状态（stale 传播关系） | R |
| Constraint-doc | `T.vnot.md` §2.2.1 标 uint8「√」，但设计期实测：kernel 计算值正确、`@tilelang.jit(out_idx=[1])` 自动分配的输出 dtype 因 `jit_npu._tvm_dtype_to_torch` 缺 uint8 而变 float32——文档未覆盖 out_idx 分配路径的 dtype 支持 | DESIGN.md#9.1 R-2；capability-gaps CG-2026-0014 | 在 `docs/Tilelang.language/内存操作/T.copy.md` 或 jit 文档补「out_idx 自动分配支持的 dtype 集」并显式标注 uint8/uint16/uint32 缺口 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| npuir `T.vnot` 原生位反替代「掩码 XOR」：单指令、少一 buffer；`hivm.hir.vxor` 标量 src1 编译失败 | D | DESIGN.md#1.6.1 #1/#2 + #3.4；探针 history_version/design_probe_vnot_int.py | `python3 -c "..."`（见设计期探针 §9，int8/16/32/64 vnot == torch.bitwise_not bit-exact） | tilelang 0.1.2 dev root build 2026-10-12 / Ascend910_9362 / CANN 9.0.0 | pattern-library/elementwise.md（或 constants.md） |
| npuir bool（i1）无法 load/store GM：以 1 字节 `int8` 存储视图 + `T.fill(1)`+`T.vxor(.,1)` 实现 bool 逻辑非（bit-exact）；直接对 bool dtype 用 v-op 触发 device `vector core timeout` | D | DESIGN.md#0.6 R3 + #9.1 R-1；探针 history_version/design_probe_bool_storage.py | `T.fill(ones_ub, T.cast(1,"int8")); T.vxor(x_ub, ones_ub, y_ub)` on `x.view(int8)` | tilelang 0.1.2 dev root build 2026-10-12 / Ascend910_9362 / CANN 9.0.0 | pattern-library/traps-runtime.md（新 trap 条） |
| `@tilelang.jit(out_idx=[1])` 对 uint8 输出自动分配为 float32（`jit_npu.py:_tvm_dtype_to_torch` 缺 uint8/uint16/uint32）；kernel 计算值本身正确 | D | capability-gaps CG-2026-0014；探针 history_version/design_probe_uint8_outidx.py | `kern_vnot("uint8")(x.npu()).dtype` 返回 torch.float32；`y.view(torch.uint8)[:N]` == ref | tilelang 0.1.2 dev root build 2026-10-12 / Ascend910_9362 / CANN 9.0.0 | capability-gaps.md（已记 CG-2026-0014） |

### Transferable Lessons（可迁移教训）

- npuir 逐元素位运算优先用原生 `T.vnot`，不要照搬 GPU「掩码 XOR」——`hivm.hir.vxor` 的第二操作数必须是 tensor（传 `T.cast(1,dtype)` 标量会编译失败），走掩码路径需额外 `T.fill` materialize 一个掩码 buffer；`T.vnot` 一步到位。
- npuir 的 bool（i1）类型不能作为 GM 张量的 dtype 做 v-op：任何 bool 逐元素算子都应以「1 字节存储视图（int8/uint8）+ 位/算术等效公式」实现，并在 wrapper 层对 bool 张量做 `.view()`；对 bool dtype 直接建 UB buffer 运行会触发 device vector-core timeout（不是编译错误，定位成本高）。
- `T.fill(buf, value)` 的 value 必须是 PrimExpr（`T.cast(1, "int8")`），传 Python int 报 `value must be a tir.PrimExpr`；`T.vbrc(scalar, buf)` 在 1D/2D 上均报 rank mismatch（scalar rank 0 vs buffer rank≥1）——常量物化用 `T.fill`。
- `@tilelang.jit(out_idx=[...])` 的输出 dtype 由 `jit_npu._tvm_dtype_to_torch` 决定，该表**缺 uint8/uint16/uint32**；凡输出含这些 dtype 的算子，设计期必须显式规划「预分配输出张量传入」或标注该缺口，不能依赖 out_idx 自动分配。
- v-prefix 一元/二元 op 以**整 buffer** 调用（`T.vnot(x_ub, y_ub)`）稳定；把动态长度 slice（`x_ub[0:tail]`）直接传给 v-op 可能报 `int() argument must be ... not 'Var'`——尾块裁剪只包 `T.copy`，v-op 作用于全 buffer。

## Stage 2 (review) Retrospective — first_review — 2026-10-12

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Constraint-doc | API 文档与设计期探针实证冲突：`T.vxor.md` §2.1 载 src1 可为「Python 标量 int / float（提升并广播）」，而设计探针/编译错误称标量 src1 被拒（`hivm.hir.vxor ... cannot be a scalar`）；且探针实测传的是 `T.cast(1,dtype)`（PrimExpr）而非 Python 标量——检视方无法在无设备环境判定哪一方成立 | `docs/Tilelang.language/逻辑操作/T.vxor.md` §2.1 vs DESIGN.md §3.4 / §9 R-3 / §1.6.0 R4；探针 `history_version/design_probe_vnot_int.py` L28–L41（`kern_vxor1` 用 `T.cast(1,dtype)`） | 在 `docs/.../T.vxor.md` §2.1 明确「Python 标量 vs PrimExpr 标量」的接受边界，并注明底层 `hivm.hir.vxor` 的 `Src1().IsTensor()` 约束是否透传到前端 | D |
| Review-flow | 本 skill 维度 8 对「负向论断与 API 文档矛盾 → fail」的严格规则仅绑定 §1.6.3 弃选行；对 §1.6.0 R4 负向淘汰（有探针佐证但与文档矛盾、且不影响 Stage 3 可实现性）无显式定级指引，检视方只能自行裁量为 warn | `.agents/skills/tilelang-design-review/SKILL.md` §3 维度 8「向量化轴与布局决策完整性」行 vs 同一维度「调研结论正确性」行 | 在维度 8「调研结论正确性（独立复核）」行补一句：负向断言有实测佐证但与所引 API 文档矛盾、且采纳项独立成立不影响实现时，记 warn 并要求 Stage 1 收敛表述（不升级为 fail） | P |
| Review-flow | 迁移/算法维度中多处依赖「设计期探针实测」（bool dtype device `vector core timeout`、uint8 out_idx dtype 映射、`int4` 失败）——检视方无设备，只能核对设计内部记录自洽与文档一致性，无法复现设备级行为 | DESIGN.md §9.1 R-1/R-2、§3.4；`history_version/design_probe_*.py` | 在 skill 检视方式中明确：设备级实测类断言以「设计内部记录自洽 + 与文档/源码不矛盾」为 pass 口径，复现交 Stage 3（不在 Stage 2 判 fail） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `T.vxor` 的 Python 标量 src1：文档称可提升广播（§2.1），前端/底层可能要求 tensor——需设备复现以裁决，检视/设计期不可据此断言 | D | `docs/Tilelang.language/逻辑操作/T.vxor.md` §2.1 vs DESIGN.md §3.4/§9 R-3；探针仅测 `T.cast(1,dtype)` | `T.vxor(buf, 1, out)`（Python 字面量）编译一次；对比 `T.vxor(buf, T.cast(1,"int8"), out)` | tilelang 0.1.2 dev root build 2026-10-12 / Ascend910_9362 / CANN 9.0.0 | docs/Tilelang.language/逻辑操作/T.vxor.md |

### Transferable Lessons（可迁移教训）

- 检视迁移/优化类负向断言（“API 不支持/代价高”）时，必须打开其引用的 API 文档按条款核对；文档与探针/实测冲突时不得默认实测正确——若采纳项独立成立，记 warn 并建议收敛表述，只有影响 Stage 3 可实现性才升 fail。
- 迁移任务中核对 block_size/并行参数的「源一致性」依据时，要区分「源运行时实际值」与「迁移 harness 的折叠约定」（本任务 source `direct` 运行时 tile=256，而 K11 折叠约定 = threads×npt=2048）；两者混写会造成可追溯性误读。
- 检视 §1.6.0 调研「无命中条目」类论断时，须亲自检索 `algorithm-candidates.md` 的算子族命中行（本任务 `ALG-elementwise` 即命中行）——设计称“无命中”但族条目实际存在时，应核对族条目的 R1–R4 是否已被实质覆盖，避免漏候选。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-12

**结果**：`[PRECISION_PASS]`。产物 `bitwise_not_fwd_kernel.py`；L0 6 例 / L1 7 例 / L2 5 例 / Boundary 8 例全过，全部 `max_diff=0.0`（bit-exact）。DESIGN.md §0.6 三项重设计（R1 网格跨步 / R2 `T.vnot` / R3 bool 存储语义）端到端落地，R-6（v-op 整 buffer + 仅 `T.copy` 动态 slice）实现约束成立。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Template | `templates/op_template.py` 未提示「UB buffer 分配必须在 block 顶层无条件进行」——首次实现按直觉把 `ones_ub` 放进编译期 `if is_bool:` 内分配，立即触发 `Undefined variable: ones_ub`（trace 期）；模板/规范缺此约束会让每个「按 dtype 分支分配额外 buffer」的算子重复踩坑 | 首次 L0 运行 stderr `error: Undefined variable: ones_ub`；绕法见 `repro/TRAP-conditional-alloc.py` | 在 op_template / skill Phase 2 增加一条「`T.alloc_ub` 一律在 Kernel 顶层无条件分配；编译期 `if` 只用于选择 op/copy，不得用于绑定 buffer」 | P |
| Test-harness | 内嵌分层测试需自造 out_idx 自动分配路径的 dtype 兜底（uint8→float32, CG-2026-0014）与 bool 存储视图包装——skill 模板无「dtype 需 host 侧重解释」的测试样例，每类算子需自行重推 | `bitwise_not_fwd_kernel.py` `_run_tensor`（bool `.view(int8)`/`.view(bool)`；uint8 `y_f32.view(torch.uint8)[:N]`） | 在 skill 或 op_template 补一小节「输出/输入需 host 侧重解释 dtype（bool i1、uint8 out_idx 缺口）时的 golden 对照写法」 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `@T.prim_func` 内**在编译期 Python-bool `if` 中绑定 buffer**（`if is_bool: ones_ub = T.alloc_ub(...)`）→ trace 期 `error: Undefined variable: ones_ub`（诊断走 fd 2，Python 异常仅 `DiagnosticError: ... error diagnostics were emitted`）；**绕法**：buffer 在 Kernel 顶层无条件分配，编译期 `if` 只选择计算 | D | Stage 3 L0 首次运行 stderr；本任务修复（unconditional alloc）；`repro/TRAP-conditional-alloc.py` | `examples/bitwise_not/bitwise_not_fwd_kernel/repro/TRAP-conditional-alloc.py`（bad form→Undefined variable；good form→bit-exact，2026-10-12 实跑 PASS） | tilelang 0.1.2 dev root build 2026-10-12 / Ascend910_9362 / CANN 9.0.0 | pattern-library/traps-compiler.md（TRAP-tvm-parser-rules 第 3 条「条件 T.alloc 非法」的精确形态补充） |
| grid-stride persistent Kernel（`T.Kernel(40, is_npu=True)` + `T.serial` + `task<n_num` 守卫）上，bool-as-int8 存储视图 `fill(1)+vxor` 与 `T.vnot`（int8/16/32/64/uint8）**端到端 bit-exact**（max_diff=0.0，含 4M 非整除尾块）；动态尾块只包 `T.copy`、v-op 作用于整 buffer（R-6）成立 | D | `bitwise_not_fwd_kernel.py` L0/L1 实跑（6 dtype × 1M，4_000_000 int32/int64，2047/100/4096/4095/1/2/10 各档） | `python examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py --level all` | tilelang 0.1.2 dev root build 2026-10-12 / Ascend910_9362 / CANN 9.0.0 | pattern-library/elementwise.md（原 Stage 1 探针仅验证单 tile/小 shape，此为真实 persistent kernel 规模复验） |
| uint8 输出经 `@tilelang.jit(out_idx=[1])` 自动分配得 float32（CG-2026-0014），但 kernel 仍把正确的 N 字节写在 buffer 基址；测试侧 `y_f32.view(torch.uint8)[:N]` 可无损取回结果 | D | `bitwise_not_fwd_kernel.py` `_run_tensor` uint8 分支（1024×1024 及 100/4095 档 exact）；设计探针 `history_version/design_probe_uint8_outidx.py` | `y = kernel(x_uint8, N); y.view(torch.uint8)[:N] == torch.bitwise_not(x)` | tilelang 0.1.2 dev root build 2026-10-12 / Ascend910_9362 / CANN 9.0.0 | capability-gaps.md CG-2026-0014（补「bytes 基址可用」实证） |

### Transferable Lessons（可迁移教训）

- 写 `@T.prim_func` 时，**任何 `T.alloc_ub`/`T.alloc_shared` 都放在 Kernel 顶层无条件分配**；编译期 Python-bool `if` 只允许包裹「执行哪条 v-op / 哪次 `T.copy`」。把 buffer 绑定放进 `if` 体内，即使条件是无-else 纯名常量（会被 trace 折叠），后续在 `if` 体外引用仍报 `error: Undefined variable: <buf>`；该诊断由 C++ 渲染到 fd 2，Python 侧只看到 `DiagnosticError`，需重定向 fd 2 才能看到真实信息。
- NPU bool（i1）逐元素算子的落地范式：kernel 按 `int8` 声明 GM/UB，host 传 `bool_tensor.view(torch.int8)`、结果 `.view(torch.bool)` 对照；计算用存储语义（bool NOT → `T.fill(ones, cast(1,"int8")) + T.vxor(x, ones, y)`，即 `x^1`）。golden 必须用 bool 语义（`torch.bitwise_not(bool)`），不可用 int8 补码 `~x` 对照（`~0 == -1` ≠ `0^1 == 1`）。
- 输出 dtype 为 uint8/uint16/uint32 时不要依赖 `@tilelang.jit(out_idx=[...])` 自动分配（`jit_npu._tvm_dtype_to_torch` 缺这三项 → 落 float32）；standalone 测试里可用 `out_f32.view(torch.uint8)[:N]` 取回 kernel 写在基址的原始字节，TileOPs 集成层应显式预分配输出张量传入。
- 动态尾块（非整除 N）只把 `tail` slice 交给 `T.copy`（`T.copy(x[off:off+tail], x_ub[0:tail])`）；v-op（`T.vnot`/`T.vxor`）必须作用在整块 UB buffer 上。整块 v-op 会读入尾部未初始化数据，但只搬出 `[0:tail]`，结果不受影响——这是 1-D elementwise 尾块处理的安全惯例。
