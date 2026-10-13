# RETROSPECTIVE.md — clamp_scalar_fwd_kernel

> 追加式日志：每次 Stage 执行追加一个章节，不覆盖历史。schema 见 `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`。

## Stage 1 (design) Retrospective — first_design（migration-harness）— 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | `gate_lint.design_lint` 的 `S1-EQUIV-EXEC` 用 `"EQUIV_FAIL" in r161` 判定「存在未通过项」，导致 §1.6.1 中**声明无 FAIL 的句子**（如「无 EQUIV_FAIL 项」）本身触发 fail——机械规则的否定式表述被当成了肯定证据 | 首次 gate 1 失败：`{"rule_id": "S1-EQUIV-EXEC", "message": "§1.6.1 存在 EQUIV_FAIL 项..."}`；DESIGN.md §1.6.1 原句「全部 16 行 EQUIV_PASS，无 EQUIV_FAIL 项」；改为「无未通过项（无 FAIL）」后 gate 通过 | `.agents/tools/gate_lint.py` 的 `S1-EQUIV-EXEC` 改为只在**结果表行**（如含「结论」列且值域为 EQUIV_FAIL）判定，或 design-template 明确要求「禁止在 §1.6.1 正文出现 EQUIV_FAIL 字面量」 | R |
| Design-flow | `gate_lint.TEMPLATE_BRACE_RE`（`\{[^{}\n]*[CJK][^{}\n]*\}`）把 LaTeX 数学公式里的**合法花括号**（`\text{或 present bound}`、`\text{...中文...}`）误判为未替换模板变量，触发 `S1-PLACEHOLDER` | 首次 gate 1 失败：`疑似模板变量未替换：{或 present bound}` / `{absent bound 侧不施加；中间 fp32，输出同输入 dtype}`；DESIGN.md §1.3 公式；把中文移出 `{}`（用 `\mathrm`/`or` 英文 + 括号外注释）后通过 | `.agents/tools/gate_lint.py` 的 `TEMPLATE_BRACE_RE` 排除 `$...$`/`\text{...}` 数学跨度；或 design-template §1.3 提示「数学公式花括号内不得出现 CJK，中文说明移到公式外」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| NaN 传播 clamp 迁移的**免常数**单点恢复链：普通 `vclamp/vmax/vmin`（丢 NaN）→ `vcmp(x,x,"ne")`（NaN 掩码）→ `vselect(mask, x, clamp, out)` **直接回选原始输入 x 的 NaN 位模式**（而非 `vbrc(canonical NaN)` + vselect）——省 1 个向量 op，且完全规避 VP-2026-0142「`T.cast(float("nan"),"float32")` NaN FloatImm 破坏 kernel cache JSON 元数据」陷阱 | P | DESIGN.md §0.6 R2 / §1.6.1 项 1、3 / §1.6.2 #6 / §3.3；VP-2026-0142（queue.md L915-931，其「可选绕法 = 算术方式构造 NaN 常数」可被本方案取代——直接回选 x 根本不需 NaN 常数） | `python examples/clamp_scalar/clamp_scalar_fwd_kernel/verify_equiv.py`（项 3：回选 x 的 NaN 位模式 vs canonical NaN，NaN 失配 0 / 0 ulp / EQUIV_PASS） | tilelang 0.1.2 dev-root（`/home/developer/tilelang-mlir-ascend/build`）+ Ascend（实查 20 AI Core）+ CANN 8.5.2 / 2026-10-13 | pattern-library/elementwise.md；并作为 VP-2026-0142 的绕法候选补入 traps-compiler.md |
| **NaN bound 全 NaN 语义**：torch.clamp（2.7.1）对 `min=NaN` / `max=NaN` 返回**整个输出全 NaN**；Ascend vmax/vmin 为 NaN-dropping（PL-1.11 P8），普通 clamp + 仅 x 的单点恢复会漏掉 NaN bound → 迁移须加 trace-time `math.isnan(bound)` 退化分支（全 NaN 输出，且仅用有限常数 0.0：`vmul(x,0.0)`+`vdiv(·,0.0)`） | P | DESIGN.md §0.1 边界语义 / §0.6 R3 / §1.6.1 项 2；本机 `torch.clamp(tensor([1.,-2.,nan]), min=float('nan'), max=0.5)` → 全 nan | `python examples/clamp_scalar/clamp_scalar_fwd_kernel/verify_equiv.py`（min=NaN / max=NaN × 3 dtype：0 ulp / 0 违反 / EQUIV_PASS） | tilelang 0.1.2 dev-root + Ascend（实查 20 AI Core）+ CANN 8.5.2 / 2026-10-13（torch 2.7.1+cpu） | pattern-library/elementwise.md（clamp 族边界语义） |
| clamp（标量边界）NPU 迁移形态：`vclamp`（both）/`vmax`（min-only）/`vmin`（max-only）按 present 分支，bound 作**标量 operand** 传 v-prefix；fp32 中转对齐 torch fp32 opmath golden | C | DESIGN.md §0.3 / §0.5 / §3.2 / §3.5.2；源 `_base.py` `ScalarParamUnaryKernel.DEFAULT_THREADS=256` + register_copy；`docs/Tilelang.language/比较操作/T.vclamp.md`（标量 min/max） | none（实现期验证） | tilelang 0.1.2 dev-root + Ascend + CANN 8.5.2 / 2026-10-13 | pattern-library/elementwise.md（clamp/relu6 族迁移形态） |

### Transferable Lessons（可迁移教训）

- **免 NaN 常数的单点恢复**：迁移带 NaN 传播语义的逐元素 clamp/激活算子时，恢复 NaN 不必构造 canonical NaN 常数再 `vbrc` 广播；把 `vselect` 的「真值」分支直接设为**原始输入 x**（`vselect(mask, x, claim_result, out)`）即可回填 x 的 NaN 位模式。这既省一个 `vbrc` op，又完全避开 VP-2026-0142（`T.cast(float("nan"),"float32")` 的 NaN FloatImm 破坏 kernel cache JSON 元数据 → 每次 miss 重编译）。前提：`vselect` 的 Cond/A/B/Out 同 shape 且 `bool` 掩码仅片上使用（不可写回 GM）。
- **NaN bound 的语义分水岭**：`torch.clamp(x, min=NaN)` / `clamp(x, max=NaN)` 返回**整个输出全 NaN**（非逐元素），而 Ascend `vmax/vmin` 丢 NaN。因此「NaN 只能源自输入 x」的单点恢复论证**仅对 finite bound 成立**；present bound 为 NaN 时必须在编译期（trace-time，bound 是构造期标量）特化出全 NaN 分支，否则 min/max 会丢 NaN 给出错误结果。构造全 NaN 亦可用有限常数：`vmul(x, 0.0)`（有限→0、±inf→NaN）后 `vdiv(·, 0.0)`（0/0→NaN）。
- **gate_lint 否定式表述陷阱**：`S1-EQUIV-EXEC` 用字符串包含判定 EQUIV_FAIL，§1.6.1 正文里「无 EQUIV_FAIL 项」这类**声明性否定句**会被误判为存在失败项——写设计时避免在 §1.6.1 出现 `EQUIV_FAIL` 字面量（用「无未通过项」表述）。
- **gate_lint 数学花括号陷阱**：`TEMPLATE_BRACE_RE` 匹配「含 CJK 的 `{}`」，LaTeX 里的 `\text{中文}` 会被误判为未替换模板变量；数学公式内的花括号不得含 CJK，中文说明移到公式外或改用英文。
- 物理核数一律以 `NPUUtils.get().get_aicore_num()` 实查为准（本机返回 20，纯 Vector ×2 = 40；与 `CONST-aicore-910B2C` 记录的 24 不一致），并在 §5.5 记录查询代码与返回值。

## Stage 3 (develop) Retrospective — first_impl（migration-harness）— 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | none（首次实现即通过：L0 7/7、L1 42/42、L2 4/4、Boundary 14/14 全 PASS，max_diff 全 0.00e+00；无编译/运行时/精度返工，≤1 attempt） | `python clamp_scalar_fwd_kernel.py --level all` → `All check passed!` | none | — |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **R2 免常数单点 NaN 恢复链在真实 NPU 端到端实证**：普通 `vclamp/vmax/vmin`（丢 NaN）+ `vcmp(x,x,"ne")` 自比较 NaN 掩码 + `vselect(mask, x32, clamp, out)` 回选原始 x 的 NaN 位模式 → NaN in -> NaN out，三 dtype 全通过，且输入 NaN 位置与输出 NaN 位置**逐位相等**（nan_count=512 全对齐）。同时证明 `npuir_cmp "ne"` 在 Ascend 上确为 IEEE 无序比较（NaN≠NaN→true，±inf/有限→false，inf 不被误判为 NaN） | D | `clamp_scalar_fwd_kernel.py` run_boundary：`[Boundary] PASS: NaN->NaN (R2 vcmp(ne)+vselect verified) dtype={float32,float16,bfloat16} nan_count=512`；`[Boundary-inf] PASS` 三 dtype | none（Boundary 套件即实证，含 NaN mask 逐位断言） | tilelang 0.1.2+417dae99 dev-root（`/home/developer/tilelang-mlir-ascend/build`）+ Ascend 910B（实查 AI Core 20，纯 Vector×2=40）+ CANN 8.5.2 / 2026-10-13 | pattern-library/elementwise.md；佐证 CONST-vcmp-ne-nan-unordered（本任务重验通过） |
| **R3 NaN-bound 全 NaN 退化分支端到端实证**：trace-time `math.isnan(bound)` 检测 → 仅用有限常数 0.0 构造全 NaN（`vmul(x,0.0)` 有限→0、±inf→NaN；`vdiv(·,0.0)` 0/0→NaN）→ 输出全 4096 NaN，与 `torch.clamp(x,min=nan)`/`max=nan` 的 golden 全 NaN 一致；三 dtype 均过 | D | `clamp_scalar_fwd_kernel.py` run_boundary：`[Boundary] PASS: NaN-bound -> all NaN dtype={...} nan_out=4096`（min=nan/max=nan 各 3 dtype 共 6 例） | none（Boundary 套件即实证） | tilelang 0.1.2+417dae99 dev-root + Ascend + CANN 8.5.2 / 2026-10-13（torch 2.7.1+cpu） | pattern-library/elementwise.md（clamp 族边界语义） |
| **vmax/vmin/vclamp 标量 operand 接受形态**：`AscendBinaryOp.buildTirCall` 对 `src1` 仅显式特判 `int/float`（→`tir.const(...)`）与 `tir.expr.Var`；但实测 `T.cast(-0.5,"float32")`（FloatImm）作 vmax 标量 operand 同样编译且数值正确（min 精确=-0.500）。本实现为稳妥统一传**原始 Python float**，两种形态均可 | D | 探针 `/tmp/probe_vmax_scalar.py`：`use_cast=False: OK min=-0.500` / `use_cast=True: OK min=-0.500` | none（非陷阱，正向确认；探针未转正——两种形态均正确，无可断言的反例） | tilelang 0.1.2+417dae99 dev-root + Ascend + CANN 8.5.2 / 2026-10-13 | docs/Tilelang.language/比较操作/T.vmax.md（可补「标量 operand 可为 Python 标量或标量表达式」） |

### Transferable Lessons（可迁移教训）

- **R2/R3 迁移配方可直接复用**：本任务把 DESIGN §0.6 的「普通 clamp + `vcmp(x,x,"ne")` 单点回选 x + NaN-bound trace-time 全 NaN 退化」在真实 NPU 三 dtype 上端到端跑通（max_diff 全 0）；后续同族 NaN 传播逐元素算子（relu6/clamp_* 等）可直接照搬：`vclamp/vmax/vmin` 按 present 选链 → `vcmp(x,x,"ne")` 掩码 → `vselect(mask, x32, clamp_out, buf2)`（非原位，写独立 buffer，规避 TRAP-vselect-inplace-carried-state）；present bound=NaN 时 trace-time 短路为 `vmul(x,0.0)`+`vdiv(·,0.0)` 全 NaN。
- **v-prefix 标量 operand 的稳健写法**：vmax/vmin/vclamp 的标量边界传**原始 Python float**（前端 `tir.const(scalar, dst.dtype)` 特判路径）最稳；`T.cast(float,"float32")`（FloatImm）实测也能用，但传原始标量语义更直白、与 `AscendBinaryOp` 的 int/float 特判一致。
- **fp32 全 buffer 显式分配（168KB）在 block_size=8192 可编译通过**：无需按 DESIGN §4.5 的 fp32 复用方案（`x32←x_ub`、`y_ub←buf2`）也能过 UB 192KB（hardsigmoid 同款结构）；复用是余量优化而非编译必需。
- **persistent 分核 `num_kernels=min(ceil(N/block_size), vector_cores)` + 核内 `T.serial(ceil(num_logical/num_kernels))` 静态边界**在本任务全档位（N=1 … 16M）均正确，尾块 `tail_size=T.min(block_size, shape-offset)` + 双显式 slice `T.copy` 无越界。
