# RETROSPECTIVE — GeluAndMulFwdOp（`gelu_and_mul_fwd_kernel`，migration-harness）

## Stage 1 (design) Retrospective — first_design — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | GPU 融合门控算子的 manifest 归属易被误指：调度 prompt 的 knowledge 预注入按族名指向 `examples/TileOPs/tileops/manifest/elementwise.yaml`，而 GPU 源 manifest 实为 `src/tileops/manifest/spec/elementwise_fused_gated.yaml`（NPU 镜像把该族并入 elementwise.yaml）；照字面引用会引到不存在的 `examples/TileOPs/tileops/manifest/elementwise_fused_gated.yaml` 而被 `S1-REF-PATHS` 判 fail | examples/TileOPs/tileops/manifest/elementwise.yaml（GeluAndMulFwdOp 在 L399）；GPU 源 `spec/elementwise_fused_gated.yaml` | harness 预注入对「族名 ≠ 文件名」的算子补一句「NPU 镜像 manifest 文件路径」（或直接给出镜像内 op 条目所在文件） | R |
| Design-flow | `S1-REF-PATHS` 的 `REF_PATH_RE` 会把省略号缩写（如 prose 里 `examples/TileOPs/.../_gqa_...py`）当作具体路径并报「不存在」——引用 examples 先例时天然想省中间目录 | DESIGN.md §0.5 初稿含 `examples/TileOPs/.../_gqa_prefill_fwd_kernel.py`，gate 报 S1-REF-PATHS；`.agents/tools/gate_lint.py` L43-45,L65-85 | REF_PATH_RE 跳过含 `...` 的 token，或 design 自检清单注明「prose 中引用长路径必须写全，不得用 `...` 缩写」 | R |
| Design-flow | 与 add 同款：`S1-EQUIV-EXEC` 以裸子串判定「EQUIV_FAIL」，否定式表述会被误判 —— 表达「全部通过」时须避开该字面量（含否定） | 本设计 §1.6.1 一律写「无未通过项」而非「无 EQUIV_FAIL」；`.agents/tools/gate_lint.py` L320-328；add RETROSPECTIVE 同型记录 | gate_lint 改为「存在 `EQUIV_FAIL` 作为结论值」，或在 design 模板注明禁止该字面量 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **NPU 原生向量 erf（`T.verf`）可整体替代 GPU 的 clamps + 10 次多项式 erf 规避链**：`docs/Tilelang.language/数学操作/T.verf.md` 载明 `T.verf(src,dst)`（定义/值域/指令映射，fp16/fp32 √、bf16 ×、Shape 无特殊要求），`customize_npuir.py` 的 `npuir_verf` 别名 `verf`。GELU/erf 类迁移应删除多项式近似（~14 向量 op → 1），bf16 经 fp32 中转；NaN 语义未文档化，须按 hardsigmoid 先例备 `vcmp(x,x,"ne")+vselect` 恢复 | P | `docs/Tilelang.language/数学操作/T.verf.md`；`tilelang/language/customize_npuir.py` `npuir_verf`/`verf` 别名；GPU 源 `_erf.py`（多项式动机） | `examples/TileOPs/tileops/kernels/elementwise/gelu_and_mul/gelu_and_mul_kernel/gelu_and_mul_fwd_kernel_DESIGN.md` §0.6 R2 / §1.6.1 | tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B2C（实查 20 AI Core） | pattern-library/constants.md 或 elementwise.md |
| **融合门控 `[M,2N]→[M,N]` 单张量布局在 npuir 的落地形态**：gate=x[m, n0:n0+tail]、value=x[m, N+n0:N+n0+tail] 为同一行内**两段连续内存**，用二维切片 `T.copy`（标量行 + 列切片，rank 折叠）等价承接 GPU 的逐元素列偏移 `x[by,col]`/`x[by,N+col]`，无需 host 侧切分、无 repack；切片+标量混合索引在 `_gqa_prefill_fwd_kernel.py` 有先例 | P | `docs/Tilelang.language/内存操作/T.copy.md`（切片语法 + rank 折叠规则）；`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`（混合索引先例） | `examples/TileOPs/tileops/kernels/elementwise/gelu_and_mul/gelu_and_mul_kernel/gelu_and_mul_fwd_kernel_DESIGN.md` §1.6.3 / §3.2 | tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B2C | pattern-library/elementwise.md |
| **多项式 erf 不可用于 fp32 路径**（若回退原生 erf）：GPU `_erf` 多项式在 fp32 下对 fp64 参照 max **5.5634** tolerance-unit、5 shape 累计 **199** 违反（远超 fp32 1e-5 容差）——源码正是因此对 fp32 改走精确 `T.erf`；迁移 fp32 路径不得沿用多项式 | D | `examples/gelu_and_mul/gelu_and_mul_fwd_kernel/verify_equiv.py`（`counter fp32-poly` 段，v1 新增）；GPU 源 `_erf.py` L67-68（out_dtype=="float32" 走 `T.erf`） | `python3 examples/gelu_and_mul/gelu_and_mul_fwd_kernel/verify_equiv.py` | tilelang 0.1.2 dev-root + CANN 8.5.2 / 2026-10-13 | pattern-library/traps-runtime.md（数值口径类） |

### Transferable Lessons（可迁移教训）

- 融合门控激活（gelu_and_mul / silu_and_mul 族）迁移：**输入是单个 `[M, 2N]` 张量，gate/value 是沿最后一维切半的前后半**——Op 层只校验并转发该张量（host 不切分，符合 ascend-constraints.md §4），kernel 内以列偏移 `N` 读 value。行内两段连续内存天然适配二维切片 `T.copy`，无需重排。
- NPU 有**原生向量误差函数 `T.verf`**（fp16/fp32），GPU 为规避昂贵 erff 而写的 clamps+10 次多项式近似**应整体删除**——这是本算子最大的发射项削减（~14 op → 1）；bf16 因 `T.verf` 不支持必经 fp32 中转。`T.verf` 的 NaN 语义未文档化，须按 hardsigmoid 先例备 `vcmp(x,x,"ne")+vselect` 单点恢复。
- erf/GELU 的 fp16/bf16 迁移应走**全 fp32 域链 + 末尾单次 cast**：既对齐 torch fp32 opmath golden，又避免 GPU 窄 dtype 尾部乘加的二次舍入（§1.6.1 项 2 机器验证 0 违反）。
- 写等价验证脚本时，**基线必须与源码逐 dtype 对齐**：本算子 fp32 源路径本就用精确 erf（无多项式），故 item 1 的多项式基线仅对 fp16/bf16 有意义，fp32 行按「无多项式」处理——否则会造出虚假的「多项式 vs 原生」差异。

## Stage 1 (design) Retrospective — revision（v0 → v1）— 2026-10-13

> 驱动：Stage 2 `REVIEW.md` 结论「不通过」（1 阻塞 + 3 建议）。上次冻结稿 `history_version/design_v0.md`。

### 修订摘要

| # | 级别 | 问题 | 处置 | 验证 |
|---|------|------|------|------|
| 1 | 阻塞 | §0.1/§0.6 R2/§8.3/§9.3 断言 `gelu(-inf)=0`，与源式 `x*half*(one+erf_val)`（`-inf*0=NaN`）及 §8.1 golden（`F.gelu(-inf)=NaN`）不符 | 重新 Read 源，四处改为 `gelu(-inf)=NaN`、`gelu(+inf)=+inf`；§8.3 Boundary 预期值改 NaN | `verify_equiv.py` 新增 `BOUNDARY` 自检（源式 + golden 双路径、三 dtype，全 OK）；设计内自洽 |
| 2 | 建议 | `T.verf` 收益「→1 op」口径不精确（lowering 为多 op 组合） | §0.6 R2/§1.6.0/§1.6.1 改为「DSL 层 1 次调用替代约 14 条显式向量 op（lowered 为少量向量 op 组合）」 | 文档口径一致 |
| 3 | 建议 | §1.6.1 fp32 多项式对照项 5.56 无脚本产出 | `verify_equiv.py` 新增 `counter fp32-poly` 段（拒绝基线控制组）；实测 max 5.5634、累计 199 违反 | 脚本重跑产出，§1.6.1 逐 shape 引用 |
| 4 | 建议 | `block_N = 256×8 = 2048` 对 fp32 不精确 | §0.3/§0.4/§5.2/§3.5.2 更正为「2048 为 NPU 选定列块宽；GPU 折叠值 fp16/bf16=2048、fp32=1024」 | 源码 `_policy.default_launch_config` 复核 |

机械复核（本版重跑）：`total rows: 30   EQUIV_PASS: 30   not-pass: 0`，`OVERALL: EQUIV_PASS`，exit 0；`counter fp32-poly: rows=5 max_ulp=5.5634 total_viol=199`；`BOUNDARY` 三 dtype 全 OK。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-review | 边界语义（`±inf`/NaN）是**语义正确性**维度的高频阻塞源：作者易用数学极限（`erf(±inf)=±1 ⇒ gelu(-inf)=0`）替代**源表达式逐项代入**，且不主动与自带 golden 互证，导致设计内自相矛盾并污染 §8.3 测试计划 | v0 §0.1 L43 同行既写「`x*0` 不特殊处理」又写 `gelu(-inf)=0`；REVIEW.md 问题 1 | `migration-analysis.md`（或 design 模板 §0.1 表）增一行硬性要求：**边界语义结论必须给出源式逐步代入过程，并标注与 §8.1 golden 的实测一致性**；`design_lint` 可加「若 §0.1 出现 `gelu(-inf)=0` 类断言而 golden 未佐证则 warn」 | R |

### Transferable Lessons（可迁移教训）

- **边界语义不能靠数学极限直觉**：`erf(±inf)=±1` 只决定 `one+erf_val`，最终值由**外层表达式**决定。本例 `x*half*(one+erf_val)` 在 `x=-inf` 处是 `-inf*0`（不定式）→ **NaN**，不是 0。凡「饱和值 × 0 / 饱和值 + 抵消项」的融合算子（gelu_and_mul、silu_and_mul 等）都必须在 `±inf` 处按源式逐步代入。
- **边界结论必须与自带 golden 互证**：本设计 §8.1 golden 为源仓基准（`F.gelu`），设计阶段对边界点实跑 golden 即可立即发现 `gelu(-inf)=0` 与 `F.gelu(-inf)=NaN` 冲突。golden 不只是 Stage 3 的比对目标，也是 Stage 1 的自检尺。
- **把边界语义写进 `verify_equiv.py` 的机器自证**（本版新增 `BOUNDARY` 段，源式与 golden 双路径）：将「一次人工阅读才能发现的语义错误」转成「Stage 2 机械复核即可拦截」，是修订防复发的最低成本手段。
- **收益口径要与 lowering 对齐**：声称 `T.verf`「1 op」是把 DSL 调用数与 lowered 指令数混为一谈；表述收益时应写清「DSL 层 1 次调用 / lowered 为若干向量 op 组合」，避免误导后端发射评估。
- **对照项数字必须有脚本出处**：设计正文引用「工具未产出、只在别处出现」的数字（如 fp32 多项式 5.56）会形成证据缺口；要么在 `verify_equiv.py` 补齐拒绝基线控制组产出，要么删除该数字。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-13

> 出口：**`[DESIGN_ERROR]`**。根因：冻结设计 §0.6 R2 选定的 erf 实现 `T.verf`
> 在本工具链上**不是 erf**——`CodeGenTileLangNPUIRAPI::VerfCodegen`
> (`tilelang-mlir-ascend/src/target/codegen_npuir_api.cc:2423`) 将其 lower 为**截断到的
> 4 项 Maclaurin 级数** `erf(x) ≈ (2/√π)(x − x³/3 + x⁵/10 − x⁷/42)`，仅 |x|≲1 可用；
> GELU 的 erf 参数 `gate/√2 ~ N(0, 0.5)` 约 16% 落在 |x|>1，误差急剧发散（x=1.877 时
> `verf=0.056` vs `erf=0.992`；x=3.6 时 `verf=−155.8`；±inf→NaN）。`T.erf` 经
> `math::ErfOp → UnaryFn::erf → hivm::VErfOp` 落到**同一** codegen，无可用替代。
> 忠实按设计实现后 `--level all`：**30/30 用例失败**（L0 12/12、L1 18/18），
> 最大绝对误差 fp16 907 / fp32 1.2e2，违反率 7.9%–38.7%。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop | 设计 Phase M1 对 `T.verf` 的可行性判定只做了「文档读 + 源码别名核对」，未做**数值域实测**；`T.verf.md` 声称值域 (-1,1)、Shape 无特殊要求，却未载明其 lowering 是低阶截断级数、有效域 | `docs/Tilelang.language/数学操作/T.verf.md`（定义/值域/指令映射，无精度/域说明）；`codegen_npuir_api.cc:2423` VerfCodegen；官方测试 `testing/npuir/arith_ops/test_verf_dev.py:60` 仅在 (-1,1) 采样、rtol=atol=1e-2 | design skill 的 API 可行性确认清单增加「对数值算子（erf/exp/log/tanh…）必须核对 **lowering 实现与有效数值域**，不能只核对签名/别名」；`docs/Tilelang.language/数学操作/*` 补「有效域/精度」列 | R |
| Develop | 算子迁移把「NPU 有原生向量 erf」当作既定事实并据此删除多项式链，但该结论源自文档标题与值域，未经小值域外实测；这是跨函数高风险点 | 本设计 §0.6 R2 / §1.6.1 全建立在 `T.verf` 上；RETROSPECTIVE（Stage 1）Value Point 亦把 `T.verf` 列为「应整体替代多项式」的 P 类结论 | 知识域中用 `T.verf` 的结论必须附**域外数值证据**（版本戳），否则降级为「未文档化假设」 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **npUIR `T.verf` 是截断 Maclaurin 级数（非 erf），有效域仅 \|x\|≲1，域外发散、±inf/NaN→NaN**：`codegen_npuir_api.cc:2423` `VerfCodegen` 生成 `(2/√π)(x−x³/3+x⁵/10−x⁷/42)`；x=1.877→0.056（erf 0.992）、x=3.6→−155.8、±inf→NaN。任何依赖全域 erf 的算子（gelu erf 形式、silu? 否、errfunc、normal cdf）**不能**用 `T.verf` | D | `tilelang-mlir-ascend/src/target/codegen_npuir_api.cc:2423-2470`；官方 test 仅 (-1,1) | `examples/gelu_and_mul/gelu_and_mul_fwd_kernel/repro/TRAP-verf-taylor-truncation.py`（自含断言，实跑通过） | tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B2C（实查 20 AI Core） | pattern-library/traps-runtime.md（数值域类） |
| **`T.erf` 与 `T.verf` 在本工具链落到同一 hivm::VErfOp codegen**，故「回退到 T.erf」不是可用替代 | D | `MathToHFusion.cpp:183` `ErfOp→UnaryFn::erf`；`HFusionToHIVM.cpp:249` → `hivm::VErfOp`；`ConvertHIVMToUpstream.cpp:1095` | 同上 repro（旁证：文档 §3 与 codegen 一致） | tilelang 0.1.2 dev-root + CANN 8.5.2 | pattern-library/traps-compiler.md |
| **融合门控 `[M,2N]→[M,N]` 的 2-D 列切片 `T.copy` 形态在本平台可编译可运行**（标量行 + 列切片 → 1-D UB；尾列 `T.min`），与 Stage 1 预判一致；失败仅来自 erf 数值，非搬运/分核/边界 | P | 本算子实现（`gelu_and_mul_fwd_kernel.py`）真实 NPU 编译并跑通 30 用例（全部因 erf 数值失败）；repro 亦用同型 | `examples/TileOPs/tileops/kernels/elementwise/gelu_and_mul/gelu_and_mul_kernel/gelu_and_mul_fwd_kernel.py` | tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B2C | pattern-library/elementwise.md |

### Transferable Lessons（可迁移教训）

- **不要把文档标题当数值承诺**：`T.verf` 文档写「误差函数」「值域 (-1,1)」，但其 lowering 是 4 项截断级数，域外灾难性发散。迁移中凡把某 API 当作数学原语（erf/exp/log/tanh/sqrt…）的前提，Stage 3 必须先做**有效域外的数值实测**再定稿，不能只核对签名与别名。
- **GELU erf 形式（以及一切全域 erf 需求）在本工具链上当前无可用实现**：`T.verf`/`T.erf` 同源同 codegen 均为截断级数。设计修订需改用**显式 clamp + 高阶 minimax 多项式**（fp16/bf16 可用 GPU 同款 clamp(±3.6)+10 次 Horner；fp32 需更高阶或有理逼近以满足 atol/rtol 1e-5），并在 host 侧对域外输入做饱和/NaN 处理（级数在 ±inf 处给 NaN，而语义要求 `gelu(+inf)=+inf`）。
- **边界语义仍需显式兜底**：即便换用多项式，`gelu(-inf)=NaN`（`-inf*0`）、`gelu(+inf)=+inf`、NaN→NaN 必须由 kernel 侧选择逻辑保证，不能依赖近似函数的隐式饱和。
- **「看似 API 缺失」应先实测 lowering**：本次在 Stage 3 才发现设计层 R2 不可实现，代价是整个 develop 尝试；后续同类任务应在设计阶段（M1 可行性）就加入「数值算子域外实测」门槛。

## Stage 1 (design) Retrospective — revision（v1 → v2）— 2026-10-13

> 驱动：Stage 3 `[DESIGN_ERROR]`（v1 选定的 `T.verf` 在本工具链 lowering 为截断 4 项 Maclaurin 级数）。上次冻结稿 `history_version/design_v1.md`。

### 修订摘要

| # | 级别 | 问题 | 处置 | 验证 |
|---|------|------|------|------|
| 1 | 阻塞 | v1 §0.6 R2 把 erf 建立在 `T.verf`（及 `T.erf`）上，二者在本工具链均为**截断级数**（非真 erf，仅 `\|x\|≲1` 有效），GELU 参数 ~16% 落在 `\|x\|>1`，kernel 全域失败 | **重做 erf 机制**：删除 `T.verf`/`T.erf`，改用**显式高阶 clamped minimax 多项式**（deg 20，`A=4`，`w=1-(t/4)²`，`c0=0.25` 精确饱和）在 fp32 域求值，供全 dtype 共用；`erf(NaN)=NaN` 由 `vcmp(x,x,"ne")+vselect` 恢复。同步改写 §0.4/§0.5/§0.6 R2/§1.4/§1.5/§1.6.0–§1.6.2/§3.1–§3.4/§4/§5/§6/§8/§9 | `verify_equiv.py` 新增 `RANGE`（全域误差 2.98e-7 / 精确饱和 / NaN 传播）与 `BOUNDARY`；**42 行采纳项全 EQUIV_PASS**（exit 0），§0.0 增「为何不会再犯同一错误」 |

保留的 v1 良好部分：gated `[M,2N]` 单张量布局、行内两段连续切片、列轴向量化、一维 `T.Kernel` + `block_N` 折叠、分核三要素、全 fp32 域链 + 末尾单次 cast、`gelu(-inf)=NaN` 边界语义。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 设计 M1「API 可行性」对数值算子只核签名/别名/文档，缺「lowering 实现 + 有效数值域」核验环节，导致跨阶段返工 | v1 §0.6 R2 建立在 `T.verf` 上（文档存在但 lowering 为截断级数）；Stage 3 `[DESIGN_ERROR]` | `tilelang-op-design` 的 API 可行性清单增加硬性行：**数值原语（erf/exp/log/tanh/sqrt…）必须核对 lowering 实现与有效域，附域外数值证据**；`docs/Tilelang.language/数学操作/*` 补「有效域 / 精度」列 | R |
| Design-flow | 等价性机器验证脚本缺「全域/边界」维度，只在随机点 + 有限角点比对，无法拦截「近似函数仅局部有效」类缺陷 | v1 `verify_equiv.py` 无 `RANGE` 段（仅有随机 shape + 角点） | design skill 的 D-1 验证模板增 `RANGE` 段（全域误差 + 饱和 + NaN 传播），对数学近似函数强制 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **本工具链可用的全域 erf 落地形态：显式 clamp(±4)+deg20 minimax 多项式（`w=1-(t/4)²`，`c0=1/4` 精确饱和）在 fp32 求值，全域 max abs err 2.98e-7、`erf(±inf)=±1`、`erf(NaN)=NaN`、`gelu(-inf)=NaN`/`gelu(+inf)=+inf`**——可作 GELU-erf 类迁移的参考实现；系数与验证脚本见算子目录 | P/D | `examples/gelu_and_mul/gelu_and_mul_fwd_kernel/verify_equiv.py`（`RANGE`/`BOUNDARY` 段，42 行全 EQUIV_PASS）；`DESIGN.md` §0.6 R2 / §4.7 | `python3 examples/gelu_and_mul/gelu_and_mul_fwd_kernel/verify_equiv.py` | tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B2C（实查 20 AI Core） | pattern-library/elementwise.md |

### Transferable Lessons（可迁移教训）

- **数值原语的可行性 = 签名存在 ∧ lowering 语义正确 ∧ 有效域覆盖**：`T.verf` 文档标题/值域都在，但 lowering 是截断级数——「文档存在」不等于「语义正确」。设计阶段须直接读 codegen（本版读 `codegen_npuir_api.cc` `VerfCodegen`）并做域外数值实测，否则错误会推迟到 Stage 3 才暴露。
- **数学近似必须用「全域」脚本机器自证**：v1 的 `verify_equiv` 只在随机点比对，无法发现「仅在 `\|x\|<1` 有效」；新增 `RANGE`（全域误差 + 饱和 + NaN）段是拦截该类缺陷的最低成本手段。
- **近似函数的饱和/NaN 需显式构造**：多项式不会自动在 `\|x\|` 大时给 `±1`，也不会自动传播 NaN。本设计用「参数 clamp + 输出 clip + `c0=1/A` 精确」保证精确 `±1`，用 `vcmp+vselect` 保证 NaN→NaN，从而使 `gelu(-inf)=NaN` 复现。
- **纯多项式逼近的 fp32 条件数**：deg≤16 的 power 基拟合在 fp32 下因系数大/抵消而超差；deg 18–20（系数幅度 ≤2）才稳定。必要时改用 Chebyshev/Clenshaw 或 exp 型 minimax（~20 op，发射项更低）。

## Stage 3 (develop) Retrospective — first_impl (attempt 1, v2 polynomial erf) — 2026-10-13

> 出口：**`[PRECISION_PASS]`**。v2 设计（显式 clamp(±4) + deg-20 多项式 erf，fp32 求值，
> `c0=1/A=0.25` 精确饱和，`vcmp(x,x,"ne")+vselect` NaN 恢复）在真实 NPU 上实现并跑通
> `--level all`：**L0 12/12、L1 全过**（阻塞层全绿），Boundary 全过，L2 仅记录（非阻塞）。
> erfc 机制不再使用 `T.verf`/`T.erf`。调试链 ≤2 个实现级 bug 修复（编译期 loop 形态、
> 操作数次序），无设计级问题。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | DESIGN §3.3 伪代码用 Python 级 `for c in ERF_COEFFS[1:]` 表达 Horner 循环，但 `@T.prim_func` 体内只接受 TIR 循环（`range`/`T.serial`/`T.unroll`…），Python 序列迭代被 parser 直接拒绝（`Expect the for loop to be one of …`），必须手工展开 20 步 | `gelu_and_mul_fwd_kernel.py`（v2 首版）编译报 `DiagnosticError`；报错指向 `for c in ERF_COEFFS[1:]` | design-template §3.3 / develop 清单注明「prim_func 内对**编译期常量序列**的循环（Horner 系数等）必须字面展开，不得写 Python `for`」 | R |
| Develop-flow | DESIGN §3.3 API 映射写 `T.vsub(w32, 1.0, w32)  # w = 1 - u^2`，误把 `vsub` 当「目标=左-右」或标量在左；实际 `vsub(src0,src1,dst)=src0-src1` 且**标量只允许在 src1**，该写法静默算出 `u²-1`，导致多项式自变量 `w∈[-1,0]`、L0 12/12 全失败 | `gelu_and_mul_fwd_kernel.py` v2 首版 L0 全 fail；repro `TRAP-vsub-operand-order.py` 实测 `vsub(a,1)=a-1` | design skill API 映射表增加硬性注记：`vsub` 语义 `src0-src1`、`src0` 不可为标量（`1-x` 须「取负再相加」）；`docs/…/T.vsub.md` §2.3 已有该限制，设计侧应引用 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **`T.vsub(src0,src1,dst)=src0-src1`，标量仅可在 `src1`；`T.vsub(w,1.0,w)` 得 `w-1` 而非 `1-w`**（静默符号错误，编译通过、精度全错）。`1-x` 须 `vmul(x,-1)` 后 `vadd(1)` | D | `examples/TileOPs/tileops/kernels/elementwise/gelu_and_mul/gelu_and_mul_kernel/gelu_and_mul_fwd_kernel.py`（v2 erf `w=1-u²` 落地）；`docs/Tilelang.language/数学操作/T.vsub.md` §2.3 | `python examples/gelu_and_mul/gelu_and_mul_fwd_kernel/repro/TRAP-vsub-operand-order.py`（自含断言，实跑通过） | tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B2C（实查 20 AI Core） | pattern-library/traps-compiler.md（API 语义类） |
| **TVM script `@T.prim_func` 体内不接受 Python 级 `for c in <常量tuple>`**，编译报「Expect the for loop to be one of range/T.serial/T.grid/T.parallel/T.vectorized/T.unroll/T.thread_binding」；Horner 等编译期系数链必须字面展开 | D/P | `gelu_and_mul_fwd_kernel.py` v2 首版编译报错 | `python gelu_and_mul_fwd_kernel.py --level L0`（未展开时立刻报该错） | tilelang 0.1.2 dev-root + CANN 8.5.2 | pattern-library/traps-compiler.md |
| **显式 clamp(±4)+deg-20 多项式 erf 在 NPU（v-prefix 链 + `vcmp/vselect`）真实跑通**：L0 fp16 max_abs_err 3.9e-3 / bf16 2.4e-4 / fp32 1.9e-6，L1 全过，Boundary 全过；`gelu(-inf)=NaN`/`gelu(+inf)=+inf`/NaN→NaN 与 golden 一致——v2 设计承接 v1 `T.verf` 缺陷的机制在硬件上成立 | D | `gelu_and_mul_fwd_kernel.py --level all` 输出；`DESIGN.md` §0.6 R2 / §3.3 | `python gelu_and_mul_fwd_kernel.py --level all` | tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B2C | pattern-library/elementwise.md |

### Transferable Lessons（可迁移教训）

- **prim_func 内只有 TIR 循环**：对编译期常量序列（Horner 系数、固定表）的循环必须**字面展开**；写 Python 级 `for c in tuple:` 会被 parser 拒绝并报 `DiagnosticError`（"Expect the for loop to be one of …"）。设计伪代码若用 Python `for` 表达多项式链，实现时必须展开。
- **`T.vsub` 是 `src0 - src1`，标量只能作 `src1`**：`T.vsub(w, 1.0, w)` 算的是 `w-1`。要算 `1 - x` 必须「取负再相加」（`vmul(x,-1)` → `vadd(...,1)`）；错误方向编译通过但结果静默错误，且难从 IR 看出。
- **多项式型 erf（及一切全域近似）的精度被外层逐步放大**：deg-20 的 `~3e-7` 绝对误差经 `0.5·|gate|·|value|` 放大；randn（O(1)）标定下无碍，但把 `|value|` 也放到 ~20 时 fp32 会突破 atol（本任务列为 L2 非阻塞）。写 Boundary/大 `|x|` 用例时，若目的是验证 erf 全域/饱和，应只放大 `gate`、保持 `value` O(1)，否则测的是 value 放大而非 erf 本身。
- **`gelu(-inf)=NaN` 的落地依赖多项式在 `|z|≥A` 精确饱和 `±1`**：须令常数项 `c0=1/A` 精确（本算例 0.25），使 `t·P(0)=±1`、`1+erf=0`、`gate·0=-inf·0=NaN`；若 `c0≠1/A` 则该边界退化为 `-inf`。本算子 Boundary 实测三 dtype 全对。

## Stage 2/3/4 复盘

（Stage 2/4 待追加）
