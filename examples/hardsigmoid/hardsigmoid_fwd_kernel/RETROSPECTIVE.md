# RETROSPECTIVE.md — hardsigmoid_fwd_kernel

> 追加式日志：每次 Stage 执行追加一个章节，不覆盖历史。schema 见 `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`。

## Stage 1 (design) Retrospective — first_design（migration-harness）— 2026-10-12

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | Ascend `vmax`/`vmin`（`tl.npuir_max/min`，`vclamp` 内部即 vmax+vmin）的 NaN 特殊值语义未在本地文档与 CANN 官方文档中明确，迫使迁移设计采用 `isnan`+`vselect` 单点恢复兜底（比纯 vclamp 多 2 个向量 op），且无法在设计期判定该兜底是否为冗余 | DESIGN.md §0.5/§0.6 R2/§9.1（已检索 CANN 9.0.0-beta1 Max/Min 文档「约束说明」仅载明地址对齐/重叠，无特殊值处理）；`/home/developer/tilelang-mlir-ascend/tilelang/language/customize_npuir.py` `npuir_max/min/clamp` 源码无 NaN 语义注释 | 在 pattern-library/traps-*.md 或 docs 增补「Ascend vmax/vmin NaN 传播行为」条目（Stage 3 实测后落 D 类常数），供后续 clamp/relu6/hardtanh/hardsigmoid/hardswish 类算子复用 | C |
| Info-source | 物理核数 `NPUUtils.get().get_aicore_num()` 本环境返回 20，与 pattern-library `CONST-aicore-910B2C`（记录 24）不一致——设备型号差异未被常数条目覆盖，设计须以实查为准并注明与文档条目的分歧 | DESIGN.md §5.5（实查 20 → 纯 Vector ×2 = 40；CONST-aicore-910B2C 记录 24） | constants.md `CONST-aicore-910B2C` 增补「本机实查 20 的其它设备型号」注释或拆分设备档位 | C |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| elementwise 激活 NaN 传播 clamp 的单点恢复化简（逐层 nan_min/nan_max → 1× isnan+vselect） | P | DESIGN.md §1.6.1 项 1（源 `_nan.py` 每次 min/max 各一次 isnan+select，因 NaN 只能源自输入 x，可化简为单点恢复） | `python examples/hardsigmoid/hardsigmoid_fwd_kernel/verify_equiv.py`（项 1：0 ulp / 0 违反 / EQUIV_PASS） | tilelang 0.1.2 dev-root（`/home/developer/tilelang-mlir-ascend/build`）+ Ascend 设备（实查 20 AI Core）+ CANN 8.5.0 | pattern-library/elementwise.md |
| sub-fp32 逐元素激活 fp32 中转链对齐 torch golden（vcast rint 升 fp32 → fp32 v-prefix 链 → vcast rint 回） | C | DESIGN.md §1.6.1/§3（复用 PL-1.17-subfp32-fp32-transit 两任务实证形态，本算子为第三个同族应用） | `examples/TileOPs/tileops/kernels/elementwise/mish/mish.py` 同款 golden 对齐 | tilelang 0.1.2 dev-root + Ascend + CANN 8.5.0 | pattern-library/elementwise.md（已存在 PL-1.17，本任务为其 known_impl 增量候选） |
| Ascend vmax/vmin NaN 传播行为未文档化（负面发现，待 Stage 3 实测标定） | C | DESIGN.md §9.1 风险点 + §0.6 R2（设计被迫采用 isnan+vselect 兜底，若硬件本身传播 NaN 则该兜底为冗余可省 2 op） | 待 Stage 3 Boundary（NaN 输入）实测判定 | tilelang 0.1.2 dev-root + Ascend + CANN 8.5.0 | pattern-library/traps-runtime.md（D 类，需 Stage 3 三件套后合入） |

### Transferable Lessons（可迁移教训）

- 迁移带 NaN 传播语义的逐元素 clamp/激活算子（hardsigmoid/hardswish/hardtanh/relu6 族）：CUDA 侧靠 `tl.max_nan`/`tl.min_nan`（fp16/bf16）或 `isnan+Select`（fp32）逐层恢复 NaN；NPU 侧 Ascend `vmax`/`vmin` NaN 语义未文档化，稳妥做法是普通 `vclamp` + 末尾 `isnan`+`vselect` 单点恢复（NaN 只能源自输入，一次检测即够，比逐层检测省 op）。设计期无法确认硬件是否原生传播 NaN，须列入风险点并让 Stage 3 的 NaN Boundary 用例实测标定。
- 物理核数实查（`NPUUtils.get().get_aicore_num()`）可能与本机 constants.md 记录的设备型号不一致，迁移设计一律以实查为准，纯 Vector 算子再 ×2，并在 §5.5 记录查询代码与返回值，勿照抄文档条目数值。
- `T.copy` 尾块用「双显式 slice」形态（`src[0:tail]`/`dst[0:tail]`）而非 base+size 标量基址形态（TRAP-T-copy-region-semantics）；`T.Tensor((N,), dtype)` 用位置参数传 dtype 避免 `name 'in_dtype' is not defined` false alarm。

## Stage 2 (review) Retrospective — first_review — 2026-10-12

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | `T.isnan` 是**标量谓词**（`isnan(x, span=None)`，入参 `PrimExpr`、第二参 `span` 非目标 buffer），`tilelang.language` 无任何向量 isnan（无 `visnan`/`npuir_isnan`）；Stage 1 设计把 `T.isnan(x_f32_ub, nan_mask)` 当作向量掩码 API 写入 §3.3/§1.6.2 并标「✅已验证」，仅凭 `dir(T) 含 isnan` 佐证——这是「凭记忆猜 API」的典型形态，而 Stage 1 的 RETROSPECTIVE 已讨论 isnan+vselect 兜底却未发现 isnan 无向量形式 | REVIEW.md 维度 1/8；DESIGN.md §3.4（`T.isnan` 行佐证仅 `dir(T) 含 isnan`）；`inspect.signature(T.isnan)` = `(x, span=None)`；全仓无向量 isnan 先例 | 在 api-quickref.md 或 docs/Tilelang.language/ 增补「`T.isnan` 仅标量谓词、无向量形式；NaN 掩码检测候选路径 = `T.vcmp(x,x,'ne')`（无序 != 待实测）或标量 isnan 兜底」条目，并提醒 design/review 维度 1/8 核对「标量谓词 vs 向量 API」不可只看符号存在 | C |
| Review-flow | `design_calc_check.py` 的 `check_ub_budget` 无法解析 §4.5 的「总计（fp16/bf16 路径）/总计（fp32 路径）」中文表头行（`TOTAL_MARK = 合计|总占用|total` 不匹配「总计」），导致 fp32 总计 155648≠172032 的混合字节口径错误被 report 为 `skip` 而非 `fail`，靠人工复核才发现 | DESIGN.md §4.5 fp32 总计行 vs 逐 buffer 求和差 16384（`y_ub` fp16/fp32 尺寸混淆）；design_calc_check.py 输出 `ub_budget: skip` | 扩展 `.agents/tools/design_calc_check.py` 的 `TOTAL_MARK` 正则加入「总计」（并考虑「（xxx 路径）」括号表头），使中文「总计」行可被识别并机械复算闭合性 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| TileLang 无向量 isnan：NaN 掩码检测须用 `T.vcmp(x,x,"ne")`（无序比较待实测）或标量 `T.isnan` 兜底，`T.isnan(x,dst)` 不是合法用法 | P | REVIEW.md 维度 1/8；`inspect.signature(T.isnan)`=`(x, span=None)`、`inspect.signature(T.vcmp)`=`(A,B,C,cmp_mod)`（cmp_mod 含 `"ne"`）；`_nan.py` 源实现仅标量 `T.isnan(T.Cast("float32", x))` | `python3 -c "import inspect,tilelang.language as T; print(inspect.signature(T.isnan)); print(inspect.signature(T.vcmp))"` | tilelang 0.1.2 dev-root（`/home/developer/tilelang-mlir-ascend/build`）+ Ascend 910B2C（实查 20 AI Core） | pattern-library/traps-compiler.md（或 docs/Tilelang.language/调试操作/） |

### Transferable Lessons（可迁移教训）

- 迁移带 NaN 传播语义的逐元素激活算子（hardsigmoid/hardswish/hardtanh/relu6 族）到 NPU 时，NaN 掩码**不能**用 `T.isnan(x, dst)` 向量形式（`T.isnan` 仅标量谓词，无向量 isnan）；向量 NaN 检测候选是 `T.vcmp(x, x, mask, "ne")`（`ne` 无序比较语义需 Stage 3 NaN Boundary 实测标定），否则回退标量 `T.isnan` 兜底。设计/检视核对 API 时须区分「符号存在」与「向量形式存在」，不可仅凭 `dir(T)` 判断。
- 检视内存预算表时，须独立复算「总计」行与逐 buffer 行是否闭合（尤其多 dtype 路径并列时，某 buffer 的 dtype 相关尺寸易在总计行张冠李戴）；`design_calc_check.py` 对中文「总计」表头尚不可解析，此类混合字节口径错误当前需人工逐行求和复核。

## Stage 1 (design) Retrospective — revision v1（migration-harness）— 2026-10-12

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 修订前未把「NaN 掩码必须用向量 API」落成可验证的 API 步骤，设计期凭 `dir(T)` 判断 `T.isnan` 向量可用（已被 Stage 2 抓出）；修订时须先 `inspect.signature` + 读 `customize_npuir.py` 源码 + grep examples 佐证，再落 §3.2/§3.3/§3.4/§1.6.2 | DESIGN.md §3.4（`T.isnan` 行弃用 + `T.vcmp`/`T.vbrc` 源码级佐证）、§3.3 L371-373（vbrc→vcmp→vselect）；`inspect.signature(T.vcmp)`=`(A,B,C,cmp_mod)`、`npuir_cmp` 断言 `cmp_mod ∈ {"eq","ne","lt","gt","ge","le"}` | api-quickref.md 增补「`T.isnan` 仅标量谓词（`(x, span=None)`）；向量 NaN 掩码用 `T.vcmp(x,x,"ne")`（无序比较待 Stage 3 标定）」 | P |
| Design-flow | NaN 广播 buffer 若作第 7 个独立 buffer 会使 fp32 驻留达 200KB > 192KB 溢出；正确做法是复用 vmul 后已死的 `buf2_ub`（c 值不再使用）作 vbrc 目标，不新增驻留 | DESIGN.md §4.3/§4.5（`nan_val` 复用 `buf2_ub`，0 新增驻留）、§3.3 L371 | design-template §4.5 增加「复用已死 buffer 承载广播/掩码常量，勿新增独立 buffer」提示 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 带 NaN 传播语义的逐元素激活（hardsigmoid/hardswish/relu6/hardtanh 族）NPU 单点恢复链：`vclamp`(vmax/vmin 丢 NaN) → `vcmp(x,x,"ne")`(IEEE 无序 NaN≠NaN→true，仅命中 NaN) → `vbrc(NaN, 复用死 buffer)` → `vselect` 回填 | P | DESIGN.md §0.6 R2 / §1.6.2 #5 / §3.3 L367-373；`npuir_cmp` 源码 `cmp_mod` 含 `"ne"`；`PL-1.11` P8 探针「vmax/vmin 钳位只治 ±Inf」 | `python examples/hardsigmoid/hardsigmoid_fwd_kernel/verify_equiv.py`（项 1：0 ulp / 0 违反 / EQUIV_PASS，数学层已验证；硬件 `"ne"` 无序语义待 Stage 3 NaN Boundary 标定） | tilelang 0.1.2 dev-root（`/home/developer/tilelang-mlir-ascend/build`）+ Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.0 | pattern-library/elementwise.md |
| `T.vselect` 的 A 参数必须为 buffer：NaN/哨兵常数经 `T.vbrc` 广播到「vmul 后已死」的中间 buffer 复用（buf2），避免第 7 个独立 buffer 把 fp32 驻留推到 200KB 溢出 | P | DESIGN.md §4.5（`nan_val` 复用 `buf2_ub`）；argmax `T.vbrc(T.float32(BIG), sent_v)` 先例 | `python3 -c "import inspect,tilelang.language as T; print(inspect.signature(T.vbrc))"`（`(src, dst)`） | tilelang 0.1.2 dev-root + Ascend 910B2C + CANN 8.5.0 | pattern-library/elementwise.md |
| sub-fp32 逐元素激活 fp32 中转的 fp32 路径 buffer 复用：`x_f32_ub`→`x_ub`（输入即 fp32 免 vcast）+ `y_ub`→`buf1_ub`（输出即 fp32 直接 copy 回 GM），168KB→104KB（−64KB） | C | DESIGN.md §4.5/§3.3 注 | none（实现期验证） | tilelang 0.1.2 dev-root + Ascend 910B2C + CANN 8.5.0 | pattern-library/elementwise.md |

### Transferable Lessons（可迁移教训）

- 带 NaN 传播语义的逐元素激活算子迁移到 NPU 的**落地 API 链**是 `vclamp`（`vmax`/`vmin` 不传播 NaN，PL-1.11 P8）→ `T.vcmp(x, x, mask, "ne")`（自比较，IEEE 无序 NaN≠NaN→true，±inf/有限位 false，掩码**仅**命中 NaN，不误伤 ±inf——区别于 `0*x` 算术掩码在 ±inf 位产生 NaN 的缺陷）→ `T.vbrc(NaN, buf2)`（NaN 常数广播到 vmul 后已死的中间 buffer，复用不新增驻留）→ `T.vselect(mask, buf2, buf1, buf1)`。`T.isnan` 是标量谓词（`(x, span=None)`），无向量形式，不可用于向量掩码。
- `T.vcmp` 输出掩码 buffer 的 dtype 是 **`bool`**（对齐 argmax `T.alloc_fragment(...,"bool")`、`_gqa_prefill_fwd_kernel.py` `T.alloc_ub(...,"bool")`），不是 `int8`。
- `T.vbrc` 的标量源须 let 绑定局部量（`nan_scalar = T.cast(float("nan"), "float32")`）再广播，字面量直接传入可能触发 rank 断言（TRAP-expert-v-operands）。
- 多 dtype 路径并列的 UB 预算表，逐 buffer 行的 dtype 相关尺寸（如 `y_ub` 的 fp16=16384 / fp32=32768）必须在「总计」行逐一核对闭合，否则出现 VP-2026-0009「混合字节口径」类算术错误。

## Stage 2 (review) Retrospective — second_review（migration-harness）— 2026-10-12

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | `T.vcmp(x,x,"ne")` 自比较检测 NaN 的**硬件无序语义**在 Stage 2 仍不可核验（需 Stage 3 实测），且源仓 `_nan.py` L22-25 注释「A self-compare is not a guard: TileLang lowers `!=` as an ordered compare」是**额外负面信号**——它证明同一「自比较检测 NaN」思路在 GPU TileLang 后端按有序比较 lower 会漏检 NaN，NPU `npuir_cmp "ne"` 是否按 IEEE 无序实现更加存疑。检视只能确认 API 存在（`npuir_cmp` 断言 `cmp_mod` 含 `"ne"`）+ 设计已如实标注风险与回退方案，无法给出「NaN 检测一定成立」的结论 | `_nan.py` L22-25；DESIGN.md §9.1/§1.6.2 #5/§0.6 R2 | constants.md/traps-*.md 增补「`npuir_cmp "ne"` 无序/有序语义」D 类常数（Stage 3 NaN Boundary 实测后合入），并把源仓「self-compare != 非 NaN guard」作为迁移检视的固定核对点 | C |
| Review-flow | `design_calc_check.py` 的 `core_split` 检查无法解析 §5.5 以散文/表格嵌入的分核方案（返回 `skip` + 误导性 `declared logical cores = 3`，实际 N=7680 时逻辑核数 = ceil(7680/8192)=1，§5.5 全文无「3」），分核算术只能靠人工独立复算 | design_calc_check.py 输出 `core_split: skip ("insufficient parsed dims {'N': 7680}; declared logical cores = 3")`；DESIGN.md §5.5 逻辑核数例（7680→1/30720→4/1048576→128/8388608→1024） | 为 `design_calc_check.py` 的 `core_split` 增加对 `num_cores = min(ceil(...), <phys>)` / `tiles_per_core = ceil(...)` 模式的解析，或在其 skip 时输出「须人工核对分核三要素」的明确提示（与 ub_budget「advisory checker; verify manually」同款措辞） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 迁移检视核对点：源算子若以「自比较 `!=`」实现 NaN 检测（如 `_nan.py`），须将其视为**后端相关语义**——GPU TileLang 按有序比较 lower（NaN≠NaN→false，漏检），NPU `npuir_cmp "ne"` 无序语义待实测；设计必须如实标注 Stage 3 探针风险 + 回退方案，不得视为已证 | P | `_nan.py` L22-25 注释（self-compare 非 guard、`!=` 有序 lower）；本 REVIEW 维度 1/8（API 存在已证、硬件语义待实测，设计已如实标注） | `sed -n '22,25p' /mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/_nan.py` | tilelang 0.1.2 dev-root + Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.0 | tilelang-design-review skill（维度 8 检视清单增补「NaN 自比较检测的后端相关语义核对」） |

### Transferable Lessons（可迁移教训）

- 检视「自比较 `vcmp(x,x,"ne")` 检测 NaN」类设计时，`T.vcmp`/`"ne"` 的 API 存在性（`npuir_cmp` 源码 `cmp_mod` 集合 + argmax/gqa examples）**不能**等同于「NaN 检测在 NPU 硬件上成立」——两者是不同层的结论：API 存在性由源码/example 佐证，硬件无序比较语义必须 Stage 3 实测标定。源仓 `_nan.py` 明言「self-compare != 非 NaN guard（有序 lower）」，是判断该语义风险等级的有力依据；设计如实标注风险 + 回退方案（标量 isnan 兜底）即为合格，不必因语义未证而判阻塞。
- `design_calc_check.py` 的 `core_split` 检查对散文嵌入的分核方案解析能力弱（返回 skip + 可能误导的「declared logical cores = N」伪影），分核三要素（逻辑核数/物理核数实查/规模判定与方案）当前必须人工按 core-split-strategy.md §1 独立复算，不可依赖其 skip 输出。

## Stage 3 (develop) Retrospective — first_impl（migration-harness）— 2026-10-12

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 设计 §0.6 R2/§3.3 推荐的 NaN 广播常数 `T.cast(float("nan"), "float32")` 会导致 tilelang kernel cache 每次构建都报 `Error loading metadata: Wrong value format for field value`（`FloatImmNode::VisitAttrs` → `ParseDouble`，JSON 无法表示 NaN）→ 缓存永远 miss、每次运行重编译（~1–2s/次）。设计/skill 均未提示「NaN 常数破坏缓存元数据序列化」这一副作用；正确性不受影响（kernel 仍编译运行正确），但属可避免的重复编译开销 | 本任务 `--level all` 输出（3 个 L0 dtype 各报一次）；`repro/CONST-vcmp-ne-nan-unordered.py` 头注（NaN vs `T.cast(1.0,"float32")` 有限常数对照：仅 NaN 触发）；`customize_npuir.py` `npuir_brc` 接受 `PrimExpr` 标量 | pattern-library/traps-compiler.md 增补「`T.cast(float("nan"),...)` 常数破坏 kernel cache JSON 元数据序列化（NaN 非 JSON 值），可选绕法 = 算术构造 NaN 或接受重编译」D 类条目（含本任务 repro） | D |
| Develop-flow | op_template.py 未覆盖「persistent 分核 + fp32 中转 buffer 复用」组合形态，实现时先尝试在 `@T.prim_func` 内做 Python buffer 别名 `x_f32 = x_ub`，触发 TVMScript parser `Undefined variable: x_f32`；正确形态是 mish.py 的「全量分配 + 同 dtype `T.copy`」或 lerp_tensor 的「两分支 factory」 | 本任务首次实现 stderr `Undefined variable: x_f32`（line 145）；mish.py L70-72（`if dtype != compute_dtype: vcast else: T.copy`）；lerp_tensor `_make_lerp_tensor_kernel.py` L189（`if use_fp32_transit:` 两分支 factory） | op_template.py 增补「fp32 中转 buffer 复用」示例 + 提示「`T.prim_func` 内禁止 Python buffer 别名赋值（`a = b` 不被 TVMScript parser 识别），用全量分配+`T.copy` 或两分支 factory」 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| Ascend `npuir_cmp "ne"` 自比较按 IEEE **无序**实现（NaN≠NaN→true、±inf/有限→false），`vcmp(x,x,"ne")`+`vselect` 单点恢复 NaN 在 NPU 实测成立（float32/fp16/bfloat16 三 dtype NaN→NaN 全过）——**正面证立** Stage 1/2 的风险点（源仓 `_nan.py`「self-compare != 非 NaN guard（有序 lower）」是 GPU 后端语义，NPU 不适用） | D | 本任务 Boundary 用例（`[Boundary] PASS: NaN->NaN (R2 vcmp(ne)+vselect verified)` ×3 dtype，nan_count=512/512）；`repro/CONST-vcmp-ne-nan-unordered.py` 头注 | `python examples/hardsigmoid/hardsigmoid_fwd_kernel/repro/CONST-vcmp-ne-nan-unordered.py`（三 dtype assert_close equal_nan + NaN mask 相等） | tilelang 0.1.2 dev-root（`/home/developer/tilelang-mlir-ascend/build`）+ Ascend 910B2C（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/traps-runtime.md（或 constants.md D 类，正面确认 `npuir_cmp "ne"` 无序语义，供 hardswish/hardtanh/relu6 族复用） |
| `T.cast(float("nan"), "float32")` NaN 常数破坏 kernel cache JSON 元数据序列化（NaN 非 JSON 值 → `FloatImmNode::VisitAttrs`→`ParseDouble` 报 `Wrong value format`，每次构建 cache miss 重编译）；有限 FloatImm 常数不受影响；正确性不受影响 | D | 本任务 `--level all` 每次构建报 `Error loading metadata`；`/tmp/probe_nan_cache2.py`（NaN vs `T.cast(1.0,"float32")` 对照：仅 NaN 报错）；`repro/CONST-vcmp-ne-nan-unordered.py` 头注 | `python examples/hardsigmoid/hardsigmoid_fwd_kernel/repro/CONST-vcmp-ne-nan-unordered.py`（运行观察 stderr `Error loading metadata`） | tilelang 0.1.2 dev-root + Ascend 910B2C + CANN 8.5.2 | pattern-library/traps-compiler.md |
| `T.prim_func` 内禁止 Python buffer 别名赋值（`x_f32 = x_ub` 触发 TVMScript `Undefined variable`）；fp32 中转 buffer 复用正确形态 = 全量分配 + 同 dtype `T.copy`（mish.py）或两分支 factory（lerp_tensor） | P | 本任务首次实现 `Undefined variable: x_f32`；mish.py L70-72 / lerp_tensor L189 佐证 | `python examples/hardsigmoid/hardsigmoid_fwd_kernel/hardsigmoid_fwd_kernel.py --level L0`（当前正确形态通过） | tilelang 0.1.2 dev-root + Ascend 910B2C + CANN 8.5.2 | pattern-library/traps-compiler.md（或 TRAP-tvm-parser-rules 增量） |

### Transferable Lessons（可迁移教训）

- 迁移带 NaN 传播语义的逐元素激活（hardsigmoid/hardswish/hardtanh/relu6 族）到 NPU：`vcmp(x, x, mask, "ne")` 自比较在 NPU 上按 IEEE **无序**实现（NaN≠NaN→true），因此 `vclamp`(vmax/vmin 丢 NaN) → `vcmp(x,x,"ne")` → `vbrc(NaN 常数, 复用死 buffer)` → `vselect` 单点恢复链是**已验证可用**的正确方案，无需回退标量 `T.isnan` 兜底。源仓 `_nan.py` 的「self-compare != 非 NaN guard」警告仅针对 GPU TileLang 后端，不可据此否决 NPU 的 `npuir_cmp "ne"` 方案——但该结论绑定工具链版本戳，工具链升级后须按 repro 重验。
- `T.prim_func` 体内做 Python 变量别名（`a = b`，把 buffer 句柄 rebind 到新名字）不被 TVMScript parser 识别，会报 `Undefined variable: a`；同 dtype 中转用 `T.copy` 显式搬移、或把 fp32-transit / native 分派提到 factory 层做两分支（lerp_tensor 形态），不要用 Python 别名省 buffer。
- 用 `T.cast(float("nan"), "float32")` 构造 NaN 广播常数会使 tilelang kernel cache 每次构建报 `Error loading metadata`（NaN 非 JSON 值、缓存元数据序列化失败）并强制重编译；若在意重复编译开销，可考虑用算术方式构造 NaN 常数（或接受该副作用）——正确性不受影响。

## Stage 1 (design) Retrospective — first_design 重设计（migration-harness）— 2026-10-13

> 本段为同一算子 `hardsigmoid_fwd_kernel` 的 **first_design 重跑**（新会话）复盘。产出：DESIGN.md（first_design 基线，无 revision 段）+ verify_equiv.py（已执行，2 项 EQUIV_PASS）+ 本节。与上文 2026-10-12 各段的关键差异：本轮**消费知识库已固化的实测结论**（`CONST-vcmp-ne-nan-unordered`，origin_task = 本算子 Stage 3），把上轮设计期标注为「待 Stage 3 实测」的 `vcmp(x,x,"ne")` 无序语义风险**降级为已验证**。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 设计期若只读 references/standards 而不消费 pattern-library 已固化条目，会把已验证结论当作未知风险重复兜底：上轮 design 对 `vcmp(x,x,"ne")` NaN 检测标「待 Stage 3 探针标定」，而 `constants.md` `CONST-vcmp-ne-nan-unordered`（`tilelang 0.1.2 dev-root + Ascend 910B2C（实查 20 核）+ CANN 8.5.2 / 2026-10-12`）已完成三 dtype 实测 | 本轮 DESIGN.md §0.6 R2 / §1.6.2 #5 / §9.1 直接引用 `CONST-vcmp-ne-nan-unordered`（verified）；`python3 .agents/tools/kb_search.py "hardsigmoid elementwise npuir min max"` 首位命中该条目 | 设计 Phase R / Phase 3 步骤 0.5 的固定动作须强调「同一 op_slug 的历史实测条目优先消费」——`kb_search.py "<op_slug>"` 命中 `CONST-*`/`CASE-*` 时以库内结论替代设计期假设 | C |
| Info-source | 迁移设计所需的 cross-repo 文档引用与 gate 的仓库路径存在性校验冲突：`core-split-strategy.md` 权威依据写 `docs/开发指南.md §3.3`，但 `docs/` 不在 npuir-migration 仓库根（实际在 tilelang 源码仓），照抄进 DESIGN.md 会被 `S1-REF-PATHS` 判失败 | 首次 `statectl gate 1` 失败：`引用的仓库路径不存在：docs/开发指南.md`；DESIGN.md §5.5 改为「该文件依据昇腾开发指南 §3.3」后 pass | 设计/检视须区分「仓库内可校验路径」与「跨仓/官方文档名称」：前者用 `docs/`/`examples/`/`testing/`/`.agents/` 前缀，后者用文档名（勿加 `docs/` 前缀），gate 只校验前者 | R |
| Info-source | 本仓 `docs/Tilelang.language/` 不存在（在 tilelang 源码仓 `/home/developer/tilelang-mlir-ascend/docs/`），故 API 佐证改用「官方 OP 文档名（Tilelang.language/数学操作/T.vadd.md 等）+ 本地 `customize_npuir.py` 源码 + `.agents/skills/tilelang-vector-skill/references/api-quickref.md`」三源交叉，既满足 docs 佐证又不触发 gate 路径校验 | DESIGN.md §3.4 逐 API 列 tilelang OP 文档名 + `customize_npuir.py` 行号 + api-quickref.md；gate pass | 设计规范对 `docs/Tilelang.language/` 的引用可显式说明「文档名形式（非本仓相对路径）」，避免设计者在仓库根找不到该目录时误判无证 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 带 NaN 传播语义逐元素 clamp/激活（hardsigmoid/hardswish/relu6/hardtanh 族）的 NPU 落地链：`vclamp`(vmax/vmin 丢 NaN) → `vcmp(x,x,"ne")`（IEEE 无序 NaN≠NaN→true，掩码仅命中 NaN，不误伤 ±inf）→ `vbrc(NaN, 复用已死 buffer)` → `vselect` 回填——**已实测证立** | D | DESIGN.md §0.6 R2 / §1.6.2 #5 / §9.1；`CONST-vcmp-ne-nan-unordered`（三 dtype NaN→NaN 全过） | `examples/hardsigmoid/hardsigmoid_fwd_kernel/repro/CONST-vcmp-ne-nan-unordered.py`；`python examples/hardsigmoid/hardsigmoid_fwd_kernel/verify_equiv.py`（数学层项 1：0 ulp / 0 违反） | tilelang 0.1.2 dev-root（/home/developer/tilelang-mlir-ascend/build）+ Ascend 910B2C（实查 20 核）+ CANN 8.5.2 | pattern-library/elementwise.md（known_impl 增量） |
| NaN 传播逐层检测的**单点恢复化简**（源 `_nan.py` 每层 min/max 各一次 isnan+select → 1 次检测 + 1 次 select）：NaN 唯一源自输入 x，数学恒等，省向量 op | P | DESIGN.md §1.6.1 项 1（0 ulp / 0 违反 / EQUIV_PASS）；源 `_nan.py` `_propagate_nan` | `python examples/hardsigmoid/hardsigmoid_fwd_kernel/verify_equiv.py` | tilelang 0.1.2 dev-root + Ascend 910B2C + CANN 8.5.0/8.5.2 | pattern-library/elementwise.md（`PL-1.17` 同族） |
| bf16 必须 fp32 中转：`vadd/vclamp/vmul/vsub` 官方 dtype 表 bf16 标 ×（仅 fp16/fp32 ✓），故 bf16 输入经 `vcast(bf16→fp32, rint)` 计算再 `vcast` 回，兼对齐 torch fp32 opmath golden | C | DESIGN.md §0.5/§3.1/§9.3；tilelang OP 文档 dtype 表 + `PL-1.17-subfp32-fp32-transit` | `examples/TileOps` 同族 mish fp32 中转形态 | tilelang 0.1.2 dev-root + Ascend 910B2C + CANN 8.5.0 | pattern-library/elementwise.md（`PL-1.17` 增补 dtype 依据） |
| `T.cast(float("nan"), "float32")` NaN 常数破坏 kernel cache JSON 元数据序列化（每次构建 cache miss 重编译）；正确性不受影响 | D | DESIGN.md §9.1/§3.3；本仓 2026-10-12 同类实测（`Error loading metadata`） | 见本目录 2026-10-12 Stage 3 段与 `repro/CONST-vcmp-ne-nan-unordered.py` 头注 | tilelang 0.1.2 dev-root + Ascend 910B2C + CANN 8.5.2 | pattern-library/traps-compiler.md |

### Transferable Lessons（可迁移教训）

- **设计重跑第一步应先 `kb_search.py "<op_slug>"`**：同一算子/同族的 Stage 3/4 实测结论（`CONST-*`/`CASE-*`）会回灌知识库；若设计期不消费，会把「已实测证立」的结论重新标成「待探针」，重复兜底并抬高风险等级。本轮据此把 `vcmp(x,x,"ne")` 无序语义由「待 Stage 3 标定」降为「已验证」。
- **gate 的 `S1-REF-PATHS` 只认仓库内相对路径**：设计正文里 `docs/`/`examples/`/`testing/`/`.agents/` 前缀的 token 会被逐一校验存在性（相对 `repo_root`）。跨仓/官方文档（如昇腾开发指南、`docs/Tilelang.language/`）不能照抄其 `docs/` 前缀，须改用文档名或「文档名形式」表述，否则门禁失败。
- **bf16 逐元素 v-prefix 链必须 fp32 中转**：tilelang 官方 OP 文档的 dtype 支持表对 `vadd/vmul/vsub/vclamp/vmax/vmin` 的 bf16 列标 ×，设计 bf16 路径时不可假设原生 bf16 向量运算；`T.vcast(bf16→fp32, rint)` → fp32 链 → `T.vcast(fp32→bf16)` 是唯一可行形态，且顺带对齐 torch fp32 opmath golden（`PL-1.17`）。

