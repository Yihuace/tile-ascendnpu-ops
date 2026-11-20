# RETROSPECTIVE.md — `examples/engram_decode/_engram_decode_kernel/`

> 本文件按 `tilelang-skill-evolution` skill 的 `references/retrospective-schema.md` 追加式记录各 Stage 复盘。Stage 1 章节由 `tilelang-op-designer` 在 DESIGN.md 产出后写入。

## Stage 1 (design) Retrospective — first_design — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | `tilelang` 顶层 import 在本环境失败（`No module named 'tvm'`：`3rdparty/tvm/python` 缺失），导致 pattern-library 中**有 repro 的条目无法在设计期重验**（`repro_runner.py` 需要 tilelang 编译）；同时 `kb_stale_check.py` 报大量 `stale`（工具链 stamp 解析为 unknown）——设计期只能按「待重验」引用并显式标注，无法把已实测常数当作已验证前提 | `examples/engram_decode/_engram_decode_kernel/DESIGN.md#1.6.0`（D-2 估算不确定性声明）；环境：`python3 -c "import tilelang"` → `ModuleNotFoundError: No module named 'tvm'`；`kb_stale_check.py` 输出 | `tilelang-op-design/SKILL.md` Phase 3 步骤 0.5 增补：环境不满足 repro 运行条件时的降级记录格式（已在 `algorithm-research.md` §6 有降级条款，可复用） | R |
| Info-source | `pattern-library/constants.md` **无 `Ascend910B3` 设备族条目**（仅 910B2C / 910_9362），而本任务实查设备为 `910B3`（20 AICore / 40 Vector Core）——设计期 roofline 的分母（带宽/发射/端口）只能跨 SoC 外推并标注「未文档化假设」 | `DESIGN.md#1.6.0`（引用 `CONST-ascend910-9362-profile` / `CONST-vector-launch-overhead` / `CONST-L1-port-bw` 均标注跨设备未重验）；实查 `NPUUtils.get().get_aicore_num()==20`、`get_arch()=="Ascend910B3"` | `tilelang-op-optimize/references/pattern-library/constants.md` 增加 910B3 设备画像条目（或拓扑等价于 910_9362 单 die 的说明）；`CONST-aicore-*` 条目补 910B3 | D |
| Constraint-doc | `docs/Tilelang.language/` 的 v-prefix 数学 op **bf16 列全线为 ✗**（vadd/vmul/vdiv/vexp/vsigmoid/vsqrt/vrsqrt/vmin），仅 `vbrc`/`vcast`/`T.gemm` 支持 bf16——这使任何含 bf16 workload 的算子被**隐性强制** fp32 中转；AGENTS.md / skill 的「v-prefix API 优先」指引未提示该 dtype 约束 | §0.5/§0.6 R2、§1.6.0「v-prefix 元算 dtype 矩阵」；逐 op doc §2.2.1 | `tilelang-vector-skill` 与 `_shared/standards` 增补「bf16 v-op 缺失 → fp32 中转」为硬约束条；`tilelang-op-design` Phase 2 检查表补 bf16 门 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| Ascend910B3 实查物理核数 = 20 AI Core / 40 Vector Core（区别于 910B2C 的 24） | D | `DESIGN.md#5.5`（实查记录）；`NPUUtils.get().get_aicore_num()` 返回 20、`get_aivector_core_num()` 返回 40、`get_arch()` 返回 `Ascend910B3` | `python3 -c "from tilelang.utils.npu_utils import NPUUtils; print(NPUUtils.get().get_aicore_num(), NPUUtils.get().get_aivector_core_num(), NPUUtils.get().get_arch())"`（需 tilelang 可用；本会话经 stub 加载 `tilelang.env` 后实查） | tilelang dev root build `/mnt/workspace/Agentkj/build`；Ascend910B3；npu-smi 25.5.0；2026-10-14 | `pattern-library/constants.md`（新增 `CONST-aicore-910B3`） |
| GEMV 的「d_mem 串行向量 FMA 链」是 NPU 迁移共性性能陷阱：op 数 = O(d_mem)（W3 两投影 2048 op）、循环携带依赖链，发射项下界 ~1ms；MAC 密集段应改用 Cube `T.gemm`（M=输出维/N=1 合同，`a_transpose` 适配 `(K,N)` 权重布局） | P | `DESIGN.md#0.5`（条目 #10）、`#0.6 R1`、`#1.6.0 R4`（GEMV = FLOPs 99.4%/流量 88%）；基线 `examples/engram/engram_decode.py`（串行 GEMV）vs 主选 Cube | `examples/gemv/example_gemv.py::naive_gemv_high_perf`（Developer `is_npu=True` + `T.gemm` GEMV 实跑先例） | 引用条目工具链见 `DESIGN.md#1.6.0`（910B2C/9362，待重验） | `pattern-library/layout.md` 或 `traps-runtime.md`（新增「GEMV 串行链 → Cube」模式/陷阱条） |
| 「多 reduce 即冗余」需按数据核对：4 次 `reduce_sum` 作用于 **h/k/vhat 三个不同向量**时非冗余（`BP_redundant_work_or_roundtrip` 的触发信号须补「同一数据」限定） | P | `DESIGN.md#1.6.0 R2`（结构依据）、`#1.6.2 #1`；`BP_redundant_work_or_roundtrip`（status=open）"多个 reduce 或多 pass 遍历同一批数据" | none（结构分析，非实测） | — | `tilelang-op-optimize/references/bottleneck-patterns.md`（补充触发信号的「同一批数据」限定词） |
| `T.copy` 跨 dtype 隐式 cast（`TRAP-C12-copy-dtype-cast`）在设计期须显式转成「`T.copy` 仅同 dtype + 显式 `T.vcast`」的纪律条——本算子多处需要 fp16/bf16↔fp32 转换（sigmoid 前置/状态 append/权重升 fp32） | C | `DESIGN.md#3.2` 注、`#3.5.1`、`#9 R-11`；`TRAP-C12-copy-dtype-cast`（status=verified） | none | 引用条目工具链（截至 2026-08-28 build，待重验） | `pattern-library/traps-compiler.md`（TRAP-C12 正文补「设计期纪律条」用法示例） |

### Transferable Lessons（可迁移教训）

- **bf16 workload 的 NPU 向量链一律用 fp32 中转**：`vadd/vmul/vdiv/vexp/vsigmoid/vsqrt/vrsqrt/vmin` 的 dtype 矩阵均不含 bf16（仅 `vbrc`/`vcast`/`T.gemm` 支持 bf16）——凡契约含 bf16 的算子，设计期即把 `vcast(rint)` 升 fp32 + 末端单次舍回写成默认路径，不要等到编译失败。
- **迁移含 GEMV/矩阵乘的算子时，先算「向量实现的 op 数」**：`T.serial(K)×T.Parallel(N)` 形式的向量乘加链 op 数 = K，在 NPU 上由发射项（~0.5µs/op，`CONST-vector-launch-overhead`）主导；K≥256 时即应改用 Cube `T.gemm`（N=1 的 GEMV 形态已被 `examples/gemv` 实证），op 数降至 K/BK。
- **`T.vrsqrt` 不可用于容差 <1e-2 的归一化族**：设备 `T.vrsqrt` 是近似指令（`TRAP-vrsqrt-plain-precision`，~2.9e-3 rel）；用 `T.vsqrt` + `T.vdiv`（`sqrt(s)/s ≡ 1/sqrt(s)`）替代，两 op 全精度（实测 ~1.07e-7）。含多次 RMSNorm/LayerNorm 的算子尤须注意误差累积。
- **conv/状态的编译期索引必须 host 层展开**：`prim_func` 内 `for p in range()` 会被解析为运行时 Var（`TRAP-C13`），致 tap 行索引成动态 slice 而非法；tap 数小的窗口/卷积算子（如 w≤4）在工厂函数体内 Python 展开或走 `@T.macro` 链。
- **`grid=batch` 的 decode 算子在 NPU 上核利用率天然低**（B≤8 vs 20 核）：这是 batch 维小的固有特征；若要提高利用率需 split-N/split-K + 跨核归约（结构级改动），应在设计中显式记录为备选而非默认。

## Stage 2 (review) Retrospective — first_review — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | `design_calc_check.py` 的 `l0c_budget` 解析器把 §5「Tiling」段内 **L1 tile 串 `128×512`**（`(BK,d_padded)` fp16 = 128KB）误当作 `block_M × block_N`，给出 `l0c_budget: fail`（256KB > 128KB）**假阳性**；真实 L0C 为 `c_frag (d_padded,1) f32`=2KB。skill 规则「l0c_budget fail → 维度 2 fail」若机械执行会误判。 | `design_calc_check.py --design examples/engram_decode/_engram_decode_kernel/DESIGN.md` → `{"check":"l0c_budget","status":"fail",...block 128x512...}`；DESIGN.md §5.3 L881（`(BK,d_padded)` fp16 = 128×512×2=128KB）为误配来源；DESIGN.md §4.5/§5.3 的 L0C=2KB 正确 | `.agents/tools/design_calc_check.py` 的 `check_l0c_budget` 收紧正则：仅在 Cube 累加器上下文（`alloc_fragment` + `T.gemm` 同段）取 `M×N`，避开 `(BK,d_padded)`/L1 显存 tile；`tilelang-design-review/SKILL.md`「机械复核」段补「`l0c_budget fail` 须先核对是否解析到 L1 tile，再定维度 2」 | R |
| Review-flow | dim-8 弃选前提核对（`negative-claim-evidence.md` §3）要求「逐格核对 dtype 矩阵」，但设计文档常以「章节号 + 结论」形式引用（如「dtype 矩阵 bf16 ×」）——本次即命中该形态（设计称 `T.transpose` bf16 ✗，实为 √，`algorithm-candidates.md` ALG-transpose 已记 VP-2026-0168 更正）。现有 skill 无机械化触发点，纯靠 reviewer 逐条开文档。 | `DESIGN.md#1.6.3` 候选 C 行（"仅二轴交换、dtype 矩阵 bf16 ×"）vs `docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1 Ascend 行 bf16=√ | `tilelang-design-review/SKILL.md` dim-8「弃选论证前提核对」行增补操作条：对每个含 dtype 断言的弃选行，用 grep 打开所引 doc 的 §2.2.1 表头+Ascend 行逐格取值（禁止只信设计转述） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `T.transpose` Ascend dtype 矩阵 **bf16=√**（支持），且 `size` 参数当前不可用——设计期「bf16 ×」为引章节号未逐格取值形态（复证 ALG-transpose VP-2026-0168） | C | `docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1 Ascend 行（bf16=√）、§2.1（`size` 传参触发 IndexError）；本任务 DESIGN.md#1.6.3 误用「bf16 ×」 | none（文档逐格核对） | — | `tilelang-op-design/references/algorithm-candidates.md` ALG-transpose（已有 VP-2026-0168，补一条「reviewer 侧已复证」指针） |

### Transferable Lessons（可迁移教训）

- **引用 API dtype 矩阵时逐格取值，不要引用章节号 + 转述结论**：`docs/Tilelang.language/*/T.*.md` 的 §2.2.1 表列次序并非总是「…fp16/fp32/bf16」，转述极易错位（本任务 `T.transpose` 被误写 bf16 ×，实为 √；同类更正见 `algorithm-candidates.md` ALG-transpose VP-2026-0168）。核对时打开文档、对齐表头列逐一取值。
- **备选方案（实验裁决用）的 UB 预算必须计「循环内同时存活」的全部 tile**：outer-product/factorize 形态常同时需要两个同形大 buffer（如 `prod_ub` 与 `e_brc` 各 64KB@BK=32,d=512），只计一个会漏算数倍；备选须按同形 tile 数 × 尺寸 × 膨胀系数核算，否则裁决选中后 Stage 4 编译期 UB 溢出。
- **机械检查的 `fail` 先核解析来源再判维度**：`design_calc_check.py` 的 `l0c_budget` 会把 Tiling 段里的 L1 tile 串（`128×512`）误当 `block_M×block_N`；reviewer 须先定位被解析的字符串，确认是 Cube 累加器 shape 还是显存 tile，再决定维度 2 的判定。

## Stage 1 (design) Retrospective — revision（revision_index=1）— 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | Stage 2 检视指出的 2 阻塞 + 3 建议**全部集中在 §1.6.3 及计数口径**，而 `tilelang-op-design` skill 的 Phase 2 检查表未要求设计者**在写弃选行时逐格取值所引 dtype 矩阵**、也未要求**备选方案的 UB 预算列全所有同存活 buffer**——两处恰是本次唯二阻塞缺陷，说明 skill 缺机械化自检点（首次设计时无 reviewer 兜底即会漏） | `REVIEW.md` 问题 1/2（阻塞）；`DESIGN.md#1.6.3` 候选 C/B 行；`docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1（bf16=√） | `tilelang-op-design/SKILL.md` Phase 2 增补两条自检：① 弃选行含 dtype 断言时逐格读所引 doc §2.2.1；② 备选/实验裁决方案的 §UB 预算须「同存活 buffer 全列 + 膨胀系数」；并在 Phase 8 复盘模板加「弃选前提核对」项 | R |
| Design-flow | 设计文档内**同一量（f32 buffer 数）在 §3.3/§4.3 与 §4.5 口径不一致**（17 vs ~11），且 fragment 计数（§0.4 #2 "×11" vs 源码 14）为纯事实计数错——正文（伪代码/内存表）与摘要表（预算表/优化手段表）之间缺一致性核对步骤 | `REVIEW.md` 问题 3/4（建议）；`DESIGN.md#4.5` vs `#3.3/#4.3` | `tilelang-op-design/SKILL.md` Phase 2/4 增补「摘要表计数须回溯正文声明同源核对」自检条（buffer 计数、fragment 计数） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 备选（实验裁决）方案的 UB 预算须**列全同存活 tile**：outer-product 型向量 GEMV 的 `prod_ub` + `e_brc` 是两张同形 `(BK,d_padded)` f32 tile，只计一张会漏算 2×；本次修订将 BK 32→8 + `e_brc` 原地复用（`T.vmul(w_f,e_brc,e_brc)`）后 raw ≈103KB/×1.7≈175KB 方可编译 | C | `REVIEW.md` 问题 2；`DESIGN.md#1.6.3`（v1 重算）；`neg-claim-evidence.md` §3 | none（解析式预算；建议 Stage 4 编译探针复证） | 引用常数绑 910B2C（待重验） | `tilelang-op-design/references/algorithm-candidates.md` ALG-gemv（补「向量分块的 UB 双 tile 约束 + BK 上界公式」） |
| 弃选行 dtype 断言必须逐格取值：`T.transpose` Ascend 矩阵 bf16=√（非 ×）；该错误在 v0 阻塞检视、v1 修正，**同一形态（引章节号未逐格取值）在 ALG-transpose/VP-2026-0168 已有记载**，本次为 design 侧复现 | C | `docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1；`DESIGN.md#1.6.3` v1 修正 | none | — | `tilelang-op-design/references/algorithm-candidates.md` ALG-transpose（补「designer 侧自查点」） |

### Transferable Lessons（可迁移教训）

- **写弃选候选行时，凡含 API dtype 断言必先逐格读所引文档 §2.2.1 表头+Ascend 行**：`T.transpose` 的 Ascend 行是 `int8/int16/int32/int64/fp16/fp32/bf16 = √`，转述为「bf16 ×」会被 Stage 2 维度 8 判为「负向断言与文档矛盾」（阻塞）。等价地，弃选理由应尽量落在**结构与收益**（如「主选已由 load 侧 `a_transpose` 零 repack」）而非 dtype 可用性。
- **实验裁决备选方案的 UB 预算必须列全「循环内同时存活」的全部 buffer**：向量分块 GEMV 至少同时需要 `w`(f32 升) + `e_brc`(f32 广播) 两张 `(BK,d_padded)` f32 tile + 累加器，只计一张会漏算 2×；BK 上界由 `2·BK·d_padded·4B·1.7 + 共享向量链 ≤ 192KB` 反解（本任务 d=512 时 BK≤8 且需原地复用）。
- **贴限预算附编译探针计划**：手工逐 buffer 预算 ≠ 可编译证明（VP-2026-0059），×1.7 膨胀后占用 >80% 时上靶前先 `tilelang.compile` 读 `ub overflow, requires N bits` 取精确实测需求。
- **摘要表的计数须回溯正文声明同源**：§4.5 预算表的 buffer 数、§0.4 优化手段表的 fragment 数应与 §3.3/§4.3 及源码一致；revision 前用 grep 交叉核对所有「×N」「~N ×」计数。

## Stage 2 (review) Retrospective — revision（revision_index=1）— 2026-10-14

> 复盘对象为**本轮检视流程自身**（v1 重新检视），非设计内容；设计问题已随 REVIEW.md 流转 Stage 1。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | `design_calc_check.py` 的 `l0c_budget` **假阳性归因**在上一轮 Retrospective 中被记为「§5.3 的 L1 tile `128×512` 被误当 `block_M×block_N`」，与实测不符：本 Agent 用 `design_calc_check` 内部函数（`_section`/`check_l0c_budget` 的正则）复现，实际首个命中的是**§1.6.3 候选 C 行（L470）的写实串 `1024×512 → 多 tile 链 … fp16/fp32/bf16`**（`.*?fp32` 命中「fp16/fp32/bf16」），把**权重矩阵 shape `d_mem×d`** 误当 block tile（§5.3 的 `128×512` 甚至不在首个匹配位）。假阳性在 v1 **仍在**。 | `python3 .agents/tools/design_calc_check.py --design examples/engram_decode/_engram_decode_kernel/DESIGN.md` → `l0c_budget: fail, block 1024x512`；复现脚本定位 `m.start()` 落在 L470 | `.agents/tools/design_calc_check.py` 的 `check_l0c_budget`：回退正则不应全文匹配裸 `<num>x<num>`（会命中权重/张量 shape 的写实串），应限定在 `T.alloc_fragment`/`T.gemm` 同段且排除 `(BK,d_padded)` 类 tile；`tilelang-design-review/SKILL.md` 机械复核段补「`fail` 须先用工具内部函数定位被解析字符串的实际行号，再判定维度」 | R |
| Review-flow | 上一轮 REVIEW 的「计数口径漂移」意见只点名 §4.5/§3.3，v1 据此只同步了 §4.5/§3.3/§5.3/§9 R-8，**漏掉 §1.6.3 主选 A 行（L468）与 §1.6.0 R3 #1 行（L348）的 `~10×(d,)f32`**——同量（f32 buffer 数）在整个文档有 4+ 处摘要出现，只改点名处必然残留。说明 review 侧提「口径不一致」时未给出**全量出现位置清单**。 | REVIEW.md（本轮）§7「非阻塞计数残留」；`grep -nE "~10×|10×\(d|17 个|17×" DESIGN.md` 命中 L348/L468 vs L857/L861/L914/L1148 | `tilelang-design-review/SKILL.md` 维度 7/8：提「计数口径漂移」类意见时，须附 `grep` 出的**同量全部出现位置**，供 Stage 1 一次改全；Skill Flow 输出增加「意见须可一次闭合」检查 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `design_calc_check.py` `l0c_budget` 假阳性的**精确触发形态**：全文回退正则 `\b(\d{2,4})\s*[x×*]\s*(\d{2,4})\b.*?(?:L0C\|累加\|accum\|fp32)` 会命中任意「写实矩阵 shape（如权重 `1024×512`）+ 同行出现 `fp32`」的散文行，与 Cube 累加器 tile 无关——含 `T.gemm` 的 GEMV/逐元素混排设计极易误报 | R | 本轮复现：`m.start()` 落 DESIGN.md L470（§1.6.3 候选 C 行）；真实 L0C=`c_frag (d_padded,1) f32`=2KB | `python3 .agents/tools/design_calc_check.py --design examples/engram_decode/_engram_decode_kernel/DESIGN.md`（输出 `block 1024x512`） | 工具无运行时依赖（纯正则）；环境 tilelang 不可 import（`No module named 'tvm'`） | `.agents/tools/design_calc_check.py`（`check_l0c_budget` 收紧回退正则 + 打印命中行号）；`tilelang-design-review/SKILL.md` 机械复核段 |
| 迁移设计文档「同量多处摘要」一致性核对：单次 review 意见若只点名一处，Stage 1 revision 会残留同名量的其他出现位（本任务 f32 buffer 数在 §4.5/§3.3/§5.3/§9 R-8 修复，§1.6.3 A 行/§1.6.0 R3 #1 残留） | R | 本轮 REVIEW.md §7；DESIGN.md L348/L468 vs L857 | none（结构核对） | — | `tilelang-design-review/SKILL.md`（维度 7/8 增「同量全量位置」举证要求） |

### Transferable Lessons（可迁移教训）

- **机械检查 `fail` 的归因必须用工具自身的解析函数复现，不得凭印象**：上一轮把 `l0c_budget` 假阳性归到 §5.3 的 `128×512`，实测命中 §1.6.3 候选 C 行的 `1024×512`——归错源会导致「可消除性」判断与复盘都失真。核对时直接 `import design_calc_check` 调 `_section`/正则并打印 `m.start()` 行号。
- **提「口径/计数不一致」类 review 意见时，先 `grep` 出同量的全部出现位置再落笔**：只点名一处，revision 就只改一处，残留会以「建议级」形式跨轮留存（本任务即发生）。
- **实验裁决「备选方案」的 UB 预算复核要独立逐格重算并检查重叠 buffer**：本轮复核备选 B 发现 `acc_k/acc_v` 与共享向量链中的 `k_f/v_f` 有 4KB 重叠（偏保守重复计），证明「列全 buffer」与「buffer 生命周期复用」是两个独立核对点，缺一即可能得出过保守或过乐观的容量结论。

## Stage 3 (develop) Retrospective — retry_impl（attempt_index=2）— 2026-10-15

> 复盘对象：`_engram_decode_kernel.py` 的实现验证流程（attempt 1 = environment RUNTIME_FAIL；
> attempt 2 = 本文件）。产物通过 L0/L1，L2/Boundary 告警不阻塞。
> 工具链：tilelang 0.1.2+a2357253662（`/mnt/workspace/Agentkj/build`，USE_NPUIR=ON）/ CANN 9.0.0 / Ascend910。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| develop-flow | 设计文档 §3.3 伪代码把「reduction 工作区」写成 `T.alloc_fragment`，而 attempt-1 实现改成 `T.alloc_ub`；该偏离在真实工具链下直接 codegen 段错误（EXIT=139，无诊断），但设计未标注「reduction buffer 的合法 scope」这一阻塞级约束。 | attempt-1 `_engram_decode_kernel.py` L236-241 用 `T.alloc_ub((1,d_pad))` + `T.reduce_sum`；本工具链实测 segfault（faulthandler 落 `tilelang/engine/lower.py:180 device_codegen`） | `tilelang-op-design/SKILL.md` 内存层级维度：对含 `T.reduce_sum` 的实现显式标注归约 buffer 的 scope 约束与已知放置陷阱（引 pattern-library） | R |
| develop-flow | 设计 §0.6 R1（Cube `T.gemm` GEMV）在真实工具链下不可实现，属**设计层**结论，但 DESIGN 未做「MIX + 向量规约」可行性前置探针；Stage 3 花大量时间逐 scope 二分才确认。 | 本任务 repro `TRAP-npuir-mix-gemm-vector-reduce.py`；capability-gaps CG-2026-0022 | `tilelang-op-design/SKILL.md` 的 API 可行性确认：凡设计采用「Cube + Vector 融合」，须在 Stage 1 做最小 MIX 探针（gemm + 归约 + 广播）并落 `verify_equiv`/probe 记录，避免设计冻结后才发现不可编译 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| npuir Developer 下**裸 out-of-place 标量广播**（`dst[j] = s[0,0] * src[j]`，仅 1 个向量操作数）触发 codegen 段错误（EXIT=139，无诊断）；含 ≥2 个向量操作数的形态（`dst[j]=src[j]*s*wt[j]`）与 in-place 形态均可编译——迁移向量算子时标量广播须带第二个向量操作数或改 in-place | D | 本任务 Stage 3 bisect（split4/split5/split6 矩阵）；`_engram_decode_kernel.py` 的 R4 物化改为 in-place `v_f[j]=v_f[j]*s` | `python examples/engram_decode/_engram_decode_kernel/repro/TRAP-npuir-vec-scalar-broadcast.py` | tilelang 0.1.2+a2357253662 / CANN 9.0.0 / Ascend910 / 2026-10-15 | `pattern-library/traps-compiler.md`（新增 TRAP-npuir-vec-scalar-broadcast）；`tilelang-vector-skill/SKILL.md` |
| npuir 下 **GM 输入张量为 fp32** 时，shared 向量规约 + 标量广播链在 BiShengIR 被拒（`expected result type with stride = ... in dim = 0`）；同形态 fp16（经 `vcast` 升 fp32）通过——fp32 GM 输入路径不可用（不影响 fp16/bf16 manifest 契约） | D | `repro/TRAP-npuir-fp32-vector-reduce.py`（实跑）；`_engram_decode_kernel.py` L2-fp32 告警 | `python examples/engram_decode/_engram_decode_kernel/repro/TRAP-npuir-fp32-vector-reduce.py` | tilelang 0.1.2+a2357253662 / CANN 9.0.0 / Ascend910 / 2026-10-15 | `pattern-library/traps-compiler.md`；`.agents/evolution/capability-gaps.md` CG-2026-0022 |
| Developer MIX kernel（Cube `T.gemm` + shared 向量规约 + 标量广播）在 BiShengIR 被内存层级 verifier 拒绝（cbuf→cc / load src==gm / stride），两半各自单独可编译——含 Cube+Vector 融合的设计须 Stage 1 前置探针 | D | `repro/TRAP-npuir-mix-gemm-vector-reduce.py`；CG-2026-0022 证据链 | `python examples/engram_decode/_engram_decode_kernel/repro/TRAP-npuir-mix-gemm-vector-reduce.py` | tilelang 0.1.2+a2357253662 / CANN 9.0.0 / Ascend910 / 2026-10-15 | `.agents/evolution/capability-gaps.md` CG-2026-0022；`tilelang-mixcv-skill/SKILL.md` |
| 多步 recurrence 测试 harness 的**窗口对齐**与 **state dtype** 两个陷阱：kernel 始终输出整窗 `max_conv_len` 行，而 golden 对短初始状态只返回新增行——须 front-pad golden 到同窗比较；`cs` 经 `_run_op` 往返后须 `.to(dtype)` 再回灌（fp32 回灌被当作 fp16 解释 → 巨大误差） | P | 本任务 L1-multistep 首跑 `state max_err=3.21`（窗口未对齐）与 `513.4`（dtype 未回灌）两次失败 | none（已固化在 `_engram_decode_kernel.py::run_multistep`） | tilelang 0.1.2+a2357253662 / CANN 9.0.0 / Ascend910 / 2026-10-15 | `tilelang-op-develop/templates/op_template.py`（recurrent-op 测试骨架注释） |

### Transferable Lessons（可迁移教训）

- **含 Cube+Vector 融合的迁移设计，在 Stage 3 第一次动手前先做「gemm + 向量规约 + 标量广播」最小探针**：本任务 R1（Cube GEMV）在真实工具链下被 BiShengIR 内存层级 verifier 整体否决（CG-2026-0022），回退向量 GEMV 才通过；探针成本远低于逐 scope 二分。
- **向量规约的 workspace scope 是阻塞级选择**：`T.reduce_sum` 的源/目标放 `T.alloc_ub` 会 codegen 段错误（EXIT=139、无诊断），`T.alloc_shared` 是参考实现的可用形态；设计伪代码里 reduction 用 fragment 是另一可用族，但三者在 MIX/纯向量下行为不一致，必须实测。
- **标量广播（kernel 独有的陷阱）**：裸 `dst[j]=s*src[j]` 段错误；写成 `dst[j]=src[j]*s*wt[j]`（≥2 向量操作数）或 in-place 即可。这是 npuir 向量算子迁移的高频坑，先按此形态写。
- **fp32 GM 输入路径单独不可用**：manifest 若含 fp32 workload 需提前探针；fp16/bf16 经 `vcast` 升 fp32 的链路是安全形态。
- **recurrent/滑窗算子的测试须对齐「整窗 vs 逻辑有效行」并回灌正确 dtype**：kernel 常恒输出整窗，golden 可能只返回有效行；`cs` 往返后不 `.to(dtype)` 会被静默按错误 dtype 解释。
