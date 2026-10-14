# RETROSPECTIVE.md — _batch_norm_bwd_wide_kernel

> 追加式日志：每次 Stage 执行追加一个章节，不覆盖历史。schema 见 `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`。

## Stage 1 (design) Retrospective — first_design（migration-harness）— 2026-10-14

### 结论

`BatchNormBwdOp` 的宽核 `_batch_norm_bwd_wide_kernel` 迁移设计完成，复用 norm 族 row-reduction 模板的 persistent 分核（`num_kernels = min(C, 40)`，实查 20 AICore ×2 = 40）+ 尾块 `T.min` + src/dst 双显式 slice + fp16/bf16 fp32 中转链。相对 fused_add_layer_norm（尾轴行归约）的四个新增设计点：(1) **中间轴归约**——逐通道平面 `x[:, c, :]` 是原生 `[N,C,S]` 的 strided 区域，用 `T.copy` 双显式 slice gather + 两步 `T.reduce`（`dims=[1]`→`(N,1)` 再 `dims=[0]`→`(1,1)`）；(2) **held/tiled 双档**——`N·S·BPE ≤ 160KB` 走 held 单扫描（复用探针已验证形态），超大 S（large-spatial `S=1048576`）走单核 S-tile 两扫描，**无跨核归并、无 atomics**（替代 GPU split kernel）；(3) **`T.reduce` size-1 维退化**——`S=1`（resnet-fc）或 `N=1` 时对应 reduce 层在 size-1 轴上无维度变化，违反 `T.reduce` Shape 规则，须工厂层 `if S>1`/`if N>1` 分支跳过；(4) **分布式 epilogue + 倒数乘（O1/O2）**——`grad_x = scale·g − per·gb − per·gw·x̂`，机器验证 EQUIV_PASS（21 案例 max ulp 0）。等价性由 `verify_equiv.py` 机器验证（O1+O2 联合全 EQUIV_PASS + raw-moment `grad_weight` 抵消判别段）。gate 1 一次通过。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | **gate 1 的 `S1-REF-PATHS` 会拒绝 `path:function` / `path:行号` 形式的引用**：`REF_PATH_RE` 把 ASCII `:` 之后的锚点并入路径 token，`os.path.exists` 对 `.../norm.py:batch_norm_backward` 判不存在 → gate fail。设计文档中凡引用「某文件的某函数/某行」都会踩中（本设计首轮 6 处） | DESIGN.md 首轮 gate 输出 `引用的仓库路径不存在：docs/examples`（同机制：`docs/examples` 斜杠拼接 token）；gate_lint.py `_ref_path_failures` + `REF_PATH_RE` | 在 `tilelang-op-design` 模板/自检清单加一句话：引用仓库文件时用「`path` 的 `func`」或 `path#Lnn` 形态，**不要**写 `path:func` 或 `path:行号`；或放宽 `_ref_path_failures` 对 ASCII `:` 后锚点的切分 | R |
| Info-source | **`T.reduce` 在 size-1 轴上的合法性未被文档显式点出**：`T.reduce` Shape 规则（「src/dst 有且仅有一个维度不同，且 dst 该维为 1」）意味着 `(N,1)→(N,1)` 的 `dims=[1]` reduce（即 S=1 退化）**非法**——两个维度都相同，无维度不同。设计须按 shape 静态跳过该层 reduce。探针形状（S≥256）未覆盖此点 | DESIGN.md §5.4/§9.2；`docs/Tilelang.language/规约操作/T.reduce.md` §2.2.2 合法/非法例（`(M,N,K)→(M,N,K)` 标为非法） | 在 `docs/Tilelang.language/规约操作/T.reduce.md` §2.2.2 补一句「src 待归约维 size=1 时该 reduce 非法（无维度变化），退化情形须在上层静态消去」 | C |
| Info-source | **规约类算子的「两步 reduce」配方（非尾轴归约）缺标准示例**：非尾轴归约（本算子通道为中间轴、归约轴 `{0,2}`）须把平面物化为 `(N,S)` 后 `dims=[1]` + `dims=[0]` 两步，`T.reduce` 文档只给单步 `dims=[1]` 例 | DESIGN.md §3.2/§0.6 R2；探针 `batch_norm_bwd_probe.py` L158-162 | 在 `docs/Tilelang.language/规约操作/T.reduce.md` §2.4 增补「两轴顺序归约到 (1,1)」示例（含 size-1 维消去注） | C |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **`path:func` / `path:行号` 引用形态触发 gate 1 `S1-REF-PATHS` 假阳性**：设计文档引用仓库源码函数/行号时，ASCII `:` 锚点被路径正则吞并，`os.path.exists` 判不存在 → gate fail。改用「`path` 的 `func`」形态即通过 | R | 本任务首轮 `statectl gate 1` 输出（`引用的仓库路径不存在：docs/examples`）+ DESIGN.md 中 6 处 `.../norm.py:batch_norm_backward` 等 | `python3 .agents/tools/statectl.py gate 1 --dir examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel`（首轮 fail、修锚点后 pass） | tilelang 0.1.2+417dae99 + Ascend910B3 + CANN 8.5.2 | .agents/tools/gate_lint.py / tilelang-op-design 模板自检清单 |
| **BN 反向 `grad_weight` 保留中心化 `Σg·x̂`（拒绝 raw-moment `rstd·(Σg·x−mean·Σg)`）**：raw-moment 代数恒等但大均值小方差 fp32 灾难性抵消（判别段 fp32 违反率 0.1553），中心化式跨 dtype 安全；迁移不得为省 1 乘 1 减而替换 | D | DESIGN.md §1.6.0 R1#3、§1.6.1 判别段 A；verify_equiv.py `_discr_moment` | `python3 examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/verify_equiv.py` | tilelang 0.1.2+417dae99（设计期，仅 torch CPU 验证）+ Ascend910B3 + CANN 8.5.2 | algorithm-candidates ALG-reduction / bottleneck-patterns（抵消式） |
| **非尾轴逐通道归约的 NPU 配方（strided plane gather + 两步 `T.reduce` + `T.vbrc` 标量广播）**：中间轴 `C` 的通道平面 `(N,S)` 在 GM 是 strided 区域（行跨步 `C·S`），`T.copy` 双显式 slice 直载 UB 后 `dims=[1]`→`dims=[0]` 两步归约到 `(1,1)`；逐通道标量经 `T.vbrc` 展开；超大 S 用单核 S-tile 两扫描（无跨核归并）——替代 GPU split kernel | P | DESIGN.md §0.6 R2/R5、§1.6.0 R1#1/#2、§3.3；探针 `examples/batch_norm_bwd/probe/PROBE.md`（真机 PASS，含超大 S） | `python examples/batch_norm_bwd/probe/batch_norm_bwd_probe.py` | tilelang 0.1.2+417dae99 + Ascend910B3（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/layout.md（非尾轴归约配方）/ cases.md（BatchNorm bwd 迁移档案） |

### Transferable Lessons（写给本迁移任务后续函数）

- **非尾轴逐通道归约**：通道轴是中间轴时，逐通道平面在 GM 是 strided 区域（行跨步 `C·S`、行内 stride=1）。配方 = `T.copy(gm[0:N, c, s0:s0+real], ub[0:N, 0:real])` 双显式 slice 直载 → `T.reduce(..., dims=[1])` 得 `(N,1)` → `T.reduce(..., dims=[0])` 得 `(1,1)`；跨 S-tile 累加用 `clear=False`（先 `T.clear` 累加器）。
- **`T.reduce` 在 size-1 轴上非法**：`S=1` 或 `N=1` 时对应维度无变化（`(N,1)→(N,1)`、`(1,1)→(1,1)`），违反「有且仅有一个维度不同且 dst 该维为 1」。必须在**工厂层** `if S>1:` / `if N>1:` 静态消去该层 reduce（不要放进 trace 体内）。
- **`path:func` / `path:行号` 引用会触发 gate 1 `S1-REF-PATHS` 假阳性**：gate 把 ASCII `:` 后的锚点当路径继续查 `os.path.exists`。引用源码用「`path` 的 `func`」或反引号分段，避免 `...py:batch_norm_backward` 这样的 token；`docs/a/b` 拼接式短语（如 `docs/examples`）同样触发，勿写。
- **one core per channel 使通道归约闭环**：BN 反向的 `grad_weight/grad_bias` 是逐通道标量、通道间独立，NPU 上「一核一通道 + 核内 S-tile」即可完成整通道归约，**无需跨核归并、无需 atomics、无需 workspace**（GPU 的 split kernel/finalize 在 NPU 上是可避免的复杂度）。
- **held/tiled 双档判据用 UB 预算**：整通道平面驻留（`N·S·BPE ≤ 预算`）走单扫描（流量最小，= manifest roofline 下界 3NCS·eb）；超出则单核 S-tile 两扫描（流量 5NCS·eb，置换 UB 有界）。BPE fp16/bf16 ≈ 18 B/元素（stage 2 + 4 个 fp32 缓冲）、fp32 ≈ 16 B/元素。

## Stage 1 (design) Retrospective — revision（v0 → v1，migration-harness）— 2026-10-14

### 结论

Stage 2 REVIEW.md「结论: 不通过」（2 阻塞 + 5 建议）触发的定点修订。**两个阻塞根因**：(1) **`out_idx` 返回序**——tilelang 的返回序 = `out_idx` 列表序且**不排序**（`tilelang/jit/jit_npu.py` 的 `JitKernel_NPU.__call__`：`return [full_args[i] for i in self.out_idx]`），v0 保持源参数序却写 `out_idx=[5,6,7]`，返回 `(grad_weight, grad_bias, grad_x)` 与契约相反；v1 改 `out_idx=[7,5,6]`（返回 `(grad_x, grad_weight, grad_bias)`），并明确 `num_inputs = total_params - len(out_idx)` ⇒ wrapper 须改 5 参调用解包。(2) **尾 tile 垃圾列**——v0 pass-1 对完整 `(N,bS)` 做 `T.reduce(dims=[1])`，尾 tile `[real:]` 垃圾列被计入行和污染 `gb/gw`；v1 用 `T.reduce(..., size=[N, real])` 截断有效 extent（`docs/Tilelang.language/规约操作/T.reduce.md` §2.1）。建议项：fp32 BPE 16→20（stage 对所有 dtype 均分配）、§1.4 shift 符号统一为 `per·gb`、§4.5 bS 表统一为公式值 2272、NPU op `numel()==0` 早退列入集成清单、§5.5 标注宽核服务范围。保留 v0 全部通过项（verify_equiv O1+O2 重跑仍 EQUIV_PASS 21 案例、判别段 A fp32 违反率 0.1553）。gate 1 pass（无 warning）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | **`out_idx` 返回序在 tilelang 文档/模板中无显式约定**：tilelang 返回序 = `out_idx` 列表序（不排序），设计者易按「host 参数序」直觉写 out_idx，导致返回三输出错位。`tilelang` 侧仅代码体现（`jit_npu.py` `__call__`），`T.reduce`/jit 文档未述 | DESIGN.md v0 §0.2/§3.3/§9.2（`out_idx=[5,6,7]`）+ REVIEW.md 问题 1；`tilelang/jit/jit_npu.py` 的 `JitKernel_NPU.__call__`（`return [full_args[i] for i in self.out_idx]`）与 `num_inputs = total_params - len(out_idx)` | 在 `tilelang-op-design` 模板 §3.3 增「多输出 jit 约定」小节：返回序 = out_idx 列表序、调用参数数 = 总参数 − 输出数、wrapper 解包须按列表序；或在 jsdoc/设计检查清单加机械核对项 | R |
| Design-flow | **尾块 reduce 的 extent 截断在设计阶段无强制核对项**：v0 用「slice 截断 I/O」即认为边界安全，遗漏「reduce 也必须限 extent」，且用错误的「dims=[1] 逐行/按列隔离」论证蒙混自检 | DESIGN.md v0 §3.3 伪代码 L510-511 + 后注；REVIEW.md 问题 2；`docs/Tilelang.language/规约操作/T.reduce.md` §2.1（`size` 参数） | 在 tilelang-op-design 自检清单「边界处理」项加：「凡对宽于有效 extent 的 buffer 做 reduce，须核对是否传 `size=[…, real]`」 | R |
| Info-source | **多输出 `out_idx` 的 `size` / 参数数语义无独立文档例**：`T.reduce.md` 有 `size` 参数但仅一行「控制 reduce 实际参与计算的数据范围」，无尾块示例；多输出 jit 亦无示例 | DESIGN.md v1 §3.3；`docs/Tilelang.language/规约操作/T.reduce.md` §2.1 | 在 `T.reduce.md` §2.1/§2.4 补「尾 tile `size=[M, real]` 截断」示例；jit 文档补多输出 `out_idx` 返回序约定 | C |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **tilelang 多输出 jit 返回序 = `out_idx` 列表序（不排序），调用参数数 = 总参数 − len(out_idx)**：设计多输出算子时，返回顺序逐位等于 `out_idx` 写法顺序；写 `[5,6,7]` 就按 `(5,6,7)` 顺序返回，不做升序归一。契约对齐须核对 `[full_args[i] for i in out_idx]` | D | DESIGN.md v1 §0.0/§0.2/§3.3；REVIEW.md 问题 1；`tilelang/jit/jit_npu.py` `JitKernel_NPU.__call__` | `python3 .agents/tools/statectl.py gate 1 --dir examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel`（读源码 + 设计核对，非真机） | tilelang 0.1.2+417dae99 + Ascend910B3 + CANN 8.5.2 | pattern-library/traps-*.md（jit 多输出 out_idx）/ tilelang-op-design 模板 §3.3 |
| **尾 tile 归约必须传 `size=[N, real]` 截断有效 extent**：对完整 `(N,bS)` 做无边 `T.reduce(dims=[1])` 会把尾 tile `[real:]` 未初始化/残留列计入行和，污染通道梯度；`dims=[1]` 不隔离垃圾列（v0 错误论证）。slice 管 I/O、`size` 管归约，二者成对 | D | DESIGN.md v1 §0.0/§3.3/§4.5/§8.3（L0-8/9/13 非整除用例）；REVIEW.md 问题 2；`docs/Tilelang.language/规约操作/T.reduce.md` §2.1 | L0-8 (4,128,1048576) fp16 bS=2272 → 尾 tile 1184（v0 形态必失败、v1 通过） | tilelang 0.1.2+417dae99 + Ascend910B3 + CANN 8.5.2 | pattern-library/layout.md（非尾轴/尾块归约）/ traps-*.md |
| **UB BPE 口径须与实际分配一致（含所有 dtype 都分配的 `stage`）**：`stage` 对所有 dtype 均分配（fp32 亦经 stage 中转到 fp32 计算缓冲），故 fp32 BPE = itemsize + 16 = 20（非 v0 的 16）；否则 bS 反推公式高估 bS → 实际 UB 溢出 | P | DESIGN.md v1 §0.0/§3.3/§4.5；REVIEW.md 问题 3 | `python3 .agents/tools/design_calc_check.py --design examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md`（ub_budget 跳过，人工复算） | tilelang 0.1.2+417dae99 + Ascend910B3 + CANN 8.5.2 | pattern-library/constants.md / tilelang-op-design §4.5 自检 |

### Transferable Lessons（写给本迁移任务后续函数）

- **多输出 jit 的返回序 = `out_idx` 列表序（tilelang 不排序）**：`program(...)` 返回 `[full_args[i] for i in out_idx]`；且调用参数数 = `total_params - len(out_idx)`（输出由 jit 自动分配，不进调用实参）。要返回 `(grad_x, grad_weight, grad_bias)` 而保持源参数序时，`out_idx` 必须写 `[7,5,6]`；wrapper 按列表序解包。
- **边界安全 = I/O slice 截断 + 归约 `size` 截断，二者缺一不可**：尾 tile 对完整 `(N,bS)` buffer 做 `T.reduce(dims=[1])` 会把 `[real:]` 的未初始化/残留列计入行和（`dims=[1]` 不隔离垃圾列——垃圾恰在求和维）。凡 reduce 宽度超出有效 extent，必须传 `size=[N, real]`。
- **UB 预算 BPE 要按「实际分配的全部 buffer」算**：若某中转缓冲（如 `stage`）对**所有** dtype 都分配，则 fp32 也要计入其 itemsize（本算子 fp32 BPE = 20，非 16）。BPE 口径错误会让 bS 反推公式失效。
- **表值与公式单一来源**：§4.5 表 bS 必须能由 §3.3/§5.2 公式复推（同 BPE、同对齐）；否则表/公式两套口径会让「是否命中尾 tile」在两者下结论不同。
- **选核策略与分核表对齐**：分核 workload 表只列「宽核被 `_holder` 选中」的 shape；被 `select_bwd_kernel` 路由到其它核（如 `spatial≥65536` 且 `C<1024` → `bwd_split`）的 shape 须标注，避免设计声称服务实际不服务的负载。

## Stage 3 (develop) Retrospective — first_impl（migration-harness）— 2026-10-14

### 结论

`_batch_norm_bwd_wide_kernel.py` 实现完成（Expert / Vector、`ASCEND_MODE="Expert"`、`out_idx=[7,5,6]`、单 `T.Scope("Vector")` + `T.alloc_ub`、一维 persistent 分核 `num_kernels = min(C, 40)`、核内 `T.serial` grid-stride、held 单扫描 / tiled 两扫描、两步 `T.reduce` + `T.reduce(size=[N, real])`、`T.vbrc` 标量广播、fp16/bf16 `T.vcast(rint)` fp32 中转 + 单次舍回、`T.min` 尾块 extent + src/dst 双显式 slice）。`python _batch_norm_bwd_wide_kernel.py --level all` 真机通过（exit 0）：**L0 14/14 PASS、L1 12/12 PASS + 5 契约 guard PASS**；L2/Boundary warn-only 记录 2 项（见下）。实现期与设计 §3.3 的**必要适配**两处（v-op 不接受运行时 slice、UB 预算改用实测多缓冲 footprint），语义不变量（双 slice I/O、reduce `size` 截断、fp32 中转、单次舍回、one core per channel）全部保留。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | **设计 §3.3 伪代码用运行时 extent 切片喂 v-op（`T.vcast(stage[0:N, 0:real], g32[0:N, 0:real], ...)`）在本工具链非法**：v-op 操作数切片须为静态边界，运行时 `real` 触发 parser 在 `tvm/tir/buffer.py __getitem__` 的 `int(lanes)`（`TypeError: int() argument ... not 'Var'`）。`T.copy` 接受运行时切片，v-op 不接受。设计/探针均未覆盖「v-op + 运行时切片」组合 | 实现首轮编译报错（trace 期 `TypeError: int() argument must be ... not 'Var'`，指向 `T.vcast(stage[0:N,0:real], ...)`）；对照 `T.copy` 同形切片编译通过 | 在 `docs/Tilelang.language/` v-op 通用说明或 `T.vcast.md` §2.3 明确「v-op 操作数仅接受静态 extent；尾块用 `T.reduce(size=...)` 限归约范围 + dst slice 限写出，v-op 在全 buffer 上运行」；tilelang-op-develop 模板尾块示例同步 | D |
| Info-source | **UB 预算 BPE 未计入 auto-multi-buffer 膨胀，tiled bS 反推偏大 → 编译期 UB 溢出**：设计 `BPE = itemsize + 16`（fp16 18 / fp32 20）与 `UB_SAFE_BYTES=163840` 得 fp16 tiled bS=2272，真机 `ub overflow, requires 1602560 bits`（= 22.04 B/elem 实测）。fp32 实测 ≈25.8 B/elem（N=128 时 128×60 → 198432 B，对齐 ada 的 ~26）。设计 §4.5 已预警膨胀但取值未落到公式 | `--level L0` 首轮 L0-8/9/13 编译溢出（`requires 1602560 bits while 1572864 bits available`）；标定脚本 `N=4 fp16 bS=2272 → 200320 B (22.04 B/elem)`、`N=2 fp32 bS=4096 → 196992 B (24.05)`、`N=128 fp32 bS=60 → 198432 B (25.84)` | 在 `pattern-library/constants.md`/`traps-compiler.md` TRAP-UB-multibuffer-inflation 补「本结构实测：transit ~22 B/elem、fp32 ~25.8 B/elem（非 18/20）」；tilelang-op-design §4.5 bS 反推公式须用实测膨胀后 BPE | D |
| Design-flow | **`T.vbrc((1,1)→(1,1))` 及 `(1,1)` 平面上的同形 v-op 均不可用**：`L=1`（N=1 且 S=1）时逐通道标量广播 dst 与 src 同形 → `vbrc op have empty broadcast dims` MLIR verify fail；改同形直接 vsub/vmul 后又报 `hivm.hir.vadd op ... operand at index 0 is vector-only`。实现加了 `scalar_plane` 工厂分支仍无法编过该单元素退化形（L2 warn 记录，非阻塞） | L2 `(1,1,1) float16` 编译失败两形态（vbrc 空广播 / vadd 非 vector） | 在 tilelang-vector-skill 或 traps-runtime 记录「1 元素平面（N=bS=1）的标量广播/同形 v-op 不可用，须由 op 层早退或升维」 | C |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **v-op 操作数切片须静态 extent；尾块正确性 = `T.reduce(size=[N, real])`（归约）+ dst 显式 slice（写出），v-op 在全 buffer 上跑**：`T.vcast(stage[0:N,0:real], ...)` 触发 parser `int(Var)`；改用全 buffer vcast 后 reduce 靠 `size` 截断、写出靠 dst slice 截断，数值正确（L0-8/9/13 尾 tile 全过） | D | 实现首轮 trace 报错；`_batch_norm_bwd_wide_kernel.py` pass-1/pass-2 全 buffer vcast；L0-8/9/13 PASS | 见本文件 `_batch_norm_bwd_wide_kernel.py`（`examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/`）L0 非整除三例 | tilelang 0.1.2+417dae99 + Ascend910B3 + CANN 8.5.2 | docs/Tilelang.language/数据类型转换操作/T.vcast.md / traps-compiler（v-op 静态边界） |
| **本 kernel 结构的 auto-multi-buffer 实测 footprint：transit ≈22.0 B/elem、fp32 ≈25.8 B/elem（nominal 18/20）**：额定 BPE 反推 bS 会 UB 溢出；用实测膨胀后 BPE + 180KiB 安全预算反推 bS 即编译通过（fp16 tiled bS=2088、fp32 N=128 bS=52） | D | `N=4 fp16 bS=2272→200320B`、`N=128 fp32 bS=60→198432B`；修正后 `--level all` exit 0 | 见 `_batch_norm_bwd_wide_kernel.py` 常量 `_UB_PE_TRANSIT/_UB_PE_FP32/_UB_SAFE_BYTES` | tilelang 0.1.2+417dae99 + Ascend910B3 + CANN 8.5.2 | pattern-library/constants.md CONST-capacity-910B2C / traps-compiler TRAP-UB-multibuffer-inflation |
| **`T.reduce(..., size=[N, real], clear=False)` 跨 S-tile 累加真机可用**：运行时 extent 的 `size` + `clear=False`（先 `T.clear` 累加器）联合形态在探针中未单独覆盖，本实现在 L0-8/9/13 非整除尾 tile 上验证通过（gb/gw 无垃圾列污染） | P | L0-8/9/13（bS=2088，尾 tile 非整除）PASS；BLOCKING REVIEW 遗留项「`T.reduce(size)+clear=False` 未单独验证」已闭环 | 见 `_batch_norm_bwd_wide_kernel.py` pass-1 两处 `T.reduce(size=[N, real], clear=False)` | tilelang 0.1.2+417dae99 + Ascend910B3 + CANN 8.5.2 | docs/Tilelang.language/规约操作/T.reduce.md §2.1/§2.3 |

### Transferable Lessons（写给本迁移任务后续函数）

- **v-op（`T.vcast`/`T.vsub`/`T.vmul`/`T.vbrc` 等）的操作数切片必须是静态 extent**：运行时 extent（如尾 tile `real`）会触发 parser `int(Var)` 拒绝。尾块正确性不靠 v-op 切片，而靠“**归约传 `size=[...]`、写出用 dst 显式 slice**、v-op 在全 buffer 上运算”；`T.copy` 则接受运行时切片（可继续用于 I/O 双 slice）。
- **UB 预算的 BPE 必须用编译期实测的 auto-multi-buffer footprint（本结构 transit ≈22、fp32 ≈25.8 B/elem），不能用 nominal buffer 字节和（18/20）**：否则 bS 反推偏大 → 编译期 `ub overflow`。先用小标定 probe 读 `requires N bits` 报文定 per-elem 成本，再定 safe budget（本例 180 KiB）与 bS。
- **避免 1 元素平面（N==1 且 bS==1）上的标量广播**：`T.vbrc((1,1)→(1,1))` 空广播 MLIR verify fail，同形直接 v-sub/v-mul 亦报 `vadd operand ... not vector-only`。此类退化 shape 应在上层（op/wrapper）早退或升维，不要依赖 kernel 内的标量广播路径。
- **持通道的 multi-output 归约实现顺序**：pass-1 用 `T.clear(累加器)` + 逐 S-tile `T.reduce(size=[N,real], clear=False)` 累加行和 → 最后一次 `dims=[0]`（`N>1`）或直接沿用（`N==1`）；`S==1` 时跳过退化的 `dims=[1]` 层（`T.copy` 平面即行和）。这些分支都放**工厂层 Python 常量**（`s_gt1`/`n_gt1`/`single_tile`/`scalar_plane`），trace 内只留纯名 if。
