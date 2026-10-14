# RETROSPECTIVE.md — _batch_norm_bwd_kernel

> 追加式日志：每次 Stage 执行追加一个章节，不覆盖历史。schema 见 `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`。

## Stage 1 (design) Retrospective — first_design（migration-harness）— 2026-10-14

### 结论

`BatchNormBwdOp` 的**通用/基础核** `_batch_norm_bwd_kernel` 迁移设计完成（Expert / Vector）。本函数是同族三核中的**通用回退**（`select_bwd_kernel` 把 `_holder` 判为 `tiled` 的 shape 路由到它；典型是大 `C ≥ _SPLIT_MAX_C(1024)` 或既非 wide 也非 split 服务的 shape），故设计核心是**通用性**：one core per channel + 一维 persistent 分核（`num_kernels = min(C, 40)`，实查 20 AICore ×2 = 40）+ UB **二维 `(bN,bS)` 流式分块**，按 UB 预算在 held 单扫描（`N·S` 入 UB）与 tiled 两扫描（超大 L / 大 N，含 N 维分块）间切换。相对同族宽核（held/tiled、固定 `N` plane）的**新增设计点**：(1) **源 flat `block_l` tile 的 NPU 重设计**——源把通道展平成一维 `L`、逐元素 `l//S, l%S` 定位（L440/451/497），非仿射地址破坏向量化，NPU 改仿射二维 `(bN,bS)` plane slice（§0.6 R4）；(2) **N 维分块**——为覆盖大 N 服务区的 UB 有界，加 `T.serial(n_tile_N)` 外层并把 N 归约改为跨 N-block `clear=False` 累加（§0.6 R7）；(3) **接口契约原样继承**——源 `out_idx=[-1]`（仅 `grad_x` 自动分配；`grad_weight/grad_bias` 由 host 预分配并以参数写入），本设计**保持**该契约，NPU wrapper 的 7 参调用**零改动**（与宽核变体统一三输出 `out_idx=[7,5,6]`、须改 5 参调用相反）。等价性由 `verify_equiv.py` 机器验证（O1 倒数乘 + O2 分布式 epilogue，21 案例 max ulp 0 全 EQUIV_PASS；raw-moment 判别段 A fp32 违反率 0.1553 → 拒采）。UB `BPE` 直接采用同族宽核 Stage 3 **实测膨胀值**（transit ≈22、fp32 ≈25.8 B/elem）+ 180 KiB 安全预算，规避了「额定 BPE 高估 bS → ub overflow」的已知陷阱。gate 1 一次通过（无 failure、无 warning）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | **同算子多变体的 `out_idx` 契约未标准化约束**：同族三核在 GPU 源里有两种约定——wide/split 用 `out_idx=[-1]`（host 预分配通道梯度/workspace），而本仓宽核 v1 设计改为 `out_idx=[7,5,6]`（统一三输出、5 参调用）。同族各变体若各自选约定，集成期易出现 wrapper 调用实参数量不匹配（wide 7→5、base 7、split 三段各自不同）。本设计选「原样继承源 host 契约」并在 §0.2/§3.3/§9.1/§10.5 逐处标注调用元数 | 本设计 §0.2/§3.3 注；宽核 `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md` §0.0 阻塞项 1（`out_idx` 返回序）；NPU wrapper `examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py` 的三处 `forward` | 在 `tilelang-op-design` 模板 §3.3「多输出 jit 约定」小节补一句：同一 op 的多个 kernel 变体应各自显式声明「保持源参数序 / out_idx 取值 / wrapper 调用实参数量」，或由 conductor 在集成清单统一；避免变体间约定漂移 | R |
| Info-source | **「通用回退核」的服务区与 manifest workload 不交集**：manifest 的 resnet-fc/stage1/2/3 由 `_holder` 路由宽核、large-spatial 路由拆分核，本函数（`bwd_kernel`）**默认不被任何 manifest workload 命中**（服务区 = `_holder` 判 `tiled` 的大 C/异形 shape）。设计期若照 manifest 列表做设计会遗漏真实服务区（大 N、大 C） | 本设计 §0.5/§5.5；`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py` 的 `_holder`/`select_bwd_kernel`；拆分核 Stage 2 §0.0（同款「服务区」误判） | 在 `tilelang-op-design` 迁移流程（Phase M0/§0.7）补一条机械步骤：从 wrapper 的选核策略反推**本函数实际服务区**（而非 manifest 全集），并在 §5.5 分核表标注 | R |
| Info-source | **同族实测 UB footprint（BPE 膨胀）已可用，但未进入设计期预算公式**：宽核 Stage 3 实测 transit ≈22 B/elem、fp32 ≈25.8 B/elem，而设计模板 §4.5 仍按额定 `itemsize+16` 举例；本设计已直接引用实测值，但该知识此前只在宽核 retrospective，未沉淀到通用参考 | 本设计 §4.5；`examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/RETROSPECTIVE.md`（Stage 3 实测） | 把「同构 UB 流式 kernel 的实测 BPE（transit ~22 / fp32 ~25.8）」写入 `pattern-library/constants.md` CONST-capacity-910B2C 或 `traps-compiler.md` TRAP-UB-multibuffer-inflation，供后续算子设计期直接引用 | C |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **通用回退核（general tiled path）NPU 配方 = one core per channel + 一维 persistent 分核 + UB 二维 `(bN,bS)` 流式分块（held 单扫描 / tiled 两扫描，含 N 维分块）**：覆盖大 C（分核）与大 N（N 维分块）的任意形状；通道归约单核闭环，无跨核归并/atomics；源 flat `block_l` tile 的非仿射 `l//S,l%S` 地址重设计为仿射二维 slice | P | 本设计 §0.6 R1/R4/R7、§5.2/§5.5；同族宽核 `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md`（held/tiled）、拆分核 `examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/DESIGN.md`（二维 tiling） | `python3 .agents/tools/statectl.py gate 1 --dir examples/batch_norm_bwd/_batch_norm_bwd_kernel`（读源码 + 结构分析，非真机） | tilelang 0.1.2+417dae99 + Ascend910B3（实查 20 AI Core）+ CANN 8.5.2 | pattern-library/layout.md（通用回退核配方）/ cases.md（BatchNorm bwd 迁移档案） |
| **同算子多变体的 `out_idx` 约定须逐变体显式声明**：源 host 契约（`out_idx=[-1]` + host 预分配 `grad_weight/grad_bias` 并以参数写入）可原样继承、wrapper 零改动；若改为统一三输出（`out_idx=[7,5,6]`）则须同步改 wrapper 调用实参数量。返回序 = `out_idx` 列表序（tilelang 不排序），调用实参数 = 总参数 − len(out_idx) | R | 本设计 §0.2/§3.3 注/§9.1/§10.5；宽核 `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md` §0.0 阻塞项 1 | `python3 .agents/tools/statectl.py gate 1 --dir examples/batch_norm_bwd/_batch_norm_bwd_kernel` | tilelang 0.1.2+417dae99 + Ascend910B3 + CANN 8.5.2 | tilelang-op-design 模板 §3.3（多输出 jit 约定）/ pattern-library/traps-*.md |
| **BN 反向 `grad_weight` 保留中心化 `Σg·x̂`（拒绝 raw-moment）**：同族统一结论，本变体复核一致（D-1 判别段 A fp32 违反率 0.1553，中心化式跨 dtype 安全） | D | 本设计 §1.6.0 R1#3、§1.6.1 判别段 A；`verify_equiv.py` 判别段 | `python3 examples/batch_norm_bwd/_batch_norm_bwd_kernel/verify_equiv.py` | tilelang 0.1.2+417dae99（设计期，仅 torch CPU 验证）+ Ascend910B3 + CANN 8.5.2 | algorithm-candidates ALG-reduction / bottleneck-patterns（抵消式） |

### Transferable Lessons（写给本迁移任务后续函数 / 同算子变体）

- **通用回退核要按「实际服务区」设计，不是 manifest 全集**：从 wrapper 的 `select_bwd_kernel`/`_holder` 反推本函数真正服务的 shape（本算子 = `tiled` 区，大 C/异形），据此决定是否必须支持大 N（→ 二维 `(bN,bS)` 分块）、大 C（→ persistent 分核）。照 manifest 列表设计会遗漏真实负载。
- **flat（`l//S, l%S`）与仿射二维 `(N,S)` plane 的取舍**：源 GPU 用 flat 展平 tile 是地址便利；NPU 上 flat 逐元素地址**非仿射**，无法用一条 `T.copy` slice 表达、且破坏向量化。NPU 一律用二维 `(bN,bS)` plane（S 为连续内轴）+ 双显式 slice；flat 段已被拆分核验证为不可取。
- **UB 预算优先用同族实测膨胀 BPE 而非额定值**：同构 UB 流式 kernel 的 auto-multibuffer 实测 footprint（transit ≈22、fp32 ≈25.8 B/elem）远高于额定 `itemsize+16`（18/20）；设计期直接引用实测值 + 180 KiB 安全预算，可避免「bS 反推偏大 → 编译期 ub overflow」。
- **变体间接口约定要各自显式声明**：同算子多 kernel 变体的 `out_idx`/wrapper 调用元数可能不同（本算子 wide=[7,5,6]/5 参 vs base=[-1]/7 参）；迁移时优先「原样继承源 host 契约」以零改动集成，若统一则须同步改 wrapper 并在设计 §9.1 记集成项。
- **继承同族已验证边界不变量**：v-op 操作数切片须**静态 extent**（运行时 `rN/rS` 触发 parser `int(Var)`；尾块靠 reduce `size` + dst 显式 slice 截断）；`T.reduce` 在 size-1 轴非法（`N=1`/`S=1` 须工厂层静态消去）；1 元素平面（`N==1 且 bS==1`）标量广播不可用——这些均为同族 Stage 3 实测结论，设计期直接作为约束注入。

## Stage 1 (design) Retrospective — revision（v0 → v1，migration-harness）— 2026-10-14

### 结论

Stage 2 `REVIEW.md`「结论: 不通过」（2 阻塞 + 3 建议）触发的定点修订。**两个阻塞根因**：(1) **pass-1 跨 N-block 双重计数**——v0 §3.3 把 `T.clear(red/red2)` 放在 `for nn in T.serial(n_tile_N)` 循环**之外**只清一次，而块内两路 `T.reduce(dims=[1], clear=False)` 跨 S-tile 累加行和、块末 `T.reduce(dims=[0], clear=False)` 再把该 N-块行和累加进 `gb/gw`；`red/red2` 不复位使第 k 块的 `red` 已含前序所有块的行和，`gb = Σ_k Σ_{j≤k} rowsum(blk_j)`，`n_tile_N≥2` 的 tiled 用例（大 C `(32,2048,512)`、large-spatial `(4,128,1048576)`）三输出全错；v1 把 `T.clear(red/red2)` **移入 `for nn` 循环内**（`gb/gw` 循环外一次），与同族拆分核 §3.3 逐字对齐。(2) **v-op 运行时 extent 切片**——v0 §3.3 写 `T.vcast(stage[0:rN,0:rS], g32[0:rN,0:rS], …)`，运行时 TIR 表达式作切片 extent 违反 v-op 静态边界（parser `int(Var)`），与 v0 **自身** §5.4/§9.2 规则直接矛盾、同族宽核 Stage 3 已实测编译失败；v1 把全部 v-op 改为**全 buffer**（`T.vcast(stage, g32, …)` 等），运行时 slice 仅留给 `T.copy` 与 `T.reduce(size=…)`。建议项：§1.6.0 R3/R4 缓冲峰值口径对齐 §4.3（4 路 fp32 + 1 路中转）、`(1,C,1)`（`L=1`）退化平面加工厂 `scalar_plane` 守卫 + L0-15、§3.3 pass-2 加 `if single_tile:` 分支落实 O3、`out_idx=[-1]` 返回措辞订正为「返回 `grad_x` 张量」。保留 v0 全部通过项（D-1 `verify_equiv` O1/O2/O3 `EQUIV_PASS` 21 案例 + 判别段 A fp32 违反率 0.1553、§1.6 调研、Expert/Vector、一维 persistent 分核、fp32 中转链、`out_idx=[-1]` 契约）。gate 1 pass。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | **「清一次累加器」与「跨块 `clear=False` 累加」的组合缺机械自检项**：设计者易把 `T.clear(red)` 放在 N/S 外层循环外一次，同时在块内用 `clear=False` 把行和累加进另一个累加器 `gb`，形成「跨块复用同一累加器却只初始化一次」的双重计数。v0 §3.3 即此形态（`red` 跨 N-block 不复位） | DESIGN.md v0 §3.3（`T.clear(red/red2)` 在 `for nn` 外）+ REVIEW.md 问题 1；同族拆分核 §3.3（`T.clear(red/red2)` 在 `for nn` 内）对照 | 在 `tilelang-op-design` 自检清单「循环与同步 / 累加」项加：「凡跨循环维（N/tile）用 `clear=False` 累加的累加器，其 `T.clear` 必须与该循环同层或更内层；若累加器跨多级维度复用，须逐级核对复位层」 | R |
| Design-flow | **设计文档自相矛盾未在自检期拦截**：v0 §3.3 用运行时 extent 喂 v-op，而同一文档 §5.4/§9.2 明写「v-op 须静态 extent」，§9.2 甚至把 `T.vcast(stage[0:rN,0:rS], …)` 列为常见错误。自检未做「伪代码 ↔ 约束章节」双向一致性核对 | DESIGN.md v0 §3.3 L548–554 vs §5.4/§9.2；REVIEW.md 问题 2 | 在 `tilelang-op-design` Phase 5 自检加一条机械核对：§3.3 伪代码中每个 v-op 的操作数切片须与 §5.4/§9.2 的静态 extent 约束逐行对照 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **跨循环维 `clear=False` 累加的行和累加器必须逐级复位**：把 `T.clear(red)` 放在 N-循环外一次、却在循环内跨块 `clear=False` 累加进 `gb`，会使 `gb = Σ_k Σ_{j≤k} rowsum(blk_j)`（前序块重复计入），`n_tile_N≥2` 全错。正解 = 行和累加器 `red/red2` 的 `T.clear` 置于 N-循环**内**每块一次、`gb/gw` 循环外一次 | D | DESIGN.md v0 §3.3（缺陷）→ v1 §3.3/§6.2/§6.4（修正）；REVIEW.md 问题 1；同族拆分核 `examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/DESIGN.md` §3.3（正确形态） | `python3 .agents/tools/statectl.py gate 1 --dir examples/batch_norm_bwd/_batch_norm_bwd_kernel`（读源码/结构核对，非真机）；L0-9 尾 N-block 门住 | tilelang 0.1.2+417dae99 + Ascend910B3 + CANN 8.5.2 | pattern-library/layout.md（分块累加复位）/ tilelang-op-design 自检清单 |
| **v-op 操作数切片须静态 extent（全 buffer）；运行时 slice 只喂 `T.copy`/`T.reduce(size=…)`**：`T.vcast(stage[0:rN,0:rS], g32[0:rN,0:rS], …)` 触发 parser `int(Var)`；v1 改 `T.vcast(stage, g32, …)`，尾块正确性 = 归约 `size` 截断 + dst 显式 slice | D | DESIGN.md v0 §3.3（缺陷）→ v1 §3.2/§3.3/§5.4/§9.2（修正）；`examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/RETROSPECTIVE.md`（Stage 3 实测首轮编译失败）；同族宽核 `.py` pass-1/pass-2 全 buffer vcast | 同族宽核 `--level all` 真机通过（全 buffer vcast + `size` 截断） | tilelang 0.1.2+417dae99 + Ascend910B3 + CANN 8.5.2 | docs/Tilelang.language/数据类型转换操作/T.vcast.md / traps-compiler（v-op 静态边界） |
| **`out_idx=[-1]`（`len(out_idx)==1`）返回标量张量而非列表**：`tilelang/jit/jit_npu.py` 的 `JitKernel_NPU.__call__` 在 `len(out_idx)==1` 时 `return full_args[self.out_idx[0]]`（张量），`len>1` 才返回列表。设计文档措辞「返回单元素列表」不准确；wrapper 的 `grad_x = program(...)` 写法正确 | R | 本设计 v0 §0.2/§3.3/§4.2/§10.5（措辞）→ v1（订正）；REVIEW.md 建议 5；`tilelang/jit/jit_npu.py` 的 `__call__` | `python3 .agents/tools/statectl.py gate 1 --dir examples/batch_norm_bwd/_batch_norm_bwd_kernel`（源码核对） | tilelang 0.1.2+417dae99 + Ascend910B3 + CANN 8.5.2 | tilelang-op-design 模板 §3.3（多输出 jit 返回约定） |

### Transferable Lessons（写给本迁移任务后续函数 / 同算子变体）

- **多级归约的累加器复位层次 = 累加所在循环的内层**：`T.clear(行和累加器)` 必须位于其累加目标维（此处 N-block）循环的**内部**每块一次；只有最终跨维累加目标（`gb/gw`）才在循环**外**清一次。凡「只清一次又跨块复用同一累加器」即双重计数缺陷；固定序累加的正确性与树序无关，复位层次错则全错。
- **设计自洽核对：§3.3 伪代码与 §5.4/§9.2 约束章节必须双向一致**：v0 一边把运行时 extent 喂 v-op、一边在约束章节禁止之——自检应逐行比对 v-op 操作数切片形态，而不是只看伪代码「像不像」。
- **`out_idx=[-1]` 的返回值是张量不是列表**：单输出 jit（`len(out_idx)==1`）直接返回张量；描述接口时按实际源码（`jit_npu.py`）写，避免 wrapper 解包误解。
- **held 档 pass-2 复用要在伪代码可见**：单 tile（`n_tile_N==n_tile_S==1`）时 epilogue 直接复用 pass-1 的 `g32/xh32`（单扫描，O3）；用工厂层 `if single_tile:` 分支显式写出，勿只留在散文说明里。
- **退化 shape 的兜底要落到实现层**：`L==1`（`N==1 且 S==1`）的 1 元素平面核内不可做标量广播——工厂 `scalar_plane` 守卫拒绝 + op/wrapper 路由（早退或受支持路径），并配 L0 回归项。
