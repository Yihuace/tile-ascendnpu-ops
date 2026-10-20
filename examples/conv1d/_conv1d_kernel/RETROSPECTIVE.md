# Stage 1 复盘 — `_conv1d_kernel`（Conv1dFwdOp 函数 2/5，一般 implicit-GEMM conv）

## Stage 1 (design) Retrospective — first_design — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 设计期实查物理核数必须用 `from tilelang.utils import NPUUtils`（`from tilelang.utils.npu import ...` 会 `ModuleNotFoundError`）；本任务实查 `get_aicore_num()=20`（Ascend 910B3，≠ `CONST-aicore-910B2C` 记录的 24） | `python3 -c "from tilelang.utils import NPUUtils; NPUUtils.get().get_aicore_num()"` → 20 | 与函数 1 复盘条目一致，无需新增（函数 1 已记录到 core-split-strategy.md） | D |
| Design-flow | 门禁 `S1-REF-PATHS` 对**连写的两个目录词**（`examples/docs`）机械判为伪路径——写作「API 均有 examples/docs 佐证」时 `examples/docs` 被 REF_PATH_RE 捕获为一条不存在的仓库路径 | 本任务首次 gate 1 失败 1 条 `S1-REF-PATHS`（`examples/docs`）；改为「本仓 examples 与 docs」后通过 | design 模板/quality-checklist 增补：引用目录佐证时禁止把两个仓库目录词用 `/` 直接连写（`examples/docs`、`docs/examples`）；应在两词间留空格或改用「A 与 B」 | R |
| Design-flow | 一般 conv（`kW>1`）的 GPU host `weight.permute(0,2,1).contiguous()` 是**内容重排**（非 metadata view），与 pointwise（`weight[:,:,0]` 是 view）本质不同；若照抄 pointwise 的「host 视图保留」判定会违反 ascend-constraints.md §4 | 源 L870 vs 函数 1 §0.1（pointwise `.contiguous()` 是 no-op） | migration-analysis.md 补一条判别式：host 侧 `.contiguous()` 仅在「原视图已连续（如尾维 size=1 的 squeeze）」时为 metadata no-op，否则须内移 kernel（本任务 §0.6 R3） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **一般 conv（kW>1）隐式 GEMM 的 A 侧须 kernel 内 weight-layout 前置**：GPU host `weight.permute(0,2,1).contiguous()`（`[C_out,C_in,K_W]→[C_out,K_W,C_in]`）是内容重排，NPU host 禁止；改用 kernel 内 Vector `T.copy`(GM→UB)+`T.transpose(perm=[0,2,1])`(UB 2 轴交换)+`T.copy`(UB→GM) 产 `w_flat`（一次性 `O(c_out·c_in·K_W)`）。K 序选 `(k_w,c_i)`（与 per-tap slice B 匹配）；替代 K 序 `(c_i,k_w)` 会让 B 每次 per-tile 物化（净亏） | P | `examples/conv1d/_conv1d_kernel/DESIGN.md` §0.6 R3、§1.6.3 候选 #1/#2 | `examples/conv1d/probe/conv1d_geomm_probe.py`（探针用 host `permute` 造 `Wf`，即此步的 GPU 形态） | tilelang 0.1.2+417dae99 + Ascend910B3（`get_aicore_num`=20）+ CANN | pattern-library / cases.md |
| **stride>1 的 im2col 列采样无法用 slice 表达（ND2NZ 要求最内维连续）→ polyphase 等价分解**：`x[ci,ol·s+c]` 内层步长 `s`，GM→L1 slice 会静默平坦误读（CG-2026-0004 同族）；polyphase `y_poly[ci,r,ol+q]`（`(q,r)` 唯一分解 `c=q·s+r`）把列采样移出 DMA 通路，后续 per-tap 连续 slice（同 stride=1）。可作常量相位表验证 | P | DESIGN.md §0.6 R4、§1.6.0 R1 候选 6、§1.6.3；源逐元素 gather L86-103 | `examples/conv1d/_conv1d_kernel/DESIGN.md` §3.3（`r_kw`/`q_kw` 相位）；smoke `test_conv1d_no_bias_matches_torch`（stride=2,pad=2）为该路径 L0 用例 | 同上 | pattern-library / cases.md |
| **跨步/重叠 im2col（stride=1）的 B 装载可用 per-tap slice `T.copy`（最内 L 连续）**——K 序 `(k_w,c_i)` 下每 tap 一行矩形；dilation 只平移起点、padding 只影响负偏移，均不破坏最内连续 | P | DESIGN.md §0.6 R4、§1.6.3 候选 #1 | `examples/conv1d/probe/conv1d_geomm_probe.py` L94-98（`xp[0:C_IN, bx*BLOCK_N+kw : ...+BLOCK_N] → l1_b[kw*C_IN:...]`） | 同上 | pattern-library / cases.md |

### Transferable Lessons（可迁移教训）

- **一般 conv 的 A/B 两个操作数之间存在「布局冲突」**：B（im2col）的连续 slice 要求 K 序 `(k_w,c_i)`（`c_i` 最内），而 weight 原生布局 `[C_out,C_in,K_W]` 的最内是 `k_w`；二者不能同时免重排——必须在「A 一次性 `T.transpose`」与「B 每次 per-tile 物化」二选一，前者净胜。设计时先固定 B 的 slice 结构，再决定 A 的重排方式。
- **`contiguous()` 的合规性取决于原视图是否已连续**：pointwise（尾维 size=1 的 squeeze）的 `.contiguous()` 是 no-op（metadata，合规）；一般 conv 的 `permute(0,2,1).contiguous()` 是真实转置（内容重排，违规）——迁移动手前必须判明。
- **stride>1 的窗口列采样是 slice 的天敌**：GM→L1/UB 的 slice 末维只能步长 1；一切「按 `s` 抽头」的采样（strided conv / 转置读 / polyphase）都须先物化连续副本或在 UB 域重排，不能指望 DMA 跨步。polyphase（把步长折进一个显式维度）是把 strided 采样还原为连续 slice 的通用手法。
- **门禁 REF_PATH_RE 会把相邻目录词连写当路径**：写「examples/docs」「docs/examples」必踩 `S1-REF-PATHS`；佐证两个目录时写「本仓 examples 与 docs」。
- **weight-layout 前置与 polyphase 物化的 UB 预算须按 `c_in·K_W` 单项评估**：`bm_w=16` 时 `2·16·c_in·K_W·2B` 在 `c_in·K_W=1280`（encodec-deep）达 80KB，逼近但未超 192KB；大 tile 需降 `bm_w` 或分段。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | DESIGN §3.3 的 polyphase 物化用 `T.Parallel(Lg, stride)` 逐元素写 GM `y_poly`；**Expert 模式 Vector 路径下 `T.Parallel` 完全不可用**（GM 写触发 `NPU IR opt failed` 编译器崩溃；先 gather 进 UB 再拷贝则 `output object was not generated`），设计直译必然失败 | 本任务探针：GM 直写 `T.Parallel` → `NPU IR opt failed ... crash backtrace`；UB 中转 → `AscendNPU IR compile reported success but output object was not generated` | `tilelang-cube-skill`/`tilelang-mixcv-skill` 增一条：Expert Vector 段禁止 `T.Parallel`（逐元素）——跨步采样须用 `T.gather`（原生 VGatherOp）或 `T.deinterleave`；design 模板 polyphase 伪代码同步 | D |
| Develop-flow | DESIGN §3.3/§6 伪代码用 `for kw in T.serial(kernel_l)`（运行时循环变量）却索引 Python 相位表 `r_kw[kw]`/`q_kw[kw]`——**tilelang 里 `range()`/`T.serial` 的循环变量是 tir.Var，不能索引 Python list**（`list indices must be integers or slices, not Var`）；且宏/普通 Python 辅助函数内含 `with T.Scope`/`T.serial` 会被 TVM script parser 误解析（`'ForFrame' object is not iterable`），必须用 `@T.macro` 或完全内联 | 实测 `list indices must be integers..., not Var`；普通函数内 `with T.Scope` → parser 报 `ForFrame not iterable`；改 `@T.macro` + 相位改运行时算术后通过 | design 模板 §3.3 自检：伪代码中凡运行时循环变量索引的编译期表，须改写为运行时算术（`(kw*d)%s`）或在 Python 层展开；跨 scope 复用代码须 `@T.macro` | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **Expert 模式 GM→UB 的「UB 目标列偏移切片」静默错值**：`T.copy(src[M,N], ub[M, PAD:PAD+N])`（`ub` 分配宽度 > N）只正确写第 0 行，行 1..M-1 落到错误 stride（1e0 量级错值，无编译错误）。修法：把列位移放到 **GM 侧**——源连续拷入同形 UB，再写 GM 切片 `gm[0:M, PAD:PAD+N]`（GM 偏移目标正确）。本算子的 `x_pad` 构造即此形态 | D | Stage 3 实测：UB-offset 变体 max_abs=5.39、15/16 行错；GM-offset 变体 max_abs=0 | `examples/conv1d/_conv1d_kernel/repro/TRAP-copy-ub-offset-stride.py`（GM-offset 断言通过；`--show-bug` 复现错值） | tilelang 0.1.2+417dae99 + Ascend910B3（`get_aicore_num`=20） | pattern-library / traps-compiler.md |
| **`T.gemm(initC=...)` 的 `&` 组合条件触发编译器段错误**：`initC=((kc==0)&(kw==0))`（两布尔 tir 表达式按位与）使 lowering 崩溃（segfault）；单比较 `initC=(k==0)` 正常。等价改写为算术单比较 `initC=(kc+kw==0)` 即通过 | D | Stage 3 实测：`&` 变体 segfault（EXIT=139）；`(kc+kw==0)` 通过且精度正确 | `examples/conv1d/_conv1d_kernel/repro/TRAP-copy-ub-offset-stride.py` 邻近的最小 gemm 核对（见 `probe20`/`probe16`） | 同上 | pattern-library / traps-compiler.md |
| **Expert Vector 段无逐元素 `T.Parallel`；跨步/解交织采样用 `T.gather`（+ 主机常量索引）**：polyphase 物化 `xdec[bn,ci,ρ·Mdec+m]=x_pad[bn,ci,m·s+ρ]` 用一次 `T.gather(ubx, ubd, host_idx, [C,l_pad])` 完成（索引 `idx[m']=(m'%Mdec)·s+m'//Mdec` 由 host 预置 int32，避免 UB 内建索引的标量写失效）。B 装载随后变成 `xdec` 的连续 per-tap slice | P | Stage 3：`T.Parallel` GM/UB 两形态均编译失败；`T.gather` 三组用例 max_abs=0；集成后 stride>1 L0/L1 全过 | `examples/conv1d/_conv1d_kernel/_conv1d_kernel.py` §prelude（gather decimation） | 同上 | pattern-library / cases.md |
| **tap-major K 分块 + L1 offset-0 装载**：per-tap 在 L1 中的**列/行偏移**（`kw·c_in`）对非 16 对齐 `c_in`（如 50/1/24）不可寻址（`ADDR_MISALIGN`）。改为「一个 tap = 一个 K 子块、目标 L1 偏移恒为 0」（`block_k` 沿 channel 再切），K 序仍 `(k_w,c_i)` 且累加仍 fp32；`c_in` 补零到 16 倍数（B 补零行使 A 补零列的垃圾值无贡献） | P | Stage 3：flat 偏移变体（probe9, c_in=24）`ADDR_MISALIGN`；tap-major offset-0（probe12/16c）通过 | `examples/conv1d/_conv1d_kernel/_conv1d_kernel.py` §`_cube`（tap-major gemm） | 同上 | pattern-library / cases.md |

### Transferable Lessons（可迁移教训）

- **Expert 模式 Vector 段不要用 `T.Parallel` 做逐元素 GM/UB 写**——它在 npuir Expert 下不可用（编译崩溃或 output 未生成）。跨步/解交织采样改用 `T.gather`（原生 VGatherOp，需 int32 索引，索引张量由 **host 预置**最稳），或 `T.deinterleave`。
- **`range()`/`T.serial` 的循环变量是 tir.Var，不能索引 Python list**；相位/常量表必须在 Python 层展开（`for kw in range()` 会被 parser 变成 serial，不能展开），或改写成运行时算术（`(kw*d)%s`、`kw//s`）。
- **跨 scope 复用的 kernel 代码必须用 `@T.macro`**；普通 Python 辅助函数若内含 `with T.Scope`/`T.serial`/`T.gemm` 会被 TVM script parser 误解析（`ForFrame not iterable`）。宏内引用闭包常量（`has_bias`/`need_poly`）由 parser 折叠，OK。
- **GM→UB 拷贝的 UB 目标切片带列偏移时会静默错值**（只第 0 行对，见 repro）；把列位移改放到 **GM 目标**侧（源先连续入同形 UB，再写 `gm[..., PAD:PAD+N]`），或 host 侧预零 + 只写有效区。
- **`initC=((a==0)&(b==0))` 会让 lowering segfault**；用单比较 `(a+b==0)`（两非负量之和为零 ⇔ 均零）或其它单比较等价式。
- **tap-major K 分块（一个 tap = 一个 K 子块，L1 目标偏移恒 0）可规避 per-tap L1 偏移对齐问题**（非 16 对齐 `c_in` 的 `kw·c_in` 偏移不可寻址）；K 序与 fp32 累加语义不变，仅分块边界变。
- **`cid` 钳位 + persistent + 深度 2 双槽 4-flag 握手（消费者侧前导）在本算子复用函数 1 修正后直接可用**：同引擎自 set→自 wait 死锁（aicore timeout 507014）的修法仍是「前导置于消费者（Vector）」。本算子 Vector 前置段（layout/x_pad）+ Vector bias 段 + Cube 段的三段结构工作正常。
- **L2 warn 层 `c_in=256`（`c_in_pad=256`, `n_kc=2`）触发 BiShengHIR pipeline 失败**（warn-only，不阻塞）；L0/L1 覆盖 `c_in≤128` 全部通过。大 K 的边界需 Stage 4 复核。
