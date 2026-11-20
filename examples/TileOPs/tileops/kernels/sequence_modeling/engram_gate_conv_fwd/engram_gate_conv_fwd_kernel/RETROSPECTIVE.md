# RETROSPECTIVE — `_engram_gate_conv_fwd_kernel`（migration-harness）

## Stage 1 (design) Retrospective — first_design — 2026-10-15

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | DESIGN.md 内引用 `docs/` 文档路径时易漏 `Tilelang.language/` 前缀（写作 `docs/创建操作/T.transpose.md`），触发 gate 1 `S1-REF-PATHS`（一次性 fail） | `examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/DESIGN.md` §1.6.3（修复前）→ `statectl gate 1` 报 path missing | tilelang-op-design skill Phase 7 输出前加一步「`docs/` 引用路径存在性自检」（或 gate 提示直接给正解路径） | R |
| Design-flow | 迁移任务的设计期结构性不确定构造（动态行索引 read / UB 仿射 shift / `if_then_else` 带 buffer）无法从源码+docs 判定，须实测；skill 未强制此步，本任务靠 conductor 预注入提示主动做 | `history_version/probe_*.py`；DESIGN.md §1.6.3 探针表 | tilelang-op-design skill 迁移分支建议「对需设备/编译验证的结构构造做 D-3 最小探针」 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `T.if_then_else` 带 buffer 操作数（动态行索引）在 npuir Developer codegen 段错误（EXIT=139 无诊断）；去掉 `if_then_else` 仅用 `T.max/T.min` clamp 索引即通过 | D | `examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/history_version/probe_c_bisect_dynrow.py`（V2 崩、V1 通）；DESIGN.md §0.6 R4/§1.6.3 | `python3 examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/history_version/probe_b_dynrow.py`（EXIT=139）/ `probe_c_bisect_dynrow.py`（V1 exit0, V2 exit-11） | tilelang 0.1.2+2562795b3a219bf7b97141f556a16e5e8ecff5f2（/mnt/workspace/Agentkj/build，USE_NPUIR=ON）/ Ascend910B3 / CANN 9.0.0 / 2026-10-15 | pattern-library/traps-compiler.md |
| npuir Developer 下 **UB 内仿射行偏移读**（`buf[i+p, j]`，p 编译期常量、i 并行索引）**合法**（conv 滑窗可在 UB 内实现，无需 GM 往返） | D | `history_version/probe_a_shift.py`（编译+运行成功）；DESIGN.md §0.6 R1/§1.6.3 | `python3 examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/history_version/probe_a_shift.py`（`PROBE_A_SHIFT_OK`） | 同上 | pattern-library/layout.md / elementwise.md |
| npuir Developer 下 **store 处 direct-cast 被 MLIR verifier 拒**（`hivm.hir.vcast` result 须 ranked tensor，实为 memref）——输出写回须先 `T.vcast` 到 dtype UB 再 `T.copy` | D | `history_version/probe_a_shift.py` 首版报错 `'hivm.hir.vcast' op result #0 must be variadic of ranked tensor ... got memref<1x4x256xf16, strided<...>>`；DESIGN.md §3.2/§3.3 注 2 | 首版 probe A（直接 `O[i,j]=T.cast(...)`） | 同上 | pattern-library/traps-compiler.md |
| `T.Parallel` 内**动态行索引读 GM**（clamp 后）在 npuir Developer 下合法（分块 tile 按运行时 `t0` 载入可行） | D | `history_version/probe_c_bisect_dynrow.py` V1；DESIGN.md §1.6.2 #3 | `python3 .../probe_c_bisect_dynrow.py`（V1 exit0） | 同上 | pattern-library/elementwise.md / traps-compiler.md |
| 纯 Vector 迁移算子（无 `T.gemm`）**不触发** `CG-2026-0022`（MIX Cube+shared 向量规约广播缺口）——设计期应以「源算子是否含 matmul」先行判定，纯 Vector 无需 MIX 探针 | P | DESIGN.md §1.6.3 MIX 可行性评估；对照 `pattern-library/cases.md` CASE-engram-decode-migration（含 `T.gemm` 才触发） | none | 同上 | pattern-library/cases.md / capability-gaps CG-2026-0022 |

### Transferable Lessons（可迁移教训）

- npuir Developer 下，**带 buffer 操作数的 `T.if_then_else` 会在 codegen 段错误（EXIT=139，无诊断）**；需要「按条件取值/清零」时，改用**索引 clamp（`T.max/T.min`）+ 算术掩码**（`T.vbrc` 广播行标量 + `T.vmul` 乘 0），不要用 select 语义。
- npuir Developer 下，**GM 写回不要用 store 处 `T.cast`**（MLIR verifier 拒 `hivm.hir.vcast` memref result）；先 `T.vcast` 到目标 dtype 的 UB buffer，再 `T.copy` 落 GM。
- UB 内的**仿射行偏移读**（`buf[i+p, j]`）合法——序列轴滑窗（因果卷积）可在 UB 内完成，无需把中间量落 GM 再从 GM 读邻居行（可省一次 kernel + 一轮 GM 往返）。
- `T.Parallel` 内**运行时行索引读 GM**（clamp 后）合法——序列分块 tile 可按运行时 `t0` 载入；但**配合 `if_then_else` 即崩**（见上）。
- 迁移设计前**先判定源算子是否含 `T.gemm`/matmul**：纯 Vector 算子不触发 `CG-2026-0022`，无需做 MIX 前置探针；含 Cube 才需。
- 设计期结构探针（最小张量、单次编译，约 1–2 分钟/个）对迁移结构决策的 ROI 极高：本任务 4 个探针即锁定了「分块 tile 载入 / UB 滑窗 / 边界掩码 / 输出写回」四条关键实现约束，避免了 Stage 3 的整轮返工。
- 引用 `docs/Tilelang.language/` 下文档时**写全 `Tilelang.language/` 路径**（子目录名单独作前缀会被 gate `S1-REF-PATHS` 判为不存在）。

## Stage 2 (review) Retrospective — first_review — 2026-10-15

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | D-5 机械复核（`design_calc_check.py`）对本设计**近乎全部 skip**（`ub_budget`/`l0c_budget`/`core_split` = skip，仅 `r3_metrics` pass），检视须回退人工复算 §4.5 UB 预算与 §5.5 分核算术——机械拦截位空转 | REVIEW.md「机械复核」段（本任务）；`python3 .agents/tools/design_calc_check.py --design examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/DESIGN.md` 输出 3 skipped | `design_calc_check.py` 解析器扩形：支持模板 §4.3「Buffer 名×Shape×dtype×层级」表与 §5.5「逻辑核数」表（纯 Vector 算子另需 `×2` 翻倍规则），否则 advisory checker 对非 GEMM 类设计覆盖不足 | R |
| Info-source | D-3 探针脚本在算子目录下直接 `python3 probe_*.py` 会 import 到 site-packages 版 tilelang（`module 'tilelang.language' has no attribute 'vcast'`），须显式 `PYTHONPATH=<repo_root>` 才用 dev-root build；探针未自带 sys.path 固化 | `history_version/probe_a_shift.py` 无 PYTHONPATH 运行 → `AttributeError: ... has no attribute 'vcast'`（首跑）；加 `PYTHONPATH=/mnt/workspace/Agentkj` 后 `PROBE_A_SHIFT_OK` | tilelang-op-design/tilelang-design-review 的 D-3 探针模板在脚本头固化 dev-root 到 `sys.path`（或注明必须带 `PYTHONPATH` 运行） | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 迁移任务描述（`op_requirements` / 用户 prompt）可能带**规格错误**（本任务把无膨胀 `EngramGateConvFwdOp` 描述为 dilated），检视须**亲自读源码**核对而非采信任务描述——同族 `engram_decode` 才含 dilation | P | `examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/DESIGN.md` §0.1 订正段；源 `engram_fwd.py` L146 `src_t=tid-(K-1)+p`（无 dilation）vs `engram_decode` L127 `(w-1-p)*dilation` | none | tilelang 0.1.2+2562795b3a219bf7b97141f556a16e5e8ecff5f2 / Ascend910B3 / CANN 9.0.0 | pattern-library/cases.md / tilelang-design-review skill 维度 0 |
| 纯 Vector 迁移设计（无 `T.gemm`）经**设备实测**确认不触发 `CG-2026-0022`：本容器探针 A/V1/V3/V4 全通过、V2 段错误，与设计期结论逐条吻合（D-3 可复现） | D | `examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/REVIEW.md`「机械复核」段（本任务复跑）；`history_version/probe_c_bisect_dynrow.py`（V1/V3/V4 exit0, V2 exit-11） | `PYTHONPATH=/mnt/workspace/Agentkj python3 examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/history_version/probe_c_bisect_dynrow.py` | tilelang 0.1.2+2562795b3a219bf7b97141f556a16e5e8ecff5f2 / Ascend910B3 / CANN 9.0.0 / 2026-10-15 | pattern-library/traps-compiler.md |

### Transferable Lessons（可迁移教训）

- **检视维度 0 必须亲读源算子源码**：迁移任务描述可能携带规格错误（本任务「dilated」实为无膨胀）；判别法是回到源码计算语句（卷积 tap 偏移是否含 dilation 因子），并与同族算子对照以确认订正方向。
- **D-5 机械复核 skip ≠ 通过**：`design_calc_check.py` 对非 GEMM / 纯 Vector 设计常解析不到 `ub_budget`/`l0c_budget`/`core_split`（本例 3/4 skip），检视须回退人工复算并把「机械跳过，人工已核」写进 REVIEW.md。
- **D-3 探针复跑须固化 tilelang 来源**：探针脚本要在脚本头把 dev-root 加入 `sys.path`（或运行时带 `PYTHONPATH=<repo_root>`），否则会静默 import 到 site-packages 版 tilelang 而报 `T.vcast` 等 API 不存在（伪失败）。
- **D-1 等价性脚本重跑要核「内嵌表 ⊂ 实跑输出」**：DESIGN.md §1.6.1 表可为实跑输出子集（本例 14/22 行），判定一致性时须核对交集行数值逐位吻合 + 实跑无 FAIL，而非行数相等。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-15

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 设计 §3.3 的标量输出写法（`for i in T.Parallel(BLK_S): buf[bid, t0+i] = rh[HALO+i, 0]`）在 npuir Developer codegen 下**不可靠**（同输入多次运行结果波动）；skill 未提示「`(R,1)` reduce buffer 的 own-rows 逐行取数 + GM 写回」是高风险构造，须实测 + 确定性复跑 | `_engram_gate_conv_fwd_kernel.py` 实现过程：T.Parallel fill 时 alpha/rrms_h/k/v 的 bad 元素在 3–28 间波动（本文件实现过程的多次同输入复跑记录），改 unroll fill 后稳定 0；REVIEW.md 未覆盖此风险 | tilelang-op-develop skill 增加检查项：「迁移算子的标量/小 buffer GM 写回须做确定性复跑（同一输入多次运行比对）」 | R |
| Info-source | 设计期 D-3 探针覆盖了「载入 / UB 滑窗 / 因果掩码 / 写回」四个大构造，但**未覆盖「`(R,1)` reduce → own-rows 标量输出」写回子形态**，Stage 3 才发现其不确定性；探针清单宜按「每个 GM 写回形态」而非「每个结构」枚举 | DESIGN.md §1.6.3 探针表（A/V1/V2/V3）；Stage 3 实测 | tilelang-op-design 探针建议增加「标量输出（`(blk,1)` → `(M,S)` 写回）形态」条目 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| npuir Developer 下，从 `(R,1)` reduce buffer 取 own-rows 写 `(M,S)` GM：**`T.Parallel` 元素级 fill + `T.copy`、以及 `T.copy` 区域切片取数均出现 nondeterministic 结果**（同输入多次运行 bad 元素在 3–28 波动）；改用**编译期 unroll 的标量 fill（`for i in range(blk): stg[i,0]=rh[halo+i,0]`）+ 单次 `T.copy`** 后完全确定（0 bad） | D | `_engram_gate_conv_fwd_kernel.py`（Stage 3 attempt 1，drv4 多 seed 复跑对比）；`repro/TRAP-npuir-parallel-scalar-store-race.py` | `PYTHONPATH=/mnt/workspace/Agentkj python3 examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/repro/TRAP-npuir-parallel-scalar-store-race.py` | tilelang 0.1.2+2562795b3a219bf7b97141f556a16e5e8ecff5f2 / Ascend910B3 / CANN 9.0.0 / 2026-10-15 | pattern-library/traps-compiler.md |
| npuir Developer 下，**1-D `T.Parallel` 标量 store 内使用运行时 `T.min`/`T.max`（运行时 `t0`）导致 codegen 段错误（SIGSEGV, exit 139）**（先打印 `Parallel cast lowering uses fixed round mode 'rint'`）；编译期 unroll 的标量语句形式正常编译运行 | D | 本文件实现过程的最小探针；`repro/TRAP-npuir-parallel-runtime-minmax.py`（parallel exit=-11 / unrolled exit=0） | `PYTHONPATH=/mnt/workspace/Agentkj python3 .../repro/TRAP-npuir-parallel-runtime-minmax.py` | 同上 | pattern-library/traps-compiler.md |
| npuir Developer 下，`T.vcast`（含 `T.Parallel` 内元素级 `T.cast` 的 lower 结果）要求 **src/dst buffer shape 完全相同**；shape 不等（如 `(R,d)→(BLK,d)`）在 `codegen_npuir_dev.cc::getBroadcastDim` 报 `Check failed: (*as_const_int(buffer_shape0[i]) == *as_const_int(buffer_shape1[i]))` | D | `_engram_gate_conv_fwd_kernel.py` 实现过程（首版 `oy[i,j]=T.cast(xt[i,j],dtype)` 失败）；`repro/TRAP-npuir-vcast-shape-mismatch.py` | `PYTHONPATH=/mnt/workspace/Agentkj python3 .../repro/TRAP-npuir-vcast-shape-mismatch.py` | 同上 | pattern-library/traps-compiler.md |
| npuir Developer 下，**元素级 GM 写回**（`Y[bid,t0+i,j]=oy[i,j]` 或 1-D `T.Parallel` 标量 GM store）**不可靠**；`T.copy` 区域写回（`T.copy(oy, Y[bid,t0:t0+blk,0:d])`）稳定正确——分块 kernel 的输出写回应一律走 `T.copy` | D | `_engram_gate_conv_fwd_kernel.py`（实现过程中元素级 store 版本错误、改 `T.copy` 后 Y/vhat 恒定正确）；`repro/TRAP-npuir-parallel-scalar-store-race.py` control 分支 | none | 同上 | pattern-library/traps-compiler.md |
| tile 宽选择改「**收缩 tile 宽直到整除 `seq_len`**」（必要时单 tile 覆盖整条序列）替代设计期「尾块 predicated store」；`blk_s=1` 不可用（`T.vbrc` empty broadcast dims）——本实现以 `cap∈{8,4}` 逐个整除性检查 + 单 tile 兜底，seq_len∈{13,7,9,3} 全部通过 | P | `_engram_gate_conv_fwd_kernel.py` `blk_s` 选择逻辑；Stage 3 实测 | `python3 .../_engram_gate_conv_fwd_kernel.py --level all`（L0-6 seq_len=13、L1-nondiv seq_len=7、B-S3/B-S9 通过） | 同上 | pattern-library/layout.md / traps-compiler.md |

### Transferable Lessons（可迁移教训）

- npuir Developer 下，把 `(R,1)` 行标量 buffer 的 own-rows 写到 `(M,S)` GM 时，**不要用 `T.Parallel` 元素级 fill，也不要用 `T.copy` 区域切片取数**——两者都会产生 nondeterministic 结果；用**编译期 unroll 的标量语句 fill 到 `(blk,1)` 暂存 buffer，再单次 `T.copy`** 落 GM。
- npuir Developer 下，**1-D `T.Parallel` 标量 store 内禁止出现运行时 `T.min`/`T.max`**（会 SIGSEGV, exit 139）；行掩码等标量式一律用**编译期 unroll（`for i in range(R)`）的标量语句**表达。
- npuir Developer 下，`T.vcast` / `T.Parallel` 内元素级 `T.cast` 的 lower 要求 **src/dst buffer shape 严格一致**；跨 shape 写回时先 `T.copy` 到同形 buffer 再 `T.vcast`，或改成 unroll 标量 cast。
- npuir Developer 下，分块 kernel 的**所有 GM 写回都用 `T.copy` 区域写回**；元素级 GM store（含 `Y[bid,t0+i,j]=...`）不可靠。
- 分块算子的 tile 宽优先保证 **`seq_len % blk_s == 0`**（对候选逐个整除性检查，必要时单 tile 覆盖全序列）；不要依赖 predicated 尾块 store。`blk_s==1` 因 `T.vbrc` empty-broadcast 不可用。
- 实现设计伪代码时，若某写法在真机上表现为**同输入多次运行结果不同**，优先怀疑「v-op 结果 → 标量/MTE 读取」的隐式同步缺失，改用编译期 unroll 标量路径或对整 buffer 做 `T.copy`。
- 本实现 UB 未触发 `ub overflow`（编译无该报文）：d=512 用 `BLK_S=4`（`R=7`），d=256 用 `BLK_S=8`（`R=11`），`(R,d)` f32 常驻 buffer 由设计的 8 个减到 5 个（复用 `prod` 作 `v_hat_norm`、复用 `sig` 作 vhat 输出暂存），UB 峰值因此低于 192KB。
