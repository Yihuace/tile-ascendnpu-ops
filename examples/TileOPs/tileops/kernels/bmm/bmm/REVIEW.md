# _bmm_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/bmm/_bmm_kernel/DESIGN.md（revision v1）
- 任务类型: 迁移（harness，Cube 批量矩阵乘族）
- 源算子: `/tmp/opencode/TileOPs-ref/tileops/kernels/bmm.py`（`_bmm_kernel` L21–L118 / `BmmKernel` L621–L684）+ `/tmp/opencode/TileOPs-ref/tileops/ops/bmm.py`（`BmmFwdOp`）；本仓 Stage 0 抽取件 `examples/TileOPs/tileops/kernels/bmm/bmm/_bmm_fwd_kernels.py`（L15–L113）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `python examples/bmm/_bmm_kernel/verify_equiv.py` → exit 0，逐行输出 `OPT-1 initC fusion  max_ulp 0  viol_rate 0.0000  EQUIV_PASS` / `OPT-2 epilogue direct  max_ulp 0  viol_rate 0.0000  EQUIV_PASS`。**与 §1.6.1 内嵌结果表（L294–L299）逐行一致，全 EQUIV_PASS**（用例含 random / ±inf / NaN / subnormal / rounding-halfway ties / 边界 shape）。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/bmm/_bmm_kernel/DESIGN.md` JSON 逐项：
  - `l1_budget`: **pass**（`L1: 1 buffer entries sum = 128.0 KB; capacity = 512.0 KB`）；
  - `l0c_budget`: **pass**（`block 128x128 x fp32 accum = 64.0 KB vs L0C 128 KB`）；
  - `core_split`: **skip**——`insufficient parsed dims {'bm':128,'bn':128}; declared logical cores = 1`。机械解析器无法解析含 batch 维的一维折叠网格（`grid = B·nt_m·nt_n`）。**人工已核**（见维度 3：§5.5 三要素齐全、物理核数实查 20、核数与 block 取值自洽、极大规模方案有据）；
  - `r3_metrics`: **pass**（`all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)`）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass
已亲自 Read 源算子全文（`tileops/kernels/bmm.py` L21–L118 `_bmm_kernel`、L621–L684 `BmmKernel`；`tileops/ops/bmm.py` `BmmFwdOp`）与本仓 Stage 0 抽取件（`_bmm_fwd_kernels.py` L15–L113），与 §0.1–§0.7 逐项核对：

- **语义理解正确性（pass）**：§0.1「严格 3D-3D、无广播 `C[b]=A[b]·B[b]`」↔ 源码 `_bmm_main`（L64–L67）与 docstring L27–L32；§0.1「沿 K 分块（`block_k`）顺序累加、累加器 fp32（`accum_dtype="float"`）、每输出 tile 由单 block 独立算完、无跨 block 归并」↔ 源码 L50 `accum_dtype = "float"` + L90–L105（`T.clear` 后逐 `ki` 累加，无 split-K）；§0.1「输入 fp16/bf16、输出 `same_as(a)`、`out_idx=[-1]` 框架分配」↔ L52–L56（`@tilelang.jit(out_idx=[-1], ...)`）+ L65–L67；§0.1 边界语义三项逐条命中：batch/K 相等与 rank 校验（`BmmFwdOp._infer_bmnk`）、`K%16==0`（源码 `BmmKernel.__init__` L641–L643 `if k % 16 != 0: raise ValueError`）、**M/N 允许非整除**（源码 docstring L31–L32 "the epilogue guards the M/N tails so m/n need not be multiples of the block sizes"）。**无一处与源码矛盾**。
- **I/O 一致（pass）**：§0.2 `a:(B,M,K)` / `b:(B,K,N)` / `c:(B,M,N)` 与源码 `T.Tensor` 签名 L65–L67 完全一致；「NN 布局、`transpose_B=False`」经核对成立——`_bmm_kernel` 的 `T.gemm`（L104–L105）未传 `transpose_B`，而 `transpose_B=True` 仅出现在 **fp8** 变体（L213/L302），设计未把 fp8 语义混入；manifest `shape_rules`（`examples/TileOPs/tileops/manifest/bmm.yaml`）确认 `b.shape[2]==a.shape[1]`（N 连续）与 `a.shape[2]%16==0`。
- **算法解读完整性（pass）**：§0.3 七步表逐句覆盖源码全部计算语句——`T.clear(c_local)`（L90）、3D grid（L69–L73）、`T.use_swizzle(10, enable=True)`（L88）、`T.Pipelined`+2×`T.copy`（L94–L103）、`T.gemm(...,policy=FullRow)`（L104–L105）、`T.copy(c_local, c_smem)`（L111）、`T.Parallel` 谓词写（L112–L114）；**无遗漏、无臆造**。数据流图覆盖 GM/SMEM/fragment 全部 buffer；host 侧（`functools.lru_cache` L21、`default_config`/`autotune_configs` L652–L677、`torch.library.custom_op` 包装 L594–L618）已列出并标注 Stage 0 承接。
- **优化手段覆盖完整性（pass）**：§0.4 十项与源码显式优化逐条对应，含易漏项 `make_swizzled_layout`（源码 L79–L83，归入 §0.4#5 SMEM swizzle）与 `T.use_swizzle(10)`（L88 → §0.4#4）；优化手段与实现层手段的层次标注合理（#10 `out_idx`/`lru_cache` 标为 host 层）。
- **耦合性判定合理性（pass；抽检 4 项）**：§0.5 每项均含四态处置 + 依据。抽检：① #4 `T.use_swizzle` 舍弃 → 依据「`T.Kernel.md` §2.1 参数表仅 blocks/threads/is_cpu/prelude/is_npu/pipeline」，已亲自打开 `docs/Tilelang.language/编译器提示操作/T.Kernel.md` 核对，参数表**确无 swizzle**，负向断言成立；② #8 `threads` 舍弃 → `TRAP-threads-kwarg-noop` 实测条目存在（`pattern-library/traps-compiler.md` L124，repro 齐全），且源码 L61 `threads: int = 128` 确实透传该 kwarg；③ #5 SMEM swizzle 舍弃 → 依据 `tilelang-cube-skill` nz-format.md「Developer 保持 ND」成立；④ #6 epilogue 重新设计 → Developer 机制（VCast）已按文档改正（见维度 8）。处置总量（保留 2 / 等价替换 3 / 重新设计 3 / 舍弃 4）与源码手段数自洽。
- **NPU 重设计可行性（pass）**：§0.6 R1–R4 每项含「源方案 → NPU 新算法 → 语义保持论证」。R1 一维折叠网格是 `cid ↔ (bz,by,bx)` 双射重编号（任务集合/计算/写回位置不变），满足「NPU ≤ 2D block」（`T.Kernel.md` §2.3 原文「最多支持 2 个 block 维度」）；R2 slice 形态 `T.copy` + `T.min` 动态 extent 满足 NPU 约束；R3 `initC` 融合经 §1.6.1 机器验证 max ulp 0；R4 移除 GPU 专属手段且逐项标注去向。附设计期探针 A–E 实测（脚本 `history_version/design_probe_bmm*.py` 已核对存在）。
- **设计与决策一致性（pass）**：§1.4/§1.5/§3.1/§3.3/§4.4/§6 描述的是 §0.6 重设计后的 NPU 算法（一维折叠网格 + L1 tiling + K 流水 + L0C→GM 直写），**无「§0 判重新设计、§3 照抄源方案」脱节**：R1 折叠网格 ↔ §3.3 伪代码 L423–L427；R2 切片 epilogue ↔ §3.3 L442–L443；R3 `initC` ↔ §3.3 L440；R4 去 GPU 手段 ↔ §3.3 无 `use_swizzle`/`threads`。
- **golden 独立性（pass）**：§8.1 `golden_bmm` 直接复用 `torch.bmm`（源仓参考 `test_bmm.py` L33 `ref_program`），**未复刻 §0.6 的 NPU 数据流**（无折叠 decode、无 `initC`、无 slice 裁剪），以 §0.1 语义为唯一依据，独立性成立。

### 1. API 可行性 — pass
§3.2/§3.4 列出的每条 API 均在 `examples/` 或 `docs/` 找到佐证：`T.Kernel(...,is_npu=True)`（`T.Kernel.md` §2.4 示例1/3）、`T.copy`（`T.copy.md` §2.1/§2.4；`examples/gemm/example_gemm.py` L28–L29）、`T.gemm(initC=)`（`T.gemm.md` §2.1/§2.4 L74 + `example_gemm.py` L30）、`T.alloc_shared`/`T.alloc_fragment`（`T.alloc_shared.md` §2.4 + `example_gemm.py` L23–L25）、`T.Pipelined`（`example_gemm.py` L27）、`T.min` 动态 extent（`T.copy.md` §2.2.2 第 2 条）、`out_idx=[-1]`（`example_gemm.py` L8）。
- **模式匹配（pass）**：`examples/gemm/example_gemm.py` 是同构的 **Developer 先例**（`os.environ["TILELANG_ASCEND_MODE"]="Developer"` L40；`T.alloc_shared`→L1 + `T.Pipelined` + `T.gemm(initC=(k==0))` + `T.copy(C_local, C[...])` epilogue），与 §3.3 伪代码结构一致；§7 采用 Developer 自动同步、无手写 set_flag/wait_flag，与模式匹配。
- **v-prefix 优先（pass）**：本算子全部计算落 `T.gemm`（Cube），无逐元素向量算子需求；§1.6.2 已论证无逐元素标量热点，故无 v-prefix API 适用点，符合「无可替代点即说明」的口径。
- **机制前提核对（本次重点复核）**：§3.2 步骤 3「GM→L1 的**显式** ND2NZ 为 Expert 路径（`T.copy.md` 支持方向表第 14 行）」、步骤 5「L0C→GM 经 Developer VCast 通用路径（第 15 行）+ §2.3 第 4 条」——亲自打开 `docs/Tilelang.language/内存操作/T.copy.md` 核对：支持方向表 L14 `GM -> L1 | ×（用 T.load_nd2nz） | √ 自动转 ND2NZ`、L15 `L0C -> GM | √（经 VCast 通用路径） | √ 自动转 Fixpipe`；§2.3 第 4 条「Developer 模式支持跨 dtype：编译器自动插入 `hivm::VCastOp`（RoundMode=RINT）」、第 6 条「Expert 模式 GM→L1 / L0C→GM 的 `T.copy` 等价 `T.load_nd2nz`/`T.store_fixpipe`」。**设计引用与文档逐条一致**（上一版的 Expert 条款错位已改正）。

### 2. 内存层级规划 — pass
- **搬运路径完整**（§4.4）：`GM[a/b] → L1[a_s/b_s] → L0C[c_l] → GM[c]`，无 GM→L0 越级；每个 buffer 标注层级（§4.3：`a_s/b_s` = L1、`c_l` = L0C fragment）。
- **L0C 容量**：`block_M·block_N·4 = 128×128×4 = 64KB ≤ 128KB` ✓（机械 `l0c_budget` pass，50% 余量）。
- **L1 预算**：`(128·64 + 64·128)·2·num_stages(2) = 64KB`，×1.7 膨胀 ≈ 111KB ≤ 512KB ✓（机械 `l1_budget` pass；上界与 `CONST-capacity-910B2C`（L1 = 512KB）一致）。
- **UB**：本设计无用户显式 UB buffer；`c_l` 在 L0C。§4.5 注记「epilogue 的 fp32→fp16/bf16 舍入由编译器自动插入的向量化 `hivm::VCastOp`（Vector 单元）承担，其内部临时张量由后端管理，不占本设计显式 UB 预算」——该表述与 `T.copy.md` §2.3 第 4 条（跨 dtype 经 VCast）一致，且**未给绝对化片流量主张**，属诚实口径。

### 3. Tiling 策略 — pass
- **Block 划分明确**（§5.2）：`block_M = block_N = 128`、`block_K = 64 if K%64==0 else 32`、`num_stages = 2`；依据 `block_M·block_N ≤ L0C/4 = 32768` 与 Cube 操作数流 `(1/bm+1/bn)` 最小化（bm=bn 处取最小），取值与约束自洽。
- **对齐约束**（§5.3）：`block_K ∈ {64,32}` → fp16 尾轴 128B/64B、`block_N=128` → 256B，均 32B 倍数；分形 `block_K ≥ 16` ✓；`K%16==0` 由 manifest 保证。
- **非整除处理**（§5.4/§6.4）：K 维由 `block_K` 选取保证整除、无 K 尾块；M/N 非整除统一为「**整块装载 + epilogue 切片裁剪**」（`tile_m/tile_n = T.min(...)` + `T.copy(c_l[0:tile_m,0:tile_n], ...)`），探针 D 实测（b4-200×300-128）；**不做 host padding**（避免额外 DtoD）。上一版 §3.3/§5.4/§9.1#4 的三处不一致已统一（见维度 5/7）。
- **分核策略三要素**（§5.5，权威 `_shared/standards/core-split-strategy.md` §1）：
  - ① **逻辑核数计算**：`num_logical_kernels = B × ⌈M/block_M⌉ × ⌈N/block_N⌉`，按 manifest 四个 workload 列算（smoke=4 / b8-1k=512 / b16-512=256 / b4-4k=4096）✓；
  - ② **物理核数依据（实查）**：给出查询代码 `NPUUtils.get().get_aicore_num()` 与实际返回值 **20**（Ascend910B2C），并说明 Cube 直接用返回值、纯 Vector 才翻倍——符合标准「禁止假设值/经验值」；且主动指出 pattern-library `CONST-aicore-910B2C` 记录的 24 为另一环境值，以本设备实查 20 为准 ✓；
  - ③ **规模判定与分核方案**：smoke（4 ≤ 20）→「无需适配」；极大规模（256/512/4096）→ **全量逻辑核 launch（over-subscription）** + 负载均衡量化（波数 26/13/205、整体效率 ≥98%）+ 两条不可缩核依据（L0C 硬上限 `block_M·block_N·4 ≤ 128KB` 无法把 4096 缩到 20 量级；`TRAP-DEVMODE-PERSIST-GEMM` 实测条目——`pattern-library/traps-runtime.md` L155 + repro 齐全——否决 Developer 核内 `T.serial` persistent）✓；核内串行未采用，故「循环边界静态」不适用。
  - **说明（非阻塞）**：极大规模分支偏离标准 §1 要素③ 字面的「固定核数 + 核内串行」范式，改采 over-subscription。该偏离有 `TRAP-DEVMODE-PERSIST-GEMM`（Developer+persistent+gemm 运行时崩溃，双模式对照 repro）与 L0C 容量硬约束双重支撑，且登记了 Stage 4 的 Expert 双 Scope 备选（`PL-1.16-expert-dualscope-bypass`，verified）+ 判定路径（§5.5 ③ 备选 / §1.6.3 裁决计划），属**有据偏离**，不判 fail。
  - 机械 `core_split` skip 已人工核（要素齐全、物理核数实查、核数与 block 自洽）。

### 4. 技术约束检测 — pass
- **三维 Kernel**：R1 已给一维折叠 + decode 方案（探针 A 实测 3D 被拒；`T.Kernel.md` §2.3 佐证）。
- **GEMM 非整除**：§5.4/§6.4 已给切片动态 extent 方案，§3.5.1 逐条登记（含 `TRAP-DEVMODE-PERSIST-GEMM`、CG-2026-0001(d)、`TRAP-threads-kwarg-noop`；三条 pattern-library / capability-gaps 条目均核对存在）。
- **L0C 溢出**：§4.5/§5.3 已核（64KB / 128KB）。

### 5. 循环与同步 — pass
- **循环结构**（§6.1/§6.2）：tile 级 block 并行 `T.Kernel(grid, is_npu=True)` + K 维 `T.Pipelined(⌈K/block_K⌉, num_stages=2)` 流水 + M/N 由 `T.gemm`（MMA 内建并行）+ epilogue `T.copy`；循环类型选择明确（无 `T.serial`/`T.Persistent`，与 §5.5 规避 `TRAP-DEVMODE-PERSIST-GEMM` 一致）。
- **同步与模式匹配**（§7）：Developer 自动同步，不手写 `T.sync_block_set/wait`、`pipe_barrier`，与 §2.1 Developer 模式匹配；手工同步仅留给 Expert 备选。
- **尾块处理**（§6.4）：K 无尾块；M/N 尾块「整块装载 + epilogue 切片裁剪」，与 §3.3 伪代码**单一形态一致**。上一版 §3.3 整块装载 vs §5.4 动态 extent 主选的冲突已消除（§11.2）。

### 6. 验证方案 — pass
- **Golden 函数**：§8.1 给 `golden_bmm`（`torch.bmm`）草案，语义独立（见维度 0）。
- **L0 测试计划**（§8.3）：smoke `(4,128,128,128,fp16)`、bf16 smoke、非整除边界 `(4,200,300,128,fp16)`、三条 host 契约负例（batch 不等 / K 不等 / rank≠3 抛 `ValueError`，对应 `test_bmm.py`）。shape/dtype 与 manifest `BmmFwdOp` workload 一致。
- **精度标准**（§8.2）：fp16 atol/rtol = 1e-3、bf16 = 1.6e-2，与 `test_bmm.py` fixture 逐值一致；`verify_equiv.py` 的 fp32 累加路径结论与该门限相容。

### 7. 完整性与一致性 — pass
- **章节齐全**：§0（含 0.1–0.7）+ §1–§11，模板 11 章节齐全。
- **无占位符**：已检索常见占位符标记（花括号占位、待办标记、未完成补全字样及各语言遗留标记），全文无命中。
- **同类引用真实**：§3.5.3 引用的 examples 路径**全部核对存在**（`examples/gemm/example_gemm.py`、`matmul.py`、`matmul_dynamic_shape.py`、`testing/npuir/linalg_ops/test_2d_grid_gemm_dyn_dev.py`、`examples/TileOPs/tileops/kernels/moe/moe_grouped_gemm_nopad/_moe_grouped_gemm_nopad_fwd_kernels.py`、`examples/TileOPs/tileops/kernels/gemm/gemm/_gemm_kernels.py`、`history_version/design_probe_bmm.py`/`design_probe_bmm_v2.py`）；§1.6 引用的 `CONST-*`（含 `CONST-capacity-910B2C`/`CONST-L1-port-bw`/`CONST-fabric-bw`/`CONST-vector-launch-overhead`/`CONST-mte2-degradation`/`CONST-aicore-910B2C`）、`PL-1.16`、`PL-1.3`、`TRAP-DEVMODE-PERSIST-GEMM`、`TRAP-threads-kwarg-noop`、`CG-2026-0001` 均核对存在。
- **内部一致**：上一版登记的三处不一致（§3.3 vs §5.4 装载形态；§0.5#6「自动 Fixpipe」vs §2.2「无 Vector」；§9.1#4「越界读未实测」vs 探针 D）**均已修复**——§5.4/§6.4/§9.1#4 统一为整块装载 + 切片裁剪、§2.2/§5.1 改述为「Cube 主体 + 编译器自动 VCast（Vector 单元）的隐式跨引擎协同」、§9.1#4 改为「探针 D 实测可容忍，残余风险按 shape 复测」。API 映射 / 内存规划 / Tiling 三处描述相互一致。

### 8. 算法优化分析 — pass
- **§1.6 章节存在性（pass）**：含 §1.6.0 算法调研、§1.6.1 数学等价优化、§1.6.2 向量化替代分析、§1.6.3 向量化轴与数据布局决策，四子节齐全。
- **算法调研存在性与完整性（pass）**：§1.6.0 四问齐全——R1 候选表**含基线（#0 源算法）** 且评估 7 项（基线/init 融合/epilogue 直写/Strassen/split-K/NT/grid 折叠）；R2 在线算法结论明确（判为 split-K，且论证基线本身已是单遍在线形态）；R3 复杂度表覆盖**四口径**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）且基线在表、口径写清；R4 逐候选硬件亲和评估（单元匹配/容量/对齐/静态边界/流水/跨核）；调研范围（§1.6.0 末）写明检索边界（algorithm-candidates.md 无 bmm 专属行、pattern-library、本仓 gemm/moe、结构分析），并说明未使用互联网检索的理由。
- **调研结论正确性（独立复核，pass）**：
  - ① **复杂度算术独立复算**：基线 FLOPs `2·4·4096³ = 5.50e11` ✓；GM 访存量 `4·(4096²·3)·2 = 4.03e8` ✓（与 manifest `roofline` 公式逐字一致）；中间缓冲 `128×128×4 = 64KB` ✓；扫描遍数 1 ✓。分块 vs 未分块口径自洽。split-K 部分和往返 `2·S·B·M·N·4`、Strassen 中间矩阵量级标注合理（按结构给量级，无漏计中间缓冲 GM 往返）。
  - ② **负向断言对照参考表复核**：Read `tilelang-op-design/references/algorithm-candidates.md`——`ALG-gemv`（R1 GEMM 分形 vs Vector 规约；R2 split-K；R4 K 分块与 L0 容量）、`ALG-conv`（R1 含 Winograd，须按单元吞吐比评估）、`ALG-triangular` 等条目存在，**无 bmm 专属化简行**，与设计自述一致；GEMM 双线性型的等价变形空间（结合律重排 = split-K、快速矩阵乘 = Strassen/Winograd）已被 §1.6.0 R1 覆盖，**未发现被漏掉的已知候选**（无归约/softmax/attention 族遗漏）。Strassen 否决前提（`T.gemm` 仅表达标准矩阵乘）经亲自核对 `T.gemm.md` §1/§2.1 成立（无 Winograd/Strassen 原语）；NT 否决前提（I/O 契约 `(B,K,N)`）经核对 `workloads/bmm.py` L30 与 `ops/bmm.py` L339–L354 的 `transpose(-2,-1).contiguous()` DtoD 告警成立。
  - ③ **亲和性淘汰有佐证**：split-K 因「manifest shape 并行度 256–4096 逻辑核 ≫ 20 物理核」被淘汰，与 §5.5 逻辑核数计算自洽；Strassen 因容量（O(N²) 中间矩阵超 L0C/UB）与单元不匹配淘汰，有结构依据。
  - ④ R3 无判定裕度小的并列候选需裁决（Strassen/split-K 均为结构性否决，非并列拍板）。
- **调研→设计一致性（pass）**：§1.4/§3.1/§6 均以 §1.6.0 选定基线族 + §1.6.1 优化后公式（`c = A_0·B_0; for k>0: c += A_k·B_k`，L290/L377）为输入，无「选 A 做 B」；源算法 #0 **未被替换**，§0.4 十项优化手段在 §0.5 逐项处置，替换项在 §1.6.0 结论末段给出意图承接（L1 tiling 承接 SMEM tiling、`T.Pipelined` 承接 K 流水、Cube MMA 承接 WGMMA；fixpipe 直写承接 SMEM 合并写意图但**仅 Expert 适用、本设计不用**）——**无静默丢弃**。
- **数学等价优化完整性（pass）**：§1.6.1 对选定算法做系统审视（恒等变形 init 融合、访存削减 epilogue 直写），两项均含四要素「原式 → 优化后公式 → 等价性论证 → 收益量化」；无更多优化空间有依据（双线性型无公共子表达式、无超越函数/除法、乘法次数达稠密矩阵乘下界）。
- **等价性论证独立推演（pass）**：
  - OPT-1（`initC=(k==0)` ≡ 清零后累加）：IEEE-754 `0 + x == x` 逐位成立（含 ±inf/NaN/subnormal），K 分块累加顺序不变（仍按 `ki` 递增）；独立推演成立。机器验证 `max_ulp 0 / viol_rate 0.0000 / EQUIV_PASS`（重跑一致）。**备注（非阻塞）**：`+0 + (-0) = +0` 而直接 `initC` 首轮可产出 `-0`，属符号零的位模式差异，数值上 `+0 == -0`、对 §8.2 的 atol/rtol 无影响，且机器验证峰值 ulp 差 0，不构成等价性缺陷。
  - OPT-2（epilogue 直写）：同一 fp32 累加器做**一次** fp32→fp16/bf16 的 round-to-nearest-even 舍入，转换次数与源一致；源 GPU 的 `T.copy(c_local, c_smem)`（fp32→fp16，RN）与 NPU Developer 的 `T.copy(c_l, c)`（编译器插入 `hivm::VCastOp`，RoundMode=RINT）舍入方向一致。独立推演成立；机器验证 `max_ulp 0 / EQUIV_PASS`（重跑一致）。
  - **机制前提复核（上一版阻塞级问题的修复验证，pass）**：§0.5#6 / §0.6 R2 / §1.2#2 / §1.6.0 R4#2 / §1.6.1 OPT-2 / §1.6.2#4·#5 / §1.6.3#1 / §2.2·§2.3 / §3.1·§3.2 / §4.4·§4.5 / §5.1 **全部统一改正**为「Developer `T.copy`(L0C→GM) 保持 ND，fp32→fp16/bf16 由编译器自动插入 `hivm::VCastOp`(RINT) 量化；`store_fixpipe`/`enable_nz2nd`/`load_nd2nz` 属 Expert-only」。亲自核对三处来源一致：`T.copy.md` 支持方向表 L15（Developer = 经 VCast 通用路径）+ §2.3 第 4 条（Developer 跨 dtype → `hivm::VCastOp`）+ §2.3 第 6 条（fixpipe 等价**仅 Expert**）。全库检索确认残留「fixpipe/nz2nd」字样**均处 Expert 备选、§11 修订说明或 CG-2026-0001(d) 原始报错文本**语境，Developer 语境已无错误机制描述。「无 Vector」表述已改述为「Cube 主体 + 编译器自动 VCast（Vector 单元）的隐式跨引擎协同、无**用户编写**的 Vector 后处理」。**OPT-2 收益已重估**：由「fixpipe 单跳、无 UB 往返、256MB 片上往返」改为「去除源方案显式 `c_smem` staging 缓冲（32KB/tile）的分配与写回路径」，并明确删去绝对化片流量主张、标注 VCast 内部临时张量由后端管理、精确片流量待 Stage 4 实测（§11.1 L779）。
- **向量化替代完整性（pass）**：§1.6.2 覆盖实现方案全部 5 个计算点（grid 索引 decode、K 维累加循环、MAC、epilogue 写回、fp32→fp16/bf16 舍入），与 §3.3 伪代码、§6.1 循环结构**交叉核对无遗漏**：§6.1 的 tile/K/M·N/epilogue 四项均对应到 §1.6.2 条目；无可向量化而未向量化的逐元素标量循环（唯一「舍入」点由编译器自动 VCast 向量化）。
- **不可替代理由充分性（pass）**：5 项均给出具体理由类别（block 索引/任务映射、Cube 专用单元、DMA/L0C→GM 专用通路、后端自动向量化 VCast），非「实现简单」式泛泛；「用 Vector 逐元素乘加替代 MAC = 反向优化」的判据与 §1.6.0 R4 计算单元匹配一致。
- **替代 API 可行性（pass）**：§1.6.2#5 未提出不存在的向量 API；其说明的 `T.vcast` 为显式替代方向且判定为「等价或更繁、无增益」，与 `T.copy.md` §2.3 第 4 条（VCast 自动插入）一致。
- **向量化轴与布局决策完整性（pass）**：§1.6.3 作为 Cube 类记录布局决策——轴质量评分维度（整除性/分形对齐、累加链形态、repack 代价、片上容量）+ **4 候选全枚举**（#1 Developer ND（选定）/ #2 Expert NZ / #3 NT / #4 host permute）+ 逐候选量化取舍；弃选理由非「未考虑」式：#2 记为 Stage 4 备选、#3 由 I/O 契约否决（引 `ops/bmm.py` L339–L354 DtoD 告警）、#4 引 `PL-1.3-host-permute`（实测 106–147µs，verified 条目存在）。迁移任务对照了源码隐式轴选择（源用 3D grid + FullRow，非直接照搬为结论）。
- **实验裁决模式完整性（pass）**：§1.6.3 备选 #2 判定裕度依赖**未实证常数**（显式 NZ 通路 vs Developer 隐式路径的流水/量化差异，pattern-library constants.md 无此条目——已核对，确无该条目）→ 给出三件套：主选 #1 Developer ND、备选 #2 Expert 双 Scope（结构完整：L1 双槽 A/B、L0C 单槽、K 循环静态边界，载体 `examples/gemm/matmul.py`，达 Stage 4 可直做深度）、裁决计划（代表 shape b8-1k/b4-4k、判定阈值「备选 latency < 主选 × 0.95 才翻转」、回写路径 `core-split-strategy.md` §2.4 / `perf-feedback.md`）+ 按 shape 分派可行性评估。
- **布局决策与实现一致（pass）**：§1.6.3 选定 Developer ND 与 §3.3 伪代码（`T.alloc_shared` + `T.copy` + `T.gemm` + L0C→GM `T.copy`）、§4.3/§4.5（buffer 形状与 L0C/L1 预算）、§6 循环结构三方一致；无重排路径（无 NT/host permute），故无融合转置链需求。
- **与下游章节一致（pass）**：§3.1 公式拆解显式以 §1.6.1 优化后公式为输入（步骤 4「首轮 `c_l ← a_s·b_s`」对应 `initC`）；§6 循环结构与 §1.6.2 结论一致（无被判定向量化的点仍存在逐元素标量循环）。

---

> 检视依据文件: `DESIGN.md`（revision v1）+ `history_version/design_v0.md`（对照）；源算子 `/tmp/opencode/TileOPs-ref/tileops/kernels/bmm.py`、`/tmp/opencode/TileOPs-ref/tileops/ops/bmm.py`、`examples/TileOPs/tileops/kernels/bmm/bmm/_bmm_fwd_kernels.py`；`docs/Tilelang.language/{内存操作/T.copy.md, 内存操作/T.alloc_shared.md, 线性代数操作/T.gemm.md, 编译器提示操作/T.Kernel.md}`；`tilelang-cube-skill/{SKILL.md, references/api-cube.md, references/nz-format.md}`；`.agents/skills/tilelang-op-design/references/algorithm-candidates.md`；`.agents/skills/tilelang-op-optimize/references/pattern-library/{constants.md, traps-runtime.md, traps-compiler.md, attention.md, layout.md, INDEX.md, repro/}`；`.agents/skills/_shared/standards/core-split-strategy.md`；`.agents/evolution/capability-gaps.md`（CG-2026-0001）；`examples/gemm/example_gemm.py`；`examples/TileOPs/tileops/manifest/bmm.yaml`；`examples/TileOPs/tests/ops/test_bmm.py`；`examples/TileOPs/tileops/workloads/bmm.py`；机械复核 `verify_equiv.py` + `.agents/tools/design_calc_check.py`。
