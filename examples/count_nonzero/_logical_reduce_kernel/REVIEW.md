# `_logical_reduce_kernel`（`CountNonzeroFwdOp`）设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/count_nonzero/_logical_reduce_kernel/DESIGN.md`
- 任务类型: 迁移（TileOPs GPU → TileLang-NPUIR，migration_mode = harness）
- 源算子: `/workspace/TileOPs-gpu/tileops/kernels/reduction/logical_reduce.py`（已 Read 全文核对）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 3（见「非阻塞改进建议」）

## 机械复核（D-1/D-5）

### D-1：等价性验证重跑（强制）

- 命令：`/workspace/venv/bin/python examples/count_nonzero/_logical_reduce_kernel/verify_equiv.py`
- 结果：**exit 0，`ALL EQUIV_PASS`**。执行输出与 DESIGN.md §1.6.1 内嵌结果表**逐行一致**：

  | 组 | 用例行 | 违反数 | verdict | 与内嵌表一致 |
  |----|-------|-------|---------|-------------|
  | E1 补式 vs 基线 | 1377 | 0 | EQUIV_PASS | ✅ |
  | E1c 对照（有序 NE 路线，应报差异） | 1377 | 42 | CONTROL_OK | ✅ |
  | E2 fp32 计数域精确性（N ≤ 2²⁴） | 21 | 0 | EQUIV_PASS | ✅ |
  | E2-boundary 计数 > 2²⁴（期望偏差） | 2 | 2 | BOUNDARY_CONFIRMED | ✅ |
  | E3 实列域 vs 256 pad 域 | 1377 | 0 | EQUIV_PASS | ✅ |
  | E3b 窄行 32B pad 补式抵消 | 1377 | 0 | EQUIV_PASS | ✅ |
  | E4 原生 dtype 比较（全 2¹⁶ 位型穷举） | 262144 | 0 | EQUIV_PASS | ✅ |
  | INFO NaN 语义矩阵 | 1 | — | INFO（源=2/候选=2/有序NE=1） | ✅ |

  门禁等价组合计 265096 行、0 违反 —— 与内嵌表「合计 265096 行 0 项失败」一致。
- **空真守卫核对（逐组）**：E1/E3/E3b 每组比较 1377 行（每行计数张量，元素数 > 0）；E2 比较 21 个累加链；E4 全 2¹⁶ = 262144 位型逐元素比较（第 285–291 行对 `(x==0)` 与计数两条路径都做了比对）；E1c 为**判别力对照**，要求 `violations > 0`（否则判 `CONTROL_BROKEN` 并自身失败）；E2-boundary 要求出现偏差。**未发现 0 元素比较导致的空真 PASS 行**（脚本 §314–320 的 6 个 gate 组均有非零比较规模与满足前置条件的 case 配置）。

### D-5：设计期估算机械核算（`design_calc_check.py`）

- 命令：`python3 .agents/tools/design_calc_check.py --design examples/count_nonzero/_logical_reduce_kernel/DESIGN.md`
- JSON checks 摘要：

  | check | status | detail |
  |-------|--------|--------|
  | `ub_budget` | **skip** | 「no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)」→ **人工已核**（见「人工复核结论」） |
  | `l0c_budget` | **skip** | 「no block_M x block_N values parsed」→ 不适用（纯 Vector，无 L0C），人工已核 |
  | `core_split` | **skip** | 「insufficient parsed dims; declared logical cores = None」→ **人工已核**（§5.5 逐负载复算，见下） |
  | `r3_metrics` | **pass** | 四口径（FLOPs / access bytes / scan passes / buffer peak）齐全 |

- **skip 项人工复核结论**：
  - `ub_budget`（§4.5）：逐 buffer 代理预算 fp16 `x_ub+zero_ub+mask_eq(1)+ind(4)` = 9 B/elem、fp32 = 13 B/elem；最大配置 `E_cap = 8192 元素/block` ⇒ fp16 72 KB / fp32 104 KB ≪ 192 KB（实测口径 8 B/elem ⇒ 64 KB）；路径 C `tn=8192` ⇒ 64 KB；硬上限 `196608/8 = 24576 元素`。**无 UB 溢出**。
  - `core_split`（§5.5）：独立复算 `num_logical = ceil(M/bm)`：smoke `32/8=4`（核 4，核内 1）；hidden `2048/2=1024`（核 40，核内 `ceil(1024/40)=26`）；seq `32/1=32`（核 32，核内 1）；3d `128/1=128`（核 40，核内 `ceil(128/40)=4`）；`(129,512)` `129/8=17`；`(32768,4)` `32768/1024=32`。**全部与设计一致**。

### 补充机械复核（设计期探针重跑，非门禁项，独立佐证）

两个 D-3 探针在本环境重跑均 `exit 0`，独立复现设计的关键未知常数：

- `history_version/design_probe_ub.py`：fp16 `2×16384`（32768 元素）报 `ub overflow, requires 2097152 bits while 1572864 bits available` ⇒ 262144 B / 32768 = **精确 8.00 B/elem**；最大可编译元素 fp16 **16384**、fp32 **8192**、bf16 **8192**；窄行 raw 宽度 `N=64/4/2` 全部 OK。IR dump 显示选定链确被 lowering 为 `hivm.hir.vcmp → hivm.hir.vcast(i1→f32) → hivm.hir.vreduce<sum> reduce_dims=[1]` ⇒ §0.6-R1 的向量链 API 组合**编译级实证**。
- `history_version/design_probe_tileloop.py`：fp16 `bm=1/2/8` × 多 tile 与 fp32 `bm=2` 多 tile **全部 OK** ⇒ `CG-2026-0014`（`bm>1` + 多 tile SIGSEGV）在本工具链不复现，§9.4 诉求成立。

> 注：以上两项为**补强证据**；D-1/D-5 是门禁机械项，已分别执行并通过。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

> 已亲自 Read 源算子**源码**全文（475 行）与 NPU 侧 harness 三工件（wrapper / Op 层 / `reduce.py`），逐项与 §0 核对。

- **语义理解正确性 — pass**：源为 `any/all/count_nonzero` 三 kind 共用工厂（源 `_LOGICAL_REDUCE_KINDS`，line 34；分支 line 132–148）。设计 §0.1/§0.2 **正确区分本迁移 kind**：`count_nonzero` 输出 **int64**（源 line 102/110/156/189 三元式 `"int8" if op_kind != "count_nonzero" else "int64"`）、走 `T.reduce_sum`（line 140）；兄弟 kind `any`→`reduce_max`、`all`→`reduce_min`（line 134/137）、int8/bool 载体（line 148/262）。设计 §0.1 明确「**不得跟随兄弟 kind 的 bool 输出契约**」（line 47）。I/O `(M,N)→(M,)`、`dim=None` flatten 为 `(M=1,N=numel)` 与源 Op 层 `_prepare_input`（line 448–458）一致。
- **单位元选择 — pass**：核对 `_pad_value_for_op`（源 line 65–69）——`all`→1.0、其余→0.0。设计 §0.1/§0.5 中 count_nonzero 采用 **0.0 单位元**且「免 pad」重设计下单位元天然中性（E3 机器验证），判定正确。
- **`_needs_pad` 分支解读 — pass**：源 `_needs_pad` 真→`T.Parallel` 掩码加载（含越界行保护，line 113–119）；假→`T.copy` 到 shared 后 cast（line 120–126）。设计 §0.3 步骤 2a/2b 逐条还原；§0.6-R3 的「免 pad raw-N」重设计经 **E3 机器验证**（实列域计数 ≡ 256 对齐 pad 域计数，pad=+0.0 中性）并辅以探针「raw 宽度可编译」实证 ⇒ 等价性保持。
- **tiled 变体迁移范围 — pass**：源 `_needs_tiling` / `_logical_reduce_kernel_tiled`（line 354–357 / 166–243）**未纳入**本设计，且 §0.5-#8/§0.6-R4 **显式区分**「GPU fragment 列上限触发」与「NPU UB 容量触发」的判据、缓冲组、尾块策略（「二者触发条件与结构均不同，不得混同」）——正是 prompt 强调的核对点，判定正确。源「`bm>1` 多 tile」探针证不复现，路径 C 仍取 `bm=1`（保守，§1.6.3-#4）。
- **算法解读完整性 — pass**：§0.3 步骤表 1–6 覆盖源基础工厂**全部计算语句**（工厂常量 line 93–95、grid line 104、载入 113–126、bool 化 129–130、三 kind 归约 132–140、cast 142–148、写回 151），无遗漏、无臆造；数据流图覆盖全部 buffer；host 侧逻辑逐条列出（`_prepare_input`/`_pre_kernel` 实读核对）。
- **优化手段覆盖完整性 — pass**：§0.4 覆盖 SMEM tiling、256 pad、bool 物化、fp32 上抛、`block_m` 48KB 启发式、`threads`、`lru_cache`、tiled 回退、结果块写、autotune、重 pad 布局约束共 11 项（对应源 line 105/93/129/126/393/98/75/166/151/415/382），并标注源缺陷（步骤 2b/6 的 `T.copy` 无 `M % block_m` 守卫，源 line 122/151）。未误判算法层/实现层。
- **耦合性判定合理性 — pass**：§0.5 每条目有**四态处置 + 依据**（保留/等价替换/重新设计/舍弃），与 `migration-analysis.md` §5.3 映射表核实一致（已打开该文件核对 line 147–151「shared memory → L1（Cube 输入）/ UB（Vector 数据）」，与 §0.5 引用吻合）。无「warp shuffle 判保留」「舍弃项无理由」类错误；`threads`/autotune 由 harness K8/K9 承接、tiled 判舍弃并说明意图去向。
- **NPU 重设计可行性 — pass**：§0.6 R1–R6 每项含「源方案 → NPU 新算法 → 语义保持论证」；R2 一维 persistent kernel（无三维 Kernel）；R4 由 UB 元素预算触发；R5 元素预算阶梯；R6 尾块三件套。均满足 NPU 约束（一维、内存层级、容量、静态边界），且经探针实证。
- **设计与决策一致性 — pass**：§0.5 舍弃 pad → §4.4/§5.3/§5.4 全链无 pad；§0.5 舍弃 `threads` → §5.5 无 `threads`；§0.6-R4 → §5.2 路径 C；§0.6-R6 → §3.2/§5.4 尾块守卫。无「§0 判重设计、§3 照抄源方案」脱节。
- **golden 独立性 — pass**：§8.1 golden = `torch.count_nonzero(x, dim) → int64`，以 §0.1 语义为唯一依据，**未复刻** §0.6 的 NPU 算法（无补式/无指示平面/无列分块），且不依赖 `torch_npu` 参考后端（`TRAP-npuir-ref-backend-golden-capability`）。`verify_equiv.py` 为纯 torch CPU + fp64 参照，与 kernel 不同源。
- **引用路径真实性 — pass**：§0/§0.7/§3.5.3 引用的 `examples/`、`testing/` 路径**逐条 `ls` 核对存在**（wrapper、Op 层、`reduce.py`、logsumexp single/tiled、`argmin` 与 `argmax` 的 DESIGN、`_primitives.py`、manifest、test/bench）。Op 层 `reduce.py` 的 `count_nonzero` 仅作 **roofline 记账**（line 379–381），设计未据此实现 —— 正确。
- **迁移范围声明 — pass**：manifest `CountNonzeroFwdOp` 仅前向（无 indices 变体）；设计只实现 `count_nonzero` kind，声明 any/all 为独立迁移单元，`sparsity-seq` 宽度 32768→16384 的下调与 manifest 注释一致。

### 1. API 可行性 — pass

§3.2/§3.4 列出的每条 API 均可在 `docs/Tilelang.language/` 或 `examples/` 找到佐证，且核心链经探针**编译级实证**（IR dump 见机械复核段）：

- `T.vcmp(..., "eq")`：`比较操作/T.vcmp.md`（shape 同形 §2.2.2；bool 不可落 GM §2.3 line 39）✓
- `T.vcast(mask,i1→f32)` 与 `f32→i64`：`数据类型转换操作/T.vcast.md` dtype 表（line 31 `f32→i64`、line 56 `i1→f32`）✓
- `T.reduce_sum(dim=1)`：`规约操作/T.reduce_sum.md`（fp32 ✓、dtype 同 src/dst）+ 家族 logsumexp 先例（`reduce_sum(x_f32, row_sum, dim=1)`，fragment dst）✓
- `T.vsub`/`T.vadd`、`T.fill`（须类型化常量）、`T.copy`（显式 slice）、`T.reshape`、`T.Kernel(is_npu=True)`+`T.serial`+动态 guard：均有文档 + 家族先例 ✓
- v-prefix 优先：设计全链用 `vcmp/vcast/vsub/vadd`，无 `npuir_xxx`。Developer 模式自动同步（§7），未写手动 `set_flag/wait_flag`——**模式匹配** ✓

### 2. 内存层级规划 — pass

- 搬运路径完整（§4.4）：`GM → UB → fragment`，无越级（无 GM→L0 直连）；纯 Vector 无 L1/L0 参与。✓
- UB 预算（§4.5）：逐配置 ≤ 192 KB（人工复算，见机械复核段），三层口径（代理 / 实测 / 硬上限）+ 退化路径补行（`M=1` 大 numel 走路径 C）。✓
- Buffer 层级标注：每个 buffer 标 UB/fragment（§4.3）。✓
- 无 Cube，L0C 不适用。✓

### 3. Tiling 策略 — pass

- Block 划分明确（§5.2）：元素预算阶梯 + 双路径，逐 workload 表给出 `bm/tn` 及算式。✓
- 对齐约束（§5.3）：`T.copy` slice 天然满足 32 B 尾轴；**免宽度 pad**（探针实测 raw N=64/4/2 可编译）。✓
- 非整除处理（§5.4）：M 尾块 `real_m=T.min` + src/dst 双 slice + 垃圾行丢弃；N 免 pad；路径 C 尾块独立静态宽度。✓
- **分核策略三要素（§5.5）齐全**：① 逻辑核数 `ceil(M/block_m)`；② 物理核数 **实查记录**（`NPUUtils.get().get_aicore_num()`=20 与 `get_aivector_core_num()`=40，纯 Vector 用 40；`ub_cap`=196608 同源实查）；③ 规模判定与方案（逻辑 ≤ 物理→无需适配；极大规模→固定 40 核 + 核内 `T.serial` 静态上界 + grid-stride + guard）。核数与 block 取值自洽、核内串行边界为工厂期静态值。✓ 与 `core-split-strategy.md` §1 及机械复算一致。

### 4. 技术约束检测 — pass

无三维 Kernel、无 GEMM/L0C/分形；UB 容量已核算；`reduce` 全程 f32（bf16 仅出现在 `vcmp` 比较域）；`-0.0`/NaN 语义、原位累加陷阱、2D `ite` 标量化、未用 alloc 不被 DCE 均逐项有处理（§3.5.1）。✓

### 5. 循环与同步 — pass

- §6.1 明确 `T.serial`（M persistent + 路径 C tile loop）与向量化原语选择；无 `T.Parallel` 融合 `ite`（规避标量化）。✓
- §7 同步策略与 Developer 模式一致（自动同步；单消费者纪律 + `E_cap` 预算为前提）。✓
- §6.4 尾块逻辑（M 尾块 / N 免 pad / 路径 C 静态尾块）。✓

### 6. 验证方案 — pass

- §8.1 golden 草案（`torch.count_nonzero → int64`）。✓
- §8.2 L0 计划：12 条 L0（含 smoke 三 dtype、persistent 多波次竞态回归、路径 C、raw 非对齐 N=300、M 尾块 `(129,512)`、NaN 门禁 L0-12、契约宽度一致性 L0-11）+ L1/L2 分层。✓
- 精度标准：int64 计数为**精确比对**（atol=rtol=0），与 harness `_exact_compare_int64` 同款；数学依据（E1–E4 整数恒等）成立。✓

### 7. 完整性与一致性 — pass

- 迁移类 §0（0.1–0.7）+ §1–§10 齐全；占位符模板标记扫描 0 命中。✓
- 引用路径真实（已逐条核对）。✓
- API 映射（§3）、内存规划（§4）、Tiling（§5）三处相互一致（`tn`/`bm`/`E_cap`/路径判定同源）。✓

### 8. 算法优化分析 — pass

- **§1.6 章节存在性 — pass**：含 §1.6.0 调研、§1.6.1 等价优化、§1.6.2 向量化替代、§1.6.3 布局决策（另 §1.6.4 实验裁决）。✓
- **算法调研存在性与完整性 — pass**：属**规约类**→完整调研。R1 候选表含**基线（GPU 源）** + C1–C7 共 8 项（每项等价性初判 + 收益方向）；R2 有**明确结论**「有，且本算子天然在线」并以「逐元素判决无跨元素依赖 + 源 tiled 的 running 累加」为结构依据；R3 **四口径**表（FLOPs / GM 访存 Bytes / 扫描遍数 / 中间缓冲峰值，基线在表、口径写清含/不含）；R4 逐候选 6 维评估。调研范围写明（候选库命中行 + `examples/` + pattern-library + 恒等变形空间）。✓
- **调研结论正确性（独立复核）— pass**：
  - ① **等价性推演**：主选 C1 `y = N − Σ[x==0]` 与原 `Σ[x!=0]` 的行内划分恒等成立；NaN/±0 判据正确（`vcmp("eq")` 为 IEEE ordered EQ，`TRAP-vcmp-ne-ordered-nan` 原文「`eq`/`le`/`ge` 为 IEEE 正确」已核对）；E4「fp16/bf16→fp32 精确单射」成立。经机器验证（E1/E1c/E3/E3b/E4 全 PASS）。
  - ② **复杂度复算**：独立复算 R3 各行——基线 `2·N_padded` FLOPs / `N_padded·elem+8` Bytes / 1 遍 / 双 4 B/elem 平面；C1 `2N` / `N·elem+8` / 1 遍 / 9 B·elem⁻¹（fp16，含 `x_ub+zero_ub+mask+ind`）；**C6 正确计入 workspace GM 往返** `M·nchunk·(4+4)B × 2`（未漏计中间缓冲 GM 流量）。数字自洽。
  - ③ **负向断言复核**：C4「`i1→i32` cast 缺失」→ 打开 `T.vcast.md` dtype 表核对（无 `i1→i32` 行）✓；C2「无 `T.vneg`」→ 枚举 `docs/Tilelang.language/数学操作/`（无取反算子）+ `tilelang/language/` grep 0 命中 ✓；C7 排序 O(N log N) ✓；C6 跨核收益面（M ≥ 32 ⇒ 核利用率 ≥ 80%）✓。**未发现被漏掉的已知化简/在线候选**（计数类单目的，恒等变形空间 C2–C7 已枚举）。
  - ④ 亲和性淘汰均引 `constants.md`/`traps-*.md` 实测条目（见下）。
  - ⑤ 并列候选（`tn`/`E_cap` 裕度）未纸面拍板：§1.6.4 给出**主选 + 备选 R′ 完整结构 + 判定阈值（<0.8×）+ 回写路径**三件套。✓
- **调研→设计一致性 — pass**：§1.4 算法描述、§3.1 公式拆解（以 §1.6.1 优化后公式为输入）、§6 循环结构均与 §1.6.0 选定 C1 一致；迁移侧 §0.5/§0.6 处置与调研选族一致，§1.6.0「源算法优化手段承接」逐项（#1–#11）说明，**无静默丢弃**。✓
- **数学等价优化完整性 — pass**：§1.6.1 采纳 E1–E4（+E3b），每项含「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素。✓
- **等价性论证独立推演 — pass**：E1 补式恒等、E2 fp32 对 `[0,2²⁴]` 整数精确、E3/E3b pad 中性、E4 比较域等价 —— 逐项独立推演成立，且与 **D-1 重跑结果一致**。✓
- **向量化替代完整性 — pass**：§1.6.2 覆盖 §1.4 两路径**全部**循环/标量点（#1–#11），与 §3.3 伪代码、§6 循环结构交叉核对无遗漏；逐元素点全部向量化（含源 `ite` 融合标量化热点）。✓
- **不可替代理由充分性 — pass**：#9 任务 guard（block 级边界）、#10 tile 循环（循环携带状态）、#11 host 元数据 —— 理由具体到本算子且非「实现简单/照源码写」。✓
- **替代 API 可行性 — pass**：替代 API 均有 `docs/`/`examples/` 佐证，且探针编译实证（IR dump）。✓
- **向量化轴与布局决策完整性 — pass**：§1.6.3 按规约类必选候选枚举（水平归约 #1 vs 垂直扫描 #2，正交 lane 映射）+ 转置 #3 / 列分块在线 #4 / pass 组织 #5 / 跨核 #6 / 超预算单缓冲 #7；逐候选量化评分（整除性/尾 lane 浪费/累加链/repack/UB 影响）；弃选有**量化理由**（#2 发射爆炸、#3 纯开销零收益、#6 收益面窄+workspace 代价），且**明确对照源码隐式轴选择**（「GPU 轴是输入非结论」）；**弃选论证前提已亲自核对**——`T.transpose.md` 证实为**核内 UB 级二轴交换链**（line 2.3），设计 §1.6.3-#3 未把核内重排写成 GM 级稻草人（合理）。✓
- **布局决策与 §3.3/§4/§6 一致 — pass**：选定轴 = N 轴（`reduce dim=1`）、布局 = 行块行主序驻留，与 §3.3 buffer 形状、§4 UB 预算、§6 循环结构一致。✓
- **弃选/代价类负向论断证据核对 — pass**：`CONST-vector-launch-overhead`（0.5 µs/op，910B2C/8.5.0）、`CONST-mte2-degradation`（33.9/30.2 GB/s/core）、`CONST-capacity-910B2C`（~1.7× 膨胀）、`CONST-msprof-task-floor-910B3`（0.23–0.3 µs/pass，同设备）、`CONST-reduce-ub-per-element`（~128 B/elem，**限定 3-D 整平面形态**）—— 逐条在 `constants.md` 命中，且设计中已按工具链戳标注适用性（910B2C 条目按「未实测假设」引用）。`TRAP-parallel2d-bm-ge2-scalarize` 的 **2026-10-15 适用域收窄**（plain 2D loop 正常向量化）已核对（traps-compiler line 218），设计 §1.6.3-#5 准确复述。✓

## 非阻塞改进建议

> 以下为**建议级**（warn）观察，不阻塞结论；供 Stage 1 revision 或后续 Stage 参考。

1. **`MAX_SINGLE_TILE_COLS` 取值标注**（DESIGN.md §0.3 line 231 / §0.4-#8 / §0.5-#8 / §3.5.2 / §1.6.0-R3）：
   设计将源 tiled 触发阈值记为 `MAX_SINGLE_TILE_COLS(32768)`。源文件 `_primitives.py` 的实际常量为 **`MAX_SINGLE_TILE_COLS = 32512`**（GPU 源与 NPU harness 两处一致；32768 出自源 `logical_reduce.py` 的 docstring 表述，line 169–170）。建议改为「`MAX_SINGLE_TILE_COLS = 32512`（源 docstring 记 32768）」以精确对齐源码常量。**影响评估**：仅涉及**未迁移**的 tiled 变体，不改变任何设计决策，属文字精度问题。

2. **§0.1 边界表 `N=0` 行**（DESIGN.md §0.1 边界表）：
   表内「torch（CPU）」列标「报错」，实测 **`torch.count_nonzero` 对 `N=0` 不报错**（`(2,0) dim=-1 → [0,0]`、`dim=None → 0`，int64）——与「报错」不符；「源 GPU 实现」列标「工厂构造即 raise」，但源工厂为惰性（`@tilelang.jit` 懒编译，`__init__`/工厂函数体内无 `N=0` 断言），此断言无源码证据。建议将该行改为「torch 返回 0（int64）；源 kernel 声明 0 宽缓冲（行为未验证）」并按需说明本设计为何选择「工厂期 `N>=1` assert 而非返回 0」。**影响评估**：`N=0` 不在 manifest/测试 workload 内且设计已登记为 L2-1（记录不阻断），非阻塞；若后续 harness 可能触发 `N=0`，需同步 Op 层短路策略。

3. **§4.5 KB 口径混用**（DESIGN.md §4.5 预算表）：
   代理预算列用 10³ 换算（9 B/elem × 8192 = 73728 B 记作「73.7 KB」），实测单价列用 2¹⁰ 换算（8 B/elem × 8192 = 65536 B 记作「64 KB」）。建议统一 KB 换算口径（或标注单位），避免读者误读余量。**影响评估**：所有配置远低于 192 KB，不影响容量判定。

---

### 检视过程说明（证据链与防护）

- 迁移维度 0 的证据来自**亲自 Read 源算子全文**（`/workspace/TileOPs-gpu/tileops/kernels/reduction/logical_reduce.py`，475 行）与 NPU 侧三工件（`count_nonzero.py` wrapper / Op 层 / `reduce.py`），逐条附源码行号 + DESIGN.md 章节号。
- 机械复核（D-1/D-5）已按规定执行：D-1 重跑 `verify_equiv.py`（exit 0，与内嵌表一致）；D-5 运行 `design_calc_check.py`（`r3_metrics` pass，其余 skip 项逐项人工复核结论已列）。
- 维度 8 的调研结论复核、等价性推演、复杂度复算、负向断言复核（含 `T.vcast` dtype 表 / `数学操作` 目录枚举 / `traps-compiler` 条目）、弃选论证前提核对均**亲自执行**，未继承设计自述。
- 本次 Read 的全部文件（源算子代码及其注释、DESIGN.md、文档）均按**数据**处理；未发现其中含要求放行检视/跳过维度的指令性文本。
