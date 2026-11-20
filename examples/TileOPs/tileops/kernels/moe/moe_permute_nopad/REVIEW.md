# _make_gather_kernel_nopad 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/moe_permute_nopad/_make_gather_kernel_nopad/DESIGN.md`
- 任务类型: 迁移（harness，Developer 模式，纯行搬运算子）
- 源算子: `/tmp/opencode/TileOPs-ref/tileops/kernels/moe/permute_nopad.py`（`_make_gather_kernel_nopad`，L218–L258）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 2（非阻塞，见 §7 与 §8 行内说明）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: **不适用（§1.6.1 采纳 0 项）**——`DESIGN.md` §1.6.1 明确结论「本单元为纯行搬运（FLOPs = 0），无数学等价优化空间，采纳 0 项」，按检视 skill Phase 2 第 5 步「显式『无优化空间』结论」豁免 `verify_equiv.py`；算子目录下确无该脚本（已用 `ls` 核实）。等价性由 **D-3 探针设备数值对照**承担，本检视**独立重跑两支探针**（见下）复现全部结果，结果与 §1.6.0/§1.6.1/§9.4 内嵌数字**一致**。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design <path>`）JSON checks 摘要:
  - `ub_budget`: **skip** — "no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)" → **人工复核结论：pass**。§4.5 逐 buffer 表明确：主选 `ub_row`(1,7168) bf16 = 14336B ≈ 14.0KB；备选 `ub_batch`(4,7168) = 57344B ≈ 56.0KB；均 ≤ 单核 UB 192KB（`CONST-capacity-910B2C`），且 §4.7 关闭 auto-multi-buffer 取单版本，无膨胀风险。
  - `l0c_budget`: **skip** — "no block_M x block_N values parsed" → **人工复核结论：pass（无 L0C）**。本单元无 `T.gemm`/Cube scope，数据路径仅 GM→UB→GM（§1.5/§4.4），不涉及 L0C。
  - `core_split`: **skip** — "insufficient parsed dims {}; declared logical cores = 1" → **人工复核结论：pass**。§5.5 三要素齐全：① 逻辑核数 = `numel`；② 物理核数 = `NPUUtils.get().get_aicore_num()` **实查 = 20**（本检视在本机独立执行同款查询，返回值 **20**，与设计一致），纯 Vector 翻倍 → `P = 40`（core-split-strategy.md §1②）；③ 极大规模 → 固定 40 核 + 核内 `T.serial`，`spc` 静态。核数与取值自洽（`numel=32768 → nk=40, spc=820`）。
  - `r3_metrics`: **pass** — "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"。
- **检查器无法机械解析的维数（dim 解析失败）已逐项人工复算**，结论见对应维度。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**检视方式**：亲自 Read 源算子 `/tmp/opencode/TileOPs-ref/tileops/kernels/moe/permute_nopad.py` 全文（391 行），与 `DESIGN.md` §0 逐项核对。

**语义核验（§0.1 ↔ 源码）**：源码 L242-254 的 `_gather_main` 恰为 `T.copy(hidden_states[permuted_idx[slot], 0:hidden_size], perm_h[slot, 0:hidden_size])`，即 `perm_h[s,:] = hidden_states[permuted_idx[s],:]`；§0.1 的语义式（含 `numel=T·K`、tight 无 padding）与源码一致。dtype 核验：源签名 L243/L245 两侧同为 `dtype`（fp16/bf16），`permuted_idx` 为 `"int32"`（L244）——§0.1「同 dtype、逐位透传、int32 索引」正确。边界核验：源 L250 `if slot < numel` 尾守卫、L364 调用方 `torch.zeros(numel)` 零初始化 → §0.1「numel 非 8 倍数尾块」「EP 未写槽位默认源行 0（dummy）」与源码/注释（L363、L382-384）一致；「无规约、无累加顺序」核验：源码无任何累加语句，正确。host 侧：源 `_make_gather_kernel_nopad` 为纯工厂、无 host 预处理（§0.1 正确）；编排在 `forward`（L331-391），§0.1 明确将其排除出本单元范围，正确。

**算法步骤覆盖（§0.3 G1–G4 ↔ 源码 L229-236、L247-254）**：`VEC=8`（L229）、`threads=min(1024,H//8)` 递减至 `H%threads==0` 且 ≥1（L230-233）、`ROWS_PER_BLOCK=8`/`grid=ceil(numel/8)`（L235-236）、`T.serial(8)` 与 `slot=bid*8+r`（L248-249）、尾守卫（L250）、直接 GM→GM 行拷贝（L253-254）**逐条命中，无遗漏、无臆造**。数据流图覆盖 `hidden_states`/`permuted_idx`/`perm_h` 全部 buffer，正确。

**优化手段核验（§0.4 O1–O7 ↔ 源码显式结构/注释）**：O1（8 槽位/block，L235 + docstring L222-227「amortizes launch/scheduling overhead」）、O2（直连无 fragment 往返，L251-252 注释）、O3（每 thread 8×bf16 128-bit `uint4`，docstring L16-17/L221-222）、O4（`threads=H//8` cap 1024，L229-233）、O5（动态源行 `permuted_idx[slot]`，L253）、O6（`grid=ceil(numel/8)`+尾守卫，L236/L250）、O7（2D 行切片，L253-254）**均识别且层级标注合理**（O5/O7 判算法层、O1-O4/O6 判硬件相关——与源码对 CUDA block/thread 模型的依赖一致，无误判）。

**耦合性判定抽检（§0.5）**：O2 判「NPU 无统一地址空间 → 等价替换为 UB 中转」有依据——已亲自打开 `docs/Tilelang.language/内存操作/T.copy.md`，其 §2.3-5 明载「**Developer 模式不支持 GM→GM 拷贝**（memref 到 memref 直连，编译期报错 "Unsupported memref to memref copy yet"），需经 UB 中转」，与设计引文一致；O4 舍弃（NPU 无 thread）依据 `TRAP-threads-kwarg-noop`（pattern-library traps-compiler.md L124-138，实测有效条目）；「gather 内联进 scan」舍弃依据 CG-2026-0027（capability-gaps.md L1147-1185，status open，含已固化复现），判定成立。四态处置均有依据，未出现「warp shuffle 判保留」类错误。

**重设计可行性（§0.6 R1–R4）**：R1（GM→GM 拆 GM→UB→GM）、R2（标量 GM 读 → `T.copy` 动态 slice 起始）、R3（固定核数 + 核内 `T.serial`）、R4（独立 kernel，`permuted_idx` 经 GM 交接）**均含「源方案 → NPU 新算法 → 语义保持论证」三要素**；R2 的「规避 CG-2026-0027 关键：`src` 读自 GM（由另一 kernel 的 scan 写入），非本 scope 向量产出」与 gap 定义（向量产出→标量消费）严格对齐，论证成立。新算法满足一维 Kernel / 无 GM→GM / 静态循环边界等 NPU 约束。

**设计与决策一致性（§1–§7 ↔ §0.5/§0.6）**：§1.4 算法描述、§3.1/§3.2 API 映射、§3.3 伪代码、§4.3/§4.4/§4.5、§6 循环结构、§7 同步策略**处处体现 §0.5/§0.6 的 NPU 形态**（UB 中转两段 DMA、标量 GM 读、固定 40 核、独立 kernel 边界），无「§0 判重设计、§3 照抄源方案」脱节。

**golden 独立性（§8.1）**：`golden_gather` 用 `torch.gather(hidden_states, 0, idx)` 纯 PyTorch 行 gather 实现，**不经** UB 中转/动态 slice/分核串行，以 §0.1 源语义为唯一依据，未复刻 §0.6 NPU 算法——独立性成立。

### 1. API 可行性 — pass

§3.2/§3.4 列出的每条 API 均有 `docs/` 或 `examples/` 佐证：`T.copy` 切片语法（`docs/Tilelang.language/内存操作/T.copy.md` §2.1/§2.4）、`T.alloc_ub`（`.../内存操作/T.alloc_ub.md`）、`T.Kernel(nk, is_npu=True)`（`.../编译器提示操作/T.Kernel.md` §2.3，已核对其 NPU 示例确为 `as (cid, _)` 双元解包，与 §3.3 `as (core_id, _subid)` 一致）、`T.serial`（`examples/cumprod/_cumulative_kernel/_cumulative_kernel.py`）。三条「不可用」API 亦亲自核验：GM→GM `T.copy`（T.copy.md §2.3-5 编译期报错）、`T.Kernel(threads=)`（`TRAP-threads-kwarg-noop`）、`T.gather` 仅尾维（T.gather.md §2.3，且 §2.4 示例 `src` 经 `T.alloc_shared` 驻 UB）。**v-prefix 优先原则**：本单元无浮点逐元素运算，§3.2 注明确说明不涉及 `vadd/vmul/vexp/vcast/vbrc`，行内向量化由 MTE2 整行 DMA 承担——该豁免理由成立（无需 v-prefix 而非遗漏）。API 与 Developer 模式匹配（自动同步、无手动 set_flag/wait_flag）。

### 2. 内存层级规划 — pass

§4.4 给出完整路径 **GM → UB → GM**（无 L1/L0A/L0B/L0C，无 GM→L0 越级）；`ub_row`/`ub_batch` 均标注 UB 层级（§4.3）。UB 预算（§4.5）主选 14.0KB、备选 56.0KB，**均远低于 A2/A3 = 192KB**；§4.7 以 `pass_configs` 关闭 auto-multi-buffer，故无 1.7× 膨胀（历史 `TRAP-UB-multibuffer-inflation` 不适用）。本单元无 Cube，L0C 容量约束不涉及。机检 `ub_budget`/`l0c_budget` 为 skip，已人工复算确认（见「机械复核」）。

### 3. Tiling 策略 — pass

**分核策略三要素齐全**（core-split-strategy.md §1/§2.2 核对）：
- ① **逻辑核数**：`num_logical_kernels = numel`（每输出行 1 工作项），代表值 smoke=8 / partial-tail=10 / prefill=32768（§5.5）。
- ② **物理核数（实查）**：§5.5 记录查询代码 `NPUUtils.get().get_aicore_num()` 与实际返回值 **20**（Ascend910B2C，2026-10-14）——**本检视在本机独立执行同一查询，返回值 = 20，与设计一致**；纯 Vector 单元按标准翻倍 → `P = 40`，符合 core-split-strategy.md §1②「纯 Vector 算子核数翻倍」，非文档假设/经验值。
- ③ **规模判定与分核方案**：`numel ≤ P` → 「无需适配」（`nk=numel`、`spc=1`）；`numel > P`（极大规模）→ **固定启动核数 = 物理核数 40 + 核内 `T.serial` 串行**，`nk=min(numel,P)`、`spc=ceildiv(numel,nk)` **为编译期常量（静态边界）**，符合标准§1③。负载均衡：连续槽位区间划分，每核槽位数差 ≤1，每槽位工作量恒定（1 行 DMA）。

Block 划分明确（1 输出行 = 1 工作项，无空间 tile）；对齐约束：§5.3 要求 `H·itemsize % 32 == 0`（fp16/bf16 → `H%16==0`），manifest workload H∈{64,128,3072,7168} 全满足；非整除处理：`if slot < numel` 尾守卫 + `numel<nk` 特例（§5.4/§6.4）。机检 `core_split` 为 skip，已人工复算确认（见「机械复核」）。

### 4. 技术约束检测 — pass

三维 Kernel：不涉及（1D `T.Kernel`，§3.5.1）；GEMM 非整除：无矩阵；L0C 溢出：无 L0C。§3.5.1 逐条对照本项目已知限制，明确 GM→GM 禁令（有 UB 中转方案）、CG-2026-0027（有 kernel 边界隔离方案）、CG-2026-0028（无 `>>`/无 `alloc_local`/尾块用 `if` 语句）、整行 32B 对齐（有约束条件）等，均给出处置方案，无遗漏项。

### 5. 循环与同步 — pass

§6.1 明确循环类型（核级 `T.Kernel(nk)` / 核内 `T.serial(spc)` 静态边界 / 行内 `T.copy` DMA），并显式声明未使用 `T.Parallel`/`T.Pipelined`/`T.Persistent` 及理由（无规约/无 spatial 分块/非持久化）。§7 同步策略与 Developer 模式匹配：scope 内编译器自动同步、不使用任何手动同步原语、kernel 间靠同 stream 顺序——与源「两 kernel 顺序 launch」语义一致。尾块处理 §6.4 完整。§7.1 显式声明「不在标量上下文读取本 scope 向量 op 写入的 UB 值」（规避 CG-2026-0027），与 §0.6 R4 闭环。

### 6. 验证方案 — pass

§8.1 含 PyTorch 参考实现草案（`golden_gather`，`torch.gather` 行 gather），且注明与 op 级 counting-sort 参考交叉验证 `permuted_idx` 语义。§8.2 按 dtype 给出精度标准：纯搬运 → **逐位精确（atol=0, rtol=0）**，与 §0.1「无算术、逐位透传」一致。§8.3 含 L0 门槛测试计划 9 例（tiny-bf16/tiny-fp16/small/partial-tail/重复源行/EP dummy/prefill/多次运行稳定/对接契约），覆盖 block 整除与非整除（numel=10）、双 dtype、尾块、dummy 语义与跨 kernel 对接。精度标准合理（搬运类比特级比对强于通用容差）。

### 7. 完整性与一致性 — pass

11 章节齐全，迁移类 §0 含 0.1–0.7 **全部小节**。占位符类标记已 grep 核验为空（无未填占位、无未完成标记）。§3.5.3 与 §11 引用的 `examples/` 路径**逐条 Glob 核实全部存在**（`examples/deepseek_v4/example_lighting_indexer_bwd_kernel.py`、`examples/moe_grouped_gemm_nopad/_tile_scheduler_kernel/`、`examples/cumprod/_cumulative_kernel/_cumulative_kernel.py`、`examples/deepseek_v32/fp8_lighting_indexer.py`、`testing/npuir/indexing_ops/test_slice_gather_dev.py|_exp.py`，以及 TileOPs 参考实现/测试/manifest）。API 映射（§3.2）、内存规划（§4）、Tiling（§5）三处描述相互一致（`nk`/`spc`/`ub_row` 形状在各章一致）。

> **建议级观察（非阻塞）**：§3.3 伪代码将 `nk = min(numel, 40)` 直接写为字面量 40，而 §5.5 定义 `P = get_aicore_num()*2`。因 kernel 由工厂按 shape 特化、`nk` 为编译期常量，字面量不改变结论；建议 Stage 3 实现时以 `NPUUtils` 实查值生成常量或注释链回 §5.5，避免工具链换机时硬编码失配。

### 8. 算法优化分析 — pass

**§1.6 章节存在性**：含 §1.6.0 调研 / §1.6.1 数学等价优化 / §1.6.2 向量化替代 / §1.6.3 向量化轴与布局决策**四子节齐全**。

**算法调研存在性与完整性（四问覆盖）**：R1 候选表 C0–C7 **含基线候选 C0（源算法）** 并逐候选举出等价性初判/收益方向；R2 在线算法**有明确结论**（「有，且本质就是单遍流式」——纯逐行映射、无跨行依赖，并严谨化说明 C3 的遍历方向反转仍属单遍）；R3 复杂度表**四口径齐全**（FLOPs/索引 op、访存 Bytes、扫描遍数、中间缓冲峰值）且**基线在表**、口径脚注清晰；R4 **逐候选六维评估**（计算单元匹配/片上容量/对齐整除/静态边界/流水融合/工具链可用性）并给出结论。调研深度与「动态索引 + 超发 grid 改造 + 工具链可行性」三处非平凡决策匹配。

**调研结论正确性（独立复核）**：
- *复杂度算术独立复算*（代表 shape T=4096,K=8,H=7168,numel=32768,bf16, elem=2B）：读 `numel·H·2` = 32768×7168×2 = 469.76MB ≈ 470MB ✓；读写合计 ≈ **939.5MB** ✓；C1 标量读 `numel·4` = 131072B = 131KB ✓；C3 读 `T·H·2` = 58.72MB + 写 470MB = **528.5MB** ✓（−43.8%≈−44%）✓；C5 `2·numel·T·H` = 1.924e12 ≈ **1.9e12 MAC** ✓；roofline `939.5MB ÷ 1.26TB/s ≈ 745.9µs ≈ 746µs` ✓；`spc = ceildiv(32768,40) = 820` ✓。**未发现中间缓冲 GM 往返漏计**（C1 仅 UB 14KB 驻留，无额外 GM 往返）。
- *负向断言主动复核*：① 「无行 gather 向量原语」——对照 `docs/Tilelang.language/索引与元素操作/T.gather.md` §2.3（「gather 当前按最后一维取数，不提供额外的 `dim` 参数」，已亲自打开核对）+ §2.4 示例（`src` 经 `T.alloc_shared` 驻 UB）+ pattern-library `TRAP-C10-strided-gather`（traps-compiler.md L29/L41，status verified），负向断言**成立**；② 「无更优替代」——对照 `references/algorithm-candidates.md` 全表（12 族：softmax/layernorm/rmsnorm/attention/conv/pooling/reduction/topk/gemv/triangular/elementwise/transpose），**确无 gather/permute 专条**，故不存在被漏掉的「已知候选行」；设计另自主枚举 C3（read-once scatter，并登记 R-4）、C4（尾维 gather 转置）、C5（选择矩阵 GEMM）、C6（host）、C7（逐元素）等候选并给出淘汰依据，覆盖度充分，**未发现明显漏掉的已知候选**；③ R2 的「无两遍/多遍物化候选」与算子结构（无 running 统计量）自洽。R3 无「判定裕度小」的并列候选（C1 实测达带宽地板，C2 仅 +3%，已由裁决阈值 ε=0.05 明确裁决）。
- *亲和性淘汰依据*：容量淘汰引用 `CONST-capacity-910B2C`（UB=192KB，已核 constants.md）；带宽地板引用 `CONST-mte2-degradation`（**已核 constants.md L82 段：256M 3:1 R/W 混合全流量 ~1.26 TB/s 为该设备地板**，与 §1.6.0 引文一致）；发射成本 `CONST-vector-launch-overhead` 明确标注「向量 op 的 ~0.5µs/op 不直接适用于 MTE2 DMA」（核 constants.md 确为向量算子口径），并如实标注为「未文档化假设 + 估算依据」，由 D-3 性能探针标定——诚实且不构成主选/备选裁决依赖。
- *互联网来源*：§1.6.0 声明「未执行互联网检索」并给出范围理由（本地信息源已覆盖 gather 候选空间）——无需来源记录。

**调研→设计一致性**：§1.6.0 选定 C1 实际驱动 §1.4（算法描述第 1/2/3 条）、§3.1（公式拆解 P1–P4）、§6（循环结构：`T.Kernel(nk)`→`T.serial(spc)`→整行 `T.copy`）——三处与调研结论同口径。迁移承接：§0.4 O1–O7 逐项给出「保留/等价替换/重新设计/舍弃 + 意图承接」说明（O1→固定核数摊薄启动、O2→形态替换（UB 中转）、O3→整行 DMA、O4→舍弃、O5/O7→保留、O6→重设计+保留尾守卫），**无静默丢弃**；§0.5/§0.6 处置与调研选族一致。

**数学等价优化完整性（§1.6.1）**：结论为**采纳 0 项**，并从「无算术可化简 / 无公共子表达式 / 无算子降代换与访存削减空间 / 唯一流量削减候选 C3 属接口外 / 代码形态级调整非数学优化」五条给出依据；未采纳项 C3 的等价性与收益（−44% 流量）明确登记为 R-4。**等价性论证独立推演**：语义式为纯行搬运 `perm_h[s,:]=hidden_states[pidx[s],:]`，两侧 dtype 一致、无 cast、无中间计算 → 恒等拷贝逐位成立，无舍入（fp16/bf16 均无误差）。**verify_equiv 豁免合法**（0 采纳项），等价性由 D-3 探针设备数值对照承担——**本检视独立重跑探针复现 max_diff=0**（见「机械复核」）。

**向量化替代完整性（§1.6.2 ↔ §3.3/§6 交叉核对）**：§1.6.2 盘点 5 个计算点（行内元素搬运、输出槽位循环、源行号读取、尾块守卫、核内任务起点），与 §3.3 伪代码、§6.1 循环结构**逐点对应、无遗漏**。第 1 项（行内元素）已由整行 `T.copy` DMA 全向量化；第 2–5 项保留为标量并**逐项给出具体到本算子的理由**：行/任意维 gather 无向量原语（附 T.gather.md §2.3 + TRAP-C10 佐证）、标量消费向量产出不可同步（CG-2026-0027）、动态边界控制流、block 级索引计算——均非逐元素热点（每行 1 次 vs 每行 7168 元素 1 条 DMA），理由充分。替代 API 可行性：本单元无 v-prefix 替代需求（无算术），所引 `T.copy` 有 docs/examples 佐证。**§6 未出现 §1.6.2 未论证的逐元素标量循环**。

**向量化轴与布局决策完整性（§1.6.3）**：候选矩阵按 gather 类枚举「连续批量 gather 粒度 × 向量轴对齐 × 索引 vs 分段拷贝」——L1（槽位串行+整行 DMA，主选）与 L2（R 行批量写，备选）为必选两类；L3–L6 弃选均附**量化理由**（L3 容量 58.7MB≫192KB + 转置链；L4 FLOPs 1.9e12 + P 物化；L5 接口缺 `fwd_idx`；L6 CPU 同步/带宽劣化），**非「未考虑」或「I/O 是行主序所以内部也行主序」式论证**。轴质量评分表（整除性/累加链/repack/UB 影响）对两候选轴（numel/H）逐项量化；**迁移任务对照了源码隐式轴选择**（§1.6.3 末段明确：源 GPU thread 轴不对应 NPU 向量轴，独立评估后映射为 DMA 整行 + 核内串行）。**弃选论证前提核对**：C4/L3 的代价机制（`T.gather` 仅尾维、`src` 须驻 UB）已亲自打开 `T.gather.md` 核对，与文档一致，**非**把核内重排写成 GM 级 transpose 之类前提错误；L3 明确标注「VGather 于 `test_slice_gather_dev.py` 可用——非 API 缺失」，负向论断与 `testing/` 佐证一致。

**实验裁决模式完整性**：§1.6.0 末给出**主选 + 备选 + 裁决计划三件套**——主选 C1（0 起点 UB dst 最简形态）；备选 C2（R=4，含完整结构：`ub_batch(R,H)` 形状、57KB UB 预算、`T.serial(spc//R)` 循环、分核三要素按 batch 粒度重算 `nk=min(numel//R,40)` 且 `spc` 取 R 倍数）；裁决计划含代表 shape（smoke/partial-tail/kimi-k2-prefill/qwen3-30b-prefill）、测量指标（编译结果 + 逐位精确 + 稳定性 + 墙钟）、**明确判定阈值 ε=0.05**、回写路径（§1.6.0/§5.3/§9）；并评估 **shape 特化分派可行性**（工厂按 shape 特化，可按 shape 分派主选/备选）。判定裕度依赖的发射成本属「未实证常数」，但已由 D-3 探针标定且**裁决不依赖它**（带宽主导），满足要求。

**布局决策与实现一致**：§1.6.3 选定 L1 与 §3.3 伪代码（`ub_row(1,H)`、`for s in T.serial(spc)`、两次 `T.copy`）、§4.3/§4.5（buffer 形状与预算）、§6（循环结构）**三方一致**；无核内重排（repack=0）。

**D-3 探针证据充分性（重点核对）**：本检视**独立重跑两支探针**：
- `design_probe_gather.py`：**PROBE_RESULT: PASS** —— bf16/fp16 × {smoke numel16 / numel10 / H=128 / variant B numel64} 共 8 例，全部 `exact=True max_diff=0.000e+00` 且 `stability(6 runs)=True`。**位精确 + 6 次稳定**成立，且覆盖尾块（numel10）与非零起点 UB dst 变体 B。
- `design_probe_gather_perf.py`：T=4096/H=7168/numel=32768 —— A=**722.1µs**、B(R=4)=**698.1µs**、B(R=8)=**694.8µs**，全部 `max_diff=0.000e+00`；A/B4 = 1.03×。与 §1.6.0 内嵌数字（719.8/696.5/694.4）差异 <0.4%（run-to-run 抖动），**互相印证**。据此 939.5MB/722.1µs = **1.30 TB/s ≥ MTE2 混合流量地板 ~1.26 TB/s**。
- 结论：D-3 探针（位精确 + 6 次稳定 + 带宽 ≥ 地板）**构成充分证据**——既闭合了迁移任务指定的关键未文档化假设（「标量 GM 读 → `T.copy` 动态源 slice 起始」首次用于驱动**读起点**），又量化了主选/备选性能差，支撑「已在 roofline 上、边际空间极小」的主选裁决。

> **建议级观察（非阻塞）**：§1.6.0 R2 引用「gather-分桶族」作为参考表条目，但 `references/algorithm-candidates.md` 实际**无** gather/permute/scatter/分桶专条（§1.6.0 末「调研范围」亦自述「无 gather/permute 专条」，两处表述略有张力）。该措辞不影响负向断言成立（已逐候选枚举 C0–C7，无漏候选），建议将「gather-分桶族」改为「按 §3 结构判据自建候选集」以与调研范围一致。

**与下游章节一致性**：§3.1 公式拆解以 §1.6.1「优化后公式 = 与 §1.3 一致」为输入（无公式级改动，自洽）；§6 循环结构与 §1.6.2 结论一致（已向量化的行内元素不再出现逐元素标量循环）。
