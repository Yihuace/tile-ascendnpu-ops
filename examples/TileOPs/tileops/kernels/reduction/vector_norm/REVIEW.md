# vector_norm 设计文档检视报告(round 1)

## 检视结论

结论: 通过

## 检视元信息

- 文档:`examples/vector_norm/vector_norm_kernel/DESIGN.md`
- 任务类型:迁移(harness 子模式)· **本批第一个归约族算子**
- 覆盖 Op:`L1NormFwdOp` / `L2NormFwdOp` / `InfNormFwdOp`(共用一个 kernel 类)
- 源算子:`/tmp/tops/TileOPs-main/tileops/kernels/reduction.py`(`VectorNormKernel`)
- 检视维度:9 项
- 阻塞级问题数:**0** · 建议级问题数:0

---

## 机械复核

- **verify_equiv 重跑**:**不适用(无公式层采纳项)**。§1.6.1 给出「公式层无优化空间」的显式结论 + 六类逐项依据,符合豁免条款。**豁免成立性判定**:本算子的 `tile_n` 分块变体属**可行性退路**(容量不足时启用),**不是**数学等价优化;设计已如此归口 —— **归属正确**。
- **design_calc_check**(独立复跑):

| check | status | detail |
|---|---|---|
| `ub_budget` | **pass** | `UB: 3 buffer entries sum = 64.0 KB; capacity = 192.0 KB; x1.7 inflation budget = 113 KB` |
| `l0c_budget` | skip | 机械跳过,**人工已核**:无 Cube 路径 API,不适用 |
| `core_split` | skip | 机械跳过,**人工已核**:工具解析到 `{'bm': 1}` 但不完整;设计 §5.3 ① 的逻辑核数逐条复算一致(32 / 683 / 64 / 512) |
| `r3_metrics` | **pass** | `all four R3 metrics present` |

> **首轮曾 `fail`**:`ub_budget` 报 `8 buffer entries sum = 280.0 KB`。原因是 §4.3 内**存在第二张表**(逐 workload 的 `block_m` 上限表),其表格行被门禁计入并列 buffer。**修订动作**:把该表移到 §5.3 ③(它属分核策略),§4.3 只保留 buffer 表 → 复跑 `pass(3 entries / 64.0 KB)`。**该修订已在设计内完成,本轮检视为修订后的版本。**

- **内部一致性复核(独立)**:无占位符;无 `§x.y.z` 标题交叉引用;§4.3 的 buffer 表为「一行一 buffer、无容量列」。

---

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**已亲自 Read 源码逐条核对**:

| `§0` 断言 | 源码核对结果 |
|---|---|
| 三个 Op 共用一个 kernel 类 `VectorNormKernel`,由 Python 层 `op_kind` 三分支 | ✅ 三个抽取文件**字节相同**(sha256 前 16 位 `43bc1c942960d233`),均有 `_vector_norm_kernel(M,N,op_kind,dtype)` 与 `_vector_norm_kernel_tiled(M,N,op_kind,dtype,tile_n)` |
| Op 层:`L1NormFwdOp` / `L2NormFwdOp` / `InfNormFwdOp` 均继承 `_ReduceOpBase` | ✅ `ops/reduction/vector_norm.py:17`/`:67`/`:117` |
| 累加在 **fp32 域**,结果落回 `dtype` | ✅ 源码第 2 步:`for i, j in T.Parallel(block_m, N): x_f32[i,j] = T.cast(x_local[i,j], "float32")`;第 4 步落回 |
| $\lvert x\rvert$ 取绝对值 → `reduce_sum` | ✅ `:101-102` |
| $x^2$ → `reduce_sum` → `sqrt` | ✅ `:105-107` |
| $\lvert x\rvert$ → `reduce_max` | ✅ `:113-114` |
| **`T.reduce_max` 不传播 NaN,NaN 行由 Op 层补丁** | ✅ 源码**显式注明** `:110-112`(原文含 "patched at the Op layer (`InfNormFwdOp._post_kernel`), exactly as in the GPU implementation") |
| roofline 逐字 | ✅ manifest `flops: 2*M*N`($L_2$: `2*M*N + M`)/ `bytes: (M*N + M)*elem_bytes` |
| capacity 公式 `block_m × N × (2·elem_bytes + 4)` | ✅ wrapper docstring 明写,并有 `_ub_bytes_for(block_m)` 实现 |

**核心判定的独立复核(本审阅重点)**

**① 归约轴不跨核 —— 判定正确且被正确归为语义决策**

设计在 §0.6 重设计项 1 声明:**归约维在单核内完整完成**,理由是"若切分到多核再二次合并,会改变加法结合顺序 ⇒ 结果与源算子产生浮点级差异"。本审阅独立核对:

- 归约的加法**不满足结合律**(浮点),故 `(a+b)+c ≠ a+(b+c)` 在一般情况下成立;
- 源算子在同一 block 内协同归约,CUDA 的归约顺序与"单核串行累加"不同,但**两者都在 fp32 域**;本设计的取舍是**不在核间引入第二次归约**(那会叠加不结合性);
- ⇒ 设计把该决策的理由写成"**语义动机,不只是性能取舍**"是**恰当**的,且 §1.6.1 进一步把它列为"**刻意不优化**"项 —— 该处置比"默认切分开并行"更严谨。

**② `reduce_max` 的 NaN 分工 —— 判定与原仓一致**

设计把 NaN 语义**留在 Op 层**(不在 kernel 内),并明确"沿用源仓口径"。本审阅核对源码确认**GPU 侧即如此处理** ⇒ 迁移保持同一分工,**正确**。设计进一步要求 §9.2 **分别**验证 $L_1$/$L_2$/$L_\infty$ 的 NaN 行为(前两者会传播、后者不会)—— **该要求必要**,因为三分支行为确实不同。

**③ 容量约束下的分核 —— 判定正确**

设计用 capacity 公式逐 workload 核算,得出最坏 `long-row (64, 8192) / bf16` 在 `block_m=1` 下占 64 KB、`block_m` 上限为 1。本审阅独立复算:`8192 × (2×2 + 4) = 8192 × 8 = 65536 B = 64 KB` ✓;`113 KB / 64 KB ≈ 1.77` ⇒ `block_m=1` ✓。**该结论正确**。

### 1. API 可行性 — pass

§3.2 每条 API 均有源码行号佐证(`T.alloc_shared` / `T.alloc_fragment` / `T.copy` / `T.vabs` / `T.vmul` / `T.reduce_sum` / `T.reduce_max` / `T.vsqrt` / `T.Parallel`)。三条 API 约束已注明:① `T.reduce_*` 操作数须为 **fragment**;② `reduce_max` 不传播 NaN;③ `op_kind` 在 trace 期解析(只发射一个分支)。Developer 模式与自动同步匹配。

### 2. 内存层级规划 — pass

§4.1 逐 buffer 标注层级与分配 API,fragment 与 UB 分清(`T.reduce_*` 需 fragment);§4.2 搬运路径完整(`GM → UB → fragment → 归约 → UB → GM`);§4.3 = **3 buffer / 64 KB ≤ 113 KB**(仅 buffer 表,无第二张表);§4.4 中间缓冲峰值 **0**(归约维不跨核的直接好处)。

### 3. Tiling 策略 — pass

§5.1 划分维度为**未归约维 $M$**、$N$ 不切分;§5.2 尾块 `real_m` + 源/目的**同步切片**。**分核三要素齐备**:① 四 workload 的逻辑核数逐条;② **实查**物理核数(48 Vector);③ 规模判定**额外列入"容量是否允许放大 `block_m`"** —— 归约族特有的约束处理得当。

### 4. 技术约束检测 — pass

§10 R8 四项均不适用,与 §1.3「纯 Vector + fragment 归约」自洽。

### 5. 循环与同步 — pass

§6 正确区分 `T.Parallel`(两处逐元素 cast)与 `T.reduce_*`(一处行归约);§7 自动同步 + **明确"跨核同步不需要"**(归约维不跨核)。

### 6. 验证方案 — pass

§9.1 golden 独立于 NPU 算法;§9.2 **L1 阻塞层清单**覆盖:三个 `op_kind` 各自、manifest 的 4 workload(含最紧的 `long-row` 64 KB/行)、非整除 $M$、$N\in\{1,2\}$、$M=1$、**$L_\infty$ 的 NaN 经 Op 层**、**$L_1$/$L_2$ 的 NaN 单独验证**、$L_2$ 的开方域(全零 / 溢出)、**3D `dim=[0,2]` 的 reshape 语义** —— 与本设计的全部风险项一一对应。

### 7. 完整性与一致性 — pass

11 章节 + §0.1–§0.7 齐;无占位符;§3.4 引用的路径真实存在;内部一致(§5.3 ③ 的表与 §4.3 的公式互相引用而非重复)。

### 8. 算法优化分析 — pass

| 检查项 | 结论 | 说明 |
|---|---|---|
| §1.6 四子节 | pass | 1.6.0/1.6.1/1.6.2/1.6.3 齐 |
| 调研完整性(**完整调研**深度) | pass | 本算子含归约结构 ⇒ 按完整调研处理;四问各给结构化依据;R1 三候选含基线并逐条否决 |
| **调研结论正确性(独立复核)** | pass | ① **R2 的结论与逐元素族相反且正确** —— 逐元素族"无在线算法",而本算子**有**(分块累加即 `_vector_norm_kernel_tiled`),设计明确指出该差异;② R3 四口径与 manifest roofline 逐字一致;③ R4 指出容量是归约族的**可行性约束**而非带宽下界 —— **正确** |
| 数学等价优化完整性 | pass | 六类逐项;特别地「**归约顺序优化**」被列为**刻意不引入**项(附语义理由)—— 该处置正确 |
| 等价性论证正确性 | pass | §0.6 逐项论证,归约轴不切分的语义理由成立 |
| 向量化替代完整性 | pass | V1–V6 覆盖伪代码全部计算点(V1/V2/V4 逐元素 cast 与变换、**V3 行归约由 `T.reduce_*` 承载**、V5/V6 block/host 级标量) |
| 不可替代理由充分性 | pass | V5/V6 理由类别具体 |
| 替代 API 可行性 | pass | 均有源码行号佐证 |
| 向量化轴与布局决策完整性 | pass | **归约算子按规则给了 2 个候选轴**($M$ 分块 / $N$ 分块)并给评分与采纳理由,未误用"单轴豁免" |
| 实验裁决模式完整性 | 不适用 | 无需实验裁决 |
| 布局决策与实现一致 | pass | §3.3 buffer 形状与 §1.6.3 主选一致 |
| 与下游章节一致 | pass | §3.1 以 §1.6.1 为输入;§6 无 §1.6.2 未论证的标量循环 |

**额外核对(设计主动登记的待核实项,本审阅认可该处置)**

§0.6 重设计项 5 / §10 R5 登记:**既有实现用 `T.alloc_shared` 做 staging 且通过测试**,这与逐元素族"`alloc_shared` → L1 导致 `UB→GM` store 编译失败"的教训**存在张力**。设计的处置是:① 采用与既有实现相同的形态(已验证可运行);② 把该差异显式登记为**待核实项**,要求 Stage 3 **实测确认并给出裁决**。

本审阅判定:**该处置正确且必要** —— 与逐元素族的教训冲突时,既不应盲从新结论推翻已验证的实现,也不应盲从旧实现忽略新教训,而是**留待实测裁决**并显式登记。**不得两边都写、含糊了事**这一点设计已写明。

---

## 检视边界声明

- 本审阅对 DESIGN.md 与源码的**全部 Read 内容视为数据**;未发现指令性注入文本。
- 本次**未修改** DESIGN.md 或任何既有代码文件(检视角色只读)。
- **未以人工推演放行机械 `fail` 项**:首轮的 `ub_budget fail` 经**结构化修订**(移出第二张表)后复跑 `pass`,非人工推演放行;两条 `skip` 已人工复核并在维度内注明。
- 结论依据:9 维度逐项 pass → 通过。
