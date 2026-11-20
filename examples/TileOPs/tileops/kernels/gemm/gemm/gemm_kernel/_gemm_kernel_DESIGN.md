# _gemm_kernel 算子设计文档

> 迁移任务（harness 模式，project=`gemm`，op=`_gemm_kernel`，编程模式 Developer）。
> 源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/gemm.py` 中的 `_gemm_kernel`（GPU SM90 warp-specialized dense GEMM）。
> 本设计只覆盖 `_gemm_kernel`；`_gemv_kernel`（m==1 / n==1 fast path）另做。
> 本章结论驱动 §1–§11 全部设计决策。

## 修订说明（revision v1 — Stage 2 检视不通过后修订）

> 依据 `REVIEW.md`（阻塞 1 条 + 建议 4 条）修订；**仅修订被检视问题**，已成立结论（warp-specialization→`T.Pipelined`、一维分核、WGMMA→`T.gemm`、动态 extent 尾块、E1/E2、分核三要素实查 20）全部保留。

### 相对上一版（design_v0）的关键调整

| # | 检视问题 | 严重程度 | 本版修订 |
|---|----------|---------|---------|
| 1 | 分核网格数公式 `min(num_logical, num_physical)` 与 §5.5 主选方案 / §0.6 语义双射论证自相矛盾，且伪代码无核内 persistent 循环 → `num_logical > 20` 时只启动 20 核、**静默漏算输出块** | 阻塞 | **统一为 `num_launched = num_logical`**（§0.6 R2、§3.3、§5.5、§6.2 四处一致）；§5.5 正文删除 `min(...)` 公式，显式写明「非 persistent 下 `num_launched` 必须等于 `num_logical`，否则漏写输出」 |
| 2 | §8.4「无 Cube↔Vector 跨引擎流水」与 VCast（Vector 单元）epilogue 事实不符 | 建议 | §5.1 / §8.4 改述为「后端自动插入的 `VCastOp` 隐式 Cube→Vector 协同，无需用户手写双 Scope」 |
| 3 | E1 基线为 strawman（源 epilogue 本就单次融合 cast），且脚本 E1 检查为恒等式 | 建议 | E1 重新定位为「不引入 fp32 中间 GM 往返」的**非回归约束**（§1.6.0 R1-5 / §1.6.1）；等价性脚本 E1 改为真正可区分的两条计算路径对照 |
| 4 | §3.3/§6.2 B 装载切片仅适用 `trans_b=False`，与默认 NT 的 `B_shared(block_N, block_K)` 不符 | 建议 | §3.3/§6.2 伪代码补 `trans_a`/`trans_b` 双分支切片；§3.3 note 同步补全 |
| 5 | §6.2 K 尾清零被注释未落伪代码；§0.1/§9.3 GEMV fast-path 条件不完整 | 建议 | §3.3/§6.2 补显式 K 尾清零分支；§0.1/§9.3 补全 GEMV 精确条件 |

### 为何不会再犯同一错误

- **网格数 vs 核内循环配对不变量**：本版确立「无非 persistent 核内多任务循环时，一维网格启动数必须恒等于逻辑块数」的显式不变量（§5.5），并在 §0.6 R2、§3.3、§6.2 三处交叉引用；`min(...)` 截断仅在「固定物理核数 + 核内 `T.serial` 串行」范式下才成立，本设计为 Developer 且规避 persistent 崩溃（`TRAP-DEVMODE-PERSIST-GEMM`），不属该范式，故全文禁止出现。
- **等价性脚本非恒等式**：E1 的两条对照分支改为不同计算路径（融合单次舍入 vs 显式 fp32 物化 + 分离 cast），脚本另给出「fp32 opmath 单次舍入」与「fp64 直接舍入」的双重舍入差异诊断，避免 `baseline == candidate` 的恒等式。
- **引擎归属措辞**：承认 epilogue 的 `VCastOp` 属 Vector 单元后，全文不再出现「无 Cube↔Vector 协同」的矛盾表述；`store_fixpipe`/`enable_nz2nd` 仍仅出现在 Expert 备选。
- **伪代码与正文一致**：K 尾清零、trans 分支等正文声明全部落到伪代码，避免「正文声明、伪代码缺失」的表述漂移。

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

- **数学语义**：通用稠密矩阵乘 `C = op(A) @ op(B)`，其中 `op(·)` 由 `trans_a` / `trans_b` 决定——`trans_a=True` 时 `op(A)=Aᵀ`（A 存储为 `[K,M]`），`trans_b=True` 时 `op(B)=Bᵀ`（B 存储为 `[N,K]`）。输出 `C` 为 `[M,N]`，无转置布局。四种布局（NT/NN/TN/TT）由 `GemmOp` 的 `(trans_a, trans_b)` 选择，默认 `(False, True)`（NT）。
- **I/O 契约**：`A: (M,K)`（或 `(K,M)`）、`B: (K,N)`（或 `(N,K)`）、`C: (M,N)`；`A/B` dtype ∈ fp16/bf16，`C` dtype 与输入相同；`K` 为收缩维。输出 shape `(M,N)` 为迁移必需字段（见 `source_output_shape`），无 transpose 输出。
- **规约语义**：沿 K 维求和；源码以 `block_k` 为粒度分块累加，累加顺序为 `k = 0, block_k, 2·block_k, …` 的顺序 K-tile 累加，累加器为 fp32（`accum_dtype="float"`，WGMMA 累加）。分块累加的加法分组与「一次 fp32 扁平求和」不同，但同为 fp32 域，误差在 fp16/bf16 容差内。
- **dtype 语义**：输入 fp16/bf16，**累加必须 fp32**（源 `accum_dtype="float"`；NPU 侧同样要求 fp32，见 `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.3：fp16×fp16 场景 dst/累加必须 fp32，否则可能运行时卡死）；输出统一在 fp32 累加器上单次舍入到输入 dtype。
- **边界语义**：
  - 非整除：源码 TMA 装载 16B 对齐约束（每个输入最内连续维须为 8 元素倍数）；K 尾部由 TMA **零填充**，M/N 尾块在 epilogue 用 `if m_start+i < m and n_start+j < n` 守卫，只写有效区域。
  - `nan`/`inf` 通过 matmul 自然传播，无掩码；空维度不出现（GEMM 无 0-size 维用例）。GEMV fast path 的**精确条件**（源码 `ops/gemm.py` 的 `_get_kernel`）为 `m==1 and not trans_a and trans_b`（lhs_row）或 `n==1 and not trans_a and not trans_b`（rhs_col）——满足其一才进 `_gemv_kernel`，否则进 `_gemm_kernel`。**注意**：默认 NT（`trans_b=True`）下 `n==1` **不**走 GEMV（rhs_col 要求 `not trans_b`），仍由本 kernel 处理。
  - Sink 语义：matmul 无 max/sum 稳定化问题，无溢出风险（fp16 输入已定值域，fp32 累加）。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现（`C = op(A) @ op(B)`，torch.matmul），未复刻 §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `a` | `(M,K)`（`trans_a=False`）或 `(K,M)`（`trans_a=True`） | float16 / bfloat16 | 左操作数，逻辑 `op(A)=[M,K]` |
| `b` | `(K,N)`（`trans_b=False`）或 `(N,K)`（`trans_b=True`） | same_as(a) | 右操作数，逻辑 `op(B)=[K,N]` |
| `c` (输出) | `(M,N)` | same_as(a) | 输出矩阵，ND 连续布局，无 transpose |

清单依据：`examples/TileOPs/tileops/manifest/gemm.yaml` 的 `GemmOp.signature` / `shape_rules`；`trans_a` 默认 `false`、`trans_b` 默认 `true`。

### 0.3 实现算法解读（怎么算）

源码 `_gemm_kernel(m, n, k, trans_a, trans_b, dtype, traced)` 为工厂函数，返回 `_gemm_func`；调用 `_gemm_func(block_m=128, block_n=128, block_k=64, num_stages=3)` 得到编译后的 `_gemm_main`。手工 warp-specialization：producer warpgroup（`tx<128`，128 线程）用 TMA 把 A/B tile 装入 `num_stages` 深 SMEM ring，consumer warpgroup（`tx>=128`，128 线程）用 WGMMA 沿 K 累加；`threads=256` 固定。`pass_configs={"tl.disable_warp_specialized": True}` 关闭自动 WS pass。

**计算步骤分解**（源码每个计算语句均归入某步骤）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | Grid 映射：`T.Kernel(ceildiv(n,block_n), ceildiv(m,block_m))` → `by=blockIdx.y`(M 块)、`bx=blockIdx.x`(N 块) | m,n,block_m,block_n | 每块的 `m_start=by*block_m`、`n_start=bx*block_n` | 输出矩阵分块 |
| 2 | 分配 `a_smem[num_stages, a_tile]`、`b_smem[num_stages, b_tile]`、`c_local[block_m,block_n]`（fp32 fragment）、`ab_full/ab_empty` barriers、迭代计数 `gi_prod/gi_cons` | — | 片上缓冲 | 累加器与输入驻留 |
| 3 | `T.annotate_layout`：`make_swizzled_layout(a_smem/b_smem)` | — | SMEM swizzle 布局 | 消 bank conflict（实现层） |
| 4 | Producer：TMA 装载 A tile（`trans_a` 决定切法）、B tile（`trans_b` 决定切法）入 ring slot `s`，`barrier_wait(ab_empty[s],…)`→`tma_copy(...,barrier=ab_full[s])`→`barrier_arrive(ab_full[s])` | GM A/B | `a_smem[s]`、`b_smem[s]` | 数据搬入 |
| 5 | Consumer：`barrier_wait(ab_full[s],phase)`→`T.wgmma_gemm(a_smem[s],b_smem[s],c_local, transpose_A=trans_a, transpose_B=trans_b, clear_accum=(ki==0))`；`T.wait_wgmma(0)`、`warpgroup_fence_operand`、`barrier_arrive(ab_empty[s])` | `a_smem[s]`,`b_smem[s]`,`c_local` | 更新 `c_local` | `C += A_tile @ B_tile`（K 累加） |
| 6 | Epilogue：`for i,j in T.Parallel(block_m,block_n): if m_start+i<m and n_start+j<n: c[m_start+i,n_start+j]=c_local[i,j]` | `c_local` | GM `c` | 输出写回（M/N 尾块守卫） |
| 7 | 寄存器重分配 `T.dec_max_nreg(24)`（producer）/`T.inc_max_nreg(240)`（consumer）；`T.alloc_var` 迭代计数；`trace.*` 时间线标记（GPU-only） | — | — | 实现层调度 |

**数据流与内存访问模式**（源硬件视角）：

```
GM[A] --TMA(cp.async.bulk)--> SMEM a_smem[num_stages,a_tile] --WGMMA--> register fragment c_local(fp32) --T.Parallel--> GM[C]
GM[B] --TMA(cp.async.bulk)--> SMEM b_smem[num_stages,b_tile] --WGMMA-->   ▲
   ring slot 由 gi%num_stages 选，phase=(gi//num_stages)%2；producer→consumer 用 ab_full/ab_empty mbarrier 配对
   K 尾由 TMA 零填充；M/N 尾由 epilogue 守卫；swizzle 布局缓解 SMEM bank conflict
```

**循环与并行结构**：

- 两级并行：grid（`ceildiv(n,block_n) × ceildiv(m,block_m)` 块，每块一个 `(M,N)` 输出 tile）+ 块内 warpgroup（producer/consumer）。
- 块内 K 迭代：`for ki in T.serial(k_iters)`，`k_iters=ceildiv(k,block_k)`；producer 与 consumer 各维护独立迭代计数，异步流水（`num_stages=3` 环形缓冲）。
- `threads=256`（两个 warpgroup × 128）。

**host 侧逻辑**：

- `GemmKernel.__init__` 调 `_gemm_kernel(m,n,k,trans_a,trans_b,dtype_str)` 建 kernel；`forward(a,b)` 用 `**self.config`（默认 `block_m=128,block_n=128,block_k=64,num_stages=3`）编译并执行；`traced=trace.enabled` 控制是否物化时间线标记。
- `GemmOp.forward` 从输入 shape 推断逻辑 `(m,n,k)`，按 `m==1 / n==1` 决定 GEMV fast path，否则用 `GemmKernel`；输出直接返回。
- 无输入预处理 / 输出后处理（无 im2col / crop / transpose）。
- `_gemm_wrapped_kernel`（`torch.library.custom_op`（GPU 侧 `top` 命名空间注册为 `gemm_wrapped_kernel`））仅用于 `torch.compile` 兼容，不在 eager 路径。

**编译配置**：`@tilelang.jit(out_idx=trace.out_idx(1,traced), pass_configs={"tl.disable_warp_specialized": True}, compile_flags=["-O3","-DENABLE_BF16"])`；`functools.lru_cache(maxsize=32)` 缓存工厂。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | SMEM tiling（`alloc_shared` + 分块搬运） | 减 GM 重复访问、提高数据复用 | A/B tile 驻留 SMEM，被同一 block 的 K 循环复用 | shared memory + L2 | 硬件强相关（NPU 有 L1 等价） |
| 2 | TMA 异步块搬运（`T.tma_copy`） | 大块高带宽搬运、免线程逐元素地址计算 | `cp.async.bulk` + 16B 对齐 | Hopper TMA 引擎 | 硬件强相关（NPU 有 MTE2/DMA 等价） |
| 3 | 多级流水环形缓冲（`num_stages=3`，`a_smem[num_stages,…]`） | 搬入与计算重叠 | stage 索引 + phase 位翻转 | 异步拷贝引擎 + SMEM | 模式可移植（`T.Pipelined`） |
| 4 | 手工 warp specialization（producer/consumer 两个 warpgroup） | 专用角色、隐藏访存延迟 | `tx<128` 分流，不同职责 | warp/warpgroup（32/128 线程） | 硬件强相关（NPU 无 warp 概念） |
| 5 | WGMMA tensor-core MMA + fp32 累加器 | 矩阵算力 | `T.wgmma_gemm` 分形 MMA | tensor core（Hopper WGMMA） | 硬件强相关（NPU 有 Cube 等价） |
| 6 | Swizzled SMEM 布局（`make_swizzled_layout`） | 消 SMEM bank conflict | 地址重排 | SMEM bank 结构 | 硬件强相关（NPU 无 bank conflict） |
| 7 | mbarrier 生产者-消费者同步（`alloc_barrier`/`barrier_wait/arrive`） | 环形缓冲正确同步 | 每 slot 一对 full/empty barrier | Hopper mbarrier | 硬件强相关 |
| 8 | 寄存器重分配（`dec_max_nreg`/`inc_max_nreg`） | 给 consumer 更多寄存器装累加器 | setmaxnreg | 大寄存器堆 | 硬件强相关 |
| 9 | `warpgroup_fence_operand` | WGMMA 操作数可见性 | 寄存器 fence | Hopper WGMMA | 硬件强相关 |
| 10 | Epilogue M/N 尾块守卫 | 支持非整除 M/N | `if` 逐元素守卫 | — | 可移植 |
| 11 | K 尾零填充（TMA 天然零填充） | 支持非整除 K | TMA OOB 补零 | TMA | 可移植（需 NPU 等价实现） |
| 12 | 工厂 `lru_cache` + 每 shape 编译 + `config` 启发式 | 降 host 开销、shape 特化 | 缓存编译产物 | — | 可移植 |
| 13 | `T.Parallel` epilogue | 输出写回并行 | 线程并行写 | SMEM/寄存器 | 可移植 |
| 14 | `trace.*` in-kernel 时间线标记 | 性能剖析 | 标记物化 + `slots` 输出 | — | GPU-only，NPU 无等价（`tileops.trace` 无 NPU 版本） |

> 识别不出 ≠ 不存在：以上 14 项为源码中可识别的优化手段，逐项在 §0.5 判定处置，避免静默丢弃。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定依据映射表：tilelang-op-design skill `references/migration-analysis.md` §5.3；NPU 能力佐证：`docs/开发指南.md` §2/§3、`docs/Tilelang.language/线性代数操作/T.gemm.md`、`docs/Tilelang.language/内存操作/T.copy.md`、`examples/gemm/example_gemm.py`、`examples/gemm/matmul.py`。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 语义（`C=op(A)@op(B)`、fp32 累加、边界守卫） | 语义 | 无 | — | **保留** | 语义层无条件保留（§0.1） |
| SMEM tiling（#1） | 优化 | shared memory | 有（L1 512KB / UB 192KB） | **等价替换** | Developer 模式 `T.alloc_shared` → L1（Cube 输入），tile 复用意图不变；`docs/开发指南.md` §2.1；`examples/gemm/example_gemm.py` |
| TMA 异步搬运（#2） | 优化 | Hopper TMA | 有（MTE2/DMA） | **等价替换** | `T.copy`（GM→L1，Developer 模式自动 ND→NZ）；`docs/Tilelang.language/内存操作/T.copy.md` §2.4 |
| 多级流水环形缓冲（#3） | 优化 | 异步引擎 + SMEM | 有 | **等价替换** | `T.Pipelined(k_iters, num_stages=S)`，stage 数按 L1 容量重算；`docs/开发指南.md` §3.2 |
| 手工 warp specialization（#4） | 优化 | warp/warpgroup | **无**（NPU 无 warp 概念，`T.Kernel(threads=)` 在 npuir 无效） | **重新设计** | → §0.6 R1：单核 `T.Pipelined` 软流水（MTE2 预取与 Cube 计算重叠） |
| WGMMA（#5） | 优化 | tensor core WGMMA | 有（Cube，`T.gemm`→`hivm.hir.mmadL1`） | **等价替换** | `T.gemm(src1,src2,dst,initC=…,a_transpose=…,b_transpose=…)`；dst fp32；`docs/Tilelang.language/线性代数操作/T.gemm.md` |
| Swizzle 消 bank conflict（#6） | 优化 | SMEM bank | **无此问题** | **舍弃** | NPU 无 SMEM bank conflict，NZ 分形布局由搬运/框架处理；migration-analysis §5.3「swizzle→舍弃」；`references/nz-format.md` |
| mbarrier 同步（#7） | 优化 | Hopper mbarrier | 有（框架自动同步） | **等价替换** | Developer 模式 `T.Pipelined` 自动插入同步（`T.alloc_barrier` 不可移植） |
| 寄存器重分配（#8） | 优化 | 大寄存器堆 setmaxnreg | **无** | **舍弃** | NPU 无 `dec/inc_max_nreg` 对应；累加器落 L0C（128KB）而非寄存器堆 |
| `warpgroup_fence_operand`（#9） | 优化 | WGMMA 寄存器 fence | **无** | **舍弃** | 随 #4/#5 一并去除；`T.gemm` 由框架管理依赖 |
| Epilogue M/N 尾块守卫（#10） | 算法/优化 | 无 | 有 | **等价替换** | 动态 extent 切片 `T.copy(C_local[0:tm,0:tn], C[m0:m0+tm,n0:n0+tn])`；`docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 规则 2 |
| K 尾零填充（#11） | 算法层 | TMA（补零） | 有（`T.clear` 或等价） | **等价替换** | 对 A/B tile 的 K 尾显式清零（§0.6 R4）；零元素不改变点积 |
| 工厂缓存 / shape 特化（#12） | 优化（host） | 无 | 有 | **保留** | 保留 `lru_cache` + 每 shape 编译；host 侧只做 shape/stride 元数据 |
| `T.Parallel` epilogue（#13） | 优化 | 线程并行 | 有 | **等价替换** | 输出写回交由 `T.copy`（MTE/VPipe 向量化），不出现逐元素标量循环（§1.6.2） |
| `trace.*` 标记（#14） | 优化（可观测性） | GPU-only tracer | **无 NPU 等价** | **舍弃** | `traced` 参数、`trace.group/range/dag/finalize/run`、`out_idx=trace.out_idx(...)` 全部去除；`target="npuir"` 不支持该仪器化 |
| Stage 0 抽取件的 `str2dtype`（仅 `_gemv_kernel` 用）/ `_Trace` 替身 | helper | GPU 专有 | 无 | **舍弃** | 均不在 `_gemm_kernel` 内，NPU 重写丢弃（任务 scaffold 提示） |
| `get_sm_version()`（op 层）/ `supported_archs=[90]` | 优化（host） | CUDA SM | 无 | **舍弃** | 已在 wrappers `kernels/gemm/gemm/gemm.py` 内改为 `supported_archs=None` |

> 判定统计：**保留 3 项**（语义、host 工厂缓存、语义层累加/fp32 精度）、**等价替换 8 项**、**重新设计 1 项**（#4，另见 R3/R4 重设计子项）、**舍弃 5 项**（#6/#8/#9/#14 + helper）。所有舍弃项均写明理由，未静默丢弃。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」及必须重写落地的条目逐项给出 NPU 新算法（重设计对照 Phase R 调研结论 §1.6.0：调研选定「单遍分块 Cube 直接法」，故此处不引入算法族替换，只做硬件组织重设计）。

**重设计项 R1：warp-specialized 异步流水 → 单核 `T.Pipelined` 软流水**

- **源方案**：producer warpgroup 用 TMA 预取、consumer warpgroup 用 WGMMA 计算，两者经 mbarrier 环形缓冲重叠；依赖 warp 分流、`threads=256`、`setmaxnreg`。性能意图 = 让 GM→SMEM 搬运（MTE2）与 MMA 计算（Cube）充分重叠，隐藏访存延迟。
- **NPU 新算法**：单一种类核心内用 `for kk in T.Pipelined(T.ceildiv(K, block_K), num_stages=S)` 组织 K 循环，由编译器自动插入 MTE2（GM→L1 装载）与 Cube（`T.gemm`）之间的多缓冲同步。这正是 `docs/开发指南.md` §2 的 NPU GEMM 结构（`examples/gemm/example_gemm.py` 官方示例同款）。
- **语义保持论证**：仅改变搬运/计算的调度方式，不改变 K 的累加顺序（仍为 k=0… 的顺序 tile 累加）与 fp32 累加域，故数学等价；边界语义沿用 §0.6 R4。性能意图（搬运-计算重叠）由 `num_stages` 承接。

**重设计项 R2：2D grid → 一维 `T.Kernel` + cid 分解 + 分核策略**

- **源方案**：`T.Kernel(ceildiv(n,block_n), ceildiv(m,block_m))` 二维 grid，`(bx,by)` 直接映射 N/M 块。
- **NPU 新算法**：`T.Kernel(num_launched, is_npu=True) as (cid, _)` 一维；`num_logical = ceildiv(M,block_M)*ceildiv(N,block_N)`，**`num_launched = num_logical`**（非 persistent：核内无多任务循环，网格数必须恒等于逻辑块数，**禁止用 `min(...)` 截断**，否则 `num_logical > 物理核数` 时只启动物理核数个核、漏写输出块），核内 `by = cid // ceildiv(N,block_N)`、`bx = cid % ceildiv(N,block_N)`。分核三要素见 §5.5。
- **语义保持论证**：因 `num_launched == num_logical`，`cid → (by,bx)` 为双射，每个逻辑输出块恰由一个核计算，无重复/遗漏写。边界：`cid ∈ [0, num_logical)` 天然成立（网格数即逻辑块数，无需额外守卫）；超发部分（`num_logical > 物理核数`）由运行时串行调度，不改变逐块计算语义。

**重设计项 R3：WGMMA transpose flags → `T.gemm` transpose 参数 + 布局相关 tile 形状**

- **源方案**：SMEM tile 形状随存储布局变化（`a_tile=(block_k,block_m)` if `trans_a` else `(block_m,block_k)`；`b_tile=(block_n,block_k)` if `trans_b` else `(block_k,block_n)`），WGMMA 用 `transpose_A/B` 调和为逻辑 `(M,K)×(K,N)`。
- **NPU 新算法**：同样按 `trans_a`/`trans_b` 决定 shared tile 形状，`T.gemm(..., a_transpose=trans_a, b_transpose=trans_b, initC=(kk==0))`。NPU `T.gemm` 的 `size=[a,b,c]` 语义（src1 `[a,b]` 或 `[b,a]`、src2 `[b,c]` 或 `[c,b]`、dst `[a,c]`）与之一致。
- **语义保持论证**：转置仅改变操作数存储访问方式，`op(A)`、`op(B)` 的逻辑值不变；`initC` 保证首次迭代清零、其余累加，与源 `clear_accum=(ki==0)` 等价。

**重设计项 R4：非整除 M/N/K 的核内处理（取代 TMA 零填充 + epilogue 守卫）**

- **源方案**：K 尾由 TMA 自动零填充；M/N 尾在 epilogue 用逐元素 `if` 守卫，只写有效区域。
- **NPU 新算法**：host 侧禁止 padding（`references/ascend-constraints.md` §4），故全部在 kernel 内：
  - `tm = T.min(block_M, M - m0)`、`tn = T.min(block_N, N - n0)`、`tk = T.min(block_K, K - k0)`；
  - 装载用动态 extent 切片（`T.copy(A[m0:m0+tm, k0:k0+tk], A_shared[0:tm,0:tk])`，`docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 规则 2 支持尾块动态 extent）；
  - **K 尾零填充**：A_shared 与 B_shared 的 K 尾区（`k>=tk`）先 `T.clear` 清零再写有效区（否则 A/B 尾区垃圾相乘会污染全部输出元素——A 行/列越界部分与 B 对应部分相乘非零）；
  - **M/N 尾无需清零**：A 的多余行只影响不被写回的 C 行，B 的多余列只影响不被写回的 C 列（matmul 行/列独立性）；
  - 写回用动态 extent 切片 `T.copy(C_local[0:tm,0:tn], C[m0:m0+tm, n0:n0+tn])`，等价源 epilogue 守卫。
- **语义保持论证**：零填充项贡献 `0·x=0`（x 为有限值时）故不改变点积——本设计对 A、B 的 K 尾**都**清零，杜绝 `0·NaN`/`0·inf` 的 IEEE 陷阱；M/N 尾只影响被守卫不写回的元素。M/N/K 非整除行为与源算子一致（输出有效区数值相同）。等价性经 §1.6.1 E2 机器验证（`EQUIV_PASS`）。

**重设计项 R5：GPU-only trace 仪器化 → 去除**

- **源方案**：`traced` 参数、`trace.group/range/dag/finalize/run`、`out_idx=trace.out_idx(1,traced)`、`slots` 附加输出。
- **NPU 新算法**：全部去除；`@tilelang.jit(out_idx=[-1], target="npuir")`，C 为唯一输出。
- **语义保持论证**：tracing 仅记时间线（disabled 时对 CUDA 亦为零成本 no-op），去除不影响数值与 I/O（`slots` 输出为 GPU 专有）。

### 0.7 标杆实现

- 源算子路径（golden 依据）：`/tmp/opencode/TileOPs-ref/tileops/kernels/gemm.py`（`_gemm_kernel`，行 301–493）；源 op 类 `/tmp/opencode/TileOPs-ref/tileops/ops/gemm.py`（`GemmOp`）。
- 源仓参考实现 / 测试基准：`examples/TileOPs/tests/ops/test_gemm.py` 的 `GemmTest.ref_program`（`torch.matmul(a.T if trans_a, b.T if trans_b)`）；`examples/TileOPs/tileops/workloads/gemm.py`（`gen_inputs`，按 trans 生成 A/B 存储布局）。
- golden 以 §0.1 语义为依据，直接移植上述 `ref_program`（matmul + 可选转置），不复刻 §0.6 的 NPU 算法。
- 源仓 Stage 0 抽取件（参考，非实现基线）：`examples/TileOPs/tileops/kernels/gemm/gemm/_gemm_kernels.py`（GPU 工厂 + `_Trace`/`str2dtype` 兼容替身）。

## 1. 概述

### 1.1 算子名称

`_gemm_kernel`（项目 `gemm`；NPU 实现文件为本算子目录下的 `_gemm_kernel.py`，由 Stage 3 产出，当前尚未生成）。

### 1.2 功能描述

稠密矩阵乘 `C = op(A) @ op(B)`，覆盖 NT/NN/TN/TT 四种存储布局，fp16/bf16 输入、fp32 累加、输出 dtype 同输入；支持非整除 M/N/K（kernel 内处理，无 host padding）。迁移后为 NPU 侧 Developer 模式 Cube 实现（tile 分块 + `T.Pipelined` K 流水 + `T.gemm`）。

### 1.3 数学公式

$$
C[m,n] = \sum_{k=0}^{K-1} \mathrm{op}(A)[m,k]\cdot \mathrm{op}(B)[k,n], \qquad
\mathrm{op}(A)=
\begin{cases}
A, & \text{trans\_a=False}\ (A:[M,K])\\
A^{\mathsf T}, & \text{trans\_a=True}\ (A:[K,M])
\end{cases},\quad
\mathrm{op}(B)=
\begin{cases}
B, & \text{trans\_b=False}\ (B:[K,N])\\
B^{\mathsf T}, & \text{trans\_b=True}\ (B:[N,K])
\end{cases}
$$

其中 `K` 为收缩维，累加在 fp32 域，输出 `C:[M,N]`。

### 1.4 算法描述

**迁移决策后的 NPU 侧算法**（与 §1.6.0 调研选定算法一致）：

1. **输出分块**：将 `C[M,N]` 划分为 `ceil(M/block_M) × ceil(N/block_N)` 个 `block_M×block_N` 输出块，一维 grid 每 slot 处理一个逻辑块（R2）。
2. **K 维分块累加**：对每个输出块，沿 K 以 `block_K` 分块，`T.Pipelined` 流水（MTE2 装载 A/B tile 到 L1 ↔ Cube `T.gemm` 累加到 fp32 `C_local`），`initC=(kk==0)` 等价 `T.clear` + 累加（R1/R3）。
3. **尾块处理**：M/N/K 非整除用动态 extent 切片装载 + K 尾零填充 + 写回守卫（R4）。
4. **写回**：`T.copy(C_local → C)`，fp32→输入 dtype 单次舍入（§1.6.1 E1）。

与源算法的结构差异（来源）：
- 源用手工 warp specialization + TMA + mbarrier 环形缓冲（§0.4 #2/#3/#4/#7）；NPU 改为单核 `T.Pipelined` 软流水（§0.6 R1）——性能意图（搬运-计算重叠）由编译器多缓冲承接。
- 源二维 grid（§0.3 步骤 1）；NPU 一维 grid + cid 分解 + 分核策略（§0.6 R2，§5.5）。
- 源 SMEM swizzle + `setmaxnreg` + `warpgroup_fence` + trace（§0.4 #6/#8/#9/#14）；NPU 全部舍弃（§0.5）。
- 源非整除用 TMA 零填充 + epilogue 守卫；NPU 用动态 extent 切片 + 显式 K 尾清零（§0.6 R4）——语义等价经 §1.6.1 E2 机器验证。

### 1.5 数据流图

```
GM[A] --T.copy(MTE2)--> L1 A_shared[block_M,block_K] ┐
GM[B] --T.copy(MTE2)--> L1 B_shared[block_K|N,...]  ┴--> T.gemm(Cube, mmadL1) --> L0C C_local[block_M,block_N] fp32
                                                                     │  (K 流水：T.Pipelined num_stages=S)
                                                                     ▼
                                       T.copy(VPipe/VCast 单次舍入 fp32→out dtype) --> GM[C]
```

### 1.6 算法调研与优化分析

> 分析对象 = §1.6.0 调研选定算法（单遍分块 Cube 直接法），经 §0.6 重设计落地的 NPU 算法。本节结论是 §3.1 与 §6 的输入。

#### 1.6.0 算法调研（Algorithm Research）

调研深度：**完整调研**（矩阵类算子，方法论见 `references/algorithm-research.md` §2；检索入口 `python3 .agents/tools/kb_search.py "gemm matmul cube L1 L0C"`）。检索源：`references/algorithm-candidates.md`（§5 R1/R2 参考表，matmul/conv 族条目）、`pattern-library/INDEX.md` 与 `attention.md`（PL-1.8 bf16 Cube 直连）、`examples/gemm/`、`examples/TileOPs/`、源算子 §0.3/§0.4。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线：分块 Cube 直接法 `C=op(A)@op(B)`（源算法） | `Σ_k a[m,k]·b[k,n]`，K 分块累加（`block_k`） | — | — | ✅（基线） |
| 1 | Strassen（2 层分块） | 2×2 分块用 7 次子乘替代 8 次 | 精确算术下恒等；浮点下**改变累加分组与误差界**，数值不稳 | 乘法次数 8→7（-12.5%），但加法次数与临时块增加 | ✅（R3 对比） |
| 2 | split-K（K 维并行分解） | 把 K 拆成 P 段独立部分和，再归约 | 数学恒等（加法结合律，fp32 分组变化） | FLOPs 不变；访存+中间缓冲增加；可并行度↑ | ✅（R3 对比） |
| 3 | 降低精度（fp16 累加 / HF32 输入） | 在 fp16 域累加 | **非等价**（显著改变数值，且 NPU fp16 dst 会卡死） | 少算无实质收益 | ❌（语义违反，`docs/Tilelang.language/线性代数操作/T.gemm.md` §2.3） |
| 4 | 转置布局改写为 NT | 经 transpose 把 TN/TT 化为 NT | 数学恒等（同一次 matmul） | 不改 FLOPs；NPU `T.gemm` 已支持 `a/b_transpose`，无需改写 | ❌（无收益，§0.6 R3） |
| 5 | epilogue cast 融合（单次舍入） | fp32 累加→store 时一次 cast（不落 fp32 中间） | 数学恒等（同一 fp32 值单次 RNE） | **非回归约束**：不引入 fp32 中间 GM 往返；源 epilogue 本就单次融合，故相对源算子增量为 0（定位见 §1.6.1） | ✅（R3 对比；采纳项 E1） |

**R2 在线算法**：**无「在线」变体适用于本语义**（结论「基线即天然单遍」）。结构依据：GEMM 的 K 累加本就是一个输出 tile 的 running accumulator（O(block) 中间状态），无需两次扫描；不存在聚合统计量需要跨 tile 重缩放。可选的**分块并行**变体是 split-K（R1 候选 2）——它不是 online 而是并行分解，当 M/N 小、K 极大（并行度不足）时才有价值。本 manifest 的 mid-m 档（m=16..96，K=2048..7168）中 m=16 时 M 分块仅 1 行，N 分块（block_N=128）已有 32 个，并行度充足，无需 split-K。收益口径（split-K）：扫描遍数不变、中间缓冲 `M·N·P` 落 GM、跨核归并代价 ↑。

**R3 复杂度对比**（四口径；基线在表；口径含/不含已注明）：

| 算法候选 | FLOPs | 访存量 Bytes | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|-------------|---------|-------------|---------------------|
| 基线（分块 Cube 直接法，ideal traffic） | `2·M·N·K`（含乘加=2） | 理想：`(M·K+N·K+M·N)·2` B（每元素读一次；含输出写）；**不含** tiling 造成的 tile 重读（由 L2 吸收） | 1（K 单遍顺序累加） | `block_M·block_N·4`（fp32 L0C，max 64KB@128×128） | 高；无跨核归并 |
| Strassen（2 层） | `~0.875·(2MNK)`（乘法↓，但加法↑，净 FLOPs 未必↓） | 与基线同阶（更多临时块） | 1 | 多个子块临时（UB/L0C 压力↑） | 高；数值不稳 |
| split-K（P 段） | `2·M·N·K`（不变） | 理想 + `M·N·P·4` 部分和写读 | 1（K 仍单遍，但分段） | `M·N·P·4`（跨核 partial 落 GM） | 更高（P×网格）；需二阶段归约或 atomic |
| 候选 5（epilogue cast 融合，**非回归约束**） | `2·M·N·K` | 相对伪基线「额外物化 fp32 输出再 cast」少 `2·M·N·4`（写 + 读回）；**相对源算子 0**（源已融合） | 1 | 无（融合） | 同基线 |

> 口径说明：FLOPs 按乘加=2；访存为 GM 流量（不把 L1/L0C 内部搬运计入）；理想口径假设张量经 L2 复用，tiling 重读不落 HBM（设计期近似）。带宽受限性：本算子为**算力/Cube 受限 + 操作数流受限**，而非纯带宽受限（见下 R4 及估算行）。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附佐证）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（分块 Cube） | ✅ MAC 密集 → Cube（`T.gemm`→`mmadL1`） | L0C 128KB：128×128×4B=64KB ✅；L1 512KB：A+B tile×stages ≤ 192KB ✅ | ✅ block_M/N≥16（分形）；block_K 16 倍数（32B 尾轴） | ✅ K/M/N 分块边界静态（M,N,K 编译期常量） | ✅ `T.Pipelined` 多缓冲；Cube-only 无需 CV 融合 | **✅ 选定** |
| Strassen | ❌ Cube 为 MAC 阵列，乘法次数减少不转化为 throughput；需额外加法/重排（Vector 侧） | 临时块增加 | ❌ 子块分形对齐复杂 | ✅ | ❌ 破坏单循环流水 | ❌ 淘汰（收益无法在 Cube 兑现 + 数值不稳） |
| split-K | ✅ Cube | ✅ | ✅ | ✅ | 需 partial 中间矩阵落 GM（`CONST-store-fixpipe-gm-only`：跨引擎强制 GM 往返）+ 二阶段归约 | ❌ 本 manifest 并行度已足，净亏（并行度收益 < 归约代价）；仅作 K-极大/ M,N-极小场景的**备选**（§1.6.3） |
| 候选 5（epilogue cast 融合） | ✅ 随基线 | ✅ 无额外缓冲 | ✅ | ✅ | ✅ 融合进 L0C→GM store | **✅ 采纳（E1）** |

**调研结论**：选定 **基线「单遍分块 Cube 直接法」+ epilogue cast 融合（E1）+ K 尾零填充（E2）**。关键依据：R3 显示 Strassen 的乘法削减在 Cube MAC 阵列上无 throughput 收益且数值不稳（R4 淘汰）；split-K 在本 manifest 的并行度下净亏（R4 淘汰）；基线在 R4 六项检查全绿，且 `examples/gemm/example_gemm.py` / `docs/开发指南.md` §2 已给出 NPU Developer 官方形态。与基线的结构差异：无（算法族沿用源算法的单遍分块结构，仅做硬件组织重设计 §0.6，不替换算法）。**调研范围**：`references/algorithm-candidates.md` matmul/conv 条目、`pattern-library/INDEX.md`（PL-1.8 bf16 Cube 直连、CONST-L1-port-bw、CONST-store-fixpipe-gm-only）、`examples/gemm/`、`examples/TileOPs/tileops/kernels/gemm/`、源算子 §0.3/§0.4、模型知识（Strassen/split-K）。源算法已识别的 14 项优化手段（§0.4）承接情况：算法层优化「K 尾零填充」由 E2 承接、「epilogue 守卫」由 R4 承接；实现层优化经 §0.5 四态处置（等价替换/重新设计/舍弃）逐项承接，无静默丢弃。

**设计期估算（D-2）**：代表 workload `ds-v3-prefill-attn-proj`（M=4096, N=4096, K=7168, bf16）。
- 流量项：理想 GM 流量 `(M·K+N·K+M·N)·2 = 150.99 MB`；按 `CONST-mte2-degradation`（2026-09-07 build / CANN 8.5.0）混合读写地板 ~1.26 TB/s ⇒ ≈ **120 µs**；按 `CONST-fabric-bw`（1.73 TB/s，乐观上界）⇒ ≈ 87 µs。
- 发射项：Cube 段向量发射 ≈ 0；MTE2 指令数 = `(K/block_K)×2` 块/核 × 逻辑核数，相对可忽略。
- 容量项：L0C 128KB 单次驻留，epilogue 单次 L0C→GM（无跨引擎往返）⇒ ≈ 0。
- `估算下界: ≈120 µs（流量 120 + 发射 ~0 + 容量 ~0，引用 CONST-mte2-degradation / CONST-fabric-bw）`。**Cube 计算项无实测常数**（pattern-library 未覆盖 Cube throughput），标注「未实测假设」：Cube 吞吐若低于流量下界对应的算力需求，则实际由算力项主导——交 Stage 4 实测对账（hardware-cost-model §3）。常数版本戳：tilelang 0.1.2+ed787bb（2026-09-07 build）/ CANN 8.5.0 / Ascend910B2C。

#### 1.6.1 数学等价优化（公式级）

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| E1 | Epilogue cast 融合（**非回归约束**，非增量优化） | 伪基线：把 `op(A)@op(B)` 的 fp32 结果额外物化回 GM，再单独 cast 到 out dtype（**非**源算子行为） | `C = round_to_dtype(op(A)@op(B))`，在 L0C→GM store 内**单次**舍入 | 数学恒等：两式均对同一 fp32 累加器做一次 RNE 舍入到目标 dtype，比特级一致；关键前提是**经过 fp32 opmath 域**（避免 fp64→fp16 直接舍入的双重舍入偏差，见 `TRAP-fp16-opmath-golden`）。**重新定位**：源算子 epilogue 本就是 `c[m,n]=c_local[i,j]`（fp32 fragment→输出 dtype 的单次融合 cast），故 E1 **不是相对源算子的增量优化，而是「不得引入 fp32 中间 GM 往返」的非回归约束** | 收益口径（相对上述伪基线，非源算子）：省 `2·M·N·4` B（fp32 写 + 读回，代表 shape ≈ 128 MB 往返）；**相对源算子 → 0 增量收益**（源已融合，本项仅冻结该行为不被回退） |
| E2 | K 尾零填充 | 对精确 K 求和 `Σ_{k<K}` | 对补齐到 `block_K` 倍数的 K 求和，A/B 尾区填零：`Σ_{k<K'} `，`K'=ceil(K/block_K)·block_K` | 数学恒等：新增项 `a[m,k]·b[k,n]=0·0=0`（A、B 尾区**都**清零，杜绝 `0·NaN`/`0·inf`）；仅改变 fp32 求和的加零操作，不改数值 | 使 `block_K ∤ K` 时免 host padding（host padding 被 `references/ascend-constraints.md` §4 禁止）；clear 代价仅尾区，可忽略 |

> 其余候选（Strassen / split-K / 精度降低 / 转置改写）在 R1/R3/R4 已判不采纳，此处不重复论证。

**机器验证（D-1）**：`examples/gemm/_gemm_kernel/verify_equiv.py`（torch CPU，fp64 参照 + fp32 opmath 域；随机张量 + 角点值 ±inf/NaN/−0/subnormal + 边界 shape；两 dtype）。**E1 的两条对照分支为不同计算路径**（融合单次舍入 vs 显式 fp32 GM 物化 + 分离 cast），非同一赋值；脚本另打印「fp64→fp16 直接舍入」与「fp64→fp32→fp16 双重舍入」的差异探针（信息性，非门禁行）。执行结果（`python3 examples/gemm/_gemm_kernel/verify_equiv.py`）：

| 采纳项 | dtype | 最大 ulp 差 | 违反率 | 结论 |
|--------|-------|------------|--------|------|
| E1-fused-vs-materialised | torch.float16 | 0.0 | 0.000000 | `EQUIV_PASS` |
| E1-fused-vs-materialised | torch.bfloat16 | 0.0 | 0.000000 | `EQUIV_PASS` |
| E2-ktail-zerofill | torch.float16 | 0.0 | 0.000000 | `EQUIV_PASS` |
| E2-ktail-zerofill | torch.bfloat16 | 0.0 | 0.000000 | `EQUIV_PASS` |

> E1 行名由 `E1-epilogue-cast-fusion` 改为 `E1-fused-vs-materialised`：反映「融合 store」与「先物化 fp32 再 cast」两条**不同代码路径**的比特级等价（非回归约束），而非原恒等式 `acc32.to(dtype) == acc32.to(dtype)`。E2 行为精确 K（`Σ_{k<K}`）与零填充 K（`Σ_{k<K'}`，A/B 双侧填零）的对照。**双重舍入探针**（信息性）：fp16 中点扫描下 `fp64→fp16` 直接舍入与 `fp64→fp32→fp16` 双重舍入失配 1599926 例，验证 E1「必须经过 fp32 opmath 域单次舍入」前提的必要性（本 torch build 的 fp64→16bit 转换本身经 fp32，故探针改用 numpy 直接舍入路径）。

**优化结论**：采纳 2 项——**E1**（epilogue cast 融合，**非回归约束**：冻结「不引入 fp32 中间 GM 往返」的行为，相对源算子 0 增量收益）与 **E2**（K 尾零填充，使 `block_K ∤ K` 时免 host padding），均 `EQUIV_PASS`。**优化后公式**（供 §3.1 使用）：
`C[m,n] = round_to_dtype( Σ_{k=0}^{K'-1} op(A)[m,k]·op(B)[k,n] )`，其中 `K'=ceil(K/block_K)·block_K`，且 `k≥K` 时 `op(A)=op(B)=0`（K 尾零填充），`round_to_dtype` 在 store 内单次舍入（E1）。Dense GEMM 的核心乘加无其他代数可约结构（Strassen 已淘汰）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | K 维累加循环 | 源 `T.serial(k_iters)` WGMMA 累加环 | Cube `T.gemm`（matrix MAC，非逐元素）；搬运由 `T.Pipelined` 流水 | ✅（Cube 承载） | — |
| 2 | A/B tile 装载 | 源 TMA `tma_copy` | `T.copy`（GM→L1，MTE2 向量化搬入） | ✅ | — |
| 3 | 输出写回 + cast | 源 `T.Parallel` 逐元素守卫写 | `T.copy(C_local[0:tm,0:tn], C[...])`（MTE/VPipe 向量化 + 单次 VCast 舍入） | ✅ | — |
| 4 | 零填充 A/B 的 K 尾 | 源 TMA 天然零填充 | `T.clear`（向量化 memset） | ✅ | — |
| 5 | block 起始索引 `m0=by*block_M`、`n0=bx*block_N`、`cid→(by,bx)` | 标量乘加/整除 | 无 | ❌ | **block 级索引 / 任务映射计算**：每 block 仅数个标量，非逐元素热点；无对应的逐元素向量等价 API（索引不构成张量） |
| 6 | `num_logical` / `num_launched` / `tm,tn,tk` 边界 | 标量算术 | 无 | ❌ | **host 侧 / kernel 元数据计算**：编译期常量或每 block 一次的标量边界，非逐元素热点 |
| 7 | K 流水循环本身（`T.Pipelined`） | 迭代 | 无 | ❌ | **tile 级顺序依赖**：`C_local` 跨 K-tile 累加，循环携带依赖；已用多缓冲流水隐藏延迟，无法用归约替代（归约需先物化全 K，破坏 tiling） |

**向量化结论**：全部**逐元素**计算已向量化（装载 `T.copy`、写回 `T.copy`+VCast、零填充 `T.clear`）；**矩阵乘由 Cube `T.gemm` 承载**（非 Vector）。保留 3 处标量 / 循环，理由类别：block 索引 / 任务映射（#5）、host 与 kernel 元数据（#6）、tile 级顺序依赖（#7）——均非逐元素热点。§6 循环结构不含任何逐元素标量循环，与本节一致。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> Cube 类的"向量化轴"退化为 **Cube 分形（NZ/fractal）布局与 epilogue 归属**决策（骨架按 tilelang-cube-skill）。I/O layout（ND，A `(M,K)`/B `(K,N)`/C `(M,N)`）是契约，核内 Cube 布局是设计变量。

**候选矩阵**（全部枚举；弃选给量化理由）：

| # | 布局方案 | Cube 数据路径 | repack 路径 | 预估收益 / 代价 | 是否采纳 |
|---|---------|--------------|------------|----------------|---------|
| 1 | **Developer 模式 ND-preserving**：`T.alloc_shared` + `T.copy`，框架自动 ND→NZ、L0C→GM | GM→L1（自动 ND2NZ）→L0A/L0B→Cube→L0C→VCast→GM | 无（框架内） | 官方示例形态，零显式 repack；`examples/gemm/example_gemm.py` | **✅ 采纳（主选）** |
| 2 | Expert 模式显式 NZ：`T.Scope("Cube")`+`T.alloc_L1`+`T.load_nd2nz`+`T.store_fixpipe` | GM→L1(`load_nd2nz`)→Cube→L0C(`store_fixpipe`) | 无（显式） | 显式控制 TMA/NZ，性能上限可能更高；但 `TRAP-DEVMODE-PERSIST-GEMM` 显示 Developer+persistent+gemm 崩溃类结构须 Expert；开发复杂度更高 | ❌ 备选（§1.6.3 结论，Stage 4 A/B） |
| 3 | host 侧 permute 输入到最优布局 | — | host permute（`PL-1.3-host-permute` 实测 106–147µs） | repack 代价 > 算子本体收益；且 host 禁改输入真实内容（`references/ascend-constraints.md` §4） | ❌ |
| 4 | 显式 `T.transpose` 把 TN/TT 转成 NT 再算 | 核内二轴交换链 | `PL-1.1-transpose-chain` ~5.4µs/链 | 无收益：NPU `T.gemm` 原生支持 `a_transpose/b_transpose`（§0.6 R3），转置是多余的核内开销 | ❌ |

**逐候选评分（主选 #1 vs 备选 #2）**：整除性——Cube 分形要求 `block_M,block_N≥16`、尾轴（K/N）16 元素（32B）对齐，两者均满足（`block_M=block_N=128`、`block_K` 16 倍数）；repack 代价——#1 由框架在搬运内完成 ND→NZ，#2 显式 `load_nd2nz`，两者均无 GM 级全量重排（正确口径：GM 流量不变，代价在 NZ 转换的 MTI/MTE 指令开销）；UB/L0C 容量——`block_M·block_N·4 ≤ 128KB`（`CONST-capacity-910B2C`），两者共用同一 L0C 预算；流水/融合——#1 用 `T.Pipelined`，#2 用 `T.serial`+`T.load_nd2nz`（显式双缓冲）。判定裕度：两者差异在于 NZ 转换是框架自动还是显式，以及 Developer/Expert 模式的能力边界（`TRAP-DEVMODE-PERSIST-GEMM`）——属未实证常数（NZ 转换指令开销 vs 框架自动路径），故按裁决模式保留 #2 为备选。

**布局决策结论**：主选 **#1 Developer 模式 ND-preserving**（`T.alloc_shared` + `T.copy` + `T.gemm`，框架自动 ND→NZ 与单次 VCast）；向量化/量化轴 = Cube 分形（`mmadL1`），无逐元素向量轴；repack 路径 = 无（框架内）。该结论同步落入 §3.3 伪代码、§4 内存规划（L1/L0C 缓冲形状）、§6 循环结构。**备选 #2（Expert 显式 NZ）**独立成节：结构 = `Scope("Cube")` + `alloc_L1/alloc_L0C` + `load_nd2nz/store_fixpipe`（见 §3.3 备选伪代码），随 DESIGN 冻结进入 Stage 4 A/B 清单。**实验裁决计划**：代表 shape = `ds-v3-prefill-attn-proj`、`wide-n-24576`、`mid-m16-attn`；指标 = kernel latency（msprof）+ 反解 NZ 转换开销；判定阈值 = 备选 latency < 主选 × 0.95 才翻转；结果回写 §1.6.3 与 pattern-library。**设计期探针**：本设计不额外起探针（主选有官方 example 佐证，NZ 转换常数交 Stage 4 实测）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式: Developer**（主选；§1.6.3 备选 #2 若在 Stage 4 胜出则切 Expert——届时为受控设计修订，非默认）。

### 2.2 选型理由

- 算子为纯 Cube（`T.gemm`），无手写 NZ/L1/L0C 需求；Developer 模式提供 `T.alloc_shared`（Cube 输入自动映射 L1）+ `T.copy`（自动 ND→NZ、自动同步）+ `T.gemm`（自动 `initC`）的完整能力，覆盖 §0.6 R1–R4 的全部需求。
- 官方 NPU Cube 示例 `examples/gemm/example_gemm.py` 与 `docs/开发指南.md` §2 均以 Developer 模式给出，是最近模板（AGENTS.md「Operator Implementation Baseline」优先改动既有案例）。
- 源 GPU 的手工 warp specialization / swizzle / `setmaxnreg` 在 NPU 无对应能力（§0.5），Developer 的自动同步/自动布局正好替代手工部分。
- `op_requirements.programming_mode = developer`（用户显式指定）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（Cube 输入 → L1）、`T.alloc_fragment`（累加器 → L0C）；不显式用 `alloc_L1/alloc_L0C` |
| 计算方式 | `T.gemm(A_shared, B_shared, C_local, initC=(kk==0), a_transpose=…, b_transpose=…)` |
| 数据搬运 | `T.copy`（GM↔shared/fragment），自动 ND→NZ、自动 dtype cast、自动同步 |
| 同步 | 自动（`T.Pipelined` 由编译器插入多缓冲同步），无手动 barrier |

> **模式边界风险**：`TRAP-DEVMODE-PERSIST-GEMM`（`traps-runtime.md`，verified）实测 **Developer + persistent 分核 + gemm 混排运行时崩溃**（"unaligned UUB addresses"），Expert 同结构正常。故本设计 Developer 主选**不采用 persistent 核内串行**（§5.5 分核方案据此选择），极大规模 persistent 化列为 Stage 4 Expert 备选（§5.5 / §9.1）。

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（K 尾零填充 + 单次舍入）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `m0 = by·block_M`，`n0 = bx·block_N` | 由一维 `cid` 恢复输出块起点 |
| 2 | `tm=min(block_M, M-m0)`，`tn=min(block_N, N-n0)` | M/N 尾块有效范围 |
| 3 | `k0 = kk·block_K`，`tk=min(block_K, K-k0)` | K 分块起点与有效长度 |
| 4 | `C_local += op(A)[m0:m0+tm, k0:k0+tk] · op(B)[k0:k0+tk, n0:n0+tn]` | 分块累加（含 K 尾零填充） |
| 5 | `C[m0:m0+tm, n0:n0+tn] = round_dtype(C_local)` | 有守卫写回 + 单次舍入 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | Grid + cid→(by,bx) | `T.Kernel(num_launched, is_npu=True)` | 一维 grid；`by=cid // ceildiv(N,block_N)`、`bx=cid % ceildiv(N,block_N)` | Developer |
| 2 | 分配 Cube 输入 | `T.alloc_shared((block_M, block_K), dtype)`；`T.alloc_shared((block_N, block_K), dtype)`（NT） | 形状随 trans 布局（§0.6 R3） | Developer |
| 3 | 分配累加器 | `T.alloc_fragment((block_M, block_N), "float32")` | fp32 累加（`T.gemm.md` §2.3） | Developer |
| 4 | K 循环流水 | `for kk in T.Pipelined(T.ceildiv(K, block_K), num_stages=S)` | `S` 按 L1 容量（§4.5） | Developer |
| 5 | 装载 A/B tile | `T.copy(A[m0:m0+rM, k0:k0+rK], A_shared[0:rM,0:rK])` | 动态 extent 尾块（`T.copy.md` §2.2.2 规则 2） | Developer |
| 6 | K 尾零填充 | `T.clear(A_shared[...])` / `T.clear(B_shared[...])`（或全 clear + 覆盖写） | 仅当 `rK < block_K`；否则整块覆盖免清零 | Developer |
| 7 | 矩阵乘累加 | `T.gemm(A_shared, B_shared, C_local, initC=(kk==0), a_transpose=trans_a, b_transpose=trans_b)` | → `hivm.hir.mmadL1` | Developer |
| 8 | 写回 + cast | `T.copy(C_local[0:tm,0:tn], C[m0:m0+tm, n0:n0+tn])` | 自动 VCast fp32→out dtype（RNE），动态 extent 守卫 | Developer |

API 依据：`docs/Tilelang.language/线性代数操作/T.gemm.md`、`docs/Tilelang.language/内存操作/T.copy.md`（§2.2.2/§2.4）、`docs/Tilelang.language/内存操作/T.clear.md`、`examples/gemm/example_gemm.py`、`examples/gemm/matmul_dynamic_shape.py`（动态 extent + 尾块）。

### 3.3 计算伪代码

```python
import tilelang
import tilelang.language as T
from tilelang.utils import NPUUtils

# --- host: 分核策略参数（编译期常量；物理核数实查值 20，见 §5.5）---
# kernel 工厂按 (m, n, k, trans_a, trans_b, dtype) 特化
@tilelang.jit(out_idx=[-1], target="npuir")
def _gemm_kernel(m, n, k, trans_a, trans_b, dtype="float16",
                 block_M=128, block_N=128, block_K=128, num_stages=3):
    accum_dtype = "float32"
    # 布局相关 tile 形状（§0.6 R3）：逻辑 op(A)=[M,K], op(B)=[K,N]
    a_rows, a_cols = (block_K, block_M) if trans_a else (block_M, block_K)
    b_rows, b_cols = (block_N, block_K) if trans_b else (block_K, block_N)
    n_blocks = T.ceildiv(n, block_N)
    num_logical = T.ceildiv(m, block_M) * n_blocks
    num_physical = NPUUtils.get().get_aicore_num()   # 实查（本机返回 20，见 §5.5）
    # 非 persistent 一维网格：网格数必须恒等于逻辑块数（禁止 min 截断，否则漏写输出块）
    num_launched = num_logical

    @T.prim_func
    def main(
        A: T.Tensor((k, m) if trans_a else (m, k), dtype),
        B: T.Tensor((n, k) if trans_b else (k, n), dtype),
        C: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(num_launched, is_npu=True) as (cid, _):
            by = cid // n_blocks
            bx = cid % n_blocks
            m0 = by * block_M
            n0 = bx * block_N
            tm = T.min(block_M, m - m0)
            tn = T.min(block_N, n - n0)

            A_shared = T.alloc_shared((a_rows, a_cols), dtype)
            B_shared = T.alloc_shared((b_rows, b_cols), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            for kk in T.Pipelined(T.ceildiv(k, block_K), num_stages=num_stages):
                k0 = kk * block_K
                rK = T.min(block_K, k - k0)
                # K 尾零填充（仅尾块不整除时）：先整块清零、再覆盖写有效区；
                # A、B 双侧都清，杜绝 0*NaN / 0*inf 污染（§0.6 R4 / E2）
                if rK < block_K:
                    T.clear(A_shared)
                    T.clear(B_shared)
                # 装载：按 trans 布局映射到逻辑 [M,K]/[K,N] 块（§0.6 R3，双分支）
                if trans_a:      # A 存储 [K,M]：切片 [k,m] → A_shared (block_K, block_M)
                    T.copy(A[k0:k0+rK, m0:m0+tm], A_shared[0:rK, 0:tm])
                else:            # A 存储 [M,K]：切片 [m,k] → A_shared (block_M, block_K)
                    T.copy(A[m0:m0+tm, k0:k0+rK], A_shared[0:tm, 0:rK])
                if trans_b:      # B 存储 [N,K]（默认 NT）：切片 [n,k] → B_shared (block_N, block_K)
                    T.copy(B[n0:n0+tn, k0:k0+rK], B_shared[0:tn, 0:rK])
                else:            # B 存储 [K,N]：切片 [k,n] → B_shared (block_K, block_N)
                    T.copy(B[k0:k0+rK, n0:n0+tn], B_shared[0:rK, 0:tn])
                T.gemm(A_shared, B_shared, C_local,
                       initC=(kk == 0), a_transpose=trans_a, b_transpose=trans_b)

            T.copy(C_local[0:tm, 0:tn], C[m0:m0+tm, n0:n0+tn])

    return main
```

> 上伪代码为 **主选方案 #1（Developer）**。`trans_a`/`trans_b` 的 `T.Tensor` 形状、`A_shared`/`B_shared` 形状与切片索引**已按转置语义展开为双分支**：`trans_a` 时 `A:[k,m]`，切片 `A[k0:k0+rK, m0:m0+tm]` 载入 `(block_K, block_M)` 的 `A_shared`；`trans_b` 时 `B:[n,k]`，切片 `B[n0:n0+tn, k0:k0+rK]` 载入 `(block_N, block_K)` 的 `B_shared`（**默认 NT 走此分支**）。数据类型直接用第二个位置参数传入（不用 `dtype=` 关键字），避免 false alarm。

**备选方案 #2（Expert 显式 NZ，随 DESIGN 冻结进 Stage 4 A/B）**：

```python
with T.Kernel(num_launched, is_npu=True) as (cid, _):
    with T.Scope("Cube"):
        by = cid // n_blocks; bx = cid % n_blocks
        A_L1 = T.alloc_L1([block_M, block_K], dtype)
        B_L1 = T.alloc_L1([block_K, block_N], dtype)
        C_L0C = T.alloc_L0C([block_M, block_N], accum_dtype)
        for kk in T.serial(T.ceildiv(k, block_K)):
            T.load_nd2nz(A[by*block_M, kk*block_K], A_L1, [block_M, block_K])
            T.load_nd2nz(B[kk*block_K, bx*block_N], B_L1, [block_K, block_N])
            T.gemm(A_L1, B_L1, C_L0C, initC=(kk == 0),
                   b_transpose=False, size=[block_M, block_K, block_N])
            T.store_fixpipe(C_L0C, C[by*block_M, bx*block_N],
                            size=[block_M, block_N], enable_nz2nd=True)
```
（备选结构依据 `examples/gemm/matmul.py`；UB/L1/L0C 预算、分核三要素、循环结构按本任务粒度重算，见 §4/§5/§6。备选仅在 Stage 4 实测胜出（< 主选 ×0.95）时启用。）

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|----------|----------|
| `T.Kernel(..., is_npu=True)` | `examples/gemm/example_gemm.py`、`docs/开发指南.md` §2 | 官方示例已验证 |
| `T.alloc_shared` / `T.alloc_fragment` | `examples/gemm/example_gemm.py` | 官方示例已验证 |
| `T.gemm(..., initC=, a_transpose=, b_transpose=)` | `docs/Tilelang.language/线性代数操作/T.gemm.md`、`examples/gemm/example_gemm.py` | 文档 + 示例 |
| `T.copy(切片, 切片)`（动态 extent 尾块） | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 规则 2、§2.4 示例 | 文档 |
| `T.clear` | `docs/Tilelang.language/内存操作/T.clear.md`、`docs/开发指南.md` §2 示例 | 文档 |
| `T.Pipelined(..., num_stages=S)` | `docs/开发指南.md` §3.2、`examples/gemm/example_gemm.py` | 官方示例已验证 |
| `NPUUtils.get().get_aicore_num()` | `docs/开发指南.md` §3.3；实查返回 20 | 实查已验证（本机） |
| `T.load_nd2nz`/`T.alloc_L1`/`T.alloc_L0C`/`T.store_fixpipe`（备选 #2） | `docs/Tilelang.language/内存操作/*.md`、`examples/gemm/matmul.py` | 文档 + 示例（备选） |
| `T.alloc_barrier`/`T.tma_copy`/`T.wgmma_gemm`/`T.dec_max_nreg`/`T.warpgroup_fence_operand`/`trace.*` | 源 GPU 代码（§0.4） | **不使用**（GPU-only，§0.5 舍弃） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | 否（源为二维 grid，本设计用一维） | 一维 `T.Kernel` + cid 分解（§0.6 R2） |
| GPU 专用 API 不可用 | 是（TMA/WGMMA/mbarrier/swizzle/setmaxnreg/trace） | 逐项四态处置（§0.5），NPU 等价 API 替换或舍弃 |
| GEMM 要求 M,N 为 block 整数倍 | **是**（m=16/32/64/96、n=2112 等非整除） | kernel 内动态 extent 切片 + K 尾零填充 + 写回守卫（§0.6 R4）；**不做 host padding** |
| L0C 容量上限 128KB | 是 | `block_M·block_N·4 ≤ 131072` ⇒ 128×128=64KB ✅（`CONST-capacity-910B2C`） |
| 物理核数限制 | 是 | §5.5 分核三要素（实查 20） |
| host 侧禁止改动输入 / padding | 是 | 全部逻辑在 kernel 内；host 只做 shape/stride 元数据（§0.3、`references/ascend-constraints.md` §4） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU SM90） | 本项目（Ascend NPU） | 转换方案 |
|--------|---------------------|---------------------|----------|
| Kernel 维度 | 二维 grid `T.Kernel(n_blocks, m_blocks)` | 一维 + cid 分解 | §0.6 R2 |
| 流水/同步 | 手工 producer/consumer warpgroup + TMA + mbarrier | `T.Pipelined` 自动多缓冲 | §0.6 R1 |
| GEMM API | `T.wgmma_gemm`（transpose_A/B） | `T.gemm`（a_transpose/b_transpose, initC） | §0.6 R3 |
| 内存层级 | SMEM + register fragment + swizzle | L1（alloc_shared）+ L0C（alloc_fragment） | §0.5 #1/#6 |
| 非整除 | TMA 零填充 + epilogue 守卫 | 动态 extent 切片 + K 尾 `T.clear` + 写回守卫 | §0.6 R4 |
| 仪器化 | `trace.*` + `slots` 输出 | 去除 | §0.6 R5 |
| dtype/累加 | fp16/bf16 → fp32 WGMMA 累加 | fp16/bf16 → fp32 `T.gemm` 累加（fp16 dst 会卡死，`T.gemm.md` §2.3） | 保留 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/gemm/example_gemm.py` | 高度相似 | Developer 模式 Cube GEMM：一维 grid + cid 分解、`alloc_shared`/`alloc_fragment`、`T.Pipelined` + `T.gemm(initC=)`、`T.copy` 写回 |
| `examples/gemm/matmul_dynamic_shape.py` | 高 | 动态 extent 尾块（`T.min`）与 Expert 显式 NZ 路径（备选 #2） |
| `examples/gemm/matmul.py` | 高 | Expert 模式 `alloc_L1`/`alloc_L0C`/`load_nd2nz`/`store_fixpipe`（备选 #2 结构） |
| `examples/TileOPs/tileops/kernels/gemm/gemm/gemm.py` | 高（wrapper） | NPU 侧 `GemmKernel`/`GemmOp` wrapper（K5/K8/K9 适配）；本 kernel 的上层契约 |
| `examples/TileOPs/tileops/kernels/gemm/gemm/_gemm_kernels.py` | 高（源） | GPU 工厂抽取件（参考语义，非实现基线） |
| `examples/TileOPs/tests/ops/test_gemm.py` | 高 | golden `ref_program`（matmul + 转置）+ 容差（fp16 1e-3 / bf16 1.6e-2） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `a` | `(m,k)` 或 `(k,m)`（trans_a） | float16 / bfloat16 | 逻辑 `op(A)=[m,k]` |
| `b` | `(k,n)` 或 `(n,k)`（trans_b） | same_as(a) | 逻辑 `op(B)=[k,n]` |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `c` | `(m,n)` | same_as(a) | ND 连续，无 transpose |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `A_shared` | `(block_M, block_K)`（trans_a=False）/ `(block_K, block_M)`（trans_a=True） | dtype | L1 | A tile（含 K 尾零填充区） |
| `B_shared` | `(block_N, block_K)`（trans_b=True）/ `(block_K, block_N)`（trans_b=False） | dtype | L1 | B tile（含 K 尾零填充区） |
| `C_local` | `(block_M, block_N)` | float32 | L0C | fp32 累加器 |

### 4.4 内存搬运路径

```
GM[A] --T.copy (MTE2, 自动 ND→NZ)--> L1 A_shared[block_M, block_K] ┐
GM[B] --T.copy (MTE2, 自动 ND→NZ)--> L1 B_shared[block_N|K, ...]   ┴--T.gemm(mmADL1)--> L0C C_local[block_M,block_N] fp32
                                                                              │    (K 流水，num_stages 深 ring)
                                                                              ▼
                              T.copy(L0C→GM, VPipe 单次 VCast fp32→out dtype, 动态 extent 守卫) --> GM[C]
宿主边界：host 仅做 (m,n,k) 推断、kernel 选择与 shape/stride 元数据，不触碰张量内容、不 padding。
```

### 4.5 UB 内存预算

> 本算子为 Cube 路径：输入驻留 **L1**（512KB），累加器驻留 **L0C**（128KB）。UB 仅承载自动 VCast 的写回中转（每 block `block_M×block_N×out_dtype`），按 `CONST-capacity-910B2C`（2026-08-24 build）UB=192KB、auto-multi-buffer 膨胀 ~1.7x 评估。

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| A_shared ×(num_stages=3) | 3 × (128,128) | fp16 | 3×32768 = 98304 |
| B_shared ×(num_stages=3) | 3 × (128,128) | fp16 | 3×32768 = 98304 |
| **L1 总计** | | | 196608 / 524288 (512KB) ✅（含 auto-multi-buffer 膨胀 ~1.12× → ~220KB，仍 < 512KB） |
| C_local | (128,128) | fp32 | 65536 / 131072 (L0C 128KB) ✅ |

说明：`block_K=128`、`num_stages=3` 时 L1 = 192KB（未计膨胀）；若工具链 auto-multi-buffer 膨胀作用于 L1，按 ~1.12×（`CONST-capacity-910B2C` 2026-09-15 重校）≈ 220KB，余量充足。`block_K=256, num_stages=2` 亦可行（256KB）。buffer 形状随 trans 布局互换但字节相同。

### 4.6 动态轴定义

首次设计按 manifest 的 `(m,n,k)` 编译期特化（每 shape 一个 kernel，`lru_cache` 缓存），**无运行时动态轴**。若要支持动态 shape（`T.symbolic`），参考 `examples/gemm/matmul_dynamic_shape.py`，但本任务 manifest 为静态 shape，不引入。

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| 无 | — | — |

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[-1], target="npuir")
def _gemm_kernel(m, n, k, trans_a, trans_b, dtype="float16",
                 block_M=128, block_N=128, block_K=128, num_stages=3):
    ...
```

（去除源 GPU 的 `pass_configs={"tl.disable_warp_specialized": True}` 与 `compile_flags=["-O3","-DENABLE_BF16"]`——npuir 目标不需要；若 Stage 3 遇到 codegen 问题再按 `tilelang-error-fixer` 加最小 pass_config。）

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Cube（仅 `T.gemm` + `T.copy`，**无用户编写的**逐元素 Vector 算子链）。

**判定依据**: 算子仅含矩阵乘 `T.gemm` + 数据搬运 `T.copy`，无用户编写的逐元素 Vector 算子链。epilogue 的 fp32→out dtype 舍入由编译器自动插入的 `VCastOp`（Vector 单元）在 L0C→GM 路径内承载，属**后端自动管理的隐式 Cube→Vector 协同**（`docs/Tilelang.language/内存操作/T.copy.md` §2.3）；它不是用户编写的 Vector 算子链，故无需手写 `Scope("Cube")/Scope("Vector")` 双 Scope。

### 5.2 Block 划分

```python
block_M = 128   # 分形 M≥16；L0C: 128*128*4=64KB ≤128KB；M 轴整除性好（16~4096 中 128 的块数合理）
block_N = 128   # 分形 N≥16；与 block_M 对称；NT 布局尾轴 K 由 block_K 保证对齐
block_K = 128   # A 尾轴（NT 下 block_K）；16 元素（32B）对齐；K/gcd = K/128 整除全部 manifest K
num_stages = 3  # L1 预算内（A/B 共 6 个 tile = 192KB）；K 迭代数 ≥ K/128 = 8~128，流水充分
num_logical = ceildiv(M, block_M) * ceildiv(N, block_N)
```

> 启发式：M 很小（m≤96）时可将 `block_M` 降到 16/32/64 以减少 M 维浪费，但需保持 ≤ L0C 预算且 ≥16；由 `default_config` 之外的启发式按 shape 选择（Stage 3/4 落地）。

### 5.3 约束分析

- **对齐约束**: `block_K=128`、`block_N=128` 均为 16 的整数倍 ⇒ fp16/bf16 尾轴（32B = 16 元素）满足；`block_M/block_N ≥ 16` 满足 Cube 分形。
- **L1 容量**: A/B shared 共 `2×(block_M×block_K + block_K×block_N)×2B×num_stages = 192KB < 512KB` ✅（`CONST-capacity-910B2C`）。
- **L0C 容量**: `block_M×block_N×4 = 64KB ≤ 128KB` ✅。
- **K 对齐**: `gcd(manifest K) = 128`，`block_K=128` 对全部 workload 整除 K ⇒ K 尾零填充在本 manifest 不触发（R4 仍实现以增强健壮性）。

### 5.4 注意事项

- **非整除 M/N**：`m ∈ {16,32,64,96,…}` 与 `n=2112` 非 `block` 整数倍，必须 kernel 内处理（§0.6 R4）：动态 extent 切片装载 + 写回守卫；**禁止 host padding**。
- **零 block 风险**：使用 `T.ceildiv` 而非整除，避免 `M < block_M` 时零 block 启动；`tm/tn` 保证有效区 >0。
- **trans 布局**：`block_K` 在 `trans_b=False` 时是 B 的**非尾轴**（B `[K,N]`，尾轴 N 由 `block_N` 保证），此时 `block_N` 需为 16 倍数；已满足。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1；依据 `docs/开发指南.md` §3.3。

- **物理核数**: `NPUUtils.get().get_aicore_num()` **实查值 = 20**（Cube 算子直接使用返回值，不翻倍）。
  查询代码与实际执行：
  ```python
  from tilelang.utils.npu_utils import NPUUtils
  print(NPUUtils.get().get_aicore_num())   # 本机实查输出: 20
  ```
  （设备：Ascend910B2C，2026-10-13 实查。）
- **逻辑核数**: `num_logical = ceil(M/block_M) × ceil(N/block_N)`。按 `block_M=block_N=128` 对 manifest workload 计算：smoke 128³=1；square-1k=64；decode-gate-up 1×17=17；decode-down 1×56=56；prefill-gate-up 32×17=544；prefill-down 32×56=1792；prefill-attn-proj 32×32=1024；k-dominant 32×56=1792；wide-n 32×192=6144；mid-m16 1×32=32；mid-m32 1×32=32；mid-m64 1×56=56；mid-m96 1×17=17。
- **规模判定**: 三档并存：
  1. **小规模**（`num_logical ≤ 20`，仅 smoke=1）→ 逻辑核数 ≤ 物理核数，**无需适配**；启动 `num_logical` 个核。
  2. **中等规模**（`20 < num_logical ≲ 60`，如 mid-m16/32=32、mid-m64=56、decode-down=56）→ 通过调整 `block_M/block_N` 使内核总数接近物理核数 20 的整数倍（按实查核数取 1×/2×/3× = 20/40/60）：如 32 ≈ 40（可调至 `block_N=128→160` 或接受 2 波），56 ≈ 60（减小 block_N 使 N 块数×M 块数 ≈ 60）。取值在 Stage 3/4 实测确定，原则是内核总数落在 20 的整数倍附近，消除"21 个内核→某核双倍任务"式负载不均。
  3. **极大规模**（`num_logical ≫ 20`，如 544~6144）→ 无法通过 block 缩减到物理核数整数倍（受 L0C 128KB 上限约束 `block_M·block_N ≤ 32768`）。标准方案是**固定启动核数 = 物理核数，核内 `T.serial` 串行处理多逻辑块**。**但 `TRAP-DEVMODE-PERSIST-GEMM`（`traps-runtime.md`，verified，toolchain 0.1.2+1990aa9fe4）实测 Developer+persistent+gemm 混排运行时崩溃，Expert 同结构正常**。故本设计：
     - **主选（Developer）**：采用**非 persistent 一维网格**，且**固定 `num_launched = num_logical`**——无论 `num_logical ≤ 20` 还是 `> 20`，都启动 `num_logical` 个核（`num_logical ≤ 20` 时无超发、无需适配；`num_logical > 20` 时超出物理核数的部分由运行时串行调度，**不裁剪内核数**）。**网格数不变量**：因本设计的核内**没有 persistent / `T.serial` 多任务循环**，`num_launched` 必须恒等于 `num_logical`；若写成 `min(num_logical, num_physical)`，`num_logical > 20` 时只启动 20 个核，`cid ∈ [0,20)` 只会覆盖前 20 个逻辑块，其余输出块**静默漏写**（本 manifest 绝大多数 workload 的 `num_logical` = 32~6144 > 20，Stage 3 首个精度用例即失败）。权衡：本 manifest 大 shape 的 K 迭代充分（K/block_K = 8~128），每块计算量大，核启动开销相对被摊薄；且 L2 复用使超发调度不显著劣化。
     - **备选（Expert，Stage 4 A/B）**：若实测显示超发调度开销显著（如 wide-n-24576），按标准 §1 要素③ 切 Expert 实现 persistent：`T.Kernel(num_physical)` + 核内 `for t in T.serial(tasks_per_kernel)`（`tasks_per_kernel` 为编译期常量，静态边界）+ `cid = t*num_physical + kernel_id`；结构见 `docs/开发指南.md` §3.3 示例。循环边界静态（`num_logical`、`num_physical` 均编译期常量）。
- **分核方案（主选，可实施）**:
  - **网格数不变量**：`num_launched = num_logical`（恒成立，非 persistent）。§0.6 R2、§3.3 伪代码、§6.2 均以本行为准。
  - `num_logical ≤ num_physical`：启动 `num_logical` 个核，一核一逻辑块，无负载不均、无超发。
  - `num_logical > num_physical`：启动 `num_logical` 个核（非 persistent，Developer 安全，**不裁剪为 20**）；每核处理一个逻辑块，超发由运行时串行调度；**不得写 `min(...)`**。
  - 不采用 `T.serial` 核内串行（避免 Developer+persistent+gemm 崩溃）；persistent 化留作 Stage 4 Expert 备选（届时才成立 `num_launched = num_physical` + 核内 `T.serial` 边界 guard 范式）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化（`T.copy`/`T.clear`/`T.gemm`）；下表循环仅指 block 级 / tile 级调度循环，**无逐元素标量循环**。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 输出块（M×N） | block 级并行 | `T.Kernel(num_launched, is_npu=True)` + cid 分解 | 每 slot 一个逻辑输出块（§0.6 R2） |
| K 方向 | tile 级流水迭代 | `for kk in T.Pipelined(ceildiv(K,block_K), num_stages=3)` | K 分块累加，MTE2 装载与 Cube 计算重叠（§0.6 R1） |
| tile 内元素 | 向量化（MTE/Cube） | `T.copy`（装载/写回）、`T.gemm`（Cube MAC）、`T.clear`（零填） | §1.6.2；无标量逐元素循环 |

### 6.2 循环伪代码

```python
# num_launched = num_logical（§5.5 网格数不变量；非 persistent，禁止 min 截断）
with T.Kernel(num_launched, is_npu=True) as (cid, _):
    by = cid // n_blocks; bx = cid % n_blocks
    m0 = by * block_M; n0 = bx * block_N
    tm = T.min(block_M, m - m0); tn = T.min(block_N, n - n0)
    A_shared = T.alloc_shared((a_rows, a_cols), dtype)
    B_shared = T.alloc_shared((b_rows, b_cols), dtype)
    C_local = T.alloc_fragment((block_M, block_N), "float32")
    for kk in T.Pipelined(T.ceildiv(k, block_K), num_stages=3):
        k0 = kk * block_K; rK = T.min(block_K, k - k0)
        # K 尾零填充（仅尾块；A、B 双侧都清，杜绝 0*NaN）
        if rK < block_K:
            T.clear(A_shared)
            T.clear(B_shared)
        # 装载：trans 双分支（默认 NT 走 trans_b 支）
        if trans_a:
            T.copy(A[k0:k0+rK, m0:m0+tm], A_shared[0:rK, 0:tm])
        else:
            T.copy(A[m0:m0+tm, k0:k0+rK], A_shared[0:tm, 0:rK])
        if trans_b:
            T.copy(B[n0:n0+tn, k0:k0+rK], B_shared[0:tn, 0:rK])
        else:
            T.copy(B[k0:k0+rK, n0:n0+tn], B_shared[0:rK, 0:tn])
        T.gemm(A_shared, B_shared, C_local, initC=(kk == 0),
               a_transpose=trans_a, b_transpose=trans_b)
    T.copy(C_local[0:tm, 0:tn], C[m0:m0+tm, n0:n0+tn])
```

### 6.3 流水线优化

使用 `T.Pipelined(k_iters, num_stages=3)`：编译器自动生成 MTE2 装载与 Cube 计算的多缓冲流水。`num_stages=3` 依据 §4.5 L1 预算（192KB < 512KB）；`k_iters = ceil(K/block_K) = 8~128`（manifest），远大于 `docs/开发指南.md` §3.2 建议的 4~6，流水可充分填充。与源 GPU 的手工 producer/consumer 双 warpgroup 流水等价（§0.6 R1）。

### 6.4 尾块处理

- **M/N 尾块**：动态 extent 切片（`tm/tn`）装载与写回（`docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 规则 2）；写回只覆盖有效区，不写坏相邻 GM（避 `TRAP-T-copy-region-semantics` 的越界写）。
- **K 尾块**：`rK < block_K` 时先 `T.clear` A_shared/B_shared 的尾区再装载，保证 `k≥K` 项贡献 0（§0.6 R4，E2）。本 manifest 的 K 均被 `block_K=128` 整除，不触发。
- **对齐填充**：尾轴非 16 倍数时编译器自动填充（`docs/开发指南.md` §3.1），`block_K/block_N=128` 规避。

---

## 7. 同步策略

### 7.1 同步模式

**模式: 自动同步**（Developer 模式）。

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|------|----------|------|
| K 流水 MTE2↔Cube | 由 `T.Pipelined` 编译器自动插入多缓冲同步 | Developer 模式自动同步；替代源 GPU 的 `alloc_barrier`/`barrier_wait`/`barrier_arrive`（§0.5 #7） |
| L0C→GM 写回 | 由 `T.copy` 自动插入依赖同步 | 保证 Cube 累加完成后才 VCast 写出 |
| 跨核 | 无需 | 每个逻辑块独立，无跨核数据交换（无 split-K） |

无手动同步点（无 `T.barrier_all`/`T.sync_block_set/wait`）。

### 7.3 pass_configs 配置

```python
pass_configs = {}   # Developer 模式默认；源 GPU 的 tl.disable_warp_specialized 不适用 npuir
```

若 Stage 3 出现 codegen 问题，按 `tilelang-error-fixer` / `TRAP-DEVMODE-PERSIST-GEMM` 的 Expert 经验（`TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False`）最小化加入，不在设计期预设。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（移植源仓 `GemmTest.ref_program`），**不得复刻 §0.6 的 NPU 算法**——保证验证独立性。

```python
import torch

def golden_gemm(a: torch.Tensor, b: torch.Tensor,
                trans_a: bool = False, trans_b: bool = True) -> torch.Tensor:
    """基于 PyTorch 的参考实现（源算子语义）：C = op(A) @ op(B)。

    a: (M,K) if not trans_a else (K,M)
    b: (K,N) if not trans_b else (N,K)
    return: (M,N), dtype == a.dtype
    """
    if trans_a:
        a = a.T
    if trans_b:
        b = b.T
    return torch.matmul(a, b)
```

（等价于 `examples/TileOPs/tests/ops/test_gemm.py` 的 `GemmTest.ref_program`；workload 生成见 `examples/TileOPs/tileops/workloads/gemm.py` 的 `GemmWorkload.gen_inputs`。）

### 8.2 精度标准

| dtype | atol | rtol | 依据 |
|-------|------|------|------|
| float16 | 1e-3 | 1e-3 | `examples/TileOPs/tests/ops/test_gemm.py` 的 `test_gemm` |
| bfloat16 | 1.6e-2 | 1.6e-2 | 同上 |
| float16（GEMV 边界档） | 1e-2 | 1e-2 | 同上（`GEMV` 档，非本任务） |
| 通用参照 | 1e-2 | 1e-2 | `docs/开发指南.md` §3 示例 |

> 累加域为 fp32、输出单次舍入（§1.6.1 E1），误差主要来自 fp32→fp16/bf16 舍入与 K 分块加法分组，落在上述容差内。

### 8.3 L0 门槛测试计划（交 `tilelang-op-develop` 执行）

| 档位 | shape (m,n,k,trans_a,trans_b,dtype) | 目的 |
|------|-----------------------------------|------|
| smoke | 128,128,128,False,True,float16 | 最小 NT 基线（index 等价性 + 编译通过） |
| smoke | 128,128,128,False,True,bfloat16 | bf16 Cube 直连（`PL-1.8-bf16-cube-direct`） |
| NN 方形 | 1024,1024,1024,False,False,float16 | trans_b=False 路径 |
| 非整除 N | 128,2112,7168,False,True,bfloat16 | N 尾块（2112 非 128 倍数） |
| 小 M | 16,4096,7168,False,True,bfloat16 | M 尾块（m=16 < block_M） |
| 小 M | 96,2112,7168,False,True,bfloat16 | M/N 双尾块 |

- 每档：`torch.testing.assert_close(c, golden_gemm(a,b,trans_a,trans_b), atol/rtol 见 §8.2)`。
- 完整分层套件（L1/L2/Boundary）交 Stage 3 的 `tilelang-op-develop`，此处不枚举。

### 8.4 CV 融合设计

**无需用户手写 CV 融合**：本算子为纯 Cube（仅 `T.gemm` + `T.copy`），无用户编写的独立 Vector 算子链，故无需 `Scope("Cube")/Scope("Vector")` 双 Scope 与手写 `sync_block_set/wait`。epilogue 的 fp32→out dtype 舍入由编译器在后端自动插入的 `VCastOp`（**Vector 单元**）于 L0C→GM 路径内完成，属**后端自动管理的隐式 Cube→Vector 协同**（`docs/Tilelang.language/内存操作/T.copy.md` §2.3 第 4 条：Developer 跨 dtype 场景插入 `VCastOp`，RoundMode=RINT）；该协同由编译器负责依赖与同步，用户不感知、不干预。若 Stage 4 引入用户自定义 Vector 侧后处理（如 bias/scale），再按 `tilelang-mixcv-skill` 评估显式 CV 融合（届时才需手写双 Scope）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **Developer + persistent + gemm 崩溃（已识别，规避）**：`TRAP-DEVMODE-PERSIST-GEMM`（`traps-runtime.md`，verified，toolchain 0.1.2+1990aa9fe4/CANN 8.5.0/Ascend910B2C）——Developer 模式 + persistent 分核 + `T.gemm` 混排运行时崩溃（"unaligned UUB addresses"），Expert 同结构正常。**本设计主选规避**：非 persistent 一维网格（§5.5）；极大规模 persistent 化列为 Expert 备选（Stage 4）。
2. **Developer 模式 GM→L1 的文档口径歧义**：`docs/Tilelang.language/内存操作/T.copy.md` §1 表格标注 Developer 模式 GM→L1 为「×（用 T.load_nd2nz）」，但 `examples/gemm/example_gemm.py`（官方 Developer Cube 示例，`docs/开发指南.md` §2 同款）用 `T.copy` 装载 `alloc_shared` 并被标记为通过。**本设计以官方示例为准**（`alloc_shared` 在 Cube kernel 内由框架映射 L1 并处理 ND→NZ），并在 Stage 3 首验证 smoke；若编译失败，改用备选 #2（Expert `load_nd2nz`）。属「未文档化假设」。
3. **非整除 M/N**：m=16/32/64/96、n=2112 非 block 整数倍；必须 kernel 内动态 extent 处理（§0.6 R4），**禁止 host padding**（`references/ascend-constraints.md` §4）；`T.ceildiv` 避免零 block。
4. **K 尾零填充**：仅当 `block_K ∤ K` 触发；本 manifest 的 `gcd(K)=128` 使 `block_K=128` 全程整除，不触发（R4 仍实现）。
5. **fp16 dst/累加会卡死**：`C_local` 必须 fp32（`docs/Tilelang.language/线性代数操作/T.gemm.md` §2.3）；输出 dtype 由 `T.copy` 单次 VCast 得到。
6. **trans 布局覆盖**：manifest 全部 workload `trans_a=false`（NN/NT），`trans_a=True`/TN/TT 路径无用例覆盖——实现时仍按 `a_transpose/b_transpose` 参数化，但测试需补 TN/TT（§8.3 可扩展）。
7. **物理核数**：实查 20（本机）；若换设备须重查（`NPUUtils.get().get_aicore_num()`），分核方案随核数变化。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| L0C 溢出 | `block_M×block_N×4 > 128KB` | segfault / 编译失败 | 保持 `block_M×block_N ≤ 32768`（128×128 或 128×256） |
| L1 溢出 | `2×(block_M·block_K+block_K·block_N)×2×num_stages > 512KB`（含 auto-multi-buffer ~1.12× 膨胀） | 编译失败 | 减小 `block_K` 或 `num_stages`（`CONST-capacity-910B2C`） |
| 尾块越界写 | 非整除 M/N 且未守卫写回 | 写坏相邻 GM（`TRAP-T-copy-region-semantics`） | 动态 extent 写回 `C_local[0:tm,0:tn]` |
| K 尾污染 | `block_K ∤ K` 且未清零 K 尾 | 全部输出元素被垃圾乘积污染 | `T.clear` A、B 的 K 尾 |
| fp16 累加 | `C_local` 用 fp16 | 运行时卡死 | 强制 fp32 累加 |
| 0×NaN | K 尾只清 A 未清 B 且 B 尾为 NaN | NaN 污染 | A、B K 尾都清零（E2） |
| persistent+gemm 崩溃 | Developer 模式核内 `T.serial` 多任务 + gemm | "unaligned UUB addresses" | 主选避免 persistent；persistent 用 Expert（§9.1.1） |

### 9.3 特殊场景处理

- **极小 M（m=16）**：M 维仅 1 块且 `tm=16`，Cube 仍按 `block_M=128` 计算（128 行中 112 行为零/垃圾但不写回），计算有浪费；可选启发式降 `block_M` 到 32/64（≥16），Stage 3/4 实测选择。
- **极小 N（n=1）**：仅当 `n==1 and not trans_a and not trans_b`（rhs_col）时 `GemmOp` 层才走 GEMV fast path；默认 NT（`trans_b=True`）下 `n==1` **不**满足该条件，仍进 `_gemm_kernel`（本 kernel 需借 N 尾块动态 extent 正确处理 n=1）。
- **极大规模（wide-n 6144 块）**：非 persistent 超发调度开销交 Stage 4 判定；若显著则切 Expert persistent（§5.5 备选）。
- **bf16 输入**：`T.gemm.md` §2.3 建议 fp32 累加（实测可用），`PL-1.8-bf16-cube-direct` 证实 bf16 Cube 直连无延迟税。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/gemm/_gemm_kernel/
├── DESIGN.md               # 本设计文档
├── RETROSPECTIVE.md        # Stage 1 复盘
├── verify_equiv.py         # D-1 等价性机器验证脚本（§1.6.1）
└── _gemm_kernel.py         # 算子实现（Stage 3 产出）
```

> 项目名 `gemm`（由 conductor 传入）；迁移 harness 模式逐函数落 `examples/{op_slug}/{func}/`。

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 设计文档 |
| `verify_equiv.py` | 已完成 | §1.6.1 E1/E2 等价性验证（4/4 `EQUIV_PASS`） |
| `RETROSPECTIVE.md` | 已完成 | Stage 1 复盘（自进化钩子） |
| `_gemm_kernel.py` | 待实现（Stage 3） | NPU Developer 模式 Cube GEMM kernel + golden |

### 10.3 命名规范

- 项目目录: `gemm`；算子/函数目录: `_gemm_kernel`；实现文件: `_gemm_kernel.py`。

### 10.4 实现顺序

1. ✅ 设计文档（`DESIGN.md`）+ 等价性验证（`verify_equiv.py`）
2. ⬜ Golden 函数（`golden_gemm`，以 §0.1 语义）
3. ⬜ 算子实现（`_gemm_kernel.py`）+ §8.3 L0 门槛精度比对
4. ⬜ 上层 wrapper（`GemmKernel`）替换 GPU 工厂调用为 NPU kernel
