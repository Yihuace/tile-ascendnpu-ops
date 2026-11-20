# `_logical_reduce_kernel`（承载 `CountNonzeroFwdOp`）算子设计文档

> **迁移任务（migration-harness）**：GPU TileOPs `CountNonzeroFwdOp` 的 TileLang kernel 函数
> `_logical_reduce_kernel` → TileLang-NPUIR（`target="npuir"`，Developer 模式）。
> **源算子**：`/workspace/TileOPs-gpu/tileops/kernels/reduction/` 下的 `logical_reduce.py`
> （基础工厂 `_logical_reduce_kernel` + `LogicalReduceKernel`；any / all / count_nonzero 三 kind 共用
> 同一工厂，由 `op_kind` 选分支。**本迁移单元只迁 `count_nonzero` kind**）。
> **本设计基于**：源码全文解读（Phase M0）→ 算法族调研（Phase R，调研四问）→ 硬件耦合性判定与 NPU 重设计
> （Phase M1，四态处置）→ 算法级优化设计（Phase 2）→ **等价性机器验证**（D-1，`verify_equiv.py`，
> E1/E1c/E2/E3/E3b/E4 全部 `EQUIV_PASS`（对照组 `CONTROL_OK`、边界组 `BOUNDARY_CONFIRMED`））
> → **设计期容量探针**（D-3，2 个探针，见下）。
> **工具链戳**：tilelang `0.1.2+b39cc30679662a13dd911768a51c7eede45d2892`（dev root build）
> - CANN 9.0.0（`/usr/local/Ascend/cann-9.0.0`）+ torch 2.9.0+cpu + **Ascend910B3（AIV=40）**。

**设备实查（本任务实测）**：

| 实查项 | 查询代码 | 返回值 |
|--------|----------|--------|
| AI Cube 核数 | `tilelang.utils.npu_utils.NPUUtils.get().get_aicore_num()` | **20** |
| AI Vector 核数 | `NPUUtils.get().get_aivector_core_num()` | **40** |
| 片上 UB 容量 | `tilelang.utils.npu_arch.AscendArch().ub_cap` | **196608 B（192 KB）** |
| 芯片名 | `AscendArch().chip_name` | `Ascend910B` |

**设计期容量探针（D-3；2 个，均落 `history_version/`，只标定未知常数、不实现算子）**：

| 探针 | 标定的未知常数 | 结论（实测） |
|------|----------------|--------------|
| `history_version/design_probe_ub.py` | 「指示链」（`vcmp(eq)` → `vcast(i1→f32)` → `reduce_sum(dim=1)`）在本工具链的 **UB 单价** | `ub overflow, requires 2097152 bits`（32768 元素）= **精确 8.00 B/元素**（编译器侧记账，与 dtype 无关）⇒ 单 block 硬上限 = 196608 / 8 = **24576 元素**；实测通过点：fp16 16384 元素、fp32/bf16 8192 元素。**同文件第二节**另证：raw 窄行宽度（N=64 / N=4 / N=2，即 128B / 8B / 4B 行宽）**均可编译通过** ⇒ 设计中**不需要任何宽度对齐 pad**（`CONST-reduce-ub-per-element` 的整平面 ~128 B/元素模型**不适用**于本 2-D 行归约形态） |
| `history_version/design_probe_tileloop.py` | 列分块形态（`T.serial(num_tiles)` 循环 + 运行累加链）在 **`bm>1` 多 tile** 下能否编译 | fp16 `2×4096`（4 tile）/ `8×512`（32 tile）/ fp32 `2×1024`（16 tile）**全部 OK** ⇒ **`CG-2026-0014`（`bm>1` + 多 tile 编译器 SIGSEGV，910B2C / CANN 8.5.0）在本工具链不复现**（版本戳更新候选，见 §9.4） |

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

**数学语义**：对已规整为 2D 的输入 `x ∈ 𝔽^{M×N}` 沿最后一维（规约维 N）统计**非零元素个数**：

$$y[i] = \#\{\,j \in [0, N) \;:\; x[i,j] \neq 0 \,\}, \qquad y \in \mathbb{Z}^{M}$$

**I/O 契约（迁移必需字段）**：输入 `(M, N)`（Op 层已把多轴输入 movedim+reshape 成 2D：M = 未规约维乘积、
N = 规约维乘积）；输出 **`(M,)`、dtype 恒 `int64`**（源 `source_output_shape: "(M,)"` 已确认；无转置布局）。
`dim=None` 全张量规约被 Op 层 flatten 成 `(M=1, N=numel)`（`_empty_dim_policy = "full"`）。

**与兄弟 kind 的关键区分（本迁移单元的判据）**：any / all / count_nonzero 共用同一工厂，但
**count_nonzero 的输出 dtype 是 `int64`**（`out = T.Tensor((M,), "int8" if op_kind != "count_nonzero" else
"int64")` 三元式 + `T.cast(result[i], "int64")`），any/all 为 `int8`（bool 载体）。**不得跟随兄弟 kind 的
bool 输出契约**。

**规约语义（累加顺序）**：计数是**整数加法**，在 fp32 精确整数域（≤ 2²⁴）内**与顺序无关**
（§1.6.1-E2 机器验证：逐元素序与树形序均 0 偏差）——与 logsumexp/var 类浮点归约不同，
**并行分组变化不影响该算子的数值结果**（迁移期的关键放宽）。

**dtype 语义**：kernel 侧输入 dtype ∈ {float16, bfloat16, float32}（bool/int32/int64/complex64/complex128
由 Op 层 `to_logical_float32` 预转 float32，见 `examples/TileOPs/tileops/kernels/reduction/count_nonzero/count_nonzero.py`
函数 `to_logical_float32`；complex 语义为 `(real != 0) | (imag != 0)`）。源在 kernel 内把数据 cast 到 **fp32**
后做 bool 化与归约；**输出恒 int64**。

**边界语义**（源码 + torch 语义 + 本工具链行为共同确定；§8.2 逐项记录）：

| 边界 | torch（CPU） | 源 GPU 实现 | 本设计 NPU 算法 | 处置 |
|------|--------------|-------------|-----------------|------|
| `±0.0` | 视为「零」（不计入） | `x != 0.0` 为 IEEE 相等判定 → ±0 均判「零」 | `x == 0.0`（**IEEE 正确**，`TRAP-vcmp-ne-ordered-nan` 明证 `eq` 为 IEEE 语义） | 等价域内 |
| **NaN** | **计入非零** | `!=` 为 **unordered NE** ⇒ NaN 计入 | 补式用 `== 0`（**ordered EQ**：NaN == 0 为假 ⇒ NaN **不计入零** ⇒ 计入非零） | **三方一致**（§1.6.1 INFO 行实测：源=2 / 候选=2 / torch=2；而 ordered-NE 路线=1，**即为被否决的形态**） |
| `±inf` / 亚正规 | 计入非零 | 计入 | 计入（比较运算不做 flush-to-zero；§9.2-R3 登记亚正规实测） | 等价域内 |
| 空规约维 `N=0` | 报错 | 工厂构造即 raise | 工厂期 raise（`N >= 1` 断言） | 保留 |
| 非整除 `N`（如 300） | — | **Op 层无 host pad**：源工厂内部 `align_up(N,256)` 派生 `N_padded`，由 `_needs_pad` 分支的**掩码加载**补齐（pad 值 0.0） | **在实列域（恰 N 列）计数，完全不 pad**（§1.6.1-E3 机器验证：pad 域计数 ≡ 实列域计数） | §0.5/§0.6-R3 |
| 窄行 `N < 32 B`（如 4） | — | 源 256 对齐 → `N_padded=256` | **raw 宽度直算**（探针实测 N=2/4/64 均编译通过） | 探针证据 |
| `M=0`（非规约维含 0） | 输出空 | 源 grid=0（未覆盖） | 列为风险（§9.2-R6） | 风险登记 |

**host 侧语义**：Op 层的 `movedim` / `reshape` / `contiguous` 均为**实现手段**而非算子语义；
`count_nonzero.py` 明示 `_kernel_handles_padding = True`（**kernel 接 raw `(M, N)`，Op 层不做 host pad**）、
`_pad_value() = 0.0`、`_kernel_handles_padding` 的语义见 `examples/TileOPs/tileops/ops/reduction/reduce.py`。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据（`torch.count_nonzero(x, dim) → int64`），
**不复刻 §0.6 的 NPU 算法**（无补式、无指示平面、无列分块），保证验证独立性。

### 0.2 源算子输入输出（含 NPU 侧契约核对）

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | **`(M, N)`（raw N）** | float16 / bfloat16 / float32 | M = 非规约维乘积，N = 规约维乘积；工厂参数携带**原始 N**（源工厂内部派生 `N_padded = align_up(N, 256)`） |
| `out` | `(M,)` | **int64** | 每行非零计数 |

**NPU 侧输入契约核对（实读目标工件）**：

| 核对项 | 结论 | 依据 |
|--------|------|------|
| 目标 Op 层 `_kernel_handles_padding` | **`True`**（`CountNonzeroFwdOp` 显式覆写） | `examples/TileOPs/tileops/ops/reduction/count_nonzero.py`（类属性块） |
| 因而 Op 传入宽度 | **raw `N`**（无 host `F.pad`） | `examples/TileOPs/tileops/ops/reduction/reduce.py` 的 `_prepare_input`（`_kernel_handles_padding` 为真时跳过 pad） |
| kernel 必须声明的宽度 | **`(M, N)`**（与 Op 传入逐元素一致） | 与源工厂「内部派生 N_padded」的差异在 §0.6-R3 处置 |
| 工厂接口（harness 固定） | `_logical_reduce_kernel(M, N, op_kind, dtype)` → `_func(block_m)` → `kernel(x)`（NPU 侧 `threads` 已去除，K9） | `examples/TileOPs/tileops/kernels/reduction/count_nonzero/count_nonzero.py` 的 wrapper |
| 本迁移单元 kind 范围 | **仅 `count_nonzero`**；any/all 为独立迁移单元（源文件由 `AllFwdOp`/`AnyFwdOp` 承载） | 用户迁移约束 + Op 层 docstring |

### 0.3 实现算法解读（怎么算）

按结构化阅读协议逐层读 `_logical_reduce_kernel`（基础工厂；`_logical_reduce_kernel_tiled` 为**非迁移对象**，
其解读仅用于耦合性判定，见 §0.4-#8）。

**计算步骤分解**（覆盖基础工厂全部计算语句）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `N_padded = align_up(N, DEFAULT_ALIGNMENT=256)`；`_needs_pad = N_padded != N`；`_pad_val = _pad_value_for_op(op_kind)`（count_nonzero → `0.0`）；`T.Kernel(T.ceildiv(M, block_m), threads)` | 工厂参数 | 编译期常量 + 一维 grid | 任务划分 |
| 2a（`_needs_pad` 为真） | `for i, j in T.Parallel(block_m, N_padded): x_f32[i,j] = T.if_then_else((pid_m*bm+i < M) and (j < N), T.cast(x[...], "float32"), T.cast(_pad_val, "float32"))` —— **掩码加载 + fp32 上抛 + 越界行保护三合一** | GM `x` | `x_f32 (bm, N_padded)` fp32 | 数据载入（非整除处理） |
| 2b（`_needs_pad` 为假） | `T.copy(x[pid_m*block_m, 0], shared_buf)` → `for i, j in T.Parallel(block_m, N_padded): x_f32[i,j] = T.cast(shared_buf[i,j], "float32")` | GM `x` → SMEM → fragment | `x_f32 (bm, N_padded)` fp32 | 数据载入（SMEM tiling + fp32 上抛） |
| 3 | `for i, j in T.Parallel(block_m, N_padded): bool_vals[i,j] = T.if_then_else(x_f32[i,j] != 0.0, 1.0, 0.0)` | `x_f32` | `bool_vals (bm, N_padded)` fp32（0.0/1.0） | **[x ≠ 0] 指示化** |
| 4（count_nonzero 分支） | `T.reduce_sum(bool_vals, result, dim=1)` | `bool_vals` | `result (bm,)` fp32 | 行计数（Σ） |
| 5（count_nonzero 分支） | `for i in T.Parallel(block_m): out_local[i] = T.cast(result[i], "int64")` | `result` | `out_local (bm,)` int64 | 计数→int64 契约 |
| 6 | `T.copy(out_local, out[pid_m * block_m])` | `out_local` | GM `out` 行块 | 输出搬运 |

**数据流（源硬件视角）**：`GM[x] →T.copy→ SMEM[shared_buf] →cast→ register/fragment[x_f32] →ite→
fragment[bool_vals] →reduce_sum→ fragment[result] →cast→ fragment[out_local] →T.copy→ GM[out]`；
`_needs_pad` 时路径 2a 合并为 GM 直接掩码载入（SMEM 被绕过）。

**循环与并行结构**：`T.Kernel(ceildiv(M, block_m), threads=threads)` —— 一维 grid × CUDA 线程两级并行；
`block_m ∈ {1,2,4,8}` 由 48 KB SMEM 预算启发式选（`default_config`）；**无流水**（无 `T.Pipelined`）。

**host 侧逻辑**：Op 层 validate → dim 归一 → `movedim` → `contiguous` → `reshape(M,N)` →
（**不 pad**，因 `_kernel_handles_padding=True`）→ 调 kernel → 输出 reshape 回原形（keepdim=false）；
`functools.lru_cache(maxsize=32)` 按 `(M, N, op_kind, dtype)` 缓存编译产物。

**编译配置**：`@tilelang.jit(out_idx=[1])`（GPU target）；无特殊 `pass_configs`；
`supported_archs = [80, 86, 89, 90]`（CUDA SM）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | SMEM tiling（`alloc_shared` + 整块 `T.copy`，非 pad 路径） | 合并访存、一次载入 | 行块驻留 SMEM 后转 fragment | shared memory | 硬件强相关（NPU 有 UB 等价） |
| 2 | **256 元素对齐 pad + 单位元掩码加载**（`align_up(N,256)` + `T.if_then_else` + `_pad_val=0.0`） | 规避 GPU `T.copy`/fragment 的 256 对齐与列上限约束 | 计算域总是 `N_padded` 的整数倍宽；pad 列用计数单位元 0.0 填充 | GPU 256 元素对齐/copy 布局推断 | 对齐约束硬件相关；**单位元技巧可移植** |
| 3 | bool 物化（1.0/0.0）+ 复用 `reduce_sum` 计数 | 用现成归约原语实现计数，避免手写计数循环 | 指示化 → 复用 sum 归约 | — | **算法层，可移植**（NPU 保留） |
| 4 | 全量 fp32 上抛（步骤 2a/2b 的 cast） | 数值稳健（对 sum/var 类归约必要） | 数据先 cast fp32 再比较与归约 | — | 对 count_nonzero **非必要**（比较是精确谓词，§1.6.1-E4） |
| 5 | `block_m` 48 KB SMEM 预算启发式 + `[1,2,4,8]` 阶梯（`default_config`） | 行块尽量大以摊薄 block 开销 | 预算 ÷ (N_padded × elem) | SMEM 容量 | 预算值硬件相关 → 需重设计 |
| 6 | `threads`（128/256）作为归约并行宽度 | CUDA 线程级并行归约 | 线程内/跨线程归约 | CUDA warp/thread | 硬件强相关（NPU 无 threads 概念） |
| 7 | `functools.lru_cache(maxsize=32)` 工厂缓存 | 消重复编译 | Python 层 `(M,N,kind,dtype)` 键 | — | 可移植 |
| 8 | **N-tiled 回退工厂** `_logical_reduce_kernel_tiled`（`tile_n` + `T.Serial(num_tiles)` + 运行累加 `acc`） | 规避 **TileLang 单 fragment 列上限 32768** 与 SMEM 预算 | 沿 N 分块，逐块 `reduce_sum` 后累加 | GPU fragment 列上限 / SMEM 预算 | 触发条件硬件相关；**本迁移单元不迁**（用户约束） |
| 9 | 结果块写 `T.copy(out_local, out[pid_m*block_m])` | 向量化写回 | fragment → GM | — | 模式可移植 |
| 10 | autotune（`autotune_configs` / `tune_by_forward`） | 配置搜索 | 逐配置实测选优 | — | harness 已移除（K8） |
| 11 | 重 pad 布局约束（`block_m × N_padded ≤ 2 × threads`，仅 N<256 生效） | 规避 GPU copy 布局推断限制 | 限制 block_m | GPU copy-layout 推断 | 硬件强相关（NPU 不适用） |

> **识别完整性**：上表覆盖源基础工厂的全部优化性结构（步骤分解表已把每条计算语句归入步骤 1–6）；
> 源为朴素「SMEM + 指示化 + 单次归约」形态，**无** warp shuffle / mma / swizzle / 异步流水等 GPU 深度优化。
> **源缺陷（本设计修正，Stage 3 不得照抄）**：步骤 6 的 `T.copy` 与步骤 2b 的 `T.copy` **均无 M 尾块守卫**
> （`M % block_m ≠ 0` 时越界读/写；harness 用例 `(129,512)` 配 `block_m=8` → `129 % 8 = 1` 即触发位形）。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（输入含 Phase R 调研结论 §1.6.0）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计数语义（`(M,N)`→`(M,)` int64，±0/NaN/±inf 边界） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden = `torch.count_nonzero`；NaN 由补式天然对齐（§0.1） |
| 指示化 + 单次行归约的两步结构（步骤 3+4） | 算法 | 无（纯算法层） | 有：Vector `vcmp`/`vcast` + `T.reduce_sum(dim=1)` | **保留** | 结构保留；比较域与指示域按 §1.6.1-E1/E4 优化（`docs/Tilelang.language/比较操作/T.vcmp.md`、`docs/Tilelang.language/规约操作/T.reduce_sum.md`） |
| 步骤 3 的 `x != 0.0` 判决式（unordered NE） | 算法 | 无 | **有但语义相反**：`T.vcmp("ne")` 是 **ordered** NE（NaN 恒假） | **等价替换** | 改为**有序 EQ 的补式** `N − Σ[x == 0]`：`eq` 为 IEEE 正确（`traps-compiler.md` 的 `TRAP-vcmp-ne-ordered-nan` 明证），且 NaN 计入非零与 torch/源一致（§1.6.1-E1 机器验证 + 对照组） |
| 步骤 2a/2b 的全量 fp32 上抛（#4） | 算法/精度 | — | — | **舍弃** | 比较是精确谓词：`[x_fp16 == 0] ≡ [fp32(x) == 0]`（fp16/bf16→fp32 精确单射，保 ±0/NaN）；§1.6.1-E4 **全 2¹⁶ 位型穷举验证** `EQUIV_PASS`。省 1 个全宽 pass + 半份 staging 字节 |
| SMEM tiling / 整块 `T.copy`（#1） | 优化 | shared memory | 有：UB（`T.alloc_shared`，实查 192 KB） | **等价替换** | GM→UB staging（`migration-analysis.md` §5.3 行「shared memory → L1/UB」）；对齐需求由 `T.copy` 切片天然满足（探针证 raw 宽度可编译） |
| 256 对齐 pad + 单位元掩码加载（#2） | 优化 | GPU 256 元素对齐 + fragment 列上限 | **不需要**：NPU 在实列域计算即精确 | **舍弃** | §0.6-R3：raw `(M, N)` 计数**不需要任何 pad**——计数语义对 pad 天然中性（pad=0.0 不计入）；§1.6.1-E3 机器验证「pad 域计数 ≡ 实列域计数」；探针实测 N=2/4/64 raw 宽度可编译 |
| `block_m` 48 KB SMEM 启发式（#5/#6） | 优化 | SMEM 容量 / CUDA 线程 | 换 **元素预算** 模型（探针实测 8 B/元素） | **重新设计** | → §0.6-R5：`E_cap = 8192 元素/block`（竞态安全）+ 阶梯（基础 `{1,2,4,8}` / 扩展 `{16..1024}`） |
| `threads` 参数（#6） | 优化 | CUDA 线程 | 无（本项目一维 `T.Kernel`，NPU 无 threads） | **舍弃** | harness K9 已去除；元素级并行交给 Vector 向量化（`docs/Tilelang.language/编译器提示操作/T.Kernel.md`） |
| N-tiled 回退工厂 `_logical_reduce_kernel_tiled`（#8） | 优化 | **TileLang 单 fragment 列上限 32768** + SMEM 预算 | 该触发条件在 NPU **不存在** | **舍弃（作为迁移对象）** | 其动机（fragment 列上限）不适用 ⇒ 不迁移该工厂、不照抄其 `tile_n` 语义。**但** NPU 侧因 **UB 容量**（而非列上限）另设列分块路径（§0.6-R4），二者**触发条件与结构均不同**，不得混同 |
| 结果块写 `T.copy`（#9） | 优化 | — | — | **等价替换** | 等价的 UB staging + 显式 slice 写（尾块 `real_m`）——模式可移植，实现形态按 NPU slice 写法适配 |
| autotune（#10） | 优化 | — | — | **舍弃** | harness K8 已移除（`init_config` 无 `tune`；启发式配置唯一入口） |
| 重 pad 布局约束（#11） | 优化 | GPU copy 布局推断 | 无此约束 | **舍弃** | GPU 特有规避；NPU 不需要 |
| M 尾块守卫缺失（源步骤 2b/6 无 `M % block_m` 检查） | 正确性（**源缺陷**） | GM 定长 block copy 无边界检查 | 有：`T.min` 标量 + src/dst 双显式 slice | **重新设计** | §0.6-R6 三件套（VP-2026-0044 待合入提案 / logsumexp 先例）；**Stage 3 不得照抄无守卫 copy 形态** |
| `lru_cache` 工厂缓存（#7） | 优化 | — | — | **保留** | 工厂层与硬件无关 |

> **无「舍弃后性能意图丢失」项**：#4 的精度意图由 §1.6.1-E4 的穷举等价性论证承接（比较谓词不需要 fp32 域）；
> #2 的对齐意图由「实列域计数 + raw 宽度可编译」承接（探针证据）；#8 的列上限规避意图随该约束消失而不再需要，
> 其性能意图（大 N 可处理）由 §0.6-R4 的 UB 容量驱动列分块承接（Pareto 上更强：触发条件是真实的容量约束）；
> #11 的布局规避意图随 GPU copy 约束消失。#6/#10 由 harness 适配（K8/K9）承接。
> 每项判定依据：`migration-analysis.md` §5.3 映射表 + 本仓 `docs/Tilelang.language/` API 文档 +
> pattern-library 实测条目 + **本任务两个设计期探针的实测结论**（逐条注明）。

### 0.6 NPU 算法重设计

**重设计项 R1：计数公式（源：fp32 上抛 → 布尔物化 → fp32 求和）**

- **源方案**：`Σ_j [fp32(x_j) != 0]`，指示值 1.0/0.0 物化为 `(bm,N_padded)` fp32 平面后 `reduce_sum`，
  末位 `T.cast(result[i], "int64")`。
- **NPU 新算法（§1.6.0 选定 C1）**：**零计数补式**（有序 EQ）：

  $$y[i] \;=\; N \;-\; \sum_{j<N}\big[\,x[i,j] == 0.0\,\big]$$

  实现链：`T.vcmp(x_ub, zero_ub, mask_eq, "eq")` → `T.vcast(mask_eq, ind, round_mode="rint")`
  （i1→f32 指示平面，0.0/1.0）→ `T.reduce_sum(ind, zc, dim=1)` → 末位 `T.vsub(n_v, zc_flat, cnt)`
  → `T.vcast(cnt, out_ub, round_mode="rint")`（int64 出栈）。
- **语义保持论证**：
  1. **数学恒等**：`#{x ≠ 0} = N − #{x = 0}`（行内逐元素划分，N 为实列数）；
  2. **NaN 安全**：`vcmp("eq")` 为 IEEE ordered EQ（`TRAP-vcmp-ne-ordered-nan` 明证「`eq`/`le`/`ge` 为 IEEE 正确」）
     ⇒ NaN 与 0 判「不等」⇒ NaN 不计入零 ⇒ **计入非零**，与 torch 及源 unordered-NE 一致；
     **反面形态被否决**：直接用 `vcmp("ne")`（ordered NE）会把 NaN 判为「零」→ 与源/torch 分歧
     （§1.6.1-E1c 对照组实测 42 行差异，`CONTROL_OK`）；`vcmp("eq")` 取反不可行——`T.vnot` 不支持
     bool/int1（`docs/Tilelang.language/逻辑操作/T.vnot.md` §2.2.1 dtype 表 bool 为 ×），故**必须**走补式；
  3. **比较域**：`[x == 0]` 在输入原生 dtype 与 fp32 域恒等（E4，全 2¹⁶ 位型穷举），±0/NaN/亚正规/±inf 全覆盖；
  4. **边界核对**：`N=1`（`(1,1)` 用例）、全零行（`y=0`）、全非零行（`y=N`）、混合符号零、NaN 行
     由 `verify_equiv.py` 角点集与 §8.2 L0 覆盖。
- **收益量化**：① 省 1 个全宽 pass（fp32 上抛）——源步骤 2 的 cast 完全消除；② 补式在 **(bm,) 微缓冲**上
  完成（1 个微元 op），**不必**在全宽平面上做 `1 − ind`（否则 +1 个 `(bm,W)` f32/全宽 pass）；③ 指示平面
  由 `vcmp+vcast` 显式向量链产生（源为 `ite` 融合形态，见 §1.6.3-#5）。

**重设计项 R2：并行结构（源：grid × CUDA threads 两级）**

- **源方案**：`T.Kernel(ceildiv(M, block_m), threads=threads)`；`threads` 同时是归约并行宽度与布局约束参数。
- **NPU 新算法**：纯 Vector 一维 persistent kernel——`T.Kernel(num_kernels, is_npu=True)`，
  `num_kernels = min(ceil(M/block_m), 40)`（**40 = 实查 `get_aivector_core_num()`**，纯 Vector 核数，§5.5）；
  核内 `T.serial(静态上界)` 任务循环（grid-stride 映射 `task = cid + s·num_kernels`）+ 动态 guard
  `if start < M`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 示例同款 guard 形态）；
  元素级并行全部交给 Vector 向量化（显式 v-prefix 链 + reduce 原语）。
- **语义保持论证**：行之间无依赖（逐行独立计数），任务循环只改变调度不改变计算；整数计数的
  累加顺序无关性（§0.1）使并行分组变化无精度影响。尾块 `real_m = T.min(bm, M − start)` + src/dst
  双显式 slice（VP-2026-0044 三件套 / logsumexp 先例），垃圾行不写出。

**重设计项 R3：宽度 / padding 策略（源：内部 `align_up(N,256)` + 掩码加载）**

- **源方案**：工厂内部派生 `N_padded = align_up(N, 256)`，`_needs_pad` 时用 `T.if_then_else` 掩码加载
  把 `[N, N_padded)` 填成单位元 0.0，随后**在 `N_padded` 全域**做指示化与归约。
- **NPU 新算法**：**完全不 pad**——kernel 声明 `(M, N)`（raw，与 Op 契约 `_kernel_handles_padding=True`
  逐元素一致），在整个计算路径上**恰在实列域（N 列）**做比较/指示/归约；列分块路径的尾块宽度
  `tail = N − num_full·tn` 亦为实列宽度（不做任何补齐）。
- **语义保持论证**：① 计数语义对「单位元 padding」天然中性：pad 列值为 0.0 ⇒ 不计入非零
  ⇒ `Σ_{j<N_padded}[x_pad_j ≠ 0] = Σ_{j<N}[x_j ≠ 0]`（§1.6.1-E3 机器验证 1377 行 0 偏差）；
  ② raw 宽度的可编译性由探针实测（N=64/4/2 均编译通过；logsumexp N=300 的已迁移先例同向）；
  ③ 不做 pad ⇒ 少 1 条掩码分支、少 1 个 `ite` 全宽 pass，且**无越界读**（源 2a 的 `pid_m*bm+i < M`
  条件同时承担了行保护，NPU 侧由尾块三件套承担）。
- **收益量化**：消除源的 `_needs_pad` 分支与其 `T.if_then_else` 全宽 pass；`N=300` 类负载的读流量
  从 `N_padded=512` 降到 `N=300`（**−41%**，D-2 实算见 §1.6.0）。

**重设计项 R4：容量驱动的路径切换（源：fragment 列上限触发的 tiled 回退）**

- **源方案**：`_needs_tiling = N_padded > MAX_SINGLE_TILE_COLS(32768) or N_padded·elem > smem_budget`，
  触发 `_logical_reduce_kernel_tiled`（`tile_n` + `T.Serial(num_tiles)` + 逐块 `reduce_sum` 累加 + int64 出栈）。
- **NPU 新算法**：路径切换的**判据换成元素预算**（探针实测 8 B/元素 ⇒ 单 block 硬上限 24576 元素；
  竞态安全预算 `E_cap = 8192` 元素/block，§4.5 记账）：
  - **路径 R（resident，`N ≤ E_cap`）**：`tn = N`（单 tile）；`bm` 由阶梯给出（`bm·N ≤ E_cap`）。
  - **路径 C（column-tiled，`N > E_cap`）**：`bm = 1`，`tn` = **≤ min(E_cap, N) 的最大 256 倍数且整除 N**
    （无整除因子时取最大 256 倍数），`num_full = N // tn`，`tail = N − num_full·tn`（静态尾块，
    独立静态宽度的 `x_tail` 缓冲组）；运行累加 `acc += tile_zc`（**no-alias**：`T.vadd(acc, zc, new_acc)`
    - `T.copy(new_acc, acc)`，规避 `TRAP-vselect-inplace-carried-state` 同族陷阱）。
- **语义保持论证**：分块归约 = 分段计数求和，由整数加法结合律与顺序无关性（§0.1）保证恒等；
  最终补式的被减数是**整行宽度 N**（= `num_full·tn + tail`），与路径 R 的 N 一致 ⇒ 两条路径输出恒等
  （§1.6.1-E3b 机器验证 pad 抵消与分段一致性）。
- **与源 tiled 工厂的区别（显式声明）**：源工厂由 **GPU fragment 列上限**触发、**不迁移**；
  本路径由 **NPU UB 容量**触发（真实硬件约束），结构上是一个**统一的列分块 + 运行累加**形态，
  与源工厂的「每行一次 `reduce_sum` 后累加」在**归约域**（局部 j vs 全局）与**等价性来源**上一致，
  但**判据、缓冲组、尾块策略均按 NPU 实测重新推导**，不照抄源的 `tile_n` 语义。
- **收益量化**：使 `N` 上限从 `E_cap`（8192）提升到 2²⁴（fp32 计数精确域，§1.6.1-E2），
  覆盖 manifest 的 16384 类负载；GM 流量**不增加**（每元素仍只读一次，见 §1.6.0-R3 对比表——
  这一点与同族 ada_layer_norm 的「N-tile 流式 +25% 流量」不同，因其流式为两遍式口径）。

**重设计项 R5：`block_m` / 配置选择（源：48 KB SMEM 启发式 + `[1,2,4,8]` 阶梯）**

- **源方案**：`smem_per_row = N_padded × elem_bytes`；`max_block_m = 48KB // smem_per_row`；
  阶梯 `[1,2,4,8]` 取不超预算的最大者 + `threads = 128`。
- **NPU 新算法**：**元素预算阶梯**（维度从「字节」改为「元素」，因为探针实测编译器 UB 单价 = 8 B/元素、
  与 dtype 无关）：
  `bm = max{p ∈ {1,2,4,8} : p·N ≤ E_cap}`；若 `ceil(M/bm) > 80`（>2 波 40 核）则启用扩展阶梯
  `p ∈ {16,32,64,128,256,512,1024}` 取同预算过滤器的最大者；`bm = min(bm, M)`。
- **语义保持论证**：纯配置选择，不影响语义；预算边界由探针的实测通过/失败点双向定标
  （通过：fp16 16384 元素 / fp32 8192 元素；失败：fp16 32768 元素 `requires 2097152 bits`）。

**重设计项 R6：M 尾块守卫（修正源缺陷）**

- **源方案**：步骤 2b/6 的 `T.copy` 无 `M % block_m` 检查（越界读/写）。
- **NPU 新算法**：`real_m = T.min(block_m, M − start)` + src/dst 双显式 slice + 垃圾行不写出。
- **语义保持论证**：仅把越界访问改为有界访问，有效行结果不变；`M=0` 另登风险（§9.2-R6）。

### 0.7 标杆实现

- **源算子**：`/workspace/TileOPs-gpu/tileops/kernels/reduction/` 下的 `logical_reduce.py`
  （本设计 §0.1–§0.4 的解读对象；基础工厂 `_logical_reduce_kernel` 为迁移对象，
  `_logical_reduce_kernel_tiled` 仅作耦合性判定输入）；Op 层策略
  `/workspace/TileOPs-gpu/tileops/ops/reduction/` 下的 `logical_reduce.py` 与 `reduce.py`；
  对齐常数 `/workspace/TileOPs-gpu/tileops/kernels/reduction/_primitives.py`。
- **NPU 侧 harness 工件（本迁移单元的输入与落点）**：
  - 提取版 GPU 参考 kernel：`examples/TileOPs/tileops/kernels/reduction/count_nonzero/_count_nonzero_fwd_kernels.py`；
  - 已移植 wrapper（K5–K9 已适配、`threads` 已去除）：
    `examples/TileOPs/tileops/kernels/reduction/count_nonzero/count_nonzero.py`；
  - Op 层：`examples/TileOPs/tileops/ops/reduction/count_nonzero.py`（`_kernel_handles_padding = True`）；
  - 规格：`examples/TileOPs/tileops/manifest/reduction.yaml` 的 `CountNonzeroFwdOp` 段
    （workloads / roofline / source）；**注意 `sparsity-seq` 宽度已由 32768 下调为 16384**；
  - 精度/性能用例：`examples/TileOPs/tests/ops/test_count_nonzero.py`、
    `examples/TileOPs/benchmarks/ops/bench_count_nonzero.py`。
- **golden 基准**：`torch.count_nonzero(x, dim=...) → int64`（§8.1）；harness 以 `torch.equal` **精确比对**
  （`_exact_compare_int64`，同时断言 dtype == int64）。
- **同族已验证参考（row-reduction 家族）**：
  `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/`（单 tile 驻留形态、
  尾块 `T.min` + 双 slice、raw N=300/32 实测通过）与
  `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_tiled/`（列循环 + 运行状态更新链）；
  `examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/_argreduce_kernel_DESIGN.md`（同族
  迁移设计结构 + 探针证据链，工具链为 910B2C/CANN 8.5.0，引用按版本戳标注）；
  `examples/argmin/_argreduce_kernel/DESIGN.md` + `examples/argmin/_argreduce_kernel/verify_equiv.py`
  （910B3/CANN 9.0.0 同族**同工具链**先例：分核三要素、UB 预算记账、D-1 脚本形态）。

---

## 1. 概述

### 1.1 算子名称

`_logical_reduce_kernel`（承载 `CountNonzeroFwdOp` 的 NPU kernel；工厂接口保留 `op_kind` 形参以维持
共享工厂契约，本单元实现 `count_nonzero` 分支）

### 1.2 功能描述

对 `(M, N)` 输入沿最后一维（规约维 N）统计**非零元素个数**，输出 `(M,)` **int64**。
输入 dtype ∈ {float16, bfloat16, float32}（其余 dtype 由 Op 层预转 float32）。
输入为 raw `(M, N)`（Op 层不做 host pad）。

### 1.3 数学公式

$$y[i] = \#\{\,j \in [0, N) : x[i,j] \neq 0\,\}$$

NPU 侧等价计算式（§1.6.1 优化后，供 §3.1 拆解；`N = num_full·tn + tail`）：

$$y[i] \;=\; N \;-\; \sum_{j<N}\big[\,x[i,j] == 0.0\,\big]$$

分段累加形态（路径 C，`N = num_full · tn + tail`；被减数恒为整行宽度 N）：

$$y[i] \;=\; N \;-\; \Big(\sum_{t<num\_full} \mathrm{reduce}_{j<tn}\big[\,x_t[i,j] == 0\,\big] \;+\; \mathrm{reduce}_{j<tail}\big[\,x_{tail}[i,j] == 0\,\big]\Big)$$

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

按 N 与元素预算的关系双路径（§0.5/§0.6 决策 + §1.6.0 调研选定）：

**路径 R（resident，`N ≤ E_cap`；覆盖 smoke `(32,256)`、hidden `(2048,4096)`、全部测试窄/常规行宽）**：

1. `T.fill(zero_ub, 0)`（**任务不变操作数**，循环外一次）；
2. 每个行块任务：`T.copy(x[start:start+real_m, 0:N], x_ub[0:real_m, 0:N])`（src/dst 双显式 slice，§0.6-R6）；
3. `T.vcmp(x_ub, zero_ub, mask_eq, "eq")` —— **IEEE ordered EQ**（NaN 不计入零，§0.6-R1）；
4. `T.vcast(mask_eq, ind, round_mode="rint")` —— i1→f32 指示平面（0.0/1.0）；
5. `T.reduce_sum(ind, zc, dim=1)` —— 行零计数 `(bm,1)` f32（整数精确域，§1.6.1-E2）；
6. `T.reshape(zc, zc_flat (bm,))` → `T.vsub(n_v, zc_flat, cnt)`（补式，`n_v ≡ N` 任务不变量）
   → `T.vcast(cnt, out_ub, round_mode="rint")` → `T.copy(out_ub[0:real_m], out[start:start+real_m])`。

**路径 C（column-tiled，`N > E_cap`；覆盖 `sparsity-seq (32,16384)` 与 `3d-multidim (128,16384)`）**：

1. `T.fill(zero_ub, 0)`、`T.fill(acc, 0)`、`T.fill(n_v, N)`（均任务不变/循环外）；
2. `for t in T.serial(num_full)`（**静态上界**）：载入第 t 个宽 `tn` 列块 → 步骤 3–5 同路径 R（`dim=1`
   归约到 `tile_zc (bm,1)`）→ **no-alias 累加** `T.vadd(acc, tile_zc, new_acc)` + `T.copy(new_acc, acc)`；
3. 尾块（`tail > 0` 时，**工厂期 Python 条件生成**的独立静态宽度代码块）：同步骤 2，用独立的
   `x_tail/zc_tail` 缓冲组（宽 = `tail`，静态常量）；
4. 出栈：同路径 R 步骤 6（被减数恒为 `N`，与分段数无关）。

> 两条路径的**逐元素计算完全相同**（`vcmp eq` → `vcast` → `reduce_sum` → 补式），差异仅在
> 「一行为一个 tile」还是「一行为 `num_full` 个 tile + 尾块」——故实现层可共享同一段原语发射代码，
> 尾块与 tile 循环由工厂期静态参数控制。

### 1.5 数据流图

```
路径 R（纯 Vector 单引擎，无 L1/L0 参与）：
GM[x] --T.copy(slice,real_m)--> UB[x_ub (bm,N) dtype] --T.vcmp("eq") vs UB[zero_ub]--> UB[mask_eq (bm,N) bool]
   --T.vcast(i1->f32, rint)--> UB[ind (bm,N) f32] --T.reduce_sum(dim=1)--> FRAG[zc (bm,1) f32]
   --T.reshape--> FRAG[zc_flat (bm,)] --T.vsub(n_v, ·)--> FRAG[cnt (bm,) f32]
   --T.vcast(rint)--> UB[out_ub (bm,) int64] --T.copy(slice)--> GM[out (M,) int64]
   无 GM 中间往返；无 workspace；无跨引擎传输。

路径 C：GM[x 列块] --T.copy--> UB[x_ub (bm,tn)] --(同上链)--> FRAG[tile_zc] --T.vadd/T.copy--> FRAG[acc]
   （逐 tile 循环；acc 全程片上驻留）→ 尾块同链 → 出栈（同上）。
```

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

> **调研对象**：「统计非零元素个数」这一数学语义的**算法族**（GPU 源算法只是基线候选之一）。
> 信息源：`references/algorithm-candidates.md` 的 `ALG-reduction` 命中行（R1「max+sum 合并扫描 / 和与平方和
> 一次扫描」、R2「分块归约 + 合并」、R3「扫描遍数 / 跨核归并代价」、R4「水平 vs 垂直 lane 映射」+
> row-reduction 迁移基准形态）+ `ALG-elementwise`（R1「公共子表达式消除 / cast 链合并」）+
> `kb_search.py` 预注入条目（`TRAP-inner-axis-reduce-scalarize` / `VP-2026-0044`（**pending 提案**）/
> `CASE-reduction-logsumexp` / `CASE-pool-maxpool1dindices-migration` / `TRAP-npuir-ref-backend-golden-capability`）
> - 本仓 `examples/`（logsumexp single/tiled、argmin/argmax 同族、`_primitives.py`）
> - pattern-library（constants / traps-compiler / traps-runtime / cases / layout）
> - **用户输入公式与源算法（基线）** + 恒等变形空间自查（1 − [x=0]、自取反判决式、位域判决式、
> 整数域累加、排序/选择类结构）。
> count_nonzero 属**规约类**（多步：指示化 + 归约）→ **完整调研**（R1 候选表 + R3 四口径对比 + R4 逐候选评估）。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| 0 | **基线：GPU 源（fp32 上抛 + 布尔物化 + `reduce_sum`）** | `Σ_j [fp32(x_j) != 0]`，指示平面 1.0/0.0，末位 cast int64 | — | — | ✅（基线） |
| 1 | **C1 零计数补式 + 原生域比较**（§1.6.1-E1/E4） | `N − Σ_j [x_j == 0]`（原生 dtype 域，ordered EQ） | 数学恒等（行内划分）；`eq` 为 IEEE 正确 ⇒ NaN/±0 与源/torch 一致 | 省 1 个全宽 pass（fp32 上抛）；补式落在 `(bm,)` 微缓冲；staging 字节减半 | ✅ |
| 2 | C2 自取反判决式 `[x ≠ 0] = [x ≠ −x]` | 取反 pass + `vcmp("ne")`（对 ±0 判等、对 NaN 判不等） | 数学上成立（对 ±0/±inf/NaN/亚正规逐一成立） | **净亏**：需 `−x` 平面（且本工具链**无 `T.vneg`**，数学操作目录枚举确认无取反算子）⇒ 等价于「把零常数量换成 −1 常数量」+ 1 个全宽 pass | ❌ |
| 3 | C3 位域判决式 `[x ≠ 0] = [bits(x) & 0x7FFF… ≠ 0]` | `vbitcast` → `vand` 掩码 → 比较 | 对 ±0/NaN/±inf 成立（IEEE 位型） | **净亏**：+2 个全宽 pass（bitcast + and）+ 零常数量不变 | ❌ |
| 4 | C4 整数域累加（int32/int64 `reduce_sum`） | 指示化到 int32 后 `reduce_sum` | 数学恒等 | 精确域从 2²⁴ 提升到 2³¹——但**bool→int32 的 `vcast` 不存在**（`T.vcast.md` §2.2.1 无 `i1→i32`）⇒ 需 `vselect(one_i32, zero_i32)` 双常数量（+2 个 `(bm,W)` 平面）；本算子 manifest `N ≤ 16384 ≪ 2²⁴`，收益为零 | ❌（留作 N>2²⁴ 备选，§1.6.1-E2） |
| 5 | C5 求和侧合并（`Σ[x≠0]` 与其它统计量一次扫描） | 无其它统计量（count_nonzero 是单目的） | — | 无可用公共子表达式 | ❌（结构原因） |
| 6 | C6 N-split 跨核 partial+merge（两段 kernel） | 列分片并行 `(M,nchunk)` 零计数 workspace + merge kernel | 数学恒等（分段求和） | 小 M 大 N 负载核利用率提升 | ❌（表下论证：本算子 M ≥ 32 ⇒ 核利用率已 ≥ 80%，workspace 往返 + 第二次 kernel 启动为净亏；对比 §1.6.4 备选不涉及跨核） |
| 7 | C7 排序/选择类结构（排序后数零） | `argsort` 后定位零段长度 | 数学恒等 | O(N log N) ≫ O(N)；且零可能在两端 | ❌（复杂度劣） |

**R2 有没有在线算法？结论：有，且本算子天然在线。**
结构判据（`algorithm-research.md` §3-R2）：计数可分解为可增量合并的 **running 统计量**（计数和），
`running_count += 本块计数` 即可流式合并；逐元素判决 `[x_j == 0]` **无跨元素依赖** ⇒ 单遍即天然在线。
收益口径：**扫描遍数 1**（源实现同为 1 遍，故在线化不减少遍数——这是 count_nonzero 与 logsumexp（两遍
max→exp-sum）/layer_norm（两遍 mean→var）的关键差别）；中间缓冲 `O(N) → O(tile)`（决定 UB 驻留可行性）
——**在线形态的价值在本算子上是「容量」而非「遍数」**：它是路径 C 存在的前提（`N > E_cap` 时无在线形态
就必须落 GM 或分两次读）。迁移任务侧，源 `_logical_reduce_kernel_tiled` 的 `acc[i] = acc[i] + tile_acc[i]`
（`T.Serial(num_tiles)` 循环内）**即为源算法族存在在线变体的直接证据**（但该工厂不迁移，§0.5）。

**R3 算法复杂度对比**（四口径；per-row 计，M 行并行摊派；本算子为**带宽受限**：
roofline `flops = 2·M·N` vs `bytes = M·N·elem + M·8`，访存主导；口径：FLOPs 按「每元素 1 次比较 + 1 次
条件加」折算，不含指示化 cast 的搬运；访存量含输入读 + 输出写，**不含** UB 内部流量）：

| 算法候选 | FLOPs（per row） | 访存量 Bytes（GM） | GM 扫描遍数 | 中间缓冲峰值（per row） | 可并行度 / 跨核代价 |
|---------|------------------|--------------------|-------------|------------------------|---------------------|
| 基线（源：fp32 上抛 + 布尔物化 + 单次 `reduce_sum`，`N_padded` 域） | `2N_padded` | `N_padded·elem + 8`（`N=300` 时 `N_padded=512` ⇒ **+71%**） | 1 | `x_f32` 4 B/elem + `bool_vals` 4 B/elem（均 `(bm,N_padded)` 平面） | M/bm 行块独立；无跨核 |
| **C1 补式 + 原生域比较（本文主选，路径 R）** | `2N` | `N·elem + 8` | 1 | `x_ub` elem + `zero_ub` elem + `mask` 1 + `ind` 4 B/elem（探针实测编译器侧统一 **8 B/elem**） | 同上；无跨核 |
| C1 + 路径 C（列分块在线累加，`N > E_cap`） | `2N + O(num_full)`（微元累加） | `N·elem + 8`（**仍 1 遍**：每元素只读一次） | 1 | `O(tn)`：`tn` 档元素 × 8 B/elem | 同上；`acc` 片上驻留，无跨核 |
| C4 整数域累加 | `2N` | `N·elem + 8` | 1 | 指示平面 int32 4 B/elem ×2（`vselect` 双常数量）⇒ 高于 C1 | 同上（且 `i1→i32` cast 缺失） |
| C6 N-split partial+merge（两段 kernel） | `2N` | `N·elem + 8` **+ workspace 往返** `M·nchunk·(4+4)B × 2` | 1（+workspace 2 遍） | `O(tn)` + workspace | 跨核 workspace + 第二次 kernel 启动 + 合并正确性面 |
| （对照）源 tiled 回退（不迁移） | `2N_padded` | `N_padded·elem + 8` | 1 | `(bm,tile_n)` 平面组 | 同 C1；触发条件（fragment 列上限）在 NPU 不存在 |

**R4 硬件亲和性评估**（检查清单逐项；实测常数引用 `pattern-library/constants.md` 条目 + 版本戳；
**本任务设备为 910B3 / CANN 9.0.0**，凡绑定 910B2C / CANN 8.5.0 的常数标注
「未实测假设：沿用同代 910B 曲线量级，Stage 4 校准」）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（`T.Parallel` 融合 `T.if_then_else` 物化布尔平面） | ⚠️ 逐元素 `ite` 形态在 `bm≥2` 标量化（`TRAP-parallel2d-bm-ge2-scalarize`：触发项含循环体内 `ite`） | ⚠️ `x_f32`+`bool_vals` 双 4 B/elem 平面 | 需 `N_padded` 256 倍数 | ✅ | ✅ | ❌ 形态需重设计（→C1） |
| **C1 补式 + 原生域显式向量链（路径 R）** | ✅ 纯 Vector（`vcmp`/`vcast`/`reduce_sum` 全向量原语；`T.reduce.md` dtype 表 fp16/fp32 ✓、**bf16 ✗**——本设计**不在 bf16 域归约**（指示平面恒 f32），故规避；`vcmp.md` dtype 表 fp16/bf16/fp32 均 ✓） | ✅ **探针实测 8 B/elem**（编译器侧，与 dtype 无关）；单 block 硬上限 24576 元素；`E_cap = 8192` 元素（竞态安全档，§4.5） | ✅ **无需任何宽度对齐**（探针实测 raw N=64/4/2 均可编译；`T.copy` 切片满足 32 B 尾轴） | ✅ M/N/tile 边界全为工厂期常量 | ✅ auto-multi-buffer（Developer 自动施加）；无跨引擎 | ✅ **路径 R 选定** |
| C1 + 路径 C（列分块在线累加） | ✅ 同 C1 + `(bm,1)` 微元 `vadd`/`copy` 累加链（**no-alias**） | ✅ tile 工作集 `tn × 8 B ≤ 64 KB` | ✅ `tn` 为 256 倍数；尾块为实列宽度（探针证 raw 宽度可编译） | ✅ `num_full/tail` 全静态（工厂期）；`bm>1` 多 tile 已由**探针实测可编译**（`CG-2026-0014` 不复现） | ✅ tile 循环 auto-multi-buffer | ✅ **路径 C 选定**（`N > E_cap`） |
| C4 整数域累加 | ✅ Vector | ⚠️ 需 2 个额外 `(bm,W)` 常数量平面 | — | — | — | ❌（`i1→i32` cast 缺失 + 零收益，R3 量化） |
| C6 N-split 跨核 | ✅ | ⚠️ workspace GM 往返（`M·nchunk·8B × 2`；`(32,16384)` fp16 实例 ≈ 128 B×… 极小但需第二次 kernel 启动） | — | ✅ | ⚠️ 两 kernel + 合并正确性面 | ❌（本算子 M ≥ 32 ⇒ 核利用率已 ≥ 80%，无收益；R3 量化） |
| C7 排序/选择 | ✅ | ⚠️ O(N) 中间缓冲 | — | — | — | ❌（O(N log N)，R3 量化） |
| （对照）源 tiled 回退 | ✅ | ✅ | ✅ | ✅ | ✅ | ⏸ 不迁移（触发条件不存在，§0.5） |

**调研结论**：选定 **C1（零计数补式 + 原生 dtype 域比较 + 单次行向 `reduce_sum`），按元素预算双路径
（路径 R 驻留 / 路径 C 列分块在线累加）** 的算法族。

- **关键依据**：R3 表——访存遍数（1）与带宽（`N` 而非 `N_padded`）均不劣于基线，指示平面 1 个
  （基线 2 个 4 B/elem 平面）；R4——全 Vector 原语可达（`vcmp`/`vcast`/`reduce_sum` 均有文档与家族先例）、
  容量经**探针实测**定标（8 B/elem）、静态边界满足一维 Kernel 约束、无需任何对齐 pad（探针证据）。
- **与基线的结构差异一句话**：「fp32 上抛 + 布尔物化 + `Σ[x≠0]`（`N_padded` 域）」→
  「原生域 `vcmp eq` + i1→f32 指示平面 + `N − Σ[x==0]`（实列域，超预算时列分块在线累加）」。
- **调研范围（「无更优替代」的查证范围）**：`algorithm-candidates.md` 的 `ALG-reduction` / `ALG-elementwise` /
  `ALG-layernorm`（row-reduction 迁移基准形态）条目；`kb_search.py` 预注入 5 条 + 本设计新增检索
  （提示链 UB 单价 / int64 输出 / 内轴 reduce 标量化）；`examples/`（logsumexp single+tiled、argmin、argmax、
  `_primitives.py`）；pattern-library（constants 全条目、traps-compiler、traps-runtime、cases、layout）；
  恒等变形空间（C2–C7 全枚举，见 R1 表）；结构判据（在线变体存在性 / 单目的性 / 排序下界 / 跨核收益前提）。
  **R1 已枚举 8 个候选（基线 + C1–C7）**，无「无更优替代」空白断言。
- **源算法优化手段承接（§0.4 逐项）**：#1 SMEM tiling → UB staging（等价替换）；#2 256 对齐 pad → 舍弃
  （实列域计数，E3 验证）；#3 指示化 + 复用 sum 归约 → **保留**（算法层，改为 `vcmp`+`vcast` 显式链）；
  #4 fp32 上抛 → 舍弃（E4 穷举验证比较域等价）；#5/`#6` 预算启发式与 threads → 重设计为元素预算阶梯 + 一维
  persistent（R2/R5）；#7 `lru_cache` 保留；#8 tiled 回退 → 作为迁移对象舍弃、其「大 N 可处理」意图由
  R4（UB 容量驱动列分块）承接；#9 出栈 → UB staging + 显式 slice；#10 autotune 舍弃（K8）；#11 GPU 布局
  约束舍弃。**无静默丢弃项**。

**设计期 roofline（D-2；协议 `_shared/standards/hardware-cost-model.md` §2；常数带版本戳）**：

| workload（manifest / 测试） | 路径/配置 | 流量项（GM 字节 ÷ 有效带宽） | 发射项（向量 op 数 × 常数 ÷ 核数） | 估算下界（µs） |
|---|---|---|---|---|
| smoke `(32,256)` fp16/bf16/fp32 | R，bm=8，4 block | 16.4 KB ÷ (33.9 GB/s × 4)〔`CONST-mte2-degradation`〕≈ **0.12** | 4 block × ~7 op ÷ 4 × 0.25–0.5〔`CONST-msprof-task-floor-910B3` / `CONST-vector-launch-overhead`〕≈ **1.8–3.5** | **≈4–6**（含 2.4 µs 固定地板） |
| `sparsity-hidden (2048,4096)` fp16/bf16 | R，bm=2，1024 block → persistent 40（核内 serial 26） | 16.8 MB ÷ (30.2 × 40) = **13.9** | 1024 × 7 ÷ 40 × 0.25–0.5 = 45–90（串行上界；auto-multi-buffer 流水隐藏后取低档） | **≈45–90** |
| `sparsity-seq (32,16384)` fp16 | C，bm=1，tn=8192，2 tile，32 block | 1.05 MB ÷ (30.2 × 32) ≈ **1.1** | 32 × (2×6+4) ÷ 32 × 0.25–0.5 ≈ **4–8** | **≈6–11** |
| `3d-multidim (128,16384)` fp16 | C，bm=1，tn=8192，2 tile，128 block → persistent 40（serial 4） | 4.2 MB ÷ (30.2 × 40) ≈ **3.5** | 128 × 16 ÷ 40 × 0.25–0.5 ≈ **13–26** | **≈16–30** |
| 测试 `(128,300)` fp16 | R，bm=8，16 block | 76.8 KB ÷ (33.9 × 16) ≈ **0.14** | 16 × 7 ÷ 16 × 0.25–0.5 ≈ **1.8–3.5** | **≈4–6** |

> **估算下界（汇总）**：`估算下界 ≈ max(流量项, 发射项, 容量项) = 发射项主导`（`(2048,4096)` 档两项接近，
> 其余档发射项主导）；**容量项不适用**——纯 Vector 单引擎、无跨引擎 S/P 往返
> （`CONST-store-fixpipe-gm-only` 不适用）。发射项按 0.25–0.5 µs/op 双档给出（
> `CONST-vector-launch-overhead` 为 0.5 µs/op 的**串行上界模型**，`CONST-msprof-task-floor-910B3`
> 实测逐元素 pass 单价 0.23–0.3 µs/pass 为低档）。
> **同形校准锚点**：`(2048,4096)` fp16 同族（argmin，bm=2、1024 block、~10 op/block、48 核）实测 47 µs /
> 估算 56 µs；本算子 op 数更少（~7/block）、核数 40，故取 ≈45–90 µs 区间的中低档。
> **常数适用性声明**：`CONST-mte2-degradation`（MTE2 曲线）与 `CONST-vector-launch-overhead` 绑定
> **910B2C / CANN 8.5.0**，本任务为 **910B3 / CANN 9.0.0** ⇒ 按「未实测假设：沿用同代 910B 曲线量级」
> 引用，Stage 4 须按 `hardware-cost-model.md` §3 回填对账；
> `CONST-msprof-task-floor-910B3` 为**本工具链同设备**实测（可信档）；
> 核数（实查 40）与 UB 容量（实查 196608 B）**已本机实查复核**，且**指示链 UB 单价 8 B/元素为本任务探针实测**
> （非假设；`CONST-reduce-ub-per-element` 的整平面 ~128 B/元素模型**经探针证不适用于本 2-D 行归约形态**——
> 该条目的适用域限定为 `(CH,1,H*W)` 3-D 整平面归约，见 §9.1-C-1）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象：§1.6.0 选定算法 C1（+ 双路径落地形态）。逐项四要素；**等价性机器验证（D-1）**：
> `verify_equiv.py`（torch CPU、fp64 参照、精确整数比对、无 NPU 依赖），结果表内嵌于下。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| **E1** | 零计数补式 + IEEE `eq` 判决（承接 §0.6-R1） | `y = Σ_j [fp32(x_j) != 0.0]`（unordered NE，指示值 1.0/0.0 物化在全宽平面） | `y = N − Σ_j [x_j == 0.0]`（**原生 dtype 域**，ordered EQ） | ① 行内逐元素划分：`#{x≠0} + #{x=0} = N` ⇒ 恒等；② 判据正确性：`T.vcmp("eq")` 为 **IEEE 正确**（`traps-compiler.md` 的 `TRAP-vcmp-ne-ordered-nan` 原文：「`eq`/`le`/`ge` 为 IEEE 正确」），故 NaN==0 为假 ⇒ NaN 计入非零，与 torch 及源 unordered-NE 一致；③ **取反路线不可用**：`T.vnot` dtype 表不含 bool/int1（`docs/Tilelang.language/逻辑操作/T.vnot.md` §2.2.1），故 `[x≠0]` 只能以补式表达；④ 机器验证：E1 组 1377 行 0 偏差，E1c 对照组证明「有序 NE 路线」在 NaN 行确有 42 处差异（判别力成立） | ① 补式在 **(bm,) 微缓冲**上完成（1 个微元 `vsub`），**避免**在全宽 `(bm,W)` 平面上做 `1 − ind`（那将是 +1 个 4 B/elem 平面与 +1 个全宽 pass）；② 与 E4 合并后整链少 1 个全宽 pass（fp32 上抛） |
| **E2** | fp32 计数域的精确性界定（工厂 assert 的数学依据） | （无公式变更；界定「fp32 累加是否丢精度」） | 记零计数与总和均以 f32 累加，且工厂期断言 `N ≤ 2²⁴` | 指示值与全部分段计数均为**整数**，任何部分和 ≤ N；f32 对 `[0, 2²⁴]` 内整数**一一精确表示** ⇒ 每个 `reduce_sum`/`vadd` 步骤精确 ⇒ 计数精确（与累加顺序、分段方式无关） | 免去 int32/int64 累加路径所需的双常数量平面与额外汇点（C4，R3 已量化其零收益）；精确域 2²⁴ ≫ manifest 最大 N=16384 |
| **E3** | 实列域计数（舍弃源的 256 对齐 pad 域） | `Σ_{j<N_padded} [x_pad_j ≠ 0]`，`N_padded = align_up(N,256)`，pad 值 0.0 | `Σ_{j<N} [x_j ≠ 0]`（恰 N 列） | pad 列恒为 `+0.0`（计数单位元）⇒ 其指示值恒为 0 ⇒ 两侧计数逐行相等（`verify_equiv` E3 组 1377 行 0 偏差，覆盖 `N ∈ {1,2,8,16,64,256,300,512,1024,16384}` × 3 dtype） | 省 1 条掩码加载分支 + 1 个 `ite` 全宽 pass；`N=300` 类负载读流量 `512→300` 列（**−41.4%**）；免 kernel 侧宽度特化代码 |
| **E3b** | 分段/对齐 pad 的中性性（支撑 R4 尾块与 **备选** 对齐路线） | 单段全行计数 | 分段（`num_full` 个 tn 块 + 尾块）计数之和 | 整数加法结合律 + 分段覆盖恰为 `[0,N)`（无重无漏）⇒ 恒等；若采「宽度对齐 pad」备选路线，pad 区全 `+0.0` ⇒ 被 EQ 判为零 ⇒ 补式被减数取 **W** 时与实列域精确抵消（E3b 组 1377 行 0 偏差） | 支撑路径 C 的正确性（本设计主选不使用对齐 pad；E3b 同时验证备选路线的中性性） |
| **E4** | 比较域下放到原生 dtype（舍弃源的 fp32 上抛） | `[fp32(x_j) == 0]`（x 先 cast fp32 的全宽 pass） | `[x_j == 0]`（fp16/bf16/fp32 原生域） | fp16→fp32 与 bf16→fp32 均为**精确单射**且保 ±0 与 NaN（`TRAP-vcmp-ne-ordered-nan` 同族对 `eq` 的 IEEE 语义明证；`vcmp` dtype 表 fp16/bf16/fp32 均 ✓）⇒ 零判决逐元素恒等；**穷举验证**：fp16 与 bf16 **全部 2¹⁶ 位型**（共 262144 行）比较结果 0 差异（E4 组） | 省 1 个全宽 cast pass；`x_ub`/`mask` 之外的指示平面恒 f32（不随输入 dtype 放大）；**注意 bf16 不被 `reduce` 支持**——本设计归约域恒为 f32 指示平面，故不受该限制（`T.reduce.md` §2.2.1） |

**机器验证结果表**（`verify_equiv.py`，2026-10-16 执行，**exit 0**；torch 2.9.0+cpu、fp64 参照、精确整数比对；
shape 覆盖 `(1,1)/(1,2)/(3,8)/(4,16)/(8,64)/(16,256)/(32,300)/(64,512)/(7,1024)/(129,512)/(5,16384)`
× {fp16,bf16,fp32} + 角点行集〔±0（两种符号）/±inf/NaN/混合 NaN 行/亚正规〔最小正规与最小亚正规〕/
最大有限/`2²⁴` 与 `2²⁴+1`/全零/全一/奇偶交替/首末单点/亚正规稀疏〕）：

| 验证项 | 用例行数 | 最大偏差 | 违反率 | 结论 |
|--------|---------|---------|--------|------|
| E1 补式（`W − Σ[x==0]`，ordered EQ）vs 基线（`Σ[x!=0]`，unordered NE） | 1377 | 0 | 0/1377 | **EQUIV_PASS** |
| E1c 判别力对照：有序 NE 路线（**故意错误候选**，NaN 判为零） | 1377 | — | **42 处**（期望 > 0） | **CONTROL_OK**（若为 0 违反则脚本判 `CONTROL_BROKEN` 并自身失败） |
| E2 fp32 计数域精确性（f32 顺序前缀和 + 树形序两种累加，`N ≤ 2²⁴`） | 21 | 0 | 0/21 | **EQUIV_PASS** |
| E2-boundary 计数 > 2²⁴（`N=2²⁵+1` 全一 / `N=2²⁶` 密度 0.5） | 2 | 16777217 / 16775511 | 2/2（**期望偏差**） | **BOUNDARY_CONFIRMED**（佐证工厂 `assert N ≤ 2²⁴`） |
| E3 实列域（N）vs 256 对齐 pad 域（`N_padded`，pad=+0.0） | 1377 | 0 | 0/1377 | **EQUIV_PASS** |
| E3b 窄行/分段宽度对齐 pad（`W = align_up(N,16)`）的补式抵消 | 1377 | 0 | 0/1377 | **EQUIV_PASS** |
| E4 原生 dtype 比较 vs fp32 上抛比较（**全 2¹⁶ 位型穷举** × {fp16,bf16}） | 262144 | 0 | 0/262144 | **EQUIV_PASS** |
| INFO：NaN 语义矩阵（行 `[NaN, 0, 1, −0]`） | 1 | — | — | **INFO**：源(unordered NE)=2 / **采纳候选(ordered EQ 补式)=2** / 有序 NE 路线=1；`torch.count_nonzero = 2` ⇒ 采纳候选与 torch 及源三方一致，反之有序-NE 路线与两者分歧（非门禁项） |
| **合计（门禁等价组 E1/E2/E3/E3b/E4）** | **265096** | — | **0 项失败** | **ALL EQUIV_PASS**（另 +1379 对照/边界行，见上） |

**优化结论**：采纳 **E1 + E2 + E3 + E3b + E4** 共 5 项（其中 E3b 同时服务于路径 C 的正确性论证与备选路线中性性）。
优化后公式 = §1.3 第二式：`y[i] = N − Σ_{j<N}[x[i,j] == 0.0]`（原生 dtype 域比较；> `E_cap` 时按列分段累加，
被减数恒为整行宽度 N）。**被否决项**：R1 表中的 C2（自取反，净亏 1 个全宽 pass 且无 `T.vneg` 佐证）、
C3（位域，+2 pass）、C4（整数域累加，`i1→i32` cast 缺失 + 零收益，登记为 N>2²⁴ 备选）、C5（无公共子表达式）、
C6（跨核 N-split，核利用率已 ≥80% 无收益）、C7（排序，复杂度劣）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点对象：§1.4 两条路径的**全部**循环与标量计算点（含 §0.3 源步骤映射）。
> 替代 API 均有 `docs/Tilelang.language/` 或 `examples/` / `testing/` 佐证（列内标注）。

| # | 计算点 | 原实现形态（源 / 朴素形态） | 向量替代方案 | 替代结论 | 不可替代理由 |
|---|--------|---------------------------|-------------|---------|--------------|
| 1 | 零判决（源步骤 3 的 `ite` 全宽循环） | 2D `T.Parallel` 融合 `T.if_then_else`（`bm≥2` 标量化，`TRAP-parallel2d-bm-ge2-scalarize`） | `T.vcmp(x_ub, zero_ub, mask_eq, "eq")`（`docs/Tilelang.language/比较操作/T.vcmp.md`；bool 结果留片上，不可落 GM——同文档 §2.3） | ✅ | — |
| 2 | 指示化（1.0/0.0 平面） | 朴素：`T.Parallel` 赋值 `1.0/0.0` | `T.vcast(mask_eq, ind, round_mode="rint")`（i1→f32，`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 表含 `i1 → f32`） | ✅ | — |
| 3 | 行归约（源步骤 4） | 源已是 `reduce_sum` 原语（朴素替代为标量累加） | `T.reduce_sum(ind, zc, dim=1)`（**同 dtype 约束**：f32→f32；`docs/Tilelang.language/规约操作/T.reduce_sum.md`、`T.reduce.md` §2.2.1） | ✅ | — |
| 4 | 补式（`N − zc`） | 朴素：全宽平面 `1 − ind`（+1 全宽 pass） | `T.vsub(n_v, zc_flat, cnt)`（`(bm,)` 微缓冲；`docs/Tilelang.language/数学操作/T.vsub.md`）；`n_v` 为任务不变量（循环外一次 `T.fill`） | ✅ | — |
| 5 | 零常数量构造 | 朴素：逐元素写 0 | `T.fill(zero_ub, <typed 0>)`（`tilelang/language/fill.py`；**值须为类型化常量**——探针实测 Python 字面量 `0` 被前端拒绝 `value must be a tir.PrimExpr`，须用 `tir.const(0, dtype)`） | ✅ | — |
| 6 | fp32 上抛（源步骤 2a/2b） | `T.Parallel` 全宽 cast 循环 | **消除**（E4：原生域比较，全 2¹⁶ 位型穷举验证） | ✅（消除） | — |
| 7 | 分段累加（路径 C） | 朴素：`acc[i] = acc[i] + tile_zc[i]` 原位累加（源 tiled 形态） | `T.vadd(acc, zc, new_acc)` + `T.copy(new_acc, acc)`（**no-alias**；`docs/Tilelang.language/数学操作/T.vadd.md`；规避 `TRAP-vselect-inplace-carried-state` 同族的原位陷阱） | ✅ | — |
| 8 | int64 出栈（源步骤 5+6） | `T.Parallel` 逐元素 `T.cast` + 尾块无守卫 `T.copy` | `T.reshape(cnt → cnt_flat (bm,))` + `T.vcast(cnt_flat, out_ub, "int64", round_mode="rint")` + `T.copy(out_ub[0:real_m], out[start:start+real_m])`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` 的 `f32 → i64` 行；`docs/Tilelang.language/shape操作/T.reshape.md`） | ✅（向量化） | — |
| 9 | 尾块行数 / 任务 guard | 源缺失（越界，§0.4 源缺陷） | `T.min` 标量 + `if start < M` 动态 guard（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 示例先例） | ❌（保留） | **block 级边界处理**：依赖运行时 pid/task 的任务边界，非逐元素热点（每 block 1 次）；属 VP-2026-0044 三件套标准形态（serial 上界静态 + guard 动态） |
| 10 | tile 循环控制（路径 C） | `T.Serial(num_tiles)`（源 tiled） | 保留 `T.serial(num_full)`（静态上界） | ❌（保留） | **tile 级顺序依赖**：`acc` 是循环携带状态（在线累加的载体）；`num_full` 为工厂期静态常量（静态边界要求），循环体内逐元素计算已全部向量化 |
| 11 | host 侧 `bm/tn/num_full/tail` 选择 | Python 工厂期计算 | — | ❌（保留） | **host 元数据计算**（不在 kernel 内；每 `(M,N,dtype)` 一次，`lru_cache` 命中后零成本） |

**向量化结论**：逐元素计算点已**全部向量化**（#1–#8，API 均有文档 + 家族先例佐证）；
保留 **3 处**标量/循环（#9 任务 guard / #10 tile 循环控制 / #11 host 元数据），理由类别分别为
「block 级边界处理」「tile 级顺序依赖」「host 元数据」——三者发射量均为每 block/tile 1 次量级，
**非逐元素热点**。源实现中的逐元素标量热点（源步骤 3 的 `ite` 融合形态在 `bm≥2` 会整体标量化）
已被 #1+#2 的显式向量链替代。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（`(M,N)` 行主序进 / `(M,)` int64 出）；**核内布局与 lane 映射是设计变量**。
> 本算子 2D `(M,N)`、规约维 = 最后一维 → 按**规约类必选候选**枚举（水平归约 vs 垂直扫描为正交决策）。

**轴质量评分**：

| 评分项 | 水平归约（lane→N 轴，`reduce dim=1`） | 垂直扫描（lane→M 行，串行步进 N 维） | 说明 |
|--------|------|------|------|
| 整除性 | `N` 任意（探针证 raw N=2/4/64/300/4096/16384 均可编译）；tn 路径取 256 倍数 | 同理 | 无差异 |
| 尾 lane 浪费率 | tn 路径为 256 倍数 ⇒ 0 浪费；raw `N` 路径（如 300、4）在最末向量组有部分空 lane，浪费上界 = `(向量宽−1)/N` ≤ 7/300 = **2.3%**（`N=4` 的测试形态另见 §9.2-R4） | 同 | 两者一致（同一条轴） |
| 累加链形态 | 归约维即向量维（`dim=1`，stride=1）——`reduce` 原生形态 ✓ | 归约维变串行步进维：每行 N 次 `(bm,1)` 微元 op（≥ N×0.25 µs 发射/行） | **决定性差异** |
| repack 代价 | 无（I/O 原生行主序即 N 连续） | 无 | — |
| UB 容量影响 | 行块 `(bm,N)` 或 tile `(bm,tn)` 驻留（8 B/elem 实测） | 同 | — |

**候选矩阵**（含 GPU 源轴选择对照——源码 thread/warp 轴是输入而非结论）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|---------|------------|--------------|------|
| 1 | **行块 × 全 N 驻留 + 水平归约**（`lane→N`，`reduce dim=1`；persistent 按 M 行块分核） | N（连续，`x_ub` 行内 stride=1） | 无 | 覆盖 `N ≤ E_cap` 的全部 manifest/测试负载；`ALG-layernorm` R4 的 row-reduction 迁移基准形态（logsumexp / ada_layer_norm 两任务实测）+ `VP-2026-0044`（**pending 提案**，非已合入规则，仅作形态参考） | ✅ **主选（路径 R）** |
| 2 | 垂直扫描（`lane→M`，串行步进 N 维） | M | 无 | 归约维串行化 → 每行 N 次微元发射（`N=4096` → ms 量级/行）——发射项灾难（`CONST-vector-launch-overhead`） | ❌（发射模型量化） |
| 3 | 核内转置（N-major）后行归约 | M（转置后连续） | `T.transpose` 二轴交换链（UB 级；`docs/Tilelang.language/创建操作/T.transpose.md`） | **纯开销零收益**：主选 #1 的向量化轴 N 即 I/O 原生连续轴，无 repack 需求；核内转置不减少 GM 流量与向量 pass 数，另需 +1 个 `(bm,W)` UB 平面（按 8 B/elem 口径 +2~4 B/elem）。**不得按 GPU 直觉把核内重排写成 GM 级重排的稻草人**——此处否决理由为「量化收益为零」而非「代价高」 | ❌ |
| 4 | **列分块 + 运行累加（在线）** | N（分段连续） | 无 | `N > E_cap` 时的**必需**维度（否则单 block 超 UB 容量）：`tn × 8 B/elem ≤ 64 KB`；GM 流量不变（1 遍）；代价 = 每 tile 2 个微元累加 op + tile 循环控制 | ✅ **主选（路径 C）** |
| 5 | 判决式/指示式的 pass 组织：**显式向量链**（`vcmp`+`vcast`）vs 2D `T.Parallel` 融合 `T.if_then_else`（同轴 #1 内的两种实现） | N | 无 | 融合式：`TRAP-parallel2d-bm-ge2-scalarize` 的触发项含**循环体内 `ite` 形态** ⇒ `bm≥2` 整体标量化；显式链：`bm=1/bm≥2` 统一向量化。**2026-10-15 适用域收窄**记录「plain 逐元素 2D loop（单向量 op/算术/cast）在 `bm≥2` 正常向量化」——故「无 `ite` 的 plain 2D 循环」可作 Stage 4 简化候选（§6.3） | ✅ **显式链主选** |
| 6 | 跨核 N-split（partial + merge 两段 kernel） | N（跨核切分） | workspace GM 往返 `(M,nchunk)×8B` | 小 M 大 N 场景可提核利用率；本算子 manifest M ≥ 32（`sparsity-seq` M=32 ⇒ 32/40 核）⇒ 收益面窄，代价 = +1 kernel 启动 + workspace 往返 + 合并正确性面 | ❌（R3 量化；登记为 Stage 4 可选，非本设计三件套） |
| 7 | **超预算单缓冲驻留**（`N ∈ (E_cap, 24576]` 时 bm=1 不双缓冲，省 1 个 tile pass） | N | 无 | 探针证该区间**可编译**（fp16 16384 元素通过、32768 失败）；代价 = 放弃 auto-multi-buffer（共享缓冲竞态面，§4.5） | ⏸ **§1.6.4 实验裁决备选** |

**布局决策结论**：选定 **#1 + #4 + #5**——核内布局 = 行块 `(bm, ·)` 行主序驻留（UB staging，
`T.alloc_shared` + 显式 slice），向量化轴 = **N 轴**（`reduce dim=1`，判决与指示化走显式
`vcmp→vcast` 链，补式落在 `(bm,)` 微缓冲），repack 无，分核 = persistent 按 M 行块（§5.5）；
`N ≤ E_cap` 走单 tile（#1），`N > E_cap` 走列分块在线累加（#4）。该结论同步落入 §3.3 伪代码与 §6 循环结构。
**弃选理由**：#2 发射爆炸（量化）；#3 纯开销零收益（主选轴即 I/O 原生连续轴）；#6 收益面窄 + workspace/kernel
启动代价；#7 为实验裁决项（三件套齐备，§1.6.4）。

#### 1.6.4 实验裁决备选方案：超预算单缓冲驻留（Stage 4 A/B 三件套）

> §1.6.3-#7 的备选完整结构 + 裁决计划（skill Phase 2 实验裁决模式）。**主选** = §1.4 双路径
> （`N ≤ E_cap` 驻留 / `N > E_cap` 列分块，`E_cap = 8192` 元素 = 竞态安全预算）。**触发形态** =
> 单 block 元素数落在 `(8192, 24576]` 的负载（manifest 的 `sparsity-seq (32,16384)` 与
> `3d-multidim (128,16384)` 恰在该区间）：主选把它们送进列分块（多 1 个 tile pass ×2 微元累加），
> 备选把它们按 `bm=1` 单 tile 驻留处理（探针证 16384 元素可编译，但放弃 auto-multi-buffer 双缓冲）。
> §3.3/§4/§6 只承载主选方案，备选结构独立成节、不与主选混排。
>
> **判定裕度的不确定性来源（未实证常数）**：① 超预算时 auto-multi-buffer 被静默放弃后，**单消费者**
> 共享缓冲（本设计每缓冲恰 1 个消费者，见 §4.5 纪律）是否仍会竞态——同族实证的竞态样本是**多消费者**
> 形态（`TRAP-UB-multibuffer-inflation` / argmin 探针），单消费者形态**无实测证据**；
> ② 「省 1 个 tile pass」的真实收益量级取决于发射单价（`CONST-vector-launch-overhead` 0.5 µs/op 上界 vs
> `CONST-msprof-task-floor-910B3` 0.23–0.3 µs/pass 低档），两档给出的收益差 ~2×。

**① 分派条件（工厂期静态谓词，per-shape 分派可行性）**：
`E_cap < N ≤ UB_HARD_ELEMS`（fp16/bf16：`8192 < N ≤ 24576`；fp32 由 8 B/elem 同口径）。工厂已按
`(M, N, op_kind, dtype)` 分派（源 `lru_cache` 语义保留，§0.4-#7），故主选/备选可按 shape 共存，
**不构成全局二选一**。

**② 备选结构（路径 R′：超预算单 tile 驻留；fp16 `N=16384, M=32` 实例）**：

```python
# 与路径 R 逐条同构，仅两处不同：
#   (a) bm 固定 = 1（单 block 元素数 = N = 16384 ≤ UB_HARD_ELEMS = 24576，探针实测可编译）；
#   (b) 不设 num_full/tail（无 tile 循环）——省 1 个 tile 的 2 个微元累加 op；
# 预算（手工代理，§4.5 口径；实测编译器单价 8 B/elem）：
#   16384 元素 × 8 B = 131072 B = 128 KB ≤ 192 KB（单缓冲）；双缓冲 256 KB > 192 KB ⇒ 自动双缓冲被放弃
with T.Kernel(T.ceildiv(M, 1), is_npu=True) as (pid, _):
    x_ub    = T.alloc_shared((1, N), dtype)          # 128 KB 档（fp16）
    zero_ub = T.alloc_shared((1, N), dtype)
    mask_eq = T.alloc_shared((1, N), "bool")
    ind     = T.alloc_shared((1, N), "float32")
    zc      = T.alloc_fragment((1, 1), "float32")
    zc_flat = T.alloc_fragment((1,), "float32")
    n_v     = T.alloc_fragment((1,), "float32")
    out_ub  = T.alloc_shared((1,), "int64")
    start = pid
    if start < M:                                     # 动态 guard（M=32 时恒成立，保留为标准形态）
        T.fill(zero_ub, tir.const(0, dtype))
        T.fill(n_v, tir.const(N, "float32"))
        T.copy(x[start:start + 1, 0:N], x_ub[0:1, 0:N])
        T.vcmp(x_ub, zero_ub, mask_eq, "eq")
        T.vcast(mask_eq, ind, round_mode="rint")
        T.reduce_sum(ind, zc, dim=1)
        T.reshape(zc, zc_flat)
        T.vsub(n_v, zc_flat, zc_flat_out)             # 补式（(1,) 微缓冲）
        T.vcast(zc_flat_out, out_ub, round_mode="rint")
        T.copy(out_ub[0:1], out[start:start + 1])
```

- **预算（备选）**：fp16 `N=16384` 单缓冲 128 KB ✓（无余量为双缓冲）；bf16 同；fp32 由 8 B/elem 单价
  同口径 ⇒ `N=16384` 亦为 128 KB 档（探针实测 fp32 8192 元素通过、更大值建议 Stage 3 复标）。
  `N > 24576` 一律走主选路径 C（硬上限）。
- **分核三要素（备选）**：逻辑核数 = `M`；物理核数 = **40**（§5.5 实查同源）；规模判定 = 逻辑 32 ≤ 40
  （`sparsity-seq`）⇒ `num_kernels = 32`、每核 1 任务；`3d-multidim` M=128 > 40 ⇒ persistent 40 +
  核内 `T.serial(4)` + guard（静态上界）。
- **风险**：单块 128 KB 无双缓冲余量 ⇒（i）auto-multi-buffer 被放弃后的单消费者竞态面（无实测证据）、
  （ii）GM 搬运与向量计算**不可重叠**（`CONST-reduce-ub-per-element` 的伴随实测：
  「整平面 `reduce` 与 GM→UB `T.copy` 不重叠」）⇒ 流量项暴露。这是本裁决的核心待测点。

**③ 实验裁决计划**：

- **代表 shape**：`sparsity-seq (32,16384)` fp16（manifest bench）+ `3d-multidim (128,16384)` fp16
  - 合成档 `(2048,16384)` fp16（放大 M 以放大流量项）。
- **指标**：bench harness `Task Duration`（`launches ≥ 30` 均值，`examples/TileOPs/benchmarks/ops/bench_count_nonzero.py`
  同款口径）+ 精度门（`torch.equal` 全 PASS 为前提；主选/备选必须逐位一致——E3b 已证分段恒等）。
- **判定阈值**：备选 R′ 端到端延迟 **< 0.8 ×** 主选同 shape 延迟 ⇒ 该 shape 类分派翻转（主选/备选
  按 shape 共存，非全局翻转）；实测同时记录「单消费者超预算是否竞态」（多次重复运行 + 逐位比对）。
- **回写路径**：实测数据与结论回写本节裁决记录 + `RETROSPECTIVE.md`；若翻转，修订 §1.4/§4.5/§5.2
  增补 R′ 分派分支并过 Stage 2 复审；Stage 3 实现期若需提前引入，走 `[DESIGN_LIMIT]` 受控修订路由。
- **常数回填**：本裁决直接标定「发射单价双档」与「单消费者超预算竞态」两个量
  （`_shared/standards/hardware-cost-model.md` §3 回填对账），并作为 `constants.md` 候选条目
  （`CONST-vector-launch-overhead` 的 910B3 数据点）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（逐元素比较 + 类型转换 + 行归约 + 微元累加），无 matmul、无 Cube | 不需要 L1/L0；`T.alloc_shared`（映射 UB）+ `T.alloc_fragment` 即可 |
| 归约 | `T.reduce_sum(dim=1)`——Developer / Expert 两模式均可用（`docs/Tilelang.language/规约操作/T.reduce_sum.md`；同族 logsumexp 单 tile 为 Developer 先例） | ✓ |
| 内存层级 | 仅 GM ↔ UB ↔ fragment，无跨引擎传输 | Developer 自动管理 |
| 同步 | 单 kernel 内顺序执行 + persistent 任务循环 + tile 循环；无核间协作 | Developer 自动同步即可 |
| 家族先例 | logsumexp single/tiled、argmin/argmax 同族迁移均为 Developer（row-reduction 迁移基准形态） | 沿用 |
| 调用方约束 | 迁移 prompt 指定 developer 模式（用户未另行指定） | ✓ |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（GM staging 与向量操作数/结果平面）+ `T.alloc_fragment`（归约目标与 `(bm,)` 微缓冲） |
| 计算方式 | v 前缀 API（`vcmp`/`vcast`/`vsub`/`vadd`）+ `T.reduce_sum` + `T.fill` + 一维 persistent Kernel |
| 同步 | Developer 自动同步（无手动 `T.sync_block_set/wait`、无 `T.set_flag/wait_flag`） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True)` 一维 persistent（`docs/Tilelang.language/编译器提示操作/T.Kernel.md`） |
| 环境变量 | `TILELANG_ASCEND_MODE=Developer`（wrapper `ascend_mode` 已声明；同族一致） |

---

## 3. API 映射设计

### 3.1 公式拆解（输入 = §1.6.1 优化后公式）

**路径 R（resident，`N ≤ E_cap`）**：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| R1 | `x_ub ← x[start:start+real_m, 0:N]` | GM→UB staging；src/dst 双显式 slice（尾块守卫） |
| R2 | `mask_eq = (x_ub == 0)` | 原生 dtype 域逐元素零判决（**IEEE ordered EQ**） |
| R3 | `ind = f32(mask_eq)` | i1→f32 指示平面（0.0/1.0） |
| R4 | `zc = Σ_j ind[:, j]` | 行零计数 `(bm,1)` f32（整数精确域） |
| R5 | `y = N − zc` | **补式**（`(bm,)` 微缓冲；`n_v ≡ N` 为任务不变量） |
| R6 | `out[start:start+real_m] ← int64(y)` | 向量化 int64 出栈 + slice 写 |

**路径 C（column-tiled，`N > E_cap`）**：tile 循环内重复 R1–R4（R4 结果记 `tile_zc`），
以 `T.vadd` + `T.copy`（**no-alias**）累加进 `acc`；尾块（`tail > 0`）用独立静态宽度缓冲组走同一链；
出栈为 R5（被减数恒为**整行宽度 N**）+ R6。

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 | 佐证 |
|------|----------|-------------|------|------|------|
| R1/C1 载入 | GM→UB | `T.copy(x[start:start+real_m, c0:c1], x_ub[0:real_m, 0:w])` | src/dst 双显式 slice；`w` 为静态宽度 | Developer | logsumexp single（`_logsumexp_kernel_single.py` 的 `T.copy` 形态）；VP-2026-0044（pending）三件套 |
| R2 零判决 | `[x == 0]` | `T.vcmp(x_ub, zero_ub, mask_eq, "eq")` | bool dst `(bm,w)`；**`eq` 为 IEEE 正确**（ordered EQ） | Developer | `docs/Tilelang.language/比较操作/T.vcmp.md` §2.2.2（同 shape）+ §2.3（bool 不可落 GM）；`traps-compiler.md` 的 `TRAP-vcmp-ne-ordered-nan` |
| R3 指示化 | `1.0/0.0` 平面 | `T.vcast(mask_eq, ind, round_mode="rint")` | i1→f32（dtype 表含 `i1 → f32`） | Developer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 |
| R4 行归约 | `Σ_j ind` | `T.reduce_sum(ind, zc, dim=1)` | dim=1；**dst dtype ≡ src dtype**（f32→f32） | Developer | `docs/Tilelang.language/规约操作/T.reduce_sum.md`、`T.reduce.md` §2.2.1（fp16/fp32 ✓、bf16 ✗） |
| R5 补式 | `N − zc` | `T.reshape(zc, zc_flat (bm,))` + `T.vsub(n_v, zc_flat, cnt)` | `(bm,)` 微缓冲；`n_v` 由 `T.fill` 预置 | Developer | `docs/Tilelang.language/shape操作/T.reshape.md`；`docs/Tilelang.language/数学操作/T.vsub.md` |
| R6 出栈 | `int64(y)` | `T.vcast(cnt, out_ub, round_mode="rint")` + `T.copy(out_ub[0:real_m], out[start:start+real_m])` | f32→i64；slice 写 | Developer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（`f32 → i64` 行）；argmin 同族出栈形态 |
| C 累加 | `acc += tile_zc` | `T.vadd(acc, tile_zc, new_acc)` + `T.copy(new_acc, acc)` | `(bm,1)`；**no-alias**（禁原位） | Developer | `docs/Tilelang.language/数学操作/T.vadd.md`；`traps-compiler.md` 的 `TRAP-vselect-inplace-carried-state`（同位形态禁令） |
| 常量构造 | `0` / `N` | `T.fill(buf, <typed const>)` | **值须为类型化 PrimExpr**（`tir.const(0, dtype)` / `T.float32(N)`）；Python 字面量被拒（探针实测 `value must be a tir.PrimExpr`） | Developer | `tilelang/language/fill.py`；探针实测错误文本 |
| Grid | persistent | `T.Kernel(num_kernels, is_npu=True)` + `T.serial(<静态上界>)` + `if start < M` | 一维 | Developer | `docs/Tilelang.language/编译器提示操作/T.Kernel.md`；`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4（guard 先例）；`layout.md` 的 `PL-1.21-vector-persistent-split` |

**dtype 路径矩阵**（比较域 = 输入原生 dtype；指示/归约域恒 f32；**每路径独立 `prim_func`，工厂期分派**——
TVMScript 禁止条件定义变量，且「未用 alloc 不被 DCE」会吃 UB 预算，故禁「无条件 alloc + 条件使用」）：

| 输入 dtype | 比较域（`vcmp`） | 预转 pass | 指示/归约域 | 手工代理 B/elem（§4.5） |
|-----------|------------------|-----------|-------------|--------------------------|
| float16 | fp16 原生域 | 无（E4） | f32 | 9 |
| bfloat16 | bf16 原生域 | 无（E4；`vcmp` 表含 bf16） | f32（**不在 bf16 域归约**，规避 `reduce` 不支持 bf16 的限制） | 9 |
| float32 | fp32 原生域 | 无 | f32 | 13 |

### 3.3 计算伪代码

```python
# 工厂期（Python，全部静态量）：N、E_cap、路径、bm、tn、num_full、tail、num_kernels、serial 上界
def _logical_reduce_kernel(M, N, op_kind, dtype):        # 接口语义参数不变（harness 契约）
    assert op_kind == "count_nonzero"                    # 本迁移单元只实现该 kind（§0.2）
    assert N >= 1 and N <= 2 ** 24                       # E2 精确域前提（超限 raise，§9.2-R7）
    E_CAP = 8192                                         # 元素/block（竞态安全档，§4.5）
    UB_HARD_ELEMS = 24576                                # = 196608 / 8（探针实测单价，§4.5）
    # ── 路径判定 + 阶梯 ──
    if N <= E_CAP:                                       # 路径 R：单 tile 驻留
        path, bm = "R", None
        bm = max([p for p in (1, 2, 4, 8) if p * N <= E_CAP] or [1])
        if ceildiv(M, bm) > 80:                          # >2 波 40 核 → 扩展阶梯
            bm = max([p for p in (16, 32, 64, 128, 256, 512, 1024)
                      if p * N <= E_CAP and p <= M] + [bm])
        bm = min(bm, M)
        tn, num_full, tail = N, 1, 0
    else:                                                # 路径 C：列分块 + 运行累加
        path, bm = "C", 1
        cap = min(E_CAP, N)
        tn = pick_tile_n(N, cap, align=256)              # ≤cap 的最大 256 倍数且整除 N；无整除因子取最大 256 倍数
        num_full, tail = N // tn, N - (N // tn) * tn
    num_kernels = min(ceildiv(M, bm), 40)                # 物理核数 = 实查 aivector = 40（§5.5）
    num_local_tasks = ceildiv(ceildiv(M, bm), num_kernels)   # 静态上界

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):                                  # K9：threads 已去除
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub    = T.alloc_shared((block_m, tn), dtype)      # GM staging（唯一按 tn 放大的输入平面）
                zero_ub = T.alloc_shared((block_m, tn), dtype)      # 零常量（任务不变，循环外 fill）
                mask_eq = T.alloc_shared((block_m, tn), "bool")     # 零判决掩码
                ind     = T.alloc_shared((block_m, tn), "float32")  # 指示平面（f32；bf16 归约限制的规避形态）
                acc     = T.alloc_fragment((block_m, 1), "float32")
                new_acc = T.alloc_fragment((block_m, 1), "float32")
                zc      = T.alloc_fragment((block_m, 1), "float32")
                zc_flat = T.alloc_fragment((block_m,), "float32")
                cnt     = T.alloc_fragment((block_m,), "float32")
                n_v     = T.alloc_fragment((block_m,), "float32")
                out_ub  = T.alloc_shared((block_m,), "int64")
                T.fill(zero_ub, tir.const(0, dtype))            # 任务不变量
                T.fill(n_v, tir.float32(N))                     # 补式被减数（任务不变量）
                for s in T.serial(num_local_tasks):             # 静态上界 + grid-stride
                    start = (cid + s * num_kernels) * block_m
                    if start < M:                              # 动态 guard（边界静态化）
                        real_m = T.min(block_m, M - start)     # M 尾块行数（§0.6-R6）
                        if path == "R":
                            _tile_body(x, start, real_m, 0, tn, ...)      # 单 tile，zc 直出
                            _epilogue(real_m, ...)
                        else:
                            T.fill(acc, tir.float32(0))
                            for t in T.serial(num_full):              # 静态边界
                                _tile_body(x, start, real_m, t * tn, tn, acc)
                            if tail > 0:                              # 静态尾块（独立宽度，工厂期条件生成）
                                _tile_body_tail(x, start, real_m, num_full * tn, tail, acc)
                            _epilogue(real_m, ...)
        return main
    return _func

# _tile_body（逐元素计算全部向量化；R2 存与 R5 互补——R5 在 epilogue 完成，故此处只算 tile_zc）
#   T.copy(x[start:start+real_m, c0:c0+tn], x_ub[0:real_m, 0:tn])
#   T.vcmp(x_ub, zero_ub, mask_eq, "eq")
#   T.vcast(mask_eq, ind, round_mode="rint")
#   T.reduce_sum(ind, zc, dim=1)                      # 路径 R：zc 即最终零计数
#                                                    # 路径 C：随后 T.vadd(acc, zc, new_acc) + T.copy(new_acc, acc)
# _epilogue（补式 + int64 出栈，R5+R6）
#   T.reshape(zc_or_acc, zc_flat) ; T.vsub(n_v, zc_flat, cnt)
#   T.vcast(cnt, out_ub, round_mode="rint") ; T.copy(out_ub[0:real_m], out[start:start+real_m])
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|----------|
| `T.vcmp(src, src, dst_bool, "eq")` | `docs/Tilelang.language/比较操作/T.vcmp.md`（shape 须同形、bool 不可落 GM）+ `traps-compiler.md` 的 `TRAP-vcmp-ne-ordered-nan`（`eq` 为 IEEE 正确） | ✅ 文档 + **本任务探针实测可编译** |
| `T.vcast(mask_bool, ind_f32, round_mode="rint")` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（`i1 → f32` 行） | ✅ 文档 + **本任务探针实测可编译**（含该链的 UB 单价标定） |
| `T.reduce_sum(ind_f32, zc_f32, dim=1)` | `docs/Tilelang.language/规约操作/T.reduce_sum.md`、`T.reduce.md` §2.2.1/§2.2.2 | ✅ 文档 + 同族 logsumexp 实测 |
| `T.vsub` / `T.vadd`（`(bm,)` / `(bm,1)` 微缓冲） | `docs/Tilelang.language/数学操作/T.vsub.md`、`docs/Tilelang.language/数学操作/T.vadd.md`；同族 logsumexp 有 in-place `vsub`/`vadd` 先例 | ✅ 文档 + 家族先例 |
| `T.fill(buf, typed const)` | `tilelang/language/fill.py` | ✅ 源码 + **探针实测**（须 `tir.const(..., dtype)`，字面量被拒） |
| `T.copy` 带显式 slice（尾块 / 列块） | `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/`、`.../_logsumexp_kernel_tiled/` | ✅ 先例 + **探针实测**（含 raw 窄宽度与列块） |
| `T.Kernel(一维, is_npu=True)` + `T.serial` + 动态 `if` guard | `docs/Tilelang.language/编译器提示操作/T.Kernel.md`；`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 示例 | ✅ 文档 + 先例 |
| `T.reshape`（形状收缩 `(bm,1)→(bm,)`） | `docs/Tilelang.language/shape操作/T.reshape.md`；argmin 同族出栈先例 | ✅ 文档 + 先例 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 persistent grid（§5.5） |
| 纯 Vector 算子核数翻倍 | Yes | **40 = 实查 `get_aivector_core_num()`**（§5.5） |
| `reduce` 混合 dtype 静默错误 | Yes（潜在） | **全程同 dtype 归约**（f32 指示平面 → f32 `zc`）；bf16 不在归约域出现 |
| bf16 不被 `reduce` 支持 | Yes（潜在） | 归约域恒为 f32（指示平面 f32），bf16 仅出现在 `vcmp` 比较域（dtype 表 ✓） |
| GEMM 非整除 / L0C 容量 / 分形限制 | No | 无 Cube 计算 |
| UB 容量 | Yes | §4.5 逐配置核算（探针实测 8 B/elem；`E_cap=8192` 元素；硬上限 24576） |
| `-0.0` / NaN 语义 | Yes | `vcmp("eq")` 为 IEEE 语义（`TRAP-vcmp-ne-ordered-nan`）；§1.6.1-E1/E1c + §8.2 L1 覆盖 |
| 原位 select/累加丢失 loop-carried 状态 | Yes（路径 C） | 累加走 **no-alias** + `T.copy` 回写（`TRAP-vselect-inplace-carried-state` 同族禁令） |
| 2D `T.Parallel` 融合 `ite` 在 `bm≥2` 标量化 | 已规避 | 判决走 `vcmp`/`vcast` **显式向量链**，不用 `ite` 融合（§1.6.3-#5） |
| host 侧禁改输入内容 / 禁 padding | Yes | kernel 接 raw `(M,N)`，**无 host pad**（Op 层 `_kernel_handles_padding=True`）；host 侧仅元数据（`bm/tn` 计算） |
| 未用 fragment alloc 不被 DCE | Yes | dtype 路径矩阵：每路径独立 `prim_func`，禁「无条件 alloc + 条件使用」 |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键差异汇总）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 并行 | `T.Kernel(grid, threads=threads)` 两级 | 一维 `T.Kernel(num_kernels, is_npu=True)` persistent | `threads` 去除（K9）；`num_kernels = min(ceil(M/bm), 40)` + 静态 serial（§0.6-R2） |
| SMEM tiling | `alloc_shared` 48 KB 预算 + 256 对齐 | UB（实查 192 KB）+ **元素预算**（探针实测 8 B/elem） | §5.2 阶梯 + §4.5 预算表 |
| 判决式 | `x != 0.0`（unordered NE） | **`vcmp("eq")` 补式**（ordered EQ，IEEE 正确） | §0.6-R1 / §1.6.1-E1（机器验证 + 对照组） |
| 比较域 | 全量 fp32 上抛 | 原生 dtype 域（fp16/bf16/fp32 均 ✓） | §1.6.1-E4（全 2¹⁶ 位型穷举） |
| 指示化 | `T.Parallel` 融合 `T.if_then_else`（`bm≥2` 标量化） | `vcmp` + `vcast` 显式向量链 | §1.6.3-#5 |
| padding | 内部 `align_up(N,256)` + 单位元掩码加载 | **完全不 pad**（实列域计数） | §0.6-R3（E3 机器验证） |
| 列分块 | `_logical_reduce_kernel_tiled`（fragment 列上限 32768 触发，**不迁移**） | UB 容量触发（`N > E_cap`），`bm=1` + 静态尾块 + no-alias 累加 | §0.6-R4（探针实测可编译，`CG-2026-0014` 不复现） |
| 尾块守卫 | 缺失（源缺陷） | `real_m = T.min(...)` + src/dst 双 slice + 垃圾行丢弃 | §0.6-R6 |
| 出栈 | `T.Parallel` cast + `T.copy` | `T.reshape` + `T.vcast(rint)` + slice copy | §3.2 R6 |
| jit 装饰器 | `@tilelang.jit(out_idx=[1])` | `@tilelang.jit(out_idx=[1], target="npuir")` | 同族一致 |
| 配置搜索 | autotune（`tune_by_forward`） | 启发式阶梯（K8 已移除 tune） | §5.2 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py` | **高**（同族 row-reduction、Developer、`reduce dim=1`、尾块 `T.min` + 双 slice、**raw 非对齐 N=300/32 实测通过**） | Kernel 结构与 raw-N 契约、JIT 形态、尾块处理 |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_tiled/_logsumexp_kernel_tiled.py` | 高（N > 驻留上限的列循环 + 运行状态更新链） | 列循环结构、运行累加形态、静态尾块 |
| `examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/_argreduce_kernel_DESIGN.md` | 高（同族迁移设计结构 + 探针证据链；910B2C / CANN 8.5.0） | §0 迁移分析组织、§1.6 调研与优化、§4.5 UB 记账、§5.2 阶梯、§5.5 分核、§9 风险清单 |
| `examples/argmin/_argreduce_kernel/DESIGN.md` + `examples/argmin/_argreduce_kernel/verify_equiv.py` | 高（**同工具链 910B3 / CANN 9.0.0**、纯 Vector、`reduce dim=1`、int64 出栈、D-1 脚本形态） | 分核三要素、UB 代理预算口径、int64 出栈（`reshape`+`vcast(rint)`）、等价性脚本骨架 |
| `examples/TileOPs/tileops/kernels/reduction/count_nonzero/count_nonzero.py` | **最高**（本算子 wrapper，K5–K9 已适配） | `default_config` 改接本设计 `_select_config`、`custom_op` 契约、`forward` 传参 |
| `examples/TileOPs/tileops/ops/reduction/count_nonzero.py` | **最高**（本算子 Op 层） | `_kernel_handles_padding = True`、`_pad_value() = 0.0`、`_noop_output_dtype() = int64` |
| `examples/TileOPs/tileops/kernels/reduction/_primitives.py` | 中（`align_up` / `compute_tile_n` / `device_smem_budget` 工具） | `tn` 选择与预算工具复用（本设计自建元素口径阶梯，工具仅作参考） |
| `testing/npuir/reduction_ops/test_reduce.py` | 中（`reduce` 原语先例） | API 可行性佐证 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(M, N)`（**raw N**，与 Op 层 `_kernel_handles_padding=True` 契约逐元素一致） | float16 / bfloat16 / float32 | 2D 连续行主序；`M`、`N` 均为工厂期常量（无动态轴） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `(M,)` | **int64** | 每行非零计数；`out_idx=[1]` 由 jit 分配返回（dtype 白名单含 int64：`tilelang/jit/jit_npu.py` 的映射表；同族 argmin int64 出参已在 910B3 上跑通） |

### 4.3 中间缓冲区

**路径 R（单 tile；缓冲形状以 `(bm, N)` 表示，路径 C 中 `N → tn` + 独立尾块组 `(bm, tail)`）**：

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | `(bm, tn)` | 输入 dtype | UB（shared） | GM staging + `vcmp` 左操作数（**单消费者**，见 §4.5 纪律） |
| `zero_ub` | `(bm, tn)` | 输入 dtype | UB（shared） | 零常量（任务不变量，循环外 `T.fill` 一次） |
| `mask_eq` | `(bm, tn)` | bool | UB（shared） | 零判决掩码（`vcmp` 输出；不可落 GM） |
| `ind` | `(bm, tn)` | float32 | UB（shared） | 指示平面（0.0/1.0；`vcast` 输出，`reduce_sum` 输入） |
| `zc` | `(bm, 1)` | float32 | fragment | 行零计数（`reduce_sum` 目标；路径 C 中记 `tile_zc`） |
| `acc` / `new_acc` | `(bm, 1)` | float32 | fragment | 路径 C 运行累加（**no-alias**：`new_acc` 为 scratch，回写 `acc`） |
| `zc_flat` / `cnt` | `(bm,)` | float32 | fragment | 形状收缩视图 / 补式结果 |
| `n_v` | `(bm,)` | float32 | fragment | 补式被减数 `≡ N`（任务不变量，循环外 `T.fill`） |
| `out_ub` | `(bm,)` | int64 | UB（shared） | 输出 staging（slice 写 GM） |
| 尾块组（路径 C，`tail > 0` 时） | `(bm, tail)` | 同各平面 | UB（shared） | `x_tail / mask_tail / ind_tail` + `zc_tail`（独立静态宽度，工厂期条件生成） |

**代理 B/elem 记账（逐 buffer 求和，§4.5）**：`x_ub` + `zero_ub`（输入 dtype）+ `mask_eq`（1）+ `ind`（4）
= **9 B/elem（fp16/bf16）/ 13 B/elem（fp32）**；`(bm,1)`/`(bm,)` 类微缓冲由 `UB_SAFETY_RESERVE_BYTES`
（8 KB）级吸收。

### 4.4 内存搬运路径

```
路径 R（纯 Vector 单引擎）：
GM[x] --T.copy(slice,real_m)--> UB[x_ub] --T.vcmp("eq") vs UB[zero_ub]--> UB[mask_eq (bool)]
  --T.vcast(i1->f32, rint)--> UB[ind (f32)] --T.reduce_sum(dim=1)--> FRAG[zc]
  --T.reshape--> FRAG[zc_flat] --T.vsub(n_v,·)--> FRAG[cnt] --T.vcast(rint)--> UB[out_ub (int64)]
  --T.copy(slice)--> GM[out]
无 L1/L0 参与；无 GM 中间往返；无 workspace。

路径 C：GM[x 列块] --T.copy--> UB[x_ub] --(同上链)--> FRAG[tile_zc]
  --T.vadd(acc,tile_zc,new_acc) + T.copy(new_acc,acc)--> FRAG[acc]（逐 tile 循环；acc 全程片上）
  → 静态尾块同链 → 出栈（同上）。
```

### 4.5 UB 内存预算（实测单价 + 双口径并列）

**口径 A：逐 buffer 代理预算（保守，可复核）**：`9 B/elem`（fp16/bf16）/ `13 B/elem`（fp32），
即 `x_ub` + `zero_ub` + `mask_eq` + `ind`；按 `hardware-cost-model.md` §5-② 的**双口径**放大后：

| 放大口径 | fp16/bf16 | fp32 | 说明 |
|---------|-----------|------|------|
| 默认 auto-multi-buffer（~1.7×，`CONST-capacity-910B2C`） | 15.3 B/elem | 22.1 B/elem | 悲观上界（设计**未显式关闭** auto-multi-buffer，故必须给出此档） |
| `--enable-auto-multi-buffer=false`（1.10–1.12×） | 9.9–10.1 B/elem | 14.3–14.6 B/elem | 关闭流水口径 |
| **实测编译器单价（本任务探针，默认 auto-multi-buffer 开）** | **8.00 B/elem** | 8.00 B/elem | `ub overflow, requires 2097152 bits`（32768 元素）= 262144 B ⇒ **精确 8.00 B/elem，与 dtype 无关** |

**容量判据（采用实测口径 C）**：

- **硬上限**：`元素数/block ≤ UB_HARD_ELEMS = 192 KB ÷ 8 = 24576`（实测通过点 fp16 16384 元素、
  fp32/bf16 8192 元素；失败点 fp16 32768 元素）。
- **竞态安全档（本设计采用）**：`E_cap = 8192 元素/block`——把实测单价 8 B/elem 代入家族
  「手工 ≤64 KB/block（×2 双缓冲 ≈ 128 KB ≤ 192 KB − 8 KB reserve）」的竞态安全档
  （row-reduction 家族两任务实测无竞态；保守取舍：按代理预算 9 B/elem 与硬上限口径均可放宽）。
- **纪律（竞态面）**：本设计的每个 shared 缓冲**恰有 1 个消费者**（`x_ub`←`vcmp`、`zero_ub`←`vcmp`、
  `mask_eq`←`vcast`、`ind`←`reduce_sum`），与同族实证的竞态样本（**多消费者**共享缓冲 +
  auto-multi-buffer 预算超限，`TRAP-UB-multibuffer-inflation` / argmin 探针）**形态不同**；
  若 Stage 3 在多波次（persistent `serial` 循环 ≥ 2 任务）测试中出现非确定性差异，
  按 §9.2-R2 的缓解路径（降 `E_cap` 或改 fragment 计算形态）处理。

**逐 workload 预算验证（路径 R；路径 C 见 §1.6.4）**：

| 配置 | 元素数/block | 代理预算（fp16） | 实测单价口径 | 结论（≤ 192 KB） |
|------|-------------|------------------|--------------|-------------------|
| smoke `(32,256)` fp16，bm=8 | 2048 | 18.0 KB | 16 KB | ✅ |
| smoke `(32,256)` fp32，bm=8 | 2048 | 26.0 KB | 16 KB | ✅ |
| `(128,512)` fp32，bm=8 | 4096 | 53.2 KB | 32 KB | ✅ |
| hidden `(2048,4096)` fp16，bm=2 | 8192（= `E_cap`） | 73.7 KB | 64 KB | ✅（同族同配置实测通过） |
| `(4,64,512)` dim=1 → `(2048,64)`，bm=128 | 8192（= `E_cap`） | 73.7 KB | 64 KB | ✅ |
| `(4,64,512)` dim=0 → `(32768,4)`，bm=1024 | 4096 | 36.9 KB | 32 KB | ✅ |
| 路径 C `(32,16384)` fp16，bm=1，tn=8192 | 8192（tile） | 73.7 KB | 64 KB | ✅ |
| 备选 R′ `(32,16384)` fp16，bm=1，tn=N | 16384 | 147 KB | 128 KB | ⏸ 单缓冲（双缓冲超限；§1.6.4） |

**退化路径容量核算（显式补行，`hardware-cost-model.md` §5-①）**：`dim=None` 全张量规约被 Op 层 flatten 成
`(M=1, N=numel)`，`numel` 可远超 `E_cap`——该类负载由**路径 C 统一承接**（`bm=1`，tile 工作集
`tn × 8 B ≤ 64 KB`，与代表 workload 无关）⇒ **回退结论：`M=1` / 大 `numel` 走路径 C，无容量风险**；
`numel > 2²⁴` 由工厂期 `assert` raise（§9.2-R7）。

### 4.6 动态轴定义

无（`M`、`N`、`dtype` 均工厂期静态；kernel 内全部循环边界静态，运行时仅 pid/task guard 的动态性）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")   # out 自动分配 (M,) int64
def _func(block_m): ...                       # K9: threads 去除；block_m 由 wrapper config 传入
```

工厂缓存：沿用 harness `LogicalReduceKernel` 的 `(M, N)` 键缓存 + 源 `lru_cache` 语义（§0.4-#7 保留）。
环境：`TILELANG_ASCEND_MODE=Developer`（wrapper `ascend_mode` 已声明）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector。

**判定依据**：全部计算为逐元素比较/类型转换/行归约/微元算术（`vcmp`、`vcast`、`reduce_sum`、`vsub`、`vadd`），
无 matmul；数据通路 GM↔UB↔fragment；无 Cube/L0 参与。

### 5.2 Block 划分（元素预算阶梯 + 路径切换 + `tn`）

```python
# 工厂期（全部静态 Python 量）；口径对照 §4.5
E_CAP        = 8192     # 元素/block（竞态安全档）
UB_HARD_ELEMS = 24576   # 192 KB / 8 B（探针实测单价）
# ── 路径切换（与 §3.3 同序）──
if N <= E_CAP:                                     # 路径 R
    block_m = max([p for p in (1, 2, 4, 8) if p * N <= E_CAP] or [1])          # 基础阶梯
    if ceildiv(M, block_m) > 80:                                              # 发射项驱动 → 扩展阶梯
        block_m = max([p for p in (16, 32, 64, 128, 256, 512, 1024)
                       if p * N <= E_CAP and p <= M] + [block_m])
    tile_n, num_full, tail = N, 1, 0
else:                                              # 路径 C
    block_m = 1
    cap = min(E_CAP, N)
    tile_n = max_divisor_pref(N, cap, align=256)   # ≤cap 的最大 256 倍数且整除 N；无整除因子 → 最大 256 倍数
    num_full, tail = N // tile_n, N - (N // tile_n) * tile_n
block_m = min(block_m, M)                          # 行数不足不超配（垃圾行污染）
num_logical = ceildiv(M, block_m)
num_kernels = min(num_logical, 40)                 # 物理核数 = 实查 aivector = 40（§5.5）
num_local_tasks = ceildiv(num_logical, num_kernels)  # 核内串行上界（静态值）
```

**`block_m` 选择理由**：阶梯内取预算最大值——行块越大，每 block 的固定开销（载入 + 出栈 ≈ 4 op）
摊得越薄（`CONST-vector-launch-overhead` 恒为每 op 固定成本）；受 `E_cap` 硬约束。
基础阶梯 `{1,2,4,8}`（源阶梯 + 家族先例）覆盖常规行宽；**小 `N` 大 `M` 负载**（如 `dim=0` 的
`(32768,4)`、`(2048,64)`）由发射项门控（`num_logical > 80` ⇒ 双波以上）启用扩展阶梯至 1024
（档位由同一预算过滤器自动封顶）。

**逐 workload 参数表**（M/N 由 manifest + Op 层 `_prepare_input` 的 movedim+reshape 推导并附算式）：

| workload（来源） | dtype | 路径 | block_m | tile_n | 元素数/block | num_logical | num_kernels | 核内任务（静态） | 依据/推导 |
|------------------|-------|------|---------|--------|--------------|-------------|-------------|------------------|-----------|
| smoke `[32,256] dim=-1` | fp16/bf16/fp32 | R | **8** | 256 | 2048 | 4 | 4 | 1 | `p·256 ≤ 8192 → p ≤ 32`；阶梯内最大 8；4 ≤ 80 不触发扩展 |
| sparsity-hidden `[2048,4096] dim=-1` | fp16/bf16 | R | **2** | 4096 | 8192 | 1024 | **40** | **26** | `p·4096 ≤ 8192 → p ≤ 2`；1024 > 80 但扩展档受预算封顶 → 2 |
| sparsity-seq `[32,16384] dim=-1` | fp16 | **C** | 1 | **8192** | 8192（tile） | 32 | 32 | 1 | `16384 > 8192` → 路径 C；`tn` = min(8192,16384) 内最大 256 倍数且整除 16384 = 8192 ⇒ `num_full=2, tail=0` |
| 3d-multidim `[4,128,4096] dim=[0,2]` | fp16 | **C** | 1 | **8192** | 8192（tile） | 128 | **40** | **4** | N = 4×4096 = 16384，M = 128（Op 层 movedim 后 reshape）；同 sparsity-seq 的 `tn` |
| 测试 `(128,512)`（basic/noncontig） | fp16/bf16/fp32 | R | 8 | 512 | 4096 | 16 | 16 | 1 | `p·512 ≤ 8192 → p ≤ 16`；阶梯内 8 |
| 测试 `(256,4096)` | fp16/bf16 | R | 2 | 4096 | 8192 | 128 | 40 | 4 | 同 sparsity-hidden |
| 测试 `(128,300)` | fp16/fp32 | R | 8 | 300 | 2400 | 16 | 16 | 1 | raw N=300（**不 pad**，§0.6-R3） |
| 测试 `(129,512)`（M 尾块） | fp16 | R | 8 | 512 | 4096 | **17** | 17 | 1 | `129 % 8 = 1` ⇒ 尾块 `real_m=1` |
| 测试 `(64,512) dim=0` → `M=512, N=64` | fp16 | R | 8 | 64 | 512 | 64 | 40 | 2 | `p·64 ≤ 8192 → p ≤ 128`；64 ≤ 80 不触发扩展 |
| 测试 `(4,64,512) dim=1 / dim=-2` → `M=2048, N=64` | fp16 | R | **128** | 64 | 8192 | 16 | 16 | 1 | 基础 bm=8 → 256 > 80 ⇒ 扩展阶梯封顶 128 |
| 测试 `(4,64,512) dim=0` → `M=32768, N=4` | fp16 | R | **1024** | 4 | 4096 | 32 | 32 | 1 | 基础 bm=8 → 4096 > 80 ⇒ 扩展封顶 1024（`p·4 ≤ 8192`） |
| 测试 `(512,)`（1D） | fp16/bf16/fp32 | R | 1 | 512 | 512 | 1 | 1 | 1 | `M=1`（Op 层 flatten）⇒ `bm = min(8, 1) = 1` |

**`tn` 推导链（路径 C，冻结值复现）**：`cap = min(8192, N)`；候选集 = `{t : t ≤ cap, t ≡ 0 (mod 256),
t | N}`；取最大者；空集则取 `≤cap` 的最大 256 倍数（此时 `tail = N % tn ≠ 0`，走静态尾块）。
`N=16384` → `8192 | 16384` ⇒ **tn=8192、num_full=2、tail=0**（零尾块）。

### 5.3 约束分析

- **对齐约束**：`T.copy` 的 slice 形态天然满足 32 B 尾轴要求；**无需任何宽度对齐 pad**（探针实测
  raw N = 64 / 4 / 2 均可编译；logsumexp N=300 已迁移先例同向）。
- **UB 容量**：§4.5 逐配置验证（代理预算 / 实测量价 / 硬上限三层口径 + 退化路径补行）。
- **L0/L1 容量**：不适用（纯 Vector）。
- **编译器陷阱约束**：判决走显式向量链（规避 2D `ite` 融合标量化）；累加走 no-alias
  （规避原位 select/累加丢失 loop-carried 状态）；`vcmp("eq")` 而非 `vcmp("ne")`（`eq` 为 IEEE 正确）；
  `T.fill` 值须类型化常量；`(bm,1)→(bm,)` 收缩经 `T.reshape`。
- **`reduce` dtype**：全程 f32（同 dtype），bf16 不进入归约域。

### 5.4 注意事项（非整除/边界处理策略）

- **M 非整除（`M % block_m ≠ 0`）**：`real_m = T.min(block_m, M − start)` + src/dst 双显式 slice +
  垃圾行不写出（§0.6-R6 三件套；VP-2026-0044（pending）/ logsumexp 先例）。
- **N 非整除（256 对齐）**：**不处理**——raw N 实列域计数（§0.6-R3；E3 机器验证 pad 域 ≡ 实列域）。
- **`tn` 非整除（路径 C 尾块）**：`tail = N − num_full·tn` 为工厂期静态常量，走**独立静态宽度的代码块**
  （独立缓冲组，非 kernel 侧掩码）；**不用 `T.if_then_else` 掩码**（logsumexp tiled 的 serial 变量条件
  有 segfault 反例，见 argmin 设计的 §9.1）。
- **`N=1`**：单列判决 → `y ∈ {0,1}`；`(1,1)` 用例由 verify_equiv 与 §8.2 L1 覆盖。
- **`M=1` / `dim=None` flatten**：`bm=min(bm,1)=1`；大 `numel` 走路径 C（§4.5 退化路径补行）。
- **`M=0`**：grid=0 边界（§9.2-R6 风险记录；harness 测试不覆盖，Op 层短路建议）。
- **`N > 2²⁴`**：工厂期 `assert` raise（§9.2-R7；fp32 计数精确域外）。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

- **① 逻辑核数**：`num_logical_kernels = ceil(M / block_m)`（纯行并行，无 N 向跨核分核；
  路径 C 的列分块在**核内**（`T.serial` tile 循环），不增加逻辑核数）。
- **② 物理核数（实查，禁止假设值）**：
  `from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()` → **20**（2026-10-16 实查）；
  **纯 Vector 算子核数**由 `NPUUtils.get().get_aivector_core_num()` 实查 = **40**（= 20 × 2，
  与 `AscendArch().aivector_core_num` 互证）；`AscendArch().ub_cap` = **196608 B（192 KB）** 同源实查。
  查询代码与返回值即本节的实查记录（`chip_name` = `Ascend910B`；同族 argmin 任务在 910B3 上得同值，
  跨任务一致）。
- **③ 规模判定与分核方案**（统一 persistent 结构：`num_kernels = min(num_logical, 40)`；
  核内 `T.serial(静态上界 = ceil(num_logical / num_kernels))` + grid-stride 任务映射
  `task = cid + s·num_kernels` + 动态 guard `if start < M`）：

| workload | `num_logical` | 判定 | `num_kernels` | 核内任务（静态上界） | 依据 |
|----------|---------------|------|---------------|----------------------|------|
| smoke (32,256) | 4（bm=8） | **逻辑 ≤ 物理（4 ≤ 40）** | 4 | 1 | 无需适配：行块独立、单波、每核 ≤1 任务 |
| sparsity-hidden (2048,4096) | 1024（bm=2） | **极大规模（1024 ≫ 40，bm 受 `E_cap` 不可再增）** | **40** | **26** | 固定 40 核 + 核内 `T.serial`（静态边界）；摊薄核启动开销 |
| sparsity-seq (32,16384) | 32（bm=1，路径 C） | **逻辑 ≤ 物理（32 ≤ 40）** | 32 | 1 | 无需适配；核利用率 32/40 = 80%（R3 已论证 N-split 跨核无收益） |
| 3d-multidim (128,16384) | 128（bm=1，路径 C） | **极大规模（128 > 40）** | **40** | **4** | 同 sparsity-hidden |
| 测试 (129,512) | 17（bm=8） | 逻辑 ≤ 物理 | 17 | 1 | 尾块行 1 行 |
| 测试 (2048,64) | 16（bm=**128**，扩展阶梯） | 逻辑 ≤ 物理 | 16 | 1 | 扩展阶梯把核数从 256 降到 16 |
| 测试 (32768,4) | 32（bm=**1024**，扩展阶梯） | 逻辑 ≤ 物理 | 32 | 1 | 扩展阶梯把核数从 4096 降到 32 |

> 核内串行循环边界**全部为静态值**（工厂期 Python 整数：`ceildiv(num_logical, num_kernels)`）；
> 动态性仅存在于 guard（`if start < M`，`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 先例形态）。
> wrapper `default_config` 须按 §5.2 公式适配（其 GPU 期 48 KB SMEM 启发式弃用，见 §5.6）。

**核数复算（供检视复算，逐负载；`num_logical = ceil(M / block_m)`）**：

- smoke：`M=32, bm=8` → **4**（≤ 40 ⇒ `num_kernels=4`，单波）
- sparsity-hidden：`M=2048, bm=2` → **1024**（> 40 ⇒ 40 核，核内上界 = `ceil(1024/40)` = **26**）
- sparsity-seq：`M=32, bm=1` → **32**（≤ 40 ⇒ 32 核，每核 1 任务 × 2 tile）
- 3d-multidim：`M=128, bm=1` → **128**（> 40 ⇒ 40 核，核内上界 = **4**）
- `(4,64,512) dim=0`：`M=32768, bm=1024` → **32**（扩展阶梯；否则 bm=8 → 4096 核）

### 5.6 集成前置核对（实读目标工件）

| 核对项 | 结论 | 实现期动作 |
|--------|------|-----------|
| 目标 Op 层 `_kernel_handles_padding` | **`True`**（`examples/TileOPs/tileops/ops/reduction/count_nonzero.py` 类属性） | **保持**；kernel 必须声明 raw `(M, N)`（宽度错位症状：行 0 正确、行 ≥1 全错且不报错） |
| 目标 Op 层 `_pad_value` | `0.0`（`count_nonzero.py` 的 `_pad_value()`）——计数单位元，方向中性 | 保持（本设计不使用该值，kernel 域内不 pad） |
| Op 层 dtype 预转 | `_pre_kernel` 对 bool/int32/int64/complex 调 `to_logical_float32` | 保持；kernel dtype 路径矩阵按 {fp16, bf16, fp32} 三路（§3.2） |
| wrapper `default_config` | 现为 GPU 期启发式（`smem_budget // (N_padded × elem)`）→ NPU 侧语义不适用 | **改为转调本设计 §5.2 的 `_select_config(M, N, dtype)`**（kernel 模块为 tiling 单一事实源） |
| wrapper 空转/约束提示 | docstring 称「callers must keep the padded reduction width within the single-tile column budget」 | 更新为「`N ≤ 2²⁴`；`N > E_cap` 自动走列分块路径」（§5.2/§5.4） |
| manifest `sparsity-seq` 宽度 | 已由 32768 下调为 16384（Stage 0 决策：tiled 变体不迁） | 与 §5.2 一致：16384 由路径 C（tn=8192）承接，无需 tiled 工厂 |
| msprof 锚点 | 现无 `msprof_kernel_name` | 增加（路径 R → `count_nonzero_resident`，路径 C → `count_nonzero_tiled`），避免 msprof 捕获 Op 层 prep 算子 |

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| M 方向（block 级） | persistent 任务循环 | `T.serial(num_local_tasks)`（静态上界 + grid-stride + guard） | 极大规模核内串行（§5.5）；摊薄核启动开销 |
| N 方向（路径 C） | tile 迭代 | `T.serial(num_full)`（静态上界） | 在线累加的循环携带状态（§1.6.2-#10）；静态边界要求 |
| 尾块（路径 C） | 无循环（独立静态宽度代码块） | — | `tail` 为工厂期常量；避免 serial 变量条件（segfault 反例） |
| 元素级判决 | 向量化 | `T.vcmp(..., "eq")` | 显式向量链（§1.6.3-#5） |
| 元素级指示化 | 向量化 | `T.vcast(mask, ind, round_mode="rint")` | i1→f32 平面 |
| 元素级归约 | 向量化归约 | `T.reduce_sum(ind, zc, dim=1)` | 原语（§3.2） |
| 分段累加（路径 C） | 向量化微元 | `T.vadd(acc, tile_zc, new_acc)` + `T.copy(new_acc, acc)` | no-alias 纪律 |
| 补式 / 出栈 | 向量化微元 | `T.vsub(n_v, zc_flat, cnt)`；`T.reshape` + `T.vcast(rint)` | `(bm,)` 级 |
| 尾块行数 / 任务 guard | 标量 | `T.min` + `if start < M` | block 级动态边界（§1.6.2-#9） |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (cid, _):
    # 任务不变操作数（循环外）：zero_ub（T.fill）、n_v（T.fill）
    for s in T.serial(num_local_tasks):                  # 静态上界
        start = (cid + s * num_kernels) * block_m
        if start < M:                                    # 动态 guard
            real_m = T.min(block_m, M - start)
            if path == "R":                              # 单 tile
                [§3.3 _tile_body(c0=0)] → zc
            else:                                        # 列分块 + 运行累加
                T.fill(acc, 0.0)
                for t in T.serial(num_full):             # 静态上界
                    [§3.3 _tile_body(c0=t*tile_n)] → zc
                    T.vadd(acc, zc, new_acc); T.copy(new_acc, acc)
                if tail > 0:                             # 静态尾块（工厂期条件生成）
                    [§3.3 _tile_body_tail(c0=num_full*tile_n, w=tail)] → zc
                    T.vadd(acc, zc, new_acc); T.copy(new_acc, acc)
            [§3.3 _epilogue：reshape → vsub(补式) → vcast(int64) → slice copy]
```

### 6.3 流水线优化

**初始版本不使用 `T.Pipelined`**（Developer 模式自动施加 auto-multi-buffer；同族已在任务/tile 循环上实测其
正确性与竞态边界，§4.5 纪律）。**Stage 4 候选（按 D-2 发射项分析排序）**：
① **§1.6.4 备选 R′（超预算单缓冲驻留）A/B 裁决**（判据与阈值已冻结）；
② 判决与指示化的 **plain 2D `T.Parallel` 形态**（2026-10-15 适用域收窄记录「plain 逐元素 2D loop
在 `bm≥2` 正常向量化」）——若成立可省 1 个显式链 op，需 §8.2 L0-2 复验向量化（前置探针）；
③ **列分块粒度扫描**（路径 C 的 `tn` ∈ 4096/8192 与 `bm>1` 组合，探针已证 `bm>1` 多 tile 可编译）；
④ `E_cap` 放宽（8192 → 12288/16384，按实测单价口径与竞态复验）；
⑤ persistent 任务循环深度/顺序调优（grid-stride 步长、按核连续性）。

### 6.4 尾块处理

见 §5.4（M 尾块：`real_m` + 双 slice + 垃圾行丢弃；N 向：**实列域无 pad**；路径 C 尾块：静态宽度独立代码块；
guard 静态化边界）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer 模式编译器管理 UB/fragment 依赖与 auto-multi-buffer 流水同步）。

### 7.2 同步点说明

无手动同步点（单 kernel、无核间协作、无跨引擎传输、无 workspace）。
persistent 任务循环与 tile 循环内的生产者-消费者依赖由编译器自动插入；前提为 §4.5 的**单消费者纪律**
（每个 shared 缓冲恰 1 个消费者）与 `E_cap` 预算纪律。

### 7.3 pass_configs 配置

无特殊 pass 配置（Developer 默认；`TILELANG_ASCEND_MODE=Developer`）。
Stage 4 探索项：`npuir.enable_auto_multi_buffer=False`（§1.6.4 备选 R′ 与 §4.5 口径 B 的显式关闭形态）。

### 7.4 CV 融合设计

**不适用**：本算子为纯 Vector（无 Cube 段、无 L0C→UB 直连需求、无核间流水），
故不涉及 `Scope("Cube")`/`Scope("Vector")` 与 `T.sync_block_set/wait`。

---

## 8. 验证方案

### 8.1 Golden 函数

> golden 以 §0.1 语义为唯一依据（`torch.count_nonzero(..., dim) → int64`），**不复刻 §0.6 的 NPU 算法**
> （无补式、无指示平面、无列分块），保证验证独立性；且**不依赖 torch_npu 参考后端**
> （`traps-runtime.md` 的 `TRAP-npuir-ref-backend-golden-capability`：参考后端自身有选择类能力缺口
> ——本算子为计数类，回退 CPU golden 的路径与该条目的处置形态一致；harness 侧 ref 为
> device 上的 `torch.count_nonzero`，本设计另提供 CPU golden 作独立基准）。

```python
def golden_count_nonzero(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """PyTorch 参考实现（CPU）：count_nonzero 语义。

    输入为 kernel 级契约的 (M, N)（op_kind = count_nonzero 只需 dim=-1 形态）；
    任意 rank / 多 dim 归约由 Op 层 movedim+reshape 承接，属 Op 层测试范围（§8.2 L1）。
    NaN 计入非零、±0 不计入 —— 即 torch.count_nonzero 的既定语义。
    """
    return torch.count_nonzero(x, dim=dim).to(torch.int64)
```

### 8.2 精度标准

**精确比对（无容差）**：`torch.equal(out, ref)` 且 `out.dtype == torch.int64`，即 `atol = rtol = 0`
（与 harness `examples/TileOPs/tests/ops/test_count_nonzero.py` 的 `_exact_compare_int64` 同款）。
**数学依据**：E1–E4 均为**整数恒等**（`verify_equiv.py` 265096 行 0 偏差）；计数是整数，任何容差放松
都会掩盖判决式（±0/NaN）、补式或出栈类缺陷。

**分层精度计划**（L0 阻塞门禁 / L1 功能覆盖 / L2 记录不阻断；完整套件交 `tilelang-op-develop` 按本节展开）：

| 层 | # | 用例 | 覆盖点 |
|----|---|------|--------|
| **L0** | L0-1 | `(32,256)` fp16 / bf16 / fp32，bm=8 | smoke 负载 ×3 dtype（比较域三路分派） |
| **L0** | L0-2 | `(2048,4096)` fp16 / bf16，bm=2 | sparsity-hidden + persistent 40×26 **多波次**（竞态回归，§4.5-R2） |
| **L0** | L0-3 | `(32,16384)` fp16，路径 C tn=8192（§5.2 冻结值） | sparsity-seq + 列分块在线累加（2 tile） |
| **L0** | L0-4 | `(128,16384)` fp16，路径 C tn=8192 | 3d-multidim（`dim=[0,2]` 等价形态）+ persistent 40×4 |
| **L0** | L0-5 | `(128,300)` fp16 / fp32 | **raw 非对齐 N（不 pad 路线）** |
| **L0** | L0-6 | `(129,512)` fp16 | M 尾块（`real_m=1`）+ 源缺陷修正复验（§0.4 源缺陷） |
| **L0** | L0-7 | `(256,4096)` fp16 | `E_cap` 边界配置（8192 元素/block） |
| **L0** | L0-8 | `(4,64,512) dim=0` → `(32768,4)`；`dim=1` → `(2048,64)` fp16 | **窄行 raw 宽度**（元素宽 8 B / 128 B）+ 扩展阶梯（bm=1024 / 128） |
| **L0** | L0-9 | `(512,)` 1D → M=1；`(1,N)` 单行 | 单行负载 + `bm=min(bm,1)` |
| **L0** | L0-10 | 构造行：全零行（y=0）/ 全非零行（y=N）/ 混合符号零 / 首末单点非零 / `N=1`、`N=2` | 补式边界与计数上界（harness workload 亦强制 `x[0]=0`、`x[1]=1`，天然覆盖两端） |
| **L0** | L0-11 | 集成宽度一致性：断言 kernel 声明宽度 == Op 传入宽度（`(M, N)` raw），`(128,300)` 用例上无行 ≥1 错位 | 契约一致性回归（VP-2026-0128 类错位缺陷） |
| **L0** | L0-12 | **NaN 行**（含 NaN + 零 + 非零混合）比对 `torch.count_nonzero` | **E1 等价性前提的机器门禁**：若 `vcmp("eq")` 实测非 IEEE，则 §1.6.1-E1 论证前提被推翻 → 走设计修订路径（不得放松断言） |
| **L1** | L1-1 | `(128,512)` fp16/bf16/fp32 + 非连续输入切片 | basic + noncontig 功能覆盖 |
| **L1** | L1-2 | `(2,64,512)` dim=-1 / `(2,4,8,512)` dim=-1 → Op 层路径 | 3D/4D 高阶形态（Op 层 movedim+reshape） |
| **L1** | L1-3 | `dim=0 / dim=1 / dim=-2` 的 `(64,512)` / `(4,64,512)` 组合 | 多 dim 语义 + Op 层 reshape 回写 |
| **L1** | L1-4 | 亚正规行（最小亚正规 fp16/fp32）/ ±inf 行 / 最大有限行 | 比较谓词的边界值（E4 角点集的 NPU 侧复验） |
| **L1** | L1-5 | dtype smoke：bool / int32 / int64 / complex64 经 Op 层预转 | Op 层 `to_logical_float32` 路径 + **int64 输出契约断言** |
| **L1** | L1-6 | `(8,300)` fp16 多行（M 尾块 + 非对齐 N 同时出现） | 两类尾块耦合 |
| **L2** | L2-1 | `M=0` / `N=0`（工厂期 raise 行为） | 记录（不阻断） |
| **L2** | L2-2 | `N > 2²⁴` 的 assert 行为 / `(1,1)` 极小 / `(tile_n+1)` 跨分界 | 记录（不阻断） |
| **L2** | L2-3 | 路径 R ↔ 路径 C 输出的逐位一致性（同数据强制两条路径） | 记录（不阻断；E3b 已证分段恒等） |

### 8.3 等价性验证工件

`verify_equiv.py`（随算子目录交付，已执行 **ALL EQUIV_PASS**，exit 0；§1.6.1 内嵌结果表）——
含 1 组**判别力对照**（有序-NE 故意错误候选必须被报出差异，否则脚本自身判 `CONTROL_BROKEN` 失败）
与 2 组边界确认（计数 > 2²⁴ 偏差）。Stage 2 检视维度 8 的重跑复核对象。

### 8.4 设计期探针工件

`history_version/design_probe_ub.py`（指示链 UB 单价 8.00 B/elem + 窄行 raw 宽度可编译性）、
`history_version/design_probe_tileloop.py`（`bm>1` 多 tile 可编译性；`CG-2026-0014` 不复现）——
供 Stage 2/3/4 复算与版本戳复核（§9.4）。

---

## 9. 风险点与注意事项

### 9.1 已知约束（工具链级；**逐条标注工具链戳与适用域**）

> 本任务工具链：tilelang `0.1.2+b39cc30679662a13dd911768a51c7eede45d2892` + **CANN 9.0.0** +
> torch 2.9.0+cpu + **Ascend910B3（AIV=40 / UB 196608 B）**。
> 标注「910B2C / CANN 8.5.0」的条目为跨工具链引用 ⇒ **待重验**；标注「本任务探针」者为同工具链实测。

| # | 约束 | 实证来源（工具链戳） | 设计内规避 |
|---|------|---------------------|-----------|
| C-1 | `CONST-reduce-ub-per-element` 的归约链 UB 单价（3-D 整平面形态最高 ~128 B/elem）**不适用于本 2-D 行归约形态** | 该条目绑定 `(CH,1,H*W)` + `reduce dim=2` 形态（910B3 / CANN 9.0.0）；**本任务探针**同工具链实测本链 = 8.00 B/elem | §4.5 采用实测单价；该条目的适用域限定写入 §9.4 回填候选 |
| C-2 | `T.vcast` **f32→f32 非恒等**（退化为四舍五入到整数） | `traps-compiler.md` 的 `TRAP-vcast-same-dtype-rounds`（910B3 / CANN 9.0.0 同工具链） | §3.2：仅用 `i1→f32` 与 `f32→i64` 跨 dtype cast，无同 dtype cast |
| C-3 | `T.vcmp("ne")` 是 **ordered** 不等（NaN 恒假）；`eq`/`le`/`ge` 为 IEEE 正确 | `traps-compiler.md` 的 `TRAP-vcmp-ne-ordered-nan`（verified） | §0.6-R1：判决走 `"eq"` 补式；§8.2 L0-12 门禁 |
| C-4 | 2D `T.Parallel` **融合 `ite`** 在 `bm≥2` 整体标量化；**2026-10-15 适用域收窄**：plain 逐元素 2D loop（单向量 op/算术/cast）在 `bm≥2` 正常向量化 | `traps-compiler.md` 的 `TRAP-parallel2d-bm-ge2-scalarize`（910B2C / CANN 8.5.0 首证；收窄记录见 `INDEX.md`） | §1.6.3-#5 显式向量链；plain 2D 形态登记为 Stage 4 候选（§6.3-②） |
| C-5 | 原位 `T.vselect`/累加在多迭代 `T.serial` 循环**丢失 loop-carried 状态** | `traps-compiler.md` 的 `TRAP-vselect-inplace-carried-state`（910B2C / CANN 8.5.0） | §1.6.2-#7：no-alias 累加 + `T.copy` 回写 |
| C-6 | `T.fill` 的值须为**类型化 PrimExpr**（Python 字面量 `0` 被前端拒绝） | **本任务探针**实测错误文本 `value must be a tir.PrimExpr or tir.BufferLoad`（同工具链） | §3.2：`tir.const(0, dtype)` / `T.float32(N)` |
| C-7 | 多 tile + `bm>1` 编译器 SIGSEGV（`CG-2026-0014`，910B2C / CANN 8.5.0） | **本任务探针**：同工具链 fp16 `2×4096`（4 tile）/`8×512`（32 tile）/fp32 `2×1024`（16 tile）**均编译通过** ⇒ 该条在本工具链**不复现** | 路径 C 允许 `bm>1`（本设计仍取 bm=1，见 §1.6.3-#4）；版本戳更新诉求写入 §9.4 |
| C-8 | auto-multi-buffer 预算超限时**静默放弃双缓冲** → **多消费者**共享缓冲竞态 | `traps-compiler.md` 的 `TRAP-UB-multibuffer-inflation` / 同族 argmin 探针（910B2C / CANN 8.5.0；多消费者形态） | §4.5：单消费者纪律 + `E_cap` 预算；§1.6.4 处置超预算档 |
| C-9 | `T.reduce` dtype 表：bf16 ✗、bool/int1 ✗；**混合 dtype reduce 静默错误** | `docs/Tilelang.language/规约操作/T.reduce.md` §2.2.1；queue VP-2026-0119 派生条目（待合入） | §3.2：归约域恒 f32、同 dtype（bf16 仅在比较域） |
| C-10 | `reduce` 的 `dims=0` 在长条 `(K,4)` 上逐列串行（灾难性慢）；`dims=1`（行向）为快路径 | `constants.md` 的 `CONST-reduce-dims0-skinny`（910B2C / CANN 8.5.0） | 本设计恒用 `dim=1`；宽度 4 的窄行（`(32768,4)`）另见 §9.2-R4 |
| C-11 | `T.arange` 的 `(M,W)` 物化按标量执行；`vbrc` 的 region dst 被静默丢弃；`vcmp` 的 bool 结果不可落 GM | `constants.md` 的 `CONST-arange-scalar-materialize` + `traps-compiler.md` 的 `TRAP-npuir-vbrc-region-dropped` + `docs/Tilelang.language/比较操作/T.vcmp.md` §2.3 | 本设计**不使用 `arange`**（无索引需求）与 `vbrc`（零常量用 `T.fill`）；bool 平面仅片上消费 |
| C-12 | 运行时 `scf.for` 循环变量作共享缓冲行偏移 → 静默错值 | `traps-compiler.md` 的 `TRAP-scf-for-row-offset-silent-miscompile`（verified） | 本设计的缓冲访问不用循环变量作行偏移（行偏移只由工厂期 `block_m × task` 与 `real_m` slice 构成，循环变量只作**列**块基址且参与 `T.copy` 的显式 slice） |

### 9.2 风险点

| # | 风险 | 影响 | 缓解 |
|---|------|------|------|
| R1 | **`vcmp("eq")` 非 IEEE 语义**（若实测与文档/同族证据不符）——这是 E1 等价性论证的前提 | NaN 行计数偏离 torch（少计） | §8.2 L0-12 机器门禁；失效则走设计修订路径（补式前提被推翻，需另寻 NaN 处理形态，不得放松断言） |
| R2 | **单消费者超预算竞态**（§4.5 纪律的形态依赖）：同族竞态实证均为**多消费者**形态，本设计的单消费者形态**无实测证据** | 非确定性错值（多波次下） | `E_cap` 保守档（8192 元素）+ L0-2/L0-4 **多波次**回归；失效时缓解路径：降到 `E_cap=4096` 或把计算改为 fragment 形态（同族竞态规避形态） |
| R3 | 亚正规 flush-to-zero（若比较路径对亚正规做 FTZ，则亚正规被误判为零） | 亚正规元素计数偏离 | 比较运算**通常不做 FTZ**（未文档化假设 + 估算依据：IEEE 比较为精确谓词）；L1-4 亚正规行实测记录；失效则按 §9.2-R1 同路径处理 |
| R4 | **窄行（`N=4`/`N=2`）性能退化**：行宽 8 B / 4 B 远低于向量宽，逐 op 效率低（`CONST-reduce-dims0-skinny` 同族现象） | `(4,64,512) dim=0` 类测试形态延迟偏高（非 manifest 基准） | 扩展阶梯把核数降到 32（bm=1024）；这是测试形态而非性能目标形态，接受 |
| R5 | `E_cap` / UB 单价绑定**本工具链**（探针同工具链实测，但**编译器版本升级后须复验**） | 配置可能过保守或不足 | §9.4 版本戳诉求；Stage 3 首跑以 L0-7/L0-2 暴露；`kb_stale_check.py` 检测 |
| R6 | **`M=0`**（非规约维含 0）：grid=0 边界未验证 | 可能启动异常或空输出 | Op 层短路建议（`forward` 加 `if M == 0: return torch.empty(0, dtype=torch.int64)`）；L2-1 记录 |
| R7 | **`N > 2²⁴`**（fp32 计数精确域外；`dim=None` 全张量 numel 超 16.7 M 时触发） | 静默计数失精（±1 量级） | 工厂期 `assert N <= 2**24` raise（§3.3）；备选 C4（int32 累加，精确域 2³¹）规格已记（§1.6.0-R1 表 C4） |
| R8 | 主要性能常数（MTE2 曲线 / 发射单价上界）绑定 910B2C / CANN 8.5.0 | D-2 估算可能系统性偏差（~2× 档） | §1.6.0 已标注「未实测假设」；Stage 4 按 `hardware-cost-model.md` §3 回填对账并回写 `constants.md` |
| R9 | 跨工具链陷阱（C-4/C-5/C-8/C-10）在 CANN 9.0.0 / 910B3 上**未逐条复验** | 规避形态可能过保守或不足 | §9.1 逐条标注；Stage 3 首跑暴露；异常时按 `[DESIGN_LIMIT]` 受控修订 |
| R10 | 路径 C 的 `tail` 为任意静态宽度（如 44）时**向量化效率低**（尾部窄块）+ 尾块代码路径分支 | 非 256 倍数 N 且超 `E_cap` 时延迟偏高 | manifest 无此形态（N ∈ {256,4096,16384} 全为 256 倍数）；L1-6 覆盖；可改用「尾块向上取整到 256 并补 0」的备选（E3b 已证 pad 中性） |

### 9.3 特殊场景处理

非整除分块（§5.4）；极小 shape（`(1,1)` / `N=1` / `N=2`，verify_equiv 角点 + L2 覆盖）；
混合精度（per-dtype 路径矩阵 §3.2）；`dim=None` 全张量（Op 层 flatten → M=1，§4.5 退化路径补行 + 路径 C）；
`dim≠-1`（Op 层 movedim 到末轴后进入同一 kernel）；NaN / ±0 / 亚正规 / ±inf（§1.6.1 角点集 + L0-12 / L1-4）；
多波次 persistent（L0-2 / L0-4 竞态回归）。

### 9.4 能力缺口与常数回填诉求（Compiler Capability Gap Rule）

本轮设计**未发现新的能力缺口**（选定的全部 API 均在本工具链编译通过——两个设计期探针为直接证据）。
需回填/复核的既有条目如下（**不在本阶段改写全局工件，交由 `tilelang-skill-evolver` 按治理分级处理**；
同时在 `RETROSPECTIVE.md` 提交 D 类价值点三件套）：

| 既有条目 | 本任务实测 | 诉求 |
|----------|-----------|------|
| `CG-2026-0014`（多 tile + `bm>1` 编译器 SIGSEGV；910B2C / CANN 8.5.0） | **不复现**：fp16 `2×4096`(4 tile) / `8×512`(32 tile) / fp32 `2×1024`(16 tile) 全部编译通过（tilelang b39cc306 + CANN 9.0.0 + 910B3） | **版本戳更新**：该触发条件在新工具链已消失（证据：`history_version/design_probe_tileloop.py`）；建议标注 `status: re-verified-not-reproduced` 或收窄适用域 |
| `CONST-reduce-ub-per-element`（归约链 UB 单价，整平面形态 ~128 B/elem；910B3 / CANN 9.0.0） | 本 2-D 行归约形态实测 **8.00 B/elem**（同设备同 CANN） | **适用域限定**：该条目单价绑定 `(CH,1,H*W)` + `dim=2` 形态；建议补一行「2-D `(bm,tn)` 行归约（`dim=1`）单价 = 8 B/elem（与 dtype 无关）」 |
| `CONST-vector-launch-overhead`（~0.5 µs/op；910B2C / CANN 8.5.0） | 未实测（Stage 4 标定） | 待 `§1.6.4` 裁决回填 910B3 数据点 |
| `TRAP-UB-multibuffer-inflation`（多消费者竞态） | 本设计为单消费者形态（未触发） | 建议补记「单消费者形态在 8 B/elem × 16384 元素（128 KB）档仍可编译」的数据点（探针证据） |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/count_nonzero/_logical_reduce_kernel/
├── DESIGN.md                          # 本设计文档
├── verify_equiv.py                    # D-1 等价性机器验证（已执行 ALL EQUIV_PASS，exit 0）
├── RETROSPECTIVE.md                   # Stage 1 复盘（本次产出）
├── history_version/
│   ├── design_probe_ub.py             # D-3 探针 #1：指示链 UB 单价 + 窄行 raw 宽度
│   └── design_probe_tileloop.py       # D-3 探针 #2：bm>1 多 tile 可编译性
└── _logical_reduce_kernel.py          # ⬜ Stage 3：算子实现（kernel + golden + L0 分层测试）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成（v0，2026-10-16） | 本文档（含 §0 迁移分析 + §1.6 调研与优化 + §5 分核三要素 + §1.6.4 实验裁决三件套） |
| `verify_equiv.py` | ✅ 已执行（exit 0） | E1 / E1c(control) / E2 / E2-boundary / E3 / E3b / E4 全组通过；门禁等价组 265096 行 0 违反 |
| `history_version/design_probe_ub.py` | ✅ 已执行 | 指示链 UB 单价 = 8.00 B/elem；硬上限 24576 元素；窄行 raw 宽度（N=64/4/2）可编译 |
| `history_version/design_probe_tileloop.py` | ✅ 已执行 | `bm>1` + 多 tile 全部编译通过（`CG-2026-0014` 不复现） |
| `RETROSPECTIVE.md` | ✅ 本次产出 | Stage 1 复盘 + 价值点提案（含 §9.4 版本戳诉求） |
| `_logical_reduce_kernel.py` | ⬜ 待实现（Stage 3） | 按本设计 §3.3/§5/§6 实现 + L0/L1 分层套件（§8.2） |

### 10.3 命名规范

- 项目目录：`examples/count_nonzero/`（project_name = count_nonzero）；算子目录：`_logical_reduce_kernel/`
  （op_name = harness 工厂函数名）。
- 实现文件：`_logical_reduce_kernel.py`（与工厂函数同名的 Developer 模式自包含形态：kernel + golden +
  L0/L1 自测，logsumexp / argmin 先例）。
- 集成落点（harness 侧）：`examples/TileOPs/tileops/kernels/reduction/count_nonzero/`
  的 wrapper 改 import 指向本 kernel + `default_config` 按 §5.2 适配（§5.6）；
  Op 层维持 `_kernel_handles_padding = True` / `_pad_value() = 0.0`。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 迁移分析（§0）
2. ✅ 等价性机器验证（`verify_equiv.py`：E1–E4 ALL PASS，含判别力对照组）
3. ✅ 设计期探针（UB 单价 8.00 B/elem + 窄行/多 tile 可编译性）
4. ✅ 设备/容量实查（`get_aicore_num()`=20 / `get_aivector_core_num()`=40 / `ub_cap`=196608）
5. ⬜ 算子实现（`_logical_reduce_kernel.py`：路径 R/C 双形态 + 工厂 + L0/L1 测试）+ golden 精确比对
6. ⬜ harness 接线（wrapper import + `default_config` → `_select_config` + msprof 锚点 +
   `test_count_nonzero.py` / `bench_count_nonzero.py` 跑通）
7. ⬜ Stage 4：性能调优 + §1.6.4 备选 R′ A/B 裁决 + 常数回填（含 §9.4 条目）
