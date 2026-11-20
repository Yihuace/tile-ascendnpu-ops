# `_simple_reduce_kernel_tiled`（AmaxFwdOp）算子设计文档

> 迁移任务（harness，Developer 模式）：GPU TileOPs `_simple_reduce_kernel_tiled`（`op_kind="amax"`）→ TileLang-NPUIR。
> 源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/reduction/reduce.py`（`ReduceKernel` + `_simple_reduce_kernel_tiled` 工厂；amax 走 `T.reduce_max` 分支，L150–L243）。
> 本设计基于：源码全文解读（Phase M0）、算法族调研（Phase R）、硬件耦合性判定与 NPU 重设计（Phase M1）、算法级优化设计（Phase 2）、**等价性机器验证**（D-1，`verify_equiv.py`）。
> 工具链戳：tilelang 0.1.2（dev root build）+ CANN 8.5.0 + **Ascend910_9362（20 AICore 实查）** + torch 2.7.1+cpu（2026-10-13 实查，见 §5.5）。
> **迁移单元范围 = tiled 流式路径**（N 超 UB 驻留上限，整行无法驻留，需沿 N 分块 + 跨 tile running max）。与已完成的兄弟单元 `_simple_reduce_kernel`（**驻留路径**，整行驻留 UB）互补：两者接口/语义衔接，由 wrapper 按 NPU UB 预算路由（§5.2 路由边界）。

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

**数学语义**：对 2D 输入 `x ∈ (M, N)` 沿最后一维（N，规约维）求逐行最大值，输出 `(M,)`：

$$y[i] = \max_{0 \le j < N} x[i,j]$$

即 `torch.amax(x, dim=-1)`（`op_kind="amax"`，manifest `AmaxFwdOp`；注意是**最大值**而非最大绝对值）。

**规约语义**：规约轴 = N（Op 层已把任意 rank 输入经 `movedim`/`flatten_for_multidim` 归一为 `(M, N)`）；keepdim 处理在 Op 层 `_reshape_output`。max 为**精确选择运算**——交换律 + 结合律 + 幂等，**无累加顺序问题、无舍入**。这是本算子与 sum/mean 类规约的本质差异：**「分块 + 跨 tile 合并」对 max 是严格数学恒等**（`max` 的幺半群结合律，见 §1.6.1-E3），不涉及 softmax/logsumexp 的 overflow 重缩放（no rescale needed）。

**dtype 语义**：输入 fp16 / bf16 / fp32；输出 `same_as(x)`。源 GPU 实现先把数据 cast 到 fp32（`tile_f32` fragment）再 reduce，结果 `T.cast` 回原 dtype（`out_local`）。

**边界语义**（源码 + torch 语义共同确定）：
- **±inf**：正常参与比较；全 `+inf` 行 → `+inf`；全 `-inf` 行 → `-inf`。
- **NaN**：`torch.amax` 传播 NaN；NPU 硬件 reduce 亦传播 NaN（argmax 探针实证，见 §9.2-R1）——两侧一致。
- **±0.0**：IEEE 相等比较；保持选择位模式。
- **非整除**：源 tiled 实现用 `N_padded = align_up(N, 256)` + `num_tiles = ceildiv(N_padded, tile_n)`，末 tile 列越界用 `T.if_then_else` 填**单位元 `-inf`**（amax 的 max 单位元，见 §1.6.1-E4）；M 尾块由 masked load 的行条件处理。
- **空维**：`N=0` 由 kernel 层构造期拒绝（源码 `N<=0` 无定义）；`M=0` 见 §9.2-R5。

**host 侧语义**：Op 层 `_prepare_input` 做 validate →（dim 归一 / `movedim` / `contiguous`）→ reshape 为 `(M, N)`；`AmaxFwdOp._kernel_handles_padding = True`（`examples/TileOPs/tileops/ops/reduction/amax.py`），故 Op 层不施加 host 侧 `F.pad`，raw `(M, N)` 直接进入 kernel。reshape/keepdim 均为**实现手段**而非算子语义，golden 不得复刻。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据（`torch.amax`，NaN 传播、±inf、-inf 单位元 pad 为文档化边界）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `(M, N)` | float16 / bfloat16 / float32 | 2D 连续行主序；M = 非规约维乘积，N = 规约维长度（**N 较大，整行无法驻留 UB**，源实现按 `N_padded = align_up(N, 256)` 处理对齐后分 `tile_n` 列块迭代） |
| `out` | `(M,)` | same_as(x) | 每行最大值（迁移必需字段：输出 shape `(M,)`，无转置布局） |

工厂契约（harness 固定）：`_simple_reduce_kernel_tiled(M, N, op_kind, dtype, tile_n)` 返回 `_func(block_m) → kernel(x)`；语义参数 `M/N/op_kind/dtype` **不变**，`tile_n` 为流式分块宽度（NPU 侧改为按 UB 预算推导），后端参数 `threads` 依 K9 去除（NPU 无 CUDA 线程概念）。

### 0.3 实现算法解读（怎么算）

源 kernel（`reduce.py` L150–L243，amax 分支）为**分块单遍流式（online running-max）**结构：沿 N 分 `tile_n` 列块逐块处理，维护跨 tile 的 running max。

**计算步骤分解**（源码每条计算语句均归入步骤，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 0 | `N_padded = align_up(N, 256)`；`num_tiles = ceildiv(N_padded, tile_n)`；`total_cols = num_tiles*tile_n`；`_needs_mask = total_cols > N`（L158–L161） | N, tile_n | 常量 | 分块规划 + 边界判定 |
| 1 | `T.fill(acc, -inf)`（L184–L185）：amax 累加器初始化为 max 单位元 | 常量 -inf | acc (bm,) f32 | 在线递推初始化 |
| 2 | **主循环** `for t in T.Serial(num_tiles)`（L189）：<br>2a `_needs_mask` 且 `t = num_tiles-1`：masked load `for i,j in T.Parallel(bm,tile_n)`：`tile_f32[i,j] = T.if_then_else(And(pid_m*bm+i<M, t*tile_n+j<N), cast(x[...], f32), cast(-inf, f32))`（L197–L208）<br>2b 其余：`T.copy(x[pid_m*block_m, t*tile_n], shared_buf)` → `tile_f32[i,j] = cast(shared_buf[i,j], f32)`（L209–L212） | x | tile_f32 (bm, tile_n, f32) | 载入 + 列/行边界（单位元填充）+ 精度提升 |
| 3 | amax 分支：`T.reduce_max(tile_f32, tile_acc, dim=1)`（L220） | tile_f32 | tile_acc (bm,) f32 | 当前 tile 行极值 $\max_{j\in\text{tile}_t}x$ |
| 4 | `for i in T.Parallel(bm)`：`acc[i] = T.max(acc[i], tile_acc[i])`（L221–L222） | acc, tile_acc | acc (bm,) f32 | **跨 tile running max 合并**（在线递推） |
| 5 | `for i in T.Parallel(bm)`：`out_local[i] = T.cast(acc[i], dtype)`（L236–L237） | acc | out_local (bm,) dtype | 结果降回输出 dtype |
| 6 | `T.copy(out_local, out[pid_m*block_m])`（L239） | out_local | GM out | 输出搬运 |

**数据流与内存访问模式**（源硬件视角）：

```
GM[x] --T.copy(或 masked Parallel load)--> SMEM[shared_buf (bm,tile_n)]
  --Parallel cast--> FRAG[tile_f32 (bm,tile_n,f32)]
  --T.reduce_max(dim=1)--> FRAG[tile_acc (bm,) f32]
  --Parallel: acc = max(acc, tile_acc)--> FRAG[acc (bm,) f32]      [循环携带]
[循环 num_tiles 次]
  --Parallel cast--> FRAG[out_local (bm,) dtype] --T.copy--> GM[out]
```

**循环与并行结构**：`T.Kernel(T.ceildiv(M, block_m), threads=threads)` 一维 grid × CUDA 线程两级并行；**N 维分块串行**（`T.Serial(num_tiles)`，循环携带 `acc`）；`block_m ∈ {1,2,4,8}` 由 48KB SMEM 预算启发式选择；`tile_n` 由 `_primitives.py` 的 `compute_tile_n`（除数优先）按 `N_padded > MAX_SINGLE_TILE_COLS(32512)` 触发；无 `T.Pipelined`（tile 循环有 carried 依赖，源未流水）。

**host 侧逻辑**：`functools.lru_cache(maxsize=32)` 按 `(M, N, op_kind, dtype, tile_n)` 缓存编译产物；Op 层 reshape/keepdim 后处理（见 §0.1 host 侧语义）；tiled 路径的 `tile_n` 由 `ReduceKernel._tile_n_for_block_m` 在 config 阶段推导。

**编译配置**：`@tilelang.jit(out_idx=[1])`（GPU target），无特殊 pass_configs。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | SMEM tile staging（分块 T.copy 载入） | 合并访存、tile 内一次载入多次使用 | 每 tile 驻留 shared memory，reduce 直接消费 | shared memory + 256 元素对齐 copy 约束 | 硬件强相关（容量/对齐不同） |
| 2 | **分块 running-max 流式（online 单遍）** | 大 N 时避免整行驻留；单遍 GM 扫描 | 逐 tile `acc=max(acc, tile_max)`，中间缓冲 O(N)→O(tile_n) | **无**（纯算法层） | **可移植（保留意图）** |
| 3 | 双路径边界处理（masked Parallel load vs 对齐 T.copy 快路径） | 非整除/非对齐时保正确性同时保留快路径 | `_needs_mask` 时末 tile 走 `T.if_then_else` + `T.And` 逐元素 masked load（填 -inf），否则走向量化 `T.copy` | GPU SMEM copy 的 256 元素对齐需求 | 硬件相关（对齐约束）；**单位元技巧可移植** |
| 4 | fp32 累加域上抛（`tile_f32` fragment） | 数值稳健（对 sum/mean/std 必要） | 全数据 cast fp32 后 reduce，结果降回原 dtype | — | 对 max **非必要**（精确选择，无舍入），见 §1.6.1-E1 |
| 5 | amax 单位元填充（-inf） | pad 列不改变结果 | `_pad_value_for_op("amax") = -inf`，max 的单位元 | — | **纯数学恒等，可移植**（承接为 §0.6-R1 / E4） |
| 6 | block_m × threads 两级并行 | 行块并行 + 线程级归约宽度 | 一维 grid × CUDA 线程；threads 同时是归约并行宽度 | CUDA 线程模型 | 硬件强相关（NPU 一维 Kernel 无 threads，K9 去除） |
| 7 | block_m 48KB SMEM 预算启发式（`[1,2,4,8]` 阶梯 + `compute_tile_n`） | 行块/tile_n 尽量大以摊薄 block 开销 | `SHARED_MEMORY_BUDGET_BYTES // (N_padded × elem)` + 阶梯取最大可行 | SMEM 容量 | 预算值硬件相关（NPU 用 UB 192KB） |
| 8 | `align_up(N,256)` + `MAX_SINGLE_TILE_COLS=32512` 分档 | 规避 TileLang GPU 向量化器 32768 列上限；>32512 切 tiled 变体 | `_needs_tiling = N_padded > MAX_SINGLE_TILE_COLS` | LLVM scalable-vector 边界 | 硬件强相关（NPU 无此上限，换 UB 容量分档） |
| 9 | `T.reduce_max` 硬件归约原语 | tile 内行归约用向量单元 | 树形归约（块内并行） | GPU reduce 原语 | NPU 有 Vector `T.reduce_max` 等价能力 |
| 10 | per-tile `T.max` 标量合并（running max） | 跨 tile 合并 | `T.Parallel(bm)` 内逐行 `T.max(acc[i], tile_acc[i])` | — | **可等价替换为单 op reduce 累加 / 向量 vmax**（§0.6-R3） |
| 11 | `lru_cache` 工厂缓存 | 消重复编译 | Python 层 `(M,N,kind,dtype,tile_n)` 键 | — | 可移植 |

> **识别完整性**：以上覆盖源码 `_simple_reduce_kernel_tiled` 的全部优化性结构（步骤分解表已把每条计算语句归入步骤 0–6，含 2a/2b 双路径）。无 warp shuffle / mma / swizzle / 异步流水等 GPU 深度优化（源码为朴素分块单遍法）。
> **源实现缺陷（Stage 3 不得照抄）**：源 L193/L210 的 `T.copy(x[pid_m*block_m, t*tile_n], shared_buf)` 无 `M` 尾块守卫（`M % block_m ≠ 0` 时越界读 shared_buf 未初始化行；步骤 5/6 的 `out_local` 亦按全 `block_m` 写）——本设计以 `real_m = T.min(bm, M−off_m)` + src/dst 双显式 slice 修正（§0.6-R1 / §5.4）。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| amax 数学语义（行最大值、输出 `(M,)` same_as(x)） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden = `torch.amax` |
| **分块 running-max 流式（#2）** | 算法 | 无（纯算法层） | — | **保留（本单元核心）** | NPU 同款 tile 循环 + running max；`max` 幺半群结合律保证数学恒等（E3）；无 rescale（对比 logsumexp 的 online softmax） |
| 步骤 3 `T.reduce_max` tile 行极值 | 算法/API | CUDA 线程归约 | 有：Vector `T.reduce_max`（dim=1） | **等价替换** | `docs/Tilelang.language/规约操作/T.reduce_max.md`；`testing/npuir/reduction_ops/test_reduce.py`（shared→shared、**同 dtype** 形态通过）；**同 dtype 约束**为 argmax 探针实证（fp16 src→fp32 dst 静默错误） |
| 步骤 4 per-tile `T.max` 标量合并（#10） | 优化/算法 | CUDA 线程内标量 | 有：reduce `clear=False` 累加 / `T.vmax` | **等价替换** | → §0.6-R3：单 op `T.reduce_max(..., clear=False)` running-max 累加（`docs/Tilelang.language/规约操作/T.reduce.md` §2.3 明确 `clear=False` 在 dst 现有值上累加）；回退 `T.vmax`（`docs/Tilelang.language/比较操作/T.vmax.md`） |
| 步骤 2a masked Parallel load + 单位元填充（-inf，#3/#5） | 优化 | GPU 256 元素对齐 | NPU **无需 256 对齐**（32B 尾轴）；且 `T.if_then_else` 引用 `T.serial` 循环变量的条件在本工具链触发编译器 segfault（logsumexp tiled 实证，§0.6-R1） | **重新设计** | → §0.6-R1：kernel 接收**原始 N**（无 N_padded、无 masked load），末尾窄 tile 用**静态尾 tile 特化**（尾宽为编译期常量，独立缓冲，无 -inf 填充）；M 尾块用 `T.min` + 双显式 slice |
| 步骤 2b fp32 上抛域（`tile_f32`，#4） | 优化/精度 | — | fp16/fp32 可**原域**等价（max 精确选择）；bf16 因 reduce 不支持而必须 cast | **等价替换（重新定域）** | → §0.6-R3 / §1.6.1-E1/E2：fp16/fp32 原域 reduce；bf16 `T.vcast→fp32` |
| SMEM tile staging + 256 对齐 copy（#1） | 优化 | shared memory | 有：UB（`T.alloc_shared`，192KB） | **等价替换** | GM→UB staging；对齐需求降为 32B 尾轴 |
| block × thread 两级并行 / `threads` 参数（#6） | 优化 | CUDA 线程 | 无（本项目一维 Kernel） | **重新设计** | → §0.6-R2：一维 persistent grid（纯 Vector 核数 = 实查 aicore×2 = 40）+ 核内 `T.serial` 静态边界任务循环；threads 依 K9 去除 |
| block_m/tile_n SMEM 预算启发式（#7/#8） | 优化 | SMEM 容量 + LLVM 向量化上限 | 换 UB 预算模型；无 32768 上限 | **重新设计** | → §0.6-R4：UB 64KB 手工预算 + 驻留/流式统一 gate（`N·B ≤ 64KB` 驻留否则 tiled） |
| 单位元 pad 的「不改变结果」意图（#5） | 优化 | — | — | **保留**（作为等价性依据） | 由 §1.6.1-E4 机器验证承接；NPU 上通过「不 padding / 静态尾 tile」实现同一意图 |
| `lru_cache` 工厂缓存（#11） | 优化 | — | — | **保留** | 工厂层与硬件无关 |

> 每项处置依据：migration-analysis.md §5.3 映射表（shared memory→UB、block×thread→一维重设计、online/算法层保留）+ 本仓 `docs/Tilelang.language/` API 文档 + `testing/npuir/reduction_ops/` 通过用例 + pattern-library 实测条目。无「舍弃后性能意图丢失」项：#2（online 流式）为本单元**核心保留**；#3/#5 的对齐/pad 意图由 R1 的静态尾 tile 与 E4 承接；#4 的精度意图对 max 冗余，由 E1/E2 论证承接（bf16 仍保留 cast）；#8 的上限分档意图由 R4 的 UB 预算 gate 承接。

### 0.6 NPU 算法重设计

**重设计项 R1：边界处理（源：256 对齐 pad + 末 tile masked load + M 无守卫 copy）**

- **源方案**：`N_padded = align_up(N, 256)`；`_needs_mask = num_tiles*tile_n > N`；末 tile 用 `T.Parallel(bm, tile_n)` + `T.if_then_else(T.And(row<M, col<N), x, -inf)` 逐元素 masked load，其余 tile 用 `T.copy` 快路径。性能意图：满足 GPU SMEM copy 的 256 元素对齐，同时把非整除正确性放进 kernel。
- **NPU 新算法**：kernel 直接接收**原始 `(M, N)`**（`AmaxFwdOp._kernel_handles_padding=True`，Op 层本就不 pad），**取消 N_padded 与 masked load**。N 分块为 `num_full = N // tile_n` 个**满 tile**（宽度 `tile_n`，编译期常量）+ 一个**静态尾 tile**（宽度 `valid_n = N − num_full·tile_n`，**编译期常量**；`valid_n == 0` 时该分支由 Python 构建期跳过）。尾 tile 使用独立缓冲 `x_tail_ub (1, valid_n)` 直接 `T.copy`，**不做 -inf 填充**（宽为编译期常量，避免 `T.serial` 循环变量条件触发编译期 segfault——logsumexp tiled 实现反例，见 §9.1）。M 尾块用 `real_m = T.min(block_m, M − off_m)` + src/dst 双显式 slice（`T.copy(x[off:off+real_m, c0:c1], x_ub[0:real_m, 0:w])`），垃圾行不写出。
- **语义保持论证**：**消除 padding 不可能改变任何输出**——pad 值为 amax 的单位元 `-inf`（`max(S ∪ {-inf...}) = max(S)`）；静态尾 tile 只搬运**真实存在**的列，等价于对原 N 列直接 reduce。M 尾块 `real_m` 仅裁掉不存在的行（每行独立规约），不改变任一有效行的结果。由 §1.6.1-E4（raw N + 静态尾 vs -inf pad 基线，0 违反）与 E3（分块 running max ≡ 整行，含非整除尾 tile，0 违反）机器验证。边界核对：全 `-inf` 行在两种处理下均得 `-inf`；NaN 行两侧均传播 NaN。

**重设计项 R2：并行结构（源：block × thread 两级 + `threads` 参数）**

- **源方案**：`T.Kernel(T.ceildiv(M, block_m), threads=threads)`；`threads ∈ {128,256}` 同时是归约并行宽度与布局约束参数。
- **NPU 新算法**：纯 Vector **一维 persistent kernel**——`T.Kernel(num_kernels, is_npu=True) as (cid, _)`，`num_kernels = min(num_logical, vector_cores)`，`vector_cores = NPUUtils.get().get_aicore_num() × 2 = 20 × 2 = 40`（纯 Vector 核数翻倍，实查见 §5.5）；核内 `T.serial(静态上界)` 任务循环（grid-stride 映射 `task = s·num_kernels + cid`）+ 动态 guard `if task < num_logical`。**tiled 路径强制 `block_m = 1`**：multitile（`num_tiles>1` 的 `T.serial` tile 循环）+ `bm>1` 组合触发编译器 SIGSEGV（`CG-2026-0014`），故 tiled 路径 `num_logical = M`，每任务处理 1 行。行内 N 分块串行（tile 循环），行间并行交给 grid。`threads` 停止出现（K9）。
- **语义保持论证**：行之间**完全独立**（per-row 规约，无跨行依赖），任务循环只改变调度、不改变计算；每行结果与 block_m/核数无关；tile 循环内 running max 是**行内**合并（`max` 结合律，E3 机器验证 228 例 0 ulp）。尾块 `real_m` + 双 slice 保证越界行不被读、垃圾行不被写。

**重设计项 R3：dtype 计算域与跨 tile 合并（源：全量 fp32 上抛 + per-tile 标量 T.max）**

- **源方案**：`tile_f32 = cast(x, f32)` 全 tile cast → `reduce_max` → `T.Parallel(bm)` 逐行 `acc[i] = T.max(acc[i], tile_acc[i])` → `out_local = cast(acc, dtype)`。
- **NPU 新算法**：**fp16 / fp32 原域 reduce_max**（`T.reduce_max(x_ub, acc, dim=1, clear=False)`，src/dst 同 dtype），输出即已为目标 dtype（省去降回 cast）；**bf16 必须 `T.vcast(x_ub, x_work, round_mode="rint")` 升 fp32 后 reduce**（`T.reduce.md` / `T.reduce_max.md` §2.2.1 dtype 表：bf16 ×；fp16/fp32 √），再 `T.cast` 回 bf16。**跨 tile running max** 用**单 op reduce 累加**替代源 per-tile 标量合并：`acc` 初始化为 `-inf`（`T.vbrc(-T.infinity(...), acc)`），每个 tile 执行 `T.reduce_max(src, acc, dim=1, clear=False)`，在 `acc` 现值上取 max 并写回（`T.reduce.md` §2.3 明确 `clear=False` 语义 = 在 dst 现有值上累加）。
- **语义保持论证**：max 是**精确选择运算**（无算术、无舍入）——fp16→fp32 为精确单射且保序保等，两域选出的极值元素相同 ⟹ 原域结果与 fp32 上抛再舍回逐位一致（E1：fp16/fp32 原域 vs fp32 上抛基线，160 例 0 ulp）。bf16↔fp32 亦为精确单射保序（E2：bf16 域 max ≡ fp32 域 max，77 例 0 违反）。跨 tile 合并 = 对 tile 极值序列再取 max，`max` 结合律/交换律保证 ≡ 整行 max（E3：228 例 0 ulp，含尾 tile 非整除）。`acc` 初始 `-inf` 为单位元，不影响结果。边界核对：±inf / NaN / 亚正规 / ±0 在单射往返下位模式保持（E1/E4 角点集覆盖）。
- **实现细节与回退**：`clear=False` 累加的循环携带性由官方 reduce 文档语义保证（`docs/Tilelang.language/规约操作/T.reduce.md` §2.3）；若 Stage 3 实测该 toolchain 的 `clear=False` 跨 `T.serial` 迭代携带异常（类比 `TRAP-vselect-inplace-carried-state` 的 in-place carried-state 陷阱），回退为**无别名三 op 链**：每 tile `T.reduce_max(src, tile_acc, dim=1, clear=True)` + `T.vmax(acc, tile_acc, new_acc)`（不别名）+ `T.vbrc(0, tmp); T.vadd(tmp, new_acc, acc)`（fragment 复制技巧，flash_attn/logsumexp 先例）。两形态数学等价（同一 running max）。

**重设计项 R4：tile_n / 驻留分档（源：48KB SMEM 启发式 + 32512 列上限）**

- **源方案**：`block_m` 由 `SHARED_MEMORY_BUDGET_BYTES(48KB) // (N_padded × elem)` + `[1,2,4,8]` 阶梯选择；`N_padded > MAX_SINGLE_TILE_COLS(32512)` 时切 tiled 变体（LLVM 32768 列向量化边界）；`tile_n` 由 `compute_tile_n` 除数优先选取。
- **NPU 新算法**：改用 **UB 手工预算 + 驻留/流式统一 gate**：`B(dtype)` = 单 row tiled 大缓冲每元素字节数（fp16 2 / fp32 4 / bf16 6，见 §4.5）；**路由 gate**（wrapper 期，替代 GPU 的 `N_padded > MAX_SINGLE_TILE_COLS`）：`N·B(dtype) ≤ 64KB` → 驻留路径（兄弟单元 `_simple_reduce_kernel`）；否则 → 本 tiled 单元。**tile_n**：在 `tile_n ≤ 64KB // B(dtype)` 且为 `N` 的因子（零尾 tile、消除近空尾）中取最大者；无因子时取预算上限 + 静态尾 tile（§5.2）。tiled 路径 `block_m = 1` 强制（CG-2026-0014）。64KB 为**竞态安全手工预算**（同族 argmax 探针定标：手工 64KB → auto-multi-buffer 双缓冲 ≤128KB ≤192KB−8KB reserve）。`MAX_SINGLE_TILE_COLS` 分档弃用（NPU 无该上限）。
- **语义保持论证**：纯配置/路由选择，不改变任何计算；预算边界由同族 argmax 探针的通过/失败对双向标定；tile_n 选取不改变结果（任意 tile 划分下 running max ≡ 整行，E3 覆盖 8 组 split）；路由 gate 只决定由哪个单元承接（两单元语义一致，§5.2）。

### 0.7 标杆实现

- **源算子**：`/tmp/opencode/TileOPs-ref/tileops/kernels/reduction/reduce.py`（`_simple_reduce_kernel_tiled`，L150–L243；§0.1–§0.4 解读对象）；非 tiled 兄弟 `_simple_reduce_kernel`（同文件 L69–L147）；常量/原语：`/tmp/opencode/TileOPs-ref/tileops/kernels/reduction/_primitives.py`。
- **NPU 侧 harness 工件**：提取版 GPU 参考 `examples/TileOPs/tileops/kernels/reduction/amax/_amax_fwd_kernels.py`（含 `_simple_reduce_kernel_tiled`，L137–L229）；已移植 wrapper（Part B，K5–K9）`examples/TileOPs/tileops/kernels/reduction/amax/amax.py`（`npub::reduce_simple_tiled_fwd` custom_op + `ReduceKernel`，`threads` 已去除）；Op 层 `examples/TileOPs/tileops/ops/reduction/amax.py`（`_kernel_handles_padding=True`）；规格 `examples/TileOPs/tileops/manifest/reduction.yaml`（`AmaxFwdOp` 段）。
- **golden 基准**：`examples/TileOPs/tests/ops/test_reduce.py`（`ReduceTest.ref_program = x.float().amax(dim=-1).to(x.dtype)`；容差 fp32 1e-4 / 其他 1e-2）。
- **同族已验证参考**：
  - `examples/amax/_simple_reduce_kernel/DESIGN.md`（**驻留兄弟，接口/语义衔接的对端**：R1–R4 重设计 + 边界处理 + persistent 分核 + §5.2 路由边界）。
  - `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_tiled/`（+DESIGN.md +实现）：**同族 N 超驻留流式先例**、`T.serial` tile 循环、online 递推、(bm,1) fragment 更新链；**其实现记录「kernel 侧 `T.if_then_else` + `T.serial` 循环变量条件 segfault」反例** → 本设计静态尾 tile 特化的依据。
  - `examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/`（`_argreduce_kernel.py` + `_argreduce_kernel_DESIGN.md`）：**persistent 分核、UB 预算、同 dtype reduce、M 尾块三件套、静态尾 tile 特化、tiled 路径 bm=1 强制（CG-2026-0014）**的直接先例。
  - `examples/TileOPs/tileops/kernels/reduction/_primitives.py`（`compute_tile_n` / `ub_slab_units` / `align_up` / `MAX_SINGLE_TILE_COLS`）。

---
## 1. 概述

### 1.1 算子名称

`_simple_reduce_kernel_tiled`（承载 `AmaxFwdOp` 的 NPU **流式（tiled）** reduce kernel；工厂接口保留 `op_kind`/`tile_n` 以复用兄弟 reduce 算子）

### 1.2 功能描述

对 `(M, N)` 输入沿 N 维（规约维）求逐行最大值，输出 `(M,)`；**N 超 UB 驻留上限时**沿 N 分 `tile_n` 列块流式处理，维护跨 tile running max，M 行间分核；M 尾块与 N 非整除尾 tile 均在内处理（无 host pad）。

### 1.3 数学公式

$$
y[i] = \max_{0 \le j < N} x[i,j]
$$

NPU 侧等价计算式（§1.6.1 优化后，供 §3.1 拆解）——按输入 dtype 分派 + 跨 tile 合并：

$$
a_t[i] = \mathrm{reduce\_max}_{j \in \text{tile}_t}\big(x[i,j]\big);\quad
m[i] = \max_{t=0}^{T-1} a_t[i]\ \ (\text{running max over tiles});\quad
y[i] = m[i]
$$

（tile 源域：fp16 / fp32 为原域 `x`；bf16 为 `vcast_{bf16\to fp32}(x)` 后的 fp32 域，输出再降回 bf16。合并 `max_t a_t[i]` 即跨 tile running max，`max` 结合律保证 ≡ 整行 max。）

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

**分块单遍流式（online running-max）算法**（per-row 独立；与源算法差异逐条标注来源）：

1. **分块规划**（工厂期）：`num_full = N // tile_n`（满 tile 数，编译期常量）；`valid_n = N − num_full·tile_n`（静态尾宽，编译期常量；`valid_n == 0` 时无尾块分支）〔§0.6-R1/R4〕；
2. **调度**：一维 persistent，`num_kernels = min(M, 40)`（tiled 路径 `block_m=1` 强制 → `num_logical = M`）；核内 `T.serial` grid-stride 任务循环 + guard〔§0.6-R2〕；
3. **初始化**：`T.vbrc(-T.infinity("float32"), acc)`（running max 初值 = max 单位元）〔§0.6-R3〕；
4. **满 tile 循环** `T.serial(num_full)`：`T.copy(x[row, t·tn:(t+1)·tn], x_ub)`；bf16 经 `T.vcast` 升 fp32；`T.reduce_max(src, acc, dim=1, clear=False)`（running max 累加）〔§0.6-R1/R3〕；
5. **静态尾 tile**（构建期 `if valid_n > 0`）：`T.copy(x[row, num_full·tn : num_full·tn+valid_n], x_tail_ub[0:1, 0:valid_n])`；bf16 vcast；`T.reduce_max(tail_src, acc, dim=1, clear=False)`（无 -inf 填充）〔§0.6-R1〕；
6. **出栈**：`out_ub[0] = acc[0,0]`（fp16/fp32 恒等）或 `= T.cast(acc[0,0], dtype)`（bf16）；`T.copy(out_ub[0:1], out[off_m : off_m+1])`〔形状收缩 (1,1)→(1,)〕；
7. **dtype 定域**：fp16 / fp32 原域 reduce；bf16 `T.vcast→fp32` 后 reduce 再降回〔§0.6-R3 / E1/E2〕。

**与源算法差异**：① `align_up(N,256)` + 末 tile masked load → 原始 N + **静态尾 tile 特化**（无 -inf 填充）+ M 尾块双 slice（§0.6-R1；源 §0.4-#3/#5）；② block×thread → 一维 persistent 且 **tiled 路径强制 bm=1**（§0.6-R2；源 §0.4-#6）；③ 全量 fp32 上抛 → fp16/fp32 原域、bf16 单独 vcast；per-tile 标量 `T.max` 合并 → **单 op reduce 累加**（§0.6-R3；源 §0.4-#4/#10）；④ 48KB SMEM 启发式 + 32512 分档 → UB 64KB 预算 + 驻留/流式统一 gate（§0.6-R4；源 §0.4-#7/#8）。**保留**：**分块 running-max 单遍流式结构**（§0.5-#2，本单元核心）与 `T.reduce_max` 硬件归约（§0.4-#9）。

### 1.5 数据流图

```
流式路径（纯 Vector，单引擎单 kernel，tiled）：
GM[x] --T.copy(slice)--> UB[x_ub (1, tile_n) dtype]      # 满 tile
   --[bf16: T.vcast]--> FRAG[x_work (1, tile_n) f32]
   --T.reduce_max(dim=1, clear=False)--> UB/FRAG[acc (1,1) 计算域]   # running max 累加（循环携带）
[重复 num_full 次]
GM[x] --T.copy(slice valid_n)--> UB[x_tail_ub (1, valid_n)]        # 静态尾 tile（valid_n 编译期常量）
   --[bf16: T.vcast]--> FRAG[x_tail_work]
   --T.reduce_max(dim=1, clear=False)--> acc                        # 合并尾 tile
   --T.Parallel(1) [恒等/cast]--> UB[out_ub (1,)]
   --T.copy(slice)--> GM[out (M,)]
无 L1/L0 参与；无 GM 中间往返；无 workspace；x 仅读一遍。
```

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：**逐行最大值规约**这一数学语义的算法族（源 tiled 算法只是基线候选之一）。amax 属规约类 → **完整调研**（R1 候选表 + R2 在线结论 + R3 四口径对比 + R4 逐候选评估）。
> 信息源：skill 参考表 `.agents/skills/tilelang-op-design/references/algorithm-candidates.md`（`ALG-reduction` 行）；`kb_search` 命中（`CASE-reduction-logsumexp` / `CONST-reduce-dims0-skinny` / `CONST-gm-to-ub-bw-dilution` / `CASE-argmax-argreduce-integration` / `CONST-vector-launch-overhead`）；本仓 `examples/`（logsumexp single/tiled、argmax、`_primitives.py`、兄弟 `_simple_reduce_kernel`）；`testing/npuir/reduction_ops/`（reduce 原语通过用例）；pattern-library（constants/traps/layout/cases）；`CG-2026-0014`（multitile+bm>1 SIGSEGV）；源算子 §0.3/§0.4；结构判据分析。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1 完成）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| 0 | **基线：GPU 源分块流式（padded/masked + fp32 上抛 + per-tile 标量 max）** | `acc = max(acc, tile_max)`，非整除列填 -inf | — | — | ✅（基线） |
| 1 | **C1 驻留原域单遍**（兄弟单元；N·B ≤ 64KB） | `y = reduce_max(x)`（整行驻留；fp16/fp32 原域） | 数学恒等（max 精确选择） | 省 fp32 cast pass + 缓冲；**但不适用于大 N** | 参考（兄弟单元） |
| 2 | **C2 分块 running-max 流式（online）+ 静态尾 tile** | 逐 tile `acc = max(acc, chunk_max)`；尾宽编译期常量、无 -inf 填充 | 数学恒等（max 为交换幂等幺半群 + 结合律） | 大 N 时单遍 GM 扫描；中间缓冲 O(N)→O(tile_n)；静态尾 tile 消除 masked load | ✅ **（本单元主选）** |
| 3 | C3 跨核 N-split partial+merge | 列分片求 partial max → workspace → merge kernel 求全局 max | 数学恒等（max 结合） | 小 M 大 N 负载核利用率提升；代价 = workspace 往返 + 第二 kernel | ⏸ Stage 4 候选 |
| 4 | C4 全排序取首 | `sort(x)[...,-1]` | 数学恒等 | O(N log N) ≫ O(N) | ❌（复杂度劣） |
| 5 | C5 变换域（log/exp 等） | 对 max 无保序等价变换可降代数 | **无**：max 是完备格上的选择运算，任何单调双射变换都不减少比较次数，非单调变换不等价 | — | ❌（无收益空间） |
| 6 | C6 树形/成对归约重排 | 平衡二叉树比较（与线性扫描同比较次数） | 数学恒等 | FLOPs 同阶；硬件 `T.reduce_max` 已实现块内树形归约 | ❌（硬件原语已覆盖） |

> **代数化简空间的结构性结论**：max 是**交换、结合、幂等**运算，任何算法至少要读每个元素一次并做 N−1 次比较（信息论下界）；不存在减少比较次数或 FLOPs 的代数化简（C5/C6 由结构判据排除，非未检索先否定）。故 R1 的可行收益集中在 **dtype 域（C1/C2 的原域）**、**扫描/缓冲结构（C2 静态尾 + 单遍流式）** 与**跨核组织（C3）**，而非比较次数。

**R2 在线算法**：**有，且本单元即采纳**——max 可分解为 running 统计量（running max），`acc = max(acc, chunk_max)` 为流式合并（结构判据满足；GPU 源 tiled 本身即用该在线结构，§0.3 步骤 4 是直接证据；同族 logsumexp tiled 已验证同族 online 递推）。**与 softmax/logsumexp 的关键差异**：amax 的在线更新**无需 rescale**（无 running sum、无 `exp(prev_max−new_max)` 校正因子）——`max` 合并即全部，故 amax 的在线结构比 online softmax 更轻（每 tile 仅 1 个合并 op，对比 logsumexp 的 vmax+vsub+vexp+vsub+vexp+vmul+vadd 链）。**收益口径**：单遍 GM 扫描、中间缓冲 O(N)→O(tile_n)、UB 可驻留性（整行不能驻留时仍可处理任意大 N）。本单元（tiled）采纳 online；驻留兄弟单元整行在片上，单遍即最优、无 online 增量收益（其 R2 结论）。

**R3 复杂度对比**（四口径；per-row 计，M 行并行摊派；口径写清含/不含；带宽受限算子——manifest flops = M·N vs bytes = (M·N+M)·elem，访存主导）：

| 算法候选 | FLOPs（比较次数，per row） | 访存量 Bytes（GM：读 x + 写 out + 中间往返） | 扫描遍数（GM） | 中间缓冲峰值（per row） | 可并行度 / 跨核代价 |
|---------|------|------|------|------|------|
| 基线（GPU padded + fp32 上抛 + 标量 max） | N−1（max）+ N（fp32 cast）≈ 2N | N·elem + elem（1 遍，**不含** cast 是片上） | 1 | x_f32 4B/elem + shared_buf elem（tile_n 级） | M/bm 行块独立，无跨核；tile 循环串行 |
| **C1 驻留原域单遍（兄弟）** | N−1（max，无 cast pass） | N·elem + elem（1 遍） | 1 | x_ub N·elem（整行驻留，fp16 2 / fp32 4 / bf16 6 B/elem） | 同上；**大 N 超 UB 不可行** |
| **C2 分块 running-max 流式（本单元）** | (N−1)（max；fp16/fp32 无 cast pass） | N·elem + elem（1 遍）+ workspace 0 | 1 | O(tile_n)：x_ub tile_n·elem（+ bf16 x_work） | 行独立；tiled 路径 bm=1 → 小 M 时核利用率低（C3 承接） |
| C3 N-split partial+merge | (N−1) + nchunk（merge 比较） | N·elem + elem + workspace 往返 `M·nchunk·(elem+elem)` | **2**（partial 写 + merge 读 workspace；x 仍 1 遍） | partial 工作集 O(tile) + workspace | 跨核 40 核可达；代价 = 第二 kernel 启动 + workspace 往返 |
| C4 全排序 | N log N | N·elem + elem（大中间） | 1 | 排序缓冲 ≫ N·elem | O(N log N) 冗余 |
| （对照）tiled 两遍式（非 online） | 2N | **2N·elem** + elem | **2** | O(tile_n) | 同 C2 |

**R4 硬件亲和性评估**（检查清单逐项；实测常数引用 pattern-library/constants.md 条目 + 版本戳；负向淘汰附佐证）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（masked load 形态） | ⚠️ `T.if_then_else` 引用 `T.serial` 循环变量条件在本工具链触发 segfault（logsumexp tiled 实现反例，§9.1）；fp32 上抛多 1 全宽 pass | tile_n 级 staging | ❌ 256 元素对齐（GPU copy 约束） | ✅ | ⚠️ | ❌ **形态需重设计**（→C2 + §0.6-R1） |
| **C1 驻留原域单遍（兄弟）** | ✅ 纯 Vector（`T.reduce_max` dim=1；fp16/fp32 ✓、**bf16 ✗→预 cast**）；`testing/npuir/reduction_ops/test_reduce.py` 通过 | ✅ N·B ≤ 64KB 才驻留（fp16 N≤32768 / fp32 N≤16384 / bf16 N≤10922） | ✅ 32B 尾轴（logsumexp N=300 先例） | ✅ | ✅ auto-multi-buffer | ✅ **驻留路径（兄弟单元）** |
| **C2 分块流式（本单元）** | ✅ 纯 Vector（`T.reduce_max` dim=1 + `clear=False` 累加 / `T.vmax` 回退） | ✅ tile 工作集 ≤64KB（tile_n·B ≤ 64KB）；大 N 任意可处理 | ✅ tile_n 静态；静态尾 tile 特化（尾宽编译期常量）；M 尾块 `T.min` 双 slice | ✅ num_full/valid_n 全工厂期常量；核内任务循环边界静态（仅 guard 动态） | ✅ tile 循环 auto-multi-buffer（tile 数据单消费者） | ✅ **tiled 路径选定（本单元）** |
| C3 N-split merge | ✅ Vector | ⚠️ workspace `M·nchunk·2·elem`（long-seq 类 ≈KB–百KB 级，≪ x 读流量） | — | ✅ | ⚠️ 两 kernel 启动 + 合并正确性面 | ⏸ **Stage 4 A/B**（小 M 大 N 核利用率） |
| C4 排序 | ✅ | ❌ 排序缓冲爆 UB | — | — | — | ❌ O(N log N) 复杂度劣 |

**调研结论**：选定 **C2（分块 running-max 流式 + 静态尾 tile 特化）为本单元主选**（N 超 UB 驻留上限的流式路径），驻留路径由兄弟单元 C1 承接。关键依据：R1/R2——max 无代数化简空间、但存在结构更优的 online 单遍（且 amax 的 online 无 rescale，比 logsumexp 更轻）；R3——C2 相对基线**省去 1 个全宽 cast pass**（fp16/fp32 原域）、遍数/缓冲不增（缓冲 O(N)→O(tile_n)）；R4——纯 Vector 原语可达（同 dtype reduce + `clear=False` 累加）、静态尾 tile 规避 segfault、tile 循环 auto-multi-buffer。与基线的结构差异一句话：**「padded/masked load + fp32 上抛 + per-tile 标量 max + block×thread」→「原始 N + 静态尾 tile + 原域 reduce + 单 op running-max 累加 + 一维 persistent（bm=1）」**。
**调研范围**（"无更优代数化简"的查证范围）：`algorithm-candidates.md` 的 `ALG-reduction` 行 + `kb_search` 命中条目（上述 5 条）+ `examples/`（logsumexp single/tiled、argmax、兄弟 `_simple_reduce_kernel`、`_primitives.py`）+ `testing/npuir/reduction_ops/`（reduce 通过用例）+ pattern-library（constants/traps/layout/cases）+ 恒等变形空间自查（C4/C5/C6 全枚举，结构判据排除）+ 互联网检索未触发（本地源已覆盖该成熟算法族，无需外部候选）。**源算法优化手段承接**（§0.4）：#2 online 流式 → C2 核心保留；#3/#5 masked 边界/pad → R1（单位元意图由 E4 承接）；#4 fp32 上抛 → E1（对 max 冗余，意图由等价论证承接，bf16 保留 cast）；#6 block×thread → R2；#7/#8 SMEM 启发式/上限分档 → R4；#10 per-tile 标量 max → R3 单 op 累加；#9 reduce 原语保留；#11 缓存保留。

**设计期 roofline（D-2；协议：hardware-cost-model.md §2；常数带版本戳）**：

| workload（manifest） | 路径/配置 | 流量项 | 发射项 | 估算下界（µs） |
|---|---|---|---|---|
| smoke (32,256) fp16 | 驻留（兄弟单元） | — | — | 本单元不适用（§5.2 路由：N·B=512B ≤ 64KB） |
| hidden-state (2048,4096) fp16/bf16 | 驻留（兄弟单元） | — | — | 本单元不适用（fp16 8KB / bf16 24KB ≤ 64KB） |
| 3d-multidim (4,128,4096) dim=[0,2] → **M=128, N=16384** fp16 | 驻留（兄弟单元） | — | — | 本单元不适用（N·B=32KB ≤ 64KB） |
| **long-seq (64,32768) bf16** | tiled，bm=1，tile_n=8192 → 4 tile，64 logical → 40 核 × 2 任务 | 4.19MB ÷ (30.2GB/s·核〔`CONST-mte2-degradation` 16M 档〕× 40 核) = 3.5µs | 512 vector op ÷ 40 核 × 0.5µs〔`CONST-vector-launch-overhead`〕= 6.4µs | **≈6.4（发射主导）** |
| （自测/large-N）(128,65536) fp16 | tiled，bm=1，tile_n=16384 → 4 tile，128 logical → 40 核 × 4 任务 | 16.8MB ÷ (30.2 × 40) = 13.9µs | 1024 op ÷ 40 × 0.5 = 12.8µs | **≈13.9（流量主导）** |
| （自测/非整除）(4,100000) fp16 | tiled，bm=1，tile_n=8192 → 12 满 tile + 尾 1696，4 logical → 4 核 × 1 | 0.8MB ÷ (30.2 × 4) = 6.6µs（核利用率 4/40） | 52 op ÷ 4 × 0.5 = 6.5µs | **≈6.6**（小 M 低核利用率 → §1.6.4 C3 动因） |

> 估算下界: max(流量项, 发射项)（容量项 n/a——纯 Vector 单引擎、无跨引擎 S/P 往返，`CONST-store-fixpipe-gm-only` 不适用）。**发射项为串行上界**——Developer auto-multi-buffer 流水下 tile 数据装载可部分隐藏发射，实际可低于该值。**设备/常数口径声明**：`CONST-mte2-degradation` 与 `CONST-vector-launch-overhead` origin 工具链为 Ascend910B2C（24 AICore）；本设备实查 **Ascend910_9362（20 AICore → 纯 Vector 40 核）**，估算按「per-core 带宽/发射不变 × 40 核」外推，属**未实测假设 + 估算依据**（Stage 4 回填对账，见 hardware-cost-model.md §3）。MTE2 按总流量档取 HBM 口径。向量 op 数口径：满 tile 路径 fp16/fp32 为 1 op/tile（reduce），bf16 为 2 op/tile（vcast+reduce）；large-N 行按 fp16 计。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（C2）。逐项四要素；**等价性机器验证（D-1）**：`verify_equiv.py`（torch CPU，fp64 参照 + dtype 域仿真 + 角点集），结果表内嵌于下。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| E1 | **fp16/fp32 原域 reduce_max**（承接 §0.6-R3） | `m_t = reduce_max(cast(x_t,fp32)); y = cast(m, dtype)` | `m_t = reduce_max(x_t)`（fp16/fp32 原域，src/dst 同 dtype） | max 是**精确选择**运算（比较 + 选择，**无算术、无舍入**）：fp16→fp32 为精确单射且保序保等，两域选出的极值元素相同 ⟹ `reduce_max_fp32(x).to(dtype) ≡ reduce_max_dtype(x)` 逐位相等。fp32 输入无上抛 | 省 1 个全 tile fp32 cast pass（每 tile n 元素）+ 2B/elem fp32 工作缓冲（fp16）；输出即目标 dtype，省降回 cast。**注**：不能跨 dtype reduce（argmax 探针实证 fp16 src→fp32 dst 静默错误），故必须同域 |
| E2 | **bf16 域不变 / 单射往返**（承接 §0.6-R3） | （GPU 原生 bf16 reduce；NPU 无此能力） | `x_work = vcast_rint(x)`（bf16→fp32）→ `m = reduce_max(x_work)` → `cast(m, bf16)` | bf16↔fp32 为精确单射（bf16→fp32 补零位、fp32→bf16 截断仅对可精确表示的 bf16 值恒等）；bf16 域 max ≡ fp32 域 max（同一被选元素），故往返结果 = golden | bf16 路径**必须**引入 1 次 vcast（`T.reduce*.md` §2.2.1 bf16 ×）——非优化项而是 API 必要性；无额外数值代价（E2 验证 0 ulp） |
| E3 | **分块 running max 合并**（承接 §0.6-R2/R3，本单元核心） | 源：逐 tile `acc[i] = T.max(acc[i], tile_acc[i])`（标量合并，`T.Parallel(bm)`） | 单 op `T.reduce_max(src_t, acc, dim=1, clear=False)`（在 acc 现值上取 max 并写回）；acc 初值 `-inf` | `max` 是**交换、结合、幂等**运算 ⟹ `max_t(max_{j∈tile_t} x[j]) ≡ max_j x[j]`（对任意 tile 划分与尾宽成立）。数学恒等，与 fp16/bf16 累加顺序无关（max 无舍入） | 每 tile 从「reduce + 标量合并」2 处收为 **1 个向量 op**（满 tile 路径），减 1 次 (1,1) 微操作发射；合并顺序改变不产生数值差（E3 验证 0 ulp） |
| E4 | **-inf 单位元 padding 消除 + 静态尾 tile**（承接 §0.6-R1） | 非整除列填 `-inf`（`N_padded`）后 masked reduce（源） | 原始 N；末尾窄 tile 用独立缓冲按**编译期常量** `valid_n` 直接 `T.copy` + reduce（无 -inf 填充） | `-inf` 是 max 的单位元：`max(S ∪ {-inf,…}) = max(S)` 对所有 `S`（含全 -inf / 含 `+inf` / 含 NaN）成立；静态尾 tile 只搬运真实存在的列 ⟹ 消除 padding 不改变任何输出 | 消除 masked `T.if_then_else` 逐元素分支（且规避 serial 循环变量条件 segfault）；N 任意宽度直承（无 N_padded 缓冲）；尾宽编译期常量 → 无动态边界 |

**机器验证结果表**（`verify_equiv.py`，2026-10-13 执行，exit 0；输入 dtype 域仿真（逐元素 `torch.maximum` 在原 dtype 内的串行 max，不离开输入域）+ fp64 参照；shape 覆盖 (1,1)…(33,4096) 与 (2,100000) × {fp16,bf16,fp32} + 角点行〔全−inf / 全+inf / +inf 夹有限 / −inf 夹有限 / 重复极值 / 亚正规极值 / 混合符号零 / dtype 极值 / 全零 / 全 NaN / NaN 夹有限 / 单调列〕）：

| 验证项 | 用例行数 | 最大 ulp 差 | 违反率 | 结论 |
|--------|---------|------------|--------|------|
| E1 原域 reduce_max（fp16/fp32）vs fp32 上抛基线 | 160（2 dtype × 7 shape + 角点） | **0** | 0/160 | **EQUIV_PASS** |
| E2 bf16 域不变（bf16 域 max == fp32 域 max） | 77（6 shape + 角点） | **0** | 0/77 | **EQUIV_PASS** |
| E3 分块 running max ≡ 整行 max（含非整除尾 tile） | 228（2 dtype × 9 split + 角点 ×3 尾宽） | **0** | 0/228 | **EQUIV_PASS** |
| E4 -inf 单位元 padding 消除（raw N + 静态尾 tile） | 210（3 dtype × 5 宽度 × 角点） | **0** | 0/210 | **EQUIV_PASS** |
| INFO：NaN 传播一致性（候选 == 基线，等价域外行为记录） | 1 | — | 0（两侧一致传播 NaN） | INFO(pass) |
| **合计** | **675** | — | **0 项失败** | **ALL EQUIV_PASS** |

**优化结论**：采纳 E1–E4 共 4 项（E1/E3/E4 为收益项，E2 为 API 必要性并附等价论证）；优化后公式 = §1.3 的分块 running-max 式（供 §3.1 拆解）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点对象：§1.4 NPU 算法的**全部**循环与标量计算点。替代 API 均有 `docs/Tilelang.language/`、`testing/` 或 `examples/` 佐证（列内标注）。

| # | 计算点 | 原实现形态（源/朴素形态） | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|---------------------------|-------------|---------|--------------|
| 1 | masked 逐元素载入（源步骤 2a） | `T.Parallel(bm,N_padded)` + `T.if_then_else(And(row<M,col<N), …)` 末 tile 全宽逐元素分支 | `T.copy(x[slice], x_ub[slice])`（MTE2 向量搬运，`docs/Tilelang.language/内存操作/T.copy.md`）；静态尾 tile 特化（E4）消除条件分支 | ✅（消除） | — |
| 2 | fp32 上抛 cast（源步骤 2b 全 tile 循环） | `T.Parallel(bm,tile_n)` 全 tile `T.cast(…, f32)` | fp16/fp32：**消除**（E1 原域）；bf16：`T.vcast(x_ub, x_work, round_mode="rint")`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`） | ✅ | — |
| 3 | tile 行极值归约（源步骤 3） | （源已是 reduce 原语；朴素替代为标量扫描） | `T.reduce_max(src, acc/tile_acc, dim=1, …)`（`docs/Tilelang.language/规约操作/T.reduce_max.md`；`testing/npuir/reduction_ops/test_reduce.py` 通过；**同 dtype** 约束） | ✅ | — |
| 4 | **跨 tile running max 合并（源步骤 4，逐行标量 `T.max`）** | `for i in T.Parallel(bm)`：`acc[i] = T.max(acc[i], tile_acc[i])`（每 tile 一次 (bm,) 标量循环） | 单 op `T.reduce_max(src, acc, dim=1, clear=False)`（在 acc 现值上累加 max，`docs/Tilelang.language/规约操作/T.reduce.md` §2.3）；回退 `T.vmax`（`docs/Tilelang.language/比较操作/T.vmax.md`） | ✅（E3） | — |
| 5 | 结果降回 cast（源步骤 5） | `T.Parallel(bm)` `T.cast(acc[i], dtype)` | fp16/fp32：恒等，降为形状收缩；bf16：`T.cast`（f32→bf16） | ✅ | — |
| 6 | (1,1)→(1,) 出栈形状收缩 | — | `for i in T.Parallel(1)`（恒等）+ `T.copy(out_ub[0:1], out[...])` | ❌（保留 Parallel） | **形状收缩类**：`T.reduce` 要求 src/dst rank 一致（`docs/Tilelang.language/规约操作/T.reduce.md`），无法直接产出 (1,)；无「cast+收缩」复合向量 API（vcast 不改 shape，logsumexp single 同款先例）；每 block 1 次发射（1 元素单发向量 op），非逐元素热点 |
| 7 | M 尾块行数 `real_m` / 任务 guard | — | `T.min(1, M − off_m)` 标量 + `if task < M` 动态 guard | ❌（保留标量） | **依赖运行时 pid 的 block 级边界**（已静态化 serial 上界，仅 guard 动态）；每任务 1 次标量，非逐元素热点 |
| 8 | 任务/块索引算术（`task = s·num_kernels+cid`、`off_m = task`） | — | 无（纯 TIR 标量算术） | ❌（保留标量） | **block 级索引 / 任务映射计算**，每任务 1 条，非逐元素热点 |
| 9 | 工厂期分块规划（`num_full`/`valid_n`/`tile_n`） | Python 工厂期计算 | — | ❌（保留） | **host 元数据计算**（不在 kernel 内；每 `(M,N,dtype,tile_n)` 一次） |

**向量化结论**：逐元素计算已全部向量化（#1–#5，API 均有佐证）；保留 4 处标量/循环（#6–#9），理由类别：形状收缩（#6，恒单发向量 op）、动态边界（#7）、block 索引（#8）、host 元数据（#9）——均非逐元素热点（每 block/任务 1 次发射量级），逐项见上表。源实现中唯一的逐元素「非原语」热点（全 tile fp32 cast、masked load 分支、per-tile 标量 max 合并）已分别被 E1/E4/E3 消除。

#### 1.6.3 向量化轴与数据布局决策（阻塞级）

> I/O layout 是契约（`(M,N)` 行主序进 / `(M,)` 出）；**核内布局与 lane 映射是设计变量**。本算子 2D（M,N）——按规约类必选候选枚举（水平归约 vs 垂直扫描为正交决策，须分别枚举）。

**轴质量评分**：

| 评分项 | 水平归约（lane→N 轴，reduce dim=1） | 垂直扫描（lane→M 行，串行步进 N） | 说明 |
|--------|------|------|------|
| 整除性 | N 任意（reduce 原语处理任意宽度；tile_n 取 N 因子或静态尾） | M 任意同理 | 无差异 |
| 尾 lane 浪费率 | 由硬件归约内部吸收（整 tile 归约非逐 lane 映射） | — | 无差异 |
| 累加链形态 | 归约维即向量维（dim=1，stride=1 连续）——reduce 原生形态 ✓ | 归约维变串行步进维（N 次 (1,1) 微操作 × ~0.5µs 发射/元素——发射项灾难） | **决定性差异** |
| repack 代价 | 无（I/O 原生行主序即 N 连续） | 无 | — |
| UB 容量影响 | tile (1, tile_n) staging（§4.5 预算表） | 同 | — |

**候选矩阵**（含源码 GPU 轴选择对照——源码 thread/warp 轴是输入非结论）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|---------|------------|--------------|------|
| 1 | **行 × N-tile 流式 + 水平归约**（lane→N，reduce dim=1；tile 串行，persistent 按 M 分核 bm=1） | N（连续） | 无 | tile ≤64KB/row（§4.5）；向量 op 1–2/tile；`T.reduce_max` dim=1 为快路径（`CONST-reduce-dims0-skinny`：dims=1 快、dims=0 长条逐列串行）；logsumexp tiled/argmax 同族两任务实证 | ✅ **主选** |
| 2 | 垂直扫描（lane→M，串行步进 N 维） | M | 无 | 归约维串行化 → 每行 N 次微操作发射（N=32768 → 数十 ms/行量级，`CONST-vector-launch-overhead`）——发射项灾难 | ❌（量化否决） |
| 3 | 核内转置（N-major）后行归约 | M（转置后连续） | `T.transpose` 二轴交换链 ×2（UB 级） | **纯开销零收益**：主选 #1 的向量化轴 N 即 I/O 原生连续轴、无 repack 需求，核内转置不减少 GM 流量与向量 op 数；另需 +1 个 (1, tile_n) UB 缓冲（吃掉预算裕度）；代价引 pattern-library `layout.md` PL-1.1-transpose-chain 实测（融合转置 ~5.4µs，verified）——即使代价小亦无收益可抵 | ❌ |
| 4 | N-split 跨核 partial+merge（C3） | N（跨核切分） | workspace GM 往返 (M,nchunk)×2 | 小 M 大 N 负载核利用率提升；代价 = 第二 kernel 启动 + workspace 往返 + 合并正确性面 | ⏸ **Stage 4 A/B**（§1.6.4） |
| 5 | 归约源选择：**UB staging 直归约** vs UB→fragment 再归约（同轴 #1 内两种缓冲组织） | N | 无 | 直归约：大缓冲仅 x_ub（fp16 2 B/elem），无额外 copy pass，单消费者无竞态；fragment 形态：+1 个 tile 缓冲（fp16 4 B/elem）但完全对齐 argmax 已验证形态 | ✅ **直归约主选（fp16/fp32）**；bf16 必需 fragment（vcast 目标） |

**布局决策结论**：选定 **#1 + #5**：核内布局 = 按行 `(1, tile_n)` 行主序 staging（UB）、向量化轴 = **N 轴**（`T.reduce_max` dim=1）、repack 无、分核 = persistent 按 M 行（bm=1，§5.5）；该结论同步落入 §3.3 伪代码与 §6 循环结构（归约即 N 轴）。弃选理由：#2 发射爆炸（量化：每行 N 次微操作）；#3 纯开销零收益（主选轴 N 即 I/O 原生连续轴，PL-1.1 实测 ~5.4µs 亦无收益可抵）；#4 为 Stage 4 实验裁决项（§1.6.4）。归约源选 UB 直归约（fp16/fp32，单消费者、无竞态、省 1 缓冲），fragment 形态作为 bf16（vcast 必需）与共享缓冲异常时的回退。

#### 1.6.4 实验裁决备选方案：C3 N-split partial+merge（Stage 4 A/B 三件套）

> §1.6.3-#4 的备选完整结构 + 裁决计划（skill Phase 2 实验裁决模式）。主选 = §1.4 单 kernel 流式（当前证据最优：正确性面最小、无 workspace、无第二 kernel）。触发形态 = 小 M 大 N 负载（如 lm-head 类，(4, 100000)，M 核利用率 4/40）。

**① 分派条件（工厂期静态谓词，per-shape 分派可行性）**：`ceildiv(M, 1) < vector_cores(40) 且 N ≥ 32768`——amax 的 manifest tiled 负载 long-seq (64,32768) 不命中（M=64 > 40）；仅当后续新增小 M 大 N 负载时命中（工厂 `lru_cache` 本就按 `(M,N,dtype,tile_n)` 分派，C3 与主选可按 shape 共存，无全局二选一约束）。

**② 备选结构（两段 kernel，均为 Developer 单 kernel 形态）**：

```python
# ── partial kernel：逻辑块 = 列分片，每块处理 (1, tn) 行链 ──
# tn：tn·B(dtype) ≤ 0.9×64KB；取 N 的 256 倍数整除因子（零尾），否则静态尾 tile
with T.Kernel(min(nchunk, vector_cores), is_npu=True) as (cid, _):     # 极大规模 → persistent
    x_ub = T.alloc_shared((1, tn), dtype)                              # GM staging
    x_w  = T.alloc_shared((1, tn), "float32")                          # 仅 bf16
    m_ub = T.alloc_shared((1, 1), acc_dtype)                           # 分片极值
    for s in T.serial(ceildiv(nchunk, vector_cores)):                  # 静态上界 + guard
        c = cid + s * vector_cores
        if c < nchunk:
            T.copy(x[0:M, c*tn:(c+1)*tn], x_ub[0:M, 0:tn])             # 列块 2D slice
            # fp16/fp32: reduce 于 x_ub ；bf16: vcast(x_ub, x_w) 后 reduce 于 x_w
            T.reduce_max(x_ub, m_ub, dim=1, clear=True)
            T.copy(m_ub[0:M, 0:1], ws_max[0:M, c:c+1])                 # workspace 列切片写出
# ── merge kernel：单块归并 ──
with T.Kernel(1, is_npu=True) as (cid, _):                             # 逻辑 1 ≤ 40，单块
    ws_f = T.alloc_shared((M, nchunk), acc_dtype)
    g    = T.alloc_shared((M, 1), acc_dtype)
    T.copy(ws_max[0:M, 0:nchunk], ws_f)                                # GM workspace → UB
    T.reduce_max(ws_f, g, dim=1, clear=True)                           # 全局极值（行向 dim=1 快路径）
    T.copy(g[0:M, 0:1], out_view)                                      # (M,1)→GM（经 (M,1) 视图 + host reshape）
```

- **workspace（GM，wrapper 层分配）**：`ws_max (M, nchunk)` 计算域 dtype（(4,100000) fp16 实例 nchunk≈13 → 4×13×2 = 104B，往返 ≈208B ≪ x 读流量）。
- **分核三要素（partial）**：逻辑核数 = nchunk；物理核数 = 40（§5.5 实查同源）；规模判定 = 极大规模（nchunk > 40 时）→ persistent 40 + 核内 `T.serial`（静态上界 + guard）。merge：逻辑 1 ≤ 40 → 单块。
- **同步**：两 kernel 内部 Developer 自动同步；kernel 间依赖由 wrapper 顺序发射保证（无核间 sync 需求）。

**③ 实验裁决计划**：代表 shape = 小 M 大 N（如 (4, 100000) fp16/bf16、(2, 262144) fp16）；指标 = bench harness Task Duration（launches≥30 均值，partial+merge 两段之和）且 golden allclose 全 PASS 为前提；**判定阈值：C3 端到端 < 0.8× 主选同 shape 延迟 → 该 shape 类分派翻转**；**回写路径**：实测数据与结论回写本节裁决记录 + `RETROSPECTIVE.md`，翻转时修订 §1.4/§3.3 增补 C3 分派分支（wrapper 双发射 + workspace 分配）并过 Stage 2 复审；Stage 3 实现期如需提前引入，走 `[DESIGN_LIMIT]` 受控修订路由。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（tile 行规约 reduce_max + running max 合并 + 微操作写回），无 matmul、无 Cube | 不需 L1/L0；`alloc_shared`→UB 即可 |
| 归约 | `T.reduce_max`（dim=1，clear/clear=False）+ `T.vcast`（bf16） | Developer 模式可用（`testing/npuir/reduction_ops/` 全部 Developer 先例） |
| 内存层级 | 仅 GM ↔ UB，无跨引擎、无 workspace（C3 才引入 workspace） | Developer 自动管理 |
| 同步 | 单 kernel 内顺序执行 + persistent 任务循环 + tile 串行循环，无核间协作 | Developer 自动同步即可 |
| 家族先例 | logsumexp single/tiled、argreduce 均为 Developer 且 L0 通过 | 沿用 |
| 调用方约束 | harness 迁移 prompt 指定 developer | ✓ |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（GM staging + 计算缓冲；编译器映射 UB）；bf16 的 fp32 工作区可 `alloc_shared`（`T.vcast` 文档示例形态） |
| 计算方式 | `T.reduce_max`（clear=False 累加 / clear=True + vmax 回退）+ `T.vcast`（bf16）+ `T.Parallel(1)` 微循环（形状收缩）；无逐元素标量循环 |
| 同步 | Developer 自动同步（无 `sync_block_set/wait`） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True)` 一维 persistent |

---

## 3. API 映射设计

### 3.1 公式拆解（输入 = §1.6.1 优化后公式）

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_ub ← x[row, t·tn : (t+1)·tn]`（满 tile）；`x_tail_ub ← x[row, num_full·tn : num_full·tn+valid_n]`（静态尾） | GM → UB tile 载入（src/dst 双 slice） |
| 2 | `x_work ← vcast_rint(x_ub)` | **仅 bf16**：bf16→fp32 精确升域；fp16/fp32 跳过（E1 原域） |
| 3 | `a_t = reduce_max_j(域(x_tile))` | tile 行极值（1,1），域 = 原域或 fp32（bf16） |
| 4 | `m ← max(m, a_t)`（初值 -inf） | **跨 tile running max 合并**（`clear=False` 累加；E3） |
| 5 | `out_ub[0] = 域→dtype(m)` | 形状收缩 (1,1)→(1,)；fp16/fp32 恒等，bf16 cast |
| 6 | `out[off_m : off_m+real_m] ← out_ub[0:real_m]` | UB → GM 输出（src/dst 双 slice） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 | 佐证 |
|------|----------|-------------|------|------|------|
| 1 载入 | GM→UB tile | `T.copy(x[row, c0:c1], x_ub[0:1, 0:w])` | 双显式 slice（w 编译期常量） | Developer | `docs/Tilelang.language/内存操作/T.copy.md`；logsumexp tiled；argreduce |
| 2 bf16 升域 | bf16→fp32 | `T.vcast(x_ub, x_work, round_mode="rint")` | round_mode="rint"（bf16→f32 精确） | Developer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 |
| 3 tile 极值 | `max_j` | `T.reduce_max(src, acc/tile_acc, dim=1, clear=…)` | src/dst **同 dtype**；dim=1 | Developer | `docs/Tilelang.language/规约操作/T.reduce_max.md`；`testing/npuir/reduction_ops/test_reduce.py`（shared→shared 同 dtype 通过） |
| 4 running max | `max(m, a_t)` | `T.reduce_max(src, acc, dim=1, clear=False)`（acc 预置 -inf） | 在 dst 现有值上累加（`T.reduce.md` §2.3） | Developer | `docs/Tilelang.language/规约操作/T.reduce.md` §2.3；回退 `T.vmax`（`docs/Tilelang.language/比较操作/T.vmax.md`） |
| 4' init acc | `-inf` | `T.vbrc(-T.infinity("float32"), acc)` | scalar → (1,1) | Developer | `docs/Tilelang.language/shape操作/T.vbrc.md`；flash_attn/logsumexp 先例 |
| 5 出栈收缩 | (1,1)→(1,) | `for i in T.Parallel(1): out_ub[i] = acc[i,0]`（bf16 加 `T.cast`） | 1 元素 Parallel | Developer | logsumexp single；argreduce |
| 6 写出 | UB→GM | `T.copy(out_ub[0:real_m], out[off:off+real_m])` | src 显式截断 | Developer | logsumexp single；argreduce |
| 分核 | persistent | `T.Kernel(num_kernels, is_npu=True) as (cid,_)` + `T.serial(num_local_tasks)` + `if task < M` | 一维 + 静态界 + 动态 guard | Developer | `docs/Tilelang.language/编译器提示操作/T.Kernel.md`；argreduce |

**dtype 路径矩阵**（工厂期分派，**每路径独立 prim_func**——TVMScript 禁止条件 alloc / 条件定义变量，`TRAP-tvm-parser-rules` + argreduce C-3「未用 fragment alloc 不被 DCE」）：

| 输入 dtype | reduce 域 | 预转 pass | `B`（大缓冲 B/elem） | 输出 cast |
|-----------|-----------|-----------|--------------------|-----------|
| float16 | fp16 原域 | 无（E1） | 2（`x_ub`+`x_tail_ub`） | 无（恒等） |
| float32 | fp32 原域 | 无（E1） | 4（`x_ub`+`x_tail_ub`） | 无（恒等） |
| bfloat16 | fp32（vcast） | `T.vcast` rint | 6（`x_ub` 2 + `x_work` 4） | `T.cast` f32→bf16 |

### 3.3 计算伪代码

TIR 原语签名中 dtype 直接作第二位置参数（模板 §3.3 规则）。工厂期全静态量：`num_full`、`valid_n`、`tile_n`、`num_kernels`、`num_local_tasks`。

```python
# 工厂期（Python）
UB_MANUAL_BUDGET = 65536                       # 字节，竞态安全手工预算（§4.5）
B = {"float16": 2, "float32": 4, "bfloat16": 6}[dtype]      # 大缓冲 B/elem
num_full = N // tile_n                         # 满 tile 数（编译期常量）
valid_n  = N - num_full * tile_n               # 静态尾宽（编译期常量；0 = 无尾块）

def _simple_reduce_kernel_tiled(M, N, op_kind, dtype, tile_n):
    # 路由 gate 与 tile_n 选择见 §5.2（本单元承载 tiled 路径；bm 强制 = 1）
    # ── fp16 / fp32（原域，无 cast）──
    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):                                     # K9：无 threads；tiled 路径 block_m = 1
        num_logical = M                                     # bm=1 → 逻辑核数 = M
        vector_cores = 40                                   # §5.5 实查 20 × 2（纯 Vector）
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = (num_logical + num_kernels - 1) // num_kernels
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), dtype)):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub   = T.alloc_shared((1, tile_n), dtype)
                acc    = T.alloc_shared((1, 1), dtype)       # running max（同 dtype）
                out_ub = T.alloc_shared((1,), dtype)
                for s in T.serial(num_local_tasks):                  # 静态上界
                    task = s * num_kernels + cid                     # grid-stride；row = task
                    if task < num_logical:                           # 动态 guard
                        T.vbrc(-T.infinity(dtype), acc)              # acc = -inf（max 单位元）
                        for t in T.serial(num_full):                 # 满 tile（静态上界）
                            T.copy(x[task, t*tile_n:(t+1)*tile_n], x_ub[0:1, 0:tile_n])
                            T.reduce_max(x_ub, acc, dim=1, clear=False)   # running max（E3）
                        # bf16: x_work=(1,tile_n) f32; vcast; reduce 于 x_work；出栈 T.cast
                        for i in T.Parallel(1):
                            out_ub[i] = acc[i, 0]                # (1,1)→(1,) 恒等
                        T.copy(out_ub[0:1], out[task:task+1])
        return main
    return _func
    # ── 静态尾 tile（构建期 valid_n > 0）：独立缓冲 (1, valid_n) + clear=False 合并（无 -inf 填充）──
    #     x_tail_ub = T.alloc_shared((1, valid_n), dtype)
    #     T.copy(x[task, num_full*tile_n : num_full*tile_n + valid_n], x_tail_ub[0:1, 0:valid_n])
    #     T.reduce_max(x_tail_ub, acc, dim=1, clear=False)
    # ── 回退（若 clear=False 跨 serial 携带异常）：per-tile clear=True + no-alias vmax + 复制技巧 ──
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.copy`（GM↔UB、双显式 slice、UB→GM 截断） | `docs/Tilelang.language/内存操作/T.copy.md`；logsumexp tiled；argreduce | ✅ 文档 + 同族 L0 通过 |
| `T.alloc_shared` | `docs/Tilelang.language/内存操作/T.alloc_shared.md`；`testing/npuir/reduction_ops/test_reduce_dev.py`（shared→shared 通过） | ✅ 文档 + 测试 |
| `T.reduce_max(buffer,out,dim,size,clear)` | `docs/Tilelang.language/规约操作/T.reduce_max.md` §2.1/§2.4；`testing/npuir/reduction_ops/test_reduce.py` 通过 | ✅ 文档 + 测试（**同 dtype** 为探针实证约束） |
| `T.reduce`（统一入口 `reduce_mode="max"`，含 `clear=False` 语义） | `docs/Tilelang.language/规约操作/T.reduce.md` §2.1/§2.3（clear=False 在 dst 现有值上累加；fp16/fp32 ✓、bf16 ✗） | ✅ 文档 |
| `T.vmax`（回退合并 op） | `docs/Tilelang.language/比较操作/T.vmax.md` §2.1/§2.2.1（fp16/fp32 ✓、bf16 ✗） | ✅ 文档 |
| `T.vbrc`（acc 初始化 -inf） | `docs/Tilelang.language/shape操作/T.vbrc.md`；flash_attn/logsumexp 先例 | ✅ 文档 + 先例 |
| `T.vcast`（bf16→f32, rint） | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 | ✅ 文档 |
| `T.cast`（f32→bf16，Parallel 内标量） | logsumexp single 先例 | ✅ 先例 |
| `T.Parallel`（微循环读、GM 禁入） | `docs/Tilelang.language/编译器提示操作/T.Parallel.md` | ✅ 文档 |
| `T.min`（尾块行数） | logsumexp single §3.3/§5.4；argreduce | ✅ 同族先例 |
| `T.serial`（小写；`T.Serial` 不存在） | logsumexp tiled §9.1 实证；`T.infinity.md` 示例 | ✅ 实证 |
| `T.Kernel(一维, is_npu=True)` + `T.serial` + 动态 `if` guard | `docs/Tilelang.language/编译器提示操作/T.Kernel.md`；argreduce；`TRAP-threads-kwarg-noop`（threads 无效果，移除） | ✅ 先例 |
| `T.infinity` | `docs/Tilelang.language/创建操作/T.infinity.md` | ✅ 文档 |
| `compute_tile_n`（仓内原语） | `examples/TileOPs/tileops/kernels/reduction/_primitives.py` | ✅ 仓内原语 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 persistent grid（§5.5） |
| 纯 Vector 算子核数翻倍 | Yes | 40 = `get_aicore_num()×2` 实查（§5.5） |
| bf16 不被 reduce/v 前缀 op 支持 | Yes | `T.vcast` bf16→fp32 预转（`T.reduce.md` §2.2.1） |
| reduce 混合 dtype 静默错误 | Yes（探针实证） | 同 dtype src/dst reduce（§3.2 注） |
| multitile + bm>1 编译 SIGSEGV | Yes | **tiled 路径强制 bm=1**（`CG-2026-0014`，§0.6-R2） |
| serial 循环变量条件 `T.if_then_else` segfault | Yes | 静态尾 tile 特化（尾宽编译期常量，§0.6-R1） |
| GEMM 要求 M,N 为 block 整数倍 | No | 无 GEMM；M 非整除用 `T.min` + 双 slice；N 非整除用静态尾 tile |
| L0C 容量上限 | No | 无 Cube 计算 |
| host 侧禁止改动输入真实内容 | Yes | 无 host pad（`_kernel_handles_padding=True`）；全部边界处理在 kernel 内 |
| UB 容量 192KB（auto-multi-buffer 膨胀） | Yes | §4.5 64KB 手工预算（×2 包络 ≤192KB） |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键差异）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 并行 | `T.Kernel(grid, threads=threads)` 两级 | 一维 `T.Kernel(n, is_npu=True)` | threads 去除（K9）；persistent + serial（§0.6-R2） |
| `target` | 无显式 target | `target="npuir"` | `@tilelang.jit(out_idx=[1], target="npuir")` |
| 循环 API | `T.Serial` | `T.serial`（小写） | `T.Serial` 不存在（logsumexp tiled 实证） |
| SMEM tiling | `alloc_shared` 48KB 预算 + 256 对齐 | UB 192KB + 32B 尾轴 | §5.2 预算 + 路由 gate（§0.6-R4） |
| 边界处理 | 256 对齐 pad + 末 tile `T.if_then_else` masked load | 原始 N + 静态尾 tile 特化（无 -inf 填充）+ M 尾块双 slice | §0.6-R1（padding 消除，E4） |
| 精度域 | 全量 fp32 上抛 + 降回 cast | fp16/fp32 原域；bf16 vcast fp32 | §1.6.1-E1/E2 |
| 跨 tile 合并 | `T.Parallel(bm)` 逐行标量 `T.max` | 单 op `T.reduce_max(..., clear=False)`（回退 `T.vmax`） | §1.6.1-E3 |
| 分档 gate | `N_padded > MAX_SINGLE_TILE_COLS(32512)` | `N·B ≤ 64KB`（UB 预算，驻留否则 tiled） | §0.6-R4；tiled 单元由驻留兄弟在超限时路由 |
| `T.fill` 初始化 | `T.fill(acc, -inf)` | `T.vbrc(-T.infinity(...), acc)` | `docs/Tilelang.language/shape操作/T.vbrc.md` |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/amax/_simple_reduce_kernel/DESIGN.md`（+实现） | 极高（**驻留兄弟**，同算子同语义，路由对端） | 重设计 R1–R4、边界三件套、persistent 分核、bf16 vcast、UB 预算、§5.2 路由边界 |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_tiled/DESIGN.md`（+实现） | 高（N>UB 流式、`T.serial` tile 循环、online 递推） | tile 循环结构、online 递推形态、**serial 循环变量条件 segfault 反例 → 静态尾 tile 依据** |
| `examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/_argreduce_kernel_DESIGN.md`（+实现） | 高（persistent 分核、UB 预算、同 dtype reduce、静态尾 tile、tiled bm=1 强制） | 分核公式、边界三件套、`CG-2026-0014` bm=1、`compute_tile_n` 用法 |
| `testing/npuir/reduction_ops/test_reduce.py` / `test_reduce_dev.py` | 高（reduce 原语合法形态） | `T.reduce_max(shared, shared, clear=…)`、同 dtype、row-reduce dim=1 |
| `examples/TileOPs/tileops/kernels/reduction/_primitives.py` | 高（`align_up`/`compute_tile_n`/`ub_slab_units`/`MAX_SINGLE_TILE_COLS`） | NPU 工具与 tile_n 选择口径复用 |
| `examples/TileOPs/tileops/kernels/reduction/amax/_amax_fwd_kernels.py` / `amax.py` | 源/上下文 | 源 kernel 基准 + 已移植 wrapper（K5–K9） |

---
## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(M, N)` | float16 / bfloat16 / float32 | 2D 连续行主序；M、N 为工厂期常量（无动态轴）；raw N（无 host pad）；**N·B(dtype) > 64KB**（超 UB 驻留上限，本单元适用范围） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `(M,)` | same_as(x) | 每行最大值；`out_idx=[1]` 由 jit 分配 |

### 4.3 中间缓冲区（tiled 路径，bm=1）

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | (1, tile_n) | 输入 dtype | UB（`alloc_shared`） | GM staging + 满 tile 归约源（fp16/fp32） |
| `x_tail_ub`（仅 valid_n>0） | (1, valid_n) | 输入 dtype | UB（`alloc_shared`） | 静态尾 tile staging + 归约源 |
| `x_work`（仅 bf16） | (1, tile_n) | float32 | UB（`alloc_shared`） | bf16→fp32 精确域（满 tile 归约源） |
| `x_tail_work`（仅 bf16 且 valid_n>0） | (1, valid_n) | float32 | UB（`alloc_shared`） | bf16 尾 tile fp32 域 |
| `acc` | (1, 1) | 计算域 dtype | UB（`alloc_shared`） | running max（reduce dst，循环携带） |
| `tile_acc`（仅回退路径） | (1, 1) | 计算域 dtype | UB（`alloc_shared`） | 回退 per-tile max |
| `out_ub` | (1,) | 输入 dtype | UB（`alloc_shared`） | 输出 staging |

### 4.4 内存搬运路径

```
纯 Vector 路径（无 Cube/L1/L0 参与；bm=1）：
GM[x] --T.copy(tile slice)--> UB[x_ub]        # 满 tile（num_full 次）
   --[bf16: T.vcast rint]--> UB[x_work f32]
   --T.reduce_max(dim=1, clear=False)--> UB[acc (1,1)]      # running max 携带
GM[x] --T.copy(tail slice valid_n)--> UB[x_tail_ub]         # 静态尾 tile（1 次，valid_n>0 时）
   --[bf16: T.vcast]--> UB[x_tail_work] --T.reduce_max(clear=False)--> UB[acc]
   --T.Parallel(1) [恒等/cast]--> UB[out_ub (1,)]
   --T.copy(slice)--> GM[out]
x 仅读一遍；无 GM 中间往返；无 workspace。

关键约束：GM 不可直达计算缓冲，必须经 UB；所有计算在 UB（`alloc_shared`）上完成；
reduce src/dst 同 dtype（fp16/fp32 路径直达；bf16 经 fp32 工作区）。
```

### 4.5 UB 内存预算

**预算公式（探针定标模型）**：`手工字节预算(row) = tile_n · B(dtype) + 微缓冲 ≤ 64KB`（tiled 路径 bm=1；`B(dtype)` = 大缓冲每元素字节数）。竞态安全实测值：×2 auto-multi-buffer 双缓冲 ≈128KB ≤ 192KB − 8KB reserve。依据链：UB=192KB（`CONST-capacity-910B2C`，反推自 `requires 1572864 bits`，含 auto-multi-buffer ~1.7x 膨胀观测）；手工 64KB → 双缓冲 ≤128KB 的通过点由同族 argmax 探针三点验证（fp16 bm=2@N=4096 = 64KB 手工 PASS；fp16 bm=4@4096 = 128KB 手工 → `requires 2097408 bits` 溢出 FAIL）与 `TRAP-UB-multibuffer-inflation` 一致。

**B/dtype 表**（tiled 路径大缓冲每元素字节，bm=1）：

| dtype | `x_ub` (B/elem) | `x_work` (B/elem) | `x_tail_ub` | `x_tail_work` | **B 合计** | 微缓冲（`acc`+`out_ub`+`tile_acc`） |
|-------|------|------|------|------|------|------|
| float16 | 2 | — | 2 | — | **2** | ≈ 4 字节（可忽略） |
| float32 | 4 | — | 4 | — | **4** | ≈ 8 字节 |
| bfloat16 | 2 | 4 | 2 | 4 | **6** | ≈ 10 字节 |

> 口径：`x_tail_ub`/`x_tail_work` 与 `x_ub`/`x_work` **不同时活跃**（尾 tile 在满 tile 循环后处理，编译器生命周期可重叠）；保守按 max(tile_n, valid_n) = tile_n 计主预算，尾缓冲计入微项。`tile_acc` 仅回退路径分配（不与主路径同时活跃）。

**逐 workload 预算验证**（tiled 路径；tile_n 由 §5.2 推导）：

| workload | dtype | N | tile_n（推导） | 手工预算 tile_n·B | ×2 双缓冲 | 结论（≤184KB 可用） |
|----------|-------|---|-----------|------------------|-----------|---------------------|
| **long-seq (64,32768)** | bf16 | 32768 | 8192（N 因子，零尾） | 48KB | 96KB | ✅ |
| （自测/large-N）(128,65536) | fp16 | 65536 | 16384（N 因子，零尾） | 32KB | 64KB | ✅ |
| （自测/非整除）(4,100000) | fp16 | 100000 | 8192（N 无 ≤32768 的更大 256 倍数因子 → 8192 + 尾 1696） | 16KB | 32KB | ✅ |
| （自测）(2, 262144) | fp16 | 262144 | 32768（= 预算上限，零尾） | 64KB | 128KB | ✅（与 argmax 验证点同包络） |
| （驻留 workload）smoke / hidden-state / 3d | fp16/bf16/fp32 | ≤16384 | — | 本单元不适用 | — | 路由兄弟驻留单元（§5.2） |

> **实现纪律（探针派生，Stage 3 必须遵守）**：① 未用 `alloc` **不被 DCE**（计入 UB 规划）→ fp16/fp32 与 bf16 必须**独立 prim_func** 分派，禁止「无条件 alloc `x_work` + 条件使用」；② 预算按上表 B 计算，宁小勿大（tile_n 降档优先于预算贴线）；③ 若编译报 `ub overflow`，优先降 tile_n 一档；④ 若共享缓冲（`x_ub`）出现数据竞争（auto-multi-buffer 预算超限），回退到「UB staging → fragment 计算」（额外 2/4 B/elem，argmax C-4 已验证形态）。
> **口径声明**：本结构活跃大缓冲数 ≤2（`x_ub`，bf16 加 `x_work`），**轻于** argmax 的 3 大缓冲（8–16 B/elem），故 64KB 手工预算包络对本结构恒成立（未实测假设 + 依据 = 同族 argmax 探针三点 + 结构更简）。

### 4.6 动态轴定义

无（M、N、dtype、tile_n 均工厂期静态；核内全部循环边界静态：`num_full`/`valid_n`/`num_local_tasks` 为 Python 整数，运行时仅 `task` guard 动态）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")   # out 自动分配 (M,)，dtype 同 x
def _func(block_m): ...                       # K9: 无 threads；tiled 路径 block_m=1，由 wrapper config 传入
```

工厂缓存：沿用 harness `ReduceKernel` 的 kernel cache（`(M,N,dtype,tile_n)` 键）+ 源码 `lru_cache` 语义（§0.4-#11 保留）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector。

**判定依据**：全部计算为归约（`T.reduce_max`）+ dtype 转换（`T.vcast`/`T.cast`）+ UB 搬运（`T.copy`），无 matmul；数据通路 GM↔UB；无 Cube/L0/L0C。

### 5.2 Block 划分（tile_n 推导 + 路由 gate）

```python
# 工厂期（全部静态 Python 量）
UB_MANUAL_BUDGET = 65536                              # 字节（§4.5 竞态安全手工预算）
B = {"float16": 2, "float32": 4, "bfloat16": 6}[dtype]      # tiled 大缓冲 B/elem
vector_cores = NPUUtils.get().get_aicore_num() * 2    # 纯 Vector 翻倍（§5.5 实查 = 40）

# ── 路由 gate（wrapper 期，替代 GPU 的 N_padded > MAX_SINGLE_TILE_COLS）──
needs_tiling = (N * B > UB_MANUAL_BUDGET)             # N·B ≤ 64KB → 兄弟驻留单元；否则本单元

# ── tile_n 推导（tiled 路径）──
tile_n_cap = (UB_MANUAL_BUDGET // B) & ~255           # 预算上限，256 对齐（≥ 本设备 32B 尾轴）
candidates = [f for f in divisors(N) if f <= tile_n_cap and f % 256 == 0]
tile_n = max(candidates) if candidates else tile_n_cap   # 整除优先（零尾）；否则上限 + 静态尾 tile
num_full = N // tile_n                                # 满 tile 数（编译期常量）
valid_n  = N - num_full * tile_n                      # 静态尾宽（编译期常量；0 = 无尾块）
assert 2 * tile_n * B <= 184 * 1024                   # 双缓冲包络 sanity（§4.5）

# ── 分核（tiled 路径 bm=1 强制）──
block_m = 1                                           # CG-2026-0014：multitile+bm>1 SIGSEGV
num_logical     = ceildiv(M, block_m)                # = M
num_kernels     = min(num_logical, vector_cores)     # = min(M, 40)
num_local_tasks = ceildiv(num_logical, num_kernels)  # 核内静态任务上界
```

**M 维并行、N 维串行分块**：M 按 `block_m` 分核（tiled 路径 bm=1，即每任务 1 行）；N 在一个 block 内串行分 `num_full` 满 tile + 静态尾 tile。

**路由边界（与兄弟单元接口契约）**：本单元适用条件 = `N·B(dtype) > 64KB`（fp16 `N > 32768`、fp32 `N > 16384`、bf16 `N > 10922`）；反之由兄弟驻留单元 `_simple_reduce_kernel` 承接（整行驻留 UB）。两者共享 `(M,)` 输出契约与 `torch.amax` 语义；wrapper `ReduceKernel.forward` 按 `needs_tiling` 分派到 `npub::reduce_simple_tiled_fwd`（本单元）或 `npub::reduce_simple_fwd`（兄弟）。manifest 的 long-seq (64,32768) bf16 落在此边界内（B=6 → N·B=196608 > 64KB），属本单元；其余 amax 负载（smoke / hidden-state / 3d-multidim）落驻留侧。**GPU 的 `N_padded > MAX_SINGLE_TILE_COLS(32512)` 分档在 NPU 弃用**（NPU 无该上限，改由 UB 预算 gate）。

### 5.3 约束分析

- **对齐约束**：尾轴 32B 对齐由 `T.copy` 切片天然满足（logsumexp N=300 先例）；tile_n 取 256 倍数（≥32B），N 无需 256 对齐（raw-N 契约）。
- **UB 容量**：§4.5 逐 workload 验证（✅/❌）。
- **L0/L1/L0C 容量**：不适用（纯 Vector）。
- **reduce dtype 约束**：src/dst 必须同 dtype（fp16/fp32 直达；bf16 经 fp32 工作区）。
- **循环携带约束**：tile 循环 `acc` 为 carried state，优先用 reduce `clear=False` 累加（官方语义）；回退无别名形式（§0.6-R3）。
- **编译器陷阱约束**：multitile + bm>1 → SIGSEGV（`CG-2026-0014`）；serial 循环变量条件 → segfault（logsumexp tiled 实证）；`T.Parallel` 只读 UB/fragment（不得读 GM）；条件 alloc / 条件定义变量禁止（`TRAP-tvm-parser-rules`）。

### 5.4 注意事项（非整除/边界处理策略）

- **N 非整除（尾 tile）**：`valid_n = N − num_full·tile_n`（编译期常量）；静态尾 tile 特化——独立缓冲 `(1, valid_n)` + `T.copy` + `clear=False` 合并，**无 -inf 填充、无循环变量条件**（规避 segfault）。`valid_n == 0` 时该分支由 Python 构建期跳过（零尾）。
- **N < tile_n**（极端）：`num_full == 0`，全部走静态尾 tile（valid_n = N）；等价于单 tile reduce。
- **M 非整除（`M % block_m ≠ 0`；tiled 路径 bm=1 → 恒整除）**：仍以 `real_m = T.min(1, M − off_m)` + src/dst 双显式 slice 统一处理（删除越界读），输出 `T.copy(out_ub[0:real_m], out[off:off+real_m])`（src 显式截断，src/dst shape 一致，符合 `TRAP-T-copy-region-semantics` 推荐形态）。
- **M=1 / N=1**：M=1 → `num_kernels=1`、单 block；N=1 → 驻留侧（N·B ≤ 64KB，非本单元）；本单元 `verify_equiv.py` (1,1)/(1,2) 用例覆盖数值。
- **M=0**：grid=0 边界（§9.2-R5 记录，harness 测试不覆盖）。
- **全 -inf / 全 +inf / NaN 行**：`-inf`/`+inf` 正常；NaN 传播（§9.2-R1）。

### 5.5 分核策略（物理核数适配）

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 docs/开发指南.md §3.3）。

- **① 逻辑核数计算**：`num_logical = ceil(M / block_m) = M`（tiled 路径 bm=1 强制；纯行并行，N 不分核、在 block 内串行分 tile）。
- **② 物理核数依据（实查）**：查询代码与返回值——
  `from tilelang.utils import NPUUtils; NPUUtils.get().get_aicore_num()` → **20**（2026-10-13 实查，设备名 `Ascend910_9362`，torch 2.7.1+cpu + CANN 8.5.0）。本算子**纯 Vector → 核数翻倍 = 20 × 2 = 40**（依据 core-split-strategy.md §1 要素② 与 ascend-constraints.md §1）。**禁止文档假设值**：本仓 `CONST-aicore-910B2C` 记录的另一设备为 24，与本设备实查值 20 不同——本设计以实查 40 为准，并在 §1.6.0 roofline 标注该设备/核数差异对常数外推的影响。
- **③ 规模判定与分核方案**（统一 persistent 结构：`num_kernels = min(num_logical, 40)`；核内 `T.serial(静态上界 = ceildiv(num_logical, num_kernels))` + grid-stride `task = s·num_kernels + cid` + 动态 guard `if task < num_logical`）：

| workload | dtype | num_logical | 判定 | num_kernels | 核内任务数（静态上界） | 依据 |
|----------|-------|-------------|------|-------------|----------------------|------|
| **long-seq (64,32768)** | bf16 | 64 | **极大规模（64 > 40，bm=1 受 multitile 陷阱钉死）** | **40** | **2**（ceildiv(64,40)） | 固定 40 核 + 核内 `T.serial` 串行（静态边界 2 + guard）；消核启动开销 |
| （自测）(128,65536) | fp16 | 128 | 极大规模（128 > 40） | 40 | 4 | 同上；负载不均（8 核 ×4 + 32 核 ×3，≤1 任务均） |
| （自测）(4,100000) | fp16 | 4 | **逻辑 ≤ 物理（4 ≤ 40）** | 4 | 1 | 无需适配：单波、每核 1 任务；核利用率 4/40 → C3 N-split 动因（§1.6.4） |
| （驻留 workload）smoke / hidden-state / 3d | fp16/bf16/fp32 | — | 路由兄弟驻留单元 | — | — | 本单元不适用（§5.2） |

> 核内串行循环边界全部为**静态值**（工厂期 Python 整数 `ceildiv(num_logical, num_kernels)`）；动态性仅存在于 guard（`if task < num_logical`）。分核参数（tile_n）选择公式见 §5.2。**负载不均说明**：极大规模下 `num_local_tasks` 为 `ceil`，存在「部分核多 1 任务」的固有 1 任务粒度不均（如 64 → 24 核 ×2 + 16 核 ×1）；该不均 ≤1 任务/核，被 persistent 结构摊薄。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block 级 / 任务级 / tile 级调度循环；唯一的 `T.Parallel(1)` 为形状收缩微循环（§1.6.2-#6）。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| M 方向（block 级） | persistent 任务循环 | `T.serial(num_local_tasks)`（静态上界 + grid-stride + guard） | 极大规模核内串行（§5.5）；消核启动开销 |
| N 方向（tile 级） | **串行迭代（循环携带 acc）** | **`T.serial(num_full)`**（小写；`T.Serial` 不存在） | 分块流式，跨 tile running max 有数据依赖，必须串行 |
| N 尾 tile | 静态展开（构建期 `if valid_n>0`） | 顺序语句（非循环，宽编译期常量） | 规避 serial 循环变量条件 segfault；零尾时跳过 |
| tile 归约（N 方向） | 硬件归约 | `T.reduce_max(src, acc, dim=1, clear=False)` | 向量单元行归约（dim=1 快路径，`CONST-reduce-dims0-skinny`）+ running max 累加 |
| 元素级（输出收缩） | 向量化微循环 | `T.Parallel(1)` | (1,1)→(1,) 形状收缩（单发向量 op） |
| dtype 转换 | 向量 op | `T.vcast`（bf16→f32，可选）；`T.cast`（f32→bf16 出栈） | 全缓冲 dtype 转换 |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (cid, _):
    for s in T.serial(num_local_tasks):                # 静态上界
        task = s * num_kernels + cid                   # grid-stride；row = task
        if task < num_logical:                         # 动态 guard
            T.vbrc(-T.infinity(dtype), acc)            # acc = -inf（max 单位元）
            for t in T.serial(num_full):               # 满 tile（静态上界）
                T.copy(x[task, t*tile_n:(t+1)*tile_n], x_ub[0:1, 0:tile_n])
                # [bf16: T.vcast(x_ub, x_work, round_mode="rint")]
                T.reduce_max(src, acc, dim=1, clear=False)     # running max
            if valid_n > 0:                            # 静态尾 tile（构建期常量条件）
                T.copy(x[task, num_full*tile_n:num_full*tile_n+valid_n],
                       x_tail_ub[0:1, 0:valid_n])
                # [bf16: T.vcast(x_tail_ub, x_tail_work, round_mode="rint")]
                T.reduce_max(tail_src, acc, dim=1, clear=False)
            for i in T.Parallel(1):
                out_ub[i] = acc[i, 0]                  # 形状收缩（bf16 加 T.cast）
            T.copy(out_ub[0:1], out[task:task+1])
```

> 注：`if valid_n > 0` 为**构建期 Python 条件**（valid_n 是工厂期常量），非 TIR 运行期分支。

### 6.3 流水线优化

**初始版本不使用 `T.Pipelined`**：tile 循环有循环携带 `acc` 依赖，tile 数据装载与计算重叠由 Developer 的 auto-multi-buffer 自动施加（`x_ub` 单消费者，无跨流水竞态面；前提为 §4.5 预算纪律）。**Stage 4 候选**：① `--enable-auto-multi-buffer=false` + 显式分配（UB 膨胀降至 ~1.10× 时 tile_n 可升档，需重验）；② C3 N-split partial+merge（§1.6.3-#4 / §1.6.4，小 M 大 N 负载）；③ tile 循环 `T.Pipelined` 软件流水（需双缓冲 `x_ub`，可能与 carried acc 交互，收益待测）。

### 6.4 尾块处理

见 §5.4：N 尾 tile 用**静态特化**（编译期 `valid_n`，独立缓冲，无 -inf 填充）；M 尾块用 `real_m = T.min(1, M − off_m)`（tiled bm=1 恒为 1）+ src/dst 双显式 slice + 垃圾行不写出。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer 模式，编译器管理 UB/fragment 依赖与 auto-multi-buffer 流水同步）。

### 7.2 同步点说明

无手动同步点（单 kernel、无核间协作、无跨引擎传输）。任务循环内 MTE2（GM→UB 载入）→ V（reduce/vcast）→ MTE3（UB→GM 写出）的生产者-消费者依赖由编译器自动插入 pipe_barrier；tile 循环内 `acc` 的读-改-写由 reduce op 自身语义 + 编译器迭代间依赖保证；`x_ub` 单消费者（reduce/vcast）无跨流水竞态面（对比 argmax 的 shared 多消费者竞态教训——本结构不涉及）。

### 7.3 pass_configs 配置

无特殊 pass 配置（Developer 默认；`TILELANG_ASCEND_MODE=Developer`）。Stage 4 探索项：`--enable-auto-multi-buffer=false`（§6.3-①）。

---

## 8. 验证方案

### 8.1 Golden 函数

> golden 以 **§0.1 源算子语义**为唯一依据实现（移植 harness 参考 `examples/TileOPs/tests/ops/test_reduce.py` 中的 `ReduceTest.ref_program`），**不复刻 §0.6 的 NPU 算法**（无静态尾 tile / 原域 reduce / persistent 分核结构）——保证验证独立性。

```python
import torch

def golden_amax_tiled(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现（CPU），对应 torch.amax(x, dim=-1)。

    输入 : x of shape (M, N), dtype ∈ {float16, bfloat16, float32}
    输出 : y of shape (M,), dtype = same_as(x)
    """
    return x.float().amax(dim=-1).to(x.dtype)
```

（harness pytest 侧 ref：`ReduceTest.ref_program` 同上；spec 接口 `test_amax_spec_dim` 用 `torch.amax(x.float(), dim=dim, keepdim=keepdim)` 覆盖 dim/keepdim。tiled 与驻留兄弟应对相同输入产出相同结果。）

### 8.2 精度标准

来自 `examples/TileOPs/tests/ops/test_reduce.py` 中的 `_tol`：

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 1e-4 | 1e-4 | 高精度 |
| float16 / bfloat16 | 1e-2 | 1e-2 | 低精度容忍 |

> 本算子是**精确选择运算 + max 结合律合并**：§1.6.1 机器验证（E1–E4）在全部 shape/split/角点上限 **0 ulp**，故实际误差应远优于上表容差；容差按 harness 契约保留（不放松亦不收紧）。

### 8.3 L0 门槛测试计划

L0 聚焦「编译通过 + 基本精度正确」，**必须覆盖大 N 流式（本单元核心用途）**、**静态尾 tile**、**非整除**与**多 dtype**；完整 L1/L2/Boundary 分层套件交由 `tilelang-op-develop` 展开。

| 测试编号 | Shape (M,N) | dtype | tile_n | num_full / valid_n | 验证目标 | 容忍度 |
|---------|-------------|-------|--------|--------------------|----------|--------|
| L0-1 | (64, 32768) | bf16 | 8192 | 4 / 0 | **manifest long-seq**（大 N 流式 + bf16 vcast + 极大规模 persistent 64→40×2） | 1e-2 |
| L0-2a/b | (128, 65536) | fp16 / fp32 | 16384 | 4 / 0 | 大 N 流式 + 极大规模（128→40×4）+ fp32 | 1e-2 / 1e-4 |
| L0-3 | (4, 100000) | fp16 | 8192 | 12 / 1696 | **静态尾 tile**（N 非整除，valid_n=1696）+ 小 M 低核利用率 | 1e-2 |
| L0-4 | (8, 40000) | fp32 | 8192 | 4 / 7232 | 静态尾 tile（fp32）+ 非整除 | 1e-4 |
| L0-5 | (16, 32769) | fp16 | 16384 | 2 / 1 | **极端尾**（valid_n=1） | 1e-2 |
| L0-6 | (2, 262144) | fp16 | 32768 | 8 / 0 | 预算上限 tile_n（64KB 手工预算贴线） | 1e-2 |
| L0-7 | (129, 65536) | fp16 | 16384 | 4 / 0 | **M 尾块**（129 → 4 任务/核不均）+ 大 N | 1e-2 |
| L0-8 | (4, 13000) | bf16 | 8192 | 1 / 4808 | bf16 + 静态尾 tile（1 满 tile + 大尾） | 1e-2 |
| L0-9 | (1, 32769) | fp16 | 16384 | 2 / 1 | M=1 单行 + 极端尾 | 1e-2 |
| L0-10 | 构造行 (M=8,N=32768)：全 −inf / 全 +inf / +inf 夹有限 / −inf 夹有限 / 重复极值 / dtype 极值 | fp16 / fp32 | 8192 | 4 / 0 | **边界值语义**（`verify_equiv.py` 角点集的 NPU 侧复验） | 1e-2 / 1e-4 |

**通过条件**：全部 L0 项通过 atol/rtol 检查；NaN 行为记录（非门禁，§9.2-R1）。**与兄弟单元交叉验证**：对小 N 输入同时跑驻留单元与本单元，结果应一致（非整除尾/分块路径回归）。

### 8.4 完整分层测试套件（交由 tilelang-op-develop 阶段）

- **L1（功能覆盖）**：1D 输入、多维 dim、keepdim=True/False、全部 dtype × 大 N 组合、manifest 全 workload（含路由到本单元者）、多 tile_n 值（8192/16384/32768）。
- **L2（边界）**：M=1、N=1、N=tile_n 恰好 1 满 tile（valid_n=0）、N=tile_n+1（valid_n=1）、N 无 ≤cap 因子、M<40 与 M>40（单波 vs 多波）。
- **Boundary**：极值输入（全 −inf、全 +inf、含 NaN、±0 混合、亚正规）；大 M persistent 多波次回归；tile_n 贴 64KB 预算（`ub overflow` 边界）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 处理方案 |
|------|------|----------|
| **bf16 不被 reduce/v 前缀 op 支持** | `T.reduce_max`/`T.vmax`/`T.vcast` 的 dtype 表 bf16 × | bf16 全程 vcast→fp32 → reduce → cast 回（§1.6.1-E2） |
| **reduce 混合 dtype 静默错误**（探针实证，与 `T.reduce_max.md` §2.4 示例 `dtype≠accum_dtype` 形式矛盾） | fp16 src→fp32 dst 静默产出错误值 | 强制同 dtype src/dst；bf16 用独立 fp32 缓冲而非跨 dtype reduce |
| **multitile + bm>1 编译 SIGSEGV**（`CG-2026-0014`，rc 139 无报错） | tiled 路径 bm>1 直接崩 | tiled 路径**强制 bm=1**（工厂期保证）；小 M 大 N 用 C3 N-split（§1.6.4） |
| **serial 循环变量条件 `T.if_then_else` segfault** | 末 tile masked load 不可移植 | **静态尾 tile 特化**（尾宽编译期常量，无循环变量条件）——logsumexp tiled 实现反例的直接绕法 |
| **循环携带 in-place 状态可能不携带**（`TRAP-vselect-inplace-carried-state`，vselect 形态） | 跨 serial 迭代的别名 carried state 静默错误 | 主选 reduce `clear=False` 累加（官方语义）；回退无别名三 op 链（§0.6-R3） |
| **未用 `alloc` 不被 DCE** | 计入 UB 规划 | fp16/fp32 与 bf16 独立 prim_func（禁止条件 alloc，`TRAP-tvm-parser-rules`） |
| **UB 192KB + auto-multi-buffer 膨胀** | 大 tile_n 探索须按膨胀后预算评估 | §4.5 64KB 手工预算（×2 包络）；编译 `ub overflow` 报文即精确需求 |
| **物理核数因设备而异** | 分核数与 persistent 任务数依赖实查值 | 实查 20 → Vector 40（§5.5）；禁止沿用其他设备文档值（如 24） |
| **`T.vcast` f32→f32 非恒等** | 破坏数据 | 同 dtype 用 `T.copy`，不做 f32→f32 vcast |
| **`T.Parallel` 禁读 GM** | 编译错误 | 所有 GM 读写经 `T.copy`；`T.Parallel` 只操作 UB/fragment |

### 9.2 风险点明细

| # | 风险 | 说明 | 处置 / 状态 |
|---|------|------|------------|
| R1 | **NaN 行为** | `torch.amax` 与 NPU 硬件 reduce 均传播 NaN（argmax 探针实证）——两者一致；NPU 具体位模式须 Stage 3 L0 用 NaN 行实测确认 | 设计期已按「传播」论证（`verify_equiv.py` INFO）；L0 记录；**不阻塞** |
| R2 | **`clear=False` 跨 `T.serial` 迭代携带性** | 官方文档明确「在 dst 现有值上累加」，但未在 serial 循环携带场景实测；同族 `TRAP-vselect-inplace-carried-state` 显示 in-place carried state 可能失效 | 主选 `clear=False`（1 op/tile）；Stage 3 L0-1 首先验证多 tile（num_full=4）结果；异常则回退无别名 `vmax`+复制链（§0.6-R3）；**不阻塞** |
| R3 | **常数设备/版本绑定** | §1.6.0 roofline 的 MTE2/发射常数 origin 为 Ascend910B2C（24 核）；本设备 `Ascend910_9362`（20 核 → Vector 40），估算按核数外推 = **未实测假设 + 估算依据** | Stage 4 按 hardware-cost-model.md §3 回填实测对账；若系统性偏差 >2× 触发 `[DESIGN_LIMIT]` 归因 |
| R4 | **UB 膨胀系数对本结构未独立实测** | 64KB 手工预算包络承自同族 argmax 探针（8–16 B/elem 结构）；本结构大缓冲更少（2–6 B/elem） | 依据充分（更简结构恒不劣于已验证包络）；Stage 3 首编译即以 `ub overflow` 报文核对；风险低 |
| R5 | **M=0（grid=0）** | 非规约维含 0 → 输出 (0,)；源实现未覆盖 | harness 测试不覆盖；列入风险，不阻塞 |
| R6 | **`compute_tile_n` 的 256 对齐与 32B 尾轴** | 尾轴实际只需 32B；用 256 倍数 tile_n 是保守选择（不损正确性，可能牺牲少量预算） | 若需更大 tile_n，可放宽到 32B 倍数（Stage 4 预算探索）；风险低 |
| R7 | **尾块动态 slice 形态** | `TRAP-T-copy-region-semantics`：base+size 形态越界写风险 | 统一用 **src/dst 双显式 slice** 形态（trap 推荐通用形态；logsumexp/argreduce L0 通过） |
| R8 | **输出形状 (1,1)→(1,)** | `T.reduce` 要求 src/dst rank 一致，无 cast+收缩复合向量 API | 单发 `T.Parallel(1)` 收缩（每 block 1 op，非热点）；fragment 形态为回退 |
| R9 | **wrapper 分档 gate 仍为 GPU 语义** | 现有 `amax.py` 的 `_needs_tiling = N_padded > MAX_SINGLE_TILE_COLS` 与 `default_config` 的 48KB 启发式是 GPU 派生 | 本设计给出 NPU 路由 gate（§5.2 `N·B > 64KB`）与 tile_n 推导；wrapper 层 `default_config`/`_needs_tiling` 的 NPU 化属 wrapper/兄弟单元职责，本设计标注接口契约 |
| R10 | **小 M 核利用率** | tiled 路径 bm=1 → 小 M（如 M=4）只用 4/40 核 | 固有（multitile 陷阱），由 C3 N-split 承接（§1.6.4，Stage 4 A/B） |

### 9.3 特殊场景处理

| 场景 | 处理方式 |
|------|----------|
| **N 非 32B 倍数** | 无 padding；`T.copy` 切片保证 32B 尾轴对齐（logsumexp N=300 先例） |
| **N 非 tile_n 整除** | 静态尾 tile 特化（valid_n 编译期常量，独立缓冲，无 -inf 填充） |
| **N < tile_n** | num_full=0，全部走静态尾 tile |
| **M 非 block_m 倍数** | tiled bm=1 恒整除；仍用 `real_m` + 双 slice 统一形态 |
| **N 超 UB 驻留上限** | 本单元（tiled 流式）；N ≤ 驻留上限时由兄弟 `_simple_reduce_kernel` 承接（§5.2 路由） |
| **dim=0 / 多维 dim / dim=None** | Op 层 `flatten_for_multidim`/`movedim` 归一为 `(M,N)`（如 (4,128,4096) dim=[0,2] → M=128,N=16384），kernel 始终沿 dim=1 归约 |
| **全 -inf / 全 +inf 行** | reduce_max 正常返回该值（E4 覆盖） |
| **NaN 行** | 传播 NaN（§9.2-R1 记录） |
| **N=1** | 驻留侧（N·B ≤ 64KB）；本单元数值用例覆盖 |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/amax/_simple_reduce_kernel_tiled/
├── _simple_reduce_kernel_tiled.py   # 算子实现（kernel + golden + L0 测试）+ 与 Golden 比对
├── DESIGN.md                        # 本设计文档
├── verify_equiv.py                  # D-1 等价性机器验证（E1–E4；ALL EQUIV_PASS）
├── RETROSPECTIVE.md                 # Stage 1 复盘（自进化钩子）
└── .stage_state.json                # 编排层状态（不由本阶段维护）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成并执行 | 等价性机器验证（§1.6.1 结果表，exit 0） |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘章节 |
| `_simple_reduce_kernel_tiled.py` | ⬜ 待实现（Stage 3） | 算子实现（kernel 工厂 + golden + L0） |

### 10.3 命名规范

- 项目目录名: `amax`；算子目录名: `_simple_reduce_kernel_tiled`
- 实现文件: `_simple_reduce_kernel_tiled.py`；工厂函数: `_simple_reduce_kernel_tiled(M, N, op_kind, dtype, tile_n)`
- Golden 函数: `golden_amax_tiled(x)`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性机器验证（verify_equiv.py）
2. ⬜ Golden 函数（§8.1）
3. ⬜ 算子实现（`_simple_reduce_kernel_tiled` 工厂 + tile_n 推导/路由 + 静态尾 tile，§3.3/§5.2/§5.5）+ L0 门槛（§8.3）
4. ⬜ 与 Golden 比对（atol/rtol §8.2），全过进入 Stage 4

### 10.5 迁移规则遵守确认

| 规则 | 遵守情况 | 说明 |
|------|----------|------|
| 算子名 = `_simple_reduce_kernel_tiled` | ✅ | 不裁剪不变换 |
| `@tilelang.jit` 函数声明不变 | ✅ | `_func(block_m)` 保留（仅移除 K9 `threads`，加 `target="npuir"`） |
| `@T.prim_func` 参数名/顺序不变 | ✅ | `main(x: T.Tensor((M,N), dtype), out: T.Tensor((M,), dtype))` |
| 从源码推断输入规格 | ✅ | 从 GPU 源码 + manifest 推断，未询问用户 |
| K9：threads 移除 | ✅ | `_func(block_m)` 无 threads |
| Host 侧不改输入 | ✅ | 无 host pad；全部边界处理在 kernel 内 |

---

## 11. 修订日志

**revision_index: 0（首次设计，2026-10-13）**

本版为 `_simple_reduce_kernel_tiled`（AmaxFwdOp 流式/tiled 路径）的首次设计，无历史回退。

**设计核心（可提炼为 4 条 GPU→NPU 关键重设计）**：

1. **边界处理**：256 对齐 pad + 末 tile `T.if_then_else` masked load → **原始 N + 静态尾 tile 特化**（尾宽编译期常量、无 -inf 填充）+ M 尾块双显式 slice（§0.6-R1）。
2. **并行结构**：block×thread + `threads` → 一维 persistent 40 核 + 核内静态 `T.serial`；**tiled 路径强制 bm=1**（`CG-2026-0014` multitile+bm>1 SIGSEGV）（§0.6-R2）。
3. **dtype 域 + 跨 tile 合并**：全量 fp32 上抛 → fp16/fp32 原域 reduce、bf16 `vcast`；per-tile 标量 `T.max` 合并 → **单 op `reduce_max(clear=False)` running max 累加**（§0.6-R3）。
4. **分块分档**：48KB SMEM 启发式 + `MAX_SINGLE_TILE_COLS` 上限 → **UB 预算 tile_n（`tile_n·B ≤ 64KB`）+ 驻留/流式统一路由 gate（`N·B ≤ 64KB`）**，tile_n 除数优先、非整除走静态尾 tile（§0.6-R4）。
