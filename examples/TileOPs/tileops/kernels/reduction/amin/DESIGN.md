# `_simple_reduce_kernel`（AminFwdOp）算子设计文档

> 迁移任务（harness，Developer 模式）：GPU TileOPs `_simple_reduce_kernel`（`op_kind="amin"`）→ TileLang-NPUIR。
> 源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/reduction/reduce.py`（`ReduceKernel` + `_simple_reduce_kernel` 工厂；amin 走「取负 → `T.reduce_max` → 取负回」分支）。
> 本设计基于：源码全文解读（Phase M0）、算法族调研（Phase R）、硬件耦合性判定与 NPU 重设计（Phase M1）、算法级优化设计（Phase 2）、**等价性机器验证**（D-1，`verify_equiv.py`）、**设备 NaN 探针**（`repro/CONST-reduce-min-nan-propagation.py`）。
> 工具链戳：tilelang 0.1.2+417dae9908 + CANN 9.0.0（`ASCEND_HOME_PATH` 实查）+ **Ascend910_9362（20 AICore 实查）** + torch 2.7.1+cpu（2026-10-13 实查，见 §5.5）。
> 本迁移单元范围 = **驻留路径**（N 可整行驻留 UB）。N 超 UB 驻留上限时由兄弟迁移单元 `_simple_reduce_kernel_tiled` 承接（§5.2 路由边界），本设计不实现 tiled 内核。
> amin 是 amax 的**对偶**（同一共享 kernel 的两个 `op_kind` 分支）：本设计复用已冻结胞弟 `examples/amax/_simple_reduce_kernel/DESIGN.md` 的迁移套路（R1 边界 / R2 分核 / R3 dtype 域 / R4 分档），差异仅在 **`T.reduce_min` + `+inf` 单位元** + 源的**取负技巧**替换（§1.6.1-E1）。

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

**数学语义**：对 2D 输入 `x ∈ (M, N)` 沿最后一维（N，规约维）求逐行最小值，输出 `(M,)`：

$$y[i] = \min_{0 \le j < N} x[i,j]$$

即 `torch.amin(x, dim=-1)`（`op_kind="amin"`，manifest `AminFwdOp`）。

**规约语义**：规约轴 = N（Op 层已把任意 rank 输入经 `movedim`/`flatten_for_multidim` 归一为 `(M, N)`，`dim=None` 全张量规约被 flatten 成 `(1, numel)`）；keepdim 后续处理在 Op 层 `_reshape_output`，不进 kernel。min 为 **精确选择运算**——交换律 + 结合律 + 幂等，**无累加顺序问题、无舍入**（这与同族 sum/mean/var 的「累加顺序影响舍入」本质不同，是 §1.6.1-E2 的基础）。

**dtype 语义**：输入 fp16 / bf16 / fp32；输出 `same_as(x)`。源 GPU 实现先把数据 cast 到 fp32（`x_f32` fragment）再规约，结果 `T.cast` 回原 dtype（`out_local`）。

**边界语义**（源码 + torch 语义 + 设备探针共同确定）：
- **±inf**：正常参与比较；全 `+inf` 行 → `+inf`；全 `-inf` 行 → `-inf`（设备探针 `repro/CONST-reduce-min-nan-propagation.py` 行 5/6 实证）。
- **NaN**：`torch.amin` 传播 NaN（含 NaN 的行输出 NaN）；NPU `T.reduce_min` 同样传播 NaN——**两侧逐位一致**（设备探针 8 行 × fp16/fp32 实证，含 NaN 居中/居首/居尾/全 NaN），见 §9.2-R1。
- **±0.0**：IEEE 相等比较；`torch.amin([+0.0, -0.0])` 与「取负取最大再取负」返回同一零位模式（torch 实测均返回 `+0.0`，见 `verify_equiv.py` 角点集），两域一致。
- **空维 / 互补形状**：`N=0` 由 kernel 层构造期拒绝（源码 `N<=0` 无定义）；`M=0` 见 §9.2-R5。
- **非整除**：源码用**单位元填充**处理对齐——`_pad_value_for_op("amin") = +inf`（`+inf` 是 min 的单位元，见 §1.6.1-E4）；pad 列的语义等价由该单位元保证。

**host 侧语义**：Op 层 `_prepare_input` 做 validate →（dim 归一 / `movedim` / `contiguous`）→ reshape 为 `(M, N)`；**`AminFwdOp._kernel_handles_padding = True`**（`examples/TileOPs/tileops/ops/reduction/amin.py` L58），故 Op 层不施加 host 侧 `F.pad`，raw `(M, N)` 直接进入 kernel。reshape/keepdim 均为**实现手段**而非算子语义，golden 不得复刻。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据（`torch.amin`，NaN 传播、±inf、`+inf` 单位元 pad 为文档化边界）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `(M, N)` | float16 / bfloat16 / float32 | 2D 连续行主序；M = 非规约维乘积，N = 规约维长度（源实现内部按 `N_padded = align_up(N, 256)` 处理对齐） |
| `out` | `(M,)` | same_as(x) | 每行最小值（迁移必需字段：输出 shape `(M,)`，无转置布局） |

工厂契约（harness 固定）：`_simple_reduce_kernel(M, N, op_kind, dtype)` 返回 `_func(block_m) → kernel(x)`；语义参数 `M/N/op_kind/dtype` **不变**，后端参数 `threads` 依 K9 去除（NPU 无 CUDA 线程概念）。

### 0.3 实现算法解读（怎么算）

源 kernel（`reduce.py` L90–L147，amin 分支 L130–L136）为**单遍三步**结构：整行载入 → **取负** → `reduce_max` → **取负回** → 出栈。源 amin 不调用 `T.reduce_min`，而是用「取负后取最大再取负」实现 min（与 amax 分支共享 `T.reduce_max` 代码路径）。

**计算步骤分解**（源码每条计算语句均归入步骤，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `T.copy(x[pid_m*block_m, 0], shared_buf)`：block 行块 GM→SMEM 整块载入（对齐路径，L115） | x | shared_buf (bm, N_padded) | 数据载入 |
| 2a | **padded 路径**（`_needs_pad`，L107–L112）：`for i,j in T.Parallel(bm,N_padded)`：`x_f32[i,j] = T.if_then_else(And(pid_m*bm+i<M, j<N), cast(x[...], f32), cast(+inf, f32))`（masked load，越界列填 `+inf`） | x, 常量 +inf | x_f32 (bm, N_padded, f32) | 载入 + 边界（单位元填充） |
| 2b | **aligned 路径**（L118–L119）：`for i,j in T.Parallel(bm,N_padded)`：`x_f32[i,j] = cast(shared_buf[i,j], f32)` | shared_buf | x_f32 (bm, N_padded, f32) | 精度提升（fp32 累加域） |
| 3 | **取负**（L132–L133）：`for i,j in T.Parallel(bm,N_padded)`：`x_f32[i,j] = -x_f32[i,j]` | x_f32 | x_f32 (bm, N_padded, f32) | min 的取负实现技巧（$-\max(-x)=\min(x)$） |
| 4 | `T.reduce_max(x_f32, acc, dim=1)`（L134） | x_f32 | acc (bm,) f32 | $\max_j(-x[i,j])$ |
| 5 | **取负回**（L135–L136）：`for i in T.Parallel(bm)`：`acc[i] = -acc[i]` | acc | acc (bm,) f32 | $\min_j x[i,j] = -\max_j(-x[i,j])$ |
| 6 | `for i in T.Parallel(bm)`：`out_local[i] = T.cast(acc[i], dtype)`（L139–L140） | acc | out_local (bm,) dtype | 结果降回输出 dtype |
| 7 | `T.copy(out_local, out[pid_m*block_m])`（L143） | out_local | GM out | 输出搬运 |

**数据流与内存访问模式**（源硬件视角）：

```
GM[x] --T.copy(或 masked Parallel load)--> SMEM[shared_buf] --Parallel cast--> FRAG[x_f32 (f32)]
  --Parallel negate--> FRAG[x_f32 (f32)] --T.reduce_max(dim=1)--> FRAG[acc (f32)]
  --Parallel negate--> FRAG[acc (f32)] --Parallel cast--> FRAG[out_local (dtype)]
  --T.copy--> GM[out]
```

**循环与并行结构**：`T.Kernel(T.ceildiv(M, block_m), threads=threads)` 一维 grid × CUDA 线程两级并行；N 维不分块（整行驻留 SMEM）；`block_m ∈ {1,2,4,8}` 由 48KB SMEM 预算启发式选择（`SHARED_MEMORY_BUDGET_BYTES // (N_padded × elem)`）；无 `T.Pipelined`。

**host 侧逻辑**：`functools.lru_cache(maxsize=32)` 按 `(M, N, op_kind, dtype)` 缓存编译产物；Op 层 reshape/keepdim 后处理（见 §0.1 host 侧语义）。

**编译配置**：`@tilelang.jit(out_idx=[1])`（GPU target），无特殊 pass_configs。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | SMEM tiling（整行块 T.copy 载入） | 合并访存、一次载入多次使用 | 行块驻留 shared memory，reduce 直接消费 | shared memory + 256 元素对齐的 copy 指令约束 | 硬件强相关 |
| 2 | 双路径边界处理（masked Parallel load vs 对齐 T.copy 快路径） | 非整除/非对齐时保证正确性同时保留快路径 | `_needs_pad = N_padded != N` 时走 `T.if_then_else` 逐元素 masked load，否则走向量化 `T.copy` | GPU SMEM copy 的 256 元素对齐需求 | 硬件相关（对齐约束）；**单位元技巧可移植** |
| 3 | fp32 累加域上抛（`x_f32` fragment） | 数值稳健（对 sum/mean/std 必要） | 全数据 cast fp32 后规约，结果降回原 dtype | — | 对 min **非必要**（精确选择，无舍入），见 §1.6.1-E2 |
| 4 | min 的「取负 → max → 取负回」实现技巧（步骤 3/5） | 复用 `T.reduce_max` 代码路径实现 min | `min(x) = -max(-x)`；两次取负分别作用于全宽 `x_f32` 与 (bm,) 的 `acc` | 源 GPU 路径未使用/未提供直接 min 归约 | **可在 NPU 上等价替换为直接 `T.reduce_min`**（省 2 个 negate pass），见 §0.5 / §1.6.1-E1 |
| 5 | amin 单位元填充（`+inf`） | pad 列不改变结果 | `_pad_value_for_op("amin") = +inf`，min 的单位元 | — | **纯数学恒等，可移植**（承接为 §0.6-R1 / E4） |
| 6 | block_m × threads 两级并行 | 行块并行 + 线程级归约宽度 | 一维 grid × CUDA 线程；threads 同时是归约并行宽度 | CUDA 线程模型 | 硬件强相关（NPU 一维 Kernel 无 threads，K9 去除） |
| 7 | block_m 48KB SMEM 预算启发式（`[1,2,4,8]` 阶梯） | 行块尽量大以摊薄 block 开销 | `SHARED_MEMORY_BUDGET_BYTES // (N_padded × elem)` + 阶梯取最大可行 | SMEM 容量 | 预算值硬件相关（NPU 用 UB 192KB） |
| 8 | `align_up(N, 256)` + `MAX_SINGLE_TILE_COLS=32512` 分档 | 规避 TileLang GPU 向量化器 32768 列上限；>32512 走 tiled 变体 | `_needs_tiling = N_padded > MAX_SINGLE_TILE_COLS` | LLVM scalable-vector 边界 | 硬件强相关（NPU 无此上限，换 UB 容量分档） |
| 9 | `T.reduce_max` 硬件归约原语（承载 min 技巧） | 行内归约用向量单元 | 树形归约（块内并行） | GPU reduce 原语 | NPU 有 Vector `T.reduce_min` 直接等价能力 |
| 10 | `lru_cache` 工厂缓存 | 消重复编译 | Python 层 `(M,N,kind,dtype)` 键 | — | 可移植 |

> **识别完整性**：以上覆盖源码 `_simple_reduce_kernel` 的 amin 分支全部优化性结构（步骤分解表已把每条计算语句归入步骤 1–7，含 2a/2b 双路径与 3/5 两次取负）。无 warp shuffle / mma / swizzle / 异步流水等 GPU 深度优化（源码为朴素单遍法）。**相较 amax 多出 #4「取负技巧」**——这是 amin 最具迁移价值的差异点（可直接等价替换为 NPU 原生 `T.reduce_min`）。
> **源实现缺陷（Stage 3 不得照抄）**：源 L115/L342 的 `T.copy` 无 `M` 尾块守卫（`M % block_m ≠ 0` 时越界读 shared_buf 未初始化行；步骤 6 的 `out_local` 亦按全 `block_m` 写）——本设计以 `real_m = T.min(bm, M−off)` + src/dst 双显式 slice 修正（§0.6-R1 / §5.4）。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| amin 数学语义（行最小值、输出 `(M,)` same_as(x)） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden = `torch.amin` |
| 单遍结构（整行载入 → 行极值 → 出栈） | 算法 | 无（纯算法层） | — | **保留** | 结构保留；实现形态分别等价替换/重设计（下几行） |
| 步骤 4 `T.reduce_max` 行极值 | 算法/API | CUDA 线程归约 | 有：Vector `T.reduce_max`（dim=1） | **等价替换** | `docs/Tilelang.language/规约操作/T.reduce_max.md`；`testing/npuir/reduction_ops/test_reduce.py`（shared→shared、**同 dtype** 形态通过） |
| 步骤 3/5「取负 → max → 取负回」min 实现技巧 | 算法/API | 源 GPU 未直接提供/未使用 min 归约 | **有**：Vector `T.reduce_min`（dim=1，直接支持） | **等价替换** | → §0.6-R5 / §1.6.1-E1：直接用 `T.reduce_min`；`docs/Tilelang.language/规约操作/T.reduce.md` §1 列 `T.reduce_min(src, dst, dims, clear)`；`testing/npuir/reduction_ops/test_reduce.py` L38 `T.reduce_min(c, s, clear=False)` 通过用例；`tilelang/language/customize_npuir.py` `reduce_min(...)`。**同 dtype** 约束同 reduce 族（fp16/fp32 ✓、bf16 ✗） |
| 步骤 2a masked Parallel load + 单位元填充（+inf） | 优化 | GPU 256 元素对齐 | NPU **无需 256 对齐**（32B 尾轴）；`T.if_then_else` 引用循环变量的条件不受支持 | **重新设计** | → §0.6-R1：kernel 接收**原始 N**（无 N_padded、无 masked load），M 尾块用 `T.min` + 双显式 slice |
| 步骤 2b fp32 上抛域（`x_f32`） | 优化/精度 | — | fp16/fp32 可**原域**等价（min 精确选择）；bf16 因 reduce 不支持而必须 cast | **等价替换（重新定域）** | → §0.6-R3 / §1.6.1-E2：fp16/fp32 原域 reduce_min；bf16 `T.vcast→fp32` |
| SMEM tiling + 256 对齐 copy | 优化 | shared memory | 有：UB（`T.alloc_shared`，192KB） | **等价替换** | GM→UB staging；对齐需求降为 32B 尾轴（migration-analysis.md §5.3 行 1） |
| block × thread 两级并行 / `threads` 参数 | 优化 | CUDA 线程 | 无（本项目一维 Kernel） | **重新设计** | → §0.6-R2：一维 persistent grid（纯 Vector 核数 = 实查 aicore×2 = 40）+ 核内 `T.serial` 静态边界任务循环；threads 依 K9 去除（`TRAP-threads-kwarg-noop`） |
| block_m 48KB SMEM 启发式（#6/#7） | 优化 | SMEM 容量 | 换 UB 预算模型 | **重新设计** | → §0.6-R4：UB 64KB 手工预算阶梯（auto-multi-buffer ×2 包络 ≤192KB，argmax 探针定标先例） |
| `align_up(N,256)` + `MAX_SINGLE_TILE_COLS` 分档（#8） | 优化 | GPU 256 对齐 / LLVM 向量化上限 | NPU 无 256 对齐要求；换 UB 容量 gate | **重新设计 / 舍并对齐** | → §0.6-R1/R4：非整除由原始 N 直承（32B 尾轴）；驻留/流式分档改由 UB 预算判定（`bm·N·B(dtype) ≤ 64KB`） |
| 单位元 pad 的「不改变结果」意图（#5） | 优化 | — | — | **保留**（作为等价性依据） | 由 §1.6.1-E4 机器验证承接；NPU 上通过「不 padding」实现同一意图 |
| `lru_cache` 工厂缓存（#10） | 优化 | — | — | **保留** | 工厂层与硬件无关 |

> 每项处置依据：migration-analysis.md §5.3 映射表（shared memory→UB 行 1、warp shuffle 归约→Vector 归约 行 3、block×thread→一维重设计 行 7、online/算法层保留 行 13）+ 本仓 `docs/Tilelang.language/` API 文档 + `testing/npuir/reduction_ops/` 通过用例 + pattern-library 实测条目。无「舍弃后性能意图丢失」项：#2 的对齐意图随 NPU 对齐约束降低而自然消失；#3 的精度意图由 E2（min 无需 fp32 域）承接并在 bf16 上保留 cast；#4 的「复用 reduce_max 路径」意图在 NPU 上无需复用（原生 `T.reduce_min` 更直接），其**计算意图**（求行 min）由 E1 完整承接。

### 0.6 NPU 算法重设计

**重设计项 R1：边界处理（源：host/kernel-side 256 对齐 pad + masked load）**

- **源方案**：`N_padded = align_up(N, 256)`；`_needs_pad` 时用 `T.Parallel(bm, N_padded)` + `T.if_then_else(T.And(row<M, col<N), x, +inf)` 逐元素 masked load（同时处理行尾与列尾），否则 `T.copy` 快路径。性能意图：满足 GPU SMEM copy 的 256 元素对齐，同时把非整除正确性放进 kernel。
- **NPU 新算法**：kernel 直接接收**原始 `(M, N)`**（`AminFwdOp._kernel_handles_padding=True`，Op 层本就不 pad），**取消 N padding 与 masked load**——NPU 尾轴对齐要求仅 32B（`T.copy` 切片天然满足，logsumexp `N=300` 先例），且 `T.if_then_else` 引用循环变量的条件在本工具链不受支持（无 padded 路径可移植）。M 尾块用 `real_m = T.min(block_m, M − off_m)` + src/dst 双显式 slice（`T.copy(x[off:off+real_m, 0:N], x_ub[0:real_m, 0:N])`），垃圾行不写出（logsumexp / argreduce 三件套先例）。
- **语义保持论证**：**消除 padding 不可能改变任何输出**——pad 值为 amin 的单位元 `+inf`（`min(S ∪ {+inf...}) = min(S)`），且源实现中 pad 列本就不参与语义结果；由 §1.6.1-E4 机器验证（含全 +inf / 全 -inf / NaN / 亚正规 / ±inf 行，0 违反）。行尾 `real_m` 仅裁掉不存在的行（每行独立规约），不改变任一有效行的结果。边界核对：全 +inf 行在两种处理下均得 +inf；NaN 行两侧均传播 NaN。

**重设计项 R2：并行结构（源：block × thread 两级 + `threads` 参数）**

- **源方案**：`T.Kernel(T.ceildiv(M, block_m), threads=threads)`；`threads ∈ {128,256}` 同时是归约并行宽度与布局约束参数。
- **NPU 新算法**：纯 Vector **一维 persistent kernel**——`T.Kernel(num_kernels, is_npu=True) as (cid, _)`，`num_kernels = min(num_logical, vector_cores)`，`vector_cores = NPUUtils.get().get_aicore_num() × 2 = 20 × 2 = 40`（纯 Vector 核数翻倍，实查见 §5.5）；核内 `T.serial(静态上界)` 任务循环（grid-stride 映射 `task = s·num_kernels + cid`）+ 动态 guard `if task < num_logical`。行内归约交给 Vector `T.reduce_min`；行块并行交给 grid。`threads` 停止出现（K9）。
- **语义保持论证**：行块之间**完全独立**（per-row 规约，无跨行依赖），任务循环只改变调度、不改变计算；每行结果与 block_m/核数无关（机器验证 E1/E2 在 (1,1)…(33,4096) 全 shape 上限 0 ulp）。尾块 `real_m` + 双 slice 保证越界行不被读、垃圾行不被写。

**重设计项 R3：dtype 计算域（源：全量 fp32 上抛 + cast 回）**

- **源方案**：`x_f32 = cast(x, f32)` 全宽 cast → `negate → reduce_max → negate` → `out_local = cast(acc, dtype)`。
- **NPU 新算法**：**fp16 / fp32 原域 reduce_min**（`T.reduce_min(x_ub, m_ub, dim=1, clear=True)`，src/dst 同 dtype），输出即已为目标 dtype（省去降回 cast）；**bf16 必须 `T.vcast(x_ub, x_work, round_mode="rint")` 升 fp32 后 reduce**（`T.reduce.md`/`T.reduce_max.md` §2.2.1 dtype 表：bf16 ×；fp16/fp32 √），再 `T.cast` 回 bf16。
- **语义保持论证**：min 是**精确选择运算**（无算术、无舍入）——fp16 域 min 选出的元素就是 fp32 域 min 选出的同一个元素（fp16→fp32 为精确单射且保序保等），降回 fp16 为精确还原 ⟹ 结果逐位一致（E2 机器验证：fp16/fp32 原域 vs fp32 上抛基线，0 ulp、0 违反）。bf16：bf16↔fp32 亦为精确单射保序，bf16 域 min ≡ fp32 域 min（E3 机器验证 0 违反）。边界核对：±inf / NaN / 亚正规 / ±0 在单射往返下位模式保持（E2/E3 角点集覆盖）。**注意**：源 L119 的 fp32 cast 对 amin 属数值冗余（对 sum/mean/std 才必要），其「稳健」意图由 E2 的等价论证承接，不构成性能意图丢失。

**重设计项 R4：block_m / 驻留分档（源：48KB SMEM 启发式 + 32512 列上限）**

- **源方案**：`block_m` 由 `SHARED_MEMORY_BUDGET_BYTES(48KB) // (N_padded × elem)` + `[1,2,4,8]` 阶梯选择；`N_padded > MAX_SINGLE_TILE_COLS(32512)` 时切 tiled 变体（LLVM 32768 列向量化边界）。
- **NPU 新算法**：改用 **UB 手工预算阶梯**：`B(dtype)` = 大缓冲每元素字节数（fp16 2 / fp32 4 / bf16 6，见 §4.5）；`block_m = max{p in L : p·N·B(dtype) ≤ 65536}`（L 为预算阶梯集合：基础阶梯 L_base = `{1,2,4,8}`；当逻辑块数 > 2×核数时启用扩展阶梯 L_ext = `{16,...,2048}`，用于小 N 大 M）；`p=1` 仍不满足 → 路由兄弟单元 `_simple_reduce_kernel_tiled`。64KB 为**竞态安全手工预算**（argmax 探针定标：手工 64KB → auto-multi-buffer 双缓冲 ≤128KB ≤192KB−8KB reserve，fp16 bm=2@N=4096 三点 PASS）；`MAX_SINGLE_TILE_COLS` 分档弃用（NPU 无该上限）。
- **语义保持论证**：纯配置选择，不改变任何计算；预算边界由 argmax 同族探针的通过/失败对（fp16 bm=2@4096 = 64KB 手工 PASS；bm=4@4096 = 128KB 手工 → `requires 2097408 bits` 溢出）双向标定；本设计因缓冲更少（无 candidate fp32 全宽缓冲）恒不劣于该已验证包络。

**重设计项 R5：行极值实现（源：取负 → `T.reduce_max` → 取负回 → 修正）**

- **源方案**：步骤 3/5 用两次取负（全宽 `x_f32` 取负 + `(bm,)` 的 `acc` 取负）把 min 表达成 `-max(-x)`，复用 amax 的 `T.reduce_max` 代码路径。
- **NPU 新算法**：直接调用 Vector `T.reduce_min(src, m_ub, dim=1, clear=True)`（src/dst 同 dtype）——`T.reduce_min` 在 NPU 上原生可用（`docs/Tilelang.language/规约操作/T.reduce.md` §1 签名 + `testing/npuir/reduction_ops/test_reduce.py` L38 通过用例），无需取负桥接。
- **语义保持论证**：`min(x) = -max(-x)` 恒等；对 ±inf、亚正规、±0、NaN 取负均为精确位翻转（NaN 取负仍是 NaN），故直接 `reduce_min` 与「取负-max-取负」逐位等价（E1 机器验证：fp16/fp32/bf16 × 7 shape + 角点集，**0 ulp、0 违反**）。边界核对：含 NaN 行两种实现均传播 NaN（设备探针实证 `T.reduce_min` 传播 NaN，与 `torch.amin` 一致）。同时 `clear=True` 将 dst 初始化为 `+inf`（min 单位元，`tilelang/language/customize_npuir.py` `reduce_min` docstring），与 E4 的 `+inf` 单位元语义自洽，不改变结果。

### 0.7 标杆实现

- **源算子**：`/tmp/opencode/TileOPs-ref/tileops/kernels/reduction/reduce.py`（`_simple_reduce_kernel`，L69–L147；amin 分支 L130–L136；§0.1–§0.4 解读对象）；Op 层 `_ReduceOpBase`：`/tmp/opencode/TileOPs-ref/tileops/ops/reduction/reduce.py`；常量/原语：`/tmp/opencode/TileOPs-ref/tileops/kernels/reduction/_primitives.py`。
- **NPU 侧 harness 工件**：提取版 GPU 参考 `examples/TileOPs/tileops/kernels/reduction/amin/_amin_fwd_kernels.py`；已移植 wrapper（Part B，K5–K9）`examples/TileOPs/tileops/kernels/reduction/amin/amin.py`（`npub::reduce_simple_fwd` custom_op + `ReduceKernel`，`threads` 已去除）；Op 层 `examples/TileOPs/tileops/ops/reduction/amin.py`（`_kernel_handles_padding=True`）；规格 `examples/TileOPs/tileops/manifest/reduction.yaml`（`AminFwdOp` 段）。
- **golden 基准**：`examples/TileOPs/tests/ops/test_reduce.py`（`ReduceTest.ref_program` 的 amin 分支 `x_f32.amin(dim=-1).to(x.dtype)`；容差 fp32 1e-4 / 其他 1e-2）。
- **同族已验证参考**：`examples/amax/_simple_reduce_kernel/`（+DESIGN.md——**对偶算子的直接先例**：min↔max、+inf↔-inf、reduce_min↔reduce_max、`T.reduce_min` 的 `clear=True` 初始化 +inf↔-inf）；`examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/`（+DESIGN.md，row-reduction 驻留形态）与 `.../_logsumexp_kernel_tiled/`（+DESIGN.md，N 超驻留的流式形态与静态尾 tile 先例）；`examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/`（`_argreduce_kernel.py` + DESIGN.md——persistent 分核、UB 预算、同 dtype reduce、M 尾块三件套的直接先例）。

---

## 1. 概述

### 1.1 算子名称

`_simple_reduce_kernel`（承载 `AminFwdOp` 的 NPU 驻留式 reduce kernel；工厂接口保留 `op_kind` 以复用兄弟 reduce 算子）

### 1.2 功能描述

对 `(M, N)` 输入沿 N 维（规约维）求逐行最小值，输出 `(M,)`；整行驻留 UB，按 M 行块分核，M 尾块与 N 非对齐均在内处理（无 host pad）。

### 1.3 数学公式

$$
y[i] = \min_{0 \le j < N} x[i,j]
$$

NPU 侧等价计算式（§1.6.1 优化后，供 §3.1 拆解）——按输入 dtype 分派：

$$
m_i = \mathrm{reduce\_min}_{j}\big(x[i,j]\big)\ \ (x\ \text{in native domain};\ \text{fp16/fp32});\qquad
m_i = \mathrm{reduce\_min}_{j}\big(\mathrm{vcast}_{bf16\to fp32}(x[i,j])\big)\ \ (\text{bf16});\qquad
y[i] = m_i
$$

（第一式：fp16/fp32 在**原域**直接归约，且**直接调用 `T.reduce_min`**——不含源实现的两次取负；第二式：bf16 升 fp32 域归约后再降回 bf16。）

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

单遍驻留算法（per-row 独立；与源算法差异逐条标注来源）：

1. **载入**：`T.copy(x[off_m : off_m+real_m, 0:N], x_ub[0:real_m, 0:N])` —— GM 行块 → UB（src/dst 双显式 slice，`real_m = T.min(block_m, M − off_m)`）〔§0.6-R1〕；
2. **dtype 定域**：fp16/fp32 直接用 `x_ub`（原域）；bf16 `T.vcast(x_ub, x_work, round_mode="rint")` 升 fp32〔§0.6-R3 / E2/E3〕；
3. **行极值**：`T.reduce_min(src, m_ub, dim=1, clear=True)`（src 为 `x_ub` 或 bf16 的 `x_work`；`m_ub` (bm,1)，同 dtype；**直接 reduce_min，无取负**）〔§0.5 等价替换 / §0.6-R5 / E1〕；
4. **出栈**：`for i in T.Parallel(block_m)`：`out_ub[i] = m_ub[i,0]`（fp16/fp32，恒等）或 `= T.cast(m_ub[i,0], dtype)`（bf16）；`T.copy(out_ub[0:real_m], out[off_m : off_m+real_m])`〔形状收缩 (bm,1)→(bm,)〕；
5. **调度**：一维 persistent，`num_kernels = min(num_logical, 40)`，核内 `T.serial` grid-stride 任务循环 + guard〔§0.6-R2〕。

**与源算法差异**：① masked-load/padding 边界 → 原始 N + M 尾块 slice（§0.6-R1，源 §0.4-#2/#5 对齐意图自然消失）；② 「取负 → reduce_max → 取负回」→ **直接 `T.reduce_min`**（§0.6-R5，源 §0.4-#4，省 2 个 negate pass）；③ 全量 fp32 上抛 → fp16/fp32 原域、bf16 单独 vcast（§0.6-R3，源 §0.4-#3）；④ block×thread → 一维 persistent（§0.6-R2，源 §0.4-#6）；⑤ 48KB SMEM 启发式 + 32512 分档 → UB 64KB 预算阶梯（§0.6-R4，源 §0.4-#7/#8）。**保留**：单遍「整行载入 → 行极值 → 出栈」结构（§0.5）与行内归约原语（§0.4-#9，由 `reduce_max` 换为 `reduce_min`）。

### 1.5 数据流图

```
驻留路径（纯 Vector，单引擎单 kernel）：
GM[x] --T.copy(slice,real_m)--> UB[x_ub (bm,N) dtype]
   --[bf16: T.vcast]--> UB/FRAG[x_work (bm,N) f32]
   --T.reduce_min(dim=1,clear=True)--> UB[m_ub (bm,1) 计算域]
   --T.Parallel(bm) [cast]--> UB[out_ub (bm,) dtype]
   --T.copy(slice)--> GM[out (M,)]
无 L1/L0 参与；无 GM 中间往返；无 workspace；x 仅读一遍。
```

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：**逐行最小值规约**这一数学语义的算法族（源算法只是基线候选之一）。amin 属规约类 → **完整调研**（R1 候选表 + R3 四口径对比 + R4 逐候选评估）。
> 信息源：skill 参考表 `.agents/skills/tilelang-op-design/references/algorithm-candidates.md`（`ALG-reduction` 行）；同族已冻结先例 `examples/amax/_simple_reduce_kernel/DESIGN.md`（对偶算子调研结论）；`kb_search` 命中（`CASE-reduction-logsumexp` / `CASE-argmax-argreduce-integration` / `CASE-norm-adalayern-stage4` / `CONST-reduce-dims0-skinny` / `TRAP-UB-multibuffer-inflation`）；本仓 `examples/`（logsumexp single/tiled、argmax、`_primitives.py`）；`testing/npuir/reduction_ops/`（含 `T.reduce_min` 的通过用例）；pattern-library（constants/traps/layout/cases）；源算子 §0.3/§0.4；结构判据分析。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1 完成）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| 0 | **基线：GPU 源单遍（取负+reduce_max+取负 + fp32 上抛）** | `y = -reduce_max(cast(-x,f32))`，非整除列填 +inf | — | — | ✅（基线） |
| 1 | **C1 驻留直接 reduce_min 原域单遍** | `y = reduce_min(x)`（fp16/fp32 原域；bf16 先 vcast fp32） | 数学恒等（`min = -max(-x)`；min 为精确选择，无舍入） | 相对基线省 1 全宽 negate pass + 1 (bm,) negate pass + 1 全宽 fp32 cast pass + 2B/elem 缓冲（fp16/fp32）；bf16 保持 1 次 vcast（API 必需） | ✅ |
| 2 | **C2 分块 running-min 流式（online）** | 逐 tile `acc = min(acc, chunk_min)`（tile-0 直写初始化，或 `+inf` 单位元起始） | 数学恒等（min 为交换幂等幺半群） | N 超 UB 驻留时单遍 GM 扫描；中间缓冲 O(N)→O(tile) | ✅（大 N） |
| 3 | C3 跨核 N-split partial+merge | 列分片求 partial min → workspace → merge kernel 求全局 min | 数学恒等（min 结合） | 小 M 大 N 负载核利用率提升；代价 = workspace 往返 + 第二 kernel | ⏸ Stage 4 候选 |
| 4 | C4 全排序取首 | `sort(x)[...,0]` | 数学恒等 | O(N log N) 远大于 O(N) | ❌（复杂度劣） |
| 5 | C5 变换域（log/exp 等） | 对 min 无保序等价变换可降代数 | **无**：min 是完备格上的选择运算，任何单调双射变换都不减少比较次数，非单调变换不等价 | — | ❌（无收益空间） |
| 6 | C6 树形/成对归约重排 | 平衡二叉树比较（与线性扫描同比较次数） | 数学恒等 | FLOPs 同阶；硬件 `T.reduce_min` 已实现块内树形归约 | ❌（硬件原语已覆盖，无额外收益） |
| 7 | C7 `min(x) = -max(-x)` 取负桥接（源实现） | 取负 → reduce_max → 取负回 | 数学恒等 | 相对 C1 **多 2 个 negate pass**（全宽 + (bm,)），无收益 | ❌（C1 严格支配，被 E1 替换） |

> **代数化简空间的结构性结论**：min 与 max 一样是**交换、结合、幂等**运算，任何算法至少要读每个元素一次并做 N−1 次比较（信息论下界）；不存在减少比较次数或 FLOPs 的代数化简（C5/C6 由结构判据排除，非未检索先否定）。故 R1 的可行收益集中在 **min 的实现方式（C1 直接 reduce_min vs C7 取负桥接，E1）**、**dtype 域（C1，E2）**、**扫描/缓冲结构（C2）** 与**跨核组织（C3）**，而非比较次数。

**R2 在线算法**：**有**——min 可分解为 running 统计量（running min），`acc = min(acc, chunk_min)` 为流式合并（结构判据满足；同族 online softmax 的 (max,sum) 对 / logsumexp tiled 已验证同族结构；源 tiled 分支亦已用 `T.min(acc, -tile_acc)` 做跨 tile 累积——源码直接证据）。**但**：amin 基线**本已是单遍**（无 softmax 的 max+sum 两遍结构，不存在「两遍→单遍」的收益），在线变体仅在 **N 超 UB 驻留的 tiled 路径**（C2，分块累加，中间缓冲 O(N)→O(tile)）有意义；驻留路径整行在片上，单遍即最优、无「在线」增量收益。故本单元（驻留）不采纳 online，只为兄弟 tiled 单元提供结构结论（C2 纳入 R3/R4 对比）。

**R3 复杂度对比**（四口径；per-row 计，M 行并行摊派；口径写清含/不含；带宽受限算子——manifest flops = M·N vs bytes = (M·N+M)·elem，访存主导）：

| 算法候选 | FLOPs（op 数，per row） | 访存量 Bytes（GM：读 x + 写 out + 中间往返） | 扫描遍数（GM） | 中间缓冲峰值（per row） | 可并行度 / 跨核代价 |
|---------|------|------|------|------|------|
| 基线（GPU 取负+max+取负 + fp32 上抛） | (N−1) max + N negate + N fp32 cast + 1 acc-negate ≈ **3N** | N·elem + elem（1 遍，**不含** negate/cast 是片上） | 1 | x_f32 4B/elem + shared_buf elem（驻留） | M/bm 行块独立，无跨核 |
| **C1 驻留直接 reduce_min 原域单遍** | (N−1) min（**无 negate、无 cast pass**）≈ **N** | N·elem + elem（1 遍） | 1 | x_ub elem（fp16/fp32 2/4B）；bf16 x_ub 2B + x_work 4B = 6B | 同上 |
| **C2 分块 running-min 流式** | (N−1) + 2·n_tiles（每 tile 1 次合并） | N·elem + elem（1 遍）+ workspace 0 | 1 | O(tile)：x_ub (bm,tile_n)·elem + 极值微缓冲 | 行独立；小 M 时核利用率低（C3 承接） |
| C3 N-split partial+merge | (N−1) + nchunk（merge 比较） | N·elem + elem + workspace 往返 `M·nchunk·(elem+elem)` | **2**（partial 写 + merge 读 workspace；x 仍 1 遍） | partial 工作集 O(tile) + workspace | 跨核 40 核可达；代价 = 第二 kernel 启动 + workspace 往返 |
| C4 全排序 | N log N | N·elem + elem（大中间） | 1 | 排序缓冲远大于 N·elem | O(N log N) 冗余 |
| C7 取负桥接（源） | (N−1) max + N negate + 1 acc-negate ≈ 2N | N·elem + elem（1 遍） | 1 | x_f32 4B/elem | 同上；相对 C1 多 2N−1 op |
| （对照）tiled 两遍式（非 online） | 2N | **2N·elem** + elem | **2** | O(tile) | 同 C2 |

**R4 硬件亲和性评估**（检查清单逐项；实测常数引用 pattern-library/constants.md 条目 + 版本戳；负向淘汰附佐证）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（masked load + 取负桥接形态） | ⚠️ `T.if_then_else` 引用循环变量条件在本工具链不受支持（logsumexp DESIGN §3.5.2；`TRAP-vbrc-scalar-shared` 的 serial 变量条件 segfault 同族）；取负桥接多 2 个 pass；fp32 上抛多 1 全宽 pass | 驻留 SMEM 需 256 对齐 | ❌ 256 元素对齐（GPU copy 约束） | ✅ | ⚠️ | ❌ **形态需重设计**（→C1 + §0.6-R1） |
| **C1 驻留直接 reduce_min 原域单遍** | ✅ 纯 Vector（`T.reduce_min` dim=1；`T.reduce.md`/`T.reduce_max.md` dtype 表 fp16/fp32 ✓、**bf16 ✗→预 cast**；`testing/npuir/reduction_ops/test_reduce.py` L38 通过） | ✅ 大缓冲仅 x_ub（fp16 2 / fp32 4 / bf16 6 B/elem），远轻于 argmax 的 8–16 B/elem；64KB 手工预算包络（`CONST-capacity-910B2C` + argmax 探针定标） | ✅ 32B 尾轴（logsumexp N=300 先例）；N 整行驻留天然覆盖任意宽 | ✅ M/N/dtype 全工厂期常量；核内循环边界静态（仅 guard 动态） | ✅ auto-multi-buffer 由 Developer 自动施加（单消费者缓冲无竞态） | ✅ **驻留路径选定** |
| C2 分块流式 | ✅ 纯 Vector + (bm,1) 微操作 `T.min` | ✅ tile 工作集 ≤64KB | ✅ tile_n 静态；静态尾 tile 特化 | ✅ num_full/tail 静态；**bm=1 强制**（`TRAP`/argmax C-5 multitile bm>1 SIGSEGV） | ✅ tile 循环 auto-multi-buffer | ✅ **tiled 路径选定（兄弟单元）** |
| C3 N-split merge | ✅ Vector | ⚠️ workspace `M·nchunk·2·elem`（amin fp16 lm-head 类约为 KB 级，远小于 x 读流量） | — | ✅ | ⚠️ 两 kernel 启动 + 合并正确性面 | ⏸ **Stage 4 A/B**（小 M 大 N 核利用率） |
| C4 排序 | ✅ | ❌ 排序缓冲爆 UB | — | — | — | ❌ O(N log N) 复杂度劣 |
| C7 取负桥接 | ✅ 可行但多 2 个 negate pass（全宽 + (bm,)） | 同 C1 | ✅ | ✅ | ✅ | ❌ 被 C1 严格支配（E1） |

**调研结论**：选定 **C1（驻留直接 `reduce_min` 原域单遍）为主选**，大 N 由 **C2（分块 running-min 流式）** 在兄弟单元承接。关键依据：R1——相对基线**省 1 全宽 cast pass + 2 个 negate pass**（fp16/fp32 缓冲加倍缩减）且不增加遍数/缓冲；R2——基线本已单遍，online 仅对大 N 有意义；R4——纯 Vector 原语可达（`T.reduce_min` 直接支持 min，无需取负桥接；同 dtype 约束下 reduce 直接消费 static staging）、驻留预算经同族探针定标。与基线的结构差异一句话：**「padded/masked load + 取负桥接 reduce_max + fp32 上抛 + block×thread」→「原始 N + 直接 reduce_min 原域 + 一维 persistent 行块」**。
**调研范围**（"无更优代数化简"的查证范围）：`algorithm-candidates.md` 的 `ALG-reduction` 行 + 对偶算子 `examples/amax/_simple_reduce_kernel/DESIGN.md` §1.6.0 的调研命中条目 + `examples/`（logsumexp single/tiled、argmax、`_primitives.py`）+ `testing/npuir/reduction_ops/`（含 4 个 reduce 通过用例，其一显式用 `T.reduce_min`）+ pattern-library（constants/traps/layout/cases）+ 恒等变形空间自查（C4/C5/C6/C7 全枚举，结构判据/支配关系排除）+ 互联网检索未触发（本地源已覆盖该成熟算法族，无需外部候选）。**源算法优化手段承接**（§0.4）：#2 masked 边界 → R1（单位元意图由 E4 论证保留）；#3 fp32 上抛 → E2（对 min 冗余，意图由等价论证承接）；#4 取负桥接 → R5/E1（计算意图「求行 min」完整承接，桥接本身被支配）；#6/#7 block×thread/SMEM 启发式 → R2/R4；#9 reduce 原语保留（换 reduce_min）；#10 缓存保留。

**设计期 roofline（D-2；协议：hardware-cost-model.md §2；常数带版本戳）**：

| workload（manifest） | 路径/配置 | 流量项 | 发射项 | 估算下界（µs） |
|---|---|---|---|---|
| smoke (32,256) fp16 | 驻留 bm=8, 4 block | 16.1KB ÷ (33.9GB/s·核〔`CONST-mte2-degradation` 1M 档〕× 40 核) ≈ 0.01µs | 4 block × 3 op ÷ 4 核 × 0.5µs〔`CONST-vector-launch-overhead`〕= 1.5µs | **≈1.5（发射主导）** |
| hidden-state (2048,4096) fp16 | 驻留 bm=8, 256 block → 40 核 × 7 任务 | 16.8MB ÷ (30.2 × 40) = 13.9µs | 7 任务 × 3 op × 0.5 = 10.5µs | **≈14（流量主导）** |
| hidden-state (2048,4096) bf16 | 驻留 bm=2, 1024 block → 40 核 × 26 任务 | 16.8MB ÷ (30.2 × 40) = 13.9µs | 26 任务 × 4 op × 0.5 = 52µs（串行上界；流水后实际低于此） | **≈14–52** |
| 3d-multidim (4,128,4096) dim=[0,2] → **M=128, N=16384** fp16 | 驻留 bm=2, 64 block → 40 核 × 2 任务 | 4.19MB ÷ (30.2 × 40) = 3.5µs | 2 任务 × 3 op × 0.5 = 3.0µs | **≈3.5** |
| long-seq (64,32768) bf16 | **路由兄弟 tiled 单元**（B=6 → bm=1 亦超 64KB） | — | — | 本单元不适用（见 §5.2 路由边界） |

> 估算下界: max(流量项, 发射项)（容量项 n/a——纯 Vector 单引擎、无跨引擎 S/P 往返，`CONST-store-fixpipe-gm-only` 不适用）。**amin 相对 amax 少 0 个流量项**（两者 GM 流量相同：读 x + 写 out）——E1/E2 削掉的是**片上 op 数**（negate/cast pass），体现在发射项：本表发射项已按 C1 的 3–4 op/block 计（基线取负桥接 + fp32 上抛为 5–6 op/block，发射项约翻倍）。**设备/常数口径声明**：`CONST-mte2-degradation`（per-core 曲线，origin 工具链 Ascend910B2C 24 核）与 `CONST-vector-launch-overhead`（同源）按「per-core 带宽/发射不变 × 40 核」外推，本设备实查为 **Ascend910_9362（20 AICore → 纯 Vector 40 核）**，属**未实测假设 + 估算依据**（设备/核数差异已标注；Stage 4 回填实测对账，见 hardware-cost-model.md §3）。发射项为串行上界，Developer auto-multi-buffer 流水下实际可低于该值（对偶 amax 同结构 Stage 3 实测 `max_diff=0` 全过，性能口径待 Stage 4）。
#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（C1）。逐项四要素；**等价性机器验证（D-1）**：`verify_equiv.py`（torch CPU，fp64 参照 + dtype 域仿真 + 角点集），结果表内嵌于下。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| E1 | **直接 `T.reduce_min` 替换取负桥接**（承接 §0.6-R5，amin 独有） | `m = -reduce_max(-x)`（源：全宽取负 → max → (bm,) 取负回） | `m = reduce_min(x)` | `min(x) = -max(-x)` 为恒等；对全部 fp 值取负均为精确位翻转（有限 / ±inf / 亚正规 / ±0 / NaN）；故两式逐位相等。min 是**精确选择**运算（比较 + 选择，无算术、无舍入）。**注**：不能跨 dtype reduce（argmax 探针实证 fp16 src→fp32 dst 静默错误），故必须同域 | 省 1 个全宽 `T.Parallel(bm,N)` negate pass + 1 个 `T.Parallel(bm)` acc-negate pass（约 2N op/行）；无需 x_f32 的「原地取负」写回压力。E1 机器验证 240 例 0 ulp |
| E2 | **fp16/fp32 原域 reduce_min**（承接 §0.6-R3） | `m = reduce_min(cast(x,fp32)); y = cast(m, dtype)` | `m = reduce_min(x)`（fp16/fp32 原域，src/dst 同 dtype） | min 是**精确选择**运算（无算术、无舍入）：fp16→fp32 为精确单射且保序保等，两域选出的极值元素相同，模型上 `reduce_min_fp32(x).to(dtype) ≡ reduce_min_dtype(x)` 逐位相等。fp32 输入无上抛 | 省 1 个全宽 fp32 cast pass（N 元素/行）+ 2B/elem fp32 工作缓冲（fp16）；输出即目标 dtype，省降回 cast。**注**：不能跨 dtype reduce，故必须同域 |
| E3 | **bf16 域不变 / 单射往返**（承接 §0.6-R3） | （GPU 原生 bf16 reduce；NPU 无此能力） | `x_work = vcast_rint(x)`（bf16→fp32）→ `m = reduce_min(x_work)` → `cast(m, bf16)` | bf16↔fp32 为精确单射（bf16→fp32 补零位、fp32→bf16 截断仅对可精确表示的 bf16 值恒等）；bf16 域 min ≡ fp32 域 min（同一被选元素），故往返结果 = golden | bf16 路径**必须**引入 1 次 vcast（`T.reduce.md`/`T.reduce_max.md` §2.2.1 bf16 ×）——非优化项而是 API 必要性；无额外数值代价（E3 验证 0 ulp） |
| E4 | **+inf 单位元 padding 消除**（承接 §0.6-R1） | 非整除列填 `+inf` 后 reduce（源 masked load） | 直接对原始 N 列 reduce（无 padding） | `+inf` 是 min 的单位元：`min(S ∪ {+inf,...}) = min(S)` 对所有 `S`（含全 +inf / 含 `-inf` / 含 NaN）成立；源实现中 pad 列本就不进入语义结果 ⟹ 消除 padding 不改变任何输出 | 消除 masked `T.if_then_else` 逐元素分支；省一次对齐 padding 处理；N 任意宽度直承（无 N_padded 缓冲） |
| E5 | **输出降回 cast 消除**（E2 的伴随） | `out_local[i] = cast(acc_f32[i], dtype)`（bm 元素 pass） | fp16/fp32：`out_ub[i] = m_ub[i,0]`（恒等，无 cast）；bf16：`cast`（fp32→bf16，必需） | 同 E2：原域 reduce 的结果已是目标 dtype，fp16/fp32 的「降回 cast」为恒等映射 | 省 fp16/fp32 路径的输出 cast 微 pass（bm 元素）；bf16 保留 |

**机器验证结果表**（`verify_equiv.py`，2026-10-13 执行，exit 0；输入 dtype 域仿真（逐元素 `torch.minimum` / `torch.maximum` 在原 dtype 内的串行归约，不离开输入域）+ fp64 参照；shape 覆盖 (1,1)/(1,2)/(3,17)/(8,256)/(16,1024)/(33,4096)/(4,100)/(M,257 角点) × {fp16,bf16,fp32} + 角点行〔全 +inf / 全 −inf / −inf 夹有限 / +inf 夹有限 / 重复极值 / 亚正规极值 / 混合符号零 / dtype 极值 / 全零 / 全 NaN / NaN 夹有限 / 单调列〕）：

| 验证项 | 用例行数 | 最大 ulp 差 | 违反率 | 结论 |
|--------|---------|------------|--------|------|
| E1 直接 reduce_min vs 源 `-reduce_max(-x)`（fp16/fp32/bf16） | 240（3 dtype × 7 shape + 角点） | **0** | 0/240 | **EQUIV_PASS** |
| E2 原域 reduce_min（fp16/fp32）vs fp32 上抛基线 | 160（2 dtype × 7 shape + 角点） | **0** | 0/160 | **EQUIV_PASS** |
| E3 bf16 域不变（bf16 域 min == fp32 域 min） | 77（6 shape + 角点） | **0** | 0/77 | **EQUIV_PASS** |
| E4 +inf 单位元 padding 消除 | 210（3 dtype × 5 宽度 + 角点） | **0** | 0/210 | **EQUIV_PASS** |
| INFO：NaN 传播一致性（候选 == 基线，等价域外行为记录） | 1 | — | 0（两侧一致传播 NaN） | INFO(pass) |
| **合计** | **687** | — | **0 项失败** | **ALL EQUIV_PASS** |

**设备侧 NaN 传播实证**（`repro/CONST-reduce-min-nan-propagation.py`，2026-10-13，`ASCEND_RT_VISIBLE_DEVICES=1`，exit 0）：shared→shared `T.reduce_min(dim=1, clear=True)` 在 8 行（有限行 / NaN 居中 / 全 NaN / NaN 居首 / NaN 居尾 / 全 +inf / +inf 夹有限 / −inf 夹有限）× {fp16, fp32} 上与 `torch.amin` **逐位一致（numpy 列表完全相同）**——证实 E4 消除 `+inf` padding 不改变 NaN 行，且 `T.reduce_min` 传播 NaN 与 `torch.amin` 一致。

**优化结论**：采纳 E1–E5 共 5 项（E1/E2/E4/E5 为收益项，E3 为 API 必要性并附等价论证）；优化后公式 = §1.3 的三式（供 §3.1 拆解）：fp16/fp32 原域 `y = reduce_min(x)`（**无取负**）；bf16 `y = cast_bf16(reduce_min(vcast_fp32(x)))`。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点对象：§1.4 NPU 算法的**全部**循环与标量计算点。替代 API 均有 `docs/Tilelang.language/`、`testing/` 或 `examples/` 佐证（列内标注）。

| # | 计算点 | 原实现形态（源/朴素形态） | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|---------------------------|-------------|---------|--------------|
| 1 | masked 逐元素载入（源步骤 2a） | `T.Parallel(bm,N_padded)` + `T.if_then_else(And(row<M,col<N), …)` 全宽逐元素分支 | `T.copy(x[slice], x_ub[slice])`（MTE2 向量搬运，`docs/Tilelang.language/内存操作/T.copy.md`）；非整除消除（E4）后无需条件分支 | ✅（消除） | — |
| 2 | fp32 上抛 cast（源步骤 2b 全宽循环） | `T.Parallel(bm,N_padded)` 全宽 `T.cast(…, f32)` | fp16/fp32：**消除**（E2 原域）；bf16：`T.vcast(x_ub, x_work, round_mode="rint")`（`数据类型转换操作/T.vcast.md` §2.2.1 bf16→f32 行） | ✅ | — |
| 3 | **全宽取负循环**（源步骤 3） | `T.Parallel(bm,N_padded)`：`x_f32[i,j] = -x_f32[i,j]` | **消除**（E1 直接 `T.reduce_min`，无需取负桥接）；如确有逐元素取负需求可 `T.vmul(x, -1, out)`（`数学操作/`，本算子不需要） | ✅（消除） | — |
| 4 | 行极值归约（源步骤 4） | （源已是 reduce 原语；朴素替代为标量扫描） | `T.reduce_min(src, m_ub, dim=1, clear=True)`（`规约操作/T.reduce.md` §1 签名；`testing/npuir/reduction_ops/test_reduce.py` L38 通过；**同 dtype** 约束） | ✅ | — |
| 5 | **acc 取负循环**（源步骤 5） | `T.Parallel(bm)`：`acc[i] = -acc[i]` | **消除**（E1 直接 `T.reduce_min` 已得 min） | ✅（消除） | — |
| 6 | 结果降回 cast（源步骤 6） | `T.Parallel(bm)` `T.cast(acc[i], dtype)` | fp16/fp32：恒等，降为形状收缩；bf16：`T.cast`（f32→bf16） | ✅ | — |
| 7 | (bm,1)→(bm,) 出栈形状收缩 | — | `for i in T.Parallel(block_m)` + `T.copy(out_ub[0:real_m], out[...])` | ❌（保留 Parallel） | **形状收缩类**：`T.reduce` 要求 src/dst rank 一致（`规约操作/T.reduce.md` §2.2.2 src (bm,N)→dst (bm,1)），无法直接产出 (bm,)；无「cast+收缩」复合向量 API（vcast 不改 shape，logsumexp single L108–116 同款先例）；每 block 1 次发射（bm 元素单发向量 op），非逐元素热点 |
| 8 | M 尾块行数 `real_m` / 任务 guard | — | `T.min(block_m, M − off_m)` 标量 + `if task < num_logical` 动态 guard | ❌（保留标量） | **依赖运行时 pid 的 block 级边界**（已静态化 serial 上界，仅 guard 动态）；VP-2026-0044 三件套标准形态；每任务 1 次标量，非逐元素热点 |
| 9 | 任务/块索引算术（`task = s·num_kernels+cid`、`off_m = task·block_m`） | — | 无（纯 TIR 标量算术） | ❌（保留标量） | **block 级索引 / 任务映射计算**，每任务 1 条，非逐元素热点；无逐元素等价 API 必要 |
| 10 | host 侧 block_m / 路径选择 | Python 工厂期计算 | — | ❌（保留） | **host 元数据计算**（不在 kernel 内；每 `(M,N,dtype)` 一次） |

**向量化结论**：逐元素计算已全部向量化或消除（#1–#6，API 均有佐证）；保留 4 处标量/循环（#7–#10），理由类别：形状收缩（#7，恒单发向量 op）、动态边界（#8）、block 索引（#9）、host 元数据（#10）——均非逐元素热点（每 block/任务 1 次发射量级），逐项见上表。源实现中的逐元素「非原语」热点（全宽 fp32 cast、masked load 分支、**两次取负**）已分别被 E2/E4/E1 消除。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（`(M,N)` 行主序进 / `(M,)` 出）；**核内布局与 lane 映射是设计变量**。本算子 2D（M,N）——按规约类必选候选枚举（水平归约 vs 垂直扫描为正交决策，须分别枚举）。

**轴质量评分**：

| 评分项 | 水平归约（lane→N 轴，reduce dim=1） | 垂直扫描（lane→M 行，串行步进 N） | 说明 |
|--------|------|------|------|
| 整除性 | N 任意（reduce 原语处理任意宽度；logsumexp N=300 先例） | M 任意同理 | 无差异 |
| 尾 lane 浪费率 | 由硬件归约内部吸收（整行归约非逐 lane 映射） | — | 无差异 |
| 累加链形态 | 归约维即向量维（dim=1，stride=1 连续）——reduce 原生形态 ✓ | 归约维变串行步进维（N 次 (bm,1) 微操作 × ~0.5µs 发射/行——发射项灾难） | **决定性差异** |
| repack 代价 | 无（I/O 原生行主序即 N 连续） | 无 | — |
| UB 容量影响 | 行块 (bm,N) 驻留（§4.5 预算表） | 同 | — |

**候选矩阵**（含源码 GPU 轴选择对照——源码 thread/warp 轴是输入非结论）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|---------|------------|--------------|------|
| 1 | **行块 × 全 N 驻留 + 水平归约**（lane→N，reduce dim=1；persistent 按 M 行块分核） | N（连续） | 无 | 驻留 ≤64KB/block（§4.5）；向量 op 3–4/block；`T.reduce_min` dim=1 为快路径（`CONST-reduce-dims0-skinny`：dims=1 快、dims=0 长条逐列串行）；logsumexp+argmax+对偶 amax 同族实证 | ✅ **主选** |
| 2 | 垂直扫描（lane→M，串行步进 N 维） | M | 无 | 归约维串行化 → 每行 N 次微操作发射（N=4096 → 2ms/行量级，`CONST-vector-launch-overhead`）——发射项灾难 | ❌（量化否决） |
| 3 | 核内转置（N-major）后行归约 | M（转置后连续） | `T.transpose` 二轴交换链 ×2（UB 级） | **纯开销零收益**：主选 #1 的向量化轴 N 即 I/O 原生连续轴、无 repack 需求，核内转置不减少 GM 流量与向量 op 数；另需 +2 个 (bm,N) UB 缓冲（直接吃掉 fp16 bm=8@4096 的预算裕度）；代价引 pattern-library `layout.md` PL-1.1-transpose-chain 实测（4 次融合转置 ~5.4µs，verified，origin_task AvgPool2dFwdOp-optimize-2026-08）——即使代价小亦无收益可抵 | ❌ |
| 4 | N-split 跨核 partial+merge（C3） | N（跨核切分） | workspace GM 往返 (M,nchunk)×2 | 小 M 大 N 负载核利用率提升；代价 = 第二 kernel 启动 + workspace 往返 + 合并正确性面 | ⏸ **Stage 4 A/B**（§6.3；完整备选结构 + 判定阈值 + 回写路径见 §1.6.4） |
| 5 | 归约源选择：**UB staging 直归约** vs UB→fragment 再归约（同轴 #1 内两种缓冲组织） | N | 无 | 直归约：大缓冲仅 x_ub（fp16 2 B/elem），无额外 copy pass，单消费者无竞态；fragment 形态：+1 个 (bm,N) 缓冲（fp16 4 B/elem）但完全对齐 argmax 已验证形态 | ✅ **直归约主选**（fragment 形态为回退，§4.5 纪律） |

**布局决策结论**：选定 **#1 + #5**：核内布局 = 行块 `(bm, N)` 行主序驻留（UB staging）、向量化轴 = **N 轴**（`T.reduce_min` dim=1）、repack 无、分核 = persistent 按 M 行块（§5.5）；该结论同步落入 §3.3 伪代码与 §6 循环结构（归约即 N 轴）。弃选理由：#2 发射爆炸（量化）；#3 纯开销零收益（主选轴 N 即 I/O 原生连续轴，PL-1.1 实测 ~5.4µs 亦无收益可抵）；#4 为 Stage 4 实验裁决项（§1.6.4）。归约源选 UB 直归约（单消费者、无竞态、省 1 缓冲），fragment 形态作为共享缓冲出现任何异常时的回退（§4.5 纪律）。

#### 1.6.4 实验裁决备选方案：C3 N-split partial+merge（Stage 4 A/B 三件套）

> §1.6.3-#4 的备选完整结构 + 裁决计划（skill Phase 2 实验裁决模式）。主选 = §1.4 单 kernel 驻留（当前证据最优：正确性面最小、无 workspace、无第二 kernel）。触发形态 = 小 M 大 N 负载（如 lm-head 类）核利用率低。

**① 分派条件（工厂期静态谓词，per-shape 分派可行性）**：`M ≤ 16 且 ceildiv(M, bm) < vector_cores(40) 且 N ≥ 16384`——amin 的 manifest 负载均不命中（smoke M=32、hidden-state M=2048、3d M=128）；仅当后续新增 lm-head 类负载时命中（工厂 `lru_cache` 本就按 `(M,N,dtype)` 分派，C3 与主选可按 shape 共存，无全局二选一约束）。

**② 备选结构（两段 kernel，均为 Developer 单 kernel 形态）**：

```python
# ── partial kernel：逻辑块 = 列分片，每块处理 (M, tn) ──
# tn：p·tn·B(dtype) ≤ 0.9×64KB（B：fp16 2 / bf16 6 / fp32 4）；取 N 的 256 倍数整除因子
with T.Kernel(min(nchunk, vector_cores), is_npu=True) as (cid, _):     # 极大规模 → persistent
    x_ub = T.alloc_shared((M, tn), dtype)                              # GM staging
    x_w  = T.alloc_shared((M, tn), "float32")                          # 仅 bf16
    m_ub = T.alloc_shared((M, 1), acc_dtype)                           # 分片极值
    for s in T.serial(ceildiv(nchunk, vector_cores)):                  # 静态上界 + guard
        c = cid + s * vector_cores
        if c < nchunk:
            T.copy(x[0:M, c*tn:(c+1)*tn], x_ub[0:M, 0:tn])             # 列块 2D slice
            # fp16/fp32: reduce 于 x_ub ；bf16: vcast(x_ub, x_w) 后 reduce 于 x_w
            T.reduce_min(x_ub, m_ub, dim=1, clear=True)
            T.copy(m_ub[0:M], ws_min[0:M, c])                          # workspace 列切片写出
# ── merge kernel：单块归并 ──
with T.Kernel(1, is_npu=True) as (cid, _):                             # 逻辑 1 ≤ 40，单块
    ws_f = T.alloc_shared((M, nchunk), acc_dtype)
    g    = T.alloc_shared((M, 1), acc_dtype)
    T.copy(ws_min[0:M, 0:nchunk], ws_f)                                # GM workspace → UB
    T.reduce_min(ws_f, g, dim=1, clear=True)                           # 全局极值（行向 dim=1 快路径）
    T.copy(g[0:M, 0:1], out_view)                                      # (M,1)→GM（经 (M,1) 视图 + host reshape）
```

- **workspace（GM，wrapper 层分配）**：`ws_min (M, nchunk)` 计算域 dtype（fp16 lm-head 类实例 M=4、nchunk≈80 → 4×80×2 = 640B，往返 ≈1.3KB 远小于 x 读流量）。
- **分核三要素（partial）**：逻辑核数 = nchunk；物理核数 = 40（§5.5 实查同源）；规模判定 = 极大规模（nchunk > 40）→ persistent 40 + 核内 `T.serial`（静态上界 + guard）。merge：逻辑 1 ≤ 40 → 单块。
- **同步**：两 kernel 内部 Developer 自动同步；kernel 间依赖由 wrapper 顺序发射保证（无核间 sync 需求）。

**③ 实验裁决计划**：代表 shape = 小 M 大 N（如 (4, 16384) fp16/bf16）；指标 = bench harness Task Duration（launches≥30 均值，partial+merge 两段之和）且 golden allclose 全 PASS 为前提；**判定阈值：C3 端到端 < 0.8× 主选同 shape 延迟 → 该 shape 类分派翻转**；**回写路径**：实测数据与结论回写本节裁决记录 + `RETROSPECTIVE.md`，翻转时修订 §1.4/§3.3 增补 C3 分派分支（wrapper 双发射 + workspace 分配）并过 Stage 2 复审；Stage 3 实现期如需提前引入，走 `[DESIGN_LIMIT]` 受控修订路由。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（整行规约 reduce_min + 微操作写回），无 matmul、无 Cube | 不需 L1/L0；`alloc_shared`→UB 即可 |
| 归约 | `T.reduce_min`（dim=1）+ `T.vcast`（bf16） | Developer 模式可用（`testing/npuir/reduction_ops/` 全部 Developer 先例，含 `T.reduce_min` 用例） |
| 内存层级 | 仅 GM ↔ UB，无跨引擎、无 workspace | Developer 自动管理 |
| 同步 | 单 kernel 内顺序执行 + persistent 任务循环，无核间协作（C3 才引入跨核 merge） | Developer 自动同步即可 |
| 家族先例 | amax（对偶）、logsumexp single/tiled、argreduce 均为 Developer 且通过 | 沿用 |
| 调用方约束 | harness 迁移 prompt 指定 developer | ✓ |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（GM staging + 计算缓冲；编译器映射 UB）；bf16 的 fp32 工作区可 `alloc_shared`（`T.vcast` 文档 §2.4 示例 2 形态） |
| 计算方式 | `T.reduce_min` + `T.vcast`（bf16）+ `T.Parallel` 微循环（形状收缩）；无逐元素标量循环 |
| 同步 | Developer 自动同步（无 `sync_block_set/wait`） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True)` 一维 persistent |

---

## 3. API 映射设计

### 3.1 公式拆解（输入 = §1.6.1 优化后公式）

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_ub ← x[off_m : off_m+real_m, 0:N]` | GM → UB 行块载入（src/dst 双 slice；real_m 尾块） |
| 2 | `x_work ← vcast_rint(x_ub)` | **仅 bf16**：bf16→fp32 精确升域；fp16/fp32 跳过（E2 原域） |
| 3 | `m_i = reduce_min_j(域(x)[i,j])` | 行极值（bm,1），域 = 原域或 fp32（bf16）；**直接 reduce_min，无取负**（E1） |
| 4 | `out_ub[i] = 域→dtype(m_i)` | 形状收缩 (bm,1)→(bm,)；fp16/fp32 恒等，bf16 cast |
| 5 | `out[off_m : off_m+real_m] ← out_ub[0:real_m]` | UB → GM 输出（src/dst 双 slice） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 | 佐证 |
|------|----------|-------------|------|------|------|
| 1 载入 | GM→UB 行块 | `T.copy(x[off:off+real_m, 0:N], x_ub[0:real_m, 0:N])` | 双显式 slice | Developer | `docs/Tilelang.language/内存操作/T.copy.md`；logsumexp single；argreduce L238 |
| 2 bf16 升域 | bf16→fp32 | `T.vcast(x_ub, x_work, round_mode="rint")` | round_mode="rint"（bf16→f32 精确） | Developer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（bf16→f32 rint） |
| 3 行极值 | `min_j` | `T.reduce_min(src, m_ub, dim=1, clear=True)` | src/dst **同 dtype**；dim=1；clear=True 将 dst 初始化为 **+inf**（min 单位元） | Developer | `docs/Tilelang.language/规约操作/T.reduce.md` §1/§2.2.1/§2.2.2、`.../T.reduce_max.md` §2.2.1；`testing/npuir/reduction_ops/test_reduce.py`（shared→shared 同 dtype 通过，含 `T.reduce_min`） |
| 4 出栈收缩 | (bm,1)→(bm,) | `for i in T.Parallel(block_m): out_ub[i] = m_ub[i,0]`（bf16 加 `T.cast`） | bm 元素 Parallel | Developer | logsumexp single L108–109；argreduce L252–253 |
| 5 写出 | UB→GM | `T.copy(out_ub[0:real_m], out[off:off+real_m])` | src 显式截断 | Developer | logsumexp single L116；argreduce L254 |
| 分核 | persistent | `T.Kernel(num_kernels, is_npu=True) as (cid,_)` + `T.serial(num_local_tasks)` + `if task < num_logical` | 一维 + 静态界 + 动态 guard | Developer | `docs/Tilelang.language/编译器提示操作/T.Kernel.md`；argreduce L224–237 |

**dtype 路径矩阵**（工厂期分派，**每路径独立 prim_func**——TVMScript 禁止条件 alloc / 条件定义变量，`TRAP-tvm-parser-rules` + argreduce C-3「未用 fragment alloc 不被 DCE」）：

| 输入 dtype | reduce 域 | 预转 pass | `B`(大缓冲 B/elem) | 输出 cast |
|-----------|-----------|-----------|--------------------|-----------|
| float16 | fp16 原域 | 无（E2） | 2（`x_ub`） | 无（恒等） |
| float32 | fp32 原域 | 无（E2） | 4（`x_ub`） | 无（恒等） |
| bfloat16 | fp32（vcast） | `T.vcast` rint | 6（`x_ub` 2 + `x_work` 4） | `T.cast` f32→bf16 |

### 3.3 计算伪代码

TIR 原语签名中 dtype 直接作第二位置参数（模板 §3.3 规则）。工厂期全静态量：`block_m` 阶梯、`num_logical/num_kernels/num_local_tasks`。

```python
# 工厂期（Python）
UB_MANUAL_BUDGET = 65536                       # 字节，竞态安全手工预算（§4.5）
B = {"float16": 2, "float32": 4, "bfloat16": 6}[dtype]      # 大缓冲 B/elem

def _simple_reduce_kernel(M, N, op_kind, dtype):
    # 档位选择与路径切换见 §5.2（本设计承载驻留路径）
    # ── fp16 / fp32（原域，无 cast、无取负）──
    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):                                     # K9：无 threads
        num_logical = (M + block_m - 1) // block_m
        vector_cores = 40                                   # §5.5 实查 20 × 2（纯 Vector）
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = (num_logical + num_kernels - 1) // num_kernels
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), dtype)):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub   = T.alloc_shared((block_m, N), dtype)
                m_ub   = T.alloc_shared((block_m, 1), dtype)
                out_ub = T.alloc_shared((block_m,), dtype)
                for s in T.serial(num_local_tasks):                  # 静态上界
                    task = s * num_kernels + cid                     # grid-stride
                    if task < num_logical:                           # 动态 guard
                        off_m  = task * block_m
                        real_m = T.min(block_m, M - off_m)
                        T.copy(x[off_m:off_m + real_m, 0:N], x_ub[0:real_m, 0:N])
                        T.reduce_min(x_ub, m_ub, dim=1, clear=True)   # 原域、直接 min
                        for i in T.Parallel(block_m):
                            out_ub[i] = m_ub[i, 0]                    # (bm,1)→(bm,) 恒等
                        T.copy(out_ub[0:real_m], out[off_m:off_m + real_m])
        return main
    return _func
    # ── bf16（独立 prim_func）：x_work 为 fp32；reduce_min 于 x_work；出栈 T.cast 回 bf16 ──
    #    其余结构同上；B=6 ⇒ block_m 档位更低（§5.2）
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.copy`（GM↔UB、双显式 slice、UB→GM 截断） | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2/§2.4；logsumexp single L79–82/L116；argreduce L238/L254 | ✅ 文档 + 同族通过 |
| `T.alloc_shared` | `docs/Tilelang.language/内存操作/T.alloc_shared.md`；`testing/npuir/reduction_ops/test_reduce.py`（`a=alloc_shared; T.reduce_min(c, s)` shared→shared 通过） | ✅ 文档 + 测试 |
| `T.reduce_min(buffer,out,dim,size,clear)` | `docs/Tilelang.language/规约操作/T.reduce.md` §1 签名/§2.2.1 dtype 表/§2.2.2 shape；`tilelang/language/customize_npuir.py` `reduce_min`（clear=True 初始化为 inf）；`testing/npuir/reduction_ops/test_reduce.py` L38 通过 | ✅ 文档 + 测试；**设备探针实证 NaN 传播**（`repro/CONST-reduce-min-nan-propagation.py`） |
| `T.reduce_max`（对偶参考，本设计不用） | `docs/Tilelang.language/规约操作/T.reduce_max.md` | ✅ 文档（amin 不调用） |
| `T.vcast`（bf16→f32, rint） | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1/§2.4 示例 2 | ✅ 文档 |
| `T.cast`（f32→bf16，Parallel 内标量） | logsumexp single L108–109 先例 | ✅ 先例 |
| `T.Parallel`（UMB 微循环读、GM 禁入） | `docs/Tilelang.language/编译器提示操作/T.Parallel.md` | ✅ 文档 |
| `T.min`（尾块行数） | logsumexp single §3.3/§5.4；argreduce L237 | ✅ 同族先例 |
| `T.Kernel(一维, is_npu=True)` + `T.serial` + 动态 `if` guard | `docs/Tilelang.language/编译器提示操作/T.Kernel.md`；argreduce L224–237；`TRAP-threads-kwarg-noop`（threads 无效果，移除） | ✅ 先例 |
| `T.infinity` | `docs/Tilelang.language/创建操作/T.infinity.md`（fp16/fp32 ✓） | 可用（本设计**未使用**——padding 消除 + `reduce_min` clear 内置 +inf；保留为回退） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 persistent grid（§5.5） |
| 纯 Vector 算子核数翻倍 | Yes | 40 = `get_aicore_num()×2` 实查（§5.5） |
| bf16 不被 reduce 支持 | Yes | `T.vcast` bf16→fp32 预转（`T.reduce.md` §2.2.1） |
| reduce 混合 dtype 静默错误 | Yes（探针实证，同族） | 同 dtype src/dst reduce（§3.2 注） |
| GEMM 要求 M,N 为 block 整数倍 | No | 无 GEMM；M 非整除用 `T.min` + 双 slice；N 整行驻留 |
| L0C 容量上限 | No | 无 Cube 计算 |
| host 侧禁止改动输入真实内容 | Yes | 无 host pad（`_kernel_handles_padding=True`）；全部边界处理在 kernel 内 |
| UB 容量 192KB（auto-multi-buffer 膨胀） | Yes | §4.5 64KB 手工预算（×2 包络 ≤192KB） |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键差异）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 并行 | `T.Kernel(grid, threads=threads)` 两级 | 一维 `T.Kernel(n, is_npu=True)` | threads 去除（K9）；persistent + serial（§0.6-R2） |
| `target` | 无显式 target | `target="npuir"` | `@tilelang.jit(out_idx=[1], target="npuir")` |
| min 实现 | 取负 → `T.reduce_max` → 取负回 | 直接 `T.reduce_min` | §0.6-R5 / E1 |
| SMEM tiling | `alloc_shared` 48KB 预算 + 256 对齐 | UB 192KB + 32B 尾轴 | §5.2 预算阶梯（§0.6-R4） |
| 边界处理 | 256 对齐 pad + `T.if_then_else` masked load | 原始 N + `real_m` 双 slice | §0.6-R1（padding 消除，E4） |
| 精度域 | 全量 fp32 上抛 + 降回 cast | fp16/fp32 原域；bf16 vcast fp32 | §1.6.1-E2/E3 |
| 单位元 | `_pad_value_for_op("amin") = +inf` | `reduce_min` clear=True 内置 +inf 初始化 | `tilelang/language/customize_npuir.py` `reduce_min` docstring；E4 |
| 分档 gate | `N_padded > MAX_SINGLE_TILE_COLS(32512)` | `bm·N·B ≤ 64KB`（UB 预算） | §0.6-R4；超限路由兄弟 tiled 单元 |
| `T.fill` 初始化 | acc 需 +inf 初始化（tiled 路径源码 L186–187） | `clear=True` 默认置 +inf，无需 fill | `reduce_min` docstring（未初始化 dst 必须 clear=True） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/amax/_simple_reduce_kernel/`（+DESIGN.md） | **对偶（最高）** | min↔max / +inf↔-inf / `reduce_min`↔`reduce_max` / `clear=True` 置 +inf↔-inf；迁移套路（R1/R2/R3/R4）逐条对偶 |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/`（+DESIGN.md） | 高度相似（同族 row-reduction、Developer、`alloc_shared`、UB→GM 尾块切片） | Kernel 结构、JIT 形态、尾块 `T.min` + 双 slice、L0 分层测试 |
| `examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/_argreduce_kernel.py`（+DESIGN.md） | 高度相似（persistent 分核、UB 预算、同 dtype reduce、`real_m` 三件套、bf16 vcast） | 分核公式、block_m 阶梯、边界三件套、探针定标常数 |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_tiled/`（+DESIGN.md） | 中（N 超驻留的流式形态——本设计的**路由目标**） | 静态尾 tile、tile 循环（兄弟单元 `_simple_reduce_kernel_tiled` 的参考） |
| `testing/npuir/reduction_ops/test_reduce.py` / `test_reduce_dev.py` | 高（reduce 原语合法形态，含 `T.reduce_min`） | `T.reduce_min(shared, shared, clear=…)`、同 dtype、row-reduce dim=1 |
| `examples/TileOPs/tileops/kernels/reduction/_primitives.py` | 高（`align_up`/`ub_slab_units`/`UB_SAFETY_RESERVE_BYTES`/`compute_tile_n`） | NPU 工具与本算子预算口径复用 |
| `examples/TileOPs/tileops/kernels/reduction/amin/_amin_fwd_kernels.py` / `amin.py` | 源/上下文 | 源 kernel 基准 + 已移植 wrapper（K5–K9） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(M, N)` | float16 / bfloat16 / float32 | 2D 连续行主序；M、N 为工厂期常量（无动态轴）；raw N（无 host pad） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `(M,)` | same_as(x) | 每行最小值；`out_idx=[1]` 由 jit 分配 |

### 4.3 中间缓冲区（驻留路径）

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | (bm, N) | 输入 dtype | UB（`alloc_shared`） | GM staging + 归约源（fp16/fp32） |
| `x_work`（仅 bf16） | (bm, N) | float32 | UB（`alloc_shared`） | bf16→fp32 精确域（归约源） |
| `m_ub` | (bm, 1) | 计算域 dtype（fp16/fp32/bf16→fp32） | UB（`alloc_shared`） | 行极值（reduce dst） |
| `out_ub` | (bm,) | 输入 dtype | UB（`alloc_shared`） | 输出 staging |

### 4.4 内存搬运路径

```
纯 Vector 路径（无 Cube/L1/L0 参与）：
GM[x] --T.copy(slice, real_m)--> UB[x_ub]
   --[bf16: T.vcast rint]--> UB[x_work f32]
   --T.reduce_min(dim=1, clear=True)--> UB[m_ub (bm,1)]
   --T.Parallel(bm) [恒等/cast]--> UB[out_ub (bm,)]
   --T.copy(slice)--> GM[out]
x 仅读一遍；无 GM 中间往返；无 workspace。

关键约束：GM 不可直达计算缓冲，必须经 UB；所有计算在 UB（`alloc_shared`）上完成；
reduce src/dst 同 dtype（fp16/fp32 路径直达；bf16 经 fp32 工作区）。
```

### 4.5 UB 内存预算

**预算公式（探针定标模型）**：`手工字节预算(block) = bm · N · B(dtype) + 微缓冲 ≤ 64KB`（竞态安全实测值；×2 auto-multi-buffer 双缓冲 ≈128KB ≤ 192KB − 8KB reserve）。依据链：UB=192KB（`CONST-capacity-910B2C`，origin_task multi_head_attention…-20260908，反推自 `requires 1572864 bits` 报错文本）；手工 64KB → 双缓冲 ≤128KB 的通过点由同族 argmax 探针三点验证（fp16 bm=2@N=4096 = 64KB 手工 PASS；fp16 bm=4@4096 = 128KB 手工 → `requires 2097408 bits` 溢出 FAIL）与 `TRAP-UB-multibuffer-inflation` 一致。

**B/dtype 表**（大缓冲每元素字节）：

| dtype | `x_ub` (B/elem) | `x_work` (B/elem) | **B 合计** | 微缓冲（`m_ub`+`out_ub`） |
|-------|------|------|------|------|
| float16 | 2 | — | **2** | ≈ 3·bm 字节（可忽略） |
| float32 | 4 | — | **4** | ≈ 5·bm 字节 |
| bfloat16 | 2 | 4 | **6** | ≈ 7·bm 字节 |

**逐 workload 预算验证**（驻留路径；bm 由 §5.2 阶梯）：

| workload | dtype | N | bm（阶梯） | 手工预算 bm·N·B | ×2 双缓冲 | 结论（≤184KB 可用） |
|----------|-------|---|-----------|------------------|-----------|---------------------|
| smoke (32,256) | fp16/bf16/fp32 | 256 | 8 | 4/12/8 KB | ≤24KB | ✅ 宽裕 |
| hidden-state (2048,4096) | fp16 | 4096 | 8 | 64KB | 128KB | ✅（与 argmax 验证点同包络） |
| hidden-state (2048,4096) | bf16 | 4096 | 2 | 48KB | 96KB | ✅ |
| hidden-state (2048,4096) | fp32 | 4096 | 4 | 64KB | 128KB | ✅ |
| 3d-multidim→(128,16384) | fp16 | 16384 | 2 | 64KB | 128KB | ✅ |
| long-seq (64,32768) | bf16 | 32768 | **不适用** | bm=1 → 192KB > 64KB | — | ❌ 路由兄弟 tiled 单元（§5.2） |

> **实现纪律（探针派生，Stage 3 必须遵守）**：① 未用 `alloc` **不被 DCE**（计入 UB 规划）→ fp16/fp32 与 bf16 必须**独立 prim_func** 分派，禁止「无条件 alloc `x_work` + 条件使用」；② 预算按上表 B 计算，宁小勿大（bm 降档优先于预算贴线）；③ 若编译报 `ub overflow`，优先降 bm 一档；④ 若共享缓冲（`x_ub`）出现数据竞争（多消费者 + auto-multi-buffer 预算超限），回退到「UB staging → fragment 计算」形态（`x_frag` 额外 2/4 B/elem，B 升为 4/8；argmax C-4 已验证形态）。
> **口径声明**：本结构大缓冲数 ≤2（`x_ub`，bf16 加 `x_work`），**轻于** argmax 的 3 大缓冲（x_ub/x_frag/cand，8–16 B/elem），故 64KB 手工预算包络对本结构恒成立（未实测假设 + 依据 = 同族 argmax 探针三点 + 结构更简）。amin 与 amax 结构对偶，缓冲口径完全相同。

### 4.6 动态轴定义

无（M、N、dtype 均工厂期静态；核内全部循环边界静态，运行时仅 `task` guard 动态）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")   # out 自动分配 (M,)，dtype 同 x
def _func(block_m): ...                       # K9: 无 threads；block_m 由 wrapper config 传入
```

工厂缓存：沿用 harness `ReduceKernel` 的 kernel cache（`(M,N)` 键）+ 源码 `lru_cache` 语义（§0.4-#10 保留）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector。

**判定依据**：全部计算为归约（`T.reduce_min`）+ dtype 转换（`T.vcast`/`T.cast`）+ UB 搬运（`T.copy`），无 matmul；数据通路 GM↔UB；无 Cube/L0/L0C。

### 5.2 Block 划分（block_m 阶梯 + 路径切换）

```python
# 工厂期（全部静态 Python 量）
UB_MANUAL_BUDGET = 65536                       # 字节（§4.5 竞态安全手工预算）
B = {"float16": 2, "float32": 4, "bfloat16": 6}[dtype]     # 大缓冲 B/elem
vector_cores = NPUUtils.get().get_aicore_num() * 2          # 纯 Vector 翻倍（§5.5 实查 = 40）

# 基础阶梯（源码/家族先例），预算最大值
block_m = max([p for p in (1, 2, 4, 8) if p * N * B <= UB_MANUAL_BUDGET] or [None])
# 扩展阶梯（发射项驱动，小 N 大 M；基础阶梯块数 > 2×核数时启用）
if block_m is not None and ceildiv(M, block_m) > 2 * vector_cores:
    block_m = max([p for p in (16, 32, 64, 128, 256, 512, 1024, 2048)
                   if p * N * B <= UB_MANUAL_BUDGET] + [block_m])
if block_m is None:
    route_to_sibling_tiled_kernel()            # N 超 UB 驻留 → _simple_reduce_kernel_tiled
else:
    block_m = min(block_m, M)                  # 不得分配超出行数（垃圾行风险）
num_logical     = ceildiv(M, block_m)          # 逻辑核数（纯行并行，无 N 向分核）
num_kernels     = min(num_logical, vector_cores)
num_local_tasks = ceildiv(num_logical, num_kernels)   # 核内静态任务上界
```

**N 不参与分块**（整行驻留）；M 按 `block_m` 分块，行间独立。

**路由边界（本迁移单元适用范围）**：驻留路径适用条件为 **存在 p ≥ 1 使 `p·N·B(dtype) ≤ 64KB`**，即 fp16 `N ≤ 32768`、fp32 `N ≤ 16384`、bf16 `N ≤ 10922`；超出则由兄弟迁移单元 `_simple_reduce_kernel_tiled` 承接（分块 running-min 流式，§1.6.0-C2）。manifest 的 long-seq (64,32768) bf16 落在此边界外（B=6 → bm=1 亦 192KB > 64KB），属兄弟单元范围。主选「块级不分档」的准确边界：`_simple_reduce_kernel` 只承载驻留；`_needs_tiling` 的判定在 wrapper 层按 UB 预算完成（替代 GPU 的 `N_padded > MAX_SINGLE_TILE_COLS`）。

### 5.3 约束分析

- **对齐约束**：尾轴 32B 对齐由 `T.copy` 切片天然满足（logsumexp N=300 先例）；N 无需 256 对齐（raw-N 契约，host pad 已由 Op 层 `_kernel_handles_padding=True` 保证不施加）。
- **UB 容量**：§4.5 逐 workload 验证（✅/❌）。
- **L0/L1/L0C 容量**：不适用（纯 Vector）。
- **reduce dtype 约束**：src/dst 必须同 dtype（fp16/fp32 直达；bf16 经 fp32 工作区）。
- **编译器陷阱约束**：`T.Parallel` 只读 UB/fragment（不得读 GM）；`T.vcast` 禁 f32→f32（`TRAP-C12`/argmax CONST）；条件 alloc / 条件定义变量禁止（`TRAP-tvm-parser-rules`）。

### 5.4 注意事项（非整除/边界处理策略）

- **M 非整除（`M % block_m ≠ 0`）**：`real_m = T.min(block_m, M − off_m)`；输入 `T.copy(x[off:off+real_m, 0:N], x_ub[0:real_m, 0:N])`（src/dst 双显式 slice，消除越界读）；输出 `T.copy(out_ub[0:real_m], out[off:off+real_m])`（src 显式截断，src/dst shape 一致，符合 `TRAP-T-copy-region-semantics` 推荐形态）；尾部垃圾行（`real_m < block_m` 时）不参与写出——每行规约独立，故不影响任一有效行结果。
- **N 非整除**：整行驻留天然覆盖（无 N 分块，无需 N_padded / masked load）。
- **`block_m = min(block_m, M)`**：避免分配超出行数导致 `x_ub` 尾行未初始化（argreduce 同款纪律）。
- **M=1 / N=1**：`bm=1`、`num_logical=1`、单 block；`verify_equiv.py` (1,1)/(1,2) 用例覆盖。
- **M=0**：grid=0 边界（§9.2-R5 记录，harness 测试不覆盖）。
- **全 +inf / 全 -inf / NaN 行**：`+inf`/`-inf` 正常；NaN 传播（§9.2-R1，设备探针实证）。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 docs/开发指南.md §3.3）。

- **① 逻辑核数计算**：`num_logical = ceil(M / block_m)`（纯行并行，N 不分核）。
- **② 物理核数依据（实查）**：查询代码与返回值——
  `from tilelang.utils import NPUUtils; NPUUtils.get().get_aicore_num()` → **20**（2026-10-13 实查，设备名 `Ascend910_9362`，torch 2.7.1+cpu + CANN 9.0.0）。本算子**纯 Vector → 核数翻倍 = 20 × 2 = 40**（依据 core-split-strategy.md §1 要素② 与 ascend-constraints.md §1）。**禁止文档假设值**：本仓 `CONST-aicore-910B2C` 记录的另一设备为 24，与本设备实查值 20 不同——本设计以实查 40 为准，并在 §1.6.0 roofline 标注该设备/核数差异对常数外推的影响。
- **③ 规模判定与分核方案**（统一 persistent 结构：`num_kernels = min(num_logical, 40)`；核内 `T.serial(静态上界 = ceildiv(num_logical, num_kernels))` + grid-stride `task = s·num_kernels + cid` + 动态 guard `if task < num_logical`）：

| workload | num_logical | 判定 | num_kernels | 核内任务数（静态上界） | 依据 |
|----------|-------------|------|-------------|----------------------|------|
| smoke (32,256) fp16/bf16/fp32 | 4（bm=8） | **逻辑 ≤ 物理（4 ≤ 40）** | 4 | 1 | 无需适配：行块独立、每核 ≤1 任务、单波无串行调度 |
| hidden-state (2048,4096) fp16 | 256（bm=8） | **极大规模（256 > 40，bm 受 UB 预算不可再增）** | **40** | **7**（ceildiv(256,40)） | 固定 40 核 + 核内 `T.serial` 串行（静态边界 7 + guard）；消核启动开销与超发串行调度 |
| hidden-state (2048,4096) bf16 | 1024（bm=2） | 极大规模 | 40 | 26 | 同上 |
| hidden-state (2048,4096) fp32 | 512（bm=4） | 极大规模 | 40 | 13 | 同上 |
| 3d-multidim→(128,16384) fp16 | 64（bm=2） | 极大规模（64 > 40） | 40 | 2 | 固定 40 核 + `T.serial`（静态 2 + guard）；负载不均 2:1（24 核 ×2 + 16 核 ×1） |
| long-seq (64,32768) bf16 | — | 路由兄弟 tiled 单元 | — | — | 本单元不适用（§5.2） |

> 核内串行循环边界全部为**静态值**（工厂期 Python 整数 `ceildiv(num_logical, num_kernels)`）；动态性仅存在于 guard（`if task < num_logical`）。分核参数（block_m）选择公式见 §5.2。**负载不均说明**：极大规模下 `num_local_tasks` 为 `ceil`，存在「部分核多 1 任务」的固有 1 任务粒度不均（如 256 → 16 核 ×7 + 24 核 ×6）；该不均 ≤1 任务/核，且被 persistent 结构摊薄（相比超发 256 核串行调度属数量级改善）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化或消除——本表循环仅指 block 级 / 任务级调度循环；唯一的 `T.Parallel(block_m)` 为形状收缩微循环（§1.6.2-#7）。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| M 方向（block 级） | persistent 任务循环 | `T.serial(num_local_tasks)`（静态上界 + grid-stride + guard） | 极大规模核内串行（§5.5）；消核启动开销 |
| 归约（N 方向） | 硬件归约 | `T.reduce_min(src, m_ub, dim=1, clear=True)` | 向量单元行归约（dim=1 快路径，`CONST-reduce-dims0-skinny`） |
| 元素级（输出收缩） | 向量化微循环 | `T.Parallel(block_m)` | (bm,1)→(bm,) 形状收缩（单发向量 op） |
| dtype 转换 | 向量 op | `T.vcast`（bf16→f32） | 全缓冲 dtype 转换 |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (cid, _):
    for s in T.serial(num_local_tasks):                # 静态上界
        task = s * num_kernels + cid                   # grid-stride
        if task < num_logical:                         # 动态 guard
            off_m  = task * block_m
            real_m = T.min(block_m, M - off_m)
            T.copy(x[off_m:off_m + real_m, 0:N], x_ub[0:real_m, 0:N])   # 载入
            # [bf16: T.vcast(x_ub, x_work, round_mode="rint")]
            T.reduce_min(src, m_ub, dim=1, clear=True)                  # 行极值（直接 min）
            for i in T.Parallel(block_m):
                out_ub[i] = m_ub[i, 0]                # 形状收缩（bf16 加 T.cast）
            T.copy(out_ub[0:real_m], out[off_m:off_m + real_m])          # 写出
```

### 6.3 流水线优化

**初始版本不使用 `T.Pipelined`**：单遍算法，每 block 仅读一次输入（无 K 维迭代）；任务循环上的 GM→UB 与计算重叠由 Developer 的 auto-multi-buffer 自动施加（argmax 探针在 1024 block 多波次下验证其正确性；前提为 §4.5 预算纪律）。**Stage 4 候选**：① `--enable-auto-multi-buffer=false` + 显式分配（UB 膨胀降至 ~1.10× 时 bm 可升档，需重验）；② C3 N-split partial+merge（§1.6.3-#4 / §1.6.4，小 M 大 N 负载）；③ 任务循环 `T.Pipelined` 软件流水（tile 级迭代——本单元无 N 迭代，仅对任务循环有效，收益待测）。

### 6.4 尾块处理

见 §5.4：M 尾块 `real_m = T.min(block_m, M − off_m)` + src/dst 双显式 slice + 垃圾行不写出；N 无尾块（整行驻留）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer 模式，编译器管理 UB/fragment 依赖与 auto-multi-buffer 流水同步）。

### 7.2 同步点说明

无手动同步点（单 kernel、无核间协作、无跨引擎传输）。任务循环内 MTE2（GM→UB 载入）→ V（reduce/vcast）→ MTE3（UB→GM 写出）的生产者-消费者依赖由编译器自动插入 pipe_barrier；单消费者缓冲（`x_ub` 仅 reduce/vcast 消费、`m_ub` 仅 Parallel 消费）无跨流水竞态面（对比 argmax 的 shared 多消费者竞态教训——本结构不涉及）。

### 7.3 pass_configs 配置

无特殊 pass 配置（Developer 默认；`TILELANG_ASCEND_MODE=Developer`）。Stage 4 探索项：`--enable-auto-multi-buffer=false`（§6.3-①）。

---

## 8. 验证方案

### 8.1 Golden 函数

> golden 以 **§0.1 源算子语义**为唯一依据实现（移植 harness 参考 `examples/TileOPs/tests/ops/test_reduce.py` 中 `ReduceTest.ref_program` 的 amin 分支），**不复刻 §0.6 的 NPU 算法**（无 padding 消除 / 原域 reduce / 直接 reduce_min / persistent 分核结构）——保证验证独立性。

```python
import torch

def golden_amin(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现（CPU），对应 torch.amin(x, dim=-1)。

    输入 : x of shape (M, N), dtype ∈ {float16, bfloat16, float32}
    输出 : y of shape (M,), dtype = same_as(x)
    """
    return x.float().amin(dim=-1).to(x.dtype)
```

（harness pytest 侧 ref：`ReduceTest.ref_program` 的 amin 分支 `x_f32.amin(dim=-1).to(x.dtype)`；spec 接口 `test_amin_spec_dim` 用 `torch.amin(x.float(), dim=dim, keepdim=keepdim)` 覆盖 dim/keepdim。）

### 8.2 精度标准

来自 `examples/TileOPs/tests/ops/test_reduce.py` 中的 `_tol`：

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 1e-4 | 1e-4 | 高精度 |
| float16 / bfloat16 | 1e-2 | 1e-2 | 低精度容忍 |

> 本算子是**精确选择运算**：§1.6.1 机器验证（E1–E4）在全部 shape/角点上限 **0 ulp**（687 用例），设备 NaN 探针亦逐位一致，故实际误差应远优于上表容差；容差按 harness 契约保留（不放松亦不收紧）。

### 8.3 L0 门槛测试计划

L0 聚焦「编译通过 + 基本精度正确」，覆盖 manifest 负载 + 边界；完整 L1/L2/Boundary 分层套件交由 `tilelang-op-develop` 展开。

| 测试编号 | Shape (M,N) | dtype | 路径/配置 | 验证目标 | 容忍度 |
|---------|-------------|-------|-----------|----------|--------|
| L0-1a/b/c | (32,256) | fp16 / bf16 / fp32 | 驻留 bm=8 | smoke 三 dtype | 1e-2 / 1e-2 / 1e-4 |
| L0-2 | (2048,4096) | fp16 | 驻留 bm=8 | manifest hidden-state fp16 + 极大规模 persistent（256 block → 40×7 任务，多波次） | 1e-2 |
| L0-3 | (2048,4096) | bf16 | 驻留 bm=2 | bf16 路径（vcast 预转 + fp32 reduce_min + 降回 cast） | 1e-2 |
| L0-4 | (128,16384) | fp16 | 驻留 bm=2 | manifest 3d-multidim（dim=[0,2] → Op 层 (M,N)=(128,16384)）| 1e-2 |
| L0-5 | (128,512) | fp32 | 驻留 bm=8 | harness ReduceBasic fp32 | 1e-4 |
| L0-6a/b | (128,300) | fp16 / bf16 | 驻留 bm=8 | **N 非对齐**（300，非 32/256 倍数） | 1e-2 |
| L0-7 | (129,512) | fp16 | 驻留 bm=8 | **M 尾块**（129 % 8 = 1） | 1e-2 |
| L0-8 | (1,512) / (1,102400 若路由) | fp16 | — | M=1 极小行 / 单行负载 | 1e-2 |
| L0-9 | 构造行 (M=8,N=257)：全 +inf / 全 −inf / +inf 夹有限 / −inf 夹有限 / 重复极值 / dtype 极值 | fp16 / fp32 | 驻留 | **边界值语义**（`verify_equiv.py` 角点集的 NPU 侧复验） | 1e-2 / 1e-4 |
| L0-10 | (4,128,4096) dim=[0,2] keepdim=False/True | fp16 | 驻留 | spec 接口 dim/keepdim（Op 层 reshape 后 M=128,N=16384） | 1e-2 |
| L0-11 | 构造 NaN 行 (M=8,N=64)：NaN 居中/居首/居尾/全 NaN + 有限行 | fp16 / fp32 | 驻留 | **NaN 传播**（对齐设备探针 `repro/CONST-reduce-min-nan-propagation.py` 的 NPU 侧复验） | equal_nan 逐位 |

**通过条件**：全部 L0 项通过 atol/rtol 检查；NaN 行为用 `equal_nan=True` 逐位核对（§9.2-R1）。

### 8.4 完整分层测试套件（交由 tilelang-op-develop 阶段）

- **L1（功能覆盖）**：1D 输入、多维 dim、keepdim=True/False、全部 dtype × shape 组合、manifest 全 workload。
- **L2（边界）**：M=1、N=1、M=block_m 整除、M=block_m+1、N 非 32B 倍数、`block_m=min(bm,M)` 触发（M<8）。
- **Boundary**：极值输入（全 +inf、全 −inf、含 NaN、±0 混合、亚正规）；大 M persistent 多波次回归。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 处理方案 |
|------|------|----------|
| **bf16 不被 reduce 支持** | `T.reduce_min/vcast` 的 dtype 表 bf16 × | bf16 全程 vcast→fp32 → reduce_min → cast 回（§1.6.1-E3） |
| **reduce 混合 dtype 静默错误**（探针实证，与 `T.reduce_max.md` §2.4 示例 `dtype≠accum_dtype` 形式矛盾） | fp16 src→fp32 dst 静默产出错误值 | 强制同 dtype src/dst；bf16 用独立 fp32 缓冲而非跨 dtype reduce；文档示例形式的差异列入 §9.2-R4 风险（Stage 3 以实机验证为准） |
| **未用 `alloc` 不被 DCE** | 计入 UB 规划 | fp16/fp32 与 bf16 独立 prim_func（禁止条件 alloc，`TRAP-tvm-parser-rules`） |
| **UB 192KB + auto-multi-buffer 膨胀** | 大块探索须按膨胀后预算评估 | §4.5 64KB 手工预算（×2 包络）；编译 `ub overflow` 报文即精确需求 |
| **物理核数因设备而异** | 分核数与 persistent 任务数依赖实查值 | 实查 20 → Vector 40（§5.5）；禁止沿用其他设备文档值（如 24） |
| **`T.vcast` f32→f32 非恒等** | 破坏数据（CONST/argmax 实证） | 同 dtype 用 `T.copy`，不做 f32→f32 vcast |
| **`T.Parallel` 禁读 GM** | 编译错误 | 所有 GM 读写经 `T.copy`；`T.Parallel` 只操作 UB/fragment |
| **M 尾块越界读/写** | 源实现无守卫（§0.3 源缺陷） | `real_m` + src/dst 双显式 slice（§5.4） |

### 9.2 风险点明细

| # | 风险 | 说明 | 处置 / 状态 |
|---|------|------|------------|
| R1 | **NaN 行为** | `torch.amin` 传播 NaN；NPU `T.reduce_min` 亦传播 NaN——**设备探针实证两者逐位一致**（8 行 × fp16/fp32，`repro/CONST-reduce-min-nan-propagation.py` PASS）；NPU 具体位模式（传播 vs 返回规范 NaN）须 Stage 3 L0-11 用 NaN 行实测确认 | 设计期已按「传播」论证并设备实证（INFO + L0-11）；**不阻塞** |
| R2 | **常数设备/版本绑定** | §1.6.0 roofline 的 MTE2/发射常数 origin 为 Ascend910B2C（24 核）；本设备 `Ascend910_9362`（20 核 → Vector 40），估算按核数外推 = **未实测假设 + 估算依据** | Stage 4 按 hardware-cost-model.md §3 回填实测对账；若系统性偏差 >2× 触发 `[DESIGN_LIMIT]` 归因 |
| R3 | **UB 膨胀系数对本结构未独立实测** | 64KB 手工预算包络承自同族 argmax 探针（8–16 B/elem 结构）；本结构大缓冲更少（2–6 B/elem），且与对偶 amax 完全相同 | 依据充分（更简结构恒不劣于已验证包络）；Stage 3 首编译即以 `ub overflow` 报文核对；风险低 |
| R4 | **reduce 同 dtype 约束 vs 文档示例差异** | `T.reduce_max.md` §2.4 用 `dtype≠accum_dtype`；argmax 探针实测混合 dtype 静默错误；本设计取**探针实证的同 dtype**路线 | 与 `testing/npuir` 通过用例一致；差异待工具链复验 |
| R5 | **M=0（grid=0）** | 非规约维含 0 → 输出 (0,)；源实现未覆盖 | harness 测试不覆盖；列入风险，不阻塞 |
| R6 | **窄 N 垃圾行** | argreduce 记录：窄 N + 垃圾尾行可能经小 N row 处理污染有效行 | 本算子 manifest 最小 N=256（非窄）；已加 `block_m=min(block_m,M)`；若 Stage 3 遇窄 N（<32）异常，加 `x_ub` 初始化（identity `+inf`）或 per-row guard |
| R7 | **尾块动态 slice 形态** | `TRAP-T-copy-region-semantics`：base+size 形态越界写风险；`size=[1,N]` 前向补 1 | 统一用 **src/dst 双显式 slice** 形态（trap 推荐通用形态；logsumexp/argreduce/对偶 amax 通过） |
| R8 | **输出形状 (bm,1)→(bm,)** | `T.reduce` 要求 src/dst rank 一致（§1.6.2-#7），无 cast+收缩复合向量 API | 单发 `T.Parallel(block_m)` 收缩（每 block 1 op，非热点）；fragment 形态为回退 |
| R9 | **wrapper 分档 gate 仍为 GPU 语义** | 现有 `amin.py` 的 `_needs_tiling = N_padded > MAX_SINGLE_TILE_COLS` 与 `default_config` 的 48KB 启发式是 GPU 派生 | 本设计给出 NPU 分档公式（§5.2）与分核参数；wrapper 层 `default_config`/`_needs_tiling` 的 NPU 化属 wrapper/兄弟单元职责，本设计标注接口契约（`block_m` 传入） |
| R10 | **负载不均** | 极大规模 `ceil` 分任务存在 ≤1 任务/核的不均 | 固有粒度不均，被 persistent 摊薄；Stage 4 可选 bm 扫描改善 |
| R11 | **取负桥接替换的边界一致性** | E1 用直接 `reduce_min` 替换源 `-max(-x)`；对 ±0 的「选中哪个零」理论上依赖实现 | `verify_equiv.py` E1 角点集（含混合符号零行）240 例 0 违反；torch 与负号技巧均返回 +0.0（实测）；Stage 3 L0-9 复验 |

### 9.3 特殊场景处理

| 场景 | 处理方式 |
|------|----------|
| **N 非 32B 倍数** | 无 padding；`T.copy` 切片保证 32B 尾轴对齐（logsumexp N=300 先例） |
| **M 非 block_m 倍数** | `real_m` + 双 slice + 垃圾行不写出 |
| **N 超 UB 驻留上限** | 路由兄弟单元 `_simple_reduce_kernel_tiled`（分块 running-min 流式） |
| **dim=0 / 多维 dim / dim=None** | Op 层 `flatten_for_multidim`/`movedim` 归一为 `(M,N)`（如 (4,128,4096) dim=[0,2] → M=128,N=16384；dim=None → M=1,N=numel），kernel 始终沿 dim=1 归约 |
| **M=1** | `bm=1`、`num_logical=1`、单 block 驻留 |
| **全 +inf / 全 -inf 行** | reduce_min 正常返回该值（vs `+inf` pad 语义一致，E4 覆盖；设备探针行 5/6 实证） |
| **NaN 行** | 传播 NaN（§9.2-R1，设备探针实证） |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/amin/_simple_reduce_kernel/
├── _simple_reduce_kernel.py     # 算子实现（kernel + golden + L0 测试）+ 与 Golden 比对
├── DESIGN.md                    # 本设计文档
├── verify_equiv.py              # D-1 等价性机器验证（E1–E4；ALL EQUIV_PASS 687 例）
├── repro/
│   └── CONST-reduce-min-nan-propagation.py  # 设备 NaN 传播探针（PASS）
├── RETROSPECTIVE.md             # Stage 1 复盘（自进化钩子）
└── .stage_state.json            # 编排层状态（不由本阶段维护）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成并执行 | 等价性机器验证（§1.6.1 结果表，exit 0，687 例 0 违反） |
| `repro/CONST-reduce-min-nan-propagation.py` | ✅ 已完成并执行 | 设备探针（`T.reduce_min` NaN 传播 vs `torch.amin`，PASS） |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘章节 |
| `_simple_reduce_kernel.py` | ⬜ 待实现（Stage 3） | 算子实现（kernel 工厂 + golden + L0） |

### 10.3 命名规范

- 项目目录名: `amin`；算子目录名: `_simple_reduce_kernel`
- 实现文件: `_simple_reduce_kernel.py`；工厂函数: `_simple_reduce_kernel(M, N, op_kind, dtype)`
- Golden 函数: `golden_amin(x)`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性机器验证（verify_equiv.py）+ 设备 NaN 探针
2. ⬜ Golden 函数（§8.1）
3. ⬜ 算子实现（`_simple_reduce_kernel` 工厂 + 分档/分核，§3.3/§5.2/§5.5）+ L0 门槛（§8.3）
4. ⬜ 与 Golden 比对（atol/rtol §8.2），全过进入 Stage 4

### 10.5 迁移规则遵守确认

| 规则 | 遵守情况 | 说明 |
|------|----------|------|
| 算子名 = `_simple_reduce_kernel` | ✅ | 不裁剪不变换 |
| `@tilelang.jit` 函数声明不变 | ✅ | `_func(block_m)` 保留（仅移除 K9 `threads`，加 `target="npuir"`） |
| `@T.prim_func` 参数名/顺序不变 | ✅ | `main(x: T.Tensor((M,N), dtype), out: T.Tensor((M,), dtype))` |
| 从源码推断输入规格 | ✅ | 从 GPU 源码 + manifest 推断，未询问用户 |
| K9：threads 移除 | ✅ | `_func(block_m)` 无 threads |
| Host 侧不改输入 | ✅ | 无 host pad；全部边界处理在 kernel 内 |

---

## 11. 修订日志

**revision_index: 0（首次设计，2026-10-13）**

本版为 `_simple_reduce_kernel`（AminFwdOp 驻留路径）的首次设计，无历史回退。设计核心（可提炼为 4 条 GPU→NPU 关键重设计）：① 边界处理（256 对齐 pad + masked load → 原始 N + M 尾块双 slice，§0.6-R1）；② 行极值实现（取负 → `reduce_max` → 取负回 → **直接 `T.reduce_min`**，§0.6-R5）；③ 并行结构（block×thread + threads → 一维 persistent 40 核 + 核内静态 serial，§0.6-R2）；④ dtype 域（全量 fp32 上抛 → fp16/fp32 原域 + bf16 vcast，§0.6-R3）。分档（§0.6-R4：48KB SMEM 启发式 + 32512 列上限 → UB 64KB 手工预算阶梯 + 超限路由兄弟 tiled 单元）与对偶 amax 一致。
