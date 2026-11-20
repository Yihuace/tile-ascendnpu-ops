# 逻辑归约 `_logical_reduce_kernel`（承载 `AllFwdOp` / `torch.all` 语义）算子设计文档

> **迁移任务（migration-harness）**：GPU TileOPs `AllFwdOp` 的 TileLang kernel 函数 `_logical_reduce_kernel`
> → TileLang-NPUIR（`target="npuir"`，Developer 模式）。
> **源算子**：`/workspace/TileOPs-gpu/tileops/kernels/reduction/logical_reduce.py`
> （基础工厂 `_logical_reduce_kernel` + `LogicalReduceKernel`；any / all / count_nonzero 三 kind 共用同一工厂，
> 由 `op_kind` 选分支。**本迁移单元只驱动 `op_kind="all"`，输出 bool（kernel 载体 int8）**；
> 兄弟 count_nonzero 为独立迁移单元，其 int64 契约与本设计无关）。
> **本设计基于**：源码全文解读（Phase M0）→ 算法族调研（Phase R，调研四问）→ 硬件耦合性判定与 NPU 重设计
> （Phase M1，四态处置）→ 算法级优化设计（Phase 2）→ **等价性机器验证**（D-1，`verify_equiv.py`，
> A1/A2/A2b/A3/A4/A5 全部 `EQUIV_PASS`，对照组 `CONTROL_OK`）
> → **设计期容量探针**（D-3，1 个探针文件，见下）。
> **工具链戳**：tilelang `0.1.2+1bc8dbc3a1c80345a12522d815a2268e555270fa`（dev root build）
> - CANN 9.0.0 + torch 2.9.0+cpu + **Ascend910B3（AIV=40）**。

**设备实查（本任务实测，2026-10-16）**：

| 实查项 | 查询代码 | 返回值 |
|--------|----------|--------|
| AI Cube 核数 | `tilelang.utils.npu_utils.NPUUtils.get().get_aicore_num()` | **20** |
| AI Vector 核数 | `NPUUtils.get().get_aivector_core_num()` | **40** |
| 片上 UB 容量 | `tilelang.utils.npu_arch.AscendArch().ub_cap` | **196608 B（192 KB）** |
| 芯片名 | `AscendArch().chip_name` | `Ascend910B` |

**设计期容量探针（D-3；1 个探针文件，落 `history_version/`，只标定未知常数、不实现算子）**：

| 探针 | 标定的未知常数 | 结论（实测） |
|------|----------------|--------------|
| `examples/all/_logical_reduce_kernel/history_version/design_probe_ub.py` | 「all 指示链」（`vcmp(eq)` → `vcast(i1→f32)` → `reduce_max(dim=1)` → 微 epilogue）在本工具链的 **UB 单价（B/元素）** 与单 block 硬上限 | **单价与 dtype 相关**：fp16 **8.00 B/元素**、fp32 **8.00 B/元素**、bf16 **12.00 B/元素**（bf16 的比较被 lowering 上抛到 f32 ⇒ 多 2 个 f32 平面，IR 实测 `vcast bf16→f32` 出现在 `vcmp` 两操作数上）。单 block **实测通过点**：fp16 24576 元素、bf16 16384 元素、fp32 16384 元素；**溢出点**：fp16 32768、bf16 24576、fp32 24576（报错文本反解 8.00/12.00 B/元素）。**同文件第二节**另证：raw 窄行宽度（N=2 / 4 / 64 / 300，即 4B / 8B / 128B / 600B 行宽）**均可编译通过** ⇒ 设计中**不需要任何宽度 pad**（源 `align_up(N,256)` 的 pad 动机在 NPU 不存在；`CONST-reduce-ub-per-element` 的整平面 ~128 B/元素模型**不适用于本 2-D 行归约形态**，该条目适用域限定为 3-D 整平面归约） |

> **探针与兄弟任务的差异（必须显式声明）**：兄弟 count_nonzero 同工厂任务用 `reduce_sum` 链标得 8.00 B/元素且
> **与 dtype 无关**；本任务实测 `reduce_max` 链在 **bf16 上为 12.00 B/元素**（比较被上抛 f32）。**常数不得互相借用**——
> 本设计的 `E_cap` 取值以本探针的**最坏 dtype 单价**推导（见 §4.5）。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

**数学语义**：对已规整为 2D 的输入 `x ∈ 𝔽^{M×N}` 沿最后一维（规约维 N）做**逻辑与归约**：

$$y[i] \;=\; \bigwedge_{j \in [0,N)} \bigl(x[i,j] \neq 0\bigr), \qquad y \in \{\text{True},\text{False}\}^{M}$$

等价地：`y[i] = (min_j [x[i,j] != 0]) == 1`，即源实现「先把元素判决为 0/1 指示量，再取行向 `reduce_min`」。

**I/O 契约（迁移必需字段）**：输入 `(M, N)`（raw N；Op 层已把多轴输入 movedim+reshape 成 2D：M = 未规约维乘积、
N = 规约维乘积）；输出 **`(M,)`、dtype `bool`**（源 `source_output_shape: "(M,)"` 已确认；无转置布局）。
kernel 侧输出载体为 **int8**（`"int8" if op_kind != "count_nonzero" else "int64"` 三元式；bool 不能作为 GM 存储
dtype），wrapper 侧 `out_raw.bool()` 还原 bool（`examples/TileOPs/tileops/kernels/reduction/all/all.py`
的 `_logical_reduce_fwd_wrapped`）。`dim=None` 全张量规约被 Op 层 flatten 成 `(M=1, N=numel)`。

**与兄弟 kind 的关键区分（本迁移单元的判据）**：any / all / count_nonzero 共用同一工厂，但
**all 的输出载体是 int8（bool，`T.cast(result[i] > 0.5, "int8")`），归约原语是 `reduce_min`**
（count_nonzero 为 `int64` + `reduce_sum`，any 为 `int8` + `reduce_max`）。**不得照抄 count_nonzero 的 int64 契约或
其计数公式**；`examples/TileOPs/tileops/ops/reduction/reduce.py` 中出现的 `"all"` 字符串只是 **roofline 记账
kind**（`reduce.py` 的 roofline 分支），**不是实现入口**，本设计不据其实现。

**规约语义（累加顺序）**：AND 是**可交换、可结合、幂等**的布尔运算，源实现把元素判决为 0/1 后取 `min`——
**纯 0/1 指示量的 min，无浮点累加**（部分和恒为 0.0 或 1.0，f32 精确表示）⇒ **并行分组 / 分段 / 累加顺序变化
不影响结果**（与 logsumexp/var/norm 类浮点归约的关键差别；§1.6.1-A4 机器验证分段与 running-max 合并位精确）。
这是本算子最重要的一条迁移放宽：**任何分段/分核重组都不引入数值风险**。

**dtype 语义**：kernel 侧输入 dtype ∈ {float16, bfloat16, float32}——bool / int32 / int64 / complex64 / complex128
由 Op 层 `to_logical_float32` 预转 float32（complex 语义 = `(real != 0) | (imag != 0)`，见
`examples/TileOPs/tileops/kernels/reduction/all/all.py` 的函数 `to_logical_float32`）。
源在 kernel 内把数据 **全宽上抛到 fp32** 后做比较与归约；**输出载体恒 int8**（bool ⇒ 1/0）。

**边界语义**（源码 + torch 语义 + 本工具链行为共同确定；§8.2 逐项记录）：

| 边界 | torch（CPU） | 源 GPU 实现 | 本设计 NPU 算法 | 处置 |
|------|--------------|-------------|-----------------|------|
| `±0.0` | 视为「零」⇒ 该行 all=False | `x != 0.0`（IEEE）⇒ ±0 均判「零」⇒ all=False | `x == 0.0`（**IEEE 正确**）⇒ ±0 均判「零」⇒ all=False | 等价域内 |
| **NaN（行内无 0）** | **NaN 计入非零** ⇒ all 可保持 True | `!=` 为 **unordered NE** ⇒ NaN 计入非零 | 补式用 `== 0`（**ordered EQ**：NaN==0 为假 ⇒ NaN **不计入零**）⇒ all 保持 True | **三方一致**（§1.6.1 INFO 行实测：源=True / 候选=True / torch=True；ordered-NE 路线=False，**即为被否决形态**） |
| NaN 混零 | 零主导 ⇒ all=False | 同 | 同（`[x==0]` 命中 ⇒ any_zero=1 ⇒ all=False，与 NaN 无关） | 等价域内 |
| `±inf` / 亚正规 | 计入非零（不破坏 all） | 计入 | 计入（比较运算不做 flush-to-zero；§9.2-R3 登记亚正规实测） | 等价域内 |
| 空规约维 `N=0` | 空 AND 恒真 ⇒ 每行 True | `N_padded = align_up(0,256) = 0` ⇒ `_needs_pad=False`，`(bm,0)` 缓冲与 0 宽 reduce（**源未覆盖**） | 工厂期 `N >= 1` 断言；`N=0` 视为退化（§9.2-R5） | 保留为风险登记 |
| 非整除 `N`（如 300） | — | 源工厂内部 `align_up(N,256)` 派生 `N_padded`，由 `_needs_pad` 分支的**掩码加载**补齐（all 的 pad 单位元 **1.0**，AND 中性） | **在实列域（恰 N 列）计算，完全不 pad**（§1.6.1-A2/A2b 机器验证：pad=1.0 域与实列域逐行恒等） | §0.5 / §0.6-R3 |
| 窄行 `N < 32 B`（如 2/4） | — | 源 256 对齐 ⇒ `N_padded=256` | **raw 宽度直算**（探针实测 N=2/4/64/300 均编译通过） | 探针证据 |
| `M=0`（非规约维含 0） | 输出空 | 源 grid=0（未覆盖） | 列为风险（§9.2-R6） | 风险登记 |
| `N=1` | 单元素行 ⇒ `y = (x != 0)` | 同 | 同（`(bm,1)` tile，`reduce_max` 恒等） | 等价域内 |

**host 侧语义**：Op 层的 `movedim` / `reshape` / `contiguous` 均为**实现手段**而非算子语义；
`examples/TileOPs/tileops/ops/reduction/all.py` 明示 `_kernel_handles_padding = True`（**kernel 接 raw `(M, N)`，
Op 层不做 host pad**）、`_pad_value() = 1.0`（all 的 AND 中性 pad 单位元）、`_empty_dim_policy = "noop"`
（`dim=[]` / `dim=()` 为 no-op：forward 返回 `x.bool()`，**不进入 kernel**），
`_noop_output_dtype() = torch.bool`。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据（`torch.all` / `x.bool().all(dim)`），
**不复刻 §0.6 的 NPU 算法**（无补式、无指示平面、无列分块），保证验证独立性。

### 0.2 源算子输入输出（含 NPU 侧契约核对）

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | **`(M, N)`（raw N）** | float16 / bfloat16 / float32 | M = 非规约维乘积，N = 规约维乘积；工厂参数携带**原始 N**（源工厂内部派生 `N_padded = align_up(N, 256)`） |
| `out` | `(M,)` | **int8**（bool 载体）→ wrapper 转 `bool` | 每行逻辑与结果（1/0） |

**NPU 侧输入契约核对（实读目标工件）**：

| 核对项 | 结论 | 依据 |
|--------|------|------|
| 目标 Op 层 `_kernel_handles_padding` | **`True`**（`AllFwdOp` 类属性覆写） | `examples/TileOPs/tileops/ops/reduction/all.py` |
| 因而 Op 传入宽度 | **raw `N`**（无 host `F.pad`） | `examples/TileOPs/tileops/ops/reduction/reduce.py` 的 `_prepare_input`（`_kernel_handles_padding` 为真时跳过 pad） |
| Op 层 `_pad_value` | `1.0`（all 的 AND 中性单位元） | `examples/TileOPs/tileops/ops/reduction/all.py` 的 `_pad_value` |
| kernel 必须声明的宽度 | **`(M, N)`**（与 Op 传入逐元素一致） | 与源工厂「内部派生 N_padded」的差异在 §0.6-R3 处置 |
| 工厂接口（harness 固定） | `_logical_reduce_kernel(M, N, op_kind, dtype)` → `_func(block_m)` → `kernel(x)`（NPU 侧 `threads` 已去除，K9） | `examples/TileOPs/tileops/kernels/reduction/all/all.py` 的 wrapper |
| 输出 dtype 契约 | **bool**（manifest `AllFwdOp.outputs.output.dtype = "bool"`；`register_fake` 返回 `torch.empty((M,), dtype=torch.bool)`） | `examples/TileOPs/tileops/manifest/reduction.yaml` 的 `AllFwdOp` 段 |
| 本迁移单元 kind 范围 | **仅 `all`**；any/count_nonzero 为独立迁移单元 | 用户迁移约束 + Op 层 docstring |

### 0.3 实现算法解读（怎么算）

按结构化阅读协议逐层读 `_logical_reduce_kernel`（基础工厂；`_logical_reduce_kernel_tiled` 为**非迁移对象**，
其解读仅用于耦合性判定与在线形态证据，见 §0.4-#8 / §1.6.0-R2）。

**计算步骤分解**（覆盖基础工厂全部计算语句；`op_kind="all"` 分支）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `N_padded = align_up(N, DEFAULT_ALIGNMENT=256)`；`_needs_pad = N_padded != N`；`_pad_val = _pad_value_for_op("all") = 1.0`；`T.Kernel(T.ceildiv(M, block_m), threads=threads)` | 工厂参数 | 编译期常量 + 一维 grid + CUDA 线程宽 | 任务划分 |
| 2a（`_needs_pad` 为真） | `for i, j in T.Parallel(block_m, N_padded): x_f32[i,j] = T.if_then_else((pid_m*bm+i < M) and (j < N), T.cast(x[...], "float32"), T.cast(1.0, "float32"))` —— **掩码加载 + fp32 上抛 + 越界行保护 + AND 中性 pad 四合一** | GM `x` | `x_f32 (bm, N_padded)` fp32 | 数据载入（非整除处理） |
| 2b（`_needs_pad` 为假） | `T.copy(x[pid_m*block_m, 0], shared_buf)` → `for i, j in T.Parallel(block_m, N_padded): x_f32[i,j] = T.cast(shared_buf[i,j], "float32")` | GM `x` → SMEM → fragment | `x_f32 (bm, N_padded)` fp32 | 数据载入（SMEM tiling + fp32 上抛） |
| 3 | `for i, j in T.Parallel(block_m, N_padded): bool_vals[i,j] = T.if_then_else(x_f32[i,j] != 0.0, 1.0, 0.0)` | `x_f32` | `bool_vals (bm, N_padded)` fp32（0.0/1.0） | **[x ≠ 0] 指示化** |
| 4（all 分支） | `T.reduce_min(bool_vals, result, dim=1)` | `bool_vals` | `result (bm,)` fp32（1.0 当且仅当全非零） | **行向逻辑与**（min over 0/1） |
| 5（all 分支） | `for i in T.Parallel(block_m): out_local[i] = T.cast(result[i] > 0.5, "int8")` | `result` | `out_local (bm,)` int8 | 逻辑与 → bool 载体 |
| 6 | `T.copy(out_local, out[pid_m * block_m])` | `out_local` | GM `out` 行块 | 输出搬运 |

**数据流（源硬件视角）**：`GM[x] →T.copy→ SMEM[shared_buf] →cast→ register/fragment[x_f32] →ite→
fragment[bool_vals] →reduce_min→ fragment[result] →compare+cast→ fragment[out_local] →T.copy→ GM[out]`；
`_needs_pad` 时路径 2a 合并为 GM 直接掩码载入（SMEM 被绕过）。

**循环与并行结构**：`T.Kernel(T.ceildiv(M, block_m), threads=threads)` —— 一维 grid × CUDA 线程两级并行；
`block_m` 由 48 KB SMEM 预算启发式在阶梯 `[1,2,4,8]` 内取不超预算的最大者（`default_config`），
`threads ∈ {128, 256}`；**无流水**（无 `T.Pipelined`）。

**host 侧逻辑**：Op 层 validate → dim 归一 → `movedim` → `contiguous` → `reshape(M,N)` →
（**不 pad**，因 `_kernel_handles_padding=True`）→ 调 kernel → 输出 reshape 回原形（`keepdim=false`）；
`functools.lru_cache(maxsize=32)` 按 `(M, N, op_kind, dtype)` 缓存编译产物。

**编译配置**：`@tilelang.jit(out_idx=[1])`（GPU target）；无特殊 `pass_configs`；
`supported_archs = [80, 86, 89, 90]`（CUDA SM）；`ascend_mode` 无（GPU 侧）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | SMEM tiling（`alloc_shared` + 整块 `T.copy`，非 pad 路径） | 合并访存、一次载入 | 行块驻留 SMEM 后转 fragment | shared memory | 硬件强相关（NPU 有 UB 等价） |
| 2 | **256 元素对齐 pad + AND 中性单位元掩码加载**（`align_up(N,256)` + `T.if_then_else` + `_pad_val=1.0`） | 规避 GPU `T.copy`/fragment 的 256 对齐与列上限约束 | 计算域总是 `N_padded` 的整数倍宽；pad 列用 **1.0（AND 中性）** 填充 | GPU 256 元素对齐/copy 布局推断 | 对齐约束硬件相关；**单位元技巧可移植** |
| 3 | 指示化（1.0/0.0）后复用 `reduce_min` 实现逻辑与 | 用现成归约原语实现 AND，避免手写判决链 | [x≠0] 指示化 → 复用 min 归约 | — | **算法层，可移植**（NPU 保留结构） |
| 4 | 全量 fp32 上抛（步骤 2a/2b 的 cast） | 数值稳健（对 sum/var 类归约必要） | 数据先 cast fp32 再比较与归约 | — | 对 all **非必要**（比较是精确谓词，§1.6.1-A3 穷举验证） |
| 5 | `block_m` 48 KB SMEM 预算启发式 + `[1,2,4,8]` 阶梯 | 行块尽量大以摊薄 block 开销 | 预算 ÷ (N_padded × elem_bytes) | SMEM 容量 | 预算值硬件相关 → 需重设计 |
| 6 | `threads`（128/256）作为归约并行宽度与布局约束参数 | CUDA 线程级并行归约 | 线程内/跨线程归约 | CUDA warp/thread | 硬件强相关（NPU 无 threads 概念） |
| 7 | `functools.lru_cache(maxsize=32)` 工厂缓存 | 消重复编译 | Python 层 `(M,N,kind,dtype)` 键 | — | 可移植 |
| 8 | **N-tiled 回退工厂** `_logical_reduce_kernel_tiled`（`tile_n` + `T.Serial(num_tiles)` + 运行累加 `acc[i] = T.min(acc[i], tile_acc[i])`） | 规避 **TileLang 单 fragment 列上限 32768** 与 SMEM 预算 | 沿 N 分块，逐块 `reduce_min` 后**运行 min 合并**（all 分支的在线形态） | GPU fragment 列上限 / SMEM 预算 | 触发条件硬件相关；**本迁移单元不迁**（用户约束）；其**运行合并结构**是「all 存在在线变体」的直接证据（§1.6.0-R2） |
| 9 | 结果块写 `T.copy(out_local, out[pid_m*block_m])` | 向量化写回 | fragment → GM | — | 模式可移植 |
| 10 | autotune（`autotune_configs` / `tune_by_forward`） | 配置搜索 | 逐配置实测选优 | — | harness 已移除（K8） |
| 11 | 重 pad 布局约束（`block_m × N_padded ≤ 2 × threads`，仅 N<256 生效） | 规避 GPU copy 布局推断限制 | 限制 block_m | GPU copy-layout 推断 | 硬件强相关（NPU 不适用） |

> **识别完整性**：上表覆盖源基础工厂的全部优化性结构（步骤分解表已把每条计算语句归入步骤 1–6）；
> 源为朴素「SMEM + 指示化 + 单次归约」形态，**无** warp shuffle / mma / swizzle / 异步流水 / 双缓冲等 GPU 深度优化
> （源逐块 `T.copy` 由编译器决定是否流水，源码未声明）。
> **源缺陷（本设计修正，Stage 3 不得照抄）**：步骤 6 的 `T.copy` 与步骤 2b 的 `T.copy` **均无 M 尾块守卫**
> （`M % block_m != 0` 时越界读/写；harness 用例 `(129,512)` 配 `block_m=8` ⇒ `129 % 8 = 1` 即触发该位形）。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（输入含 Phase R 调研结论 §1.6.0）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 逻辑与语义（`(M,N)`→`(M,)` bool，±0/NaN/±inf 边界） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden = `torch.all`；NaN 由补式天然对齐（§0.1） |
| 指示化 + 单次行归约的两步结构（步骤 3+4） | 算法 | 无（纯算法层） | 有：Vector `vcmp`/`vcast` + `T.reduce_max(dim=1)` | **保留** | 结构保留；比较域与指示域按 §1.6.1-A1/A3 优化（`docs/Tilelang.language/比较操作/T.vcmp.md`、`docs/Tilelang.language/规约操作/T.reduce_max.md`） |
| 步骤 3 的 `x != 0.0` 判决式（unordered NE）+ 步骤 4 的 `reduce_min` | 算法 | 无 | **有但语义相反**：`T.vcmp("ne")` 是 **ordered** NE（NaN 恒假） | **等价替换** | 改为**有序 EQ 的补式**：`all = ¬any[x==0] = (max_j [x==0]) == 0`（`eq` 为 IEEE 正确，`traps-compiler.md` 的 `TRAP-vcmp-ne-ordered-nan` 明证），NaN 计入非零与 torch/源一致（§1.6.1-A1 + 对照 A1c + INFO 行） |
| 步骤 2a/2b 的全量 fp32 上抛（#4） | 算法/精度 | — | — | **舍弃** | 比较是精确谓词：`[x_fp16 == 0] ≡ [fp32(x) == 0]`（fp16/bf16→fp32 精确单射，保 ±0/NaN）；§1.6.1-A3 **全 2¹⁶ 位型穷举** `EQUIV_PASS`。省 1 个全宽 pass + 半份 staging 字节 |
| SMEM tiling / 整块 `T.copy`（#1） | 优化 | shared memory | 有：UB（`T.alloc_shared`，实查 192 KB） | **等价替换** | GM→UB staging（`migration-analysis.md` §5.3 行「shared memory → L1/UB」）；对齐需求由 `T.copy` 切片天然满足（探针证 raw 宽度可编译） |
| 256 对齐 pad + AND 中性掩码加载（#2） | 优化 | GPU 256 元素对齐 + fragment 列上限 | **不需要**：NPU 在实列域计算即精确 | **舍弃** | §0.6-R3：raw `(M, N)` 计算**不需要任何 pad**——AND 语义对 pad 天然中性（pad=1.0 不改变结果）；§1.6.1-A2/A2b 机器验证「pad=1.0 域 ≡ 实列域」；探针实测 N=2/4/64/300 raw 宽度可编译 |
| `block_m` 48 KB SMEM 启发式（#5/#6） | 优化 | SMEM 容量 / CUDA 线程 | 换 **元素预算** 模型（探针实测 8/12 B/元素，与 dtype 有关） | **重新设计** | → §0.6-R5：`E_cap = 8192 元素/block`（竞态安全档）+ 阶梯（基础 `[1,2,4,8]` / 扩展 `[16..1024]`） |
| `threads` 参数（#6） | 优化 | CUDA 线程 | 无（本项目一维 `T.Kernel`，NPU 无 threads） | **舍弃** | harness K9 已去除；元素级并行交给 Vector 向量化（`docs/Tilelang.language/编译器提示操作/T.Kernel.md`） |
| N-tiled 回退工厂 `_logical_reduce_kernel_tiled`（#8） | 优化 | **TileLang 单 fragment 列上限 32768** + SMEM 预算 | 该触发条件在 NPU **不存在** | **舍弃（作为迁移对象）** | 其动机（fragment 列上限）不适用 ⇒ 不迁移该工厂、不照抄其 `tile_n` 语义。**但** NPU 侧因 **UB 容量**（而非列上限）另设列分块路径（§0.6-R4），二者**触发条件与结构均不同**，不得混同；其「运行 min 合并」的**在线意图**由 §0.6-R4 的运行 max 合并承接 |
| 结果块写 `T.copy`（#9） | 优化 | — | — | **等价替换** | 等价的 UB staging + 显式 slice 写（尾块 `real_m`）——模式可移植，实现形态按 NPU slice 写法适配 |
| autotune（#10） | 优化 | — | — | **舍弃** | harness K8 已移除（`init_config` 无 `tune`；启发式配置唯一入口） |
| 重 pad 布局约束（#11） | 优化 | GPU copy 布局推断 | 无此约束 | **舍弃** | GPU 特有规避；NPU 不需要 |
| M 尾块守卫缺失（源步骤 2b/6 无 `M % block_m` 检查） | 正确性（**源缺陷**） | GM 定长 block copy 无边界检查 | 有：`T.min` 标量 + src/dst 双显式 slice | **重新设计** | §0.6-R6 三件套（VP-2026-0044 待合入提案 / logsumexp 先例）；**Stage 3 不得照抄无守卫 copy 形态** |
| `lru_cache` 工厂缓存（#7） | 优化 | — | — | **保留** | 工厂层与硬件无关 |

> **无「舍弃后性能意图丢失」项**：#4 的精度意图由 §1.6.1-A3 的穷举等价性论证承接（比较谓词不需要 fp32 域）；
> #2 的对齐意图由「实列域计算 + raw 宽度可编译」承接（探针证据），其 AND 中性意图由 A2/A2b 的中性性证明承接
> （备选对齐路线仍被证明中性）；#8 的列上限规避意图随该约束消失而不再需要，其性能意图（大 N 可处理 + 运行合并）
> 由 §0.6-R4 的 UB 容量驱动列分块 + 运行 max 承接（Pareto 上更强：触发条件是真实容量约束）；#11 的布局规避意图
> 随 GPU copy 约束消失。#6/#10 由 harness 适配（K8/K9）承接。
> 每项判定依据：`migration-analysis.md` §5.3 映射表 + 本仓 `docs/Tilelang.language/` API 文档 +
> pattern-library 实测条目 + **本任务设计期探针的实测结论**（逐条注明）。

### 0.6 NPU 算法重设计

**重设计项 R1：判决与归约公式（源：fp32 上抛 → `[x!=0]` 指示化 → `reduce_min`）**

- **源方案**：`AND_j [fp32(x_j) != 0]`（unordered NE，NaN 计入非零），指示值 1.0/0.0 物化为 `(bm,N_padded)` fp32
  平面后 `reduce_min`，末位 `T.cast(result[i] > 0.5, "int8")`。
- **NPU 新算法（§1.6.0 选定 C1）**：**零命中补式**（有序 EQ）：

  $$y[i] \;=\; \neg\Bigl(\max_{j<N}\bigl[\,x[i,j] == 0.0\,\bigr]\Bigr) \;=\; 1 \;-\; \max_{j<N}\bigl[\,x[i,j] == 0.0\,\bigr]$$

  实现链：`T.vcmp(x_ub, zero_ub, mask_eq, "eq")` → `T.vcast(mask_eq, ind, round_mode="rint")`
  （i1→f32 指示量，0.0/1.0，**命中零处为 1.0**）→ `T.reduce_max(ind, mx, dim=1, clear=True)`
  （1.0 当且仅当「行内存在零」）→ `T.reshape(mx, mx_flat)` → `T.vsub(one_v, mx_flat, all_v)`
  （**(bm,) 微缓冲**上的补式）→ `T.vcast(all_v, all_f16, "rint")` → `T.vcast(all_f16, out_ub, "rint")`（int8 出栈）。
- **语义保持论证**：
  1. **数学恒等（De Morgan）**：`AND_j (x_j != 0) ≡ ¬(OR_j (x_j == 0))`，行内逐元素划分 ⇒ 恒等；
     2 值域：EQ 指示量 ∈ {0.0, 1.0}，其 **max 恒为 0.0 或 1.0（f32 精确）**，`1 - max` 亦精确 ∈ {0.0, 1.0}
     （§1.6.1-A5 机器验证 0 ulp）；
  2. **NaN 安全**：`vcmp("eq")` 为 IEEE ordered EQ（`TRAP-vcmp-ne-ordered-nan` 明证「`eq`/`le`/`ge` 为 IEEE 正确」）
     ⇒ NaN 与 0 判「不等」⇒ NaN 不计入零 ⇒ **不破坏 all**，与 torch 及源 unordered-NE 一致；
     **反面形态被否决**：直接用 `vcmp("ne")`（ordered NE，NaN 恒假）会把 NaN 判为「零」⇒ 含 NaN 行误判 False
     （§1.6.1-A1c 对照组实测 69 处差异，`CONTROL_OK`；INFO 行实测 源/候选/torch 三方 True 而 ordered-NE=False）。
     **取反不可行**：`T.vnot` dtype 表不含 bool/int1（`docs/Tilelang.language/逻辑操作/T.vnot.md` §2.2.1
     「bool/int1 = ×」）⇒ 逻辑「非」**必须**以补式（或等价的最大值判零）表达，故本形态是唯一可行方向；
  3. **补式落点**：补式在 **(bm,) 微缓冲**上完成（1 个微元 `vsub`），**避免**在全宽 `(bm,W)` 平面上做 `1 - ind`
     （那将是 +1 个 4 B/元素的 f32 平面与 +1 个全宽 pass）；若改用「nonzero 指示量 + `reduce_min`」则必须新增
     一个全宽常数量平面（C5 弃选，§1.6.0-R1）；
  4. **比较域**：`[x == 0]` 在输入原生 dtype 与 fp32 域恒等（A3，全 2¹⁶ 位型穷举），±0/NaN/亚正规/±inf 全覆盖；
  5. **边界核对**：`N=1`、全零行（y=False）、全非零行（y=True）、NaN 无零行（y=True）、NaN 混零行（y=False）、
     混合符号零（`±0`）由 `verify_equiv.py` 角点集与 §8.2 L0 覆盖。
- **收益量化**：① 省 1 个全宽 pass（fp32 上抛）——源步骤 2 的 cast 完全消除；② 补式在 **(bm,) 微缓冲**完成
  （1 个微元 op）；③ 指示平面由 `vcmp+vcast` 显式向量链产生（源为 `ite` 融合形态，见 §1.6.3-#5）。

**重设计项 R2：并行结构（源：grid × CUDA threads 两级）**

- **源方案**：`T.Kernel(ceildiv(M, block_m), threads=threads)`；`threads` 同时是归约并行宽度与布局约束参数。
- **NPU 新算法**：纯 Vector 一维 persistent kernel——`T.Kernel(num_kernels, is_npu=True)`，
  `num_kernels = min(ceil(M/block_m), 40)`（**40 = 实查 `get_aivector_core_num()`**，纯 Vector 核数，§5.5）；
  核内 `T.serial(静态上界)` 任务循环（grid-stride 映射 `task = cid + s·num_kernels`）+ 动态 guard
  `if start < M`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 示例同款 guard 形态）；
  元素级并行全部交给 Vector 向量化（显式 v-prefix 链 + reduce 原语）。
- **语义保持论证**：行之间无依赖（逐行独立判决），任务循环只改变调度不改变计算；AND 指示量与 max 的
  结合/交换/幂等性（§0.1）使分段与并行分组变化**无精度影响**（A4 位精确验证）。尾块
  `real_m = T.min(bm, M − start)` + src/dst 双显式 slice（VP-2026-0044 三件套 / logsumexp 先例），垃圾行不写出。

**重设计项 R3：宽度 / padding 策略（源：内部 `align_up(N,256)` + AND 中性掩码加载）**

- **源方案**：工厂内部派生 `N_padded = align_up(N, 256)`，`_needs_pad` 时用 `T.if_then_else` 掩码加载把
  `[N, N_padded)` 填成单位元 **1.0**，随后**在 `N_padded` 全域**做指示化与归约。
- **NPU 新算法**：**完全不 pad**——kernel 声明 `(M, N)`（raw，与 Op 契约 `_kernel_handles_padding=True`
  逐元素一致），在整个计算路径上**恰在实列域（N 列）**做比较/指示/归约；列分块路径的尾块宽度
  `tail = N − num_full·tn` 亦为实列宽度（不做任何补齐）。
- **语义保持论证**：① AND 语义对「单位元 padding」天然中性：pad 列值为 1.0 ⇒ 其「命中零」指示量为 0
  ⇒ `max_{j<N_padded}[x_pad_j == 0] = max_{j<N}[x_j == 0]`，两侧 all 逐行相等（§1.6.1-A2/A2b 机器验证 1377 行
  0 偏差，覆盖 `N ∈ {1,2,8,16,64,300,512,1024,16384}` × 3 dtype；**这正是用户提示「若走掩码路线 pad 值须为 1
  而非 0」的中性性证明**——本设计走「不 pad」路线，A2 同时证明备选的 pad=1.0 路线亦中性，pad=0.0 路线会误判）；
  ② raw 宽度的可编译性由探针实测（N=2/4/64/300 均编译通过；logsumexp N=300 的已迁移先例同向）；
  ③ 不做 pad ⇒ 少 1 条掩码分支、少 1 个 `ite` 全宽 pass，且**无越界读**（源 2a 的 `pid_m*bm+i < M`
  条件同时承担了行保护，NPU 侧由尾块三件套承担）。
- **收益量化**：消除源的 `_needs_pad` 分支与其 `T.if_then_else` 全宽 pass；`N=300` 类负载的读流量
  从 `N_padded=512` 降到 `N=300`（**−41%**，D-2 实算见 §1.6.0）。

**重设计项 R4：容量驱动的路径切换（源：fragment 列上限触发的 tiled 回退）**

- **源方案**：`_needs_tiling = N_padded > MAX_SINGLE_TILE_COLS(32768) or N_padded·elem > smem_budget`，
  触发 `_logical_reduce_kernel_tiled`（`tile_n` + `T.Serial(num_tiles)` + 逐块 `reduce_min` 后
  `acc[i] = T.min(acc[i], tile_acc[i])` 运行合并 + int8 出栈）。
- **NPU 新算法**：路径切换的**判据换成元素预算**（本任务探针实测单价 ⇒ 见 §4.5 记账口径；
  竞态安全预算 `E_cap = 8192` 元素/block）：
  - **路径 R（resident，`N <= E_cap`）**：`tn = N`（单 tile）；`bm` 由阶梯给出（`bm·N <= E_cap`）。
  - **路径 C（column-tiled，`N > E_cap`）**：`bm = 1`，`tn` = **不超 min(E_cap, N) 的最大 256 倍数且整除 N**
    （无整除因子时取最大 256 倍数），`num_full = N // tn`，`tail = N − num_full·tn`（静态尾块，
    独立静态宽度的 `x_tail` 缓冲组）；运行合并 = **max**（`T.vmax(acc, tile_mx, new_acc)` +
    `T.copy(new_acc, acc)`，**no-alias**：规避 `TRAP-vselect-inplace-carried-state` 同族陷阱），
    `acc` 初值 `T.fill(acc, 0.0)`（非负指示量的 max 单位元）。
- **语义保持论证**：分块 AND = 分段「命中零」指示量的 **max 合并**，由 max 的结合/交换/幂等性与
  「分段覆盖恰为 [0,N)、无重无漏」保证恒等（A4 机器验证：整行 vs 分段 TN ∈ {8192, 4096, 300}
  位精确，f32 running max 累加器 0 偏差）；最终补式的被减数恒为 **1**（与路径 R 一致），
  两条路径输出恒等。**注意 all 的合并是 max（命中零的 OR），而非 min**——若照抄源 tiled 的
  `T.min(acc, tile_acc)` 形态（对 `[x!=0]` 指示量取 min）在补式表示下会**语义翻错**，此处为显式差异。
- **与源 tiled 工厂的区别（显式声明）**：源工厂由 **GPU fragment 列上限**触发、**不迁移**；
  本路径由 **NPU UB 容量**触发（真实硬件约束），结构上是一个**统一的列分块 + 运行 max 合并**形态，
  与源工厂「每块归约后运行合并」在**归约域**（局部 j vs 全局）与**在线意图**上一致，
  但**判据、缓冲组、合并算子、尾块策略均按 NPU 实测重新推导**，不照抄源的 `tile_n` 语义。
- **收益量化**：使 `N` 上限从 `E_cap`（8192）提升到 UB 硬上限（实测 fp16 24576 元素/block）；
  GM 流量**不增加**（每元素仍只读一次，见 §1.6.0-R3 对比表）。

**重设计项 R5：`block_m` / 配置选择（源：48 KB SMEM 启发式 + `[1,2,4,8]` 阶梯）**

- **源方案**：`smem_per_row = N_padded × elem_bytes`；`max_block_m = 48KB // smem_per_row`；
  阶梯 `[1,2,4,8]` 取不超预算的最大者 + `threads = 128`。
- **NPU 新算法**：**元素预算阶梯**（维度从「字节」改为「元素」，因为探针实测编译器 UB 单价
  与「归约链形态 × 元素数」绑定、并按 dtype 分档——bf16 因比较上抛 f32 更贵）：
  `bm = max{p ∈ [1,2,4,8] : p·N <= E_cap}`；若 `ceil(M/bm) > 80`（>2 波 40 核）则启用扩展阶梯
  `p ∈ [16,32,64,128,256,512,1024]` 取同预算过滤器的最大者；`bm = min(bm, M)`。
- **语义保持论证**：纯配置选择，不影响语义；预算边界由探针的实测通过/失败点双向定标
  （通过：fp16 24576 / bf16 16384 / fp32 16384 元素；失败：fp16 32768 / bf16 24576 / fp32 24576）。

**重设计项 R6：M 尾块守卫（修正源缺陷）**

- **源方案**：步骤 2b/6 的 `T.copy` 无 `M % block_m` 检查（越界读/写）。
- **NPU 新算法**：`real_m = T.min(block_m, M − start)` + src/dst 双显式 slice + 垃圾行不写出。
- **语义保持论证**：仅把越界访问改为有界访问，有效行结果不变；`M=0` 另登风险（§9.2-R6）。

### 0.7 标杆实现

- **源算子**：`/workspace/TileOPs-gpu/tileops/kernels/reduction/logical_reduce.py`
  （本设计 §0.1–§0.4 的解读对象；基础工厂 `_logical_reduce_kernel` 为迁移对象，
  `_logical_reduce_kernel_tiled` 仅作耦合性判定与在线形态证据）；Op 层策略
  `/workspace/TileOPs-gpu/tileops/ops/reduction/logical_reduce.py` 与 `reduce.py`；
  对齐常数 `/workspace/TileOPs-gpu/tileops/kernels/reduction/_primitives.py`。
- **NPU 侧 harness 工件（本迁移单元的输入与落点）**：
  - 提取版 GPU 参考 kernel：`examples/TileOPs/tileops/kernels/reduction/all/_all_fwd_kernels.py`；
  - 已移植 wrapper（K5–K9 已适配、`threads` 已去除、`npub::all_logical_reduce_fwd` 命名空间隔离）：
    `examples/TileOPs/tileops/kernels/reduction/all/all.py`；
  - Op 层：`examples/TileOPs/tileops/ops/reduction/all.py`（`_kernel_handles_padding = True`、
    `_pad_value() = 1.0`、`_empty_dim_policy = "noop"`）；
  - 规格：`examples/TileOPs/tileops/manifest/reduction.yaml` 的 `AllFwdOp` 段（workloads / roofline / source）；
  - 精度/性能用例：`examples/TileOPs/tests/ops/test_all.py`、
    `examples/TileOPs/benchmarks/ops/bench_all.py`；
  - 工作负载生成器：`examples/TileOPs/tileops/workloads/reduction.py`（`AllWorkload`）。
- **golden 基准**：`torch.all(x.bool(), dim=...) -> bool`（§8.1）；harness 以 `torch.equal` **精确比对**
  （`examples/TileOPs/tests/ops/test_all.py` 的 `_exact_compare`，同时断言 dtype == bool）。
- **同族已验证参考（row-reduction 家族）**：
  `examples/count_nonzero/_logical_reduce_kernel/DESIGN.md` 与
  `examples/count_nonzero/_logical_reduce_kernel/verify_equiv.py`（**同 GPU 源工厂、同工具链**的姊妹迁移单元：
  迁移分析骨架、D-1 脚本形态、UB 预算记账口径、分核三要素）；
  `examples/count_nonzero/_logical_reduce_kernel/history_version/design_probe_ub.py`（探针模板基线）；
  `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single`（单 tile 驻留形态、
  尾块 `T.min` + 双 slice、raw N=300/32 实测通过）与
  `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_tiled`（列循环 + 运行状态更新链）；
  `examples/argmin/_argreduce_kernel/DESIGN.md` + `examples/argmin/_argreduce_kernel/verify_equiv.py`
  （910B3/CANN 9.0.0 同族**同工具链**先例：分核三要素、UB 预算记账、D-1 脚本形态）。

---
## 1. 概述

### 1.1 算子名称

`_logical_reduce_kernel`（承载 `AllFwdOp`，`torch.all` 语义；GPU 侧为 `LogicalReduceKernel` 的 `op_kind="all"` 分支）。

### 1.2 功能描述

对 2D 输入 `(M, N)` 沿最后一维做**逻辑与归约**：每行只要存在一个「零」元素，该行输出 `False`；整行全非零则输出
`True`（NaN 视为非零，不破坏结果；±0 视为零）。输出 `(M,)` bool（kernel 载体 int8）。

### 1.3 数学公式

定义式（源语义，unordered NE）：

$$y[i] \;=\; \bigwedge_{j<N} \bigl(x[i,j] \neq 0\bigr)$$

本设计采纳式（§1.6.1 优化后，ordered EQ 补式）：

$$y[i] \;=\; 1 \;-\; \max_{j<N}\bigl[\,x[i,j] == 0.0\,\bigr]$$

其中 `[·]` 为 0/1 指示量（EQ 为 IEEE 有序相等），`max` 在 f32 指示平面上做行向归约，末位做 `f32 → int8` 出栈；
`N > E_cap` 时 `max` 由「分段 max + 运行 max 合并」实现（被减数恒为 1）。

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

> 本节描述 §0.5/§0.6 决策后的 NPU 算法，与源算法存在差异处逐条注明来源。

1. **判决**（源步骤 3 → 本设计 §0.6-R1）：`T.vcmp(x_ub, zero_ub, mask_eq, "eq")`，**有序 EQ**（IEEE 正确）。
   *与源差异*：源用 `ite(x_f32 != 0.0, 1, 0)`（unordered NE + fp32 上抛）；差异来源 R1（NaN 语义等价替换）
   与 A3（原生域比较，舍弃 fp32 上抛）。
2. **指示化**：`T.vcast(mask_eq, ind, round_mode="rint")`（i1→f32，命中零处 1.0）。源为全宽 `ite` 融合形态；
   本设计改显式向量链（§1.6.3-#5，规避 `TRAP-parallel2d-bm-ge2-scalarize` 的 `ite` 标量化）。
3. **行向归约**：`T.reduce_max(ind, mx, dim=1, clear=True)`。*与源差异*：源为 `reduce_min` over `[x!=0]`；
   本设计对「命中零」指示量取 max（等价置换，补式落点见 R1 论证）。
4. **补式**：`T.reshape(mx, mx_flat)` → `T.vsub(one_v, mx_flat, all_v)`（(bm,) 微缓冲）。
5. **出栈**：`T.vcast(all_v, all_f16, "rint")` → `T.vcast(all_f16, out_ub, "rint")`（int8 载体；
   `f32 → i8` 无直接 `vcast` 表项，故走 `f32 → f16 → i8` 两步，见 §3.2）。
6. **写出**：`T.copy(out_ub[0:real_m], out[start:start+real_m])`（尾块守卫，R6）。
7. **双路径**（源用 `_needs_tiling` 分派；本设计改按**元素预算**分派，R4）：
   **路径 R**（`N <= E_cap`，单 tile 驻留 + 单次 `reduce_max`）；
   **路径 C**（`N > E_cap`，列分块 + 逐块 `reduce_max` + **运行 max 合并**，末尾一次补式）。
8. **分核**：一维 persistent `T.Kernel(num_kernels, is_npu=True)`，`num_kernels = min(ceil(M/bm), 40)`，
   核内 `T.serial` 静态上界 + grid-stride + 动态 guard（R2、§5.5）。

与源算法的结构差异一句话：**「fp32 上抛 + `[x!=0]` 指示化 + `reduce_min`（`N_padded` 域）」→
「原生域 `vcmp eq` + i1→f32 指示平面 + `max` + `(bm,)` 微缓冲补式（实列域，超预算时列分块在线 max 合并）」。**

### 1.5 数据流图

```
输入 x (M, N) 原生 dtype (fp16/bf16/fp32)
   │  路径 R:  T.copy 行块 (bm,N)          ─┐
   │  路径 C:  T.copy 列块 (bm,tn) 逐块      │
   ▼                                        │
 UB[x_ub] ──vcmp(eq, 常数量 zero_ub)──▶ UB[mask_eq (bool, 片上)]
   │
   │──vcast(i1→f32, rint)──▶ UB[ind (f32 0/1 指示量)]
   │
   │──reduce_max(dim=1)──▶ frag[mx (bm,1)]
   │                      │ 路径 C: vmax 运行合并 acc = max(acc, mx)（no-alias）
   │                      ▼
   │                  reshape → frag[mx_flat (bm,)]
   │                      │
   │                  vsub(one_v, mx_flat) → frag[all_v (bm,) f32 ∈ 0.0/1.0]
   │                      │
   │                  vcast(f32→f16) → vcast(f16→i8) → UB[out_ub (bm,) int8]
   ▼
 T.copy(out_ub[0:real_m], out[start:start+real_m]) ──▶ 输出 out (M,) int8（wrapper 转 bool）
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先**调研**——对「逻辑与行归约」这一数学语义的算法族回答调研四问（1.6.0），
> 选定算法族；再在保证数学等价的前提下优化公式（1.6.1）；再把保留下来的计算全部交给向量单元
> （1.6.2）；最后回答「在哪个轴上向量化」（1.6.3）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。
> 分析对象 = §1.6.0 选定、经 §0.6 重设计落地的 NPU 算法（非源公式直译）。

#### 1.6.0 算法调研（Algorithm Research）⭐

> **调研对象**：「逐行逻辑与归约」（`out[i] = AND_j (x_j != 0)`）这一数学语义的**算法族**（GPU 源算法只是基线候选之一）。
> 信息源：`references/algorithm-candidates.md` 的 `ALG-reduction` 命中行（R1「等价化简 / 指示量置换」、
> R2「分块归约 + 合并」、R3「扫描遍数 / 跨核归并代价」、R4「水平 vs 垂直 lane 映射」+ row-reduction 迁移基准形态）
> - `ALG-elementwise`（R1「公共子表达式消除 / cast 链合并」）+ `kb_search.py` 预注入条目
> （`CONST-reduce-ub-per-element` / `CG-2026-0026` / `CG-2026-0027` / `CG-2026-0028` / `VP-2026-0263`）
> - 本仓 `examples/`（count_nonzero 姊妹单元、logsumexp single/tiled、argmin/argmax 同族、`_primitives.py`）
> - pattern-library（constants / traps-compiler / cases / layout）+ **用户输入公式与源算法（基线）**
> - 恒等变形空间自查（De Morgan 补式、直判 ordered NE、幅度式 min|x|、位域式、乘积式、vselect 指示量、
> 整数域累加、分段在线合并、排序/选择结构）。
> all 属**规约类**（多步：判决 + 指示化 + 行归约）→ **完整调研**（R1 候选表 + R3 四口径对比 + R4 逐候选评估）。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| 0 | **基线：GPU 源（fp32 上抛 + `[x!=0]` 指示化 + `reduce_min`）** | `AND_j [fp32(x_j) != 0]`，指示平面 1.0/0.0（`N_padded` 域），末位 `T.cast(result > 0.5, "int8")` | — | — | ✅（基线） |
| 1 | **C1 零命中补式 + 原生域比较（本文主选）** | `y = 1 − max_j [x_j == 0]`（原生 dtype 域，ordered EQ；补式落 `(bm,)` 微缓冲） | 数学恒等（De Morgan + 行内划分）；`eq` 为 IEEE 正确 ⇒ NaN/±0 与源/torch 一致（A1/A5 机器验证） | 省 1 个全宽 pass（fp32 上抛）；补式落 `(bm,)` 微缓冲；staging 字节减半 | ✅ |
| 2 | C2 直判式 `vcmp("ne")` + `reduce_min`（nonzero 指示量） | `AND_j [x_j != 0]`（ordered NE） | **数学上等价但硬件语义相反**：`vcmp("ne")` 为 ordered NE（NaN 恒假）⇒ NaN 被判「零」⇒ 含 NaN 行误判 False | 结构最短（省补式），但**语义破裂** | ❌（A1c 对照实测 69 处差异；`TRAP-vcmp-ne-ordered-nan` 明证） |
| 3 | C3 幅度式 `min_j |x_j| != 0` | 等价于 `min_j |x_j|` 然后判零 | 对有限值/±0/±inf 成立；但 (`零` 与 `NaN` 同行) 时结果依赖 `reduce_min` 的 NaN 传播语义（**未文档化**：若 NaN 被忽略则 min=0 ⇒ False 正确；若 NaN 传播则 min=NaN ⇒ 判「非零」⇒ **误判 True**） | 需 +1 个全宽绝对值 pass + 依赖未文档化 NaN 语义 | ❌（未文档化假设 + 额外面；`docs/Tilelang.language/规约操作/T.reduce_max.md` 未记录 NaN 传播条款） |
| 4 | C4 乘积式 `prod_j [x_j != 0] == 1` | 连乘 nonzero 指示量 | 数学恒等 | **净亏**：nonzero 指示量需补式 ⇒ +1 个全宽平面；vmul 链引入乘法器与更长的向量链 | ❌ |
| 5 | C5 `vselect` 构造 nonzero 指示量 + `reduce_min`（Path B） | `vselect(mask_eq, 0, 1, nz_ind)` → `reduce_min` | 数学恒等（省去微缓冲补式） | 少 1 个微元 op，但需 **2 个全宽常数量平面**（zero/one）且**该链的 UB 单价未实测**（本任务探针标定的是 `vcmp+vcast` 链 8/12 B/元素） | ❌（无实测常数支撑 + 多 1 个全宽平面；登记为 Stage 4 备选，见 §6.3） |
| 6 | C6 位域式 `[bits(x) & 0x7FFF… != 0]` | `vbitcast` → `vand` 掩码 → 比较 | 对 ±0/NaN/±inf 成立（IEEE 位型） | **净亏**：+2 个全宽 pass（bitcast + and） | ❌ |
| 7 | C7 整数域判决（int16/int32 比较后归约） | 指示化到整数域后 `reduce_max` | 数学恒等 | 零收益（指示量与 max 本就在 f32 精确域；`reduce` dtype 表支持整型但无精度需求） | ❌（结构原因：无精度动机） |
| 8 | C8 分段（列分块）在线 AND + 运行 max 合并（路径 C） | `acc = max(acc, tile_max)` 逐块合并，末位一次补式 | 数学恒等（max 结合/交换/幂等 + 分段覆盖无重无漏；A4 机器验证位精确） | 容量：中间缓冲 O(N) → O(tile)，使大 N 可处理 | ✅（`N > E_cap` 时） |
| 9 | C9 N-split 跨核 partial + merge（两段 kernel） | 列分片并行 `(M,nchunk)` 命中零 workspace + merge kernel | 数学恒等（分段 max 合并） | 小 M 大 N 场景核利用率提升 | ❌（本算子 manifest M >= 32 ⇒ 核利用率已 >= 80%，workspace 往返 + 第二次 kernel 启动为净亏） |
| 10 | C10 排序/选择类结构（排序后判零段） | `argsort` 后定位零 | 数学恒等 | O(N log N) 远劣于 O(N)，且 N 维无局部性 | ❌（复杂度劣） |

**R2 有没有在线算法？结论：有，且本算子天然在线。**
结构判据（`algorithm-research.md` §3-R2）：AND 可分解为可增量合并的 **running 统计量**——
「命中零」指示量的 **max**（`acc = max(acc, tile_max)`）即是结合/交换/幂等的合并算子；
逐元素判决 `[x_j == 0]` **无跨元素依赖** ⇒ 单遍即天然在线。
收益口径：**扫描遍数 1**（源实现同为 1 遍，故在线化不减少遍数——这是 all 与 logsumexp（两遍 max→exp-sum）
的关键差别）；中间缓冲 `O(N) → O(tile)`——**在线形态的价值在本算子上是「容量」而非「遍数」**：
它是路径 C 存在的前提（`N > E_cap` 时无在线形态就必须落 GM 或分两次读）。
迁移任务侧，源 `_logical_reduce_kernel_tiled` 的 `acc[i] = T.min(acc[i], tile_acc[i])`（`T.Serial(num_tiles)` 循环内）
**即为源算法族存在在线变体的直接证据**（该工厂不迁移，§0.5；其运行合并算子由 min 改为 max，§0.6-R4）。

**R3 算法复杂度对比**（四口径；per-row 计，M 行并行摊派；本算子为**带宽受限**：
roofline `flops = M·N`（每输出 N−1 次逻辑与）vs `bytes = M·N·elem_bytes + M`，访存主导；
口径：FLOPs 按「每元素 1 次比较 + 1 次指示量归约合并」折算，**不含**指示化 cast 的搬运；
访存量含输入读 + 输出写，**不含** UB 内部流量）：

| 算法候选 | FLOPs（per row） | 访存量 Bytes（GM） | GM 扫描遍数 | 中间缓冲峰值（per row） | 可并行度 / 跨核代价 |
|---------|------------------|--------------------|-------------|------------------------|---------------------|
| 基线（源：fp32 上抛 + 指示化 + 单次 `reduce_min`，`N_padded` 域） | `2N_padded` | `N_padded·elem + 1`（`N=300` 时 `N_padded=512` ⇒ **+71%**） | 1 | `x_f32` 4 B/elem + `bool_vals` 4 B/elem（均 `(bm,N_padded)` 平面） | M/bm 行块独立；无跨核 |
| **C1 补式 + 原生域比较（本文主选，路径 R）** | `2N` | `N·elem + 1` | 1 | `x_ub` elem + `zero_ub` elem + `mask` 1 + `ind` 4 B/elem（探针实测编译器侧统一 **8 B/elem（fp16/fp32）/ 12 B/elem（bf16）**） | 同上；无跨核 |
| C1 + 路径 C（列分块在线 max 合并，`N > E_cap`） | `2N + O(num_full)`（微元 max 合并） | `N·elem + 1`（**仍 1 遍**：每元素只读一次） | 1 | `O(tn)`：`tn` 档元素 × 8/12 B/elem | 同上；`acc` 片上驻留，无跨核 |
| C3 幅度式 | `3N`（+1 个全宽 abs pass） | `N·elem + 1` | 1 | `abs` 平面 4 B/elem 额外 | 同上（另有 NaN 语义不确定性） |
| C5 vselect 指示量 | `2N` | `N·elem + 1` | 1 | `zero`/`one` 两个全宽常数量 + `nz_ind` ⇒ 高于 C1（单价未实测） | 同上 |
| C9 N-split partial+merge（两段 kernel） | `2N` | `N·elem + 1` **+ workspace 往返** `M·nchunk·(1+1)B × 2` | 1（+workspace 2 遍） | `O(tn)` + workspace | 跨核 workspace + 第二次 kernel 启动 + 合并正确性面 |
| （对照）源 tiled 回退（不迁移） | `2N_padded` | `N_padded·elem + 1` | 1 | `(bm,tile_n)` 平面组 | 同 C1；触发条件（fragment 列上限）在 NPU 不存在 |

**R4 硬件亲和性评估**（检查清单逐项；实测常数引用 `pattern-library/constants.md` 条目 + 版本戳；
**本任务设备为 910B3 / CANN 9.0.0**，凡绑定 910B2C / CANN 8.5.0 的常数标注
「未实测假设：沿用同代 910B 曲线量级，Stage 4 校准」）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（`T.Parallel` 融合 `T.if_then_else` 物化布尔平面） | ⚠️ 逐元素 `ite` 形态在 `bm>=2` 标量化（`TRAP-parallel2d-bm-ge2-scalarize`：触发项含循环体内 `ite`） | ⚠️ `x_f32` + `bool_vals` 双 4 B/elem 平面 | 需 `N_padded` 256 倍数 | ✅ | ✅ | ❌ 形态需重设计（→C1） |
| **C1 补式 + 原生域显式向量链（路径 R）** | ✅ 纯 Vector（`vcmp`/`vcast`/`reduce_max` 全向量原语；`T.reduce.md` dtype 表 fp16/fp32 ✓、**bf16 ✗**——本设计**不在 bf16 域归约**（指示平面恒 f32），故规避；`vcmp.md` dtype 表 fp16/bf16/fp32 均 ✓） | ✅ **探针实测 8 B/elem（fp16/fp32）、12 B/elem（bf16）**；单 block 实测通过点 fp16 24576 / bf16 16384 / fp32 16384 元素；`E_cap = 8192` 元素（竞态安全档，§4.5） | ✅ **无需任何宽度对齐**（探针实测 raw N=2/4/64/300 均可编译；`T.copy` 切片满足 32 B 尾轴） | ✅ M/N/tile 边界全为工厂期常量 | ✅ auto-multi-buffer（Developer 自动施加）；无跨引擎 | ✅ **路径 R 选定** |
| C1 + 路径 C（列分块在线 max 合并） | ✅ 同 C1 + `(bm,1)` 微元 `vmax`/`copy` 合并链（**no-alias**） | ✅ tile 工作集 `tn × 8 B <= 64 KB`（fp16/fp32；bf16 按 12 B/elem ⇒ `tn × 12 B <= 96 KB`） | ✅ `tn` 为 256 倍数；尾块为实列宽度（探针证 raw 宽度可编译） | ✅ `num_full/tail` 全静态（工厂期）；`bm>1` 多 tile 由姊妹单元探针实测可编译（`CG-2026-0014` 不复现） | ✅ tile 循环 auto-multi-buffer | ✅ **路径 C 选定**（`N > E_cap`） |
| C3 幅度式 | ✅ Vector | ⚠️ +1 个全宽 f32 平面 | — | — | — | ❌（NaN 语义未文档化 + R3 量化） |
| C5 vselect 指示量 | ✅ Vector | ⚠️ 需 2 个额外 `(bm,W)` 常数量平面；单价未实测 | — | — | — | ❌（无实测常数支撑，登记 Stage 4 备选） |
| C9 N-split 跨核 | ✅ | ⚠️ workspace GM 往返（`M·nchunk·2B × 2`；本算子 M >= 32 时极小但需第二次 kernel 启动） | — | ✅ | ⚠️ 两 kernel + 合并正确性面 | ❌（核利用率已 >= 80%，无收益；R3 量化） |
| C10 排序/选择 | ✅ | ⚠️ O(N) 中间缓冲 | — | — | — | ❌（O(N log N)，R3 量化） |
| （对照）源 tiled 回退 | ✅ | ✅ | ✅ | ✅ | ✅ | ⏸ 不迁移（触发条件不存在，§0.5）；其在线意图由 C8 承接 |

**调研结论**：选定 **C1（零命中补式 + 原生 dtype 域比较 + 行向 `reduce_max` + `(bm,)` 微缓冲补式），
按元素预算双路径（路径 R 驻留 / 路径 C 列分块在线 max 合并）** 的算法族。

- **关键依据**：R3 表——访存遍数（1）与带宽（`N` 而非 `N_padded`）均不劣于基线，指示平面 1 个
  （基线 2 个 4 B/elem 平面）；R4——全 Vector 原语可达（`vcmp`/`vcast`/`reduce_max` 均有文档与家族先例）、
  容量经**本任务探针实测**定标（8/12 B/elem，按 dtype 分档）、静态边界满足一维 Kernel 约束、
  无需任何对齐 pad（探针证据 + A2/A2b 中性性证明）。
- **与基线的结构差异一句话**：「fp32 上抛 + `[x!=0]` 指示化 + `reduce_min`（`N_padded` 域）」→
  「原生域 `vcmp eq` + i1→f32 命中零指示平面 + `reduce_max` + `(bm,)` 微缓冲补式（实列域，
  超预算时列分块在线 max 合并）」。
- **调研范围（负向断言与「无更优替代」的查证范围）**：`references/algorithm-candidates.md` 的
  `ALG-reduction` / `ALG-elementwise` / row-reduction 迁移基准形态条目；`kb_search.py` 预注入 5 条
  （`CONST-reduce-ub-per-element`（**先核对形态适用域后否决其对本 2-D 行归约的适用性**，见 §9.1-C-1）/
  `CG-2026-0026` / `CG-2026-0027` / `CG-2026-0028` / `VP-2026-0263`（pending））+ 本设计新增检索
  （`all logical reduce reduction bool keepdim 归约 npuir developer`）；
  `examples/`（count_nonzero 姊妹单元、logsumexp single+tiled、argmin、argmax、`_primitives.py`）；
  pattern-library（constants 全条目、traps-compiler、traps-runtime、cases、layout）；
  恒等变形空间（C2–C10 全枚举，见 R1 表）；结构判据（在线变体存在性 / 单目的性 / 排序下界 / 跨核收益前提）。
  **R1 已枚举 11 个候选（基线 + C1–C10）**，无「无更优替代」空白断言。
- **源算法优化手段承接（§0.4 逐项）**：#1 SMEM tiling → UB staging（等价替换）；#2 256 对齐 pad → 舍弃
  （实列域计算，A2/A2b 验证 pad=1.0 中性）；#3 指示化 + 复用归约原语 → **保留**（算法层，改为
  `vcmp` + `vcast` 显式链 + `reduce_max`）；#4 fp32 上抛 → 舍弃（A3 穷举验证比较域等价）；
  #5/#6 预算启发式与 threads → 重设计为元素预算阶梯 + 一维 persistent（R2/R5）；#7 `lru_cache` 保留；
  #8 tiled 回退 → 作为迁移对象舍弃、其「大 N 可处理 + 运行合并」意图由 R4/C8（UB 容量驱动列分块 +
  运行 max）承接（合并算子由 min 改 max，见 R4 论证）；#9 出栈 → UB staging + 显式 slice；#10 autotune 舍弃（K8）；
  #11 GPU 布局约束舍弃。**无静默丢弃项**。

**设计期 roofline（D-2；协议 `_shared/standards/hardware-cost-model.md` §2；常数带版本戳）**：

| workload（manifest / 测试） | 路径 / 配置 | 流量项（GM 字节 ÷ 有效带宽） | 发射项（向量 op 数 × 常数 ÷ 核数） | 估算下界（µs） |
|---|---|---|---|---|
| smoke `(32,256)` fp16/fp32 | R，bm=8，4 block | 16.4 KB ÷ (33.9 GB/s × 4)〔`CONST-mte2-degradation`〕≈ **0.12** | 4 block × 约 10 op ÷ 4 × 0.25–0.5〔`CONST-msprof-task-floor-910B3` / `CONST-vector-launch-overhead`〕≈ **2.5–5.0** | **约 5–7**（含 2.4 µs 固定地板） |
| mask-validation-4k `(32,4096)` bool→f32 | R，bm=2，16 block | 524 KB ÷ (33.9 × 16) ≈ **0.97** | 16 × 10 ÷ 16 × 0.25–0.5 ≈ **2.5–5.0** | **约 5–7** |
| mask-validation-32k `(32,32768)` bool→f32 | C，bm=1，tn=8192，4 tile，32 block | 4.19 MB ÷ (30.2 × 32) ≈ **4.3** | 32 × 4 × 12 ÷ 32 × 0.25–0.5 ≈ **12–24** | **约 12–24** |
| 3d-multidim `(4,128,4096)` dim=[0,2] ⇒ M=128, N=16384，bool→f32 | C，bm=1，tn=8192，2 tile，128 block → persistent 40（核内 serial 4） | 8.39 MB ÷ (30.2 × 40) ≈ **6.9** | 128 × 2 × 12 ÷ 40 × 0.25–0.5 ≈ **19–38** | **约 19–38** |
| 测试 `(128,300)` fp16 | R，bm=8，16 block | 76.8 KB ÷ (33.9 × 16) ≈ **0.14** | 16 × 10 ÷ 16 × 0.25–0.5 ≈ **2.5–5.0** | **约 5–7** |

> **估算下界（汇总）**：`估算下界 = max(流量项, 发射项, 容量项) = 发射项主导`（`(32,32768)` 与 `3d-multidim` 档
> 两项相差约 3–5 倍，发射项主导）；**容量项不适用**——纯 Vector 单引擎、无跨引擎 S/P 往返
> （`CONST-store-fixpipe-gm-only` 不适用）。
> 发射项按 0.25–0.5 µs/op 双档给出（`CONST-vector-launch-overhead` 为 0.5 µs/op 的**串行上界模型**，
> `CONST-msprof-task-floor-910B3` 实测逐元素 pass 单价 0.23–0.3 µs/pass 为低档）。
> **同形校准锚点**：`(2048,4096)` fp16 同族（argmin，bm=2、1024 block、约 10 op/block、48 核）实测 47 µs /
> 估算 56 µs；姊妹 count_nonzero 同工厂同形任务（`(2048,4096)` 估算 45–90 µs）在本设备上按同口径估算，
> 本算子 op 数相近（约 10/block，多 1 个微元 vsub、少 1 个微元 vsub 与 vadd 相抵），故量级一致。
> **常数适用性声明**：`CONST-mte2-degradation`（MTE2 曲线）与 `CONST-vector-launch-overhead` 绑定
> **910B2C / CANN 8.5.0**，本任务为 **910B3 / CANN 9.0.0** ⇒ 按「未实测假设：沿用同代 910B 曲线量级」引用，
> Stage 4 须按 `hardware-cost-model.md` §3 回填对账；`CONST-msprof-task-floor-910B3` 为**本工具链同设备**
> 实测（可信档）；核数（实查 40）与 UB 容量（实查 196608 B）**已本机实查复核**；
> **指示链 UB 单价 8/12 B/元素为本任务探针实测**（非假设；`CONST-reduce-ub-per-element` 的整平面
> ~128 B/元素模型**经探针证不适用于本 2-D 行归约形态**——该条目的适用域限定为 `(CH,1,H*W)` 3-D 整平面归约，
> 见 §9.1-C-1）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象：§1.6.0 选定算法 C1（+ 双路径落地形态）。逐项四要素；**等价性机器验证（D-1）**：
> `verify_equiv.py`（torch 2.9.0+cpu、fp64 参照、bool 精确比对、无 NPU 依赖），结果表内嵌于下。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| **A1** | De Morgan 补式 + IEEE `eq` 判决（承接 §0.6-R1） | `y = AND_j [fp32(x_j) != 0.0]`（unordered NE，指示值 1.0/0.0 物化在全宽平面） | `y = 1 − max_j [x_j == 0.0]`（**原生 dtype 域**，ordered EQ） | ① De Morgan + 行内划分：`AND_j (x_j != 0) ≡ ¬(OR_j (x_j == 0))` ⇒ 恒等；② 判据正确性：`T.vcmp("eq")` 为 **IEEE 正确**（`traps-compiler.md` 的 `TRAP-vcmp-ne-ordered-nan` 原文「eq/le/ge 为 IEEE 正确」），NaN==0 为假 ⇒ NaN 不计入零 ⇒ all 保持 True，与 torch 及源 unordered-NE 一致；③ **取反路线不可用**：`T.vnot` dtype 表不含 bool/int1（`docs/Tilelang.language/逻辑操作/T.vnot.md` §2.2.1），故「非」只能以补式表达；④ 机器验证：A1 组 1377 行 0 偏差，A1c 对照组证明「有序 NE 路线」在 NaN 行确有 69 处差异（判别力成立），INFO 行三方一致 | ① 补式在 **(bm,) 微缓冲**完成（1 个微元 `vsub`），**避免**在全宽 `(bm,W)` 平面上做 `1 − ind`（那将是 +1 个 4 B/elem 平面与 +1 个全宽 pass）；② 与 A3 合并后整链少 1 个全宽 pass（fp32 上抛） |
| **A2** | 实列域计算（舍弃源的 256 对齐 pad 域） | `AND_{j<N_padded} [x_pad_j != 0]`，`N_padded = align_up(N,256)`，pad 值 **1.0** | `AND_{j<N} [x_j != 0]`（恰 N 列） | pad 列恒为 **+1.0（AND 中性）** ⇒ 其「命中零」指示量恒为 0 ⇒ `max_{N_padded}[x==0] = max_N[x==0]` ⇒ 两侧逐行相等（A2 组 1377 行 0 偏差，覆盖 `N ∈ {1,2,8,16,64,300,512,1024,16384}` × 3 dtype）。**同时证明备选 pad 路线的中性性，且 pad 值必须为 1 而非 0** | 省 1 条掩码加载分支 + 1 个 `ite` 全宽 pass；`N=300` 类负载读流量 `512 → 300` 列（**−41.4%**）；免 kernel 侧宽度特化代码 |
| **A2b** | 窄行 32B 对齐 pad 的中性性 | 单段实列域 | `W = align_up(N,16)` 的 pad 域 | 同上（pad=+1.0 中性）；A2b 组 1377 行 0 偏差 | 支撑「备选对齐路线」与 Stage 4 简化候选的中性性（本设计主选**不使用**对齐 pad） |
| **A3** | 比较域下放到原生 dtype（舍弃源的 fp32 上抛） | `[fp32(x_j) == 0]`（x 先 cast fp32 的全宽 pass） | `[x_j == 0]`（fp16/bf16/fp32 原生域） | fp16→fp32 与 bf16→fp32 均为**精确单射**且保 ±0 与 NaN（`TRAP-vcmp-ne-ordered-nan` 同族对 `eq` 的 IEEE 语义明证；`vcmp.md` dtype 表 fp16/bf16/fp32 均 ✓）⇒ 零判决逐元素恒等；**穷举验证**：fp16 与 bf16 **全部 2¹⁶ 位型**（共 262144 行）比较结果与 AND 判决 0 差异（A3 组） | 省 1 个全宽 cast pass；`x_ub`/`mask` 之外的指示平面恒 f32（不随输入 dtype 放大）；**注意 bf16 不被 `reduce` 支持**——本设计归约域恒为 f32 指示平面，故不受该限制（`docs/Tilelang.language/规约操作/T.reduce.md` §2.2.1），但 bf16 的 `vcmp` 被 lowering 上抛 f32 ⇒ 单价升为 12 B/elem（探针实测，§4.5 记账） |
| **A4** | 分段在线 AND + f32 running-max 合并（支撑路径 C） | 整行单次 `max_j [x_j == 0]` | `num_full` 段「命中零」指示量的分段 max 再取 max（`acc = max(acc, tile_max)`），末位一次补式 | ① 分段覆盖恰为 `[0, N)`（无重无漏）+ max 的**结合/交换/幂等**⇒ 恒等；② 累加器恒为 0.0/1.0（f32 精确，无舍入）；③ 机器验证：整行 vs 分段 `tn ∈ {8192, 4096, 300}`（含非整除尾块 300）390 行 **0 偏差**，且 f32 running max 与整行「命中零」逐行位精确 | 支撑 `N > E_cap` 的容量路径（中间缓冲 O(N) → O(tile)）；GM 流量**不增加**（仍 1 遍） |
| **A5** | 微 epilogue 的精确性与 cast 链（int8 载体） | 源：`T.cast(result > 0.5, "int8")`（`T.Parallel(bm)` 逐元素比较 + cast） | `all_v = 1 − max`（f32）→ `vcast` f32→f16（rint）→ `vcast` f16→i8（rint） | ① `max ∈ {0.0, 1.0}` ⇒ `1 − max ∈ {0.0, 1.0}` 精确（0 ulp，A5 组对值域断言 0 违反）；② `f32 → f16 → i8`（rint）对 0.0/1.0 逐位还原 bool（A5 组 1377 行 0 差异）；③ 依据：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` 的 `f32→f16`、`f16→i8` 行；**`f32→i8` 无表项**，故必须两步 | 两条微元 cast 替代「逐元素比较 + cast」的 `T.Parallel(bm)` 标量形态；值域精确性由 A5 保证 |

**机器验证结果表**（`verify_equiv.py`，2026-10-16 执行，**exit 0**；torch 2.9.0+cpu、fp64 参照、bool 精确比对；
shape 覆盖 `(1,1)/(1,2)/(3,8)/(4,16)/(8,64)/(16,256)/(32,300)/(64,512)/(7,1024)/(129,512)/(5,16384)`
× {fp16,bf16,fp32} + 角点行集〔±0（两种符号）/±inf/NaN/NaN 无零行/NaN 混零行/亚正规〔最小正规与最小亚正规〕/
最大有限（fp16 与 fp32）/全零/全一/奇偶交替/首末单零/单 NaN/全 NaN/亚正规稀疏〕）：

| 验证项 | 用例行数 | 最大偏差 | 违反率 | 结论 |
|--------|---------|---------|--------|------|
| A1 补式（`¬any[x==0]`，ordered EQ）vs 基线（`AND[x!=0]`，unordered NE） | 1377 | 0 | 0/1377 | **EQUIV_PASS** |
| A1c 判别力对照：有序 NE 路线（**故意错误候选**，NaN 判为零） | 1377 | — | **69 处**（期望 > 0） | **CONTROL_OK**（若为 0 违反则脚本判 `CONTROL_BROKEN` 并自身失败） |
| A2 实列域（N）vs 256 对齐 pad 域（`N_padded`，pad=+1.0） | 1377 | 0 | 0/1377 | **EQUIV_PASS** |
| A2b 窄行 32B 对齐 pad（`W = align_up(N,16)`，pad=+1.0） | 1377 | 0 | 0/1377 | **EQUIV_PASS** |
| A3 原生 dtype 比较 vs fp32 上抛比较（**全 2¹⁶ 位型穷举** × {fp16,bf16}，mask 与 AND 判决双口径） | 262144 | 0 | 0/262144 | **EQUIV_PASS** |
| A4 分段 AND + f32 running-max 合并 vs 整行（`tn = 8192/4096/300`，含非整除尾块） | 390 | 0 | 0/390 | **EQUIV_PASS** |
| A5 微 epilogue 链 `f32 0/1 → f16 → i8` 位精确 + 值域精确性 | 1377 | 0 | 0/1377 | **EQUIV_PASS** |
| INFO：NaN 语义 `[NaN,1,-1,5.96e-8]`（NaN 无零行） | 1 | — | — | **INFO**：源(unordered NE)=True / **采纳候选(ordered EQ 补式)=True** / 有序 NE 路线=False / `torch.all`=True ⇒ 采纳候选与 torch 及源三方一致，反之有序-NE 路线与两者分歧（非门禁项） |
| INFO：NaN 语义 `[NaN,0,1,-0]`（NaN 混零行） | 1 | — | — | **INFO**：四方一致 False（零主导） |
| **合计（门禁等价组 A1/A2/A2b/A3/A4/A5）** | **268042** | — | **0 项失败** | **ALL EQUIV_PASS**（另 +1379 对照/INFO 行，见上；对照组 `CONTROL_OK`） |

**优化结论**：采纳 **A1 + A2 + A2b + A3 + A4 + A5** 共 6 项（其中 A2b 服务于备选对齐路线的中性性，
A4 服务于路径 C 的正确性，A5 界定 int8 载体的出栈链）。优化后公式 = §1.3 第二式：
`y[i] = 1 − max_{j<N}[x[i,j] == 0.0]`（原生 dtype 域比较；`N > E_cap` 时按列分段运行 max 合并，被减数恒为 1）。
**被否决项**：R1 表中的 C2（直判 ordered NE，语义破裂、A1c 实证 69 处差异）、C3（幅度式，依赖未文档化的
`reduce_min` NaN 传播语义 + 1 个全宽 pass）、C4（乘积式，+1 全宽平面）、C5（vselect 指示量，无实测单价 +
多 1 个全宽常数量，登记 Stage 4 备选）、C6（位域式，+2 pass）、C7（整数域判决，无精度动机）、
C9（跨核 N-split，核利用率已 >= 80% 无收益）、C10（排序，复杂度劣）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点对象：§1.4 两条路径的**全部**循环与标量计算点（含 §0.3 源步骤映射）。
> 替代 API 均有 `docs/Tilelang.language/` 或 `examples/` / `testing/` 佐证（列内标注）。

| # | 计算点 | 原实现形态（源 / 朴素形态） | 向量替代方案 | 替代结论 | 不可替代理由 |
|---|--------|---------------------------|-------------|---------|--------------|
| 1 | 零命中判决（源步骤 3 的 `ite` 全宽循环） | 2D `T.Parallel` 融合 `T.if_then_else`（`bm>=2` 标量化，`TRAP-parallel2d-bm-ge2-scalarize`） | `T.vcmp(x_ub, zero_ub, mask_eq, "eq")`（`docs/Tilelang.language/比较操作/T.vcmp.md`；bool 结果留片上，不可落 GM——同文档 §2.3） | ✅ | — |
| 2 | 指示化（1.0/0.0 平面） | 朴素：`T.Parallel` 赋值 `1.0/0.0` | `T.vcast(mask_eq, ind, round_mode="rint")`（i1→f32，`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 表含 `i1 → f32`） | ✅ | — |
| 3 | 行归约（源步骤 4） | 源已是 `reduce_min` 原语（朴素替代为标量扫描） | `T.reduce_max(ind, mx, dim=1, clear=True)`（**同 dtype 约束**：f32→f32；`docs/Tilelang.language/规约操作/T.reduce_max.md`、`T.reduce.md` §2.2.1） | ✅ | — |
| 4 | 补式（`1 − max`） | 朴素：全宽平面 `1 − ind`（+1 全宽 pass） | `T.vsub(one_v, mx_flat, all_v)`（`(bm,)` 微缓冲；`docs/Tilelang.language/数学操作/T.vsub.md`）；`one_v` 为任务不变量（循环外一次 `T.fill`） | ✅ | — |
| 5 | 零常数量构造 | 朴素：逐元素写 0 | `T.fill(zero_ub, tir.const(0, dtype))`（**值须为类型化常量**——姊妹单元探针实测 Python 字面量 `0` 被前端拒绝 `value must be a tir.PrimExpr`，须用 `tir.const(0, dtype)`） | ✅ | — |
| 6 | fp32 上抛（源步骤 2a/2b） | `T.Parallel` 全宽 cast 循环 | **消除**（A3：原生域比较，全 2¹⁶ 位型穷举验证） | ✅（消除） | — |
| 7 | 分段运行合并（路径 C） | 源 tiled：`acc[i] = T.min(acc[i], tile_acc[i])`（原位、且算子对补式表示已翻错） | `T.vmax(acc, mx, new_acc)` + `T.copy(new_acc, acc)`（**no-alias**；`docs/Tilelang.language/比较操作/T.vmax.md` 的 fp32 ✓；规避 `TRAP-vselect-inplace-carried-state` 同族原位陷阱） | ✅ | — |
| 8 | int8 出栈（源步骤 5+6） | `T.Parallel` 逐元素 `T.cast(result > 0.5, "int8")` + 尾块无守卫 `T.copy` | `T.reshape(mx → mx_flat)` + `T.vsub`(补式) + `T.vcast(f32→f16, "rint")` + `T.vcast(f16→i8, "rint")` + `T.copy(out_ub[0:real_m], out[start:start+real_m])`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` 的 `f32→f16` 与 `f16→i8` 行；`docs/Tilelang.language/shape操作/T.reshape.md`；A5 组验证位精确） | ✅（向量化） | — |
| 9 | 尾块行数 / 任务 guard | 源缺失（越界，§0.4 源缺陷） | `T.min` 标量 + `if start < M` 动态 guard（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 示例先例） | ❌（保留） | **block 级边界处理**：依赖运行时 task 的任务边界，非逐元素热点（每 block 1 次）；属 VP-2026-0044 三件套标准形态（serial 上界静态 + guard 动态） |
| 10 | tile 循环控制（路径 C） | `T.Serial(num_tiles)`（源 tiled） | 保留 `T.serial(num_full)`（静态上界） | ❌（保留） | **tile 级顺序依赖**：`acc` 是循环携带状态（在线 max 的载体）；`num_full` 为工厂期静态常量（静态边界要求），循环体内逐元素计算已全部向量化 |
| 11 | host 侧 `bm / tn / num_full / tail` 选择 | Python 工厂期计算 | — | ❌（保留） | **host 元数据计算**（不在 kernel 内；每 `(M,N,dtype)` 一次，`lru_cache` 命中后零成本） |

**向量化结论**：逐元素计算点已**全部向量化**（#1–#8，API 均有文档 + 家族先例佐证）；
保留 **3 处**标量/循环（#9 任务 guard / #10 tile 循环控制 / #11 host 元数据），理由类别分别为
「block 级边界处理」「tile 级顺序依赖」「host 元数据」——三者发射量均为每 block/tile 1 次量级，
**非逐元素热点**。源实现中的逐元素标量热点（源步骤 3 的 `ite` 融合形态在 `bm>=2` 会整体标量化）
已被 #1 + #2 的显式向量链替代；源步骤 5 的 `T.Parallel(bm)` 逐元素比较 + cast 已由 #8 的两条向量 cast 替代。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（`(M,N)` 行主序进 / `(M,)` int8 出，wrapper 转 bool）；**核内布局与 lane 映射是设计变量**。
> 本算子 2D `(M,N)`、规约维 = 最后一维 ⇒ 按**规约类必选候选**枚举（水平归约 vs 垂直扫描为正交决策）。

**轴质量评分**：

| 评分项 | 水平归约（lane→N 轴，`reduce_max dim=1`） | 垂直扫描（lane→M 行，串行步进 N 维） | 说明 |
|--------|------|------|------|
| 整除性 | `N` 任意（探针证 raw N=2/4/64/300/4096/16384/32768 均可编译）；tn 路径取 256 倍数 | 同理 | 无差异 |
| 尾 lane 浪费率 | tn 路径为 256 倍数 ⇒ 0 浪费；raw `N` 路径（如 300、4）在最末向量组有部分空 lane，浪费上界 = `(向量宽−1)/N` <= 7/300 = **2.3%**（`N=4` 的测试形态另见 §9.2-R4） | 同 | 两者一致（同一条轴） |
| 累加链形态 | 归约维即向量维（`dim=1`，stride=1）——`reduce_max` 原生形态 ✓ | 归约维变串行步进维：每行 N 次 `(bm,1)` 微元 op（>= N×0.25 µs 发射/行） | **决定性差异** |
| repack 代价 | 无（I/O 原生行主序即 N 连续） | 无 | — |
| UB 容量影响 | 行块 `(bm,N)` 或 tile `(bm,tn)` 驻留（8 B/elem fp16/fp32、12 B/elem bf16 实测） | 同 | — |

**候选矩阵**（含 GPU 源轴选择对照——源码 thread/warp 轴是输入而非结论）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|---------|------------|--------------|------|
| 1 | **行块 × 全 N 驻留 + 水平归约**（`lane→N`，`reduce_max dim=1`；persistent 按 M 行块分核） | N（连续，`x_ub` 行内 stride=1） | 无 | 覆盖 `N <= E_cap` 的全部 manifest/测试负载；row-reduction 迁移基准形态（logsumexp / ada_layer_norm / 姊妹 count_nonzero 三任务实测）+ `VP-2026-0044`（**pending 提案**，非已合入规则，仅作形态参考） | ✅ **主选（路径 R）** |
| 2 | 垂直扫描（`lane→M`，串行步进 N 维） | M | 无 | 归约维串行化 ⇒ 每行 N 次微元发射（`N=4096` ⇒ ms 量级/行）——发射项灾难（`CONST-vector-launch-overhead`） | ❌（发射模型量化） |
| 3 | 核内转置（N-major）后行归约 | M（转置后连续） | `T.transpose` 二轴交换链（UB 级；`docs/Tilelang.language/创建操作/T.transpose.md`） | **纯开销零收益**：主选 #1 的向量化轴 N 即 I/O 原生连续轴，无 repack 需求；核内转置不减少 GM 流量与向量 pass 数，另需 +1 个 `(bm,W)` UB 平面（按 8/12 B/elem 口径 +2~4 B/elem）。**不得按 GPU 直觉把核内重排写成 GM 级重排的稻草人**——此处否决理由为「量化收益为零」而非「代价高」 | ❌ |
| 4 | **列分块 + 运行 max 合并（在线）** | N（分段连续） | 无 | `N > E_cap` 时的**必需**维度（否则单 block 超 UB 容量）：`tn × 8 B <= 64 KB`（fp16/fp32）/ `tn × 12 B <= 96 KB`（bf16）；GM 流量不变（1 遍）；代价 = 每 tile 2 个微元合并 op + tile 循环控制 | ✅ **主选（路径 C）** |
| 5 | 判决式/指示式的 pass 组织：**显式向量链**（`vcmp`+`vcast`）vs 2D `T.Parallel` 融合 `T.if_then_else`（同轴 #1 内的两种实现） | N | 无 | 融合式：`TRAP-parallel2d-bm-ge2-scalarize` 的触发项含**循环体内 `ite` 形态** ⇒ `bm>=2` 整体标量化；显式链：`bm=1/bm>=2` 统一向量化 | ✅ **显式链主选** |
| 6 | 补式落点：`(bm,)` 微缓冲 `1 − max` vs 全宽平面 `1 − ind` | N | 无 | 全宽路线 = +1 个 `(bm,W)` f32 平面 + 1 个全宽 pass（UB 单价 +4 B/elem，fp16 `E_cap` 档降约 1/3）；微缓冲路线 = 1 个 `(bm,)` 级 op（A5 验证精确） | ✅ **微缓冲主选** |
| 7 | 归约算子选择：eq 指示量 + `reduce_max`（本设计）vs nonzero 指示量 + `reduce_min`（C5 vselect） | N | 无 | 后者需 `vselect` + 2 个全宽常数量平面且单价未实测；前者复用与姊妹任务同源、已由本任务探针标定的 `vcmp+vcast+reduce` 链 | ✅ **eq + max 主选** |
| 8 | 跨核 N-split（partial + merge 两段 kernel） | N（跨核切分） | workspace GM 往返 `(M,nchunk)×1B` | 小 M 大 N 场景可提核利用率；本算子 manifest M >= 32（mask-validation-4k M=32 ⇒ 32/40 核）⇒ 收益面窄，代价 = +1 kernel 启动 + workspace 往返 + 合并正确性面 | ❌（R3 量化；登记为 Stage 4 可选，非本设计三件套） |
| 9 | **超预算单缓冲驻留**（`N ∈ (E_cap, 硬上限]` 时 bm=1 不双缓冲，省 1 个 tile pass） | N | 无 | 探针证该区间**可编译**（fp16 24576 元素通过、32768 失败；bf16/fp32 16384 通过、24576 失败）；代价 = 放弃 auto-multi-buffer（共享缓冲竞态面，§4.5） | ⏸ **§1.6.4 实验裁决备选** |

**布局决策结论**：选定 **#1 + #4 + #5 + #6 + #7**——核内布局 = 行块 `(bm, ·)` 行主序驻留（UB staging，
`T.alloc_shared` + 显式 slice），向量化轴 = **N 轴**（`reduce_max dim=1`，判决与指示化走显式
`vcmp → vcast` 链，补式落 `(bm,)` 微缓冲），repack 无，分核 = persistent 按 M 行块（§5.5）；
`N <= E_cap` 走单 tile（#1），`N > E_cap` 走列分块在线 max 合并（#4）。
**弃选理由**：#2 发射爆炸（量化）；#3 纯开销零收益（主选轴即 I/O 原生连续轴）；
#6 的全宽补式路线（+1 个全宽平面与 pass）；#8 收益面窄 + workspace/kernel 启动代价；#9 为实验裁决项
（三件套齐备，§1.6.4）。该结论同步落入 §3.3 伪代码与 §6 循环结构（累加循环的内层向量维 = N）。

#### 1.6.4 实验裁决备选方案：超预算单缓冲驻留（Stage 4 A/B 三件套）

> §1.6.3-#9 的备选完整结构 + 裁决计划（skill Phase 2 实验裁决模式）。**主选** = §1.4 双路径
> （`N <= E_cap` 驻留 / `N > E_cap` 列分块，`E_cap = 8192` 元素 = 竞态安全预算）。**触发形态** =
> 单 block 元素数落在 `(8192, 硬上限]` 的负载（manifest 的 `3d-multidim (4,128,4096) dim=[0,2]` ⇒
> M=128, N=16384 恰在该区间；fp16/fp32 语义下 16384 元素亦在区间内）：主选把它们送进列分块
> （多 1 个 tile pass ×2 微元合并），备选把它们按 `bm=1` 单 tile 驻留处理（探针证 fp16 16384 元素可编译，
> 但放弃 auto-multi-buffer 双缓冲）。§3.3/§4/§6 只承载主选方案，备选结构独立成节、不与主选混排。
>
> **判定裕度的不确定性来源（未实证常数）**：① 超预算时 auto-multi-buffer 被静默放弃后，**单消费者**
> 共享缓冲（本设计每缓冲恰 1 个消费者，见 §4.5 纪律）是否仍会竞态——同族实证的竞态样本是**多消费者**
> 形态（`TRAP-UB-multibuffer-inflation` / argmin 探针），单消费者形态**无实测证据**；
> ② 「省 1 个 tile pass」的真实收益量级取决于发射单价（`CONST-vector-launch-overhead` 0.5 µs/op 上界 vs
> `CONST-msprof-task-floor-910B3` 0.23–0.3 µs/pass 低档），两档给出的收益差约 2 倍。

**① 分派条件（工厂期静态谓词，per-shape 分派可行性）**：
`E_cap < N <= 硬上限(dtype)`（fp16：`8192 < N <= 24576`；bf16：`8192 < N <= 16384`；fp32：`8192 < N <= 16384`，
硬上限取探针实测**通过点**）。工厂已按 `(M, N, op_kind, dtype)` 分派（源 `lru_cache` 语义保留，§0.4-#7），
故主选/备选可按 shape 共存，**不构成全局二选一**。

**② 备选结构（路径 R′：超预算单 tile 驻留；fp16 `N=16384, M=32` 实例）**：

```python
# 与路径 R 逐条同构，仅两处不同：
#   (a) bm 固定 = 1（单 block 元素数 = N = 16384 <= 硬上限(bf16/fp32) = 16384，探针实测可编译）；
#   (b) 不设 num_full/tail（无 tile 循环）——省 1 个 tile 的 2 个微元合并 op；
# 预算（手工代理，§4.5 口径；实测编译器单价 8 B/elem（fp16/fp32）/ 12 B/elem（bf16））：
#   单块 16384 元素 × 8 B = 131072 B = 128 KB <= 192 KB（单缓冲）；双缓冲 256 KB > 192 KB ⇒ 自动双缓冲被放弃
with T.Kernel(T.ceildiv(M, 1), is_npu=True) as (pid, _):
    x_ub    = T.alloc_shared((1, N), dtype)            # 128 KB 档（fp16）
    zero_ub = T.alloc_shared((1, N), dtype)
    mask_eq = T.alloc_shared((1, N), "bool")
    ind     = T.alloc_shared((1, N), "float32")
    mx      = T.alloc_fragment((1, 1), "float32")
    mx_flat = T.alloc_fragment((1,), "float32")
    one_v   = T.alloc_fragment((1,), "float32")
    all_v   = T.alloc_fragment((1,), "float32")
    all_f16 = T.alloc_fragment((1,), "float16")
    out_ub  = T.alloc_shared((1,), "int8")
    start = pid
    if start < M:                                      # 动态 guard（M=32 时恒成立，保留为标准形态）
        T.fill(zero_ub, tir.const(0, dtype))
        T.fill(one_v, tir.const(1, "float32"))
        T.copy(x[start:start + 1, 0:N], x_ub[0:1, 0:N])
        T.vcmp(x_ub, zero_ub, mask_eq, "eq")           # IEEE ordered EQ
        T.vcast(mask_eq, ind, round_mode="rint")       # i1 -> f32
        T.reduce_max(ind, mx, dim=1, clear=True)       # 1.0 iff any zero
        T.reshape(mx, mx_flat)
        T.vsub(one_v, mx_flat, all_v)                  # 补式（(1,) 微缓冲）
        T.vcast(all_v, all_f16, round_mode="rint")     # f32 -> f16
        T.vcast(all_f16, out_ub, round_mode="rint")    # f16 -> i8
        T.copy(out_ub[0:1], out[start:start + 1])
```

- **预算（备选）**：fp16 `N=16384` 单缓冲 128 KB ✓（无余量给双缓冲）；bf16 按 12 B/elem 同档
  （`16384 × 12 = 192 KB` 恰触上限，建议 Stage 3 复标）；fp32 按 8 B/elem ⇒ 128 KB 档。
  `N > 24576`（fp16）一律走主选路径 C（硬上限）。
- **分核三要素（备选）**：逻辑核数 = `M`；物理核数 = **40**（§5.5 实查同源）；规模判定 = 逻辑 32 <= 40
  （`3d-multidim` 的 M=128）⇒ `num_kernels = min(M, 40)`、M=128 时 persistent 40 + 核内 `T.serial(4)` + guard
  （静态上界）。
- **风险**：单块 128 KB 无双缓冲余量 ⇒（i）auto-multi-buffer 被放弃后的单消费者竞态面（无实测证据）、
  （ii）GM 搬运与向量计算**不可重叠**（`CONST-reduce-ub-per-element` 的伴随实测：
  「整平面 `reduce` 与 GM→UB `T.copy` 不重叠」）⇒ 流量项暴露。这是本裁决的核心待测点。

**③ 实验裁决计划**：

- **代表 shape**：`3d-multidim (4,128,4096) dim=[0,2]` bool→f32（manifest bench）+ `mask-validation-32k`
  `(32,32768)` fp32（硬上限之外，应走主选；作为对照）+ 合成档 `(2048,16384)` fp32（放大 M 以放大流量项）。
- **指标**：bench harness `Task Duration`（`launches >= 30` 均值，`examples/TileOPs/benchmarks/ops/bench_all.py`
  同款口径）+ 精度门（`torch.equal` 全 PASS 为前提；主选/备选必须逐位一致——A4 已证分段恒等）。
- **判定阈值**：备选 R′ 端到端延迟 **< 0.8 ×** 主选同 shape 延迟 ⇒ 该 shape 类分派翻转（主选/备选
  按 shape 共存，非全局翻转）；实测同时记录「单消费者超预算是否竞态」（多次重复运行 + 逐位比对）。
- **回写路径**：实测数据与结论回写本节裁决记录 + `RETROSPECTIVE.md`；若翻转，修订 §1.4/§4.5/§5.2
  增补 R′ 分派分支并过 Stage 2 复审；Stage 3 实现期若需提前引入，走 `[DESIGN_LIMIT]` 受控修订路由。
- **常数回填**：本裁决直接标定「发射单价双档」「单消费者超预算竞态」「bf16 归约链单价」三个量
  （`_shared/standards/hardware-cost-model.md` §3 回填对账），并作为 `constants.md` 候选条目
  （`CONST-reduce-ub-per-element` 的 2-D 行归约形态补充 + 910B3 `CONST-vector-launch-overhead` 数据点）。

---
## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer

（NPU 侧 `ascend_mode = "Developer"`，与 `examples/TileOPs/tileops/kernels/reduction/all/all.py` 的
`LogicalReduceKernel.ascend_mode = "Developer"` 一致；conductor 亦指定 `programming_mode = developer`。）

### 2.2 选型理由

- **计算类型为纯 Vector**：全部计算是逐元素比较（`vcmp`）、类型转换（`vcast`）、行向归约（`reduce_max`）
  与 `(bm,)` 微元算术（`vsub`/`vmax`），无 matmul ⇒ 不需要 Cube/L1/L0 的显式管理，Expert 模式的手工
  流水/同步收益不成立。
- **归约由原语直接承担**：`T.reduce_max(dim=1)` 在 Developer 模式下由编译器完成 UB 内的行向归约 lowering
  （姊妹 count_nonzero 任务在**同工具链同设备**上已实证该形态可用），无需手工分片累加。
- **流水交给编译器**：task / tile 循环上的 auto-multi-buffer 由 Developer 模式自动施加（姊妹任务实测其
  正确性与竞态边界），无需 `T.Pipelined` 手工 stage 管理。
- **收益/成本**：Developer 模式代码更短、UB 记账可按探针实测单价线性估算（§4.5），符合「先把算法与
  容量算对，再考虑 Expert 手工调度」的次序；Expert 模式留作 Stage 4 若实测表明 auto-multi-buffer
  不生效时的备选路径（§6.3）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（UB 上的行块 tile 与指示平面、输出载体）+ `T.alloc_fragment`（`(bm,1)`/`(bm,)` 微缓冲） |
| 计算方式 | 显式 v-prefix 向量链（`T.vcmp` → `T.vcast` → `T.reduce_max` → `T.vsub` → `T.vcast`×2），不使用 `T.Parallel` 融合 `ite` |
| 线程 / 核组织 | 一维 `T.Kernel(num_kernels, is_npu=True)` + 核内 `T.serial` 任务循环（persistent），无 `threads`（harness K9 已去除） |
| 同步 | 自动同步（编译器管理 UB/fragment 依赖与 auto-multi-buffer），无手工 flag |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（§1.3 第二式）；每步优先映射 v-prefix 向量 API，与 §1.6.2 的
> 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `z_j = [\,x_j == 0\,]`（原生 dtype 域） | 零命中指示（ordered EQ，IEEE 正确） |
| 2 | `i_j = cast_f32(z_j)`（i1 → f32，取值 0.0/1.0） | 指示化，归约域恒 f32（规避 bf16 不被 `reduce` 支持） |
| 3 | `m = max_{j<W} i_j`（W = 列块宽 tn 或整行 N） | 行向归约：1.0 当且仅当「本块存在零」 |
| 4 | `m = max(m, m_tile)`（路径 C，逐 tile） | 在线合并（max 结合/交换/幂等） |
| 5 | `a = 1 − m` | 补式（**(bm,) 微缓冲**） |
| 6 | `y = cast_i8(cast_f16(a))` | int8 载体出栈（`f32→i8` 无表项，走两步舍入） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0a | 零常数量 | `T.fill(zero_ub, tir.const(0, dtype))` | 类型化常量（字面量 `0` 会被前端拒绝） | Developer |
| 0b | 补式被减数常量 1 | `T.fill(one_v, tir.const(1, "float32"))` | `(bm,)` 微缓冲，任务循环外一次 | Developer |
| 0c | 路径 C 合并累加器初值 | `T.fill(acc, tir.const(0, "float32"))` | max 单位元（指示量非负） | Developer |
| 1 | 数据搬入 | `T.copy(x[start:start+real_m, c0:c0+W], x_ub[0:real_m, 0:W])` | src/dst **双显式 slice**（尾块守卫，R6） | Developer |
| 2 | `[x == 0]` | `T.vcmp(x_ub, zero_ub, mask_eq, "eq")` | `"eq"` = ordered EQ（`"ne"` 为 ordered NE，NaN 语义相反，禁用） | Developer |
| 3 | `i = cast_f32(z)` | `T.vcast(mask_eq, ind, round_mode="rint")` | i1 → f32（表项存在） | Developer |
| 4 | `m = max_j i_j` | `T.reduce_max(ind, mx, dim=1, clear=True)` | 同 dtype（f32→f32），`clear=True` 由编译期初始化 | Developer |
| 5 | `m = max(m, m_tile)`（路径 C） | `T.vmax(acc, mx, new_acc)` + `T.copy(new_acc, acc)` | **no-alias**（规避原位 carried-state 陷阱） | Developer |
| 6 | `(bm,1) → (bm,)` 形状桥 | `T.reshape(mx, mx_flat)` | 同元素数、rank 降一 | Developer |
| 7 | `a = 1 − m` | `T.vsub(one_v, mx_flat, all_v)` | 同 rank 同 shape（`(bm,)`） | Developer |
| 8 | `y = cast_i8(cast_f16(a))` | `T.vcast(all_v, all_f16, round_mode="rint")` → `T.vcast(all_f16, out_ub, round_mode="rint")` | `f32→f16` + `f16→i8`（两表项均存在） | Developer |
| 9 | 写出 | `T.copy(out_ub[0:real_m], out[start:start+real_m])` | 尾块不写出垃圾行 | Developer |
| 10 | 任务边界 | `real_m = T.min(block_m, M - start)` + `if start < M:` | 动态 guard（serial 上界静态） | Developer |

**API 可行性确认（逐条来源；均为**本仓文档 + 家族先例**双证）**：

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.vcmp(A, B, C, cmp_mod)` | `docs/Tilelang.language/比较操作/T.vcmp.md`（fp16/bf16/fp32 ✓；bool 输出须留片上） | 已验证（姊妹任务同形态实测通过） |
| `T.vcast(src, dst, round_mode)` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md`（`i1→f32` ✓、`f32→f16` ✓、`f16→i8` ✓；**`f32→i8` 无表项** ⇒ 两步） | 已验证（本任务探针三种 dtype 全编译通过，含两步 cast） |
| `T.reduce_max(buffer, out, dim, clear)` | `docs/Tilelang.language/规约操作/T.reduce_max.md`、`T.reduce.md`（dtype 表 fp16/fp32 ✓、bf16 ✗、int 族 ✓；`(M,N)→(M,1)` 形状规则） | 已验证（本任务探针编译通过；归约域恒 f32） |
| `T.vsub(src0, src1, dst)` | `docs/Tilelang.language/数学操作/T.vsub.md`（fp32 ✓；同 rank；支持广播） | 已验证（姊妹任务同形态实测通过） |
| `T.vmax(src0, src1, dst)` | `docs/Tilelang.language/比较操作/T.vmax.md`（fp32 ✓） | 待 Stage 3 实证（路径 C 合并链；同 rank 同 shape、no-alias 写法与 `vsub` 同族） |
| `T.reshape(src, dst)` | `docs/Tilelang.language/shape操作/T.reshape.md` | 已验证（姊妹任务同用法实测通过） |
| `T.copy(src_slice, dst_slice)` | `docs/Tilelang.language/内存操作/T.copy.md` | 已验证（姊妹任务同用法实测通过） |
| `T.fill(buf, tir.const(v, dtype))` | 姊妹任务探针实测（Python 字面量被拒 `value must be a tir.PrimExpr`） | 已验证 |
| `T.Kernel(n, is_npu=True)` / `T.serial(n)` | `docs/Tilelang.language/编译器提示操作/T.Kernel.md`；serial 上界须静态 | 已验证（姊妹任务同结构实测通过） |

> 全部 API 均**先枚举名字再在 `docs/Tilelang.language/` 全部子目录检索**（含 `创建操作/`、`索引与元素操作/`、
> `条件操作/`、`排序操作/`、`逻辑操作/`、`原子操作/` 等 AGENTS.md 路由未映射目录）；未使用任何本仓无法佐证的 API。

### 3.3 计算伪代码

> 迁移规则：`@tilelang.jit` 函数名保持 `_logical_reduce_kernel`；`@T.prim_func` 参数名与顺序保持
> `x`、`out`；`T.Tensor` 的 dtype 以**第二个位置参数**传入（不用 `dtype=` 关键字）。

```python
# 工厂期（Python，全部静态量）：N、dtype、E_CAP、路径、bm、tn、num_full、tail、num_kernels、serial 上界
# _tile_body(x, out, c0, W) —— 单个列块的判决 + 指示化 + 归约（逐元素计算全部向量化）
#   T.copy(x[start:start+real_m, c0:c0+W], x_ub[0:real_m, 0:W])   # 双显式 slice
#   T.vcmp(x_ub, zero_ub, mask_eq, "eq")                          # IEEE ordered EQ
#   T.vcast(mask_eq, ind, round_mode="rint")                      # i1 -> f32（0.0/1.0，命中零为 1.0）
#   T.reduce_max(ind, mx, dim=1, clear=True)                      # mx: 1.0 iff 本块存在零
# _epilogue(mx) —— 补式 + int8 出栈
#   T.reshape(mx, mx_flat)
#   T.vsub(one_v, mx_flat, all_v)                                 # (bm,) 微缓冲：all = 1 - any_zero
#   T.vcast(all_v, all_f16, round_mode="rint")                    # f32 -> f16
#   T.vcast(all_f16, out_ub, round_mode="rint")                   # f16 -> i8
#   T.copy(out_ub[0:real_m], out[start:start+real_m])

with T.Kernel(num_kernels, is_npu=True) as (cid, _):
    # 任务不变量（循环外）：zero_ub（T.fill）、one_v（T.fill）
    for s in T.serial(num_local_tasks):                # 静态上界
        start = (cid + s * num_kernels) * block_m
        if start < M:                                  # 动态 guard
            real_m = T.min(block_m, M - start)
            if path == "R":                            # 单 tile 驻留
                [§3.3 _tile_body(c0=0, W=N)] -> mx
                [§3.3 _epilogue(mx)]
            else:                                      # 列分块 + 运行 max 合并（在线）
                T.fill(acc, tir.const(0, "float32"))   # max 单位元
                for t in T.serial(num_full):           # 静态上界
                    [§3.3 _tile_body(c0=t*tile_n, W=tile_n)] -> mx
                    T.vmax(acc, mx, new_acc)           # no-alias 合并
                    T.copy(new_acc, acc)
                if tail > 0:                           # 静态尾块（工厂期条件生成独立代码块）
                    [§3.3 _tile_body(c0=num_full*tile_n, W=tail)] -> mx
                    T.vmax(acc, mx, new_acc)
                    T.copy(new_acc, acc)
                [§3.3 _epilogue(acc)]
```

> **结构性条件一律提到工厂层**（不迁就姊妹任务的教训）：路径分支、`tail` 的有无、`num_full` 的取值
> 全部是 Python 期的静态量 ⇒ 每个变体返回**独立的 prim_func**；`@T.prim_func` 内**不出现**解析期 `if`
> 定义 buffer 的写法（分支内绑定的名字不逃逸到 sibling 分支，条件为真时报 `ParserError: Undefined variable`）。

### 3.4 API 可行性确认

见 §3.2 末表（含每条 API 的文档路径与验证状态）。**无「需扩展」项**；唯一 Stage 3 待实证项为
`T.vmax`（路径 C 合并链），其失败退路明确：改用 `T.reduce_max` 的双缓冲写法或 `T.vsub` 取负后
`T.vmin`（不改变语义，仅换合并算子），并在 §8.2 L0 复验。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **否** | 一维 `T.Kernel(num_kernels, is_npu=True)`；M 行块与 N 列块均为核内结构，不引入第二/三维 grid |
| bool（i1）不可 load/store GM | **是** | bool 只作片上中间量（`mask_eq`）；输出载体用 int8，wrapper `out_raw.bool()` 还原 bool（源同款） |
| `reduce` 不支持 bf16 | **是（已规避）** | 归约域恒为 f32 指示平面（`ind`），bf16 只出现在 `x_ub`/`zero_ub`（`vcmp` 上抛由编译器处理） |
| `T.vcast` 无 `f32 → i8` 表项 | **是** | 走 `f32 → f16 → i8`（两表项均存在）；A5 机器验证位精确 |
| 循环边界须静态 | **是** | `num_full` / `num_local_tasks` / `tile_n` / `tail` 全为工厂期 Python 整数；动态性仅在 guard |
| UB 容量 192 KB（实查 196608 B） | **是** | §4.5 逐配置核算 + 探针实测单价 + 硬上限；`E_cap = 8192` 元素 |
| L0C / L1 容量 | **否** | 纯 Vector，不涉及 L0/L1 |
| GEMM 非整除 / L0C 溢出 | **否** | 无 Cube 计算 |
| 尾轴 32B 对齐 | **否（探针已证）** | raw N=2/4/64/300 均编译通过；`T.copy` 切片满足 32B 尾轴要求 |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键差异汇总）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 维度 / 并行两级 | 一维 grid × CUDA `threads` | 一维 `T.Kernel` persistent + 核内 `T.serial` | `threads` 去除（K9）；元素级并行交 Vector 向量化（§0.6-R2） |
| 共享内存层级 | `T.alloc_shared`（SMEM） | `T.alloc_shared`（UB，实查 192 KB） | 等价替换；容量与对齐按探针实测重算（§0.6-R5） |
| 归约算子 | `reduce_min` over `[x!=0]` | `reduce_max` over `[x==0]` + 微缓冲补式 | 语义等价替换（NaN 方向相反，§0.6-R1） |
| 判决式 | `x != 0.0`（unordered NE，fp32 域） | `vcmp("eq")`（ordered EQ，原生域） | ordered NE 在 NPU 语义相反 ⇒ 必须走补式（A1c 实证） |
| 对齐 pad | 内部 `align_up(N,256)` + 掩码（pad=1.0） | 完全不 pad（实列域） | A2/A2b 证明 pad=1.0 中性；探针证 raw 宽度可编译 |
| 列分块回退 | `_needs_tiling`（fragment 列上限 / SMEM 预算）触发 tiled 工厂 | `N > E_cap`（UB 容量）触发列分块 + 运行 **max** 合并 | 不迁移源 tiled 工厂；判据/合并算子/尾块策略按 NPU 重推（§0.6-R4） |
| 尾块守卫 | 缺失（越界） | `real_m = T.min(...)` + 双显式 slice | 修正源缺陷（§0.6-R6） |
| 出栈 | `T.cast(result > 0.5, "int8")`（`T.Parallel(bm)`） | `f32→f16→i8` 两条向量 cast | 向量化（A5 位精确） |
| dtype 支持 | fp32 上抛后统一计算 | 原生域比较 + f32 指示平面 | A3 穷举等价（省 1 个全宽 pass） |
| out_idx dtype | `int8`（bool 载体） | `int8`（同） | 保持（wrapper 转 bool） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/count_nonzero/_logical_reduce_kernel/DESIGN.md` | **极高**（同 GPU 源工厂、同工具链、同 2-D 行归约） | 迁移分析骨架、UB 单价探针方法、`E_cap` 记账、分核三要素、D-1 脚本形态 |
| `examples/count_nonzero/_logical_reduce_kernel/_logical_reduce_kernel.py` | **极高** | 可运行参考实现：`vcmp`→`vcast`→`reduce`→微缓冲 链的精确写法与 `T.copy` 双 slice 形态 |
| `examples/TileOPs/tileops/kernels/reduction/all/all.py` | 高（本迁移单元的 wrapper） | 工厂签名 `(M, N, op_kind, dtype)`、`default_config` 需替换（§5.6） |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single` | 高 | 单 tile 驻留 + `T.min` 尾块 + 双 slice（raw N=300 先例） |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_tiled` | 中高 | 列循环 + 运行状态更新链（本设计路径 C 的结构祖先） |
| `examples/argmin/_argreduce_kernel/DESIGN.md` | 中高 | 同族同工具链迁移设计结构 + 探针证据链（核数/UB 实查口径） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(M, N)` | float16 / bfloat16 / float32 | raw N（Op 层预转：bool/int32/int64/complex → float32） |
| | | | M = 非规约维乘积；N = 规约维乘积 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `(M,)` | **int8**（bool 载体） | 每行逻辑与结果（1/0）；wrapper `out_raw.bool()` 转为 manifest 契约的 bool |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | `(bm, W)`，W ∈ {N, tn, tail} | 输入 dtype | UB | 行块/列块输入 tile |
| `zero_ub` | `(bm, W)` | 输入 dtype | UB | 零常数量（比较右操作数；任务外一次 `T.fill`） |
| `mask_eq` | `(bm, W)` | bool | UB（片上，不可落 GM） | 零命中掩码 |
| `ind` | `(bm, W)` | float32 | UB | 0/1 指示平面（归约域，恒 f32） |
| `mx` | `(bm, 1)` | float32 | fragment | 本块「是否存在零」（1.0/0.0） |
| `acc` / `new_acc` | `(bm, 1)` | float32 | fragment | 路径 C 运行 max 合并（no-alias 双缓冲） |
| `mx_flat` / `one_v` / `all_v` | `(bm,)` | float32 | fragment | 形状桥 / 补式被减数 / 补式结果 |
| `all_f16` | `(bm,)` | float16 | fragment | 出栈中间态 |
| `out_ub` | `(bm,)` | int8 | UB | 输出行块载体 |

### 4.4 内存搬运路径

```
GM[x] ──T.copy（双显式 slice，含尾块 real_m）──▶ UB[x_ub]
                                                   │
                          UB[zero_ub]（T.fill 一次）─┤
                                                   ▼
                                     T.vcmp → UB[mask_eq (bool, 片上)]
                                                   │
                                     T.vcast(i1→f32) → UB[ind]
                                                   │
                                     T.reduce_max(dim=1) → frag[mx (bm,1)]
                                                   │  （路径 C：T.vmax + T.copy 循环合并）
                                     T.reshape → frag[mx_flat (bm,)]
                                                   │
                          frag[one_v]（T.fill 一次）┤
                                                   ▼
                                     T.vsub → frag[all_v (bm,)]
                                                   │
                          T.vcast(f32→f16) → T.vcast(f16→i8) → UB[out_ub]
                                                   │
GM[out] ◀──T.copy（双显式 slice）──────────────────┘
```

**无 Cube 段、无 L1/L0 参与**；GM↔UB 单向各一次（读 1 遍、写 1 遍）。

### 4.5 UB 内存预算（逐 buffer 手工核算 + 探针实测量价 + 硬上限三层口径）

**口径说明**：手工代理预算按**声明 buffer 清单**求和（保守）；**实测量价** = 探针从编译器
`ub overflow, requires N bits` 反解的**链总需求**（含 lowering 物化的中间张量）——后者才是容量判据。
本工具链实测（`history_version/design_probe_ub.py`，2026-10-16，910B3 / CANN 9.0.0）：

| dtype | 实测单价（B/元素） | 单 block 实测通过元素数 | 实测溢出元素数 | 溢出报错 |
|-------|------------------|------------------------|---------------|---------|
| float16 | **8.00** | **24576** | 32768（`requires 2097408 bits`；同档 32768 另测 6.00 B/elem 报 `1573632 bits`） | `ub overflow, requires N bits while 1572864 bits available` |
| bfloat16 | **12.00**（比较上抛 f32，IR 可见 `vcast bf16→f32` 作用于 `vcmp` 两操作数） | **16384** | 24576（`requires 2359296 bits`）、32768（`3145728 bits`） | 同上 |
| float32 | **8.00** | **16384** | 24576（`requires 1573632 bits`，超 96 B）、32768（`2097920 bits`） | 同上 |

**竞态安全预算 `E_cap` 的推导**：把「探针实测单价」视为编译器侧**总需求**（已含其自身施加的缓冲/物化），
再额外留出一倍余量给 auto-multi-buffer 的第二缓冲与竞态面，即 `2 × 单价 × 元素数 <= 196608 B`
⇒ `E_cap = 196608 // 2 // 12（最坏 dtype bf16）= 8192` 元素（等价表述：**最坏 dtype 的实测量价 <= UB/2 = 96 KB**）。
**取 bf16 最坏档作为全 dtype 的统一档**（保守、实现简单）；fp16/fp32 的理论放宽档
（`196608 // 2 // 8 = 12288`）登记为 Stage 4 调优候选（§6.3）。
另有**硬上限档**（物理判据，不预留余量）：`196608 // 单价`，由实测通过点确认
（fp16 24576 / bf16 16384 / fp32 16384 元素）。

**逐配置核算**（统一口径：**手工代理（单缓冲）** = 声明 buffer 清单求和
`(2 × elem_bytes + 1 + 4) B/元素 × 元素数`（fp16/bf16 = 9、fp32 = 13 B/元素）；
**实测量价** = 探针反解单价 × 元素数（编译器侧，含其 lowering 物化的中间张量）；
判据 = ① `元素数 <= E_cap`（= 8192，等价于「最坏 dtype 的实测量价 <= UB/2 = 96 KB」）
且 ② `实测量价 <= 硬上限档`）：

| 配置（workload / 测试） | dtype | 路径 | 每 block 元素数 | 手工代理（单缓冲，声明清单） | 实测量价（探针单价 × 元素数） | ① 元素数 <= 8192 | 结论 |
|---|---|---|---|---|---|---|---|
| smoke `(32,256)` | fp16 | R，bm=8 | 2048 | 9 × 2048 = 18.4 KB | 8 × 2048 = **16 KB** | ✓ | ✓ |
| smoke `(32,256)` | fp32 | R，bm=8 | 2048 | 13 × 2048 = 26.6 KB | 8 × 2048 = **16 KB** | ✓ | ✓ |
| mask-validation-4k `(32,4096)` | fp32 | R，bm=2 | 8192 | 13 × 8192 = 106.5 KB | 8 × 8192 = **64 KB** | ✓（= 上限档） | ✓ |
| `(256,4096)` 测试 | fp16 | R，bm=2 | 8192 | 9 × 8192 = 73.7 KB | 8 × 8192 = **64 KB** | ✓ | ✓ |
| `(256,4096)` 测试 | bf16 | R，bm=2 | 8192 | 9 × 8192 = 73.7 KB | 12 × 8192 = **96 KB** | ✓（最坏档，恰 = UB/2） | ✓（余量最小档，Stage 3 复标） |
| `(128,300)` 测试 | fp16 | R，bm=8 | 2400 | 9 × 2400 = 21.6 KB | 8 × 2400 = **19.2 KB** | ✓ | ✓ |
| `(128,300)` 测试 | fp32 | R，bm=8 | 2400 | 13 × 2400 = 31.2 KB | 8 × 2400 = **19.2 KB** | ✓ | ✓ |
| `(64,512)` dim=0 ⇒ M=512, N=64 | fp16 | R，bm=8 | 512 | 9 × 512 = 4.6 KB | 8 × 512 = **4 KB** | ✓ | ✓ |
| `(4,64,512)` dim=1 ⇒ M=2048, N=64 | fp16 | R，bm=128（扩展阶梯） | 8192 | 9 × 8192 = 73.7 KB | 8 × 8192 = **64 KB** | ✓ | ✓ |
| `(4,64,512)` dim=0 ⇒ M=32768, N=4 | fp16 | R，bm=1024（扩展阶梯） | 4096 | 9 × 4096 = 36.9 KB | 8 × 4096 = **32 KB** | ✓ | ✓ |
| mask-validation-32k `(32,32768)` | fp32 | C，bm=1，tn=8192 | 8192（tile） | 13 × 8192 = 106.5 KB | 8 × 8192 = **64 KB** | ✓ | ✓ |
| 3d-multidim（M=128, N=16384） | fp32 | C，bm=1，tn=8192 | 8192（tile） | 13 × 8192 = 106.5 KB | 8 × 8192 = **64 KB** | ✓ | ✓ |
| 退化路径：`dim=None` flatten ⇒ M=1, N=numel（如 1M） | fp16 | C，bm=1，tn=8192，num_full=ceil(numel/8192) | 8192（tile） | 9 × 8192 = 73.7 KB | 8 × 8192 = **64 KB** | ✓ | ✓（`num_full` 静态值随 shape 变，但每 tile 驻留恒定） |
| 退化路径：`dim=None` flatten | fp32 | 同上 | 8192（tile） | 13 × 8192 = 106.5 KB | 8 × 8192 = **64 KB** | ✓ | ✓ |
| 退化路径：`N=1`（`(M,1)` 测试） | fp16 | R，bm = min(8, M) | 8 | 9 × 8 = 72 B | 8 × 8 = **64 B** | ✓ | ✓ |
| 反例登记：单块 32768 元素（fp16） | fp16 | — | 32768 | — | **262176 B > 196608 B（硬上限档）** | ✗ | ✗ ⇒ 判据 ① 强制该 shape 走路径 C |

> **口径纪律（`_shared/standards/hardware-cost-model.md` §5）**：
> ① **退化/几何特化路径已单列**（最后两行）——`dim=None` 全量 flatten（M=1、N=numel）走路径 C，
> 每 tile 驻留恒定 64 KB（fp16/fp32）⇒ **`E_cap` 的容量结论对退化路径同样有效**（与池化类「整平面缓冲随
> H·W 线性放大」的退化路径不同，本算子路径 C 的 `tn` 被 `E_cap` 封顶，不随 N 放大）；`N=1` 行块极小。
> ② **UB 膨胀系数双口径并列**：默认 auto-multi-buffer 开时 `CONST-capacity-910B2C` 记录 ~1.7x 膨胀
> （保守上限）；`--enable-auto-multi-buffer=false` 下实测 1.10–1.12（另有显式 alloc 结构下 ≈0.98 的反例）。
> **本设计未显式关闭 auto-multi-buffer**，故容量判据按**额外一倍余量（2x）**核算——判据 ①
> 的 `E_cap = UB/2 ÷ 最坏单价` 即该 2x 口径；若按 1.7x 解释，各档余量更大。**BishengIR 实际需求以探针反解为准**
> （探针从编译器报错文本取 `required bytes`，已含其内部膨胀），两条口径互相印证。
> ③ **手工代理 vs 实测口径的差异已显式给出**：fp16 档手工代理（约 36 KB/8192 元素 ≈ 4.5 B/elem）
> 远小于实测量价（8 B/elem）——正是姊妹任务与本任务探针共同证明的「归约链 lowering 物化额外中间张量」；
> 容量判据一律以**实测量价**为准。

### 4.6 动态轴定义

无。kernel 声明 `(M, N)` 为编译期常量（源工厂 `lru_cache` 按 `(M, N, op_kind, dtype)` 缓存编译产物，
NPU 侧保持同一语义）；`M` / `N` 运行时变化时由 Op 层触发新编译。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")
```

- `out_idx=[1]`：输出 `out` 由框架按声明 dtype（int8）分配并返回（与源一致）；
- `target="npuir"`：Ascend 后端；
- 工厂外层保留 `functools.lru_cache(maxsize=32)`（源语义保留，§0.4-#7）；
- 无特殊 `pass_configs`（沿用 Developer 模式默认，auto-multi-buffer 开启）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector。

**判定依据**：全部计算为逐元素比较/类型转换/行归约/微元算术（`vcmp`、`vcast`、`reduce_max`、`vsub`、
`vmax`），无 matmul；数据通路 GM↔UB↔fragment；无 Cube/L0/L1 参与。

### 5.2 Block 划分（元素预算阶梯 + 路径切换 + `tn`）

```python
# 工厂期（全部静态 Python 量）；口径对照 §4.5
UB_BYTES      = 196608                     # 实查 AscendArch().ub_cap
UNIT_PRICE    = {"float16": 8, "float32": 8, "bfloat16": 12}   # 探针实测 B/元素
E_CAP         = UB_BYTES // 2 // max(UNIT_PRICE.values())      # = 8192 元素（竞态安全档，最坏 dtype bf16）
HARD_CAP      = {"float16": 24576, "bfloat16": 16384, "float32": 16384}  # 探针实测通过点
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
    tile_n = max_divisor_pref(N, cap, align=256)   # 不超 cap 的最大 256 倍数且整除 N；无整除因子 → 最大 256 倍数
    num_full, tail = N // tile_n, N - (N // tile_n) * tile_n
block_m = min(block_m, M)                          # 行数不足不超配（垃圾行污染）
num_logical = ceildiv(M, block_m)
num_kernels = min(num_logical, 40)                 # 物理核数 = 实查 aivector = 40（§5.5）
num_local_tasks = ceildiv(num_logical, num_kernels)  # 核内串行上界（静态值）
```

**`block_m` 选择理由**：阶梯内取预算最大值——行块越大，每 block 的固定开销（载入 + 出栈约 5 op）
摊得越薄（`CONST-vector-launch-overhead` 恒为每 op 固定成本）；受 `E_cap` 硬约束。
基础阶梯 `[1,2,4,8]`（源阶梯 + 家族先例）覆盖常规行宽；**小 `N` 大 `M` 负载**（如 `(32768,4)`、`(2048,64)`）
由发射项门控（`num_logical > 80` ⇒ 双波以上）启用扩展阶梯至 1024（档位由同一预算过滤器自动封顶）。

**逐 workload 参数表**（M/N 由 manifest + Op 层 `_prepare_input` 的 movedim+reshape 推导并附算式；
manifest workload 的 `dim` 缺省：bench 侧 `extra.setdefault("dim", -1)` 归约最后一维）：

| workload（来源） | dtype（kernel 侧） | 路径 | block_m | tile_n | 元素数/block | num_logical | num_kernels | 核内任务（静态） | 依据/推导 |
|------------------|-------|------|---------|--------|--------------|-------------|-------------|------------------|-----------|
| smoke `[32,256] dim=-1` | fp16/fp32（manifest 列 bool/fp16/fp32） | R | **8** | 256 | 2048 | 4 | 4 | 1 | `p·256 <= 8192 → p <= 32`；阶梯内最大 8；4 <= 80 不触发扩展 |
| mask-validation-4k `[32,4096]`（dim 缺省 ⇒ -1） | bool→f32 | R | **2** | 4096 | 8192 | 16 | 16 | 1 | `p·4096 <= 8192 → p <= 2`；16 <= 80 不触发扩展 |
| mask-validation-32k `[32,32768]`（dim 缺省 ⇒ -1） | bool→f32 | **C** | 1 | **8192** | 8192（tile） | 32 | 32 | 1 | `32768 > 8192` → 路径 C；`tn` = min(8192,32768) 内最大 256 倍数且整除 32768 = 8192 ⇒ `num_full=4, tail=0` |
| 3d-multidim `[4,128,4096] dim=[0,2]` | bool→f32 | **C** | 1 | **8192** | 8192（tile） | 128 | **40** | **4** | 归约维 {0,2} ⇒ N = 4×4096 = 16384，M = 128（Op 层 movedim 后 reshape）；`tn`=8192 ⇒ `num_full=2, tail=0` |
| 测试 `(128,512)`（basic/noncontig） | fp16/bf16/fp32 | R | 8 | 512 | 4096 | 16 | 16 | 1 | `p·512 <= 8192 → p <= 16`；阶梯内 8 |
| 测试 `(256,4096)` | fp16/bf16 | R | 2 | 4096 | 8192 | 128 | 40 | 4 | 同 mask-validation-4k 的 `p` 上界；128 > 80 但扩展档受预算封顶 → 2 |
| 测试 `(128,300)` | fp16/fp32 | R | 8 | 300 | 2400 | 16 | 16 | 1 | raw N=300（**不 pad**，§0.6-R3） |
| 测试 `(129,512)`（M 尾块） | fp16 | R | 8 | 512 | 4096 | **17** | 17 | 1 | `129 % 8 = 1` ⇒ 尾块 `real_m=1` |
| 测试 `(64,512) dim=0` ⇒ M=512, N=64 | fp16/fp32 | R | 8 | 64 | 512 | 64 | 40 | 2 | `p·64 <= 8192 → p <= 128`；64 <= 80 不触发扩展 |
| 测试 `(4,64,512) dim=1 / dim=-2` ⇒ M=2048, N=64 | fp16/bf16 | R | **128** | 64 | 8192 | 16 | 16 | 1 | 基础 bm=8 → 256 > 80 ⇒ 扩展阶梯封顶 128 |
| 测试 `(4,64,512) dim=0` ⇒ M=32768, N=4 | fp16 | R | **1024** | 4 | 4096 | 32 | 32 | 1 | 基础 bm=8 → 4096 > 80 ⇒ 扩展封顶 1024（`p·4 <= 8192`） |
| 测试 `(512,)`（1D） | fp16/bf16/fp32 | R | 1 | 512 | 512 | 1 | 1 | 1 | `M=1`（Op 层 flatten）⇒ `bm = min(8, 1) = 1` |
| keepdim `(64,512) dim=-1` | fp16 | R | 8 | 512 | 4096 | 16 | 16 | 1 | 与被归约维无关的 keepdim 只影响 Op 层 shape 还原 |
| `dim=None` flatten（如 `(32,32768)` 全量） | fp32 | **C** | 1 | 8192 | 8192（tile） | 1 | 1 | 1 | `M=1, N=numel`；`num_full = ceil(numel/8192)` 静态 |

**`tn` 推导链（路径 C，冻结值复现）**：`cap = min(8192, N)`；候选集 = `[t : t <= cap, t ≡ 0 (mod 256), t | N]`；
取最大者；空集则取不超 cap 的最大 256 倍数（此时 `tail = N % tn != 0`，走静态尾块）。
`N=16384` → `8192 | 16384` ⇒ **tn=8192、num_full=2、tail=0**；`N=32768` → **tn=8192、num_full=4、tail=0**。

### 5.3 约束分析

- **对齐约束**：`T.copy` 的 slice 形态天然满足 32 B 尾轴要求；**无需任何宽度对齐 pad**（探针实测
  raw N = 2 / 4 / 64 / 300 均可编译；logsumexp N=300 已迁移先例同向）。
- **UB 容量**：§4.5 逐配置验证（代理预算 / 实测量价 / 硬上限三层口径 + 退化路径补行）。
- **L0/L1 容量**：不适用（纯 Vector）。
- **编译器陷阱约束**：判决走显式向量链（规避 2D `ite` 融合标量化）；合并走 no-alias
  （规避原位 select/累加丢失 loop-carried 状态）；`vcmp("eq")` 而非 `vcmp("ne")`（`eq` 为 IEEE 正确，
  `ne` 为 ordered NE 与 all 语义相反）；`T.fill` 值须类型化常量；`(bm,1)→(bm,)` 收缩经 `T.reshape`；
  `f32→i8` 无直接 cast ⇒ 两步。
- **`reduce` dtype**：全程 f32（同 dtype），bf16 不进入归约域。
- **路径切换判据的静态性**：`N`、`E_CAP`、`tile_n`、`num_full`、`tail` 均为工厂期常量 ⇒ 每个变体
  生成独立 prim_func（§3.3 注）。

### 5.4 注意事项（非整除/边界处理策略）

- **M 非整除（`M % block_m != 0`）**：`real_m = T.min(block_m, M − start)` + src/dst 双显式 slice +
  垃圾行不写出（§0.6-R6 三件套；VP-2026-0044（pending）/ logsumexp 先例）。
- **N 非整除（256 对齐）**：**不处理**——raw N 实列域计算（§0.6-R3；A2/A2b 机器验证 pad 域 ≡ 实列域）。
- **`tn` 非整除（路径 C 尾块）**：`tail = N − num_full·tn` 为工厂期静态常量，走**独立静态宽度的代码块**
  （独立缓冲组，非 kernel 侧掩码）；**不用 `T.if_then_else` 掩码**（logsumexp tiled 的 serial 变量条件
  有 segfault 反例，见 argmin 设计的 §9.1）。
- **`N=1`**：单列判决 ⇒ `y = (x != 0)`；`(1,1)` 用例由 verify_equiv 与 §8.2 L1 覆盖。
- **`M=1` / `dim=None` flatten**：`bm = min(bm, 1) = 1`；大 `numel` 走路径 C（§4.5 退化路径补行）。
- **`N=0`（空规约维）**：torch 语义为空 AND 恒真（每行 True），源工厂未覆盖（`N_padded=0`）；本设计在
  工厂期 `assert N >= 1` 并在 Op 层短路（§9.2-R5）。
- **`M=0`**：grid=0 边界（§9.2-R6 风险记录；harness 测试不覆盖，Op 层短路建议）。
- **`N > 硬上限`**：一律路径 C（`tn <= E_cap`），无形状长度上限（`num_full` 可为任意静态整数）。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

- **① 逻辑核数**：`num_logical_kernels = ceil(M / block_m)`（纯行并行，无 N 向跨核分核；
  路径 C 的列分块在**核内**（`T.serial` tile 循环），不增加逻辑核数）。
- **② 物理核数（实查，禁止假设值）**：
  `from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()` → **20**（2026-10-16 实查）；
  **纯 Vector 算子核数**由 `NPUUtils.get().get_aivector_core_num()` 实查 = **40**（= 20 × 2，
  与 `AscendArch().aivector_core_num` 互证）；`AscendArch().ub_cap` = **196608 B（192 KB）** 同源实查。
  查询代码与返回值即本节的实查记录（`chip_name` = `Ascend910B`；同族 argmin / 姊妹 count_nonzero 任务
  在同设备上得同值，跨任务一致）。
- **③ 规模判定与分核方案**（统一 persistent 结构：`num_kernels = min(num_logical, 40)`；
  核内 `T.serial(静态上界 = ceil(num_logical / num_kernels))` + grid-stride 任务映射
  `task = cid + s·num_kernels` + 动态 guard `if start < M`）：

| workload | `num_logical` | 判定 | `num_kernels` | 核内任务（静态上界） | 依据 |
|----------|---------------|------|---------------|----------------------|------|
| smoke (32,256) | 4（bm=8） | **逻辑核数 <= 物理核数（4 <= 40）** | 4 | 1 | 无需适配：行块独立、单波、每核 <= 1 任务 |
| mask-validation-4k (32,4096) | 16（bm=2） | **逻辑 <= 物理（16 <= 40）** | 16 | 1 | 无需适配；核利用率 16/40 = 40%，但 M=32 行块已用满并行维（N 向跨核已在 R3 中否决） |
| mask-validation-32k (32,32768) | 32（bm=1，路径 C） | **逻辑 <= 物理（32 <= 40）** | 32 | 1 | 无需适配；核利用率 32/40 = 80%（R3 已论证 N-split 跨核无收益） |
| 3d-multidim (M=128, N=16384) | 128（bm=1，路径 C） | **极大规模（128 > 40）** | **40** | **4** | 固定 40 核 + 核内 `T.serial`（静态边界）；摊薄核启动开销 |
| 测试 (256,4096) | 128（bm=2） | **极大规模（128 > 40）** | **40** | **4** | 同 3d-multidim |
| 测试 (129,512) | 17（bm=8） | 逻辑 <= 物理 | 17 | 1 | 尾块行 1 行 |
| 测试 (512,64) | 64（bm=8） | **中等规模（64 > 40，可对齐）** | **40** | **2** | 物理核数的 1.6 倍 ⇒ 按 40 核 persistent（`ceildiv(64,40)=2`）；亦可把 bm 提到 16（`32 <= 40`）对齐为单波，登记 Stage 4 调优 |
| 测试 (2048,64) | 16（bm=**128**，扩展阶梯） | 逻辑 <= 物理 | 16 | 1 | 扩展阶梯把核数从 256 降到 16 |
| 测试 (32768,4) | 32（bm=**1024**，扩展阶梯） | 逻辑 <= 物理 | 32 | 1 | 扩展阶梯把核数从 4096 降到 32 |
| smoke / 测试 (128,300) | 16（bm=8） | 逻辑 <= 物理 | 16 | 1 | raw N=300 |

> 核内串行循环边界**全部为静态值**（工厂期 Python 整数：`ceildiv(num_logical, num_kernels)`）；
> 动态性仅存在于 guard（`if start < M`，`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 先例形态）。
> wrapper `default_config` 须按 §5.2 公式适配（其 GPU 期 48 KB SMEM 启发式弃用，见 §5.6）。

**核数复算（供检视复算，逐负载；`num_logical = ceil(M / block_m)`）**：

- smoke：`M=32, bm=8` → **4**（<= 40 ⇒ `num_kernels=4`，单波）
- mask-validation-4k：`M=32, bm=2` → **16**（<= 40 ⇒ 16 核）
- mask-validation-32k：`M=32, bm=1` → **32**（<= 40 ⇒ 32 核，每核 1 任务 × 4 tile）
- 3d-multidim：`M=128, bm=1` → **128**（> 40 ⇒ 40 核，核内上界 = **4**）
- `(256,4096)`：`M=256, bm=2` → **128**（> 40 ⇒ 40 核，核内上界 = **4**）
- `(4,64,512) dim=0`：`M=32768, bm=1024` → **32**（扩展阶梯；否则 bm=8 → 4096 核）
- `(4,64,512) dim=1`：`M=2048, bm=128` → **16**（扩展阶梯；否则 bm=8 → 256 核）

### 5.6 集成前置核对（实读目标工件）

| 核对项 | 结论 | 实现期动作 |
|--------|------|-----------|
| 目标 Op 层 `_kernel_handles_padding` | **`True`**（`examples/TileOPs/tileops/ops/reduction/all.py` 类属性） | **保持**；kernel 必须声明 raw `(M, N)`（宽度错位症状：行 0 正确、行 >= 1 全错且不报错） |
| 目标 Op 层 `_pad_value` | `1.0`（all 的 AND 中性） | 保持（本设计不使用该值，kernel 域内不 pad） |
| 目标 Op 层 `_empty_dim_policy` | `"noop"`（`dim=[]` / `dim=()` → `x.bool()`，不进 kernel） | 保持（kernel 无需处理空 dim 列表） |
| Op 层 dtype 预转 | `_pre_kernel` 对 bool/int32/int64/complex 调 `to_logical_float32` | 保持；kernel dtype 路径矩阵按 {fp16, bf16, fp32} 三路（§3.2） |
| 输出 dtype 契约 | manifest `bool`；kernel 载体 `int8` + wrapper `out_raw.bool()` | 保持 int8 载体声明（`"int8" if op_kind != "count_nonzero" else "int64"` 三元式原样保留） |
| wrapper `default_config` | 现为 GPU 期启发式（`smem_budget // (N_padded × elem)`）→ NPU 侧语义不适用 | **改为转调本设计 §5.2 的 `_select_config(M, N, dtype)`**（kernel 模块为 tiling 单一事实源） |
| wrapper 空转/约束提示 | docstring 称「kernel handles 256-element alignment padding internally」 | 更新为「实列域计算、无 pad；`N > E_cap` 自动走列分块路径」（§5.2/§5.4） |
| factory 签名 | `_logical_reduce_kernel(M, N, op_kind, dtype)` → `_func(block_m)`（K9 已去 `threads`） | 保持（迁移公共规则：`@tilelang.jit` 函数名与 `@T.prim_func` 参数名/顺序不变） |
| msprof 锚点 | 现无 `msprof_kernel_name` | 增加（路径 R → `all_resident`，路径 C → `all_tiled`），避免 msprof 捕获 Op 层 prep 算子 |

---
## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block 级 / tile 级调度循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| M 方向（block 级） | persistent 任务循环 | `T.serial(num_local_tasks)`（静态上界 + grid-stride + guard） | 极大规模核内串行（§5.5）；摊薄核启动开销 |
| N 方向（路径 C） | tile 迭代 | `T.serial(num_full)`（静态上界） | 在线 max 合并的循环携带状态（§1.6.2-#10）；静态边界要求 |
| 尾块（路径 C） | 无循环（独立静态宽度代码块） | — | `tail` 为工厂期常量；避免 serial 变量条件（segfault 反例） |
| 元素级判决 | 向量化 | `T.vcmp(..., "eq")` | 显式向量链（§1.6.3-#5） |
| 元素级指示化 | 向量化 | `T.vcast(mask, ind, round_mode="rint")` | i1→f32 平面 |
| 元素级归约 | 向量化归约 | `T.reduce_max(ind, mx, dim=1, clear=True)` | 原语（§3.2） |
| 分段合并（路径 C） | 向量化微元 | `T.vmax(acc, mx, new_acc)` + `T.copy(new_acc, acc)` | no-alias 纪律 |
| 补式 / 出栈 | 向量化微元 | `T.reshape` + `T.vsub(one_v, ·)` + `T.vcast(f32→f16)` + `T.vcast(f16→i8)` | `(bm,)` 级 |
| 尾块行数 / 任务 guard | 标量 | `T.min` + `if start < M` | block 级动态边界（§1.6.2-#9） |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (cid, _):
    # 任务不变操作数（循环外）：zero_ub（T.fill）、one_v（T.fill）
    for s in T.serial(num_local_tasks):                  # 静态上界
        start = (cid + s * num_kernels) * block_m
        if start < M:                                    # 动态 guard
            real_m = T.min(block_m, M - start)
            if path == "R":                              # 单 tile
                [§3.3 _tile_body(c0=0, W=N)] -> mx
            else:                                        # 列分块 + 运行 max 合并
                T.fill(acc, tir.const(0, "float32"))
                for t in T.serial(num_full):             # 静态上界
                    [§3.3 _tile_body(c0=t*tile_n, W=tile_n)] -> mx
                    T.vmax(acc, mx, new_acc); T.copy(new_acc, acc)
                if tail > 0:                             # 静态尾块（工厂期条件生成）
                    [§3.3 _tile_body(c0=num_full*tile_n, W=tail)] -> mx
                    T.vmax(acc, mx, new_acc); T.copy(new_acc, acc)
            [§3.3 _epilogue：reshape → vsub(补式) → vcast(f32→f16) → vcast(f16→i8) → slice copy]
```

### 6.3 流水线优化

**初始版本不使用 `T.Pipelined`**（Developer 模式自动施加 auto-multi-buffer；姊妹单元已在任务/tile 循环上
同工具链实测其正确性与竞态边界，§4.5 纪律）。**Stage 4 候选（按 D-2 发射项分析排序）**：
① **§1.6.4 备选 R′（超预算单缓冲驻留）A/B 裁决**（判据与阈值已冻结）；
② 判决与指示化的 **plain 2D `T.Parallel` 形态**（姊妹设计记录的适用域收窄「plain 逐元素 2D loop 在 `bm>=2`
正常向量化」）——若成立可省 1 个显式链 op，需 §8.2 L0-2 复验向量化（前置探针）；
③ **`E_cap` 按 dtype 放宽**（fp16/fp32 8 B/elem ⇒ `196608/2/8 = 12288`；bf16 保持 8192）——
可把 `(256,4096)` 类负载的 `num_logical` 从 128 降到 64（配合扩展阶梯），减少核启动与出栈开销；
④ **C5（vselect 指示量 + `reduce_min`）A/B**：先以探针标定其 UB 单价（未实测），再判定是否较本设计
省 1 个微元 `vsub` 更划算；
⑤ **列分块粒度扫描**（路径 C 的 `tn ∈ 4096/8192` 与 `bm>1` 组合；姊妹探针已证 `bm>1` 多 tile 可编译）；
⑥ persistent 任务循环深度/顺序调优（grid-stride 步长、按核连续性）。

### 6.4 尾块处理

见 §5.4（M 尾块：`real_m` + 双 slice + 垃圾行丢弃；N 向：**实列域无 pad**；路径 C 尾块：静态宽度独立代码块；
guard 静态化边界）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer 模式编译器管理 UB/fragment 依赖与 auto-multi-buffer 流水同步）。

### 7.2 同步点说明

无手工同步点。理由：本算子无跨核通信（无 workspace、无 `T.sync_block_set/wait`）、无 CV 融合
（无 Cube 段）、无跨引擎依赖（无 L0/L1）；UB 内 RAW/WAR 依赖（`T.copy` → `vcmp` → `vcast` → `reduce_max`
→ 微元链）与 task/tile 循环的双缓冲同步全部由 Developer 模式编译器自动插入。
**Stage 3 须复核**：若实测发现 auto-multi-buffer 在 `(bm, W)` 大块形态下未生效（姊妹任务已实测生效；
`CONST-reduce-ub-per-element` 的伴随实测提示「整平面 reduce 与 GM→UB copy 不重叠」是**3-D 整平面形态**
的行为，本设计为 2-D 行归约），则在 Stage 4 按 §6.3-①/⑤ 处理，不改变本节的自动同步选择。

### 7.3 pass_configs 配置

```python
pass_configs = {}
```

即沿用 Developer 模式默认（auto-multi-buffer 开启）。**不显式关闭** auto-multi-buffer——
§4.5 的容量判据按最保守的双缓冲 2x 口径核算，与该默认一致。

### 7.4 CV 融合设计

**不适用**：本算子为纯 Vector（无 matmul），无 Cube 段可分派给 `Scope("Cube")`，无跨核
`T.sync_block_set/wait` 需求，故无 CV 融合。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（优先移植源仓参考实现/测试基准），
> **不得复刻 §0.6 的 NPU 算法**（无补式、无指示平面、无列分块），保证验证独立性。

```python
import torch


def golden_all_logical_reduce(x: torch.Tensor, dim=-1, keepdim: bool = False) -> torch.Tensor:
    """torch.all 语义的逻辑与归约（bool 输出）。

    语义依据 = DESIGN.md §0.1：out = AND_j (x_j != 0)（NaN 视为非零；±0 视为零）。
    实现方式 = 源仓参考实现/测试基准的写法：x.bool().all(dim=...)（torch 权威 API），
    与 Op 层 test 的 ref_program 完全一致（examples/TileOPs/tests/ops/test_all.py）。
    """
    return x.bool().all(dim=dim, keepdim=keepdim)


def golden_all_logical_reduce_2d(x: torch.Tensor, dim=-1, keepdim: bool = False) -> torch.Tensor:
    """等价入口：先 movedim+reshape 成 2D 再归约（供 kernel 级 L0 用例使用）。"""
    d = dim if isinstance(dim, int) else None
    if d is not None and d < 0:
        d += x.ndim
    x2d = x if d is None else x.movedim(d, -1).reshape(-1, x.shape[d])
    out = x2d.bool().all(dim=-1)
    return out.reshape(*x.shape[:d], *x.shape[d + 1:]) if not keepdim else out
```

harness 比对口径：`torch.equal` **精确比对**且断言 dtype == bool
（`examples/TileOPs/tests/ops/test_all.py` 的 `_exact_compare`）。

**判别性输入生成纪律（必须遵守）**：`randn` 输入的零密度约为 0 ⇒ `all` 恒为 True，**恒等实现亦会通过**。
所有测试输入必须注入判别性信号：约 30% 元素显式置零 + 强制角点（`x[0,0] = 0`、`x[0,1] = 1`、
`-0.0`、`NaN`、`+inf`、亚正规），并至少构造：全零行、全非零行、仅一处零的行、NaN 无零行、NaN 混零行
（`examples/TileOPs/tileops/workloads/reduction.py` 的 `_make_logical_input` 已强制首行全零 / 次行全非零，
在其基础上补 NaN 与 ±0 角点）。

### 8.2 精度标准

| dtype（kernel 侧） | atol | rtol | 说明 |
|-------|------|------|------|
| float16 | — | — | **输出为 bool：逐位精确比对**（`torch.equal`），不适用 atol/rtol |
| bfloat16 | — | — | 同上 |
| float32 | — | — | 同上 |
| bool / int32 / int64 / complex（Op 层预转 float32） | — | — | 同上（Op 层转换后再比对 bool 输出） |

> 本算子输出为布尔判决，**容差比对不适用**；`atol/rtol` 仅作为「不得放宽」的显式声明保留。

**L0 门槛测试计划**（由 `tilelang-op-develop` 落地执行）：

| 编号 | 内容 | 判据 |
|------|------|------|
| L0-1 | 路径 R 代表 shape：`(32,256)` fp16/fp32、`(128,512)` fp16/bf16/fp32、`(128,300)` fp16/fp32、`(129,512)` fp16（M 尾块）、`(1,1)`/`(512,)`（退化） | `torch.equal` 全 PASS + dtype == bool |
| L0-2 | 判别性角点集：全零行 / 全非零行 / 单零行 / `±0` / `NaN` 无零行 / `NaN` 混零行 / `+inf` / 亚正规 / `N=1` | `torch.equal` 全 PASS；**NaN 混零行必须为 False、NaN 无零行必须为 True**（若出现相反结论即为 ordered-NE 误用，回到 §3.2 复核 `"eq"`） |
| L0-3 | 路径 C 代表 shape：`(32,32768)` fp32、`(4,128,4096) dim=[0,2]`（M=128, N=16384）、合成 `(2048,16384)` fp32、非整除尾块合成档（`tn` 不整除 N） | `torch.equal` 全 PASS；与路径 R 同 shape（若可编译）逐位一致 |
| L0-4 | 分核边界：`num_logical` 恰等于 / 略小于 / 远大于 40 的三档（如 M=40、M=32、M=128）；`num_logical % 40 != 0` 的负载 | `torch.equal` 全 PASS（尤其尾块 guard 与 grid-stride 映射） |
| L0-5 | 未向量化点复核：kernel IR 检查 `vcmp`/`vcast`/`reduce_max` 为向量形态（无逐元素标量 `ite` 循环）；`bm>=2` 档必须确认未整体标量化（`TRAP-parallel2d-bm-ge2-scalarize`） | IR/汇编复核 + 时延不出现数量级劣化 |

### 8.3 等价性验证工件

| 工件 | 路径 | 状态 |
|------|------|------|
| D-1 等价性机器验证 | `examples/all/_logical_reduce_kernel/verify_equiv.py` | **已执行 exit 0**，A1/A2/A2b/A3/A4/A5 全部 `EQUIV_PASS`（§1.6.1 结果表） |
| 设计期容量探针（D-3） | `examples/all/_logical_reduce_kernel/history_version/design_probe_ub.py` | **已执行**，标定 8/12 B/elem 单价与硬上限（§4.5） |

---

## 9. 风险点与注意事项

### 9.1 已知约束

- **C-1（形态适用域，重要）**：`pattern-library/constants.md` 的 `CONST-reduce-ub-per-element`
  （910B3 / CANN 9.0.0，`repro/TRAP-reduce-ub-per-element.py`）记录整平面 3-D 归约链单价最高约
  128 B/元素；该条目的**适用域限定为 `(CH,1,H*W)` 3-D 整平面 + `dim=2`**。
  本设计的 2-D 行归约形态实测为 **8 B/元素（fp16/fp32）/ 12 B/元素（bf16）**（本任务探针），
  与该条目相差约 16 倍——**该常数不得用于本形态的容量估算**，须以本任务探针实测值为准
  （该条目本身 `status: verified`，仅在形态不匹配处不适用）。该条目在本次 kb_stale_check 中被标为
  `stale`，引用前已按「引用条目前置重验」核对形态适用域。
- **C-2（bf16 更贵）**：bf16 的 `vcmp` 被 lowering 上抛到 f32（IR 可见 `vcast bf16→f32` 作用于两操作数），
  UB 单价 12 B/元素（较 fp16/fp32 高 50%）。`E_cap` 取最坏档 8192 元素，避免为 bf16 单独设档；
  `(256,4096)` bf16 档双缓冲后恰触 192 KB 上限（§4.5），Stage 3 须复标该档并优先以 E_cap 降阶或
  切 C5/路径 C 规避。
- **C-3（能力缺口登记，交 evolver 蒸馏）**：本设计消费了三条 open 能力缺口条目——
  `CG-2026-0026`（BishengIR `T.reduce_*` 归约 lowering 的 UB 记账模型：本任务探针为其提供了
  **2-D 行归约 + dtype 依赖（8 vs 12 B/元素）** 的新数据点）、`CG-2026-0027`（归约与 GM→UB 搬运的流水调度：
  §1.6.4 裁决的核心待测点）、`CG-2026-0028`（reduce 链 UB 预算 + Frontend `vsel` 操作数形态：
  本设计因单价未实测而弃选 C5）。**occurrence 登记与 capability-gaps.md 回写由
  `tilelang-skill-evolver` 在任务终态执行**（本 Stage 只产设计与探针工件；见 `RETROSPECTIVE.md`）。
  另：`T.vcast` 表缺 `f32 → i8` 表项（以 `f32→f16→i8` 两步规避，语义/精度无损，A5 验证），
  属 Frontend API 面的记录性缺口（**不构成算法迫使替换**）。
- **C-4（`VP-2026-0263` 的使用口径）**：该 queue 条目（pending，非已合入规则）给出「`vcmp(eq) → vcast(i1→f32)
  → reduce_sum` 指示链 UB 单价 8.00 B/元素，**独立于 dtype**」。本任务以同族形态（`reduce_max`）独立标定后
  **部分推翻其「独立于 dtype」的外推**（bf16 为 12 B/元素）——引用时按「同链族、dtype 分档」口径使用，
  并作为该条目的修订证据（交 evolver）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|---------|------|----------|
| 用 `vcmp("ne")` 直判非零 | 「直觉最短」实现 | 含 NaN 行误判 False（与 torch/源分歧） | 必须走 `"eq"` + 微缓冲补式（§0.6-R1、A1c、L0-2） |
| 用 `T.vnot` 取反 eq 掩码 | 「非」语义实现 | 编译失败（dtype 表无 bool/int1） | 补式或 `max` 判零（§0.6-R1） |
| 照抄 count_nonzero 的 `T.vadd` 合并 | 路径 C 合并 | 语义错（AND 需 max 合并，非加法） | `T.vmax` + no-alias（§0.6-R4） |
| 照抄源 tiled 的 `T.min(acc, tile_acc)` | 路径 C 合并 | 在补式表示下语义翻错 | 同上 |
| 补式落全宽平面（`1 - ind`） | 微缓冲省事写法 | +1 个全宽 f32 平面 + 1 个全宽 pass，`E_cap` 档降约 1/3 | 补式落 `(bm,)` 微缓冲（§1.6.3-#6） |
| 2D `T.Parallel` 融合 `T.if_then_else` | 判决/指示化 | `bm>=2` 整体标量化（`TRAP-parallel2d-bm-ge2-scalarize`） | 显式 `vcmp`+`vcast` 链（§1.6.3-#5） |
| `T.fill` 传 Python 字面量 | 常量构造 | 前端拒绝 `value must be a tir.PrimExpr` | `tir.const(v, dtype)` |
| `T.copy` 无尾块守卫 | M 非整除 | 越界读/写（源缺陷） | `real_m` + 双显式 slice（§0.6-R6） |
| `prim_func` 内用解析期 `if` 定义 buffer | 路径分支写在一个函数里 | 条件为真报 `ParserError: Undefined variable`（缺陷随 shape 潜伏） | 结构性条件提到 `@tilelang.jit` 工厂层，按变体返回独立 prim_func（§3.3 注） |
| 单块超 UB | `bm·W` 过大 | 编译报 `ub overflow, requires N bits while 1572864 bits available` | `E_cap` / 硬上限约束（§4.5）；反例：fp16 单块 32768 元素需 262176 B |
| `reduce` 域用 bf16 | 直接对 bf16 平面归约 | 编译失败（`reduce` dtype 表 bf16 = ×） | 归约域恒 f32 指示平面（§3.5.1） |
| 用 randn 测试 | 判别性缺失 | all 恒 True，恒等实现亦通过 | 30% 置零 + 角点注入（§8.1） |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| 非整除分块（M） | 尾块三件套（§0.6-R6） |
| 非整除宽度（N / tn） | N 不 pad（实列域）；tn 尾块用独立静态宽度代码块（§5.4） |
| 极小 shape（`(1,1)`、`(512,)`、`N=1`） | `bm = min(bm, M)`；`(bm,1)` tile 上 `reduce_max` 恒等（§5.2 表） |
| 大 M 小 N（`(32768,4)`、`(2048,64)`） | 扩展阶梯把逻辑核数压到 32 / 16（§5.5） |
| 空规约维 `N=0` | 工厂期 `assert N >= 1`；torch 语义为空 AND 恒真，建议 Op 层短路（§9.2-R5） |
| `M=0` | grid=0 边界（§9.2-R6 风险记录，Op 层短路建议） |
| 混合 `±0` / NaN 行 | 由 `"eq"` 的 IEEE 语义天然覆盖（A1/A5 + L0-2） |
| Op 层预转 dtype（bool/int/complex） | kernel 侧只见 {fp16, bf16, fp32}（§0.1） |

> 编号说明：上表 `§9.2-R5` / `§9.2-R6` / `§9.2-R4` 指「本表所属小节的风险条目序号」，与 §0.6 的重设计项编号
> （R1–R6）无关；实现期以语义描述为准。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/all/_logical_reduce_kernel/
├── DESIGN.md                       # 本设计文档
├── verify_equiv.py                 # D-1 等价性机器验证（已执行，exit 0）
├── RETROSPECTIVE.md                # Stage 1 复盘（自进化钩子）
└── history_version/
    └── design_probe_ub.py          # D-3 设计期容量探针（已执行）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `examples/all/_logical_reduce_kernel/DESIGN.md` | 已完成 | 设计文档（含 §0 迁移分析、§1.6 算法调研与优化） |
| `examples/all/_logical_reduce_kernel/verify_equiv.py` | 已完成并执行 | D-1 等价性验证（A1/A2/A2b/A3/A4/A5 全部通过） |
| `examples/all/_logical_reduce_kernel/history_version/design_probe_ub.py` | 已完成并执行 | D-3 容量探针（8/12 B/elem 单价与硬上限） |
| 本算子目录下的 `_logical_reduce_kernel.py` | 待实现 | 算子实现 + 分层测试（Stage 3） |
| `examples/all/_logical_reduce_kernel/RETROSPECTIVE.md` | 已完成 | Stage 1 复盘章节 |

### 10.3 命名规范

- 项目目录名：`all`（manifest family=`reduction` / op_slug=`all`）
- 算子目录名：`_logical_reduce_kernel`（迁移单元名 = GPU 源工厂名）
- 实现文件：`_logical_reduce_kernel.py`
- 测试文件：`test__logical_reduce_kernel.py`（可选，放入 `testing/`）

### 10.4 实现顺序

1. ✅ 设计文档（`DESIGN.md`）+ D-1 验证脚本 + D-3 容量探针
2. ⬜ Golden 函数（§8.1，与 L0-1/L0-2 同步落地）
3. ⬜ 算子实现（`_logical_reduce_kernel.py`）+ 分层测试（L0/L1/L2/boundary，`--level` 入口）
4. ⬜ Stage 2 设计检视复核 `verify_equiv.py` 与探针结论

---

## 11. 附：与姊妹迁移单元（count_nonzero）的关系与差异汇总

> 两者**共用同一 GPU 源工厂**（`_logical_reduce_kernel`）与同一 NPU wrapper 结构（`LogicalReduceKernel`），
> 但为**独立迁移单元、独立 op 目录、独立输出契约**。本设计的迁移分析骨架同源，以下差异必须逐条保持：

| 维度 | 姊妹 count_nonzero | 本单元 all |
|------|-------------------|-----------|
| kind | `op_kind="count_nonzero"` | `op_kind="all"` |
| 输出 | `(M,)` **int64**（精确整数比对） | `(M,)` **bool**（kernel 载体 **int8**） |
| 公式 | `N − Σ_j [x_j == 0]`（计数补式） | `1 − max_j [x_j == 0]`（**集合补式**） |
| 归约算子 | `reduce_sum` | **`reduce_max`** |
| 路径 C 合并 | `T.vadd`（求和） | **`T.vmax`**（求最大；照抄 vadd 即语义错） |
| 微缓冲补式 | `T.vsub(n_v, zc_flat)`（被减数 = N） | `T.vsub(one_v, mx_flat)`（被减数 = 1） |
| 出栈 cast | `f32 → i64`（单步，表项存在） | **`f32 → f16 → i8`（两步，`f32→i8` 无表项）** |
| UB 单价（实测） | 8.00 B/元素（`reduce_sum` 链，**与 dtype 无关**） | **8.00（fp16/fp32）/ 12.00（bf16）B/元素**（`reduce_max` 链，**dtype 相关**） |
| NaN 语义方向 | unordered NE 与 ordered EQ 补式**同向**（NaN 计入非零） | 同向（NaN 计入非零 ⇒ 不破坏 all）；但**误用 ordered NE 的后果相反**：计数少 1 vs **整行翻成 False** |
| 精度域 | 整数计数需 `N <= 2²⁴` 精确域断言 | **无精度域问题**（0/1 指示量的 max 恒精确） |
| `E_cap` 依据 | 8 B/元素（dtype 无关） | **12 B/元素（最坏 dtype bf16）** |

> **不得互相借用**：单价、`E_cap` 依据、补式被减数、合并算子、出栈 cast 链、输出 dtype 六项在
> 两个单元间**均不可照抄**（姊妹任务的 `RETROSPECTIVE.md` 已把「参数与签名不得照抄」列为可迁移教训）。
