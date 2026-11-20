# `_cumulative_kernel`（CumprodFwdOp · 前缀积 scan）算子设计文档

> 迁移任务（harness，**Developer** 模式）：GPU TileOPs `_cumulative_kernel`（`op_kind="prod"`）→ TileLang-NPUIR。
> 源算子：`/tmp/opencode/TileOPs-ref/tileops/kernels/reduction/cumulative.py`（`_cumulative_kernel` 工厂；`op_kind="prod"` 分支）。
> 本设计基于：源码全文解读（Phase M0）、算法族调研（Phase R）、硬件耦合性判定与 NPU 重设计（Phase M1）、算法级优化设计（Phase 2）、**等价性机器验证**（D-1，`verify_equiv.py`）、**设计期探针**（D-3，`history_version/design_probe_p1_logdomain_scan.py` + `design_probe_p2_sign_debug.py`，均在实机 NPU 上执行）。
> 工具链戳：tilelang 0.1.2（dev root build）+ CANN 8.5.0 + **Ascend910_9362（`NPUUtils.get().get_aicore_num()` 实查 = 20，2026-10-13）** + torch 2.7.1（实机 NPU 探针 2026-10-13 实跑）。
> 迁移单元范围 = **`_cumulative_kernel` 的 `prod` 分支**（前缀积）；源文件中的 `sum` 分支与 3 个 `_parallel_scan_*` 辅助 kernel（仅 `op_kind="sum"` 使用，见 §0.3）不在本单元。

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 `tilelang-op-design` skill 的 `references/migration-analysis.md`）：先彻底读懂源算子（§0.1–§0.4），再做算法调研（结论落 §1.6.0，执行位置在 M0 与 M1 之间），最后判定硬件耦合性并给出 NPU 算法设计决策（§0.5–§0.6）。本章结论驱动 §1–§11 的所有设计决策。

### 0.1 源算子语义（做什么）

**数学语义**：对二维输入 `x ∈ (M, N)` 沿最后一维（scan 维 `N`）求**含前缀积（inclusive prefix product）**，输出与输入同 shape：

$$
y[i,j] = \prod_{k=0}^{j} x[i,k] \qquad (0 \le i < M,\ 0 \le j < N)
$$

即 `torch.cumprod(x, dim=-1)`（`op_kind="prod"`，manifest `CumprodFwdOp`，见 `examples/TileOPs/tileops/manifest/scan.yaml` L6–L39；`ref_api: "torch.cumprod"`）。

**前缀语义（依赖顺序）**：`y[i,j]` 依赖同一行内 `x[i,0..j]` 的**有序乘积**——这是一个 **scan（前缀扫描）**，存在行内顺序依赖；行与行之间（`i` 维）**完全独立**。乘法满足交换律与结合律（运算本身无序），但浮点乘法的**舍入结果依赖分组顺序**（`((a·b)·c)` 与 `(a·(b·c))` 可能差 1 ulp）——这正是本算子与纯逐元素映射的本质区别，也是重设计（§0.6）必须论证累加/乘积分组变化的精度影响的原因。

**dtype 语义**：输入 fp16 / bf16 / fp32；输出 `same_as(x)`。源 GPU 实现**全程 fp32 计算域**：载入后 `T.cast(..., "float32")`（L214/L237），running 累加器 `acc` 为 fp32（L195），逐元素乘积与写出在 fp32（L240–L243），输出写回时 `T.cast(out_f32, dtype)`（L247）降回原 dtype。**关键量化点**：产品值域极宽（可上溢至 ±inf、下溢至 0），fp32 中间域是数值必需（见 §1.6.1-E4）。

**边界语义**（源码 + torch 语义共同确定）：
- **非整除 N**：源用 `N_padded = align_up(N, 256)`；越界列以**单位元 1.0 填充**（`_identity = 1.0`，L88）——`prod(S ∪ {1,...}) = prod(S)`，填充列为乘法单位元、不改变结果（L147/L227）。
- **M 尾块**：源码对 `pid_m*block_m + i < M` 有 guard（L210/L233），越界行不载入/不写出。
- **0 / ±inf / NaN**：`torch.cumprod` 按 IEEE 传播——遇 `x=0` 后前缀积恒为 `±0`；遇 `±inf` 后为 `±inf`；`0·inf = NaN`；NaN 传播。源实现（纯逐元素乘）天然满足。
- **空维**：`N=0` / `M=0` 在源实现中无定义（`align_up(0,256)=0`、`n_tiles=0`），本设计列为 §9 风险（harness 不覆盖）。

**host 侧语义**：Op 层 `CumulativeOp._run`（`examples/TileOPs/tileops/ops/scan/cumprod.py` L129–L147）做 validate →（`dim != ndim-1` 时 `movedim` / `contiguous`）→ `reshape(-1, N)` 归一到 `(M, N)` → kernel → `reshape` 回原形 →（`movedim` 还原）。kernel 侧写 `(M, N_padded)` 完整行，Op 层再裁剪 `y[:, :N]`（L147/kernel forward L247）。**`movedim`/`reshape`/裁剪均为实现手段而非算子语义**，golden 不得复刻。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（`x.float().cumprod(dim=-1).to(dtype)`，移植 harness 参考 `examples/TileOPs/tests/ops/test_cumulative.py` 的 `CumprodFwdTest.ref_program`），**独立于本设计的 NPU 算法**。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `(M, N)` | float16 / bfloat16 / float32 | 2D 连续行主序；`M = prod(shape[:dim]) · prod(shape[dim+1:])`，`N = shape[dim]`（Op 层 `movedim`+`reshape` 后） |
| `y` | `(M, N)` | same_as(x) | 逐行前缀积（**非转置布局，与输入同 layout**；迁移必需字段确认） |

工厂契约（harness 固定）：`_cumulative_kernel(M, N, op_kind, dtype)` 返回 `_func(block_m, block_n) → kernel(x)`；语义参数 `M/N/op_kind/dtype` **不变**，后端参数 `threads` 依 K9 去除（NPU 无 CUDA 线程概念，`examples/TileOPs/tileops/kernels/scan/cumprod/cumprod.py` L109/L226–L245）。

### 0.3 实现算法解读（怎么算）

**源 kernel 结构**（`cumulative.py` L176–L254，prod 分支）：`@tilelang.jit(out_idx=[1])` 工厂，`T.Kernel(T.ceildiv(M, block_m), threads=threads)` 一维 grid；每 block 处理 `block_m` 行；`N` 方向按 `block_n` 分 tile 顺序处理，跨 tile 用 per-row running 累加器 `acc[i]` 携带。

**计算步骤分解**（源码每条计算语句均归入步骤，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `for i in T.Parallel(block_m): acc[i] = T.float32(1)`（L198–L199，单位元初始化） | 常量 1 | `acc`（bm,）f32 | 空前缀积的单位元 |
| 2 | `for tile_idx in T.Serial(n_tiles)`（L202，`n_tiles=N_padded/block_n`） | — | 循环 | N 向分块 |
| 3a | 对齐快路径：`T.Parallel(bm, bn)` 逐元素 `shared_in[i,j] = x[...]`（L210–L214）再 `tile_f32[i,j] = cast(shared_in,f32)`（L213–L214） | x | `shared_in`(bm,bn+pad) → `tile_f32`(bm,bn) f32 | 载入 tile + 精度提升 |
| 3b | 尾 tile masked 路径：`tile_f32[i,j] = if_then_else(And(row<M,col<N), cast(x), cast(1.0,f32))`（L215–L228，**1.0 单位元填充**） | x, 1.0 | `tile_f32` | 载入 + 边界（单位元） |
| 4 | **含前缀积（行内顺序依赖）**：`for i in T.Parallel(block_m): for j in T.Serial(block_n): acc[i] = acc[i] * tile_f32[i,j]; out_f32[i,j] = acc[i]`（L240–L243） | tile_f32, acc | `acc`(bm,) f32, `out_f32`(bm,bn) f32 | $y[i,j]=\prod_{k\le j}x[i,k]$ |
| 5 | `shared_out[i,j] = cast(out_f32[i,j], dtype)`（L246–L247） | out_f32 | shared_out(bm,bn+pad) dtype | 精度降回 |
| 6 | `with T.If(row < M): y[row, col] = shared_out[i,j]`（L248–L252） | shared_out | y (GM) | 输出（M 尾块 guard） |

**数据流图**（源硬件视角，覆盖全部 buffer）：

```
GM[x] ─T.Parallel 逐元素载入(+守卫)─▶ SMEM[shared_in (bm,bn+8)] ─T.Parallel cast─▶ FRAG[tile_f32 (bm,bn) f32]
                                                         │
             acc (bm,) f32（跨 tile 驻留的 running 前缀积，单位元 1.0 初始化）
                                                         ▼
   L240–243: for j in Serial(bn): acc[i] *= tile_f32[i,j]; out_f32[i,j] = acc[i]   ← 行内顺序 scan
                                                         │
                       FRAG[out_f32 (bm,bn) f32] ─T.Parallel cast─▶ SMEM[shared_out] ─T.Parallel(守卫)─▶ GM[y]
```
标注：`shared_in`/`shared_out` 带 `_SMEM_PAD=8` 列填充（消 bank conflict）；`N_padded = align_up(N, 256)`（`T.copy` 256 元素对齐约束）。

**循环与并行结构**：`grid = ceildiv(M, block_m)`（一维 block）× `threads`（CUDA 线程）两级；**N 向串行**（外层 `T.Serial(n_tiles)`，内层 `T.Serial(block_n)`），行向并行（`T.Parallel(block_m)`）。默认 `block_n=128`（`_DEFAULT_BLOCK_N`，L52），`block_m` 由 48KB SMEM 预算启发式取 `{1,2,4,8,16}`（L386–L392）。

**host 侧逻辑**：`functools.lru_cache(maxsize=32)` 按 `(M,N,op_kind,dtype)` 缓存编译产物（L63）；Op 层 `movedim`/`reshape`/裁剪（§0.1）。无 im2col 等预处理。

**并行 scan 分支说明**：源另有 `_parallel_scan_{local,carry,propagate}_kernel`（L453–L572）三遍并行 scan，但 `use_parallel = (M < 128 and N > 8192 and op_kind == "sum")`（L354）——**`op_kind="prod"` 永远走顺序 scan**，源码注释明示 "cumprod has no parallel implementation"（L352–L354）。故这三遍 scan 不属于本迁移单元，本设计不对其负责。

**编译配置**：`@tilelang.jit(out_idx=[1])`（GPU target），无特殊 `pass_configs`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | **SMEM tiling（N 向按 block_n 分块）** | 降 shared 用量（`bm·bn` 而非 `bm·N_padded`）、改善访存 | 分块载入 + 块内 running 携带 | shared memory + 256 元素对齐 copy | 硬件强相关（NPU 有 UB 等价，预算/对齐不同） |
| 2 | **running 前缀积（register/SMEM 累加器 `acc`）** | 维持行内 scan 顺序依赖 | per-row 标量累加器跨 tile 驻留 | — | **算法层**（scan 的携带依赖本质），可移植但 NPU 上须改用硬件 scan（见 §0.6-R1） |
| 3 | **fp32 计算域（`tile_f32`/`out_f32`/`acc` 全 fp32）** | 数值稳健（值域极端宽） | 载入 cast fp32 → 计算 → 写回 cast | — | **可移植**（纯数值），NPU 保留（§1.6.1-E4） |
| 4 | **单位元 1.0 填充（越界列）** | 非整除 N 时 pad 列不改变结果 | `_identity=1.0`；tail tile `if_then_else` | — | **纯数学恒等，可移植**（承接为 §0.6-R3 / E3） |
| 5 | **SMEM bank-conflict padding（`_SMEM_PAD=8`）** | 消 32-way bank conflict | 行 stride 68 bank-words（非 32 倍数） | SMEM 32 bank × 4B 结构 | 硬件强相关（NPU 无 bank conflict 概念）→ **舍弃** |
| 6 | **256 元素对齐（`align_up(N,256)`）** | 满足 `T.copy` shared 指令对齐 | pad 到 256 倍数 | GPU copy 向量化边界 | 硬件强相关 → NPU 降到 32B 尾轴（raw N，无需 pad） |
| 7 | **block_m × threads 两级并行** | 行块并行 + 线程级映射 | 一维 grid × CUDA 线程；`threads` 同时是并行宽度 | CUDA 线程模型 | 硬件强相关（NPU 一维 Kernel 无 threads）→ **重新设计** |
| 8 | **block_m 48KB SMEM 预算启发式** | 行列块尽量大以摊薄 block 开销 | `SHARED_MEMORY_BUDGET_BYTES//(2·(bn+8)·elem)` + 阶梯 | SMEM 容量 | 预算值硬件相关 → NPU 换 UB 预算 |
| 9 | **`lru_cache` 工厂缓存** | 消重复编译 | Python 层 `(M,N,kind,dtype)` 键 | — | 可移植 |

> **识别完整性**：以上覆盖源码 `_cumulative_kernel` prod 分支的全部优化性结构（步骤表已把每条计算语句归入步骤 1–6，含 3a/3b 双路径）。无 warp shuffle / mma / swizzle / 异步流水 / online 算法等更深层优化（源实现为朴素顺序 scan + SMEM tiling）。
> **源实现注意点**：源 kernel 无 `N=0`/`M=0` 定义；`_SMEM_PAD` 只影响 bank 冲突不影响结果。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定依据：`migration-analysis.md` §5.3 映射表 / `examples/` 佐证 / `docs/Tilelang.language/` 条目；Phase R 调研结论〔§1.6.0〕作为判定输入——调研确认「log 域 + 硬件 scan 原语」为该语义在 NPU 上的最优算法族，源顺序 scan 被替换。）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 前缀积数学语义（行内有序乘积、输出 same_as(x)、±inf/NaN/0 传播） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 以此为基线（§8.1） |
| running 累加器 `acc` + N 向串行 scan（#2） | 算法 | 无（纯算法层），但**串行形态在 NPU 上代价致命** | 有：硬件 scan 原语 `T.cumsum`（`npuir_cumsum` → `hivm.hir.vcumsum`） | **重新设计** | → §0.6-R1：**log 域变换** `y = exp(cumsum(ln|x|))·sign`；NPU 用 1 次 `T.cumsum` 原语替代 `N` 次串行向量 op（源码内层 `T.Serial(block_n)` 在 NPU 上会发射 `block_n` 个向量 op/tile，见 §1.6.0-R3） |
| fp32 计算域（#3） | 优化/精度 | — | 有（Vector fp32） | **保留** | 全 dtype 统一 fp32 中间域（fp16/bf16 升 fp32，输出 cast 回）；`T.vln`/`T.vexp`/`T.cumsum` dtype 表 bf16 × → bf16 必须升域 |
| 单位元 1.0 填充（#4） | 优化 | — | — | **保留**（作为等价性依据） | 由 §1.6.1-E3 论证承接；NPU 上通过「不 padding / raw N」实现同一意图 |
| SMEM bank-conflict padding（#5） | 优化 | SMEM bank 结构 | **无此问题** | **舍弃** | **舍弃理由**：NPU 无 SMEM bank conflict 概念（migration-analysis.md §5.3「swizzle 防 bank conflict → 舍弃」）；等价收益由 MTE 搬运原生获得 |
| 256 元素对齐 + N_padded（#6） | 优化 | GPU copy 向量化边界 | 无此要求（32B 尾轴） | **等价替换** | kernel 直接接收**原始 N**（Op 层本就不 pad）；`T.copy` 切片天然满足 32B 尾轴；非整除由 slice 尾块处理（§0.6-R3） |
| block_m × threads 两级并行 / `threads` 参数（#7） | 优化 | CUDA 线程 | **无**（本项目一维 Kernel） | **重新设计** | → §0.6-R2：一维 persistent grid（纯 Vector 核数 = 实查 aicore × 2 = 40）+ 核内 `T.serial` 静态边界任务循环；`threads` 依 K9 去除 |
| block_m 48KB SMEM 启发式（#8） | 优化 | SMEM 容量 | 换 UB 预算模型 | **重新设计** | → §0.6-R4：UB 手工预算阶梯 `bm·bn·B_elem ≤ UB_BUDGET`（`B_elem` = 大缓冲每元素字节，见 §4.5） |
| `lru_cache` 工厂缓存（#9） | 优化 | — | — | **保留** | 工厂层与硬件无关 |

> **四态处置统计**：保留 4 项（语义 / fp32 域 / 单位元意图 / lru_cache）｜等价替换 1 项（256 对齐 → 32B/raw N）｜重新设计 3 项（串行 scan → log 域硬件 scan〔R1〕、block×threads → 一维 persistent〔R2〕、48KB 启发式 → UB 预算〔R4〕）｜舍弃 1 项（SMEM bank padding，附理由）。
> **无「舍弃后性能意图丢失」项**：#5 的 bank-conflict 消除意图在 NPU 不存在对应问题；#7/#8 的并行/预算意图由 R2/R4 承接；#2 的顺序 scan 意图（正确的行内前缀依赖）由 R1 的 log 域扫描严格承接（§1.6.1-E1/E2 机器验证）；#4 的单位元意图由 E3 承接。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（对照 Phase R 调研结论 §1.6.0：源算法族〔顺序 scan〕因**无 cumprod 硬件原语 + 串行形态发射灾难**被替换为 log 域硬件 scan 族；源算法优化手段的意图承接已在 §0.5 逐项标注）。

**重设计项 R1：行内 scan（源：`T.Serial(block_n)` running 前缀积）→ log 域硬件 scan**

- **源方案**：`for i in Parallel(block_m): for j in Serial(block_n): acc[i] *= tile_f32[i,j]; out_f32[i,j] = acc[i]`——行内 `block_n` 步串行乘积，跨 tile 由 `acc[i]` 携带。性能意图：正确的行内前缀依赖。
- **NPU 新算法**：**对数域变换 + NPU 硬件 scan 原语**。对每个元素做对数变换，使前缀**积**变为前缀**和**，再用 NPU 的 `T.cumsum`（`hivm::VCumsumOp`，一条向量原语完成整行扫描）求前缀和，最后指数还原；符号单独精确处理：

  $$
  y[i,j] \;=\; \underbrace{(-1)^{\,c_{ij}}}_{\text{sign}}\;\cdot\;\underbrace{\exp\!\Big(\textstyle\sum_{k\le j}\ln|x[i,k]|\Big)}_{\text{magnitude}}\;,\qquad
  c_{ij}=\#\{\,k\le j:\ x[i,k]<0\,\}
  $$

  其中幅值前缀由 `T.cumsum` 沿 scan 轴一次完成（`exp(cumsum(ln|x|))`），符号前缀由**负号计数的奇偶性**精确给出（`c_{ij}` 亦由一次 `T.cumsum` 得到）。N 超 UB 驻留时，跨 tile 用 per-row **加性**标量 carry（log 域与计数域各一）携带——加法 carry 比源的乘法 carry 更简单、无上溢风险。
- **语义保持论证**：**数学恒等（对 `x≠0`）**——`∏ x_k = (∏ sign(x_k))·∏|x_k| = (-1)^{#neg}·exp(Σ ln|x_k|)`；`x=0` 时 `ln|0| = -inf` → 前缀和 `-inf` → `exp(-inf)=0`，与 `torch.cumprod` 遇零后恒零一致；符号奇偶由整数计数给出、**精确无舍入**。**舍入差异**：log 域乘积的累加顺序与源顺序乘积不同（前缀和 vs 顺序乘积），属 `fp32 ulp` 量级；§1.6.1 机器验证（E1/E2）在 fp32/fp16/bf16 × 全部 shape 与角点上**违反率 0**（fp32 最大相对误差 3.5e-6 ≪ 1e-3；fp16 4.3e-4、bf16 3.9e-3 ≪ 5e-2）。已知边界：`|x|` 恰为 `dtype` 最大可表示值附近时 `exp(ln|x|)` 往返可能上溢（§1.6.1 INFO 行 + §9 风险 R3）。
- **意图承接**：源 #2（running 累加器维持顺序依赖）的**正确性意图**由 log 域扫描严格承接（前缀依赖仍在，但实现方式由串行改为硬件 scan）；源 #1（SMEM tiling 降 shared 用量）由 R4 UB 分块承接。

**重设计项 R2：并行结构（源：block × thread + `threads`）→ 一维 persistent 分核**

- **源方案**：`T.Kernel(ceildiv(M, block_m), threads=threads)`；`threads∈{128}` 同时是并行宽度与布局约束。
- **NPU 新算法**：纯 Vector **一维 persistent kernel**——`T.Kernel(num_kernels, is_npu=True) as (cid, _)`，`num_kernels = min(num_logical, vector_cores)`，`vector_cores = NPUUtils.get().get_aicore_num() × 2 = 20 × 2 = 40`（纯 Vector 核数翻倍，实查见 §5.5）；核内 `T.serial(静态上界)` 任务循环（grid-stride 映射 `task = s·num_kernels + cid`）+ 动态 guard `if task < num_logical`。行间独立 → 行块并行；行内 scan 交给 `T.cumsum`。`threads` 停止出现（K9）。
- **语义保持论证**：行块之间**完全独立**（per-row scan，无跨行依赖），任务循环只改变调度、不改变计算；每行结果与 `block_m`/核数无关。M 尾块 `real_m=T.min(block_m, M−off_m)` 只裁掉不存在的行，不改变任一有效行结果。

**重设计项 R3：边界处理（源：256 对齐 pad + masked load + M 尾块 guard）→ raw N + slice 尾块**

- **源方案**：`N_padded=align_up(N,256)`；tail tile 用 `T.if_then_else(And(row<M,col<N), x, 1.0)` 逐元素 masked load（单位元 1.0 填越界列）。
- **NPU 新算法**：kernel 直接接收**原始 `(M,N)`**（Op 层本就不施加 host pad）；N 方向 slice 尾块 `tn=T.min(block_n, N−col0)` + `T.copy(x[off_m:off_m+real_m, col0:col0+tn], x_ub[0:real_m, 0:tn])`（src/dst 双显式 slice）；M 尾块 `real_m=T.min(block_m, M−off_m)` + 双 slice，垃圾行不写出。**取消 masked `if_then_else`**（NPU 上引用循环变量的条件不受支持，且 log 域下单位元 1.0 对应的 `ln(1)=0` 天然由 slice 裁剪替代）。
- **语义保持论证**：**消除 padding 不改变任何输出**——源 pad 值为乘法单位元 1.0，`∏(S ∪ {1,…}) = ∏S`；且源实现中 pad 列本就不进入语义结果。slice 尾块只裁剪越界列/行（每行 scan 独立），不改变任一有效元素的前缀结果。边界核对：零/±inf/NaN 在两侧行为一致（§1.6.1 角点集覆盖）。

**重设计项 R4：block_m / 驻留分档（源：48KB SMEM 启发式 + N 向 tile）→ UB 预算阶梯 + 跨 tile 加性 carry**

- **源方案**：`block_m` 由 `SHARED_MEMORY_BUDGET_BYTES(48KB)//(2·(bn+8)·elem)` + `{1,2,4,8,16}` 阶梯选择；N 向固定按 `block_n=128` 分 tile，乘法 carry。
- **NPU 新算法**：`block_n = min(N, BLOCK_N_MAX)`（`BLOCK_N_MAX=4096`，使 N≥4096 时仍可对齐 UB）；`block_m = max(1, min(M, UB_BUDGET // (block_n·B_elem)))`，`B_elem` = 大缓冲每元素字节（§4.5）。`N ≤ block_n` 时**单 tile / 每行**（无 carry、无分块）；`N > block_n` 时按 `block_n` 分 tile，跨 tile 用 **log 域加性 carry `c_l`（Σ ln|x|）+ 计数 carry `c_k`（#neg）** 携带。carry 更新用 `T.reduce_sum` 求 tile 内 log 和 / 计数和（或折叠进 cumsum 首元素），避免列抽取。
- **语义保持论证**：纯配置/调度选择，不改变任一元素的前缀和值（加法结合律；分组顺序变化属 fp32 ulp 量级，§1.6.1-E3 机器验证 chunk=128 的跨块 carry 与单块结果违反率 0）。**无上溢风险**：log 域 carry 是加法（源的乘法 carry 会在跨 tile 累积中上溢/下溢），对本算子（长 scan 值域极宽）反而更稳健。

### 0.7 标杆实现

- **源算子**：`/tmp/opencode/TileOPs-ref/tileops/kernels/reduction/cumulative.py`（`_cumulative_kernel` prod 分支 L176–L254；§0.1–§0.4 解读对象）；Op 层 `/tmp/opencode/TileOPs-ref/tileops/ops/reduction/cumulative.py`（`CumulativeOp` / `CumprodFwdOp`）。
- **NPU 侧 harness 工件**：Stage 0 抽取件 `examples/TileOPs/tileops/kernels/scan/cumprod/_cumprod_fwd_kernels.py`（`_cumulative_kernel` + 3 个 `_parallel_scan_*`；本任务只做 `_cumulative_kernel`）；已移植 wrapper `examples/TileOPs/tileops/kernels/scan/cumprod/cumprod.py`（`npub::cumulative_fwd` custom_op + `CumulativeKernel`，`threads` 已去除，K5–K9 完成）；Op 层 `examples/TileOPs/tileops/ops/scan/cumprod.py`；规格 `examples/TileOPs/tileops/manifest/scan.yaml`（`CumprodFwdOp` 段）。
- **golden 基准**：`examples/TileOPs/tests/ops/test_cumulative.py`（`CumprodFwdTest.ref_program = x.float().cumprod(dim=-1).to(x.dtype)`；容差 fp32 1e-3 / fp16·bf16 5e-2）；bench `examples/TileOPs/benchmarks/ops/bench_cumulative.py`；workload `examples/TileOPs/tileops/workloads/scan.py`（`CumprodFwdWorkload`，默认 `use_small_range=True` → 正值近 1）。
- **同族已验证参考**：`examples/amax/_simple_reduce_kernel/DESIGN.md`（reduction 布局/persistent 分核/UB 预算三件套参照）；`examples/TileOPs/tileops/kernels/reduction/_primitives.py`（`make_cumulative_scan` 顺序 scan macro，NPU 上保留为**对照/回退**）；`testing/npuir/scan_ops/test_cumsum_dev.py`（`T.cumsum` NPU 合法形态）；`docs/Tilelang.language/排序操作/T.vcumsum.md`（VCumsumOp 规格）。

---

## 1. 概述

### 1.1 算子名称

`_cumulative_kernel`（承载 `CumprodFwdOp` 的 NPU 前缀积 scan kernel；工厂接口保留 `op_kind` 以复用兄弟 sum 算子——`sum` 分支在 NPU 上直接映射 `T.cumsum`，见 §3.2 注）。

### 1.2 功能描述

对 `(M, N)` 输入沿 N 维（scan 维）求逐行**含前缀积**，输出与输入同 shape/dtype；行间独立并按 M 行块分核，行内 scan 用 NPU 硬件 `T.cumsum` 原语在 log 域完成，符号精确单独处理；N 超 UB 驻留时按 tile 分块、用加性 carry 跨块携带。

### 1.3 数学公式

$$
y[i,j] = \prod_{k=0}^{j} x[i,k]\qquad(0 \le i < M,\ 0 \le j < N)
$$

NPU 侧等价计算式（§1.6.1 优化后，供 §3.1 拆解）——对 `x ≠ 0`：

$$
m_{ij} = \textstyle\sum_{k\le j}\ln|x[i,k]|,\qquad
c_{ij} = \#\{k\le j:\ x[i,k]<0\},\qquad
y[i,j] = (-1)^{c_{ij}}\cdot \exp\!\big(m_{ij}\big)
$$

（`x=0` 由 `ln|0|=-inf → exp(-inf)=0` 自动得到零前缀；符号由整数计数奇偶性精确给出。全部中间量在 fp32 域计算，写回时下降为输入 dtype。）

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

纯 Vector 的 **persistent 行块 + log 域硬件 scan** 算法（per-row 独立；与源算法差异逐条标注来源）：

1. **载入**：`T.copy(x[off_m:off_m+real_m, col0:col0+tn], x_f32[0:real_m, 0:tn])` —— GM 行块 → UB，Developer 跨 dtype 自动 `VCast(RINT)` 升 fp32〔§0.6-R3〕；
2. **幅值 + 符号判据**：`T.vabs(x_f32, w_f32)`（`w=|x|`）；`T.vcmp(x_f32, w_f32, cond, "lt")`（`x<|x| ⟺ x<0`，**免零缓冲**）；`T.vln(w_f32, l_f32)`（`l=ln|x|`）〔§0.6-R1〕；
3. **幅值前缀和**：`T.cumsum(l_f32, s_f32, dim=1)`（**硬件 scan，一条原语完成整行**）；多 tile 时 `T.vadd(s_f32, carry_l, s_f32)`（行广播）〔§0.6-R1/R4〕；
4. **符号前缀**：`T.vcast(cond, w_f32)`（bool→0.0/1.0）；`T.cumsum(w_f32, l_f32, dim=1)`（负号计数前缀）；多 tile 时 `T.vadd(l_f32, carry_k, l_f32)`；奇偶性 `par = k − 2·floor(k/2)`（`vcast(round_mode="floor")`，`T.vfloor` 在当前 target 不可用，见 §9 R2），`sign = 1 − 2·par`〔§0.6-R1〕；
5. **还原与写出**：`T.vexp(s_f32, s_f32)`；`T.vmul(s_f32, l_f32, s_f32)`（`y=mag·sign`）；`T.copy(s_f32[0:real_m,0:tn], y[off_m:off_m+real_m, col0:col0+tn])`（UB→GM，自动降回 dtype）；
6. **carry 更新**（仅多 tile）：`T.reduce_sum(局部 log, >, dim=1)` 与计数和 → `T.vadd(carry, 局部和, carry)`〔§0.6-R4〕；
7. **调度**：一维 persistent，`num_kernels = min(num_logical, 40)`，核内 `T.serial` grid-stride 任务循环 + guard〔§0.6-R2〕。

**与源算法差异**：① 行内串行乘积 → log 域 `T.cumsum` 硬件 scan（§0.6-R1）；② block×threads → 一维 persistent（§0.6-R2）；③ 256 对齐 pad + masked load → raw N + slice 尾块（§0.6-R3）；④ 48KB SMEM 启发式 + 乘法 carry → UB 预算阶梯 + 加性 carry（§0.6-R4）。**保留**：fp32 计算域（§0.4-#3）、单位元「不改变结果」意图（#4，由 E3 承接）、`lru_cache`（#9）。

### 1.5 数据流图

```
驻留路径（纯 Vector，单引擎单 kernel；单 tile/行时无 carry）：
GM[x] --T.copy(slice, 自动 cast)--> UB[x_f32 (bm,tn) fp32]
   --T.vabs--> UB[w_f32=|x|] --T.vcmp(x_f32,w_f32)-> UB[cond bool]
                             --T.vln--> UB[l_f32=ln|x|]
   --T.cumsum(dim=1)--> UB[s_f32 = 局部 log 前缀和]  --(+carry_l 行广播)-->
   --T.vexp--> UB[mag]
   cond --T.vcast--> UB[neg 0/1] --T.cumsum(dim=1)--> UB[k 计数前缀] --(+carry_k)-->
        --parity(vmul/vcast-floor/vmul/vsub)--> UB[sign ±1]
   --T.vmul(mag, sign)--> UB[y_f32] --T.copy(slice, 自动 cast)--> GM[y]

无 L1/L0 参与；无 GM 中间往返；无 workspace；x 仅读一遍、y 仅写一遍。
carry（仅 N>block_n）：per-row 标量 c_l（log 域和）、c_k（负号计数），跨 tile 加法携带。
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先**调研**——对同一数学语义的算法族回答调研四问（§1.6.0；迁移任务在源算子解读〔§0〕后、耦合性判定与重设计〔§0.5/§0.6〕前完成，调研结论已作为 M1 判定输入）；再在保证数学等价的前提下优化公式（§1.6.1，**含机器验证**）；再把保留下来的计算全部交给向量单元（§1.6.2）；最后回答"在哪个轴上向量化"（§1.6.3，阻塞级）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：**沿一维的前缀积（scan）** 这一数学语义的算法族。源算法（顺序 running 乘积 scan）只是基线候选之一。调研深度：**完整调研**（scan/规约类 → R1 候选表 + R3 四口径 + R4 逐候选评估）。
> 信息源：skill 参考表 `.agents/skills/tilelang-op-design/references/algorithm-candidates.md`（`ALG-reduction`/scan 命中行）；`kb_search "cumprod cumsum prefix scan 前缀积"`（命中 `VP-2026-0114`/`CG-2026-0011`/`CG-2026-0010`/`PL-1.18`/`PL-1.16` 等 SSD chunk scan 同族条目——已消费：SSD scan 族以 Cube gemm 双路径落地、其瓶颈结论（段数墙/mte2）与本题纯 Vector 形态不通用，作为"并行 scan 在 NPU 的既有形态"参照）；本仓 `examples/`（`_primitives.py:make_cumulative_scan` 顺序 scan、logsumexp/argmax row-reduction）；`docs/Tilelang.language/排序操作/T.vcumsum.md`（**NPU 硬件 scan 原语**）；`testing/npuir/scan_ops/test_cumsum_dev.py` / `testing/python/language/test_tilelang_language_cumsum.py`（scan 原语合法形态）；源算子 §0.3/§0.4；**设计期探针**（D-3，实机 NPU：`T.cumsum(dim=1)` 可用性 + log 域正确性 + 符号路径正确性，见下文）；结构判据分析。**互联网检索**：本地源已覆盖该成熟算法族（前缀扫描 = 经典并行算法），未触发。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| 0 | **基线：源顺序 running 乘积 scan** | `acc_j = acc_{j-1}·x_j`（行内 `T.Serial(N)`，跨 tile 乘法 carry） | — | — | ✅（基线） |
| 1 | **C1 log 域变换 + 硬件 scan**（选定） | `y = exp(cumsum(ln|x|))·(-1)^{cumsum(x<0)}`；跨块加性 carry | **数学恒等**（对 `x≠0`；`x=0` 由 `-inf→0` 自然处理）；舍入分组变化属 fp32 ulp | 行内 scan 由 `N` 次向量 op → **1 次 `T.cumsum` 原语**；跨块 carry 变加法（无乘积上溢） | ✅ |
| 2 | **C2 Hillis-Steele 移位-乘扫描（直接乘积）** | `for d in 1,2,4,…: y = y · shift(y, d)`（log₂(bn) 步），加乘法 carry | 数学恒等（结合律）；**无需 log/exp、符号天然精确** | op 数 ∝ log₂(bn)（bn=4096 → ≥24 op/tile），高于 C1（~6 op/tile） | ✅（备选，见 §1.6.3） |
| 3 | **C3 两/三遍并行 scan（local + carry + propagate）** | 源 `_parallel_scan_*` 结构（块局部 scan → 块间 exclusive scan → 广播乘） | 数学恒等 | 提升并行度；但需 workspace GM 往返 + 第二/三 kernel（prod 分支源码**未实现**，`use_parallel` 仅 sum） | ✅（复杂度对比） |
| 4 | **C4 全序列矩阵物化** | `cumsum(v) = L̃·v`（下三角全 1 矩阵 @ 向量），log 域后 exp | 数学恒等 | `O(N²)` MAC/行（N=4096 → 1.7e7）——远超 `O(N)` | ❌（复杂度劣） |
| 5 | **C5 乘积转求幂（`cumprod = exp2(cumsum(log2|x|))`）** | 同 C1 换底 e→2 | 数学恒等 | 与 C1 同族；`T.vexp2`/`T.vlog2` 需额外 `tmp` 缓冲（`docs/Tilelang.language/数学操作/T.vexp2.md`/`T.vlog2.md`），小幅劣势 | ❌（C1 的等价变体，无额外收益） |
| 6 | **C6 排序/查找类** | 前缀积无排序或查找表述 | — | — | ❌（无对应结构） |

> **代数化简空间的结构性结论**：前缀积是**结合但非交换、非幂等**的幺半群（乘法群）上的 scan，**任何算法至少要读每个元素一次并做 N−1 次乘法**（信息论下界）。不存在降低乘法次数的代数化简；可行的收益集中在**扫描结构（如何用一条原语替代 N 次串行）**与**值域（log 域使乘法可复用加法 scan 硬件）**。C1/C2 即两条可行路线。

**R2 在线算法**：**有，且基线本已是在线（单遍流式）变体**——前缀积天然是 running 乘积（`acc *= x_j`），单遍、`O(1)` per-row 状态；源的 tiled 形式已经是「chunk 内在线 + chunk 间 carry」的在线算法。**收益口径**：本算子不是「两遍 → 单遍」型（基线已单遍），在线性的价值在于**中间缓冲 O(N) → O(tile)**——C1 的分块 carry 形态正是该在线变体（中间缓冲 = `bm·tn`），用于 N 超 UB 驻留的场景。故 R2 结论 = **在线变体存在且被 C1 的跨块 carry 结构直接采用**（不引入额外遍数）。

**R3 复杂度对比**（四口径；以 manifest 最大负载 `hidden-state (2048,4096)` fp16 标定，M 行并行摊派；口径注明含/不含；带宽受限 vs 发射受限并列标注）：

| 算法候选 | FLOPs（乘法/加法，per 行） | 访存量 Bytes（GM：读 x + 写 y） | 扫描遍数（GM） | 中间缓冲峰值 | **向量 op 数 / 核（本设备 40 核）** |
|---------|------|------|------|------|------|
| **基线（顺序 scan）** | N−1 乘 | N·elem ×2（读写各 1 遍） | 1 | `bm·bn` tile + `acc` | **≈ (M/bm)·N / 40**：bm≈1 → 2048·4096/40 ≈ **2.1e5 op/核**（每核 2.1e5 次 0.5µs 发射 → **灾难性**） |
| **C1 log 域硬件 scan（选定）** | N 次 ln + N 次 exp + (N−1) 加 + 计数链 ≈ 4N 超越/逐元素（但**发射为 O(1)/tile**） | 同上（1 读 1 写） | 1 | `bm·bn` tile（4 fp32 缓冲）+ carry | **≈ (M·N/(bm·bn))·op_tile / 40**：op_tile≈16 → 8.4e6/4096·16/40 ≈ **820 op/核** |
| C2 Hillis-Steele 移位乘 | (N−1) 乘（分步） | 同上 | 1（+bon 次 UB 内移位） | `bm·bn` tile（2 缓冲：y + shift） | **≈ (M·N/(bm·bn))·(2·log₂(bn)+carry) / 40**：bn=4096 → 820·(24+4)/16 ≈ **1435 op/核** |
| C3 三遍并行 scan | (N−1) + 2·n_tiles 合并 | N·elem×2 + workspace 往返 `2·M·n_tiles·elem` | **2–3**（partial 写 + propagate 读） | tile + workspace | 结构更复杂（跨核/多 kernel）；prod 分支源码无实现 |

> **口径说明**：① FLOPs 只计乘法/加法，超越函数（ln/exp）按逐元素 1 op 计；② 访存量含输入读与输出写（`2·M·N·elem`），不含 on-chip 搬运；③ **向量 op 数**为 NPU 上的真实成本口径（`CONST-vector-launch-overhead` ~0.5µs/op 固定发射主导）——这是本算子 NPU 侧的**主导项**（见 R4）；④ 基线 op 数按源码 `T.Parallel(bm)+T.Serial(bn)` 形态估计（内层 `bn` 步串行、每步 1 个行向量 op）。

**R4 硬件亲和性评估**（逐候选对照检查清单；**实测常数/探针证据**同附）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（顺序 scan） | ❌ **主导计算是 N 步串行行向量乘**——每步 1 个向量 op（`bm` 元素），`(M/bm)·N` op 总量，发射主导 → **≈2.1e5 op/核**（0.5µs/op → 数量级灾难，`CONST-vector-launch-overhead`） | ✅ | ✅ | ✅ | ❌ 串行链阻断流水 | ❌ **形态不可用**（→C1） |
| **C1 log 域硬件 scan（选定）** | ✅ 纯 Vector：`T.cumsum`（`hivm::VCumsumOp`，**探针 P1 实证 dim=1 可用**）+ `vabs/vln/vexp/vcmp/vcast/vmul/vadd/vsub`（均 fp32 ✓，`docs/Tilelang.language/数学操作/*`、`比较操作/T.vcmp.md`、`数据类型转换操作/T.vcast.md`） | ✅ 大缓冲仅 ~4 个 `bm·bn` fp32（§4.5，`CONST-capacity-910B2C` UB 192KB） | ✅ 尾轴 32B（`T.copy` 切片天然满足）；dtype 分派（fp16/fp32 用域内、bf16 升 fp32） | ✅ M/N/dtype 全工厂期常量；核内循环边界静态（仅 guard 动态） | ✅ auto-multi-buffer 由 Developer 自动施加（载入-计算-写出） | ✅ **选定** |
| C2 Hillis-Steele 移位乘 | ✅ 纯 Vector（`vmul`+`T.copy` 移位，无异构算子）；符号天然精确 | ✅ 仅 2 个 `bm·bn` 缓冲（最省） | ✅ 移位切片按 `2^d` 对齐 | ✅ | ⚠️ log₂(bn) 步串行移位链，流水困难 | ✅ **备选**（op 数 ≈1.7× C1；无 log/exp 精度损失，见 §1.6.3） |
| C3 三遍并行 scan | ✅ Vector | ⚠️ workspace `2·M·n_tiles·elem`（KB 级） | ✅ | ✅ | ⚠️ 2–3 kernel 启动 + workspace 往返 | ⏸ **Stage 4 候选**（小 M 大 N 负载；prod 分支需自建，首版不做） |
| C4 矩阵物化 | ✅ Vector/Cube | ❌ `[N,N]` 中间爆 UB | — | — | — | ❌ `O(N²)` 复杂度劣 |

**调研结论**：选定 **C1（log 域变换 + `T.cumsum` 硬件 scan + 精确符号前缀）为主选**。关键依据：**R3/R4** —— 源基线的 NPU 成本是 `(M/bm)·N` 个向量 op（发射灾难），C1 把它降为 `(M·N/(bm·bn))·O(1)`（每 tile 固定 ~16 op）；C1 相对 C2 的 op 数优势来自 **`T.cumsum` 一条原语替代 log₂(bn) 步移位链**（bn=4096 时 16 vs 28 op/tile）。**与基线的结构差异一句话**：**「行内 N 步串行乘法 + 跨块乘法 carry」→「log 域：1 次硬件 `cumsum` 求前缀和 + 指数还原 + 精确符号前缀 + 跨块加性 carry」**。
**调研范围**（"无更优算法族替代"的查证范围）：`algorithm-candidates.md` 的 `ALG-reduction`/scan 命中行；`kb_search` 命中条目（SSD chunk scan 同族 5 条已评估——其 Cube gemm 双路径形态为**并行 scan 的 Cube 实现**，与本题纯 Vector + 值域极宽场景不匹配，且引入 Expert MixCV 复杂度，不采纳）；`examples/TileOPs/tileops/kernels/reduction/_primitives.py`（顺序 scan macro）；logsumexp/argmax row-reduction 例；`docs/Tilelang.language/排序操作/T.vcumsum.md` 与 scan 测试；源算子 §0.3/§0.4；**实机探针**（P1/P2）；恒等变形空间自查（C4/C5/C6 全枚举，结构判据排除）；互联网检索未触发（本地源 + 经典算法知识已收敛）。**源算法优化手段承接**（§0.4）：#2 顺序 scan → R1 严格承接（E1/E2 验证）；#4 单位元意图 → E3；#5 bank padding 舍弃（NPU 无此问题）；#6 对齐 → R3；#7/#8 → R2/R4；#3 fp32 域、#9 缓存保留。

**设计期 roofline（D-2；协议：`hardware-cost-model.md` §2；常数带版本戳）**：

器件口径声明：`CONST-mte2-degradation` / `CONST-vector-launch-overhead` / `CONST-capacity-910B2C` 的 origin 工具链为 **Ascend910B2C（24 AICore）**；本设备实查为 **Ascend910_9362（20 AICore → 纯 Vector 40 核）**。下述估算按「per-core 带宽/发射不变 × 40 核」外推——属**未实测假设 + 估算依据**（设备/核数差异已标注；Stage 4 回填实测对账，见 `hardware-cost-model.md` §3）。

| workload（manifest） | dtype | 配置（bm×bn） | (M·N/(bm·bn)) tiles | 流量项 | 发射项（op_tile≈16，0.5µs/op） | **估算下界 = max** |
|---|---|---|---|---|---|---|
| smoke (128,512) | fp16/bf16/fp32 | bm=9 × bn=512 | 15 | 0.26MB ÷ (40·33.9GB/s) ≈ 0.19µs | 15/40·16·0.5 = **3.0µs** | **≈3µs（发射主导）** |
| hidden-state (2048,4096) | fp16 | bm=1 × bn=4096 | 2048 | 33.6MB ÷ (40·20.9GB/s) ≈ 40µs | 2048/40·16·0.5 = **410µs** | **≈410µs（发射主导）** |
| hidden-state (2048,4096) | bf16 | bm=1 × bn=4096 | 2048 | 33.6MB ≈ 40µs | **410µs** | **≈410µs** |
| non-aligned (512,300) | fp16/bf16 | bm=15 × bn=300 | 35 | 0.61MB ÷ (40·33.9GB/s) ≈ 0.45µs | 35/40·16·0.5 = **7.0µs** | **≈7µs** |
| long-seq (64,32768) | bf16 | bm=1 × bn=4096（8 tile/行，带 carry） | 512 | 8.4MB ÷ (40·20.9GB/s) ≈ 10µs | 512/40·20·0.5 = **128µs** | **≈128µs** |

> 估算下界: max(流量项, 发射项)（容量项 n/a——纯 Vector 单引擎、无跨引擎 S/P 往返，`CONST-store-fixpipe-gm-only` 不适用）。**结论：本设计为发射/计算主导（非访存主导）**——这是「无 cumprod 硬件原语、须以 log 域 + 多趟向量链模拟 scan」的物理代价。**Stage 4 优化杠杆**（§9 R4）：① 减少 `op_tile`（`vcos` 替代 4-op 奇偶、合并两条 cumsum、消除 `vabs`）；② 减少大缓冲数以增大 `bm·bn`（inline/复用缓冲，`bm=1→2` 即发射减半）；③ 实验裁决 C2（C1 vs C2）。发射项为串行上界，Developer auto-multi-buffer 下实际可低于该值。**精度乐观性**：本估算的 0.5µs/op 常数在 40 核外推下未实测——偏差 >2× 时 Stage 4 触发 `[DESIGN_LIMIT]` 归因（`hardware-cost-model.md` §3）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法 C1（log 域硬件 scan）。逐项四要素（原式 → 优化后公式 → 等价性论证 → 收益量化）；**等价性机器验证（D-1）**：`verify_equiv.py`（torch CPU，fp64 参照 + dtype 域仿真），结果表内嵌于下；**实机探针（D-3）**：`history_version/design_probe_p1_logdomain_scan.py`（log 域 + `cumsum(dim=1)`）与 `design_probe_p2_sign_debug.py`（符号路径），2026-10-13 实机 NPU 执行。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| E1 | **乘法 scan → log 域加法 scan**（承接 §0.6-R1） | `y_j = ∏_{k≤j} x_k`（行内 N−1 次顺序乘） | `mag_j = exp(Σ_{k≤j} ln|x_k|)`（前缀和 + 指数还原） | **数学恒等**（对 `x_k≠0`）：`∏|x_k| = exp(Σ ln|x_k|)`，对数把乘法群同构到加法群。`x_k=0`：`ln|0|=-inf` → 前缀和 `-inf` → `exp(-inf)=0`，与 `torch.cumprod` 遇零恒零一致。舍入：前缀和 vs 顺序乘积的分组差异属 fp32 ulp（机器验证 E1 违反率 0，fp32 max rel 3.5e-6） | 行内 scan 由 **N 次向量 op → 1 次 `T.cumsum` 硬件原语**；跨块 carry 由乘法（会累积上溢/下溢）→ **加法**（值域中性、更稳） |
| E2 | **符号前缀分离（精确奇偶）**（承接 §0.6-R1） | 顺序乘积自动携带符号（`sign(y_j)=∏sign(x_k)`） | `sign_j = (-1)^{c_j}`，`c_j=Σ_{k≤j}[x_k<0]`（整数计数前缀） | **精确等价**：`∏x_k = (∏sign(x_k))·∏|x_k| = (-1)^{#neg}·∏|x_k|`；计数为整数、奇偶性**无舍入**（机器验证 E2 违反率 0）。`[x<0]` 用 `x<\|x\|` 判据（免零缓冲） | 使 E1 在符号一般输入下仍成立（否则 E1 仅对正值成立）；代价 = 1 次 `T.cumsum` + 奇偶链（§1.6.2-#5/#6） |
| E3 | **单位元 pad 消除 + 跨块加性 carry**（承接 §0.6-R3/R4） | 越界列填 `1.0`（源 masked load）；跨 tile 乘法 carry `acc *= tile_total` | 直接对**原始 N** slice 处理（无 padding）；跨 tile `carry_l += tile_log_total`、`carry_k += tile_neg_total` | **数学恒等**：`1.0` 是乘法单位元且源 pad 列本不进语义结果，消除 padding 不改变任何输出（机器验证 E3 chunk vs 单块，违反率 0）；curry 由乘法改加法 = E1 的直接推论 | 消除 masked `if_then_else` 逐元素分支；N 任意宽度直承（无 `N_padded` 缓冲）；carry 无上溢 |
| E4 | **全 dtype fp32 计算域**（承接 §0.4-#3） | 源：载入 cast fp32 → 计算 → 写回 cast（已 fp32） | 同源：`T.copy` 跨 dtype 自动升 fp32 → log 域计算 → `T.copy` 自动降回 dtype | **保留源数值路径**（非新优化）；且是 API 必要性——`T.cumsum`/`T.vln`/`T.vexp` dtype 表 **bf16 ×**（`docs/Tilelang.language/排序操作/T.vcumsum.md` §2.2.1、`数学操作/T.vln.md`/`T.vexp.md` §2.2.1），bf16 必须升 fp32。舍入点与源一致（fp32 中间 + 输出 cast） | 与源同精度基线；bf16 路径额外 1 次 `VCast`（API 必需，非损失） |

**机器验证结果表**（`verify_equiv.py`，2026-10-13 执行，torch CPU，exit 0；fp64 参照 `x.double().cumprod(-1)`；baseline = 源数值路径逐元素复刻；candidate = E1+E2（+E3 chunked）fp32 域仿真；判定口径：harness 容差违反数 viol（fp32 atol/rtol 1e-3，fp16/bf16 5e-2）+ 非有限模式一致）：

| 验证项 | dtype | 用例数（gated） | 最大绝对误差 | 最大相对误差 | 违反数 | 结论 |
|--------|-------|------|-------------|-------------|--------|------|
| E1 幅值 log 域（正值） | float32 | 13 | 5.5e-8 | 3.45e-6 | 0 | **EQUIV_PASS** |
| E1 幅值 log 域（正值） | float16 | 13 | 2.4e-4 | 4.3e-4 | 0 | **EQUIV_PASS** |
| E1 幅值 log 域（正值） | bfloat16 | 13 | 1.95e-3 | 3.9e-3 | 0 | **EQUIV_PASS** |
| E2 符号分离（符号一般） | float32 | 23 | 5.6e-8 | 3.45e-6 | 0 | **EQUIV_PASS** |
| E2 符号分离（符号一般） | float16 | 23 | 2.4e-4 | 4.3e-4 | 0 | **EQUIV_PASS** |
| E2 符号分离（符号一般） | bfloat16 | 23 | 1.95e-3 | 3.9e-3 | 0 | **EQUIV_PASS** |
| E3 跨块 carry（chunk=128） | float32/16/bf16 | 5×3 | 同 E2 | 同 E2 | 0 | **EQUIV_PASS** |
| INFO：near-max（`\|x\|≈dtype_max`） | float32 / bfloat16 | 1×2 | — | 1.7e-6 | 模式不一致 1 | **INFO（已知边界，见 §9 R3）** |
| INFO：near-max | float16 | 1 | 1.6 | 2.4e-5 | 0 | **INFO(pass)** |
| **合计（gated）** | — | **123** | — | — | **0** | **ALL EQUIV_PASS** |

**实机 NPU 探针结果**（D-3，2026-10-13，Ascend910_9362；脚本 `history_version/design_probe_p1_logdomain_scan.py`、`design_probe_p2_sign_debug.py`）：

| 探针项 | 输入 | 结果 | 结论 |
|--------|------|------|------|
| P1-A `T.cumsum(src,dst,dim=1)` on (2,64) fp32 UB | 随机 | max_abs_err **4.77e-7** | ✅ **`cumsum(dim=1)` 可用**（本地仅有 dim=0 的 4×4 用例——本探针消除最大风险） |
| P1-B log 域 cumprod（正值）on (2,64) fp32 UB | `rand·0.01+0.99` | max_rel_err **8.4e-8** | ✅ 幅值链路端到端正确 |
| P2 符号分离（negcount + 奇偶 sign）on (1,8) fp32 UB | `[-2,3,-4,5,-6,7,8,-9]` | negcnt `[1,1,2,2,3,3,3,4]`、sign `[-1,-1,1,1,-1,-1,-1,1]`、maxerr 0.031（值 ~4e5，rel 8e-8） | ✅ 符号路径正确；确认 **`T.vfloor` 在本 target 不可用**（仅 A5 codegen 注册，`src/target/codegen_npuir_dev_a5.cc:4215`），改用 `T.vcast(round_mode="floor")`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 f32→f32 floor） |

**优化结论**：采纳 E1–E4 共 4 项（E1/E2/E3 为收益项，E4 为保留源数值路径 + API 必要性并附论证）；优化后公式 = §1.3 的两式（供 §3.1 拆解）：`y = exp(cumsum(ln|x|))·(-1)^{cumsum(x<0)}`，全 fp32 域。**已知边界**：`|x|` 恰为 dtype 最大可表示值的极小邻域内，`exp(ln|x|)` 往返可能上溢（INFO 行；§9 R3），非 manifest 负载形态，不阻塞。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点对象：§1.4 NPU 算法的**全部**循环与标量计算点。替代 API 均有 `docs/Tilelang.language/`、`testing/` 或实机探针佐证（列内标注）。

| # | 计算点 | 原实现形态（源/朴素形态） | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|--------|---------------------------|-------------|---------|--------------|
| 1 | **行内 N 步串行乘积 scan**（源步骤 4） | `T.Parallel(bm){T.Serial(bn): acc*=tile[:,j]}`（N 次行向量 op） | **log 域 + `T.cumsum(src,dst,dim=1)` 硬件原语**（1 op/tile；探针 P1 实证） | ✅（消除） | — |
| 2 | 幅值变换（`\|x\|`、`ln`） | 逐元素循环 | `T.vabs` + `T.vln`（`docs/Tilelang.language/数学操作/T.vabs.md`、`T.vln.md`，fp32 ✓） | ✅ | — |
| 3 | 指数还原 | 逐元素 `exp` | `T.vexp`（`数学操作/T.vexp.md`，fp32 ✓） | ✅ | — |
| 4 | 符号判据 + 指示量 | 逐元素 `if x<0` | `T.vcmp(x, \|x\|, cond, "lt")` + `T.vcast(cond, f32)`（`比较操作/T.vcmp.md` §2.3 片上使用 + i1→f32 rint，`数据类型转换操作/T.vcast.md` §2.2.1） | ✅ | — |
| 5 | 符号计数前缀 | 逐元素累加 | `T.cumsum(neg, k, dim=1)`（同 #1 原语） | ✅ | — |
| 6 | 奇偶 → 符号（`par=k−2·floor(k/2)`、`sign=1−2·par`） | 逐元素标量分支/取模 | `T.vmul`/`T.vcast(round_mode="floor")`/`T.vsub` 向量链（探针 P2 实证；**`T.vfloor` 本 target 不可用**，见 §9 R2） | ✅ | — |
| 7 | 幅值×符号 | 逐元素乘 | `T.vmul`（`数学操作/T.vmul.md`，行广播 ✓） | ✅ | — |
| 8 | GM↔UB 载入/写出（含 dtype 升降） | 逐元素 `T.Parallel` + `T.cast` | `T.copy` slice（Developer 跨 dtype 自动 `VCast(RINT)`，`内存操作/T.copy.md` §2.2.1/§2.3-4） | ✅ | — |
| 9 | 跨块 carry 注入（`vadd(s, carry, s)`，行广播 `[bm,tn]+[bm,1]`） | 逐元素加 | `T.vadd`（`数学操作/T.vadd.md` §2.2.2 行广播 ✓） | ✅ | — |
| 10 | tile 内 log/计数的和（carry 更新源） | 逐元素累加 | `T.reduce_sum(src, dst, dim=1)`（`规约操作/T.reduce_sum.md`） | ✅ | — |
| 11 | persistent 任务循环 `for s in T.serial(num_local_tasks)` | 块级串行 | 无 | ❌（保留） | **block 级任务映射 + grid-stride 调度**：核内任务序（`task=s·num_kernels+cid`）是分核结构本身（`core-split-strategy.md` 要素③ 极大规模方案），非逐元素热点；无逐元素等价 API |
| 12 | 任务/块索引算术（`task`、`off_m`、`col0`） | 标量整除/取余 | 无（纯 TIR 标量算术） | ❌（保留） | **block 级索引/任务映射计算**：每 tile O(1) 条，非逐元素热点 |
| 13 | 尾块尺寸（`real_m=T.min(bm,M−off_m)`、`tn=T.min(bn,N−col0)`） | 标量 min | 无 | ❌（保留） | **依赖运行时 pid 的 block 级边界**（循环边界已静态化，仅尾块尺寸动态）；进 slice extents，每 tile O(1) 条 |
| 14 | host 侧 block 参数/路径选择 | Python 工厂期标量 | 无 | ❌（保留） | **host 元数据计算**（不在 kernel 内；每 `(M,N,dtype)` 一次） |

**向量化结论**：逐元素计算点（#1–#10）**全部向量化**（替代 API 均有 `docs/` 条目或实机探针佐证：`T.cumsum`/`T.vabs`/`T.vln`/`T.vexp`/`T.vcmp`/`T.vcast`/`T.vmul`/`T.vadd`/`T.vsub`/`T.reduce_sum`/`T.copy`）；保留 4 类标量/循环（#11–#14），理由类别：分核与任务调度串行（#11）、block 索引（#12）、动态尾块边界（#13）、host 元数据（#14）——**均非逐元素热点**（每 tile/任务 1 次发射量级），与 §6 循环结构一致（§6 无任何逐元素标量循环）。源实现中唯一的逐元素热点（行内 N 步串行扫描与 masked 载入分支）已被 #1/#4/#8 消除。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（`(M,N)` 行主序进、`(M,N)` 同 layout 出）；**核内布局与 lane 映射是设计变量**。本算子 scan 语义为 **scan 类**（规约类的顺序依赖变体）——须枚举「水平 scan（lane→N，沿 scan 轴）vs 垂直 scan（lane→M，串行步进 N）」为正交决策，并评估核内重排候选。

**轴质量评分**：

| 评分项 | 水平 scan（lane→N 轴，`cumsum(dim=1)`） | 垂直 scan（lane→M 行，串行步进 N） | 说明 |
|--------|------|------|------|
| 整除性 | N 由 `T.cumsum` 处理任意宽度（slices 尾块） | M 任意 | 无差异 |
| 尾 lane 浪费率 | 由硬件 scan 内部吸收 | — | 无差异 |
| **scan 链形态** | **硬件 scan 原语一次完成**（`hivm::VCumsumOp`） | 串行步进维 = N 次行向量 op（`(M/bm)·N` op 总量） | **决定性差异**（`CONST-vector-launch-overhead`：≈2.1e5 op/核 vs ≈820 op/核） |
| repack 代价 | 无（I/O 原生行主序即沿 dim=1 连续） | 无 | — |
| UB 容量影响 | `(bm,bn)` 驻留（§4.5） | 同 | — |

**候选矩阵**（含源码 GPU 轴选择对照——源码 thread→行 映射是输入非结论）：

| # | 布局方案 | 向量化轴 / 关键结构 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|--------------------|------------|--------------|------|
| 1 | **行块 × scan 轴驻留 + 硬件 `cumsum(dim=1)` + log 域 + 精确符号**（persistent 按 M 分核） | N（连续，scan 轴） | 无 | `(bm,bn)` 驻留（§4.5）；op_tile≈16（发射主导，§1.6.0 roofline）；探针 P1/P2 实证 | ✅ **主选** |
| 2 | 垂直 scan（lane→M，串行步进 N） | M | 无 | scan 维串行化 → `(M/bm)·N` 次向量 op（2048×4096 → ~2.1e5 op/核，0.5µs/op ≡ 数量级灾难） | ❌（量化否决） |
| 3 | **核内转置（N-major）后用 `cumsum(dim=0)`** | 转置后 M（=原 scan 轴）不连续 | `T.transpose` 二轴交换链 ×2（UB 级） | **纯开销、零收益前提被探针推翻**：探针 P1 已证 **`cumsum(dim=1)` 直接可用**，无需以 dim=0 绕行；核内转置不减少 GM 流量与向量 op 数，反增 2 个 `(bm,bn)` 缓冲 + VTransposeOp 开销（`pattern-library/layout.md` PL-1.1-transpose-chain 实测链代价） | ❌（P1 已推翻该前提） |
| 4 | **Hillis-Steele 移位-乘核内扫描（直接乘积，log-free）** | N（连续） | 无（移位用 slice `T.copy`） | op_tile≈2·log₂(bn)+carry（bn=4096 → ≈28），**高于主选（16）**，但**无超越函数、符号天然精确**、大缓冲更少（2 vs 4） | ⏸ **备选 / Stage 4 实验裁决项**（§1.6.4） |
| 5 | C3 三/两遍并行 scan（局部 scan + 块间 exclusive scan + 传播乘） | N（跨核切分） | workspace GM 往返 | 提升小 M 大 N 核利用率；prod 分支源码无实现，需自建 + workspace + 多 kernel | ⏸ Stage 4 候选（首版不做；§1.6.4 注） |
| 6 | 缓冲组织：**UB staging 直链** vs 额外 fragment 中转 | N | 无 | 直链（本设计）：大缓冲 4×fp32 `(bm,bn)`；如需降缓冲数可探索 in-place（`T.vadd/vsub/vmul` 文档明示支持原地，`T.vcast`/`T.cumsum`/`T.vln`/`T.vexp` 原地性未文档化——**Stage 3 验证后择优**） | ✅ **直链主选**（in-place 列为 Stage 4 杠杆 §9 R4） |

**布局决策结论**：选定 **#1**：核内布局 = 行块 `(bm, bn)` 行主序驻留（UB）、向量化轴 = **scan 轴 N**（`T.cumsum(dim=1)`）、repack 无、分核 = persistent 按 M 行块（§5.5）。该结论同步落入 §3.3 伪代码与 §6 循环结构（scan/逐元素循环内层向量维 = N 轴；buffer 形状 = `(bm,bn)` + `(bm,1)` carry）。弃选理由：#2 量化发射灾难；#3 前提被探针 P1 推翻（dim=1 直接可用）；#4 列入 Stage 4 实验裁决（§1.6.4）；#5 首版不做。

#### 1.6.4 实验裁决备选方案：Hillis-Steele 移位-乘扫描（Stage 4 A/B 三件套）

> §1.6.3-#4 的备选完整结构 + 裁决计划（skill Phase 2 实验裁决模式）。**触发前提**：主选 C1（log 域）与备选 C2（移位-乘）的判定裕度依赖**未文档化常数**——超越函数（`vln`/`vexp`）与普通 `vmul` 的 per-op 成本之比。`CONST-vector-launch-overhead` 记「固定发射成本 ~0.5µs/op 主导」（据此 op 数决定：C1 16 vs C2 28 → C1 胜），但该常数未区分 op 类型，故存在不确定区间。**主选 = C1（当前证据下 op 数占优、实现已验证 P1/P2）**。

**① 分派条件（工厂期静态谓词，per-shape 分派可行性）**：两者均为纯 Vector 单 kernel、同一分核结构，可按 `(M,N,dtype)` 在 `lru_cache` 工厂层共存分派（`_cumulative_kernel` 本就形如 `lru_cache` 特化，无全局二选一约束）。

**② 备选结构（C2，可直接进 Stage 3）**：
- **缓冲**：`y_ub (bm,bn)` fp32、`sh_ub (bm,bn)` fp32（移位副本）；B_elem = 8 B/elem（**优于主选的 17**）→ 同预算下 `bm·bn` 可达 2 倍。
- **算法（inclusive Hillis-Steele，直接乘积，符号天然）**：
  ```
  T.copy(x[off:off+real_m, col0:col0+tn], y_ub)          # 载入升 fp32
  for d in [1,2,4,...,2^⌈log2(bn)⌉/2]:                    # 静态展开 log2(bn) 步
      T.copy(y_ub[0:real_m, 0:tn-d], sh_ub[0:real_m, d:tn])   # 移位副本（首 d 列不写=乘 1）
      T.vmul(y_ub[0:real_m, d:tn], sh_ub[0:real_m, d:tn], y_ub[0:real_m, d:tn])
  # 局部 scan 完成 → 由 slice 取末列或 reduce 得 tile 总积，乘性 carry 传播
  T.copy(y_ub[0:real_m, 0:tn], y[off:off+real_m, col0:col0+tn])
  ```
  - op_tile ≈ 2·log₂(bn) + carry（bn=512 → 22；bn=4096 → 28）。
  - **区域形态注**：`vmul`/`copy` 使用列区间 `[d:tn]`（非全缓冲）——`_get_extent` 支持 slice region，但**区间形态须 Stage 3 首日验证**（探针未覆盖；若区间 op 不可用，退化为全缓冲 + 「首 d 列乘 1」的 filler，op 数 +1/步）。
- **分核三要素**：同主选（`num_logical=ceil(M/bm)`；物理核 40；persistent + 静态 serial）。B_elem=8 → `bm·bn ≤ UB_BUDGET/8`，bm 可 2×（发射减半）。
- **同步**：Developer 自动同步（同主选）。
- **数值**：直接乘积 = 与 golden 同路径（`x.float().cumprod` 的移位分组），**可能比 log 域更接近 golden**（无 log/exp 往返）；符号精确（无需 E2 计数扫描）。

**③ 实验裁决计划**：
- **代表 shape**：`(2048,4096)` fp16/bf16（manifest 最大负载）+ `(64,32768)` bf16（长 scan）。
- **测量指标**：bench harness Task Duration（`bench_cumulative.py`，launch≥30 均值）+ 精度 allclose（harness 容差）为前提。
- **判定阈值**：**C2 端到端 < 0.9× C1 同 shape 延迟 → 该 shape 类分派翻转**（否则维持 C1）。
- **回写路径**：实测数据与结论回写本节裁决记录 + `RETROSPECTIVE.md`；翻转时修订 §1.4/§3.3/§6 增补 C2 分派分支并过 Stage 2 复审；Stage 3 实现期如需提前引入，走 `[DESIGN_LIMIT]` 受控修订路由。
- **设计期探针（D-3）**：**不额外做**——探针预算（≤2）已用于最高风险的 P1/P2（`cumsum(dim=1)` 可用性与 log/符号正确性，均为「不做则设计不成立」的前提）；超越 vs 普通 op 成本比属**运行期吞吐画像**，设计期最小探针难标定，交 Stage 4 实测（符合「探针不能替代 Stage 4 实验裁决」）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（harness 迁移 prompt 指定 developer）。

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（`T.cumsum` 硬件 scan + `vabs/vln/vexp/vcmp/vcast/vmul/vadd/vsub` 链），无 matmul、无 Cube | 不需 L1/L0；`T.alloc_shared`→UB 即可 |
| scan 原语 | `T.cumsum`（Developer + Expert 均可用，`docs/Tilelang.language/排序操作/T.vcumsum.md`；`testing/npuir/scan_ops/test_cumsum_dev.py` 为 **Developer** 通过用例） | Developer 可用 |
| 内存层级 | 仅 GM ↔ UB，无跨引擎、无 workspace | Developer 自动管理 |
| 同步 | 单 kernel 内顺序执行 + persistent 任务循环，无核间协作 | Developer 自动同步即可 |
| 家族先例 | amax/logsumexp/argmax row-vector 均为 Developer 且 L0 通过 | 沿用 |
| 调用方约束 | harness 迁移 prompt 指定 developer | ✓ |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（GM staging + 计算缓冲 + carry；编译器映射 UB） |
| 计算方式 | `T.cumsum`（scan）+ `T.vabs/vln/vexp/vcmp/vcast/vmul/vadd/vsub`（逐元素）+ `T.reduce_sum`（carry 源）；无逐元素标量循环 |
| 同步 | Developer 自动同步（无 `sync_block_set/wait`） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True)` 一维 persistent |

---

## 3. API 映射设计

### 3.1 公式拆解（输入 = §1.6.1 优化后公式）

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_f32 ← x[off_m:off_m+real_m, col0:col0+tn]` | GM → UB 行块载入（跨 dtype 自动升 fp32；src/dst 双 slice） |
| 2 | `w ← \|x_f32\|` | 幅值（同时供符号判据） |
| 3 | `l ← ln(w)` | log 域变换（幅值） |
| 4 | `s ← cumsum_j(l)` | 局部 log 前缀和（硬件 scan，dim=1） |
| 5 | `s ← s + c_l` | 跨块 log carry（行广播；单 tile 时省略） |
| 6 | `cond ← (x_f32 < w)` | 符号判据（`x<\|x\| ⟺ x<0`，免零缓冲） |
| 7 | `n ← cast_f32(cond)`；`k ← cumsum_j(n)`；`k ← k + c_k` | 负号计数前缀（+ 跨块 carry） |
| 8 | `par ← k − 2·floor(k/2)`；`sg ← 1 − 2·par` | 奇偶 → 符号（`par∈{0,1}`、`sg∈{−1,+1}`） |
| 9 | `y_f32 ← exp(s)·sg` | 指数还原 × 符号 |
| 10 | `y[off_m:off_m+real_m, col0:col0+tn] ← y_f32` | UB → GM 写出（自动降回 dtype） |
| 11 | `c_l ← c_l + Σ_j l`；`c_k ← c_k + Σ_j n` | carry 更新（多 tile；用局部和） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 | 佐证 |
|------|----------|-------------|------|------|------|
| 1 载入 | GM→UB 行块（升 fp32） | `T.copy(x[off_m:off_m+real_m, col0:col0+tn], x_f32[0:real_m, 0:tn])` | 双显式 slice；跨 dtype 自动 `VCast(RINT)` | Developer | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.1/§2.3-4、§2.2.2-2 |
| 2 幅值 | `\|·\|` | `T.vabs(x_f32, w_f32)` | 同 shape fp32 | Developer | `docs/Tilelang.language/数学操作/T.vabs.md` |
| 3 log | `ln` | `T.vln(w_f32, l_f32)` | fp32 ✓（bf16 ×→须升域） | Developer | `docs/Tilelang.language/数学操作/T.vln.md` §2.2.1 |
| 4 前缀和 | `cumsum_j` | `T.cumsum(l_f32, s_f32, dim=1, reverse=False)` | **dim=1**；fp16/fp32 ✓（bf16 ×） | Developer | `docs/Tilelang.language/排序操作/T.vcumsum.md` §2.2.1；`testing/npuir/scan_ops/test_cumsum_dev.py`；**探针 P1-A 实证 dim=1** |
| 5 log carry | 行广播加 | `T.vadd(s_f32, c_l, s_f32)` | `[bm,tn]+[bm,1]` 行广播 | Developer | `docs/Tilelang.language/数学操作/T.vadd.md` §2.2.2 |
| 6 符号判据 | `x<\|x\|` | `T.vcmp(x_f32, w_f32, cond, "lt")` | bool 输出留片上 | Developer | `docs/Tilelang.language/比较操作/T.vcmp.md` §2.3（bool 不写 GM）；**探针 P2 实证** |
| 7a 指示量 | bool→f32 | `T.vcast(cond, n_f32, round_mode="rint")` | i1→f32 rint | Developer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 |
| 7b 计数前缀 | `cumsum_j` | `T.cumsum(n_f32, k_f32, dim=1, reverse=False)` | dim=1 | Developer | 同步骤 4 |
| 8 奇偶 | `k−2⌊k/2⌋` | `T.vmul(k_f32,0.5,t)` → `T.vcast(t,t,round_mode="floor")` → `T.vmul(t,2,t)` → `T.vsub(k_f32,t,t)`；`sg=1−2·t`：`T.vmul(t,2,t)` → `T.vsub(1.0,t,t)` | **`T.vfloor` 本 target 不可用**（§9 R2）→ 用 `vcast` floor | Developer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（f32→f32 floor）；`数学操作/T.vmul.md`、`T.vsub.md`；**探针 P2 实证** |
| 9 还原 | `exp·sign` | `T.vexp(s_f32, s_f32)`；`T.vmul(s_f32, t, s_f32)` | fp32 ✓ | Developer | `docs/Tilelang.language/数学操作/T.vexp.md` |
| 10 写出 | UB→GM（降 dtype） | `T.copy(s_f32[0:real_m, 0:tn], y[off_m:off_m+real_m, col0:col0+tn])` | src 显式截断 | Developer | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 |
| 11 carry 更新 | 局部和 | `T.reduce_sum(l_f32, rs, dim=1, clear=True)`；`T.vadd(c_l, rs, c_l)` | 同 dtype | Developer | `docs/Tilelang.language/规约操作/T.reduce_sum.md` |
| 分核 | persistent | `T.Kernel(num_kernels, is_npu=True) as (cid,_)` + `T.serial(num_local_tasks)` + `if task < num_logical` | 一维 + 静态界 + 动态 guard | Developer | `docs/Tilelang.language/编译器提示操作/T.Kernel.md`；amax DESIGN §5.5 |

> **注（`sum` 分支前瞻）**：工厂保留 `op_kind`。兄弟单元 `CumsumFwdOp`（`op_kind="sum"`）在 NPU 上可**直接** `T.cumsum(x_f32, y_f32, dim=1)`（无需 log/符号），单位元为 0。本设计只实现 `prod` 分支。

### 3.3 计算伪代码

工厂期全静态量：`block_n`、`block_m`、`num_logical/num_kernels/num_local_tasks/n_tiles`。dtype 分派：`float32` 原域（`T.copy` 无 cast）；`float16`/`bfloat16` 由 `T.copy` 自动升 fp32（bf16 必需，dtype 表 ×）。

```python
# 工厂期（Python）
BLOCK_N_MAX = 4096
B_ELEM = 4 * 4 + 1          # 4 个 fp32 大缓冲 + 1 个 bool（大缓冲每元素字节，§4.5）
UB_BUDGET = 81920           # 80KB 手工预算（×~1.7 auto-multi-buffer ≈ 136KB ≤ 192KB）
vector_cores = 40           # NPUUtils 实查 20 × 2（纯 Vector）

def _cumulative_kernel(M, N, op_kind, dtype):        # op_kind == "prod"
    assert op_kind == "prod", "sum 分支为兄弟迁移单元"
    block_n = min(N, BLOCK_N_MAX)
    block_m = max(1, min(M, UB_BUDGET // (block_n * B_ELEM)))
    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        num_logical = (M + block_m - 1) // block_m
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = (num_logical + num_kernels - 1) // num_kernels
        n_tiles = (N + block_n - 1) // block_n
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M, N), dtype)):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                xf  = T.alloc_shared((block_m, block_n), "float32")
                wf  = T.alloc_shared((block_m, block_n), "float32")
                lf  = T.alloc_shared((block_m, block_n), "float32")
                sf  = T.alloc_shared((block_m, block_n), "float32")
                cond= T.alloc_shared((block_m, block_n), "bool")
                c_l = T.alloc_shared((block_m, 1), "float32")   # log carry
                c_k = T.alloc_shared((block_m, 1), "float32")   # 计数 carry
                rs  = T.alloc_shared((block_m, 1), "float32")
                for s in T.serial(num_local_tasks):             # 静态上界
                    task = s * num_kernels + cid                # grid-stride
                    if task < num_logical:                      # 动态 guard
                        off_m  = task * block_m
                        real_m = T.min(block_m, M - off_m)
                        if n_tiles == 1:                        # 单 tile/行：无 carry
                            T.copy(x[off_m:off_m+real_m, 0:N], xf[0:real_m, 0:N])
                            T.vabs(xf, wf)
                            T.vcmp(xf, wf, cond, "lt")
                            T.vln(wf, lf)
                            T.cumsum(lf, sf, dim=1, reverse=False)
                            T.vcast(cond, wf, round_mode="rint")
                            T.cumsum(wf, lf, dim=1, reverse=False)
                            _parity_sign(lf, wf)                # 见下 macro
                            T.vexp(sf, sf)
                            T.vmul(sf, wf, sf)
                            T.copy(sf[0:real_m, 0:N], y[off_m:off_m+real_m, 0:N])
                        else:                                   # 多 tile + 加性 carry
                            for i in T.Parallel(block_m):
                                c_l[i, 0] = T.float32(0); c_k[i, 0] = T.float32(0)
                            for t in T.serial(n_tiles):
                                col0 = t * block_n
                                tn   = T.min(block_n, N - col0)
                                T.copy(x[off_m:off_m+real_m, col0:col0+tn], xf[0:real_m, 0:tn])
                                T.vabs(xf, wf); T.vcmp(xf, wf, cond, "lt"); T.vln(wf, lf)
                                T.cumsum(lf, sf, dim=1, reverse=False)      # 局部 log 前缀
                                T.vadd(sf, c_l, sf)                         # + log carry
                                T.vcast(cond, wf, round_mode="rint")
                                T.cumsum(wf, lf, dim=1, reverse=False)      # 局部计数前缀
                                T.vadd(lf, c_k, lf)                         # + 计数 carry
                                _parity_sign(lf, wf)
                                T.vexp(sf, sf); T.vmul(sf, wf, sf)
                                T.copy(sf[0:real_m, 0:tn], y[off_m:off_m+real_m, col0:col0+tn])
                                # carry 更新（用局部和；须在 cumsum 前取，实现见 §3.4 注）
                                # T.reduce_sum(局部 l / n, rs, dim=1, clear=True); T.vadd(c_l, rs, c_l)
        return main
    return _func

# macro：奇偶 → 符号（wf 为 scratch，结果落 wf）
def _parity_sign(kf, wf):
    T.vmul(kf, 0.5, wf)                    # wf = k/2
    T.vcast(wf, wf, round_mode="floor")    # wf = floor(k/2)   （T.vfloor 本 target 不可用）
    T.vmul(wf, 2.0, wf)                    # wf = 2*floor(k/2)
    T.vsub(kf, wf, wf)                     # wf = parity ∈ {0,1}
    T.vmul(wf, 2.0, wf)                    # wf = 2*parity
    T.vsub(1.0, wf, wf)                    # wf = sign ∈ {-1,+1}
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.cumsum(src, dst, dim=1, reverse=False)` | `docs/Tilelang.language/排序操作/T.vcumsum.md`；`testing/npuir/scan_ops/test_cumsum_dev.py`（Developer，dim=0）；`testing/python/language/test_tilelang_language_cumsum.py`（dim=1 GPU 用例） | ✅ 文档 + **探针 P1-A 实机 dim=1 通过**（max_abs_err 4.8e-7） |
| `T.vabs` / `T.vln` / `T.vexp` | `docs/Tilelang.language/数学操作/T.vabs.md` / `T.vln.md` / `T.vexp.md`（fp16/fp32 ✓） | ✅ 文档 + **探针 P1-B 实机通过** |
| `T.vcmp`（bool 输出）+ `T.vcast(i1→f32)` | `docs/Tilelang.language/比较操作/T.vcmp.md` §2.2.1/§2.3；`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 | ✅ 文档 + **探针 P2 实机通过** |
| `T.vcast(f32→f32, round_mode="floor")` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 | ✅ 文档 + **探针 P2 实机通过** |
| `T.vmul` / `T.vadd` / `T.vsub`（行广播 `[M,N]⊙[M,1]`、原地 dst） | `docs/Tilelang.language/数学操作/T.vmul.md`、`T.vadd.md`、`T.vsub.md` §2.2.2/§2.3 | ✅ 文档 |
| `T.reduce_sum(src, dst, dim=1, clear=True)` | `docs/Tilelang.language/规约操作/T.reduce_sum.md` | ✅ 文档（同族 reduce 用例通过） |
| `T.copy`（GM↔UB、slice 尾块、Developer 跨 dtype 自动 cast） | `docs/Tilelang.language/内存操作/T.copy.md` §2.2.1/§2.2.2/§2.3-4 | ✅ 文档 + 探针 P1 跨 dtype 载入通过 |
| `T.Kernel(一维, is_npu=True)` + `T.serial` + 动态 `if` guard | `docs/Tilelang.language/编译器提示操作/T.Kernel.md`；amax DESIGN §5.5 | ✅ 先例 |
| `T.min`（尾块尺寸） | amax/logsumexp 同族先例 | ✅ 先例 |
| ~~`T.vfloor`~~ | **本 target 不可用**（仅 A5 codegen 注册，`src/target/codegen_npuir_dev_a5.cc:4215`；`codegen_npuir_api.cc` 无 floor 分发；探针 P2 实证原地 vfloor 无效） | ❌ **改用 `T.vcast(round_mode="floor")`**（§9 R2） |

**carry 更新实现注**：carry 更新需 tile 的 log 和 / 计数和（`Σ_j l`、`Σ_j n`）。若在 `T.cumsum` **之后**取和，`l`/`n` 已被前缀和覆盖，故两点须在覆盖前完成——推荐**用独立缓冲保存局部 raw 值再做 `T.reduce_sum`**，或**将 carry 折叠进 cumsum 首元素**（`l[0] += c_l`、`n[0] += c_k` 后再 cumsum，则末列即新 carry，但需列抽取）。**实现形态在 Stage 3 以最简可编译路径定**（两种均为 O(1) 发射/块，不改变算法语义与复杂度）；首版建议：额外 `T.reduce_sum` 于 `T.vcmp`/`T.vln` 之后、`T.cumsum` 之前对 `lf`/`ln` 副本取和。

---

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 persistent grid（§5.5） |
| 纯 Vector 算子核数翻倍 | Yes | 40 = `get_aicore_num()×2` 实查（§5.5） |
| bf16 不被 scan/log/exp 支持 | Yes | 全 dtype `T.copy` 自动升 fp32 后计算（§1.6.1-E4） |
| `T.vfloor` 本 target 不可用 | Yes | `T.vcast(round_mode="floor")`（§9.2-R2） |
| GEMM 要求 M/N 为 block 整数倍 | No | 无 GEMM；N/M 非整除用 slice 尾块（§5.4） |
| L0C 容量上限 | No | 无 Cube 计算 |
| host 侧禁止改动输入真实内容 | Yes | 无 host pad；全部边界处理在 kernel 内 |
| UB 容量 192KB（auto-multi-buffer 膨胀） | Yes | §4.5 80KB 手工预算（×~1.7 ≤192KB） |
| bool 不可写 GM | Yes | `cond` 仅片上消费（`T.vcmp` → `T.vcast`） |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键差异）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 并行 | `T.Kernel(grid, threads=threads)` 两级 | 一维 `T.Kernel(n, is_npu=True)` | threads 去除（K9）；persistent + serial（§0.6-R2） |
| `target` | 无显式 target | `target="npuir"` | `@tilelang.jit(out_idx=[1], target="npuir")` |
| 行内 scan | `T.Serial(block_n)` running 乘积 | `T.cumsum(dim=1)` 硬件 scan（log 域） | §0.6-R1 / §1.6.1-E1/E2 |
| SMEM tiling | `alloc_shared` 48KB 预算 + 256 对齐 | UB 192KB + 32B 尾轴 | §4.5 预算阶梯（§0.6-R4） |
| 边界处理 | 256 对齐 pad + `T.if_then_else` masked load | 原始 N + slice 尾块 | §0.6-R3（padding 消除，E3） |
| 精度域 | 全量 fp32 上抛 + 降回 cast | 全 dtype `T.copy` 自动升 fp32 + 输出 cast | §1.6.1-E4 |
| 分档 gate | 48KB SMEM 启发式 | `bm·bn·B_ELEM ≤ UB_BUDGET` | §0.6-R4 |

#### 3.5.3 CV 融合设计（不适用）

本算子为**纯 Vector**（`T.cumsum` + v-prefix 链），无 Cube/matmul 段，无跨引擎数据通路、无 workspace、无 `sync_block_set/wait`。CV 融合（Cube epilogue 归 Vector、双 Scope 流水等）**无可融合对象**——故不设独立 CV 融合章节。若未来 `sum` 分支或长序列变体引入 Cube（如矩阵物化 C4），另评估；本迁移单元明确不做。

#### 3.5.4 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `testing/npuir/scan_ops/test_cumsum_dev.py` | 高（`T.cumsum` NPU Developer 合法形态） | `T.alloc_shared` + `T.cumsum(src,dst,dim,reverse)` + `T.copy` 结构；dtype 表 |
| `docs/Tilelang.language/排序操作/T.vcumsum.md` | 高（scan 原语规格） | 参数/`cum_dims`/dtype 表/示例 |
| `examples/amax/_simple_reduce_kernel/DESIGN.md` | 高（row-vector Developer、persistent 分核、UB 预算三件套） | 一维 persistent 结构、分核公式、UB 手工预算、L0 分层 |
| `examples/TileOPs/tileops/kernels/reduction/_primitives.py`（`make_cumulative_scan`） | 中（顺序 scan macro，NPU 上的对照/回退） | 顺序 scan 的确切语义（对照本设计的 log 域等价） |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/`（+DESIGN.md） | 中（row-reduction Developer，尾块 slice + 分层测试） | 尾块 `T.min` + 双 slice、Kernel 结构、L0 测试 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(M, N)` | float16 / bfloat16 / float32 | 2D 连续行主序；M、N 为工厂期常量（无动态轴）；raw N（无 host pad；Op 层本就不施加 `F.pad`） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(M, N)` | same_as(x) | 同输入 shape/layout；`out_idx=[1]` 由 jit 分配（无 `N_padded` 裁剪需求——直接写 N 列） |

### 4.3 中间缓冲区（驻留/tile 路径）

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_f32` | `(bm, bn)` | float32 | UB（`alloc_shared`） | GM staging + 符号判据源 |
| `w_f32` | `(bm, bn)` | float32 | UB | `\|x\|` → 后续作指示量/奇偶 scratch |
| `l_f32` | `(bm, bn)` | float32 | UB | `ln\|x\|` → 计数前缀 |
| `s_f32` | `(bm, bn)` | float32 | UB | log 前缀和 → 指数还原 → 输出 |
| `cond` | `(bm, bn)` | bool | UB | `x<\|x\|` 判据（片上，不写 GM） |
| `c_l` / `c_k` | `(bm, 1)` | float32 | UB | 跨 tile log / 计数 carry（多 tile 时） |
| `rs` | `(bm, 1)` | float32 | UB | reduce 结果（carry 更新） |

### 4.4 内存搬运路径

```
纯 Vector 路径（无 Cube/L1/L0 参与）：
GM[x] --T.copy(slice, 跨 dtype 自动升 fp32)--> UB[x_f32]
   --T.vabs--> UB[w_f32] --T.vcmp(x_f32,w_f32)-> UB[cond]
                         --T.vln--> UB[l_f32]
   --T.cumsum(dim=1)--> UB[s_f32] --(+c_l 行广播)-->
   cond --T.vcast--> UB[w_f32 0/1] --T.cumsum(dim=1)--> UB[l_f32] --(+c_k)-->
        --parity(vmul/vcast-floor/vmul/vsub)--> UB[w_f32 = sign]
   --T.vexp(s_f32)--> UB[mag];  --T.vmul(mag,sign)--> UB[s_f32]
   --T.copy(slice, 自动降 dtype)--> GM[y]
x 仅读一遍、y 仅写一遍；无 GM 中间往返；无 workspace。

关键约束：GM 不可直达计算缓冲，必须经 UB；所有计算在 UB（alloc_shared）上完成；
         T.cumsum/T.vln/T.vexp dtype 表 bf16 × → 全 dtype 统一 fp32 域。
```

### 4.5 UB 内存预算

**预算公式**：`block_m·block_n·B_ELEM + 微缓冲 ≤ UB_BUDGET`。本设计大缓冲 = 4 个 fp32 `(bm,bn)`（`x_f32`/`w_f32`/`l_f32`/`s_f32`）+ 1 个 bool `(bm,bn)` → `B_ELEM = 4×4 + 1 = 17` B/elem；微缓冲（`c_l/c_k/rs`）≈ 3·bm·4 字节（可忽略）。`UB_BUDGET = 81920`（80KB 手工预算）。依据链：UB = 192KB（`CONST-capacity-910B2C`，`pattern-library/constants.md`）；auto-multi-buffer ~1.7× 膨胀 → 80KB×1.7 ≈ 136KB ≤ 192KB ✓（amax 同族先例用 64KB 更保守；本设计缓冲更多，取 80KB 并在 Stage 3 以 `ub overflow` 报文核对）。**discipline**：编译器不 DCE 未用 `alloc`——dtype/路径分派若产生未用缓冲须拆独立 prim_func（`TRAP-tvm-parser-rules`）。

**逐 workload 预算验证**（`block_n = min(N, 4096)`，`block_m = max(1, min(M, 4818 // block_n))`）：

| workload | dtype | N | bn | bm | `bm·bn·17` | ×1.7 | 结论 |
|----------|-------|---|----|----|-----------|------|------|
| smoke (128,512) | fp16/bf16/fp32 | 512 | 512 | 9 | 78.3KB | 133KB | ✅（贴限，Stage 3 核对） |
| hidden-state (2048,4096) | fp16/bf16 | 4096 | 4096 | 1 | 69.6KB | 118KB | ✅ |
| non-aligned (512,300) | fp16/bf16 | 300 | 300 | 16 | 81.6KB | 139KB | ✅（贴限） |
| long-seq (64,32768) | bf16 | 32768 | 4096 | 1 | 69.6KB | 118KB | ✅（8 tile/行 + carry） |

> **实现纪律（Stage 3）**：① 若编译报 `ub overflow`，优先降 `block_n`/`block_m` 一档；② **Stage 4 杠杆**：`T.vadd/vsub/vmul` 文档明示支持原地操作，`T.vcast/T.cumsum/T.vln/T.vexp` 的原地性未文档化——若 Stage 3 验证可原地复用，`B_ELEM` 可降至 ~9，`block_m·block_n` 翻倍、发射项减半（§9 R4）；③ bf16 输入由 `T.copy` 自动升 fp32，无需额外 `x_work` 缓冲。

### 4.6 动态轴定义

无（M、N、dtype 均工厂期静态；核内全部循环边界静态，运行时仅 `task` guard 与尾块尺寸 `real_m`/`tn` 动态）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")   # out 自动分配 (M,N)，dtype 同 x
def _func(): ...                              # K9：无 threads；block 参数由工厂闭包固化
```
工厂缓存：沿用 harness `CumulativeKernel` 的 kernel cache + 源码 `lru_cache` 语义（§0.4-#9 保留）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector（scan + 逐元素链）。**判定依据**：全部计算为 `T.cumsum`（Vector scan 原语）+ v-prefix 逐元素 + `T.reduce_sum` + UB 搬运，无 matmul；数据通路 GM↔UB；无 Cube/L1/L0/L0C。

### 5.2 Block 划分

```python
# 工厂期（全部静态 Python 量）
BLOCK_N_MAX = 4096
B_ELEM = 17                       # 4 fp32 大缓冲 + 1 bool（§4.5）
UB_BUDGET = 81920                 # 80KB 手工预算
vector_cores = NPUUtils.get().get_aicore_num() * 2     # 纯 Vector 翻倍（§5.5 实查 = 40）

block_n = min(N, BLOCK_N_MAX)                          # scan 轴 tile
block_m = max(1, min(M, UB_BUDGET // (block_n * B_ELEM)))
n_tiles = ceildiv(N, block_n)                          # 1 → 无 carry；>1 → 加性 carry
num_logical     = ceildiv(M, block_m)
num_kernels     = min(num_logical, vector_cores)
num_local_tasks = ceildiv(num_logical, num_kernels)     # 核内静态任务上界
```

**scan 轴不跨核**（行内依赖；一条 scan 链由单核完成）；**M 向按 `block_m` 行块分核**（行间独立）。`block_n` 优先取大（减少 `n_tiles` 与 carry 开销，且使 N≤4096 时单 tile/行）。

### 5.3 约束分析

- **对齐**：尾轴 32B 由 `T.copy` 切片天然满足（`N=300` 非 32B 倍数，slice 尾块处理）。
- **UB 容量**：§4.5 逐 workload 验证。
- **dtype 约束**：`T.cumsum`/`T.vln`/`T.vexp` dtype 表 **bf16 ×**（`排序操作/T.vcumsum.md`、`数学操作/T.vln.md`/`T.vexp.md` §2.2.1）→ 全 dtype 升 fp32。
- **`T.vfloor` 不可用**（本 target）→ `T.vcast(round_mode="floor")`（§9 R2）。
- **bool 不写 GM**（`比较操作/T.vcmp.md` §2.3）→ `cond` 仅片上消费。
- **编译器陷阱**：`T.Parallel` 只读 UB（不得读 GM）；条件 alloc / 条件定义变量禁止（`TRAP-tvm-parser-rules`）→ dtype/路径分派拆独立 prim_func。

### 5.4 注意事项（非整除/边界处理策略）

- **N 非整除 `block_n`**：`tn = T.min(block_n, N − col0)` + `T.copy(x[...:col0+tn], x_ub[0:real_m, 0:tn])`（src/dst 双显式 slice）。
- **M 非整除 `block_m`**：`real_m = T.min(block_m, M − off_m)` + 双 slice；垃圾行不写出（每行 scan 独立，不影响有效行）。
- **`N=0` / `M=0`**：源实现未定义，harness 不覆盖 —— 列入 §9 风险 R6（工厂期可加 `if N==0 or M==0: return 恒等/空`）。
- **零输入**：`ln|0| = -inf` → 前缀和 `-inf` → `exp(-inf)=0`（与 golden 一致，§1.6.1 E2/角点覆盖）。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 docs/开发指南.md §3.3）。

- **① 逻辑核数计算**：`num_logical = ceil(M / block_m)`（纯行并行，scan 轴 N 不跨核）。
- **② 物理核数依据（实查）**：查询代码与返回值——
  `from tilelang.utils import NPUUtils; NPUUtils.get().get_aicore_num()` → **20**（2026-10-13 实查，设备名 `Ascend910_9362`，torch 2.7.1 + CANN 8.5.0）。本算子**纯 Vector → 核数翻倍 = 20 × 2 = 40**（依据 core-split-strategy.md §1 要素② 与项目约定）。**禁止文档假设值**：本仓 `CONST-aicore-910B2C` 记的另一设备为 24，与本设备实查 20 不同——本设计以实查 40 为准，并在 §1.6.0 roofline 标注该设备/核数差异对常数外推的影响。
- **③ 规模判定与分核方案**（统一 persistent 结构：`num_kernels = min(num_logical, 40)`；核内 `T.serial(静态上界 = ceil(num_logical/num_kernels))` + grid-stride `task = s·num_kernels + cid` + 动态 guard `if task < num_logical`）：

| workload | block_m×block_n | num_logical | 判定 | num_kernels | 核内任务数（静态上界） | 依据 |
|----------|-----------------|-------------|------|-------------|----------------------|------|
| smoke (128,512) fp16/bf16/fp32 | 9×512 | 15 | **逻辑 ≤ 物理（15 ≤ 40）** | 15 | 1 | 无需适配：行块独立、每核 ≤1 任务、单波 |
| non-aligned (512,300) fp16/bf16 | 16×300 | 32 | **逻辑 ≤ 物理（32 ≤ 40）** | 32 | 1 | 无需适配 |
| hidden-state (2048,4096) fp16/bf16 | 1×4096 | 2048 | **极大规模（2048 > 40，bm 受 UB 预算不可再增）** | **40** | **52**（ceil(2048/40)） | 固定 40 核 + 核内 `T.serial`（静态边界 52 + guard） |
| long-seq (64,32768) bf16 | 1×4096 | 64 | **极大规模（64 > 40）** | 40 | 2 | 固定 40 核 + `T.serial`（静态 2 + guard）；负载不均 2:1 |

> 核内串行循环边界全部为**静态值**（工厂期 Python 整数 `ceil(num_logical / num_kernels)`）；动态性仅存在于 guard（`if task < num_logical`）与尾块尺寸。**负载不均说明**：极大规模下存在「部分核多 1 任务」的固有 ≤1 任务粒度不均，被 persistent 结构摊薄（相比超发 2048 核串行调度属数量级改善）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block 级 / tile 级调度循环；scan 由硬件原语完成。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| M 方向（block 级） | persistent 任务循环 | `T.serial(num_local_tasks)`（静态上界 + grid-stride + guard） | 极大规模核内串行（§5.5）；消核启动开销 |
| N 方向（tile 级） | tile 循环 | `T.serial(n_tiles)`（静态边界） | N 超 UB 驻留时分块 + 加性 carry（§0.6-R4）；单 tile 时省略 |
| **scan（N 方向行内）** | **硬件原语** | `T.cumsum(src, dst, dim=1, reverse=False)` | 一条原语替代 N 次串行向量 op（§1.6.2-#1） |
| 逐元素（幅值/log/符号/还原） | 向量 op | `T.vabs/T.vln/T.vexp/T.vcmp/T.vcast/T.vmul/T.vadd/T.vsub` | 全缓冲整块向量化 |
| carry 源 | 向量归约 | `T.reduce_sum(src, dst, dim=1, clear=True)` | (bm,bn)→(bm,1) 行和 |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (cid, _):
    for s in T.serial(num_local_tasks):                    # 静态上界
        task = s * num_kernels + cid                       # grid-stride
        if task < num_logical:                             # 动态 guard
            off_m  = task * block_m
            real_m = T.min(block_m, M - off_m)
            for t in T.serial(n_tiles):                    # 多 tile（单 tile 时循环体退化为 1 次）
                col0 = t * block_n
                tn   = T.min(block_n, N - col0)
                T.copy(x[off_m:off_m+real_m, col0:col0+tn], xf[0:real_m, 0:tn])   # 载入
                T.vabs(xf, wf)                                                    # |x|
                T.vcmp(xf, wf, cond, "lt")                                        # x<0
                T.vln(wf, lf)                                                     # ln|x|
                T.cumsum(lf, sf, dim=1, reverse=False)                            # 幅值前缀和
                T.vadd(sf, c_l, sf)                                               # +log carry（单 tile 可省）
                T.vcast(cond, wf, round_mode="rint")                              # 指示量
                T.cumsum(wf, lf, dim=1, reverse=False)                            # 计数前缀
                T.vadd(lf, c_k, lf)                                               # +计数 carry
                # 奇偶 → 符号（落 wf）
                T.vmul(lf, 0.5, wf); T.vcast(wf, wf, round_mode="floor")
                T.vmul(wf, 2.0, wf); T.vsub(lf, wf, wf)
                T.vmul(wf, 2.0, wf); T.vsub(1.0, wf, wf)
                T.vexp(sf, sf)                                                    # 幅值还原
                T.vmul(sf, wf, sf)                                                # ×符号
                T.copy(sf[0:real_m, 0:tn], y[off_m:off_m+real_m, col0:col0+tn])   # 写出
                # carry 更新（多 tile；源值须在 cumsum 覆盖前取，见 §3.4 注）
```

### 6.3 流水线优化

**初始版本不使用 `T.Pipelined`**：tile 循环上的 GM→UB 载入与计算重叠由 Developer 的 auto-multi-buffer 自动施加；**Stage 4 候选**：① `--enable-auto-multi-buffer=false` + 显式缓冲（若 UB 膨胀降为 ~1.10× 可升 `bm`）；② C2 移位-乘备选（§1.6.4）；③ tile 循环 `T.Pipelined` 软件流水（长 N 负载有潜在收益，待测）。

### 6.4 尾块处理

见 §5.4：M 尾块 `real_m` + 双 slice；N 尾块 `tn` + 双 slice；`N=0/M=0` 见 §9 R6。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer 模式，编译器管理 UB 依赖与 auto-multi-buffer 流水同步）。

### 7.2 同步点说明

无手动同步点（单 kernel、无核间协作、无跨引擎传输）。任务循环内 MTE2（GM→UB 载入）→ V（`cumsum`/逐元素）→ MTE3（UB→GM 写出）的生产者-消费者依赖由编译器自动插入 `pipe_barrier`。`T.cumsum` 的 src/dst 为不同缓冲（`lf→sf`、`wf→lf`），无原地竞态面；carry 为 per-row 小缓冲（每 tile 末尾更新、下一 tile 头消费），其依赖由编译器识别。

### 7.3 pass_configs 配置

无特殊 pass 配置（Developer 默认；`TILELANG_ASCEND_MODE=Developer`）。Stage 4 探索项：`--enable-auto-multi-buffer=false`（§6.3-①）。

---

## 8. 验证方案

### 8.1 Golden 函数

> golden 以 **§0.1 源算子语义**为唯一依据实现（移植 harness 参考 `examples/TileOPs/tests/ops/test_cumulative.py` 的 `CumprodFwdTest.ref_program`），**不复刻 §0.6 的 NPU 算法**（无 log 域变换、无符号分离、无 persistent 分核）——保证验证独立性。

```python
import torch

def golden_cumprod(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """PyTorch 参考实现（CPU），对应 torch.cumprod(x, dim)。

    输入 : x, dtype ∈ {float16, bfloat16, float32}
    输出 : y, shape == x.shape, dtype = same_as(x)
    """
    return x.float().cumprod(dim=dim).to(x.dtype)
```

（harness pytest 侧 ref：`CumprodFwdTest.ref_program = x.float().cumprod(dim=-1).to(x.dtype)` 同上。）

### 8.2 精度标准

来自 `examples/TileOPs/tests/ops/test_cumulative.py` 的 `_cumprod_tol`：

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 1e-3 | 1e-3 | 高精度 |
| float16 / bfloat16 | 5e-2 | 5e-2 | 低精度容忍（前缀积更数值敏感） |

> §1.6.1 机器验证显示 log 域变换的实际误差远优于容差（fp32 max rel 3.5e-6；fp16 4.3e-4；bf16 3.9e-3），容差按 harness 契约保留（不放松亦不收紧）。

### 8.3 L0 门槛测试计划

L0 聚焦「编译通过 + 基本精度正确」，覆盖 manifest 负载 + 边界；完整 L1/L2/Boundary 分层套件交由 `tilelang-op-develop` 展开。

| 测试编号 | Shape (M,N) | dtype | 路径/配置 | 验证目标 | 容忍度 |
|---------|-------------|-------|-----------|----------|--------|
| L0-1a/b/c | (128,512) | fp16 / bf16 / fp32 | 单 tile/行 bn=512 bm=9 | smoke 三 dtype | 5e-2 / 5e-2 / 1e-3 |
| L0-2 | (2048,4096) | fp16 | 单 tile/行 bn=4096 bm=1；极大规模 persistent（2048→40×52） | manifest hidden-state fp16 + 多波次 | 5e-2 |
| L0-3 | (2048,4096) | bf16 | 同上（bf16 升域） | manifest hidden-state bf16 | 5e-2 |
| L0-4 | (64,32768) | bf16 | **多 tile**（8 tile/行）+ 加性 carry | manifest long-seq + carry 路径 | 5e-2 |
| L0-5a/b | (512,300) | fp16 / bf16 | 单 tile/行 bn=300 bm=16 | **N 非对齐**（300 非 32/256 倍数） | 5e-2 |
| L0-6 | (129,512) | fp16 | bm=9（129%9=3） | **M 尾块**（real_m 路径） | 5e-2 |
| L0-7a/b | (128,300) | fp16 / bf16 | — | harness `test_cumulative.py` 非对齐用例 | 5e-2 |
| L0-8 | (1,512) / (512,1) | fp16/fp32 | bm=1/bn=min(N,4096) | M=1 极小行 / N=1 退化 | 5e-2 / 1e-3 |
| L0-9 | 构造 (8,257)：含 0 / 负数 / 交替符号 / 全 1 / 近 1 | fp16/fp32 | — | **角点语义**（`verify_equiv.py` 角点集的 NPU 侧复验） | 5e-2 / 1e-3 |
| L0-10 | (2,64,512) dim=-1 / (2,512,256) dim=1 | fp16 | Op 层 movedim/reshape 后 (M,N) | spec 多维 dim + `test_cumprod_dim_axis1` | 5e-2 |

**通过条件**：全部 L0 项通过 atol/rtol 检查。**NaN/±inf 行为记录**（非门禁，§9 R1）。

### 8.4 完整分层测试套件（交由 tilelang-op-develop 阶段）

- **L1（功能覆盖）**：1D/2D/3D/4D 输入、dim=-1/dim=1、非连续输入、全部 dtype × manifest workload。
- **L2（边界）**：M=1、N=1、M%block_m≠0、N 非 32B 倍数、N>block_n（carry 路径）、动态 shape kernel cache（harness `test_cumprod_dynamic_shape_kernel_cache`）。
- **Boundary**：全 1 / 含 0 / 含负号 / 交替符号 / 近 1 / 大值增长序列 / 长 scan 下溢。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 处理方案 |
|------|------|----------|
| **无 `T.cumprod` 硬件原语** | 前缀积须经 log 域变换 | `exp(cumsum(ln\|x\|))` + 符号分离（§1.6.1-E1/E2；探针 P1/P2 实证） |
| **`T.cumsum`/`T.vln`/`T.vexp` bf16 ×** | bf16 不能直接 scan/log/exp | 全 dtype 升 fp32（`T.copy` 自动 cast；§1.6.1-E4） |
| **`T.vfloor` 本 target 不可用** | 奇偶计算不能直接用 floor | `T.vcast(round_mode="floor")`（探针 P2 实证；§9.2-R2） |
| **UB 192KB + auto-multi-buffer 膨胀** | 大块探索须按膨胀后预算 | §4.5 80KB 手工预算；编译 `ub overflow` 报文即精确需求 |
| **未用 `alloc` 不被 DCE** | 计入 UB 规划 | dtype/路径分派拆独立 prim_func（`TRAP-tvm-parser-rules`） |
| **`T.Parallel` 禁读 GM** | 编译错误 | GM 读写全经 `T.copy`；`T.Parallel` 只用于 (bm,1) carry 初始化 |
| **物理核数因设备而异** | 分核数与 persistent 任务数依赖实查 | 实查 20 → Vector 40（§5.5）；禁沿用其他设备值（如 24） |
| **发射/计算主导** | 性能受向量 op 数限制 | §1.6.0 roofline；§9.2-R4 Stage 4 杠杆 |

### 9.2 风险点明细

| # | 风险 | 说明 | 处置 / 状态 |
|---|------|------|------------|
| R1 | **NaN/±inf/0 传播** | 期望与 `torch.cumprod` 一致（§0.1）；`ln(0)=-inf → exp=0` 已由 §1.6.1 角点验证；但 NPU 上 `T.vln(0)`/`T.vexp(-inf)` 的具体位模式须 Stage 3 L0 用含 0/含 inf/含 NaN 用例实测确认 | 设计期已按语义论证（E2/角点机器验证）；L0-9 记录；**不阻塞** |
| R2 | **`T.vfloor` 不可用** | 仅 A5 codegen 注册（`src/target/codegen_npuir_dev_a5.cc:4215`），本 target（`codegen_npuir_api.cc`）无分发；探针 P2 实测原地 `vfloor` 无效（sign 恒 1） | 改用 `T.vcast(round_mode="floor")`（探针 P2 实证，sign 正确）；**已解决** |
| R3 | **log 域 near-max 上溢** | `\|x\|` 恰为 dtype 最大可表示值邻域时 `exp(ln\|x\|)` 往返可能上溢为 inf（`verify_equiv.py` near-max **INFO** 行：fp32/bf16 模式不一致 1、rel 1.7e-6；fp16 无此现象） | 非 manifest 负载形态；列入已知边界，不阻塞；若 Stage 3 遇极端输入，可对 `\|x\|` 加 `T.vmin(\|x\|, fin.max·(1−ε))` 或对首元素特判 |
| R4 | **发射/计算主导的性能** | §1.6.0 roofline 估算 hidden-state ≈410µs（发射主导），远高于访存地板 ~40µs | Stage 4 杠杆：① 减少 `op_tile`（`vcos` 替代 4-op 奇偶、合并两条 cumsum、消除 `vabs`）；② 减少大缓冲数以增大 `bm·bn`（in-place 复用，`B_ELEM` 17→~9，发射减半）；③ §1.6.4 C2 实验裁决。设计期估算 >2× 偏差触发 `[DESIGN_LIMIT]` 归因（`hardware-cost-model.md` §3） |
| R5 | **roofline 常数设备/版本绑定** | `CONST-mte2-degradation`/`CONST-vector-launch-overhead` origin 为 910B2C（24 核）；本设备 910_9362（20 核 → Vector 40），估算按核数外推 = **未实测假设 + 估算依据** | Stage 4 回填实测对账（`hardware-cost-model.md` §3） |
| R6 | **`N=0` / `M=0`** | 源实现未定义（`align_up(0,256)=0`、`n_tiles=0`）；harness 不覆盖 | 工厂期可加早退（`N==0` → 恒等/空；`M==0` → 空输出）；列入风险，不阻塞 |
| R7 | **carry 更新取值时序** | carry 需 tile 的局部 log 和/计数和，须在 `T.cumsum` 覆盖源缓冲前取（§3.4 注）；实现形态（独立缓冲 + `reduce_sum` vs 折叠首元素）待 Stage 3 定 | 两种均为 O(1) 发射；不影响算法语义；L0-4 覆盖 carry 路径 |
| R8 | **尾块动态 slice 形态** | 统一用 src/dst 双显式 slice（`T.copy.md` §2.2.2 尾块借用规则） | 同 amax/logsumexp 已通过形态；L0-6/L0-5 覆盖 |
| R9 | **wrapper 分档 gate 仍为 GPU 语义** | 现有 `cumprod.py` 的 `default_config`（48KB 启发式 + `threads` 已去）与 `use_parallel` 判定为 GPU 派生；本设计给出 NPU 的 `block_n/block_m/n_tiles` 公式（§5.2） | wrapper 层 `default_config` 的 NPU 化属 wrapper/兄弟单元职责，本设计标注接口契约（`block_m`/`block_n` 传入） |
| R10 | **动态 shape kernel cache** | harness `test_cumprod_dynamic_shape_kernel_cache` 要求每 `(M,N)` 独立缓存条目 | 沿用 `CumulativeKernel._kernel_cache`（`(M,N,dtype,device)` 键）；本设计工厂按 `(M,N)` 特化，兼容 |

### 9.3 特殊场景处理

| 场景 | 处理方式 |
|------|----------|
| N 非 32B 倍数 | 无 padding；`T.copy` 切片保证 32B 尾轴（L0-5） |
| N > block_n（长 scan） | 多 tile + 加性 carry `c_l`/`c_k`（L0-4） |
| M 非 block_m 倍数 | `real_m` + 双 slice + 垃圾行不写出（L0-6） |
| dim=-1 / dim=1 / 多维 | Op 层 `movedim`/`reshape` 归一为 `(M,N)`，kernel 始终沿 dim=1（L0-10） |
| 含 0 / 含负号 / 交替符号 | log 域 + 符号分离天然处理（角点集覆盖） |
| 全 1 | `ln(1)=0` → 前缀和 0 → `exp(0)=1`；符号 +1（L0-9） |
| M=1 / N=1 | `block_m=1`、单块；`block_n=min(1,4096)=1`（L0-8） |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/cumprod/_cumulative_kernel/
├── _cumulative_kernel.py          # 算子实现（kernel 工厂 + golden + L0 测试）【Stage 3】
├── DESIGN.md                      # 本设计文档
├── verify_equiv.py                # D-1 等价性机器验证（E1–E3；ALL EQUIV_PASS，123 gated）
├── history_version/
│   ├── design_probe_p1_logdomain_scan.py   # D-3 探针 P1（cumsum dim=1 + log 域）
│   └── design_probe_p2_sign_debug.py       # D-3 探针 P2（符号路径 + vfloor 缺口）
└── RETROSPECTIVE.md               # Stage 1 复盘（自进化钩子）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成并执行 | 等价性机器验证（§1.6.1 结果表，exit 0，123 gated 全 PASS） |
| `history_version/design_probe_p1_logdomain_scan.py` | ✅ 已完成并实机执行 | 探针 P1 |
| `history_version/design_probe_p2_sign_debug.py` | ✅ 已完成并实机执行 | 探针 P2 |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘章节 |
| `_cumulative_kernel.py` | ⬜ 待实现（Stage 3） | 算子实现（kernel 工厂 + golden + L0） |

### 10.3 命名规范

- 项目目录名：`cumprod`；算子目录名：`_cumulative_kernel`
- 实现文件：`_cumulative_kernel.py`；工厂函数：`_cumulative_kernel(M, N, op_kind, dtype)`
- Golden 函数：`golden_cumprod(x, dim=-1)`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性机器验证（verify_equiv.py）+ 设计期探针（P1/P2）
2. ⬜ Golden 函数（§8.1）
3. ⬜ 算子实现（`_cumulative_kernel` 工厂 + 分档/分核 + log 域 scan + 符号分离，§3.3/§5.2/§5.5）+ L0 门槛（§8.3）
4. ⬜ 与 Golden 比对（atol/rtol §8.2），全过进入 Stage 4

### 10.5 迁移规则遵守确认

| 规则 | 遵守情况 | 说明 |
|------|----------|------|
| 算子名 = `_cumulative_kernel` | ✅ | 不裁剪不变换 |
| `@tilelang.jit` 函数声明不变 | ✅ | 保留工厂签名；移除 K9 `threads`，加 `target="npuir"` |
| `@T.prim_func` 参数名/顺序不变 | ✅ | `main(x: T.Tensor((M,N), dtype), y: T.Tensor((M,N), dtype))` |
| 从源码推断输入规格 | ✅ | 从 GPU 源码 + manifest 推断，未询问用户 |
| Host 侧不改输入 | ✅ | 无 host pad；全部边界处理在 kernel 内 |
| golden 独立于 NPU 算法 | ✅ | §8.1 以 §0.1 语义实现 |

---

## 11. 修订日志

**revision_index: 0（首次设计，2026-10-13）**

本版为 `_cumulative_kernel`（`CumprodFwdOp`，前缀积）的首次设计，无历史回退。

**设计核心（可提炼为 4 条 GPU→NPU 关键重设计）**：
1. **行内 scan**：源「N 步串行 running 乘积」→ **log 域变换 `y = exp(cumsum(ln|x|))·(-1)^{cumsum(x<0)}` + NPU 硬件 `T.cumsum` 原语**（§0.6-R1；无 cumprod 原语故以对数同构复用加法 scan；符号精确分离；探针 P1/P2 实机实证）；
2. **并行结构**：源「block × threads 两级」→ **一维 persistent 分核（实查 20 AICore → 纯 Vector 40 核）+ 核内静态 `T.serial` 任务循环**（§0.6-R2）；
3. **边界处理**：源「256 对齐 pad + masked load + 单位元填充」→ **raw N + slice 尾块**（§0.6-R3；E3 论证 padding 消除不改变结果）；
4. **分档与 carry**：源「48KB SMEM 启发式 + N 向 tile + 乘法 carry」→ **UB 预算阶梯（`B_ELEM` 手工预算）+ 跨 tile 加性 carry（log 域/计数域）**（§0.6-R4；加法 carry 无乘积上溢）。

**验证印记**：`verify_equiv.py` 123 gated 用例 ALL EQUIV_PASS（fp32/fp16/bf16 × shape/角点/chunked）；实机探针 P1 证 `T.cumsum(dim=1)` 可用、log 域正确；P2 证符号路径正确并暴露 `T.vfloor` 缺口（改用 `T.vcast(round_mode="floor")`）。


---

## 附录 A：Stage 4 性能反馈（`[DESIGN_LIMIT]`，归档于 2026-10-16）

> 本节为 `perf_opt/perf_feedback.md` 的**归档副本**（Stage 4 调优期间实测得出）。
> 按 TUNING→DESIGN 受控逆向反馈流程，经用户路由选择「附录补记」归档；原始反馈文件不改写。
> 结论限定于本设计族（Stage-3 C1：log 域 + `T.cumsum`）。

# 性能反馈（[DESIGN_LIMIT]）

## 触发判定

- 设计层归因: `_cumulative_kernel_DESIGN.md` §1.6.0（R4 硬件亲和性）/§1.6.3（布局决策 #1）/§6.1 把
  `T.cumsum` 当作**硬件 scan 原语**（"1 次 `T.cumsum` 原语替代 N 次串行向量 op"、"op_tile≈16"），整套
  C1 选型（log 域 `exp(cumsum(ln|x|))·(-1)^(cumsum(x<0))`）与其 roofline 估算均由此成立。该前提在本工具链
  上不成立：npnir `hivm.hir.vcumsum` 被降为**标量指令**，故 C1 的结构性成本是 `2·M·N` 次标量 element-step，
  非 tiling/参数可解（配置扫描 6765–7318 us 全落 ~1% 噪声带）。
- 实测与假设矛盾: DESIGN §1.6.0 估算最大负载 `hidden-state`（2048×4096 fp16）≈ **410 us**；实测 baseline
  **8422 us（20.5x，>2x 门槛）**，最优调优结果 **6766 us（16.5x）**。编译器对 `via VCumsumOp` 直接给出
  `will execute by scalar instruction with low effiency`，与"硬件 scan 原语一次完成"假设直接矛盾；profile 显示
  `aiv_scalar_ratio = 0.9898`、`aiv_vec_ratio = 0.0101`（99% 标量）。DESIGN §1.6.4 预注册的纯向量 C2 备选亦
  实测慢于 baseline（region-offset 数据流标量化），故结构性 >2x 加速无法在本设计族内兑现。

## 参照锚定

- 参照实现: 同族已在仓库落地的 scan 实现 `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/_ssd_chunk_scan_fwd_kernel.py`
  （CASE-ssd-chunkscan-migration，MixCV Expert Cube+Vector chunk scan）；顺序 scan 基元
  `examples/TileOPs/tileops/kernels/reduction/_primitives.py`（其 `make_cumulative_scan` 与本题 baseline
  同型，无性能参照价值）；`docs/Tilelang.language/排序操作/T.vcumsum.md` 仅声明 dtype/shape 规格，无吞吐/向量性条款。
- 不可移植论证: SSD chunk scan 的 scan 维是**chunk 级状态递推**（少量 chunk 的 Cube gemm 双路径），并非
  cumprod 所需的**逐元素前缀积/前缀和**；移植到本算子需要把逐元素 scan 改写成 chunk 内 Cube 三角矩阵乘 +
  chunk 间传播乘，属算法族替换 + Expert MixCV 跨引擎重写，超出 Stage 4 调优授权范围。**未实证假设声明**：本
  阶段未实现该 Cube 路线，故"其能否 >2x 优于 scalar cumsum"为未实证推断，非实测结论；本反馈不据此断言绝对
  上限，仅断言**冻结的 C1（log 域 + `T.cumsum`）设计族**在当前工具链上的局部地板。未找到与本算子同形（纯
  Vector、逐元素、无 Cube 状态）的快速 scan 参照实现。

## 假设

`_cumulative_kernel_DESIGN.md` §1.6.0 调研结论与 §1.6.3 布局决策 #1：以 `T.cumsum(dim=1)`（`hivm::VCumsumOp`）
作为硬件向量 scan 原语，`op_tile≈16`，据此估算 `hidden-state` ≈410 us、`long-seq` ≈128 us，并给出
"发射/计算主导"而非"访存主导"的结论（§1.6.0 roofline 表）；§9 R4 亦列出"减少 op_tile"作为 Stage 4 杠杆。
DESIGN 自身已标注：若 0.5 us/op 常数外推偏差 >2x，Stage 4 应触发 `[DESIGN_LIMIT]` 归因。

## 实测

msprof op 口径（kernel `main`，launch-count 20，warm-up 5；device Ascend910_9362，40 向量核）：

| workload | dtype | baseline Task Duration (us) | final Task Duration (us) | 备注 |
|---|---|---:|---:|---|
| hidden-state-scan 2048×4096 | fp16 / bf16 | 8422.01 / 8425.57 | 6766.00 / 6767.38 | ratio 0.423→0.446 % |
| long-seq-scan 64×32768 | bf16 | 2185.68 | 2089.52 | ratio 0.483→0.438 %（向量 op 数下降） |
| non-aligned-cumprod 512×300 | fp16 / bf16 | 200.00 / 200.62 | 127.12 / 127.28 | ratio 0.429→0.444 % |

- 诊断信号: `aiv_scalar_ratio = 0.9898`、`aiv_vec_ratio = 0.0101`（hidden-state baseline）；
  编译期告警 `Op 'hivm.hir.vcumsum' will execute by scalar instruction with low effiency`。
- 对照实测（单测探针，2048×4096 fp16 同 tiling）：纯搬运 16.0 us；搬入+全缓冲 `T.vmul`+搬出 16.5 us；
  单次 region-copy + separate-dst 全缓冲 `T.vmul` 15.7 us；而 12 步 region-offset / 链式 region-copy 数据流
  20858–20886 us（全部标量化）。即：向量逐元素算力可用（~16 us 地板），但 scan 所需的移位/前缀数据流在本
  工具链上无法保持向量化。
- raw profile: `perf_opt/profiles/{baseline,final,candidate_ks}/{workload}/`；配置扫描与探针日志见
  `perf_opt/logs/`。
- DESIGN §1.6.4 预注册 C2 备选（纯向量 shift-multiply）实测：Kogge-Stone 变体 9856 us（hidden-state）、
  3038 us（long-seq）、473 us（non-aligned），**全 workload 慢于 baseline**，故 A/B 裁决维持 C1。

## 影响面

- 算子族/shape: 所有依赖 `T.cumsum` / `hivm.hir.vcumsum` 的 scan 类算子（cumsum/cumprod、SSD/linear-attention
  的混合状态 scan 中若走该原语的分支），以及任何把 scan 当 "1 原语/tile" 计入 op 预算的设计估算；本算子全
  shape（N 从 300 到 32768、fp16/bf16/fp32）均受同一标量地板约束。
- 改设计预期影响: 需修订 §1.6.0 roofline（常数校准为"cumsum 为标量"）、§1.6.3 布局决策 #1 与 §9 R4 杠杆
  清单；若要突破该地板，须走算法族替换（Cube 三角矩阵前缀积 / chunk 化两遍 scan + workspace），预计属于
  Stage 1（designer, mode=revision）工作量级。
- 交付影响: 本 Stage 4 交付仍为当前设计族内的最优版本（`perf_opt/_cumulative_kernel.py`，1.05–1.58x，
  Roofline % 不过 0.5%）；60% Roofline 目标在本设计族内结构性不可达。

## 反馈结论

`T.cumsum` 在本工具链被降为标量指令，使 C1（log 域硬件 scan）设计的真实上限是 `O(2·M·N)` 标量成本，
实测比 DESIGN 估算慢 16.5–20.5x 且不可由 tiling 改善——60% Roofline 目标结构性不可达，需设计层修订
（校准 roofline 常数或替换 scan 算法族）。

建议路由: 附录补记
