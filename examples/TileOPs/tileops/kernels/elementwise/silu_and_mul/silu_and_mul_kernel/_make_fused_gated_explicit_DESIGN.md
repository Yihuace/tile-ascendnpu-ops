# `_make_fused_gated_explicit`（silu_and_mul / `SiluAndMulFwdOp`）算子设计文档

> **任务属性**：迁移（harness 子模式，`developer`）。源算子 = TileOPs GPU `FusedGatedKernel._make_fused_gated_explicit`（op 类 `SiluAndMulFwdOp`）。
> **源算子文件**：`examples/TileOPs/tileops/kernels/elementwise/silu_and_mul/_silu_and_mul_fwd_kernels.py`（Stage 0 提取件，接口契约冻结基准）；GPU 上下文 `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（`_make_fused_gated_explicit` @ L646、`FusedGatedKernel` @ L1130、`SiluAndMulFwdKernel` @ L1926）、`/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/activations.py`（`SiluAndMulFwdOp` @ L328）、`/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/_base.py`（`FusedGatedOp` @ L818）。
> **NPU 集成目标**：wrapper `examples/TileOPs/tileops/kernels/elementwise/silu_and_mul/silu_and_mul.py`（`SiluAndMulFwdKernel`，工厂 `(M, N, dtype, op_func)` + 可调用 `(block_size)`，`default_config = {"block_size": 1024}`）；Op 层 `examples/TileOPs/tileops/ops/elementwise/silu_and_mul.py`（`SiluAndMulFwdOp`）。
> **算子族**：elementwise 逐元素融合（gate/value 切片 + SiLU + 乘），纯 Vector，无规约、无 matmul。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

- **数学语义**：单输入 `x`（shape `(M, 2N)`）在末维拆两半——`gate = x[:, :N]`、`value = x[:, N:]`；输出 `y[m, n] = silu(x[m, n]) * x[m, N+n]`，`silu(g) = g · sigmoid(g) = g / (1 + e^{-g})`（SwiGLU 前向的门控激活，`F.silu(x[..., :N]) * x[..., N:]`）。源定义：`_get_effective_op_func` 的 `fused_op(gate, value) = activation_func(gate) * value`（`/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py` L1187–L1200），`SiluAndMulFwdKernel.activation_func(x) = x / (1 + exp2(-x·log2e))`（同文件 L1931–L1937）。
- **I/O 契约**：单输入 `(M, 2N)`，输出 `(M, N)`，同 dtype；`x.shape[1]` 必须为偶数（`FusedGatedOp._validate_runtime_input`），`x` 调用前 `.contiguous()`（`_eager_forward`）。输出 shape `(M, N)` 由 `source_output_shape` 确认。
- **规约语义**：**无规约、无跨元素依赖**——`y[m, n]` 只依赖 `x[m, n]` 与 `x[m, N+n]`，与任何其他输出元素无关；无累加顺序问题（分组/并行/分核变化不改变数值）。
- **dtype 语义**：`float16 | bfloat16 | float32`（manifest `SiluAndMulFwdOp` signature；`SiluAndMulFwdKernel.SUPPORTED_DTYPES`）。**中间计算精度为 fp32**：`activation_func` 显式 `g = T.Cast("float32", x)`（源 L1934），silu 在 fp32 域算，再与 `value`（被提升到 fp32）相乘，末端回舍原 dtype。fp8 路径不在本算子契约内（wrapper 已移除 fp8 包装）。
- **边界语义**：NaN 传播（`silu(NaN)=NaN`，`NaN·v=NaN`）；`silu(+inf)=+inf`（`+inf/(1+e^{-inf})=+inf/1`）；`silu(-inf)=NaN`（`-inf/(1+e^{+inf})=-inf/inf`，与 `torch` 的 `-inf·sigmoid(-inf)=-inf·0` 一致）；大负 `g` 时 `silu→0⁻`（`exp(-g)` 可能溢出为 `inf`，`g/inf→0`，正确）；非整除 shape 由工厂 `T.ceildiv` + 尾块处理；`M` 可为 1（decode）。
- **host 侧语义**：`@functools.lru_cache(maxsize=32)` 按 `(M, N, dtype, ...)` 特化编译；`out_idx=[1]`；Op 层只做 `.contiguous()`（纯视图，无数据物化）。无 im2col / 无广播物化 / 无 reshape 契约。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（`silu(x.float()[:, :N]) * x.float()[:, N:]`，移植 `examples/TileOPs/tests/ops/test_silu_and_mul.py` 的 `SiluAndMulTest.ref_program`），**不复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x`（单输入，打包） | `(M, 2N)` | float16 / bfloat16 / float32 | `gate = x[:, :N]`（列 `0..N-1`，行内连续）、`value = x[:, N:]`（列 `N..2N-1`，行内连续）；行 stride = `2N` |
| `y`（输出） | `(M, N)` | **same_as(x)** | `out_idx=[1]`（`main(x, y)` 的 index 1）；无转置、无布局变换 |

**manifest workload**（`examples/TileOPs/tileops/manifest/elementwise.yaml` `SiluAndMulFwdOp`）：`smoke-2m [1024, 2048]`（fp32/fp16/bf16）、`llama-3.1-8b-swiglu-prefill [2048, 28672]`（fp16/bf16）、`llama-3.1-8b-swiglu-decode [1, 28672]`（bf16）、`ffn-square-8m [4096, 4096]`（fp16/bf16）。roofline：`flops = 6·M·N`、`bytes = 3·M·N·elem_bytes`（2 读 1 写）。

### 0.3 实现算法解读（怎么算）

源文件：`examples/TileOPs/tileops/kernels/elementwise/silu_and_mul/_silu_and_mul_fwd_kernels.py`（Stage 0 提取件）；GPU 上下文 `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（`_make_fused_gated_explicit` @ L646）。

**计算步骤分解**（源码每个计算语句均归入某步骤，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1（host/build） | `block_N = threads · num_per_thread`；grid = `(ceildiv(N, block_N), M)` | threads=128(fp16/bf16)/256(fp32)、npt=8/4 | 编译期常量 / CUDA grid | 任务划分 |
| 2 | `col = (bx·threads + i)·npt + j` | `bx`(列块) `by`(行) 与线程索引 | 输出列下标 `col` | 索引映射 |
| 3 | `gate = x[by, col]`；`value = x[by, N + col]` | GM `x` | 寄存器标量 | gate/value 切片读取 |
| 4 | `g = T.Cast("float32", gate)`；`s = g / (1 + exp2(-g·log2e))` | `gate` | fp32 标量 `s = silu(gate)` | `silu(gate)`（exp2 降代换形态） |
| 5 | `y[by, col] = s * value` | `s`(fp32)、`value`（提升 fp32） | GM `y`（回舍原 dtype） | `· value` |

**数据流与内存访问模式**（GPU 视角）：

```
GM[x] ──标量 coalesced 读（gate 段 + value 段，行内连续）──> register ──cast f32 → exp2 → div → mul──> register ──标量写──> GM[y]
（无 SMEM / 无 alloc_shared；FusedGated 工厂直接在全局内存上做逐线程标量读写）
```

**循环与并行结构**：CUDA grid `(ceildiv(N, block_N), M)` 两维（`bx`=列块、`by`=行）；block 内 `for i, j in T.Parallel(threads_arg, npt_arg)` 二维线程并行，每线程标量处理 `npt` 个连续元素（`col = (bx·threads + i)·npt + j`，即线程 `i` 独占 `npt` 个连续列 → 128-bit 向量化访存）。无 `T.Pipelined`、无 fragment 驻留、无中间 GM 往返。

**host 侧逻辑**：`@functools.lru_cache(maxsize=32)` 按 `(M, N, dtype, op_func, threads, npt)` 特化工厂编译；`@tilelang.jit(out_idx=[1])`；`SiluAndMulFwdKernel.forward` 预编译缓存 `_compiled_fn`，Op 层 `.contiguous()` 后调用。策略机（`direct` / `explicit_parallel`）在 build 期由 Python `if` 静态选择，默认 `explicit_parallel`（`FusedGatedKernel.DEFAULT_STRATEGY`）。

### 0.4 优化手段解读（为什么快）

> 对照 migration-analysis.md §4.1 清单逐项核对。FusedGated 工厂**无 SMEM tiling、无 warp shuffle、无 swizzle、无异步流水、无 online 算法、无 persistent**——显式优化手段如下。

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | `explicit_parallel`：`T.Parallel(threads, npt)` | 满并行 + 每线程多元素 ILP | 每线程独占 `npt` 个连续元素（`col=(bx·t+i)·npt+j`） | CUDA block/thread 线程模型 | 硬件强相关 |
| 2 | 连续段 coalesced 读写 | 访存带宽 | gate/value 两段行内连续；`npt` 连续元素 → 128-bit 事务 | CUDA memory coalescing | 硬件强相关 |
| 3 | `npt=8`（fp16/bf16）/`npt=4`（fp32） | 加宽加载事务 | 每线程 `npt` 连续元素拼成 128-bit（fp16×8 / fp32×4） | GPU LDG.128 | 硬件强相关 |
| 4 | **fp32 计算域**（`T.Cast("float32", x)`） | 精度 / 与参考一致 | silu 全链在 fp32，末端回舍原 dtype | 无（纯数值路径） | 可移植（NPU 侧为**必需**，§0.5） |
| 5 | **`exp2` 降代换**（`exp2(-g·log2e)` 替代 `exp(-g)`） | 减超越函数代价 | CUDA 上 `exp2f` = 1 条 MUFU.EX2，`expf` 为多 op 序列 | GPU MUFU 单元 | 硬件强相关（NPU 无此收益，§0.5） |
| 6 | 二维 grid `(列块, 行)` | 行级任务独立 | `by` 直接映射输出行，`bx` 映射列块 | CUDA grid | 硬件强相关（NPU 需一维化，§0.5） |
| 7 | 寄存器级标量计算（无 SMEM） | 省片上共享内存与同步 | 每元素就地算完即写 | GPU 大寄存器堆 | 硬件强相关 |
| 8 | 工厂闭包 + `lru_cache(32)` | 摊薄 JIT 编译开销 | 每参数组合只编译一次 | 无 | 可移植 |
| 9 | 策略机（`direct` vs `explicit_parallel`，默认后者） | 依 dtype/shape 选最优 body | build 期 Python `if` 选择 | 无 | 可移植（契约只保留 explicit_parallel，wrapper E6） |

> 识别不出来 ≠ 不存在。上面 9 项均需在 §0.5 逐项处置；源**未**使用 SMEM tiling / swizzle / warp shuffle / 异步流水 / online 算法（对照清单无遗漏）。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 数学语义 `y = silu(gate)·value` | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.4 规则 1）；逐元素无累加顺序问题 |
| I/O 契约（`(M,2N)`→`(M,N)`，same dtype，`out_idx=[1]`） | 语义 | 无 | — | **保留** | §0.2 已确认输出 shape；无转置 / 无 host 布局变换 |
| **fp32 计算域**（手段 4） | 语义（dtype） | 无 | 有（`T.vcast` 升/降 + fp32 域 v-prefix 链） | **保留（必需）** | PL-1.17-subfp32-fp32-transit：vadd/vsub/vmul/vexp 的 bf16 支持缺失（`docs/Tilelang.language/数学操作/T.vexp.md` §2.2.1 bf16「×」）→ bf16 唯一计算路径是 fp32；且 fp16 golden 经 fp32 opmath + 单次舍回，原生域逐步舍入差 2–3 ulp |
| `exp2(-g·log2e)` 降代换（手段 5） | 优化 | GPU MUFU | 有（`T.vexp`，同 `T.vexp2` 均单指令） | **等价替换（撤销降代换）** | §0.6 R1 / §1.6.1 Item 1；`docs/Tilelang.language/数学操作/T.vexp.md` §3（→`hivm.hir.vexp`）与 `T.vexp2.md` §3——NPU 上两者同为单向量指令，GPU 侧省 MUFU 的前提不成立 |
| `T.Parallel(threads, npt)` 逐元素标量链（手段 1/3/7） | 优化 | CUDA 线程模型 + LDG.128 + 寄存器堆 | 无直接等价（NPU 无逐元素标量快路径） | **重新设计** | → §0.6 R1：UB 内 v-prefix 向量链（128-bit lane 承载 `npt` 意图） |
| 二维 grid `(列块, 行)`（手段 6） | 算法/优化 | CUDA grid | 有（一维 `T.Kernel(n, is_npu=True)`，grid 维展开） | **重新设计** | → §0.6 R2：`(row, chunk)` 逻辑 task 展平为一维 + persistent 分核（ascend-constraints 一维 Kernel） |
| 连续段 coalesced 访问（手段 2） | 优化 | CUDA coalescing | 有（GM↔UB 连续段 `T.copy`） | **等价替换** | `docs/Tilelang.language/内存操作/T.copy.md` §2.1/§2.4；`examples/elementwise/vec_add_1d.py` L34–L38 |
| 寄存器级驻留计算（手段 7） | 优化 | GPU 寄存器堆 | 有（UB） | **等价替换** | `T.alloc_ub`（v-prefix 要求操作数在 UB，`T.vmul.md` §2.3 第 1 条）；`T.alloc_fragment` 在双张量操作数下被分配到 `<zero>` 空间（`examples/clamp_max/_make_clamp_tensor_kernel/DESIGN.md` §9.2 实测） |
| `threads=` / `num_per_thread` kwarg（手段 1/3） | 优化 | CUDA | 无 | **舍弃** | npuir `T.Kernel` 不接受 `threads=`；wrapper K11 已把 `threads·npt` 折叠为单一 `block_size`，无性能意图丢失 |
| 策略机 `direct` / `explicit_parallel`（手段 9） | 优化 | CUDA | 无 | **舍弃** | wrapper E6：契约只保留 `explicit_parallel` 形态；`direct`（1 元素/线程）在 NPU 无对应收益 |
| `op_func` 标量闭包（`activation_func(gate)·value`） | 优化（API） | 无 | 无（闭包产出标量 TIR 表达式，非向量 op） | **重新设计** | → §0.6 R3：silu 数学**内联**为 v-prefix 链；工厂签名保留 `op_func` 仅作形态标识，不再用它逐元素求值 |
| 工厂闭包 + `lru_cache` 特化（手段 8） | 优化（host） | 无 | — | **保留** | wrapper `forward` 惰性构建 + 按 `block_size` 缓存（`silu_and_mul.py` L157–L160） |

**判定统计**：保留 **4** 项（数学语义 / I/O 契约 / fp32 计算域 / 工厂缓存）、等价替换 **3** 项（`exp2` 降代换、连续段搬运、寄存器驻留→UB）、重新设计 **3** 项（R1 逐元素标量链 / R2 并行与分核 / R3 op_func 内联）、舍弃 **2** 项（`threads/npt` kwarg、策略机）。

### 0.6 NPU 算法重设计

**重设计项 R1: 逐元素标量链 → UB 内 v-prefix 向量链（silu 数学内联，fp32 中转）**

- **源方案**：`T.Parallel(threads, npt)` 内每元素标量链：`g=cast(gate,f32)` → `exp2(-g·log2e)` → `1+·` → `g/·` → `·value` → 回舍（§0.3 步骤 3–5）。性能意图：以「CUDA 多线程 + 每线程 `npt` 个元素 ILP + 128-bit 事务」把逐元素切碎的标量链摊到 GPU 吞吐。
- **NPU 新算法**：整块在 UB 上做 v-prefix 链——`vcast` 升 fp32 后 `vmul/vexp/vadd/vdiv/vmul`，末端 `vcast` 舍回原 dtype（优化后公式见 §1.6.1）。256 个元素的 lane 并行承载源的「满并行」意图；连续段 `T.copy` 承载 128-bit 事务意图。
- **语义保持论证**：**数学恒等**——向量 op 逐 lane 独立，与源的逐元素标量链逐点等价（无累加、无跨 lane 依赖，映射顺序不影响结果）；fp32 计算域与源一致（源亦在 fp32 域算后单次舍回），`exp2` 撤销为 `exp` 经 `verify_equiv.py` 机器验证（15/15 `EQUIV_PASS`，0 违反，含 NaN/±Inf/±0/subnormal 角点）。边界（NaN / `silu(-inf)=NaN` / 大负 `g→0⁻`）与 §0.1 一致。
- **意图承接**：手段 1（满并行 + ILP）→ 向量 128-bit lane；手段 3（`npt` 加宽事务）→ lane 宽度；手段 4（fp32 域）**保留**（PL-1.17 强制）；手段 5（`exp2`）**经论证撤销**（非静默丢弃，§1.6.1）；手段 7（寄存器驻留）→ `T.alloc_ub`。

**重设计项 R2: 并行结构与分核（CUDA 二维 grid×thread → 一维 Kernel + persistent）**

- **源方案**：`T.Kernel(ceildiv(N, block_N), M)` 的 CUDA 两维 grid（`bx` 列块 × `by` 行），block 内 `T.Parallel(threads, npt)`；grid 大小任意（prefill 下 14×2048 = 28672 个 block），超发由 CUDA 调度器吞吐。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`（本项目不支持三维 Kernel，`examples/clamp_max/_make_clamp_tensor_kernel/DESIGN.md` §3.5.1 同款约束）；把源的 `(by, bx)` 两维逻辑任务展平为 `(row, chunk)` 并按 raster 序编号（`task = row·chunks + chunk`），核内 `for i in T.serial(num_local)`（**静态边界**）以 grid-stride 推进 `task = i·num_kernels + cid`，`if task < num_logical` 屏蔽越界。核数按 §5.5 分核。
- **语义保持论证**：逻辑 task 与输出 tile 一一对应、各 task 完全独立（无跨 task 依赖、无竞写），映射顺序 / 分核数 / `block_M` 变化不改变任何输出值（§0.1 无规约）。
- **意图承接**：手段 6（二维 grid）→ 一维 Kernel + `(row, chunk)` 展平（任务映射等价）；`threads/npt` 折叠为 `block_size`（wrapper K11），无性能意图丢失。

**重设计项 R3: `op_func` 标量闭包 → v-prefix 链内联（契约签名保留）**

- **源方案**：工厂接收 `op_func(gate, value)` 标量闭包（`activation_func(gate)·value`），在 `T.Parallel` 内逐元素求值（`_silu_and_mul_fwd_kernels.py` L37）。
- **NPU 新算法**：silu-and-mul 的数学**内联**为 v-prefix 向量链（`op_func` 的标量语义无法映射到向量 op，NPU 无「向量化标量闭包」机制）。工厂签名 `(M, N, dtype, op_func)` 保留以维持 wrapper 契约（`silu_and_mul.py` L121–L126），`op_func` 不再参与 kernel body。
- **语义保持论证**：内联链 = `activation_func(gate)·value` 的向量化等价实现（§1.6.1 优化后公式），逐元素等价；仅当 `op_func` 不是 SiLU 门控时语义会偏差——本算子契约固定为 `SiluAndMulFwdKernel`（wrapper `_effective_op_func` 恒定返回 SiLU），故无影响。**风险登记**：若未来复用到 `gelu_and_mul` 等兄弟算子，须为各自激活重新内联向量链（§9.1）。

**优化意图承接总表**（§0.4 → NPU 方案，无静默丢弃）：

| §0.4 优化手段 | NPU 承接 |
|---------------|----------|
| 1 `T.Parallel(threads, npt)` 满并行 + ILP | Vector 128-bit lane + `block_size` tile（R1/R2） |
| 2 连续段 coalesced | `T.copy` 连续段 GM↔UB（等价替换，同款语义） |
| 3 `npt` 加宽事务 | 向量 lane 宽度（128-bit） |
| 4 fp32 计算域 | **保留（必需）**——`vcast` 中转 + fp32 域链（PL-1.17） |
| 5 `exp2` 降代换 | **撤销**（NPU `vexp`/`vexp2` 均单指令，无收益；§1.6.1 Item 1） |
| 6 二维 grid | 一维 Kernel + `(row, chunk)` 展平（R2） |
| 7 寄存器驻留 | `T.alloc_ub`（等价替换） |
| 8 工厂 + `lru_cache` | wrapper 惰性构建 + 缓存（保留） |
| 9 策略机 | 舍弃（契约只保留 `explicit_parallel`） |
| `op_func` 闭包 | 内联 v-prefix 链（R3） |

### 0.7 标杆实现

- **源算子文件**：`examples/TileOPs/tileops/kernels/elementwise/silu_and_mul/_silu_and_mul_fwd_kernels.py`（Stage 0 提取件，接口契约冻结基准）；GPU 原始上下文 `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（`_make_fused_gated_explicit` @ L646、`FusedGatedKernel` @ L1130、`SiluAndMulFwdKernel` @ L1926）、`/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/activations.py`（`SiluAndMulFwdOp` @ L328）、`/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/_base.py`（`FusedGatedOp` @ L818）。
- **NPU 集成目标（wrapper）**：`examples/TileOPs/tileops/kernels/elementwise/silu_and_mul/silu_and_mul.py`（`SiluAndMulFwdKernel`：工厂 `(M, N, dtype, op_func)` → 可调用 `(block_size)`；`default_config = {"block_size": 1024}`（K11 = `threads·npt`：fp16/bf16 `128·8`、fp32 `256·4`）；`ascend_mode = "Developer"`）；NPU Op 层 `examples/TileOPs/tileops/ops/elementwise/silu_and_mul.py`（`SiluAndMulFwdOp`，`out_idx=[1]` 语义、`.contiguous()` + `eval_roofline`）。
- **规格来源**：NPU manifest `examples/TileOPs/tileops/manifest/elementwise.yaml` `SiluAndMulFwdOp`（4 workloads：`smoke-2m [1024,2048]`、`llama-3.1-8b-swiglu-prefill [2048,28672]`、`llama-3.1-8b-swiglu-decode [1,28672]`、`ffn-square-8m [4096,4096]`；roofline `flops=6·M·N` / `bytes=3·M·N·elem_bytes`）。
- **测试 / 基准**：`examples/TileOPs/tests/ops/test_silu_and_mul.py`（`SiluAndMulTest.ref_program` 为 golden 依据；tolerance fp32 1e-5、fp16 1e-2、bf16 1.6e-2；含 shape/dtype 契约用例）、`examples/TileOPs/benchmarks/ops/bench_silu_and_mul.py`（`torch-ref` 基线 `F.silu(x[..., :half]) * x[..., half:]`）。
- **同族已踩坑参考**：`examples/clamp_max/_make_clamp_tensor_kernel/DESIGN.md`（同族纯 Vector elementwise 迁移基准：§0/§1.6 结构、persistent 分核、`T.alloc_ub` 铁律、`T.alloc_shared`≠UB 陷阱）、`examples/ge/ge_fwd_kernel/DESIGN.md`（同族迁移组织）、`testing/autotune/SiluAndMul/silu_and_mul_2d.py`（**NPU 官方 autotune 先例**：`T.npuir_sigmoid` + `T.vmul` + `T.copy(A[by*bm, bx*bn], A_shared)` 2-D 分块形态）、`testing/autotune/Silu/silu_2d.py`、`testing/npuir/arith_ops/test_sigmoid.py`（`T.npuir_sigmoid` = `T.vsigmoid` 直接先例）、`examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/`（elementwise 多 op 融合迁移同族）。
- golden 以 §0.1 语义为依据（`F.silu(gate_f32)*value_f32` 后回舍），见 §8.1。

---

## 1. 概述

### 1.1 算子名称

`_make_fused_gated_explicit`（Op 层名 `SiluAndMulFwdOp` / 算子族 `silu_and_mul`）

### 1.2 功能描述

融合门控激活前向核：对单个打包张量 `x`（`(M, 2N)`，末维前半为 `gate`、后半为 `value`）计算 `y = silu(gate) * value`，输出 `(M, N)`，dtype 与输入一致（fp16/bf16/fp32）。SwiGLU / MoE FFN 的激活前端（`F.silu(x[..., :N]) * x[..., N:]`）。纯逐元素、无规约、无跨元素依赖。

### 1.3 数学公式

$$
y[m, n] = \operatorname{silu}\big(x[m, n]\big)\cdot x[m, N+n],\qquad
\operatorname{silu}(g) = g\cdot\sigma(g) = \frac{g}{1+e^{-g}},\qquad
m=0..M{-}1,\ n=0..N{-}1
$$

参考 API：`torch.nn.functional.silu(x[..., :N]) * x[..., N:]`。

**公式级优化后形式（§1.6.1）**：`y = (gate·value) / (1 + exp(-gate))`（sub-fp32 dtype 在 fp32 域计算后单次 `rint` 舍回）。

### 1.4 算法描述

**（迁移决策后的 NPU 侧算法；与源算法的结构差异逐条注明来源）**

单遍逐元素融合向量算法（§1.6.0 调研选定基线公式——`silu` 的直接 exp 形式，无更优算法族替代）：

1. **任务分解（编译期）**：输出 `(M, N)` 按 raster 序划为 `(row, chunk)` 逻辑 task；`block_N = block_size`（wrapper 可调用参数，默认 1024），`block_M` 为工厂常量（默认 2），`chunks = ceildiv(N, block_N)`，`num_logical = ceildiv(M, block_M) · chunks`。
2. **核间**：一维 `T.Kernel(num_kernels, is_npu=True)`（persistent，`num_kernels = min(num_logical, 物理 Vector 核数)`，§5.5）。
3. **核内（每 task 一块）**：
   - `T.copy` 载入 gate 块（`x` 列 `[n0, n0+bn)`，行 stride `2N`）与 value 块（`x` 列 `[N+n0, N+n0+bn)`）；
   - sub-fp32 dtype：`vcast(→f32, rint)` 升域；fp32 dtype：直接计算；
   - `p = vmul(g32, v32)`；`e = vexp(vmul(g32, -1.0))`；`d = vadd(e, 1.0)`；`y32 = vdiv(p, d)`；
   - `vcast(→原 dtype, rint)` 舍回；`T.copy` 写回。
4. **尾块**：`n_tail = T.min(block_N, N - n0)`，`T.copy` 用 runtime 边界切片；向量 op 对整 buffer 计算，仅 `[0:n_tail]` 写回（垃圾不落 GM）。

**与源算法的结构差异**：① `T.Parallel(threads, npt)` 逐元素标量链 → **UB 内 v-prefix 向量链**（§0.6 R1）；② CUDA 二维 grid×thread → **一维 Kernel + persistent 分核**（§0.6 R2）；③ **撤销 GPU 专属 `exp2(-g·log2e)` 降代换**（§1.6.1 Item 1）；④ `op_func` 标量闭包 → **内联向量链**（§0.6 R3）；⑤ 保留源的 fp32 计算域（PL-1.17 强制）。

### 1.5 数据流图

```
任务分解：输出 (M, N) → block_M × block_N raster tile → task=(row-block, chunk) → 一维 num_logical

sub-fp32 (fp16/bf16) 路径（每 task）：
  GM[x] ─T.copy(gate 段: 列 [n0, n0+bn))─> UB[g_ub(dtype)] ─vcast(rint)→ UB[g32(f32)] ┐
  GM[x] ─T.copy(value 段: 列 [N+n0, N+n0+bn))─> UB[v_ub(dtype)] ─vcast(rint)→ UB[v32(f32)] ┤
                                                                                          ├─vmul→ UB[p(f32)]
   UB[g32] ─vmul(·,-1)→ UB[ng] ─vexp→ UB[e] ─vadd(+1)→ UB[d] ─────────────────────────────┤
                                                                                          ├─vdiv(p/d)→ UB[y32(f32)] ─vcast(rint)→ UB[y_ub(dtype)] ─T.copy─> GM[y]
fp32 路径同构，无两级 vcast（直接 fp32 域链）
全程 UB 内计算，无 L1/L0，无中间 GM 往返（GM 流量 = 3·M·N·E 下界）
```

### 1.6 算法调研与优化分析 ⭐

> 设计第一优先级：先调研（1.6.0，迁移任务在 Phase M0 后、M1 前完成，结论已驱动 §0.5/§0.6），再公式级优化（1.6.1），再循环/标量的向量化替代（1.6.2），最后向量化轴与布局决策（1.6.3）。本节分析对象 = §1.6.0 选定、经 §0.6 重设计落地的 NPU 算法。`silu_and_mul` 为**单步逐元素融合类**（无规约 / 无窗口 / 无矩阵），按 algorithm-research.md §2 属**单步逐元素**族；但含超越函数（exp）与除法，故 R1 做**候选表级**调研（不止一行结论），R2–R4 按单步族给明确结论。

#### 1.6.0 算法调研（Algorithm Research）⭐

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判。算子维度为「每输出元素的标量向量 op 数」，`silu` 本体与 `·value` 合并计）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线（源算法）** | `y = (g/(1+exp2(-g·log2e)))·v`，fp32 域 | — | — | ✅（基线） |
| 1 | **exp 直式 + 分子合并** | `y = (g·v)/(1+exp(-g))`，fp32 域（`vexp`） | **数学恒等**（`exp2(x·log2e)≡exp(x)`；乘法结合性）；舍入路径差异见 §1.6.1（机器验证 0 违反） | **正向**：撤销 GPU 专属 `log2e` 乘（等代换）；相对 `vsigmoid` 组合路省 1 op | ✅ **采纳**（§1.6.1 Item 1） |
| 2 | `vsigmoid` 组合 | `y = (g·vsigmoid(g))·v` | 数学恒等 | **负收益**：`vsigmoid` 展开为 `VAdd+VSub+VExp+VDiv` 4 op（`docs/Tilelang.language/数学操作/T.sigmoid.md` §3），加 2 次 `vmul` = **6 op** > 候选 1 的 5 op | ✅（对比后弃选） |
| 3 | 正指数形（免取负） | `y = (g·v)·e/(1+e)`，`e = vexp(g)` | 数学恒等（`g/(1+e^{-g}) = g·e/(1+e)`） | **持平**：仍为 5 op（`vexp`+`vadd`+2`vmul`+`vdiv`），只是把「取负乘」换成「分子乘 e」 | ✅（对比后弃选：无收益） |
| 4 | tanh 形 | `y = 0.5·g·(1+tanh(g/2))·v` | 数学恒等（`tanh(g/2)=2σ(g)-1`） | **持平**：`vmul`+`vtanh`+`vadd`+2`vmul` = 5 op；但 golden 基于 `sigmoid`/`exp`，tanh 路多一层代数变形（精度对齐更远） | ✅（对比后弃选） |
| 5 | `vrec` 倒数形 | `y = (g·v)·vrec(1+exp(-g))` | 数学恒等 | **负收益**：`vrec`+`vmul` 2 op 替代 `vdiv` 1 op → 6 op | ✅（对比后弃选） |
| 6 | Hardswish / 分段近似 | `g·clamp(g+3,0,6)/6` 等 | **非等价**（不同函数，偏离 `silu` 远超 tol） | 语义被改，禁止 | ❌（语义不等价） |
| 7 | 位/查表 sigmoid 近似 | int 位技巧 / 分段多项式 | **容差内不可控**（近似误差随定义域分布，无法保证 1e-2 全区间）；且引入整型路径 | 语义风险 + op 数不降 | ❌（语义风险） |
| 8 | 原生 dtype（fp16）计算 | 省两级 `vcast` | **非等价**：fp16 域 `vexp`/`vdiv` 逐步舍入 + bf16 算术无原生支持（PL-1.17）；golden 经 fp32 | **负收益**：精度不可控；bf16 直接不可行 | ❌（dtype 不可行） |

**R2 在线算法**：**无在线变体需求（天然单遍）**。结构依据：`y[m, n]` 仅依赖 `(x[m, n], x[m, N+n])`，**无 running 统计量**（无和 / 最大值 / 计数 / 在线 softmax 类状态）、无跨元素依赖——逐元素映射本身即单遍流式（algorithm-research.md §3 R2 结构判据「逐元素映射无跨元素依赖，单遍即天然在线」）；源算法亦为单遍、无 online 手段（§0.4）。收益口径（扫描遍数=1、中间缓冲 O(tile)、UB 驻留）在所有候选间无差异。

**R3 复杂度对比**（带宽受限算子——算术强度 ≈ 6 FLOP ÷ 6 B = 1 FLOP/B，**访存为主导项**；口径：`M·N` 为输出元素数、`E = elem_bytes`；op 数按「每输出元素标量 op」与「每 tile 向量 op」双列；sub-fp32 的 `vcast` 另计）：

| 算法候选 | FLOPs / 向量 op 数（口径） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|---------------------------|---------------|---------|-------------|---------------------|
| **基线（源：exp2 + fp32 域，T.Parallel 标量链）** | 标量链 8 op/元素（2 cast + vmul(log2e) + exp2 + vadd + vdiv + vmul·v + cast↓）；逐条向量化 = **8 向量 op/tile** | 3MN·E（gate+value 读 + y 写） | 1 | O(block_N·npt) 寄存器 | N·M/block_N 独立线程；标量链为热点 |
| **候选 1（exp 直式 + 分子合并，fp32 域）** | **5 向量 op/tile**（vmul(-1) + vexp + vadd + vmul(g·v) + vdiv） | 3MN·E（同基线） | 1 | O(block_M·block_N)（UB，§4.5） | num_logical 独立 task，零跨核同步；persistent 摊薄启动 |
| 候选 1 + fp32 输入（无中转） | 5 向量 op/tile（省 3 `vcast`） | 3MN·E | 1 | 同 | 同 |
| **候选 1 + sub-fp32 中转（**采纳实现**）** | **8 向量 op/tile**（3 `vcast` + 5） | 3MN·E | 1 | O(block_M·block_N)（fp16 中转 ≈20 B/elem、bf16 ≈24 B/elem，PL-1.17） | 同 |
| 候选 2（`vsigmoid` 组合） | 6+3 = **9 向量 op/tile** | 3MN·E | 1 | 同 | 同（每 tile 多 1 op） |
| 候选 3（正指数形） | 5+3 = **8**（同候选 1 实现，无收益） | 3MN·E | 1 | 同 | 同 |
| 候选 4（tanh 形） | 5+3 = **8** | 3MN·E | 1 | 同 | 同 |
| 候选 5（`vrec` 倒数） | 6+3 = **9** | 3MN·E | 1 | 同 | 同 |

**op 数下界分析**：`silu(g)·v = g·v/(1+e^{-g})` 的最短链 = 1 条超越函数（`vexp`）+ 1 条 `vadd` + 1 条 `vdiv` + 2 条 `vmul`（`g·v` 与取负 `·(-1)`）= **5 条向量 op**，为纯向量实现的 op 数下界——候选 2–5 均 ≥5（含 2 条 vmul 的变体）或需要额外取负 / 倒数，无更低者；`vneg` 原语在本项目不存在（`tilelang/language/__init__.py` L72–L176 的 v-prefix 清单无 `vneg`），故「取负」必须占 1 条 `vmul`。访存量 3MN·E 为输入读 + 输出写的下界，所有候选相同。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附 `docs/` / `examples/` / pattern-library 佐证）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 可融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（源逐元素标量 cast/exp2/div/mul 链） | ❌ 逐元素标量链落错单元（NPU 无逐元素标量快路径；须逐条向量化后 op 数 ×8） | OK | 标量链无向量 lane | 尾块依赖动态 shape | 单阶段 | ❌ 淘汰：改为向量链即回到候选 1 |
| **候选 1（exp 直式 + 分子合并，fp32 域）** | ✅ 全落 Vector：`vexp`/`vdiv`/`vmul`/`vadd`（fp32 原生，`T.vexp.md` §2.2.1 fp32 √） | ✅ tile ≤4096 元素（fp16 中转 20 B/elem ×1.7 ≈ 139KB < 192KB，§4.5） | ✅ `block_size·E` 32B 整倍数；fp32×4 / fp16,bf16×8 整除，尾 lane 0% | ✅ N/block_size 全编译期特化，仅 1 个运行时 `n_tail` | 单阶段无流水；纯 Vector 无 CV 融合 | ✅ **选定** |
| 候选 2（`vsigmoid`） | ✅ Vector（但 `vsigmoid` 展开 4 op + 2 vmul = 6，比候选 1 多 1 op） | OK | OK | OK | OK | ❌ 弃选：op 数 6 > 5（`T.sigmoid.md` §3 展开证据） |
| 候选 3（正指数形） | ✅ Vector（`vexp` + 额外 1 vmul） | OK | OK | OK | OK | ❌ 弃选：op 数持平，无收益 |
| 候选 4（tanh 形） | ✅ Vector（`vtanh` 单指令） | OK | OK | OK | OK | ❌ 弃选：op 数持平，且与金标 exp 路径精度对齐更远 |
| 候选 5（`vrec` 倒数） | ✅ Vector（`vrec` 未见于 `docs/Tilelang.language/`——仅 `tilelang/language/customize_npuir.py` 导出，属未文档化） | OK | OK | OK | OK | ❌ 弃选：op 数 6 > 5；API 未文档化（未文档化假设：勿依赖） |
| 候选 6–7（近似 / 位技巧） | Vector 或整型路径 | OK | OK | OK | OK | ❌ 弃选：**语义不等价**（hardswish 等偏离 `silu` 远超 1e-2；近似误差无法全区间约束） |

**调研结论**：选定 **候选 1（`y = (g·v)/(1+exp(-g))`，fp32 域，sub-fp32 dtype 经 `vcast` 中转）**，即 §0.6 R1 重设计落地的 NPU 算法。关键依据：R3 中候选 1 的 5 op 为纯向量实现下界（候选 2/5 为 6 op、候选 3/4 持平、候选 6/7 语义不等价），访存 3MN·E 为所有候选共同下界；R4 中仅候选 1 对计算单元（全 Vector、fp32 原生）、UB 容量、对齐整除、静态边界全部亲和，且 bf16 由 PL-1.17 强制走 fp32。与基线的**结构差异**：① 逐元素标量链 → 一条条向量 op；② GPU 专属 `exp2(-g·log2e)` 降代换撤销为 `exp`；③ CUDA 二维 grid×thread → 一维 Kernel + persistent 分核；④ `vsigmoid` 组合路（6 op）被排除，改用 exp 直式（5 op）。

**「无更优替代 / 无更优公式」的调研范围**：① algorithm-research.md §5 参考表命中行 `ALG-elementwise`（R1：公共子表达式消除——`g`/`v` 各用一次，无公共子式；除法转乘倒数——`vdiv` 单指令优于 `vrec+vmul` 2 op；降代换——`exp2` 在 NPU 无收益，已在 §1.6.1 论证；R2 逐元素天然单遍）；② 恒等变形自查（候选 2–5：`vsigmoid` 展开 / 正指数形 / tanh 形 / 倒数形，均 op 数 ≥5 或持平，见 R1/R3）；③ 本仓 `examples/elementwise/vec_add_1d.py`、`examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/`（mish = `x·tanh(softplus)` 多 op 融合）、`testing/autotune/Silu/silu_2d.py`、`testing/autotune/SiluAndMul/silu_and_mul_2d.py`（**NPU 官方先例用 `npuir_sigmoid`+`vmul` 组合**——本设计经 op 数分析排除该路）、`testing/npuir/arith_ops/test_sigmoid.py`、`examples/deepseek_v4/engram/engram_gate_kernel_fwd.py`（`T.vsigmoid` 直接用例）——均为直排 v-prefix 单遍结构，无 `silu` 公式变形先例；④ pattern-library `elementwise.md`（PL-1.6 copy-floor / PL-1.10 loads-first / PL-1.17 fp32 中转 / PL-1.14 / PL-1.19——均访存 / 布局调度类）+ `constants.md`（发射 / 带宽常数），无 `silu` 公式级条目；⑤ 源算子结构分析（§0.3：唯一可优化点是 `exp2` 降代换，已在 §1.6.1 承接）；⑥ 互联网检索**未执行**（环境 `webfetch` 未验证可用；且本算子族（SiLU/SwiGLU）公式已收敛为 `g·σ(g)`，R1 候选空间经本地恒等变形已穷尽，无检索增益——如实记录，不编造来源）。源算法优化手段的意图承接见 §0.6 末尾总表，无静默丢弃。

**设计期 roofline 估算（D-2）**（估算下界 = max(流量项, 发射项, 容量项)；常数引用 `pattern-library/constants.md`；本机实测 `NPUUtils.get().get_aicore_num() = 20`（查询路径 `tilelang.utils.npu_utils.NPUUtils`，2026-10-14 实查）⇒ 纯 Vector 核数 = `get_aivector_core_num()` = **40**，与 `CONST-aicore-910B2C`（910B2C=24/48）不同源，流量项按 40/48 折算并标注；`block_size = 1024`（wrapper 默认），`block_M = 2`）：

| 代表 workload | 流量项（3MN·E ÷ 40/48 折算带宽） | 发射项（num_local × op 数 × 0.5µs，**保守上界**） | 容量项 | 估算下界 |
|---------------|--------------------------------|--------------------------------------------------|--------|----------|
| smoke `[1024,2048]` fp16（N=1.05M，num_logical=512, num_local=13） | 6.29MB ÷ 1.05 TB/s ≈ **6.0µs** | 13 × 8 × 0.5µs ≈ **52µs** | 豁免（轻量 tile 级，无跨引擎往返） | **≈52µs**（小 shape：发射可能暴露） |
| prefill `[2048,28672]` fp16（N=29.4M，num_logical=1024·14=14336, num_local=359） | 176.2MB ÷ 1.05 TB/s ≈ **167.8µs** | 359 × 8 × 0.5µs ≈ **1436µs** | 豁免 | **≈168µs**（流量主导；见下注） |
| ffn-square `[4096,4096]` fp16（N=8.39M，num_logical=2048·2=4096, num_local=103） | 50.3MB ÷ 1.05 TB/s ≈ **47.9µs** | 103 × 8 × 0.5µs ≈ **412µs** | 豁免 | **≈48µs**（流量主导） |
| decode `[1,28672]` bf16（N=14336，num_logical=1·14=14, num_local=1） | 86KB ÷ 1.05 TB/s ≈ **0.08µs** | 1 × 8 × 0.5µs ≈ **4µs** | 豁免 | **≈5µs**（延迟主导，单波） |

> 引用常数：`CONST-mte2-degradation`（MTE2 有效带宽随总流量退化：1M 33.9 → 16M 30.2 → 64M 20.9 → 256M 18.4 GB/s/core；256M 混合 R/W 全流量地板 ~1.26 TB/s @48 核，toolchain tilelang 0.1.2+ed787bb/910B2C/CANN 8.5.0）——流量项按 40/48 折算（≈1.05 TB/s），**标注为跨核数外推的未实测假设**；`CONST-vector-launch-overhead`（~0.5µs/op，toolchain 0.1.2+3a214cde/910B2C，与本次不同源——**跨版本外推的未实测假设**）；`CONST-copy-floor-method`（**关键反证**：lerp_tensor 7-pass fp32-transit 链在 ≥16M 档 auto-multi-buffer 全隐藏于 MTE2 窗口，delta ≤1.1µs）——故上表**发射项为不重叠上界，非可信下界**；本算子每 tile 8 条向量 op（含 3 `vcast`），与 lerp 的 7-pass 同量级，**≥16M 档（prefill/ffn）应以流量项为设计下界**，发射项仅在 <16M 档（smoke）可能暴露。容量项按轻量调研豁免（纯 Vector，tile 级缓冲无跨引擎往返）。**注**：上表「估算下界」列对 prefill/ffn 取流量项、对 smoke/decode 取发射项，为该档主导项；`block_size` 增大可摊薄发射项（`num_local ∝ 1/block_size`），Stage 4 扫（§5.4）。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的候选 1。逐类别核查（**采纳 1 项**，含四要素 + 机器验证）：

| # | 优化类别 | 核查结论 | 依据 |
|---|----------|----------|------|
| 1 | 恒等变形降代价（除法转乘倒数 / sqrt→rsqrt / 换底） | **部分采纳（并入 Item 1）**：`exp2(-g·log2e)` 换底为 `exp(-g)`；`vdiv` 保留（`vrec+vmul` 为 2 op，劣于 1 op） | §1.6.0 R1 候选 5 / R3 |
| 2 | 稳定化变形（减 max / logsumexp 改写） | **无需**：`silu` 无溢出放大结构——大正 `g` 时 `exp(-g)→0`、`d→1`；大负 `g` 时 `exp(-g)→inf`、`g/inf→0`（= 正确的 `silu→0⁻`）。无中间量溢出 | §0.1 边界语义 |
| 3 | 公共子表达式消除 | **无公共子式**：`g32` 用 2 次（`vmul(-1)` 与 `vmul(g·v)`）但均为叶值，CSE 无收益；`e`/`d` 各用 1 次 | §1.6.0 R3 |
| 4 | 算子降代换 | **采纳 Item 1**：`exp2`+`log2e` 乘 → `exp` 直式；同时以 exp 直式替代 `vsigmoid` 组合（6→5 op） | 见下 |
| 5 | 访存量削减 | **本体内达下界** 3MN·E（gate/value 各读一次 + y 写一次）；无中间 GM 往返（全 UB）；无重复读取 | §1.6.0 R3 |
| 6 | 归约结构优化 | **无归约** | §0.1 |
| 7 | 分子合并（`s=silu(g); y=s·v` → `y=(g·v)/d`） | **并入 Item 1**：op 数持平（5），合并后缩短 `g32` 活区间（可提前复用其 buffer） | §1.6.0 R1 候选 1 |

**采纳项 1：撤销 GPU 专属 `exp2(-g·log2e)` 降代换 + 采用 exp 直式（分子合并）**

- **原式**（GPU 源，`activation_func` + `fused_op`）：`g32 = cast(gate, f32)`；`s = g32 / (1 + exp2(-g32·log2e))`；`y = cast(s · value, dtype)`。
- **优化后公式**（NPU）：`g32 = vcast(gate, f32, rint)`；`v32 = vcast(value, f32, rint)`；`p = g32·v32`；`e = vexp(g32·(-1))`；`d = e + 1`；`y32 = p / d`；`y = vcast(y32, dtype, rint)`。
- **等价性论证**：
  - **`exp2` 撤销**：`exp2(x·log2e) ≡ exp(x)` 为数学恒等（`log2e = 1/ln2`）；GPU 侧用 `exp2` 是因 CUDA `expf` 为多 op 而 `exp2f` 为 1 条 `MUFU.EX2`；NPU 上 `vexp` 与 `vexp2` **同为单向量指令**（`docs/Tilelang.language/数学操作/T.vexp.md` §3 → `hivm.hir.vexp`；`T.vexp2.md` §3 → `hivm.hir.vexp2`），故降代换无性能收益，反而引入 `log2e` 的一次舍入。撤销后 op 数不变（均 5 条），数值路径为数学直式。
  - **分子合并**：`(g/(1+e^{-g}))·v ≡ (g·v)/(1+e^{-g})` 为乘法结合律的实数恒等。
  - **容差内等价说明**：`exp2` 撤销改变的是 `log2e` 舍入（sub-ulp 级），fp16/bf16 舍回后**逐位一致**（见机器验证）；分子合并改变的是 fp32 域乘除次序，回舍后差异 ≤1 ulp（fp16/bf16 由 dtype 精度主导）。`verify_equiv.py` 机器验证（E1/E2）在 fp64 参照下 **0 违反**。
  - **相对 `vsigmoid` 组合路的 op 数**：`y = g·vsigmoid(g)·v` = `vsigmoid` 展开 4 op（`T.sigmoid.md` §3：`VAdd+VSub+VExp+VDiv`）+ 2 `vmul` = **6 op**；本采纳式为 **5 op**——排除 `vsigmoid` 组合路是本项的实现收益。
- **收益量化**：相对**源 GPU 算法**：向量 op 数 8 → 8（sub-fp32 中转下均含 3 `vcast`；fp32 输入下 5），**无 op 数净收益**（`exp2` 撤销与分子合并均不改变 op 数）——收益为**数值保真（消除 `log2e` 舍入源）+ 代码简化（少一个常量操作数）**，经机器验证确认等价。相对 **NPU 官方 autotune 先例**（`npuir_sigmoid`+`2×vmul`，`testing/autotune/SiluAndMul/silu_and_mul_2d.py` L116–L118）：**6 → 5 向量 op**（每 tile 省 1 条，约 −17% 向量链）。访存量不变（3MN·E）。

**无进一步 op 数优化（显式结论与依据）**：纯向量实现的 op 数下界为 **5**（1 `vexp` + 1 `vadd` + 1 `vdiv` + 2 `vmul`；`vneg` 原语不存在于 v-prefix 清单，取负须占 1 条 `vmul`）——候选 3/4 持平（同 5），候选 2/5 为 6，候选 6/7 语义不等价（§1.6.0 R1/R3/R4）；访存量 3MN·E 为读 2N + 写 N 的下界，无中间 GM 往返可削减；无公共子表达式、无归约、无多轮扫描（单遍）。故**除 Item 1 外无进一步数学等价优化空间**。

**等价性机器验证（D-1）** — `examples/silu_and_mul/_make_fused_gated_explicit/verify_equiv.py`（torch CPU，fp64 参照 + 目标 dtype 舍入对照；角点含 ±0/±1/±2/±8/±10/±11/±20/±40/±88/±89/±100/±Inf/NaN/finfo.max/tiny/smallest_normal + 4096 随机点；本次执行于 2026-10-14，本机 aicore 20）：

| 项 | 对照对象 | 用例 | 违反数 | 最大 ulp | 结论 |
|----|----------|------|--------|----------|------|
| **E1-{fp16,bf16,fp32}-cand(exp)-vs-golden** | 采纳式 `g/(1+exp(-g))` vs fp64 `g·sigmoid(g)`（回舍目标 dtype） | 各 4120 点 | 0 / 0 / 0 | 0.00 | `EQUIV_PASS` |
| **E1-{fp16,bf16,fp32}-base(exp2)-vs-golden** | 源式 `g/(1+exp2(-g·log2e))` vs fp64 金标 | 各 4120 点 | 0 / 0 / 0 | 0.00 | `EQUIV_PASS` |
| **E1-{fp16,bf16,fp32}-cand-vs-base** | 采纳式 vs 源式（含 NaN/Inf 模式） | 各 4120 点 | 0 / 0 / 0 | **0.00**（回舍后逐位一致） | `EQUIV_PASS` |
| **E2-{fp16,bf16,fp32}-fused-cand-vs-golden** | 全融合 `silu_exp(g)·v` vs `silu_ref(g)·v` | 各 4120 点 | 0 / 0 / 0 | 0.00 | `EQUIV_PASS` |
| **E2-{fp16,bf16,fp32}-fused-base-vs-golden** | 源式全融合 vs 金标 | 各 4120 点 | 0 / 0 / 0 | 0.00 | `EQUIV_PASS` |

**OVERALL: `EQUIV_PASS`（15/15）**（脚本随算子目录留存，Stage 2 维度 8 可重跑）。结论：`exp2(-g·log2e)` → `exp(-g)` 的撤销在 fp16/bf16/fp32 回舍后**逐位一致**（`log2e` 舍入低于目标 dtype 精度），分子合并同样逐位一致；NaN / ±Inf 模式匹配（`silu(-inf) = NaN` 两式一致）。§3.1 公式拆解以「优化后公式 = `y = (g·v)/(1+exp(-g))`（fp32 域）」为唯一输入。

> **采纳项对源优化手段的影响**：源手段 5（`exp2` 降代换）经上述论证**撤销**（非静默丢弃，§0.6 承接总表已登记）；其余手段（满并行、coalesced、`npt` 加宽、fp32 域、二维 grid、寄存器驻留、工厂缓存）意图由 §0.6 R1/R2 承接。

**优化结论**：本设计**采纳 1 项**数学等价优化——**撤销 GPU 专属 `exp2(-g·log2e)` 降代换并采用 exp 直式（分子合并）**（采纳项 1）。满足四要素：数学恒等（`exp2(x·log2e)≡exp(x)`、乘法结合律）成立，经 `verify_equiv.py` **机器验证**（E1/E2，15/15 `EQUIV_PASS`，回舍后逐位一致）。**收益**：相对 NPU 官方 `vsigmoid` 组合先例 **6 → 5 向量 op/tile**；相对源算法 op 数持平（8，sub-fp32 中转下）但消除 `log2e` 舍入源；访存量不变。**优化后公式 = `y = (g·v)/(1+exp(-g))`（fp32 域，sub-fp32 经 `vcast` 中转）**，为 §3.1 公式拆解唯一输入。**未采纳**候选 2（`vsigmoid` 组合，6 op）/ 3（正指数形，持平）/ 4（tanh 形，持平）/ 5（`vrec`，6 op）/ 6–7（近似，语义不等价），依据见 §1.6.0 R1/R3/R4；其余类别（稳定化 / 公共子式 / 访存 / 归约 / 多轮扫描）经核查**无进一步可采纳项**。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案（§0.6 重设计后的 NPU 算法）中**全部**循环与标量计算点：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | gate/value 切片读取 | 源 `gate=x[by,col]` / `value=x[by,N+col]` 标量索引逐元素 | `T.copy` 连续/跨行段（GM→UB） | ✅ | —（`docs/Tilelang.language/内存操作/T.copy.md` §2.1/§2.4；`testing/autotune/SiluAndMul/silu_and_mul_2d.py` L114–L115 先例） |
| 2 | fp32 升域 cast（gate/value） | 源 `T.Cast("float32", x)` 逐元素标量 | `T.vcast(g_ub, g32, round_mode="rint")` | ✅ | —（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1：f16→f32 / bf16→f32 均 `rint`；PL-1.17） |
| 3 | 取负 `-g32` | 源 `exp2(-g·log2e)` 内含标量取负 | `T.vmul(g32, -1.0, ng)`（标量 src2） | ✅ | —（`docs/Tilelang.language/数学操作/T.vmul.md` §2.3 第 4 条「仅支持 src2 为标量」；`vneg` 原语不存在） |
| 4 | 指数 `exp(-g)` | 源 `T.exp2(-g·log2e)` 逐元素 | `T.vexp(ng, e)` | ✅ | —（`docs/Tilelang.language/数学操作/T.vexp.md` §2.4 示例即整 buffer 向量 `vexp`；fp32 √） |
| 5 | `1 + e` | 源标量 `one + ...` | `T.vadd(e, 1.0, d)`（标量 src1） | ✅ | —（`docs/Tilelang.language/数学操作/T.vadd.md` §2.3 第 4 条「仅支持 src1 为标量」） |
| 6 | 分子 `g·v` | 源 `s * value` 标量乘 | `T.vmul(g32, v32, p)` | ✅ | —（`T.vmul.md` §2.4 同形 2D 示例） |
| 7 | 除法 `p/d` | 源 `s/den` 标量除 | `T.vdiv(p, d, y32)` | ✅ | —（`docs/Tilelang.language/数学操作/T.vdiv.md` §2.4 示例） |
| 8 | 回舍原 dtype | 源 `y[...] = ...` 隐式截断 | `T.vcast(y32, y_ub, round_mode="rint")` | ✅ | —（`T.vcast.md` §2.2.1：f32→f16 / f32→bf16 均支持 `rint`） |
| 9 | task 分解与索引（`row = task // chunks`、`chunk = task % chunks`、`n0 = chunk·block_N`、`x[m0, ...]` / `x[m0, N + n0]` 基址） | 源 `col=(bx·t+i)·npt+j` 逐元素索引 | **无**（block 级标量） | ❌ | **block 级索引 / 任务映射计算**：每 task 仅 O(1) 次乘除 + O(1) 次加（编译期常量因子），非逐元素热点；无逐元素等价 API（向量 op 不承载标量基址计算） |
| 10 | 尾块长度 `n_tail = T.min(block_N, N - n0)` | 源无显式尾块（依赖 `LegalizeSafeMemoryAccess`） | **无** | ❌ | **依赖动态 shape 的边界处理**：非整除时逐 task 边界不同；沿用 `examples/elementwise/vec_add_1d.py` L31–L33 的已验证 `T.min` 模式（`docs/Tilelang.language/比较操作/T.min.md`）；padding 违反 host 禁改输入约束 |
| 11 | 核内 task 调度循环 `for i in T.serial(num_local)` | 循环 | **无**（persistent 结构需循环推进 task） | ❌ | **tile 级调度循环**（§5.5 persistent 分核标准结构，core-split-strategy.md §1 要素③）；边界为编译期静态值，循环体一次一个完整 tile（内部全向量化） |
| 12 | host 侧 `chunks / num_logical / num_kernels / num_local / block_M` 计算 | Python 标量 | — | ❌ | **host 元数据计算**（不在 kernel 内；工厂闭包编译期完成，符合 decision-tree §1） |

**向量化结论**：全部逐元素计算点（#1 切片读取、#2 升域、#3 取负、#4 指数、#5 加一、#6 分子乘、#7 除法、#8 回舍）**均已替代为向量 op**——所用 API（`T.copy`/`T.vcast`/`T.vmul`/`T.vexp`/`T.vadd`/`T.vdiv`）全部有 `docs/Tilelang.language/` 佐证且均为 fp32 原生支持；替代方案即 §1.6.1 优化后公式的逐 op 映射。保留 4 处标量 / 循环（#9 任务索引、#10 动态尾块、#11 tile 调度循环、#12 host 元数据），逐项理由见上表——§6 循环结构与本结论一致（kernel 内**无任何逐元素标量循环**）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（packed `(M, 2N)` 进、`(M, N)` 出，均行主序连续），但**核内 tile 形状与 lane 映射是自由变量**。本节枚举候选并量化取舍。迁移任务：源 GPU 的 `threads×npt` 轴与二维 grid 是**硬件组织的输入而非结论**，须独立评估。**本算子特有约束**：gate 与 value 是同一行的两个**连续半段**（列 stride=1，段间 offset = `N`），非交织，无需 deinterleave / 重排。

**轴质量评分维度**（候选共用的向量化轴 = 输出 N 轴，段内 stride=1）：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | `block_N` 对向量宽度（fp16/bf16 ×8、fp32 ×4）整除 | `block_N ∈ {1024, 2048}` 均整除 ✓ |
| 尾 lane 浪费率 | `(ceil(len/vw)·vw − len)/len` | 仅最后一 chunk 有 `(block_N − n_tail)` 垃圾 lane，不落 GM，浪费率 ≈0 |
| 累加链形态 | 该轴为纯向量维（stride=1），无串行累加维 | 无阻碍（逐元素无累加） |
| repack 代价 | 改变核内 tile 形状所需重排 | 两种候选均 **无 repack**（直接 `T.copy` 从 I/O 原生布局取段） |
| UB 容量影响 | tile 尺寸 → 驻留缓冲 | `block_M·block_N·(中转账 B/elem) ≤ 192KB/1.7`（§4.5） |

**候选矩阵**（全部枚举，弃选必须给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | **I/O 原生 packed + raster 2-D tile `(block_M, block_N)`**，`block_N=block_size`（默认 1024）、`block_M=2` | 输出 N 轴（段内 stride=1） | 无 | 基线：gate 段 `[n0,n0+bn)`、value 段 `[N+n0,N+n0+bn)` 各为 `block_M` 条 `bn·E` 连续 burst（行 stride `2N`）；task 数 ÷`block_M`（发射项摊薄，对 <16M 档有益）；UB `2·1024·20 B×1.7 ≈ 68KB` ✓ | ✅ **采纳（主选）** |
| 2 | I/O 原生 packed + 逐行连续 chunk（`block_M=1`） | 输出 N 轴 | 无 | 备选：每 task 恰 1 条 gate burst + 1 条 value burst（最大连续度，`bn·E` 单段）；UB 最省（`1024·20×1.7 ≈ 34KB`）；task 数 ×`block_M`（prefill 28672 vs 14336）→ 发射项 ×2 | ⚠️ **备选**（见裁决计划） |
| 3 | host 侧把 `x` 预拆为 `gate`/`value` 两块连续张量（`x[:, :N].contiguous()` + `x[:, N:].contiguous()`） | 输出 N 轴 | host `.contiguous()` 全量拷贝 | ❌ **负收益**：host 物化 `+2MN·E` 写 + 读（prefill +117MB），违反 ascend-constraints §4（host 禁改输入真实内容），且 `T.copy(x[m0, N+n0], v_ub)` 已能直接取跨段行——无必要 | ❌ |
| 4 | 核内 UB 重排（`T.deinterleave` / `T.transpose`）后再算 | — | 核内转置链 | ❌ **负收益**：gate/value 本非交织，deinterleave 无意义；`T.transpose` 引入 `VTransposeOp` 开销 + UB 驻留（`T.transpose.md`：permutation 限两轴交换、必须显式传 permutation），零收益 | ❌ |
| 5 | 三维/多维 tile 展开（`(block_M, block_N)` 再分形） | — | 无 | ❌ 无收益：逐元素无分形约束；NPU 一维 Kernel 已由 raster 展平覆盖 | ❌ |
| 6 | 逐元素标量 `T.Parallel` 直译（源形态） | 无向量轴 | 无 | ❌ **负收益**：NPU 无逐元素标量快路径（落 Scala，无 128-bit lane）；§1.6.2 #1–#8 已判向量化 | ❌ |

**布局决策结论**：选定方案 **#1**——核内布局 = packed `(M, 2N)` 上按 raster 序的 2-D tile `(block_M, block_N)`（`block_M=2` 默认、`block_N = block_size`），向量化轴 = **输出 N 轴（段内 stride=1）**，repack = 无；gate/value 段各自用 `T.copy` 从 GM 原生布局直接取（value 段列偏移 `N`）。该结论同步落入 §3.3 伪代码（buffer 形状 `(block_M, block_N)`、`T.copy` 操作数即段）与 §6 循环结构；§4.5 UB 预算按此布局计算。方案 #2（`block_M=1`）作为备选 + 裁决计划随设计冻结进 Stage 4。

**源码轴对照（迁移任务要求）**：源 `col=(bx·threads+i)·npt+j` 的隐式轴向是「输出 N 轴 + 每线程 `npt` 连续元素」，`by` 为行轴。主选 #1 的「输出 N 轴向量化 + `block_M` 行」与之同轴（N 轴连续、行轴独立），仅把「行轴由 grid 承载」改为「行轴并入 tile 的 `block_M` 维 + raster 展平」——轴选择无需额外重设计。

**实验裁决模式（主选 #1 vs 备选 #2）**：两候选的判定裕度依赖**未实证常数**（MTE2 跨行 stride 访问的段合并效率 vs 每 task 固定开销的暴露量），故按 §1.6.3 实验裁决模式产出三件套：

- **主选**：方案 #1（2-D tile `(block_M=2, block_N=1024)`，进 Stage 3）。
- **备选**：方案 #2（`block_M=1` 逐行连续 chunk）。结构完整可实现：buffer 形状 `(1, block_N)`（或 `(block_N,)`），`T.copy(x[m0 : m0+1, n0 : n0+bn], g_ub)` / `T.copy(x[m0 : m0+1, N+n0 : N+n0+bn], v_ub)` 单段连续载入；分核三要素按 `block_M=1` 粒度重算（`num_logical = M·chunks`，§5.5 附两套核数）；UB 预算 ≈34KB（fp16，§4.5）。
- **裁决计划**：Stage 4 在代表 shape（prefill `[2048,28672]` fp16、ffn `[4096,4096]` fp16、smoke `[1024,2048]` fp16/fp32、decode `[1,28672]` bf16）实测主选 vs 备选；指标 = msprof op Task Duration + `vec_ratio` + `mte2_active_bw`；判定阈值 = **备选 latency < 主选 × 0.97 才翻转**；结果回写 §1.6.3。**shape 特化工厂（`lru_cache`）评估**：wrapper 按 `(M, N, dtype)` 特化编译，天然支持按 shape 分派主选/备选（如大 M 用 `block_M=2`、`M` 小或 UB 紧张用 `block_M=1`），无全局二选一需求。**设计期探针（D-3，可选前置）**：本轮未执行硬件探针（公式已由 CPU 机器验证锁定；MTE2 段合并效率属 Stage 4 实测范畴），留待裁决计划内标定。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（`op_requirements.programming_mode` 指定；与算子特征判定一致；wrapper `SiluAndMulFwdKernel.ascend_mode = "Developer"` 亦固定）

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（逐元素向量链 + 连续/跨行段 `T.copy`），无 matmul、无归约、无 Cube | 无 L1/L0/L0C 需求，不需要 MixCV |
| 复杂度级别 | 单步逐元素融合（载入 → 5~8 条向量 op → 写回），无多阶段流水 | 无精细 buffer 生命周期 / 手动同步需求 |
| 同步 | 每 tile 独立、无核间依赖 | 无手动 `sync_block_set/wait` 需求 |
| API | `T.copy` + `T.alloc_ub` + v-prefix（`vcast/vmul/vexp/vadd/vdiv`）均为 Developer 可用形态 | `docs/Tilelang.language/` 各 §2.4 示例即 `@tilelang.jit(target="npuir")` Developer 形态 |
| 先例 | `examples/elementwise/vec_add_1d.py`、`testing/autotune/SiluAndMul/silu_and_mul_2d.py`、`testing/npuir/arith_ops/test_sigmoid.py` | 同构算子已验证 |

决策树路径：纯 element-wise 单步 → Developer 模式（decision-tree.md §2）。**非 persistent + gemm 混合算子**，不落入 `CG-2026-0010`（Developer persistent+gemm 崩溃）域（本算子纯 Vector、无 `T.gemm`），故 Developer 安全。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub(shape, dtype)` 显式 UB（v-prefix 要求操作数在 UB；`T.alloc_shared` 会落到 cbuf/L1 导致 store 报错，见 §9.2） |
| 计算方式 | v-prefix 向量 API，无 `T.Parallel` 逐元素标量循环 |
| 同步 | Developer 自动同步（copy→compute→copy 依赖由编译器插入） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)`（一维） |
| 环境变量 | `TILELANG_ASCEND_MODE=Developer`（wrapper `ascend_mode="Developer"`；`examples/elementwise/example_elementwise_add.py` 先例） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式 = §1.6.1 优化后公式：`y = (gate·value)/(1 + exp(-gate))`（sub-fp32 在 fp32 域计算后 `rint` 舍回）。拆解与 §1.6.2 向量化结论一致（逐元素计算点全部向量化，无逐元素标量循环）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `g_ub ← x[m0:m0+bm, n0:n0+bn]`；`v_ub ← x[m0:m0+bm, N+n0:N+n0+bn]` | gate / value 段载入（GM→UB） |
| 2 | `g32 = f32(g)`；`v32 = f32(v)`（仅 sub-fp32） | 升域（`rint`） |
| 3 | `ng = g32 · (−1)` | 取负（`vneg` 缺失，需 `vmul` 标量） |
| 4 | `e = exp(ng)` | 指数 |
| 5 | `d = e + 1` | 分母 |
| 6 | `p = g32 · v32` | 分子（合并两处乘法之一） |
| 7 | `y32 = p / d` | silu·value |
| 8 | `y_ub = dtype(y32)`（仅 sub-fp32）；`y ← y_ub` | 回舍（`rint`）+ 写回（UB→GM） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| task 分派 | `task = i·num_kernels + cid` | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` + `T.serial(num_local)` | num_kernels/num_local 编译期常量（§5.5） | Developer |
| 段载入 | `x[r0:r1, c0:c1] ↔ g_ub[:rs, :cs]` | `T.copy(x[m0:m0+m_tail, n0:n0+n_tail], g_ub[0:m_tail, 0:n_tail])` | 切片语法，runtime 尾块 extents（`T.copy.md` §2.1/§2.2.2 第 2 条） | Developer |
| 升域 | `g32 = f32(g)` | `T.vcast(g_ub, g32, round_mode="rint")` | f16→f32 / bf16→f32 仅 `rint`（`T.vcast.md` §2.2.1） | Developer |
| 取负 | `ng = g32·(−1)` | `T.vmul(g32, -1.0, e32)` | 标量 src2（`T.vmul.md` §2.3 第 4 条） | Developer |
| 指数 | `e = exp(ng)` | `T.vexp(e32, e32)` | 同形原地（`T.vexp.md` §2.4） | Developer |
| 加一 | `d = e + 1` | `T.vadd(e32, 1.0, d32)` | 标量 src1（`T.vadd.md` §2.3 第 4 条） | Developer |
| 分子 | `p = g32·v32` | `T.vmul(g32, v32, p32)` | 同形 2D（`T.vmul.md` §2.4） | Developer |
| 除法 | `y32 = p/d` | `T.vdiv(p32, d32, y32)` | 同形 2D（`T.vdiv.md` §2.4） | Developer |
| 回舍 | `y_ub = dtype(y32)` | `T.vcast(y32, y_ub, round_mode="rint")` | f32→f16 / f32→bf16 支持 `rint`（`T.vcast.md` §2.2.1；PL-1.17） | Developer |
| 段写回 | `y[r0:r1, n0:n1]` | `T.copy(y_ub[0:m_tail, 0:n_tail], y[m0:m0+m_tail, n0:n0+n_tail])` | 同上 | Developer |
| 尾块长度 | `n_tail = min(bn, N − n0)` | `T.min(block_N, N - n0)` | 标量 min（`T.min.md`；`vec_add_1d.py` L31–L33） | Developer |
| 整除向上取整 | `ceildiv(N, bn)` | `T.ceildiv` | `docs/Tilelang.language/数学操作/T.ceildiv.md` | Developer |
| buffer 分配 | UB 片上缓冲 | `T.alloc_ub((block_M, block_N), dtype)` | `docs/Tilelang.language/内存操作/T.alloc_ub.md` | Developer |

> **备选 API 路径（可选优化，未采纳）**：Developer 模式 `T.copy` 支持跨 dtype（自动插 `hivm::VCastOp`，`T.copy.md` §2.2.1）——可用 `T.copy(x[...], g32)` 把升域**折进搬运**，省去独立 `g_ub` 暂存（UB 少 1 buffer/输入），但 VCastOp 仍在向量管线内执行，op 数不变，收益仅 UB；本设计保留显式 `vcast`（可读性与 dst 形状可控）。

### 3.3 计算伪代码

> dtype 位置传参（模板规则：`T.Tensor((M, 2 * N), dtype)` 不用 `dtype=` 关键字）。分解量在 `@tilelang.jit` 函数体内用被调参 `block_size` 计算（trace 期折叠；bitwise_and Stage 3 TRAP-jit-blocksize-decomposition）。`block_M` 为工厂常量（默认 2）。

```python
from tilelang.utils.npu_utils import NPUUtils   # 实查物理核数（§5.5）

def _make_fused_gated_explicit(M, N, dtype, op_func=None,
                               block_size=1024, block_M=2, output_dtype=None):
    # op_func 仅作形态标识（wrapper 契约）；silu-and-mul 数学在下方内联（§0.6 R3）
    out_dtype = output_dtype or dtype
    sub_f32 = dtype in ("float16", "bfloat16")     # bf16 无原生算术 + fp16 golden 对齐（PL-1.17）
    block_M = min(block_M, M)
    vector_cores = NPUUtils.get().get_aivector_core_num()   # = get_aicore_num()*2 = 40

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(block_size_arg):
        block_N = block_size_arg
        chunks = T.ceildiv(N, block_N)             # trace 期常量
        m_blocks = T.ceildiv(M, block_M)
        num_logical = m_blocks * chunks
        num_kernels = min(num_logical, vector_cores)      # persistent 分核（§5.5）
        num_local = T.ceildiv(num_logical, num_kernels)   # 静态边界

        @T.prim_func
        def main(x: T.Tensor((M, 2 * N), dtype), y: T.Tensor((M, N), out_dtype)):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                g_ub = T.alloc_ub((block_M, block_N), dtype)
                v_ub = T.alloc_ub((block_M, block_N), dtype)
                y_ub = T.alloc_ub((block_M, block_N), out_dtype)
                if sub_f32:
                    g32 = T.alloc_ub((block_M, block_N), "float32")
                    v32 = T.alloc_ub((block_M, block_N), "float32")
                p32 = T.alloc_ub((block_M, block_N), "float32")
                t32 = T.alloc_ub((block_M, block_N), "float32")   # ng -> e 原地复用
                d32 = T.alloc_ub((block_M, block_N), "float32")

                for i in T.serial(num_local):          # 核内串行，静态边界
                    task = i * num_kernels + cid
                    if task < num_logical:
                        mb = task // chunks
                        ck = task % chunks
                        m0 = mb * block_M
                        n0 = ck * block_N
                        m_tail = T.min(block_M, M - m0)
                        n_tail = T.min(block_N, N - n0)
                        T.copy(x[m0 : m0 + m_tail, n0 : n0 + n_tail],
                               g_ub[0:m_tail, 0:n_tail])
                        T.copy(x[m0 : m0 + m_tail, N + n0 : N + n0 + n_tail],
                               v_ub[0:m_tail, 0:n_tail])
                        if sub_f32:
                            T.vcast(g_ub, g32, round_mode="rint")
                            T.vcast(v_ub, v32, round_mode="rint")
                            T.vmul(g32, v32, p32)              # 分子 p = g*v
                            T.vmul(g32, -1.0, t32)             # t = -g
                            T.vexp(t32, t32)                   # t = exp(-g)
                            T.vadd(t32, 1.0, d32)              # d = 1 + exp(-g)
                            T.vdiv(p32, d32, p32)              # y32 = p/d（原地）
                            T.vcast(p32, y_ub, round_mode="rint")
                        else:
                            T.vmul(g_ub, v_ub, p32)
                            T.vmul(g_ub, -1.0, t32)
                            T.vexp(t32, t32)
                            T.vadd(t32, 1.0, d32)
                            T.vdiv(p32, d32, y_ub)
                        T.copy(y_ub[0:m_tail, 0:n_tail],
                               y[m0 : m0 + m_tail, n0 : n0 + n_tail])

        return main

    return kernel
```

> 实现注记：① 两路 op 序列由 Python `if sub_f32` **编译期**分支选择：sub-fp32 路算到 `p32`（fp32）后 `vcast(rint)` 舍入写 `y_ub`（原 dtype）；fp32 路无中转，`p32` 与 `y_ub` 同为 fp32，`T.vdiv(p32, d32, y_ub)` 直接落 `y_ub`。② v-prefix 整 buffer 操作在尾块（`m_tail<block_M` / `n_tail<block_N`）时对 `[tail:]` 段一并计算，仅 `[0:m_tail, 0:n_tail]` 写回，垃圾不落 GM（`vec_add_1d.py` 官方先例同款）。③ `x[m0:m0+m_tail, N+n0 : N+n0+n_tail]` 为跨段切片（列偏移 `N`，行 stride `2N`），TileLang 切片语法直接支持；若前端对「偏移 + 动态 extents」组合报错，退化为索引+size 形式 `T.copy(x[m0, N + n0], v_ub, size=[m_tail, n_tail])`（`T.copy.md` §2.4 示例 3）。④ `t32` 承担 `-g → exp → +1` 三步原地复用（活区间不与 `g32` 冲突：`g32` 在 `vmul` 后不再被 `t32` 链读取）。

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|----------|----------|
| `T.vmul`（`npuir_mul`） | `docs/Tilelang.language/数学操作/T.vmul.md` §2.1/§2.3（标量 src2）/§2.4 | ✅ 文档确认（fp16/fp32 原生；bf16「×」→ 本设计不依赖 bf16 算术） |
| `T.vexp`（`npuir_exp`） | `docs/Tilelang.language/数学操作/T.vexp.md` §2.2.1（fp16/fp32 √，bf16 ×）/§3 | ✅ 文档确认（fp32 原生） |
| `T.vadd`（`npuir_add`） | `docs/Tilelang.language/数学操作/T.vadd.md` §2.3（标量 src1）/§2.4 | ✅ 文档确认 |
| `T.vdiv`（`npuir_div`） | `docs/Tilelang.language/数学操作/T.vdiv.md` §2.1/§2.4 | ✅ 文档确认 |
| `T.vcast`（`npuir_cast`） | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（f16/bf16→f32 `rint`；f32→f16/bf16 `rint`） | ✅ 文档确认（PL-1.17 双证实测） |
| `T.copy`（切片 + runtime extents） | `docs/Tilelang.language/内存操作/T.copy.md` §2.1/§2.2.2 第 2/4 条/§2.4 示例 1/3 | ✅ 文档确认 + `vec_add_1d.py` / `testing/autotune/SiluAndMul/silu_and_mul_2d.py` L114–L120 先例 |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`；`vec_add_1d.py` L28–L30 | ✅（注意 `T.alloc_shared`→cbuf/L1 不可用于纯 Vector，§9.2） |
| `T.min`（标量） | `docs/Tilelang.language/比较操作/T.min.md`；`vec_add_1d.py` L31–L33 | ✅ |
| `T.ceildiv` | `docs/Tilelang.language/数学操作/T.ceildiv.md` | ✅ |
| `T.Kernel(grid, is_npu=True)` | `vec_add_1d.py` L27；`T.vexp.md` §2.4 示例 | ✅ |
| `T.serial` + `task = i·K + cid` + `if` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 示例 2（persistent 分发模式） | ✅ |
| `@tilelang.jit(out_idx=[1], target="npuir")` | 源 `out_idx=[1]` 契约（`_silu_and_mul_fwd_kernels.py` L28）+ `example_elementwise_add.py` L13 | ✅ |
| `NPUUtils.get().get_aivector_core_num()` | `tilelang/utils/npu_utils.py` L121–L123 | ✅ 实查（本机 20 → 40，§5.5） |
| `T.vsigmoid`（**未采纳**） | `docs/Tilelang.language/数学操作/T.sigmoid.md` §3（展开 4 op） | ✅ 存在但弃用（op 数 6 > 5，§1.6.0 R1 候选 2） |
| `T.vrec`（**未采纳**） | 无 `docs/Tilelang.language/` 条目；仅 `tilelang/language/customize_npuir.py` 导出 | ⚠️ 未文档化（弃选，§1.6.0 R1 候选 5） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No（源为二维 grid，NPU 一维 `T.Kernel(num_kernels)`） | 不涉及 |
| GPU 专用 API（`threads=` kwarg / `T.Parallel(threads,npt)`） | Yes | `is_npu=True` + v-prefix 取代（§0.5 舍弃项 / §0.6 R1/R2） |
| GEMM 非整除 | No（无 GEMM；非 GEMM 的尾块由 §6.4 处理） | §6.4 尾块策略 |
| L0C 溢出 | No（纯 Vector，无 L0C） | 不涉及 |
| 物理核数限制（分核三要素） | Yes（逻辑核 14–14336 vs 物理 Vector 核 40） | §5.5 persistent 分核 |
| Host 侧输入操作约束 | No（Op 层仅 `.contiguous()` 纯视图，无 expand / padding） | 合规（ascend-constraints §4） |
| uint8 / bool 张量限制 | No（输入/输出同 dtype，float-only；无 bool/uint8） | 不涉及 |
| bf16 无原生向量算术 | Yes（bf16 为 manifest dtype） | fp32 中转（PL-1.17；§1.6.1） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 并行 | `T.Kernel(ceildiv(N, block_N), M, threads=…)` 二维 + thread | 一维 `T.Kernel(n, is_npu=True)` | §0.6 R2 |
| 元素并行 | `T.Parallel(threads, npt)` 标量循环 | v-prefix 向量 API（无逐元素循环） | §1.6.2 #1–#8 |
| silu 本体 | 标量 `T.Cast` + `T.exp2` + 除 + 乘 | 向量 `vcast/vmul/vexp/vadd/vdiv` 链（fp32 域） | §1.6.1 / §0.6 R1 |
| 指数 | `exp2(-g·log2e)`（GPU MUFU 降代换） | `vexp(-g)`（NPU 两者同价） | §1.6.1 Item 1 |
| 门控数值 | fp32 域算后回舍 | 同（fp32 中转，PL-1.17） | §0.5 保留 |
| 片上驻留 | register（逐线程） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| grid 规模 | 任意（CUDA 调度器） | persistent `min(num_logical, 40)` | §0.6 R2 / §5.5 |
| 策略机 | `direct` / `explicit_parallel` | 舍弃（契约只保留 explicit_parallel） | §0.5 舍弃 |
| 工厂参数 | `(M, N, dtype, op_func, threads, npt, output_dtype)` | `(M, N, dtype, op_func, block_size, block_M, output_dtype)`；callable(`block_size`) | wrapper K11（`threads·npt` → `block_size`） |
| 激活闭包 | `op_func` 逐元素求值 | 内联向量链（签名保留 `op_func`） | §0.6 R3 |
| JIT 装饰 | `@tilelang.jit(out_idx=[1])` | `@tilelang.jit(out_idx=[1], target="npuir")` | K1 适配 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `testing/autotune/SiluAndMul/silu_and_mul_2d.py` | 高度相似（**本算子 NPU 官方 autotune 先例**） | `T.copy(A[by*block_M, bx*block_N], A_shared)` 二维分块搬运、`npuir_sigmoid`+`vmul` 组合、persistent `cid` 展平（`by = cid // nblk; bx = cid % nblk`）；**差异**：本设计改用 exp 直式（5 op vs 6）与单打包输入取段 |
| `testing/npuir/arith_ops/test_sigmoid.py` | 高度相似（`T.npuir_sigmoid` = `T.vsigmoid` 直接先例） | `T.alloc_ub` + `T.copy` + 向量激活 + `tilelang.compile(func, target="npuir")` 调用形态 |
| `examples/elementwise/vec_add_1d.py` | 高度相似（一维逐元素 + 动态尾块 + v-prefix + persistent） | `T.Kernel(n, is_npu=True)`、`T.alloc_ub`、`tail = T.min(...)` 尾块模式、runtime 边界 `T.copy`、v-prefix 用法 |
| `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/` | 中度相似（elementwise 多 op 融合迁移同族） | 多 op 向量链组织（`x·tanh(softplus(x))`）、fp32 中转 |
| `examples/clamp_max/_make_clamp_tensor_kernel/DESIGN.md` | 中度相似（同族纯 Vector 迁移结构基准） | §0/§1.6 章节组织、persistent 分核、`T.alloc_ub` 铁律、`T.alloc_shared`≠UB 陷阱、L0 测试结构 |
| `examples/ge/ge_fwd_kernel/DESIGN.md` | 中度相似（同族迁移组织） | 一维 Kernel + 切片 + 广播路径处理结构 |
| `examples/deepseek_v4/engram/engram_gate_kernel_fwd.py` | 低压相似 | `T.vsigmoid` 在真实 BlendCV/Vector kernel 中的用法（L185） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype（kernel 承载） | 说明 |
|--------|-------|---------------------|------|
| `x` | `(M, 2N)` | float16 / bfloat16 / float32 | 打包的 gate（列 `[0,N)`）与 value（列 `[N,2N)`）；行 stride = `2N` |

> **dtype 覆盖（对照 manifest signature）**：fp16/bf16/fp32 全部在 §1.6.0 给出路径——fp32 直接 fp32 链；fp16/bf16 走 fp32 中转（PL-1.17），无遗漏。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(M, N)` | **same_as(x)** | `out_idx=[1]`；无布局变换 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `g_ub` | `(block_M, block_N)` | 原 dtype | UB | gate 段载入（列 `[n0, n0+bn)`） |
| `v_ub` | `(block_M, block_N)` | 原 dtype | UB | value 段载入（列 `[N+n0, N+n0+bn)`） |
| `g32`（仅 sub-fp32） | `(block_M, block_N)` | float32 | UB | gate 升域 |
| `v32`（仅 sub-fp32） | `(block_M, block_N)` | float32 | UB | value 升域 |
| `p32` | `(block_M, block_N)` | float32 | UB | 分子 `g·v`（sub-fp32 亦承载 `y32` 结果原地） |
| `t32` | `(block_M, block_N)` | float32 | UB | `-g → exp(-g) → 1+exp(-g)` 三步原地复用链 |
| `d32` | `(block_M, block_N)` | float32 | UB | 分母（也可与 `t32` 合并，Stage 3 视分配） |
| `y_ub` | `(block_M, block_N)` | 原 dtype | UB | 回舍结果 / 输出 staging |

> **缓冲复用**：`g_ub`/`y_ub` 可按活区间合并；`t32`/`d32` 可合并（`vadd` 原地）。fp32 路径无 `g32`/`v32`/`y_ub`（`y_ub` 即 `p32`，fp32 域）。无 fp8 中转 buffer（fp8 不在契约）。

### 4.4 内存搬运路径

```
纯 Vector 单阶段（每 task，sub-fp32 路径）：
  GM[x] ─T.copy(gate 段: [m0:m0+bm, n0:n0+bn])─> UB[g_ub(dtype)] ─vcast─> UB[g32(f32)] ┐
  GM[x] ─T.copy(value 段: [m0:m0+bm, N+n0:N+n0+bn])─> UB[v_ub(dtype)] ─vcast─> UB[v32(f32)] ┤
                                                                                              ├─ vmul→ UB[p32]
   g32 ─vmul(−1)→t32 ─vexp→t32 ─vadd(+1)→d32 ────────────────────────────────────────────────┤
                                                                                              └─ vdiv(p32/d32)→ p32 ─vcast(rint)→ UB[y_ub(dtype)] ─T.copy─> GM[y]
fp32 路径同构，无两级 vcast（直接 fp32 域链）
全程 UB 内，无 L1/L0 参与（纯 Vector），无中间 GM 往返；GM 流量 = 3·M·N·E（下界）
```

### 4.5 UB 内存预算

`block_M = 2`、`block_N = block_size = 1024`（tile = 2048 元素）。中转账按 PL-1.17 口径（fp16 ≈20 B/elem、bf16 ≈24 B/elem），按 ×1.7 auto-multi-buffer 膨胀上界（`CONST-capacity-910B2C`）：

| dtype 路径 | 活跃 buffer（B/elem） | 手工合计 | ×1.7 膨胀 | 目标 UB | 余量 |
|-----------|----------------------|----------|-----------|---------|------|
| fp16 @ (2,1024) | 20（g2+v2+g32 4+v32 4+p32 4+t32 4） | 40 KB | ~68 KB | 192 KB | ✓ 充裕 |
| bf16 @ (2,1024) | 24（同上，dtype 侧 2+2） | 48 KB | ~82 KB | 192 KB | ✓ 充裕 |
| fp32 @ (2,1024) | 12（g4+v4+p4，`y_ub`=`p32`） | 24 KB | ~41 KB | 192 KB | ✓ 充裕 |

> 膨胀系数引用 `CONST-capacity-910B2C`（auto-multi-buffer ~1.7x；另有「显式 alloc 结构 ~1.12x / 反例 0.981x」结构依赖记录，本设计取保守 1.7x）。**tile 元素上限**（`floor(192KB/(B_per_elem × 1.7))`）：fp16 ≤ 5650、bf16 ≤ 4706、fp32 ≤ 9411——`block_M·block_N = 2048` 远低于上限，UB 余量充裕。**Stage 4 可选优化**：增大 `block_size`（2048/4096）以减少 `num_local`、摊薄发射项（代价：尾块浪费率上升 + UB 收紧），交 Stage 4 扫（受 §9.1 环境双态约束）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| 无 | 无 TIR 动态符号——工厂闭包按 `(M, N, dtype, block_size, block_M)` **编译期特化**（源同款模式，wrapper `lru_cache` 语义） | 不同 Op 实例触发一次 JIT 特化；kernel 边界全静态 |

> `chunks`/`m_blocks`/`num_logical`/`num_kernels`/`num_local` 全为 Python int / trace 期常量；kernel 内唯一运行时标量为 `m_tail`、`n_tail`（各 1 次/task）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")
# out_idx=[1]：main(x, y) 的 y（index 1）为输出，承接源 out_idx=[1] 契约
#   （_silu_and_mul_fwd_kernels.py L28）
# target="npuir"：K1 适配（GPU 版无 target；example_elementwise_add.py L13 先例）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 仅逐元素 silu-and-mul（`vcast/vmul/vexp/vadd/vdiv`）+ 段搬运（`T.copy`），无 matmul、无规约、无数据重排 → 全部计算落 Vector 单元，仅需 GM↔UB 搬运（decision-tree §2 纯 element-wise 分支）。

### 5.2 Block 划分

```python
block_N = block_size = 1024   # 列（输出 N 轴）tile 宽；wrapper 可调用参数（K11 = threads*npt）
block_M = 2                   # 行 tile 高；工厂常量（UB 允许下 Stage 4 可扫 2/4）
chunks  = ceildiv(N, block_N)          # 列块数
m_blocks = ceildiv(M, block_M)         # 行块数
num_logical = m_blocks * chunks        # 逻辑 task 数（raster: task = mb*chunks + ck）
# 选择理由（采纳 wrapper 默认 block_size=1024，Stage 4 可扫）：
# ① 尾轴 32B 对齐：1024×{2,4}B = 2KB/4KB，均 32 整除 ✓
# ② 向量宽度整除：fp16/bf16 ×8、fp32 ×4 均整除 1024 → 尾 lane 浪费 0%（尾部 chunk 除外）
# ③ UB 预算：tile 2048 元素 ≤ 82KB（×1.7）< 192KB ✓ 余量充裕（§4.5）
# ④ 发射项：block_M=2 使 task 数较 block_M=1 减半，摊薄 <16M 档发射暴露（§1.6.0）
# ⑤ 与同族迁移一致（源 threads*npt=1024），接口契约冻结（wrapper config）
```

### 5.3 约束分析

| 约束 | 状态 | 说明 |
|------|------|------|
| 尾轴 32B 对齐 | ✓ | `block_N·E` 为 32 的整数倍 |
| 向量宽度整除 | ✓ | 1024 对 fp16×8、bf16×8、fp32×4 均整除 |
| UB 容量 | ✓ | tile 2048 元素：fp16 ≈68KB / bf16 ≈82KB / fp32 ≈41KB，远低于 192KB |
| 非整除 M/N | ✓ | 行尾块 `m_tail = T.min(block_M, M − m0)`、列尾块 `n_tail = T.min(block_N, N − n0)` 截断（§6.4） |
| 静态边界 | ✓ | `num_local` 编译期常量；仅 `m_tail`/`n_tail` 运行时单标量 |
| 跨段切片（value 列偏移 `N`） | ✓ | 切片语法支持列偏移 + 动态 extents（`T.copy.md` §2.1/§2.2.2；退化写法见 §3.3 注记 ③） |

### 5.4 注意事项

- **非整除**：`M/N` 非 `block_M/block_N` 整数倍时，尾 tile 用 `T.min` 截断 extents；向量 op 对整 buffer 计算，仅 `[0:m_tail, 0:n_tail]` 写回。**不采用 host padding**（违反 ascend-constraints §4）。
- **`M=1`（decode）**：`block_M = min(2, M) = 1`，`m_blocks = 1`，`num_logical = chunks = ceildiv(14336, 1024) = 14 ≤ 40` → 无需 persistent 适配（§5.5）；该 shape 数据量极小（86KB），延迟主导，14 核足够。
- **`M < block_M` 或 `N < block_N`**：`m_blocks=1` / `chunks=1`，单 tile 覆盖，尾块逻辑统一处理。
- **block_size 调优**：增大 `block_size` 可摊薄发射项但收紧 UB；`block_size=4096` 时 fp16 手工 80KB ×1.7 = 136KB < 192KB ✓（bf16 更紧）。交 Stage 4 扫。

### 5.5 分核策略（物理核数适配）⭐

> 三要素（core-split-strategy.md §1；依据 docs/开发指南.md §3.3）。一句话：物理核数必须 `NPUUtils.get().get_aicore_num()` 实查并记录查询代码与返回值（纯 Vector 算子核数翻倍），禁止文档假设/经验值。

**① 逻辑核数计算**：`num_logical = ceildiv(M, block_M) · ceildiv(N, block_N)`（`block_M=2`、`block_N=1024`）：

| workload | M | N | m_blocks | chunks | num_logical |
|----------|---|---|----------|--------|-------------|
| smoke-2m `[1024, 2048]` | 1024 | 1024 | 512 | 1 | **512** |
| prefill `[2048, 28672]` | 2048 | 14336 | 1024 | 14 | **14 336** |
| decode `[1, 28672]` | 1 | 14336 | 1 | 14 | **14** |
| ffn-square-8m `[4096, 4096]` | 4096 | 2048 | 2048 | 2 | **4 096** |

**② 物理核数依据**（实查，禁止假设值）：

```python
from tilelang.utils.npu_utils import NPUUtils
n = NPUUtils.get().get_aicore_num()          # → 20（本机实查，2026-10-14）
vector_cores = NPUUtils.get().get_aivector_core_num()   # = n * 2 = 40（纯 Vector 算子核数翻倍）
```

**实查记录**：命令 `python3 -c "from tilelang.utils.npu_utils import NPUUtils; print(NPUUtils.get().get_aicore_num(), NPUUtils.get().get_aivector_core_num())"` 返回 **20 / 40**（2026-10-14）。参考 `CONST-aicore-910B2C`（910B2C = 24 AI Core / 48 Vector）与本机不同源，流量项已按 40/48 折算并标注（§1.6.0）。

**③ 规模判定与分核方案**：

| workload | num_logical | 规模判定 | num_kernels | num_local（编译期静态） |
|----------|-------------|----------|-------------|------------------------|
| smoke-2m | 512 | **中等偏大**（> 40，可对齐） | 40 | 13（40×13 = 520 ≥ 512，尾核空转 <1.6%） |
| prefill | 14 336 | **极大规模** | 40 | 359（40×359 = 14 360 ≥ 14 336） |
| decode | 14 | **逻辑核数 ≤ 物理核数** | 14 | 1（无需适配：14 ≤ 40，逻辑核不超发、无串行调度开销） |
| ffn-square-8m | 4 096 | **极大规模** | 40 | 103（40×103 = 4 120 ≥ 4 096） |

- **中等 / 极大规模**：固定启动内核数 = 物理核数（`num_kernels = min(num_logical, 40)`），核内 `T.serial(num_local)` 串行处理多个逻辑 tile，`num_local = T.ceildiv(num_logical, num_kernels)` 为**编译期静态值**（满足 core-split-strategy.md §1 要素③）；`task = i·num_kernels + cid` 的 raster 映射保证核间负载均衡（各核 task 数差 ≤1）。
- **逻辑核数 ≤ 物理核数（decode）**：`num_kernels = num_logical = 14`，`num_local = 1`——**无需适配**（逻辑核不超发，无串行调度开销；14 个核并行处理 14 个 tile）。
- **Stage 4 可评估**：调整 `block_M`/`block_size` 使 `num_logical` 更贴合 40 的整数倍（如 `block_M=4` → prefill 7 168、ffn 2 048），消除负载不均（§4.5、§5.4）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block 级 / tile 级调度循环。kernel 内**无任何逐元素标量循环**。

| 循环层级 | 类型 | 作用域 | 边界 | 说明 |
|----------|------|--------|------|------|
| 核间 | `T.Kernel(num_kernels, is_npu=True)` | 40（或 14）个 Vector 核 | 静态 | persistent 分核（§5.5） |
| 核内 task 推进 | `T.serial(num_local)` | 每核 | **静态**（`num_local` 编译期常量） | grid-stride：`task = i·num_kernels + cid` |
| tile 内逐元素计算 | **无循环** | — | — | 由 v-prefix 向量 op 覆盖整 tile（§1.6.2） |
| 尾块 | `if task < num_logical` + `m_tail/n_tail = T.min(...)` | 每 task | 运行时单标量 | 尾块截断 |

### 6.2 循环伪代码

见 §3.3。核内只有一个 `T.serial(num_local)` 调度循环（循环体 = 两次段载入 + 5~8 条向量 op + 一次段写回）；**无任何逐元素标量循环**（与 §1.6.2 结论一致）。

### 6.3 流水线优化

单阶段，无 `T.Pipelined`。auto multi-buffer（默认开启）可将 `T.copy`（MTE2）与向量链（VEC）流水重叠——`CONST-copy-floor-method` 记录 ≥16M 档计算链可被隐藏于 MTE2 窗口（lerp 7-pass fp32-transit 链实测 delta ≤1.1µs）；本算子每 tile 8 条向量 op（含 3 `vcast`），与 lerp 同量级，**≥16M 档（prefill/ffn）预期充分重叠、流量主导**；<16M 档（smoke）发射可能暴露，`block_M=2` + Stage 4 `block_size` 扫描用以摊薄（§1.6.0 roofline 注）。

### 6.4 尾块处理

`m_tail = T.min(block_M, M − m0)`、`n_tail = T.min(block_N, N − n0)`；`T.copy` 用 runtime 边界切片（`vec_add_1d.py` L34–L38 已验证模式）；向量 op 整 buffer 计算，仅 `[0:m_tail, 0:n_tail]` 写回。**不采用 host padding**（违反 ascend-constraints §4）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: Developer 自动同步（编译器插入 `T.copy` ↔ 向量链 ↔ `T.copy` 的依赖同步）。每 tile 内部依赖链 `GM→UB 载入 → 向量链 → UB→GM 写回`，无跨 tile / 跨核依赖。

### 7.2 同步点说明

| 同步点 | 类型 | 说明 |
|--------|------|------|
| `T.copy(gate/value → UB)` → `vcast` | 自动（数据依赖） | 段载入完成后方可升域/计算 |
| 向量链内部（`vmul`→`vexp`→`vadd`→`vdiv`） | 自动（数据依赖） | 逐级依赖由编译器保证 |
| `vdiv`/`vcast` → `T.copy(UB → GM)` | 自动（数据依赖） | 计算完成后方可写回 |
| 核间 | 无需 | 各核独立处理互不相交的 tile，零跨核同步 |

> 本算子非 MixCV，**无需手动 `sync_block_set/wait`**；`T.alloc_ub` 独立缓冲下无额外同步要求（区别于 Expert 双 Scope）。

### 7.3 pass_configs 配置

无特殊 pass_configs（默认 `--enable-auto-multi-buffer=true`、`--disable-hivm-auto-inject-sync=false`）；Developer 模式由 wrapper `ascend_mode="Developer"` 固定（不受环境 `TILELANG_ASCEND_MODE` 影响）。

---

## 8. 验证方案

### 8.1 Golden 函数

> Golden 以 §0.1 语义为**唯一依据**（`F.silu(gate_f32) * value_f32` 后回舍原 dtype），**不复刻** §0.6 的 NPU 算法（不模拟 `vexp`/分核/尾块），保证与源算子语义独立对齐。优先移植源仓参考实现（`examples/TileOPs/tests/ops/test_silu_and_mul.py` 的 `SiluAndMulTest.ref_program` L49–L53）。

```python
import torch
import torch.nn.functional as F

def silu_and_mul_golden(x: torch.Tensor) -> torch.Tensor:
    """Reference: F.silu(gate) * value with x packed (M, 2N), output (M, N).

    Semantics (§0.1): gate = x[:, :N], value = x[:, N:];
    y = silu(gate) * value; compute in fp32 (op math) then round back to x.dtype
    (single rounding), matching the source test's ref_program.
    """
    half = x.shape[1] // 2
    x_f32 = x.float()
    gate = x_f32[:, : half]
    value = x_f32[:, half:]
    return (F.silu(gate) * value).to(x.dtype)
```

### 8.2 精度标准

对齐 manifest conformance 测试（`examples/TileOPs/tests/ops/test_silu_and_mul.py` `_get_tolerances`）：

| dtype | atol | rtol | equal_nan | 说明 |
|-------|------|------|-----------|------|
| float16 | 1e-2 | 1e-2 | True | `silu` 在 fp32 域算后单次回舍（PL-1.17 对齐；预期远优于容差） |
| bfloat16 | 1.6e-2 | 1.6e-2 | True | 同上（bf16 走 fp32 中转） |
| float32 | 1e-5 | 1e-5 | True | 直接 fp32 链，无 dtype 舍回差异 |

`equal_nan=True` 从 §0.1 边界语义来（`silu(-inf) = NaN`、NaN 传播）；对 NaN / ±Inf 位置额外断言「输出模式与 golden 一致」（`verify_equiv.py` E1/E2 已在 fp64 域覆盖）。

### 8.3 L0 门槛测试计划

| 编号 | 用例 | 覆盖点 |
|------|------|--------|
| L0-1 | smoke `x=[1024, 2048]` fp16/bf16/fp32 | 主路径 + 全 dtype（含 **bf16** fp32 中转路径） |
| L0-2 | prefill `x=[2048, 28672]` fp16/bf16 | 大规模 workload、`chunks=14`、persistent 分核 |
| L0-3 | decode `x=[1, 28672]` bf16 | `M=1`、`block_M=min(2,M)=1`、`num_logical ≤ 40` 无需适配路径 |
| L0-4 | ffn `x=[4096, 4096]` fp16/bf16 | `M/N` ≠ block 倍数、`chunks=2` |
| L0-5 | 非整除：`x=[7, 2050]`（N=1025）、`x=[3, 2]`（N=1）| 行/列尾块截断（§6.4）、`m_tail`/`n_tail=1` |
| L0-6 | 角点：NaN / ±Inf / ±0 / 大负 `g`（`exp(-g)` 溢出）/ subnormal，fp16/fp32，`equal_nan=True` | §0.1 边界语义（`silu(-inf)=NaN`、大负 `g→0⁻`、NaN 传播） |
| L0-7 | Op 层契约：`SiluAndMulFwdOp.forward` 输出 shape == `(M, N)`、dtype == `x.dtype`；`.contiguous()` 输入 | I/O 契约（§0.2） |
| L0-8 | `eval_roofline` 返回 `(6·M·N, 3·M·N·elem_bytes)` | manifest roofline 契约 |
| L0-9 | 奇数列 `x.shape[1]` 为奇数 | Op 层应 raise（`_validate_runtime_input`） |

> 分层套件 L1/L2/Boundary 的完整枚举交由 `tilelang-op-develop`；本节仅列 L0 门槛。**L0-1/L0-2 的 bf16 用例为强制**（bf16 走 fp32 中转，验证 PL-1.17 路径）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| # | 风险 | 依据 / 缓解 |
|---|------|------------|
| R1 | **bf16 无原生向量算术**（`vexp`/`vmul`/`vdiv` 文档标 bf16「×」）：若 fp32 中转被误删，bf16 workload 编译失败 | `docs/Tilelang.language/数学操作/T.vexp.md` §2.2.1、`T.vmul.md` §2.2.1、`T.vdiv.md` §2.2.1 均标 bf16 ×；PL-1.17-subfp32-fp32-transit（两任务实证：v7 fp16 化变体在 bf16 workload 编译失败）。**本设计强制 fp32 中转**（§1.6.1）；L0-1/L0-2 bf16 用例复测 |
| R2 | **跨段切片（value 列偏移 `N` + 动态 extents）前端行为未实测** | `T.copy(x[m0:m0+m_tail, N+n0 : N+n0+n_tail], v_ub)` 依赖切片语义（`T.copy.md` §2.1）；无「列偏移 + 动态 extents」的直接先例（`silu_and_mul_2d.py` 为双输入无偏移）。**回退**：索引+size 形式 `T.copy(x[m0, N+n0], v_ub, size=[m_tail, n_tail])`（`T.copy.md` §2.4 示例 3）；L0-5 覆盖 |
| R3 | **`T.alloc_shared` 会被分配到 cbuf/L1，纯 Vector store 报错** | `examples/clamp_max/_make_clamp_tensor_kernel/DESIGN.md` §9.2 实测（`T.alloc_shared` → `#hivm.address_space<cbuf>`；`hivm.hir.store` 仅支持 ub→gm）。**本设计统一 `T.alloc_ub`** |
| R4 | **`T.vrec` 未文档化**（无 `docs/Tilelang.language/` 条目） | 仅 `tilelang/language/customize_npuir.py` 导出；本设计不用（`vdiv` 单指令更优，§1.6.0 R1 候选 5） |
| R5 | **发射项 / `block_size` 调优受环境双态影响** | 本算子每 tile 8 向量 op，<16M 档发射可能暴露；Stage 4 若扫 `block_size` 须遵守 run 间环境波动（多条 run 取中位） |
| R6 | **跨版本/跨核数外推假设** | 流量项按 40 核 / 常数 48 核折算、发射项 0.5µs/op 取自不同工具链——均为**未实测假设**（§1.6.0），Stage 4 以 msprof 实测标定 |
| R7 | **`op_func` 契约仅覆盖 SiLU** | 本设计内联 SiLU 向量链（§0.6 R3）；若复用工厂到 `gelu_and_mul` 等，须为各自激活重新内联（wrapper 目前只落地 `SiluAndMulFwdOp`） |
| R8 | **实验裁决未决（主选 #1 vs 备选 #2）** | §1.6.3 两候选判定裕度依赖未实证的 MTE2 段合并常数；Stage 4 按裁决计划实测（阈值 0.97），结果回写 §1.6.3 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| bf16 直接算术 | 删除 fp32 中转、用 bf16 直算 | 编译失败（`T.vmul` bf16 不支持） | 保留 `vcast` 中转（PL-1.17，§1.6.1） |
| `T.alloc_shared` 当 UB 用 | 照 GPU / 旧示例写 `alloc_shared` | `hivm.hir.store` 报错 / L1 分配 | 改 `T.alloc_ub`（§9.1 R3） |
| UB 溢出 | `block_M·block_size` 过大（如 `block_size=4096`+大 `block_M`） | 编译报 `ub overflow` | 减小 tile；按 §4.5 上限（fp16 tile ≤5650 元素 ×1.7）核算 |
| 越界读 | gate/value 段 extents 未截断（非整除） | 越界 / 错误值 | `m_tail`/`n_tail = T.min(...)`（§6.4） |
| 3D Kernel | 照源二维 grid 写 `T.Kernel(ax, ay, is_npu=True)` | 编译不支持 | 一维 Kernel + `(row, chunk)` raster 展平（§0.6 R2） |
| 探针 / 验证脚本 device 陷阱 | CPU 建张量后 `.npu()` 产生新张量 | 误判「全零」 | 先建 device 张量再调用，比较 device 张量 `.cpu()`（clamp 实测踩坑） |

### 9.3 特殊场景处理

- **`M=1`（decode）**：`block_M=1`、`m_blocks=1`、`num_logical=14 ≤ 40` → 无需 persistent 适配（§5.5）；延迟主导。
- **`N > block_N`（prefill）**：`chunks>1`，同一行的 gate/value 段被切成多块，每块独立 task。
- **`N` 非 `block_N` 倍数**：列尾块截断（§6.4）。
- **大负 `gate`**：`exp(-gate)→inf`，`d→inf`，`p/inf→0`（= 正确 `silu→0⁻`）；fp16/bf16 回舍后为 0（L0-6 覆盖）。
- **`gate=0`**：`silu(0)=0`，`y=0`（精确）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/silu_and_mul/_make_fused_gated_explicit/
├── DESIGN.md              # 本设计文档
├── verify_equiv.py        # D-1 等价性机器验证（15/15 EQUIV_PASS）
└── RETROSPECTIVE.md       # Stage 1 复盘
```

### 10.2 文件清单

| 文件 | 说明 | 状态 |
|------|------|------|
| `DESIGN.md` | 算子设计文档（§0 迁移分析 + §1–§11） | ✅ 已完成 |
| `verify_equiv.py` | 等价性机器验证脚本（E1/E2，fp64 参照） | ✅ 已执行 EQUIV_PASS |
| `RETROSPECTIVE.md` | Stage 1 复盘 | ✅ 已完成 |

> 实现文件由 `tilelang-op-develop` 产出（`examples/silu_and_mul/_make_fused_gated_explicit/` 下），集成目标为 wrapper `examples/TileOPs/tileops/kernels/elementwise/silu_and_mul/silu_and_mul.py`。

### 10.3 命名规范

- 项目目录 `silu_and_mul`（`op_slug`）；算子目录 `_make_fused_gated_explicit`（提取函数名）；实现文件 `_make_fused_gated_explicit.py`。

### 10.4 实现顺序

1. 在 `examples/elementwise/vec_add_1d.py` 骨架上把 `vadd` 替换为「载 gate/value 段 + v-prefix silu-and-mul 链」，验证 fp16 L0-1。
2. 加 fp32 中转（`vcast` 升/降）+ fp32 路径分支，验证 bf16（L0-1/L0-2）。
3. 加 raster 2-D tile（`block_M=2`）+ persistent 分核（§5.5）+ 尾块 `m_tail`/`n_tail`（§6.4）。
4. 跑 L0 门槛套件（§8.3），重点 bf16（L0-1/L0-2）、跨段切片（L0-5）、角点（L0-6）。
5. （Stage 4）按 §1.6.3 裁决计划实测主选 #1 vs 备选 #2。

---

## 11. 设计核心与 GPU→NPU 关键重设计（下游报告提炼用）

1. **逐元素标量链 → UB 内单遍向量链（silu 数学内联）**：源 `T.Parallel(threads, npt)` 内每元素标量 `cast→exp2→div→mul` 链（向量化口径 8 向量 op/tile）替换为 **5 条 fp32 域 v-prefix op**（`vmul` 取负 → `vexp` → `vadd` → `vmul` 分子 → `vdiv`）+ 3 条 `vcast`（sub-fp32 中转）——撤销源 `exp2(-g·log2e)` 降代换、并列 `y=(g·v)/(1+exp(-g))` 分子合并，相对 NPU 官方 `vsigmoid` 组合先例 **6 → 5 op/tile**（§0.6 R1，§1.6.0 R1/R3，§1.6.1 Item 1，`verify_equiv.py` 15/15 `EQUIV_PASS`）。
2. **fp32 计算域强制保留（PL-1.17）**：bf16 无原生向量算术（`vexp`/`vmul`/`vdiv` 文档 bf16「×」）+ fp16 需对齐 torch fp32 golden——sub-fp32 输入一律 `vcast(rint)` 升 fp32 域计算后单次 `rint` 舍回；这是**保留**（非优化），误删会导致 bf16 编译失败（§0.5，§9.1 R1）。
3. **并行结构重设计**：CUDA 二维 grid `(列块, 行)` × block/thread → **一维 `T.Kernel(num_kernels, is_npu=True)` + raster `(row, chunk)` 展平 + persistent 分核**（`num_kernels = min(num_logical, 40)`，物理 Vector 核数 = `NPUUtils.get().get_aivector_core_num()` = 40，本机实查 20→40；核内 `T.serial(num_local)` 静态边界；decode `num_logical=14 ≤ 40` 走无需适配分支）（§0.6 R2，§5.5）。
4. **门控取段与 `op_func` 内联**：单打包 `(M, 2N)` 上 gate 段（列 `[n0,n0+bn)`）与 value 段（列 `[N+n0,N+n0+bn)`）各用一条 `T.copy` 直接取段（无 host 物化、无核内重排）；`op_func` 标量闭包无法向量化 → silu 数学内联为 v-prefix 链，工厂签名保留 `op_func` 仅作形态标识（§0.6 R3，§1.6.3 方案 #1 vs 备选 #2 裁决）。
