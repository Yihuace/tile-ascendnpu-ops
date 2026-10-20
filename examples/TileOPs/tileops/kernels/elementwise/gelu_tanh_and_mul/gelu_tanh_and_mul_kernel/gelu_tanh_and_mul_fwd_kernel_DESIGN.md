# GeluTanhAndMulFwdOp 算子设计文档（Stage 1 — 迁移设计）

> 迁移来源：GPU 仓 `TileOPs-main`（`src/tileops/kernels/elementwise/activations.py` 的 `GeluTanhAndMulFwdKernel`，复用 `_base.py` 的 `FusedGatedKernel`（`_get_effective_op_func`：`fused_op(gate, value) = activation_func(gate) * value`）与 `_builders.py` 的 `make_fused_gated_explicit`）。
> 迁移目标：TileLang-NPUIR（Ascend NPU），接口函数 `gelu_tanh_and_mul_fwd_kernel`，`target="npuir"`。
> 迁移工作区：`examples/gelu_tanh_and_mul/gelu_tanh_and_mul_fwd_kernel/`（project=`gelu_tanh_and_mul`，op=`gelu_tanh_and_mul_fwd_kernel`，scenario=migration，mode=harness）。
> 本设计按 skill `tilelang-op-design` 的三问框架（Phase M0）→ 算法调研（Phase R）→ 硬件耦合性判定与 NPU 重设计（Phase M1）→ 通用设计流程（Phase 1–4）产出；programming_mode = `developer`，target = `npuir`。
> 本版为 `first_design`（首次设计）。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 skill `tilelang-op-design` 的 `references/migration-analysis.md`）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。

### 0.1 源算子语义（做什么）

**数学语义**：融合门控激活（flashinfer `gelu_tanh_and_mul`）。输入是**单个** `x`，shape `[M, 2*N]`，沿最后一维切成两半：`gate = x[:, :N]`、`value = x[:, N:]`；输出 `y = gelu_tanh(gate) * value`（shape `[M, N]`），其中 `gelu_tanh` 为 **tanh 近似形式**（非 erf 精确形式）：

$$
\operatorname{gelu\_tanh}(g) = \frac{1}{2}\,g\left(1 + \tanh\!\big(u(g)\big)\right),
\qquad u(g) = \sqrt{\tfrac{2}{\pi}}\left(g + 0.044715\,g^{3}\right),
\qquad y_{m,n} = \operatorname{gelu\_tanh}\!\big(x_{m,n}\big)\cdot x_{m,\,N+n}
$$

源实现（`GeluTanhAndMulFwdKernel.activation_func`，`activations.py` L414-423；融合由 `FusedGatedKernel._get_effective_op_func` 的 `fused_op(gate, value) = act(gate) * value` 构造）：

```python
# GeluTanhAndMulFwdKernel.activation_func（x 即 gate）
sqrt_2_over_pi = T.cast(SQRT_2_OVER_PI, "float32")   # sqrt(2/pi)
coeff = T.cast(GELU_TANH_COEFF, "float32")           # 0.044715
half = T.cast(0.5, x.dtype); one = T.cast(1.0, x.dtype)
x_f32 = T.Cast("float32", x)
inner = sqrt_2_over_pi * (x_f32 + coeff * x_f32 * x_f32 * x_f32)
tanh_val = T.Cast(x.dtype, T.tanh(inner))            # T.tanh：CUDA 上为忠实 tanh
return half * x * (one + tanh_val)
# fused_op(gate, value) = activation_func(gate) * value
```

| 语义要素 | 结论（从源码解读得出） |
|---------|----------------------|
| 数学映射 | 逐元素：`y = gelu_tanh(gate)*value`；`gate`/`value` 是**同一张量 `[M,2N]` 的前后半**（非两个独立张量）；无跨元素依赖 |
| I/O 契约 | 输入**单张量** `x [M, 2*N]`（gate/value 沿最后维拼接）；输出 `[M, N]`，与输入同 dtype；无转置、无 in-place |
| 规约语义 | **无规约**，纯逐元素；无累加顺序问题 |
| dtype 语义 | 输入 `float16 \| bfloat16 \| float32`（`SUPPORTED_DTYPES = FLOAT_DTYPES`）；输出 `same_as(input)`。tanh 的**参数与求值在 fp32**（`x_f32 = T.Cast("float32", x)`，`inner` 为 fp32，`T.tanh` 为 fp32），随后 `tanh_val` 被 cast 回 `x.dtype`，`half`/`one`/`x`/`value` 均为 `x.dtype`，**末尾乘积在窄 dtype 内**完成（两次舍入路径） |
| tanh 计算（关键） | 源用 `T.tanh`（fp32）——这是 CUDA 上的**忠实** tanh；本工具链 npuir 的向量 tanh 是 `T.vtanh`（见 §0.4 / §9.1），**两条模式路径都不是全范围忠实**（默认/Expert 为截断级数；Developer 为 exp 恒等式、`+inf`/大参数溢出为 NaN），故须显式替代（§0.6 R2） |
| 边界语义 | `±inf`：`gelu_tanh(+inf)=+inf`（`u=+inf`，`tanh=1`，`0.5·(+inf)·2=+inf`）；**`gelu_tanh(-inf)=NaN`**——源式 `half*x*(one+tanh_val)` 在 `x=-inf` 处为 `(-inf)·(1+(-1)) = (-inf)·0 = NaN`（IEEE；`one+tanh_val` 恰为 `0`）。`NaN` 输入 → `NaN`（经 `inner`/`tanh` 传播）。`0`/`-0` 保持（`u(-0.0)=-0.0`，`tanh=-0.0`，`half·(-0.0)·(1-0)=-0.0`）。`value=NaN/±inf` 经最终乘法按 IEEE 传播（`x*0` 不特殊处理——这正是 `gelu_tanh(-inf)` 为 NaN 的同一机制） |
| 空维度 / 非整除 | `M`/`N` 为构造期参数；`N` 由 `make_fused_gated_explicit` 的 `T.ceildiv(N, block_N)` 处理非整除尾列（`col` 超界不写） |

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（移植源仓测试基准 `examples/TileOPs/tests/ops/test_gelu_tanh_and_mul.py` 的 `ref_program`——其在 CPU 上用**显式 tanh 公式** `0.5*g*(1+torch.tanh(inner))*value`（fp32）计算，不复刻 NPU 算法）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `[M, 2*N]` | `float16 \| bfloat16 \| float32` | 单张量；`x[:, :N]` 为 gate，`x[:, N:]` 为 value |
| `value`（切分视图） | `[M, N]` | 同 `x` | `x[:, N:]`，kernel 内以列偏移 `N+col` 读取，不物化 |
| `output`（y） | `[M, N]` | `same_as(x)` | 输出；`y = gelu_tanh(gate) * value`，无转置 |

> Op 层契约（`examples/TileOPs/tileops/ops/elementwise/gelu_tanh_and_mul.py` 的 `GeluTanhAndMulFwdOp`）只接收/校验/转发这一个 `[M, 2N]` 张量；**不在 host 侧做切分**（切分是 kernel 内的列偏移索引）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句均已归入某步骤，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式部分 |
|------|------|------|------|-----------------|
| 1（host/Op） | 校验 `x.shape == [M, 2N]`、dtype、`x.contiguous()` | `x [M,2N]` | 同一张量（视图） | I/O 契约 |
| 2（loader） | 逐元素取 `gate = x[by, col]`、`value = x[by, N+col]` | `x` | `gate, value`（标量） | 门控切分 |
| 3 | `gate_f32 = fp32(gate)` | `gate` | `gate_f32` | 参数域提升 |
| 4 | `g2 = gate_f32·gate_f32`；`g3 = g2·gate_f32` | `gate_f32` | `g2, g3` | `g³` |
| 5 | `inner = sqrt_2_over_pi·(gate_f32 + coeff·g3)` | `gate_f32, g3` | `inner` | `u(g)` |
| 6 | `tanh_val = cast_dtype(tanh(inner))` | `inner` | `tanh_val`（窄 dtype） | `tanh(u)` |
| 7 | `gelu = half·gate·(one + tanh_val)`（窄 dtype） | `gate, tanh_val` | `gelu` | `½g(1+tanh)` |
| 8 | `y[by, col] = gelu · value`（窄 dtype） | `gelu, value` | `y` | `gelu_tanh(gate)·value` |
| 9（Op） | 返回 `y [M,N]` | `y` | 输出 | I/O 契约 |

其中 `tanh(inner)`（步骤 6）在源仓 GPU 上由 `T.tanh` 直接给出（CUDA 忠实实现）；NPU 侧无忠实向量 tanh（§0.4 #3 / §9.1），故该步在 §0.6 R2 被显式 sigmoid 机制替代。

**数据流与内存访问模式**（源 GPU 硬件视角）：

```
GM[x (M,2N)] --逐元素读取(x[by,col] / x[by,N+col])--> 寄存器 gate,value
  --fp32 提升--> g2,g3 --inner=u(g)--> 寄存器 inner --tanh--> tanh_val(窄 dtype)
  --窄 dtype 乘加--> gelu --* value--> y --写出--> GM[y (M,N)]
```

无 SMEM 分块（`explicit_parallel` 策略逐元素直读 GM；`FusedGatedKernel.DEFAULT_STRATEGY = "explicit_parallel"`）。

**循环与并行结构**：grid 二维 `T.Kernel(T.ceildiv(N, block_N), M, threads=256)`（`make_fused_gated_explicit`；`block_N = threads × num_per_thread`，GPU 默认 `npt = max(MIN_NUM_PER_THREAD, BYTES_PER_THREAD//elem_bytes)`，故 **fp16/bf16 为 256×8 = 2048、fp32 为 256×4 = 1024**——折叠值随 dtype 变，非单一 2048）；block 内 `T.Parallel(threads, npt)` 两级并行；`col = (bx*threads + i)*npt + j`。`by` 是行索引。

**host 侧逻辑**：`GeluTanhAndMulFwdOp.forward` 仅 `x.contiguous()` + 校验 + 调 kernel；`GeluTanhAndMulFwdKernel.forward` 调编译后 callable。无广播/reshape/crop；**不改输入张量内容**（满足 ascend-constraints.md §4）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | `explicit_parallel` 逐元素直读（`T.Parallel(threads, npt)`） | 尾轴连续、单遍 | 每线程 `npt` 元素连续读写 | CUDA 线程模型 / coalescing | 硬件强相关 |
| 2 | `threads × npt` 两级并行执行模型 | 每线程多元素放大吞吐 | thread/warp 组织，`block_N = threads×npt`（fp16/bf16 256×8=2048；fp32 256×4=1024） | CUDA 线程模型 | 硬件强相关 |
| 3 | `T.tanh`（fp32 内建） | 一次调用得忠实 tanh | CUDA `tanhf` 内建 | CUDA 数学库（忠实向量 tanh） | **硬件强相关**（NPU 的 `T.vtanh` 按模式分派且均非全范围忠实，须显式重设计，§0.6 R2） |
| 4 | gate/value 同张量列偏移索引（`x[by,col]`、`x[by,N+col]`） | 免切分物化、单输入契约 | 行内两段连续访存 | 无（地址算术） | 可移植 |
| 5 | fp32 参数域求值（`x_f32`）、末尾窄 dtype 乘积 | 提升 tanh 参数精度 | `T.Cast("float32", x)` 后求 `inner` 与 tanh | 无（纯精度策略） | 可移植 |
| 6 | 常数 `SQRT_2_OVER_PI` / `GELU_TANH_COEFF` 预计算（`T.cast(...,"float32")`） | 除法/开方转常数乘 | 编译期常数 | 无（纯算法层） | 可移植 |
| 7 | factory `@lru_cache(maxsize=32)` | 每 `(M,N,dtype)` 特化缓存 | host 缓存 | 无 | 可移植 |

> 源实现无 swizzle、无 mma、无 warp shuffle、无 online 算法（逐元素无跨元素依赖，天然单遍）。识别不出来 ≠ 不存在——上表为 `activations.py`（`GeluTanhAndMulFwdKernel`）+ `_base.py`（`FusedGatedKernel`）+ `_builders.py`（`make_fused_gated_explicit`）的逐项核对结果。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级（语义/算法/优化） | 源硬件依赖 | NPU 有等价能力？ | 处置（保留/等价替换/重新设计/舍弃） | NPU 对应方案 / 依据 |
|------|----------------------|-----------|-----------------|-----------------------------------|---------------------|
| 计算语义 `y=gelu_tanh(gate)*value`（tanh 近似、NaN 传播、输出同 dtype、`gelu_tanh(-inf)=NaN`） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| 逐元素单遍、无规约、无跨元素依赖 | 算法 | 无 | — | **保留** | 纯算法层（migration-analysis.md §5.3「online/分块累加 → 保留」同类） |
| 常数 `SQRT_2_OVER_PI`/`GELU_TANH_COEFF` 预计算（#6） | 算法 | 无 | — | **保留** | 纯算法层恒等变形（编译期常数） |
| fp32 参数域求值 + 末尾精度策略（#5） | 算法 | 无 | — | **保留→强化为完整 fp32 域链 + 末尾单次 cast** | 对齐 torch fp32 opmath golden（PL-1.17-subfp32-fp32-transit）；bf16 无原生 v-prefix 算术（§0.6 R2 / §1.6.1 项 2） |
| `T.tanh`（fp32 内建，#3） | 优化 | CUDA 忠实向量 tanh | **无**（npuir `T.vtanh` 默认/Expert 为截断 Maclaurin、Developer 为 exp 恒等式但 `+inf`/大参数溢出为 NaN） | **重新设计** | → §0.6 R2（**显式 sigmoid 机制** + 恒等式折叠：`0.5g(1+tanh(u))=g·σ(2u)`；σ 用显式 `vexp/vdiv`） |
| gate/value 同张量列偏移索引（#4） | 优化 | 地址算术 | 有 | **等价替换** | `T.copy` 二维切片 `x[m, N+n0 : N+n0+tail]`（`docs/Tilelang.language/内存操作/T.copy.md`；`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` 混合标量/切片先例） |
| `threads × npt` 两级并行 + `T.Parallel(threads,npt)`（#1/#2） | 优化 | CUDA 线程模型 | 无直接等价（一维 Kernel；`threads=` 在 npuir 无效果） | **重新设计** | → §0.6 R1（`block_N` 折叠 + 一维 `T.Kernel` + UB 向量化） |
| fragment/寄存器分块装载、coalesced 连续访问 | 优化 | SMEM/寄存器/coalescing | 有（UB + MTE 连续搬运） | **等价替换** | `T.alloc_ub` + `T.copy` 连续切片（GM→UB→GM），ascend-constraints.md §1 / decision-tree.md §3 |
| factory `@lru_cache`（#7） | 优化 | 无 | 有 | **等价替换** | NPU 侧同款 `@functools.lru_cache(maxsize=32)` |
| swizzle / mma / warp shuffle | 优化 | SMEM bank / tensor core / warp | 不存在于本算子 | **不适用** | 源实现无此三者，无需处置 |

> 统计：**保留 4 项、等价替换 3 项、重新设计 2 项、舍弃 0 项**。每项依据：migration-analysis.md §5.3 映射表条目 + `examples/` 同族迁移佐证（hardsigmoid / add / gelu_and_mul）+ `docs/Tilelang.language/` v-prefix API 条目。无「舍弃」项（源 `T.tanh` 机制被重新设计承接——其「一次调用得 tanh 因子」的意图由**恒等式折叠后的单次 sigmoid**承接，见 R2）。

### 0.6 NPU 算法重设计

**重设计项 R1：并行结构重设计（二维 `T.Kernel(ceildiv(N,block_N), M, threads=256)` + `T.Parallel` → 一维 `T.Kernel` + `block_N` 折叠 + UB 向量化）**

- **源方案**：二维 grid（列块 × 行 `M`）+ `threads=256`，block 内 `T.Parallel(threads, npt)` 逐元素；`block_N = threads×npt = 2048`。`T.Kernel` 三维/二维与 `threads=` 在 NPU 均不可用（ascend-constraints.md「仅支持一维 Kernel」；`TRAP-threads-kwarg-noop`：`T.Kernel(..., threads=)` 在 npuir 仅消费 `grid_size`）。
- **NPU 新算法**：一维 `T.Kernel(num_cores, is_npu=True) as (cid, _)`；把 GPU 的 `(列块, 行)` 二维任务空间**折叠为一维任务序** `task = m * n_num + n0/block_N`（`n_num = ceildiv(N, block_N)`，**行优先**使相邻任务访问相邻内存）；元素级并行交给 Vector 单元（v-prefix 整块 API 对 UB buffer 计算，天然向量化）；分块搬运用 `T.copy`（GM→UB→GM）。大规模（逻辑核数 > 物理向量核数）用核内 `T.serial` persistent 串行（§5.5）。
- **语义保持论证**：逐元素映射不变（每输出元素独立，无跨元素依赖、无规约）——`threads/npt/二维 grid` 到 `block_N/一维核` 的任务重排不改变任何输出值，每个 `y[m,n]` 的表达式与求值顺序仍与源码逐步一致（在 fp32 域，见 R2/项 2）；边界语义：尾列（`N % block_N ≠ 0`）仅 `T.copy` 有效列，v-prefix 对全 buffer 计算、垃圾尾部不写回，与源 `col` 越界不写等价。

**重设计项 R2：tanh 求值重设计（CUDA 忠实 `T.tanh` → 显式 sigmoid 机制 + 恒等式折叠；统一 fp32 域与末尾单次 cast）**

- **源方案**：`activation_func` 用 `T.tanh(inner)`——CUDA 上一个忠实 tanh 调用。这是 GPU 上「有忠实向量 tanh」的直接体现。
- **NPU 现状（关键负向证据，本工具链实测）**：npuir 的向量 tanh 是 `T.vtanh`，其低级实现**按 `TILELANG_ASCEND_MODE` 分派且两条路径都不是全范围忠实的**；`T.tanh` 则是 TIR **标量**表达式（`tilelang/language/tir/op.py`），无向量形态。
    - **默认模式（`TILELANG_ASCEND_MODE` 未设 / Expert）**：`CodeGenTileLangNPUIRAPI::VtanhCodegen`（`tilelang-mlir-ascend/src/target/codegen_npuir_api.cc` ~2526）把 `tl.npuir_vtanh` lower 为**截断的 4 项 Maclaurin 级数** `tanh(x) ≈ x − x³/3 + 2x⁵/15 − 17x⁷/315`，仅 `|x|≲0.5` 可用——实测 `vtanh(2.0)=3.31` vs `tanh=0.964`（符号错）、`vtanh(3.0)=−91.63`、`vtanh(10)=−5.27e5`、`vtanh(±inf)=NaN`。
    - **Developer 模式**：`CodeGenTileLangNPUIRDEV::VtanhCodegen`（`codegen_npuir_dev.cc` ~3365）用 exp 恒等式 `tanh(x)=(e^{2x}−1)/(e^{2x}+1)`，对有限 `|x|` 忠实，但 `u→+∞` 时 `e^{2u}` 溢出 → `∞/∞=NaN`（实测 `vtanh(50)=NaN`、`vtanh(+inf)=NaN`；`−inf→−1`）；fp32 下 `|x|>~44`（`e^{88}` 溢出）即变 NaN。
    - 两模式硬件实测见 `repro/TRAP-vtanh-truncated-taylor.py`（默认路径，截断断言）与 `history_version/design_probe_vtanh_truthfulness.py`（逐模式对照），2026-10-13，Ascend910B3。**注意**：该编译产物按 `TILELANG_ASCEND_MODE` 分派，而该变量为**进程级、trace-time 读取**（CG-2026-0013）——模式不稳定；且两条路径分别对「小 `|u|` 全错」（默认，仅 `|u|≲0.5`）与「大 `|u|`/`+inf` 出错」（Developer，溢出/`+inf→NaN`）不忠实。
    - 因此无论走哪条路径，`T.vtanh` 都不能覆盖 GELU-tanh 需要的完整实轴（标准正态 gate 已使 `|u|≈10`；`g³` 使 `|u|` 随 gate 快速增大，边界含 `±inf`）——故 `T.vtanh` 必须弃用（与同工具链 `T.verf` 截断 CG-2026-0021 同族）。
- **NPU 新算法（本设计核心）**：用恒等式把 tanh 消去，直接以**单次 sigmoid**求 GELU-tanh 因子：

  $$
  1 + \tanh(u) = 2\,\sigma(2u)
  \;\Longrightarrow\;
  \operatorname{gelu\_tanh}(g) = \tfrac{1}{2}\,g\big(1+\tanh(u)\big) = g\,\sigma(2u),
  \qquad
  \sigma(v) = \frac{1}{1+e^{-v}}
  $$

  其中 `u = sqrt(2/pi)·(g + 0.044715 g³)`。`σ` 在 **fp32 域**显式求值：`vi = −2u`（`vmul`）→ `ex = vexp(vi)`（`vexp`，溢出自动为 `+inf`/`0`）→ `den = ex + 1`（`vadd`）→ `s = 1/den`（`vdiv`，被除数用 `vbrc` 填 1 的 ones buffer，因 `vdiv` 仅允许第二操作数为标量）。据此：
  - ① 全程在 **fp32 域**求值（`g32 = fp32(gate)`），单次 sigmoid 对全实轴有效且**自饱和**：`u→+∞ ⇒ ex=exp(−∞)=0 ⇒ s=1 ⇒ gelu=g`；`u→−∞ ⇒ ex=exp(+∞)=+inf ⇒ s=1/(+inf)=0 ⇒ gelu=0`；无溢出为 NaN 的路径（与 `(exp(2u)−1)/(exp(2u)+1)` 的 inf/inf 形态不同）；
  - ② `NaN` 由 `vexp` / `vdiv` 自然传播（`σ(NaN)=NaN`），无需额外掩码；
  - ③ `−0.0` 由因子 `g` 保持（`gelu_tanh(−0.0)=(−0.0)·σ(−0.0)=−0.0`）；
  - ④ `g·σ(2u)·value` 全在 fp32 完成后**一次 `T.vcast` 落回输出 dtype**（对齐 torch golden 的 fp32 opmath，并避免 GPU 窄 dtype 尾部乘加的二次舍入）；
  - ⑤ 折叠消去一个加（`1+tanh`）、一个乘（`×0.5`），相对「先算 tanh 再 `0.5g(1+tanh)`」少 2 个向量 op，且避开近 0 处的相消。
- **语义保持论证**：
  - **数学等价（精确恒等 + 容差内）**：折叠式与源式**代数精确相等**（`1+tanh(u)=2σ(2u)`）；`σ` 用标准 `1/(1+e^{-v})` 精确实现（`T.vsigmoid` 文档 §3 亦记载其 lowering 为 VAdd/VSub/VExp/VDiv，即同一忠实形态）。数值上两式差仅来自舍入：`verify_equiv.py` 42 行采纳项全 `EQUIV_PASS`（§1.6.1），`RANGE` 段实测候选 tanh 机制全域（`u∈[−50,50]`）对 fp64 tanh max abs err **1.78e-7**、折叠 gelu 全域（`g∈[−100,100]`，`u` 涨到 ~3.6e4）max abs err **3.78e-6**（均 < fp32 atol 1e-5）；硬件探针（`history_version/design_probe_sigmoid_gelu_mechanism.py`）实测 fp32 1.9e-7 / fp16 9.4e-4 / bf16 6.8e-3。
  - **精度路径澄清**：源「inner+tanh 在 fp32，tanh 结果 cast 回窄 dtype，窄 dtype 内乘加」→ 新「全 fp32 链 + 末尾单 cast」，数学等价、舍入更接近 golden（§1.6.1 项 2 机器验证 0 违反）。
  - **边界语义核对**：`±inf` —— `u(±inf)=±inf`，`σ(+inf)=1`、`σ(−inf)=0`，故 `gelu_tanh(+inf)=+inf`、**`gelu_tanh(-inf)=(-inf)·0=NaN`**（与源式及 §8.1 golden `F.gelu(-inf,approximate="tanh")=NaN` 一致）；`NaN→NaN`；`0`/`-0` 分别保持 `+0.0`/`-0.0`（`RANGE` 段机器核验 `gelu(-0.0)` signbit）。`value` 的 NaN/±inf 经最终乘法按 IEEE 传播。
  - **验证方式**：§8.1 golden（§0.1 语义，CPU 显式 tanh）+ L0 精度对比（容差见表 §8.2）；`verify_equiv.py` 机器验证（§1.6.1，含 `RANGE`/`BOUNDARY`）。
- **源优化手段意图承接**（§0.4）：#3（忠实 tanh 因子）意图由**恒等式折叠后的单次 sigmoid**承接（且更省 2 个 op）；#1/#2（连续逐元素并行）意图由 UB `T.copy` + v-prefix 整块计算承接；#4 列偏移由二维切片承接；#5（fp32 域）保留并强化为完整 fp32 链；#6（常数）保留；#7（lru_cache）保留。无静默丢弃。

> §0.5 全部「重新设计」条目（R1/R2）已重设计；其余为保留/等价替换。§1–§11 均基于 R1/R2 落地后的 NPU 算法展开。

### 0.7 标杆实现

- 源算子路径：`/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/activations.py`（`GeluTanhAndMulFwdKernel.activation_func`）；`/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/_base.py`（`FusedGatedKernel._get_effective_op_func`）；`/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/_builders.py`（`make_fused_gated_explicit`）；`/mnt/workspace/TileOPs-main/src/tileops/ops/elementwise/activations.py`（`GeluTanhAndMulFwdOp`）。
- 源 manifest 条目：`/mnt/workspace/TileOPs-main/src/tileops/manifest/spec/elementwise_fused_gated.yaml`（`GeluTanhAndMulFwdOp`，~L56）。
- NPU 侧提取参考（合并 factory）：`examples/TileOPs/tileops/kernels/elementwise/gelu_tanh_and_mul/_gelu_tanh_and_mul_fwd_kernels.py`（factory `gelu_tanh_and_mul_fwd_kernel(M, N, dtype, threads=256, num_per_thread=8)`，`activation_func` 已内联 + `* value`，`T.tanh(inner)`）。
- NPU wrapper/op/test/bench：`examples/TileOPs/tileops/kernels/elementwise/gelu_tanh_and_mul/gelu_tanh_and_mul.py`、`examples/TileOPs/tileops/ops/elementwise/gelu_tanh_and_mul.py`、`examples/TileOPs/tests/ops/test_gelu_tanh_and_mul.py`、`examples/TileOPs/benchmarks/ops/bench_gelu_tanh_and_mul.py`。
- 源仓参考实现 / 测试基准：`examples/TileOPs/tests/ops/test_gelu_tanh_and_mul.py` 的 `ref_program`（CPU 显式 tanh 公式 `0.5*g*(1+torch.tanh(inner))*value`，fp32 opmath）；容差见同文件 `_get_tolerances`（fp32 1e-5；fp16 1e-2；bf16 1.6e-2）。
- golden 以 §0.1 语义为依据，优先移植源仓参考实现，**不复刻 NPU 算法**（保证验证独立性）。

---

## 1. 概述

### 1.1 算子名称

GeluTanhAndMulFwdOp（manifest 键）；迁移目标 TileLang 接口函数 `gelu_tanh_and_mul_fwd_kernel`。

### 1.2 功能描述

融合门控 tanh 近似 GELU 激活：输入单张量 `x [M, 2*N]`（`gate=x[:, :N]`、`value=x[:, N:]`），输出 `[M, N]`，`y = gelu_tanh(gate) * value`，`gelu_tanh` 为 tanh 近似形式（`torch.nn.functional.gelu(approximate="tanh")`）。用于 Llama/Qwen 等 SwiGLU 风格 FFN 的中间激活与门控相乘。仅浮点 dtype（fp16/bf16/fp32）；NaN 传播为 NaN。

### 1.3 数学公式

$$
y_{m,n} = \operatorname{gelu\_tanh}\!\left(x_{m,n}\right)\cdot x_{m,\,N+n},
\qquad
\operatorname{gelu\_tanh}(g) = \frac{1}{2}\,g\Big(1+\tanh\big(\sqrt{\tfrac{2}{\pi}}\,(g+0.044715\,g^{3})\big)\Big),
\qquad
m\in[0,M),\ n\in[0,N)
$$

（fp16/bf16 经 fp32 域求值后按输出 dtype 舍入一次。）

### 1.4 算法描述

单步逐元素映射（多输入 → 单输出）：Op 层把**单个** `[M, 2N]` 张量直接交给 kernel；kernel 以行优先的一维任务序处理输出 `[M, N]`，每个任务取行 `m` 的一段列块 `n0:n0+tail`：`gate` 读自 `x[m, n0:n0+tail]`，`value` 读自 `x[m, N+n0:N+n0+tail]`；在 fp32 域计算 `y = g32·σ(2·u32)·v32`，其中 `u32 = sqrt(2/pi)·(g32 + 0.044715·g32³)`、`σ` 为显式 sigmoid（§0.6 R2），单次 cast 落回输出 dtype。无规约、无窗口、无矩阵、无跨元素依赖，**天然单遍**。本算法与 §1.6.0 调研选定算法一致。与源算法结构差异：① 并行组织（二维 grid + `T.Parallel` → 一维 Kernel + `block_N` 折叠，§0.6 R1）；② tanh 求值（源 GPU 忠实 `T.tanh` → 显式 sigmoid + 恒等式折叠，§0.6 R2）；③ 精度路径（窄 dtype 尾部乘加 → 全 fp32 链 + 单次 cast，§0.6 R2 / §1.6.1 项 2）。计算表达式本身（tanh 近似 GELU 门控）与源语义一致。

### 1.5 数据流图

```
输入 x [M, 2N] --Op 校验/contiguous（不切分）--> kernel
  x[m, n0:n0+tail]     --T.copy--> UB gate_ub --vcast--> UB g32  (gate32)
  x[m, N+n0:N+n0+tail] --T.copy--> UB value_ub --vcast--> UB v32 (value32)
  g32 --vmul(×g32)--> g2 --vmul(×g32)--> g3 --vmul(×c)--> cg3
      --vadd(+g32)--> u_in --vmul(×√(2/π))--> u32
      --vmul(×(−2))--> vi32 --vexp--> ex32 --vadd(+1)--> den32
      --vdiv(1/den)--> s32 (=σ(2u)) --vmul(×g32)--> s32 (=gelu_tanh)
      --vmul(×v32)--> s32(y32) --vcast--> out_ub
  out_ub --T.copy--> y[m, n0:n0+tail]  （y [M, N]）
```

### 1.6 算法调研与优化分析 ⭐

> 设计第一优先级：先调研（§1.6.0 四问）→ 数学等价优化（§1.6.1）→ 向量化替代（§1.6.2）→ 向量化轴/布局（§1.6.3）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：`y = gelu_tanh(gate)*value`（tanh 近似形式）的算法族——所有能算出相同结果的算法结构。源算法/输入公式只是基线候选之一。本算子为**单步逐元素（多输入 → 单输出）**，按 algorithm-research.md §2 判为**轻量调研**（四问逐问回答并给依据）。对照 algorithm-candidates.md 的 elementwise 命中行（R1：公共子表达式消除 / 除法转乘倒数 / 高代价算子降代换；R2：逐元素映射天然单遍；R3：融合消除中间 GM 往返）与 pattern-library 的 `PL-1.17-subfp32-fp32-transit`、`CONST-vector-launch-overhead`。知识库预注入消费：`CASE-norm-adalayern-migration`（迁移一次过纪律：原生指令替代+机器验证）、`TRAP-threads-kwarg-noop`（`threads=` 无效，促 R1 折叠）、`CG-2026-0021`（`T.verf` 截断——本工具链原生超越函数的语义陷阱，促 R2 对 `T.vtanh` 做同款核查）。**本版新增负向证据**：`T.vtanh` 在本工具链按 `TILELANG_ASCEND_MODE` 分派两条非全范围忠实的路径（默认/Expert：截断 Maclaurin；Developer：exp 恒等式，`+inf`/大参数溢出为 NaN），不可用作 tanh——见 §0.6 R2 / §9.1。

**R1 等价化简公式候选**（基线候选必须在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线（源算法） | `gelu_tanh(g)=½g(1+tanh(u))`，`u=√(2/π)(g+0.044715g³)`；`y=gelu_tanh(gate)*value`；`tanh` 用 CUDA 忠实 `T.tanh` | — | — | ✅（基线） |
| 1 | **恒等式折叠 + 显式 sigmoid**：`½g(1+tanh(u)) = g·σ(2u)`，`σ(v)=1/(1+e^{−v})` 用 `vexp/vdiv` 显式 fp32 求值 | `gelu_tanh(g) = g·σ(2u)` | **代数精确恒等**（`1+tanh(u)=2σ(2u)`）；σ 为标准形式 | 消 1 加 + 1 乘；避开近 0 相消；且**无需忠实 tanh**（绕开 `T.vtanh` 不可用） | ✅ |
| 2 | 显式 tanh 机制 `1−2/(e^{2u}+1)` 或 `2σ(2u)−1`，再接 `½g(1+tanh)` | `tanh(u)=1−2/(e^{2u}+1)` | 容差内等价（σ/exp 标准形） | 可用（自饱和、NaN 传播），但比候选 1 多 2 个 op（未折叠） | ✅（候选 1 的未折叠形态） |
| 3 | 常数乘（`√(2/π)`、`0.044715` 预计算） | `g·√(2/π)` 等 | 数学恒等（常数预计算） | 免运行时开方/除法（基线已含） | ✅（基线即此式；无结构差异，不单列） |
| 4 | 因子重排 `½g(1+tanh)·v ↔ g·(½(1+tanh))·v` | 结合/交换律 | 数学恒等（浮点结合序差 ≤1 ulp） | 无（op 数相同） | ❌（无结构差异） |
| 5 | **`T.vtanh`/`T.tanh` 原生 tanh** | `tanh_val = vtanh(u)` | **不等价（语义错误，且按模式分派两种失败）**：默认/Expert 模式 lower 为截断 Maclaurin `x−x³/3+2x⁵/15−17x⁷/315`（仅 `\|x\|≲0.5`、域外发散）；Developer 模式用 `(e^{2x}−1)/(e^{2x}+1)`（有限 `\|x\|` 忠实，但 `+\|x\|>~44` 溢出、`+inf→NaN`）；`T.tanh` 为 TIR 标量表达式 | 名义 op 数少 | ❌（两条路径均非全范围忠实：默认实测 `vtanh(2)=3.31` vs `0.964`、`vtanh(3)=−91.63`；Developer 实测 `vtanh(+inf)=NaN`；硬件 repro 已证伪） |
| 6 | exp 型 tanh 近似 `(e^{2u}−1)/(e^{2u}+1)` | 直接除 | 容差内等价于 tanh，但 `u→+∞` 时 `∞/∞=NaN`（不饱和） | op 数 ≈ 候选 2 | ❌（溢出非饱和：`gelu_tanh(+inf)` 会变 NaN，违反边界语义；候选 1/2 的 `1/(1+e^{−v})` 形态自饱和） |
| 7 | 多项式/rational tanh（minimax） | e.g. deg-N rational | 容差内等价 | 可用，但需额外系数 + 全域裁剪，op 数不低于候选 1 | ❌（候选 1 已用标准初等函数以更少 op 达到 fp32 精度；不引入自定多项式） |

**R2 在线算法**：**无「在线/离线」之分**（即天然单遍）——`gelu_tanh(gate)*value` 是逐元素映射，每个输出元素只依赖对应 gate/value，无跨元素依赖、无 running 统计量、无归约；源算法（§0.3）已单遍。不存在需要两遍扫描 / 中间缓冲的变体。结论明确「无（单遍即最优）」，结构依据：逐元素无跨元素依赖（algorithm-research.md §3 R2 结构判据）。

**R3 复杂度对比**（逐候选四口径；带宽受限算子访存/遍数主导，不得只比 FLOPs。manifest roofline：`flops = 10*M*N`，`bytes = 3*M*N*elem_bytes`）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线（源 tanh 近似，CUDA 忠实 `T.tanh`） | 10·M·N（manifest 口径） | 3·M·N·elem（读 x 的 2·M·N + 写 y 的 M·N，无中间 GM 往返） | 1 | O(block_N)（UB 驻留，无 GM 中间缓冲） | 完全并行（逐元素独立，无跨核归并） |
| 候选 1（恒等式折叠 + 显式 sigmoid，本设计） | 10·M·N（manifest 口径不变；向量链 ~15 op） | 3·M·N·elem（同基线） | 1 | O(block_N)（无中间 GM 缓冲，UB 内复用） | 完全并行 |
| 候选 2（显式 tanh 机制，未折叠） | 10·M·N | 3·M·N·elem | 1 | O(block_N) | 完全并行（op 数比候选 1 多 2） |
| 候选 5（截断级数 `T.vtanh`，已证伪） | 10·M·N（名义） | 3·M·N·elem | 1 | O(block_N) | 完全并行（但域外发散，淘汰） |

**结论**：候选间 FLOPs / 访存 / 遍数 / 中间缓冲四口径基本同（带宽受限，访存主导）；**基线即最优访存下的最简门控式**；唯一可行且正确的 tanh 实现是**候选 1（恒等式折叠 + 显式 sigmoid）**——源依赖的忠实 `T.tanh`/`T.vtanh` 在本工具链不可用（=候选 5 被证伪），候选 1 通过代数恒等把 tanh 消去，是**全 dtype 唯一正确且最短 op 数的实现**；候选 2 为候选 1 的未折叠等价形态（列为 Stage 4 备选仅用于对照，op 数更高不采纳）。无并列争议。

**R4 硬件亲和性评估**（逐候选对照检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水可融合性 / 跨核结构；负向淘汰须附 `docs/` 与 `examples/` 佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（源 tanh / `T.tanh`） | 逐元素 → Vector（vcast/vmul/vadd/vexp/vdiv），落 Vector 正确；但忠实 tanh 在本工具链无向量形态（候选 5 已被证伪） | 工作集 = O(block_N) UB 驻留，≤51KB（§4.5） | 尾列连续；`block_N=2048` 为 32B 整数倍且整除向量宽（fp16/bf16 ×8、fp32 ×4）✓ | `ceildiv(N,block_N)`/`ceildiv(tiles,num_cores)` 均编译期常量（M/N 为 JIT 参数）✓ | 单步链可 `T.Pipelined` 双缓冲（Stage 4 可选） | ❌（作为 tanh 实现不可用，被候选 1 替换） |
| 候选 1（恒等式折叠 + 显式 sigmoid） | 逐元素 → Vector；`vmul`（g²/g³/cg³/arg/×(−2)）+ `vadd`（u_in/+1）+ `vexp` + `vdiv`（1/den）+ `vcast`，纯初等向量算子、无需特殊函数单元 | 同上（~51KB） | 无 Shape 特殊要求；尾轴同基线 ✓ | 同上 ✓ | 同上 | ✅ **采纳（并入选定算法）** |
| 候选 2（显式 tanh 未折叠） | Vector 可算（多 2 op：`vsub`/`vmul`） | 同上 | 同上 | 同上 | 同上 | ⭕（等价但更长，列为 Stage 4 对照） |
| 候选 5（截断级数 `T.vtanh`） | Vector 可算，但域外发散、`±inf→NaN` | — | — | — | — | ❌ 淘汰：本工具链 lowering 非 tanh，违反 §0.1 契约 |
| 候选 6（`(e^{2u}−1)/(e^{2u}+1)`） | Vector 可算，但正无穷处 `∞/∞=NaN` | — | — | — | — | ❌ 淘汰：溢出非饱和，违反 `gelu_tanh(+inf)=+inf` 边界 |

**定量 roofline（D-2，轻量调研豁免容量项，流量项必算）**：采用 pattern-library/constants.md 实测常数条目（版本戳 `tilelang 0.1.2+ed787bb`（2026-09-07 build）/ CANN 8.5.0 / Ascend910B2C；本机实查 20 AI Core）：

- 流量项：GM 读写总字节 ÷ 有效带宽（`CONST-mte2-degradation` 按总流量档位取值）；
- 发射项：向量 op 数 × ~0.5µs/op（`CONST-vector-launch-overhead`；本设计候选 1 链长约 15 op）；
- 容量项：豁免（单步逐元素，无跨引擎中间矩阵往返，`CONST-store-fixpipe-gm-only` 不适用——无 Cube）。

```
估算下界(large prefill M=2048,N=14336, fp16, 总流量 176.16MB):
  max(流量 ≈226µs, 发射 ≈7.5µs) ≈ 226µs   ← 带宽受限
  流量项: 输入 2·M·N·2B = 117,440,512B + 输出 M·N·2B = 58,720,256B = 176,160,768B ≈ 176.16MB;
          CONST-mte2-degradation 176M 档 ≈ 19.5 GB/s/core × 40 向量核 ≈ 780 GB/s → 226µs
  发射项: 链 vcast×2 + vmul(g2)+vmul(g3)+vmul(cg3)+vadd(u_in)+vmul(u32)+vmul(vi)
          +vexp+vadd(den)+vbrc(常数 1)+vdiv(s)+vmul(gelu)+vmul(×value)+vcast ≈ 15 op × 0.5µs ≈ 7.5µs
  容量项: 豁免（单步逐元素，无跨引擎往返）
估算下界(small decode M=1,N=14336, bf16, 7 核):
  max(流量 ≈0.03µs, 发射 ≈7.5µs) ≈ 7.5µs（发射/启动主导；launch 可与搬运重叠 → 实测可能更低）
```

**调研结论**：选定算法族 = **逐元素门控 tanh 近似 GELU（单遍），经恒等式折叠为 `y = g·σ(2u)·value`（σ 用显式 `vexp/vdiv` 在 fp32 域求值 + 末尾单 cast）**。关键依据：R1 无更少 op 的代数形式（候选 4 无差异、**候选 5 被工具链证伪**、候选 6/7 无收益），候选 1 是**全 dtype 唯一正确且最短的 tanh 实现**（代数精确恒等 `1+tanh(u)=2σ(2u)`，省 2 个向量 op 且避开近 0 相消）；R2 天然单遍；R3 四口径基线即最优访存、候选 1 在等访存下发射项 ~7.5µs 远低于流量项；R4 纯 Vector 亲和（纯初等向量算子、UB 驻留、32B 对齐、静态边界）。与基线结构差异：**代数表达式经恒等折叠等价重写**（`½g(1+tanh(u)) → g·σ(2u)`），tanh 求值由忠实内建改为显式 sigmoid，精度路径由窄 dtype 尾部改为全 fp32 + 单 cast。「无更优替代」调研范围：algorithm-candidates.md elementwise 命中行、恒等变形自查（R1 候选 3/4/5/6/7）、模式库 `PL-1.17-subfp32-fp32-transit` 与 `CONST-vector-launch-overhead`、源码 §0.3/§0.4 的算法与优化手段、工具链 `codegen_npuir_api.cc`（证伪候选 5）。源优化手段（§0.4）在调研后被 §0.6 R1/R2 承接，无静默丢弃。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法；调研候选中初判采纳的项在此完成等价论证与收益量化（与 1.6.0 不重复论证）。含采纳项，故生成并执行 `verify_equiv.py`（结果表见下）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | **恒等式折叠 + 显式 sigmoid**：把源 `T.tanh`（GPCD 忠实）替换为 `1+tanh(u)=2σ(2u)` 折叠后的单次 sigmoid | `gelu_tanh(g)=½g(1+tanh(u))`，`tanh` 由源 `T.tanh` 给出（NPU 侧 `T.vtanh` 为截断级数，不可用） | `gelu_tanh(g)=g·σ(2u)`，`σ(v)=1/(1+e^{−v})`，fp32 显式 `vexp/vdiv` | **代数精确恒等**（`1+tanh(u)=2σ(2u)`）+ 容差内等价（σ 标准形；对 fp64 参照 21 行 0 违反、0 NaN 失配）。机器验证：候选 vs 源式 max ≤0.0527 tolerance-unit（fp32）；全域 `RANGE` max abs err 1.78e-7（tanh 机制）/ 3.78e-6（折叠 gelu） | 相对源忠实 `T.tanh`（不可用）是**唯一可行且正确**的实现；相对未折叠显式 tanh（候选 2）省 1 加 + 1 乘（~2 op）；避开近 0 相消 |
| 2 | 全 fp32 域链 + 末尾单次 cast（对齐 torch golden fp32 opmath） | `tanh_val=cast(tanh(inner), x.dtype); y = half·x·(one+tanh_val)·value`（窄 dtype 内两次舍入；源 `activation_func`） | `y = cast_to(dtype)( g32·σ(2·u32)·v32 )`（fp32 乘加，一次舍入） | **容差内等价**：两式均实现 `gelu_tanh(gate)*value`，舍入实现不同，差 ≤1 ulp；新式对齐 golden（fp32 opmath）。机器验证（下表）0 违反、0 NaN 失配 | 免 bf16 无原生 v-prefix 算术（唯一可行路径）；对齐 golden 使 L0 精度稳定；不加 op（链长不变） |

**机器验证结果表**（`examples/gelu_tanh_and_mul/gelu_tanh_and_mul_fwd_kernel/verify_equiv.py`，torch CPU + fp64 参照 + 随机/角点值（±inf/NaN/−0.0/subnormal/±max/±4.5/±1e-4）+ 逐 dtype 舍入，已执行；shape ∈ {(16,128),(8,256),(1,128),(4,64),(32,128),(64,512),(3,257)}；每采纳项 21 行，共 42 行；另含 `counter npuir-vtanh` 对照与 `RANGE` / `BOUNDARY` 自检）：

| 采纳项 | max 差（tolerance-unit，cand vs 基线） | 违反率 | NaN 失配 | cand vs fp64 参照 | 结论 |
|--------|------------------------------------------|--------|---------|------------------|------|
| 1. 恒等式折叠 + 显式 sigmoid（`g·σ(2u)`） | fp32 ≤0.0182 / fp16 ≤0.0379 / bf16 ≤0.0526 | 0 / 0 | 0 | fp32 ≤0.0493 / fp16 ≤0.0493 / bf16 ≤0.0493 | EQUIV_PASS |
| 2. 全 fp32 链 + 末尾单 cast | fp32 ≤0.0182 / fp16 ≤0.0860 / bf16 ≤0.4310 | 0 / 0 | 0 | fp32 ≤0.0493 / fp16 ≤0.0493 / bf16 ≤0.0493 | EQUIV_PASS |

**全域 / 边界自检**（同脚本）：

- `RANGE`（覆盖「gate 可大可小、`g³` 增长快」这一全域盲区）：候选 tanh 机制 `2σ(2x)−1` 在 `|x|≤50` 全域对 fp64 tanh **max abs err = 1.78e-7**；折叠 `gelu_tanh(g)=g·σ(2u)` 在 `|g|≤100` 全域（`u` 涨到 ~3.6e4）对 fp64 **max abs err = 3.78e-6**（均 < fp32 atol 1e-5）；**精确饱和**：`tanh(+inf)=+1`、`tanh(-inf)=-1`、`tanh(NaN)=NaN`、`gelu_tanh(+inf)=+inf`、`gelu_tanh(-inf)=NaN`、`gelu_tanh(NaN)=NaN`；`gelu_tanh(-0.0)=-0.0`（signbit 保持）。
- `BOUNDARY`：源式与 golden（`F.gelu(approximate="tanh")`）双路径三 dtype 均 `gelu_tanh(-inf)=NaN`、`gelu_tanh(+inf)=+inf`。

> 对照项（`verify_equiv.py` 的 `counter npuir-vtanh` 段，**拒绝基线控制组，非采纳项**）：**npuir `T.vtanh` 截断级数在 `|x|>0.5` 即发散**——`vtanh(1.0)=0.746` vs `tanh=0.762`、`vtanh(2.0)=−3.308` vs `0.964`、`vtanh(3.0)=−91.63`、`vtanh(10)=−5.27e5`；`±inf→NaN`。这既解释了本设计**不得沿用 `T.vtanh`**、也印证 §0.6 R2 的显式机制必要。全部 42 行**采纳项** EQUIV_PASS，无未通过项，可采纳。脚本随算子目录留存（重跑：`total rows: 42   EQUIV_PASS: 42   not-pass: 0`，`OVERALL: EQUIV_PASS`，exit 0）。

**优化结论**：采纳 2 项（1 个恒等折叠 + 显式 sigmoid 的 tanh 求值项 + 1 个精度等价项）。优化后公式为：

$$
y_{m,n} = \operatorname{cast}_{dtype}\!\Big( g_{32}\cdot \sigma\!\big(2\cdot u_{32}\big)\cdot v_{32} \Big),
\qquad
u_{32}=\sqrt{\tfrac{2}{\pi}}\big(g_{32}+0.044715\,g_{32}^{3}\big),
\qquad
\sigma(z)=\frac{1}{1+e^{-z}}
$$

其中 `g_{32}=fp32(x_{m,n})`、`v_{32}=fp32(x_{m,N+n})`，`σ` 用 `vexp/vdiv` 显式 fp32 求值。供 §3.1 公式拆解使用。**无其他可采纳优化项**：单次逐元素门控映射，无公共子表达式（`u` 只用一次）、无除法/开方可省（`√(2/π)` 为编译期常数；`1/(1+e^{−v})` 的除法不可免）、无 cast 链冗余（输入 1 次升 + 输出 1 次降）、无中间 GM 往返；候选 6（`(e^{2u}−1)/(e^{2u}+1)`）因溢出非饱和被拒；`T.vtanh` 因工具链证伪被拒；候选 2（未折叠显式 tanh）op 数更高列为 Stage 4 对照（§5.4）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | gate 升 fp32（fp16/bf16） | 逐元素 `T.Cast("float32", gate)` | `T.vcast(gate_ub, g32, round_mode="rint")`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`；mish/lerp 佐证） | ✅ | — |
| 2 | value 升 fp32（fp16/bf16） | 逐元素 `T.Cast("float32", value)` | `T.vcast(value_ub, v32, round_mode="rint")`（同上） | ✅ | — |
| 3 | `g²`、`g³` | 逐元素 `g*g`、`g*g*g` | `T.vmul(g32, g32, g2)`；`T.vmul(g2, g32, g3)`（`docs/Tilelang.language/数学操作/T.vmul.md`） | ✅ | — |
| 4 | `cg³`、`g+cg³`、乘 `√(2/π)` | 逐元素 | `T.vmul(g3, c, cg3)`；`T.vadd(g32, cg3, u_in)`；`T.vmul(u_in, sqrt_2_over_pi, u32)`（`T.vmul.md`/`T.vadd.md`） | ✅ | — |
| 5 | `−2u` | 逐元素 | `T.vmul(u32, -2.0, vi32)`（标量 src2；`T.vmul.md`） | ✅ | — |
| 6 | `e^{−2u}` | 逐元素 `exp` | `T.vexp(vi32, ex32)`（`docs/Tilelang.language/数学操作/T.vexp.md`；fp32 ✓） | ✅ | — |
| 7 | `1+e^{−2u}` | 逐元素 `+1` | `T.vadd(ex32, 1.0, den32)`（标量 src2；`T.vadd.md`） | ✅ | — |
| 8 | 常数 1（被除数 buffer） | 逐元素填 1 | `T.vbrc(T.cast(1.0,"float32"), one32)`（`docs/Tilelang.language/shape操作/T.vbrc.md`；hardsigmoid 常数填块先例） | ✅ | — |
| 9 | `σ(2u)=1/den` | 逐元素除法 | `T.vdiv(one32, den32, s32)`（src1 为张量、src2 张量；`docs/Tilelang.language/数学操作/T.vdiv.md`） | ✅ | — |
| 10 | `×gate`（gelu_tanh） | 逐元素 `gate·(...)` | `T.vmul(g32, s32, s32)`（tensor×tensor，原地；`T.vmul.md` dst 可同 src） | ✅ | — |
| 11 | `×value`（门控乘） | 逐元素 `gelu·value` | `T.vmul(s32, v32, s32)`（tensor×tensor，原地） | ✅ | — |
| 12 | 输出降 dtype | 逐元素 `T.Cast(dtype, y32)` | `T.vcast(s32, out_ub, round_mode="rint")`（`T.vcast.md`） | ✅ | — |
| 13 | 任务行列解算 `m = task//n_num`、`n0 = (task%n_num)*block_N` | 标量 divmod | 无向量 API | ❌ | block 级任务映射索引，每任务一次，非逐元素热点（可接受理由类别①） |
| 14 | 尾列 `tail = T.min(block_N, N-n0)` | 标量 min | 无（`T.min` 标量形） | ❌ | 依赖动态 shape 的边界处理，每任务一次，非逐元素热点（可接受理由类别④；mish/hardsigmoid 同款尾块） |
| 15 | `num_cores`/`tiles_per_core` 分核元数据、Op 层 shape 校验 | trace-time Python 标量 | 无（编译期） | ❌ | host/trace 侧 shape/分核元数据计算，不在 kernel 计算环内（可接受理由类别②） |

**向量化结论**：核内逐元素计算（2 次升 cast / `g²` / `g³` / `cg³` / `+g` / `×√(2/π)` / `×(−2)` / `vexp` / `+1` / `vbrc(1)` / `vdiv` / `×gate` / `×value` / 降 cast）已**全部向量化**（替代 API 均有 `docs/Tilelang.language/` 文档佐证：vcast/vmul/vadd/vexp/vdiv/vbrc）。保留 3 处标量：任务行列解算（任务映射）、尾列大小（动态边界）、分核/Op 元数据（host/trace 侧），均非逐元素热点，理由见上表。与 §6 循环结构一致（§6 不出现逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> **背景**：I/O layout 是契约（输入 `[M, 2N]` 单张量进、输出 `[M, N]` 出）；核内布局与 lane 映射是自由变量。本算子有 2 个非 batch 维（行 `M`、列 `N`），且存在 gate/value **沿最后一维切半**的结构，故必须枚举候选并量化弃选。

**轴质量评分维度**：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | `block_N=2048`：fp16/bf16 ×8、fp32 ×4 均整除（2048=256×8=512×4）✓；`N=14336=7×2048` 整除 ✓ | 整除 ✓ |
| 尾 lane 浪费率 | 整除档 0%；尾列（任意 `N`）仅 `T.copy` 有效列，垃圾 lane 不写回 | 0（整除档） |
| 累加链形态 | **无累加**（逐元素）——列轴 stride=1 连续，无跨步系数阻碍向量化 | 无累加链 |
| repack 代价 | 主选方案无 repack（gate/value 各为行内一段连续内存，直接切片搬运）；备选若需重排则代价见下 | 0 |
| UB 容量影响 | 驻留 = O(block_N)（§4.5），布局候选间无差异 | 见 §4.5 |

**候选矩阵**（枚举有意义候选，弃选给量化理由，禁止「未考虑」）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | 行优先任务序 + 行内列块：`x[m, n0:n0+tail]`（gate）与 `x[m, N+n0:N+n0+tail]`（value）两段连续切片，列轴向量化 | 列轴（行内 stride=1 连续） | 无（gate/value 天然为行内两段连续内存） | 基线：连续读写、无核内 stride 索引、无 repack；UB=O(block_N) | ✅ **采纳** |
| 2 | host 侧先切分为两个 `[M,N]` 张量（`x[:,:N]`、`x[:,N:]` 各自 contiguous）再逐元素 | 列轴 | host 两次切片/物化（value 需一次设备端拷贝） | 引入一次额外设备端物化流量（写 ~M·N + 读）+ 破坏单张量契约（ascend-constraints.md §4 host 不得改输入真实内容） | ❌（纯增流量，且与单输入契约不符） |
| 3 | 视 `[M,2,N]` 并在核内转置为 `[2,M,N]` 分别处理 | 列轴 | 核内 `T.transpose` 重排 | 逐元素无窗口/归约轴可借助重排，重排不改访存模式，纯增 VTransposeOp 管线开销与 UB 容量（PL-1.1 家族 ~µs 级但零收益） | ❌（零收益纯增代价） |
| 4 | 整行 2N 一次载入 UB 后核内 deinterleave 出 gate/value | 列轴 | 核内解交织重排 | 单行 2N 缓冲更大（UB 压力），仍需重排；无收益 | ❌（无收益且增 UB） |
| 5 | 把 `[M,2N]` 展平为一维、按输出 `[M,N]` 连续块处理（假设块整除 N） | 展平后尾轴 | 无（块内连续） | 与候选 1 同构（块整除 N 时每个连续输出块恰在一行内，gate/value 仍为两个等长连续段）；候选 1 的行列解算更直观，不易在 `N%block_N≠0` 时越行 | ❌（与候选 1 数学同构，无额外收益） |

> 迁移对照：GPU 源码的并行轴选择（`T.Parallel(threads,npt)` 的 thread/tpt 轴、二维 grid 的 `(列块, 行)`）是**输入而非结论**——GPU thread/warp 轴与 NPU Vector 列轴不对应；GPU 的 gate/value 列偏移索引（`x[by,col]`/`x[by,N+col]`）在 NPU 由候选 1 的二维切片等价承接。本节独立评估后取「行优先任务序 + 行内列块两段连续切片 + 列轴向量化」。

**布局决策结论**：选定方案 #1「行优先任务序 + 行内列块两段连续切片（gate=`x[m,n0:n0+tail]`、value=`x[m,N+n0:N+n0+tail]`）+ 列轴向量化」，无核内 repack、无重新布局。候选 #2/#3/#4/#5 弃选理由见矩阵（增流量 / 零收益重排 / 增 UB / 数学同构），均已按文档合法形态核对（`docs/Tilelang.language/内存操作/T.copy.md` 切片语法与 rank 折叠规则；`PL-1.1-transpose-chain`；ascend-constraints.md §4）。该结论同步落入 §3.3 伪代码与 §6 循环结构（向量维 = 行内列轴，buffer 均为 1-D 列块）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer（迁移默认）。

### 2.2 选型理由

纯 Vector 逐元素算子：无 matmul（不涉及 Cube）、无归约、无跨核协作，数据流线性（GM→UB→计算→GM），无手动内存层级控制或手动同步需求。Developer 模式由编译器自动管理同步与 buffer 分配，与 mish / hardsigmoid / add 同族 elementwise 先例一致。源算子 NPU wrapper 亦声明 `ascend_mode = "Developer"`（`examples/TileOPs/tileops/kernels/elementwise/gelu_tanh_and_mul/gelu_tanh_and_mul.py`）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub` 显式指定 UB buffer（v-prefix 要求 UB 驻留数据） |
| 计算方式 | v-prefix 整块逐元素 API（vcast / vmul / vadd / vexp / vdiv / vbrc），无显式 `T.Parallel` 元素循环 |
| 同步 | 自动同步（Developer 模式编译器插入，无需手动 `set_flag`/`wait_flag`） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（`y = g·σ(2u)·value`）；每步映射 v-prefix 向量 API，与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 0 | gate 切片 `x[m, n0:n0+tail]`、value 切片 `x[m, N+n0:N+n0+tail]` | 行内两段连续切片（§1.6.3 方案 #1） |
| 1 | `g32 = cast(gate)`、`v32 = cast(value)`（fp16/bf16 时） | 升 fp32（fp32 输入走同 dtype `T.copy`），对齐 torch golden 的 fp32 opmath |
| 2 | `g2 = g32 · g32` | `g²` |
| 3 | `g3 = g2 · g32` | `g³` |
| 4 | `cg3 = g3 · c` | `c·g³`（`c=0.044715`） |
| 5 | `u_in = g32 + cg3` | `g + c g³` |
| 6 | `u32 = u_in · √(2/π)` | `u(g)` |
| 7 | `vi32 = u32 · (−2)` | `−2u` |
| 8 | `ex32 = exp(vi32)` | `e^{−2u}`（溢出自动 ±inf/0） |
| 9 | `den32 = ex32 + 1` | `1 + e^{−2u}` |
| 10 | `s32 = 1 / den32` | `σ(2u)`（被除数 1 用 ones buffer） |
| 11 | `s32 = g32 · s32` | `g·σ(2u) = gelu_tanh(g)` |
| 12 | `s32 = s32 · v32` | `y = gelu_tanh(gate)·value`（门控相乘） |
| 13 | `out = cast(s32)`（fp16/bf16 时） | 降回输出 dtype（一次舍入） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0a | 装载 gate | `T.copy(x[m, n0:n0+tail], gate_ub[0:tail])` | 二维切片（标量行 + 列切片）→ 1-D UB；尾列 `tail` 裁剪（TRAP-T-copy-region-semantics） | Developer |
| 0b | 装载 value | `T.copy(x[m, N+n0:N+n0+tail], value_ub[0:tail])` | 列起点偏移 `N` | Developer |
| 1a | gate 升 fp32 | `T.vcast(gate_ub, g32, round_mode="rint")` | fp16/bf16 → fp32（fp32 走 `T.copy`） | Developer |
| 1b | value 升 fp32 | `T.vcast(value_ub, v32, round_mode="rint")` | 同上 | Developer |
| 2 | `g2 = g32·g32` | `T.vmul(g32, g32, g2)` | tensor×tensor | Developer |
| 3 | `g3 = g2·g32` | `T.vmul(g2, g32, g3)` | tensor×tensor | Developer |
| 4 | `cg3 = g3·c` | `T.vmul(g3, GELU_TANH_COEFF, cg3)` | `c` 为 fp32 标量 | Developer |
| 5 | `u_in = g32+cg3` | `T.vadd(g32, cg3, u32)` | tensor×tensor（原地写 u32 暂存 u_in） | Developer |
| 6 | `u32 = u_in·√(2/π)` | `T.vmul(u32, SQRT_2_OVER_PI, u32)` | 标量 src2，原地 | Developer |
| 7 | `vi32 = u32·(−2)` | `T.vmul(u32, -2.0, vi32)` | 标量 src2 | Developer |
| 8 | `ex32 = exp(vi32)` | `T.vexp(vi32, ex32)` | fp32 exp | Developer |
| 9 | `den32 = ex32+1` | `T.vadd(ex32, 1.0, den32)` | 标量 src2 | Developer |
| 10 | `σ = 1/den32` | `T.vbrc(T.cast(1.0,"float32"), one32)`；`T.vdiv(one32, den32, s32)` | 常数广播 + 张量除（src1 张量、src2 张量） | Developer |
| 11 | `gelu = g32·σ` | `T.vmul(g32, s32, s32)` | tensor×tensor，原地 | Developer |
| 12 | `y = gelu·v32` | `T.vmul(s32, v32, s32)` | tensor×tensor，原地 | Developer |
| 13 | 降回输出 dtype | `T.vcast(s32, out_ub, round_mode="rint")` | fp32 → fp16/bf16（fp32 走 `T.copy`） | Developer |
| 14 | 写回 | `T.copy(out_ub[0:tail], y[m, n0:n0+tail])` | 二维切片，尾列裁剪 | Developer |

> fp32 输入路径跳过步骤 1a/1b/13（无 vcast，走同 dtype `T.copy`——**本工具链 `T.vcast(fp32→fp32)` 会损坏数据**，硬件探针实证，见 §9.2）。常数 `SQRT_2_OVER_PI`、`GELU_TANH_COEFF`、`-2.0`、`1.0` 均以 fp32 标量传递。标量 operand 用原生 Python float（同 add 先例：NPU 二元/一元 op 前端只接受裸标量）。**无 `FusedGatedKernel` 的 `fused_op` 闭包**：NPU 侧 `activation_func(gate)*value` 直接内联展开为步骤 2–12。**无 `T.vtanh`/`T.tanh`**（本工具链不可用，见 §0.6 R2 / §9.1）。

### 3.3 计算伪代码

> 约定：`T.Tensor` 的数据类型以**第二个位置参数**传入（不用 `dtype=` 关键字），避免产生 `name 'in_dtype' is not defined` 类 false alarm（design-template §3.3）。

```python
import functools
import tilelang
import tilelang.language as T
from tilelang.utils import NPUUtils

SQRT_2_OVER_PI = 0.7978845608028654   # sqrt(2/pi)
GELU_TANH_COEFF = 0.044715            # tanh-GELU 三次项系数


@functools.lru_cache(maxsize=32)
def gelu_tanh_and_mul_fwd_kernel(M, N, dtype, output_dtype=None, block_N=2048, num_cores=None):
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"
    n_num = T.ceildiv(N, block_N)                 # 每行逻辑列块数（编译期常量）
    num_logical_kernels = M * n_num               # 逻辑核数（行 × 列块）
    vector_cores = NPUUtils.get().get_aicore_num() * 2   # 物理核数实查 ×2（纯 Vector）
    if num_cores is None or num_cores <= 0 or num_cores >= num_logical_kernels:
        num_cores = min(num_logical_kernels, vector_cores)
    tiles_per_core = T.ceildiv(num_logical_kernels, num_cores)   # 编译期常量，静态边界

    sqrt_2_over_pi = T.cast(SQRT_2_OVER_PI, "float32")
    coeff = T.cast(GELU_TANH_COEFF, "float32")

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(_M, _N, _block_N, _num_cores):
        @T.prim_func
        def main(
            x: T.Tensor((_M, 2 * _N), dtype),
            y: T.Tensor((_M, _N), out_dtype),
        ):
            with T.Kernel(_num_cores, is_npu=True) as (cid, _):
                gate_ub = T.alloc_ub((_block_N,), dtype)
                value_ub = T.alloc_ub((_block_N,), dtype)
                g32 = T.alloc_ub((_block_N,), compute_dtype)
                v32 = T.alloc_ub((_block_N,), compute_dtype)
                g2 = T.alloc_ub((_block_N,), compute_dtype)     # g^2
                g3 = T.alloc_ub((_block_N,), compute_dtype)     # g^3
                cg3 = T.alloc_ub((_block_N,), compute_dtype)    # c * g^3
                u32 = T.alloc_ub((_block_N,), compute_dtype)    # u(g) / 复用链
                vi32 = T.alloc_ub((_block_N,), compute_dtype)   # -2u
                ex32 = T.alloc_ub((_block_N,), compute_dtype)   # exp(-2u)
                den32 = T.alloc_ub((_block_N,), compute_dtype)  # 1 + exp(-2u)
                one32 = T.alloc_ub((_block_N,), compute_dtype)  # 常数 1
                s32 = T.alloc_ub((_block_N,), compute_dtype)    # sigmoid / gelu / 结果链
                out_ub = T.alloc_ub((_block_N,), out_dtype)

                for i in T.serial(tiles_per_core):           # 核内串行 tile 循环（静态边界）
                    task = i * _num_cores + cid
                    if task < _M * T.ceildiv(_N, _block_N):
                        m = task // T.ceildiv(_N, _block_N)  # 行号
                        n0 = (task % T.ceildiv(_N, _block_N)) * _block_N
                        tail = T.min(_block_N, _N - n0)

                        # --- 行内两段连续切片装载 ---
                        T.copy(x[m, n0 : n0 + tail], gate_ub[0:tail])
                        T.copy(x[m, _N + n0 : _N + n0 + tail], value_ub[0:tail])

                        # --- 升 fp32（fp16/bf16）或同 dtype copy（fp32）---
                        if dtype != compute_dtype:
                            T.vcast(gate_ub, g32, round_mode="rint")
                            T.vcast(value_ub, v32, round_mode="rint")
                        else:
                            T.copy(gate_ub, g32)
                            T.copy(value_ub, v32)

                        # --- u(g) = sqrt(2/pi) * (g + 0.044715 g^3) ---
                        T.vmul(g32, g32, g2)             # g^2
                        T.vmul(g2, g32, g3)              # g^3
                        T.vmul(g3, coeff, cg3)           # c g^3
                        T.vadd(g32, cg3, u32)            # g + c g^3
                        T.vmul(u32, sqrt_2_over_pi, u32) # u(g)

                        # --- 显式 sigmoid：gelu_tanh(g) = g * sigmoid(2u) ---
                        T.vmul(u32, -2.0, vi32)          # -2u
                        T.vexp(vi32, ex32)               # exp(-2u)（溢出 -> +inf/0）
                        T.vadd(ex32, 1.0, den32)         # 1 + exp(-2u)
                        T.vbrc(T.cast(1.0, "float32"), one32)  # 常数 1（被除数）
                        T.vdiv(one32, den32, s32)        # sigmoid(2u) = 1/(1+exp(-2u))
                        T.vmul(g32, s32, s32)            # gelu_tanh(g) = g * sigmoid(2u)

                        # --- 门控相乘：y = gelu_tanh(gate) * value ---
                        T.vmul(s32, v32, s32)

                        # --- 降回输出 dtype（fp16/bf16）或同 dtype copy（fp32）---
                        if out_dtype != compute_dtype:
                            T.vcast(s32, out_ub, round_mode="rint")
                        else:
                            T.copy(s32, out_ub)

                        # --- 写回（尾列裁剪）---
                        T.copy(out_ub[0:tail], y[m, n0 : n0 + tail])

        return main

    return kernel(M, N, block_N, num_cores)
```

> 注：常数 1 用 `T.vbrc`（标量广播到整块）填入 `one32`（`T.div`/`vdiv` 仅允许第二操作数为标量，故被除数须为张量 buffer；hardsigmoid 常数填块先例同款）。若 `T.vbrc` 的标量源须 let 绑定，则先 `one_c = T.cast(1.0, "float32")`。`g32`/`v32`/`u32`/`vi32`/`s32` 复用为 fp32 计算缓冲（`T.vmul`/`T.vadd`/`T.vdiv` 支持 `dst` 与 `src` 同 buffer 原地操作）。fp32 路径（`dtype==compute_dtype`）可将 `g32←gate_ub`、`v32←value_ub`、`out_ub←s32` 复用，驻留从 ≤51KB 降至 ≤40KB（§4.5）。`T.ceildiv(_N,_block_N)` 与 `tiles_per_core` 均为编译期常量，核内循环边界静态。**无 `T.vtanh`/`T.tanh` 调用**。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.vmul(src0, src1, dst)` | `docs/Tilelang.language/数学操作/T.vmul.md`（src2 可标量；dst 可原地；fp16/fp32 ✓、bf16 ×） | ✅ 已验证（文档读 + 硬件探针） |
| `T.vadd(src0, src1, dst)` | `docs/Tilelang.language/数学操作/T.vadd.md`（src1 可标量；dst 可原地） | ✅ 已验证（文档读 + 硬件探针） |
| `T.vexp(src, dst)` | `docs/Tilelang.language/数学操作/T.vexp.md`（fp16/fp32 ✓、bf16 ×；整块 exp） | ✅ 已验证（文档读 + 硬件探针；溢出 → +inf） |
| `T.vdiv(src1, src2, dst)` | `docs/Tilelang.language/数学操作/T.vdiv.md`（src2 可标量、src1 必须张量；dst 可原地；fp16/fp32 ✓） | ✅ 已验证（文档读 + 硬件探针） |
| `T.vbrc(scalar, dst)` | `docs/Tilelang.language/shape操作/T.vbrc.md`（标量广播填块）；hardsigmoid 先例 `T.vbrc(nan_scalar, buf)` | ✅ 已验证（文档 + examples + 硬件探针） |
| `T.vcast(src, dst, round_mode)` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md`；mish/lerp 佐证；**仅用于 fp16/bf16↔fp32，不用于 fp32→fp32** | ✅ 已验证（文档读 + 硬件探针发现 fp32→fp32 陷阱） |
| `T.copy`（切片，二维 + 标量行） | `docs/Tilelang.language/内存操作/T.copy.md`（切片语法 + rank 折叠规则 1/2；`_gqa_prefill_fwd_kernel.py` 混合标量/切片先例） | ✅ 已验证（文档 + examples） |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`；mish/gelu_tanh_and_mul wrapper | ✅ 已验证 |
| `T.Kernel(一维, is_npu=True) as (cid, _)` | mish / hardsigmoid / add 同族先例 | ✅ 已验证 |
| `T.ceildiv` / `T.min`（标量） | `docs/Tilelang.language/数学操作/T.ceildiv.md`；mish 尾块 `T.min` 先例 | ✅ 已验证 |
| `NPUUtils.get().get_aicore_num()` | `docs/开发指南.md` §3.3；本机实查（§5.5） | ✅ 已实查（=20） |
| `T.vtanh(src, dst)` / `T.tanh` | 文档 `docs/Tilelang.language/数学操作/T.vtanh.md` **存在**，但工具链按模式分派：`codegen_npuir_api.cc` `VtanhCodegen`（默认/Expert）为截断 4 项级数，`codegen_npuir_dev.cc` `VtanhCodegen`（Developer）为 exp 恒等式（`+inf`/溢出 → NaN）；`T.tanh` 为 TIR 标量表达式 | ❌ **不可用（已证伪，弃用）** |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | Yes（源为二维 `T.Kernel(ceildiv(N,block_N), M)`） | 折叠为一维 `T.Kernel(num_cores, is_npu=True)`，行列并入任务序（§0.6 R1） |
| GPU 专用 API | Yes（`T.Parallel(threads,npt)`、`threads=`、CUDA 忠实 `T.tanh`） | 全部改写为 NPU v-prefix API + 显式 sigmoid 机制（§0.5/§0.6） |
| 无可用忠实向量 tanh（`T.vtanh` 为截断级数、`T.tanh` 为标量） | Yes | 恒等式折叠 + 显式 sigmoid（§0.6 R2 / §9.1）；**禁止 `T.vtanh`/`T.tanh`** |
| `T.vcast(fp32→fp32)` 损坏数据 | Yes（fp32 输入/输出路径） | fp32 路径改 `T.copy`（同 dtype），仅跨精度转换用 `T.vcast`（§9.2） |
| GEMM 非整除 | 不适用（无 GEMM） | — |
| L0C 溢出 | 不适用（无 Cube） | — |
| 分核策略 | Yes | §5.5 三要素（逻辑核数 / 物理核数实查 / 规模判定与分核方案） |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键技术差异汇总）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 维度 | 二维 `T.Kernel(ceildiv(N,block_N), M, threads=256)` | 一维 `T.Kernel(num_cores, is_npu=True)` | §0.6 R1 |
| 元素并行 | `T.Parallel(threads, npt)` 显式两级循环 | v-prefix 整块 API（Vector 单元隐式向量化） | §0.6 R1 |
| 内存分配 | 直接读 GM（`explicit_parallel`） | `T.alloc_ub` + `T.copy`（GM→UB→GM） | §0.5 等价替换 |
| tanh 求值 | CUDA 忠实 `T.tanh(inner)` | 恒等式折叠 + 显式 sigmoid（`g·σ(2u)`，fp32） | §0.6 R2 |
| 精度路径 | tanh 结果 cast 回窄 dtype，尾部窄 dtype 乘加 | 全 fp32 链 + 末尾单 cast | §0.6 R2 / §1.6.1 项 2 |
| 门控切分 | `x[by,col]` / `x[by,N+col]` 逐元素列偏移 | `T.copy` 二维切片（gate/value 两段连续） | §0.5 等价替换 |
| `threads×npt` | fp16/bf16 256×8=2048、fp32 256×4=1024 | `block_N=2048`（NPU 选定列块，取窄 dtype 折叠值 256×8） | §5.2 |
| factory 缓存 | `@lru_cache(maxsize=32)` | `@lru_cache(maxsize=32)` | §0.5 等价替换 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/elementwise/gelu_and_mul/gelu_and_mul_kernel/gelu_and_mul_fwd_kernel_DESIGN.md` | 最高（**同族门控激活迁移**，仅激活函数不同：erf vs tanh） | 一维 Kernel + block_N 折叠、fp32 中转、显式超越函数机制、分核三要素、UB 预算口径、`verify_equiv` 结构、边界语义（`gelu(-inf)=NaN`） |
| `examples/TileOPs/tileops/kernels/elementwise/gelu_tanh_and_mul/gelu_tanh_and_mul.py` | 最高（本算子 NPU scaffold） | `_register` wrapper、`ascend_mode="Developer"`、SUPPORTED_DTYPES 浮点集、单 `[M,2N]` 输入契约、`default_config` 返回 `{}` |
| `examples/TileOPs/tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/hardsigmoid_fwd_kernel_DESIGN.md` | 高度相似（同族 elementwise 迁移设计） | 一维 Kernel + block_size 折叠、fp32 中转、分核三要素、UB 预算口径、`vbrc` 常数填块先例、`vclamp`/`vcmp`+`vselect` NaN 先例 |
| `examples/TileOPs/tileops/kernels/elementwise/add/add_kernel/add_fwd_kernel_DESIGN.md` | 高度相似（同族 elementwise 迁移设计） | 多输入契约、fp32 中转、persistent 分核、UB 预算、`verify_equiv` 结构 |
| `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py` | 高度相似（同族 unary 激活迁移） | Developer 模式、`T.alloc_ub`、`T.copy` 尾块、v-prefix 链、fp32 中转、`exp` 饱和裁剪先例 |
| `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel` | 相似（同族 elementwise + fp32 中转） | fp32 中转模式（PL-1.17）、`T.min` 尾块 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `[M, 2*N]` | `float16 \| bfloat16 \| float32` | **单张量**（gate/value 沿最后维拼接）；kernel 内 `gate=x[m,n0:n0+tail]`、`value=x[m,N+n0:N+n0+tail]`，不物化切分 |
| `M` / `N` | 标量 | int | JIT 编译期参数（每 `(M,N,dtype)` 一特化） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `[M, N]` | `same_as(x)` | 输出；无转置、与输入同 layout |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `gate_ub` | `(block_N,)` | input dtype | UB | gate tile（`x[m,n0:n0+tail]`→UB） |
| `value_ub` | `(block_N,)` | input dtype | UB | value tile（`x[m,N+n0:N+n0+tail]`→UB） |
| `g32` | `(block_N,)` | float32 | UB | gate fp32 中转（vcast 目标；fp32 时即 gate_ub） |
| `v32` | `(block_N,)` | float32 | UB | value fp32 中转（fp32 时即 value_ub） |
| `g2` | `(block_N,)` | float32 | UB | `g²` |
| `g3` | `(block_N,)` | float32 | UB | `g³` |
| `cg3` | `(block_N,)` | float32 | UB | `c·g³` |
| `u32` | `(block_N,)` | float32 | UB | `g+cg³` 暂存，随后为 `u(g)`（原地复用） |
| `vi32` | `(block_N,)` | float32 | UB | `−2u` |
| `ex32` | `(block_N,)` | float32 | UB | `e^{−2u}` |
| `den32` | `(block_N,)` | float32 | UB | `1+e^{−2u}` |
| `one32` | `(block_N,)` | float32 | UB | 常数 1（`vdiv` 被除数，`vbrc` 填） |
| `s32` | `(block_N,)` | float32 | UB | sigmoid / gelu_tanh / 结果链（原地复用） |
| `out_ub` | `(block_N,)` | output dtype | UB | 输出 tile（UB→GM） |

### 4.4 内存搬运路径

```
GM[x (M,2N)] --T.copy(x[m,n0:n0+tail])-->      UB[gate_ub]  --vcast--> UB[g32]
GM[x (M,2N)] --T.copy(x[m,N+n0:N+n0+tail])-->  UB[value_ub] --vcast--> UB[v32]
  g32 --vmul(×g32)--> g2 --vmul(×g32)--> g3 --vmul(×c)--> cg3
      --vadd(+g32)--> u32 --vmul(×√(2/π))--> u32 --vmul(×(−2))--> vi32
      --vexp--> ex32 --vadd(+1)--> den32 --vdiv(one32/den32)--> s32 (σ(2u))
      --vmul(×g32)--> s32 (gelu_tanh) --vmul(×v32)--> s32 (y32)
  UB[s32] --vcast--> UB[out_ub] --T.copy(out_ub[0:tail])--> GM[y (M,N)]
```

纯 Vector：GM ↔ UB 两级，无 L1/L0A/L0B/L0C（无 Cube）。数据仅 GM→UB 一次读入、UB→GM 一次写出，无中间 GM 往返。

### 4.5 UB 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `gate_ub` | (2048,) | fp16/bf16 | 4096（fp32 时 8192） |
| `value_ub` | (2048,) | fp16/bf16 | 4096（fp32 时 8192） |
| `g32` | (2048,) | fp32 | 8192 |
| `v32` | (2048,) | fp32 | 8192 |
| `g2` | (2048,) | fp32 | 8192 |
| `g3` | (2048,) | fp32 | 8192 |
| `cg3` | (2048,) | fp32 | 8192 |
| `u32` | (2048,) | fp32 | 8192 |
| `vi32` | (2048,) | fp32 | 8192 |
| `ex32` | (2048,) | fp32 | 8192 |
| `den32` | (2048,) | fp32 | 8192 |
| `one32` | (2048,) | fp32 | 8192 |
| `s32` | (2048,) | fp32 | 8192 |
| `out_ub` | (2048,) | fp16/bf16 | 4096（fp32 时 8192） |
| **总计（fp16/bf16 路径，全 buffer 显式）** | | | **106496（104KB）** |
| **总计（fp32 路径，全 buffer 显式）** | | | **122880（120KB）** |
| **总计（fp16/bf16 路径，复用 `g32←gate_ub`、`v32←value_ub`、`out_ub←s32`）** | | | **94208（92KB）** |
| **总计（fp32 路径，复用 `g32←gate_ub`、`v32←value_ub`、`out_ub←s32`）** | | | **98304（96KB）** |

对照目标平台 UB 容量 **192KB**（`CONST-capacity-910B2C`：UB=192KB/AIV；version stamp `tilelang 0.1.2+3a214cde` + CANN 8.5.0）。该条目记录 auto-multi-buffer 约 1.7x 膨胀、或（显式 alloc 结构）基础分配差 ≈ 手工预算 ×1.10–1.12。按 ×1.12 上偏系数：

- **fp16/bf16 路径（全显式）**：104KB ×1.12 ≈ 117KB < 192KB ✓，余量 ~39%。
- **fp32 路径（全显式）**：120KB ×1.12 ≈ 134KB < 192KB ✓，余量 ~30%。
- **fp16/bf16 路径（复用）**：92KB ×1.12 ≈ 103KB，余量 ~46%。
- **fp32 路径（复用，推荐）**：96KB ×1.12 ≈ 108KB，余量 ~44%。

结论：全部 dtype 路径均不溢出，`block_N=2048` 有余量；Stage 4 可据此扫描更大的 `block_N`（受 192KB 约束）。注：本设计的向量链（~15 buffer）较短，`block_N` 上限受 UB 约束但仍宽松。

### 4.6 动态轴定义

无动态轴。`M`/`N` 为 JIT 编译期参数（每 `(M,N,dtype)` 一特化，`@lru_cache`）；`N` 的编译期值驱动 `n_num`/`tiles_per_core` 静态边界。尾列由 `tail = T.min(block_N, N - n0)` 处理；不引入运行时动态轴。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")
```

无额外 `pass_configs`（Developer 默认）。**tanh 机制常数**：`SQRT_2_OVER_PI = 0.7978845608028654`、`GELU_TANH_COEFF = 0.044715`（与源仓 `tileops/kernels/constants.py` 一致），以 fp32 标量传入（§3.3）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector。

**判定依据**：仅逐元素运算（cast / 常数乘 / 乘加 / exp / div / 比较无关的纯算术链），无 matmul、无归约，不涉及 Cube 单元。

### 5.2 Block 划分

```python
block_N = 2048              # NPU 选定行内列块宽（取自 fp16/bf16 GPU 折叠值 threads×npt=256×8）
n_num = T.ceildiv(N, block_N)   # 每行列块数
num_logical_kernels = M * n_num # 逻辑核数（行 × 列块）
```

- **选择理由**：2048 为**本平台 NPU 选定列块宽**，取自 fp16/bf16 GPU 折叠值 `threads(256) × npt(8)`（K11 迁移规则：CUDA `threads×npt` 折叠为单一 `block_size`，由 factory 参数传入）。注：GPU 折叠值随 dtype 变（`npt = max(MIN_NUM_PER_THREAD, BYTES_PER_THREAD//elem_bytes)`，**fp32 为 256×4=1024**），故 2048 是 NPU 设计取值而非源通用值；32B 对齐（2048×2/4B 均为 32B 整数倍）；整除向量宽（fp16/bf16 ×8、fp32 ×4）；整除 manifest `N=14336`（14336/2048=7）；UB 预算 92–120KB ≪ 192KB（§4.5），块宽足够使向量链发射开销摊薄（`CONST-vector-launch-overhead`：宽块 op 恒优于窄块 op 翻倍）。
- **块宽敏感性**：`block_N` 增大 → 任务数减少（MTE 指令数下降）、但单任务 UB 增大且小 `M` workload 的并行度下降；Stage 4 可扫描 `block_N`（受 UB 192KB 约束）与 `T.Pipelined`。

### 5.3 约束分析

- **对齐约束**：`block_N=2048`，fp16/bf16 尾轴 2048×2B=4KB、fp32 2048×4B=8KB，均 32B 对齐 ✓；整除向量宽 ✓。
- **UB 容量**：92–120KB < 192KB（§4.5）✓。
- **L0/L1/L0C 容量**：无 Cube 计算，不适用。

### 5.4 注意事项

- **非整除尾列**：`tail = T.min(block_N, N - n0)`，仅 `T.copy` 有效列进/出；v-prefix 对全 buffer 计算，尾部垃圾元素不写回（与 mish 同款）。
- **极小 shape**：`N < block_N` 时 `n_num=1`，`tail=N`，单列块；`M=1` 时逻辑核数 `= n_num`（decode 场景 7 核）。
- **gate/value 偏移**：value 列起点恒为 `N+n0`（`x` 的第 2 段），与输出列 `n0` 同段长 `tail`。
- **tanh 机制 op 数权衡（Stage 4 备选）**：本设计用恒等式折叠 + 单次 sigmoid（~15 向量 op）。若 Stage 4 需进一步压低 decode 时延，可评估候选 2（未折叠显式 tanh，多 2 op，无收益）或复用 `T.vsigmoid`（文档 §3：lowering 为 VAdd/VSub/VExp/VDiv，与显式链同源）以省去 `vbrc(1)`——均列为 Stage 4 对照项，本版基线保持显式链（无对工具链语义的额外假设、已硬件验证）。
- **fp32 路径 vcast 陷阱**：见 §9.2（`T.vcast(fp32→fp32)` 损坏数据，须走 `T.copy`）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。物理核数必须 `NPUUtils.get().get_aicore_num()` 实查并记录查询代码与返回值（Cube/混合直接用返回值、纯 Vector 翻倍），禁止文档假设/经验值。

- **② 物理核数依据（实查）**：查询代码与实际返回值记录于此（禁止文档假设/经验值替代）：

```python
from tilelang.utils import NPUUtils
physical_cores = NPUUtils.get().get_aicore_num()   # 实查输出：20
vector_cores = physical_cores * 2                   # 纯 Vector 算子核数翻倍：40
```

实查命令与输出（2026-10-13 本机执行，toolchain tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B3）：`get_aicore_num()` 返回 **20**。本算子**纯 Vector → 核数翻倍 = 20 × 2 = 40**。（备注：`CONST-aicore-910B2C` 记录 910B2C 为 24，与实查 20 不一致——设备档/型号差异，**一律以实查为准**，不以 24 假设。）

- **① 逻辑核数**：`num_logical_kernels = M × ceil(N / block_N)`，随 workload（NPU manifest `examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `GeluTanhAndMulFwdOp`）变化：

| workload | M | N | dtype | n_num=ceil(N/2048) | 逻辑核数 M×n_num | 每核串行 tile 数 ceil(logical/40) |
|----------|---|---|-------|--------------------|------------------|----------------------------------|
| smoke | 16 | 128 | f32/f16/bf16 | 1 | 16 | 1（num_cores=16） |
| llama-8b-ffn-prefill | 2048 | 14336 | f16/bf16 | 7 | 14336 | 359 |
| llama-8b-ffn-decode | 1 | 14336 | bf16 | 7 | 7 | 1（num_cores=7） |

- **③ 规模判定与分核方案**：workload 跨越「无需适配」到「极大规模」两档，设计采用**统一 persistent 结构**（成熟 elementwise 形态，hardsigmoid/add/mish 同款）：

  - **小规模（逻辑核数 ≤ 40）**：`num_cores = num_logical_kernels`，每核 ≤1 tile，`tiles_per_core=1`，无串行循环、无负载不均——结论「**无需适配**」（逻辑核数 ≤ 物理核数）。覆盖 smoke（16）、decode（7）。
  - **极大规模（逻辑核数 > 40）**：固定启动内核数 = 40（= 物理向量核数），核内 `T.serial(tiles_per_core)` 串行处理多个逻辑块（`tiles_per_core = T.ceildiv(num_logical_kernels, 40)`，编译期常量，**循环边界静态**；prefill 359），摊薄核启动开销，**内核总数 = 物理核数整数倍（1×）**，无负载不均。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅 block 级 / tile 级调度循环，无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级并行 | 核级并行 | `T.Kernel(num_cores, is_npu=True) as (cid, _)` | 每核处理若干逻辑块（§5.5 分核） |
| tile 级迭代 | 核内串行 | `T.serial(tiles_per_core)` | persistent 串行处理多个逻辑块，边界静态（§5.5） |
| 元素级 | 向量化 | v-prefix 整块 API（vcast / vmul / vadd / vexp / vdiv / vbrc） | 块内逐元素由 Vector 单元并行（§1.6.2） |

### 6.2 循环伪代码

```python
with T.Kernel(num_cores, is_npu=True) as (cid, _):
    # UB buffer 分配（§4.3）
    for i in T.serial(tiles_per_core):          # 核内串行 tile 循环（静态边界）
        task = i * num_cores + cid
        if task < M * T.ceildiv(N, block_N):    # 尾任务 guard
            m = task // T.ceildiv(N, block_N)
            n0 = (task % T.ceildiv(N, block_N)) * block_N
            tail = T.min(block_N, N - n0)
            # 装载 -> v-prefix 链（显式 sigmoid tanh 门控）-> 写回（见 §3.3）
```

### 6.3 流水线优化

Stage 4 可选：对主路径采用 guard-free 2-stage `T.Pipelined(tiles_per_core, num_stages=2)` 双缓冲流水（mish perf_opt 形态），将向量链隐藏于 MTE2 窗口（PL-1.6-copy-floor：≥16M 档计算链被完全隐藏）。Stage 3 基线不启用（正确性优先，`T.serial` 顺序即可）。

### 6.4 尾块处理

非整除 `N`：`tail = T.min(block_N, N - n0)`；`T.copy` 仅装载/写回 `[0:tail]` 有效列；v-prefix 对全 `block_N` buffer 计算，垃圾尾部不写回（与 mish 同款）。`task < M*ceil(N/block_N)` guard 覆盖最后一波不完全 tile。manifest workload 的 `N` 均被 2048 整除（14336=7×2048），尾列路径服务任意 `N` 的通用情形。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer）。

### 7.2 同步点说明

无手动同步点。Developer 模式下编译器自动管理 GM↔UB 搬运与向量计算间的依赖（`T.copy` 与后续 v-prefix op、以及各 v-op 之间的数据依赖由自动同步保证），无跨核协作（逐元素无跨元素依赖，各核独立）。

### 7.3 pass_configs 配置

Developer 模式默认配置，无需额外 `pass_configs`。环境变量 `TILELANG_ASCEND_MODE = "Developer"`（可选，默认即 Developer；NPU wrapper 声明 `ascend_mode = "Developer"`）。无运行时分支（dtype 分支为 trace-time 选定）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：Golden 以 **§0.1 源算子语义**为唯一依据（单 `[M,2N]` 张量、gate/value 切半、**tanh 近似 GELU**、fp32 opmath、输出同 dtype），移植源仓测试基准 `examples/TileOPs/tests/ops/test_gelu_tanh_and_mul.py` 的 `ref_program`（CPU 显式 tanh 公式），**不复刻 §0.6 的 NPU 算法**（不含显式 sigmoid、fp32 链、persistent 结构）。

```python
import torch

def golden_gelu_tanh_and_mul(x):
    """GeluTanhAndMul 参考实现：x [M,2N] -> y [M,N] = gelu_tanh(gate)*value。

    与源仓测试基准一致（examples/TileOPs/tests/ops/test_gelu_tanh_and_mul.py 的
    ref_program：CPU 上用显式 tanh 公式，注意该 build 的设备 F.gelu(approximate=
    "tanh") 不可靠，故显式展开）。
    独立于 NPU 算法：仅依赖 torch 原生 tanh 与乘法，gate/value 切半由 torch 视角完成。
    """
    half = x.shape[-1] // 2
    xc = x.detach().float().cpu()
    gate = xc[..., :half]
    value = xc[..., half:]
    inner = (2.0 / 3.141592653589793) ** 0.5 * (gate + 0.044715 * gate ** 3)
    gelu = 0.5 * gate * (1.0 + torch.tanh(inner))
    return (gelu * value).to(x.dtype)
```

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-2 | 1e-2 |
| bfloat16 | 1.6e-2 | 1.6e-2 |

（与源仓测试 `_get_tolerances` 一致。kernel 用全 fp32 链 + 显式 sigmoid，对齐 torch fp32 opmath golden；§1.6.1 机器验证候选对 fp64 参照均 0 违反，对基线 max ≤0.431 tolerance-unit；硬件探针 fp32 1.9e-7 / fp16 9.4e-4 / bf16 6.8e-3。）

### 8.3 L0 门槛测试计划

L0 门槛（Stage 3 必过）：`M=2048, N=14336`（manifest 主 workload），`float16` / `bfloat16` 两 dtype，以及 `M=1, N=14336`（decode）与 smoke `(16,128)` 三 dtype，`randn` 输入（gate/value 两半独立 randn），vs golden（§8.1 显式 tanh 公式）在对应容差下 `torch.testing.assert_close`。完整分层套件（L1 非整除 `N`、多 shape；L2 边界 size；Boundary 特殊值含 NaN/±inf/−0.0/subnormal）交由 `tilelang-op-develop` 规划，本设计不枚举。

**特别提醒（tanh 全域与边界语义验证）**：Boundary 用例必须覆盖 ① 大 `|gate|` 饱和（`u(±inf)=±inf`；**预期 `gelu_tanh(-inf)=NaN`**——源式 `-inf*0`，与 §8.1 golden 一致，**不得断言 0**；`gelu_tanh(+inf)=+inf`；另可由大负有限值 `gelu_tanh(-大值) → -0.0` 作饱和验证点）；② **`|u|>0.5` 全域精度**（`T.vtanh` 盲区——须覆盖 `|u|` 至 ~12 附近的样本，确认显式 sigmoid 不漏精度）；③ `NaN` 输入 → NaN 输出（§0.1 语义层要求）；④ fp32 精度（1e-5 容差对显式 sigmoid 精度的实测标定）；⑤ `-0.0` 输入 → `-0.0` 输出。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 说明 |
|------|------|
| **`T.vtanh` 不是真 tanh（本工具链，按模式两条路径）** | `tl.npuir_vtanh` 按 `TILELANG_ASCEND_MODE` 分派（进程级、trace-time 读取，CG-2026-0013）：**默认/Expert** → `CodeGenTileLangNPUIRAPI::VtanhCodegen`（`codegen_npuir_api.cc`）为**截断的 4 项 Maclaurin 级数** `tanh(x) ≈ x − x³/3 + 2x⁵/15 − 17x⁷/315`，仅 `\|x\|≲0.5` 有效（实测 `vtanh(2)=3.31` vs `0.964`、`vtanh(3)=−91.63`、`vtanh(±inf)=NaN`）；**Developer** → `CodeGenTileLangNPUIRDEV::VtanhCodegen`（`codegen_npuir_dev.cc`）为 exp 恒等式 `(e^{2x}−1)/(e^{2x}+1)`，有限 `\|x\|` 忠实但 `+\|x\|>~44` 溢出、`vtanh(+inf)=NaN`（实测）。`T.tanh` 为 TIR 标量表达式（无向量形态）。**本设计全篇禁用 `T.vtanh`/`T.tanh`**，改用恒等式折叠 + 显式 sigmoid（§0.6 R2）；Stage 3 若误用默认路径会在 L0（`\|u\|>0.5` 样本）立即失败，Developer 路径则在 `+inf`/大 `gate` 边界失败 |
| 显式 sigmoid 精度未在 `vmul/vadd/vexp/vdiv` 上逐指令实测（硬件探针已整链验证） | 设计期以 torch fp32（模拟 vmul/vadd/vexp/vdiv 的舍入）为代理，`verify_equiv.py` `RANGE` 段全域 max abs err 1.78e-7（tanh）/ 3.78e-6（折叠 gelu）；硬件探针整链实测 fp32 1.9e-7（含 exp/div）。若 Stage 3 发现硬件 `vexp`/`vdiv` 舍入导致超差，可改用 `T.vsigmoid`（同源 lowering）或提高中间精度，风险低 |
| **`T.vcast(fp32→fp32)` 损坏数据** | 硬件探针实证：对 fp32 buffer 调 `T.vcast` 使输出错误（阈值化现象）；fp32 输入/输出路径必须用同 dtype `T.copy`（§3.3）。仅跨精度（fp16/bf16↔fp32）用 `T.vcast` |
| bf16 无原生 v-prefix 算术 | 多数 v-prefix op（vmul/vadd/vexp/vdiv）dtype 表 bf16 标 `×`；bf16 路径必经 fp32 中转（`T.vcast` 双端），本设计已覆盖（§3.3） |
| UB 预算 | 92–120KB < 192KB（§4.5），余量 ≥30%；`block_N` 可上调 |
| 发射项低 | 显式 sigmoid 链 ~15 向量 op（远低于同族 gelu_and_mul 的 deg20 erf ~55 op）；发射项 ~7.5µs，主 workload（prefill）仍带宽受限（226µs） |
| 分核 | 物理核数实查 20（×2=40），与 `CONST-aicore-910B2C`（24）不一致——本环境设备返回 20，以实查为准 |
| 知识库提示 | `TRAP-threads-kwarg-noop`（`threads=` 无效）已促 R1 折叠；`CG-2026-0021`（`T.verf` 截断）促 R2 对 `T.vtanh` 做同款核查（同族工具链陷阱）；`CONST-vector-launch-overhead`（宽块 op 优先）指导 `block_N` 取值 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 误用 `T.vtanh`/`T.tanh` | 依赖「NPU 有原生 tanh」直接调 `T.vtanh` | 默认路径域外发散（`\|x\|>0.5`）使 L0/L1 全失败（实测 9e1 量级）；Developer 路径在 `+inf`/大 `gate`（`\|u\|>~44`）返回 NaN | **禁用**；用 §3.3 恒等式折叠 + 显式 sigmoid |
| `T.vcast` 用在 fp32→fp32 | fp32 输入/输出路径照搬窄 dtype 的 vcast | 输出错误（阈值化），精度测试失败 | fp32 路径改 `T.copy`（同 dtype）；仅跨精度转换用 `T.vcast` |
| 用 `(e^{2u}−1)/(e^{2u}+1)` 求 tanh | 直译数学定义 | `u→+∞` 时 `∞/∞=NaN`，`gelu_tanh(+inf)` 变 NaN（违反边界） | 用候选 1 折叠式 `g·σ(2u)`（`1/(1+e^{−2u})` 自饱和） |
| sigmoid 被除数写成标量 | `T.vdiv(1.0, den, s)` | 编译失败（vdiv 仅允许第二操作数为标量） | 用 `T.vbrc` 填 ones buffer 作被除数（§3.3 步骤 10） |
| bf16 直接调 v-prefix 算术 | 对 bf16 buffer 调 `vmul`/`vadd`/`vexp`/`vdiv` | 编译失败（dtype 表 bf16 ×） | 先 `T.vcast(bf16→fp32)` 中转（§3.3） |
| UB 溢出 | `block_N` 过大（fp32 路径全 buffer 链） | 编译失败（`ub overflow`） | 维持 `block_N=2048`；fp32 路径复用 buffer（§4.5） |
| 尾列越界 | `T.copy` 用 base+size 形态写尾列 | MTE 越界写坏相邻 tensor | 双显式切片 `T.copy(out_ub[0:tail], y[m, n0:n0+tail])`（TRAP-T-copy-region-semantics） |
| value 列偏移写错 | 误把 value 读自 `x[m, n0:n0+tail]` | 输出错误（用 gate 当 value） | value 列起点恒为 `N+n0`（§3.3/§4.3） |
| `dtype=` 关键字 false alarm | `T.Tensor((M,2*N), dtype=...)` | 编译报 `name 'in_dtype' is not defined` | 位置参数传 dtype（design-template §3.3） |

### 9.3 特殊场景处理

- **非整除 `N`**：尾列 `tail` + `task` guard（§6.4）。
- **极小 shape**：`N < block_N` 时单列块；`M=1` 时逻辑核数 `=n_num`。
- **大 `|gate|` / 全域**：`u` 全域由显式 sigmoid 自饱和（`u→±∞ ⇒ gelu_tanh→g/0`）；`RANGE` 段机器验证。
- **NaN / ±inf 输入**：NaN→NaN（`vexp`/`vdiv` 传播）；±inf 经 sigmoid 饱和：`gelu_tanh(+inf)=+inf`，而 **`gelu_tanh(-inf)=NaN`**（源式 `(-inf)*0`，与 §8.1 golden 一致；机器自证见 `verify_equiv.py` 的 `BOUNDARY` 段）。
- **混合精度**：fp16/bf16 走 fp32 中转（vcast），fp32 直算（走 `T.copy`）；输出 dtype 与输入一致。
- **单张量契约**：Op 层不做 host 切分；gate/value 由 kernel 内 `T.copy` 列偏移读取（§0.1/§1.6.3）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/gelu_tanh_and_mul/gelu_tanh_and_mul_fwd_kernel/
├── DESIGN.md               # 本设计文档
├── verify_equiv.py         # §1.6.1 等价性机器验证脚本（含 RANGE / BOUNDARY）
├── RETROSPECTIVE.md        # Stage 1 复盘（Phase 8）
├── repro/
│   └── TRAP-vtanh-truncated-taylor.py   # T.vtanh 截断（硬件 repro）
└── history_version/
    ├── design_probe_vtanh_truthfulness.py        # D-3 探针 #1
    └── design_probe_sigmoid_gelu_mechanism.py    # D-3 探针 #2
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 本设计文档（first_design） |
| `verify_equiv.py` | 已完成 | §1.6.1 等价性验证（42 行全 EQUIV_PASS + RANGE/BOUNDARY） |
| `RETROSPECTIVE.md` | 已完成 | Stage 1 复盘（Phase 8） |
| `repro/TRAP-vtanh-truncated-taylor.py` | 已完成 | `T.vtanh` 截断硬件 repro |
| `history_version/design_probe_*.py` | 已完成 | D-3 设计期探针（2 个） |
| `gelu_tanh_and_mul_fwd_kernel.py` | Stage 3 待实现 | kernel + golden + 分层测试 |

### 10.3 命名规范

- 项目目录名：`gelu_tanh_and_mul`；算子目录名：`gelu_tanh_and_mul_fwd_kernel`；接口函数：`gelu_tanh_and_mul_fwd_kernel`（manifest 键 `GeluTanhAndMulFwdOp`）。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py，含全域 tanh 验证）+ D-3 探针
2. ⬜ Golden 函数（§8.1）
3. ⬜ 算子实现（`gelu_tanh_and_mul_fwd_kernel`，恒等式折叠 + 显式 sigmoid）+ 与 Golden 精度比对（Stage 3，`tilelang-op-develop`）

---

## 11. 参考实现与追溯

- 源仓（GPU）：`/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/activations.py`（`GeluTanhAndMulFwdKernel`）、`_base.py`（`FusedGatedKernel`）、`_builders.py`（`make_fused_gated_explicit`）、`ops/elementwise/activations.py`（`GeluTanhAndMulFwdOp`）、`manifest/spec/elementwise_fused_gated.yaml`。
- NPU 迁移工作区：`examples/TileOPs/tileops/kernels/elementwise/gelu_tanh_and_mul/`（scaffold + 提取 factory）、`examples/TileOPs/tileops/ops/elementwise/gelu_tanh_and_mul.py`、`examples/TileOPs/tests/ops/test_gelu_tanh_and_mul.py`、`examples/TileOPs/benchmarks/ops/bench_gelu_tanh_and_mul.py`、`examples/TileOPs/tileops/manifest/elementwise.yaml`。
- 工具链证据：`tilelang-mlir-ascend/src/target/codegen_npuir_api.cc`（`VtanhCodegen`，默认/Expert：截断级数）、`codegen_npuir_dev.cc`（`VtanhCodegen`，Developer：exp 恒等式、`+inf`/溢出 → NaN）、`tilelang/language/customize_npuir.py`（`npuir_vtanh`）、`tilelang/language/tir/op.py`（标量 `tanh`）；`tilelang/engine/lower.py`（按 `TILELANG_ASCEND_MODE` 分派 codegen）。
- 同族先例：`examples/TileOPs/tileops/kernels/elementwise/gelu_and_mul/gelu_and_mul_kernel/gelu_and_mul_fwd_kernel_DESIGN.md`（erf 版）、`examples/TileOPs/tileops/kernels/elementwise/hardsigmoid/hardsigmoid_kernel/hardsigmoid_fwd_kernel_DESIGN.md`、`examples/TileOPs/tileops/kernels/elementwise/add/add_kernel/add_fwd_kernel_DESIGN.md`。
- 知识库：`.agents/evolution/capability-gaps.md`（CG-2026-0021 `T.verf` 截断、CG-2026-0022 torch_npu `F.gelu` 近似）；`.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`（`CONST-capacity-910B2C` / `CONST-vector-launch-overhead` / `CONST-mte2-degradation` / `CONST-aicore-910B2C`）。
- 本设计的两项能力缺口（建议登记）：① `T.vtanh` 截断（同 CG-2026-0021，可增补 occurrences）；② `T.vcast(fp32→fp32)` 损坏数据（新缺口，见 §9.2）。
