# GeluAndMulFwdOp 算子设计文档（Stage 1 — 迁移设计）

> 迁移来源：GPU 仓 `TileOPs-main`（`src/tileops/kernels/elementwise/activations.py` 的 `GeluAndMulFwdKernel`，复用 `_base.py` 的 `FusedGatedKernel`（`_get_effective_op_func`：`fused_op(gate, value) = activation_func(gate) * value`）、builder `_builders.py` 的 `make_fused_gated_explicit`、erf 多项式 `_erf.py`、NaN 工具 `_nan.py`）。
> 迁移目标：TileLang-NPUIR（Ascend NPU），接口函数 `gelu_and_mul_fwd_kernel`，`target="npuir"`。
> 迁移工作区：`examples/gelu_and_mul/gelu_and_mul_fwd_kernel/`（project=`gelu_and_mul`，op=`gelu_and_mul_fwd_kernel`，scenario=migration，mode=harness）。
> 本设计按 skill `tilelang-op-design` 的三问框架（Phase M0）→ 算法调研（Phase R）→ 硬件耦合性判定与 NPU 重设计（Phase M1）→ 通用设计流程（Phase 1–4）产出；programming_mode = `developer`，target = `npuir`。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 skill `tilelang-op-design` 的 `references/migration-analysis.md`）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。

### 0.0 相对上一版的关键调整（mode=revision，v0 → v1）

> 本版为针对 Stage 2 `REVIEW.md`（结论：**不通过**）的修订版；上一版冻结于 `history_version/design_v0.md`。仅改动检视指出的 **1 项阻塞 + 3 项建议**，其余章节保持 v0（v0 的机械复核 30/30 EQUIV_PASS、gated 列索引与源一致、`T.verf` 文档核验等结论均维持有效）。修订前已按检视要求**重新 Read 源**（`activations.py` `GeluAndMulFwdKernel.activation_func` L395-402、`_erf.py`、`_base.py` `FusedGatedKernel._get_effective_op_func`，及 `_nan.py`）逐项复核边界实际行为。

| # | REVIEW 问题 | 级别 | 涉及章节 | 本版处置 |
|---|------------|------|---------|---------|
| 1 | 断言 `gelu(-inf)=0`，与源实现（`x*half*(one+erf_val)` 在 `x=-inf` 处 `-inf*0=NaN`）及 §8.1 golden（`F.gelu(-inf)=NaN`）均不符 | **阻塞** | §0.1 / §0.6 R2 / §8.3 / §9.3 | 重新 Read 源确认：`erf(-inf)=-1` → `one+erf_val=0` → `x*half=-inf` → `-inf*0=NaN`；`F.gelu(-inf)=NaN`（fp32/fp16/bf16 实测）。四处断言全部更正为 **`gelu(-inf)=NaN`**、`gelu(+inf)=+inf`；§8.3 Boundary 预期值改为 NaN（保留 `+inf` 与「大负有限值 → `-0.0`」作饱和验证点） |
| 2 | `T.verf` 收益「→1 op」不精确（其 lowering 为 `arith::DivFOp + VMulOp + VAddOp` 组合） | 建议 | §0.6 R2 / §1.6.0 / §1.6.1 | 改为准确口径：「**DSL 层 1 次 `T.verf` 调用替代约 14 条向量 op 的显式多项式链**（lowered 后为 `arith::DivFOp`+`hivm::VMulOp`+`hivm::VAddOp` 少量向量 op 组合，非单条硬件指令）」；带宽受限主结论不变 |
| 3 | §1.6.1 fp32 多项式对照项（5.56）无 `verify_equiv.py` 产出对应行 | 建议 | §1.6.1 / `verify_equiv.py` | **补齐**：`verify_equiv.py` 新增 `counter fp32-poly` 段（fp32 多项式臂 vs fp64 参照，拒绝基线控制组，非采纳项），实测 **max 5.5634 tolerance-unit、5 shape 累计 199 违反**；§1.6.1 改引脚本实测值并列出逐 shape 分解 |
| 4 | `block_N = 256×8 = 2048` 表述对 fp32 不成立（fp32 GPU 折叠值为 256×4=1024） | 建议 | §0.3 / §0.4 / §5.2 / §3.5.2 | 更正为「2048 为**本平台 NPU 选定列块宽**，取自 fp16/bf16 GPU 折叠值 256×8；GPU 折叠值随 dtype 变（fp32 为 256×4=1024），2048 非源通用值」 |

**为何不会再犯同一错误（revision gate 要求）**：

- **根因**：v0 把「`erf(±inf)=±1` ⇒ `gelu(-inf)=0`」当作数学极限直觉结论，**未按源码实际求值顺序**（`x * half * (one + erf_val)` 的 `-inf * 0`）逐步复算，**也未与 §8.1 golden 交叉验证**——属「边界结论未经源式逐步代入 + 未与 golden 互证」。该错误还自相矛盾（同行已写「`x*0` 不特殊处理」，却给出 `gelu(-inf)=0`）。
- **本版纪律（可复用防线，已落入本设计）**：
  1. **边界语义必须按源式逐步代入复算**：任何 `±inf`/NaN/`−0.0`/subnormal 结论都要用**源表达式逐项代入**（而非数学极限），并显式列出产生该结果的每一步（本例：`x*half=-inf`、`one+erf_val=0`、`-inf*0=NaN`）。
  2. **边界结论必须与 §8.1 golden 实测互证**：设计阶段即对每个边界点用 golden（`F.gelu`）实跑比对，二者不一致即判为设计缺陷（本版已在 §0.1/§0.6/§8.3 标注「与 §8.1 golden 一致」）。
  3. **边界语义纳入机器自证**：`verify_equiv.py` 新增 `BOUNDARY` 自检，对**源式与 golden 双路径**均断言 `gelu(-inf)=NaN`、`gelu(+inf)=+inf`（三 dtype），使同类错误在 Stage 2 机械复核阶段即可被拦截，而非仅靠人工阅读。

### 0.1 源算子语义（做什么）

**数学语义**：融合门控激活（flashinfer `gelu_and_mul`）。输入是**单个** `x`，shape `[M, 2*N]`，沿最后一维切成两半：`gate = x[:, :N]`、`value = x[:, N:]`；输出 `y = gelu(gate) * value`（shape `[M, N]`），其中 `gelu` 为**精确 erf 形式**（非 tanh 近似）：

$$
\operatorname{gelu}(g) = g \cdot \frac{1}{2}\left(1 + \operatorname{erf}\!\left(\frac{g}{\sqrt{2}}\right)\right),
\qquad
y_{m,n} = \operatorname{gelu}\!\big(x_{m,n}\big)\cdot x_{m,\,N+n}
$$

源实现（`GeluAndMulFwdKernel.activation_func`，`activations.py` L395-402；融合由 `FusedGatedKernel._get_effective_op_func` 的 `fused_op(gate, value) = act(gate) * value` 构造，`_base.py` L458-467）：

```python
# GeluAndMulFwdKernel.activation_func（x 即 gate）
inv_sqrt2 = T.cast(INV_SQRT2, "float32")     # 1/sqrt(2)
half = T.cast(0.5, x.dtype); one = T.cast(1.0, x.dtype)
x_f32 = T.Cast("float32", x)
erf_val = T.Cast(x.dtype, erf(x_f32 * inv_sqrt2, x.dtype))   # erf 见 _erf.py
return x * half * (one + erf_val)                          # 窄 dtype 内乘积
# fused_op(gate, value) = activation_func(gate) * value
```

| 语义要素 | 结论（从源码解读得出） |
|---------|----------------------|
| 数学映射 | 逐元素：`y = gelu(gate)*value`；`gate`/`value` 是**同一张量 `[M,2N]` 的前后半**（非两个独立张量）；无跨元素依赖 |
| I/O 契约 | 输入**单张量** `x [M, 2*N]`（gate/value 沿最后维拼接）；输出 `[M, N]`，与输入同 dtype；无转置、无 in-place |
| 规约语义 | **无规约**，纯逐元素；无累加顺序问题 |
| dtype 语义 | 输入 `float16 \| bfloat16 \| float32`（`SUPPORTED_DTYPES = FLOAT_DTYPES`）；输出 `same_as(input)`。erf 的**参数与求值在 fp32**（`x_f32 = T.Cast("float32", x)`）；随后 `half`/`one`/`erf_val` 与 `x` 均为 `x.dtype`，**末尾乘积在窄 dtype 内**完成（两次舍入路径） |
| erf 计算（关键） | `_erf.erf(x, out_dtype)`：`out_dtype == "float32"` 时用 `T.erf`（fp32 内建，2 ulp）；**其余（fp16/bf16）用 clamps + 最高次在前的 10 次多项式**（`_CLAMP=3.6`、`_POLY_COEFFS` 11 个系数，`fminf/fmaxf` 钳位丢 NaN，故用 `keep_nan` 回填）——注释明言「erff 2 ulp fp32 精度只有 fp32 结果装得下，多项式指令数约为其一半」 |
| 边界语义 | `erf` 对 NaN 参数返回 NaN（`keep_nan` 恢复；torch.erf 语义）。`±inf`：`erf(±inf)=±1`，但 **`gelu(-inf)=NaN`**——源式 `x*half*(one+erf_val)` 在 `x=-inf` 处为 `(-inf)*(1+(-1)) = (-inf)*0 = NaN`（IEEE；`one+erf_val` 恰为 `0`）；`gelu(+inf)=+inf`（`(+inf)*(1+1)=+inf`）。与 §8.1 golden `F.gelu(-inf)=NaN`（fp32/fp16/bf16 实测一致）。`0`/`-0` 保持。`value=NaN/±inf` 经最终乘法按 IEEE 传播（`x*0` 不特殊处理——这正是 `gelu(-inf)` 为 NaN 的同一机制） |
| 空维度 / 非整除 | `M`/`N` 为构造期参数；`N` 由 `make_fused_gated_explicit` 的 `T.ceildiv(N, block_N)` 处理非整除尾列（`col` 超界不写） |

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（移植源仓测试基准 `examples/TileOPs/tests/ops/test_gelu_and_mul.py` 的 `ref_program = (F.gelu(gate) * value).to(dtype)`，fp32 opmath），**不复刻 NPU 算法**。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `[M, 2*N]` | `float16 \| bfloat16 \| float32` | 单张量；`x[:, :N]` 为 gate，`x[:, N:]` 为 value |
| `value`（切分视图） | `[M, N]` | 同 `x` | `x[:, N:]`，kernel 内以列偏移 `N+col` 读取，不物化 |
| `output`（y） | `[M, N]` | `same_as(x)` | 输出；`y = gelu(gate) * value`，无转置 |

> Op 层契约（`examples/TileOPs/tileops/ops/elementwise/gelu_and_mul.py` 的 `GeluAndMulFwdOp`）只接收/校验/转发这一个 `[M, 2N]` 张量；**不在 host 侧做切分**（切分是 kernel 内的列偏移索引）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句均已归入某步骤，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式部分 |
|------|------|------|------|-----------------|
| 1（host/Op） | 校验 `x.shape == [M, 2N]`、dtype、`x.contiguous()` | `x [M,2N]` | 同一张量（视图） | I/O 契约 |
| 2（loader） | 逐元素取 `gate = x[by, col]`、`value = x[by, N+col]` | `x` | `gate, value`（标量） | 门控切分 |
| 3 | `gate_f32 = fp32(gate)` | `gate` | `gate_f32` | erf 参数域提升 |
| 4 | `erf_val = cast_dtype(erf(gate_f32 * inv_sqrt2, gate.dtype))` | `gate_f32` | `erf_val`（窄 dtype） | `erf(g/√2)` |
| 5 | `gelu = gate * half * (one + erf_val)`（窄 dtype） | `gate, erf_val` | `gelu` | `g·½·(1+erf)` |
| 6 | `y[by, col] = gelu * value`（窄 dtype） | `gelu, value` | `y` | `gelu(gate)·value` |
| 7（Op） | 返回 `y [M,N]` | `y` | 输出 | I/O 契约 |

其中 `erf(...)` 展开为（`_erf.py`）：fp32 参数 → 若 `out_dtype=="float32"` 用 `T.erf`；否则 `clamp(x,-3.6,3.6)` → `w = 1-(t/3.6)^2` → 10 次 Horner 多项式 → `clip(±1)` → `keep_nan` 恢复 NaN。

**数据流与内存访问模式**（源 GPU 硬件视角）：

```
GM[x (M,2N)] --逐元素读取(x[by,col] / x[by,N+col])--> 寄存器 gate,value
  --fp32 提升--> erf 参数 --多项式/T.erf--> erf_val(窄 dtype)
  --窄 dtype 乘加--> gelu --* value--> y --写出--> GM[y (M,N)]
```

无 SMEM 分块（`explicit_parallel` 策略逐元素直读 GM；`FusedGatedKernel.DEFAULT_STRATEGY = "explicit_parallel"`）。

**循环与并行结构**：grid 二维 `T.Kernel(T.ceildiv(N, block_N), M, threads=256)`（`make_fused_gated_explicit`；`block_N = threads × num_per_thread`，GPU 默认 `npt = max(MIN_NUM_PER_THREAD, BYTES_PER_THREAD//elem_bytes)`，故 **fp16/bf16 为 256×8 = 2048、fp32 为 256×4 = 1024**——折叠值随 dtype 变，非单一 2048）；block 内 `T.Parallel(threads, npt)` 两级并行；`col = (bx*threads + i)*npt + j`。`by` 是行索引。

**host 侧逻辑**：`GeluAndMulFwdOp.forward` 仅 `x.contiguous()` + 校验 + 调 kernel；`GeluAndMulFwdKernel.forward` 调编译后 callable，`_restore_output_dtype`。无广播/reshape/crop；**不改输入张量内容**（满足 ascend-constraints.md §4）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | `explicit_parallel` 逐元素直读（`T.Parallel(threads, npt)`） | 尾轴连续、单遍 | 每线程 `npt` 元素连续读写 | CUDA 线程模型 / coalescing | 硬件强相关 |
| 2 | `threads × npt` 两级并行执行模型 | 每线程多元素放大吞吐 | thread/warp 组织，`block_N = threads×npt`（fp16/bf16 256×8=2048；fp32 256×4=1024） | CUDA 线程模型 | 硬件强相关 |
| 3 | gate/value 同张量列偏移索引（`x[by,col]`、`x[by,N+col]`） | 免切分物化、单输入契约 | 行内两段连续访存 | 无（地址算术） | 可移植 |
| 4 | **erf 的 clamps + 降次多项式近似**（`_erf.py`） | fp16/bf16 下避免昂贵的 erff、复用单精度多项式 | `clamp`→`w`→10 次 Horner→`clip` | CUDA `fminf/fmaxf`（丢 NaN）+ 大寄存器堆 | 硬件强相关 |
| 5 | fp32 参数域求值（`x_f32`）、末尾窄 dtype 乘积 | 提升 erf 参数精度 | `T.Cast("float32", x)` 后求 erf | 无（纯精度策略） | 可移植 |
| 6 | `keep_nan`（一次 `isnan` + `Select`） | 恢复 `fminf/fmaxf` 丢掉的 NaN | 对参数一次 select 回填 | CUDA `isnan` | 硬件强相关 |
| 7 | 常数 `INV_SQRT2` 预计算（`T.cast(...,"float32")`） | 除法/开方转常数乘 | 编译期常数 | 无（纯算法层） | 可移植 |
| 8 | `_erf` 按 `out_dtype` 分派（fp32→`T.erf`，其余→多项式） | 各 dtype 取最合适 erf 路径 | 编译期分支 | CUDA `T.erf`/`erff` | 硬件强相关 |
| 9 | factory `@lru_cache(maxsize=32)` | 每 `(M,N,dtype)` 特化缓存 | host 缓存 | 无 | 可移植 |

> 源实现无 swizzle、无 mma、无 warp shuffle、无 online 算法（逐元素无跨元素依赖，天然单遍）。识别不出来 ≠ 不存在——上表为 `activations.py` + `_base.py`（`FusedGatedKernel`）+ `_builders.py`（`make_fused_gated_explicit`）+ `_erf.py` + `_nan.py` 的逐项核对结果。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级（语义/算法/优化） | 源硬件依赖 | NPU 有等价能力？ | 处置（保留/等价替换/重新设计/舍弃） | NPU 对应方案 / 依据 |
|------|----------------------|-----------|-----------------|-----------------------------------|---------------------|
| 计算语义 `y=gelu(gate)*value`（精确 erf、NaN 传播、输出同 dtype） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| 逐元素单遍、无规约、无跨元素依赖 | 算法 | 无 | — | **保留** | 纯算法层（migration-analysis.md §5.3「online/分块累加 → 保留」同类） |
| 常数 `INV_SQRT2` 预计算（#7） | 算法 | 无 | — | **保留** | 纯算法层恒等变形（编译期常数） |
| fp32 参数域求值 + 末尾精度策略（#5） | 算法 | 无 | — | **保留→强化为完整 fp32 域链 + 末尾单次 cast** | 对齐 torch fp32 opmath golden（PL-1.17-subfp32-fp32-transit）；bf16 无原生 v-prefix 算术（§0.6 R2 / §1.6.1 项 2） |
| gate/value 同张量列偏移索引（#3） | 优化 | 地址算术 | 有 | **等价替换** | `T.copy` 二维切片 `x[m, N+n0 : N+n0+tail]`（`docs/Tilelang.language/内存操作/T.copy.md`；`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` 混合标量/切片先例） |
| `threads × npt` 两级并行 + `T.Parallel(threads,npt)`（#1/#2） | 优化 | CUDA 线程模型 | 无直接等价（一维 Kernel；`threads=` 在 npuir 无效果） | **重新设计** | → §0.6 R1（`block_N` 折叠 + 一维 `T.Kernel` + UB 向量化） |
| fragment/寄存器分块装载、coalesced 连续访问 | 优化 | SMEM/寄存器/coalescing | 有（UB + MTE 连续搬运） | **等价替换** | `T.alloc_ub` + `T.copy` 连续切片（GM→UB→GM），ascend-constraints.md §1 / decision-tree.md §3 |
| erf 的 clamps + 降次多项式近似 + `keep_nan`（#4/#6） | 优化 | CUDA `fminf/fmaxf` + 大寄存器堆 | 有（原生向量 erf） | **重新设计** | → §0.6 R2（`T.verf` 原生向量 erf；NaN 由 verf 传播，必要时按 hardsigmoid 先例恢复） |
| `_erf` 按 `out_dtype` 分派（#8） | 优化 | CUDA `T.erf`/`erff` | 有（`T.verf` fp16/fp32） | **重新设计** | 并入 R2：统一走 `T.verf`（fp32 求值，bf16 经 fp32 中转） |
| factory `@lru_cache`（#9） | 优化 | 无 | 有 | **等价替换** | NPU 侧同款 `@functools.lru_cache(maxsize=32)` |
| swizzle / mma / warp shuffle | 优化 | SMEM bank / tensor core / warp | 不存在于本算子 | **不适用** | 源实现无此三者，无需处置 |

> 统计：**保留 4 项、等价替换 4 项、重新设计 2 项、舍弃 0 项**。每项依据：migration-analysis.md §5.3 映射表条目 + `examples/` 同族迁移佐证（hardsigmoid / add）+ `docs/Tilelang.language/` v-prefix API 条目。无「舍弃」项（erf 多项式被重新设计承接，其「避免昂贵 erff、指令数可控」意图由原生 `T.verf` 更好承接，见 R2）。

### 0.6 NPU 算法重设计

**重设计项 R1：并行结构重设计（二维 `T.Kernel(ceildiv(N,block_N), M, threads=256)` + `T.Parallel` → 一维 `T.Kernel` + `block_N` 折叠 + UB 向量化）**

- **源方案**：二维 grid（列块 × 行 `M`）+ `threads=256`，block 内 `T.Parallel(threads, npt)` 逐元素；`block_N = threads×npt = 2048`。`T.Kernel` 三维/二维与 `threads=` 在 NPU 均不可用（ascend-constraints.md「仅支持一维 Kernel」；`TRAP-threads-kwarg-noop`：`T.Kernel(..., threads=)` 在 npuir 仅消费 `grid_size`）。
- **NPU 新算法**：一维 `T.Kernel(num_cores, is_npu=True) as (cid, _)`；把 GPU 的 `(列块, 行)` 二维任务空间**折叠为一维任务序** `task = m * n_num + n0/block_N`（`n_num = ceildiv(N, block_N)`，**行优先**使相邻任务访问相邻内存）；元素级并行交给 Vector 单元（v-prefix 整块 API 对 UB buffer 计算，天然向量化）；分块搬运用 `T.copy`（GM→UB→GM）。大规模（逻辑核数 > 物理向量核数）用核内 `T.serial` persistent 串行（§5.5）。
- **语义保持论证**：逐元素映射不变（每输出元素独立，无跨元素依赖、无规约）——`threads/npt/二维 grid` 到 `block_N/一维核` 的任务重排不改变任何输出值，每个 `y[m,n]` 的表达式与求值顺序仍与源码逐步一致（在 fp32 域，见 R2/项 2）；边界语义：尾列（`N % block_N ≠ 0`）仅 `T.copy` 有效列，v-prefix 对全 buffer 计算、垃圾尾部不写回，与源 `col` 越界不写等价。

**重设计项 R2：erf 求值重设计（CUDA clamps + 降次多项式 → NPU 原生向量 `T.verf`；并统一 fp32 域与末尾单次 cast）**

- **源方案**：`_erf.erf` 对 fp16/bf16 用 `clamp(x,-3.6,3.6)` → `w=1-(t/3.6)^2` → 10 次 Horner 多项式 → `clip(±1)` → `keep_nan`（一次 `isnan`+`Select`）恢复 `fminf/fmaxf` 丢掉的 NaN；对 fp32 用 `T.erf`。这是 GPU 上「无廉价高精度窄 dtype erff」的**硬件特化规避手段**，约 14 个向量指令。
- **NPU 新算法**：NPU 提供**原生向量误差函数** `T.verf`（文档 `docs/Tilelang.language/数学操作/T.verf.md`：`T.verf(src, dst)`，定义 `erf(x)=(2/√π)∫e^{-t²}dt`，值域 `(-1,1)`，支持 **fp16/fp32**、bf16 `×`；lower 为 `arith::DivFOp + hivm::VMulOp + hivm::VAddOp` 组合）。据此：① 全程在 **fp32 域**求值（`g32 = fp32(gate)`，`T.verf(g32 * INV_SQRT2)`），彻底删去多项式、`clamp/w/clip` 与 `keep_nan` 链；② 末尾 `g*0.5*(1+erf)*value` 全在 fp32 完成后**一次 `T.vcast` 落回输出 dtype**（对齐 torch golden 的 fp32 opmath，并避免 GPU 窄 dtype 尾部乘加的二次舍入）；③ bf16 因 `T.verf` 不支持 bf16，经 fp32 中转（`T.vcast(bf16→fp32)` → `T.verf` fp32 → 链 → `T.vcast(fp32→bf16)`）。
- **语义保持论证**：
  - **数学等价（容差内）**：两臂均计算 `erf(g/√2)`；多项式是最坏 1.7e-5 绝对误差的近似，`T.verf` 是硬件原生近似，二者在输出 dtype 容差内等价（§1.6.1 项 1 机器验证：fp16/bf16 相对多项式基线 max ≤0.14 tolerance-unit、0 违反）。**fp32 路径源码本就用精确 erf（`T.erf`），换 `T.verf` 是同类硬件 erf 替换**，不引入多项式误差。
  - **精度路径澄清**：源「erf 结果 cast 回窄 dtype + 窄 dtype 乘积」→ 新「全 fp32 链 + 末尾单 cast」，数学等价、舍入更接近 golden（§1.6.1 项 2 机器验证：max ≤0.37 tolerance-unit、0 违反）。
  - **边界语义核对**：NaN —— `torch.erf` 对 NaN 返回 NaN；NPU `T.verf` 的 NaN 语义未文档化，按 fallback 计划保留 NaN 传播（若 Stage 3 实测 verf 丢 NaN，则按 hardsigmoid 先例 `vcmp(x,x,"ne") + vselect` 单点恢复，见 §9.1）；`±inf` —— `erf(±inf)=±1`，故 `gelu(+inf)=+inf`（`+inf*2`），而 **`gelu(-inf)=NaN`**——源式 `x*half*(one+erf_val)` 在 `x=-inf` 处 `one+erf_val=0`、`x*half=-inf`，`-inf*0=NaN`（IEEE），与源实际行为及 §8.1 golden `F.gelu(-inf)=NaN` 一致（该结论已由 `verify_equiv.py` 的 `BOUNDARY` 自检对源式 + golden 双路径三 dtype 机器核验）；`-0.0`/subnormal 由 fp32 域保留。
  - **验证方式**：§8.1 golden（§0.1 语义）+ L0 精度对比（容差见表 §8.2）；`verify_equiv.py` 机器验证（§1.6.1）。
- **源优化手段意图承接**（§0.4）：#4/#6/#8（多项式/keep_nan/erf 分派）的「在窄 dtype 上低成本求 erf」意图由 `T.verf`（原生向量 erf；DSL 层 1 次调用替代约 14 条显式多项式向量 op，lowered 后为 `arith::DivFOp`+`hivm::VMulOp`+`hivm::VAddOp` 少量向量 op 组合）更好承接；#1/#2（连续逐元素并行）意图由 UB `T.copy` + v-prefix 整块计算承接；#3 列偏移由二维切片承接；#5/#7（fp32 域、常数）保留并强化。无静默丢弃。

> §0.5 全部「重新设计」条目（R1/R2）已重设计；其余为保留/等价替换。§1–§11 均基于 R1/R2 落地后的 NPU 算法展开。

### 0.7 标杆实现

- 源算子路径：`/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/activations.py`（`GeluAndMulFwdKernel.activation_func`）；`/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/_base.py`（`FusedGatedKernel._get_effective_op_func`）；`/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/_builders.py`（`make_fused_gated_explicit`）；`/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/_erf.py`（`erf` 多项式）；`/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/_nan.py`（`keep_nan`）；`/mnt/workspace/TileOPs-main/src/tileops/ops/elementwise/activations.py`（`GeluAndMulFwdOp`）。
- 源 manifest 条目：`/mnt/workspace/TileOPs-main/src/tileops/manifest/spec/elementwise_fused_gated.yaml`（`GeluAndMulFwdOp`，~L31）。
- NPU 侧提取参考（合并 factory）：`examples/TileOPs/tileops/kernels/elementwise/gelu_and_mul/_gelu_and_mul_fwd_kernels.py`（factory `gelu_and_mul_fwd_kernel(M, N, dtype, threads=256, num_per_thread=8)`，`activation_func` 已内联 + `* value`）。
- NPU wrapper/op/test/bench：`examples/TileOPs/tileops/kernels/elementwise/gelu_and_mul/gelu_and_mul.py`、`examples/TileOPs/tileops/ops/elementwise/gelu_and_mul.py`、`examples/TileOPs/tests/ops/test_gelu_and_mul.py`、`examples/TileOPs/benchmarks/ops/bench_gelu_and_mul.py`。
- 源仓参考实现 / 测试基准：`examples/TileOPs/tests/ops/test_gelu_and_mul.py` 的 `ref_program = (torch.nn.functional.gelu(gate.float()) * value.float()).to(dtype)`；容差见同文件 `_get_tolerances`（fp32 1e-5；fp16 1e-2；bf16 1.6e-2）。
- golden 以 §0.1 语义为依据，优先移植源仓参考实现，**不复刻 NPU 算法**（保证验证独立性）。
---

## 1. 概述

### 1.1 算子名称

GeluAndMulFwdOp（manifest 键）；迁移目标 TileLang 接口函数 `gelu_and_mul_fwd_kernel`。

### 1.2 功能描述

融合门控精确 GELU 激活：输入单张量 `x [M, 2*N]`（`gate=x[:, :N]`、`value=x[:, N:]`），输出 `[M, N]`，`y = gelu(gate) * value`，`gelu` 为 erf 精确形式（`torch.nn.functional.gelu(approximate="none")`）。用于 Llama/Qwen 等 SwiGLU 风格 FFN 的中间激活与门控相乘。仅浮点 dtype（fp16/bf16/fp32）；NaN 传播为 NaN。

### 1.3 数学公式

$$
y_{m,n} = \operatorname{gelu}\!\left(x_{m,n}\right)\cdot x_{m,\,N+n},
\qquad
\operatorname{gelu}(g) = g\cdot\frac{1}{2}\left(1+\operatorname{erf}\!\left(\frac{g}{\sqrt{2}}\right)\right),
\qquad
m\in[0,M),\ n\in[0,N)
$$

（fp16/bf16 经 fp32 域求值后按输出 dtype 舍入一次。）

### 1.4 算法描述

单步逐元素映射（多输入 → 单输出）：Op 层把**单个** `[M, 2N]` 张量直接交给 kernel；kernel 以行优先的一维任务序处理输出 `[M, N]`，每个任务取行 `m` 的一段列块 `n0:n0+tail`：`gate` 读自 `x[m, n0:n0+tail]`，`value` 读自 `x[m, N+n0:N+n0+tail]`；在 fp32 域计算 `y = g32*0.5*(1+verf(g32/√2))*v32`，单次 cast 落回输出 dtype。无规约、无窗口、无矩阵、无跨元素依赖，**天然单遍**。本算法与 §1.6.0 调研选定算法一致。与源算法结构差异：① 并行组织（二维 grid + `T.Parallel` → 一维 Kernel + `block_N` 折叠，§0.6 R1）；② erf 求值（多项式/keep_nan → 原生 `T.verf`，§0.6 R2）；③ 精度路径（窄 dtype 尾部乘加 → 全 fp32 链 + 单次 cast，§0.6 R2 / §1.6.1 项 2）。计算表达式本身（精确 erf GELU 门控）与源语义一致。

### 1.5 数据流图

```
输入 x [M, 2N] --Op 校验/contiguous（不切分）--> kernel
  x[m, n0:n0+tail]     --T.copy--> UB gate_ub --vcast--> UB g32  (gate32)
  x[m, N+n0:N+n0+tail] --T.copy--> UB value_ub --vcast--> UB v32 (value32)
  g32 --vmul(inv√2)--> w32 --verf--> e32 --vadd(+1)--> e32 --vmul(×0.5)--> e32
      --vmul(×g32)--> e32(gelu) --vmul(×v32)--> e32(y32) --vcast--> out_ub
  out_ub --T.copy--> y[m, n0:n0+tail]  （y [M, N]）
```

### 1.6 算法调研与优化分析 ⭐

> 设计第一优先级：先调研（§1.6.0 四问）→ 数学等价优化（§1.6.1）→ 向量化替代（§1.6.2）→ 向量化轴/布局（§1.6.3）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：`y = gelu(gate)*value`（精确 erf 形式）的算法族——所有能算出相同结果的算法结构。源算法/输入公式只是基线候选之一。本算子为**单步逐元素（多输入 → 单输出）**，按 algorithm-research.md §2 判为**轻量调研**（四问逐问回答并给依据）。对照 algorithm-candidates.md 的 elementwise 命中行（R1：公共子表达式消除 / 除法转乘倒数 / 高代价算子降代换；R2：逐元素映射天然单遍；R3：融合消除中间 GM 往返）与 pattern-library 的 `PL-1.17-subfp32-fp32-transit`、`CONST-vector-launch-overhead`。知识库预注入消费：`CASE-norm-adalayern-migration`（迁移一次过纪律：原生指令替代+机器验证）、`TRAP-threads-kwarg-noop`（`threads=` 无效，促 R1 折叠）、`CG-2026-0003`（Expert fixpipe 版本配对缺口，与本 Developer 纯 Vector 路径无关）。

**R1 等价化简公式候选**（基线候选必须在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线（源算法） | `gelu(g)=g·0.5·(1+erf(g/√2))`；`y=gelu(gate)*value`；erf 用多项式（fp16/bf16）/`T.erf`（fp32） | — | — | ✅（基线） |
| 1 | erf 求值降代换：原生向量 erf（`T.verf`）替代 10 次多项式 | `e = verf(g32·inv√2)`（fp32） | 容差内等价（两者均为 erf 近似，硬件 erf 精度不低于 1.7e-5 多项式） | 少算：DSL 层 1 次 `T.verf` 调用替代约 14 条向量 op 的显式多项式链（lowered 后为 `arith::DivFOp`+`hivm::VMulOp`+`hivm::VAddOp` 少量向量 op 组合，非单条硬件指令） | ✅ |
| 2 | 常数乘（除法/开方转乘） | `g·(1/√2)`（源已含 `INV_SQRT2`） | 数学恒等（常数预计算） | 免运行时 sqrt/div（基线已含） | ✅（基线即此式；无结构差异，不单列） |
| 3 | 因子重排 `g·0.5·(1+e)·v` ↔ `0.5·g·(1+e)·v` | 结合/交换律 | 数学恒等（浮点结合序差 ≤1 ulp） | 无（op 数相同，仍 3 乘） | ❌（无结构差异） |
| 4 | **tanh 近似** `0.5g(1+tanh(√(2/π)(g+0.044715g³)))` | tanh-GELU | **不等价**（近似公式，非 erf 精确形式） | 少 1 次 erf（tanh 便宜） | ❌（语义错误：契约要求精确 erf，`GeluTanhAndMulFwdOp` 为另一 manifest 算子） |
| 5 | 合并 `0.5·(1+erf)` 与 `×gate`/`×value` 少一乘 | 任意结合 | 无法消去任一乘（`0.5`、`gate`、`value` 三因子独立） | 无 | ❌（无更少 op 的代数形式） |
| 6 | erf 的 clamp `±3.6` 饱和裁剪 | `erf(±3.6)≈±1` | 数学近似（源多项式已含，属近似内部） | 无（原生 verf 无需） | ❌（并入候选 1） |

**R2 在线算法**：**无「在线/离线」之分**（即天然单遍）——`gelu(gate)*value` 是逐元素映射，每个输出元素只依赖对应 gate/value，无跨元素依赖、无 running 统计量、无归约；源算法（§0.3）已单遍。不存在需要两遍扫描 / 中间缓冲的变体。结论明确「无（单遍即最优）」，结构依据：逐元素无跨元素依赖（algorithm-research.md §3 R2 结构判据）。

**R3 复杂度对比**（逐候选四口径；带宽受限算子访存/遍数主导，不得只比 FLOPs。manifest roofline：`flops = 6*M*N`，`bytes = 3*M*N*elem_bytes`）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线（多项式 erf，fp16/bf16） | 6·M·N（manifest 口径：div/erf/add/mul-half/mul-gate + mul-value） | 3·M·N·elem（读 x 的 2·M·N + 写 y 的 M·N，无中间 GM 往返） | 1 | O(block_N)（UB 驻留，无 GM 中间缓冲） | 完全并行（逐元素独立，无跨核归并） |
| 候选 1（原生 `T.verf`） | 6·M·N（erf 以 1 条向量指令计，FLOPs 口径不变） | 3·M·N·elem（同基线） | 1 | O(block_N) | 完全并行 |
| 候选 4（tanh 近似） | ~10·M·N（cube+mul+add+mul+tanh+…） | 3·M·N·elem | 1 | O(block_N) | 完全并行（但语义不等价，淘汰） |

**结论**：候选间 FLOPs / 访存 / 遍数 / 中间缓冲四口径基本同（带宽受限，访存主导）；**基线即最优访存下的最简门控式**；唯一有实质收益的候选是**候选 1（erf 求值降代换：原生 `T.verf` 替代多项式）**——它在同一访存下把 erf 段从 ~14 条显式向量 op 降为 1 次 `T.verf` 调用（lowered 后为少量向量 op 组合），其余代数结构不可再简。无并列争议。

**R4 硬件亲和性评估**（逐候选对照检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水可融合性 / 跨核结构；负向淘汰须附 `docs/` 与 `examples/` 佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（多项式 erf） | 逐元素 → Vector（vcast/vmul/vadd/vmin/vmax），落 Vector 正确；但 erf 段 10 次 Horner + clamps + keep_nan ≈ 14 向量 op，发射主导 | 工作集 = O(block_N) UB 驻留，≤56KB（§4.5） | 尾列连续；`block_N=2048` 为 32B 整数倍且整除向量宽（fp16/bf16 ×8、fp32 ×4）✓ | `ceildiv(N,block_N)`/`ceildiv(tiles,num_cores)` 均编译期常量（M/N 为 JIT 参数）✓ | 单步链可 `T.Pipelined` 双缓冲（Stage 4 可选） | ✅ 采纳但被候选 1 改进 |
| 候选 1（原生 `T.verf`） | 逐元素 → Vector；`T.verf` 原生向量 erf（DSL 单次调用，lowered 为少量向量 op 组合；`docs/Tilelang.language/数学操作/T.verf.md`，fp16/fp32 ✓、bf16 ×），与元素级算力匹配 | 同上（删除多项式中间缓冲后更小） | `T.verf` Shape 无特殊要求；尾轴同基线 ✓ | 同上 ✓ | 同上 | ✅ **采纳（并入选定算法）** |
| 候选 4（tanh 近似） | Vector 可算，但语义错误 | — | — | — | — | ❌ 淘汰：非精确 erf 形式，违反 §0.1 契约 |

**定量 roofline（D-2，轻量调研豁免容量项，流量项必算）**：采用 pattern-library/constants.md 实测常数条目（版本戳 `tilelang 0.1.2+ed787bb`（2026-09-07 build）/ CANN 8.5.0 / Ascend910B2C；本机实查 20 AI Core）：

- 流量项：GM 读写总字节 ÷ 有效带宽（`CONST-mte2-degradation` 按总流量档位取值）；
- 发射项：向量 op 数 × ~0.5µs/op（`CONST-vector-launch-overhead`；`T.verf` 1 次调用替代多项式 ~14 op 的显式链，发射项大幅下降）；
- 容量项：豁免（单步逐元素，无跨引擎中间矩阵往返，`CONST-store-fixpipe-gm-only` 不适用——无 Cube）。

```
估算下界(large prefill M=2048,N=14336, fp16, 总流量 176.16MB):
  max(流量 ≈226µs, 发射 ≈4.5µs) ≈ 226µs
  流量项: 输入 2·M·N·2B = 117,440,512B + 输出 M·N·2B = 58,720,256B = 176,160,768B ≈ 176.16MB;
          CONST-mte2-degradation 176M 档 ≈ 19.5 GB/s/core × 40 向量核 ≈ 780 GB/s → 226µs
  发射项: 链 vcast×2 + vmul(inv√2) + verf(1) + vadd(+1) + vmul(×0.5) + vmul(×gate) + vmul(×value) + vcast = 9 op × 0.5µs ≈ 4.5µs
  容量项: 豁免（单步逐元素，无跨引擎往返）
估算下界(small decode M=1,N=14336, bf16, 7 核):
  max(流量 ≈0.4µs, 发射 ≈4.5µs) ≈ 4.5µs（发射/启动主导；launch 可与搬运重叠 → 实测可能更低）
```

**调研结论**：选定算法族 = **逐元素门控精确 GELU（单遍，`y = g·0.5·(1+verf(g/√2))·v`，fp32 域求值 + 末尾单 cast，erf 用原生 `T.verf`）**。关键依据：R1 无更少 op 的代数形式（候选 3/5 无差异、候选 4 语义错误），唯候选 1（erf 降代换）有实质收益；R2 天然单遍；R3 四口径基线即最优访存、候选 1 在等访存下最优发射；R4 纯 Vector 亲和（`T.verf` fp16/fp32 原生、UB 驻留、32B 对齐、静态边界）。与基线结构差异：**代数表达式不变**，仅 erf 求值（多项式 → `T.verf`）与精度路径（窄 dtype → 全 fp32 + 单 cast）不同。「无更优替代」调研范围：algorithm-candidates.md elementwise 命中行、恒等变形自查（R1 候选 3/5/6）、模式库 `PL-1.17-subfp32-fp32-transit` 与 `CONST-vector-launch-overhead`、源码 §0.3/§0.4 的算法与优化手段。源优化手段（§0.4）在调研后被 §0.6 R1/R2 承接，无静默丢弃。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法；调研候选中初判采纳的项在此完成等价论证与收益量化（与 1.6.0 不重复论证）。含采纳项，故生成并执行 `verify_equiv.py`（结果表见下）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| 1 | erf 求值降代换：原生向量 `T.verf` 替代 GPU 降次多项式（fp16/bf16 路径；fp32 路径源码本就用精确 `T.erf`，换 `T.verf` 为同类硬件 erf 替换） | `e = ψ(g32·inv√2)`，`ψ = clamp/w/clip 10 次 Horner + keep_nan`（`_erf.py`） | `e = T.verf(g32·inv√2)`（原生向量 erf，fp32 求值） | **容差内等价**：两者均为 `erf` 的近似；多项式最坏绝对误差 1.7e-5（源注释），原生 erf 精度不低于此。机器验证（下表）fp16/bf16 相对多项式基线 max ≤0.14 tolerance-unit、0 违反、0 NaN 失配；对 fp64 参照均 0 违反 | DSL 层 1 次 `T.verf` 调用替代约 14 条向量 op 的显式多项式链（少 ~13 条显式向量 op；lowered 后 `T.verf` 自身展开为 `arith::DivFOp`+`hivm::VMulOp`+`hivm::VAddOp` 少量向量 op 组合，仍远少于 14 条 Horner 链）；`CONST-vector-launch-overhead` 下发射项 ~7µs → 明显下降 |
| 2 | 全 fp32 域链 + 末尾单次 cast（对齐 torch golden fp32 opmath） | `erf_val=cast(ψ, x.dtype); y = x·half·(one+erf_val)·value`（窄 dtype 内两次舍入；源 `activation_func`） | `y = cast_to(dtype)( g32·0.5·(1+verf(g32·inv√2))·v32 )`（fp32 乘加，一次舍入） | **容差内等价**：两式均实现 `gelu(gate)*value`，舍入实现不同，差 ≤1 ulp；新式对齐 golden（fp32 opmath）。机器验证（下表）fp32 逐位相同（0.0）、fp16 ≤0.11、bf16 ≤0.38 tolerance-unit、0 违反、0 NaN 失配；变体基线（窄链）对 fp64 参照 max 0.37 亦 0 违反 | 免 bf16 无原生 v-prefix 算术（唯一可行路径）；对齐 golden 使 L0 精度稳定；不加 op（链长不变） |

**机器验证结果表**（`examples/gelu_and_mul/gelu_and_mul_fwd_kernel/verify_equiv.py`，torch CPU + fp64 参照 + 随机/角点值（±inf/NaN/−0.0/subnormal/±max）+ 逐 dtype 舍入，已执行；shape ∈ {(16,128),(8,256),(1,128),(4,64),(32,128)}；每采纳项 15 行，共 30 行；另含 `counter fp32-poly` 对照 5 行与 `BOUNDARY` 边界自检）：

| 采纳项 | max ulp 差（tolerance-unit，cand vs 基线） | 违反率 | NaN 失配 | cand vs fp64 参照 | 结论 |
|--------|------------------------------------------|--------|---------|------------------|------|
| 1. 原生 `T.verf` 替代多项式 erf | fp32 0.0000 / fp16 ≤0.0521 / bf16 ≤0.1371 | 0 / 0 | 0 | fp16 ≤0.0002 / bf16 ≤0.0540 | EQUIV_PASS |
| 2. 全 fp32 链 + 末尾单 cast | fp32 0.0000 / fp16 0.1024 / bf16 ≤0.3720 | 0 / 0 | 0 | fp16 ≤0.0185 / bf16 ≤0.0540 | EQUIV_PASS |

> 对照项（`verify_equiv.py` 新增 `counter fp32-poly` 段，**拒绝基线控制组，非采纳项**）：**GPU 多项式 erf 基线在 fp32 下对 fp64 参照 max 5.5634 tolerance-unit、5 个 shape 累计 199 次违反**（逐 shape：`16×128`=34、`8×256`=44、`1×128`=4、`4×64`=8、`32×128`=109）——这正是源码对 fp32 改用精确 `T.erf` 而非多项式的原因，也是本设计 fp32 路径必须用原生 erf、不得沿用多项式的证据。全部 30 行**采纳项** EQUIV_PASS，无未通过项，可采纳。脚本随算子目录留存（Stage 2 维度 8 重跑复核：`total rows: 30   EQUIV_PASS: 30   not-pass: 0`，另有 5 行 fp32 多项式对照与 3 行 `BOUNDARY` 自检，`OVERALL: EQUIV_PASS`）。

**优化结论**：采纳 2 项（1 个求值降代换项 + 1 个精度等价项）。优化后公式为：

$$
y_{m,n} = \operatorname{cast}_{dtype}\!\Big( g_{32}\cdot \tfrac{1}{2}\Big(1 + \operatorname{verf}\big(g_{32}\cdot \tfrac{1}{\sqrt{2}}\big)\Big)\cdot v_{32} \Big),
\qquad g_{32}=\operatorname{fp32}(x_{m,n}),\ v_{32}=\operatorname{fp32}(x_{m,N+n})
$$

供 §3.1 公式拆解使用。**无其他可采纳优化项**：单次逐元素门控映射，无公共子表达式（erf 参数只用一次）、无除法/开方（`1/√2` 为编译期常数）、无 cast 链冗余（输入 1 次升 + 输出 1 次降）、无中间 GM 往返；tanh 近似因语义不成立被拒。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | gate 升 fp32（fp16/bf16） | 逐元素 `T.Cast("float32", gate)` | `T.vcast(gate_ub, g32, round_mode="rint")`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`；mish/lerp 佐证） | ✅ | — |
| 2 | value 升 fp32（fp16/bf16） | 逐元素 `T.Cast("float32", value)` | `T.vcast(value_ub, v32, round_mode="rint")`（同上） | ✅ | — |
| 3 | erf 参数 `g32·(1/√2)` | 逐元素 `gate_f32 * inv_sqrt2` | `T.vmul(g32, inv_sqrt2, w32)`（标量 src2；`docs/Tilelang.language/数学操作/T.vmul.md`） | ✅ | — |
| 4 | 误差函数 `erf(w32)` | 逐元素多项式（10 Horner + clamps + clip） | `T.verf(w32, e32)`（原生向量 erf；`docs/Tilelang.language/数学操作/T.verf.md`，fp16/fp32 ✓） | ✅ | — |
| 5 | `1 + erf` | 逐元素 `one + erf_val` | `T.vadd(e32, 1.0, e32)`（标量 src2，原地；`docs/Tilelang.language/数学操作/T.vadd.md`） | ✅ | — |
| 6 | `×0.5` | 逐元素 `half * (...)` | `T.vmul(e32, 0.5, e32)`（标量 src2，原地；`T.vmul.md`） | ✅ | — |
| 7 | `× gate`（gelu） | 逐元素 `gate * (...) ` | `T.vmul(g32, e32, e32)`（tensor×tensor，原地；`T.vmul.md` dst 可同 src） | ✅ | — |
| 8 | `× value`（门控乘） | 逐元素 `gelu * value` | `T.vmul(e32, v32, e32)`（tensor×tensor，原地） | ✅ | — |
| 9 | 输出降 dtype | 逐元素 `T.Cast(dtype, y32)` | `T.vcast(e32, out_ub, round_mode="rint")`（`T.vcast.md`） | ✅ | — |
| 10 | 任务行列解算 `m = task//n_num`、`n0 = (task%n_num)*block_N` | 标量 divmod | 无向量 API | ❌ | block 级任务映射索引，每任务一次，非逐元素热点（可接受理由类别①） |
| 11 | 尾列 `tail = T.min(block_N, N-n0)` | 标量 min | 无（`T.min` 标量形） | ❌ | 依赖动态 shape 的边界处理，每任务一次，非逐元素热点（可接受理由类别④；mish/hardsigmoid 同款尾块） |
| 12 | `num_cores`/`tiles_per_core` 分核元数据、Op 层 shape 校验 | trace-time Python 标量 | 无（编译期） | ❌ | host/trace 侧 shape/分核元数据计算，不在 kernel 计算环内（可接受理由类别②） |

**向量化结论**：核内逐元素计算（2 次升 cast / erf 参数乘 / **原生 `T.verf`** / `+1` / `×0.5` / `×gate` / `×value` / 降 cast）已**全部向量化**（替代 API 均有 `docs/Tilelang.language/` 文档佐证：vcast/vmul/vadd/verf）。保留 3 处标量：任务行列解算（任务映射）、尾列大小（动态边界）、分核/Op 元数据（host/trace 侧），均非逐元素热点，理由见上表。与 §6 循环结构一致（§6 不出现逐元素标量循环）。

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
| 2 | host 侧先切分为两个 `[M,N]` 张量（`x[:,:N]`、`x[:,N:]` 各自 contiguous）再逐元素 | 列轴 | host 两次切片/物化（value 需一次设备端拷贝） | 引入一次额外设备端物化流量（写 ~M·N + 读）+ 破坏单张量契约（ascend-constraints.md §4 host 不得改输入真实内容；`expand/contiguous` 仅视图可，物化 value 属多余搬运） | ❌（纯增流量，且与单输入契约不符） |
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

纯 Vector 逐元素算子：无 matmul（不涉及 Cube）、无归约、无跨核协作，数据流线性（GM→UB→计算→GM），无手动内存层级控制或手动同步需求。Developer 模式由编译器自动管理同步与 buffer 分配，与 mish / hardsigmoid / add 同族 elementwise 先例一致。源算子 NPU wrapper 亦声明 `ascend_mode = "Developer"`（`examples/TileOPs/tileops/kernels/elementwise/gelu_and_mul/gelu_and_mul.py`）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub` 显式指定 UB buffer（v-prefix / `T.verf` 要求 UB 驻留数据） |
| 计算方式 | v-prefix 整块逐元素 API（vcast / vmul / vadd / verf），无显式 `T.Parallel` 元素循环 |
| 同步 | 自动同步（Developer 模式编译器插入，无需手动 `set_flag`/`wait_flag`） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**；每步映射 v-prefix 向量 API，与 §1.6.2 向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 0 | gate 切片 `x[m, n0:n0+tail]`、value 切片 `x[m, N+n0:N+n0+tail]` | 行内两段连续切片（§1.6.3 方案 #1） |
| 1 | `g32 = cast(gate)`、`v32 = cast(value)`（fp16/bf16 时） | 升 fp32（fp32 输入跳过），对齐 torch golden 的 fp32 opmath |
| 2 | `w32 = g32 · (1/√2)` | erf 参数（常数乘） |
| 3 | `e32 = verf(w32)` | **原生向量 erf**（§1.6.1 项 1） |
| 4 | `e32 = 1 + e32` | `1 + erf` |
| 5 | `e32 = 0.5 · e32` | `×½` |
| 6 | `e32 = g32 · e32` | `gate · ½(1+erf) = gelu` |
| 7 | `e32 = e32 · v32` | `y = gelu · value`（门控相乘） |
| 8 | `out = cast(e32)`（fp16/bf16 时） | 降回输出 dtype（一次舍入） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0a | 装载 gate | `T.copy(x[m, n0:n0+tail], gate_ub[0:tail])` | 二维切片（标量行 + 列切片）→ 1-D UB；尾列 `tail` 裁剪（TRAP-T-copy-region-semantics） | Developer |
| 0b | 装载 value | `T.copy(x[m, N+n0:N+n0+tail], value_ub[0:tail])` | 列起点偏移 `N` | Developer |
| 1a | gate 升 fp32 | `T.vcast(gate_ub, g32, round_mode="rint")` | fp16/bf16 → fp32（fp32 走 `T.copy`） | Developer |
| 1b | value 升 fp32 | `T.vcast(value_ub, v32, round_mode="rint")` | 同上 | Developer |
| 2 | `w32 = g32·(1/√2)` | `T.vmul(g32, inv_sqrt2, w32)` | `inv_sqrt2` 为 fp32 标量 | Developer |
| 3 | `e32 = verf(w32)` | `T.verf(w32, e32)` | 原生向量误差函数 | Developer |
| 4 | `e32 = 1 + e32` | `T.vadd(e32, 1.0, e32)` | 标量 1.0，原地 | Developer |
| 5 | `e32 = 0.5·e32` | `T.vmul(e32, 0.5, e32)` | 标量 0.5，原地 | Developer |
| 6 | `e32 = g32·e32` | `T.vmul(g32, e32, e32)` | tensor×tensor，原地 | Developer |
| 7 | `e32 = e32·v32` | `T.vmul(e32, v32, e32)` | tensor×tensor，原地 | Developer |
| 8 | 降回输出 dtype | `T.vcast(e32, out_ub, round_mode="rint")` | fp32 → fp16/bf16（fp32 走 `T.copy`） | Developer |
| 9 | 写回 | `T.copy(out_ub[0:tail], y[m, n0:n0+tail])` | 二维切片，尾列裁剪 | Developer |

> fp32 输入路径跳过步骤 1a/1b/8（无 vcast，走同 dtype `T.copy`）。常数 `inv_sqrt2` 与 `1.0`/`0.5` 均以 fp32 传递。标量 operand 用原生 Python float（同 add 先例：NPU 二元/一元 op 前端只接受裸标量）。**无 `FusedGatedKernel` 的 `fused_op` 闭包**：NPU 侧 `activation_func(gate)*value` 直接内联展开为步骤 2–7。

### 3.3 计算伪代码

> 约定：`T.Tensor` 的数据类型以**第二个位置参数**传入（不用 `dtype=` 关键字），避免产生 `name 'in_dtype' is not defined` 类 false alarm（design-template §3.3）。

```python
import functools
import tilelang
import tilelang.language as T
from tilelang.utils import NPUUtils

INV_SQRT2 = 0.7071067811865476  # 1/sqrt(2)

@functools.lru_cache(maxsize=32)
def gelu_and_mul_fwd_kernel(M, N, dtype, output_dtype=None, block_N=2048, num_cores=None):
    out_dtype = output_dtype or dtype
    compute_dtype = "float32"
    n_num = T.ceildiv(N, block_N)                 # 每行逻辑列块数（编译期常量）
    num_logical_kernels = M * n_num               # 逻辑核数（行 × 列块）
    # 物理核数实查 ×2（纯 Vector），见 §5.5
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    if num_cores is None or num_cores <= 0 or num_cores >= num_logical_kernels:
        num_cores = min(num_logical_kernels, vector_cores)
    tiles_per_core = T.ceildiv(num_logical_kernels, num_cores)   # 编译期常量，静态边界

    inv_sqrt2 = T.cast(INV_SQRT2, "float32")

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
                e32 = T.alloc_ub((_block_N,), compute_dtype)
                w32 = T.alloc_ub((_block_N,), compute_dtype)
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

                        # --- erf 参数 + 原生向量 erf：gelu(gate) ---
                        T.vmul(g32, inv_sqrt2, w32)      # w32 = gate / sqrt(2)
                        T.verf(w32, e32)                 # e32 = erf(w32)  （原生向量 erf）
                        T.vadd(e32, 1.0, e32)            # e32 = 1 + erf
                        T.vmul(e32, 0.5, e32)            # e32 = 0.5 * (1 + erf)
                        T.vmul(g32, e32, e32)            # e32 = gate * 0.5 * (1 + erf) = gelu

                        # --- 门控相乘：y = gelu(gate) * value ---
                        T.vmul(e32, v32, e32)

                        # --- 降回输出 dtype（fp16/bf16）或同 dtype copy（fp32）---
                        if out_dtype != compute_dtype:
                            T.vcast(e32, out_ub, round_mode="rint")
                        else:
                            T.copy(e32, out_ub)

                        # --- 写回（尾列裁剪）---
                        T.copy(out_ub[0:tail], y[m, n0 : n0 + tail])

        return main

    return kernel(M, N, block_N, num_cores)
```

> 注：`g32`/`v32`/`e32`/`w32` 复用为 fp32 计算缓冲（`T.vadd`/`T.vmul` 支持 `dst` 与 `src` 同 buffer 原地操作；`T.verf` 用独立 `e32` 以免依赖原地 verf）。fp32 路径（`dtype==compute_dtype`）可将 `g32←gate_ub`、`v32←value_ub`、`out_ub←e32` 复用，驻留从 ≤56KB 降至 ≤32KB（§4.5）。`T.ceildiv(_N,_block_N)` 与 `tiles_per_core` 均为编译期常量，核内循环边界静态。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|---------|------|
| `T.verf(src, dst)` | `docs/Tilelang.language/数学操作/T.verf.md`（原生向量误差函数，fp16/fp32 ✓、bf16 ×，Shape 无特殊要求；`tilelang/language/customize_npuir.py` 的 `npuir_verf` 别名为 `verf`） | ✅ 已验证（文档读 + 源码读） |
| `T.vcast(src, dst, round_mode)` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md`；mish/lerp 佐证 | ✅ 已验证（文档读） |
| `T.vmul(src0, src1, dst)` | `docs/Tilelang.language/数学操作/T.vmul.md`（src2 可标量；dst 可原地；fp16/fp32 ✓、bf16 ×） | ✅ 已验证（文档读） |
| `T.vadd(src0, src1, dst)` | `docs/Tilelang.language/数学操作/T.vadd.md`（src 可标量；dst 可原地） | ✅ 已验证（文档读） |
| `T.copy`（切片，二维 + 标量行） | `docs/Tilelang.language/内存操作/T.copy.md`（切片语法 + rank 折叠规则 1/2；`_gqa_prefill_fwd_kernel.py` 混合标量/切片先例） | ✅ 已验证（文档 + examples） |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`；mish/gelu_and_mul wrapper | ✅ 已验证 |
| `T.Kernel(一维, is_npu=True) as (cid, _)` | mish / hardsigmoid / add 同族先例 | ✅ 已验证 |
| `T.ceildiv` / `T.min`（标量） | `docs/Tilelang.language/数学操作/T.ceildiv.md`；mish 尾块 `T.min` 先例 | ✅ 已验证 |
| `NPUUtils.get().get_aicore_num()` | `docs/开发指南.md` §3.3；本机实查（§5.5） | ✅ 已实查 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | Yes（源为二维 `T.Kernel(ceildiv(N,block_N), M)`） | 折叠为一维 `T.Kernel(num_cores, is_npu=True)`，行列并入任务序（§0.6 R1） |
| GPU 专用 API | Yes（`T.Parallel(threads,npt)`、`threads=`、CUDA `fminf/fmaxf` 多项式 erf、`keep_nan`） | 全部改写为 NPU v-prefix API + `T.verf` + UB 路径（§0.5/§0.6） |
| GEMM 非整除 | 不适用（无 GEMM） | — |
| L0C 溢出 | 不适用（无 Cube） | — |
| 分核策略 | Yes | §5.5 三要素（逻辑核数 / 物理核数实查 / 规模判定与分核方案） |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键技术差异汇总）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 维度 | 二维 `T.Kernel(ceildiv(N,block_N), M, threads=256)` | 一维 `T.Kernel(num_cores, is_npu=True)` | §0.6 R1 |
| 元素并行 | `T.Parallel(threads, npt)` 显式两级循环 | v-prefix 整块 API（Vector 单元隐式向量化） | §0.6 R1 |
| 内存分配 | 直接读 GM（`explicit_parallel`） | `T.alloc_ub` + `T.copy`（GM→UB→GM） | §0.5 等价替换 |
| erf 求值 | fp16/bf16：clamps + 10 次多项式 + `keep_nan`；fp32：`T.erf` | 原生向量 `T.verf`（统一 fp32 求值） | §0.6 R2 |
| 精度路径 | erf cast 回窄 dtype，尾部窄 dtype 乘加 | 全 fp32 链 + 末尾单 cast | §0.6 R2 / §1.6.1 项 2 |
| 门控切分 | `x[by,col]` / `x[by,N+col]` 逐元素列偏移 | `T.copy` 二维切片（gate/value 两段连续） | §0.5 等价替换 |
| `threads×npt` | fp16/bf16 256×8=2048、fp32 256×4=1024 | `block_N=2048`（NPU 选定列块，取窄 dtype 折叠值 256×8） | §5.2 |
| factory 缓存 | `@lru_cache(maxsize=32)` | `@lru_cache(maxsize=32)` | §0.5 等价替换 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/elementwise/gelu_and_mul/gelu_and_mul.py` | 最高（本算子 NPU scaffold） | `_register` wrapper、`ascend_mode="Developer"`、`SUPPORTED_DTYPES` 浮点集、单 `[M,2N]` 输入契约、`default_config` 返回 `{}`（block_size 由 factory 定） |
| `examples/hardsigmoid/hardsigmoid_fwd_kernel/DESIGN.md` | 高度相似（同族 elementwise 迁移设计） | 一维 Kernel + block_size 折叠、fp32 中转、分核三要素、UB 预算口径、`T.verf` 类激活先例、NaN 恢复先例 |
| `examples/add/add_fwd_kernel/DESIGN.md` | 高度相似（同族 elementwise 迁移设计） | 多输入契约、fp32 中转、persistent 分核、UB 预算、`verify_equiv` 结构 |
| `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py` | 高度相似（同族 unary 激活迁移） | Developer 模式、`T.alloc_ub`、`T.copy` 尾块、v-prefix 链、fp32 中转 |
| `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py` | 相似（同族 elementwise + fp32 中转） | fp32 中转模式（PL-1.17）、`T.min` 尾块 |

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
| `w32` | `(block_N,)` | float32 | UB | erf 参数 `g32·(1/√2)` |
| `e32` | `(block_N,)` | float32 | UB | erf/gelu/结果链（原地复用） |
| `out_ub` | `(block_N,)` | output dtype | UB | 输出 tile（UB→GM） |

### 4.4 内存搬运路径

```
GM[x (M,2N)] --T.copy(x[m,n0:n0+tail])-->      UB[gate_ub]  --vcast--> UB[g32]
GM[x (M,2N)] --T.copy(x[m,N+n0:N+n0+tail])-->  UB[value_ub] --vcast--> UB[v32]
  g32 --vmul(inv√2)--> UB[w32] --verf--> UB[e32] --vadd(+1)--> UB[e32]
      --vmul(×0.5)--> UB[e32] --vmul(×g32)--> UB[e32](gelu) --vmul(×v32)--> UB[e32]
  UB[e32] --vcast--> UB[out_ub] --T.copy(out_ub[0:tail])--> GM[y (M,N)]
```

纯 Vector：GM ↔ UB 两级，无 L1/L0A/L0B/L0C（无 Cube）。数据仅 GM→UB 一次读入、UB→GM 一次写出，无中间 GM 往返。

### 4.5 UB 内存预算

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `gate_ub` | (2048,) | fp16/bf16 | 4096（fp32 时 8192） |
| `value_ub` | (2048,) | fp16/bf16 | 4096（fp32 时 8192） |
| `g32` | (2048,) | fp32 | 8192 |
| `v32` | (2048,) | fp32 | 8192 |
| `w32` | (2048,) | fp32 | 8192 |
| `e32` | (2048,) | fp32 | 8192 |
| `out_ub` | (2048,) | fp16/bf16 | 4096（fp32 时 8192） |
| **总计（fp16/bf16 路径）** | | | **45056（44KB）** |
| **总计（fp32 路径，全 buffer 显式）** | | | **57344（56KB）** |
| **总计（fp32 路径，复用 g32←gate_ub + v32←value_ub + out_ub←e32）** | | | **32768（32KB）** |

对照目标平台 UB 容量 **192KB**（`CONST-capacity-910B2C`：UB=192KB/AIV；version stamp `tilelang 0.1.2+3a214cde` + CANN 8.5.0）。按该条目 ×1.10–1.12 上偏系数：

- **fp16/bf16 路径**：44KB ×1.12 ≈ 49KB < 192KB ✓，余量 ~74%。
- **fp32 路径（全 buffer 显式）**：56KB ×1.12 ≈ 63KB < 192KB ✓，余量 ~67%。
- **fp32 路径（复用方案，推荐）**：32KB ×1.12 ≈ 36KB，余量 ~81%。

结论：全部 dtype 路径均远不溢出，`block_N=2048` 有极大余量；Stage 4 可据此扫描更大的 `block_N`（受 192KB 约束，理论可到 ~12K 列块）。

### 4.6 动态轴定义

无动态轴。`M`/`N` 为 JIT 编译期参数（每 `(M,N,dtype)` 一特化，`@lru_cache`）；`N` 的编译期值驱动 `n_num`/`tiles_per_core` 静态边界。尾列由 `tail = T.min(block_N, N - n0)` 处理；不引入运行时动态轴。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector。

**判定依据**：仅逐元素运算（cast / 常数乘 / `T.verf` / 加 / tensor 乘），无 matmul、无归约，不涉及 Cube 单元。

### 5.2 Block 划分

```python
block_N = 2048              # NPU 选定行内列块宽（取自 fp16/bf16 GPU 折叠值 threads×npt=256×8）
n_num = T.ceildiv(N, block_N)   # 每行列块数
num_logical_kernels = M * n_num # 逻辑核数（行 × 列块）
```

- **选择理由**：2048 为**本平台 NPU 选定列块宽**，取自 fp16/bf16 GPU 折叠值 `threads(256) × npt(8)`（K11 迁移规则：CUDA `threads×npt` 折叠为单一 `block_size`，由 factory 参数传入）。注：GPU 折叠值随 dtype 变（`npt = max(MIN_NUM_PER_THREAD, BYTES_PER_THREAD//elem_bytes)`，**fp32 为 256×4=1024**），故 2048 是 NPU 设计取值而非源通用值；32B 对齐（2048×2/4B 均为 32B 整数倍）；整除向量宽（fp16/bf16 ×8、fp32 ×4）；整除 manifest `N=14336`（14336/2048=7，`N=12288`/2048=6）；UB 预算 32–56KB ≪ 192KB（§4.5），块宽足够使向量链发射开销摊薄（`CONST-vector-launch-overhead`：宽块 op 恒优于窄块 op 翻倍）。
- **块宽敏感性**：`block_N` 增大 → 任务数减少（MTE 指令数下降）、但单任务 UB 增大且小 `M` workload 的并行度下降；Stage 4 可扫描 `block_N`（受 UB 192KB 约束）与 `T.Pipelined`。

### 5.3 约束分析

- **对齐约束**：`block_N=2048`，fp16/bf16 尾轴 2048×2B=4KB、fp32 2048×4B=8KB，均 32B 对齐 ✓；整除向量宽 ✓。
- **UB 容量**：32–56KB < 192KB（§4.5）✓。
- **L0/L1/L0C 容量**：无 Cube 计算，不适用。

### 5.4 注意事项

- **非整除尾列**：`tail = T.min(block_N, N - n0)`，仅 `T.copy` 有效列进/出；v-prefix/`T.verf` 对全 buffer 计算，尾部垃圾元素不写回（与 mish 同款）。
- **极小 shape**：`N < block_N` 时 `n_num=1`，`tail=N`，单列块；`M=1` 时逻辑核数 `= n_num`（decode 场景 7 核）。
- **gate/value 偏移**：value 列起点恒为 `N+n0`（`x` 的第 2 段），与输出列 `n0` 同段长 `tail`。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。物理核数必须 `NPUUtils.get().get_aicore_num()` 实查并记录查询代码与返回值（Cube/混合直接用返回值、纯 Vector 翻倍），禁止文档假设/经验值。

- **② 物理核数依据（实查）**：查询代码与实际返回值记录于此（禁止文档假设/经验值替代）：

```python
from tilelang.utils import NPUUtils
physical_cores = NPUUtils.get().get_aicore_num()   # 实查输出：20
vector_cores = physical_cores * 2                   # 纯 Vector 算子核数翻倍：40
```

实查命令与输出（2026-10-13 本机执行，toolchain tilelang 0.1.2 dev-root + CANN 8.5.2）：`get_aicore_num()` 返回 **20**。本算子**纯 Vector → 核数翻倍 = 20 × 2 = 40**。（备注：`CONST-aicore-910B2C` 记录 910B2C 为 24，与实查 20 不一致——设备档/型号差异，**一律以实查为准**，不以 24 假设。）

- **① 逻辑核数**：`num_logical_kernels = M × ceil(N / block_N)`，随 workload（NPU manifest `examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `GeluAndMulFwdOp`）变化：

| workload | M | N | dtype | n_num=ceil(N/2048) | 逻辑核数 M×n_num | 每核串行 tile 数 ceil(logical/40) |
|----------|---|---|-------|--------------------|------------------|----------------------------------|
| smoke | 16 | 128 | f32/f16/bf16 | 1 | 16 | 1（num_cores=16） |
| llama-8b-ffn-prefill | 2048 | 14336 | f16/bf16 | 7 | 14336 | 359 |
| llama-8b-ffn-decode | 1 | 14336 | bf16 | 7 | 7 | 1（num_cores=7） |
| qwen3-8b-swiglu-b32 | 32 | 12288 | f16/bf16 | 6 | 192 | 5 |

- **③ 规模判定与分核方案**：workload 跨越「无需适配」到「极大规模」两档，设计采用**统一 persistent 结构**（成熟 elementwise 形态，hardsigmoid/add/mish 同款）：

  - **小规模（逻辑核数 ≤ 40）**：`num_cores = num_logical_kernels`，每核 ≤1 tile，`tiles_per_core=1`，无串行循环、无负载不均——结论「**无需适配**」（逻辑核数 ≤ 物理核数）。覆盖 smoke（16）、decode（7）。
  - **中 / 极大规模（逻辑核数 > 40）**：固定启动内核数 = 40（= 物理向量核数），核内 `T.serial(tiles_per_core)` 串行处理多个逻辑块（`tiles_per_core = T.ceildiv(num_logical_kernels, 40)`，编译期常量，**循环边界静态**；prefill 359；qwen3 5），摊薄核启动开销，**内核总数 = 物理核数整数倍（1×）**，无负载不均。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅 block 级 / tile 级调度循环，无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级并行 | 核级并行 | `T.Kernel(num_cores, is_npu=True) as (cid, _)` | 每核处理若干逻辑块（§5.5 分核） |
| tile 级迭代 | 核内串行 | `T.serial(tiles_per_core)` | persistent 串行处理多个逻辑块，边界静态（§5.5） |
| 元素级 | 向量化 | v-prefix 整块 API（vcast / vmul / vadd / `T.verf`） | 块内逐元素由 Vector 单元并行（§1.6.2） |

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
            # 装载 -> v-prefix 链（verf 门控）-> 写回（见 §3.3）
```

### 6.3 流水线优化

Stage 4 可选：对主路径采用 guard-free 2-stage `T.Pipelined(tiles_per_core, num_stages=2)` 双缓冲流水（mish perf_opt 形态），将向量链隐藏于 MTE2 窗口（PL-1.6-copy-floor：≥16M 档计算链被完全隐藏）。Stage 3 基线不启用（正确性优先，`T.serial` 顺序即可）。

### 6.4 尾块处理

非整除 `N`：`tail = T.min(block_N, N - n0)`；`T.copy` 仅装载/写回 `[0:tail]` 有效列；v-prefix / `T.verf` 对全 `block_N` buffer 计算，垃圾尾部不写回（与 mish 同款）。`task < M*ceil(N/block_N)` guard 覆盖最后一波不完全 tile。manifest workload 的 `N` 均被 2048 整除（14336=7×2048、12288=6×2048），尾列路径服务任意 `N` 的通用情形。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer）。

### 7.2 同步点说明

无手动同步点。Developer 模式下编译器自动管理 GM↔UB 搬运与向量计算间的依赖（`T.copy` 与后续 v-prefix/`T.verf` op、以及各 v-op 之间的数据依赖由自动同步保证），无跨核协作（逐元素无跨元素依赖，各核独立）。

### 7.3 pass_configs 配置

Developer 模式默认配置，无需额外 `pass_configs`。环境变量 `TILELANG_ASCEND_MODE = "Developer"`（可选，默认即 Developer；NPU wrapper 声明 `ascend_mode = "Developer"`）。无运行时分支（dtype 分支为 trace-time 选定）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：Golden 以 **§0.1 源算子语义**为唯一依据（单 `[M,2N]` 张量、gate/value 切半、精确 erf GELU、fp32 opmath、输出同 dtype），移植源仓测试基准，**不复刻 §0.6 的 NPU 算法**（不含 `T.verf`、fp32 链、persistent 结构）。

```python
import torch
import torch.nn.functional as F

def golden_gelu_and_mul(x):
    """GeluAndMul 参考实现：x [M,2N] -> y [M,N] = gelu(gate) * value（精确 erf）。

    与源仓测试基准一致（examples/TileOPs/tests/ops/test_gelu_and_mul.py 的
    ref_program = (F.gelu(gate.float()) * value.float()).to(dtype)）。
    独立于 NPU 算法：仅依赖 torch.nn.functional.gelu（approximate="none"，erf 精确形式）
    与 torch 原生乘法；gate/value 切半由 torch 视角完成。
    """
    half = x.shape[-1] // 2
    gate = x[..., :half].float()
    value = x[..., half:].float()
    return (F.gelu(gate) * value).to(x.dtype)
```

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-2 | 1e-2 |
| bfloat16 | 1.6e-2 | 1.6e-2 |

（与源仓测试 `_get_tolerances` 一致。kernel 用全 fp32 链 + 原生 `T.verf`，对齐 torch fp32 opmath golden；§1.6.1 机器验证候选对 fp64 参照均 0 违反，对多项式/窄链基线 max ≤0.37 tolerance-unit。）

### 8.3 L0 门槛测试计划

L0 门槛（Stage 3 必过）：`M=2048, N=14336`（manifest 主 workload），`float16` / `bfloat16` 两 dtype，以及 `M=1, N=14336`（decode）与 smoke `(16,128)` 三 dtype，`randn` 输入（gate/value 两半独立 randn），vs golden（`F.gelu(gate)*value`）在对应容差下 `torch.testing.assert_close`。完整分层套件（L1 非整除 `N`、多 shape；L2 边界 size；Boundary 特殊值含 NaN/±inf/−0.0/subnormal）交由 `tilelang-op-develop` 规划，本设计不枚举。

**特别提醒（erf 边界与 NaN 语义验证）**：Boundary 用例必须覆盖 ① 大 `|gate|` 饱和（`erf(±inf)=±1`；**预期 `gelu(-inf)=NaN`**——源式 `-inf*0`，与 §8.1 golden `F.gelu(-inf)=NaN` 一致，**不得断言 0**；`gelu(+inf)=+inf`；另可由大负有限值 `gelu(-大值) → -0.0` 作饱和验证点）；② `NaN` 输入 → NaN 输出（§0.1 语义层要求，据此确认 Ascend `T.verf` 的 NaN 行为，见 §9.1）；③ fp32 精度（1e-5 容差对 `T.verf` 精度的实测标定）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 说明 |
|------|------|
| `T.verf` 的 NaN 语义未文档化 | `docs/Tilelang.language/数学操作/T.verf.md` 仅给出定义/值域/指令映射，未载明特殊值（NaN/±inf）处理。`torch.erf` 对 NaN 返回 NaN；若 Stage 3 实测 Ascend `T.verf` 丢 NaN（如 `fminf/fmaxf` 类），则按 hardsigmoid 先例用 `T.vcmp(x,x,"ne")` 自比较得 NaN 掩码 + `T.vselect` 单点恢复（语义层 NaN→NaN 不可妥协） |
| `T.verf` 精度未实测 | 文档未给 ulp；fp32 容差 1e-5 要求 `T.verf` 精度 ≤~1e-6。§1.6.1 以 torch.erf（fp32）为代理验证链结构，但硬件 `T.verf` 绝对精度须 Stage 3 L0 实测标定；若不达标，回退方案为移植源多项式（fp32 路径用多项式最坏 5.56 tolerance-unit 会超差，故 fp32 仍须精确 erf，可考虑 fp64 辅助或 API 侧确认） |
| `T.verf` 不支持 bf16 | 文档 dtype 表 bf16 标 `×`；bf16 路径必经 fp32 中转（`T.vcast` 双端），本设计已覆盖（§3.3） |
| UB 预算 | 32–56KB ≪ 192KB（§4.5），余量充足；`block_N` 可上调 |
| 分核 | 物理核数实查 20（×2=40），与 `CONST-aicore-910B2C`（24）不一致——本环境设备返回 20，以实查为准 |
| 知识库提示 | `CG-2026-0003`（Expert fixpipe lowering 版本配对缺口）与本设计 Developer 纯 Vector 路径无关；`TRAP-threads-kwarg-noop`（`threads=` 无效）已促 R1 折叠；`CASE-norm-adalayern-migration` 的「原生指令替代 + 机器验证」纪律已应用于 R2 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| bf16 直接调 `T.verf` | 对 bf16 buffer 调 `T.verf` | 编译失败（dtype 表 bf16 ×） | 先 `T.vcast(bf16→fp32)` 中转（§3.3） |
| UB 溢出 | `block_N` 过大（≫2048 的 fp32 全 buffer 链） | 编译失败（`ub overflow`） | 维持 `block_N=2048`；fp32 路径复用 buffer（§4.5） |
| 尾列越界 | `T.copy` 用 base+size 形态写尾列 | MTE 越界写坏相邻 tensor | 双显式切片 `T.copy(out_ub[0:tail], y[m, n0:n0+tail])`（TRAP-T-copy-region-semantics） |
| value 列偏移写错 | 误把 value 读自 `x[m, n0:n0+tail]` | 输出错误（用 gate 当 value） | value 列起点恒为 `N+n0`（§3.3/§4.3） |
| 原地 verf 依赖 | 写成 `T.verf(e32, e32)` 依赖 in-place | 不确定是否支持 | 用独立 buffer（`T.verf(w32, e32)`），不依赖原地 verf |
| `dtype=` 关键字 false alarm | `T.Tensor((M,2*N), dtype=...)` | 编译报 `name 'in_dtype' is not defined` | 位置参数传 dtype（design-template §3.3） |

### 9.3 特殊场景处理

- **非整除 `N`**：尾列 `tail` + `task` guard（§6.4）。
- **极小 shape**：`N < block_N` 时单列块；`M=1` 时逻辑核数 `=n_num`。
- **NaN / ±inf 输入**：按 §9.1 的 NaN 计划保证 NaN→NaN；±inf 经 erf 饱和（`erf(±inf)=±1`）：`gelu(+inf)=+inf`，而 **`gelu(-inf)=NaN`**（源式 `x*half*(one+erf_val)` 的 `-inf*0`，与 §8.1 golden 一致；机器自证见 `verify_equiv.py` 的 `BOUNDARY` 段）。
- **混合精度**：fp16/bf16 走 fp32 中转（vcast），fp32 直算；输出 dtype 与输入一致。
- **单张量契约**：Op 层不做 host 切分；gate/value 由 kernel 内 `T.copy` 列偏移读取（§0.1/§1.6.3）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/gelu_and_mul/gelu_and_mul_fwd_kernel/
├── DESIGN.md               # 本设计文档
├── verify_equiv.py         # §1.6.1 等价性机器验证脚本（已执行，结果内嵌）
└── RETROSPECTIVE.md        # Stage 1 复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 本设计文档 |
| `verify_equiv.py` | 已完成 | §1.6.1 等价性验证（30 行全 EQUIV_PASS） |
| `RETROSPECTIVE.md` | 已完成 | Stage 1 复盘（Phase 8） |
| `gelu_and_mul_fwd_kernel.py` | 待实现（Stage 3） | kernel + golden + 分层测试 |
| `test_gelu_and_mul.py` | 待实现（Stage 3） | 精度测试 |

### 10.3 命名规范

- 项目目录名：`gelu_and_mul`；算子目录名：`gelu_and_mul_fwd_kernel`；接口函数：`gelu_and_mul_fwd_kernel`（manifest 键 `GeluAndMulFwdOp`）。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）
2. ⬜ Golden 函数（§8.1）
3. ⬜ 算子实现（`gelu_and_mul_fwd_kernel`）+ 与 Golden 精度比对（Stage 3，`tilelang-op-develop`）
