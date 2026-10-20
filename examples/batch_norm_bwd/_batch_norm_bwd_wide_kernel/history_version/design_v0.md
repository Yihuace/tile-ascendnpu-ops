# _batch_norm_bwd_wide_kernel 算子设计文档

> 迁移任务（harness 模式）：BatchNormBwdOp 的宽核（one core per channel，单核持有整通道并归约）。
> 源算子（GPU 原始源码）：`TileOPs-main/src/tileops/kernels/norm/batch_norm.py` 的 `_batch_norm_bwd_wide_kernel`（L1658）+ host 类 `BatchNormBwdWideKernel`（L1981）。
> Stage 0 提取件（本函数迁移的精确源）：`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/_batch_norm_bwd_kernels.py`。
> NPU wrapper：`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py`；op `examples/TileOPs/tileops/ops/norm/batch_norm_bwd.py`；manifest `examples/TileOPs/tileops/manifest/norm.yaml`；test `examples/TileOPs/tests/ops/test_batch_norm_bwd.py`；bench `examples/TileOPs/benchmarks/ops/bench_batch_norm_bwd.py`。
> 可行性探针（Stage 0，真机 PASS，见 `examples/batch_norm_bwd/probe/PROBE.md` + `examples/batch_norm_bwd/probe/batch_norm_bwd_probe.py`）：逐一验证了 strided 逐通道 plane gather、双累加器两步 `T.reduce`、`T.vbrc` 标量广播、fp16/bf16 fp32 中转链、one-core-per-channel，以及超大 S 的核内 S-tiling（无跨核归并、无 atomics）。
> 工具链版本戳：tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b（2026-10-10 build）+ CANN 8.5.2 + Ascend910B3（`NPUUtils.get().get_aicore_num()` 实查 = 20，纯 Vector ×2 = 40）+ torch 2.7.1+cpu + torch_npu 2.7.1.post4。

## 0. 源算子解读与迁移分析

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 tilelang-op-design skill 的 `references/migration-analysis.md`）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。

### 0.1 源算子语义（做什么）

**数学语义**（BatchNorm 训练反向，从训练前向保存的逐通道统计量反推梯度）。输入 `grad_out, x : [N, C, S]`、`weight, mean, rstd : [C]`（fp32），记 `L = N*S`、`x_hat[c,l] = (x[c,l] - mean[c]) * rstd[c]`，逐通道 `c`：

$$
\text{grad\_bias}[c] = \sum_{l<L} \text{grad\_out}[c,l],\qquad
\text{grad\_weight}[c] = \sum_{l<L} \text{grad\_out}[c,l]\cdot \hat{x}[c,l]
$$

$$
\text{grad\_x}[c,l] = \frac{w[c]\cdot \text{rstd}[c]}{L}\Big(L\cdot\text{grad\_out}[c,l] - \text{grad\_bias}[c] - \hat{x}[c,l]\cdot\text{grad\_weight}[c]\Big)
$$

**I/O 契约**：`grad_out, x` 形状 `[N, C, S]`、dtype ∈ {float32, float16, bfloat16}（`DType[float32|float16|bfloat16]`）；`weight, mean, rstd` 形状 `[C]` fp32；输出 `grad_x : [N, C, S]` 同输入 dtype、`grad_weight : [C]` / `grad_bias : [C]` fp32（manifest `BatchNormBwdOp` signature；host 侧把任意 `(N, C, *spatial)` contiguous view 成 `(N, C, S)`，S=spatial 维乘积，无转置、非 in-place）。**输出 shape 从源码/op 回填确认**：host `forward` 末尾 `grad_x.view(x.shape)`（`TileOPs-main/src/tileops/ops/norm/batch_norm.py:251`）。

**规约语义**：逐通道 `c` 沿展平轴 `N·S`（原生 layout 的 `{0, 2}` 轴）归约两个独立标量——`grad_bias = Σ grad_out`、`grad_weight = Σ grad_out·x_hat`。**通道轴 `C` 是中间轴**：逐通道平面 `x[:, c, :]` 在 GM 中是 strided 区域（行跨步 `C*S`，行内 stride=1），归约轴集合不是尾轴。`weight/mean/rstd` 是逐通道输入标量。**累加顺序**：源宽核用固定 shuffle 树（`T.shfl_xor`，butterfly 深度 5）合并 warp 内部分量，再核内串行跳过 per-warp 总量，顺序固定 → 每次运行结果一致（源码 L1666-1668 docstring 明确）。

**dtype 语义**：累加全程 fp32（`accum_dtype = "float32"`）；`grad_out/x` cast 到 fp32 参与计算；`grad_x` 单次舍回输入 dtype；`grad_weight/grad_bias` 输出 fp32。即「fp32 opmath + 单次舍回」。

**边界语义**：`L % (threads*num_per_thread) != 0` 时源宽核走 `else` 分支逐元素 `if l < L` 谓词（L1677 `exact` 判定）；NaN/Inf 自然传播（无特殊处理）；`L=0`（空张量）由 op 层拦截返回零通道梯度（`TileOPs-main/src/tileops/ops/norm/batch_norm.py:237`）；`N≥1, C≥1, S≥1`。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现（移植源仓参考实现 `examples/TileOPs/tileops/workloads/norm.py 的 batch_norm_backward`），不复刻 NPU 算法。

### 0.2 源算子输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `grad_out` | 输入 | `(N, C, S)` | float16 / bfloat16 / float32 | 上游梯度；S=spatial 维乘积 |
| `x` | 输入 | `(N, C, S)` | same_as(grad_out) | 原始输入 |
| `weight` | 输入 | `(C,)` | float32 | 仿射缩放 gamma |
| `mean` | 输入 | `(C,)` | float32 | 前向保存的逐通道 batch mean |
| `rstd` | 输入 | `(C,)` | float32 | 前向保存的逐通道 reciprocal std |
| `grad_x` | 输出 | `(N, C, S)` | same_as(grad_out) | `out_idx=7`（源 `_bn_bwd_wide` 末参数） |
| `grad_weight` | 输出 | `(C,)` | float32 | `out_idx=5` |
| `grad_bias` | 输出 | `(C,)` | float32 | `out_idx=6` |

> 源 `@T.prim_func _bn_bwd_wide` 参数顺序为 `(grad_out, x, weight, mean, rstd, grad_weight, grad_bias, grad_x)`，`out_idx=[-1]`（仅 `grad_x` 自动分配，`grad_weight/grad_bias` 由 host 预分配传入，源码 L2038-2040）。本设计保持参数名/顺序，`out_idx` 改为 `[5, 6, 7]` 由 jit 统一分配三输出（§3.3）。

### 0.3 实现算法解读（怎么算）

**源码结构**（GPU，工厂模式）：`_batch_norm_bwd_wide_kernel(N, C, S, dtype)` → `@tilelang.jit(out_idx=[-1], compile_flags=["-O3","-DENABLE_BF16"])` 内 `_bn_bwd_wide_func(threads, num_per_thread)` → `@T.prim_func _bn_bwd_wide(...)`。host `BatchNormBwdWideKernel`（L1981）用 `_block_launch(call, 2)` 选 `(threads, num_per_thread)`，`forward` 预分配 `grad_weight/grad_bias` 后调用。

**计算步骤分解**（源码每个计算语句均已归入；`threads` 个 CUDA 线程协作一个通道，每线程持 `steps*num_per_thread` 元素）：

| 步骤 | 计算（源码行） | 输入 | 输出 | 对应语义公式部分 |
|------|----------------|------|------|-----------------|
| P0 | `params[0..2] = weight[bc], mean[bc], rstd[bc]`（L1696-1698） | GM 逐通道 3 标量 | 本地 register | 逐通道参数读入 |
| P1a | 装载 `go_held`：`T.vectorized` 读 `grad_out[head//S, bc, head%S + i]`（L1710-1711） | GM grad_out | register `go_held` (steps·npt, dtype) | 输入搬运 |
| P1b | 装载 `x_held`：同上（L1712-1713） | GM x | register `x_held` | 输入搬运 |
| P1c | 累加：`g=cast(go); x_hat=(cast(x_held)-params[1])*params[2]; acc+=g; acc_xhat+=g*x_hat`（L1714-1720） | go_held, x_held, params | `acc`,`acc_xhat` fp32 | grad_bias / grad_weight 局部贡献 |
| P1d | 尾块掩码分支：`if l<L` 逐元素（L1721-1732） | 同上 | 同上 | 非整除边界 |
| R1 | warp shuffle 树：`acc += T.shfl_xor(acc, 1<<step)`（butterfly 5 步，L1734-1736） | acc, acc_xhat | 同 register | 线程级归约（warp 内） |
| R2 | per-warp 写 shared + barrier + 串行合并（L1738-1748） | acc | `acc`/`acc_xhat` 通道总量 | 跨 warp 归约 |
| W1 | `if tx==0: grad_bias[bc]=acc; grad_weight[bc]=acc_xhat`（L1750-1752） | acc | GM 逐通道 2 标量 | 写 grad_bias/grad_weight |
| E0 | 逐通道标量预折叠：`scale=w*rstd; per_elem=scale/L; shift=-per_elem*acc; xhat_coef=per_elem*acc_xhat`（L1755-1758） | params, acc | 本地标量 | epilogue 系数 |
| E1 | 逐元素 epilogue：`out[i]=cast(scale*cast(go_held)+shift-xhat_coef*((cast(x_held)-mean)*rstd), dtype)`（L1760-1773） | go_held, x_held, 系数 | register `out` (npt) | grad_x 公式 |
| E2 | `T.vectorized` 写 `grad_x[head//S, bc, head%S+i]`（L1774） | out | GM grad_x | 输出搬运 |

**数据流与内存访问模式**（源硬件视角）：

```
GM[weight/mean/rstd] --scalar load--> REG[params]
GM[grad_out] --T.vectorized strided (head//S, bc, head%S)--> REG[go_held]
GM[x]        --T.vectorized strided--> REG[x_held]
  --(逐元素 cast + x_hat + 两个 += )--> REG[acc, acc_xhat]
  --T.shfl_xor butterfly--> SMEM[warp_sum]* --barrier--> REG[acc] (通道总量)
REG[acc] --tx==0--> GM[grad_bias, grad_weight]
REG[go_held, x_held] --epilogue 标量复合式--> REG[out] --T.vectorized--> GM[grad_x]
```

**循环与并行结构**：`T.Kernel(C, threads=threads)` 一维 grid（**每 block 一个通道**，逻辑核数 = C，无核内任务循环）；block 内 `threads` 个线程协作通道，每线程持 `steps*num_per_thread` 元素（`steps = ceil(L/(threads*num_per_thread))`）；归约分两级——warp 内 shuffle 树 + 跨 warp shared 串行合并。**无核间协作**（通道间独立，无跨 block 归并，源码无 atomics）。

**host 侧逻辑**：`BatchNormBwdWideKernel.__init__` 计算 `launch = _block_launch(call, 2)`（`threads/num_per_thread`）；`forward` 做 `grad_out/x.contiguous()`、预分配 `grad_weight/grad_bias`，调用 kernel。op 层 `BatchNormBwdOp.forward` 把任意 `(N,C,*spatial)` view 成 `(N,C,S)`、`contiguous()`，返回 `grad_x.view(x.shape)`。**无数据内容预处理**（纯 shape 元数据视图）。`@functools.lru_cache(maxsize=32)` 工厂按 `(N,C,S,dtype)` 特化。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | 通道 register 驻留（go_held/x_held） | 消 GM 重读（grad_out/x 单遍读） | `T.alloc_local([steps*npt])` 持通道元素 | 大寄存器堆 | 硬件强相关 |
| 2 | warp shuffle 树形归约（`T.shfl_xor`） | 低延迟线程级归约 | butterfly 5 步 XOR 交换 | warp（32 lane）+ shuffle 指令 | 硬件强相关 |
| 3 | two-accumulator 融合归约（acc / acc_xhat） | 单遍同时得 grad_bias 与 grad_weight | 同一遍内两个 `+=`（L1719-1720） | 无（纯算法层融合） | 可移植 |
| 4 | 固定归约顺序（shuffle 树 + 串行合并） | `grad_weight/grad_bias` 逐次运行确定 | 固定深度树 + 固定 warp 序 | warp 拓扑 | 硬件强相关（意图可移植） |
| 5 | epilogue 标量预折叠（scale/per_elem/shift/xhat_coef） | 逐元素只做 2 mul + 1 sub + 1 sub | 逐通道标量在循环外算一次 | 无（纯算法层） | 可移植 |
| 6 | fp32 opmath + 单次舍回 | 精度（fp32 累加，输出 dtype） | `T.cast(...,"float32")` 累加、赋值截断回 dtype | 寄存器 | 硬件强相关 |
| 7 | 向量化访存（`T.vectorized`） | 访存带宽 | 每线程 `num_per_thread` 连续元素一次读写 | memory coalescing | 模式可移植 |
| 8 | 整除快路径 / 尾块掩码路径（`exact` + `if l<L`） | 整除去谓词开销 | `exact` 编译期分支 + 逐元素守卫 | 无（模式可移植） | 模式可移植 |
| 9 | 一通道一 block 映射 | 通道间并行、无跨核归并 | `T.Kernel(C)` | CTA 调度 | 硬件强相关 |
| 10 | lru_cache 工厂 + per-shape JIT 特化 | 摊编译开销 | `functools.lru_cache(maxsize=32)` | 无（host 层） | 可移植 |
| 11 | `out_idx` 输出自动分配（host 预分配 channel 梯度） | 接口契约 | jit out_idx | 无 | 可移植（接口契约） |

> 识别不出来 ≠ 不存在。Stage 0 观察项（threads/npt 复杂度、split 变体、三 kernel 选择策略）已并入上表与 §0.5 处置。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定依据：`references/migration-analysis.md` §5 三层模型 + §5.3 映射表；`examples/` 佐证；`docs/` 条目；pattern-library 实测条目；探针 `examples/batch_norm_bwd/probe/PROBE.md`。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（逐通道两标量归约 + grad_x 公式 + fp32 opmath + 单次舍回） | 语义 | 无 | — | **保留** | 语义层无条件保留（§0.1） |
| two-accumulator 融合归约（#3） | 算法 | 无 | — | **保留** | 同遍 `T.reduce` 双路（探针验证） |
| epilogue 标量预折叠（#5） | 算法 | 无 | — | **保留** | 标量在核内循环外算一次（§1.6.1 O2 承接） |
| 通道 register 驻留（#1） | 优化 | register | 有（UB 192KB） | **等价替换** | `T.alloc_ub`（Expert 映射 UB，`docs/Tilelang.language/内存操作/T.alloc_ub.md`）；整通道平面驻留 UB，预算见 §4.5 |
| warp shuffle 树形归约（#2） | 优化 | warp 32 lane + shuffle | 无直接等价 | **重新设计** | → §0.6 R2：UB 中转 + `T.reduce`（dims=[1]→[0]）整平面归约 |
| 固定归约顺序（#4） | 优化 | warp 拓扑 | 有（reduce 定序） | **等价替换** | `T.reduce` 降级为 `hivm::VReduceOp`（`docs/Tilelang.language/规约操作/T.reduce.md` §3），每核固定归约序（探针 PROBE §6） |
| 向量化访存（#7） | 优化 | coalesced | 有（slice copy + 向量宽度） | **等价替换** | `T.copy` 双显式 slice（GM↔UB，`docs/Tilelang.language/内存操作/T.copy.md`）；向量宽 fp16/bf16 ×8、fp32 ×4 |
| fp32 opmath + 单次舍回（#6） | 优化 | 寄存器 cast | 有（UB fp32 + vcast） | **等价替换** | `T.vcast(rint)` 升 fp32 → fp32 v-prefix 链 → `T.vcast(rint)` 单次舍回（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`；bf16 无 Vector 算术 → 中转必需） |
| 整除快路径 / 尾块掩码（#8） | 优化 | 无 | 有更优形态 | **重新设计** | → §0.6 R3：`T.min` + src/dst 双显式 slice + 垃圾元素丢弃 |
| 一通道一 block 映射（#9） | 优化 | CTA 调度 | 无（一维 Kernel + 物理核数限制） | **重新设计** | → §0.6 R1：一维 `T.Kernel` persistent 分核（`num_kernels = min(C, 40)`） |
| `threads` 参数 / block×thread 两级并行 | 优化 | CUDA 线程 | 无 | **舍弃** | npuir `T.Kernel(threads=)` 无效果；并行交给 Vector lane（§1.6.2/§1.6.3） |
| `T.shfl_xor` / `T.shift_left` / `T.sync_threads` / `T.get_thread_binding` | 优化 | CUDA warp/线程 | 无 | **舍弃** | 后端不适用；归约改 `T.reduce`（§0.6 R2） |
| lru_cache 工厂 + JIT 特化（#10） | host | 无 | — | **保留** | host 层硬件无关；NPU wrapper 用 Kernel 类实例缓存承接 |
| 三 kernel 选择策略（wide/split/general） | host | CUDA `sm_count` | 有（物理核数） | **等价替换** | `_AI_CORE_COUNT=40` 替代 `call.sm_count`（wrapper 已移植，`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py 的 L83`） |

### 0.6 NPU 算法重设计

**重设计项 R1: 并行结构（一通道一 block 超发 → 一维 persistent 分核）**

- **源方案**：`T.Kernel(C, threads=threads)`——逻辑 block 数 = C，依赖 CUDA 运行时把 C 个 block 调度到 SM。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`，`num_kernels = min(C, 40)`（40 = 20 AICore × 2，纯 Vector 翻倍，§5.5 实查）；核内 `for i in T.serial(num_local_tasks)` 以 grid-stride 任务映射 `ch = i*num_kernels + cid`，`if ch < C` 守卫；`num_local_tasks` 为 host 层 Python 编译期常量（静态边界）。意图承接：消除超发逻辑内核的串行调度开销（`core-split-strategy.md` §1 极大规模分支）。参照 `examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/DESIGN.md` §0.6 R1 同款 persistent 分核。

**重设计项 R2: 归约（warp shuffle 树 + shared 合并 → UB 整平面 `T.reduce`）**

- **源方案**：每线程持 `steps*num_per_thread` 元素，warp 内 `T.shfl_xor` butterfly 5 步，per-warp totals 写 shared，barrier 后串行合并。
- **NPU 新算法**：整通道平面 gather 到 UB `(N, S)` fp32，两次 `T.reduce(reduce_mode="sum")`——`(N,S) --dims=[1]--> (N,1) --dims=[0]--> (1,1)`（探针主路径验证；S=1 时退化为单次 `dims=[0]`，见 §3.3）。逐通道 Scalar 广播用 `T.vbrc`。
- **语义保持论证**：归约是加法结合律下的同一数学和；NPU 用固定树序（`hivm::VReduceOp`），每核内定序 → 通道梯度确定（探针 PROBE §6）。累加分组与 GPU 不同 → fp16/bf16 舍入差异在 op 容差内（§8.2：fp16 1e-2、bf16 2e-2，探针最坏 max_abs_err 4.883e-04）。

**重设计项 R3: 边界处理（逐元素谓词 → `T.min` + 双显式 slice + 垃圾元素丢弃）**

- **源方案**：`exact` 编译期分支 + 非整除时逐元素 `if l < L` 掩码装载/写出。
- **NPU 新算法**：统一单路径（去 `exact` 分支）：`real_s = T.min(bS, S - s0)`（`docs/Tilelang.language/比较操作/T.min.md` 标量 min），装载/写出用 src/dst 双显式 slice；S=1 与尾 tile 的越界元素由 slice extent 截断。S-tiling 的核内 `T.serial(n_tiles)` 边界静态。
- **语义保持论证**：越界元素不参与装载/写出；reduce 在有效 extent 上求和（非整除时尾 tile 用 `T.min` 截尾）；与源谓词掩码等价。参照 `examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/DESIGN.md` §0.6 R3。

**重设计项 R4: dtype 路径（寄存器隐式 cast → 显式 `T.vcast(rint)` fp32 中转链）**

- **源方案**：`T.cast(..., "float32")` 上 cast + 赋值隐式截断回 dtype。
- **NPU 新算法**：fp16/bf16 输入：`T.vcast(stage, x32, round_mode="rint")` 升 fp32（f16→f32 / bf16→f32 rint 无损）→ fp32 v-prefix 链 → `T.vcast(..., round_mode="rint")` 单次舍回；fp32 输入无 cast。**bf16 无 Vector 算术**（`T.vmul.md`/`T.vsub.md`/`T.vadd.md`/`T.reduce_sum.md` §2.2.1 bf16 列为 ×）→ fp32 中转必需。
- **语义保持论证**：fp16/bf16→fp32 精确无损；fp32→目标 dtype 单次 RNE 舍回 = 源赋值截断语义；跨 dtype `T.copy` 的隐藏 VCast 不用作 cast 融合。

**重设计项 R5: 超大 S 的核内 S-tiling（源 split kernel → 单核 S-tile 两遍，无跨核归并）**

- **源方案**：GPU 的 `_batch_norm_bwd_split_kernel`（L1795）把通道切成 `splits` 个 block，`partial_sum[C,splits]` + finalize kernel 合并。
- **NPU 新算法**：一个通道始终由**一个核**持有——当 `N·S` 超 UB 预算时，核内 `T.serial(n_tiles)` 分块：pass 1 逐 S-tile 累加 `acc_g/acc_gx (N,bS)` fp32，最后一次 `T.reduce` 得通道标量；pass 2 重读 S-tile 做 grad_x epilogue（探针 `_scratch_tiled.py` 验证 S=512/bS=128 超大 S PASS）。**无第二个 block、无跨核归并、无 atomics**。
- **语义保持论证**：S-tile 累加 = 展平 `N·S` 求和的结合律重排；`acc_g[n,j]` 列向累加后再 `reduce(dims=[1],[0])` 与整平面一次性求和同值（fp32 舍入差在容差内）。承接源 split 的「统计遍 + 应用遍」两扫描意图，但把跨核归并消除（one core per channel 使通道归约天然闭环）。

**舍弃项的意图承接汇总**（不得静默丢弃，migration-analysis §6.4）：
- register 驻留（#1）意图 = 消 GM 重读 → UB 整平面驻留承接（held 档单扫描）；
- shuffle 树（#2）意图 = 低延迟树归约 → `T.reduce` 硬件树归约承接；
- threads（#9 部分）意图 = 块内并行度 → Vector lane 向量化承接（§1.6.2/§1.6.3）；
- 谓词掩码（#8）意图 = 越界安全 → `T.min` + 双显式 slice 承接。

### 0.7 标杆实现

- 源算子：`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/_batch_norm_bwd_kernels.py`（Stage 0 提取件，迁移精确源）；GPU 原始 `TileOPs-main/src/tileops/kernels/norm/batch_norm.py`（`_batch_norm_bwd_wide_kernel` L1658）。
- 参考实现 / 测试基准：`examples/TileOPs/tileops/workloads/norm.py 的 batch_norm_backward`（独立纯 PyTorch，fp32 中间量 + 舍回输入 dtype）——golden 以 §0.1 语义为依据移植此实现，不复刻 NPU 算法。
- manifest：`examples/TileOPs/tileops/manifest/norm.yaml` `BatchNormBwdOp` 条目（5 输入 3 输出；workloads resnet-fc/stage1/2/3/large-spatial；roofline `flops = 9*B*C*prod(L)`，`eval_roofline` nbytes `= (3*N*C*S + 5*C)*elem_bytes`）。
- 性能基准：`examples/TileOPs/benchmarks/ops/bench_batch_norm_bwd.py`（manifest 驱动，5 输入多输入形态）。
- 探针：`examples/batch_norm_bwd/probe/PROBE.md` + `examples/batch_norm_bwd/probe/batch_norm_bwd_probe.py`（真机 PASS，含超大 S 核内 tiling）。

---
## 1. 概述

### 1.1 算子名称

`_batch_norm_bwd_wide_kernel`（BatchNormBwdOp 的宽核变体；NPU 侧 one core per channel）

### 1.2 功能描述

对 `(N, C, S)` 输入，逐通道沿展平轴 `N·S` 归约两个标量 `grad_bias = Σ grad_out` 与 `grad_weight = Σ grad_out·(x−mean)·rstd`，并据此逐元素写出 `grad_x`（NPU 重设计后算法，与源算法差异见 §0.6/§1.6.0：一维 persistent 分核、UB `T.reduce` 替代 warp shuffle、held/tiled 双档、fp32 中转链、`T.vbrc` 标量广播）。

### 1.3 数学公式

$$
\hat{x}[c,l] = (x[c,l] - \text{mean}[c])\cdot \text{rstd}[c],\qquad
\text{gb}[c] = \sum_{l<L}\text{grad\_out}[c,l],\qquad
\text{gw}[c] = \sum_{l<L}\text{grad\_out}[c,l]\cdot\hat{x}[c,l]
$$

$$
\text{grad\_x}[c,l] = \frac{w[c]\cdot\text{rstd}[c]}{L}\Big(L\cdot\text{grad\_out}[c,l] - \text{gb}[c] - \hat{x}[c,l]\cdot\text{gw}[c]\Big),\qquad L = N\cdot S
$$

`grad_x` 在 fp32 域按上式算得后单次舍回输入 dtype；`grad_weight/grad_bias` 为 fp32。

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

与 §1.6.0 调研选定算法一致（中心化 `x̂` + 双标量归约 + 单核持有通道；held 档单扫描 / tiled 档两扫描）。每核处理一个通道任务：

1. **逐通道参数装载**：`weight[c]`、`mean[c]`、`rstd[c]` GM→UB `(1,1)` fp32（`T.copy` 标量 slice）。
2. **逐通道平面 gather**：`grad_out[0:N, c, 0:S]`、`x[0:N, c, 0:S]` 经 `T.copy` 双显式 slice 搬入 UB `(N, bS)`（strided 区域，行跨步 `C·S`，行内 stride=1）。
3. **fp32 中转**：fp16/bf16 输入经 `T.vcast(rint)` 升 fp32；fp32 输入同 dtype 直拷。
4. **`x̂` 计算**：`x̂ = (x − mean)·rstd`——`mean/rstd` 经 `T.vbrc` 广播到 `(N,bS)` 后 `T.vsub`/`T.vmul`。
5. **双累加归约**：`acc_g += grad_out`、`acc_gx += grad_out·x̂`（held 档直接对整平面；tiled 档逐 S-tile 累加，列向后一次 `T.reduce`）；通道标量 `gb = Σ acc_g`、`gw = Σ acc_gx` 经两步 `T.reduce`（`dims=[1]`→`(N,1)` 再 `dims=[0]`→`(1,1)`）；写 `grad_bias[c]`、`grad_weight[c]`。
6. **epilogue 系数预折叠**：`scale = w·rstd`、`per = scale·fl(1/L)`、`shift = −per·gb`、`xhat_coef = per·gw`（逐通道标量，循环外一次）。
7. **`grad_x` 逐元素写出**：`grad_x = scale·grad_out − shift_brc − xhat_coef·x̂`（`shift/xhat_coef` 经 `T.vbrc` 广播），`T.vcast(rint)` 舍回 dtype，`T.copy` 双显式 slice 写 GM。

与源算法差异及来源：并行组织从「一通道一 block + threads」改为「一维 persistent 分核 + 核内 serial」（§0.6 R1）；归约从「warp shuffle 树 + shared 合并」改为「UB 整平面 `T.reduce` 两步」（§0.6 R2）；边界从「逐元素谓词」改为「`T.min` + 双显式 slice」（§0.6 R3）；dtype 从隐式 cast 改为显式 `T.vcast(rint)` 中转链（§0.6 R4）；超大 S 从「跨 block split + finalize」改为「单核 S-tile 两扫描、无跨核归并」（§0.6 R5）；epilogue 预折叠（§1.6.1 O2）。**保留**源算法意图：two-accumulator 单遍融合、epilogue 标量预折叠、fp32 opmath 单次舍回。

### 1.5 数据流图

```
GM[weight/mean/rstd] --T.copy(标量 slice)--> UB[w1/m1/r1 (1,1) fp32]
GM[grad_out] --T.copy(双slice,N×bS)--> UB[stage_g(dtype)] --vcast(rint)--> UB[g32 (N,bS) fp32]
GM[x]        --T.copy(双slice,N×bS)--> UB[stage_x(dtype)] --vcast(rint)--> UB[x32 (N,bS) fp32]
UB[x32] --vbrc(m1)+vsub+vbrc(r1)+vmul--> UB[xh32 (N,bS) fp32]  (= x_hat)
UB[g32] --(T.reduce dims=[1]→(N,1)→dims=[0]→(1,1))--> UB[gb (1,1)] --T.copy--> GM[grad_bias]
UB[g32]·UB[xh32] --vmul--> UB[prod] --(T.reduce 两步)--> UB[gw (1,1)] --T.copy--> GM[grad_weight]
 UB[gb],UB[gw] + UB[w1],UB[r1] --标量预折叠--> UB[scale/per/shift/xhat_coef (1,1)]
UB[g32],UB[xh32] + UB[scale/shift/xhat_coef] --vbrc 广播 + vmul/vsub--> UB[gx32 (N,bS)]
UB[gx32] --vcast(rint)--> UB[stage_g(dtype)] --T.copy(双slice)--> GM[grad_x]
（全部计算在 UB 内闭环；held 档 g32/xh32 常驻 UB 单扫描；tiled 档逐 S-tile 装载/重读，UB 驻留有界）
```

### 1.6 算法调研与优化分析 ⭐

> 调研对象：同一数学语义（逐通道 BN 反向，从 forward 保存的 mean/rstd 反推三梯度）的算法族。源算法（warp shuffle 树 + register 驻留）只是基线候选之一。调研深度：**完整调研**（规约/统计类）。信息源：`references/algorithm-candidates.md` ALG-reduction 族命中行 + 本仓 examples（`examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/DESIGN.md` 同族行归约迁移、`examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/DESIGN.md` 尾块模式）+ pattern-library（`constants.md` CONST-mte2-degradation / CONST-vector-launch-overhead / CONST-aicore-910B2C）+ 源码 §0.3/§0.4 + 探针真机结论 + 结构判据。无互联网检索（本地源全覆盖四问所需候选）。

#### 1.6.0 算法调研（Algorithm Research）⭐

**R1 等价化简公式候选**（基线候选必须在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3？ |
|---|------|------------|-----------|---------|----------|
| 0 | 基线（源 `_batch_norm_bwd_wide_kernel`） | 中心化 `x̂` + 双 `+=` 累加 + warp shuffle 树归约 + register 驻留 + 逐元素谓词 | — | — | ✅（基线） |
| 1 | held 单扫描整平面驻留 | 同基线数学；整通道平面 `(N,S)` 驻留 UB，单遍读 `grad_out/x`，两次 `T.reduce`，epilogue 就地 | 数学恒等（同和同式） | 少一遍 `grad_out/x` 读（流量 3·L·C·eb，Manifest roofline 下界） | ✅（主选 A，N·S 落 UB 预算内） |
| 2 | tiled 两扫描（核内 S-tile） | 同基线数学；`(N,bS)` 分块驻留，pass1 累加 + final reduce，pass2 重读做 epilogue | 数学恒等（结合律重排） | UB 有界、任意 S（代价：+2 读 → 5·L·C·eb） | ✅（主选 B，超大 S） |
| 3 | raw-moment `grad_weight` | `gw = rstd·(Σ grad_out·x − mean·Σ grad_out)` | 代数恒等，但大均值小方差 fp32 **灾难性抵消**（D-1 判别段 A） | 归约遍省每元素 1 sub + 1 mul（≈L op/通道） | ❌（数值否决，§1.6.1 判别段 A 机器证据） |
| 4 | 跨核 split 通道（源 `_batch_norm_bwd_split_kernel`） | 通道切 `splits` 段跨 block，部分和 `[C,splits]` + finalize 合并 | 数学恒等（容差内） | 通道内并行度提升 | ❌（需跨核同步/atomics；one core per channel 已闭环，见 R2/R4） |
| 5 | online / Welford 变体 | 逐 tile 增量统计 | — | — | ❌（本反向的两和是**输入无关函数的可加一阶矩**，归约天然单遍；无 variance/max 可在线更新——见 R2） |

**R2 在线算法**：**无新增收益（等价于基线）。** 结构依据：BN 反向的 `mean/rstd` 是**前向输入**（非本算子计算），本算子只算两个加性一阶矩 `Σ grad_out` 与 `Σ grad_out·x̂`——任何分块/流式归约就是把这两和的求和项分组，本身就是「单遍 streaming reduce」，不存在需要重缩放（rescale）的 running 统计量（对比 online softmax 的 running max/sum）。因此「在线化」在本算子上退化为 R1#1/#2 的 held/tiled 结构选择（UB 驻留 vs 分块），不构成独立算法族。源算法 `_batch_norm_bwd_kernel`（L1487）已含 `clear=False` 累加式分块（源码证据：分块累加是该族既有结构）。

**R3 复杂度对比**（每通道、`L = N·S`、`eb` = dtype 字节；FLOPs 量纲 = 向量操作数（含 cast 计数为 1 op/元素），无超越函数；GM Bytes 口径 = `grad_out` 读 + `x` 读 + `grad_x` 写 + `weight/mean/rstd` 与通道梯度小项 5C；扫描遍数 = 同一输入张量的 GM 重复读取次数）：

| 算法候选 | FLOPs（每通道） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线 shuffle 树（=候选 0） | ≈9L（x̂ 2L + 双累加 3L + epilogue 4L；与 manifest roofline flops=9·B·C·L 一致） | 3·L·C·eb（单遍读） | grad_out 1、x 1 | 2 寄存器数组 (steps·npt) | 通道间并行；无跨核 |
| held 单扫描（主选 A） | ≈9L | **3·L·C·eb**（+5C） | grad_out 1、x 1 | 2 fp32 `(N,S)` 平面 + 1 scratch | 通道间并行；无跨核 |
| tiled 两扫描（主选 B） | ≈9L | 5·L·C·eb（+5C） | grad_out 2、x 2 | 有界 `(N,bS)`·≈18B | 通道间并行；无跨核 |
| raw-moment（候选 3） | ≈8L（省归约遍 x̂ 的 2L） | 同 held/tiled | 同 held/tiled | 同 held/tiled | 同；**抵消风险** |
| 跨核 split（候选 4） | ≈9L + 归并开销 | 5·L·C·eb + `[C,splits]` 部分和往返 | 2 + 部分和 | +`[C,splits]` | 通道内并行；**需跨核归并**（sync_block_set/wait 或两阶段 kernel） |

prefill 代表（resnet-stage1 `N=8,C=64,S=1024` fp16，L=8192）：held 3·8192·64·2 ≈ 3.1MB；large-spatial（`N=4,C=128,S=1048576`，L=4194304）：tiled 5·4194304·128·2 ≈ 5.37GB。带宽受限算子上访存/遍数是主导项（`constants.md` CONST-mte2-degradation）。

**R4 硬件亲和性评估**（检查清单逐项；负向淘汰附已核对的 docs 条款或 examples 佐证，或「未文档化假设 + 估算依据」）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| held 单扫描（A） | Vector：`T.reduce` + v-prefix 逐元素（`docs/Tilelang.language/规约操作/T.reduce.md` / `T.vmul.md`），无 MAC 段 ✓ | 2 fp32 `(N,S)` + scratch ≈ 18·N·S B ≤ 160KB（manifest 除 large-spatial 外全落；§4.5）✓ | S 尾轴连续 stride=1；fp16/bf16 ×8、fp32 ×4；S=900/784 有尾 lane 浪费 ≤ 1%（§1.6.3） | N/C/S/bS 编译期常量 ✓ | 顺序依赖链；装载/计算 overlap 交 auto multibuffer ✓ | ✅ 主选 A |
| tiled 两扫描（B） | Vector ✓ | 有界 `(N,bS)` ✓（任意 S） | bS 取 8 的倍数，尾 tile `T.min` 截尾 ✓ | `n_tiles` 静态 ✓ | 同上 | ✅ 主选 B（超大 S） |
| raw-moment（3） | Vector ✓ | 同 A/B | 同 A/B | ✓ | ✓ | ❌ 数值否决（R1#3 判别段 A） |
| 跨核 split（4） | Vector ✓ | +部分和缓冲 | ✓ | ✓ | 需核间同步 | ❌（`T.atomic_add`/finalize 可用但无必要；`docs/Tilelang.language/原子操作/T.atomic_add.md`） |
| online/Welford（5） | 合并链对 Vector 失分 | 同 | 同 | 同 | 依赖链长 | ❌（R2：无对应 running 统计量） |

**调研结论**：选定**「中心化 `x̂` + 双标量 `T.reduce` 归约 + one core per channel」算法族**（候选基线数学内核 + 候选 1/2 的 NPU 驻留实现），按 UB 预算在 held 单扫描（主选 A）与 tiled 两扫描（主选 B）间切换；采纳 §1.6.1 的 O1/O2/O3。关键依据：R3 表中 held 的 GM 流量 3·L·C·eb 为全候选最小（tiled +66% 流量但 UB 有界，是超大 S 的唯一可行档）；R4 全项 Vector 亲和；raw-moment 因大均值小方差抵消被机器证据否决（§1.6.1 判别段 A）；跨核 split 的归并/atomics 在 one core per channel 下无收益。与基线的结构差异一句话：数学内核不变（中心化 `x̂` + 双和），并行组织从 grid 超发改 persistent 分核、归约从 warp shuffle 改 UB `T.reduce`、边界从谓词改 `T.min`+双 slice、dtype 改显式 fp32 中转、超大 S 改单核 S-tile（消除跨核归并）。源算法优化手段意图承接见 §0.6 末尾汇总。调研范围：`algorithm-candidates.md` ALG-reduction 族命中行、`examples/fused_add_layer_norm` 同族迁移设计、`examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/DESIGN.md` 尾块模式、`constants.md` CONST-* 三条、源码 §0.3/§0.4、探针 PROBE.md、结构判据（R2）。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)；常数引用 `constants.md` CONST-mte2-degradation 3:1 R/W 地板 ~1.26 TB/s、CONST-vector-launch-overhead ~0.5µs/op）**：

- **resnet-stage1** `8×64×1024` fp16（held，流量 3.1MB）：流量项 ≈ 3.1MB / 1.26TB/s ≈ **2.5µs**；发射项 ≈ 20 op × 0.5µs × 通道轮次（40 核 × 2 通道轮 ≈ 每核 2 通道）≈ 20µs？——**未实测假设**：发射项按每核 op 数 × 0.5µs 粗估、未含 auto-multibuffer 隐藏，标「未实测假设 + 估算依据（CONST-vector-launch-overhead）」，交 Stage 4 校准。取 max → **≈ 20µs 量级**。
- **large-spatial** `4×128×1048576` fp16（tiled，流量 5.37GB）：流量项 ≈ 5.37GB / 1.26TB/s ≈ **4.26ms**（发射项：40 核 × 4 任务 × 512 S-tile × 20 op × 0.5µs ≈ 0.82ms，被 MTE2 窗口隐藏，取 max = 流量项）。
- **resnet-fc** `32×64×1`（held，L=32，流量 24KB）：发射/延迟主导 ≈ **10–20µs**（`constants.md` CONST-vector-launch-overhead）。
- 容量项：0（纯 Vector，无跨引擎往返）。**⚠️ stale 标注**：CONST-mte2-degradation（tilelang 0.1.2+ed787bb，2026-09-07）被 kb_stale_check 标记 stale（当前 417dae99，2026-10-10）；数字仅用于候选排序与 Stage 4 对账，结构性结论（held 流量最小、tiled +66%）不依赖其精确值。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定算法（中心化 `x̂` + 双标量归约，held/tiled）。逐项四要素：

| # | 优化项 | 原式（源/数学） | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| O1 | 除法转乘倒数（L 为编译期常量） | `coef = w·rstd / L` | `per = (w·rstd) · fl(1/L)`（`L_inv` 工厂期 Python 常量） | L 编译期常量；`fl(1/L)` 相对误差 ≤ 2⁻²⁴（L=2^k 时精确）；D-1 机器验证 21 案例（3 dtype × 7 shape）全过：候选式 vs 基线式 **max ulp = 0、违反率 0** | 逐通道 1 次 vdiv → 1 次 vmul（(1,1) buffer，op 型替换，零风险） |
| O2 | distributed epilogue（折 L 与 coef 入逐通道标量） | `grad_x = coef·(L·grad_out − gb − x̂·gw)` | `grad_x = scale·grad_out − per·gb − per·gw·x̂`，`scale = w·rstd` | 分配律恒等：`coef·L·grad_out = (w·rstd)·grad_out = scale·grad_out`；`coef·gb = per·gb`；`coef·gw·x̂ = per·gw·x̂`。D-1：**max ulp = 0、违反率 0**（三 dtype 全过） | 逐元素由 3 mul（`L·g`、`x̂·gw`、末乘 `coef`）降为 2 mul（`scale·g`、`per·gw·x̂`），每元素省 1 向量 mul（≈L op/通道） |
| O3 | `x̂` 公共子表达式消除（CSE） | 归约遍与 epilogue 各算一次 `(x−mean)·rstd` | held 档 `x̂` 驻留 UB 一次、归约与 epilogue 两处复用 | 同值复用（同一 buffer 的恒等读取），无浮点语义变化 | held 档每元素省 1 vsub + 1 vmul；**tiled 档无法驻留**，pass2 重算（在 R3 流量中已计）|

**D-1 等价性机器验证结果表**（脚本 `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/verify_equiv.py` 已执行，2026-10-14，torch CPU、fp64 参照 + 角点值；判定准则：候选式 vs 基线式在 dtype 容差内且违反率 = 0）：

| 验证项 | 案例数 | fp16/bf16 舍回后 max ulp | fp32 max ulp | 违反率 | 结论 |
|--------|--------|---------|---------|--------|------|
| O1+O2 联合（候选式 vs 基线式） | 21（3 dtype × 7 shape：(32,1)/(8,1024)/(4,1024)/(4,784)/(4,2048)/(1,1)/(2,3)） | **0（逐位一致）** | **0** | **0**（三 dtype 全 0） | **EQUIV_PASS** |
| 判别段 A：raw-moment `gw`（拒绝证据，非采纳项） | 7 shape × 3 dtype | — | fp32 max 违反率 **0.1553**（N=4,S=2048 大均值小方差） | — | **验证失败（FAIL）→ 拒绝采纳**（机器证据支撑 R1#3 否决；fp16/bf16 宽容差掩蔽，正是中心化式跨 dtype 保留的理由） |

形态覆盖：randn 梯度、大均值小方差 `x`（≈3e3 ± 0.5，raw-moment 抵消压力源）、常量 `x`（零方差通道）、大动态范围梯度、次正规点位、±inf、NaN、极值 `weight`、退化 shape N=1/2/S=1。

**优化结论**：采纳 O1/O2/O3。优化后公式（§3.1 唯一输入，`r = fl(1/L)`）：

$$
\text{scale} = w\cdot\text{rstd},\quad \text{per} = \text{scale}\cdot r,\quad
\text{shift} = -\text{per}\cdot\text{gb},\quad \text{xc} = \text{per}\cdot\text{gw}
$$
$$
\text{grad\_x} = \text{scale}\cdot\text{grad\_out} + \text{shift} - \text{xc}\cdot\hat{x}
$$

**否决项及依据**（负向断言举证）：
- raw-moment `gw = rstd·(Σ grad_out·x − mean·Σ grad_out)`：D-1 判别段 A 机器证据（fp32 违反率 0.1553），非纸面论证；
- fp16/bf16 原生域计算链（不经 fp32 中转）：`docs/Tilelang.language/数学操作/T.vmul.md` / `T.vsub.md` / `T.vadd.md` §2.2.1 bf16 列为 ×（无 Vector 算术）——fp32 中转为**必需**；
- 跨 dtype `T.copy` 作 cast 融合：`docs/Tilelang.language/内存操作/T.copy.md` §3 跨 dtype copy 不减少 vector pass 且舍入不可控——统一走显式 `T.vcast(rint)`。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态（GPU 源） | 向量替代方案（API 佐证） | 是否替代 | 不可替代理由 |
|---|--------|---------------------|--------------------------|---------|--------------|
| 1 | 逐通道平面 gather | `T.vectorized` 循环 `grad_out[head//S, bc, head%S+i]` | `T.copy` 双显式 slice（strided 区域，`docs/Tilelang.language/内存操作/T.copy.md`） | ✅ | — |
| 2 | 上 cast fp32 | `T.cast(..., "float32")` 逐元素 | `T.vcast(stage, x32, round_mode="rint")`（f16→f32 / bf16→f32 rint ✓，`docs/Tilelang.language/数据类型转换操作/T.vcast.md`） | ✅ | — |
| 3 | `x̂ = (x − mean)·rstd` | 逐元素 `(cast(x_held)-params[1])*params[2]` | `T.vbrc(m1, brc)` + `T.vsub(x32, brc, x32)` + `T.vbrc(r1, brc)` + `T.vmul(x32, brc, x32)`（`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2） | ✅ | — |
| 4 | 双 `+=` 累加 | 逐元素 `acc[0]+=g; acc_xhat[0]+=g*x_hat` | `T.vadd(acc_g, g32, acc_g)` + `T.vmul(g32, xh32, prod)` + `T.vadd(acc_gx, prod, acc_gx)`（`docs/Tilelang.language/数学操作/T.vadd.md` / `T.vmul.md`） | ✅ | — |
| 5 | 通道和 `gb` | warp shuffle 树 + shared 合并 | 两步 `T.reduce(g, red, dims=[1], reduce_mode="sum", clear=True)` + `T.reduce(red, gb, dims=[0], ...)`（`docs/Tilelang.language/规约操作/T.reduce.md` §2.2.2 / `T.reduce_sum.md`） | ✅ | — |
| 6 | 通道和 `gw` | 同 shuffle 树 | 同上（对 `prod`） | ✅ | — |
| 7 | epilogue 系数 | 逐线程标量 `scale/per/shift/xc` | (1,1) 上 `T.vmul`/`T.vsub` 标量 op（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 标量 ✓） | ✅ | — |
| 8 | `grad_x` epilogue | 逐元素复合标量式 | `T.vbrc(shift,xc)` 广播 + `T.vmul`/`T.vsub`（O2 分布式，`docs/Tilelang.language/shape操作/T.vbrc.md`） | ✅ | — |
| 9 | 下 cast + 写出 | 赋值隐式截断 + `T.vectorized` 写 | `T.vcast(gx32 → stage, rint)`（f32→f16 / f32→bf16 rint ✓）+ `T.copy` 双 slice | ✅ | — |
| 10 | 通道任务循环（persistent） | 一通道一 block（无核内循环） | `for i in T.serial(num_local_tasks)` + `if ch < C` 守卫 | ❌ 保留 | **block 级任务映射计算**：每 task 数个标量索引（ch），非逐元素热点；serial 循环是 persistent 分核（§0.6 R1）的载体，无逐元素等价 API |
| 11 | S-tile 循环（tiled 档） | 源无核内 S 循环（split kernel 跨 block） | `for t in T.serial(n_tiles)`（静态边界） | ❌ 保留 | **tile 级调度循环**：`n_tiles` 为 host 编译期常量；承载 UB 有界与两扫描（§0.6 R5），非逐元素计算 |
| 12 | 尾 tile 有效长度 | 逐元素 `if l < L` 谓词 | `real_s = T.min(bS, S - s0)`（标量，1 次/tile，`docs/Tilelang.language/比较操作/T.min.md`） | ❌ 保留 | **动态边界元数据**：S % bS ≠ 0 的尾 tile extent；per-tile 1 次标量 min + slice 界 |
| 13 | 逐通道标量装载 | 线程读 `weight[bc]/mean[bc]/rstd[bc]` | `T.copy(weight[c:c+1], w1)` 标量 slice | ✅ | — |

**向量化结论**：逐元素计算（gather / cast / x̂ / 累加 / epilogue / 舍回）与通道归约**已全部向量化/归约化**（替代 API 均有 docs+探针佐证，上表）。保留 3 类标量/循环——通道任务映射（#10）、S-tile 调度（#11）、尾 tile `T.min`（#12），逐项理由如上，无「实现简单」类理由。§6 循环结构与本表一致（kernel 内无逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

规约类算子必选候选：**水平归约（lane→归约轴）vs 垂直扫描（lane→独立行）**；另评估核内重排（转置）与跨核 split。归约轴是展平 `N·S`（非尾轴），核内布局把逐通道平面物化为 `(N, S)` 二维区域。

**轴质量评分**（主选向量化轴 = S；昇腾 Vector 128bit：fp16/bf16 ×8、fp32 ×4）：

| 评分项 | 判定 |
|--------|------|
| 整除性 | 逐通道平面行内 stride=1（尾轴 S）；fp16/bf16 向量宽 ×8：S=1024（128×8 ✓）；fp32 ×4：S=784（196×4 ✓） |
| 尾 lane 浪费率 | S=1024/784 均为向量宽整数倍（0%）；S=900（test 30×30）→ ceil(900/8)·8−900 = 4，浪费 0.44% |
| 累加链形态 | 归约维 = `dims=[1]`（S，连续 stride=1，硬件树归约）+ `dims=[0]`（N，小）；无跨步系数；N 维不参与向量化（外层 reduce） |
| repack 代价 | **无**（I/O 原生 `(N,C,S)` 行内连续 = 核内平面行；逐通道 gather 是 strided GM 区域，`T.copy` 直接支持，非重排） |
| UB 容量影响 | held 档 `N·S·18B ≤ 160KB`（§4.5）；tiled 档 `bS` 由预算反推（§4.5） |

**候选矩阵**（全部枚举，弃选必须给量化理由，禁止「未考虑」）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 采纳？ |
|---|---------|---------|------------|--------------|--------|
| 1 | I/O 原生 `(N,C,S)` + 逐通道 `(N,S)` 平面物化（held 单扫描 / tiled 两扫描） | S（尾轴连续；`reduce dims=[1]` 水平归约 + `dims=[0]`） | 无（GM strided slice 直载） | held 流量 3·L·C·eb（全候选最小）；tiled UB 有界 | ✅ **主选** |
| 2 | 核内 flatten 平面 `(N,S)→(L,)` 一维归约 | L | UB 内 `T.reshape`（`docs/Tilelang.language/shape操作/T.reshape.md`） | 与 #1 同流量；但 1D reduce 丢失 `(N,1)` 两步中间量、且 GM 侧平面是 strided 无法 flatten——无收益且 API 形态不如两步 `T.reduce` 有先例 | ❌（等价流量、无收益） |
| 3 | 转置布局 `(N,C,S)→(N,S,C)` 使通道尾轴 | S/C | host permute（百 µs 级）或核内 `T.transpose` 链 | 归约轴不变（仍是 `N·S`）；repack 纯开销；且 op 契约输出 `(N,C,S)` 需再转回 | ❌ |
| 4 | 垂直扫描（lane→独立行，串行步进 `N·S`） | 行 | 无 | 归约维访问跨步 → 无向量宽度；跨步累加退化为标量（结合 R3/R4 无向量 API 支撑） | ❌ |

**GPU 源码隐式轴选择的独立复核**（迁移任务要求）：GPU 把 `threads` 映射到通道内扁平 `L` 的连续段（`head = (k*threads+tx)*num_per_thread`，源码 L1708），即 lane 沿「连续访存轴」。本设计独立评估后**结论一致、理由不同**：NPU 选 S 为向量轴（行内 stride=1）因 v-op 广播形态是行内（`[M,N]↔[M,1]`，`docs/Tilelang.language/数学操作/T.vsub.md` §2.2.2）、且 held 档整平面驻留使 `grad_out/x` 单遍扫描（R3 流量最小）；GPU 的 warp shuffle 归约轴与 NPU `T.reduce` 归约轴**不一一对应**（GPU 跨线程树 vs NPU 二维两步 reduce），故归约结构按 NPU 硬件重设计（§0.6 R2）。

**实验裁决模式评估**：主选 #1（held）vs #2-tiled 的判定裕度 = 流量差（held 3 vs tiled 5，+66%）远超未实证常数的不确定区间 → 无需三件套；唯一未实证量是 **auto-multibuffer 膨胀系数**（决定 held 档 `bS=S` 是否需回退为 tiled）——这是**参数级**（bS 取值）而非结构级决策：主选结构对 held/tiled 均成立，设计期取保守预算（§4.5），bS 留 Stage 4 实测（编译期 UB 溢出报错即天然 guard）。**不启动设计期探针（D-3）**：结构决策不依赖该常数，且探针已实测 held 144KB 通过。

**布局决策结论**：选定 #1：核内布局 = I/O 原生 `(N,C,S)`、逐通道平面 `(N,S)` 物化到 UB、向量化轴 = S（尾轴）、repack = 无；held（`N·S·18B ≤ 160KB`）单扫描 / tiled 两扫描由 UB 预算切换。该结论同步落入 §3.3 伪代码与 §6 循环结构（归约/广播的内层向量维 = S；buffer 形状 = `(N, bS)`）；弃选方案的量化理由见候选矩阵。

---
## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Expert（Vector 路径）

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（`T.reduce` + v-prefix 逐元素 + `T.vbrc` 标量广播），无 matmul | 无 Cube/L0/L1 参与；单 `T.Scope("Vector")` |
| 归约 | `T.reduce` dims=[1]/[0] | `references/decision-tree.md`：`T.reduce` 在 Developer 与 Expert 两模式都可使用——模式选择取决于是否需显式控制 UB 层级与同步，而非用了哪个 API（故 Expert 不排斥本归约） |
| 内存层级 | 仅 GM↔UB，需显式 UB 放置 | `T.alloc_ub`（Expert 显式申请 UB，`docs/Tilelang.language/内存操作/T.alloc_ub.md`：`T.alloc_ub [Expert mode]`） |
| 同步 | 单 Vector scope 内顺序依赖链、无核间/跨引擎协作（one core per channel） | Vector scope 内默认顺序执行/自动流水；无需手动 `set_flag/wait_flag`（§7） |
| 先例 | `examples/conv1d/_conv1d_direct_kernel/_conv1d_direct_kernel.py`（纯 Vector、`T.Scope("Vector")` + `T.alloc_ub`、persistent，Expert 已交付，含 `T.reduce`/`T.vmul`/`T.vcast`） | Expert/Vector 形态已验证可行 |
| 探针 | `examples/batch_norm_bwd/probe/batch_norm_bwd_probe.py` 真机 PASS——其构造（strided `T.copy` slice / 两步 `T.reduce` / `T.vbrc` / `T.vcast`）均**模式无关**（decision-tree 明确 `T.reduce` 两模式可用），Expert 下由 `T.Scope("Vector")` + `T.alloc_ub` 承载 | 核心构造已真机验证 |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub`（显式 UB）——v-op 操作数必须在 UB（`docs/Tilelang.language/数学操作/T.vmul.md` §2.3.1） |
| Scope | 单 `with T.Scope("Vector"):`（无 Cube scope，非 MixCV） |
| 计算方式 | v-prefix API（vcast/vmul/vsub/vadd/vbrc）+ `T.reduce` + `T.min` 标量边界 |
| 同步 | Vector scope 内默认顺序/自动流水；无手动同步（§7） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)`（一维） |
| 一致性 | 实现期设 `ASCEND_MODE = "Expert"`；Stage 0 wrapper `examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py` 现为 `ascend_mode = "Developer"`，须在集成期同步改为 `"Expert"`（记入 §9.1 风险） |

---

## 3. API 映射设计

### 3.1 公式拆解

输入公式 = §1.6.1 优化后公式（fp32 计算域；`r = fl(1/L)`；`scale = w·rstd`、`per = scale·r`、`shift = per·gb`、`xc = per·gw`）：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `g32 = fp32(grad_out)`；`x32 = fp32(x)` | 上 cast（fp32 输入直拷） |
| 2 | `xhat = (x32 − mean)·rstd` | 逐通道标量广播 |
| 3 | `gb = Σ xhat 的 grad_out 和 = Σ g32`；`gw = Σ g32·xhat` | 双标量归约 |
| 4 | `scale = w·rstd`；`per = scale·r`；`shift = per·gb`；`xc = per·gw` | epilogue 系数（O1） |
| 5 | `grad_x = scale·g32 − shift − xc·xhat` | 分布式 epilogue（O2） |
| 6 | `grad_x_dtype = dtype_round(grad_x)` | 单次舍回 |
| 7 | `grad_bias = gb`；`grad_weight = gw` | 通道梯度输出 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 任务映射 | `c = i·K + cid` | `T.serial(num_local_tasks)` + `if c < C` | 静态边界（host 折叠） | Expert |
| 尾 tile | `real = min(bS, S−s0)` | `T.min(bS, S - s0)` | 标量（`docs/Tilelang.language/比较操作/T.min.md`） | Expert |
| 平面 gather | GM strided 区域→UB | `T.copy(grad_out[0:N, c, s0:s0+real], stage[0:N, 0:real])` | src/dst 双显式 slice，dst 零起点 | Expert |
| 上 cast | fp16/bf16→fp32 | `T.vcast(stage[0:N,0:real], g32[0:N,0:real], round_mode="rint")` | f16→f32 / bf16→f32 rint ✓ | Expert |
| 标量广播 | `[1,1]→[N,bS]` | `T.vbrc(m1, brc)` | src (1,1) → dst (N,bS) ✓（`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2） | Expert |
| `x̂` 中心化 | `(x−mean)·rstd` | `T.vsub(xh32, brc, xh32)` + `T.vmul(xh32, brc, xh32)` | 同形 (N,bS) 就地 | Expert |
| 归约（S） | `Σ` over S | `T.reduce(g32, red, dims=[1], reduce_mode="sum", clear=True/False)` | (N,bS)→(N,1)（`docs/Tilelang.language/规约操作/T.reduce.md` §2.2.2） | Expert |
| 归约（N） | `Σ` over N | `T.reduce(red, gb, dims=[0], reduce_mode="sum", clear=True)` | (N,1)→(1,1) | Expert |
| 通道梯度写 | `gb/gw → GM` | `T.copy(gb, grad_bias[c:c+1])` | 标量 slice | Expert |
| epilogue 系数 | `scale/per/shift/xc` | `T.vmul(w1, r1, coef)` 等 (1,1) 标量 op | 标量广播 ✓（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2） | Expert |
| epilogue | `scale·g − shift − xc·x̂` | `T.vbrc` + `T.vmul` + `T.vsub` | 同形 (N,bS) | Expert |
| 舍回 + 写出 | fp32→dtype→GM | `T.vcast(prod, stage, round_mode="rint")` + `T.copy(stage[0:N,0:real], grad_x[0:N, c, s0:s0+real])` | f32→f16 / f32→bf16 rint ✓ | Expert |

### 3.3 计算伪代码

> 按模板 §3.3 注记：`T.Tensor` 的 dtype 以第二个位置参数传入，不用 `dtype=` 关键字。**Expert 模式**：`with T.Kernel(...)` 内套 `with T.Scope("Vector"):`，UB 缓冲用 `T.alloc_ub`（`docs/Tilelang.language/内存操作/T.alloc_ub.md`）。工厂接口与源一致 `(N, C, S, dtype)`；移除 `threads/num_per_thread`（CUDA 执行模型产物，npuir `T.Kernel(threads=)` 无效果）。输出 `out_idx=[5, 6, 7]`（`grad_weight/grad_bias/grad_x`）。held/tiled 两档与 `if S>1`/`if N>1` reduce 退化分支在**工厂层**解析（trace 体之外，规避 parser 作用域陷阱）；held 档 `n_tiles == 1` 无 pass-2 重读。

```python
def _batch_norm_bwd_wide_kernel(N, C, S, dtype):
    L = N * S
    L_inv = 1.0 / float(L)                                  # O1: 编译期常量
    L_f = float(L)
    use_transit = dtype in ("float16", "bfloat16")          # bf16 无 Vector 算术 -> 必需中转
    bpe = 18 if use_transit else 16                          # stage 2 + g32/xh32/prod/brc 各 4
    UB_SAFE_BYTES = 163_840                                  # 160 KiB（探针 held 档实测 144KB 通过）
    vector_cores = NPUUtils.get().get_aicore_num() * 2       # 实查 20 -> 40（纯 Vector 翻倍，§5.5）
    held = N * S * bpe <= UB_SAFE_BYTES
    if held:
        bS_default = S
    else:
        vw = 8 if use_transit else 4                          # 向量宽
        bS_default = max(vw, ((UB_SAFE_BYTES // (N * bpe)) // vw) * vw)

    @tilelang.jit(out_idx=[5, 6, 7], target="npuir")
    def _func(bS=bS_default):
        if N * bS * bpe > UB_SAFE_BYTES:
            raise ValueError(f"bS={bS} exceeds UB budget for N={N}, dtype={dtype}")
        n_tiles = (S + bS - 1) // bS                          # held -> 1
        num_kernels = min(C, vector_cores)                    # 一维 persistent 分核（§5.5）
        num_local_tasks = (C + num_kernels - 1) // num_kernels  # 静态边界

        @T.prim_func
        def main(
            grad_out: T.Tensor((N, C, S), dtype),
            x: T.Tensor((N, C, S), dtype),
            weight: T.Tensor((C,), "float32"),
            mean: T.Tensor((C,), "float32"),
            rstd: T.Tensor((C,), "float32"),
            grad_weight: T.Tensor((C,), "float32"),           # out_idx=5
            grad_bias: T.Tensor((C,), "float32"),             # out_idx=6
            grad_x: T.Tensor((N, C, S), dtype),               # out_idx=7
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
              with T.Scope("Vector"):                       # Expert: 单 Vector scope
                stage = T.alloc_ub((N, bS), dtype)          # g/x 中转 -> gx 舍回
                g32 = T.alloc_ub((N, bS), "float32")
                xh32 = T.alloc_ub((N, bS), "float32")       # x_hat（held 档 pass-2 复用）
                prod = T.alloc_ub((N, bS), "float32")       # 乘积/广播 scratch
                brc = T.alloc_ub((N, bS), "float32")
                red = T.alloc_ub((N, 1), "float32")
                red2 = T.alloc_ub((N, 1), "float32")
                gb = T.alloc_ub((1, 1), "float32")
                gw = T.alloc_ub((1, 1), "float32")
                w1 = T.alloc_ub((1, 1), "float32")
                m1 = T.alloc_ub((1, 1), "float32")
                r1 = T.alloc_ub((1, 1), "float32")
                coef = T.alloc_ub((1, 1), "float32")        # scale
                per = T.alloc_ub((1, 1), "float32")
                shift = T.alloc_ub((1, 1), "float32")
                xc = T.alloc_ub((1, 1), "float32")

                for i in T.serial(num_local_tasks):
                    c = i * num_kernels + cid
                    if c < C:
                        T.copy(weight[c:c + 1], w1)
                        T.copy(mean[c:c + 1], m1)
                        T.copy(rstd[c:c + 1], r1)

                        # ---- pass 1: 逐 S-tile 累加 gb / gw ----
                        T.clear(red)
                        T.clear(red2)
                        for t in T.serial(n_tiles):
                            s0 = t * bS
                            real = T.min(bS, S - s0)
                            T.copy(grad_out[0:N, c, s0:s0 + real], stage[0:N, 0:real])
                            if use_transit:
                                T.vcast(stage[0:N, 0:real], g32[0:N, 0:real], round_mode="rint")
                            else:
                                T.copy(stage[0:N, 0:real], g32[0:N, 0:real])
                            T.copy(x[0:N, c, s0:s0 + real], stage[0:N, 0:real])
                            if use_transit:
                                T.vcast(stage[0:N, 0:real], xh32[0:N, 0:real], round_mode="rint")
                            else:
                                T.copy(stage[0:N, 0:real], xh32[0:N, 0:real])
                            T.vbrc(m1, brc)
                            T.vsub(xh32, brc, xh32)                 # x - mean
                            T.vbrc(r1, brc)
                            T.vmul(xh32, brc, xh32)                 # x_hat
                            T.vmul(g32, xh32, prod)                 # g * x_hat
                            # 累加（held: 整平面；tiled: 逐 tile，clear=False 跨 tile 累加）
                            T.reduce(g32, red, dims=[1], reduce_mode="sum", clear=False)
                            T.reduce(prod, red2, dims=[1], reduce_mode="sum", clear=False)
                        T.reduce(red, gb, dims=[0], reduce_mode="sum", clear=True)
                        T.reduce(red2, gw, dims=[0], reduce_mode="sum", clear=True)
                        T.copy(gb, grad_bias[c:c + 1])
                        T.copy(gw, grad_weight[c:c + 1])

                        # ---- epilogue 系数（O1/O2） ----
                        T.vmul(w1, r1, coef)                    # scale = w * rstd
                        T.vmul(coef, L_inv, per)                # per = scale / L
                        T.vmul(per, gb, shift)                  # shift = per * gb（正号）
                        T.vmul(per, gw, xc)                     # xc = per * gw

                        # ---- pass 2: grad_x = scale*g - shift - xc*x_hat ----
                        for t in T.serial(n_tiles):
                            s0 = t * bS
                            real = T.min(bS, S - s0)
                            T.copy(grad_out[0:N, c, s0:s0 + real], stage[0:N, 0:real])
                            if use_transit:
                                T.vcast(stage[0:N, 0:real], g32[0:N, 0:real], round_mode="rint")
                            else:
                                T.copy(stage[0:N, 0:real], g32[0:N, 0:real])
                            T.copy(x[0:N, c, s0:s0 + real], stage[0:N, 0:real])
                            if use_transit:
                                T.vcast(stage[0:N, 0:real], xh32[0:N, 0:real], round_mode="rint")
                            else:
                                T.copy(stage[0:N, 0:real], xh32[0:N, 0:real])
                            T.vbrc(m1, brc); T.vsub(xh32, brc, xh32)
                            T.vbrc(r1, brc); T.vmul(xh32, brc, xh32)   # x_hat
                            T.vbrc(coef, brc)
                            T.vmul(g32, brc, prod)                  # scale * g
                            T.vbrc(shift, brc)
                            T.vsub(prod, brc, prod)                 # - per*gb
                            T.vbrc(xc, brc)
                            T.vmul(xh32, brc, brc)                  # xc * x_hat
                            T.vsub(prod, brc, prod)                 # - xc*x_hat
                            if use_transit:
                                T.vcast(prod, stage, round_mode="rint")
                            else:
                                T.copy(prod[0:N, 0:real], stage[0:N, 0:real])
                            T.copy(stage[0:N, 0:real], grad_x[0:N, c, s0:s0 + real])

        return main

    return _func
```

> 尾块说明：tiled 档 v-op 作用于完整 `(N, bS)` buffer，尾 tile `[real:]` 列为未初始化垃圾——按列/行隔离（`reduce dims=[1]` 逐行、广播逐元素），装载/写出的双显式 slice 只取有效 extent（`examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/DESIGN.md` §0.6 R3 同款）。**held 档优化**：`n_tiles == 1` 时 pass-1 的 `g32/xh32` 即 epilogue 所需，实现层以工厂层 `if held:` 分支全省 pass-2 重读（单扫描；O3 的 `x̂` CSE 即此）。**接口保持**：`main(grad_out, x, weight, mean, rstd, grad_weight, grad_bias, grad_x)` 参数名/顺序与源 `_bn_bwd_wide` 一致；源 `out_idx=[-1]` 改为 `out_idx=[5,6,7]`（三输出统一由 jit 分配，host 不再预分配通道梯度）。

### 3.4 API 可行性确认

| API | 文档路径 / 佐证 | 验证状态 | 备注 |
|-----|----------------|----------|------|
| `T.Kernel(一维, is_npu=True)` | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` | ✅ 已核对 | 1D；`(cid, _)` 解包 |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md` | ✅ 已核对 | Expert 显式 UB；1D~2D ✓ |
| `T.copy`（双显式 slice） | `docs/Tilelang.language/内存操作/T.copy.md` | ✅ 已核对 | GM↔UB ✓；strided 区域（源 slice 非单位步长）✓ |
| `T.vcast` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` | ✅ 已核对 | f16→f32 rint ✓；bf16→f32 rint ✓；f32→f16 rint ✓；f32→bf16 rint ✓；shape 须一致 |
| `T.reduce`（dims/clear） | `docs/Tilelang.language/规约操作/T.reduce.md` | ✅ 已核对 | fp16/fp32 ✓ bf16 ×；`(N,bS)→(N,1)`、`(N,1)→(1,1)`；`clear=False` 跨 tile 累加 |
| `T.reduce_sum` | `docs/Tilelang.language/规约操作/T.reduce_sum.md` | ✅ 已核对 | `(M,N)→(M,1)`；等价可用 |
| `T.vmul`/`T.vsub`/`T.vadd` | `docs/Tilelang.language/数学操作/T.vmul.md` / `T.vsub.md` / `T.vadd.md` | ✅ 已核对 | fp16/fp32 ✓ bf16 ×；行广播 `[M,N]↔[M,1]`、列广播 `[1,N]↔[1,1]`、标量 ✓；就地 ✓ |
| `T.vbrc` | `docs/Tilelang.language/shape操作/T.vbrc.md` | ✅ 已核对 | src (1,1)→(N,bS) 合法；bf16 支持 |
| `T.min` | `docs/Tilelang.language/比较操作/T.min.md` | ✅ 已核对 | 标量 min（fp16/fp32/int ✓），尾 tile extent |
| `T.clear` | `docs/Tilelang.language/内存操作/T.clear.md` | ✅ 已核对 | 累加器初始化 |
| `T.serial` + if 守卫 | `examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/DESIGN.md` §3.3 | ✅ 佐证 | persistent grid-stride 同款 |
| 探针真机 | `examples/batch_norm_bwd/probe/batch_norm_bwd_probe.py` | ✅ 真机 PASS | strided plane gather / 双 reduce / vbrc / vcast / held + 超大 S tiled |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(num_kernels, is_npu=True)` |
| GPU 专用 API | Yes | `T.shfl_xor`/`T.shift_left`/`T.sync_threads`/`T.get_thread_binding`/`threads=` 不迁移（§0.5 舍弃清单） |
| GEMM 分形/非整除 | No | 无 GEMM；S 非整除经 `T.min`+slice |
| L0C 容量 | No | 纯 Vector，无 L0 |
| 物理核数适配 | Yes | §5.5 三要素（实查 20→40） |
| bf16 无 Vector 算术 | Yes | §0.6 R4 fp32 中转链（必需） |
| 中间轴归约（非尾轴） | Yes | 逐通道 `(N,S)` 平面 strided gather + 两步 `T.reduce`（探针验证） |

#### 3.5.2 参考实现差异说明（GPU → NPU，汇总影响 API 选型项；完整差异见 §0.5/§0.6）

| 差异项 | GPU 源 | 本项目（Ascend） | 转换方案 |
|--------|--------|-----------------|----------|
| threads 参数 | `T.Kernel(C, threads=)` | npuir 无效 | 移除；一维 persistent `T.Kernel(num_kernels)` |
| register 计算域 | `T.alloc_local` | v-op 要求 UB 操作数 | 全部 `T.alloc_ub` |
| warp shuffle 归约 | `T.shfl_xor` butterfly | 无 warp | `T.reduce` 两步（§0.6 R2） |
| `T.vectorized` 逐元素循环 | 线程内向量化 | v-prefix 向量 op | `T.copy`/`T.vcast`/v-op（§1.6.2） |
| 逐元素谓词（`if l<L`） | masked load/store | `T.min`+双显式 slice | §0.6 R3 |
| 隐式 dtype 截断 | `T.cast` 赋值 | 显式 `T.vcast(rint)` | §0.6 R4 |
| split kernel（跨 block） | `[C,splits]`+finalize | 单核 S-tile 两扫描 | §0.6 R5 |
| `out_idx=[-1]` | 仅 grad_x 自动 | `out_idx=[5,6,7]` | 三输出统一分配 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/conv1d/_conv1d_direct_kernel/_conv1d_direct_kernel.py` | 高（**Expert/Vector** 纯 Vector 迁移先例，已交付） | `ascend_mode="Expert"` + 单 `T.Scope("Vector")` + `T.alloc_ub`、persistent 分核、`T.reduce`/`T.vmul`/`T.vcast` fp32 中转、尾块 slice 截断 |
| `examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/DESIGN.md` | 极高（同族行归约迁移，已交付；Developer streaming 形态） | persistent 分核、尾块 `T.min`+双 slice、fp32 计算域、bf16 fp32 中转、UB 预算表、分核实查 |
| `examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py` | 极高（本算子 NPU wrapper） | 接口参数顺序、`_AI_CORE_COUNT=40` 分核策略、`ascend_mode` 集成期须由 `"Developer"` 改 `"Expert"`（§2.3） |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/DESIGN.md` | 高（同族 row-reduction 设计） | 尾块 `T.min`+双显式 slice+垃圾行丢弃、`(bm,1)` reduce 输出形态 |
| `examples/batch_norm_bwd/probe/batch_norm_bwd_probe.py` | 极高（本算子可行性探针，真机 PASS） | strided plane gather、两步 `T.reduce`、`T.vbrc` 广播、held/tiled 结构 |
| `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/_make_lerp_tensor_kernel.py` | 高（persistent 分核已调优） | `min(logical, cores)` 分核 + grid-stride serial、UB 预算 guard |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `grad_out` | `(N, C, S)` | float16 / bfloat16 / float32 | 上游梯度；N=C 轴外、S=spatial 乘积 |
| `x` | `(N, C, S)` | same_as(grad_out) | 原始输入 |
| `weight` | `(C,)` | float32 | 仿射缩放 |
| `mean` | `(C,)` | float32 | 前向保存 mean |
| `rstd` | `(C,)` | float32 | 前向保存 rstd |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `grad_x` | `(N, C, S)` | same_as(grad_out) | `out_idx=7` 自动分配 |
| `grad_weight` | `(C,)` | float32 | `out_idx=5` 自动分配 |
| `grad_bias` | `(C,)` | float32 | `out_idx=6` 自动分配 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途（角色复用时序） |
|-----------|-------|-------|----------|---------------------|
| `stage` | `(N, bS)` | dtype | UB | g 中转 → x 中转 → gx 舍回中转 |
| `g32` | `(N, bS)` | float32 | UB | grad_out 平面（held 档 pass-2 复用） |
| `xh32` | `(N, bS)` | float32 | UB | x 32 → x_hat（held 档 pass-2 复用） |
| `prod` | `(N, bS)` | float32 | UB | `g·x̂` 乘积 / `scale·g` / 减法累加 |
| `brc` | `(N, bS)` | float32 | UB | `T.vbrc` 标量广播 scratch |
| `red`/`red2` | `(N, 1)` | float32 | UB | dims=[1] 归约中间/累加器 |
| `gb`/`gw`/`w1`/`m1`/`r1`/`coef`/`per`/`shift`/`xc` | `(1, 1)` | float32 | UB | 逐通道标量（各 4B，可忽略） |

### 4.4 内存搬运路径

```
纯 Vector 路径（无 Cube/L0/L1 参与）：
GM[grad_out] --T.copy(双slice)--> UB[stage] --vcast(rint)--> UB[g32]
GM[x]        --T.copy(双slice)--> UB[stage] --vcast(rint)--> UB[xh32]
UB[xh32] --vbrc(m1)+vsub+vbrc(r1)+vmul--> UB[xh32 = x_hat]
UB[g32]·UB[xh32] --vmul--> UB[prod] --T.reduce(dims=[1],clear=False)--> UB[red] --T.reduce(dims=[0])--> UB[gb]
UB[g32] --T.reduce(dims=[1],clear=False)----------------------------------> UB[red2] --T.reduce(dims=[0])--> UB[gw]
UB[gb]/UB[gw] --T.copy--> GM[grad_bias]/GM[grad_weight]
UB[w1]·UB[r1] --vmul--> UB[coef]; ... --> UB[per/shift/xc]  (标量预折叠)
UB[g32],UB[xh32] + UB[coef/shift/xc] --vbrc+vmul+vsub--> UB[prod=grad_x]
UB[prod] --vcast(rint)--> UB[stage] --T.copy(双slice)--> GM[grad_x]
（全部计算在 UB 内闭环，无中间 GM 往返；held 档 n_tiles=1 单扫描无 pass-2 重读）
```

关键约束：v-op 操作数必须在 UB（`docs/Tilelang.language/数学操作/T.vmul.md` §2.3.1）；GM 不可直达 fragment。

### 4.5 UB 内存预算

预算目标：总驻留 ≤ UB 容量 192KB，取 `UB_SAFE_BYTES = 163_840`（160 KiB）——依据：**探针 held 档（`examples/batch_norm_bwd/probe/batch_norm_bwd_probe.py`）在 `(N=8,C=64,S=1024)` fp16 分配 `stage+g+xh+prod+brc` ≈ 144KB 真机编译运行 PASS**；160KB 留 32KB 余量。BPE 口径：fp16/bf16 = 18 B/元素（stage 2 + g32/xh32/prod/brc 各 4）；fp32 = 16 B/元素（无 stage）。总驻留 = `BPE·N·bS` + `(N,1)` 小项。

**held/tiled 判定与 bS**：`held = N·S·BPE ≤ 163_840`（held 时 `bS = S`）；否则 `bS = align_down(163_840 // (N·BPE), vw)`（vw = 向量宽 8/4）。

| workload (N,C,S) | dtype | N·S·BPE | 档 | bS | n_tiles | 驻留 (B) |
|------------------|-------|---------|----|----|---------|----------|
| resnet-fc (32,64,1) | fp16 | 576 | held | 1 | 1 | 576 |
| resnet-stage1 (8,64,1024) | fp16/bf16 | 147,456 | held | 1024 | 1 | 147,456 |
| resnet-stage2 (4,128,1024) | fp16 | 73,728 | held | 1024 | 1 | 73,728 |
| resnet-stage3 (4,256,784) | fp16 | 56,448 | held | 784 | 1 | 56,448 |
| resnet-stage3 (4,256,784) | fp32 | 50,176 | held | 784 | 1 | 50,176 |
| large-spatial (4,128,1048576) | fp16 | 75,497,472 | tiled | 2048 | 512 | 147,456 |
| test (4,64,4096) | fp16 | 294,912 | tiled | 2048 | 2 | 147,456 |

默认公式：`bS_default = S if held else align_down(163_840 // (N·BPE), vw)`；工厂层 guard 拒绝超预算 `bS`（编译期 UB 溢出报错为天然兜底）。**注**：auto-multibuffer 可能膨胀（TRAP-UB-multibuffer-inflation），但探针 held 档实测 144KB 通过表明本结构（显式 `alloc_ub`）膨胀有限；held 临界（resnet-stage1 147KB）留 Stage 4 实测，超预算即回退 tiled。

### 4.6 动态轴定义

无运行时动态轴。N、C、S、bS 均为工厂期常量（per-build JIT 特化，源 lru_cache 工厂同款语义）；尾 tile extent `real_s` 是 kernel 内 `T.min` 标量。op 层把任意 `(N, C, *spatial)` view 成 `(N, C, S)`——host 元数据视图操作。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[5, 6, 7], target="npuir")   # grad_weight/grad_bias/grad_x
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector

**判定依据**：仅 `T.reduce` + v-prefix 逐元素运算 + `T.vbrc` 广播，无 matmul。

### 5.2 Block 划分

```python
bS = S if (N * S * BPE <= UB_SAFE_BYTES) else align_down(UB_SAFE_BYTES // (N * BPE), vw)  # §4.5
num_logical = C                              # 逻辑任务数 = 通道数（一通道一任务）
num_kernels = min(num_logical, 40)           # 实际启动内核数（≤ 物理核，§5.5）
num_local_tasks = ceil(num_logical / num_kernels)  # 核内串行任务数（静态）
```

**选择理由**：归约轴 `N·S` 的维度上——S（尾轴）作为**向量化/归约内维**（连续 stride=1），N 作为外层 reduce 维；整通道平面驻留 UB（held）使 `grad_out/x` 单遍扫描（R3 流量最小）；超大 S 用核内 S-tile（tiled）保证 UB 有界。

### 5.3 约束分析

- **对齐约束**：UB 32B 对齐由编译器管理（`docs/Tilelang.language/内存操作/T.alloc_ub.md`）；逐通道平面行内 stride=1；S 尾 lane 浪费率见 §1.6.3（≤ 0.5%）。
- **UB 容量**：§4.5 表，全负载域 held ≤ 147,456B、tiled ≤ 147,456B（< 160KB 预算）✓。
- **L0/L1/分形**：不适用（纯 Vector）。

### 5.4 注意事项

- **S % bS ≠ 0（tiled 尾 tile）**：`real_s = T.min(bS, S − s0)`；装载/写出双显式 slice 截断；reduce 在有效 extent 上求和。
- **N=1（单 batch）**：`(1, bS)` 平面；`dims=[0]` reduce 在 size-1 上退化（`(1,1)→(1,1)`），实现层 `if N > 1:` 工厂分支跳过（避免非法 reduce；等价地在 `red` 上取即得）。
- **S=1（resnet-fc）**：平面 `(N,1)`，`dims=[1]` reduce 在 size-1 上退化 → 实现层 `if S > 1:` 工厂分支跳过第一层 reduce，直接 `(N,1) --dims=[0]--> (1,1)`（`T.reduce` Shape 规则要求恰好一维不同，size-1 维不满足）。
- **held 档 pass-2 省重读**：`n_tiles == 1` 时 epilogue 复用 pass-1 的 `g32/xh32`（O3，单扫描）。
- **wrapper 默认 bS**：NPU wrapper 由工厂默认治理；集成期/Stage 4 以 §4.5 表值或扫描值覆盖。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 docs/开发指南.md §3.3）。

- **物理核数**：**20**（AI Core，Ascend910B3）。实查记录（2026-10-14，本设计执行）：

```python
from tilelang.utils.npu_utils import NPUUtils
n = NPUUtils.get().get_aicore_num()   # -> 20
vector_cores = n * 2                  # 纯 Vector 算子翻倍 -> 40
```

  与 `constants.md` CONST-aicore-910B2C 的 2026-10-12 注（本机 `get_aicore_num()` 返回 20，设备档 910B3）一致；禁止按 24 假设。
- **逻辑核数**：`num_logical = C`（一通道一逻辑任务）：

| workload | shape (N,C,S) | dtype | logical | num_kernels=min(C,40) | num_local_tasks |
|----------|---------------|-------|---------|-----------------------|-----------------|
| resnet-fc | 32×64×1 | fp16 | 64 | 40 | 2 |
| resnet-stage1 | 8×64×1024 | fp16/bf16 | 64 | 40 | 2 |
| resnet-stage2 | 4×128×1024 | fp16 | 128 | 40 | 4 |
| resnet-stage3 | 4×256×784 | fp16/fp32 | 256 | 40 | 7 |
| large-spatial | 4×128×1048576 | fp16 | 128 | 40 | 4 |

- **规模判定与分核方案**：全部 manifest 的 `C ∈ {64,128,256} > 40` → **极大规模分支**——固定启动 `num_kernels = 40`，核内 `T.serial(num_local_tasks)` grid-stride（`c = i·40 + cid`），`if c < C` 守卫；**循环边界静态**（`num_local_tasks` 为 host 层编译期常量）。负载不均 ≤1 任务（64=40+24、128=40·3+8、256=40·6+16）。该形态消除逻辑核超发的串行调度开销（core-split-strategy §1），并获 grid-stride 聚集窗口的 MTE2 访问效益（`constants.md` CONST-mte2-degradation 实测 +3.2%）。**通道归约完全在单核内闭环**（one core per channel），无需核间 `T.sync_block_set/wait` 或 workspace。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表仅 block/tile 级调度循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 通道任务维 | 核级并行 + 核内串行 | `T.Kernel(num_kernels)` + `T.serial(num_local_tasks)` | persistent 分核（§5.5）；通道间独立 |
| S-tile 维（tiled 档） | 核内串行 | `T.serial(n_tiles)`（pass 1 / pass 2） | UB 有界（§0.6 R5）；`n_tiles` 静态 |
| 元素维（S） | 向量化（无循环） | `T.copy`/`T.vcast`/v-op/`T.reduce` 全 buffer 操作 | §1.6.2 #1-#9 |
| 尾 tile | 标量 min + slice | `T.min` + 双显式 slice | §1.6.2 #12 |

### 6.2 循环伪代码

见 §3.3（结构：`with T.Kernel(num_kernels, is_npu=True) as (cid, _):` → `with T.Scope("Vector"):` → buffer 分配（`T.alloc_ub`，serial 外，跨 task 复用）→ `for i in T.serial(num_local_tasks):` → `if c < C:` → pass-1 累加归约 → epilogue 系数 → pass-2 写 grad_x）。

### 6.3 流水线优化

**不使用显式 `T.Pipelined`**。理由：单任务内为顺序依赖链（装载→cast→x̂→累加→reduce→系数→epilogue），无 K 维迭代可重叠；任务间装载/计算 overlap 交由编译器 auto multi-buffer。Stage 4 选项：显式双缓冲（预取下一 S-tile）或 `T.Pipelined` 包裹 serial 循环——若 baseline 偏离 §1.6.0 估算下界 >2x 再评估。

### 6.4 尾块处理

**tiled 尾 tile**：`S % bS ≠ 0` 时最后一个 tile 的 `real_s < bS`——装载/写出双显式 slice 截断。**通道尾任务**：`C % num_kernels ≠ 0` 时 `if c < C` 守卫跳过越界通道。**N=1 / S=1**：工厂层分支跳过退化的 reduce 层（§5.4）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：Expert（显式 UB + 单 `T.Scope("Vector")`）；无核间/跨引擎协作。

### 7.2 同步点说明

Expert 模式下 UB 分配与数据放置显式，但**单个 Vector scope 内的顺序依赖链仍由编译器按 MTE2→V→MTE3 拓扑自动插 pipe_barrier**（`examples/conv1d/_conv1d_direct_kernel/_conv1d_direct_kernel.py` 同款 Expert/Vector 形态：`T.copy`/`T.vcast`/`T.vmul`/`T.reduce` 顺序书写，未手写 sync 即编译运行）。本算子无跨引擎（无 Cube）与跨核协作，故无需 `T.sync_block_set/wait`、无需 `T.set_flag/T.wait_flag`：

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy`(GM→UB) 后 | 编译器自动 | MTE2→V 拓扑自动 pipe_barrier |
| v-op / reduce / vbrc / vcast 之间 | 编译器自动（顺序执行） | 同 Vector 引擎顺序依赖 |
| `T.copy`(UB→GM) 前 | 编译器自动 | V→MTE3 拓扑自动 pipe_barrier |
| serial 通道任务 / S-tile 间 | 编译器自动 | 同核串行，无并发 |
| 跨核（通道间） | 无需 | one core per channel，通道归约单核闭环；无 workspace/atomics |

**显式同步备选**：若 Stage 3 引入显式双缓冲（预取下一 S-tile）后实测出现 WAR/RAW 冒险，按 `references/decision-tree.md` 用 `T.set_flag`/`T.wait_flag`（Expert 手动同步形态）管理；本设计默认不引入（见 §6.3）。

### 7.3 pass_configs 配置

无特殊 pass_configs。Expert Vector scope 默认编译流水线（`examples/conv1d/_conv1d_direct_kernel/_conv1d_direct_kernel.py` 同款）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 §0.1 源算子语义为唯一依据（移植源仓参考实现 `examples/TileOPs/tileops/workloads/norm.py 的 batch_norm_backward`），**不复刻 §0.6 的 NPU 算法**（无 vcast / fp32 中转 / persistent / 两步 reduce / 分布式 epilogue）——保证验证独立性。`mean/rstd` 由一次前向（`torch.var_mean` + `rsqrt`）产生。

```python
import torch

def golden_batch_norm_bwd(grad_out, x, weight, mean, rstd):
    """独立纯 PyTorch CPU BN 反向（fp32 中间量 + 舍回输入 dtype）。

    输入: grad_out/x (N, C, S) 同形同 dtype；weight/mean/rstd (C,) fp32
    输出: (grad_x, grad_weight, grad_bias) —— grad_x 同输入 dtype；通道梯度 fp32
    语义（§0.1）: x_hat = (x - mean) * rstd；grad_bias = sum_{N,S}(grad_out)；
                  grad_weight = sum_{N,S}(grad_out * x_hat)；
                  grad_x = (w*rstd/L) * (L*grad_out - grad_bias - x_hat*grad_weight)
    """
    N, C, S = grad_out.shape
    L = N * S
    shape = (1, C, 1)
    x_hat = (x.float() - mean.view(shape)) * rstd.view(shape)
    grad_bias = grad_out.float().sum(dim=(0, 2))            # (C,)
    grad_weight = (grad_out.float() * x_hat).sum(dim=(0, 2))  # (C,)
    gb = grad_bias.view(shape)
    gw = grad_weight.view(shape)
    grad_x = (weight.view(shape) * rstd.view(shape) / L) * (
        L * grad_out.float() - gb - x_hat * gw
    )
    return grad_x.to(x.dtype), grad_weight, grad_bias
```

**独立性说明**：golden 直接调用 `torch` 归约（`sum`）与逐元素运算，不含 NPU 设计中的 persistent 分核、两步 `T.reduce`、`T.vbrc` 广播、fp32 中转链、分布式 epilogue 等任何算法特征；NPU 算法若与 golden 数值一致即证明迁移正确。另可用 `examples/TileOPs/tests/ops/test_batch_norm_bwd.py 的 BatchNormBwdTest` 的 `gen_inputs`（用 `torch.var_mean` 从 `x` 生成自洽的 mean/rstd）作为入口。

### 8.2 精度标准

来自 `examples/TileOPs/tileops/workloads/norm.py 的 batch_norm_bwd_tolerance`（`grad_x` 逐元素；通道梯度用更宽的累加界 `sum_atol`）：

| dtype | atol | rtol | sum_atol（通道梯度） |
|-------|------|------|----------------------|
| float32 | 1e-5 | 1e-5 | 1e-4 |
| float16 | 1e-2 | 1e-2 | 1e-2 |
| bfloat16 | 2e-2 | 2e-2 | 2e-2 |

fp32 中转链对齐 golden opmath（D-1 verify_equiv：fp16/bf16 舍回后逐位一致）；探针最坏 max_abs_err 4.883e-04（fp16 grad_x）在容差内。

### 8.3 L0 门槛测试计划

L0 聚焦「编译通过 + 基本精度正确」（完整 L1/L2/Boundary 分层套件交由 tilelang-op-develop）。对照 = §8.1 golden（CPU）。

| # | Shape (N, C, S) | dtype | 档 | 验证目标 | 容忍度 |
|---|-----------------|-------|----|---------|--------|
| L0-1 | (32, 64, 1) | fp32 | held | resnet-fc（S=1 退化 reduce 路径） | 1e-5 |
| L0-2 | (32, 64, 1) | fp16 | held | resnet-fc fp16 | 1e-2 |
| L0-3 | (8, 64, 1024) | fp16 | held | resnet-stage1（held 临界 BPE） | 1e-2 |
| L0-4 | (8, 64, 1024) | bf16 | held | resnet-stage1 bf16（fp32 中转） | 2e-2 |
| L0-5 | (4, 128, 1024) | fp16 | held | resnet-stage2 | 1e-2 |
| L0-6 | (4, 256, 784) | fp32 | held | resnet-stage3 fp32（非 2^k S） | 1e-5 |
| L0-7 | (4, 256, 784) | fp16 | held | resnet-stage3 fp16 | 1e-2 |
| L0-8 | (4, 128, 1048576) | fp16 | tiled | large-spatial（512 S-tile，尾 tile 截尾） | 1e-2 |
| L0-9 | (4, 64, 4096) | fp16 | tiled | 中档 tiled（2 S-tile） | 1e-2 |
| L0-10 | (8, 64, 900) | fp16 | held | test 30×30（S 非向量宽整数倍，尾 lane） | 1e-2 |
| L0-11 | (2, 8, 3) | fp16 | held | 微型 shape（N,S 均小） | 1e-2 |
| L0-12 | (1, 4, 16) | bf16 | held | N=1 退化 reduce 路径 | 2e-2 |

**通过条件**：全部项通过 atol/rtol（`grad_x` 逐元素 + `grad_weight/grad_bias` 用 `sum_atol`）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 处理方案 |
|------|------|----------|
| **held 档 UB 临界**（resnet-stage1 147KB，预算 160KB） | 若 auto-multibuffer 膨胀超 32KB 余量则编译失败 | 工厂 guard + 回退 tiled 档（`bS < S`，两扫描）；探针 held 档 144KB 真机通过佐证余量 |
| **中间轴归约（非尾轴）** | 逐通道平面 strided gather | `T.copy` 双显式 slice（探针真机验证）；行跨步 `C·S` |
| **bf16 无 Vector 算术** | bf16 直算编译失败 | fp32 中转链（§0.6 R4，必需） |
| **N=1 或 S=1 的退化 reduce** | `(1,1)` 上 dims=[1]/[0] reduce 非法（无维度变化） | 工厂层 `if N > 1:` / `if S > 1:` 分支跳过对应 reduce 层（§5.4） |
| **N 无对齐 padding** | reduce 在有效 extent 上求和 | `T.min` 截尾 + 双显式 slice |
| **未初始化垃圾列（tiled 尾 tile）** | 尾 tile 垃圾列统计量无意义 | 按列隔离（reduce dims=[1] 逐行/广播逐元素），slice 截断；L0-8/L0-9 捕获 |
| **roofline 常数 stale** | §1.6.0 估算引用 CONST-* 被 kb_stale_check 标记 | 估算仅作排序与 Stage 4 对账 |
| **通道任务负载不均** | C 非 40 整数倍（64/128/256）→ 部分核多 1 任务 | 不均 ≤1 任务（≤ 3.8%）；Stage 4 可评估分核粒度 |
| **inf/NaN 传播** | 全 inf 通道 → NaN 输出（公式自然传播） | 与 golden/源一致；测试用例为有限随机输入 |
| **逐通道标量装载** | 每通道 3 次 `T.copy` 标量 slice | 少量标量，非热点 |
| **wrapper `ascend_mode` 不一致** | Stage 0 wrapper `batch_norm_bwd.py` 现为 `ascend_mode = "Developer"`，本设计选 Expert | 集成期（Stage 5）把 wrapper 的 `ascend_mode` 改为 `"Expert"`；实现期 `ASCEND_MODE = "Expert"`（§2.3） |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | bS 超预算 / held 膨胀超余量 | 编译失败 `ub overflow` | 降 bS / 回退 tiled；§4.5 表 |
| bf16 直传 v-op/reduce | 未做 vcast fp32 | 编译错误（dtype 不支持） | fp32 中转链 |
| reduce 退化 | N=1 或 S=1 仍做 dims=[1]/[0] | MLIR verify fail（无维度变化） | 工厂层 `if N>1` / `if S>1` 分支 |
| `clear=False` 未初始化 | 跨 tile 累加前未 `T.clear(red)` | 静默数值错误（`docs/Tilelang.language/规约操作/T.reduce.md` §2.3） | 累加前 `T.clear` |
| 跨 dtype `T.copy` 当 cast | GM fp16 → UB fp32 一步 | 隐藏 VCast 舍入不可控（`docs/Tilelang.language/内存操作/T.copy.md` §3） | 显式 `T.vcast(rint)` + 同 dtype copy |
| 列广播直接 vsub | 想省 vbrc 用 `[N,bS]−[1,1]` | 未文档化形态（文档仅列 `[M,N]−[M,1]`/`[1,N]−[1,1]`/标量） | `T.vbrc` 显式展开（探针同款） |
| out_idx 顺序反 | `out_idx=[7,5,6]` | wrapper 三输出错位 | `out_idx=[5,6,7]`（grad_weight, grad_bias, grad_x，对齐源参数序） |
| if 体内新定义变量跨块引用 | TVM parser 作用域 | Undefined variable | if 体内只写预分配 buffer；dtype/held 分支置于工厂层 |
| 通道索引常量折叠遗漏 | `num_local_tasks` 非常量 | 动态循环边界 | host 层编译期常量（§5.5） |

### 9.3 特殊场景处理

| 场景 | 处理方式 |
|------|----------|
| S=1（resnet-fc） | 平面 `(N,1)`；跳过 `dims=[1]` reduce，直接 `dims=[0]`；向量化轴退化为 N（小负载，延迟 bound） |
| N=1 | `(1, bS)`；`dims=[0]` reduce 退化 → 工厂分支跳过；L0-11/L0-12 覆盖 |
| large-spatial（S=1048576） | tiled 档，512 S-tile；pass-1 累加、pass-2 重读；无跨核归并 |
| 3D 输入 `(N, C, H, W)` | op 层 reshape `(N, C, H*W)` 后调 kernel（host 元数据视图） |
| C > 40 | persistent 分核（§5.5）；核内 serial 多通道 |
| 零方差通道（x 常量） | `x̂ = 0`；`gw = 0`；grad_x = scale·g（公式自然）；verify_equiv 角点已覆盖 |
| 全 inf / NaN 通道 | 自然传播（与源/golden 一致）；verify_equiv 角点已覆盖 |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/
├── DESIGN.md                          # 本设计文档
├── verify_equiv.py                    # D-1 等价性机器验证脚本（已执行，EQUIV_PASS）
├── _batch_norm_bwd_wide_kernel.py     # 算子实现（Stage 3 产出）
└── RETROSPECTIVE.md                   # Stage 1 复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成并执行 | O1+O2 EQUIV_PASS（21 案例）；raw-moment 拒绝证据（判别段 A） |
| `_batch_norm_bwd_wide_kernel.py` | ⬜ 待实现（Stage 3） | 工厂 + kernel + golden + L0 测试 |

### 10.3 命名规范

- 项目目录名：`batch_norm_bwd`；算子目录名 / 实现文件：`_batch_norm_bwd_wide_kernel`（snake_case，不裁剪不变换）
- Golden 函数：`golden_batch_norm_bwd(grad_out, x, weight, mean, rstd)`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）——本文档
2. ✅ 等价性机器验证（verify_equiv.py，D-1）
3. ⬜ Golden 函数（§8.1，验证基准）
4. ⬜ 算子实现 `_batch_norm_bwd_wide_kernel.py`（§3.3 伪代码 → 代码）+ L0 门槛测试（§8.3 全 12 项）
5. ⬜ 精度比对（vs golden，三输出）

### 10.5 迁移规则遵守确认

| 规则 | 遵守情况 | 说明 |
|------|----------|------|
| 接口参数不变 | ✅ | 工厂签名 `(N, C, S, dtype)` 与源一致 |
| `@T.prim_func` 参数名/顺序不变 | ✅ | `main(grad_out, x, weight, mean, rstd, grad_weight, grad_bias, grad_x)`；源 `out_idx=[-1]` → `[5,6,7]` |
| threads 移除 | ✅ | 移除 `threads/num_per_thread` |
| host 侧不改输入 | ✅ | 全部计算在 kernel 内；op 层仅 reshape view |
| 从源码推断规格（不问用户） | ✅ | §0.1/§0.2 从源码 + manifest 解读；输出 shape 已回填标注依据 |
| 迁移纪律（不照抄 GPU warp shuffle/threads/谓词） | ✅ | §0.5/§0.6 四态判定 + NPU 重设计 |

---

## 11. 修订日志

**revision_index: 0（first_design）**

首次设计（migration-harness，`_batch_norm_bwd_wide_kernel`，2026-10-14）。无上一版；无历史否决路径。

**关键设计决策摘要**：

| # | 决策 | 依据 |
|---|------|------|
| 1 | one core per channel + UB `T.reduce` 两步（替代 warp shuffle 树） | §0.6 R2 / 探针真机 PASS |
| 2 | held 单扫描（`N·S·18B ≤ 160KB`）/ tiled 两扫描（超大 S）由 UB 预算切换 | §1.6.0 R1#1/#2 / §1.6.3 / §4.5 |
| 3 | 一维 persistent 分核 `num_kernels = min(C, 40)`（实查 20→40） | §0.6 R1 / §5.5 |
| 4 | 尾块 `T.min` + src/dst 双显式 slice | §0.6 R3 |
| 5 | fp16/bf16 → fp32 中转链（显式 `T.vcast(rint)`）；bf16 必需 | §0.6 R4 |
| 6 | 分布式 epilogue + 倒数乘（O1/O2），D-1 EQUIV_PASS | §1.6.1 |
| 7 | 保留中心化 `x̂` 归约（拒绝 raw-moment，抵消证据） | §1.6.1 判别段 A |
| 8 | 三输出 `out_idx=[5,6,7]`（grad_weight/grad_bias/grad_x） | §0.1 接口契约 |
| 9 | 编程模式 **Expert**（`T.Scope("Vector")` + `T.alloc_ub`） | §2 / `conv1d_direct` Expert/Vector 先例 |
