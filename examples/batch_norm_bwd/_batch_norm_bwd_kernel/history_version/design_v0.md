# _batch_norm_bwd_kernel 算子设计文档

> 迁移任务（harness 模式）：`BatchNormBwdOp` 的**通用/基础核**（general tiled path，one core per channel，通道经 UB **流式分块**处理；其余两个变体是它的特化）。
> **first_design**（首次设计；无上一版，无历史否决路径）。
> 源算子（GPU 原始源码）：`TileOPs-main/src/tileops/kernels/norm/batch_norm.py` 的 `_batch_norm_bwd_kernel`（L1487）+ host 类 `BatchNormBwdKernel`（L2146）+ 配置源 `_TiledPath`（L66）。
> Stage 0 提取件（本函数迁移的精确源）：`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/_batch_norm_bwd_kernels.py` 的 `_batch_norm_bwd_kernel`（L357）。
> NPU wrapper：`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py` 的 `BatchNormBwdKernel`（L327）+ 选核 `select_bwd_kernel`（L150）；op `examples/TileOPs/tileops/ops/norm/batch_norm_bwd.py`；manifest `examples/TileOPs/tileops/manifest/norm.yaml`（`BatchNormBwdOp`）；test `examples/TileOPs/tests/ops/test_batch_norm_bwd.py`；bench `examples/TileOPs/benchmarks/ops/bench_batch_norm_bwd.py`。
> 可行性探针（Stage 0，真机 PASS，见 `examples/batch_norm_bwd/probe/PROBE.md` + `examples/batch_norm_bwd/probe/batch_norm_bwd_probe.py`）：验证了 strided 逐通道 plane gather、双累加器两步 `T.reduce`、`T.vbrc` 标量广播、fp16/bf16 fp32 中转链、one-core-per-channel 与超大 S 的核内 S-tiling（无跨核归并、无 atomics）。
> 同族参考（已完成）：宽核 `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md`（held 形态 + Stage 3 实测 UB footprint）与拆分核 `examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/DESIGN.md`（二维 `(bN,bS)` tiling + 跨段 workspace）；两者 `RETROSPECTIVE.md` 的可迁移教训。
> 工具链版本戳：tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b（2026-10-10 build）+ CANN 8.5.2 + Ascend910B3（`NPUUtils.get().get_aicore_num()` 实查 = 20，纯 Vector ×2 = 40）+ torch 2.7.1+cpu + torch_npu 2.7.1.post4。

## 0. 源算子解读与迁移分析

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 `tilelang-op-design` skill 的 `.agents/skills/tilelang-op-design/references/migration-analysis.md`）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。

### 0.0 首次设计说明（first_design）

本版为 `_batch_norm_bwd_kernel` 的首次迁移设计（first_design，migration-harness，2026-10-14）；无上一版、无已否决路径。§0.0 记录本函数的**定位**（它是同族三核中的**通用回退**，另两个变体是它的特化），供 Stage 2 检视与后续 revision 对照：

- 三个 GPU 变体共用同一数学语义（逐通道 BN 反向），差异只在**通道如何被持有/切分**：`_batch_norm_bwd_wide_kernel`（channel 入寄存器，one block per channel，held；本仓已交付）、`_batch_norm_bwd_split_kernel`（channel 切 `splits` 段跨 block：部分和 → finalize → apply；本仓已交付）、`_batch_norm_bwd_kernel`（**本函数**：one block per channel，channel 经 shared memory 以 `block_l` 为单位**流式分块**，`block_l ≥ L` 时退化为单扫描 persistent）。
- NPU 侧 `select_bwd_kernel`（`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py`）按 `_holder` 策略路由：`thread`/`block` → `bwd_wide`；`split` → `bwd_split`；其余（`tiled`，典型是大 `C ≥ _SPLIT_MAX_C(1024)` 或既非 wide 也非 split 服务的 shape）→ **本函数 `bwd_kernel`**。故本函数在 NPU 上的服务区 = **通用回退区**，设计须能覆盖任意 N/C/S（含大 N 与大 C），而 manifest 的 resnet-fc/stage1/2/3 默认路由 wide、large-spatial 默认路由 split。

### 0.1 源算子语义（做什么）

**数学语义**（BatchNorm 训练反向，从训练前向保存的逐通道统计量反推梯度）。输入 `grad_out, x : [N, C, S]`、`weight, mean, rstd : [C]`（fp32），记 `L = N*S`、`x_hat[c,l] = (x[c,l] - mean[c]) * rstd[c]`，逐通道 `c`：

$$
\text{grad\_bias}[c] = \sum_{l<L} \text{grad\_out}[c,l],\qquad
\text{grad\_weight}[c] = \sum_{l<L} \text{grad\_out}[c,l]\cdot \hat{x}[c,l]
$$

$$
\text{grad\_x}[c,l] = \frac{w[c]\cdot \text{rstd}[c]}{L}\Big(L\cdot\text{grad\_out}[c,l] - \text{grad\_bias}[c] - \hat{x}[c,l]\cdot\text{grad\_weight}[c]\Big)
$$

**I/O 契约**：`grad_out, x` 形状 `[N, C, S]`、dtype ∈ {float32, float16, bfloat16}（manifest `DType[float32|float16|bfloat16]`）；`weight, mean, rstd` 形状 `[C]` fp32；输出 `grad_x : [N, C, S]` 同输入 dtype、`grad_weight : [C]` / `grad_bias : [C]` fp32（manifest `BatchNormBwdOp` signature）。host 侧把任意 `(N, C, *spatial)` contiguous view 成 `(N, C, S)`（S = spatial 维乘积，无转置、非 in-place）。**输出 shape 从源码/op 回填确认**：host `forward` 末尾 `grad_x.view(x.shape)`（`examples/TileOPs/tileops/ops/norm/batch_norm_bwd.py`）。

**源算子的元素定位约定（本变体特征）**：源码 L375 明言「Element *l* of channel *c* is `[l // S, c, l % S]` of every tensor」——本变体把通道展平成**一维 `L`**，`block_l` 是**展平轴的 tile 宽度**（`go_shared[j] = grad_out[j // S, bc, j % S]`，L440），而非二维 `(N,S)` 平面。**通道轴 `C` 是中间轴**：逐通道平面 `x[:, c, :]` 在 GM 中是 strided 区域（行跨步 `C*S`，行内 stride=1），归约轴集合 `{0, 2}` 不是尾轴。

**规约语义**：逐通道 `c` 沿展平轴 `N·S` 归约两个独立标量——`grad_bias = Σ grad_out`、`grad_weight = Σ grad_out·x_hat`；累加顺序为「one block per channel + 块内 `T.reduce_sum(dim=1)` 硬件归约树」（L470-471），每通道单块闭环、无跨块归并、无 atomics。

**dtype 语义**：累加全程 fp32（`accum_dtype = "float32"`）；`grad_out/x` cast 到 fp32 参与计算（L433-434 `T.cast(..., accum_dtype)`）；`grad_x` 单次舍回输入 dtype（L490/L509 `T.cast(gx, dtype)`）；`grad_weight/grad_bias` 输出 fp32。即「fp32 opmath + 单次舍回」。

**边界语义**：`ragged = L % block_l != 0`（L391）——整除时走**无守卫**路径（源码注释 L390「A predicated load costs a read」，谓词装载一次判 = 多读一次），非整除时末 tile 用 `T.if_then_else(j < L, ..., 0)` 掩码装载/写出（L423-437/L488-492）。NaN/Inf 自然传播（无特殊处理）；`N≥1, C≥1, S≥1`。**空张量 `L=0`**：GPU op 在 `numel()==0` 时早退，但本迁移使用的 **NPU op** `examples/TileOPs/tileops/ops/norm/batch_norm_bwd.py` 的 `forward` **无** `numel()==0` 守卫——该拦截须由 NPU op/wrapper 承接（集成期补偿，见 §9.1/§9.3）。

**持久路径 vs 非持久路径（本变体的核心结构）**：源码 docstring L377-381 明言——
- **Persistent path（`block_l ≥ L`）**：pass 1 在累加 `grad_bias/grad_weight` 的同时把 `grad_out/x` 存入 shared memory，**pass 2 直接从 shared 计算 `grad_x`，消除第二次全局读**；
- **Non-persistent path（`block_l < L`）**：**两次全局读**（经典两遍 BN 反向）。

**host 侧 block_l 来源**：`_TiledPath.for_length(L)`（`TileOPs-main/src/tileops/kernels/norm/batch_norm.py` L66）——`PERSISTENT_MAX_L = 8192`、`MAX_BLOCK_L = 512`、`REDUCE_THREADS = (256,128,64,32)`；`L ≤ 8192` 时首选 `block_l = L`（单 tile = persistent），否则取 `block_l ≤ 512` 的 L 的因子（多 tile = non-persistent）；均不整除时末 tile 掩码。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现（移植源仓参考实现 `examples/TileOPs/tileops/workloads/norm.py` 的 `batch_norm_backward`），不复刻 NPU 算法。

### 0.2 源算子输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `grad_out` | 输入 | `(N, C, S)` | float16 / bfloat16 / float32 | 上游梯度；S = spatial 维乘积 |
| `x` | 输入 | `(N, C, S)` | same_as(grad_out) | 原始输入 |
| `weight` | 输入 | `(C,)` | float32 | 仿射缩放 gamma |
| `mean` | 输入 | `(C,)` | float32 | 前向保存的逐通道 batch mean |
| `rstd` | 输入 | `(C,)` | float32 | 前向保存的逐通道 reciprocal std |
| `grad_weight` | 输出 | `(C,)` | float32 | 源 host **预分配并传入**（非 `out_idx`） |
| `grad_bias` | 输出 | `(C,)` | float32 | 源 host **预分配并传入**（非 `out_idx`） |
| `grad_x` | 输出 | `(N, C, S)` | same_as(grad_out) | 源 `out_idx=[-1]`（唯一自动分配） |

> 源 `@T.prim_func _bn_bwd` 参数顺序为 `(grad_out, x, weight, mean, rstd, grad_weight, grad_bias, grad_x)`，`out_idx=[-1]`（仅 `grad_x` 自动分配；`grad_weight/grad_bias` 由 host 预分配传入，源码 L388 + host `BatchNormBwdKernel.forward` L2182-2211）。本设计**保持源参数序与 `out_idx=[-1]`**（见 §3.3 注 / §10.5）：`num_inputs = 8 - 1 = 7`，调用 `program(grad_out, x, weight, mean, rstd, grad_weight, grad_bias)` 返回单元素列表 `[grad_x]`；NPU wrapper `BatchNormBwdKernel.forward`（`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py` L383-384）**无需改动**。这与宽核变体（把三输出统一为 `out_idx=[7,5,6]`、须改 5 参调用）不同——本函数的选择是把源 host 契约原样继承，理由见 §3.3 注。

### 0.3 实现算法解读（怎么算）

**源码结构**（GPU，工厂模式）：`_batch_norm_bwd_kernel(N, C, S, dtype)` → `@tilelang.jit(out_idx=[-1], compile_flags=["-O3","-DENABLE_BF16"])` 内 `_bn_bwd_func(block_l, threads)` → `@T.prim_func _bn_bwd(...)`。host `BatchNormBwdKernel`（L2146）用 `_TiledPath.for_length(L)[0]` 选 `(block_l, threads)`，`forward` 预分配 `grad_weight/grad_bias` 后以 7 参调用。

**计算步骤分解**（源码每个计算语句均已归入；`threads` 个 CUDA 线程协作一个通道的一个 `block_l` tile）：

| 步骤 | 计算（源码行） | 输入 | 输出 | 对应语义公式部分 |
|------|----------------|------|------|-----------------|
| P0 | `mean_val=rstd_val=w_val = mean[bc], rstd[bc], weight[bc]`（L409-411） | GM 逐通道 3 标量 | 本地标量 | 逐通道参数读入 |
| P1a | pass 1 装载（persistent/非 persistent 两路）：`go_shared[j] = grad_out[j//S, bc, j%S]`，`x_shared[j] = x[...]`（L440-441 / L449-463） | GM grad_out/x | shared `go_shared/x_shared [block_l]` | 输入搬运 |
| P1b | 累加：`go_val=cast(go_shared[j]); x_hat=(cast(x_shared[j])-mean_val)*rstd_val; do_frag+=go_val; do_xhat_frag+=go_val*x_hat`（L432-446 / L464-466） | go_shared/x_shared | `do_frag/do_xhat_frag [1,block_l]` fp32 | grad_bias / grad_weight 贡献 |
| P1c | 尾块掩码：`if_then_else(j<L, ..., 0)` / `if l<L`（L423-437 / L452-460） | 同上 | 同上 | 非整除边界 |
| R1 | 块内归约：`T.reduce_sum(do_frag, sum_do, dim=1)`、`T.reduce_sum(do_xhat_frag, sum_do_xhat, dim=1)`（L470-471） | do_frag/do_xhat_frag | `sum_do/sum_do_xhat [1]` | 通道两标量 |
| W1 | `grad_bias[bc]=sum_do[0]; grad_weight[bc]=sum_do_xhat[0]`（L473-474） | sum_do | GM 逐通道 2 标量 | 写 grad_bias/grad_weight |
| E0 | 逐通道标量预折叠：`w_rstd_over_L = w_val*rstd_val/cast(L)`（L477） | w_val/rstd_val | 本地标量 | epilogue 系数 |
| E1 | pass 2：`go_val=cast(go_shared[j]); x_hat=(cast(x_shared[j])-mean_val)*rstd_val; gx=w_rstd_over_L*(L*go_val - sum_do[0] - x_hat*sum_do_xhat[0])`（persistent，L482-487；非 persistent 重读 GM，L496-520） | go_shared/x_shared（或重读 GM g/x） | `gx` fp32 | grad_x 公式 |
| E2 | 写出：`grad_x[j//S, bc, j%S] = T.cast(gx, dtype)`（L490/L492 / L509/L520） | gx | GM grad_x | 输出搬运 |

**数据流与内存访问模式**（源硬件视角）：

```
GM[weight/mean/rstd] --scalar load--> REG[mean_val/rstd_val/w_val]
GM[grad_out] --T.Parallel/if_then_else(flat l=j//S*S+j%S, strided)--> SMEM[go_shared (block_l,)]
GM[x]        --同上--> SMEM[x_shared (block_l,)]
  --(逐元素 cast + x_hat + 两个 += 到 fragment)--> FRAG[do_frag, do_xhat_frag (1, block_l)]
  --T.reduce_sum(dim=1)--> FRAG[sum_do, sum_do_xhat (1,)]
FRAG[sum_do] --写 GM--> GM[grad_bias, grad_weight]
w_rstd_over_L = w_val * rstd_val / L
路径 A (persistent, block_l>=L)：SMEM[go_shared/x_shared 仍在] --epilogue--> grad_x --写 GM--> GM[grad_x]
路径 B (non-persistent, block_l<L)：GM[grad_out/x] --第二次 flat 装载--> --epilogue--> grad_x --写 GM--> GM[grad_x]
```

**循环与并行结构**：`T.Kernel(C, threads=threads)` 一维 grid（**每 block 一个通道**，逻辑核数 = C，无核内任务循环）；block 内 `threads` 个线程协作：`T.Parallel(1, block_l)` 逐元素并行 / `T.Pipelined(tiles, num_stages=0)` 分 tile 循环（**显式禁用流水**，注释 L448「T.copy inside T.Pipelined races with the async copy」）；通道内归约 = `T.reduce_sum(dim=1)` 单级线程树。**无核间协作**（通道间独立，无跨 block 归并，源码无 atomics）。

**host 侧逻辑**：`BatchNormBwdKernel.__init__` 建工厂；`default_config` = `_TiledPath.for_length(L)[0]`（`{block_l, threads}`），`autotune_configs` = 整张表；`forward` 做 `contiguous()`、预分配 `grad_weight/grad_bias`、7 参调用。op 层把任意 `(N,C,*spatial)` view 成 `(N,C,S)`、`contiguous()`，返回 `grad_x.view(x.shape)`。**无数据内容预处理**（纯 shape 元数据视图）。`@functools.lru_cache(maxsize=32)` 工厂按 `(N,C,S,dtype)` 特化。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | shared-memory 流式分块（`go_shared/x_shared` + `block_l`） | 按块驻留、限制 register 压力、tile 宽可调 | `T.alloc_shared([block_l], dtype)` + `T.Parallel(1, block_l)` 协作 | shared memory | 硬件强相关（可等价替换为 UB） |
| 2 | persistent 路径（`block_l ≥ L`，单 tile 持有整通道） | pass 2 复用 shared，**消第二次全局读** | 单 tile 时两 buffer 仍持通道，pass 2 就地 epilogue（L480-492） | shared memory + 寄存器/线程容量 | 硬件强相关（意图可移植） |
| 3 | two-accumulator 融合归约（`do_frag` / `do_xhat_frag`） | 单遍同时得 grad_bias 与 grad_weight | 同一遍两个 `+=`（L436-437/L445-446） | 无（纯算法层融合） | 可移植 |
| 4 | 块内归约树（`T.reduce_sum(dim=1)`） | 低延迟线程级归约 | 单级 `AllReduce` 模板（线程幂次） | warp/CTA 线程 + 模板 | 硬件强相关 |
| 5 | 分 tile 循环（`T.Pipelined(tiles, num_stages=0)`） | 控制 UB/寄存器压力（大 `L`） | `num_stages=0` 显式不流水（规避 T.copy race） | 无（调度结构） | 模式可移植 |
| 6 | 整除无守卫快路径 / 尾 tile 谓词（`ragged` + `if_then_else`） | 整除去谓词开销 | 编译期 `ragged` 分支 + 逐元素守卫 | 无（模式可移植） | 模式可移植 |
| 7 | epilogue 逐通道标量预折叠（`w_rstd_over_L`） | 逐元素只做 2 mul + 2 sub | 逐通道标量在循环外算一次（L477） | 无（纯算法层） | 可移植 |
| 8 | fp32 opmath + 单次舍回 | 精度（fp32 累加，输出 dtype） | `T.cast(...,"float32")` 累加、赋值截断回 dtype | 寄存器 | 硬件强相关 |
| 9 | 一通道一 block 映射 | 通道间并行、无跨核归并 | `T.Kernel(C)` | CTA 调度 | 硬件强相关 |
| 10 | lru_cache 工厂 + `_TiledPath` per-shape 特化/autotune | 摊编译开销、按 L 选 tile | `functools.lru_cache(maxsize=32)` + `for_length(L)` | 无（host 层） | 可移植 |
| 11 | `out_idx=[-1]` 单输出自动分配（host 预分配通道梯度） | 接口契约 | jit `out_idx` + host 预分配 | 无 | 可移植（接口契约） |

> 识别不出来 ≠ 不存在。未列出的优化会在迁移中被静默丢弃，导致性能回退无法追溯。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定依据：`.agents/skills/tilelang-op-design/references/migration-analysis.md` §5 三层模型 + §5.3 映射表；`examples/` 佐证；`docs/` 条目；pattern-library 实测条目；探针 `examples/batch_norm_bwd/probe/PROBE.md`；同族宽核 `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md`（含其 Stage 3 实测 UB footprint）。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（逐通道两标量归约 + grad_x 公式 + fp32 opmath + 单次舍回） | 语义 | 无 | — | **保留** | 语义层无条件保留（§0.1） |
| two-accumulator 融合归约（#3） | 算法 | 无 | — | **保留** | 同遍 `T.reduce` 双路（探针真机验证） |
| epilogue 标量预折叠（#7） | 算法 | 无 | — | **保留** | 标量在核内循环外算一次（§1.6.1 O2 承接） |
| persistent 单扫描（#2） | 优化 | shared + 线程容量 | 有（UB 192KB） | **等价替换** | `T.alloc_ub` 整通道平面驻留（held 档，`docs/Tilelang.language/内存操作/T.alloc_ub.md`）；预算见 §4.5 |
| shared-memory 流式分块（#1） | 优化 | shared memory | 有（UB 192KB） | **等价替换** | `T.alloc_ub` + 核内 `T.serial` S/N 分块（tiled 档）；bS/bN 由 UB 预算反推 |
| 块内归约树 `T.reduce_sum(dim=1)`（#4） | 优化 | warp/CTA 线程 | 无直接等价 | **重新设计** | → §0.6 R2：UB 二维平面 + 两步 `T.reduce`（`dims=[1]`→`(bN,1)` 再 `dims=[0]`→`(1,1)`） |
| 分 tile 循环 `T.Pipelined(num_stages=0)`（#5） | 优化 | CUDA 线程/流水 | 有更优形态 | **重新设计** | → §0.6 R3：核内 `T.serial(n_tile_N) × T.serial(n_tile_S)`（静态边界）；显式不流水（规避异步 copy race，与源同意图） |
| 逐元素谓词 `if_then_else(j<L)`（#6） | 优化 | 无 | 有更优形态 | **重新设计** | → §0.6 R5：`T.min` + src/dst 双显式 slice（I/O）+ reduce `size` 截断（归约） |
| fp32 opmath + 单次舍回（#8） | 优化 | 寄存器 cast | 有（UB fp32 + vcast） | **等价替换** | `T.vcast(rint)` 升 fp32 → fp32 v-prefix 链 → `T.vcast(rint)` 单次舍回（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`；bf16 无 Vector 算术 → 中转必需） |
| 一通道一 block 映射（#9） | 优化 | CTA 调度 | 无（一维 Kernel + 物理核数限制） | **重新设计** | → §0.6 R1：一维 `T.Kernel` persistent 分核（`num_kernels = min(C, 40)`） |
| flat 元素定位 `l//S, l%S`（#1 的地址形态） | 优化 | 逐线程地址计算 | 无（非仿射 → 破坏向量化） | **重新设计** | → §0.6 R4：二维 `(bN,bS)` plane（S 为连续内轴）；见拆分核教训（flat 段 `l//S`/`l%S` 非仿射） |
| `threads` 参数 / block×thread 两级并行 | 优化 | CUDA 线程 | 无 | **舍弃** | npuir `T.Kernel(threads=)` 无效果；并行交给 Vector lane（§1.6.2/§1.6.3） |
| `T.Parallel(1, block_l)` / `T.Pipelined` | 优化 | CUDA 线程 | 无 | **舍弃** | 后端不适用；改 UB v-op/`T.reduce` + 核内 `T.serial`（§0.6 R2/R3） |
| lru_cache 工厂 + JIT 特化（#10） | host | 无 | — | **保留** | host 层硬件无关；NPU wrapper 用 Kernel 类实例缓存承接 |
| 三 kernel 选择策略（wide/split/general） | host | CUDA `sm_count` | 有（物理核数） | **等价替换** | wrapper 已移植 `_AI_CORE_COUNT=40` + `select_bwd_kernel`（`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py`） |

### 0.6 NPU 算法重设计

**重设计项 R1: 并行结构（一通道一 block 超发 → 一维 persistent 分核）**

- **源方案**：`T.Kernel(C, threads=threads)`——逻辑 block 数 = C，依赖 CUDA 运行时把 C 个 block 调度到 SM。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`，`num_kernels = min(C, 40)`（40 = 20 AICore × 2，纯 Vector 翻倍，§5.5 实查）；核内 `for i in T.serial(num_local_tasks)` 以 grid-stride 任务映射 `c = i*num_kernels + cid`，`if c < C` 守卫；`num_local_tasks` 为 host 层 Python 编译期常量（静态边界）。意图承接：消除超发逻辑内核的串行调度开销（`.agents/skills/_shared/standards/core-split-strategy.md` §1 极大规模分支）。参照同族 `examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/DESIGN.md` 与宽核 `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md` 同款 persistent 分核。
- **语义保持论证**：`C` 个逻辑任务只是源 grid 的重新映射；通道间独立，任务到通道的映射不改变任何任务的计算语义。

**重设计项 R2: 归约（块内线程树 → UB 二维平面两步 `T.reduce`）**

- **源方案**：`do_frag/do_xhat_frag (1, block_l)` 逐元素累加，再 `T.reduce_sum(dim=1)` 线程树归约到 `(1,)`。
- **NPU 新算法**：把每通道平面物化为 UB `(bN, bS)` fp32，逐 tile 两路 `T.reduce(g32, red, dims=[1], reduce_mode="sum", clear=False, size=[rN, rS])` 得行和 `(bN,1)`（跨 S-tile 累加），再 `T.reduce(red, gb, dims=[0], reduce_mode="sum", clear=False, size=[rN,1])` 得 `(1,1)` 通道标量（跨 N-block 累加）。`N=1` 时工厂层跳过退化的 `dims=[0]` 层，`S=1` 时跳过退化的 `dims=[1]` 层（`T.reduce` Shape 规则要求恰好一维不同且 dst 该维为 1）。
- **语义保持论证**：归约是加法结合律下的同一数学和；NPU 用固定树序（`hivm::VReduceOp`，`docs/Tilelang.language/规约操作/T.reduce.md` §3），每核内定序、N-block/S-tile 按固定 `serial` 序分组累加 → 通道梯度确定（探针 PROBE §6）。累加分组与 GPU 不同 → fp16/bf16 舍入差异在 op 容差内（§8.2，探针最坏 max_abs_err 4.883e-04）。

**重设计项 R3: 流式分块调度（flat tile + `T.Pipelined(num_stages=0)` → 核内 `T.serial` 二维分块）**

- **源方案**：`T.Pipelined(tiles, num_stages=0)` 对展平 `L` 的 `block_l` tile 循环（显式不流水，规避 T.copy race）。
- **NPU 新算法**：核内 `for nn in T.serial(n_tile_N)`（N-子块）× `for tt in T.serial(n_tile_S)`（S-子块）双静态循环；`bN/bS` 由 UB 预算反推；held 档 `n_tile_N=n_tile_S=1`（单 tile 覆盖整通道，对应源 persistent `block_l ≥ L`）。**不引入显式 `T.Pipelined`**（同源意图：源自身用 `num_stages=0` 禁用流水）；装载/计算 overlap 交编译器 auto multi-buffer。
- **语义保持论证**：分块只是把展平 `L` 的求和项按 `(nn, tt)` 重新分组，结合律下与一次性求和同值；`n_tile_N/n_tile_S` 为 host 编译期常量。

**重设计项 R4: 地址/布局（flat `l//S, l%S` → 二维 `(bN,bS)` plane）**

- **源方案**：`grad_out[j // S, bc, j % S]` 逐元素 flat 定位（`l` 跨 N 行时不连续）。
- **NPU 新算法**：逐通道 plane 物化为 UB `(bN, bS)`（S 为连续内轴，行内 stride=1），`T.copy` 双显式 slice 沿 S 轴取子块 `grad_out[n0:n0+rN, c, s0:s0+rS]`。**承接**源的「一通道展平分块」意图，但把 flat 逐元素地址换成仿射二维 slice（flat `l//S`/`l%S` 在 NPU 上非仿射、逐元素且破坏向量化——拆分核 `RETROSPECTIVE.md` 已验证 flat 段被拒、S 轴仿射子段可用）。
- **语义保持论证**：二维 plane 与 flat 展平是同一数据的不同索引（`l = n*S + s`，一一对应），元素集合不变。

**重设计项 R5: 边界处理（逐元素谓词 → `T.min` + 双显式 slice + reduce `size` 截断）**

- **源方案**：`ragged` 编译期分支 + 非整除时逐元素 `if_then_else(j<L, ...)` / `if l<L` 掩码装载/写出。
- **NPU 新算法**：统一单路径（去 `ragged` 分支）：子块 extent `rS = T.min(bS, S - s0)`、`rN = T.min(bN, N - n0)`（`docs/Tilelang.language/比较操作/T.min.md` 标量 min），装载/写出用 src/dst 双显式 slice；**归约同样须截断**：`T.reduce(dims=[1])` 传 `size=[rN, rS]`、`T.reduce(dims=[0])` 传 `size=[rN, 1]`，只对有效范围求和（否则尾 tile 垃圾列/行计入 `gb/gw`——宽核 v0 缺陷根因）。
- **语义保持论证**：越界元素不参与装载/写出；reduce 以 `size` 限定有效 extent（非整除时尾 tile/尾 N-block 用 `T.min` 截尾）；与源谓词掩码等价。参照同族 `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md` §0.6 R3 并叠加 N 维截尾（拆分核 §0.6 R5）。

**重设计项 R6: dtype 路径（隐式 cast → 显式 `T.vcast(rint)` fp32 中转链）**

- **源方案**：`T.cast(..., "float32")` 上 cast + 赋值隐式截断回 dtype。
- **NPU 新算法**：fp16/bf16 输入：`T.vcast(stage, x32, round_mode="rint")` 升 fp32（f16→f32 / bf16→f32 rint 无损）→ fp32 v-prefix 链 → `T.vcast(..., round_mode="rint")` 单次舍回；fp32 输入无 cast。**bf16 无 Vector 算术**（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.1 bf16 列为 ×）→ fp32 中转必需。
- **语义保持论证**：fp16/bf16→fp32 精确无损；fp32→目标 dtype 单次 RNE 舍回 = 源赋值截断语义；跨 dtype `T.copy` 的隐藏 VCast 不用作 cast 融合（`docs/Tilelang.language/内存操作/T.copy.md` §3）。

**重设计项 R7: 超大 `L` 的处理（源多 tile 两遍 → 核内二维分块两扫描，无跨核归并）**

- **源方案**：`block_l < L` 的 non-persistent 路径，两次全局读（两遍 BN 反向）。
- **NPU 新算法**：一个通道始终由**一个核**持有——当 `N·S` 超 UB 预算时，核内 `(bN,bS)` 分块：pass 1 逐 tile 累加 `red/red2`，末尾一次 `T.reduce` 得通道标量；pass 2 重读 tile 做 `grad_x` epilogue（探针 `_scratch_tiled.py` 验证超大 S 分块 PASS）。**无第二个 block、无跨核归并、无 atomics**。
- **语义保持论证**：分块累加 = 展平 `N·S` 求和的结合律重排；两扫描承接源 non-persistent 的「统计遍 + 应用遍」意图；held 档（`n_tile=1`）单扫描承接源 persistent 意图（消第二次读）。

**舍弃项的意图承接汇总**（不得静默丢弃，`migration-analysis.md` §6.4）：
- shared 流式分块（#1）意图 = 按块驻留限制压力 → UB 二维分块承接；
- persistent 单扫描（#2）意图 = 消第二次全局读 → held 档复算 UB 承接；
- 线程树归约（#4）意图 = 低延迟块内归约 → `T.reduce` 硬件树归约承接；
- `threads`（#9 part）意图 = 块内并行度 → Vector lane 向量化承接（§1.6.2/§1.6.3）；
- 谓词掩码（#6）意图 = 越界安全 → `T.min` + 双显式 slice + reduce `size` 承接。

### 0.7 标杆实现

- 源算子：`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/_batch_norm_bwd_kernels.py` 的 `_batch_norm_bwd_kernel`（Stage 0 提取件，迁移精确源，L357）；GPU 原始 `TileOPs-main/src/tileops/kernels/norm/batch_norm.py`（`_batch_norm_bwd_kernel` L1487 + host `BatchNormBwdKernel` L2146 + `_TiledPath` L66）。
- 参考实现 / 测试基准：`examples/TileOPs/tileops/workloads/norm.py` 的 `batch_norm_backward`（独立纯 PyTorch，fp32 中间量 + 舍回输入 dtype）——golden 以 §0.1 语义为依据移植此实现，不复刻 NPU 算法。
- manifest：`examples/TileOPs/tileops/manifest/norm.yaml` 的 `BatchNormBwdOp` 条目（5 输入 3 输出；workloads resnet-fc/stage1/2/3/large-spatial；roofline `flops = 9*B*C*prod(L)`，`eval_roofline` nbytes `= (3*N*C*S + 5*C)*elem_bytes`）。
- 性能基准：`examples/TileOPs/benchmarks/ops/bench_batch_norm_bwd.py`（manifest 驱动，5 输入多输入形态）。
- 探针：`examples/batch_norm_bwd/probe/PROBE.md` + `examples/batch_norm_bwd/probe/batch_norm_bwd_probe.py`（真机 PASS，含超大 S 核内 tiling）。
- 同族已交付设计：`examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md`（held + Stage 3 实测 UB footprint）、`examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/DESIGN.md`（二维 `(bN,bS)` tiling）。

---
## 1. 概述

### 1.1 算子名称

`_batch_norm_bwd_kernel`（BatchNormBwdOp 的通用/基础核变体；NPU 侧 one core per channel + UB 二维流式分块）

### 1.2 功能描述

对 `(N, C, S)` 输入，逐通道沿展平轴 `N·S` 归约两个标量 `grad_bias = Σ grad_out` 与 `grad_weight = Σ grad_out·(x−mean)·rstd`，并据此逐元素写出 `grad_x`。本变体是**通用回退**：一个通道由一个核持有，经 UB 以 `(bN, bS)` 二维子块**流式**处理——当整通道可驻留 UB 时退化为单扫描（对应源 persistent 路径），否则两扫描（对应源 non-persistent 路径）（NPU 重设计后算法与源差异见 §0.6/§1.6.0：一维 persistent 分核、二维 plane + 两步 `T.reduce` 替代 flat tile + 线程树、核内 `T.serial` 替代 `T.Pipelined`、`T.min`+双显式 slice+reduce `size` 替代逐元素谓词、fp32 中转链、`T.vbrc` 标量广播）。

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

与 §1.6.0 调研选定算法一致（中心化 `x̂` + 双标量归约 + one core per channel 的 held/tiled 双档；数学内核同源，结构按 NPU 重设计）。每核处理一个通道任务：held 档单扫描、tiled 档两扫描。

1. **逐通道参数装载**：`weight[c]`、`mean[c]`、`rstd[c]` GM→UB `(1,1)` fp32（`T.copy` 标量 slice）。
2. **逐通道平面 gather（二维）**：`grad_out[n0:n0+rN, c, s0:s0+rS]`、`x[...]` 经 `T.copy` 双显式 slice 搬入 UB `(rN, rS)`（strided 区域，行跨步 `C·S`，行内 stride=1）。
3. **fp32 中转**：fp16/bf16 输入经 `T.vcast(rint)` 升 fp32；fp32 输入同 dtype 直拷。
4. **`x̂` 计算**：`x̂ = (x − mean)·rstd`——`mean/rstd` 经 `T.vbrc` 广播后 `T.vsub`/`T.vmul`。
5. **双累加归约**：pass 1 逐 S-tile 两路 `T.reduce(g32, red, dims=[1], clear=False, size=[rN, rS])` 行和累加进 `red/red2 (bN,1)`，每 N-block 末 `T.reduce(red, gb, dims=[0], clear=False, size=[rN,1])` 累加进 `(1,1)`；写 `grad_bias[c]`、`grad_weight[c]`。
6. **epilogue 系数预折叠**：`scale = w·rstd`、`per = scale·fl(1/L)`、`shift = per·gb`、`xhat_coef = per·gw`（逐通道标量，循环外一次；与 §1.6.1/§3.1/§3.3 同号）。
7. **`grad_x` 逐元素写出**：`grad_x = scale·grad_out − shift_brc − xhat_coef·x̂`（`shift/xhat_coef` 经 `T.vbrc` 广播），`T.vcast(rint)` 舍回 dtype，`T.copy` 双显式 slice 写 GM。held 档直接复用 pass 1 的 `g32/xh32`（单扫描，省 pass 2 重读）；tiled 档 pass 2 重读。

与源算法差异及来源：并行组织从「一通道一 block + threads」改为「一维 persistent 分核 + 核内 serial」（§0.6 R1）；归约从「flat tile + `T.reduce_sum(dim=1)` 线程树」改为「二维 plane + 两步 `T.reduce`」（§0.6 R2）；调度从「`T.Pipelined(tiles, num_stages=0)`」改为「核内 `T.serial(n_tile_N) × T.serial(n_tile_S)`」（§0.6 R3）；地址从「flat `l//S, l%S`」改为「仿射二维 slice」（§0.6 R4）；边界从「`ragged` + `if_then_else` 谓词」改为「`T.min` + 双显式 slice + reduce `size`」（§0.6 R5）；dtype 从隐式 cast 改为显式 `T.vcast(rint)` 中转链（§0.6 R6）；超大 `L` 从「两遍全局读」改为「单核二维分块两扫描、无跨核归并」（§0.6 R7）；epilogue 预折叠（§1.6.1 O2）。**保留**源算法意图：two-accumulator 单遍融合、persistent 单扫描、epilogue 标量预折叠、fp32 opmath 单次舍回、整除快路径。

### 1.5 数据流图

```
GM[weight/mean/rstd] --T.copy(标量 slice)--> UB[w1/m1/r1 (1,1) fp32]
for nn in serial(n_tile_N):                                   # N-子块
 for tt in serial(n_tile_S):                                  # S-子块（held 档均退化为 1）
  GM[grad_out] --T.copy(双slice,rN×rS)--> UB[stage(dtype)] --vcast(rint)--> UB[g32 (bN,bS) fp32]
  GM[x]        --T.copy(双slice,rN×rS)--> UB[stage(dtype)] --vcast(rint)--> UB[xh32 (bN,bS) fp32]
  UB[xh32] --vbrc(m1)+vsub+vbrc(r1)+vmul--> UB[xh32 = x̂]
  UB[g32] --T.reduce(dims=[1],clear=False,size=[rN,rS])--> UB[red (bN,1)]
  UB[g32]·UB[xh32] --vmul--> UB[prod] --T.reduce(dims=[1],clear=False,size=[rN,rS])--> UB[red2 (bN,1)]
 UB[red]/UB[red2] --T.reduce(dims=[0],clear=False,size=[rN,1])--> UB[gb/gw (1,1)]
UB[gb/gw] --T.copy--> GM[grad_bias]/GM[grad_weight]
UB[gb],UB[gw] + UB[w1],UB[r1] --标量预折叠--> UB[scale/per/shift/xhat_coef (1,1)]
for nn in serial(n_tile_N):
 for tt in serial(n_tile_S):
  GM[grad_out]/GM[x] --T.copy(双slice)--> UB[stage] --vcast(rint)--> UB[g32/xh32]
  UB[g32],UB[xh32] + UB[scale/shift/xhc] --vbrc 广播 + vmul/vsub--> UB[gx32]
  UB[gx32] --vcast(rint)--> UB[stage(dtype)] --T.copy(双slice)--> GM[grad_x]
（全部计算在 UB 内闭环；held 档 n_tile_N=n_tile_S=1，pass 2 复用 pass 1 的 g32/xh32 单扫描）
```

### 1.6 算法调研与优化分析 ⭐

> 调研对象：同一数学语义（逐通道 BN 反向，从 forward 保存的 mean/rstd 反推三梯度）的算法族。源算法（one block per channel 的 flat shared 流式 + threads）只是基线候选之一。调研深度：**完整调研**（规约/统计类）。信息源：`.agents/skills/tilelang-op-design/references/algorithm-candidates.md` ALG-reduction 族命中行 + 本仓 examples（同族宽核 `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md`、拆分核 `examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/DESIGN.md`、`examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/DESIGN.md`、`examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/DESIGN.md`）+ pattern-library（`constants.md` CONST-mte2-degradation / CONST-vector-launch-overhead / CONST-aicore-910B2C / CONST-capacity-910B2C）+ 源码 §0.3/§0.4 + 探针真机结论 + 结构判据。无互联网检索（本地源全覆盖四问所需候选）。

#### 1.6.0 算法调研（Algorithm Research）⭐

**R1 等价化简公式候选**（基线候选必须在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3？ |
|---|------|------------|-----------|---------|----------|
| 0 | 基线（源 `_batch_norm_bwd_kernel`） | 中心化 `x̂` + 双 `+=` 累加 + flat `block_l` tile（`l//S,l%S`）+ 线程树 `T.reduce_sum` + `T.Pipelined(num_stages=0)` | — | — | ✅（基线） |
| 1 | held 单扫描整平面驻留 | 同基线数学；整通道平面 `(N,S)` 驻留 UB，单遍读 `grad_out/x`，两次 `T.reduce`，pass 2 复用 | 数学恒等（同和同式） | 少一遍 `grad_out/x` 读（流量 3·L·C·eb，Manifest roofline 下界） | ✅（主选 A，`N·S` 落 UB 预算内） |
| 2 | tiled 两扫描（核内二维分块） | 同基线数学；`(bN,bS)` 分块驻留，pass1 累加 + final reduce，pass2 重读做 epilogue | 数学恒等（结合律重排） | UB 有界、任意 N/S（代价：+2 读 → 5·L·C·eb） | ✅（主选 B，超大 L / 大 N） |
| 3 | raw-moment `grad_weight` | `gw = rstd·(Σ grad_out·x − mean·Σ grad_out)` | 代数恒等，但大均值小方差 fp32 **灾难性抵消**（D-1 判别段 A） | 归约遍省每元素 1 sub + 1 mul（≈L op/通道） | ❌（数值否决，§1.6.1 判别段 A 机器证据） |
| 4 | 跨核 split 通道（源 `_batch_norm_bwd_split_kernel`） | 通道切 `splits` 段跨 block，部分和 `[C,splits]` + finalize 合并 | 数学恒等（容差内） | 通道内并行度提升 | ❌（需跨核同步/atomics；本函数服务区是大 C，`C ≥ 1024` 时并行度已足，见 R4；拆分核已覆盖 split 区） |
| 5 | online / Welford 变体 | 逐 tile 增量统计 | — | — | ❌（本反向的两和是**输入无关函数的可加一阶矩**，归约天然单遍；无 variance/max 可在线更新——见 R2） |

**R2 在线算法**：**无新增收益（等价于基线）。** 结构依据：BN 反向的 `mean/rstd` 是**前向输入**（非本算子计算），本算子只算两个加性一阶矩 `Σ grad_out` 与 `Σ grad_out·x̂`——任何分块/流式归约就是把这两和的求和项分组，本身就是「单遍 streaming reduce」，不存在需要重缩放（rescale）的 running 统计量（对比 online softmax 的 running max/sum）。源 `_batch_norm_bwd_kernel` 的分块累加（L449-466）即该族既有 streaming 结构。因此「在线化」在本算子上退化为 R1#1/#2 的 held/tiled 结构选择，不构成独立算法族。

**R3 复杂度对比**（每通道、`L = N·S`、`eb` = dtype 字节；FLOPs 量纲 = 向量操作数（含 cast 计数为 1 op/元素），无超越函数；GM Bytes 口径 = `grad_out` 读 + `x` 读 + `grad_x` 写 + `weight/mean/rstd` 与通道梯度小项 5C；扫描遍数 = 同一输入张量的 GM 重复读取次数）：

| 算法候选 | FLOPs（每通道） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线 flat 流式（=候选 0） | ≈9L（x̂ 2L + 双累加 3L + epilogue 4L；与 manifest roofline `flops=9·B·C·L` 一致） | 3·L·C·eb（persistent）或 5·L·C·eb（非 persistent） | grad_out 1–2、x 1–2 | 2 shared (block_l) + 2 fragment | 通道间并行；无跨核 |
| held 单扫描（主选 A） | ≈9L | **3·L·C·eb**（+5C） | grad_out 1、x 1 | 2 fp32 `(N,S)` 平面 + scratch | 通道间并行；无跨核 |
| tiled 两扫描（主选 B） | ≈9L | 5·L·C·eb（+5C） | grad_out 2、x 2 | 有界 `(bN,bS)`·BPE | 通道间并行；无跨核 |
| raw-moment（候选 3） | ≈8L（省归约遍 x̂ 的 2L） | 同 held/tiled | 同 held/tiled | 同 held/tiled | 同；**抵消风险** |
| 跨核 split（候选 4） | ≈9L + 归并开销 | 5·L·C·eb + `[C,splits]` 部分和往返 | 2 + 部分和 | +`[C,splits]` | 通道内并行；**需跨核归并** |

prefill 代表（resnet-stage1 `N=8,C=64,S=1024` fp16，L=8192）：held 3·8192·64·2 ≈ 3.1MB；大 C 服务区代表（`N=32,C=2048,S=512` fp16，L=16384）：tiled 5·16384·2048·2 ≈ 320MB；large-spatial（`N=4,C=128,S=1048576`，L=4194304）：tiled 5·4194304·128·2 ≈ 5.37GB。带宽受限算子上访存/遍数是主导项（`constants.md` CONST-mte2-degradation）。

**R4 硬件亲和性评估**（检查清单逐项；负向淘汰附已核对的 docs 条款或 examples 佐证，或「未文档化假设 + 估算依据」）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| held 单扫描（A） | Vector：`T.reduce` + v-prefix 逐元素（`docs/Tilelang.language/规约操作/T.reduce.md` / `T.math/T.vmul`），无 MAC 段 ✓ | 2 fp32 `(N,S)` + scratch ≈ 22·N·S B ≤ 180KB（实测膨胀 BPE，§4.5）✓ | S 尾轴连续 stride=1；fp16/bf16 ×8、fp32 ×4；S=784/900 有尾 lane 浪费 ≤ 1%（§1.6.3） | N/C/S/bN/bS/n_tile_* 编译期常量 ✓ | 顺序依赖链；装载/计算 overlap 交 auto multibuffer ✓ | ✅ 主选 A |
| tiled 两扫描（B） | Vector ✓ | 有界 `(bN,bS)` ✓（任意 N/S） | bS 取向量宽倍数，尾 tile/尾 N-block `T.min` 截尾 ✓ | `n_tile_N/n_tile_S` 静态 ✓ | 同上 | ✅ 主选 B（超大 L / 大 N） |
| raw-moment（3） | Vector ✓ | 同 A/B | 同 A/B | ✓ | ✓ | ❌ 数值否决（R1#3 判别段 A） |
| 跨核 split（4） | Vector ✓ | +部分和缓冲 | ✓ | ✓ | 需核间同步 | ❌（`T.atomic_add`/finalize 可用但无必要；`docs/Tilelang.language/原子操作/T.atomic_add.md`；本函数服务区 C 大、并行度足） |
| online/Welford（5） | 合并链对 Vector 失分 | 同 | 同 | 同 | 依赖链长 | ❌（R2：无对应 running 统计量） |

**调研结论**：选定**「中心化 `x̂` + 双标量 `T.reduce` 归约 + one core per channel + UB 二维流式分块（held 单扫描 / tiled 两扫描）」算法族**（候选基线数学内核 + 候选 1/2 的 NPU 驻留实现），按 UB 预算在 held（主选 A）与 tiled（主选 B）间切换；采纳 §1.6.1 的 O1/O2。关键依据：R3 表中 held 的 GM 流量 3·L·C·eb 为全候选最小（tiled +66% 流量但 UB 有界，是超大 L / 大 N 的唯一可行档）；R4 全项 Vector 亲和；raw-moment 因大均值小方差抵消被机器证据否决（§1.6.1 判别段 A）；跨核 split 的归并/atomics 在 one core per channel 下无收益、且本函数服务区 `C` 大并行度已足。与基线的结构差异一句话：数学内核不变（中心化 `x̂` + 双和），并行组织从 grid 超发改 persistent 分核、归约从 flat 线程树改 UB 二维两步 `T.reduce`、调度从 `T.Pipelined` 改 `T.serial`、地址从 flat `l//S` 改仿射二维、边界从谓词改 `T.min`+双 slice+reduce `size`、dtype 改显式 fp32 中转、大 L 改单核二维分块（消除跨核归并）。源算法优化手段意图承接见 §0.6 末尾汇总。调研范围：`algorithm-candidates.md` ALG-reduction 族命中行、`examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md`、`examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/DESIGN.md`、`examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/DESIGN.md`、`examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/DESIGN.md`、`constants.md` CONST-* 四条、源码 §0.3/§0.4、探针 PROBE.md、结构判据（R2）。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)；常数引用 `constants.md` CONST-mte2-degradation 3:1 R/W 地板 ~1.26 TB/s、CONST-vector-launch-overhead ~0.5µs/op）**：

- **resnet-stage1** `8×64×1024` fp16（held，流量 3.1MB）：流量项 ≈ 3.1MB / 1.26TB/s ≈ **2.5µs**；发射项 ≈ 20 op × 0.5µs × 通道轮次（40 核 × 2 通道轮 ≈ 每核 2 通道）≈ 20µs？——**未实测假设**：发射项按每核 op 数 × 0.5µs 粗估、未含 auto-multibuffer 隐藏，标「未实测假设 + 估算依据（CONST-vector-launch-overhead）」，交 Stage 4 校准。取 max → **≈ 20µs 量级**。
- **大 C 服务区代表** `32×2048×512` fp16（tiled，bN=16、bS=512、n_tile_N=2、n_tile_S=1，流量 320MB）：流量项 ≈ 320MB / 1.26TB/s ≈ **254µs**（发射项：40 核 × 52 任务 × 2 N-block × ~10 op × 0.5µs ≈ 20.8ms？——同属**未实测假设**，被 MTE2 窗口隐藏的判断交 Stage 4；保守取 max = 发射项与流量项同量级）。
- **large-spatial** `4×128×1048576` fp16（tiled，bN=2、bS=4184、n_tile_N=2、n_tile_S=251，流量 5.37GB）：流量项 ≈ 5.37GB / 1.26TB/s ≈ **4.26ms**。
- **resnet-fc** `32×64×1`（held，L=32，流量 24KB）：发射/延迟主导 ≈ **10–20µs**（`constants.md` CONST-vector-launch-overhead）。
- 容量项：0（纯 Vector，无跨引擎往返）。**⚠️ stale 标注**：CONST-mte2-degradation（tilelang 0.1.2+ed787bb，2026-09-07）被 kb_stale_check 标记 stale（当前 417dae99，2026-10-10）；数字仅用于候选排序与 Stage 4 对账，结构性结论（held 流量最小、tiled +66%）不依赖其精确值。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定算法（中心化 `x̂` + 双标量归约，held/tiled）。逐项四要素：

| # | 优化项 | 原式（源/数学） | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| O1 | 除法转乘倒数（L 为编译期常量） | `coef = w·rstd / L`（源码 `w_rstd_over_L = w_val*rstd_val/cast(L)`，L477） | `per = (w·rstd) · fl(1/L)`（`L_inv` 工厂期 Python 常量） | L 编译期常量；`fl(1/L)` 相对误差 ≤ 2⁻²⁴（L=2^k 时精确）；D-1 机器验证 21 案例（3 dtype × 7 shape）全过：候选式 vs 基线式 **max ulp = 0、违反率 0** | 逐通道 1 次 vdiv → 1 次 vmul（(1,1) buffer，op 型替换，零风险） |
| O2 | distributed epilogue（折 L 与 coef 入逐通道标量） | `grad_x = coef·(L·grad_out − gb − x̂·gw)`（源码 L485-487） | `grad_x = scale·grad_out − per·gb − per·gw·x̂`，`scale = w·rstd` | 分配律恒等：`coef·L·grad_out = (w·rstd)·grad_out = scale·grad_out`；`coef·gb = per·gb`；`coef·gw·x̂ = per·gw·x̂`。D-1：**max ulp = 0、违反率 0**（三 dtype 全过） | 逐元素由 3 mul（`L·g`、`x̂·gw`、末乘 `coef`）降为 2 mul（`scale·g`、`per·gw·x̂`），每元素省 1 向量 mul（≈L op/通道） |
| O3 | `x̂` 公共子表达式消除（CSE） | 归约遍与 epilogue 各算一次 `(x−mean)·rstd` | held 档 `x̂` 驻留 UB 一次、归约与 epilogue 两处复用 | 同值复用（同一 buffer 的恒等读取），无浮点语义变化 | held 档每元素省 1 vsub + 1 vmul；**tiled 档无法驻留**，pass2 重算（在 R3 流量中已计） |

**D-1 等价性机器验证结果表**（脚本 `examples/batch_norm_bwd/_batch_norm_bwd_kernel/verify_equiv.py` 已执行，2026-10-14，torch CPU、fp64 参照 + 角点值；判定准则：候选式 vs 基线式在 dtype 容差内且违反率 = 0）：

| 验证项 | 案例数 | fp16/bf16 舍回后 max ulp | fp32 max ulp | 违反率 | 结论 |
|--------|--------|---------|---------|--------|------|
| O1+O2 联合（候选式 vs 基线式） | 21（3 dtype × 7 shape：(32,1)/(8,1024)/(4,1024)/(4,784)/(4,2048)/(1,1)/(2,3)） | **0（逐位一致）** | **0** | **0**（三 dtype 全 0） | **EQUIV_PASS** |
| 判别段 A：raw-moment `gw`（拒绝证据，非采纳项） | 7 shape × 3 dtype | — | fp32 max 违反率 **0.1553**（N=4,S=2048 大均值小方差） | — | **验证失败（FAIL）→ 拒绝采纳**（机器证据支撑 R1#3 否决；fp16/bf16 宽容差掩蔽，正是中心化式跨 dtype 保留的理由） |

形态覆盖：randn 梯度、大均值小方差 `x`（≈3e3 ± 0.5，raw-moment 抵消压力源）、常量 `x`（零方差通道）、大动态范围梯度、次正规点位、±inf、NaN、极值 `weight`、退化 shape N=1/2/S=1。

**优化结论**：采纳 O1/O2/O3。优化后公式（§3.1 唯一输入，`r = fl(1/L)`）：

$$
\text{scale} = w\cdot\text{rstd},\quad \text{per} = \text{scale}\cdot r,\quad
\text{shift} = \text{per}\cdot\text{gb},\quad \text{xc} = \text{per}\cdot\text{gw}
$$
$$
\text{grad\_x} = \text{scale}\cdot\text{grad\_out} - \text{shift} - \text{xc}\cdot\hat{x}
$$

**否决项及依据**（负向断言举证）：
- raw-moment `gw = rstd·(Σ grad_out·x − mean·Σ grad_out)`：D-1 判别段 A 机器证据（fp32 违反率 0.1553），非纸面论证；
- fp16/bf16 原生域计算链（不经 fp32 中转）：`docs/Tilelang.language/数学操作/T.vmul.md` / `T.vsub.md` / `T.vadd.md` §2.2.1 bf16 列为 ×（无 Vector 算术）——fp32 中转为**必需**；
- 跨 dtype `T.copy` 作 cast 融合：`docs/Tilelang.language/内存操作/T.copy.md` §3 跨 dtype copy 不减少 vector pass 且舍入不可控——统一走显式 `T.vcast(rint)`。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态（GPU 源） | 向量替代方案（API 佐证） | 是否替代 | 不可替代理由 |
|---|--------|---------------------|--------------------------|---------|--------------|
| 1 | 逐通道平面 gather | flat `T.Parallel` 循环 `grad_out[j//S, bc, j%S]` | `T.copy` 双显式 slice（仿射二维 strided 区域，`docs/Tilelang.language/内存操作/T.copy.md`） | ✅ | — |
| 2 | 上 cast fp32 | `T.cast(..., "float32")` 逐元素 | `T.vcast(stage, x32, round_mode="rint")`（f16→f32 / bf16→f32 rint ✓，`docs/Tilelang.language/数据类型转换操作/T.vcast.md`） | ✅ | — |
| 3 | `x̂ = (x − mean)·rstd` | 逐元素 `(cast(x_shared[j])-mean_val)*rstd_val` | `T.vbrc(m1, brc)` + `T.vsub(xh32, brc, xh32)` + `T.vbrc(r1, brc)` + `T.vmul(xh32, brc, xh32)`（`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2） | ✅ | — |
| 4 | 双 `+=` 累加 | 逐元素 `do_frag[j]+=g; do_xhat_frag[j]+=g*x_hat` | `T.vmul(g32, xh32, prod)` + 两路 `T.reduce(...)`（`docs/Tilelang.language/数学操作/T.vmul.md` / `docs/Tilelang.language/规约操作/T.reduce.md`） | ✅ | — |
| 5 | 通道和 `gb` | `T.reduce_sum(do_frag, dim=1)` 线程树 | 两步 `T.reduce(g32, red, dims=[1], reduce_mode="sum", clear=False, size=[rN,rS])` + `T.reduce(red, gb, dims=[0], ...)` | ✅ | — |
| 6 | 通道和 `gw` | 同 `T.reduce_sum(do_xhat_frag, dim=1)` | 同上（对 `prod`） | ✅ | — |
| 7 | epilogue 系数 | 逐线程标量 `w_rstd_over_L` | (1,1) 上 `T.vmul`/`T.vsub` 标量 op（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 标量 ✓） | ✅ | — |
| 8 | `grad_x` epilogue | 逐元素复合标量式 | `T.vbrc(shift,xc)` 广播 + `T.vmul`/`T.vsub`（O2 分布式，`docs/Tilelang.language/shape操作/T.vbrc.md`） | ✅ | — |
| 9 | 下 cast + 写出 | `T.cast(gx, dtype)` 逐元素 + `T.Parallel` 写 | `T.vcast(gx32 → stage, rint)`（f32→f16 / f32→bf16 rint ✓）+ `T.copy` 双 slice | ✅ | — |
| 10 | 通道任务循环（persistent） | 一通道一 block（无核内循环） | `for i in T.serial(num_local_tasks)` + `if c < C` 守卫 | ❌ 保留 | **block 级任务映射计算**：每 task 数个标量索引（c），非逐元素热点；serial 循环是 persistent 分核（§0.6 R1）的载体，无逐元素等价 API |
| 11 | N/S 子块循环（tiled 档） | `T.Pipelined(tiles, num_stages=0)`（flat `block_l`） | `for nn in T.serial(n_tile_N)` × `for tt in T.serial(n_tile_S)`（静态边界） | ❌ 保留 | **tile 级调度循环**：`n_tile_*` 为 host 编译期常量；承载 UB 有界与两扫描（§0.6 R3/R7），非逐元素计算 |
| 12 | 尾块有效长度 | 逐元素 `if j < L` / `if l < L` 谓词 | `rS = T.min(bS, S-s0)`、`rN = T.min(bN, N-n0)`（标量，1 次/tile，`docs/Tilelang.language/比较操作/T.min.md`） | ❌ 保留 | **动态边界元数据**：`S % bS ≠ 0` / `N % bN ≠ 0` 的尾块 extent；per-tile 1 次标量 min + slice 界 |
| 13 | 逐通道标量装载 | 线程读 `weight[bc]/mean[bc]/rstd[bc]` | `T.copy(weight[c:c+1], w1)` 标量 slice | ✅ | — |

**向量化结论**：逐元素计算（gather / cast / x̂ / 累加 / epilogue / 舍回）与通道归约**已全部向量化/归约化**（替代 API 均有 docs+探针佐证，上表）。保留 3 类标量/循环——通道任务映射（#10）、N/S 子块调度（#11）、尾块 `T.min`（#12），逐项理由如上，无「实现简单」类理由。§6 循环结构与本表一致（kernel 内无逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

规约类算子必选候选：**水平归约（lane→归约轴）vs 垂直扫描（lane→独立行）**；另评估核内重排（转置）、flat 展平与跨核 split。归约轴是展平 `N·S`（非尾轴），核内布局把逐通道平面物化为 `(bN, bS)` 二维区域。

**轴质量评分**（主选向量化轴 = S；昇腾 Vector 128bit：fp16/bf16 ×8、fp32 ×4）：

| 评分项 | 判定 |
|--------|------|
| 整除性 | 逐通道平面行内 stride=1（尾轴 S）；fp16/bf16 向量宽 ×8：S=1024（128×8 ✓）；fp32 ×4：S=784（196×4 ✓） |
| 尾 lane 浪费率 | S=1024/784 均向量宽整数倍（0%）；S=900（test 30×30）→ ceil(900/8)·8−900 = 4，浪费 0.44% |
| 累加链形态 | 归约维 = `dims=[1]`（S，连续 stride=1，硬件树归约）+ `dims=[0]`（N，小）；无跨步系数；N 维不参与向量化（外层 reduce） |
| repack 代价 | **无**（I/O 原生 `(N,C,S)` 行内连续 = 核内平面行；逐通道 gather 是 strided GM 区域，`T.copy` 直接支持，非重排） |
| UB 容量影响 | held 档 fp16/bf16 `N·S·22B`（fp32 `N·S·26B`，实测膨胀 BPE，§4.5）；tiled 档 bS/bN 由预算反推 |

**候选矩阵**（全部枚举，弃选必须给量化理由，禁止「未考虑」）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 采纳？ |
|---|---------|---------|------------|--------------|--------|
| 1 | I/O 原生 `(N,C,S)` + 逐通道 `(bN,bS)` 平面物化（held 单扫描 / tiled 两扫描） | S（尾轴连续；`reduce dims=[1]` 水平归约 + `dims=[0]`） | 无（GM strided slice 直载） | held 流量 3·L·C·eb（全候选最小）；tiled UB 有界 | ✅ **主选** |
| 2 | flat 展平平面 `(N,S)→(L,)` 一维归约（源形态） | L | 无（但逐元素 `l//S`/`l%S` 非仿射地址） | 与 #1 同流量；**flat 逐元素地址破坏向量化**、无仿射 slice 可载；拆分核 `RETROSPECTIVE.md` 已验证 flat 段被拒、S 轴仿射子段可用 | ❌（非仿射地址，向量化障碍） |
| 3 | 转置布局 `(N,C,S)→(N,S,C)` 使通道尾轴 | S/C | host permute（百 µs 级）或核内 `T.transpose` 链（`docs/Tilelang.language/创建操作/T.transpose.md`） | 归约轴不变（仍是 `N·S`）；repack 纯开销；且 op 契约输出 `(N,C,S)` 需再转回 | ❌ |
| 4 | 垂直扫描（lane→独立行，串行步进 `N·S`） | 行 | 无 | 归约维访问跨步 → 无向量宽度；跨步累加退化为标量 | ❌ |

**GPU 源码隐式轴选择的独立复核**（迁移任务要求）：GPU 把 `threads` 映射到通道内**扁平 `L`** 的连续段（`l = l_tile*block_l + j`，源码 L451/L497），即 lane 沿「flat 连续访存轴」。本设计独立评估后**结论一致（向量轴落在连续访存方向）、理由与轴选择均不同**：NPU 选 S 为向量轴（行内 stride=1）因 v-op 广播形态是行内（`[M,N]↔[M,1]`，`docs/Tilelang.language/数学操作/T.vsub.md` §2.2.2）、且二维 plane 使 gather 成为仿射 slice（flat `l//S`/`l%S` 非仿射，见候选 #2）；GPU 的 warp/thread 归约轴与 NPU `T.reduce` 归约轴**不一一对应**，故归约结构按 NPU 硬件重设计（§0.6 R2/R4）。

**实验裁决模式评估**：主选 #1（held）vs #2-tiled 的判定裕度 = 流量差（held 3 vs tiled 5，+66%）远超未实证常数的不确定区间 → 无需三件套；唯一未实证量是 **auto-multibuffer 膨胀系数**（决定 held 档 `bS=S` 是否需回退为 tiled）——这是**参数级**（bS/bN 取值）而非结构级决策：主选结构对 held/tiled 均成立。设计期采用**同族宽核 Stage 3 实测的膨胀后 BPE**（transit ≈22、fp32 ≈25.8 B/elem，见 `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/RETROSPECTIVE.md`）与 180KiB 安全预算反推，规避了「额定 BPE 高估 bS → UB 溢出」的已知陷阱。**不启动设计期探针（D-3）**：结构决策不依赖未知常数，膨胀系数已有同族同构 kernel 的实测值，探针 held 档 144KB 亦真机通过。

**布局决策结论**：选定 #1：核内布局 = I/O 原生 `(N,C,S)`、逐通道平面 `(bN,bS)` 物化到 UB、向量化轴 = S（尾轴）、repack = 无；held（fp16/bf16 `N·S·22B ≤ 180KiB`）单扫描 / tiled 两扫描（含 N 维分块）由 UB 预算切换。该结论同步落入 §3.3 伪代码与 §6 循环结构（归约/广播的内层向量维 = S；buffer 形状 = `(bN, bS)`）；弃选方案的量化理由见候选矩阵。

---
## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Expert（Vector 路径）

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（`T.reduce` + v-prefix 逐元素 + `T.vbrc` 标量广播），无 matmul | 无 Cube/L0/L1 参与；单 `T.Scope("Vector")` |
| 归约 | `T.reduce` dims=[1]/[0] | `references/decision-tree.md`：`T.reduce` 在 Developer 与 Expert 两模式都可使用——模式选择取决于是否需显式控制 UB 层级与同步，而非用了哪个 API |
| 内存层级 | 仅 GM↔UB，需显式 UB 放置 | `T.alloc_ub`（Expert 显式申请 UB，`docs/Tilelang.language/内存操作/T.alloc_ub.md`：`T.alloc_ub [Expert mode]`） |
| 同步 | 单 Vector scope 内顺序依赖链、无核间/跨引擎协作（one core per channel） | Vector scope 内默认顺序执行/自动流水；无需手动 `set_flag/wait_flag`（§7） |
| 先例 | `examples/conv1d/_conv1d_direct_kernel/DESIGN.md`（纯 Vector、`T.Scope("Vector")` + `T.alloc_ub`、persistent，Expert 已交付）、同族 `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md`（Expert/Vector） | Expert/Vector 形态已验证可行 |
| 探针 | `examples/batch_norm_bwd/probe/PROBE.md` 真机 PASS——其构造（strided `T.copy` slice / 两步 `T.reduce` / `T.vbrc` / `T.vcast`）均**模式无关**（decision-tree 明确 `T.reduce` 两模式可用），Expert 下由 `T.Scope("Vector")` + `T.alloc_ub` 承载 | 核心构造已真机验证 |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub`（显式 UB）——v-op 操作数必须在 UB（`docs/Tilelang.language/数学操作/T.vmul.md` §2.3.1） |
| Scope | 单 `with T.Scope("Vector"):`（无 Cube scope，非 MixCV） |
| 计算方式 | v-prefix API（vcast/vmul/vsub/vadd/vbrc）+ `T.reduce` + `T.min` 标量边界 |
| 同步 | Vector scope 内默认顺序/自动流水；无手动同步（§7） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)`（一维） |
| 一致性 | 实现期设 `ASCEND_MODE = "Expert"`；NPU wrapper `examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py` 的 `_BatchNormBwdBase.ascend_mode` 现为 `"Developer"`，须在集成期同步改为 `"Expert"`（记入 §9.1 风险） |

---

## 3. API 映射设计

### 3.1 公式拆解

输入公式 = §1.6.1 优化后公式（fp32 计算域；`r = fl(1/L)`；`scale = w·rstd`、`per = scale·r`、`shift = per·gb`、`xc = per·gw`）：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `g32 = fp32(grad_out)`；`x32 = fp32(x)` | 上 cast（fp32 输入直拷） |
| 2 | `xhat = (x32 − mean)·rstd` | 逐通道标量广播 |
| 3 | `gb = Σ g32`；`gw = Σ g32·xhat` | 双标量归约（跨 tile 累加） |
| 4 | `scale = w·rstd`；`per = scale·r`；`shift = per·gb`；`xc = per·gw` | epilogue 系数（O1） |
| 5 | `grad_x = scale·g32 − shift − xc·xhat` | 分布式 epilogue（O2） |
| 6 | `grad_x_dtype = dtype_round(grad_x)` | 单次舍回 |
| 7 | `grad_bias = gb`；`grad_weight = gw` | 通道梯度输出 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 任务映射 | `c = i·K + cid` | `T.serial(num_local_tasks)` + `if c < C` | 静态边界（host 折叠） | Expert |
| 尾块 | `rN = min(bN, N−n0)`；`rS = min(bS, S−s0)` | `T.min(...)` | 标量（`docs/Tilelang.language/比较操作/T.min.md`） | Expert |
| 平面 gather | GM strided 区域→UB | `T.copy(grad_out[n0:n0+rN, c, s0:s0+rS], stage[0:rN, 0:rS])` | src/dst 双显式 slice，dst 零起点 | Expert |
| 上 cast | fp16/bf16→fp32 | `T.vcast(stage[0:rN,0:rS], g32[0:rN,0:rS], round_mode="rint")` | f16→f32 / bf16→f32 rint ✓ | Expert |
| 标量广播 | `[1,1]→[rN,rS]` | `T.vbrc(m1, brc)` | src (1,1) → dst (bN,bS) ✓（`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2） | Expert |
| `x̂` 中心化 | `(x−mean)·rstd` | `T.vsub(xh32, brc, xh32)` + `T.vmul(xh32, brc, xh32)` | 同形 `(bN,bS)` 就地 | Expert |
| 归约（S） | `Σ` over S | `T.reduce(g32, red, dims=[1], reduce_mode="sum", clear=False, size=[rN, rS])` | `(bN,bS)→(bN,1)`；`size` 截断有效列（`docs/Tilelang.language/规约操作/T.reduce.md` §2.1/§2.2.2） | Expert |
| 归约（N） | `Σ` over N | `T.reduce(red, gb, dims=[0], reduce_mode="sum", clear=False, size=[rN, 1])` | `(bN,1)→(1,1)`，跨 N-block 累加 | Expert |
| 通道梯度写 | `gb/gw → GM` | `T.copy(gb, grad_bias[c:c+1])` | 标量 slice（写传入的 GM 参数，见 §3.3 注） | Expert |
| epilogue 系数 | `scale/per/shift/xc` | `T.vmul(w1, r1, coef)` 等 (1,1) 标量 op | 标量广播 ✓（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2） | Expert |
| epilogue | `scale·g − shift − xc·x̂` | `T.vbrc` + `T.vmul` + `T.vsub` | 同形 `(bN,bS)` | Expert |
| 舍回 + 写出 | fp32→dtype→GM | `T.vcast(prod, stage, round_mode="rint")` + `T.copy(stage[0:rN,0:rS], grad_x[n0:n0+rN, c, s0:s0+rS])` | f32→f16 / f32→bf16 rint ✓ | Expert |

### 3.3 计算伪代码

> 按模板 §3.3 注记：`T.Tensor` 的 dtype 以第二个位置参数传入，不用 `dtype=` 关键字。**Expert 模式**：`with T.Kernel(...)` 内套 `with T.Scope("Vector"):`，UB 缓冲用 `T.alloc_ub`（`docs/Tilelang.language/内存操作/T.alloc_ub.md`）。工厂接口与源一致 `(N, C, S, dtype)`；移除 `threads`（CUDA 执行模型产物，npuir `T.Kernel(threads=)` 无效果），移除 `block_l`（GPU tile 宽度；NPU 改由 `bN/bS` 表达）。
> **接口保持（`out_idx=[-1]`）**：源 `@T.prim_func _bn_bwd` 参数顺序 `(grad_out, x, weight, mean, rstd, grad_weight, grad_bias, grad_x)` 逐字保持，`out_idx=[-1]`（唯一自动分配 `grad_x`）。`num_inputs = 8 - 1 = 7`，调用 `program(grad_out, x, weight, mean, rstd, grad_weight, grad_bias)` 返回 `[grad_x]`。`grad_weight/grad_bias` 是**由 host 预分配、以参数传入并写入的 GM 张量**（非 out_idx）——与本仓拆分核把 `partial_sum[C,splits]` / `scale/shift/centered_coef` 作为「无 out_idx 的传入 GM workspace」写法同款（`examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/DESIGN.md` §0.2/§3.3），故 NPU wrapper `BatchNormBwdKernel.forward`（`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py` L383-384）**无需改动**。这是本函数与宽核变体（三输出统一 `out_idx=[7,5,6]`、须改 5 参调用）的接口差异，选择依据 = 源 host 契约原样继承、集成零改动。
> held/tiled 两档与 `N=1`/`S=1` 的 reduce 退化分支在**工厂层**解析（trace 体之外，规避 parser 作用域陷阱）；held 档 `n_tile_N == n_tile_S == 1`，pass-2 复用 pass-1 的 `g32/xh32`（单扫描，O3）。

```python
def _batch_norm_bwd_kernel(N, C, S, dtype):
    L = N * S
    L_inv = 1.0 / float(L)                                  # O1: 编译期常量
    use_transit = dtype in ("float16", "bfloat16")          # bf16 无 Vector 算术 -> 必需中转
    elem_bytes = 2 if use_transit else 4
    # UB BPE：同族宽核 Stage 3 实测 auto-multibuffer 膨胀后 footprint（§4.5）
    bpe = 22 if use_transit else 26
    UB_SAFE_BYTES = 184_320                                  # 180 KiB
    vec = 8 if use_transit else 4                            # 向量宽
    vector_cores = NPUUtils.get().get_aicore_num() * 2       # 实查 20 -> 40（纯 Vector 翻倍，§5.5）
    held = N * S * bpe <= UB_SAFE_BYTES
    if held:
        bN, bS = N, S
    else:
        bS = min(S, ((UB_SAFE_BYTES // (2 * bpe)) // vec) * vec)   # 至少留 2 行 N
        bS = max(vec, (bS // vec) * vec)
        bN = min(N, max(2, UB_SAFE_BYTES // (bS * bpe)))           # ≥2 行使 dims=[0] 合法
    n_tile_N = (N + bN - 1) // bN
    n_tile_S = (S + bS - 1) // bS
    n_gt1, s_gt1 = N > 1, S > 1

    @tilelang.jit(out_idx=[-1], target="npuir")             # 源契约：仅 grad_x 自动分配
    def _func(bN=bN, bS=bS):
        if bN * bS * bpe > UB_SAFE_BYTES:
            raise ValueError(f"bN={bN},bS={bS} exceeds UB budget")
        num_kernels = min(C, vector_cores)                   # 一维 persistent 分核（§5.5）
        num_local_tasks = (C + num_kernels - 1) // num_kernels   # 静态边界

        @T.prim_func
        def main(
            grad_out: T.Tensor((N, C, S), dtype),
            x: T.Tensor((N, C, S), dtype),
            weight: T.Tensor((C,), "float32"),
            mean: T.Tensor((C,), "float32"),
            rstd: T.Tensor((C,), "float32"),
            grad_weight: T.Tensor((C,), "float32"),           # 传入 GM（非 out_idx）
            grad_bias: T.Tensor((C,), "float32"),             # 传入 GM（非 out_idx）
            grad_x: T.Tensor((N, C, S), dtype),               # out_idx=-1
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
              with T.Scope("Vector"):                         # Expert: 单 Vector scope
                stage = T.alloc_ub((bN, bS), dtype)           # g/x 中转 -> gx 舍回
                g32 = T.alloc_ub((bN, bS), "float32")
                xh32 = T.alloc_ub((bN, bS), "float32")        # x_hat（held 档 pass-2 复用）
                prod = T.alloc_ub((bN, bS), "float32")        # 乘积 / 广播 scratch
                brc = T.alloc_ub((bN, bS), "float32")
                red = T.alloc_ub((bN, 1), "float32")
                red2 = T.alloc_ub((bN, 1), "float32")
                gb = T.alloc_ub((1, 1), "float32")
                gw = T.alloc_ub((1, 1), "float32")
                w1 = T.alloc_ub((1, 1), "float32")
                m1 = T.alloc_ub((1, 1), "float32")
                r1 = T.alloc_ub((1, 1), "float32")
                scale = T.alloc_ub((1, 1), "float32")
                per = T.alloc_ub((1, 1), "float32")
                shift = T.alloc_ub((1, 1), "float32")
                xc = T.alloc_ub((1, 1), "float32")

                for i in T.serial(num_local_tasks):
                    c = i * num_kernels + cid
                    if c < C:
                        T.copy(weight[c:c + 1], w1)
                        T.copy(mean[c:c + 1], m1)
                        T.copy(rstd[c:c + 1], r1)

                        # ---- pass 1: 逐 (N,S) 子块累加 gb / gw ----
                        T.clear(red)
                        T.clear(red2)
                        T.clear(gb)
                        T.clear(gw)
                        for nn in T.serial(n_tile_N):
                            n0 = nn * bN
                            rN = T.min(bN, N - n0)
                            for tt in T.serial(n_tile_S):
                                s0 = tt * bS
                                rS = T.min(bS, S - s0)
                                T.copy(grad_out[n0:n0 + rN, c, s0:s0 + rS], stage[0:rN, 0:rS])
                                if use_transit:
                                    T.vcast(stage[0:rN, 0:rS], g32[0:rN, 0:rS], round_mode="rint")
                                else:
                                    T.copy(stage[0:rN, 0:rS], g32[0:rN, 0:rS])
                                T.copy(x[n0:n0 + rN, c, s0:s0 + rS], stage[0:rN, 0:rS])
                                if use_transit:
                                    T.vcast(stage[0:rN, 0:rS], xh32[0:rN, 0:rS], round_mode="rint")
                                else:
                                    T.copy(stage[0:rN, 0:rS], xh32[0:rN, 0:rS])
                                T.vbrc(m1, brc)
                                T.vsub(xh32, brc, xh32)                 # x - mean
                                T.vbrc(r1, brc)
                                T.vmul(xh32, brc, xh32)                 # x_hat
                                T.vmul(g32, xh32, prod)                 # g * x_hat
                                # size=[rN, rS] 把行和截断到有效范围（尾块垃圾列不计入）
                                if s_gt1:
                                    T.reduce(g32, red, dims=[1], reduce_mode="sum", clear=False, size=[rN, rS])
                                    T.reduce(prod, red2, dims=[1], reduce_mode="sum", clear=False, size=[rN, rS])
                                else:                                   # S=1: 行和即平面本身
                                    T.copy(g32[0:rN, 0:1], red[0:rN, 0:1])
                                    T.copy(prod[0:rN, 0:1], red2[0:rN, 0:1])
                            if n_gt1:
                                T.reduce(red, gb, dims=[0], reduce_mode="sum", clear=False, size=[rN, 1])
                                T.reduce(red2, gw, dims=[0], reduce_mode="sum", clear=False, size=[rN, 1])
                            else:
                                T.copy(red[0:1, 0:1], gb); T.copy(red2[0:1, 0:1], gw)
                        T.copy(gb, grad_bias[c:c + 1])
                        T.copy(gw, grad_weight[c:c + 1])

                        # ---- epilogue 系数（O1/O2） ----
                        T.vmul(w1, r1, scale)                   # scale = w * rstd
                        T.vmul(scale, L_inv, per)               # per = scale / L
                        T.vmul(per, gb, shift)                  # shift = per * gb（正号）
                        T.vmul(per, gw, xc)                     # xc = per * gw

                        # ---- pass 2: grad_x = scale*g - shift - xc*x_hat ----
                        for nn in T.serial(n_tile_N):
                            n0 = nn * bN
                            rN = T.min(bN, N - n0)
                            for tt in T.serial(n_tile_S):
                                s0 = tt * bS
                                rS = T.min(bS, S - s0)
                                T.copy(grad_out[n0:n0 + rN, c, s0:s0 + rS], stage[0:rN, 0:rS])
                                if use_transit:
                                    T.vcast(stage[0:rN, 0:rS], g32[0:rN, 0:rS], round_mode="rint")
                                else:
                                    T.copy(stage[0:rN, 0:rS], g32[0:rN, 0:rS])
                                T.copy(x[n0:n0 + rN, c, s0:s0 + rS], stage[0:rN, 0:rS])
                                if use_transit:
                                    T.vcast(stage[0:rN, 0:rS], xh32[0:rN, 0:rS], round_mode="rint")
                                else:
                                    T.copy(stage[0:rN, 0:rS], xh32[0:rN, 0:rS])
                                T.vbrc(m1, brc); T.vsub(xh32, brc, xh32)
                                T.vbrc(r1, brc); T.vmul(xh32, brc, xh32)   # x_hat
                                T.vbrc(scale, brc)
                                T.vmul(g32, brc, prod)                  # scale * g
                                T.vbrc(shift, brc)
                                T.vsub(prod, brc, prod)                 # - per*gb
                                T.vbrc(xc, brc)
                                T.vmul(xh32, brc, brc)                  # xc * x_hat
                                T.vsub(prod, brc, prod)                 # - xc*x_hat
                                if use_transit:
                                    T.vcast(prod, stage, round_mode="rint")
                                else:
                                    T.copy(prod[0:rN, 0:rS], stage[0:rN, 0:rS])
                                T.copy(stage[0:rN, 0:rS], grad_x[n0:n0 + rN, c, s0:s0 + rS])

        return main

    return _func
```

> **尾块说明**：tiled 档 v-op 作用于完整 `(bN, bS)` buffer，尾块的 `[rN:]` 行 / `[rS:]` 列为未初始化/上一 tile 残留垃圾。**装载/写出**用双显式 slice 只取有效 extent；**归约**用 `size=[rN, rS]`（S 维）与 `size=[rN, 1]`（N 维）把和截断到有效范围——slice 管 I/O、`size` 管归约，二者成对，缺一即污染 `gb/gw`（同族宽核 v0 缺陷根因）。**held 档优化**：`n_tile_N == 1 and n_tile_S == 1` 时 pass-1 的 `g32/xh32` 即 epilogue 所需，实现层以工厂层 `if held:` 分支全省 pass-2 重读（单扫描；O3 的 `x̂` CSE 即此）。**接口保持**：`main(grad_out, x, weight, mean, rstd, grad_weight, grad_bias, grad_x)` 参数名/顺序与源 `_bn_bwd` 一致；`out_idx=[-1]`（返回 `[grad_x]`）。**wrapper 调用/解包约定**：`BatchNormBwdKernel.forward` 保持 7 参调用 `grad_x = program(grad_out, x, weight, mean, rstd, grad_weight, grad_bias)[0]`（或等效解包），无需改接口。

### 3.4 API 可行性确认

| API | 文档路径 / 佐证 | 验证状态 | 备注 |
|-----|----------------|----------|------|
| `T.Kernel(一维, is_npu=True)` | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` | ✅ 已核对 | 1D；`(cid, _)` 解包 |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md` | ✅ 已核对 | Expert 显式 UB；1D~2D ✓ |
| `T.copy`（双显式 slice） | `docs/Tilelang.language/内存操作/T.copy.md` | ✅ 已核对 | GM↔UB ✓；strided 区域（源 slice 非单位步长）✓；接受运行时 extent |
| `T.vcast` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` | ✅ 已核对 | f16→f32 rint ✓；bf16→f32 rint ✓；f32→f16 rint ✓；f32→bf16 rint ✓；shape 须一致；**操作数切片须静态 extent**（同族宽核 Stage 3 实测，见 §9.2） |
| `T.reduce`（dims/clear/size） | `docs/Tilelang.language/规约操作/T.reduce.md` | ✅ 已核对 | fp16/fp32 ✓ bf16 ×；`(bN,bS)→(bN,1)`、`(bN,1)→(1,1)`；`clear=False` 跨 tile/N-block 累加；**`size=[rN,rS]` 截断有效 extent**（§2.1）；`clear=False` 前须 `T.clear` |
| `T.reduce_sum` | `docs/Tilelang.language/规约操作/T.reduce_sum.md` | ✅ 已核对 | `(M,N)→(M,1)`；等价可用 |
| `T.vmul`/`T.vsub`/`T.vadd` | `docs/Tilelang.language/数学操作/T.vmul.md` / `T.vsub.md` / `T.vadd.md` | ✅ 已核对 | fp16/fp32 ✓ bf16 ×；行广播 `[M,N]↔[M,1]`、列广播 `[1,N]↔[1,1]`、标量 ✓；就地 ✓ |
| `T.vbrc` | `docs/Tilelang.language/shape操作/T.vbrc.md` | ✅ 已核对 | src (1,1)→(bN,bS) 合法；bf16 支持；**1 元素平面不可用**（同族宽核 Stage 3 实测，见 §9.2） |
| `T.min` | `docs/Tilelang.language/比较操作/T.min.md` | ✅ 已核对 | 标量 min（fp16/fp32/int ✓），尾块 extent |
| `T.clear` | `docs/Tilelang.language/内存操作/T.clear.md` | ✅ 已核对 | 累加器初始化 |
| `T.serial` + if 守卫 | `examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/DESIGN.md` §3.3 | ✅ 佐证 | persistent grid-stride 同款 |
| 探针真机 | `examples/batch_norm_bwd/probe/PROBE.md` | ✅ 真机 PASS | strided plane gather / 双 reduce / vbrc / vcast / held + 超大 S tiled |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(num_kernels, is_npu=True)` |
| GPU 专用 API | Yes | `threads=`/`T.Parallel`/`T.Pipelined`/`T.alloc_shared`/`T.alloc_fragment`/`T.reduce_sum(dim=1)` 线程树不迁移（§0.5 舍弃清单） |
| GEMM 分形/非整除 | No | 无 GEMM；S/N 非整除经 `T.min`+slice |
| L0C 容量 | No | 纯 Vector，无 L0 |
| 物理核数适配 | Yes | §5.5 三要素（实查 20→40） |
| bf16 无 Vector 算术 | Yes | §0.6 R6 fp32 中转链（必需） |
| 中间轴归约（非尾轴） | Yes | 逐通道 `(bN,bS)` 平面 strided gather + 两步 `T.reduce`（探针验证） |
| 大 N / 大 C 通用性 | Yes | 一维 persistent 分核覆盖大 C；`(bN,bS)` 二维分块覆盖大 N（§0.6 R7） |

#### 3.5.2 参考实现差异说明（GPU → NPU，汇总影响 API 选型项；完整差异见 §0.5/§0.6）

| 差异项 | GPU 源 | 本项目（Ascend） | 转换方案 |
|--------|--------|-----------------|----------|
| threads 参数 | `T.Kernel(C, threads=)` | npuir 无效 | 移除；一维 persistent `T.Kernel(num_kernels)` |
| shared/fragment 计算域 | `T.alloc_shared` + `T.alloc_fragment` | v-op 要求 UB 操作数 | 全部 `T.alloc_ub` |
| flat tile（`l//S, l%S`） | 展平 `block_l` tile | flat 逐元素地址非仿射 | 二维 `(bN,bS)` 仿射 slice（§0.6 R4） |
| 线程树归约 | `T.reduce_sum(do_frag, dim=1)` | 无 warp | `T.reduce` 两步（§0.6 R2） |
| `T.Parallel` 逐元素循环 | 线程内并行 | v-prefix 向量 op | `T.copy`/`T.vcast`/v-op（§1.6.2） |
| `T.Pipelined(num_stages=0)` | 分 tile 循环 | `T.serial` 静态循环 | §0.6 R3 |
| 逐元素谓词（`if j<L`） | masked load/store | `T.min`+双显式 slice+reduce `size` | §0.6 R5 |
| 隐式 dtype 截断 | `T.cast` 赋值 | 显式 `T.vcast(rint)` | §0.6 R6 |
| 超大 L 两遍 | non-persistent 两全局读 | 单核二维分块两扫描 | §0.6 R7 |
| `out_idx=[-1]` | 仅 grad_x 自动（host 预分配 gw/gb） | **保持 `out_idx=[-1]`** | 源契约原样继承；wrapper 7 参调用不变（§3.3 注） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/DESIGN.md` | 极高（**同算子同族**，已交付 v1；held 形态 + Stage 3 实测 UB footprint） | one core per channel、held 单扫描/tiled 两扫描、两步 `T.reduce`+`size` 截断、fp16/bf16 fp32 中转、UB 预算实测膨胀 BPE、per-chip 分核实查 |
| `examples/batch_norm_bwd/_batch_norm_bwd_split_kernel/DESIGN.md` | 极高（同算子同族，已交付 v1） | 二维 `(bN,bS)` tiling（大 N）、跨块 workspace 无 atomics、`floordiv/mod` 索引真机验证 |
| `examples/fused_add_layer_norm/_fused_add_layer_norm_kernel/DESIGN.md` | 极高（同族行归约迁移，已交付；Developer streaming 形态） | persistent 分核、尾块 `T.min`+双 slice、fp32 计算域、bf16 fp32 中转、UB 预算表、分核实查 |
| `examples/conv1d/_conv1d_direct_kernel/DESIGN.md` | 高（**Expert/Vector** 纯 Vector 迁移先例，已交付） | `ascend_mode="Expert"` + 单 `T.Scope("Vector")` + `T.alloc_ub`、persistent 分核、`T.reduce`/`T.vmul`/`T.vcast` fp32 中转、尾块 slice 截断 |
| `examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py` | 极高（本算子 NPU wrapper） | `BatchNormBwdKernel.forward` 7 参调用 + 预分配 gw/gb（接口保持）、`select_bwd_kernel` 路由、`_AI_CORE_COUNT=40`、集成期 `ascend_mode` 须由 `"Developer"` 改 `"Expert"`（§2.3） |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/DESIGN.md` | 高（同族 row-reduction 设计） | 尾块 `T.min`+双显式 slice+垃圾行丢弃、`(bm,1)` reduce 输出形态 |
| `examples/batch_norm_bwd/probe/PROBE.md` | 极高（本算子可行性探针，真机 PASS） | strided plane gather、两步 `T.reduce`、`T.vbrc` 广播、held/tiled 结构、超大 S 核内 tiling |

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
| `grad_x` | `(N, C, S)` | same_as(grad_out) | `out_idx=[-1]`（唯一自动分配） |
| `grad_weight` | `(C,)` | float32 | 由 host 预分配、传入并写入（非 out_idx，源契约） |
| `grad_bias` | `(C,)` | float32 | 同上 |

> `program(grad_out, x, weight, mean, rstd, grad_weight, grad_bias)` 返回 `[grad_x]`；host 侧再以 `grad_weight`/`grad_bias` 组装 `(grad_x, grad_weight, grad_bias)`（源 wrapper 契约；见 §0.2 / §3.3 注）。

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途（角色复用时序） |
|-----------|-------|-------|----------|---------------------|
| `stage` | `(bN, bS)` | dtype | UB | g 中转 → x 中转 → gx 舍回中转 |
| `g32` | `(bN, bS)` | float32 | UB | grad_out 平面（held 档 pass-2 复用） |
| `xh32` | `(bN, bS)` | float32 | UB | x 32 → x_hat（held 档 pass-2 复用） |
| `prod` | `(bN, bS)` | float32 | UB | `g·x̂` 乘积 / `scale·g` / 减法累加 |
| `brc` | `(bN, bS)` | float32 | UB | `T.vbrc` 标量广播 scratch |
| `red`/`red2` | `(bN, 1)` | float32 | UB | dims=[1] 行和中间/累加器 |
| `gb`/`gw`/`w1`/`m1`/`r1`/`scale`/`per`/`shift`/`xc` | `(1, 1)` | float32 | UB | 逐通道标量（各 4B，可忽略） |

### 4.4 内存搬运路径

```
纯 Vector 路径（无 Cube/L0/L1 参与）：
GM[grad_out] --T.copy(双slice)--> UB[stage] --vcast(rint)--> UB[g32]
GM[x]        --T.copy(双slice)--> UB[stage] --vcast(rint)--> UB[xh32]
UB[xh32] --vbrc(m1)+vsub+vbrc(r1)+vmul--> UB[xh32 = x_hat]
UB[g32]·UB[xh32] --vmul--> UB[prod] --T.reduce(dims=[1],clear=False,size=[rN,rS])--> UB[red]
UB[g32] --T.reduce(dims=[1],clear=False,size=[rN,rS])-----------------------------> UB[red2]
UB[red]/UB[red2] --T.reduce(dims=[0],clear=False,size=[rN,1])--> UB[gb]/UB[gw]
UB[gb]/UB[gw] --T.copy--> GM[grad_bias]/GM[grad_weight]
UB[w1]·UB[r1] --vmul--> UB[scale]; ... --> UB[per/shift/xc]  (标量预折叠)
UB[g32],UB[xh32] + UB[scale/shift/xc] --vbrc+vmul+vsub--> UB[prod=grad_x]
UB[prod] --vcast(rint)--> UB[stage] --T.copy(双slice)--> GM[grad_x]
（全部计算在 UB 内闭环，无中间 GM 往返；held 档 n_tile=1 单扫描无 pass-2 重读）
```

关键约束：v-op 操作数必须在 UB（`docs/Tilelang.language/数学操作/T.vmul.md` §2.3.1）；GM 不可直达 fragment。

### 4.5 UB 内存预算

预算目标：总驻留 ≤ UB 容量 192KB，取 `UB_SAFE_BYTES = 184_320`（180 KiB）——依据：**同族宽核 Stage 3 真机实测**（`examples/batch_norm_bwd/_batch_norm_bwd_wide_kernel/RETROSPECTIVE.md`）在额定 `BPE=18/20` 时 `bS=2272` 触发 `ub overflow, requires 1602560 bits`（= 22.04 B/elem），标定得 **auto-multibuffer 膨胀后 footprint：transit ≈22 B/elem、fp32 ≈25.8 B/elem**；改用膨胀后 BPE + 180KiB 安全预算后 `--level all` 编译运行通过。本 kernel 缓冲区集合与宽核同构（`stage + g32 + xh32 + prod + brc` 四路 fp32 + 中转），故直接沿用实测 BPE（向上取整 `BPE_transit = 22`、`BPE_fp32 = 26`）。**额定口径**（作对照）：`BPE = itemsize + 16`（fp16/bf16 = 18、fp32 = 20）。总驻留 ≈ `BPE·bN·bS` + `(bN,1)` 两项 + `(1,1)` 小项（可忽略）。

**held/tiled 判定与 bN/bS**：`held = N·S·BPE ≤ 184_320`（held 时 `bN = N`、`bS = S`）；否则 `bS = align_down(min(S, 184_320 // (2·BPE)), vw)`、`bN = min(N, max(2, 184_320 // (bS·BPE)))`（`vw` = 向量宽 8/4）。该公式即 §3.3 工厂的 `bN/bS` 推导，表值与公式**同源可复推**。

| workload (N,C,S) | dtype | N·S·BPE | 档 | bN | bS | n_tile_N | n_tile_S | 驻留 (B) |
|------------------|-------|---------|----|----|----|---------|---------|----------|
| resnet-fc (32,64,1) | fp16 | 704 | held | 32 | 1 | 1 | 1 | 704 |
| resnet-stage1 (8,64,1024) | fp16/bf16 | 180,224 | held | 8 | 1024 | 1 | 1 | 180,224 |
| resnet-stage2 (4,128,1024) | fp16 | 90,112 | held | 4 | 1024 | 1 | 1 | 90,112 |
| resnet-stage3 (4,256,784) | fp16 | 68,992 | held | 4 | 784 | 1 | 1 | 68,992 |
| resnet-stage3 (4,256,784) | fp32 | 81,536 | held | 4 | 784 | 1 | 1 | 81,536 |
| 大 C 服务区 (32,2048,512) | fp16 | 360,448 | tiled | 16 | 512 | 2 | 1 | 180,224 |
| large-spatial (4,128,1048576) | fp16 | 92,274,688 | tiled | 2 | 4184 | 2 | 251 | 184,096 |

> bN/bS 推导（fp16 `BPE=22`）：tiled 时 `184_320 // (2·22) = 4189`；`(32,2048,512)` → `bS = align_down(min(512,4189),8)=512`，`bN = min(32, max(2, 184320//(512·22)=16))=16`，驻留 `16·512·22 = 180,224 ≤ 184,320` ✓；`large-spatial` → `bS = align_down(min(1048576,4189),8)=4184`，`bN = min(4, max(2, 184320//(4184·22)=2))=2`，驻留 `2·4184·22 = 184,096 ≤ 184,320` ✓（余 224B）；`n_tile_S = ceil(1048576/4184) = 251`（尾 tile `rS = 1048576 − 250·4184 = 2576`）。两 tiled 用例中 large-spatial 命中尾 S-tile（非整除 reduce 路径）、大 C 用例命中尾 N-block（`rN = 32 − 16 = 16`）。

默认公式：`held ? (N,S) : (bN, bS)`（见上）；工厂层 guard 拒绝超预算组合（编译期 UB 溢出报错为天然兜底）。**注**：`BPE` 采用同族实测膨胀值；held 临界（resnet-stage1 180,224B ≈ 176KiB）留 Stage 4 实测，超预算即回退 tiled。

### 4.6 动态轴定义

无运行时动态轴。N、C、S、bN、bS 均为工厂期常量（per-build JIT 特化，源 lru_cache 工厂同款语义）；尾块 extent `rN/rS` 是 kernel 内 `T.min` 标量。op 层把任意 `(N, C, *spatial)` view 成 `(N, C, S)`——host 元数据视图操作。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[-1], target="npuir")   # 源契约：仅 grad_x 自动分配
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector

**判定依据**：仅 `T.reduce` + v-prefix 逐元素运算 + `T.vbrc` 广播，无 matmul。

### 5.2 Block 划分

```python
held = N * S * BPE <= UB_SAFE_BYTES            # 整通道入 UB
bN, bS = (N, S) if held else (                        # §4.5 公式
    min(N, max(2, UB_SAFE_BYTES // (align_down(min(S, UB_SAFE_BYTES // (2 * BPE)), vw) * BPE))),
    align_down(min(S, UB_SAFE_BYTES // (2 * BPE)), vw),
)
n_tile_N = ceil(N / bN); n_tile_S = ceil(S / bS)
num_logical = C                              # 逻辑任务数 = 通道数（一通道一任务）
num_kernels = min(num_logical, 40)           # 实际启动内核数（≤ 物理核，§5.5）
num_local_tasks = ceil(num_logical / num_kernels)  # 核内串行任务数（静态）
```

**选择理由**：归约轴 `N·S` 的维度上——S（尾轴）作为**向量化/归约内维**（连续 stride=1），N 作为外层 reduce 维；`(bN,bS)` 二维分块使 UB 在任何 N/S 下都有界（通用回退的必要条件）。

### 5.3 约束分析

- **对齐约束**：UB 32B 对齐由编译器管理（`docs/Tilelang.language/内存操作/T.alloc_ub.md`）；逐通道平面行内 stride=1；bS 取向量宽倍数，S 尾 lane 浪费率见 §1.6.3（≤ 0.5%）。
- **UB 容量**：§4.5 表，held ≤ 180,224B、tiled ≤ 184,096B（< 184,320B 预算）✓。
- **L0/L1/分形**：不适用（纯 Vector）。

### 5.4 注意事项

- **S % bS ≠ 0 或 N % bN ≠ 0（tiled 尾块）**：`rS = T.min(bS, S − s0)`、`rN = T.min(bN, N − n0)`；装载/写出双显式 slice 截断 **且** reduce 传 `size=[rN, rS]`（S 维）/ `size=[rN, 1]`（N 维）限定有效范围——I/O 与归约两侧都须截断，缺一即把尾块垃圾列/行计入 `gb/gw`（同族宽核 v0 缺陷根因）。
- **N=1（单 batch）**：`dims=[0]` reduce 在 size-1 上退化（`(1,1)→(1,1)`），实现层 `if n_gt1:` 工厂分支跳过（避免非法 reduce；等价地在 `gb = red` 取即得）。
- **S=1（resnet-fc）**：平面 `(bN,1)`，`dims=[1]` reduce 在 size-1 上退化 → 实现层 `if s_gt1:` 工厂分支跳过第一层 reduce，`red` 直接取平面列（`T.reduce` Shape 规则要求恰好一维不同，size-1 维不满足）。
- **held 档 pass-2 省重读**：`n_tile_N == n_tile_S == 1` 时 epilogue 复用 pass-1 的 `g32/xh32`（O3，单扫描）。
- **v-op 操作数切片须静态 extent**：`stage`/`g32` 等 v-op 切片一律用**编译期常量** `(bN, bS)`（不要传运行时 `rN/rS` 给 v-op）——同族宽核 Stage 3 实测运行时切片触发 parser `int(Var)`；尾块用 reduce `size` + dst 显式 slice 截断（`T.copy` 可接受运行时 extent）。
- **wrapper 默认 bN/bS**：NPU wrapper 由工厂默认治理；集成期/Stage 4 以 §4.5 表值或扫描值覆盖。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

- **物理核数**：**20**（AI Core，Ascend910B3）。实查记录（2026-10-14，本设计执行）：

```python
from tilelang.utils.npu_utils import NPUUtils
n = NPUUtils.get().get_aicore_num()   # -> 20
vector_cores = n * 2                  # 纯 Vector 算子翻倍 -> 40
```

  与 `constants.md` CONST-aicore-910B2C 注（本机 `get_aicore_num()` 返回 20，设备档 910B3）一致；禁止按 24 假设。
- **逻辑核数**：`num_logical = C`（一通道一逻辑任务）：

| workload | shape (N,C,S) | dtype | logical | num_kernels=min(C,40) | num_local_tasks |
|----------|---------------|-------|---------|-----------------------|-----------------|
| resnet-fc | 32×64×1 | fp16 | 64 | 40 | 2 |
| resnet-stage1 | 8×64×1024 | fp16/bf16 | 64 | 40 | 2 |
| resnet-stage2 | 4×128×1024 | fp16 | 128 | 40 | 4 |
| resnet-stage3 | 4×256×784 | fp16/fp32 | 256 | 40 | 7 |
| 大 C 服务区 | 32×2048×512 | fp16 | 2048 | 40 | 52 |
| large-spatial ※ | 4×128×1048576 | fp16 | 128 | 40 | 4 |

> **※ 选核口径**：manifest 的 resnet-fc/stage1/2/3 默认由 `_holder` 判为 `thread`/`block` → 路由宽核；large-spatial 判为 `split` → 路由拆分核。本函数（`bwd_kernel`）实际服务 `_holder` 判为 `tiled` 的 shape（大 C，如表中「大 C 服务区」行）。上表 manifest 行仅表示「若本函数被选中时的分核参数」，本设计不改选核策略。

- **规模判定与分核方案**：本函数服务区 `C` 通常 ≫ 40（典型 `C ≥ 1024`）→ **极大规模分支**——固定启动 `num_kernels = 40`，核内 `T.serial(num_local_tasks)` grid-stride（`c = i·40 + cid`），`if c < C` 守卫；**循环边界静态**（`num_local_tasks` 为 host 层编译期常量）。负载不均 ≤1 任务（2048=40·51+8）。该形态消除逻辑核超发的串行调度开销（core-split-strategy §1），并获 grid-stride 聚集窗口的 MTE2 访问效益（`constants.md` CONST-mte2-degradation 实测 +3.2%）。**通道归约完全在单核内闭环**（one core per channel），无需核间 `T.sync_block_set/wait` 或 workspace。

---
## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表仅 block/tile 级调度循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 通道任务维 | 核级并行 + 核内串行 | `T.Kernel(num_kernels)` + `T.serial(num_local_tasks)` | persistent 分核（§5.5）；通道间独立 |
| N-子块维（tiled 档） | 核内串行 | `T.serial(n_tile_N)` | UB 有界（§0.6 R7）；`n_tile_N` 静态 |
| S-子块维（tiled 档） | 核内串行 | `T.serial(n_tile_S)`（pass 1 / pass 2） | UB 有界；`n_tile_S` 静态 |
| 元素维（S） | 向量化（无循环） | `T.copy`/`T.vcast`/v-op/`T.reduce` 全 buffer 操作 | §1.6.2 #1-#9 |
| 尾块 | 标量 min + slice（I/O）+ size（归约） | `T.min` + 双显式 slice + `T.reduce(size=[rN,rS])` | §1.6.2 #12；reduce 须同步限 extent |

### 6.2 循环伪代码

见 §3.3（结构：`with T.Kernel(num_kernels, is_npu=True) as (cid, _):` → `with T.Scope("Vector"):` → buffer 分配（`T.alloc_ub`，serial 外，跨 task 复用）→ `for i in T.serial(num_local_tasks):` → `if c < C:` → pass-1（`nn/tt` 双 serial 累加归约）→ epilogue 系数 → pass-2 写 grad_x）。

### 6.3 流水线优化

**不使用显式 `T.Pipelined`**。理由：单任务内为顺序依赖链（装载→cast→x̂→累加→reduce→系数→epilogue），无 K 维迭代可重叠；且**源自身用 `T.Pipelined(num_stages=0)` 显式禁用流水**（源码 L448 注释「T.copy inside T.Pipelined races with the async copy」），NPU 侧等价承接为核内 `T.serial`。任务间/子块间装载与计算 overlap 交由编译器 auto multi-buffer。Stage 4 选项：显式双缓冲（预取下一子块）或 `T.Pipelined` 包裹 serial 循环——若 baseline 偏离 §1.6.0 估算下界 >2x 再评估。

### 6.4 尾块处理

**tiled 尾块**：`S % bS ≠ 0` 或 `N % bN ≠ 0` 时最后一个子块 `rN < bN` 或 `rS < bS`——装载/写出双显式 slice 截断 **且** pass-1 两路 `T.reduce` 传 `size=[rN, rS]`（S 维）+ `size=[rN, 1]`（N 维）（仅对有效范围求和；否则垃圾列/行污染 `gb/gw`）。**通道尾任务**：`C % num_kernels ≠ 0` 时 `if c < C` 守卫跳过越界通道。**N=1 / S=1**：工厂层分支跳过退化的 reduce 层（§5.4）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：Expert（显式 UB + 单 `T.Scope("Vector")`）；无核间/跨引擎协作。

### 7.2 同步点说明

Expert 模式下 UB 分配与数据放置显式，但**单个 Vector scope 内的顺序依赖链仍由编译器按 MTE2→V→MTE3 拓扑自动插 pipe_barrier**（`examples/conv1d/_conv1d_direct_kernel/DESIGN.md` 同款 Expert/Vector 形态：`T.copy`/`T.vcast`/`T.vmul`/`T.reduce` 顺序书写，未手写 sync 即编译运行）。本算子无跨引擎（无 Cube）与跨核协作，故无需 `T.sync_block_set/wait`、无需 `T.set_flag/T.wait_flag`：

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy`(GM→UB) 后 | 编译器自动 | MTE2→V 拓扑自动 pipe_barrier |
| v-op / reduce / vbrc / vcast 之间 | 编译器自动（顺序执行） | 同 Vector 引擎顺序依赖 |
| `T.copy`(UB→GM) 前 | 编译器自动 | V→MTE3 拓扑自动 pipe_barrier |
| serial 通道任务 / N/S 子块间 | 编译器自动 | 同核串行，无并发 |
| 跨核（通道间） | 无需 | one core per channel，通道归约单核闭环；无 workspace/atomics |

**显式同步备选**：若 Stage 3 引入显式双缓冲（预取下一子块）后实测出现 WAR/RAW 冒险，按 `.agents/skills/tilelang-op-design/references/decision-tree.md` 用 `T.set_flag`/`T.wait_flag`（Expert 手动同步形态）管理；本设计默认不引入（见 §6.3）。

### 7.3 pass_configs 配置

无特殊 pass_configs。Expert Vector scope 默认编译流水线（`examples/conv1d/_conv1d_direct_kernel/DESIGN.md` 同款）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 §0.1 源算子语义为唯一依据实现（移植源仓参考实现 `examples/TileOPs/tileops/workloads/norm.py` 的 `batch_norm_backward`），**不复刻 §0.6 的 NPU 算法**（无 vcast / fp32 中转 / persistent / 两步 reduce / 分布式 epilogue）——保证验证独立性。**golden 体直接调用 `torch.*` 模块函数**（`torch.sum` / `torch.mul` / `torch.sub` / `torch.div`），不使用等价的张量方法链，满足 Stage 3 的 golden-torch AST 检查。`mean/rstd` 由一次前向（`torch.var_mean` + `torch.rsqrt`）产生。

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
    go = grad_out.to(torch.float32)
    xf = x.to(torch.float32)
    w = weight.reshape(shape)
    m = mean.reshape(shape)
    r = rstd.reshape(shape)
    x_hat = torch.mul(torch.sub(xf, m), r)
    grad_bias = torch.sum(go, dim=(0, 2))                    # (C,)
    grad_weight = torch.sum(torch.mul(go, x_hat), dim=(0, 2))  # (C,)
    gb = grad_bias.reshape(shape)
    gw = grad_weight.reshape(shape)
    inner = torch.sub(torch.sub(torch.mul(float(L), go), gb), torch.mul(x_hat, gw))
    grad_x = torch.mul(torch.div(torch.mul(w, r), float(L)), inner)
    return grad_x.to(grad_out.dtype), grad_weight, grad_bias
```

**独立性说明**：golden 直接调用 `torch` 归约（`torch.sum`）与逐元素模块函数，不含 NPU 设计中的 persistent 分核、二维 `(bN,bS)` 分块、两步 `T.reduce`、`T.vbrc` 广播、fp32 中转链、分布式 epilogue 等任何算法特征；NPU 算法若与 golden 数值一致即证明迁移正确。另可用 `examples/TileOPs/tests/ops/test_batch_norm_bwd.py` 的输入生成（用 `torch.var_mean` 从 `x` 生成自洽的 mean/rstd）作为入口。

### 8.2 精度标准

来自 `examples/TileOPs/tileops/workloads/norm.py` 的 batch-norm-bwd tolerance（`grad_x` 逐元素；通道梯度用更宽的累加界 `sum_atol`）：

| dtype | atol | rtol | sum_atol（通道梯度） |
|-------|------|------|----------------------|
| float32 | 1e-5 | 1e-5 | 1e-4 |
| float16 | 1e-2 | 1e-2 | 1e-2 |
| bfloat16 | 2e-2 | 2e-2 | 2e-2 |

fp32 中转链对齐 golden opmath（D-1 verify_equiv：fp16/bf16 舍回后逐位一致）；探针最坏 max_abs_err 4.883e-04（fp16 grad_x）在容差内。

### 8.3 L0 门槛测试计划

L0 聚焦「编译通过 + 基本精度正确」（完整 L1/L2/Boundary 分层套件交由 `tilelang-op-develop`）。对照 = §8.1 golden（CPU）。

| # | Shape (N, C, S) | dtype | 档 | 验证目标 | 容忍度 |
|---|-----------------|-------|----|---------|--------|
| L0-1 | (32, 64, 1) | fp32 | held | resnet-fc（S=1 退化 reduce 路径） | 1e-5 |
| L0-2 | (32, 64, 1) | fp16 | held | resnet-fc fp16 | 1e-2 |
| L0-3 | (8, 64, 1024) | fp16 | held | resnet-stage1（held 临界 BPE） | 1e-2 |
| L0-4 | (8, 64, 1024) | bf16 | held | resnet-stage1 bf16（fp32 中转） | 2e-2 |
| L0-5 | (4, 128, 1024) | fp16 | held | resnet-stage2 | 1e-2 |
| L0-6 | (4, 256, 784) | fp32 | held | resnet-stage3 fp32（非 2^k S） | 1e-5 |
| L0-7 | (4, 256, 784) | fp16 | held | resnet-stage3 fp16 | 1e-2 |
| L0-8 | (4, 128, 1048576) | fp16 | tiled | large-spatial（bN=2,bS=4184 → 251 S-tile，尾 tile 2576；**非整除 reduce 路径**） | 1e-2 |
| L0-9 | (32, 2048, 512) | fp16 | tiled | **大 C 服务区代表**（bN=16 → 2 N-block，尾 N-block 16；**非整除 N 路径**） | 1e-2 |
| L0-10 | (8, 64, 900) | fp16 | held | test 30×30（S 非向量宽整数倍，尾 lane） | 1e-2 |
| L0-11 | (2, 8, 3) | fp16 | held | 微型 shape（N,S 均小且 S=3 尾 lane） | 1e-2 |
| L0-12 | (1, 4, 16) | bf16 | held | N=1 退化 reduce 路径 | 2e-2 |
| L0-13 | (4, 64, 4097) | fp16 | tiled | 微型非整除 tiled（bS=4184 → 2 S-tile，尾 tile 4097−4184<0 → 单 tile 覆盖、需 `T.min` 截尾） | 1e-2 |
| L0-14 | (2, 8, 0) | fp16 | — | 空张量（`L=0`）：须由 op 层 `numel()==0` 早退拦截（§9.1），kernel 不应收到 `S=0` | 精确零 |

**通过条件**：全部项通过 atol/rtol（`grad_x` 逐元素 + `grad_weight/grad_bias` 用 `sum_atol`）。L0-8/L0-9/L0-13 为尾块（S 尾 + N 尾 + 微型非整除）回归必测点。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 处理方案 |
|------|------|----------|
| **held 档 UB 临界**（resnet-stage1 180,224B，预算 184,320B） | 若 auto-multibuffer 膨胀超余量则编译失败 | 工厂 guard + 回退 tiled 档（`bN·bS` 缩水，两扫描）；BPE 已用同族实测膨胀值 |
| **中间轴归约（非尾轴）** | 逐通道平面 strided gather | `T.copy` 双显式 slice（探针真机验证）；行跨步 `C·S` |
| **bf16 无 Vector 算术** | bf16 直算编译失败 | fp32 中转链（§0.6 R6，必需） |
| **N=1 或 S=1 的退化 reduce** | `(1,1)` 上 dims=[1]/[0] reduce 非法（无维度变化） | 工厂层 `if n_gt1:` / `if s_gt1:` 分支跳过对应 reduce 层（§5.4） |
| **v-op 运行时切片非法** | `T.vcast`/`T.vmul` 等操作数切片为运行时 `rN/rS` 触发 parser `int(Var)` | v-op 一律用编译期常量 `(bN,bS)` 全 buffer 运算；尾块靠 reduce `size` + dst 显式 slice 截断（§5.4；同族宽核 Stage 3 实测） |
| **1 元素平面标量广播不可用** | `N==1 且 bS==1`（如 `L=1`）时 `T.vbrc((1,1)→(1,1))` 空广播 MLIR verify fail | 该形态由 op 层早退或升维；L0 不主动造 `(1,1,1)` 用例（同族宽核 Stage 3 实测） |
| **未初始化垃圾列/行（tiled 尾块）** | 尾子块 `[rN:]`/`[rS:]` 为未初始化/残留垃圾，无边 reduce 会将其计入 | 归约传 `size=[rN, rS]`（S 维）+ `size=[rN, 1]`（N 维）+ 双显式 slice；L0-8/L0-9/L0-13 门住 |
| **空张量 `L=0`** | NPU op `examples/TileOPs/tileops/ops/norm/batch_norm_bwd.py` 无 `numel()==0` 早退 → 空张量进入 kernel 构建（`S=0` 的 `T.copy`/`T.reduce` 退化非法） | 集成期在 NPU op/wrapper 增补 `numel()==0` 早退（返回 `empty_like(x)` + 零通道梯度，对齐 GPU op 语义，§0.1/§9.3/L0 清单补一项） |
| **roofline 常数 stale** | §1.6.0 估算引用 CONST-* 被 kb_stale_check 标记 | 估算仅作排序与 Stage 4 对账 |
| **通道任务负载不均** | C 非 40 整数倍（64/128/256/2048）→ 部分核多 1 任务 | 不均 ≤1 任务；Stage 4 可评估分核粒度 |
| **inf/NaN 传播** | 全 inf 通道 → NaN 输出（公式自然传播） | 与 golden/源一致；测试用例为有限随机输入 |
| **wrapper `ascend_mode` 不一致** | NPU wrapper `_BatchNormBwdBase.ascend_mode` 现为 `"Developer"`，本设计选 Expert | 集成期（Stage 5）把 wrapper 的 `ascend_mode` 改为 `"Expert"`；实现期 `ASCEND_MODE = "Expert"`（§2.3）；**接口（7 参调用 + out_idx=[-1]）无需改** |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | bN·bS 超预算 / held 膨胀超余量 | 编译失败 `ub overflow` | 缩 bN·bS / 回退 tiled；§4.5 表与实测 BPE |
| bf16 直传 v-op/reduce | 未做 vcast fp32 | 编译错误（dtype 不支持） | fp32 中转链 |
| reduce 退化 | N=1 或 S=1 仍做 dims=[1]/[0] | MLIR verify fail（无维度变化） | 工厂层 `if n_gt1` / `if s_gt1` 分支 |
| `clear=False` 未初始化 | 跨 tile/N-block 累加前未 `T.clear(red/red2/gb/gw)` | 静默数值错误（`docs/Tilelang.language/规约操作/T.reduce.md` §2.3） | 累加前 `T.clear` |
| 跨 dtype `T.copy` 当 cast | GM fp16 → UB fp32 一步 | 隐藏 VCast 舍入不可控（`docs/Tilelang.language/内存操作/T.copy.md` §3） | 显式 `T.vcast(rint)` + 同 dtype copy |
| v-op 运行时切片 | `T.vcast(stage[0:rN,0:rS], ...)` 传运行时 extent | parser `int(Var)`（`TypeError`） | v-op 用编译期 `(bN,bS)` 全 buffer；reduce `size` + dst slice 截断（同族宽核 Stage 3 实测） |
| 列广播直接 vsub | 想省 vbrc 用 `[bN,bS]−[1,1]` | 未文档化形态（文档仅列 `[M,N]−[M,1]`/`[1,N]−[1,1]`/标量） | `T.vbrc` 显式展开（探针同款） |
| 尾块 reduce 垃圾列/行 | 对完整 `(bN,bS)` 做 reduce 未限 extent | 未初始化/残留列/行计入 → `gb/gw` 偏大 | reduce 传 `size=[rN,rS]` / `size=[rN,1]`（§3.3 / §5.4） |
| if 体内新定义变量跨块引用 | TVM parser 作用域 | Undefined variable | if 体内只写预分配 buffer；dtype/held 分支置于工厂层 |
| 通道索引常量折叠遗漏 | `num_local_tasks` 非常量 | 动态循环边界 | host 层编译期常量（§5.5） |

### 9.3 特殊场景处理

| 场景 | 处理方式 |
|------|----------|
| S=1（resnet-fc） | 平面 `(bN,1)`；跳过 `dims=[1]` reduce，直接把列当行和；向量化轴退化为 N（小负载，延迟 bound） |
| N=1 | `(1,bS)`；`dims=[0]` reduce 退化 → 工厂分支跳过；L0-12 覆盖 |
| 大 C（本函数服务区） | 一维 persistent `num_kernels = min(C,40)` grid-stride（§5.5）；L0-9 覆盖 |
| large-spatial（S=1048576） | tiled 档，bN=2、bS=4184 → 251 S-tile（尾 tile 2576）；pass-1 累加（`size=[rN,rS]` 截断）、pass-2 重读；无跨核归并。**注**：默认选核把该 shape 路由 `bwd_split`（§5.5），本路径仅在 base 核被选中时适用 |
| 3D 输入 `(N, C, H, W)` | op 层 reshape `(N, C, H*W)` 后调 kernel（host 元数据视图） |
| 空张量 `L=0` | 集成期由 NPU op/wrapper 早退（`numel()==0` → `empty_like(x)` + 零通道梯度）；kernel 不接收 `S=0`（§0.1/§9.1） |
| 零方差通道（x 常量） | `x̂ = 0`；`gw = 0`；grad_x = scale·g（公式自然）；verify_equiv 角点已覆盖 |
| 全 inf / NaN 通道 | 自然传播（与源/golden 一致）；verify_equiv 角点已覆盖 |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/batch_norm_bwd/_batch_norm_bwd_kernel/
├── DESIGN.md                       # 本设计文档
├── verify_equiv.py                 # D-1 等价性机器验证脚本（已执行，EQUIV_PASS）
├── _batch_norm_bwd_kernel.py       # 算子实现（Stage 3 产出）
└── RETROSPECTIVE.md                # Stage 1 复盘（Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成并执行 | O1+O2 EQUIV_PASS（21 案例）；raw-moment 拒绝证据（判别段 A） |
| `_batch_norm_bwd_kernel.py` | ⬜ 待实现（Stage 3） | 工厂 + kernel + golden + L0 测试 |

### 10.3 命名规范

- 项目目录名：`batch_norm_bwd`；算子目录名 / 实现文件：`_batch_norm_bwd_kernel`（snake_case，不裁剪不变换）
- Golden 函数：`golden_batch_norm_bwd(grad_out, x, weight, mean, rstd)`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）——本文档
2. ✅ 等价性机器验证（verify_equiv.py，D-1）
3. ⬜ Golden 函数（§8.1，验证基准）
4. ⬜ 算子实现 `_batch_norm_bwd_kernel.py`（§3.3 伪代码 → 代码）+ L0 门槛测试（§8.3 全 14 项）
5. ⬜ 精度比对（vs golden，三输出）；集成期同步 wrapper `ascend_mode="Expert"` + `numel()==0` 早退（§9.1；接口保持 `out_idx=[-1]`、7 参调用不改）

### 10.5 迁移规则遵守确认

| 规则 | 遵守情况 | 说明 |
|------|----------|------|
| 接口参数不变 | ✅ | 工厂签名 `(N, C, S, dtype)` 与源一致 |
| `@T.prim_func` 参数名/顺序不变 | ✅ | `main(grad_out, x, weight, mean, rstd, grad_weight, grad_bias, grad_x)`；源 `out_idx=[-1]` 保持（仅 grad_x 自动分配） |
| out_idx 返回序契约 | ✅ | 返回 `[grad_x]`，与源 host 契约一致；wrapper 7 参调用 + 预分配 gw/gb 不改（§3.3 注） |
| threads 移除 | ✅ | 移除 `threads`（源 block_l 由 NPU `bN/bS` 表达，非 CDU 执行参数） |
| host 侧不改输入 | ✅ | 全部计算在 kernel 内；op 层仅 reshape view（+集成期 `numel()==0` 早退） |
| 从源码推断规格（不问用户） | ✅ | §0.1/§0.2 从源码 + manifest 解读；输出 shape 已回填标注依据 |
| 迁移纪律（不照抄 GPU threads/flat tile/谓词） | ✅ | §0.5/§0.6 四态判定 + NPU 重设计 |

---

## 11. 修订日志

**revision_index: 0（first_design）**

首次设计（migration-harness，`_batch_norm_bwd_kernel`，2026-10-14）。无上一版；无历史否决路径。

**关键设计决策摘要**（v0 / first_design）：

| # | 决策 | 依据 |
|---|------|------|
| 1 | one core per channel + UB 二维 `(bN,bS)` 流式分块 + 两步 `T.reduce` | §0.6 R2/R4 / 探针真机 PASS |
| 2 | held 单扫描（fp16/bf16 `N·S·22B ≤ 180KiB`；fp32 26B）/ tiled 两扫描（超大 L / 大 N）由 UB 预算切换 | §1.6.0 R1#1/#2 / §1.6.3 / §4.5 |
| 3 | 一维 persistent 分核 `num_kernels = min(C, 40)`（实查 20→40） | §0.6 R1 / §5.5 |
| 4 | 尾块 `T.min` + src/dst 双显式 slice + reduce `size=[rN,rS]`/`[rN,1]` | §0.6 R5 / 同族宽核 v0 缺陷教训 |
| 5 | fp16/bf16 → fp32 中转链（显式 `T.vcast(rint)`）；bf16 必需 | §0.6 R6 |
| 6 | 分布式 epilogue + 倒数乘（O1/O2），D-1 EQUIV_PASS 21 案例 | §1.6.1 |
| 7 | 保留中心化 `x̂` 归约（拒绝 raw-moment，抵消证据） | §1.6.1 判别段 A |
| 8 | 单输出 `out_idx=[-1]`（保持源 host 契约；grad_weight/grad_bias 传入并写入） | §0.1/§0.2 / §3.3 注 |
| 9 | 编程模式 **Expert**（`T.Scope("Vector")` + `T.alloc_ub`） | §2 / `conv1d_direct` Expert/Vector 先例 |
| 10 | UB `BPE` 用同族实测膨胀值（transit 22、fp32 26）+ 180KiB 安全预算 | §4.5 / 同族宽核 Stage 3 实测 |



