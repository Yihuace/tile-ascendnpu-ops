# _make_clamp_tensor_kernel（clamp_max / ClampMaxFwdOp）算子设计文档

> 迁移任务（`scenario=migration`，`migration_mode=harness`，GPU 仓为 TileOPs 同构工程）。源算子 = GPU 仓 TileOPs 的 `_make_clamp_tensor_kernel`（`ClampTensorFwdKernel` 的 builder 工厂；`ClampMaxFwdOp` 用其 `has_max`-only 分支：`y = min(x, hi)`），迁往 TileLang-NPUIR（`target="npuir"`，华为昇腾 NPU），编程模式固定 `Developer`。
>
> 本文 §0 记录源算子三问解读 → 算法调研（Phase R，结论见 §1.6.0）→ 硬件耦合性判定 → NPU 算法重设计；§1–§11 基于**重设计后的 NPU 算法**展开。
>
> **本算子相对同族 `ge`/`le`/`eq`（比较类）与 `bitwise_*`（位运算类）的关键差异**（由 §0.6 / §1.6.1 论证、`verify_equiv.py` + 设计期探针机器验证）：
> 1. **输出 dtype = `same_as(input)`（不是 bool）**：clamp_max 是「数值截断」而非比较——输出仍是 fp16/bf16/fp32，**无 bool→int8 承载链、无 `T.vcmp`、无升域链**，这是本算子相对比较族的最大简化。
> 2. **`T.vmin` 的原生能力远强于 `T.vcmp`**：设计期探针实测（见下）`T.vmin` ① **原生支持 bf16**（与文档 `docs/Tilelang.language/比较操作/T.vmin.md` §2.2.1 的 bf16「×」相反，以实测为准）；② **接受标量 src2**；③ **原生支持一维广播**（`[M,N]` vs `[M,1]`）；④ **原生传播 NaN**（x 或 hi 为 NaN → NaN）。故**无需** `T.vbrc` 显式同形物化，也无需升域。
> 3. **GPU 源的两条冗余计算可消除**（§1.6.1，`verify_equiv.py` 机器验证）：① 显式 `isnan` 守卫（NPU `VMinOp` 已传播 NaN，与 `torch.clamp_max` 逐点一致）；② fp32 中转（`min` 无累加，原生 dtype 结果 bit-exact）。
> 4. **manifest 含广播 workload**（`cnn-feat-broadcast [16,256,56,56]^[256,1,1]`、`row-broadcast [4096,4096]^[1,4096]`）——源 Op 层 host `_expand_flat` 会把上界**物化**到全 shape（`+2N·E` GM 流量，且违反 ascend-constraints.md §4 的 host 禁物化约束）。NPU 重设计把广播**内移 kernel**（行分解 + `T.vmin` 原生标量/一维广播 + `T.vbrc` 兜底）→ §0.6 R3（**必需**，非可选）。
>
> 关键能力结论来自**设计期探针实测**（D-3；脚本 `examples/clamp_max/_make_clamp_tensor_kernel/history_version/design_probe_vmin.py`，最小张量，总墙钟 << 10 min）：
> - `T.vmin` 同形 fp16 / bf16 / fp32 最小值 bit-correct（探针 Q-A 全 OK，含 bf16）。
> - **`T.vmin` 传播 NaN / ±Inf 与 `torch.clamp_max` 逐点一致**（探针 Q-B：NaN-in-x / NaN-in-hi / ±Inf 角点全 match）→ GPU 的 `isnan` 守卫冗余。
> - `T.vmin(x, scalar, y)` 标量 src2 可用（探针 Q-C1）。
> - `T.vmin([M,N], [M,1])` 一维广播可用（探针 Q-C2）。
> - **文档偏差**：`docs/Tilelang.language/比较操作/T.vmin.md` §2.2.1 dtype 矩阵把 bf16 标为「×」，实测 bf16 通过 → 以实测为准，列 §9.1 R1 待 L0 复核（含 fp32 中转回退）。
> 工具链 `tilelang 0.1.2+417dae9908ad` / torch_npu 2.7.1.post4 / CANN 8.5.0 / 本机 NPU（`NPUUtils.get().get_aicore_num()` 实查 = 20 ⇒ 纯 Vector 核 40；探针与 `verify_equiv.py` 均执行于该工具链）。

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

- **数学语义**：逐元素上界截断 `y[i] = min(x[i], hi[i])`，等价于 `torch.clamp_max(x, hi)`（Tensor 上界变体；`torch.clamp(x, min=None, max=hi)` 同义）。**NaN 语义与 PyTorch 一致**：`x` 或 `hi` 在位置 i 为 NaN 时 `y[i]` 为 NaN（源文件注释明确：CUDA `T.min`/`fmin` 不传播 NaN，故显式加 `isnan` 守卫，见 `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py` L3034–L3039）。
- **I/O 契约**：两输入（数据张量 `x` + 上界张量 `hi`）一输出；三者 **same dtype**（fp16/bf16/fp32）；输出 shape = `broadcast_shapes(input.shape, max.shape)`。Op 层在 host 侧把两输入 `expand` 到 out_shape 后 `contiguous().view(-1)` 展平为 1-D `(N_total,)` 连续张量，kernel 只处理同形 1-D（`examples/TileOPs/tileops/ops/elementwise/clamp_max.py` L104–L125）。
- **规约语义**：**无规约、无跨元素依赖**——每个输出元素只依赖同位置的 `x`/`hi`（经广播），`y[i]` 与 `y[j]` 完全独立，故并行组织 / 分核 / 分组的变化不影响任何数值结果（无浮点累加顺序问题）。
- **dtype 语义**：输入 / 输出族 `float16 | bfloat16 | float32`（manifest `ClampMaxFwdOp` signature；`ClampTensorFwdKernel.SUPPORTED_DTYPES`）。GPU 非 fp8 路径把操作数升 fp32 计算后回舍原 dtype（`T.cast(...,"float32")` → `T.min` → `T.Cast(dtype, r)`，源 L3148–L3160）；fp8 路径不在本算子契约内（`ClampMaxFwdOp` 的 manifest dtype 排除 fp8）。
- **边界语义**：NaN 传播（见上）；±Inf 按实数序参与 `min`；`±0` 的 `min(+0,-0)` 结果按 IEEE minimum 语义（PyTorch 与 NPU 实测一致，探针 Q-B）；非整除 shape 由尾块处理；0-dim / 广播上界由 Op 层 `_expand_flat` 处理。
- **host 侧语义**：`_expand_flat`（`expand → contiguous → view(-1)`）的广播物化、`reshape(out_shape)` 回形状、`eval_roofline` 元数据——属 Op 层契约（实现手段，非数学语义）；NPU 侧 Op 层 `examples/TileOPs/tileops/ops/elementwise/clamp_max.py` 已承接（O6「op-specific flow preserved unchanged」）。**注意**：`_expand_flat` 对广播 / 标量上界会触发 host 侧 `.contiguous()` **物化**（非纯视图），与 ascend-constraints.md §4 存在张力 → §0.5 / §0.6 R3。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（`torch.clamp_max` 广播直算），**不复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x`（input） | `(N_total,)` | float16 / bfloat16 / float32 | Op 层已广播并展平的 1-D 连续张量 |
| `hi`（max，上界） | `(N_total,)`（NPU 内移广播后为 `(max_numel,)`） | same_as(x) | 同上（源 Op 层已 `expand` 到 out_shape 后展平；广播时原 `max_numel ≤ N_total`） |
| `y`（output） | `(N_total,)` | **same_as(x)** | `out_idx=[2]`；一维展平视图，Op 层再 `reshape(out_shape)`；无转置 / 布局变换 |

**广播 workload（NPU manifest）**：4 条 workload 中 2 条为广播——`cnn-feat-broadcast` `[16,256,56,56]` 输入配 `[256,1,1]` 上界（上界沿空间维广播，等价于「每 (n,c) 行一个标量上界」）、`row-broadcast` `[4096,4096]` 配 `[1,4096]`（上界沿行广播，每行一条 `(4096,)` 向量）。源 Op 层 `_expand_flat` 会把上界 host 物化到全 shape（§0.5 / §0.6 R3）。

**输出 shape/dtype 依据**：源 `_make_clamp_tensor_kernel` 非 fp8 `has_max`-only 分支 `y: T.Tensor((N,), dtype)`（`examples/TileOPs/tileops/kernels/elementwise/clamp_max/_clamp_max_fwd_kernels.py` L202–L225）；manifest `ClampMaxFwdOp` `outputs.output.dtype: "same_as(input)"`、`shape_rules: output.shape == broadcast_shapes(input.shape, max.shape)`；wrapper `ClampTensorFwdKernel.output_dtype = dtype`（`clamp_max.py` L151）；NPU Op `ClampMaxFwdOp` `self.out_shape = broadcast_shapes(input, max)`、`N_total = prod(out_shape)`（`examples/TileOPs/tileops/ops/elementwise/clamp_max.py` L88–L91）。

### 0.3 实现算法解读（怎么算）

源文件：`examples/TileOPs/tileops/kernels/elementwise/clamp_max/_clamp_max_fwd_kernels.py`（Stage 0 提取件）；GPU 上下文 `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（`_make_clamp_tensor_kernel` @ L3017、`ClampTensorFwdKernel` @ L3227）、`/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/clamp.py`（`ClampMaxFwdOp` @ L209）。

**计算步骤分解**（源码每个计算语句均归入某步骤，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1（host） | `out_shape = broadcast_shapes(input.shape, max.shape)`；`N_total = prod(out_shape)` | input/max shape | out_shape / N_total（编译期常量） | 广播 lowering（实现手段） |
| 2（host） | `_expand_flat`：`t.expand(out_shape).contiguous().view(-1)`（input、max 各一次） | GM input / max | GM 展平 1-D `(N_total,)` | 广播物化（Op 层契约，非语义） |
| 3 | `block_size = threads*npt`；grid = `ceildiv(N, block_size)` | threads=512 / npt（fp32=4, 非 fp32=8） | CUDA 任务划分 | 任务划分 |
| 4 | `T.copy(x[bx·bs:(bx+1)·bs] → x_reg)`；`T.copy(hi[...] → hi_reg)` | GM x / hi | fragment `x_reg` / `hi_reg` | 数据搬入 |
| 5 | `for i,j in T.Parallel(threads,npt)`: `k=i·npt+j`; `x32=cast(x_reg[k],f32)`; `hi32=cast(hi_reg[k],f32)`; `r=T.min(x32,hi32)` | fragment | `r`（fp32） | `y = min(x, hi)`（逐元素本体） |
| 6 | `r = if_then_else(isnan(hi32), hi32, r)`; `r = if_then_else(isnan(x32), x32, r)` | `r` | `r`（NaN 传播） | NaN 语义修正 |
| 7 | `x_reg[k] = T.Cast(dtype, r)` | `r`（fp32） | fragment（原 dtype） | 回舍原 dtype |
| 8 | `T.copy(x_reg → y[bx·bs:(bx+1)·bs])` | fragment | GM y | 数据搬出 |
| 9（host） | Op 层 `result.view(out_shape)` | `(N_total,)` | broadcast shape | 后处理（Op 层契约） |

> 非 fp8 路径**无显式尾块边界判断**（`has_max`-only 分支，源 L3196–L3224）：依赖 TileLang `LegalizeSafeMemoryAccess` 隐式保护尾块。`has_min`/`has_max`/`is_fp8` 三分支在**工厂 build 期**（Python `if`）静态选择；`ClampMaxFwdOp` 只走 `has_max`-only 分支。

**数据流与内存访问模式**（GPU 视角）：

```
GM[x]  ──T.copy(连续段)──> fragment(x_reg) ┐
GM[hi] ──T.copy(连续段)──> fragment(hi_reg)┼─ 逐元素 cast→f32 · T.min · isnan 守卫 · cast→dtype ──> fragment ──T.copy(连续段)──> GM[y]
（kernel 内两操作数同形 (N_total,)；广播已由 Op 层 host 侧物化，kernel 不做索引广播）
```

**循环与并行结构**：CUDA grid（`ceildiv(N, block_size)` 个 block）× block 内 `T.Parallel(threads=512, npt)` 二维线程并行，每线程标量处理 `npt` 个元素，`block_size` = 2048（fp32）/ 4096（fp16/bf16）。无 `T.Pipelined`、无 fragment→GM 中间往返。

**host 侧逻辑**：`@functools.lru_cache(maxsize=32)` 按 `(N, dtype, has_min, has_max, ...)` 特化编译；`out_idx=[2]`；Op 层 `_expand_flat` 广播物化 + 展平、`reshape` 回形状。NPU wrapper（`examples/TileOPs/tileops/kernels/elementwise/clamp_max/clamp_max.py`）冻结工厂为 `_make_clamp_tensor_kernel(N, dtype, has_min, has_max)` 返回可调用 `(block_size)`，`default_config` 给 `{"block_size": 2048(fp32)/4096(fp16,bf16)}`（K11：`threads*npt` 折叠）。

### 0.4 优化手段解读（为什么快）

> 对照 migration-analysis.md §4.1 清单逐项核对；源码**显式**优化手段如下（无 SMEM tiling / warp shuffle / swizzle / 异步流水 / online 算法，对照清单无遗漏）。

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | `explicit_parallel`：`T.Parallel(threads, num_per_thread)` | 满并行 + 每线程多元素 ILP | 每线程标量处理 `npt` 个元素（`k=i·npt+j`） | CUDA block/thread 线程模型 | 硬件强相关 |
| 2 | 连续段 coalesced 读写 | 访存带宽 | 按 `block_size` 连续段访问，尾轴连续 | CUDA memory coalescing | 硬件强相关 |
| 3 | `register_copy` 策略（`alloc_fragment` + `T.copy` 载入→计算→写回） | 数据驻留寄存器 + 128bit 向量访存 | `T.alloc_fragment` 片上缓冲 | GPU 寄存器堆 + fragment 机制 | 硬件强相关 |
| 4 | fp32 中转计算（`cast→f32` → `min` → `cast→dtype`） | 精度 / 与参考实现一致 | 操作数升 fp32 后再 `min`，末端回舍 | 无（纯数值路径） | 可移植，但**对 min 冗余**（见 §1.6.1） |
| 5 | 显式 NaN 守卫（`isnan` + `if_then_else`，2 条/元素） | 对齐 PyTorch `clamp_max` 的 NaN 传播 | 逐元素 `isnan` 判定后 `select` 覆盖 | CUDA `fmin` **不传播 NaN**（语言/硬件语义，非调度） | 需 NPU 侧确认（NPU `VMinOp` 已传播 NaN → 舍弃） |
| 6 | 编译期分支（`has_min`/`has_max`/`is_fp8` Python `if`） | 零运行时分派开销 | build 期选择 kernel body | 无 | 可移植 |
| 7 | 工厂闭包 + `lru_cache(maxsize=32)` | 摊薄 JIT 编译开销 | 每参数组合只编译一次 | 无 | 可移植 |

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 数学语义 `y = min(x, hi)` | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.4 规则 1）；逐元素无累加顺序问题 |
| I/O 契约（1-D 进 1-D 出、broadcast out_shape） | 语义 | 无 | — | **保留** | 输出 shape 由 §0.2 确认；回形状在 Op 层 |
| dtype 契约（fp16/bf16/fp32） | 语义 | 无 | 有（探针 Q-A：`T.vmin` bf16/fp16/fp32 全 OK） | **保留** | 原生 dtype 直接计算（§1.6.1）；文档 bf16「×」以实测为准（§9.1 R1） |
| Op 层 `_expand_flat` 广播 + 展平 | 语义（Op 契约） | 无 | 有（`T.vmin` 原生一维广播 / 标量 + `T.vbrc`） | **重新设计**（广播 / 标量上界路径，必需） | §0.6 R3：manifest `cnn-feat-broadcast` / `row-broadcast` 两条广播 workload 触发 host 物化（`+2N·E`，违反 ascend-constraints.md §4）→ 广播内移 kernel；**同形路径**（smoke / elementwise-16M）`_expand_flat` 为纯视图（`shape` 相等 → 无 expand、`contiguous` no-op），合规 → 保留 |
| 一维 grid `ceildiv(N, block_size)` | 算法 | 弱（CUDA grid） | 有（`T.Kernel(一维, is_npu=True)`） | **保留** | 一维 Kernel 为本项目支持形态（ascend-constraints.md §1；`examples/elementwise/vec_add_1d.py`）；核数按 §5.5 分核适配（→ R2） |
| `T.Parallel(threads, npt)` 两级线程并行 | 优化 | CUDA 线程模型 | 无直接等价 | **重新设计** | → §0.6 R1/R2（一维 Kernel + Vector 向量化；npt 意图由向量 lane / block tile 承接） |
| `threads=` / `num_per_thread` kwarg | 优化 | CUDA | 无 | **舍弃** | npuir `T.Kernel` 不接受 `threads=`；wrapper K11 已把 `threads*npt` 折叠为单一 `block_size`，无性能意图丢失 |
| 连续段 coalesced 搬运 | 优化 | CUDA coalescing | 有（GM↔UB 连续段 `T.copy`） | **等价替换** | `T.copy` 连续段 GM↔UB（`docs/Tilelang.language/内存操作/T.copy.md`；`vec_add_1d.py` 同款） |
| `register_copy`（fragment 驻留） | 优化 | GPU 寄存器堆 | 有（UB） | **等价替换** | `T.alloc_fragment`→`T.alloc_ub`（migration-analysis.md §5.3「register→UB 中间缓冲」；v-prefix 要求操作数在 UB） |
| **标量逐元素 `T.min` 本体** | 优化 | CUDA 线程模型 | 有（`T.vmin`，向量） | **等价替换** | → §0.6 R1：`T.vmin(x_ub, hi_ub, y_ub)` 一条向量指令覆盖整块（探针 Q-A） |
| **fp32 中转计算** | 优化 | 无 | 冗余（`min` 结果必为操作数之一） | **舍弃** | → §0.6 R1 / §1.6.1 Item 2：原生 dtype 直接 `T.vmin`，bit-exact（`verify_equiv.py` E2） |
| **显式 NaN 守卫（2 条/元素）** | 优化 | CUDA `fmin` 不传播 NaN | NPU `VMinOp` **已传播 NaN**（探针 Q-B） | **舍弃** | → §0.6 R1 / §1.6.1 Item 1：`T.vmin` 裸式即匹配 `torch.clamp_max` NaN 语义（`verify_equiv.py` E1） |
| 编译期分支（has_min/has_max） | 优化（host） | 无 | — | **保留** | wrapper 只在 `has_max`-only 分支落地，无运行时分派 |
| 工厂闭包 + `lru_cache` 特化 | 优化（host） | 无 | — | **保留** | wrapper `forward` 已按 `block_size` 惰性构建 + 缓存（`clamp_max.py` L188–L190） |

**判定统计**：保留 6 项、等价替换 3 项、重新设计 2 项（R2 并行与分核 / R3 广播与标量上界内移）、舍弃 3 项（`threads=`/npt kwarg、fp32 中转、显式 NaN 守卫）。

> **与比较族（ge/le/eq）的判定差异**：比较族因 `T.vcmp` **不支持广播**（CG-2026-0021）与 bool 输出（bool GM 挂死，CG-2026-0022）必须做「显式同形物化 + bool→int8 承载 + （uint8）无符号域修正」三层重设计；本算子输出为数值 dtype、且 `T.vmin` 原生支持广播/标量/NaN 传播 → **无上述三层**，重设计量显著更小，反而多为「消除源冗余」（fp32 中转、NaN 守卫）。

### 0.6 NPU 算法重设计

**重设计项 R1: 逐元素截断本体（标量 `T.min` 循环 → 一条向量 `T.vmin`；同步消除 NaN 守卫与 fp32 中转）**

- **源方案**：`T.Parallel(threads,npt)` 二维线程循环内逐元素：升 fp32 → `T.min` → 2 条 `isnan` 守卫 → 回舍原 dtype（§0.3 步骤 5–7）。性能意图：以「CUDA 多线程 + 每线程 npt 个元素 ILP」把逐元素切碎的标量链摊到 GPU 吞吐。
- **NPU 新算法**：整块一次 `T.vmin(x_ub, hi_ub, y_ub)`——**同形**操作数均在 UB，输出 `y_ub` 同 dtype。**不带** `isnan` 守卫（NPU `VMinOp` 已传播 NaN，探针 Q-B），**不带** fp32 中转（`min` 结果必为操作数之一，原生 dtype 精确，§1.6.1 Item 2）。
- **语义保持论证**：**数学恒等**——`T.vmin` 的逐元素 `min` 与 PyTorch `clamp_max` 的截断语义逐点相同，且 NaN 传播（NaN-in-x / NaN-in-hi → NaN）与 ±Inf/±0 由硬件 `VMinOp` 承接（探针 Q-B 逐点 match；`verify_equiv.py` E1 在 fp64 参考下守卫式 ≡ 裸式 ≡ `torch.clamp_max`）。fp32 中转消除的 bit-exact 由 `verify_equiv.py` E2 机器验证（fp16/bf16 各 4621 角点/随机点，0 违反）。
- **意图承接**：源手段 1（满并行 + ILP）→ 向量 128bit lane；手段 4（fp32 中转）**经论证冗余后舍弃**（非静默丢弃，见 §1.6.1）；手段 5（NaN 守卫）**经 NPU 能力实测后舍弃**（非静默丢弃）；手段 6（编译期分支）保留。

**重设计项 R2: 并行结构与分核（CUDA block×thread 两级并行 + grid 超发 → 一维 Kernel + Vector 向量化 + persistent 分核）**

- **源方案**：`T.Kernel(ceildiv(N, block_size), threads=512)` 的 CUDA grid×block 两级并行，block 内 `T.Parallel(threads, npt)` 二维线程映射；grid 大小任意（256M workload 下 65536 个 block），超发由 CUDA 硬件调度器吞吐。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`；核内 `for i in T.serial(num_local)`（**静态边界**）以 grid-stride 映射逻辑 task `= i*num_kernels + cid`，`if task < num_logical` 屏蔽越界。逐元素 `min` 交 Vector 单元一条 `T.vmin` 覆盖整块（128bit 向量 lane）。核数按 §5.5 分核（`num_kernels = min(num_logical, 物理 Vector 核数)`，物理核数 = `NPUUtils.get().get_aicore_num() * 2`，本机 20 ⇒ 40）。
- **语义保持论证**：task 与输出连续块一一对应、各 task 完全独立（无跨 task 依赖、无竞写），映射顺序 / 分核数变化不改变任何输出值；向量化只是把逐元素标量 `min` 换成等价的逐元素向量 `min`（lane 间独立）。无累加顺序问题。
- **意图承接**：手段 1（`T.Parallel` 满并行 + ILP）→ Vector 128bit lane + `block_size` tile 粒度；手段 2（coalesced）→ `T.copy` 连续段；手段 3（fragment 驻留）→ `T.alloc_ub` 原地缓冲；`threads=`/npt 折叠为 `block_size`（wrapper K11），无性能意图丢失。

**重设计项 R3: 上界广播内移 kernel（host 物化 → 行分解 + `T.vmin` 原生标量/一维广播 + `T.vbrc` 兜底；ascend-constraints.md §4）**

- **源方案**：Op 层 `_expand_flat` 在 host 侧把 `max` `expand` 到 out_shape 后 `.contiguous()` 物化为完整张量再展平（`examples/TileOPs/tileops/ops/elementwise/clamp_max.py` L104–L112）——**同形路径**（`max.shape == out_shape`）下是纯视图（无 expand、`contiguous` no-op），合规；**广播上界**（manifest `cnn-feat-broadcast [16,256,56,56]^[256,1,1]`、`row-broadcast [4096,4096]^[1,4096]`）下触发 host 侧**数据物化**，违反 ascend-constraints.md §4「host 侧禁止改动输入张量真实内容 / 禁止 Padding」，并引入 `+2N·E` 额外 GM 流量（物化写 + kernel 读）。
- **NPU 新算法**：把上界广播内移 kernel。**行分解**（与比较族 ge/le 的 R1 同构，但无逐元素 divmod，仅每 task 一次行 base）：
  1. wrapper 传入未展开的 `hi`（连续 `(max_numel,)`）+ `coalesced_shape`（输出合并维）+ `hi_strides`（`hi` 在合并维上的 stride，`0` = 广播维）——`coalesce_broadcast_dims` 复用源域工具（wrapper 层）。
  2. 输出视为 `R = prod(coalesced_shape[:-1])` 行 × 行长 `L = coalesced_shape[-1]`；`k=1`（全同形 / 全标量）退化为 `R=1, L=N_total`（同形路径 = 单行，与 R1 统一）。
  3. 每 task（行 `row` + 块 `chunk`）：`x_base = row·L + chunk·block_size`；`tail = T.min(block_size, L − chunk·block_size)`；`T.copy(x[x_base:x_base+tail] → x_ub[0:tail])`。上界按其在合并维的末维 stride 分两路（末维必有 `hi_strides[-1] ∈ {0,1}`，因输出末维 size>1 时上界末维要么同 size（stride 1）要么 size 1（stride 0））：
     - **`hi_strides[-1] == 1`**（末维非广播，如 `row-broadcast` 的 `[1,4096]`、同形）：`hi_base = Σ_{i<k-1} coord_i(row)·hi_strides[i]`；`T.copy(hi[hi_base+chunk·block_size : …] → hi_ub[0:tail])`；**同形** `T.vmin(x_ub, hi_ub, y_ub)`（`row-broadcast` 时 `hi_strides=[0,1]` ⇒ `hi_base=0` 恒等 → 上界整行可**在 task 循环前装载一次**驻留 UB 复用，消除 4096 次重读）；
     - **`hi_strides[-1] == 0`**（末维广播，如 `cnn-feat-broadcast` 的 `[256,1,1]` 合并后 `[1,256,1]`）：`hi_base = Σ coord_i(row)·hi_strides[i]`（指向该行对应的上界元素）；`T.copy(hi[hi_base:hi_base+1] → hi1[0:1])`；`T.vmin(x_ub, hi1, y_ub)`（`(1,)`→`(block_size,)` 一维广播，`T.vmin.md` §2.2.2「src2 可以使用标量」/「仅一维不同且 size=1 → 带 brc 的 min」；探针 Q-C2 实测一维广播 OK）；**兜底** `T.vbrc(hi1 → hi_c[block_size])` 后同形 `T.vmin(x_ub, hi_c, y_ub)`（`docs/Tilelang.language/shape操作/T.vbrc.md`）。
  4. 单元素 / 0-dim 上界（`hi_strides` 全 0）退化为恒定 `hi1`（每 task 同值，可提升出循环），或直接 `T.vmin(x_ub, scalar, y_ub)`（探针 Q-C1）。
- **语义保持论证**：**值域恒等**——`min(x[i], bound[i])` 与 `min(x[i], expand(bound)[i])` 逐元素相同（广播只在索引层复制标量值，不改变任何数值）；行分解给出的 `hi_base` 与源 Op 层 `expand` 后的展平索引**逐点一致**（合并维的行/内层分解恰为 broadcast 索引，末维 stride ∈{0,1}）。`verify_equiv.py` E2 覆盖行/内层映射（E2 已含同形映射）、E3（标量 == 满张量）、E4（一维广播 == 显式 expand）在 fp64 参考下 **0 违反**；探针 Q-C1/C2 锁定 NPU 原生标量/广播行为。
- **意图承接**：源手段（host `_expand_flat` 通用广播 + 展平）的「让 kernel 只见 1-D」意图以「行 base + kernel 内原生广播」承接；`coalesce_broadcast_dims`（源工具）在 wrapper 层复用。**边界**：本项需 wrapper 工厂签名扩展（传入 `hi` 未展开 + 合并 shape/strides），是**必需**重设计（manifest 含 2 条广播 workload）。若实现暂保持冻结的 `(N, dtype, has_min, has_max)` 签名 + Op 层预展开，则同形 kernel 仍正确，但广播 workload 保留 host 物化（列 §9.1 R3 风险，交 Stage 2/3 与 Op 层契约方确认）。

**优化意图承接总表**（§0.4 → NPU 方案，无静默丢弃）：

| §0.4 优化手段 | NPU 承接 |
|---------------|----------|
| 1 `T.Parallel` 满并行 + ILP | Vector 向量化（`T.vmin` 128bit lane）+ 单一 `block_size` tile（R2） |
| 2 连续段 coalesced | `T.copy` 连续段 GM↔UB（同款语义） |
| 3 `register_copy` fragment 驻留 | `T.alloc_ub` 原地缓冲（同上） |
| 4 fp32 中转 | **舍弃**（冗余）：`min` 无累加，原生 dtype bit-exact（§1.6.1 Item 2；`verify_equiv.py` E2） |
| 5 显式 NaN 守卫 | **舍弃**（冗余）：NPU `VMinOp` 传播 NaN（探针 Q-B；§1.6.1 Item 1；`verify_equiv.py` E1） |
| 6 编译期分支 | 原样保留（wrapper 只在 `has_max`-only 落地） |
| 7 工厂 + lru_cache 特化 | wrapper `forward` 惰性构建 + 缓存（原样保留） |

### 0.7 标杆实现

- **源算子文件**：`examples/TileOPs/tileops/kernels/elementwise/clamp_max/_clamp_max_fwd_kernels.py`（Stage 0 提取件，接口契约冻结基准）；GPU 原始上下文 `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（`_make_clamp_tensor_kernel` @ L3017、`ClampTensorFwdKernel` @ L3227）、`/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/clamp.py`（`ClampMaxFwdOp` @ L209）。
- **NPU 集成目标（wrapper）**：`examples/TileOPs/tileops/kernels/elementwise/clamp_max/clamp_max.py`（`ClampTensorFwdKernel`：工厂签名 `(N, dtype, has_min, has_max)` → 可调用 `(block_size)`；`default_config` block_size = 2048(fp32)/4096(fp16/bf16)）；NPU Op 层 `examples/TileOPs/tileops/ops/elementwise/clamp_max.py`（`ClampMaxFwdOp`）。
- **规格来源**：NPU manifest `examples/TileOPs/tileops/manifest/elementwise.yaml` `ClampMaxFwdOp`（**4 workloads**：`smoke-1k [1024]`、`elementwise-16M [4096,4096]^[4096,4096]`、`cnn-feat-broadcast [16,256,56,56]^[256,1,1]`、`row-broadcast [4096,4096]^[1,4096]`；roofline `flops=N / bytes=3·N·elem_bytes`）；GPU manifest 同段在 `/tmp/opencode/TileOPs-ref/tileops/manifest/elementwise_unary_activation.yaml` `ClampMaxFwdOp`（2 workloads）。测试 GPU 仓 `tests/ops/test_special_elementwise_conformance.py`（`test_clamp_max_tensor` L253–L267、`test_clamp_max_nan_propagation` L325–L337、`test_clamp_max_init_signature_pytorch_aligned` L270–L276）；编译冒烟 `tests/ops/test_elementwise_compile.py` L775–L800；基准 GPU 仓 `benchmarks/ops/bench_independent_elementwise.py` L155–L170。**注**：调度 prompt 提到的 `tests/ops/test_clamp_max.py` / `benchmarks/ops/bench_clamp_max.py` **在源仓与 NPU 仓均不存在**（NPU manifest `source.test/bench` 指向它们但文件缺失，已实查 `find`）——clamp_max 的测试 / 基准实际落在上述共享文件（`test_special_elementwise_conformance.py` / `test_elementwise_compile.py` / `bench_independent_elementwise.py`）。
- **同族已踩坑参考**：`examples/ge/ge_fwd_kernel/DESIGN.md`、`examples/le/le_fwd_kernel/DESIGN.md`（比较族迁移先例，§0/§1.6 结构基准；差异见本文件头）、`examples/TileOPs/tileops/kernels/elementwise/mish`、`lerp_tensor`（elementwise 族 v-prefix 先例）。
- golden 以 §0.1 语义为依据（`torch.clamp_max` 广播直算），见 §8.1。

---

## 1. 概述

### 1.1 算子名称

`_make_clamp_tensor_kernel`（Op 层名 `ClampMaxFwdOp` / 算子族 `clamp_max`）

### 1.2 功能描述

逐元素上界截断前向核：对两个已展平为一维的同 dtype 张量（数据 `x` 与上界 `hi`）计算 `y = min(x, hi)`，支持 NumPy/PyTorch 广播（输出 shape = `broadcast_shapes(x_shape, hi_shape)`），输出 dtype = 输入 dtype（fp16/bf16/fp32）。纯逐元素、无规约、无跨元素依赖。**NaN 语义与 `torch.clamp_max` 一致**：`x` 或 `hi` 为 NaN → 输出 NaN。

### 1.3 数学公式

$$
y[i] = \min\big(x[\mathrm{off}_x(i)],\ hi[\mathrm{off}_{hi}(i)]\big), \qquad i = 0, 1, \ldots, N_{total}-1
$$

其中 `off_x`/`off_hi` 为广播映射（同形时 `off(i)=i`；源 Op 层在 host 侧把广播物化后展平，NPU 侧 §0.6 R3 可选内移 kernel 由 `T.vmin` 原生广播承接）；`min` 为 IEEE minimum（NaN 传播）。参考 API：`torch.clamp_max(x, hi)`。

**公式级优化后形式（§1.6.1）**：`y = vmin(x, hi)`（原生 dtype，无 fp32 中转、无 `isnan` 守卫）。

### 1.4 算法描述

**（迁移决策后的 NPU 侧算法；与源算法的结构差异逐条注明来源）**

单遍逐元素向量算法（§1.6.0 调研选定基线公式，无更优算法族替代）：

1. **任务分解（编译期）**：`num_logical = ceildiv(N_total, block_size)`（`k=1` 展平一维；`block_size ∈ {2048(fp32), 4096(fp16/bf16)}`，wrapper K11 传入）。
2. **核间**：`T.Kernel(num_kernels, is_npu=True)`（persistent，`num_kernels = min(num_logical, 物理 Vector 核数)`，§5.5）。
3. **核内（每 task 一块）**：连续段 `T.copy` 载入 `x`、`hi` 到 UB → **一条 `T.vmin(x_ub, hi_ub, y_ub)`**（原生 dtype，NaN 传播）→ 连续段 `T.copy` 写回（尾块按 `tail` 截断）。
4. **广播 / 标量上界（§0.6 R3，可选内移）**：`T.vmin(x_ub, scalar, y_ub)` 或 `T.vmin(x_ub, hi_ub, y_ub)`（一维广播）——规避 Op 层 host 物化。

与源算法的**结构差异**：① `T.Parallel(threads,npt)` 逐元素标量 `T.min` 链 → **一条向量 `T.vmin`**（§0.6 R1）；② CUDA grid×thread 两级并行 → 一维 Kernel + persistent 分核（§0.6 R2）；③ **消除源的两条冗余**：fp32 中转（`min` 无累加 → 原生 dtype bit-exact）与显式 `isnan` 守卫（NPU `VMinOp` 已传播 NaN）（§0.6 R1 / §1.6.1）；④ 广播 / 标量上界内移 kernel（§0.6 R3）。

**与同族 `ge`/`le`/`eq`（比较族）的算法差异（迁移视角）**：比较族输出 **bool** 且 `T.vcmp` **不支持广播**（CG-2026-0021）→ 必须「显式同形物化（`vbrc`）+ bool→int8 承载 + （uint8）无符号域修正」；本算子输出 **same_as(input) 数值 dtype**、`T.vmin` **原生支持一维广播 / 标量 / bf16 / NaN 传播** → **无 bool 承载链、无升域链、无 `vbrc` 物化**。这是本算子相对比较族的最大结构简化。

### 1.5 数据流图

```
任务分解（§0.6 R3 行分解；同形 / 全标量退化为 R=1）：
  output 合并为 R 行 × 末维 L；task=(row,chunk)；num_logical = R·ceildiv(L, block_size)
同形 / 末维非广播（§0.6 R1 / R3 分支 1；row-broadcast 上界整行可常驻 UB 复用）：
  GM[x]  ──T.copy(连续段, 尾块截断)──> UB[x_ub] ┐
  GM[hi] ──T.copy(连续段, 尾块截断)──> UB[hi_ub]┼─ T.vmin(原生 dtype, NaN 传播) ──> UB[y_ub] ──T.copy(连续段)──> GM[y]
末维广播（§0.6 R3 分支 2；如 cnn-feat [256,1,1] 每行一个标量上界；兜底 T.vbrc）：
  GM[hi[hi_base:hi_base+1]] ──T.copy 1 元素──> UB[hi1(1,)] ──T.vmin(x_ub, hi1 一维广播)──> UB[y_ub] ──T.copy──> GM[y]
全路径无 L1/L0 参与（纯 Vector）；无中间 GM 往返；无 fp32 中转 / 无 isnan 守卫
```

### 1.6 算法调研与优化分析 ⭐

> 设计第一优先级：先调研（1.6.0，迁移任务在 Phase M0 后、M1 前完成，结论已驱动 §0.5/§0.6），再公式级优化（1.6.1），再循环/标量的向量化替代（1.6.2），最后向量化轴与布局决策（1.6.3）。本节分析对象 = §1.6.0 选定、经 §0.6 重设计落地的 NPU 算法。`clamp_max` 为**单步逐元素类**（无规约/窗口/矩阵/多步融合），按 algorithm-research.md §2 采用**轻量调研**深度（四问各一行结论 + 依据），但四问不可跳过。

#### 1.6.0 算法调研（Algorithm Research）⭐

**R1 等价化简公式候选**（基线在表；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线（源算法 / 输入公式） | `y = min(x, hi)`（同形）+ **fp32 中转** + **2 条 `isnan` 守卫**（源 §0.3） | — | — | ✅（基线） |
| 1 | **裸 min（原生 dtype）** | `y = vmin(x, hi)`（原生 dtype，无守卫） | **数学恒等**：IEEE minimum（NaN 传播）≡ `torch.clamp_max`；fp32 中转对 `min` 冗余（结果必为操作数之一，精确）（E1/E2 机器验证 + 探针 Q-B） | **正向**：每元素省 2 cast + 2 isnan + 2 select（6 op/elem）→ 向量 op 数 8→1 | ✅ **采纳**（R1 本体） |
| 2 | 负号对偶 | `min(x,hi) = −max(−x, −hi)` | 恒等（含 NaN：`−max(−NaN,·)` = `−NaN` = NaN） | **负收益**：1 min → 2 取负 + 1 max（3 op/元素） | ❌ op 数 ×3 |
| 3 | ReLU 形 | `min(x,hi) = x − relu(x − hi)` | 实数恒等；**浮点非严格等价**：`x−hi` 引入舍入、大数相减失真，`x==hi` 附近可能翻转 | **负收益**：2~3 op/元素 + 浮点失真 | ❌ op 数 ×2+、浮点不等价 |
| 4 | 算术平均 | `min(x,hi) = (x + hi − abs(x−hi))/2` | 实数恒等；**浮点非严格等价**：加法/减法/除 2 各引舍入 + 溢出风险 | **负收益**：≥3 op/元素 + 浮点失真 | ❌ op 数 ×3、浮点不等价 |
| 5 | `T.vclamp` 组合 | `vclamp(x, y, −inf, hi)` = `max(x,−inf)` then `min(·, hi)` | 数学恒等（`max(x,−inf)=x`） | **负收益**：1 min → 1 max + 1 min（2 op/元素） | ❌ op 数 ×2 |
| 6 | 位模式比较选择 | 有符号位模式选最小 | 浮点（NaN/±0）不成立；整型 op 数 ≥2 | **负收益** | ❌ op 数 ×2+、浮点不成立 |
| 7 | **fp32 中转消除** | 原生 dtype `min` vs `min(cast(x,f32),cast(hi,f32))` 回舍 | **bit-exact 等价**（fp16/bf16 在 fp32 精确 + min 结果为操作数之一；E2 机器验证） | **正向**：省 2 cast/元素（2 向量 op/task） | ✅ **采纳**（§1.6.1 Item 2） |
| 8 | **NaN 守卫消除** | 裸 `vmin` vs `min` + 2 守卫 | **等价**（NPU `VMinOp` 传播 NaN = `clamp_max`；探针 Q-B + E1 机器验证） | **正向**：省 2 isnan + 2 select/元素 | ✅ **采纳**（§1.6.1 Item 1） |

**R2 在线算法**：**无在线变体需求（天然单遍）**。结构依据：`y_i` 仅依赖 `(x_{off_x(i)}, hi_{off_hi(i)})`，**无 running 统计量**（和/最大值/计数/在线 softmax 类状态）、无跨元素依赖——逐元素映射本身即单遍流式（algorithm-research.md §3 R2 结构判据「逐元素映射无跨元素依赖，单遍即天然在线」）；源算法亦为单遍（§0.4 无 online 手段），一致。收益口径（扫描遍数=1、中间缓冲 O(block_size)、UB 驻留）在候选间无差异。

**R3 复杂度对比**（带宽受限算子——算术强度 ≈ 1 min/elem ÷ (3·E) 字节，fp32 下 < 0.09 op/B，**访存为主导项**；口径：N 为展平元素数，E = elem_bytes，输出同 E；op 数按「每元素标量 op」与「每 task 向量 op」双列）：

| 算法候选 | FLOPs（口径：标量 op 数 / 每 task 向量 op 数） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-----------------------------------------------|---------------|---------|-------------|---------------------|
| **基线（GPU 源：fp32 中转 + 守卫）** | **8N 标量 op**（2 cast↑ + 1 min + 2 isnan + 2 select + 1 cast↓）/ **8 向量 op**（按标量链逐条向量化） | 3N·E（x + hi 读 + y 写）；**广播上界另 +2N·E**（host `_expand_flat` 物化写 + kernel 读） | 1 | O(block_size) fragment | N/block 独立 task；标量链 / 守卫为热点 |
| **NPU 重设计（§0.6 R1，裸 `T.vmin` 原生 dtype）** | **1N 向量 op**（每 task 一条 `T.vmin`）/ **1 向量 op** | 3N·E（连续段；同形） | 1 | O(block_size)（UB：3 buffer，§4.5） | num_logical 独立 task，零跨核同步；persistent 摊薄启动 |
| **NPU 重设计（§0.6 R3，广播内移）** | 1N（+ 每 row ≤1 条 `vbrc` 兜底） | **3N·E**（上界仅读其原生 `max_numel·E`，如 `row-broadcast` 上界 16KB、`cnn-feat` 上界 1KB；**免去源 host 物化的 `+2N·E`**） | 1 | O(block_size)（+ `hi1 (1,)` 标量 / 整行上界常驻） | 同（+ 每 row 行 base 标量） |
| 候选 2（负号对偶） | 3N / 3 | 同 | 1 | O(block_size) | 同 |
| 候选 3（ReLU 形） | 2~3N / 2~3 | 同基线 | 1 | O(block_size) | 同 |
| 候选 5（`vclamp`） | 2N / 2 | 同 | 1 | O(block_size) | 同 |
| **候选 7-2D（`[R_blk, block]` tile，§1.6.3 备选）** | 同 NPU 重设计（每 tile 一条 `T.vmin`），但 **task 数 ÷ R_blk** → 发射项 ÷ ~R_blk | 同 NPU 重设计 | 1 | O(R_blk·block_size) | task 数 ÷ R_blk，零跨核代价 |

基线（`min` 本体）已是 1 op/元素的下界；NPU 重设计在**截断本体**上无更优公式（候选 2/3/4/5/6 均 op 数 ≥2，或对浮点非严格等价），其收益全在**消除源的两条冗余**（fp32 中转 + NaN 守卫 → 向量 op 数 8→1，每 task 少 7 条向量 op）与**广播访存削减**（§0.6 R3 内移，消除 host 物化 `+2N·E`）。**注**：本算子 Op 层已把两操作数展平同形，故**无比较族那样的逐元素广播索引 divmod 热点**。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附 `docs/` / `examples/` / pattern-library 佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 可融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（源逐元素标量 cast/min/isnan/select 链） | ❌ 逐元素标量链 + 2 守卫落错单元（标量密集，非 Vector 单条指令） | OK | 标量链无向量 lane | 尾块依赖动态 shape | 单阶段 | ❌ 淘汰：NPU 无逐元素标量快路径，逐条向量化后 op 数 ×8 |
| **NPU 重设计（裸 `T.vmin` 原生 dtype）** | ✅ `min` → Vector `T.vmin`（fp16/bf16/fp32 同形 √，探针 Q-A；一维广播 / 标量 √，Q-C1/C2） | ✅ 2 buffer × block_size ≤ 192KB（§4.5，含 1.7x 膨胀后仍达标） | ✅ block_size×E = 32B 整倍数；向量宽度整除（fp16/bf16 ×8、fp32 ×4）；尾 lane 浪费 0% | ✅ N/block_size 全编译期特化；tail 仅 1 次/task | 单阶段无需流水；纯 Vector 无 CV 融合 | ✅ **选定** |
| 候选 2（负号对偶） | ✅ 仍落 Vector（但 op 数 ×3：2 取负 + max） | OK | OK | OK | OK | ❌ 无收益（op 数 ×3） |
| 候选 3（ReLU 形） | ✅ Vector（但需 `vsub`+`vrelu` 且引入浮点舍入） | OK | OK | OK | OK | ❌ op 数 ×2+、浮点非严格等价 |
| 候选 5（`vclamp`） | ✅ Vector（`T.vclamp`，但需传 `min_val=−inf` → 额外 `vmax`） | OK | OK | OK | OK | ❌ op 数 ×2（无收益） |
| 候选 7-2D（2-D tile） | ✅ 同形 `T.vmin`（rank-2 tile） | ✅ `R_blk×block_size×E×2×1.7`（fp32 R_blk=2,bs=4096 → 96KB×1.7 ≈163KB <192KB ✓） | ✅ 二维 tile 连续 | ✅ 静态 | 单阶段 | ⚠️ **备选**（发射项主导时可摊薄，见 §1.6.3） |

**调研结论**：选定 **基线截断语义（`y = min(x, hi)`）的 NPU 向量重设计算法**（§0.6 R1：一条原生 dtype `T.vmin`，同形 / 标量 / 一维广播），叠加 **§0.6 R2 一维 Kernel + persistent 分核**与 **§0.6 R3 上界广播内移**。关键依据：R3 表中 `min` 本体已是 op 数下界（1 op/元素），候选 2/3/4/5/6 均无访存收益且 op 数 ≥2（候选 3/4 对浮点非严格等价）；R4 中仅 NPU 重设计对 Vector / UB 容量 / 对齐 / 静态边界全部亲和（基线逐元素标量链在 NPU 落错单元被淘汰）。与基线的**结构差异**：① 逐元素标量 cast/min/isnan/select 链 → 一条向量 `T.vmin`（原生 dtype）；② 源的两条冗余（fp32 中转、NaN 守卫）消除；③ CUDA 两级并行 → 一维 Kernel + persistent；④ 上界广播**内移 kernel**（行分解 + `T.vmin` 原生标量/一维广播 + `T.vbrc` 兜底，**必需**——manifest 含 2 条广播 workload，`+2N·E` host 物化消除）。**「无更优替代」的调研范围**：① algorithm-research.md §5 参考表命中行 `ALG-elementwise`（R1：公共子表达式消除——`x`/`hi` 各用一次，无公共子式；除法乘倒数——无除法；cast 链合并——**本算子恰为 cast 链消除的命中项**（fp32 中转冗余）；R2 逐元素天然单遍——与本算子一致）；② 恒等变形自查——`min` 的最短形态为 1 条向量 `min`，负号对偶 / ReLU / 算术平均 / `vclamp` / 位分解均 op 数 ≥2 或对浮点非严格等价（R1 候选 2–6）；③ 本仓 `examples/elementwise/vec_add_1d.py`、`examples/TileOPs/tileops/kernels/elementwise/mish`、`lerp_tensor`、同族 `ge`/`le`/`eq`/`bitwise_*` 迁移——均直排 v-prefix 单遍结构，无 `min`/clamp 类公式变形先例；④ pattern-library `elementwise.md`/`cases.md`——elementwise 条目均为访存 / 布局调度类（copy-floor、loads-first、MTE3 strided、scatter-col）+ `PL-1.17-subfp32-fp32-transit`（sub-fp32 的 fp32 中转模式，**本算子经 §1.6.1 Item 2 论证该中转对 min 冗余**），无 `min` 谓词公式级条目；⑤ `kb_search "vmin vmax NaN"` 命中 `PL-1.11`（attention.md：**vmax/vmin 钳位只治 ±Inf、NaN 不被清除** → 佐证 vmin 传播 NaN）、`VP-2026-0087`（**vmin 族内原生支持一般列广播**，规则不族内统一）——与探针 Q-B/Q-C2 一致；⑥ 源算子结构分析（§0.3，逐元素标量链 + fp32 中转 + 守卫是唯一可优化点，已在 R1 承接）；⑦ **文档与实测不符项**（`T.vmin.md` §2.2.1 bf16「×」 vs 探针实测 OK）记 §9.1 R1。源算法优化手段的意图承接见 §0.6 末尾总表，无静默丢弃。

**设计期 roofline 估算（D-2）**（估算下界 = max(流量项, 发射项, 容量项)；常数引用 `pattern-library/constants.md`；本机实测 `NPUUtils.get().get_aicore_num()=20` ⇒ 40 Vector 核，与 `CONST-aicore-910B2C` 的 910B2C=24/48 不同源，流量项按 40/48 比例折算并标注；block_size=4096（fp16/bf16）/2048（fp32））：

| 代表 workload | 流量项 | 发射项 | 容量项 | 估算下界 |
|---------------|--------|--------|--------|----------|
| smoke `[1024]` fp32（N=1K，block_size=2048，chunks=1） | 3·N·4 = 12KB ÷ 1.05 TB/s ≈ **0.01µs** | 逻辑核 1 / 40 = 1 × 1 vmin × 0.5µs ≈ **0.5µs** | 豁免（轻量调研；tile 级缓冲无跨引擎往返） | **≈0.5µs** |
| elementwise-16M `[4096,4096]` fp16（N=16.78M，bs=4096，num_logical=4096） | 3·N·2 = 100.7MB ÷ 1.05 TB/s ≈ **95.9µs** | 4096 / 40 ≈ 103 × 1 op × 0.5µs ≈ **51.2µs** | 豁免 | **≈96µs**（流量主导） |
| elementwise-16M `[4096,4096]` fp32（N=16.78M，bs=2048，num_logical=8192） | 3·N·4 = 201.3MB ÷ 1.05 TB/s ≈ **191.7µs** | 8192 / 40 ≈ 205 × 1 op × 0.5µs ≈ **102.4µs** | 豁免 | **≈192µs**（流量主导） |
| cnn-feat-broadcast `[16,256,56,56]^[256,1,1]` fp16（N=12.85M，合并末维 L=3136，chunks=1，R=4096） | **内移**：(x 25.7MB + y 25.7MB + 上界 1KB) ÷ 1.05 ≈ **48.9µs**；**源 host 物化**：+ 物化上界 25.7MB×2 = 51.4MB → **≈97.9µs** | 4096 / 40 ≈ 103 × 1 op（`T.vmin(x_ub,hi1)` 原生广播）≈ **51.2µs**；vbrc 兜底 ×2 ≈ 102.4µs | 豁免 | **内移 ≈51µs**（发射主导）/ 源 host 物化 ≈98µs |
| row-broadcast `[4096,4096]^[1,4096]` fp16（N=16.78M，L=4096，chunks=1，R=4096） | **内移**：(x 33.6MB + y 33.6MB + 上界整行 8KB 常驻) ÷ 1.05 ≈ **64.0µs**；**源 host 物化**：+ 物化上界 33.6MB×2 = 67.1MB → **≈127.9µs** | 4096 / 40 ≈ 103 × 1 op ≈ **51.2µs** | 豁免 | **内移 ≈64µs**（流量主导）/ 源 host 物化 ≈128µs |

> 引用常数：`CONST-mte2-degradation`（MTE2 有效带宽随总流量退化，256M 档 3:1 R/W 混合全流量地板 ~1.26 TB/s @48 核，toolchain tilelang 0.1.2+ed787bb/910B2C/CANN 8.5.0，**kb_stale_check 标记 stale，引用前需重验**）；`CONST-vector-launch-overhead`（~0.5µs/op，toolchain 0.1.2+3a214cde/910B2C，与本次 417dae99 不同源——**标注为跨版本外推的未实测假设**）；`CONST-copy-floor-method`（≥16M 档计算链可被 auto multi-buffer 隐藏于 MTE2 窗口，佐证发射项可隐藏于流量窗口）。流量项按本机 40 核 / 常数 48 核折算（1.26×40/48≈1.05 TB/s），**标注为跨核数外推的未实测假设**。**本算子每 task 仅 1 条向量 op（vmin），发射项恒低于/接近流量项**——与比较族（每 task 3–11 op）相反，故**无需**发射项驱动的 2-D tile 主设计；2-D tile（§1.6.3 备选）仅在 `block_size` 偏小 / 极端 dtype 时有边际意义。容量项按轻量调研豁免（纯 Vector，2 buffer tile 级）。manifest workload 仅 fp16/bf16/fp32，无 uint8；fp16/bf16 同字节口径。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的基线截断算法。逐类别核查（**采纳 2 项**，均有四要素 + 机器验证）：

| # | 优化类别 | 核查结论 | 依据 |
|---|----------|----------|------|
| 1 | 恒等变形降代价（除法转乘倒数 / sqrt→rsqrt / 换底） | **无适用项**：公式为单次 `min`，无除法、无开方、无超越函数 | §1.3；`docs/Tilelang.language/数学操作/`（T.rsqrt/T.vexp 等无关） |
| 2 | 稳定化变形（减 max / logsumexp 改写） | **无需求**：无指数/求和/溢出放大结构；NaN/±Inf 由 `T.vmin` 承接 | §0.1 边界语义；探针 Q-B |
| 3 | 公共子表达式消除 | **无公共子式**：`x`/`hi` 各引用一次 | §1.3 |
| 4 | 算子降代换 | **无可降的高代价算子**：仅 1 条 `min`，已是 op 数下界；负号对偶/ReLU/算术/位/`vclamp` 分解均 op 数 ≥2 或对浮点非严格等价（§1.6.0 R1） | §1.6.0 R1 候选表 |
| 5 | 访存量削减 | **本体内达下界**（3N·E）；广播 / 标量上界的 host 物化 `+2N·E` 由 §0.6 R3 内移削减——属索引/操作数形态重设计 | §0.6 R3；§1.6.0 R3 |
| 6 | 归约结构优化 | **无归约** | §0.1 |
| 7 | **cast 链消除（算子降代换）** | **采纳 Item 2**：fp32 中转对 `min` 冗余 | 见下 |
| 8 | **谓词/守卫消除** | **采纳 Item 1**：显式 `isnan` 守卫冗余（NPU `VMinOp` 传播 NaN） | 见下 |

**采纳项 1：NaN 显式守卫消除**

- **原式**（GPU 源，2 条守卫/元素）：`r = min(x32, hi32); r = if_then_else(isnan(hi32), hi32, r); r = if_then_else(isnan(x32), x32, r)`。
- **优化后公式**：`y = vmin(x, hi)`（裸式）。
- **等价性论证**：**数学恒等（在 IEEE minimum 语义下）**——IEEE 754-2019 `minimum(x, NaN) = NaN`（NaN 传播），而 `torch.clamp_max` 的 NaN 语义即「x 或 hi 为 NaN → NaN」，故显式守卫在 NaN 传播的 `min` 上**冗余**。机器验证：`verify_equiv.py` E1（fp64 参考，含 NaN / ±Inf / ±0 / subnormal / 随机角点；fp16/bf16/fp32 各 525 点）候选式 ≡ 基线守卫式 ≡ `torch.clamp_max`，**0 违反**。NPU 硬件行为由探针 Q-B 实测锁定：`T.vmin` 在 NaN-in-x / NaN-in-hi / ±Inf 角点与 `torch.clamp_max` 逐点一致（fp16/fp32）。
- **收益量化**：每元素省 2 次 `isnan` + 2 次 `select`（按标量链向量化口径 = 每 task 省 **4 条向量 op**）；无访存变化。

**采纳项 2：fp32 中转消除（原生 dtype 计算）**

- **原式**（GPU 源）：`x32 = cast(x, f32); hi32 = cast(hi, f32); r = min(x32, hi32); y = cast(r, dtype)`。
- **优化后公式**：`y = vmin(x, hi)`（原生 dtype）。
- **等价性论证**：**bit-exact 等价**——`min(a, b)` 的结果**必为 `a` 或 `b` 之一**；fp16/bf16 的每个值在 fp32 中**精确可表示**（fp16: 11-bit 有效位 ⊂ fp32 24-bit；bf16: 8-bit 有效位 ⊂ fp32），故 `min` 在 fp32 域得 `r ∈ {a, b}`，回舍原 dtype 对 `{a, b}` 无损 → 无损往返。机器验证：`verify_equiv.py` E2（fp16/bf16 各 4621 角点 + 随机点，与原式 bit 对照）**0 违反**；探针 Q-A 直接证实 NPU 原生 dtype `T.vmin` bit-correct。
- **收益量化**：每元素省 2 次 `cast`（每 task 省 **2 条向量 op**）；UB 缓冲减少 2 个 fp32 upcast buffer（§4.5）。
- **阈值/例外**：对本算子（逐元素 `min`，无累加）成立；若未来引入累加/归约则需重新评估（累加域仍需 fp32）。

**采纳项收益合并**：每 task 向量 op 数 **8 → 1**（原：2 cast↑ + 1 vmin + 2 isnan-vcmp + 2 vselect + 1 cast↓）；无访存量变化。**优化后公式 = `y = vmin(x, hi)`（原生 dtype）**，为 §3.1 公式拆解的唯一输入。

**其余类别无进一步优化项（显式结论）**：无公共子表达式、无除法/超越函数、无归约、无可合并的多轮扫描（单遍）——逐类别核查结论见上表。

**等价性机器验证（D-1）** — `examples/clamp_max/_make_clamp_tensor_kernel/verify_equiv.py`（torch CPU，fp64 参照 + 目标 dtype 舍入对照），结果表（本次执行于工具链 `tilelang 0.1.2+417dae9908ad` / torch_npu 2.7.1.post4 / CANN 8.5.0 / 本机 aicore 20）：

| 项 | 对照对象 | 用例 | 违反数 | 结论 |
|----|----------|------|--------|------|
| **E1-{fp16,bf16,fp32}-candidate-vs-ref** | 裸 `min`（采纳项 1 候选式）vs `torch.clamp_max`（fp64 参考） | 各 525 点（±0/±1/±2/max/-max/tiny/-tiny/±Inf/NaN + 随机） | 0 / 0 / 0 | `EQUIV_PASS` |
| **E1-{fp16,bf16,fp32}-candidate-vs-baseline** | 裸 `min` vs 源守卫式（`min` + 2 `isnan` 守卫） | 同上 | 0 / 0 / 0 | `EQUIV_PASS` |
| **E2-{fp16,bf16}-native-vs-fp32transit** | 原生 dtype `min` vs fp32 中转 `min` 回舍（采纳项 2） | 各 4621 点（角点 + 4096 随机） | 0 / 0 | `EQUIV_PASS` |
| **E3-{fp16,bf16,fp32}-scalar-vs-full** | `vmin(x, scalar)` vs `vmin(x, full(scalar))`（§0.6 R3 标量路径） | 9 角点（含 NaN/±Inf/max） | 0 / 0 / 0 | `EQUIV_PASS` |
| **E4-{fp16,bf16,fp32}-bcast-vs-expand** | `min([M,N],[M,1])` 原生广播 vs 显式 expand（§0.6 R3） | 512 随机点 | 0 / 0 / 0 | `EQUIV_PASS` |

**OVERALL: `EQUIV_PASS`（14/14）**（脚本随算子目录留存，Stage 2 维度 8 可重跑）。**NPU 硬件层**的 `VMinOp` NaN 传播 / bf16 / 标量 / 一维广播由 `history_version/design_probe_vmin.py` 探针实测（Q-A/B/C1/C2 全 OK，D-3 设计期探针）——CPU 机器验证锁定公式等价，探针锁定硬件行为，二者互补。§3.1 公式拆解以「优化后公式 = `y = vmin(x, hi)`（原生 dtype）」为唯一输入。

> **采纳项对 GPU 源优化手段的影响**：源手段 4（fp32 中转）、手段 5（NaN 守卫）经上述论证后**舍弃**（非静默丢弃，§0.6 承接总表已登记）；其余手段（满并行、coalesced、fragment 驻留、编译期分支、工厂缓存）意图由 §0.6 R1/R2 承接。

**优化结论**：本设计**采纳 2 项**数学等价优化——① **NaN 显式守卫消除**（采纳项 1）、② **fp32 中转消除**（采纳项 2）。两项均满足四要素：数学等价成立（① IEEE 754 `minimum(x,NaN)=NaN` ≡ `torch.clamp_max` NaN 语义；② `min(a,b)` 结果必为 `a` 或 `b` 之一，fp16/bf16 值在 fp32 精确可表示 → 原生 dtype bit-exact），并经 `verify_equiv.py` **机器验证**（E1/E2，`EQUIV_PASS`，0 违反）与设计期探针（Q-A/Q-B）双重锁定。**收益**：每 task 向量 op 数 **8 → 1**，访存量不变（3N·E）；**优化后公式 = `y = vmin(x, hi)`（原生 dtype）**，为 §3.1 公式拆解唯一输入。**未采纳**其余候选（负号对偶 / ReLU 形 / 算术平均 / `vclamp` / 位模式）：op 数 ≥2 或对浮点非严格等价，依据见 §1.6.0 R1 候选表；公共子表达式 / 除法·超越函数 / 归约 / 多轮扫描各类别经核查**无进一步可采纳项**（依据见上表与 §1.6.0 R1）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案（§0.6 重设计后的 NPU 算法）中**全部**循环与标量计算点：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 逐元素截断 `min(x, hi)` | 源 `T.Parallel(threads,npt)` 标量 `T.min` 循环 | `T.vmin(x_ub, hi_ub, y_ub)`（同形，原生 dtype） | ✅ | —（佐证：`docs/Tilelang.language/比较操作/T.vmin.md` §2.1/§2.4 示例即向量 `vmin`；探针 Q-A 实测 bit-correct） |
| 2 | fp32 升域 cast | 源 `T.cast(x, "float32")` 逐元素 | **消除**（无需替代，采纳项 2） | ✅（消除） | —（`min` 无累加，原生 dtype bit-exact；E2 机器验证） |
| 3 | 回舍 cast | 源 `T.Cast(dtype, r)` 逐元素 | **消除**（无需替代，采纳项 2） | ✅（消除） | —（同上） |
| 4 | `isnan` 守卫 ×2 | 源 `T.isnan` + `T.if_then_else` 逐元素 | **消除**（无需替代，采纳项 1） | ✅（消除） | —（NPU `VMinOp` 传播 NaN；探针 Q-B + E1 机器验证） |
| 5 | 标量上界 / 一维广播上界（§0.6 R3） | 源 Op 层 host `_expand_flat` 物化 | `T.vmin(x_ub, scalar, y_ub)` / `T.vmin(x_ub, hi1, y_ub)`（`(1,)`→`(block_size,)` 原生广播）/ 兜底 `T.vbrc(hi1→hi_c)` + `T.vmin` | ✅ | —（佐证：`T.vmin.md` §2.2.2「src2 可以使用标量」+「仅一维不同且 size=1 → 带 brc 的 min」；`docs/Tilelang.language/shape操作/T.vbrc.md`；探针 Q-C1/C2） |
| 6 | 行 base 标量（§0.6 R3：`hi_base`/`x_base` 与 row 分解） | 源 host 侧 `expand` 物化（无 kernel 标量） | **无**（行级标量，每 task 一次） | ❌ | **block 级索引/任务映射计算**：每 task 仅 O(k) 次标量乘加（k≤3），非逐元素热点；行分解的 `hi_strides` 合并维由 wrapper 编译期给出 |
| 7 | 尾块长度 `tail = T.min(block_size, N − off)` | 标量比较 | 无 | ❌ | **依赖动态 shape 的边界处理**：非整除时逐 task 不同边界；已采用 `examples/elementwise/vec_add_1d.py` 的已验证 `T.min` 模式（`docs/Tilelang.language/比较操作/T.min.md`）；padding 方案违反 host 禁改输入约束（ascend-constraints.md §4） |
| 8 | 核内 task 调度循环 `for i in T.serial(num_local)` | 循环 | 无（persistent 结构需循环推进 task） | ❌ | **tile 级调度循环**（§5.5 persistent 分核标准结构，core-split-strategy.md §1 要素③）；边界为编译期静态值，循环体一次一个完整 task（内部全向量化） |
| 9 | host 侧 `num_logical/num_kernels/chunks` 计算、block_size 配置 | Python 标量 | — | ❌ | **host 元数据计算**（不在 kernel 内；工厂闭包编译期完成，符合 decision-tree §1） |

**向量化结论**：全部逐元素计算点（#1 截断本体、#2/#3 fp32 中转、#4 NaN 守卫、#5 上界广播）**均已处理**——#1 用一条向量 `T.vmin`，#2–#4 **直接消除**（采纳项 1/2，非"保留标量"），#5 用 `T.vmin` 原生标量/广播能力；替代/消除 API 均有 docs + 机器验证双佐证。保留 4 处标量/循环（#6 行 base / #7 动态边界 / #8 tile 调度循环 / #9 host 元数据），逐项理由见上表——§6 循环结构与本结论一致（kernel 内**无任何逐元素标量循环**）。**注**：本算子每 task 仅 1 条向量 op，是 §1.6.3 布局决策「发射项非主导」的依据。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（Op 层已把两输入展平为一维连续、输出回 broadcast shape），但**核内布局与 lane 映射是自由变量**。本节枚举候选并量化取舍。迁移任务：源 GPU 的 `threads×npt` 轴是**硬件组织的输入而非结论**（thread/warp 轴与 NPU Vector 轴不对应），须独立评估。**本算子特有约束**：`T.vmin` 原生支持一维广播 / 标量（探针 Q-C1/C2）→ 无比较族的「显式同形物化」需求；输出为数值 dtype → 无承载 dtype 选择。

**候选矩阵**（弃选给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | 展平一维 + **行分解**（`R` 行 × 末维 `L`）+ 整块向量 `T.vmin`（同形 / 标量 / 一维广播） | 展平 N 轴 / 合并末维 L | 无 | 基线：连续段 `T.copy`，访存 MTE2 最优；对齐 `block_size×E` 32B 整除；UB 3 buffer ≤192KB（§4.5）；每 task 1 条向量 op；广播上界经 `hi1`/`vbrc` 内移（§0.6 R3） | ✅ **采纳**（主选） |
| 2 | 2-D 块化 tile `[R_blk, block]`：连续段载入，`T.vmin([R_blk,block],[R_blk,block])` 同形 | 末维（+ 行轴） | 无 | 备选：**发射项按 R_blk 摊薄**——但本算子每 task 仅 1 op，发射项本已非主导（§1.6.0），收益边际；代价 UB `R_blk×block×E×3×1.7` | ⚠️ **备选**（仅极端小 block / 发射受限时，见裁决计划） |
| 3 | host 侧预广播物化 `max`（源 Op 契约） | 展平 N | host `expand→contiguous` | 同形路径合规（纯视图）；**广播路径违反 ascend-constraints.md §4 + 额外 `2N·E` GM 流量**（cnn-feat +51MB / row-broadcast +67MB，§1.6.0） | ❌（广播路径弃选，改用 §0.6 R3 行分解内移） |
| 4 | 逐元素标量 `T.Parallel` 截断（源直译） | 无向量轴 | 无 | ❌ **负收益**：NPU 无逐元素标量快路径（落 Scala、无 128bit lane）；§1.6.2 #1 已判定向量化 | ❌ |
| 5 | 位模式选最小（int 专属） | int 专属 | 无 | ❌ 浮点不成立（NaN/±0）；op 数 ×2+；见 §1.6.0 R1 候选 6 | ❌ |
| 6 | 先 `vclamp` 再取（`max`+`min` 两步） | 展平 N | 无 | ❌ 负收益：2 op/元素（§1.6.0 R1 候选 5） | ❌ |

**轴质量评分（主选 #1）**：整除性——向量宽度 fp16/bf16 ×8、fp32 ×4；`block_size ∈ {2048, 4096}` 均为宽度整倍数，尾 lane 浪费 0%（`N_total` 非整除时仅尾部 1 个 task 有 `(block_size−tail)` 垃圾 lane，不落 GM）；累加链——无累加（逐元素，无跨步串行累加维，不阻碍向量化）；repack 代价——无（直接用 I/O 原生连续段）；UB 容量——3 buffer × block_size ≤ 41KB（×1.7，§4.5），余量充裕；广播上界（`row-broadcast` 整行 8KB / `cnn-feat` 标量 1KB）另计仍远低于上限。

**源码轴对照（迁移任务要求）**：源 `T.Parallel(threads, npt)` 的隐式轴向是「连续 N 轴 + 每线程 npt 连续元素」，与主选 #1 的「连续段 + 展平 N 轴向量化」一致；源为标量 `T.min`，NPU 换一条 `T.vmin` 覆盖整块——轴选择无需额外重设计（无广播索引 divmod，与比较族不同）。

**实验裁决模式（2-D 块化 tile 备选 #2）**：主选 #1 每 task 仅 1 条向量 op，发射项本已低于流量项（§1.6.0），备选 #2 收益依赖**未实证常数**（向量发射摊销律、二维 `T.vmin` 每 op 开销），判定裕度小但存在（尤其若 Stage 4 采用更小 block_size），故：
- **主选**：方案 #1（展平一维，(block_size,) 1-D tile，进 Stage 3）。
- **备选**：方案 #2（2-D `[R_blk, block]` tile；推荐 fp16/bf16 R_blk=2、fp32 R_blk=2；结构：buffer 形状 `(R_blk, block)`、连续段载入、`T.vmin` 在 rank-2 tile 上执行、尾块按 `tail` 截断；分核三要素按 2-D task 粒度重算 `num_logical = ceildiv(R, R_blk) × ceildiv(N, block)`）。
- **裁决计划**：Stage 4 在 `[4096,4096]` fp16 与大 shape 代表 shape 实测方案 #1 vs #2；指标 = msprof op Task Duration + vec_ratio + mte2_active_bw；判定阈值 = 备选 latency < 主选 × 0.97 才翻转；**shape 特化工厂（lru_cache）评估**：wrapper 按 `(N_total, dtype)` 特化，天然支持按 shape 分派，无全局二选一需求——结果回写 §1.6.3。

**布局决策结论**：选定方案 #1——核内布局 = `(block_size,)` UB buffer（`x_ub`、`hi_ub`、`y_ub`，均原生 dtype；广播路径另用 `hi1 (1,)` 标量源与可选的整行上界常驻 buffer），向量化轴 = **展平 N 轴 / 合并末维 L**，repack = 无；广播 / 标量上界经 §0.6 R3 行分解由 `T.vmin` 原生标量/一维广播（+ `T.vbrc` 兜底）承接。该结论同步落入 §3.3 伪代码（buffer 形状 `(block_size,)`、`T.vmin` 操作数即整 buffer）与 §6 循环结构；§4.5 UB 预算按此布局计算。方案 #2 作为备选 + 裁决计划随设计冻结进 Stage 4。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（`op_requirements.programming_mode` 指定；且与算子特征判定一致）

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（逐元素 `T.vmin` 一条向量指令 + 连续段 `T.copy`），无 matmul、无规约 | 无 Cube/L0 需求，不需要 MixCV |
| 复杂度级别 | 单步算子（一次搬运 + 一条 vmin + 一次搬出，广播路径额外原生 brc） | 无多阶段流水/精细 buffer 生命周期控制需求 |
| 同步 | 每 task 独立、无核间依赖 | 无手动 `sync_block_set/wait` 需求 |
| API | v-prefix（`T.vmin`）+ `T.copy` + `T.alloc_ub` 均为 Developer 可用形态 | `docs/Tilelang.language/比较操作/T.vmin.md` §2.4 示例即 `@tilelang.jit(target="npuir")` Developer 形态；探针 Q-A/B/C1/C2 均于 Developer 模式实测通过 |
| 先例 | `examples/elementwise/vec_add_1d.py`（一维逐元素 + `alloc_ub` + v-prefix，Developer） | 同构算子已验证 |

决策树路径：纯 element-wise 单步 → Developer 模式（decision-tree.md §2；非 persistent 混合算子，无 CG-2026-0010 的 Developer/persistent+gemm 限制，本算子纯 Vector persistent）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub((block_size,), dtype)` 显式 UB（v-prefix 要求操作数在 UB，`T.vmin.md` §2.1） |
| 计算方式 | v-prefix 向量 API（`T.vmin`），无 `T.Parallel` 逐元素标量循环 |
| 同步 | Developer 自动同步（copy→compute→copy 顺序依赖由编译器保证） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` |
| 环境变量 | `TILELANG_ASCEND_MODE=Developer`（`examples/elementwise/example_elementwise_add.py` 先例；wrapper `ascend_mode="Developer"`） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式 = §1.6.1 优化后公式：`y = vmin(x, hi)`（原生 dtype）。拆解与 §1.6.2 向量化结论一致（逐元素计算点全部向量化/消除，无逐元素标量循环）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `off = task * block_size`；`tail = min(block_size, N − off)` | 块偏移 + 尾块长度（标量，每 task 一次） |
| 2 | `x_ub[0:tail] ← x[off : off+tail]` | 数据连续段载入（GM→UB） |
| 3 | `hi_ub[0:tail] ← hi[off : off+tail]`（或标量 `hi1`，§0.6 R3） | 上界连续段载入 / 单元素载入 |
| 4 | `y_ub = min(x_ub, hi_ub)` | 逐元素截断（向量 `T.vmin`，原生 dtype，NaN 传播） |
| 5 | `y[off : off+tail] ← y_ub[0:tail]` | 结果写回（UB→GM，尾块截断） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| task 分派 | `task = i*num_kernels + cid` | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` + `T.serial(num_local)` | num_kernels/num_local 编译期常量（§5.5） | Developer |
| 连续段搬入 | `x[off:off+tail] ↔ x_ub[0:tail]` | `T.copy(src_slice, dst_slice)` | 连续段切片拷贝（runtime 边界 OK；`docs/Tilelang.language/内存操作/T.copy.md`；`vec_add_1d.py` L34–L38） | Developer |
| 逐元素最小值 | `y_ub = min(x_ub, hi_ub)` | `T.vmin(x_ub, hi_ub, y_ub)`（`npuir_min`） | 同形 / 一维广播 / 标量 src2 均支持（`docs/Tilelang.language/比较操作/T.vmin.md` §2.2.2；探针 Q-A/C1/C2）；dtype fp16/bf16/fp32 | Developer |
| 标量上界 | `y_ub = min(x_ub, c)` | `T.vmin(x_ub, c, y_ub)` | c 为 Python/TIR 标量（`T.vmin.md` §2.2.2「src2 可以使用标量」；探针 Q-C1） | Developer |
| 尾块长度 | `tail = min(block_size, N − off)` | `T.min(a, b)` | `docs/Tilelang.language/比较操作/T.min.md`（标量 `arith.minimumf`） | Developer |
| 整除向上取整 | `ceildiv(N, block_size)` | `T.ceildiv` | `docs/Tilelang.language/数学操作/T.ceildiv.md` | Developer |
| buffer 分配 | UB 片上缓冲 | `T.alloc_ub((block_size,), dtype)` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`（1D 支持 ✓） | Developer |

### 3.3 计算伪代码

> dtype 位置传参（模板规则：`T.Tensor((N,), dtype)` 不用 `dtype=` 关键字）。工厂签名保留源契约（wrapper 已冻结调用形态）。block_size 相关分解**必须在 `@tilelang.jit` 函数体内**用被调用 block_size 计算（bitwise_and Stage 3 TRAP-jit-blocksize-decomposition；`T.ceildiv` 在 trace 期折叠）。

```python
def clamp_max_fwd_kernel(N_total, dtype, has_min, has_max,
                         coalesced_shape=None, hi_strides=None, hi_numel=None,
                         block_size=4096):
    # has_min=False, has_max=True（ClampMaxFwdOp 契约）；dtype ∈ {float16, bfloat16, float32}
    # coalesced_shape/hi_strides 仅在广播路径需要（§0.6 R3）；同形路径为 k=1 退化
    out_dtype = dtype                 # same_as(input)（§0.2）
    vector_cores = NPUUtils.get().get_aicore_num() * 2   # 纯 Vector 翻倍；实查见 §5.5

    same_shape = (coalesced_shape is None) or (len(coalesced_shape) == 1)
    if same_shape:
        L = N_total
        R = 1
    else:
        L = coalesced_shape[-1]                  # 合并末维
        R = 1
        for d in coalesced_shape[:-1]:
            R *= d
    hi_last_stride = None if same_shape else hi_strides[-1]   # ∈ {0,1}（广播维）

    @tilelang.jit(out_idx=[2], target="npuir")   # 源 has_max-only 为 out_idx=[2]
    def kernel(block_size=block_size):
        chunks = T.ceildiv(L, block_size)                # trace 期折叠
        num_logical = R * chunks
        num_kernels = min(num_logical, vector_cores)     # persistent 分核（§5.5）
        num_local = T.ceildiv(num_logical, num_kernels)  # 静态边界

        @T.prim_func
        def main(
            x: T.Tensor((N_total,), dtype),
            hi: T.Tensor((hi_numel or N_total,), dtype),
            y: T.Tensor((N_total,), out_dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_ub((block_size,), dtype)
                hi_ub = T.alloc_ub((block_size,), dtype)     # 末维非广播时用
                hi1 = T.alloc_ub((1,), dtype)                # 末维广播（标量上界）时用
                y_ub = T.alloc_ub((block_size,), dtype)
                for i in T.serial(num_local):                # 核内串行，静态边界
                    task = i * num_kernels + cid
                    if task < num_logical:
                        row = task // chunks
                        inner_off = (task % chunks) * block_size
                        tail = T.min(block_size, L - inner_off)
                        x_base = row * L + inner_off
                        T.copy(x[x_base : x_base + tail], x_ub[0:tail])
                        if same_shape or hi_last_stride == 1:
                            # 末维非广播：同形整段（row-broadcast 时 hi_base 恒 0 → 可提升出循环常驻）
                            hi_base = 0 if same_shape else <outer base from hi_strides>(row)
                            T.copy(hi[hi_base + inner_off : hi_base + inner_off + tail],
                                   hi_ub[0:tail])
                            T.vmin(x_ub, hi_ub, y_ub)         # 同形，原生 dtype，NaN 传播
                        else:
                            # 末维广播（如 cnn-feat [256,1,1]）：每行一个标量上界
                            hi_base = <outer base from hi_strides>(row)
                            T.copy(hi[hi_base : hi_base + 1], hi1[0:1])
                            T.vmin(x_ub, hi1, y_ub)           # (1,)→(block_size,) 一维广播
                            # 兜底：T.vbrc(hi1, hi_ub); T.vmin(x_ub, hi_ub, y_ub)
                        T.copy(y_ub[0:tail], y[x_base : x_base + tail])

        return main

    return kernel
```

> 实现注记：① 源 `has_max`-only 分支无尾块显式判断（依赖 `LegalizeSafeMemoryAccess`），NPU 版显式用 `tail = T.min(block_size, L − inner_off)` 以保证 UB 切片边界静态可证（`vec_add_1d.py` 同款）。② **无 fp32 cast、无 isnan 守卫**（§1.6.1 采纳项 1/2）。③ 标量 / 0-dim 上界（`hi_numel==1`）：`hi_base=0`、走末维广播分支，`hi1` 装载一次。④ **同形路径**（smoke / elementwise-16M）为 `R=1, L=N_total` 的单行退化——`num_logical = ceildiv(N_total, block_size)`，与 §5.5 一致。⑤ `<outer base from hi_strides>(row)` 为 Python 层按 `coalesced_shape[:-1]`/`hi_strides` 生成的 `T.floordiv`/`T.floormod` 展开链（因子全为编译期常量，仅每 task 一次）。⑥ v-prefix 整 buffer 操作在尾块（`tail<block_size`）时对 `[tail:]` 段一并计算，仅 `[0:tail]` 写回，垃圾不落 GM（`vec_add_1d.py` 官方先例同款）。

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|----------|----------|
| `T.vmin`（`npuir_min`） | `docs/Tilelang.language/比较操作/T.vmin.md` §2.1/§2.2.2/§2.4；`tilelang/language/customize_npuir.py` L186（`B: Second input tensor or scalar value`） | ✅ **探针实测**：fp16/bf16/fp32 同形 bit-correct（Q-A）；NaN 传播 = `clamp_max`（Q-B）；标量 src2 可用（Q-C1）；一维广播可用（Q-C2）。**文档 bf16 标「×」与实测不符**（§9.1 R1） |
| `T.vmax` / `T.vclamp`（备选路线） | `docs/Tilelang.language/比较操作/T.vmax.md`/`T.vclamp.md` | ✅ 存在，但本算子弃用（op 数 ≥2，§1.6.0 R1 候选 5；`T.vclamp` 展开为 `npuir_max`+`npuir_min`，见 `T.vclamp.md` §3） |
| `T.min`（标量） | `docs/Tilelang.language/比较操作/T.min.md`（scalar min，`arith::MinimumFOp` for float） | ✅ 尾块长度标量用（`vec_add_1d.py` L31–L33 先例） |
| `T.copy`（gm↔ub、切片、runtime 边界） | `docs/Tilelang.language/内存操作/T.copy.md`；`vec_add_1d.py` L34–L38 | ✅ |
| `T.alloc_ub(shape, dtype)` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`；`vec_add_1d.py` L28–L30 | ✅（1D 支持） |
| `T.ceildiv` | `docs/Tilelang.language/数学操作/T.ceildiv.md` | ✅ |
| `T.Kernel(grid, is_npu=True)` | `examples/elementwise/vec_add_1d.py` L27；`T.vmin.md` §2.4 示例 | ✅（examples 先例） |
| `T.serial` + `task = i·K + cid` + `if` | `T.vcast.md` §2.4 示例（persistent 分发模式） | ✅ |
| `@tilelang.jit(out_idx=[2], target="npuir")` | 源 `out_idx=[2]` 契约（`_clamp_max_fwd_kernels.py` L202）+ `example_elementwise_add.py` L13 | ✅ |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No（源即一维 grid；NPU 一维 `T.Kernel(num_kernels)`） | 不涉及 |
| GPU 专用 API（`threads=` kwarg / `T.Parallel(threads,npt)`） | Yes | `is_npu=True` + v-prefix 取代（§0.5 舍弃项 / §0.6 R1/R2） |
| GEMM 非整除 | No（无 GEMM；非 GEMM 的尾块由 §6.4 处理） | §6.4 尾块策略 |
| L0C 溢出 | No（纯 Vector，无 L0C） | 不涉及 |
| 物理核数限制（分核三要素） | Yes（逻辑核 1–65536 vs 物理 Vector 核 40） | §5.5 persistent 分核 |
| Host 侧输入操作约束 | Yes（Op 层 `_expand_flat` 对广播 / 标量上界做 host 物化） | 同形路径合规（纯视图）；广播 / 标量路径由 §0.6 R3 内移 kernel（`T.vmin` 原生广播/标量）——符合 ascend-constraints.md §4 |
| uint8 张量 / bool 输出限制 | No（本算子输出 same_as(input)，无 bool/uint8） | 不涉及（区别于比较族） |
| `T.vcmp` 广播 / int8 限制 | No（本算子不用 `T.vcmp`） | 不涉及 |
| `T.vmin` 文档 bf16「×」 | Yes（bf16 为 manifest dtype） | 探针 Q-A 实测 bf16 OK（以实测为准）；§9.1 R1 待 L0 复核 + fp32 中转回退 |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 并行 | `T.Kernel(grid, threads=512)` CUDA 两级 | 一维 `T.Kernel(n, is_npu=True)` | §0.6 R2 |
| 元素并行 | `T.Parallel(threads, npt)` 标量循环 | v-prefix 向量 API（无逐元素循环） | §1.6.2 #1 |
| 截断本体 | 原生域标量 `T.min` + fp32 中转 | `T.vmin`（原生 dtype）一条向量指令 | §0.6 R1 / §1.6.1 |
| NaN 语义 | 显式 `isnan` 守卫（CUDA fmin 不传播） | `T.vmin` 原生传播（无需守卫） | §1.6.1 Item 1 |
| fp32 中转 | `cast↑` + `min` + `cast↓` | 消除（原生 dtype bit-exact） | §1.6.1 Item 2 |
| 上界广播 | Op 层 host `_expand_flat` 物化 | kernel 内 `T.vmin` 原生标量/一维广播（可选内移） | §0.6 R3 |
| 片上驻留 | `T.alloc_fragment`（寄存器） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| grid 规模 | 任意（CUDA 调度器） | persistent `min(num_logical, 40)` | §0.6 R2 / §5.5 |
| JIT 装饰 | `@tilelang.jit(out_idx=[2])` | `@tilelang.jit(out_idx=[2], target="npuir")` | K1 适配 |
| 工厂参数 | `(N, dtype, has_min, has_max, output_dtype, is_fp8, threads=256, npt=8)` | `(N, dtype, has_min, has_max)`；callable(`block_size`) | wrapper K11（接口契约冻结） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/elementwise/vec_add_1d.py` | 高度相似（一维逐元素 + 动态尾块 + v-prefix + persistent） | `T.Kernel(n, is_npu=True)`、`T.alloc_ub`、`tail = T.min(...)` 尾块模式、runtime 边界 `T.copy`、v-prefix 用法 |
| `examples/ge/ge_fwd_kernel/DESIGN.md`、`examples/le/le_fwd_kernel/DESIGN.md` | 中度相似（同族迁移组织 + elementwise v-prefix，比较族） | §0/§1.6 章节组织、persistent 分核、L0 测试结构、wrapper 契约；**差异**：本算子无 bool 承载 / 无升域 / 无 vbrc 物化 |
| `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/` | 中度相似（elementwise v-prefix 单步） | elementwise 族 v-prefix 组织（mish 用 `T.min(T.max(...))`/clamp） |
| `testing/npuir/arith_ops/test_clamp_dev.py`、`test_clamp_vec_dev.py`、`test_clamp_exp.py`、`test_clamp_vec_exp.py` | 高度相似（**本算子 vclamp/clamp 的直接先例**） | `T.vclamp(src_ub, dst_ub, scalar|tensor, ...)` 可用性、`alloc_shared`(cbuf) vs `alloc_ub`/`alloc_fragment` 空间差异、`tilelang.compile(func, target="npuir")` 调用形态 |
| `testing/npuir/parallel_ops/test_min_max_parallel.py` | 高度相似（`T.min`/`T.max` 标量与张量路径） | `T.min(scalar, x_sh)` / `T.max(i,j)` 标量与循环变量路径（`docs/Tilelang.language/比较操作/T.min.md`） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype（kernel 承载） | 说明 |
|--------|-------|---------------------|------|
| `x` | `(N_total,)` | float16 / bfloat16 / float32 | Op 层已展平的 1-D 连续张量 |
| `hi`（max） | `(N_total,)`（内移广播后可为 `(max_numel,)`） | same_as(x) | 上界；同形契约下与 x 等长 |

> **dtype 覆盖（对照 manifest signature）**：manifest `ClampMaxFwdOp` inputs 声明 `float16 | bfloat16 | float32`——**全部**在 §1.6.0 R4 中给出原生路径（探针 Q-A 实测 bf16 OK），无遗漏。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(N_total,)` | **same_as(x)** | `out_idx=[2]`；Op 层 `reshape(out_shape)` 回广播形状 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | `(block_size,)` | 原生 dtype | UB | 数据连续段载入 |
| `hi_ub` | `(block_size,)` | 原生 dtype | UB | 上界连续段载入（同形路径）|
| `y_ub` | `(block_size,)` | 原生 dtype | UB | `T.vmin` 输出 |
| `hi1`（仅标量上界路径，§0.6 R3） | `(1,)` | 原生 dtype | UB | 标量上界单元素（`T.vmin` 标量可直接用，无需 buffer） |

> **无 fp32 upcast buffer、无 isnan buffer、无承载 dtype buffer**（§1.6.1 采纳项 1/2 消除）。buffer 复用：`x_ub`/`y_ub` 可原地（`T.vmin(x_ub, hi_ub, x_ub)`，`T.vmin.md` §2.3「无特殊限制」未禁别名）；`hi_ub` 活区间与 `y_ub` 重叠可合并（Stage 3 视分配）。

### 4.4 内存搬运路径

```
纯 Vector 单阶段（每 task）：
  GM[x]  ──T.copy(连续段)──> UB[x_ub] ┐
  GM[hi] ──T.copy(连续段)──> UB[hi_ub]┼─ T.vmin(原生 dtype, NaN 传播) ──> UB[y_ub] ──T.copy(连续段)──> GM[y]
（标量上界：GM[hi(1)] ──T.copy 1 元素──> UB[hi1] ──T.vmin(x_ub, hi1 广播)──> UB[y_ub]）
全程 UB 内，无 L1/L0 参与，无中间 GM 往返
```

### 4.5 UB 内存预算

block_size = 4096（fp16/bf16）/ 2048（fp32，wrapper 默认，§5.2）。活跃 buffer 仅 3 个（`x_ub`/`hi_ub`/`y_ub`），按 `×1.7` auto-multi-buffer 膨胀上界（`CONST-capacity-910B2C`）：

| dtype 路径 | 活跃 buffer（字节/元素） | 手工合计 | ×1.7 膨胀 | 目标 UB | 余量 |
|-----------|------------------------|----------|-----------|---------|------|
| fp16 / bf16 @4096 | 2+2+2 = **6** | 24 KB | ~41 KB | 192 KB | ✓ 充裕 |
| fp32 @2048 | 4+4+4 = **12** | 24 KB | ~41 KB | 192 KB | ✓ 充裕 |

> 膨胀系数引用 `CONST-capacity-910B2C`（auto-multi-buffer ~1.7x；另有「显式 alloc 结构 ~1.12x / 反例 0.981x」结构依赖记录，本设计取保守 1.7x）。block_size 上限（`floor(192KB / (3 × E × 1.7))`）：fp16/bf16 ≤ 19267、fp32 ≤ 9634——**wrapper 默认值（4096/2048）远低于上限**，UB 有大量余量。**Stage 4 可选优化**：增大 `block_size`（如 8192/16384）以减少 task 数、摊薄发射项（代价：尾块浪费率上升），交 Stage 4 扫（受 §9.1 R4 环境双态约束）。block_size 由 wrapper config 传入（`ClampTensorFwdKernel._default_block_size`：fp32 2048 / fp16,bf16 4096）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| 无 | 无 TIR 动态符号——工厂闭包按 `(N_total, dtype, has_min, has_max, block_size)` **编译期特化**（源同款模式，wrapper `lru_cache` 语义） | 不同 Op 实例触发一次 JIT 特化；kernel 边界全静态 |

> `num_logical`、`num_kernels`、`num_local` 全为 Python int / trace 期常量；kernel 内唯一运行时标量为 `tail`（1 次/task）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[2], target="npuir")
# out_idx=[2]：main(x, hi, y) 的 y（index 2）为输出，承接源 has_max-only 契约
#   （_clamp_max_fwd_kernels.py L202/L3197）
# target="npuir"：K1 适配（GPU 版无 target；example_elementwise_add.py L13 先例）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 仅逐元素截断（1 条 `T.vmin`）+ 连续段搬运（`T.copy`），无 matmul、无规约、无数据重排 → 全部计算落 Vector 单元，仅需 GM↔UB 搬运（decision-tree §2 纯 element-wise 分支）。

### 5.2 Block 划分

```python
block_size = 4096   # fp16/bf16（wrapper 默认，K11 = threads*npt = 512*8）
block_size = 2048   # fp32（wrapper 默认，K11 = 512*4）
# 选择理由（采纳 wrapper 默认，Stage 4 可扫）：
# ① 尾轴 32B 对齐：2048×4B=8KB / 4096×{2,4}B=8K/16K，均 32 整除 ✓
# ② 向量宽度整除：fp16/bf16 ×8、fp32 ×4 均整除 2048/4096 → 尾 lane 浪费 0%（尾部 task 除外）
# ③ UB 预算：3 buffer ≤ 41KB（×1.7）< 192KB ✓ 余量充裕（§4.5）
# ④ 发射项：N/block 个 task，每 task 1 条 vmin；发射项恒 < 流量项（§1.6.0）
# ⑤ 与同族迁移一致（源 threads*npt），接口契约冻结（wrapper config）
# ⑤ 广播路径（§0.6 R3）：合并维末维 L 与 block_size 关系决定 chunks；cnn-feat L=3136<4096 → chunks=1；
#    row-broadcast L=4096 → chunks=1；同形路径 R=1, L=N_total（由 chunks 分块）
L = N_total if same_shape else coalesced_shape[-1]     # 合并末维
R = 1 if same_shape else prod(coalesced_shape[:-1])    # 行数
chunks = ceildiv(L, block_size)
num_logical = R * chunks                     # 逻辑 task 数（行 × 块）
```

### 5.3 约束分析

| 约束 | 状态 | 说明 |
|------|------|------|
| 尾轴 32B 对齐 | ✓ | block×E 为 32 的整数倍 |
| 向量宽度整除 | ✓ | 2048/4096 对 fp16×8、bf16×8、fp32×4 均整除 |
| UB 容量 | ✓ | 3 buffer ≤ 41KB，远低于 192KB |
| 非整除 N | ✓ | 尾块 `tail = T.min(block_size, N − off)` 截断（§6.4） |
| 静态边界 | ✓ | 全部 tile 分解编译期常量，仅 `tail` 运行时单标量 |

### 5.4 注意事项

- 源 `has_max`-only 分支**无显式尾块判断**（依赖 `LegalizeSafeMemoryAccess`）；NPU 版显式 `tail` 切片更稳妥（与 `vec_add_1d.py` 一致）。
- `T.vmin` 整 buffer 操作在尾块会对 `[tail:]` 段一并计算（值无害），仅 `[0:tail]` 写回。
- 标量上界路径不需 `hi` 张量（`T.vmin(x_ub, scalar, ...)`），但 wrapper 契约（同形 `hi`）下仍以张量传入——内移 §0.6 R3 为可选优化。

### 5.5 分核策略（物理核数适配）⭐

> 三要素（core-split-strategy.md §1；依据 docs/开发指南.md §3.3）。

**① 逻辑核数计算**：展平一维后 `num_logical = ceildiv(N_total, block_size)`：

| workload | N_total | 结构（R×L, chunks） | block_size | num_logical |
|----------|---------|--------------------|-----------|-------------|
| smoke `[1024]` | 1,024 | R=1, L=1024, chunks=1 | 2048(fp32)/4096 | 1 |
| elementwise-16M `[4096,4096]^[4096,4096]` | 16,777,216 | R=1, L=16.78M, chunks=4096 | 4096 | 4,096 |
| cnn-feat-broadcast `[16,256,56,56]^[256,1,1]` | 12,845,056 | R=4096, L=3136, chunks=1 | 4096 | 4,096 |
| row-broadcast `[4096,4096]^[1,4096]` | 16,777,216 | R=4096, L=4096, chunks=1 | 4096 | 4,096 |

**② 物理核数依据**（实查，禁止假设值）：

```python
from tilelang.utils.npu import NPUUtils   # 或等价导入
n = NPUUtils.get().get_aicore_num()       # → 20（本机实查，2026-10-13）
vector_cores = n * 2                       # 纯 Vector 算子核数翻倍 → 40
```

**实查记录**：查询代码如上，命令 `python3 -c "from tilelang.utils.npu import NPUUtils; print(NPUUtils.get().get_aicore_num())"` 返回 **20**（纯 Vector 核数 = 20×2 = **40**）；参考 `CONST-aicore-910B2C`（910B2C = 24 AI Core，48 Vector）与本机不同源，流量项已按 40/48 折算并标注（§1.6.0）。

**③ 规模判定与分核方案**：`num_logical`（1–4096）**超过**物理核数 40（除 smoke 1 例外）→ 按 **极大规模** 处置：固定启动内核数 = 物理核数（`num_kernels = min(num_logical, 40)` = 40），核内 `T.serial(num_local)` 串行处理多个逻辑块任务，`num_local = T.ceildiv(num_logical, num_kernels)`（**编译期静态值**）：

| workload | num_kernels | num_local（静态） | 每核摊薄 |
|----------|-------------|-------------------|----------|
| smoke `[1024]` | 1 | 1 | 无需适配（逻辑核 1 ≤ 40，退化 1 核） |
| elementwise-16M `[4096,4096]` | 40 | 103 | 40×103 = 4120 ≥ 4096（尾核空转 <1%） |
| cnn-feat-broadcast `[16,256,56,56]^[256,1,1]` | 40 | 103 | 40×103 = 4120 ≥ 4096 |
| row-broadcast `[4096,4096]^[1,4096]` | 40 | 103 | 40×103 = 4120 ≥ 4096 |

> 分核方案收敛：persistent 结构把 4096 个逻辑 task 常驻 ≤40 个 Vector 核，摊薄核启动开销，避免超发被串行调度。`num_local` 为编译期静态常量，满足 core-split-strategy.md §1 要素③。Stage 4 可评估 `block_size` 增大以调整 `num_logical`（§4.5；若 block_size 增至 8192，row-broadcast 的 L=4096 变 `chunks=1` 不变、同形路径 num_logical 减半）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 循环层级 | 类型 | 作用域 | 边界 | 说明 |
|----------|------|--------|------|------|
| 核间 | `T.Kernel(num_kernels, is_npu=True)` | 40 个 Vector 核 | 静态 | persistent 分核（§5.5） |
| 核内 task 推进 | `T.serial(num_local)` | 每核 | **静态**（`num_local` 编译期常量） | grid-stride：`task = i*num_kernels + cid` |
| 逐元素计算 | **无循环** | — | — | 由 `T.vmin` 一条向量指令覆盖整块（§1.6.2 #1） |
| 尾块 | `if task < num_logical` + `tail = T.min(...)` | 每 task | 运行时单标量 | 尾块截断 |

### 6.2 循环伪代码

见 §3.3。核内只有一个 `T.serial(num_local)` 调度循环（循环体 = 一次连续段载入 ×2 + 一条 `T.vmin` + 一次连续段写回）；**无任何逐元素标量循环**（与 §1.6.2 结论一致）。

### 6.3 流水线优化

单阶段，无 `T.Pipelined`。auto multi-buffer（默认开启）可将 `T.copy`（MTE2）与 `T.vmin`（VEC）流水重叠——`CONST-copy-floor-method` 记录 ≥16M 档计算链可被隐藏于 MTE2 窗口，本算子每 task 1 条向量 op（计算量极低），MTE2 主导，预期重叠充分。

### 6.4 尾块处理

`tail = T.min(block_size, N_total − off)`；`T.copy` 用 runtime 边界切片 `x[off : off+tail]`（`vec_add_1d.py` L34–L38 已验证模式）；`T.vmin` 整 buffer 计算，仅 `[0:tail]` 写回。**不采用 host padding**（违反 ascend-constraints.md §4）。

---

## 7. 同步策略

### 7.1 同步模式

**Developer 自动同步**（编译器插入 `T.copy`↔`T.vmin` 的依赖同步）。每 task 内部依赖链 `GM→UB 载入 → vmin → UB→GM 写回`，无跨 task / 跨核依赖。

### 7.2 同步点说明

| 同步点 | 类型 | 说明 |
|--------|------|------|
| `T.copy(x/hi → UB)` → `T.vmin` | 自动（数据依赖） | 载入完成后方可计算 |
| `T.vmin` → `T.copy(y_ub → GM)` | 自动（数据依赖） | 计算完成后方可写回 |
| 核间 | 无需 | 各核独立处理互不相交的 task，零跨核同步 |

> `T.vmin.md` §2.3「特殊限制说明：无」；`T.vmin` 输出到独立 buffer 时无额外同步要求。**无需手动 `sync_block_set/wait`**（非 MixCV）。

### 7.3 pass_configs 配置

无特殊 pass_configs（默认 `--enable-auto-multi-buffer=true`、`--disable-hivm-auto-inject-sync=false`）；Developer 模式由 wrapper `ascend_mode="Developer"` 固定。

---

## 8. 验证方案

### 8.1 Golden 函数

> Golden 以 §0.1 语义为**唯一依据**（`torch.clamp_max` 广播直算），**不复刻** §0.6 的 NPU 算法（不模拟 `T.vmin` / 分核 / 尾块），保证与源算子语义独立对齐。优先移植 GPU 仓参考实现（`tests/ops/test_special_elementwise_conformance.py::test_clamp_max_tensor` L258–L267）。

```python
import torch

def clamp_max_golden(input: torch.Tensor, max: torch.Tensor) -> torch.Tensor:
    """Reference: torch.clamp_max with NumPy/PyTorch broadcasting.

    Semantics (§0.1): y = min(input, max) element-wise; NaN in input or max
    propagates (NaN -> NaN); output dtype == input dtype; output shape ==
    broadcast_shapes(input.shape, max.shape).
    """
    return torch.clamp_max(input, max)
```

### 8.2 精度标准

`min` 是**操作数选择**（非算术），同 dtype 下 golden 与 kernel 结果**应逐位一致**。判定标准（对齐 manifest conformance 测试）：

| dtype | atol | rtol | equal_nan | 说明 |
|-------|------|------|-----------|------|
| float16 | 1e-3 | 1e-3 | True | `torch.clamp_max` 同 dtype 直算（预期 bit-exact，容差为余量） |
| bfloat16 | 1.6e-2 | 1.6e-2 | True | 同上 |
| float32 | 1e-5 | 1e-5 | True | 同上 |

`equal_nan=True` 是**必需**的（NaN 传播契约，§0.1）；对 NaN 位置额外断言「输出 NaN 模式与 golden 一致」。

### 8.3 L0 门槛测试计划

| 编号 | 用例 | 覆盖点 |
|------|------|--------|
| L0-1 | 同形 `[4096,4096]^[4096,4096]`，fp16/bf16/fp32 | 主路径 + 全 dtype（含 **bf16**，复核 §9.1 R1） |
| L0-2 | 广播 manifest workload：`cnn-feat-broadcast [16,256,56,56]^[256,1,1]`、`row-broadcast [4096,4096]^[1,4096]`（fp16/bf16/fp32）；另 `[4,8]^[4,1]`、`[4,8]^()`（0-dim 上界） | 上界广播 / 标量路径（§0.6 R3 行分解；`test_clamp_max_tensor` 用例） |
| L0-3 | NaN 传播：`x=[NaN,...]` / `hi=[...,NaN]`，fp16/fp32，`equal_nan=True` | **NaN 语义**（§1.6.1 Item 1；`test_clamp_max_nan_propagation` 用例） |
| L0-4 | 角点：±Inf、±0、max/-max、subnormal | IEEE 边界（探针 Q-B 已覆盖 NaN/±Inf；subnormal 待 L0） |
| L0-5 | 非整除 N（如 N=4097，block_size=4096） | 尾块截断（§6.4） |
| L0-6 | 标量上界路径（若 §0.6 R3 内移实现） | `T.vmin(x_ub, scalar, y_ub)`（探针 Q-C1 同款） |
| L0-7 | Op 层一致性：`ClampMaxFwdOp.forward` 输出 shape == broadcast_shapes | I/O 契约（§0.2）|
| L0-8 | `eval_roofline` 返回 `(N_total, 3*N_total*elem_bytes)` | manifest roofline 契约 |

> 分层套件 L1/L2/Boundary 的完整枚举交由 `tilelang-op-develop`；本节仅列 L0 门槛。**L0-1 的 bf16 用例为强制**（文档 bf16「×」与探针实测矛盾——以 L0 复测定案）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| # | 风险 | 依据 / 缓解 |
|---|------|------------|
| R1 | **`T.vmin` 文档 dtype 矩阵标 bf16「×」，与探针实测矛盾** | `docs/Tilelang.language/比较操作/T.vmin.md` §2.2.1 标 bf16 ×；探针 Q-A 实测 bf16 同形 bit-correct（`design_probe_vmin.py`）。**以实测为准**；L0-1 强制复测。**回退方案**：若某工具链真不支持 bf16，走 `T.vcast(bf16→f32)` → fp32 `T.vmin` → `T.vcast(f32→bf16)`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1；对 min 无损，§1.6.1 Item 2 的逆命题）——牺牲 2 条 op/task |
| R2 | **`VMinOp` NaN 传播为未文档化假设**（仅探针实测） | 探针 Q-B（fp16/fp32 NaN/±Inf 角点 match）+ `PL-1.11`（vmax/vmin 钳位不清除 NaN）+ `arith::MinimumFOp`（float NaN 传播，`T.min.md` §3）。**L0-3 强制 NaN 角点**；回退：显式 `isnan` 守卫（需 NPU 侧 NaN 判定 API，未文档化——优先依赖 L0 复测） |
| R3 | **Op 层 host `_expand_flat` 广播物化 vs ascend-constraints.md §4** | 同形路径合规（纯视图）；广播 / 标量路径触发 host `.contiguous()` 物化。缓解：§0.6 R3 内移 kernel（`T.vmin` 原生广播/标量）。若 wrapper 签名保持冻结（`(N,dtype,has_min,has_max)`），则广播路径保留 host 物化——需与 Op 层契约方确认，属 Stage 2/3 升级项 |
| R4 | 发射项 / block_size 调优受环境双态影响 | 本算子每 task 1 op、发射项非主导；若 Stage 4 调 `block_size`，须遵守 `VP-2026-0001`（run 间 ±2–3% 环境双态，<3% 结论须多 run） |
| R5 | 尾块 `[tail:]` 段 vector op 计算垃圾值 | 仅 `[0:tail]` 写回，垃圾不落 GM（`vec_add_1d.py` 官方先例）；无正确性影响 |
| R6 | subnormal / 极端指数值未经 NPU 实测 | 探针未覆盖 subnormal；L0-4 覆盖。`min` 为操作数选择，理论无舍入（fp16/bf16 subnormal 在 fp32 精确）——低风险 |

### 9.2 常见错误

- **探针 / 验证脚本的 `.npu()` 陷阱**（本次实测踩坑）：`torch.zeros(...)`（CPU）→ kernel 调用时 `.npu()` 产生**新** device 张量，写回不反映到原 CPU 张量；须先建 device 张量再调用、对 device 张量 `.cpu()` 比较（否则误判"全零"）。
- **`T.alloc_shared` ≠ UB**：本机 IR 实测 `T.alloc_shared` → `#hivm.address_space<cbuf>`（L1），`T.copy(cbuf → GM)` 报 `'hivm.hir.store' op only support store ub to gm`；`T.alloc_fragment` 在双张量操作数下被分配到 `<zero>` 空间同样报错。**纯 Vector elementwise 须用 `T.alloc_ub`**（`vec_add_1d.py` / `T.vmin.md` §2.4 同款）。
- **`T.vmin`/`T.vmax`/`T.vclamp` 是 `npuir_min/max/clamp` 别名**（`tilelang/language/__init__.py` L77–L79/L136）——二者等价，新代码优先 v-prefix（AGENTS.md API 约定）。

### 9.3 特殊场景处理

- **0-dim / 标量上界**：Op 层 `torch.broadcast_shapes` 把 0-dim 归一为 `(1,)`；kernel 侧 `T.vmin` 原生标量（§0.6 R3）。
- **`x` 与 `hi` 完全相同 shape**：退化为纯逐元素，无广播开销（manifest workload 主路径）。
- **空张量**：`N_total` 由 Op 层 `prod(out_shape)`，空维度由上层拦截。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/clamp_max/_make_clamp_tensor_kernel/
├── DESIGN.md                              # 本文件
├── verify_equiv.py                        # D-1 等价性机器验证（14/14 EQUIV_PASS）
├── RETROSPECTIVE.md                       # Stage 1 复盘
└── history_version/
    └── design_probe_vmin.py               # D-3 设计期探针（vmin dtype/NaN/标量/广播）
```

### 10.2 文件清单

| 文件 | 说明 | 状态 |
|------|------|------|
| `DESIGN.md` | 算子设计文档（§0 迁移分析 + §1–§11） | ✅ |
| `verify_equiv.py` | 等价性机器验证脚本（E1–E4） | ✅ 已执行 EQUIV_PASS |
| `history_version/design_probe_vmin.py` | D-3 探针（Q-A/B/C1/C2） | ✅ 已执行全 OK |
| `RETROSPECTIVE.md` | Stage 1 复盘 | ✅ |

### 10.3 命名规范

- 算子目录 `clamp_max`（`op_slug`）；kernel 目录 `_make_clamp_tensor_kernel`（提取函数名）；实现文件由 `tilelang-op-develop` 产出（`_make_clamp_tensor_kernel.py` 或 wrapper 指向的 `clamp_max_kernel/`）。

### 10.4 实现顺序

1. 在 `examples/elementwise/vec_add_1d.py` 骨架上把 `vadd` 替换为 `vmin`（同形路径），验证 fp16/bf16/fp32 L0-1。
2. 加 persistent 分核（§5.5）+ 尾块 `tail`。
3. 加广播 / 标量上界路径（§0.6 R3，若 wrapper 允许）。
4. 跑 L0 门槛套件（§8.3），重点 L0-1（bf16）/L0-3（NaN）。

---

## 11. 设计核心与 GPU→NPU 关键重设计（下游报告提炼用）

1. **逐元素截断链 → 单条向量 `T.vmin`**：源 `T.Parallel(threads, npt)` 内逐元素标量 `T.min`（含 fp32 升域 + 2 条 `isnan` 守卫 + 回舍，共 **8 标量 op/元素**）替换为 **一条向量 `T.vmin(x_ub, hi_ub, y_ub)`**（原生 dtype，NaN 传播）——每 task 向量 op 数 **8→1**（§0.6 R1，§1.6.1）。
2. **消除源的两条冗余计算**（`verify_equiv.py` 机器验证 + 探针实测）：① **fp32 中转消除**——`min` 无累加，结果必为操作数之一，原生 dtype bit-exact（Item 2，E2 0 违反）；② **NaN 显式守卫消除**——NPU `VMinOp` 原生传播 NaN，与 `torch.clamp_max` 逐点一致（Item 1，探针 Q-B + E1 0 违反）。
3. **并行结构重设计**：CUDA `grid×block` 两级并行 + `threads/npt` → 一维 `T.Kernel(num_kernels, is_npu=True)` + `T.alloc_ub` + persistent 分核（`num_kernels = min(num_logical, NPUUtils.get().get_aicore_num()*2)`，本机 20⇒40；核内 `T.serial` 静态边界）——摊薄核启动、消除超发（§0.6 R2，§5.5）。
4. **上界广播 / 标量内移**：借 `T.vmin` 原生一维广播与标量 src2（探针 Q-C1/C2），把 Op 层 host `_expand_flat` 的广播物化内移 kernel，规避 ascend-constraints.md §4 的 host 物化约束（§0.6 R3；`verify_equiv.py` E3/E4 0 违反）。
