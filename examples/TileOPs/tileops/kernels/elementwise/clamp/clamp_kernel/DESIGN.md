# _make_clamp_tensor_kernel 算子设计文档（迁移）

> **项目**：`clamp` ｜ **算子目录**：`examples/clamp/_make_clamp_tensor_kernel/` ｜ **编程模式**：Developer ｜ **target**：`npuir`
>
> **任务性质**：迁移（migration-harness）。把 GPU TileLang 工厂函数 `_make_clamp_tensor_kernel`（Tensor-bound `y = min(max(x, lo), hi)`，非 fp8 分支**已在 fp32 域** clamp 后单次舍回，并带显式 NaN 守卫）重设计为 NPU（`target="npuir"`）实现。
>
> **本版核心决策**：① **计算域 = fp32 中转**（**保留源既有行为**并对齐契约 golden `torch.clamp(x.float(), lo.float(), hi.float()).to(dtype)`）；bf16 因 `T.vclamp`/`T.vmin`/`T.vmax` dtype 矩阵不含 bf16 而**强制** fp32 中转，fp16/fp32 承接源同域行为（PL-1.27 ②：同域 = 保留并对齐，非「新增优化」）。② **算法级采纳优化 = 源码显式 NaN 守卫消除**（3×`isnan` + 3×`if_then_else` → 0）：源守卫是 **CUDA 语义补丁**（`T.max`/`T.min` 走 `fmaxf`/`fminf`，不传播 NaN），而 NPU `VMaxOp`/`VMinOp` **原生传播 NaN**——本设计以**设备探针实测**（`history_version/design_probe_clamp_nan.py`，PROBE PASS）确认张量边界 `vclamp`/`vmin`/`vmax` 的 NaN 传播与 `lo>hi → hi` 语义，故守卫在 NPU 逐位冗余；`verify_equiv.py` 23/23 `EQUIV_PASS`。③ **广播处理 = host 侧物化（Op 层 `_expand_flat` 既有契约，保留）+ kernel 扁平一维**（`§0.6 R4`：核内 2D+vbrc 候选经量化评估否决）。④ **并行/内存重设计**：GPU `fragment + block×thread` → 一维 `T.Kernel(40, is_npu=True)` + 核内 `T.serial` 静态边界网格跨步 + UB 整块 `T.copy` + 整 buffer 向量 op。
>
> **源优化意图承接**：源 #1 `register_copy`（fragment load→compute→store）→ NPU `GM→UB→GM` 整块 `T.copy` + 整 buffer v-op；源 #2 block×thread 两级并行 → 一维 `T.Kernel(40, is_npu=True)` + 核内网格跨步；源 #3 fp32 计算域 → **保留**（契约对齐 + bf16 强制）；源 #4 `npt` 对齐 → K11 `block_size` 折叠 + 32B 对齐复算；源 #5 NaN 守卫 → **舍去**（NPU 原生语义覆盖，§1.6.1 #1）；源 #6 `has_min`/`has_max` 工厂分支特化 → 保留（三个 kernel 变体）；源 #7 `lru_cache` 特化 → 保留（wrapper 惰性构建 + 缓存）。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 tilelang-op-design skill 的 [references/migration-analysis.md](../../../.agents/skills/tilelang-op-design/references/migration-analysis.md)）：先彻底读懂源算子（0.1–0.4），再做算法调研（Phase R，结论见 §1.6.0），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。

**源算子路径（源真相）**：`/home/developer/TileOPs/tileops/kernels/elementwise.py`

- 本任务唯一目标：工厂 `_make_clamp_tensor_kernel(N, dtype, has_min, has_max, output_dtype=None, is_fp8=False, threads=256, npt=8)`（**L3018–L3224**）+ 类 `ClampTensorFwdKernel(ParametricUnaryKernel)`（**L3227–L3265**）。
- **变体辨识（避免误取）**：`_make_clamp_kernel`（L2938，scalar 边界 `ClampFwdKernel`）**不是**本任务目标；`ClampMinFwdOp` / `ClampMaxFwdOp` / fp8 分支亦不在迁移范围。本设计只覆盖 **Tensor-bound**（`lo`/`hi` 为张量）三输入广播语义。
- 抽取件（Stage 0 脚手架产物，接口契约冻结基准）：`examples/TileOPs/tileops/kernels/elementwise/clamp/_clamp_fwd_kernels.py`（L13–L219，与源 L3018–L3224 逐字一致）。
- NPU 集成 wrapper（已冻结，本 kernel 的唯一调用方）：`examples/TileOPs/tileops/kernels/elementwise/clamp/clamp.py`（`ClampTensorFwdKernel`）；Op 层：`examples/TileOPs/tileops/ops/elementwise/clamp.py`（`ClampFwdOp`）。

### 0.1 源算子语义（做什么）

- **数学语义**：逐元素三输入 clamp（Tensor-bound `torch.clamp`）
  $$y[i] = \min\big(\max(x[i], lo[i]),\ hi[i]\big),\quad i \in [0, N)$$
  `lo` / `hi` 为**同 dtype 张量**（非标量），与 `x` 广播；`has_min` / `has_max` 两个构造期布尔量在三种变体间选择（both / min-only / max-only）。`has_min = has_max = False` 在构造期抛 `ValueError`（源 L3041–L3044 / 类 L3241–L3244；契约测试 `test_clamp_both_none_rejected`）。
- **I/O 契约**：三输入一输出，**全部展平为连续一维 `(N,)`**（`N = N_total = prod(broadcast_shapes(input, min, max))`）；三输入同 dtype；输出 dtype = 输入 dtype（`output_dtype` 缺省回退 `dtype`，本算子恒等）。fp8 不在搬运范围（manifest/`SUPPORTED_DTYPES = (fp16, bf16, fp32)`，构造期拒绝非浮点与 fp8）。
- **规约语义**：无规约、无跨元素依赖（每个输出元素只依赖同下标的三输入元素），**不存在累加顺序问题**——并行组织变化不改变数值结果。
- **dtype 语义（关键：计算域 = fp32）**：源**非 fp8 分支**为
  ```python
  x32  = T.cast(x_reg[k],  "float32")     # L3150（x 升 fp32）
  lo32 = T.cast(lo_reg[k], "float32")     # L3151
  hi32 = T.cast(hi_reg[k], "float32")     # L3152
  r    = T.min(T.max(x32, lo32), hi32)    # L3153–L3154（fp32 域 clamp）
  x_reg[k] = T.Cast(dtype, r)             # L3160（单次舍回原 dtype）
  ```
  即 **fp32 中转**：三操作数升 fp32、clamp、结果单次舍回；中间**无逐步 dtype 域舍入**。契约测试基准同域：`ClampTensorTest.ref_program = torch.clamp(x.float(), mn.float(), mx.float()).to(x.dtype)`（`examples/TileOPs/tests/ops/test_special_elementwise.py` L115–L116）。→ 源与非 fp8 契约 golden **同域**（PL-1.27 ②：同域 = 保留并对齐）。
- **边界语义**：
  - **NaN 传播**（torch 契约）：`x`、`lo`、`hi` 任一为 NaN → 输出 NaN。源以**显式守卫**实现（L3157–L3159：`r = if_then_else(isnan(hi), hi, r)` → `isnan(lo)` → `isnan(x)`），原因见源注释 L3034–L3039：「`T.max`/`T.min` on CUDA do not propagate NaN by themselves (they return the non-NaN operand)」——CUDA 的 `T.max`/`T.min` 即 `fmaxf`/`fminf`（IEEE `maxNum`，非传播）。契约测试 `test_clamp_tensor_nan_propagation` 以 `atol=rtol=0.0, equal_nan=True` **逐位断言** NaN 语义。
  - **`lo > hi`**：`min(max(x, lo), hi)` 在 `lo > hi` 时恒返回 `hi`（PyTorch 语义）——**顺序敏感**，交换 min/max 顺序即非等价（§1.6.1 非门禁诊断实证 6.3 万点违反）。契约测试 workload `(4,8)/(1,8)/(4,1)` 中 `mn = randn-0.5`、`mx = randn+0.5` 会实际触发 `lo > hi`。
  - **±inf / ±0 / 次正规**：IEEE 语义，clamp 是投影（选择三值之一），除 fp32↔原 dtype 单次舍回外不引入舍入。
  - **0-dim / 标量边界张量**：`min_shape == ()`（契约测试 `(4,8), (), ()`）——`expand` 到输出 shape 后与全 shape 张量等价，**kernel 视角无差异**。
  - **空张量 / 极小 shape**：`out_shape == ()` 时 Op 层以 `(1,)` 兜底（`clamp.py` L131/L136）；`N = 0` 由上层拦截，kernel 不单独处理。
- **host 侧语义**：三输入**预广播 → 展平**为 `(N,)` 连续张量、输出 `view` 回 `out_shape`，由 Op 层 `ClampFwdOp._expand_flat`（`t.expand(out_shape).contiguous().view(-1)`）与 `_eager_forward` 承接——**属于算子既有契约（O6「preserved unchanged」）**，kernel 无感知；详见 §0.6 R4 与 §9.1 R-3。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（`torch.clamp(x.float(), lo.float(), hi.float()).to(dtype)`，与源仓测试基准一致），**不复刻** NPU 算法（不复用 vclamp/fp32 中转/persistent 结构）——保证验证独立性。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `(N,)` | float16 / bfloat16 / float32 | 数据张量（Op 层已预广播展平物化），`N` = broadcast 后元素总数 |
| `lo` | `(N,)` | same_as(x) | 下界张量；**仅当 `has_min` 时存在**（源 main 签名随分支变化） |
| `hi` | `(N,)` | same_as(x) | 上界张量；**仅当 `has_max` 时存在** |
| `y` | `(N,)` | same_as(x) | 输出；**源码解读确认输出为一维 `(N,)`**（`main(... y: T.Tensor((N,), dtype))`，源 L3139/L3173/L3203），Op 层再 `view(out_shape)` 还原——`source_output_shape = broadcast_shapes(input, min, max)` |

**输出 shape 依据**：源 L3139 `y: T.Tensor((N,), dtype)`；`ClampFwdOp._eager_forward`（`clamp.py` L136）`result.view(self.out_shape if self.out_shape else ())` 佐证 kernel 输出为一维展平视图，无转置/布局变换。`out_idx` 随分支：both `[3]`、min-only `[2]`、max-only `[2]`（源 L3132 / L3167 / L3197）。

### 0.3 实现算法解读（怎么算）

源文件：`examples/TileOPs/tileops/kernels/elementwise/clamp/_clamp_fwd_kernels.py`（L13–L219；Stage 0 从 GPU 仓 `/home/developer/TileOPs/tileops/kernels/elementwise.py` L3018–L3224 提取，Pattern B 工厂闭包）。

**计算步骤分解**（覆盖源码全部计算语句，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1（host，Op 层） | 三输入 `expand(out_shape) → contiguous() → view(-1)` | x / lo / hi 任意 broadcast shape | 三个 `(N,)` 连续张量 | 预广播展平（Op 契约，wrapper/Op 承接） |
| 2 | `block_size = threads × npt`；grid = `T.ceildiv(N, block_size)` | threads_arg, npt_arg | CUDA grid 配置 | 任务划分 |
| 3 | `T.copy(x[bx·bs:(bx+1)·bs], x_reg)`（×2 或 ×3，按 has_min/has_max） | GM x / lo / hi | fragment `x_reg`/`lo_reg`/`hi_reg` | 数据搬入（连续段 coalesced） |
| 4 | `for i,j in T.Parallel(threads, npt): k=i·npt+j; x32=cast(x_reg[k],f32); lo32=cast(lo_reg[k],f32); hi32=cast(hi_reg[k],f32); r=min(max(x32,lo32),hi32); r=where(isnan(hi32),hi32,r); r=where(isnan(lo32),lo32,r); r=where(isnan(x32),x32,r); x_reg[k]=T.Cast(dtype,r)` | x_reg / lo_reg / hi_reg | x_reg（**原地覆写**） | 核心公式 + fp32 中转 + NaN 守卫 |
| 5 | `T.copy(x_reg, y[bx·bs:(bx+1)·bs])` | x_reg | GM y | 数据搬出 |
| 6（host，Op 层） | `result.view(out_shape)` | `(N,)` 输出 | broadcast shape | 后处理（Op 契约） |

**分支结构**（源共 6 个 kernel 变体，字段矩阵 = {is_fp8} × {min+max, min-only, max-only}；本迁移只覆盖 `is_fp8 = False` 的 3 个）：三者步骤骨架相同，差异仅在搬入 buffer 数与 `out_idx`；`has_min`/`has_max` 在**构造期**决定编译哪个变体（编译期特化，无运行期分支）。

**数据流与内存访问模式**（GPU 视角）：

```
GM[x]  ──T.copy(连续 block_size 段)──> fragment x_reg  ┐
GM[lo] ──T.copy──────────────────────> fragment lo_reg ┼─ 逐线程标量：升 fp32 → min(max) → isnan 守卫 ──> x_reg（原地）──T.copy──> GM[y]
GM[hi] ──T.copy──────────────────────> fragment hi_reg ┘
（无 shared memory 中转、无 bank conflict 处理、无异步流水；单阶段 load→compute→store）
```

**循环与并行结构**：CUDA grid（`ceildiv(N, block_size)` 个 block）× block 内 `T.Parallel(threads, npt)` 二维线程并行；每线程处理 `npt` 个连续元素（`k = i·npt + j` 的线程→元素映射）。工厂默认 `threads=256, npt=8`（block_size=2048）；GPU Kernel 类实际 `threads=512`，`npt=4`（fp32）/`8`（fp16/bf16）→ block_size = 2048 / 4096（NPU wrapper `_DEFAULT_BLOCK_SIZE` 已按此折叠）。

**host 侧逻辑**：`@tilelang.jit(out_idx=[…])` 工厂闭包按 `(N, dtype, has_min, has_max, output_dtype, is_fp8, threads, npt)` 特化编译（GPU 仓叠加 `functools.lru_cache(maxsize=32)`）；`del` 无关参数；`out_dtype = output_dtype or dtype`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | `register_copy`（fragment `alloc_fragment` + load→compute→store 单阶段） | 数据驻留寄存器、消除重复 GM 访问 | 三输入 tile 装入 fragment，算术全程寄存器内完成（`x_reg[k]` 原地覆写为输出） | GPU 大寄存器堆 + fragment 机制 | 硬件强相关 |
| 2 | 连续段 coalesced 搬运 | 访存带宽 | `T.copy` 按 block_size 连续段读写，尾轴连续 → CUDA memory coalescing | CUDA coalescing 硬件 | 硬件强相关 |
| 3 | thread × npt 两级并行 + 每线程多元素（ILP） | 并行度与指令级并行 | `T.Parallel(threads, npt)`，每线程连续处理 npt 个元素 | CUDA block/thread 线程模型 | 硬件强相关 |
| 4 | fp32 计算域（`cast→min/max→cast` 单次往返） | 对齐 torch 比较/舍回语义、免逐步舍入 | 三操作数升 fp32，clamp 后单次舍回 | GPU 支持全 dtype fragment 算术 | 语义层（契约对齐；NPU bf16 需 API 强制中转） |
| 5 | 显式 NaN 守卫（`isnan` + `if_then_else` ×3） | **正确性**（复现 torch NaN 传播） | CUDA `fmaxf/fminf` 不传播 NaN → 补 3 步选择性覆盖 | CUDA `T.max/T.min` 的 maxNum 语义 | **语义补丁（源硬件特有）**，非性能优化 |
| 6 | `has_min`/`has_max` 工厂分支（3 个 kernel 变体） | 免运行期分支、免多余输入 | 构造期选定变体，含单边界变体（省 1 输入 + 1 段搬入） | 无（编译期特化） | 可移植 |
| 7 | 一维展平（Op 层物化） | 消除多维索引开销、kernel 极简 | 三输入预广播展平为一维连续 | 无（通用） | 可移植 |
| 8 | 工厂闭包 + `lru_cache` 按 `(N, dtype, …)` 编译特化 | 摊薄 JIT 编译开销 | 每个参数组合只编译一次 | 无（host 层） | 可移植 |

> 源码未使用 SMEM tiling / warp shuffle / 异步流水 / swizzle / persistent kernel——本表已穷尽源文件全部显式优化结构（对照 migration-analysis.md §4.1 清单逐项核对；对照 `clamp` 家族源码 L2938–L3265 全段，含 scalar/fp8 变体确认无遗漏）。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 Phase R 调研结论 §1.6.0）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 数学语义 `min(max(x,lo),hi)` | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.4 规则 1）；§1.6.0 R1 确认为最优公式形态（含顺序敏感性） |
| I/O 契约（`(N,)`×3 进 `(N,)` 出、同 dtype、`out_idx` 随分支） | 语义 | 无 | — | **保留** | 输出 shape 一维由 §0.2 源码解读确认；`out_idx` 与 wrapper 调用点逐分支对齐（§3.2/§9.1 R-5） |
| dtype 契约（fp16/bf16/fp32） | 语义 | 无 | — | **保留** | wrapper `SUPPORTED_DTYPES` 沿用；bf16 **计算路径**因 API 限制需 fp32 中转（→ 重设计项 R2，契约本身不变） |
| 边界语义（NaN 传播 / `lo>hi→hi` / ±inf / ±0） | 语义 | 无 | — | **保留** | 设备 `VMaxOp`/`VMinOp` 原生传播 NaN 且 `min(max)` 顺序不变（**设备探针实测**，§0.6 R5）；golden 按 §0.1 语义实现 |
| host 预广播/展平/回 view | 语义（Op 契约） | 无 | — | **保留** | `ClampFwdOp._expand_flat` / `_eager_forward` 已承接（O6 preserved）；broadcast 代价与 host 约束核对见 §0.6 R4 + §9.1 R-3 |
| 一维 grid `ceildiv(N, block_size)` | 算法 | 弱（CUDA grid） | 有（`T.Kernel(一维, is_npu=True)`） | **保留** | 一维 Kernel 为本项目支持形态（ascend-constraints.md §1 第 1 行；`examples/elementwise/vec_add_1d.py` / `examples/elementwise/vec_add_2d_dynamic_shape.py` 先例）；核数按分核策略适配（→ R3） |
| `has_min`/`has_max` 工厂分支特化 | 算法（host） | 无 | — | **保留** | NPU 工厂保持三变体（§3.3 伪代码），契约/精度路径与源一致 |
| 工厂闭包 `lru_cache` 特化 | 算法（host） | 无 | — | **保留** | NPU wrapper `forward` 已实现按 block_size 惰性构建 + 缓存（`clamp.py` L201–L207） |
| fragment 驻留（#1） | 优化 | GPU 寄存器堆 | 有（UB） | **等价替换** | `T.alloc_fragment` → `T.alloc_ub`（migration-analysis.md §5.3「shared memory→UB」+「register 累加→UB 中间缓冲」行；v-prefix API 要求操作数在 UB：`docs/Tilelang.language/数学操作/T.vadd.md` §2.3 第 1 条）；意图承接：数据驻留片上、单阶段 load→compute→store |
| 连续段 coalesced 搬运（#2） | 优化 | CUDA coalescing | 有（向量化 copy） | **等价替换** | `T.copy` GM↔UB 连续段搬运（`docs/Tilelang.language/内存操作/T.copy.md`；`vec_add_1d.py` 同款模式）；意图承接：带宽利用 |
| thread × npt CUDA 两级并行（#3） | 优化 | CUDA 线程模型 | 无直接等价 | **重新设计** | → §0.6 重设计项 R1（一维 grid + Vector 128bit 向量化，npt 的「每线程多元素」意图由向量 lane 承接） |
| fp32 计算域（#4） | 语义/优化 | GPU 全 dtype fragment 算术 | fp16/fp32 有（`vclamp` ✓）；**bf16 无**（dtype 矩阵 bf16 ×） | **等价替换**（bf16 路径）+ **保留**（fp16/fp32） | → §0.6 重设计项 R2：bf16 经 `T.vcast` 升 fp32 clamp、舍回 bf16；依据 `docs/Tilelang.language/比较操作/T.vclamp.md` §2.2.1（fp16 √ / fp32 √ / bf16 ×）、`T.vmin.md` / `T.vmax.md` 同 |
| 显式 NaN 守卫（#5） | 语义补丁 | CUDA `T.max/T.min`（maxNum 不传播 NaN） | 有（NPU 原生传播） | **舍弃**（NPU 上冗余） | → §0.5 说明 + §1.6.1 #1：NPU `VMaxOp`/`VMinOp` 传播 NaN（**设备探针 `history_version/design_probe_clamp_nan.py` PROBE PASS**：张量边界 `vclamp`/`vmin`/`vmax` 对 x/lo/hi 的 NaN 逐位传播、`lo>hi→hi` 一致），守卫逐位冗余 → 消 6 op/元素。**舍弃理由 = 语义已被硬件原生覆盖，保留反增 op**（非「实现简单」） |
| `T.Kernel(..., threads=threads_arg)` 的 threads kwarg | 优化 | CUDA | 无 | **舍弃** | npuir `T.Kernel` 不接受 `threads=`；`is_npu=True` 取代（`vec_add_1d.py` L27 的 `T.Kernel(n_num, is_npu=True)` 佐证）。舍弃理由：CUDA 线程维度概念在 NPU 不存在，threads/npt 双参数折叠为单一 `block_size`（K11/wrapper `_DEFAULT_BLOCK_SIZE`），无性能意图丢失（并行组织由 R1 承接） |
| GPU grid 自由超发（逻辑核任意多） | 优化 | CUDA 硬件调度器 | 无（超发被串行调度 + 核启动开销） | **重新设计** | → §0.6 重设计项 R3（persistent 分核，依据 core-split-strategy.md §1） |
| host 侧广播物化（`expand().contiguous()`） | 优化/契约 | 无 | 部分（核内 vbrc 能力有限） | **保留**（并评估核内替代） | → §0.6 重设计项 R4（host 物化保留 = Op 契约；核内 2D+vbrc 候选经量化评估否决） |

**判定统计**：保留 11 项、等价替换 2 项、重新设计 3 项（R1 并行结构 / R2 bf16 路径 / R3 分核策略）+ 评估性重设计 1 项（R4 广播）、舍弃 2 项（NaN 守卫、threads kwarg，理由见上表）。

### 0.6 NPU 算法重设计

**重设计项 R1: 并行结构（CUDA block×thread 两级并行 → NPU 一维 Kernel + Vector 向量化）**

- **源方案**：`T.Kernel(ceildiv(N, block_size), threads=threads_arg)` 的 CUDA grid×block 两级并行；block 内 `T.Parallel(threads_arg, npt_arg)` 二维线程映射，每线程标量处理 npt 个连续元素（`k = i·npt + j` 索引 + 逐元素标量算术 + NaN 守卫）。性能意图：满并行 + 每线程多元素 ILP。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`（核数按 R3 取值）；核内 tile 的 clamp 计算**替换为整 buffer v-prefix 向量 API**——`T.vclamp(xc, yc, lo_c, hi_c)`（前端展开 `npuir_max` + `npuir_min`，即源 `min(max(v32,lo),hi)` 的融合向量形态），配合 `T.copy` GM↔UB 连续段搬运。源方案「每线程 npt 元素」的 ILP 意图由 Vector 单元 128bit 向量 lane 承接（fp16 ×8 / fp32 ×4 元素/指令）；「threads×npt = block_size」的 tile 粒度由单一 `block_size` 参数承接（K11 折叠）。
- **语义保持论证**：逐元素映射不变（每元素独立计算 `min(max(x,lo),hi)`，无跨元素依赖、无规约），并行组织与线程→元素映射的变化不改变任何输出值——数学上每个 `y[i]` 的表达式与求值顺序（先 `max(x,lo)` 后 `min(·,hi)`）与源码逐步一致（**顺序不可交换**，§1.6.1 诊断实证）。fp32/fp16 路径在 fp32 域 clamp 后单次舍回，与源舍入路径一致（IEEE RN，单次舍回位置相同）；bf16 见 R2。NaN/±inf/±0 行为由 R5 探针核对一致。

**重设计项 R2: bf16/fp16 计算路径（原生 dtype 域 → vcast 升 fp32 clamp → 舍回）**

- **源方案**：源**非 fp8 分支本身即 fp32 中转**（`cast→min(max)→单次舍回`），对所有 fp16/bf16/fp32 一致。故 fp16/fp32 的 fp32 域**保留**（等价替换无变化）；bf16 在 NPU 因 API 限制**强制** fp32 中转。
- **NPU 新算法**：`T.vcast` 三操作数 原 dtype→fp32（`round_mode="rint"`，升精度无损）→ fp32 域 `T.vclamp(x32, y32, lo32, hi32)` → `T.vcast` fp32→原 dtype（`round_mode="rint"`）→ `T.copy` 回 GM。依据：`docs/Tilelang.language/比较操作/T.vclamp.md` §2.2.1 dtype 矩阵 **bf16 ×**（`T.vmin.md` / `T.vmax.md` 同）；`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 支持 `bf16→f32 (rint)` 与 `f32→bf16 (rint)`。
  - fp16 亦可走原生 `vclamp`（fp16 ✓），但**保留 fp32 中转**：与源非 fp8 分支逐位一致、与契约 golden fp32 opmath 域一致、且 bf16 必经此路径（统一代码路径、零额外 GM 流量）；fp16 原生 clamp 的等价性与 ROI 已按 PL-1.27 ② 记录为「非必要优化」（clamp 与 round 可交换，不构成正确性理由）。
- **语义保持论证**：数学恒等（同一公式同一次序）；数值上 fp32 域 clamp 后单次舍回 = 源非 fp8 分支与契约 golden 的**同一舍入路径**——`verify_equiv.py` #2 三 dtype 全边界组（含 `lo>hi` / `±inf`）`max_abs=0.0`、`viol=0`。边界语义：vcast 对 NaN/±inf 原样传播（IEEE 转换）；`lo>hi→hi` 由 min/max 顺序保证（设备探针确认）。
- **意图承接说明**：源优化 #4（fp32 计算域）完整保留（fp16/fp32 直接承接；bf16 为 API 强制的中转形态，意图不变）。

**重设计项 R3: 分核策略（GPU grid 自由超发 → persistent 核数适配）**

- **源方案**：CUDA grid 大小 `ceildiv(N, block_size)` 任意取值（最大 workload 下 65536 个 block），超发 block 由 CUDA 硬件调度器吞吐（无额外软件开销）。
- **NPU 新算法**：NPU 物理 AI Core 有限且超发逻辑核会被运行时**串行调度并引入额外核启动开销**（core-split-strategy.md §1）。本算子 manifest workload 的逻辑核数 256–65536 全部超过物理 Vector 核数（实查 20×2=40，见 §5.5），属极大规模 → persistent 化：`num_kernels = min(num_logical, 40)`（编译期常量），核内 `for i in T.serial(num_local_tasks)`（**静态边界**，`num_local_tasks = ceildiv(num_logical, num_kernels)`）以 grid-stride 方式映射 `block_id = i × num_kernels + cid`，`if block_id < num_logical` 屏蔽越界任务。已验证先例：`examples/elementwise/vec_add_2d_dynamic_shape.py`、`docs/Tilelang.language/比较操作/T.vmin.md` §2.4 示例（`T.serial + block_id = i×BLOCK + cid + if` 结构）。
- **语义保持论证**：任务映射改变（grid-stride 分配 tile）不改逐元素结果——每个 tile 的计算与写出完全独立、无跨核依赖，tile 处理顺序对输出无影响（无竞写：`block_id ↔ tile` 一一对应）。静态边界由 N 编译期特化保证（工厂闭包按 N 特化，N 为 Python int → `num_logical`、`num_local_tasks` 均为编译期常量）。
- **意图承接说明**：源方案无 persistent 结构（CUDA 不需要）；R3 不是源优化的丢弃而是 NPU 硬件差异的必要适配（core-split-strategy.md §1 要素③ 极大规模分支的标准要求）。

**重设计项 R4: 广播语义处理（host 侧物化保留 vs 核内广播候选）**

> 本算子的**核心难点**：三输入张量 + 一般广播语义（含 0-dim/标量边界张量）。下表三候选逐项量化。

| # | 广播处理方案 | 结构 | 语义保持 | 代价 | 处置 |
|---|-------------|------|---------|------|------|
| A | **host 侧物化（Op 层 `_expand_flat`）+ kernel 扁平一维** | `t.expand(out_shape).contiguous().view(-1)`（Op 契约，O6 preserved）；kernel 只见 3×`(N,)` | 逐位等价：`expand+contiguous` 产出与广播语义逐一对应的物化值（每个输出元素 i 取 `(x[i], lo[i], hi[i])` 的广播值） | manifest 全 workload 三输入同 shape → `expand` 为视图、`contiguous` 无拷贝、`view` 免费 → **零额外代价**；真广播（测试级微 shape，如 `(4,8)/(1,8)/(4,1)`、`()`）→ 每个被广播操作数一次 O(N) 物化（host↔device 流量） | ✅ **采纳**（Op 层既有契约，kernel I/O 冻结为扁平一维） |
| B | **核内 2D + `vbrc`/`vmin` 自动广播** | kernel 接收原始 shape + broadcast 规则，核内按轴 `T.vbrc` 展开 | 可等价（需逐轴正确实现） | `T.vmin` §2.2.2：自动广播**仅限「有且仅有一个维度不同且该维为 1」**；`T.vclamp` §2.2.2：张量边界需与 src/dst **逐元素对应（通常同 shape）** → 一般三路广播（`(4,8)/(1,8)/(4,1)` 两维同时不同、0-dim 沿全轴）**无法由单次 vclamp 覆盖**，须逐操作数 `T.vbrc` + 多维 tiling + 轴规则 → 额外 UB buffer 与向量 op；且需**修改冻结的 wrapper/Op 契约**（调用点已传扁平一维） | ❌ **否决**：① 需改冻结契约（超出 Stage-1 kernel 范围，`clamp.py` O6「preserved」）；② 广播仅出现在测试级微 shape，manifest/benchmark 全 workload 无广播 → 收益≈0；③ `vclamp` 张量边界同 shape 限制 + `vmin` 单维广播限制使一般三路广播不可由单 op 承接，实现复杂度/UB 代价高（依据：`T.vclamp.md` §2.2.2、`T.vmin.md` §2.2.2） |
| C | **核内 stride-0 索引广播** | kernel 按 broadcast stride 直接索引 GM | 可等价 | `T.copy` 要求连续段（`T.copy.md` 示例均为连续切片）；stride-0 视图**非连续**、无 `T.copy` 支持；逐元素 gather 无向量 API（核内索引类无 v-prefix 逐元素 gather） | ❌ **否决**：API 缺失（`docs/Tilelang.language/内存操作/T.copy.md` 无 stride-0/gather 形态；核内无逐元素索引向量 API）；代价类论断标注「未文档化假设」，但连续段搬运是本项目唯一已验证 GM↔UB 通路 |

- **选择**：方案 A（host 物化 + 扁平 kernel），与源算子契约、NPU wrapper 调用点、`lerp_tensor` 同族先例（`lerp_tensor.py` 文档「Op layer is responsible for broadcasting」）三方一致。
- **语义保持论证**：`expand(out_shape)` 是**零拷贝视图**（stride 含 0），`contiguous()` 产出按 out_shape 行主序物化的连续张量，`view(-1)` 得到 `(N,)`——对每个展平下标 `i`，其对应广播坐标可从 `out_shape` 唯一还原，取值与 `torch.clamp` 的广播结果逐一相同（`_expand_flat` 与 PyTorch `broadcast_shapes` 语义一致，Op 层用 `torch.broadcast_shapes` 计算 `N_total`）。kernel 内不再有任何广播逻辑，`y[i] = min(max(x[i],lo[i]),hi[i])` 直接落位。
- **代价量化**：manifest 4 workload（`[4096,4096]` / `[16384,16384]`，三输入同 shape）→ 物化次数 0、拷贝 0 字节（`expand` 返回视图 + `contiguous()` 对已连续张量返回自身）。真广播场景（契约测试微 shape）→ 每个 `shape != out_shape` 的操作数一次 O(N) 物化，量级 = 该操作数单次读 + 一次写（与 kernel 主流量同阶但仅发生在测试级小 shape）。**未文档化假设**：manifest/benchmark 是否存在真实广播 workload —— 经 `elementwise.yaml` `ClampFwdOp.workloads` 逐条核对（4 条全为三输入同 shape），标注为已核对结论而非假设。
- **host 约束核对**（ascend-constraints.md §4）：`_expand_flat` 的 `contiguous()` **不是**纯 stride/shape 视图操作（会物化内容），属 host 侧预处理的**既有契约（O6 preserved）**，不在本 kernel 的重设计范围；kernel 内 host 侧零数据操作（仅 N/block_size 元数据）。该张力列入 §9.1 R-3 并给出根因、影响与升级路径（如需严格合规须改 Op/wrapper 契约，属 Stage 2/3 决策）。

**重设计项 R5: NaN 语义实现（显式守卫 → 依赖设备原生 NaN 传播）**

- **源方案**：`r = min(max(x,lo),hi)` 后追加三步覆盖 `r = where(isnan(hi), hi, r); r = where(isnan(lo), lo, r); r = where(isnan(x), x, r)`（合计 +3 `isnan` +3 `if_then_else` = 6 op/元素），因 CUDA `T.max`/`T.min`（`fmaxf`/`fminf`）返回非 NaN 操作数。
- **NPU 新算法**：**不写守卫**，直接 `T.vclamp(xc, yc, lo_c, hi_c)`——NPU `VMaxOp`/`VMinOp` **原生传播 NaN**（非 IEEE `maxNum`），`max(x,lo)` 与 `min(·,hi)` 任一侧 NaN 即输出 NaN，与 torch `maximum`/`minimum` 一致。
- **语义保持论证**：在「max/min 传播 NaN」语义下，三步守卫的谓词永为假（NaN 已经由 max/min 传播到 r），故守卫是**恒等冗余**——`verify_equiv.py` #1「无守卫式 vs 源守卫式」fp64 三 dtype `max_abs=0.0 / viol=0`。前提「设备传播 NaN」为 **设备探针实测**（`history_version/design_probe_clamp_nan.py`：张量边界 `T.vclamp(x_ub,y_ub,lo_ub,hi_ub)` 与 `T.vmin`/`T.vmax` 对 x/lo/hi 的 NaN 逐位传播、与 `torch.clamp` bit-exact）；对照 `verify_equiv.py` 非门禁诊断「maxNum（`torch.fmax/fmin`，不传播）vs 守卫式」出现违反 → 说明守卫只在 maxNum 语义下必要（即源 CUDA 场景）。
- **意图承接说明**：源 #5 是**正确性补丁**而非性能优化；NPU 由硬件语义承接同一正确性，op 数由 8 → 2（`vclamp` 展开 2 op），性能与可读性双收益。若未来设备语义变更导致 NaN 不传播，回退方案 = `T.vcmp(x,x,"ne")` 得 NaN 掩码 + `T.vselect`（`T.vcmp.md` §2.2.3：NaN ≠ NaN；`T.vcmp` dtype 矩阵含 fp16/fp32/**bf16**）——本设计记为 §9.2 回退路径。

**优化意图承接总表**（§0.4 → NPU 方案，无静默丢弃）：

| §0.4 优化手段 | NPU 承接 |
|---------------|----------|
| 1 register 驻留 | UB 驻留（`T.alloc_ub` + 整 buffer v-op），单阶段 load→compute→store 结构不变 |
| 2 coalesced 搬运 | `T.copy` 连续段 GM↔UB（同款语义） |
| 3 thread×npt ILP | Vector 128bit 向量 lane（fp16×8/fp32×4）+ 单一 block_size tile 粒度 |
| 4 fp32 计算域 | 完整保留（fp16/fp32 直承接；bf16 API 强制 fp32 中转，域不变） |
| 5 NaN 守卫 | 舍去——NPU 原生 NaN 传播承接正确性（R5，机器 + 设备双证） |
| 6 has_min/has_max 变体特化 | 原样保留（三变体，§3.3） |
| 7 一维展平 | 原样保留（Op 层） |
| 8 lru_cache 特化 | 原样保留（wrapper forward 惰性构建 + 缓存） |

### 0.7 标杆实现

- **源算子文件（源真相）**：`/home/developer/TileOPs/tileops/kernels/elementwise.py`（`_make_clamp_tensor_kernel` @ **L3018**、`ClampTensorFwdKernel` @ **L3227**）
- **Stage 0 抽取件（接口契约冻结基准）**：`examples/TileOPs/tileops/kernels/elementwise/clamp/_clamp_fwd_kernels.py`
- **参考实现 / 测试基准**：`torch.clamp(x.float(), mn.float(), mx.float()).to(dtype)`（源仓 `examples/TileOPs/tests/ops/test_special_elementwise.py` `ClampTensorTest.ref_program` L115–L116；NaN 用例 L196–L208）
- **manifest 规格**：`examples/TileOPs/tileops/manifest/elementwise.yaml` `ClampFwdOp` 条目（4 workloads + roofline `flops=N` / `bytes=4·N·elem_bytes`）
- **性能基准**：`examples/TileOPs/benchmarks/ops/bench_independent_elementwise.py`
- **NPU 集成目标（wrapper/Op，冻结）**：`examples/TileOPs/tileops/kernels/elementwise/clamp/clamp.py`（`ClampTensorFwdKernel`；`_DEFAULT_BLOCK_SIZE = {fp32:2048, fp16:4096, bf16:4096}`）、`examples/TileOPs/tileops/ops/elementwise/clamp.py`（`ClampFwdOp`）
- **同族迁移先例**：`examples/hardtanh/_make_hardtanh_kernel/`（scalar-bound clamp 家族，PL-1.23 第 4 证）、`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/`（三输入 elementwise 广播）
- golden 以 §0.1 语义为依据（优先移植源仓参考 `torch.clamp` fp32 域），见 §8.1
---

## 1. 概述

### 1.1 算子名称

`_make_clamp_tensor_kernel`（Op 层名 `ClampFwdOp` / `clamp`；Tensor-bound clamp）

### 1.2 功能描述

Tensor-bound clamp 前向核：对三个已展平为一维 `(N,)` 的同 dtype 张量计算 `y = min(max(x, lo), hi)`，纯逐元素、无规约、无跨元素依赖；`lo`/`hi` 的存在性由构造期 `has_min`/`has_max` 决定。

### 1.3 数学公式

$$
y[i] = \min\big(\max(x[i],\ lo[i]),\ hi[i]\big),\quad i \in [0, N)
$$

参考 API：`torch.clamp(input, min: Tensor, max: Tensor)`（Tensor-bound 形式；`lo > hi` 时结果取 `hi`）。

### 1.4 算法描述

**（迁移决策后的 NPU 侧算法；与源算法结构差异逐条注明出处）**

单遍逐元素三输入向量算法（§1.6.0 调研选定基线公式 `min(max(...))`，无等价化简替代）：

1. **搬入**：每 tile 将 `x` / `lo` / `hi` 三段连续 `block_size` 数据 `T.copy` GM→UB（单边界变体只搬 2 段；尾块按 `tail` 截断，§6.4）；
2. **计算**：UB 内 fp32 域整 buffer 向量链——fp16/bf16 先 `T.vcast`(→fp32, rint) 升精度，`T.vclamp(xc, yc, lo_c, hi_c)`（前端展开 `npuir_max` + `npuir_min`，即 `min(max(·,lo),hi)`），再 `T.vcast`(→原 dtype, rint) 舍回；fp32 直接 `T.vclamp`。**无 NaN 守卫**（§0.6 R5）；
3. **搬出**：`T.copy` UB→GM（尾块按 `tail` 截断）。

与源算法的结构差异：① CUDA threads×npt 二维线程并行 → 一维 Kernel + v-prefix 向量化（R1）；② bf16 计算因 API 强制 fp32 中转（fp16/fp32 承接源同域）（R2）；③ grid 按逻辑核数自由超发 → persistent 核数适配（R3）；④ 显式 NaN 守卫 → 依赖设备原生 NaN 传播（R5）；⑤ 广播由 Op 层 host 物化承接（R4，源一致）。数据流骨架（单阶段 load→compute→store、片上驻留、连续段搬运、fp32 中转）与源一致。

### 1.5 数据流图

```
fp32 主路径（每 tile，block_size=2048）：
GM[x]  ──T.copy(tail 截断)──> UB[x_ub:fp32]  ─┐
GM[lo] ──T.copy──────────────> UB[lo_ub:fp32] ┼─ vclamp(x_ub → y_ub, lo_ub, hi_ub) ─> UB[y_ub] ──T.copy──> GM[y]
GM[hi] ──T.copy──────────────> UB[hi_ub:fp32] ┘

fp16 / bf16 路径（R2，block_size=4096）：
GM[x/lo/hi:dt] ──T.copy──> UB[x/lo/hi_ub:dt] ──vcast(rint)──> UB[x32/lo32/hi32:fp32]
  ──vclamp(x32 → y32, lo32, hi32)──> UB[y32:fp32] ──vcast(rint)──> UB[y_ub:dt] ──T.copy──> GM[y]
（vcast 全程 UB 内，零额外 GM 流量；无 L1/L0 参与——纯 Vector 路径；无中间 GM 往返）
```

### 1.6 算法调研与优化分析 ⭐

> 设计第一优先级：先调研（1.6.0，迁移任务在 Phase M0 后、M1 前完成，结论已驱动 §0.5/§0.6），再公式级优化（1.6.1），再循环/标量的向量化替代（1.6.2），最后向量化轴与布局决策（1.6.3）。本节分析对象 = §1.6.0 选定、经 §0.6 重设计落地的 NPU 算法。clamp（Tensor-bound）为**单步逐元素类**（无规约/窗口/矩阵/多步融合），按 algorithm-research.md §2 采用轻量调研深度（四问各一行结论 + 依据），但四问不可跳过。

#### 1.6.0 算法调研（Algorithm Research）⭐

**R1 等价化简公式候选**（基线在表；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线（源算法 / 输入公式）** | `min(max(x, lo), hi)`（`T.vclamp`，前端 2 op） | — | — | ✅（基线） |
| 1 | 交换 min/max 顺序 | `max(min(x, hi), lo)` | **非等价**：`lo > hi` 时基线恒为 `hi`、本式恒为 `lo`（PyTorch 语义顺序敏感；`verify_equiv.py` 非门禁诊断实测 6.3 万点违反、`max_abs≈4.97`） | — | ❌ 非等价（已机器证伪） |
| 2 | 融合元算 `T.vclamp` | `T.vclamp(x,x,lo,hi)` → `npuir_max`+`npuir_min` | 逐位恒等（前端按 max→min 顺序展开，与基线同序） | op 数不变（仍 2 op）；单 API、免手写两步 | ✅（基线形态，非 op 削减） |
| 3 | relu 分解 | `lo + relu(x−lo) − relu(x−hi)` | `lo ≤ hi` 时恒等；**`lo > hi` 时非等价**（区间内非常量）；op 数 4（+100%） | **负收益**（4 op vs 2 op） | ❌ op 数更多且 `lo>hi` 失效 |
| 4 | median 形式 | `median(lo, x, hi)` | `lo ≤ hi` 恒等；`lo > hi` 时 median 返回 `x`（截断）≠ `hi`（PyTorch）**非等价** | op 数不变但语义偏离 | ❌ 非等价 |
| 5 | 单边界退化特化（`lo=-inf`→仅 vmin；`hi=+inf`→仅 vmax） | `vmin(x,hi)` / `vmax(x,lo)` | 逐位恒等（一侧无穷时 clamp 退化为单侧） | 1 op（-50%，仅退化边界） | ✅ 评估（见 §1.6.1 #3） |
| 6 | **消除显式 NaN 守卫**（源守卫在 NPU 冗余） | `min(max(x,lo),hi)`（不写 `isnan`/`if_then_else`） | 在「max/min 传播 NaN」语义下逐位恒等（设备探针实测前提） | **6 op → 0**（8 op → 2 op） | ✅ **采纳**（§1.6.1 #1） |

> 候选 5/6 的**边界为张量**（非常量），故**不存在** hardtanh 式的「编译期边界常量折叠」空间（`min_val`/`max_val` 是 tensor，非 Python 常量；合并进 §1.6.1 表格说明）。候选 1/3/4 均因 `lo > hi` 语义（PyTorch 固定 `min(max)` 顺序）被证伪——这是本算子区别于无界 elementwise 的关键语义约束。

**R2 在线算法**：**天然单遍，无需在线变体**。结构依据：输出 `y[i]` 仅依赖 `(x[i], lo[i], hi[i])`，无 running 统计量（和 / 最大值 / 计数）、无跨元素依赖——逐元素映射本身即单遍流式（algorithm-research.md §3 R2 结构判据「逐元素映射天然单遍」）；源算法亦为单遍（§0.4 无 online 手段），一致。

**R3 复杂度对比**（带宽受限算子——算术强度 = `N` flops / `4N·elem_bytes`：fp16 = 0.125 flop/B、fp32 = 0.0625 flop/B，远低于 Vector 单元平衡点，**访存为主导项**）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线 `min(max)`（无守卫） | 2N（max + min 各 N；无超越函数） | **4N·B**（3 读 + 1 写，无中间 GM 往返——tile 驻留 UB）= I/O 下界，与 manifest roofline `bytes=4·N·elem_bytes` 一致 | 1（每元素每张量恰好读一次写一次） | O(block_size) tile 级（UB：32–96KB，§4.5） | N/block_size 个完全独立 tile，零跨核同步 |
| 源基线（含 6 op 守卫） | 8N（+3 isnan 比较 +3 select） | 4N·B（相同；守卫在寄存器/UB 内） | 1 | 同上 | 同上 |
| 候选 3 relu 分解 | 4N（+2 relu +2 加减） | 4N·B | 1 | 同上 | 同上 |
| 候选 5 退化边界特化 | 1N | 4N·B（输入仍按契约搬满） | 1 | 同上 | 同上 |

**结论**：选定候选 0（基线公式）形态、**采纳候选 6（消除守卫）**——基线在四口径上全部并列最优（FLOPs 最少、访存达 I/O 下界、单遍、缓冲 tile 级），候选 3 仅增加 FLOPs、候选 1/4 非等价；候选 5 仅对退化边界（`±inf` 张量）有效且需运行期判定（边界为张量，无法编译期特化）→ 不采纳（§1.6.1 #3）。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附 `docs/` 与 `examples/` 目录佐证）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（`vclamp` 2 op） | 逐元素 min/max → Vector 单元完美匹配（`T.vclamp` fp16 √ / fp32 √，`T.vclamp.md` §2.2.1）；bf16 需 R2 中转（`vcast` bf16↔f32 √） | fp32：8 buf × 2048×4B = 64KB（无条件全量分配，§4.5）；fp16：4×8KB(dt) + 4×16KB(f32) = 96KB；bf16 同 fp16 = 96KB（均 ≤ UB 192KB，含 multi-buffer 膨胀见 §4.5） | block_size×B 均 32B 整除（2048×4B=8KB / 4096×2B=8KB）；向量宽度整除（fp16×8、fp32×4 均整除 block_size）；N 非整除由尾块 `tail` 处理（§6.4） | N 编译期特化 → grid/循环边界全静态；尾块 `tail` 为运行时标量但仅 1 次/tile | 单阶段无需流水；2 op（`vclamp` 展开）已是最简融合 | ✅ **选定** |
| 候选 3 relu 分解 | 同（多 2 次 vrelu/vadd 链长 100%） | 同 | 同 | 同 | 同 | ❌ R3 FLOPs +100% 且 `lo>hi` 非等价 |
| 候选 5 退化边界特化 | 同（1 op） | 同 | 同 | ❌ 边界为**张量**→ 需运行期全元素判定（额外 N 次比较 + 全核归约）才知是否退化，代价 > 省下的 1 op | 同 | ❌ 边界非常量，不适用编译期折叠 |
| 候选 1/4 顺序/median 变体 | — | — | — | — | — | ❌ 非等价（`lo>hi`），直接淘汰 |

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)）**：

- 设备：**Ascend910_9362 单 die**（本机实查 `NPUUtils.get().get_aicore_num()=20` → 40 Vector 核）；核→GM ≈ **1.06 TB/s**（单 die 主口径）、HBM ≈ 1.64 TB/s（辅口径）——引用 `CONST-ascend910-9362-profile`（`Ascend910_9362 / torch_npu 2.7.1.post4 / CANN 9.0.0 / 2026-10-13`）。
- **流量项** = `4·N·elem_bytes ÷ 1.06 TB/s`（3 读 + 1 写；该口径与 `BitwiseNotFwdOp` 实测饱和点 1.06–1.10 TB/s 一致）：
  | workload | N | dtype | 流量 | 流量项 |
  |----------|---|-------|------|--------|
  | elementwise-16m | 2²⁴ | fp16 | 134.2 MB | **126.6 µs** |
  | elementwise-16m | 2²⁴ | fp32 | 268.4 MB | **253.2 µs** |
  | elementwise-256m | 2²⁸ | fp16 | 2.147 GB | **2026 µs** |
- **发射项** = 向量 op 数 × ~0.5µs/op ≈ 6 op × 0.5 = **3 µs**（固定项；`CONST-vector-launch-overhead` 为 Ascend910B2C 实测 + 「9362 未重验」注记 → 标注**待重验假设**，且被 auto-multi-buffer 隐藏于 MTE2 窗口内，PL-1.6-copy-floor）。
- **容量项**：**不适用**（纯 Vector，无跨引擎 S/P 往返，`CONST-store-fixpipe-gm-only` 不涉及）。
- → **估算下界 = max(126.6, 3) = 126.6 µs（16M fp16）/ 253.2 µs（16M fp32）/ 2026 µs（256M fp16）**（流量项主导；1M 档受 L2 驻留影响，估算偏保守，标注「未实测假设 + L2 效应依据」）。**依据**：`CONST-ascend910-9362-profile` + `CONST-vector-launch-overhead`（版本戳见上）。**对账**：`BitwiseNotFwdOp` 64M int32（512MB）实测 487.81µs → 1.1006 TB/s、256M int32（2GB）实测 2020.44µs → 1.0629 TB/s，与本法计算同阶（偏差 < 5%），佐证流量项口径。

**调研结论**：选定**基线公式 `y = min(max(x,lo),hi)` 的单遍逐元素 Vector 算法**（`T.vclamp` 融合元算 + fp32 域 + persistent 分核），并**采纳候选 6（消除 NaN 守卫）**。关键依据：R3 表中基线 FLOPs 最少（2N）且访存已达 I/O 下界（4N·B，与 manifest roofline `flops=N / bytes=4N·elem_bytes` 一致，注意 manifest flops 按「1 次融合 compare-and-select」计、本设计按 max+min 2 op 计，口径差异不影响访存主导结论）；R4 基线对 Vector 单元/UB 容量/对齐/静态边界全部亲和；候选 1/3/4 因 `lo>hi` 语义非等价（已机器证伪），候选 5 因边界为张量不适用编译期折叠。与基线（源算法）的**结构差异**：① 消 NaN 守卫（6 op → 0）；② 融合为单 `T.vclamp`；③ 实现载体变化（fragment 标量循环 → 整 buffer v-prefix 链；persistent 分核）。算法族本身无差异。**「无更优替代」的调研范围**：① algorithm-research.md §5 参考表本体 `references/algorithm-candidates.md` 的 `ALG-elementwise` 条目（R1 记「公共子表达式消除；除法乘倒数；cast 链合并」；R2 记「逐元素映射天然单遍」；R3 记「融合消除中间 GM 往返——访存收益主导」；R4 记「单遍 Vector 流；dtype 路径」）——本条目无 clamp 专门候选，故以恒等变形自查补足（R1 候选 1–6）；② 恒等变形空间（交换顺序 / relu 分解 / median / 单侧退化，逐项评估见 R1）；③ `examples/elementwise/`（`vec_add_1d`/`vec_add_2d*`/`example_elementwise_add`/`exp2`/`log2`——均为直排 v-prefix 单遍结构，无 clamp 公式变形先例）与 `examples/hardtanh`（scalar 边界 clamp：其「边界常量折叠」不能迁移，因本算子边界为张量）；④ pattern-library（预注入条目消费：`PL-1.6-copy-floor`（多输入 elementwise copy-floor 与 MTE2 退化曲线——本算子 3 输入直接相关）、`PL-1.23-vop-chain-inplace-alias`（vclamp 原地别名，含 NaN 传播设备语义）、`PL-1.17-subfp32-fp32-transit`（sub-fp32 fp32 中转）、`PL-1.27-same-family-precedent-precondition`（同族先例承接前提核对——支持「源 fp32 域 = 保留而非优化」判定）、`CASE-elementwise-lerp-opt`（三输入 elementwise 参考档案））；⑤ 源算子结构分析（§0.3，单遍 2 op + 6 op 守卫）。源算法优化手段的意图承接见 §0.6 末尾总表，无静默丢弃。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定的基线算法（经 §0.6 R1/R2/R5 落地）。逐类别核查 + 采纳项四要素；采纳项经 `verify_equiv.py` 机器验证（结果表见本节末）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| **1** | **显式 NaN 守卫消除（采纳）** | `r=min(max(x,lo),hi)`；`r=where(isnan(hi),hi,r)`；`r=where(isnan(lo),lo,r)`；`r=where(isnan(x),x,r)`（8 op） | `r = min(max(x, lo), hi)`（`T.vclamp` 展开 2 op） | 在「max/min 传播 NaN」语义下，`max(x,lo)`/`min(·,hi)` 已把任一 NaN 操作数传播到 `r`，三步守卫谓词恒假 → **逐位恒等**。`verify_equiv.py` #1（无守卫式 vs 源守卫式，fp64，三 dtype）`max_abs=0.0 / viol=0`。前提由**设备探针实测**（张量边界 `vclamp`/`vmin`/`vmax` NaN 逐位传播 + `lo>hi→hi`）；对照非门禁诊断「maxNum 语义 vs 守卫式」出现违反（证明守卫仅 CUDA maxNum 场景必需） | **6 op → 0**（每元素 op 数 8 → 2，-75% 向量指令）；带宽口径不变（4N·B，守卫不产生 GM 流量）→ 带宽受限算子下收益主要体现在发射项与流水窗口占用（PL-1.6：计算链被 MTE2 隐藏，ROI 属「减指令 + 可读性」，非吞吐级） |
| **2** | fp32 中转计算域（**保留并对齐**，非新增采纳） | 源非 fp8 分支 `cast→min(max)→cast`（fp32 域） | 同式（`vcast(rint)`×3 → `vclamp` fp32 → `vcast(rint)`×1） | **保留**源既有行为：与契约 golden `torch.clamp(x.float(), lo.float(), hi.float()).to(dtype)` 同域 → 逐位一致（PL-1.27 ②）。`verify_equiv.py` #2 三 dtype × 全边界组（含 `lo>hi`/`±inf`）`max_abs=0.0 / viol=0`；`#2 roundtrip lossless` 证中转无损。bf16 为 API 强制（`T.vclamp.md` §2.2.1 bf16 ×） | op 数不变（中转 4 次 vcast，UB 内）；bf16 无替代路径 |

**未采纳项（记录理由，避免遗漏优化）**：

| # | 候选 | 理由 |
|---|------|------|
| E1 | 编译期边界特化折叠（hardtanh 式 `lo=-inf→vmin` 等） | **不可行**：本算子 `lo`/`hi` 是**运行时张量**（非 hardtanh 的编译期 Python 常量），无编译期取值可折叠；若要按「整张量恒为 ±inf」特化需运行期全量归约判定，代价 > 收益（§1.6.0 R4 候选 5） |
| E2 | fp16 原生 dtype clamp（跳过 fp32 中转） | clamp 与 round 可交换 → 数值亦等价（PL-1.27 ②），但：① bf16 无 `vclamp`（API 硬限制）；② 源非 fp8 分支与契约 golden 均为 fp32 域（保留对齐优先）；③ 带宽受限下 ROI≈0。记为 Stage 4 备选（不改变结构） |
| E3 | relu 分解 / 顺序交换 / median（§1.6.0 R1 候选 1/3/4） | `lo > hi` 语义非等价（已机器证伪）+ op 数更高 |
| E4 | 公共子表达式消除（CSE） | 无公共子式：`x`/`lo`/`hi` 各引用一次，`max(x,lo)` 结果仅被 `min(·,hi)` 用一次 |
| E5 | 除法转乘倒数 / rsqrt / 超越函数降代换 | 无除法、无开方、无超越函数（纯比较选择） |
| E6 | 归约结构优化 | 无规约（§0.1） |

**优化结论**：采纳 **1 项**（#1 NaN 守卫消除，8 op → 2 op）；#2 为**契约域保留**（机器验证逐位一致）。**优化后公式** = `y[i] = min(max(x[i], lo[i]), hi[i])`，在 fp32 域求值后按目标 dtype 单次舍回（bf16/fp16 经 `vcast` 进出，§0.6 R2）——此式即 §3.1 公式拆解的唯一输入。

**等价性机器验证（D-1）**：脚本 `examples/clamp/_make_clamp_tensor_kernel/verify_equiv.py`（CPU；候选式 = NPU 向量链逐 op 复现；基线式 = 源守卫式在 fp64 参照下的输出；域项对照契约 golden `torch.clamp(x.float(), lo.float(), hi.float()).to(dtype)`）。**执行结果表**（脚本实际输出，`python3 verify_equiv.py`，**ALL 23 ITEMS EQUIV_PASS**）：

| 对照项 | 候选式 | 基线 / 参照 | 样本数 | 最大 ulp 差 (max_abs) | 违反率 | 结论 |
|--------|--------|------------|--------|----------------------|--------|------|
| `#1 no-guard vs source-guarded (fp64)` (fp32) | 无守卫 `min(max)` | 源式 + 3 守卫 | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#1 no-guard(fp64→dt) vs GOLDEN` (fp32) | 无守卫 + 舍回 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#2 fp32-transit vs GOLDEN [default(-1,1)]` (fp32) | fp32 中转链 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#2 fp32-transit vs GOLDEN [nondefault(-0.5,2)]` (fp32) | fp32 中转链 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#2 fp32-transit vs GOLDEN [lo>hi(2,1)]` (fp32) | fp32 中转链 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#2 fp32-transit vs GOLDEN [lo=-inf]` (fp32) | fp32 中转链 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#2 fp32-transit vs GOLDEN [hi=+inf]` (fp32) | fp32 中转链 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#1 no-guard vs source-guarded (fp64)` (fp16) | 无守卫 | 源式 + 守卫 | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#1 no-guard(fp64→dt) vs GOLDEN` (fp16) | 无守卫 + 舍回 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#2 fp32-transit vs GOLDEN [default(-1,1)]` (fp16) | fp32 中转链 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#2 fp32-transit vs GOLDEN [nondefault(-0.5,2)]` (fp16) | fp32 中转链 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#2 fp32-transit vs GOLDEN [lo>hi(2,1)]` (fp16) | fp32 中转链 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#2 fp32-transit vs GOLDEN [lo=-inf]` (fp16) | fp32 中转链 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#2 fp32-transit vs GOLDEN [hi=+inf]` (fp16) | fp32 中转链 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#2 roundtrip lossless` (fp16) | f16→f32→f16(rint) | `x`（恒等） | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#1 no-guard vs source-guarded (fp64)` (bf16) | 无守卫 | 源式 + 守卫 | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#1 no-guard(fp64→dt) vs GOLDEN` (bf16) | 无守卫 + 舍回 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#2 fp32-transit vs GOLDEN [default(-1,1)]` (bf16) | fp32 中转链 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#2 fp32-transit vs GOLDEN [nondefault(-0.5,2)]` (bf16) | fp32 中转链 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#2 fp32-transit vs GOLDEN [lo>hi(2,1)]` (bf16) | fp32 中转链 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#2 fp32-transit vs GOLDEN [lo=-inf]` (bf16) | fp32 中转链 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#2 fp32-transit vs GOLDEN [hi=+inf]` (bf16) | fp32 中转链 | 契约 golden | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| `#2 roundtrip lossless` (bf16) | bf16→f32→bf16(rint) | `x`（恒等） | 262,160 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |

> 判定准则 = Stage 3 门禁同口径 `|cand−base| ≤ atol + rtol·|base|`（fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2），违反率 0 即 PASS；覆盖随机张量 `randn`（2²⁰）+ 角点（NaN 分别置于 x/lo/hi、±0、±inf、次正规 1e-45、边界邻域、`lo>hi`、退化边界组）。

**非门禁诊断（机器输出，不进 gate —— 证被拒候选与前提敏感性）**：

| 被拒候选 / 前提 | dtype | max_abs | 违反数 | 结论 |
|----------------|-------|---------|--------|------|
| 交换顺序 `max(min(x,hi),lo)` vs 守卫式 | fp32 | 4.973e+00 | 62,720 | 非等价（`lo>hi` 触发）→ 拒绝 |
| 交换顺序 `max(min(x,hi),lo)` vs 守卫式 | fp16 | 5.152e+00 | 63,167 | 非等价 → 拒绝 |
| maxNum（`torch.fmax/fmin`，不传播 NaN）vs 守卫式 | fp32 | 0.000e+00 | 2 | 仅 2 个 NaN 角点违反 → 证明**守卫仅在 maxNum 语义（源 CUDA）下必要**，NPU 原生传播语义下冗余（采纳 #1 的前提成立） |

**前提的设备实测（D-3 设计期探针）**：`examples/clamp/_make_clamp_tensor_kernel/history_version/design_probe_clamp_nan.py`（**PROBE PASS**，tilelang 0.1.2+c3a94f1d / CANN 9.0.0 / Ascend910_9362 / 2026-10-13）——断言 [1] 张量边界 `T.vclamp(x_ub,y_ub,lo_ub,hi_ub)` == `torch.clamp` bit-exact（NaN-aware）；[2] NaN 分别置于 x / lo / hi 均传播；[3] `lo>hi → hi`（min/max 顺序）成立；[4a/4b] 张量边界 `T.vmin`/`T.vmax` NaN 传播。该探针是采纳项 #1 的**前提证据**（非完整算子实现，符合 D-3 预算：1 个探针、<3 min）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案（§0.6 重设计后的 NPU 算法）中**全部**循环与标量计算点：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 逐元素 `min(max(x,lo),hi)` | 源 `T.Parallel(threads, npt)` 标量循环逐元素比较 | `T.vclamp(xc, yc, lo_c, hi_c)`（整 UB buffer；前端 `npuir_max`+`npuir_min`） | ✅ | —（佐证：`docs/Tilelang.language/比较操作/T.vclamp.md` §2.4；`T.vmin.md` §2.4 官方 persistent 用法） |
| 2 | fp16/bf16 逐元素升/降精度 | 源 `T.cast(x_reg[k],"float32")` / `T.Cast(dtype,r)` 逐元素 | `T.vcast(src_ub, dst_ub, round_mode="rint")`（整 buffer） | ✅ | —（佐证：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1） |
| 3 | 源 NaN 守卫（3×`isnan` + 3×`if_then_else`） | 逐元素标量条件 | **整体消除**（§1.6.1 #1：设备原生 NaN 传播已覆盖） | ✅（消除） | —（`docs/Tilelang.language/比较操作/` 目录无 `T.isnan` 条目；NPU 原生 `VMax/VMin` 传播语义已覆盖，探针实测） |
| 4 | tile 起始偏移 `off = block_id × block_size`、`block_id = i×num_kernels + cid` | 标量整数乘加 | 无逐元素等价 API（索引计算非数据计算） | ❌ | **block 级索引 / 任务映射计算**：每 tile 仅 2–3 次标量整数运算，非逐元素热点（占比 < 0.1% 指令量）；`T.vmin.md` §2.4 官方示例同款保留 |
| 5 | 尾块长度 `tail = T.min(block_size, N − off)` | 标量比较 | 无 | ❌ | **依赖动态 shape 的边界处理**：N 非整除时末 tile 边界量；已采用 `examples/elementwise/vec_add_1d.py` L31–L33 的已验证 `T.min` 模式（`docs/Tilelang.language/比较操作/T.min.md`），无法静态化（N 编译期常量但整除性随 N 变化；padding 方案违反 host 侧禁改输入约束，ascend-constraints.md §4） |
| 6 | 核内多 tile 调度循环 `for i in T.serial(num_local_tasks)` | 循环 | 无（tile 间无数据依赖，但 persistent 结构需循环推进任务） | ❌ | **tile 级调度循环**（R3 persistent 分核的标准结构，core-split-strategy.md §1 要素③；`T.vmin.md` §2.4 示例同款）；循环边界 `num_local_tasks` 为编译期静态值，循环体一次处理一个完整 tile（内部全向量化），非逐元素热点 |
| 7 | host 侧 `num_kernels/num_logical` 计算、`block_size` 配置 | Python 标量 | — | ❌ | **host 元数据计算**（不在 kernel 内；工厂闭包编译期完成，符合决策树 §1「维度参数自推导」） |
| 8 | host 侧三输入预广播物化（`expand().contiguous().view(-1)`） | host 张量操作 | — | ❌ | **host 侧 Op 层契约**（不在本 kernel 内；§0.6 R4 方案 A）；广播语义处理的选择与代价见 §0.6 R4 与 §9.1 R-3 |

**向量化结论**：3 类逐元素计算点（#1–#3）**已全部向量化或消除**，替代/消除方案均有 docs + examples 佐证；保留 5 处标量/循环（#4–#8），理由类别分别为 block 索引 / 动态边界 / tile 级调度循环 / host 元数据 / host 广播预处理，逐项见上表——§6 循环结构与本结论一致（kernel 内无任何逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

**单轴豁免判定**：本算子为**一维展平单轴算子**（Op 层已将三输入物化为 `(N,)` 连续一维，`clamp.py` wrapper 契约），数据向量轴仅存在一个候选 → 按 SKILL.md Phase 2 第 3 条「纯 GEMV/单轴算子可写明仅一个候选轴豁免」处理。**但本算子的阻塞级布局决策不在数据轴，而在「广播语义放在哪一侧」（§0.6 R4）**——按「≥2 非 batch 维度」的广播类候选清单补充完整分析：

**候选矩阵（广播处理 × 向量化轴）**：

| # | 布局 / 广播方案 | 向量化轴 | repack 路径 | 预估收益 / 代价 | 是否采纳 |
|---|----------------|---------|------------|----------------|---------|
| 1 | **一维连续（I/O 原生，Op 层已展平）+ host 物化广播** | 展平 N 轴（stride=1 连续） | 无（Op 侧 `expand/contiguous/view`，见 R4-A） | 基线：整除性 fp32 2048%4=0 / fp16 4096%8=0，尾 lane 浪费 0%；无累加链；UB 预算 §4.5 达标；manifest workload 同 shape → 物化零代价 | ✅ **采纳** |
| 2 | 核内 2D tiling + `T.vbrc` 逐操作数广播 | 行内 cols 轴 | 核内广播展开 | 一般三路广播（两维同时不同 / 0-dim 沿全轴）无法由单次 `vclamp` 覆盖（`T.vclamp.md` §2.2.2 张量边界需同 shape；`T.vmin.md` §2.2.2 仅单维自动广播）→ 需改冻结 wrapper 契约 + 多维 tiling + 额外 UB；广播仅见测试级微 shape → 收益≈0 | ❌（见 §0.6 R4-B 量化理由） |
| 3 | 核内 stride-0 索引广播 | — | 无 repack（直接索引） | `T.copy` 仅支持连续段（`T.copy.md`），stride-0 非连续；核内无逐元素 gather 向量 API | ❌（API 缺失，§0.6 R4-C） |
| 4 | 核内重排布局（转置 / 分块交织） | — | T.transpose 链 | 无意义：无窗口/无跨步/无规约轴可优化；pattern-library 转置链适用于 NCHW 类窗口算子，本算子无此结构 | ❌（无适用场景） |

**源码轴对照（迁移任务要求）**：GPU 源码的 `T.Parallel(threads, npt)` 二维轴是 **CUDA 硬件组织的输入而非结论**（thread/warp 映射与 NPU Vector 轴不对应）；其「每线程 npt 个连续元素」的隐式轴向选择（连续 N 轴、npt 为向量宽度雏形）与候选 #1 的 N 轴选择一致——源码轴选择经独立评估后确认采纳同一连续轴，Vector 128bit（fp16×8/fp32×4）承接 npt 的多元素意图。**广播维对照**：源算法**也没有核内广播**（Op 层已物化展平），故「host 物化」不是绕过 NPU 限制的妥协，而是与源算法结构一致的迁移承接。

**布局决策结论**：选定方案 #1——核内布局 = 一维 `(block_size,)` UB buffer（fp32 路径 ×4；fp16/bf16 路径 ×7 = 3 dtype 入 + 3 fp32 + 1 dtype 出，§4.3/§4.5），向量化轴 = 展平 N 轴（stride=1），无 repack；广播语义由 Op 层 host 物化承接（R4-A）。该结论同步落入 §3.3 伪代码（buffer 形状 `(block_size,)`、v-prefix 操作数即整 buffer）与 §6 循环结构（无逐元素循环，向量操作覆盖全 tile）；§4.5 UB 预算按此布局计算。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（`op_requirements.programming_mode` 指定，且与算子特征判定一致）

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（逐元素 max/min 比较选择 + bf16 cast），无 matmul、无归约 | 无 Cube/L0 需求，不需要 MixCV |
| 复杂度级别 | 单步算子（一次搬运 + 2 op 向量链 + 一次搬出） | 无多阶段流水/精细 buffer 生命周期控制需求 |
| 同步 | 每 tile 独立、无核间依赖 | 无手动 `sync_block_set/wait` 需求 |
| API | v-prefix（`vclamp`/`vcast`）+ `T.copy` + `T.alloc_ub` 均为 Developer 可用形态 | `T.vclamp.md` §2.4 示例即 `@tilelang.jit(target="npuir")` Developer 形态 |
| 先例 | `examples/elementwise/vec_add_1d.py`（一维逐元素 + alloc_ub + vadd）、`examples/hardtanh/_make_hardtanh_kernel/`（clamp 家族 Developer 实现） | 同构算子已验证 |

决策树路径：纯 element-wise 单步 → Developer 模式（decision-tree.md §2）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub((block_size,), dtype)` 显式 UB（v-prefix API 要求操作数在 UB，`T.vadd.md` §2.3 第 1 条） |
| 计算方式 | v-prefix 向量 API（`T.vclamp` + `T.vcast`）整 buffer 操作；无 `T.Parallel` 标量循环 |
| 同步 | Developer 自动同步（copy→compute→copy 顺序依赖由编译器保证） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` |
| 环境变量 | `TILELANG_ASCEND_MODE=Developer`（`examples/elementwise/example_elementwise_add.py` 先例；`clamp.py` wrapper `ascend_mode = "Developer"` L134） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式 = §1.6.1 **优化后公式**（无守卫）：`y = min(max(x, lo), hi)`；fp16/bf16 路径含 R2 的 cast 前后处理。拆解与 §1.6.2 向量化结论一致。

**fp32 路径**：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_ub ← x[off:off+tail]`，`lo_ub/hi_ub` 同 | 三输入 tile 搬入 GM→UB（尾块截断） |
| 2 | `y = min(max(x, lo), hi)` | `T.vclamp(x_ub, y_ub, lo_ub, hi_ub)`（2 op，前端 `npuir_max`+`npuir_min`） |
| 3 | `y[off:off+tail] ← y_ub[0:tail]` | 结果搬出 UB→GM |

**fp16 / bf16 路径**（R2）：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_ub/lo_ub/hi_ub ← x/lo/hi[off:off+tail]`（原 dtype） | 搬入 |
| 2 | `x32/lo32/hi32 ← vcast(x_ub/lo_ub/hi_ub, rint)` | 升 fp32（无损） |
| 3 | `y32 = min(max(x32, lo32), hi32)` | fp32 域 `T.vclamp(x32, y32, lo32, hi32)` |
| 4 | `y_ub ← vcast(y32, rint)` | 舍回原 dtype（单次） |
| 5 | `y[off:off+tail] ← y_ub[0:tail]` | 搬出 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| tile 分派 | `block_id = i·num_kernels + cid` | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` + `T.serial(num_local_tasks)` | num_kernels/num_local_tasks 为编译期常量（§5.5） | Developer |
| 搬入/搬出 | `x[off:off+tail] ↔ x_ub[0:tail]` | `T.copy(src_slice, dst_slice)` | 连续段切片拷贝（`T.copy.md`：gm-ub / ub-gm 均支持，bf16 √） | Developer |
| 尾块长度 | `tail = min(block_size, N − off)` | `T.min(a, b)` | `docs/Tilelang.language/比较操作/T.min.md` | Developer |
| 两侧 clamp | `y = min(max(x, lo), hi)` | `T.vclamp(src=x_ub, dst=y_ub, min=lo_ub, max=hi_ub)` | 张量边界 ✓（`T.vclamp.md` §2.2.2）；fp16 √ / fp32 √ / bf16 ×（§2.2.1） | Developer |
| fp16/bf16 升精度 | `fp32 ← fp16/bf16` | `T.vcast(src=x_ub, dst=x32, round_mode="rint")` | fp16/bf16→f32 仅 rint（`T.vcast.md` §2.2.1） | Developer |
| fp16/bf16 舍回 | `fp16/bf16 ← fp32` | `T.vcast(src=y32, dst=y_ub, round_mode="rint")` | f32→fp16/bf16 支持 rint（RN-even，与 torch 转换语义一致） | Developer |
| 单边界变体 | `y = max(x, lo)` / `y = min(x, hi)` | `T.vmax(x_ub, lo_ub, y_ub)` / `T.vmin(x_ub, hi_ub, y_ub)` | 张量操作数 ✓（`T.vmax.md` §2.1 / `T.vmin.md` §2.1）；fp16 √ / fp32 √ / bf16 × | Developer |
| 整除向上取整 | `ceildiv(N, block_size)` | `T.ceildiv` | `docs/Tilelang.language/数学操作/T.ceildiv.md` | Developer |
| buffer 分配 | UB 片上缓冲 | `T.alloc_ub((block_size,), dtype)` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`（1D 支持 ✓） | Developer |

### 3.3 计算伪代码

> dtype 位置传参（模板 §3.3 规则：`T.Tensor((N,), dtype)` 不用 `dtype=` 关键字，避免 false alarm）；`N` 为编译期 Python int（工厂特化），**无运行时 `shape` 标量**（尾块由编译期常量 N 推导，同 `examples/elementwise/vec_add_1d.py` / lerp_tensor 先例）。

```python
def _make_clamp_tensor_kernel(N, dtype, has_min, has_max):   # E1：工厂签名（wrapper 契约）
    vector_cores = NPUUtils.get().get_aicore_num() * 2       # 纯 Vector 翻倍；实查见 §5.5

    # 编译期分支：has_min / has_max 选定变体（对应源 3 个 kernel 变体）
    n_in = (1 if has_min else 0) + (1 if has_max else 0)     # 输入 buffer 数（1 或 2）

    @tilelang.jit(out_idx=[1 + n_in], target="npuir")        # both: out_idx=[3]; 单边界: [2]
    def kernel(block_size):
        num_logical = T.ceildiv(N, block_size)
        num_kernels = min(num_logical, vector_cores)         # persistent 分核（R3）
        num_local_tasks = T.ceildiv(num_logical, num_kernels)  # 静态边界

        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            b0: T.Tensor((N,), dtype),     # lo（has_min 时）或 hi（仅 has_max 时）
            b1: T.Tensor((N,), dtype),     # hi（both 时；单边界变体下该参数被删除）
            y: T.Tensor((N,), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                # 1. UB buffer 全量无条件分配（R-7：条件 alloc 非法）
                x_ub = T.alloc_ub((block_size,), dtype)
                b0_ub = T.alloc_ub((block_size,), dtype)
                b1_ub = T.alloc_ub((block_size,), dtype)
                y_ub = T.alloc_ub((block_size,), dtype)
                x32 = T.alloc_ub((block_size,), "float32")   # fp32 中转（R2）
                b032 = T.alloc_ub((block_size,), "float32")
                b132 = T.alloc_ub((block_size,), "float32")
                y32 = T.alloc_ub((block_size,), "float32")
                need_cast = dtype != "float32"

                # 2. persistent 核内串行处理多个 tile（静态边界）
                for i in T.serial(num_local_tasks):
                    block_id = i * num_kernels + cid
                    if block_id < num_logical:
                        off = block_id * block_size
                        tail = T.min(block_size, N - off)     # 尾块截断（§6.4）

                        # 3. 数据搬入（GM → UB，尾块段拷贝，零起点 dst）
                        T.copy(x[off : off + tail], x_ub[0:tail])
                        T.copy(b0[off : off + tail], b0_ub[0:tail])
                        if has_max and has_min:
                            T.copy(b1[off : off + tail], b1_ub[0:tail])

                        # 4. 计算（整 buffer 向量链）
                        if need_cast:
                            T.vcast(x_ub, x32, round_mode="rint")
                            T.vcast(b0_ub, b032, round_mode="rint")
                            if has_max and has_min:
                                T.vcast(b1_ub, b132, round_mode="rint")
                            if has_max and has_min:
                                T.vclamp(x32, y32, b032, b132)      # min(max(x32,b032),b132)
                            elif has_min:
                                T.vmax(x32, b032, y32)              # max(x32, lo32)
                            else:
                                T.vmin(x32, b032, y32)              # min(x32, hi32)
                            T.vcast(y32, y_ub, round_mode="rint")
                            T.copy(y_ub[0:tail], y[off : off + tail])
                        else:
                            if has_max and has_min:
                                T.vclamp(x_ub, y_ub, b0_ub, b1_ub)
                            elif has_min:
                                T.vmax(x_ub, b0_ub, y_ub)
                            else:
                                T.vmin(x_ub, b0_ub, y_ub)
                            T.copy(y_ub[0:tail], y[off : off + tail])

        return main

    return kernel
```

> **实现注记**：
> 1. 上面的 `main` 签名按「both」变体示意（`b0=lo, b1=hi`）；单边界变体（`has_min xor has_max`）在工厂内以编译期分支生成**少一个输入参数**的签名（`main(x, bound, y)`，`out_idx=[2]`），与源三变体及 wrapper 调用点逐一对应（§9.1 R-5）。
> 2. `vclamp`/`vmin`/`vmax`/`vcast` 作用于**整 UB buffer**（非 tail 切片）——与 `T.vmin.md` §2.4 / hardtanh 先例一致；`[tail:block_size]` 陈旧段参与计算但**不落 GM**（搬出仅拷 `[0:tail]`），正确性无碍（`vec_add_1d.py` L34–L38 官方先例形态）。尾块为**动态值**，不可传给 v-op（会编译失败，hardtanh R-8）。
> 3. 所有 `T.alloc_ub` 无条件在 `T.Kernel` 顶层分配（条件 alloc 触发 `Undefined variable`，hardtanh R-7）；编译期 `if need_cast:` / `has_*` 分支只选择运行哪些 v-op 与哪些 `T.copy`。
> 4. `y_ub` 与 `y32` 独立分配（非原地复用），规避原地别名不确定性——UB 预算已充足（§4.5）；若 Stage 4 需压 UB，可按 PL-1.23 将 `y32` 复用为 `b132`（读取先于覆写）。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|----------|------|
| `T.Kernel(grid, is_npu=True)` | `examples/elementwise/vec_add_1d.py` L27；`T.vmin.md` §2.4 | ✅ 已验证（examples 先例） |
| `T.alloc_ub(shape, dtype)` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`（1D–2D）；`vec_add_1d.py` L28–L30；hardtanh 实现 | ✅ |
| `T.copy`（gm↔ub、切片、bf16） | `docs/Tilelang.language/内存操作/T.copy.md`（dtype 矩阵含 bf16 √；示例即切片拷贝） | ✅ |
| `T.vclamp`（张量边界） | `docs/Tilelang.language/比较操作/T.vclamp.md` §2.1/§2.2.2（min/max 可 scalar 或 tensor）+ §2.4 张量边界示例；**本任务设计期设备探针实测张量边界 bit-exact** | ✅ ✅（文档 + 设备实测） |
| `T.vmin(src1, src2, dst)` | `docs/Tilelang.language/比较操作/T.vmin.md` §2.1（src2 tensor or scalar）/ §2.4 | ✅ |
| `T.vmax(src0, src1, dst)` | `docs/Tilelang.language/比较操作/T.vmax.md` §2.1（双 tensor）/ §2.4 | ✅ |
| `T.vcast`（fp16/bf16↔f32） | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（rint） | ✅ |
| `T.min / T.ceildiv` | `docs/Tilelang.language/比较操作/T.min.md`；`docs/Tilelang.language/数学操作/T.ceildiv.md` | ✅ |
| `T.serial` + `block_id = i·K + cid` + `if` 边界 | `T.vmin.md` §2.4 示例（官方 persistent 分发模式）；hardtanh 实现 | ✅ |
| `@tilelang.jit(out_idx=[…], target="npuir")` | 源算子 `out_idx` 契约 + `examples/elementwise/example_elementwise_add.py` 的 `target="npuir"` | ✅ |
| NaN 传播（`VMaxOp`/`VMinOp`） | 设计期设备探针 `history_version/design_probe_clamp_nan.py`（PROBE PASS）+ `PL-1.23-vop-chain-inplace-alias`（scalar 边界设备实证） | ✅（设备实测） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No（源即一维 `ceildiv(N, block)`；NPU 一维 `T.Kernel(num_kernels)`） | 不涉及 |
| GPU 专用 API（`threads=` kwarg） | Yes（源 `T.Kernel(..., threads=threads_arg)`） | `is_npu=True` 取代（§0.5 舍弃项） |
| GEMM 非整除（M/N block 整数倍） | No（无 GEMM；非 GEMM 的 N 非整除由尾块 `tail` 处理，§6.4） | §6.4 尾块策略 |
| L0C 溢出 | No（纯 Vector，无 L0C；`block_M×block_N` 不适用） | 不涉及 |
| 物理核数限制（分核三要素） | Yes（逻辑核 256–65536 >> 40） | §5.5 persistent 分核 |
| bf16 Vector 算术缺失（`vclamp`/`vmin`/`vmax`） | Yes（manifest 契约含 bf16） | §0.6 R2：vcast fp32 中转 |
| Host 侧输入操作约束 | **部分涉及**（Op 层预广播物化 `contiguous()`） | §0.6 R4-A：预广播在 Op 层 wrapper（既有契约 O6 preserved）；kernel 内 host 侧零数据操作（仅 N/block_size 元数据）→ kernel 自身合规；Op 层物化的张力与升级路径见 §9.1 R-3 |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 并行 | `T.Kernel(grid, threads=)` CUDA 两级 | 一维 `T.Kernel(n, is_npu=True)` | §0.6 R1 + K11 折叠 |
| 元素并行 | `T.Parallel(threads, npt)` 标量循环 | v-prefix 整 buffer 向量 op（无逐元素循环） | §1.6.2 #1–#2 |
| 片上驻留 | `T.alloc_fragment`（寄存器） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| clamp 计算 | 标量 `T.min(T.max(v32, lo32), hi32)` | 融合 `T.vclamp`（前端 `npuir_max`+`npuir_min`） | §0.6 R1 |
| NaN 处理 | CUDA `T.max/T.min`（maxNum）→ 显式 3 步守卫 | NPU 原生 NaN 传播 → 无守卫 | §0.6 R5（设备探针实测） |
| bf16 计算 | fragment 全 dtype 算术（升 fp32 域） | `vclamp` 无 bf16 → `vcast` fp32 中转 | §0.6 R2 |
| grid 规模 | 任意（CUDA 调度器） | persistent `min(逻辑核, 40)` | §0.6 R3 / §5.5 |
| JIT 装饰 | `@tilelang.jit(out_idx=[…])` | `@tilelang.jit(out_idx=[…], target="npuir")` | K1 适配 |
| 工厂参数 | `(N, dtype, has_min, has_max, output_dtype, is_fp8, threads, npt)` | `(N, dtype, has_min, has_max)`；callable(block_size) | wrapper E1/E2（接口契约冻结，`clamp.py` 已按此实现） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/hardtanh/_make_hardtanh_kernel/_make_hardtanh_kernel.py` | **极高**（clamp 家族 + fp32 中转 + persistent + vclamp） | `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步、`T.alloc_ub` 全量无条件分配（R-7）、`T.vclamp`/`T.vmin`/`T.vmax` 整 buffer + `vcast(rint)`、尾块零起点切片、`TILELANG_ASCEND_MODE=Developer` |
| `examples/elementwise/vec_add_1d.py` | 高度相似（一维逐元素 + 动态尾块 + v-prefix） | `T.Kernel(n, is_npu=True)`、`T.alloc_ub`、`tail_size = T.min(...)` 尾块模式、`T.vadd` 用法 |
| `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel_DESIGN.md` | 高度相似（**三输入 elementwise + host 预广播 + persistent**，同 harness 迁移） | §0 迁移分析组织、host 预广播契约、persistent 分核三要素、UB 预算表、L0 测试计划结构 |
| `examples/TileOPs/tileops/kernels/elementwise/clamp/clamp.py` | 集成契约（wrapper，冻结） | `_DEFAULT_BLOCK_SIZE`、`out_idx`/调用点约定、`ascend_mode = "Developer"` |
| `examples/elementwise/example_elementwise_add.py` | 相似（jit 装饰与模式） | `@tilelang.jit(..., target="npuir")`、`TILELANG_ASCEND_MODE=Developer`、`T.ceildiv` 网格展开 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(N,)` | float16 / bfloat16 / float32 | Op 层已预广播物化的连续一维张量 |
| `lo` | `(N,)` | same_as(x) | 下界张量；仅当 `has_min`（源/契约中 `min` 为 Tensor） |
| `hi` | `(N,)` | same_as(x) | 上界张量；仅当 `has_max` |

`N` 为编译期常量（工厂按 N 特化，`lru_cache` 语义由 wrapper 承接）；代表 workload N ∈ {2²⁰, 2²⁴, 2²⁶, 2²⁸}（manifest `ClampFwdOp.workloads`）。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(N,)` | same_as(x) | `out_idx = 1 + n_in`（both `[3]` / 单边界 `[2]`）；Op 层 `view` 回 `broadcast_shapes(input, min, max)` |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub / b0_ub / b1_ub` | `(block_size,)` | 输入 dtype | UB | 三输入 tile 驻留（单边界变体少一个） |
| `y_ub` | `(block_size,)` | 输出 dtype | UB | 舍回 / 输出 tile |
| `x32 / b032 / b132` | `(block_size,)` | float32 | UB | fp16/bf16 路径 fp32 中转（R2；fp32 路径不分配使用） |
| `y32` | `(block_size,)` | float32 | UB | fp16/bf16 路径 clamp 结果（fp32 域） |

> fp32 路径的 UB buffer 命名为 `x_ub`/`b*_ub`/`y_ub`（皆为 fp32）；`x32`/`b*32`/`y32` 仅在 `need_cast` 分支使用（因禁止条件 alloc，全量分配但 fp32 路径不使用——冗余 UB 占用计入 §4.5 最坏路径预算）。

### 4.4 内存搬运路径

```
纯 Vector 单阶段（每 tile）：
GM[x]  ──T.copy(尾块截断)──> UB[x_ub]  ─┐
GM[lo] ──T.copy────────────> UB[b0_ub] ┼─ T.vclamp（整 buffer，前端 npuir_max+npuir_min）──> UB[y_ub] ──T.copy──> GM[y]
GM[hi] ──T.copy────────────> UB[b1_ub] ┘
fp16/bf16 变体（R2）：UB[x/b0/b1_ub:dt] ──vcast(rint)──> UB[x32/b032/b132:f32] ──vclamp──> UB[y32:f32] ──vcast(rint)──> UB[y_ub:dt]
无 L1/L0 参与纯 Vector 路径；无中间 GM 往返（tile 一次进出）；cast 全程 UB 内（零额外 GM 流量）
```

### 4.5 UB 内存预算

block_size 按 wrapper 默认（K11 承接：fp32 = 2048、fp16/bf16 = 4096）；**禁止条件 alloc** → 按 fp16/bf16 路径（7 buffer）为最坏预算：

| Buffer | fp32 @2048 | fp16/bf16 @4096 |
|--------|-----------|-----------------|
| `x_ub` + `b0_ub` + `b1_ub`（输入 dtype） | 3 × 2048×4B = 24576 B | 3 × 4096×2B = 24576 B |
| `y_ub`（输出 dtype） | 2048×4B = 8192 B | 4096×2B = 8192 B |
| `x32` + `b032` + `b132`（f32 中转，仅非 fp32 使用但**无条件分配**） | 3 × 2048×4B = 24576 B（与输入同 dtype、未使用） | 3 × 4096×4B = 49152 B |
| `y32`（f32 结果，同上） | 2048×4B = 8192 B（未使用） | 4096×4B = 16384 B |
| **手工总计（无条件全量分配）** | **65536 B (64KB)** | **98304 B (96KB)** |
| × auto-multi-buffer 膨胀（平台系数，见注） | ~109KB | ~167KB |
| 目标平台 UB 容量 | 196608 B (192KB) | 196608 B (192KB) |

- **结论**：两路径均在 UB 上限内（fp32：手工 64KB / 膨胀 ~109KB；fp16/bf16：手工 96KB / 膨胀 ~167KB）。**注**：`CONST-capacity-910B2C` 记 auto-multi-buffer 膨胀 ~1.7x（910B2C 结构实测；9362 未重验），按 1.7x 计算 fp16/bf16 路径 ≈167KB < 192KB（余量 ~25KB，偏紧）；**显式逐 buffer alloc 结构实测可达 0.98x**（同条目第六轮反例）——本设计为逐 buffer 显式 `T.alloc_ub` 结构，实际需求预计 < 96KB×1.1。若 9362 编译报 UB 溢出，降 block_size（4096 → 2048）或按 §3.3 注记 4 复用 `y32`（省 16KB）。fp32 路径的 4 个 `*32` 中转 buffer 因禁止条件 alloc 而空占 ~32KB（可通过 dtype 特化的 `xc` host 别名按角色复用压到 4 buffer，hardtanh 同款结构；Stage 3 按编译实测决定）。
- block_size 上界（按无条件全量分配的手工预算反推）：fp16/bf16 ≤ 8192（= 192KB ÷ (4×2 + 4×4) B/elem = 192KB ÷ 24B）；fp32 ≤ 6144（= 192KB ÷ 32B/elem = 8 buffer × 4B）。默认值（4096 / 2048）余量充足。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| `N_total` | 无 TIR 动态符号——工厂闭包按 N **编译期特化**（源算子同款模式，wrapper 惰性构建 + 缓存） | 不同 Op 实例 N 不同（1 ~ 2²⁸+），每 N 触发一次 JIT 特化；kernel 边界全静态 |

> 与 hardtanh 的差异：hardtanh 用运行时 `shape: T.int32` 标量以复用同一编译内核处理 ≤N 的运行时长度；本算子沿 lerp_tensor 先例**不引入运行时标量**（`tail = T.min(block_size, N − off)`，N 编译期常量），使 kernel 入参个数与 wrapper 调用点 `_compiled_fn(x, lo, hi)` 完全一致（见 §9.1 R-5）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1 + n_in], target="npuir")
# out_idx = [3]（both，main(x, lo, hi, y)）或 [2]（单边界，main(x, bound, y)）——承接源算子契约
# target="npuir"：K1 适配（GPU 版无 target，默认 CUDA；example_elementwise_add.py 先例）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 仅逐元素 max/min 比较选择（+bf16/fp16 cast），无 matmul、无规约、无数据重排 → 全部计算落 Vector 单元，仅需 GM↔UB 搬运（决策树 §2 纯 element-wise 分支）；无 Cube 段 → **CV 融合不适用**。

### 5.2 Block 划分

```python
block_size = 4096   # fp16/bf16 默认（= GPU threads(512) × npt(8)，K11/wrapper _DEFAULT_BLOCK_SIZE 承接）
block_size = 2048   # fp32 默认（= GPU threads(512) × npt(4)）
# 选择理由：
# ① 尾轴 32B 对齐：4096×2B = 8192B、2048×4B = 8192B，均 32B 整除 ✓
# ② 向量宽度整除：fp16×8 lane、fp32×4 lane 均整除 block_size → 尾 lane 浪费 0%
# ③ UB 预算：fp32 64KB / fp16·bf16 96KB（无条件全量分配，§4.5）≤ 192KB ✓
# ④ tile 搬运粒度 8KB/buffer，兼顾 copy 带宽利用率与 persistent 任务粒度
# ⑤ GPU 语义承接：与源 Kernel 类默认 block_size（threads×npt）一致，跨平台行为可对照
num_logical = T.ceildiv(N, block_size)      # 逻辑 tile 数
```

一维划分（无 block_M/block_N 之分）——本算子为展平一维逐元素，GEMM 类分形/非整除策略不适用。

### 5.3 约束分析

- **对齐约束**: block_size×elem_bytes = 8KB（32B 的 256 倍）✓；GM 侧 tile 起点 `off = block_id × block_size` 恒为 8KB 对齐 ✓（尾块 `tail` 段长度非 32B 倍数时的拷贝行为依赖 `vec_add_1d.py` 先例形态，风险点 §9.2 并入 L0 测试）
- **UB 容量**: 见 §4.5，两 dtype 路径 64–96KB ≤ 192KB ✓（multi-buffer 膨胀系数为 9362 未重验项，列入 §9.1 R-6）
- **L0 容量**: 不适用（纯 Vector，无 L0C）

### 5.4 注意事项

- **非整除处理**：尾块 `tail = T.min(block_size, N − off)` 截断拷贝（§6.4）；manifest workload 的 N 均为 2 的幂、被 block_size 整除（无尾块），尾块路径服务任意 broadcast shape 的动态 N（如 `test_clamp_tensor` 的 `(4,8)` → N=32，`(4,)` → N=4）。
- block_size 是 wrapper config 可调参数（`{"block_size": ...}`），kernel 不硬编码——Stage 4 可扫描调优（§1.6.3 候选 #1 已锁定轴向，调优仅粒度维度）；上界见 §4.5。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1 三要素表（依据 docs/开发指南.md §3.3）。

- **① 逻辑核数计算**：`num_logical = ceildiv(N, block_size)`（一维展平，无 N 方向二维网格），按 manifest `ClampFwdOp.workloads`：

  | workload | N | dtype（block_size） | num_logical |
  |----------|---|--------------------|-------------|
  | smoke-1m（1024×1024） | 2²⁰ | fp32（2048） | 512 |
  | smoke-1m | 2²⁰ | fp16/bf16（4096） | 256 |
  | elementwise-16m（4096×4096） | 2²⁴ | fp32（2048） | 8192 |
  | elementwise-16m | 2²⁴ | fp16/bf16（4096） | 4096 |
  | elementwise-64m（8192×8192） | 2²⁶ | fp16/bf16（4096） | 16384 |
  | elementwise-256m（16384×16384） | 2²⁸ | fp16/bf16（4096） | 65536 |

- **② 物理核数依据**：目标设备物理核数 **实查**（2026-10-13 本会话，Ascend910_9362）：

  ```python
  from tilelang.utils.npu_utils import NPUUtils
  NPUUtils.get().get_aicore_num()        # 实际返回 20
  NPUUtils.get().get_aivector_core_num() # 实际返回 40（= aicore × 2）
  ```

  → **纯 Vector 物理核数 = 20 AI Core × 2 = 40 Vector Core**（core-split-strategy.md §1 要素②「纯 Vector 算子核数翻倍」；与 `CONST-aicore-910_9362` 实查值一致）。注：`pattern-library/constants.md` 的 `CONST-aicore-910B2C` 记 24 核，**绑定 Ascend910B2C，跨 SoC 不适用**——以本机实查 20/40 为准。工厂闭包每次 JIT 特化时实查（§3.3 伪代码），**无硬编码**。

- **③ 规模判定与分核方案**：**极大规模**——全部 manifest workload 的 num_logical（256–65536）均 > 物理 Vector 核数 40，且无法通过增大 block_size 缩减到 ≤ 40（UB 容量上限：fp16/bf16 block ≤ 8192 → 2²⁰ 仍需 128 核 > 40；256M 规模更不可能）。按 core-split-strategy.md §1 要素③「极大规模——固定启动内核数 = 物理核数，核内 `T.serial` 串行处理多个逻辑块任务」：
  - 固定启动内核数 `num_kernels = min(num_logical, 40)`（编译期常量；小 N 场景自动退化为 `num_kernels = num_logical ≤ 40`，同一模板全覆盖，工厂层静态判定无运行时开销）
  - 核内串行：`for i in T.serial(num_local_tasks)`，`num_local_tasks = ceildiv(num_logical, num_kernels)` —— **静态边界**（N / block_size / 40 均为编译期常量）
  - 任务映射：`block_id = i × num_kernels + cid`（grid-stride），`if block_id < num_logical` 屏蔽尾部越界
  - 各 workload 任务分布：1M fp16 → 40 核 × ≤7 任务；16M fp16 → 40 × ≤103；64M → 40 × ≤410；256M → 40 × ≤1639——核启动次数固定 40 次（对比超发方案 65536 次串行调度），**负载不均 ≤ 1 任务/核**（最坏 1639 vs 1638 → 0.06% 可忽略）
  - **失败信号**（供 Stage 3/4 对照）：① `num_local_tasks` 依赖运行时值（非静态）→ 编译失败或违反标准；② msprof 显示核串行排队/启动开销占比异常 → 检查 num_kernels 是否误用逻辑核数；③ 负载不均（末核任务数 2 倍）→ 确认 `min()` 取值与 `ceildiv` 上界；④ 40 与本机 `get_aivector_core_num()` 不符 → 跨设备部署需重算（无硬编码）
---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化/消除——本表循环仅 block/tile 级调度循环，无任何逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| tile 级（核间） | persistent 并行 | `T.Kernel(num_kernels, is_npu=True)` | §5.5 分核：40 Vector 核常驻，每核串行消化多个 tile |
| tile 级（核内多任务） | 串行 | `T.serial(num_local_tasks)`（**静态边界**） | R3 persistent 标准结构；摊薄核启动开销 |
| 元素级 | 向量化 | `T.vclamp`（+`T.vcast` cast 链）整 buffer | §1.6.2 #1–#3；无 `T.Parallel` 逐元素循环（与源码差异：R1） |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (cid, _):        # 核间：persistent 40 核
    x_ub, b0_ub, b1_ub, y_ub = T.alloc_ub(...), ...          # §3.3 完整版（+ f32 中转 buffer）
    for i in T.serial(num_local_tasks):                      # 核内：静态边界串行
        block_id = i * num_kernels + cid
        if block_id < num_logical:
            # tile 处理（T.copy ×(2/3) → vcast/vclamp → T.copy ×1，全向量化）
            ...
```

### 6.3 流水线优化

**不使用 T.Pipelined**。理由：单阶段 load→compute→store 算子（每 tile 一次进出），无 K 维迭代累加结构可供双缓冲；纯 Vector 简单结构下流水的收益上限低——`PL-1.6-copy-floor` 实测该同族结构（3 载入 + 向量链 + 1 写出）在 ≥16M 档 auto multi-buffer 已把全部向量 pass 隐藏于 MTE2 窗口内（delta ≤1.1µs），流水收益≈0。若 Stage 4 的 msprof 显示小 N 档（1M）copy/计算暴露，可试验多 tile 双缓冲（属参数级调优，不改变设计结构）。

### 6.4 尾块处理

`N % block_size ≠ 0` 时（任意 broadcast shape 的动态 N，如 `(4,8)`→N=32、`(4,)`→N=4；manifest workload 全整除不触发）：

```python
off = block_id * block_size
tail = T.min(block_size, N - off)          # 最后一个 tile 的有效长度
T.copy(x[off : off + tail], x_ub[0:tail])  # 零起点 dst 切片（规避 UB 非零起点对齐陷阱）
...（v-prefix 整 buffer 操作；UB 中 [tail:block_size] 陈旧段不落 GM）
T.copy(y_ub[0:tail], y[off : off + tail])
```

模式来源：`examples/elementwise/vec_add_1d.py` L31–L38（官方已验证形态）。不采用 host 侧 padding（违反 host 禁改输入约束，ascend-constraints.md §4）；不采用 Kernel 动态 block（N 已编译期特化，`tail` 标量一次/tile 开销可忽略）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

无手动同步点。依据：① 每 tile 计算完全独立、无核间数据依赖（无 workspace、无跨核归约）→ 不需要 `T.sync_block_set/wait`；② tile 内 `T.copy → v-prefix → T.copy` 的顺序依赖由 Developer 模式自动同步保证（`T.vclamp.md` / `T.vmin.md` 等 v-prefix 文档均为无显式同步示例；`clamp.py` wrapper `ascend_mode = "Developer"`）；③ persistent 核内 `T.serial` 顺序执行天然串行。

### 7.3 pass_configs 配置

无特殊 pass 配置（默认管线）。环境变量：`TILELANG_ASCEND_MODE=Developer`（§2.3；`example_elementwise_add.py` 先例）。

> **CV 融合（§8.2 模板项）**：**不适用**——本算子为纯 Vector（无 Cube 段），不存在 Cube↔Vector 协作或跨引擎同步需求。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 §0.1 源算子语义为唯一依据（优先移植源仓参考实现/测试基准 `ClampTensorTest.ref_program`），**不复刻 §0.6 的 NPU 算法**（不复用 vclamp/fp32 中转/persistent 结构/无守卫形态）——保证验证独立性。

```python
def golden_clamp_tensor(x, lo, hi):
    """Tensor-bound clamp 参考实现（源算子语义 + 契约测试基准）。

    契约（examples/TileOPs/tests/ops/test_special_elementwise.py
    ClampTensorTest.ref_program）：
        torch.clamp(x.float(), lo.float(), hi.float()).to(x.dtype)
    —— fp32 opmath 域 clamp 后单次舍回；输入为已广播对齐的同 dtype 张量。
    NaN 语义：任一操作数 NaN → 输出 NaN（torch 契约，equal_nan 逐位比对）；
    lo > hi → 结果取 hi。
    """
    return torch.clamp(x.float(), lo.float(), hi.float()).to(x.dtype)
```

### 8.2 精度标准

（承接源仓 `_get_tolerances`，`examples/TileOPs/tests/ops/test_special_elementwise.py` L91–L96）

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 1e-5 | 1e-5 | 同式同 fp32 域，预期 bit 级一致 |
| float16 | 1e-3 | 1e-3 | fp32 中转链与契约 golden 同域，预期 bit 级一致 |
| bfloat16 | 1.6e-2 | 1.6e-2 | API 强制 fp32 中转，契约同域，预期 bit 级一致 |
| NaN 用例 | 0.0 | 0.0 | `test_clamp_tensor_nan_propagation`：`equal_nan=True` **逐位断言** |

### 8.3 L0 门槛测试计划

> L0 为 Stage 3 开发的精度门槛（golden 对比必须全过）；完整 L1/L2/Boundary 分层套件由 `tilelang-op-develop` 展开。

| # | 用例 | 输入 | 通过标准 |
|---|------|------|----------|
| L0-1 | smoke 全 dtype（both 变体） | N=2²⁰，fp16/bf16/fp32，`x~randn`、`lo~randn-0.5`、`hi~randn+0.5` | vs golden，§8.2 容差 |
| L0-2 | 大规模抽检 | N=2²⁴ fp16 + fp32（elementwise-16m 代表） | vs golden |
| L0-3 | **NaN 传播** | x/lo/hi 分别在若干位置置 NaN（fp32 + fp16，契约用例形态） | vs golden，`atol=rtol=0, equal_nan=True`（**逐位**） |
| L0-4 | **`lo > hi`** | `lo` 全域大于 `hi`（如 lo=2, hi=1）及逐元素混合（`mn=randn-0.5, mx=randn+0.5`） | vs golden（结果取 `hi`） |
| L0-5 | 单边界变体 | `has_min only`（`torch.clamp(x, lo, None)`）/ `has_max only`（`torch.clamp(x, None, hi)`） | vs golden + 调用点 `_compiled_fn(x, bound)` 无参数量错 |
| L0-6 | **广播 / 0-dim 边界** | `(4,8)/(1,8)/(4,1)` 三路广播；`(4,8)/()/()` 0-dim 边界；`min>max` 广播 | vs golden（Op 层物化路径，§0.6 R4-A） |
| L0-7 | 尾块非整除 | N ∈ {1000, 4097, 5000}（非 block 整除、非 32B 对齐尾段），fp16/fp32 | vs golden（覆盖 §6.4/§9.2 R-1） |
| L0-8 | 极小 shape | N ∈ {1, 7, 32}（单 tile 非满 / num_logical=1 退化） | vs golden |
| L0-9 | 端点 / 极值 | ±0、±inf、次正规 1e-45、65504、±88.8（fp32/fp16/bf16） | vs golden |
| L0-10 | persistent 边界 | N 使 num_logical ∈ {39, 40, 41}（核数切换边界） | vs golden + 无越界（末 tile 屏蔽正确） |

---

## 9. 风险点与注意事项

### 9.1 已知约束

- **R-1 尾块非 32B 对齐拷贝**：`tail` 段长度非 32B 倍数时（如 N=1000）的 `T.copy` 行为依赖 `vec_add_1d.py` 同款零起点切片先例；L0-7 强制覆盖；若复现按 pattern-library「UB dst 非零起点」陷阱条目排查（本设计 UB dst 恒零起点已规避主形态）。
- **R-2 设备 NaN 语义**：采纳项 #1 依赖 `VMaxOp`/`VMinOp` 传播 NaN。已由**设计期设备探针**（`history_version/design_probe_clamp_nan.py`，PROBE PASS）与 PL-1.23 第 4 证双证；仍保留 **L0-3 逐位 NaN 用例**作为 Stage 3 回归门（工具链升级后须重验，`kb_stale_check.py`）。
- **R-3 host 侧广播物化与 host 约束的张力（重要）**：Op 层 `_expand_flat` 的 `t.expand(out_shape).contiguous().view(-1)` 会**物化**被广播的操作数（`contiguous()` 非纯 stride/shape 视图操作）。这是**源/GPU harness 既有 Op 契约（`clamp.py` O6「preserved unchanged」）**，不在本 kernel 的重设计范围；kernel 内 host 侧零数据操作。影响：manifest 4 workload 三输入同 shape → `expand` 视图 + `contiguous` 无拷贝 → **零代价**；真广播仅见契约测试微 shape（`(4,8)/(1,8)/(4,1)`、`()`）。**升级路径**：若 Stage 2/3 判定须严格符合 ascend-constraints.md §4「仅视图操作」，则须改 Op/wrapper 契约（把广播下移至核内 2D+vbrc，见 §0.6 R4-B）——该改动超出 Stage 1 kernel 设计范围，需 conductor 决策；本设计已把 R4-B/C 的量化评估留在 §0.6 备查。
- **R-4 物理核数 20/40 为本机实查值**（2026-10-13）：跨设备部署时 `num_kernels` 须随 `NPUUtils.get().get_aicore_num()` 重算——工厂闭包每次 JIT 特化时实查（§3.3 伪代码），无硬编码。
- **R-5 kernel 输出约定 / 调用点逐算子核对（VP-2026-0189 / VP-2026-0133 / VP-2026-0119）**：本 kernel **不引入运行时 `shape: T.int32` 标量**（N 编译期特化、静态尾块），故入参个数 = `total_params − len(out_idx)` 恰为 wrapper 传入实参数（both：`main(x,lo,hi,y)` out_idx=[3] → wrapper `_compiled_fn(x, lo, hi)`；单边界：`main(x,bound,y)` out_idx=[2] → `_compiled_fn(x, lo)` / `(x, hi)`）。**Stage 3/5 须按 kernel 实际签名逐个核对**（同 family 内逐算子不同的输出约定不适用本算子）；若实现期改为引入运行时 `shape` 标量，则须同步按 VP-2026-0133 修正 wrapper 调用点。
- **R-6 UB multi-buffer 膨胀系数为 9362 未重验项**：`CONST-capacity-910B2C` 的 ~1.7x 系 910B2C 实测、9362 未重验；本设计逐 buffer 显式 alloc 结构预计更低（同条目反例 0.98x）。若编译 UB 溢出，降 block_size 或复用 `y32`（§4.5）。
- **R-7 persistent 结构依赖 N 编译期特化**：动态 N 场景（每次新 shape）触发重编译，语义正确但首次调用有 JIT 延迟——与源算子 `lru_cache` 行为一致，非回退。
- **R-8 bf16 无原生 Vector 算术**（`T.vclamp`/`T.vmin`/`T.vmax` §2.2.1 bf16 ×）：fp32 中转为唯一路径；cast 指令 ~4 次/tile（UB 内，零额外 GM 流量），带宽口径不变——bf16 性能预期略低于 fp16，属 API 硬约束。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 交换 `min`/`max` 顺序 | 误写 `max(min(x,hi),lo)` | `lo > hi` 时结果 = `lo`（应为 `hi`）→ 精度失败（6.3 万点违反） | 严格用 `T.vclamp`（前端 max→min）或 `T.min(T.max(...))`；L0-4 强制覆盖 |
| 保留 NaN 守卫或误以为需自建 | 照抄源守卫 | op 数 8 而非 2（性能损失、无正确性收益） | §1.6.1 #1：NPU 原生传播，直接 `T.vclamp`；L0-3 验证 |
| 尾块非 32B 对齐拷贝异常 | L0-7 的 N=1000/4097（tail 段长度非 32B 倍数） | `T.copy` 报错或数据错乱 | `vec_add_1d.py` 同款零起点切片；L0-7 强制覆盖 |
| `T.copy` 静默跨 dtype 转换 | copy 的 src/dst dtype 不一致时被静默 cast | 精度路径失控 | 设计中所有 `T.copy` 严格同 dtype（fp16/bf16↔fp32 一律显式 `T.vcast`）；实现期 lint 自查 |
| v-op 传入动态 tail 切片 | 把 `x_ub[0:tail]` 传给 `T.vclamp`/`T.vcast` | 编译失败（hardtanh R-8） | v-op 一律作用于**整 buffer**，仅 `T.copy` 用 tail 切片（§3.3 注记 2） |
| 条件 `T.alloc_ub` | 在 `if need_cast:` 内 alloc | `Undefined variable` 编译失败（hardtanh R-7） | 所有 UB buffer 无条件在 `T.Kernel` 顶层分配（§3.3 注记 3） |
| `out_idx` / 入参个数与 wrapper 不符 | both 变体误写 `out_idx=[2]`，或引入未同步的 `shape` 标量 | `Expected K inputs, got M` 或输出错位 | §3.3/§9.1 R-5 逐分支对齐（both [3]、单边界 [2]）；Stage 3 逐签名核对 |
| num_kernels 误用逻辑核数 | 照抄 `T.ceildiv(N, block_size)` 作 grid | 65536 核串行调度、启动开销剧增 | §5.5 分核方案为准（`min(num_logical, 40)`） |
| block_size 超 UB 上限 | wrapper config 传入 > 8192（fp16/bf16）/ > 12288（fp32） | UB 溢出编译失败 | §4.5 上界文档化；实现期 assert |
| bf16 直接调 `T.vclamp` | 未走 fp32 中转 | 编译失败（dtype 矩阵 bf16 ×） | §0.6 R2 强制 `vcast` 中转 |

### 9.3 特殊场景处理

- **三输入不同 broadcast shape**（如 `(4,8)/(1,8)/(4,1)`、`(4,8)/()/()`）：Op 层预广播物化为 `(N,)`——kernel 无感知（§0.6 R4-A；契约测试 `test_clamp_tensor` 的 full 用例覆盖该路径）
- **`lo > hi`**：由 `min(max(...))` 顺序天然给出 `hi`（§0.1 / 探针 [3]），无需特殊分支
- **单边界（`min`/`max` 之一为 `None`）**：构造期 `has_min`/`has_max` 选定变体（`T.vmax`/`T.vmin` 单侧），wrapper 调用点相应省略实参（§9.1 R-5）
- **`N == 0` / 空张量**：`ClampFwdOp` 以 `out_shape or (1,)` 兜底或上层拦截，kernel 不触达（与源一致）
- **混合 dtype 输入**：Op 层 `forward` 已拒绝（dtype 一致性校验）——kernel 无需防御
- **fp8**：不在 manifest `ClampFwdOp` 契约内（源 fp8 分支不迁移），wrapper `SUPPORTED_DTYPES` 构造期拒绝

---

## 10. 交付清单

### 10.1 目录结构

```
examples/clamp/_make_clamp_tensor_kernel/
├── _make_clamp_tensor_kernel.py   # 算子实现（kernel 工厂 + golden + L0~Boundary 测试入口）
├── DESIGN.md                      # 本设计文档
├── verify_equiv.py                # 等价性机器验证（D-1，23/23 EQUIV_PASS）
├── history_version/
│   └── design_probe_clamp_nan.py  # 设计期设备探针（D-3，NaN 传播前提，PROBE PASS）
├── RETROSPECTIVE.md               # Stage 1 复盘（skill Phase 8）
└── perf_opt/                      # （Stage 4 预留）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本文档 |
| `verify_equiv.py` | ✅ 已完成并实跑 | §1.6.1 机器验证（23 项全 EQUIV_PASS） |
| `history_version/design_probe_clamp_nan.py` | ✅ 已完成并实跑 | §1.6.1 采纳项 #1 前提的设备实测（PROBE PASS） |
| `_make_clamp_tensor_kernel.py` | ⬜ 待实现（Stage 3） | E1/E2 契约：`_make_clamp_tensor_kernel(N, dtype, has_min, has_max)` → callable(block_size) → `@T.prim_func main(...)`（三变体，§3.3）；集成目标 wrapper `examples/TileOPs/tileops/kernels/elementwise/clamp/clamp.py`（Stage 5 接入） |
| golden | 随实现文件交付（§8.1） | `torch.clamp(x.float(), lo.float(), hi.float()).to(dtype)` |
| `RETROSPECTIVE.md` | ✅ 已追加 | skill Phase 8 |

### 10.3 命名规范

- 项目目录：`clamp`（conductor 指定）；算子目录：`_make_clamp_tensor_kernel`（源算子函数名，harness 提取件同名）
- 实现文件：`_make_clamp_tensor_kernel.py`；测试入口内嵌（`if __name__ == "__main__"`，`vec_add_1d.py` / hardtanh 先例）
- 命令行：`python _make_clamp_tensor_kernel.py --level all`（L0 阻塞，L1/L2/Boundary 记录不阻塞）

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）+ 设计期探针（design_probe_clamp_nan.py）
2. ⬜ Golden 函数（§8.1，验证基准）
3. ⬜ 算子实现（`_make_clamp_tensor_kernel.py`：fp32 主路径 → fp16/bf16 fp32 中转分支 → 三变体 → L0 测试全绿）
