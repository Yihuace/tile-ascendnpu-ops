# _make_hardtanh_kernel 算子设计文档（迁移）

> **项目**：`hardtanh` ｜ **算子目录**：`examples/hardtanh/_make_hardtanh_kernel/` ｜ **编程模式**：Developer ｜ **target**：`npuir`
>
> **任务性质**：迁移（migration-harness）。把 GPU TileLang 工厂函数 `_make_hardtanh_kernel`（`y = min(max(x, min_val), max_val)`，非 fp8 分支**已在 fp32 域**做两侧 clamp 后单次舍回）重设计为 NPU（`target="npuir"`）实现。
>
> **本版核心决策**：① **计算域 = fp32 中转**（**保留源既有行为**并对齐契约 golden）。与同族 `leaky_relu`（源为输入 dtype 域、需改为 fp32）不同，**hardtanh 源码非 fp8 分支本身即 `v32 = T.cast(val, "float32")` → `T.min(T.max(v32, lo), hi)` → `T.Cast(val.dtype, …)`**，即源与契约 `F.hardtanh(x.float(), lo, hi).to(dtype)` 都已用 fp32 域；且 **bf16 因 `T.vclamp`/`T.vmin`/`T.vmax` 的 dtype 矩阵不含 bf16 而 API 强制 fp32 中转**。故本算子**无计算域争议**，直接承接源结构。② **算法级优化 = 编译期边界特化（恒等折叠）**：`min_val`/`max_val` 为构造期编译期常量，按取值在 host 侧裁剪向量链——`lo=-inf → 仅 vmin`（1 op）、`hi=+inf → 仅 vmax`（1 op）、两侧皆 inf → 纯 `T.copy`（0 op）、`lo==hi → 常量填充`（0 数据 op）；默认 `(-1.0, 1.0)` 两侧有限**不触发折叠**（诚实记录，2 op）。折叠经 `verify_equiv.py` 机器验证逐位一致（20/20 `EQUIV_PASS`）。③ **并行/内存重设计**：GPU `fragment + block×thread + 256 threads/block` → 一维 `T.Kernel(40, is_npu=True)` + 核内 `T.serial` 网格跨步 + UB 整块 `T.copy` + 整 buffer 向量 op。
>
> **源优化意图承接**：源 #1 `register_copy`（fragment load→compute→store）→ NPU `GM→UB→GM` 整块 `T.copy` + 整 buffer v-op；源 #2 block×thread 两级并行 → 一维 `T.Kernel(40, is_npu=True)` + 核内网格跨步；源 #3 fp32 计算域 → **保留**（契约对齐 + bf16 强制）；源 #4 `npt` 对齐 → K11 `block_size` 折叠 + 32B 对齐复算；源 #5 无 autotune → 保留；源 #6 `min(max)` 两 op 无分支 → `T.vclamp`（前端展开 `npuir_max`+`npuir_min`）；源 #7 `lru_cache` 常量特化 → 保留并强化为边界折叠特化。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织：先彻底读懂源算子（0.1–0.4），再做算法调研（Phase R，结论见 §1.6.0），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。

**源算子路径（源真相）**：`/home/developer/TileOPs/tileops/kernels/elementwise.py`

- 具体算子：工厂 `_make_hardtanh_kernel(N, dtype, min_val, max_val, output_dtype=None, is_fp8=False, threads=256, npt=8)`（L2564–L2608）+ `HardtanhFwdKernel(ParametricUnaryKernel)`（L2611–L2624，`min_val`/`max_val` compile-time 特化）：

  ```python
  @functools.lru_cache(maxsize=32)
  def _make_hardtanh_kernel(N, dtype, min_val, max_val, output_dtype=None,
                            is_fp8=False, threads=256, npt=8):
      out_dtype = output_dtype or dtype
      block_size = threads * npt
      if is_fp8:                                     # fp8 分支（本迁移不涉及）
          @tilelang.jit(out_idx=[1])
          def kernel(threads_arg, npt_arg):
              @T.prim_func
              def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
                  with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                      for i, j in T.Parallel(threads_arg, npt_arg):
                          idx = (bx * threads_arg + i) * npt_arg + j
                          if idx < N:
                              val = x[idx]
                              lo = T.cast(min_val, "float32")
                              hi = T.cast(max_val, "float32")
                              v32 = T.cast(val, "float32")
                              y[idx] = T.Cast(out_dtype, T.min(T.max(v32, lo), hi))
              return main
      else:                                          # 非 fp8 分支（本迁移对象）
          @tilelang.jit(out_idx=[1])
          def kernel(threads_arg, npt_arg):
              @T.prim_func
              def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
                  with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                      x_reg = T.alloc_fragment((block_size,), dtype)
                      y_reg = T.alloc_fragment((block_size,), dtype)
                      T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                      for i, j in T.Parallel(threads_arg, npt_arg):
                          val = x_reg[i * npt_arg + j]
                          lo = T.cast(min_val, "float32")          # ★ fp32 域边界
                          hi = T.cast(max_val, "float32")
                          v32 = T.cast(val, "float32")             # ★ fp32 中转
                          y_reg[i * npt_arg + j] = T.Cast(val.dtype, T.min(T.max(v32, lo), hi))
                      T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])
              return main
      return kernel
  ```

- 基类与模板：`ParametricUnaryKernel`（GPU 端共享基类，提供 `_builder_fn`/`_builder_args`/`default_config`/`forward`；NPU 侧不提供该基类，集成 wrapper `hardtanh.py` 已内联该逻辑）。
- 提取产物（脚手架）：`examples/TileOPs/tileops/kernels/elementwise/hardtanh/_hardtanh_fwd_kernels.py`（Pattern B）；集成 wrapper：同目录 `hardtanh.py`（K11：`block_size` = 1024 fp32 / 2048 fp16·bf16；`npub::hardtanh_fwd`；NPU 工厂签名 `_make_hardtanh_kernel(N, dtype, min_val, max_val, block_size=...)`，返回 `main(x, y, shape)`，`out_idx=[1]`）。
- **wrapper 明确契约（本迁移必须遵守）**：`hardtanh.py` docstring 载明「The fp32 transit in the extracted non-fp8 branch … **is semantically required and must be preserved** by the NPU re-implementation: it reproduces `torch.nn.functional.hardtanh`'s fp32 clamping (the bounds are compared in fp32 before the downcast back to the input dtype)」——本设计承接该要求（§0.6）。

### 0.1 源算子语义（做什么）

**数学语义**：`y = clamp(x, min_val, max_val) = min(max(x, min_val), max_val)`，逐元素（manifest `HardtanhFwdOp.ref_api = torch.nn.functional.hardtanh`，`examples/TileOPs/tileops/manifest/elementwise.yaml` `HardtanhFwdOp` 段）。

**契约语义 = 测试 golden `F.hardtanh(x.float(), -1.0, 1.0).to(dtype)`**（`examples/TileOPs/tests/ops/test_activation.py` L209–L225，`HardtanhTest.ref_program`；注释明示「Every golden below uses the same default bounds (-1.0, 1.0) that the op is constructed with」）。容差：fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2（`_get_tolerances`，同族 leaky_relu 一致）。

**计算域语义（本算子无争议点，与 leaky_relu 对照）**：

| 域 | 表示 | 与契约 golden 的一致性 |
|----|------|------------------------|
| **契约 golden** | `F.hardtanh(x.float(), lo, hi).to(dtype)` | 基线：fp32 域 clamp，末端单次舍回原 dtype |
| 源 GPU 非 fp8 分支 | `v32 = T.cast(val, "float32")`；`lo/hi = T.cast(bound, "float32")`；`T.min(T.max(v32, lo), hi)`；`T.Cast(val.dtype, …)` | **与契约 golden 逐位一致**（同为 fp32 域 clamp + 单次舍回）——源与契约同域，**无需域替换** |

→ 与同族 `leaky_relu`（源为输入 dtype 域，须改为 fp32 中转）**不同**：hardtanh 源**本身即 fp32 中转**（这也正是 wrapper docstring 强调「must be preserved」的原因）。本迁移**保留 fp32 计算域**。

**dtype 语义（迁移必须保持的关键点）**：源非 fp8 分支全程在 fp32 域做 `max`/`min`，仅在末端 `T.Cast(val.dtype, …)` 一次舍回输入 dtype。即 fp16/bf16 的 clamp 在 fp32 域完成、单次舍回——与 golden 的 fp32 opmath 语义一致。**bf16 额外强制**：NPU 侧 `T.vclamp`/`T.vmin`/`T.vmax` 的 dtype 矩阵均不含 bf16（各 op doc §2.2.1），bf16 必须 `T.vcast` 升 fp32 计算、末端舍回（本设计因此对 fp16·bf16 统一走 fp32 中转，见 §0.6/§0.5）。

**`min_val` / `max_val` 语义**：compile-time 标量（`HardtanhFwdKernel._builder_args → (min_val, max_val)`；工厂经 `functools.lru_cache` 按参数特化）。Op 层参数默认 `min_val=-1.0` / `max_val=1.0`（manifest `params.min_val.default=-1.0`、`params.max_val.default=1.0`）；manifest `shape_rules: min_val <= max_val`。本设计据此把两边界当**编译期 Python 常量**贯穿，并按常数取值做编译期特化（§0.6 R2 / §1.6.1 #1）。

**`inplace` 语义**：host 侧处理，不在 kernel 内（manifest `params.inplace` 默认 false）；迁移保持：kernel 只处理 1-D `[N]`，无 inplace 分支。

**规约语义**：无规约（逐元素映射，无跨元素依赖），不存在累加顺序问题。

**边界语义**（逐元素，clamp 为投影）：`clamp(0) = 0`（-1≤0≤1）；`clamp(+∞) = hi`；`clamp(−∞) = lo`；`clamp(NaN) = NaN`（torch `hardtanh`/`clamp` 传播 NaN）。**空维度/非整除 shape** 由 host 侧 flatten 成 1-D `[N]`。**次正规/极值**：clamp 为选择，不引入除舍回外的舍入。**NaN 传播注意**：设备 `VMaxOp`/`VMinOp` 的 NaN 语义（IEEE `maxNum` vs 传播）须在 Stage 3 设备探针核对——本算子契约/测试 workload（randn + 有限角点）不含 NaN，故不阻塞设计，列入 §9 风险。

**host 侧语义**：`HardtanhFwdKernel`/`HardtanhFwdOp` 把任意 shape 展平为 `N_total = numel`；`forward` 仅调用编译后 kernel，无 host 预处理；`inplace` 后处理如上。

**语义保持基线**：§8.1 golden 以本节语义（`F.hardtanh(x.float(), min_val, max_val).to(dtype)`）为唯一依据实现，**不得复刻 §0.6 的 NPU 结构决策**（保证验证独立性）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `(N,)`（原始任意 shape 被 host 侧展平；manifest `[1048576]` / `[2048,4096]` / `[16,256,56,56]` / `[4096,4096]`） | `float16 \| bfloat16 \| float32` | 1-D 展平输入向量，`N = product(input.shape)` |
| `out` | `same_as_input`（输出 shape = 输入 shape，展平后 `(N,)`） | `same_as(input)` | 逐元素 hardtanh 输出 |

**输出 shape（迁移必需）**：`same_as_input`（manifest `shape_rules: output.shape == input.shape`；`source_output_shape = same_as_input`）。
**layout**：1-D 连续（contiguous flatten），无转置。
**动态轴**：`N` 在编译期由宿主 shape 决定（`lru_cache` 按 N 特化）；尾块守卫以运行时 `shape: T.int32` 参数支持任意 N。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（非 fp8 分支，`register_copy` 式 fragment load→compute→store，源 L2590–L2606）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `x_reg ← x[bx*bs:(bx+1)*bs]`（fragment 载入，`bs=threads×npt`） | GM `x` | fragment `x_reg` | 元素定位 / 搬入 |
| 2 | `val = x_reg[i*npt+j]`；`lo = cast(min_val, fp32)`；`hi = cast(max_val, fp32)`；`v32 = cast(val, fp32)` | `x_reg`、常量 `min_val`/`max_val` | 标量 `val`/`lo`/`hi`/`v32` | 常量特化 + fp32 升级 |
| 3 | `y_reg[i*npt+j] = cast(val.dtype, min(max(v32, lo), hi))` | `v32`,`lo`,`hi` | `y_reg` | 两侧 clamp（fp32 域） + 单次舍回 |
| 4 | `y[bx*bs:(bx+1)*bs] ← y_reg`（fragment 写回） | `y_reg` | GM `y` | 搬出 |

**数据流与内存访问模式**（源硬件 = GPU）：

```
GM[x] --(fragment load, coalesced)--> registers[x_reg]
   --(T.Parallel(threads,npt) 逐线程标量: cast↑fp32 → max(v32,lo) → min(·,hi) → cast↓dtype)--> registers[y_reg]
   --(fragment store, coalesced)--> GM[y]
（无显式 SMEM；fragment 双缓冲由编译器管理；block_size = threads × npt）
```

**循环与并行结构**：`T.Kernel(T.ceildiv(N, block_size), threads=256)`——一维 grid（`bx`）+ 256 线程/block 两级并行；每线程 `npt` 元素，`T.Parallel(threads_arg, npt_arg)`。`block_size = 256 × npt`：fp32 `npt=4`→1024，fp16/bf16 `npt=8`→2048（`ParametricUnaryKernel.default_config`；K11 折叠至 `_DEFAULT_BLOCK_SIZE`）。

**host 侧逻辑**：`N_total`、`dtype`、`min_val`、`max_val` 由 Op 层传入；`forward` 直接调用编译后 kernel；flatten 由 Op 层完成（wrapper `hardtanh.py` 的 `_DEFAULT_BLOCK_SIZE` = {fp32:1024, fp16:2048, bf16:2048}）；`is_fp8` 分支由 GPU fp8 专用路径触发，本迁移不涉及。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | `register_copy`（fragment load→compute→store） | 向量化 / 合并访存 | `T.alloc_fragment` + `T.copy` 切片 + `T.Parallel` 标量映射 | GPU 大寄存器堆 + memory coalescing | **硬件强相关**（fragment/register 模型） |
| 2 | block×thread 两级并行（大 grid + 256 线程/block） | 用大规模线程/grid 吃满内存带宽 | `T.Kernel(ceildiv(N,bs), threads=256)` + `T.Parallel` | GPU thread/warp 执行模型 | **硬件强相关**（GPU 线程模型） |
| 3 | **fp32 计算域**（`v32=cast(val,f32)`、边界 cast fp32、`min(max)`、单次舍回） | 匹配 `F.hardtanh` 的 fp32 clamp 语义 | clamp 在 fp32 域比较、末端一次舍回 | 半精度 ALU 下需中转 | **语义层**（计算域，须与契约 golden 核对）；**API 层可移植** |
| 4 | strategy-aware `npt`（fp32=4、fp16/bf16=8） | 访存对齐（4×4B / 8×2B = 128bit） | `default_config` 启发式 | memory coalescing（对齐） | 实现层（NPU 折叠为 `block_size`） |
| 5 | 无 autotune（elementwise 固定启发式配置） | 带宽受限 + 单步逐元素，搜索 ROI 低 | `default_config` 固定 `threads=256`+`npt` | 无硬件依赖 | 可移植（NPU 侧同样固定单实现） |
| 6 | 单条 `min(max)` 完成两侧 clamp（2 op，无分支、无超越函数） | 极低 op 数 | `T.min`/`T.max` + 一次 fp32 往来 | GPU 逐元素 ALU | **算法层，可移植**：NPU 有融合 `T.vclamp`（前端展开 `npuir_max`+`npuir_min`），且可按边界常量折叠（§1.6.1 #1） |
| 7 | 工厂 `lru_cache` 按 `(N, dtype, min_val, max_val, …)` 特化 | 消运行时参数分派 | compile-time 常量固化 | 无硬件依赖 | 可移植（NPU 侧同用编译期常量特化，并强化为边界折叠） |

> 识别不出来 ≠ 不存在。上述 7 项为源码中可识别的全部优化/规避手段；后续迁移逐项处置（§0.5）。**未识别优化会被静默丢弃**，故此处列全。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 Phase R 调研结论 §1.6.0）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `clamp(x, lo, hi)`（含 I/O、边界、`inplace` host 契约） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 依 §0.1；计算核不做语义替代 |
| 计算算法（步骤 1–4 逐元素映射结构，含 fp32 域 `min(max)`） | 算法 | 无（纯数学顺序、无跨元素依赖） | — | **保留（结构）** | 逐元素映射结构保留；仅并行/内存实现重设计；fp32 域计算核**原样承接** |
| 优化 #3 fp32 计算域（边界与数据同升 fp32、`min(max)`、单次舍回） | 语义/优化 | 无 | **有**（`T.vclamp`/`vcast` 支持 fp16·fp32） | **保留**（契约对齐 + bf16 强制） | → §0.6 R2：fp16/bf16 用 `T.vcast` 升 fp32 → `T.vclamp` → `T.vcast` 舍回；bf16 因 `vclamp` 无 bf16 **强制**中转（§1.6.1 #2） |
| 优化 #1 `register_copy`（fragment + `T.Parallel` 标量映射） | 优化 | GPU register/fragment 模型 | 无直接等价（NPU 无一维 `threads=`；v-prefix op 作用于整 UB buffer） | **重新设计** | → §0.6 R1：GM→UB→GM + 整 buffer v-prefix 向量链 |
| 优化 #2 block×thread 两级并行 | 优化 | GPU thread/warp 模型 | 无直接等价（NPU 一维 Kernel） | **重新设计** | → §0.6 R1：一维 `T.Kernel(40, is_npu=True)` + 核内 `T.serial` 网格跨步 |
| 优化 #6 计算核（`min(max)`，2 op） | 算法 | GPU 逐元素 ALU | **有**：`T.vclamp`（前端 `npuir_max`+`npuir_min`）/`T.vmin`/`T.vmax` | **等价替换（并优化）** | → §0.6 R2 + §1.6.1 #1：默认 `T.vclamp`（2 op）；退化边界按编译期常量折叠为 `vmin`/`vmax`（1 op）/`copy`（0 op） |
| 优化 #4 strategy-aware `npt` 对齐 | 优化 | memory coalescing（对齐） | 有（32B 尾轴对齐 / 向量宽度 128bit） | **等价替换** | K11：`threads × npt` 折叠为 `block_size`（§5.2），对齐约束按 NPU 32B 复算（§5.3） |
| 优化 #5 无 autotune | 优化 | 无 | 有 | **保留** | NPU 侧同固定单实现（无 strategy/autotune） |
| 优化 #7 `lru_cache` 常量特化 | 优化 | 无 | 有 | **保留（并强化）** | NPU 工厂保持 `min_val`/`max_val`/`N`/`dtype` 编译期常量；并借此做边界折叠特化（§1.6.1 #1） |
| fp8 分支（`is_fp8`，`float8_*` + `out_dtype` 独立） | 优化 | GPU fp8 专用路径 | 无（NPU manifest 不支持 fp8，`SUPPORTED_DTYPES` 仅 float16/bf16/float32） | **舍弃** | manifest 输入 dtype 仅 fp16/bf16/fp32；fp8 不在契约内，不迁移（依据 `elementwise.yaml` HardtanhFwdOp `inputs.input.dtype` + `hardtanh.py` `SUPPORTED_DTYPES`） |
| host 侧 flatten / `inplace`（非 kernel 逻辑） | 算法（host） | 无 | — | **保留** | Op 层传入 `N_total`，kernel 只处理 1-D；`inplace` 由 host（§0.1） |

> **依据**：#3 的 vcast/中转可用性由 `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（`f16→f32` / `f32→f16` / `bf16→f32` / `f32→bf16` 支持 `rint`）佐证；#6 的 `vclamp` 由 `docs/Tilelang.language/比较操作/T.vclamp.md` §2.1/§2.3（标量或张量边界、前端展开 `max`+`min`）佐证，且 `testing/npuir/arith_ops/test_clamp_dev.py`（Developer 模式 fp16/fp32 实跑）为直接先例。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（承接源方案性能意图 + 语义保持论证）。

**重设计项 R1：并行结构重设计（thread 级并行 → 一维 Kernel + 核内网格跨步）**

- **源方案**：`T.Kernel(ceildiv(N, block_size), threads=256)` + `T.Parallel(256, npt)`，每线程 `npt` 元素；性能意图 = 用大规模线程/grid 吃满内存带宽。
- **NPU 新算法**：一维 `T.Kernel(core_num, is_npu=True)`（`core_num` = 实查物理 Vector 核数 **40**，§5.5 ②）+ 核内 `for i in T.serial(T.ceildiv(n_num, core_num))` 网格跨步任务映射（`task = i * core_num + cid`，`if task < n_num` 守卫）；每 tile 经 `T.copy` GM→UB、整 buffer 向量 op、`T.copy` UB→GM。
- **语义保持论证**：任务映射只重排「哪个核处理哪个元素区间」，各元素计算完全独立（无跨元素依赖、无累加顺序变化），数学语义逐元素恒等；边界：`T.serial` 循环上界 `T.ceildiv(n_num, core_num)` 为编译期静态值，守卫 `task < n_num` 覆盖非整除尾块，不产生越界。
- **模式先例**：`examples/elementwise/vec_add_2d.py`、同族迁移 `examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py`（`PL-1.22-persistent-1d-grid-stride`）。

**重设计项 R2：计算核重设计（标量 `min(max)` + 逐元素 cast → 向量链 + 编译期边界折叠）**

- **源方案**：`T.Parallel` 内标量逐元素 `cast↓(min(max(cast↑(val), cast↑(lo)), cast↑(hi)))`——GPU 标量逐元素 2 op（fp32 域）+ 2 cast。
- **NPU 新算法（整 UB buffer 向量链 + 编译期边界特化）**：
  - **默认（两侧有限）**：`T.vclamp(xc, xc, lo, hi)`——前端展开 `npuir_max(xc, lo)` → `npuir_min(·, hi)`，即源 `min(max(v32, lo), hi)` 的融合向量形态（2 op）。
  - **折叠形态（编译期常量守卫）**：`lo == -inf → T.vmin(xc, hi, xc)`（1 op）；`hi == +inf → T.vmax(xc, lo, xc)`（1 op）；两侧皆 inf → 纯 `T.copy`（0 op）；`lo == hi → T.fill`（常量，0 数据 op）。
  - **fp16/bf16**：先 `T.vcast(x_ub, xf, round_mode="rint")` 升 fp32，向量链在 fp32 域，末端 `T.vcast(xc, y_ub, round_mode="rint")` 单次舍回。
- **语义保持论证**：
  1. **默认形态与源逐位等价**：`T.vclamp(src, dst, a, b)` 前端即 `npuir_max(src, a, dst)` + `npuir_min(dst, b, dst)`（`customize_npuir.py::npuir_clamp` L1128+），与源 `T.min(T.max(v32, lo), hi)` 同为「先 max 后 min」，fp32 域、单次舍回，逐位一致（`verify_equiv.py` #2 三 dtype `max_abs=0.0`）。
  2. **折叠形态逐位等价**：一侧无穷时 `max(x, -inf) = x`（clamp 上界失效）与 `min(x, +inf) = x`（下界失效）为浮点恒等（含有限值/±inf；NaN 经 torch 语义核对一致，设备语义见风险 R-4）；两侧皆 inf 为恒等映射；`lo==hi==c` 时 `clamp(x,c,c)=c`（常量填充）。`verify_equiv.py` #1 各折叠行 vs 全式三 dtype `max_abs=0.0`。
  3. **计算域**：fp16/bf16 经 fp32 中转，与源（同 fp32）及契约 golden 一致；`vcast` 升降均 `rint`（仅支持）对应单次舍回。`verify_equiv.py` #2 vs golden 三 dtype、三边界组 `max_abs=0.0`；中转无损行 `0.0`。
  4. **边界语义**：`clamp(+∞)=hi`、`clamp(−∞)=lo`、`clamp(NaN)=NaN`、`±0` 原样；非整除尾块由 `tail` 守卫。折叠形态在 `lo=-inf` 时 `vmin(+∞, hi)=hi`、`vmin(−∞,hi)=−∞`（无下界，正确）——与全式一致。
- **源优化意图承接**：源 #3「fp32 计算域」→ **保留**（R2 显式承接，且 bf16 为 API 强制）；源 #6「极低 op 数（2 op）」→ 默认 `T.vclamp`（2 op）承接，退化边界下进一步降至 1/0 op；源 #1/#2 见 R1。

> 承接说明（源优化意图不丢弃）：§0.4 #1「向量化访存」→ R1/R2 以「GM→UB 整块 `T.copy` + 向量 op 整 buffer」承接；#2「大规模并行吃带宽」→ R1 网格跨步并行承接；#3「fp32 域」→ **保留**（R2）；#4「对齐」→ K11 折叠 `block_size` + §5.3 32B 对齐复算；#5 保留；#6「极低 op 数」→ R2 `vclamp`（2 op）+ 边界折叠（1/0 op）；#7「常量特化」→ R2 编译期边界折叠 + 工厂 `min_val`/`max_val` 常量贯穿。

### 0.7 标杆实现

- 源算子文件（golden 语义依据）：`/home/developer/TileOPs/tileops/kernels/elementwise.py`（`_make_hardtanh_kernel` L2564–L2608、`HardtanhFwdKernel` L2611–L2624）。
- 源仓测试基准：`examples/TileOPs/tests/ops/test_activation.py`（`test_hardtanh` = randn 1M × fp32/fp16/bf16 + 4M × fp16/bf16，`ref_program = F.hardtanh(x.float(), -1.0, 1.0).to(dtype)`；`test_hardtanh_edge` = fp32 4096 填 `repeat([-88.8,-1.0,-0.5,0.0,0.5,1.0,88.8])`；`test_hardtanh_rejects_non_float_dtype`；容差 `_get_tolerances` fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2）。
- manifest：`examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `HardtanhFwdOp`（`ref_api=torch.nn.functional.hardtanh`；params `min_val=-1.0`/`max_val=1.0`/`inplace=false`；shape_rules `min_val <= max_val`、`output.shape == input.shape`；workloads `[1048576]` fp32/fp16/bf16、`[2048,4096]` fp16/bf16、`[16,256,56,56]` fp16/bf16、`[4096,4096]` fp16/bf16/fp32；roofline flops=`N`，bytes=`2*N*elem_bytes`）。
- golden 以 §0.1 语义为依据（`F.hardtanh(x.float(), min_val, max_val).to(dtype)`），**不复刻 §0.6 的向量链结构**（保证验证独立性）。
- 同族迁移先例（只读参考）：`examples/leaky_relu/_make_leaky_relu_kernel/`、`examples/elu/_make_elu_kernel/`、`examples/bitwise_not/bitwise_not_fwd_kernel/`、`examples/erf/erf_fwd_kernel/`。

---

## 1. 概述

### 1.1 算子名称

`_make_hardtanh_kernel`（项目 `hardtanh`，目录 `examples/hardtanh/_make_hardtanh_kernel/`）。

### 1.2 功能描述

对 1-D 展平张量逐元素计算 `y = clamp(x, min_val, max_val)`，语义等价于 `torch.nn.functional.hardtanh`；支持 `float16 | bfloat16 | float32`，输出与输入 shape/dtype 一致。单步纯 Vector 逐元素算子，无规约、无矩阵运算、无跨元素依赖、无超越函数。

**迁移决策后的 NPU 侧算法与源算法的关键差异**：① 计算核由「标量 `T.Parallel` 内 `min(max)`」改为「整 UB buffer 的 `T.vclamp` 向量链」，并按编译期边界常量折叠为 `vmin`/`vmax`/`copy`/`fill`；② 并行/内存结构由 GPU fragment+thread 改为 UB + 一维核 + 网格跨步。计算域与源一致（**均为 fp32 中转 + 单次舍回**），计算核语义与源逐位一致（逐元素 clamp）。

### 1.3 数学公式

$$
y_i = \operatorname{hardtanh}(x_i)=\min\bigl(\max(x_i,\ \text{min\_val}),\ \text{max\_val}\bigr),\quad i=0,1,\ldots,N-1
$$

NPU 计算核（§1.6.1 优化后，按编译期边界常量特化；sub-fp32 在 fp32 计算域构造、末端单次舍回）：

$$
y =
\begin{cases}
x & \text{min\_val}=-\infty,\ \text{max\_val}=+\infty &(\text{copy},\ 0\text{ op})\\[2pt]
\min(x,\ \text{max\_val}) & \text{min\_val}=-\infty &(\texttt{vmin},\ 1\text{ op})\\[2pt]
\max(x,\ \text{min\_val}) & \text{max\_val}=+\infty &(\texttt{vmax},\ 1\text{ op})\\[2pt]
\text{min\_val}\ (\text{constant}) & \text{min\_val}=\text{max\_val} &(\texttt{fill},\ 0\ \text{data op})\\[3pt]
\operatorname{vclamp}(x,\ \text{min\_val},\ \text{max\_val}) & \text{default (both finite)} &(\texttt{npuir\_max}+\texttt{npuir\_min},\ 2\text{ op})
\end{cases}
$$

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法（§0.6），与源算法在结构上等价、在并行/内存实现上有差异，计算域无差异：**

- **计算核心**（§0.6 R2 / §1.6.1 #1）：默认两侧有限边界用 `T.vclamp(xc, xc, lo, hi)`（前端展开 `npuir_max` + `npuir_min`，2 op，即源 `min(max(v32, lo), hi)` 的融合向量形态）；退化边界按编译期常量折叠为 `T.vmin`（1 op）/`T.vmax`（1 op）/`T.copy`（0 op）/`T.fill`（常量）。源为 `T.Parallel` 内标量 `T.min(T.max(v32, lo), hi)`；NPU 改为整 UB buffer 的无分支向量链。
- **计算域**（§0.6 R2 / §1.6.1 #2）：**保留源既有 fp32 中转**——fp16/bf16 先 `T.vcast(rint)` 升 fp32 → 向量链 → `T.vcast(rint)` 舍回；fp32 直算。**匹配契约 golden 的 fp32 opmath 语义**（源本身即 fp32；bf16 为 API 强制）。
- **编译期特化**（§0.6 R2 / §1.6.1 #1）：`min_val`/`max_val` 为编译期 Python 常量（源 `lru_cache` 特化同源），按取值选择向量链形态；默认 `(-1.0, 1.0)` 不触发折叠。

除上述外，算法与 §1.6.0 调研选定的算法族（单步逐元素，访存为理论下界）一致：一趟扫描、无中间 GM 往返、无跨元素依赖。

### 1.5 数据流图

```
fp32 路径（xc = x_ub，链尾原地复用为结果）：
GM[x] --T.copy--> UB[x_ub] --[hardtanh 向量链: vclamp（默认 2 op）或 vmin/vmax（折叠 1 op）]--> UB[x_ub] --T.copy--> GM[y]

fp16/bf16 路径（fp32 中转，xc = xf）：
GM[x(dtype)] --T.copy--> UB[x_ub(dtype)] --T.vcast(rint)--> UB[xf(fp32)]
     --[hardtanh 向量链（fp32 域）]--> UB[xf(fp32)] --T.vcast(rint)--> UB[y_ub(dtype)] --T.copy--> GM[y]
```

向量链内部（全部 UB 级、原地复用）：

```
默认(两侧有限): vclamp(xc, xc, lo, hi)     # = npuir_max(xc,lo) → npuir_min(xc,hi)，结果落回 xc
折叠 lo=-inf:   vmin(xc, hi, xc)           # 结果落回 xc
折叠 hi=+inf:   vmax(xc, lo, xc)           # 结果落回 xc
折叠 两侧inf:   copy(x_ub → y_ub)          # 恒等
折叠 lo==hi:    fill(y_ub, cast(lo,dtype)) # 常量（不读 x 数据）
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（对同一数学语义的算法族回答调研四问，选定算法族），再在数学等价前提下优化公式，再审视循环/标量计算的向量化替代。迁移任务：1.6.1–1.6.3 的分析对象为 §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 本算子为**单步逐元素类**（§0.3 步骤 4 步、无规约/窗口/矩阵/超越函数），按算法调研 §2 属轻量调研范畴，但因存在**多种等价/特化形态**与**计算域/折叠选择**，R1 给出候选表、R3 给出四口径对比、R4 逐候选评估（达到完整调研深度）。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向（op/元素） | 纳入 R3 对比 |
|---|------|------------|-----------|--------------------|-------------|
| 0 | **基线：源算法**（`min(max)`，fp32 域） | `T.min(T.max(v32, lo), hi)`（2 op） | 迭代定义式；源即 fp32 域，与契约 golden 同域 | 2 op（fp32）/4 op（fp16·bf16，+2 vcast） | ✅（基线） |
| 1 | **融合 `T.vclamp`（采纳，默认路径）** | `T.vclamp(xc, xc, lo, hi)`（前端 `npuir_max`+`npuir_min`，2 op） | 与基线**结构同式**（前端即「先 max 后 min」）；仅 API 融合 | 2 op（同基线），单条向量调用 | ✅（采纳） |
| 2 | **编译期边界特化折叠（采纳）** | `lo=-inf→vmin`（1 op）/`hi=+inf→vmax`（1 op）/两侧 inf→`copy`（0 op）/`lo==hi→fill`（0 数据 op） | **逐位恒等**（一侧无穷时 clamp 退化为单侧；NaN 经 torch 语义核对一致） | **1/0 op（退化边界）**；默认不触发 | ✅（采纳，§1.6.1 #1） |
| 3 | fp16 原生 dtype clamp（跳过 fp32 中转） | `T.vclamp` on fp16 | 对 clamp **亦逐位等价**（round 与 clamp 可交换，机器诊断证）；但 bf16 无 `vclamp`，且 wrapper 契约要求 fp32 | 2 op（fp16）vs 4 op | ✅（评估，**不采纳**，§1.6.1 E1） |
| 4 | relu 分解：`lo + relu(x-lo) - relu(x-hi)` | `vsub`+`vrelu`+`vsub`+`vadd`（≥4 op） | 数学等价（`lo ≤ hi`） | ≥4 op（多于 2） | ✅（评估，不采纳） |
| 5 | median 三分量形式 | `max(min(a,b), min(max(a,b), c))`（≥4 op） | 数学等价（`lo≤hi`） | ≥4 op | ✅（评估，不采纳） |
| 6 | compare/select 形式 | `vcmp` + `vselect`（3 op） | 数学等价 | 3 op（多于 2）；`vcmp` bool 需片上 | ✅（评估，不采纳） |
| 7 | 超越函数逼近 | — | **不适用**：hardtanh 为分段线性，无超越函数可降代 | 无 | ❌（无适用性） |

**R2 在线算法**：**无在线化收益**——逐元素映射本身天然单遍（无跨元素依赖），不存在可被「单遍/流式」消去的多遍基线。结构判据：本算子无 running 统计量（无归约/无 max/sum 归约、无跨元素依赖），在线变体定义不适用。依据：源算法即单遍直写（§0.3 步骤 1–4 无循环携带依赖）；`examples/elementwise/` 全系逐元素实现（`vec_add_1d.py`、`vec_add_2d.py`）单遍；`algorithm-candidates.md` 的 `ALG-elementwise` R2 记「逐元素映射天然单遍」。

**R3 复杂度对比**（四口径；本算子**带宽/发射受限**，访存/遍数是主导项）：

| 算法候选 | FLOPs（v-op 数/元素） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|----------------------|---------------|---------|-------------|---------------------|
| 基线（源 `min(max)`，fp32 域） | 2（max1+min1） | `2·N·elem_bytes` | 1 遍 | O(block_size)（~4 UB buffer） | 可分块给多核，无跨核归并 |
| #1 融合 `vclamp`（**采纳**） | 2 | `2·N·elem_bytes` | 1 遍 | O(block_size) | 同上 |
| #2 边界折叠（**采纳**，退化边界） | 1（`vmin`/`vmax`）/ 0（copy、fill） | `2·N·elem_bytes` | 1 遍 | 同量级（复制形态仅 1 输入 1 输出） | 同上 |
| #3 fp16 原生 clamp | 2（无 vcast） | `2·N·elem_bytes` | 1 遍 | 略小（省 2 fp32 buffer） | 同上（bf16 不适用） |
| #4 relu 分解 | ≥4 | `2·N·elem_bytes` | 1 遍 | 同量级 | 同上 |
| #5 median | ≥4 | `2·N·elem_bytes` | 1 遍 | 同量级 | 同上 |
| #6 compare/select | 3（+bool buffer） | `2·N·elem_bytes` | 1 遍 | 同量级（+bool buffer） | 同上 |

> 口径说明：FLOPs 以「每元素 v-op 数」计（无超越函数，1 op = 1 单元操作）；fp16/bf16 默认路径各 +2 `vcast`（升+降，candidate #3 除外）。访存 = 输入读 + 输出写，**全部候选同为理论下界 `2·N·elem_bytes`**（无中间 GM 往返）；扫描遍数 = GM 读取次数 = 1；中间缓冲 = UB 单 tile 驻留（非 O(N)）。候选差异在 **op 数（发射成本）**、**退化边界覆盖** 与 **API 可行性（bf16）**。

**R4 硬件亲和性评估**（逐候选对照检查清单；容量/带宽淘汰引用 `constants.md` 条目）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **#1 vclamp + #2 折叠（采纳）** | Vector（`vclamp`→`VMaxOp`+`VMinOp`；`vclamp`/`vmin`/`vmax` fp16·fp32 ✓、bf16 ✗；`vcast` f16↔f32/bf16↔f32 支持 rint） | 小（~5 工作 buffer，bs2048 fp16 ≈ 38 KB，§4.5）✓ | 尾轴 32B 对齐 ✓；v-op 作用于整 buffer | N 编译期常量 ✓ | 单步无多阶段；跨 tile 重叠由 auto-multi-buffer 处理 | ✅ **采纳** |
| 基线（源 `min(max)`） | Vector（`vmin`/`vmax` fp16·fp32 ✓） | 同 | ✓ | ✓ | 同 | ⚠️ 等价（融合 `vclamp` 同结构，取 #1） |
| #3 fp16 原生 clamp | Vector（`vclamp` fp16 ✓，bf16 ✗） | 略小 | ✓ | ✓ | 同 | ❌ ROI≈0（MTE2 受限）+ wrapper 契约要求 fp32 + bf16 不适用（§1.6.1 E1） |
| #4 relu 分解 / #5 median | Vector | 同 | ✓ | ✓ | 同 | ❌ op 更多（≥4 vs 2），无增益 |
| #6 compare/select | Vector（`vcmp`/`vselect`） | 同 +bool buffer | ✓ | ✓ | 同 | ❌ op 更多（3 vs 2）+ bool 片上约束 |
| #7 超越函数逼近 | — | — | — | — | — | ❌ 数学不适用（分段线性） |

**v-prefix 元算 dtype 矩阵（决定 bf16 必经 fp32 中转）**：`vclamp`/`vmin`/`vmax` **不含 bf16**（`docs/Tilelang.language/比较操作/T.vclamp.md`·`T.vmin.md`·`T.vmax.md` §2.2.1：fp16 ✓ / fp32 ✓ / bf16 ✗）；`vcast` 支持 `f16→f32`、`bf16→f32`、`f32→f16/bf16`（均 `rint`）。→ bf16 一律经 fp32 中转。

**调研结论**：**选定「融合 `T.vclamp`（默认两侧有限）+ 编译期边界折叠（退化边界）+ fp32 中转计算域」**（候选 #1 + #2 + 域保留），无更优可行替代。

关键依据：① R3 全部候选访存同为下界 `2N·elem_bytes`，差异在 op 数、退化边界覆盖与 API 可行性——候选 #1 与基线同为最少 op（2）且为源 `min(max)` 的融合向量形态；候选 #2 在退化边界下进一步降至 1/0 op；候选 #3（fp16 原生）虽亦 2 op 且机器证逐位等价，但 **bf16 不可用 + wrapper 契约要求 fp32 + ROI≈0**，不采纳；候选 #4/#5/#6 op 更多、#7 数学不适用，均淘汰。② **计算域**：源本身即 fp32 中转、契约 golden 亦 fp32，**保留 fp32 域无争议**；bf16 为 API 强制。③ 选定算法的全部 op（`vclamp`/`vmin`/`vmax`/`vcast`）均有本仓 docs 佐证与 `testing/npuir/arith_ops/test_clamp_dev.py`（Developer fp16/fp32）实跑先例（§3.4）。

**调研范围**（「无更优替代」依据）：① 本仓算法候选库 `tilelang-op-design/references/algorithm-candidates.md` 的 `ALG-elementwise` 条目（R1 记「公共子表达式消除；除法乘倒数；cast 链合并」；R2 记「逐元素映射天然单遍」；R3 记「融合消除中间 GM 往返」；R4 记「单遍 Vector 流；dtype 路径」）——本仓候选库不含 hardtanh/clamp 专门条目，故以恒等变形自查补足（候选 #1–#6）；② 本仓 `examples/` 全系（`elementwise/*`、`leaky_relu`、`elu`、`erf`、`expm1`、`bitwise_not`、`mish`）：`elu`/`erf`/`expm1` 为超越函数自建逼近（hardtanh 不适用，纯分段线性）、`bitwise_not` 为有原生元算直译形态、`leaky_relu` 为分段线性代数重排 + 标量参数特化（**本算子最接近**）——hardtanh 属「clamp 直译 + 边界常量折叠」形态；③ pattern-library（`kb_search "clamp hardtanh elementwise activation"` 消费预注入条目）：`PL-1.28-const-scalar-param-fold`（标量参数编译期特化——**直接支持候选 #2**）、`PL-1.17-subfp32-fp32-transit`（sub-fp32 逐元素 fp32 中转——支持计算域决策，**其第三触发「标量参数半精度不可表示」提示核对边界表示性**）、`PL-1.22-persistent-1d-grid-stride`（一维网格跨步）、`PL-1.23-vop-chain-inplace-alias`（v-op 链原地复用）、`CONST-ascend910-9362-profile`（设备带宽画像）、`TRAP-expert-v-operands`（v-op 操作数规则）；④ **外部已知算法族**（模型知识 + PyTorch/CUDA 生态）：clamp 的代数重写（relu 分解 / median / min-max）与 torch 原生 min/max——均属分段线性代数重排，由 §1.6.1 逐项论证；无超越函数逼近空间；⑤ `docs/Tilelang.language/比较操作/` 全目录枚举确认存在**原生 `vclamp` 融合元算**（前端展开 `npuir_max`+`npuir_min`），故默认路径无需手写 `min(max)`（但两者同结构）。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)；纯 Vector 逐元素豁免容量项）**：

- 流量项 = `bytes ÷ 有效带宽`，`bytes = 2·N·elem_bytes`（manifest roofline 同口径）。引用 **`CONST-ascend910-9362-profile`**（`constants.md`，`status: verified`，版本戳 Ascend910_9362 / CANN 9.0.0 / tilelang dev build 0.1.1.10 / 2026-10-13）：单 die 核→GM ≈ **1.06 TB/s**、HBM ≈1.64 TB/s；BitwiseNotFwdOp 大 shape 实测饱和 **1.06–1.10 TB/s（≈核→GM）**。取 **1.10 TB/s** 作流量分母（大 N 主口径）。
- 发射项 = `(ceil(n_num/core_num)) × k × ~0.5µs/op`（`CONST-vector-launch-overhead`，绑定 910B2C **且 `kb_stale_check` 标记 stale**，本设备 910_9362 未重验，**未文档化假设 + 估算依据**）；`k` = 每 tile 向量 op 数：默认路径 fp32 `k=2`、fp16/bf16 `k=4`（+2 vcast）。
- 容量项 = 0（纯 Vector，无跨引擎往返；`CONST-store-fixpipe-gm-only` 不适用）。

以 dtype 对应 `block_size`（fp32=1024、fp16/bf16=2048，K11 折叠）、物理 Vector 核 = 40、默认 `(-1.0, 1.0)`（不折叠，`k` 如上）估算：

| workload | dtype | N | block_size | n_num | bytes | 流量项 | 发射项 (k) | **估算下界 max** |
|----------|---|------|-----------|-------|-------|---------------------|----------------------------|------------------|
| smoke-1m | fp32 | 1,048,576 | 1024 | 1,024 | 8.4 MB | ≈ 7.6 µs | 26.0 µs (k=2) | **≈ 26.0 µs** |
| smoke-1m | fp16 | 1,048,576 | 2048 | 512 | 4.2 MB | ≈ 3.8 µs | 26.0 µs (k=4) | **≈ 26.0 µs** |
| bounded-hidden `[2048,4096]` | fp16 | 8,388,608 | 2048 | 4,096 | 33.5 MB | ≈ 30.5 µs | ≈ 206 µs (k=4) | **≈ 206 µs** |
| bounded-conv-feat `[16,256,56,56]` | fp16 | 12,845,056 | 2048 | 6,272 | 51.4 MB | ≈ 46.7 µs | ≈ 314 µs (k=4) | **≈ 314 µs** |
| elementwise-16m `[4096,4096]` | fp16 | 16,777,216 | 2048 | 8,192 | 67.1 MB | ≈ 61.0 µs | ≈ 410 µs (k=4) | **≈ 410 µs** |
| elementwise-16m `[4096,4096]` | fp32 | 16,777,216 | 1024 | 16,384 | 134.2 MB | ≈ 122 µs | ≈ 410 µs (k=2) | **≈ 410 µs** |

**设计期估算（D-2 roofline）**：代表性 `smoke-1m fp16` 估算下界 ≈ **26.0 µs**（流量 3.8 + 发射 26.0 + 容量 0，**发射项主导**）；大 shape 由发射项主导（本算子为 **2–4-op 短链**，发射成本是第一驱动）。**估算不确定性声明（未文档化假设 + 估算依据）**：① `CONST-vector-launch-overhead` 的 0.5µs/op 绑定 910B2C 且 stale，本设备 910_9362 未重验，故发射项可能**上偏**；② **对照证据（实测）**：同族 `bitwise_not`（k=1，1M int32）实测 7.72µs、16m int32 93.62µs、256m int32 2058µs——均**贴近流量项而非发射项**（1M：流量 7.5µs vs 实测 7.72µs），说明**短链逐元素算子在实测中由流量项主导、发射项被 MTE2 窗口隐藏**（与 `PL-1.6-copy-floor`「≥16M 计算链被 MTE2 窗口完全隐藏」一致，该条 stale 仅作背景）；故上表发射项**系统性上偏**，真实下界应更接近流量项；③ `block_size` 是摊薄发射的第一杠杆（发射 ∝ 1/block_size，§5.2）；④ Stage 4 baseline 采集后按 `hardware-cost-model.md` §3 对账并回填常数。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（vclamp + 边界折叠 + fp32 中转）。以下各项完成「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；采纳项的等价性经 `verify_equiv.py` 机器验证（结果表见本节末）。

| # | 优化项 | 原式（基线） | 优化后公式（采纳） | 等价性论证 | 收益量化 |
|---|--------|------|-----------|-----------|---------|
| 1 | **编译期边界特化折叠**（采纳） | 默认 `vclamp(x, lo, hi)`（2 op） | `lo=-inf` → `vmin(x, hi)`（1 op）；`hi=+inf` → `vmax(x, lo)`（1 op）；两侧 inf → `copy`（0 op）；`lo==hi` → `fill`（常量，0 数据 op） | **逐位恒等**：一侧无穷时 `max(x,-inf)=x`（含 `x` 有限/±inf）、`min(x,+inf)=x`；两侧 inf 为恒等映射；`lo==hi==c` 时 `clamp(x,c,c)=c`。`verify_equiv.py` #1 各折叠行 vs 全式三 dtype `max_abs=0.0` | 退化边界下 op 2→1→0（发射项 −50%~−100%）；**默认 `(-1,1)` 不触发**（诚实记录） |
| 2 | **fp32 中转计算域**（采纳 = 保留源既有 + 契约对齐） | 概念上「输入 dtype 域直译」（`cast(lo,dtype)` 域内 clamp） | `vcast↑f32(rint) → vclamp(fp32, 边界为 fp32 常量) → vcast↓dtype(rint)` | 与契约 golden（fp32 域）**逐位一致**（`verify_equiv.py` #2 三 dtype × 三边界组 `max_abs=0.0`；中转无损）；**源本身即 fp32 中转**（`v32=T.cast(val,fp32)`）→ 本项为**保留并对齐**，非结构变更 | 正确性：与源/契约同域；**bf16 为 API 强制**（`vclamp`/`vmin`/`vmax` 无 bf16）；代价 +2 vcast（fp16/bf16） |
| E1 | 原生 dtype clamp / 跳过 fp32 中转（**不采纳**） | — | `vclamp` on fp16（2 op） | **机器诊断证其亦逐位等价**（round 与 clamp 可交换，`verify_equiv.py` 诊断行三 dtype `max_abs=0.0`），**但 bf16 无 `vclamp` 且 wrapper 契约明确要求保留 fp32 中转** | ROI≈0（MTE2 受限，见 §1.6.0 设计期估算对照证据）→ **不采纳**；理由记录于此，列为 Stage 4 备选 |
| E2 | 代数重写（relu 分解 / median / compare-select）（**不采纳**） | — | 多种 | 数学等价（`lo≤hi`） | op 更多（≥3 vs 2），无增益 |
| E3 | 浮点边界往返 / 舍入合并（**不采纳**） | — | — | 计算域已为 fp32 且单次舍回，无额外舍入可消 | 无（域已对齐） |

**优化结论**：采纳 **2 项**（#1 编译期边界折叠、#2 fp32 中转计算域）。**公式层面已无进一步计算量/访存量可优化空间**——默认 `(-1, 1)` 两侧有限边界的 `clamp` 已是**最短闭式（2 op = 源的最小 op 数）**，访存固定在理论下界 `2N·elem_bytes`（1 读 1 写、无中间 GM 往返）；唯一的额外 op 是 fp16/bf16 的 2 次 `vcast`（bf16 为 API 强制；fp16 因 wrapper 契约与统一路径保留，E1 记录其可省但 ROI≈0）。**优化后公式**（供 §3.1 使用）：
- 默认 `(-1,1)`：`float32` → `vclamp(x, lo, hi)`（2 op）；`float16/bfloat16` → `vcast_dtype(vclamp(vcast_f32(x), lo, hi))`（+2 vcast）
- 退化边界：`vmin`（lo=-inf，1 op）/ `vmax`（hi=+inf，1 op）/ `copy`（两侧 inf，0 op）/ `fill`（lo==hi，常量）

**等价性机器验证（D-1）**：脚本 `examples/hardtanh/_make_hardtanh_kernel/verify_equiv.py`（CPU；候选式 = NPU 向量链逐 op 复现，fp32 域 clamp 后按目标 dtype 单次舍回；基线式：折叠项对照「全式」`min(max(x,lo),hi)`，域项对照契约 golden `F.hardtanh(x.float(), lo, hi).to(dtype)`）。**执行结果表**（脚本实际输出，`python3 verify_equiv.py`，**ALL 20 ITEMS EQUIV_PASS**）：

| 采纳项 / 对照 | 候选式 | 基线式 | samples | 最大 abs 差 | 违反比 | 结论 |
|---------------|--------|--------|---------|------------|--------|------|
| #2 fp32 中转 vs GOLDEN `[default(-1,1)]` (fp32) | fp32 中转链 | `F.hardtanh(x,lo,hi)` | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #2 fp32 中转 vs GOLDEN `[nondefault(-0.5,2)]` (fp32) | fp32 中转链 | `F.hardtanh` | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #2 fp32 中转 vs GOLDEN `[nonrepresentable(0.01,0.5)]` (fp32) | fp32 中转链 | `F.hardtanh` | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #1 fold `lo=-inf`(vmin) vs 全式 (fp32) | `min(x,hi)` | `min(max(x,-inf),hi)` | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #1 fold `hi=+inf`(vmax) vs 全式 (fp32) | `max(x,lo)` | `min(max(x,lo),+inf)` | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #1 fold 两侧 inf(copy) vs 全式 (fp32) | `x` | `min(max(x,-inf),+inf)` | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #2 fp32 中转 vs GOLDEN `[default(-1,1)]` (fp16) | fp32 中转链 | `F.hardtanh` | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #2 fp32 中转 vs GOLDEN `[nondefault(-0.5,2)]` (fp16) | fp32 中转链 | `F.hardtanh` | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #2 fp32 中转 vs GOLDEN `[nonrepresentable(0.01,0.5)]` (fp16) | fp32 中转链 | `F.hardtanh` | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #1 fold `lo=-inf`(vmin) vs 全式 (fp16) | `min(x,hi)` | 全式 | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #1 fold `hi=+inf`(vmax) vs 全式 (fp16) | `max(x,lo)` | 全式 | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #1 fold 两侧 inf(copy) vs 全式 (fp16) | `x` | 全式 | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #2 中转无损 (fp16) | f16→f32→f16(rint) | `x`（恒等） | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #2 fp32 中转 vs GOLDEN `[default(-1,1)]` (bf16) | fp32 中转链 | `F.hardtanh` | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #2 fp32 中转 vs GOLDEN `[nondefault(-0.5,2)]` (bf16) | fp32 中转链 | `F.hardtanh` | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #2 fp32 中转 vs GOLDEN `[nonrepresentable(0.01,0.5)]` (bf16) | fp32 中转链 | `F.hardtanh` | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #1 fold `lo=-inf`(vmin) vs 全式 (bf16) | `min(x,hi)` | 全式 | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #1 fold `hi=+inf`(vmax) vs 全式 (bf16) | `max(x,lo)` | 全式 | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #1 fold 两侧 inf(copy) vs 全式 (bf16) | `x` | 全式 | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #2 中转无损 (bf16) | bf16→f32→bf16(rint) | `x`（恒等） | 1,048,605 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |

> 判定准则 = Stage 3 门禁同口径 `|cand−base| ≤ atol + rtol·|base|`（fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2），违反率 0 即 PASS；覆盖随机张量 `randn`（2²⁰）+ 角点（±0/±inf/NaN/次正规 1e-45/边界邻域 0.01±/±1/±88.8/65504）。

**非门禁诊断（机器输出）——被拒候选 E1 的等价性证据**（说明「fp16 原生 clamp」在数值上亦逐位等价，其弃选理由是 ROI 与契约而非正确性）：

| 被拒候选 | 边界组 | dtype | max_abs | 违反数 | 结论 |
|----------|--------|-------|---------|--------|------|
| 原生 dtype clamp（跳过中转） | 0.01/0.5、-0.1/0.1 | fp32/fp16/bf16 | 0.000e+00 | 0 | 数值上逐位等价（round 与 clamp 可交换）→ 弃选理由 = bf16 不可用 + wrapper 契约 + ROI≈0，非正确性（§1.6.1 E1） |

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 两侧 clamp | 源标量逐元素 `T.min(T.max(v32, lo), hi)` | `T.vclamp(xc, xc, lo, hi)`（融合向量；佐证 `docs/.../比较操作/T.vclamp.md` §2.4 + `testing/npuir/arith_ops/test_clamp_dev.py` 实跑）；退化边界折叠为 `T.vmin`/`T.vmax`（佐证 `比较操作/T.vmin.md`·`T.vmax.md` §2.4） | ✅ | — |
| 2 | fp32 中转 cast（fp16/bf16） | 源标量 `T.cast(val,"float32")` / `T.Cast(val.dtype,…)` | `T.vcast(src, dst, round_mode="rint")`（升/降；佐证 `docs/.../数据类型转换操作/T.vcast.md` §2.2.1） | ✅ | — |
| 3 | 逐元素 `T.Parallel(threads,npt)` 循环 | 源线程级元素映射循环 | **消除**：整 UB buffer 向量 op 一次完成（无 `T.Parallel` 标量循环） | ✅ | — |
| 4 | `if idx < N` 逐元素守卫（fp8 分支） | 源标量分支（本分支无） | 改由 tile 级 `tail = T.min(block_size, shape-off)` 动态切片（尾部整体裁） | ✅ | — |
| 5 | 边界常量 `cast(min_val, f32)` | 源逐元素重复 `T.cast` | 提为 host 侧 Python 编译期常量（参与折叠特化） | ✅ | — |
| 6 | 核内网格跨步任务循环 | `for i in T.serial(ceildiv(n_num, core_num))` | 无（核内调度循环，非逐元素计算） | ❌ | tile/block 级任务调度循环；每核迭代数 `ceil(n_num/40)`（1M fp32 bs1024 = 26 次，fp16 bs2048 = 13 次），非逐元素热点；元素级并行由向量指令整 buffer 完成。先例 `vec_add_2d.py`、`PL-1.22-persistent-1d-grid-stride` |
| 7 | block 起始索引 / 任务映射 | `task=i*core_num+cid`、`off=task*block_size` | 无 | ❌ | block 级索引/任务映射，每 tile 少量标量，非逐元素热点；无逐元素等价 API |
| 8 | 尾块边界（非整除 N） | `tail = T.min(block_size, shape-off)` | 无 | ❌ | 依赖动态 shape 的边界处理；manifest 全部 workload 的 N 均整除 `block_size`，设计保留 `shape: T.int32` 守卫以支持任意 N（测试 4,000,000 于 bs2048 余 1216）（纯静态化对通用 N 不成立） |

**向量化结论**：逐元素计算点（#1–#5）**已全部向量化**（`T.vclamp/vmin/vmax/vcast`，均有 docs 佐证 + `test_clamp_dev.py` 实跑先例）；保留 3 处标量/循环（#6 核内调度循环、#7 block 索引、#8 动态边界），理由类别：tile 级调度循环 + block 索引 + 动态边界，逐项见上表。与 §6 循环结构一致——kernel 内**无逐元素标量循环**。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

**单轴算子豁免**：本算子为 1-D `[N]` 单轴算子（host 侧把任意 shape 展平为一维），不存在「在哪个轴向量化」的轴选择问题——唯一候选向量化轴即该 1D 元素轴本身。按 skill Phase 2 第 3 项「纯 GEMV/单轴算子可写明『仅一个候选轴』豁免」，记录豁免依据：

- **向量化轴** = 唯一 1D 元素轴（stride=1，连续）。
- **轴质量**：fp32 元素 4B，向量宽 ×4 → `block_size=1024` 每 tile 256 向量槽，尾 lane 浪费 0%（1024 整除 4）；fp16/bf16 元素 2B，向量宽 ×8 → `block_size=2048` 每 tile 256 槽，浪费 0%；均按 32B 尾轴对齐（1024×4=4096B、2048×2=4096B 为 32B 整数倍）。无累加链（无串行累加维，`vclamp`/`vmin`/`vmax` 均为纯逐元素）。无 repack（I/O 布局 = 核内布局，一维连续，不需 `T.transpose` 或 host permute）。UB 驻留与 §4.5 一致。
- **迁移注**：GPU 源码的并行轴选择（thread `i`×`npt j` → 元素 `idx`）是**输入而非结论**——NPU Vector 轴与之不对应，§0.6 R1 已独立评估为「核内 grid-stride + 向量指令整 buffer」。
- **决策落地**：主选轴 = 元素轴，同步落入 §3.3 伪代码与 §6 循环结构（向量链各 op 直接作用于 UB 一维 buffer；无逐元素标量循环）。

**布局决策结论**：选定方案（唯一候选轴）：核内布局 = 一维连续 `[block_size]`（I/O 布局即核内布局，无 repack），向量化轴 = 唯一 1D 元素轴，repack 路径 = 无。弃选候选：无（单轴算子，无第二轴可枚举，按 skill Phase 2 第 3 项豁免）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer

### 2.2 选型理由

| 算子特征 | 分析 | 结论 |
|----------|------|------|
| 计算类型 | 纯 Vector（逐元素 clamp 向量链），无 matmul | 不需 Cube/L0C/L1 |
| 是否含归约 | 无归约 | 不需 Expert 级 buffer 控制 |
| 是否需流水线 | 单数据通路 GM→UB→GM，无核间协作 | 不需 CV 融合 |
| 向量链 API | 全部 v-prefix（`T.vclamp/vmin/vmax/vcast`） | 用 v-prefix API |
| 同步需求 | UB 内顺序计算，数据依赖线性 | Developer 自动同步即可 |

决策树路径：`纯 element-wise → 单步逐元素 → 无需精细 buffer 控制 → Developer 模式`。与 `op_requirements.programming_mode=developer` 一致；参考 `examples/elementwise/vec_add_1d.py`（Developer + `T.alloc_ub` + v-prefix）、`examples/leaky_relu/_make_leaky_relu_kernel/`（同族参数化逐元素）与 `testing/npuir/arith_ops/test_clamp_dev.py`（Developer `T.vclamp` fp16/fp32 先例）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub((block_size,), dtype)` 显式 UB 分配（v-prefix API 要求 buffer 在 UB） |
| 计算方式 | v-prefix API 向量链，非 `T.Parallel` 标量循环 |
| 同步 | Developer 自动同步（线性数据依赖，编译器自动插 barrier） |
| 分核 | 一维 `T.Kernel(core_num, is_npu=True)` + `T.serial` 网格跨步（§5.5） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（clamp 向量链 + 边界编译期折叠 + fp16/bf16 fp32 中转；默认路径 = `T.vclamp`）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_tile = x[off : off+bs]` | 搬入一维 tile 到 UB |
| 2 | `xf = cast(x_tile)`（仅 fp16/bf16） | fp32 中转（无损升级） |
| 3a | `yc = min(max(xc, lo), hi)`（默认） | 两侧 clamp（`T.vclamp`，fp32 域） |
| 3b | `yc = min(xc, hi)` / `max(xc, lo)` / `xc` / `fill(lo)`（折叠） | 退化边界形态 |
| 4 | `y_tile = cast(yc)`（仅 fp16/bf16） | 单次舍回原 dtype |
| 5 | `y[off : off+bs] = y_tile` | 搬出结果到 GM |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 搬入 | `x_tile = x[...]` | `T.copy` | `T.copy(x[off:off+tail], x_ub[0:tail])` | Developer |
| cast 升 fp32（fp16/bf16） | `xf = cast(x, fp32)` | `T.vcast` | `T.vcast(x_ub, xf, round_mode="rint")` | Developer |
| 两侧 clamp（默认） | `min(max(xc,lo),hi)` | `T.vclamp` | `T.vclamp(xc, xc, lo, hi)`（标量边界；结果原地落回 `xc`） | Developer |
| 单侧 clamp（折叠 lo=-inf） | `min(xc, hi)` | `T.vmin` | `T.vmin(xc, hi, xc)`（标量 src2） | Developer |
| 单侧 clamp（折叠 hi=+inf） | `max(xc, lo)` | `T.vmax` | `T.vmax(xc, lo_ub, xc)`（边界常量 buffer，见下注） | Developer |
| 恒等（折叠 两侧 inf） | `xc` | `T.copy` | `T.copy(x_ub[0:tail], y[off:off+tail])` | Developer |
| 常量（折叠 lo==hi） | `c` | `T.fill` | `T.fill(y_ub, T.cast(lo, dtype))` | Developer |
| cast 舍回（fp16/bf16） | `y_tile = cast(yc,dtype)` | `T.vcast` | `T.vcast(xc, y_ub, round_mode="rint")` | Developer |
| 搬出 | `y[...] = y_tile` | `T.copy` | `T.copy((y_ub if need_cast else x_ub)[0:tail], y[off:off+tail])` | Developer |
| 核内网格跨步 | 任务映射 | `T.Kernel` + `T.serial` | `T.Kernel(core_num, is_npu=True)` + `for i in T.serial(ceildiv(n_num, core_num))` | Developer |
| 边界常量 buffer（hi=+inf 折叠） | `lo` | `T.fill` | `T.fill(lo_ub, T.cast(lo, f32))`（循环外一次） | Developer |

> **注（`vmax` 标量边界规避）**：`docs/.../比较操作/T.vmin.md` 明示 src2 可用标量，而 `T.vmax.md` 仅示例张量-张量（`TRAP-expert-v-operands`：`T.vcmp` 标量操作数有拒绝 `tir.Cast` 的历史）。故 **`hi=+inf` 折叠形态用预填 fp32 常量 buffer `lo_ub` 作 `vmax` 第二操作数**（循环外 `T.fill` 一次，只读跨 tile 有效），规避标量操作数不确定性；默认形态用 `T.vclamp`（doc 明示支持标量边界）。

### 3.3 计算伪代码

> 迁移规则：`@T.prim_func` 参数**数据类型直接作为第二个位置参数**（不用 `dtype=` 关键字，避免 `false alarm`）；`@T.prim_func` 参数名 `x, y` 及顺序不变，新增 `shape: T.int32` 用于动态 N 尾块守卫；**v-prefix op 只作用于整 UB buffer，动态 slice 只包在 `T.copy` 里**（`PL-1.22`/`PL-1.23`）；**buffer 在 `T.Kernel` 顶层无条件分配**（避免 `TRAP-conditional-alloc`）。

```python
import tilelang
import tilelang.language as T

# Hardtanh: y = clamp(x, min_val, max_val)   (min_val/max_val 编译期常量; 默认 -1.0 / 1.0)
@tilelang.jit(out_idx=[1], target="npuir")
def _make_hardtanh_kernel(N, dtype, min_val, max_val, block_size=2048):
    f32 = "float32"
    need_cast = (dtype != f32)                 # 编译期常量（fp16/bf16 -> fp32 中转）
    lo = float(min_val)
    hi = float(max_val)
    NEG_INF, POS_INF = float("-inf"), float("inf")
    # 编译期边界折叠（host Python 常量，非运行时分支；默认 (-1,1) 不触发）
    only_upper = (lo == NEG_INF and hi != POS_INF)   # -> vmin
    only_lower = (hi == POS_INF and lo != NEG_INF)   # -> vmax
    identity   = (lo == NEG_INF and hi == POS_INF)   # -> copy
    const_val  = (lo == hi)                          # -> fill
    n_num = T.ceildiv(N, block_size)                 # N 编译期常量 -> 静态值
    core_num = 40                                    # 物理 Vector 核数（实查 20 aicore x 2）

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), dtype),
        shape: T.int32,
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)   # 输入 tile
            y_ub = T.alloc_ub((block_size,), dtype)   # 输出/舍回 tile
            xf = T.alloc_ub((block_size,), f32)       # fp32 计算域（fp16/bf16 中转入口）
            lo_ub = T.alloc_ub((block_size,), f32)    # 常量 lo（仅 hi=+inf 折叠用，只读）

            if only_lower:                            # 循环外一次填充（只读跨 tile 有效）
                T.fill(lo_ub, T.cast(lo, f32))

            xc = xf if need_cast else x_ub            # 编译期别名：计算域

            for i in T.serial(T.ceildiv(n_num, core_num)):   # 静态循环上界
                task = i * core_num + cid
                if task < n_num:
                    off = task * block_size
                    tail = T.min(block_size, shape - off)

                    if const_val:                     # 折叠：输出=常量（不读 x 数据）
                        T.fill(y_ub, T.cast(lo, dtype))
                        T.copy(y_ub[0:tail], y[off:off + tail])
                    else:
                        T.copy(x[off:off + tail], x_ub[0:tail])
                        if need_cast:
                            T.vcast(x_ub, xf, round_mode="rint")

                        # --- hardtanh 向量链（fp32 域，整 buffer）---
                        if identity:                  # 折叠：恒等
                            pass
                        elif only_upper:              # 折叠 lo=-inf -> vmin
                            T.vmin(xc, T.cast(hi, f32), xc)
                        elif only_lower:              # 折叠 hi=+inf -> vmax
                            T.vmax(xc, lo_ub, xc)
                        else:                         # 默认两侧有限 -> vclamp (2 op)
                            T.vclamp(xc, xc, T.cast(lo, f32), T.cast(hi, f32))

                        if need_cast:
                            T.vcast(xc, y_ub, round_mode="rint")
                            T.copy(y_ub[0:tail], y[off:off + tail])
                        else:
                            T.copy(xc[0:tail], y[off:off + tail])

    return main
```

> **伪代码说明**：① `if need_cast:` / `if only_upper:` / … 为 Python 编译期条件（JIT 参数编译期确定），不产生运行时分支；**buffer 在 `with T.Kernel` 顶层无条件分配**（避免 `TRAP-conditional-alloc` 的 `Undefined variable`）；② `lo_ub` 仅在 `hi=+inf` 折叠时填充（循环外一次，只读）；③ v-op 全部作用于**整 buffer**（`PL-1.22`），仅 `T.copy` 用动态 slice；④ **`xc` 计算域别名**：fp32 路径 `xc=x_ub`（无中转），fp16/bf16 路径 `xc=xf`；⑤ **`xc` 在链尾原地承载结果**（`vclamp(xc,xc,..)` / `vmin(xc,..,xc)` / `vmax(xc,..,xc)`，`PL-1.23`）；⑥ `y_ub` 仅用于 fp16/bf16 的舍回与 `lo==hi` 常量填充。**op 计数（默认）= 2（fp32）/ 4（fp16·bf16）；折叠形态 = 1/0**。
>
> **接口适配说明**：集成 wrapper `hardtanh.py` 约定 NPU 工厂签名为 `_make_hardtanh_kernel(N, dtype, min_val, max_val, block_size=...)`，返回的 `prim_func` 为 `main(x, y, shape)`（`out_idx=[1]`），调用 `_make_hardtanh_kernel(N, dtype_str, min_val, max_val, block_size=block_size)(x, N)`。本设计按此契约（保留 x/y 名与顺序 + 新增 `shape` 尾守卫）。`is_fp8` 分支为 GPU fp8 专用路径，NPU manifest 不支持 fp8，**不迁移**（§0.5）。
>
> **wrapper 契约遵守**：wrapper docstring 要求「fp32 transit … must be preserved」——本设计**完全遵守**：fp16/bf16 一律 fp32 中转，默认路径 `T.vclamp` 即源 `min(max)` 的融合向量形态。

### 3.4 API 可行性确认

| API | 来源确认（本地） | 状态 |
|-----|-----------------|------|
| `T.vclamp(src, dst, min_val, max_val)`（标量边界，元素级裁剪，前端展开 `npuir_max`+`npuir_min`） | `docs/Tilelang.language/比较操作/T.vclamp.md` §2.1/§2.3/§2.4；`tilelang/language/customize_npuir.py::npuir_clamp` L1128+（`npuir_max`→`npuir_min`）；`testing/npuir/arith_ops/test_clamp_dev.py`（Developer fp16/fp32 实跑） | ✅ 真实可用（dtype fp16 ✓ / fp32 ✓ / bf16 ✗） |
| `T.vmin(src1, src2/dst)`（src2 可标量） | `docs/Tilelang.language/比较操作/T.vmin.md` §2.1/§2.4（`hivm::VMinOp`） | ✅ 真实可用（bf16 ✗） |
| `T.vmax(src0, src1, dst)`（张量-张量） | `docs/Tilelang.language/比较操作/T.vmax.md` §2.4（`hivm::VMaxOp`） | ✅ 真实可用（bf16 ✗） |
| `T.vcast(src, dst, round_mode)` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（`f16→f32` / `bf16→f32` / `f32→f16/bf16` 支持 `rint`） | ✅ 真实可用 |
| `T.copy`（slice） | `examples/elementwise/vec_add_1d.py` | ✅ 已核对 |
| `T.alloc_ub` / `T.fill` | `vec_add_1d.py`；`bitwise_not_fwd_kernel.py` 常量 buffer | ✅ 已核对 |
| `T.Kernel(n, is_npu=True)` + `T.serial` | `vec_add_2d.py`；`PL-1.22-persistent-1d-grid-stride` | ✅ 已核对 |
| `T.cast(scalar, f32)`（编译期常量参与链） | `T.vmul.md` §2.3；`leaky_relu` 同族实现 | ✅ 已核对 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(core_num, is_npu=True)` |
| GPU 专用 API | Yes（源用标量 `T.cast` + `T.min/T.max` + `T.Parallel` + `threads=`） | 换为 v-prefix 向量链 + `is_npu=True`（§0.5/§0.6） |
| GEMM 非整除 | No | 非 GEMM；非整除由 `tail` + `T.ceildiv` 处理（§5.4） |
| L0C 溢出 | No | 无 Cube 计算 |
| 物理核数限制 | Yes | 逻辑核数 = `ceildiv(N, block_size)` 远超物理核 40 → 核内 `T.serial` 网格跨步（§5.5） |
| **bf16 v-prefix 数学 API 缺失** | Yes | `T.vclamp/vmin/vmax` 不含 bf16 → bf16 走 fp32 中转（§0.6 R2） |
| 条件 `T.alloc` 非法 | Yes（编译期分支易误用） | buffer 全部在 `T.Kernel` 顶层无条件分配（§3.3） |
| `T.vmax` 标量边界不确定 | Yes（`hi=+inf` 折叠） | 用预填 fp32 常量 buffer `lo_ub` 作第二操作数（§3.2 注） |
| v-op 动态 slice 不可用 | Yes | v-op 作用于整 buffer，仅 `T.copy` 用 `[0:tail]`（§3.3） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU 源） | 本项目（Ascend npuir） | 转换方案 |
|--------|-------------------|----------------------|----------|
| Kernel 装饰器 | `@tilelang.jit(out_idx=[1])` | `@tilelang.jit(out_idx=[1], target="npuir")` | K1 加 `target="npuir"` |
| Kernel 声明 | `T.Kernel(ceildiv(N,bs), threads=256)` | `T.Kernel(core_num, is_npu=True)` + `T.serial` 网格跨步 | K2/K3 去 `threads=`、加 `is_npu=True`；§0.6 R1 |
| 并行模型 | `T.Parallel(threads, npt)` 线程级 | 一维核 + 向量指令整 buffer | K11：`threads×npt` 折叠为 `block_size` |
| **计算核** | 标量 `T.min(T.max(v32,lo),hi)`（2 op） | 融合 `T.vclamp`（2 op）+ 边界折叠（1/0 op） | §0.6 R2 + §1.6.1 #1 |
| **计算域** | fp32 中转（`v32=T.cast(val,fp32)`） | **同 fp32 中转**（保留源；bf16 API 强制） | §0.6 R2（无域差异） |
| Buffer 层级 | `T.alloc_fragment`（register） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| prim_func 参数 | `main(x, y)` | `main(x, y, shape: T.int32)` | 不改 x/y 顺序与名称，新增 shape 用于尾块守卫 |
| strategy/autotune | `register_copy` + `threads`/`npt` | 无 strategy 维度、无 autotune | K8/K9 移除 |
| fp8 分支 | `is_fp8` 路径（`float8_*` + 独立 `out_dtype`） | 无 | 不迁移（NPU manifest 不支持 fp8） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/leaky_relu/_make_leaky_relu_kernel/` | 高度相似（同族参数化逐元素迁移） | 迁移文档结构、一维核 + 网格跨步、`shape: T.int32` 尾块、`verify_equiv.py` 范式、fp32 中转与标量参数特化 |
| `examples/elu/_make_elu_kernel/` | 高度相似（同族迁移） | 标量参数编译期特化（恒等折叠 / 形态切换）、fp32 中转 |
| `testing/npuir/arith_ops/test_clamp_dev.py` | 高度相似（API 先例） | `T.vclamp(src_ub, dst_ub, lo, hi)` Developer fp16/fp32 用法 |
| `examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py` | 高度相似（迁移形态） | `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步 + `T.fill` 常量 buffer + `shape: T.int32` |
| `examples/elementwise/vec_add_1d.py`·`vec_add_2d.py` | 高度相似 | 一维 `T.Kernel`、`T.alloc_ub`、`T.copy` slice、`T.serial` 网格跨步 |
| `examples/TileOPs/tileops/kernels/elementwise/hardtanh/hardtanh.py` | 迁移目标 | 集成 wrapper + `block_size`（K11）+ `main(x, y, shape)` 契约 + fp32 中转要求 |
| `docs/Tilelang.language/比较操作/T.vclamp.md`·`T.vmin.md`·`T.vmax.md`·`数据类型转换操作/T.vcast.md` | API | 向量链各 op 规格、dtype 矩阵与用法 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(N,)`（原始任意 shape 展平） | `float16 \| bfloat16 \| float32` | 1-D 展平输入向量，`N = product(shape)` 编译期常量 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(N,)` | `same_as(x)` | 输出，shape/dtype 与输入一致 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | `(block_size,)` | `dtype` | UB | 输入 tile；**fp32 路径复用为计算域 `xc` 与结果缓冲** |
| `y_ub` | `(block_size,)` | `dtype` | UB | 输出 tile（**仅 fp16/bf16** 承载 `vcast` 舍回；或 `lo==hi` 常量填充） |
| `xf` | `(block_size,)` | `float32` | UB | fp32 计算域 `xc`（fp16/bf16 = vcast 升级入口，链尾原地复用为结果；fp32 路径空闲） |
| `lo_ub` | `(block_size,)` | `float32` | UB | 常量 `lo`（**仅 `hi=+inf` 折叠**用，只读，循环外 `T.fill` 一次） |

> **buffer 分配策略**：全部在 `with T.Kernel` 顶层**无条件分配**（避免 `TRAP-conditional-alloc`）；`lo_ub` 仅在 `hi=+inf` 折叠时填充；**`xc` 计算域别名**（fp32→`x_ub` / fp16/bf16→`xf`）并在链尾原地承载结果（§3.3，`PL-1.23`）。

### 4.4 内存搬运路径

```
fp32 路径（纯 Vector，无 L1/L0；xc = x_ub）：
GM[x] --T.copy--> UB[x_ub] --[hardtanh 链: vclamp 2 op（原地复用 x_ub 承载结果）]--> UB[x_ub] --T.copy--> GM[y]

fp16/bf16 路径（fp32 中转；xc = xf）：
GM[x(dtype)] --T.copy--> UB[x_ub(dtype)] --T.vcast(rint)--> UB[xf(fp32)]
     --[hardtanh 链（原地复用 xf 承载结果）]--> UB[xf(fp32)] --T.vcast(rint)--> UB[y_ub(dtype)] --T.copy--> GM[y]
```

### 4.5 UB 内存预算

K11 折叠 `block_size`：fp32 = 1024，fp16/bf16 = 2048。

| 路径 | x_ub | y_ub | xf | lo_ub | 合计 | ×1.7（`TRAP-UB-multibuffer-inflation` 最坏膨胀） |
|------|------|------|----|-------|------|---------------------------------------------------|
| fp32 (bs=1024) | 4 KB | 4 KB | 4 KB | 4 KB | **16 KB** | ≈ 28 KB < 192 KB ✓ |
| fp16 (bs=2048) | 4 KB | 4 KB | 8 KB | 8 KB | **24 KB** | ≈ 41 KB ✓ |
| bf16 (bs=2048) | 4 KB | 4 KB | 8 KB | 8 KB | **24 KB** | ≈ 41 KB ✓ |

> UB 硬上限 = 192 KB/AIV（`CONST-capacity-910B2C`，绑定 Ascend910B2C **且 stale**；本设备 Ascend910_9362 同规格量级）。**膨胀系数注（CG-2026-0008）**：显式 alloc 结构实测 ~12% 余量即足（正负两向偏差均无文档，故按最坏上偏 1.7× 保守核对）。**block_size 放大空间**：`bs=4096` → fp16 48 KB（×1.7 ≈ 82 KB，可行）、fp32 64 KB（×1.7 ≈ 109 KB，可行）；`bs=8192` → fp16 96 KB（×1.7 ≈ 163 KB，可行）、fp32 128 KB（×1.7 ≈ 218 KB，**超限**）。故 `bs` 放大是 Stage 4 发射项调优杠杆（§5.2），默认取 K11 折叠值。

> **dtype 差异注**：K11 折叠值 fp32=1024、fp16/bf16=2048（源 `threads(256)×npt`，fp32 npt=4 / fp16·bf16 npt=8）。为可追溯与对齐源，本设计**按 dtype 分别使用 K11 折叠值**（wrapper `_DEFAULT_BLOCK_SIZE`）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| `N` 的实际长度 | `shape: T.int32`（prim_func 参数） | 1 ~ manifest 最大 16,777,216 |

> `N` 同时作为编译期 `T.Tensor((N,), dtype)` 的 shape（按特定 N 特化编译）与运行时 `shape` 参数（尾块守卫）。manifest 全部 workload 的 N 均整除 `block_size`，但保留 `tail` 守卫以支持任意 N（测试 4,000,000 于 bs=2048 余 1216）。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[1],        # y 是 prim_func 第 2 个参数（index=1）
    target="npuir",
)
def _make_hardtanh_kernel(N, dtype, min_val, max_val, block_size=2048):
    ...
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅含逐元素向量链（`vclamp`/`vmin`/`vmax`/`vcast`），无 matmul、无归约、无 Cube 计算；仅需 UB 层级，不涉及 L1/L0A/L0B/L0C。

### 5.2 Block 划分

```python
block_size = {"float32": 1024, "float16": 2048, "bfloat16": 2048}[dtype]   # K11 折叠
n_num = T.ceildiv(N, block_size)   # 逻辑 tile 数（编译期静态值）
```

**block_size 选择理由**：
1. 与源 K11 折叠值一致（fp32 = `threads(256)×npt(4)` = 1024、fp16/bf16 = `threads(256)×npt(8)` = 2048），迁移可追溯；v-prefix 整 buffer 形态下默认值无 128-bit 对齐差异；
2. UB 对齐：block_size×elem_bytes 是 32B 的整数倍（fp32 4096B、fp16/bf16 4096B）✓；
3. UB 容量：最大路径（fp16/bf16 bs=2048）≈ 24 KB，×1.7 最坏膨胀 ≈ 41 KB < 192 KB ✓；
4. **发射项权衡**：本算子为 2–4-op 短链，§1.6.0 估算显示发射项名义主导（发射 ∝ 1/block_size），但实测对照（`bitwise_not`）显示短链逐元素算子实际由流量项主导、发射项被隐藏；`bs` 放大受 UB 膨胀系数不确定性约束。默认取 K11 折叠值；**bs=2048/4096 列为 Stage 4 调优首选杠杆**（须以编译探针校准 UB 后采纳，属分核参数调优，见 `core-split-strategy.md` §2.4）。

### 5.3 约束分析

- **对齐约束**：fp32 bs=1024，元素 4B → 每 tile 256 向量槽（128bit×4），尾 lane 浪费 0%；fp16/bf16 bs=2048，元素 2B → 256 向量槽（128bit×8），浪费 0%；均满足 32B 尾轴对齐 ✓。
- **UB 容量**：最大 24 KB（fp16/bf16 bs=2048），×1.7 最坏膨胀 ≈ 41 KB < 192 KB ✓。
- **L0/L1 容量**：不适用（纯 Vector）。
- **分形限制**：不适用（无 Cube）。

### 5.4 注意事项（非整除处理策略）

**非整除场景预判**：manifest workload 的 N 为 2 的幂（1,048,576 / 8,388,608 / 12,845,056 / 16,777,216）——`[2048,4096]` 与 `[4096,4096]` 及 `[1048576]` 均整除 `block_size`（fp32 1024 / fp16·bf16 2048）；`[16,256,56,56]` = 12,845,056，12,845,056 % 2048 = 512 ≠ 0 → **存在尾块**。测试 shape 4,000,000 **不整除 2048**（余 1216）→ 存在尾块。

**策略**：Kernel 内动态尾块（不改动 host 输入张量、不做 host padding，遵守 `ascend-constraints.md` §4）：最后一个 tile 用 `tail = T.min(block_size, shape - off)` 裁剪 `T.copy` 的搬入/搬出量；向量 op 仍作用于**整 buffer**（尾部未初始化数据不影响仅搬出的 `[0:tail]`）。参考 `examples/elementwise/vec_add_1d.py`。任务网格用 `T.ceildiv(N, block_size)` 保证覆盖最后一个不完整 tile。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

**① 逻辑核数计算**：本算子为 1-D 输出网格 `[N]`，仅 N 轴（无 M 轴，M 项取 1）：

```
num_logical_kernels = ceil(N / block_size)
```

| workload | dtype | N | block_size | 逻辑核数 |
|----------|-------|---|-----------|----------|
| smoke-1m | fp32 | 1,048,576 | 1024 | 1,024 |
| smoke-1m | fp16 | 1,048,576 | 2048 | 512 |
| bounded-hidden `[2048,4096]` | fp16 | 8,388,608 | 2048 | 4,096 |
| bounded-conv-feat `[16,256,56,56]` | fp16 | 12,845,056 | 2048 | 6,272 |
| elementwise-16m `[4096,4096]` | fp32 | 16,777,216 | 1024 | 16,384 |

**② 物理核数依据（实查）**：目标设备 `Ascend910_9362`，实查代码与返回值（纯 Vector 算子核数翻倍）：

```python
from tilelang.utils.npu_utils import NPUUtils
NPUUtils.get().get_aicore_num()        # 实际返回 20
NPUUtils.get().get_aivector_core_num() # 实际返回 40（= aicore × 2）
```

→ **纯 Vector 物理核数 = 20 × 2 = 40**（2026-10-13 本会话实查；与 `CONST-aicore-910_9362` 一致）。注：`pattern-library/constants.md` 的 `CONST-aicore-910B2C` 记 24 核，绑定 Ascend910B2C，**以实查为准**。

**③ 规模判定与分核方案**：**极大规模**——逻辑核数（512 ~ 16,384）远超物理核数 40，且 N 为 2 的幂、物理核数 40 = 2³×5 含非 2 因子，数学上**无法**通过调整 block_size（2 的幂）使逻辑核数对齐 40 的整数倍（5 ∤ 2ⁿ）。按标准 §1 要素③「极大规模」处置：

- **固定启动内核数 = 物理核数 40**；
- 核内 `T.serial` 网格跨步处理多个逻辑 tile：
  ```python
  for i in T.serial(T.ceildiv(n_num, core_num)):   # 循环上界静态（n_num、core_num 均编译期常量）
      task = i * core_num + cid                    # 网格跨步（grid-stride）
      if task < n_num:                             # 守卫
          off = task * block_size
          ...
  ```
- **循环边界静态** ✓（`n_num`、`core_num` 均编译期常量，`T.ceildiv(n_num, core_num)` 为静态整数）；
- **残留负载不均**（显式记录）：以 smoke-1m fp32（bs1024）为例，`tasks_per_core = ceildiv(1024,40) = 26`；网格跨步下 `1024 = 40×25 + 24` → 前 24 个核执行 26 次、后 16 个核执行 25 次 → 临界路径 ≈ **+1.56%**（26 / 25.6）。smoke-1m fp16（bs2048）`ceildiv(512,40)=13`（512 = 40×12+32），临界 **+1.56%**；bounded-hidden fp16 `ceildiv(4096,40)=103`（4096 = 40×102+16），临界 **+0.20%**；bounded-conv-feat fp16 `ceildiv(6272,40)=157`（6272 = 40×156+32），临界 **+0.13%**；elementwise-16m fp16/fp32 `ceildiv(8192,40)=205` / `ceildiv(16384,40)=410`，临界 **+0.10%**。相较「直接 `T.Kernel(n_num)` 由运行时串行调度多波次」，核内 `T.serial` 把核启动固定开销从「每 tile 一次」摊薄为「每核一次」，净收益为正。
- **模式先例**：`examples/elementwise/vec_add_2d.py`；同族 `bitwise_not_fwd_kernel.py`（`PL-1.22-persistent-1d-grid-stride`）。

> 备选（供 Stage 4 参考，不在本设计实现）：直接 `T.Kernel(n_num)` 由运行时调度（无核内串行）——保留为对称性对照，若实测核内串行反而引入同步开销可回退；本设计按标准要求取核内串行 persistent 形态。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block 级 / tile 级调度循环；kernel 内无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级（任务网格） | block 级并行 | `T.Kernel(core_num, is_npu=True)` | 40 个物理 Vector 核各处理一组 tile |
| tile 级（核内） | 串行迭代（网格跨步） | `T.serial(ceildiv(n_num, core_num))` | 每核串行处理 `ceildiv(n_num,40)` 个 tile，摊薄核启动开销（§5.5） |
| 元素级 | 向量化 | `T.vclamp`（默认 2 op）或 `T.vmin`/`T.vmax`（折叠 1 op）+ `vcast` | 整 buffer 一次完成，无 `T.Parallel` 标量循环 |

**T.Persistent 选择说明**：本设计以 `T.serial` 网格跨步实现 persistent 语义（每核固定处理多 tile），与 `T.Persistent` 等价且为同族先例形态（`PL-1.22`）；不额外引入 `T.Persistent`（核数固定 40 已足够）。

### 6.2 循环伪代码

见 §3.3（`with T.Kernel(core_num, ...)` + `for i in T.serial(...)` + `if task < n_num`，循环内为默认 2-op（`vclamp`）或折叠 1/0-op 的 hardtanh 向量链）。

### 6.3 流水线优化

**当前设计**: 不使用 `T.Pipelined`。理由：单 tile 内数据依赖线性（copy-in → [vcast↑] → 1–2-op 链 → [vcast↓] → copy-out），无 K 维迭代；跨 tile 迭代（`T.serial`）的 MTE2/VEC 重叠由 Developer 模式 auto-multi-buffer 自动处理。设计阶段保持简洁；若 Stage 4 需进一步优化（persistent double buffer / 更大 block_size），参照同族 `T.serial` 多 tile 形态。

### 6.4 尾块处理

见 §5.4。`N % block_size != 0` 时最后一个 tile 用 `tail = T.min(block_size, shape - off)` 动态裁剪 `T.copy` 量；manifest `[16,256,56,56]`（12,845,056 % 2048 = 512）与测试 4,000,000（余 1216）触发尾块。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下编译器自动插入同步。核内数据依赖链严格线性（每 tile 内）：

```
T.copy(GM→UB) → [vcast↑] → vclamp (npuir_max → npuir_min) → [vcast↓] → T.copy(UB→GM)
```

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy(GM→UB)` 后 | 自动 | 编译器插 MTE2→V 等待 |
| `vclamp` 内部 `npuir_max → npuir_min` | 自动 | 线性数据依赖（同 buffer 原地），插 V→V barrier |
| `T.copy(UB→GM)` 前 | 自动 | 编译器插 V→MTE3 等待 |
| tile 迭代之间（`T.serial`） | 自动 | 同 buffer 复用由 auto-multi-buffer 管理跨迭代依赖 |

无手动 `T.sync_block_set/wait`（无跨引擎/跨核协作，无 CV 融合）。

### 7.3 pass_configs 配置

```python
# Developer 模式默认配置，无需额外 pass_configs
# os.environ["TILELANG_ASCEND_MODE"] = "Developer"  # 可选，默认即 Developer
```

### 7.4 CV 融合设计

**n/a**：本算子是纯 Vector 逐元素算子，无 Cube 段（无 `T.gemm`）、无跨引擎协作、无核间 workspace，不涉及 CV 融合。据 AGENTS.md「Developer-mode MixCV trigger rule」——kernel 内既无 Cube 侧 `T.gemm` 亦无跨核流水，**非 MixCV**，不触发 mixcv skill。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（`F.hardtanh(x.float(), min_val, max_val).to(dtype)`，保持契约 fp32 opmath 语义），**不得复刻 §0.6 的 NPU 向量链结构**——保证验证独立性。

```python
import torch
import torch.nn.functional as F


def golden_hardtanh(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0) -> torch.Tensor:
    """Hardtanh 参考实现，等价于 torch.nn.functional.hardtanh（契约 fp32 opmath 语义）。

    契约（test_activation.py::HardtanhTest.ref_program）：
        F.hardtanh(x.float(), -1.0, 1.0).to(x.dtype)
    即对 fp16/bf16 输入先升 fp32 计算，末端单次舍回原 dtype；fp32 输入直接计算。
    与源 GPU 非 fp8 分支同域（源亦 fp32 中转）；golden 复用库实现、不复刻 NPU 结构。
    """
    x_f32 = x.detach().cpu().to(torch.float32)
    return F.hardtanh(x_f32, min_val, max_val).to(x.dtype)
```

> 备注：golden **独立于** NPU 向量链（直接调用 `F.hardtanh` 库实现，不复用 §3.3 的 op 序列），满足验证独立性。

### 8.2 精度标准

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 1e-5 | 1e-5 | 来源：`examples/TileOPs/tests/ops/test_activation.py` `_get_tolerances` |
| float16 | 1e-3 | 1e-3 | 来源：同上；NPU 在 fp32 域计算、单次舍回 |
| bfloat16 | 1.6e-2 | 1.6e-2 | 来源：同上；bf16 精度低，容忍度放宽 |

**误差预算核对**：`verify_equiv.py` 实测 fp32 中转链 vs 契约 golden **三 dtype × 三边界组 `max_abs=0.0`（逐位一致）**，远优于门禁（fp32/fp16/bf16 均 0 误差）；clamp 为投影选择、单次舍回，与 golden 舍入路径一致。

### 8.3 L0 门槛测试计划

> L0 = 门槛 smoke 测试（最小可用性验证，进 Stage 3 前必过）；完整分层套件 L1/L2/Boundary 交由 `tilelang-op-develop`，不在此枚举。**显式包含上下边界处/内外、x=0、各 dtype、非整除尾块、退化边界折叠路径**。

| 用例 | 输入 | 期望 | 判定 |
|------|------|------|------|
| L0-1 fp32 smoke（randn 全域） | `randn([1048576], fp32)`（NPU） | 与 `golden_hardtanh(x,-1,1)` 比对 ≤1e-5 | 门槛通过 |
| L0-2 fp16 smoke（randn 全域） | `randn([1048576], fp16)`（NPU） | 比对 ≤1e-3 | 门槛通过 |
| L0-3 bf16 smoke（randn 全域） | `randn([1048576], bf16)`（NPU） | 比对 ≤1.6e-2 | 门槛通过 |
| L0-4 fp32 角点（边界内外） | `repeat([-88.8,-1.0,-0.5,0.0,0.5,1.0,88.8], 4096, fp32)`（NPU，`test_hardtanh_edge` 同款） | 与 golden 比对 ≤1e-5（`-88.8→-1.0`、`-1.0→-1.0`、`-0.5→-0.5`、`0→0`、`0.5→0.5`、`1.0→1.0`、`88.8→1.0`） | 门槛通过 |
| L0-5 上下界邻域 | `linspace(-1.1, 1.1, 1<<20, fp32)`（NPU） | 与 golden 比对 ≤1e-5（覆盖两侧触发与线性内区） | 门槛通过 |
| L0-6 极值 | `repeat([-65504.0, 65504.0, float('-inf'), float('inf')], 4096, fp32)`（NPU） | `±65504 → ±1.0`、`+inf→1.0`、`-inf→-1.0` | 门槛通过 |
| L0-7 非整除尾块 | `randn([4000000], fp16)`（NPU） | 比对 ≤1e-3（覆盖 4000000 % 2048 = 1216 尾块） | 门槛通过 |
| L0-8 manifest 多 shape | `[2048,4096]`（fp16/bf16，8,388,608）、`[16,256,56,56]`（fp16/bf16，12,845,056 % 2048 = 512 尾块）、`[4096,4096]`（fp16/bf16/fp32）（NPU） | 比对 ≤1e-3 / ≤1.6e-2 / ≤1e-5 | 门槛通过 |
| L0-9 退化边界折叠（`lo=-inf`） | `randn([1048576], fp32)`，`min_val=-inf, max_val=1.0`（NPU） | 与 `golden_hardtanh(x,-inf,1.0)` 比对 ≤1e-5（覆盖 `vmin` 折叠路径；`verify_equiv.py` #1 证逐位一致） | 门槛通过 |
| L0-10 退化边界折叠（`hi=+inf`） | `randn([1048576], fp32)`，`min_val=-1.0, max_val=+inf`（NPU） | 与 `golden_hardtanh(x,-1.0,+inf)` 比对 ≤1e-5（覆盖 `vmax`+`lo_ub` 折叠路径） | 门槛通过 |

- **执行方式**：`tilelang.compile(func, target="npuir")` 后单次调用，`torch.testing.assert_close(y.cpu(), golden, atol, rtol, equal_nan=True)`（golden = §8.1）。
- **门槛标准**：L0-1..L0-10 全过即进入 Stage 3；任一失败不得继续。
- 输入生成参考 `examples/TileOPs/tileops/workloads/elementwise.py`；边界用例参考 `test_activation.py::HardtanhTest` / `ActivationEdgeFixture`（`repeat([-88.8,-1.0,-0.5,0.0,0.5,1.0,88.8])`）。
- **额外设计期落点**：`verify_equiv.py`（D-1 机器验证，20/20 `EQUIV_PASS`，§1.6.1）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 缓解方案 |
|------|------|----------|
| **R-1 bf16 v-prefix 数学 API 缺失**（`vclamp`/`vmin`/`vmax` 各 op doc §2.2.1） | bf16 直传编译失败 | bf16 走 fp32 中转（§0.6 R2，API 强制） |
| **R-2 计算域 = fp32 中转**（保留源 + 契约对齐） | 若误改为输入 dtype 域 fp16 直算，**数值上仍逐位等价（机器证）**但违背 wrapper 契约且 bf16 不可用 | 全 sub-fp32 统一 fp32 中转（§0.6 R2 / §1.6.1 #2）；wrapper docstring 契约遵守 |
| **R-3 边界折叠默认不触发** | 默认 `(-1,1)` 两侧有限，op 数仍为 2（与源一致），fold 无默认收益 | 诚实记录（§1.6.1 #1）；fold 面向 `±inf`/`lo==hi` 退化参数域；默认路径 2 op 已为最短闭式 |
| **R-4 NaN 传播（设备语义未验证）** | torch `clamp`/`hardtanh` 传播 NaN；设备 `VMaxOp`/`VMinOp` 若为 IEEE `maxNum` 语义则可能不传播 | 契约/测试 workload（randn + 有限角点）不含 NaN；Stage 3 增加 NaN 设备探针核对；`equal_nan=True` 口径；列入待确认 |
| **R-5 发射项主导（名义）** | §1.6.0 估算示 emission 名义主导；但实测对照（`bitwise_not`）显示短链逐元素实际流量主导 | 标注「未文档化假设 + 估算依据」（常数绑 910B2C、stale、可能上偏）；block_size 为杠杆（§5.2）；Stage 4 baseline 后按 `hardware-cost-model.md` §3 对账 |
| **R-6 UB 容量** | bs 放大受限 | 默认按 dtype K11 折叠值（最大 24 KB×1.7≈41 KB 安全）；buffer 复用（`xc` 原地）；bs 放大须编译探针校准 |
| **R-7 条件 `T.alloc` 非法**（`TRAP-conditional-alloc`） | 编译期 `if` 内 `T.alloc_ub` trace 失败 | buffer 全部在 `T.Kernel` 顶层无条件分配（§3.3）；`xf`/`lo_ub` 在未用路径仍分配 |
| **R-8 v-op 整 buffer vs 动态 slice** | 传动态 slice 报错 | v-op 作用于整 buffer，仅 `T.copy` 用 `[0:tail]`（§3.3） |
| **R-9 vcast round_mode 限制**（`T.vcast.md` §2.2.1） | `f16→f32` / `bf16→f32` / `f32→f16/bf16` **仅 `rint`** | 中转升降统一 `round_mode="rint"`（与 golden 单次舍回一致） |
| **R-10 `T.vmax` 标量边界不确定** | 折叠 `hi=+inf` 若用 `vmax(xc, scalar, xc)` 有操作数规则风险（`TRAP-expert-v-operands` 同族） | 用预填 fp32 常量 buffer `lo_ub` 作第二操作数（§3.2 注） |
| **R-11 分核残留不均** | 40 核含非 2 因子 5，2ⁿ 逻辑核无法整除 | §5.5 网格跨步核内串行，残留临界 +0.10%~+1.56%，已显式记录 |
| **R-12 常数表设备/版本** | `CONST-vector-launch-overhead`/`CONST-capacity-910B2C` 绑 910B2C 且 `kb_stale_check` 标记 **stale**；本设备 910_9362 | 带宽/发射/容量数字标注「待重验」；流量项优先用 `CONST-ascend910-9362-profile`（verified/实测）；Stage 4 baseline 后对账 |
| **R-13 元算设备语义** | 文档语义可能与设备不一致 | 关键 API `vclamp`（→`VMaxOp`/`VMinOp`）/`vcast` 有 `test_clamp_dev.py` Developer 实跑先例；工具链升级后须重验 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| bf16 直接传入数学 op | 未做 vcast 上转 | 编译失败 | bf16 先 `vcast` 到 fp32（R-1） |
| vcast 用非 rint 模式升级 | `f16→f32` 用 `round`/`floor` | 编译失败（仅 `rint`） | 升降统一 `round_mode="rint"`（R-9） |
| 条件 alloc | 编译期 `if need_cast:` 内 `T.alloc_ub` | trace 报 `Undefined variable` | buffer 顶层无条件分配（R-7） |
| v-op 传动态 slice | `T.vclamp(x_ub[0:tail], ...)` | 可能编译/运行错误 | v-op 用整 buffer（R-8） |
| `vmax` 传标量边界 | `T.vmax(x_ub, lo, ...)` | 可能编译失败（标量操作数） | 用常量 buffer `lo_ub`（R-10） |
| UB 溢出 | block_size 过大 | 编译失败 | 维持 K11 折叠值（默认）；§4.5 预算 + 编译探针（R-6） |
| 尾块越界 | N 非 bs 整数倍且无 tail 裁剪 | 越界读/写 GM | `tail = T.min(block_size, shape-off)`（§5.4） |
| 边界常量未填 | 折叠 `hi=+inf` 直接用未初始化 `lo_ub` | `vmax` 语义错误 | 循环外 `T.fill(lo_ub, cast(lo,f32))`（§3.3） |
| 零输入崩溃 | N=0 空张量 | MTE DDR 崩溃（`TRAP-zero-input-crash`） | host 侧空张量守卫；本任务 N ≥ 1e6，不触发 |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| **非整除 N** | Kernel 内 `tail` 动态裁剪（§5.4），无需 host padding |
| **极小 N（N < block_size）** | `n_num=1`，`tail=N`，仅 `task==0` 的核工作，其余核空转 |
| **极大 N（>1.6e7）** | 核内 `T.serial` 网格跨步（§5.5），逻辑核数静态 |
| **`±inf` 输入** | `+inf → min(max(+inf,lo),hi)=hi`；`-inf → lo`（§0.1 边界语义） |
| **`x=0` / `-0.0`** | `clamp(0)=0`、`clamp(-0.0)= -0.0`（在区间内原样，含符号） |
| **`x` 为大正/负数** | 投影到 `hi`/`lo`（无溢出，clamp 为选择） |
| **退化边界** | `lo=-inf` → `vmin`；`hi=+inf` → `vmax`+`lo_ub`；两侧 inf → copy；`lo==hi` → 常量 `fill`（§1.6.1 #1） |
| **混合 dtype** | 按 dtype 编译期特化（每 dtype 一份 kernel），输出 `same_as(input)` |
| **`inplace=True`** | host 侧 `input.copy_(result)`（不在 kernel 内，§0.1） |

---

## 10. 交付清单

| 交付物 | 路径 | 说明 |
|--------|------|------|
| 设计文档 | `examples/hardtanh/_make_hardtanh_kernel/DESIGN.md` | 本文件 |
| 等价性机器验证 | `examples/hardtanh/_make_hardtanh_kernel/verify_equiv.py` | D-1：20/20 `EQUIV_PASS`（§1.6.1） |
| NPU kernel（下游 Stage 3 产出，尚未创建） | 算子目录下 `_make_hardtanh_kernel.py` | 按 §3.3 伪代码 + §5 分核实现 |
| Golden（下游） | 同上（kernel 文件内 `golden_hardtanh`） | 依 §8.1 |
| 迁移集成 wrapper（已存在，只读参照） | `examples/TileOPs/tileops/kernels/elementwise/hardtanh/hardtanh.py` | K11 `block_size` + `main(x,y,shape)` 契约 + fp32 中转要求 |
| 抽取脚手架（已存在，只读参照） | `examples/TileOPs/tileops/kernels/elementwise/hardtanh/_hardtanh_fwd_kernels.py` | Pattern B 源直译参照 |
| 精度用例（已存在，只读参照） | `examples/TileOPs/tests/ops/test_activation.py` | `test_hardtanh` / `test_hardtanh_edge` |
| 性能用例（已存在，只读参照） | `examples/TileOPs/benchmarks/ops/bench_elementwise_manifest.py` | manifest-driven bench |

**实现要点速览（Stage 3 输入）**：
1. **计算域 = fp32 中转**（fp16/bf16 先 `vcast(rint)` 升 fp32，链尾 `vcast(rint)` 舍回；fp32 直算）——**遵守 wrapper 契约**；
2. **计算核 = `T.vclamp`（默认 2 op）+ 编译期边界折叠**（`lo=-inf→vmin` 1 op / `hi=+inf→vmax` 1 op / 两侧 inf→copy 0 op / `lo==hi→fill` 常量）；
3. 结构：一维 `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步 + `tail` 尾块守卫；
4. `block_size` = K11 折叠值（fp32 1024 / fp16·bf16 2048）；buffer 顶层无条件分配；v-op 作用于整 buffer；`hi=+inf` 折叠用 `lo_ub` 常量 buffer 规避 `vmax` 标量不确定性；
5. `bf16` **必须** fp32 中转（`vclamp`/`vmin`/`vmax` 无 bf16）。
