# _make_leaky_relu_kernel 算子设计文档（迁移）

> **项目**：`leaky_relu` ｜ **算子目录**：`examples/leaky_relu/_make_leaky_relu_kernel/` ｜ **编程模式**：Developer ｜ **target**：`npuir`
>
> **任务性质**：迁移（migration-harness）。把 GPU TileLang 工厂函数 `_make_leaky_relu_kernel`（`y = T.if_then_else(val > zero, val, slope * val)`，非 fp8 分支**在输入 dtype 域**计算）重设计为 NPU（`target="npuir"`）实现。
>
> **本版核心决策（本算子最关键点）**：**计算域由「输入 dtype 域」改为「fp32 中转域」**。契约 golden 是 `F.leaky_relu(x.float(), 0.01).to(dtype)`（**fp32 域**计算），而源 GPU 非 fp8 分支用 `slope = T.cast(0.01, val.dtype)` 在**输入 dtype 域**相乘——`0.01` 在 fp16/bf16 不可精确表示（`cast(0.01,fp16)=0.010002136`、`cast(0.01,bf16)=0.010009766`）。设备探针（`history_version/design_probe_leaky_relu.py`）+ `verify_equiv.py` **双实测**：输入 dtype 域对 golden 的角点违反比达 **0.2597**（fp16，`max_abs=4.883e-4`）、bf16 达 **0.1297**（仍 PASS 但余量小）；fp32 中转域对 golden **逐位一致**（`max_abs=0.0`，违反比 0）。且 bf16 因 `T.vmul` 不支持 bf16（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.1）**必须** fp32 中转。故选定 **fp32 中转计算域**（更贴近契约，余量最大，且 bf16 强制）。
>
> **第二决策（算法级优化）**：编译期特化——`0 < negative_slope <= 1`（默认 0.01 命中）时用 **`y = max(x, slope·x)`**（`vmul + vmax`，**2 op**）替代 `vcmp + vmul + vselect`（3 op）；`negative_slope>1` 或 `<=0` 回退 where 形式。设备探针 + `verify_equiv.py` 双证 `max` 形式与 where 形式/ golden 在 `0<s<=1` 上**逐位一致**；`s>1` 诊断证明 guard 必要（diagnostic 非门禁）。**负向事实**：默认 `slope=0.01` 一般**不能**省 `vmul`（不像 `slope==1.0` 的恒等折叠）——`0.01` 既非 1 亦非 0，乘法不可消除。
>
> **源优化意图承接**：源 #1 `register_copy`（fragment load→compute→store）→ NPU `GM→UB→GM` 整块 `T.copy` + 整 buffer v-prefix；源 #2 block×thread 两级并行 → 一维 `T.Kernel(40, is_npu=True)` + 核内 `T.serial` 网格跨步；源 #3 输入 dtype 域算术 → **等价替换为 fp32 中转**（实测更贴近 golden，见上）；源 #4 strategy `npt` 对齐 → K11 `block_size` 折叠 + 32B 对齐复算；源 #5 无 autotune → 保留。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织：先彻底读懂源算子（0.1–0.4），再做算法调研（Phase R，结论见 §1.6.0），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。

**源算子路径（源真相）**：`/home/developer/TileOPs/tileops/kernels/elementwise.py`

- 具体算子：工厂 `_make_leaky_relu_kernel`（L2427–L2478）+ `LeakyReluFwdKernel(ParametricUnaryKernel)`（L2481–L2493，`negative_slope` compile-time 特化）：

  ```python
  @functools.lru_cache(maxsize=32)
  def _make_leaky_relu_kernel(N, dtype, negative_slope, output_dtype=None,
                              is_fp8=False, threads=256, npt=8):
      out_dtype = output_dtype or dtype
      block_size = threads * npt
      if is_fp8:                                     # fp8 分支（本迁移不涉及）
          accum = _fp8_accum_dtype_str()             # = "float16"
          @tilelang.jit(out_idx=[1])
          def kernel(threads_arg, npt_arg):
              @T.prim_func
              def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
                  with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                      for i, j in T.Parallel(threads_arg, npt_arg):
                          idx = (bx * threads_arg + i) * npt_arg + j
                          if idx < N:
                              val = x[idx]
                              v = T.cast(val, accum)          # fp16 accum
                              zero = T.cast(0, accum)
                              slope = T.cast(negative_slope, accum)
                              result = T.if_then_else(v > zero, v, slope * v)
                              y[idx] = T.Cast(out_dtype, result)
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
                          zero = T.cast(0, val.dtype)
                          slope = T.cast(negative_slope, val.dtype)   # ★ 输入 dtype 域
                          y_reg[i * npt_arg + j] = T.if_then_else(val > zero, val, slope * val)
                      T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])
              return main
      return kernel
  ```

- 基类与模板：`ParametricUnaryKernel`（GPU 端共享基类，提供 `_builder_fn`/`_builder_args`/`default_config`/`forward`；NPU 侧不提供该基类，`leaky_relu.py` 已内联，E6）。
- 提取产物（脚手架）：`examples/TileOPs/tileops/kernels/elementwise/leaky_relu/_leaky_relu_fwd_kernels.py`（Pattern B，含 `_fp8_accum_dtype_str` 手工补齐）；集成 wrapper：同目录 `leaky_relu.py`（K11：`block_size` = 1024 fp32 / 2048 fp16·bf16；`npub::leaky_relu_fwd`；签名 `_make_leaky_relu_kernel(N, dtype, negative_slope, block_size=...)`，返回 `main(x, y, shape)`）。

### 0.1 源算子语义（做什么）

**数学语义**：`y = x if x > 0 else negative_slope * x`，逐元素（manifest `LeakyReluFwdOp.ref_api = torch.nn.functional.leaky_relu`，`examples/TileOPs/tileops/manifest/elementwise.yaml` L263–L305）。

**契约语义 = 测试 golden `F.leaky_relu(x.float(), 0.01).to(dtype)`**（`examples/TileOPs/tests/ops/test_activation.py` L145–L190，`LeakyReluTest.ref_program`）。容差：fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2（`_get_tolerances`）。

**关键核对（本算子的核心分歧点）——golden 的 `0.01` 在哪个域参与乘法？**

| 域 | 表示 | 与 torch golden 的一致性（实测） |
|----|------|---------------------------------|
| **契约 golden** | `F.leaky_relu(x.float(), 0.01).to(dtype)` | 基线：fp32 域算 `0.01·x`，末端单次舍回原 dtype |
| 源 GPU 非 fp8 分支 | `slope = T.cast(0.01, dtype)`，`slope · val` 在**输入 dtype 域** | fp16：`cast(0.01,fp16)=0.010002136`（rel 2.14e-4）；bf16：`0.010009766`（rel 9.77e-4）；fp32：`0.00999999978`（rel 2.24e-8，≈恒等） |

→ **源实现在 fp16/bf16 上是在输入 dtype 域做乘法，与 fp32 域的契约 golden 存在可量化的偏差**（见 §0.6 R3 / §1.6.1 的实测：fp16 角点违反比 0.2597、bf16 0.1297）。该分支的注释“matching `torch.nn.functional.leaky_relu`'s per-dtype arithmetic”只在“对 fp16 张量直接调用 `F.leaky_relu`”时成立；**本任务的契约 golden 显式 `x.float()` 上转**，故契约域是 fp32。这一分支必须在迁移中显式处置（§0.5 / §0.6 R3）。

**dtype 语义（迁移必须保持的关键点）**：源**非 fp8 分支不做 fp32 中转**——`x_reg`/`y_reg` 与 `slope` 均为输入 dtype，即 fp16/bf16 全程半精度乘 + 半精度二路选择，仅一次隐式舍回（写 `y_reg`）。而 golden 是 fp32 opmath + 一次舍回。二者语义不同（浮点层面），须择一并论证（§0.6 R3）。

**`negative_slope` 语义**：compile-time 标量（`LeakyReluFwdKernel._builder_args → (negative_slope,)`；工厂经 `functools.lru_cache` 按 `negative_slope` 特化）。Op 层参数默认 0.01（manifest `params.negative_slope.default = 0.01`）；本设计据此把 `negative_slope` 当**编译期 Python 常量**贯穿，并按常数取值做编译期特化（§0.6 R4 / §1.6.1 #2）。

**`inplace` 语义**：host 侧处理，不在 kernel 内（manifest `params.inplace` 默认 false；同族 `elu.py` 的 `forward` 在 `inplace=True` 时 `input.copy_(result.reshape(input.shape)); return input`）。迁移保持：kernel 无 inplace 分支，别名由 host `copy_` 完成（kernel 只处理 1-D `[N]`）。

**规约语义**：无规约（逐元素映射，无跨元素依赖），不存在累加顺序问题。

**边界语义**（逐元素，两支在 `x=0` 同值）：`leaky_relu(0)=0`（`0>0` 假 → `0.01·0=0`，两支同值）；`leaky_relu(+∞)=+∞`；`leaky_relu(−∞)=0.01·(−∞)=−∞`；`leaky_relu(NaN)=NaN`（`NaN>0` 假 → `0.01·NaN=NaN`）。**注意**：当 `negative_slope==0` 时 golden 在 `x=−∞` 处为 `0·(−∞)=NaN`（torch 实测），故 `max(x,0)`/relu 形式在 `slope==0` 的 `−∞` 角点**非等价**（本设计 guard 只覆盖 `0<s<=1`，`slope==0` 走 where 回退，回退式在该角点亦得 `0·(−∞)=NaN`，与 golden 一致）。空维度/非整除 shape 由 host 侧 flatten 成 1-D `[N]`。

**host 侧语义**：`ParametricUnaryKernel`/`LeakyReluFwdOp` 把任意 shape 展平为 `N_total = numel`；`forward` 仅调用编译后 kernel（`_leaky_relu_fwd_wrapped`），无 host 预处理；`inplace` 后处理如上。

**语义保持基线**：§8.1 golden 以本节语义（`F.leaky_relu(x.float(), negative_slope).to(dtype)`）为唯一依据实现，**不得复刻 §0.6 的 NPU 结构决策**（保证验证独立性）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `(N,)`（原始 4-D shape 被 host 侧展平；manifest `[1048576]` / `[16,256,64,64]` / `[16,512,32,32]` / `[4096,4096]`） | `float16 \| bfloat16 \| float32` | 1-D 展平输入向量，`N = product(input.shape)` |
| `out` | `same_as_input`（输出 shape = 输入 shape，展平后 `(N,)`） | `same_as(input)` | 逐元素 LeakyReLU 输出 |

**输出 shape（迁移必需）**：`same_as_input`（manifest `shape_rules: output.shape == input.shape`；`source_output_shape = same_as_input`）。
**layout**：1-D 连续（contiguous flatten），无转置。
**动态轴**：`N` 在编译期由宿主 shape 决定（`lru_cache` 按 N 特化）；尾块守卫以运行时 `shape: T.int32` 参数支持任意 N。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（非 fp8 分支，`register_copy` 式 fragment load→compute→store，源 L2461–L2476）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `x_reg ← x[bx*bs:(bx+1)*bs]`（fragment 载入，`bs=threads×npt`） | GM `x` | fragment `x_reg` | 元素定位/搬入 |
| 2 | `val = x_reg[i*npt+j]`；`zero = cast(0, val.dtype)`；`slope = cast(negative_slope, val.dtype)` | `x_reg` | 标量 `val`/`zero`/`slope` | 常量特化（输入 dtype 域） |
| 3 | `y_reg[i*npt+j] = if_then_else(val > zero, val, slope * val)` | `val`,`zero`,`slope` | `y_reg` | 负分支乘 + 二路选择（输入 dtype 域） |
| 4 | `y[bx*bs:(bx+1)*bs] ← y_reg`（fragment 写回） | `y_reg` | GM `y` | 搬出 |

**数据流与内存访问模式**（源硬件 = GPU）：

```
GM[x] --(fragment load, coalesced)--> registers[x_reg]
   --(T.Parallel(threads,npt) 逐线程标量：cast→cmp→mul→if_then_else)--> registers[y_reg]
   --(fragment store, coalesced)--> GM[y]
（无显式 SMEM；fragment 双缓冲由编译器管理；block_size = threads × npt）
```

**循环与并行结构**：`T.Kernel(T.ceildiv(N, block_size), threads=256)`——一维 grid（`bx`）+ 256 线程/block 两级并行；每线程 `npt` 元素，`T.Parallel(threads_arg, npt_arg)`。`block_size = 256 × npt`：fp32 `npt=4`→1024，fp16/bf16 `npt=8`→2048（`ParametricUnaryKernel.default_config`）。

**host 侧逻辑**：`N_total`、`dtype`、`negative_slope` 由 Op 层传入；`forward` 直接调用编译后 kernel；flatten 由 Op 层完成（wrapper `leaky_relu.py` 的 `_DEFAULT_BLOCK_SIZE` = {fp32:1024, fp16:2048, bf16:2048}）；`is_fp8` 分支由 GPU fp8 专用路径触发，本迁移不涉及。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | `register_copy`（fragment load→compute→store） | 向量化/合并访存 | `T.alloc_fragment` + `T.copy` 切片 + `T.Parallel` 标量映射 | GPU 大寄存器堆 + memory coalescing | **硬件强相关**（fragment/register 模型） |
| 2 | block×thread 两级并行（大 grid + 256 线程/block） | 用大规模线程/grid 吃满内存带宽 | `T.Kernel(ceildiv(N,bs), threads=256)` + `T.Parallel` | GPU thread/warp 执行模型 | **硬件强相关**（GPU 线程模型） |
| 3 | **输入 dtype 域算术**（`slope = T.cast(negative_slope, val.dtype)`） | 免去 fp32 中转的 cast 开销 + 对齐“对 fp16 张量直接调用 `F.leaky_relu`”的逐 dtype 语义 | slope 常量在输入 dtype 表示，全程半精度乘 | 半精度 mul（GPU fp16 原生高吞吐） | **语义层**（计算域选择，须与契约 golden 核对）；**API 层可移植** |
| 4 | strategy-aware `npt`（fp32=4、fp16/bf16=8） | 访存对齐（4×4B / 8×2B = 128bit） | `default_config` 启发式 | memory coalescing（对齐） | 实现层（NPU 折叠为 `block_size`） |
| 5 | 无 autotune（elementwise 固定启发式配置） | 带宽受限 + 单步逐元素，搜索 ROI 低 | `default_config` 固定 `threads=256`+`npt` | 无硬件依赖 | 可移植（NPU 侧同样固定单实现） |
| 6 | 单条比较+乘+选择完成逐元素映射（3 op，无超越函数） | 极低 op 数 | `if_then_else` + 一次 `mul` | GPU 逐元素 ALU | **算法层，可移植**：NPU 有 `vcmp/vselect/vmul`，且有更短的 `vmul+vmax` 等价形态（§1.6.1 #2） |
| 7 | 工厂 `lru_cache` 按 `(N, dtype, negative_slope, ...)` 特化 | 消运行时参数分派 | compile-time 常量固化 | 无硬件依赖 | 可移植（NPU 侧同用编译期常量特化） |

> 识别不出来 ≠ 不存在。上述 7 项为源码中可识别的全部优化/规避手段；后续迁移逐项处置（§0.5）。**未识别优化会被静默丢弃**，故此处列全。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 Phase R 调研结论 §1.6.0）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `y = x if x>0 else slope*x`（含 I/O、边界、`inplace` host 契约） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 依 §0.1；计算核不做语义替代 |
| 计算算法（步骤 1–4 逐元素映射结构，含二路选择） | 算法 | 无（纯数学顺序、无跨元素依赖） | — | **保留（结构）** | 逐元素映射结构保留；仅并行/内存/计算域重设计 |
| 优化 #3 输入 dtype 域算术（半精度乘 + 半精度选择） | 语义/优化 | GPU 半精度 mul 原生 | **有**（`T.vmul` 支持 fp16） | **等价替换**（→ fp32 中转） | → §0.6 R3：契约 golden 在 fp32 域，实测 fp32 中转**逐位一致**而输入 dtype 域偏差（fp16 角点违反比 0.2597）；且 bf16 `T.vmul` 缺失强制 fp32 中转。**性能意图（免 cast）让位于正确性（匹配契约）** |
| 优化 #1 `register_copy`（fragment + `T.Parallel` 标量映射） | 优化 | GPU register/fragment 模型 | 无直接等价（NPU 无一维 `threads=`；v-prefix op 作用于整 UB buffer） | **重新设计** | → §0.6 R1：GM→UB→GM + 整 buffer v-prefix 向量链 |
| 优化 #2 block×thread 两级并行 | 优化 | GPU thread/warp 模型 | 无直接等价（NPU 一维 Kernel） | **重新设计** | → §0.6 R1：一维 `T.Kernel(40, is_npu=True)` + 核内 `T.serial` 网格跨步 |
| 优化 #6 计算核（`cmp+mul+select`） | 算法 | GPU 逐元素 ALU | **有**：`vcmp/vselect/vmul/vmax` | **等价替换（并优化）** | → §0.6 R2 + §1.6.1 #2：`0<s<=1` 用 `vmul+vmax`（2 op）替代 `vcmp+vmul+vselect`（3 op），其余走 where 回退 |
| 优化 #4 strategy-aware `npt` 对齐 | 优化 | memory coalescing（对齐） | 有（32B 尾轴对齐 / 向量宽度 128bit） | **等价替换** | K11：`threads × npt` 折叠为 `block_size`（§5.2），对齐约束按 NPU 32B 复算（§5.3） |
| 优化 #5 无 autotune | 优化 | 无 | 有 | **保留** | NPU 侧同固定单实现（无 strategy/autotune） |
| 优化 #7 `lru_cache` 常量特化 | 优化 | 无 | 有 | **保留** | NPU 工厂保持 `negative_slope`/`N`/`dtype` 编译期常量；并借此做 slope 编译期特化（§1.6.1 #2） |
| fp8 分支（`is_fp8`，fp16 accum + `float8_*`） | 优化 | GPU fp8 专用路径 | 无（NPU manifest 不支持 fp8，`SUPPORTED_DTYPES` 仅 float16/bf16/float32） | **舍弃** | manifest 输入 dtype 仅 fp16/bf16/fp32；fp8 不在契约内，不迁移（依据 `elementwise.yaml` L271 + `leaky_relu.py` `SUPPORTED_DTYPES`） |
| host 侧 flatten / `inplace` copy_（非 kernel 逻辑） | 算法（host） | 无 | — | **保留** | Op 层传入 `N_total`，kernel 只处理 1-D；`inplace` 由 host（§0.1） |

> **依据**：#3 的 vcast/中转可用性由 `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（`f16→f32` / `f32→f16/bf16` 支持 `rint`）佐证；#6 的 `vmul+vmax` 由 `docs/Tilelang.language/数学操作/T.vmul.md`·`比较操作/T.vmax.md` §2.2.1 佐证；两者另有本会话设备探针实测（§3.4）。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（承接源方案性能意图 + 语义保持论证）。

**重设计项 R1：并行结构重设计（thread 级并行 → 一维 Kernel + 核内网格跨步）**

- **源方案**：`T.Kernel(ceildiv(N, block_size), threads=256)` + `T.Parallel(256, npt)`，每线程 `npt` 元素；性能意图 = 用大规模线程/grid 吃满内存带宽。
- **NPU 新算法**：一维 `T.Kernel(core_num, is_npu=True)`（`core_num` = 实查物理 Vector 核数 **40**，§5.5 ②）+ 核内 `for i in T.serial(T.ceildiv(n_num, core_num))` 网格跨步任务映射（`task = i * core_num + cid`，`if task < n_num` 守卫）；每 tile 经 `T.copy` GM→UB、整 buffer 向量 op、`T.copy` UB→GM。
- **语义保持论证**：任务映射只重排「哪个核处理哪个元素区间」，各元素计算完全独立（无跨元素依赖、无累加顺序变化），数学语义逐元素恒等；边界：`T.serial` 循环上界 `T.ceildiv(n_num, core_num)` 为编译期静态值，守卫 `task < n_num` 覆盖非整除尾块，不产生越界。
- **模式先例**：`examples/elementwise/vec_add_2d.py` L28–L42、同族迁移 `examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py` L89–L108（`PL-1.22-persistent-1d-grid-stride`）。

**重设计项 R2：计算核重设计（标量 `if_then_else` → 向量链，含编译期特化）** ⭐

- **源方案**：`y = T.if_then_else(val > zero, val, slope * val)`——GPU 标量逐元素比较 + 乘 + 选择（3 op）。
- **NPU 新算法（两形态，按编译期 `negative_slope` 特化）**：

  $$
  \text{max path}(0<s\le 1):\quad y = \max(x,\; s\cdot x)\qquad(\texttt{vmul}+\texttt{vmax},\ 2\text{ op})
  $$
  $$
  \text{where path}(s\le 0 \text{ or } s>1):\quad y = \operatorname{vselect}(x>0,\; x,\; s\cdot x)\quad(\texttt{vcmp}+\texttt{vmul}+\texttt{vselect},\ 3\text{ op})
  $$

  两形态均为**单条无分支向量链**（整 UB buffer）；`T.vmax` 无标量广播，故默认路径用 `vmul(xc, s, t)` 生成 `t`，再 `vmax(xc, t, xc)` 选大。

- **语义保持论证（逐值等价 + 边界核对）**：
  1. **默认路径 `max(x, s·x)` 与契约公式逐位等价（`0<s≤1`）**：`x>0` 时 fp32 域 `round(s·x) ≤ x`（真值 `s·x<x` 且 round-to-nearest 不会越过 `x`）→ `max=x`；`x≤0` 时 `round(s·x) ≥ x` → `max=round(s·x)=s·x`，而 `s·x` 是单次 fp32 乘、与 golden 负分支同一个乘积 → 逐位一致。`verify_equiv.py` #2 证（fp32/fp16/bf16 `max_abs=0.0`），设备探针证（fp32 角点 `max_abs=0.0`）。
  2. **回退路径**：与源公式同式（`vcmp+vmul+vselect`），在 fp32 域计算，与 golden 负分支同一个 `s·x` 乘积；`verify_equiv.py` #5 证（slope 0.5/2.0 精确）。
  3. **分支点 `x=0`**：`max(0, s·0)=0`；golden `0`；`where` 回退 `0>0` 假 → `s·0=0`。三式同值 0，无跳变。
  4. **极值/边界**：`x=+∞ → max(+∞, s·(+∞)=+∞)=+∞`（golden `+∞` ✓）；`0<s≤1` 时 `x=−∞ → max(−∞, −∞)=−∞`（golden `s·(−∞)=−∞` ✓）；`x=NaN → max(NaN,NaN)=NaN`（golden `NaN` ✓）。**`s==0` 的 `−∞` 角点**（golden `0·(−∞)=NaN`）落在回退路径（`s=0` 不满足 `0<s`），回退式 `0·(−∞)=NaN` 与 golden 一致——故 guard 定义为**严格** `0<s≤1`。
  5. **设备端到端实测**（本会话探针）：fp32 域 where/max 两形态 fp32 `max_abs=0.0`、fp16 randn `3.05e-5`（viol_ratio 0.0296）、bf16 randn `2.44e-4`（0.0148）；fp16/bf16 角点 `0.0`；小 x 敏感段 `linspace(-2e-3,2e-3)` `0.0`；非整除尾块（4,000,000 % 2048 = 1216）`0.0`；slope 0.5/1.0/0.001 fp32 `0.0`——全部远优于门禁。
  6. **拒绝项（明确记录）**：**不采用**更激进的 “`s==1.0` 恒等 / `s==0.0` relu” 专用分支——`s==1` 已被 `max(x,x)=x` 精确覆盖（`verify_equiv.py` #3 证），`s==0` 的 `−∞` 角点被回退路径正确处理；额外分支无默认/测试用例收益，徒增代码路径（评估结论见 §1.6.1 #4）。
- **源优化意图承接**：源 #3「输入 dtype 域免 cast」意图 → 让位于正确性（R3）；源 #6「极低 op 数」意图 → 默认路径 2 op（较源 3 op 更少）承接。

**重设计项 R3：计算域重设计（输入 dtype 域 → fp32 中转域）** ⭐（本算子关键）

- **源方案**：`zero = T.cast(0, val.dtype)`、`slope = T.cast(negative_slope, val.dtype)`，全程在输入 dtype 域计算；性能意图 = 省一次 fp32 往返 cast + 对齐半精度直接调用语义。
- **NPU 新算法**：
  - **fp32 路径**：直接在 fp32 UB buffer 上跑 R2 向量链（无中转）；
  - **fp16/bf16 路径**：`T.vcast(x_ub, xf, round_mode="rint")` → R2 向量链（fp32 域，slope 为 fp32 常量）→ `T.vcast(xc, y_ub, round_mode="rint")`；
  - **bf16 为 API 强制中转**：`T.vmul` 与 `T.vmax` 的 dtype 矩阵均**不含 bf16**（`docs/Tilelang.language/数学操作/T.vmul.md`·`比较操作/T.vmax.md` §2.2.1），bf16 必须 `vcast` 升 fp32 计算、末端舍回。
- **语义保持论证（含与 golden 的一致性核对——本任务要求项）**：

  | 计算域 | fp32 | fp16（角点） | fp16（randn） | bf16（角点） | bf16（randn） | 与契约 golden |
  |--------|------|-------------|----------------|-------------|----------------|---------------|
  | 契约 golden = 输入 dtype 域？ | — | — | — | — | — | 基线 |
  | **源直译（输入 dtype 域）** | 一致 | `max_abs=4.883e-4`，违反比 **0.2597** | `3.05e-5` | `3.906e-3`，违反比 **0.1297** | `2.44e-4` | 偏差（仍 PASS 但余量小） |
  | **fp32 中转域（采纳）** | 一致 | **`0.0`** | `3.05e-5` | **`0.0`** | `2.44e-4` | **逐位一致（角点/敏感段/尾块 `max_abs=0.0`）** |

  → **fp32 中转域与契约 golden 逐位一致**（`verify_equiv.py` #1 行 `max_abs=0.0`）；输入 dtype 域因 `cast(0.01,dtype)` 的表示误差在角点处偏差达 0.26 违反比。故**采纳 fp32 中转域**：数学上两域都算出同一 LeakyReLU，差异仅在浮点舍入路径；契约 golden 是 fp32，故 fp32 中转是**契约精确实现**。边界（`±∞/NaN/次正规`）经设备探针角点核对一致。
- **源优化意图承接说明**：源 #3 的“免 cast”性能意图在该域下**放弃**（+2 vcast for fp16/bf16）；收益是契约精确（角点余量从 0.26 降到 0）与 bf16 可行性。这是一个**正确性优先于微观 op 数**的取舍，已在 §0.5 记录。

> 承接说明（源优化意图不丢弃）：§0.4 #1「向量化访存」→ R1/R2 以「GM→UB 整块 `T.copy` + 向量 op 整 buffer」承接；#2「大规模并行吃带宽」→ R1 网格跨步并行承接；#3「输入 dtype 域」→ R3 **等价替换为 fp32 中转**（实测更贴近 golden，理由见上）；#4「对齐」→ K11 折叠 `block_size` + §5.3 32B 对齐复算；#5 保留；#6「极低 op 数」→ R2 默认路径 `vmul+vmax`（2 op，较源 3 op 更少）；#7「常量特化」→ R2 编译期 slope 特化 + 工厂 `negative_slope` 常量贯穿。

### 0.7 标杆实现

- 源算子文件（golden 语义依据）：`/home/developer/TileOPs/tileops/kernels/elementwise.py`（`_make_leaky_relu_kernel` L2427–L2478、`LeakyReluFwdKernel` L2481–L2493）。
- 源仓测试基准：`examples/TileOPs/tests/ops/test_activation.py`（`test_leaky_relu` = `randn` 1M × fp32/fp16/bf16 + 4M × fp16/bf16，`ref_program = F.leaky_relu(x.float(), 0.01).to(dtype)`；`test_leaky_relu_edge` = fp32 4096 填 `repeat([-88.8,-1.0,-1e-7,0.0,1e-7,1.0,88.8])`；容差 `_get_tolerances` fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2）。
- manifest：`examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `LeakyReluFwdOp`（L263–L305；workloads `[1048576]` fp32/fp16/bf16、`[16,256,64,64]` fp16/bf16、`[16,512,32,32]` fp16/bf16、`[4096,4096]` fp16/bf16/fp32；roofline flops=`2*N`，bytes=`2*N*elem_bytes`）。
- golden 以 §0.1 语义为依据（`F.leaky_relu(x.float(), negative_slope).to(dtype)`），**不复刻 §0.6 的向量链结构**（保证验证独立性）。
- 同族迁移先例（只读参考）：`examples/elu/_make_elu_kernel/`、`examples/expm1/expm1_fwd_kernel/`、`examples/erf/erf_fwd_kernel/`、`examples/bitwise_not/bitwise_not_fwd_kernel/`。
---

## 1. 概述

### 1.1 算子名称

`_make_leaky_relu_kernel`（项目 `leaky_relu`，目录 `examples/leaky_relu/_make_leaky_relu_kernel/`）。

### 1.2 功能描述

对 1-D 展平张量逐元素计算 `y = x if x > 0 else negative_slope * x`，语义等价于 `torch.nn.functional.leaky_relu`；支持 `float16 | bfloat16 | float32`，输出与输入 shape/dtype 一致。单步纯 Vector 逐元素算子，无规约、无矩阵运算、无跨元素依赖、无超越函数。

**迁移决策后的 NPU 侧算法与源算法的关键差异**：① 计算域由**输入 dtype 域**改为 **fp32 中转域**（匹配契约 golden，bf16 强制）；② 默认 `0<negative_slope<=1` 时计算核用 `max(x, s·x)`（2 op）替代源 where 形式（3 op）；③ 并行/内存结构由 GPU fragment+thread 改为 UB + 一维核 + 网格跨步。计算核语义与源一致（逐元素 LeakyReLU）。

### 1.3 数学公式

$$
y_i = \operatorname{leaky\_relu}(x_i)= \begin{cases} x_i & x_i > 0\\[2pt] s\cdot x_i & x_i \le 0\end{cases},\quad s=\text{negative\_slope}\ (\text{default }0.01),\ i=0,1,\ldots,N-1
$$

NPU 计算核（§1.6.1 优化后，按编译期 `s` 特化；对所有 dtype 在 fp32 计算域构造，末端单次舍回原 dtype）：

$$
y = \begin{cases}
\max(x,\ s\cdot x) & 0 < s \le 1 \quad(\text{默认路径, 2 op})\\[3pt]
\operatorname{vselect}(x>0,\ x,\ s\cdot x) & s \le 0 \text{ or } s > 1 \quad(\text{回退路径, 3 op})
\end{cases}
$$

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法（§0.6），与源算法在结构上等价、在计算域/并行/内存实现上有差异：**

- **计算核心**（§0.6 R2 / §1.6.1 #2）：默认 `0<s<=1` 用 `t = vmul(xc, s)` → `y = vmax(xc, t)`（2 op）；回退用 `vcmp(xc>0)` → `vmul` → `vselect`（3 op）。源为 `T.Parallel` 内标量 `if_then_else(val>0, val, cast(s,dtype)*val)`；NPU 改为整 UB buffer 的无分支向量链。
- **计算域**（§0.6 R3）：fp32 直算；fp16/bf16 先 `T.vcast(rint)` 升 fp32 → 向量链 → `T.vcast(rint)` 舍回。**匹配契约 golden 的 fp32 opmath 语义**（源为输入 dtype 域，实测偏差，见 §0.6 R3）。
- **编译期特化**（§0.6 R2 / §1.6.1）：`negative_slope` 为编译期 Python 常量（源 `lru_cache` 特化同源），按取值选择向量链形态。

除上述外，算法与 §1.6.0 调研选定的算法族（单步逐元素，访存为理论下界）一致：一趟扫描、无中间 GM 往返、无跨元素依赖。

### 1.5 数据流图

```
fp32 路径（xc = x_ub，链尾原地复用为结果）：
GM[x] --T.copy--> UB[x_ub] --[LeakyReLU 向量链 2 op: vmul/vmax]--> UB[x_ub] --T.copy--> GM[y]

fp16/bf16 路径（fp32 中转，xc = xf）：
GM[x(dtype)] --T.copy--> UB[x_ub(dtype)] --T.vcast(rint)--> UB[xf(fp32)]
     --[LeakyReLU 向量链 2 op]--> UB[xf(fp32)] --T.vcast(rint)--> UB[y_ub(dtype)] --T.copy--> GM[y]
```

向量链内部（全部 UB 级、原地复用）：

```
默认路径: t = vmul(xc, s)          （s 为 fp32 编译期常量）
          xc = vmax(xc, t)         （结果落回 xc，正分支自然取原值 x）
回退路径: cond = vcmp(xc, 0, "gt"); t = vmul(xc, s); xc = vselect(cond, xc, t)
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（对同一数学语义的算法族回答调研四问，选定算法族），再在数学等价前提下优化公式，再审视循环/标量计算的向量化替代。迁移任务：1.6.1–1.6.3 的分析对象为 §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 本算子为**单步逐元素类**（§0.3 步骤 4 步、无规约/窗口/矩阵/超越函数），按 `_shared/standards/algorithm-research.md` §3 属轻量调研范畴，但因其存在**多种等价/特化形态**与**计算域分歧**，R1 给出候选表、R3 给出四口径对比、R4 逐候选评估（达到完整调研深度）。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向（op/元素） | 纳入 R3 对比 |
|---|------|------------|-----------|--------------------|-------------|
| 0 | **基线：源算法**（where 形式，输入 dtype 域） | `vselect(x>0, x, s·x)`（3 op） | 迭代定义式；输入 dtype 域与 golden 有浮点偏差（§0.6 R3） | 3 op（fp32）/5 op（fp16·bf16） | ✅（基线） |
| 1 | **max 形式（采纳，默认路径）** | `max(x, s·x)`（2 op） | **`0<s≤1` 时数学恒等 + fp32 域逐位一致**（`max` 退化为 where 两支，§1.6.1 #2） | **2 op（fp32）/4 op（fp16·bf16）** | ✅（采纳） |
| 2 | relu/min-max 形式（torch 语义写作） | `max(x,0) + s·min(x,0)`（4 op） | 数学恒等（任一 `s`） | 4 op（多于 #1）；`s=0` 时 `−∞` 角点与 golden 的 `NaN` 不一致 | ✅（但 op 更多） |
| 3 | `x · select(x>0, 1, s)` | `vmul(x, vselect(cmp,1,s))`（3 op） | 数学恒等（任一 `s`） | 与基线同 op（3 op），无增益 | ✅ |
| 4 | `s==1.0` 恒等 / `s==0.0` relu 专用分支 | 恒等 = 纯 `T.copy`；relu = `vrelu`（1 op） | `s==1` 数学恒等；`s==0` 的 `−∞` 角点 `vrelu(−∞)=0` ≠ golden `NaN`（仅该角点） | 非默认参数、无测试覆盖 | ✅（评估，不采纳，§1.6.1 #4） |
| 5 | **fp32 中转计算域（采纳）** | `cast↑f32 → 链 → cast↓dtype` | **对契约 golden 逐位一致**（基线输入 dtype 域有偏差） | +2 vcast（fp16/bf16）；正确性收益 | ✅（采纳，§1.6.1 #1） |
| 6 | `exp`/分段逼近（同族 elu/erf/expm1 的路线） | — | **不适用**：LeakyReLU 为分段线性，无超越函数可降代 | 无 | ❌（无适用性） |

> **候选 #6 说明**：同族 `elu`/`erf`/`expm1` 任务的核心是「超越函数逼近」，LeakyReLU 是**纯分段线性**函数（无 `exp/log`），该路线在数学上不适用。

**R2 在线算法**：**无在线化收益**——逐元素映射本身天然单遍（无跨元素依赖），不存在可被「单遍/流式」消去的多遍基线。结构判据：本算子无 running 统计量（无归约/无 max/sum 归约、无跨元素依赖），在线变体定义不适用。依据：源算法即单遍直写（§0.3 步骤 1–4 无循环携带依赖）；`examples/elementwise/` 全系逐元素实现（`vec_add_1d.py`、`vec_add_2d.py`）单遍；`ALG-elementwise`（`algorithm-candidates.md`）R2 记「逐元素映射天然单遍」。

**R3 复杂度对比**（四口径；本算子**带宽/发射受限**，访存/遍数是主导项）：

| 算法候选 | FLOPs（v-op 数/元素） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|----------------------|---------------|---------|-------------|---------------------|
| 基线（源 where，输入 dtype 域） | 3（cmp1+mul1+select1） | 2·N·elem_bytes | 1 遍 | O(block_size)（~4 UB buffer） | 可分块给多核，无跨核归并 |
| #1 max 形式（**采纳**，0<s≤1） | **2**（mul1+max1） | 2·N·elem_bytes | 1 遍 | O(block_size)（~4 UB buffer） | 同上 |
| #2 relu/min-max | 4（min1+max1+mul1+add1） | 2·N·elem_bytes | 1 遍 | 同量级 | 同上 |
| #3 `x·select(1,s)` | 3 | 2·N·elem_bytes | 1 遍 | 同量级 | 同上 |
| #4 slope 恒等/relu 特化 | 0（s=1）/1（s=0） | 2·N·elem_bytes | 1 遍 | 同量级 | 同上（非默认参数） |
| **#5 fp32 中转（采纳）** | 与所选链同 +2 vcast（fp16/bf16） | 2·N·elem_bytes | 1 遍 | O(block_size) | 同上 |

> 口径说明：FLOPs 以「每元素 v-op 数」计（无超越函数，故 1 op = 1 单元操作）；fp16/bf16 各候选均 +2 `vcast`（升+降）。访存 = 输入读 + 输出写，**全部候选同为理论下界 `2·N·elem_bytes`**（无中间 GM 往返）；扫描遍数 = GM 读取次数 = 1；中间缓冲 = UB 单 tile 驻留（非 O(N)）。候选差异在 **op 数（发射成本）** 与 **正确性（与契约 golden 的一致性）**。
> **注意**：`#5`（fp32 中转）不改变访存/遍数，只增加 2 个 vcast；其采纳理由是**正确性**（契约 golden 精确匹配 + bf16 可行性），非 op 数收益。`#1`（max 形式）与 `#5` 正交，可叠加（默认路径 = 两者同时生效）。

**R4 硬件亲和性评估**（逐候选对照检查清单；容量/带宽淘汰引用 `constants.md` 条目）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **#1 max + #5 fp32 中转（采纳）** | Vector（`vmul`/`vmax`/`vcast`；`vmax` fp16·fp32 ✓、bf16 ✗；`vcast` f16↔f32 / f32↔bf16 支持 rint） | 小（~4 工作 buffer + 零/cond @bs2048 fp16 ≈ 34 KB，§4.5）✓ | 尾轴 32B 对齐 ✓；v-op 作用于整 buffer | N 编译期常量 ✓ | 单步无多阶段；跨 tile 重叠由 auto-multi-buffer 处理 | ✅ **采纳** |
| 基线（where，输入 dtype 域） | Vector（`vcmp/vselect/vmul` fp16·fp32 ✓） | 同 | ✓ | ✓ | 同 | ⚠️ 可运行但与 golden 有偏差（§0.6 R3） |
| #2 relu/min-max | Vector（`vmin/vmax` fp16·fp32 ✓） | 略多 | ✓ | ✓ | 同 | ❌ op 更多（4 vs 2）+ `s=0` 角点不一致 |
| #3 `x·select(1,s)` | Vector | 同 | ✓ | ✓ | 同 | ❌ 无增益（3 op） |
| #4 slope 恒等/relu | Vector（`vrelu` fp16·fp32 ✓） | 同 | ✓ | ✓ | 同 | ⚠️ 非默认参数，无收益路径 |
| #6 超越函数逼近 | — | — | — | — | — | ❌ 数学不适用（分段线性） |

**v-prefix 元算 dtype 矩阵（决定 bf16 必经 fp32 中转）**：`vmul`/`vmax`/`vrelu` **不含 bf16**（`docs/Tilelang.language/数学操作/T.vmul.md`·`比较操作/T.vmax.md`·`数学操作/T.vrelu.md` §2.2.1：fp16 ✓ / fp32 ✓ / bf16 ✗）；`vcast` 支持 `f16→f32`、`bf16→f32`、`f32→f16/bf16`（均 `rint`）。→ bf16 一律经 fp32 中转。

**调研结论**：**选定「max 形式（`0<s≤1`）+ fp32 中转计算域」**（候选 #1 + #5），回退路径为源 where 形式；无更优可行替代。

关键依据：① R3 全部候选访存同为下界 `2N·elem_bytes`，差异在 op 数与正确性——候选 #1 在默认参数域给出**最少 op（2）**且与契约 golden **逐位一致**；候选 #5 把 fp16/bf16 的计算域对齐契约 golden（逐位一致），bf16 更是 API 强制；② R4 逐候选淘汰：候选 #2（op 更多 + `s=0` 角点不一致）、候选 #3（无增益）、候选 #6（数学不适用）均不采纳；③ 选定算法的全部 op（`vmul/vmax/vcmp/vselect/vcast`）均在**目标设备实测**为真实可用元算（§3.4 探针表）。

**调研范围**（「无更优替代」依据）：① 参考表 `tilelang-op-design/references/algorithm-candidates.md` 的 `ALG-elementwise` 条目（R1 记「公共子表达式消除；除法乘倒数；cast 链合并」；R2 记「逐元素映射天然单遍」；R3 记「融合消除中间 GM 往返」；R4 记「单遍 Vector 流；dtype 路径」+ `TRAP-fp16-opmath-golden`「sub-fp32 逐元素算子 fp32 中转链对齐 torch golden ≤1 ulp」——**该实测直接支持本设计的 fp32 中转决策**）——本仓算法候选库不含 LeakyReLU 专门条目；② 本仓 `examples/` 全系（`elementwise/*`、`elu`、`erf`、`expm1`、`bitwise_not`）：`elu`/`erf`/`expm1` 为**超越函数自建逼近**（LeakyReLU 不适用，纯分段线性）、`mish`/`bitwise_not` 为有原生元算直译形态——LeakyReLU 属「分段线性直译 + 代数重排」形态；③ pattern-library：`kb_search "leaky_relu leaky relu activation elementwise"` 命中 `PL-1.23-vop-chain-inplace-alias`（v-op 链原地复用纪律）、`PL-1.22-persistent-1d-grid-stride`（一维网格跨步）、`CONST-elementwise-op-precision-910_9362`（设备元算精度矩阵）、`ALG-elementwise`；④ **外部已知算法族**（模型知识 + PyTorch/CUDA 生态）：`max(x,0)+s·min(x,0)`（torch 语义写法）、`x·select`、`max(x,s·x)`——均属分段线性代数重排，由 §1.6.1 逐项论证；无超越函数逼近空间；⑤ `docs/Tilelang.language/` 全目录枚举确认**无原生 `vleaky_relu`/`vleakyrelu` 元算**（`数学操作/` 仅有 `vabs/vadd/vcos/vexp/vexp2/vln/vlog2/vmul/vpow/vrelu/vrsqrt/vsigmoid/vsin/vsqrt/vsub/vtanh/...`——`vrelu` 语义为 `max(x,0)` 非 LeakyReLU），故须自建向量链。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)；纯 Vector 逐元素豁免容量项）**：

- 流量项 = `bytes ÷ 有效带宽`，`bytes = 2·N·elem_bytes`（manifest roofline 同口径）。引用 `CONST-mte2-degradation`（`constants.md`，版本戳 tilelang 0.1.2+ed787bb / CANN 8.5.0 / 2026-09-07；fp16 4-set 轮换 1M 33.9 → 16M 30.2 → 64M 20.9 → 256M 18.4 GB/s/core，256M 混合读写地板 ~1.26 TB/s）——**该常数绑定 Ascend910B2C，本设备为 Ascend910_9362，且 `kb_stale_check` 标记 stale，数字仅作量级估算（待重验）**。本设备实查 Vector 核 40（§5.5 ②）。
- 发射项 = `(n_num / core_num) × k × ~0.5µs/op`（`CONST-vector-launch-overhead`，绑定 910B2C 且 stale，**未文档化假设 + 估算依据**）；`k` = 每 tile 向量 op 数：**默认路径 fp32 `k=2`、fp16/bf16 `k=4`（+2 vcast）**；回退路径各 +1。
- 容量项 = 0（纯 Vector，无跨引擎往返；`CONST-store-fixpipe-gm-only` 不适用）。

以 dtype 对应 `block_size`（fp32=1024、fp16/bf16=2048，K11 折叠）、物理 Vector 核 = 40 估算（取默认 `slope=0.01` → max 路径）：

| workload | dtype | N | block_size | n_num | bytes | 流量项 | 发射项 (k) | **估算下界 max** |
|----------|---|------|-----------|-------|-------|---------------------|----------------------------|------------------|
| smoke-1m | fp32 | 1,048,576 | 1024 | 1024 | 8.4 MB | ≈ 6.2 µs | 25.6 µs (k=2) | **≈ 25.6 µs** |
| smoke-1m | fp16 | 1,048,576 | 2048 | 512 | 4.2 MB | 3.1 µs | 25.6 µs (k=4) | **≈ 25.6 µs** |
| gan-feat `[16,256,64,64]` | fp16 | 16,777,216 | 2048 | 8192 | 33.5 MB | ≈ 27 µs | ≈ 410 µs (k=4) | **≈ 410 µs** |
| gan-feat-deep `[16,512,32,32]` | fp16 | 16,777,216 | 2048 | 8192 | 33.5 MB | ≈ 27 µs | ≈ 410 µs (k=4) | **≈ 410 µs** |
| elementwise-16m `[4096,4096]` | fp32 | 16,777,216 | 1024 | 16384 | 134.2 MB | ≈ 107 µs | ≈ 410 µs (k=2) | **≈ 410 µs** |
| elementwise-16m `[4096,4096]` | fp16 | 16,777,216 | 2048 | 8192 | 67.1 MB | ≈ 53 µs | ≈ 410 µs (k=4) | **≈ 410 µs** |

**设计期估算（D-2 roofline）**：代表性 `smoke-1m fp32` 估算下界 ≈ **25.6 µs**（流量 6.2 + 发射 25.6 + 容量 0，**发射项主导**）；全部 workload 由发射项主导（本算子为 **2–4-op 短链**，发射成本是第一驱动；较同族 erf（21-op）/expm1（16-op）低 4–8×，较 elu（4–6-op）低约 2×）。

> **估算不确定性声明（未文档化假设 + 估算依据）**：① `k` 与「每 tile op 链跨 tile 重叠程度」均无本仓实测——`CONST-vector-launch-overhead` 的 0.5µs/op 绑定 910B2C 且 stale，auto-multi-buffer 下向量 op 可与 MTE2 重叠（同源 `PL-1.6-copy-floor`（≥16M 档计算链被 MTE2 窗口完全隐藏，计算链优化 ROI≈0）标记 **stale**，仅作背景，**不作为决策依据**），故上表发射项可能**上偏**；② 对照证据：同族 `bitwise_not`（k=1）实测 256m 1932µs 低于其设计估算 2918µs（实测优于估算 ≈1.5×）；③ `block_size` 是摊薄发射的第一杠杆（发射 ∝ 1/block_size，§5.2）；④ Stage 4 baseline 采集后按 `hardware-cost-model.md` §3 对账并回填常数。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（max 形式 + fp32 中转 + slope 编译期特化）。以下各项完成「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；采纳项的等价性经 `verify_equiv.py` 机器验证（结果表见本节末）。

| # | 优化项 | 原式（基线） | 优化后公式（采纳） | 等价性论证 | 收益量化 |
|---|--------|------|-----------|-----------|---------|
| 1 | **fp32 中转计算域**（采纳） | 源：输入 dtype 域算 `cast(s,dtype)·x` + 输入 dtype 域选择 | `vcast↑f32(rint) → 向量链(fp32, s 为 fp32 常量) → vcast↓dtype(rint)` | 与契约 golden（fp32 域）**逐位一致**（`verify_equiv.py` #1 `max_abs=0.0`）；输入 dtype 域相对 golden 角点违反比 0.2597（fp16）/0.1297（bf16） | 正确性：角点余量 0.26→0；bf16 API 强制；代价 +2 vcast（fp16/bf16） |
| 2 | **max 形式（`0<s≤1`）**（采纳，默认路径） | `vcmp + vmul + vselect`（3 op） | `t=vmul(xc,s); xc=vmax(xc,t)`（2 op） | **`0<s≤1` 数学恒等**：`x>0 → round(s·x)≤x → max=x`；`x≤0 → round(s·x)≥x → max=s·x`（与 golden 负分支同一乘积）；`x=0/±∞/NaN` 逐值一致。`verify_equiv.py` #2（三种 dtype `max_abs=0.0`）+ 设备探针（角点/小 x/尾块 `0.0`） | fp32 3→2 op、fp16/bf16 5→4 op（**−1 op/tile，发射项 −25%~−33%**）；无运行时分支（编译期特化） |
| 3 | **CSE / 缓冲原地复用**（采纳） | 各中间量独立 buffer | 默认路径链尾结果原地落回 `xc`（`vmax(xc,t,xc)`）；`x` 只搬入一次；fp32 路径 `xc=x_ub`（零中转 buffer） | 数学恒等（中间量生命期结束后复用，不改变数值）；链尾原地复用先例 `PL-1.23-vop-chain-inplace-alias` | 省 1 个结果 buffer（§4.5）；`x` 仅 1 次 GM 读 |
| E1 | **`s==1.0` 恒等 / `s==0.0` relu 专用分支（拒绝，非采纳）** | `max(x,s·x)` / where | 恒等 = 纯 `T.copy`；relu = `vrelu` | 恒等数学成立（`verify_equiv.py` E1 行 `max_abs=0.0`）；**但 `s==1` 已被 `max(x,x)=x` 精确覆盖、`s==0` 的 `−∞` 角点 `vrelu(−∞)=0` ≠ golden `NaN`** | 无默认/测试用例收益，且增第三代码路径——**不采纳**，理由记录于此 |
| E2 | **relu/min-max 形式（拒绝）** | where | `max(x,0)+s·min(x,0)` | 数学恒等（任一 `s`）；但 `s=0` 的 `−∞` 角点与 golden 不一致 | op 更多（4 vs 2），无增益 |

**优化结论**：采纳 3 项（#1 fp32 中转计算域、#2 max 形式、#3 CSE/原地复用）。**公式层面已无进一步计算量/访存量可优化空间**——LeakyReLU 的默认路径 `max(x,s·x)` 已是**最短闭式（2 op）**，访存固定在理论下界 `2N·elem_bytes`（1 读 1 写、无中间 GM 往返）；唯一的额外 op 是 fp16/bf16 的 2 次 `vcast`（bf16 为 API 强制，fp16 为契约精确所需）。**优化后公式**（供 §3.1 使用）：
- `float32`：`y = vmax(x, s·x)`（默认 `0<s≤1`，2 op）；回退 `vselect(x>0, x, s·x)`（3 op）
- `float16 / bfloat16`：`y = vcast_dtype(vmax(xf, s·xf))`（+2 vcast）

**等价性机器验证（D-1）**：脚本 `examples/leaky_relu/_make_leaky_relu_kernel/verify_equiv.py`（CPU；候选式 = NPU 向量链逐 op 复现，fp32 域计算后按目标 dtype 单次舍回；基线式 = 源输入 dtype 域 where 形式；契约 golden = `F.leaky_relu(x.float(), s).to(dtype)`）。**执行结果表**（脚本实际输出，`python3 verify_equiv.py`，**ALL 20 ITEMS EQUIV_PASS**）：

| 采纳项 / 对照 | 候选式 | 基线式 | samples | 最大 abs 差 | 违反比 | 结论 |
|---------------|--------|--------|---------|------------|--------|------|
| #1 fp32 中转 vs 源输入 dtype（fp32） | fp32 where 链 | 源输入 dtype 域 | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #1 fp32 中转 vs GOLDEN（fp32） | fp32 where 链 | `F.leaky_relu(x,0.01)` | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #2 max vs where（fp32） | `max(x,s·x)` | `vselect` 链 | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #2 max vs GOLDEN（fp32） | `max(x,s·x)` | `F.leaky_relu(x,0.01)` | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #1 fp32 中转 vs 源输入 dtype（fp16） | fp32 where 链 | 源输入 dtype 域 | 1,048,611 | 4.883e-04 | 0.2597 | **EQUIV_PASS** |
| #1 fp32 中转 vs GOLDEN（fp16） | fp32 where 链 | `F.leaky_relu(x,0.01)` | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #2 max vs where（fp16） | `max(x,s·x)` | `vselect` 链 | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #2 max vs GOLDEN（fp16） | `max(x,s·x)` | `F.leaky_relu(x,0.01)` | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #1 中转无损（fp16） | f16→f32→f16(rint) | `x`（恒等） | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #1 fp32 中转 vs 源输入 dtype（bf16） | fp32 where 链 | 源输入 dtype 域 | 1,048,611 | 3.906e-03 | 0.1297 | **EQUIV_PASS** |
| #1 fp32 中转 vs GOLDEN（bf16） | fp32 where 链 | `F.leaky_relu(x,0.01)` | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #2 max vs where（bf16） | `max(x,s·x)` | `vselect` 链 | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #2 max vs GOLDEN（bf16） | `max(x,s·x)` | `F.leaky_relu(x,0.01)` | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| #1 中转无损（bf16） | bf16→f32→bf16(rint) | `x`（恒等） | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| E1 s==1 恒等 vs where（fp32） | 恒等 copy | `vselect` 链 | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS**（评估行） |
| E1 s==1 恒等 vs GOLDEN（fp32） | 恒等 copy | `F.leaky_relu(x,1.0)` | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS**（评估行） |
| E1 s==1 恒等 vs GOLDEN（fp16） | 恒等 copy | `F.leaky_relu(x,1.0)` | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS**（评估行） |
| #2 max(s=0.5) vs GOLDEN（fp32） | `max(x,0.5x)` | `F.leaky_relu(x,0.5)` | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| 回退 where(s=0.5) vs GOLDEN（fp32） | `vselect` 链 | `F.leaky_relu(x,0.5)` | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |
| 回退 where(s=2.0) vs GOLDEN（fp32） | `vselect` 链 | `F.leaky_relu(x,2.0)` | 1,048,611 | 0.000e+00 | 0.0000 | **EQUIV_PASS** |

> 判定准则 = Stage 3 门禁同口径 `|cand−base| ≤ atol + rtol·|base|`（fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2），违反率 0 即 PASS；覆盖随机张量 `randn`（2²⁰）+ 角点（±0/±inf/NaN/次正规 1e-45/小 x 敏感段 1e-7..1e-3/±1/±88.8/65504）；全部超差计数为 0。

**非门禁诊断（机器输出）——`max` 形式 guard 必要性的证据**（证明 `s>1` 时 `max(x,s·x)` 非等价）：

| negative_slope | `max(x, s·x)` vs GOLDEN max_abs | 违反数 | 结论 |
|----------------|-----------------------------------|--------|------|
| 2.0 | 6.550e+04 | 1,048,591 | **非等价**（guard 外，不采纳） |
| 100.0 | 6.485e+06 | 1,048,602 | **非等价**（guard 外，不采纳） |

→ 故 §1.6.1 #2 的 max 形式**必须**由编译期 `0.0 < negative_slope <= 1.0` 严格守卫；guard 外走 where 回退（上表 #5 s=2.0 行证回退精确）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 二路条件选择 / 负分支选路 | 源标量 `T.if_then_else(val>zero, val, slope*val)` | 默认路径 `T.vmax(xc, t, xc)`（数学等价，免条件）；回退路径 `T.vcmp(xc, z, cond, "gt")` + `T.vselect(cond, xc, t, xc)`（佐证：`docs/.../比较操作/T.vcmp.md` §2.4；`条件操作/T.vselect.md` §2.4；设备探针实跑） | ✅ | — |
| 2 | `×negative_slope` | 源标量 `slope * val`，`slope=T.cast(negative_slope, val.dtype)` | `T.vmul(xc, T.cast(slope,f32), t)`（整 buffer；佐证 `docs/.../数学操作/T.vmul.md` §2.3 标量广播；设备探针 slope 0.5/1.0/0.001 实跑） | ✅ | — |
| 3 | 元素级取大（默认路径） | 源无此点（新增优化） | `T.vmax(xc, t, xc)`（佐证 `docs/.../比较操作/T.vmax.md` §2.4；设备探针 fp32/fp16/bf16 实跑） | ✅ | — |
| 4 | fp32 中转（fp16/bf16） | 源为**输入 dtype 域**（无转换；`T.cast` 仅用于常量） | `T.vcast(src, dst, round_mode="rint")`（升/降）（佐证 `docs/.../数据类型转换操作/T.vcast.md` §2.2.1；设备探针 fp16/bf16 实跑） | ✅ | — |
| 5 | 零常量 | 源标量 `zero = T.cast(0, val.dtype)` | `T.fill(z, 0.0)`（循环外一次；回退路径 `vcmp` 需同 shape tensor 操作数），佐证：`bitwise_not_fwd_kernel.py` `T.fill` 常量 buffer | ✅ | — |
| 6 | 核内网格跨步任务循环 | `for i in T.serial(ceildiv(n_num, core_num))` | 无（核内调度循环，非逐元素计算） | ❌ | tile/block 级任务调度循环；每核迭代数 `ceildiv(n_num,40)`（1M fp32 bs1024 = 26 次，fp16 bs2048 = 13 次），非逐元素热点；元素级并行由向量指令整 buffer 完成。先例 `vec_add_2d.py` L32、`PL-1.22-persistent-1d-grid-stride` |
| 7 | block 起始索引 / 任务映射 | `task=i*core_num+cid`、`off=task*block_size` | 无 | ❌ | block 级索引/任务映射，每 tile 少量标量，非逐元素热点；无逐元素等价 API |
| 8 | 尾块边界（非整除 N） | `tail = T.min(block_size, shape-off)` | 无 | ❌ | 依赖动态 shape 的边界处理；manifest 全部 workload 的 N 整除 `block_size`，设计保留 `shape: T.int32` 守卫以支持任意 N（测试 4,000,000 于 bs2048 余 1216）（纯静态化对通用 N 不成立） |

**向量化结论**：逐元素计算点（#1–#5）**已全部向量化**（`T.vmax/vcmp/vselect/vmul/vcast/fill`，均有 docs 佐证 + 设备探针实测）；保留 3 处标量/循环（#6 核内调度循环、#7 block 索引、#8 动态边界），理由类别：tile 级调度循环 + block 索引 + 动态边界，逐项见上表。与 §6 循环结构一致——kernel 内**无逐元素标量循环**。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

**单轴算子豁免**：本算子为 1-D `[N]` 单轴算子（host 侧把任意 4-D shape 展平为一维），不存在「在哪个轴向量化」的轴选择问题——唯一候选向量化轴即该 1D 元素轴本身。按 skill Phase 2 第 3 项「纯 GEMV/单轴算子可写明『仅一个候选轴』豁免」，记录豁免依据：

- **向量化轴** = 唯一 1D 元素轴（stride=1，连续）。
- **轴质量**：fp32 元素 4B，向量宽 ×4 → `block_size=1024` 每 tile 256 向量槽，尾 lane 浪费 0%（1024 整除 4）；fp16/bf16 元素 2B，向量宽 ×8 → `block_size=2048` 每 tile 256 槽，浪费 0%；均按 32B 尾轴对齐（1024×4=4096B、2048×2=4096B 为 32B 整数倍）。无累加链（无串行累加维，`vmax`/`vmul` 均为纯逐元素）。无 repack（I/O 布局 = 核内布局，一维连续，不需 `T.transpose` 或 host permute）。UB 驻留与 §4.5 一致。
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
| 计算类型 | 纯 Vector（逐元素 LeakyReLU 向量链），无 matmul | 不需 Cube/L0C/L1 |
| 是否含归约 | 无归约 | 不需 Expert 级 buffer 控制 |
| 是否需流水线 | 单数据通路 GM→UB→GM，无核间协作 | 不需 CV 融合 |
| 向量链 API | 全部 v-prefix（`T.vmul/vmax/vcmp/vselect/vcast`） | 用 v-prefix API |
| 同步需求 | UB 内顺序计算，数据依赖线性 | Developer 自动同步即可 |

决策树路径：`纯 element-wise → 单步逐元素 → 无需精细 buffer 控制 → Developer 模式`。与 `op_requirements.programming_mode=developer` 一致；参考 `examples/elementwise/vec_add_1d.py`（Developer + `T.alloc_ub` + v-prefix）与同族迁移 `examples/bitwise_not/bitwise_not_fwd_kernel/`（`T.fill` + v-op + `T.serial` 网格跨步）。

> **模式敏感性提示**：本设计仅用 Developer 实测语义（§3.4 探针）。`T.vrelu` 在 Developer/Expert 下实现不同——本设计**不使用** `vrelu`（slope=0 走 where 回退，见 §1.6.1 #4），规避该差异。

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

> 输入公式为 §1.6.1 的**优化后公式**（LeakyReLU 向量链 + slope 编译期特化 + fp16/bf16 fp32 中转；默认路径 = max 形式）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_tile = x[off : off+bs]` | 搬入一维 tile 到 UB |
| 2 | `xf = cast(x_tile)`（仅 fp16/bf16） | fp32 中转（无损升级） |
| 3a | `t = s · xc` | 负分支缩放（`xc` = 计算域，fp32） |
| 3b | `yf = max(xc, t)`（默认 `0<s≤1`） | 二路取值（正分支自然取 `xc`） |
| 3b' | `cond = (xc > 0)`；`yf = select(cond, xc, t)`（回退） | `s≤0` 或 `s>1` 时的严格二路选择 |
| 4 | `y_tile = cast(yf)`（仅 fp16/bf16） | 单次舍回原 dtype |
| 5 | `y[off : off+bs] = y_tile` | 搬出结果到 GM |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 搬入 | `x_tile = x[...]` | `T.copy` | `T.copy(x[off:off+tail], x_ub[0:tail])` | Developer |
| cast 升 fp32（fp16/bf16） | `xf = cast(x, fp32)` | `T.vcast` | `T.vcast(x_ub, xf, round_mode="rint")` | Developer |
| 负分支缩放 | `t = s·xc` | `T.vmul` | `T.vmul(xc, T.cast(slope, f32), t)` | Developer |
| 二路取值（默认） | `yf = max(xc, t)` | `T.vmax` | `T.vmax(xc, t, xc)`（结果原地落回 `xc`） | Developer |
| 比较掩码（回退） | `cond = (xc>0)` | `T.vcmp` | `T.vcmp(xc, z, cond, "gt")`（`z` 为零常量 buffer） | Developer |
| 选路（回退） | `yf = cond?xc:t` | `T.vselect` | `T.vselect(cond, xc, t, xc)`（复用 `xc`） | Developer |
| cast 舍回（fp16/bf16） | `y_tile = cast(yf,dtype)` | `T.vcast` | `T.vcast(xc, y_ub, round_mode="rint")` | Developer |
| 搬出 | `y[...] = y_tile` | `T.copy` | `T.copy((y_ub if need_cast else x_ub)[0:tail], y[off:off+tail])` | Developer |
| 核内网格跨步 | 任务映射 | `T.Kernel` + `T.serial` | `T.Kernel(core_num, is_npu=True)` + `for i in T.serial(ceildiv(n_num, core_num))` | Developer |
| 零常量 buffer | `z = 0` | `T.fill` | `T.fill(z, T.cast(0.0, f32))`（循环外） | Developer |

### 3.3 计算伪代码

> 迁移规则：`@T.prim_func` 参数名 `x, y` 及顺序不变，新增 `shape: T.int32` 用于动态 N 尾块守卫；**v-prefix op 只作用于整 UB buffer，动态 slice 只包在 `T.copy` 里**（前一任务 R-6 实证陷阱；`PL-1.22`/`PL-1.23`）；**零常量 buffer 在 tile 循环外一次性填充**；**buffer 在 `T.Kernel` 顶层无条件分配**（避免 `TRAP-conditional-alloc`）。

```python
import tilelang
import tilelang.language as T

# LeakyReLU: y = x if x > 0 else negative_slope * x   (negative_slope 编译期常量; 默认 0.01)
@tilelang.jit(out_idx=[1], target="npuir")
def _make_leaky_relu_kernel(N, dtype, negative_slope, block_size=2048):
    f32 = "float32"
    need_cast = (dtype != f32)                 # 编译期 Python 常量（fp16/bf16 -> fp32 中转）
    slope = float(negative_slope)
    use_max = (0.0 < slope <= 1.0)             # 编译期特化（默认 0.01 命中, §1.6.1 #2）
    n_num = T.ceildiv(N, block_size)           # N 编译期常量 -> 静态值
    core_num = 40                              # 物理 Vector 核数（实查 20 aicore x 2）

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), dtype),
        shape: T.int32,
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)   # 输入 tile
            y_ub = T.alloc_ub((block_size,), dtype)   # 输出 tile（fp16/bf16 舍回缓冲）
            xf = T.alloc_ub((block_size,), f32)       # fp32 计算域（fp16/bf16 中转入口）
            t = T.alloc_ub((block_size,), f32)        # 负分支 s*x
            z = T.alloc_ub((block_size,), f32)        # 常量 0（回退路径用，只读）
            cond = T.alloc_ub((block_size,), "bool")  # 回退路径掩码

            # 零常量 buffer：tile 循环外一次性填充（只读，跨 tile 有效）
            T.fill(z, T.cast(0.0, f32))

            # Python 编译期别名：fp32 路径直接以 x_ub 为计算域，fp16/bf16 用 xf
            xc = xf if need_cast else x_ub

            for i in T.serial(T.ceildiv(n_num, core_num)):   # 静态循环上界
                task = i * core_num + cid
                if task < n_num:
                    off = task * block_size
                    tail = T.min(block_size, shape - off)
                    T.copy(x[off:off + tail], x_ub[0:tail])

                    if need_cast:                            # 编译期分支：fp16/bf16
                        T.vcast(x_ub, xf, round_mode="rint")

                    # --- LeakyReLU 向量链（fp32 域，作用于整 buffer）---
                    if use_max:                              # 默认路径（0<s<=1）: 2 op
                        T.vmul(xc, T.cast(slope, f32), t)    # t = s * x
                        T.vmax(xc, t, xc)                    # xc = max(x, s*x)
                    else:                                    # 回退路径（s<=0 或 s>1）: 3 op
                        T.vcmp(xc, z, cond, "gt")            # cond = x > 0
                        T.vmul(xc, T.cast(slope, f32), t)    # t = s * x
                        T.vselect(cond, xc, t, xc)           # xc = x>0 ? x : s*x

                    if need_cast:                            # 编译期分支：单次舍回
                        T.vcast(xc, y_ub, round_mode="rint")
                        T.copy(y_ub[0:tail], y[off:off + tail])
                    else:
                        T.copy(xc[0:tail], y[off:off + tail])

    return main
```

> **伪代码说明**：① `if need_cast:` / `if use_max:` 为 Python 编译期条件（JIT 参数编译期确定），不产生运行时分支；**buffer 在 `with T.Kernel` 顶层无条件分配**（避免 `TRAP-conditional-alloc` 的 `Undefined variable`）；② 零常量 `z` 在 `T.serial` **循环外** `T.fill`（只读、跨 tile 有效，仅回退路径使用）；③ v-op 全部作用于**整 buffer**（R-6 陷阱），仅 `T.copy` 用动态 slice；④ **`xc` 计算域别名**：fp32 路径 `xc=x_ub`（无中转、零额外 buffer 开销），fp16/bf16 路径 `xc=xf`（`vcast` 升级后计算）；⑤ **`xc` 在链尾原地承载结果**（`vmax(xc,t,xc)` / `vselect(...,xc)`，`PL-1.23`）——故无需独立结果 buffer；⑥ `y_ub` 仅用于 fp16/bf16 的单次舍回（fp32 路径直接从 `x_ub` 搬出）。**op 计数 = 2（fp32, 默认）/ 4（fp16·bf16, 默认）/ 3（fp32, 回退）/ 5（fp16·bf16, 回退）**。
>
> **接口适配说明**：迁移 wrapper `leaky_relu.py` 约定 NPU 工厂签名为 `(N, dtype, negative_slope, block_size)`，返回的 `prim_func` 为 `main(x, y, shape)`（`out_idx=[1]`），调用 `_make_leaky_relu_kernel(N, dtype_str, negative_slope, block_size=block_size)(x, N)`（`leaky_relu.py` L114）。本设计按此契约（保留 x/y 名与顺序 + 新增 `shape` 尾守卫）。`is_fp8` 分支为 GPU fp8 专用路径（`torch.float8_*`），NPU manifest 不支持 fp8，**不迁移**（§0.5）。
>
> **注意**：wrapper `leaky_relu.py` 的 docstring 曾主张“输入 dtype 域算术语义必需、必须保留”；本设计以**契约 golden（fp32 域）**为语义权威，用设备探针 + `verify_equiv.py` 实测证明 **fp32 中转域对 golden 逐位一致、输入 dtype 域有偏差**（§0.6 R3），据此选定 fp32 中转域。wrapper 的**接口契约**（签名/`block_size`/`main(x,y,shape)`）被完整遵守，仅其算术域注释被实测依据取代。

### 3.4 API 可行性确认

> **纪律（前车之鉴 TRAP-vrsqrt-plain-precision / CG-2026-0015 / TRAP-verf-maclaurin-truncation）**：NPU 文档声称的元算语义/精度**必须对目标设备实测**。本表全部关键 API 附目标设备（Ascend910_9362）实测（探针脚本 `history_version/design_probe_leaky_relu.py`，2026-10-13 本会话实跑）。

| API | 来源确认（本地） | 设备实测（目标设备） | 状态 |
|-----|-----------------|---------------------|------|
| `T.vmul(src1, src2/dst)`（标量广播） | `docs/Tilelang.language/数学操作/T.vmul.md` §2.3（标量 src2、原地）；dtype fp16 ✓/fp32 ✓/bf16 ✗ | 链端到端 fp32 `max_abs=0.0`；slope 0.5/1.0/0.001 实跑通过 | ✅ 真实可用（bf16 除外） |
| `T.vmax(src0, src1, dst)`（tensor-tensor） | `docs/Tilelang.language/比较操作/T.vmax.md` §2.4（`hivm::VMaxOp`）；dtype fp16 ✓/fp32 ✓/bf16 ✗ | fp32dom-max 链 fp32 `max_abs=0.0`、fp16/bf16 实跑通过 | ✅ 真实可用（bf16 除外） |
| `T.vcmp(A,B,C,mode)` + `T.vselect(Cond,A,B,Out)` | `docs/.../比较操作/T.vcmp.md` §2.4（输出 bool，片上用）；`条件操作/T.vselect.md` §2.4；dtype fp16 ✓/fp32 ✓/bf16 ✓ | 回退链角点实跑逐值一致 | ✅ 真实可用 |
| `T.vcast(src, dst, round_mode)` | `docs/.../数据类型转换操作/T.vcast.md` §2.2.1（`f16→f32` / `f32→f16/bf16` / `bf16→f32` 支持 `rint`） | fp16/bf16 中转链实跑（fp16 randn 3.05e-5 / bf16 randn 2.44e-4；角点 0.0） | ✅ 真实可用 |
| `T.copy`（slice） | `examples/elementwise/vec_add_1d.py` L34 | 探针实跑（含非整除尾块 4e6 fp16） | ✅ 已核对 |
| `T.alloc_ub` / `T.fill` | `vec_add_1d.py` L28；`bitwise_not_fwd_kernel.py` 常量 buffer | 探针实跑 | ✅ 已核对 |
| `T.Kernel(n, is_npu=True)` + `T.serial` | `vec_add_2d.py` L28–L42 | 探针实跑 | ✅ 已核对 |
| 原生 `vleaky_relu`/`npuir_leaky_relu` 元算 | `docs/Tilelang.language/数学操作/` 全目录枚举 + `customize_npuir.py` 无 `npuir_leaky_relu` | — | ❌ **不存在**（须自建 2–5-op 向量链，见 §1.6.0 调研范围 ⑤） |

**链端到端设备实测**（`design_probe_leaky_relu.py`，全部 PASS）：fp32 域 where/max 两形态 fp32 randn `max_abs=0.0`；fp16 randn `3.05e-5`（viol_ratio 0.0296）；bf16 randn `2.44e-4`（0.0148）；fp32/fp16/bf16 角点 `0.0`；fp32 小 x 敏感段 `linspace(-2e-3,2e-3)` `0.0`；fp16 非整除尾块（4,000,000 % 2048 = 1216）`0.0`；fp32 域 max 形式 slope 0.5/1.0/0.001 `0.0`；输入 dtype 域直译（源）fp16 角点 `4.883e-4`（viol_ratio 0.2597，仍 PASS 但余量小——对比依据）。全部远优于门禁容差。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(core_num, is_npu=True)` |
| GPU 专用 API | Yes（源用标量 `T.cast` + `T.if_then_else` + `T.Parallel` + `threads=`） | 换为 v-prefix 向量链 + `is_npu=True`（§0.5/§0.6） |
| GEMM 非整除 | No | 非 GEMM；非整除由 `tail` + `T.ceildiv` 处理（§5.4） |
| L0C 溢出 | No | 无 Cube 计算 |
| 物理核数限制 | Yes | 逻辑核数 = `ceildiv(N, block_size)` 远超物理核 40 → 核内 `T.serial` 网格跨步（§5.5） |
| **bf16 v-prefix 数学 API 缺失** | Yes | `T.vmul/vmax` 不含 bf16 → bf16 走 fp32 中转（§0.6 R3） |
| 无原生 `vleaky_relu` 元算 | Yes | 自建 2–5-op 向量链（§0.6 R2 / §1.6.0） |
| 条件 `T.alloc` 非法 | Yes（编译期分支易误用） | buffer 全部在 `T.Kernel` 顶层无条件分配（§3.3） |
| bool 不可写回 GM | Yes（`vcmp` 输出 bool） | `cond` 仅片上使用（配 `vselect`），从不 `T.copy` 到 GM（`T.vcmp.md` §2.3） |
| 标量操作数需在 `T.Kernel` 内、`T.Scope` 外定义 | Yes | `slope`/常数以 Python 编译期常量传入（`T.vmul.md` §2.3 注 2） |
| `vmax` 无标量广播 | Yes | 默认路径先 `vmul` 生成 tensor `t`，再 `vmax(xc,t,xc)` tensor-tensor（§1.6.1 #2） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU 源） | 本项目（Ascend npuir） | 转换方案 |
|--------|-------------------|----------------------|----------|
| Kernel 装饰器 | `@tilelang.jit(out_idx=[1])` | `@tilelang.jit(out_idx=[1], target="npuir")` | K1 加 `target="npuir"` |
| Kernel 声明 | `T.Kernel(ceildiv(N,bs), threads=256)` | `T.Kernel(core_num, is_npu=True)` + `T.serial` 网格跨步 | K2/K3 去 `threads=`、加 `is_npu=True`；§0.6 R1 |
| 并行模型 | `T.Parallel(threads, npt)` 线程级 | 一维核 + 向量指令整 buffer | K11：`threads×npt` 折叠为 `block_size` |
| **计算域** | **输入 dtype 域**（`slope=T.cast(negative_slope, val.dtype)`） | **fp32 中转域**（`vcast` 升 fp32，slope 为 fp32 常量） | §0.6 R3 等价替换（实测更贴近 golden） |
| **LeakyReLU 计算核** | `if_then_else(v>0, v, s*v)`（3 op） | 默认 `vmax(x, s*x)`（2 op）/ 回退 `vcmp+vselect`（3 op） | §0.6 R2 + §1.6.1 #2（编译期 slope 特化） |
| Buffer 层级 | `T.alloc_fragment`（register） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| prim_func 参数 | `main(x, y)` | `main(x, y, shape: T.int32)` | 不改 x/y 顺序与名称，新增 shape 用于尾块守卫 |
| strategy/autotune | `register_copy` + `threads`/`npt` | 无 strategy 维度、无 autotune | K8/K9 移除 |
| fp8 分支 | `is_fp8` 路径（`float8_*` + fp16 accum） | 无 | 不迁移（NPU manifest 不支持 fp8） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/elementwise/vec_add_1d.py` | 高度相似 | 一维 `T.Kernel`、`T.alloc_ub`、`T.copy` slice、v-prefix 一元 API、`shape` 动态尾块 |
| `examples/elementwise/vec_add_2d.py` | 高度相似 | `T.Kernel(BLOCK_SIZE, is_npu=True)` + `T.serial(ceildiv(...))` + `if block_id < 总任务` 网格跨步（分核先例） |
| `examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py` | 高度相似（迁移形态） | `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步 + `T.fill` 常量 buffer + `shape: T.int32`（结构参照） |
| `examples/elu/_make_elu_kernel/DESIGN.md` | 高度相似（同族迁移） | 迁移文档结构、设备探针方法论、候选表/等价机器验证范式、fp32 中转与 `vcmp/vselect` 选路；**其 fp32 中转结论与本算子一致** |
| `examples/expm1/expm1_fwd_kernel/DESIGN.md`·`examples/erf/erf_fwd_kernel/DESIGN.md` | 相似（同族迁移） | 自建向量链的设备探针方法论、fp32 中转、`vcmp/vselect` 选路（**反面对照**：其超越函数逼近路线对分段线性的 LeakyReLU 不适用） |
| `examples/TileOPs/tileops/kernels/elementwise/leaky_relu/` | 迁移目标 | 集成 wrapper + `block_size`（K11）+ `main(x, y, shape)` 契约 + `negative_slope` 特化 |
| `docs/Tilelang.language/数学操作/T.vmul.md`·`比较操作/T.vmax.md`·`比较操作/T.vcmp.md`·`条件操作/T.vselect.md`·`数据类型转换操作/T.vcast.md` | API | 向量链各 op 规格、dtype 矩阵与用法 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(N,)`（原始 4-D shape 展平） | `float16 \| bfloat16 \| float32` | 1-D 展平输入向量，`N = product(shape)` 编译期常量 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(N,)` | `same_as(x)` | 输出，shape/dtype 与输入一致 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | `(block_size,)` | `dtype` | UB | 输入 tile；**fp32 路径复用为计算域 `xc` 与结果缓冲**（搬出前原地复用） |
| `y_ub` | `(block_size,)` | `dtype` | UB | 输出 tile（**仅 fp16/bf16 路径**承载 `vcast` 单次舍回结果） |
| `xf` | `(block_size,)` | `float32` | UB | fp32 计算域 `xc`（fp16/bf16 = vcast 升级入口，链尾原地复用为结果；fp32 路径空闲） |
| `t` | `(block_size,)` | `float32` | UB | 负分支 `s·x`（`vmul` 结果，供 `vmax`/`vselect` 作第二操作数） |
| `z` | `(block_size,)` | `float32` | UB | 常量 0（**只读**，循环外 `T.fill`，仅回退路径供 `vcmp` 同 shape 操作数） |
| `cond` | `(block_size,)` | `bool` | UB | `x > 0` 掩码（**仅回退路径**；片上，从不写回 GM） |

> **buffer 分配策略**：全部在 `with T.Kernel` 顶层**无条件分配**（避免 `TRAP-conditional-alloc`）；零常量循环外填充；**`xc` 计算域别名**（fp32→`x_ub` / fp16/bf16→`xf`）并在链尾原地承载结果（§3.3，`PL-1.23`）。

### 4.4 内存搬运路径

```
fp32 路径（纯 Vector，无 L1/L0；xc = x_ub）：
GM[x] --T.copy--> UB[x_ub] --[LeakyReLU 链 2 op（vmul/vmax，原地复用 x_ub 承载结果）]--> UB[x_ub] --T.copy--> GM[y]

fp16/bf16 路径（fp32 中转；xc = xf）：
GM[x(dtype)] --T.copy--> UB[x_ub(dtype)] --T.vcast(rint)--> UB[xf(fp32)]
     --[LeakyReLU 链 2 op（原地复用 xf 承载结果）]--> UB[xf(fp32)] --T.vcast(rint)--> UB[y_ub(dtype)] --T.copy--> GM[y]
```

### 4.5 UB 内存预算

K11 折叠 `block_size`：fp32 = 1024，fp16/bf16 = 2048。

| 路径 | x_ub | y_ub | xf | t | z | cond | 合计 | ×1.7（`TRAP-UB-multibuffer-inflation` 最坏膨胀） |
|------|------|------|----|----|---|------|------|---------------------------------------------------|
| fp32 (bs=1024) | 4 KB | 4 KB | 4 KB | 4 KB | 4 KB | 1 KB | **21 KB** | ≈ 36 KB < 192 KB ✓ |
| fp16 (bs=2048) | 4 KB | 4 KB | 8 KB | 8 KB | 8 KB | 2 KB | **34 KB** | ≈ 58 KB ✓ |
| bf16 (bs=2048) | 4 KB | 4 KB | 8 KB | 8 KB | 8 KB | 2 KB | **34 KB** | ≈ 58 KB ✓ |

> UB 硬上限 = 192 KB/AIV（`CONST-capacity-910B2C`，绑定 Ascend910B2C **且 stale**；本设备 Ascend910_9362 同规格量级）。**膨胀系数注（CG-2026-0008）**：显式 alloc 结构实测 ~12% 余量即足（正负两向偏差均无文档，故按最坏上偏 1.7× 保守核对）。**block_size 放大空间**：`bs=4096` → fp16 68 KB（×1.7 ≈ 116 KB，可行）、fp32 84 KB（×1.7 ≈ 143 KB，可行）；`bs=8192` → fp16 136 KB（×1.7 ≈ 231 KB，**超限**）。故 `bs=2048/4096` 列为 Stage 4 发射项调优杠杆（§5.2），默认 `block_size` 按 dtype 取 K11 折叠值。

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
def _make_leaky_relu_kernel(N, dtype, negative_slope, block_size=2048):
    ...
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅含逐元素向量链（`vmul/vmax/vcmp/vselect/vcast`），无 matmul、无归约、无 Cube 计算；仅需 UB 层级，不涉及 L1/L0A/L0B/L0C。

### 5.2 Block 划分

```python
block_size = {"float32": 1024, "float16": 2048, "bfloat16": 2048}[dtype]   # K11 折叠
n_num = T.ceildiv(N, block_size)   # 逻辑 tile 数（编译期静态值）
```

**block_size 选择理由**：
1. 与源 K11 折叠值一致（fp32 = `threads(256)×npt(4)` = 1024、fp16/bf16 = `threads(256)×npt(8)` = 2048），迁移可追溯；v-prefix 整 buffer 形态下 fp32 无 128-bit 对齐约束差异；
2. UB 对齐：block_size×elem_bytes 是 32B 的整数倍（fp32 4096B、fp16/bf16 4096B）✓；
3. UB 容量：最大路径（fp16/bf16 bs=2048）≈ 34 KB，×1.7 最坏膨胀 ≈ 58 KB < 192 KB ✓；
4. **发射项权衡**：本算子 §1.6.0 估算显示**发射项绝对主导**（发射 ∝ 1/block_size），但 bs 放大受 UB 膨胀系数不确定性约束。默认取 K11 折叠值；**bs=2048/4096 列为 Stage 4 调优首选杠杆**（须以编译探针校准 UB 后采纳，分核参数调优，见 `core-split-strategy.md` §2.4）。

### 5.3 约束分析

- **对齐约束**：fp32 bs=1024，元素 4B → 每 tile 256 向量槽（128bit×4），尾 lane 浪费 0%；fp16/bf16 bs=2048，元素 2B → 256 向量槽（128bit×8），浪费 0%；均满足 32B 尾轴对齐 ✓。
- **UB 容量**：最大 34 KB（fp16/bf16 bs=2048），×1.7 最坏膨胀 ≈ 58 KB < 192 KB ✓。
- **L0/L1 容量**：不适用（纯 Vector）。
- **分形限制**：不适用（无 Cube）。

### 5.4 注意事项（非整除处理策略）

**非整除场景预判**：manifest workload 的 N（1,048,576 / 16,777,216）均整除 `block_size`（fp32 1024 与 fp16/bf16 2048 均整除，因 N 为 2 的幂 ≥ 2²⁰）；测试 shape 4,000,000 **不整除 2048**（余 1216）→ 存在尾块。

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
| gan-feat `[16,256,64,64]` | fp16 | 16,777,216 | 2048 | 8,192 |
| gan-feat-deep `[16,512,32,32]` | fp16 | 16,777,216 | 2048 | 8,192 |
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
- **残留负载不均**（显式记录）：以 smoke-1m fp32（bs1024）为例，`tasks_per_core = ceildiv(1024,40) = 26`；网格跨步下前 `1024 - 40×25 = 24` 个核执行 26 次、后 16 个核执行 25 次 → 临界路径 ≈ **+1.56%**（26 / 25.6）。gan-feat（fp16, bs2048）`ceildiv(8192,40)=205`，前 32 核 205 次、后 8 核 204 次 → 临界 **+0.10%**；elementwise-16m fp32 `ceildiv(16384,40)=410`，临界 **+0.10%**。相较「直接 `T.Kernel(n_num)` 由运行时串行调度多波次」，核内 `T.serial` 把核启动固定开销从「每 tile 一次」摊薄为「每核一次」，净收益为正。
- **模式先例**：`examples/elementwise/vec_add_2d.py` L28–L42；同族 `bitwise_not_fwd_kernel.py` L89–L108（`PL-1.22-persistent-1d-grid-stride`）。

> 备选（供 Stage 4 参考，不在本设计实现）：直接 `T.Kernel(n_num)` 由运行时调度（无核内串行）——保留为对称性对照，若实测核内串行反而引入同步开销可回退；本设计按标准要求取核内串行 persistent 形态。
---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block 级 / tile 级调度循环；kernel 内无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级（任务网格） | block 级并行 | `T.Kernel(core_num, is_npu=True)` | 40 个物理 Vector 核各处理一组 tile |
| tile 级（核内） | 串行迭代（网格跨步） | `T.serial(ceildiv(n_num, core_num))` | 每核串行处理 `ceildiv(n_num,40)` 个 tile，摊薄核启动开销（§5.5） |
| 元素级 | 向量化 | `T.vmul/vmax`（默认 2 op）或 `T.vcmp/vselect/vmul`（回退 3 op）+ `vcast` | 整 buffer 一次完成，无 `T.Parallel` 标量循环 |

**T.Persistent 选择说明**：本设计以 `T.serial` 网格跨步实现 persistent 语义（每核固定处理多 tile），与 `T.Persistent` 等价且为同族先例形态（`PL-1.22`）；不额外引入 `T.Persistent`（核数固定 40 已足够）。

### 6.2 循环伪代码

见 §3.3（`with T.Kernel(core_num, ...)` + `for i in T.serial(...)` + `if task < n_num`，循环内为 2-op（默认）/ 3-op（回退）LeakyReLU 向量链）。

### 6.3 流水线优化

**当前设计**: 不使用 `T.Pipelined`。理由：单 tile 内数据依赖线性（copy-in → [vcast↑] → 2–3-op 链 → [vcast↓] → copy-out），无 K 维迭代；跨 tile 迭代（`T.serial`）的 MTE2/VEC 重叠由 Developer 模式 auto-multi-buffer 自动处理。**发射项主导下**，若 Stage 4 需进一步优化（persistent double buffer / 更大 block_size），参照 `example_elementwise_exp2.py` 的 `T.serial` 多 tile 形态；设计阶段保持简洁。

### 6.4 尾块处理

见 §5.4。`N % block_size != 0` 时最后一个 tile 用 `tail = T.min(block_size, shape - off)` 动态裁剪 `T.copy` 量；manifest 全部 workload 整除（§5.4），仅测试 4,000,000 触发尾块（余 1216，bs=2048）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下编译器自动插入同步。核内数据依赖链严格线性（每 tile 内）：

```
T.copy(GM→UB) → [vcast↑] → (vmul → vmax) 或 (vcmp ∥ vmul → vselect) → [vcast↓] → T.copy(UB→GM)
```

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy(GM→UB)` 后 | 自动 | 编译器插 MTE2→V 等待 |
| 各 v-op 之间（2–3 op 链） | 自动 | 线性数据依赖（默认路径 `vmul→vmax` 串行；回退路径 `vcmp` 与 `vmul` 并行分支后汇于 `vselect`），插 V→V barrier |
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

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（`F.leaky_relu(x.float(), negative_slope).to(dtype)`，保持契约 fp32 opmath 语义），**不得复刻 §0.6 的 NPU 向量链结构**——保证验证独立性。

```python
import torch
import torch.nn.functional as F


def golden_leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """LeakyReLU 参考实现，等价于 torch.nn.functional.leaky_relu（契约 fp32 opmath 语义）。

    契约（test_activation.py::LeakyReluTest.ref_program）：
        F.leaky_relu(x.float(), negative_slope).to(dtype)
    即对 fp16/bf16 输入先升 fp32 计算，末端单次舍回原 dtype；fp32 输入直接计算。
    源 GPU 非 fp8 分支实为**输入 dtype 域**算术（slope=cast(s,dtype)），与契约 fp32
    域有小偏差（§0.6 R3）；golden 严格以契约为准，不复刻源或 NPU 结构。
    """
    x_f32 = x.detach().cpu().to(torch.float32)
    return F.leaky_relu(x_f32, negative_slope).to(x.dtype)
    # 与契约 golden 同式；独立于 NPU 向量链实现（不复用其结构）
```

> 备注：golden **独立于** NPU 向量链（直接调用 `F.leaky_relu` 库实现，不复用 §3.3 的 op 序列），满足验证独立性。**关键**：契约 golden 在 fp32 域计算 `0.01·x`，故 NPU 侧**必须**用 fp32 中转域（不得用输入 dtype 域直译，§0.6 R3）。

### 8.2 精度标准

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 1e-5 | 1e-5 | 来源：`examples/TileOPs/tests/ops/test_activation.py` `_get_tolerances` |
| float16 | 1e-3 | 1e-3 | 来源：同上；NPU 在 fp32 域计算、单次舍回 |
| bfloat16 | 1.6e-2 | 1.6e-2 | 来源：同上；bf16 精度低，容忍度放宽 |

**误差预算核对（设计期设备实测，`design_probe_leaky_relu.py`）**：默认路径 fp32 域 `vmul+vmax` 对 golden —— fp32 `max_abs=0.0`、fp16 randn `3.05e-5`（viol_ratio 0.0296）、bf16 randn `2.44e-4`（0.0148）；角点/小 x/尾块 `0.0`。全部 ≫ 优于门禁（fp32 精确、fp16 ≥30× 余量、bf16 ≥60× 余量）。

### 8.3 L0 门槛测试计划

> L0 = 门槛 smoke 测试（最小可用性验证，进 Stage 3 前必过）；完整分层套件 L1/L2/Boundary 交由 `tilelang-op-develop`，不在此枚举。**显式包含负分支小 x、x=0 分支点、正负饱和、各 dtype、非整除尾块、非默认 slope**。

| 用例 | 输入 | 期望 | 判定 |
|------|------|------|------|
| L0-1 fp32 smoke（randn 全域） | `randn([1048576], fp32)`（NPU） | 与 `golden_leaky_relu(x,0.01)` 比对 ≤1e-5 | 门槛通过 |
| L0-2 fp16 smoke（randn 全域） | `randn([1048576], fp16)`（NPU） | 比对 ≤1e-3 | 门槛通过 |
| L0-3 bf16 smoke（randn 全域） | `randn([1048576], bf16)`（NPU） | 比对 ≤1.6e-2 | 门槛通过 |
| L0-4 fp32 角点（分发/边界） | `repeat([-88.8,-1.0,-1e-7,0.0,1e-7,1.0,88.8], 4096, fp32)`（NPU，`test_leaky_relu_edge` 同款） | 与 golden 比对 ≤1e-5（`-88.8→-0.888`、`±1e-7` 各归其支、`0→0`、`88.8→88.8`；**设计期设备实测角点 max_abs 0.0**） | 门槛通过 |
| L0-5 负分支小 x 敏感段 | `linspace(-2e-3, 0, 1<<20, fp32)`（NPU） | 与 golden 比对 ≤1e-5（设计期设备实测 max_abs 0.0） | 门槛通过 |
| L0-6 正负饱和极值 | `repeat([-65504.0, 65504.0, float('-inf'), float('inf'), float('nan')], 4096, fp32)`（NPU） | `-65504→-655.04`、`+inf→+inf`、`-inf→-inf`、`nan→nan`（`equal_nan=True`） | 门槛通过 |
| L0-7 x=0 分支点 + 全域敏感区 | `linspace(-9,9,1<<20, fp32)`（NPU） | 与 golden 比对 ≤1e-5（设计期设备实测 0.0） | 门槛通过 |
| L0-8 非整除尾块 | `randn([4000000], fp16)`（NPU） | 比对 ≤1e-3（覆盖 4000000 % 2048 = 1216 尾块；设计期设备实测 0.0） | 门槛通过 |
| L0-9 各 dtype 边界 shape | `[16,256,64,64]` / `[16,512,32,32]` fp16·bf16（NPU，manifest） | 比对 ≤1e-3 / ≤1.6e-2 | 门槛通过 |
| L0-10 非默认 slope（回退路径） | `randn([1048576], fp32)`，`negative_slope=2.0`（NPU） | 与 `golden_leaky_relu(x,2.0)` 比对 ≤1e-5（覆盖 where 回退路径；`verify_equiv.py` #5 证精确） | 门槛通过 |

- **执行方式**：`tilelang.compile(func, target="npuir")` 后单次调用，`torch.testing.assert_close(y.cpu(), golden, atol, rtol, equal_nan=True)`（golden = §8.1）。
- **门槛标准**：L0-1..L0-10 全过即进入 Stage 3；任一失败不得继续。
- 输入生成参考 `examples/TileOPs/tileops/workloads/elementwise.py`；边界用例参考 `test_activation.py::LeakyReluTest` / `ActivationEdgeFixture`（`repeat([-88.8,-1.0,-1e-7,0.0,1e-7,1.0,88.8])`）。
- **额外设计期落点**：`history_version/design_probe_leaky_relu.py`（设备探针：fp32 域 where/max 两形态端到端 + 输入 dtype 域对照 + 角点/小 x/尾块/slope）、`verify_equiv.py`（D-1 机器验证，20/20 `EQUIV_PASS`）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 缓解方案 |
|------|------|----------|
| **R-1 计算域选择（本算子核心）** | 若误用源「输入 dtype 域」直译，fp16 角点对 golden 违反比达 0.2597、bf16 0.1297（余量小） | 采纳 **fp32 中转域**（对 golden 逐位一致，§0.6 R3 / §1.6.1 #1）；设备探针 + `verify_equiv.py` 双证 |
| **R-2 无原生 `vleaky_relu` 元算** | 须自建 2–5-op 向量链 | 直译 + 代数重排（§0.6 R2）；op 数远低于同族 erf/expm1 |
| **R-3 bf16 v-prefix 数学 API 缺失**（`vmul`/`vmax`/`vrelu` 各 op doc §2.2.1） | bf16 直传编译失败 | bf16 走 fp32 中转（§0.6 R3，且为 API 强制） |
| **R-4 max 形式 guard** | `s>1` 时 `max(x,s·x)` 非等价（诊断 max_abs 6.55e4） | 编译期严格守卫 `0.0 < negative_slope <= 1.0`；guard 外走 where 回退（`verify_equiv.py` #5 证精确） |
| **R-5 发射项主导**（2–4 op/tile，`CONST-vector-launch-overhead`） | §1.6.0 估算示 emission 远超 flow（25.6µs vs 6.2µs @smoke-1m fp32） | 标注「未文档化假设 + 估算依据」（常数绑 910B2C、stale、可能上偏）；block_size 为杠杆（§5.2）；Stage 4 baseline 后按 `hardware-cost-model.md` §3 对账 |
| **R-6 UB 容量** | bs 放大受限 | 默认按 dtype K11 折叠值（最大 34 KB×1.7≈58 KB 安全）；buffer 复用（`xc` 原地、`t` 独立）；bs=2048/4096 须编译探针校准 |
| **R-7 条件 `T.alloc` 非法**（`TRAP-conditional-alloc`） | 编译期 `if` 内 `T.alloc_ub` trace 失败 | buffer 全部在 `T.Kernel` 顶层无条件分配（§3.3）；`z`/`cond` 在默认路径未用但仍分配 |
| **R-8 v-op 整 buffer vs 动态 slice** | 传动态 slice 报错 | v-op 作用于整 buffer，仅 `T.copy` 用 `[0:tail]`（§3.3） |
| **R-9 vcast round_mode 限制**（`T.vcast.md` §2.2.1） | `f16→f32` / `f32→f16/bf16` / `bf16→f32` **仅 `rint`** | 中转升降统一 `round_mode="rint"`（与 golden 单次舍回一致） |
| **R-10 `vcmp` bool 不可写回 GM**（`T.vcmp.md` §2.3） | 误写回编译失败 | `cond` 仅片上配 `vselect`（仅回退路径），从不 `T.copy` 到 GM |
| **R-11 `vmax` 无标量广播** | `vmax(x, scalar)` 不可用 | 先 `vmul(xc,s,t)` 生成 tensor `t`，再 `vmax(xc,t,xc)` tensor-tensor（§1.6.1 #2） |
| **R-12 分核残留不均** | 40 核含非 2 因子 5，2ⁿ 逻辑核无法整除 | §5.5 网格跨步核内串行，残留临界 +1.56%（smoke-1m fp32）/ +0.10%（大 N），已显式记录 |
| **R-13 常数表设备/版本** | `CONST-mte2-degradation`/`CONST-vector-launch-overhead`/`CONST-capacity-910B2C` 绑 910B2C 且 `kb_stale_check` 标记 **stale**；本设备 910_9362 | 带宽/发射/容量数字标注「待重验」，仅作量级估算；不以 stale 条目为决策依据（**本设计的关键正确性结论来自本会话设备探针，不依赖 stale 条目**）；Stage 4 baseline 后对账 |
| **R-14 元算设备语义** | 文档语义可能与设备不一致（`TRAP-vrsqrt-plain-precision`、`TRAP-verf-maclaurin-truncation`） | 本设计关键 API `vmul/vmax/vcmp/vselect/vcast` 全部经**目标设备探针实测**（§3.4）；工具链升级后须重验 |
| **R-15 wrapper 注释与实测冲突** | `leaky_relu.py` docstring 主张「输入 dtype 域算术必需」与本设计实测结论相反 | 以**契约 golden** 为语义权威（§0.6 R3 实测证据）；wrapper 的接口契约（签名/`block_size`/`main(x,y,shape)`）完整遵守，仅算术域注释被实测取代——**列入风险交 Stage 3 核对**，建议同步更新该注释避免误导 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 误用输入 dtype 域直译 | 照抄源非 fp8 分支 | fp16/bf16 角点对 golden 违反比 0.26/0.13 | 用 fp32 中转域（R-1） |
| `s>1` 仍用 max 形式 | 未加编译期 guard | 正分支错误放大（max_abs 6.55e4） | `0.0 < slope <= 1.0` 严格守卫（R-4） |
| `s==0` 用 relu 形式 | 未识别 `−∞` 角点 | `vrelu(−∞)=0` ≠ golden `NaN` | `s==0` 走 where 回退（R-4） |
| bf16 直接传入数学 op | 未做 vcast 上转 | 编译失败 | bf16 先 `vcast` 到 fp32（R-3） |
| vcast 用非 rint 模式升级 | `f16→f32` 用 `round`/`floor` | 编译失败（仅 `rint`） | 升降统一 `round_mode="rint"`（R-9） |
| bool `cond` 写回 GM | `T.copy(cond_ub, ...)` | 编译失败（i1 不可 load/store GM） | `cond` 仅片上（R-10） |
| 条件 alloc | 编译期 `if need_cast:` 内 `T.alloc_ub` | trace 报 `Undefined variable` | buffer 顶层无条件分配（R-7） |
| v-op 传动态 slice | `T.vmax(x_ub[0:tail], ...)` | 可能编译/运行错误 | v-op 用整 buffer（R-8） |
| `vmax` 传标量 | `T.vmax(x_ub, 0.01, ...)` | 编译失败（无标量广播） | 先 `vmul` 生成 tensor（R-11） |
| UB 溢出 | block_size 过大 | 编译失败 | 维持 K11 折叠值（默认）；§4.5 预算 + 编译探针（R-6） |
| 尾块越界 | N 非 bs 整数倍且无 tail 裁剪 | 越界读/写 GM | `tail = T.min(block_size, shape-off)`（§5.4） |
| 零常量 buffer 未填充 | 直接用未初始化的 `z`（回退路径） | `vcmp` 语义错误 | 循环外 `T.fill(z, 0.0)`（§3.3） |
| `vcmp` 操作数 shape 不一致 | 用标量 `0` 代替 tensor | 编译失败（A/B 须同 shape） | `T.fill(z,...)` 生成同 shape 常量 buffer（§1.6.2 #5） |
| 零输入崩溃 | N=0 空张量 | MTE DDR 崩溃（`TRAP-zero-input-crash`） | host 侧空张量守卫；本任务 N ≥ 1e6，不触发 |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| **非整除 N** | Kernel 内 `tail` 动态裁剪（§5.4），无需 host padding |
| **极小 N（N < block_size）** | `n_num=1`，`tail=N`，仅 `task==0` 的核工作，其余核空转 |
| **极大 N（>1.6e7）** | 核内 `T.serial` 网格跨步（§5.5），逻辑核数静态 |
| **`±inf/NaN` 输入** | `+inf → max(+inf,+inf)=+inf`；`-inf → max(-inf,s·(-inf)=-inf)=-inf`（`0<s`）；`NaN → NaN>0 为假 → max(NaN,NaN)=NaN`（§0.6 R2 论证 4；设备探针角点实跑一致） |
| **`x=0` / `-0.0`** | `max(0, s·0)=0` / `max(-0.0, s·(-0.0))=-0.0`；与 golden 同值（±0 视为相等，容差口径） |
| **`x` 为大正数** | 正分支直接取 `x`（`max(x,s·x)=x`）→ 无上溢，`x=88.8 → 88.8`（golden 一致，设备实测） |
| **`x` 为大负数** | `s·x` 有限（`0<s`），`max(s·x, x)=s·x` → `-88.8 → -0.888`（golden 一致） |
| **`negative_slope` 特殊值** | `s==1` → `max(x,x)=x`（恒等）；`s==0` → 回退 where（`x=−∞` 得 `NaN` 与 golden 一致）；`s>1` → 回退 where |
| **混合 dtype** | 按 dtype 编译期特化（每 dtype 一份 kernel），输出 `same_as(input)` |
| **`inplace=True`** | host 侧 `input.copy_(result)`（不在 kernel 内，§0.1） |

---

## 10. 交付清单

| 交付物 | 路径 | 说明 |
|--------|------|------|
| 设计文档 | `examples/leaky_relu/_make_leaky_relu_kernel/DESIGN.md` | 本文件 |
| 等价性机器验证 | `examples/leaky_relu/_make_leaky_relu_kernel/verify_equiv.py` | D-1：20/20 `EQUIV_PASS`（§1.6.1） |
| 设计期设备探针 | `examples/leaky_relu/_make_leaky_relu_kernel/history_version/design_probe_leaky_relu.py` | D-3：fp32 域 where/max 两形态 + 输入 dtype 域对照 + 角点/小 x/尾块/slope（§3.4） |
| NPU kernel（下游 Stage 3 产出，尚未创建） | 算子目录下 `_make_leaky_relu_kernel.py` | 按 §3.3 伪代码 + §5 分核实现 |
| Golden（下游） | 同上（kernel 文件内 `golden_leaky_relu`） | 依 §8.1 |
| 迁移集成 wrapper（已存在，只读参照） | `examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu.py` | K11 `block_size` + `main(x,y,shape)` 契约 |
| 精度用例（已存在，只读参照） | `examples/TileOPs/tests/ops/test_activation.py` | `test_leaky_relu` / `test_leaky_relu_edge` |
| 性能用例（已存在，只读参照） | `examples/TileOPs/benchmarks/ops/bench_elementwise_manifest.py` | manifest-driven bench |

**实现要点速览（Stage 3 输入）**：
1. **计算域 = fp32 中转**（fp16/bf16 先 `vcast(rint)` 升 fp32，链尾 `vcast(rint)` 舍回；fp32 直算）——**不可**用输入 dtype 域直译；
2. **计算核 = 编译期 slope 特化**：默认 `0<slope<=1` 用 `vmul(xc,s,t) → vmax(xc,t,xc)`（2 op）；否则 `vcmp(xc,z,cond,"gt") → vmul → vselect`（3 op）；
3. 结构：一维 `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步 + `tail` 尾块守卫；
4. `block_size` = K11 折叠值（fp32 1024 / fp16·bf16 2048）；buffer 顶层无条件分配；零常量循环外 `T.fill`；v-op 作用于整 buffer；
5. **`max` 形式必须由编译期 `0.0 < negative_slope <= 1.0` 严格守卫**（`s>1` 非等价）。
