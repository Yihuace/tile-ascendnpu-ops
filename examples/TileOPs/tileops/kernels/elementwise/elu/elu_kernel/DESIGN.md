# _make_elu_kernel 算子设计文档（迁移）

> **项目**：`elu` ｜ **算子目录**：`examples/elu/_make_elu_kernel/` ｜ **编程模式**：Developer ｜ **target**：`npuir`
>
> **任务性质**：迁移（migration-harness）。把 GPU TileLang 工厂函数 `_make_elu_kernel`（`y = T.if_then_else(v32 > 0, val, T.Cast(dtype, a*(T.exp(v32) - 1)))`）重设计为 NPU（`target="npuir"`）实现。
>
> **本版核心决策**：**源公式与测试 golden 天然一致，计算核不做算法替代**。源 GPU 的负分支 `alpha*(exp(x)-1)` 与测试 golden `F.elu(x.float(), alpha).to(dtype)` 的负分支**逐值一致**——CPU 与本会话目标设备（`Ascend910_9362`）双证：`F.elu` 负分支用 **`exp(x)-1` 而非 `expm1(x)`**（设备实测 `F.elu(-1e-7) = -1.1920929e-7 = exp(-1e-7)-1`，而 `expm1(-1e-7) = -9.99999e-8`）。故迁移只在**并行/内存结构层**重设计（GPU thread → NPU 一维核 + 核内网格跨步、fragment → UB、标量 `T.cast`/`T.exp` → 向量 `vcast`/`vexp`），计算核**直译**源公式（NPU 有原生真 `T.vexp`，设备实测 rel 1.182e-7）——**明确拒绝**同族 expm1 任务的小 `|x|` Taylor 消抵消方案（该方案会与 golden 的 `exp-1` 产生小 x 偏差：`x=-1e-7` 处相对差 16.1%，见 `verify_equiv.py` 非门禁诊断）。设备端到端探针 + `verify_equiv.py`（8/8 `EQUIV_PASS`）双证。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织：先彻底读懂源算子（0.1–0.4），再做算法调研（Phase R，结论见 §1.6.0），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。

**源算子路径（源真相）**：`/home/developer/TileOPs/tileops/kernels/elementwise.py`

- 具体算子：`EluFwdKernel(ParametricUnaryKernel)`（L2548–L2557，`alpha=1.0` compile-time 特化）+ 工厂 `_make_elu_kernel`（L2497–L2546）：

  ```python
  @functools.lru_cache(maxsize=32)
  def _make_elu_kernel(N, dtype, alpha, output_dtype=None, is_fp8=False,
                       threads=256, npt=8):
      out_dtype = output_dtype or dtype
      block_size = threads * npt
      if is_fp8:                                   # fp8 分支（本迁移不涉及）
          ...
      else:                                        # 非 fp8 分支（本迁移对象）
          @tilelang.jit(out_idx=[1])
          def kernel(threads_arg, npt_arg):
              @T.prim_func
              def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
                  with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                      x_reg = T.alloc_fragment((block_size,), dtype)
                      y_reg = T.alloc_fragment((block_size,), dtype)
                      T.copy(x[bx * block_size:(bx + 1) * block_size], x_reg)
                      for i, j in T.Parallel(threads_arg, npt_arg):
                          val = x_reg[i * npt_arg + j]
                          zero = T.cast(0, "float32")
                          a = T.cast(alpha, "float32")
                          one = T.cast(1.0, "float32")
                          v32 = T.cast(val, "float32")
                          y_reg[i * npt_arg + j] = T.if_then_else(
                              v32 > zero, val, T.Cast(val.dtype, a * (T.exp(v32) - one)),
                          )
                      T.copy(y_reg, y[bx * block_size:(bx + 1) * block_size])
              return main
      return kernel
  ```

- 基类与模板：`ParametricUnaryKernel`（GPU 端共享基类，提供 `_builder_fn`/`_builder_args`/`default_config`/`forward`；NPU 侧不提供该基类，`elu.py` 已内联，E6）；
- 提取产物（脚手架）：`examples/TileOPs/tileops/kernels/elementwise/elu/_elu_fwd_kernels.py`（Pattern B，含 `is_fp8` 与非 fp8 两分支）；集成 wrapper：同目录 `elu.py`（K11：`block_size` = 1024 fp32 / 2048 fp16·bf16；`npub::elu_fwd`）；`ops/elementwise/elu.py`（`EluFwdOp`，`alpha`/`inplace` host 处理 + `_validate_alpha`）。

### 0.1 源算子语义（做什么）

**数学语义**：`y = x if x > 0 else alpha * (exp(x) - 1)`，逐元素（manifest `EluFwdOp.ref_api = torch.nn.functional.elu`，`examples/TileOPs/tileops/manifest/elementwise.yaml` L219–L261）。

**契约语义 = 测试 golden `F.elu(x.float(), alpha).to(dtype)`**（`examples/TileOPs/tests/ops/test_activation.py` L50–L51；`alpha` 默认 1.0）。**关键核对（前序 erf/expm1 血的教训）**：`F.elu` 的负分支究竟用 `exp(x)-1` 还是 `expm1(x)`？——**双证为 `exp(x)-1`**：

| 输入 `x` | `F.elu(x,1.0)`（CPU/设备一致） | `exp(x)-1` | `expm1(x)` |
|---------|------------------------------|-----------|-----------|
| `-1e-7` | `-1.1920929e-7` | **`-1.1920929e-7`** ✓ | `-9.99999e-8` ✗ |
| `-1e-6` | `-1.0132790e-6` | **`-1.0132790e-6`** ✓ | `-9.9999954e-7` ✗ |
| `-1e-3` | `-9.9951029e-4` | **`-9.9951029e-4`** ✓ | `-9.9950016e-4` ✗ |

- CPU 实测（本会话，`torch 2.7.1`）：`F.elu` 负分支 == `exp(x)-1` 逐值一致；
- **目标设备实测**（`Ascend910_9362`，本会话探针 `history_version/design_probe_elu.py` 第 1 项 + 直接 `torch_npu` 调用）：设备 `F.elu` 输出与 `exp(x)-1` 逐值一致，与 CPU 结论**一致**。

→ **源 GPU 的 `alpha*(exp(x)-1)` 与 golden 天然一致**；迁移的语义基线即源公式本身，**不存在 expm1 任务那种「源实现精度不足、需算法替代」的差距**。

**dtype 语义（迁移必须保持的关键点）**：源实现**强制 fp32 中转**——`v32 = T.cast(val, "float32")`、`a = T.cast(alpha, "float32")`，`T.exp` 的输入恒为 `float32`。即：
- 对 fp16/bf16 输入：`x → cast fp32 → exp(fp32) − 1 → ×alpha → if_then_else → 隐式舍回原 dtype`（`y_reg` dtype = 输入 dtype，写回一次舍入）；
- 对 fp32 输入：`cast` 为恒等。
该 `fp32 opmath + 一次舍回` 语义与 golden `F.elu(x.float(), ...)`（fp32 域计算后 `.to(dtype)`）一致，是迁移**不得丢失**的 dtype 语义（NPU 侧以 `T.vcast(rint)` 承接）。

**`alpha` 语义**：compile-time 标量（`EluFwdKernel._builder_args → (alpha,)`；`_make_elu_kernel` 经 `functools.lru_cache` 按 `alpha` 特化）。**host 侧约束**（`ops/elementwise/elu.py::_validate_alpha`，移植自 `_validate_scalar_param_repr` 的 float 分支）：`alpha` 须为 int/float；`NaN` 接受；`±Inf` 与超出 dtype `finfo` 范围的值**拒绝**（float-only 无整数表示分支）。设计据此把 `alpha` 当**编译期 Python 常量**贯穿。

**`inplace` 语义**：host 侧处理，不在 kernel 内——`EluFwdOp.forward` 在 `inplace=True` 时 `input.copy_(result.reshape(input.shape)); return input`，kernel 始终算进新 buffer（`ops/elementwise/elu.py` L178–L184）。迁移保持（kernel 无 inplace 分支，别名由 host `copy_` 完成）。

**规约语义**：无规约（逐元素映射，无跨元素依赖），不存在累加顺序问题。

**边界语义**：`elu(0)=0`（正分支 `0>0` 为假 → 负分支 `alpha*(exp(0)-1)=0`，两支同值 0）；`elu(+∞)=+∞`（`+∞>0` → 取 `x`）；`elu(−∞)=alpha*(exp(−∞)−1)=−alpha`（alpha=1 时 `−1`）；`elu(NaN)=NaN`（`NaN>0` 为假 → `exp(NaN)−1=NaN`）；`x ≳ 88.72`（fp32 `ln(finfo.max)`）时正分支直接取 `x`（有限），负分支仅作用于 `x≤0` 不发生上溢。空维度/非整除 shape 由 host 侧 flatten 成 1-D `[N]`。

**host 侧语义**：`ParametricUnaryKernel`/`EluFwdOp` 把任意 shape 展平为 `N_total = numel`；`forward` 仅调用编译后 kernel（`_elu_fwd_wrapped`），无 host 预处理；`inplace` 后处理如上。

**语义保持基线**：§8.1 golden 以本节语义（`F.elu(x.float(), alpha).to(dtype)`）为唯一依据实现，**不得复刻 §0.6 的 NPU 结构决策**（保证验证独立性；本算子 golden 与源公式同式，故独立实现即 `F.elu` 调用）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `(N,)`（原始 shape 被 host 侧展平；manifest `[1048576]` / `[2048,4096]` / `[2048,8192]` / `[4096,4096]`） | `float16 \| bfloat16 \| float32` | 1-D 展平输入向量，`N = product(input.shape)` |
| `y` | `same_as_input`（输出 shape = 输入 shape，展平后 `(N,)`） | `same_as(input)` | 逐元素 ELU 输出 |

**输出 shape（迁移必需）**：`same_as_input`（manifest `shape_rules: output.shape == input.shape`）。
**layout**：1-D 连续（contiguous flatten），无转置。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（非 fp8 分支，`register_copy` 式 fragment load→compute→store，源 L2533–L2540）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `x_reg ← x[bx*bs:(bx+1)*bs]`（fragment 载入） | GM `x` | fragment `x_reg` | 元素定位/搬入 |
| 2 | `v32 = cast(x_reg[i],'f32')` | `x_reg[i]` | 标量 `v32` | fp32 中转 |
| 3 | `t = a*(T.exp(v32) - 1.0)`（`a=cast(alpha,'f32')`） | `v32` | 标量 `t` | 负分支 `alpha*(exp(x)-1)` |
| 4 | `y_reg[i] = if_then_else(v32 > 0, val, cast(t, dtype))` | `v32`,`val`,`t` | `y_reg[i]` | 二路选择（含舍回原 dtype） |
| 5 | `y[bx*bs:(bx+1)*bs] ← y_reg`（fragment 写回） | `y_reg` | GM `y` | 搬出 |

**数据流与内存访问模式**（源硬件 = GPU）：

```
GM[x] --(fragment load)--> registers[x_reg]
   --(T.Parallel(threads,npt) 逐线程标量：cast→exp→sub→mul→if_then_else→cast)--> registers[y_reg]
   --(fragment store)--> GM[y]
（无显式 SMEM；fragment 双缓冲由编译器管理；block_size = threads × npt）
```

**循环与并行结构**：`T.Kernel(T.ceildiv(N, block_size), threads=256)`——一维 grid（`bx`）+ 256 线程/block 两级并行；每线程 `npt` 元素，`T.Parallel(threads_arg, npt_arg)`。`block_size = 256 × npt`：fp32 `npt=4`→1024，fp16/bf16 `npt=8`→2048。

**host 侧逻辑**：`N_total`、`dtype`、`alpha` 由 Op 层传入；`forward` 直接调用编译后 kernel；flatten 由 Op 层完成（`wrapper elu.py` 的 `_DEFAULT_BLOCK_SIZE` = {fp32:1024, fp16:2048, bf16:2048}；`is_fp8` 分支由 GPU fp8 专用路径触发，本迁移不涉及）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | `register_copy`（fragment load→compute→store） | 向量化访存 | `T.alloc_fragment` + `T.copy` 切片 + `T.Parallel` 标量映射 | GPU 大寄存器堆 + memory coalescing | **硬件强相关**（fragment/register 模型） |
| 2 | block×thread 两级并行（大 grid + 256 线程/block） | 用大规模线程/grid 吃满内存带宽 | `T.Kernel(ceildiv(N,bs), threads=256)` + `T.Parallel` | GPU thread/warp 执行模型 | **硬件强相关**（GPU 线程模型） |
| 3 | **fp32 中转**（`T.cast(val,'float32')` → `T.exp`） | 以 fp32 域计算保住精度（规避半精度 exp） | 半精度输入先升 fp32，末端一次舍回原 dtype | fp32 算术单元 | **语义层**（fp32 opmath 精度语义，可移植）；**API 层不可移植** |
| 4 | strategy-aware `npt`（fp32=4、fp16/bf16=8） | 访存对齐（4×4B / 8×2B） | `_strategy_npt` 启发式 | memory coalescing（对齐） | 实现层（NPU 折叠为 `block_size`） |
| 5 | 无 autotune（elementwise 固定启发式配置） | 带宽受限 + 单步逐元素，搜索 ROI 低 | `default_config` 固定 `threads=256`+`npt` | 无硬件依赖 | 可移植（NPU 侧同样固定单实现） |
| 6 | 单条超越函数指令（`T.exp`）完成逐元素映射 | 低 op 数 | 库函数 `expf` | GPU 超越函数单元 | **算法层，可移植**：NPU 有原生真 `T.vexp`（设备实测 rel 1.182e-7），且源 `exp−1` 正是 golden 负分支——**直译即正确** |

> 识别不出来 ≠ 不存在。上述 6 项为源码中可识别的全部优化/规避手段；后续迁移逐项处置（§0.5）。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 Phase R 调研结论 §1.6.0）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `y = x if x>0 else alpha*(exp(x)-1)`（含 fp32 opmath 精度语义） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 依 §0.1（`F.elu`）；**计算核不做算法替代**（§0.6 R2） |
| 计算算法（步骤 1–5 逐元素映射结构，含二路选择） | 算法 | 无（纯数学顺序、无跨元素依赖） | — | **保留（结构）** | 逐元素映射结构保留；仅并行/内存结构重设计 |
| 优化 #1 `register_copy`（fragment + `T.Parallel` 标量映射） | 优化 | GPU register/fragment 模型 | 无直接等价（NPU 无一维 `threads=`；v-prefix op 作用于整 UB buffer） | **重新设计** | → §0.6 R1：GM→UB→GM + 整 buffer v-prefix 向量链 |
| 优化 #2 block×thread 两级并行 | 优化 | GPU thread/warp 模型 | 无直接等价（NPU 一维 Kernel） | **重新设计** | → §0.6 R1：一维 `T.Kernel(core_num, is_npu=True)` + 核内 `T.serial` 网格跨步 |
| 优化 #3 fp32 中转（标量 `T.cast` 升 fp32） | 优化/语义 | CUDA 半精度 exp 缺失 + fp32 算术 | 有（`T.vcast` 向量升/降，`rint`） | **等价替换** | → §0.6 R3：fp16/bf16 `vcast(rint)` 升 fp32 计算、单次 `vcast(rint)` 舍回（bf16 为 v-prefix 数学 API 缺失下的强制中转） |
| 优化 #6 计算核 `T.exp(v32) − 1`（+`if_then_else`） | 算法 | CUDA `expf` + `T.cast` | **有**：NPU 有真 `T.vexp`（设备 rel 1.182e-7），且 `exp−1` 正是 golden 负分支语义 | **保留（直译）** | → §0.6 R2：`T.vexp(xf)`→`T.vsub(…,1)`→`T.vmul(…,alpha)`→`T.vcmp`+`T.vselect`；**不替代**（区别于同族 expm1 任务） |
| 优化 #4 strategy-aware `npt` 对齐 | 优化 | memory coalescing（对齐） | 有（32B 尾轴对齐 / 向量宽度 128bit） | **等价替换** | K11：`threads × npt` 折叠为 `block_size`（§5.2），对齐约束按 NPU 32B 复算（§5.3） |
| 优化 #5 无 autotune | 优化 | 无 | 有 | **保留** | NPU 侧同固定单实现（无 strategy/autotune） |
| host 侧 flatten / `inplace` copy_（非 kernel 逻辑） | 算法（host） | 无 | — | **保留** | Op 层传入 `N_total`，kernel 只处理 1-D；`inplace` 由 host `input.copy_`（§0.1） |

> **依据**：#3 的 vcast 可用性由 `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（`f16→f32` 支持 `rint`；`f32→f16/bf16` 支持 `rint`）+ 设备探针（§3.4）双证；#6 的「`vexp` 为真 exp 且 `exp−1` 与 golden 一致」由设备探针（本会话 `design_probe_elu.py`：设备 `F.elu` == `exp−1`；链端到端 fp32 max_abs 5.96e-8）+ `CONST-elementwise-op-precision-910_9362`（`T.vexp` rel 1.182e-7、`T.vsub` 精确）三证。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（承接源方案性能意图 + 语义保持论证）。

**重设计项 R1：并行结构重设计（thread 级并行 → 一维 Kernel + 核内网格跨步）**

- **源方案**：`T.Kernel(ceildiv(N, block_size), threads=256)` + `T.Parallel(256, npt)`，每线程 `npt` 元素；性能意图 = 用大规模线程/grid 吃满内存带宽。
- **NPU 新算法**：一维 `T.Kernel(core_num, is_npu=True)`（`core_num` = 实查物理 Vector 核数 **40**，§5.5 ②）+ 核内 `for i in T.serial(T.ceildiv(n_num, core_num))` 网格跨步任务映射（`task = i * core_num + cid`，`if task < n_num` 守卫）；每 tile 经 `T.copy` GM→UB、整 buffer 向量 op、`T.copy` UB→GM。**模式先例**：`examples/elementwise/vec_add_2d.py` L28–L42、同族迁移 `examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py` L89–L108（`PL-1.22-persistent-1d-grid-stride` 两证）。
- **语义保持论证**：任务映射只重排「哪个核处理哪个元素区间」，各元素计算完全独立（无跨元素依赖、无累加顺序变化），数学语义逐元素恒等；边界：`T.serial` 循环上界 `T.ceildiv(n_num, core_num)` 为编译期静态值，守卫 `task < n_num` 覆盖非整除尾块，不产生越界。

**重设计项 R2：计算核直译源公式（`exp−1` 向量链）——本迁移与同族 expm1 任务的分水岭** ⭐

- **源方案**：`y = if_then_else(v32 > 0, val, cast(a*(T.exp(v32) - 1), dtype))`——用原生 exp 后减 1、乘 alpha、二路选择；性能意图 = 单条超越函数指令完成逐元素映射 + fp32 精度。
- **NPU 现状（设备实测）**：NPU **有**原生真 exp 元算（`T.vexp`，设备实测 rel 1.182e-7，`CONST-elementwise-op-precision-910_9362`；本会话探针链端到端复现）。`exp(x)−1` 结构**可直接移植且与 golden 一致**。
- **NPU 新算法（向量链直译，不替代）**：

  $$
  y = \begin{cases} x & x > 0\\ \alpha\cdot(\operatorname{vexp}(x)-1) & x \le 0\end{cases}
  $$

  实现为**单条无分支向量链**（两分支皆算 + `T.vselect` 选路），op 顺序 = §3.3 伪代码（fp32 **5 op**：`vcmp+vexp+vsub+vmul+vselect`；fp16/bf16 **7 op**，+2 `vcast`）。**编译期 `alpha==1.0` 时省去 `vmul`**（×1.0 恒等，§1.6.1 #1）。

- **语义保持论证（逐值等价 + 边界核对）**：
  1. **负分支**：`vselect` 假支路 = `alpha*(vexp(x)−1)`，与源/golden 负分支同式；`T.vsub(·,1)` 精确（设备实测 `max_abs=0.0`），`vexp` 相对误差 1.182e-7，`vmul` 精确——负分支相对误差 ≲ 1.2e-7 ≪ rtol 1e-5（≥80× 余量）。
  2. **正分支**：`vselect` 真支路 = `xc`（原值 `x`，含 fp32 中转对 fp16/bf16 无损）——与源 `v32 > 0 → val` 逐位一致。
  3. **分支点 `x=0`**：`0 > 0` 为假 → 负分支 `alpha*(vexp(0)−1)=alpha*0=0`；正分支在 `x=0` 亦不选中，两支同值 0，无跳变（与 golden `F.elu(0)=0` 一致）。
  4. **极值/边界**：`x=+∞ → +∞>0` → 取 `x=+∞`（golden `+∞` ✓）；`x=−∞ → vexp(−∞)−1=0−1=−1`（alpha=1；golden `−1` ✓）；`x=NaN → NaN>0 为假 → vexp(NaN)−1=NaN`（golden `NaN` ✓）；`x=88.8`（正）→ 取 `x=88.8` 有限（golden `88.8` ✓，无上溢）；`x=−88.8 → exp(−88.8)−1≈−1`（golden `−1` ✓）。**无饱和 clamp 需求**（正分支直接输出 `x`）。
  5. **设备端到端实测**（本会话探针）：fp32 `max_abs=5.96e-8`、fp16 `4.88e-4`、bf16 `3.9e-3`、角点 `0.0`、小 x 敏感段 `5.96e-8`、非整除尾块 fp16 `7.6e-6`、`alpha=0.5` `2.98e-8`——全部远优于门禁（rtol/atol）。
- **拒绝项（明确记录）**：**不采用** expm1 任务的小 `|x|` 分段 Taylor 消抵消方案。理由：本算子 golden `F.elu` 用 `exp−1`，Taylor 修正是相对 `expm1` 的改进，反而**偏离** `exp−1` 契约——`verify_equiv.py` 非门禁诊断实测 `x=-1e-7` 处 expm1-based 候选相对 golden 差 **16.1%**、`x=-1e-3` 差 1.0e-5。故该方案在本算子**非等价**，禁止采纳。
- **源优化意图承接**：源 #3「fp32 精度」意图 → R2（向量链在 fp32 域构造）+ R3（fp32 中转）承接；源 #6「单条超越函数指令」性能意图在 NPU 上以 **1× `vexp` + 4 个算术/选择 op（alpha=1 时）** 落地，与源 op 数同量级。

**重设计项 R3：dtype 路径重设计（标量 `T.cast` → 向量 `T.vcast` fp32 中转链）**

- **源方案**：标量 `T.cast(val, "float32")` 使计算在 fp32 域进行，末端 `T.Cast(val.dtype, ...)` 隐式舍回原 dtype；性能意图 = 以 fp32 精度保住结果。
- **NPU 新算法**：
  - **fp32 路径**：直接在 fp32 UB buffer 上跑 R2 向量链；
  - **fp16/bf16 路径**：`T.vcast(x_ub, xf, round_mode="rint")` → R2 向量链（fp32 域）→ `T.vcast(xc, y_ub, round_mode="rint")`；
  - **bf16 为 API 强制中转**：`T.vexp/vmul/vsub` 的 dtype 矩阵均**不含 bf16**（`docs/Tilelang.language/数学操作/*.md` §2.2.1），故 bf16 必须 `vcast` 升 fp32 计算、末端舍回。
- **语义保持论证**：fp32 域算 ELU 后单次舍回原 dtype ≡ 源 fp32 中转语义；`vcast` 升 fp32 无损、降回 `rint` 单次舍入。**设备实测**（§3.4）：fp16 `max_abs=4.88e-4`（≪ rtol 1e-3）、bf16 `3.9e-3`（≪ rtol 1.6e-2）；`verify_equiv.py` #2/#3/#7/#8 `EQUIV_PASS`。

**重设计项 R4：二路条件选择（源 `if_then_else` → 无分支 `vcmp`+`vselect`）**

- **源方案**：`T.if_then_else(v32 > zero, val, ...)`（GPU 分支）。
- **NPU 新算法**：`T.fill(z, 0.0)` 常量零 buffer（循环外一次性）；`T.vcmp(xc, z, cond, "gt")` 得 `cond = x>0`（bool 片上）；`T.vselect(cond, xc, neg, xc)` 无分支选路（`neg = alpha*(vexp(xc)−1)`）。
- **语义保持论证**：SIMD 屏蔽无语义分支发散，两分支同一元素至多一个被选中；`x=0`/`±∞`/`NaN` 均验证与 golden 一致（R2 论证 3/4）。`cond` 为 bool 仅片上使用（配 `vselect`），从不 `T.copy` 到 GM（`docs/Tilelang.language/比较操作/T.vcmp.md` §2.3：i1 不可 load/store GM）。

> 承接说明（源优化意图不丢弃）：§0.4 #1「向量化访存」→ R2/R4 以「GM→UB 整块 `T.copy` + 向量 op 整 buffer」承接；#2「大规模并行吃带宽」→ R1 网格跨步并行承接；#3「fp32 精度」→ R3 fp32 中转链承接；#4「对齐」→ K11 折叠 `block_size` + §5.3 32B 对齐复算；#5 保留；#6「单超越指令」→ 以 1× `vexp` + 4–6 op 承接（NPU 有原生真 `vexp`，无算法缺口）。

### 0.7 标杆实现

- 源算子文件（golden 语义依据）：`/home/developer/TileOPs/tileops/kernels/elementwise.py`（`EluFwdKernel` L2548–L2557、`_make_elu_kernel` L2497–L2546）。
- 源仓测试基准：`examples/TileOPs/tests/ops/test_activation.py`（`test_elu` = `randn` + `F.elu(x.float(),1.0)`；`test_elu_edge` = `repeat([-88.8,-1.0,-1e-7,0.0,1e-7,1.0,88.8])`；`test_elu_alpha` = alpha 0.5/2.0；容差 `_get_tolerances` fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2）。
- manifest：`examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `EluFwdOp`（L219–L261；workloads `[1048576]` fp32/fp16/bf16、`[2048,4096]` fp16/bf16、`[2048,8192]` fp16/bf16、`[4096,4096]` fp16/bf16/fp32；roofline flops=`4*N`，bytes=`2*N*elem_bytes`）。
- golden 以 §0.1 语义为依据（`F.elu(x.float(), alpha).to(dtype)`），**不复刻 §0.6 的向量链结构**（保证验证独立性）。
- 同族迁移先例：`examples/expm1/expm1_fwd_kernel/`、`examples/erf/erf_fwd_kernel/`、`examples/bitwise_not/bitwise_not_fwd_kernel/`（只读参考）。

---

## 1. 概述

### 1.1 算子名称

`_make_elu_kernel`（项目 `elu`，目录 `examples/elu/_make_elu_kernel/`）。

### 1.2 功能描述

对 1-D 展平张量逐元素计算 `y = x if x > 0 else alpha * (exp(x) − 1)`，语义等价于 `torch.nn.functional.elu`；支持 `float16 | bfloat16 | float32`，输出与输入 shape/dtype 一致。单步纯 Vector 逐元素算子，无规约、无矩阵运算、无跨元素依赖。保持源算子的 **fp32 中转**精度语义（fp16/bf16 → fp32 域计算 → 单次舍回）。

**本版核心变化**：**计算核直译源公式**（源 `exp−1` 正是 golden 负分支语义，NPU 有原生真 `T.vexp`，设备实测 rel 1.182e-7）；仅并行/内存结构按 NPU 重设计（一维核 + 网格跨步、fragment→UB、标量 cast/exp→向量 vcast/vexp）。**明确不采用**同族 expm1 任务的小 `|x|` Taylor 消抵消方案（会偏离 golden 的 `exp−1`：`x=-1e-7` 相对差 16.1%）。

### 1.3 数学公式

$$
y_i = \operatorname{elu}(x_i)= \begin{cases} x_i & x_i > 0\\ \alpha\,(e^{x_i}-1) & x_i \le 0\end{cases},\quad i=0,1,\ldots,N-1
$$

保持源 fp32 中转语义（§0.1）：对所有 dtype，`exp`/算术在 fp32 域计算后单次舍回原 dtype。NPU 计算核为分支无发散向量链（§1.4 / §1.6）：

$$
y = \operatorname{vselect}\big(x>0,\; x,\; \alpha\,(\operatorname{vexp}(x)-1)\big)
$$

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法（§0.6），与源算法在结构上等价、在并行/内存实现上有差异：**

- **计算核心**（§0.6 R2 / §1.6.1）：**直译源 `exp−1`**——`vexp(xc) → vsub(·,1) →（alpha≠1 时）vmul(·,alpha)`，正分支直接取 `xc`，`vcmp + vselect` 无分支选路。源为 `T.Parallel` 内标量 `a*(exp(cast(x,f32))−1)`；NPU 改为整 UB buffer 的 5-op（fp32）/7-op（fp16·bf16）向量链（含 `vexp` 真 exp 元算）。
- **dtype 路径**（§0.6 R3）：fp32 直算；fp16/bf16 先 `T.vcast(rint)` 升 fp32 → 向量链 → `T.vcast(rint)` 舍回。保持源 fp32 opmath 语义。
- **二路选择**（§0.6 R4）：`T.vcmp(xc, 0, "gt")` + `T.vselect` 无分支选路。
- **并行结构**（§0.6 R1）：一维 `T.Kernel(core_num, is_npu=True)` + 核内 `T.serial` 网格跨步。

除上述外，算法与 §1.6.0 调研选定的算法族（单步逐元素直译，访存为理论下界）一致：一趟扫描、无中间 GM 往返、无跨元素依赖。

### 1.5 数据流图

```
fp32 路径（xc = x_ub，链尾原地复用为结果）：
GM[x] --T.copy--> UB[x_ub] --[ELU 向量链 5 op in fp32：vcmp/vexp/vsub/(vmul)/vselect]--> UB[x_ub] --T.copy--> GM[y]

fp16/bf16 路径（fp32 中转，xc = xf）：
GM[x(dtype)] --T.copy--> UB[x_ub(dtype)] --T.vcast(rint)--> UB[xf(fp32)]
     --[ELU 向量链 5 op]--> UB[xf(fp32)] --T.vcast(rint)--> UB[y_ub(dtype)] --T.copy--> GM[y]
```

向量链内部（全部 UB 级、原地复用）：
```
z=0(常量) → vcmp(xc, z, "gt")=cond
  ∥ vexp(xc)=e; vsub(e,1)=e; [vmul(e,alpha)=e]        （负分支）
  → vselect(cond, xc, e)=xc                            （正分支 = xc 原值）
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（对同一数学语义的算法族回答调研四问，选定算法族），再在数学等价前提下优化公式，再做循环/标量向量化替代分析。迁移任务：1.6.1–1.6.3 的分析对象为 §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 本算子为**单步逐元素类**（§0.3 步骤 5 步、无规约/窗口/矩阵），按 `_shared/standards/algorithm-research.md` §3 执行调研，四问逐问回答（R1 因存在多种等价/近似形态，给出候选表）。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线：源算法（GPU，直译到 NPU） | `vselect(x>0, x, alpha*(vexp(x)-1))` | **与契约 golden `F.elu` 逐值一致**（负分支同用 `exp−1`） | 5 op（fp32）/7 op | ✅（基线，NPU 可移植且正确） |
| 1 | expm1-based（同族 expm1 任务方案） | 负分支用 `expm1(x)`（小 \|x\| Taylor） | **非等价**：golden 用 `exp−1`，`x=-1e-7` 相对差 16.1% | 无（且偏差） | ❌（诊断证据见 §1.6.1 / `verify_equiv.py`） |
| 2 | relu/min-max 形式 | `max(x,0) + alpha*(exp(min(x,0))-1)` | 容差内等价（含 `x=0`） | 6 op（多 vmin/vmax） | ✅（但 op 多于基线） |
| 3 | `exp2` 换底 | `alpha*(vexp2(x*log2e)-1)` | 容差内等价（数学恒等） | 与基线同 op（+1 mul），`vexp` 已是真 exp，无增益 | ✅ |
| 4 | `alpha*(exp(x)) - alpha` 重排 | `vmul(e,alpha)` 后 `vsub(t,alpha)` | 数学恒等 | 同 op 数（2 op），无增益 | ❌（同式重排） |
| 5 | 编译期 `alpha==1.0` 折叠 | 省去 `vmul(·,alpha)` | 数学恒等（×1.0） | 5→4 op（fp32）/7→6 op（fp16·bf16） | ✅（**采纳**，§1.6.1 #1） |
| 6 | `vcmp` 比较模式 `ge` 替代 `gt` | `x>=0` 选正分支 | `x=0` 两支同值 0，容差内等价 | 同 op 数 | ❌（无增益，且偏离源严格 `>`） |

> **候选 #1 为何在表内**：它是同族 expm1 任务的核心结论（小 `|x|` Taylor 消抵消）。本算子必须显式核对该方案是否适用——**结论是不适用**（golden 用 `exp−1`，修 expm1 反而引入偏差），这正是本迁移与 expm1 任务的分水岭。

**R2 在线算法**：**无在线化收益**——逐元素映射本身天然单遍（无跨元素依赖，不存在可被「单遍/流式」消去的多遍基线）。结构判据：本算子无 running 统计量（无归约/无 max/无 sum），在线变体定义不适用。依据：源算法即单遍直写；`examples/elementwise/` 全系逐元素实现（`vec_add_1d.py`、`vec_add_2d.py`、`example_elementwise_add.py`）单遍。

**R3 复杂度对比**（四口径；本算子带宽/发射受限，访存/遍数是主导项）：

| 算法候选 | FLOPs（v-op 数/元素） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|----------------------|---------------|---------|-------------|---------------------|
| 基线（源直译 `vexp`+select） | **5**（cmp1+exp1+sub1+mul1+select1） | 2·N·elem_bytes | 1 遍 | O(block_size)（~4 个 fp32 UB buffer，§4.5） | 可分块给多核，无跨核归并 |
| 候选 #2（relu/min-max） | 6（+vmin/vmax 但省 select） | 2·N·elem_bytes | 1 遍 | 同量级 | 同上 |
| 候选 #3（exp2 换底） | 6（+1 mul） | 2·N·elem_bytes | 1 遍 | 同量级 | 同上 |
| 候选 #5（**采纳：alpha 折叠**） | **4**（alpha=1）/ 5（alpha≠1） | 2·N·elem_bytes | 1 遍 | 同量级 | 同上 |
| 候选 #1（expm1 Taylor，**非等价**） | 16 | 2·N·elem_bytes | 1 遍 | ~6 buffer | 同上（但小 x 偏离 golden） |

> 口径说明：FLOPs 以「每元素 v-op 数」计（超越函数 `vexp` 按 1 op 记，估算口径）；fp16/bf16 各候选均 +2 `vcast`（7 op）。访存 = 输入读 + 输出写，**全部候选同为理论下界 `2·N·elem_bytes`**（无中间 GM 往返）；扫描遍数 = GM 读取次数 = 1；中间缓冲 = UB 单 tile 驻留（非 O(N)）。候选差异在 **op 数（发射成本）** 与 **正确性**。

**R4 硬件亲和性评估**（逐候选对照检查清单）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **基线 / 候选 #5（采纳）** | Vector（`T.vexp` 真 exp，设备实测 rel 1.182e-7；`vcmp/vselect/vsub/vmul` 均设备实测可用，§3.4） | 小（~4 fp32 工作 buffer + 零 buffer + cond，@bs2048 fp32 ≈ 42 KB，§4.5）✓ | 尾轴 32B 对齐 ✓；v-op 作用于整 buffer | N 编译期常量 ✓ | 单步无多阶段；跨 tile 重叠由 auto-multi-buffer 处理 | ✅ **采纳** |
| 候选 #2（relu/min-max） | Vector（`vmin/vmax`） | 略多 | ✓ | ✓ | 同 | ⚠️ 可等价但 op 更多（6 vs 4/5），无增益 |
| 候选 #3（exp2 换底） | Vector（`vexp2` + mul） | 同 | ✓ | ✓ | 同 | ⚠️ 等价但 +1 op，`vexp` 已是真 exp，无增益 |
| 候选 #1（expm1 Taylor） | Vector | 略多 | ✓ | ✓ | 同 | ❌ **非等价**（golden 用 `exp−1`；`x=-1e-7` 差 16.1%）——正确性否决 |
| 候选 #6（ge） | Vector | 同 | ✓ | ✓ | 同 | ❌ 无增益，偏离源严格 `>` |

**调研结论**：**选定「基线直译 + 编译期 alpha 折叠」（候选 #0/#5 合流），无更优可行替代。**

关键依据：① R3 全部候选访存同为下界 `2N·elem_bytes`，差异在 op 数——候选 #5 在「正确性（与 golden 逐值一致）」与「op 数（alpha=1 时 4 op，本算子所有默认用例）」间取最优；② R4 逐候选淘汰：候选 #2/#3 等价但 op 更多无增益、候选 #1 **正确性否决**（本算子 golden 用 `exp−1`，Taylor 修 expm1 反而引入 16.1% 偏差）、候选 #6 无增益；③ 选定算法的全部 op 均在**目标设备实测**为真实可用元算（§3.4 探针表）。

**调研范围**（「无更优替代」依据）：① 参考表 `tilelang-op-design/references/algorithm-candidates.md` 的 `ALG-elementwise` 条目（R1 记「公共子表达式消除；除法乘倒数；cast 链合并」；R2 记「逐元素映射天然单遍」）——本仓算法候选库不含 ELU 专门条目；② 本仓 `examples/` 全系（`elementwise/*`、`erf`、`bitwise_not`、`expm1`）中 `expm1` 为分段 Taylor 自建逼近（**本算子不适用**，见候选 #1）、`erf` 由自建 A&S 逼近、`mish`/`bitwise_not` 为有原生元算直译形态——ELU 属「有原生元算直译」形态；③ pattern-library：`kb_search "elu activation elementwise sub-fp32 fp32 transit"` 命中 `PL-1.23-vop-chain-inplace-alias`（v-op 链原地复用纪律）、`PL-1.17-subfp32-fp32-transit`（**stale**，仅背景）、`CONST-elementwise-op-precision-910_9362`（设备元算精度矩阵）；④ **外部已知算法族**（log-sum-exp 稳定化、`2·sinh` 恒等、`exp2` 换底、Cephes 有理）由模型知识补充，属 algorithm-research.md §6 第 5 层「只取算法思路」——对本**连续且已含原生元算**的 ELU 均无 op 数/精度增益（ELU 无 `log`/`sum` 归约，稳定化变形不适用）；⑤ 本地 `docs/Tilelang.language/` 全目录枚举确认**无原生 `velu` 元算**（`数学操作/` 目录仅有 `T.vexp/vln/vlog2/vsqrt/vrsqrt/vpow/vsin/vcos/.../vrelu/vabs/vadd/vmul/vsub/vdiv`，其中 `vrelu` 仅 fp32/fp16 且语义为 `max(x,0)` 非 ELU）——故必须自建向量链，直译即最优可行集。

**v-prefix 元算 dtype 矩阵（决定 bf16 必经 fp32 中转）**：`vexp/vmul/vsub` 均仅 `fp16|fp32`（不含 bf16），`vcmp/vselect` 含 `fp16|fp32|bf16`（`docs/Tilelang.language/数学操作/*.md`·`比较操作/T.vcmp.md`·`条件操作/T.vselect.md` §2.2.1）。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)；纯 Vector 逐元素豁免容量项）**：

- 流量项 = `bytes ÷ 有效带宽`，`bytes = 2·N·elem_bytes`（manifest roofline 同口径）。引用 `CONST-mte2-degradation`（`constants.md`，版本戳 tilelang 0.1.2+ed787bb / CANN 8.5.0 / 2026-09-07；fp16 4-set 轮换 1M 33.9 → 16M 30.2 → 64M 20.9 → 256M 18.4 GB/s/core）：**该常数绑定 Ascend910B2C，本设备为 Ascend910_9362，且 `kb_stale_check` 标记 stale，数字仅作量级估算（待重验）**。本设备实查 Vector 核 40（§5.5 ②）。
- 发射项 = `(n_num / core_num) × k × ~0.5µs/op`（`CONST-vector-launch-overhead`，绑定 910B2C=`fixed dispatch` 且 stale，**未文档化假设 + 估算依据**）；`k` = 每 tile 向量 op 数：**fp32 `k=4`（alpha=1）/ 5（alpha≠1）**，fp16/bf16 `k=6`（alpha=1）/7（alpha≠1）（+2 vcast）。
- 容量项 = 0（纯 Vector，无跨引擎往返；`CONST-store-fixpipe-gm-only` 不适用）。

以 dtype 对应 `block_size`（fp32=1024、fp16/bf16=2048，K11 折叠）、物理 Vector 核 = 40 估算（取 alpha=1 默认路径）：

| workload | dtype | N | block_size | n_num | bytes | 流量项 | 发射项 (k) | **估算下界 max** |
|----------|---|------|-----------|-------|-------|---------------------|----------------------------|------------------|
| smoke-1m | fp32 | 1,048,576 | 1024 | 1024 | 8.4 MB | ≈ 7 µs | 51.2 µs (k=4) | **≈ 51 µs** |
| smoke-1m | fp16 | 1,048,576 | 2048 | 512 | 4.2 MB | 3.5 µs | 38.4 µs (k=6) | **≈ 38 µs** |
| mlp-hidden `[2048,4096]` | fp16 | 8,388,608 | 2048 | 4096 | 33.6 MB | ≈ 28 µs | ≈ 307 µs (k=6) | **≈ 307 µs** |
| mlp-hidden-wide `[2048,8192]` | fp16 | 16,777,216 | 2048 | 8192 | 67.1 MB | ≈ 56 µs | ≈ 614 µs (k=6) | **≈ 614 µs** |
| elementwise-16m `[4096,4096]` | fp32 | 16,777,216 | 1024 | 16384 | 134.2 MB | ≈ 111 µs | ≈ 819 µs (k=4) | **≈ 819 µs** |

**设计期估算（D-2 roofline）**：代表性 `smoke-1m fp32` 估算下界 ≈ **51 µs**（流量 7 + 发射 51.2 + 容量 0，发射项主导）；**全部 workload 由发射项主导**（本算子为 4–6-op 短链，发射成本是第一驱动，但远低于同族 expm1 的 16-op 链）。

> **估算不确定性声明（未文档化假设 + 估算依据）**：① `k` 与「每 tile op 链跨 tile 重叠程度」均无本仓实测——`CONST-vector-launch-overhead` 的 0.5µs/op 绑定 910B2C 且 stale，auto-multi-buffer 下向量 op 可与 MTE2 重叠（同源 `PL-1.6-copy-floor` / `PL-1.17-subfp32-fp32-transit` 均标记 **stale**，仅作背景，**不作为决策依据**），故上表发射项可能**上偏**；② 对照证据：同族 `bitwise_not`（k=1）实测 256m 1932µs 低于其设计估算 2918µs（实测优于估算 ≈1.5×）；③ 本算子发射项为 v0 型 2-op 链的 2–3×，为同族 erf（21-op）/expm1（16-op）的 1/3–1/4；④ `block_size` 是摊薄发射的第一杠杆（发射 ∝ 1/block_size，§5.2）。Stage 4 baseline 采集后按 `hardware-cost-model.md` §3 对账并回填常数。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（源直译 + alpha 折叠 + fp32 中转）。以下各项完成「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；采纳项 #1–#3 的等价性经 `verify_equiv.py` 机器验证（结果表见本节末）。

| # | 优化项 | 原式（基线） | 优化后公式（采纳） | 等价性论证 | 收益量化 |
|---|--------|------|-----------|-----------|---------|
| 1 | **编译期 `alpha==1.0` 折叠**（采纳） | 恒做 `vmul(e, alpha)` | `alpha==1.0` 时 Python 编译期跳过 `vmul`（`if float(alpha) != 1.0: T.vmul(...)`） | 数学恒等（×1.0 为恒等元；golden `F.elu(x,1.0)` 内部亦乘 1.0，逐位一致）；`verify_equiv.py` #4 `EQUIV_PASS`（max_abs 0.0） | 默认路径（alpha=1.0，本算子全部 `test_elu`/`test_elu_edge` 用例）：fp32 5→4 op、fp16/bf16 7→6 op（**−1 op/tile ≥20%**）；无运行时分支（编译期常量特化） |
| 2 | **fp16/bf16 fp32 中转计算域**（采纳） | 源语义：fp32 域算 + 单次舍回 | `vcast↑f32(rint) → 向量链(fp32) → vcast↓dtype(rint)` | fp32 域算后单次舍回 ≡ 源语义；fp16↔fp32 / bf16↔fp32 往返无损；`verify_equiv.py` #2/#3/#7/#8 `EQUIV_PASS` | 精度对齐 golden（设备实测 fp16 max_abs 4.88e-4、bf16 3.9e-3）；代价 +2 vcast；bf16 为 API 强制 |
| 3 | **CSE / 缓冲复用**（采纳） | 各中间量独立 buffer | `xc` 计算域链尾原地承载结果（`vselect` dest = `xc`）；`e` 原地 `vsub`；零常量 buffer 循环外一次性 `T.fill` | 数学恒等（中间量在生命期结束后复用，不改变数值）；前置读取 op（`vcmp`）先于 `vcmp` 后对 `xc` 的覆写（`PL-1.23-vop-chain-inplace-alias`） | 省 1 个结果 buffer（§4.5）；`x` 只搬入一次；零 buffer 不产生 per-tile op |
| — | **expm1 小 \|x\| 分段 Taylor 消抵消（拒绝，非采纳）** | `exp−1`（基线） | 小 `\|x\|` 改用 `expm1`（Taylor） | **非等价**：本算子 golden 用 `exp−1`；`verify_equiv.py` 非门禁诊断实测 `x=-1e-7` expm1-based 候选相对 golden 差 **16.1%**、`x=-1e-6` 差 1.31%、`x=-1e-3` 差 1.0e-5（超/贴 fp32 rtol） | 无（且引入偏差）；同族 expm1 任务的方案**不迁移**，理由已记录于 §0.6 R2 |
| — | **`alpha*(exp(x))−alpha` 重排 / `exp2` 换底（拒绝）** | `alpha*(exp−1)` | 见 R1 候选 #3/#4 | 数学恒等但 op 数不降 | 无增益（同 op 数或 +1 op） |

**优化结论**：采纳 3 项。**公式层面已无进一步计算量/访存量可优化空间**——ELU 的负分支 `exp−1` 已是原生元算最短闭式（1× `vexp` + 1 sub + 1 mul），正分支为恒等取值，唯一可省的是 `alpha==1.0` 的 `vmul`（已采纳）；链内 CSE/原地复用已完成（`xc` 原地、`e` 原地、零 buffer 循环外一次填充）。**优化后公式**（供 §3.1 使用）：
- `float32`：`y = vselect(x>0, x, alpha*(vexp(x)-1))`（alpha=1 时 4 op，否则 5 op）
- `float16 / bfloat16`：`y = vcast(vselect(x>0, x, alpha*(vexp(x)-1)), dtype)`（+2 vcast）

**等价性机器验证（D-1）**：脚本 `examples/elu/_make_elu_kernel/verify_equiv.py`（CPU；候选式 = NPU 向量链逐 op 复现，fp32 域计算后按目标 dtype 单次舍回；基线式 = `F.elu(x.float(), alpha).to(dtype)` golden）。**执行结果表**（脚本实际输出，`python3 verify_equiv.py`，ALL 8 ITEMS EQUIV_PASS）：

| 采纳项 | 候选式 | 基线式 | samples | 最大 abs 差 | 违反率(atol+rtol) | 结论 |
|--------|--------|--------|---------|------------|--------|------|
| #1 链路 fp32（alpha=1） | `sel(x>0, x, exp(x)-1)`(fp32 arith) | `F.elu(x,1.0)` golden | 1,048,611 | 0.000e+00 | 0.000000 | **EQUIV_PASS** |
| #2 链路 fp16 | `round_f16(chain fp32)` | `F.elu(x,1.0)` fp16 golden | 1,048,611 | 0.000e+00 | 0.000000 | **EQUIV_PASS** |
| #3 链路 bf16 | `round_bf16(chain fp32)` | `F.elu(x,1.0)` bf16 golden | 1,048,611 | 0.000e+00 | 0.000000 | **EQUIV_PASS** |
| #1 采纳项：alpha==1.0 折叠（省 vmul） | 折叠链 | `F.elu(x,1.0)` golden | 1,048,611 | 0.000e+00 | 0.000000 | **EQUIV_PASS** |
| #2 通用 alpha=0.5（vmul 路径） | `sel(x>0, x, 0.5*(exp(x)-1))` | `F.elu(x,0.5)` golden | 1,048,611 | 2.980e-08 | 0.000000 | **EQUIV_PASS** |
| #2 通用 alpha=2.0（vmul 路径） | `sel(x>0, x, 2.0*(exp(x)-1))` | `F.elu(x,2.0)` golden | 1,048,611 | 1.192e-07 | 0.000000 | **EQUIV_PASS** |
| #3 中转无损 fp16 | `f16→f32→f16(rint)` | `x`（恒等） | 1,048,611 | 0.000e+00 | 0.000000 | **EQUIV_PASS** |
| #3 中转无损 bf16 | `bf16→f32→bf16(rint)` | `x`（恒等） | 1,048,611 | 0.000e+00 | 0.000000 | **EQUIV_PASS** |

> 说明：① 判定准则 = Stage 3 门禁同口径 `|cand−base| ≤ atol + rtol·|base|`（fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2），违反率 0 即 PASS；② fp32/fp16/bf16 链路与 golden **max_abs 0.0（逐位一致）**，因 CPU `F.elu` 与候选式同用 `exp−1`（设备探针亦证设备 `F.elu` 同用 `exp−1`）；③ 覆盖随机张量 `randn`（2²⁰）+ 角点（±0/±inf/NaN/次正规 1e-45/小 x 敏感段 1e-7..1e-3/边界 ±1/±88.8/65504/测试角点）；④ 全部超差计数为 0。

**非门禁诊断（机器输出）——拒绝 expm1-based 方案的证据**（证明本算子 golden 用 `exp−1`，Taylor 修 expm1 反而偏离）：

| x | `F.elu(x,1.0)`（golden） | expm1-based 候选 | 相对差 |
|---|--------------------------|------------------|--------|
| -1e-7 | -1.19209290e-07 | -9.99999941e-08 | **1.611e-01** |
| -1e-6 | -1.01327896e-06 | -9.99999543e-07 | 1.311e-02 |
| -1e-5 | -1.00135803e-05 | -9.99994973e-06 | 1.361e-03 |
| -1e-4 | -1.00016594e-04 | -9.99949989e-05 | 2.159e-04 |
| -1e-3 | -9.99510288e-04 | -9.99500160e-04 | 1.013e-05 |

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 二路条件选择 `x>0 ? x : neg` | 源标量 `T.if_then_else` | `T.vcmp(xc, z, cond, "gt")` + `T.vselect(cond, xc, neg, xc)`（佐证：`docs/.../比较操作/T.vcmp.md` §2.3；`条件操作/T.vselect.md` §2.3；设备探针"corner"实跑） | ✅ | — |
| 2 | `e^x` | 源标量 `T.exp` | `T.vexp(xc, e)`（佐证：`docs/.../数学操作/T.vexp.md` §3 `hivm.hir.vexp`；`CONST-elementwise-op-precision-910_9362` rel 1.182e-7） | ✅ | — |
| 3 | `−1` | 源标量 `-1.0` | `T.vsub(e, 1.0, e)`（佐证：`docs/.../数学操作/T.vsub.md` §2.3 标量广播；`CONST-...` `vsub` 精确） | ✅ | — |
| 4 | `×alpha` | 源标量 `a * (...)`, `a=T.cast(alpha,'f32')` | `T.vmul(e, alpha, e)`（整 buffer）（佐证：`docs/.../数学操作/T.vmul.md` §2.3 标量广播）；alpha==1.0 编译期折叠 | ✅ | — |
| 5 | fp32 中转（fp16/bf16） | 源标量 `T.cast` | `T.vcast(src, dst, round_mode="rint")`（佐证：`docs/.../数据类型转换操作/T.vcast.md` §2.2.1；设备探针 fp16/bf16 实跑） | ✅ | — |
| 6 | 零常量 | 源标量 `zero = T.cast(0,"f32")` | `T.fill(z, 0.0)`（循环外一次；`vcmp` 需同 shape tensor 操作数），佐证：`bitwise_not_fwd_kernel.py` L104 `T.fill` 常量 buffer | ✅ | — |
| 7 | 核内网格跨步任务循环 | `for i in T.serial(ceildiv(n_num, core_num))` | 无（核内调度循环，非逐元素计算） | ❌ | tile/block 级任务调度循环；每核迭代数 `ceildiv(n_num,40)`（1M@bs2048 = 13 次），非逐元素热点；向量 op 作用于整 buffer，元素级并行由向量指令完成。先例 `vec_add_2d.py` L32、`PL-1.22-persistent-1d-grid-stride` |
| 8 | block 起始索引 / 任务映射 | `task=i*core_num+cid`、`off=task*block_size` | 无 | ❌ | block 级索引/任务映射，每 tile 少量标量，非逐元素热点；无逐元素等价 API |
| 9 | 尾块边界（非整除 N） | `tail = T.min(block_size, shape-off)` | 无 | ❌ | 依赖动态 shape 的边界处理；manifest 全部 workload 的 N 整除 `block_size`，设计保留 `shape: T.int32` 守卫以支持任意 N（纯静态化对通用 N 不成立） |

**向量化结论**：逐元素计算点（#1–#6）**已全部向量化**（`T.vcmp/vselect/vexp/vsub/vmul/vcast/fill`，均有 docs 佐证 + 设备探针实测）；保留 3 处标量/循环（#7 核内调度循环、#8 block 索引、#9 动态边界），理由类别：tile 级调度循环 + block 索引 + 动态边界，逐项见上表。与 §6 循环结构一致——kernel 内**无逐元素标量循环**。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

**单轴算子豁免**：本算子为 1-D `[N]` 单轴算子（host 侧把任意 shape 展平为一维），不存在「在哪个轴向量化」的轴选择问题——唯一候选向量化轴即该 1D 元素轴本身。按 skill Phase 2 第 3 项「纯 GEMV/单轴算子可写明『仅一个候选轴』豁免」，记录豁免依据：

- **向量化轴** = 唯一 1D 元素轴（stride=1，连续）。
- **轴质量**：manifest 对齐 → fp32 元素 4B，向量宽 ×4 → `block_size=2048` 每 tile 512 向量槽，尾 lane 浪费 0%（2048 整除 4）；fp16/bf16 元素 2B，向量宽 ×8 → 256 槽，浪费 0%；均按 32B 尾轴对齐（2048×elem_bytes 为 32B 整数倍）。无累加链（无串行累加维）。无 repack（I/O 布局 = 核内布局，一维连续，不需 `T.transpose` 或 host permute）。UB 驻留与 §4.5 一致。
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
| 计算类型 | 纯 Vector（逐元素 ELU 向量链），无 matmul | 不需 Cube/L0C/L1 |
| 是否含归约 | 无归约 | 不需 Expert 级 buffer 控制 |
| 是否需流水线 | 单数据通路 GM→UB→GM，无核间协作 | 不需 CV 融合 |
| 向量链 API | 全部 v-prefix（`T.vcmp/vselect/vexp/vsub/vmul/vcast`） | 用 v-prefix API |
| 同步需求 | UB 内顺序计算，数据依赖线性 | Developer 自动同步即可 |

决策树路径：`纯 element-wise → 单步逐元素 → 无需精细 buffer 控制 → Developer 模式`。与 `op_requirements.programming_mode=developer` 一致；参考 `examples/elementwise/vec_add_1d.py`（Developer + `T.alloc_ub` + v-prefix）与同族迁移 `examples/bitwise_not/bitwise_not_fwd_kernel/`（`T.fill` + v-op + `T.serial` 网格跨步）。

> **模式敏感性提示**：`T.vtanh`/`T.vexp2`/`T.verf` 等的 Developer/Expert 实现不同（`TRAP-verf-maclaurin-truncation`）。本设计仅用 Developer 实测语义（§1.6.0 R4 / §3.4 探针）。

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

> 输入公式为 §1.6.1 的**优化后公式**（ELU 向量链 + alpha 折叠 + fp16/bf16 fp32 中转）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_tile = x[off : off+bs]` | 搬入一维 tile 到 UB |
| 2a | `xf = cast(x_tile)`（仅 fp16/bf16） | fp32 中转（无损升级） |
| 2b | `cond = (xc > 0)` | 正分支掩码 |
| 2c | `e = vexp(xc)`；`e = e − 1` | 负分支 `exp(x)−1` |
| 2d | `e = e · alpha`（alpha≠1.0） | 负分支缩放的 alpha 因子（alpha=1.0 编译期折叠） |
| 2e | `yf = select(cond, xc, e)` | 二路选路（fp32 域结果） |
| 2f | `y_tile = cast(yf)`（仅 fp16/bf16） | 单次舍回原 dtype |
| 3 | `y[off : off+bs] = y_tile` | 搬出结果到 GM |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 搬入 | `x_tile = x[...]` | `T.copy` | `T.copy(x[off:off+tail], x_ub[0:tail])` | Developer |
| cast 升 fp32（fp16/bf16） | `xf = cast(x, fp32)` | `T.vcast` | `T.vcast(x_ub, xf, round_mode="rint")` | Developer |
| 正分支掩码 | `cond = (xc>0)` | `T.vcmp` | `T.vcmp(xc, z, cond, "gt")`（`z` 为零常量 buffer） | Developer |
| 指数 | `e = vexp(xc)` | `T.vexp` | `T.vexp(xc, e)` | Developer |
| 负分支 | `e = e−1` | `T.vsub` | `T.vsub(e, T.cast(1.0,f32), e)` | Developer |
| alpha 缩放 | `e = e·alpha` | `T.vmul` | `T.vmul(e, T.cast(alpha,f32), e)`（alpha==1.0 时省） | Developer |
| 选路 | `yf = cond?xc:e` | `T.vselect` | `T.vselect(cond, xc, e, xc)`（复用 `xc`） | Developer |
| cast 舍回（fp16/bf16） | `y_tile = cast(yf,dtype)` | `T.vcast` | `T.vcast(xc, y_ub, round_mode="rint")` | Developer |
| 搬出 | `y[...] = y_tile` | `T.copy` | `T.copy((y_ub if need_cast else x_ub)[0:tail], y[off:off+tail])` | Developer |
| 核内网格跨步 | 任务映射 | `T.Kernel` + `T.serial` | `T.Kernel(core_num, is_npu=True)` + `for i in T.serial(ceildiv(n_num, core_num))` | Developer |
| 零常量 buffer | `z = 0` | `T.fill` | `T.fill(z, T.cast(0.0, f32))`（循环外） | Developer |

### 3.3 计算伪代码

> 迁移规则：`@T.prim_func` 参数名 `x, y` 及顺序不变，新增 `shape: T.int32` 用于动态 N 尾块守卫；**v-prefix op 只作用于整 UB buffer，动态 slice 只包在 `T.copy` 里**（前一任务 R-6 实证陷阱；`PL-1.22`/`PL-1.23`）；**零常量 buffer 在 tile 循环外一次性填充**（只读，不产生 per-tile op）。

```python
import tilelang
import tilelang.language as T

# ELU: y = x if x > 0 else alpha*(exp(x)-1)   (alpha 编译期常量; 默认 1.0)
@tilelang.jit(out_idx=[1], target="npuir")
def _make_elu_kernel(N, dtype, alpha, output_dtype=None, block_size=2048):
    out_dtype = output_dtype or dtype
    f32 = "float32"
    need_cast = (dtype != f32)          # 编译期 Python 常量
    need_mul = (float(alpha) != 1.0)    # 编译期 alpha==1.0 折叠（§1.6.1 #1）
    n_num = T.ceildiv(N, block_size)    # N 编译期常量 → 静态值
    core_num = 40                       # 物理 Vector 核数（实查 20 aicore × 2）

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)      # 输入 tile
            y_ub = T.alloc_ub((block_size,), out_dtype)  # 输出 tile（fp16/bf16 舍回缓冲）
            xf = T.alloc_ub((block_size,), f32)          # fp32 计算域（fp16/bf16 中转入口）
            e = T.alloc_ub((block_size,), f32)           # exp(x) -> exp(x)-1 -> *alpha
            z = T.alloc_ub((block_size,), f32)           # 常量 0（只读）
            cond = T.alloc_ub((block_size,), "bool")

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

                    # --- ELU 向量链（fp32 域，作用于整 buffer）---
                    T.vcmp(xc, z, cond, "gt")                # cond = x > 0
                    T.vexp(xc, e)                            # e = exp(x)
                    T.vsub(e, T.cast(1.0, f32), e)           # e = exp(x) - 1
                    if need_mul:                             # 编译期：alpha == 1.0 折叠
                        T.vmul(e, T.cast(alpha, f32), e)     # e = alpha*(exp(x)-1)
                    T.vselect(cond, xc, e, xc)               # xc = x>0 ? x : alpha*(exp-1)

                    if need_cast:                            # 编译期分支：单次舍回
                        T.vcast(xc, y_ub, round_mode="rint")
                        T.copy(y_ub[0:tail], y[off:off + tail])
                    else:
                        T.copy(xc[0:tail], y[off:off + tail])

    return main
```

> **伪代码说明**：① `if need_cast:` / `if need_mul:` 为 Python 编译期条件（JIT 参数编译期确定），不产生运行时分支；**buffer 在 `with T.Kernel` 顶层无条件分配**（避免 `TRAP-conditional-alloc` 的 `Undefined variable`）；② 零常量 `z` 在 `T.serial` **循环外** `T.fill`（只读、跨 tile 有效）；③ v-op 全部作用于**整 buffer**（R-6 陷阱），仅 `T.copy` 用动态 slice；④ **`xc` 计算域别名**：fp32 路径 `xc=x_ub`（无中转、零额外 buffer 开销），fp16/bf16 路径 `xc=xf`（`vcast` 升级后计算）；⑤ **`e` 复用**：`vexp` 结果 → `vsub` → `vmul`（原地）；**`xc` 在链尾原地承载结果**（`vselect` 原地，`PL-1.23`）——故无需独立结果 buffer；⑥ `y_ub` 仅用于 fp16/bf16 的单次舍回（fp32 路径直接从 `x_ub` 搬出）。**op 计数 = 4（fp32, alpha=1）/ 5（fp32, alpha≠1）/ 6（fp16·bf16, alpha=1）/ 7（fp16·bf16, alpha≠1）**。

> **接口适配说明**：迁移 wrapper `elu.py` 约定 NPU 工厂签名为 `(N, dtype, alpha, block_size)`，返回的 `prim_func` 为 `main(x, y, shape)`（`out_idx=[1]`），调用 `_make_elu_kernel(N, dtype_str, alpha, block_size=block_size)(x, N)`（`elu.py` L109）。本设计按此契约（保留 x/y 名与顺序 + 新增 `shape` 尾守卫）。`is_fp8` 分支为 GPU fp8 专用路径（`torch.float8_*`），NPU manifest 不支持 fp8，**不迁移**。

### 3.4 API 可行性确认

> **纪律（前车之鉴 TRAP-vrsqrt-plain-precision / CG-2026-0015 / TRAP-verf-maclaurin-truncation）**：NPU 文档声称的元算语义/精度**必须对目标设备实测**。本表全部关键 API 附目标设备（Ascend910_9362）实测（探针脚本 `history_version/design_probe_elu.py`，2026-10-13 本会话实跑）。

| API | 来源确认（本地） | 设备实测（目标设备） | 状态 |
|-----|-----------------|---------------------|------|
| `T.vexp(src, dst)` | `docs/Tilelang.language/数学操作/T.vexp.md`（§3 `hivm.hir.vexp`）；`customize_npuir.py:295 npuir_exp` | 链端到端 fp32 max_abs 5.96e-8；`CONST-elementwise-op-precision-910_9362` 独立实证 rel 1.182e-7（真 exp） | ✅ 真实可用 |
| `T.vsub(src0, src1, dst)` | `docs/.../数学操作/T.vsub.md`（§2.3 标量 src1；§3） | 链角点 max_abs 0.0；`CONST-...` `vsub(x,1)` 精确 | ✅ 真实可用 |
| `T.vmul(src1, src2, dst)`（标量广播） | `docs/.../数学操作/T.vmul.md` §2.3（标量 src2） | `alpha=0.5` 链 max_abs 2.98e-8 实跑通过 | ✅ 真实可用 |
| `T.vcmp(A,B,C,mode)` + `T.vselect(Cond,A,B,Out)` | `docs/.../比较操作/T.vcmp.md` §2.3（输出 bool，片上用）；`条件操作/T.vselect.md` §2.3；`customize_npuir.py:481/501` | 链角点（含 x=0/±inf/nan）实跑逐值一致 | ✅ 真实可用 |
| `T.vcast(src, dst, round_mode)` | `docs/.../数据类型转换操作/T.vcast.md` §2.2.1（`f16→f32` / `f32→f16/bf16` 支持 `rint`） | fp16/bf16 中转链实跑（fp16 4.88e-4 / bf16 3.9e-3） | ✅ 真实可用 |
| `T.copy`（slice） | `examples/elementwise/vec_add_1d.py` L34 | 探针实跑（含非整除尾块 fp16 7.6e-6） | ✅ 已核对 |
| `T.alloc_ub` / `T.fill` | `vec_add_1d.py` L28；`bitwise_not_fwd_kernel.py` L104 | 探针实跑 | ✅ 已核对 |
| `T.Kernel(n, is_npu=True)` + `T.serial` | `vec_add_2d.py` L28–L42 | 探针实跑 | ✅ 已核对 |
| 原生 `velu` / `npuir_elu` 元算 | `docs/Tilelang.language/数学操作/` 全目录枚举 + `customize_npuir.py` 无 `npuir_elu` | — | ❌ **不存在**（须自建向量链，见 §1.6.0 调研范围 ⑤） |

**链端到端设备实测**（`design_probe_elu.py`，全部 PASS）：fp32 randn 1M `max_abs=5.96e-8`；fp16 1M `4.88e-4`；bf16 1M `3.9e-3`；fp32 角点 `0.0`；fp32 小 x 敏感段 `linspace(-1e-3,1e-3)` `5.96e-8`；fp32 `linspace(-9,9)` `5.96e-8`；fp16 非整除尾块（4,000,000 % 2048 = 1216）`7.6e-6`；fp32 `alpha=0.5` `2.98e-8`。全部远优于门禁容差。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(core_num, is_npu=True)` |
| GPU 专用 API | Yes（源用标量 `T.exp` + `T.cast` + `T.if_then_else` + `T.Parallel` + `threads=`） | 换为 v-prefix 向量链 + `is_npu=True`（§0.5/§0.6） |
| GEMM 非整除 | No | 非 GEMM；非整除由 `tail` + `T.ceildiv` 处理（§5.4） |
| L0C 溢出 | No | 无 Cube 计算 |
| 物理核数限制 | Yes | 逻辑核数 = `ceildiv(N, block_size)` 远超物理核 40 → 核内 `T.serial` 网格跨步（§5.5） |
| **bf16 v-prefix 数学 API 缺失** | Yes | `T.vexp/vadd/vmul/vsub` 不含 bf16 → bf16 走 fp32 中转（§0.6 R3） |
| 无原生 `velu` 元算 | Yes | 自建 4–7-op 向量链（§0.6 R2 / §1.6.0） |
| 条件 `T.alloc` 非法 | Yes（编译期分支易误用） | buffer 全部在 `T.Kernel` 顶层无条件分配（§3.3） |
| bool 不可写回 GM | Yes（`vcmp` 输出 bool） | `cond` 仅片上使用（配 `vselect`），从不 `T.copy` 到 GM（`T.vcmp.md` §2.3） |
| 标量操作数需在 `T.Kernel` 内、`T.Scope` 外定义 | Yes | `alpha`/常数以 Python 编译期常量传入（`T.vmul.md` §2.3 注 2） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU 源） | 本项目（Ascend npuir） | 转换方案 |
|--------|-------------------|----------------------|----------|
| Kernel 装饰器 | `@tilelang.jit(out_idx=[1])` | `@tilelang.jit(out_idx=[1], target="npuir")` | K1 加 `target="npuir"` |
| Kernel 声明 | `T.Kernel(ceildiv(N,bs), threads=256)` | `T.Kernel(core_num, is_npu=True)` + `T.serial` 网格跨步 | K2/K3 去 `threads=`、加 `is_npu=True`；§0.6 R1 |
| 并行模型 | `T.Parallel(threads, npt)` 线程级 | 一维核 + 向量指令整 buffer | K11：`threads×npt` 折叠为 `block_size` |
| **ELU 计算核** | `if_then_else(v32>0, val, cast(a*(exp(v32)-1)))` | **同式向量链**（`vcmp/vselect/vexp/vsub/vmul`） | §0.6 R2 直译（不替代，与 expm1 任务不同） |
| 计算精度 | `T.cast` 标量升 fp32 | `T.vcast` 向量升/降 fp32 | 保持 fp32 opmath 语义（§0.1） |
| Buffer 层级 | `T.alloc_fragment`（register） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| prim_func 参数 | `main(x, y)` | `main(x, y, shape: T.int32)` | 不改 x/y 顺序与名称，新增 shape 用于尾块守卫 |
| strategy/autotune | `register_copy` + `threads`/`npt` | 无 strategy 维度、无 autotune | K8/K9 移除 |
| fp8 分支 | `is_fp8` 路径（`float8_*`） | 无 | 不迁移（NPU manifest 不支持 fp8） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/elementwise/vec_add_1d.py` | 高度相似 | 一维 `T.Kernel`、`T.alloc_ub`、`T.copy` slice、v-prefix 一元 API、`shape` 动态尾块 |
| `examples/elementwise/vec_add_2d.py` | 高度相似 | `T.Kernel(BLOCK_SIZE, is_npu=True)` + `T.serial(ceildiv(...))` + `if block_id < 总任务` 网格跨步（分核先例） |
| `examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py` | 高度相似（迁移形态） | `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步 + `T.fill` 常量 buffer + `shape: T.int32`（结构参照） |
| `examples/expm1/expm1_fwd_kernel/DESIGN.md` | 高度相似（同族迁移） | **反面对照**：其分段 Taylor 消抵消方案**不适用于 ELU**（golden 用 `exp−1`）；迁移文档结构、设备探针方法论、候选表/等价机器验证范式 |
| `examples/erf/erf_fwd_kernel/erf_fwd_kernel.py` | 相似（同族迁移） | 自建逼近链的设备探针方法论、fp32 中转、`vcmp/vselect` 选路 |
| `examples/TileOPs/tileops/kernels/elementwise/elu/` | 迁移目标 | 集成 wrapper + `block_size`（K11）+ `main(x, y, shape)` 契约 + `alpha` 特化 |
| `docs/Tilelang.language/数学操作/T.vexp.md`·`T.vsub.md`·`T.vmul.md` | API | 向量链各 op 规格与 dtype 矩阵 |
| `docs/Tilelang.language/比较操作/T.vcmp.md`·`条件操作/T.vselect.md` | API | 二路选择的 bool 片上用法 |
| `docs/Tilelang.language/数据类型转换操作/T.vcast.md` | API | fp16/bf16↔fp32 中转（`rint`） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(N,)`（原始 shape 展平） | `float16 \| bfloat16 \| float32` | 1-D 展平输入向量，`N = product(shape)` 编译期常量 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(N,)` | `same_as(x)` | 输出，shape/dtype 与输入一致 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | `(block_size,)` | `dtype` | UB | 输入 tile；**fp32 路径复用为计算域 `xc` 与结果缓冲**（搬出前原地复用） |
| `y_ub` | `(block_size,)` | `out_dtype` | UB | 输出 tile（**仅 fp16/bf16 路径**承载 `vcast` 单次舍回结果） |
| `xf` | `(block_size,)` | `float32` | UB | fp32 计算域 `xc`（fp16/bf16 = vcast 升级入口，链尾原地复用为结果；fp32 路径空闲） |
| `e` | `(block_size,)` | `float32` | UB | `exp(x)` → `exp(x)−1` → `×alpha`（负分支，全程原地） |
| `z` | `(block_size,)` | `float32` | UB | 常量 0（**只读**，循环外 `T.fill`，供 `vcmp` 同 shape 操作数） |
| `cond` | `(block_size,)` | `bool` | UB | `x > 0` 掩码（片上，从不写回 GM） |

> **buffer 分配策略**：全部在 `with T.Kernel` 顶层**无条件分配**（避免 `TRAP-conditional-alloc`）；零常量循环外填充；**`xc` 计算域别名**（fp32→`x_ub` / fp16/bf16→`xf`）并在链尾原地承载结果（§3.3，`PL-1.23`）。

### 4.4 内存搬运路径

```
fp32 路径（纯 Vector，无 L1/L0；xc = x_ub）：
GM[x] --T.copy--> UB[x_ub] --[ELU 链 4–5 op（原地复用 x_ub 承载结果）]--> UB[x_ub] --T.copy--> GM[y]

fp16/bf16 路径（fp32 中转；xc = xf）：
GM[x(dtype)] --T.copy--> UB[x_ub(dtype)] --T.vcast(rint)--> UB[xf(fp32)]
     --[ELU 链 4–5 op（原地复用 xf 承载结果）]--> UB[xf(fp32)] --T.vcast(rint)--> UB[y_ub(dtype)] --T.copy--> GM[y]
```

### 4.5 UB 内存预算

`block_size = 2048`：

| 路径 | x_ub | y_ub | xf | e | z | cond | 合计 | ×1.12（`CONST-capacity-910B2C` 上偏系数，stale） |
|------|------|------|----|----|---|------|------|------------------------------------------------|
| fp32 | 8 KB | 8 KB | 8 KB | 8 KB | 8 KB | 2 KB | **42 KB** | ≈ 47 KB < 192 KB ✓ |
| fp16 | 4 KB | 4 KB | 8 KB | 8 KB | 8 KB | 2 KB | **34 KB** | ≈ 38 KB ✓ |
| bf16 | 4 KB | 4 KB | 8 KB | 8 KB | 8 KB | 2 KB | **34 KB** | ≈ 38 KB ✓ |

> UB 硬上限 = 192 KB/AIV（`CONST-capacity-910B2C`，绑定 Ascend910B2C **且 stale**；本设备 Ascend910_9362 同规格量级）。**膨胀系数注**：`TRAP-UB-multibuffer-inflation` 记 auto-multi-buffer 可能膨胀 ~1.7×；本设计最大路径 42 KB，×1.7 ≈ 71 KB 仍 <192 KB（**安全**）。**block_size 放大空间**：`bs=4096` → fp32 84 KB（×1.7 ≈ 143 KB，可行）；`bs=8192` → fp32 168 KB（×1.7 ≈ 286 KB，**超限**）。故 `bs=3072/4096` 列为 Stage 4 发射项调优杠杆（§5.2），默认 `block_size` 按 dtype 取 K11 折叠值（fp32=1024 / fp16·bf16=2048）。

> **dtype 差异注**：K11 折叠值 fp32=1024、fp16/bf16=2048（源 `threads(256)×npt`，fp32 npt=4 / fp16·bf16 npt=8）。fp32 路径在 bs=1024 下最大 42/2 ≈ 21 KB，放大空间更大；为可追溯与对齐源，本设计**按 dtype 分别使用 K11 折叠值**（wrapper `_DEFAULT_BLOCK_SIZE`）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| `N` 的实际长度 | `shape: T.int32`（prim_func 参数） | 1 ~ manifest 最大 16,777,216 |

> `N` 同时作为编译期 `T.Tensor((N,), dtype)` 的 shape（按特定 N 特化编译）与运行时 `shape` 参数（尾块守卫）。manifest 全部 workload 的 N 均整除 `block_size`，但保留 `tail` 守卫以支持任意 N（测试 4,000,000 触发尾块）。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[1],        # y 是 prim_func 第 2 个参数（index=1）
    target="npuir",
)
def _make_elu_kernel(N, dtype, alpha, output_dtype=None, block_size=2048):
    ...
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅含逐元素向量链（`vcmp/vselect/vexp/vsub/vmul/vcast`），无 matmul、无归约、无 Cube 计算；仅需 UB 层级，不涉及 L1/L0A/L0B/L0C。

### 5.2 Block 划分

```python
block_size = {"float32": 1024, "float16": 2048, "bfloat16": 2048}[dtype]   # K11 折叠
n_num = T.ceildiv(N, block_size)   # 逻辑 tile 数（编译期静态值）
```

**block_size 选择理由**：
1. 与源 K11 折叠值一致（fp32 = `threads(256)×npt(4)` = 1024、fp16/bf16 = `threads(256)×npt(8)` = 2048），迁移可追溯；v-prefix 整 buffer 形态下 fp32 无 128-bit 对齐约束差异；
2. UB 对齐：block_size×elem_bytes 是 32B 的整数倍（fp32 4096B、fp16/bf16 4096B）✓；
3. UB 容量：最大 fp32 路径（bs=1024）≈ 21 KB，×1.7 最坏膨胀 ≈ 36 KB < 192 KB ✓；
4. **发射项权衡**：本算子 §1.6.0 估算显示**发射项绝对主导**（发射 ∝ 1/block_size），但 bs 放大受 UB 膨胀系数不确定性约束。默认取 K11 折叠值；**bs=3072/4096 列为 Stage 4 调优首选杠杆**（须以编译探针校准 UB 后采纳，分核参数调优，见 `core-split-strategy.md` §2.4）。

### 5.3 约束分析

- **对齐约束**：fp32 bs=1024，元素 4B → 每 tile 256 向量槽（128bit×4），尾 lane 浪费 0%；fp16/bf16 bs=2048，元素 2B → 256 向量槽（128bit×8），浪费 0%；均满足 32B 尾轴对齐 ✓。
- **UB 容量**：最大 42 KB（fp32, bs=2048 换算）/ bs=1024 实际 ≈ 21 KB（含 1.7× 最坏膨胀 < 192 KB）✓。
- **L0/L1 容量**：不适用（纯 Vector）。
- **分形限制**：不适用（无 Cube）。

### 5.4 注意事项（非整除处理策略）

**非整除场景预判**：manifest workload 的 N（1,048,576 / 8,388,608 / 16,777,216）均整除 `block_size`（fp32 1024 与 fp16/bf16 2048 均整除，因 N 为 2 的幂 ≥ 2²⁰）；测试 shape 4,000,000 **不整除 2048**（余 1216）→ 存在尾块。

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
| mlp-hidden `[2048,4096]` | fp16 | 8,388,608 | 2048 | 4,096 |
| mlp-hidden-wide `[2048,8192]` | fp16 | 16,777,216 | 2048 | 8,192 |
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
          ...
  ```
- **循环边界静态** ✓（`n_num`、`core_num` 均编译期常量，`T.ceildiv(n_num, core_num)` 为静态整数）；
- **残留负载不均**（显式记录）：以 mlp-hidden（fp16, bs2048）为例，`tasks_per_core = ceildiv(4096,40) = 103`；网格跨步下前 `4096 - 40×102 = 16` 个核执行 103 次、后 24 个核执行 102 次 → 临界路径 ≈ **+0.98%**。smoke-1m fp32（bs1024）`ceildiv(1024,40)=26`，临界 +2.3%；elementwise-16m fp32 `ceildiv(16384,40)=410`，临界 +0.24%。相较「直接 `T.Kernel(n_num)` 由运行时串行调度多波次」，核内 `T.serial` 把核启动固定开销从「每 tile 一次」摊薄为「每核一次」，净收益为正。
- **模式先例**：`examples/elementwise/vec_add_2d.py` L28–L42；同族 `bitwise_not_fwd_kernel.py` L89–L108（`PL-1.22-persistent-1d-grid-stride` 第二证）。

> 备选（供 Stage 4 参考，不在本设计实现）：直接 `T.Kernel(n_num)` 由运行时调度（无核内串行）——保留为对称性对照，若实测核内串行反而引入同步开销可回退；本设计按标准要求取核内串行 persistent 形态。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block 级 / tile 级调度循环；kernel 内无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级（任务网格） | block 级并行 | `T.Kernel(core_num, is_npu=True)` | 40 个物理 Vector 核各处理一组 tile |
| tile 级（核内） | 串行迭代（网格跨步） | `T.serial(ceildiv(n_num, core_num))` | 每核串行处理 `ceildiv(n_num,40)` 个 tile，摊薄核启动开销（§5.5） |
| 元素级 | 向量化 | `T.vcmp/vselect/vexp/vsub/vmul/vcast`（4–7 op） | 整 buffer 一次完成，无 `T.Parallel` 标量循环 |

### 6.2 循环伪代码

见 §3.3（`with T.Kernel(core_num, ...)` + `for i in T.serial(...)` + `if task < n_num`，循环内为 4–7-op ELU 向量链）。

### 6.3 流水线优化

**当前设计**: 不使用 `T.Pipelined`。理由：单 tile 内数据依赖线性（copy-in → vcast → 4–7-op 链 → vcast → copy-out），无 K 维迭代；跨 tile 迭代（`T.serial`）的 MTE2/VEC 重叠由 Developer 模式 auto-multi-buffer 自动处理。**发射项主导下**，若 Stage 4 需进一步优化（persistent double buffer / 更大 block_size），参照 `example_elementwise_exp2.py` 的 `T.serial` 多 tile 形态；设计阶段保持简洁。

### 6.4 尾块处理

见 §5.4。`N % block_size != 0` 时最后一个 tile 用 `tail = T.min(block_size, shape - off)` 动态裁剪 `T.copy` 量；manifest 全部 workload 整除（§5.4），仅测试 4,000,000 触发尾块（余 1216）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下编译器自动插入同步。核内数据依赖链严格线性（每 tile 内）：

```
T.copy(GM→UB) → [vcast↑] → vcmp ∥ (vexp → vsub → [vmul]) → vselect → [vcast↓] → T.copy(UB→GM)
```

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy(GM→UB)` 后 | 自动 | 编译器插 MTE2→V 等待 |
| 各 v-op 之间（4–7 op 链） | 自动 | 线性数据依赖（`vcmp` 与 `vexp` 链并行分支后汇于 `vselect`），插 V→V barrier |
| `T.copy(UB→GM)` 前 | 自动 | 编译器插 V→MTE3 等待 |
| tile 迭代之间（`T.serial`） | 自动 | 同 buffer 复用由 auto-multi-buffer 管理跨迭代依赖 |

无手动 `T.sync_block_set/wait`（无跨引擎/跨核协作，无 CV 融合）。

### 7.3 pass_configs 配置

```python
# Developer 模式默认配置，无需额外 pass_configs
# os.environ["TILELANG_ASCEND_MODE"] = "Developer"  # 可选，默认即 Developer
```

---

### 7.4 CV 融合设计

**n/a**：本算子是纯 Vector 逐元素算子，无 Cube 段（无 `T.gemm`）、无跨引擎协作、无核间 workspace，不涉及 CV 融合。据 AGENTS.md「Developer-mode MixCV trigger rule」——kernel 内既无 Cube 侧 `T.gemm` 亦无跨核流水，**非 MixCV**，不触发 mixcv skill。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（`F.elu(x.float(), alpha).to(dtype)`，保持源 fp32 中转语义），**不得复刻 §0.6 的 NPU 向量链结构**——保证验证独立性。

```python
import torch
import torch.nn.functional as F


def golden_elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """ELU 参考实现，等价于 torch.nn.functional.elu（保持源 fp32 opmath 语义）。

    源算子计算核 op = alpha*(T.exp(cast(x,'f32')) - 1)，二路选择 if x>0 取 x。
    对 fp16/bf16 输入先升 fp32 计算，末端单次舍回原 dtype；fp32 输入直接计算。
    CPU / 目标设备实测均证明 F.elu 负分支用 exp(x)-1（非 expm1），与源公式逐值
    一致，故 golden 直接采用 F.elu(x.float(), alpha).to(x.dtype)。
    """
    x_f32 = x.detach().cpu().to(torch.float32)
    return F.elu(x_f32, alpha).to(x.dtype)
    # 与源 fp32 中转语义一致；golden 独立于 NPU 向量链实现（不复用其结构）
```

> 备注：golden **独立于** NPU 向量链（直接调用 `F.elu` 库实现，不复用 §3.3 的 op 序列），满足验证独立性。**关键**：本算子 golden 用 `exp−1`（非 `expm1`），故 NPU 侧负分支**必须**用 `exp−1`，不得用 expm1/Taylor（见 §0.6 R2 / §1.6.1 诊断）。

### 8.2 精度标准

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 1e-5 | 1e-5 | 来源：`examples/TileOPs/tests/ops/test_activation.py` `_get_tolerances` |
| float16 | 1e-3 | 1e-3 | 来源：同上；NPU 在 fp32 域计算、单次舍回 |
| bfloat16 | 1.6e-2 | 1.6e-2 | 来源：同上；bf16 精度低，容忍度放宽 |

**误差预算核对（设计期设备实测，`design_probe_elu.py`）**：负分支相对误差 ≲ 1.2e-7（`vexp` rel 1.182e-7 + 精确 `vsub`/`vmul`）→ 设备端到端 fp32 `max_abs=5.96e-8`、fp16 `4.88e-4`、bf16 `3.9e-3`，角点 `0.0`（**全部 ≫ 优于门禁，fp32 ≥160× 余量**）。

### 8.3 L0 门槛测试计划

> L0 = 门槛 smoke 测试（最小可用性验证，进 Stage 3 前必过）；完整分层套件 L1/L2/Boundary 交由 `tilelang-op-develop`，不在此枚举。**显式包含负分支小 x、x=0/正负边界、各 dtype 用例**。

| 用例 | 输入 | 期望 | 判定 |
|------|------|------|------|
| L0-1 fp32 smoke（randn 全域） | `randn([1048576], fp32)`（NPU） | 与 `golden_elu(x,1.0)` 比对 ≤1e-5 | 门槛通过 |
| L0-2 fp16 smoke（randn 全域） | `randn([1048576], fp16)`（NPU） | 比对 ≤1e-3 | 门槛通过 |
| L0-3 bf16 smoke（randn 全域） | `randn([1048576], bf16)`（NPU） | 比对 ≤1.6e-2 | 门槛通过 |
| L0-4 fp32 角点（分发/边界） | `repeat([-88.8,-1.0,-1e-7,0.0,1e-7,1.0,88.8], 4096, fp32)`（NPU，`test_elu_edge` 同款） | 与 `golden_elu(x,1.0)` 比对 ≤1e-5（`-88.8→-1.0`、`±1e-7` 各归其支、`0→0`、`88.8→88.8`；**设计期设备实测角点 max_abs 0.0**） | 门槛通过 |
| L0-5 负分支小 x 敏感段（核心精度用例） | `linspace(-1e-3, 0, 1<<20, fp32)`（NPU） | 与 `golden_elu(x,1.0)` 比对 ≤1e-5（**必须用 `exp−1` 而非 expm1**；设计期设备实测 max_abs 5.96e-8） | 门槛通过 |
| L0-6 全域敏感区（含分支点 x=0） | `linspace(-9,9,1<<20, fp32)`（NPU） | 与 `golden_elu(x,1.0)` 比对 ≤1e-5（设计期设备实测 5.96e-8） | 门槛通过 |
| L0-7 非整除尾块 | `randn([4000000], fp16)`（NPU） | 比对 ≤1e-3（覆盖 4000000 % 2048 = 1216 尾块；设计期设备实测 7.6e-6） | 门槛通过 |
| L0-8 非默认 alpha | `randn([1048576], fp32)`，`alpha=0.5`（NPU） | 与 `golden_elu(x,0.5)` 比对 ≤1e-5（设计期设备实测 2.98e-8；覆盖 `vmul` 路径） | 门槛通过 |
| L0-9 各 dtype 边界 shape | `[2048,4096]`/`[2048,8192]` fp16·bf16（NPU，manifest） | 比对 ≤1e-3 / ≤1.6e-2 | 门槛通过 |

- **执行方式**：`tilelang.compile(func, target="npuir")` 后单次调用，`torch.testing.assert_close(y.cpu(), golden, atol, rtol)`（golden = §8.1）。
- **门槛标准**：L0-1..L0-9 全过即进入 Stage 3；任一失败不得继续。
- 输入生成参考 `examples/TileOPs/tileops/workloads/elementwise.py`（`EluWorkload`）；边界用例参考 `test_activation.py` 的 `ActivationEdgeFixture`（`repeat([-88.8,-1.0,-1e-7,0.0,1e-7,1.0,88.8])`）。
- **额外设计期落点**：`history_version/design_probe_elu.py`（设备探针：设备 `F.elu` 语义 + 向量链端到端 fp32/fp16/bf16 + 角点/小 x/尾块/alpha）、`verify_equiv.py`（D-1 机器验证，8/8 `EQUIV_PASS`）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 缓解方案 |
|------|------|----------|
| **R-1 无原生 `velu` 元算** | 须自建 4–7-op 向量链 | 直译源公式（§0.6 R2）；op 数远低于同族 expm1（16） |
| **R-2 bf16 v-prefix 数学 API 缺失**（各 op doc §2.2.1） | bf16 直传编译失败 | bf16 走 fp32 中转（§0.6 R3） |
| **R-3 与 golden 语义核对（前序 erf/expm1 教训）** | 若误用 expm1/Taylor 消抵消，会偏离 golden（`x=-1e-7` 差 16.1%） | **不采用** expm1 方案；负分支严格用 `vexp−1`；CPU + 设备双证 + `verify_equiv.py` 诊断（§0.6 R2 / §1.6.1） |
| **R-4 发射项主导**（4–7 op/tile，`CONST-vector-launch-overhead`） | §1.6.0 估算示 emission 远超 flow（51µs vs 7µs @smoke-1m fp32） | 标注「未文档化假设 + 估算依据」（常数绑 910B2C、fixed dispatch、stale、可能上偏）；block_size 为杠杆（§5.2）；Stage 4 baseline 后按 `hardware-cost-model.md` §3 对账 |
| **R-5 UB 容量** | bs 放大受限 | 默认按 dtype K11 折叠值（最大 42 KB×1.7≈71 KB 安全）；buffer 复用（`e` 原地、`xc` 原地）；bs=3072/4096 须编译探针校准 |
| **R-6 条件 `T.alloc` 非法**（`TRAP-conditional-alloc`） | 编译期 `if` 内 `T.alloc_ub` trace 失败 | buffer 全部在 `T.Kernel` 顶层无条件分配（§3.3） |
| **R-7 v-op 整 buffer vs 动态 slice**（`PL-1.22` R-6） | 传动态 slice 报错 | v-op 作用于整 buffer，仅 `T.copy` 用 `[0:tail]`（§3.3） |
| **R-8 vcast round_mode 限制**（`T.vcast.md` §2.2.1） | `f16→f32` / `f32→f16/bf16` **仅 `rint`** | 中转升降统一 `round_mode="rint"`（与 golden 单次舍回一致） |
| **R-9 `vcmp` bool 不可写回 GM**（`T.vcmp.md` §2.3） | 误写回编译失败 | `cond` 仅片上配 `vselect`，从不 `T.copy` 到 GM |
| **R-10 分核残留不均** | 40 核含非 2 因子 5，2ⁿ 逻辑核无法整除 | §5.5 网格跨步核内串行，残留临界 +0.98%（mlp-hidden）/ +2.3%（smoke fp32），已显式记录 |
| **R-11 alpha 折叠前提** | 折叠仅在 `float(alpha)==1.0` 时启用 | 编译期 Python 判定（精确 `==`，无浮点误差风险）；`verify_equiv.py` #4 证恒等；alpha≠1.0 走 `vmul` 路径（#5/#6 证等价） |
| **R-12 常数表设备/版本** | `CONST-mte2-degradation`/`CONST-vector-launch-overhead`/`CONST-capacity-910B2C` 绑 910B2C 且 `kb_stale_check` 标记 **stale**；本设备 910_9362 | 带宽/发射/容量数字标注「待重验」，仅作量级估算；不以 stale 条目为决策依据（**本设计的关键正确性结论来自本会话设备探针，不依赖 stale 条目**）；Stage 4 baseline 后对账 |
| **R-13 元算设备语义** | 文档语义可能与设备不一致（`TRAP-vrsqrt-plain-precision`、`TRAP-verf-maclaurin-truncation`） | 本设计关键 API `vexp/vsub/vmul/vcmp/vselect/vcast` 全部经**目标设备探针实测**（§3.4）；工具链升级后须重验 |
| **R-14 host 测试 harness alpha golden 偏差（待确认）** | 移植的 `test_activation.py::EluAlphaFixture` / `test_elu_alpha` 用 `EluTest.ref_program`（硬编码 `F.elu(x,1.0)`）比对 `op(alpha=0.5/2.0)` 的输出——**alpha 用例的 golden 与 op 的 alpha 不一致**（疑似移植测试 harness 缺陷，与 kernel 设计无关） | **设计侧不受影响**（kernel 通用 alpha 正确，见 L0-8 / `verify_equiv` #5/#6）；**列入风险交 Stage 3 核对**，建议 Stage 3 为该 fixture 补 `alpha` 感知的 `ref_program`（`F.elu(x.float(), alpha)`），或确认 harness 预期 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 误用 expm1/Taylor 消抵消 | 照搬同族 expm1 任务方案 | 小 x 处偏离 golden（`x=-1e-7` 差 16.1%） | 负分支严格 `vexp−1`（R-3） |
| bf16 直接传入数学 op | 未做 vcast 上转 | 编译失败 | bf16 先 `vcast` 到 fp32（R-2） |
| vcast 用非 rint 模式升级 | `f16→f32` 用 `round`/`floor` | 编译失败（仅 `rint`） | 升级/舍回统一 `round_mode="rint"`（R-8） |
| bool `cond` 写回 GM | `T.copy(cond_ub, ...)` | 编译失败（i1 不可 load/store GM） | `cond` 仅片上（R-9） |
| 条件 alloc | 编译期 `if need_cast:` 内 `T.alloc_ub` | trace 报 `Undefined variable` | buffer 顶层无条件分配（R-6） |
| v-op 传动态 slice | `T.vabs(x_ub[0:tail], ...)` | 可能编译/运行错误 | v-op 用整 buffer（R-7） |
| UB 溢出 | block_size 过大 | 编译失败 | 维持 K11 折叠值（默认）；§4.5 预算 + 编译探针（R-5） |
| 尾块越界 | N 非 bs 整数倍且无 tail 裁剪 | 越界读/写 GM | `tail = T.min(block_size, shape-off)`（§5.4） |
| 零常量 buffer 未填充 | 直接用未初始化的 `z` | `vcmp` 语义错误 | 循环外 `T.fill(z, 0.0)`（§3.3） |
| `vcmp` 操作数 shape 不一致 | 用标量 `0` 代替 tensor | 编译失败（A/B 须同 shape） | `T.fill(z,...)` 生成同 shape 常量 buffer（§1.6.2 #6） |
| 零输入崩溃 | N=0 空张量 | MTE DDR 崩溃（`TRAP-zero-input-crash`） | host 侧空张量守卫；本任务 N ≥ 1e6，不触发 |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| **非整除 N** | Kernel 内 `tail` 动态裁剪（§5.4），无需 host padding |
| **极小 N（N < block_size）** | `n_num=1`，`tail=N`，仅 `task==0` 的核工作，其余核空转 |
| **极大 N（>1.6e7）** | 核内 `T.serial` 网格跨步（§5.5），逻辑核数静态 |
| **`±inf/NaN` 输入** | `+inf → 取 x = +inf`；`−inf → vexp(−inf)−1 = −1`（alpha=1）；`NaN → NaN>0 为假 → vexp(NaN)−1 = NaN`（§0.6 R2 论证 4；设备探针角点实跑一致） |
| **`x=0` / `-0.0`** | `0>0` 为假 → 负分支 `alpha*(vexp(0)−1)=0`；与 golden 同值 0（±0 视为相等，容差口径） |
| **`x` 为大正数** | 正分支直接取 `x`（不经过 `vexp`）→ 无上溢，`x=88.8 → 88.8`（golden 一致，设备实测） |
| **`x` 为大负数** | `vexp` 下溢至 0 → `alpha*(0−1) = −alpha`（golden `−alpha`） |
| **混合 dtype** | 按 dtype 编译期特化（每 dtype 一份 kernel），输出 `same_as(input)` |
| **`inplace=True`** | host 侧 `input.copy_(result)`（不在 kernel 内，§0.1） |

---

## 10. 交付清单

| 交付物 | 路径 | 说明 |
|--------|------|------|
| 设计文档 | `examples/elu/_make_elu_kernel/DESIGN.md` | 本文件 |
| 等价性机器验证 | `examples/elu/_make_elu_kernel/verify_equiv.py` | D-1：8/8 `EQUIV_PASS`（§1.6.1） |
| 设计期设备探针 | `examples/elu/_make_elu_kernel/history_version/design_probe_elu.py` | D-3：设备 `F.elu` 语义 + 向量链端到端（§3.4） |
| NPU kernel（下游 Stage 3 产出，尚未创建） | 算子目录下 `_make_elu_kernel.py` | 按 §3.3 伪代码 + §5 分核实现 |
| Golden（下游） | 同上（kernel 文件内 `golden_elu`） | 依 §8.1 |
| 迁移集成 wrapper（已存在，只读参照） | `examples/TileOPs/tileops/kernels/elementwise/elu/elu.py` | K11 `block_size` + `main(x,y,shape)` 契约 |
| 精度用例（已存在，只读参照） | `examples/TileOPs/tests/ops/test_activation.py` | `test_elu` / `test_elu_edge` / `test_elu_alpha` |

**实现要点速览（Stage 3 输入）**：
1. 计算核 = `vcmp(xc,0,"gt")` → `vexp(xc)` → `vsub(·,1)` →（alpha≠1）`vmul(·,alpha)` → `vselect(cond,xc,neg,xc)`（4–7 op）；
2. dtype：fp32 直算；fp16/bf16 `vcast(rint)` 升 fp32 → 链 → `vcast(rint)` 舍回；
3. 结构：一维 `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步 + `tail` 尾块守卫；
4. `alpha==1.0` 编译期折叠 `vmul`；buffer 顶层无条件分配；零常量循环外 `T.fill`；
5. **禁止** expm1/Taylor 消抵消（golden 用 `exp−1`）。
