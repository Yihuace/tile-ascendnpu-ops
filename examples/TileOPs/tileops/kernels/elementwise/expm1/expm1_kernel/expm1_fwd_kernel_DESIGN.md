# expm1_fwd_kernel 算子设计文档（迁移）

> **项目**：`expm1` ｜ **算子目录**：`examples/expm1/expm1_fwd_kernel/` ｜ **编程模式**：Developer ｜ **target**：`npuir`
>
> **任务性质**：迁移（migration-harness）。把 GPU TileLang 函数 `expm1_fwd_kernel`（`op_func = T.exp(T.cast(x,"float32")) - 1.0`）重设计为 NPU 实现。
>
> **本版核心决策**：源 `exp(x)-1` 在 NPU 上可直接落为原生 `T.vexp`（设备实测真 exp，rel 1.18e-7），但 `exp(x)-1` 在小 `|x|` 处**灾难性抵消**（设备实测 x=1e-7 相对误差 19.2%），不符合 `torch.expm1`（小 `|x|` 相对精确）语义。故核心算法**重设计为分段 expm1**：`|x| < 0.5·ln2` 用 6 项 Taylor（`x·Σxⁿ/(n+1)!`，消抵消，相对误差 ≤ 5e-7），`|x| ≥ 0.5·ln2` 用 `vexp(x)-1`（此时 `exp(x)≥√2`，抵消可忽略）——经目标设备端到端实测与 `verify_equiv.py`（6/6 `EQUIV_PASS`）双证。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。

**源算子路径（源真相）**：`/home/developer/TileOPs/tileops/kernels/elementwise.py`

- 具体算子：`Expm1FwdKernel`（L2142–L2147，`FloatUnaryKernel` 子类）：

  ```python
  class Expm1FwdKernel(FloatUnaryKernel):
      """Element-wise exp(x) - 1."""
      @staticmethod
      def op_func(x):
          return T.exp(T.cast(x, "float32")) - T.cast(1.0, "float32")
  ```

- 基类与模板：`FloatUnaryKernel`（L1297–L1300，`SUPPORTED_DTYPES=_FLOAT_DTYPES`）、`UnaryKernel`（L680–L887，`DEFAULT_STRATEGY="register_copy"` L700）；
- 策略工厂：`_make_unary_regcopy`（L344–L363，fragment load → compute → fragment store）；
- 提取产物（脚手架）：`examples/TileOPs/tileops/kernels/elementwise/expm1/_expm1_fwd_kernels.py`（`main(x, y)` + `T.Parallel(threads,npt)`）；集成包装：同目录 `expm1.py`（K11：`block_size` = 1024 fp32 / 2048 fp16·bf16）。

### 0.1 源算子语义（做什么）

**数学语义**：`y = expm1(x) = e^x − 1`，逐元素（manifest `ref_api = torch.expm1`，`examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `Expm1FwdOp`）。

**契约语义 = `torch.expm1`**（测试 golden）：数学意义上 `expm1` 在小 `|x|` 处必须**相对精确**（`expm1(x) ≈ x + x²/2`），这正是 `expm1` 相对 `exp(x)-1` 存在的意义（`torch.expm1` 由 libm 特化实现，非 `exp-1`）。

**⚠️ 源实现与契约的差距（迁移必须正视）**：源 `op_func` 是 `exp(x)-1` 的**直译**，在小 `|x|` 处发生灾难性抵消，其相对精度**不满足** `torch.expm1` 语义（设备实测：`x=1e-7` 输出 `1.192e-7` vs 真值 `1.000e-7`，相对误差 19.2%）。源 GPU kernel 之所以通过测试，是因为测试门禁带 **atol=1e-5**（绝对值口径），抵消产生的是**相对**误差而其绝对误差仅 ~2e-8。因此：
- 迁移的**语义基线**是 `torch.expm1`（§8.1 golden），不是源 `op_func` 的数值输出；
- NPU 侧**允许且应当**用更精确的算法实现 `torch.expm1` 的契约语义（比源实现更接近契约），这**不构成语义改变**。

**dtype 语义（迁移必须保持的关键点）**：源实现**强制 fp32 中转**——`T.exp` 的输入恒为 `float32`（`T.cast(x,"float32")`）。即：
- 对 fp16/bf16 输入：`x → cast fp32 → exp(fp32) − 1 → 隐式舍回原 dtype`（`y_reg` dtype = 输入 dtype，写回一次舍入）；
- 对 fp32 输入：`cast` 为恒等。
该 `fp32 opmath + 一次舍回` 语义与 CPU golden `torch.expm1` 的 fp32 opmath 一致，是迁移**不得丢失**的 dtype 语义（NPU 侧以 `T.vcast(rint)` 承接）。

**规约语义**：无规约（逐元素映射，无跨元素依赖），不存在累加顺序问题。

**边界语义**：`expm1(0)=0`；`expm1(+∞)=+∞`；`expm1(−∞)=−1`；`expm1(NaN)=NaN`；fp32 下 `x ≳ 88.72`（`ln(finfo.max)`) 时 `expm1(x)` 溢出为 `+∞`（`torch.expm1` 同样为 `+∞`，fp16 下阈值更低）。空维度/非整除 shape 由 host 侧 flatten 成 1-D `[N]` 后进入 kernel。

**host 侧语义**：`UnaryKernel.__init__` 把任意 shape 展平为 `N_total = numel`；`forward` 仅调用编译后 kernel，无 host 预处理/后处理。

**语义保持基线**：§8.1 golden 以本节语义（`torch.expm1`，保持源 fp32 中转语义）为唯一依据实现，**不得复刻 §0.6 的 NPU 分段算法**（保证验证独立性）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `(N,)`（原始 shape 被 host 侧展平；manifest 有 `[1048576]` / `[4096,4096]` / `[8192,8192]` / `[16384,16384]`） | `float16 \| bfloat16 \| float32` | 1-D 展平输入向量，`N = product(input.shape)` |
| `y` | `same_as_input`（输出 shape = 输入 shape，展平后 `(N,)`） | `same_as(input)` | 逐元素 `expm1` 输出 |

**输出 shape（迁移必需）**：`same_as_input`（manifest `shape_rules: output.shape == input.shape`）。
**layout**：1-D 连续（contiguous flatten），无转置。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（`register_copy` 策略，`_make_unary_regcopy`，L344–L363）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `x_reg ← x[bx*bs:(bx+1)*bs]`（fragment 载入） | GM `x` | fragment `x_reg` | 元素定位/搬入 |
| 2 | `y_reg[i] = T.exp(T.cast(x_reg[i],'float32')) - 1.0`（逐线程 `i,j` 标量映射） | `x_reg[i]` | `y_reg[i]` | `y = expm1(x)`（fp32 域） |
| 3 | `y[bx*bs:(bx+1)*bs] ← y_reg`（fragment 写回） | `y_reg` | GM `y` | 搬出 |

**数据流与内存访问模式**（源硬件 = GPU）：

```
GM[x] --(fragment load, 128-bit)--> registers[x_reg]
   --(T.Parallel(threads,npt) 标量 exp(cast(x,'f32')) - 1)--> registers[y_reg]
   --(fragment store)--> GM[y]
（无显式 SMEM；fragment 双缓冲由编译器管理；block_size = threads × npt）
```

**循环与并行结构**：`T.Kernel(T.ceildiv(N, block_size), threads=256)`——一维 grid（`bx`）+ 256 线程/block 两级并行；每线程 `npt` 元素，`T.Parallel(threads_arg, npt_arg)`。`block_size = 256 × npt`：fp32 `npt=4`→1024，fp16/bf16 `npt=8`→2048。

**host 侧逻辑**：`N_total`、`dtype` 由 Op 层传入；`forward` 直接调用编译后 kernel；flatten 由 Op 层完成（`wrapper expm1.py` 的 `_DEFAULT_BLOCK_SIZE` = {fp32:1024, fp16:2048, bf16:2048}）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | `register_copy` 策略（fragment load→compute→store） | 向量化 128-bit 访存 | `T.alloc_fragment` + `T.copy` 切片 + `T.Parallel` 标量映射 | GPU 大寄存器堆 + memory coalescing | **硬件强相关**（fragment/register 模型） |
| 2 | block×thread 两级并行（大 grid + 256 线程/block） | 用大规模线程/grid 吃满内存带宽 | `T.Kernel(ceildiv(N,bs), threads=256)` + `T.Parallel` | GPU thread/warp 执行模型 | **硬件强相关**（GPU 线程模型） |
| 3 | **fp32 中转**（`T.cast(x,'float32')` → `T.exp`） | 以 fp32 域计算保住精度（规避半精度 exp） | 半精度输入先升 fp32，末端一次舍回原 dtype | fp32 算术单元 | **语义层**（fp32 opmath 精度语义，可移植）；**API 层不可移植**（详见 §0.5 #3） |
| 4 | strategy-aware `npt`（fp32=4、fp16/bf16=8） | 128-bit 访存对齐（4×4B / 8×2B） | `_strategy_npt` 启发式 | memory coalescing（对齐） | 实现层（NPU 折叠为 `block_size`） |
| 5 | 无 autotune（elementwise 固定启发式配置） | 带宽受限 + 单步逐元素，搜索 ROI 低 | `default_config` 固定 `threads=256`+`npt` | 无硬件依赖 | 可移植（NPU 侧同样固定单实现） |
| 6 | 单条超越函数指令（`T.exp`）完成逐元素映射 | 低 op 数 | 库函数 `expf` | GPU 超越函数单元 | **算法层**：NPU 有原生 `T.vexp`（设备实测真 exp）；但**直译 `exp-1` 破坏 `torch.expm1` 小 x 语义**（本迁移核心问题，§0.6 R2） |

> 识别不出来 ≠ 不存在。上述 6 项为源码中可识别的全部优化/规避手段；后续迁移逐项处置（§0.5）。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 Phase R 调研结论 §1.6.0）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `y = expm1(x)`（含 fp32 opmath 精度语义） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 依 §0.1（`torch.expm1`） |
| 计算算法（步骤 1–3 逐元素映射结构） | 算法 | 无（纯数学顺序、无跨元素依赖） | — | **保留（结构）** | 逐元素映射结构保留；计算核需重设计（下述 #3/#6） |
| 优化 #1 `register_copy`（fragment + `T.Parallel` 标量映射） | 优化 | GPU register/fragment 模型 | 无直接等价（NPU 无一维 `threads=`；v-prefix op 作用于整 UB buffer） | **重新设计** | → §0.6 R1：GM→UB→GM + 整 buffer v-prefix 向量链 |
| 优化 #2 block×thread 两级并行 | 优化 | GPU thread/warp 模型 | 无直接等价（NPU 一维 Kernel） | **重新设计** | → §0.6 R1：一维 `T.Kernel(core_num, is_npu=True)` + 核内 `T.serial` 网格跨步 |
| 优化 #3 fp32 中转（`T.cast` 标量升 fp32） | 优化/语义 | CUDA 半精度 exp 缺失 + fp32 算术 | 有（`T.vcast` 向量升/降，`rint`） | **等价替换** | → §0.6 R3：fp16/bf16 `vcast(rint)` 升 fp32 计算、单次 `vcast(rint)` 舍回（bf16 为 v-prefix 数学 API 缺失下的强制中转） |
| 优化 #6 计算核 `T.exp(x) − 1` | 算法 | CUDA `expf` + `T.cast` | **部分**：NPU 有真 `T.vexp`（rel 1.18e-7），但 `exp−1` 在小 \|x\| 处灾难性抵消 → **不满足契约语义** | **重新设计（算法替代）** | → §0.6 R2：分段 expm1（小 \|x\| Taylor6 消抵消 + 大 \|x\| `vexp−1`） |
| 优化 #4 strategy-aware `npt` 对齐 | 优化 | memory coalescing（对齐） | 有（32B 尾轴对齐 / 向量宽度 128bit） | **等价替换** | K11：`threads × npt` 折叠为 `block_size`（§5.2），对齐约束按 NPU 32B 复算（§5.3） |
| 优化 #5 无 autotune | 优化 | 无 | 有 | **保留** | NPU 侧同固定单实现（无 strategy/autotune） |
| host 侧 flatten（非 kernel 逻辑） | 算法（host） | 无 | — | **保留** | Op 层传入 `N_total`，kernel 只处理 1-D |

> **依据**：#3 的 vcast 可用性由 `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（f32↔f16/bf16 支持 `rint`）+ 设备探针 P4 双证；#6 的「`vexp` 为真 exp 但 `exp−1` 抵消」由设备探针 P0/P1/P3 三证（见 §1.6.0 R4 / §3.4）。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（承接源方案性能意图 + 语义保持论证）。

**重设计项 R1：并行结构重设计（thread 级并行 → 一维 Kernel + 核内网格跨步）**

- **源方案**：`T.Kernel(ceildiv(N, block_size), threads=256)` + `T.Parallel(256, npt)`，每线程 `npt` 元素；性能意图 = 用大规模线程/grid 吃满内存带宽。
- **NPU 新算法**：一维 `T.Kernel(core_num, is_npu=True)`（`core_num` = 实查物理 Vector 核数 **40**，§5.5 ②）+ 核内 `for i in T.serial(T.ceildiv(n_num, core_num))` 网格跨步任务映射（`task = i * core_num + cid`，`if task < n_num` 守卫）；每 tile 经 `T.copy` GM→UB、整 buffer 向量 op、`T.copy` UB→GM。**模式先例**：`examples/elementwise/vec_add_2d.py` L28–L42、同族迁移 `examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py` L89–L108（`PL-1.22-persistent-1d-grid-stride` 两证）。
- **语义保持论证**：任务映射只重排「哪个核处理哪个元素区间」，各元素计算完全独立（无跨元素依赖、无累加顺序变化），数学语义逐元素恒等；边界：`T.serial` 循环上界 `T.ceildiv(n_num, core_num)` 为编译期静态值，守卫 `task < n_num` 覆盖非整除尾块，不产生越界。

**重设计项 R2：计算核算法替代（源 `exp(x)−1` → 分段 expm1）⭐ 本迁移核心**

- **源方案**：`y = T.exp(T.cast(x,'float32')) - 1.0`——用原生 exp 后减 1；性能意图 = 单条超越函数指令完成逐元素映射 + fp32 精度。
- **NPU 现状（设备实测）**：NPU **有**原生真 exp 元算（`T.vexp`，设备实测 rel 1.18e-7，§3.4 P1），故 `exp(x)−1` 结构可直接移植；**但直译会继承灾难性抵消**（设备 P0 实测：`x=1e-7` 相对误差 **19.2%**、`x=1e-6` **4.6%**、`x=1e-3` 2.3e-5），与 `torch.expm1` 的小 `|x|` 相对精确语义不符。
- **NPU 新算法（算法替代，分段 expm1）**：设阈值 `T = 0.5·ln2 ≈ 0.34657359`（`exp(T)=√2`）：

  $$
  \operatorname{expm1}(x)\;\approx\;
  \begin{cases}
  x\cdot\displaystyle\sum_{n=0}^{5}\frac{x^{n}}{(n{+}1)!}=x\Bigl(1+\tfrac{x}{2}+\tfrac{x^2}{6}+\tfrac{x^3}{24}+\tfrac{x^4}{120}+\tfrac{x^5}{720}\Bigr), & |x|<T\\[6pt]
  \operatorname{vexp}(x)-1, & |x|\ge T
  \end{cases}
  $$

  实现为**单条无分支向量链**（两分支皆算 + `T.vselect` 选路），op 顺序 = §3.3 伪代码（16 op fp32 / 18 op fp16·bf16）。
- **语义保持论证（容差内等价 + 边界核对）**：
  1. **小 `|x|` 分支误差界**：`expm1(x) = x·Σ_{n≥0} xⁿ/(n+1)!` 的 Taylor 截断余项 `|R| = |x|·|Σ_{n≥6} xⁿ/(n+1)!| ≤ |x|⁷/5040·e^{|x|} ≤ T⁷/5040·e^T = 1.19e-7×1.414 ≈ 1.68e-7`（绝对）；相对误差 `≤ |x|⁶/5040·e^T ≤ T⁶/5040·e^T ≈ 4.86e-7`。`x→0` 时 `expm1≈x`，多项式首项即 `x`，**相对误差不退化**（设备实测 `x=1e-7` rel=7.1e-8）。
  2. **大 `|x|` 分支误差界**：`|x| ≥ T ⇒ exp(x) ≥ √2 ⇒ expm1(x) = exp(x)−1 ≥ 0.414`，减法抵消放大因子 `exp(x)/expm1(x) ≤ √2/(√2−1) = 3.414`；故减 1 的舍入相对误差 `≤ 3.414 × 2^-24 ≈ 2.0e-7`，叠加 `vexp` 自身相对误差 `≤ 3.414×1.18e-7 ≈ 4.0e-7`（边界处最大，随 `exp(x)` 增大递减至 ~1.18e-7）。**综合两分支相对误差 ≲ 5e-7 ≪ rtol 1e-5（≥20× 余量）**；近 0 处绝对误差 `≤ T·4.9e-7 ≈ 1.7e-7 ≪ atol 1e-5`。
  3. **阈值连续性**：两分支在 `|x|=T` 处均为解析值 `expm1(T)`，两侧误差分别 ≤4.9e-7 与 ≤4.0e-7，无跳变。
  4. **极值/边界**：`x=0 → 0·g(0)=0`；`x=+∞ → vexp(∞)−1=∞`（golden `+∞` ✓）；`x=−∞ → vexp(−∞)−1=0−1=−1`（golden `−1` ✓）；`x=NaN → |NaN|<T 为 false → vexp(NaN)−1=NaN`（golden `NaN` ✓）；`x=88.8 → exp 溢出 → ∞`（golden `torch.expm1(88.8)=∞` ✓）。**大 `|x|` 饱和无需额外 clamp**（由 `vexp` 的 inf/0 行为天然覆盖）。
- **源优化意图承接**：源 #3「fp32 精度」意图 → R2（分段链在 fp32 域构造）+ R3（fp32 中转）承接，且**精度优于源直译**（小 x 相对误差 19.2% → 7.1e-8）；源 #6「单条超越函数指令」性能意图在 NPU 上以 **1× `vexp` + 15 个算术/选择 op** 落地（无原生 `vexp1m` 元算，见 §1.6.0 R4），代价已量化并作为工程风险显式登记（§1.6.0 估算 / §9）。

**重设计项 R3：dtype 路径重设计（标量 `T.cast` → 向量 `T.vcast` fp32 中转链）**

- **源方案**：标量 `T.cast(x, "float32")` 使计算在 fp32 域进行，末端隐式舍回原 dtype；性能意图 = 以 fp32 精度保住结果。
- **NPU 新算法**：
  - **fp32 路径**：直接在 fp32 UB buffer 上跑 R2 分段链；
  - **fp16/bf16 路径**：`T.vcast(x_ub, xf, round_mode="rint")` → R2 分段链（fp32 域）→ `T.vcast(xc, y_ub, round_mode="rint")`；
  - **bf16 为 API 强制中转**：`T.vexp/vmul/vadd/vsub/vabs` 的 dtype 矩阵均**不含 bf16**（`docs/Tilelang.language/数学操作/*.md` §2.2.1），故 bf16 必须 `vcast` 升 fp32 计算、末端舍回。
- **语义保持论证**：fp32 域算 `expm1` 后单次舍回原 dtype ≡ 源 fp32 中转语义；`vcast` 升 fp32 无损、降回 `rint` 单次舍入。**设备实测**（§3.4 P4）：fp16 路径 `max_abs=0.0`、bf16 路径 `max_abs=0.0`（对 `torch.expm1` 逐值 bit-exact），0 违例。

**重设计项 R4：阈值条件选择（源无此结构，NPU 新增）**

- **源方案**：无（源为单一 `exp-1` 链）。
- **NPU 新算法**：`T.vabs(xc, xa)` 取 `|x|`；`T.vcmp(xa, thr, cond, "lt")` 得 `cond = |x|<T`（`thr` 为循环外一次性 `T.fill` 的只读常量 buffer）；`T.vselect(cond, x·g, vexp(x)−1, xc)` 选路。
- **语义保持论证**：无分支发散（SIMD 屏蔽），两分支同一元素至多一个被选中；`x=0`/`±∞`/`NaN` 均验证与 golden 一致（R2 论证 4）。`cond` 为 bool 仅片上使用（配 `vselect`），从不 `T.copy` 到 GM（`T.vcmp.md` §2.3）。

> 承接说明（源优化意图不丢弃）：§0.4 #1「向量化 128-bit 访存」→ R1/R2 以「GM→UB 整块 `T.copy` + 向量 op 整 buffer」承接；#2「大规模并行吃带宽」→ R1 网格跨步并行承接；#3「fp32 精度」→ R3 fp32 中转链承接；#4「对齐」→ K11 折叠 `block_size` + §5.3 32B 对齐复算；#5 保留；#6「单超越指令」→ 以 1× vexp + 15 op 承接（NPU 无 vexpm1，代价量化登记）。

### 0.7 标杆实现

- 源算子文件（golden 语义依据）：`/home/developer/TileOPs/tileops/kernels/elementwise.py`（`Expm1FwdKernel`，L2142–L2147）。
- 源仓测试基准：`examples/TileOPs/tests/ops/test_unary_math.py`（`test_expm1` = `randn` + `torch.expm1`；`test_expm1_edge` = `repeat([0.0, 88.8, -88.8, 1e-7])`；容差 `_get_tolerances` fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2）。
- manifest：`examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `Expm1FwdOp`（workloads `[4096,4096]` fp16/bf16/fp32、`[16384,16384]` fp16/bf16；roofline flops=`2*N`，bytes=`2*N*elem_bytes`）。
- golden 以 §0.1 语义为依据（`torch.expm1`，保持 fp32 中转语义），**不复刻 §0.6 的分段算法**（保证验证独立性）。
---

## 1. 概述

### 1.1 算子名称

`expm1_fwd_kernel`（项目 `expm1`，目录 `examples/expm1/expm1_fwd_kernel/`）。

### 1.2 功能描述

对 1-D 展平张量逐元素计算 `y = expm1(x) = e^x − 1`，语义等价于 `torch.expm1`；支持 `float16 | bfloat16 | float32`，输出与输入 shape/dtype 一致。单步纯 Vector 逐元素算子，无规约、无矩阵运算、无跨元素依赖。保持源算子的 **fp32 中转**精度语义（fp16/bf16 → fp32 域计算 → 单次舍回）。

**本版核心变化**：源 `exp(x)−1` 直译（`T.vexp − 1`）在小 `|x|` 处灾难性抵消（设备 P0 实测 `x=1e-7` 相对误差 19.2%），不满足 `torch.expm1` 语义；NPU 侧改为**分段 expm1**（小 `|x|` 6 项 Taylor + 大 `|x|` `vexp−1`），设备端到端实测小 x 相对误差 ≤1.2e-7、全域 0 违例（§3.4）。

### 1.3 数学公式

$$
y_i = \operatorname{expm1}(x_i)=e^{x_i}-1,\quad i=0,1,\ldots,N-1
$$

保持源 fp32 中转语义（§0.1）：对所有 dtype，`expm1` 在 fp32 域计算后单次舍回原 dtype。NPU 计算核为分段形式（§1.4 / §1.6.1）：

$$
y = \begin{cases} x\,\bigl(1+\tfrac{x}{2}+\tfrac{x^2}{6}+\tfrac{x^3}{24}+\tfrac{x^4}{120}+\tfrac{x^5}{720}\bigr) & |x| < 0.5\ln 2\\[4pt] e^{x}-1 & |x| \ge 0.5\ln 2 \end{cases}
$$

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法（§0.6），与源算法存在差异：**

- **计算核心**（§0.6 R2 / §1.6.1）：**分段 expm1**——`|x| < T`（`T=0.5·ln2`）用 6 项 Taylor `x·Σ_{n=0}^{5}xⁿ/(n+1)!`（消灾难性抵消）；`|x| ≥ T` 用 `T.vexp(x) − 1`。源为 `T.Parallel` 内标量 `exp(cast(x,f32))−1`；NPU 改为整 UB buffer 的 16-op 向量链（含 `vexp` 真 exp 元算）。
- **dtype 路径**（§0.6 R3）：fp32 直算；fp16/bf16 先 `T.vcast(rint)` 升 fp32 → 分段链 → `T.vcast(rint)` 舍回。保持源 fp32 opmath 语义。
- **阈值选路**（§0.6 R4）：`T.vabs` + `T.vcmp` + `T.vselect` 无分支选路。
- **并行结构**（§0.6 R1）：一维 `T.Kernel(core_num, is_npu=True)` + 核内 `T.serial` 网格跨步。

除上述外，算法与 §1.6.0 调研选定的算法族（单步逐元素分段逼近，访存为理论下界）一致：一趟扫描、无中间 GM 往返、无跨元素依赖。

### 1.5 数据流图

```
fp32 路径（xc = x_ub，链尾原地复用为结果）：
GM[x] --T.copy--> UB[x_ub] --[分段 expm1 向量链 16 op in fp32]--> UB[x_ub] --T.copy--> GM[y]

fp16/bf16 路径（fp32 中转，xc = xf）：
GM[x(dtype)] --T.copy--> UB[x_ub(dtype)] --T.vcast(rint)--> UB[xf(fp32)]
     --[分段 expm1 向量链 16 op]--> UB[xf(fp32)] --T.vcast(rint)--> UB[y_ub(dtype)] --T.copy--> GM[y]
```

分段链内部（全部 UB 级、原地复用）：
```
xc(=x_ub 或 xf) → vabs(=|x|)=xa → vcmp(|x|<T)=cond
  ∥ vmul(xc,C5)→g;vadd(g,C4);vmul(g,xc);vadd(g,C3);vmul(g,xc);vadd(g,C2);vmul(g,xc);vadd(g,C1);vmul(g,xc);vadd(g,C0)  [Horner]
  → vmul(xc,g)=xa(小分支)
  ∥ vexp(xc)=e; vsub(e,1)=e(大分支)
  → vselect(cond, xa, e)=xc
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（对同一数学语义的算法族回答调研四问，选定算法族），再在数学等价前提下优化公式，再做循环/标量向量化替代分析。迁移任务：1.6.1–1.6.3 的分析对象为 §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 本算子为**单步逐元素类**（§0.3 步骤 3 步、无规约/窗口/矩阵），按 `_shared/standards/algorithm-research.md` §3 执行调研，四问逐问回答（R1 因存在成熟的数值逼近算法族，给出候选表）。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线：源算法（GPU） | `y = T.exp(cast(x,'f32')) − 1` | 与契约 `torch.expm1` **仅容差内等价**（小 \|x\| 相对误差 19%） | — | ✅（基线，NPU 可移植但精度不足） |
| 1 | 直译 `vexp(x) − 1`（NPU 原生） | `T.vexp(x) − 1` | 同上；NPU 有真 `vexp`（rel 1.18e-7），但抵消不变 | op 最少（2 op） | ✅ |
| 2 | **分段 Taylor（采纳）** | `\|x\|<0.5ln2: x·Σ_{n=0}^{5}xⁿ/(n+1)!；否则 vexp−1` | **容差内等价**：小 \|x\| 相对误差 ≤4.9e-7，大 \|x\| ≤4.0e-7，均 ≪ rtol 1e-5 | 修抵消，+14 op | ✅ |
| 3 | Cephes expm1f 有理逼近 | `\|x\|<0.5ln2: x + x·x²·P(x²)/Q(x²)；否则 vexp−1` | 容差内等价（libm 级精度） | 有理式需 vdiv/多系数，op 多于候选 2 | ✅ |
| 4 | 全局单一多项式 | `Σ aₙ xⁿ` 全域 | **非等价**（大幅值指数增长，多项式发散） | 无 | ❌（结构上不适用全域，同候选 1 病根） |
| 5 | `2·sinh(x/2)·exp(x/2)` | 双曲恒等式 | 容差内等价，但小 x 下 `sinh` 自身抵消（需再用 Taylor） | 无增益 | ❌（NPU 无 `vsinh`；且未消抵消） |
| 6 | `log1p` 关系反解 | `y` 满足 `log1p(y)=x` | 需迭代/求逆，非闭式 | 无 | ❌（非直接公式，代价高） |
| 7 | `exp2` 换底 | `expm1 = vexp2(x·log2e) − 1` | 同候选 1，抵消不变 | 无 | ❌（未消抵消，`T.vexp2` 见 `docs/.../T.vexp2.md`） |
| 8 | libm 区间约减 + 指数拼装 | 拆 `x = k·ln2 + r`，`expm1 = 2^k(1+expm1(r))−1` | 容差内等价 | op 更多、需位操作 | ❌（op 多于候选 2） |
| 9 | 恒等变形（Horner/CSE/降代换） | 候选 2 内部求值重排 | 数学恒等 | 见 §1.6.1 | ✅（部分纳入 §1.6.1） |

**R2 在线算法**：**无在线化收益**——逐元素映射本身天然单遍（无跨元素依赖，不存在可被「单遍/流式」消去的多遍基线）。结构判据：本算子无 running 统计量（无归约/无 max/无 sum），在线变体定义不适用。依据：源算法即单遍直写；`examples/elementwise/` 全系逐元素实现（`vec_add_1d.py`、`example_elementwise_add.py`）单遍。

**R3 复杂度对比**（四口径；本算子带宽/发射受限，访存/遍数是主导项）：

| 算法候选 | FLOPs（v-op 数/元素） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|----------------------|---------------|---------|-------------|---------------------|
| 基线（源 `exp−1` 直译） | 2（exp1+sub1） | 2·N·elem_bytes | 1 遍 | O(block_size) | 可分块给多核，无跨核归并 |
| 候选 1（`vexp−1`） | 2 | 2·N·elem_bytes | 1 遍 | O(block_size) | 同上；**小 x 相对误差 19%** |
| **候选 2（分段 Taylor，采纳）** | **16**（abs1+cmp1+Horner10+mul1+exp1+sub1+select1） | 2·N·elem_bytes | 1 遍 | O(block_size)（~6 个 fp32 UB buffer，§4.5） | 同上；无跨核归并 |
| 候选 3（Cephes 有理） | ~24（含 vdiv） | 2·N·elem_bytes | 1 遍 | 同量级 +P/Q 缓冲 | 同上 |
| 候选 8（区间约减） | ~20（含位操作） | 2·N·elem_bytes | 1 遍 | 同量级 | 同上 |

> 口径说明：FLOPs 以「每元素 v-op 数」计（超越函数 `vexp` 按 1 op 记，估算口径）；候选 2 计入 fp16/bf16 的 +2 `vcast`（合计 18）；访存 = 输入读 + 输出写，**全部候选同为理论下界 `2·N·elem_bytes`**（无中间 GM 往返）；扫描遍数 = GM 读取次数 = 1；中间缓冲 = UB 单 tile 驻留（非 O(N)）。候选差异在 **op 数（发射成本）**。

**R4 硬件亲和性评估**（逐候选对照检查清单）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线 / 候选 1（`exp−1`） | Vector（`T.vexp` 真 exp，设备实测 rel 1.18e-7） | 小（2 buffer） | ✓ | ✓ | 单步 | ⚠️ 可跑但**小 x 相对误差不满足契约**（设备 P0） |
| **候选 2（分段 Taylor，采纳）** | 全部 op 落 Vector（`vabs/vcmp/vmul/vadd/vexp/vsub/vselect`，均为设备实测可用元算，§3.4） | ~6 fp32 工作 buffer + 常量，@bs2048 fp32 ≈ 58 KB（§4.5）✓ | 尾轴 32B 对齐 ✓；v-op 作用于整 buffer | N 编译期常量 ✓ | 单步无多阶段；跨 tile 重叠由 auto-multi-buffer 处理 | ✅ **采纳**（精度达标、容量充足、op 全部设备验证） |
| 候选 3（Cephes 有理） | Vector（需 `vdiv`） | 略多 | ✓ | ✓ | 同 | ⚠️ 备选：精度更高但 op 更多（~24 vs 16），本仓无实测 |
| 候选 5（sinh）/ 6（log1p 反解）/ 7（exp2 换底） | Vector | — | — | — | — | ❌ NPU 无 `vsinh`（`docs/Tilelang.language/数学操作/` 枚举无此 op）；6 非闭式；7 未消抵消 |
| 候选 4（全局多项式）/ 8（区间约减） | Vector | — | — | — | — | ❌ 4 结构上发散；8 op 更多、无增益 |

**调研结论**：**选定候选 2（分段 expm1：小 `|x|` 6 项 Taylor + 大 `|x|` `vexp−1`），无更优可行替代。**

关键依据：① R3 全部候选访存同为下界 `2N·elem_bytes`，差异在 op 数——候选 2（16 op）在「精度（小 x 相对误差 ≤4.9e-7，20× 余量）」与「op 数」间取平衡，且**修掉候选 1 的抵消**（本迁移契约的核心要求）；② R4 逐候选淘汰：候选 1 精度不足、候选 3 为可接受备选（精度更高但 op 更多、无实测）、候选 4/5/6/7/8 结构不适用或无增益；③ 候选 2 的全部 op 均在**目标设备实测**为真实可用元算（§3.4 探针表）。

**调研范围**（「无更优替代」依据）：① 参考表 `tilelang-op-design/references/algorithm-candidates.md` 的 `ALG-elementwise` 条目（R1 记「公共子表达式消除；除法乘倒数；cast 链合并」；R2 记「逐元素映射天然单遍」；**已知实测**：sub-fp32 逐元素算子 fp32 中转链）——本仓算法候选库不含 expm1 专门逼近条目；② 本仓 `examples/` 全系（`elementwise/*`、`erf`、`bitwise_not`、`mish`）中 `erf` 由自建 A&S 逼近、`mish`/`bitwise_not` 为有原生元算直译形态，**无 expm1 逼近先例**；③ pattern-library：`kb_search "expm1 elementwise sub-fp32 fp32 transit"` 命中 `PL-1.17-subfp32-fp32-transit`（fp32 中转模式，**stale**）、`VP-2026-0002`、`ALG-elementwise`、`VP-2026-0141`（设备元算精度矩阵：`T.vexp` rel 1.182e-7）、`TRAP-vrsqrt-plain-precision`；④ **外部已知算法族**（Cephes expm1f 有理逼近；Taylor 截断；libm 区间约减）由模型知识补充，属 algorithm-research.md §6 第 5 层「只取算法思路」，API 存在性另由本地 `docs/` + 设备探针佐证；⑤ 本地 `docs/Tilelang.language/` 全目录枚举确认**无原生 `vexpm1`/`expm1` 向量 op**（`数学操作/` 目录仅有 `T.vexp/vln/vlog2/vsqrt/vrsqrt/vpow/vsin/vcos/vtanh/vexp2/vabs/vadd/vmul/vsub/vdiv`；`tilelang/language/customize_npuir.py` 无 `npuir_expm1`）——故必须自建，逼近族即最优可行集。

**v-prefix 元算 dtype 矩阵（决定 bf16 必经 fp32 中转）**：`vexp/vmul/vadd/vsub/vabs` 均仅 `fp16|fp32`（不含 bf16），`vcmp/vselect` 含 `fp16|fp32|bf16`（`docs/Tilelang.language/数学操作/*.md`·`比较操作/T.vcmp.md`·`条件操作/T.vselect.md` §2.2.1）。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)；纯 Vector 逐元素豁免容量项）**：

- 流量项 = `bytes ÷ 有效带宽`，`bytes = 2·N·elem_bytes`（manifest roofline 同口径）。引用 `CONST-mte2-degradation`（`constants.md`，版本戳 tilelang 0.1.2+ed787bb / CANN 8.5.0 / 2026-09-07；fp16 4-set 轮换 1M 33.9 → 16M 30.2 → 64M 20.9 → 256M 18.4 GB/s/core）：**该常数绑定 Ascend910B2C，本设备为 Ascend910_9362，且 `kb_stale_check` 标记 stale，数字仅作量级估算（待重验）**。本设备实查 Vector 核 40（§5.5 ②）。
- 发射项 = `(n_num / core_num) × k × ~0.5µs/op`（`CONST-vector-launch-overhead`，绑定 910B2C=`fixed dispatch` 且 stale，**未文档化假设 + 估算依据**）；`k` = 每 tile 向量 op 数：**fp32 `k=16`**，fp16/bf16 `k=18`（+2 vcast）。
- 容量项 = 0（纯 Vector，无跨引擎往返；`CONST-store-fixpipe-gm-only` 不适用）。

以 `block_size = 2048`、物理 Vector 核 = 40 估算：

| workload | dtype | N | bytes | 流量项 | 发射项 (k) | **估算下界 max** |
|----------|---|------|-------|--------------------|----------------------------|------------------|
| smoke-1m | fp32 | 1,048,576 | 8.4 MB | 6.2 µs | 102.4 µs (k=16) | **≈ 102 µs** |
| smoke-1m | fp16 | 1,048,576 | 4.2 MB | 3.5 µs | 115.2 µs (k=18) | **≈ 115 µs** |
| elementwise-16m | fp32 | 16,777,216 | 134.2 MB | ≈ 111 µs | ≈ 1638 µs (k=16) | **≈ 1638 µs** |
| elementwise-16m | fp16 | 16,777,216 | 67.1 MB | ≈ 56 µs | ≈ 1843 µs (k=18) | **≈ 1843 µs** |
| elementwise-256m | fp16 | 268,435,456 | 1073.7 MB | ≈ 1459 µs | ≈ 29491 µs (k=18) | **≈ 29491 µs** |

**设计期估算（D-2 roofline）**：代表性 `smoke-1m fp32` 估算下界 ≈ **102 µs**（流量 6.2 + 发射 102.4 + 容量 0，发射项主导）；**全部 workload 由发射项主导**（本算子计算链为 16-op 分段链，op 数是发射成本的第一驱动）。

> **估算不确定性声明（未文档化假设 + 估算依据）**：① `k` 与「每 tile op 链跨 tile 重叠程度」均无本仓实测——`CONST-vector-launch-overhead` 的 0.5µs/op 绑定 910B2C 且 stale，auto-multi-buffer 下向量 op 可与 MTE2 重叠（同源 `PL-1.6-copy-floor` / `PL-1.17-subfp32-fp32-transit` 均标记 **stale**，仅作背景，**不作为决策依据**），故上表发射项可能**上偏**；② 对照证据：同族 `bitwise_not`（k=1）实测 256m 1932µs 低于其设计估算 2918µs（实测优于估算 ≈1.5×）；③ 本算子发射项远重于 v0 型 2-op 链（16/2 = 8×），但**低于同族 erf 的 21-op 链**——这是「修抵消」的固有代价，也是本设计的核心精度/性能权衡；④ `block_size` 是摊薄发射的第一杠杆（发射 ∝ 1/block_size，§5.2）。Stage 4 baseline 采集后按 `hardware-cost-model.md` §3 对账并回填常数。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（分段 expm1 + fp32 中转）。以下 5 项均完成「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；采纳项 #1–#4 的等价性经 `verify_equiv.py` 机器验证（结果表见本节末）。

| # | 优化项 | 原式（基线） | 优化后公式（采纳） | 等价性论证 | 收益量化 |
|---|--------|------|-----------|-----------|---------|
| 1 | **小 `\|x\|` 抵消修复（分段 Taylor 替代 `exp−1`）** | `exp(x)−1`（源；小 x 相对误差 19.2%） | `\|x\|<0.5ln2: x·Σ_{n=0}^{5}xⁿ/(n+1)!` | **容差内等价**：Taylor 截断余项 `|R|≤|x|⁷/5040·e^{|x|}≤1.68e-7`，相对误差 `≤|x|⁶/5040·e^T ≤4.86e-7 ≪ rtol 1e-5`；`x→0` 首项即 `x`（相对误差不退化）；设备实测小 x rel ≤1.2e-7；`verify_equiv.py` #1/#2/#3 | 小 x 相对误差 19.2%→7.1e-8（**−6 个数量级**）；代价 = 大分支仍 `vexp−1`，小分支 +14 op |
| 2 | **Horner 嵌套求值**（多项式求值降 op） | 直接展开 `Σ_{n=0}^{5} cₙ xⁿ`：`x²,x³,x⁴,x⁵` 4 mul + 6 系数 mul + 5 add = 15 op | `g=(((((C5·x+C4)·x+C3)·x+C2)·x+C1)·x+C0)` 再 `x·g`：5 mul + 5 add + 1 mul = **11 op** | 数学恒等（Horner 法则，多项式求值的结合律重排） | 15 → 11 op（**−4 op/tile**，约 −27% 多项式段）；`verify_equiv.py` 以 Horner 形式逐 op 复现 |
| 3 | **阈值选路替代分支**（源无此结构） | 分正/负半轴或大/小区间写多条链（易引入重复系数） | 两分支皆算 + `T.vselect(cond,…)` 单链 | 数学恒等（条件函数按 `\|x\|<T` 逐元素选取；SIMD 屏蔽无发散） | 计算路径唯一、无常量重复；`x=0/±∞/NaN` 角点设备实测与 golden 一致 |
| 4 | **fp16/bf16 fp32 中转计算域** | 源语义：fp32 域算 + 单次舍回 | `vcast↑f32(rint) → 分段链(fp32) → vcast↓dtype(rint)` | fp32 域算后单次舍回 ≡ 源语义；fp16↔fp32 / bf16↔fp32 往返无损；与 `torch.expm1(dtype)` golden 门禁内一致（`verify_equiv.py` #2/#3/#4/#5） | 精度对齐 golden（设备实测 fp16/bf16 max_abs 0.0）；代价 +2 vcast；bf16 为 API 强制 |
| 5 | **CSE / 缓冲复用** | 各中间量独立 buffer | `xa` 由 `\|x\|` 复用为小分支结果 `x·g`；`xc` 链尾原地承载结果；系数为编译期常量 | 数学恒等（中间量在生命期结束后复用，不改变数值） | 省 1 个 fp32 UB buffer（§4.5）；`x` 只搬入一次 |

**优化结论**：采纳 5 项。**公式层面已无进一步计算量/访存量可优化空间**——`expm1` 的实现核心即「小 x 消抵消 + 大 x 用原生 exp」，分段 Taylor 已是该约束下的最小 op 闭式；链内 CSE 已完成（`|x|` 复用、`xc` 原地、Horner 求值重排）。**优化后公式**（供 §3.1 使用）：
- `float32`：`y = piecewise_expm1(x)`（16 op 向量链，§3.3）
- `float16 / bfloat16`：`y = vcast(piecewise_expm1(vcast(x, 'float32')), dtype)`（18 op）

**等价性机器验证（D-1）**：脚本 `examples/expm1/expm1_fwd_kernel/verify_equiv.py`（CPU；候选式 = 分段 expm1 逐 op 以 fp32 舍入路径复现设备链，末端按目标 dtype 单次舍回；基线式 = `torch.expm1` 同 dtype golden）。**执行结果表**（脚本实际输出，`python3 verify_equiv.py`，ALL 6 ITEMS EQUIV_PASS）：

| 采纳项 | 候选式 | 基线式 | samples | 最大 abs 差 | 最大 ulp | 违反率(atol+rtol) | 结论 |
|--------|--------|--------|---------|------------|---------|--------|------|
| #1 fp32 | 分段(fp32 arith) | `torch.expm1(x)` (fp32 golden) | 1,048,615 | 3.815e-06 | 5.0000 | 0.000000 | **EQUIV_PASS** |
| #2 fp16 | `round_f16(分段 fp32)` | `torch.expm1(x)` (fp16 golden) | 1,048,615 | 0.000e+00 | 0.0000 | 0.000000 | **EQUIV_PASS** |
| #3 bf16 | `round_bf16(分段 fp32)` | `torch.expm1(x)` (bf16 golden) | 1,048,615 | 0.000e+00 | 0.0000 | 0.000000 | **EQUIV_PASS** |
| #4 fp16 中转无损 | `f16→f32→f16(rint)` | `x`（恒等） | 1,048,615 | 0.000e+00 | 0.0000 | 0.000000 | **EQUIV_PASS** |
| #5 bf16 中转无损 | `bf16→f32→bf16(rint)` | `x`（恒等） | 1,048,615 | 0.000e+00 | 0.0000 | 0.000000 | **EQUIV_PASS** |
| #6 intrinsic 截断界 | 分段(fp64) | `expm1_fp64(x)` | 1,048,615 | 1.245e-07 | n/a | 0.000000 | **EQUIV_PASS** |

> 说明：① 判定准则 = Stage 3 门禁同口径 `|cand−base| ≤ atol + rtol·|base|`（fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2），违反率 0 即 PASS；#6 判据为分段算法的 fp64 实现相对 `expm1_fp64` 的最大绝对误差 <1e-6（实测 1.245e-7，复核 Taylor 截断界 R2 论证 1）。② 分段式非位级恒等，故 **max_ulp 不作判据**（大 |x| 处 `exp(x)` 巨大、ulp 极小无意义；#1 的 max_abs 3.8e-6 出现在大 x（rtol 头部），allowed=1e-5·|base| 远大于 3.8e-6）——正确判据是门禁合规。③ 覆盖随机张量 `randn`（2²⁰）+ 角点（±0/±inf/NaN/次正规/小 x 敏感段 1e-7..1e-3/阈值边界 ±0.3466/溢出 88.8/−88.8）——显式含测试角点 `[0.0,88.8,−88.8,1e-7]`。④ 全部超差计数为 0。

**小 `|x|` 相对精度诊断（非门禁，机器输出）**——证明采纳项修掉抵消：

| x | torch.expm1 | 基线 `exp−1` 相对误差 | 分段（采纳）相对误差 |
|---|-------------|----------------------|----------------------|
| 1e-7 | 1.000000e-07 | 1.921e-01 | 5.000e-08 |
| 1e-6 | 1.000000e-06 | 4.633e-02 | 4.525e-08 |
| 1e-5 | 1.000005e-05 | 1.353e-03 | 2.204e-09 |
| 1e-4 | 1.000050e-04 | 1.160e-04 | 8.859e-08 |
| 1e-3 | 1.000500e-03 | 2.334e-05 | 6.985e-08 |

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 逐元素 `\|x\|` | 源标量 `T.abs`（隐式） | `T.vabs(xc, xa)`（佐证：`docs/Tilelang.language/数学操作/T.vabs.md` §3 `hivm::VAbsOp`；设备探针 P3 实测） | ✅ | — |
| 2 | 阈值判定 `\|x\|<T` | 标量 `if x>T` 分支 | `T.vcmp(xa, thr, cond, "lt")`（佐证：`docs/.../比较操作/T.vcmp.md` §2.3；探针 P3） | ✅ | — |
| 3 | Horner 多项式（6 系数） | 标量循环 | 10 × `T.vmul/T.vadd`（整 buffer）（佐证：`docs/.../T.vmul.md`·`T.vadd.md` §2.3 标量广播；探针 P3） | ✅ | — |
| 4 | 小分支 `x·g(x)` | 标量乘 | `T.vmul(xc, g, xa)`（原地复用 `xa`） | ✅ | — |
| 5 | `e^x` | 源标量 `T.exp` | `T.vexp(xc, e)`（佐证：`docs/.../T.vexp.md` §3 `hivm.hir.vexp`；探针 P1 rel 1.182e-7） | ✅ | — |
| 6 | `−1` | 源标量 `-1.0` | `T.vsub(e, 1.0, e)`（佐证：`docs/.../T.vsub.md` §2.3 标量广播；探针 P2 精确） | ✅ | — |
| 7 | 阈值选路 | 标量条件分支 | `T.vselect(cond, xa, e, xc)`（佐证：`docs/.../条件操作/T.vselect.md` §2.3；探针 P3） | ✅ | — |
| 8 | fp32 中转（fp16/bf16） | 标量 `T.cast` | `T.vcast(src, dst, round_mode="rint")`（佐证：`docs/.../数据类型转换操作/T.vcast.md` §2.2.1；探针 P4） | ✅ | — |
| 9 | 核内网格跨步任务循环 | `for i in T.serial(ceildiv(n_num, core_num))` | 无（核内调度循环，非逐元素计算） | ❌ | tile/block 级任务调度循环；每核迭代数 `ceildiv(n_num,40)`（1M@bs2048 = 13 次），非逐元素热点；向量 op 作用于整 buffer，元素级并行由向量指令完成。先例 `vec_add_2d.py` L32、`PL-1.22` |
| 10 | block 起始索引 / 任务映射 | `task=i*core_num+cid`、`off=task*block_size` | 无 | ❌ | block 级索引/任务映射，每 tile 少量标量，非逐元素热点；无逐元素等价 API |
| 11 | 尾块边界（非整除 N） | `tail = T.min(block_size, shape-off)` | 无 | ❌ | 依赖动态 shape 的边界处理；manifest 全部 workload 的 N 整除 `block_size`，设计保留 `shape: T.int32` 守卫以支持任意 N（纯静态化对通用 N 不成立） |
| 12 | 阈值常量 buffer 初始化 | 每元素标量赋值 | `T.fill(thr, T.cast(T, f32))`（佐证：`bitwise_not_fwd_kernel.py` L104） | ✅ | 循环外一次性填充（per-kernel，非逐元素热点） |

**向量化结论**：逐元素计算点（#1–#8、#12）**已全部向量化**（`T.vabs/vcmp/vmul/vadd/vexp/vsub/vselect/vcast`，均有 docs 佐证 + 设备探针实测）；保留 3 处标量/循环（#9 核内调度循环、#10 block 索引、#11 动态边界），理由类别：tile 级调度循环 + block 索引 + 动态边界，逐项见上表。与 §6 循环结构一致——kernel 内**无逐元素标量循环**。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

**单轴算子豁免**：本算子为 1-D `[N]` 单轴算子（host 侧把任意 shape 展平为一维），不存在「在哪个轴向量化」的轴选择问题——唯一候选向量化轴即该 1D 元素轴本身。按 skill Phase 2 第 3 项「纯 GEMV/单轴算子可写明『仅一个候选轴』豁免」，记录豁免依据：

- **向量化轴** = 唯一 1D 元素轴（stride=1，连续）。
- **轴质量**：manifest 对齐 → fp32 元素 4B，向量宽 ×4 → `block_size=2048` 每 tile 512 向量槽，尾 lane 浪费 0%（2048 整除 4）；fp16/bf16 元素 2B，向量宽 ×8 → 256 槽，浪费 0%；均按 32B 尾轴对齐（2048×elem_bytes 为 32B 整数倍）。无累加链（无串行累加维）。无 repack（I/O 布局 = 核内布局，一维连续，不需 `T.transpose` 或 host permute）。UB 驻留与 §4.5 一致。
- **迁移注**：GPU 源码的并行轴选择（thread `i`×`npt j` → 元素 `idx`）是**输入而非结论**——NPU Vector 轴与之不对应，§0.6 R1 已独立评估为「核内 grid-stride + 向量指令整 buffer」。
- **决策落地**：主选轴 = 元素轴，同步落入 §3.3 伪代码与 §6 循环结构（分段链各 op 直接作用于 UB 一维 buffer；无逐元素标量循环）。

**布局决策结论**：选定方案（唯一候选轴）：核内布局 = 一维连续 `[block_size]`（I/O 布局即核内布局，无 repack），向量化轴 = 唯一 1D 元素轴，repack 路径 = 无。弃选候选：无（单轴算子，无第二轴可枚举，按 skill Phase 2 第 3 项豁免）。
---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer

### 2.2 选型理由

| 算子特征 | 分析 | 结论 |
|----------|------|------|
| 计算类型 | 纯 Vector（逐元素分段 expm1 链），无 matmul | 不需 Cube/L0C/L1 |
| 是否含归约 | 无归约 | 不需 Expert 级 buffer 控制 |
| 是否需流水线 | 单数据通路 GM→UB→GM，无核间协作 | 不需 CV 融合 |
| 分段链 API | 全部 v-prefix（`T.vabs/vcmp/vmul/vadd/vexp/vsub/vselect/vcast`） | 用 v-prefix API |
| 同步需求 | UB 内顺序计算，数据依赖线性 | Developer 自动同步即可 |

决策树路径：`纯 element-wise → 单步逐元素 → 无需精细 buffer 控制 → Developer 模式`。与 `op_requirements.programming_mode=developer` 一致；参考 `examples/elementwise/vec_add_1d.py`（Developer + `T.alloc_ub` + v-prefix）与同族迁移 `examples/bitwise_not/bitwise_not_fwd_kernel/`（`T.fill` + v-op + `T.serial` 网格跨步）。

> **模式敏感性提示**：`T.vtanh`/`T.vexp2` 等的 Developer/Expert 实现不同。本设计仅用 Developer 实测语义（§1.6.0 R4 / §3.4 探针）。

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

> 输入公式为 §1.6.1 的**优化后公式**（分段 expm1 向量链 + fp16/bf16 fp32 中转）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_tile = x[off : off+bs]` | 搬入一维 tile 到 UB |
| 2a | `xf = cast(x_tile)`（仅 fp16/bf16） | fp32 中转（无损升级） |
| 2b | `xa = \|xf\|` | 绝对值（阈值判定） |
| 2c | `cond = (xa < T)` | 小分支掩码（`T=0.5·ln2`） |
| 2d | `g = Horner(C; xf)` | `expm1(x)/x` 的 6 项 Taylor |
| 2e | `ys = xf · g` | 小分支 `expm1` |
| 2f | `e = vexp(xf)`；`e = e − 1` | 大分支 `exp(x)−1` |
| 2g | `yf = select(cond, ys, e)` | 分段选路（fp32 域结果） |
| 2i | `y_tile = cast(yf)`（仅 fp16/bf16） | 单次舍回原 dtype |
| 3 | `y[off : off+bs] = y_tile` | 搬出结果到 GM |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 搬入 | `x_tile = x[...]` | `T.copy` | `T.copy(x[off:off+tail], x_ub[0:tail])` | Developer |
| cast 升 fp32（fp16/bf16） | `xf = cast(x, fp32)` | `T.vcast` | `T.vcast(x_ub, xf, round_mode="rint")` | Developer |
| 绝对值 | `xa=\|xc\|` | `T.vabs` | `T.vabs(xc, xa)`（`xc = xf if need_cast else x_ub`） | Developer |
| 阈值判定 | `cond=(xa<T)` | `T.vcmp` | `T.vcmp(xa, thr, cond, "lt")` | Developer |
| Horner 主体 | `g=(((((C5·x+C4)·x+C3)·x+C2)·x+C1)·x+C0)` | `T.vmul`/`T.vadd` ×10 | 见 §3.3 | Developer |
| 小分支 | `ys=x·g` | `T.vmul` | `T.vmul(xc, g, xa)`（复用 `xa`） | Developer |
| 指数 | `e=vexp(x)` | `T.vexp` | `T.vexp(xc, e)` | Developer |
| 大分支 | `e=e−1` | `T.vsub` | `T.vsub(e, T.cast(1.0,f32), e)` | Developer |
| 选路 | `yf=cond?ys:e` | `T.vselect` | `T.vselect(cond, xa, e, xc)`（复用 `xc`） | Developer |
| cast 舍回（fp16/bf16） | `y_tile=cast(yf,dtype)` | `T.vcast` | `T.vcast(xc, y_ub, round_mode="rint")` | Developer |
| 搬出 | `y[...]=y_tile` | `T.copy` | `T.copy((y_ub if need_cast else x_ub)[0:tail], y[off:off+tail])` | Developer |
| 核内网格跨步 | 任务映射 | `T.Kernel` + `T.serial` | `T.Kernel(core_num, is_npu=True)` + `for i in T.serial(ceildiv(n_num, core_num))` | Developer |
| 阈值常量 buffer | `thr = T` | `T.fill` | `T.fill(thr, T.cast(0.34657359, f32))`（循环外） | Developer |

### 3.3 计算伪代码

> 迁移规则：`@T.prim_func` 参数名 `x, y` 及顺序不变，新增 `shape: T.int32` 用于动态 N 尾块守卫；**v-prefix op 只作用于整 UB buffer，动态 slice 只包在 `T.copy` 里**（前一任务 R-6 实证陷阱；`PL-1.22` 同款）；**常量 buffer 在 tile 循环外一次性填充**（只读，不产生 per-tile op）。

```python
import tilelang
import tilelang.language as T

# 分段 expm1：|x|<0.5*ln2 用 expm1(x)/x 的 6 项 Taylor，否则 vexp(x)-1
_T   = 0.34657359027997264          # 0.5*ln2
_C5  = 1.0 / 720.0                  # 1/6!
_C4  = 1.0 / 120.0                  # 1/5!
_C3  = 1.0 / 24.0                   # 1/4!
_C2  = 1.0 / 6.0                    # 1/3!
_C1  = 1.0 / 2.0                    # 1/2!
_C0  = 1.0                          # 1/1!

@tilelang.jit(out_idx=[1], target="npuir")
def expm1_fwd_kernel(N, dtype, output_dtype=None, block_size=2048):
    out_dtype = output_dtype or dtype
    f32 = "float32"
    need_cast = (dtype != f32)              # 编译期 Python 常量
    n_num = T.ceildiv(N, block_size)        # N 编译期常量 → 静态值
    core_num = 40                           # 物理 Vector 核数（实查 20 aicore × 2）

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), out_dtype),
        shape: T.int32,
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)          # 输入 tile
            y_ub = T.alloc_ub((block_size,), out_dtype)      # 输出 tile（fp16/bf16 舍回缓冲）
            xf = T.alloc_ub((block_size,), f32)              # fp32 计算缓冲（fp16/bf16 中转入口）
            xa = T.alloc_ub((block_size,), f32)              # |x| -> 小分支结果
            g = T.alloc_ub((block_size,), f32)               # Horner 累加器
            e = T.alloc_ub((block_size,), f32)               # exp(x) -> exp(x)-1（大分支）
            thr = T.alloc_ub((block_size,), f32)             # 常量阈值 T（只读）
            cond = T.alloc_ub((block_size,), "bool")

            # 常量 buffer：tile 循环外一次性填充（只读，跨 tile 有效）
            T.fill(thr, T.cast(_T, f32))

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

                    # --- 分段 expm1 向量链（fp32 域，作用于整 buffer）---
                    T.vabs(xc, xa)                           # xa = |x|
                    T.vcmp(xa, thr, cond, "lt")              # cond = |x| < T
                    # g = (((((C5*x + C4)*x + C3)*x + C2)*x + C1)*x + C0)   (Horner)
                    T.vmul(xc, _C5, g)
                    T.vadd(g, T.cast(_C4, f32), g)
                    T.vmul(g, xc, g)
                    T.vadd(g, T.cast(_C3, f32), g)
                    T.vmul(g, xc, g)
                    T.vadd(g, T.cast(_C2, f32), g)
                    T.vmul(g, xc, g)
                    T.vadd(g, T.cast(_C1, f32), g)
                    T.vmul(g, xc, g)
                    T.vadd(g, T.cast(_C0, f32), g)
                    T.vmul(xc, g, xa)                        # xa = x*g（小分支，复用 xa）
                    T.vexp(xc, e)                            # e = exp(x)
                    T.vsub(e, T.cast(1.0, f32), e)           # e = exp(x) - 1（大分支）
                    T.vselect(cond, xa, e, xc)               # xc = |x|<T ? x*g : exp(x)-1

                    if need_cast:                            # 编译期分支：单次舍回
                        T.vcast(xc, y_ub, round_mode="rint")
                        T.copy(y_ub[0:tail], y[off:off + tail])
                    else:
                        T.copy(xc[0:tail], y[off:off + tail])

    return main
```

> **伪代码说明**：① `if need_cast:` 为 Python 编译期条件（JIT 参数编译期确定），不产生运行时分支；**buffer 在 `with T.Kernel` 顶层无条件分配**（避免 `TRAP-conditional-alloc` 的 `Undefined variable`）；② 常量 `thr` 在 `T.serial` **循环外** `T.fill`（只读、跨 tile 有效）；③ v-op 全部作用于**整 buffer**（`PL-1.22` R-6 陷阱），仅 `T.copy` 用动态 slice；④ **`xc` 计算域别名**：fp32 路径 `xc=x_ub`（无中转、零额外 buffer 开销），fp16/bf16 路径 `xc=xf`（`vcast` 升级后计算）；⑤ **`xa` 复用**：先存 `|x|`（供 `vcmp`），`vcmp` 后复用为小分支结果 `x·g`；**`xc` 在链尾复用**为大分支 `exp−1` 与最终结果缓冲（`vselect` 原地）——故无需独立结果 buffer；⑥ `y_ub` 仅用于 fp16/bf16 的单次舍回（fp32 路径直接从 `x_ub` 搬出）。**op 计数 = 16**（fp32：abs1+cmp1+Horner10+mul1+exp1+sub1+select1）/ 18（fp16/bf16，+2 vcast）。

### 3.4 API 可行性确认

> **纪律（前车之鉴 TRAP-vrsqrt-plain-precision / CG-2026-0015）**：NPU 文档声称的元算语义/精度**必须对目标设备实测**。本表全部关键 API 附目标设备（Ascend910_9362）实测（探针脚本 `history_version/design_probe_expm1_ops.py`，2026-10-12 实跑）。

| API | 来源确认（本地） | 设备实测（目标设备） | 状态 |
|-----|-----------------|---------------------|------|
| `T.vexp(src, dst)` | `docs/Tilelang.language/数学操作/T.vexp.md`（§3 `hivm.hir.vexp`）；`customize_npuir.py:295 npuir_exp` | **rel err 1.182e-7**（±10 全域）/ **1.192e-7**（近 0）——真 exp | ✅ 真实可用 |
| `T.vsub(src0, src1, dst)` | `docs/.../数学操作/T.vsub.md`（§2.3 标量 src1；§3） | `vsub(x,1)` **max_abs 0.0**（精确） | ✅ 真实可用 |
| `T.vabs(src, dst)` | `docs/.../数学操作/T.vabs.md`（§3 `hivm::VAbsOp`） | 分段链探针 P3 实跑（角点一致、小 x 精确） | ✅ 真实可用 |
| `T.vmul/vadd`（含标量广播） | `docs/.../数学操作/T.vmul.md`·`T.vadd.md` §2.3（标量 src） | 分段链探针 P3 实跑通过 | ✅ 真实可用 |
| `T.vcmp(A,B,C,mode)` + `T.vselect(Cond,A,B,Out)` | `docs/.../比较操作/T.vcmp.md` §2.3（输出 bool，片上用）；`条件操作/T.vselect.md` §2.3 | 分段链探针 P3 选路实跑通过 | ✅ 真实可用 |
| `T.vcast(src, dst, round_mode)` | `docs/.../数据类型转换操作/T.vcast.md` §2.2.1（f16/bf16↔f32 rint） | 探针 P4（fp16/bf16 中转，max_abs 0.0、0 违例） | ✅ 真实可用 |
| `T.copy`（slice） | `examples/elementwise/vec_add_1d.py` L34 | 探针实跑 | ✅ 已核对 |
| `T.alloc_ub` / `T.fill` | `vec_add_1d.py` L28；`bitwise_not_fwd_kernel.py` L104 | 探针实跑 | ✅ 已核对 |
| `T.Kernel(n, is_npu=True)` + `T.serial` | `vec_add_2d.py` L28–L42 | 探针实跑 | ✅ 已核对 |
| ~~原生 `vexpm1` / `expm1` 元算~~ | `docs/Tilelang.language/` 全目录枚举 + `customize_npuir.py` 无 `npuir_expm1` | — | ❌ **不存在**（须自建，见 §1.6.0 调研范围 ⑤） |

**分段链端到端设备实测**：`fp32 max_abs=1.907e-6 / viol 0`（randn + 角点）；小 x `x=1e-7 → 1.000000e-07`（rel 7.1e-8）`… x=1e-3 → rel 1.16e-7`，与 `torch.expm1` 逐值一致；角点 `[0,-0,1e-7,…,0.3466,0.5,1,3,8,10,88,88.8,-88.8,inf,-inf,nan]` → 与 `torch.expm1` **逐值一致**（含 `88.8 → inf`、`-88.8 → -1.0`、`nan → nan`）；`fp16/bf16 max_abs=0.0`（对 `torch.expm1` bit-exact）。对照基线（P0，源 `exp−1` 直译）小 x 相对误差 19.2%/4.6%/1.4e-3 → **采纳项修掉抵消**。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(core_num, is_npu=True)` |
| GPU 专用 API | Yes（源用标量 `T.exp` + `T.cast` + `T.Parallel` + `threads=`） | 换为 v-prefix 向量链 + `is_npu=True`（§0.5/§0.6） |
| GEMM 非整除 | No | 非 GEMM；非整除由 `tail` + `T.ceildiv` 处理（§5.4） |
| L0C 溢出 | No | 无 Cube 计算 |
| 物理核数限制 | Yes | 逻辑核数 = `ceildiv(N, 2048)` 远超物理核 40 → 核内 `T.serial` 网格跨步（§5.5） |
| **bf16 v-prefix 数学 API 缺失** | Yes | `T.vexp/vadd/vmul/vsub/vabs` 不含 bf16 → bf16 走 fp32 中转（§0.6 R3） |
| **无原生 `vexpm1` 元算** | Yes | 自建分段 expm1（§0.6 R2 / §1.6.0） |
| 条件 `T.alloc` 非法 | Yes（编译期分支易误用） | buffer 全部在 `T.Kernel` 顶层无条件分配（§3.3） |
| bool 不可写回 GM | Yes（`vcmp` 输出 bool） | `cond` 仅片上使用（配 `vselect`），从不 `T.copy` 到 GM（`T.vcmp.md` §2.3） |
| 标量操作数需在 `T.Kernel` 内、`T.Scope` 外定义 | Yes | 系数为 Python 编译期常量传入（`T.vmul.md` §2.3 注 2） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU 源） | 本项目（Ascend npuir） | 转换方案 |
|--------|-------------------|----------------------|----------|
| Kernel 装饰器 | `@tilelang.jit(out_idx=[1])` | `@tilelang.jit(out_idx=[1], target="npuir")` | K1 加 `target="npuir"` |
| Kernel 声明 | `T.Kernel(ceildiv(N,bs), threads=256)` | `T.Kernel(core_num, is_npu=True)` + `T.serial` 网格跨步 | K2/K3 去 `threads=`、加 `is_npu=True`；§0.6 R1 |
| 并行模型 | `T.Parallel(threads, npt)` 线程级 | 一维核 + 向量指令整 buffer | K11：`threads×npt` 折叠为 `block_size` |
| **expm1 计算核** | `T.exp(cast(x,f32)) - 1`（小 x 抵消） | **分段 expm1（16-op 向量链）** | §0.6 R2 / §1.6.1 #1 |
| 计算精度 | `T.cast` 标量升 fp32 | `T.vcast` 向量升/降 fp32 | 保持 fp32 opmath 语义（§0.1） |
| Buffer 层级 | `T.alloc_fragment`（register） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| prim_func 参数 | `main(x, y)` | `main(x, y, shape: T.int32)` | 不改 x/y 顺序与名称，新增 shape 用于尾块守卫 |
| strategy/autotune | `register_copy` + `threads`/`npt` | 无 strategy 维度、无 autotune | K8/K9 移除 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/elementwise/vec_add_1d.py` | 高度相似 | 一维 `T.Kernel`、`T.alloc_ub`、`T.copy` slice、v-prefix 一元 API、`shape` 动态尾块 |
| `examples/elementwise/vec_add_2d.py` | 高度相似 | `T.Kernel(BLOCK_SIZE, is_npu=True)` + `T.serial(ceildiv(...))` + `if block_id < 总任务` 网格跨步（分核先例） |
| `examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py` | 高度相似（迁移形态） | `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步 + `T.fill` 常量 buffer + `shape: T.int32`（结构参照） |
| `examples/erf/erf_fwd_kernel/DESIGN.md` | 高度相似（同族迁移 + 自建逼近） | 无原生元算时自建逼近链、设备探针方法论、fp32 中转、候选表/等价机器验证范式 |
| `examples/TileOPs/tileops/kernels/elementwise/expm1/` | 迁移目标 | 集成 wrapper + `block_size`（K11）+ `main(x, y, shape)` 契约 |
| `docs/Tilelang.language/数学操作/T.vexp.md`·`T.vsub.md`·`T.vabs.md`·`T.vmul.md`·`T.vadd.md` | API | 分段链各 op 规格与 dtype 矩阵 |
| `docs/Tilelang.language/比较操作/T.vcmp.md`·`条件操作/T.vselect.md` | API | 阈值判定的 bool 片上用法 |
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
| `xa` | `(block_size,)` | `float32` | UB | `\|x\|` →（`vcmp` 后）**复用**为小分支结果 `x·g` |
| `g` | `(block_size,)` | `float32` | UB | Horner 累加器 |
| `e` | `(block_size,)` | `float32` | UB | `exp(x)` → `exp(x)−1`（大分支，原地） |
| `thr` | `(block_size,)` | `float32` | UB | 常量阈值 `T=0.5·ln2`（**只读**，循环外 `T.fill`） |
| `cond` | `(block_size,)` | `bool` | UB | `\|x| < T` 掩码（片上，从不写回 GM） |

> **buffer 分配策略**：全部在 `with T.Kernel` 顶层**无条件分配**（避免 `TRAP-conditional-alloc`）；阈值常量循环外填充；**`xa` 复用**（`|x|`→小分支结果）、**`xc` 计算域别名**（fp32→`x_ub` / fp16/bf16→`xf`）并在链尾原地承载结果（§3.3）。

### 4.4 内存搬运路径

```
fp32 路径（纯 Vector，无 L1/L0；xc = x_ub）：
GM[x] --T.copy--> UB[x_ub] --[分段 expm1 链 16 op（原地复用 x_ub 承载结果）]--> UB[x_ub] --T.copy--> GM[y]

fp16/bf16 路径（fp32 中转；xc = xf）：
GM[x(dtype)] --T.copy--> UB[x_ub(dtype)] --T.vcast(rint)--> UB[xf(fp32)]
     --[分段 expm1 链 16 op（原地复用 xf 承载结果）]--> UB[xf(fp32)] --T.vcast(rint)--> UB[y_ub(dtype)] --T.copy--> GM[y]
```

### 4.5 UB 内存预算

`block_size = 2048`：

| 路径 | x_ub | y_ub | xf | xa | g | e | thr | cond | 合计 | ×1.12（`CONST-capacity-910B2C` 上偏系数，stale） |
|------|------|------|----|----|---|----|-----|------|------|------------------------------------------------|
| fp32 | 8 KB | 8 KB | 8 KB | 8 KB | 8 KB | 8 KB | 8 KB | 2 KB | **58 KB** | ≈ 65 KB < 192 KB ✓ |
| fp16 | 4 KB | 4 KB | 8 KB | 8 KB | 8 KB | 8 KB | 8 KB | 2 KB | **50 KB** | ≈ 56 KB ✓ |
| bf16 | 4 KB | 4 KB | 8 KB | 8 KB | 8 KB | 8 KB | 8 KB | 2 KB | **50 KB** | ≈ 56 KB ✓ |

> UB 硬上限 = 192 KB/AIV（`CONST-capacity-910B2C`，绑定 Ascend910B2C **且 stale**；本设备 Ascend910_9362 同规格量级）。**膨胀系数注**：`TRAP-UB-multibuffer-inflation` 记 auto-multi-buffer 可能膨胀 ~1.7×；本设计单 buffer 最大路径 58 KB，×1.7 ≈ 99 KB 仍 <192 KB（**安全**）。**block_size 放大空间**：`bs=4096` → fp32 116 KB（×1.7 ≈ 197 KB，**贴限**，须编译探针校准）；`bs=3072` → fp32 87 KB（×1.7 ≈ 148 KB，可行）。故默认 `block_size=2048`（保守、与源 fp16/bf16 的 K11 折叠值一致）；`bs=3072/4096` 列为 Stage 4 发射项调优杠杆（§5.2）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| `N` 的实际长度 | `shape: T.int32`（prim_func 参数） | 1 ~ manifest 最大 268,435,456 |

> `N` 同时作为编译期 `T.Tensor((N,), dtype)` 的 shape（按特定 N 特化编译）与运行时 `shape` 参数（尾块守卫）。manifest 全部 workload 的 N 均整除 `block_size=2048`，但保留 `tail` 守卫以支持任意 N。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[1],        # y 是 prim_func 第 2 个参数（index=1）
    target="npuir",
)
def expm1_fwd_kernel(N, dtype, output_dtype=None, block_size=2048):
    ...
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅含逐元素向量链（`vabs/vcmp/vmul/vadd/vexp/vsub/vselect/vcast`），无 matmul、无归约、无 Cube 计算；仅需 UB 层级，不涉及 L1/L0A/L0B/L0C。

### 5.2 Block 划分

```python
block_size = 2048   # K11: 源 threads(256) × npt(8)（fp16/bf16）折叠为单一参数
n_num = T.ceildiv(N, block_size)   # 逻辑 tile 数（编译期静态值）
```

**block_size=2048 选择理由**：
1. 与源 fp16/bf16 一致（K11 折叠 `threads × npt = 256 × 8 = 2048`），迁移可追溯；fp32 源为 1024，本设计统一取 2048（v-prefix 整 buffer 形态下 fp32 无 128-bit 对齐约束差异）；
2. UB 对齐：2048×elem_bytes 是 32B 的整数倍（fp32 8192B、fp16/bf16 4096B）✓；
3. UB 容量：最大 fp32 路径 58 KB（§4.5），×1.7 最坏膨胀 ≈ 99 KB < 192 KB ✓；
4. **发射项权衡**：本算子 §1.6.0 估算显示**发射项绝对主导**（发射 ∝ 1/block_size），但 bs 放大受 UB 膨胀系数不确定性约束。默认取 2048（保守）；**bs=3072/4096 列为 Stage 4 调优首选杠杆**（须以编译探针校准 UB 后采纳，分核参数调优，见 `core-split-strategy.md` §2.4）。

### 5.3 约束分析

- **对齐约束**：`block_size=2048`，fp32 元素 4B → 每 tile 512 向量槽（128bit×4），尾 lane 浪费 0%；fp16/bf16 元素 2B → 256 向量槽（128bit×8），浪费 0%；均满足 32B 尾轴对齐 ✓。
- **UB 容量**：最大 99 KB（fp32，含 1.7× 最坏膨胀）< 192 KB ✓。
- **L0/L1 容量**：不适用（纯 Vector）。
- **分形限制**：不适用（无 Cube）。

### 5.4 注意事项（非整除处理策略）

**非整除场景预判**：manifest 全部 workload 的 N（1048576 / 16,777,216 / 67,108,864 / 268,435,456）均整除 `block_size=2048`（尾块不存在）；测试 shape 4,000,000 **不整除 2048**（余 1216）→ 存在尾块。

**策略**：Kernel 内动态尾块（不改动 host 输入张量、不做 host padding，遵守 `ascend-constraints.md` §4）：最后一个 tile 用 `tail = T.min(block_size, shape - off)` 裁剪 `T.copy` 的搬入/搬出量；向量 op 仍作用于**整 buffer**（尾部未初始化数据不影响仅搬出的 `[0:tail]`）。参考 `examples/elementwise/vec_add_1d.py`。任务网格用 `T.ceildiv(N, block_size)` 保证覆盖最后一个不完整 tile。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

**① 逻辑核数计算**：本算子为 1-D 输出网格 `[N]`，仅 N 轴（无 M 轴，M 项取 1）：

```
num_logical_kernels = ceil(N / block_size) = ceildiv(N, 2048)
```

| workload | N | 逻辑核数 `ceildiv(N,2048)` |
|----------|---|---------------------------|
| smoke-1m | 1,048,576 | 512 |
| elementwise-16m | 16,777,216 | 8,192 |
| elementwise-64m | 67,108,864 | 32,768 |
| elementwise-256m | 268,435,456 | 131,072 |

**② 物理核数依据（实查）**：目标设备 `Ascend910_9362`，实查代码与返回值（纯 Vector 算子核数翻倍）：

```python
from tilelang.utils.npu_utils import NPUUtils
NPUUtils.get().get_aicore_num()        # 实际返回 20
NPUUtils.get().get_aivector_core_num() # 实际返回 40（= aicore × 2）
```

→ **纯 Vector 物理核数 = 20 × 2 = 40**（2026-10-12 本会话实查，与 `CONST-aicore-910_9362` 一致）。注：`pattern-library/constants.md` 的 `CONST-aicore-910B2C` 记 24 核，绑定 Ascend910B2C，**以实查为准**。

**③ 规模判定与分核方案**：**极大规模**——逻辑核数（512 ~ 131,072）远超物理核数 40，且 N 为 2 的幂、物理核数 40 = 2³×5 含非 2 因子，数学上**无法**通过调整 block_size（2 的幂）使逻辑核数对齐 40 的整数倍（5 ∤ 2ⁿ）。按标准 §1 要素③「极大规模」处置：

- **固定启动内核数 = 物理核数 40**；
- 核内 `T.serial` 网格跨步处理多个逻辑 tile：
  ```python
  for i in T.serial(T.ceildiv(n_num, core_num)):   # 循环上界静态（n_num、core_num 均编译期常量）
      task = i * core_num + cid                    # 网格跨步（grid-stride）
      if task < n_num:                             # 守卫
          ...
  ```
- **循环边界静态** ✓（`n_num`、`core_num` 均编译期常量，`T.ceildiv(n_num, core_num)` 为静态整数）；
- **残留负载不均**（显式记录）：以 1M@bs2048 为例，`tasks_per_core = ceildiv(512,40) = 13`；网格跨步下前 `512 - 40×12 = 32` 个核执行 13 次、后 8 个核执行 12 次 → 临界路径 = 13/12.8 ≈ **+1.6%**。相较「直接 `T.Kernel(n_num=512)` 由运行时串行调度 12.8 波次」，核内 `T.serial` 把核启动固定开销从「每 tile 一次」摊薄为「每核一次」，净收益为正。大 N 档逻辑核数更大，残留不均占比进一步下降（16m：`ceildiv(8192,40)=205`，临界 +0.24%；256m：+0.02%）。
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
| 元素级 | 向量化 | `T.vabs/vcmp/vmul/vadd/vexp/vsub/vselect/vcast`（16 op） | 整 buffer 一次完成，无 `T.Parallel` 标量循环 |

### 6.2 循环伪代码

见 §3.3（`with T.Kernel(core_num, ...)` + `for i in T.serial(...)` + `if task < n_num`，循环内为 16-op 分段向量链）。

### 6.3 流水线优化

**当前设计**: 不使用 `T.Pipelined`。理由：单 tile 内数据依赖线性（copy-in → vcast → 16-op 链 → vcast → copy-out），无 K 维迭代；跨 tile 迭代（`T.serial`）的 MTE2/VEC 重叠由 Developer 模式 auto-multi-buffer 自动处理。**发射项主导下**，若 Stage 4 需进一步优化（persistent double buffer / 更大 block_size），参照 `example_elementwise_exp2.py` 的 `T.serial` 多 tile 形态；设计阶段保持简洁。

### 6.4 尾块处理

见 §5.4。`N % block_size != 0` 时最后一个 tile 用 `tail = T.min(block_size, shape - off)` 动态裁剪 `T.copy` 量；manifest 全部 workload 整除（§5.4 表），仅测试 4,000,000 触发尾块（余 1216）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下编译器自动插入同步。核内数据依赖链严格线性（每 tile 内）：

```
T.copy(GM→UB) → [vcast↑] → vabs → vcmp → Horner(10) → vmul → vexp → vsub → vselect → [vcast↓] → T.copy(UB→GM)
```

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy(GM→UB)` 后 | 自动 | 编译器插 MTE2→V 等待 |
| 各 v-op 之间（16 op 链） | 自动 | 线性数据依赖，插 V→V barrier |
| `T.copy(UB→GM)` 前 | 自动 | 编译器插 V→MTE3 等待 |
| tile 迭代之间（`T.serial`） | 自动 | 同 buffer 复用由 auto-multi-buffer 管理跨迭代依赖 |

无手动 `T.sync_block_set/wait`（无跨引擎/跨核协作，无 CV 融合）。**CV 融合设计**：n/a（纯 Vector 算子，无 Cube 段）。

### 7.3 pass_configs 配置

```python
# Developer 模式默认配置，无需额外 pass_configs
# os.environ["TILELANG_ASCEND_MODE"] = "Developer"  # 可选，默认即 Developer
```

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（`torch.expm1`，保持源 fp32 中转语义），**不得复刻 §0.6 的 NPU 分段算法**——保证验证独立性。

```python
import torch

def golden_expm1(x: torch.Tensor) -> torch.Tensor:
    """expm1 参考实现，等价于 torch.expm1（保持源 fp32 opmath 语义）。

    源算子 op_func = T.exp(T.cast(x, "float32")) - 1.0：对 fp16/bf16 输入先升
    fp32 计算，末端单次舍回原 dtype；fp32 输入直接计算。torch.expm1 自身即
    原生精确 expm1（小 |x| 相对精确），故 golden 直接采用 torch.expm1。
    """
    x_f32 = x.to(torch.float32)
    return torch.expm1(x_f32).to(x.dtype)
    # 与 torch.expm1(x) 在 fp16/bf16 下 bit-exact（torch.expm1 自身亦 fp32 opmath）
```

> 备注：golden **独立于** NPU 分段算法（不复用 Taylor 系数，直接用 libm 级 `torch.expm1`），满足验证独立性。**注意**：源 GPU `op_func` 是 `exp−1` 直译，其数值输出在小 `|x|` 处与 golden 有 ~19% 相对差（但绝对差 <atol）——因此 golden 取契约语义 `torch.expm1`，而非源直译的数值。

### 8.2 精度标准

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 1e-5 | 1e-5 | 来源：`examples/TileOPs/tests/ops/test_unary_math.py` `_get_tolerances` |
| float16 | 1e-3 | 1e-3 | 来源：同上；NPU 在 fp32 域计算、单次舍回 |
| bfloat16 | 1.6e-2 | 1.6e-2 | 来源：同上；bf16 精度低，容忍度放宽 |

**误差预算核对（设计期实测）**：分段算法相对误差：小 `|x|` ≤4.9e-7（Taylor 截断界，设备实测 ≤1.2e-7）、大 `|x|` ≤4.0e-7（抵消放大界，设备实测含 `vexp` rel 1.18e-7）→ 设备端到端 fp32 `max_abs=1.9e-6`、`viol 0`（**≪ atol 1e-5，≥5× 余量**）；fp16/bf16 实测 `max_abs=0.0`（bit-exact）。

### 8.3 L0 门槛测试计划

> L0 = 门槛 smoke 测试（最小可用性验证，进 Stage 3 前必过）；完整分层套件 L1/L2/Boundary 交由 `tilelang-op-develop`，不在此枚举。**显式包含小 x 抵消用例（L0-5/L0-6）**，确保不重蹈源直译的抵消缺陷。

| 用例 | 输入 | 期望 | 判定 |
|------|------|------|------|
| L0-1 fp32 smoke（randn 全域） | `randn([1048576], fp32)`（NPU） | 与 `torch.expm1` 比对 ≤1e-5 | 门槛通过 |
| L0-2 fp16 smoke（randn 全域） | `randn([1048576], fp16)`（NPU） | 比对 ≤1e-3 | 门槛通过 |
| L0-3 bf16 smoke（randn 全域） | `randn([1048576], bf16)`（NPU） | 比对 ≤1.6e-2 | 门槛通过 |
| **L0-4 fp32 角点（测试 `[0.0,88.8,-88.8,1e-7]`）** | `repeat([0.0,88.8,-88.8,1e-7],[1024], fp32)`（NPU） | 与 `torch.expm1` 比对 ≤1e-5（`88.8→inf`、`-88.8→-1.0`、`1e-7→1e-7`；**设计期设备实测逐值一致**） | 门槛通过 |
| **L0-5 小 x 抵消敏感段（核心精度用例）** | `linspace(-1e-3, 1e-3, 1<<20, fp32)`（NPU） | 与 `torch.expm1` 比对，且**相对误差** ≤1e-5（naive 直译此处 19% 失败；**设计期设备实测 rel ≤1.16e-7**） | 门槛通过 |
| L0-6 全域敏感区 | `linspace(-9,9,1<<20, fp32)`（NPU） | 与 `torch.expm1` 比对 ≤1e-5（含阈值边界 ±0.3466） | 门槛通过 |
| L0-7 非整除尾块 | `randn([4000000], fp16)`（NPU） | 比对 ≤1e-3（覆盖 4000000 % 2048 = 1216 尾块） | 门槛通过 |

- **执行方式**：`tilelang.compile(func, target="npuir")` 后单次调用，`torch.testing.assert_close(y.cpu(), golden, atol, rtol)`（golden = §8.1）。
- **门槛标准**：L0-1..L0-7 全过即进入 Stage 3；任一失败不得继续。
- 输入生成参考 `examples/TileOPs/tileops/workloads/elementwise.py`；边界用例参考 `test_unary_math.py` 的 `MathEdgeFixture`（`repeat([0.0, 88.8, -88.8, 1e-7])`）。
- **额外设计期落点**：`history_version/design_probe_expm1_ops.py`（设备探针：native naive 抵消 + vexp/vsub 语义 + 分段端到端 + fp16/bf16）、`verify_equiv.py`（D-1 机器验证，6/6 `EQUIV_PASS`）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 缓解方案 |
|------|------|----------|
| **R-1 无原生 `vexpm1` 元算** | 小 x 抵消须自建分段（16 op） | 分段 Taylor（§0.6 R2，设备实测达标）；**已登记能力缺口**（见 §9.4） |
| **R-2 小 x 抵消（本算子核心）** | 源直译 `exp−1` 小 x 相对误差 19%，不满足 `torch.expm1` 语义 | 分段 Taylor 修抵消（设备实测 rel ≤1.2e-7，20× 余量）；`verify_equiv.py` 6/6 PASS；L0-5 显式覆盖敏感段 |
| **R-3 大 x 抵消放大** | `|x|` 跨过阈值处抵消放大因子 ≤3.414 | 阈值取 `0.5·ln2`（`exp=√2`），放大因子上界 3.414；大分支相对误差 ≤4.0e-7 ≪ rtol |
| **R-4 发射项主导**（16 op/tile，`CONST-vector-launch-overhead`） | §1.6.0 估算显示 emission 远超 flow（102µs vs 6.2µs @smoke-1m） | 标注「未文档化假设 + 估算依据」（常数绑 910B2C、fixed dispatch、stale、可能上偏）；block_size 为杠杆（§5.2）；Stage 4 baseline 后按 `hardware-cost-model.md` §3 对账 |
| **R-5 UB 容量**（分段链 6 个 fp32 工作/常量 buffer，@bs2048 fp32 58 KB） | bs 放大受限（bs=4096 ×1.7 ≈ 197 KB 贴限） | 默认 bs=2048（58 KB×1.7≈99 KB 安全）；buffer 复用（`xa`→小分支、`xc` 原地）；bs=3072/4096 须编译探针校准 |
| **R-6 bf16 v-prefix 数学 API 缺失**（各 op doc §2.2.1） | bf16 直传编译失败 | bf16 走 fp32 中转（§0.6 R3） |
| **R-7 条件 `T.alloc` 非法**（`TRAP-conditional-alloc`） | 编译期 `if` 内 `T.alloc_ub` trace 失败 | buffer 全部在 `T.Kernel` 顶层无条件分配（§3.3） |
| **R-8 v-op 整 buffer vs 动态 slice**（`PL-1.22` R-6） | 传动态 slice 报错 | v-op 作用于整 buffer，仅 `T.copy` 用 `[0:tail]`（§3.3） |
| **R-9 vcast round_mode 限制**（`T.vcast.md` §2.2.1） | f16→f32 / bf16→f32 **仅 `rint`** | 中转升降统一 `round_mode="rint"`（与 golden 单次舍回一致） |
| **R-10 `vcmp` bool 不可写回 GM**（`T.vcmp.md` §2.3） | 误写回编译失败 | `cond` 仅片上配 `vselect`，从不 `T.copy` 到 GM |
| **R-11 分核残留不均** | 40 核含非 2 因子 5，2ⁿ 逻辑核无法整除 | §5.5 网格跨步核内串行，残留临界 +1.6%（1M）/ +0.24%（16m），已显式记录 |
| **R-12 常数表设备/版本** | `CONST-mte2-degradation`/`CONST-vector-launch-overhead`/`CONST-capacity-910B2C` 绑 910B2C 且 `kb_stale_check` 标记 **stale**；本设备 910_9362 | 带宽/发射/容量数字标注「待重验」，仅作量级估算；不以 stale 条目为决策依据（**本设计的关键精度结论来自本会话设备探针，不依赖 stale 条目**）；Stage 4 baseline 后对账 |
| **R-13 元算设备语义**（承诺与实测一致性） | 文档语义可能与设备不一致（`TRAP-vrsqrt-plain-precision`：`T.vrsqrt` plain 近似 2.87e-3） | 本设计关键 API `vexp/vsub/vabs/vmul/vadd/vcmp/vselect/vcast` 全部经**目标设备探针实测**（§3.4）：`vexp` 真 exp rel 1.18e-7、`vsub` 精确、分段端到端 0 违例；后续工具链升级须重验 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 直译 `T.vexp(x) − 1` | 沿用源 `op_func` | 小 \|x\| 相对误差 19%（违反契约语义） | 禁用；改分段 expm1（R-1/R-2） |
| bf16 直接传入数学 op | 未做 vcast 上转 | 编译失败 | bf16 先 `vcast` 到 fp32（R-6） |
| vcast 用非 rint 模式升级 | f16/bf16→f32 用 `round`/`floor` | 编译失败（仅 `rint`） | 升级/舍回统一 `round_mode="rint"`（R-9） |
| bool `cond` 写回 GM | `T.copy(cond_ub, ...)` | 编译失败（i1 不可 load/store GM） | `cond` 仅片上（R-10） |
| 条件 alloc | 编译期 `if need_cast:` 内 `T.alloc_ub` | trace 报 `Undefined variable` | buffer 顶层无条件分配（R-7） |
| v-op 传动态 slice | `T.vabs(x_ub[0:tail], ...)` | 可能编译/运行错误 | v-op 用整 buffer（R-8） |
| UB 溢出 | block_size 过大 | 编译失败 | 维持 bs ≤ 2048（默认）；§4.5 预算 + 编译探针（R-5） |
| 尾块越界 | N 非 bs 整数倍且无 tail 裁剪 | 越界读/写 GM | `tail = T.min(block_size, shape-off)`（§5.4） |
| `xa` 复用冲突 | 在 `vcmp` 尚未用完 `xa` 时写入小分支结果 | 结果错误 | 仅在 `vcmp` 完成后将 `xa` 复用为 `x·g`（§3.3 顺序） |
| 阈值 buffer 未填充 | 直接用未初始化的 `thr` | 选路错误 | 循环外 `T.fill(thr, T)`（§3.3） |
| 零输入崩溃 | N=0 空张量 | MTE DDR 崩溃（`TRAP-zero-input-crash`） | host 侧空张量守卫；本任务 N ≥ 1e6，不触发 |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| **非整除 N** | Kernel 内 `tail` 动态裁剪（§5.4），无需 host padding |
| **极小 N（N < block_size）** | `n_num=1`，`tail=N`，仅 `task==0` 的核工作，其余核空转 |
| **极大 N（>2.6e8）** | 核内 `T.serial` 网格跨步（§5.5），逻辑核数静态 |
| **`±inf/NaN` 输入** | 分段链：`+inf → vexp−1 = inf`、`−inf → 0−1 = −1`、`NaN → cond=false → vexp(NaN)−1 = NaN`（§0.6 R2 论证 4；设备探针 P3 角点逐值一致） |
| **大 `\|x\|` 溢出** | `x ≳ 88.72` 时 `vexp` 溢出为 `inf`，结果 `inf`，与 `torch.expm1` 一致（设备实测 `x=88.8 → inf`） |
| **`x` 近 0** | 走小分支 `x·g(x)`，首项即 `x`，相对误差不退化（设备实测 `x=1e-7 → 1.000000e-07`，rel 7.1e-8） |
| **阈值边界 `\|x\|=0.5·ln2`** | 两分支解析值相等，误差分别 ≤4.9e-7 / ≤4.0e-7，无跳变（verify_equiv 角点含 ±T） |
| **混合 dtype** | 按 dtype 编译期特化（每 dtype 一份 kernel），输出 `same_as(input)` |

### 9.4 能力缺口登记（Compiler Capability Gap Rule）

- **缺口**：NPU npuir 无原生 `expm1` 向量元算（`docs/Tilelang.language/数学操作/` 无 `T.vexpm1`；`tilelang/language/customize_npuir.py` 无 `npuir_expm1`）。
- **影响层**：Frontend API（缺 `vexpm1` 元算）。
- **本任务处置**：自建分段 expm1（16-op v-prefix 链），精度达标但 op 数高于原生元算（发射成本，§1.6.0 估算）。
- **证据**：本地 `docs/` 全目录枚举 + 代码 registry 枚举（§1.6.0 调研范围 ⑤）；`vexp` 可用性由设备探针 P1 证明。
- **建议**（若后续出现 ≥2 个独立任务）→ 可向编译器层提 capability-fill 提案（新增 `hivm.hir.vexpm1`），并登记 `.agents/evolution/capability-gaps.md`（仅登记，本设计不改编译器）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/expm1/expm1_fwd_kernel/
├── DESIGN.md                       # 本设计文档
├── verify_equiv.py                 # §1.6.1 等价性机器验证脚本（D-1，6/6 EQUIV_PASS）
├── RETROSPECTIVE.md                # Stage 1 复盘（自进化钩子）
├── history_version/
│   └── design_probe_expm1_ops.py   # 设备探针（naive 抵消 + vexp/vsub 语义 + 分段端到端 + fp16/bf16）
└── expm1_fwd_kernel.py             # 算子实现（Stage 3 产出）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成 | 6 项等价性机器验证全部 EQUIV_PASS（§1.6.1） |
| `history_version/design_probe_expm1_ops.py` | ✅ 已完成 | 设备探针（2026-10-12 实跑） |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘 |
| `expm1_fwd_kernel.py` | ⬜ 待实现 | 算子实现（Stage 3） |
| `test_expm1.py` | ⬜ 待实现 | 分层测试（Stage 3） |

### 10.3 命名规范

- 项目目录名: `expm1`；算子目录名: `expm1_fwd_kernel`；实现文件: `expm1_fwd_kernel.py`（保留源工厂函数名 `expm1_fwd_kernel`）；测试文件: `test_expm1.py`。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）
2. ✅ 设备探针（`design_probe_expm1_ops.py`，naive 抵消 + 分段算法可用性实测）
3. ✅ 等价性验证（`verify_equiv.py`，6/6 EQUIV_PASS）
4. ⬜ Golden 函数（`golden_expm1`，验证基准）
5. ⬜ 算子实现（`expm1_fwd_kernel.py`：kernel + wrapper + L0 测试）
6. ⬜ L0 门槛测试（L0-1..L0-7：randn 全域 3 档 + 角点 + 小 x 敏感段 + 阈值边界 + 非整除尾块）

### 10.5 迁移执行规则遵循确认

| 规则 | 遵循情况 |
|------|----------|
| `@tilelang.jit` 装饰函数命名为 `expm1_fwd_kernel` | ✅ 保留源工厂函数名 |
| Kernel 参数语义保持（`threads`/`npt` 折叠为 `block_size`） | ✅ K11：256 × 8 = 2048 |
| `@T.prim_func` 参数名 `x, y` 及顺序不变 | ✅ `main(x, y, shape)`，新增 `shape` 用于尾块 |
| 覆盖全部 dtype（fp16/bf16/fp32） | ✅ fp32 直算；fp16/bf16 fp32 中转（bf16 API 强制中转） |
| 保持源 fp32 中转语义 | ✅ `vcast↑f32 → 分段链 → vcast↓dtype`；`verify_equiv.py` 门禁内一致 |
| K1–K4 | ✅ `target="npuir"`、`is_npu=True`、移除 `threads=` |
| K8/K9 | ✅ 移除 autotune / `threads` |
