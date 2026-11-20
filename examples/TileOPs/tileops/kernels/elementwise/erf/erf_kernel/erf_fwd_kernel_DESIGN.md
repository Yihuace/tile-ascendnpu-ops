# erf_fwd_kernel 算子设计文档

> **修订版本（revision 1）**：本版针对 Stage 3 `[DESIGN_ERROR]`（DESIGN v0 采纳的核心元算 `T.verf` 在目标设备上不是真 erf，而是 7 阶 Maclaurin 截断多项式，详见 `repro/TRAP-verf-maclaurin-truncation.py` 与能力缺口 `CG-2026-0015`）**重做核心算法**：以设备实测可用的 v-prefix 元算自建 A&S 7.1.26 有理-指数逼近，并对全部关键 API 做了**目标设备实测探针**（`history_version/design_probe_erf_ops.py` / `design_probe_erf_ops2.py`）。相对上一版的关键调整见 §10.6。

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。

**源算子路径（源真相）**：`/home/developer/TileOPs/tileops/kernels/elementwise.py`
- 具体算子：`ErfFwdKernel`（L2118–L2127，`FloatUnaryKernel` 子类）；
- 基类与模板：`FloatUnaryKernel`（L1297–L1300，`SUPPORTED_DTYPES=_FLOAT_DTYPES`）、`UnaryKernel`（L680–L887，`DEFAULT_STRATEGY="register_copy"` L700）；
- 策略工厂：`_make_unary_regcopy`（L344–L363，fragment load → compute → fragment store）；
- 提取产物（脚手架）：`examples/TileOPs/tileops/kernels/elementwise/erf/_erf_fwd_kernels.py`；集成包装：同目录 `erf.py`。

### 0.1 源算子语义（做什么）

**数学语义**：`y = erf(x)`，逐元素误差函数：

$$
\operatorname{erf}(x)=\frac{2}{\sqrt{\pi}}\int_0^x e^{-t^2}\,dt,\qquad y=\operatorname{erf}(x)\in(-1,\,1)
$$

`ref_api = torch.erf`（manifest `ErfFwdOp`，`examples/TileOPs/tileops/manifest/elementwise.yaml` L140–L176）。源码 `op_func`（L2125–L2127）：

```python
@staticmethod
def op_func(x):
    return T.erf(T.cast(x, "float32"))   # 强制先升 fp32 再算 erf
```

**dtype 语义（迁移必须保持的关键点）**：源实现**强制 fp32 中转**——`T.erf` 的输入恒为 `float32`。源码注释（L2119–L2123）：*"Casts to fp32 before calling `T.erf` because the half-precision intrinsic `herf` is not a valid CUDA built-in."* 即：
- 对 fp16/bf16 输入：`x → cast fp32 → erf(fp32) → 隐式舍回原 dtype`（`y_reg` dtype = 输入 dtype，写回时一次性舍入）；
- 对 fp32 输入：`cast` 为恒等，直接 `erf(fp32)`。

该 fp32 中转恰好与 torch CPU golden 的 **fp32 opmath + 一次舍回** 语义一致。这是本迁移**不得丢失**的语义。

**规约语义**：无规约（逐元素映射，无跨元素依赖），不存在累加顺序问题。

**边界语义**：`erf(0)=0`、`erf(±∞)=±1`、`erf(NaN)=NaN`；`erf(-x)=-erf(x)`（奇函数）；大 `|x|` 下 `erf(x)→sign(x)`（fp32 下 `|x|≳3.9` 时已舍入为 ±1）。空维度/非整除 shape 由 host 侧 flatten 成 1-D `[N]` 后进入 kernel。fp16/bf16 输入经 fp32 中转，`±inf/NaN` 的 IEEE 传播与舍入路径无关。

**host 侧语义**：`UnaryKernel.__init__` 把任意 shape 的输入展平为 `N_total = numel`；`forward` 仅调用编译后 kernel，无 host 预处理/后处理。

**语义保持基线**：§8.1 golden 以本节语义（`torch.erf`，保持源 fp32 中转语义）为唯一依据实现，**不得复刻 §0.6 的 NPU 算法**。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `(N,)`（原始 shape 被 host 侧展平；manifest 有 `[1048576]` / `[4096,4096]` / `[8192,8192]` / `[16384,16384]`） | `float16 \| bfloat16 \| float32` | 1-D 展平输入向量，`N = product(input.shape)` |
| `y` | `same_as_input`（输出 shape = 输入 shape，展平后 `(N,)`） | `same_as(input)` | 逐元素 `erf` 输出 |

**输出 shape（迁移必需）**：`same_as_input`（manifest `shape_rules: output.shape == input.shape`）。
**layout**：1-D 连续（contiguous flatten），无转置。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（`register_copy` 策略，`_make_unary_regcopy`，L344–L363）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `x_reg ← x[bx*bs:(bx+1)*bs]`（fragment 载入） | GM `x` | fragment `x_reg` | 元素定位/搬入 |
| 2 | `y_reg[i] = T.erf(T.cast(x_reg[i],'float32'))`（逐线程 `i,j` 标量映射） | `x_reg[i]` | `y_reg[i]` | `y = erf(x)`（fp32 域） |
| 3 | `y[bx*bs:(bx+1)*bs] ← y_reg`（fragment 写回） | `y_reg` | GM `y` | 搬出 |

**数据流与内存访问模式**（源硬件 = GPU）：

```
GM[x] --(fragment load, 128-bit)--> registers[x_reg]
   --(T.Parallel(threads,npt) 标量 T.erf(T.cast(x,'f32')))--> registers[y_reg]
   --(fragment store)--> GM[y]
（无显式 SMEM；fragment 双缓冲由编译器管理；block_size = threads × npt）
```

**循环与并行结构**：`T.Kernel(T.ceildiv(N, block_size), threads=256)`——一维 grid（`bx`）+ 256 线程/block 两级并行；每线程 `npt` 元素，`T.Parallel(threads_arg, npt_arg)`。`block_size = 256 × npt`：fp32 `npt=4`→1024，fp16/bf16 `npt=8`→2048。

**host 侧逻辑**：`N_total`、`dtype` 由 Op 层传入；`forward` 直接调用编译后 kernel；flatten 由 Op 层完成。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | `register_copy` 策略（fragment load→compute→store） | 向量化 128-bit 访存 | `T.alloc_fragment` + `T.copy` 切片 + `T.Parallel` 标量映射 | GPU 大寄存器堆 + memory coalescing | **硬件强相关**（fragment/register 模型） |
| 2 | block×thread 两级并行（大 grid + 256 线程/block） | 用大规模线程/grid 吃满内存带宽 | `T.Kernel(ceildiv(N,bs), threads=256)` + `T.Parallel` | GPU thread/warp 执行模型 | **硬件强相关**（GPU 线程模型） |
| 3 | **fp32 中转**（`T.cast(x,'float32')` → `T.erf`） | ① 规避 CUDA 无 `herf` 内建（codegen 限制）；② 以 fp32 域计算保住精度 | 半精度输入先升 fp32，末端一次舍回原 dtype | CUDA codegen 限制 + fp32 算术单元 | **语义层**（fp32 opmath 精度语义，可移植）；**API 层不可移植**（详见 §0.5 #3） |
| 4 | strategy-aware `npt`（fp32=4、fp16/bf16=8） | 128-bit 访存对齐（4×4B / 8×2B） | `_strategy_npt` 启发式 | memory coalescing（对齐） | 实现层（NPU 折叠为 `block_size`） |
| 5 | 无 autotune（elementwise 固定启发式配置） | 带宽受限 + 单步逐元素，搜索 ROI 低 | `default_config` 固定 `threads=256`+`npt` | 无硬件依赖 | 可移植（NPU 侧同样固定单实现） |

> 识别不出来 ≠ 不存在。上述 5 项为源码中可识别的全部优化/规避手段；后续迁移逐项处置（§0.5）。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 Phase R 调研结论 §1.6.0）

> **v0 教训（本表相对 v0 的关键修正）**：v0 在此表把「`T.erf` → 原生 `T.verf`」判为「等价替换」并**只信文档**；设备实测证伪（`T.verf` 为截断多项式）。本版对关键 API 一律以**目标设备实测**为准（探针见 §1.6.0 / §3.4）。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `y = erf(x)`（含 fp32 opmath 精度语义） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 依 §0.1 |
| 计算算法（步骤 1–3 逐元素映射结构） | 算法 | 无（纯数学顺序、无跨元素依赖） | — | **保留（结构）** | 逐元素映射结构保留；计算核需重设计（下述 #3） |
| 优化 #1 `register_copy`（fragment + `T.Parallel` 标量映射） | 优化 | GPU register/fragment 模型 | 无直接等价（NPU 无一维 `threads=`；v-prefix op 作用于整 UB buffer） | **重新设计** | → §0.6 R1：GM→UB→GM + 整 buffer v-prefix 向量链 |
| 优化 #2 block×thread 两级并行 | 优化 | GPU thread/warp 模型 | 无直接等价（NPU 一维 Kernel） | **重新设计** | → §0.6 R1：一维 `T.Kernel(core_num, is_npu=True)` + 核内 `T.serial` 网格跨步 |
| 优化 #3 erf 计算核（源 `T.erf` 真 erf + fp32 中转） | 算法 | CUDA `herf` 缺失 + fp32 算术 | **否**——NPU 无可用真 erf 元算（`T.verf` = 7 阶 Maclaurin 截断多项式，CG-2026-0015；`T.erf` 标量路径编译期段错误） | **重新设计（算法替代）** | → §0.6 R2：自建 A&S 7.1.26 有理-指数逼近（仅用设备实测可用 v-op）；fp32 中转语义 → R3 承接 |
| 优化 #4 strategy-aware `npt` 对齐 | 优化 | memory coalescing（对齐） | 有（32B 尾轴对齐 / 向量宽度 128bit） | **等价替换** | K11：`threads × npt` 折叠为 `block_size`（§5.2），对齐约束按 NPU 32B 复算（§5.3） |
| 优化 #5 无 autotune | 优化 | 无 | 有 | **保留** | NPU 侧同固定单实现（无 strategy/autotune） |
| host 侧 flatten（非 kernel 逻辑） | 算法（host） | 无 | — | **保留** | Op 层传入 `N_total`，kernel 只处理 1-D |

> 依据：#3 的「无可用真 erf」由设备实测复现（`repro/TRAP-verf-maclaurin-truncation.py`，2026-10-12 PASS）+ codegen 源码证据（`src/target/codegen_npuir_api.cc:2419–2524` / `codegen_npuir_dev.cc:3227–3272` / `codegen_npuir_api_a5.cc` 均展开截断多项式；仅 `codegen_npuir_dev_a5.cc:4002` 用 `mlir::math::ErfOp`，本设备不适用）+ `CG-2026-0015` 支撑；替代元算的可用性由 `docs/Tilelang.language/数学操作/` 与设备探针双证（§3.4）。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（承接源方案性能意图 + 语义保持论证）。

**重设计项 R1：并行结构重设计（thread 级并行 → 一维 Kernel + 核内网格跨步）**

- **源方案**：`T.Kernel(ceildiv(N, block_size), threads=256)` + `T.Parallel(256, npt)`，每线程 `npt` 元素；性能意图 = 用大规模线程/grid 吃满内存带宽。
- **NPU 新算法**：一维 `T.Kernel(core_num, is_npu=True)`（`core_num` = 实查物理 Vector 核数 **40**，§5.5 ②）+ 核内 `for i in T.serial(T.ceildiv(n_num, core_num))` 网格跨步任务映射（`task = i * core_num + cid`，`if task < n_num` 守卫）；每 tile 经 `T.copy` GM→UB、整 buffer 向量 op、`T.copy` UB→GM。**模式先例**：`examples/elementwise/vec_add_2d.py` L28–L42、同族迁移 `examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py` L89–L108（`PL-1.22-persistent-1d-grid-stride` 两证）。
- **语义保持论证**：任务映射只重排「哪个核处理哪个元素区间」，各元素计算完全独立（无跨元素依赖、无累加顺序变化），数学语义逐元素恒等；边界：`T.serial` 循环上界 `T.ceildiv(n_num, core_num)` 为编译期静态值，守卫 `task < n_num` 覆盖非整除尾块，不产生越界。

**重设计项 R2：计算核算法替代（源真 erf → 自建 A&S 7.1.26 逼近）⭐ 本版核心修正**

- **源方案**：`T.Parallel` 内标量 `T.erf(T.cast(x_reg[i], 'float32'))`——调用真误差函数（CUDA/codegen 实现）；性能意图 = 单条超越函数指令完成逐元素映射。
- **NPU 现状（设备实测，v0 被证伪处）**：NPU **无可用真 erf 元算**：
  - `T.verf`（文档 §3 称转译为 `Constant/DivF/VMul/VAdd`）在 non-A5 codegen 路径上**实际是 7 阶 Maclaurin 截断多项式** `(2/√π)(x − x³/3 + x⁵/10 − x⁷/42)`，仅 `|x|≲0.5` 有效；实测 `x=1` 误差 4.3e-3、`x=3` 误差 37.9、`x=8` 输出 52828.5（真值 1.0）。证据：`CG-2026-0015` + `repro/TRAP-verf-maclaurin-truncation.py`（设备实跑 PASS）+ `src/target/codegen_npuir_api.cc:2419–2524` 等。
  - `T.erf`（GPU 源形态的 TIR 标量 erf）在 NPU 编译期**段错误**（EXIT=139），不可用（CG-2026-0015 evidence）。
  - 仅 A5 代 `codegen_npuir_dev_a5.cc:4002` 用 `mlir::math::ErfOp`（真 erf），本设备 `Ascend910_9362` 不适用。
- **NPU 新算法（算法替代）**：以设备实测可用的 v-prefix 元算自建 **Abramowitz & Stegun 7.1.26** 有理-指数逼近（x ≥ 0，最大绝对误差 1.5e-7，配合奇对称延拓到负半轴）：

  $$
  t=\frac{1}{1+p\,|x|},\quad
  \operatorname{erf}(x)\;\approx\;\operatorname{sign}(x)\Bigl[1-\bigl(a_1t+a_2t^2+a_3t^3+a_4t^4+a_5t^5\bigr)e^{-x^2}\Bigr]
  $$
  系数：`p=0.3275911`，`a1=0.254829592`，`a2=−0.284496736`，`a3=1.421413741`，`a4=−1.453152027`，`a5=1.061405429`（A&S 7.1.26，来源：Abramowitz & Stegun, *Handbook of Mathematical Functions*, 1964，p.299；数值核对见 `verify_equiv.py`）。

  逐元素向量链（fp32 域，op 顺序 = §3.3 伪代码）：
  `vabs → vmul(·p) → vadd(+1) → vrec → Horner(9 op) → vmul(x²) → vmul(·−1) → vexp → vmul(·poly) → vsub(1−·) → vcmp → vselect → vmul(·sign)`（共 21 op）。
- **语义保持论证（容差内等价 + 边界核对）**：
  1. **误差界**：A&S 7.1.26 的截断误差 `|ε(x)| ≤ 1.5×10⁻⁷`（x≥0，闭式有理-指数逼近），叠加 fp32 逐 op 舍入后仍远低于门禁 `atol=1e-5`（**设计期实测**：设备 fp32 全域 `max_abs=3.725e-7`，`|x|` 分区间 [0,0.5)/[0.5,1)/[1,2)/[2,4)/[4,8) 分别 3.7e-7 / 2.4e-7 / 1.8e-7 / 1.8e-7 / 0；探针 `history_version/design_probe_erf_ops.py`）。
  2. **奇对称**：`erf(−x)=−erf(x)` 严格成立；实现为「对 `|x|` 走单链 + 末端 `select(x≥0,+,−)` 恢复符号」，数学恒等。唯一非光滑点 `x=0`：`y_even(0)=1−Σaᵢ=1−0.999999999≈1e-9`，`sign` 任取 ±1 结果均为 ~1e-9，误差 < atol。
  3. **饱和**：`|x|≳6` 时 `e^{−x²}` 在 fp32 下下溢为 0 → `y_even→1`，结果 = `sign(x)`，与真值 `erf` 的 fp32 舍入结果（`|x|≳3.9` 即舍入为 ±1）**一致**（实测 x=±100 输出 ±1.0 vs golden ±1.0）。
  4. **边界**：`x=+∞`：`t=vrec(∞)=0` → Horner 全 0 → `e=exp(−∞)=0` → `y_even=1`，sign=+1 → 1 ✓；`x=−∞` → −1 ✓；`x=NaN`：`vcmp(NaN≥0)=false` → sign=−1，`y_even=NaN` → `NaN` ✓。
- **源优化意图承接**：源 #3「以 fp32 精度保住结果」意图 → 本项在 fp32 域构造全部逼近链承接（精度实测 3.7e-7 ≪ 门禁）；源的「单条超越函数指令」性能意图在 NPU 上**无法承接**（无元算），改以 21-op 向量链落地，代价已量化并作为工程风险显式登记（§1.6.0 估算 / §9）。

**重设计项 R3：dtype 路径重设计（标量 `T.cast` → 向量 `T.vcast` fp32 中转链）**

- **源方案**：标量 `T.cast(x, "float32")` 使 erf 在 fp32 域计算，末端隐式舍回原 dtype；性能意图 = 以 fp32 精度保住结果。
- **NPU 新算法**：
  - **fp32 路径**：直接在 fp32 UB buffer 上跑 R2 向量链。
  - **fp16/bf16 路径**：`T.vcast(x_ub, xf, round_mode="rint")` → R2 向量链（fp32 域，结果原地留在 `xf`）→ `T.vcast(xf, y_ub, round_mode="rint")`。
  - **bf16 为 API 强制中转**：`T.vexp/vadd/vmul/vsub/vabs/vrec` 的 dtype 矩阵均**不含 bf16**（见 `docs/Tilelang.language/数学操作/*.md` §2.2.1），故 bf16 必须 `vcast` 升 fp32 计算、末端舍回。
- **语义保持论证**：fp32 域算 erf 后单次舍回原 dtype ≡ 源语义；`vcast` 升 fp32 无损、降回 `rint` 单次舍入。**设备实测**（`design_probe_erf_ops2.py`）：fp16 路径 `max_abs=1.221e-4`（< 1 个 fp16 ulp@1.0 ≈ 9.8e-4）、0 违例；bf16 路径 `max_abs=0.0`、0 违例。

**重设计项 R4：符号恢复（源 GPU 由库内 erf 实现自带；NPU 需显式构造）**

- **源方案**：负半轴由 `T.erf` 库实现内部处理，源码无显式符号逻辑。
- **NPU 新算法**：`T.vabs` 取 |x| 走正半轴链；`T.vcmp(xc, zero, cond, "ge")` + `T.vselect(cond, ones, negones, t)`（`t` 复用为 `sign`）+ `T.vmul(xc, t, xc)` 恢复符号。`ones`/`negones`/`zero` 为**只读常量 buffer，在 tile 循环外一次性 `T.fill`**（不产生 per-tile 发射）。
- **语义保持论证**：`sign` 由 `x≥0` 逐元素判定，`y=sign·y_even`；`x=0` 时 `y_even≈1e-9`，符号无关（见 R2 论证 2）。设备实测角点 `[0,3,-3,100]` → `[0.0, 0.999978, -0.999978, 1.0]`，与 `torch.erf` 的 fp32 结果逐值一致。

> 承接说明（源优化意图不丢弃）：§0.4 #1「向量化 128-bit 访存」→ R1/R2 以「GM→UB 整块 `T.copy` + 向量 op 整 buffer」承接；#2「大规模并行吃带宽」→ R1 网格跨步并行承接；#3「fp32 精度」→ R3 fp32 中转链承接（其 API 动因不可移植部分由 R2 算法替代吸收）；#4「对齐」→ K11 折叠 `block_size` + §5.3 32B 对齐复算；#5 保留。

### 0.7 标杆实现

- 源算子文件（golden 语义依据）：`/home/developer/TileOPs/tileops/kernels/elementwise.py`（`ErfFwdKernel`，L2118–L2127）。
- 源仓测试基准：`examples/TileOPs/tests/ops/test_unary_math.py`（`ref_program = torch.erf`；容差 `_get_tolerances` fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2）；边界用例 `MathEdgeFixture` = `repeat([0.0, 3.0, -3.0, 100.0])`。
- manifest：`examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `ErfFwdOp`（L140–L176）。
- golden 以 §0.1 语义为依据（`torch.erf`，保持 fp32 中转语义），**不复刻 §0.6 的 NPU 算法**（保证验证独立性）。

---
## 1. 概述

### 1.1 算子名称

`erf_fwd_kernel`（项目 `erf`，目录 `examples/erf/erf_fwd_kernel/`）。

### 1.2 功能描述

对 1-D 展平张量逐元素计算误差函数 `y = erf(x)`，语义等价于 `torch.erf`；支持 `float16 | bfloat16 | float32`，输出与输入 shape/dtype 一致。单步纯 Vector 逐元素算子，无规约、无矩阵运算、无跨元素依赖。保持源算子的 **fp32 中转**精度语义（fp16/bf16 → fp32 域算 erf → 单次舍回）。

**本版核心变化**：NPU 无可用的真 erf 元算（`T.verf` 实为截断多项式，见 §0.6 R2），故 erf 由**自建 A&S 7.1.26 有理-指数逼近**在 fp32 域计算（误差实测 3.7e-7 ≪ 门禁 1e-5）。

### 1.3 数学公式

$$
y_i = \operatorname{erf}(x_i)=\frac{2}{\sqrt{\pi}}\int_0^{x_i} e^{-t^2}\,dt,\quad i=0,1,\ldots,N-1,\qquad y_i\in(-1,1)
$$

保持源 fp32 中转语义（§0.1）：对所有 dtype，erf 在 fp32 域计算后单次舍回原 dtype。

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法（§0.6），与源算法存在差异：**

- **计算核心**（§0.6 R2）：**自建 A&S 7.1.26 有理-指数逼近**——
  `t = vrec(1 + p·|x|)`；`poly(t) = Horner(a1..a5)`；`erf ≈ sign(x)·(1 − poly(t)·vexp(−x²))`。
  源为 `T.Parallel` 内标量真 `T.erf`；NPU 因无可用 erf 元算改为上述 21-op 整 UB buffer 向量链。
- **dtype 路径**（§0.6 R3）：fp32 直算；fp16/bf16 先 `T.vcast(rint)` 升 fp32 → 逼近链 → `T.vcast(rint)` 舍回。保持源 fp32 opmath 语义。
- **符号恢复**（§0.6 R4）：`T.vabs` + `T.vcmp/T.vselect` 对 |x| 单链计算后恢复符号。
- **并行结构**（§0.6 R1）：一维 `T.Kernel(core_num, is_npu=True)` + 核内 `T.serial` 网格跨步。

除上述外，算法与 §1.6.0 调研选定的算法族（单步逐元素逼近，访存为理论下界）一致：一趟扫描、无中间 GM 往返、无跨元素依赖。

### 1.5 数据流图

```
fp32 路径（xc = x_ub，链尾原地复用）：
GM[x] --T.copy--> UB[x_ub] --[A&S erf 向量链 21 op in fp32]--> UB[x_ub] --T.copy--> GM[y]

fp16/bf16 路径（fp32 中转，xc = xf）：
GM[x(dtype)] --T.copy--> UB[x_ub(dtype)] --T.vcast(rint)--> UB[xf(fp32)]
     --[A&S erf 向量链 21 op]--> UB[xf(fp32)] --T.vcast(rint)--> UB[y_ub(dtype)] --T.copy--> GM[y]
```

A&S 向量链内部（全部 UB 级、原地复用）：
```
x_f32 → vabs(=|x|) → vmul(·p) → vadd(+1) → vrec(=t)
      → Horner(t; a1..a5) → vmul(t,t)=x² → vmul(·−1)=−x² → vexp=e^{−x²}
      → vmul(poly,e)=prod → vsub(1−prod)=y_even → [vcmp/vselect=sign] → vmul(y_even,sign)=y
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（对同一数学语义的算法族回答调研四问，选定算法族），再在数学等价前提下优化公式，再做循环/标量向量化替代分析。迁移任务：1.6.1–1.6.3 的分析对象为 §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 本算子为**单步逐元素类**（§0.3 步骤 3 步、无规约/窗口/矩阵），按 `_shared/standards/algorithm-research.md` §3 执行**轻量调研**（四问各给结论 + 依据），但四问逐问回答。
>
> **本轮调研的驱动事件（v0 失败教训）**：v0 把「`T.verf` 直算」当作唯一算法，未把「同语义算法族」拉出来比较，也**未对设备实测 API 语义**——结果核心假设被设备证伪。本版调研四问以「**无原生 erf 元算**」为前提，从零枚举 erf 的数值逼近算法族。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线：源算法（真 `T.erf`，GPU） | `y = T.erf(T.cast(x,'f32'))` → 舍回 dtype | — | — | ✅（基线，**NPU 不可用**） |
| 1 | `T.verf` 直算（v0 采纳） | `verf(x)`（1 v-op） | ⚠️ **等价性不成立**：设备实测 non-A5 为 7 阶 Maclaurin 截断多项式（仅 `\|x\|≲0.5` 有效） | — | ✅（**已证伪，弃**） |
| 2 | **自建 A&S 7.1.26 有理-指数逼近**（采纳） | `sign(x)·(1−poly(t)e^{−x²})`, `t=1/(1+p\|x\|)` | **容差内等价**：截断误差 ≤1.5e-7（x≥0），fp32 舍入后实测 3.7e-7 ≪ atol 1e-5 | 少算：单超越函数 → 1 vexp + 20 算术/向量 op；少搬：整块 UB | ✅ |
| 3 | A&S 7.1.28（4 项 erfc 有理-指数） | `sign(x)·(1−(b1t+..+b4t⁴)e^{−x²})` | 容差内等价，误差 ≤3e-7（略大） | 比 #2 少 2 个 Horner op（7 vs 9） | ✅ |
| 4 | Winitzki sqrt-exp 逼近 | `sign(x)·sqrt(1−e^{−x²(4/π+a x²)/(1+a x²)})` | 容差内等价但 **误差 ~1.2e-4 ≫ atol 1e-5** | 少算（~13 op） | ❌（精度不达标，见 R4） |
| 5 | 分段 Chebyshev/有理逼近 | 按 `\|x\|` 区间选多项式/有理式 | 容差内等价（可任意精度） | 需分段 `vselect` + 多组系数，op 数超过 #2 | ❌（op 更多、系数成本高） |
| 6 | 直接多项式（Maclaurin/Taylor） | Σ 奇次项 | **非等价**（大 `\|x\|` 发散，同候选 1 病根） | 无 | ❌（结构上不适用全域） |
| 7 | tanh 型逼近 | `erf(x)≈tanh(αx+βx³)` | 容差内等价但误差 ~1e-3，达不到 fp32 门禁 | — | ❌（精度不达标） |
| 8 | 经用户批准补齐 `npuir_verf`（CG-2026-0015） | 编译器层修复 | 数学恒等（真 erf） | 最优（1 op） | ❌（**本任务内不可行**，需编译器层改动 + 用户批准，见 R4） |
| 9 | 恒等变形（CSE / 换底 / 降代换） | — | erf 为非初等超越函数（无初等闭式），无恒等化简；逼近式内部的 CSE/Horner 收益见 §1.6.1 | 见 §1.6.1 | ✅（部分纳入 §1.6.1） |

**R2 在线算法**：**无在线化收益**——逐元素映射本身天然单遍（无跨元素依赖，不存在可被「单遍/流式」消去的多遍基线）。结构判据：本算子无 running 统计量（无归约/无 max/无 sum），在线变体定义不适用。依据：源算法即单遍直写；`examples/elementwise/` 全系逐元素实现（`vec_add_1d.py`、`example_elementwise_add.py`）单遍。

**R3 复杂度对比**（四口径；本算子带宽/发射受限，访存/遍数是主导项）：

| 算法候选 | FLOPs（v-op 数/元素） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|----------------------|---------------|---------|-------------|---------------------|
| 基线（源真 `T.erf` 直译） | 1 超越 + cast（NPU 无对应） | 2·N·elem_bytes | 1 遍 | O(block_size) | 可分块给多核，无跨核归并 |
| 候选 1（`T.verf`） | 1（有效 3） | 2·N·elem_bytes | 1 遍 | O(block_size) | 同上；**精度不达标** |
| **候选 2（A&S 7.1.26，采纳）** | **21**（abs1+mul+add+rec1+Horner9+mul(neg)1+exp1+mul1+sub1+cmp1+select1+mul1） | 2·N·elem_bytes | 1 遍 | O(block_size)（~10 个 fp32 UB buffer，§4.5） | 同上；无跨核归并 |
| 候选 3（A&S 7.1.28） | 19（Horner 7） | 2·N·elem_bytes | 1 遍 | 同上（少 1 个 buffer） | 同上 |
| 候选 4（Winitzki） | ~13 | 2·N·elem_bytes | 1 遍 | 同量级 | 同上；**精度不达标** |
| 候选 8（真 erf 元算） | 1 | 2·N·elem_bytes | 1 遍 | O(block_size) | 同上；**需编译器层修复** |

> 口径说明：FLOPs 以「每元素 v-op 数」计（超越函数 vexp 按 1 op 记，估算口径）；候选 2/3 计入 fp16/bf16 的 +2 `vcast`；访存 = 输入读 + 输出写，**全部候选同为理论下界 `2·N·elem_bytes`**（无中间 GM 往返）；扫描遍数 = GM 读取次数 = 1；中间缓冲 = UB 单 tile 驻留（非 O(N)）。候选 2/3/4 访存相同，差异在 **op 数（发射成本）**。

**R4 硬件亲和性评估**（逐候选对照检查清单）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（源真 `T.erf` + `T.Parallel`） | erf → Vector；但 NPU 无 `T.Parallel` 标量融合路径 | — | — | — | — | ❌ 不可直接移植（§0.6 R1/R2） |
| 候选 1（`T.verf`） | Vector | 小 | ✓ | ✓ | 单步 | ❌ **设备证伪**：`T.verf` 是截断多项式（`repro/TRAP-verf-maclaurin-truncation.py`），非真 erf |
| **候选 2（A&S 7.1.26 自建，采纳）** | 全部 op 落 Vector（`vabs/vmul/vadd/vrec/vexp/vcmp/vselect`，均为设备实测真实可用元算，§3.4） | ~10 fp32 buffer，@bs2048 ≈ 82 KB（§4.5）✓ | 尾轴 32B 对齐 ✓；v-op 作用于整 buffer | N 编译期常量 ✓ | 单步无多阶段；跨 tile 重叠由 auto-multi-buffer 处理 | ✅ **采纳**（精度实测达标、容量充足、op 全部设备验证） |
| 候选 3（A&S 7.1.28） | 同上 | 同上（略省） | ✓ | ✓ | 同 | ⚠️ 备选：误差 3e-7 仍达标，但无本仓实测且仅省 2 op（<10%），**首选精确度余量更大的 7.1.26** |
| 候选 4（Winitzki） | Vector | 略省 | ✓ | ✓ | 同 | ❌ 误差 ~1.2e-4 > atol 1e-5（来源：Winitzki, *A handy approximation for the error function and its inverse* (2008), 模型知识 + 数学分析；未文档化假设 = 无本仓实测，估算依据 = 该逼近公示误差界）；**已就地数学核验：在 erf 值域内其相对误差量级 1e-4，超 fp32 门禁** |
| 候选 6（Maclaurin） | Vector | — | — | — | — | ❌ 大 `\|x\|` 发散（同候选 1 设备证据） |
| 候选 7（tanh 型） | Vector（`T.vtanh` 实测真 tanh） | — | — | — | — | ❌ 逼近误差 ~1e-3 超 fp32 门禁（模型知识；未文档化假设 + 数学分析） |
| 候选 8（真 erf 元算 / `npuir_verf` 补齐） | Vector（最优） | 最小 | ✓ | ✓ | 单步 | ❌ **本任务内不可行**：需 non-A5 codegen 修复（CG-2026-0015，layer: codegen），非本任务范围且须用户批准；已登记能力缺口 |

**⚠️ 设备实测附带证伪（v0 残留疑虑的澄清）**：设计错误摘要提到「同类元算 `vtanh` 亦为截断式多项式（建议一并排查）」。**本轮设备探针证伪该疑虑**：`T.vtanh` 在 Developer 模式实测 `max_abs=1.192e-7`（对比 `torch.tanh`，域 ±10），为**真 tanh**（与 `docs/.../T.vtanh.md` §3「Developer 模式用 `(exp(2x)−1)/(exp(2x)+1)`」一致）——截断多项式仅存在于 Expert 模式。故 `vtanh` 可用；但 tanh 型 erf 逼近精度不足（候选 7），不作为采纳路线。

**调研结论**：**选定候选 2（自建 A&S 7.1.26 有理-指数逼近），无更优可行替代。**

关键依据：① R3 全部候选访存同为下界 `2N·elem_bytes`，差异在 op 数——候选 2（21 op）在「精度余量（实测 3.7e-7 vs 门禁 1e-5，27× 余量）」与「op 数」间取平衡；② R4 逐候选淘汰：候选 1 设备证伪、候选 4/7 精度不达标、候选 6 结构不适用、候选 3 为可接受备选（略省 op 但无实测）、候选 5 op 更多、候选 8 本任务不可行；③ 候选 2 的全部 op 均在**目标设备实测**为真实元算（§3.4 探针表）。

**调研范围**（「无更优替代」依据）：① 参考表 `tilelang-op-design/references/algorithm-candidates.md` 对「逐元素超越函数（erf/exp/log/tanh）」无专门逼近候选条目（本仓算法候选库不含 erf 逼近族）；② 本仓 `examples/` 全系（`elementwise/*`、`bitwise_not`、mish）均为「有原生元算时直译」形态，**无 erf 逼近先例**（本算子为首例）；③ pattern-library：`kb_search "erf verf maclaurin truncation"` 命中 `CG-2026-0015`（capability-gaps）；`traps-runtime.md` 的 `TRAP-verf-maclaurin-truncation`（repro 见 `examples/erf/erf_fwd_kernel/repro/`）；④ **外部已知算法族**（Abramowitz & Stegun 1964 §7.1.26/7.1.28；Winitzki 2008；分段 Chebyshev）由模型知识补充，属 algorithm-research.md §6 第 5 层「只取算法思路」，API 存在性另由本地 `docs/` + 设备探针佐证；⑤ 结构分析：erf 非初等超越函数（无初等闭式），无恒等化简便无「零 op」替代，逼近族即最优可行集。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)；纯 Vector 逐元素豁免容量项）**：

- 流量项 = `bytes ÷ 有效带宽`，`bytes = 2·N·elem_bytes`（manifest roofline 同口径）。引用 `CONST-mte2-degradation`（`constants.md`，版本戳 tilelang 0.1.2+ed787bb / CANN 8.5.0 / 2026-09-07；fp16 4-set 轮换 1M 33.9 → 16M 30.2 → 64M 20.9 → 256M 18.4 GB/s/core）：**该常数绑定 Ascend910B2C，本设备为 Ascend910_9362，数字标注「待重验」，仅作量级估算**。本设备实查 Vector 核 40（§5.5 ②）。
- 发射项 = `(n_num / core_num) × k × ~0.5µs/op`（`CONST-vector-launch-overhead`，绑定 910B2C=`fixed dispatch`，**未文档化假设 + 估算依据**）；`k` = 每 tile 向量 op 数：**fp32 `k=21`**，fp16/bf16 `k=23`（+2 vcast）。
- 容量项 = 0（纯 Vector，无跨引擎往返；`CONST-store-fixpipe-gm-only` 不适用）。

以 `block_size = 2048`、物理 Vector 核 = 40 估算：

| workload | dtype | N | bytes | 流量项 | 发射项 (k) | **估算下界 max** |
|----------|---|------|-------|--------------------|----------------------------|------------------|
| smoke-1m | fp32 | 1,048,576 | 8.4 MB | 6.2 µs | 134.4 µs (k=21) | **≈ 134 µs** |
| smoke-1m | fp16 | 1,048,576 | 4.2 MB | 3.1 µs | 147.2 µs (k=23) | **≈ 147 µs** |
| elementwise-16m | fp32 | 16,777,216 | 134.2 MB | ≈ 111 µs | ≈ 2150 µs (k=21) | **≈ 2150 µs** |
| elementwise-16m | fp16 | 16,777,216 | 67.1 MB | ≈ 56 µs | ≈ 2355 µs (k=23) | **≈ 2355 µs** |
| elementwise-256m | fp16 | 268,435,456 | 1073.7 MB | ≈ 1459 µs | ≈ 37683 µs (k=23) | **≈ 37683 µs** |

**设计期估算（D-2 roofline）**：代表性 `smoke-1m fp32` 估算下界 ≈ **134 µs**（流量 6.2 + 发射 134.4 + 容量 0，发射项主导）；**全部 workload 由发射项主导**（本算子计算链为无原生 erf 元算下的 21-op 逼近链，op 数是发射成本的第一驱动）。

> **估算不确定性声明（未文档化假设 + 估算依据）**：① `k` 与「每 tile op 链跨 tile 重叠程度」均无本仓实测——`CONST-vector-launch-overhead` 的 0.5µs/op 绑定 910B2C 且为 **fixed dispatch 成本**，auto-multi-buffer 下向量 op 可与 MTE2 重叠（`PL-1.6-copy-floor` 与同源 `PL-1.17-subfp32-fp32-transit` 按调度提示为 **stale**，仅作背景，**不作为决策依据**），故上表发射项可能**上偏**；② 对照证据：同族 `bitwise_not`（k=1）实测 256m 1932µs 低于其设计估算 2918µs（实测优于估算 ≈1.5×）；③ **本算子的发射项远重于 v0 的 3-op 链**（21/3 = 7×），这是「无原生元算」的固有代价——**唯一根本解是 CG-2026-0015 编译器层修复**（候选 8）。真实下界应落在 `flow`（firm 带宽地板）与上表 emission 之间；Stage 4 baseline 采集后按 `hardware-cost-model.md` §3 对账并回填常数。**block_size 是摊薄发射的第一杠杆**（发射 ∝ 1/block_size），但因 §4.5 UB 预算限制为 2048/3072，放大空间有限。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（自建 A&S 7.1.26 + fp32 中转）。以下 5 项均完成「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；采纳项 #1–#4 的等价性经 `verify_equiv.py` 机器验证（结果表见本节末）。

| # | 优化项 | 原式（基线） | 优化后公式（采纳） | 等价性论证 | 收益量化 |
|---|--------|------|-----------|-----------|---------|
| 1 | **erf 计算核：自建 A&S 7.1.26 逼近替代（不可用的）真 erf 元算** | `erf(x)`（源 `T.erf`；NPU 无可用元算） | `sign(x)·(1−poly(t)e^{−x²})`, `t=1/(1+p\|x\|)`，`poly` 为 5 项 Horner | **容差内等价**：截断误差 ≤1.5e-7（A&S 7.1.26 公布界，x≥0）；设备 fp32 实测 `max_abs=3.725e-7`，全域分区间 ≤3.7e-7，均 ≪ `atol=1e-5`（27× 余量）；`verify_equiv.py` #1/#2/#3 | 使 fp32/fp16/bf16 三档在无原生 erf 的 NPU 上**可落地**（v0 不可行）；代价 = 21 op（vs 源 1 op），已量化登记 |
| 2 | **Horner 嵌套求值**（多项式求值降 op） | 直接展开 `a1t+a2t²+a3t³+a4t⁴+a5t⁵`：4 次幂乘 + 5 次系数乘 + 4 次加 = 13 op（幂乘可复用：t²,t³,t⁴,t⁵ 4 mul） | `poly=((( (a5t+a4)t +a3)t +a2)t +a1)t`（Horner）：5 mul + 4 add = **9 op** | 数学恒等（Horner 法则，多项式求值的结合律重排） | 13 → 9 op（**−4 op/tile**，约 −19% 计算链）；`verify_equiv.py` 以 Horner 形式逐 op 复现 |
| 3 | **倒数 `vrec` 替代除法**（分母为标量加 \|x\| 的仿射式） | `t = 1/(1+p\|x\|)` 用 `vdiv(ones, den)`：需 1 个常 1 buffer + 1 op | `T.vrec(den, t)`：1 op，**免 ones buffer** | 数学恒等（`1/a ≡ rec(a)`）；设备实测 `vrec` 与 `vdiv` 精度同量级（rel 1.189e-7 vs 1.189e-7，探针 P1b/P1c） | 省 1 个 fp32 UB buffer（§4.5）；op 数持平但减少常量 buffer 与一次依赖 |
| 4 | **奇对称复用** `erf(−x)=−erf(x)` | 对正/负半轴各设一组系数/分支（或对大 \|x\| 另设饱和分支） | 对 `\|x\|` 走**唯一** A&S 链，末端 `sign(x)=select(x≥0,+1,−1)` 恢复符号 | 数学恒等（erf 奇函数）；`x=0` 处 `y_even≈1e-9`，符号无关（误差 <atol）；设备实测角点 `[0,3,-3,100]→[0,0.999978,−0.999978,1.0]` 与 golden 一致 | 计算路径唯一（无负半轴分支/重复系数），**code/branch 最小化**；大 \|x\| 饱和由 `e^{−x²}→0` 自然实现（无需 clamp op） |
| 5 | **fp16/bf16 fp32 中转计算域** | 源语义：fp32 域算 erf + 单次舍回 | `vcast↑f32(x) → A&S(fp32) → vcast↓dtype(rint)` | fp32 域算后单次舍回 ≡ 源语义；fp16↔fp32 / bf16↔fp32 往返无损；与 `torch.erf(dtype)` golden 门禁内一致（`verify_equiv.py` #2/#3/#4/#5） | 精度对齐 golden（fp16 实测 max_abs 1.221e-4 <1 ulp@1.0；bf16 max_abs 3.1e-5）；代价 +2 vcast；bf16 为 API 强制（数学元算不含 bf16） |

**优化结论**：采纳 5 项。**公式层面已无进一步计算量/访存量可优化空间**——erf 非初等超越函数，逼近族即最优集；链内 CSE 已完成（`|x|` 复用为 `x²`、系数复用）；剩余 op 均为必要算子（abs/仿射/倒数/Horner 主体/exp/饱和/符号）。**优化后公式**（供 §3.1 使用）：
- `float32`：`y = A&S_7_1_26(x)`（21 op 向量链，§3.3）
- `float16 / bfloat16`：`y = vcast(A&S_7_1_26(vcast(x, 'float32')), dtype)`

**等价性机器验证（D-1）**：脚本 `examples/erf/erf_fwd_kernel/verify_equiv.py`（CPU；候选式 = A&S 7.1.26 逐 op 以 fp32 舍入路径复现设备链，末端按目标 dtype 单次舍回；基线式 = `torch.erf` 同 dtype golden）。**执行结果表**（脚本实际输出，`python3 verify_equiv.py`，ALL 6 ITEMS EQUIV_PASS）：

| 采纳项 | 候选式 | 基线式 | samples | 最大 abs 差 | 违反率(atol+rtol) | 结论 |
|--------|--------|--------|---------|------------|--------|------|
| #1 fp32 | A&S(fp32 arith) | `torch.erf(x)` (fp32 golden) | 1,048,599 | 5.662e-07 | 0.000000 | **EQUIV_PASS** |
| #2 fp16 | `round_f16(A&S fp32)` | `torch.erf(x)` (fp16 golden) | 1,048,599 | 1.221e-04 | 0.000000 | **EQUIV_PASS** |
| #3 bf16 | `round_bf16(A&S fp32)` | `torch.erf(x)` (bf16 golden) | 1,048,599 | 3.052e-05 | 0.000000 | **EQUIV_PASS** |
| #4 fp16 中转无损 | `f16→f32→f16(rint)` | `x`（恒等） | 1,048,599 | 0.000e+00 | 0.000000 | **EQUIV_PASS** |
| #5 bf16 中转无损 | `bf16→f32→bf16(rint)` | `x`（恒等） | 1,048,599 | 0.000e+00 | 0.000000 | **EQUIV_PASS** |
| #6 intrinsic 逼近界 | A&S(fp64) | `erf_fp64(x)` | 1,048,599 | 1.394e-07 | 0.000000 | **EQUIV_PASS** |

> 说明：① 判定准则 = Stage 3 门禁同口径 `|cand−base| ≤ atol + rtol·|base|`（fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2），违反率 0 即 PASS；#6 判据为逼近截断误差 <1e-6（A&S 7.1.26 公布界 1.5e-7，实测 1.39e-7 复核通过）。② 近似式非位级恒等，故 **max_ulp 不作判据**（近 `x→0` 处 golden 值近 0、ulp 极小，比值无意义）——正确判据是门禁合规，已按此实现。③ 覆盖随机张量 `randn·2`（2²⁰）+ 角点（±0/±inf/NaN/次正规/敏感尾段 ±3,±3.9,±5/饱和 ±100）——显式含 v0 §8.3 必失败的角点 `[0,3,-3,100]` 与 randn 全域。④ 前 6 项超差计数全 0，全部通过（无一超差）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 逐元素 `\|x\|` | 源标量 `T.abs` | `T.vabs(xc, xa)`（佐证：`docs/.../数学操作/T.vabs.md` §3 `hivm::VAbsOp`；设备探针 P2 实测） | ✅ | — |
| 2 | 仿射 `1+p\|x\|` | 标量乘加 | `T.vmul(xa, p, t)` + `T.vadd(t, 1.0, t)`（佐证：`T.vmul.md`/`T.vadd.md` §2.3 标量广播；设备探针 P2） | ✅ | — |
| 3 | 倒数 `1/den` | 标量除法 | `T.vrec(den, t)`（佐证：`T.rsqrt.md` 同族；`tilelang/language/customize_npuir.py:386 npuir_rec`；设备探针 P1b rel 1.189e-7） | ✅ | — |
| 4 | Horner 5 项多项式 | 标量循环 | 9 × `T.vmul/T.vadd`（整 buffer）（佐证：`T.vmul/vadd` §2.3；探测 P2 实测） | ✅ | — |
| 5 | `x²` / `−x²` | 标量乘 | `T.vmul(xa,xa,xa)` + `T.vmul(xa,−1,xa)`（原地） | ✅ | — |
| 6 | `e^{−x²}` | 标量超越函数 | `T.vexp(xa, e)`（佐证：`docs/.../T.vexp.md` §3 `hivm.hir.vexp`；探针 P1a rel 1.182e-7） | ✅ | — |
| 7 | `1−prod` | 标量减 | `T.vsub(ones, poly, xc)`（`ones` 循环外 `T.fill`）（佐证：`T.vsub.md` §2.3） | ✅ | — |
| 8 | 符号判定 + 恢复 | 标量 `if x>=0` 分支 | `T.vcmp(xc, zero, cond, "ge")` + `T.vselect(cond, ones, negones, t)` + `T.vmul(xc, t, xc)`（佐证：`docs/.../比较操作/T.vcmp.md` §2.3、`条件操作/T.vselect.md` §2.3；探针 P2 角点实测） | ✅ | — |
| 9 | 核内网格跨步任务循环 | `for i in T.serial(ceildiv(n_num, core_num))` | 无（核内调度循环，非逐元素计算） | ❌ | tile/block 级任务调度循环；`T.verf`→已弃，改用 `T.serial`；每核迭代数 `ceildiv(n_num,40)`（1M@bs2048 = 13 次），非逐元素热点；向量 op 作用于整 buffer，元素级并行由向量指令完成。先例 `vec_add_2d.py` L32、`PL-1.22` |
| 10 | block 起始索引 / 任务映射 | `task=i*core_num+cid`、`off=task*block_size` | 无 | ❌ | block 级索引/任务映射，每 tile 少量标量，非逐元素热点；无逐元素等价 API |
| 11 | 尾块边界（非整除 N） | `tail = T.min(block_size, shape-off)` | 无 | ❌ | 依赖动态 shape 的边界处理；manifest 全部 workload 的 N 整除 `block_size`，设计保留 `shape: T.int32` 守卫以支持任意 N（纯静态化对通用 N 不成立） |
| 12 | 常量 buffer 初始化（ones/negones/zero） | 每元素标量赋值 | `T.fill(ones, ...)`（佐证：`bitwise_not_fwd_kernel.py` L104） | ✅ | 循环外一次性填充（per-kernel，非逐元素热点） |

**向量化结论**：逐元素计算点（#1–#8）**已全部向量化**（`T.vabs/vmul/vadd/vrec/vexp/vsub/vcmp/vselect`，均有 docs 佐证 + 设备探针实测）；保留 3 处标量/循环（#9 核内调度循环、#10 block 索引、#11 动态边界），理由类别：tile 级调度循环 + block 索引 + 动态边界，逐项见上表。与 §6 循环结构一致——kernel 内**无逐元素标量循环**。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

**单轴算子豁免**：本算子为 1-D `[N]` 单轴算子（host 侧把任意 shape 展平为一维），不存在「在哪个轴向量化」的轴选择问题——唯一候选向量化轴即该 1D 元素轴本身。按 skill Phase 2 第 3 项「纯 GEMV/单轴算子可写明『仅一个候选轴』豁免」，记录豁免依据：

- **向量化轴** = 唯一 1D 元素轴（stride=1，连续）。
- **轴质量**：manifest 对齐 → fp32 元素 4B，向量宽 ×4 → `block_size=2048` 每 tile 512 向量槽，尾 lane 浪费 0%（2048 整除 4）；fp16/bf16 元素 2B，向量宽 ×8 → 256 槽，浪费 0%；均按 32B 尾轴对齐（2048×elem_bytes 为 32B 整数倍）。无累加链（无串行累加维）。无 repack（I/O 布局 = 核内布局，一维连续，不需 `T.transpose` 或 host permute）。UB 驻留与 §4.5 一致。
- **迁移注**：GPU 源码的并行轴选择（thread `i`×`npt j` → 元素 `idx`）是**输入而非结论**——NPU Vector 轴与之不对应，§0.6 R1 已独立评估为「核内 grid-stride + 向量指令整 buffer」。
- **决策落地**：主选轴 = 元素轴，同步落入 §3.3 伪代码与 §6 循环结构（逼近链各 op 直接作用于 UB 一维 buffer；无逐元素标量循环）。

**布局决策结论**：选定方案（唯一候选轴）：核内布局 = 一维连续 `[block_size]`（I/O 布局即核内布局，无 repack），向量化轴 = 唯一 1D 元素轴，repack 路径 = 无。弃选候选：无（单轴算子，无第二轴可枚举，按 skill Phase 2 第 3 项豁免）。

---
## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer

### 2.2 选型理由

| 算子特征 | 分析 | 结论 |
|----------|------|------|
| 计算类型 | 纯 Vector（逐元素 erf 逼近链），无 matmul | 不需 Cube/L0C/L1 |
| 是否含归约 | 无归约 | 不需 Expert 级 buffer 控制 |
| 是否需流水线 | 单数据通路 GM→UB→GM，无核间协作 | 不需 CV 融合 |
| 逼近链 API | 全部 v-prefix（`T.vabs/vmul/vadd/vrec/vexp/vsub/vcmp/vselect/vcast`） | 用 v-prefix API |
| 同步需求 | UB 内顺序计算，数据依赖线性 | Developer 自动同步即可 |

决策树路径：`纯 element-wise → 单步逐元素 → 无需精细 buffer 控制 → Developer 模式`。与 `op_requirements.programming_mode=developer` 一致；参考 `examples/elementwise/vec_add_1d.py`（Developer + `T.alloc_ub` + v-prefix）与同族迁移 `examples/bitwise_not/bitwise_not_fwd_kernel/`（`T.fill` + v-op + `T.serial` 网格跨步）。

> **模式敏感性提示**：`T.vtanh`/`T.vexp2` 等的 Developer/Expert 实现不同（`T.vtanh` Expert 走多项式）；本设计仅用 Developer 实测语义（§1.6.0 R4 探针）。

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

> 输入公式为 §1.6.1 的**优化后公式**（A&S 7.1.26 向量链 + fp16/bf16 fp32 中转）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_tile = x[off : off+bs]` | 搬入一维 tile 到 UB |
| 2a | `xf = cast(x_tile)`（仅 fp16/bf16） | fp32 中转（无损升级） |
| 2b | `xa = |xf|` | 取绝对值（奇对称前提） |
| 2c | `t = 1/(1 + p·xa)` | 倒数域变量（`p=0.3275911`） |
| 2d | `poly = Horner(a1..a5; t)` | 5 项多项式（Horner） |
| 2e | `e = exp(−xa²)` | 指数核 |
| 2f | `y_even = 1 − poly·e` | 偶部 |
| 2g | `sign = select(xf ≥ 0, +1, −1)` | 符号 |
| 2h | `yf = sign · y_even` | fp32 域 erf |
| 2i | `y_tile = cast(yf)`（仅 fp16/bf16） | 单次舍回原 dtype |
| 3 | `y[off : off+bs] = y_tile` | 搬出结果到 GM |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 搬入 | `x_tile = x[...]` | `T.copy` | `T.copy(x[off:off+tail], x_ub[0:tail])` | Developer |
| cast 升 fp32（fp16/bf16） | `xf = cast(x, fp32)` | `T.vcast` | `T.vcast(x_ub, xf, round_mode="rint")` | Developer |
| 绝对值 | `xa=abs(xc)` | `T.vabs` | `T.vabs(xc, xa)`（`xc = xf if need_cast else x_ub`） | Developer |
| 仿射 | `t=1+p·xa` | `T.vmul` + `T.vadd` | `T.vmul(xa, P, t)`；`T.vadd(t, 1.0, t)` | Developer |
| 倒数 | `t=1/t` | `T.vrec` | `T.vrec(t, t)`（原地） | Developer |
| Horner | `poly=(((a5t+a4)t+a3)t+a2)t+a1)t` | `T.vmul`/`T.vadd` ×9 | 见 §3.3 | Developer |
| `x²` | `xa=xa·xa` | `T.vmul` | `T.vmul(xa, xa, xa)`（原地） | Developer |
| `−x²` | `xa=−xa` | `T.vmul` | `T.vmul(xa, -1.0, xa)` | Developer |
| 指数 | `e=exp(xa)` | `T.vexp` | `T.vexp(xa, e)` | Developer |
| 乘积 | `prod=poly·e` | `T.vmul` | `T.vmul(poly, e, poly)`（原地） | Developer |
| 偶部 | `y_even=1−prod` | `T.vsub` | `T.vsub(ones, poly, xc)`（复用 `xc`） | Developer |
| 符号判定 | `cond=(x≥0)` | `T.vcmp` | `T.vcmp(xc, zero, cond, "ge")` | Developer |
| 符号选择 | `sign=cond?+1:−1` | `T.vselect` | `T.vselect(cond, ones, negones, t)`（复用 `t`） | Developer |
| 符号恢复 | `y=sign·y_even` | `T.vmul` | `T.vmul(xc, t, xc)`（原地） | Developer |
| cast 舍回（fp16/bf16） | `y_tile=cast(yf,dtype)` | `T.vcast` | `T.vcast(xc, y_ub, round_mode="rint")` | Developer |
| 搬出 | `y[...]=y_tile` | `T.copy` | `T.copy((y_ub if need_cast else x_ub)[0:tail], y[off:off+tail])` | Developer |
| 核内网格跨步 | 任务映射 | `T.Kernel` + `T.serial` | `T.Kernel(core_num, is_npu=True)` + `for i in T.serial(ceildiv(n_num, core_num))` | Developer |
| 常量 buffer | ones/negones/zero | `T.fill` | `T.fill(ones, T.cast(1.0, f32))`（循环外） | Developer |

### 3.3 计算伪代码

> 迁移规则：`@T.prim_func` 参数名 `x, y` 及顺序不变，新增 `shape: T.int32` 用于动态 N 尾块守卫；`T.Tensor` 的 dtype 作为第二位置参数。**v-prefix op 只作用于整 UB buffer，动态 slice 只包在 `T.copy` 里**（前一任务 R-6 实证陷阱；`PL-1.22` 同款）；**常量 buffer 在 tile 循环外一次性填充**（只读，不产生 per-tile op）。

```python
import tilelang
import tilelang.language as T

# A&S 7.1.26 (max abs err 1.5e-7, x >= 0)
_P  = 0.3275911
_A1 = 0.254829592
_A2 = -0.284496736
_A3 = 1.421413741
_A4 = -1.453152027
_A5 = 1.061405429

@tilelang.jit(out_idx=[1], target="npuir")
def erf_fwd_kernel(N, dtype, output_dtype=None, block_size=2048):
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
            xa = T.alloc_ub((block_size,), f32)              # |x| -> x^2
            t = T.alloc_ub((block_size,), f32)               # 倒数域变量 -> sign
            poly = T.alloc_ub((block_size,), f32)            # Horner -> prod
            e = T.alloc_ub((block_size,), f32)               # exp(-x^2)
            ones = T.alloc_ub((block_size,), f32)            # 常数 1（只读）
            negones = T.alloc_ub((block_size,), f32)         # 常数 -1（只读）
            zero = T.alloc_ub((block_size,), f32)            # 常数 0（只读）
            cond = T.alloc_ub((block_size,), "bool")

            # 常量 buffer：tile 循环外一次性填充（只读，跨 tile 有效）
            T.fill(ones, T.cast(1.0, f32))
            T.fill(negones, T.cast(-1.0, f32))
            T.fill(zero, T.cast(0.0, f32))

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

                    # --- A&S 7.1.26 erf 向量链（fp32 域，作用于整 buffer）---
                    T.vcmp(xc, zero, cond, "ge")             # cond = x >= 0（xc 转作结果前用完）
                    T.vabs(xc, xa)                           # xa = |x|
                    T.vmul(xa, _P, t)                        # t = p*|x|
                    T.vadd(t, T.cast(1.0, f32), t)           # t = 1 + p*|x|
                    T.vrec(t, t)                             # t = 1/(1+p|x|)
                    # poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
                    T.vmul(t, _A5, poly)
                    T.vadd(poly, _A4, poly)
                    T.vmul(poly, t, poly)
                    T.vadd(poly, _A3, poly)
                    T.vmul(poly, t, poly)
                    T.vadd(poly, _A2, poly)
                    T.vmul(poly, t, poly)
                    T.vadd(poly, _A1, poly)
                    T.vmul(poly, t, poly)
                    T.vmul(xa, xa, xa)                       # xa = x^2
                    T.vmul(xa, T.cast(-1.0, f32), xa)        # xa = -x^2
                    T.vexp(xa, e)                            # e = exp(-x^2)
                    T.vmul(poly, e, poly)                    # poly = poly*e
                    T.vsub(ones, poly, xc)                   # 复用 xc：xc = 1 - poly*e
                    T.vselect(cond, ones, negones, t)        # t = sign (复用 t buffer)
                    T.vmul(xc, t, xc)                        # xc = sign * y_even

                    if need_cast:                            # 编译期分支：单次舍回
                        T.vcast(xc, y_ub, round_mode="rint")
                        T.copy(y_ub[0:tail], y[off:off + tail])
                    else:
                        T.copy(xc[0:tail], y[off:off + tail])

    return main
```

> **伪代码说明**：① `if need_cast:` 为 Python 编译期条件（JIT 参数编译期确定），不产生运行时分支；**buffer 在 `with T.Kernel` 顶层无条件分配**（避免 `TRAP-conditional-alloc` 的 `Undefined variable`）；② 常量 `ones/negones/zero` 在 `T.serial` **循环外** `T.fill`（只读、跨 tile 有效）；③ v-op 全部作用于**整 buffer**（`PL-1.22` R-6 陷阱），仅 `T.copy` 用动态 slice；④ **`xc` 计算域别名**：fp32 路径 `xc=x_ub`（无中转、零额外 buffer 开销），fp16/bf16 路径 `xc=xf`（`vcast` 升级后计算）；⑤ **`xc` 在链尾复用**为 fp32 结果缓冲（`vsub`/`vmul` 原地），且 `t` 在 Horner 后复用为 `sign`——故无需独立 fp32 结果 buffer；⑥ `y_ub` 仅用于 fp16/bf16 的单次舍回（fp32 路径直接从 `x_ub` 搬出）。**op 计数 = 21**（fp32）/ 23（fp16/bf16，+2 vcast）。

### 3.4 API 可行性确认

> **纪律（v0 教训）**：不仅文档核对，**全部关键 API 附目标设备（Ascend910_9362）实测**（探针脚本 `history_version/design_probe_erf_ops.py` / `design_probe_erf_ops2.py`，2026-10-12 实跑）。

| API | 来源确认（本地） | 设备实测（目标设备） | 状态 |
|-----|-----------------|---------------------|------|
| `T.vexp(src, dst)` | `docs/Tilelang.language/数学操作/T.vexp.md`（§3 `hivm.hir.vexp`）；`customize_npuir.py:295 npuir_exp` | rel err **1.182e-7**（±10，fp32） | ✅ 真实可用 |
| `T.vrec(src, dst)` | `customize_npuir.py:386 npuir_rec`（`T.rsqrt.md` 同族；无独立 md） | rel err **1.189e-7**（0.1–10，fp32） | ✅ 真实可用 |
| `T.vdiv(a, b, c)` | `docs/.../数学操作/T.vdiv.md`（§3 `hivm::VDivOp`） | rel err **1.189e-7**（同 vrec，作对照） | ✅ 真实可用（本设计改用 vrec 省 buffer） |
| `T.vtanh(src, dst)` | `docs/.../数学操作/T.vtanh.md`（§3 Developer 用 exp 形式） | max_abs **1.192e-7**（±10）= 真 tanh | ✅ 真实（**证伪 v0 残留疑虑**） |
| `T.vabs(src, dst)` | `docs/.../数学操作/T.vabs.md`（§3 `hivm::VAbsOp`） | 逼近链 P2 实跑（角点一致） | ✅ 真实可用 |
| `T.vmul/vadd/vsub`（含标量广播） | `docs/.../数学操作/T.vmul.md`·`T.vadd.md`·`T.vsub.md` §2.3（标量 src2/src1） | 逼近链 P2 实跑通过 | ✅ 真实可用 |
| `T.vcmp(A,B,C,mode)` + `T.vselect(Cond,A,B,Out)` | `docs/.../比较操作/T.vcmp.md` §2.3（输出 bool，片上用）；`条件操作/T.vselect.md` §2.3 | 逼近链 P2 符号恢复实跑通过 | ✅ 真实可用 |
| `T.vcast(src, dst, round_mode)` | `docs/.../数据类型转换操作/T.vcast.md` §2.2.1（f16/bf16↔f32 rint） | 探针 P2（fp16/bf16 中转，0 违例） | ✅ 真实可用 |
| `T.copy`（slice） | `examples/elementwise/vec_add_1d.py` L34 | 探针实跑 | ✅ 已核对 |
| `T.alloc_ub` / `T.fill` | `vec_add_1d.py` L28；`bitwise_not_fwd_kernel.py` L104 | 探针实跑 | ✅ 已核对 |
| `T.Kernel(n, is_npu=True)` + `T.serial` | `vec_add_2d.py` L28–L42 | 探针实跑 | ✅ 已核对 |
| ~~`T.verf`~~ | `docs/.../T.verf.md` | ❌ **截断多项式（非真 erf）**，`repro/TRAP-verf-maclaurin-truncation.py` | ❌ **弃用**（CG-2026-0015） |
| ~~`T.erf`（标量 TIR）~~ | 源 GPU 形态 | ❌ 编译期段错误 EXIT=139（CG-2026-0015 evidence） | ❌ **弃用** |

**逼近链端到端设备实测**：`A&S fp32 max_abs=3.725e-7`（全域 ±8，分区间 ≤3.7e-7）；`fp16 max_abs=1.221e-4 / bf16 0.0`（门禁 1e-3 / 1.6e-2，0 违例）；角点 `[0,3,-3,100]` → `[0.0,0.999978,-0.999978,1.0]`（与 `torch.erf` fp32 一致）。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(core_num, is_npu=True)` |
| GPU 专用 API | Yes（源用标量 `T.erf` + `T.Parallel` + `threads=`） | 换为 v-prefix 向量链 + `is_npu=True`（§0.5/§0.6） |
| GEMM 非整除 | No | 非 GEMM；非整除由 `tail` + `T.ceildiv` 处理（§5.4） |
| L0C 溢出 | No | 无 Cube 计算 |
| 物理核数限制 | Yes | 逻辑核数 = `ceildiv(N, 2048)` 远超物理核 40 → 核内 `T.serial` 网格跨步（§5.5） |
| **bf16 v-prefix 数学 API 缺失** | Yes | `T.vexp/vadd/vmul/vsub/vabs/vrec` 不含 bf16 → bf16 走 fp32 中转（§0.6 R3） |
| **无原生真 erf 元算** | Yes | `T.verf` = 截断多项式、`T.erf` 段错误 → 自建 A&S 逼近（§0.6 R2 / CG-2026-0015） |
| 条件 `T.alloc` 非法 | Yes（编译期分支易误用） | buffer 全部在 `T.Kernel` 顶层无条件分配（§3.3） |
| bool 不可写回 GM | Yes（`vcmp` 输出 bool） | `cond` 仅片上使用（配 `vselect`），从不 `T.copy` 到 GM（`T.vcmp.md` §2.3） |
| 标量操作数需在 `T.Kernel` 内、`T.Scope` 外定义 | Yes | 系数为 Python 编译期常量传入（`T.vmul.md` §2.3 注 2） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU 源） | 本项目（Ascend npuir） | 转换方案 |
|--------|-------------------|----------------------|----------|
| Kernel 装饰器 | `@tilelang.jit(out_idx=[1])` | `@tilelang.jit(out_idx=[1], target="npuir")` | K1 加 `target="npuir"` |
| Kernel 声明 | `T.Kernel(ceildiv(N,bs), threads=256)` | `T.Kernel(core_num, is_npu=True)` + `T.serial` 网格跨步 | K2/K3 去 `threads=`、加 `is_npu=True`；§0.6 R1 |
| 并行模型 | `T.Parallel(threads, npt)` 线程级 | 一维核 + 向量指令整 buffer | K11：`threads×npt` 折叠为 `block_size` |
| **erf 计算核** | `T.erf`（真 erf 元算） | **自建 A&S 7.1.26 向量链（21 op）** | §0.6 R2 / §1.6.1 #1 |
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
| `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/` | 相似（迁移形态） | 同族 unary elementwise 迁移（v-prefix + UB + fp32 中转 + 尾块） |
| `docs/Tilelang.language/数学操作/T.vexp.md`·`T.vrec.md`(customize_npuir)·`T.vmul.md`·`T.vadd.md` | API | 逼近链各 op 规格与 dtype 矩阵 |
| `docs/Tilelang.language/比较操作/T.vcmp.md`·`条件操作/T.vselect.md` | API | 符号判定的 bool 片上用法 |

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
| `xa` | `(block_size,)` | `float32` | UB | `\|x\|` → `x²` → `−x²`（原地复用） |
| `t` | `(block_size,)` | `float32` | UB | 仿射分母 → `t=1/(1+p\|x\|)` → Horner 变量 → **复用为 sign** |
| `poly` | `(block_size,)` | `float32` | UB | Horner 累加器 → `prod=poly·e`（原地） |
| `e` | `(block_size,)` | `float32` | UB | `exp(−x²)` |
| `ones` / `negones` / `zero` | `(block_size,)` | `float32` | UB | 常量 1 / −1 / 0（**只读**，循环外 `T.fill`） |
| `cond` | `(block_size,)` | `bool` | UB | `x ≥ 0` 掩码（片上，从不写回 GM） |

> **buffer 分配策略**：全部在 `with T.Kernel` 顶层**无条件分配**（避免 `TRAP-conditional-alloc`）；常量 buffer 循环外填充；**`xc` 计算域别名**（fp32→`x_ub` / fp16/bf16→`xf`）在链尾原地复用为 fp32 结果、`t` 复用为 `sign`、`xa/poly` 原地复用（§3.3）。**相对 v0 新增 ~7 个 fp32 工作/常量 buffer**（原 v0 仅 4 个），这是逼近链的固有内存代价（§4.5 预算核对）。

### 4.4 内存搬运路径

```
fp32 路径（纯 Vector，无 L1/L0；xc = x_ub）：
GM[x] --T.copy--> UB[x_ub] --[A&S 链 21 op（原地复用 x_ub 承载结果）]--> UB[x_ub] --T.copy--> GM[y]

fp16/bf16 路径（fp32 中转；xc = xf）：
GM[x(dtype)] --T.copy--> UB[x_ub(dtype)] --T.vcast(rint)--> UB[xf(fp32)]
     --[A&S 链 21 op（原地复用 xf 承载结果）]--> UB[xf(fp32)] --T.vcast(rint)--> UB[y_ub(dtype)] --T.copy--> GM[y]
```

### 4.5 UB 内存预算

`block_size = 2048`：

| 路径 | x_ub | y_ub | xf | xa | t | poly | e | ones/negones/zero | cond | 合计 | ×1.12（`CONST-capacity-910B2C` 上偏系数） |
|------|------|------|----|----|---|------|----|-------------------|------|------|------------------------------------------|
| fp32 | 8 KB | 8 KB | 8 KB | 8 KB | 8 KB | 8 KB | 8 KB | 24 KB | 2 KB | **82 KB** | ≈ 92 KB < 192 KB ✓ |
| fp16 | 4 KB | 4 KB | 8 KB | 8 KB | 8 KB | 8 KB | 8 KB | 24 KB | 2 KB | **74 KB** | ≈ 83 KB ✓ |
| bf16 | 4 KB | 4 KB | 8 KB | 8 KB | 8 KB | 8 KB | 8 KB | 24 KB | 2 KB | **74 KB** | ≈ 83 KB ✓ |

> UB 硬上限 = 192 KB/AIV（`CONST-capacity-910B2C`，绑定 Ascend910B2C；本设备 Ascend910_9362 同规格）。**膨胀系数注**：`CONST-capacity-910B2C` 记录「显式 alloc 结构（逐 buffer 显式分配、首维 = half）实测 0.981×手工 / ~12% 余量即足」（2026-09-16 反例数据点）——本设计为逐 buffer 显式 `T.alloc_ub`，按 ×1.12 保守估算仍 <192 KB。**block_size 放大空间有限**：`bs=3072` → 123 KB（×1.12 ≈ 138 KB）可行；`bs=4096` → 164 KB（×1.12 ≈ 184 KB）**接近上限**，须以编译探针（`ub overflow` 报文）实测校准（`TRAP-UB-multibuffer-inflation`：auto-multi-buffer 下 footprint 结构类平台值 ≈20/26 B/elem，本结构 resident ≈41 B/elem，**须实测而非外推**）。故默认 `block_size=2048`；`bs=3072/4096` 列为 Stage 4 发射项调优杠杆（§5.2）。

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
def erf_fwd_kernel(N, dtype, output_dtype=None, block_size=2048):
    ...
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅含逐元素向量链（`vabs/vmul/vadd/vrec/vexp/vsub/vcmp/vselect/vcast`），无 matmul、无归约、无 Cube 计算；仅需 UB 层级，不涉及 L1/L0A/L0B/L0C。

### 5.2 Block 划分

```python
block_size = 2048   # K11: 源 threads(256) × npt(8)（fp16/bf16）折叠为单一参数
n_num = T.ceildiv(N, block_size)   # 逻辑 tile 数（编译期静态值）
```

**block_size=2048 选择理由**：
1. 与源一致（K11 折叠 `threads × npt = 256 × 8 = 2048`，fp16/bf16），迁移可追溯；
2. UB 对齐：2048×elem_bytes 是 32B 的整数倍（fp32 8192B、fp16/bf16 4096B）✓；
3. UB 容量：最大 fp32 路径 82 KB（§4.5），×1.12 ≈ 92 KB < 192 KB ✓；
4. **发射项权衡**：本算子 §1.6.0 估算显示**发射项绝对主导**（发射 ∝ 1/block_size），故取较大 block_size=2048（而非源 fp32 的 1024）。**bs=3072/4096 可继续摊薄发射**（4096 减半），但 §4.5 显示 bs=4096 接近 UB 上限——**列为 Stage 4 调优首选杠杆**，须以编译探针校准后采纳（分核参数调优，见 `core-split-strategy.md` §2.4）。

### 5.3 约束分析

- **对齐约束**：`block_size=2048`，fp32 元素 4B → 每 tile 512 向量槽（128bit×4），尾 lane 浪费 0%；fp16/bf16 元素 2B → 256 向量槽（128bit×8），浪费 0%；均满足 32B 尾轴对齐 ✓。
- **UB 容量**：最大 92 KB（fp32，含 1.12x）< 192 KB ✓。
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
| 元素级 | 向量化 | `T.vabs/vmul/vadd/vrec/vexp/vsub/vcmp/vselect/vcast`（21 op） | 整 buffer 一次完成，无 `T.Parallel` 标量循环 |

### 6.2 循环伪代码

见 §3.3（`with T.Kernel(core_num, ...)` + `for i in T.serial(...)` + `if task < n_num`，循环内为 21-op 向量链）。

### 6.3 流水线优化

**当前设计**: 不使用 `T.Pipelined`。理由：单 tile 内数据依赖线性（copy-in → vcast → 21-op 链 → vcast → copy-out），无 K 维迭代；跨 tile 迭代（`T.serial`）的 MTE2/VEC 重叠由 Developer 模式 auto-multi-buffer 自动处理。**发射项主导下**，若 Stage 4 需进一步优化（persistent double buffer / 更大 block_size），参照 `example_elementwise_exp2.py` 的 `T.serial` 多 tile 形态与 `PL-1.10-loads-first-decoupling`（多输入 staging 前置）；设计阶段保持简洁。

### 6.4 尾块处理

见 §5.4。`N % block_size != 0` 时最后一个 tile 用 `tail = T.min(block_size, shape - off)` 动态裁剪 `T.copy` 量；manifest 全部 workload 整除（§5.4 表），仅测试 4,000,000 触发尾块（余 1216）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下编译器自动插入同步。核内数据依赖链严格线性（每 tile 内）：

```
T.copy(GM→UB) → [vcast↑] → vabs → vmul → vadd → vrec → Horner(9) → vmul → vmul → vexp
              → vmul → vsub → vcmp → vselect → vmul → [vcast↓] → T.copy(UB→GM)
```

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy(GM→UB)` 后 | 自动 | 编译器插 MTE2→V 等待 |
| 各 v-op 之间（21 op 链） | 自动 | 线性数据依赖，插 V→V barrier |
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

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（`torch.erf`，保持源 fp32 中转语义），**不得复刻 §0.6 的 NPU 算法**——保证验证独立性。

```python
import torch

def golden_erf(x: torch.Tensor) -> torch.Tensor:
    """Erf 参考实现，等价于 torch.erf（保持源 fp32 opmath 语义）。

    源算子 op_func = T.erf(T.cast(x, "float32"))：对 fp16/bf16 输入先升 fp32
    计算 erf，末端单次舍回原 dtype；fp32 输入直接计算。
    """
    x_f32 = x.to(torch.float32)
    return torch.erf(x_f32).to(x.dtype)
    # 与 torch.erf(x) 在 fp16/bf16 下 bit-exact（torch.erf 自身亦 fp32 opmath）
```

> 备注：`torch.erf` 对 fp16/bf16 亦经 fp32 opmath 计算，故 `golden_erf(x)` 与 `torch.erf(x)` 等价。golden **独立于** NPU 逼近链（不复用 A&S 系数），满足验证独立性。

### 8.2 精度标准

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 1e-5 | 1e-5 | 来源：`examples/TileOPs/tests/ops/test_unary_math.py` `_get_tolerances` |
| float16 | 1e-3 | 1e-3 | 来源：同上；NPU 在 fp32 域计算、单次舍回 |
| bfloat16 | 1.6e-2 | 1.6e-2 | 来源：同上；bf16 精度低，容忍度放宽 |

**误差预算核对（设计期实测）**：A&S 7.1.26 截断误差 ≤1.5e-7（理论）+ fp32 链舍入 ≈ 若干 ulp → 设备实测 max_abs **3.725e-7** ≪ fp32 atol 1e-5（**27× 余量**）；fp16 实测 1.221e-4（<1 ulp@1.0）、bf16 实测 3.1e-5，均远低于门禁。

### 8.3 L0 门槛测试计划

> L0 = 门槛 smoke 测试（最小可用性验证，进 Stage 3 前必过）；完整分层套件 L1/L2/Boundary 交由 `tilelang-op-develop`，不在此枚举。**显式包含 v0 必失败的用例（L0-4 角点、L0-1..3 randn 全域）**，确保本版不重蹈覆辙。

| 用例 | 输入 | 期望 | 判定 |
|------|------|------|------|
| L0-1 fp32 smoke（randn 全域） | `randn([1048576], fp32)`（NPU） | 与 `torch.erf` 比对 ≤1e-5（覆盖 `\|x\|` 至 ~5.5） | 门槛通过 |
| L0-2 fp16 smoke（randn 全域） | `randn([1048576], fp16)`（NPU） | 比对 ≤1e-3 | 门槛通过 |
| L0-3 bf16 smoke（randn 全域） | `randn([1048576], bf16)`（NPU） | 比对 ≤1.6e-2 | 门槛通过 |
| **L0-4 fp32 角点尾段（v0 必失败用例）** | `repeat([0,3,-3,100],[4096], fp32)`（NPU） | 与 `torch.erf` 比对 ≤1e-5（**设计期设备实测通过：`[0.0,0.999978,-0.999978,1.0]` vs golden 逐值一致**） | 门槛通过 |
| L0-5 全域敏感区（v0 必失败区） | `linspace(-8,8,1<<20, fp32)`（NPU） | 与 `torch.erf` 比对 ≤1e-5（**设计期探针实测 max_abs 3.725e-7**） | 门槛通过 |
| L0-6 非整除尾块 | `randn([4000000], fp16)`（NPU） | 比对 ≤1e-3（覆盖 4000000 % 2048 = 1216 尾块） | 门槛通过 |

- **执行方式**：`tilelang.compile(func, target="npuir")` 后单次调用，`torch.allclose(y.cpu(), golden, atol, rtol)`（golden = §8.1）。
- **门槛标准**：L0-1..L0-6 全过即进入 Stage 3；任一失败不得继续。
- 输入生成参考 `examples/TileOPs/tileops/workloads/elementwise.py` 的 `ErfWorkload`（`ShapedRandnWorkload`）；边界用例参考 `test_unary_math.py` 的 `MathEdgeFixture`（`repeat([0.0,3.0,-3.0,100.0])`）。
- **额外设计期落点**：`repro/TRAP-verf-maclaurin-truncation.py`（v0 根因证据）、`history_version/design_probe_erf_ops.py`（新算法设备实测）、`history_version/design_probe_erf_ops2.py`（fp16/bf16 路径实测）、`verify_equiv.py`（D-1 机器验证）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 缓解方案 |
|------|------|----------|
| **R-1 无原生真 erf 元算**（CG-2026-0015，open）：`T.verf`=截断多项式、`T.erf` 段错误 | v0 核心方案证伪；算子上限降低 | 自建 A&S 7.1.26 逼近（§0.6 R2，设备实测达标）；能力缺口已登记；根本解（编译器层修复）本任务外 |
| **R-2 逼近精度余量**：A&S 截断 1.5e-7 + fp32 链舍入 | fp32 门禁 1e-5，余量 27× | 设备实测 max_abs 3.7e-7（分区间 ≤3.7e-7）；`verify_equiv.py` 6/6 PASS；L0-4/L0-5 显式覆盖敏感区 |
| **R-3 发射项主导**（21 op/tile，`CONST-vector-launch-overhead`） | §1.6.0 估算显示 emission 远超 flow（134µs vs 6.2µs @smoke-1m） | 标注「未文档化假设 + 估算依据」（常数绑 910B2C、fixed dispatch、可能上偏）；block_size 为杠杆（§5.2，bs=3072 可行/4096 贴限）；Stage 4 baseline 后按 `hardware-cost-model.md` §3 对账 |
| **R-4 UB 容量**（逼近链 ~10 个 fp32 buffer，41 B/elem） | bs 放大受限，制约发射项优化 | 默认 bs=2048（§4.5，92 KB×1.12）；buffer 复用（t→sign、xa/poly 原地）+ 舍回结果复用 x_ub；bs=4096 须编译探针校准（`TRAP-UB-multibuffer-inflation`） |
| **R-5 bf16 v-prefix 数学 API 缺失**（各 op doc §2.2.1） | bf16 直传编译失败 | bf16 走 fp32 中转（§0.6 R3） |
| **R-6 条件 `T.alloc` 非法**（`TRAP-conditional-alloc`） | 编译期 `if` 内 `T.alloc_ub` trace 失败 | buffer 全部在 `T.Kernel` 顶层无条件分配（§3.3） |
| **R-7 v-op 整 buffer vs 动态 slice**（`PL-1.22` R-6） | 传动态 slice 报错 | v-op 作用于整 buffer，仅 `T.copy` 用 `[0:tail]`（§3.3） |
| **R-8 vcast round_mode 限制**（`T.vcast.md` §2.2.1） | f16→f32 / bf16→f32 **仅 `rint`** | 中转升降统一 `round_mode="rint"`（与 golden 单次舍回一致） |
| **R-9 `vcmp` bool 不可写回 GM**（`T.vcmp.md` §2.3） | 误写回编译失败 | `cond` 仅片上配 `vselect`，从不 `T.copy` 到 GM |
| **R-10 分核残留不均** | 40 核含非 2 因子 5，2ⁿ 逻辑核无法整除 | §5.5 网格跨步核内串行，残留临界 +1.6%（1M）/ +0.24%（16m），已显式记录 |
| **R-11 常数表设备/版本** | `CONST-mte2-degradation`/`CONST-vector-launch-overhead` 绑 910B2C 且同源 `PL-1.6-copy-floor`/`PL-1.17-subfp32-fp32-transit` 为 stale；本设备 910_9362 | 带宽/发射数字标注「待重验」，仅作量级估算；不以 stale 条目为决策依据；Stage 4 baseline 后对账 |
| **R-12 `vrec`/`vdiv`/`vexp`/`vtanh` 设备语义**（承诺与实测一致性） | 文档语义可能与设备不一致（v0 教训） | 全部经目标设备探针实测（§3.4）：精度 1e-7 量级、真 tanh；后续工具链升级须重验 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 用 `T.verf` 算 erf | 沿用 v0 方案 | 输出远超门禁（x=8 → 5.3e4） | 禁用；改自建 A&S 链（R-1） |
| 用 `T.erf` 标量 | 移植 GPU 源形态 | NPU 编译期段错误 EXIT=139 | 禁用（R-1） |
| bf16 直接传入数学 op | 未做 vcast 上转 | 编译失败 | bf16 先 `vcast` 到 fp32（R-5） |
| vcast 用非 rint 模式升级 | f16/bf16→f32 用 `round`/`floor` | 编译失败（仅 `rint`） | 升级/舍回统一 `round_mode="rint"`（R-8） |
| bool `cond` 写回 GM | `T.copy(cond_ub, ...)` | 编译失败（i1 不可 load/store GM） | `cond` 仅片上（R-9） |
| 条件 alloc | 编译期 `if need_cast:` 内 `T.alloc_ub` | trace 报 `Undefined variable` | buffer 顶层无条件分配（R-6） |
| v-op 传动态 slice | `T.vabs(x_ub[0:tail], ...)` | 可能编译/运行错误 | v-op 用整 buffer（R-7） |
| UB 溢出 | block_size 过大（~10 fp32 buffer） | 编译失败 | 维持 bs ≤ 3072（默认 2048）；§4.5 预算 + 编译探针（R-4） |
| 尾块越界 | N 非 bs 整数倍且无 tail 裁剪 | 越界读/写 GM | `tail = T.min(block_size, shape-off)`（§5.4） |
| `t` buffer 复用冲突 | 在 Horner 尚未用完 `t` 时写入 sign | 结果错误 | 仅在 Horner 完成后将 `t` 复用为 sign（§3.3 顺序） |
| 零输入崩溃 | N=0 空张量 | MTE DDR 崩溃（`TRAP-zero-input-crash`） | host 侧空张量守卫；本任务 N ≥ 1e6，不触发 |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| **非整除 N** | Kernel 内 `tail` 动态裁剪（§5.4），无需 host padding |
| **极小 N（N < block_size）** | `n_num=1`，`tail=N`，仅 `task==0` 的核工作，其余核空转 |
| **极大 N（>2.6e8）** | 核内 `T.serial` 网格跨步（§5.5），逻辑核数静态 |
| **`±inf/NaN` 输入** | fp32 链对 inf/NaN 幂等：`erf(±inf)=±1`、`erf(NaN)=NaN`（§0.6 R2 论证 4；`verify_equiv.py` 角点覆盖） |
| **大 `\|x\|` 饱和** | `e^{−x²}` 下溢为 0 → 结果 = sign(x)；fp32 下 `\|x\|≳3.9` 与真值舍入一致（§0.6 R2 论证 3） |
| **`x` 近 0（`y_even≈0`）** | `x=0` 时 `y_even≈1e-9`，符号无关；误差 <atol（§0.6 R2 论证 2） |
| **混合 dtype** | 按 dtype 编译期特化（每 dtype 一份 kernel），输出 `same_as(input)` |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/erf/erf_fwd_kernel/
├── DESIGN.md                       # 本设计文档（revision 1）
├── verify_equiv.py                 # §1.6.1 等价性机器验证脚本（D-1，6/6 EQUIV_PASS）
├── RETROSPECTIVE.md                # Stage 1 复盘（自进化钩子）
├── history_version/
│   ├── design_v0.md                # v0（已证伪）
│   ├── design_probe_erf_ops.py     # 设备探针 1（op 语义矩阵 + A&S fp32 精度）
│   └── design_probe_erf_ops2.py    # 设备探针 2（fp16/bf16 fp32 中转路径）
├── repro/
│   └── TRAP-verf-maclaurin-truncation.py   # v0 根因设备证据（Stage 3 产出）
└── erf_fwd_kernel.py               # 算子实现（Stage 3 产出）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档（revision 1） |
| `verify_equiv.py` | ✅ 已完成 | 6 项等价性机器验证全部 EQUIV_PASS（§1.6.1） |
| `history_version/design_probe_erf_ops.py` | ✅ 已完成 | 设备探针 1（2026-10-12 实跑） |
| `history_version/design_probe_erf_ops2.py` | ✅ 已完成 | 设备探针 2（2026-10-12 实跑） |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘（revision 1 追加） |
| `erf_fwd_kernel.py` | ⬜ 待实现 | 算子实现（Stage 3） |
| `test_erf.py` | ⬜ 待实现 | 分层测试（Stage 3） |

### 10.3 命名规范

- 项目目录名: `erf`；算子目录名: `erf_fwd_kernel`；实现文件: `erf_fwd_kernel.py`（保留源工厂函数名 `erf_fwd_kernel`）；测试文件: `test_erf.py`。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md revision 1）
2. ✅ 设备探针（design_probe_erf_ops.py / ops2.py，新算法可用性实测）
3. ✅ 等价性验证（verify_equiv.py，6/6 EQUIV_PASS）
4. ⬜ Golden 函数（`golden_erf`，验证基准）
5. ⬜ 算子实现（`erf_fwd_kernel.py`：kernel + wrapper + L0 测试）
6. ⬜ L0 门槛测试（L0-1..L0-6：randn 全域 3 档 + v0 必失败角点 + 敏感区 + 非整除尾块）

### 10.5 迁移执行规则遵循确认

| 规则 | 遵循情况 |
|------|----------|
| `@tilelang.jit` 装饰函数命名为 `erf_fwd_kernel` | ✅ 保留源工厂函数名 |
| Kernel 参数语义保持（`threads`/`npt` 折叠为 `block_size`） | ✅ K11：256 × 8 = 2048 |
| `@T.prim_func` 参数名 `x, y` 及顺序不变 | ✅ `main(x, y, shape)`，新增 `shape` 用于尾块 |
| 覆盖全部 dtype（fp16/bf16/fp32） | ✅ fp32 直算；fp16/bf16 fp32 中转（bf16 API 强制中转） |
| 保持源 fp32 中转语义 | ✅ `vcast↑f32 → A&S → vcast↓dtype`；`verify_equiv.py` 门禁内一致 |
| K1–K4 | ✅ `target="npuir"`、`is_npu=True`、移除 `threads=` |
| K8/K9 | ✅ 移除 autotune / `threads` |

### 10.6 相对上一版（design_v0）的关键调整 ⭐（revision 专属）

**本次为「路径 B：实施期设计错误」的修订（revision_index=1）。**

| # | v0 的设计选择 | v1 的关键调整 | 为什么不会再犯同一错误 |
|---|---------------|---------------|----------------------|
| 1 | **核心 API = 原生 `T.verf`**（§1.6.0 采纳候选 1） | **改用自建 A&S 7.1.26 有理-指数逼近**（21-op v-prefix 向量链，§0.6 R2 / §1.6.1 #1） | v0 只核对 `docs/.../T.verf.md` 与源码转译**目标**，未核对 codegen **实际展开式**、也未做**设备实测**。v1 对每个关键 API 一律**在目标设备实测语义/精度**（§3.4 探针表），并复现了截断证据（`repro/TRAP-verf-maclaurin-truncation.py`） |
| 2 | `T.verf` 的「等价替换」判定依据 = 文档 §2.2.1 + §3 | 判定依据 = **设备实测**（`T.verf` 证伪；`vexp/vrec/vdiv/vtanh/vabs` 实测真实可用） | 建立「负向/正向断言均须设备实测或显式标注未文档化假设」的硬约束（§1.6.0 R4、§3.4）；v0 的 `verify_equiv.py` 只做 CPU 域等价（**只证算法域，不证设备实现**），v1 在其上叠加设备探针 |
| 3 | §0.5「`T.erf`→`T.verf` 等价替换」 | §0.5 该项改为「**重新设计（算法替代）**」，并新增「无原生真 erf 元算」判定 | 耦合性判定必须以设备能力为准；能力缺口登记 `CG-2026-0015` |
| 4 | §1.6.0 候选表把 `T.verf` 当唯一算法 | §1.6.0 从零枚举 **9 个同语义算法候选**（基线 + 8 逼近/替代），逐项做 R3/R4 | 落实「调研先于优化、输入算法只是候选之一」；负向淘汰逐项给依据（R4 表） |
| 5 | §1.6.0 复杂性只算 3 条 verf 转译 op | §1.6.0 按 **21-op 实测链**重算发射项（估算下界 134µs @smoke-1m） | 诚实量化「无原生元算」的性能代价，登记为风险 R-3；给出 block_size 杠杆与 Stage 4 对账路径 |
| 6 | §4.5 UB 预算 4 buffer（24 KB） | §4.5 预算 **~10 个 fp32 buffer（82 KB @bs2048）** | 逼近链的固有内存代价显式建模；buffer 复用（`t`→sign、`xa/poly` 原地）已计入 |
| 7 | §8.3 的 L0 角点 `[0,3,-3,100]` 在 v0 方案下**必失败** | L0-4 显式保留该角点 + 新增 L0-5 敏感区 `linspace(-8,8)` | 把 v0 的失分点固化为门槛用例；设计期已设备实测通过（角点逐值一致、全域 max_abs 3.7e-7） |
| 8 | 遗留疑虑「`vtanh` 亦为截断多项式」 | 探针**证伪**：`T.vtanh` Developer 模式为真 tanh（max_abs 1.19e-7） | 不把未验证的疑虑当结论；同一探针体系一次性澄清 |

**「为何不会再犯同一错误」的机制性保证**：① 核心 API 一律经目标设备探针实测（`history_version/design_probe_erf_ops*.py`，含 dtype/语义/精度矩阵）；② 设计采纳项的精度以**设备端到端实测**为准，不依赖文档声称；③ 无法本地满足的能力（真 erf 元算）以能力缺口形式显式登记（`CG-2026-0015`）而非变通掩盖；④ 负向断言全部附设备证据或标注未文档化假设（`negative-claim-evidence.md` 口径）；⑤ 角点/敏感区用例在设计期即设备验证，并固化为 L0 门槛。

---

## 附录 A · Stage 4 调优反馈（`[DESIGN_LIMIT]` 补记，2026-10-13）

> 来源：`perf_opt/perf_feedback.md`（ErfFwdOp / `erf_fwd_kernel`，scenario=optimize，Stage 4）。**本附录为设计文档补记，不修改正文结论**；路由 = 附录补记（archive，不耗修订预算）。

- **反馈结论**：§1.6.0 R4 对候选 7「tanh 型逼近」的淘汰依据（原文「容差内等价但误差 ~1e-3，达不到 fp32 门禁」，公式限定为 `erf(x)≈tanh(αx+βx³)`）**仅在其 2 系数形态下成立**。
- **实测证伪（本机 Ascend910_9362，同会话 `msprof op` 口径）**：改用 `erf(x) ≈ tanh(x · Q(x²))`（Q 为偶多项式，`tanh(odd)=odd` 天然消去全部 sign 处理）后——
  - **k=4（5 系数）**：fp32 `max_abs = 2.98e-6`（门禁 1e-5 → **3.4× 余量**），op 数 **12**（fp32）/ **14**（fp16·bf16），远低于 §1.6.0 R3 记录的下界 21/23；
  - **k=6（7 系数）**：fp32 `max_abs = 6.2e-8`（**>160× 余量**），op 数 16。
  - 逐 op 消融实测：`vselect`（bool 掩码选择）单项 ≈ **6.5×** 普通 v-op，是 A&S 链的最大结构成本；`vrec` 次之；`vexp` 较便宜。
- **结构性加速**：**2.15×–2.75×**（16m f16 2.19× / bf16 2.15× / f32 2.75×；64m ~2.2×；256m f16 2.23× / bf16 2.19×），7/7 tune workload 均 > 2×。
- **对设计结论的刷新**：§1.6.0「选定候选 2 …无更优可行替代」与 op 数下界（21/23）被本次实测刷新为「**op 数下界 ≥ 12（tanh-奇多项式族）**」；**若将来重做本算子设计，应以本附录为输入**（R4 候选 7 不应再以 2 系数形态淘汰）。
- **精度留档**：tanh k=4 路径 fp32 余量由 16.9× 降至 **3.4×**（仍过门禁）；若需更大余量可用 k=6（16 op，仍 2.0× 加速）。
