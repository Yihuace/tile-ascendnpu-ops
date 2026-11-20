# _make_dropout_kernel 算子设计文档（迁移）

> **项目**：`dropout` ｜ **算子目录**：`examples/dropout/_make_dropout_kernel/` ｜ **编程模式**：Developer ｜ **target**：`npuir`
>
> **任务性质**：迁移（migration-harness）。把 GPU TileLang 工厂函数 `_make_dropout_kernel`（inverted dropout，用 `T.rng_init`/`T.rng_rand_float`（cuRAND Philox4_32_10）逐线程生成 uniform[0,1)，`keep = rand >= p`，`y = if_then_else(keep, x*scale, 0)`，`scale = 1/(1-p)`）重设计为 NPU（`target="npuir"`）实现。
>
> **本版核心决策（本算子最关键点，conductor 预探 + 本 Agent 独立复核）**：**`T.rng_init` / `T.rng_rand_float` 在 npuir 工具链上不存在**（见 §1.6.0 R1 复核证据：`dir(T)` 无任何 rng/rand 属性、`docs/Tilelang.language/` 全 16 子目录无 RNG 条目、`customize_npuir.py` 无 rng 算子、全仓 grep 仅命中脚手架文件）。故**源 RNG 算法整体重设计**为**核内 counter-based 无状态 hash**（右移无关 xorshift-left + 回绕乘 + 阈值掩码，int32 向量元算自实现），并由设备探针实测：与 numpy int32 模型逐位一致、top-11 χ²=2004（dof 2047）、lag-1 χ²=1119（dof 1023）、同 seed 逐位重放、异 seed 掩码差异 0.496、全 dtype 全 p 值 drop rate 落在 3σ 内、p=0 恒等 / p=1 全零。
>
> **第二决策（算法级优化）**：掩码施加由源 `if_then_else(keep, x*scale, 0)`（选择式）改为**算术 0/1 掩码乘**（`y = (x*scale) * m`，`m∈{0,1}`，由 `vsub/vmax/vmin` 从阈值比较导出，全程无分支）。二者在有限输入上**逐元素恒等**；在 `±inf/NaN` 角点上，算术掩码乘**与契约 golden（`torch.nn.functional.dropout` 的乘性掩码语义 `x*(m*scale)`，实测 p=1 对 `±inf` 返回 NaN）逐值一致**，而源选择式反而偏离（返回精确 0）。且算术掩码乘避开了 910_9362 上 `vselect` ≈ 6.5x 普通 v-op 的高代价（`PL-1.29-vselect-cost`）。
>
> **第三决策（计算域）**：fp16/bf16 走 **fp32 中转域**（`vcast` 升 → fp32 向量链 → `vcast` 舍回）；bf16 为 API 强制（`vmul`/`vmax`/`vmin`/`vselect` 均不含 bf16）。
>
> **源优化意图承接**：源 #1 on-chip cuRAND RNG → **重设计**为核内 counter-based hash（确定性重放意图由「纯函数 of (seed, 元素索引)」承接）；源 #2 block×thread 两级并行 → 一维 `T.Kernel(40, is_npu=True)` + 核内 `T.serial` 网格跨步；源 #3 索引置换 `(bx*threads+i)*npt+j` → **舍弃**（GPU coalescing/RNG 布局专用；NPU 用自然展平序 + `arange` ramp，掩码分配序差异经契约核对可接受）；源 #4 输入 dtype 域乘 → **等价替换为 fp32 中转**（契约一致 + bf16 强制）；源 #5 `register/fragment` → **等价替换为 UB**；源 #6 选择式计算核 → **等价替换（并优化）为算术掩码乘**；源 #7 `lru_cache` 常量特化 → 保留；源 #8 无 autotune → 保留。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织：先彻底读懂源算子（0.1–0.4），再做算法调研（Phase R，结论见 §1.6.0），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。

**源算子路径（源真相）**：`/home/developer/TileOPs/tileops/kernels/dropout.py`

- 具体算子：工厂 `_make_dropout_kernel(N, dtype, p, seed, threads=256, num_per_thread=8)`（L29–L67）+ 类 `DropoutKernel(Kernel)`（L70–L122）。
- 源关键代码（L45–L65）：

  ```python
  block_size = threads * num_per_thread
  scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0

  @tilelang.jit(out_idx=[1])
  def kernel(threads_arg, npt_arg):
      @T.prim_func
      def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
          with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
              T.rng_init(seed)                              # ★ per-thread RNG state (cuRAND Philox4_32_10)
              for i, j in T.Parallel(threads_arg, npt_arg):
                  idx = (bx * threads_arg + i) * npt_arg + j   # ★ 置换索引
                  rand_val = T.rng_rand_float()                # ★ uniform[0,1)
                  if idx < N:
                      keep = rand_val >= T.cast(p, "float32")
                      y[idx] = T.if_then_else(keep, x[idx] * T.cast(scale, x[idx].dtype),
                                              T.cast(0.0, x[idx].dtype))
      return main
  return kernel
  ```

- 提取产物（脚手架）：`examples/TileOPs/tileops/kernels/regularization/dropout/_dropout_kernels.py`（Pattern B，逐句保留 GPU 原形态）。
- 集成 wrapper（NPU 接口契约）：同目录 `dropout.py`（K11：`_make_dropout_kernel(N, dtype, p, seed)` 返回「接受单一 `block_size` 的可调用」，`forward` 调 `self.kernel(block_size)(x)`；`default_config = {"block_size": 1024·fp32 / 2048·fp16·bf16}`）。

### 0.1 源算子语义（做什么）

**数学语义**：inverted dropout（倒置丢弃）。逐元素

$$
y_i = \begin{cases} x_i \cdot \dfrac{1}{1-p} & \text{keep, prob } 1-p\\[4pt] 0 & \text{drop, prob } p\end{cases},\qquad i=0,\ldots,N-1
$$

（式中：以概率 `1-p` 保留并放大 `1/(1-p)`，以概率 `p` 置 0。）

等价于 `y = x * mask / (1-p)`，`mask ~ Bernoulli(1-p)`（`mask` 与 `x` 逐元素独立）。`p=1` 时 `scale=0`（全部丢弃）。manifest `ref_api = torch.nn.functional.dropout`，`params: p=0.5 / seed=0 / training=true`。

**契约 golden = `torch.nn.functional.dropout`（training=True）**。关键核对（本算子与常规确定性算子的最大区别）：

1. **dropout 是随机算子**——不存在可逐元素复现的解析 golden；测试契约（`examples/TileOPs/tests/ops/test_dropout.py`）由四类性质组成：
   - **统计**：丢弃比例落在 p 的 3σ 内（`test_dropout_statistical_rate`，N=4M）；
   - **确定性重放**：同 `(seed, 实现)` → 逐位相同输出（`test_dropout_deterministic_replay`）；不同 seed → 不同输出（`test_dropout_different_seeds`）；
   - **缩放正确**：非零元素 == `x * 1/(1-p)`，容差按 dtype（`test_dropout_scale_factor`，atol/rtol fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2）；
   - **边界**：`p=0` 恒等、`p=1` 全零、`training=False` 恒等、输出 shape/dtype == 输入。
   故本设计的等价性判据 = **分布等价 + 确定性重放 + 缩放契约**（非「与 GPU mask 逐位相同」——cuRAND 流跨设备不可复现，见 §0.5 / §1.6.0）。
2. **契约 golden 的掩码语义是「乘性」而非「选择式」**（实测）：`torch.nn.functional.dropout` 对 `±inf` 输入在丢弃位置返回 **NaN**（`inf * 0 = NaN`，p=1 实测 `[nan,nan,nan,0.0,-0.0]`），对 `-0.0` 保留 `-0.0`。而源 GPU `T.if_then_else(keep, x*scale, 0)` 对丢弃位置强制精确 `0`。→ 源实现在 `±inf/NaN` 角点与契约 golden **不一致**（见 §0.5 / §1.6.1 #1 决策证据）。

**dtype 语义**：源在**输入 dtype 域**相乘（`x[idx] * T.cast(scale, x.dtype)`），无 fp32 中转。`scale` 为 Python float 常量，编译期特化。

**规约语义**：无规约（逐元素映射，无跨元素依赖），不存在累加顺序问题。

**边界语义**：`p=0` → `scale=1`、`keep` 恒真 → 恒等；`p=1` → `scale=0`、`keep` 恒假 → 全零；`training=False` → host 侧直接返回输入（不经 kernel，manifest 记为短路径）。`±inf/NaN/-0.0` 见上（契约 = torch 乘性掩码）。

**host 侧语义**：`N_total = numel`（任意 shape 展平为 1-D `[N]`）；`training=False` 的恒等、`p` 的合法性校验在 host 完成；`forward` 仅调用编译后 kernel。

**语义保持基线**：§8.1 golden 以本节契约（分布/重放/缩放/边界）为唯一依据实现，**不得复刻 §0.6 的 NPU PRNG 结构**（保证验证独立性）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | `(N,)`（原始任意 shape 被 host 侧展平；manifest `[1024,1024]` / `[1024,4096]` / `[1024,10240]`） | `float16 \| bfloat16 \| float32` | 1-D 展平输入，`N = product(input.shape)` |
| `y` | `same_as_input`（展平后 `(N,)`） | `same_as(x)` | 逐元素 dropout 输出 |

**输出 shape（迁移必需）**：`same_as_input`（manifest `shape_rules: output.shape == input.shape`；`source_output_shape = same_as_input`）。
**layout**：1-D 连续（contiguous flatten），无转置。
**动态轴**：`N` 由 `lru_cache` 按宿主 shape 编译期特化；实测/测试的 N 均为编译期常量。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源，L52–L63）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `T.rng_init(seed)`：每线程初始化 cuRAND 状态（Philox4_32_10，按 block/thread id 派生 key） | 编译期 `seed` + 线程 id | 每线程 RNG 状态 | 掩码随机源（种子固化） |
| 2 | 元素定位：`idx = (bx*threads+i)*npt+j` | grid `bx` + 线程/元素编号 | 全局元素索引 `idx` | 元素寻址（**置换序**，非自然展平序） |
| 3 | `rand_val = T.rng_rand_float()`：线程内顺序推进 RNG，每次产出一个 uniform[0,1) | RNG 状态 | `rand_val` | 掩码采样 `mask_i` |
| 4 | `keep = rand_val >= cast(p, fp32)` | `rand_val`、`p` | 布尔 `keep` | `mask_i = 1{rand_i ≥ p}`（保留概率 1−p） |
| 5 | `y[idx] = if_then_else(keep, x[idx]*cast(scale,dtype), cast(0,dtype))` | `x[idx]`、`keep`、`scale` | `y[idx]` | 保留 → 缩放；丢弃 → 0 |

**数据流与内存访问模式**（源硬件 = GPU）：

```
GM[x] --(coalesced load, per-thread npt 元素)--> registers
   --(T.rng_init: 每线程 cuRAND 状态；T.Parallel(threads,npt) 内逐元素: rng_rand_float → cmp → mul → if_then_else)--> registers
   --(coalesced store)--> GM[y]
（无显式 SMEM；RNG 状态驻留寄存器；block_size = threads × npt）
```

**循环与并行结构**：`T.Kernel(T.ceildiv(N, block_size), threads=256)`——一维 grid（`bx`）+ 256 线程/block；每线程 `npt` 元素（`T.Parallel(threads_arg, npt_arg)`）。`block_size = 256 × npt`：fp32 `npt=4`→1024，fp16/bf16 `npt=8`→2048（`DropoutKernel.default_config`）。

**host 侧逻辑**：`N_total`、`dtype`、`p`、`seed` 由 Op 层传入；factory 经 `functools.lru_cache(maxsize=32)` 按 `(N, dtype, p, seed, threads, num_per_thread)` 特化；`forward` 调 `self.kernel(cfg["threads"], cfg["num_per_thread"])(x)`；任意 shape 由 Op 层展平为 `[N]`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | **on-chip 硬件 RNG**（`T.rng_init`/`T.rng_rand_float`，cuRAND Philox4_32_10） | 高吞吐随机数，免显式表/状态外存 | 每线程寄存器态 Philox；硬件/库实现 | **cuRAND（NVIDIA 专用）** | **硬件强相关**（NPU 无对应原语 → 重设计） |
| 2 | **每线程 RNG 状态 + 线程内顺序推进** | 使同 seed + 同线程布局 → 同 mask（可重放） | `rng_init(seed)` 按 block/thread id 派生 key；线程内顺序消费 num_per_thread 个值 | GPU thread 执行模型 | **硬件强相关**（线程模型 → 重设计为无状态 counter-based） |
| 3 | **置换索引** `(bx*threads+i)*npt+j` | 访存合并 + 掩码随线程布局分布 | 线程 i 的元素在内存中相隔 npt，逐线程连续 npt | memory coalescing | **硬件强相关**（GPU coalescing 布局，NPU 不适用） |
| 4 | **输入 dtype 域乘**（`x[idx] * cast(scale, x.dtype)`） | 免 fp32 往返 cast，半精度原生高吞吐 | scale 常量在输入 dtype 表示，全程半精度乘 | GPU 半精度 mul | **语义/优化层**（须与契约核对；NPU 有 bf16 缺失约束） |
| 5 | **register/fragment 级计算** | 数据驻留寄存器，合并访存 | 逐线程标量 load→compute→store | GPU 大寄存器堆 | **硬件强相关**（NPU → UB 整块向量） |
| 6 | **选择式计算核** `if_then_else(keep, x*scale, 0)`（2 op/元素：cmp + select + mul） | 极低 op 数 | 标量二路选择 | GPU 逐元素 ALU | **算法层**（NPU `vselect` 代价高 → 有更优等价形态） |
| 7 | 单遍逐元素（无中间落回、无重复读） | 访存最小 | 一次读 x、一次写 y | 无硬件依赖 | 可移植 |
| 8 | 工厂 `lru_cache` 按 `(N, dtype, p, seed, …)` 特化 | 消运行时参数分派（p/seed 编译期常量） | compile-time 常量固化 | 无硬件依赖 | 可移植 |
| 9 | 无 autotune（固定启发式 config） | 带宽/发射受限 + 单步，搜索 ROI 低 | `default_config` 固定 | 无硬件依赖 | 可移植 |

> 识别不出来 ≠ 不存在。上述 9 项为源码中可识别的全部优化/规避手段；后续迁移逐项处置（§0.5）。**未识别优化会被静默丢弃**，故此处列全。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 Phase R 调研结论 §1.6.0）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（inverted dropout、I/O、缩放契约、`p=0/1`、`training` host 契约） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 依 §0.1；kernel 不做语义替代 |
| 「同 seed → 同 mask」**可重放契约** | 语义 | 无（契约本身） | — | **保留** | 由核内纯函数 `h(seed, 元素索引)` 承接（无状态、可重放） |
| **随机数算法（cuRAND Philox4_32_10 + 每线程顺序态）** | 算法 | cuRAND 库 + GPU 线程模型 | **无**（npuir 无任何 RNG 原语） | **重新设计** | → §0.6 R2：核内 counter-based 无状态 hash（右移无关 xorshift-left + 回绕乘），设备探针实测统计与重放 |
| 优化 #2 每线程 RNG 状态 / 线程内顺序推进 | 算法/优化 | GPU thread 模型 | 无 | **重新设计** | → §0.6 R2：改为 `idx` 播种的无状态 hash（天然并行、无跨元素顺序依赖） |
| **on-chip 硬件 RNG 的吞吐意图** | 优化 | cuRAND 专用 | 无 | **重新设计（降级）** | 见 §0.6 R2 + §1.6.0 结论：以 ~9-op 向量 hash 近似替代；**性能回退显式记录**（16m fp16 实测 366µs vs 带宽地板 ~63µs） |
| 优化 #3 置换索引 `(bx*threads+i)*npt+j` | 优化 | GPU coalescing / thread 布局 | 无 | **舍弃** | NPU 用自然展平序 + `arange` ramp（`idx = tile_base + lane`）；差异仅「哪个元素配哪个随机值」，契约（统计+自洽重放）不含与 GPU 掩码逐位一致的要求。理由记录于此，防性能/语义回退无法追溯 |
| 优化 #4 输入 dtype 域乘 | 语义/优化 | GPU 半精度 mul | **有**（fp16 ✓） | **等价替换（→ fp32 中转）** | bf16 `vmul`/`vmax`/`vmin` 缺失（API 强制 fp32 中转）；fp16 转 fp32 后对契约 golden 偏差 5.96e-8（≤1e-5，`verify_equiv` #1 证），且掩码乘在统一 fp32 域更稳。见 §0.6 R3 / §1.6.1 #2 |
| 优化 #5 register/fragment | 优化 | GPU 寄存器堆 | 有（UB） | **等价替换** | `T.alloc_ub` 整块 buffer + v-prefix 向量链（§0.6 R1） |
| 优化 #2「大规模并行吃带宽」意图（block×thread） | 优化 | GPU thread/warp | 无直接等价（一维 Kernel） | **重新设计** | → §0.6 R1：一维 `T.Kernel(40, is_npu=True)` + 核内 `T.serial` 网格跨步（§5.5） |
| 优化 #6 选择式计算核 | 算法 | GPU 逐元素 ALU | 有（`vcmp/vselect`） | **等价替换（并优化）** | → §0.6 R4 + §1.6.1 #1：算术 0/1 掩码乘替代 `if_then_else`（免 `vselect` 高价；且对 `±inf/NaN` 更贴合契约 torch 乘性语义） |
| 优化 #7 单遍逐元素 | 算法 | 无 | — | **保留** | 单遍访存下界 `2·N·elem_bytes`（§1.6.0 R3） |
| 优化 #8 `lru_cache` 常量特化 | 优化 | 无 | 有 | **保留** | 工厂保持 `(N, dtype, p, seed)` 编译期常量（p/seed 固化） |
| 优化 #9 无 autotune | 优化 | 无 | 有 | **保留** | NPU 侧同固定单实现 |
| `threads=256` kwarg / `npt` | 优化 | GPU 线程数 | 无效果（`TRAP-threads-kwarg-noop`） | **舍弃** | 移除 `threads=`；`threads×npt` 折叠为 `block_size`（K11，wrapper 契约） |
| `training=False` 短路径 / 输入展平 | 算法（host） | 无 | — | **保留** | host 侧处理（§0.1），kernel 只处理 `[N]` |

> **依据**：NPU 无 RNG 原语由 §1.6.0 R1/R4 的复核证据佐证（`dir(T)` / docs 全子目录 / `customize_npuir.py` / BishengIR `linalg.fill_rng_2d` 未暴露）；`vcast` 中转可用性由 `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 佐证；`vmul/vmax/vmin` 的 dtype 矩阵（fp16·fp32 ✓ / bf16 ✗）由各 op doc §2.2.1 佐证；`vselect` 高代价由 `PL-1.29-vselect-cost`（verified）佐证。全部关键项另有本会话设备探针实测（§3.4）。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（承接源方案性能意图 + 语义保持论证）。

**重设计项 R1：并行结构重设计（block×thread 两级并行 → 一维 Kernel + 核内网格跨步）**

- **源方案**：`T.Kernel(ceildiv(N, block_size), threads=256)` + `T.Parallel(256, npt)`，每线程 `npt` 元素；性能意图 = 大规模并行吃满带宽。
- **NPU 新算法**：一维 `T.Kernel(core_num, is_npu=True)`（`core_num` = 实查物理 Vector 核数 **40**，§5.5 ②）+ 核内 `for i in T.serial(ceildiv(n_num, core_num))` 网格跨步（`task = i*core_num + cid`，`if task < n_num` 守卫）；每 tile 经 `T.copy` GM→UB、整 buffer 向量链、`T.copy` UB→GM。
- **语义保持论证**：任务映射只重排「哪个核处理哪个元素区间」，各元素计算完全独立（无跨元素依赖、无累加顺序变化），逐元素语义恒等；边界：`T.serial` 上界 `ceildiv(n_num, core_num)` 为编译期静态值，守卫覆盖非整除尾块，不越界。
- **模式先例**：`examples/elementwise/vec_add_2d.py` L28–L42；`examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py` L89–L108（`PL-1.22-persistent-1d-grid-stride`）。

**重设计项 R2：随机数算法重设计（cuRAND Philox 状态机 → 核内无状态 counter-based hash）** ⭐（本算子关键）

- **源方案**：`T.rng_init(seed)` 建每线程 cuRAND 状态，`T.rng_rand_float()` 线程内顺序产出 uniform[0,1)；性能意图 = 硬件级高吞吐 + 同 seed 同布局可重放。
- **NPU 新算法**（元素级纯函数，右移无关，int32 向量元算）：

  ```
  # 每元素（lane 级，整 UB buffer 向量化）
  idx = tile_base + lane                 # vbrc(base=off) + vadd(lane, base)  —— 自然展平序索引
  h   = idx ^ seed                       # vxor（seed 为编译期常量 buffer）
  h  ^= h << 13                          # vshl + vxor
  h   = h * 0x9E3779B1  (int32 wrap)     # vmul（设备实测回绕）
  h  ^= h << 7                           # vshl + vxor
  h   = h * 0x85EBCA6B  (int32 wrap)     # vmul
  h  ^= h << 13                          # vshl + vxor
  hm  = h & 0x7FFFFFFF                   # vand —— 取非负 31 bit
  d   = hm - (256*Tp - 1)                # vsub，Tp = round(p·2^23)
  m   = clamp(d, 0, 1)                   # vmax(d,0) + vmin(·,1) —— 0/1 掩码（keep iff (hm>>8) ≥ Tp）
  mf  = cast(m, fp32)                    # vcast
  yf  = (xf * scale) * mf                # vmul ×2（scale=1/(1-p) 或 0）
  ```

  设计要点：① **无状态**（`h` 仅为 `(seed, idx)` 的函数）→ 天然并行、可重放，且**不依赖线程布局**（重放契约更强于源）；② **右移无关**——`vshr`(int32) 设备实测为**算术右移且退化为标量指令**（编译告警），故以「左移 + 回绕乘」构造 avalanche 并用 `hm ≥ 256·Tp ⟺ (hm>>8) ≥ Tp` 的等价变形**完全消除右移**；③ `seed` 编译期常量（工厂 `lru_cache` 特化）；④ 阈值 `Tp = round(p·2^23)` → 丢弃概率 `256·Tp/2^31 = Tp/2^23 ≈ p`，误差 ≤ 2^-24。
- **语义保持论证**：
  1. **统计等价**：设备实测（`design_probe_dropout_rng.py`）hash 与 numpy int32 模型**逐位一致**；top-11 χ²=2004（dof 2047，3σ 阈 2303）、lag-1 χ²=1119（dof 1023，3σ 阈 1204）→ 边际均匀 + 无 lag-1 相关；drop rate 在 p∈{0.1,0.3,0.5,0.7} 全部 3σ 内（fp32/fp16/bf16 三角点一致）。
  2. **重放契约**：`m` 为 `(seed, idx)` 的纯函数 → 同 seed 逐位重放（实测 ✓）；不同 seed 掩码差异 0.496 ≈ 0.5（实测 ✓）。
  3. **边界**：`p=0` → `Tp=0`、`d=hm+1≥1` → `m≡1`、`scale=1` → 恒等（实测 ✓）；`p=1` → `Tp=2^23`、`d≤0` → `m≡0`、`scale=0` → 全零（实测 ✓）。
  4. **与源语义的差异（显式记录）**：源掩码由「每线程顺序 cuRAND 流 + 置换索引」决定，本算法由「元素自然序索引的 hash」决定——**具体哪些元素被丢弃不同**，但契约（统计 + 自洽重放 + 缩放）完全满足。该差异是分布式随机算子的固有性质（跨设备 RNG 流不可逐位复现），不构成语义违约。
- **源优化意图承接**：源 #1「on-chip 硬件 RNG 吞吐」意图 → **放弃**（NPU 无 RNG 原语，被迫软件实现，性能回退显式记录）；源 #2「每线程态 + 可重放」意图 → 由「无状态纯函数 `(seed, idx)`」承接（重放契约更强：与线程布局无关）；源 #3「置换索引」意图 → 舍弃（见 §0.5）。

**重设计项 R3：计算域重设计（输入 dtype 域 → fp32 中转域）**

- **源方案**：`x[idx] * T.cast(scale, x.dtype)`，全程输入 dtype 域；性能意图 = 省一次 fp32 往返 cast。
- **NPU 新算法**：fp32 直算；fp16/bf16 先 `T.vcast(x_ub, xf, "rint")` 升 fp32 → 掩码乘链（fp32）→ `T.vcast(xf, y_ub, "rint")` 舍回。
- **语义保持论证**：bf16 为 **API 强制**（`vmul`/`vmax`/`vmin` 均不含 bf16）；fp16 收敛到统一 fp32 掩码乘域使「mask 施加 + scale 乘」在同一精度域，对契约 golden 偏差实测 5.96e-8（≤ 1e-5，`verify_equiv` #1）与 0.0（bf16 相对输入 dtype 基线）。数学上两域都算出同一 dropout（差异仅浮点舍入路径），契约 tol 远宽于该差异。
- **源优化意图承接**：源 #4「免 cast」意图让位于 bf16 可行性与统一的 fp32 掩码域（+2 vcast for fp16/bf16）。

**重设计项 R4：掩码施加重设计（选择式 → 算术 0/1 掩码乘）** ⭐

- **源方案**：`T.if_then_else(keep, x*scale, 0)`（cmp + mul + select）。GPU 上 3 op 极廉价。
- **NPU 新算法**：`yf = (xf * scale) * mf`（`mf ∈ {0.0, 1.0}` fp32），由 R2 的 `clamp(d,0,1) + vcast` 导出（`vsub + vmax + vmin + vcast + vmul + vmul`，全部廉价向量 op），**无 `vselect`、无分支**。
- **语义保持论证**：
  1. **对契约 golden 逐值一致**（`verify_equiv` #1）：有限域 fp32/fp16/bf16 全 p 值 max|d|=0；`±inf/NaN/-0.0` 角点亦 0 差异（`mask-mul vs GOLDEN(inf/nan)` 全 EQUIV_PASS）——**算术掩码乘恰好复现 torch `native_dropout` 的 `x*(m·scale)` 乘性语义**（丢弃 `±inf` → NaN，保留 `-0.0`）。
  2. **相对源选择式**：有限输入逐元素恒等；`±inf/NaN` 丢弃位源返回精确 0、算术乘返回 NaN——**算术乘更贴合契约**（`verify_equiv` 诊断行：源选择式对 `±inf/NaN` 有 2–3/6 差异）。
  3. **收益**：避开 910_9362 `vselect` ≈ 6.5x 普通 v-op 的掩码选择高代价（`PL-1.29-vselect-cost`）；掩码导出为 `vsub+vmax+vmin+vmul + vmul` 廉价链。
- **源优化意图承接**：源 #6「极低 op 数」意图 → 以 `vsub+vmax+vmin+vcast+vmul` 廉价链承接（较 `vcmp+vselect+vmul` 的无 `vselect` 形态，发射代价更低）。

> 承接说明（源优化意图不丢弃）：§0.4 #1 → R2 重设计（软件 hash，回退记录）；#2 → R2 无状态纯函数（重放更强）；#3 → 舍弃（有理由）；#4 → R3 fp32 中转；#5 → R1/R4 以 UB 整块向量链承接；#6 → R4 算术掩码乘（免 `vselect`）；#7 保留（单遍）；#8 → 工厂常量特化保留（p/seed）；#9 保留。

### 0.7 标杆实现

- 源算子文件（golden 语义依据）：`/home/developer/TileOPs/tileops/kernels/dropout.py`（`_make_dropout_kernel` L29–L67、`DropoutKernel` L70–L122）。
- 源仓测试基准：`examples/TileOPs/tests/ops/test_dropout.py`（统计 3σ / 确定性重放 / 异 seed / 缩放 / `p=0` 恒等 / `p=1` 全零 / `training=False` 恒等 / shape·dtype 保持 / 自定义 config `block_size` 回归）。
- 性能用例：`examples/TileOPs/benchmarks/ops/bench_dropout.py`（manifest 驱动，torch `F.dropout` 基线）。
- manifest：`examples/TileOPs/tileops/manifest/regularization.yaml`（`DropoutOp`；workloads smoke `[1024,1024]` fp32·fp16·bf16、`[1024,4096]` fp16、`[1024,10240]` bf16、`[1024,4096]` fp32；roofline flops=`N if 0<p<1 else 0`、bytes=`2·N·elem_bytes`）。
- golden 以 §0.1 契约（分布/重放/缩放/边界）为依据（§8.1），**不复刻 §0.6 的 PRNG/掩码结构**（保证验证独立性）。
- 同族迁移先例（只读参考）：`examples/leaky_relu/_make_leaky_relu_kernel/`、`examples/elu/_make_elu_kernel/`、`examples/bitwise_not/bitwise_not_fwd_kernel/`、`examples/expm1/expm1_fwd_kernel/`。

---


## 1. 概述

### 1.1 算子名称

`_make_dropout_kernel`（项目 `dropout`，目录 `examples/dropout/_make_dropout_kernel/`）。

### 1.2 功能描述

对 1-D 展平张量逐元素计算 inverted dropout：`y = x * mask/(1-p)`，`mask ~ Bernoulli(1-p)`，语义等价于 `torch.nn.functional.dropout`（training=True）；支持 `float16 | bfloat16 | float32`，输出与输入 shape/dtype 一致。单步纯 Vector 逐元素算子，无规约、无矩阵运算、无跨元素依赖、无超越函数。**唯一特殊点**：需确定性可重放的随机掩码（源用 cuRAND；NPU 用核内 counter-based hash 重设计）。

**迁移决策后的 NPU 侧算法与源算法的关键差异**：① 随机源由 cuRAND 每线程顺序态改为**核内无状态 counter-based hash**（`(seed, 元素索引)` 的纯函数）；② 掩码施加由 `if_then_else` 选择改为**算术 0/1 掩码乘**（免 `vselect`，且更贴合 torch 乘性语义）；③ 计算域由输入 dtype 域改为 **fp32 中转域**（bf16 API 强制）；④ 并行/内存结构由 GPU fragment+thread 改为 UB + 一维核 + 网格跨步；⑤ 索引由置换序改为**自然展平序**。逐元素 dropout 语义与源一致。

### 1.3 数学公式

$$
y_i = \begin{cases} x_i \cdot \dfrac{1}{1-p} & \text{keep, prob } 1-p\\[4pt] 0 & \text{drop, prob } p\end{cases},\qquad \text{equiv. } y = x \odot \operatorname{mask} \cdot \frac{1}{1-p},\ \ \operatorname{mask}_i\sim\operatorname{Bernoulli}(1-p)
$$

（式中：以概率 `1-p` 保留并放大 `1/(1-p)`，以概率 `p` 置 0；`equiv.` 为等价表达。）

NPU 计算核（§1.6.1 优化后，全 dtype 在 fp32 计算域，末端单次舍回）：

$$
\text{mask}_i = \operatorname{clamp}\!\big(hm_i - (256\,T_p - 1),\ 0,\ 1\big),\quad
hm_i = h_i \,\&\, \texttt{0x7FFFFFFF},\quad
h_i = \operatorname{mix}\!\big(\text{idx}_i \oplus \text{seed}\big),\quad
T_p = \operatorname{round}(p\cdot 2^{23})
$$
$$
y_i = \big(x_i \cdot \tfrac{1}{1-p}\big)\cdot \text{mask}_i
$$

其中 `mix` = 右移无关 xorshift-left/回绕乘（§0.6 R2）——`h` 是 `(seed, idx)` 的确定性纯函数（可重放）。

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法（§0.6），与源算法在结构上等价、在随机源/掩码施加/并行/内存实现上有差异：**

- **随机源**（§0.6 R2）：核内 counter-based 无状态 hash，输入 = `(seed, 元素自然序索引)`。源为 `T.rng_init`+`T.rng_rand_float` 的每线程顺序态（cuRAND Philox4_32_10）——NPU 无该原语（§1.6.0 R1）。
- **掩码施加**（§0.6 R4）：`yf = (xf*scale) * mf`，`mf∈{0,1}` fp32（由阈值比较经 `vsub/vmax/vmin/vcast` 导出），**无 `vselect`、无分支**；源为 `if_then_else`。
- **计算域**（§0.6 R3）：fp32 直算；fp16/bf16 先 `vcast(rint)` 升 fp32 → 链 → `vcast(rint)` 舍回；源为输入 dtype 域。
- **编译期特化**：`p`/`seed` 为编译期 Python 常量（源 `lru_cache` 同源），阈值 `Tp`、`scale`、`seed` 常量 buffer 均编译期固化。

除上述外，算法与 §1.6.0 调研选定的算法族（单步逐元素 + 核内 counter-based PRNG）一致：一趟扫描、无中间 GM 往返、无跨元素依赖、访存为理论下界。

### 1.5 数据流图

```
fp32 路径（xc = x_ub）：
GM[x] --T.copy--> UB[x_ub] --[idx → hash → mask → vcast → ×scale → ×mask]--> UB[x_ub] --T.copy--> GM[y]

fp16/bf16 路径（fp32 中转，xc = xf）：
GM[x(dtype)] --T.copy--> UB[x_ub(dtype)] --T.vcast(rint)--> UB[xf(fp32)]
     --[idx → hash → mask → vcast → ×scale → ×mask]--> UB[xf(fp32)] --T.vcast(rint)--> UB[y_ub(dtype)] --T.copy--> GM[y]
```

掩码支路（与数据支路并行、只依赖 idx）：
```
lane=arange(bs) ─┐
                 ├─ vadd → idx ─ vxor(seed) → [xorshift-left ×2 + 回绕乘 ×2] → h
base=vbrc(off) ──┘                                                              ↓ vand(0x7FFFFFFF)
                                          mf = vcast( vmin(vmax(vsub(hm, tp), 0), 1) ) ← hm
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（对同一数学语义的算法族回答调研四问，选定算法族），再在数学等价前提下优化公式，再审视循环/标量计算的向量化替代。迁移任务：1.6.1–1.6.3 的分析对象为 §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 本算子表面是「单步逐元素」，但含两个算法族子层：**(A) 随机掩码生成**（RNG 算法族，本算子真正难点）与 **(B) 掩码施加/缩放**（0/1 掩码算术族）。二者均按完整调研深度展开（存在多种等价/近似等价形态且有硬件能力分歧）。

**R1 等价化简 / 替代方案候选表**（等价化简公式候选，基线在表内；正式等价论证与收益量化在 §1.6.1）：

子层 A（随机掩码生成）：

| # | 候选（RNG 族） | 结构 / 公式 | 等价性初判 | 收益 / 代价方向（op/元素） | 纳入 R3/R4 |
|---|---------------|------------|-----------|--------------------------|-----------|
| **A0** | **基线：源 cuRAND Philox4_32_10** | `T.rng_init(seed)` + 每线程 `T.rng_rand_float()` 顺序态 | 迭代定义式（mask~Bernoulli）；流跨设备不可复现 | RNG 由硬件/库承担（≈0 显式 op） | ❌ **不可用**（npuir 无 RNG 原语） |
| A1 | **host 侧物化 mask 后传入** | host 生成 mask 张量，kernel 只做 `load mask + select` | 分布等价（host 高质量 RNG）；但 RNG 计算被移出 kernel | kernel 省 ~9 op；**+N·eb_mask GM 流量** | ❌ **否决**（见下） |
| A2 | **BishengIR `linalg.fill_rng_2d`**（counter-based LCG，2 mul + 2 add） | 编译器 named op，`min/max/seed` + 元素索引播种 | 分布等价（LCG，文档声明「不保证分布」） | kernel ≈4 op | ❌ **前端不可达**（CG-2026-0021） |
| A3 | **counter-based 纯 LCG**（单 mul + 单 add） | `h = a·idx + c mod 2^32`，取高位 | 分布等价性弱（晶格结构，连续 idx 相关） | 2 op | ✅（评估，见 R4） |
| A4 | **counter-based xorshift32**（3×(右移+xor)） | 经典 xorshift32 | 分布好（最大周期 2^32−1） | 6 op，但**依赖 `vshr`** | ✅（评估，见 R4） |
| **A5** | **counter-based 右移无关 hash（采纳）** | `idx^seed` → 3×(左移+xor) 与 2×(回绕乘) 交错 | 分布等价（设备实测 χ²/lag-1 达标、与 numpy 逐位一致） | **~9 op（全向量）** | ✅ **采纳** |

子层 B（掩码施加 / 缩放）：

| # | 候选（掩码族） | 结构 / 公式 | 等价性初判 | 收益 / 代价方向 | 纳入 R3/R4 |
|---|---------------|------------|-----------|----------------|-----------|
| **B0** | **基线：源 `if_then_else` 选择式** | `y = keep ? x·scale : 0` | 有限域等价 golden；`±inf/NaN` 丢弃位返回 0，**与 torch 乘性 golden 不一致** | cmp + select + mul（`vselect` 6.5x） | ✅（基线） |
| **B1** | **算术 0/1 掩码乘（采纳）** | `y = (x·scale)·m`，`m=clamp(d,0,1)` | 有限域恒等；`±inf/NaN` 与 torch 乘性 golden **一致** | `vsub+vmax+vmin+vcast+vmul+vmul`（无 `vselect`） | ✅ **采纳** |
| B2 | `vcmp + vselect` 选择式（NPU 直译） | `y = vselect(cond, x·scale, 0)` | 同 B0（精确 0） | `vselect` 高价 | ✅（弃选，§1.6.1 #1） |
| B3 | `vcmp + vcmp→vand→vselect`（掩码张量化，VP-2026-0014 形态） | 多条掩码链 | 同 B0 | op 数更多 | ❌（无增益） |

> **子层 A 候选 A1（host 侧物化 mask）否决依据**（负向断言举证）：`.agents/skills/tilelang-op-design/references/ascend-constraints.md` §4「Host 侧输入操作约束」——「算子的**所有核心计算逻辑**（数据搬运、数学运算、归约、维度重排、padding 等）必须在 `@tilelang.jit` 装饰的 kernel 函数内部完成」，随机掩码生成属核心计算，不得移出 kernel；且额外 `+N·eb_mask` GM 流量（fp16 mask 时总流量 3N·eb vs 下界 2N·eb，+50%）。
> **子层 A 候选 A2 不可达依据**（负向断言举证）：`build/bishengir/dialects/linalg/opdsl/ops/core_named_ops.py` L1814–1845 定义 `fill_rng_2d`（LCG named op），但 `tilelang/language/customize_npuir.py`（全 2010 行）与 `docs/Tilelang.language/` 全 16 子目录均无任何映射/条目 → TileLang npuir 前端不可达（登记 CG-2026-0021）。

**R2 在线算法**：**无在线化收益/不适用**——逐元素映射无跨元素依赖，本身即单遍（§0.3 无循环携带依赖），不存在可被「单遍/流式」消去的多遍基线；结构判据：本算子无 running 统计量（无归约/无 max/sum），在线变体定义不适用。依据：源码即单遍直写；`examples/elementwise/` 全系逐元素实现单遍；`.agents/skills/tilelang-op-design/references/algorithm-candidates.md` 的 `ALG-elementwise` R2 记「逐元素映射天然单遍」。**RNG 视角**：源 cuRAND 是「顺序/流式」态，但那是实现手段；本设计改为无状态 counter-based（比顺序态更适合向量化，且重放契约更强）。

**R3 复杂度对比**（四口径；本算子 **发射/向量吞吐受限**，op 数与访存/遍数并列为主导项）。口径说明：FLOPs 以「每元素向量 op 数」计（无超越函数，1 op = 1 单元操作）；访存 = GM 流量（输入读 + 输出写 + 中间往返）；扫描遍数 = GM 读取次数；中间缓冲 = UB 驻留峰值（均 O(block_size) 单 tile）。

| 算法候选 | op 数/元素（fp32 / fp16·bf16） | 访存 Bytes | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------------------------------|-----------|---------|-------------|---------------------|
| **A0 基线（cuRAND + B0 select）** | ~1（RNG 硬件）+ 3（cmp/select/mul） | 2·N·eb | 1 | 寄存器 O(npt) | GPU 规模并行；不含于本设备 |
| A1 host mask + select | ~3（load/cmp/select）+2 vcast | **3·N·eb**（+mask 读） | 1 | O(tile) | 同下（但流量 +50%） |
| A2 `fill_rng_2d`（若可达）+ B1 | ~4（LCG）+ 6（掩码/缩放） | 2·N·eb | 1 | O(tile) | 同下 |
| A3 纯 LCG + B1 | ~2（LCG）+ 6 + vand/vshl? | 2·N·eb | 1 | O(tile) | 相关性风险（见 R4） |
| A4 xorshift32 + B1 | ~6 +1 seed +3 idx + 6 = 16 | 2·N·eb | 1 | O(tile)（~10 int32 buffer） | 均匀多核，无跨核归并；**`vshr` 退化标量** |
| **A5 右移无关 hash + B1（采纳）** | **~9 +1 seed +3 idx +6 = 19（fp32）/ 21（fp16·bf16，+2 vcast）** | **2·N·eb（下界）** | **1** | O(tile)（~10 int32 buffer，实测 bs=2048 可编译） | 均匀多核，无跨核归并 |
| B2 `vcmp+vselect`（替代 B1） | A5 的掩码段 6→3，但含 `vselect`（6.5x） | 2·N·eb | 1 | 同 | 同（但发射代价更高） |

> **口径注**：全部候选访存同为理论下界 `2·N·eb`（A1 除外，+mask 读），扫描遍数 = 1；差异集中在 **op 数（发射/向量吞吐成本）**、**掩码选择的单 op 代价（`vselect` 6.5x）** 与 **分布质量**。基线 A0 的 RNG 由硬件承担（≈0 显式 op）——这是本迁移无法在 NPU 复现的核心落差（§1.6.0 结论 / §9 R-1）。

**R4 硬件亲和性评估**（逐候选对照检查清单；容量/带宽淘汰引用 `constants.md` 条目）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|------|-------------|---------|------------|---------|------------|------|
| **A5+B1（采纳）** | Vector（`vxor/vshl/vmul(回绕)/vand/vsub/vmax/vmin/vcast/vmul` 全向量；`vshr` 被消除） | 小（~10 int32 buffer + 4 其它，bs=2048 实测可编译，§4.5）✓ | 尾轴 32B 对齐 ✓；v-op 作用于整 buffer | N 编译期常量 ✓ | 单步无多阶段；跨 tile 重叠由 auto-multi-buffer 处理 | ✅ **采纳** |
| A3 纯 LCG | Vector（`vmul`+`vadd`） | 更小 | ✓ | ✓ | 同 | ❌ 分布质量弱（LCG 晶格；连续 idx 高位相关），3σ 统计有风险 → 不采纳 |
| A4 xorshift32 | Vector（但 **`vshr` 设备实测：算术右移 + 退化为标量指令**，编译告警 "will execute by scalar instruction with low efficiency"） | 同 | ✓ | ✓ | 同 | ❌ `vshr` 算术语义破坏 hash 分布 + 标量退化 → 不采纳 |
| A1 host mask | Vector | 小 | ✓ | 需要 host↔device 同步 + 额外流量 | 无 | ❌ 违反 host 计算约束 + 流量 +50% |
| A2 `fill_rng_2d` | Vector（LCG） | 小 | ✓ | ✓ | 同 | ❌ 前端不可达（CG-2026-0021） |
| B2 `vcmp+vselect` | Vector（`vselect` 6.5x，`PL-1.29-vselect-cost`） | 同 | ✓ | ✓ | 同 | ❌ 掩码段代价高于 B1（且 `±inf` 语义偏离契约） |
| A0 基线 cuRAND | — | — | — | — | — | ❌ npuir 无 RNG 原语（`dir(T)` 无 rng 属性） |

**v-prefix 元算 dtype 矩阵（决定 bf16 必经 fp32 中转与掩码可选 op）**：`vmul`/`vmax`/`vmin`/`vselect` **不含 bf16**（`docs/Tilelang.language/数学操作/T.vmul.md`·`比较操作/T.vmax.md`·`比较操作/T.vmin.md`·`条件操作/T.vselect.md` §2.2.1：fp16 ✓ / fp32 ✓ / bf16 ✗）；`vcast` 支持 `f16→f32`、`bf16→f32`、`f32→f16/bf16`（均 `rint`）。`vxor/vand/vshl` int32 ✓；`vshr` int32 为**算术右移 + 标量退化**（设备实测）；`vmul` int32 **回绕 mod 2^32**（设备实测）。

**RNG 原语复核（conductor 预探的独立复核）**：

| 复核项 | 命令 / 路径 | 结论 |
|--------|------------|------|
| `T` 有无 rng/rand 属性 | `python -c "import tilelang.language as T; print([a for a in dir(T) if 'rng' in a.lower() or 'rand' in a.lower()])"` | `[]`（**不存在**） |
| 前端 RNG 算子 | `tilelang/language/customize_npuir.py`（全 2010 行） | 无 `rng`/`random` 算子 |
| 文档 RNG 条目 | `docs/Tilelang.language/` 全 16 子目录枚举 | 无 RNG/随机子目录或条目 |
| 全仓 rng 命中 | grep `rng_init\|rng_rand_float` | 仅命中脚手架 `examples/TileOPs/.../dropout*`（源拷贝）+ kb_search 日志 —— 无实现 |
| 编译器中立能力 | `build/bishengir/.../core_named_ops.py` L1814 `fill_rng_2d`；`HIVMOps.td` 无 rng | 编译器有 LCG named op，**前端不可达**（CG-2026-0021） |

→ **结论：`T.rng_init`/`T.rng_rand_float` 不可用**；采用 **A5（核内右移无关 counter-based hash）**。

**调研结论**：**选定「A5 核内无状态 counter-based hash + B1 算术 0/1 掩码乘 + fp32 中转计算域」**；无更优可行替代。

关键依据：① R3 全部可行候选访存同为下界 `2N·eb`、单遍，差异在 op 数/掩码单 op 代价/分布质量——A5+B1 在可行方案中给出**已实测达标的分布质量**（χ²/lag-1/3σ 全过）与**最低掩码段代价**（免 `vselect`）；② R4 逐候选淘汰：A1（违反 host 约束 + 流量）、A2（前端不可达）、A3（分布弱）、A4（`vshr` 算术+标量退化）、B2（`vselect` 高价 + `±inf` 语义偏离）、A0（原语缺失）；③ 选定算法全部 op（`vxor/vshl/vmul/vand/vsub/vmax/vmin/vcast`）均在**目标设备实测**为真实可用元算（§3.4 探针表）；④ 选定 hash 与 numpy int32 模型**逐位一致**（设备验证）。

**调研范围**（「无更优替代」依据）：① 参考表 `.agents/skills/tilelang-op-design/references/algorithm-candidates.md`——本仓算法候选库**无 dropout/RNG 专门条目**（`kb_search "dropout RNG random mask"` 命中 `CASE-elementwise-expm1-migration`、`VP-2026-0014`（掩码向量链）、`PL-1.29-vselect-cost` 等间接条目）；② 本仓 `examples/` 全系（`elementwise/*`、`leaky_relu`、`elu`、`expm1`、`erf`、`bitwise_not`、`argmax`）：**无任何 RNG/随机算子先例**；③ pattern-library：`kb_search` 命中 `PL-1.29-vselect-cost`（掩码选择代价）、`PL-1.22-persistent-1d-grid-stride`（一维网格跨步）、`PL-1.23-vop-chain-inplace-alias`（v-op 链原地复用）、`PL-1.19-devmode-1d-runtime-shape`（一维 kernel 落盘形态）、`constants.md`（容量/带宽/发射常数）；④ **源算子及依赖库**（§0.3/§0.4）：源 RNG 为 cuRAND Philox4_32_10（`T.rng_init`/`T.rng_rand_float`）——「该算子族存在什么算法」的直接证据，但依赖 NVIDIA 专用能力；⑤ **外部已知算法族**（模型知识）：counter-based PRNG（Philox/threefry、xorshift、LCG、splitmix/murmur 终结器、Squares）——均为 `f(seed, counter)` 无状态形态，适合向量化；BishengIR `fill_rng_2d` 的 LCG 实现印证「counter-based per-element seed」是编译器层认可路线；⑥ 前端不可达的 A2 已登记能力缺口（CG-2026-0021），后续工具链若暴露则回访评估去绕化。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)；纯 Vector 逐元素豁免容量项）**：

- **流量项** = `bytes ÷ 有效带宽`，`bytes = 2·N·eb`（manifest roofline 同口径）。引用 `CONST-ascend910-9362-profile` / `CONST-mte2-degradation`〔9362 注记〕：9362 单 die 核→GM ≈ **1.06 TB/s**、HBM ≈1.64 TB/s（绑定 Ascend910_9362；`kb_stale_check` 对 910B2C 曲线条目标 stale，**跨 SoC 沿用待重验**）。16m fp16：`bytes = 67.1 MB ÷ 1.06 TB/s ≈ 63 µs`。
- **发射项** = `op 数 × 单 op 边际代价`。引用 `PL-1.29-vselect-cost` / `CONST-vector-op-cost-910_9362`（910_9362 实测：普通 v-op ≈ **9.15 µs / 16m 元素**，`vselect` ≈ 59.1 µs，`vexp` 13.8，`vrec` 22.9；绑定 910_9362 / CANN 9.0.0 / 2026-10-13）。本设计 op 数：fp32 `k=19`、fp16·bf16 `k=21`（+2 vcast）→ 16m 发射项 ≈ `19×9.15 ≈ 174 µs`（fp32）/`21×9.15 ≈ 192 µs`（fp16·bf16）。
- **容量项** = 0（纯 Vector，无跨引擎往返）。
- **实测校准**（本会话设备实测，§3.4）：16m fp16 ≈ **366 µs**、fp32 ≈375 µs、bf16 ≈402 µs；1m fp32 131 µs / fp16 125 µs / bf16 129 µs。→ 实测 ≈ 发射项估算同量级（估算略偏低，因 copy floor 与多 buffer 搬运未计入估算），**发射/向量吞吐主导**（约为流量地板 63 µs 的 5.8×）。

| workload | dtype | N | block_size | n_num | bytes | 流量项（@1.06TB/s） | 发射项（k×9.15µs/16m） | **实测 / 估算下界 max** |
|----------|---|------|-----------|-------|-------|---------------------|------------------------|--------------------------|
| smoke `[1024,1024]` | fp32/fp16/bf16 | 1,048,576 | 1024 | 1024 | 4.2/8.4 MB | ≈ 4–8 µs | ≈ 11–12 µs | **实测 125–131 µs** |
| `[1024,4096]` | fp16 | 4,194,304 | 2048 | 2048 | 16.8 MB | ≈ 16 µs | ≈ 48 µs | **估算 ≈ 173 µs** |
| `[1024,10240]` | bf16 | 10,485,760 | 2048 | 5120 | 41.9 MB | ≈ 40 µs | ≈ 120 µs | **估算 ≈ 293 µs** |
| `[1024,4096]` | fp32 | 4,194,304 | 1024 | 4096 | 33.6 MB | ≈ 32 µs | ≈ 44 µs | **估算 ≈ 180 µs** |
| 16m（调优基准确认） | fp16 | 16,777,216 | 2048 | 8192 | 67.1 MB | ≈ 63 µs | ≈ 192 µs | **实测 366 µs** |

**设计期估算（D-2）结论**：代表性 `16m fp16` 实测 **366 µs**（估算下界 ≈ 192 µs，流量地板 63 µs）——**发射/向量吞吐主导，约为带宽地板的 5.8×**。根因：软件 PRNG（~19–21 op）取代了源硬件 RNG（≈0 op），这是本迁移的**固有性能落差**（§9 R-1）。

> **估算不确定性声明（未文档化假设 + 估算依据）**：① 单 op 边际代价 9.15 µs/16m 取自 `PL-1.29`（910_9362 实测，同设备，可信度高），但该值是「同结构消融」边际，链内 op 的实际重叠/流水可能使其上下浮动；② 1M 实测（125–131 µs）远高于按 16m/16 的线性外推（≈23 µs）→ **小 N 由固定开销（~40 核启动 + 多 buffer 首末搬运）主导**，故 `[1024,4096]`/`[1024,10240]` 的插值估算偏保守；③ 9362 核→GM 1.06 TB/s 引自 `CONST-ascend910-9362-profile`（值可信；`CONST-mte2-degradation` 的带宽*曲线*绑 910B2C 且 stale，仅作背景）；④ Stage 4 baseline 采集后按 `hardware-cost-model.md` §3 对账并回填。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（A5 hash + B1 算术掩码乘 + fp32 中转 + 编译期 p/seed 特化）。以下各项完成「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；采纳项的等价性经 `verify_equiv.py` 机器验证（结果表见本节末）。

| # | 优化项 | 原式（基线） | 优化后公式（采纳） | 等价性论证 | 收益量化 |
|---|--------|------|-----------|-----------|---------|
| 1 | **算术 0/1 掩码乘**（采纳） | `vcmp + vmul + vselect`（`y = keep ? x·scale : 0`，`vselect` 6.5x） | `mf = vcast(clamp(hm − (256Tp−1), 0, 1))`；`y = (x·scale)·mf` | 有限域逐元素恒等（`m∈{0,1}`）；`±inf/NaN/-0.0` 与契约 golden（torch `x*(m·scale)`）**逐值一致**（`verify_equiv` #1/#2 全 EQUIV_PASS）；源选择式在 `±inf/NaN` 反而偏离契约 | 掩码段由 `vselect`(6.5x)+`vcmp` 降为 `vsub+vmax+vmin+vcast`（廉价链）；16m 省 ≈ 59.1−4×9.15 ≈ **22 µs** |
| 2 | **fp32 中转计算域**（采纳） | 源：输入 dtype 域算 `x·cast(scale,dtype)` + 输入 dtype 域掩码 | `vcast↑f32(rint) → fp32 链 → vcast↓dtype(rint)` | bf16 **API 强制**（`vmul/vmax/vmin` 无 bf16）；fp16 对输入-dtype 基线 max\|d\|=5.96e-8 ≤ 门禁 1e-5（`verify_equiv` #3），bf16 = 0 | 统一 fp32 掩码域；代价 +2 vcast（fp16/bf16） |
| 3 | **右移无关化**（采纳） | xorshift32 的 `h ^= h>>17`（需 `vshr`：**算术右移 + 标量退化**） | `hm ≥ 256·Tp ⟺ (hm>>8) ≥ Tp`（阈值放大），hash 用「左移 + 回绕乘」构造 avalanche | 代数恒等：`hm≥0` 时 `⌊hm/256⌋ ≥ Tp ⟺ hm ≥ 256·Tp`；左侧整数除/取整方向已核对（err ≤ 2^-24）。hash 分布经 χ²/lag-1 实测达标（与 A4 xorshift32 同级质量，`verify_equiv` #4/#5） | **消除全部 `vshr`**（避免算术语义错误 + 标量退化性能陷阱，`TRAP` 级）；−0 op（用回绕乘补 avalanche，见 #5） |
| 4 | **阈值概率精确化**（采纳） | `keep = rand ≥ p`（浮点比较） | `keep ⟺ (hm>>8) ≥ Tp`，`Tp = round(p·2^23)` | 丢弃概率 `= 256·Tp/2^31`，与 p 误差 ≤ 2^-24（`verify_equiv` #8：p=0.1/0.3/0.5/0.7 全 ≤ 2^-24） | 免浮点比较与浮点乘；纯整数阈值比较 |
| 5 | **CSE / buffer 复用**（采纳） | 各中间量独立 buffer | `hm` 复用 `t`（hash 后 `t` 空闲）；`idx` 复用为 `h`；`seed/m31/zerob/tp` 常量 buffer 循环外一次性 `T.fill` | 数学恒等（中间量生命期结束后复用，不改数值）；链尾原地写回先例 `PL-1.23-vop-chain-inplace-alias` | 减 2 个 int32 buffer（§4.5）；常量 `T.fill` 从每 tile 降为每 kernel 一次 |
| E1 | **大 block_size（如 4096）**（评估，不默认采纳） | `block_size=2048`（K11 折叠值） | `block_size=4096` | 数学无关（仅 tiling）；UB 预算须按 §4.5 放大核对 | 发射 ∝ 1/block_size，可能受益；列为 Stage 4 A/B 杠杆（§5.2） |
| E2 | **B2 `vcmp+vselect` 形态**（拒绝） | B1 算术掩码乘 | `vselect(cond, x·scale, 0)` | 与源选择式同（`±inf/NaN` 偏离契约，非采纳） | 计入 `vselect` 6.5x，无收益——**不采纳** |
| E3 | **A4 xorshift32（含 `vshr`）**（拒绝） | A5 右移无关 hash | xorshift32 | hash 分布可能等价，但 `vshr` 算术语义 + 标量退化使正确性/性能均劣 | **不采纳**，理由记录于此 |

**优化结论**：采纳 5 项（#1 算术掩码乘、#2 fp32 中转、#3 右移无关化、#4 阈值概率精确化、#5 CSE/buffer 复用）。**公式层面已无进一步计算量/访存量可优化空间**：访存固定在理论下界 `2N·eb`（1 读 1 写、无中间 GM 往返）；掩码段已是免 `vselect` 的最廉价整数链；唯一「额外」成本是 fp16/bf16 的 2 次 `vcast`（bf16 强制、fp16 为统一掩码域所需）与 PRNG 的 ~9 op（§1.6.0 固有落差，无更优可行替代）。**优化后公式**（供 §3.1 使用）：
- `float32`：`hm = (mix(idx ^ seed) & 0x7FFFFFFF); y = (x * scale) * cast(clamp(hm - (256Tp-1), 0, 1))`
- `float16 / bfloat16`：同上，但 `x` 先 `vcast` 升 fp32、`y` 末端 `vcast` 舍回。

**等价性机器验证（D-1）**：脚本 `examples/dropout/_make_dropout_kernel/verify_equiv.py`（torch CPU / numpy；fp64 参照；候选式 = NPU 链的 numpy int32 精确模型 + fp64 数值，基线式 = 源选择式 / 输入-dtype 算术，契约 golden = torch 乘性掩码语义）。运行 `python3 verify_equiv.py`，**结果：FAIL items = 0 → ALL EQUIV_PASS**：

| 采纳项 / 对照 | 候选式 | 基线式 / 契约 | samples | 最大 abs 差 | 违反比 | 结论 |
|---------------|--------|---------------|---------|------------|--------|------|
| #1 算术掩码乘 vs GOLDEN（fp32，p∈{0.1,0.3,0.5,0.7,1.0}） | `(x·scale)·m` | torch 乘性 golden | 5×1,048,586 | 0.000e+00 | 0 | **EQUIV_PASS** |
| #1 算术掩码乘 vs GOLDEN（fp16，全 p） | 同上 | torch 乘性 golden | 5×1,048,586 | 0.000e+00 | 0 | **EQUIV_PASS** |
| #1 算术掩码乘 vs GOLDEN（bf16，全 p） | 同上 | torch 乘性 golden | 5×1,048,586 | 0.000e+00 | 0 | **EQUIV_PASS** |
| #1 算术掩码乘 vs GOLDEN（`±inf/NaN` 角点，3 dtype×2 p） | 同上 | torch 乘性 golden | 6×6 | 0.000e+00 | 0 | **EQUIV_PASS** |
| #2 fp32 中转 vs 输入-dtype 基线（fp16, p=0.5） | fp32 域链 | 输入 dtype 域 | 1,048,576 | 5.960e-08 | 0 | **EQUIV_PASS** |
| #2 fp32 中转 vs 输入-dtype 基线（bf16, p=0.5） | fp32 域链 | 输入 dtype 域 | 1,048,576 | 0.000e+00 | 0 | **EQUIV_PASS** |
| #3/#4 PRNG 均匀性 top-11 χ² | A5 hash | 均匀分布 | 4,194,304 | χ²=2004（dof 2047，3σ 阈 2303） | — | **EQUIV_PASS** |
| #3 PRNG lag-1 独立性 χ²（top5×5） | A5 hash | 独立均匀 | 4,194,303 | χ²=1119（dof 1023，3σ 阈 1204） | — | **EQUIV_PASS** |
| 重放契约 | A5 hash 纯函数 | 同 seed 二次计算 | 65,536 | 逐位相同 | 0 | **EQUIV_PASS** |
| seed 分离 | A5 mask(seed=42) | A5 mask(seed=99) | 65,536 | 差异比 0.496 ≈ 0.5 | — | **EQUIV_PASS** |
| #4 阈值概率精确性（p=0.1/0.3/0.5/0.7） | `256Tp/2^31` | p | — | err ≤ 4.77e-08（≤ 2^-24） | 0 | **EQUIV_PASS** |
| #4 经验丢弃率（p=0.1/0.5/0.7, N=8M） | A5 mask | 3σ 界 | 8,388,608 | dev ≤ 2.80e-04（< 3σ 4.75e-04..5.18e-04） | 0 | **EQUIV_PASS** |

> 判定准则 = Stage 3 门禁同口径 `|cand−base| ≤ atol + rtol·|base|`（fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2，`equal_nan`）；覆盖随机张量 `randn`（2²⁰）+ 角点（±0/−0/±1/次正规 1e-38·1e-45/±65504/±inf/NaN）。**全部违规计数为 0 → ALL EQUIV_PASS**。

**非门禁诊断（机器输出）——源选择式相对契约 golden 的 `±inf/NaN` 偏离**（证明采纳算术乘的必要性，非采纳源选择式）：

| 形态 | `±inf/NaN/-0.0` 角点 vs torch GOLDEN | 结论 |
|------|--------------------------------------|------|
| 源 `if_then_else(keep, x·scale, 0)` | 差异 2/6（p=0.5）、3/6（p=1.0），3 dtype 一致 | **偏离契约**（丢弃 `±inf` 返回 0，torch 返回 NaN） |
| 采纳 `(x·scale)·m` | 差异 0/6 | **与契约一致** |

→ 采纳算术掩码乘不仅更廉价，且**比源选择式更贴合契约 golden**（torch `native_dropout` 为乘性掩码）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 元素的全局索引生成 | 源标量 `idx=(bx*threads+i)*npt+j`（线程级标量） | `T.arange(lane, [1], 0)`（元素 ramp）+ `T.fill(base, off)`（vbrc）+ `T.vadd(lane, base, idx)` | ✅ | —（设备实测 1-D ramp 与 numpy 逐位一致） |
| 2 | 随机数生成（掩码采样） | 源标量 `T.rng_rand_float()`（每线程顺序态） | 核内 counter-based hash 整 buffer 向量链（`vxor/vshl/vmul(回绕)/vand`，§0.6 R2） | ✅ | —（设备实测与 numpy 逐位一致） |
| 3 | 掩码阈值比较 | 源标量 `keep = rand_val >= p` | 整数链 `vsub` + `vmax` + `vmin`（阈值放大，免 `vshr`） | ✅ | — |
| 4 | 掩码施加 / 缩放 | 源标量 `if_then_else(keep, x*scale, 0)` | `vcast(i32→f32)` + `vmul(x,scale)` + `vmul(·, mf)`（无 `vselect`、无分支） | ✅ | — |
| 5 | fp32 中转（fp16/bf16） | 源无（输入 dtype 域） | `T.vcast(..., "rint")` 升/降 | ✅ | — |
| 6 | 常量 buffer 填充（seed/m31/zero/tp） | 源标量 `T.cast(scale, dtype)` 常量 | `T.fill(buf, T.cast(const, dtype))`（vbrc；循环外一次） | ✅ | — |
| 7 | tile 级调度循环 | `for i in T.serial(ceildiv(n_num, core_num))`（核内网格跨步） | 无（block 级任务调度循环） | ❌ | tile/block 级任务调度；每核迭代 `ceildiv(n_num,40)`（16m bs2048 = 205 次），非逐元素热点；元素级并行由向量指令整 buffer 完成。先例 `vec_add_2d.py` L32、`PL-1.22-persistent-1d-grid-stride` |
| 8 | block 起始索引 / 任务映射 | `task=i*core_num+cid`、`off=task*block_size` | 无 | ❌ | block 级索引/任务映射，每 tile 少量标量，非逐元素热点；无逐元素等价 API |
| 9 | 尾块边界（非整除 N） | `tail = T.min(block_size, N - off)` | 无 | ❌ | 依赖 N（编译期）与 `off`（运行时）的边界处理；manifest 全部 workload 的 N 整除 `block_size`（manifest N 为 2 的幂；测试 4,000,000 % 2048 = 1216 触发尾块），纯静态化对通用 N 不成立 |

**向量化结论**：逐元素计算点（#1–#6）**已全部向量化**（`vxor/vshl/vmul/vand/vsub/vmax/vmin/vcast/fill/arange/vadd`，均有 docs 佐证 + 设备探针实测）；保留 3 处标量/循环（#7 核内调度循环、#8 block 索引、#9 动态边界），理由类别：**tile 级调度循环 + block 索引 + 动态边界**。与 §6 循环结构一致——kernel 内**无逐元素标量循环**。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

**单轴算子豁免**：本算子为 1-D `[N]` 单轴算子（host 侧把任意 shape 展平为一维），不存在「在哪个轴向量化」的轴选择——唯一候选向量化轴即该 1D 元素轴本身。按 skill Phase 2 第 3 项「纯 GEMV/单轴算子可写明『仅一个候选轴』豁免」，记录豁免依据：

- **向量化轴** = 唯一 1D 元素轴（stride=1，连续）。
- **轴质量**：fp32 元素 4B，向量宽 ×4 → `block_size=1024` 每 tile 256 向量槽，尾 lane 浪费 0%（1024 整除 4）；fp16/bf16 元素 2B，向量宽 ×8 → `block_size=2048` 每 tile 256 槽，浪费 0%；均按 32B 尾轴对齐（1024×4=4096B、2048×2=4096B 为 32B 整数倍）。无累加链（`vmul/vmax/vmin` 均纯逐元素）。无 repack（I/O 布局 = 核内布局，一维连续，不需 `T.transpose` 或 host permute）。UB 驻留与 §4.5 一致。
- **迁移注**：GPU 源码的并行轴选择（thread `i`×`npt j` → 置换索引 `idx`）是**输入而非结论**——NPU Vector 轴与之不对应，§0.6 R1/R2 已独立评估为「核内 grid-stride + 向量指令整 buffer + 自然展平序索引」。
- **决策落地**：主选轴 = 元素轴，同步落入 §3.3 伪代码与 §6 循环结构（向量链各 op 直接作用于 UB 一维 buffer；无逐元素标量循环）。

**布局决策结论**：选定方案（唯一候选轴）：核内布局 = 一维连续 `[block_size]`（I/O 布局即核内布局，无 repack），向量化轴 = 唯一 1D 元素轴，repack 路径 = 无。弃选候选：无（单轴算子，无第二轴可枚举，按 skill Phase 2 第 3 项豁免）。

**A/B 实验裁决计划（§1.6.3 要求；主选 vs 备选）**：主选 = A5 hash（3 回合约 9 op，设备实测统计达标）；**备选 = A5′ 简化 hash**（去掉第 1 个 `vshl+vxor` 回合与第 2 个 `vmul`，约 5 op，结构完整可实现、仅链更短——但 avalanche 较弱，须重跑全部统计门禁）。裁决指标：16m fp16/bf16 latency（主选 vs 备选）+ 备选必须仍满足 `verify_equiv` 的 χ²/lag-1/3σ 全过。判定阈值：备选 latency < 主选 × 0.97 **且** 统计门禁全过 → 翻转。裁决结果回写 §1.6.3/pattern-library。`block_size`（2048 vs 4096）作为独立调优杠杆交 Stage 4（§5.2）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer

### 2.2 选型理由

| 算子特征 | 分析 | 结论 |
|----------|------|------|
| 计算类型 | 纯 Vector（逐元素掩码 + 缩放向量链），无 matmul | 不需 Cube/L0C/L1 |
| 是否含归约 | 无归约 | 不需 Expert 级 buffer 控制 |
| 是否需流水线 | 单数据通路 GM→UB→GM，无核间协作 | 不需 CV 融合 |
| 向量链 API | 全部 v-prefix（`vxor/vshl/vmul/vand/vsub/vmax/vmin/vcast/vadd`） | 用 v-prefix API |
| 同步需求 | UB 内顺序计算，数据依赖线性 | Developer 自动同步即可 |

决策树路径：`纯 element-wise → 单步逐元素 → 无需精细 buffer 控制 → Developer 模式`。与 `op_requirements.programming_mode=developer` 一致；参考 `examples/elementwise/vec_add_1d.py`（Developer + `T.alloc_ub` + v-prefix）与同族迁移 `examples/bitwise_not/bitwise_not_fwd_kernel/`（int 位运算 + `T.fill` + `T.serial` 网格跨步）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub(...)` 显式 UB 分配（v-prefix API 要求 buffer 在 UB） |
| 计算方式 | v-prefix API 向量链，非 `T.Parallel` 标量循环 |
| 同步 | Developer 自动同步（线性数据依赖，编译器自动插 barrier） |
| 分核 | 一维 `T.Kernel(core_num, is_npu=True)` + `T.serial` 网格跨步（§5.5） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（无状态 counter hash + 算术 0/1 掩码乘 + fp32 中转 + 编译期 p/seed 特化）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `lane = [0..bs)`；`base = off`；`idx = lane + base` | 元素自然序全局索引 |
| 2 | `h = idx ^ seed` | 种子混合（编译期 `seed`） |
| 3 | `h = mix(h)`（3 回合：`h ^= h<<13`；`h *= A`；`h ^= h<<7`；`h *= B`；`h ^= h<<13`） | avalanche（右移无关） |
| 4 | `hm = h & 0x7FFFFFFF` | 取非负 31 bit |
| 5 | `m = clamp(hm − (256·Tp − 1), 0, 1)`；`mf = cast(m, f32)` | 0/1 掩码（keep ⟺ `hm ≥ 256·Tp`） |
| 6 | `xf = cast(x)`（仅 fp16/bf16） | fp32 中转（无损升级） |
| 7 | `xs = xf · scale`（`scale = 1/(1-p)` 或 `0`） | 保留元素缩放 |
| 8 | `yf = xs · mf` | 掩码施加（丢弃 → 0 / `±inf`→NaN，同 torch） |
| 9 | `y = cast(yf)`（仅 fp16/bf16）；写出 | 单次舍回原 dtype |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 搬入 | `x_tile = x[...]` | `T.copy` | `T.copy(x[off:off+tail], x_ub[0:tail])` | Developer |
| 元素索引 ramp | `lane = [0..bs)` | `T.arange` | `T.arange(lane, [1], 0)`（循环外一次） | Developer |
| tile 基址广播 | `base = off` | `T.fill`（= `vbrc`） | `T.fill(base, off)` | Developer |
| 全局索引 | `idx = lane + base` | `T.vadd` | `T.vadd(lane, base, idx)` | Developer |
| 种子混合 | `h = idx ^ seed` | `T.vxor` | `T.vxor(idx, seedb, h)`（`seedb` 常量 buffer，循环外 `T.fill`） | Developer |
| 左移 | `h << k` | `T.vshl` | `T.vshl(h, 13, t)`（标量移位） | Developer |
| xor 混合 | `h ^= t` | `T.vxor` | `T.vxor(h, t, h)`（tensor-tensor；src1 须 tensor，`VP-2026-0130`） | Developer |
| 回绕乘 | `h *= A` | `T.vmul`（int32 回绕，实测） | `T.vmul(h, A_int32, h)`（`A_int32` 为 0x9E3779B1 的有符号形式） | Developer |
| 掩码位与 | `hm = h & 0x7FFFFFFF` | `T.vand` | `T.vand(h, m31, hm)`（`m31` 常量 buffer） | Developer |
| 阈值平移 | `d = hm − (256·Tp−1)` | `T.vsub` | `T.vsub(hm, tp, hm)` | Developer |
| 下界裁剪 | `d = max(d, 0)` | `T.vmax` | `T.vmax(hm, zerob, hm)`（`vmax` 无标量广播，用 `zerob`） | Developer |
| 上界裁剪 | `m = min(d, 1)` | `T.vmin` | `T.vmin(hm, 1, hm)`（`vmin` 允许标量 src2） | Developer |
| 掩码转 fp32 | `mf = cast(m)` | `T.vcast` | `T.vcast(hm, mf, round_mode="rint")` | Developer |
| 升 fp32（fp16/bf16） | `xf = cast(x)` | `T.vcast` | `T.vcast(x_ub, xf, round_mode="rint")` | Developer |
| 缩放 | `xs = xf·scale` | `T.vmul` | `T.vmul(xc, T.cast(scale, "float32"), xc)` | Developer |
| 掩码施加 | `yf = xs·mf` | `T.vmul` | `T.vmul(xc, mf, xc)` | Developer |
| 舍回（fp16/bf16） | `y = cast(yf,dtype)` | `T.vcast` | `T.vcast(xc, y_ub, round_mode="rint")` | Developer |
| 搬出 | `y[...] = y_tile` | `T.copy` | `T.copy((y_ub if need_cast else x_ub)[0:tail], y[off:off+tail])` | Developer |
| 核内网格跨步 | 任务映射 | `T.Kernel` + `T.serial` | `T.Kernel(core_num, is_npu=True)` + `for i in T.serial(ceildiv(n_num, core_num))` | Developer |
| 常量 buffer | seed/m31/zero/tp | `T.fill` | `T.fill(buf, T.cast(const, dtype))`（循环外一次） | Developer |

### 3.3 计算伪代码

> 迁移规则：`@T.prim_func` 参数名 `x, y` 及顺序不变（wrapper 契约：`_make_dropout_kernel(N,dtype,p,seed)(block_size)` 编译后以 `fn(x)` 调用，`out_idx=[1]` 自动分配 `y`，**无运行时 `shape` 参数**——尾块用编译期 `N` 静态计算）；**v-prefix op 只作用于整 UB buffer，动态 slice 只包在 `T.copy` 里**（`PL-1.22`/`PL-1.23`、R-6 陷阱）；**常量 buffer 在 tile 循环外一次性填充**；**buffer 在 `T.Kernel` 顶层无条件分配**（避免 `TRAP-conditional-alloc`）。

```python
import functools
import tilelang
import tilelang.language as T

_A_I32 = -1640531527   # 0x9E3779B1 as signed int32 (vmul wraps mod 2^32)
_B_I32 = -2048144777   # 0x85EBCA6B as signed int32
_M31 = 0x7FFFFFFF
_F32 = "float32"
_CORE_NUM = 40         # physical Vector cores (NPUUtils: 20 AI cores x 2)

@functools.lru_cache(maxsize=32)
def _make_dropout_kernel(N, dtype, p, seed, default_block_size=2048):
    """Inverted dropout, y = x*mask/(1-p), deterministic replay via in-core
    counter-based PRNG (cuRAND unavailable on npuir -- see DESIGN.md 0.6 R2)."""
    scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0
    tp = int(round(p * float(1 << 23)))          # drop threshold on bits [8..30]
    tp_shift = 256 * tp - 1                      # keep iff hm >= 256*tp

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(block_size):
        need_cast = dtype != _F32
        n_num = T.ceildiv(N, block_size)

        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            with T.Kernel(_CORE_NUM, is_npu=True) as (cid, _):
                # ---- buffers (unconditionally allocated; see TRAP-conditional-alloc) ----
                x_ub = T.alloc_ub((block_size,), dtype)
                y_ub = T.alloc_ub((block_size,), dtype)   # fp16/bf16 round-back
                xf = T.alloc_ub((block_size,), _F32)      # fp32 compute domain
                lane = T.alloc_ub((block_size,), "int32")
                base = T.alloc_ub((block_size,), "int32")
                h = T.alloc_ub((block_size,), "int32")    # idx -> hash (in place)
                t = T.alloc_ub((block_size,), "int32")    # hash temp -> mask
                seedb = T.alloc_ub((block_size,), "int32")
                m31 = T.alloc_ub((block_size,), "int32")
                zerob = T.alloc_ub((block_size,), "int32")
                tp_b = T.alloc_ub((block_size,), "int32")
                mf = T.alloc_ub((block_size,), _F32)

                # ---- loop-invariant constants (filled once, outside tile loop) ----
                T.fill(seedb, T.cast(seed, "int32"))
                T.fill(m31, T.cast(_M31, "int32"))
                T.fill(zerob, T.cast(0, "int32"))
                T.fill(tp_b, T.cast(tp_shift, "int32"))
                T.arange(lane, [1], 0)                    # lane[i] = i

                xc = xf if need_cast else x_ub

                for i in T.serial(T.ceildiv(n_num, _CORE_NUM)):   # static upper bound
                    task = i * _CORE_NUM + cid
                    if task < n_num:                              # grid-stride guard
                        off = task * block_size
                        tail = T.min(block_size, N - off)         # tail via compile-time N

                        # ---- PRNG: idx -> hash (right-shift-free, int32 wrap) ----
                        T.fill(base, off)
                        T.vadd(lane, base, h)          # h = idx
                        T.vxor(h, seedb, h)            # h = idx ^ seed
                        T.vshl(h, 13, t); T.vxor(h, t, h)
                        T.vmul(h, _A_I32, h)
                        T.vshl(h, 7, t);  T.vxor(h, t, h)
                        T.vmul(h, _B_I32, h)
                        T.vshl(h, 13, t); T.vxor(h, t, h)

                        # ---- 0/1 mask: keep iff (h & 0x7FFFFFFF) >= 256*tp ----
                        T.vand(h, m31, t)          # hm
                        T.vsub(t, tp_b, t)         # d = hm - (256*tp - 1)
                        T.vmax(t, zerob, t)        # d = max(d, 0)
                        T.vmin(t, 1, t)            # m = min(d, 1)
                        T.vcast(t, mf, round_mode="rint")

                        # ---- data path (fp32 compute domain) ----
                        T.copy(x[off:off + tail], x_ub[0:tail])
                        if need_cast:
                            T.vcast(x_ub, xc, round_mode="rint")
                        T.vmul(xc, T.cast(scale, _F32), xc)   # x * scale
                        T.vmul(xc, mf, xc)                    # * mask (0/1)
                        if need_cast:
                            T.vcast(xc, y_ub, round_mode="rint")
                            T.copy(y_ub[0:tail], y[off:off + tail])
                        else:
                            T.copy(xc[0:tail], y[off:off + tail])

        return main

    return kernel
```

> **伪代码说明**：① `if need_cast:` 为 Python 编译期条件（JIT 参数编译期确定），不产生运行时分支；**buffer 在 `with T.Kernel` 顶层无条件分配**（避免 `TRAP-conditional-alloc` 的 `Undefined variable`）；② 常量 buffer（`seedb/m31/zerob/tp_b`）与 `arange` 在 `T.serial` **循环外**填充一次（只读、跨 tile 有效）；③ v-op 全部作用于**整 buffer**（R-6 陷阱），仅 `T.copy` 用动态 `[0:tail]` slice；④ **`xc` 计算域别名**：fp32 路径 `xc=x_ub`（无中转、零额外开销），fp16/bf16 路径 `xc=xf`（`vcast` 升级后计算）；⑤ `h` 原地承载 idx→hash，`t` 原地承载 hash temp→掩码（`PL-1.23` 原地复用纪律）——故无需独立结果 buffer；⑥ `y_ub` 仅用于 fp16/bf16 的单次舍回（fp32 路径直接从 `x_ub` 搬出）。**每 tile 向量 op 计数 = 19（fp32）/ 21（fp16·bf16，+2 vcast）**。
>
> **接口适配说明**：迁移 wrapper `dropout.py` 约定 NPU 工厂 `_make_dropout_kernel(N, dtype, p, seed)` 返回「接受单一 `block_size` 的可调用」，`forward` 调 `self.kernel(block_size)(x)`（`out_idx=[1]` 自动分配 `y`）。本设计按此契约：外层工厂 `lru_cache` 特化 `(N, dtype, p, seed)`，内层 `@tilelang.jit kernel(block_size)`，`main(x, y)`（**无 `shape` 参数**，与 `compiled_fn(x)` 调用一致）；`block_size` 默认按 dtype 取 K11 折叠值（wrapper `_DEFAULT_BLOCK_SIZE`：fp32 1024 / fp16·bf16 2048）。

### 3.4 API 可行性确认

> **纪律（前车之鉴 `TRAP-vrsqrt-plain-precision` / `TRAP-verf-maclaurin-truncation` / `TRAP-threads-kwarg-noop`）**：NPU 文档声称的元算语义/精度/有效性**必须对目标设备实测**。本表全部关键 API 附目标设备（Ascend910_9362）实测（探针脚本 `history_version/design_probe_dropout_rng.py`，2026-10-14 本会话实跑，**ALL PASS**）。

| API | 来源确认（本地） | 设备实测（目标设备） | 状态 |
|-----|-----------------|---------------------|------|
| `T.vmul`（int32，回绕） | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.1（int32 ✓ / bf16 ✗）；§2.3 标量 src2 ✓ | 回绕实测：`2147483647×3→2147483645`、`-2147483648×3→-2147483648` —— **mod 2^32 回绕** ✓ | ✅ 真实可用 |
| `T.vxor`（int32） | `docs/.../逻辑操作/T.vxor.md` §2.2.1（int32 ✓）；§2.1 称 src1 可标量，但 `VP-2026-0130` 实测 `hivm.hir.vxor` src1 须 tensor | hash 链 tensor-tensor 形态实跑（与 numpy 逐位一致） | ✅ 真实可用（**src1 须 tensor**） |
| `T.vshl`（int32，标量移位） | `docs/.../逻辑操作/T.vshl.md` §2.2.1（int32 ✓）、§2.2.2（B 可为标量） | hash 链 `T.vshl(h,k,t)` 标量移位实跑（无告警） | ✅ 真实可用 |
| `T.vand`（int32） | `docs/.../逻辑操作/T.vand.md` §2.2.1（int32 ✓） | `vand(h,m31,t)` 实跑 | ✅ 真实可用 |
| `T.vshr`（int32） | `docs/.../逻辑操作/T.vshr.md` §2.2.1（int32 ✓）；§2.2.2（B 须 tensor/(1,)） | **算术右移**（`-1>>1 = -1`）；且编译告警 "will execute by **scalar** instruction with low efficiency"；shift 量须为 tensor | ⚠️ **可用但劣（算术+标量退化）→ 本设计避免使用** |
| `T.vadd`（int32） | `docs/.../数学操作/T.vadd.md` §2.2.1（int32 ✓） | `vadd(lane,base,h)` 实跑 | ✅ 真实可用 |
| `T.vsub`/`T.vmax`/`T.vmin`（int32） | `docs/.../数学操作/T.vsub.md`、`比较操作/T.vmax.md`·`T.vmin.md` §2.2.1（int32 ✓；`vmax` 无标量 src1、`vmin` src2 可标量） | 掩码链实跑 | ✅ 真实可用 |
| `T.vcast`（i32→f32 / f16↔f32 / bf16↔f32） | `docs/.../数据类型转换操作/T.vcast.md` §2.2.1（`i32→f32`、`f16→f32`、`f32→f16/bf16`、`bf16→f32` 支持 `rint`；**无 `i32→f16`/`u32`**） | 掩码 `vcast(i32→f32)` + 中转链实跑 | ✅ 真实可用 |
| `T.vmul`/`T.vmax`/`T.vmin`（fp32） | 同上 op doc §2.2.1（fp32 ✓ / bf16 ✗） | fp32 链实跑（全 dtype） | ✅ 真实可用（bf16 除外） |
| `T.arange`（1-D ramp） | `docs/.../创建操作/T.arange.md`；`VP-2026-0171` 记 `(1,N)` 单行 tile 静默产错尾元素（首维须 ≥2） | **1-D `(block_size,)` int32 ramp 与 numpy `arange` 逐位一致**（hash 与 numpy 逐位一致即证）→ 秩-1 buffer 未触发该 bug | ✅ 真实可用（1-D） |
| `T.fill`（vbrc） | `docs/.../内存操作/`；`bitwise_not_fwd_kernel.py` 常量 buffer；`customize_npuir.py::npuir_fill` 接受 `tir.PrimExpr` | `T.fill(base, off)`（运行时 PrimExpr 广播）+ 常量填充实跑 | ✅ 真实可用 |
| `T.copy`（slice） | `examples/elementwise/vec_add_1d.py` L34 | 实跑（含 16m 大 N） | ✅ 已核对 |
| `T.Kernel(n, is_npu=True)` + `T.serial` | `vec_add_2d.py` L28–L42 | 实跑（40 核网格跨步） | ✅ 已核对 |
| 源 `T.rng_init`/`T.rng_rand_float` | 无 `docs/` 文档或源码导出（§1.6.0 RNG 复核表） | — | ❌ **不存在**（须自建 hash，CG-2026-0021） |

**链端到端设备实测**（`design_probe_dropout_rng.py`，**ALL PASS**）：hash 与 numpy int32 模型**逐位一致**；top-11 χ²=2004（dof 2047）、lag-1 χ²=1119（dof 1023）；fp32/fp16/bf16 × p∈{0.1,0.3,0.5,0.7} 的 drop rate 全部 3σ 内；结构检查（每元素为 0 或 `x·scale`）全 dtype 通过；同 seed 逐位重放；异 seed 掩码差异 32522/65536（≈0.5）；`p=0` 恒等 / `p=1` 全零。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(_CORE_NUM, is_npu=True)` |
| GPU 专用 API | Yes（源用 `T.rng_init`/`T.rng_rand_float`、标量 `T.if_then_else`、`T.Parallel`、`threads=`） | RNG 重设计为核内 hash；选择式换算术掩码乘；换 v-prefix 向量链 + `is_npu=True`（§0.5/§0.6） |
| GEMM 非整除 | No | 非 GEMM；非整除由 `tail`（编译期 N）处理（§5.4） |
| L0C 溢出 | No | 无 Cube 计算 |
| 物理核数限制 | Yes | 逻辑核数 = `ceildiv(N, block_size)` 远超物理核 40 → 核内 `T.serial` 网格跨步（§5.5） |
| **无 RNG 前端原语** | Yes | 核内 counter-based hash（§0.6 R2，CG-2026-0021） |
| **bf16 v-prefix 数学 API 缺失** | Yes | `vmul/vmax/vmin` 不含 bf16 → bf16 走 fp32 中转（§0.6 R3） |
| **`vshr` 算术右移 + 标量退化** | Yes | 设计避免 `vshr`（右移无关 hash + 阈值放大，§1.6.1 #3） |
| 条件 `T.alloc` 非法（`TRAP-conditional-alloc`） | Yes（编译期分支易误用） | buffer 全部在 `T.Kernel` 顶层无条件分配（§3.3） |
| bool 不可写回 GM | No | 本设计不做 bool `vcmp→GM`（掩码为 int32/fp32，全程片上） |
| `vxor` src1 须 tensor（`VP-2026-0130`） | Yes | hash 的 xor 均为 tensor-tensor（`T.vxor(h,t,h)`） |
| `vmax` 无标量广播 | Yes | 下界裁剪用 `zerob` 常量 buffer（`vmax(h,zerob,h)`） |
| 标量操作数需在 `T.Kernel` 内定义 | Yes | `scale` 以 Python 编译期常量传入 `vmul` 标量位 |
| `T.Kernel(threads=)` 无效果（`TRAP-threads-kwarg-noop`） | Yes（源用 `threads=256`） | 移除 `threads=`；`threads×npt` 折叠为 `block_size` |
| **host 侧禁核心计算**（`ascend-constraints.md` §4） | Yes | 掩码生成在 kernel 内（否决 host 物化 mask，§1.6.0 R1-A1） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU 源） | 本项目（Ascend npuir） | 转换方案 |
|--------|-------------------|----------------------|----------|
| Kernel 装饰器 | `@tilelang.jit(out_idx=[1])` | `@tilelang.jit(out_idx=[1], target="npuir")` | K1 加 `target="npuir"` |
| Kernel 声明 | `T.Kernel(ceildiv(N,bs), threads=256)` | `T.Kernel(_CORE_NUM, is_npu=True)` + `T.serial` 网格跨步 | K2/K3 去 `threads=`、加 `is_npu=True`；§0.6 R1 |
| 并行模型 | `T.Parallel(threads, npt)` 线程级 | 一维核 + 向量指令整 buffer | K11：`threads×npt` 折叠为 `block_size` |
| **随机源** | **cuRAND Philox4_32_10**（`rng_init`/`rng_rand_float`） | **核内 counter-based hash**（无状态，`(seed,idx)` 纯函数） | §0.6 R2 重设计（CG-2026-0021） |
| **索引** | 置换序 `(bx*threads+i)*npt+j` | 自然展平序 `tile_base + arange(lane)` | §0.5 舍弃置换（NPU 无 coalescing 约束） |
| **掩码施加** | `if_then_else(keep, x*scale, 0)` | `(x*scale)*cast(clamp(d,0,1))`（算术掩码乘） | §0.6 R4 + §1.6.1 #1 |
| **计算域** | 输入 dtype 域 | fp32 中转域（fp16/bf16） | §0.6 R3（bf16 强制 + 契约一致） |
| Buffer 层级 | register（隐式） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| prim_func 参数 | `main(x, y)` | `main(x, y)`（不变） | wrapper 契约 `fn(x)`，无 `shape` 参数 |
| strategy/autotune | `threads`/`npt` + `default_config` | 无 strategy、无 autotune；单一 `block_size` | K8/K9/K11 移除 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py` | **高度相似**（迁移形态 + int 位运算） | `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步 + `T.fill` 常量 buffer + int 位运算（`vxor`/`vnot`）形态（结构参照） |
| `examples/elementwise/vec_add_1d.py` | 高度相似 | 一维 `T.Kernel`、`T.alloc_ub`、`T.copy` slice、v-prefix 一元 API |
| `examples/elementwise/vec_add_2d.py` | 高度相似 | `T.Kernel(BLOCK_SIZE, is_npu=True)` + `T.serial(ceildiv(...))` + `if block_id < 总任务` 网格跨步（分核先例） |
| `examples/leaky_relu/_make_leaky_relu_kernel/DESIGN.md` | 高度相似（同族迁移） | 迁移文档结构、设备探针方法论、`verify_equiv` 范式、fp32 中转与 `vcmp/vselect` 选路（**反面对照**：本算子以算术掩码乘替代其 `vselect`，因掩码选择价高且需贴合 torch 乘性语义） |
| `examples/elu/_make_elu_kernel/DESIGN.md`、`examples/expm1/expm1_fwd_kernel/DESIGN.md` | 相似（同族迁移） | 自建向量链 + `vcmp/vselect` 选路 + fp32 中转的设备探针方法论 |
| `examples/TileOPs/tileops/kernels/regularization/dropout/dropout.py` | 迁移目标 | 集成 wrapper：`_make_dropout_kernel(N,dtype,p,seed)(block_size)` + `block_size`（K11）+ `fn(x)` 调用契约 |
| `docs/Tilelang.language/数学操作/T.vmul.md`·`逻辑操作/T.vxor.md`·`T.vshl.md`·`T.vand.md`·`比较操作/T.vmax.md`·`T.vmin.md`·`数据类型转换操作/T.vcast.md`·`创建操作/T.arange.md`·`shape操作/T.vbrc.md` | API | 向量链各 op 规格、dtype 矩阵与用法 |
| `.agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md`（`PL-1.22`/`PL-1.23`/`PL-1.29`）·`constants.md`·`traps-compiler.md`（`TRAP-threads-kwarg-noop`） | 实测模式/陷阱 | 网格跨步模板、原地复用纪律、`vselect` 代价、向量单 op 代价、`threads=` 无效 |

---


## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(N,)`（原始任意 shape 展平） | `float16 \| bfloat16 \| float32` | 1-D 展平输入，`N = product(shape)` 编译期常量 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y` | `(N,)` | `same_as(x)` | 输出，shape/dtype 与输入一致（`out_idx=[1]` 自动分配） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | `(block_size,)` | `dtype` | UB | 输入 tile；**fp32 路径复用为计算域 `xc` 与结果缓冲** |
| `y_ub` | `(block_size,)` | `dtype` | UB | 输出 tile（**仅 fp16/bf16 路径**承载 `vcast` 舍回结果） |
| `xf` | `(block_size,)` | `float32` | UB | fp32 计算域 `xc`（fp16/bf16 = vcast 升级入口，链尾原地复用为结果；fp32 路径空闲） |
| `lane` | `(block_size,)` | `int32` | UB | 元素 ramp `[0..bs)`（循环外 `arange` 一次） |
| `base` | `(block_size,)` | `int32` | UB | tile 基址 `off`（每 tile `vbrc`） |
| `h` | `(block_size,)` | `int32` | UB | 全局索引 `idx` → hash（原地复用） |
| `t` | `(block_size,)` | `int32` | UB | hash 移位临时量 → `hm` → 0/1 掩码（原地复用） |
| `seedb` | `(block_size,)` | `int32` | UB | 常量 `seed`（**只读**，循环外 `T.fill`） |
| `m31` | `(block_size,)` | `int32` | UB | 常量 `0x7FFFFFFF`（**只读**，循环外） |
| `zerob` | `(block_size,)` | `int32` | UB | 常量 `0`（`vmax` 下界，**只读**，循环外） |
| `tp_b` | `(block_size,)` | `int32` | UB | 常量 `256·Tp − 1`（阈值平移，**只读**，循环外） |
| `mf` | `(block_size,)` | `float32` | UB | 0/1 掩码（fp32，供掩码乘） |

> **buffer 分配策略**：全部在 `with T.Kernel` 顶层**无条件分配**（避免 `TRAP-conditional-alloc`）；常量 buffer 循环外填充；`h`/`t` 原地复用（§3.3，`PL-1.23`）。

### 4.4 内存搬运路径

```
fp32 路径（纯 Vector，无 L1/L0；xc = x_ub）：
GM[x] --T.copy--> UB[x_ub] --[idx→hash→掩码→vcast; ×scale; ×mask（原地复用 x_ub）]--> UB[x_ub] --T.copy--> GM[y]

fp16/bf16 路径（fp32 中转；xc = xf）：
GM[x(dtype)] --T.copy--> UB[x_ub(dtype)] --T.vcast(rint)--> UB[xf(fp32)]
     --[idx→hash→掩码→vcast; ×scale; ×mask（原地复用 xf）]--> UB[xf(fp32)] --T.vcast(rint)--> UB[y_ub(dtype)] --T.copy--> GM[y]
```

掩码支路（只依赖 `lane`/`base`，与数据支路并行，全程 UB 级）：
```
arange(lane) + vbrc(base=off) → vadd(idx) → vxor(seed) → [vshl/vxor ×2 + 回绕 vmul ×2] → h
  → vand(0x7FFFFFFF) → vsub(tp) → vmax(0) → vmin(1) → vcast(fp32) → mf
```

### 4.5 UB 内存预算

K11 折叠 `block_size`：fp32 = 1024，fp16/bf16 = 2048。单位 KB（1 KB = 1024 B）。

| 路径 (bs) | x_ub | y_ub | xf | mf | lane/base/h/t/seedb/m31/zerob/tp_b（8×int32） | 合计 | ×1.7（`TRAP-UB-multibuffer-inflation`） |
|-----------|------|------|----|----|---------------------------------------------|------|----------------------------------------|
| fp32 (bs=1024) | 4 | 4 | 4 | 4 | 8 × 4 = 32 | **48 KB** | ≈ 82 KB < 192 KB ✓ |
| fp16 (bs=2048) | 4 | 4 | 8 | 8 | 8 × 8 = 64 | **88 KB** | ≈ 150 KB < 192 KB ✓ |
| bf16 (bs=2048) | 4 | 4 | 8 | 8 | 8 × 8 = 64 | **88 KB** | ≈ 150 KB < 192 KB ✓ |
| fp16 (bs=4096，E1 变体) | 8 | 8 | 16 | 16 | 8 × 16 = 128 | **176 KB** | ≈ 299 KB **> 192 KB ✗** |

> UB 硬上限 = 192 KB/AIV（`CONST-capacity-910B2C` 的 9362 适用性注记：容量数值与 9362 单 die 一致；膨胀系数 1.7x 为 910B2C 结构实测、9362 未重验）。**膨胀系数注（CG-2026-0008）**：显式逐 buffer alloc 结构实测 ~12% 余量即足；本预算按最坏 1.7x 保守核对。**设备实测佐证**：`bs=2048` 全 dtype 在目标设备**编译并运行成功**（探针 16m 用例），即真实占用在 UB 上限内。**`bs=4096` 在 8 个 int32 buffer 下 ×1.7 超限**——E1 变体若采纳须先削减 int32 buffer 数（如常量 buffer 合并/复用），故 E1 列为「须编译探针校准」的 Stage 4 杠杆（§5.2）。

> **dtype 差异注**：K11 折叠值 fp32=1024、fp16/bf16=2048（源 `threads(256)×npt`，fp32 npt=4 / fp16·bf16 npt=8），与 wrapper `_DEFAULT_BLOCK_SIZE` 一致。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| `N` 的实际长度 | 编译期 `T.Tensor((N,), dtype)`（`lru_cache` 按宿主 shape 特化） | 1 ~ manifest 最大 16,777,216 |

> `N` 为编译期常量（wrapper 按 `N_total` 构建一次 kernel，`forward` 复用）。尾块 `tail = T.min(block_size, N - off)` 用编译期 `N` 静态计算，**无需运行时 `shape` 参数**（与 wrapper 的 `fn(x)` 调用一致）。manifest 全部 workload 的 N 整除 `block_size`；测试 4,000,000 于 bs=2048 余 1216（触发尾块）。

### 4.7 JIT 配置

```python
@functools.lru_cache(maxsize=32)
def _make_dropout_kernel(N, dtype, p, seed):          # 编译期特化 (N, dtype, p, seed)
    ...
    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(block_size):                           # wrapper: self.kernel(block_size)
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            ...
        return main
    return kernel
# 调用：_make_dropout_kernel(N, dtype_str, p, seed)(block_size)(x)
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅含逐元素向量链（`vxor/vshl/vmul/vand/vsub/vmax/vmin/vcast/vadd`），无 matmul、无归约、无 Cube 计算；仅需 UB 层级，不涉及 L1/L0A/L0B/L0C。

### 5.2 Block 划分

```python
block_size = {"float32": 1024, "float16": 2048, "bfloat16": 2048}[dtype]   # K11 折叠
n_num = T.ceildiv(N, block_size)   # 逻辑 tile 数（编译期静态值）
```

**block_size 选择理由**：
1. 与源 K11 折叠值一致（fp32 = `threads(256)×npt(4)` = 1024、fp16/bf16 = `threads(256)×npt(8)` = 2048），迁移可追溯；v-prefix 整 buffer 形态下无 128-bit 对齐差异；
2. UB 对齐：`block_size×elem_bytes` 是 32B 的整数倍（fp32 4096B、fp16/bf16 4096B）✓；
3. UB 容量：最大路径（fp16/bf16 bs=2048）≈ 88 KB，×1.7 最坏膨胀 ≈ 150 KB < 192 KB ✓（设备实测编译通过）；
4. **发射项权衡**：§1.6.0 估算显示**发射/向量吞吐主导**（op 数 ~19–21；发射 ≈ 1/block_size），但 bs 放大受 UB 膨胀约束（bs=4096 超限，§4.5）——故默认取 K11 折叠值；`bs=4096`（须先削 int32 buffer）列为 Stage 4 调优杠杆。

### 5.3 约束分析

- **对齐约束**：fp32 bs=1024，元素 4B → 每 tile 256 向量槽（128bit×4），尾 lane 浪费 0%；fp16/bf16 bs=2048，元素 2B → 256 向量槽（128bit×8），浪费 0%；均满足 32B 尾轴对齐 ✓。
- **UB 容量**：最大 88 KB（fp16/bf16 bs=2048），×1.7 最坏膨胀 ≈ 150 KB < 192 KB ✓。
- **L0/L1 容量**：不适用（纯 Vector）。
- **分形限制**：不适用（无 Cube）。

### 5.4 注意事项（非整除处理策略）

**非整除场景预判**：manifest workload 的 N（1,048,576 / 4,194,304 / 10,485,760）均整除 `block_size`（fp32 1024 与 fp16/bf16 2048 均整除，因 N 为 2 的幂或 2·5 幂且 ≥ 2²⁰；`4,194,304 = 2²²`、`10,485,760 = 2²¹·5` 均整除 2048）；测试 shape 4,000,000 **不整除 2048**（余 1216）→ 存在尾块。

**策略**：Kernel 内动态尾块（不改动 host 输入张量、不做 host padding，遵守 `ascend-constraints.md` §4）：最后一个 tile 用 `tail = T.min(block_size, N - off)`（`N` 编译期常量）裁剪 `T.copy` 的搬入/搬出量；向量 op 仍作用于**整 buffer**（尾部未初始化数据不影响仅搬出的 `[0:tail]`）。参考 `examples/elementwise/vec_add_1d.py`。任务网格用 `T.ceildiv(N, block_size)` 保证覆盖最后一个不完整 tile。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

**① 逻辑核数计算**：本算子为 1-D 输出网格 `[N]`，仅 N 轴（无 M 轴，M 项取 1）：

```
num_logical_kernels = ceil(N / block_size)
```

| workload | dtype | N | block_size | 逻辑核数 |
|----------|---|------|-----------|----------|
| smoke `[1024,1024]` | fp32 | 1,048,576 | 1024 | 1,024 |
| smoke `[1024,1024]` | fp16/bf16 | 1,048,576 | 2048 | 512 |
| `[1024,4096]` | fp16 | 4,194,304 | 2048 | 2,048 |
| `[1024,10240]` | bf16 | 10,485,760 | 2048 | 5,120 |
| `[1024,4096]` | fp32 | 4,194,304 | 1024 | 4,096 |

**② 物理核数依据（实查）**：目标设备 `Ascend910_9362`，实查代码与返回值（纯 Vector 算子核数翻倍）：

```python
from tilelang.utils.npu_utils import NPUUtils
NPUUtils.get().get_aicore_num()        # 实际返回 20
NPUUtils.get().get_aivector_core_num() # 实际返回 40（= aicore × 2）
```

→ **纯 Vector 物理核数 = 20 × 2 = 40**（2026-10-14 本会话实查；与 `CONST-aicore-910_9362` / `CG-2026-0019/0020` 的设备戳一致）。注：`pattern-library/constants.md` 的 `CONST-aicore-910B2C` 记 24 核，绑定 Ascend910B2C，**以实查为准**。

**③ 规模判定与分核方案**：**极大规模**——逻辑核数（512 ~ 5,120）远超物理核数 40，且 N 为 2 的幂/含因子 5，物理核数 40 = 2³×5 含非 2 因子，数学上**无法**通过调整 block_size（2 的幂）使逻辑核数对齐 40 的整数倍（5 ∤ 2ⁿ）。按标准 §1 要素③「极大规模」处置：

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
- **残留负载不均**（显式记录）：以 smoke fp32（bs1024）为例 `ceildiv(1024,40)=26`；网格跨步下前 24 个核执行 26 次、后 16 个核 25 次 → 临界路径 ≈ **+1.56%**；fp16/bf16 smoke（bs2048）`ceildiv(512,40)=13`，前 32 核 13 次、后 8 核 12 次 → 临界 **+4.2%**；`[1024,10240]` bf16 `ceildiv(5120,40)=128` → 临界 **+0.78%**。相较「直接 `T.Kernel(n_num)` 由运行时串行调度多波次」，核内 `T.serial` 把核启动固定开销从「每 tile 一次」摊薄为「每核一次」，净收益为正。
- **模式先例**：`examples/elementwise/vec_add_2d.py` L28–L42；同族 `bitwise_not_fwd_kernel.py` L89–L108（`PL-1.22-persistent-1d-grid-stride`）。

> 备选（供 Stage 4 参考，不在本设计实现）：直接 `T.Kernel(n_num)` 由运行时调度（无核内串行）——保留为对称性对照；本设计按标准要求取核内串行 persistent 形态。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block 级 / tile 级调度循环；kernel 内无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级（任务网格） | block 级并行 | `T.Kernel(_CORE_NUM, is_npu=True)` | 40 个物理 Vector 核各处理一组 tile |
| tile 级（核内） | 串行迭代（网格跨步） | `T.serial(ceildiv(n_num, core_num))` | 每核串行处理 `ceildiv(n_num,40)` 个 tile，摊薄核启动开销（§5.5） |
| 元素级 | 向量化 | `vxor/vshl/vmul/vand/vsub/vmax/vmin/vcast/vadd`（整 buffer） | 一次完成，无 `T.Parallel` 标量循环 |

**T.Persistent 选择说明**：本设计以 `T.serial` 网格跨步实现 persistent 语义（每核固定处理多 tile），与 `T.Persistent` 等价且为同族先例形态（`PL-1.22`）；不额外引入 `T.Persistent`（核数固定 40 已足够）。

### 6.2 循环伪代码

见 §3.3（`with T.Kernel(_CORE_NUM, ...)` + `for i in T.serial(...)` + `if task < n_num`，循环内为 PRNG 向量链 + 掩码链 + 数据链）。

### 6.3 流水线优化

**当前设计**: 不使用 `T.Pipelined`。理由：单 tile 内数据依赖线性（copy-in → [vcast↑] → PRNG/掩码链 → 标量乘 / 掩码乘 → [vcast↓] → copy-out），无 K 维迭代；跨 tile 迭代（`T.serial`）的 MTE2/VEC 重叠由 Developer 模式 auto-multi-buffer 自动处理。**发射主导下**，若 Stage 4 需进一步优化（persistent double buffer / 更大 block_size），参照 `example_elementwise_exp2.py` 的 `T.serial` 多 tile 形态；设计阶段保持简洁。

### 6.4 尾块处理

见 §5.4。`N % block_size != 0` 时最后一个 tile 用 `tail = T.min(block_size, N - off)` 动态裁剪 `T.copy` 量；manifest 全部 workload 整除（§5.4），仅测试 4,000,000 触发尾块（余 1216，bs=2048）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下编译器自动插入同步。核内数据依赖链严格线性（每 tile 内）：

```
T.copy(GM→UB) → [vcast↑] → (PRNG 链 ∥ 掩码链) → ×scale → ×mask → [vcast↓] → T.copy(UB→GM)
```

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy(GM→UB)` 后 | 自动 | 编译器插 MTE2→V 等待 |
| 各 v-op 之间（PRNG 链 + 掩码链） | 自动 | 线性数据依赖（hash 回合串行、掩码链串行、数据链汇入 `×mask`），插 V→V barrier |
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

> 迁移任务：golden 以 **§0.1 源算子契约语义**为唯一依据实现，**不得复刻 §0.6 的 NPU PRNG 结构**——保证验证独立性。dropout 是**随机算子**，不存在可逐元素复现的解析 golden：验证由「契约性质」构成（统计 + 重放 + 缩放 + 边界），另提供一个 torch 语义基线用于**性质对照**（非逐元素比对）。

```python
import torch
import torch.nn.functional as F


def golden_from_mask(x: torch.Tensor, mask: torch.Tensor, p: float) -> torch.Tensor:
    """给定掩码的确定性契约参考（torch native_dropout 乘性语义 x*(m*scale)）。

    mask ∈ {0,1}，与 x 同 shape；输出 in-place 于 fp32 opmath 后舍回原 dtype。
    **独立于 NPU 向量链**（不复用 §3.3 的 op 序列；掩码由实现提供，算术由 torch 公式给出）。
    """
    x_f = x.detach().cpu().to(torch.float32)
    m = mask.detach().cpu().to(torch.float32)
    scale = 0.0 if p >= 1.0 else 1.0 / (1.0 - p)
    return (x_f * (m * scale)).to(x.dtype)


def golden_dropout_torch(x: torch.Tensor, p: float, training: bool = True,
                         seed: int = 0) -> torch.Tensor:
    """torch 契约语义基线（用于性质对照与 p=0/1、training=False 边界的期望值）。

    注意：其 RNG 与 NPU 实现不同，**掩码不逐位可比**；只用于校验统计/边界性质。
    """
    if not training or p == 0.0:
        return x.detach().cpu().clone()
    g = torch.Generator().manual_seed(seed)
    return F.dropout(x.detach().cpu().to(torch.float32), p, training=True).to(x.dtype)


def check_dropout_contract(y, x, p, *, y2=None, dtype_tol=(1e-5, 1e-3, 1.6e-2)):
    """契约性质检查（L0 门槛用）：结构 + 缩放 + 统计 + 重放。

    返回 (structure_ok, scale_ok, rate, rate_ok, replay_ok)。
    容差按 dtype：fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2（来源 test_dropout.py）。
    """
    y = y.detach().cpu()
    x = x.detach().cpu()
    scale = 0.0 if p >= 1.0 else 1.0 / (1.0 - p)
    at = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 1.6e-2}[x.dtype]
    yf = y.to(torch.float32)
    is_zero = yf == 0
    is_scaled = torch.isclose(yf, torch.full_like(yf, scale), atol=at, rtol=0)
    structure_ok = bool((is_zero | is_scaled).all())
    scale_ok = bool(is_scaled[is_scaled].numel() == 0 or
                    torch.isclose(yf[is_scaled], torch.full_like(yf[is_scaled], scale),
                                  atol=at, rtol=0).all())
    n = x.numel()
    rate = float(is_zero.to(torch.float32).mean())
    sigma3 = 3.0 * (p * (1 - p) / n) ** 0.5 if 0 < p < 1 else 0.0
    rate_ok = bool(abs(rate - p) <= max(sigma3, 1e-6))
    replay_ok = None if y2 is None else bool(torch.equal(y, y2.detach().cpu()))
    return structure_ok, scale_ok, rate, rate_ok, replay_ok
```

> 备注：`golden_from_mask` 与 `golden_dropout_torch` **独立于** NPU 向量链（算术来自 torch 公式 / 统计来自理论）。**关键**：§1.6.1 的等价性机器验证（`verify_equiv.py`）以 fp64 参照证明了掩码乘算术对契约的逐值一致（有限域与 `±inf/NaN` 角点）；统计/重放由 §8.3 L0 门槛与设备探针覆盖。

### 8.2 精度标准

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 1e-5 | 1e-5 | 来源：`examples/TileOPs/tests/ops/test_dropout.py` `test_dropout_scale_factor` |
| float16 | 1e-3 | 1e-3 | 来源：同上；NPU 在 fp32 域掩码乘、单次舍回 |
| bfloat16 | 1.6e-2 | 1.6e-2 | 来源：同上；bf16 精度低，容忍度放宽 |

**误差预算核对（设计期，`verify_equiv.py`）**：掩码乘算术对契约 golden 在有限域与 `±inf/NaN` 角点 **max|d| = 0**（3 dtype）；fp32 中转 vs 输入-dtype 基线 max|d| = 5.96e-8（fp16） / 0（bf16），远优于门禁。统计：drop rate 全 p 值落在 3σ 内（设备实测）。

### 8.3 L0 门槛测试计划

> L0 = 门槛 smoke 测试（最小可用性验证，进 Stage 3 前必过）；完整分层套件 L1/L2/Boundary 交由 `tilelang-op-develop`，不在此枚举。**显式包含统计、确定性重放、异 seed、缩放、`p=0/1` 边界、各 dtype、非整除尾块、非默认 `block_size`**。

| 用例 | 输入 | 期望 | 判定 |
|------|------|------|------|
| L0-1 统计率（fp32/fp16/bf16, p=0.5） | `ones([4194304])`（NPU） | drop rate 落在 p 的 3σ 内（`check_dropout_contract`） | 门槛通过 |
| L0-2 统计率（fp16, p∈{0.1,0.3}） | `ones([4194304])`（NPU） | 3σ 内（`test_dropout_statistical_rate` 同款） | 门槛通过 |
| L0-3 缩放正确 | `ones([1000000])` 各 dtype/p | 非零元素 == `1/(1-p)`（dtype 容差） | 门槛通过 |
| L0-4 确定性重放 | `randn([1000000], fp16/fp32)`，同 seed 两次 | `torch.equal(y1, y2)` | 门槛通过 |
| L0-5 异 seed 分离 | `ones([262144])`，seed 42 vs 99 | `not torch.equal(y1, y2)`（设备实测掩码差异 ≈0.5） | 门槛通过 |
| L0-6 `p=0` 恒等 | `randn([1000000])`，p=0.0 | `allclose(y, x)`（设备实测 ✓） | 门槛通过 |
| L0-7 `p=1` 全零 | `randn([1000000])`，p=1.0 | `torch.equal(y, 0)`（设备实测 ✓） | 门槛通过 |
| L0-8 `training=False` | `randn([1000000])`，training=False | host 短路径直接返回输入 | 门槛通过 |
| L0-9 shape/dtype 保持 | `randn((100, N//100))` 各 dtype | `y.shape==x.shape`、`y.dtype==x.dtype` | 门槛通过 |
| L0-10 非整除尾块 | `randn([4000000], fp16)`（4000000 % 2048 = 1216） | 契约性质全过（设备实测 bs=2048 尾块 ✓） | 门槛通过 |
| L0-11 非默认 block_size（回归） | `8192` fp16 `block_size=512` / `8192` fp32 `block_size=128` / `65536` fp16 `block_size=1024`，p=0 恒等 | 输出正确（`test_dropout_custom_config_*` 同款，验证 grid 维度与 launch config 一致） | 门槛通过 |
| L0-12 `±inf/NaN/-0.0` 角点（诊断） | `repeat([inf,-inf,nan,-0.0,0.0,3.0], ...)`，p=0.5/1.0 | 语义与 torch 乘性一致（丢弃 `±inf`→NaN，`-0.0` 保留） | 记录（非门禁） |

- **执行方式**：`tilelang.compile(func, target="npuir")` 后单次调用；性质检查用 §8.1 `check_dropout_contract`；`p=0/1`、`training=False` 用 `golden_dropout_torch` 期望值。
- **门槛标准**：L0-1..L0-11 全过即进入 Stage 3；任一失败不得继续。
- 输入生成参考 `examples/TileOPs/tileops/workloads/regularization.py`（`DropoutOpWorkload`）；边界用例参考 `test_dropout.py::DropoutEdgeCaseFixture` / `DropoutCustomConfigFixture`。
- **额外设计期落点**：`history_version/design_probe_dropout_rng.py`（设备探针，ALL PASS）、`verify_equiv.py`（D-1 机器验证，FAIL items = 0）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 编号 | 约束 | 影响 | 缓解方案 |
|------|------|------|----------|
| **R-1** | **无 RNG 前端原语**（`T.rng_init`/`T.rng_rand_float` 不可用）→ 软件 PRNG 取代硬件 RNG | 固有性能落差：16m fp16 实测 366µs（约为带宽地板 63µs 的 5.8×，发射主导） | 核内 counter-based hash（§0.6 R2，CG-2026-0021 登记）；列明落差；若前端未来暴露 `fill_rng_2d`/RNG 元算则去绕化 |
| **R-2** | **`vshr` 算术右移 + 标量退化** | 若误用 `vshr` 构造 hash：分布被算术语义破坏 + 标量指令性能陷阱 | 设计**完全避免 `vshr`**（右移无关 hash + 阈值放大 `hm ≥ 256Tp`，§1.6.1 #3）；E3 明确拒绝 A4 形态 |
| **R-3** | **bf16 v-prefix 数学 API 缺失**（`vmul/vmax/vmin` 无 bf16） | bf16 直传编译失败 | bf16 走 fp32 中转（§0.6 R3，API 强制） |
| **R-4** | **随机算子无解析 golden** | 「与 golden 逐元素比对」不适用 | 契约化验证：统计 3σ + 重放 + 缩放 + 边界（§8.1）；算术等价由 `verify_equiv.py` 机器验证 |
| **R-5** | **跨设备掩码不可逐位复现**（cuRAND ≠ 本 hash） | 与 GPU 输出不逐位相同 | 契约核对：分布式随机算子的固有性质；`different seeds differ` 与自洽重放满足（§0.6 R2 论证 4） |
| **R-6** | **`±inf/NaN` 掩码语义** | 算术掩码乘对丢弃 `±inf` 返回 NaN（≠ 源选择式的精确 0） | **这是契约正确行为**（torch `native_dropout` 乘性语义，实测 p=1 返回 NaN）；`verify_equiv` 证候选与 golden 一致、源选择式偏离 |
| **R-7** | **UB 容量（bs=4096 超限）** | E1 大 block_size 变体不可行（176KB×1.7 > 192KB） | 默认 bs 取 K11 折叠值（最大 88KB×1.7≈150KB 安全）；bs=4096 须先削 int32 buffer 并编译探针校准 |
| **R-8** | **发射/向量吞吐主导**（op 数 19–21） | 性能天花板由向量 op 吞吐决定 | 标注「未文档化假设 + 估算依据」（§1.6.0 D-2）；block_size/简化 hash（备选）为 Stage 4 杠杆；Stage 4 baseline 后对账 |
| **R-9** | **条件 `T.alloc` 非法**（`TRAP-conditional-alloc`） | 编译期 `if` 内 `T.alloc_ub` trace 失败 | buffer 全部在 `T.Kernel` 顶层无条件分配（§3.3） |
| **R-10** | **v-op 整 buffer vs 动态 slice** | 传动态 slice 报错 | v-op 作用于整 buffer，仅 `T.copy` 用 `[0:tail]`（§3.3） |
| **R-11** | **`vxor` src1 须 tensor**（`VP-2026-0130`） | 标量 src1 报 "cannot be a scalar" | hash 的 xor 均 tensor-tensor（`T.vxor(h,t,h)`） |
| **R-12** | **`vmax` 无标量广播** | `vmax(d,0)` 不可用 | 下界裁剪用 `zerob` 常量 buffer（`vmax(t,zerob,t)`） |
| **R-13** | **`vcast` round_mode 限制**（`T.vcast.md` §2.2.1） | `f16→f32`/`f32→f16/bf16`/`bf16→f32` 仅 `rint`；**无 `i32→f16`/`u32`** | 中转统一 `round_mode="rint"`；掩码 `i32→f32` 后用于 fp32 乘（无 i32→f16 需求） |
| **R-14** | **`T.arange` 单行 tile 陷阱**（`VP-2026-0171`，`(1,N)` 静默错尾元素） | 1-D buffer 若被当单行可能出错 | **设备实测**：1-D `(block_size,)` ramp 与 numpy 逐位一致（未触发）；若 Stage 3 遇异常，退路 = 2-D `(2, bs/2)` ramp + `reshape` |
| **R-15** | **分核残留不均**（40 核含非 2 因子 5） | 2ⁿ/含 5 逻辑核无法整除 40 | §5.5 网格跨步核内串行，残留临界 +0.78%~+4.2%，已显式记录 |
| **R-16** | **常数表设备/版本** | `CONST-mte2-degradation` 带宽曲线绑 910B2C 且 stale；`CONST-vector-op-cost-910_9362` 绑 910_9362 | 本设计关键正确性结论来自**本会话设备探针**（不依赖 stale 条目）；带宽/发射数字标注「待重验」，仅作量级估算；Stage 4 baseline 后对账 |
| **R-17** | **元算设备语义** | 文档语义可能与设备不一致 | 本设计关键 API 全部经**目标设备探针实测**（§3.4）；工具链升级后须重验 |
| **R-18** | **掩码分配序与源不同** | 与 GPU 掩码不逐位一致 | §0.5 显式记录（契约不含逐位要求）；列入风险交 Stage 3 核对 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 照抄 `T.rng_init`/`T.rng_rand_float` | 直接移植源 | 编译失败（属性不存在） | 用核内 counter-based hash（R-1） |
| hash 用 `vshr` | 移植 xorshift32 | 算术右移破坏分布 + 标量退化 | 右移无关 hash + 阈值放大（R-2） |
| 用 `vselect` 施加掩码 | 直译源选择式 | `vselect` 6.5x + `±inf` 语义偏离契约 | 算术 0/1 掩码乘（§1.6.1 #1） |
| bf16 直接传入数学 op | 未做 vcast 上转 | 编译失败 | bf16 先 `vcast` 到 fp32（R-3） |
| 掩码常量用 `u32`/`i32→f16` | 误用无支撑 dtype | 编译失败 | 全程 int32 + fp32 掩码（R-13） |
| `vmax` 传标量 0 | 下界裁剪 | 编译失败（无标量广播） | 用 `zerob` 常量 buffer（R-12） |
| `vxor` 传标量 seed | 种子混合 | 编译失败（src1 须 tensor） | 用 `seedb` 常量 buffer（R-11） |
| 条件 alloc | 编译期 `if need_cast:` 内 `T.alloc_ub` | trace 报 `Undefined variable` | buffer 顶层无条件分配（R-9） |
| v-op 传动态 slice | `T.vmul(x_ub[0:tail], ...)` | 可能编译/运行错误 | v-op 用整 buffer（R-10） |
| 零输入崩溃 | N=0 空张量 | MTE DDR 崩溃（`TRAP-zero-input-crash`） | host 侧空张量守卫；本任务 N ≥ 2²⁰，不触发 |
| UB 溢出 | block_size 过大（4096） | 编译失败 | 维持 K11 折叠值（R-7） |
| hash 常量超 int32 | `vmul(h, 0x9E3779B1, h)` | 编译报 `Literal exceeds maximum of int32` | 用有符号形式（`-1640531527`） |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| **非整除 N** | Kernel 内 `tail`（编译期 N）动态裁剪（§5.4），无需 host padding |
| **极小 N（N < block_size）** | `n_num=1`，`tail=N`，仅 `task==0` 的核工作，其余核空转 |
| **极大 N（>1.6e7）** | 核内 `T.serial` 网格跨步（§5.5），逻辑核数静态 |
| **`p=0`** | `Tp=0` → 阈值平移 `-1` → `d=hm+1≥1` → 掩码恒 1；`scale=1` → 恒等（实测 ✓） |
| **`p=1`** | `Tp=2²³` → `d≤0` → 掩码恒 0；`scale=0` → 全零（实测 ✓） |
| **`±inf/NaN/-0.0` 输入** | 与 torch 乘性语义一致：保留 → `±inf`；丢弃 → NaN；`-0.0` 保留 `-0.0`（`verify_equiv` 证） |
| **不同 seed** | `seed` 编译期特化（`lru_cache` 按 seed 建独立 kernel）→ 掩码不同（实测差异 ≈0.5） |
| **混合 dtype** | 按 dtype 编译期特化（每 dtype 一份 kernel），输出 `same_as(input)` |
| **`training=False`** | host 侧直接返回输入（不在 kernel 内，§0.1） |

---

## 10. 交付清单

| 交付物 | 路径 | 说明 |
|--------|------|------|
| 设计文档 | `examples/dropout/_make_dropout_kernel/DESIGN.md` | 本文件 |
| 等价性机器验证 | `examples/dropout/_make_dropout_kernel/verify_equiv.py` | D-1：FAIL items = 0（ALL EQUIV_PASS）（§1.6.1） |
| 设计期设备探针 | `examples/dropout/_make_dropout_kernel/history_version/design_probe_dropout_rng.py` | D-3：hash 逐位一致 + χ²/lag-1 + 3σ + 重放 + seed 分离 + 边界（ALL PASS，§3.4） |
| 探针计时脚本 | `examples/dropout/_make_dropout_kernel/history_version/design_probe_dropout_rng_timing.py` | 16m/1m × 3 dtype 端到端时延（§1.6.0 D-2） |
| 能力缺口登记 | `.agents/evolution/capability-gaps.md`（CG-2026-0021） | npuir 无 RNG 前端原语（§1.6.0 / §0.6 R2） |
| NPU kernel（下游 Stage 3 产出，尚未创建） | 算子目录下 `_make_dropout_kernel.py` | 按 §3.3 伪代码 + §5 分核实现 |
| Golden（下游） | 同上（kernel 文件内 `golden_from_mask` / `check_dropout_contract`） | 依 §8.1 |
| 迁移集成 wrapper（已存在，只读参照） | `examples/TileOPs/tileops/kernels/regularization/dropout/dropout.py` | K11 `block_size` + `fn(x)` 调用契约 |
| 精度用例（已存在，只读参照） | `examples/TileOPs/tests/ops/test_dropout.py` | 统计/重放/异 seed/缩放/边界/custom config |
| 性能用例（已存在，只读参照） | `examples/TileOPs/benchmarks/ops/bench_dropout.py` | manifest-driven bench（torch `F.dropout` 基线） |

**实现要点速览（Stage 3 输入）**：
1. **随机源 = 核内 counter-based hash**（`idx^seed` → 3×(左移+xor) 与 2×(回绕乘) 交错；`& 0x7FFFFFFF`）；**禁用 `vshr`**（算术 + 标量退化）；
2. **掩码 = `clamp(hm − (256·Tp−1), 0, 1)`**（`vsub/vmax/vmin`，免 `vshr`/`vselect`），`Tp=round(p·2²³)`；`mf=vcast(i32→f32)`；
3. **施加 = `(x·scale)·mf`**（算术掩码乘，贴合 torch 乘性语义；**不**用 `vselect`）；
4. **计算域 = fp32 中转**（fp16/bf16 先 `vcast(rint)` 升、链尾 `vcast(rint)` 舍回；bf16 强制）；
5. 结构：一维 `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步 + `tail`（编译期 N）尾块守卫；`block_size` = K11 折叠值（fp32 1024 / fp16·bf16 2048）；
6. buffer 顶层无条件分配；常量 buffer（seed/m31/zero/tp）与 `arange` 循环外一次 `T.fill`；v-op 作用于整 buffer；hash 常量用有符号 int32 形式；
7. **确定性重放契约**由「`h` 为 `(seed, idx)` 纯函数」保证；`p=0/1` 无需特判（链自然给出恒等/全零）。
