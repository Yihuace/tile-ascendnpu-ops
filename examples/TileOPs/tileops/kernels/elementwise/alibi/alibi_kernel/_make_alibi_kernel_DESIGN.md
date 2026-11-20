# _make_alibi_kernel 算子设计文档（迁移 · 修订第 2 版）

> **项目**：`alibi` ｜ **算子目录**：`examples/alibi/_make_alibi_kernel/` ｜ **编程模式**：Developer ｜ **target**：`npuir`
>
> **任务性质**：迁移（migration-harness）。把 GPU TileLang 工厂函数 `_make_alibi_kernel`（`@tilelang.jit(out_idx=[0])`，`main(out)` 仅输出、无张量输入）重设计为 NPU（`target="npuir"`）实现。本算子是**生成式（generative, zero-tensor-input）**算子：输出 `bias[H,S,S]` 完全由构造期整数参数 `seq_len`/`num_heads` + `dtype` 合成。
>
> **本版（rev2）修订说明**：v1/rev2 因 **§1.6.3 候选 #2 的 `T.transpose` dtype 负向前提与所引文档矛盾**（文档记 bf16 √，设计记 bf16 ×）被 Stage 2 检视判**不通过**。本版**仅做最小修订**——删除该 dtype 限制前提（弃选 #2 仅保留「无收益」真实理由）、全文逐条核对 dtype 断言、落盘 UB 天花板探针、修正生成式算子的 host permute 表述；**技术结论（div/mod 可实现性、性能裁决 59.1×、arange ramp 选定、4 项等价优化、`verify_equiv.py` 46/46、分核三要素）全部不变**。详见 §11.1。
>
> **本版（rev1）修订说明**：v0 因**两条负向能力断言被 Stage 2 检视推翻**而判不通过——(1)「NPU 无向量整数 div/mod」；(2)「`T.Parallel` 写 UB（Developer）不可用」。rev1**以目标设备实测重做 §0.5 耦合性判定 / §1.6.0 调研四问 / §1.6.1 / §1.6.2**，并逐条修正受影响的负向断言（§3.4/§3.5.1/§1.6.3）。**坐标生成方案的结论不变（仍选 arange ramp + D 共享）**，但**裁决依据由「基线不可实现」改为「基线可实现但慢 59×」的目标设备实测**（`design_probe_alibi_v1.py` Q1/Q2/Q4）。详见 §0.6 与 §1.6.0。
>
> **本版核心决策 1（索引生成，本算子最关键点）**：源 GPU 用「flat 索引 → `h=flat//S²`、`rem=flat%S²`、`row=rem//S`、`col=rem%S`」的逐元素整数除法/取模生成坐标。NPU **有**整数 div/mod 能力（`T.vdiv` int64、`npuir_floordiv`/`vfloordiv`、Developer `T.Parallel` 内 int64 `//`,`%`——§3.4 能力实证），故基线**可实现**（设备实测 Q1 全 PASS）；但它**逐元素标量**执行（~10 scalar op/元素 + 每元素一次 exp2），`S=2048,H=32,fp16` 实测 **39.03 ms**。改用**结构化分离**：`|i-j|` 与 head 无关，用 **int16 `T.npuir_arange` 行/列 ramp**（`[1,0]`/`[0,1]`）生成 `(bi,bj)` 坐标矩阵 → `vcast` int16→fp32 → `vsub` → `vabs` 得到**精确**距离矩阵 `D`（fp32）；`slope_h` 按 head **标量** `T.exp2(-8(h+1)/H)`（源实现式）计算。**`D` 在 head 维共享（CSE，§1.6.1 #1）**——每空间 tile 只算一次距离，`H` 次复用。同规模实测 **0.660 ms**，较基线 **快 59.1×**（设备实测 Q4）。两条坐标路径数值均与 golden 一致（Q1/Q2 均 0 违反），故裁决为**纯性能裁决**。
>
> **本版核心决策 2（调用约定，实测强制）**：源用 `out_idx=[0]` 自动分配输出。**设备探针**：`@tilelang.jit(out_idx=[...])` + `T.npuir_arange` **运行期设备崩溃**（`MTE DDR out of range`，retCode 507057；同 kernel 体改显式输出参数即 bit-exact）。故 NPU kernel 改为**显式输出参数 `main(out)`（无 `out_idx`）**，集成 wrapper 侧 `torch.empty` 分配并传入。已登记能力缺口 CG-2026-0016。
>
> **本版核心决策 3（slope API，设备探针）**：`slope_h` 用**标量 `T.exp2`**（源实现式）；设备实测 `max_abs=2.98e-8`（vs `torch.pow(2,·)` fp32，rel ≤2.5e-7@H=32）。**不使用向量 `T.vexp2`**——同设备 Developer 模式 `T.vexp2` 在 (8,64) tile 上部分元素产出 `inf`（CG-2026-0018）。
>
> **源优化意图承接（编号对应 §0.4）**：源 #1「一维 grid + 大线程数」→ **重新设计**为一维 `T.Kernel(40,is_npu=True)` + 核内 `T.serial` 网格跨步（R2）；源 #2「flat→坐标 div/mod」→ **等价替换为 arange ramp + vsub/vabs**（结构化分离；基线实测慢 59×，R1）；源 #3「slope=f(h) 标量 exp2」→ **保留**（每 head 标量 + D 共享，R3）；源 #4「fp32 计算 + 单次舍回」→ 向量 `T.vcast(rint)` 承接（R4）；源 #5「`if flat<N_total` 尾守卫」→ `task<n_tiles` 网格守卫 + 整除分块（S 编译期）；源 #6「`if_then_else` 求 abs」→ `T.vabs`（向量）/ `T.max(d,-d)`（若走 `T.Parallel` 标量路径）；源 #7「`lru_cache` 常量特化」→ 编译期 S/H 贯穿；源 #8「无 autotune」→ 保留。源**无** `lru_cache`/autotune 之外的布局优化（生成式、无输入读）。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织：先彻底读懂源算子（0.1–0.4），再做算法调研（Phase R，结论见 §1.6.0），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。

**源算子路径（源真相）**：`/home/developer/TileOPs/tileops/kernels/elementwise.py`

- 具体算子：工厂 `_make_alibi_kernel`（L3557–L3591）+ `AlibiFwdKernel(Kernel)`（L3594–L3642）：

  ```python
  @functools.lru_cache(maxsize=32)
  def _make_alibi_kernel(seq_len, num_heads, dtype, threads=256, npt=8):
      N_total = num_heads * seq_len * seq_len
      block_size = threads * npt
      S2 = seq_len * seq_len

      @tilelang.jit(out_idx=[0])                     # ★ 自动分配输出（NPU 侧实测不可与 arange 共存）
      def kernel(threads_arg, npt_arg):
          @T.prim_func
          def main(out: T.Tensor((N_total,), dtype)):  # ★ 无输入；out 展平 1-D
              with T.Kernel(T.ceildiv(N_total, block_size), threads=threads_arg) as bx:
                  for i, j in T.Parallel(threads_arg, npt_arg):
                      flat = (bx * threads_arg + i) * npt_arg + j
                      if flat < N_total:                    # 尾守卫
                          h = flat // S2                    # 整数 div/mod（NPU 可实现，见 §3.4）
                          rem = flat % S2
                          row = rem // seq_len
                          col = rem % seq_len
                          exp_val = T.cast(-8.0,"f32") * T.cast(h+1,"f32") / T.cast(num_heads,"f32")
                          slope = T.exp2(exp_val)           # fp32 slope
                          dist = T.cast(row - col, "float32")
                          abs_dist = T.if_then_else(dist > T.cast(0,"f32"), dist, -dist)
                          out[flat] = T.Cast(dtype, -slope * abs_dist)   # 单次舍回
          return main
      return kernel
  ```

- 基类/集成：`AlibiFwdKernel(Kernel)`（`supported_archs=[80,86,89,90]`、`default_config={"threads":256,"num_per_thread": npt}`，`npt=4`(fp32)/`8`(fp16·bf16)，`forward()` **无参** → `self._compiled_fn().reshape(H,S,S)`）。Op：`tileops/ops/elementwise/alibi.py::AlibiFwdOp`（**eager-only，无 `torch.library.custom_op`**；`forward()` 无参）。
- 提取产物（脚手架）：`examples/TileOPs/tileops/kernels/elementwise/alibi/_alibi_fwd_kernels.py`（Pattern B，仅含 `_make_alibi_kernel`）；集成 wrapper：同目录 `alibi.py`（`_DEFAULT_BLOCK_SIZE = {fp32:1024, fp16:2048, bf16:2048}`，K11 折叠 `threads*npt`；注释声明「NPU 组件重写为 `target="npuir"`，签名 `_make_alibi_kernel(seq_len, num_heads, dtype)`，返回可调用对象接 `block_size`」）。
- manifest：`examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `AlibiFwdOp`（L307–L360；`ref_api: "none"`；`params` `seq_len`/`num_heads`/`dtype`；workloads 4 条 `2048/4096 × 32 × {fp16,bf16}`；`roofline flops=3*H*S*S`、`bytes=H*S*S*elem_bytes`——**write-only**）。

### 0.1 源算子语义（做什么）

**数学语义**：`bias[h,i,j] = -slope_h * |i-j|`，其中 `slope_h = 2^(-8*(h+1)/H)`，`h∈[0,H)`，`i,j∈[0,S)`（manifest `AlibiFwdOp`；ALiBi 位置编码偏置，Attention with Linear Biases）。**无张量输入**——输出完全由构造期整数参数 `S=seq_len`、`H=num_heads` 与 `dtype` 合成。

> **⚠ 源 docstring / 实现不一致（本设计的关键陷阱，显式标注）**：工厂 `_make_alibi_kernel` 的 docstring（L3561）写 `slope_h = 2^(-8*h/H)`（指数为 `-8h/H`，h 从 0 起 → `h=0` 时 slope=1.0），而**实现**（L3582 `exp_val = T.cast(-8.0,"float32") * T.cast(h+1,"float32") / T.cast(num_heads,"float32")`）为 `2^(-8(h+1)/H)`（h 从 0 起但用 `h+1`）。**`AlibiFwdKernel` 类 docstring（L3598）写 `2^(-8*(h+1)/H)`，与实现一致**；golden 测试 `test_alibi` 用 `torch.arange(1, H+1)` 亦为 `h+1`。故唯一自相矛盾的是工厂函数 docstring。**本设计以「实现 + golden」为准**（`2^(-8(h+1)/H)`）；若误信工厂 docstring 将得 `h=0 → slope=1.0`（偏 2^(8/H) 倍）的错误结果。本差异已在 §0.4 与 §3.4 交叉标注，供 Stage 3 与维护者核对。

**契约语义 = 测试 golden**（`examples/TileOPs/tests/ops/test_special_elementwise.py::test_alibi`，L46–L61）：

```python
positions = torch.arange(seq_len, dtype=torch.float32)
dist = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()          # (S,S) fp32
slopes = torch.pow(2.0, -8.0*torch.arange(1,H+1,dtype=torch.float32)/H) # (H,) fp32
ref = (-slopes[:,None,None] * dist[None,:,:]).to(dtype)                 # 单次舍回
```

即：**fp32 域**算 `|i-j|`、`slope_h`、乘积，末端**单次舍回** `dtype`；`torch.testing.assert_close` 容差 fp16 `atol=rtol=1e-2`、fp32 `1e-5`。用例 `(seq_len=128, num_heads=8)` × `{fp16, fp32}`。Bench 参考 `_alibi_reference`（`bench_independent_elementwise.py`）与 golden 同式。

**dtype 语义（迁移必须保持的关键点）**：源用 `T.cast(row-col, "float32")` + `exp_val` fp32 + `T.exp2` fp32 + `T.Cast(dtype, -slope*abs_dist)`——**全程 fp32 计算、末端单次舍回**。与 golden 的 fp32 opmath 语义**一致**（本算子**无** leaky_relu 那种「源输入 dtype 域 vs 契约 fp32 域」分歧）。

**`slope_h` 语义**：`2^(-8(h+1)/H)`，fp32。`H=8` 时指数为整数 → slopes 恰为 2 的整数次幂（`1/2,1/4,…,1/256`，fp32/fp16 精确可表示）；`H=32` 时指数为 `-(h+1)/4`（0.25 的倍数），非精确幂。

**规约语义**：**无规约**（逐元素映射，无跨元素依赖、无累加顺序问题）。

**边界语义**：`i=j` → `|i-j|=0` → 输出 `-slope·0 = 0`（精确 0）；`|i-j|` 最大 `S-1`；无 NaN/Inf 产生（无输入张量、无除零、无超越函数上溢——`slope≤2^(-8/H)≤0.84`，`|i-j|≤4095`，乘积有限）。无非整除边界（输出 shape 由 S 决定）；S 非整除分块时由任务网格守卫处理（§5.4）。

**host 侧语义**：`AlibiFwdKernel.__init__` 由 `(seq_len, num_heads, dtype)` 构造；`forward()` **无参**调用编译后 kernel；`AlibiFwdOp.forward()` 把 1-D 输出 `reshape(H,S,S)`。**无 host 预处理**（无输入、无 flatten）。

**语义保持基线**：§8.1 golden 以本节语义（移植 `test_alibi` 的 torch 参考式）为唯一依据实现，**不得复刻 §0.6 的 NPU 结构决策**（保证验证独立性）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `out` | `(N_total,)`，`N_total = H*S*S`（host 侧 `reshape(H,S,S)`） | `float16 \| bfloat16 \| float32` | **唯一张量参数（输出）**；本算子**无张量输入** |

**输出 shape（迁移必需字段）**：`(num_heads, seq_len, seq_len)`（`source_output_shape`；manifest `outputs.output.shape = "[num_heads, seq_len, seq_len]"`）。
**layout**：输出为 `(H,S,S)` 连续内存（row-major）；源展平为 1-D `(N_total,)` 逐元素写。
**动态轴**：无（S、H 均为构造期编译期常量；`lru_cache` 按 `(seq_len,num_heads,dtype)` 特化）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源 `T.Parallel(threads,npt)` 内逐元素标量；源码每个计算语句均归入下表）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `flat = (bx*threads+i)*npt+j` | grid/thread 索引 | 标量 `flat` | 元素定位 |
| 2 | `h=flat//S²`；`rem=flat%S²`；`row=rem//S`；`col=rem%S` | `flat` | 标量 `h,row,col` | 坐标反解（flat→(h,i,j)） |
| 3 | `exp_val = -8*(h+1)/H`；`slope = T.exp2(exp_val)` | `h,H` | 标量 `slope` (fp32) | `slope_h` |
| 4 | `dist=cast(row-col,f32)`；`abs_dist=if_then_else(dist>0,dist,-dist)` | `row,col` | 标量 `abs_dist` | `\|i-j\|` |
| 5 | `out[flat] = T.Cast(dtype, -slope*abs_dist)` | `slope,abs_dist` | GM `out[flat]` | 乘积 + 单次舍回 |

**数据流与内存访问模式**（源硬件 = GPU）：

```
（无输入）
grid(bx) × threads(256) × npt(8/4)  →  逐线程标量：div/mod 反解坐标 → exp2 slope → |row-col| → 乘 → 舍回
   → GM[out]（coalesced 连续写，write-only；无 SMEM、无 fragment 显式缓冲）
```

**循环与并行结构**：`T.Kernel(T.ceildiv(N_total, block_size), threads=256)`——一维 grid（`bx`）+ 256 线程/block 两级并行；每线程 `npt` 元素；`T.Parallel(threads_arg, npt_arg)`。`block_size = 256×npt`：fp32 `npt=4`→1024，fp16/bf16 `npt=8`→2048。

**host 侧逻辑**：`N_total`、`dtype`、`S`、`H` 由 Op/Kernel 层传入；`forward()` 调编译后 kernel；`reshape(H,S,S)` 由 Op 层完成；工厂 `functools.lru_cache` 按 `(seq_len,num_heads,dtype)` 特化编译。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | 一维 grid + 大线程数（coalesced 连续写） | 用大规模线程/grid 吃满写出带宽 | `T.Kernel(ceildiv(N,bs), threads=256)` + `T.Parallel`；`out[flat]` 连续 | GPU thread/warp 执行模型 + memory coalescing | **硬件强相关**（GPU 线程模型） |
| 2 | **flat→坐标 逐元素 div/mod** | 由 1-D flat 索引反解 (h,i,j) | 标量整数 `//`/`%` | GPU 逐元素整数 ALU | **实现层**（NPU **有** int64 div/mod，但仅逐元素标量，见 §0.5/§3.4） |
| 3 | **slope 只依赖 h（`T.exp2` fp32）** | 每元素一次超越函数 | 标量 `exp2`（无查表/共享） | GPU 超越函数单元 | **算法层**（数学结构可移植：slope 与 (i,j) 无关） |
| 4 | **fp32 计算 + 单次舍回** | 精度对齐 golden | `cast` fp32 → 算 → `T.Cast(dtype,·)` | fp32 ALU | **语义层**（可移植；NPU `vcast(rint)` 承接） |
| 5 | `if flat < N_total` 尾守卫 | 覆盖非整除 N | 标量条件 | 无硬件依赖 | 可移植（NPU 网格守卫） |
| 6 | `if_then_else` 求 abs（避免 `T.abs` 整数问题） | 逐元素绝对值 | 标量二路选择 | GPU 逐元素 ALU | **算法层**（NPU `T.vabs` 原生；`T.Parallel` 标量路径需 `T.max(d,-d)` 替代，见 §3.4） |
| 7 | 工厂 `lru_cache` 按 `(seq_len,num_heads,dtype)` 特化 | 消运行时参数分派 | compile-time 常量固化 | 无硬件依赖 | 可移植（NPU 侧同用编译期常量） |
| 8 | 无 autotune、固定启发式 `npt` | 带宽受限 + 单步，搜索 ROI 低 | `default_config` 固定 | 无硬件依赖 | 可移植（NPU 固定单实现 + §5.2 block 选择） |

> **识别不出来 ≠ 不存在**。上述 8 项为源码可识别的全部优化/规避手段；后续迁移逐项处置（§0.5）。注意源**没有**为「head 无关的 `|i-j|` 复用」做任何优化（源每元素重算 `slope` 与 `dist`）——这是 §0.6/§1.6.1 引入的**新**优化。另：源 docstring 与实现在 slope 指数上不一致（§0.1），本设计以实现为准。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定输入含 Phase R 调研结论 §1.6.0）

> **本版重要修正**：v0 以「NPU 无向量 div/mod」「`T.Parallel` 写 UB 不可用」两条**错误**负向断言把源 #2（flat div/mod）与源 `T.Parallel` 结构判为「不可实现」。**本版依据下表所列的本地文档/源码/在仓测试/目标设备实测证据**，认定二者**均可实现**；源 #2 的处置改为**「重新设计」——但依据是性能（实测慢 59×）而非能力缺失**。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义 `bias[h,i,j]=-slope_h*|i-j|`（含 I/O、fp32 计算域、单次舍回、边界） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 依 §0.1；计算域与舍回路径均保持 |
| 计算算法（步骤 1–5 逐元素映射；无规约、无跨元素依赖） | 算法 | 无 | — | **保留（结构）** | 逐元素映射结构保留；仅并行/内存/坐标生成重设计 |
| 优化 #2 flat→坐标 逐元素 div/mod | 实现/算法 | GPU 逐元素整数 ALU | **有**：int64 向量 div（`T.vdiv` int64 √，`T.vdiv.md` §2.2.1）、地板除（`npuir_floordiv`/`vfloordiv`，`customize_npuir.py:425`）、`T.Parallel` 内 int64 `//`,`%`（`testing/npuir/parallel_ops/test_floordiv_floormod_parallel.py`）、目标设备实测 Q1 PASS | **重新设计**（**性能依据**，非能力缺失） | → §0.6 R1：结构化分离（arange 行/列 ramp + vsub/vabs），`|i-j|` 与 head 无关；基线可实现但**逐元素标量**，实测 39.03 ms vs arange 0.66 ms（2k/32/fp16，Q4，快 59.1×） |
| 优化 #1 grid+thread 两级并行（大 grid 吃带宽） | 优化 | GPU thread/warp 模型 | 无直接等价（NPU 一维 Kernel；纯 Vector 核 40） | **重新设计** | → §0.6 R2：一维 `T.Kernel(40,is_npu=True)` + 核内 `T.serial` 网格跨步 |
| 优化 #3 slope=f(h)（`T.exp2` fp32，与 (i,j) 无关） | 算法 | GPU 超越函数单元 | **有**：标量 `T.exp2`（设备实测 1 ulp vs `torch.pow`，Q3） | **保留**（结构落地面见 §0.6 R3，属「保留」层） | → §0.6 R3：把 slope 提升为**每 head 标量**（`T.serial(H)` 内一次），并借「D 与 h 无关」在 head 维**复用 D**（§1.6.1 #1）；**不使用**向量 `T.vexp2`（设备实测 inf，CG-2026-0018） |
| 优化 #4 fp32 计算 + 单次舍回 | 优化/语义 | 无 | 有（`T.vcast(rint)` f32↔f16/bf16） | **等价替换**（结构落地面见 §0.6 R4） | → §0.6 R4：`vcast(rint)` 单次舍回；fp32 路径直写 |
| 优化 #6 `if_then_else` 求 abs | 算法 | GPU 逐元素 ALU | 有（`T.vabs` fp32，向量；`T.max(d,-d)` 标量等价式，设备实测 Q1 PASS） | **等价替换** | → §0.6 R1：向量路径 `T.vabs`（fp32 距离矩阵）；`T.Parallel` 标量路径须用 `T.max(d,-d)`（`if_then_else` 在 `T.Parallel` 内崩溃，§3.4） |
| 优化 #5 尾守卫 `if flat<N_total` | 优化 | 无 | 有 | **等价替换** | `task<n_tiles` 任务网格守卫 + 整除分块（§5.4） |
| 优化 #7 `lru_cache` 常量特化 | 优化 | 无 | 有 | **保留** | NPU 工厂保持 `(seq_len,num_heads,dtype)` 编译期常量；`S`/`H` 参与 index/循环静态化 |
| 优化 #8 无 autotune | 优化 | 无 | 有 | **保留** | NPU 侧固定单实现 + §5.2 block 选择 |
| `out_idx=[0]` 自动输出分配 | 优化/host | 无（框架机制） | **无（实测冲突）** | **重新设计** | → §0.6 R5：改**显式输出参数**（无 `out_idx`），wrapper 侧 `torch.empty` 分配；依据设备探针（CG-2026-0016） |
| 源 `T.Parallel(threads,npt)` 并行结构（作为整体形态） | 优化/结构 | GPU 线程模型（threads 语义） | **有（写 UB 可用）**：`T.Parallel.md` §2.3–2.4 官方 `@tilelang.jit(target="npuir")`+`T.Parallel`+`alloc_shared` 示例；`testing/npuir/parallel_ops/`（12 个 Developer 测试） | **重新设计**（结构形态；性能依据） | `T.Parallel` 写 UB **可用**（设备实测 Q1：baseline kernel 全 PASS）；但源 `T.Parallel(threads=256)` 的 `threads=` 语义在 npuir 无效果（TRAP-threads-kwarg-noop）；且逐元素标量形态慢 59× → R1/R2 改用 arange + 向量链 |
| fp8（`_get_fp8_output_dtypes`） | — | — | 无（manifest 仅 fp16/bf16/fp32） | **舍弃** | `AlibiFwdKernel.SUPPORTED_DTYPES` 仅三种 float；fp8 不在契约内 |

> **依据汇总（证据链）**：
> - **div/mod 可用**：`docs/Tilelang.language/数学操作/T.vdiv.md` §2.2.1（Ascend 行 int64 √、fp16/fp32 √；int16/int32/uint* ×）；`tilelang/language/customize_npuir.py:158 npuir_div`（导出 `vdiv`，`__init__.py:82–83`）、`:425 npuir_floordiv`（导出 `vfloordiv`，`__init__.py:98–99`）；`testing/npuir/parallel_ops/test_floordiv_floormod_parallel.py`（Developer `T.Parallel`+`alloc_shared`，int64 `//`/`%`，`rtol=0,atol=0` 精确；commit f987008）——**目标设备实测重跑 12/12 passed**。
> - **`T.Parallel` 写 UB 可用**：`docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.3（"循环体内的变量必须是 alloc 申请出的 UB 上的 Buffer"）、§2.4（官方 `@tilelang.jit(target="npuir")`+`T.Parallel(block_M, block_N)`+`alloc_shared` 示例）；`testing/npuir/parallel_ops/`（`test_cast_parallel.py`、`test_min_max_parallel.py`、`test_unary_math_parallel.py`、`test_floordiv_floormod_parallel.py`、`test_loop_var_compute.py` 等 12 个 Developer 测试）——**目标设备实测**（Q1 baseline `T.Parallel` kernel 全 PASS）。
> - **`T.Parallel` 的文档限制**：`T.Parallel.md` §2.2.1「当前不支持直接使用循环变量的条件语句，例如 `T.if_then_else(i>j, xxx, xxx)`」+ §2.3「`T.Parallel` 内下标不能做变换」——目标设备实测 V4/V5/V7 复现编译期 segfault，V6/V6b/V8/V9（`T.max(d,-d)` / inline `T.fabs`）可用（§3.4）。
> - **标量 `T.exp2` 可用**：`tilelang/language/tir/op.py:1837`；设备实测 Q3（1 ulp）。
> - **`T.vabs` 无 int16**：`docs/Tilelang.language/数学操作/T.vabs.md` §2.2.1（fp16/fp32 ✓，int16 ×）。
> - R5 由设备探针记录（CG-2026-0016）——**唯一**成立的能力性（而非性能性）否决。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（承接源方案性能意图 + 语义保持论证）。

**重设计项 R1：坐标生成重设计（flat→div/mod → arange ramp + vsub/vabs，距离矩阵 head 维共享）** ⭐（本算子最关键）

- **源方案**：对每个 flat 元素做 4 次整数 div/mod 反解 `(h,row,col)`；性能意图 = 无额外缓冲地由 flat 索引得到坐标。
- **基线可行性（本版修正）**：源基线在 NPU **可实现**——`T.Parallel(block)` + `alloc_shared`（UB）+ int64 `//`,`%`。**但**：① 逐元素**标量**执行（~10 scalar op/元素 + 每元素一次 `exp2`）；② 源 `if_then_else(dist>0,dist,-dist)` 形态在 `T.Parallel` 内**编译期 segfault**（文档 §2.2.1 明示条件语句不支持循环变量），须改用 `T.max(dist,-dist)` 等价式；③ 1-D flat 分块 `block` 需 ≥ `ceil(N_total/65535)`（coreDim ≤65535 上限；2k/32 时 N=134M → block≥2049，实测 block=2048 触发 `KernelLaunch failed ... coreDim ... 65536`，block=4096 可行）。**目标设备实测修正后的基线全 PASS**（Q1：`(128,8)` bit-exact / fp32 3.815e-6 / `(2048,32)` 3.906e-3，0 违反）。
- **NPU 新算法**：**结构化分离**——`|i-j|` 只依赖 `(i,j)`、`slope_h` 只依赖 `h`。以 `(bi,bj)` 空间 tile 为单位：
  1. `T.npuir_arange(R, [1,0], i0)` → `R[r,c]=i0+r`（行索引）；`T.npuir_arange(C, [0,1], j0)` → `C[r,c]=j0+c`（列索引），均为 int16（索引 ≤4095 < 32767 精确）；
  2. `T.vcast(R, Rf, rint)`、`T.vcast(C, Cf, rint)`（int16→fp32，精确）；
  3. `T.vsub(Rf, Cf, Rf)`；`T.vabs(Rf, Rf)` → `D[r,c]=|(i0+r)-(j0+c)|`（fp32，整数精确）；
  4. **`D` 在 head 维复用**：`h` 循环内只做 `T.vmul(D, slope_h_neg, O)`（+`vcast`），不再重算距离。
- **语义保持论证**：`D` 与 `h` 无关（数学恒等，§1.6.1 #1）；int16 ramp 对 `|i-j|≤4095` 精确（int16 上界 32767）；int16→fp32 无损；vsub/vabs 对整数精确。设备 Q1（基线）/Q2（arange）端到端对比：两条路径**同 golden、同 0 违反**（Q2 `test` fp16 bit-exact、fp32 max_abs 3.815e-6<1e-5；`manifest-2k` fp16 3.906e-3 / bf16 4.0（1 ulp）0 违反）。边界：`i=j`→`D=0`→输出精确 0（golden 同）。
- **源优化意图承接**：源 #2「由 flat 反解坐标」的意图由「arange ramp 直接给坐标矩阵」承接，且**额外**省去 head 维重复距离计算（源无此复用）。**裁决依据 = 实测性能**（Q4：arange 0.660 ms vs 基线 39.029 ms，2k/32/fp16，**快 59.1×**；两条路径数值一致，故为纯性能裁决）。

**重设计项 R2：并行结构重设计（thread 级并行 → 一维 Kernel + 核内网格跨步）**

- **源方案**：`T.Kernel(ceildiv(N,bs), threads=256)` + `T.Parallel(256,npt)`；意图 = 大规模并行吃满写出带宽。
- **NPU 新算法**：一维 `T.Kernel(core_num, is_npu=True)`（`core_num` = 实查物理 Vector 核数 **40**，§5.5 ②）+ 核内 `for t in T.serial(T.ceildiv(n_tiles, core_num))` 网格跨步（`task=t*core_num+cid`，`if task<n_tiles` 守卫）；每 tile 由 `(i0,j0)` 定位，距离链 + head 循环。
- **语义保持论证**：任务映射只重排「哪个核处理哪个 (i,j) 空间 tile」，各输出元素独立（无跨元素依赖、无累加顺序），数学语义逐元素恒等；`T.serial` 上界 = 编译期静态值（`n_tiles`、`core_num` 均编译期常量），守卫覆盖非整除尾任务，不越界。
- **模式先例**：`examples/elementwise/vec_add_2d.py` L28–L42；`PL-1.22-persistent-1d-grid-stride`。

**重设计项 R3：slope 计算与强度削减（每元素 exp2 → 每 head 标量 exp2；D 共享）**（§0.5 处置层 = 「保留」——此处为保留层内的结构落地）

- **源方案**：每个元素一次 `slope = T.exp2(-8(h+1)/H)`（GPU 超越函数单元）。
- **NPU 新算法**：`slope` 提升为 **每个 head 一次的标量**：`for h in T.serial(H): e = T.cast(-8/H,f32)*T.cast(h+1,f32); slp = -T.exp2(e); T.vmul(D, slp, O)`。**不使用**向量 `T.vexp2`（设备实测 (8,64) tile 部分元素 `inf`，CG-2026-0018）。
- **语义保持论证**：`slope` 与 (i,j) 无关（数学恒等，§1.6.1 #1/#4）；标量 `T.exp2` 与 golden `torch.pow(2,·)` 设备实测 1 ulp 内（rel ≤2.5e-7@H=32，Q3）；`-(slope)*D` 与 golden `(-slope)*dist` 逐位一致（§1.6.1 #3）。设备 Q2 证。
- **源优化意图承接**：源 #3「每元素一次 exp2」→ 降为「每 head 一次」+ D 共享，op 数由 `H·(每元素 exp2)` 量级降为 `H·(每 tile 一次)`（§1.6.1 收益量化）。

**重设计项 R4：dtype 路径（标量 `T.cast` → 向量 `T.vcast` fp32 域 + 单次舍回）**（§0.5 处置层 = 「等价替换」——此处为等价替换层内的结构落地）

- **源方案**：标量 `T.cast(row-col,"f32")`、`T.Cast(dtype,·)` 单次舍回。
- **NPU 新算法**：fp32 路径直写 `O`→GM；fp16/bf16 路径 `T.vcast(O, O16, rint)` 单次舍回后 `T.copy(O16, out[...])`。dtype 矩阵：`vcast` f32→f16/bf16 支持 `rint`（`docs/.../T.vcast.md` §2.2.1）。
- **语义保持论证**：fp32 域算后单次 `rint` 舍回 ≡ 源/golden 语义；设备 Q2（fp16 `test` bit-exact、fp32 3.815e-6）。

**重设计项 R5：调用约定（`out_idx` 自动输出 → 显式输出参数）** ⭐（实测强制，唯一能力性否决）

- **源方案**：`@tilelang.jit(out_idx=[0])`，输出由框架自动分配。
- **NPU 新算法**：`@tilelang.jit(target="npuir")`（**无 `out_idx`**），`main(out: T.Tensor((H,S,S), dtype))`；集成 wrapper 侧 `out = torch.empty(H,S,S,...)`，调用 `kernel(out)`。
- **语义保持论证**：仅改变输出张量的分配位置（框架 vs wrapper），输出 shape/dtype/layout 与内容不变；设备实测用该形态全 PASS（Q1/Q2）。
- **依据**：设备实测 `out_idx=[0]` + `T.npuir_arange` 运行期设备崩溃（`MTE DDR out of range`，retCode 507057；同 kernel 体显式输出即 bit-exact）；`testing/npuir/creation_ops/test_arange.py` 亦用显式输出。CG-2026-0016。

> **承接说明（源优化意图不丢弃）**：§0.4 #1「大规模并行吃带宽」→ R2 网格跨步 + 大写出连续性承接；#2「flat 反解坐标」→ R1 arange ramp 承接（且新增 head 维 D 复用；裁决依据从「不可实现」改为实测性能）；#3「每元素 exp2」→ R3 每 head 标量 exp2 + D 共享承接（强度削减）；#4「fp32 + 单次舍回」→ R4 vcast 承接；#5 尾守卫 → §5.4 网格守卫；#6 abs → R1 `T.vabs`（向量）/ `T.max(d,-d)`（标量 `T.Parallel` 路径）；#7 常量特化 → 编译期 S/H 贯穿；#8 无 autotune → 保留。

### 0.7 标杆实现

- 源算子文件（golden 语义依据）：`/home/developer/TileOPs/tileops/kernels/elementwise.py`（`_make_alibi_kernel` L3557–L3591、`AlibiFwdKernel` L3594–L3642）；Op 层 `/home/developer/TileOPs/tileops/ops/elementwise/alibi.py`。
- 源仓测试基准：`examples/TileOPs/tests/ops/test_special_elementwise.py::test_alibi`（`seq_len=128, num_heads=8` × `{fp16, fp32}`；ref = fp32 `pow(2,·)`/`|i-j|`/乘积 → `.to(dtype)`；`assert_close` 容差 fp16 1e-2 / fp32 1e-5）。
- manifest：`examples/TileOPs/tileops/manifest/elementwise.yaml` 的 `AlibiFwdOp`（workloads `{2048,4096}×32×{fp16,bf16}`；roofline `flops=3*H*S*S`、`bytes=H*S*S*elem_bytes`）。
- golden 以 §0.1 语义为依据（移植 `test_alibi` 的 torch 参考式），**不复刻 §0.6 的 NPU 结构**（保证验证独立性）。
- 同族迁移先例（只读参考）：`examples/leaky_relu/_make_leaky_relu_kernel/`、`examples/elu/_make_elu_kernel/`、`examples/erf/erf_fwd_kernel/`（有输入逐元素；结构对照）。
- 索引生成先例（只读）：`testing/npuir/creation_ops/test_arange.py`；`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` L421–435（`T.arange` 行/列 ramp 求 `i-j`）；`examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/_ssd_chunk_scan_fwd_kernel.py` L349–353（`T.arange`+`vsub`+`vmin`）。
- **div/mod + `T.Parallel` 先例（本版新增，只读）**：`testing/npuir/parallel_ops/`（12 个 Developer 测试；`test_floordiv_floormod_parallel.py` = int64 `//`,`%`，`test_cast_parallel.py`、`test_min_max_parallel.py`、`test_unary_math_parallel.py`、`test_loop_var_compute.py` 等）。

---

## 1. 概述

### 1.1 算子名称

`_make_alibi_kernel`（项目 `alibi`，目录 `examples/alibi/_make_alibi_kernel/`；集成类 `AlibiFwdKernel` / `AlibiFwdOp`）。

### 1.2 功能描述

生成 ALiBi（Attention with Linear Biases）位置编码偏置张量 `bias[H,S,S]`：`bias[h,i,j] = -slope_h * |i-j|`，`slope_h = 2^(-8*(h+1)/H)`。**生成式算子（无张量输入）**——输出完全由构造期整数参数 `S=seq_len`、`H=num_heads` 与 `dtype` 合成，支持 `float16 | bfloat16 | float32`。纯 Vector、逐元素、write-only（无 GM 读、无规约、无矩阵运算）；核内需**索引生成**（坐标矩阵）是本算子区别于有输入逐元素算子的核心。

**迁移决策后的 NPU 侧算法与源算法的关键差异**：① 坐标生成由「flat→div/mod 逐元素标量」改为「int16 arange 行/列 ramp + vsub/vabs 向量链」；② `|i-j|` 距离矩阵在 head 维**复用**（源每元素重算）；③ `slope` 由每元素 `exp2` 降为每 head 标量 `exp2`；④ 调用约定由 `out_idx` 自动输出改为显式输出参数（实测强制）；⑤ 并行/内存由 GPU grid+thread 改为 UB + 一维核 + 网格跨步。计算语义与源一致（逐元素 `-slope_h*|i-j|`，fp32 域 + 单次舍回）。**坐标生成差异为性能裁决（实测基线慢 59×）**，两条路径数值一致（§1.6.0 R3/R4）。

### 1.3 数学公式

$$
\text{bias}[h,i,j] = -\text{slope}_h \cdot |i-j|,\qquad
\text{slope}_h = 2^{\,-8\,(h+1)/H},\qquad h\in[0,H),\ i,j\in[0,S)
$$

NPU 计算核（§1.6.1 优化后；`D[i,j]=|i-j|` 在 head 维共享，`s_h=-\text{slope}_h` 为每 head 编译期/设备标量）：

$$
\text{bias}[h,i,j] = s_h \cdot D[i,j],\qquad
D = \operatorname{vabs}\big(\operatorname{vsub}(\operatorname{vcast}_{f32}(R),\ \operatorname{vcast}_{f32}(C))\big),\quad
R[r,c]=i_0{+}r,\ C[r,c]=j_0{+}c
$$

### 1.4 算法描述

**本节描述迁移决策后的 NPU 侧算法（§0.6），与源算法在结构上等价、在坐标生成/并行/调用约定上有差异：**

- **坐标生成**（§0.6 R1）：`(bi,bj)` 空间 tile 上，`T.npuir_arange` 生成 int16 行/列 ramp（`R[r,c]=i0+r`、`C[r,c]=j0+c`）→ `T.vcast(rint)` 升 fp32 → `T.vsub` → `T.vabs` 得 `D[r,c]=|(i0+r)-(j0+c)|`（fp32，精确）。
- **head 维复用**（§0.6 R1 / §1.6.1 #1）：`D` 与 `h` 无关，每空间 tile 只算一次；`h` 循环内仅 `vmul(D, s_h, O)`（+`vcast`）。
- **slope**（§0.6 R3）：`for h in T.serial(H): s_h = -T.exp2(-8*(h+1)/H)`（标量，源实现式；**不用**向量 `T.vexp2`）。
- **dtype**（§0.6 R4）：fp32 域计算；fp16/bf16 末端 `T.vcast(rint)` 单次舍回。
- **并行**（§0.6 R2）：一维 `T.Kernel(40, is_npu=True)` + 核内 `T.serial` 网格跨步。

除上述外，算法与 §1.6.0 调研选定的算法族（结构化分离的单遍 write-only 生成）一致：一趟写出、无中间 GM 往返、无跨元素依赖、无跨核归并。

### 1.5 数据流图

```
（无输入；S、H 为编译期常量）
with T.Kernel(40, is_npu=True)：
  GM 无读
  空间 tile (i0,j0)：
    arange(R[i16]) ─┐
    arange(C[i16]) ─┴─ vcast(rint)→fp32 ─ vsub ─ vabs ─→ UB D[bi,bj] (fp32)
    for h in serial(H):
        s_h = -exp2(-8(h+1)/H)                  （标量）
        O = vmul(D, s_h)                        （fp32）
        [fp16/bf16] O16 = vcast(O, rint)        （单次舍回）
        T.copy(O16 | O, out[h, i0, j0]) ────────→ GM out[H,S,S]（write-only）
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（对同一数学语义的算法族回答调研四问，选定算法族），再在数学等价前提下优化公式，再做循环/标量计算的向量化替代。迁移任务：1.6.1–1.6.3 的分析对象为 §1.6.0 调研选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 本算子为**生成式单步逐元素类**（无输入、无规约/窗口/矩阵、无跨元素依赖），按 `_shared/standards/algorithm-research.md` §3 属轻量调研范畴；但因**坐标生成（索引数学）**是本算子的核心分歧点、且存在多种等价结构（含**可实现**的 div/mod 基线），R1 给出候选表、R3 给出四口径对比、R4 逐候选评估（达到完整调研深度）。
>
> **本版重要修正**：v0 的 R1 把基线（flat div/mod）标为「NPU 不可直接实现」，属**错误负向断言**（见 §0.5 证据链）。本版把基线列为**可实现候选**，并以**目标设备实测的性能/精度/可实现性**裁决。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 可实现性 | 收益方向 / 实测证据 | 纳入 R3 对比 |
|---|------|------------|-----------|---------|---------------------|-------------|
| 0 | **基线：源算法**（flat→div/mod 逐元素标量 + 每元素 exp2） | `flat→(h,row,col)`；`out=-exp2(-8(h+1)/H)·\|row-col\|` | 迭代定义式（fp32 域） | ✅ **可实现**（`T.Parallel`+UB+int64 `//`,`%`；Q1 全 PASS） | — | ✅（基线；实测 39.03 ms @2k） |
| 1 | **结构化分离 + arange ramp + D head 维共享（采纳）** | `D=abs(vsub(vcast(R),vcast(C)))`；`out[h]=s_h·D` | 数学恒等（`D` 与 `h` 无关；`s_h=-slope_h`） | ✅（Q2 全 PASS） | 距离生成指令降 `H×`；无逐元素 div/mod；实测 **0.660 ms**（快基线 59.1×） | ✅（采纳） |
| 2 | 不共享 D（每 head 重算距离） | 每 (h,i,j) 现算 `\|i-j\|` | 数学恒等 | ✅ | 距离生成指令 `×H`（劣于 #1） | ✅（对照，劣） |
| 2b | div/mod **向量化**（flat ramp + `vfloordiv`/`vfloormod` int64） | `arange(F,[bj,1],base)` → int64 vfloordiv/vfloormod → 坐标；slope 需向量 exp2/gather | 数学恒等 | ⚠️ 可构造 | int64 中间（8B/elem，UB ×2）+ 需可靠的向量 exp2（`vexp2` 不可靠）+ 仍需 flat ramp（即仍需 arange）→ 复杂度高于 #1 无增益 | ✅（对照，劣） |
| 3 | 距离用 `max(i,j)-min(i,j)` | `vsub(vmax(R,C), vmin(R,C))` | 数学恒等（整数域） | ✅ | 与 `vsub+vabs` 同 op 数（vmax+vmin+vsub=3 vs vsub+vabs=2），更差 | ✅（劣） |
| 4 | host 编译期 slope 常量 | Python `-2**(-8(h+1)/H)` 作 vmul 标量 | 数学恒等 | ❌ | `for h in range(H)` 被 TVMScript 降为 `T.serial`（h 为 Var），列表索引/枚举均报错（v0 实测） | ❌ 不可行 |
| 5 | 向量 `T.vexp2` 计算 slope | `slope_vec = vexp2(exponents)` | 容差内等价 | ❌ | 设备实测 `inf`（(8,64) tile 部分元素，CG-2026-0018） | ❌ 设备不可靠 |
| 6 | fp16 arange ramp（替代 int16） | `R,C` fp16 | **非精确** | — | fp16 仅能精确表示 ≤2048 及 2048 以上偶数，S=4096 时 `4095` 不可表示 | ❌ 正确性否决（S>2048） |
| 7 | 距离 profile gather（查表） | 预存 `D` profile 再 gather | 数学上可等价 | ❌ | 需核内 gather + 索引 math（无核内向量 gather 先例，见 R4） | ❌（负向，见 R4） |
| 8 | output 由「外积」构造 | `out[h]=row_scale ⊗ col_scale` | **非等价**（`\|i-j\|` 不可分解为外积） | — | — | ❌ 数学不成立 |
| 9 | 源 `T.Parallel(threads,npt)` 直译（含 `if_then_else` abs） | 逐元素 flat 结构整体移植 | 数学等价 | ⚠️ **部分**：`if_then_else(dist>0,…)` 在 `T.Parallel` 内**编译期 segfault**（§3.4 设备实测 V4/V7；`T.Parallel.md` §2.2.1 明示不支持循环变量条件）；`T.max(d,-d)` 等价式可用 | 逐元素标量，慢 59× | ✅（对照，= #0 形态，劣） |

**R2 在线算法**：**无在线化收益**——生成式单遍写出算子，本身即「天然单遍」（无 GM 读、无跨元素依赖、无 running 统计量）。结构判据：本算子无归约目标（无 max/sum/running），在线变体的「分块 running + 重缩放」定义不适用。依据：源算法即单遍直写（§0.3 步骤 1–5 无循环携带依赖）；manifest roofline `bytes=H*S*S*elem_bytes` 为 write-only 理论下界；`ALG-elementwise`（algorithm-candidates.md）记「逐元素映射天然单遍」。

**R3 复杂度对比**（四口径；本算子 **write-only、发射/带宽受限**，访存为理论下界，差异在**指令数（发射成本）**与**正确性**）：

| 算法候选 | FLOPs（op 形态 / 输出元素） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 | **实测 (2k/32/fp16)** |
|---------|------------------------------|---------------|---------|-------------|---------------------|----------------------|
| **#0 基线（源，flat div/mod 标量 + 每元素 exp2）** | **~10 scalar op/元素**（4 int64 div/mod + exp2 + sub + abs + mul + neg + cast） | `H·S²·elem_bytes`（无读） | 写出 1 遍 | O(`block`) UB（1-D） | 逻辑块 `N/block`（≥65536 需 block≥2049）；标量吞吐受限 | **39.029 ms** |
| **#1 结构化分离 + D 共享（采纳）** | `(6 + 2H)/(H·bi·bj)` 每元素 ≈ `2/(bi·bj)`（H 大时）；**距离项 6/(H·bi·bj)** | `H·S²·elem_bytes`（无读） | 写出 1 遍 | 12 B/elem × `bi·bj`（实测，§4.5） | 可分块多核；无跨核归并 | **0.660 ms** |
| #2 不共享 D | `(8H)/(H·bi·bj)` ≈ `8/(bi·bj)`（距离项不摊薄） | 同 | 1 遍 | 同 | 同 | —（构造上劣） |
| #2b div/mod 向量化 | ~8 向量 op/元素 + slope 处理 | 同（int64 中间双倍 UB） | 1 遍 | ~20 B/elem | 同 | —（构造上劣） |
| #3 max−min 距离 | 同 #1 +1 op/tile | 同 | 1 遍 | 同 | 同 | — |
| #5 向量 vexp2 slope | 同 #1（slope 摊到向量） | 同 | 1 遍 | 同 | **设备不可用** | — |
| #6 fp16 ramp | 同 #1 | 同 | 1 遍 | 同 | **S>2048 不精确** | — |

> 口径说明：FLOPs 以「每输出元素的 op 数」计（生成式无输入读，`exp2` 按 1 op 记；#1 的距离 ramp 生成按每 tile 固定 op 数摊到 `H·bi·bj` 输出元素）。访存 = **只写** `H·S²·elem_bytes`（理论下界，无中间 GM 往返）；扫描遍数 = 写出 1 遍；中间缓冲 = UB 单空间 tile（非 O(N)）。**全部可行候选访存同为下界**，差异在**指令数**——#1 因 D 共享把距离项摊薄 `H×` 且以向量指令整 buffer 执行，是唯一在默认 H=32 下距离开销可忽略、且发射成本最低的形态；#0 因逐元素标量（~10 op/元素）而指令数爆炸，**实测慢 59.1×**（Q4）。
>
> **柱状对比（S=2048,H=32,fp16，设备实测 Q4）**：`#0 基线 = 39.029 ms` → `#1 arange = 0.660 ms`；两条路径数值一致（Q1/Q2 各 0 违反），故调整为纯性能裁决。

**R4 硬件亲和性评估**（逐候选对照检查清单；容量/带宽淘汰引用 `constants.md` 条目）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **#1 结构化分离 + D 共享（采纳）** | Vector（`vcast/vsub/vabs/vmul`；`vabs` fp16/fp32 ✓、int16 ✗ → 距离在 fp32 域；`arange` int16/fp16/fp32 ✓） | 小（实测峰值 12 B/elem；`bi·bj=8192` → 96 KB，§4.5）✓ | `bi·bj` 整除 32B ✓；v-op 作用于整 buffer | S、H 编译期常量 ✓ | 单步；跨 tile 重叠由 auto-multi-buffer 处理（默认 ON） | ✅ **采纳** |
| #0 基线（flat div/mod 标量 `T.Parallel`） | Vector 标量域（int64 `//`,`%` 可用；`test_floordiv_floormod_parallel.py` + 设备 Q1 实证） | 1-D `block` buffer（小） | ✓ | ✓（需 block≥ceil(N/65535) 避 coreDim 上限） | 逐元素标量串行，无向量吞吐 | ⚠️ **可实现但慢 59×**（Q1/Q4）→ 不采纳 |
| #2 不共享 D | Vector | 同 | ✓ | ✓ | 同 | ⚠️ 可行但指令 `×H`（劣于 #1） |
| #2b div/mod 向量化 | Vector（int64 vfloordiv/vfloormod） | int64 中间 8B/elem（UB ×2） | ✓ | ✓ | 需可靠向量 exp2（`vexp2` ✗，CG-2026-0018） | ⚠️ 复杂且无增益 |
| #3 max−min | Vector（`vmax/vmin/vsub`） | 同 | ✓ | ✓ | 同 | ❌ op 更多（3 vs 2）无增益 |
| #5 向量 vexp2 slope | Vector（`vexp2`） | 同 | ✓ | ✓ | 同 | ❌ **设备实测 inf**（CG-2026-0018） |
| #6 fp16 ramp | Vector（`vcast` 有 i16→f32）；fp16 ramp S>2048 不精确 | 同 | ✓ | ✓ | 同 | ❌ **正确性否决**（S=2048/4096） |
| #7 距离 profile gather | Vector 核内 gather（`T.gather` 存在但无核内向量 gather 实测先例） | 需 profile 驻留 | 未知 | 需索引 | 同 | ❌ 负向：`docs/Tilelang.language/索引与元素操作/T.gather.md` 存在但**无核内向量 gather 先例/实测**（未文档化假设 + 估算依据：per-element 索引代价高）；且 §R1 #1 已足够 |
| #8 外积分解 | — | — | — | — | — | ❌ 数学不成立 |
| #9 源 `T.Parallel` 直译 | Vector 标量域（写 UB 可用） | 1-D `block` | ✓ | ✓ | 逐元素标量 | ❌ 形态 = #0，慢 59×（且含 `if_then_else` 崩溃形态） |

**v-prefix 元算 dtype 矩阵（决定距离域与舍回）**：`vabs` 支持 int32/int64/fp16/fp32（**不含 int16**）→ 距离在 **fp32 域**（`vcast` i16→f32，精确）；`vcast` 支持 `i16→f32`(rint)、`f32→f16/bf16`(rint)（`docs/.../T.vcast.md` §2.2.1）；`vmul` 标量广播 fp16/fp32 ✓、bf16 ✗（本设计 vmul 恒在 fp32 域，bf16 仅末端 vcast 舍回）。

**调研结论**：**选定「结构化分离（int16 arange ramp + vsub/vabs 生成距离）+ D 在 head 维共享 + 每 head 标量 `T.exp2` slope + fp32 域单次舍回」**（候选 #1）。

关键依据（**本版修正为性能裁决**）：① R3 全部可行候选访存同为 write-only 下界 `H·S²·elem_bytes`，差异在**指令数**——#1 把距离生成摊薄 `H×` 且以向量指令整 buffer 执行（H=32 时距离开销占比 <5%）；② **#0 基线可实现**，但逐元素标量，**目标设备实测 39.029 ms vs #1 0.660 ms（2k/32/fp16，快 59.1×，Q4）**，且需规避 `if_then_else` 崩溃（用 `T.max`）与 coreDim 上限（block≥2049）；③ 其余候选淘汰（#2 指令 ×H、#2b 复杂无增益、#3 op 更多、#5 设备 inf、#6 S>2048 不精确、#7 无核内 gather 先例、#8 数学不成立、#9 = #0）；④ 选定算法的全部 op（`arange/vcast/vsub/vabs/vmul` + 标量 `T.exp2`）均在**目标设备实测**为真实可用（§3.4 Q1–Q3）。

**调研范围**（「无更优替代」依据）：① 参考表 `tilelang-op-design/references/algorithm-candidates.md` 的 `ALG-elementwise` 条目（R1「公共子表达式消除 / cast 链合并」；R2「逐元素映射天然单遍」；R3「融合消除中间 GM 往返」）——本仓无 ALiBi/生成式专门条目；② 本仓 `examples/` 同类实现：无输入逐元素先例（`elementwise/*` 均有输入）；索引生成先例 `testing/npuir/creation_ops/test_arange.py`、`_gqa_prefill_fwd_kernel.py` L421–435、`_ssd_chunk_scan_fwd_kernel.py` L349–353；③ pattern-library：`kb_search "alibi generative bias arange"` / `"integer div mod floordiv vector"` 命中 `CG-2026-0016/0017/0018`、`PL-1.22-persistent-1d-grid-stride`、`PL-1.23-vop-chain-inplace-alias`、`CASE-deepseek-v4-highperf`（int16 索引矩阵 + `T.arange` strides）等；**KB/pattern-library 无 ALiBi/generative 专属条目**（设计期已 `kb_search` 确认）；④ **外部已知算法族**（模型知识 + PyTorch 生态）：ALiBi（Press et al. 2021）的 bias 是确定性解析式，无「在线/近似/变换域」替代空间；FFT/递推（`D` 可用差分递推）在 write-only 生成下无收益（写出占比不变）；⑤ 本地 `docs/Tilelang.language/` + `tilelang/language/customize_npuir.py` 导出清单 + `testing/npuir/` 在仓测试**三查**确认整数 div/mod 能力（§0.5 证据链），并确认 `T.Parallel` 写 UB 可用——**故不依赖任何「API 不可用」类负向断言**（唯一能力性否决是 `out_idx`+arange 组合崩溃，CG-2026-0016，有设备实证）。

**设计期估算（D-2 roofline，估算下界 = max(流量项, 发射项, 容量项)；纯 Vector 豁免容量项）**：

- **流量项** = `bytes ÷ 有效带宽`，`bytes = H·S²·elem_bytes`（manifest roofline 同口径，**write-only**）。引用 `CONST-mte2-degradation`（`constants.md`；HBM 混合读写全流量地板 ~1.26 TB/s）——**该常数绑定 Ascend910B2C，本设备 Ascend910_9362，且标记 stale，数字仅作量级估算（待重验）**。本设备实查 Vector 核 40（§5.5 ②）。
- **发射项** = `n_tiles × k × ~0.5µs/op ÷ core_num`（`CONST-vector-launch-overhead`，绑 910B2C 且 stale，**未文档化假设 + 估算依据**）；`k` = 每空间 tile 的向量 op 数 = `6 + 2H`。
- **容量项** = 0（纯 Vector，无跨引擎往返）。

以 `bi·bj = 8192`、物理 Vector 核 = 40 估算：

| workload | dtype | H | S | n_tiles | bytes | 流量项 | 发射项 (k=6+2H) | **估算下界 max** | **实测 (rev1)** |
|----------|---|------|---|---------|-------|--------|------------------|------------------|------------------|
| test `(128,8)` | fp16 | 8 | 128 | 2 | 0.26 MB | <1 µs | ≈ 0.55 µs | **≈ 0.6 µs** | — |
| llama-prefill-2k | fp16 | 32 | 2048 | 512 | 268 MB | ≈ 213 µs | ≈ 448 µs | **≈ 448 µs** | **660 µs**（Q4） |
| llama-prefill-4k | fp16 | 32 | 4096 | 2048 | 1.07 GB | ≈ 850 µs | ≈ 1792 µs | **≈ 1792 µs** | — |

**设计期估算（D-2 roofline）**：代表性 `llama-prefill-2k fp16` 估算下界 ≈ **448 µs**；**实测 660 µs（Q4，bi=4/bj=2048）**，为估算的 ~1.47×，落在「发射项主导 + 0.5µs/op 常数 stale」的不确定区间内。全部 workload 由**发射项主导**（本算子 write-only、`bytes` 为下界且被摊薄，指令数成为第一驱动）。`bi·bj` 是摊薄发射的第一杠杆（发射 ∝ 1/(bi·bj)；实测 UB 天花板 = 16384 elems，§4.5）。

> **估算不确定性声明（未文档化假设 + 估算依据）**：① `CONST-vector-launch-overhead` 的 0.5µs/op 绑定 910B2C 且 stale；实测 660µs 较估算 448µs 高 ~47%，说明该常数偏乐观（**实测反解 ≈0.74µs/op**）——Stage 4 以 msprof 标定后回填；② `CONST-mte2-degradation` 的 1.26 TB/s 为**读写混合**地板，本算子 write-only，实际写出带宽可能更高，故流量项上偏；③ UB 实测天花板 16384 elems（§4.5），`bi·bj` 取 8192 留 2× 余量；④ Stage 4 baseline 采集后按 `hardware-cost-model.md` §3 对账并回填常数。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定算法（结构化分离 + D 共享 + 标量 exp2 + fp32 域）。以下各项完成「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；采纳项的等价性经 `verify_equiv.py` 机器验证（结果表见本节末）。

| # | 优化项 | 原式（基线） | 优化后公式（采纳） | 等价性论证 | 收益量化 |
|---|--------|------|-----------|-----------|---------|
| 1 | **CSE / D 在 head 维共享**（采纳，本算子核心） | 每 (h,i,j) 现算 `\|i-j\|`（`H` 次重复） | 每空间 tile 算一次 `D=\|i-j\|`，`h` 循环内复用 | **数学恒等**：`\|i-j\|` 不含 `h`；`-s_h·D` 与逐元素 `-s_h·\|i-j\|` 同值。`verify_equiv.py` #1（三 dtype `max_abs=0.0`，`violations=0`） | 距离生成指令数 `×H → ×1`（H=32 时距离开销占比 <5%）；D 驻留 UB 跨 head 复用 |
| 2 | **负号折入 slope 标量**（采纳） | `out = -(\text{slope}\cdot D)` | `s=-slope`；`out = s\cdot D` | **数学恒等 + 逐位一致**（`(-a)*b == -(a*b)`，IEEE 乘对符号精确）。`verify_equiv.py` #3（三 dtype `max_abs=0.0`） | 免一次 per-head 取负 op |
| 3 | **int16 ramp 距离域**（采纳） | 距离在 fp32（源 `cast(row-col,f32)`） | int16 arange ramp → `vcast` i16→f32 → vsub/vabs | **值精确等价**：`\|i-j\|≤4095<32767`，int16 精确表示；i16→f32 无损。`verify_equiv.py` #2（`max_abs=0.0`）+ #8（S=4096，`max_abs=0.0`） | 省 UB 字节（int16 ramp 2B/elem vs fp32 4B）；距离仍 fp32 精确 |
| 4 | **slope 标量 `T.exp2`（每 head 一次）**（采纳） | 每元素 `T.exp2`（源） | `for h in serial(H): s_h=-T.exp2(-8(h+1)/H)` | **容差内等价**：设备 `T.exp2` vs `torch.pow(2,·)` fp32 `max_abs=2.98e-8`（rel ≤2.5e-7@H=32，Q3）；末端舍回 dtype 吸收。`verify_equiv.py` #4（CPU `exp2` vs `pow` 三 dtype `max_abs=0.0`） | 每元素 1 次 exp2 → 每 head 1 次（`H` 次，相对 `H·S²` 可忽略） |
| E1 | **host 编译期 slope 常量（拒绝）** | — | Python `-2**(-8(h+1)/H)` | 数学上等价，但 `for h in range(H)` 被降为 `T.serial`（`h` 为 Var）→ 列表索引/枚举报错（v0 实测） | 不可行 |
| E2 | **向量 `T.vexp2` slope（拒绝）** | — | `vexp2` 批量 | 数学上等价，但**设备实测 inf**（CG-2026-0018） | 不可用 |
| E3 | **fp16 ramp（拒绝）** | — | fp16 arange | **非等价**（S=2048/4096 时 `\|i-j\|>2048` 不可精确表示） | 正确性否决 |
| E4 | **`max-min` 距离（拒绝）** | `vsub+vabs` | `vsub(vmax,vmin)` | 数学恒等 | op 更多（3 vs 2），无增益 |

**优化结论**：采纳 4 项（#1 D 共享、#2 负号折叠、#3 int16 ramp、#4 每 head 标量 exp2）。**公式层面已无进一步计算量/访存量可优化空间**——本算子 write-only、访存固定在理论下界 `H·S²·elem_bytes`（无中间 GM 往返）；距离生成经 D 共享已摊薄 `H×`；slope 已由每元素降为每 head。**优化后公式**（供 §3.1 使用）：
- `D = vabs(vsub(vcast_{f32}(arange_{i16}[1,0]@i0), vcast_{f32}(arange_{i16}[0,1]@j0)))`（fp32，每空间 tile 一次）
- `float32`：`out[h] = vmul(D, s_h)`（s_h = 标量）
- `float16 / bfloat16`：`out[h] = vcast_{dtype}(vmul(D, s_h), rint)`

**等价性机器验证（D-1）**：脚本 `examples/alibi/_make_alibi_kernel/verify_equiv.py`（CPU；候选式 = 优化后公式，基线式 = 原式，fp64 参照；判定准则 = Stage-3 门禁同口径 `|cand-base| ≤ atol + rtol·|base|`，fp32 1e-5 / fp16 1e-2 / bf16 1.6e-2，违反率 0 即 PASS）。**执行结果表**（脚本实际输出，`python3 verify_equiv.py`，**ALL 46 ITEMS EQUIV_PASS**；覆盖 `S∈{128,256}`×`H∈{8,32}`×`{fp32,fp16,bf16}` + S=4096 int16-range + 角点对角/h 极值）：

| 采纳项 / 对照 | 候选式 | 基线式 | samples | 最大 abs 差 | 违反数 | 结论 |
|---------------|--------|--------|---------|------------|--------|------|
| #2 int16-ramp（fp32/fp16/bf16） | int16 ramp D | exact fp32 D | 16384–65536 | 0.000e+00 | 0 | **EQUIV_PASS** |
| #1 CSE / D-shared（fp32/fp16/bf16） | broadcast D→scale per head | per-head 重算 | 131072–2097152 | 0.000e+00 | 0 | **EQUIV_PASS** |
| #3 neg-fold（fp32/fp16/bf16） | `(-s)·D` | `-(s·D)` | 131072–2097152 | 0.000e+00 | 0 | **EQUIV_PASS** |
| #4 exp2 slope vs pow slope（fp32/fp16/bf16） | `exp2(e)` | `pow(2,e)` | 131072–2097152 | 0.000e+00 | 0 | **EQUIV_PASS** |
| #5 full candidate vs golden（fp32/fp16/bf16） | 优化后全式 | `test_alibi` golden | 131072–2097152 | 0.000e+00 | 0 | **EQUIV_PASS** |
| #6 baseline-divmod vs golden（fp32/fp16/bf16） | 源 flat div/mod 坐标 + fp64 pow | `test_alibi` golden | 131072–2097152 | 0.000e+00（fp32 S=256/H=32 为 1.526e-05 < 容差） | 0 | **EQUIV_PASS** |
| #7 int64 div/mod 坐标往返（fp32/fp16/bf16） | `h·S²+row·S+col == flat` | 精确相等 | 16384–2097152 | 0 | 0 | **EQUIV_PASS** |
| #8 int16-range S=4096（`\|i-j\|>2048`） | int16 ramp D | exact fp32 D | 16777216 | 0.000e+00 | 0 | **EQUIV_PASS** |
| #5 角点（对角 0 / h 极值） | 优化后全式 | golden | 131072 | 0.000e+00 | 0 | **EQUIV_PASS** |

> 说明：① CPU 上 `torch.exp2` 与 `torch.pow(2,·)` 对 ALiBi 指数逐位一致（`max_abs=0.0`），#4 无违反；设备 `T.exp2` 与 `pow` 有 1 ulp 差（§3.4 Q3），但末端 dtype 舍回后**端到端 0 违反**（§3.4 Q1/Q2）。② 本算子无张量输入，故「角点值」为结构性角点（对角 `i=j` 精确 0、h=0/H-1 极值 slope、`i=0`/`j=S-1` 极值距离），已覆盖；③ #6（基线 div/mod 坐标）与 #5（arange 候选）**同 golden 同 0 违反**——机器验证坐实「坐标方案之争是性能裁决而非数值/正确性裁决」；④ 全部违反数为 0（46/46 `EQUIV_PASS`）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 行/列坐标生成 | 源标量 `flat//S²`、`%S²`、`//S`、`%S`（或 `T.Parallel` int64 标量） | `T.npuir_arange(R,[1,0],i0)` + `T.npuir_arange(C,[0,1],j0)`（佐证：`test_arange.py`；`_gqa_prefill_fwd_kernel.py` L421–435；`_ssd_chunk_scan_fwd_kernel.py` L349–353；设备 Q2 实测） | ✅ | — |
| 2 | 距离 `\|i-j\|` | 源标量 `row-col` + `if_then_else` | `T.vcast(i16→f32,rint)`×2 + `T.vsub` + `T.vabs`（佐证：`docs/.../数据类型转换操作/T.vcast.md` §2.2.1、`数学操作/T.vsub.md` §2.3、`数学操作/T.vabs.md` §2.4；设备 Q2） | ✅ | — |
| 3 | 乘积 `slope_h·\|i-j\|` | 源标量 `-slope*abs_dist` | `T.vmul(D, s_h, O)`（整 buffer；佐证 `docs/.../数学操作/T.vmul.md` §2.3 标量广播；设备 Q2） | ✅ | — |
| 4 | 末端舍回 | 源标量 `T.Cast(dtype,·)` | `T.vcast(O, O16, round_mode="rint")`（佐证 `docs/.../数据类型转换操作/T.vcast.md` §2.2.1；设备 Q2） | ✅ | — |
| 5 | **每 head slope 标量 `T.exp2`** | 源标量 `T.exp2`（每元素） | 无（保持标量；**不能**用向量 `T.vexp2`——设备 inf，CG-2026-0018） | ❌ | **API 不可靠**：向量 `vexp2` 设备实测 inf；标量是 `T.vmul` 的合法标量操作数（源原式）。代价：每 head 1 次（共 `H` 次），相对 `H·S²` 输出可忽略（§1.6.1 #4） |
| 6 | head 循环 `for h in T.serial(H)` | — | 无（tile 级调度循环，非逐元素） | ❌ | **tile 级调度循环**：每空间 tile 迭代 `H` 次，每次 2 个整-buffer 向量 op；非逐元素热点；元素级并行由向量指令整 buffer 完成。此循环承载 D 共享（§1.6.1 #1） |
| 7 | 核内网格跨步任务循环 | — | 无（核内调度循环） | ❌ | **tile/block 级任务调度循环**：`T.serial(ceildiv(n_tiles,40))`（S=4096@8192 = 52 次/核），非逐元素；先例 `PL-1.22` |
| 8 | tile 起始索引 / 任务映射 | `task=t*40+cid`；`i0=(task//nb)*bi`；`j0=(task%nb)*bj` | 无 | ❌ | **block 级索引/任务映射**，每 tile 少量标量，非逐元素热点；无逐元素等价 API |
| 9 | 尾块边界（S 非整除分块） | — | 无 | ❌ | **依赖 S 的分块整除性**：设计取 `bi\|S`、`bj\|S`（S 编译期，§5.4）；不可整除时保留 guard/回退（见 §5.4），纯静态化对任意 S 不成立 |

**向量化结论**：逐元素计算点（#1–#4）**已全部向量化**（`arange/vcast/vsub/vabs/vmul`，均有 docs 佐证 + 设备 Q1/Q2 实测）；保留 5 处标量/循环（#5 每 head 标量 exp2、#6 head 调度循环、#7 核内网格跨步、#8 block 索引、#9 动态边界），理由类别：**API 不可靠（vexp2）+ tile 级调度循环 + block 索引 + 动态边界**，逐项见上表。与 §6 循环结构一致——kernel 内**无逐元素标量循环**。
>
> **对照说明**：源 `T.Parallel` 逐元素结构**本身可用**（写 UB，Q1 实证），其逐元素计算点亦可被向量化（如用 `T.vfloordiv` 走 int64 向量域）——但那样需 flat ramp（即仍需 arange）+ int64 中间（UB ×2）+ 可靠的向量 exp2（不可得）。**故「向量化替代」在本算子的最优解就是 #1 的结构化分离 arange 链**，而非把 div/mod 逐元素向量化。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> 本算子输出为 3-D `(H,S,S)`。I/O layout 是契约（`(H,S,S)` row-major）；核内布局/向量化轴是**设计变量**。头维 `H` 由 tile 内 `T.serial` 循环承载（§1.6.1 #1 D 共享），故核内 tile 是 `(bi,bj)` 的 **(i,j) 二维平面**；向量化轴 = 尾轴 `bj`（连续）。候选枚举：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | **I/O 原生 (bi,bj)，bj\|S**（含 bj=S 全宽） | 尾轴 `bj`（stride=1 连续） | 无（I/O 布局 = 核内布局） | 基线；bj=S 时写出完全连续 | ✅ **采纳** |
| 2 | 核内转置为 (bj,bi)（`T.transpose` 两轴交换） | `bi` | 需 `T.transpose`（核内 UB 级、µs 级；`docs/Tilelang.language/创建操作/T.transpose.md` §2.3 仅支持两轴交换，>2 轴可分解为相邻交换链） | **无收益**（距离矩阵对称、`bj` 已可作向量轴；转置不减少 op 数/访存，纯增 VTransposeOp 向量管线开销） | ❌ 负向：**无收益**（`T.transpose` 的 dtype 支持见 `docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1：int8/int16/int32/uint64/int64/fp16/fp32/**bf16 均 √**，仅 uint8/uint16/uint32/bool ×，故**不以 dtype 为弃选理由**） |
| 3 | 输出侧**一次** host permute | — | 一次 host permute（百 µs 级，`PL-1.3-host-permute`） | repack 代价 > 算子本体流量，净亏；且输出为契约 layout 不得改 | ❌ 负向：host 契约禁改、代价净亏（引 `PL-1.3` 实测 106–147µs）。**本算子为零张量输入生成式，输入侧无 permute，仅可能对输出做一次 permute** |
| 4 | 3-D tile (bh,bi,bj)（head 入 tile） | `bj` | 无 | 可省 head 循环，但需 3-D 广播（`vmul` 3-D 广播**设备实测编译失败**：`value must be a tir.PrimExpr or T.BufferLoad`；`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 仅列 2-D 广播规则） | ❌ 负向：3-D 广播不支持（设备实测 + 文档仅 2-D 规则） |

> **弃选前提取证说明（rev2 增补）**：本节弃选候选的 API dtype/代价前提**已逐条打开对应文档的 dtype 矩阵核对**——#2 `T.transpose`（`docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1：bf16 及 int8/int16/int32/uint64/int64/fp16/fp32 均 √）、#4 `T.vmul` 广播（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 仅列 2-D 规则）；#3 host permute 代价引 pattern-library `PL-1.3-host-permute`（实测量级 106–147µs）。凡文档未覆盖者显式标注「未文档化假设 + 估算依据」（如 §1.6.0 R4 #7 gather）。

**轴质量评分（采纳方案 #1）**：
- **整除性**：`bj` 取 S 的因子（如 128/256/512/S），`bj·elem_bytes` 为 32B 整数倍（bj=128 fp16=256B、fp32=512B）✓；向量宽 fp16/bf16 ×8、fp32 ×4，`bj%8==0`（bj≥128）→ 尾 lane 浪费 0%。
- **尾 lane 浪费率**：0%（bj 为 8 的整数倍）。
- **累加链形态**：无串行累加维（纯逐元素），无跨步系数阻碍向量化。
- **repack 代价**：0（I/O 布局即核内布局）。
- **UB 容量影响**：tile `bi·bj` 决定工作 buffer 预算（§4.5；实测峰值 12 B/elem，天花板 16384 elems）。
- **写出连续性**：`bj=S` 时 tile 覆盖整行 → `T.copy` 为 `bi·S` 连续元素（MTE3 单段连续流）；`bj<S` 时为 `bi` 条 `bj·elem_bytes` 连续段（bj≥128 → 段 ≥256B）。**优先取 `bj=S`**（在 UB 允许时）。〔**未文档化假设 + 估算依据**：连续写出优于跨步写出是保守结构判断；`PL-1.14-mte3-strided-ws` 记跨步行写出有税但其**本身在 stale 前置重验清单**（不得作决策依据），故本条仅以「连续流段更长」的常识判断为依据，Stage 4 以实测裁决。〕

**布局决策结论**：选定方案 #1：核内布局 = `(bi,bj)` 二维连续 tile（I/O 布局即核内布局，无 repack），向量化轴 = 尾轴 `bj`，repack 路径 = 无；**倾向 `bj=S`（写出完全连续）**并由 §5.2 依据 UB 预算取 `bi=max(2, 8192//bj)`（`bi≥2` 为设备实测约束，CG-2026-0017）。弃选 #2/#3/#4 的量化理由见候选矩阵。该结论同步落入 §3.3 伪代码与 §6 循环结构（tile buffer 形状 `(bi,bj)`，逐元素向量 op 作用于该 buffer 的整体）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer

### 2.2 选型理由

| 算子特征 | 分析 | 结论 |
|----------|------|------|
| 计算类型 | 纯 Vector（arange ramp + 逐元素 v-链），无 matmul | 不需 Cube/L0C/L1 |
| 是否含归约 | 无归约 | 不需 Expert 级 buffer 控制 |
| 是否需流水线 | 单数据通路 UB→GM（write-only），无核间协作 | 不需 CV 融合 |
| 索引/向量链 API | `T.npuir_arange` + v-prefix（`T.vcast/vsub/vabs/vmul`） | Developer 模式（`arange` 与 `T.Parallel` 均设备实测于 Developer 通路可用，见 §3.4） |
| 同步需求 | UB 内顺序计算，数据依赖线性 | Developer 自动同步即可 |

决策树路径：`纯 element-wise（生成式） → 单步逐元素 + 核内索引生成 → 无需精细 buffer 控制 → Developer 模式`。与 `op_requirements.programming_mode=developer` 一致；参考 `examples/elementwise/vec_add_2d.py`（Developer + `T.alloc_ub` + v-prefix + `T.serial` 网格跨步）与 `testing/npuir/creation_ops/test_arange.py`（Developer `arange` 用例）、`testing/npuir/parallel_ops/`（Developer `T.Parallel` 用例）。

> **模式敏感性提示**：`T.vexp2` 在 Developer/Expert 下实现不同，且本会话 Developer 模式实测 inf（CG-2026-0018）——本设计**不使用**向量 `vexp2`；仅用设备实测过的标量 `T.exp2`（§3.4 Q3）。
>
> **模式与 `T.Parallel`**：`T.Parallel` 写 UB（Developer）**可用**（Q1 实证）。本设计不用 `T.Parallel` 承载逐元素计算，**不是**因其不可用，而是因为向量 arange 链的发射成本低 59×（§1.6.0 R3/Q4）；`T.Parallel` 仅作为「可实现但劣」的基线候选记录在 §1.6.0。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub((bi,bj), dtype)` 显式 UB 分配（v-prefix/arange 要求 buffer 在 UB）；全部在 `T.Kernel` 顶层无条件分配 |
| 计算方式 | `T.npuir_arange` + v-prefix 向量链（逐元素计算全部向量化）。**注**：`T.Parallel` 写 UB（Developer）本身**可用**（`T.Parallel.md` §2.3–2.4 + `testing/npuir/parallel_ops/` + 设备 Q1），但逐元素标量形态发射成本高 59×（Q4），故不采用 |
| 同步 | Developer 自动同步（线性数据依赖，编译器自动插 barrier） |
| 分核 | 一维 `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步（§5.5） |
| 调用约定 | **显式输出参数**（无 `out_idx`）——实测强制（§0.6 R5） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（结构化分离 + D 共享 + 负号折叠 + int16 ramp + 每 head 标量 exp2）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `i0=(task//nb)·bi`，`j0=(task%nb)·bj` | 空间 tile 定位（block 级标量） |
| 2 | `R[r,c]=i0+r`，`C[r,c]=j0+c` | int16 行/列坐标 ramp |
| 3 | `Rf = cast_{f32}(R)`，`Cf = cast_{f32}(C)` | int16→fp32（精确） |
| 4 | `D = \|Rf − Cf\|` | 距离矩阵（fp32，整数精确） |
| 5 | `e = −8·(h+1)/H`；`s_h = −2^{e}` | 每 head slope 标量（`h` 为 `T.serial` Var） |
| 6 | `O = s_h · D` | 逐 head 缩放（D 复用） |
| 7 | `O16 = cast_{dtype}(O)`（fp16/bf16） | 单次舍回（`rint`） |
| 8 | `out[h, i0, j0] = O16`（或 `O`） | 写出（write-only） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 坐标 ramp | `R[r,c]=i0+r` | `T.npuir_arange` | `T.npuir_arange(R, [1, 0], i0)`（strides=每维系数，offset） | Developer |
| 坐标 ramp | `C[r,c]=j0+c` | `T.npuir_arange` | `T.npuir_arange(C, [0, 1], j0)` | Developer |
| 升 fp32 | `Rf=cast(R)` | `T.vcast` | `T.vcast(R, Rf, round_mode="rint")` | Developer |
| 距离 | `D=\|Rf−Cf\|` | `T.vsub` + `T.vabs` | `T.vsub(Rf, Cf, Rf)`；`T.vabs(Rf, Rf)` | Developer |
| slope | `s_h=−exp2(e)` | `T.exp2`（标量）+ Python 取负 | `e = T.cast(-8.0,f32)*T.cast(h+1,f32)/T.cast(H,f32)`；`slp = -T.exp2(e)` | Developer |
| 缩放 | `O=s_h·D` | `T.vmul` | `T.vmul(Rf, slp, O)`（标量广播） | Developer |
| 舍回 | `O16=cast(O)` | `T.vcast` | `T.vcast(O, O16, round_mode="rint")` | Developer |
| 写出 | `out[h,i0,j0]=O16` | `T.copy` | `T.copy(O16, out[h, i0, j0])` | Developer |
| 任务映射 | grid-stride | `T.Kernel` + `T.serial` | `T.Kernel(40, is_npu=True)` + `for t in T.serial(ceildiv(n_tiles,40))` | Developer |

### 3.3 计算伪代码

> 迁移规则：`@T.prim_func` 输出参数名 `out`（源一致）；**不使用 `out_idx`**（§0.6 R5）；`T.Tensor` 的 dtype 以**第二位置参数**传入（不用 `dtype=`）；v-op 只作用于整 UB buffer（动态 slice 只包在 `T.copy` 里）；buffer 在 `T.Kernel` 顶层无条件分配（避免 `TRAP-conditional-alloc`）；`T.npuir_arange` 的 tile 首维必须 ≥2（CG-2026-0017）。

```python
import tilelang
import tilelang.language as T

# ALiBi: bias[h,i,j] = -slope_h * |i-j|, slope_h = 2^(-8(h+1)/H)
# 生成式：main 只有输出参数；不使用 out_idx（实测 out_idx + arange 设备崩溃，CG-2026-0016）
@tilelang.jit(target="npuir")
def _make_alibi_kernel(seq_len, num_heads, dtype, bi=None, bj=None, budget=8192):
    S, H = seq_len, num_heads
    bj = bj or S                     # 优先全宽（写出连续）
    # bi 默认按 UB 预算推导（§5.2；budget=8192 留 2× 余量，实测天花板 16384）
    bi = bi if bi is not None else max(2, budget // bj)
    bi = max(2, bi)                  # arange tile 首维 ≥2（CG-2026-0017）
    nb = S // bj                     # 列块数（bi|S、bj|S）
    n_tiles = (S // bi) * nb         # 逻辑 tile 数（编译期静态）
    f32 = "float32"
    need_cast = dtype != f32
    core_num = 40                    # 物理 Vector 核（实查 20 aicore × 2）

    @T.prim_func
    def main(out: T.Tensor((H, S, S), dtype)):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            R = T.alloc_ub((bi, bj), "int16")     # 行坐标 ramp
            C = T.alloc_ub((bi, bj), "int16")     # 列坐标 ramp
            Rf = T.alloc_ub((bi, bj), f32)        # 行坐标 fp32 / 距离 D
            Cf = T.alloc_ub((bi, bj), f32)        # 列坐标 fp32
            O = T.alloc_ub((bi, bj), f32)         # 每 head 缩放结果
            O16 = T.alloc_ub((bi, bj), dtype)     # 舍回缓冲（fp16/bf16）

            for t in T.serial(T.ceildiv(n_tiles, core_num)):   # 静态循环上界
                task = t * core_num + cid
                if task < n_tiles:                             # 尾任务守卫
                    i0 = (task // nb) * bi
                    j0 = (task % nb) * bj
                    T.npuir_arange(R, [1, 0], i0)              # R[r,c] = i0 + r
                    T.npuir_arange(C, [0, 1], j0)              # C[r,c] = j0 + c
                    T.vcast(R, Rf, round_mode="rint")          # int16 -> fp32 (无损)
                    T.vcast(C, Cf, round_mode="rint")
                    T.vsub(Rf, Cf, Rf)                         # Rf = i - j
                    T.vabs(Rf, Rf)                             # Rf = |i-j| = D  (head 无关)
                    for h in T.serial(H):
                        e = T.cast(-8.0, f32) * T.cast(h + 1, f32) / T.cast(H, f32)
                        slp = -T.exp2(e)                       # s_h = -slope_h (标量)
                        T.vmul(Rf, slp, O)                     # O = s_h * D
                        if need_cast:                          # 编译期分支
                            T.vcast(O, O16, round_mode="rint") # 单次舍回
                            T.copy(O16, out[h, i0, j0])
                        else:
                            T.copy(O, out[h, i0, j0])
    return main
```

> **伪代码说明**：① `if need_cast:` 为 Python 编译期条件（JIT 参数编译期确定），不产生运行时分支；buffer 顶层无条件分配；② 距离链（步骤 2–4）每空间 tile 一次，`h` 循环（`T.serial(H)`）内只有 `vmul`+`vcast`——**D 共享**（§1.6.1 #1）；③ `slp=-T.exp2(e)` 生成标量 `Var`（LetStmt），作为 `T.vmul` 的合法标量操作数（`AscendBinaryOp` 接受 `tir.expr.Var`）；④ `R`/`C` 为 int16（2B/elem）以省 UB 并覆盖 `|i-j|≤4095`；⑤ v-op 全部作用于整 buffer；⑥ **`bi` 默认值**：`bi = max(2, budget//bj)`（`budget` 默认 8192，§5.2），与 §5.2 推荐取值一致；调用方可显式传 `bi`。
>
> **接口适配说明**：集成 wrapper `tileops/kernels/elementwise/alibi/alibi.py` 约定 NPU 工厂签名 `_make_alibi_kernel(seq_len, num_heads, dtype)`，返回可调用对象。因改用**显式输出参数**（§0.6 R5），wrapper 的 `forward()` 需由「`self._compiled_fn()`」改为「`out = torch.empty(H,S,S, device=..., dtype=...); self._compiled_fn(out); return out`」（Stage 3 据此实现；输出 shape/dtype 不变）。`_DEFAULT_BLOCK_SIZE`（K11 折叠）在 NPU 侧映射为 `(bi,bj)` tile 选择（§5.2）。
>
> **op 计数 = 每空间 tile `6 + 2H`**（距离链 6：arange×2+vcast×2+vsub+vabs；head 循环 `2H`：vmul+vcast）——H=32 时 70。

### 3.4 API 可行性确认

> **纪律（前车之鉴 TRAP-verf-maclaurin-truncation / CG-2026-0015）**：NPU 文档声称的元算语义/精度**必须对目标设备实测**；**负向断言必须三查**（`tilelang/language/` 导出 + `testing/npuir/*/` 在仓测试 + `docs/` 官方示例），禁止凭记忆/GPU 先验下结论（本版核心修正的根因）。本节全部关键 API 附目标设备（Ascend910_9362，toolchain tilelang 0.1.2+87a3307b2a83d60ee8af09086ff280a8bb6c26c5）实测（探针脚本 `history_version/design_probe_alibi_v1.py` 的 Q1–Q4，2026-10-13 本会话实跑；v0 的 `design_probe_alibi.py` 结论已按新工具链复核）。**其中 `T.Parallel` 能力 bisect 为同会话实测但未落盘（见下），UB 天花板已落盘复现于 `history_version/design_probe_alibi_ub_ceiling.py`。**

| API | 来源确认（本地） | 设备实测（目标设备） | 状态 |
|-----|-----------------|---------------------|------|
| `T.npuir_arange(dst, strides, offset)`（2-D tile，int16） | `docs/Tilelang.language/创建操作/T.arange.md` §2.2.1（int16/int32/int64/fp16/fp32、`hivm::VArangeOp`）；`test_arange.py`；`_gqa_prefill_fwd_kernel.py` L421–435；`_ssd_chunk_scan_fwd_kernel.py` L349–353 | `(32,64)/(64,128)` tile 端到端 0 违反（Q2） | ✅ 真实可用（**tile 首维 ≥2**；`(1,N)` 尾元素垃圾，CG-2026-0017） |
| `T.vcast(src,dst,rint)` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（`i16→f32` rint、`i16→f16`；`f32→f16/bf16` rint） | 端到端 Q2 实跑（i16→f32 无损、f32→f16/bf16 单次舍回） | ✅ 真实可用 |
| `T.vsub(src0,src1,dst)` | `docs/Tilelang.language/数学操作/T.vsub.md` §2.3 | 距离链实跑（Q2） | ✅ 真实可用 |
| `T.vabs(src,dst)` | `docs/Tilelang.language/数学操作/T.vabs.md` §2.2.1（fp16/fp32 ✓，int16 ✗） | 距离链实跑（fp32 域，Q2） | ✅ 真实可用 |
| `T.vmul(src1, scalar, dst)`（标量广播） | `docs/Tilelang.language/数学操作/T.vmul.md` §2.3（标量 src2；dst 可原地） | `T.vmul(D, slp, O)` 实跑（Q2）；标量为标量 Var | ✅ 真实可用 |
| 标量 `T.exp2(e)`（源 slope 实现式） | `tilelang/language/tir/op.py:1837`；`AscendBinaryOp` 接受标量 Var/const | `max_abs=2.98e-8` vs `torch.pow(2,·)`（rel ≤2.5e-7@H=32，Q3）；端到端 Q1/Q2 全 PASS | ✅ 真实可用 |
| `T.Kernel(n, is_npu=True)` + `T.serial` | `examples/elementwise/vec_add_2d.py` L28–L42 | Q1/Q2 实跑 | ✅ 真实可用 |
| `T.copy(UB→GM 区域)` | `test_arange.py`；`vec_add_1d.py` | Q1/Q2 实跑（含 `out[h,i0,j0]` 区域） | ✅ 真实可用 |
| **`T.Parallel(block)` 写 UB（Developer，含 `alloc_shared`）** | `docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.3（"循环体内的变量必须是 alloc 申请出的 UB 上的 Buffer"）、§2.4（官方 `@tilelang.jit(target="npuir")`+`T.Parallel`+`alloc_shared` 示例）；`testing/npuir/parallel_ops/`（12 测试） | **Q1 baseline kernel 全 PASS**；`test_floordiv_floormod_parallel.py` 设备重跑 **12/12 passed** | ✅ **真实可用**（v0 断言「不可用」被推翻） |
| **向量/标量整数 div/mod** | `docs/.../T.vdiv.md` §2.2.1（int64 √，int16/int32 ×）；`customize_npuir.py:158 npuir_div`→`vdiv`、`:425 npuir_floordiv`→`vfloordiv`；`test_floordiv_floormod_parallel.py`（Developer int64 `//`,`%`） | Q1（int64 `//`,`%` 于 `T.Parallel`）全 PASS | ✅ **可用**：int64 有（标量 `T.Parallel` 与向量 `vdiv/vfloordiv`）；**int16/int32 无独立 v-op**（v0「NPU 无向量整数 div/mod」为过度断言） |
| **`T.Parallel` 内 `T.if_then_else` / `T.fabs`（循环变量条件/局部变量）** | `docs/.../T.Parallel.md` §2.2.1「当前不支持直接使用循环变量的条件语句，例如 `T.if_then_else(i>j, xxx, xxx)`」 | **编译期 segfault**（V4/V5/V7）；`T.max(d,-d)`（V6/V6b）与 inline `T.fabs`（V8/V9）可用 | ⚠️ 条件式受限（**有文档条款**）：abs 须用 `T.max(d,-d)` 等价式，不得用 `if_then_else` |
| **`@tilelang.jit(out_idx=[...])` + `T.npuir_arange`** | — | **运行期设备崩溃**（`MTE DDR out of range`，retCode 507057；同 kernel 体显式输出 bit-exact） | ❌ **不可用**（CG-2026-0016；改用显式输出参数）——**唯一能力性否决** |
| **向量 `T.vexp2`（Developer）** | `docs/Tilelang.language/数学操作/T.vexp2.md` §1（自述 `exp(A*ln2)`） | (8,64) fp32 tile 部分元素 `inf`（输入域有限 [-2,13.75]） | ❌ **不可靠**（CG-2026-0018；不用） |
| **3-D v-op 广播 `[bh,1,1]·[1,bi,bj]`** | `docs/.../数学操作/T.vmul.md` §2.2.2 仅列 2-D 规则 | 编译失败（`value must be a Tir.PrimExpr or T.BufferLoad`） | ❌ 不支持（故 head 由 `T.serial` 承载，§1.6.3 #4） |

**端到端设备实测（`design_probe_alibi_v1.py` Q1/Q2，判定 = `atol+rtol·|ref|`，全部 0 违反 PASS）**：

| 用例 | 来源 | dtype | 坐标方案 | 结果 |
|------|------|-------|---------|------|
| `S=128,H=8` | `test_alibi` | fp16 | **基线 flat div/mod（Q1）** | **exact=True**，max_abs=0.0，违反 0/131072 |
| `S=128,H=8` | `test_alibi` | fp32 | 基线（Q1） | max_abs=3.815e-6（<1e-5），违反 0 |
| `S=2048,H=32` | manifest smoke | fp16 | 基线（Q1, blk=4096） | max_abs=3.906e-3，违反 0/134217728 |
| `S=128,H=8` | `test_alibi` | fp16 | **arange（Q2）** | **exact=True**，违反 0 |
| `S=128,H=8` | `test_alibi` | fp32 | arange（Q2） | max_abs=3.815e-6，违反 0 |
| `S=2048,H=32` | manifest smoke | fp16 | arange（Q2, bi=64,bj=128） | max_abs=3.906e-3，违反 0 |
| `S=2048,H=32` | manifest smoke | bf16 | arange（Q2） | max_abs=4.0（1 ulp），违反 0 |

> **Q3（slope 语义）**：标量 `T.exp2(-8(h+1)/H)` vs `torch.pow(2,·)` fp32：H=8 `max_abs=2.98e-8` rel 5.96e-8；H=32 `max_abs=2.98e-8` rel 2.53e-7 → **约为 `pow` 的 1 ulp**。
>
> **Q4（性能，S=2048,H=32,fp16）**：基线 flat div/mod `T.Parallel` = **39.029 ms**；arange + D 共享 = **0.660 ms**；**ratio = 59.1×**。此为坐标方案裁决的核心设备证据。
>
> **`T.Parallel` 能力 bisect（**会话内实测，未落盘**；同会话同工具链）**：V0 `o[i]=cast(i//4)` ✓ / V1 加 block 偏移 ✓ / V2 加 `%` ✓ / V3 加标量 `exp2` ✓ / V4 `if_then_else(dist>0,dist,-dist)` ✗ segfault / V5 `fabs(local)` ✗ segfault / V6 `max(d,-d)` inline ✓ / V6b `max(d,-d)` 局部变量 ✓ / V7 `if_then_else`（flat 派生）✗ segfault / V8 inline `fabs` ✓ / V9 inline `fabs` ✓——**div/mod + UB 可用；条件式（`if_then_else`）受限（有文档条款）；`T.max`/inline `fabs` 为可用 abs 等价式**。〔**未落盘说明（rev2）**：该 bisect 未落盘为独立脚本（其 V8/V9 变体的原始写法未留存，重建复跑不能保证逐变体一致），故此处显式标注为会话内一次性实测；其结论只用于支撑「`T.Parallel` 写 UB 可用 + 条件式受限」，不构成 §4/§5 的操作界。〕
>
> **UB 天花板探针（设备实测，已落盘复现 `history_version/design_probe_alibi_ub_ceiling.py`，§4.5）**：`bi·bj ∈ {8192,12288,16384}`（fp16）与 `{8192,16384}`（fp32）compile+run OK；`24576` 报 `ub overflow, requires 2359296 bits while 1572864`（=192KB）。故实测峰值 ≈ **12 B/elem**（liveness 复用后的距离链峰值），**天花板 = 16384 elems**（v0 的「192KB/1.12×、16384 超限」为过度保守的解析估计，实测 16384 恰好贴合 192KB）。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(40, is_npu=True)`；输出 `(H,S,S)` 由 tile 区域定位 |
| GPU 专用 API（`T.Parallel(threads,npt)` 的 `threads=` 语义、`out_idx`） | Yes | `threads=` 在 npuir 无效果（TRAP-threads-kwarg-noop）；**`T.Parallel` 本身可用**（§3.4），本设计改用 arange + v-prefix（性能）；**显式输出参数**（§0.6 R5） |
| GEMM 非整除 | No | 非 GEMM；分块整除性见 §5.4 |
| L0C 溢出 | No | 无 Cube 计算 |
| 物理核数限制 | Yes | 逻辑 tile 数（512–2048）远超物理核 40 → 核内 `T.serial` 网格跨步（§5.5） |
| **整数 div/mod（int64 可用 / int16·int32 无独立 v-op）** | Yes | **有能力**（§3.4）；不采用逐元素标量 div/mod（慢 59×）→ arange ramp + vsub/vabs（§0.6 R1） |
| **`T.Parallel` 内条件式（`if_then_else` 循环变量）崩溃** | 仅基线路径 | `T.Parallel.md` §2.2.1 文档限制；若走 `T.Parallel` 须用 `T.max(d,-d)` 替代 abs（本设计走 arange，不涉及） |
| **`out_idx` + arange 崩溃** | Yes | 显式输出参数（§0.6 R5；CG-2026-0016） |
| **`arange` tile 首维 ≥2** | Yes | `bi≥2`（§5.2；CG-2026-0017） |
| **向量 `vexp2` 不可靠** | Yes | 标量 `T.exp2`（§0.6 R3；CG-2026-0018） |
| **3-D v-op 广播不支持** | Yes | head 由 `T.serial` 循环承载（§1.6.3 #4） |
| `vabs` 不支持 int16 | Yes | 距离在 fp32 域（`vcast` 升 f32 后 `vabs`） |
| 条件 `T.alloc` 非法 | Yes | buffer 全部顶层无条件分配（§3.3） |
| coreDim ≤ 65535 | 仅基线路径 | 基线 1-D flat 需 `block ≥ ceil(N_total/65535)`（2k/32 → ≥2049；block=2048 实测 `KernelLaunch failed`）；arange 方案固定 40 核，不涉及 |
| **零张量输入** | Yes（源即如此） | 无可读张量；kernel 仅输出参数；无 N=0 空张量风险（S>0） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU 源） | 本项目（Ascend npuir） | 转换方案 |
|--------|-------------------|----------------------|----------|
| Kernel 装饰器 | `@tilelang.jit(out_idx=[0])` | `@tilelang.jit(target="npuir")`（**无 out_idx**） | §0.6 R5（实测强制） |
| Kernel 声明 | `T.Kernel(ceildiv(N,bs), threads=256)` | `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步 | §0.6 R2 |
| 并行模型 | `T.Parallel(threads,npt)` 线程级 | 一维核 + arange ramp + 整-buffer v-op | §0.6 R1/R2（`T.Parallel` 可用但慢 59×） |
| **坐标生成** | **flat→`//`,`%` 逐元素 div/mod** | **int16 arange 行/列 ramp + vsub/vabs** | §0.6 R1（性能裁决，实测 59×） |
| **距离复用** | 源每元素重算 `\|i-j\|` | **D 在 head 维共享** | §0.6 R1 / §1.6.1 #1（新增优化） |
| slope | 每元素 `T.exp2` | 每 head 标量 `T.exp2` | §0.6 R3 |
| abs | `if_then_else(dist>0,dist,-dist)` | `T.vabs`（向量 fp32 域）；基线路径可用 `T.max(d,-d)` | §0.6 R1（`if_then_else` 在 `T.Parallel` 崩溃，§3.4） |
| Buffer 层级 | 无显式 SMEM（逐元素直写） | `T.alloc_ub` | §0.5 等价替换 |
| prim_func 参数 | `main(out: T.Tensor((N,),dtype))` | `main(out: T.Tensor((H,S,S),dtype))`（显式输出，无 out_idx） | §0.6 R5；输出形状更自然 |
| strategy/autotune | `register_copy` 无；`threads/npt` 启发式 | 无 strategy、无 autotune；`(bi,bj)` tile 选择 | K8/K9/K11 折叠为 tile |
| fp8 | `_get_fp8_output_dtypes` 分支 | 无 | 舍弃（manifest 仅 fp16/bf16/fp32） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `testing/npuir/creation_ops/test_arange.py` | 高度相似（索引生成） | Developer `T.npuir_arange` 的 `[1,0]`/`[0,1]` 行/列 ramp + 显式输出张量（**证明 arange 需配显式输出**） |
| `testing/npuir/parallel_ops/`（12 测试；尤其 `test_floordiv_floormod_parallel.py`、`test_cast_parallel.py`、`test_min_max_parallel.py`、`test_unary_math_parallel.py`、`test_loop_var_compute.py`） | 高度相似（**div/mod + `T.Parallel` 写 UB 的能力证据**） | Developer `T.Parallel`+`alloc_shared`(UB) 逐元素算子；int64 `//`,`%` `rtol=0,atol=0`；**判断「标量计算形态是否可 Developer 化」的一手证据位点**（v0 遗漏，本版补列） |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` | 高度相似（索引生成） | L421–435 `T.arange(colmat,[0,1],0)`/`T.arange(rowmat,[1,0],0)` + `vsub`（distance 同形） |
| `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/_ssd_chunk_scan_fwd_kernel.py` | 高度相似（索引生成） | L349–353 `arange` 行/列 ramp + `vsub` |
| `examples/elementwise/vec_add_2d.py` | 高度相似 | `T.Kernel(BLOCK_SIZE, is_npu=True)` + `T.serial(ceildiv(...))` + `if block_id < 总任务` 网格跨步 |
| `examples/leaky_relu/_make_leaky_relu_kernel/DESIGN.md` | 相似（同族迁移方法论） | 迁移文档结构、设备探针方法论、候选表/`verify_equiv.py` 范式、`out_idx`→显式输出的接口注意 |
| `docs/Tilelang.language/创建操作/T.arange.md`·`数学操作/T.vsub.md`·`T.vabs.md`·`T.vmul.md`·`T.vdiv.md`·`数据类型转换操作/T.vcast.md`·`编译器提示操作/T.Parallel.md` | API | 各 op 规格、dtype 矩阵与用法 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

**无张量输入**（生成式算子）。`seq_len`/`num_heads`/`dtype` 为构造期编译期参数（非张量）。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `(H, S, S)`（`H=num_heads`，`S=seq_len`） | `float16 \| bfloat16 \| float32` | ALiBi bias；**唯一张量参数（显式输出，无 out_idx）**；write-only |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `R` | `(bi, bj)` | `int16` | UB | 行坐标 ramp（`i0+r`） |
| `C` | `(bi, bj)` | `int16` | UB | 列坐标 ramp（`j0+c`） |
| `Rf` | `(bi, bj)` | `float32` | UB | 行坐标 fp32 → **距离 `D`（原地）** |
| `Cf` | `(bi, bj)` | `float32` | UB | 列坐标 fp32 |
| `O` | `(bi, bj)` | `float32` | UB | 每 head 缩放结果（`s_h·D`），**跨 head 原地复用** |
| `O16` | `(bi, bj)` | `dtype` | UB | `vcast` 单次舍回缓冲（仅 fp16/bf16 路径） |

> **buffer 分配策略**：全部在 `with T.Kernel` 顶层无条件分配（避免 `TRAP-conditional-alloc`）；`D` 原地落在 `Rf`；`O` 每 head 原地复用（先读 D，再写 O，生命期不交）。**可选 UB 优化**（须 Stage 3 重验）：把 `Cf` 复用为 `O`（`vcast(C,O)` → `vsub(Rf,O,Rf)` → `vabs` 后 `O` 释放供 head 循环用），省 4B/elem。

### 4.4 内存搬运路径

```
（GM 无读）
GM                                                              UB
────                                                            ────
                                             arange → R,C (int16)
                                     vcast(rint) → Rf,Cf (fp32)
                                     vsub/vabs   → Rf = D (fp32)
  for h in serial(H):
      (标量 exp2) → s_h
                                     vmul(D,s_h) → O (fp32)
      [fp16/bf16] vcast(rint)       → O16
      T.copy(O16|O, out[h,i0,j0]) ←──────────────────────────────
```

### 4.5 UB 内存预算

> **本版修正**：v0 用解析式 `int16×2 + f32×3 + dtype×1`（fp16 18 B/elem）并乘 1.12×，得 8192 elems = 144KB、「16384 超限」。**设备实测**（`history_version/design_probe_alibi_ub_ceiling.py` UB 天花板探针，已落盘复现）显示 BishengIR 按**生命期复用**分配，实际峰值 = **12 B/elem**（距离链 `R+C+Rf+Cf` = 2+2+4+4 = 12 B/elem 为峰值；head 循环 `Rf+O+O16` ≤12 B/elem），UB 硬上限 = **1572864 bits = 192 KB/AIV**（报错报文实证）。

| 路径 / `bi·bj` | 实测峰值（12 B/elem） | 编译结果（设备） |
|---------------|----------------------|------------------|
| fp16 / bf16 @ 8192 | 96 KB | ✅ compile+run OK |
| fp16 @ 12288 | 144 KB | ✅ OK |
| fp16 / fp32 @ 16384 | 192 KB（= 上限） | ✅ OK（恰贴合） |
| fp16 @ 24576 | 288 KB | ❌ `ub overflow, requires 2359296 bits while 1572864` |

- **UB 硬上限** = 192 KB/AIV（`1572864 bits`，设备报文实证；与 `CONST-capacity-910B2C` 同量级，该条目绑 910B2C **且 stale**，以本设备实测为准）。
- **实测天花板** = **`bi·bj ≤ 16384`**（192KB / 12 B/elem）。**默认取 8192**（96 KB，留 2× 余量）。
- **`auto-multi-buffer` 状态（问题 3 回应）**：非 A5 设备 `--enable-auto-multi-buffer` **默认 True**（`tilelang/jit/jit_npu.py:1890–1893`）；本设计**不显式关闭**。**关键实测**：`bi·bj ∈ {8192,12288,16384}` 在 auto-multi-buffer 默认 ON 下均 compile+run OK，说明本 kernel（显式 `alloc_ub` 结构）**未受 `TRAP-UB-multibuffer-inflation`（~1.7×）影响**——后者绑 attention 类 Pipelined 多缓冲场景（`CONST-capacity-910B2C` 头号条目），本 kernel 无 `T.Pipelined`，故 auto-multi-buffer 不膨胀这些 buffer。v0 的 1.12× 解析系数与 1.7× 口径**均不必采用**：以实测 12 B/elem + 192KB 上限为操作界。
- **block 放大空间**：`bi·bj=4096` → 48 KB（宽裕）；`8192` → 96 KB（默认）；`16384` → 192 KB（贴限，留 0 余量，不建议）。故 `bi·bj ∈ {4096, 8192}` 为 Stage 4 发射项调优杠杆（§5.2），默认 8192。
- **Stage 3/4 校准**：若编译器版本变化导致分配膨胀，按 BishengIR 报文 `ub overflow, requires N bits` 反推实际需求（`CG-2026-0008` 的探针步骤同口径）；`bi·bj` 下调即可。

### 4.6 动态轴定义

**无动态轴**：`S`、`H` 均为构造期编译期常量（`lru_cache` 按 `(seq_len,num_heads,dtype)` 特化），参与 index/循环静态化。输出 shape 完全由编译期常量决定。

> **KB 预注入条目适用性**：`PL-1.19-devmode-1d-runtime-shape`（Developer 一维 kernel 的运行时 `shape: T.int32` 标量参数）**不适用**——本算子无输入、无动态轴（S 编译期），故无运行时 shape 尾守卫需求。`PL-1.22-persistent-1d-grid-stride`（核内网格跨步）与 `PL-1.23-vop-chain-inplace-alias`（UB 原地复用纪律）**适用**并在 §0.6 R2 / §1.6.1 #3 引用。

### 4.7 JIT 配置

```python
@tilelang.jit(
    target="npuir",
    # 注意：不使用 out_idx（实测 out_idx + arange 设备崩溃，§0.6 R5 / CG-2026-0016）
)
def _make_alibi_kernel(seq_len, num_heads, dtype=..., bi=None, bj=None, budget=8192):
    ...
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector（生成式）

**判定依据**: 算子仅含 arange 索引生成 + 逐元素向量链（`vcast/vsub/vabs/vmul`），无 matmul、无归约、无 Cube 计算；仅需 UB 层级，不涉及 L1/L0A/L0B/L0C。

### 5.2 Block 划分

```python
# (bi, bj)：空间 (i,j) 平面 tile；bi≥2（CG-2026-0017）；bi|S、bj|S；优先 bj=S（写出连续）
BUDGET = 8192                          # UB 实测天花板 16384（12 B/elem），默认取 8192 留 2× 余量
bj = S                                 # 若 S ≤ BUDGET//2；否则 bj = 最大满足 bi·bj ≤ BUDGET 的 S 因子
bi = max(2, BUDGET // bj)              # 默认；与 §3.3 伪代码 bi 默认值一致
nb = S // bj                           # 列块数
n_tiles = (S // bi) * nb               # 逻辑 tile 数
```

**推荐取值**（`BUDGET = 8192`）：

| workload | S | H | bj | bi | bi·bj |
|----------|---|---|----|----|-------|
| `test_alibi` | 128 | 8 | 128（=S，全宽） | 64 | 8192 |
| llama-prefill-2k | 2048 | 32 | 2048（=S，全宽） | 4 | 8192 |
| llama-prefill-4k | 4096 | 32 | 4096（=S，全宽） | 2 | 8192 |

**选择理由**：
1. **bj=S** 时 `T.copy` 为 `bi·S` 连续元素（连续写出更优；`PL-1.14` stale，不作决策依据，见 §1.6.3）；
2. **bi≥2** 为设备实测约束（`(1,N)` arange 尾元素垃圾，CG-2026-0017）；
3. **bi·bj≤8192** 满足 UB 预算（实测 96 KB，天花板 16384=192KB）；
4. **发射项权衡**：发射 ∝ 1/(bi·bj)（§1.6.0 估算发射项主导），8192 在 UB 约束下接近上限；`bi·bj=4096/8192` 列为 Stage 4 调优杠杆（16384 实测可编译但贴 UB 上限，不推荐）。

> **与 §3.3 默认值的关系**：§3.3 伪代码签名 `bi=None` → 运行时按 `bi = max(2, budget//bj)`（budget 默认 8192）推导，与本表推荐值一致（test → bi=64、2k → bi=4、4k → bi=2）；调用方可显式覆盖 `bi`（Stage 4 调优）。

### 5.3 约束分析

- **对齐约束**：`bj·elem_bytes` 为 32B 整数倍（bj=S ≥128；fp16 bj=128→256B、fp32→512B）✓；向量宽 fp16/bf16 ×8、fp32 ×4，`bj%8==0` → 尾 lane 浪费 0%。
- **UB 容量**：实测峰值 12 B/elem；8192 elems → 96 KB < 192 KB ✓（设备 compile+run 实证）。
- **L0/L1 容量**：不适用（纯 Vector）。
- **分形限制**：不适用（无 Cube）。
- **`arange` tile 首维**：`bi≥2` ✓（CG-2026-0017）。
- **int16 ramp 值域**：`i0+bi-1 ≤ S-1 ≤ 4095 < 32767` ✓（S≤4096 时精确；更大 S 需评估 int32 ramp）。
- **coreDim ≤ 65535**：arange 方案固定 40 核 ✓（基线 1-D flat 才涉及，见 §3.5.1）。

### 5.4 注意事项（非整除处理策略）

**非整除场景预判**：manifest `S ∈ {2048,4096}`、test `S=128` 均为 2 的幂，`bi|S`、`bj|S` 成立（推荐取值全整除）。**通用 S 策略**（kernel 内，不改动 host、不做 padding）：

- `S` 为编译期常量，故 `S` 的因子在编译期已知：选 `bj ∈ {最大 ≤ BUDGET//2 的 S 因子（优先 =S）}`、`bi = max(2, BUDGET//bj)`（须 `bi | S`）；`nb = S//bj`、`n_tiles = (S//bi)·nb`。
- 若 `S` 无合适因子（如 S 为大素数），退化为 **`bi=2`、`bj=2`/最小因子** 或**尾块守卫**：向量 op 仍作用于整 buffer（arange 生成超出 S 的索引无害，因为只 `T.copy` 有效区），行方向以 `i_tail = T.min(bi, S-i0)`、`j_tail = T.min(bj, S-j0)` 裁剪 `T.copy`（`T.copy(O16[0:i_tail,0:j_tail], out[h, i0, j0])`）。**注意**：`bi` 非整除 S 时行尾 tile 会越界写，故行方向**必须** `T.min` 裁剪。此回退为 Stage 3 待实现/待验证项（列入 §9 风险）。
- **本任务范围内（S∈{128,2048,4096}）全部整除，无需尾块处理。**

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

**① 逻辑核数计算**：输出网格为 `(i,j)` 空间 tile（head 在 tile 内 `T.serial`）：

```
num_logical_kernels = ceil(S/bi) × ceil(S/bj) = n_tiles
```

| workload | S | H | bi | bj | 逻辑核数 |
|----------|---|---|----|----|----------|
| test_alibi | 128 | 8 | 64 | 128 | 2×1 = **2** |
| llama-prefill-2k | 2048 | 32 | 4 | 2048 | 512×1 = **512** |
| llama-prefill-4k | 4096 | 32 | 2 | 4096 | 2048×1 = **2048** |

**② 物理核数依据（实查）**：目标设备 `Ascend910_9362`，实查代码与返回值（纯 Vector 算子核数翻倍）：

```python
from tilelang.utils.npu_utils import NPUUtils
NPUUtils.get().get_aicore_num()        # 实际返回 20
NPUUtils.get().get_aivector_core_num() # 实际返回 40（= aicore × 2）
```

→ **纯 Vector 物理核数 = 20 × 2 = 40**（2026-10-13 本会话实查；与 `CONST-aicore-910_9362` 一致）。注：`CONST-aicore-910B2C` 记 24 核，绑定 Ascend910B2C，**以实查为准**。

**③ 规模判定与分核方案**：

- **test_alibi（逻辑核数 2）**：逻辑核数 ≤ 物理核数 40 → **无需适配**（1 波次，其余核空转；`T.serial(ceildiv(2,40)) = T.serial(1)` 单次迭代）。依据：逻辑核数 2 < 40。
- **manifest（逻辑核数 512 / 2048）**：**极大规模**——逻辑核数远超物理核数 40，且 S 为 2 的幂、`bi` 为 2 的幂，`n_tiles` 为 2 的幂，物理核 40 = 2³×5 含非 2 因子，数学上**无法**通过对齐 block 使逻辑核数为 40 的整数倍（5 ∤ 2ⁿ）。按标准 §1 要素③「极大规模」处置：
  - **固定启动内核数 = 物理核数 40**；
  - 核内 `T.serial` 网格跨步处理多个逻辑 tile：
    ```python
    for t in T.serial(T.ceildiv(n_tiles, core_num)):   # 上界静态（n_tiles、40 均编译期常量）
        task = t * core_num + cid                      # 网格跨步
        if task < n_tiles:                             # 守卫
            ...
    ```
  - **循环边界静态** ✓（`n_tiles`、`core_num` 均编译期常量，`T.ceildiv(n_tiles,40)` 为静态整数）；
  - **残留负载不均**（显式记录）：llama-prefill-4k（n_tiles=2048）`ceildiv(2048,40)=52`，前 `2048-40×51=8` 核执行 52 次、后 32 核 51 次 → 临界路径 ≈ **+1.3%**（52/51.2）；llama-prefill-2k（512）`ceildiv(512,40)=13`，前 32 核 13 次、后 8 核 12 次 → 临界 **+1.6%**。相较「直接 `T.Kernel(n_tiles)` 由运行时串行调度多波次」，核内 `T.serial` 把核启动固定开销从「每 tile 一次」摊薄为「每核一次」，净收益为正。
- **模式先例**：`examples/elementwise/vec_add_2d.py` L28–L42；`PL-1.22-persistent-1d-grid-stride`。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block 级 / tile 级调度循环；kernel 内无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核级（任务网格） | block 级并行 | `T.Kernel(40, is_npu=True)` | 40 个物理 Vector 核各处理一组空间 tile |
| tile 级（核内） | 串行迭代（网格跨步） | `T.serial(ceildiv(n_tiles,40))` | 每核串行处理 `ceildiv(n_tiles,40)` 个空间 tile，摊薄核启动开销（§5.5） |
| head 级（tile 内） | 串行迭代 | `T.serial(H)` | 每空间 tile 内按 head 复用距离 D（§1.6.1 #1）；每次 1 `vmul`(+1 `vcast`) |
| 元素级 | 向量化 | `T.npuir_arange` + `T.vcast/vsub/vabs/vmul` | 整 buffer 一次完成，无 `T.Parallel` 标量循环 |

**T.Persistent 选择说明**：以 `T.serial` 网格跨步实现 persistent 语义（每核固定处理多 tile），与 `T.Persistent` 等价且为同族先例形态（`PL-1.22`）；不额外引入 `T.Persistent`（核数固定 40 已足够）。

### 6.2 循环伪代码

见 §3.3（`with T.Kernel(40, is_npu=True)` + `for t in T.serial(...)` + `if task < n_tiles`；tile 内 `for h in T.serial(H)` + 距离链/vmul/vcast）。

### 6.3 流水线优化

**当前设计**：不使用 `T.Pipelined`。理由：write-only 单数据通路（无 GM→UB 载入可重叠），单 tile 内数据依赖线性（arange→vcast→vsub→vabs→[vmul→vcast→copy]×H）；跨 tile 迭代（`T.serial`）的 MTE3 写出与 VEC 计算重叠由 Developer 模式 **auto-multi-buffer（默认 ON，`jit_npu.py:1890`）** 自动处理。**发射项主导下**，若 Stage 4 需进一步优化（更大 `bi·bj` / head 循环与写出重叠），参照 `example_elementwise_exp2.py` 的 `T.serial` 多 tile 形态；设计阶段保持简洁。

### 6.4 尾块处理

见 §5.4。本任务范围（S∈{128,2048,4096}）全部整除分块，无尾块；通用 S 的尾块守卫（行/列 `T.min` 裁剪 `T.copy`）列为 Stage 3 待实现项（§9 风险）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下编译器自动插入同步。核内数据依赖链严格线性：

```
arange(R) ∥ arange(C) → vcast(Rf) ∥ vcast(Cf) → vsub → vabs → [vmul → vcast → T.copy] × H
```

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `arange` 后 | 自动 | 编译器插标量/V 依赖 |
| 各 v-op 之间 | 自动 | 线性数据依赖（`vsub→vabs` 串行；`vmul→vcast` 串行） |
| `T.copy(UB→GM)` 前 | 自动 | 编译器插 V→MTE3 等待 |
| head 迭代之间 | 自动 | `O` 原地复用由 auto-multi-buffer 管理跨迭代依赖 |
| tile 迭代之间 | 自动 | 同 buffer 复用管理 |

无手动 `T.sync_block_set/wait`（无跨引擎/跨核协作，无 CV 融合）。

### 7.3 pass_configs 配置

```python
# Developer 模式默认配置，无需额外 pass_configs
# os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
# auto-multi-buffer 默认 True（jit_npu.py:1890），本设计不改动
```

### 7.4 CV 融合设计

**n/a**：纯 Vector 生成式算子，无 Cube 段（无 `T.gemm`）、无跨引擎协作、无核间 workspace，不涉及 CV 融合。据 AGENTS.md「Developer-mode MixCV trigger rule」——kernel 内既无 Cube 侧 `T.gemm` 亦无跨核流水，**非 MixCV**，不触发 mixcv skill。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（移植 `test_special_elementwise.py::test_alibi` 的 torch 参考式），**不得复刻 §0.6 的 NPU 结构**——保证验证独立性（不复用 arange ramp / D 共享结构）。

```python
import torch

def golden_alibi(seq_len: int, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """ALiBi bias 参考实现（移植 test_special_elementwise.py::test_alibi）。

    bias[h,i,j] = -slope_h * |i-j|,  slope_h = 2^(-8*(h+1)/H)
    fp32 域计算 |i-j| 与 slope（pow(2,·)），末端单次 .to(dtype)。
    独立于 NPU 的 arange-ramp / D-shared 结构（验证独立性）。
    注：以「实现 + golden」的 (h+1) 指数为准（源工厂 docstring 的 2^(-8h/H) 为过时描述，见 §0.1）。
    """
    positions = torch.arange(seq_len, dtype=torch.float32)
    dist = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()          # (S,S)
    slopes = torch.pow(2.0, -8.0 * torch.arange(1, num_heads + 1,
                                                dtype=torch.float32) / num_heads)
    return (-slopes[:, None, None] * dist[None, :, :]).to(dtype)           # (H,S,S)
```

> 备注：golden **独立于** NPU 结构（不与 `D-shared`/`arange` 复用），满足验证独立性。**契约 = fp32 域 `pow(2,·)` + `|i-j|` + 单次舍回**；NPU 侧以 fp32 域 + 设备 `exp2`（1 ulp）实现，末端舍回后与 golden 一致（§3.4 Q1/Q2 实测 0 违反）。

### 8.2 精度标准

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 1e-5 | 1e-5 | 来源：`test_special_elementwise.py::test_alibi`（fp32 分支） |
| float16 | 1e-2 | 1e-2 | 来源：同上（fp16 分支） |
| bfloat16 | 1.6e-2 | 1.6e-2 | 来源：本仓 elementwise bf16 通用容差（`test_activation.py::_get_tolerances` 同口径；alibi test 未单列 bf16）——bf16 尾数 8 bit，大距离处 1 ulp 约 4–8 |

**误差预算核对（设计期设备实测，`design_probe_alibi_v1.py` Q1/Q2）**：fp16 `test` **bit-exact**（两条坐标路径）；fp32 `test` max_abs=3.815e-6（<1e-5）；fp16 `manifest-2k` 违反 0；bf16 `manifest-2k` max_abs 4.0（1 ulp，落在 `rtol·|ref|` 内）；`int16-range` S=4096 违反 0。全部 **0 违反**。

### 8.3 L0 门槛测试计划

> L0 = 门槛 smoke 测试（最小可用性验证，进 Stage 3 前必过）；完整分层套件 L1/L2/Boundary 交由 `tilelang-op-develop`，不在此枚举。**显式包含 test 用例、H=8 精确幂 slope、各 dtype、大 S、int16 值域、对角零、非整除尾块**。

| 用例 | 输入（构造参数） | 期望 | 判定 |
|------|-----------------|------|------|
| L0-1 test fp16（**含 H=8 slope 精确幂**） | `(S=128, H=8, fp16)` | 与 `golden_alibi` 比对 ≤1e-2（设计期设备实测 **bit-exact**） | 门槛通过 |
| L0-2 test fp32 | `(S=128, H=8, fp32)` | 比对 ≤1e-5（设计期实测 3.815e-6） | 门槛通过 |
| L0-3 manifest smoke fp16 | `(S=2048, H=32, fp16)` | 比对 ≤1e-2（设计期实测 3.906e-3，违反 0） | 门槛通过 |
| L0-4 manifest smoke bf16 | `(S=2048, H=32, bf16)` | 比对 ≤1.6e-2（设计期实测 1 ulp，违反 0） | 门槛通过 |
| L0-5 大 S fp16 | `(S=4096, H=32, fp16)` | 比对 ≤1e-2 | 门槛通过 |
| L0-6 int16 值域（`\|i-j\|>2048`，S 超 fp16 精确域） | `H=1, S=4096, fp16`（或 H=32） | 比对 ≤1e-2（设计期实测 0 违反） | 门槛通过 |
| L0-7 对角零 | 设计期检查 `out[:, i, i] == 0` | 精确 0（设计期 corner 实测 True） | 门槛通过 |
| L0-8 小 S / 单 tile | `(S=64, H=4, fp16)` | 比对 ≤1e-2 | 门槛通过 |
| L0-9 各 dtype 覆盖 | `{fp16, bf16, fp32}` × `(S=256, H=32)` | 各容差比对 | 门槛通过 |
| L0-10 非整除尾块（Stage 3 待实现） | 如 `S=1000`（非 2 幂） | 比对 ≤1e-2（**需先实现 §5.4 尾块守卫**） | 门槛通过 |

- **执行方式**：`kernel = _make_alibi_kernel(S, H, dtype, bi=..., bj=...)`；`out = torch.empty(H,S,S, device="npu", dtype=...)`；`kernel(out)`；`torch.testing.assert_close(out.cpu(), golden_alibi(S,H,dtype), atol, rtol, equal_nan=True)`（golden = §8.1）。
- **门槛标准**：L0-1..L0-9 全过即进入 Stage 3；L0-10 在尾块守卫实现后补。
- **额外设计期落点**：`history_version/design_probe_alibi_v1.py`（设备探针 Q1–Q4）、`history_version/design_probe_alibi_ub_ceiling.py`（UB 天花板，rev2 落盘）、`history_version/design_probe_alibi.py`（v0 P1–P5，前序证据）、`verify_equiv.py`（D-1 机器验证，46/46 `EQUIV_PASS`）。**注**：§3.4 的 `T.Parallel` 能力 bisect 为会话内实测、未落盘（见 §3.4）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| 约束 | 影响 | 缓解方案 |
|------|------|----------|
| **R-1 `out_idx` + `T.npuir_arange` 设备崩溃**（本算子关键，CG-2026-0016） | 若照源用 `out_idx=[0]` 自动输出，运行期 `MTE DDR out of range`（507057） | **显式输出参数**（无 `out_idx`），wrapper 侧 `torch.empty` 分配（§0.6 R5）；设备实测该形态全 PASS |
| **R-2 `arange` tile 首维 ≥2**（CG-2026-0017） | `bi=1` 时 arange 尾元素垃圾（静默错误） | `bi≥2`（§5.2）；`(1,N)` 形态禁用 |
| **R-3 向量 `vexp2` 不可靠**（CG-2026-0018） | Dev 模式部分元素 inf | 用**标量 `T.exp2`**（§0.6 R3）；不用 vexp2 |
| **R-4 整数 div/mod 形态**（本版修正） | int64 可用（标量 `T.Parallel` 与向量 `vdiv/vfloordiv`）；**int16/int32 无独立 v-op**；`T.Parallel` 内 `if_then_else`/`fabs(局部)` 崩溃 | 本设计走 arange ramp，不涉及 div/mod；若回退基线须用 `T.max(d,-d)`（§3.4）。**不得再断言「NPU 无 div/mod」** |
| **R-5 3-D v-op 广播不支持** | head 不能入 tile 与 (i,j) 广播 | head 由 `T.serial` 循环承载（§1.6.3 #4） |
| **R-6 发射项主导**（`6+2H` op/tile，`CONST-vector-launch-overhead`） | §1.6.0 估算发射项远超流量（448µs vs 213µs @2k），**实测 660µs** | 标注「未文档化假设 + 估算依据」（常数绑 910B2C、stale）；`bi·bj` 为杠杆（§5.2）；Stage 4 baseline 后按 `hardware-cost-model.md` §3 对账 |
| **R-7 UB 容量** | `bi·bj` 放大受限 | 实测峰值 12 B/elem、天花板 **16384**（192KB，设备报文实证）；默认 `bi·bj=8192`（96KB）；auto-multi-buffer 默认 ON 且本 kernel 未膨胀（§4.5） |
| **R-8 bf16 1-ulp 差异** | S=2048 大距离处 bf16 与 golden 差 1 ulp（max_abs 4.0） | 落在 `rtol·|ref|` 内（违反 0，§3.4/§8.2）；bf16 容差取 1.6e-2 |
| **R-9 slope 设备 exp2 精度** | 设备 `T.exp2` vs `pow` 1 ulp（rel ≤2.5e-7@H=32） | 末端 dtype 舍回吸收（Q1/Q2 0 违反）；fp32 `test` 3.815e-6<1e-5 |
| **R-10 非整除 S 尾块未实现** | 通用 S（非 2 幂）行/列越界写 | 本任务 S∈{128,2048,4096} 全整除；通用 S 需 §5.4 尾块守卫（Stage 3 待实现/验证） |
| **R-11 条件 `T.alloc` 非法** | 编译期 `if` 内 `T.alloc_ub` trace 失败 | buffer 全部顶层无条件分配（§3.3） |
| **R-12 常数表设备/版本** | `CONST-capacity-910B2C`/`CONST-mte2-degradation`/`CONST-vector-launch-overhead` 绑 910B2C 且 stale；本设备 910_9362 | 容量/带宽/发射数字标注「待重验」，仅作量级估算；**关键结论来自本会话设备探针**；UB 上限用本设备报文（1572864 bits）实测替代 |
| **R-13 开发期 cache 污染** | 会话中 `tilelang.cache` 命中旧 kernel 导致误判 | 关键探针用 `TILELANG_DISABLE_CACHE=1` + 独立进程；Stage 3 亦同 |
| **R-14 `arange` 仅在 ≥2 维、显式输出、Developer/tilelang.compile 通路验证** | 其他通路（Expert/Scope）未验证 | 本设计固定 Developer + 显式输出 + 2-D tile；工具链升级后须重验 |
| **R-15 `T.Parallel` 内条件式崩溃（本版新增发现）** | `if_then_else(cond_on_loopvar,...)`、`T.fabs(局部变量)` 在 `T.Parallel` 内**编译期 segfault**（`T.Parallel.md` §2.2.1 有条款） | 本设计不依赖 `T.Parallel`；若走基线须用 `T.max(d,-d)`/inline `fabs` 替代（§3.4）。此为**有文档条款**的限制，不得泛化为「T.Parallel 不可用」 |
| **R-16 coreDim ≤ 65535（仅基线）** | 1-D flat 基线块数超限（2k/32 需 block≥2049） | arange 方案固定 40 核，不涉及 |
| **R-17 工具链版本** | 本设计设备结论绑 tilelang 0.1.2+87a3307b / CANN 9.0.0 / Ascend910_9362 | 工具链升级后须重验 Q1–Q4 与 UB 天花板（`kb_stale_check.py` 口径） |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 用 `out_idx` 自动输出 | 照抄源装饰器 | 运行期设备崩溃（507057） | 显式输出参数（R-1） |
| `bi=1` tile | 追求最大写出连续性 | arange 尾元素垃圾（静默错误） | `bi≥2`（R-2） |
| 用向量 `T.vexp2` 算 slope | 想省标量 exp2 | 部分元素 inf | 标量 `T.exp2`（R-3） |
| **误判「NPU 无 div/mod」** | 凭记忆/GPU 先验 | 错误否决可实现基线 | 三查（`customize_npuir.py` 导出 + `testing/npuir/parallel_ops/` + `docs/Tilelang.language/数学操作/T.vdiv.md`）（R-4） |
| **误判「`T.Parallel` 写 UB 不可用」** | 把 CG-2026-0016（out_idx+arange）泛化 | 错误否决在仓已验证形态 | `T.Parallel.md` §2.3–2.4 + `parallel_ops` 12 测试 + 设备 Q1（R-15） |
| 在 `T.Parallel` 内用 `if_then_else` 求 abs | 照抄源 `if_then_else(dist>0,dist,-dist)` | 编译期 segfault | 用 `T.max(d,-d)`/inline `fabs`；或走向量 `T.vabs`（R-15） |
| fp16 ramp 生成坐标 | 想省 UB | S>2048 时不精确（静默错误） | int16 ramp（R-4 / §1.6.1 #3） |
| `vabs` 作用于 int16 | 想在 int16 域取绝对 | 编译失败（vabs 无 int16） | `vcast` 升 fp32 后 `vabs` |
| 3-D v-op 广播 | 想把 head 入 tile | 编译失败 | head 用 `T.serial`（R-5） |
| v-op 传动态 slice | `T.vmul(D[:, 0:tail], ...)` | 编译/运行错误 | v-op 用整 buffer，仅 `T.copy` 用尾 slice |
| 条件 alloc | 编译期 `if need_cast:` 内 `T.alloc_ub` | trace `Undefined variable` | buffer 顶层无条件分配（R-11） |
| UB 溢出 | `bi·bj` > 16384 | `ub overflow, requires N bits` | 维持 `bi·bj≤8192`（R-7） |
| 基线块数超 coreDim | 1-D flat `block` 过小 | `KernelLaunch failed ... coreDim ... 65536` | 仅基线涉及；arange 方案固定 40 核（R-16） |
| 非整除 S 无尾守卫 | S 非 2 幂 | 行/列越界写 | §5.4 尾块守卫（R-10） |
| 开发期误判 | `cache` 命中旧 kernel | 结论错误 | `TILELANG_DISABLE_CACHE=1` + 独立进程（R-13） |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| **`i=j`（对角）** | `D=0` → 输出精确 `0`（golden 同；设备 corner 实测 True） |
| **`H=8`（slope 精确幂）** | `slope_h=2^-k` 精确；设备 `exp2` 1 ulp 内；`test` 用例 bit-exact |
| **`H=32`（slope 非精确幂）** | `exp2(-(h+1)/4)` 1 ulp 内；末端 dtype 舍回吸收（0 违反） |
| **大 S（4096，`\|i-j\|>2048`）** | int16 ramp 精确覆盖（≤4095<32767）；fp16 无法表示 4095 故**必须 int16/更高**，已用 int16 |
| **极小 S（S<bi）** | 取 `bi=2`、`bj=S`（S≥2）或 `bi=S`；逻辑核数 1，仅 `task==0` 核工作 |
| **S 非 2 幂** | §5.4 尾块守卫（Stage 3 待实现） |
| **bf16** | fp32 域计算 + `vcast(rint)` 单次舍回（尾数 8 bit，容差 1.6e-2） |
| **fp32** | 无需 `vcast` 舍回（`O` 直写 GM）；`O16` 可复用/空置 |
| **需回退 `T.Parallel` 标量路径时** | 用 `T.max(d,-d)` 求 abs；`block ≥ ceil(N_total/65535)` 避 coreDim 上限；`threads=` 无效果（TRAP-threads-kwarg-noop） |

---

## 10. 交付清单

| 交付物 | 路径 | 说明 |
|--------|------|------|
| 设计文档 | `examples/alibi/_make_alibi_kernel/DESIGN.md` | 本文件（修订第 2 版） |
| 等价性机器验证 | `examples/alibi/_make_alibi_kernel/verify_equiv.py` | D-1：46/46 `EQUIV_PASS`（§1.6.1） |
| 设计期设备探针（rev1） | `examples/alibi/_make_alibi_kernel/history_version/design_probe_alibi_v1.py` | Q1 基线 div/mod + Q2 arange 端到端 + Q3 exp2 + Q4 性能对比（§3.4） |
| 设计期设备探针（UB 天花板，rev2 落盘复现） | `examples/alibi/_make_alibi_kernel/history_version/design_probe_alibi_ub_ceiling.py` | §4.5 UB 天花板：8192/12288/16384 OK；24576 报 `ub overflow requires 2359296 bits while 1572864`（实跑复现 §3.4/§4.5 结论） |
| 设计期设备探针（v0，前序） | `examples/alibi/_make_alibi_kernel/history_version/design_probe_alibi.py` | P1–P5（v0 证据，新工具链已复核） |
| NPU kernel（下游 Stage 3 产出，尚未创建） | 本算子目录下 `_make_alibi_kernel.py` | 按 §3.3 伪代码 + §5 分核实现（显式输出、`bi≥2`、int16 ramp、标量 exp2） |
| Golden（下游） | 同上（kernel 文件内 `golden_alibi`） | 依 §8.1 |
| 能力缺口登记 | `.agents/evolution/capability-gaps.md` | CG-2026-0016（out_idx+arange）/ CG-2026-0017（1 行 arange）/ CG-2026-0018（vexp2 inf） |
| 迁移集成 wrapper（已存在，需按 §0.6 R5 调整 `forward`） | `examples/TileOPs/tileops/kernels/elementwise/alibi/alibi.py` | 显式输出：wrapper 侧 `torch.empty` 分配并传入 |
| 精度用例（已存在，只读参照） | `examples/TileOPs/tests/ops/test_special_elementwise.py::test_alibi` | `(128,8)` × `{fp16,fp32}` |
| 性能用例（已存在，只读参照） | `examples/TileOPs/benchmarks/ops/bench_independent_elementwise.py` | generative bench（`ref_api: none`） |

**实现要点速览（Stage 3 输入）**：
1. **显式输出参数**（无 `out_idx`）：`main(out: T.Tensor((H,S,S), dtype))`；wrapper `torch.empty` 分配并调用 `kernel(out)`；
2. **坐标生成**：`T.npuir_arange(R,[1,0],i0)` / `[0,1],j0`（int16）→ `vcast(rint)`→fp32 → `vsub` → `vabs` 得 `D`；**tile 首维 bi≥2**；
3. **D 共享 + 每 head 标量 slope**：`for h in T.serial(H)` 内 `slp=-T.exp2(-8(h+1)/H)` → `vmul(D, slp, O)` → `vcast(rint)` → `copy`；
4. 结构：一维 `T.Kernel(40, is_npu=True)` + `T.serial` 网格跨步 + `if task<n_tiles` 守卫；
5. tile：`bj=S`（写出连续）、`bi=max(2, 8192//bj)`；buffer 顶层无条件分配；距离在 fp32 域（vabs 无 int16）；
6. **不用**：`out_idx`、向量 `vexp2`、3-D v-op 广播、fp16 ramp（S>2048）；
7. **若走 `T.Parallel` 基线备选**：可用（写 UB），但须用 `T.max(d,-d)` 替代 `if_then_else`、`block≥ceil(N/65535)`；性能预计慢 ~59×（仅作对照，不作交付）。

---

## 11. 相对上一版的关键调整（revision 说明）

### 11.1 相对 v1 的关键调整（rev2，本次）

> 本节回应 Stage 2 检视（round 2，v1/rev2）的 **1 个阻塞项 + 2 个建议项**，逐条说明修改与「为何不会再犯同一错误」。技术结论（div/mod 可实现性、性能裁决 59.1×、arange ramp 选定、4 项优化、`verify_equiv.py` 46/46、分核三要素）**全部保持**。

| 检视项 | 上一版（v1）的问题 | 本版（rev2）的修正 | 为何不会再犯 |
|--------|-------------------|-------------------|--------------|
| **阻塞：§1.6.3 候选 #2 的 dtype 前提与所引文档矛盾** | 以「`T.transpose` bf16 ×」为弃选 #2 的**前提之一**，并自述「已亲自核对」；而 `docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1 dtype 矩阵实为 **bf16 √**（int8/int16/int32/uint64/int64/fp16/fp32 均 √，仅 uint8/uint16/uint32/bool ×） | §1.6.3 #2 **删去 dtype 限制前提**，弃选理由仅保留「**无收益**（距离对称、`bj` 已可作向量轴；转置不减少 op 数/访存）」；并新增「弃选前提取证说明」逐条核对 #2/#3/#4 所引 dtype 矩阵与代价条款 | 对 API dtype 负向断言一律**逐格回文档矩阵取值**（本次已全量重扫 §1.6.0/§1.6.3/§0.5/§0.6/§3.5.1 全部 dtype 断言，见 §11.1 末自查结论）；不再以「已亲自核对」自述代替实读 |
| **建议 2：bisect / UB 天花板探针未落盘** | §3.4 bisect（V0–V9）与 §4.5 UB 天花板结论归于 `design_probe_alibi_v1.py`，但该脚本仅 Q1–Q4 | ① **落盘** `history_version/design_probe_alibi_ub_ceiling.py`（UB 天花板，实跑复现同一报文）；② bisect 就地改述为「**会话内实测、未落盘**」并说明不构成 §4/§5 操作界 | 「设备实测」结论须「脚本 + 运行命令」可复现，否则显式标注「会话内实测（未落盘）」 |
| **建议 3：生成式算子 host permute 表述** | §1.6.3 #3 写「两次 host permute（进出）」，与零张量输入语义不符 | 改为「输出侧**一次** host permute」，并注明生成式输入侧无 permute | 表述与 §1.2 生成式（无张量输入）语义对齐 |

**rev2 dtype 断言全文自查结论**：逐条打开对应文档核对——① §1.6.3 #2 `T.transpose`（原「bf16 ×」→ 更正为 bf16 √，删除该前提）；② §1.6.3 #4 `T.vmul` 广播（`T.vmul.md` §2.2.2 仅列 2-D ✓）；③ §0.6 R4 / §1.6.1 #4 `vcast` f32→f16/bf16 `rint`（`T.vcast.md` §2.2.1 表 ✓）；④ §1.6.0 R4 / §3.5.1 / §9 `vabs` 无 int16（`T.vabs.md` §2.2.1：int16 ×、fp16/fp32/int32/int64 √ ✓）；⑤ §1.6.0 R4 / §3.5.1 `vdiv` 仅 int64（`T.vdiv.md` §2.2.1 ✓）；⑥ §1.6.0 R4 / §3.5.1 `vmul` 标量广播 fp16/fp32、bf16 ✗（`T.vmul.md` §2.2.1/§2.2.2 ✓）；⑦ §3.4 `T.vexp2`（文档 §1 自述 `exp(A*ln2)` ✓）；⑧ §1.6.0 R4 #7 `gather`（标「未文档化假设 + 估算依据」，不构成 dtype 断言）。**除 #1 外无其他 dtype 矩阵与文档矛盾。**

### 11.2 相对 v0 的关键调整（rev1）

> 本节回应 Stage 2 检视（round 1，v0）的 2 个阻塞项 + 3 个建议项，逐条说明修改与「为何不会再犯同一错误」。

| 检视项 | v0 的问题 | 本版（rev1）的修正 | 为何不会再犯 |
|--------|-----------|-------------------|--------------|
| **阻塞 1：负向断言被推翻（「NPU 无向量整数 div/mod」）** | §0.5/§0.6 R1/§1.6.0 R4/§3.4/§3.5.1 以「无向量 div/mod」否决源基线 | §0.5 处置改为「重新设计（**性能依据**）」；§1.6.0 基线列为**可实现候选**并补 op 数/访存/精度口径；§3.4 修正为「int64 有（`vdiv`/`vfloordiv`/`T.Parallel` `//`,`%`）；int16/int32 无独立 v-op」；§1.6.0/§1.6.1 新增**设备实测** `39.03 ms vs 0.66 ms（59.1×）` 作为裁决证据；`verify_equiv.py` 新增 #6/#7 证明基线数值等效 | 新增 R-4「三查」纪律（`customize_npuir.py` 导出 + `testing/npuir/parallel_ops/` + `docs/Tilelang.language/数学操作/T.vdiv.md`）；负向断言逐条附证据；设备探针（Q1）落地 |
| **阻塞 2：负向断言被推翻（「`T.Parallel` 写 UB 不可用」）** | §2.3/§3.4/§3.5.1/§1.6.2 断言 `T.Parallel` 写 UB 不可用 | §3.4 改为「**真实可用**」（`T.Parallel.md` §2.3–2.4 + `parallel_ops` 12 测试 + 设备 Q1）；§0.5 源 `T.Parallel` 结构列「重新设计（性能依据）」；区分「`T.Parallel` 本身可用」与「`out_idx`+arange 组合崩溃（CG-2026-0016）」；§3.5.3 补列 `testing/npuir/parallel_ops/` | 明确「CG-2026-0016 是该组合缺口，不可泛化」；新增 R-15 记录 `T.Parallel` 的**文档化**条件式限制（`if_then_else`/`fabs` 局部变量崩溃）但不泛化 |
| **建议 3：UB 膨胀口径（1.12× vs 1.7×）** | §4.5 用 1.12×，§6.3 依赖 auto-multi-buffer，口径未对齐 | §4.5 声明 `auto-multi-buffer` **默认 ON**（`jit_npu.py:1890`），给出**设备实测**峰值 12 B/elem + 天花板 16384（`ub overflow requires 2359296 bits while 1572864` = 192KB）；说明本 kernel 无 `Pipelined` 故不受 1.7× 膨胀影响；§6.3/§7.3 同步声明 | 用设备实测替代解析系数；口径全文一致 |
| **建议 4：四态口径 / 默认值一致性** | §0.5 把 #3/#4 标「保留/等价替换」，§0.6 又列为重设计项 R3/R4；§3.3 默认 `bi=2` 与 §5.2 推荐 `bi=64` 未说清 | §0.5 明确 #3/#4 处置层为「保留/等价替换」，§0.6 R3/R4 注明「属该层内的结构落地」；§3.3 伪代码 `bi=None` → `max(2, budget//bj)`（默认=§5.2 推荐），并加说明 | 统一「处置层」与「重设计项」措辞；默认值与推荐值显式绑定 |
| **建议 5：源 docstring/实现不一致** | §0.1 未标注 | §0.1 显式标注：工厂 docstring `2^(-8h/H)` vs 实现 `2^(-8(h+1)/H)`，**以实现 + golden 为准**；§0.4/§8.1 交叉引用 | 关键陷阱显式记录，防 Stage 3 误信 docstring |
