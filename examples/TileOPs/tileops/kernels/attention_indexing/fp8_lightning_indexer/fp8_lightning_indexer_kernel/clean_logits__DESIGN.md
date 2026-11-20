# clean_logits_ 算子设计文档（迁移）

> **项目**：`fp8_lightning_indexer` ｜ **算子目录**：`examples/fp8_lightning_indexer/clean_logits_/` ｜ **编程模式**：Developer ｜ **target**：`npuir`
>
> **任务性质**：迁移（migration-harness）。把 GPU TileLang 工厂函数 `clean_logits_`（`/home/developer/TileOPs/tileops/kernels/fp8_lightning_indexer.py` L152–L184）重设计为 NPU（`target="npuir"`）实现。本函数是 `FP8LightningIndexerOp` 的**第 2 个迁移单元**（同任务第 1 个单元 `_fp8_lightning_indexer_kernel` 已完成 Stage 1–3，见 `examples/fp8_lightning_indexer/_fp8_lightning_indexer_kernel/`）。
>
> **本版最关键的三项决策**：
>
> 1. **线程级谓词写 → block 级向量化三段式扫描**：源用 `T.get_thread_binding()` + `threads` + `if idx < n` 做**逐线程谓词标量写**（CUDA 专有，NPU 无线程绑定）→ 重设计为 **块对齐 tile 扫描 + 三态分类**（窗口外 tile 常量 `-inf` 直写、窗口内 tile 幂等跳过、边界 tile 掩码 `vselect`），全部 DMA 偏移保持块对齐（§0.6 R1/R2）。
> 2. **不把窗口端点当 DMA 偏移（规避同族实测 mis-compile 陷阱）**：同任务前序函数实测「标量 min/max 累加链作 DMA 偏移」在本工具链 mis-compile（`MPU address access is invalid`，见其 RETROSPECTIVE「Transferable Lessons」第 2 条）。本设计的**所有 `T.copy` 偏移 = `t*block_N`（循环导出）**，窗口标量 `cu_k_s/cu_k_e` 只出现在**标量分支条件**与**向量比较的操作数**里（`T.vcmp`/`T.vbrc` 的运行时标量形在本仓 NPU 先例中已实跑：`_gqa_prefill_fwd_kernel.py` L551/L872、`flash_qla/tilelang_get_warmup_chunks.py` L91/L95）。
> 3. **数学等价优化（已机器验证）**：把「逐元素条件投影 `out = cond ? x : -inf`」改写为**补集分解 + 只写不改**（窗口内值恒等不触碰）→ 窗口内零读零写；再加**幂等跳过**（窗口内 tile 无操作）与**常量直写**（窗口外 tile 免读免比较）→ `verify_equiv.py` 全项 `EQUIV_PASS`（逐位 max|d|=0，含 ±inf/NaN/−0.0/subnormal/空窗/满窗角点）。相对「整行 load+select+store」候选，主 workload 流量由 536MB 降至 ≤268MB（§1.6.0 R3）。

---

## 相对上一版的关键调整（revision 1，2026-10-15）

> **触发**：Stage 2 检视不通过（`REVIEW.md`：阻塞级 3 / 建议级 2）。本版**仅修订被检视指出的落地形态缺陷**，§0 源算子解读、§1.6.0 调研结论、§1.6.1 等价优化项（E1–E4）、§5.5 分核三要素等**已验证结论全部保留**，不推翻。

| # | 问题 | 上一版（design_v0） | 本版（revision 1） | 落地章节 |
|---|------|---------------------|--------------------|---------|
| 阻塞 1 | UB/shared buffer 用动态维 `T.dynamic("kv_group")` → `memref.alloc` 维度数校验失败 | `T.alloc_shared([block_N, kv_group], ...)`（`kv_group = T.dynamic("kv_group")`） | **编译期静态上界 `G_MAX`**（工厂参数，默认 2）：buffer 声明 `[block_N, G_MAX]`；`T.copy` 保留动态 extent 切片 `[0:tail, 0:kv_group]` | §0.6 R1、§1.6.3、§3.3、§4.3、§4.5、§4.6 |
| 阻塞 2 | v-op 缓冲用 `T.alloc_shared` → 运行时标量 `vcmp/vand/vselect` 链编译失败（`InternalError: cannot find variable clo`） | `T.alloc_shared`（§2.3 明写为 v-op 载体） | **全部 v-op 缓冲改 `T.alloc_ub`**（与本设计自引先例 `_gqa_prefill_fwd_kernel.py` L421–426 一致；`TRAP-vbrc-scalar-shared` 佐证） | §2.3、§3.3、§3.4、§4.3 |
| 阻塞 3 | `T.vcmp(idx_ub, cu_k_s - k0, ...)` 内联复合标量 → 前端 `<Sub> has no attribute buffer` | 内联 `cu_k_s - t*block_N` 作第二操作数 | **let-bind** `s_off = cu_k_s - k0` / `e_off = cu_k_e - k0` 后再传入 | §0.6 R2、§3.1、§3.2、§3.3、§1.6.2-#4、§9.1 |
| 建议 4 | §5.5「4096 档残留不均 +0.4%」复算不符 | `16 核 103、24 核 102 ≈ +0.4%` | 复算 `103/102.4 − 1 ≈ +0.59%` → 订正 **+0.6%** | §5.5 |
| 建议 5 | §1.6.0 D-2 候选 3 的 `vselect` 估算 ≈1 ms 偏高约 4× | `vselect ≈1 ms`、候选 3 ≈2 ms、选定≈8× 更优 | 按 `CONST-vector-op-cost-910_9362` 复算：`vselect ≈0.25 ms`；含 2×`vcmp`+1×`vand` 的完整 v-op 链 ≈0.36 ms → 候选 3 ≈**0.87 ms**、选定≈**3.5×** 更优 | §1.6.0 R4、§1.6.0 D-2 |

**为何不会再犯同一错误**：上一版把「算法正确性」与「落地形态可编译性」当作可互推（`verify_equiv` 全绿即认为 §3.3 伪代码可用）。本版新增 **D-3 设计期编译探针**（`history_version/design_probe_forms.py`，本机同工具链 `target="npuir"`），对三条硬约束逐一**编译实证**：① 静态形状 UB（动态维不可作 buffer 维）；② `T.alloc_ub` 承载 v-op 链；③ `vcmp` 标量操作数须 let-bound（`PROBE_ALL_AS_EXPECTED`，见 §3.4/§9.1）。后续任何 v-op 掩码链设计在冻结前先过该探针。

> **补充说明（建议 4/5 的结论不变性）**：两项建议均为**数字订正**，不改变对应结论——§5.5 4096 档仍属「动态 grid 无法对齐兜底」的记录项；§1.6.0 候选 3 仍被「流量 2×–4×（R3）」否决，`vselect` 修正后仅使选定算法的领先倍数由 8× 收敛到 3.5×（量级结论不变）。

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 tilelang-op-design skill 的 [references/migration-analysis.md](../../../.agents/skills/tilelang-op-design/references/migration-analysis.md)）：先彻底读懂源算子（0.1–0.4），再做算法调研（Phase R，结论见 §1.6.0），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。

**源算子路径（源真相）**：`/home/developer/TileOPs/tileops/kernels/fp8_lightning_indexer.py`

- 本单元工厂：`clean_logits_(threads: int = 512)`（L152–L184，`@tilelang.jit`），内部 `@T.prim_func clean_logits_kernel(Logits, CuSeqLenKS, CuSeqLenKE)`（L162–L182）。
- 同文件另一单元（**非本设计**）：主打分 kernel `_fp8_lightning_indexer_kernel`（L19–L149，已完成 Stage 1–3）。
- 集成 wrapper（NPU 侧接口契约）：`examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer.py` L87–L105——`if clean_logits: clean_logits_()(Logits, CuSeqLenKS, CuSeqLenKE)`。**K9 已去掉 `threads`**，故 NPU 工厂以**无参可调用**（`clean_logits_()(...)`）为契约；形状经 `T.dynamic` 从张量推断，不额外传 shape 参数。
- 抽取副本（脚手架，源真相镜像）：`examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/_f_p8_lightning_indexer_kernels.py` L157–L189。

### 0.1 源算子语义（做什么）

**数学语义**（对 Logits 做**原地**窗口掩码清理：窗口 `[cu_k_s(s), cu_k_e(s))` 内保留、窗口外置 `-inf`）：

$$
\text{Logits}[b,s,t,g] \leftarrow
\begin{cases}
\text{Logits}[b,s,t,g], & \text{if } \mathrm{cu\_k\_s}(s) \le t < \mathrm{cu\_k\_e}(s)\\
-\infty, & \text{otherwise}
\end{cases}
$$

（即：窗口内保留原值，窗口外置精确 −∞）

$$
\mathrm{cu\_k\_s}(s)=\max\big(\min(\text{CuSeqLenKS}[s], S_{kv}),\,0\big),\qquad
\mathrm{cu\_k\_e}(s)=\max\big(\min(\text{CuSeqLenKE}[s], S_{kv}),\,\mathrm{cu\_k\_s}(s)\big)
$$

- **规约语义**：**无归约**——逐元素条件投影（projection class）。每个 `(b,s,t,g)` 输出仅由同位置输入与两个按 `s` 变化的标量边界决定，元素间无依赖。
- **dtype 语义**：`Logits` = `T.float`（源码 L159）= **fp32**；`CuSeqLenKS/KE` = `T.int32`（L160）。哨兵 `-T.infinity(dtype)` = fp32 的**精确 −∞**（非有限大数）。无中间 dtype 提升/截断。
- **原地语义**：`Logits` 是 **in-place** 写入（源 prim_func 无 out_idx，wrapper 以 `mutates_args=("Logits",)` 声明）。窗口内**不被读也不被写**（源只在窗口外写）。
- **边界语义**：`cu_k_s` 被 clamp 到 `[0, S_kv]`；`cu_k_e` 以 `cu_k_s` 为下界（`ke < ks` → 退化空窗 → 整行 `-inf`）；空窗（`ke == ks`）合法，全窗（`ks=0, ke=S_kv`）合法（此时全行不被触碰）。窗口外区域输入若为 NaN/±inf，**一律覆写为 −inf**；窗口内 NaN/±inf **原样保留**。`S_kv` 可与 `t` 轴不等（非方形）。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（移植源仓参考实现的窗口掩码部分，见 §0.7）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `Logits`（**输入兼输出**，原地） | `[batch, seq_len, seq_len_kv, kv_group]` | float32 | 打分缓冲；本算子就地掩码（窗口外 −inf） |
| `CuSeqLenKS` | `[seq_len]` | int32 | 每 query 行窗口下界 |
| `CuSeqLenKE` | `[seq_len]` | int32 | 每 query 行窗口上界 |

`Logits` 输出 shape 与源 `source_output_shape` 一致：**`(batch, seq_len, seq_len_kv, kv_group)`**（同布局、同 dtype，原地）。`Logits` 为**显式入参、非 out_idx**（源签名如此；NPU 保持一致）。

> 与前序主 kernel 的契约（§0.2 引用）：主 kernel 只保证写窗口的**块对齐超集**，且**不**保证超集内落在各行 `[ks,ke)` 之外的元素为 −inf——该职责**完全**落于本单元。本单元须覆盖 `[0, ks) ∪ [ke, S_kv)` 全区间（含主 kernel 未写的区域），与源 `clean_logits_` 行为一致。

> **本单元不涉 fp8**：`Logits` 为 fp32、`CuSeqLen*` 为 int32，无 fp8 dtype/量化/GEMM → 前序单元的能力缺口 **CG-2026-0022（npui 无 fp8 全链路）对本单元不适用**，不产生新能力缺口。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句都归入某步骤；括号内为源码行）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式部分 |
|------|------|------|------|-----------------|
| 1 | `with T.Kernel(seq_len, batch, threads=threads) as (bx, by)`（L168） | — | 2D grid（`bx=s`, `by=b`）× 512 线程 | 任务划分（每 `(b,s)` 一个 block） |
| 2 | `tx = T.get_thread_binding()`（L169） | — | 线程 lane 索引 | 线程映射（实现手段） |
| 3 | `cu_k_s = max(min(CuSeqLenKS[bx], seq_len_kv), 0)`；`cu_k_e = max(min(CuSeqLenKE[bx], seq_len_kv), cu_k_s)`（L170–L171） | CuSeqLenKS/KE | 两个 int32 标量 | 窗口边界（含 clamp/退化空窗） |
| 4 | **左区**：`for n_i in serial(ceildiv(cu_k_s, threads)): idx = n_i*threads + tx; if idx < cu_k_s: for g in serial(kv_group): Logits[by,bx,idx,g] = -inf`（L173–L177） | `cu_k_s`, lane | GM `Logits` 左区 = −inf | 补集左段 `[0, ks)` |
| 5 | **右区**：`for n_i in serial(ceildiv(seq_len_kv - cu_k_e, threads)): idx = cu_k_e + n_i*threads + tx; if idx < seq_len_kv: for g in serial(kv_group): Logits[by,bx,idx,g] = -inf`（L178–L182） | `cu_k_e`, lane | GM `Logits` 右区 = −inf | 补集右段 `[ke, S_kv)` |

**数据流与内存访问模式**（源硬件视角，CUDA）：

```
GM[CuSeqLenKS/KE] --标量 load--> 标量 cu_k_s / cu_k_e
GM[Logits] --线程 lane 谓词写(-inf)--> GM[Logits]   （仅窗口外；SMEM 不参与）
```

- **无 SMEM 中转、无 fragment、无矩阵单元**：纯寄存器标量读 + 谓词化 GM 写。
- 每个 `(b,s)` block 覆盖整行 `[0, S_kv)` 的补集（两段连续区间）。

**循环与并行结构**：`T.Kernel(seq_len, batch, threads=512)` —— **2D grid**（bx=s、by=b）× **512 线程**；块内 kv 方向 `T.serial(ceildiv(...))` 分段 + 线程 lane 展开（每线程一步处理一个 kv 位置）；`kv_group` 为最内 `T.serial`。

**host 侧逻辑**：本单元 kernel **无 host 侧预处理/后处理**（wrapper 仅在 `clean_logits=True` 时追加调用本 kernel，见 wrapper L104–L105）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | **线程 lane 展开 + 谓词写**（`tx = get_thread_binding()`；`idx = n_i*threads + tx`；`if idx < cu_k_s`，L169/L174–L176/L180–L182） | 用 512 线程并行覆盖被写区域；谓词裁掉越界 lane | threadIdx 索引 + 每 lane 逐元素谓词 | CUDA threadIdx / warp（512 线程） | **强相关 → 重新设计**（NPU 无线程绑定） |
| O2 | **补集两段分解**（L173–L182：只写 `[0,ks)` 与 `[ke,Skv)`，不碰窗口） | 只写窗口外，**窗口内零读零写** | 集合补集分解 `[0,Skv) \ [ks,ke)` | 无（纯算法层） | **可移植（关键保留）** |
| O3 | **单次 min/max clamp 边界**（L170–L171） | 防越界、统一退化空窗 | `max(min(x,Skv),0)` / 以 ks 为下界 | 无 | **可移植** |
| O4 | **`-inf` 精确哨兵**（L177/L182 `-T.infinity(dtype)`） | 契约要求精确 −∞（下游 softmax/比较语义） | 常量标量写 | 无 | **保留**（契约） |
| O5 | **`kv_group` 串行内循环**（L176/L181） | 覆盖全部 group（`g` 最内） | serial | 无 | **可移植（可并入向量尾轴）** |
| O6 | **2D grid（seq_len × batch）**（L168） | 每个 `(b,s)` 一个 block | grid 划分 | CUDA grid | **等价替换**（NPU ≤2D grid / 1D 展平） |
| O7 | **`threads` 配置项**（L153 默认 512） | 调节 CUDA 并行度 | 编译期线程数 | CUDA blockDim | **舍弃**（NPU 无对应；K9 已移除） |

> 识别不出来 ≠ 不存在。未列出的优化会在迁移中被静默丢弃，导致性能回退无法追溯。本单元无 SMEM tiling / 无流水 / 无向量化 / 无 online 手段——**核心优化是 O2（补集只写）+ O1（线程并行）**。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 1 **计算语义**（窗口外 −inf、窗口内恒等、原地） | 语义 | 无 | — | **保留** | 语义层无条件保留（§0.1）；§8.1 golden 以此为准 |
| 2 **线程 lane 展开 + 谓词写（O1）** | 优化 | CUDA threadIdx/warp（512 线程） | **无对应物** | **重新设计** | 见 §0.6 R1。NPU 无 threadIdx（`docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3：`is_npu=True` 只到 2D block，无线程绑定）→ 逐元素谓词写改 **block 级 tile 向量操作**（`T.copy` 直写 + `T.vcmp`/`T.vselect` 掩码） |
| 3 **补集两段分解（O2，只写不改）** | 算法 | 无 | **有** | **保留（关键）** | 纯集合分解，与硬件无关；NPU 同样只写窗口外（省掉整行读+选择性写），并可进一步做 tile 级跳过（§1.6.1 E1/E2） |
| 4 **单次 min/max clamp（O3）** | 算法 | 无 | 有 | **保留** | `T.max/T.min` 标量形支持；clamp 语义不变 |
| 5 **精确 `-inf` 哨兵（O4）** | 语义/算法 | 无 | 有 | **保留** | `T.vbrc(-T.infinity("float32"), buf)` 物化 −inf 常量缓冲（`docs/Tilelang.language/创建操作/T.infinity.md` §2.4；先例 `_gqa_prefill_fwd_kernel.py` L432 `T.vbrc(value_neg, ub_neg)`） |
| 6 **`kv_group` 串行内循环（O5）** | 优化 | 无 | 有 | **等价替换** | 最内 serial 维并入 tile 尾轴 `[block_N, G_MAX]`（`G_MAX` 为编译期静态上界、`kv_group` 连续 stride 1，拷贝用动态 extent）；`kv_group=1` 即连续 1-D 逻辑 tile（manifest 4 workload 全为 1，用例含 2） |
| 7 **2D grid（O6）** | 优化 | CUDA grid | 有（≤2D / 1D） | **等价替换** | 1D 展平 `T.Kernel(batch*seq_len, is_npu=True)` + `cid` 解码（`T.Kernel.md` §2.4 例 1；先例 `examples/deepseek_v32/sparse_mla_fwd.py` L66 的动态乘积 grid） |
| 8 **`threads` 配置项（O7）** | 实现 | CUDA blockDim | **无对应物** | **舍弃** | NPU 无 blockDim 语义；脚手架 K9 已从 wrapper/工厂签名移除 `threads` |
| 9 **标量 min/max 结果作 DMA 偏移**（前序单元实测陷阱的潜在形态，非源码显式手段） | 实现 | 无（工具链缺陷） | — | **规避（设计约束）** | 前序单元 `_fp8_lightning_indexer_kernel` Stage 3 实测该形态 mis-compile（`MPU address access is invalid`）——本设计**强制所有 `T.copy` 偏移块对齐**（§0.6 R2） |

> 依据：`migration-analysis.md` §5.3 能力映射 + 本仓 `examples/` 同族先例 + 逐条 `docs/Tilelang.language/` 条款。舍弃项理由均见上表「NPU 对应方案/依据」列。

### 0.6 NPU 算法重设计

对 §0.5 中判定为「重新设计」的条目逐项给出 NPU 新算法。

**重设计项 R1: 线程 lane 谓词写（O1）→ block 对齐三段式向量化扫描**

- **源方案**：CUDA `threads=512` + `tx = T.get_thread_binding()`；每 lane 处理一个 kv 位置，用 `if idx < cu_k_s` / `if idx < seq_len_kv` 谓词裁掉越界元素；`kv_group` 内层 `T.serial` 逐 g 标量写。
- **NPU 新算法**：把「逐元素谓词写」改为**块对齐 tile 扫描 + 三态分类**（tile = `[block_N, G_MAX]` 静态 UB tile，拷贝动态 extent）：
  1. **窗口外 tile**（`tile_end ≤ cu_k_s` 或 `tile_start ≥ cu_k_e`）→ 从常量缓冲 `neg_ub`（= −inf）`T.copy` **直接整块写**（免读、免比较）；
  2. **窗口内 tile**（`cu_k_s ≤ tile_start 且 tile_end ≤ cu_k_e`）→ **完全跳过**（不读不写，窗口值恒等保留）；
  3. **边界 tile**（跨窗口端点，每行 ≤2 个）→ `T.copy` 载入 tile，`T.arange` 建 kv 索引 → `T.vcmp`×2 + `T.vand` 建窗口掩码 → `T.vselect(cond, log_ub, neg_ub, log_ub)` 原地选择 → `T.copy` 写回。
  `kv_group` 并入 tile 尾轴（**编译期静态上界** `[block_N, G_MAX]`，见 §3.3/§4.6——动态维不可作 UB buffer 形状，revision 1 阻塞 1），不做内层 serial；拷贝用动态 extent `[0:tail, 0:kv_group]`。
- **语义保持论证**：输出集合与源完全相同——窗口外 tile 写精确 −inf（与源逐元素写同值）；窗口内 tile 不触碰（原地缓冲值恒等保留，等价于源「不写」）；边界 tile 用 `vselect` 在窗口内位置取原值、窗口外位置取 −inf（`vselect` 是精确 selection，无舍入）。**边界语义核对**：空窗（`ke≤ks`）→ 全部 tile 落入「窗口外」→ 整行 −inf（与源一致）；全窗（`ks=0,ke=S_kv`）→ 全部 tile 落入「窗口内」→ 全行不触碰（与源一致）；窗口外 NaN/±inf → 覆写 −inf；窗口内 NaN/±inf → 保留（`vselect` 精确）。机器验证见 §1.6.1（`verify_equiv.py`，逐位 max|d|=0）。

**重设计项 R2: 窗口端点（`cu_k_s`/`cu_k_e`）的 DMA 偏移用法 → 强制块对齐偏移 + 标量比较/向量操作数**

- **源方案**：`idx = n_i*threads + tx` 与 `idx = cu_k_e + n_i*threads + tx`——窗口端点出现在写入地址表达式中（右区起始就是 `cu_k_e`）。
- **NPU 新算法**：**所有 `T.copy` 的偏移一律为块对齐的 `t*block_N`（循环导出量）**，绝不使用 GM 派生的标量（`cu_k_s`/`cu_k_e`）作偏移；窗口标量只出现在：
  - **标量分支条件**（tile 三态分类：`tile_end ≤ cu_k_s` 等）；
   - **向量比较的操作数**（须 **let-bind**：先 `s_off = cu_k_s - t*block_N` 再 `T.vcmp(idx_ub, s_off, cond, "ge")`。运行时标量形先例 `_gqa_prefill_fwd_kernel.py` L544/L551 `K_blk = ...` 后 `T.vcmp(ub_diff, K_blk, ub_cond, "le")`；**内联复合表达式**（`T.vcmp(idx, cu_k_s - k0, ...)`）会触发前端 `<class 'tvm.tir.expr.Sub'> has no attribute buffer`，revision 1 阻塞 3）。
  这是对前序单元实测陷阱（标量 min/max 链作 DMA 偏移 mis-compile）的**结构性规避**——该陷阱的修法（「对齐到块边界的扫描 + 单次 `T.min` clamp」，见其 RETROSPECTIVE）正是本设计的偏移纪律。
- **语义保持论证**：地址计算的改写不改变被写元素集合（块对齐 tile 的并集 ⊇ 补集区间，边界 tile 由掩码精确裁到窗口外），输出与源逐位一致（§1.6.1 机器验证）。

**重设计项 R3: 2D grid → 1D 展平 grid**

- **源方案**：`T.Kernel(seq_len, batch, threads=threads)` 2D grid。
- **NPU 新算法**：1D `T.Kernel(batch * seq_len, is_npu=True) as (cid, _)`，`by = cid // seq_len`（batch）、`bx = cid % seq_len`（query 行）。
- **语义保持论证**：grid 展平是纯任务映射重排，每个逻辑任务仍是「一个 `(b,s)` 行」，输出集合不变。1D 非持久结构与本任务前序单元（`_fp8_lightning_indexer_kernel` 的 1D grid）一致。

**其余（§0.5 保留/等价替换/舍弃项）** 无需重设计，但在 §1–§6 按 NPU 能力落地（UB tile、1D grid、`T.copy`/v-prefix、Developer 自动同步）。

### 0.7 标杆实现

- **源算子路径标注（golden 来源）**：`/home/developer/TileOPs/tileops/kernels/fp8_lightning_indexer.py`（`clean_logits_` L152–L184）。源仓对「窗口掩码」的参考实现位于整体 op 的测试 golden：`examples/TileOPs/tests/ops/test_fp8_lightning_indexer.py`（`FP8LightningIndexerTest.ref_program`，L82–L93：`mask = (ar>=ks)&(ar<ke)` → `logits.masked_fill(~mask, -inf)`；比较器 `_validate_tensor_match` 用 **isfinite 掩码 + 非有限值逐位一致 + 有限域 correlation ≤1e-3**，L41–L65）。
- **迁移 golden 策略**：§8.1 直接移植该窗口掩码语义（`masked_fill(-inf)`），**不复刻** §0.6 的 NPU 三态扫描算法（保证验证独立性）。
- **同族 NPU 先例**：`examples/indexer/indexer_fwd.py` L84–L92（同文件的窗口掩码形态，标量版）；`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` L425–L434/L551–L565（NPU 掩码标准形态：`T.arange` 建索引 + `T.vcmp` + `T.vand` + `T.vselect(-inf)`）；`examples/flash_qla/tilelang_get_warmup_chunks.py` L91–L105（`T.vbrc(运行时标量)` + `T.vcmp` + `T.vselect`）；`examples/deepseek_v32/sparse_mla_fwd_dynamic_shape.py` L263/L416（动态 shape 下的 arange+vcmp 掩码）。

---
## 1. 概述

### 1.1 算子名称

`clean_logits_`（Lightning Indexer 打分缓冲的窗口掩码清理 kernel，迁移自 GPU TileOPs `FP8LightningIndexerOp` 的第 2 个 kernel 工厂）。

### 1.2 功能描述

对前序主 kernel 产出的 fp32 `Logits[b, s, t, g]` **原地**清理：把每条 query 行 `s` 的可见窗口 `[cu_seqlen_ks[s], cu_seqlen_ke[s])` **之外**的元素置为精确 `-inf`，窗口内的元素原样保留。本单元是主 kernel「块对齐超集写」契约的收口步骤。

### 1.3 数学公式

$$
\text{Logits}[b,s,t,g] \leftarrow
\begin{cases}
\text{Logits}[b,s,t,g], & \mathrm{cu\_k\_s}(s) \le t < \mathrm{cu\_k\_e}(s)\\
-\infty, & \text{otherwise}
\end{cases}
\qquad
\begin{aligned}
\mathrm{cu\_k\_s}(s)&=\max(\min(\text{CuSeqLenKS}[s], S_{kv}),0)\\
\mathrm{cu\_k\_e}(s)&=\max(\min(\text{CuSeqLenKE}[s], S_{kv}),\mathrm{cu\_k\_s}(s))
\end{aligned}
$$

**§1.6.1 优化后（采纳 E1–E3）**：等价改写为**补集只写**（窗口内恒等保留、不读不写）：

$$
\text{Logits}[b,s,t,g] \leftarrow -\infty \quad \forall\, t \notin [\mathrm{cu\_k\_s}(s),\mathrm{cu\_k\_e}(s))
\;(\text{window untouched})
$$

### 1.4 算法描述

**迁移决策后的 NPU 侧算法**（基于 §0.5/§0.6；与源算法结构差异逐条注明）：

1. **Grid**：1D 非持久 grid `batch * seq_len`（源为 2D grid × 512 线程；差异来源 §0.6 R3/R1——NPU 无线程绑定，`(b,s)` 对任务展平成一维）。每逻辑 block 处理一条 query 行的全部 `[0, S_kv)`。
2. **窗口边界（块级标量）**：`cu_k_s = max(min(CuSeqLenKS[bx], S_kv), 0)`、`cu_k_e = max(min(CuSeqLenKE[bx], S_kv), cu_k_s)`（源 L170–L171 同义；承接 O3）。
3. **块对齐 tile 扫描**：对 `t = 0 … ceil(S_kv/block_N)-1`，tile = `[t*block_N, t*block_N + tail)`（`tail = min(block_N, S_kv - t*block_N)`，尾块动态 extent 切片）。**所有 DMA 偏移 = `t*block_N`（块对齐）**（§0.6 R2）。
4. **三态分类**（承接 O2 补集只写 + §1.6.1 E1–E3）：
   - 窗口外 tile（`tile_end ≤ cu_k_s` 或 `tile_start ≥ cu_k_e`）→ 常量 `-inf` 直写（免读免比较，§1.6.1 E3）；
   - 窗口内 tile（`cu_k_s ≤ tile_start` 且 `tile_end ≤ cu_k_e`）→ **跳过**（无操作，§1.6.1 E2）；
   - 边界 tile（跨窗口端点，每行 ≤2 个）→ 载入 + 掩码 + `T.vselect` 原地选择 + 写回（§1.6.1 E4：窗宽 ≥ `block_N` 时单边掩码）。
5. **掩码构建**：`T.arange(idx_ub, [1,0])` 建 `[block_N, G_MAX]` 行索引 → **let-bind** `s_off = cu_k_s - t*block_N` / `e_off = cu_k_e - t*block_N`，再 `T.vcmp(idx_ub, s_off, cond_lo, "ge")`、`T.vcmp(idx_ub, e_off, cond_hi, "lt")` → `T.vand(cond_lo, cond_hi, cond)`（先例 `_gqa_prefill_fwd_kernel.py` L872–L874）→ `T.vselect(cond, log_ub, neg_ub, log_ub)`（原地）。全部 v-op 缓冲为 `T.alloc_ub`（revision 1 阻塞 2/3）。
6. **原地写**：`Logits` 是输入兼输出，无额外输出张量；窗口内元素不触碰（承接 O2）。

与源算法的差异来源：① 线程 lane 谓词写 → block tile 三态扫描（§0.6 R1）；② 窗口端点不再作 DMA 偏移（§0.6 R2）；③ 2D grid → 1D（§0.6 R3）；④ 补集只写 + 幂等跳过 + 常量直写（§1.6.1 E1–E3）。

### 1.5 数据流图

```
GM[CuSeqLenKS/KE] ──标量 load──> cu_k_s, cu_k_e (块级标量)
GM[Logits] ──────────────────────────────────────────────┐
   │     窗口外 tile ── T.copy(neg_ub) ──────────────> GM[Logits]  (−inf 直写)
   │     窗口内 tile ── (跳过，不触碰) ───────────────> (无访问)
   └─    边界 tile  ── T.copy ─> log_ub ─vand/vcmp/vselect→ log_ub ─T.copy─> GM[Logits]
                                        ↑
                      idx_ub(T.arange) + cu_k_s/cu_k_e(标量) + neg_ub(−inf 常量)
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先**调研**（1.6.0 调研四问）选算法族；再数学等价优化公式（1.6.1）；再把计算交给向量单元（1.6.2）；最后定向量化轴与核内布局（1.6.3）。本节结论是 §3.1 与 §6 的输入。分析对象 = §1.6.0 选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

算子类别：**窗口 + 逐元素条件投影（masked constant-fill / projection class）**——单步、无规约、无矩阵、无跨元素依赖。按 `algorithm-research.md` §2，属「窗口类」→ **完整调研**（R1 候选表 + R3 对比表 + R4 逐候选评估）。

调研范围：`algorithm-candidates.md`（注意本族无直接命中条目——见 R1 说明）；本仓 `examples/indexer/indexer_fwd.py`、`examples/deepseek_v32/*`、`examples/flash_qla/*`、`pattern-library/elementwise.md`（PL-1.6/PL-1.17/PL-1.22/PL-1.23/PL-1.28/PL-1.29）、`pattern-library/attention.md`（PL-1.11 causal-mask + 算术惩罚掩码 + `-inf`/有限哨兵分析）、`queue.md`（VP-2026-0228 算术掩码乘）；结构分析（§0.3：补集两段连续、无跨元素依赖）。

**R1 等价化简公式候选**（基线候选必须在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|--------|
| 0 | **基线：源算法** | 线程 lane 逐元素谓词写 `[0,ks)∪[ke,Skv)` | — | — | ✅（基线） |
| 1 | **补集分解 + 只写不改**（采纳，§1.6.1 E1） | `out = x` 于窗口；`out = -inf` 于补集（窗口**零读零写**） | 精确等价（窗口内恒等）=源语义；机器逐位 max\|d\|=0 | 窗口内免读免写 | ✅ |
| 2 | **tile 级幂等跳过**（采纳，§1.6.1 E2） | 窗口内 tile 完全无操作 | 精确等价（无操作 = 保持原值） | 窗口区零访问 | ✅ |
| 3 | **窗口外 tile 常量直写**（采纳，§1.6.1 E3） | 窗口外整 tile `T.copy(neg_ub)` | 精确等价（整 tile 均为 −inf） | 免读免 vcmp/vselect | ✅ |
| 4 | **边界单边掩码**（采纳，§1.6.1 E4） | 窗宽 ≥ block_N 时边界 tile 只建单边 cond | 精确等价（另一边 cond 恒真） | 边界 v-op 减半 | ✅ |
| 5 | **整行 load+vselect+store（无条件）** | `out = vselect(keep, x, -inf)` 全行 | 数学等价 | **负向**：整行读+写 + 每 tile vselect | ❌（流量 ≈2×、vselect 高代价值，见 R3/R4） |
| 6 | **有限哨兵 `-1e38` 替代 `-inf`**（PL-1.11 形态） | 窗口外 `x + (−1e38)` / `-1e38` 填 | **非等价**：`isfinite(-1e38)=True` ≠ `isfinite(-inf)=False` | — | ❌（负向，机器诊断 D1：isfin_mismatch=2977） |
| 7 | **算术 0/1 掩码乘**（VP-2026-0228 形态） | `out = x*m + s*(1−m)` | **非等价**：`0·(±inf)=NaN`，无法产出 `−inf` | — | ❌（负向，机器诊断 D2：isfin_mismatch=1119） |
| 8 | **融合进主 kernel** | 主 kernel 内做窗口掩码、省本单元 | 语义可行 | — | ❌（**范围/契约理由**：本单元是独立迁移单元、主 kernel 契约只写块对齐超集；融合会改变主 kernel 契约并重复掩码，见 §0.1/§0.2 契约） |

**R2 在线算法**：**无（不需要）**。结构依据：本算子是**逐元素条件投影**——每个 `(b,s,t,g)` 输出仅依赖同位置输入与两个按 `s` 变化的标量边界，**无跨元素归约、无跨 tile running 统计量**（`algorithm-research.md` §3 R2 判据「逐元素映射无跨元素依赖，单遍即天然在线」）。因此「online 变体」在本语义下退化为**单遍扫描**（本设计即单遍 block 扫描），不存在两遍↔单遍的算法族差异可优化。源算法亦为单遍（§0.4 无 online 手段）。结论：**单遍块扫描即最优扫描形态**。

**R3 复杂度对比**（四口径；`N = batch·seq_len·seq_len_kv·kv_group` 元素，`E = 4 B/elem`；代表 workload = `b1/s8192/h32/d64/skv8192/g1` → `N = 67.1e6`，`N·E = 268 MB`；口径 = HBM 无 L2 复用、`block_N=256`、`G=1`；`w` = 窗口占行比例）：

| 候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度/跨核 |
|------|-------|----------------|---------|-------------|--------------|
| **0 基线（源）** | 每窗口外元素 1 cmp = `(1−w)·N` cmp | 写 `(1−w)·N·E`；读 0（窗口内不读） | Logits 1（写 1 遍）+ 标量边界读 | O(threads)（无 tile） | grid `S·B`；**但逐元素标量 → NPU 无线程承载** |
| **1 补集分解（只写）** | ~0 | 写 `(1−w)·N·E` | 1 | O(1)（直接写） | `S·B`；**需窗口端点作 DMA 偏移（陷阱）** |
| **2 三态扫描（采纳）** | 窗口外 ~0；边界 `≤2·block_N·G` cmp/行 | 写 `(1−w)·N·E` + 边界 tile 读 `≤2·block_N·G·E`/行（≈0.3% 级） | 1 | UB `O(block_N·G)`（§4.5） | `S·B`；块对齐偏移 |
| **3 整行 select（否决）** | `N·(2 cmp + 1 and + 1 sel)` | **读 `N·E` + 写 `N·E` = 2·N·E = 536 MB** | 2（读+写） | UB `O(block_N·G)` | `S·B` |
| **6 有限哨兵（否决）** | `N·(≤5)` | 同 3 | 2 | 同 3 | 契约不符（非算力问题） |
| **7 算术掩码乘（否决）** | `N·(≤4)` | 同 3 | 2 | 同 3 | 契约不符（`0·inf=NaN`） |

口径说明：**FLOPs** 按「比较 = 1 op、选择 = 1 op」折算（无乘加、无超越函数）；**访存量**按 HBM（写 + 读；本算子无中间 GM 往返）；**扫描遍数**按对 GM `Logits` 的读写趟数；**中间缓冲峰值**决定 UB 驻留。**不含**主 kernel 与其它单元的流量。候选 3 的 536 MB 是候选 2 最坏情形（`w=0`，268 MB）的 **2×**，`w>0` 时差距更大（`w=0.5` 时 536 MB vs 134 MB = **4×**）。

**R4 硬件亲和性评估**（逐候选对照检查清单）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐/整除 | 静态边界 | 流水/融合 | 结论 |
|------|-------------|---------|-----------|---------|-----------|------|
| 0 基线 | Vector（逐元素），但**经 CUDA thread lane**→ NPU 无线程映射 | — | — | — | — | ❌ **不可实现**（NPU 无 `get_thread_binding`/`threads`，`T.Kernel.md` §2.3） |
| 1 补集分解（只写） | Vector（`T.copy` 整段） | O(1) | tail 需 `T.min` | 窗口端点**被用作 DMA 偏移** | — | ⚠️ **可写但规避**（前序单元实测该形态 mis-compile；改块对齐后即候选 2） |
| **2 三态扫描** | Vector（`T.copy` 整块 + 少量 v-op）；**匹配佳** | UB ~KB（§4.5）✓ | 偏移 `t*block_N` 块对齐 ✓；`tail=T.min` 单次 clamp ✓ | 循环 trip = `ceildiv(S_kv, block_N)`（动态，`T.serial` 支持） | 无流水需求（纯 store）；无跨核同步 | ✅ **选定** |
| 3 整行 select | Vector（含 `vselect`） | UB ~KB ✓ | 块对齐 ✓ | 动态 trip ✓ | 无 | ✅ 可实现但 ❌ **弃选**：流量 2×–4×（R3）+ `vselect ≈6.5×` 普通 v-op（`CONST-vector-op-cost-910_9362`）→ 主 workload 估算 ≈0.87 ms（流量 505 µs + v-op 链 ≈0.36 ms，revision 1 建议 5 订正）vs 候选 2 ≈0.25 ms（≈3.5×） |
| 5/6/7 | — | — | — | — | — | ❌ 非等价（6/7 机器诊断留痕）或范围外（5 见 R3） |

**调研结论**：选定**候选 1+2+3+4 的组合**（补集只写 + tile 级幂等跳过 + 窗口外常量直写 + 边界单边掩码），结构 = **1D 块扫描 + 三态 tile 分类 + 边界 `vselect`**。与基线的结构差异两点：① 线程 lane 谓词写 → 块级 tile 向量操作（§0.6 R1）；② 补集只写在基线上**进一步**做 tile 级幂等跳过与常量直写（§1.6.1 E2/E3）。**源优化意图承接**：O1（线程并行）→ 块级向量化（本结论）；O2（补集只写）→ 保留并加强（E2/E3）；O3（clamp）→ 保留；O4（−inf 哨兵）→ 保留（`T.vbrc` 物化）；O5（g 串行）→ 并入 tile 尾轴；O6（2D grid）→ 1D 展平；O7（threads）→ 舍弃（NPU 无对应）。负向候选（5/6/7/8）依据见 R1/R3/R4 表与 §1.6.1 机器诊断。

**设计期估算（D-2）**：代表 workload（`b1/s8192/skv8192/g1`，`N=67.1e6`，`N·E=268 MB`，`block_N=256`）：

- **流量项**：补集写 = `(1−w)·268 MB`。**最坏（空窗 `w=0`）= 268 MB**；取核→GM 带宽 `CONST-ascend910-9362-profile`（**9362 单 die ≈1.06 TB/s，2026-10-13**）⇒ **≈253 µs**；典型 `w=0.5` ⇒ ≈127 µs。常数**绑定 9362**，本机为 **Ascend910B3**（`get_aicore_num()=20` 实查）——「**未实测假设 + 估算依据**」：910B3 无独立带宽常数，沿用 9362 值并交 Stage 4 回填。
- **发射项**：bulk 窗口外 tile 为 `T.copy` 整块 store（无逐元素 v-op）；仅边界 tile 有 v-op（每行 ≤2 tile × ≤4 v-op），总量 = `B·S·8` v-op ≈ 6.6e4（代表 shape），按 `CONST-vector-op-cost-910_9362`（普通 v-op 16m≈9.15 µs）折算，单 tile（`256` elem）级 ≈0.15 µs → 总计 ≈10 µs 量级，**远低于流量项**。
- **容量项**：无 GM workspace、无 Cube（L0C 不涉及）、UB 驻留 ~KB（§4.5）⇒ **≈0**。
- **估算下界** = max(253 µs, ~10 µs, 0) ≈ **0.25 ms（流量项主导，空窗最坏）**，引用 `CONST-ascend910-9362-profile` + `CONST-vector-op-cost-910_9362`（均 **9362 绑定、910B3 未重验**，标注未实测假设）。对比候选 3（整行 select）估算 ≈**0.87 ms**（流量 536 MB ≈ 505 µs + v-op 链 ≈0.36 ms；其中 `vselect` 单 op 按 `CONST-vector-op-cost-910_9362`（59.1 µs / 16m-elem）折算 `59.1 µs × (67.1e6/16e6) ≈ 0.25 ms`，2×`vcmp`+1×`vand` 为普通 v-op（9.15 µs/16m）≈0.11 ms——**revision 1 建议 5 订正，原 ≈1 ms 偏高约 4×**）→ 选定的三态扫描约 **3.5× 更优**（量级结论不变：候选 3 仍因流量 2×–4× 被否决）。
  估算下界: ≈0.25 ms（流量 0.25 ms + 发射 ~0.01 ms + 容量 ≈0，引用 CONST-ascend910-9362-profile / CONST-vector-op-cost-910_9362）。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定算法。本算子是**精确投影类**（输出恒为 `{原值, −∞}` 之一，无算术舍入），故所有采纳项的等价性是**逐位精确**（非容差内），机器验证按 `max|d|=0` + isfinite 掩码一致门禁。

| # | 优化项 | 原式（源） | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|-----------|-----------|-----------|---------|
| **E1** | **补集分解 + 只写不改** | `out[t] = cond(t) ? x[t] : −∞`（逐元素投影） | `out[t] = −∞ ∀ t∉[ks,ke)`；**窗口内不触碰** | **精确等价**：投影式在 `cond=真` 支恒返回 `x[t]`（`vselect` 是精确 selection），而原地缓冲窗口内本就等于 `x[t]`；「不写」=「写同值」（对多核/无别名场景下等价——本算子无别名写）。机器验证：`verify_equiv.py` E1-combo 逐位 `max\|d\|=0` | 窗口区**零读零写**：省 `w·N·E` 读 + `w·N·E` 写（主 workload `w=0.5` 时各省 134 MB） |
| **E2** | **tile 级幂等跳过** | 全行逐元素投影 | 窗口内 tile 完全无操作 | **精确等价**：无操作保持缓冲原值；缓冲窗口内值即目标值（原地语义） | 窗口区彻底零访问（UB/带宽 0） |
| **E3** | **窗口外 tile 常量直写** | 逐元素 `cond ? x : −∞` | 整 tile `T.copy(neg_ub)` = −∞ | **精确等价**：窗口外 tile 全部位置 `cond=假` → 全 −∞；常量写与逐元素选择同值 | 免读 tile、免 `vcmp`/`vand`/`vselect`（每 tile 省 ~4 v-op） |
| **E4** | **边界单边掩码** | 每边界 tile 建双边 cond | 窗宽 ≥ `block_N` 时只建含端点那一侧的 cond | **精确等价**：tile 未跨另一端点时该侧 `cond` 恒真，第 2 个 `vselect` 为恒等 | 边界 v-op 由 4 降至 2/行（代表 shape 省 `B·S·2` v-op） |
| — | **−inf 常量物化**（不构成公式变化） | 逐元素 `-T.infinity` | `T.vbrc(-T.infinity("float32"), neg_ub)` 一次 | 逐位同值（fp32 −∞） | 常量一次物化、循环复用 |
| **否决** | 有限哨兵 `-1e38`（PL-1.11 形态） | — | — | **非等价**：契约要求 `isfinite=−∞` 精确；`-1e38` 的 `isfinite=True` → 比较器 `isfinite` 掩码不符（`verify_equiv` D1：`isfin_mismatch=2977`） | 否决（负向，机器诊断留痕） |
| **否决** | 算术 0/1 掩码乘（VP-2026-0228 形态） | — | `x·m + s·(1−m)` | **非等价**：`0·(±inf)=NaN`，无法产出精确 `−∞`（`verify_equiv` D2：`isfin_mismatch=1119`、`nan_mismatch=1122`） | 否决（负向；VP-2026-0228 的适用前提是**乘性契约**，本算子是**加性 −∞ 填充**契约，不适用） |
| **否决** | 整行无条件 `vselect` | — | — | 数学等价但**非优化** | 否决：流量 2×–4×（R3）+ `vselect 6.5×`（R4） |

**优化结论**：采纳 **4 项**（E1 + E2 + E3 + E4），均为精确等价。**优化后公式**（供 §3.1 使用）：

```
for t in 0 .. ceil(S_kv/block_N)-1:
    tile = [t*block_N, min((t+1)*block_N, S_kv))
    if tile ∩ [cu_k_s, cu_k_e) == ∅:      write -inf to tile          # E3
    elif tile ⊆ [cu_k_s, cu_k_e):          no-op                      # E2
    else:                                  tile = vselect(cond, tile, -inf)   # E1 (E4: single edge)
```

**D-1 机器验证结果表**（`verify_equiv.py`，torch CPU；脚本存于算子目录）：

| 项 | 用例 | max\|d\|（fp32 逐位） | isfinite 掩码不符 | NaN 位不符 | 非有限一致 | 结论 |
|----|------|----------------------|-------------------|-----------|-----------|------|
| E1–E4 | 随机窗（S=64,Skv=1024,G=1） | **0** | 0 | 0 | True | **EQUIV_PASS** |
| E1–E4 | 非对齐端点（ks%block≠0, ke%block≠0） | **0** | 0 | 0 | True | **EQUIV_PASS** |
| E1–E4 | 空窗（ke≤ks） | **0** | 0 | 0 | True | **EQUIV_PASS** |
| E1–E4 | 全窗（ks=0,ke=Skv） | **0** | 0 | 0 | True | **EQUIV_PASS** |
| E1–E4 | 反转窗（ke<ks → clamp 空窗） | **0** | 0 | 0 | True | **EQUIV_PASS** |
| E1–E4 | `kv_group=2` | **0** | 0 | 0 | True | **EQUIV_PASS** |
| E1–E4 | `Skv=700` 非 `block_N` 倍数（尾块） | **0** | 0 | 0 | True | **EQUIV_PASS** |
| E1–E4 | 4 seed × block_N∈{64,128,256,512} | **0** | 0 | 0 | True | **EQUIV_PASS** |
| E2 | `skip_inside=False`（对照：跳过与不跳过同值） | **0** | 0 | 0 | True | **EQUIV_PASS** |
| E3 | `constant_store=False`（对照：整块写 vs 逐元素选） | **0** | 0 | 0 | True | **EQUIV_PASS** |
| E4 | `single_edge=False`（对照：单边 vs 双边掩码） | **0** | 0 | 0 | True | **EQUIV_PASS** |
| 角点 | 张量含 `±inf/NaN/−0.0/±1e-45/subnormal/±1e38` | **0** | 0 | 0 | True | **EQUIV_PASS**（窗口内 ±inf/NaN 保留、窗口外覆写 −inf） |
| 诊断(非门禁) | 有限哨兵 `−1e38` | — | **2977** | 0 | False | DIAGNOSTIC（**证明否决必要**） |
| 诊断(非门禁) | 算术掩码乘 `x·m+s·(1−m)` | — | **1119** | **1122** | False | DIAGNOSTIC（**证明否决必要**） |

`ALL_GATE_ITEMS EQUIV_PASS`（全部门禁项逐位精确；2 项非门禁诊断证否决必要）。

> **对照行独立性（revision 1）**：门禁行的左侧 = 候选算法 `algo_tile`（三态块扫描），右侧 = 参考语义 `baseline`（`torch.where(keep, x, -inf)`），二者为**不同实现**。两项诊断候选 `cand_finite_sentinel` / `cand_arithmetic_blend` 均**由窗口掩码独立构造**（不经 `algo_tile` 输出后处理），杜绝 `return <other_fn>(...)` 式自比较/恒等对照（VP-2026-0090「对照行独立性核对」）。订正后 D1/D2 判据数值不变（D1 `isfin=2977`、D2 `isfin=1119/nan=1122`），但证据链独立成立。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案（§1.4/§3.3）中**全部**循环与标量计算点：

| # | 计算点 | 原实现形态（源） | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------------|-------------|---------|------------------------------|
| 1 | 窗口外元素写 −inf | `for n_i in serial(...)` + 线程 lane 逐元素谓词写（L173–L177/L178–L182） | `T.copy(neg_ub, Logits[..., k0:k0+tail, :])` **整块常量 store**（E3；隐式向量化） | ✅ | — （`docs/Tilelang.language/内存操作/T.copy.md` §2.1；先例 `_gqa_prefill_fwd_kernel.py` 整 tile 写） |
| 2 | 窗口内元素保留 | 源不写（已向量化语义） | 三态分类**跳过**（E2） | ✅ | — |
| 3 | 边界 tile 掩码（kv 索引） | 无（源用线程 lane 隐含逐元素定位） | `T.arange(idx_ub, [1,0])` 建 `[block_N, G_MAX]` 行索引（一次，静态形状 buffer） | ✅ | — （`docs/.../创建操作/T.arange.md` §2.2.1 int32 ✓、§2.4；先例 `_gqa_prefill_fwd_kernel.py` L430–L431、`sparse_mla_fwd_dynamic_shape.py` L416） |
| 4 | 边界 tile 条件比较 | `if idx < cu_k_s`（逐元素谓词） | **let-bind** `s_off = cu_k_s−k0` / `e_off = cu_k_e−k0` 后 `T.vcmp(idx_ub, s_off, cond_lo, "ge")` / `T.vcmp(idx_ub, e_off, cond_hi, "lt")`（运行时标量第二操作数须 let-bound） | ✅ | — （`docs/.../比较操作/T.vcmp.md` §2.1/§2.2.1 int32 ✓；先例 `_gqa_prefill_fwd_kernel.py` L544/L551 先 `K_blk = ...` 再 `vcmp`；内联复合标量报 `<Sub> has no attribute buffer`，revision 1 阻塞 3） |
| 5 | 边界 tile 掩码合并 | 无（谓词天然合取） | `T.vand(cond_lo, cond_hi, cond)`（bool 片上） | ✅ | — （`docs/.../逻辑操作/T.vand.md` §2.3 广播；bool 片上合法 §2.4；先例 `_gqa_prefill_fwd_kernel.py` L874、`sparse_mla_fwd_dynamic_shape.py` L268） |
| 6 | 边界 tile 条件选择 | `if idx < cu_k_s: Logits[...] = -inf`（逐元素分支写） | `T.vselect(cond, log_ub, neg_ub, log_ub)`（原地选择） | ✅ | — （`docs/.../条件操作/T.vselect.md` §2.2.1 fp32 ✓；先例 `_elu_kernel.py` L174、`_gqa_prefill_fwd_kernel.py` L565） |
| 7 | −inf 常量 | 逐元素 `-T.infinity(dtype)` | `T.vbrc(-T.infinity("float32"), neg_ub)` 一次物化 | ✅ | — （`docs/.../创建操作/T.infinity.md` §2.4 fp32 ✓ + `docs/.../shape操作/T.vbrc.md` §2.2 标量形；先例 `_gqa_prefill_fwd_kernel.py` L432） |
| 8 | **窗口边界 clamp**（`cu_k_s`/`cu_k_e`） | 源 `T.max/T.min` 标量（L170–L171） | 无逐元素等价 API（且不应向量化） | ❌ | **block 级标量调度计算**：每行仅 2 次 min + 2 次 max 得到 2 个调度标量，非逐元素热点；向量化需先物化 UB 数组，收益为负。编译期/块级标量即可（承接源 O3） |
| 9 | **tile 三态分类** | 无（源无线程无关的分类） | 无逐元素等价（且不应向量化） | ❌ | **block 级标量分支**：每 tile ≤3 次标量比较（调度），非逐元素热点；这是 §0.6 R2 规避 DMA 偏移陷阱的关键（窗口标量只作标量比较） |
| 10 | **逻辑任务解码**（`cid → (by, bx)`） | 源 2D grid 自带 | 无 | ❌ | **block 级任务映射**：每核 1 次整数除/模，非逐元素热点（`T.Kernel.md` §2.4 例 1 同形） |
| 11 | **kv 循环 trip count**（`ceildiv(S_kv, block_N)`） | 源 `T.serial(ceildiv(..., threads))` | 无 | ❌ | **依赖动态 shape 的循环边界**：`seq_len_kv` 为 `T.dynamic`（保接口契约），trip count 运行时确定；`T.serial` 动态上界在 NPU 先例中支持（`examples/elementwise/vec_add_2d_dynamic_shape.py` L70） |
| 12 | **尾块 extent**（`tail = T.min(block_N, S_kv − k0)`） | 源 `if idx < n` 谓词 | 无逐元素等价（且不应向量化） | ❌ | **依赖动态 shape 的边界处理**：单次 `T.min` clamp（PL-1.22 形态，`vec_add_2d_dynamic_shape.py` L86–L93 同形），v-op 作用于整 tile、只拷出 `[0:tail]`；不可静态化（`S_kv` 运行时值） |

**向量化结论**：逐元素计算点（#1–#7）**全部向量化**（整块 `T.copy` / v-prefix / `T.vselect`），替代 API 均有 `docs/` + `examples/` 佐证；保留 5 处非逐元素标量/循环（#8 block 级边界 clamp、#9 block 级 tile 分类、#10 block 任务映射、#11 动态 trip、#12 动态尾块），逐项理由见上表，**无「实现简单」类理由**。§6 循环结构中**无逐元素标量循环**残留。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

I/O layout 是**契约**（`Logits[B,S,Skv,G]`，尾轴 `kv_group` 连续、stride 1），核内布局是**设计变量**。算子非 batch 维 ≥2（S/Skv/G），逐候选枚举：

| # | 核内布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 采纳 |
|---|-------------|---------|------------|--------------|------|
| 1 | **tile `[block_N, G_MAX]`（kv 行 × group，G_MAX 为编译期静态上界），处理整条 `(b,s)` 行** | **`(kv, G)` 合并最内维**（G 连续，stride 1） | 无（原位读写，`T.copy` 保形；拷贝 extent 用动态 `[0:tail, 0:kv_group]`） | 基线；`block_N` G=1 时 fp32 尾轴 `block_N·4 B`（256×4=1024 B ≥32B ✓）；G=2 时整块连续 2 KB（连续 DMA）。**静态上界是硬约束**：动态维不可作 UB buffer 形状（`memref.alloc` 维度数校验失败，revision 1 阻塞 1） | ✅ **主选** |
| 2 | 逐 `g` 切片 `[block_N]`（stride G） | kv（stride = G） | 无 | `G>1` 时每行 stride `G·4 B`（8 B < 32B）→ 非连续 DMA，MTE3 分段退化（`PL-1.14-mte3-strided-ws`：跨步写段级延迟不摊销） | ❌（主选否决）：G=1 与 #1 等价；G>1 引入跨步写退化。**仅作阻塞 1 的退路**（1-D 静态 tile `[block_N]` + `for g in T.serial(kv_group)`——`T.serial` 动态上界合法，编译探针已验证；G>1 时为跨步写） |
| 3 | host permute `Logits→[B,G,S,Skv]` | kv | host 重排（~百 µs，`PL-1.3-host-permute`） | repack 代价 ≫ 收益，且**违反 `ascend-constraints.md` §4**（host 不得改输入物理排布） | ❌ |
| 4 | 贯穿整张 `Logits` 的 1-D 扁平化 `[B·S·Skv·G]` | 全局线性索引 | 无 | 窗口按 `s` 变化 → 扁平索引需 `// G` 还原 kv 位置（**整数除法逐元素**，退化）；且 tile 跨 `s` 边界时分类失效 | ❌：窗口语义按 `s` 行定义，全局扁平化破坏 tile 三态分类 |

**布局决策结论**：选定 **#1**：核内 tile = `[block_N, G_MAX]`（kv 行 × group 连续块，`G_MAX` 为**编译期静态上界**），**向量化轴 = 合并的 `(kv, G)` 最内连续维**，repack 路径 = **无**（原位 `T.copy` 读写，动态 extent `[0:tail, 0:kv_group]`）。该结论同步落入 §3.3 伪代码与 §6 循环结构：`neg_ub`/`idx_ub`/`log_ub`/`cond_*` 的缓冲形状均为 `[block_N, G_MAX]`（静态），最内维与本节选定轴一致，**且 v-op 操作数为全静态形状 buffer**（`TRAP-expert-v-operands`：「v-op 切片仅接受静态边界」）。轴质量评分：`block_N=256` fp32（向量宽 4）→ 整除 ✓、尾 lane 浪费 0%；`G=1` 全连续；`G=2` 整块连续 2 KB。**弃选理由均为量化/契约性**（#2 跨步写退化【仅阻塞 1 退路】；#3 host permute 净亏且违约束；#4 破坏按行窗口语义），非「未考虑」。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（非持久，单 kernel 内 block 级向量操作，编译器自动放置 UB / 自动同步）。

### 2.2 选型理由

- 算子为**纯 Vector**（无矩阵乘、无归约、无 Cube）：仅需 UB tile + `T.copy` + v-prefix。Developer 模式对本形态有大量可运行先例：`examples/elementwise/vec_add_2d_dynamic_shape.py`（动态 shape + 1D grid + UB tile + `T.copy`）；`examples/elu/_make_elu_kernel/_make_elu_kernel.py`（`T.vcmp` + `T.vselect` 掩码链）；`examples/deepseek_v32/sparse_mla_fwd_dynamic_shape.py`（动态 shape + `T.arange`/`T.vcmp`/`T.vand` 掩码）。
- 脚手架 wrapper 已设 `ascend_mode = "Developer"`，且 `op_requirements.programming_mode = developer`。
- 相对 Expert：免手写 Cube/Vector 双 Scope（`T.Scope`）+ `sync_block_set/wait`；本算子无 Cube 段、无跨引擎协同，Expert 无收益。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | **`T.alloc_ub`**（UB tile，Developer 下 v-op/`T.copy` 的片上载体；v-op 缓冲须为 `alloc_ub`——`alloc_shared` 下运行时标量 `vcmp/vand/vselect` 链编译失败 `cannot find variable clo`，revision 1 阻塞 2 / `TRAP-vbrc-scalar-shared`）；buffer 形状全为**编译期静态**（`[block_N, G_MAX]`，动态维不可作 buffer 维）；无 `alloc_fragment`/`alloc_L1`/`alloc_L0C`（无 Cube） |
| 计算方式 | `T.copy`（整块常量写/搬入搬出）+ v-prefix（`T.vbrc`/`T.arange`/`T.vcmp`/`T.vand`/`T.vselect`） |
| 同步 | **自动同步**（Developer 由编译器插入 GM↔UB 依赖同步），无手写 flag |
| 流水 | 无（`T.serial` 单遍；纯 store 无计算可重叠） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 **优化后公式**。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `k0 = t·block_N`；`tail = min(block_N, S_kv − k0)` | 块对齐 kv 游标 + 尾块 extent |
| 2 | `cu_k_s = max(min(CuSeqLenKS[s], S_kv), 0)`；`cu_k_e = max(min(CuSeqLenKE[s], S_kv), cu_k_s)` | 窗口边界（块级标量） |
| 3 | `OUT(tile) = −∞`，当 `k0+tail ≤ cu_k_s` 或 `k0 ≥ cu_k_e` | 窗口外整 tile（E3） |
| 4 | `OUT(tile) = x(tile)`（不触碰），当 `cu_k_s ≤ k0` 且 `k0+tail ≤ cu_k_e` | 窗口内整 tile（E2） |
| 5 | `cond = (idx ≥ cu_k_s − k0) ∧ (idx < cu_k_e − k0)`，`idx_i = i` | 边界 tile 窗口掩码 |
| 6 | `OUT(tile)[i] = cond_i ? x(tile)_i : −∞` | 边界 tile 选择（E1/E4） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| grid | 逻辑任务 | `T.Kernel(batch * seq_len, is_npu=True)` | 1D 非持久 grid；`cid → (by=batch, bx=s)` | Developer |
| 2 | 窗口边界 | `T.max(T.min(CuSeqLenKS[bx], seq_len_kv), 0)` | 标量形 `T.min/T.max` | Developer |
| 3 | −inf 常量 | `T.vbrc(T.float32(-T.infinity("float32")), neg_ub)` | 标量→`[block_N, G_MAX]` 广播（一次；`neg_ub` 为 `T.alloc_ub`） | Developer |
| 5 | kv 索引 | `T.arange(idx_ub, [1, 0], 0)` | `[block_N, G_MAX]` int32，`idx[i,j]=i`（一次） | Developer |
| 3 | 窗口外整块写 | `T.copy(neg_ub[0:tail, 0:kv_group], Logits[by, bx, k0:k0+tail, 0:kv_group])` | 4-D 切片（**切片语法**，非 `size=`；UB 侧静态 buffer + 动态 extent） | Developer |
| 5 | 掩码比较 | `s_off = cu_k_s − k0`；`e_off = cu_k_e − k0`；`T.vcmp(idx_ub, s_off, cond_lo, "ge")` / `T.vcmp(idx_ub, e_off, cond_hi, "lt")` | **运行时标量须 let-bind**（内联复合式报错） | Developer |
| 5 | 掩码合并 | `T.vand(cond_lo, cond_hi, cond_lo)` | bool 原地（先例 gqa L874） | Developer |
| 6 | 载入边界 tile | `T.copy(Logits[by, bx, k0:k0+tail, 0:kv_group], log_ub[0:tail, 0:kv_group])` | 4-D 切片 | Developer |
| 6 | 选择 | `T.vselect(cond_lo, log_ub, neg_ub, log_ub)` | 原地（dest=srcA，先例 elu L174） | Developer |
| 6 | 写回边界 tile | `T.copy(log_ub[0:tail, 0:kv_group], Logits[by, bx, k0:k0+tail, 0:kv_group])` | 4-D 切片 | Developer |
| 扫描 | kv 块循环 | `T.serial(T.ceildiv(seq_len_kv, block_N))` | 动态上界（先例 vec_add_2d_dynamic_shape L70） | Developer |

### 3.3 计算伪代码

> 依 `design-template.md` 提示：`T.Tensor` 的 dtype 直接作第二位置参数传入（不用 `dtype=`）。

```python
@tilelang.jit(target="npuir")
def clean_logits_(block_N: int = 256, G_MAX: int = 2):
    dtype = "float32"
    index_dtype = "int32"

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    kv_group = T.dynamic("kv_group")          # 仅用于 GM 形状与拷贝 extent（动态）

    @T.prim_func
    def clean_logits_kernel(
        Logits: T.Tensor([batch, seq_len, seq_len_kv, kv_group], dtype),   # 原地
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),
    ):
        with T.Kernel(batch * seq_len, is_npu=True) as (cid, _):
            by = cid // seq_len              # batch
            bx = cid % seq_len               # query 行 s

            # 片上 buffer：编译期静态形状 [block_N, G_MAX]（阻塞 1）；v-op 载体 = alloc_ub（阻塞 2）
            neg_ub = T.alloc_ub([block_N, G_MAX], dtype)
            idx_ub = T.alloc_ub([block_N, G_MAX], index_dtype)
            cond_lo = T.alloc_ub([block_N, G_MAX], "bool")
            cond_hi = T.alloc_ub([block_N, G_MAX], "bool")
            log_ub = T.alloc_ub([block_N, G_MAX], dtype)

            # 常量/索引一次物化（承接 O4/E1）
            T.vbrc(T.float32(-T.infinity("float32")), neg_ub)
            T.arange(idx_ub, [1, 0], 0)      # idx_ub[i, j] = i

            # 窗口边界（块级标量；承接 O3）
            cu_k_s = T.max(T.min(CuSeqLenKS[bx], seq_len_kv), 0)
            cu_k_e = T.max(T.min(CuSeqLenKE[bx], seq_len_kv), cu_k_s)

            n_kv = T.ceildiv(seq_len_kv, block_N)
            for t in T.serial(n_kv):
                k0 = t * block_N                          # 块对齐偏移（§0.6 R2）
                tail = T.min(block_N, seq_len_kv - k0)
                # 三态分类（§1.6.1 E2/E3）
                if (k0 + tail <= cu_k_s) or (k0 >= cu_k_e):
                    # 窗口外 → -inf 直写（免读免比较）；静态 buffer + 动态 extent 切片
                    T.copy(neg_ub[0:tail, 0:kv_group], Logits[by, bx, k0:k0 + tail, 0:kv_group])
                elif (cu_k_s <= k0) and (k0 + tail <= cu_k_e):
                    pass                                   # 窗口内 → 幂等跳过
                else:
                    # 边界：搬入 + 掩码 + 选择 + 写回
                    T.copy(Logits[by, bx, k0:k0 + tail, 0:kv_group], log_ub[0:tail, 0:kv_group])
                    s_off = cu_k_s - k0          # let-bound 标量（阻塞 3）
                    e_off = cu_k_e - k0          # let-bound 标量（阻塞 3）
                    T.vcmp(idx_ub, s_off, cond_lo, "ge")
                    T.vcmp(idx_ub, e_off, cond_hi, "lt")
                    T.vand(cond_lo, cond_hi, cond_lo)
                    T.vselect(cond_lo, log_ub, neg_ub, log_ub)
                    T.copy(log_ub[0:tail, 0:kv_group], Logits[by, bx, k0:k0 + tail, 0:kv_group])

    return clean_logits_kernel
```
> **接口契约**：工厂签名 `clean_logits_(block_N=256, G_MAX=2)`（**无 `threads`**，K9；**无 shape 参数**——`batch/seq_len/seq_len_kv/kv_group` 经 `T.dynamic` 从张量推断；`G_MAX` 是编译期静态上界、非 shape 参数，默认 2 保证 `clean_logits_()` 无参可调用）；`prim_func` 3 个张量参数顺序与源一致；`Logits` 显式入参（原地、非 out_idx）。wrapper 调用 `clean_logits_()(Logits, CuSeqLenKS, CuSeqLenKE)` 与契约一致。

> **注意**：`or`/`and` 在 TVMScript 中的运行时条件写法已由 **D-3 设计期编译探针**（`history_version/design_probe_forms.py` form `ab`）实证可编译，非仅「以 Stage 3 首编为准」；仍**不得**在 Python `if` 分支内赋值后于分支外引用（TVMScript 作用域规则，见前序单元 RETROSPECTIVE 与 `TRAP-tvm-parser-rules`）。

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.alloc_ub(static_shape, dtype)` | `docs/.../内存操作/T.alloc_ub.md`（**注**：该文档标 `[Expert mode]` 与实测不符——Developer 模式亦可用，见 `queue.md` VP-2026-0198）；`_gqa_prefill_fwd_kernel.py` L421–L426（全部 v-op 缓冲用 `T.alloc_ub`） | 文档+先例（**v-op 载体**；`T.alloc_shared` 承载运行时标量 `vcmp/vselect` 链编译失败——D-3 探针 form `c1` / `TRAP-vbrc-scalar-shared`） |
| `T.copy`（4-D 切片读写、动态 extent；静态 UB buffer） | `docs/.../内存操作/T.copy.md` §2.1/§2.2.2；`examples/elementwise/vec_add_2d_dynamic_shape.py` L91–L93 | 文档+先例（**用切片语法，非 `size=`**——前序单元实测 ≥3-D `size=` 前置补 1 错区）+ **D-3 探针 form `ab` 编译通过**（静态 `[256,2]` buffer + 动态 extent `[0:tail, 0:kv_group]`） |
| `T.dynamic` + 动态 grid + 动态 `T.serial` | `examples/deepseek_v32/sparse_mla_fwd.py` L66（动态乘积 grid）；`vec_add_2d_dynamic_shape.py` L70 | 先例（NPU） |
| `T.arange(dst, [1,0], 0)` | `docs/.../创建操作/T.arange.md` §2.2.1 int32 ✓、§2.4；`_gqa_prefill_fwd_kernel.py` L430–L431 | 文档+先例（静态形状 buffer） |
| `T.vcmp(A, scalar, C, "ge"/"lt")`（**scalar 须 let-bound**） | `docs/.../比较操作/T.vcmp.md` §2.1/§2.2.1 int32 ✓；`_gqa_prefill_fwd_kernel.py` L544/L551（先 `K_blk = ...` 再 `vcmp`） | 文档+先例 + **D-3 探针**（form `ab` let-bound 通过；form `c2` 内联复合式失败 `<Sub> has no attribute buffer`） |
| `T.vand(A, B, C)`（bool 片上） | `docs/.../逻辑操作/T.vand.md` §2.3/§2.4；`_gqa_prefill_fwd_kernel.py` L874、`sparse_mla_fwd_dynamic_shape.py` L268 | 文档+先例 |
| `T.vselect(cond, A, B, A)`（原地） | `docs/.../条件操作/T.vselect.md` §2.2.1 fp32 ✓、§2.4；`_elu_kernel.py` L174 | 文档+先例 |
| `T.vbrc(scalar, dst)`（运行时标量广播） | `docs/.../shape操作/T.vbrc.md` §2.2 标量形；`flash_qla/tilelang_get_warmup_chunks.py` L91（运行时标量） | 文档+先例（`dst` 为 UB） |
| `T.infinity("float32")` | `docs/.../创建操作/T.infinity.md` §2.2.1 fp32 ✓、§2.4 | 文档+先例 |
| `T.Kernel(<expr>, is_npu=True)` / `T.serial` / `T.ceildiv` / `T.min` / `T.max` | `docs/.../编译器提示操作/T.Kernel.md` §2.3/§2.4；`数学操作/T.ceildiv.md`；`比较操作/T.min.md`/`比较操作/T.max.md` | 文档+先例 |

> **D-3 设计期编译探针**（`history_version/design_probe_forms.py`，本机同工具链 `tilelang 0.1.2+c144c4b2` / `target="npuir"`，2026-10-15）：`PROBE_ALL_AS_EXPECTED`——form `ab`（静态 `[256,2]` `alloc_ub` + 全 v-op 掩码链 + let-bound 标量）**编译通过**；form `c1`（同链用 `T.alloc_shared`）**失败**；form `c2`（内联复合标量作 `vcmp` 操作数）**失败**。三处形态即 revision 1 的三个阻塞级修复的落地形态。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **否** | 用 **1D** grid（`batch*seq_len`），未触 2D 上限，更无三维 |
| GPU 专用 API（`get_thread_binding`/`threads`） | **是** | §0.5 条目 2/8 处置（重新设计/舍弃）；K9 已从签名移除 `threads` |
| FP8 dtype / fp8 GEMM | **否** | 本单元只处理 fp32 `Logits` + int32 `CuSeqLen*`；CG-2026-0022 不适用（§0.2） |
| GEMM 非整除 | **否** | 无 GEMM |
| L0C 溢出 | **否** | 无 Cube / 无 L0C |
| 分核策略缺失/不适配 | **是** | 见 §5.5（逻辑核 `batch·seq_len` 动态；物理核实查 40；无法整数倍对齐兜底 + 残留负载不均记录） |
| Host 侧改输入内容（`ascend-constraints.md` §4） | **否** | kernel 内完成全部清理；host 无预处理/后处理 |
| 标量 min/max 结果作 DMA 偏移（工具链陷阱） | **是（已规避）** | §0.6 R2：所有 `T.copy` 偏移块对齐；窗口标量只作标量比较/向量操作数 |
| **动态维作 UB/shared buffer 形状**（`memref.alloc` 维度数校验失败） | **是（已规避）** | 尾轴用**编译期静态上界** `G_MAX`：buffer `[block_N, G_MAX]` + 动态 extent 拷贝（revision 1 阻塞 1；`T.copy.md` §2.2.2「UB 侧缓冲 shape 须为静态」） |
| **v-op 缓冲用 `T.alloc_shared`**（运行时标量 `vcmp/vselect` 链 `cannot find variable`） | **是（已规避）** | 全部 v-op 缓冲改 `T.alloc_ub`（revision 1 阻塞 2；先例 `_gqa_prefill_fwd_kernel.py` L421–L426；`TRAP-vbrc-scalar-shared`） |
| **`T.vcmp` 第二操作数为内联复合标量**（`<Sub> has no attribute buffer`） | **是（已规避）** | 标量 let-bind 后再传入（revision 1 阻塞 3；`TRAP-expert-v-operands`：vcmp 标量操作数拒绝 `tir.Cast`/复合式） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 维度 | 2D `T.Kernel(seq_len, batch, threads=512)` | 1D `T.Kernel(batch*seq_len, is_npu=True)` | §0.6 R3 |
| 线程模型 | 512 线程 + `get_thread_binding` + 逐元素谓词写 | 无线程；块级 tile 扫描 + 三态分类 | §0.6 R1 |
| 写地址表达式 | `idx = n_i*threads + tx` / `cu_k_e + n_i*threads + tx`（含窗口端点） | `k0 = t*block_N`（块对齐；窗口端点仅作比较） | §0.6 R2 |
| 掩码应用 | 逐元素 `if idx < n: Logits[...] = -inf` | 整块常量写 + 边界 `vselect` | §0.6 R1 / §1.6.1 |
| `kv_group` | 最内 `T.serial(kv_group)` | 并入 tile 尾轴 `[block_N, G_MAX]`（静态上界；拷贝动态 extent `0:kv_group`） | §0.5 条目 6 / §0.6 R1 |
| 流水 | 无（源亦无） | 无 | — |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/indexer/indexer_fwd.py` | **极高**（同族窗口掩码） | L84–L92 窗口掩码形态（`k_idx < ks or k_idx >= ke → -inf`，源同语义）；L51/L64 动态形状 + 1D grid |
| `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py` | **极高**（NPU 掩码标准形态） | L430 索引 arange、L432 −inf vbrc、L551/L872 `vcmp(标量)`、L874 `vand`、L565 `vselect`（与本设计 §3.3 同形） |
| `examples/deepseek_v32/sparse_mla_fwd_dynamic_shape.py` | 高 | L66 动态乘积 grid、L263/L268/L416 `vcmp`/`vand`/`arange` 掩码（动态 shape） |
| `examples/elementwise/vec_add_2d_dynamic_shape.py` | 高（动态 shape 纯 Vector 模板） | 1D grid + 动态 `T.serial` + `T.min` 尾块 extent + 2D 切片 `T.copy` |
| `examples/flash_qla/tilelang_get_warmup_chunks.py` | 中 | L91–L105 运行时标量 `T.vbrc` + `T.vcmp` + `T.vselect` |
| `examples/elu/_make_elu_kernel/_make_elu_kernel.py` | 中 | `T.vcmp` + `T.vselect` 原地形态与 UB 预算范式 |

---
## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `Logits`（**输入兼输出，原地**） | `[batch, seq_len, seq_len_kv, kv_group]` | float32 | 打分缓冲；就地掩码 |
| `CuSeqLenKS` | `[seq_len]` | int32 | 每 query 行窗口下界 |
| `CuSeqLenKE` | `[seq_len]` | int32 | 每 query 行窗口上界 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `Logits` | `[batch, seq_len, seq_len_kv, kv_group]` | float32 | **原地输出**（同输入，无独立输出张量） |

### 4.3 中间缓冲区（主选 `block_N=256`，**静态上界 `G_MAX=2`**）

> 全部缓冲用 **`T.alloc_ub`**（v-op 载体），形状为**编译期静态** `[block_N, G_MAX]`；运行时 `kv_group ≤ G_MAX`，拷贝用动态 extent `[0:tail, 0:kv_group]`（revision 1 阻塞 1/2）。

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `neg_ub` | `[block_N, G_MAX]` | float32 | UB | −inf 常量（一次物化，全 tile 复用） |
| `idx_ub` | `[block_N, G_MAX]` | int32 | UB | kv 行索引（`arange`，一次物化） |
| `cond_lo` | `[block_N, G_MAX]` | bool | UB | 掩码（`vcmp`→`vand` 原地；仅边界 tile 用） |
| `cond_hi` | `[block_N, G_MAX]` | bool | UB | 掩码（`vcmp`；仅边界 tile 用） |
| `log_ub` | `[block_N, G_MAX]` | float32 | UB | 边界 tile 数据（搬入→`vselect` 原地→搬出） |

### 4.4 内存搬运路径

```
GM[CuSeqLenKS/KE] --标量 load--> cu_k_s, cu_k_e (块级标量)
GM[Logits] --窗口外 tile: T.copy(neg_ub)--> GM[Logits]        (整块 -inf 直写，无读取)
GM[Logits] --窗口内 tile: (跳过，无访问)--> (无)
GM[Logits] --边界 tile: T.copy--> UB log_ub --vcmp/vand/vselect--> log_ub --T.copy--> GM[Logits]
```

**关键点**：全程 **GM ↔ UB 二级**；无 L1/L0（无 Cube）；无 GM workspace 往返；窗口内 tile 与 GM 零交互（E2）。搬入/搬出仅发生在 ≤2 个边界 tile/行。

### 4.5 UB 内存预算（`block_N=256`，**静态 `G_MAX=2`**）

> 缓冲形状为**编译期静态** `[block_N, G_MAX]`，UB 占用与运行时 `kv_group`（1 或 2）无关，恒按下表计（revision 1 阻塞 1 的代价：G=1 workload 也按 `G_MAX=2` 分配，仍 ≪ 上限）。

**UB**

| Buffer | Shape | dtype | 大小 |
|--------|-------|-------|------|
| neg_ub | (256, 2) | float32 | 2048 B |
| idx_ub | (256, 2) | int32 | 2048 B |
| cond_lo | (256, 2) | bool | 512 B |
| cond_hi | (256, 2) | bool | 512 B |
| log_ub | (256, 2) | float32 | 2048 B |
| 合计（手工） | | | 7168 B |

- **×auto-multi-buffer 膨胀**（`CONST-capacity-910B2C`：默认 auto-multi-buffer 开 ~1.7×；`auto-multi-buffer=false` ×1.10–1.12；**本机 910B3 未重验**——未实测假设）⇒ 预算 ≤ 7168 B × 1.7 ≈ **12.2 KB** ≪ 192 KB ✅。
- **L0C**：不涉及（无 Cube）。**L1**：不涉及（纯 Vector）。
- 若后续 workload 的 `kv_group > 2`，须同步增大 `G_MAX`（每 +1 组 ≈ +3.5 KB 手工预算），仍 ≪ 上限。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| `batch` | `T.dynamic("batch")` | manifest 均为 1 |
| `seq_len` | `T.dynamic("seq_len")` | 1024 / 4096 / 8192 |
| `seq_len_kv` | `T.dynamic("seq_len_kv")` | 1024 / 8192 |
| `kv_group` | `T.dynamic("kv_group")` | 1（manifest 全为 1；测试含 2）；**仅用于 GM 形状与拷贝 extent，不作 buffer 维** |

`block_N` 与 `G_MAX` 为编译期常量（工厂参数，默认 256 / 2）。GM 张量形状全部动态——**保接口契约**（wrapper 以 `clean_logits_()` 无参调用，形状经张量推断；与源 `T.dynamic` 一致）。**片上 buffer 形状 = 编译期静态 `[block_N, G_MAX]`**（动态维不可作 UB buffer 维，revision 1 阻塞 1）。

**`G_MAX` 取值来源与越界行为**：`G_MAX = 2` 覆盖全部已知 workload（manifest `attention_indexing.yaml` 四 workload 的 `kv_group` 全为 1；精度用例含 `kv_group=2`）。**契约**：运行时必须满足 `kv_group ≤ G_MAX`。`G_MAX` 为工厂参数（默认 2，保证 `clean_logits_()` 无参可调用），上游若引入 `kv_group > 2` 的 workload，须在集成层以 `clean_logits_(G_MAX=<新上界>)` 显式提高（UB 增量极小，见 §4.5）。**越界行为**：`kv_group > G_MAX` 时动态 extent 拷贝 `[0:tail, 0:kv_group]` 会越出静态 buffer 界 → 未定义（编译期无法静态捕获，因形状动态）。**缓解**：Stage 3 建议在 wrapper 侧加 `assert kv_group <= G_MAX`（host 元数据，零 kernel 代价），列入 §9.1 风险 9。

### 4.7 JIT 配置

```python
@tilelang.jit(target="npuir")   # 无 out_idx（Logits 为原地显式入参）；无 pass_configs（无超越函数）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**纯 Vector**（逐元素条件投影 / window mask）。判定依据：无矩阵乘、无归约、无跨元素依赖；仅需 UB tile + `T.copy` + v-prefix。

### 5.2 Block 划分

```python
block_N = 256                       # kv 方向 tile（含 G 连续尾轴）
M  = 8192                           # 逻辑任务数 = batch*seq_len（full 代表 shape；smoke=1024）
bm = 1                              # 每个逻辑 block 处理一条 query 行
num_logical = ceil(M / bm) = 8192   # 逻辑核数 = batch*seq_len
```

**选择理由**：`block_N` 是该算子唯一的 tile 旋钮。候选矩阵：

| `block_N` | 每行 tile 数（Skv=8192） | 边界 tile 选择代价 | 常量 store 粒度 | 备注 |
|-----------|--------------------------|-------------------|----------------|------|
| 64 | 128 | 小（≤2×64 elem/行） | 256 B | tile 多、循环开销大 |
| 128 | 64 | ≤2×128 elem/行 | 512 B | 折中 |
| **256** | **32** | **≤2×256 elem/行（≈6% 行内元素，仅边界行）** | **1 KB** | **主选**（粒度/循环开销平衡） |
| 512 | 16 | ≤2×512 elem/行 | 2 KB | 边界浪费略大 |

边界 tile 的额外读写（≤2 tile/行）+ v-op ≤2×4 = 8/行，相对整行（Skv=8192）占比 <1%；故 `block_N` 对总代价影响小，主选 **256**（DMA 粒度 ≥1 KB、`T.serial` 迭代 32 次/行）。`block_N` 为工厂参数，Stage 4 可扫。

### 5.3 约束分析

- **对齐约束**：`block_N=256` fp32 尾轴 `256×4 = 1024 B`（≥32B ✓）；`G=2` 时整块连续 2 KB（连续 DMA）。
- **UB 容量**：手工 ≈7.0 KB（静态 `G_MAX=2`），膨胀后 ≤ ~12.2 KB ≪ 192 KB ✓（§4.5）。
- **L0/L1**：不涉及（无 Cube）。

### 5.4 注意事项（尾块 / 非整除 / 契约）

- **`seq_len_kv` 非 `block_N` 整数倍（尾块）**：`tail = T.min(block_N, seq_len_kv − k0)` 动态 extent 切片（PL-1.22 形态）；v-op 作用于整 tile、只拷出 `[0:tail]`，尾块外 UB 陈旧段不落盘。
- **窗口端点非块对齐（通用）**：`cu_k_s`/`cu_k_e` 为运行时值，一般非 `block_N` 倍数 → 由**边界 tile 的掩码选择**精确处理；**绝不把窗口端点当 DMA 偏移**（§0.6 R2）。
- **空窗（`ke ≤ ks`）**：所有 tile 落入「窗口外」→ 整行 −inf（与源一致）。
- **全窗（`ks=0, ke=S_kv`）**：所有 tile 落入「窗口内」→ 全行不触碰（与源一致）。
- **与主 kernel 的契约**：本单元须覆盖 `[0,ks) ∪ [ke,S_kv)` 全区间（含主 kernel 未写的 `[0, cu_k_s_min)` 与 `[cu_k_e_max_rounded, S_kv)`）——本设计按每行 `[ks,ke)` 分类，天然覆盖全区间，不依赖主 kernel 写到哪里。
- **`kv_group`**：并入 tile 尾轴，缓冲为**静态 `[block_N, G_MAX=2]`**（§4.6）；`kv_group=1`（manifest 全部）为连续 1-D 逻辑 tile，`kv_group=2` 时整 tile 连续（§1.6.3 #1）。运行时须 `kv_group ≤ G_MAX`（越界行为与缓解见 §4.6/§9.1）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`../../../.agents/skills/_shared/standards/core-split-strategy.md` §1/§2.1（依据 `docs/开发指南.md` §3.3）。

- **① 物理核数依据（实查）**：

  ```python
  from tilelang.utils.npu_utils import NPUUtils
  NPUUtils.get().get_aicore_num()          # 实查返回: 20  （本机 Ascend910B3）
  NPUUtils.get().get_aivector_core_num()   # 实查返回: 40  （= 20 × 2）
  ```

  → **物理核数 = 40**（`= get_aivector_core_num()`）。依据：本算子是**纯 Vector 算子**（无 Cube），标准 §1 要素②规定「纯 Vector 算子核数翻倍，即 `get_aicore_num() * 2`」→ **40**。查询时间：2026-10-15，工具链 `tilelang 0.1.2+a2357253662babf317ada194080078f928d155b3`（dev root build）/ CANN 9.0.0。**禁止文档假设/经验值**；本机设备名 `Ascend910B3`，核数 20/40 与 `CONST-aicore-910_9362` 一致但 SoC 名不同——此为**实查值**记录。

- **② 逻辑核数计算**：`num_logical_kernels = batch × seq_len`（每条 query 行一个逻辑任务；因形状动态，下表为 manifest 代表值）：
  - smoke `s1024`：1 × 1024 = **1024**
  - full `s4096`：1 × 4096 = **4096**
  - full `s8192`：1 × 8192 = **8192**

- **③ 规模判定与分核方案**：逻辑核数（1024–8192）**远大于**物理核数（40）→ 属**极大规模**。
  - **无法整数倍对齐**（数学依据）：`seq_len` 均为 2 的幂（1024/4096/8192），`batch=1` → 逻辑核数恒为 2 的幂；物理核数 **40 = 2³·5** 含非 2 因子 5；`gcd(2ⁿ, 40) = 8` → 数学上**不存在**任何 2 的幂逻辑核数能被 40 整除。
  - **方案（无法对齐兜底）**：取**非持久动态 grid = `batch·seq_len`**（每个逻辑任务一个 block），由运行时按 40 物理核分波调度，并**显式记录残留负载不均**：`8192 = 40×204 + 32` → 32 核执行 205 个任务、8 核执行 204 个，临界路径惩罚 ≈ 205/204.8 − 1 = **+0.1%**；`4096 = 40×102 + 16` → 16 核 103、24 核 102，临界路径 ≈ 103/102.4 − 1 = **+0.6%**（revision 1 建议 4：上一版写 +0.4%，按 `ceil(logical/phys)`/均值口径复算为 +0.59%）；smoke `1024 = 40×25 + 24` → 24 核 26、16 核 25，≈ **+1.6%**。
  - **为何不用「固定核数 + 核内 `T.serial`」持久化**：标准 §1 要素③「极大规模」选项要求**核内串行循环边界为静态值**；本算子接口契约为**动态 shape**（工厂以 `clean_logits_()` 无参调用，`num_logical` 非编译期常量），无法给出静态核内串行上界 → 落到「无法对齐兜底」。**残余可用优化**：若 Stage 4 实测核启动开销显著，可将逻辑粒度改为 `batch·seq_len·ceil(S_kv/block_N)`（更细），或改用 PL-1.22 持久模板（固定 40 核 + 核内 grid-stride），届时按静态/近似静态上界重评——列入 §9.1，非本阶段冻结项。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 逻辑任务（batch × query 行） | block 级并行 | `T.Kernel(batch*seq_len, is_npu=True)` | 1D 非持久动态 grid；每核处理一条 query 行的全部 kv |
| kv 块扫描 | tile 级顺序 | `T.serial(ceildiv(seq_len_kv, block_N))` | 动态上界（形状动态）；单遍无流水需求 |
| tile 三态分类 | block 级标量分支 | `if / elif / else`（嵌套，避免跨分支赋值） | 每 tile ≤3 次标量比较；窗口标量只作标量比较（§0.6 R2） |
| 元素级（写/掩码/选择） | 向量化 | `T.copy` 整块 / `T.vbrc` / `T.arange` / `T.vcmp` / `T.vand` / `T.vselect` | 依 §1.6.2 已全部向量化，无逐元素标量循环 |
| 逐元素并行循环 | **不用 `T.Parallel`** | — | 无逐元素标量循环需 `T.Parallel`：整块 `T.copy` + v-prefix 已覆盖；`T.Parallel` 下标不可变换（`docs/.../编译器提示操作/T.Parallel.md` §2.3），本设计无其用武之地 |
| 核内持久串行多任务 | **不用 `T.Persistent`** | — | 动态 shape 契约（§5.5）；非持久动态 grid |

### 6.2 循环伪代码

见 §3.3（本设计的循环结构即 §3.3 伪代码：`T.Kernel` → 一次常量/索引物化 → `T.serial(n_kv)` 内三态分类）。

### 6.3 流水线优化

- **无 `T.Pipelined`**：纯 store / 少量边界 load，无计算可与搬注重叠，无跨 tile 复用；双缓冲无收益（且增加 UB 占用）。
- **常量/索引物化外提**：`neg_ub`（−inf）与 `idx_ub`（arange）在 `T.serial` 循环**外**物化一次，循环内复用（承接 O4，§1.6.1）。

### 6.4 尾块处理

见 §5.4：`tail = T.min(block_N, seq_len_kv − k0)` 动态 extent + 零起点 UB 切片（`neg_ub[0:tail]` / `log_ub[0:tail]`），v-op 作用于整 tile、只拷出 `[0:tail]`。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**自动同步**（Developer 模式由编译器插入 GM↔UB 依赖同步，用户不手写 flag）。

### 7.2 同步点说明

无手写同步点。Developer 自动处理：边界 tile 的 `T.copy`(GM→UB) 与后续 `T.vselect` 的 RAW 依赖、`T.copy`(UB→GM) 的 WAR/WAW 依赖；整块常量 store 的 MTE3 写依赖。**若 Stage 3 遇同步相关崩溃**，回退手段见 §9.2（转 Expert 显式 `sync_block_set/wait`）。

### 7.3 pass_configs 配置

```python
pass_configs = {}   # 无。源无 pass_configs（clean_logits_ 未设 TL_ENABLE_FAST_MATH）
```

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据，**直接移植源仓参考实现的窗口掩码部分**（`examples/TileOPs/tests/ops/test_fp8_lightning_indexer.py` `ref_program` L82–L93），**不复刻** §0.6 的 NPU 三态扫描算法——保证验证独立性。

```python
def golden_clean_logits(logits, cu_seqlen_ks, cu_seqlen_ke):
    """In-place window mask: set -inf outside [ks, ke) per query row.

    logits: [B, S, Skv, G] fp32 (not modified in place; returns masked copy)
    cu_seqlen_ks/ke: [S] int32
    returns [B, S, Skv, G] fp32, -inf outside each row's window.
    """
    B, S, Skv, G = logits.shape
    ks = torch.clamp(cu_seqlen_ks, min=0, max=Skv)          # source L170
    ke = torch.clamp(cu_seqlen_ke, max=Skv)                 # source L171
    ke = torch.maximum(ke, ks)                              # source L171 (ke >= ks)
    ar = torch.arange(0, Skv, device=logits.device)
    keep = (ar[None, :] >= ks[:, None]) & (ar[None, :] < ke[:, None])   # [S, Skv]
    keep = keep[None, :, :, None]                                       # [1, S, Skv, 1]
    return torch.where(keep, logits, torch.full_like(logits, float("-inf")))
```

**比较器**（沿用源：`_validate_tensor_match`，L41–L65）：先比对 `isfinite` 掩码一致、非有限值逐位一致（`atol=0, rtol=0, equal_nan=True`），再在有限域（窗口内）用 **correlation** `1 − 2·Σab/Σ(a²+b²) ≤ 1e-3`。

### 8.2 精度标准

| 域 | 判据 | 容差 |
|----|------|------|
| 非有限域（窗口外 −inf、窗口内原 ±inf/NaN） | `isfinite` 掩码逐位 + 非有限值逐位一致（含 NaN） | **精确**（本算子为精确投影，无舍入） |
| 窗口内有限域 | correlation difference `1 − corr` | ≤ **1e-3**（源测试契约） |
| 窗口内有限域（辅助） | rtol/atol | 1e-5 / 1e-5（窗口内应逐位保留，bounded 输入应完全一致） |

### 8.3 L0 门槛测试计划（本阶段规划，交 `tilelang-op-develop`）

| 编号 | 用例 | 目的 |
|------|------|------|
| L0-1 | smoke `b1/s1024/skv1024/g1`，全窗（`ks=0,ke=Skv`） | 最小正确性（全跳过路径）、编译通过 |
| L0-2 | `b1/s1024/skv1024/g1`，随机窗 | 三态分类基础（含左/右窗口外 + 边界） |
| L0-3 | full `b1/s4096/skv8192/g1`，**非对齐端点** | 边界 tile 掩码 + 左区/右区常量写 |
| L0-4 | full `b1/s8192/skv8192/g1`，**空窗**（`ke=ks`） | 整行 −inf（全窗口外路径） |
| L0-5 | `b1/s512/skv1000/g1`（`skv` 非 `block_N` 倍数） | 尾块动态 extent |
| L0-6 | `b1/s1024/skv1024/g2` | `kv_group=2` 尾轴合并且不退化 |
| L0-7 | 窗口内注入 `±inf` / `NaN` / `−0.0` / subnormal | 边界语义（窗口内保留、窗口外覆写） |
| L0-8 | 反转窗（`ke < ks`） | clamp 退化空窗 |
| Boundary | 单行窗口（`ke=ks+1`）、窗宽 < `block_N`（单 tile 双端点） | 边界特例 |

（完整 L1/L2/Boundary 套件交 `tilelang-op-develop`，此处不枚举。）

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **动态 shape 契约（低风险，revision 1 已编译实证）**：wrapper 以 `clean_logits_()` 无参调用 → GM 形状全部 `T.dynamic`（grid / `T.serial` 上界 / 拷贝 extent 均动态）。**D-3 探针 form `ab` 已实证**「动态乘积 grid（`batch*seq_len`）+ 动态 trip + `or`/`and` 运行时条件」整链 `target="npuir"` **编译通过**；设备端运行时行为仍交 Stage 3 首编/L0 验证。
2. **运行时标量作 `T.vcmp`/`T.vbrc` 操作数（低风险，revision 1 已修订）**：**须 let-bind**——`s_off = cu_k_s − k0` 后再 `T.vcmp(idx_ub, s_off, ...)`（先例 `_gqa_prefill_fwd_kernel.py` L544/L551 即此形态）。**D-3 探针**：let-bound 通过（form `ab`）、内联复合式失败（form `c2`：`<Sub> has no attribute buffer`）。若仍遇前端限制，回退：`T.vbrc(s_off, bound_ub)` 物化 `[block_N, G_MAX]` 后 `vcmp(idx_ub, bound_ub, ...)`。
3. **`vcmp` int32 可能标量化（低-中风险，D-3 探针已观测告警）**：`PL-1.11-causal-mask-scalartrap` 实证 **int16 `vcmp` 全形态标量化**（236 µs/链 @[32,256]）。**本机探针**编译 int32 索引链时输出 `warning: Op 'hivm.hir.vcmp' will execute by scalar instruction with low efficiency`（`TRAP-expert-v-operands` 同源）。本设计 `vcmp` 仅用于 ≤2 个边界 tile/行（非逐 tile 热点），影响有界。**回退（按序）**：① **fp32 索引**——探针实测 `T.arange(idx_fp32, [1,0])` + fp32 `vcmp` **编译通过且无标量化告警**（边界偏移为 ≤512 的小整数，fp32 精确表示；**设备端数值正确性待 Stage 3 验证**，属「未文档化但本地实测可用」）；② 如需规避 fp32 arange 的文档缺口，退回 int32 并接受边界 tile 的标量化（占比 <1%）。
4. **`vand` 的 bool 支持（低风险）**：`T.vand.md` §2.2 表标 bool `×`，但 `_gqa_prefill_fwd_kernel.py` L874、`sparse_mla_fwd_dynamic_shape.py` L268 实测用 bool `vand`。若 Stage 3 bool 形态失败，回退：两次 `T.vselect` 嵌套（`vselect` 恒等支），或 `vcast` 到 int32 后 `vand`。
5. **估算常数绑定 9362、本机 910B3 未实测**（§1.6.0 估算下界/§4.5 UB 膨胀系数）：带宽/发射常数取 `CONST-ascend910-9362-profile` / `CONST-vector-op-cost-910_9362`，标注「未实测假设 + 估算依据」；Stage 4 须回填实测对账（`hardware-cost-model.md` §3）。
6. **负向候选的机器诊断留痕**：有限哨兵 `−1e38`（D1）与算术掩码乘（D2）均已由 `verify_equiv.py` 证非等价（`isfin_mismatch>0`）——非纸面断言。
7. **分核对齐**：逻辑核数（2 的幂）与物理核数 40（含因子 5）不可整数倍对齐 → 非持久动态 grid + 残留不均记录（§5.5）；smoke 惩罚 +1.6%（短任务，绝对影响小）。
8. **`or`/`and` 运行时条件写法（已探针实证）**：`if (k0 + tail <= cu_k_s) or (k0 >= cu_k_e):` 由 D-3 探针（form `ab`）实证可编译（无需退化为嵌套 `if/else`）；仍**不得**在 Python `if` 分支内赋值后于分支外引用（前序单元 RETROSPECTIVE 实测 `Undefined variable`，`TRAP-tvm-parser-rules`）。
9. **`G_MAX` 静态上界契约（revision 1 新增，低风险）**：片上 buffer 为静态 `[block_N, G_MAX=2]`，运行时须 `kv_group ≤ G_MAX`（§4.6）。全部已知 workload（manifest 四 workload `kv_group=1`、用例含 2）满足；上游若引入更大 `kv_group` 须显式提高 `G_MAX`（UB 增量 ≈3.5 KB/组，≪ 上限）。**缓解**：Stage 3 在 wrapper 侧加 host 侧 `assert kv_group <= G_MAX`（零 kernel 代价），越界时 fail-fast 而非越界访问。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| `MPU address access is invalid` / NaN | 用运行时窗口标量作 `T.copy` 偏移 | 结果错/设备报错 | **本设计已规避**（§0.6 R2）；禁止把 `cu_k_s`/`cu_k_e` 放进 `T.copy` 偏移 |
| `size=` 在 ≥3-D 前置补 1 | 用 `T.copy(..., size=...)` 写 4-D `Logits` | **静默写错区域** | 一律用**切片语法**（`Logits[by,bx,k0:k0+tail,0:kv_group]`，UB 侧缓冲静态 + 动态 extent）；见前序单元 repro `TRAP-copy-size-4d-extent-misalign.py` |
| bool 张量写 GM 失败 | 把 `cond_ub` 落盘 | 编译失败（i1 不可 load/store GM） | bool 仅留片上（`vand`/`vselect` 消费）；不 `T.copy` 到 GM |
| `0·(±inf)=NaN` | 用算术掩码乘产出 −inf | 契约不符（窗口外应为 −inf） | 用 `vselect`（精确选择）或常量直写，**不用**算术乘 |
| `-1e38` 哨兵 | 用有限大数替 −inf | `isfinite` 掩码不符 | 用精确 `-T.infinity("float32")` |
| **动态维作 UB buffer 形状** | `T.alloc_ub/alloc_shared([N, T.dynamic("kv_group")])` | `memref.alloc` 维度数校验失败 / MLIR verify 失败 | 尾轴用**编译期静态上界** `G_MAX`（revision 1 阻塞 1；D-3 探针 form `ab`） |
| **v-op 缓冲用 `T.alloc_shared`** | 运行时标量 `vcmp/vselect` 链搭在 `alloc_shared` 上 | `InternalError: cannot find variable clo` | v-op 缓冲用 `T.alloc_ub`（revision 1 阻塞 2；D-3 探针 form `c1` vs `ab`） |
| **`T.vcmp` 内联复合标量** | `T.vcmp(idx, cu_k_s - k0, cond, "ge")` | `<class 'tvm.tir.expr.Sub'> has no attribute buffer` | 先 let-bind `s_off = cu_k_s - k0`（revision 1 阻塞 3；D-3 探针 form `c2`） |
| UB 溢出（`ub overflow`） | `block_N`/`G_MAX` 过大 / auto-multi-buffer 膨胀 | 编译失败 | 本次预算 ≤12.2 KB（G_MAX=2）；若超限减小 `block_N` 或 `G_MAX`（§5.2/§4.5） |
| `[M,N]*[1,N]` 静默算错 | 误用 `vmul` 列因子 | M==N 静默错 | 本设计不使用 `vmul` 广播（用 `vbrc` + `vselect`） |
| 分支内赋值跨分支引用 | `if/else` 内给 Python 变量赋值 | `Undefined variable` | 嵌套 `if/else`，分支只含不跨分支共享的语句 |

### 9.3 特殊场景处理

- **极小 shape**：smoke `s1024`（逻辑核 1024，负载不均 +1.6%，§5.5）。
- **窗宽 < `block_N`**：单 tile 含双端点 → 走双边掩码路径（§1.6.1 E4 的单边特化不适用，回退双边，正确性不变）。
- **全窗 / 空窗**：分别走全跳过 / 全常量写（§5.4）。
- **`kv_group` 动态（仅 GM/拷贝 extent）**：片上尾轴为**静态上界** `[block_N, G_MAX]`，`kv_group=1` 连续；`kv_group=2` 整块连续（§1.6.3/§4.6）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/fp8_lightning_indexer/clean_logits_/
├── DESIGN.md               # 本设计文档（revision 1）
├── verify_equiv.py         # D-1 等价性机器验证脚本（已产出并执行：ALL_GATE_ITEMS EQUIV_PASS）
├── RETROSPECTIVE.md        # Stage 1 复盘（Phase 8，追加写入）
├── history_version/
│   ├── design_v0.md        # 上一版设计备份（Stage 2 不通过版）
│   └── design_probe_forms.py   # D-3 设计期编译探针（revision 1；PROBE_ALL_AS_EXPECTED）
└── clean_logits_.py        # 算子实现（待 Stage 3）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成（revision 1） | 设计文档 |
| `verify_equiv.py` | ✅ 已完成并执行（`ALL_GATE_ITEMS EQUIV_PASS`；revision 1 消除自比较项） | D-1 等价验证 |
| `history_version/design_probe_forms.py` | ✅ 已完成并执行（`PROBE_ALL_AS_EXPECTED`） | D-3 设计期编译探针（三形态） |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘（Phase 8） |
| `clean_logits_.py` | ⬜ 待实现 | NPU kernel 实现（Stage 3） |

### 10.3 命名规范

- 项目目录：`fp8_lightning_indexer`；算子目录：`clean_logits_`；实现文件：`clean_logits_.py`。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）
2. ✅ 等价性验证（verify_equiv.py）
3. ⬜ Golden 函数（§8.1）+ NPU kernel 实现（依 §3.3 伪代码）
4. ⬜ L0 门槛测试（§8.3）→ 集成（wrapper `clean_logits_()(Logits, CuSeqLenKS, CuSeqLenKE)` 契约不变）

---

## 11. 参考与工件索引

| 类型 | 路径 / 条目 |
|------|------------|
| 源算子 | `/home/developer/TileOPs/tileops/kernels/fp8_lightning_indexer.py`（`clean_logits_` L152–L184） |
| 抽取副本 | `examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/_f_p8_lightning_indexer_kernels.py`（L157–L189） |
| NPU 接口契约 | `examples/TileOPs/tileops/kernels/attention_indexing/fp8_lightning_indexer/fp8_lightning_indexer.py`（L87–L105） |
| 同任务前序产物 | `examples/fp8_lightning_indexer/_fp8_lightning_indexer_kernel/{DESIGN.md,REVIEW.md,_fp8_lightning_indexer_kernel.py,RETROSPECTIVE.md}` |
| 规格 manifest | `examples/TileOPs/tileops/manifest/attention_indexing.yaml` |
| 精度用例 | `examples/TileOPs/tests/ops/test_fp8_lightning_indexer.py`（L41–L94） |
| 性能用例 | `examples/TileOPs/benchmarks/ops/bench_fp8_lightning_indexer.py` |
| 同族先例 | `examples/indexer/indexer_fwd.py`；`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/_gqa_prefill_fwd_kernel.py`；`examples/deepseek_v32/sparse_mla_fwd_dynamic_shape.py`；`examples/elementwise/vec_add_2d_dynamic_shape.py`；`examples/flash_qla/tilelang_get_warmup_chunks.py`；`examples/elu/_make_elu_kernel/_make_elu_kernel.py` |
| D-3 设计期探针 | `history_version/design_probe_forms.py`（revision 1；`PROBE_ALL_AS_EXPECTED`——静态 UB + `alloc_ub` v-op 链 + let-bound 标量编译通过；`alloc_shared` v-op 链 / 内联 `vcmp` 标量失败） |
| 模式库引用 | `PL-1.11-causal-mask-scalartrap`（int16 vcmp 标量化 + 有限哨兵分析）、`PL-1.14-mte3-strided-ws`（跨步写退化）、`PL-1.22-persistent-1d-grid-stride`（动态尾块/UB 零起点切片）、`PL-1.23-vop-chain-inplace-alias`（`vselect` 原地）、`PL-1.29-vselect-cost`（`vselect` ≈6.5×）、`TRAP-expert-v-operands`（v-op 切片仅静态边界 / vcmp 标量操作数 / v-op dtype 矩阵）、`TRAP-vbrc-scalar-shared`（vbrc 标量→shared 不可用）、`TRAP-tvm-parser-rules` + `TRAP-conditional-alloc`（解析器作用域 / 条件 alloc 非法）、`constants.md`（`CONST-vector-op-cost-910_9362` / `CONST-ascend910-9362-profile` / `CONST-capacity-910B2C` / `CONST-aicore-910_9362`）、`queue.md`（**VP-2026-0228** 算术掩码乘——本算子以机器诊断证明其不适用；**VP-2026-0090** 对照行独立性——本版据此消除 D1 自比较；**VP-2026-0198** `T.alloc_ub.md` 的 `[Expert mode]` 标签与实测不符） |
| 共享标准 | `.agents/skills/_shared/standards/{algorithm-research.md, core-split-strategy.md, negative-claim-evidence.md, hardware-cost-model.md}` |
| 文档条款 | `docs/Tilelang.language/{内存操作/T.copy.md, 创建操作/T.arange.md, 创建操作/T.infinity.md, 比较操作/T.vcmp.md, 比较操作/T.vmin.md, 逻辑操作/T.vand.md, 条件操作/T.vselect.md, shape操作/T.vbrc.md, 编译器提示操作/T.Kernel.md, 编译器提示操作/T.Parallel.md, 比较操作/T.min.md, 比较操作/T.max.md, 数学操作/T.ceildiv.md}` |

---

## 性能目标（Stage 4 / optimize 场景；由 conductor 追加）

- 性能目标类型：`best_effort`（用户未指定目标类型 / 数值 / baseline）
- **本任务是 `clean_logits_` 的专项调优**（此前它与 primary 合在一个 optimize 任务里、被判定 `no_gain` 且未改造；实测它已成为 `FP8LightningIndexerOp` 的**主导开销**：s4k 约 11.8 ms / s8k 约 23.5 ms，而 primary 约 5.3 / 10.5 ms）。
- 性能 workload 来源：`examples/TileOPs/benchmarks/ops/bench_fp8_lightning_indexer.py::test_fp8_lightning_indexer_bench` 的 manifest 参数化案例所对应的 `Logits [b, s, s_kv, g]`（fp32）规模；完整集合由 optimizer 展开并分类，conductor 不写死 tune 用例与数量。
- **记录约定（多 kernel 目录）**：本目录 `statectl` 的 `kernel_opt_py_path` 现为 `perf_opt/clean_logits_.py`；`perf_opt/perf_records.jsonl` 须**只含 `clean_logits_` 的记录**，primary 的记录移至 `perf_records_primary.jsonl`（保留 `perf_records_all.audit.jsonl` 作为不可变并集）。
- 回归入口：`python {kernel_dir}/perf_opt/clean_logits_.py --level all`（L0/L1 失败阻塞）；采纳后从 `examples/TileOPs/` 运行 `python -m tileops.reporting.cli run --op FP8LightningIndexerOp --prof-mode msprof`
- 关注 shape：未指定；已知瓶颈为「`[block_N, kv_group=1]` 窄内维的逐行标量 copy」（代价 ∝ block 数，约 69 µs/block），`block_N` 增大反而 +81%。
