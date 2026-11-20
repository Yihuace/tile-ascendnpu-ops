# AdaptiveMaxPool2dIndicesFwdOp（`_adaptive_max_pool2d_with_indices_kernel`）算子设计文档

> **迁移任务（TileOPs 同构仓迁移，harness 子模式）**：源算子 = `AdaptiveMaxPool2dIndicesFwdOp`
> （family=`pool`，`variant_of` = `AdaptiveMaxPool2dFwdOp`），本次重实现的 TileLang kernel 函数 =
> `_adaptive_max_pool2d_with_indices_kernel`（**双输出**：values + int64 indices）。
> 本文按「源算子三问解读（§0）→ 算法调研（§1.6.0）→ 硬件耦合性判定与 NPU 重设计（§0.5/§0.6）
> → 通用设计流程（§1–§10）」组织。§1–§10 一律以 §0.6 重设计后的 NPU 算法为输入，不照抄源方案。
>
> - project = `adaptive_max_pool2d_with_indices`，op = `_adaptive_max_pool2d_with_indices_kernel`
> - 编程模式：**Developer**（迁移 prompt 指定）
> - 源算子路径（kernel）：`/workspace/TileOPs-gpu/tileops/kernels/pool/adaptive_max_pool2d.py`
>   （函数 `_adaptive_max_pool2d_with_indices_kernel`，第 119–206 行；同文件 13–74 行为单输出变体，不在本次范围）
> - 源 helper：`/workspace/TileOPs-gpu/tileops/kernels/pool/common.py`
>   （`adaptive_bin`、`max_adaptive_bin_extent`、`AdaptivePool2dKernelBase`）
> - 源 host 层（op）：`/workspace/TileOPs-gpu/tileops/ops/pool.py`（共享池化模块；`AdaptiveMaxPool2dIndicesFwdOp`
>   与 `_AdaptivePool2dFwdOpBase` / `_adaptive_pool2d_roofline` 均在该文件；源仓无独立 `ops/pool/` 目录文件）
> - NPU 侧抽取参考件（Stage 0 产物）：`examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d_with_indices/_adaptive_max_pool2d_indices_fwd_kernels.py`
> - NPU 侧 wrapper / Op（Stage 0 产物）：`examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d_with_indices/adaptive_max_pool2d_with_indices.py`、`examples/TileOPs/tileops/ops/pool/adaptive_max_pool2d_with_indices.py`
> - 迁移元数据：`examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d_with_indices/.migration_meta.json`
> - 规格来源（workloads / roofline）：`examples/TileOPs/tileops/manifest/pool.yaml`（`AdaptiveMaxPool2dIndicesFwdOp`）
> - 设备常数（本设计会话实查，2026-10-14）：`NPUUtils.get().get_aicore_num()` = **20**（Ascend910B3，
>   `npu-smi info` 显示 `910B3`）→ 纯 Vector 算子核数翻倍 = **40**；
>   toolchain `tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2` + `torch_npu 2.9.0.post8`。
> - 同族前作（骨架复用对象）：`examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/DESIGN.md`
>   （单输出 max 变体，CASE-pool-adaptivemaxpool2d-migration）、`examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/DESIGN.md`。
> - 版本：**revision v1**（`revision_index = 1`）；上一版 = `history_version/design_v0.md`；修订触发 = Stage 2 检视不通过（`REVIEW.md`，2 阻塞 + 5 建议）。

---

## 本次修订说明（revision v1 相对 v0）⭐

> 触发：Stage 2 检视不通过（`REVIEW.md` 的 2 个阻塞级 + 5 个建议级问题）。本节汇总相对
> `history_version/design_v0.md` 的**关键调整**与**根因对策**。除下列修正外，v0 的**正确冻结决策全部保持不变**：
> 取值集合 / 向量轴 = C / 分核粒度与单元定义（`num_logical` 唯一事实源）/ C4-C4b 特化分支 / 编程模式（Developer）/
> API 选型 / N1-N2-N3 公式与 `first_valid` 消解。§0 的源算子语义解读经检视独立 torch 实测复核成立，**沿用**（重读源码复核一致性见 §0.1）。

### R-v1-1 关键调整：first-hit 与值累加的实现次序（对应阻塞级 1）

- **v0 缺陷**：§3.3 / §6.2 伪代码把 `T.vmax(acc, row_f32[iw], acc)`（值累加）写在
  `T.vcmp(row_f32[iw], acc, gt_b, "gt")`（first-hit 判据）**之前** ⇒ 判据退化为 `v > max(a, v)`，对任意 `(a,v)` **恒假**
  ⇒ `max_idx` 恒等于初值 `f0`（窗口首元素），与 §1.6.1 N3「记录集合取 max」的**严格前缀最大值**前提冲突。
  反例：`arange(12).reshape(1,1,3,4)` 归约 `(2,2)` 的 `bin(1,1)` 应为 11，v0 次序得 6。
  §1.4 第 3 步的行内次序本已正确，故 v0 文档内部自相矛盾——**Stage 3 会按被声明为冻结决策的 §3.3/§6.2 实现**。
- **v1 修正**：§3.3 / §6.2 调换次序——`vcmp(gt)`（恒与**更新前**的 `acc` 比较）→ `vselect` → `vmax(max_idx)` →
  再 `vmax(acc)` → `vmul(h_flag)`；并在 §3.3 / §6.2 / §1.6.2 #1、#4 显式写入「**相对次序约束**」注释。
  §1.6.1 N3 与 §9.2 增补**实现级检查项**；§8.3 增补机制级证据要求（指数不在窗口首位的 shape 可复现）；
  `verify_equiv.py` 新增 **`N3-ord` 次序回归守卫**：正确次序必须复现基线 + `arange(12)→(2,2) bin(1,1)=11` 硬编码反证用例 +
  「值优先次序在一切『真 first-hit 离开窗口首元素』的非 NaN 窗口上必须被检出」（342 checks，全 `EQUIV_PASS`）。

### R-v1-2 关键调整：退化路径 UB 预算与容量判据（对应阻塞级 2）

- **v0 缺陷**：§4.5 的 C4 ≈90 KB / C4b ≈64 KB **低报**（本设计实算 132.8 KB / 112.1 KB；C4b 漏计
  `idx1d`/`idxp4`/`idxb4`/`idxbl` ≈51 KB）。默认档 ×1.7 后 226/195 KB **超 192 KB/AIV**，
  与 §3.3/§4.3 的 buffer 清单不符，「两档均在容量内」对退化路径**不成立**——而 C4 正是覆盖 manifest
  `global-1x1`（L0-5、三组 full workload 之一）的路径，Stage 3 若据 90 KB 启用 C4 将触发 BishengIR UB 溢出。
- **v1 修正**：退化路径通道块**收窄为 `CH_deg = min(C, 32)`**（一般路径保持 `CH = min(C, 64)`），
  按 §3.3/§4.3 的**实际 buffer 清单**重算两行为 C4 ≈66.6 KB / C4b ≈14.7 KB（manifest 恒等 case `(1,8,9,11)`，C=8 ⇒ `CH_deg=8`；
  `CH_deg=32` 的一般上界为 C4 ≈66.6 KB / C4b ≈56.5 KB），默认档 ×1.7 后 ≤ 113 / ≤96 KB，**两档均 < 192 KB ✓**；
  容量判据改写为**带字节数的不等式**（§4.5），并与一般路径同口径（逐 buffer 求和、保守「`T.reshape` 视图各自物化」）；
  保留「`CH_deg=8` 仍超则回退一般路径」兜底。§5.2/§5.5 的分核三要素按 `CH_deg` 重算（global-1x1 → 512 单元）。

### R-v1-3 建议级修正（5 项）

1. **§1.6.2 覆盖补全**：新增 #18「`for ow in T.serial(out_W)` 输出列调度循环」；把原 #16 拆为
   #16「静态界 `max_adaptive_bin_extent`（trace 期编译期常数）」与 #17「**运行时**标量 bin 边界算术（`adaptive_bin(o_h,…)`，块/索引级标量 TIR）」，
   消除 v0「#16 把运行时 `adaptive_bin` 当 trace 期常数」的错误；保留项枚举同步更新为 7 处。
2. **§1.6.0 R1 #8 负向断言修正**：v0 的「信息论不可行（值序需全 32 位）」前提在本算子输入域不成立——
   fp16→fp32 低 **13** 位恒零、bf16→fp32 低 **16** 位恒零，平面索引 `< 2^13`（fp16）/`< 2^16`（bf16）可塞入低位，
   manifest 平面 49–3136 全落在可塞区间。**结论（不采纳）保留**，但改按**量化/工程性依据**论证（收益有限 + 位运算/并列方向/NaN 位型代价 + 与 argmax 族一致性）。
3. **§0.6 R3 笔误**：承接行 `gt`+**`vmin`** → `gt`+**`vmax`**（与 §1.6.1 N3、§3.2 9c 一致；`vmin` 恰在本设计 §9.2 的错误清单上）。
4. **§0.4 #4 机制引用**：由单输出变体第 62 行改为本次目标函数（双输出变体）第 180–185 行。
5. **§8.3 语义门禁**：新增 L0-9b（bf16 语义门禁，与 L0-9 同注入集合：入窗 NaN / 两个 NaN / 全并列 / `{+inf,-inf}` / 全 `-inf`）。

### 为什么不会再犯同一错误

- **次序类（r-v1-1）**：v0 根因是「**公式级**验证通过 ≠ **实现级** op 次序正确」——`verify_equiv.py` 只对照公式、不验证 §3.3 的 op 相对次序。
  v1 把实现级约束**机器化**：`N3-ord` 以「正确次序必须复现基线 + 错误次序必须被检出」的**双向非负向断言**覆盖，
  反证用例取最小可复现 shape（硬编码 `arange(12)→(2,2)`），并以「真 first-hit 离开窗口首元素的非 NaN 窗口」为判据保证非空。
  同时 §3.3/§6.2/§1.6.2 的相对次序写死于注释与伪代码，消除文档内部矛盾——**同类错误在 D-1 层即可拦截**。
- **容量类（r-v1-2）**：v0 根因有二——「代表 shape 的容量结论被外推为**整个特化路径**的结论」，
  且退化路径 buffer 清单求和**漏项**（`T.reshape` 视图物化未计入）。
  v1 把容量判据写成**逐 buffer 带字节数的不等式**并与 §3.3 的 `alloc` 清单一一对应（退化路径按实际清单重算），
  数值上予退化路径独立 `CH_deg` 裕度并保留 `CH_deg=8` 回退一般路径——**不再以代表 shape 的结论代替退化路径结论**。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 方法论：`tilelang-op-design` skill 的 [references/migration-analysis.md](../../../.agents/skills/tilelang-op-design/references/migration-analysis.md)。
> 逐行读源文件（`adaptive_max_pool2d.py` 全文 253 行、`common.py` 全文 192 行、`ops/pool.py` 的
> `_AdaptivePool2dFwdOpBase` / `AdaptiveMaxPool2dIndicesFwdOp` / `_adaptive_pool2d_roofline` /
> `_normalize_output_size`），**未臆测语义**。
> 本次迁移范围 = **双输出变体** `_adaptive_max_pool2d_with_indices_kernel`
> （manifest `AdaptiveMaxPool2dIndicesFwdOp`，`ref_api` = `torch.nn.functional.adaptive_max_pool2d(..., return_indices=True)`，
> `variant_of` = `AdaptiveMaxPool2dFwdOp`）；单输出变体 `_adaptive_max_pool2d_kernel` 已由同族前作迁移，本文不重复。

### 0.1 源算子语义（做什么）

**数学语义**（`torch.nn.functional.adaptive_max_pool2d(x, output_size, return_indices=True)`，**双输出**）：

对输出空间位置 $(o_h,o_w)$，输入空间轴按 PyTorch adaptive-bin 规则划分为半开区间
（源实现由 `common.py` 的 `adaptive_bin` 给出）：

$$
\text{bin}_H(o_h)=\big[\lfloor o_h H_{in}/H_{out}\rfloor,\ \lceil (o_h+1)H_{in}/H_{out}\rceil\big),\qquad
\text{bin}_W(o_w)=\big[\lfloor o_w W_{in}/W_{out}\rfloor,\ \lceil (o_w+1)W_{in}/W_{out}\rceil\big)
$$

$$
\text{out}[n,c,o_h,o_w]=\max_{i_h\in\text{bin}_H(o_h)}\ \max_{i_w\in\text{bin}_W(o_w)}\ x[n,c,i_h,i_w]
$$

$$
\text{indices}[n,c,o_h,o_w]= f_{H,W}(i_h^\star, i_w^\star)
$$

其中 **选中的元素** $(i_h^\star,i_w^\star)$ 与 **索引的编码** 由下列**精确规则**给出（源实现的状态机，
`adaptive_max_pool2d.py` 第 161–202 行；本设计已用 **torch 实测复核**，见下）：

1. **索引编码 = H×W 平面内的行主序扁平下标**（**不是**二维坐标对）：
   $f_{H,W}(i_h,i_w) = i_h \cdot W_{in} + i_w$，dtype **int64**，shape 同 output。
   源码用 `flat_idx = T.cast(ih * w_in + iw, idx_dtype)` 构造（`adaptive_max_pool2d.py` 第 179 行），
   末尾 `T.cast(..., "int64")` 落盘（第 200–202 行）。
2. **并列（tie）取首个命中（first hit）**：源码更新条件为严格大于
   `elif val > max_val: max_val = val; max_idx = flat_idx`（第 190–192 行）→ 相等时不更新，故先访问者胜出。
3. **窗口含 NaN 时：值取 NaN，索引取窗口内「最后命中的那个 NaN」**（last NaN visited）：
   源码对每个 NaN 元素执行 `nan_idx = flat_idx; has_nan = True`（第 181–185 行，注释原文
   "PyTorch records the last NaN visited in a pooling window"）→ 后访问的 NaN 覆盖前者；
   末尾 `indices = has_nan ? nan_idx : max_idx`（第 200–202 行）。
4. **访问顺序 = 行主序**：`for kh in T.serial(max_kh): for kw in T.serial(max_kw)`（第 173–174 行），
   即先 `kh`（行）后 `kw`（列），元素坐标 $(i_h,i_w) = (i_h^{start}+kh,\ i_w^{start}+kw)$。
   「首个命中」与「最后 NaN」均以该顺序为准。
5. **`{+inf,-inf}` 同窗 → 值取 `+inf`（不是 NaN）**；`-inf` 按 IEEE 最大值语义参与比较。
6. **bin 永不为空**（`common.py` 的 `adaptive_bin` 注释：即使 `size_out > size_in` 时 bin 也非空）；
   非整除时相邻 bin 重叠一行/列；bin 边界用**整数差**（`adaptive_bin` 整数算术），非比例近似。

**本设计的 torch 实测复核**（torch 2.9.0 CPU，2026-10-14，`F.adaptive_max_pool2d(..., return_indices=True)`）：

| 输入（窗口内取值） | 实测 `val` | 实测 `idx`（int64） | 结论 |
|---|---|---|---|
| `x=arange(12).reshape(1,1,3,4)`，`output_size=(2,2)` | `[[5,7],[9,11]]` | `[[5,7],[9,11]]` | 索引 = `ih*W_in+iw` 扁平下标（W_in=4）✓ |
| 全 0 窗口（全部并列） | `0` | `0` | tie → first hit ✓ |
| 平面内两个最大值（扁平 0 与 2 均为 9） | `9` | `0` | tie → first hit（行主序先者）✓ |
| 平面内 NaN 在扁平 0 与 2（两个 NaN） | `nan` | `2` | NaN → **last NaN**（不是首 NaN）✓ |
| `{+inf,-inf}` 同窗 | `inf` | `0` | ±inf → `+inf`（**非 NaN**）✓ |
| 全 `-inf` 窗口 | `-inf` | `0` | first hit（`-inf` 也是合法最大值）✓ |
| `inf` + `nan` 同窗 | `nan` | `1` | NaN 胜出，索引为 NaN 的扁平下标 ✓ |
| `bf16` NaN 窗口 | `nan` | `0` | bf16 路径同语义 ✓ |
| 3D CHW `(2,3,4)` → `(2,2)` | — | int64，shape `(2,2,2)` | CHW 保持 rank，索引仍为平面扁平下标 ✓ |
| 扩张 bin（`2×2` → `3×3`） | `[[1,2,2],[3,4,4],[3,4,4]]` | `[[0,1,1],[2,3,3],[2,3,3]]` | 扩张 bin 语义一致 ✓ |

> **注意（与其它参考实现的分野，知识预注入 `VP-2026-0125`）**：NPU 的 device argmax / reduce 对 NaN 传播
> 常以「哨兵索引 `2^30`」或「首 NaN 索引」收场，本算子**两者都不是**——必须显式实现「值 NaN + last NaN 扁平索引」，
> **不得**依赖 `T.reduce_max` / `T.vmax` 的 NaN 行为（未文档化），也不得套用哨兵语义。

- **输出尺寸规则**（`_normalize_output_size` + `_resolve_out_dims`）：`output_size` 取
  `None | int | tuple/list[int|None, int|None]`；`None`（含 tuple 中的 `None`）→ 取对应输入轴尺寸。
  `out_H = H_in if oH is None else oH`，`out_W` 同理；标量 int 按 `(v,v)` 展开；`bool` 拒绝；`<=0` 拒绝。
- **dtype 语义**：输入仅 **float16 / bfloat16**（`_validate_adaptive_pool_input_dtypes` 显式拒绝 fp32）；
  源 kernel 内 `accum_dtype = "float"`（`adaptive_max_pool2d.py` 第 129 行）——逐元素先
  `T.cast(x[...], "float")` 再参与比较，末尾 `T.cast(result, dtype)` 单次舍回。
  **对 max 而言 fp32 是「比较域」而非「累加域」**：输出即被选中的那个输入元素，`dtype→fp32→dtype` 无损，
  故 `out` 逐位等于窗口内某输入元素（或 NaN）——这正是精度用例 `atol=rtol=0` 的依据
  （`examples/TileOPs/tests/ops/test_pool.py` 用例 `test_adaptive_max_pool2d_indices` 附近注释
  "Max is exact (no accumulation) and the index comparison is exact too"）。
- **索引 dtype 语义**：源码 `idx_dtype = "int32" if h_in * w_in < 2**31 else "int64"`（第 139 行，注释
  "keeps the div/mod off the 64-bit path; the stored index stays int64 to match PyTorch"）——
  **kernel 内以 int32 承载扁平下标，落盘时 cast 到 int64**；本设计保持该语义（§0.5 保留项）。
- **边界语义**：bin 非空；非整除重叠行/列；`±inf` 按 IEEE max；NaN 按「值胜出 + last NaN 索引」；
  3D CHW 输入经 op 层 `unsqueeze(0)` 成 4D、两路输出各自 `squeeze(0)` 回 3D（恒等 rank 保持）。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（直接移植源仓测试基准
`F.adaptive_max_pool2d(..., return_indices=True)`），**不复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x`（input） | `(N, C, H_in, W_in)`（op 层同时接受 3D `(C,H_in,W_in)`） | float16 \| bfloat16 | NCHW 连续布局，尾轴 `W_in` 连续 |
| `out`（output） | `(N, C, out_H, out_W)`；3D 输入时输出 3D `(C, out_H, out_W)` | same_as_input | NCHW 连续，`out_idx=[1,2]` 中的第 1 路 |
| `indices`（output） | `(N, C, out_H, out_W)`；3D 输入时输出 3D | **int64** | NCHW 连续，`out_idx=[1,2]` 中的第 2 路 |

**输出 shape（迁移必需字段）**：`(N, C, out_H, out_W)` **两路同形**，其中
`out_H = H_in if oH is None else oH`、`out_W` 同理。两路输出均为原生 NCHW 布局（**不是转置布局**）；
`indices` 与 `out` 的 shape 完全一致，仅 dtype 不同。

### 0.3 实现算法解读（怎么算）

源码结构化阅读（`kernels/pool/adaptive_max_pool2d.py` 第 119–206 行，逐行）：

**计算步骤分解**（源码每个计算语句均归入某步骤）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1（host，trace 期） | `total_output = n*c_in*out_h*out_w`；`max_kh = max_adaptive_bin_extent(h_in,out_h)`；`max_kw = max_adaptive_bin_extent(w_in,out_w)`；`idx_dtype = int32 if h_in*w_in < 2**31 else int64` | 编译期 shape 常量 | Python int / 字符串 | bin 循环**静态上界** + 索引承载宽度（TileLang 拒绝动态 `T.serial` 边界） |
| 2（grid） | `T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx` | `block_m`、`threads` | 一维 block 网格 | 输出元素级并行划分（**两路输出共用同一网格**） |
| 3（lane 索引解码） | `out_idx = bx*block_m + i`；`ow=out_idx%out_w`；`oh=(out_idx//out_w)%out_h`；`c=(out_idx//(out_w*out_h))%c_in`；`batch=out_idx//(out_w*out_h*c_in)`；经 `if out_idx < total_output` 守卫 | `out_idx` | 标量 `(batch,c,oh,ow)` | 输出坐标 $(n,c,o_h,o_w)$ |
| 4（bin 边界） | `ih_start,ih_end = adaptive_bin(oh,h_in,out_h)`；`iw_start,iw_end = adaptive_bin(ow,w_in,out_w)` | 标量 | 标量对 | $\text{bin}_H(o_h)$ / $\text{bin}_W(o_w)$ |
| 5（状态初值） | `max_val = cast(-inf, float)`；`has_nan = False`；`max_idx = cast(0, idx_dtype)`；`nan_idx = cast(0, idx_dtype)`；`first_valid = True` | — | 标量 fp32 / bool / idx_dtype×2 | 最大值初值 + NaN 标记 + 两个索引累加器 + 首有效标记 |
| 6（bin 扫描：比较 + 索引 + NaN 三路更新） | `for kh in T.serial(max_kh): for kw in T.serial(max_kw): if ih_start+kh<ih_end and iw_start+kw<iw_end:` 内依次 `ih=ih_start+kh`；`iw=iw_start+kw`；`val=cast(x[batch,c,ih,iw],"float")`；`flat_idx=cast(ih*w_in+iw, idx_dtype)`；`is_nan=T.isnan(val)`；`if is_nan: nan_idx=flat_idx; has_nan=True` / `elif first_valid: max_val=val; max_idx=flat_idx; first_valid=False` / `elif val>max_val: max_val=val; max_idx=flat_idx` | `x` 的逐元素 GM 读 | 标量 fp32 / bool / idx_dtype×2 | $\max_{i_h}\max_{i_w}$ + **first-hit 索引** + **last-NaN 索引** + NaN 胜出标记 |
| 7（结果 + 双写出） | `result = T.if_then_else(has_nan, cast(NaN,"float"), max_val)`；`out[batch,c,oh,ow] = T.cast(result, dtype)`；`indices[batch,c,oh,ow] = T.cast(T.if_then_else(has_nan, nan_idx, max_idx), "int64")` | 标量 | 标量 + 两个 GM 写 | NaN 判决 + 值/索引两路写出 |

**数据流与内存访问模式（源硬件视角）**：

```
GM[x] --逐元素标量读 x[batch,c,ih,iw]--> register(max_val fp32, has_nan bool,
                                              max_idx idx_dtype, nan_idx idx_dtype, first_valid bool)
        --NaN 判决 + 单次 cast--> GM[out]
        --has_nan?nan_idx:max_idx + cast int64--> GM[indices]
（无 SMEM/L1 中转；每个输出元素现场从 GM gather 其 bin 的全部元素；相邻输出元素重复读重叠行/列；
  两路输出共用同一次 bin 扫描——索引「免费」获得，不产生第二次 GM 读）
```

**循环与并行结构**：
- grid = `ceildiv(total_output, block_m)` 个 block，每 block `threads` 个 CUDA 线程；
- `T.Parallel(block_m)`：每个线程（lane）负责**一个输出元素**，lane 内全为**标量**计算；
- lane 内循环：`T.serial(max_kh) × T.serial(max_kw)` **静态边界** + 运行时 `if` 守卫跳过界外 lane；
- **lane 内无归约原语、无向量化**（每 lane 独立标量状态机）；
- **索引与值在同一遍扫描内同时产生**（关键结构事实，§0.4 优化 #12）；
- 无流水线（无 `T.Pipelined`）、无共享内存、无 SIMD。

**host 侧逻辑**（`ops/pool.py` 的 `_AdaptivePool2dFwdOpBase` / `AdaptiveMaxPool2dIndicesFwdOp`）：
- `_normalize_output_size`（`output_size` 规范成 2 元组）、`_resolve_out_dims`（`None` → 输入尺寸）；
- 3D→4D `unsqueeze(0)`；**双输出各自 `squeeze(0)` 回 3D**（`if self._returns_indices: return out.squeeze(0), indices.squeeze(0)`）；
- `x = x.contiguous()`（无实际拷贝时 no-op）；
- `_kernel_cache`（按 `(n,c,h,w,out_h,out_w,dtype,device)` 缓存 kernel 实例）；`dtype` gate（仅 fp16/bf16）；
- `functools.lru_cache(maxsize=32)` 缓存 kernel factory（按 shape/dtype）；
- `_adaptive_pool2d_roofline(op, indices=True)`（仅用于 manifest 性能规格，不参与计算；`bytes_` 额外加 `N*C*out_H*out_W*8`）；
- 编译配置：`@tilelang.jit(out_idx=[1,2], compile_flags=["-O3","-DENABLE_BF16"])`（`adaptive_max_pool2d.py` 第 134 行）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | **静态边界 bin 循环 + 运行时守卫** | 规避 TileLang「拒绝动态 `T.serial` 边界」，同时不虚增循环体 | `max_adaptive_bin_extent` 在 trace 期算出最宽 bin 作静态上界；守卫跳过界外 lane | 无（语言/编译器级约束） | **可移植**（NPU 同样只接受静态 `T.serial` 边界） |
| 2 | **fp32 比较域 + 单次输出舍入** | 与 PyTorch 比较语义一致 | 逐元素 `T.cast(x,"float")` 比较，末尾一次 `T.cast(result,dtype)` | 无（纯数值语义，max 下无损） | **保留**（语义层） |
| 3 | **bin 大小整数边界取整** | 与 PyTorch adaptive 语义逐位一致 | `adaptive_bin` 整数算术（非比例近似） | 无 | **保留**（语义层） |
| 4 | **显式 NaN 追踪 + 值覆盖** | 复刻「NaN 胜出」，不依赖 `T.max` 的 NaN 行为 | 本变体（**双输出**，第 180–185 行）：`is_nan = T.isnan(val)`；`if is_nan: nan_idx = flat_idx; has_nan = True`；末尾 `result = has_nan ? NaN : max_val`（第 194–198 行）（单输出变体形态见第 62 行 `has_nan = has_nan \| T.isnan(val)`，语义等价但**不在本次迁移范围**） | 无（纯数值语义） | **保留**（语义层，需在 NPU 上向量化） |
| 5 | **first-hit 严格大于规则** | 复刻 PyTorch tie 语义 | `elif val > max_val`（严格 `>` ⇒ 相等不更新 ⇒ 先访问者胜出） | 无 | **保留**（语义层，需在 NPU 上向量化） |
| 6 | **last-NaN 索引覆盖** | 复刻 PyTorch NaN 索引语义 | 每个 NaN 元素 `nan_idx = flat_idx`（后 NaN 覆盖先 NaN）；末尾 `has_nan ? nan_idx : max_idx` | 无 | **保留**（语义层，需在 NPU 上向量化） |
| 7 | **int32 承载扁平索引** | 把 div/mod 从 64 位路径移开（宿主注释原文） | `idx_dtype = int32`（当 `h_in*w_in < 2**31`）；仅在落盘时 cast 到 int64 | 无（位宽选择） | **保留**（数值/位宽层） |
| 8 | **双输出同轮融合** | 索引不额外付 GM 读代价 | 值与索引在同一 bin 扫描内同时产生；仅多一次尾部 select + 一次 GM 写 | 无（融合结构） | **保留/等价替换**（NPU 沿用单遍融合） |
| 9 | **一维 block 网格 + 输出元素扁平映射** | 把 `(N,C,oH,oW)` 四维问题压成一维可并行任务 | `T.ceildiv(total_output,block_m)` + `%`/`//` 解码 | GPU block 网格（非 hardware-specific） | **等价替换**（一维 Kernel） |
| 10 | **`T.Parallel(block_m)` 线程级 lane 并行** | 每线程处理一个输出元素，隐藏 gather 延迟 | CUDA thread/warp 两级并行模型 | **CUDA threads / warp**（NPU 无等价） | **重新设计**（→§0.6 R1） |
| 11 | **lane 内标量状态机（比较链 + 索引链 + NaN 链）** | 单线程顺序更新、寄存器驻留 | `max_val/max_idx/nan_idx/has_nan/first_valid` 寄存器 + 逐元素 GM gather | **GPU 大寄存器堆 + 寄存器比较链** | **重新设计**（→§0.6 R2/R3） |
| 12 | **逐元素 GM gather（无 tile 搬运 / 无 coalescing）** | 窗口不规则，直接按 bin 坐标逐元素取数 | 直接索引 `x[batch,c,ih,iw]` | GPU 的 L1/L2 缓存吸收重复访问 | **重新设计**（→§0.6 R4） |
| 13 | **`functools.lru_cache` factory 缓存 + `_kernel_cache` 实例缓存** | 避免重复 trace/编译 | host 侧缓存 | 无 | **保留**（host 侧） |
| 14 | **`adaptive_bin` / `max_adaptive_bin_extent` trace 期常数求值** | 把 bin 算术提前到 trace 期 | Python 整数运算（`adaptive_bin` 注释 `Evaluated while tracing`） | 无 | **保留**（host/trace 侧） |

> 识别不出来 ≠ 不存在。以上 14 项为源码中可见的全部优化/结构性选择（含 indices 特有的 #5/#6/#7/#8）；
> 未识别的项会在迁移中被静默丢弃。相对单输出变体，本变体**新增** #5 first-hit、#6 last-NaN、#7 int32 索引、#8 双输出融合。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定问题：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（adaptive bin 取最大值、bin 整数边界、输出尺寸规则） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1） |
| **NaN 胜出 + ±inf 最大值语义（#4）** | 语义 | 无 | — | **保留** | 数值语义；NPU 侧用 IEEE 正确的 `eq` 掩码取反实现（§0.6 R2；`ne` 为 ordered-NE 不可用，见 `TRAP-vcmp-ne-ordered-nan`） |
| **first-hit tie 语义（#5）** | 语义 | 无 | — | **保留** | 用严格 `gt` 掩码 + 索引 `vmax`（last-record）累计（§0.6 R3），语义完全一致 |
| **last-NaN 索引语义（#6）** | 语义 | 无 | — | **保留** | 改写为「NaN 位置上扁平下标的**最大值**」（访问序 = 扁平递增，故 last = max），用 `vmax` 累计（§0.6 R3 + §1.6.1 N2） |
| **索引编码 = 平面扁平下标 `ih*W_in+iw`（#7 的语义面）** | 语义 | 无 | — | **保留** | 契约（int64、shape 同 output、行主序扁平）；NPU 侧同样构造该扁平值 |
| **int32 承载索引 + 落盘 cast int64（#7）** | 数值/位宽 | 无 | 有 | **保留** | NPU 侧索引累加器用 int32（`T.vselect`/`T.vmax`/`T.vcmp`/`T.vbrc`/`T.transpose` 均支持 int32；`T.vcast` i32→i64 `rint` ✓）；工厂期守卫 `h_in*w_in < 2**31` |
| **双输出同轮融合（#8）** | 优化 | 无 | 有 | **保留** | 单遍扫描内同时产出值与索引（同一 UB tile），不产生第二次 GM 读 |
| I/O 契约（NCHW→NCHW 双输出、fp16/bf16、值 dtype 同输入、索引 int64） | 语义 | 无 | — | **保留** | layout 是契约 |
| fp32 比较域 + 单次输出舍入（#2） | 语义 | 无 | — | **保留** | NPU Vector 支持 fp32 `T.vmax`/`T.vcast`（`docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1 fp32 ✓） |
| 非整除重叠 bin、空 bin 不存在 | 语义 | 无 | — | **保留** | 语义层 |
| 静态边界 bin 循环 + 守卫（#1） | 算法 | 无（语言约束，非硬件） | 有（同款约束） | **保留** | NPU `T.serial` 同样只接受静态边界（`TRAP-tir-range-loop-not-unrolled`：Python `range` 亦降为串行 TIR 循环） |
| `adaptive_bin` / `max_adaptive_bin_extent`（#14） | 算法/host | 无 | 有 | **保留** | trace 期 Python 整数算术 |
| lru_cache / `_kernel_cache`（#13） | host | 无 | 有 | **保留** | host 缓存 |
| 一维 block 网格 + 输出扁平映射（#9） | 优化 | GPU block 网格 | 有（一维 `T.Kernel`，NPU 支持 ≤2 维） | **等价替换** | migration-analysis.md §5.3「block×thread 两级并行 → block 级任务划分 + 向量化元素并行」；`docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3 |
| **`T.Parallel(block_m)` 线程级 lane 并行（#10）** | 优化 | CUDA threads / warp | **无** | **重新设计** → §0.6 R1 | NPU 无 CUDA thread 概念；每 lane 一个输出元素会退化为**标量**执行（pattern-library `PL-1.2-caxis-accum` 同族实测窗口累加 98.3% 标量） |
| **lane 内标量状态机（#11）** | 优化 | GPU 寄存器堆 + 寄存器比较链 | **无直接等价** | **重新设计** → §0.6 R2/R3 | migration-analysis.md §5.3「register 累加 → UB 中间缓冲」；NPU 侧改成 C 轴向量最大值 + 向量掩码索引追踪 |
| **逐元素 GM gather（#12）** | 优化 | GPU L1/L2 缓存 | **无等价（且有害）** | **重新设计** → §0.6 R4 | NPU 逐元素 gather 无 coalescing、无 cache 吸收；改为 UB tile 搬运 + 向量化（`PL-1.2-caxis-accum`、`PL-1.1-transpose-chain`） |
| 退化网格（`1×1` / 恒等）无特化 | 优化 | 无 | 有 | **重新设计** → §0.6 R5 | 源对所有网格一律走双重 bin 循环；NPU 侧补结构特化（§1.6.1 C4/C4b，`ALG-pooling` 几何特化模式集） |

> 每项处置依据：migration-analysis.md §5.3 映射表条目 / `docs/Tilelang.language/` 条目 / pattern-library 实测条目（见上表「依据」列）。
> **舍弃项：无。** 源码未使用 warp shuffle / SMEM swizzle / mma / 三维 Kernel / cp.async 等 GPU 专有手段，故无舍弃条目。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（每项含「源方案 → NPU 新算法 → 语义保持论证」）。
**重设计输入含 §1.6.0 算法调研结论**：调研选定「直接自适应 bin 取最大值的**单遍在线状态机** + C 轴向量化 +
显式向量 NaN 追踪 + **向量化 first-hit/last-NaN 索引追踪** + 核内融合转置」算法族
（与源算法同族，但归约/追踪单元由标量改为 C 向量），故 §0.6 的重设计直接承接调研结论；
源算法已识别优化手段（§0.4）的优化意图承接见每项「承接」行。

**重设计项 R1：并行结构重设计（线程级 lane 并行 → 一维 Kernel + C 轴向量任务划分）**

- **源方案**：`T.Kernel(ceildiv(total_output,block_m), threads=threads)` + `T.Parallel(block_m)`，每 CUDA 线程处理一个输出元素。
  性能意图：把 `N*C*out_H*out_W` 个输出元素摊到大量线程上并行，靠 warp/线程切换隐藏逐元素 GM gather 延迟。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True)`；**逻辑任务单元 = `(n, o_h, c_block)`**，
  其中 `c_block` 是长度 `CH` 的**通道向量块**；`num_logical = N * out_H * ceil(C/CH)`（一般网格；
  退化路径的单元 = 整张 `(n, c_block)` 平面，见 §5.2/§5.5 ①）。
  单元内以 **C 轴为向量维**、`o_w`/`kw` 为 `T.serial(max_bin_*)` + 守卫的调度循环、`o_h` 由单元 id 运行时解码；
  `num_logical > vector_cores` 时按 persistent grid-stride 模式挂 `T.serial` 核内循环。
- **语义保持论证**：任务划分只是把互不相关的输出元素分给不同单元，**不改变任何窗口的取值集合、比较结果与访问顺序**
  （每个输出仍对其 bin 按行主序取最大值与索引）；无跨单元归约，无边界语义变化。
  承接：源 #9 的一维扁平映射被保留为「一维 Kernel」，#10 的 lane 并行意图由「单元并行 + C 轴向量并行」承接。

**重设计项 R2：值归约重设计（per-lane 标量比较链 + 标量 NaN 标记 → UB 内 C 轴向量取最大值 + 向量 NaN 掩码）**

- **源方案**：每输出元素在寄存器里做 `max_val = T.max(max_val, val)` 标量比较链，并行
  `has_nan = has_nan | T.isnan(val)` 标量 bool 累积，初值 `-inf` / `False`。
- **NPU 新算法**：C 轴向量化（向量轴 = C，§1.6.3）：
  1. 逐元素 cast 改为 **tile 级向量 cast**：`T.vcast(row_dst, row_f32, round_mode="rint")` 一次把整行 `(W_in, CH)` 提到 fp32。
  2. 取最大值改为 **C 轴向量最大值**：`T.vmax(acc[ow], row_f32[iw,:], acc[ow])`（原地），
     `acc`（`out_W × CH` fp32）经 `T.vbrc(-T.infinity("float32"), acc)` 初始化为 `-inf`（**单位元是 `-inf`，非 0**）。
  3. **NaN 追踪向量化为「乘性 eq 掩码」**（max 特有，语义关键）：行级
     `T.vcmp(row_f32, row_f32, eq_mask, "eq")`（`x==x` 为 1、NaN 为 0）→ `T.vcast(eq_mask, eq_f32, "rint")`；
     逐元素 `T.vmul(h_flag, eq_f32[iw], h_flag)`（`h_flag` 初值 1，0 吸收、1 中性）。**不依赖 `vmax` 的 NaN 行为**。
     依据 `TRAP-vcmp-ne-ordered-nan`（`ne` 为 ordered-NE，NaN 恒假不可用）。
  4. 输出：`T.vcast(out32, acc16, round_mode="rint")`（单次舍回 dtype）→ `T.transpose(acc16, acc_t, [1,0])` → `T.copy` 写回。
- **语义保持论证**：
  - **数学等价**：分块（`kh`/`kw` 循环）不改变取最大值的元素集合与访问顺序；C 轴是**独立轴**（每通道独立取最大值，§0.1），
    把 `c` 放进向量 lane 只是 SIMD 数据并行，**不改变任何 `c` 的取值集合**。
  - **NaN/±inf 语义核对**（逐条对齐 §0.1 实测）：`{+inf,-inf}` 无 NaN → 掩码乘积保持 1 → 输出 `acc`，
    `T.vmax(-inf,+inf)=+inf` ✓；窗口含 NaN → 乘积变 0 → 值输出 NaN ✓。
  - **dtype/舍入**：与源一致（fp32 比较域、末尾单次 `rint`）；`vcast` fp16/bf16→fp32 无损上转。
  - 承接：源 #2（fp32 + 单次舍入）与 #4（NaN 追踪）分别由「fp32 向量比较域」与「向量 eq 掩码乘积」承接；
    #11 的「寄存器比较链」意图由「UB fp32 向量最大值累加器」承接。

**重设计项 R3：索引追踪重设计（per-lane 标量 `max_idx`/`nan_idx` 状态机 → 向量化 first-hit / last-NaN 累加器）**

- **源方案**：每输出元素在寄存器里维护 `max_idx`（first-hit，严格 `>` 更新）与 `nan_idx`（每个 NaN 覆盖）、
  `first_valid` 标记，末尾 `has_nan ? nan_idx : max_idx`。
  性能意图：单线程顺序状态机、寄存器驻留、**与值归约同轮完成**（#8 融合意图）。
- **NPU 新算法**（C 轴向量，每个 lane = 一个通道）：
  1. 索引累加器（int32，匹配源 `idx_dtype`）：`max_idx`（`out_W × CH` int32）与 `nan_idx`（`out_W × CH` int32）。
  2. `max_idx` 初值 = **窗口首个被访问元素的扁平下标** `ih_start*W_in + iw_start(o_w)`（`T.vbrc` 标量填充）；
     `nan_idx` 初值 = `-1`（哨兵：`-1` 表示「该 lane 窗口内尚未见 NaN」）。
  3. 逐元素（`vs` = `row_f32[iw,:]`，`fv` = `T.vbrc((ih_start+kh)*W_in + iw, fvec)` 的扁平下标向量）：
     - **first-hit**：`T.vcmp(vs, acc[ow], gt, "gt")`（严格大于；NaN 参与时为假，与源 NaN 分支互斥一致）→
       `T.vselect(gt, fv, neg1, cand)`（非破纪录元素贡献哨兵 `-1`）→ `T.vmax(max_idx[ow], cand, max_idx[ow])`
       （原地 `vmax`）。**次序约束**：本步的 `acc[ow]` 必须是**本元素累加之前**的值（= 源 `val > max_val` 里「此前元素的最大值」），
       故值累加 `T.vmax(acc, vs, acc)` **必须在本步之后**执行（见 §3.3/§6.2 的注释与 §9.2）。等价性：源在递增扁平访问序上以严格 `>` 更新 `max_idx`，其**最后一次**更新即
       「值首次达到窗口最终最大值」的元素；破纪录元素按扁平**严格递增**出现，故「破纪录元素扁平下标的最大值」=
       最后一条记录 = first-hit 索引。`max_idx` 初值 = 首个被访问元素的扁平下标 `ih_start*W_in + iw_start(o_w)`，
       用于覆盖「全 `-inf` 窗口（无任何破纪录元素）」的情形——`-1` 哨兵恒小于窗口内任何合法扁平值，
       故初值不会被破纪录元素的下标污染。
     - **last-NaN**：`T.vcmp(eq_f32[iw], zero_elem, isnan, "eq")` → `T.vselect(isnan, fv, neg1, nanc)` →
       `T.vmax(nan_idx[ow], nanc, nan_idx[ow])`（原地 `vmax`；访问序扁平严格递增 ⇒ **最大值 = 最后访问的 NaN**）。
  4. 单元末尾：`cfinal = T.vcmp(nan_idx, zeros, "ge")`（`nan_idx ≥ 0` ⟺ 窗口含 NaN）→
     `idx32 = T.vselect(cfinal, nan_idx, max_idx, idx32)`（`has_nan ? nan_idx : max_idx`，与源第 200–202 行同式）。
  5. 输出：`T.transpose(idx32, idx_t, [1,0])`（int32 二轴交换）→ `T.vcast(idx_t, idx64, round_mode="rint")`
     （i32→i64）→ `T.copy` 写回 `indices`（行片 `(real_c, out_W)`）。
- **语义保持论证**：
  - **first-hit 等价**：源用严格 `>` 在**递增扁平序**上更新 `max_idx`，故记录的是**最后一条记录**，
    即「值首次达到窗口最终最大值」的元素（后续相等值不再更新）。破纪录元素按扁平严格递增出现，
    故「破纪录元素扁平下标的最大值」= 最后一条记录 = 该元素。向量式用 `vselect(gt, fv, -1)` + `vmax` 取
    该最大值，结果相同。并列（相等）时 `gt` 为假 ⇒ 不覆盖 ⇒ 先访问者胜出 ✓
    （与 §0.1 实测「平面内两个最大值 → idx 0」一致）。
  - **last-NaN 等价**：源对每个 NaN 覆盖 `nan_idx`；访问序扁平严格递增 ⇒ 最后一次覆盖 = NaN 位置集合中扁平**最大**者。
    向量式用 `vmax` 累计 `isnan ? f : -1`，结果同 ✓（与 §0.1 实测「两个 NaN → idx 2（后一个）」一致）。
  - **NaN 窗口下 `max_idx` 无关性**：源在 `has_nan` 时输出 `nan_idx`，`max_idx` 的后续更新不影响结果；
    向量式同样无条件更新 `max_idx` 但由 `cfinal` 判决取舍 ⇒ 一致。
  - **`first_valid` 被消解**：源用 `first_valid` 处理「窗口首个非 NaN 元素即 `-inf`」的情形；本设计把 `max_idx` 初值直接
    置为「首个被访问元素」的扁平下标，使「全 `-inf` 窗口」无需任何更新即得到 first-hit 正确结果
    （与源 `first_valid` 分支在被触发时写出的值相同）；证明见 §1.6.1 N3 与 `verify_equiv.py` 的 N3/N3b 用例组。
  - **`vmax` 的 NaN 行为无关性**（重要）：`acc` 的 NaN 传播/忽略两种约定下结果**逐位相同**
    （`verify_equiv.py` 的 N3 与 N3b 两行均 `EQUIV_PASS`）——因为含 NaN 的窗口由 `nan_idx` 独立判决，
    不含 NaN 的窗口两种约定等价。故本设计**不依赖**未文档化的 `T.vmax` NaN 行为。
  - **int32 位宽**：源 `idx_dtype` 语义保持（守卫 `h_in*w_in < 2**31`）；落盘 cast 到 int64 与源同。
  - 承接：源 #5（first-hit）、#6（last-NaN）、#7（int32 索引）、#8（双输出融合）分别由
    「`gt`+**`vmax`** first-hit 累加器」「`isnan`+`vmax` last-NaN 累加器」「int32 累加器 + i32→i64 vcast」「单遍扫描内同轮产出」承接，
    **无静默丢弃**。（注：first-hit 累加器为 `gt`+**`vmax`**——「破纪录元素扁平下标的最大值」；`gt` 必须与**更新前**的值累加器比较，见 §3.3/§6.2 的次序约束与 §9.2。）

**重设计项 R4：数据搬运重设计（逐元素 GM gather → UB tile 搬运 + 核内融合转置）**

- **源方案**：直接在 GM 上按 `x[batch,c,ih_start+kh,iw_start+kw]` 逐元素取值。性能意图：接口简单；
  GPU 靠 L1/L2 缓存吸收重叠行/列的重复访问。
- **NPU 新算法**：按 C 轴向量化所需布局（C 最内连续）搬运（与同族前作同款）：
  1. GM→UB：按 `(real_c, W_in)` 行片整片搬运 `T.copy(x[n, c0:c0+real_c, ih, 0:W_in], row_src[0:real_c,0:W_in])`；
  2. UB 内**融合转置链**：`T.transpose(row_src, row_dst, [1,0])` 把 `(CH, W_in)` 变成 `(W_in, CH)`（C 最内）；
     依据 `PL-1.1-transpose-chain`（核内二轴交换链，实测 ~µs 级；`docs/Tilelang.language/创建操作/T.transpose.md` §2.3 仅支持二轴交换）；
  3. 转置后 C 轴连续 → `T.vcast` 到 fp32 → 向量取最大值 / `gt` 比较 / eq 掩码；
  4. 输出侧：值与索引各自 `T.transpose([1,0])` 回转成 `(CH, out_W)` 后 `T.copy` 回 GM（值先 `vcast`、索引先 `vcast` 到 int64）。
- **语义保持论证**：搬运/转置是**纯数据重排**（双射），不改变元素值与访问顺序；输出写回后与 NCHW 契约一致。无精度影响（不引入任何算术）。
  承接：源 #12「靠缓存吸收重复访问」的意图由「UB tile 常驻 + 重叠行只搬一次（同一 `kh` 行的 w-bin 复用）」承接。

**重设计项 R5：算法替代/结构特化（退化网格）**

> 该项属「算法替代」模式（migration-analysis.md §6.2），采纳项与 §1.6.0 调研结论一致、并在 §1.6.1 完成等价性论证与机器验证。

- **源方案**：对所有输出网格一律走双重 bin 循环（含 `out_H=out_W=1` 全局池化、以及 `out==in` 恒等情形）。
- **NPU 新算法**（factory 级结构分发，形状静态可判定；仅在 UB 预算内启用）：
  - **C4（全局池化，`out_H==1 and out_W==1`）**：bin(0,0) = `[0,H_in)×[0,W_in)` 即整个平面 →
    按 `(CH, H_in, W_in)` 连续块搬运 + `T.vcast` fp32 + `T.reduce_max(dim=2→1)` 得值；
    索引侧按 argmax 形态：平面扁平下标向量 `k_j`（`T.arange` 1-D → `reshape` → `T.vbrc`）构造，
    `cand = T.vselect(vcmp(flat32, gbrc, "eq"), k_j, bigk)` → `T.reduce_min(dim=2→1)` 得 first-hit 索引；
    `candn = T.vselect(mb, k_j, neg1w)` → `T.reduce_max(dim=2→1)` 得 last-NaN 索引；
    判决 `has_nan = (nmin == 0)`（`eq` 掩码经 `reduce_min`）→ `idx = has_nan ? nank : firstk`。
    **NaN 判决不依赖 `reduce_max` 的 NaN 行为**（无论它传播还是忽略，`nmin` 与 `NaN 掩码归约` 独立给出正确判决）。
  - **C4b（恒等，`out_H==H_in and out_W==W_in`）**：所有 bin 大小 1 → `out = x`（纯 UB 中转拷贝，同 dtype 不 cast）；
    **索引 = 平面扁平下标常量模式**（与 `n`/`c` 无关）：`T.arange`（1-D，长度 `H_in*W_in`）→ `T.reshape` 成 `(1,1,H_in,W_in)`
    → `T.vbrc` 广播到 `(CH,1,H_in,W_in)` → `T.reshape` 成 `(CH,H_in,W_in)` → `T.vcast` 到 int64，
    该模式**在任务循环外构建一次**（任务不变），每个单元只做一次 `T.copy` 写入 `indices`。
- **语义保持论证**：见 §1.6.1（C4/C4b 均与基线公式**逐位等价**，`verify_equiv.py` 全部 `EQUIV_PASS`）。
- 承接：源「一般 bin 循环」在非退化网格上保持不变；退化网格特化**不改变语义**，仅减少冗余循环。

**重设计后的 NPU 算法一句话总结**：NCHW 进 / 双 NCHW 出，核内按 `(n,o_h,c_block)` 单元把 C 轴作为向量轴做自适应 bin 的
fp32 向量取最大值（`T.vmax` + eq 掩码 NaN 追踪）**并在同一遍扫描内**用 int32 向量累加器同时产出
first-hit 最大值索引（`gt`+`vselect`+`vmax` last-record）与 last-NaN 索引（`isnan`+`vselect`+`vmax`），用核内二轴转置把 C 搬到最内连续轴；
退化网格（全局池化 → 整平面 reduce 型 argmax / 恒等 → arange 常量索引模式）走结构特化路径。

### 0.7 标杆实现

- 源算子路径（获取 golden 语义依据）：`/workspace/TileOPs-gpu/tileops/kernels/pool/adaptive_max_pool2d.py`
  （函数 `_adaptive_max_pool2d_with_indices_kernel`）。
- 源仓参考实现 / 测试基准（NPU 侧移植后）：
  - `examples/TileOPs/tests/ops/test_pool.py`（用例 `test_adaptive_max_pool2d_indices`、
    `test_adaptive_max_pool2d_indices_accepts_chw`、`test_adaptive_max_pool2d_indices_nan_window`；
    `ref_program` = `F.adaptive_max_pool2d(input, size, return_indices=True)`，torch CPU 参考）；
  - 同文件 GPU 侧对应测试。
- **golden 以 §0.1 语义为唯一依据**，直接使用 `torch.nn.functional.adaptive_max_pool2d(..., return_indices=True)`，
  **不复刻 §0.6 的 NPU 算法**（保证验证独立性）。
- 规格来源：`examples/TileOPs/tileops/manifest/pool.yaml` 的 `AdaptiveMaxPool2dIndicesFwdOp`（3 组 workloads + 4 组测试 shape）。
- 基准来源：`examples/TileOPs/benchmarks/ops/bench_pool.py` 中的函数 `test_adaptive_max_pool2d_indices_bench`。
---

## 1. 概述

### 1.1 算子名称

`_adaptive_max_pool2d_with_indices_kernel`（迁移自 TileOPs `AdaptiveMaxPool2dIndicesFwdOp` /
`AdaptiveMaxPool2dWithIndicesKernel`）。

### 1.2 功能描述

自适应二维最大池化前向（**双输出**）：把 NCHW 输入的空间轴按 PyTorch adaptive-bin 规则
（`[floor(o*in/out), ceil((o+1)*in/out))`）划分，每个输出元素为其 bin 内的最大值，
并同时输出该最大值元素在 `H_in×W_in` 平面内的**行主序扁平下标**（int64）。
值语义：窗口内任一元素为 NaN 时输出 NaN（`{+inf,-inf}` 同窗取 `+inf`，非 NaN）；
索引语义：并列取首个命中（first hit），窗口含 NaN 时取**最后访问的 NaN**的扁平下标。

### 1.3 数学公式

$$
\text{out}[n,c,o_h,o_w]=\max_{i_h\in\text{bin}_H(o_h)}\ \max_{i_w\in\text{bin}_W(o_w)}\ x[n,c,i_h,i_w],
\qquad
\text{indices}[n,c,o_h,o_w]=i_h^\star\cdot W_{in}+i_w^\star
$$

$$
\text{bin}_H(o_h)=\big[\lfloor o_h H_{in}/H_{out}\rfloor,\ \lceil (o_h+1)H_{in}/H_{out}\rceil\big],
\qquad
\text{bin}_W(o_w)=\big[\lfloor o_w W_{in}/W_{out}\rfloor,\ \lceil (o_w+1)W_{in}/W_{out}\rceil\big]
$$

选中元素 $(i_h^\star,i_w^\star)$ 按 §0.1 的精确规则确定（行主序访问；first-hit 并列；NaN 胜出时索引取 last NaN）。
`H_out = H_in if oH is None else oH`，`W_out` 同理。

### 1.4 算法描述

**（迁移决策后的 NPU 侧算法，见 §0.6）** 单步 + 窗口归约类算子，纯 Vector：

1. host/trace 期（factory 内）：由 `(H_in,out_H)`/`(W_in,out_W)` 算出每个 `o_h`/`o_w` 的 bin 边界 `(start,end)`
   （`adaptive_bin`）与静态上界 `max_bin_h = max_adaptive_bin_extent(H_in,out_H)`、`max_bin_w`（编译期常数）；
   并判定退化网格（C4 全局 / C4b 恒等）与 UB 预算（含退化路径通道块 `CH_deg = min(C, 32)` 与容量判据，§4.5）。
2. 逻辑任务单元 `(n, o_h, c_block)`（`CH` 通道向量块；一般网格 `CH = min(C,64)`）；单元内：
   - `acc`（`out_W × CH` fp32）经 `T.vbrc(-inf)` 置 `-inf`；`h_flag`（`out_W × CH` fp32）经 `T.vbrc(1.0)` 置 1（NaN 免标记）；
     `max_idx`（`out_W × CH` int32）初值 = 首个被访问元素扁平下标 `ih_start*W_in + iw_start(o_w)`；
     `nan_idx`（`out_W × CH` int32）初值 = `-1`；
   - `for kh in T.serial(max_bin_h)`：若 `kh < bin_h(o_h)`（运行时守卫），搬入输入行 `(CH, W_in)` → 核内转置 `(W_in, CH)`
     → `vcast` fp32；随后对每个 `o_w`、每个 `kw`（均为 `T.serial(max_bin_w)` + 守卫）**按下列固定次序**执行七条向量 op：
     `vcmp(gt)`（**与更新前的 `acc` 比较——次序约束，须先于 `vmax(acc)`**）/ `vselect(max 索引候选)` / `vmax(max_idx)` /
     `vmax(acc)` / `vcmp(isnan)` / `vselect(NaN 索引候选)` / `vmax(nan_idx)`，外加 `vmul(h_flag, eq_f32)` 与 `vbrc(fv)`（见 §1.6.2 表）；
   - 末尾：`cfinal = (nan_idx ≥ 0)`（⟺ 窗口含 NaN）→ 值 `vselect(cfinal, NaN, acc)`、索引 `vselect(cfinal, nan_idx, max_idx)`
     → 值 `vcast` 回 dtype、索引 `vcast` 到 int64 → 二者各自 `transpose([1,0])` 回转 `(CH, out_W)` → 写回 GM。
3. 退化网格（`1×1` 全局池化 / `out==in` 恒等）走 §0.6 R5 的结构特化路径（C4 整平面 reduce 型 argmax；C4b 纯拷贝 + arange 常量索引），
   通道块取 `CH_deg = min(C, 32)`（容量驱动，§4.5）。

**与源算法的结构差异**（来源标注）：
- 归约单元：源「每输出元素一个线程 + 标量比较链」→ NPU「每单元一个 C 向量块 + `T.vmax` 向量比较」（§0.6 R2）。
- NaN 追踪：源「标量 bool `has_nan \| isnan(val)`」→ NPU「C 向量 eq 掩码乘性累积」（§0.6 R2；语义不变）。
- 索引追踪：源「标量 `max_idx`/`nan_idx` 状态机 + `first_valid` 分支」→ NPU「int32 向量累加器
  （`gt`+`vmax` last-record / `isnan`+`vmax` last-NaN）」（§0.6 R3；语义不变，`first_valid` 被初值消解）。
- 访存：源「逐元素 GM gather」→ NPU「UB tile 搬运 + 核内融合转置」（§0.6 R4）。
- 并行：源「block×thread 两级 + lane/元素」→ NPU「一维 Kernel + `(n,o_h,c_block)` 单元 + C 轴向量 lane」（§0.6 R1）。
- 退化网格特化：源无 → NPU 有（§0.6 R5，见 §1.6.1）；索引侧特化为「整平面 argmax」/「arange 常量模式」。
- **不变**：bin 整数边界算术、`kh`→`kw` 的访问顺序、first-hit 并列规则、NaN 胜出 + last-NaN 索引 + ±inf 最大值语义、
  fp32 比较域 + 单次输出舍入、int32 承载索引 + 落盘 cast int64、静态边界 bin 循环 + 守卫、NCHW 双输出契约、双输出同轮融合。

### 1.5 数据流图

```
GM[x: (N,C,H_in,W_in) fp16/bf16]
   │  T.copy 行片 (CH, W_in)  [同一 c 的整行连续]
   ▼
UB[row_src: (CH, W_in) dtype] --T.transpose [1,0]--> UB[row_dst: (W_in, CH) dtype] --T.vcast(rint)--> UB[row_f32: (W_in, CH) f32]
   │                                                              └--T.vcmp(eq)--> UB[eq_mask bool] --T.vcast--> UB[eq_f32 f32]
   │  T.serial kh (静态界+守卫)
   ▼
   ├─ 值：   UB[acc (out_W,CH) f32 <-- vmax(acc, row_f32[iw])]  ∥  UB[h_flag f32 <-- vmul(h_flag, eq_f32[iw])]
   │        ⚠ 次序约束：本步的 vmax(acc) 必须在下方索引 `vcmp(gt)` 之后（gt 与更新前的 acc 比较）
   └─ 索引： UB[max_idx (out_W,CH) i32 <-- vmax(max_idx, vselect(gt, fv, -1))]   # gt 先于 vmax(acc) 取掩码
             UB[nan_idx (out_W,CH) i32 <-- vmax(nan_idx, vselect(isnan, fv, -1))]
   │
   │  cfinal = vcmp(nan_idx, 0, "ge")                       # 窗口含 NaN ⟺ nan_idx >= 0
   ├─> UB[out32 f32] = vselect(cfinal, NaN, acc)  --vcast(rint)--> UB[acc16] --transpose[1,0]--> UB[acc_t (CH,out_W)] --T.copy--> GM[out]
   └─> UB[idx32 i32] = vselect(cfinal, nan_idx, max_idx) --transpose[1,0]--> UB[idx_t] --vcast(rint, i32->i64)--> UB[idx64] --T.copy--> GM[indices]
（退化路径 C4：GM → UB 连续 (CH,H*W) → vcast f32 → reduce_max 得值 + arange 下标向量 → vselect(eq,vmax) → reduce_min 得 first-hit
             + vselect(isnan,vmax) → reduce_max 得 last-NaN；eq 掩码 reduce_min 判 has_nan；vselect 判决 → vcast → GM 双输出）
（退化路径 C4b：GM → UB 整平面 (CH,H_in,W_in) → GM 纯拷贝；索引侧 = 任务不变的 arange 常量模式 (CH,H_in,W_in) int64 → T.copy 到 GM[indices]）
（无 GM 中间往返、无 Cube/L1/L0C 路径）
```

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象 = **同一数学语义的算法族**（所有能算出「adaptive-bin 最大值 + 其平面扁平下标」的结构）：
> **基线 = 源算法**（直接自适应 bin 扫描 + 单遍标量状态机）。
> 本算子属**窗口/归约类 + 索引输出类** → 完整调研（R1 候选表 + R3 四口径表 + R4 逐候选评估）。
> 参考表命中行：`algorithm-candidates.md` 的 `ALG-pooling`（family=pooling）——`R1: avg pool = 常数权重卷积；sum pool × 常数 = avg pool`；
> `R2: 滑窗天然流式`；`R3: 窗口重叠数据的重复读`；`R4: C 轴向量轴；跨步系数`；
> `known_impl` = `avg_pool2d` 系列，`kb_links` = `PL-1.2-caxis-accum / PL-1.3-host-permute / PL-1.4-tiling-heuristic`，
> 并含「几何特化路径模式集」（`out=1×1` 整平面归约 / `out==in` 纯拷贝 / 退化路径容量判据 / 单元粒度规则）。
> **命中行已全部评估**（见 R1/R3/R4）；**但 ALG-pooling 的 R1 命中项（常数权重卷积/GEMM）对 max 不成立**
> （max 非线性），且该命中行**未覆盖索引输出**——「索引」是本算子与同族 avg/max 单输出变体的关键差异，
> 故 R1 额外补充索引特有的候选（复合键 / 两遍 / 可分离 / 逐元素编码）。
> 另查 `ALG-reduction`（family=reduction/statistics）系列与 argmax 族（本仓 `examples/TileOPs/tileops/kernels/reduction/argmax/`，
> `CASE-argmax-argreduce-stage4`）——argmax 是「索引输出」的最近邻算法族，其候选（掩码候选 + `reduce_min` first-occurrence、
> 在线 running (extreme, idx) 递推、`T.deinterleave` 窄 N 分解）均纳入 R1/R4 评估。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线：源算法**（直接自适应 bin 扫描 + 单遍标量状态机，值与索引同轮） | $\max_{kh}\max_{kw}x$；索引按严格 `>` first-hit / NaN 覆盖 last-NaN | — | — | ✅（基线） |
| 0' | **基线向量化 = 单遍在线状态机 + C 轴向量**（值、first-hit 索引、last-NaN 索引三路累加器） | 同 #0（取值集合、访问顺序、判决完全一致），仅 SIMD 化；索引重写为「破纪录元素扁平取 max」「NaN 位置扁平取 max」 | **数学恒等**（C 轴独立 + 访问序不变；见 §1.6.1 N1/N2/N3） | 标量链 → 向量链；索引不额外付 GM 读 | ✅（**选定**） |
| 1 | 常数权重卷积 / im2col+Cube GEMM（ALG-pooling R1 命中行） | $\text{out}=W_0\otimes x$（0/1 权重） | **❌ 不等价**：`max` 非线性；该命中行仅对 avg/sum 成立 | — | ❌（数学不等价） |
| 2 | 可分离 2D（H/W 两轴分两遍）+ 逐轴 argmax 合并 | $\max_{i_h}(\max_{i_w}x)$；索引 = 行 argmax 再跨行合并 | 数学恒等（max 可交换/结合；行主序 first-hit 在「行内 first-hit + 行序首次破纪录」下保持） | 名义减少比较次数，但见 R3 | ✅（量化后弃） |
| 3 | 前缀最大值 + 差分 | $P[b]-P[a]$ | **❌ 不等价**：`max` **无逆元**（`x=[5,1,3]` 区间 `[1,3)` max=3 而 `P[3]-P[1]=0`；索引同样无法从差分恢复） | — | ❌（数学不等价） |
| 4 | 退化网格结构特化（`1×1` / 恒等） | `1×1`: 整平面 max + argmax；恒等: `out=x`、索引 = `ih*W+iw` 常量 | 数学恒等（bin(0,0)=全平面；单位 bin 的最大值=元素本身、其索引即该元素扁平下标） | 全局池化：$H{\cdot}W$ 次向量 `vmax` → 1 次向量归约 + 1 次 argmax 归约；恒等：消除全部归约 | ✅ |
| 5 | sum-pool × 常数（ALG-pooling 命中行） | $(\sum x)\cdot c$ | **❌ 不等价**（该式给 avg/sum，不给 max，亦不给索引） | — | ❌（数学不等价） |
| 6 | 归约融合：值/索引/NaN 三路**在同一轮扫描内**产出（源已是同轮） | 单轮：`vmax` + `gt`/`isnan` 掩码索引累加 | 数学恒等（同轮与分轮结果相同） | 避免第二遍扫描（源 #8 已是同轮，NPU 沿用） | ✅（并入 #0'） |
| 7 | **两遍式**：先算值（一遍扫描），再回扫选索引 | `pass1: max`；`pass2: first f with v==max / last f with isnan` | 数学恒等（索引 = `min{f:v_f=max}` / `max{f:isnan(v_f)}`，见 §1.6.1 N2/N3） | **负收益**：需第二遍 GM 读（+1 遍扫描）或整窗口 UB 驻留 | ❌（访问量翻倍；见 R3） |
| 8 | **复合键编码**：把 `(value, index)` 打包成单一可比较量，用**一次** `vmax` 同时更新值与索引 | 若值 dtype 与索引可共用一个有序键（如 fp32 值 + 低位塞索引） | **❌ 不采纳（工程性，非信息论）**：fp16→fp32 的表示低 **13** 位恒零、bf16→fp32 低 **16** 位恒零，故平面索引 `< 2^13`（fp16）/`< 2^16`（bf16）**可**塞入低位，本算子 manifest 与测试 shape 的平面量级 49–3136（56×56、55×57）**全落在可塞区间内** ⇒ 该路径在本输入域**并非数学不可实现**。仍**不采纳**，理由：① **收益有限**——每元素仅省 `vcmp`+`vselect`+`vmax` 3 条 op，而本算子按 §1.6.0 设计期估算为**带宽受限**、发射项非主导；② **代价与风险**——需 `T.vand`/`T.vor` 位运算构造键包、需反转索引编码以保持 first-hit 并列方向、需证明 NaN/±inf 位型与索引位在同一 32 位内共存的正确性、需按 `h_in*w_in` 分段加守卫与回退路径；③ **与 argmax 族已验证形态不一致**（本仓 argmax 用独立索引累加器）。 | — | ❌（工程性弃选，R3 标 N/A） |
| 9 | **逐元素编码 + `reduce_min/max`**（argmax 族形态：`cand = ite(v==ext, idx, BIG)` + `reduce_*`） | 窗口整体驻留 UB 后一次归约 | 数学恒等（argmax 语义） | 适合**窗口可整体驻留**的退化网格（C4） | ✅（**仅用于 C4**；一般网格窗口跨行无法整体驻留，见 R4） |

**R1 初判小结**：数学等价的化简结构有三类——① **索引重写**（顺序覆盖状态 → 可交换的 `max` 归约，§1.6.1 N1/N2/N3，
使索引可向量化，是**本算子相对单输出变体的关键算法级优化**）；② **退化网格特化**（#4，C4/C4b）；
③ **三路归约融合**（#6，源已有）。与 avg/sum 不同，max 的**全部代数化简路径（#1/#3/#5）因 `max` 非线性/无逆元而不成立**；
索引侧另有两条**不采纳**路径（#7 两遍式：访存翻倍；#8 复合键编码：**工程性弃选**，非数学不可实现——见上表 #8 依据）。

**R2 在线算法**：**有**（且基线本身已是单遍）。结构判据：算子可分解为 **running 最大值** + **running first-hit 索引**
- **running last-NaN 索引** + **running NaN 标记**，四者均为「只依赖已扫描前缀」的累加量，无跨元素未决依赖，
故天然存在单遍流式变体——每个输入元素读一次，参与其归属输出 bin 的值/索引/NaN 三路更新，
运行状态为 `O(out_W·CH)` 的累加器（远小于 `O(H·W)` 物化）。
**收益口径**：扫描遍数 1（无两遍式）、中间缓冲 `O(out_W·CH)`、UB 驻留完全可行（见 §4.5 预算）。
迁移任务佐证：源算法 §0.4 #8 已是值/索引同轮，§0.4 #11 是 running 比较状态机，构成「存在在线变体」的直接证据。
结论：**在线单遍变体 = 采用（= 基线结构），无第二遍扫描、无整窗口物化**。

**R3 复杂度对比**（逐候选四口径；本算子是**带宽受限**算子，访存/遍数为主导项，不只看 FLOPs）：

以 `spp-6x6`（`N=2,C=128,H=W=56,out=6×6`，`scan_H=scan_W=60`，fp16）为口径示例；口径：FLOPs 计「元素访问次数」
（一次最大比较 = 1）；访存 Bytes = GM 读 + GM 写（**含** `indices` 的 int64 写出，**不含** UB 内搬运）；
扫描遍数 = 输入 GM 读取遍数；中间缓冲峰值 = 片上驻留量。

| 算法候选 | FLOPs（元素访问次数） | 访存量 Bytes（GM） | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|----------------------|--------------------|---------|-------------|---------------------|
| 0 基线（直接标量） | `N·C·scan_H·scan_W` = 921 600 | `(N·C·H·W + N·C·out_H·out_W)·2 + N·C·out_H·out_W·8` = **1 697 792** | 1 | `O(1)`（寄存器标量） | 元素级并行 `N·C·out_H·out_W = 9216`；**无跨核归约** |
| 0' 基线向量化（**选定**，C 轴） | 921 600 元素访问；**向量 op ≈ 8×14400 + 行开销 ≈ 117 K** | 同上 **1 697 792**（+ 重叠行重复读，已含在 `scan_*` 内） | 1 | `acc+h_flag+max_idx+nan_idx` + `row` 系列 ≈ **65 KB**（§4.5） | 单元并行 `N·out_H·ceil(C/CH) = 24`；**无跨核归约** |
| 1 常数权重卷积 / GEMM | **不适用**（非线性，无法表达） | — | — | — | — |
| 2 可分离 2D（+ 逐轴 argmax 合并） | 每平面 `H_in·scan_W + scan_H·out_W` = 3360+360 = **3720/平面**（> 3600 = `scan_H·scan_W`）；合并阶段另有 `out_H·out_W` 次双 `vselect` | 同上 | 1 | + 行部分最大值/索引缓冲 `R(H_in·out_W·CH)`（spp CH=64 ≈ 86 KB） | 同上，但多一中间缓冲 + 阶段间串行 |
| 3 前缀最大值 + 差分 | **不适用**（max 无逆元） | — | — | — | — |
| 4 退化特化（`1×1` / 恒等） | `1×1`: **1 次向量归约 + 1 次 argmax 归约**（vs 基线 `H·W` 次向量 `vmax`）；恒等: **0 次比较**（纯拷贝 + 常量索引） | 同上 | 1 | `(CH_deg,H·W)` 平面块 + 下标/候选/哨兵向量 ≈ 66.6 KB（CH_deg=32，见 §4.5） | 同上 |
| 7 两遍式（值遍 + 索引遍） | 921 600 元素访问 ×2（或窗口物化后 1 遍 + 归约） | 同上 **+ 第二遍读 802 816 B**（或窗口物化内存） | **2**（或 1 遍 + `O(H_in·W_in·CH)` 驻留） | 整窗口物化需 `O(scan_H·scan_W·CH)` ≫ UB | 不可行 |
| 9 逐元素编码 + reduce（argmax 族） | `out_H·out_W` 次窗口归约（每次 `bw·bh` 条向量 op） | 同上 | 1 | 需窗口整体驻留 `O(bh·bw·CH)` | 仅适用于窗口可驻留的 C4 |

> 口径含：GM 输入读 + 双输出写（值 + int64 索引）；不含：UB 内搬运/转置/掩码（片上，计入 §1.6.0 估算行的发射项）。
> **基线在表内**（#0）；选定候选 #0'（基线向量化）。候选 1/3/5 因**数学不等价**、候选 8 因**工程性弃选**（非信息论，见 R1 #8）不参与复杂度对比（标注 N/A）。
> 候选 2 的运算量为**我方独立复算**（每平面 `H_in·scan_W + scan_H·out_W` vs `scan_H·scan_W`，见上行）。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附 `docs/`/`examples/` 佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 0 基线（标量） | **Vector 单元浪费**（逐元素标量比较 + 标量状态机，`PL-1.2-caxis-accum` 同族实测窗口累加 98.3% 标量） | 寄存器 | — | 静态（`max_bin_*` 界） | 无 | ❌ 计算单元失配（重设计为 #0'） |
| **0' 基线向量化（C 轴，选定）** | **Vector 逐元素/归约** ✓（`T.vmax` fp32/int32 ✓ / `T.vcmp` fp32/int32 ✓ / `T.vselect` fp32/int32 ✓ / `T.vmul` fp32/int32 ✓ / `T.vbrc` int32 ✓ / `T.transpose` fp32/int32 ✓ / `T.vcast` i32→i64 ✓） | `acc+h_flag+row 系列` ≈ 65 KB ≪ UB 192 KB（`CONST-capacity-910B2C`） | **C 轴恒为 8 的整数倍**（C=8/16/64/96/128/2048；fp16 向量宽 8）；`W_in`=55/56/57 仅作 row 长度，不作向量轴 | ✓ 全部编译期静态（`adaptive_bin` trace 期整数；动态量仅运行时标量 `o_h`/`kh` 守卫） | 无跨引擎流水；退化网格可 reduce 融合 | ✅ **选定** |
| 2 可分离 2D | Vector ✓ | 需 + 行部分最大值/索引缓冲（spp CH=64 ≈ 86 KB） | 同 #0' | ✓ | 两阶段串行 | ❌ **运算量更大**（3720 > 3600/平面）+ 额外缓冲 + 跨行合并需要 `dim=0` 归约（`CONST-reduce-dims0-skinny` 记录 `dims=0` 在长条形上逐列串行，灾难性慢）；详见 §1.6.1 |
| 4 退化特化 | Vector reduce ✓ | `(CH_deg,H·W)` 平面块 + 下标/候选/哨兵向量 ≈ 66.6 KB（resnet C4，`CH_deg=min(C,32)=32`，见 §4.5） | ✓ | ✓ | 单归约 | ✅ 采纳（`1×1` / 恒等；`ALG-pooling` 几何特化模式集） |
| 9 逐元素编码 + reduce（一般网格） | Vector ✓ | **UB 不可行**：整窗口驻留需 `O(bh·bw·CH)`；`scan_H·scan_W` 窗（如 10×10）在 CH=64 下 ≈ 256 KB > 192 KB | ✓ | ✓ | 归约 | ❌ 容量否决（`CONST-capacity-910B2C`）；仅 C4 可用（窗口固定 = 整平面且已有容量判据） |
| 1 常数权重卷积 / 6 GEMM | **不适用**（max 非线性，非 MAC 密集） | — | — | — | — | ❌ 数学不等价，无需亲和评估 |
| 7 两遍式 | Vector ✓ | 第二遍需重读 GM 或整窗口驻留 | ✓ | ✓ | 两遍 | ❌ 访存翻倍 + 容量（同 #9） |
| 8 复合键编码 | — | — | — | — | — | ❌ 工程性弃选（非信息论，R1 #8：收益有限 + 位运算 / 并列方向 / NaN 位型代价） |

**R4 弃选理由的证据引用（negative-claim-evidence 口径）**：
- 候选 1/3/5：**数学事实**（`max` 非线性、无逆元）——不作 API 检索需求；候选 8：**工程性弃选**（非信息论，见 R1 #8），
  其位运算包需 `T.vand`/`T.vor`（本设计未采用该路径，故不构成 API 检索需求）。
- 候选 2：运算量为**我方独立复算**；其额外缓冲与 `dims=0` 归约风险引用 `CONST-reduce-dims0-skinny`（已实测常数条目）。
- 候选 9：容量否决引用 `CONST-capacity-910B2C`（UB 192KB + 膨胀系数），并给出 `O(bh·bw·CH)` 估算。
- 候选 0 的「标量占比」反例引用 `PL-1.2-caxis-accum` 诊断触发器（窗口累加 98.3% 标量源于跨步系数阻碍向量化）。
- 候选 0'/4 的 dtype 支持矩阵来自 `docs/Tilelang.language/`：`T.vmax`（§2.2.1 含 int32/fp16/fp32）、
  `T.vcmp`（§2.2.1 含 int32）、`T.vselect`（§2.2.1 含 int32/int64/fp16/fp32/bf16）、
  `T.vmul`（§2.2.1 含 int32）、`T.vbrc`（§2.2.1 含 int32）、`T.transpose`（§2.2.1 含 int32/int64）、
  `T.vcast`（§2.2.1 含 i32→i64 `rint`）、`T.reduce_max`（§2.2.1 含 int32/fp16/fp32）、`T.reduce`（§2.2.1 含 int32）。

**设计期估算（D-2 roofline）** —— 协议见 `_shared/standards/hardware-cost-model.md` §2，常数引用 `pattern-library/constants.md`：

| workload | 流量项（`bytes ÷ 有效带宽`） | 发射项（向量 op 数） | 容量项 | 估算下界 |
|----------|------------------------------|----------------------|--------|---------|
| `global-1x1`(8,2048,7,7)→1×1 | 1 769 472 B ÷ 1.26 TB/s = **1.40 µs**（HBM 3:1 混合地板，`CONST-mte2-degradation`）；L2 驻留口径（sets=1）÷ 2.07 TB/s ≈ 0.85 µs | C4 特化：512 单元（`CH_deg=32`）× ~12 op/单元 ≈ 6.1 K op | 0（无跨引擎往返） | **≈ 1.4 µs** |
| `spp-6x6`(2,128,56,56)→6×6 | 1 697 792 B ÷ 1.26 TB/s = **1.35 µs** | 8 op × 14 400 元素片 + 行开销 ≈ **117 K 向量 op** | 0 | **≈ 1.4 µs** |
| `nondiv-7x7`(2,64,55,57)→7×7 | 865 280 B ÷ 1.26 TB/s = **0.69 µs** | 8 op × 4 800 ≈ **38 K 向量 op** | 0 | **≈ 0.7 µs** |

> **估算下界: ≈ 1.4 µs（流量 1.35–1.40 + 发射「未文档化假设」 + 容量 0，引用 `CONST-mte2-degradation` / `CONST-capacity-910B2C`）；端到端实测地板大概率由固定开销主导（NPU kernel launch + Event 计时 ~50–60 µs，`examples/TileOPs/docs/gpu_to_npu_adaptation.md`）。**
> **发射项口径注**：`CONST-vector-launch-overhead`（`~0.5µs/op`，toolchain 2026-09-09 / 3a214cde）是 **bn=512 宽块大 op** 的实测值，
> 对本设计 ≤128 元素的窄向量 op **不能直接外推**（按 0.5µs/op 会得 ~58 ms 的荒谬值）——窄向量 op 代价**未逐点实测**，标注「未文档化假设」；
> **方向证据已有实测**：`PL-1.4-tiling-heuristic` 同族池化实测「CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs」（更宽 C 向量更优）。
> **容量项为 0**：本设计无 `store_fixpipe` 类跨引擎中间矩阵往返（纯 Vector 单 kernel）。
> **注（索引相对单输出 max 的额外发射代价）**：保持 first-hit / last-NaN 语义需每元素额外 4 条向量 op
> （`vcmp(gt)`+`vselect`+`vmax` 索引、`vcmp(isnan)`+`vselect`+`vmax` NaN 索引，与值路径共用 `acc`/`eq_f32`），
> 故 spp 发射量约为同族单输出 max 的 4×；这是输出索引的**必要代价**（见 R2/R4 与 §1.6.2），非可省项。

**引用的 pattern-library 条目重验状态（`kb_stale_check.py` 口径，本设计会话 2026-10-14 实跑）**：
本会话工具链为 `tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2`，下列被引条目因版本戳落后被标记 `stale`
（`kb_stale_check` 输出 157 条 stale）——按纪律**须注明其支撑方式**，不得当成未经审慎的既成事实：

| 被引条目 | stale? | 本设计如何使用 | 重验 / 支撑方式 |
|---|---|---|---|
| `PL-1.4-tiling-heuristic` | **否（fresh）** | 决定主选 `CH=64` 的**主证据**（CH64 21.6µs < CH32 < CH16） | 版本戳与当前工具链同代；仍列入 §1.6.3 的 D-3 裁决计划做 A/B 复测 |
| `PL-1.1-transpose-chain` / `PL-1.2-caxis-accum` / `PL-1.3-host-permute` | **否（fresh）** | 布局候选 #2/#3 的取舍与核内转置代价 | 同上；布局决策另由 §1.6.3 候选矩阵自洽论证 |
| `TRAP-vselect-inplace-carried-state` | **否（fresh）** | 禁用 in-place `vselect`，改用 scratch + 原地 `vmax`/`vmul` | 同族前作 Stage 3（2026-10-13，近当前工具链）实测通过其绕法 |
| `TRAP-vcmp-ne-ordered-nan` | 是（stamp `db1783d`） | NaN 探测改用 `eq` 取反（§1.6.1 N1） | **本会话独立复核**：`verify_equiv.py` 的 N1 以 `eq` 语义构造并以 fp64 参照逐位对照通过；且同族前作 Stage 3（`db1783d`）在该工具链上实测复现该陷阱并给出合法绕法 |
| `TRAP-tir-range-loop-not-unrolled` | 是（stamp `417dae99`） | `o_w`/`kw` 用 `T.serial` + 守卫而非 `range` | 同族前作 Stage 3 两度实测（avg + max 迁移）确认；本设计保守采用 `T.serial`（即使假设失效也不损失正确性，仅多一层循环） |
| `CONST-capacity-910B2C` | 是（stamp `3a214cde`） | UB 192 KB 容量判据与 §4.5 预算 | 用作**硬件规格级**常数（UB 容量），并以**自洽的逐 buffer 算术**（§4.5，本设计自行求和）+ **编译期容量判据 + 回退一般路径**兜底；预算是本设计自己的算术结论，不单依赖该条目 |
| `CONST-mte2-degradation` | 是（stamp `ed787bb`） | 流量项带宽地板 1.26 TB/s（HBM 3:1 混合） | 用作**量级参照**（流量项 1.35–1.40 µs 与发射项量级差 3 个数量级），结论方向（带宽受限、发射非主导）不因具体数值 ±30% 而翻转；已标注为设计期估算，Stage 4 按 `hardware-cost-model.md` §3 做「估算 vs 实测」对账 |
| `CONST-vector-launch-overhead` | 是（stamp `3a214cde`） | 仅用于**否定其可外推性**（已在估算注中显式标注「未文档化假设」） | 不承载任何决策；CH 决策由 fresh 的 `PL-1.4` 承载 |
| `CONST-reduce-dims0-skinny` | 是（stamp `417dae99`） | 候选 2（可分离 2D）弃选的**次要素**（主因是运算量更大：3720 > 3600） | 弃选不依赖该条目：即使 `dims=0` 无惩罚，候选 2 仍因运算量更大 + 额外缓冲被弃 |

> 结论：**本设计的承载决策的条目（`PL-1.4` 与布局族 `PL-1.1/1.2/1.3`、`TRAP-vselect-inplace-carried-state`）均为 fresh**；
> stale 条目或只作**硬件规格级/量级级**参照（并有本设计自洽算术或回退路径兜底），或**不承载决策**（`CONST-vector-launch-overhead`），
> 或已由**本会话机器验证 / 同族 Stage 3 实测**独立佐证（`TRAP-vcmp-ne-ordered-nan`、`TRAP-tir-range-loop-not-unrolled`）。

**调研结论**：选定算法族 = **「单遍在线状态机 + fp32 比较域 + C 轴向量化 + 显式向量 NaN 追踪 + 向量化 first-hit/last-NaN 索引追踪 + 核内融合转置（NCHW→C 最内）」**
（R3 表候选 #0'；另采纳候选 #4 退化特化 → §1.6.1 **C4/C4b**，候选 #6 融合扫描并入 #0'，候选 #9 仅用于 C4；
两项采纳与 §1.6.1 优化结论**逐项一致**）。
关键依据：R3 表显示 #0' 的 FLOPs（= `N·C·scan_H·scan_W`，与 manifest `roofline.flops` 逐字一致）与
访存量（= manifest `roofline.bytes`，含 int64 索引的 8 B/elem 项）均为各候选最小量级；候选 2 不降反升且引入 `dims=0` 归约风险；
候选 1/3/5 数学上不可行、候选 8 工程性弃选（非信息论，见 R1 #8）；R4 表显示 #0' 唯一同时满足「Vector 计算单元匹配 + C 轴向量整除 + 全静态边界 + 无跨核归约 + 容量充裕」。
**与基线的结构差异一句话**：数学公式、取值集合、访问顺序与 NaN/并列判决完全不变，执行单元由「每输出一线程的标量状态机」
改为「`(n,o_h,c_block)` 单元的 C 轴向量比较器 + 三路向量累加器（值 / first-hit 索引 / last-NaN 索引）」。
**调研范围声明**（确有两类结构特化被采纳，但**无更低复杂度算法族**）：
本调研覆盖 ① `algorithm-candidates.md` 的 `ALG-pooling` 命中行全部候选（R1 常数权重卷积 / sum-pool×常数 / R2 滑窗流式 /
R3 窗口重叠重复读 / R4 C 轴向量轴 / 几何特化模式集）——并**逐项判定其对 max 与对索引输出的适用性**；
② `ALG-reduction` 与 argmax 族（`examples/TileOPs/tileops/kernels/reduction/argmax/`，`CASE-argmax-argreduce-stage4`）的索引输出候选
（掩码候选 + `reduce_min` first-occurrence、在线 running 递推、`T.deinterleave` 窄 N 分解——其中「掩码候选 + 归约」形态被采纳用于 C4）；
③ 恒等变形空间（可分离两轴、前缀最大值差分、两遍式、复合键编码、归约融合）；
④ pattern-library 同类实现（`PL-1.1-transpose-chain` / `PL-1.2-caxis-accum` / `PL-1.3-host-permute` / `PL-1.4-tiling-heuristic` /
`CONST-vector-launch-overhead` / `CONST-reduce-dims0-skinny` / `TRAP-vselect-inplace-carried-state` / `TRAP-vcmp-ne-ordered-nan`）；
⑤ 本仓 `examples/` 同类实现（§3.5.3）与**同族前作** `examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/`
（单输出 max 迁移，CASE-pool-adaptivemaxpool2d-migration）与 `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/`（avg 迁移）。
**互联网检索未使用**（本算子算法族在本地参考表 + 源码 + argmax 族实现 + pattern-library 已完整覆盖，属窗口/归约类，无需联网补充）。
**下游约束**：§1.4 算法描述、§1.6.1–§1.6.3 的分析对象 = 选定算法 #0'；§0.5/§0.6 的处置与之**一致**
（§0.6 R2 = 值/NaN 向量化、§0.6 R3 = 索引向量化、§0.6 R4 = C 轴布局、§0.6 R5 = 候选 #4 + #9）；
源算法 §0.4 已识别优化手段的意图承接逐项见 §0.6（**无静默丢弃**：14 项全部有处置）。
#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定的算法 #0'（= 基线单遍在线状态机 + C 轴向量化）。调研候选（R1 表）中初判可采纳的项在此完成四要素论证。
> 索引相关的 **N1/N2/N3** 是公式级重写（把「顺序覆盖状态」改写为「可交换归约」），是使索引在 NPU 上可向量化的**前提**。

| # | 优化项 | 原式（源/基线） | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| N1 | **NaN 标记：顺序 OR → 乘性 eq 掩码** | $\text{has\_nan}=\bigvee_{k}\ \text{isnan}(v_k)$（标量 bool 顺序或） | $h=\prod_k \text{eq}(v_k,v_k)\in\{0,1\}$，$\text{has\_nan} = (h=0)$；其中 $\text{eq}(v,v)=1$ 当且仅当 $v$ 非 NaN | **数学恒等**：$\text{eq}$ 是 `isnan` 的 IEEE 补集，对 $\{0,1\}$ 掩码 $\prod_k m_k = 0 \iff \exists k: m_k=0$（De Morgan 的乘性形式）；`0` 吸收、`1` 中性 | 把每元素标量或 + `isnan` 替换为整 tile `T.vcmp(eq)`/`T.vcast`（每行 2 op）+ 每元素 1 条原地 `T.vmul`；消除数据相关分支 |
| N2 | **last-NaN 索引：顺序覆盖 → 位置集合上取 max** | $\text{nan\_idx}$：对每个 NaN 顺序覆写为当前扁平下标（最后一次覆写生效） | $\text{nan\_idx}=\max\{\,f_k:\text{isnan}(v_k)\,\}$，无 NaN 时为哨兵 $-1$ | **数学恒等**：访问序按扁平下标**严格递增**，故「最后一次覆写」= NaN 位置集合中扁平最大者；$\max$ 可交换/可结合，与扫描顺序无关 | 把「数据相关分支覆写」替换为 `T.vselect(isnan, f, -1)` + 原地 `T.vmax`（每元素 2 op），可向量化且无分支 |
| N3 | **first-hit 最大索引：顺序条件覆写 → 破纪录集合上取 max（并消解 `first_valid`）** | $\text{max\_idx}$：`elif val > max_val: max_idx = f`（最后一条记录生效）；另有 `first_valid` 分支处理首个有效元素 | $\text{max\_idx}=\max\big(f_0,\ \max\{f_k: v_k > \max_{j<k} v_j\}\big)$，其中 $f_0$ = 首个被访问元素的扁平下标 | **数学恒等**：源的 `max_val` 单调不减，故「记录」= 严格超过此前最大值的元素，记录按扁平**严格递增**出现；源的最后一次记录 = 值首次达到窗口最终最大值的元素 = 破纪录集合里扁平最大者。并列（相等）时 `v > max_val` 为假 ⇒ 不覆写 ⇒ 先访问者胜出，与式中的严格 `>` 一致。`first_valid` 分支在触发时写出的正是「首个被访问元素」的值与下标；把 `max_idx` 初值置为 $f_0$ 后，全 `-inf` 窗口（无任何记录）自然得到 $f_0$，而任何记录的下标 $\ge f_0$（窗口内扁平最小值即 $f_0$）故初值不被污染 ⇒ 与源逐位一致 | 替换标量条件覆写 + `first_valid` 分支为 `T.vselect(gt, f, -1)` + 原地 `T.vmax`（每元素 2 op）；**消解一个状态变量**（`first_valid`），减少寄存器/分支。**实现次序约束（关键）**：`gt = (v > acc)` 必须用**更新前**的值累加器计算，即 `T.vcmp(gt)` 必须先于 `T.vmax(acc)` 执行（否则判据 `v > max(a,v)` 恒假 ⇒ `max_idx` 退化为 $f_0$，见 §3.3/§6.2「次序约束」）；该约束由 `N3-ord` 守卫机器化 |
| C4 | **退化网格结构特化：全局池化（`out_H==1 and out_W==1`）** | 一般 bin 双循环取最大值与索引（`bin(0,0)` 由通用公式给出 $\max_{kh<H}\max_{kw<W}x$，索引为逐元素状态机结果） | 值 $=\max_{\text{all}}x$（整平面 flatten 后单次 `T.reduce_max` 链）；索引 $=\begin{cases}\max\{k:\text{isnan}\}& \text{if-any-NaN}\\ \min\{k: v_k=\max\}& \text{else}\end{cases}$（`vselect` + `reduce_min/max` + 电平判决）；NaN 判决用 eq 掩码 `reduce_min` 独立得出 | **数学恒等**：`out_H=out_W=1` 时 $\lfloor 0\cdot H/1\rfloor=0$、$\lceil 1\cdot H/1\rceil=H$，故 `bin(0,0)=[0,H)×[0,W)` = 整平面；取值集合与访问序（扁平递增）完全相同 ⇒ 值与索引均**逐位等价**（max/argmax 是选择，无算术）。索引的两种形态正是 N2/N3 在整平面上的闭式表达 | `global-1x1`（`H=W=7`）：每单元 $H{\times}W=49$ 次向量 `vmax` ×(值+索引 op) → **1 次向量归约 + 1 次掩码归约**（≈ 15×）；覆盖 3 组 manifest workload 中单元数最多者（512 单元，`CH_deg=32`） |
| C4b | **退化网格结构特化：恒等（`out_H==H_in and out_W==W_in`）** | 一般 bin 双循环（每个 bin 大小 1，max 即元素本身、索引即该元素扁平下标） | `out = x`（纯 UB 中转拷贝，同 dtype 不 cast）；`indices` = 与 `n`/`c` 无关的**平面扁平下标常量模式** `k=ih*W_in+iw`（`T.arange` 1-D → reshape → `vbrc` → int64） | **数学恒等**：`out_H=H_in` 时 `bin_H(o_h)=[o_h,o_h+1)`（$\lfloor o_h\rfloor=o_h$、$\lceil o_h+1\rceil=o_h+1$），单位 bin 的最大值 = 元素本身、其扁平下标 = `ih*W+iw`；逐元素恒等（NaN 也随元素原样搬出，值语义一致；索引为常量模式，与元素值无关） | 覆盖 op 层 `output_size=None` 的恒等用例与 `(None,…)`/`(…,None)` 半恒等轴；消除全部归约与全部索引累加 |

> **无「倒数乘 / 归一化」优化空间**（显式结论）：与同族 avg 池化不同，max 池化**无除法/归一化**（结果即被选元素），
> 故 avg 迁移的 `sum/d → sum·(1/d)` 在本算子**不存在**——§1.6.1 无对应项，非遗漏。
> **无公共子表达式消除**：每元素仅参与一次比较与一次掩码更新（`eq_f32` 行级已复用，`fvec` 行内复用）。
> **无高代价算子降代换**（无 exp/rsqrt 类）。
> **检索新增**：R1 #8 的「复合键编码」（用一次 `vmax` 同时更新值与索引）为**工程性弃选**（**非**信息论不可行：
> fp16/bf16→fp32 低 13/16 位恒零，平面索引在本算子输入域内可塞入低位；弃选依据为收益有限 + 位运算/并列方向/NaN 位型代价，见 §1.6.0 R1 #8），
> **非遗漏**；R1 #7 两遍式因访存翻倍弃选。

**等价性机器验证（D-1）**：全部采纳优化项由 `examples/adaptive_max_pool2d_with_indices/_adaptive_max_pool2d_with_indices_kernel/verify_equiv.py`
在 **fp64 参照**下对照基线（源顺序状态机，含 `first_valid` 分支）验证（torch CPU）。共 **7 项 × 用例矩阵 × 2 dtype**：
- **N1 / N2 / N3 / N3b**：语义重写项，判据 = 值舍入到目标 dtype 后与基线**逐位一致**（NaN-vs-NaN 计相等、NaN-vs-finite 恒判失败）
  **且索引逐元素完全相等**（整数精确比对，`VP-2026-0131`：int64 索引 `torch.equal` 是免费 oracle）。
  `N3b` = 与 `N3` 同式但把值累加器换成 **NaN 忽略**约定（`fmax`），验证结果**不依赖 `T.vmax` 的未文档化 NaN 行为**。
- **N3-ord**：**实现次序回归守卫**（非公式项，覆盖 v0 的阻塞级 1 缺陷）——① 正确次序（`gt` 与**更新前**累加器比较）必须复现基线；
  ② **值优先次序**（`gt` 与**更新后**累加器比较，即 v0 §3.3/§6.2 形态）必须在**一切「真 first-hit 离开窗口首个被访问元素」的非 NaN 窗口**上被检出
  （即错误次序给出的索引必 ≠ 基线索引）；另含硬编码反证用例 `arange(12).reshape(1,1,3,4) → (2,2)` 的 `bin(1,1)`：
  正确索引 **11**（窗口最大在扁平 11），v0 次序退化为种子 **6**（= `1*4+2`）。该守卫以「错误次序必须被检出」的**双向断言**给出，
  而非负向断言——D-1 层即可拦截同类次序回归。
- **C4 / C4b**：结构特化项，判据 = 值与索引均与**基线一般网格公式**逐位一致。
- 用例矩阵：3 组 shape 群（general 9 组 / global 4 组 / identity 4 组，共 17 组 shape×out 组合，含非整除重叠、扩张 bin、
  部分 `None`、单输出列、通道尾块、`1×1` 极端）× 10 种数据（`rand` / `nan` / `nan_first`（首元素即 NaN）/
  `inf`（同窗 `{+inf,-inf}`）/ `all_equal`（全并列）/ `all_neg_inf` / `tie_two_max` / `subnormal` / `large` / `zeros`）
  × 2 dtype（fp16 / bf16）。
- **运行结论**：全部 **7 项 `EQUIV_PASS`，worst `max_abs_diff = 0.0`，0 mismatch**。结果表内嵌如下：

```
item   | checks | worst ulp-ish diff |  miss | viol_rate | status
----------------------------------------------------------------------
N1     |    340 |            0.0e+00 |     0 |     0.0% | EQUIV_PASS
N2     |    340 |            0.0e+00 |     0 |     0.0% | EQUIV_PASS
N3     |    340 |            0.0e+00 |     0 |     0.0% | EQUIV_PASS
N3b    |    340 |            0.0e+00 |     0 |     0.0% | EQUIV_PASS
N3-ord |    342 |            0.0e+00 |     0 |     0.0% | EQUIV_PASS
C4     |    120 |            0.0e+00 |     0 |     0.0% | EQUIV_PASS
C4b    |    120 |            0.0e+00 |     0 |     0.0% | EQUIV_PASS
----------------------------------------------------------------------
criterion: exact (0 mismatches) after rounding to the target dtype (value diff = 0.0, i.e. no ULP deviation); NaN-vs-NaN counts equal, NaN-vs-finite fails; indices compared exactly.
total (shape, out, kind) cases swept: 340 x 2 dtypes
[EQUIV] ALL PASS
```

> 列口径：`worst ulp-ish diff` = 值在目标 dtype 舍入后与基线的最大绝对差（本算子为**选择**，恒为 `0.0e+00`，即 0 ULP）；
> `miss` = 值不一致 + 索引不一致的元素总数；`viol_rate` = `miss / checks`；全项 0 → `EQUIV_PASS`。

> `checks` = 该判定项被实际评估的 (shape,out,kind,dtype) 用例数（N1/N2/N3/N3b 覆盖 340 例；`N3-ord` = 340 例 + 2 条硬编码反证检查 = 342；
> C4/C4b 仅在其适用的 shape 群上评估，各 120 例）。完整逐条输出可复现：
> `/workspace/venv/bin/python examples/adaptive_max_pool2d_with_indices/_adaptive_max_pool2d_with_indices_kernel/verify_equiv.py`
> （脚本随算子目录留存，Stage 2 维度 8 会重跑复核）。
> **守卫非空性证据**（2026-10-14 本设计会话）：general 矩阵（fp64）× 10 种数据上，满足「真 first-hit ≠ 窗口首元素且非 NaN」的
> 输出元素共 **9779 个**，全部被 `N3-ord` 用来核对「错误次序必被检出」；硬编码反证用例 `arange(12)→(2,2)` 的 `bin(1,1)` 实测
> 基线 = 11、正确次序 = 11、值优先次序 = 6。

**优化结论**：采纳 **5 项**——**N1**（乘性 eq 掩码）、**N2**（last-NaN = 位置集合 max）、**N3**（first-hit = 破纪录集合 max，消解 `first_valid`）、
**C4**（全局池化特化）、**C4b**（恒等特化）；外加 **N3b** 作为稳健性对照（同式不同 NaN 约定，亦 PASS）与
**N3-ord** 作为**实现次序回归守卫**（正确次序复现基线 + 值优先次序必被检出，亦 PASS）。
候选 C2（可分离 2D）经量化后不采纳（每平面 3720 > 3600，运算量更大 + 额外缓冲 + `dims=0` 归约风险）；
候选 #3/#5（前缀最大值差分 / sum-pool×常数）与 #1（常数权重卷积）因**数学不等价**不采纳；
候选 #7（两遍式）因访存翻倍不采纳；候选 #8（复合键编码）因工程性代价（收益有限 + 位运算/并列方向/NaN 位型风险，见 §1.6.0 R1 #8）不采纳。
**「倒数乘」优化在本算子不存在**（max 无归一化）。**优化后公式（供 §3.1 公式拆解使用）**：

```
一般网格 (out_H×out_W，非退化)；行主序访问序 (kh 外, kw 内)，扁平 f = ih*W_in + iw：
  值:      acc      = max over window of cast_f32(x)                    # 初值 -inf
  索引:    cfinal   = (nan_idx >= 0)                  # 见下，NaN 判决
           max_idx  = max(f(ih0,iw0), max{ f : cast_f32(x) > running-max })
           nan_idx  = max{ f : isnan(x) }             # 无 NaN 时 = -1
           out      = cfinal ? NaN : acc
           indices  = cfinal ? nan_idx : max_idx      # 再 cast int64
退化网格:
  out_H==1 and out_W==1       :  out = max(plane);  indices = (含NaN) ? max{k:isnan} : min{k: v_k=max}
  out_H==H_in and out_W==W_in :  out = x;          indices = ih*W_in + iw
```

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点实现方案（§1.4 + §3.3 伪代码）中**全部**循环与标量计算点，逐项判定能否用向量操作等价替代。
> 替代方案所用 API 均以 `docs/Tilelang.language/` 条目佐证（下表「佐证」列）。

| # | 计算点 | 原实现形态（源 GPU） | 向量替代方案 | 是否替代 | 不可替代理由 / 佐证 |
|---|--------|---------------------|-------------|---------|--------------------|
| 0 | **累加器 / 追踪器初始化**（`acc ← -inf`、`h_flag ← 1`、`max_idx ← f0`、`nan_idx ← -1`） | 源 `max_val = cast(-inf)`、`has_nan=False`、`max_idx=cast(0)`、`nan_idx=cast(0)`、`first_valid=True`（每输出元素 1 次标量） | `T.vbrc(-T.infinity("float32"), acc)`、`T.vbrc(1.0, h_flag)`、`T.vbrc(ih0*W_in+iw0, max_idx[ow:ow+1,:])`、`T.vbrc(-1, nan_idx[ow:ow+1,:])` | ✅ | 佐证：`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.1（int32/fp32 ✓）、§2.2.2（scalar→任意 shape）；`创建操作/T.infinity.md` |
| 1 | bin 取最大值 `max_val = T.max(max_val, val)` | 逐元素标量比较（每输出元素一条） | `T.vmax(acc[ow], row_f32[iw,:], acc[ow])`（C 轴向量，原地） | ✅ | 佐证：`docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1 fp32 ✓；`PL-1.2-caxis-accum`。**次序约束（关键）**：该原地 `vmax(acc)` 必须在 #4 的 `vcmp(gt)` **之后**执行——`gt` 恒须与**更新前**的 `acc` 比较，否则判据 `v > max(a,v)` 恒假 ⇒ `max_idx` 退化为初值 `f0`（v0 缺陷；见 §1.6.1 N3 与 §9.2） |
| 2 | 逐元素 `T.cast(x[...],"float")` | 每元素一次标量 cast | `T.vcast(row_dst, row_f32, round_mode="rint")`（整 tile） | ✅ | 佐证：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（f16/bf16→f32 `rint` ✓） |
| 3 | **NaN 标记** `has_nan = has_nan \| T.isnan(val)` | 每元素标量 `isnan` + 标量 bool 或（本变体第 180–185 行：`is_nan = T.isnan(val); if is_nan: nan_idx = flat_idx; has_nan = True`；单输出变体为第 62 行 `has_nan = has_nan \| T.isnan(val)`，语义等价） | 行级 `T.vcmp(row_f32,row_f32,eq_mask,"eq")` + `T.vcast(eq_mask,eq_f32,"rint")`；每元素 `T.vmul(h_flag, eq_f32[iw], h_flag)` | ✅ | 佐证：`比较操作/T.vcmp.md` §2.2.1（fp32 ✓，`eq` 为 IEEE 正确）；`数学操作/T.vmul.md` §2.2.1（fp32 ✓）。**语义正确性**：`eq` 为 `isnan` 的 IEEE 补集（`ne` 是 ordered-NE，NaN 恒假不可用，见 `TRAP-vcmp-ne-ordered-nan`） |
| 4 | **first-hit 索引** `elif val>max_val: max_idx=flat` | 每元素标量条件覆写 | `T.vcmp(vs, acc[ow], gt, "gt")` + `T.vselect(gt, fv, neg1, cand)` + `T.vmax(max_idx[ow], cand, max_idx[ow])`（int32） | ✅ | 佐证：`比较操作/T.vcmp.md` §2.2.1（int32/fp32 ✓，"gt" 模式）、`条件操作/T.vselect.md` §2.2.1（int32 ✓）、`比较操作/T.vmax.md` §2.2.1（int32 ✓）。**等价性**见 §1.6.1 N3（last-record = 破纪录集合 max；`verify_equiv.py` N3/N3b 全 PASS）。**次序约束（关键）**：这三条 op 必须在 #1 的原地 `vmax(acc)` **之前**执行（`gt` 恒与更新前的 `acc` 比较）；`verify_equiv.py` 的 `N3-ord` 守卫（正确次序复现基线 + 值优先次序必被检出）机器化该约束 |
| 5 | **last-NaN 索引** `if is_nan: nan_idx=flat` | 每元素标量条件覆写 | `T.vcmp(eq_f32[iw], zero_elem, isnan, "eq")` + `T.vselect(isnan, fv, neg1, nanc)` + `T.vmax(nan_idx[ow], nanc, nan_idx[ow])`（int32） | ✅ | 佐证：同 #4。**等价性**见 §1.6.1 N2（位置集合 max） |
| 6 | **扁平下标广播** `flat_idx = cast(ih*w_in+iw, idx_dtype)` | 每元素一次标量乘加 | `T.vbrc((ih0+kh)*W_in + iw, fvec)`（int32 标量 → `(1,CH)` 向量） | ✅ | 佐证：`shape操作/T.vbrc.md` §2.2.1（int32 ✓）、§2.2.2（标量 src 无 shape 限制） |
| 7 | 窗口级 NaN 判决 `if_then_else(has_nan, NaN, max_val)` / `has_nan?nan_idx:max_idx` | 每元素标量分支 | `T.vcmp(nan_idx, zeros, cfinal, "ge")` + `T.vselect(cfinal, nanbuf, acc, out32)`（值）+ `T.vselect(cfinal, nan_idx, max_idx, idx32)`（索引） | ✅ | 佐证：`条件操作/T.vselect.md` §2.2.1（fp32/int32 ✓）；`nanbuf = inf - inf`（`数学操作/T.vsub.md` §2.2.1 fp32 ✓） |
| 8 | 最终输出 cast（值 + 索引） | 每元素一次标量 cast | 值 `T.vcast(out32, acc16, "rint")`；索引 `T.transpose` 后 `T.vcast(idx_t, idx64, "rint")`（i32→i64） | ✅ | 佐证：`数据类型转换操作/T.vcast.md` §2.2.1（f32→f16 ✓、**i32→i64 `rint` ✓**） |
| 9 | 退化网格全平面归约（值 + 索引） | （源无此路径） | 值 `T.reduce_max(flat32, mid, dim=2)` + `T.reduce_max(mid, gmax, dim=1)`；索引 `T.vselect(eq, k_j, bigk)`+`T.reduce_min(dim=2→1)`（first-hit）与 `T.vselect(mb, k_j, neg1w)`+`T.reduce_max(dim=2→1)`（last-NaN） | ✅ | 佐证：`规约操作/T.reduce_max.md` §2.2.1（fp32 ✓，`clear=True`）；`规约操作/T.reduce.md` §2.1（`reduce_min` + int32 ✓）；`创建操作/T.arange.md` §2.3.2（1-D arange + reshape + vbrc 合法形态） |
| 10 | 退化网格 NaN 探测（C4） | （源无此路径） | `T.vcmp(flat32,flat32,mb,"eq")` + `T.vcast(mb,mf32,"rint")` + `T.reduce_min` 链得 `nmin ∈ {0,1}` | ✅ | 佐证：`T.vcast` §2.2.1 `i1→f32 rint ✓`；掩码为 {0,1} 无 NaN，归约行为无关 |
| 11 | 退化网格索引常量模式（C4b） | （源无此路径） | `T.arange(idx1d (H*W,),[1],0)` → `T.reshape` → `(1,1,H,W)` → `T.vbrc` → `(CH,1,H,W)` → `T.reshape` → `(CH,H,W)` → `T.vcast` int64 | ✅ | 佐证：`创建操作/T.arange.md` §2.2.1（int32 ✓）、§2.3.2（1-D 优先，禁 2-D 标量物化）；`shape操作/T.reshape.md`（int32 ✓ 元数据级）；`shape操作/T.vbrc.md` §2.2.2（rank 一致 + 单维 1 广播） |
| 12 | 布局重排（C→最内 与 回转） | （源无；GPU 靠 coalesced 尾轴） | `T.transpose(src, dst, [1,0])`（值 fp32、索引 int32 各一次） | ✅ | 佐证：`创建操作/T.transpose.md` §2.3（二轴交换 ✓，fp16/fp32/**int32/int64** ✓）；`PL-1.1-transpose-chain` |
| 13 | **bin 的 H 方向循环 `for kh in T.serial(max_bin_h)`** | `T.serial` 静态界 + 守卫（源同款） | 无（保持 `T.serial` + `if kh < bh` 守卫） | ❌ | **理由（tile 级顺序依赖 + 动态边界）**：`kh` 是**三路累加器的循环携带维**（跨 `kh` 累加到同一 `acc`/索引累加器），且 bin 上界 `bin_h(o_h)` 依赖运行时解码的 `o_h`（**依赖动态 index 的边界**）；TileLang 只接受静态 `T.serial` 边界（且 `TRAP-tir-range-loop-not-unrolled` 表明 Python `range` 亦降为串行 TIR 循环）。**但循环体内全部计算已向量化**（#1–#6），故该循环是 §6 的 block/tile 级调度循环，非逐元素标量循环。 |
| 14 | **bin 的 W 方向循环 `for kw in T.serial(max_bin_w)`** | 内层标量比较 | 循环体内为向量 op（#1/#3/#4/#5）；循环本身保持 `T.serial` + `if kw < bw` 守卫 | ❌ | **理由（tile 级顺序依赖 + 动态边界）**：同 #13（`bw = iw1-iw0` 依赖运行时 `ow`）；`TRAP-tir-range-loop-not-unrolled` 已实证 Python `range` 不展开为向量 op 序列，故与 `kh` 同形处理。**循环体内已向量化**，非标量热点。 |
| 15 | 单元任务解码 `(n, o_h, c_block)` | — | 无 | ❌ | **理由（block 级索引 / 任务映射计算）**：每单元少量整数 `%`/`//`，非逐元素热点；无对应逐元素向量 API |
| 16 | **bin 循环静态上界 `MAX_BIN_H`/`MAX_BIN_W`**（`max_adaptive_bin_extent`） | trace 期 Python 整数（编译期常数） | 无 | ❌ | **理由（host/trace 侧元数据）**：`max_adaptive_bin_extent` 在 factory tracing 期求值为 Python 整数，是**编译期常数**（TileLang 只接受静态 `T.serial` 边界）；不在 kernel 运行时逐元素执行 |
| 17 | **运行时标量 bin 边界算术** `adaptive_bin(o_h,…)` / `bin_h(o_h)` / `bw = iw1-iw0` | （源同款，第 158–159 行）逐元素 `div/mod` | 无 | ❌ | **理由（block/索引级标量 TIR 算术 + 依赖运行时解码值）**：`o_h`（由 `uid` 运行时解码）与 `ow`（`T.serial` 变量）使 `ih0/ih1`/`iw0/iw1` 为**核内运行时标量**（`adaptive_bin` 的整数 div/mod，见 `docs/Tilelang.language/` 无数值级向量 API 可表达「随 index 变化的边界」）；其值供 `T.copy` 切片与 `if kh < bh` 守卫使用，**非逐元素数据热点**，每单元仅 `O(out_W + 1)` 次 |
| 18 | **输出列调度循环 `for ow in T.serial(out_W)`** | 任务内 `o_w` 调度（每单元 `out_W` 次） | 无（保持 `T.serial(out_W)`） | ❌ | **理由（tile 级调度 + 运行时候选边界）**：循环内 `iw0, iw1 = adaptive_bin(ow, W_in, out_W)`（见 #17）依赖运行时 `ow`；`out_W` 为静态小值（本算子 1–13，非逐元素维）；`TRAP-tir-range-loop-not-unrolled` 表明 Python `range` 亦降为串行 TIR 循环，同族于 #13/#14。**循环体内全为向量 op**（#0/#1/#4/#5/#6），非标量热点 |
| 19 | host 侧 `unsqueeze/squeeze/contiguous` + 双输出 `squeeze(0)` | — | 无 | ❌ | **理由（host 侧元数据）**：只改 shape/stride 视图，符合 `ascend-constraints.md` §4；不在 kernel 内 |

**向量化结论**：逐元素计算（累加器/追踪器初始化 / 取最大值 / NaN 掩码 / first-hit 索引 / last-NaN 索引 / 扁平下标构造 /
NaN 判决 / cast / 归约 / arange 常量模式 / 布局重排）**已全部向量化**，替代方案 API 均有 `docs/` 条目佐证。
保留 **7 处** 循环 / 标量点，理由类别：**#13/#14** tile 级顺序依赖 + 动态边界（循环体内已向量化）/
**#15** block 级任务映射 / **#16** host/trace 侧编译期常数 / **#17** 运行时标量 bin 边界算术（块/索引级标量 TIR）/
**#18** 输出列 tile 级调度循环（循环体内已向量化）/ **#19** host 侧视图操作。
无「以实现简单为由保留标量循环」的情形。**§6 循环结构中不出现逐元素标量循环**（与本结论一致；#13/#14/#18 的循环体均为 v-prefix 向量 op）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> **背景**：I/O layout 是契约（NCHW 进 / NCHW 双出），但**核内布局与 lane 映射是自由变量**。本节枚举「窗口/池化类」必选候选
> （reference：algorithm-research.md Phase 2 第 3 项「I/O 原生布局 + 最内连续轴」与「核内重排布局 + 高整除性轴」）。
> 迁移任务：源的并行轴选择（GPU 逐元素 lane）是**输入而非结论**，对照 §0.3 独立评估。

**轴质量评分维度**：

| 评分项 | 说明 | 候选 1（NCHW / W 轴） | 候选 2（核内 NHWC / C 轴） | 候选 3（host permute / C 轴） | 候选 4（窗口 tile 驻留） |
|--------|------|----------------------|---------------------------|------------------------------|------------------------|
| 整除性 | 轴长 vs 向量宽度（fp16 ×8） | `W_in`=55/56/57：56 ✓，55/57 ✗（尾 lane 浪费 1.8%/1.4%）；且**窗口宽 bin_w≈9–10 非 8 倍数** | **C 恒为 8 的整数倍**（8/16/64/96/128/2048）✓ | 同候选 2 | 同候选 2（向量仍是 C） |
| 尾 lane 浪费率 | `(ceil(len/vw)·vw − len)/len` | W 轴 3.6%（55→56）；**段长 9–10 → ~12%** | **0%** | 0% | 0% |
| 累加链形态 | 该轴作串行累加维（stride≠1）还是向量维（stride=1） | W 轴作段内归约需**变长段最大值**（非单一 op）或前缀最大（不可行） | C 轴作向量维（stride=1，转置后连续）；空间轴作串行归约维（`kh`/`kw`）——`max`/argmax 结构天然 | 同候选 2 | 窗口 tile `(bh,bw,CH)`，空间归约用 `reduce_min/max`（需 `dim=2→1` 链） |
| repack 代价 | 布局重排代价 | 无 repack，但需变长段处理 | 核内融合转置链（`T.transpose` 二轴交换，`PL-1.1` 实测 ~5.4µs / 4 次） | **host permute 106–147µs**（`PL-1.3`，已量化证伪） | 无 repack，但窗口 gather 为跨步拷贝（`bw < W_in`） |
| UB 容量影响 | 布局对驻留缓冲的影响 | 小（row `(W_in)` 级） | `row(·,CH)` + `acc/out32/zeros/infbuf/nanbuf` + 索引 4 组 + `row_f32/eq_f32` ≈ 65 KB | 同候选 2 | 需窗口整体驻留 `O(bh·bw·CH)`（`scan_H·scan_W` 窗在 CH=64 ≈ 256 KB）→ 超 UB |

**候选矩阵**（全部枚举，弃选必须给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益 / 代价 | 是否采纳 |
|---|---------|---------|------------|----------------|---------|
| 1 | I/O 原生 NCHW | W（行内连续） | 无 | 基线：窗口段长 9–10 非 8 倍数、需变长段最大值（`T.reduce_max` 要求静态形状；变长段需逐元素或 gather）；`W_in`=55/57 非对齐；索引需在 W 轴做段内 argmax（变长段 + `dims=0` 归约风险） | ❌ 量化理由：段长 ~12% 尾 lane 浪费 + 无单一向量 op 表达变长段归约，且 `T.reduce_max` 不支持 bf16（`docs/Tilelang.language/规约操作/T.reduce_max.md` §2.2.1） |
| 2 | **核内 NHWC（融合转置链）** | **C（恒为 8 的整数倍）** | 逐行二轴 `T.transpose(src,dst,[1,0])`（`PL-1.1` 族，实测 ~µs 级） | 尾 lane 0%；转置代价 ~µs 级（远低于算子本体）；UB 增量 ≈ 65 KB（§4.5） | ✅ **选定** |
| 3 | host permute NCHW↔NHWC 进出 | C | 两次 host permute | **106–147µs**（`PL-1.3`，实测）≥ 多数中等算子本体；host 侧连续化会产生新张量（无收益） | ❌ 量化理由：repack 代价 > 算子本体流量（估算 ~1.4µs 流量项）两个数量级，净亏 |
| 4 | Cube/implicit-GEMM（0/1 权重矩阵） | Cube N 轴（out_W） | 权重矩阵构造 | **数学不可行**（max 非线性，无 0/1 权重矩阵可表达；索引更无法线性化）；且 `out_W=6/7 < 16` 违 Cube 分形下限 | ❌ 见 §1.6.0 R1/R4（数学不等价优先于此） |
| 5 | 窗口 tile 驻留 + `reduce_min/max` argmax（argmax 族形态） | C（窗口 tile `(bh,bw,CH)`） | 窗口 gather（跨步拷贝 `bw<W_in`） | 一般网格 UB 不可行：`scan_H·scan_W` 窗在 CH=64 ≈ 256 KB > 192 KB（`CONST-capacity-910B2C`）；若改为逐输出元素 gather 则 GM 过读 `W_in/bw` 倍（spp ≈ 5.6×） | ❌ 量化理由：容量否决 + 过读；**仅在 C4（窗口固定为整平面且已有容量判据）采纳**（`ALG-pooling` 几何特化模式集） |

**布局决策结论**：选定 **方案 #2：核内布局 = 逐行 NHWC（C 最内连续）、向量化轴 = C、repack 路径 = 核内逐行二轴 `T.transpose`（`[1,0]`）+ 输出侧值与索引各自 `T.transpose([1,0])` 回转**。
该结论已同步至：§3.3 伪代码（`row_src/row_dst/row_f32/eq_f32/acc/h_flag/max_idx/nan_idx` 形状与内层向量维 = C）、
§4 内存规划（§4.3/§4.5 的 buffer 形状与预算）、§6 循环结构（累加内层向量维 = C）。

**D-3 实验裁决模式（主选 + 备选 + 裁决计划）**：主选与备选的分歧落在**窄向量 op 的每-op 发射代价**这一**未实测外推区间**——
`CONST-vector-launch-overhead` 只给了 bn=512 大 op 的 0.5µs/op 实测值，本设计 ≤128 元素的窄 op 未逐点实测；
且本算子每元素比同族单输出 max 多 4 条向量 op（索引追踪），op 数放大使该外推更敏感（§1.6.0 发射项注）。
判定裕度（`CH=64` 少一半 op 但核数少一半）落在该外推的不确定区间内，故保留备选与裁决计划。
**方向证据**：`PL-1.4-tiling-heuristic` 直接实测「CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs——更宽 C 向量 + 更少 block 胜过更高空间并行度」，
与 `CONST-vector-launch-overhead`「宽块 op 恒优于窄块 op 翻倍」方向一致；**主选 `CH=64` 恰为该池化族已实测范围（CH16/32/64）内的最优档**，
故主选以实测条目为主证据，不属「未实证的纸面单选」。

- **主选（进 Stage 3）**：一般网格 `CH = min(C, 64)`（8 的整数倍），逻辑单元 = `(n, o_h, c_block)`，`num_logical > vector_cores` 时 persistent（见 §5.5）；
  **退化路径（C4/C4b）另有容量驱动的通道块 `CH_deg = min(C, 32)`**（见 §4.5，是**容量约束**而非性能裁决——独立于本节主选的 D-3 A/B 旋钮）。
  代表值（一般，`CH=64`）：`spp` → 24 单元；`nondiv` → 14 单元；代表值（退化，`CH_deg=32`）：`global-1x1` → 512 单元（persistent，40 核 × 13 任务）。
  **依据**：`PL-1.4-tiling-heuristic` 实测 CH64 为该池化族最优档（21.6µs，优于 CH32/CH16）；与本设计 C 轴向量化形态同族。
  （退化路径取 `CH_deg=32` 的原因：`CH=64` 时 C4 ≈ 132.8 KB / C4b ≈ 112.1 KB，默认档 ×1.7 后超 192 KB，见 §4.5。）
- **备选（≤1 个，结构完整，可直接实现）**：**一般网格 `CH = min(C, 8)`（最小 CH）**，即把 op 数放大 8×、核数放大 8×（`spp` → 192 单元）。
  这是「发射代价主导 vs 吞吐主导」的另一端；buffer 形状与主选相同仅 CH 变小（UB 预算同步缩小 ≈ 8 KB 级），
  分核三要素按 `num_logical = 192` 重算（persistent 40 核 × 5 任务）。
  （**注**：备选只改**一般网格**的 `CH`；退化路径的 `CH_deg=32` 由**容量判据**决定、不受本 A/B 影响——若备选被采纳，退化路径可同步取 `CH_deg=8`
  以进一步降容量，但那是容量决策而非性能裁决。）
  （备选存在的意义**仅限于**：验证比 `PL-1.4` 已测范围（CH16/32/64）**更窄**的 CH=8 是否使趋势翻转——`PL-1.4` 未覆盖 CH<16，该点仍属未实测外推。）
- **裁决计划**：代表 shape = `spp-6x6`、`global-1x1`（各 1 组）；指标 = msprof op Task Duration；测量未知常数 = 窄向量 op 的每-op 代价
  （用 `perf_opt/` 下两种 CH 的实测反解）；判定阈值 = **备选 latency < 主选 × (1 − 0.10) 才翻转**（10% 稳健性门槛）；
  裁决结果回写路径 = Stage 4 实测数据经附录补记固化 §1.6.3 判定依据，若翻转则按 `perf-feedback.md` 受控路由回写 DESIGN.md。
- **shape 分派评估**：本 kernel 由 `functools.lru_cache` 按 `(n,c,h,w,out_h,out_w,dtype)` 特化，**支持按 shape 分派**主选/备选——
  裁决计划须评估 `global-1x1` 等大 N 形状走大 CH、`spp/nondiv` 走小 CH 的**按负载分派**（而非全局二选一）。
---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（迁移 prompt 指定；`os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`，
`ASCEND_MODE = "Developer"` 与 Stage 5 集成的 wrapper 常量一致）。

### 2.2 选型理由

- 算子是**纯 Vector**（无 matmul / 无 Cube），仅需逐元素向量运算 + 归约 + 掩码选择 + 布局重排——正是 Developer 模式的目标域
  （`examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/`、`examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/`、
  `examples/TileOPs/tileops/kernels/reduction/argmax/` 均为 Developer）。
- 无跨引擎流水、无手动 L1/L0 分级需求（`T.alloc_ub` 直映 UB）；编译器管理 UB 分配与自动同步（§7）。
- 索引追踪所需的 `T.vselect` / `T.vmax` / `T.vcmp` / `T.vmul` / `T.transpose` / `T.vcast` 全为 v-prefix Developer API。
- 迁移 prompt 明确要求 developer 模式。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 环境变量 | `TILELANG_ASCEND_MODE=Developer` |
| 内存分配 | `T.alloc_ub(...)`（GM→UB 路径） |
| 计算方式 | v-prefix 向量 API（`T.vmax/T.vcmp/T.vselect/T.vmul/T.vcast/T.vbrc` + `T.reduce_max`/`T.reduce_min` + `T.arange`/`T.reshape`）+ `T.transpose` + `T.copy` |
| 同步 | 编译器自动同步（§7）；无手动 `T.set_flag/T.wait_flag` |
| Kernel 维度 | 一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)` |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**；每步优先映射 v-prefix 向量 API，与 §1.6.2 的向量化结论一致。
> 索引相关步骤标 **[idx]**；退化路径标 **[C4]** / **[C4b]**。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `(ih0,ih1) ← bin_H(o_h)`；`(iw0,iw1) ← bin_W(o_w)` | 编译期（`o_w` 经 `T.serial` 变量→运行时标量）/ 运行时标量 bin 边界；`bh=ih1-ih0`、`bw=iw1-iw0` |
| 2 | `row_src[c, iw] ← x[n, c0+c, ih0+kh, iw]` | 行片搬运 `(real_c, W_in)` |
| 3 | `row_dst[iw, c] ← row_src[c, iw]` | 核内二轴转置 → C 最内 |
| 4 | `row_f32[iw, c] ← cast_f32(row_dst[iw, c])` | 整 tile fp32 上转（`rint`） |
| 5 | `eq_f32[iw, c] ← cast_f32(row_f32[iw,c] == row_f32[iw,c])` | NaN 掩码（`eq` 为 IEEE 正确；`1`=数值、`0`=NaN） |
| 6 | `acc[o_w, c] ← max(acc[o_w, c], row_f32[iw0(o_w)+kw, c])` | C 轴向量取最大值（`kh`/`kw` serial + 守卫） |
| 7 | `h_flag[o_w, c] ← h_flag[o_w, c] · eq_f32[iw0+kw, c]` | NaN 标记乘性累积（初值 1） |
| 8 **[idx]** | `f ← (ih0+kh)·W_in + (iw0+kw)`；`fvec ← brc(f)` | 扁平下标标量 → `(1,CH)` 向量 |
| 9 **[idx]** | `gt ← (row_f32[iw0+kw] > acc)`；`max_idx[o_w] ← max(max_idx[o_w], gt ? fvec : −1)` | first-hit（last-record）索引 |
| 10 **[idx]** | `isnan ← (eq_f32[iw0+kw] == 0)`；`nan_idx[o_w] ← max(nan_idx[o_w], isnan ? fvec : −1)` | last-NaN 索引（初值 −1） |
| 11 | `cfinal[o_w,c] ← (nan_idx[o_w,c] ≥ 0)` | 窗口含 NaN 判决（与值路径的 `h_flag==0` 等价；二选一，本设计用索引侧 `nan_idx` 以保证值/索引判决**同源**） |
| 12 | `out32[o_w,c] ← cfinal ? NaN : acc`；`idx32[o_w,c] ← cfinal ? nan_idx : max_idx` | 值/索引 NaN 判决 |
| 13 | `acc16 ← cast_dtype(out32)`；`idx_t ← cast_i32(idx32)ᵀ`；`idx64 ← cast_i64(idx_t)` | 值单次舍回；索引转置后 cast 到 int64 |
| 14 | `out[n, c0+c, o_h, o_w] ← acc_t[c, o_w]`；`indices[n, c0+c, o_h, o_w] ← idx64[c, o_w]` | 两路行片写回 `(real_c, out_W)` |
| C4 | `out[n,c,0,0] ← max_{ih,iw} x`；`indices ← (含NaN) ? max{k:isnan} : min{k: v_k=max}` | 退化：整平面 `reduce_max` + argmax 归约 + eq 掩码 `reduce_min` 判决 |
| C4b | `out[n,c,ih,iw] ← x[n,c,ih,iw]`；`indices[n,c,ih,iw] ← ih*W_in+iw` | 退化：恒等拷贝 + arange 常量索引模式 |

> **执行次序约束（关键，v1 修正）**：上表按「值 / 掩码 / 索引」分组排列，**并非执行次序**。单元内的正确执行次序为
> **#9（`vcmp(gt)` → `vselect` → `vmax(max_idx)`，其中 `gt` 与**更新前**的 `acc` 比较）→ #6（原地 `vmax(acc)`）→ #7（`vmul(h_flag)`）→ #10（last-NaN）**。
> 即 `gt ← (row_f32[iw0+kw] > acc)` 中的 `acc` 必须是**本元素累加之前**的值（= 源 `val > max_val` 里那个「此前元素的最大值」）；
> 若把 #6 置于 #9 之前，判据退化为 `v > max(a,v)`（恒假）⇒ `max_idx` 恒等于初值 `f0`，索引系统性错误。
> 实现形态与次序约束见 §3.3 / §6.2；机器验证见 §1.6.1 的 `N3-ord` 守卫。

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0a | `acc` 置 `-inf` | `T.vbrc(src, dst)` | `src = -T.infinity("float32")`（标量），`dst = acc` | Developer |
| 0b | `h_flag` 置 1 | `T.vbrc(src, dst)` | `src = 1.0`，`dst = h_flag` | Developer |
| 0c **[idx]** | `max_idx` 置 `ih0*W_in+iw0` | `T.vbrc(src, dst)` | `src = ih0*W_in + iw0`（int32 标量），`dst = max_idx[ow:ow+1, :]` | Developer |
| 0d **[idx]** | `nan_idx` 置 `-1` | `T.vbrc(src, dst)` | `src = -1`，`dst = nan_idx` | Developer |
| 2 | 行片 GM→UB | `T.copy(src, dst)` | `src = x[n, c0:c0+real_c, ih0+kh, 0:W_in]`，`dst = row_src[0:real_c, 0:W_in]` | Developer |
| 3 | 二轴转置 | `T.transpose(src, dst, permutation)` | `permutation=[1,0]`（`docs/Tilelang.language/创建操作/T.transpose.md`） | Developer |
| 4 | fp32 上转 | `T.vcast(src, dst, round_mode)` | `round_mode="rint"` | Developer |
| 5 | NaN 掩码 | `T.vcmp(A, B, C, cmp_mod)` + `T.vcast` | `A=B=row_f32`，`C=eq_mask`（bool 片上），`cmp_mod="eq"`；`T.vcast(eq_mask, eq_f32, "rint")` | Developer |
| 6 | C 轴取最大值 | `T.vmax(src0, src1, dst)` | 切片 `(1,CH)`：`src0=acc[ow:ow+1,:]`，`src1=row_f32[iw:iw+1,:]`，`dst=acc[ow:ow+1,:]`（原地）。**次序约束**：本行须在 #9a 之后执行（`gt` 与更新前的 `acc` 比较） | Developer |
| 7 | NaN 标记累积 | `T.vmul(src1, src2, dst)` | `src1=src2=dst=h_flag[ow:ow+1,:]`，`src2=eq_f32[iw:iw+1,:]`（原地） | Developer |
| 8 **[idx]** | 扁平下标广播 | `T.vbrc(src, dst)` | `src = (ih0+kh)*W_in + (iw0+kw)`（int32 标量），`dst = fvec`（`(1,CH)` int32） | Developer |
| 9a **[idx]** | 严格大于掩码 | `T.vcmp(A, B, C, cmp_mod)` | `A=row_f32[iw:iw+1,:]`，`B=acc[ow:ow+1,:]`，`C=gt_bool`，`cmp_mod="gt"`。**次序约束**：本行须在 #6（原地 `vmax(acc)`）之前执行 | Developer |
| 9b **[idx]** | 索引候选 | `T.vselect(Cond, A, B, Out)` | `Cond=gt_bool`，`A=fvec`，`B=neg1`（`(1,CH)` int32 全 −1），`Out=cand` | Developer |
| 9c **[idx]** | first-hit 累积 | `T.vmax(src0, src1, dst)` | `src0=dst=max_idx[ow:ow+1,:]`（int32 原地），`src1=cand` | Developer |
| 10a **[idx]** | NaN 掩码（逐元素） | `T.vcmp` | `A=eq_f32[iw:iw+1,:]`，`B=zero_elem`（`(1,CH)` fp32 全 0），`C=isnan_bool`，`cmp_mod="eq"` | Developer |
| 10b **[idx]** | NaN 索引候选 | `T.vselect` | `Cond=isnan_bool`，`A=fvec`，`B=neg1`，`Out=nanc` | Developer |
| 10c **[idx]** | last-NaN 累积 | `T.vmax` | `src0=dst=nan_idx[ow:ow+1,:]`（int32 原地），`src1=nanc` | Developer |
| 11 | NaN 判决条件 | `T.vcmp` | `A=nan_idx`，`B=zeros`（int32 全 0），`C=cfinal`，`cmp_mod="ge"` | Developer |
| 12a | `cfinal?NaN:max` | `T.vselect` | `Cond=cfinal`，`A=nanbuf`（`inf-inf`），`B=acc`，`Out=out32` | Developer |
| 12b **[idx]** | `cfinal?nan_idx:max_idx` | `T.vselect` | `Cond=cfinal`，`A=nan_idx`，`B=max_idx`，`Out=idx32`（int32） | Developer |
| 13a | 值 dtype 回转 | `T.vcast(src, dst, round_mode)` | `round_mode="rint"` | Developer |
| 13b **[idx]** | 索引转置回转 | `T.transpose(src, dst, [1,0])` | `idx32 (out_W,CH) → idx_t (CH,out_W)`（int32） | Developer |
| 13c **[idx]** | 索引 i32→i64 | `T.vcast(src, dst, round_mode)` | `T.vcast(idx_t, idx64, round_mode="rint")`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 `i32→i64 rint` ✓） | Developer |
| 14 | 行片 UB→GM（两路） | `T.copy(src, dst)` | 值 `acc_t[0:real_c,0:out_W] → out[n,c0:c0+real_c,oh,0:out_W]`；索引 `idx64[0:real_c,0:out_W] → indices[...]` | Developer |
| C4 | 全平面取最大值 | `T.reduce_max(buffer, out, dim)` | `(CH,H,W)→(CH,H,1)`（dim=2）→`(CH,1,1)`（dim=1），`clear=True` | Developer |
| C4 **[idx]** | 平面下标向量 | `T.arange` + `T.reshape` + `T.vbrc` | `T.arange(k1d (H*W,), [1], 0)` → `T.reshape` `(1,1,H*W)` → `T.vbrc` → `(CH,1,H*W)` → `T.reshape` → `k_j (CH,H,W)`（int32） | Developer |
| C4 **[idx]** | first-hit 索引 | `T.vselect` + `T.reduce_min` | `eqk_b=vcmp(flat32,gbrc,"eq")`；`cand=vselect(eqk_b, k_j, bigk)`；`T.reduce_min(cand, midk, dim=2)` + `T.reduce_min(midk, first_k, dim=1)` | Developer |
| C4 **[idx]** | last-NaN 索引 | `T.vselect` + `T.reduce_max` | `candn=vselect(mb, k_j, neg1w)`；`T.reduce_max(candn, midn2, dim=2)` + `T.reduce_max(midn2, nan_k, dim=1)` | Developer |
| C4 | NaN 判决 | `T.vcmp`+`T.vcast`+`T.reduce_min` | `mb=vcmp(eq)`；`mf32=vcast(i1→f32,"rint")`；`reduce_min` 链 → `nmin∈{0,1}`；`cond=vcmp(nmin,zero,"eq")` | Developer |
| C4b **[idx]** | 常量索引模式 | `T.arange`+`T.reshape`+`T.vbrc`+`T.vcast` | `(H*W,)` → `(1,1,H,W)` → `vbrc` → `(CH,1,H,W)` → `reshape (CH,H,W)` → `vcast` int64（任务循环外构建一次） | Developer |
| 边界差 | 整数 bin 边界 | Python `adaptive_bin`（host/trace） | 复用源 `common.py` 逻辑 | host |

### 3.3 计算伪代码

> 注（迁移模板要求）：`T.Tensor` 的数据类型以**第二个位置参数**传入，不用 `dtype=` 关键字。
> 依据 `TRAP-tir-range-loop-not-unrolled`：`@T.prim_func` 内 Python `range` **不是**编译期展开，故 `o_w`/`kw` 与 `kh` 同形
> 用 `T.serial(max_*)` + `if k < extent` 守卫（同族前作 Stage 3 实测确认的可行形态）。
> 注（**退化路径通道块**）：本伪代码中「一般网格」分支的通道块为 `CH = min(C, 64)`；「C4/C4b」两个退化分支使用
> **容量驱动的 `CH_deg = min(C, 32)`**（见 §4.5；两分支缓冲形状中的 `CH` 应理解为 `CH_deg`）。
> 注（**执行次序**）：一般网格分支内 `vcmp(gt)` 先于 `vmax(acc)`——见下方注释与 §3.1/§1.6.2 的次序约束。

```python
# 迁移决策后的 NPU 算法伪代码（Developer 模式；C 轴向量化 + 三路向量累加器 + 核内转置）
@tilelang.jit(out_idx=[1, 2], target="npuir")     # 双输出：out / indices
def _build(block_m=None):      # block_m 为源接口兼容参数（K9 保留），NPU tiling 由 shape 派生，值被校验后忽略
    @T.prim_func
    def main(
        x:       T.Tensor((N, C, H_in, W_in), dtype),      # dtype 位置参数（非 dtype=）
        out:     T.Tensor((N, C, out_H, out_W), dtype),
        indices: T.Tensor((N, C, out_H, out_W), "int64"),
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            # ---- C4b 恒等（out_H==H_in and out_W==W_in；单元 = 整张 (n, c_block) 平面）----
            if IS_IDENTITY:
                idx1d = T.alloc_ub((H_in * W_in,), "int32")
                idxp4 = T.alloc_ub((1, 1, H_in, W_in), "int32")     # reshape 视图（元数据）
                idxb4 = T.alloc_ub((CH, 1, H_in, W_in), "int32")
                idxbl = T.alloc_ub((CH, H_in, W_in), "int32")
                idx64 = T.alloc_ub((CH, H_in, W_in), "int64")
                stage = T.alloc_ub((CH, H_in, W_in), dtype)
                T.arange(idx1d, [1], 0)                    # 平面扁平下标 0..H*W-1（1-D 优先形态）
                T.reshape(idx1d, idxp4)
                T.vbrc(idxp4, idxb4)                       # (1,1,H,W) -> (CH,1,H,W)
                T.reshape(idxb4, idxbl)
                T.vcast(idxbl, idx64, round_mode="rint")   # 索引常量模式（任务不变，循环外构建一次）
                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < NUM_LOGICAL:                  # NUM_LOGICAL = N * num_cblk
                        nb = uid // num_cblk; cb = uid % num_cblk
                        real_c = T.min(CH, C - cb*CH)
                        T.copy(x[nb, cb*CH:cb*CH+real_c, 0:H_in, 0:W_in], stage[0:real_c, 0:H_in, 0:W_in])
                        T.copy(stage[0:real_c, 0:H_in, 0:W_in], out[nb, cb*CH:cb*CH+real_c, 0:H_in, 0:W_in])
                        T.copy(idx64[0:real_c, 0:H_in, 0:W_in],
                               indices[nb, cb*CH:cb*CH+real_c, 0:H_in, 0:W_in])
            # ---- C4 全局池化（out_H==1 and out_W==1；单元 = 整张 (n, c_block) 平面）----
            elif IS_GLOBAL:
                flat   = T.alloc_ub((CH, H_in, W_in), dtype)
                flat2  = T.alloc_ub((CH, H_in, W_in), "float32")
                mb     = T.alloc_ub((CH, H_in, W_in), "bool")   # 1=数值, 0=NaN
                mf32   = T.alloc_ub((CH, H_in, W_in), "float32")
                k_j    = T.alloc_ub((CH, H_in, W_in), "int32")  # 平面扁平下标（各通道同值）
                cand   = T.alloc_ub((CH, H_in, W_in), "int32")
                candn  = T.alloc_ub((CH, H_in, W_in), "int32")
                neg1w  = T.alloc_ub((CH, H_in, W_in), "int32")
                bigk   = T.alloc_ub((CH, H_in, W_in), "int32")
                eqk_b  = T.alloc_ub((CH, H_in, W_in), "bool")
                gbrc   = T.alloc_ub((CH, H_in, W_in), "float32")
                mid    = T.alloc_ub((CH, H_in, 1), "float32")
                gmax   = T.alloc_ub((CH, 1, 1), "float32")
                midn   = T.alloc_ub((CH, H_in, 1), "float32")
                nmin   = T.alloc_ub((CH, 1, 1), "float32")
                midk   = T.alloc_ub((CH, H_in, 1), "int32")     # 索引候选中间归约
                midn2  = T.alloc_ub((CH, H_in, 1), "int32")
                firstk = T.alloc_ub((CH, 1, 1), "int32")
                nank   = T.alloc_ub((CH, 1, 1), "int32")
                zeros  = T.alloc_ub((CH, 1, 1), "float32")
                cond   = T.alloc_ub((CH, 1, 1), "bool")
                idx32  = T.alloc_ub((CH, 1, 1), "int32")
                idx64  = T.alloc_ub((CH, 1, 1), "int64")
                out32  = T.alloc_ub((CH, 1, 1), "float32")
                infbuf = T.alloc_ub((CH, 1, 1), "float32")
                nanbuf = T.alloc_ub((CH, 1, 1), "float32")
                gout   = T.alloc_ub((CH, 1, 1), dtype)
                k1d    = T.alloc_ub((H_in * W_in,), "int32")    # 1-D arange（优先形态）
                k1r    = T.alloc_ub((1, 1, H_in * W_in), "int32")
                k1w    = T.alloc_ub((CH, 1, H_in * W_in), "int32")
                T.arange(k1d, [1], 0)                            # 0..H*W-1 = 平面行主序扁平下标
                T.reshape(k1d, k1r)
                T.vbrc(k1r, k1w)                                 # (1,1,H*W) -> (CH,1,H*W)
                T.reshape(k1w, k_j)                              # -> (CH,H,W)（reshape 为元数据视图）
                T.vbrc(-1, neg1w)                                # last-NaN 候选哨兵
                T.vbrc(H_in * W_in, bigk)                        # first-hit 候选哨兵（> 任何平面扁平）
                T.clear(zeros)
                for t in T.serial(num_local):
                    uid = t * num_kernels + cid
                    if uid < NUM_LOGICAL:                  # NUM_LOGICAL = N * num_cblk
                        nb = uid // num_cblk; cb = uid % num_cblk
                        real_c = T.min(CH, C - cb*CH)
                        T.copy(x[nb, cb*CH:cb*CH+real_c, 0:H_in, 0:W_in], flat[0:real_c, 0:H_in, 0:W_in])
                        T.vcast(flat, flat2, round_mode="rint")
                        # --- 值：整平面 max + eq 掩码 NaN 判决 ---
                        T.reduce_max(flat2, mid, dim=2, clear=True)      # (CH,H,W) -> (CH,H,1)
                        T.reduce_max(mid,   gmax, dim=1, clear=True)     # -> (CH,1,1)
                        T.vbrc(gmax, gbrc)                               # 广播到平面
                        T.vcmp(flat2, flat2, mb, "eq")                   # 1=数值, 0=NaN
                        T.vcast(mb, mf32, round_mode="rint")
                        T.reduce_min(mf32, midn, dim=2, clear=True)
                        T.reduce_min(midn, nmin, dim=1, clear=True)      # 0 iff 含 NaN
                        T.vcmp(nmin, zeros, cond, "eq")                  # has_nan
                        T.vbrc(T.infinity("float32"), infbuf); T.vsub(infbuf, infbuf, nanbuf)  # NaN = inf-inf
                        T.vselect(cond, nanbuf, gmax, out32)             # has_nan ? NaN : max
                        T.vcast(out32, gout, round_mode="rint")
                        T.copy(gout[0:real_c, 0:1, 0:1], out[nb, cb*CH:cb*CH+real_c, 0:1, 0:1])
                        # --- 索引：first-hit = min{k: v_k = max}；last-NaN = max{k: isnan(v_k)} ---
                        T.vcmp(flat2, gbrc, eqk_b, "eq")
                        T.vselect(eqk_b, k_j, bigk, cand)
                        T.reduce_min(cand,  midk,  dim=2, clear=True)
                        T.reduce_min(midk,  firstk, dim=1, clear=True)
                        T.vselect(mb, k_j, neg1w, candn)
                        T.reduce_max(candn, midn2, dim=2, clear=True)
                        T.reduce_max(midn2, nank,  dim=1, clear=True)
                        T.vselect(cond, nank, firstk, idx32)             # has_nan ? nank : firstk
                        T.vcast(idx32, idx64, round_mode="rint")         # i32 -> i64
                        T.copy(idx64[0:real_c, 0:1, 0:1], indices[nb, cb*CH:cb*CH+real_c, 0:1, 0:1])
            # ---- 一般网格（单元 = (n, o_h, c_block)；NUM_LOGICAL = N*out_H*num_cblk）----
            else:
                acc      = T.alloc_ub((out_W, CH), "float32")   # 值累加器（向量轴 = C）
                h_flag   = T.alloc_ub((out_W, CH), "float32")   # NaN 免标记（初值 1）
                max_idx  = T.alloc_ub((out_W, CH), "int32")     # first-hit 索引
                nan_idx  = T.alloc_ub((out_W, CH), "int32")     # last-NaN 索引（初值 -1）
                zeros_i  = T.alloc_ub((out_W, CH), "int32")     # int32 零（供 vcmp "ge" 用）
                zeros    = T.alloc_ub((out_W, CH), "float32")
                infbuf   = T.alloc_ub((out_W, CH), "float32")
                nanbuf   = T.alloc_ub((out_W, CH), "float32")
                out32    = T.alloc_ub((out_W, CH), "float32")
                idx32    = T.alloc_ub((out_W, CH), "int32")
                idx_t    = T.alloc_ub((CH, out_W), "int32")
                idx64    = T.alloc_ub((CH, out_W), "int64")
                cfinal   = T.alloc_ub((out_W, CH), "bool")
                cand     = T.alloc_ub((1, CH), "int32")
                nanc     = T.alloc_ub((1, CH), "int32")
                fvec     = T.alloc_ub((1, CH), "int32")
                neg1     = T.alloc_ub((1, CH), "int32")
                gt_b     = T.alloc_ub((1, CH), "bool")
                isn_b    = T.alloc_ub((1, CH), "bool")
                zero_e   = T.alloc_ub((1, CH), "float32")
                row_src  = T.alloc_ub((CH, W_in), dtype)
                row_dst  = T.alloc_ub((W_in, CH), dtype)
                row_f32  = T.alloc_ub((W_in, CH), "float32")
                eq_mask  = T.alloc_ub((W_in, CH), "bool")
                eq_f32   = T.alloc_ub((W_in, CH), "float32")
                acc16    = T.alloc_ub((out_W, CH), dtype)
                acc_t    = T.alloc_ub((CH, out_W), dtype)
                T.vbrc(0.0, zero_e)
                T.vbrc(-1, neg1)                             # 索引候选哨兵（-1 < 任何合法扁平）
                T.clear(zeros); T.clear(zeros_i)
                for t in T.serial(num_local):                # persistent（静态界）
                    uid = t * num_kernels + cid
                    if uid < NUM_LOGICAL:
                        nb  = uid // (out_H * num_cblk)
                        r   = uid %  (out_H * num_cblk)
                        o_h = r // num_cblk
                        cb  = r %  num_cblk
                        real_c = T.min(CH, C - cb*CH)
                        ih0, ih1 = adaptive_bin(o_h, H_in, out_H)     # 运行时标量
                        bh = ih1 - ih0
                        T.vbrc(-T.infinity("float32"), acc)          # 值单位元 = -inf
                        T.vbrc(1.0, h_flag)                          # NaN 免标记初值 1
                        for ow in T.serial(out_W):
                            iw0, iw1 = adaptive_bin(ow, W_in, out_W)
                            T.vbrc(ih0 * W_in + iw0, max_idx[ow:ow+1, :])   # first-hit 初值 = 首访问元素
                            T.vbrc(-1, nan_idx[ow:ow+1, :])                # last-NaN 哨兵
                        for kh in T.serial(MAX_BIN_H):               # 静态界 + 守卫
                            if kh < bh:
                                T.copy(x[nb, cb*CH:cb*CH+real_c, ih0+kh, 0:W_in],
                                       row_src[0:real_c, 0:W_in])
                                T.transpose(row_src, row_dst, permutation=[1, 0])
                                T.vcast(row_dst, row_f32, round_mode="rint")
                                T.vcmp(row_f32, row_f32, eq_mask, "eq")     # IEEE 正确的 NaN 探测
                                T.vcast(eq_mask, eq_f32, round_mode="rint")
                                for ow in T.serial(out_W):
                                    iw0, iw1 = adaptive_bin(ow, W_in, out_W)
                                    bw = iw1 - iw0
                                    for kw in T.serial(MAX_BIN_W):
                                        if kw < bw:
                                            T.vbrc((ih0 + kh) * W_in + iw0 + kw, fvec)   # 扁平下标
                                            # --- first-hit 索引（last-record = 破纪录扁平取 max）---
                                            # 次序约束（关键）：gt 必须与 **更新前** 的 acc 比较
                                            #（= 源 `val > max_val` 里 max_val 是此前元素的最大值）。
                                            # 故 vcmp(gt) 必须先于下方的 vmax(acc)；若颠倒，
                                            # 判据退化为 v > max(a,v)（恒假）⇒ max_idx 退化为 f0。
                                            T.vcmp(row_f32[iw0+kw:iw0+kw+1, 0:CH],
                                                   acc[ow:ow+1, 0:CH], gt_b, "gt")
                                            T.vselect(gt_b, fvec, neg1, cand)
                                            T.vmax(max_idx[ow:ow+1, 0:CH], cand, max_idx[ow:ow+1, 0:CH])
                                            # --- 值（在 gt 取掩码之后再原地更新 acc）---
                                            T.vmax(acc[ow:ow+1, 0:CH],
                                                   row_f32[iw0+kw:iw0+kw+1, 0:CH],
                                                   acc[ow:ow+1, 0:CH])
                                            T.vmul(h_flag[ow:ow+1, 0:CH],
                                                   eq_f32[iw0+kw:iw0+kw+1, 0:CH],
                                                   h_flag[ow:ow+1, 0:CH])
                                            # --- last-NaN 索引（NaN 位置扁平取 max）---
                                            T.vcmp(eq_f32[iw0+kw:iw0+kw+1, 0:CH], zero_e, isn_b, "eq")
                                            T.vselect(isn_b, fvec, neg1, nanc)
                                            T.vmax(nan_idx[ow:ow+1, 0:CH], nanc, nan_idx[ow:ow+1, 0:CH])
                        T.vcmp(nan_idx, zeros_i, cfinal, "ge")           # cfinal = (nan_idx >= 0)
                        T.vselect(cfinal, nan_idx, max_idx, idx32)
                        T.transpose(idx32, idx_t, permutation=[1, 0])
                        T.vcast(idx_t, idx64, round_mode="rint")         # i32 -> i64
                        T.copy(idx64[0:real_c, 0:out_W],
                               indices[nb, cb*CH:cb*CH+real_c, o_h, 0:out_W])
                        T.vbrc(T.infinity("float32"), infbuf)
                        T.vsub(infbuf, infbuf, nanbuf)                   # NaN = inf - inf（IEEE，运行时）
                        T.vselect(cfinal, nanbuf, acc, out32)            # has_nan ? NaN : max
                        T.vcast(out32, acc16, round_mode="rint")
                        T.transpose(acc16, acc_t, permutation=[1, 0])
                        T.copy(acc_t[0:real_c, 0:out_W],
                               out[nb, cb*CH:cb*CH+real_c, o_h, 0:out_W])
    return main
```

（伪代码为**决策示意**，Stage 3 可细化局部次序、buffer 复用与视图（`reshape`）细节；但
「向量轴 = C」「核内二轴转置（值与索引各一次）」「静态 bin 界 + 守卫」「eq 掩码 NaN 追踪 + `nan_idx>=0` 判决」
「`gt`+`vmax` first-hit / `isnan`+`vmax` last-NaN」「**`gt` 恒与更新前的 `acc` 比较（`vcmp(gt)` 先于 `vmax(acc)`）**」
「int32 承载索引 + 转置后 cast int64」「C4/C4b 特化 + 其单元粒度 + 退化路径 `CH_deg=min(C,32)`」
是冻结决策。）

### 3.4 API 可行性确认

| API | 来源确认 | 是否验证 |
|-----|---------|---------|
| `T.Kernel(num, is_npu=True) as (cid, _)` | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3（NPU ≤2 维） | ✓ 现有 examples 广泛使用 |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md` §2.2.2（1D~2D 充分验证；3D 见下注） | ✓（同族前作同款） |
| `T.clear`（buffer 清零，1~5 维） | `docs/Tilelang.language/内存操作/T.clear.md` §2.2.1（fp32/int32 ✓） | ✓ |
| `T.copy`（切片 GM↔UB） | `docs/Tilelang.language/内存操作/T.copy.md` §2.1/§2.2.2 | ✓ |
| `T.transpose`（二轴交换；fp16/fp32/**int32/int64**） | `创建操作/T.transpose.md` §2.2.1/§2.3 | ✓（`PL-1.1` 实测 ~µs 级；**int32 形态无本仓先例，见 §9.1 风险**） |
| `T.vcast`（f16/bf16→f32 `rint`；f32→f16 `rint`；i1→f32 `rint`；**i32→i64 `rint`**） | `数据类型转换操作/T.vcast.md` §2.2.1 | ✓（argmax 用 f32→i64） |
| `T.vmax`（fp16/fp32/**int16/int32/int64**；`dst` 可原地） | `比较操作/T.vmax.md` §2.2.1 | ✓（原地形态 `PL-1.2` 实测） |
| `T.vcmp`（fp16/fp32/bf16/**int16/int32/int64**；"eq"/"gt"/"ge" 模式；输出 bool 片上） | `比较操作/T.vcmp.md` §2.2.1/§2.3 | ✓（**"ne" 为 ordered-NE，禁用**，见 `TRAP-vcmp-ne-ordered-nan`） |
| `T.vselect`（fp16/fp32/bf16/**int16/int32/int64**；bool 条件；同形 A/B/Out） | `条件操作/T.vselect.md` §2.2.1 | ✓（argmax 同款；**禁 in-place 别名**，见 `TRAP-vselect-inplace-carried-state`） |
| `T.vmul`（fp16/fp32/**int32**；支持广播/标量 src2） | `数学操作/T.vmul.md` §2.2.1/§2.2.2 | ✓ |
| `T.vbrc`（**int16/int32/int64**/fp16/fp32/bf16；scalar→任意 shape；tensor src 需同 rank + 单维 1） | `shape操作/T.vbrc.md` §2.2.1/§2.2.2 | ✓（`CONST-arange-scalar-materialize` 的 (1,N)+首轴广播绕法） |
| `T.arange`（int16/int32/int64/fp16/fp32；**1-D 优先**） | `创建操作/T.arange.md` §2.2.1/§2.3.2 | ✓（argmax r6 用 1-D + reshape 形态） |
| `T.reshape`（元数据级；int32 ✓） | `shape操作/T.reshape.md` §2.2.1 | ✓（argmax r5/r6 用） |
| `T.reduce_max` / `T.reduce_min`（fp32/int32 ✓，`clear` 参数） | `规约操作/T.reduce_max.md` §2.2.1；`规约操作/T.reduce.md` §2.1 | ✓（单输出 max 前作 C4 实测） |
| `T.infinity("float32")` / `T.vsub`（fp32） | `创建操作/T.infinity.md` §2.2.1；`数学操作/T.vsub.md` §2.2.1 | ✓ |

> **dtype 约束提醒**：`T.vmax`/`T.reduce_max`/`T.vsub` 的 dtype 表**不含 bf16**（见各 §2.2.1），
> `T.vcmp`/`T.vselect`/`T.transpose`/`T.vcast`/`T.vmul` 支持 bf16。故 **bf16 输入必须经 `T.vcast` 上转到 fp32 后计算**——
> 这与「fp32 比较域」语义一致（§0.6 R2），无需额外代价。索引累加器为 int32（与 dtype 无关，bf16/fp16 共用）。
> **`bool`（i1）不可 load/store GM**（`T.vcmp` §2.3）：本设计的 `eq_mask`/`gt_b`/`isn_b`/`cfinal`/`mb`/`cond` 均为**片上中间结果**，不写回 GM ✓。
> **注（3D UB buffer）**：`T.alloc_ub` 文档称 3D~5D「有待充分验证」；C4/C4b 路径用 3D buffer 的形态与同族前作
> `examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/_adaptive_max_pool2d_kernel.py`（C4 `flat/flat32/mid/gmax` 均为 3D）一致，
> 该前作 Stage 3 已实测通过——故本条属**已有同族实证**；若 Stage 3 遇 3D UB 问题，回退为 2D `(CH, H*W)` + reshape（Stage 3 细节）。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel（`T.Kernel` 一维/最多二维 block） | 否（用一维 `T.Kernel(num_kernels)`；任务维度在 kernel 内解码） | 一维网格；`(n,o_h,c_block)` 由 `uid = t*num_kernels + cid` 解码 |
| GPU 专用 API | 否（全部 v-prefix / `T.copy` / `T.transpose` / `T.arange` / `T.reshape`，均为 NPU API） | — |
| GEMM 非整除 / L0C 溢出 | 否（无 Cube 计算，`L0C` 不适用） | — |
| 分核策略 | 见 §5.5（三要素齐全） | persistent grid-stride |
| host 侧禁止改输入内容 | 否（host 只做 `unsqueeze/squeeze/contiguous` 视图操作） | `ascend-constraints.md` §4 |
| 动态 `T.serial` 边界 | 涉及（`kh` 上界 `bin_h(o_h)`、`kw` 上界 `bw` 依赖运行时解码值） | 静态上界 `MAX_BIN_H`/`MAX_BIN_W` + `if k < extent` 守卫（源同款） |
| Python `range` 非编译期展开 | 涉及（`o_w`/`kw` 若写成 `range` 会降为串行 TIR 循环） | 统一写 `T.serial(max_*)` + 守卫（`TRAP-tir-range-loop-not-unrolled`） |
| `target="npuir"` 的 `tilelang.jit` 无 `compile_flags` 形参 | 涉及（源 `compile_flags=["-O3","-DENABLE_BF16"]`） | 不传该形参（同族前作已弃用；bf16 支持不依赖该 flag） |
| int32 索引位宽上限 | 涉及（`h_in*w_in` 需 < 2**31） | 工厂期守卫（§9.1） |
| `T.vcmp("ne")` 为 ordered-NE | 涉及（NaN 探测） | 改用 `eq`（IEEE 正确）取反（§1.6.1 N1） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 维度 / 并行模型 | `T.Kernel(ceildiv(total,block_m), threads)` + `T.Parallel(block_m)`（线程 lane 级） | 一维 `T.Kernel(num_kernels, is_npu=True)` | §0.6 R1（`(n,o_h,c_block)` 单元 + C 向量 lane） |
| 归约单元 | 每线程标量比较链 | UB fp32 向量比较器（C 轴） | §0.6 R2 |
| NaN 追踪 | 标量 bool `has_nan \| isnan(val)` | C 向量 eq 掩码乘性累积（`h_flag`） | §0.6 R2（语义不变） |
| **索引追踪** | 标量 `max_idx`/`nan_idx` + `first_valid` | int32 向量累加器（`gt`+`vmax` / `isnan`+`vmax`） | §0.6 R3 + §1.6.1 N2/N3（语义不变） |
| **双输出** | `out_idx=[1,2]`，同一线程写两路 | `out_idx=[1,2]`，同一单元写两路（同一遍扫描） | 保留融合 #8 |
| 数据搬运 | 逐元素 GM gather | UB tile 搬运 + 核内二轴转置（值/索引各一次回转） | §0.6 R4 |
| `threads` 参数 | factory `(block_m, threads)`、`default_config={"block_m":256,"threads":256}` | **移除 `threads`**（NPU 无 CUDA 线程）；callable 只留 `block_m` | 脚手架已按 K9 移除 |
| 退化网格 | 无特化 | C4/C4b 结构特化（含索引侧 argmax 归约 / arange 常量模式） | §0.6 R5 / §1.6.1 |
| autotune | `autotune_configs` | 移除（K8），启发式 config | 脚手架已移除 |
| 编译标记 | `compile_flags=["-O3","-DENABLE_BF16"]` | **不传**（本分支 `tilelang.jit` 无该形参） | 同族前作已弃用 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/_adaptive_max_pool2d_kernel.py` | **最高（同族前作，CASE-pool-adaptivemaxpool2d-migration）** | Developer 一维 `T.Kernel(num_kernels,is_npu=True)`；`CH=min(C,64)`；`(n,o_h,c_block)` 解码；`T.vbrc`/`T.clear`/`T.vcast`/`T.vmax`/`T.transpose`/`T.vmul` 用法；eq 掩码 NaN 追踪；C4/C4b 特化与容量判据；persistent 分核与 `real_c` 尾块 |
| `examples/adaptive_max_pool2d/_adaptive_max_pool2d_kernel/DESIGN.md` | 最高（同族设计骨架） | §1.6.0 调研表 / §1.6.3 布局候选矩阵 / §5.5 分核三要素 / §8.3 L0 计划的**直接复用对象**；本设计在其上增加索引输出维度 |
| `examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py` | 高（同族前作，CASE-pool-adaptiveavgpool2d-migration） | C 轴向量化 + 核内转置 + C4/C4b 的完整 NPU 实现形态 |
| `examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/perf_opt/_argreduce_kernel.py` | **高（索引输出算法族参考）** | **first-occurrence 索引的向量化形态**（`vselect` 掩码候选 + `reduce_min`；`BIG` 哨兵；fp32 索引 + `reshape`+`vcast` 转 int64）；`T.arange` 1-D→reshape→`vbrc` 索引构造；`T.vcmp eq` 用法；in-place `vselect` 陷阱注释；`dim=0` 归约风险 |
| `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py` | 高（结构模板） | persistent grid-stride `T.serial(num_local)` + `if uid < num_logical` 守卫；`T.alloc_ub`；尾块 gate |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py` | 高（归约模板） | `T.reduce_max` 用法、fp32 中转、尾部切片 |
| `examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d_with_indices/_adaptive_max_pool2d_indices_fwd_kernels.py` | 高（本迁移的源函数抽取件） | GPU 源 kernel + `adaptive_bin`/`max_adaptive_bin_extent` helper（迁移时逐字对照） |
| `examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d_with_indices/adaptive_max_pool2d_with_indices.py` | 高（NPU wrapper 契约） | `npub::` custom_op 双输出 `Tuple[Tensor,Tensor]`、`out_idx=[1,2]`、`default_config={"block_m":256}`、`ascend_mode="Developer"` |
| `examples/TileOPs/tileops/ops/pool/adaptive_max_pool2d_with_indices.py` | 高（Op 层契约） | `_returns_indices` 双输出 `_eager_forward`、3D unsqueeze/squeeze、`_kernel_cache`、`eval_roofline(indices=True)` |
| `examples/TileOPs/tests/ops/test_pool.py` / `examples/TileOPs/benchmarks/ops/bench_pool.py` / `examples/TileOPs/tileops/manifest/pool.yaml` | 高（规格/golden/bench） | 用例 `test_adaptive_max_pool2d_indices*` 的 workload、`ref_program` 双输出、`atol=rtol=0` |
| pattern-library `layout.md` `PL-1.1/1.2/1.3/1.4` + `traps-compiler.md` `TRAP-vcmp-ne-ordered-nan` / `TRAP-vselect-inplace-carried-state` / `TRAP-tir-range-loop-not-unrolled` | 高（池化/索引类实测与陷阱） | 核内融合转置链、C 轴切片累加、host permute 证伪、tiling 启发式；NaN 探测合法形态；in-place vselect 陷阱；Python `range` 不展开 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(N, C, H_in, W_in)` | float16 \| bfloat16 | NCHW 连续；3D CHW 由 op 层 `unsqueeze(0)` 适配 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `(N, C, out_H, out_W)` | same_as_input | NCHW 连续；`out_idx=[1,2]` 第 1 路 |
| `indices` | `(N, C, out_H, out_W)` | **int64** | NCHW 连续；`out_idx=[1,2]` 第 2 路 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `row_src` | `(CH, W_in)` | dtype | UB | 行片 GM→UB 暂存 |
| `row_dst` | `(W_in, CH)` | dtype | UB | 核内二轴转置输出（C 最内） |
| `row_f32` | `(W_in, CH)` | float32 | UB | fp32 上转比较源 |
| `eq_mask` / `eq_f32` | `(W_in, CH)` | bool / float32 | UB | 行级 NaN 掩码（`x==x`）与其 fp32 {0,1} 副本 |
| `acc` | `(out_W, CH)` | float32 | UB | C 轴向量最大值累加器（初值 `-inf`） |
| `h_flag` | `(out_W, CH)` | float32 | UB | NaN 免标记乘性累积器（初值 1） |
| `max_idx` | `(out_W, CH)` | **int32** | UB | first-hit 索引（初值 `ih0*W_in+iw0`） |
| `nan_idx` | `(out_W, CH)` | **int32** | UB | last-NaN 索引（初值 −1） |
| `zeros` / `infbuf` / `nanbuf` / `out32` | `(out_W, CH)` | float32 | UB | 零常量 / `+inf` / NaN（`inf-inf`）/ 值判决结果 |
| `idx32` / `idx_t` | `(out_W, CH)` / `(CH, out_W)` | int32 | UB | 索引判决结果 + 转置回转 |
| `idx64` | `(CH, out_W)` | int64 | UB | 索引落盘前的 i64 副本 |
| `cfinal` | `(out_W, CH)` | bool | UB | 窗口级 has_nan 判决条件（**片上**） |
| `cand` / `nanc` / `fvec` / `neg1` / `gt_b` / `isn_b` / `zero_e` | `(1, CH)` | int32 / bool / float32 | UB | 每元素索引候选、扁平下标向量、−1 常量、掩码（**均片上**） |
| `acc16` / `acc_t` | `(out_W, CH)` / `(CH, out_W)` | dtype | UB | 值舍回 + 转置回转 |
| `flat` / `flat2`（C4） | `(CH, H_in, W_in)` dtype / `(CH, H_in*W_in)` float32 | UB | 全局池化退化路径（`flat2` 为展平视图） |
| `mb` / `mf32`（C4） | `(CH,H_in*W_in)` | bool / float32 | UB | NaN 掩码（bool 片上）与 fp32 掩码 |
| `k_j`（C4） | `(CH, H_in, W_in)` | int32 | UB | 平面扁平下标向量（`arange` 1-D → reshape → `vbrc` → reshape） |
| `cand` / `candn` / `neg1w` / `bigk`（C4） | `(CH,H_in,W_in)` 各 | int32 | UB | 索引候选与两个哨兵张量（`vselect` 需全张量操作数） |
| `mid` / `gmax` / `gbrc` / `midn` / `nmin` / `midk` / `midn2`（C4） | `(CH,H_in,1)` / `(CH,1,1)` / `(CH,H_in,W_in)` | float32 / int32 | UB | 值与索引的归约中间/结果 |
| `zeros` / `cond` / `idx32` / `idx64` / `out32` / `infbuf` / `nanbuf` / `gout`（C4） | `(CH,1,1)` 等 | float32 / bool / int32 / int64 / dtype | UB | 判决、NaN 常量、索引与值输出 |
| `idx1d` / `idxp4` / `idxb4` / `idxbl` / `idx64` / `stage`（C4b） | `(H*W,)` / `(1,1,H,W)` / `(CH,1,H,W)` / `(CH,H,W)` / `(CH,H,W)` i64 / `(CH,H,W)` dtype | UB | 恒等路径中转 + 索引常量模式（任务循环外构建一次） |

> **注（退化路径通道块）**：C4/C4b 两个退化分支的 `CH` 一律取 **`CH_deg = min(C, 32)`**（容量驱动，见 §4.5），
> 与一般网格分支的 `CH = min(C, 64)` 不同；§4.3 上表 C4/C4b 行中的 `CH` 即 `CH_deg`。

### 4.4 内存搬运路径

```
【一般网格】
GM[x] --T.copy 行片(real_c,W_in)--> UB[row_src] --T.transpose[1,0]--> UB[row_dst] --T.vcast(rint)--> UB[row_f32]
UB[row_f32] --T.vcmp(eq)+T.vcast--> UB[eq_f32]
  ┌ 值:   UB[acc] <- vmax(acc, row_f32[iw])        ∥  UB[h_flag] <- vmul(h_flag, eq_f32[iw])
  ├ 索引: UB[max_idx] <- vmax(max_idx, vselect(gt, fvec, -1))
  └ 索引: UB[nan_idx] <- vmax(nan_idx, vselect(isnan, fvec, -1))
UB[nan_idx] --vcmp(ge 0)--> UB[cfinal]
UB[acc]/UB[nan_idx]/UB[max_idx] --vselect(cfinal,·,·)--> UB[out32] --vcast(rint)--> UB[acc16] --transpose[1,0]--> UB[acc_t] --T.copy 行片--> GM[out]
UB[idx32] = vselect(cfinal, nan_idx, max_idx) --transpose[1,0]--> UB[idx_t] --vcast(rint, i32->i64)--> UB[idx64] --T.copy 行片--> GM[indices]
【C4 全局池化】GM[x] --T.copy 连续块(CH,H,W)--> UB[flat] --vcast--> UB[flat2]
   UB[flat2] --reduce_max(dim=2→1)--> UB[gmax] --vbrc--> gbrc
   UB[flat2] --vcmp(eq)+vcast(i1→f32)+reduce_min(dim=2→1)--> UB[nmin] --vcmp(eq 0)--> cond(has_nan)
   索引：T.arange(1-D)→reshape→vbrc→reshape--> k_j(平面扁平下标)
         UB[flat2] --vcmp(flat2==gbrc)+vselect(k_j,bigk)+reduce_min(dim=2→1)--> firstk
         UB[mb] --vselect(k_j,neg1w)+reduce_max(dim=2→1)--> nank
   cond --vselect(nanbuf,gmax)--> out32 --vcast--> GM[out] ; cond --vselect(nank,firstk)--> idx32 --vcast(i32→i64)--> GM[indices]
【C4b 恒等】GM[x] --T.copy 整平面(CH,H,W)--> UB[stage] --T.copy--> GM[out]
            索引：T.arange→reshape→vbrc→reshape→vcast i64 --> UB[idx64] --T.copy--> GM[indices]（单元 = 整张 (n,c_block) 平面）
（无 GM 中间往返、无 Cube/L1/L0C 路径）
```

### 4.5 UB 内存预算

| Buffer | Shape（代表 workload：`spp-6x6`，CH=64，W_in=56，out_W=6，fp16） | dtype | 大小 (Bytes) |
|--------|------------------------|-------|-------------|
| `row_src` / `row_dst` | `(64, 56)` / `(56, 64)` | fp16 | 7 168 × 2 |
| `row_f32` / `eq_f32` | `(56, 64)` 各 | fp32 | 14 336 × 2 |
| `eq_mask` | `(56, 64)` | bool | 3 584 |
| `acc` / `out32` / `zeros` / `infbuf` / `nanbuf` | `(6, 64)` 各 | fp32 | 1 536 × 5 |
| `max_idx` / `nan_idx` / `idx32` / `idx_t` | `(6, 64)` 各 | int32 | 1 536 × 4 |
| `idx64` | `(64, 6)` | int64 | 3 072 |
| `cfinal` | `(6, 64)` | bool | 384 |
| `cand` / `nanc` / `fvec` / `neg1` | `(1, 64)` 各 | int32 | 256 × 4 |
| `zero_e` | `(1, 64)` | fp32 | 256 |
| `acc16` / `acc_t` | `(6,64)` / `(64,6)` | fp16 | 768 × 2 |
| **总计（spp-6x6，CH=64）** | | | **≈ 65.1 KB** / 196 608 (192 KB, `CONST-capacity-910B2C`) ✓ |
| **C4（resnet `global-1x1`，`H=W=7`，plane 49，`CH_deg=32`，fp16）** | 见下方逐 buffer 复算 | — | **68 200 B ≈ 66.6 KB**（含 `T.reshape` 视图各自物化，与一般路径同口径）✓ |
| **C4b（恒等 `(1,8,9,11)→(9,11)`，plane 99，`CH_deg=8`，fp16）** | 见下方逐 buffer 复算 | — | **15 048 B ≈ 14.7 KB** ✓ |

**退化路径（C4/C4b）逐 buffer 复算（v1 修正，阻塞级 2）** —— 上表两行为**修正后**数值。v0 曾记 C4 ≈ 90 KB / C4b ≈ 64 KB，
与 §3.3/§4.3 的 buffer 清单**不符**（低报；C4b 更漏计 `idx1d`/`idxp4`/`idxb4`/`idxbl` ≈ 51 KB）。
以下按 §3.3 退化分支的**实际 `alloc` 清单**逐 buffer 求和（保守口径：`T.reshape` 视图可能各自物化），并同时列出 **v0 的 `CH=64`** 以说明为何必须收窄 `CH`：

| 组 | C4（plane=7×7=49），`CH_deg=32` | C4（`CH=64`，v0） | C4b（plane=9×11=99），`CH_deg=8` | C4b（`CH=64`，v0） |
|----|----------------------------------|-------------------|----------------------------------|--------------------|
| 12 个 `(CH,H,W)` 平面缓冲（C4：`flat`/`flat2`/`mb`/`mf32`/`k_j`/`cand`/`candn`/`neg1w`/`bigk`/`eqk_b`/`gbrc`/`k1w`） | 62 720 | 125 440 | — | — |
| 6 个平面缓冲（C4b：`stage`(dtype)/`idx1d`/`idxp4`/`idxb4`/`idxbl`/`idx64`） | — | — | 15 048 | 114 840 |
| 归约/判决小缓冲（`mid`/`midn`/`midk`/`midn2` `(CH,H,1)` + `gmax`/`nmin`/`firstk`/`nank`/`zeros`/`cond`/`idx32`/`idx64`/`out32`/`infbuf`/`nanbuf`/`gout` `(CH,1,1)`） | 5 088 | 10 176 | — | — |
| `k1d`+`k1r`（`(H*W,)`/`(1,1,H*W)`，CH 无关） | 392 | 392 | — | — |
| **合计** | **68 200 B（≈66.6 KB）** | **136 008 B（≈132.8 KB）** | **15 048 B（≈14.7 KB）** | **114 840 B（≈112.1 KB）** |

> **膨胀后（默认档 ×1.7）**：C4 `CH_deg=32` = 115 940 B ≈ **113.2 KB ✓**；C4b `CH_deg=8` = 25 582 B ≈ **25.0 KB ✓**。
> 对照 v0 的 `CH=64`：C4 = 231 214 B ≈ **225.8 KB ✗**、C4b = 195 228 B ≈ **190.7 KB ✗**（C4 超 192 KB、C4b 贴限）
> —— 这正是 v0「两档均在容量内」对退化路径不成立的原因，也是 v1 把退化路径通道块收窄为 `CH_deg` 的依据。

> **UB 膨胀口径两档**（`CONST-capacity-910B2C`，entry 见 `constants.md`；与一般路径同口径）：
> - **默认档（auto-multi-buffer 开，本设计默认，§7.3）**：膨胀 **~1.7x** → 一般 spp ≈ 111 KB ✓；**C4（`CH_deg=32`）≈ 113.2 KB ✓；C4b（`CH_deg=8`）≈ 25.0 KB ✓**。
> - **关闭档（`--enable-auto-multi-buffer=false`）**：膨胀 1.10–1.12 → 一般 spp ≈ 72 KB；C4 ≈ 73–75 KB；C4b ≈ 16–17 KB。
> **一般路径与修正后的退化路径在两档下均 < 192 KB**（v0 的退化路径两行在默认档下超容量，已由 `CH_deg` 修正消除）；
> §7.3「溢出时再评估关 multi-buffer」以本两档为可比基线。

> **退化路径容量判据（强制，`hardware-cost-model.md` §5.1）** —— **带字节数的不等式**（与 §3.3 的 `alloc` 清单一一对应，与一般路径同口径）：
> $$
> \texttt{degenerate\_bytes}(\texttt{CH\_deg}, H_{in}, W_{in}, \texttt{dtype}) \times \texttt{inflate} \le 192\,\text{KB} = 196\,608\ \text{B},\qquad \texttt{CH\_deg}=\min(C,32),
> $$
> 其中 `inflate` = 默认档 ~1.7 / 关闭档 ~1.11（来源同上），`degenerate_bytes` = §3.3 退化分支 alloc 清单逐 buffer 求和（保守：`T.reshape` 视图各自物化）：
> - **C4**：`CH_deg·H_in·W_in·(2+4+1+4+4+4+4+4+4+1+4) + CH_deg·H_in·W_in·4 + 小缓冲(CH_deg) + 392`
>   （前项 = 11 个 `(CH,H,W)` 平面缓冲的字节和：`flat`(2)+`flat2`(4)+`mb`(1)+`mf32`(4)+`k_j`(4)+`cand`(4)+`candn`(4)+`neg1w`(4)+`bigk`(4)+`eqk_b`(1)+`gbrc`(4)；
>   中项 = 第 12 个平面缓冲 `k1w`(int32)；后项 = `k1d`+`k1r`）
> - **C4b**：`CH_deg·H_in·W_in·(2+4+8) + CH_deg·H_in·W_in·4 + H_in·W_in·4 + H_in·W_in·4`
>   （`stage`(fp16,2)+`idxb4`(int32,4)+`idx64`(int64,8)；`idxbl`(int32,4)；`idxp4`(int32,4)；`idx1d`(int32,4)）
> **判据不满足时（含 `CH_deg=8` 仍超）→ 该 shape 回退一般路径**（纯优化，语义与精度不变；判据形态与同族前作 `identity_bytes` / `global_bytes` 一致）。
> 代表 shape 判定：`global-1x1`（plane 49，`CH_deg=32`）默认档 113.2 KB ≤ 192 KB ✓ 走 C4；恒等 `(1,8,9,11)`（plane 99，`CH_deg=8`）25.0 KB ✓ 走 C4b；
> 更大平面（如 `(1,8,32,32)→1×1`，plane 1024）实算 `degenerate_bytes ≈ 340 268 B ≈ 332 KB` ≫ 192 KB ⇒ **回退一般路径**
> （与 `VP-2026-0146`「退化路径 UB 随 H·W 线性放大、需编译期判据否则回退」的实测结论一致；同族前作实测回退且 `max_diff=0`）。

### 4.6 动态轴定义

**无动态轴**：`N, C, H_in, W_in, out_H, out_W` 全部在 factory 构建期（trace 期）确定，kernel 由 `functools.lru_cache` 按 shape 特化。
（`o_h`/`ih0`/`bin_h`/`bw` 在 kernel 内是运行时标量，但它们是**由编译期 shape 推出的运行时值**，不是动态轴。）

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1, 2], target="npuir")
def _adaptive_max_pool2d_with_indices_func(block_m=None) -> ...   # 去掉源 GPU 的 threads 参数（K9）
```

> 本分支 `target="npuir"` 的 `tilelang.jit` **没有 `compile_flags` 形参**（同族前作 Stage 3 实测确认，`tilelang/jit/__init__.py`），
> 故源配置里的 `compile_flags=["-O3","-DENABLE_BF16"]` 直接弃用；bf16 支持不依赖该 flag。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**纯 Vector**。

**判定依据**：算子仅含逐元素向量运算（`T.vmax`/`T.vcmp`/`T.vselect`/`T.vmul`）、归约（`T.reduce_max`/`T.reduce_min`）、
cast（`T.vcast`）、索引构造（`T.vbrc`/`T.arange`/`T.reshape`）与布局重排（`T.transpose`），
无 `T.gemm` / matmul → 不需要 L1 + L0A/L0B/L0C（§1.6.0 R1/R4 已弃 Cube 路径——数学上不可表达）。
故核数按**纯 Vector 算子翻倍**规则取物理核数 ×2（§5.5）。

### 5.2 Block 划分

本算子的分块维度不是传统 GEMM 的 `(M,N)`，而是 **(通道块 CH) × (输出行 o_h)** 的任务维度：

```python
CH     = min(C, 64)                # 一般网格的通道向量块（8 的整数倍，且 ≥ 8）；主选值，D-3 备选见 §1.6.3
CH_deg = min(C, 32)                # 退化路径（C4/C4b）的通道向量块（容量驱动，见 §4.5；8 的整数倍）
# 一般网格用 CH；C4/C4b 用 CH_deg。下式 num_cblk 按分支取（C4/C4b 用 CH_deg，一般用 CH）：
num_cblk     = ceil(C / CH_deg) if DEGENERATE else ceil(C / CH)
# 逻辑单元数按编译期分支取（**单元粒度是 num_logical 的唯一定义源**，见 §5.5 ①）：
#   一般网格  单元 = (n, o_h, c_block)  ->  NUM_LOGICAL = N * out_H * num_cblk
#   C4 全局   单元 = (n, c_block)        ->  NUM_LOGICAL = N * num_cblk（out_H==1，与通用式同值）
#   C4b 恒等  单元 = (n, c_block)        ->  NUM_LOGICAL = N * num_cblk（整平面；≠ N*out_H*num_cblk）
NUM_LOGICAL = N * num_cblk if (out_H == 1 and out_W == 1) or (out_H == H_in and out_W == W_in) \
              else N * out_H * num_cblk
```

> **两个通道块常量**（v1 新增 `CH_deg`）：**一般网格** `CH = min(C, 64)`（`PL-1.4` 实测最优档，§1.6.3）；
> **退化路径（C4/C4b）** `CH_deg = min(C, 32)`——容量驱动（`CH=64` 时退化路径默认档超 192 KB，见 §4.5），**非性能裁决**。
> 两分支的单元粒度（`num_logical` 分支口径）与 §3.3 解码恒一致。

- **为什么 `num_logical` 按分支取**：`num_logical` 必须与「一个逻辑单元处理多少数据」**严格同粒度**，否则任务解码越界。
  一般网格单元为 `(n, o_h, c_block)`，退化路径（C4/C4b）单元为 `(n, c_block)` **整张平面**，所需核数不同；
  C4 因 `out_H==1` 两种口径同值，**唯一会分叉的是 C4b**：`(1,8,9,11)` 恒等 case 通用式给 9，而整平面单元只需 1 ——
  若沿用通用式 9 配 §3.3 解码 `n = uid // num_cblk` 将产生 `n=0..8`（`N=1`）**越界读/写**。
  本版采用整平面单元口径 `NUM_LOGICAL = N*num_cblk`，与 §3.3 解码、§4.3 C4b buffer、§5.5 ① / §6 全文唯一自洽。
- **CH = 64 的选择理由**（一般网格）：① C 轴为向量轴，CH 需为 8（fp16 向量宽）的整数倍——64 为 8 的整数倍；
  ② UB 预算充裕（spp ≈65 KB，§4.5）；③ 与 §5.5 分核方案配合使 `num_logical` 落在物理核数附近。
- **CH_deg = 32 的选择理由**（退化路径）：容量判据 `degenerate_bytes(CH_deg,…)·inflate ≤ 192 KB` 在 `CH_deg=32` 下满足
  （C4 ≈66.6 KB / C4b ≈14.7 KB，默认档 ×1.7 后 113.2 / 25.0 KB ✓），`CH=64` 时不满足（§4.5）。
- **`MAX_BIN_H = max_adaptive_bin_extent(H_in, out_H)`、`MAX_BIN_W = max_adaptive_bin_extent(W_in, out_W)`**：
  bin 循环的静态上界（编译期常数）；`bh`/`bw` 为运行时边界差。

**代表性 shape 的分块结果**（`CH` 列对退化路径给 `CH_deg`）：

| workload | N | C | H_in×W_in | out | CH / CH_deg | num_cblk | NUM_LOGICAL（按单元粒度） |
|----------|---|---|-----------|-----|----|----------|-------------|
| smoke `(1,8,4,4)` | 1 | 8 | 4×4 | 2×2 | 8 | 1 | 2（一般） |
| global-1x1 `(8,2048,7,7)` | 8 | 2048 | 7×7 | 1×1 | **32（CH_deg）** | 64 | **512**（C4 特化，`N*num_cblk` = 通用式同值） |
| spp-6x6 `(2,128,56,56)` | 2 | 128 | 56×56 | 6×6 | 64 | 2 | 24（一般） |
| nondiv-7x7 `(2,64,55,57)` | 2 | 64 | 55×57 | 7×7 | 64 | 1 | 14（一般） |
| test `(2,64,56,56)`→6×6 | 2 | 64 | 56×56 | 6×6 | 64 | 1 | 12（一般） |
| test `(1,96,28,30)`→7×5 | 1 | 96 | 28×30 | 7×5 | 64 | 2 | 14（一般；通道尾块 `real_c=32`） |
| test `(1,128,55,57)`→7×7 | 1 | 128 | 55×57 | 7×7 | 64 | 2 | 14（一般） |
| test `(1,16,11,13)`→`(5,None)` | 1 | 16 | 11×13 | 5×13 | 16 | 1 | 5（一般；H 非恒等，**非** C4b） |
| 3D `(16,11,13)`→(5,4)（op 层 unsqueeze 后 `(1,16,11,13)`） | 1 | 16 | 11×13 | 5×4 | 16 | 1 | 5（一般） |
| 恒等 `(1,8,9,11)`→`(9,11)` | 1 | 8 | 9×11 | 9×11 | **8（CH_deg）** | 1 | **1**（C4b 整平面单元） |

### 5.3 约束分析

- **对齐约束**：向量轴 C 恒为 8 的整数倍（fp16/bf16 向量宽 8）；`CH=64`（一般）与 `CH_deg=32`（退化）均为 8 倍数 ✓。`W_in`（55/56/57）仅作转置后的 row 长度，不作向量轴，不要求整除 ✓。
- **UB 容量**：一般网格 spp `CH=64` 总 ≈65 KB（默认 auto-multi-buffer ×1.7 → ≈111 KB；关闭档 ×1.12 → 72 KB）< 192 KB ✓；
  退化路径 C4（`CH_deg=32`）≈66.6 KB、C4b（`CH_deg=8`）≈14.7 KB，默认档 ×1.7 后 113.2 / 25.0 KB < 192 KB ✓（§4.5）。
  C4/C4b 另有编译期容量判据不等式 + 回退一般路径（§4.5）。
- **L0 容量**：无 Cube 计算，不适用 ✓。
- **索引位宽**：int32 累加器要求 `h_in*w_in < 2**31`（工厂期守卫；与源 `idx_dtype` 同界）。
- **非整除**：本算子**不存在 GEMM 式整除问题**——`total_output`、`W_in` 无需是 block 的整数倍：
  任务维度用 persistent `T.serial` + `if uid < NUM_LOGICAL` 守卫（尾块丢弃），row 长度用静态 `W_in`；
  通道尾块用 `real_c = min(CH, C - cb*CH)` 动态 copy 范围（值与索引两路写回**都只用 `[0:real_c]`**），padded lane 永不写回 GM。
  bin 非整除（重叠）由整数 `adaptive_bin` 精确处理（语义本来就要求）。

### 5.4 注意事项

- **尾单元**：`NUM_LOGICAL` 非 `num_kernels` 整数倍时（global-1x1: 512 / 40 = 12.8 → `num_local = 13`），核内 `T.serial(num_local)` 的 `if uid < NUM_LOGICAL` 守卫丢弃越界任务。
  **退化路径的 `NUM_LOGICAL` 按其整平面单元粒度取**（`N*num_cblk`）。
- **通道尾块**：`C % CH != 0`（如 C=96 → CH=64 + `real_c=32`）——值/索引两路的 copy/写回**均**用 `real_c` 动态范围；
  `row_src/row_dst/row_f32/eq_f32` 的 padded lane 残留旧值，但 padded lane 的 `acc`/`max_idx`/`nan_idx` 永不写回 GM（各 lane 独立），无污染。
- **`kh`/`kw` 越界守卫**：`MAX_BIN_H`/`MAX_BIN_W` 是 bin 的**最宽**上界；对较窄的 `o_h`/`o_w`，守卫跳过越界元素（不读取 `>= ih1`/`>= iw1` 的输入）——与源守卫同构（§0.4 #1）。
- **零输出保护**：`out_H/out_W > 0` 由 op 层保证（`_normalize_output_size` 拒绝 `<=0`）。
- **NaN / 索引语义**：必须在所有输入（含 `±inf`/subnormal）下保持「NaN 胜出 + last-NaN 索引 + ±inf 取 +inf + 并列取 first-hit」——
  由 §1.6.1 N1/N2/N3 的向量化形态保证（`verify_equiv.py` 7 项全 `EQUIV_PASS`）。
- **值/索引判决同源**：值与索引的 NaN 判决都取 `cfinal = (nan_idx >= 0)`（而非值路径的 `h_flag==0`），确保二者**永不出现「值为 NaN 而索引取 max_idx」的错配**（`h_flag` 仅作为 N1 的独立机制保留与校验）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

- **① 逻辑核数计算**（**单元粒度 = `num_logical` 的唯一定义源**）：
  - **一般网格**（非恒等、非全局）：单元 = `(n, o_h, c_block)` → `num_logical = N × out_H × ceil(C / CH)`（`CH=min(C,64)`；pure-Vector 无 `block_N` 维度，故只有通道块维 × 输出行维）。
  - **退化路径 C4 全局**（`out_H==out_W==1`）：单元 = 整张 `(n, c_block)` 平面 → `num_logical = N × ceil(C / CH_deg)`（`CH_deg=min(C,32)`）。因 `out_H==1`，该值与通用式**同值**（global-1x1 `8 × ceil(2048/32) = 8 × 64 = 512`），无分叉。
  - **退化路径 C4b 恒等**（`out_H==H_in and out_W==W_in`）：单元 = 整张 `(n, c_block)` 平面 → `num_logical = N × ceil(C / CH_deg)`，**显式不同于**通用式 `N × out_H × ceil(C/CH)`。
    依据：若沿用通用式而单元仍为整平面，§3.3 解码 `n = uid // num_cblk` 在 `N=1` 下会产生越界（见 §5.2）。
  代表值见 §5.2 表：smoke 2 / global-1x1 **512**（C4，`CH_deg=32`）/ spp 24 / nondiv 14 / 恒等 1。

- **② 物理核数依据（实查）**：
  本设计环境实际执行查询（2026-10-14，项目 venv 下）：
  ```python
  from tilelang.utils.npu_utils import NPUUtils
  NPUUtils.get().get_aicore_num()
  # 实查返回值 = 20   (设备 Ascend910B3，`npu-smi info` 显示 910B3，本设计会话)
  ```
  **纯 Vector 算子核数翻倍**：`vector_cores = 20 × 2 = 40`。
  （**不采用** pattern-library `CONST-aicore-910B2C` 的 910B2C 值 24；本设备为 910B3，必须实查。实查须在项目 venv
  `/workspace/venv/bin/python` 下执行，系统 `python3` 缺 `torch` 会误报「环境无运行时」。）

- **③ 规模判定与分核方案**（三选一，逐 workload 判定）：

  | workload | NUM_LOGICAL | 物理核 | 判定 | 方案 |
  |----------|-------------|--------|------|------|
  | smoke `(1,8,4,4)`→2×2 | 2 | 40 | **逻辑核数 ≤ 物理核数** | **无需适配**：`num_kernels = min(2,40) = 2`，2 个单元各占 1 核 |
  | spp / nondiv / 多数测试（12–24） | 5–24 | 40 | **逻辑核数 ≤ 物理核数** | **无需适配**：`num_kernels = NUM_LOGICAL`，每核 1 个单元 |
  | `(1,8,9,11)` 恒等（**C4b**） | 1 | 40 | **逻辑核数 ≤ 物理核数** | **无需适配**：`num_kernels = 1`，1 核执行 1 个整平面单元（`N*num_cblk=1`）；**消除**了「通用式 9 + 整平面单元 → 越界」 |
  | global-1x1 `(8,2048,7,7)` | **512** | 40 | **极大规模**（无法通过调小 block 缩减核数到 ≤40，且核数非 40 的整数倍） | **核内串行（persistent）**：固定启动 `num_kernels = min(NUM_LOGICAL, vector_cores) = 40`；核内 `for t in T.serial(num_local)` 串行，`num_local = ceil(512/40) = 13`（**静态值**）；`uid = t*num_kernels + cid`，`if uid < NUM_LOGICAL` 守卫 |

  **通用规则（factory 级）**：`NUM_LOGICAL` 按分支粒度取（见 ①），`num_kernels = min(NUM_LOGICAL, vector_cores)`，
  `num_local = ceil(NUM_LOGICAL / num_kernels)`（**Python 整数编译期常量**，非 `T.ceildiv` 表达式）。
  当 `NUM_LOGICAL ≤ vector_cores` → `num_local = 1`，退化为「每核 1 单元 = 无需适配」；当 `NUM_LOGICAL > vector_cores` → 核内串行（persistent）。
  **核内串行循环边界 `num_local` 为静态值**（`NUM_LOGICAL`/`num_kernels` 均为编译期常数）——满足 `core-split-strategy.md` §1 要素③硬要求。
  > **不做空核的保证**：因 `num_kernels = min(NUM_LOGICAL, vector_cores)`，故 `num_kernels ≤ NUM_LOGICAL`，每个物理核的首个任务 `uid = cid < NUM_LOGICAL` 恒有效。
  > 分核增益预期：global-1x1 若按 GPU 式超发 512 核，NPU 运行时会将 472 核串行排入 40 核 → 引入核启动开销；persistent 化后固定 40 核、每核 13 个静态任务（512 = 40×12 + 32，负载差 1 个任务）。
---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block/tile 级调度循环。**不出现逐元素标量循环**。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 逻辑单元（task） | block 级并行 | `T.Kernel(num_kernels)` | 每核处理 1 个或多个 `(n,o_h,c_block)` 单元 |
| 初始化（值/标记/两个索引） | **整块填充/清除**（无循环） | `T.vbrc(-inf, acc)` + `T.vbrc(1.0, h_flag)` + `T.vbrc(ih0*W_in+iw0, max_idx[ow])` + `T.vbrc(-1, nan_idx)` | 单元入口整 buffer 初始化（§1.6.2 #0a–#0d） |
| 核内多任务 | 迭代（persistent） | `T.serial(num_local)`（静态界）+ `if uid<NUM_LOGICAL` | 极大规模时摊薄核启动开销（§5.5）；静态边界 |
| bin 的 H 方向 | 迭代（归约链） | `T.serial(MAX_BIN_H)` + `if kh < bh` | tile 级顺序依赖 + 动态边界（§1.6.2 #13）；循环体内是向量 op |
| 输出列 `o_w` | 迭代（任务内调度） | `T.serial(out_W)` + bin 边界内联 | 与 `kh` 同形（§1.6.2 #18：tile 级调度 + 运行时候选边界，循环体内是向量 op；`TRAP-tir-range-loop-not-unrolled`：Python `range` 亦降为串行 TIR 循环）；`out_W` 静态且小 |
| bin 的 W 方向 | 迭代（归约链） | `T.serial(MAX_BIN_W)` + `if kw < bw` | tile 级顺序依赖 + 动态边界（§1.6.2 #14）；循环体内是向量 op |
| 元素级（C 轴） | 向量化 | `T.vmax` + `T.vcmp` + `T.vselect` + `T.vmul`（向量轴 = C，§1.6.3 决策） | C 轴 lane 并行，无逐元素循环 |
| 索引构造（退化） | 向量化 | `T.arange` + `T.reshape` + `T.vbrc` | 1-D arange 优先形态，禁 2-D 标量物化 |
| 退化路径归约 | 单次归约 | `T.reduce_max` / `T.reduce_min`（C4） | 全局池化无 bin 循环 |

### 6.2 循环伪代码

```python
# num_kernels / num_local 由 NUM_LOGICAL 导出；NUM_LOGICAL 按编译期分支的单元粒度取（§3.3/§5.5①）：
#   一般 = N*out_H*num_cblk；C4 全局 = N*num_cblk；C4b 恒等 = N*num_cblk（整平面）
with T.Kernel(num_kernels, is_npu=True) as (cid, _):        # 一维网格（≤2 维合法）
    for t in T.serial(num_local):                            # persistent（静态界）
        uid = t * num_kernels + cid
        if uid < NUM_LOGICAL:                                # 尾任务守卫（与 NUM_LOGICAL 同粒度）
            # 解码为 (n, o_h, c_block)【一般】/ (n, c_block)【C4/C4b】后进入 §3.3 的单元体
            ...
            for kh in T.serial(MAX_BIN_H):                   # 静态界
                if kh < bh:                                  # bin H 边界守卫
                    [T.copy → row_src; T.transpose; T.vcast; T.vcmp(eq)→T.vcast(eq_f32)]
                    for ow in T.serial(out_W):               # 任务内调度（非展开）
                        for kw in T.serial(MAX_BIN_W):       # 静态界
                            if kw < bw:                      # bin W 边界守卫
                                T.vbrc(flat, fvec)
                                # first-hit 索引：gt 必须与 **更新前** 的 acc 比较（次序约束）
                                T.vcmp(row_f32[iw], acc[ow], gt_b, "gt")             # first-hit（先于 vmax(acc)）
                                T.vselect(gt_b, fvec, neg1, cand)
                                T.vmax(max_idx[ow], cand, max_idx[ow])
                                T.vmax(acc[ow], row_f32[iw], acc[ow])                 # 值（后更新 acc）
                                T.vmul(h_flag[ow], eq_f32[iw], h_flag[ow])            # NaN 标记
                                T.vcmp(eq_f32[iw], zero_e, isn_b, "eq")              # last-NaN
                                T.vselect(isn_b, fvec, neg1, nanc)
                                T.vmax(nan_idx[ow], nanc, nan_idx[ow])
```

> **次序约束（关键，v1 修正）**：`T.vcmp(row_f32[iw], acc[ow], gt_b, "gt")` **必须先于** `T.vmax(acc[ow], row_f32[iw], acc[ow])`
> 执行（使 `gt` 与**更新前**的 `acc` 比较）。若颠倒（v0 形态），`gt ≡ (v > max(a,v)) ≡ False` ⇒ `cand ≡ −1` ⇒ `max_idx` 恒为初值 `f0`
> （索引系统性错误）。见 §3.3 与 §1.6.1 `N3-ord` 守卫。

### 6.3 流水线优化

**不使用 `T.Pipelined`**。理由：
- 单元内只有一条 `GM→UB→计算→UB→GM` 单阶段流，无多阶段 K 循环可流水；`kh` 循环次数少（`MAX_BIN_H` 1–10），双缓冲收益有限。
- Developer 模式下编译器可自动 multi-buffer；若 Stage 4 实测显示 MTE2 未与 Vector 重叠，可在 `kh` 循环上探索 `T.Pipelined(num_stages=2)`（列入 Stage 4 候选，不改主结构）。
- C4/C4b 退化路径为「1 次搬运 + 1 次归约 / 纯拷贝」，无流水空间。

### 6.4 尾块处理

- **任务尾块**：`NUM_LOGICAL` 非 `num_kernels` 整数倍 → `if uid < NUM_LOGICAL` 丢弃越界任务。退化路径 `NUM_LOGICAL` 按整平面单元粒度取（§5.5 ①）。
- **通道尾块**：`real_c = min(CH, C - cb*CH)` 动态 copy/写回范围（**值与索引两路都用**）；padded lane 永不落 GM。
- **bin 尾块（H 方向）**：`kh >= bh` → 守卫跳过，不读入界外行。
- **bin 尾块（W 方向）**：`kw >= bw` → 守卫跳过，精确覆盖 bin，无越界、无掩码需求。
- **无 GEMM 式整除要求**（§5.3）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**自动同步（Developer 模式）**。

### 7.2 同步点说明

无手动同步点。Developer 模式下编译器在 `T.copy`（MTE2 搬运）与 v-prefix 向量 op（Vector 单元）之间自动插入同步标志，
`T.transpose`（VTransposeOp）、`T.reduce_max`/`T.reduce_min`（VReduceOp）、`T.arange`（VArangeOp）与 `T.vbrc`（VBrcOp）
亦由编译器编排。本算子无跨核通信、无 Cube/Vector 跨引擎依赖：

| 位置 | 同步 API | 理由 |
|------|----------|------|
| （无） | — | Developer 自动同步；本算子单引擎（Vector）、单阶段，无 `sync_block_set/wait` 需求 |

> **注（`T.arange` 的流水语义）**：`docs/Tilelang.language/创建操作/T.arange.md` §2.3.1 记录「1-D `arange` 走
> `varange strides[%c1]` + VADDS 扩展」路径（PIPE_V 建模一致），而 2-D 形态走标量写且缺跨流水同步。
> 本设计**只用 1-D 形态**（C4/C4b 的 `k1d`/`idx1d`）+ `T.reshape`（元数据级），规避该风险；若 Stage 3 观测到
> arange 与后继向量读的可见性问题，按文档 §2.3.2 建议在两个 op 之间插入 `T.pipe_barrier("PIPE_ALL")`（保守同步，仍属自动同步范畴的局部加固）。

### 7.3 pass_configs 配置

`@tilelang.jit(out_idx=[1, 2], target="npuir")`；不设置自定义 `pass_configs`（与 `lerp_tensor` / `logsumexp` / 同族前作一致）。
若 Stage 3/4 出现 UB 膨胀导致溢出，可评估 `--enable-auto-multi-buffer=false`（`CONST-capacity-910B2C` 2026-09-16 反例数据点）。

### 7.4 CV 融合设计

**不适用**：本算子为**纯 Vector**（无 `T.gemm`、无 Cube 引擎参与），不存在 Cube↔Vector 的核间流水与跨引擎同步，
故无 CV 融合设计（与同族 avg/max 前作一致）。跨核协同亦不需要（无跨核归约、无 workspace 中继）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（优先移植源仓参考实现/测试基准），**不得复刻 §0.6 的 NPU 算法**——保证验证独立性。

```python
import torch
import torch.nn.functional as F


def golden_adaptive_max_pool2d_with_indices(x: torch.Tensor, output_size):
    """PyTorch reference for adaptive_max_pool2d(return_indices=True), per DESIGN.md section 0.1.

    Directly uses the source-repo test benchmark
    (see ``ref_program`` in examples/TileOPs/tests/ops/test_pool.py).  Accepts 3D CHW or
    4D NCHW; ``None`` entries keep the input size.  Returns (values, indices) where
    ``indices`` is int64 and holds the flat ``ih * W_in + iw`` index (ties keep the first
    hit; a NaN in the window makes the value NaN and records the last NaN visited).
    Independent of the NPU algorithm (no transpose / no C-vectorisation / no index
    accumulator / no C4-C4b specialisation).  Runs on CPU and does not require an NPU device.
    """
    size = (None, None) if output_size is None else output_size
    val, idx = F.adaptive_max_pool2d(x.detach().to(torch.float32), size, return_indices=True)
    return val.to(x.dtype), idx
```

> **索引校验方式**：`indices` 为 int64，是**精确可判定 oracle**（`VP-2026-0131`：int64 索引 `torch.equal` 是免费 oracle），
> 用 `torch.testing.assert_close(idx, ref_idx, atol=0, rtol=0)`（整数不设容差）。

### 8.2 精度标准

| 输出 | atol | rtol | 说明 |
|------|------|------|------|
| `out`（值） | 0 | 0 | max 是**选择**而非累加，输出逐位等于窗口内某输入元素（§0.1）；NaN 位置用 `equal_nan=True` |
| `indices` | 0 | 0 | 整数精确相等（int64） |

> 依据：`examples/TileOPs/tests/ops/test_pool.py` 用例 `test_adaptive_max_pool2d_indices` 注释
> "Max is exact (no accumulation) and the index comparison is exact too"，用例 `test.check(op, *inputs, atol=0, rtol=0)`；
> 与本设计的 fp32 比较域 + 单次 `rint` 舍回一致（fp16/bf16→fp32→dtype 无损）。
> **无 fp32 用例**（op 层 dtype gate 拒绝 fp32）。值比较在含 NaN 的用例上须 `equal_nan=True`。

### 8.3 L0 门槛测试计划（必须通过）

| # | 用例 | 目的 |
|---|------|------|
| L0-1 | `(2,64,56,56)` out `(6,6)` fp16 | 测试用例 smoke-spp-6x6（一般网格） |
| L0-2 | `(1,96,28,30)` out `(7,5)` bf16 | smoke-asymmetric-bf16（**bf16 上转 + 非对称 bin + 通道尾块 `real_c=32`**） |
| L0-3 | `(1,128,55,57)` out `(7,7)` fp16 | full-nondiv-fp16（非整除重叠 bin） |
| L0-4 | `(1,16,11,13)` out `(5,None)` fp16 | full-partial-none-fp16（部分 None → H 非恒等，一般网格） |
| L0-5 | `(8,2048,7,7)` out `(1,1)` fp16 | manifest global-1x1（**C4 特化 + persistent 512 单元（`CH_deg=32`，`num_local=13`）**） |
| L0-6 | `(1,8,9,11)` out None fp16 | **C4b 恒等**（`NUM_LOGICAL=1` 收敛路径 + arange 常量索引模式） |
| L0-7 | `(1,8,4,4)` out `(7,7)` fp16 | **扩张 bin**（段长 1–2，`MAX_BIN_*` 静态容纳） |
| L0-8 | 3D `(16,11,13)` out `(5,4)` fp16（op 层 unsqueeze） | CHW rank 保持（**两路输出均回 3D**） |
| L0-9 | **语义门禁（fp16）**：NaN/±inf/并列注入（`(1,16,11,13)`→`(5,4)`） | 值：`equal_nan=True` 全量比对；专项断言「入窗 NaN → 值 NaN」、**「两个 NaN → 索引取后一个」**、**「并列 → 索引取首者」**、`{+inf,-inf}` 同窗 → 值 `+inf` 且索引 = `+inf` 的扁平下标、全 `-inf` 窗 → 值 `-inf` 且索引 = 首个元素扁平下标 |
| L0-9b | **语义门禁（bf16）**：与 L0-9 **同注入集合**（入窗 NaN / 两个 NaN / 全并列 / `{+inf,-inf}` / 全 `-inf`），shape `(1,16,11,13)`→`(5,4)` bf16 | 覆盖 op 层 dtype gate 的**第二条路径**（bf16 须先 `vcast` 到 fp32 再比较，§9.1）：断言值与 fp16 同口径（`equal_nan=True` 全量比对）+ 索引 `atol=rtol=0` 精确比对 + `cfinal` 机制观测（同 L0-9） |
| L0-10 | `block_m` guard（传 0/负） + `block_m=256` 派发 | factory 接口兼容性（K9 保留但校验）+ wrapper 契约 |
| L0-11 | `indices` 专项：形状/dtype 断言 | `idx.shape == out.shape == (N,C,out_H,out_W)` 且 `idx.dtype == torch.int64` |

> 完整分层套件（L1 功能全覆 / L2 极端尺寸 / Boundary 特殊值）交由 `tilelang-op-develop`，不在此枚举。
> 精度对照：值 `torch.testing.assert_close(out, ref_val, atol=0, rtol=0, equal_nan=True)`；
> 索引 `torch.testing.assert_close(idx, ref_idx, atol=0, rtol=0)`。
> **机制级证据要求**（同族前作教训：门禁用例通过 ≠ 语义机制生效）：
> ① `h_flag` 的 NaN 免标记与 `nan_idx>=0` 判决须给出 **机制级观测**（例如注入 NaN 后检查 `cfinal` 分支确被走到 / 最小 repro），
> 不可仅以「值输出 NaN」通过——该结果可能由 `T.vmax` 的 NaN 传播兜底产生（同族前作首版即出现该假阳性）。
> ② **first-hit 次序**须给出**实现级检查**：用「窗口最大不在首个被访问元素」的最小 shape（如 `arange(12)` 3×4 → `(2,2)`，`bin(1,1)` 索引须为 11）
> 断言索引 = 破纪录元素扁平下标，而非窗口首元素 `f0`（v0 的次序缺陷会在此暴露；§1.6.1 `N3-ord` 守卫在 D-1 层已拦住同型回归）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

- **NaN 语义必须由 IEEE 正确的 `eq` 掩码实现**：不得依赖 `T.vmax` / `T.reduce_max` 的 NaN 行为（**未文档化**）；
  **不得**用 `T.vcmp(...,"ne")` 做 NaN 探测（`ne` 是 ordered-NE，NaN 操作数下恒假，见 `TRAP-vcmp-ne-ordered-nan`）。
  本设计用 `eq = (x == x)` 取反 + 掩码乘性累积（`h_flag`）+ 索引侧 `nan_idx >= 0` 判决。
  已由 `verify_equiv.py` 的 N3/N3b 两行证明结果**与 `vmax` 的 NaN 约定无关**。
- **int32 索引的转置形态**（**本设计新增风险，须 Stage 3 首要验证**）：`docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1
  明确列出 int32/int64 受支持，但**本仓 examples 无 int32 `T.transpose` 先例**（argmax 用 fp32 承载索引）。
  **回退方案（已设计）**：改为 **fp32 承载索引**（`max_idx`/`nan_idx`/`fvec`/`cand`/`nanc`/`idx32`/`idx_t` 均为 fp32，
  与 argmax 族完全同形），转置 fp32 有先例；此时须加工厂期守卫 `h_in*w_in <= 2**24`（fp32 整数精确域），
  且 `-1` 哨兵改为 `-1.0`、`BIG` 类常量按 fp32 语义调整。两条路径的**语义完全一致**（索引均为整数），
  仅位宽与守卫界不同——Stage 3 若遇 int32 转置问题按此回退，不构成设计层阻塞。
- **in-place 向量运算的 loop-carried 陷阱**：**禁用** `T.vselect(cond, A, B, B)`（dst 别名某源）在 `T.serial` 循环里做
  loop-carried 更新（`TRAP-vselect-inplace-carried-state`，实测评测：停在 tile-0 局部值）。
  本设计的 loop-carried 更新只用**原地 `T.vmax`/`T.vmul`**（`PL-1.2-caxis-accum` 已验证形态，同族单输出前作 Stage 3 实测通过），
  且 `T.vselect` 一律写入**独立 scratch**（`cand`/`nanc`/`idx32`/`out32`），dst 不与任何源别名。
- **bf16 计算路径**：`T.vmax`/`T.reduce_max`/`T.reduce_min`/`T.vsub` 不支持 bf16 → bf16 必须先 `vcast` 到 fp32
  （与 fp32 比较域一致，无需额外代价）；索引累加器 int32 与 dtype 无关。
- **bool 不可落 GM**：`eq_mask`/`gt_b`/`isn_b`/`cfinal`/`mb`/`cond` 必须留片上（`T.vcmp` §2.3）。
- **Python `range` 非编译期展开**：`o_w`/`kw` 一律写 `T.serial(max_*)` + 守卫（`TRAP-tir-range-loop-not-unrolled`）。
- **int32 索引位宽守卫**：工厂期要求 `h_in*w_in < 2**31`（与源 `idx_dtype` 同界）；超出时报错
  （**已确认的限制**：源在 `>= 2**31` 时切 int64 承载，本设计不支持——该量级（单片 ≥21 亿元素）在任何现实 workload 与 manifest 中都不存在，列入风险点）。
- **UB 膨胀**：默认 auto-multi-buffer ~1.7x（关闭档 ~1.11）；退化路径（C4/C4b）驻留缓冲随 `H·W` 线性放大，
  故其通道块**收窄为 `CH_deg = min(C, 32)`**，并须以**编译期容量判据不等式** `degenerate_bytes(CH_deg,…) × inflate ≤ 192 KB` 决定是否走特化，
  否则**回退一般路径**（`hardware-cost-model.md` §5.1；同族前作实测回退且无精度损失）。§4.5 给出 C4/C4b 的逐 buffer 复算与代表 shape 判定。
- **3D UB buffer**：`T.alloc_ub` 文档称 3D~5D「有待充分验证」；C4/C4b 用 3D 形态与同族前作一致（已有实证）。
  若 Stage 3 遇阻，回退 2D `(CH, H*W)` + `reshape` 视图（Stage 3 细节）。
- **`compile_flags` 形参不存在**：本分支 `tilelang.jit`（`target="npuir"`）无该形参，勿传。
- **被引 pattern-library 条目的重验状态**：`PL-1.4-tiling-heuristic` / `PL-1.1` / `PL-1.2` / `PL-1.3` /
  `TRAP-vselect-inplace-carried-state` 为 fresh；`CONST-capacity-910B2C` / `CONST-mte2-degradation` /
  `CONST-vector-launch-overhead` / `TRAP-vcmp-ne-ordered-nan` / `TRAP-tir-range-loop-not-unrolled` 被 `kb_stale_check.py`
  标记 `stale`（版本戳落后于本会话工具链 `ba60e99c…`）。各条的使用方式与重验/兜底见 §1.6.0 末「引用的 pattern-library 条目重验状态」
  表——**stale 条目或只作硬件规格级/量级级参照（有自洽算术或回退路径兜底），或不承载决策，或已由本会话机器验证 / 同族 Stage 3 实测独立佐证**。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| `acc` 未置 `-inf` | 用 `T.clear` 置 0 后直接 `vmax` | **全负窗口输出 0（错误）** | 必须 `T.vbrc(-T.infinity("float32"), acc)`（max 的单位元是 `-inf`） |
| NaN 探测用 `vcmp("ne")` | 试图用 `v != v` 探测 | **掩码恒假**（ordered-NE）；端到端可能被 `vmax` 的 NaN 传播兜底而"假通过" | 用 `T.vcmp(...,"eq")`（IEEE 正确）取反 + 掩码乘性累积（`h_flag`），并用 `nan_idx>=0` 做索引侧判决 |
| **索引追踪用 `vmin`/最小扁平** | 误以为 first-hit = 「最小值对应的最小扁平」 | **索引错误**（§1.6.1 初版即此错，`verify_equiv.py` 拦下：`N3` 首轮 20426 mismatch） | 源记录的是**最后一条记录**；破纪录元素扁平递增 ⇒ first-hit = **破纪录集合的扁平最大值**，用 `vselect(gt, fv, -1)` + `vmax` |
| **first-hit 与值累加的实现次序颠倒** | 先 `T.vmax(acc, row, acc)` 再 `T.vcmp(row, acc, gt, "gt")`（v0 缺陷，§3.3/§6.2） | **索引系统性错误**：`gt ≡ (v > max(a,v)) ≡ False` ⇒ `cand ≡ −1` ⇒ `max_idx` 恒为初值 `f0`（窗口首元素；`arange(12)→(2,2)` 的 `bin(1,1)` 得 6 而非 11） | `T.vcmp(gt)` **必须先于**原地 `vmax(acc)`（`gt` 与更新前的 `acc` 比较）；机器验证见 §1.6.1 `N3-ord`；机制级检查见 §8.3 |
| `max_idx` 初值缺失/置 0 | 全 `-inf` 窗口无任何记录 | 索引落到 0（错误） | `max_idx` 初值 = 首个被访问元素扁平 `ih0*W_in+iw0`（用 `T.vbrc` 填） |
| in-place `T.vselect(cond,A,B,B)` | loop-carried 索引更新 | 静默丢状态（停在 tile-0 值） | 写独立 scratch + 原地 `vmax`/`vmul` |
| `NUM_LOGICAL` 与单元粒度不一致（C4b） | 恒等路径沿用通用式 `N*out_H*num_cblk` | `n = uid//num_cblk` 越界读写 | C4b/C4 用 `N*num_cblk`（§5.5 ①） |
| `kh`/`kw` 动态边界 | 直接 `T.serial(bin_h(o_h))` | TileLang 拒绝动态边界 | 静态 `MAX_BIN_H`/`MAX_BIN_W` + `if k < extent` 守卫 |
| 值与索引的 NaN 判决不同源 | 值用 `h_flag==0`、索引用 `nan_idx>=0` 各判一次 | 极端情形下二者可能错配（值为 NaN 而索引取 max_idx） | 统一取 `cfinal = (nan_idx >= 0)` 作为两路判决条件（§5.4）；`h_flag` 保留作 N1 机制校验 |
| UB 溢出 | block/CH 过大或退化路径平面过大 | 编译失败 | 减小 CH；或关 auto-multi-buffer（§7.3）；或让退化路径回退一般路径（§4.5） |
| 通道尾块越界 | `C % CH != 0` 时按 CH copy | 越界读/写 | `real_c = min(CH, C - cb*CH)`（**两路输出都用**） |
| vcmp/vselect 形状不一致 | cond 用 `(CH,)` 而数据 `(1,CH)` | 编译/校验失败 | cond 与数据切片同形 `(1,CH)`；常量 buffer 也用 `(1,CH)` |
| `T.arange` 用 2-D 形态 | 直接 `T.arange(buf (M,N), [N,1], 0)` | 标量物化 + 跨流水可见性风险 | 用 1-D `T.arange` + `T.reshape`（+ `T.vbrc`） |

### 9.3 特殊场景处理

- **非整除分块**：通道尾块 `real_c` 动态；任务尾块 `if uid < NUM_LOGICAL`；bin 重叠由整数 `adaptive_bin` 精确处理。
- **极小 shape**：`(1,1,1,1)`（global == identity，identity 优先 → 纯拷贝 + 索引 0）；`C<8`（`CH=8` + `real_c<8` 处理）。
- **扩大输出**（`out > in`，如 `(1,8,4,4)`→`(7,7)`）：bin 宽度 1–2，`MAX_BIN_H/W` 静态容纳；语义 = 元素复制/邻居最大值（含索引跟随）。
- **混合精度**：bf16 经 fp32 中转；值单次 `rint` 舍回；索引 int32→int64。
- **NaN/±inf**：所有路径（一般网格 / C4 / C4b）均保持语义；C4b 恒等路径天然保留 NaN（纯拷贝），索引为常量模式（与元素值无关）。
- **并列**：严格 `gt` 保证 first-hit；C4 的整平面 argmax 用 `min{k : v_k = max}` 显式实现 first-hit（与一般网格同语义）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/adaptive_max_pool2d_with_indices/_adaptive_max_pool2d_with_indices_kernel/
├── DESIGN.md                                   # 本设计文档
├── verify_equiv.py                             # §1.6.1 采纳优化项的等价性机器验证（D-1）
├── RETROSPECTIVE.md                            # Stage 1 复盘（Phase 8）
└── _adaptive_max_pool2d_with_indices_kernel.py      # 算子实现（Stage 3 产出）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 设计文档（本文件） |
| `verify_equiv.py` | ✅ 已完成（**7 项**全 `EQUIV_PASS`；含 `N3-ord` 次序回归守卫） | §1.6.1 等价性机器验证（N1/N2/N3/N3b/**N3-ord**/C4/C4b） |
| `_adaptive_max_pool2d_with_indices_kernel.py` | ⬜ 待实现（Stage 3） | NPU kernel（双输出）+ golden + 分层测试 |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘 |

### 10.3 命名规范

- 项目目录名：`adaptive_max_pool2d_with_indices`（snake_case，与 manifest op_slug 一致）
- 算子目录名：`_adaptive_max_pool2d_with_indices_kernel`（snake_case，与迁移函数名一致）
- 实现文件：`_adaptive_max_pool2d_with_indices_kernel.py`
- 输出契约与参数名保持 `_adaptive_max_pool2d_with_indices_kernel(n, c_in, h_in, w_in, out_h, out_w, dtype)` 不变（`threads` 已按 K9 移除）。
- 返回双输出 `(out, indices)`（`out_idx=[1, 2]`）。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py，7 项全 PASS，含 `N3-ord` 次序回归守卫）
2. ⬜ Golden 函数（§8.1，验证基准）
3. ⬜ 算子实现（`_adaptive_max_pool2d_with_indices_kernel.py`）+ L0 精度比对（值 `equal_nan=True, atol=rtol=0`；索引精确）
4. ⬜ 接入 NPU wrapper（`examples/TileOPs/tileops/kernels/pool/adaptive_max_pool2d_with_indices/`），跑 `test_pool.py` / `bench_pool.py`

### 10.5 与 GPU 源接口的对照（迁移契约）

| 项 | GPU 源 | NPU 目标 |
|----|--------|---------|
| factory 签名 | `_adaptive_max_pool2d_with_indices_kernel(n,c_in,h_in,w_in,out_h,out_w,dtype="float16")` | **不变** |
| 返回 callable | `_build(block_m, threads)` | `_build(block_m)`（K9：`threads` 移除，`block_m` 保留兼容/被忽略） |
| `@tilelang.jit` | `out_idx=[1, 2], compile_flags=[...]` | `out_idx=[1, 2], target="npuir"`（**无 `compile_flags`**） |
| 输出 | `(out, indices)`；`indices` int64 同形 | **不变** |
| wrapper custom_op | `top::adaptive_max_pool2d_with_indices_wrapped_kernel` | `npub::adaptive_max_pool2d_with_indices_wrapped_kernel`（脚手架已改） |
| `default_config` | `{"block_m": 256, "threads": 256}` | `{"block_m": 256}`（脚手架已改） |
| 索引承载位宽 | kernel 内 `int32 if h_in*w_in < 2**31 else int64` | **不变**（int32 累加 + 落盘 cast int64；`>= 2**31` 报错，见 §9.1） |
| 索引语义 | 扁平 `ih*W_in+iw`；first-hit 并列；NaN → last-NaN 索引 | **不变**（§1.6.1 N2/N3 机器验证） |
