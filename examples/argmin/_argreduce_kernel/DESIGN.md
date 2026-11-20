# `_argreduce_kernel`（承载 `ArgminFwdOp`）算子设计文档

> **迁移任务（migration-harness）**：GPU TileOPs `ArgminFwdOp` 的 TileLang kernel 函数
> `_argreduce_kernel` → TileLang-NPUIR（`target="npuir"`，Developer 模式）。
> **源算子**：`/workspace/TileOPs-gpu/tileops/kernels/` 下 `reduction` 目录的 `argreduce.py`（`_argreduce_kernel` 工厂 +
> `ArgreduceKernel`；GPU 侧 `ArgminFwdOp` 与 `ArgmaxFwdOp` 共用本 kernel，`op_kind` 选分支）。
> **本设计基于**：源码全文解读（Phase M0）→ 算法族调研（Phase R，调研四问）→ 硬件耦合性判定与 NPU 重设计
> （Phase M1，四态处置）→ 算法级优化设计（Phase 2）→ **等价性机器验证**（D-1，`verify_equiv.py`，
> E1–E5 全部 `EQUIV_PASS`；v1 补强 E2/E4 判别力并加负向对照，见「修订说明」与 §1.6.1）。
> **工具链戳**：tilelang `0.1.2+ac0522ed260caa64c885a8c57b46d3651efca7ce`（dev root build；仓库 HEAD
> `2026-10-14 12:49`）+ CANN 9.0.0 + torch_npu 2.9.0.post8 + **Ascend910B3**（npu-smi 26.1.1）。
> **设备实查（本任务实测，与同族 argmax 迁移任务的 910B2C/24 核不同——常数与陷阱的跨设备复用风险已按条目版本戳逐条标注）**：

| 实查项 | 查询代码 | 返回值 |
|--------|----------|--------|
| AI Cube 核数 | `tilelang.utils.npu_utils.NPUUtils.get().get_aicore_num()` | **20** |
| AI Vector 核数 | `tilelang.utils.npu_arch.AscendArch().aivector_core_num`（等价 `NPUUtils.get().get_aivector_core_num()`） | **40** |
| 片上 UB 容量 | `AscendArch().ub_cap`（等价 `max_smem_usage`） | **196608 B（192 KB）** |
| 芯片名 | `AscendArch().chip_name` | `Ascend910B` |

> 物理核数与片上容量**均为实查值**（§5.5 分核三要素要素②的实查记录），不使用文档假设或经验值。

---

## 修订说明（v1，revision mode）⭐

> 本版由 Stage 2 检视（结论「通过」、0 阻塞级）后的 5 条**建议级改进**触发定向修订。**§0 / §1.6 / §5 等
> 已通过门禁 1 的实质结论保持稳定**，本版仅作定点最小编辑，不改变算法族（C1+C2 双路径）、API 选型、
> 分核方案与 Tiling 主选。

**相对上一版（`history_version/design_v0.md`）的关键调整**：

| # | 触发建议 | 上一版 | 本版调整 |
|---|---------|--------|---------|
| A1 | 建议 1（验证覆盖） | `verify_equiv.py` E2 基线含死赋值（`-max(-x)` 结果被 NaN-忽略最小值的等价实现覆盖）、E4 为精确单射恒真——两组结构性无判别力 | 重写脚本：E2 基线改为**真正的 `-max(-x)` 变换域路径**（fmax NaN-忽略语义化，无死赋值），与直接域 `reduce_min` 真实对照；E4 改为 **fp16/bf16 原生 dtype 流水 vs fp32 上抛流水**的真实跨域对照；**新增「判别力对照」组**（末次出现 / 过小哨兵 / 有损粗化域三个故意错误候选）把「脚本能报出差异」变成机器断言；重跑回写 §1.6.1 结果表 |
| A2 | 建议 2（内存记账） | §4.5 称 `idx_*`（含随 bm 放大的 `idx_j`）计入 8 KB reserve——叙述不准确 | §4.5 记账改为**探针定标的代理预算**：仅计 `x_ub`/`x_frag(+work)`/`cand`，其余 `(bm,N_padded)` 类缓冲由编译器 PlanMemory 活跃期复用/别名吸收；明确跨设备（910B3）须 Stage 3 复标定，代理预算不作为可编译证明 |
| A3 | 建议 3（内部一致性） | §1.6.2 结论段「保留 4 处（#8–#11）」与表体（#9 已 ✅ 向量化）矛盾 | 结论改为「保留 **3 处**（#8 / #10 / #11）」，同步「替代结论」列语义（#9 明确属已替代侧） |
| A4 | 建议 4（Tiling 阶梯） | §5.2/§3.3 基础阶梯对 `p=1` 误用 `B_multi`（bm=1 实含 `ext_brc`） | 基础梯改为 `p · N_padded · (B_bm1 if p == 1 else B_multi) <= UB_MANUAL_BUDGET`（同族已验证 Stage-4 形态），并补预算过滤器注 |
| A5 | 建议 5（复杂度口径） | §1.6.0-R3 C1 记「≈3·N_padded 向量 op」（未含 `vbrc` 广播） | 改记「**≈4–5·N_padded 向量 op**（含 `vbrc` 广播与候选 `reduce_min`）」，与 §1.6.2-#6 一一对应 |

**为何不会再犯同一错误**：

- A1：脚本现对每组断言「候选与基线**不共用同一实现路径**」——E2 基线走 `-max(-x)`（不再有覆盖语句），
  E4 基线走 fp32 上抛流水而候选走原生 dtype 流水；并新增独立「判别力对照」组：若脚本判别力失效，
  对照组将报告 0 违反 → 脚本自身判定失败（`CONTROL_BROKEN`）。判别力由机器断言而非人工目检保证。
- A2：记账口径改为「代理预算 + PlanMemory 别名吸收」并显式声明其**非可编译证明**属性——不再把任意
  per-block 放大缓冲声称为 8 KB reserve 可吸收；跨设备复标定要求落入 §9.2-R4 与 Stage 3 L0-4 探针。
- A3/A4/A5：三处均为**表述/计数一致性**问题（表体已正确，结论段/表达式回调了未修正形态）——本版以
  「表体与结论逐项对齐 + 与同族已验证 Stage-4 形态对账」方式修正，属可机械复核项。
- **机械误触规避（本版附加）**：本版把源路径改为「`reduction` 目录下 `argreduce.py`」的分列写法，并避免
  正文出现表示「不适用」的裸占位词（统一写「不适用」）——防止下游检视报告机械引用同一连写串时误触
  `S2-MIG-DIM0` 规则（该规则对维度 0 正文的相邻三字符序列做字面匹配）。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

**数学语义**：对 2D 输入 `x ∈ ℝ^{M×N}` 沿最后一维（规约维 N）求**最小值位置索引**：

$$\mathrm{argmin}(x)[i] = \min\{\, j \in [0, N) \;:\; x[i,j] = \min_{k<N} x[i,k] \,\}$$

即 **first-occurrence（首个极值位置）语义**——tie 时返回最小索引，与 `torch.argmin` 的首现保证一致
（源码以「串行扫描 + 首个匹配 `T.loop_break()`」实现该语义；harness 测试用 `torch.equal` **精确比对**索引）。

**规约语义**：规约轴 = N。Op 层已把任意 rank 输入 reshape/movedim 到 `(M, N)`（`dim=None` 全张量规约被
flatten 成 `(1, numel)`，见 `_ReduceOpBase._prepare_input`）；keepdim 不进 kernel（Op 层后处理）；输出
`(M,)` int64。**min 是精确选择运算**——无累加顺序问题、无舍入（这是 argreduce 与 sum 类规约的本质差异，
§1.6.1-E4 的基础）。

**dtype 语义**：输入 fp16/bf16/fp32；GPU 源码把数据先 cast 到 **fp32** 再做 reduce 与匹配（对 argmin 而言
该 cast 数值上是非必要的，见 §0.5 处置与 §1.6.1-E4）；**输出恒 int64**。

**边界语义**（源码 + torch 语义 + 本工具链行为共同确定；§8.2 L2 逐项记录）：

| 边界 | torch（CPU） | 源 GPU 实现 | 本设计 NPU 算法 | 处置 |
|------|--------------|-------------|-----------------|------|
| `+inf` / `-inf` | 正常参与比较 | 正常参与 | 正常参与 | 等价域内 |
| **NaN** | 返回**首个 NaN 索引** | fmax 语义**忽略 NaN**；全 NaN 行 `row_extreme` 保持初值 → 扫描永不匹配 → **输出 0**（源码 `T.fill(out_idx, 0)` 回退值） | `reduce_min` **传播 NaN** → 掩码匹配全失败 → 输出哨兵 `2^30` | **三方各异，文档化边界（非等价域）**；`verify_equiv.py` INFO 行实测记录，不进 L0 门禁 |
| `±0.0` | IEEE 相等（`-0.0 == +0.0`）→ 取首现 | IEEE 相等 | IEEE 相等（`vcmp "eq"`） | kernel 与 **torch-CPU golden 一致**；torch-NPU 设备实现把 ±0 视为可区分全序（argmin 偏好 `-0.0`），混合符号零且行极值恰为 0 的行上两者分歧；`randn` 输入下概率为 0 |
| 空维 `N=0` | 报错 | kernel 构造即 raise | 工厂期 raise | 保留 |
| 非整除 `N` | — | Op 层 host `F.pad` 到 `align_up(N,256)`，pad 值 = 单位元（argmin → **+inf**） | **主选保留该 pad 契约**（§0.2/§0.6-R3） | §5.4 |
| `M=0`（非规约维含 0） | 输出空 | 源实现未覆盖（grid=0） | 列为风险（§9.2-R6），Op 层短路建议 | 风险登记 |

**host 侧语义**：reshape/transpose/pad/keepdim 均为 **Op 层实现手段**而非算子语义（`tileops/ops/reduction/reduce.py`
模块注释明言 kernel 转换完成后应消除 host pad——"until their kernels are converted"）；golden 不得复刻这些
实现细节。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据（`torch.argmin`，first-occurrence + int64 + IEEE 相等
匹配；torch NaN 语义为文档化边界）。

### 0.2 源算子输入输出（含 NPU 侧输入契约与集成前置核对）

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x` | 源契约 `(M, N_padded)`，`N_padded = align_up(N, 256)` | float16 / bfloat16 / float32 | M = 非规约维乘积，N = 原始规约维长度（工厂参数携带**原始 N**）；GPU 契约下 Op 层已 pad（argmin pad **`+inf`**） |
| `out` | `(M,)` | int64 | 每行一个首现最小值索引（迁移必需字段：输出 shape `(M,)`，无转置布局） |

**工厂契约（harness 固定，接口约束）**：`_argreduce_kernel(M, N, op_kind, dtype)` 返回
`_func(block_m) → kernel(x)`（GPU 侧还带 `threads`，NPU 侧按 K9 去除——`TRAP-threads-kwarg-noop` 亦证
`T.Kernel(threads=)` 在 npuir 无效）。语义参数 `M/N/op_kind/dtype` 不变；`op_kind ∈ {argmax, argmin}`
（本迁移单元的测试范围是 **argmin**；argmax 为独立迁移单元，但工厂接口保留双 kind 支持，见 §8.2 L0-10）。

**NPU 侧输入契约决策（集成前置核对，VP-2026-0134 / VP-2026-0128）**：

| 契约 | Op 层 `_kernel_handles_padding` | `_pad_value()` | Op 实际传入宽度 | kernel 必须声明的宽度 | 本设计处置 |
|------|--------------------------------|----------------|-----------------|----------------------|-----------|
| **主选：pad 契约** | `False`（目标 Op 层现值，继承 `_ReduceOpBase` 默认） | **`+inf`**（`examples/TileOPs/tileops/ops/reduction/argmin.py` 第 76–78 行） | `N_padded = align_up(N, 256)` | **`N_padded`** | ✅ 采用（零 Op 层改动） |
| 备选：raw-N 契约 | `True`（需一行属性翻转） | 不参与（无 host pad） | `N` | `N` | ⏸ 记录（触发条件见下） |

**核对动作（实读目标工件）**：

1. `examples/TileOPs/tileops/ops/reduction/argmin.py`：`ArgminFwdOp` **未覆写** `_kernel_handles_padding`
   → 继承 `examples/TileOPs/tileops/ops/reduction/reduce.py` 第 91 行 `_kernel_handles_padding = False`；
   `_pad_value()` 返回 `float("inf")`——**argmin 的 pad 方向为 `+inf`**（argmax 为 `-inf`；共享 kernel
   接入两个算子时必须核对这一方向性，方向反了会让 pad 列在 argmin 中胜出，属静默错值）。
2. `examples/TileOPs/tileops/ops/reduction/reduce.py` 第 460–472 行：`_kernel_handles_padding=False` 时
   Op 层执行 `F.pad(x, (0, N_padded - N), value=self._pad_value())`，即把 `(M, N)` 变 `(M, N_padded)` 后
   才调 kernel。
3. **一致性规则（写入实现约束）**：本设计主选下，kernel 工厂内部派生 `N_padded = align_up(N, DEFAULT_ALIGNMENT)`
   并**声明 `T.Tensor((M, N_padded), dtype)`**（与源同构）——声明宽度与 Op 传入宽度**逐元素一致**。
   宽度错位的症状（VP-2026-0128 / CASE-argmax-argreduce-integration 实证）：**行 0 正确、行 ≥1 全错且不报错**
   （行步长错绑）。§8.2 L0-6 以非对齐 `(256,300) → N_padded=512` 用例作该缺陷的判别用例。
4. **wrapper 侧（`examples/TileOPs/tileops/kernels/reduction/argmin/argmin.py`）**：现 `default_config`
   仍用 GPU 期 48 KB SMEM 启发式（`SHARED_MEMORY_BUDGET_BYTES // (N_padded × elem)`）——CASE 第 ② 条实证该
   启发式在 NPU kernel 上直接 `ub overflow`。实现期须改为转调 kernel 模块的 `_select_config(M, N, dtype)`
   （kernel 为 tiling 单一事实源，§5.2）。

**主选依据（为何保留 pad 契约而非翻转 raw-N）**：

- 目标 Op 层**此刻就是** `_kernel_handles_padding=False` + `_pad_value=+inf`（实读结论），保留契约＝零 Op 层
  改动、集成面最小；
- **argmin 的 manifest 负载不含窄 N 负载**（`examples/TileOPs/tileops/manifest/reduction.yaml` 的 `ArgminFwdOp`
  段：smoke `[32,256]`、lm-head `[4,102400]`、hidden-state `[2048,4096]`、non-aligned `[256,300]`——全部
  `dim=-1`，N ∈ {256, 102400, 4096, 300}）。仅 `300 → 512` 一处产生 pad，读流量 +72% 但**绝对增量为
  108.5 KB/负载**（§1.6.0 D-2 实算），非带宽主导；raw-N 契约在本算子上**没有 manifest 级收益**
  （与 argmax 的 3D `dim=0` 负载不同——那里 pad 造成 32× 总流量膨胀）；
- 附带收益：pad 契约下 `N_padded ≥ 256` 恒成立 → **窄内维（N < 256）域在本算子不可达**，同族 argmax
  迁移中出现的窄 N 陷阱族（`(2048,4)` 手工 64 KB → 实测 ~4× 膨胀、deinterleave 宽视图路径）整体不适用，
  设计空间显著收窄（§5.2）。

**备选 raw-N 契约（记录，不在本次主选）**：翻转 `ArgminFwdOp._kernel_handles_padding = True`，kernel 声明
`(M, N)`，非整除由**静态尾 tile** 处理（同族 argmax 已验证形态）。触发条件：① 集成层要求 argmin 承接
`dim=0`/中间维负载（如新增 3D 非末轴 workload）；② Stage 4 实测显示 pad 读取开销成为带宽瓶颈。
两契约输出完全一致（pad 值为单位元，§1.6.1-E5 机器验证 `EQUIV_PASS`）。

> **与 queue VP-2026-0128 的关系（显式说明，避免误读为忽略该 pending 提案）**：VP-2026-0128 把
> 「`_kernel_handles_padding=True` + kernel 原始 N 契约」标为**小 N 负载的最高杠杆项**（最高 64× 流量改善）
> ——该结论来自 argmax 的 3D `dim=0` 负载（`N=4` → pad 到 256），其前提是 **manifest 含窄 N 负载**。
> 本算子（argmin）的 manifest **不含**窄 N 负载（N ∈ {256, 102400, 4096, 300}，§0.2），最高杠杆不成立
> （唯一 pad 位形 `300→512` 仅 +108.5 KB 读流量）；而 pad 契约方向正确性是**正确性关键项**（`+inf` 而非
> `-inf`），保留契约恰好把该关键项固定在 Op 层单点。故本单元**先行采用 pad 契约**（零改动、最小集成面），
> 并把 raw-N 登记为可激活备选——与 VP-2026-0128 不矛盾（其高杠杆前提在本算子不满足），亦与
> VP-2026-0134 的「先核对再决定」要求一致（核对结论已落 §5.6 表）。

### 0.3 实现算法解读（怎么算）

GPU 源码核心（`_argreduce_kernel`，两步法）：

**计算步骤分解**（覆盖源码全部计算语句；`_argreduce_kernel` 第 35–107 行）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `T.copy(x[pid_m * block_m, 0], shared_buf)`：block 行块 GM→SMEM 整块载入 | x (M,N_padded) | shared_buf (bm,N_padded) | 数据载入 |
| 2 | `T.Parallel(block_m, N_padded)` 循环 `T.cast` → fp32 | shared_buf | x_f32 fragment | 精度提升（对 argmin 非必要，§0.5） |
| 3a | argmax 分支：`T.fill(row_extreme, -inf)` + `T.reduce_max(x_f32, row_extreme, dim=1, clear=False)` | x_f32 | row_extreme (bm,) | 行最大值 |
| 3b | **argmin 分支**：`T.Parallel` 取负 → `T.reduce_max(neg_x, row_extreme, dim=1, clear=False)` → `T.Parallel` 取负还原（`-max(-x)` 变换） | x_f32 | neg_x fragment + row_extreme | 行最小值 |
| 3c | `T.fill(out_idx, T.cast(0, "int64"))`（第 95 行）：扫描回退值初始化 | — | out_idx (bm,) int64 | **无匹配行的回退语义**（全 NaN 行 → 输出 0，§0.1） |
| 4 | 串行扫描：`for i in T.Parallel(block_m): for j in T.Serial(N): if x_f32[i,j]==row_extreme[i]: out_idx[i]=j; T.loop_break()`（第 96–100 行） | x_f32, row_extreme | out_idx (bm,) int64 | **首现索引**（early-exit 平均 N/2 次标量比较；扫描上界为**原始 N** 而非 N_padded） |
| 5 | `T.copy(out_idx, out[pid_m * block_m])` | out_idx | GM out | 输出搬运 |

**数据流**（源硬件视角）：`GM[x] →T.copy→ SMEM[shared_buf] →T.Parallel cast→ register/fragment[x_f32]
→reduce/scan→ fragment[row_extreme, out_idx] →T.copy→ GM[out]`；argmin 分支比 argmax 多一个 `neg_x`
fragment 往返（取负 + 还原两遍全宽 pass）。

**循环与并行结构**：`T.Kernel(T.ceildiv(M, block_m), threads=threads)` —— 一维 grid × CUDA 线程两级并行；
`block_m ∈ {1,2,4,8}` 由 48 KB SMEM 预算启发式选择（`default_config`，第 182–216 行）；`N < 256`（重 pad）
时附加 `block_m × N_padded ≤ 2 × threads` 的 GPU copy-layout 约束；**无流水**（无 `T.Pipelined`）。

**host 侧逻辑**：Op 层 validate →（dim 归一 / `movedim` / `contiguous`）reshape `(M,N)` → `F.pad` 到
`N_padded`（单位元 `+inf`）→ 调 kernel → 输出 reshape 回原始形状（keepdim 处理）；
`functools.lru_cache(maxsize=32)` 按 `(M, N, op_kind, dtype)` 缓存编译产物。

**编译配置**：`@tilelang.jit(out_idx=[1])`（GPU target）；无特殊 `pass_configs`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | SMEM tiling（256 元素对齐整块 `T.copy`） | 合并访存、一次载入多次使用 | 行块驻留 SMEM，reduce 与 scan 共享 | shared memory + 256 对齐 copy 指令约束 | 硬件强相关 |
| 2 | 两步法（并行 reduce 找极值 + 串行扫首匹配） | 极值用并行归约（快），首现索引用标量扫描（语义必需） | `reduce_max` 树形归约 + early-exit 标量扫描 | CUDA 线程级并行 reduce 原语 | 算法层可移植；scan 的标量形态硬件相关 |
| 3 | fp32 累加域 cast | 数值稳健（对 sum 类规约必要） | 全数据 cast fp32 后再 reduce | — | 对 argmin **非必要**（min 为精确选择运算，§1.6.1-E4） |
| 4 | argmin 的 `-max(-x)` 变换 | 复用 argmax 通路（一次实现双算子） | 取负 → `reduce_max` → 取负还原 | — | 纯数学恒等，可移植；但 NPU 有 `T.reduce_min` → 变换冗余（§0.5） |
| 5 | host pad + 单位元填充（argmin `+inf`） | 满足 256 对齐的非整除处理 | Op 层 `F.pad`，pad 列不改变结果 | GPU copy 的 256 对齐需求 | 对齐约束硬件相关（NPU 仅需 32B 尾轴）；**单位元技巧可移植** |
| 6 | `block_m` SMEM 预算启发式 | 行块尽量大以摊薄 block 开销 | 48 KB 预算 + `[1,2,4,8]` 阶梯 | SMEM 容量 | 预算值硬件相关 |
| 7 | `lru_cache` 工厂缓存 | 消重复编译 | Python 层 `(M,N,kind,dtype)` 键 | — | 可移植 |
| 8 | 重 pad 布局约束（`bm×N_padded ≤ 2×threads`） | 规避 TileLang GPU copy 布局推断限制 | 限制 `block_m` | GPU copy-layout 推断 | 硬件强相关（NPU 不适用） |

> **识别完整性**：以上覆盖源码全部优化性结构（步骤分解表已把每条计算语句归入步骤 1–5，含 3a–3c 分支）；
> 源码为朴素两步法，**无** warp shuffle / mma / swizzle / 异步流水等 GPU 深度优化。
> **源实现缺陷（本设计修正，Stage 3 不得照抄）**：第 68 行 / 第 103 行的 `T.copy` **无 M 尾块守卫**
> （`M % block_m ≠ 0` 时越界读/写；harness 用例 `(129,512)` 且 `block_m=8` → `129 % 8 = 1` 即触发位形）
> ——见 §0.5 对应行与 §5.4 修正三件套。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（输入含 Phase R 调研结论 §1.6.0）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| first-occurrence argmin 语义（含 int64 输出） | 语义 | 无 | — | **保留** | 语义层无条件保留；golden = `torch.argmin` |
| 两步法结构（极值归约 → 首现索引） | 算法 | 无（纯算法层） | — | **保留** | 结构保留；两步的**实现形态**分别等价替换/重设计（下两行） |
| 步骤 3b 极值归约（`-max(-x)` + `T.reduce_max`） | 算法 | CUDA 线程归约 | 有：Vector `T.reduce_min/​max(dim=1)` | **等价替换** | 直接 `T.reduce_min`（§0.6-R1/E2）；`docs/Tilelang.language/规约操作/T.reduce_max.md`、`docs/Tilelang.language/规约操作/T.reduce.md`。**实测约束：reduce dst dtype 必须与 src 相同**（fp16 src→fp32 dst 静默产出错误值，queue VP-2026-0119 派生条目 + `TRAP-reduce-mixed-dtype-silent-corrupt` 待合入）；**bf16 不被 reduce 支持**（`T.reduce.md` §2.2.1）→ 预 cast fp32 |
| 步骤 4 串行扫描 + `T.loop_break` | 算法 | CUDA 标量线程 early-exit | 无等价（NPU Vector 无 per-lane 标量 early-exit） | **重新设计** | → §0.6-R1：向量化「掩码索引取 min」：`cand[j] = j if x[j]==m else BIG`；`first = reduce_min(cand)`（§1.6.1-E1，机器验证 `EQUIV_PASS`） |
| 步骤 2 fp32 cast（fp16/bf16 输入） | 算法/精度 | — | — | **舍弃（fp16）/ 保留改造（bf16）** | fp16：min 为精确选择运算，fp16 域 reduce + 匹配与 fp32 域**索引恒等**（§1.6.1-E4，`EQUIV_PASS`），省 1 个全宽 pass + 2 B/elem；bf16：reduce 不支持 + bf16→fp32 精确，保留 cast（`T.vcast` `rint`）。**禁同 dtype `T.vcast`**（`TRAP-vcast-same-dtype-rounds`：f32→f32 退化为四舍五入到整数、静默坏值）→ fp32 路径用同 dtype `T.copy` |
| 步骤 3b `-max(-x)` argmin 变换（#4） | 算法 | — | 有更优：直接 `T.reduce_min` | **等价替换** | 直接 `reduce_min` + 同一首现索引机制；索引恒等（§1.6.1-E2，`EQUIV_PASS`）；省 2 个全宽 pass（取负 + 还原） |
| SMEM tiling / 256 对齐 `T.copy`（#1） | 优化 | shared memory | 有：UB（`T.alloc_shared`，实查 192 KB） | **等价替换** | GM→UB staging + fragment 计算（`migration-analysis.md` §5.3 行「shared memory → L1/UB」）；对齐需求降为 32B 尾轴（`examples/TileOPs/tileops/kernels/reduction/logsumexp/` 的 N=300 先例） |
| host pad + 单位元（非整除处理，#5） | 优化 | GPU 256 对齐 | pad **语义**可移植；256 对齐约束非必要 | **保留（pad 单位元语义）/ 舍弃（256 对齐约束）** | §0.6-R3：主选保留 Op 层 pad 契约（`+inf` 单位元，零改动的集成面）+ kernel 声明 `N_padded`；**同时**论证「在 N_padded 全域计算候选不改变输出」（§1.6.1-E5），故该契约无正确性代价 |
| block × thread 两级并行 / `threads` 参数（#8） | 优化 | CUDA 线程 | 无（本项目一维 `T.Kernel`） | **重新设计** | → §0.6-R2：一维 persistent grid（实查 **40** 个 Vector 核）+ `T.serial` 静态边界任务循环；`threads` 参数按 K9 去除 |
| `block_m` 48 KB SMEM 启发式 + 布局约束（#6/#8） | 优化 | SMEM 容量 / GPU copy 布局推断 | 换 UB 预算模型 | **重新设计** | → §0.6-R4：沿用同族已验证的 UB 预算公式（§5.2 表：手工 ≤64 KB/block，×2 双缓冲 ≤128 KB < 192 KB） |
| M 尾块守卫缺失（源第 68/103 行 `T.copy` 无 `M % block_m` 边界检查） | 正确性（**源缺陷**） | GM 定长 block copy 无边界检查 | 有：`T.min` 标量 + src/dst 双显式 slice | **重新设计** | §5.4 三件套（`real_m = T.min(bm, M−start)` + 双 slice + 垃圾行不写出；VP-2026-0044 / logsumexp 先例）；**Stage 3 不得照抄源码的无守卫 copy 形态** |
| `lru_cache` 工厂缓存（#7） | 优化 | — | — | **保留** | 工厂层与硬件无关 |

> **无「舍弃后性能意图丢失」项**：#3 的精度意图由 §1.6.1-E4 的等价性论证承接（min 为精确选择无需 fp32 域）；
> #8 的布局规避意图随 GPU copy 约束消失而不再需要。每项判定依据：`migration-analysis.md` §5.3 映射表 +
> 本仓 `docs/Tilelang.language/` API 文档 + pattern-library 实测条目（逐条注明）。

### 0.6 NPU 算法重设计

**重设计项 R1：首现索引的向量化（源：串行扫描 + `T.loop_break`）**

- **源方案**：`T.Serial(N)` 标量扫描逐元素比较、命中即 `T.loop_break()`（平均 N/2 次标量操作；依赖 CUDA
  线程标量执行模型；性能意图 = early-exit 减少平均比较次数）。
- **NPU 新算法**：**掩码索引取最小**（§1.6.0 选定 C1）：

  $$\mathrm{first}[i] = \min_{j<N_{padded}}\;\big(\,j \cdot \mathbb{1}[x[i,j]=m[i]] + \mathrm{BIG}\cdot \mathbb{1}[x[i,j]\neq m[i]]\,\big),\qquad \mathrm{BIG}=2^{30}$$

  实现：`T.reduce_min`（极值 m，同 dtype）→ **显式向量 op 链** `T.vbrc`（广播极值）→ `T.vcmp(...,"eq")`
  → `T.vselect(cmp, idx_j, sent_v, cand)` → `T.reduce_min(cand, first, dim=1)` → int64 出栈。
  **为何用显式向量链而非 2D `T.Parallel` 融合 `T.if_then_else`**：queue CG-2026-0015 派生条目
  `TRAP-parallel2d-bm-ge2-scalarize` 实测「2D `T.Parallel(bm,N)` 融合 `T.if_then_else` 在 `bm≥2` 整体
  标量化（scalar ratio 0.84–0.95，(2,4096) 222µs；`bm=1` 同形态向量化正常）」，且该结论与操作数形态无关
  （三变体对照证伪）。故：**`bm≥2` 强制走显式向量链**；`bm=1` 两形态皆可（链式为统一形态）。
  early-exit 意图由「向量单元整行并行」承接（O(N) 向量 pass ≪ O(N/2) 标量操作，且标量 op 每次约 0.5 µs
  发射不可向量化）。
- **语义保持论证**：m 由实际数据归约得出、至少一个 j 命中（有限/±inf 输入）；非命中候选 `BIG=2^30`
  （fp32 精确 2 的幂）> 任何合法 j（`j < N_padded ≤ 2^24`，fp32 精确整数域）；`reduce_min` 取到**最小命中 j
  = 首现索引**，与源扫描逐元素等价（§1.6.1-E1 机器验证 264 行 `EQUIV_PASS`，含 tie/全 ±inf/亚正规/混合零
  /N∈{1,2,8,256,300,512,1024,4096}×3 dtype）。边界：NaN 行不命中（输出哨兵 `2^30`，§9.2-R1 文档化）。

**重设计项 R2：并行结构（源：block×thread 两级 + `threads` 参数）**

- **源方案**：一维 grid × CUDA threads；`threads` 同时是归约并行宽度与布局约束参数。
- **NPU 新算法**：纯 Vector 一维 persistent kernel——`T.Kernel(num_kernels, is_npu=True)`，
  `num_kernels = min(ceil(M/block_m), 40)`（**40 = 实查 `aivector_core_num`**，纯 Vector 核数；见 §5.5）；
  核内 `T.serial(静态上界)` 任务循环（grid-stride 映射 `task = cid + s·num_kernels`）+ 动态 guard
  `if start < M`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 示例先例；`PL-1.21-vector-persistent-split`）。
  元素级并行全部交给 Vector 向量化（显式 v-prefix op 链 + reduce 原语；`bm≥2` 不用 2D Parallel 融合循环，
  见 R1）。**计算缓冲全部用 fragment**（`T.alloc_shared` 仅作 GM staging）——同族实证 `shared` 多消费者缓冲
  在 auto-multi-buffer 流水下当双缓冲预算超限时产生**非确定性数据竞争**（fragment 计算形态在多波次下稳定）。
- **语义保持论证**：行块间无依赖（per-row 独立规约），任务循环只改变调度不改变计算；尾块
  `real_m = T.min(bm, M − start)` + src/dst 双显式 slice（logsumexp / VP-2026-0044 三件套先例），
  垃圾行不写出。

**重设计项 R3：padding / 宽度策略（源：host `F.pad` 到 256 对齐 + 单位元）**

- **源方案**：Op 层 `F.pad(x, (0, N_padded−N), value=单位元)`，满足 GPU SMEM copy 的 256 元素对齐；kernel
  声明 `(M, N_padded)` 但只扫描前 N 列（串行扫描上界为原始 N）。
- **NPU 新算法（主选）**：**保留 Op 层 pad 契约**——`ArgminFwdOp` 维持 `_kernel_handles_padding=False` +
  `_pad_value() = +inf`（零改动）；kernel 工厂内部派生 `N_padded = align_up(N, DEFAULT_ALIGNMENT)` 并声明
  `(M, N_padded)`；**候选计算在 N_padded 全域进行**（不再像源那样把扫描上界钉在 N）。
- **语义保持论证**：① pad 值为 argmin 的单位元（`+inf`），pad 列永不成为行最小值（除非全行皆 `+inf`，
  此时实列同样命中且索引更小）；② **在 N_padded 全域计算候选不改变输出**：实列索引恒 `< N ≤` pad 列索引，
  故「pad 列命中」⟹「某个更小索引的实列命中」⟹ `reduce_min` 取到的必然仍是实列首现索引——**消除 host pad
  不可能改变任何输出**（§1.6.1-E5 机器验证 264 行 `EQUIV_PASS`，含全 `+inf` 行）。收益：无需 kernel 侧掩码 /
  尾 tile 特化代码路径，控制流最简。代价：`N=300` 类负载多读 108.5 KB（§1.6.0 D-2），非带宽主导。
- **备选（记录）**：raw-N 契约 + 静态尾 tile（同族 argmax 形态）——触发条件见 §0.2。

**重设计项 R4：`block_m` / `tile_n` 选择（源：48 KB SMEM 启发式 + GPU 布局约束）**

- **源方案**：`SHARED_MEMORY_BUDGET_BYTES(48 KB) // (N_padded × elem)` + `[1,2,4,8]` 阶梯 + 重 pad 布局约束。
- **NPU 新算法**：UB 预算公式（沿用同族已验证模型，§5.2）：基础阶梯
  `block_m = max{p ∈ {1,2,4,8} : p·N_padded·B(dtype) ≤ 64 KB}`（`B` = fragment 形态字节数/元素，§4.5 表；
  64 KB = 竞态安全手工预算，×2 双缓冲 ≈ 128 KB < 192 KB 实查容量）；基础阶梯下逻辑 block 数 > 2×40 时
  启用扩展阶梯 `{16..2048}`（小 `N_padded`/大 M 负载，档位由同一预算过滤器封顶）；`N_padded·B > 64 KB 且
  p=1 仍超` 时切 **tiled online 路径**（强制 `bm=1`），`tile_n` 由 `compute_tile_n`（仓内原语
  `examples/TileOPs/tileops/kernels/reduction/_primitives.py`）按同预算 + 整除优先 + 0.9 裕度后置校正选取。
- **语义保持论证**：纯配置选择，不影响语义；预算边界由同族探针与 Stage 3/5 实测双向定标
  （通过点：fp16 bm=2@4096 = 64 KB、fp32/bf16 bm=1@4096；失败点：fp16 bm=4@4096 编译溢出）。
- **与源启发式的差异说明**：源启发式绑定 GPU SMEM 容量与 CUDA 布局推断；NPU 侧以**实查 UB 容量**与
  **40 核 persistent 分核**为约束重构（不照抄）。

**重设计项 R5：M 尾块守卫（修正源缺陷）**

- **源方案**：`T.copy` 无 `M % block_m` 检查（越界读/写风险，`(129,512)` bm=8 即触发）。
- **NPU 新算法**：`real_m = T.min(block_m, M − start)` + src/dst 双显式 slice，垃圾行不写出（VP-2026-0044 /
  logsumexp 三件套）。
- **语义保持论证**：只改变越界访问为有界访问，有效行结果不变；`M=0` 边界另登风险（§9.2-R6）。

### 0.7 标杆实现

- **源算子**：`/workspace/TileOPs-gpu/tileops/kernels/` 下 `reduction` 目录的 `argreduce.py`（本设计 §0.1–§0.4 解读对象）；
  Op 层 reshape/pad 策略 `/workspace/TileOPs-gpu/tileops/ops/` 下 `reduction` 目录的 `argreduce.py`（`_pad_value` 方向性）
  与 `/workspace/TileOPs-gpu/tileops/ops/reduction/reduce.py`（`_prepare_input` 的 pad/N_padded 契约）；
  对齐常数 `/workspace/TileOPs-gpu/tileops/kernels/reduction/_primitives.py`。
- **NPU 侧 harness 工件（本迁移单元的输入与落点）**：
  - 提取版 GPU 参考 kernel：`examples/TileOPs/tileops/kernels/reduction/argmin/_argmin_fwd_kernels.py`；
  - 已移植 wrapper（Part B，K5–K9 已适配、`threads` 已去除）：
    `examples/TileOPs/tileops/kernels/reduction/argmin/argmin.py`；
  - Op 层：`examples/TileOPs/tileops/ops/reduction/argmin.py`（`_pad_value() → +inf`）；
  - 规格：`examples/TileOPs/tileops/manifest/reduction.yaml` 的 `ArgminFwdOp` 段；
  - 精度/性能用例来源：`examples/TileOPs/tests/ops/test_argmin.py`、`examples/TileOPs/benchmarks/ops/bench_argmin.py`。
- **已迁移的同 kernel 姊妹算子（最高相似度参考）**：
  `examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/_argreduce_kernel_DESIGN.md`（v1 设计 +
  探针证据链）与 `examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/perf_opt/_argreduce_kernel.py`
  （Stage 4 终版 kernel：resident / narrow / tiled / C6 四路径）；wrapper
  `examples/TileOPs/tileops/kernels/reduction/argmax/argmax.py`。**注意**：该姊妹任务在 910B2C + CANN 8.5.0
  上完成（核数 24 → Vector 48），本任务为 910B3 + CANN 9.0.0（核数 20 → Vector 40）——引用的陷阱与性能数字
  均按版本戳标注「跨工具链/跨设备，待重验」（§9.1）。
- **golden 基准**：`torch.argmin`（§8.1）；harness 测试以 `x.argmin(dim=...)` 为 ref、`torch.equal` 精确比对。
- **同族已验证参考**：`examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/` 与
  `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_tiled/`（row-reduction 迁移基准形态）。

---

## 1. 概述

### 1.1 算子名称

`_argreduce_kernel`（承载 `ArgminFwdOp` 的 NPU kernel；工厂接口保留 `argmax` kind 以维持共享 kernel 契约）

### 1.2 功能描述

对 `(M, N_padded)` 输入沿 N 维（规约维）求**首个**最小值位置，输出 `(M,)` int64 索引；first-occurrence
tie-break，与 `torch.argmin` 精确一致。

### 1.3 数学公式

$$y[i] = \min\{\, j \in [0, N) : x[i,j] = \min_{k<N} x[i,k] \,\}$$

NPU 侧等价计算式（§1.6.1 优化后，供 §3.1 拆解；`N_padded = align_up(N, 256)`，pad 列值 = `+inf`）：

$$m[i] = \mathrm{reduce}_{\min,\,j}\,x[i,j];\qquad y[i] = \mathrm{int64}\Big(\min_{j<N_{padded}}\ \mathrm{ite}\big(x[i,j]=m[i],\ j,\ \mathrm{BIG}\big)\Big),\quad \mathrm{BIG}=2^{30}$$

大 N（超 UB 驻留）时的在线单遍递推（§1.6.0-C2，tile-0 初始化 + **严格小于**更新）：

$$\text{tile}_t:\ (c_m, c_f)=\Big(\min_{j\in t}x[\cdot,j],\ \min_{j\in t}\mathrm{ite}(x[\cdot,j]=\textstyle\min_t, j, \mathrm{BIG})\Big);\quad (r_m,r_f)\leftarrow\begin{cases}(c_m,\,t{\cdot}tn{+}c_f) & c_m<r_m\\ (r_m,r_f) & \text{else}\end{cases}$$

### 1.4 算法描述（迁移决策后的 NPU 侧算法）

按 N 驻留性双路径（§0.5/§0.6 决策 + §1.6.0 调研选定；与源算法差异逐条标注来源）：

**路径 S（resident，`N_padded × B(dtype) × bm ≤ 64 KB`，整行驻留 UB）**——覆盖 smoke `(32,256)`、
non-aligned `(256,300→512)`、hidden-state `(2048,4096)` 全部 manifest 负载：

1. `T.copy` GM 行块 → UB（src/dst 双显式 slice 到 `real_m`，§0.6-R5）→ 同 dtype `T.copy` UB → **fragment**
   （fp32 直接 `T.copy`，**禁 f32→f32 `T.vcast`**）；
2. `T.reduce_min(x_frag, m, dim=1)`（**同 dtype**；fp16 原域、bf16 先 `T.vcast(rint)` 预转 fp32、fp32 原域）→ 行极值 `m (bm,1)`；
3. `T.vbrc(m → ext_brc (bm,N_padded))`——极值广播同形化（`bm=1` 时 `(·,1)` 不变操作数在 Parallel 条件中
   触发编译器 i1 误型 bug；统一 `vbrc` 形态使 `bm=1/bm≥2` 同构）；
4. `T.vcmp(x_frag, ext_brc, cmp_eq, "eq")`（`docs/Tilelang.language/比较操作/T.vcmp.md`）；
5. `T.vselect(cmp_eq, idx_j, sent_v, cand)`（`docs/Tilelang.language/条件操作/T.vselect.md`）——候选
   `cand = j if x==m else BIG`（`idx_j` 为核内构建的列索引，见 §3.2 R6 形态）；
6. `T.reduce_min(cand, first, dim=1)` → 首现索引（fp32，精确整数域）；
7. `T.reshape(first, first_flat)` + `T.vcast(first_flat, out_ub, round_mode="rint")` 向量化 int64 出栈 →
   `T.copy(out_ub[0:real_m], out[start:start+real_m])`。

> 步骤 4–5 的**显式向量链**是 `bm≥2` 的强制形态（2D `T.Parallel(bm,N)` 融合 `T.if_then_else` 在 `bm≥2`
> 整体标量化，见 §0.6-R1 与 §9.1-C-2）；该链在 `bm=1` 与 `bm≥2` 统一使用。

**路径 T（tiled online，`N_padded·B > 64 KB 且 bm=1 仍超`）**——覆盖 lm-head `(4,102400)`：

1. tile 0（静态宽度 `tile_n`）：copy → fragment → `reduce_min` **直接写入 `running_min`** → 链式候选
   （局部 j）→ `reduce_min` **直接写入 `running_idx`**（tile-0 直写初始化，消除哨兵初始化在全 `+inf` 行上
   的边界陷阱——严格小于更新下若初值设 `BIG` 会丢失首现，§1.6.1-E3 论证）；
2. `T.serial(num_full-1)`（静态边界）逐全 tile：copy → fragment → `chunk_min` → `vbrc` → `vcmp/vselect` 候选
   → `reduce_min` → `chunk_first_local`；`T.Parallel(bm)` 全局化 `chunk_global = t·tn + local`；
   `vcmp(chunk_min, running_min, "lt")` + 两个 `vselect` **无别名**更新（`T.vselect(cond, A, B, scratch)` +
   `T.copy` 回写——原位 `vselect(cond, A, B, B)` 在多迭代 `serial` 循环丢失 loop-carried 状态，
   见 §9.1-C-7）；
3. 静态尾 tile（`tail = N_padded mod tile_n` 工厂期常量，独立静态宽度代码路径）同 2；
4. 出栈同路径 S。

**与源算法差异**：串行扫描 → 显式向量链 + `reduce_min`（§0.6-R1）；fp16 fp32 cast 舍弃（§1.6.1-E4）；
argmin `-max(-x)` 直译为直接 `reduce_min`（§1.6.1-E2）；并行结构 persistent 一维 40 核（§0.6-R2）；
`block_m`/`tile_n` UB 预算（§0.6-R4）；M 尾块守卫（§0.6-R5）。算法结构（两步：极值 → 首现索引）**保留**
自源码（§0.5）。

### 1.5 数据流图

```
路径 S（resident，纯 Vector，单引擎单 kernel）：
GM[x] --T.copy(slice,real_m)--> UB[x_ub] --T.copy--> FRAG[x_frag]
  --T.reduce_min(dim=1)--> FRAG[m (bm,1)]
  --T.vbrc--> FRAG[ext_brc (bm,N_padded)]
  --T.vcmp("eq")--> FRAG[cmp_eq (bm,N_padded) bool]
  --T.vselect(idx_j, sent_v)--> FRAG[cand (bm,N_padded) fp32]
  --T.reduce_min(dim=1)--> FRAG[first (bm,1) fp32]
  --T.reshape + T.vcast(rint)--> UB[out_ub int64] --T.copy(slice)--> GM[out]

路径 T（tiled online）：每 tile 重复 [GM→UB→FRAG→chunk 极值→链式候选→reduce_min→全局化→lt 更新]，
running_(min,idx) (bm,1) fragment 跨 tile 保持在片上；尾 tile 静态宽度独立缓冲。
```

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

> **调研对象**：first-occurrence argmin 这一数学语义的**算法族**（GPU 源算法只是基线候选之一）。
> 信息源：`references/algorithm-candidates.md`（`ALG-reduction` 命中行：R1 「max+sum 合并扫描 / 和与平方和
> 一次扫描」、R2「分块归约+合并 / Welford」、R4「水平 vs 垂直 lane 映射」+ row-reduction 迁移基准形态）+
> `kb_search.py` 预注入条目（CASE-argmax-argreduce-integration / VP-2026-0128 / VP-2026-0134 /
> VP-2026-0125 / TRAP-npuir-arange-pattern-with-whole-plane-reduce）+ 本仓 `examples/`（logsumexp single/tiled、
> argmax 姐妹 kernel、ssd_chunk_scan）+ pattern-library（constants / traps-compiler / layout / elementwise /
> cases）+ **用户输入公式与源算法（基线）** + 恒等变形空间自查。
> argreduce 属**规约类**（多步：极值 + 索引）→ **完整调研**（R1 候选表 + R3 四口径对比 + R4 逐候选评估）。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| 0 | **基线：GPU 源两步法** | `-max(-x)` 求极值 + 串行扫描首匹配（early-exit） | — | — | ✅（基线） |
| 1 | **C1 向量化两步（驻留式）** | `first = reduce_min_j(ite(x_j=m, j, BIG))` | 数学恒等（m 必被 ≥1 个 j 命中；`BIG>N_padded` 且 fp32 精确） | 串行标量扫描 → 2 个向量 pass；消除 `loop_break` 标量依赖 | ✅ |
| 2 | **C2 在线单遍（流式）** | running `(min, first-idx)` 对 + **严格小于**更新（tile-0 直写初始化） | 数学恒等（比较-only，无值域算术；归纳论证见 §1.6.1-E3） | 大 N 单遍 GM 扫描；中间缓冲 O(N)→O(tile)；UB 驻留可行 | ✅ |
| 3 | C3 int64 复合键打包 | `key = monotonic_bits(x)·2³² + (2³²−1−j)` 单次 `reduce_max` 后解码 | 数学恒等（IEEE 位序单调映射；argmin 用保序取反） | 单 reduce 取代两步——但键构造需 `vbitcast`+移位/按位或链 ~5+ 全宽 pass，**净亏**（fp32 键 mantissa 24 bit 不足以容纳 值 11 + j 17 bit 的打包需求，**结构上必须 int64**；int64 键 8 B/elem 亦吃预算） | ❌（pass 数劣于 C1/C2） |
| 4 | C4 全排序取首 | `argsort(x)[0]` | 数学恒等 | O(N log N) ≫ O(N) | ❌（复杂度劣） |
| 5 | C5 argmin 的 `-max(-x)` 变换（源码方案） | 取负 → argmax → 还原 | 数学恒等（严格单调且保等） | 复用单实现；但 NPU 有 `reduce_min` → 变换冗余（2 个全宽 pass 纯开销） | ✅（作为 E2 基线对照） |
| 6 | C6 跨核 N-split partial+merge 两段 kernel | 列分片并行 → workspace `(M,nchunk)×(min,idx)` → merge kernel 归并 | 数学恒等（merge 用**严格小于**跨 chunk 的 first-idx 归并，与 C1 同构） | 小 M 大 N 负载核利用率 `ceil(M/bm)/40 → min(nchunk,40)/40` | ⏸ Stage 4 候选（完整结构 + 判定阈值 + 回写路径见 §1.6.4；`PL-1.20-nsplit-perrow-partial` 已验证） |

**R2 有没有在线算法？结论：有**。argmin 的在线变体 = running `(极值, 首现索引)` 对（结构判据：argmin 可分解为
可增量合并的 running 统计量 `(min, first-idx)`；与 online softmax 的 `(max,sum)` 对同构，logsumexp tiled
已验证同族结构）。收益口径：tiled 场景 GM 扫描 **2 遍 → 1 遍**（两遍式 = pass1 找全局极值 + pass2 重读找首现）；
中间缓冲 `O(N) → O(tile)`；UB 驻留可行性：tile 级 `(bm,tile_n)` 工作集 ≤ 64 KB（§5.2 表）。驻留场景（路径 S）
数据已在片上，「在线」无额外收益（片上两 pass 零 GM 代价）——**仅在 tiled 路径采纳**。

**R3 算法复杂度对比**（四口径；per-row 计，M 行并行摊派；本算子为带宽受限：roofline `flops = M·N_padded`
vs `bytes = M·N_padded·elem + M·8`，访存主导）：

| 算法候选 | FLOPs（比较次数，口径：per row） | 访存量 Bytes（GM；读 x + 写 out + 中间往返） | GM 扫描遍数 | 中间缓冲峰值（per row） | 可并行度 / 跨核代价 |
|---------|----------------------------------|---------------------------------------------|-------------|------------------------|---------------------|
| 基线（GPU 两步） | `N_padded`（reduce）+ `N/2` 平均（标量扫描） | `N_padded·elem + 8` | 1 | `x_f32` fragment 4 B/elem（驻留） | M/bm 行块独立，无跨核 |
| C1 驻留两步向量化 | `N_padded`（reduce_min）+ `N_padded`（vbrc 广播）+ `N_padded`（vcmp/vselect 链）+ `N_padded`（候选 reduce_min）≈ **4–5·N_padded 向量 op**（含 `vbrc` 广播与候选 `reduce_min`；下述 C2/C3 同口径） | `N_padded·elem + 8` | 1 | `x_frag` `B`/2 + idx/cand/sent fp32 ≈ **8–16 B/elem**（fp16 8/10、bf16 10/14、fp32 12/16；bm≥2 / bm=1 含 `ext_brc`，与 §4.5 同口径） | 同上 |
| C2 在线单遍（tiled） | 每 tile：`tn`（chunk min）+ `tn`（候选链）+ `tn`（min）+ `O(1)` 更新 ≈ **3N_padded + 3·n_tiles 微操作** | `N_padded·elem + 8`（**1 遍**）+ workspace 0 | **1** | `O(tile)`：`(bm,tn)×(2..4 B)` ≈ 10–16 B/elem·tn | 行独立；lm-head 类 `M<40` 时核利用率低（§1.6.3-#4 / §1.6.4 C6 承接） |
| C3 int64 打包 | `N_padded`（键构造 ~5 pass）+ `N_padded`（reduce）+ 解码 ≈ **7·N_padded** | `N_padded·elem + 8` | 1 | 键 int64 8 B + 输入 `B`/2 ≈ 10–12 B/elem | 同 C1 |
| （对照）tiled 两遍式 | `2·N_padded` | **`2·N_padded·elem + 8`**（重读） | **2** | `O(tile)` | 同 C2 |

**R4 硬件亲和性评估**（检查清单逐项；实测常数引用 `pattern-library/constants.md` 条目 + 版本戳；**本任务设备
为 910B3，凡绑定 910B2C 的常数标注「未实测假设：沿用同代 910B 曲线量级，Stage 4 校准」**）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（串行扫描形态） | ❌ 标量 early-exit 扫描与 Vector 逐元素/规约模型不符（N/2 次标量 op × ~0.5 µs 发射 ≫ 向量 pass，`CONST-vector-launch-overhead`） | — | — | — | — | ❌ 形态需重设计（→C1） |
| C1 驻留两步向量化 | ✅ 纯 Vector（`reduce_min` + `vcmp/vselect` 链 + `reduce_min` 全向量原语；`T.reduce.md` dtype 表 ✓ fp16/fp32/i64，**bf16 ✗→预 cast**） | ✅ 驻留 ≤64 KB/block（实查 UB 192 KB ÷ auto-multi-buffer 膨胀 1.6–2.0×〔`CONST-capacity-910B2C`〕；fragment 计算形态规避多消费者竞态） | ✅ 32 B 尾轴对齐即可（logsumexp N=300 先例）；N_padded 恒为 256 倍数 | ✅ M/N_padded/tile 边界全工厂期常量 | ✅ auto-multi-buffer 流水（Developer 自动施加）；无跨引擎往返 | ✅ **驻留路径选定** |
| C2 在线单遍 | ✅ 纯 Vector + `(bm,1)` 微操作链（`vcmp`/`vselect`） | ✅ tile 工作集 ≤64 KB（fp16 tn≤6400、bf16/fp32 tn≤4096–4608） | ✅ `tile_n` 256 倍数（`compute_tile_n` 除数优先）；**静态尾 tile** 特化（kernel 侧掩码有编译器 segfault 反例，故用静态特化） | ✅ `num_full/tail/serial` 边界全静态；**`bm=1` 强制**（CG-2026-0014：`bm>1`+`num_tiles>1` → 编译器 SIGSEGV） | ✅ tile 循环 auto-multi-buffer | ✅ **tiled 路径选定** |
| C3 int64 打包 | ✅ Vector | ⚠️ 键 8 B/elem 预算减半 | — | — | ⚠️ 键构造链多 pass | ❌ pass 数劣（R3 量化） |
| C6 N-split merge | ✅ | ⚠️ workspace GM 往返（`M·nchunk·(elem+4)B ×2`；fp16 lm-head 实例 ≈3.8 KB ≪ x 读流量 800 KB） | — | ✅ | ⚠️ 两 kernel 启动 + 合并正确性面（同族实例曾有首版合并 mismatch，终版修复并 `PL-1.20` 固化） | ⏸ Stage 4（**完整备选结构 + 判定阈值 + 回写路径见 §1.6.4**） |

**调研结论**：选定 **C1（驻留两步向量化）+ C2（在线单遍）按 N 驻留阈值双路径**的算法族。

- **关键依据**：R3 表——两口径（GM 遍数、向量 pass 数）均不劣于基线且消除标量扫描；R4——全 Vector 原语
  可达、驻留/tile 预算经同族实测定标、静态边界满足一维 Kernel 约束。
- **与基线的结构差异一句话**：「`-max(-x)` 极值归约 + 串行扫描」→「`reduce_min` 极值归约 + 掩码/选择索引
  `reduce_min`（驻留）/ 在线 running-(min,idx)（流式）」。
- **调研范围（「无更优替代」的查证范围）**：`algorithm-candidates.md` 的 `ALG-reduction` / `ALG-layernorm`
  R1–R4 行与 row-reduction 迁移基准形态；`kb_search` 预注入 5 条 + 本设计的 queue 复核；`examples/`
  （logsumexp single+tiled、argmax 姐妹 kernel、ssd_chunk_scan）；pattern-library（constants 全条目、
  traps-compiler、layout PL-1.20/1.21/1.22、elementwise、cases）；恒等变形空间（C3/C4/C5 全枚举）；
  结构判据（在线变体存在性 / 复合键打包可行性 / 排序下界）。**R1 已枚举 6 个候选（含基线）**，
  无「无更优替代」空白断言。
- **源算法优化手段承接（§0.4 逐项）**：#2 两步结构保留；#3 fp32 cast 由 E4 论证舍弃（意图 = 数值稳健，
  对精确选择运算冗余）；#4 的 `-max(-x)` 变换由 E2 替换为直接 `reduce_min`（意图 = 单实现双算子，保留——
  同一 kernel 双 kind 分支）；#5 单位元 pad 由 R3 保留（意图 = 非整除正确性），256 对齐约束舍弃；
  #1 SMEM tiling 由 UB+fragment 承接；#6/#8 由 R4 重构；#7 保留。

**设计期 roofline（D-2；协议 `_shared/standards/hardware-cost-model.md` §2；常数带版本戳）**：

| workload（manifest） | 路径/配置 | 流量项 | 发射项（向量 op 数 × 0.5 µs〔`CONST-vector-launch-overhead`〕÷ 核数） | 估算下界（µs） |
|---|---|---|---|---|
| smoke (32,256) fp16/bf16/fp32 | S, bm=8, 4 block | 16.3 KB ÷ (33.9 GB/s × 4 核) ≈ 0.12 µs | 4×10 op ÷ 4 核 × 0.5 = 5.0 µs | **≈5** |
| non-aligned (256,300→**512**) fp16/bf16 | S, bm=8, 32 block | 258.5 KB ÷ (33.9 × 32) ≈ 0.24 µs（含 pad 108.5 KB） | 32×10 ÷ 32 × 0.5 = 5.0 µs | **≈5** |
| hidden-state (2048,4096) fp16 | S, bm=2, 1024 block→persistent 40 | 16.8 MB ÷ (30.2 × 40) = 13.9 µs | 1024×10 ÷ 40 × 0.5 = 128 µs（串行上界）→ 流水隐藏后 ≈**56 µs**（校准锚点：同形态 48 核实测 47 µs，按 40/48 折算） | **≈56–128** |
| hidden-state (2048,4096) bf16 | S, bm=1, 2048 block→persistent 40 | 16.8 MB ÷ (30.2 × 40) = 13.9 µs | 2048×11 ÷ 40 × 0.5 = 282 µs（串行上界）→ 流水隐藏后 ≈**90 µs**（同形态 48 核实测 55 µs 校准） | **≈90–282** |
| lm-head (4,102400) fp16 | T, bm=1, tn=5120, 20 tile, 4 block | 800 KB ÷ (30.2 × 4) = 6.6 µs | (7 + 19×13 + 3) op × 0.5 ≈ 129 µs/核（4 核并行 4 行）→ 流水后 ≈**30–60 µs**（同形态实测 32.5 µs 校准） | **≈30–60** |
| lm-head (4,102400) bf16 | T, bm=1, tn=4096, 25 tile, 4 block | 800 KB ÷ (30.2 × 4) = 6.6 µs | (7 + 11 + 24×13) × 0.5 ≈ 165 µs/核 → 流水后 ≈**40–75 µs** | **≈40–75** |

> **估算下界（汇总）**：`估算下界 ≈ max(流量项, 发射项, 容量项) = 发射项主导`（容量项不适用——纯 Vector 单引擎、
> 无跨引擎 S/P 往返，`CONST-store-fixpipe-gm-only` 不适用）。发射项按 `CONST-vector-launch-overhead`
> （~0.5 µs/op 串行模型）为**上界**；三处用同形态实测值（47/55/32.5 µs）校准并标注偏差方向
> （auto-multi-buffer 流水使实际低于串行模型 ~2–3×）。
> **常数适用性声明**：`CONST-mte2-degradation` 与 `CONST-vector-launch-overhead` 均绑定 **910B2C**
> （CANN 8.5.0），本任务设备为 **910B3**（CANN 9.0.0）——本表按「未实测假设：沿用同代 910B 曲线量级」
> 引用，Stage 4 须用本机实测校准（`hardware-cost-model.md` §3 回填对账）。核数(`CONST-aicore-910B2C`;
> 跨型号数据点已书 910B3=20)与 UB 容量（`CONST-capacity-910B2C`）**已在本机实查复核**
> （20 / 196608 B，§5.5），无假设成分。

#### 1.6.1 数学等价优化（公式级）

> 分析对象：§1.6.0 选定算法（C1+C2）的公式。逐项四要素；**等价性机器验证（D-1）**：`verify_equiv.py`
> （torch CPU，fp64 参照 + 输入 dtype 舍入仿真 + 角点），结果表内嵌于下。**v1 补强（建议 1）**：E2 基线改为
> 真正的 `-max(-x)` 变换域路径（fmax NaN-忽略语义化，无死赋值），E4 改为原生 dtype 流水 vs fp32 上抛流水的
> 真实跨域对照；另加「判别力对照」组（3 个故意错误候选必须被报出差异），消除退化对照。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| E1 | 首现索引向量化（承接 §0.6-R1） | 串行扫描 `for j: if x[j]==m: break`（N/2 平均标量比较） | `first = reduce_min_j( ite(x_j = m, j, BIG) )`，`BIG=2^30` | m 必被 ≥1 个 j 命中（有限/±inf 输入）；非命中候选 `BIG=2^30`（fp32 精确 2 的幂）> 任何 `j < N_padded ≤ 2^24`（fp32 精确整数域）；`reduce_min` 恰取最小命中 j = 首现。实现形态：显式 `vbrc→vcmp→vselect` 链（`bm≥2` 强制，CG-2026-0015）。NaN 行不命中（输出哨兵）——NaN 为文档化边界非等价域（§9.2-R1） | 标量 N/2 次 op（每次 ~0.5 µs 发射不可向量化）→ 2 个向量 pass（链式候选 1 + reduce_min 1）；GPU `loop_break` 依赖消除 |
| E2 | argmin 直译（替换源 `-max(-x)` 变换） | `argmin(x) = argmax(-x)`（取负 pass + `reduce_max` + 还原 pass） | `argmin(x) = first-match(min(x))`：直接 `T.reduce_min` + 同 E1 候选机制 | 取负严格单调且保等：`x_i < x_j ⟺ −x_i > −x_j`，`x_i = x_j ⟺ −x_i = −x_j` ⟹ `min` 的首现 j ≡ `−x` 的 `max` 的首现 j（索引恒等，不经值域算术） | 省 2 个全宽 pass（取负 + 还原）；NPU `T.reduce_min` 存在（`testing/npuir/reduction_ops/test_reduce.py` 先例） |
| E3 | 在线单遍递推（承接 §1.6.0-C2） | 两遍式：pass1 全局极值 → pass2 重读找首现（GM 2 遍） | running `(r_m, r_f)`：tile-0 直写初始化；后续 tile **严格小于**才更新 `(c_m, t·tn+c_f)` | 归纳：处理完 tiles `0..t` 后 `(r_m, r_f)` = (已见列的 min, 其首现)。`c_m<r_m` → 新极值来自本 tile、首现 = 本 tile 内首现（更早列值 ≥ r_m > c_m 不可能命中）；`c_m=r_m` → 相等时保留更早索引（升序 tile 下本 tile 索引必大于已见索引）；`c_m>r_m` → 保持。tile-0 直写消除哨兵初始化（否则全 `+inf` 行 `c_m=+inf=r_m` 不触发严格小于、首现丢失——机器验证含全 `+inf` 用例）。比较-only：无值域算术 → 无舍入路径 | tiled 路径 GM 2 遍→**1 遍**（lm-head：−800 KB 读流量 ≈ −6.6 µs@4 核）；中间缓冲 `O(N)→O(tile)` |
| E4 | fp16 免 fp32 上抛（舍弃源全量 cast） | `m = reduce_min(cast(x, fp32))`，匹配在 fp32 域 | `m = reduce_min(x)`（fp16 原域），匹配 `x[j] == m`（fp16 域，同 dtype 比较） | min 为**精确选择**运算（无舍入）：fp16 域 min 的结果 = fp32 域 min 经 fp16↔fp32 单射往返的结果（fp16→fp32 精确单射保序保等）；等值比较在两域判定一致 ⟹ 首现索引恒等。bf16 不适用此优化（reduce 不支持 bf16，`T.reduce.md` §2.2.1——bf16 保留 cast fp32，非精度需求而是 **API 需求**）；fp32 输入无需 cast | fp16 路径省 1 个全宽 pass + 2 B/elem 缓冲（驻留预算等效放宽 8→6 B/elem）；**注意实测陷阱**：reduce 混合 dtype（fp16 src→fp32 dst）静默产出错误值——本优化的「同域比较」正是其规避形态 |
| E5 | `+inf` padding 的索引中性（主选契约的正确性基础） | 源：在 N_padded 张量上只扫描前 N 列（扫描上界钉在 N） | 在 N_padded 全域计算候选：`first = min_{j<N_padded} ite(x[j]=m, j, BIG)`（pad 列 = `+inf`） | 实列索引恒 `< N ≤` pad 列索引；pad 列取到 `+inf` 只在「全行皆为 `+inf`」时成为 min，而此时实列（含 j=0）同样命中且索引更小 ⟹ `reduce_min` 取到的**必然仍是实列首现索引**。故「在 N_padded 全域计算」与「只在实列范围计算」索引恒等 ⟹ host pad 不改变任何输出（对全 `+inf` 行、tie 行、`±inf` 混合行逐一机器验证） | 免去 kernel 侧尾 tile 掩码 / 宽度特化代码路径（控制流最简）；**代价**：`N=300` 类负载多读 108.5 KB（§1.6.0 D-2，非带宽主导） |

**机器验证结果表**（`verify_equiv.py`，2026-10-14 执行，exit 0；输入 dtype 舍入仿真 + fp64 参照；shape 覆盖
`(1,1)/(1,2)/(3,8)/(4,256)/(8,300)/(16,512)/(7,1024)/(4,4096)` × {fp16,bf16,fp32} + 角点行〔全 0 / 全 `+inf`
/ 全 `-inf` / `±inf` 混合 / 混合符号零（两种方向）/ 重复极值 tie / 亚正规最小〕；E3 另覆盖 `(4,700)`/`(3,700)`
的 `tile_n ∈ {128,256,300,512,700}`、跨 tile tie、全 `+inf` 行、尾 tile 唯一命中）：

| 验证项 | 用例行数 | 最大偏差 | 违反率 | 结论 |
|--------|---------|---------|--------|------|
| E1 掩码候选 vs 串行扫描基线 | 264 | 0（索引精确相等） | 0/264 | **EQUIV_PASS** |
| E2 直接 `reduce_min`（直接域）vs **真 `-max(-x)` 变换域**基线（fmax NaN-忽略，无死赋值） | 264 | 0 | 0/264 | **EQUIV_PASS** |
| E3 在线递推 vs 直接 argmin（`tile_n` 扫描 + 跨 tile tie + 尾 tile + 全 `+inf`） | 90 | 0 | 0/90 | **EQUIV_PASS** |
| E4 fp16/bf16 **原生 dtype 流水** vs fp32 上抛流水（真实跨域对照，非精确单射恒真构造） | 264 | 0 | 0/264 | **EQUIV_PASS** |
| E5 `+inf` pad 全域（`N_padded`）vs 实列宽 | 264 | 0 | 0/264 | **EQUIV_PASS** |
| 判别力对照（3 个故意错误候选：末次出现 / 过小哨兵 / 有损粗化域） | 147 | > 0（期望违反） | 72 / 57 / 2 处 | **CONTROL_OK**（证明脚本可报出差异；若为 0 违反则判 `CONTROL_BROKEN` 并失败——非门禁等价项） |
| INFO：NaN 行为（等价域外，行为记录） | 4 | 全部含 NaN 行：npu-candidate=哨兵 `2^30` / 源 fmax 路径=非 NaN 极值或回退 0 / torch=首个 NaN 索引——**三方各异，符合声明的边界行为**；该行同时证明 E2 变换域基线与 NaN 传播候选**结构上可分** | — | **INFO(diverge)**，非门禁项 |
| **合计（门禁等价组）** | **1146** | — | **0 项失败** | **ALL EQUIV_PASS**（另 +147 对照行，见上行） |

**优化结论**：采纳 E1–E5 共 5 项；优化后公式 = §1.3 第二、三式（供 §3.1 拆解）：驻留式
`first = reduce_min(ite(x = reduce_min(x), j, 2^30))`（fp16/fp32 原域，bf16 预 cast fp32；argmin 用
`reduce_min` 直译；实现形态为显式向量链）；流式在线递推（tile-0 直写 + 严格小于更新）。
（v1：E2/E4 两组已按上文判别力补强重写并重跑，结果表逐行更新——5 组门禁等价全通过 + 3 组判别力对照全 OK。）

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点对象：§1.4 NPU 算法的**全部**循环与标量计算点。替代 API 均有 `docs/Tilelang.language/` 或
> `examples/` / `testing/` 佐证（列内标注）。

| # | 计算点 | 原实现形态（源码/朴素形态） | 向量替代方案 | 替代结论（✅ 已替代 / ❌ 保留） | 不可替代理由 |
|---|--------|---------------------------|-------------|---------|--------------|
| 1 | 首现索引扫描（源步骤 4） | `T.Serial(N)` 标量比较 + `T.loop_break` | `T.vbrc` + `T.vcmp("eq")` + `T.vselect` + `T.reduce_min(dim=1)`（`docs/Tilelang.language/条件操作/T.vselect.md`、`docs/Tilelang.language/比较操作/T.vcmp.md`、`docs/Tilelang.language/规约操作/T.reduce.md`） | ✅ | — |
| 2 | 极值归约（源步骤 3b） | （源已是 reduce 原语；朴素替代为标量扫描） | `T.reduce_min(dim=1)`（`docs/Tilelang.language/规约操作/T.reduce_max.md`；**同 dtype 约束**） | ✅ | — |
| 3 | argmin 取负 / 还原（源步骤 3b 两处全宽循环） | `T.Parallel` 逐元素取负 ×2 | 直接 `T.reduce_min`（E2；`testing/npuir/reduction_ops/test_reduce.py` 先例） | ✅（消除） | — |
| 4 | fp32 cast（源步骤 2） | `T.Parallel` 全宽 cast 循环 | fp16：**消除**（E4，原域 reduce + 比较）；bf16：`T.vcast(rint)`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`；**禁 f32→f32 `vcast`**）；fp32：无 cast（同 dtype `T.copy`） | ✅ | — |
| 5 | 索引序列生成 | 朴素：`T.Parallel` 赋值 `idx[i,j]=j` | 核内 1-D `T.arange(idx_src (N_padded,), [1], 0)` → `T.reshape` 到 `(1,N_padded)` → `T.vbrc` 到 `(bm,N_padded)`；`bm=1` 时 `T.reshape` 直接进 `idx_j`（同形 `vbrc` 触发空广播维 verify fail，`TRAP-vbrc-same-shape-empty-broadcast` / CG-2026-0017） | ✅ | — |
| 6 | 极值广播到 `(bm,N_padded)` | 朴素：`(bm,1)` 循环填充 | `T.vbrc(row_min, ext_brc)`（`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2 张量广播 `(bm,1)→(bm,N)`） | ✅ | — |
| 7 | 在线更新（tiled） | 朴素：逐元素条件更新 | `T.vcmp(..., "lt")`（`docs/Tilelang.language/比较操作/T.vcmp.md`）+ `T.vselect` **无别名** ×2（`T.vselect(cond, A, B, scratch)` + `T.copy` 回写；(bm,1) 微缓冲） | ✅ | — |
| 8 | tile 基址全局化（tiled） | — | `T.Parallel(bm)` 标量算术 `chunk_global = t·tn + chunk_first[i,0]`（`bm=1` 恒 1 次迭代；纯 TIR 算术无 select codegen） | ❌（保留） | tile 级顺序依赖载体（在线更新的跨 tile 携带状态）；`bm=1`（CG-2026-0014 强制）下为单元素语句，非逐元素热点；融合进 value 分支含 `serial` 变量的条件有 logsumexp 同类 codegen segfault 反例（`TRAP-vbrc-scalar-shared`）——列为 Stage 4 简化候选 |
| 9 | int64 出栈 epilogue | — | `T.reshape(first, first_flat (bm,))` + `T.vcast(first_flat, out_ub, round_mode="rint")`（向量化）+ `T.copy` UB→GM | ✅（向量化） | 形状收缩（`(bm,1)→(bm,)`）由 `T.reshape`（元数据）桥接后 `vcast` 逐元素向量执行；`rint` 在整数域与源 `fptosi` 截断值恒等（first 恒为 `j<N_padded` 或 `BIG`，均 fp32 精确整数） |
| 10 | 尾块行数 `real_m` / 任务 guard | — | `T.min` 标量 + `if start < M` 动态 guard（`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 示例先例） | ❌（保留） | 依赖运行时 pid 的 block 级边界处理（非逐元素热点）；VP-2026-0044 三件套标准形态（serial 上界静态 + guard 动态） |
| 11 | host 侧 block_m/tile_n/路径选择 | Python 工厂期计算 | — | ❌（保留） | host 元数据计算（不在 kernel 内；每 `(M,N,dtype)` 一次） |

**向量化结论**：逐元素计算已全部向量化（#1–#7、#9，API 均有佐证）；保留 **3 处**标量/循环（#8 / #10 / #11），
理由类别：tile 级顺序依赖（#8）、形状收缩/动态边界（#10）、host 元数据（#11）——均非逐元素热点（发射量 1 op 量级），
逐项见上表（#9 属**已替代**侧，不计入保留集）。源实现中唯一的逐元素标量热点（串行扫描）已被 E1 向量化替代。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（`(M,N_padded)` 行主序进 / `(M,)` int64 出）；**核内布局与 lane 映射是设计变量**。
> 本算子 2D `(M,N_padded)`——按规约类必选候选枚举。

**轴质量评分**（规约类：水平归约 vs 垂直扫描为正交决策，须分别枚举）：

| 评分项 | 水平归约（lane→N 轴，reduce `dim=1`） | 垂直扫描（lane→M 行，串行步进 N） | 说明 |
|--------|------|------|------|
| 整除性 | N_padded 恒为 256 倍数（`reduce`/`vcmp`/`vselect` 原语处理任意宽度）；尾 lane 浪费由硬件归约吸收 | M 任意同理 | 无差异 |
| 累加链形态 | 归约维即向量维（`dim=1` 连续 stride=1）——reduce 原生形态 ✓ | 归约维变串行步进维（N 次 `(bm,1)` 微操作 × 0.5 µs 发射 = `N/2·0.5 µs/行`——发射爆炸） | **决定性差异** |
| repack 代价 | 无（I/O 原生行主序即 N 连续） | 无 | — |
| UB 容量影响 | 行块 `(bm,N_padded)` 驻留（§4.5 预算表） | 同 | — |

**候选矩阵**（含 GPU 源轴选择对照——源码 thread/warp 轴是输入非结论）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 采纳 |
|---|---------|---------|------------|--------------|------|
| 1 | **行块 × 全 N_padded 驻留 + 水平归约**（lane→N，reduce `dim=1`；persistent 按 M 行块分核） | N_padded（连续） | 无 | 驻留 ≤64 KB/block；向量 pass ~10/block；`PL-1.21-vector-persistent-split` + `ALG-reduction` row-reduction 基准形态（logsumexp+ada 两任务实证） | ✅ **主选** |
| 2 | 垂直扫描（lane→M，串行步进 N 维） | M | 无 | 归约维串行化 → 每行 N 次微操作发射（N_padded=4096 → ms 量级/行）——发射项灾难 | ❌（发射模型量化） |
| 3 | 核内转置（N-major）后行归约 | M（转置后连续） | `T.transpose` 二轴交换链 ×2（UB 级；`docs/Tilelang.language/创建操作/T.transpose.md`；实测代价引 `PL-1.1-transpose-chain` ~5.4 µs） | **纯开销零收益**：主选 #1 的向量化轴 N_padded 即 I/O 原生连续轴、无 repack 需求，任何核内转置不减少 GM 流量与向量 pass 数；另需 +2 个 `(bm,N_padded)` UB 缓冲（按 §4.5 口径 +4~8 B/elem，直接吃掉 hidden-state fp16 bm=2 的预算裕度） | ❌ |
| 4 | N-split 跨核（C6：列分片 partial+merge 两段 kernel） | N（跨核切分） | workspace GM 往返 `(M,nchunk)×2`（fp16 lm-head 实例 ≈3.8 KB） | 小 M 大 N（lm-head）核利用 `4/40 → 40/40`；同形态实测锚点 2.2–2.6×（`PL-1.20-nsplit-perrow-partial`）；代价：+1 kernel 启动、workspace 往返、**合并 tie-break 正确性面** | ⏸ **Stage 4 A/B**（结构 + 阈值 + 回写见 §1.6.4） |
| 5 | 候选构造形态：显式向量链 vs 2D `T.Parallel` 融合 `T.if_then_else`（同轴 #1 内的两种 pass 组织） | N_padded | 无 | 融合式：`bm=1` 向量化正常（(1,5120) 25.9 µs）但 **`bm≥2` 整体标量化**（scalar ratio 0.84–0.95，(2,4096) 222 µs）；链式：pass 数 +2、`bm=1/bm≥2` 统一向量化 | ✅ **链式主选**（CG-2026-0015 / `TRAP-parallel2d-bm-ge2-scalarize`；融合式仅作 `bm=1` 的等价回退形态） |

**布局决策结论**：选定方案 **#1 + #5**：核内布局 = 行块 `(bm,N_padded)` 行主序驻留（UB staging + fragment
计算），向量化轴 = **N_padded 轴**（reduce `dim=1`，候选构造走显式 `vbrc→vcmp→vselect` 链），repack 无，
分核 = persistent 按 M 行块（§5.5）。该结论同步落入 §3.3 伪代码与 §6 循环结构。弃选理由：#2 发射爆炸（量化）；
#3 纯开销零收益（主选轴即 I/O 原生连续轴、无 repack 需求，`PL-1.1` 实测代价小亦无收益可抵；`T.transpose.md`
的 dtype 表在本算子无需核对——该路径已被量化收益否决而非能力否决）；#4 为 Stage 4 实验裁决项，三件套齐备（§1.6.4）。

#### 1.6.4 实验裁决备选方案：C6 N-split partial+merge（Stage 4 A/B 三件套）

> §1.6.3-#4 的备选完整结构 + 裁决计划（skill Phase 2 实验裁决模式）。**主选** = §1.4 单 kernel 双路径
> （当前证据下最优：正确性面最小、无 workspace 往返）；**触发形态** = 小 M 大 N 负载（lm-head 类）
> 核利用率 `ceil(M/bm)/40` 偏低（`M=4` → `4/40`）。§3.3/§4/§6 只承载主选方案，本节结构独立成节、不与主选混排。

**① 分派条件（工厂期静态谓词，per-shape 分派可行性）**：`M ≤ 16 且 ceil(M/bm) < 40 且 N_padded ≥ 32768`
——lm-head `(4,102400)` 命中；hidden-state / non-aligned / smoke 不命中（走主选）。工厂 `lru_cache` 本就按
`(M, N, op_kind, dtype)` 分派，C6 与主选可按 shape 共存（无全局二选一约束）。

**② 备选结构（两段 kernel，均为 Developer 单 kernel 形态；fp16 lm-head 实例参数：bm=M=4、tn=1280、nchunk=80）**：

```python
# ── partial kernel（argreduce_partial）：逻辑块 = nchunk 列分片，每块处理 (M, tn) ──
# tn/nchunk：§5.2 同款规则在 partial 预算下取值——fp16：slab = ub_slab_units(2, 2, 1) = 4
#   （x_ub/x_frag fp16 + cand fp32），4×4×tn×2B ≤ 0.9×64KB → tn ≤ 1843 → N 的 256 倍数
#   整除因子取最大 → tn=1280、nchunk=80（102400 = 80×1280，零尾）；手工预算 40960B = 40KB
with T.Kernel(min(nchunk, 40), is_npu=True) as (cid, _):        # 分核：逻辑 80 > 40 → persistent
    x_ub   = T.alloc_shared((M, tn), dtype)                      # GM staging（§4.5 纪律②）
    x_frag = T.alloc_fragment((M, tn), dtype)                    # 计算缓冲（bm=M≥2，链式无需 ext_brc）
    m      = T.alloc_fragment((M, 1), dtype)
    cand   = T.alloc_fragment((M, tn), "float32")
    first  = T.alloc_fragment((M, 1), "float32")
    gidx   = T.alloc_fragment((M,), "float32")
    m_ub, gidx_ub = T.alloc_shared((M,), dtype), T.alloc_shared((M,), "float32")
    for s in T.serial(ceildiv(nchunk, num_kernels)):             # 静态上界（80/40 → 2）+ guard
        c = cid + s * num_kernels
        if c < nchunk:
            T.copy(x[0:M, c*tn:(c+1)*tn], x_ub[0:M, 0:tn])       # 列块 2D slice
            T.copy(x_ub, x_frag)
            T.reduce_min(x_frag, m, dim=1)                        # 驻留路径 S2–S5 同款链（E4：原域）
            for i, j in T.Parallel(M, tn):                        # 候选（局部 j）——M=4 且 tn 窄，
                cand[i, j] = T.if_then_else(x_frag[i, j] == m[i, 0],   # 或退到链式（见 §1.6.3-#5 注）
                                            T.cast(j, "float32"), T.float32(BIG))
            T.reduce_min(cand, first, dim=1)
            for i in T.Parallel(M):                               # 全局化（§1.6.2-#8 同款标量算术）
                gidx[i] = c * tn + first[i, 0]
                m_ub[i] = m[i, 0]                                 # (M,1)→(M,) 形状收缩（#9 同款）
            T.copy(m_ub[0:M], ws_min[0:M, c])                     # workspace 列切片写出
            T.copy(gidx_ub[0:M], ws_idx[0:M, c])                  # （UB staging + slice copy 纪律）

# ── merge kernel（argreduce_merge）：单块归并，E1 同构 tie-break ──
with T.Kernel(1, is_npu=True) as (cid, _):                        # 分核：逻辑 1 ≤ 40，单块
    ws_min_f = T.alloc_fragment((M, nchunk), cdtype)              # workspace → fragment
    ws_idx_f = T.alloc_fragment((M, nchunk), "float32")
    g      = T.alloc_fragment((M, 1), cdtype)
    candm  = T.alloc_fragment((M, nchunk), "float32")
    fout   = T.alloc_fragment((M, 1), "float32")
    T.copy(ws_min[0:M, 0:nchunk], ws_min_f)                       # GM workspace → fragment
    T.copy(ws_idx[0:M, 0:nchunk], ws_idx_f)
    T.reduce_min(ws_min_f, g, dim=1)                              # 全局极值（行向，已验证形态）
    for i, c in T.Parallel(M, nchunk):                            # tie-break：与 E1 同构
        candm[i, c] = T.if_then_else(ws_min_f[i, c] == g[i, 0],
                                     ws_idx_f[i, c], T.float32(BIG))
    T.reduce_min(candm, fout, dim=1)                              # 首现 = 命中 chunk 的最小全局索引
    T.reshape(fout, fout_flat)                                    # 向量化 int64 出栈（主选同款）
    T.vcast(fout_flat, out_ub, round_mode="rint")
    T.copy(out_ub[0:M], out[0:M])
```

- **workspace（GM，wrapper 层分配）**：`ws_min (M, nchunk)` 计算域 dtype + `ws_idx (M, nchunk)` fp32——
  fp16 lm-head 实例 `4×80×(2+4)B ≈ 1.9 KB`，往返 ≈3.8 KB ≪ x 读流量 800 KB。**tie-break 正确性**：
  `ws_idx[i,c] = c·tn + 首现局部索引`（已全局化），chunk 索引随列位单调递增 → 命中集上取 min 即全局首现
  （与 E1 机器验证同构）。
- **UB 预算（partial，fp16 tn=1280）**：slab=4 × bm=4 × tn=1280 × 2 B = 40960 B = 40 KB ≤ 0.9×64 KB ✓；
  merge 侧 `(M,nchunk)=(4,80)` 微缓冲 ≪ 预算。bf16 partial：`x_ub` bf16 + `x_work`/`cand` fp32
  → slab = `ub_slab_units(2, 1, 2)` = 5 → `5×4×tn×2B ≤ 0.9×64KB` → tn ≤ 1474 → 整除因子取 1280、预算 50 KB ✓；
  fp32 partial：slab = `ub_slab_units(4, 3, 0)` = 3 → tn=1024/nchunk=100。
- **分核三要素（partial）**：逻辑核数 = `nchunk`（80）；物理核数 = **40**（§5.5 实查同源）；规模判定 =
  极大规模（80 > 40）→ persistent 40 + 核内 `T.serial`（静态上界 2 + guard）。merge：逻辑 1 ≤ 物理 → 单块直发。
- **同步**：两 kernel 内部 Developer 自动同步（§7 同款）；kernel 间依赖由 wrapper 顺序发射保证（无核间 sync）。

**③ 实验裁决计划**：代表 shape = lm-head fp16 `(4,102400)` + lm-head bf16 `(4,102400)`；指标 = bench harness
Task Duration（launches≥30 均值，partial+merge 两段之和）且 golden `torch.equal` 全 PASS 为前提；
**判定阈值：C6 端到端 < 0.8× 主选单 kernel 同 shape 延迟 → 该 shape 类分派翻转**（同族同形态实测锚点
≈0.46–0.54×，若本轮主选实测维持 ≥30 µs 则大概率翻转）；**回写路径**：实测数据与结论回写本节裁决记录 +
`RETROSPECTIVE.md`，翻转时修订 §1.4/§3.3 增补 C6 分派分支（wrapper 双发射 + workspace 分配）并过 Stage 2 复审；
Stage 3 实现期如需提前引入，走 `[DESIGN_LIMIT]` 受控修订路由。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（归约 + 逐元素比较/选择 + 微操作），无 matmul、无 Cube | 不需要 L1/L0；`alloc_shared`→UB + `alloc_fragment` 即可 |
| 归约 | `reduce_min`（dim=1）+ `reduce_min`（候选）——Developer 模式可用（`testing/npuir/reduction_ops/` 全部 Developer 先例） | ✓ |
| 内存层级 | 仅 GM ↔ UB ↔ fragment，无跨引擎 | Developer 自动管理 |
| 同步 | 单 kernel 内顺序执行 + persistent 任务循环，无核间协作（C6 才是两 kernel 组合，仍各自内部自动同步） | Developer 自动同步即可 |
| 家族先例 | logsumexp single/tiled、argmax 姊妹 kernel 均为 Developer（`PL-1.21-vector-persistent-split` row-reduction 基准形态） | 沿用 |
| 调用方约束 | harness 迁移 prompt 指定 developer 模式 | ✓ |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（GM staging，编译器映射 UB）+ `T.alloc_fragment`（全部计算缓冲——竞态规避形态，§4.5 纪律②） |
| 计算方式 | v 前缀 API（`vbrc`/`vcmp`/`vselect`/`vcast`）+ `reduce_min` + 一维 persistent Kernel |
| 同步 | Developer 自动同步（无手动 `sync_block_set/wait`） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True)` 一维 persistent |

---

## 3. API 映射设计

### 3.1 公式拆解（输入 = §1.6.1 优化后公式）

**驻留路径 S**（argmin）：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| S1 | `x_ub ← x[block 行块, 0:N_padded]`；`x_frag ← x_ub` | GM→UB→fragment（src/dst 双 slice 到 `real_m`） |
| S2 | `m = reduce_min(x_frag, dim=1)` | 行极值 `(bm,1)`；fp16/fp32 原域、bf16 先 `vcast(x_ub→x_work, rint)`（fp32 域） |
| S3 | `ext = vbrc(m → (bm,N_padded))` | 极值广播同形化（统一 `bm=1/bm≥2`，规避 `(·,1)` 不变操作数 i1 误型） |
| S4 | `cmp_eq = vcmp(x_frag, ext, "eq")`；`cand = vselect(cmp_eq, idx_j, sent_v)`，`sent_v ≡ BIG=2^30` | 链式候选（向量化；`bm≥2` 强制） |
| S5 | `first = reduce_min(cand, dim=1)` | 首现索引 `(bm,1)` fp32 |
| S6 | `reshape(first, first_flat)` + `vcast(first_flat, out_ub, rint)`；`out[block] ← out_ub[0:real_m]` | 向量化 int64 出栈 + GM 写（slice） |

**流式路径 T**（tiled online；tile-0 为 S2/S4/S5 直写 running，全 tile/尾 tile 追加更新链）：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| T1 | tile-0：`running_min = reduce_min(tile0)`；`running_idx = reduce_min(ite(tile0=running_min, j, BIG))` | 直写初始化（E3；局部 j 即全局 j） |
| T2 | 全 tile t：`c_m = reduce_min(tile_t)`；`c_f = reduce_min(ite(tile_t=c_m, j, BIG))`；`c_g = t·tn + c_f`；`if c_m < running_min: (running_min, running_idx) ← (c_m, c_g)` | 严格小于更新（`vcmp "lt"` + `vselect` ×2 **无别名**） |
| T3 | 尾 tile（静态宽度 `tail`）：同 T2，基址 `num_full·tn` | 静态特化路径 |
| T4 | 同 S6 | 出栈 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 | 佐证 |
|------|----------|-------------|------|------|------|
| S1/T1 载入 | GM→UB | `T.copy(x[start:start+real_m, c0:c1], buf[0:real_m, 0:w])` | 双显式 slice | Developer | logsumexp single；VP-2026-0044 三件套 |
| S1' UB→fragment | UB→FRAG | `T.copy(x_ub, x_frag)`（同 dtype） | 全缓冲 | Developer | **禁 `vcast` f32→f32**（`TRAP-vcast-same-dtype-rounds`） |
| S2 bf16 预转 | bf16→fp32 | `T.vcast(x_ub, x_work, round_mode="rint")` | rint（bf16→f32 精确） | Developer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 |
| S2 极值 | 行极小值 | `T.reduce_min(x_frag, m, dim=1)` | dim=1；**dst dtype ≡ src dtype** | Developer | `docs/Tilelang.language/规约操作/T.reduce_max.md`；`testing/npuir/reduction_ops/test_reduce.py` |
| S3 广播 | `(bm,1)→(bm,N_padded)` | `T.vbrc(m, ext_brc)` | 张量广播（尺寸 1 维扩展） | Developer | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2 |
| S4 比较 | `x == m` 元素级 | `T.vcmp(x_frag, ext_brc, cmp_eq, "eq")` | 布尔 dst `(bm,N)` | Developer | `docs/Tilelang.language/比较操作/T.vcmp.md` |
| S4 候选 | `ite(x==m, j, BIG)` | `T.vselect(cmp_eq, idx_j, sent_v, cand)` | 元素级三操作数 | Developer | `docs/Tilelang.language/条件操作/T.vselect.md` |
| S4 索引 | 列索引 `j` | `T.arange(idx_src (N_padded,), [1], 0)` + `T.reshape(idx_src, idx_row (1,N_padded))` + `T.vbrc(idx_row, idx_j (bm,N_padded))`；`bm=1` 时 `T.reshape(idx_src, idx_j)` | 1-D arange → reshape → vbrc | Developer | `docs/Tilelang.language/创建操作/T.arange.md`；`CONST-arange-scalar-materialize`；同形 `vbrc` 禁用（`TRAP-vbrc-same-shape-empty-broadcast` / CG-2026-0017）；region dst 禁用（`TRAP-npuir-vbrc-region-dropped`） |
| S5 首现 | `reduce_min` dim=1 | `T.reduce_min(cand, first, dim=1)` | fp32 | Developer | `docs/Tilelang.language/规约操作/T.reduce.md`（`reduce_mode="min"`） |
| T2 比较/选择 | `lt` 更新 | `T.vcmp(c_m, r_m, cond, "lt")` + `T.vselect(cond, c_m, r_m, new_m)` + `T.vselect(cond, c_g, r_f, new_i)` + `T.copy(new_*, r_*)` | `(bm,1)` **无别名**（scratch + 回写） | Developer | `docs/Tilelang.language/比较操作/T.vcmp.md`、`docs/Tilelang.language/条件操作/T.vselect.md`；`TRAP-vselect-inplace-carried-state`（原位形态禁用） |
| T2' 基址 | `t·tn+local` | `T.Parallel(bm)` + TIR 算术 `T.cast(t*tile_n,"float32") + local[i,0]` | 纯算术（`bm=1` 单元素） | Developer | §1.6.2-#8 |
| S6 出栈 | fp32→int64 | `T.reshape(first, first_flat)` + `T.vcast(first_flat, out_ub, round_mode="rint")`；`T.copy(out_ub[0:real_m], out[start:start+real_m])` | 向量化 cast（`rint` 整数域恒等）+ slice 写 | Developer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1；`docs/Tilelang.language/shape操作/T.reshape.md` |
| Grid | persistent | `T.Kernel(num_kernels, is_npu=True)` + `T.serial(静态上界)` + `if start < M` | 一维 | Developer | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 示例（guard 先例）；`PL-1.21-vector-persistent-split` |

**dtype 路径矩阵**（S2/S4 按输入 dtype 工厂期分派；TVMScript 禁止条件定义变量 → 每路径独立 `prim_func`，工厂选择；
**未用 fragment alloc 不被 DCE**，故禁止「无条件 alloc + 条件使用」——CG-2026-0008 小 buffer 漏计/膨胀实证）：

| 输入 dtype | 极值/比较域 | 预转 pass | 候选域 | B/elem（bm≥2 / bm=1 含 `ext_brc`） |
|-----------|------------|----------|--------|------------------|
| float16 | fp16 原域 | 无（E4） | fp32 | 8 / 10 |
| float32 | fp32 原域 | 无（同 dtype `T.copy`） | fp32 | 12 / 16 |
| bfloat16 | fp32（`vcast` 预转） | `vcast` rint | fp32 | 10 / 14 |

### 3.3 计算伪代码

TIR 原语函数签名中 dtype 直接作第二个位置参数（避免 `dtype=` 关键字触发 false-alarm）。

```python
# 工厂期（Python，全部静态量）：N_padded、block_m 阶梯、tile_n、num_full/tail、num_kernels、serial 上界
def _argreduce_kernel(M, N, op_kind, dtype):          # 接口语义参数不变（harness 契约）
    N_padded = align_up(N, 256)                       # 主选 pad 契约：声明宽度 == Op 传入宽度（§0.2）
    assert N_padded <= 2 ** 24                        # E1/E3 fp32 索引精确域前提（超限 raise）
    # 两段式 bm 解析（与 §5.2 同序求解——基础/扩展阶梯逐档用该档 B/elem，无解回退 bm=1，仍无解转 tiled）
    B_multi = B_elem(dtype, bm_class="multi")         # §4.5 表 bm≥2 列：fp16 8 / fp32 12 / bf16 10
    B_bm1 = B_elem(dtype, bm_class="bm1")             # §4.5 表 bm=1 列（含 ext_brc）：fp16 10 / fp32 16 / bf16 14
    block_m = max([p for p in (1, 2, 4, 8)
                   if p * N_padded * (B_bm1 if p == 1 else B_multi) <= 65536] or [None])  # 基础阶梯
    if block_m is not None and ceildiv(M, block_m) > 80:            # 发射项驱动 → 扩展阶梯
        block_m = max([p for p in (16, 32, 64, 128, 256, 512, 1024, 2048)
                       if p * N_padded * B_multi <= 65536] + [block_m])
    if block_m is None and N_padded * B_bm1 <= 65536:
        block_m = 1                                   # bm=1 驻留（含 ext_brc 预算列）
    if block_m is None:                               # 仍无解 → 路径 T（tiled online）
        block_m = 1                                   # CG-2026-0014 强制
        tile_n = pick_divisor_with_margin(N_padded, dtype, UB_MANUAL_BUDGET, margin=0.9)
    block_m = min(block_m, M)                         # 行数不足不超配（垃圾行污染）
    # ── 路径 S（resident）伪代码（fp16 形态；bf16/fp32 同构换域 + 独立 prim_func 分派）──
    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(bm):                                     # K9: threads 已去除
        @T.prim_func
        def main(x: T.Tensor((M, N_padded), dtype), out: T.Tensor((M,), "int64")):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub    = T.alloc_shared((bm, N_padded), dtype)      # GM staging
                x_frag  = T.alloc_fragment((bm, N_padded), dtype)    # 计算缓冲（竞态规避）
                m       = T.alloc_fragment((bm, 1), dtype)
                ext_brc = T.alloc_fragment((bm, N_padded), dtype)
                cmp_eq  = T.alloc_fragment((bm, N_padded), "bool")
                idx_src = T.alloc_shared((N_padded,), "float32")     # 任务不变操作数（循环外构建）
                idx_row = T.alloc_shared((1, N_padded), "float32")
                idx_j   = T.alloc_shared((bm, N_padded), "float32")
                sent_v  = T.alloc_fragment((bm, N_padded), "float32")
                cand    = T.alloc_fragment((bm, N_padded), "float32")
                first   = T.alloc_fragment((bm, 1), "float32")
                first_flat = T.alloc_fragment((bm,), "float32")
                out_ub  = T.alloc_shared((bm,), "int64")
                T.arange(idx_src, [1], 0)                    # 1-D arange（r6 形态）
                if bm >= 2:
                    T.reshape(idx_src, idx_row); T.vbrc(idx_row, idx_j)
                else:
                    T.reshape(idx_src, idx_j)                # 同形 vbrc 禁用
                T.vbrc(T.float32(BIG), sent_v)
                for s in T.serial(num_local_tasks):          # 静态上界 + grid-stride
                    start = (cid + s * num_kernels) * bm
                    if start < M:                            # 动态 guard（边界静态化）
                        real_m = T.min(bm, M - start)
                        T.copy(x[start:start+real_m, 0:N_padded], x_ub[0:real_m, 0:N_padded])
                        T.copy(x_ub, x_frag)
                        T.reduce_min(x_frag, m, dim=1)       # argmax 分支则 reduce_max（保留双 kind）
                        T.vbrc(m, ext_brc)
                        T.vcmp(x_frag, ext_brc, cmp_eq, "eq")
                        T.vselect(cmp_eq, idx_j, sent_v, cand)
                        T.reduce_min(cand, first, dim=1)
                        T.reshape(first, first_flat)
                        T.vcast(first_flat, out_ub, round_mode="rint")
                        T.copy(out_ub[0:real_m], out[start:start+real_m])
        return main
    return _func
# 路径 T（tiled online）：tile-0 直写 running → T.serial(num_full-1) 全 tile（chunk 链 + lt 无别名更新）
# → 静态尾 tile → 出栈（§3.1 T1–T4；同族形态见 argmax perf_opt）
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|----------|
| `T.reduce_min(dim=1)` | `docs/Tilelang.language/规约操作/T.reduce_max.md`、`docs/Tilelang.language/规约操作/T.reduce.md`；`testing/npuir/reduction_ops/test_reduce.py` | ✅ 文档 + 测试 + 同族实测 |
| `T.vbrc`（张量广播 `(bm,1)→(bm,N)`） | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2 | ✅ 文档 + 同族实测 |
| `T.vcmp("eq"/"lt")` | `docs/Tilelang.language/比较操作/T.vcmp.md` | ✅ 文档 + 同族实测 |
| `T.vselect` | `docs/Tilelang.language/条件操作/T.vselect.md` | ✅ 文档 + 同族实测 |
| `T.vcast(bf16→f32, rint)` / `T.vcast(f32→int64, rint)` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 | ✅ 文档（f32→f32 禁用为同族 + 工具链同级实证） |
| `T.arange` + `T.reshape` + `T.vbrc` 索引链 | `docs/Tilelang.language/创建操作/T.arange.md`、`docs/Tilelang.language/shape操作/T.reshape.md`；`CONST-arange-scalar-materialize` | ✅ 文档 + 同族实测（1-D 形态） |
| `T.copy` 带显式 slice（尾块） | `examples/TileOPs/tileops/kernels/reduction/logsumexp/` 系列 | ✅ 先例 |
| `T.Kernel(一维, is_npu=True)` + `T.serial` + 动态 `if` guard | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 示例；logsumexp 家族；`PL-1.21-vector-persistent-split` | ✅ 先例 + 实测 |
| `T.if_then_else` 元素级（仅 `bm=1` 回退形态） | `testing/npuir/parallel_ops/test_if_then_else_cmp_cond.py`（通过测试） | ✅ 测试（`bm≥2` 标量化 → 仅 `bm=1` 用） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 persistent grid（§5.5） |
| 纯 Vector 算子核数翻倍 | Yes | **40 = 实查 `aivector_core_num`**（§5.5） |
| bf16 不被 reduce 支持 | Yes | `vcast` bf16→fp32 预转（`T.reduce.md` §2.2.1 dtype 表） |
| reduce 混合 dtype 静默错误 | Yes | 同 dtype reduce + 同 dtype 比较（§3.2 约束注） |
| `(·,1)` 不变操作数 Parallel 条件 bug（`bm=1`） | Yes | `vbrc` 同形化（§1.6.2-#6） |
| shared 多消费者竞态（auto-multi-buffer 预算超限） | Yes | fragment 计算形态 + 64 KB/block 手工预算（§4.5） |
| multitile + `bm>1` 编译器 SIGSEGV | Yes（tiled 路径） | tiled 强制 `bm=1`（CG-2026-0014） |
| `vcast` f32→f32 数据破坏 | fp32 路径潜在 | 全程同 dtype `T.copy`（§3.2 注） |
| 2D `T.Parallel` 融合 `if_then_else` `bm≥2` 标量化 | Yes | 候选构造走显式向量链（CG-2026-0015 / `TRAP-parallel2d-bm-ge2-scalarize`） |
| 原位 `T.vselect(cond,A,B,B)` 丢 loop-carried 状态 | Yes（tiled 更新链） | 无别名（scratch + `T.copy` 回写，`TRAP-vselect-inplace-carried-state`） |
| L0C / 分形 / GEMM 非整除 | No | 无 Cube 计算 |

#### 3.5.2 参考实现差异说明（GPU → Ascend 关键差异汇总）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 并行 | `T.Kernel(grid, threads=threads)` 两级 | 一维 `T.Kernel(n, is_npu=True)` | `threads` 去除（K9）；persistent + `serial`（§0.6-R2） |
| SMEM tiling | `alloc_shared` 48 KB 预算 + 256 对齐 | UB 192 KB（实查）+ fragment（auto-multi-buffer 膨胀 1.6–2.0×） | §5.2 预算公式 + §4.5 B/elem 表 |
| 串行首现扫描 | `T.Serial` + `T.loop_break` | 显式向量链 + `reduce_min`（E1） | §0.6-R1 |
| argmin 极值 | `-max(-x)` + `reduce_max` | 直接 `reduce_min`（E2） | §0.6-R1 |
| host pad | Op 层 `F.pad` 256 对齐 | **保留 pad 契约**（`_kernel_handles_padding=False` + `+inf`），kernel 声明 `N_padded` | §0.2 / §0.6-R3 |
| jit 装饰器 | `@tilelang.jit(out_idx=[1])` | `@tilelang.jit(out_idx=[1], target="npuir")` | logsumexp / argmax 同款 |
| 精度域 | 全量 fp32 上抛 | fp16 原域（E4）/ bf16 fp32（API 需求）/ fp32 原域 | §1.6.1-E4 |
| int64 输出 | `T.copy(out_idx, out)` | `T.reshape` + `T.vcast(rint)` 向量化 + slice copy | §3.2 S6 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/_argreduce_kernel_DESIGN.md` | **最高**（共享 kernel 的姊妹迁移设计，910B2C/CANN 8.5.0） | §0 迁移分析结构、§1.6 调研与优化、§4.5 UB 预算、§5.2 阶梯、§5.5 分核、§9 风险清单 |
| `examples/TileOPs/tileops/kernels/reduction/argmax/argmax_kernel/perf_opt/_argreduce_kernel.py` | **最高**（同 kernel 的 Stage 4 终版实现，含链式候选 / 索引链 / tiled 无别名更新 / C6） | 显式向量链形态、`T.arange` 1-D 索引链、`bm=1`/`bm≥2` 分派、tiled 更新链、UB 预算常量 |
| `examples/TileOPs/tileops/kernels/reduction/argmin/argmin.py` | 高（本算子 wrapper，K5–K9 已适配） | `default_config` 改接 `_select_config`、`custom_op` 契约、`forward` 传参 |
| `examples/TileOPs/tileops/ops/reduction/argmin.py` | 高（本算子 Op 层） | `_pad_value() → +inf`、`_prepare_input` 的 pad 契约 |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/`（+ 其 DESIGN） | 高度相似（同族 row-reduction、Developer、`reduce` dim=1、尾块 `T.min` + 双 slice） | Kernel 结构、JIT 形态、尾块处理、L0 分层测试 |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_tiled/`（+ 其 DESIGN） | 高度相似（N > UB 流式、`T.serial` tile 循环、在线递推、`(bm,1)` fragment 更新链；**尾 tile kernel 侧掩码 segfault 反例 → 本设计静态尾 tile 特化依据**） | tile 循环结构、在线递推形态 |
| `examples/TileOPs/tileops/kernels/reduction/_primitives.py` | 高（`compute_tile_n` / `ub_slab_units` / `device_smem_budget` / `UB_SAFETY_RESERVE_BYTES` NPU 工具） | `tile_n` 选择与 UB 预算复用 |
| `testing/npuir/parallel_ops/test_if_then_else_cmp_cond.py` + `testing/npuir/reduction_ops/test_reduce.py` | 中（元素级条件 + reduce 先例） | API 可行性佐证 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(M, N_padded)`，`N_padded = align_up(N, 256)`（主选 pad 契约） | float16 / bfloat16 / float32 | 2D 连续行主序；`M`、`N` 为工厂期常量（无动态轴）。备选 raw-N 契约下为 `(M, N)`（§0.2） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `(M,)` | int64 | 首现最小值索引；`out_idx=[1]` 由 jit 分配返回 |

### 4.3 中间缓冲区（路径 S，fp16 形态；bf16/fp32 见 §3.2 矩阵）

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | `(bm, N_padded)` | 输入 dtype | UB（shared） | GM staging（唯一 shared **数据**缓冲） |
| `x_frag` | `(bm, N_padded)` | 输入 dtype | fragment | 计算源（竞态规避：多消费者全走 fragment） |
| `x_work`（仅 bf16） | `(bm, N_padded)` | float32 | fragment | bf16→fp32 精确域 |
| `m` | `(bm, 1)` | 计算域 dtype | fragment | 行极值 |
| `ext_brc` | `(bm, N_padded)` | 计算域 dtype | fragment | 极值同形广播（统一 bm 形态） |
| `cmp_eq` | `(bm, N_padded)` | bool | fragment | 相等掩码 |
| `cand` | `(bm, N_padded)` | float32 | fragment | 掩码索引候选 |
| `first` / `first_flat` | `(bm,1)` / `(bm,)` | float32 | fragment | 首现索引 / 出栈前扁平视图 |
| `idx_src` / `idx_row` / `idx_j` | `(N_padded,)` / `(1,N_padded)` / `(bm,N_padded)` | float32 | UB（shared，**任务不变操作数**） | 列索引链 |
| `sent_v` | `(bm, N_padded)` | float32 | fragment | `BIG` 哨兵广播 |
| `out_ub` | `(bm,)` | int64 | UB（shared） | 输出 staging |

路径 T 追加（全部 fragment，`(bm,1)` 类微缓冲）：`running_min/running_idx/chunk_min/chunk_first/chunk_global/cond_lt/new_min/new_idx`
- 尾 tile 独立静态宽度组 `x_tail/xw_tail/tail_brc/cand_tail/tail_min/tail_first/tail_global/cond_tail`
（尾 tile 的 `x_tail` 为 shared staging，其余 fragment）。

### 4.4 内存搬运路径

```
路径 S（纯 Vector 单引擎）：
GM[x] --T.copy(slice,real_m)--> UB[x_ub] --T.copy--> FRAG[x_frag]
  --T.reduce_min(dim=1)--> FRAG[m] --T.vbrc--> FRAG[ext_brc]
  --T.vcmp("eq")--> FRAG[cmp_eq] --T.vselect(idx_j, sent_v)--> FRAG[cand]
  --T.reduce_min(dim=1)--> FRAG[first] --reshape+vcast(rint)--> UB[out_ub] --T.copy(slice)--> GM[out]
无 L1/L0 参与；无 GM 中间往返；无 workspace。

路径 T：GM[x tile] --T.copy--> UB[x_ub] --T.copy/vcast--> FRAG --(chunk 链)--> FRAG[running_*]
（逐 tile 循环；running 状态全程片上驻留）
```

### 4.5 UB 内存预算（同族定标模型 + 本机实查容量复核）

**预算公式**：`手工字节预算(block) ≤ 64 KB`（竞态安全实测值；×2 auto-multi-buffer 双缓冲 ≈ 128 KB ≤ 192 KB − 8 KB reserve）。
依据链：UB = **实查 196608 B（192 KB）**（§0 实查表，`AscendArch().ub_cap`）；auto-multi-buffer 膨胀实测两数据点
（**1.60× = 含未用 `ext_brc` 的首版清单**；**2.0× = 净版清单**——不同 buffer 清单的配置，正是「未用 alloc 不被 DCE」的成因；
`CONST-capacity-910B2C` / `TRAP-UB-multibuffer-inflation`）；**预算超限不报错而是静默放弃双缓冲 → 多消费者 shared
缓冲数据竞争**（同族实测 15–25 行非确定 BIG 行；fragment 化 + 64 KB 预算后稳定）。

**B/elem 表**（fragment 形态，`bm≥2` / `bm=1`（含 `ext_brc`））：

| dtype | `x_ub` | `x_frag`(+`work`) | `cand` | bm≥2 合计 | bm=1 合计 |
|-------|--------|-------------------|--------|-----------|-----------|
| fp16 | 2 | 2 | 4 | **8** | 10 |
| fp32 | 4 | 4 | 4 | **12** | 16 |
| bf16 | 2 | 4 | 4 | **10** | 14 |

> **记账口径（v1 修正，建议 2：探针定标的代理预算）**：B/elem 为**探针定标的代理预算**——仅计 `x_ub` /
> `x_frag`(+`work`) / `cand` 三类主要计算缓冲（同族同清单探针 P3a 在 910B2C 标定）。其余 `(bm,N_padded)` 类缓冲
> （`ext_brc` / `sent_v` / `cmp_eq` / `idx_j`）与任务不变的 `idx_src` / `idx_row` 由编译器 **PlanMemory 的活跃期
> 复用/别名吸收**（同形 `(bm,N_padded)` 缓冲生命周期不重叠，不叠加计入峰值）——这些缓冲的字面和显著大于代理预算，
> 故「逐字相加」不是可编译上界的正确口径。其中 `idx_j`（`(bm,N_padded)` fp32）**随 bm 放大**，由活跃期复用吸收
> 而非 8 KB reserve；`(bm,1)` 类微缓冲仍按 `UB_SAFETY_RESERVE_BYTES`（8 KB）吸收。
> **跨设备适用性**：本机为 **910B3**（CANN 9.0.0），探针标定来自 910B2C（CANN 8.5.0）——须按 §9.2-R4 在
> Stage 3 以 L0-4 编译探针**复标定**；**代理预算不作为可编译证明**，实际超限由编译期报错显性暴露。

**逐 workload 预算验证**（驻留路径；tiled 见 §5.2）：

| 配置 | 手工预算 | ×2 双缓冲 | 结论（≤ 192 KB − 8 KB reserve） |
|------|---------|-----------|-------------------------------|
| fp16 bm=8 @ N_padded=256（smoke） | 16 KB | 32 KB | ✅ |
| fp16 bm=8 @ N_padded=512（non-aligned） | 32 KB | 64 KB | ✅ |
| fp16 bm=2 @ N_padded=4096（hidden-state） | 64 KB | 128 KB | ✅（同族同配置实测通过） |
| bf16 bm=1 @ N_padded=4096 | 56 KB | 112 KB | ✅ |
| fp32 bm=1 @ N_padded=4096 | 64 KB | 128 KB | ✅ |
| fp16 bm=4 @ N_padded=4096（**不采纳**，仅边界对照） | 128 KB | 256 KB | ❌ 编译拒绝（同族实测 `requires 2097408 bits`） |

**退化 / 极薄路径容量核算（显式补行，`hardware-cost-model.md` §5-①）**：`dim=None` 全张量规约被 Op 层 flatten 成
`(M=1, N=numel)`，以及 `M=1` 单行负载——**上述「代表 workload 的驻留结论不覆盖该路径**：`numel` 可远超驻留上限
（`N_padded` 最大 2^24）。该路径由 **tiled 路径（强制 `bm=1`）**统一承接：工作集 = `(1, tile_n)×slab‧B ≤ 64 KB`
（tile 级预算，与代表 workload 无关）→ **回退结论：M=1/大 numel 走 tiled，无容量风险**；`numel > 2^24` 由工厂期
`assert` raise（E1/E3 等价域外，§9.2-R11）。

### 4.6 动态轴定义

无（`M`、`N`、`dtype`、`op_kind` 均工厂期静态；kernel 内全部循环边界静态，运行时仅 pid/task guard）。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1], target="npuir")   # out 自动分配 (M,) int64
def _func(block_m): ...                       # K9: threads 去除；block_m 由 wrapper config 传入
```

工厂缓存：沿用 harness `ArgreduceKernel` 的 kernel cache（`(M,N)` 键）+ 源码 `lru_cache` 语义（§0.4-#7 保留）。
环境：`TILELANG_ASCEND_MODE=Developer`（wrapper `ascend_mode = "Developer"` 已声明）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector。

**判定依据**：全部计算为逐元素/规约/比较选择（`reduce_min`、`vcmp`/`vselect` 链、`reduce_min`），无 matmul；
数据通路 GM↔UB↔fragment；无 Cube/L0 参与。

### 5.2 Block 划分（`block_m` 阶梯 + 路径切换 + `tile_n`）

```python
# 工厂期（全部静态 Python 量）；N_padded = align_up(N, 256)（主选 pad 契约）
UB_MANUAL_BUDGET = 65536                             # 字节（§4.5 竞态安全手工预算）
B_multi = B_elem(dtype, bm_class="multi")            # §4.5 表 bm≥2 列：fp16 8 / fp32 12 / bf16 10
B_bm1   = B_elem(dtype, bm_class="bm1")              # §4.5 表 bm=1 列（含 ext_brc）：fp16 10 / fp32 16 / bf16 14
# 路径切换（两段式，与 §3.3 同序）：驻留阈值 = 64KB 手工预算
block_m = max([p for p in (1, 2, 4, 8)
               if p * N_padded * (B_bm1 if p == 1 else B_multi)
               <= UB_MANUAL_BUDGET] or [None])   # 基础阶梯（p=1 用含 ext_brc 的 B_bm1）
# 扩展阶梯（发射项驱动）：基础阶梯下逻辑 block 数 > 2×40 = 80（两波）时启用 {16..2048} 档，
# 档位由同一预算过滤器自动封顶。依据：小 N_padded 大 M 负载（如 dim=0 → N_padded=256、M=2048）
# 若停在 bm=8 → 256 block、发射项 ≈256×10/40×0.5 = 32µs；bm=32 → 64 block → 8µs。
if block_m is not None and ceildiv(M, block_m) > 80:
    block_m = max([p for p in (16, 32, 64, 128, 256, 512, 1024, 2048)
                   if p * N_padded * B_multi <= UB_MANUAL_BUDGET] + [block_m])
if block_m is None and N_padded * B_bm1 <= UB_MANUAL_BUDGET:   # bm≥2 无解 → bm=1 试驻留
    block_m = 1
if block_m is None:                                  # 仍无解 → 路径 T（tiled online）
    block_m = 1                                      # CG-2026-0014 强制
    # compute_tile_n（仓内原语 examples/TileOPs/tileops/kernels/reduction/_primitives.py）：budget 单位
    # 为【字节】、num_buffers = 同形 (bm,tile_n) 大缓冲折算数（ub_slab_units）
    slab = ub_slab_units(elem_bytes, dtype_slabs, fp32_slabs)   # fp16 3+1→5 / bf16 1+3→7 / fp32 4+0→4
    tile_n_cap = compute_tile_n(block_m=1, elem_bytes=elem_bytes, N_padded=N_padded,
                                alignment=256, budget=UB_MANUAL_BUDGET, num_buffers=slab)
    # wrapper 侧整除优先 + 裕度后置校正：候选集 = N_padded 的因子 ∩ 256 倍数 ∩ ≤ cap
    #   ∩ slab·elem·tn ≤ 0.9·UB_MANUAL_BUDGET，取最大者（零尾 tile + ≥10% 预算裕度）；空集回退 cap + 静态尾 tile
    tile_n = pick_divisor_with_margin(N_padded, tile_n_cap, slab, elem_bytes,
                                      UB_MANUAL_BUDGET, margin=0.9)
    # 非整除残余 tail = N_padded - num_full·tile_n 走静态尾 tile 特化（§5.4）
block_m = min(block_m, M)                            # 行数不足不超配（垃圾行污染，同族实测）
num_row_blocks = ceildiv(M, block_m)
assert N_padded <= 2 ** 24                           # E1/E3 fp32 索引精确域前提；超限 raise
```

**lm-head `tile_n` 推导链（冻结值复现，供 Stage 2/3 复算）**：

| dtype | slab | `compute_tile_n` 返回（cap） | 后置校正（整除因子 ∩ ≤cap ∩ ≤0.9×64 KB） | 冻结 `tile_n` / tiles / 手工预算 |
|-------|------|------------------------------|------------------------------------------|--------------------------------|
| fp16 | 5 | 6400（65536//10=6553 → align_down 256） | 6400 预算 64000 B > 58982 B ✗ → 次大因子 **5120**（51200 B ✓，20 tile 零尾） | **5120 / 20 / 50 KB** |
| bf16 | 7 | 4608（65536//14=4681 → align_down 256） | 4608 预算 64512 B > 58982 B ✗ → 因子 **4096**（57344 B ✓，25 tile 零尾） | **4096 / 25 / 56 KB** |

**逐 workload 参数表**（manifest 负载；M/N 由 manifest + `_ReduceOpBase._prepare_input` 推导并附算式）：

| workload（manifest） | dtype | 路径 | block_m | tile_n | 手工预算 | 依据/推导 |
|----------------------|-------|------|---------|--------|---------|-----------|
| smoke `[32,256] dim=-1` | fp16/bf16/fp32 | S | 8 | — | 16/20/24 KB | `M=32, N_padded=256`；基础阶梯 `p·256·B ≤ 64 KB` → 三者均 ≤32 → 取阶梯内最大 8；block 数 4 ≤ 80（不触发扩展） |
| non-aligned `[256,300] dim=-1` | fp16/bf16 | S | 8 | — | 32/40 KB | `M=256, N_padded=512`；fp16 `p≤16`、bf16 `p≤12.8` → 阶梯内 8；block 数 32 ≤ 80 |
| hidden-state `[2048,4096] dim=-1` | fp16 | S | **2** | — | 64 KB | `p·4096·8 ≤ 64 KB → p≤2`；block 数 1024 > 80 但扩展档受预算封顶（p≤2）→ bm=2 |
| hidden-state `[2048,4096] dim=-1` | bf16 | S | **1** | — | 56 KB | `p·4096·10 ≤ 64 KB → p≤1` |
| lm-head `[4,102400] dim=-1` | fp16 | T | 1 | **5120** | 50 KB | 驻留无解（102400×8 ≫ 64 KB）→ tiled；推导链见上表 |
| lm-head `[4,102400] dim=-1` | bf16 | T | 1 | **4096** | 56 KB | 同上 |

> **预算过滤器（`p=1` 修正，v1 建议 4）**：基础阶梯的预算过滤器对 `p=1` 使用 `B_bm1`（含 `ext_brc` 的更大值），
> 与 §4.5 B/elem 表及同族已验证 Stage-4 形态（`p==1 → B_bm1`）一致——旧形态对 `p=1` 误用 `B_multi`，
> 会把 `N_padded ∈ (4096, 5461] × fp32` 一类形状错误判为「bm=1 驻留可行」（选出实际超 64 KB 预算的 bm=1）。
> 修正后此类形状正确回落 **tiled 路径**（bm=1 + `tile_n`）。manifest 内负载不受影响：hidden-state fp32@4096
> = 64 KB（恰在上限）、bf16@4096 = 56 KB（B_bm1=14），均在预算内。

**`block_m` 选择理由**：阶梯内取预算最大值——行块越大每 block 摊薄发射越优（`CONST-vector-launch-overhead`），
受 64 KB 竞态安全预算硬约束（§4.5）。基础阶梯 `{1,2,4,8}`（源码与家族先例，逐档 B/elem：`p=1` 用 `B_bm1`、
`p≥2` 用 `B_multi`）覆盖 `N_padded ≥ 512` 类负载；
**小 `N_padded` 大 M 负载**由发射项门控（block 数 > 80）启用扩展阶梯至 2048（档位 = 预算过滤器自动封顶）。
**为何无需同族 argmax 的窄 N 预算减半**：主选 pad 契约下 `N_padded ≥ 256` 恒成立 → 窄内维（N < 256）域**不可达**，
同族窄 N 的 ~4× 膨胀与 deinterleave 路径整体不适用（§0.2 主选依据）。

### 5.3 约束分析

- **对齐约束**：尾轴 32 B 对齐由 `T.copy` 切片天然满足（logsumexp N=300 先例）；`N_padded` 为 256 倍数（pad 契约保证）。
- **UB 容量**：§4.5 预算表逐配置验证（✅/❌ 双向）；本机 UB 实查 196608 B。
- **L0/L1 容量**：不适用（纯 Vector）。
- **编译器陷阱约束**：tiled 路径 `bm=1` 强制（CG-2026-0014）；`bm≥2` 禁 2D `T.Parallel` 融合 `if_then_else`
  （CG-2026-0015）；tiled 更新链禁原位 `vselect`（`TRAP-vselect-inplace-carried-state`）；同形 `vbrc` 禁用
  （`TRAP-vbrc-same-shape-empty-broadcast` / CG-2026-0017）。

### 5.4 注意事项（非整除/边界处理策略）

- **M 非整除（`M % block_m ≠ 0`）**：`real_m = T.min(block_m, M − start)` + src/dst 双显式 slice `T.copy` +
  垃圾行不写出（VP-2026-0044 三件套；§0.5 源缺陷修正）。
- **N 非整除（主选 pad 契约）**：由 Op 层 `F.pad`（`+inf` 单位元）+ kernel 声明 `N_padded` 处理；kernel 内在
  **N_padded 全域**计算候选，与只算实列索引恒等（§1.6.1-E5 机器验证）。**无 kernel 侧掩码、无静态尾 tile**——
  N 向控制流最简。
- **`tile_n` 非整除（tiled 路径）**：`compute_tile_n` 除数优先；残余 `tail` 走**静态尾 tile** 代码路径
  （宽度为工厂期常量；**不用 kernel 侧 `if_then_else` 掩码**——logsumexp tiled 的 serial 变量条件 segfault 反例）。
- **M=1 / `dim=None` flatten**：`M=1`、`numel` 可超驻留上限 → tiled 路径承接（§4.5 退化路径核算行）。
- **N=1**：`N_padded=256`；候选唯一命中 j=0（机器验证 `(1,1)/(1,2)` 用例）。
- **M=0**：grid=0 边界（§9.2-R6 风险记录，harness 测试不覆盖）。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（依据 `docs/开发指南.md` §3.3）。

- **① 逻辑核数**：`num_logical_kernels = ceil(M / block_m)`（纯行并行，无 N 向分核；tiled 路径 `bm=1` →
  `num_logical_kernels = M`）。
- **② 物理核数（实查，禁止假设值）**：
  `from tilelang.utils.npu_utils import NPUUtils; NPUUtils.get().get_aicore_num()` → **20**（2026-10-14 15:13 实查，
  Ascend910B3，npu-smi 26.1.1）；**纯 Vector 算子核数**由 `AscendArch().aivector_core_num`（等价
  `NPUUtils.get().get_aivector_core_num()`）实查 = **40**（= 20 × 2）。查询代码与返回值记录于本节（本节即实查记录）。
  `ub_cap` = **196608 B（192 KB）** 同源实查。**该实查值与同族 argmax 任务的 910B2C（24 → 48）不同——常数与陷阱
  的跨设备复用均按版本戳标注（§9.1）。**
- **③ 规模判定与分核方案**（persistent 统一结构：`num_kernels = min(num_logical_kernels, 40)`；核内
  `T.serial(静态上界 = ceildiv(num_logical_kernels, num_kernels))` + grid-stride 任务映射
  `task = cid + s·num_kernels` + 动态 guard `if start < M`）：

| workload | `num_logical_kernels` | 判定 | `num_kernels` | 核内任务数（静态上界） | 依据 |
|----------|----------------------|------|---------------|----------------------|------|
| smoke (32,256) | 4（bm=8） | **逻辑 ≤ 物理（4 ≤ 40）** | 4 | 1 | 无需适配：行块独立、每核 ≤1 任务、单波 |
| non-aligned (256,300→512) | 32（bm=8） | **逻辑 ≤ 物理（32 ≤ 40）** | 32 | 1 | 同上 |
| hidden-state (2048,4096) fp16 | 1024（bm=2） | **极大规模（1024 ≫ 40，bm 受 UB 预算不可再增）** | **40** | **26**（ceildiv(1024,40)） | 固定 40 核 + 核内 `T.serial`（静态边界 26 + guard）；消核启动开销与串行调度（`PL-1.21-vector-persistent-split`） |
| hidden-state (2048,4096) bf16 | 2048（bm=1） | 极大规模 | 40 | 52 | 同上 |
| lm-head (4,102400) fp16/bf16 | 4（bm=1, tiled） | **逻辑 ≤ 物理（4 ≤ 40）** | 4 | 1 | 核利用率 4/40——C6 N-split 为 Stage 4 候选（§1.6.3-#4/§1.6.4） |
| dim=0 类（测试）(2048,256) | 256（bm=8）→ 64（bm=**32**，扩展阶梯） | 极大规模 → 调整 block 后 64 > 40 | 40 | 2 | 扩展阶梯把小 N_padded/大 M 的核数从 256 降到 64，再 persistent 到 40 核 |

> 核内串行循环边界全部为**静态值**（工厂期 Python 整数：`ceildiv(num_logical_kernels, num_kernels)`）；
> 动态性仅存在于 guard（`if start < M`，`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 先例形态）。
> wrapper `default_config` 须按 §5.2 公式适配（其 GPU 48 KB SMEM 启发式弃用，见 §5.6）。

**核数复算（供检视复算，逐负载；`num_logical_kernels = ceil(M / block_m)`）**：

- smoke：`M = 32`、`block_m = 8` → num_logical = **4**（≤ 40 → num_kernels = 4，单波）
- non-aligned：`M = 256`、`block_m = 8` → num_logical = **32**（≤ 40 → num_kernels = 32）
- hidden-state：`M = 2048`、`block_m = 2` → num_logical = **1024**（> 40 → num_kernels = 40，核内 `T.serial` 上界 = ceil(1024/40) = **26**）
- lm-head（tiled）：`M = 4`、`block_m = 1` → num_logical = **4**（≤ 40 → num_kernels = 4）
- dim=0 类（扩展阶梯后）：`M = 2048`、`block_m = 32` → num_logical = **64**（> 40 → num_kernels = 40，核内上界 = **2**）

### 5.6 集成前置核对（VP-2026-0134 / VP-2026-0128）

| 核对项 | 结论（实读目标工件） | 实现期动作 |
|--------|----------------------|-----------|
| 目标 Op 层 `_kernel_handles_padding` | `examples/TileOPs/tileops/ops/reduction/argmin.py` **未覆写** → 继承 `_ReduceOpBase._kernel_handles_padding = False`（`examples/TileOPs/tileops/ops/reduction/reduce.py` 第 91 行） | 保持（主选 pad 契约） |
| 目标 Op 层 `_pad_value` | `float("inf")`（`argmin.py` 第 76–78 行）——**argmin 必须 `+inf`**（`-inf` 会让 pad 列在 argmin 中胜出，静默错值） | 保持；集成回归须断言 pad 方向 |
| kernel 声明宽度 vs Op 传入宽度 | Op 在 `_prepare_input` pad 到 `align_up(N,256)` 后传入 → kernel 必须声明 `(M, N_padded)` | 工厂内部派生 `N_padded` 并按其声明（宽度错位症状：行 0 对、行 ≥1 全错不报错） |
| wrapper `default_config` | 现为 GPU 期 48 KB SMEM 启发式（`SHARED_MEMORY_BUDGET_BYTES // (N_padded × elem)`）→ 在 NPU kernel 上 `ub overflow` | 改为转调 kernel 模块 `_select_config(M, N, dtype)`（§5.2 为单一事实源） |
| msprof 锚点 | 现无 `msprof_kernel_name` | 增加（`tiled` → `argreduce_tiled`，`resident` → `main`），避免 msprof 捕获 Op 层 prep 算子 |

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| M 方向（block 级） | persistent 任务循环 | `T.serial(num_local_tasks)`（静态上界 + grid-stride + guard） | 极大规模核内串行（§5.5）；消核启动开销 |
| N 方向（tiled 路径） | tile 迭代 | `T.serial(num_full − 1)`（静态；tile-0 剥离初始化） | 在线递推的跨 tile 顺序依赖（E3）；静态边界要求 |
| 元素级（候选） | 向量化（显式链） | `T.vbrc` + `T.vcmp` + `T.vselect` | `bm≥2` 禁 2D `T.Parallel` 融合（CG-2026-0015） |
| 元素级（极值/首现） | 向量化归约 | `T.reduce_min(dim=1)` ×2 | 原语（§3.2） |
| 索引构建 | 向量化 | `T.arange`（1-D）+ `T.reshape` + `T.vbrc` | `CONST-arange-scalar-materialize` 的 1-D 形态 |
| 出栈 | 向量化 | `T.reshape` + `T.vcast(rint)` + `T.copy` | 形状收缩由 reshape 桥接（§1.6.2-#9） |
| 尾块行数 / 任务 guard | 标量 | `T.min` + `if start < M` | 动态边界（§1.6.2-#10） |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (cid, _):
    # 任务不变操作数（循环外构建）：idx_j / sent_v
    # 路径 S：单层任务循环
    for s in T.serial(num_local_tasks):              # 静态上界
        start = (cid + s * num_kernels) * block_m
        if start < M:                                # 动态 guard
            [§3.3 路径 S 主体：copy→frag→reduce_min→vbrc→vcmp→vselect→reduce_min→出栈]
    # 路径 T：任务循环内嵌 tile 循环
    #   tile-0 直写初始化（无循环）
    #   for t in T.serial(num_full - 1): [chunk 链 + 严格小于无别名更新]
    #   静态尾 tile（无循环，独立代码路径）
    #   出栈
```

### 6.3 流水线优化

**初始版本不使用 `T.Pipelined`**（auto-multi-buffer 由 Developer 模式自动施加；同族已实测其在任务/tile 循环上的
正确性与竞态边界——§4.5 纪律）。**Stage 4 候选（按 D-2 发射项分析排序）**：① `--enable-auto-multi-buffer=false`
显式分配结构（可放宽 UB 预算、允许 bm 升档；需重验竞态）；② tiled 路径 tile 循环软件流水（tn/bm 扫描）；
③ **C6 N-split partial+merge**（§1.6.4；同族实测 ≈0.46–0.54× 主选）；④ 融合候选 value 分支内联全局索引
`t·tn+j`（消 base-add 微操作，需探针验证 serial 变量于 value 分支的 codegen）；⑤ 扩展阶梯 bm 扫描
（小 N_padded/大 M 负载 bm ∈ {8,16,32} A/B）。

### 6.4 尾块处理

见 §5.4（M 尾块：`real_m` + 双 slice + 垃圾行丢弃；N 向：主选 pad 契约全宽计算、tiled 尾 tile 静态宽度特化；
guard 静态化边界）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer 模式编译器管理 UB/fragment 依赖与 auto-multi-buffer 流水同步）。

### 7.2 同步点说明

无手动同步点（单 kernel、无核间协作、无跨引擎传输；C6 备选的两段 kernel 之间由 wrapper 顺序发射保证依赖，
无核间 sync 需求）。persistent 任务循环与 tile 循环内的生产者-消费者依赖由编译器自动插入——前提为 §4.5 的
fragment 纪律（计算一律读 fragment，`shared` 仅 GM staging）。

### 7.3 pass_configs 配置

无特殊 pass 配置（Developer 默认；`TILELANG_ASCEND_MODE=Developer` 环境变量）。Stage 4 探索项：
`--enable-auto-multi-buffer=false`（§6.3-①）。

---

## 8. 验证方案

### 8.1 Golden 函数

> golden 以 §0.1 语义为唯一依据（`torch.argmin`，first-occurrence + int64 + IEEE 相等匹配；torch NaN 语义为
> 文档化边界），**不复刻 §0.6 NPU 算法**（无掩码候选 / `reduce_min` / 在线递推结构）——保证验证独立性。

```python
def golden_argreduce(x: torch.Tensor, op_kind: str = "argmin") -> torch.Tensor:
    """PyTorch 参考实现（CPU）：argmin first-occurrence 语义。
    输入任意 rank；与 harness ArgminFwdOp 的 dim/keepdim 语义对齐由 Op 层测试覆盖，
    kernel 级 golden 只承接 (M, N) -> (M,) 契约。"""
    return torch.argmin(x, dim=-1) if op_kind == "argmin" else torch.argmax(x, dim=-1)
```

（harness pytest 侧 ref：`x.argmin(dim=...)` on device——§9.2-R2 记录其 ±0.0 设备实现差异对 `randn` 测试无影响。）

### 8.2 精度标准

**精确比对**（索引语义，无 atol/rtol 容差）：`torch.equal(out, ref)` 且 `out.dtype == torch.int64`
（与 harness `test_argmin.py` 同款）。数学依据：E1–E5 为**索引恒等**（§1.6.1 机器验证 0 偏差），任何容差放松
都会掩盖 tie-break / 哨兵类 bug。

**L0 门槛测试计划**（gate 用例；完整 L1/L2/Boundary 分层套件交由 `tilelang-op-develop` 按本节展开）：

| # | 用例 | 覆盖点 |
|---|------|--------|
| L0-1 | `(32,256)` fp16 / bf16 / fp32，bm=8 | smoke 负载 ×3 dtype |
| L0-2 | `(4,102400)` fp16，tiled tn=5120（§5.2 冻结值） | lm-head + tiled 路径 + 多 tile 在线递推 |
| L0-3 | `(4,102400)` bf16，tiled tn=4096（§5.2 冻结值） | bf16 tiled（`vcast` 预转 + 整除 `tile_n`） |
| L0-4 | `(2048,4096)` fp16 bm=2 | hidden-state + persistent 40×26 任务循环（多波次竞态回归） |
| L0-5 | `(2048,4096)` bf16 bm=1 | 预算边界 dtype 路径 |
| L0-6 | `(256,300)` fp16 / bf16（N_padded=512） | **非对齐 N + pad 契约宽度一致性判别用例**（宽度错位 → 行 ≥1 全错） |
| L0-7 | `(129,512)` fp16 | M 尾块（`129 % bm ≠ 0`）+ 源缺陷修正复验 |
| L0-8 | `(1, N)` 与 `(numel,)` flatten（`dim=None` → M=1） | 单行负载 + M=1 驻留/tiled 分界（§4.5 退化路径） |
| L0-9 | 构造行：重复最小值（跨 tile tie）/ 全 `+inf` / 全 `-inf` / `±inf` 混合 / 亚正规最小 / N=1、N=2 | 首现 tie-break + 边界值（`verify_equiv.py` 角点集的 NPU 侧复验） |
| L0-10 | argmax kind 镜像（`(32,256)`×3 dtype + `(4,102400)` fp16） | 工厂双 kind 契约（共享 kernel 接口完整） |
| L0-11 | 集成宽度一致性：`(8,300)` fp16 → 断言 kernel 声明 `(8,512)` 且输出与 `torch.argmin` 全等 | VP-2026-0128 错位缺陷回归 |

L2（记录不阻断）：NaN 行为（§9.2-R1，断言**记录值** = 哨兵 `2^30` 而非崩溃）、混合符号零行（记录 vs
torch-CPU / torch-NPU 两基准）、`M=0`、`(1,1)` 极小、`tile_n+1` 恰跨分界、`N_padded > 2^24` 工厂断言行为。

### 8.3 等价性验证工件

`verify_equiv.py`（随算子目录交付，已执行 ALL EQUIV_PASS，§1.6.1 结果表；v1 起含 3 组**判别力对照**——
故意错误候选必须被报出差异，否则脚本自身失败）——Stage 2 检视维度 8 重跑复核对象。

---

## 9. 风险点与注意事项

### 9.1 已知约束（工具链级；**逐条标注工具链戳与是否需重验**）

> 本任务工具链：tilelang dev build `ac0522ed` + **CANN 9.0.0** + torch_npu 2.9.0.post8 + **Ascend910B3**。
> 下表中绑定 910B2C / CANN 8.5.0 的条目为**跨工具链/跨设备引用 → 待重验**（`kb_stale_check.py` 亦报相关条目 stale）；
> 绑定 CANN 9.0.0 / 910B3 的条目与本任务同工具链，可信度更高。

| # | 约束 | 实证来源（工具链戳） | 设计内规避 |
|---|------|---------------------|-----------|
| C-1 | reduce 混合 dtype（fp16 src→fp32 dst）**静默产出错误值** | 同族探针 dbg-B（argmax 任务，CANN 8.5.0 / 910B2C，**待重验**）+ queue VP-2026-0119 派生 `TRAP-reduce-mixed-dtype-silent-corrupt`（待合入） | §3.2 强制同 dtype reduce + 同 dtype 比较 |
| C-2 | 2D `T.Parallel(bm,N)` 融合 `if_then_else` 在 `bm≥2` **整体标量化**（与操作数形态无关） | `TRAP-parallel2d-bm-ge2-scalarize` / CG-2026-0015（CANN 8.5.0 / 910B2C，**待重验**） | §0.6-R1/§1.6.3-#5：`bm≥2` 走显式向量链（融合式仅 `bm=1` 回退） |
| C-3 | `bm=1` 时 Parallel 条件中的 `(·,1)` 不变操作数被误 lower 为 `f16→i1` 广播 | 同族探针 P2（CANN 8.5.0 / 910B2C，**待重验**）；CG-2026-0016 亲缘 | §1.6.2-#6：统一 `vbrc` 同形化 |
| C-4 | **未用 fragment alloc 不被 DCE**（计入 UB 规划） | 同族探针（**待重验**）；`CONST-capacity-910B2C` 小 buffer 漏计条目 | §3.2 dtype 路径矩阵：每路径独立 `prim_func`，禁「无条件 alloc + 条件使用」 |
| C-5 | **shared 多消费者缓冲在 auto-multi-buffer 预算超限时静默竞态** | 同族探针（**待重验**） | §4.5 纪律②：计算全走 fragment + 64 KB/block 预算 |
| C-6 | multitile + `bm>1` → 编译器 **SIGSEGV** | CG-2026-0014（CANN 8.5.0 / 910B2C，**待重验**；触发矩阵 bm=1 任意 tile ✓ / bm>1 单 tile ✓ / bm>1 多 tile ✗） | §5.2：tiled 路径强制 `bm=1` |
| C-7 | 原位 `T.vselect(cond, A, B, B)` 在多迭代 `T.serial` 循环**丢失 loop-carried 状态** | `TRAP-vselect-inplace-carried-state`（CANN 8.5.0 / 910B2C，**待重验**） | §3.2 T2：无别名（scratch + `T.copy` 回写） |
| C-8 | `T.vcast` **f32→f32 非恒等**（退化为四舍五入到整数） | `TRAP-vcast-same-dtype-rounds`（CANN 9.0.0 / **910B3 同工具链**，双工具链复现） | §3.2：fp32 路径一律同 dtype `T.copy` |
| C-9 | 同形 `T.vbrc`（无尺寸 1 差维）→ 空广播维 MLIR verify fail | `TRAP-vbrc-same-shape-empty-broadcast` / CG-2026-0017（CANN 8.5.0 / 910B2C，**待重验**） | §1.6.2-#5：`bm=1` 用 `T.reshape` 直进，不用同形 `vbrc` |
| C-10 | `T.arange` 的 `(M,W)` 物化按标量执行（~1 ns/elem）；`vbrc` 的 region dst 写入被静默丢弃 | `CONST-arange-scalar-materialize`（CANN 8.5.0 / 910B2C，**待重验**）+ `TRAP-npuir-vbrc-region-dropped`（CANN 9.0.0 / **910B3 同工具链**） | §1.6.2-#5：1-D `arange` + reshape + 整 buffer `vbrc`（非 region） |
| C-11 | `T.if_then_else` 条件引用 `T.serial` 循环变量 → select codegen **segfault** | `TRAP-vbrc-scalar-shared`（CANN 8.5.0 / 910B2C，**待重验**）；logsumexp tiled 反例 | §1.6.2-#8：候选条件只含 buffer 元素；tile 基址走 value 算术 |

### 9.2 风险点

| # | 风险 | 影响 | 缓解 |
|---|------|------|------|
| R1 | **NaN 行为分歧**：NPU `reduce_min` 传播 NaN → 匹配不成立 → 输出哨兵 `2^30`；torch（CPU/NPU）返回首个 NaN 索引；源 GPU fmax 返回非 NaN 极值位置（全 NaN 行 → 0）——**三方各异**（`verify_equiv.py` INFO 行实测） | NaN 输入行索引不符合 torch 语义 | 语义域文档化（§0.1）；harness 测试全 `randn`（无 NaN）不受影响；L2 记录行为；如需对齐 torch 可 Stage 4 加 `vcmp(x,x,"eq")` 取反的 NaN 预检 pass（成本 +1 全宽 pass，暂不做；注意 `T.vcmp "ne"` 是 ordered-NE 不可用于 NaN 探测，`TRAP-vcmp-ne-ordered-nan`） |
| R2 | **±0.0 混合零行 vs torch-NPU 设备实现分歧**：本 kernel IEEE 相等（与 torch-CPU golden 一致）；torch-NPU argmin 偏好 `-0.0` | 行极值恰为 0 且含混合符号零的行，harness pytest（device ref）可能不等 | `randn` 输入下概率 0（需构造）；kernel 级 golden（CPU）一致；L2 记录；不改设计（torch 自身 CPU/NPU 不一致，无单一正确基准） |
| R3 | tiled 路径 `bm=1` 由编译器陷阱（C-6）强制，行块不可放大 | lm-head 类负载核利用率 4/40 | 主选接受（D-2 估算 ≈30–60 µs）；C6 N-split 为 Stage 4 备选（§1.6.4 三件套齐备；同族实测锚点约 0.46–0.54×） |
| R4 | 竞态规避依赖 fragment 纪律 + 64 KB 预算（经验值，非编译器契约） | 新 shape/新配置贴线时可能复现 C-5 | §4.5 双向实测表 + L0-4 多波次回归用例；实现纪律写入 §4.5；Stage 4 探索 auto-multi-buffer=false 结构 |
| R5 | `(·,1)` 广播 bug（C-3）的规避依赖 bm 分派正确性 | `bm=1`/`bm≥2` 误用 → 编译失败（显性，非静默） | 编译期失败即可发现；L0-8 M=1 用例覆盖 `bm=1` 路径 |
| R6 | **M=0**（非规约维含 0）：grid=0 边界未验证 | 可能启动异常或空输出 | Op 层短路建议（wrapper `forward` 加 `if M == 0: return torch.empty(0, dtype=torch.int64)`）；harness 测试不覆盖；L2 记录 |
| R7 | 主选 pad 契约依赖 Op 层维持 `_kernel_handles_padding=False` + `_pad_value=+inf` | 若集成期翻转标志而 kernel 仍声明 `N_padded` → 宽度错位（行 ≥1 全错） | §0.2 一致性规则 + §5.6 核对表 + L0-6/L0-11 判别用例；备选 raw-N 契约规格已备（§0.2） |
| R8 | 主要性能常数（MTE2 曲线 / 发射开销）绑定 910B2C，本机为 910B3 | D-2 估算可能系统性偏差 | §1.6.0 已标注「未实测假设」；Stage 4 按 `hardware-cost-model.md` §3 回填对账并回写 constants.md |
| R9 | 跨工具链陷阱（C-1..C-7、C-9..C-11）在 CANN 9.0.0 / 910B3 上**未复验** | 规避形态可能过保守（性能损失）或不足（新形态） | §9.1 逐条标注；Stage 3 首跑以 L0-4/L0-6 快速暴露；异常时按 `[DESIGN_LIMIT]` 受控修订 |
| R10 | **`tile_n` 静态尾 tile 的 bf16 预算裕度 12.5%**（57344 B / 65536 B，0.9 裕度规则内） | 编译器版本行为变化可能溢出/竞态 | `tile_n` 规则参数化（可降 tn=2048：32768 B，50% 裕度）；L0-3 回归 |
| R11 | **`N_padded > 2^24` 超界输入**（fp32 索引不可精确表示，E1/E3 等价域外；`dim=None` 全张量 numel > 16.7 M 时触发） | 静默索引失精 | §5.2 工厂期 `assert N_padded <= 2**24`（超限 raise 并提示分块）；manifest 最大 N=102400 安全；L2 记录断言行为 |

### 9.3 特殊场景处理

非整除分块（§5.4）；极小 shape（`(1,1)` 驻留 bm=1，`verify_equiv.py` 覆盖）；混合精度（per-dtype 路径矩阵 §3.2）；
`dim=None` 全张量（Op 层 flatten → M=1，§5.4）；跨 tile tie（L0-9 构造行 + `verify_equiv.py` E3 专项）；
`dim≠-1`（Op 层 movedim 到末轴后进入同一 kernel，新增负载按 §0.2 评估主选/备选契约）。

### 9.4 能力缺口登记（Compiler Capability Gap Rule）

本轮设计确认的缺口**均已存在条目**（dedupe 结果；本轮以「消费 + 版本戳重验诉求」方式引用，不重复登记）：

| 缺口 | 层级 | 已有条目 | 本设计角色 |
|------|------|---------|-----------|
| multitile + `bm>1` 编译器 SIGSEGV | TileLangIR pass（向量化/分块） | `CG-2026-0014`（已存在） | 规避形态（tiled 强制 bm=1） |
| 2D `T.Parallel` 融合 `if_then_else` `bm≥2` 标量化 | TileLangIR（npu_loop_vectorize） | `CG-2026-0015`（已存在） | 规避形态（显式向量链） |
| reduce 混合 dtype 静默错误值 | BishengIR（VReduce 类型校验缺失） | queue VP-2026-0119 派生条目（**待合入** `TRAP-reduce-mixed-dtype-silent-corrupt`） | 设计约束（同 dtype） |
| `(·,1)` 不变操作数 i1 误型广播（`bm=1`） | BishengIR verifier | `CG-2026-0016`（已存在） | 规避形态（`vbrc` 同形化） |
| 同形 `T.vbrc` 空广播维 verify fail | BishengIR（vbrc verifier） | `CG-2026-0017`（已存在） | 规避形态（`reshape` 直进） |
| auto-multi-buffer 预算超限静默竞态 | BishengIR PlanMemory/流水 | queue VP-2026-0121 / `CONST-capacity-910B2C` 条目 | 预算纪律（64 KB + fragment） |

> **重验诉求**：上述条目中绑定 910B2C / CANN 8.5.0 者与本任务（910B3 / CANN 9.0.0）工具链**不同**，
> 按 `capability-gaps.md` 的版本戳规则**须重验后方可继续引用**——建议 Stage 3/4 在 L0-4/L0-6 首跑时顺带记录
> 各条在本工具链的触发状态，并在 `RETROSPECTIVE.md` 提交版本戳更新（D 类价值点三件套）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/argmin/_argreduce_kernel/
├── DESIGN.md                          # 本设计文档
├── verify_equiv.py                    # D-1 等价性机器验证（已执行 ALL EQUIV_PASS）
├── RETROSPECTIVE.md                   # Stage 1 复盘（本次产出）
├── _argreduce_kernel.py               # ⬜ Stage 3：算子实现（kernel + golden + L0 测试）
└── perf_opt/                          # ⬜ Stage 4：调优工作区（含 §1.6.4 C6 备选 A/B）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成（v1，2026-10-14） | 本文档（含 §0 迁移分析 + §1.6 调研与优化 + §5 分核三要素；v1 采纳 5 条检视建议的定向修订，见「修订说明」） |
| `verify_equiv.py` | ✅ 已执行（v1 重跑） | 5 组门禁等价全部 `EQUIV_PASS`（1146 行 / 0 违反）+ 3 组判别力对照全部 `CONTROL_OK`；NaN 边界 INFO |
| `RETROSPECTIVE.md` | ✅ 本次产出 | Stage 1 复盘 + 价值点提案 |
| `_argreduce_kernel.py` | ⬜ 待实现（Stage 3） | 按本设计 §3.3/§5/§6 实现 + L0 套件 §8.2 |
| `perf_opt/_argreduce_kernel.py` | ⬜ 待实现（Stage 4） | 调优 + §1.6.4 C6 A/B（含 `argreduce_partial`/`argreduce_merge`） |

### 10.3 命名规范

- 项目目录：`examples/argmin/`（project_name=argmin）；算子目录：`_argreduce_kernel/`（op_name，harness 工厂函数名）。
- 实现文件：`_argreduce_kernel.py`（与工厂函数同名的 Developer 模式自包含形态：kernel + golden + L0-L2 自测，
  logsumexp / argmax 先例）。
- 集成落点（harness 侧）：`examples/TileOPs/tileops/kernels/reduction/argmin/` 的 wrapper 改 import 指向本 kernel
  - `default_config` 按 §5.2 适配 + Op 层维持 `_kernel_handles_padding=False`/`_pad_value=+inf`（§5.6）。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）
2. ✅ 等价性机器验证（`verify_equiv.py`，E1–E5 ALL PASS）
3. ✅ 设备/容量实查（`get_aicore_num()`=20 / `aivector_core_num`=40 / `ub_cap`=196608）
4. ⬜ 算子实现（`_argreduce_kernel.py`：路径 S/T 双 `prim_func` 族 + 工厂 + L0 测试）+ golden 精确比对
5. ⬜ harness 接线（wrapper import + `default_config` + msprof 锚点 + `test_argmin.py` / `bench_argmin.py`）
6. ⬜ Stage 4：性能调优 + §1.6.4 C6 A/B 裁决 + 常数回填
