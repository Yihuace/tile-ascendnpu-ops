# MaxPool1dIndicesFwdOp（`_max_pool1d_with_indices_kernel`）算子设计文档

> **迁移任务（TileOPs 同构仓迁移，harness 子模式，逐函数）**：源算子 = `MaxPool1dIndicesFwdOp`
> （family=`pool`，`variant_of` = `MaxPool1dFwdOp`），本次重实现的 TileLang kernel 函数 =
> `_max_pool1d_with_indices_kernel`（→ Kernel class `MaxPool1dWithIndicesKernel`，**双输出**：
> values + int64 indices）。本文按「源算子三问解读（§0）→ 算法调研（§1.6.0）→ 硬件耦合性判定与
> NPU 重设计（§0.5/§0.6）→ 通用设计流程（§1–§10）」组织。§1–§10 一律以 §0.6 重设计后的 NPU 算法为输入。
>
> - project = `max_pool1d_with_indices`，op = `_max_pool1d_with_indices_kernel`
> - 编程模式：**Developer**
> - 源算子路径（kernel）：`/workspace/TileOPs-gpu/tileops/kernels/pool/max_pool1d.py`
>   （函数 `_max_pool1d_with_indices_kernel`，第 212–288 行；同文件第 15–68 行是单输出变体，
>   已由同族前作迁移，**不在本次范围**）
> - 源 helper：`/workspace/TileOPs-gpu/tileops/kernels/pool/common.py`（`pool_output_dim`）
> - 源 host 层（op）：`/workspace/TileOPs-gpu/tileops/ops/pool.py`（`MaxPool1dIndicesFwdOp` 与
>   `_MaxPoolFwdOpBase` / `_max_pool_roofline` 均在该文件）
> - NPU 侧抽取参考件（Stage 0 产物）：`examples/TileOPs/tileops/kernels/pool/max_pool1d_with_indices/_max_pool1d_indices_fwd_kernels.py`
> - NPU 侧集成包（host 层 / Kernel 类）：`examples/TileOPs/tileops/kernels/pool/max_pool1d_with_indices/max_pool1d_with_indices.py`
> - 迁移元数据：`examples/TileOPs/tileops/kernels/pool/max_pool1d_with_indices/.migration_meta.json`
> - 规格来源（workloads / roofline / shape_rules）：`examples/TileOPs/tileops/manifest/pool.yaml`（`MaxPool1dIndicesFwdOp`）
> - 精度用例：`examples/TileOPs/tests/ops/test_pool.py`；性能用例：`examples/TileOPs/benchmarks/ops/bench_pool.py`
> - 设备常数（本设计会话实查，2026-10-14）：`NPUUtils.get().get_aicore_num()` = **20**（Ascend910B3，
>   `npu-smi info` 显示 `910B3`）→ 纯 Vector 算子核数翻倍 = **40**
> - **toolchain stamp（pattern-library 口径）**：`tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2（dev root build）`
>   `+ torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3`（dev root build hash 取自 pattern-library
>   `traps-compiler.md` 同会话 2026-10-14 记录；CANN 版本取自 `examples/TileOPs/reports/tileops/` 下
>   `run.json` 的 `setup.environment.cann` = `9.0.0`）
> - **任务快照（task snapshot，单独记录，非 tilelang 版本）**：`892e2c18d80c37795727236b5ebbe145626a8313`
>   （`tilelang.__version__` 后缀与 `git rev-parse HEAD` 同值——该串是**本仓快照**，不是 tilelang 版本）
> - 同族前作（骨架复用对象）：`examples/max_pool1d/_max_pool1d_kernel/DESIGN.md`（单输出 1D 变体，已过 Stage 3）、
>   `examples/adaptive_max_pool2d_with_indices/_adaptive_max_pool2d_with_indices_kernel/DESIGN.md`（双输出索引变体）
> - 版本：**revision v1**（Stage 2 检视「不通过」后回退修订；上一版备份 = `history_version/design_v0.md`）。

---

## 相对上一版的关键调整（revision v1 vs `design_v0.md`）

> 触发：Stage 2 检视结论「**不通过**」（2 阻塞 + 8 建议）。本表逐条列出相对 `design_v0.md` 的修改与
> **「为什么不会再犯同一错误」**。§0.1 的边界域断言（阻塞 #2）属 Phase M0 语义解读，本版已**重读源码重做**该环节；
> 阻塞 #1（容量系数）属 §4.5 核算，本版重导全部派生量。

| # | 上一版（v0） | 本版（v1） | 依据 / 为什么不会再犯 |
|---|---|---|---|
| A1 | §4.5 容量闭式 `G = CH·LSPAN·(2·eb+9) + BL·CH·(53+2·eb) + 256`——字节系数 `53 = 25+28` **漏计** `idx32`/`idx_t`（8）+ `idx64`（8），与同行分量注释（求和 `69`）自相矛盾 | 系数改为 **`(69 + 2·eb)`**（`= 25+2·eb` 单输出基式 + `7×i32` 28 + `idx32/idx_t` 8 + `idx64` 8）；分量注释求和改为 `69`；`+256`（`fv`）保留 | 逐 buffer 表逐项求和 = 闭式值（双重佐证），且 pattern-library `layout.md` `PL-1.4` 的 1D **单输出**版式即为 `BL·CH·(25+2·eb)`。**不会再犯**：新增自检纪律——闭式括号内分量求和必须等于系数，逐 buffer 求和必须等于闭式值，二者本版已互相印证 |
| A2 | 修正系数后 `sincnet` fp16（`BL=18`）`G=129 280 B`、`ecg` bf16（`BL=20`）`G=132 800 B`，`×1.7` 后 **219.8 / 225.8 KiB > 192 KiB**（v0 误判 ✓） | 取**路线 1（按默认档 ×1.7 降 `BL`）**：**`sincnet BL=16`**（`LSPAN=48`，`G=114 944 B`，`×1.7=195 405 B ≤ 192 KiB`）、**`ecg BL=17`**（`LSPAN=41`，`G=113 792 B`，`×1.7=193 446 B ✓`）；连带重算 `num_oblk`/`num_logical`/`num_local` 及 §1.6.3 备选档 | 路线 2（改证 `INFLATE`）需额外声明 auto-multi-buffer 关闭机制 + stage 回退，本版不采用。**不会再犯**：新增「容量档变更 ⇒ 全部派生量（`LSPAN`→`num_oblk`→`num_logical`→`num_local`）连带重算」纪律 |
| A3 | §0.1 断言「`p_W ≤ ⌊k_W/2⌋`、`s_W>0`、`d_W>0` ⇒ 每个输出窗口至少含一个有效输入元素（无需空窗口兜底）」——**不成立**（`d_W ≥ 2` 反例族） | 改写为**逐条件**的「非空窗口保证」（`d_W==1` 是不可缺条件），并**显式定义空窗口契约**：**对齐源**（`max_idx==BIG ∧ ¬has_nan ⇒ idx=0`），使「与源等价」在全部 manifest-legal 配置下为真 | 反例 `L_in=1,k=2,s=1,p=1,d=2`（窗口 lane `{−1,1}` 全落界外）；与 `algorithm-candidates.md` 窗口池化域条件①「`d==1` 必不可少」同源。**不会再犯**：新增「凡『参数约束 ⇒ 集合关系（非空/全覆盖/退化）』断言，必须逐条件推导真实充分条件并枚举反例」纪律 |
| A4 | N3 与 §0.6 R3 的 `first_valid` 消解论证无「非空窗口」限定；空窗口下 `max_idx` 输出 `BIG=L_in`（与源 `0` 不一致） | N3 与 R3 的等价前提补「**非空窗口**（≥1 in-bounds lane）」，并把空窗口子句写成 N3 的**第 4 个等价前提分支**；§3.3/§6.2 伪代码可见该子句（复用 `cfinal` 布尔缓冲，**不新增 UB**） | 空窗口子句 O(1)/单元（2 条 `(BL,CH)` 级向量 op）；`idx32==BIG` 检测与空窗口一一对应（非空窗口恒有 `max_idx<BIG`） |
| A5 | §8.3 无空窗口控制组 | §8.3 增补**负向控制组 L0-15**（`d_W≥2 ∧ p_W>0` 且存在空窗口的用例，断言 = 空窗口契约，**不断言与 torch 相等**——该 regime torch 索引越界，不可作 oracle）；`verify_equiv.py` 增 `EW` 项 + `empty_window_evidence()` | 对齐 `algorithm-candidates.md` 窗口池化域条件③「逐子句隔离负例」纪律 |
| A6 | §1.6.1 C1 收益栏「读次数降 `k_W` 倍」；sincnet halo 比 ≈1.06× | 改写为「访存削减 = 窗口重叠复用 + 连续 span 合并访存（非纯读次数）」；`k/s>1` 读次数比 `(num_oblk·LSPAN)/(L_out·k_W)`（ecg ≈0.48×），`k=s` 读次数不变而改访存模式；sincnet 比 ≈1.01× | 复算 `num_oblk·LSPAN/L_in`：sincnet `86·48/4096` ≈1.008、ecg `61·41/2048` ≈1.22 |
| A7 | §0.4 #4 / §1.6.0 R2 引文指向单输出工厂第 57 行 | 改引本算子行号：`elif val > max_val: max_val = val; max_idx = flat_idx`（第 270–272 行）、`if is_nan: nan_idx = flat_idx; has_nan = True`（第 261–265 行） | 语义结论不变，引文指向本算子代码 |
| A8 | §5.2 表 `test (1,4,97)` 行 `LSPAN=96`（笔误） | 改为 `LSPAN=97` | `min((32−1)·3+(5−1)·1+1, 97)=min(98,97)=97` |
| A9 | §4.5 逐 buffer 表合计 `129 720 / 134 800 B` 与逐项和不一致；自注「逐 buffer 略低（闭式更保守）」方向反 | 表合计改为逐项和（候选档 `BL=18/20` 修正合计 `129 280 / 132 800 B`；选定档 `BL=16/17` 合计 `114 944 / 113 792 B`）；自注改为「逐 buffer 求和 = 闭式值」 | 与 A1/A2 同步 |
| A10 | §2.2 把 `T.reduce_max` 列入所需算子（设计不使用） | 删除 `T.reduce_max`（最大值由 `T.vmax` 窗口累加实现） | 与 §3.4/§9.2 一致 |
| A11 | §1.6.2 未显式覆盖 persistent 调度循环与尾块标量 `real_c`/`BL_real` | 改写 #19 为「单元任务解码 + persistent 调度任务 id + 尾块边界（`real_c`/`BL_real`）」，理由类别 = block 级标量（非逐元素） | 使循环/标量清单与 §3.3/§6 一一对应 |
| A12 | §1.6.1 措辞「fp64 参照」「逐行原样粘贴」与脚本实际不符；耗时记 ≈38.5 s | 改为「torch CPU fp32 参照 + 目标 dtype 往返比对（0 ULP 判据）」「按 config 聚合后的输出（数值与计数已逐项核对）」；耗时更新为本次实测区间 | 与 `verify_equiv.py` 实际实现一致（非阻塞项） |

> **未变（用户补充约束 S1/S2/S3 全部保持）**：C4/C4b「仅数学等价性论证、**不采用**」；空窗口契约（A3）
> **只给索引 epilogue 加一条全 shape 统一子句**，**不引入任何几何特化分支**（§3/§5/§6 仍无 C4/C4b 分支）；
> toolchain stamp 与任务快照分列；`verify_equiv.py` 分支采样（本版仅新增空窗口分支，仍为少量代表组）。

---

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 方法论：`tilelang-op-design` skill 的 [references/migration-analysis.md](../../../.agents/skills/tilelang-op-design/references/migration-analysis.md)。
> 逐行读源文件（`max_pool1d.py` 全文 348 行、`common.py`、`ops/pool.py` 的 `_MaxPoolFwdOpBase` /
> `MaxPool1dIndicesFwdOp` / `_max_pool_roofline`），**未臆测语义**。
> 本次迁移范围 = **双输出变体** `_max_pool1d_with_indices_kernel`（manifest `MaxPool1dIndicesFwdOp`，
> `ref_api` = `torch.nn.functional.max_pool1d(..., return_indices=True)`）；单输出变体不在范围内。

### 0.1 源算子语义（做什么）

**数学语义**（`torch.nn.functional.max_pool1d(x, kernel_size, stride, padding, dilation, ceil_mode,
return_indices=True)`，**双输出**）：对每个 `ow ∈ [0, L_out)`，

$$
\text{out}[n,c,o_w]=\max_{0\le k_w<k_W}\ x_{\text{pad}}\big[n,c,\ o_w s_W-p_W+k_w d_W\big],\qquad
\text{indices}[n,c,o_w]=i_w^\star
$$

其中 $x_{\text{pad}}$ 在 $[0,L_{in})$ 之外取 $-\infty$（源实现以「越界 lane **跳过**」表达，
`max_pool1d.py` 第 257 行 `if iw >= 0 and iw < l_in:`，初值 `max_val = T.cast(float("-inf"), accum_dtype)`
第 250 行），$i_w^\star$ 是赢家元素沿 $L$ 轴的位置。选中元素 $i_w^\star$ 与值/索引的精确规则由源状态机给出
（`max_pool1d.py` 第 255–284 行）：

1. **索引编码 = 沿 L 轴的位置 `iw`（int64）**：源码 `flat_idx = T.cast(iw, "int64")`（第 259 行），
   末尾 `T.cast(...)` 落盘（`out_idx=[1,2]` 的第 2 路，第 280–284 行）。对 1D 而言「平面扁平下标」就是 `iw` 本身
   （`iw ∈ [0, L_in)`）。
2. **并列（tie）取首个命中（first hit）**：更新条件为严格大于
   `elif val > max_val: max_val = val; max_idx = flat_idx`（第 270–272 行）⇒ 相等不更新 ⇒ 先访问者胜出；
   窗口内**首个**非 NaN 元素走 `first_valid` 分支写入 `max_val`/`max_idx`（第 266–269 行）。
3. **窗口含 NaN 时：值取 NaN，索引取窗口内「最后命中的那个 NaN」**（last NaN visited）：
   源码对每个 NaN 执行 `nan_idx = flat_idx; has_nan = True`（第 261–264 行，注释原文
   "PyTorch records the last NaN visited in a pooling window"）→ 后访问的 NaN 覆盖前一个；
   末尾 `indices = has_nan ? nan_idx : max_idx`（第 280–284 行）。
4. **访问顺序 = `k_w` 递增**（`for kw in T.serial(kernel_w)`，第 255 行），即 `iw = o_w s_W - p_W + k_w d_W`
   **沿 `k_w` 严格递增**（`d_W>0`）。「首个命中」与「最后 NaN」均以该顺序为准。

**本设计的 torch 实测复核**（torch 2.9.0 CPU，2026-10-14，
`F.max_pool1d(x, k, s, p, d, ceil_mode, return_indices=True)`；脚本 `verify_equiv.py` 的
`semantics_evidence()` 逐行打印，命令：
`/workspace/venv/bin/python examples/max_pool1d_with_indices/_max_pool1d_with_indices_kernel/verify_equiv.py`）：

| 输入（窗口内取值） | 实测 `val` | 实测 `idx`（int64） | 结论 |
|---|---|---|---|
| `x=[3,1,4,1,5,9]`, k2s1p0 | `[3,4,4,5,9]` | `[0,2,2,4,5]` | 索引 = `iw` 沿 L 轴位置 ✓ |
| `x=[5,5,5,1]`, k3s1p0 | `[5,5]` | `[0,1]` | tie → first hit ✓ |
| `x=[9,1,9,1]`, k3s1p0（两个最大在 iw=0,2） | `[9,9]` | `[0,2]` | tie → first hit（前一个）✓ |
| `x=[nan,5,nan,2]`, k3s1p0（两个 NaN） | `[nan,nan]` | `[2,2]` | NaN → **last NaN**（不是首 NaN）✓ |
| `x=[1,nan,3,nan]`, k3s1p0 | `[nan,nan]` | `[1,3]` | NaN → last NaN（逐窗口）✓ |
| `x=[-inf,-inf,-inf,-inf]`, k3s1p1（前 padding，全 `-inf`） | `[-inf,-inf,-inf,-inf]` | `[0,0,1,2]` | 越界 lane **跳过**；索引 = 窗口内**首个有效元素**（含 `-inf`）✓ |
| `x=[2,7,-inf,-inf]`, k3s1p0 | `[7,7]` | `[1,1]` | `-inf` 是合法最大值，first hit ✓ |
| `x=[inf,-inf,1,2]`, k2s1p0 | `[inf,1,2]` | `[0,2,3]` | `{+inf,-inf}` → `+inf`（**非 NaN**）✓ |
| `x=[inf,nan,1]`, k2s1p0 | `[nan,nan]` | `[1,1]` | NaN 胜出，索引为 NaN 位置 ✓ |
| `x=[4,1,9,1]`, k2s2p0 **d2** | `[9]` | `[2]` | dilation>1 索引 = 实际命中位置 ✓ |
| `x=[1,8,2,3,7]`, k2s3p0 | `[8,7]` | `[1,4]` | `s≠k` 尾窗部分覆盖 ✓ |

> **注意（与几种常见 NPU reduce 形态的分野）**：device argmax / reduce 对 NaN 常以「哨兵索引」或
> 「首 NaN 索引」收场，本算子**两者都不是**——必须显式实现「值 NaN + **last NaN** 的 `iw`」，
> **不得**依赖 `T.vmax` / `T.reduce_max` 的 NaN 行为（未文档化），也不得套用哨兵语义。

- **输出尺寸规则**：`out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)`
  （源第 225 行）。helper 含 `ceil_mode` 两处修正（`effective_kernel = d*(k-1)+1`；ceil 分支向上取整；
  末尾「最后一窗起点落在 padding 内则回退一窗」）。**该公式不得在任何地方重算**（抽取件注释原文），
  本设计逐字引用抽取件的内联副本（`_max_pool1d_indices_fwd_kernels.py` 第 100–117 行）。
- **dtype 语义**：门禁 = `{float16, bfloat16, float32}`（源类构造第 192–195 行；集成包第 192–195 行；
  `ops/pool.py` 的 `_validate_pool_input_dtypes`）。源 kernel 内 `accum_dtype = "float"`（第 224 行）：
  逐元素 `T.cast(x[...], accum_dtype)` 参与比较，末尾 `T.cast(result, dtype)` 单次舍回。
  **对 max 而言 fp32 是「比较域」而非「累加域」**：输出即被选中的那个输入元素，`dtype→fp32→dtype` 无损
  ⇒ `out` 逐位等于窗口内某输入元素（或 NaN）——这正是精度标准 `atol=rtol=0` 的依据。
- **索引 dtype 语义**：**int64**（源第 234/250–259 行；`T.cast(iw, "int64")`），shape 同 output。
- **I/O 契约**：输入 `x`：`(N, C, L_in)` **恰好 3D NCL**（`ops/pool.py` 的 `_resolve_input` 要求
  `input.ndim == nd + 2`，nd=1 ⇒ 3D，**无 2D C×L 路径**），连续布局，尾轴 `L_in` 连续；
  输出两路同形 `(N, C, L_out)`，`out` 同输入 dtype、`indices` **int64**，均为原生 NCL
  （**不是转置布局**），`out_idx=[1,2]`。
- **边界语义（逐条件，v1 修订）**：
  - **非空窗口保证**：`d_W == 1 ∧ 0 ≤ p_W ≤ ⌊k_W/2⌋ ∧ s_W > 0` ⇒ 每个输出窗口至少含 1 个 in-bounds 元素。
    **`d_W ≥ 2` 时该保证不成立**——窗口的输入覆盖集合是等差子集 `{o_w·s_W − p_W + k_w·d_W}`，须逐窗口判空
    （最小反例：`L_in=1, k=2, s=1, p=1, d=2` → 窗口 lane 集合 `{−1, 1}` 全落 `[0,L_in)` 外）。
    与 `algorithm-candidates.md` 窗口池化域条件①「`d==1` 必不可少」同源。
  - **空窗口契约（v1 显式定义，对齐源）**：当某窗口**无任何 in-bounds lane** 时，源状态机输出 `(val, idx) = (−∞, 0)`
    （`max_pool1d.py` 第 250 行 `max_val=−inf`、第 252 行 `max_idx=0` 初值，第 257 行越界跳过）。
    本设计**对齐源**：`max_idx` 保持哨兵 `BIG=L_in` 且无 NaN 时，epilogue 追加子句 `max_idx==BIG ∧ nan_idx<0 ⇒ idx=0`，
    故「与源等价」在**全部 manifest-legal 配置**下为真（含空窗口 regime）。契约可见性见 §3.3/§6.2 伪代码与 §9.3。
    **注（trade-off）**：该 regime 下 torch `F.max_pool1d(..., return_indices=True)` 返回**最后一条候选 lane 的 `iw`**
    （可越出 `[0,L_in)`，属其 CPU kernel 的 loop-artifact），与源/本设计**三方互异**；该 regime 为产线外
    （值为 `−∞`、索引越界），无任何 workload / bench / L0 用例覆盖，故 **torch 在此不可作 oracle**
    （§8.3 L0-15 断言契约而非 torch 相等；`verify_equiv.py` 的 `empty_window_evidence()` 逐行留证）。
  - `{+inf,-inf}` 按 IEEE 最大值取 `+inf`；NaN 按「值胜出 + last NaN 索引」。

**语义保持基线**：§8.1 golden 以本节语义为唯一依据实现（直接移植源仓测试基准
`F.max_pool1d(..., return_indices=True)`），**不复刻** §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x`（input） | `(N, C, L_in)`（**恰好 3D**，无 C×L 路径） | float16 \| bfloat16 \| float32 | NCL 连续；尾轴 `L_in` 连续 |
| `out`（output） | `(N, C, L_out)` | same_as_input | NCL 连续；`out_idx=[1,2]` 的第 1 路 |
| `indices`（output） | `(N, C, L_out)` | **int64** | NCL 连续；`out_idx=[1,2]` 的第 2 路 |

**输出 shape（迁移必需字段）**：`(N, C, L_out)` **两路同形**，
`L_out = pool_output_dim(L_in, k_W, s_W, p_W, ceil_mode, d_W)`。两路输出均为原生 NCL 布局
（**不是转置布局**）；`indices` 与 `out` 的 shape 完全一致，仅 dtype 不同。

### 0.3 实现算法解读（怎么算）

源码结构化阅读（`max_pool1d.py` 第 212–288 行，逐行）：

**计算步骤分解**（源码每个计算语句均归入某步骤）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1（host，trace 期） | `out_l = pool_output_dim(...)`；`total_output = n*c_in*out_l`（第 225–226 行） | 编译期 shape 常量 | Python int | 输出长度 + `ceil_mode` 修正 |
| 2（grid） | `T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx`（第 236 行） | `block_m`、`threads` | 一维 block 网格 | 输出元素级并行划分（**两路输出共用同一网格**） |
| 3（lane 扁平索引） | `out_idx = bx*block_m + i`；`ow = out_idx % out_l`；`channel_batch_idx = out_idx // out_l`；`c_idx = channel_batch_idx % c_in`；`batch = channel_batch_idx // c_in`；经 `if out_idx < total_output` 守卫（第 237–243 行） | `out_idx` | 标量 `(batch,c_idx,ow)` | 输出坐标 $(n,c,o_w)$ |
| 4（状态初值） | `max_val = T.cast(-inf, "float")`；`has_nan = False`；`max_idx = T.cast(0,"int64")`；`nan_idx = T.cast(0,"int64")`；`first_valid = True`（第 245–254 行） | — | 标量 | 最大值初值 + NaN 标记 + 两个 int64 索引累加器 + 首有效标记 |
| 5（窗口扫描：值 + 索引 + NaN 三路更新） | `for kw in T.serial(kernel_w): iw = ow*stride_w - pad_w + kw*dilation_w; if 0<=iw<l_in:` 内 `val=cast(x[...],"float")`；`flat_idx=cast(iw,"int64")`；`is_nan=T.isnan(val)`；`if is_nan: nan_idx=flat_idx; has_nan=True` / `elif first_valid: max_val=val; max_idx=flat_idx; first_valid=False` / `elif val>max_val: max_val=val; max_idx=flat_idx`（第 255–272 行） | `x` 的逐元素 GM 读 | 标量 fp32 / bool / int64×2 | $\max_{k_w}$ + **first-hit 索引** + **last-NaN 索引**（越界 lane 跳过 ≡ $-\infty$ padding） |
| 6（结果 + 双写出） | `result = T.if_then_else(has_nan, cast(nan,"float"), max_val)`；`out[...] = cast(result, dtype)`；`indices[...] = T.if_then_else(has_nan, nan_idx, max_idx)`（第 274–284 行） | 标量 | 标量 + 两个 GM 写 | NaN 判决 + 值/索引两路写出 |

**数据流与内存访问模式（源硬件视角）**：

```
GM[x] --逐元素标量读 x[batch,c_idx,iw]--> register(max_val fp32, has_nan bool,
                                                  max_idx int64, nan_idx int64, first_valid bool)
        --NaN 判决 + 单次 cast--> GM[out]
        --has_nan?nan_idx:max_idx (int64)--> GM[indices]
（无 SMEM/L1 中转；每个输出元素现场从 GM gather 其窗口的全部元素；相邻输出窗口重叠时重复读；
  两路输出共用同一次窗口扫描——索引「免费」获得，不产生第二次 GM 读）
```

**循环与并行结构**：
- grid = `ceildiv(total_output, block_m)` 个 block，每 block `threads` 个 CUDA 线程；
- `T.Parallel(block_m)`：每个线程（lane）负责**一个输出元素**，lane 内全为**标量**计算；
- lane 内循环：`T.serial(kernel_w)` **静态边界**（`k_W` 是 trace 期常量）+ 运行时 `if` 守卫跳过越界 lane；
- **lane 内无归约原语、无向量化**（每 lane 独立标量状态机）；
- **索引与值在同一遍扫描内同时产生**（关键结构事实，§0.4 #13）；
- 无流水线（无 `T.Pipelined`）、无共享内存、无 SIMD。

**host 侧逻辑**（`ops/pool.py` 的 `_MaxPoolFwdOpBase` / `MaxPool1dIndicesFwdOp`）：
- 参数规范化（`kernel_size`/`stride`/`padding`/`dilation` 的 int↔1-tuple 展开、`None` stride → kernel）
  与校验（`padding ≤ k//2`、`stride>0`、`dilation>0`、3D NCL、dtype 门禁）在 op 层完成；
- `x = x.contiguous()`（无实际拷贝时 no-op）；`_kernel_cache` 按 `(n,c,l_in,dtype,device)` 缓存 kernel 实例；
- `functools.lru_cache(maxsize=32)` 缓存 kernel factory（按 shape/dtype）；
- `_max_pool_roofline(op, indices=True)`（仅 manifest 性能规格，不参与计算；`bytes_` 额外加 `N*C*out_l*8`）；
- 编译配置：`@tilelang.jit(out_idx=[1,2], compile_flags=["-O3","-DENABLE_BF16"])`（源第 228 行）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | **一维 block 网格 + 输出元素扁平映射** | 把 `(N,C,L_out)` 三维问题压成一维可并行任务 | `T.ceildiv(total_output,block_m)` + `%`/`//` 反解 | GPU block 网格（非 hardware-specific） | **等价替换**（一维 `T.Kernel`） |
| 2 | **`T.Parallel(block_m)` 线程级 lane 并行** | 每线程一个输出元素、隐藏逐元素 gather 延迟 | CUDA thread/warp 两级并行模型 | **CUDA threads / warp**（NPU 无等价） | **重新设计** → §0.6 R1 |
| 3 | **静态边界窗口循环 + 运行时越界守卫** | `k_W` 为 trace 期常量、循环上界静态；守卫跳过 padding lane | `T.serial(kernel_w)` + `if iw>=0 and iw<l_in` | 无（语言/编译器级约束） | **保留** |
| 4 | **标量 max 比较链 + `-inf` 初值** | 单线程顺序比较、寄存器驻留；`-inf` 是 `max` 单位元 | 本算子：`elif val > max_val: max_val = val; max_idx = flat_idx`（`max_pool1d.py` 第 270–272 行）+ `first_valid` 首次 seed（第 266–269 行），初值 `max_val = T.cast(float("-inf"), accum_dtype)`（第 250 行），均在 `T.alloc_var` 上（第 245–249 行） | **GPU 大寄存器堆 + 寄存器比较链** | **重新设计** → §0.6 R2 |
| 5 | **first-hit 严格大于规则（`first_valid` + `val>max_val`）** | 复刻 PyTorch tie 语义 | 严格 `>` ⇒ 相等不更新 ⇒ 先访问者胜出；首个有效元素走 `first_valid` | 无（纯数值语义） | **保留**（语义层，需向量化） |
| 6 | **last-NaN 索引覆盖** | 复刻 PyTorch NaN 索引语义 | 每个 NaN 元素 `nan_idx = flat_idx`（后 NaN 覆盖先 NaN）；末尾 `has_nan ? nan_idx : max_idx` | 无（纯数值语义） | **保留**（语义层，需向量化） |
| 7 | **标量 NaN 追踪 + 值覆盖** | 复刻「NaN 胜出」，不依赖 `T.max` 的 NaN 行为 | `has_nan = has_nan \| T.isnan(val)`；末尾 `has_nan ? NaN : max_val` | 无（纯数值语义） | **保留**（语义层，需向量化） |
| 8 | **int64 扁平索引承载** | 与 PyTorch 索引 dtype 契约一致 | `flat_idx = cast(iw, "int64")` | 无（位宽选择） | **保留**（数值/位宽层） |
| 9 | **双输出同轮融合** | 索引不额外付 GM 读代价 | 值与索引在同一窗口扫描内同时产生；仅多一次尾部 select + 一次 GM 写 | 无（融合结构） | **保留/等价替换**（NPU 沿用单遍融合） |
| 10 | **fp32 比较域 + 单次输出舍入** | 与 PyTorch 比较语义一致；max 下无损 | `accum_dtype="float"`，逐元素 cast，末尾一次 cast 回 | 无（纯数值语义） | **保留**（语义层） |
| 11 | **逐元素 GM gather（无 tile 搬运 / 无 coalescing）** | 接口简单，按窗口坐标直接取值 | 直接索引 `x[batch,c_idx,iw]` | GPU 的 L1/L2 缓存吸收重复访问 | **重新设计** → §0.6 R4 |
| 12 | **`T.alloc_var` 标量寄存器状态** | lane 内状态（`max_val`/`max_idx`/`nan_idx`/`has_nan`/`first_valid`）寄存器驻留 | `T.alloc_var` + 顺序更新 | GPU 寄存器 + 单线程执行模型 | **重新设计** → §0.6 R1/R2/R3 |
| 13 | **`out_idx < total_output` 尾任务守卫** | 处理 `total_output` 非 `block_m` 整数倍 | 运行时谓词跳过越界 lane | 无 | **保留**（模式可移植） |
| 14 | **`functools.lru_cache` factory 缓存 + op 层 kernel 实例缓存** | 避免重复 trace/编译 | host 侧缓存 | 无 | **保留**（host 侧） |

> 识别不出来 ≠ 不存在。以上 14 项为源码中可见的全部优化/结构性选择（含 indices 特有的 #5/#6/#8/#9）；
> 相对单输出变体**新增** #5 first-hit、#6 last-NaN、#8 int64 索引、#9 双输出融合。
> 源码**未**使用 warp shuffle / SMEM swizzle / mma / 三维 Kernel / cp.async / `T.Pipelined`，故无「舍弃」类 GPU 专有手段。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定问题：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义（窗口取最大值；越界跳过 ≡ padding $-\infty$；`ceil_mode` 输出尺寸规则） | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.1）；已由 §1.6.1 E1 机器验证 |
| **NaN 胜出 + ±inf 最大值语义（#7）** | 语义 | 无 | — | **保留** | 数值语义；NPU 侧用 IEEE 正确的 `eq` 掩码乘性累积实现（§0.6 R2；`ne` 为 ordered-NE，NaN 恒假，不可用作 NaN 探测） |
| **first-hit tie 语义（#5）** | 语义 | 无 | — | **保留** | 等价改写为「值等于窗口最终最大值的最小 `iw`」（§0.6 R3 + §1.6.1 N3），逐位一致 |
| **last-NaN 索引语义（#6）** | 语义 | 无 | — | **保留** | 改写为「NaN 位置上 `iw` 的**最大值**」（访问序 `iw` 严格递增 ⇒ last = max），用 `vmax` 累计（§0.6 R3 + §1.6.1 N2） |
| **索引编码 = `iw`（#8 的语义面）** | 语义 | 无 | — | **保留** | 契约（int64、shape 同 output、`iw ∈ [0,L_in)`）；NPU 侧同样构造该值 |
| int64 索引承载（#8） | 数值/位宽 | 无 | 有 | **保留** | NPU 侧中间以 **int32** 承载（`T.vmin`/`T.vmax`/`T.vcmp`/`T.vselect`/`T.vmul`/`T.transpose` 均支持 int32；`T.vcast` i32→i64 `rint` ✓），落盘 cast int64。工厂期守卫 `l_in < 2**31` |
| **双输出同轮融合（#9）** | 优化 | 无 | 有 | **保留** | 单遍扫描内同时产出值与索引（同一 UB tile），不产生第二次 GM 读 |
| I/O 契约（3D NCL→NCL 双输出、三 dtype、值同 dtype、索引 int64） | 语义 | 无 | — | **保留** | layout 是契约 |
| fp32 比较域 + 单次输出舍入（#10） | 语义 | 无 | — | **保留** | NPU Vector 支持 fp32 `T.vmax`/`T.vcast`（`docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1 fp32 ✓） |
| 边界域：`d_W==1` 下窗口非空保证、`d_W≥2` 逐窗口判空、**空窗口契约（对齐源 `idx=0`）**、`ceil_mode` 末窗回退 | 语义 | 无 | — | **保留** | 语义层；空窗口契约见 §0.1（v1 修订）与 §0.6 R3 第 4 等价分支 |
| 静态边界窗口循环 + 守卫（#3） | 算法 | 无（语言约束） | 有（同款约束） | **保留** | NPU `T.serial` 同样只接受静态边界；本算子 `k_W` 是**编译期常量** |
| `-inf` 初值 + 越界跳过 | 算法 | 无 | 有 | **保留** | `T.vbrc(-T.infinity("float32"), acc)` 整 buffer 广播 |
| 尾任务守卫（#13） | 算法 | 无 | 有 | **保留** | `if uid < num_logical` 运行时谓词 |
| lru_cache / kernel 缓存（#14） | host | 无 | 有 | **保留** | host 缓存 |
| 一维 block 网格 + 输出扁平映射（#1） | 优化 | GPU block 网格 | 有（一维 `T.Kernel`） | **等价替换** | migration-analysis.md §5.3；`docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3 |
| **`T.Parallel(block_m)` 线程级 lane 并行（#2）** | 优化 | CUDA threads / warp | **无** | **重新设计** → §0.6 R1 | NPU 无 CUDA thread 概念；每 lane 一个输出元素退化为**标量**（pattern-library `layout.md` 的 `PL-1.2-caxis-accum` 同族实测窗口累加 98.3% 标量） |
| **标量 max 比较链（#4）** | 优化 | GPU 寄存器堆 | **无直接等价** | **重新设计** → §0.6 R2 | migration-analysis.md §5.3「register 累加 → UB 中间缓冲 / Vector 归约」；NPU 侧改为 C 轴向量 `T.vmax` |
| **标量 first-hit / last-NaN 状态机（#5/#6/#12）** | 优化 | GPU 单线程执行模型 | **无直接等价** | **重新设计** → §0.6 R3 | 状态改为 UB 向量累加器（int32 `vmin`/`vmax`），lane 映射到 C 轴 |
| **逐元素 GM gather（#11）** | 优化 | GPU L1/L2 缓存 | **无等价（且有害）** | **重新设计** → §0.6 R4 | NPU 逐元素 gather 无 coalescing、无 cache 吸收；改为 UB tile 搬运 + 向量化 |
| 退化特化（源无；同族模式集） | 优化 | 无 | 有 | **重新设计** → §0.6 R5 | 源对所有长度一律走通用窗口循环；NPU 侧评估退化特化，**本设计判定不采用**（`TRAP-npuir-arange-pattern-with-whole-plane-reduce`；见 §0.6 R5） |

> 每项处置依据：migration-analysis.md §5.3 映射表条目 / `docs/Tilelang.language/` 条目 / pattern-library 实测条目（见「依据」列）。
> **舍弃项：无。** 源码未使用 warp shuffle / SMEM swizzle / mma / 三维 Kernel / cp.async 等 GPU 专有手段，故无舍弃条目。

### 0.6 NPU 算法重设计

对 §0.5 判定为「重新设计」的条目逐项给出 NPU 新算法（每项含「源方案 → NPU 新算法 → 语义保持论证」）。
**重设计输入含 §1.6.0 算法调研结论**：调研选定「直接窗口取最大值 + 显式块级向量 eq 掩码 NaN 追踪 +
**C 轴向量化索引追踪（int32 累加器）** + fp32 比较域 + 核内融合转置 + L 轴连续 span 载入分块 +
在线单遍」算法族（与源算法同族，但归约/索引/追踪单元由标量改为 C 向量），故 §0.6 直接承接调研结论；
源算法已识别优化手段（§0.4）的优化意图承接见每项「承接」行。

**重设计项 R1：并行结构重设计（线程级 lane 并行 → 一维 Kernel + `(n, o_blk, c_block)` 单元 + C 轴向量 lane）**

- **源方案**：`T.Kernel(ceildiv(total_output,block_m), threads=threads)` + `T.Parallel(block_m)`，
  每 CUDA 线程处理一个输出元素，lane 内 `T.alloc_var` 标量状态。
  性能意图：把 `N*C*L_out` 个输出元素摊到大量线程上并行，靠 warp/线程切换隐藏逐元素 GM gather 延迟。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True)`；**逻辑任务单元 = `(n, o_blk, c_block)`**，
  其中 `c_block` 是长度 `CH` 的**通道向量块**、`o_blk` 是长度 `BL` 的**输出位置块**；
  `num_logical = N * num_oblk * ceil(C/CH)`（**全部 shape 走一般网格**——C4/C4b 未采用，见 R5）。
  单元内以 **C 轴为向量维**；`o_blk` 由单元 id 运行时解码；`k_w` 与块内输出位置 `owl` 为 `T.serial` 内层循环
  （`k_W`、`BL` 均为编译期常量），循环体为 C 向量 op。`num_logical > vector_cores` 时按 persistent
  grid-stride 模式挂 `T.serial` 核内循环（静态界，见 §5.5）。
- **为何必须沿 L 分块**（1D 相对 2D 的关键差异，沿用同族前作结论）：2D 的 `W_in`（≤112）可整行驻留 UB；
  1D 的 `L_in` 可达 4096（`sincnet-speaker-local`），`(CH, L_in)` 整线驻留远超 UB ⇒ 单元必须同时沿输出轴
  （`o_blk`）分块，且载入区为一个**连续的输入 span**（`LSPAN` 长度）。
- **语义保持论证**：任务划分只是把互不相关的输出元素分给不同单元，**不改变任何窗口的取值集合、比较结果与访问顺序**
  （每个输出仍对其窗口取最大值与索引，越界 lane 仍按 $-\infty$ 处理）；无跨单元归约，无边界语义变化。
  承接：源 #1 的一维扁平映射被保留为「一维 Kernel」，#2 的 lane 并行意图由「单元并行 + C 轴向量并行」承接。

**重设计项 R2：值归约重设计（per-lane 标量比较链 + 标量 NaN 标记 → UB 内 C 轴向量取最大值 + 向量 NaN 掩码）**

- **源方案**：每输出元素在寄存器里做 `max_val = T.max(max_val, T.cast(x[...],"float"))` 标量比较链，
  并行 `has_nan = has_nan | T.isnan(val)` 标量 bool 累积，初值 $-\infty$/`False`。
- **NPU 新算法**：C 轴向量化（向量轴 = C，§1.6.3）：
  1. 累加器 `acc`（`BL × CH` fp32）经 `T.vbrc(-T.infinity("float32"), acc)` 置 $-\infty$（**单位元是 $-\infty$，非 0**）；
     NaN 追踪器 `flag`（`BL × CH` fp32）经 `T.vbrc(T.float32(1.0), flag)` 置 1。
  2. 逐元素 cast 改为 **tile 级向量 cast**：`T.vcast(row_dst, row_f32, round_mode="rint")` 一次把整块
     `(LSPAN, CH)` 提到 fp32（**fp32 输入时改用 `T.copy`**——同 dtype `vcast` 会退化为 round-to-integer 误编译，见 §9.2）。
  3. 取最大值改为 **C 轴向量最大值**：`T.vmax(acc[owl:owl+1,:], row_f32[j:j+1,:], acc[owl:owl+1,:])`（原地），
     其中 `j = iw - s0`（相对缓冲偏移，见 §1.6.1 C1）。
  4. **NaN 追踪向量化为「乘性 eq 掩码」**（max 特有，语义关键）：**每单元一次**
     `T.vcmp(row_f32, row_f32, eq_mask, "eq")`（$x{=}x$ 为真、NaN 为假）→ `T.vcast(eq_mask, eq_f32, "rint")`；
     窗口内逐 (kw, owl) `T.vmul(flag[owl:owl+1,:], eq_f32[j:j+1,:], flag[owl:owl+1,:])`（`flag` 初值 1：0 吸收、1 中性）。
     **不依赖 `vmax` 的 NaN 行为**（该行为未文档化；`ne` 为 ordered-NE 亦不可用，见 §9.2）。
  5. 输出：`cfinal = T.vcmp(flag, zeros, "eq")`（`flag==0` ⟺ 窗口含 NaN）→
     `T.vselect(cfinal, nanbuf, acc, out32)`（`nanbuf = inf - inf` = NaN）→
     `T.vcast(out32, acc16, round_mode="rint")`（单次舍回 dtype；fp32 输入改用 `T.copy`）→
     `T.transpose(acc16, acc_t, [1,0])` → `T.copy` 写回。
- **语义保持论证**：
  - **数学等价**：窗口扫描不改变取最大值的元素集合；C 轴是**独立轴**（每通道独立取最大值，§0.1），
    把 `c` 放进向量 lane 只是 SIMD 数据并行，**不改变任何 `c` 的取值集合**。
  - **NaN/±inf 语义核对**（逐条对齐 §0.1）：`{+inf,-inf}` 无 NaN → `eq_mask` 恒真 → `flag` 保持 1 →
    `cfinal` 恒假 → 输出 `acc`，而 `T.vmax(-inf,+inf)=+inf` ✓；窗口含 NaN → 对应 lane 的掩码为 0 →
    `flag` 变 0（吸收）→ `cfinal` 真 → 输出 NaN ✓。
  - **dtype/舍入**：与源一致（fp32 比较域、末尾单次 `rint`）；`vcast` fp16/bf16→fp32 为无损上转。
  - 承接：源 #4 的「寄存器比较链」意图由「UB fp32 向量最大值累加器」承接；#7 的「标量 bool NaN 追踪」
    由「C 向量 eq 掩码乘性累积」承接；#10 由「fp32 向量比较域 + 单次 `rint`」承接。

**重设计项 R3：索引追踪重设计（per-lane 标量 `max_idx`/`nan_idx` 状态机 + `first_valid` → C 轴向量化 int32 累加器）**

- **源方案**：每输出元素在寄存器里维护 `max_idx`（first-hit，严格 `>` 更新；首个有效元素走 `first_valid`）与
  `nan_idx`（每个 NaN 覆盖），末尾 `has_nan ? nan_idx : max_idx`。
  性能意图：单线程顺序状态机、寄存器驻留、**与值归约同轮完成**（#9 融合意图）。
- **NPU 新算法**（C 轴向量，每个 lane = 一个通道；两个索引累加器均 int32）：
  1. `max_idx`（`BL × CH` int32）初值 = `BIG`（= `L_in` 哨兵，经整 buffer 广播）；`nan_idx`（`BL × CH` int32）
     初值 = `-1`（哨兵：`-1` 表示「该 lane 窗口内尚未见 NaN」）。
  2. **first-hit（两遍法的第二遍；`acc` 已定稿）**：逐 (kw, owl)，若 `0<=iw<L_in`：
     - `T.vcmp(row_f32[j:j+1,:], acc[owl:owl+1,:], eqb, "eq")`（值等于窗口最大值；NaN 与任何值都不等 ⇒ 恒假）；
     - `T.vselect(eqb, fv, bigv, cand)`（命中取 `iw` 向量 `fv`，否则取哨兵 `BIG`）；
     - `T.vmin(max_idx[owl:owl+1,:], cand, max_idx[owl:owl+1,:])`（原地 `vmin`）。
     末尾 `max_idx` = 「窗口内值等于 `acc` 的**最小** `iw`」。
  3. **last-NaN（与值同轮）**：逐 (kw, owl)，若 `0<=iw<L_in`：
     - `T.vcmp(row_f32[j:j+1,:], row_f32[j:j+1,:], eq2b, "eq")`（非 NaN 为真）；`T.vselect(eq2b, neg1v, fv, nanc)`
       （非 NaN 取 `-1`、NaN 取 `iw`）；`T.vmax(nan_idx[owl:owl+1,:], nanc, nan_idx[owl:owl+1,:])`（原地 `vmax`）。
  4. 单元末尾：`cfinal = T.vcmp(nan_idx, zero_i32, "ge")`（`nan_idx >= 0` ⟺ 窗口含 NaN）→
     `idx32 = T.vselect(cfinal, nan_idx, max_idx, idx32)`（`has_nan ? nan_idx : max_idx`，与源第 280–284 行同式）。
4b. **空窗口子句（v1 新增，对齐源）**：`T.vcmp(idx32, bigv, cfinal, "eq")`（`idx32==BIG` ⟺ 无 NaN 且
     `max_idx` 未命中 ⟺ 窗口无 in-bounds lane；非空窗口恒有 `max_idx < BIG`）→
     `T.vselect(cfinal, zero_i32, idx32, idx32)`（空窗口 `idx = 0`，与源 `max_idx` 初值同值）。
     **复用 `cfinal` 布尔缓冲**（须排在 `out32` 计算之后），**不新增 UB**；成本 O(1)/单元（2 条 `(BL,CH)` 级向量 op）。
  5. 输出：`T.transpose(idx32, idx_t, [1,0])`（int32 二轴交换）→ `T.vcast(idx_t, idx64, round_mode="rint")`
     （i32→i64）→ `T.copy` 写回 `indices`（行片 `(real_c, BL_real)`）。
- **语义保持论证**：
  - **first-hit 等价（非空窗口）**：源用严格 `>` 在**递增 `iw` 序**上更新 `max_idx`，记录的是「值首次达到窗口最终最大值」的元素；
    该元素 = **窗口内值等于 `acc` 的最小 `iw`**（`verify_equiv.py` N2N3 组机器验证）。并列时值相等但 `vmin` 取更小 `iw`
    ⇒ 先访问者胜出 ✓（与 §0.1 实测「tie → first」一致）。**`acc` 必须已定稿**（本设计用两遍法：第一遍只算 `acc`/`flag`/`nan_idx`，
    第二遍算 first-hit）⇒ 无「与更新前/后累加器比较」的次序陷阱。**前提限定：非空窗口**（≥1 in-bounds lane）。
  - **last-NaN 等价**：源对每个 NaN 覆盖 `nan_idx`；访问序 `iw` 严格递增 ⇒ 最后一次覆盖 = NaN 位置集合中 `iw` **最大**者。
    向量式用 `vmax` 累计 `isnan ? iw : -1`，结果同 ✓（与 §0.1 实测「两个 NaN → idx 2」一致）。
  - **`first_valid` 被消解（非空窗口）**：源用 `first_valid` 处理「窗口内无破纪录元素」（如全 `-inf`）时 `max_idx` 应取首个有效元素；
    向量式 `min{iw : x[iw]==acc}` 在全 `-inf` 窗口下 acc=-inf、所有有效元素都等于 acc ⇒ 取最小有效 `iw` = 首个有效元素 ✓
    （§0.1 实测行 `x=[-inf,-inf,-inf,-inf]` k3s1p1 → `[0,0,1,2]`；`verify_equiv.py` 的 `P1`/`all_neg_inf` 用例机器验证）。
    同样限定「非空窗口」。
  - **空窗口分支等价（v1 新增，N3 第 4 分支）**：窗口无 in-bounds lane 时源不更新 `max_idx`（保持初值 0）且 `has_nan=False`，
    输出 `(val, idx)=(−∞, 0)`；向量式 `acc=−inf`、`max_idx=BIG`、`flag=1` ⇒ 空窗口子句把 `idx` 置 0，与源同值 ✓
    （`verify_equiv.py` 的 `EW` 项 3 配置 × 3 dtype × 3 数据全 `EQUIV_PASS`；`empty_window_evidence()` 逐行留证
    source/design `idx=0`、torch `idx∈{1,2}` 越界）。
  - **NaN 窗口下 `max_idx` 无关性**：源在 `has_nan` 时输出 `nan_idx`，`max_idx` 不影响结果；向量式同样由 `cfinal` 取舍 ⇒ 一致。
    （空窗口 ⟹ 无 in-bounds lane ⟹ 无 NaN，故空窗口子句与 NaN 分支互斥。）
  - **int32 位宽**：工厂期守卫 `l_in < 2**31`（1D 索引上界），落盘 cast 到 int64 与源同。
  - 承接：源 #5（first-hit）、#6（last-NaN）、#8（int64 索引）、#9（双输出融合）分别由
    「`eq`+`vmin` first-hit 累加器」「`isnan`+`vmax` last-NaN 累加器」「int32 累加器 + i32→i64 vcast」「单遍扫描内同轮产出」承接，
    空窗口子句补全「与源等价」的边界域，**无静默丢弃**。

**重设计项 R4：数据搬运重设计（逐元素 GM gather → UB 连续 span 载入 + 核内融合转置）**

- **源方案**：直接在 GM 上按 `x[batch,c_idx,iw]` 逐元素取值。性能意图：接口简单；GPU 靠 L1/L2 缓存吸收重叠窗口的重复访问。
- **NPU 新算法**：按 C 轴向量化所需布局（C 最内连续）搬运，且沿 L 轴做**连续 span 载入 + 钳制**：
  1. 单元解码后求载入范围（**纯标量，每单元 2 次钳制**）：`base = ow0*sW - pW`、
     `span = (BL-1)*sW + (k_W-1)*d_W + 1`、`LSPAN = min(span, L_in)`、`s0 = max(0, min(base, L_in - LSPAN))`；
  2. GM→UB：`T.copy(x[nb, cb*CH:cb*CH+real_c, s0:s0+LSPAN], row_src[0:real_c, 0:LSPAN])`
     —— **源起始 `s0` 为运行时值但恒在界内、长度为静态 `LSPAN`**，避免动态偏移 subview（见 §9.2 陷阱）；
  3. UB 内**融合转置**：`T.transpose(row_src, row_dst, [1,0])` 把 `(CH, LSPAN)` 变成 `(LSPAN, CH)`（C 最内）；
  4. 转置后 C 轴连续 → `T.vcast` 到 fp32 → 向量取最大值 / eq 掩码 / 索引追踪；
  5. 输出侧：值 `(BL,CH)` → `vcast` → `acc16` → `T.transpose([1,0])` → `(CH,BL)` → `T.copy` 回 GM；
     索引 `(BL,CH)` int32 → `T.transpose([1,0])` → `vcast` i64 → `T.copy` 回 GM。
- **语义保持论证**：搬运/转置是**纯数据重排**（双射），不改变元素值与窗口取值集合；输出写回后与 NCL 契约一致。
  无精度影响（不引入任何算术）。载入钳制只影响**载入的地址范围**，读出时仍按 `0 ≤ iw < L_in` 逐 lane 守卫
  （§1.6.1 C1 的等价改写保证两者一致，`verify_equiv.py` C1 组机器验证）。
  承接：源 #11「靠缓存吸收重复访问」的意图由「UB 连续 span 常驻 + 同一输入元素被其归属的所有 (kw, owl) 复用」承接。

**重设计项 R5：结构特化评估（整线归约 C4 / 恒等 C4b）——判定「不采用」，仅保留数学等价性论证**

> **这一节是对 USER SUPPLEMENT S1 的显式回答，也是本设计与同族单输出前作的关键分歧。**

- **源方案**：对所有输出长度一律走通用窗口循环（含 `L_out==1` 的全局池化、`k=1` 的恒等情形）。
- **候选特化（数学上成立，但只作论证）**：
  - **C4（整线归约，值）**：`L_out==1 ∧ p_W==0 ∧ d_W==1 ∧ (k_W-1)*d_W+1 >= L_in` 时窗口恰为 $[0,L_{in})$，
    值 = 整线最大值；NaN 判决由独立 `eq` 掩码归约（`reduce_min==0`）给出。**`d_W==1` 必不可少**
    （窗口输入覆盖集合是 $\{k_w d_W\}\cap[0,L_{in})$，仅 $d_W=1$ 时等于整线；反例 `L_in=5,k=2,s=1,p=0,d=4`：窗口仅 `{0,4}`，
    整线归约 = 1.0 而正确值 = 0.0）。
  - **C4b（恒等）**：`k_W==1 ∧ d_W==1 ∧ p_W==0 ∧ s_W==1` 时窗口恰为单元素 ⇒ `out = x`。
- **判定：C4/C4b 仅为「数学等价性论证（math-equivalence argument）」，本设计不采用（NOT adopted）。**
  - **理由（值半成立、索引半无法证明独立于已知断裂链）**：值半（整线 `reduce_max` / 恒等拷贝）数学等价已在
    `verify_equiv.py` 的 `specialisation_math_only()` 以 **value-only** 形式机器确认（3 dtype 全 `EQUIV_PASS`）。
    但**双输出变体**的 C4 还需要构造 first-hit / last-NaN 的**整线 argmax 索引**，其自然形态是由
    `T.arange` 派生的**线内下标 pattern** 在 `vselect + reduce`（`reduce_min`/`reduce_max`）链内消费——
    这正是本工具链**已验证断裂**的链路：`TRAP-npuir-arange-pattern-with-whole-plane-reduce`
    （`arange` 派生下标 pattern standalone 精确、in-chain 被读回损坏；同族 `AdaptiveMaxPool2dIndicesFwdOp`
    的整平面 argmax 特化因此 values 全对而索引大面积错，仅因 int64 索引 `torch.equal` oracle 才被发现，
    最终处置 = 交回一般路径）。
  - **本设计无法证明索引构造不依赖该断裂链**（「由派生 pattern 供 `vselect`/`reduce` 消费」就是该链本身；
    1D 下线内下标仍是同一 pattern 形态），故按 S1「无法证明 ⇒ 不采用，走一般路径」执行——
    与同族前作的最终处置一致。**全部 shape（含 manifest `textcnn-global`，`L_out==1`）走一般网格。**
  - **不采用的代价与收益核对**：一般路径在 `L_out==1` 时 `BL=1`、`LSPAN=min(k_W, L_in)`，其正确性与
    一般 shape 完全同构（同一 code path，无分支分叉）；放弃的只是「整线单次归约」的纯性能收益
    （§1.6.3 评估：`textcnn-global` 一般路径流量下界 3.46 µs，仍为流量主导、非瓶颈），**语义零风险**。
- 承接：源「通用窗口循环」保持；**不存在优化意图丢弃**（源本就无特化）；亦**不引入**任何结构性分支
  ⇒ §3/§5/§6 无 C4/C4b 分支（与 §1.6.1 的「不采用」结论一致）。

**重设计后的 NPU 算法一句话总结**：NCL 进 / 双 NCL 出，核内按 `(n, o_blk, c_block)` 单元把 C 轴作为向量轴，
沿 L 轴做一次**钳制连续 span 载入**（`(CH, LSPAN)`），对 `k_W` 窗口做 fp32 向量取最大值（`T.vmax`），
并在同一遍扫描内用块级 eq 掩码乘性累积（`vcmp(eq)`+`vcast`+`vmul`）追踪 NaN、用 int32 向量累加器
（first-hit = `eq`+`vmin` 取「等于最大值的最小 `iw`」、last-NaN = `isnan`+`vmax` 取「NaN 位置的最大 `iw`」）
产出索引，用核内二轴转置把 C 搬到最内连续轴；**全部 shape 走一般网格（C4/C4b 不采用）**。

### 0.7 标杆实现

- 源算子路径（获取 golden 语义依据）：`/workspace/TileOPs-gpu/tileops/kernels/pool/max_pool1d.py`
  （函数 `_max_pool1d_with_indices_kernel`）。
- 源仓参考实现 / 测试基准（NPU 侧移植后）：
  - `examples/TileOPs/tests/ops/test_pool.py`（双输出池化用例，`ref_program` =
    `F.max_pool1d(..., return_indices=True)`，torch CPU 参考，值 `atol=rtol=0`，索引 int64 精确比对）；
  - 源仓同文件（GPU）`tests/ops/test_pool.py`。
- **golden 以 §0.1 语义为唯一依据**，直接使用 `torch.nn.functional.max_pool1d(..., return_indices=True)`，
  **不复刻 §0.6 的 NPU 算法**（保证验证独立性）。
- 规格来源：`examples/TileOPs/tileops/manifest/pool.yaml` 的 `MaxPool1dIndicesFwdOp`
  （`flops = N*C*L_out*k_W`、`bytes = (N*C*L_in + N*C*L_out)*elem_bytes + N*C*L_out*8`）。
- 基准来源：`examples/TileOPs/benchmarks/ops/bench_pool.py`（双输出 1D 池化 bench）。

---

## 1. 概述

### 1.1 算子名称

`_max_pool1d_with_indices_kernel`（迁移自 TileOPs `MaxPool1dIndicesFwdOp` / `MaxPool1dWithIndicesKernel`）。

### 1.2 功能描述

窗口式（非自适应）一维最大池化前向（**双输出**）：对每个输出位置 $o_w$，在其
`kernel_size × stride × padding × dilation` 决定的窗口内取最大值，并同时输出该最大值元素沿 L 轴的
位置 `iw`（int64）；窗口越界部分按 $-\infty$ 填充（源实现以「越界 lane 跳过」表达）；
窗口内任一元素为 NaN 时输出 NaN（NaN 胜出），索引取窗口内**最后访问的 NaN** 的 `iw`；
并列（tie）取首个命中（first hit）。输入**恰好 3D NCL**，dtype 支持 `float16 / bfloat16 / float32`，
`out` 同 dtype、`indices` int64。

### 1.3 数学公式

$$
\text{out}[n,c,o_w]=\max_{0\le k_w<k_W}\ x_{\text{pad}}\big[n,c,\ o_w s_W-p_W+k_w d_W\big],
\qquad
\text{indices}[n,c,o_w]=i_w^\star
$$

$$
x_{\text{pad}}[n,c,i_w]=x[n,c,i_w]\ \ \text{if}\ \ 0\le i_w<L_{in};
\qquad x_{\text{pad}}=-\infty\ \ \text{otherwise}
$$

选中元素 $i_w^\star$ 按 §0.1 的精确规则确定（`k_w` 递增访问；first-hit 并列；NaN 胜出时值 NaN、索引取 last NaN）。
输出长度 $L_{out}=$ `pool_output_dim`$(L_{in},k_W,s_W,p_W,\text{ceil\_mode},d_W)$（helper 逐字取自源仓 `common.py`，**不重算**）。

### 1.4 算法描述

**（迁移决策后的 NPU 侧算法，见 §0.6）** 单步 + 窗口归约类算子，纯 Vector：

1. host/trace 期（factory 内）：由 `(N,C,L_in,k_W,s_W,p_W,d_W,ceil_mode)` 算出输出长度 `L_out`
   （`pool_output_dim`）；由 UB 容量判据（§4.5）选出通道向量块 `CH` 与**输出位置块 `BL`**；
   `num_logical / num_kernels / num_local` 在 factory 层求为编译期常量。**全 shape 走一般网格**
   （C4/C4b 不采用，§0.6 R5）。
2. 逻辑任务单元 `(n, o_blk, c_block)`（`BL` 输出位置 × `CH` 通道）。单元内：
   - 计算载入范围 `base/span/LSPAN/s0`（纯标量，2 次钳制）；
   - `acc`（`BL × CH` fp32）经 `T.vbrc(-inf)` 置 $-\infty$；`flag`（`BL × CH` fp32）经 `T.vbrc(1.0)` 置 1；
     `max_idx`（`BL × CH` int32）经整 buffer 广播置哨兵 `BIG=L_in`；`nan_idx`（`BL × CH` int32）置 `-1`；
   - `T.copy` 载入 `(real_c, LSPAN)` 连续 span → 核内转置 `(LSPAN, CH)` → `vcast` fp32（fp32 输入用 `T.copy`）；
   - 块级一次 `T.vcmp(row_f32, row_f32, eq_mask, "eq")` + `T.vcast` → `eq_f32`（非 NaN 掩码）；
   - **第一遍**（值 + NaN + last-NaN）：`for kw in T.serial(k_W): for owl in T.serial(BL):` 算 `iw = base + owl*s_W + kw*d_W`；
     若 `0<=iw<L_in`（逐 lane 守卫，相对缓冲下标 `j = iw - s0`）则 `T.vmax` 更新 `acc[owl]`、
     `T.vmul(flag[owl], eq_f32[j], flag[owl])`、并用 `vselect(eq2b,-1,fv)`+`T.vmax` 更新 `nan_idx[owl]`；
   - **第二遍**（first-hit，`acc` 已定稿）：同 `(kw,owl)` 循环，`T.vcmp(row_f32[j], acc[owl], "eq")` →
     `T.vselect(eqb, fv, BIG)` → `T.vmin(max_idx[owl], cand, max_idx[owl])`；
   - 末尾：`cfinal = (nan_idx >= 0)` → 值 `out32 = vselect(cfinal, NaN, acc)`、索引 `idx32 = vselect(cfinal, nan_idx, max_idx)`；
     再追加**空窗口子句**：`idx32 = vselect(idx32==BIG, 0, idx32)`（对齐源，见 §0.1）
     → 值 `vcast` 回 dtype、索引 `vcast` 到 int64 → 二者各自 `transpose([1,0])` 回转 `(CH, BL)` → 写回 GM。

**与源算法的结构差异**（来源标注）：
- 归约/追踪单元：源「每输出元素一个线程 + 标量状态机」→ NPU「每单元一个 C 向量块 + `T.vmax`/`vmin` 向量累加器」（§0.6 R1/R2/R3）。
- NaN 追踪：源「标量 bool `has_nan \| isnan(val)`」→ NPU「C 向量 eq 掩码乘性累积」（§0.6 R2）。
- 索引追踪：源「标量 `max_idx`/`nan_idx` + `first_valid` 分支」→ NPU「int32 向量累加器（`eq`+`vmin` first-hit / `isnan`+`vmax` last-NaN）」（§0.6 R3）。
- 访存：源「逐元素 GM gather」→ NPU「UB 连续 span 载入 + 核内融合转置」（§0.6 R4）。
- 并行：源「block×thread 两级 + lane/元素」→ NPU「一维 Kernel + `(n,o_blk,c_block)` 单元 + C 轴向量 lane」（§0.6 R1）。
- L 轴分块（1D 新增）：源无（逐元素未载入）；NPU 因 `L_in` 可达 4096 必须沿输出轴做 `BL` 块 + 连续 span 载入。
- 结构特化：源无 → NPU **评估后不采用**（§0.6 R5）。
- **不变**：窗口取值集合、$-\infty$ padding 语义、first-hit 并列规则、NaN 胜出 + last-NaN 索引 + $\pm\infty$ 取 $+\infty$、
  `ceil_mode` 输出尺寸规则、fp32 比较域 + 单次输出舍入、int64 索引契约、NCL 双输出契约、双输出同轮融合。

### 1.5 数据流图

```
GM[x: (N,C,L_in) fp16/bf16/fp32]
   │  T.copy 连续 span (real_c, LSPAN)  [s0 钳制, 静态长度]
   ▼
UB[row_src: (CH,LSPAN) dtype] --T.transpose[1,0]--> UB[row_dst: (LSPAN,CH) dtype] --T.vcast(rint)--> UB[row_f32 (LSPAN,CH) f32]
   │  块级一次 NaN 探针：vcmp(row_f32,row_f32,"eq") + vcast --> UB[eq_mask / eq_f32]
   │  第一遍: T.serial kw (k_W) × T.serial owl (BL) + 逐 lane 守卫
   ▼
   ├─ 值：   UB[acc  (BL,CH) f32 <- vmax(acc[owl], row_f32[j])，初值 -inf]
   ├─ NaN：  UB[flag (BL,CH) f32 <- vmul(flag[owl], eq_f32[j])，初值 1]
   └─ last-NaN：UB[nan_idx (BL,CH) i32 <- vmax(nan_idx[owl], vselect(eq2,fv,-1))，初值 -1]
   │  第二遍: T.vcmp(row_f32[j], acc[owl], "eq") -> vselect(eqb, fv, BIG) -> vmin(max_idx[owl], cand)
   │        UB[max_idx (BL,CH) i32，初值 BIG=L_in] = min{iw : x[iw]==acc}
   │
   │  cfinal = vcmp(nan_idx, 0, "ge")                 # 窗口含 NaN ⟺ nan_idx >= 0
   ├─> UB[out32 f32] = vselect(cfinal, nanbuf, acc)  --vcast(rint)--> UB[acc16] --transpose[1,0]--> UB[acc_t (CH,BL)] --T.copy--> GM[out]
   └─> UB[idx32 i32] = vselect(cfinal, nan_idx, max_idx) --空窗口子句(复用 cfinal: idx32==BIG → 0)--> --transpose[1,0]--> UB[idx_t] --vcast(rint,i32->i64)--> UB[idx64 (CH,BL)] --T.copy--> GM[indices]
（无 GM 中间往返、无 Cube / L1 / L0C 路径）
```

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先**调研**——对同一数学语义的算法族回答调研四问，选定算法族（§1.6.0）；
> 再在保证数学等价的前提下优化公式（§1.6.1）；再把保留下来的计算全部交给向量单元（§1.6.2）；
> 最后回答「在哪个轴向量化」（§1.6.3）。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象 = **同一数学语义的算法族**（所有能算出「窗口最大值 + 赢家 `iw`」的结构）。
> 基线候选 = 输入公式 / 源算法（直接窗口取最大值，逐元素标量 + 标量 first-hit / last-NaN 状态机 + 标量 NaN 追踪）。
> 本算子属**窗口 / 归约 + argmax 索引类** → **完整调研**（R1 候选表 + R3 四口径表 + R4 逐候选评估）。
> 参考表命中行：`algorithm-candidates.md` 的 `ALG-pooling`（family=pooling，`R1` 命中「avg pool = 常数权重卷积」、
> `R2`「滑窗天然流式」、`R3`「窗口重叠数据的重复读」、`R4`「C 轴向量轴」，`known_impl` 指向池化系列，
> `kb_links = PL-1.2-caxis-accum / PL-1.3-host-permute / PL-1.4-tiling-heuristic`）——**命中行已全部评估**，
> 其中 `R1` 命中项对 **max 不成立**（见候选 1/2）。另命中 pattern-library 的 `PL-1.22-maxreduce-nan-tracking`
> （max 池化/归约的 NPU 向量实现形态，多任务独立实证）与 `TRAP-npuir-arange-pattern-with-whole-plane-reduce`
> （argmax 型索引特化的断裂链——**本设计据此否决 R5 特化，见 §0.6 R5**）。

**R1 等价化简公式候选**（基线候选必须在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线：源算法**（直接窗口取最大值 + 标量 first-hit/last-NaN 状态机 + 标量 NaN 追踪） | $\max_{k_w}x_{\text{pad}}$ + 状态机索引 | — | — | ✅（基线） |
| 0' | **基线向量化**（C 轴向量 `vmax`/`vmin` + 块级向量 eq 掩码 NaN 追踪 + int32 向量索引累加器 + L 轴 span 载入分块 + 两遍 first-hit） | 同 #0 | **数学恒等**（C 轴独立，仅 SIMD 化；索引 first-hit 改「等于最大值的最小 `iw`」、last-NaN 改「NaN 位置的最大 `iw`」，逐位等价；L 分块只改访存不改取值集合） | 标量链 → 向量链（`/CH` 次向量 op）；逐元素 gather → span 载入复用 | ✅（选定） |
| 1 | 常数权重卷积 / im2col+Cube GEMM（`ALG-pooling` 的 `R1` 命中行） | 线性加权和 | **❌ 不等价**：$\max$ 非线性，不存在权重矩阵使 $\max$ 等于线性加权和；且 argmax 索引无法由线性算子给出；`L_out` 作 Cube N 轴常 < 分形下限 16 | — | ❌（数学不等价） |
| 2 | sum-pool × 常数（`ALG-pooling` 命中行） | $(\sum x)\cdot c$ | **❌ 不等价**（给出 avg/sum，不给 max，更不给索引） | — | ❌（数学不等价） |
| 3 | 前缀最大值 + 差分 | $P[b]-P[a]$ | **❌ 不等价**：$\max$ **无逆元**（例 `x=[5,1,3]`，区间 $[1,3)$ 的 max $=3$ 而 $P[3]{-}P[1]=0$）；索引亦不可得 | — | ❌（数学不等价） |
| 4 | 归约结构优化：值 / NaN / 索引**同一遍扫描内融合**（源已同轮） | 单轮：`vmax` + eq 掩码 + 索引累加器 | 数学恒等（同轮与分轮结果相同） | 避免第二遍扫描 | ✅（并入 #0'） |
| 5 | **访存量削减：连续 span 载入 + 块内复用**（源为逐元素 GM gather，同一输入被 `k_W` 个窗口重复读） | 单元载入一次 `(CH, LSPAN)`，块内 `k_W·BL` 次读取全命中 UB | 数学恒等（读同一批元素；载入钳制只改地址范围，读出仍按 `0<=iw<L_in` 守卫） | GM 读次数由 `N·C·L_out·k_W` 降至 `N·C·(L_in + halo)` | ✅（并入 #0'） |
| 6 | **索引追踪的代数改写**（标量状态机 → 集合极值） | first-hit = `min{iw : x[iw]==acc}`；last-NaN = `max{iw : isnan(x[iw])}` | 数学恒等（递增 `iw` 访问序下，first-hit = 最终最大值的最小位置、last-NaN = NaN 位置的最大位置；D-1 机器验证） | 顺序覆写 → 可交换 `vmin`/`vmax` 归约（可向量化） | ✅（并入 #0'） |
| 7 | 退化/几何结构特化（整线归约 C4 / 恒等 C4b） | `L_out==1` 整线 max；`k==1` 恒等 | 值半**数学恒等**（仅窗口覆盖整线 / 恰为单位窗时）；**索引半在本工具链不可证明可行**（`TRAP-npuir-arange-pattern-with-whole-plane-reduce`） | 整线：多次向量 `vmax`/`vmin` → 单次归约 | ❌ **不采用**（§0.6 R5；仅保留数学等价性论证） |
| 8 | 可分离 1D→无（本算子已是 1 个空间维） | — | **不适用**（1D 无可分离结构） | — | ❌（结构不适用） |

**R1 初判小结**：唯一数学等价的化简结构是「**索引追踪的代数改写**」（#6：顺序状态机 → 集合极值，使索引可向量化）、
「**值与 NaN 判决融合**」（#4）与「**连续 span 载入的访存量削减**」（#5，1D 相对源的关键补强）；
与 avg 池化不同，max 的**全部代数化简路径（#1/#2/#3）因 $\max$ 非线性 / 无逆元而不成立**——
这是 max 归约相对 sum 归约的本质限制（数学事实，非 API 或性能问题）。退化特化（#7）数学上成立但**索引半受已知陷阱阻断，不采用**。

**R2 在线算法**：**有**（且基线本身已是单遍）。结构判据：算子可分解为 **running 最大值** +
**running NaN 标记** + **running first-hit/last-NaN 索引**（窗口内可增量更新，无跨元素依赖），
故天然存在单遍流式变体——每个输入元素读一次、参与其归属（窗口重叠时若干）输出的 `vmax`/`vmin`/NaN 判决。
**收益口径**：扫描遍数 **1**（无两遍式 GM 扫描；索引追踪所需的「两遍」均在 **UB 驻留的同一 span** 上，非 GM 二读）、
中间缓冲 `O(BL·CH + LSPAN·CH)`（远小于 `O(L_in·CH)` 的物化）、UB 驻留可行（§4.5）。
迁移任务佐证：源算法 §0.4 #4/#5/#6/#7 已是 running 比较 / running 标记 / running 索引——本算子
（`max_pool1d.py` 第 212–288 行）对应语句：`elif val > max_val: max_val = val; max_idx = flat_idx`（第 270–272 行）、
`if is_nan: nan_idx = flat_idx; has_nan = True`（第 261–265 行）、`first_valid` 首次 seed（第 266–269 行）
（机制与单输出工厂第 57 行的 `max_val = T.max(max_val,val)` / `has_nan = has_nan | isnan(val)` 同族同构），
是「存在在线变体」的直接证据。
结论：**在线单遍变体 = 采用（= 基线结构）**，无第二遍 GM 扫描 / 无中间矩阵物化。

**R3 复杂度对比**（逐候选四口径；本算子是**带宽受限**算子，访存 / 遍数为主导项，不只看 FLOPs）：

口径：FLOPs = 元素比较次数（一次最大比较 = 1）；访存 Bytes = GM 读 + GM 写（**不含** UB 内搬运；中间缓冲不落 GM；
索引输出 int64 = 8 B/元素）；扫描遍数 = 输入 GM 读取遍数；中间缓冲峰值 = 片上驻留量。

| 算法候选 | FLOPs（比较次数） | 访存量 Bytes（GM） | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|------------------|--------------------|---------|-------------|---------------------|
| 0 基线（直接标量） | `N·C·L_out·k_W` | `(N·C·L_in + N·C·L_out)·eb + N·C·L_out·8` | 1（窗口重叠时靠缓存吸收重复读） | `O(1)`（寄存器标量） | 元素级并行 `N·C·L_out`；**无跨核归约** |
| **0' 基线向量化（选定，C 轴 + span 载入分块 + 向量索引）** | 同上（`/CH` 次向量 `vmax`/`vmin`） | 同上 + span 重读比 `num_oblk·LSPAN/L_in`（sincnet `k=s` ≈ 1.01；ecg ≈ 1.22） | 1 | `acc+flag+max_idx+nan_idx+row_f32+eq_*` = `O((BL + LSPAN)·CH)` | 单元并行 `N·num_oblk·ceil(C/CH)`；**无跨核归约** |
| 1 常数权重卷积 / GEMM | **不适用**（非线性，无法表达；索引亦不可得） | — | — | — | — |
| 2 sum-pool × 常数 | **不适用**（给出的不是 max） | — | — | — | — |
| 3 前缀最大值 + 差分 | **不适用**（max 无逆元） | — | — | — | — |
| 7 退化特化（整线归约 / 恒等） | **不采用**（索引半受 `arange` 陷阱阻断） | — | — | — | — |

> 口径含：GM 输入读 + 值输出写 + **索引 int64 输出写**；不含：UB 内搬运 / 转置 / NaN 追踪 / 索引追踪 / 守卫
> （片上，计入 §1.6.0 估算行的发射项）。**基线在表内**（#0）；选定候选 #0'。
> 候选 1/2/3 因**数学不等价**不参与复杂度对比（标注 N/A）；候选 4/5/6 并入 #0'；候选 7 不采用；候选 8 结构不适用。
> **注（双输出相对单输出的额外代价）**：索引引入 ① 每元素额外 1 条 `vmax`（last-NaN 候选）+ 每元素 1 条 `vmin`
> （first-hit 候选，含 `vcmp`+`vselect`）；② GM 侧 int64 索引写 = 8 B/元素（对 `L_in` 较小的 shape 占比高，
> 例如 `textcnn-global` 的索引写 131 072 B 约占该 workload 总流量 3%）。这是保持「first-hit / last-NaN」语义的**必要代价**。

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附 `docs/` / `examples/` 佐证或「未文档化假设」标注）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 0 基线（标量） | **Vector 单元浪费**（逐元素标量比较，`PL-1.2` 同族实测窗口累加 98.3% 标量） | 寄存器 | — | 静态（`k_W` 编译期常量） | 无 | ❌ 计算单元失配（重设计为 #0'） |
| **0' 基线向量化（C 轴，选定）** | **Vector** ✓（`T.vmax`/`T.vmin`/`T.vcmp` fp32+int32 ✓ / `T.vselect` ✓ / `T.vcast` ✓ / `T.vbrc` ✓ / `T.transpose` ✓） | 见 §4.5（代表档 ≈ 110–112 KiB 原始 / ×1.7 后 ≈ 188–191 KiB，对照 192 KiB） | **C 轴恒为 8 的整数倍**（C=8/80/128/256）；`L_in`(32/128/2048/4096) 与 `LSPAN` 仅作转置后 row 长度，不作向量轴，不要求整除 | ✓（`k_W`/`BL`/`LSPAN` 编译期常量；动态量仅 `o_w`/`iw` 派生标量守卫与 2 次钳制） | 无跨引擎流水 | ✅ **选定** |
| 7 退化特化 | Vector reduce ✓ | `(CH,L_in)` 块 | ✓ | ✓ | 单归约 | ❌ **不采用**：索引半依赖 `arange` pattern 链（`TRAP-npuir-arange-pattern-with-whole-plane-reduce`，与 `AdaptiveMaxPool2dIndicesFwdOp` 同构）⇒ 无法证明索引正确，走一般路径（§0.6 R5） |
| 1 / 2 / 3 | **不适用**（max 非线性 / 无逆元，非 MAC 密集） | — | — | — | — | ❌ 数学不等价，无需亲和评估 |
| 8 可分离 | **不适用**（1D 无可分离结构） | — | — | — | — | ❌ 结构不适用 |

**R4 弃选理由的证据引用（negative-claim-evidence 口径）**：

- 候选 1/2/3：**数学事实**（$\max$ 非线性、无逆元），非 API / 性能断言——不作 API 检索需求（不存在可检索的「最大化简公式」）。
- 候选 0 的「标量占比」反例引用 pattern-library `layout.md` 的 `PL-1.2-caxis-accum` 与「诊断触发器」
  （「窗口累加 98.3% 标量源于 `j·sW`（`sW≠1`）跨步系数阻碍向量化，换轴后 5.6x」）。
- 候选 7 的索引侧不可行引用 `TRAP-npuir-arange-pattern-with-whole-plane-reduce`（`traps-compiler.md`，
  status verified，device 910B3，本会话同 stamp，有 repro `repro/TRAP-npuir-arange-pattern-with-whole-plane-reduce.py`）：
  `arange` 派生下标 pattern 在「`vselect` + `reduce`」链内被读回损坏；处置 = 交回一般路径。
- 候选 0'/6 的 dtype 支持矩阵来自 `docs/Tilelang.language/`（`T.vmax`/`T.vmin`/`T.vcmp`/`T.vselect`/`T.vcast`/`T.vbrc`/`T.transpose` 各 §2.2.1 表，见 §3.4）。
- **bf16 注意**：`T.vmax`/`T.vmin` 的 dtype 表**不含 bf16**（`docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1、
  `docs/Tilelang.language/比较操作/T.vmin.md` §2.2.1），故 bf16 输入必须经 `T.vcast` 上转 fp32 后计算——
  与「fp32 比较域」语义一致（§0.6 R2），无额外代价。
- **`T.vmin` 存在性**：`docs/Tilelang.language/比较操作/T.vmin.md` §2.2.1（int16/int32/int64/fp16/fp32；含标量广播形态）✓。

**设计期估算（D-2 roofline）** —— 协议见 `_shared/standards/hardware-cost-model.md` §2，
常数引用 `pattern-library/constants.md`：`CONST-mte2-degradation`（HBM 3:1 混合读写地板 1.26 TB/s，
版本戳 `tilelang 0.1.2+15ad002b3d / 2026-09-21`）；`CONST-vector-launch-overhead`（0.5µs/op，
版本戳 `tilelang 0.1.2+3a214cde / 2026-09-09`）；`CONST-capacity-910B2C`（192 KB/AIV，
版本戳 `tilelang 0.1.2+1990aa9fe4 / 2026-09-17`）。三者均绑定 **Ascend910B2C**，本设备为 **Ascend910B3**（实查 20×2=40）。

以 dispatch 代表 workload 计算（`CH`/`BL` 按 §4.5 容量档；`eb` = 输入元素宽度）：

| workload | 流量项（`bytes ÷ 有效带宽 1.26 TB/s`） | 发射项（向量 op 数） | 容量项 | 估算下界 |
|----------|--------------------------------------|----------------------|--------|---------|
| `smoke`(1,8,32, k3s2p1, CH=8,BL=16) | 1 792 B ÷ 1.26 TB/s = **1.4 ns** | 1 单元 × ~（16·3·3）op | 0 | **launch 主导**（见注） |
| `sincnet-speaker-local`(32,80,4096, k3s3p0, CH=64,BL=16) | 55 915 520 B ÷ 1.26 TB/s = **44.4 µs** | 5 504 单元 × ~（16·3·3）op = **0.79 M op** | 0 | **≈ 44.4 µs**（流量主导；发射项不确定） |
| `textcnn-global`(64,256,128, k128s1p0, CH=64,BL=1) | 4 358 144 B ÷ 1.26 TB/s = **3.46 µs** | 256 单元 × ~（1·128·3）op = **98 k op** | 0 | **≈ 3.46 µs** |
| `ecg-cnn-dilated`(16,128,2048, k5s2p2 d2 ceil, CH=64,BL=17) | 29 339 648 B ÷ 1.26 TB/s = **23.3 µs** | 1 952 单元 × ~（17·5·3）op = **0.50 M op** | 0 | **≈ 23.3 µs** |

> **估算下界（dispatch 代表 = `sincnet-speaker-local`）: ≈ 44.4 µs**（流量 44.4 + 发射「未文档化假设」 + 容量 0；
> 引用 `CONST-mte2-degradation` / `CONST-capacity-910B2C`）。**端到端实测地板大概率由固定开销主导**
> （NPU kernel launch + Event 计时 ~50–60 µs 量级，见 `examples/TileOPs/docs/gpu_to_npu_adaptation.md`）。
> **发射项口径注（未文档化假设）**：`CONST-vector-launch-overhead`（~0.5µs/op）是 **bn=512 宽块大 op** 的实测值，
> 对本设计 ≤64 元素的窄向量 op 不能直接外推；窄向量 op 的每-op 代价**未逐点实测**，标注「未文档化假设 + 估算依据」。
> **本算子每元素约 3 条向量 op**（`vmax` 值 + `vmax` last-NaN + `vmin` first-hit，另加 `vcmp`/`vselect`），
> 是同族单输出（~2 条/元素）的 **~1.5 倍**，且索引多 8 B/元素 GM 写 ⇒ 发射项**可能成为 `ecg` 的实际主导项** —— 列入 §9 风险点。
> **设备差异注**：三条常数均绑定 Ascend910B2C（48 向量核）；本设备 Ascend910B3（实查 40）——跨型号复用属**未实测假设**。
> **版本戳注**：本任务 toolchain stamp = `ba60e99c`（dev root build），与上述常数记录版本不同——跨 stamp 复用属未实测假设（§9.4）。

**调研结论**：选定算法族 = **「直接窗口取最大值 + fp32 比较域 + C 轴向量化 + 显式块级向量 eq 掩码 NaN 追踪 +
C 轴向量化 int32 索引追踪（first-hit = `eq`+`vmin` 取「等于最大值的最小 `iw`」；last-NaN = `isnan`+`vmax` 取
「NaN 位置的最大 `iw`」）+ 核内融合转置（NCL→C 最内）+ L 轴连续 span 载入分块 + 在线单遍」**（R3 表候选 #0'；
候选 #4 融合扫描、#5 span 载入削减、#6 索引代数改写均并入 #0'；**候选 #7 退化特化不采用**，见 §0.6 R5）。
关键依据：R3 表显示 #0' 的 FLOPs（= `N·C·L_out·k_W`，与 manifest `roofline.flops` 逐字一致）与访存量
（= manifest `roofline.bytes` + halo 放大）均为各候选最小量级（候选 1/2/3 数学不等价）；R4 表显示 #0' 唯一同时满足
「Vector 计算单元匹配 + C 轴向量整除 + 全静态框界 + 无跨核归约 + 容量充裕」。
**与基线的结构差异一句话**：数学公式、取值集合与 NaN 判决完全不变，索引由「标量顺序状态机」改写为「C 向量 int32
累加器 + 集合极值」，执行单元由「每输出一线程的标量链 + 逐元素 GM gather」改为「`(n,o_blk,c_block)` 单元的
C 轴向量比较器 + 连续 span 载入 + 向量索引/NaN 追踪器」。
**调研范围声明**（确有一项候选（退化特化）被评估后否决，但**无更低复杂度算法族**）：
本调研覆盖 ① `algorithm-candidates.md` 的 `ALG-pooling` 命中行全部候选（R1 常数权重卷积 / sum-pool×常数 /
R2 滑窗流式 / R3 窗口重叠重复读 / R4 C 轴向量轴）——并**逐项判定其对 max+argmax 的适用性**（R1 命中项对 max 因非线性不成立）；
② 恒等变形空间（前缀最大值差分、可分离 1D→无、归约融合、span 载入访存削减、**索引状态机的集合极值改写**）；
③ pattern-library 同类实现（`PL-1.1-transpose-chain` / `PL-1.2-caxis-accum` / `PL-1.3-host-permute` /
`PL-1.4-tiling-heuristic` / `PL-1.22-maxreduce-nan-tracking`）与陷阱条目（`TRAP-npuir-arange-pattern-with-whole-plane-reduce`、
`TRAP-npuir-vbrc-region-dropped`）；④ 本仓 `examples/` 同类实现与同族前作（§3.5.3）。
**互联网检索未使用**（本算子算法族在本地参考表 + 源码 + pattern-library 已完整覆盖，属窗口 / 归约 + argmax 索引类，
无需联网补充）。
**下游约束**：§1.4 算法描述、§1.6.1–§1.6.3 的分析对象 = 选定算法 #0'；§0.5/§0.6 的处置与之**一致**
（§0.6 R2 = #0' 的值/NaN 向量化、§0.6 R3 = #0' 的索引向量化、§0.6 R4 = #0' 的 span 载入、§0.6 R5 = 候选 #7 不采用）；
源算法 §0.4 已识别优化手段的意图承接逐项见 §0.6（**无静默丢弃**：14 项全部有处置）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定的算法 #0'。R1 表中初判可采纳的项在此完成四要素论证。

**语义等价性锚点 E1（基线公式的 NPU 表述）**：源以「越界 lane 跳过」表达 padding；
NPU 表述为「累加器初值 $-\infty$ + 仅 fold in-bounds lane」。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| **E1** | 语义等价锚点：越界跳过 ≡ $-\infty$ padding + 累加器初值 $-\infty$ | `if in_bounds: max_val = max(max_val, val)`，`max_val` 初值 $-\infty$ | `acc ← -inf`；仅对 in-bounds lane 做 `acc ← max(acc, …)` | **数学恒等**：$\max(-\infty,x)=x$，跳过 lane 与 fold 一个 $-\infty$ 逐位等价（**空窗口两侧同为 `−∞`**）；已由 `verify_equiv.py` E1 组（vs `F.max_pool1d`，torch CPU fp32 参照 + 目标 dtype 往返比对，0 ULP 判据）机器验证为**逐位一致** | 无（表达式改写，非降本项）——作为后续优化项的语义锚 |

**采纳的数学等价优化项**：

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| **C1** | 连续 span 载入的**钳制等价改写**（替代逐元素 GM gather 的地址计算） | 逐元素读 `x[batch,c,iw]`，判据 `0<=iw<L_in`（每输出元素 `k_W` 次 strided gather） | 载入连续 span `[s0, s0+LSPAN)`（`s0=max(0,min(base, L_in-LSPAN))`，`LSPAN=min(span,L_in)`，`base=ow0*sW-pW`，`span=(BL-1)*sW+(k_W-1)*d_W+1`）；块内读 `j = iw - s0`，判据 `0<=iw<L_in` | **数学恒等**：对所有窗口内 `iw∈[base, base+span)`，`s0<=iw<s0+LSPAN ⟺ 0<=iw<L_in`；且 `0≤s0`、`s0+LSPAN≤L_in` 保证载入恒在界内。已由 `verify_equiv.py` C1 组（`BL∈{2,8}` × 3 配置 × 3 dtype × 6 数据）机器验证为**逐位一致**并逐 lane 断言下标落界 | **访存削减 = 窗口重叠复用 + 连续 span 合并访存（非纯读次数）**：GM 读次数比 = `(num_oblk·LSPAN)/(L_out·k_W)`；`k/s > 1` 的 workload 降为 ecg ≈ `61·41/(1023·5)` = **0.48×**；`k=s` 的 workload（sincnet `86·48/4096` ≈ **1.01×**、textcnn）读次数 ≈ 不变而改为**访存模式改善（连续 span 取代跨步 gather）**。**注**：同一输入元素被其归属的 `k_W·BL` 个 (kw,owl) 读取全命中 UB |
| **N2** | **last-NaN 索引的集合极值改写** | 源逐 NaN 覆盖 `nan_idx = iw`（末写生效），末尾 `has_nan ? nan_idx : max_idx` | `nan_idx = max{iw : in-bounds ∧ isnan(x[iw])}`（int32，初值 -1）；`has_nan = (nan_idx >= 0)` | **数学恒等**：访问序 `iw` 随 `k_w` **严格递增** ⇒ 源「最后一条 NaN 覆盖」= NaN 位置集合中 `iw` **最大**者。已由 `verify_equiv.py` N2N3 组（含 `nan_two`/`nan_and_inf` 数据）机器验证 | 顺序覆写 → 可交换 `vmax` 归约（可 C 向量化）；无独立收益量，属**可向量化前提** |
| **N3** | **first-hit 索引的集合极值改写（等价前提：非空窗口）** | 源严格 `>` 更新 `max_idx` + `first_valid`（首写生效记录最后一条），末尾 `has_nan ? nan_idx : max_idx` | **非空窗口**（≥1 in-bounds lane）：`max_idx = min{iw : in-bounds ∧ x[iw] == acc}`（int32，初值 `BIG=L_in`）；**空窗口分支**：`max_idx==BIG ∧ ¬has_nan ⇒ idx=0`（对齐源 `max_idx` 初值，见 §0.1 空窗口契约）；总式 `idx = has_nan ? nan_idx : (max_idx==BIG ? 0 : max_idx)` | **数学恒等（非空窗口）**：源在**递增 `iw` 序**上以严格 `>` 更新 ⇒ 记录的是值首次达到窗口最终最大值的元素 = 「值等于 `acc` 的最小 `iw`」；全 `-inf` 窗口下 `min{iw : x[iw]==-inf}` = 首个有效元素（消解源 `first_valid`）。**空窗口分支与源同值**（源 `max_idx` 初值 0）。已由 `verify_equiv.py` N2N3 组（含 `all_neg_inf`/`tie_two_max`）与 EW 组/`empty_window_evidence()` 机器验证 | 值比较 + 顺序覆写 → `vmin` 归约（可 C 向量化）；`verify_equiv.py` N2N3-ord 守卫确认「误把比较放在累加之后」的形态会被检出（39/72 cases）；空窗口子句 O(1)/单元 |

> **两遍法 vs 单遍法（次序约束声明）**：N3 的等价前提是 `acc` 为**窗口最终最大值**，故本设计采用**两遍法**
> （第一遍算 `acc`/`flag`/`nan_idx`，第二遍算 first-hit）——两遍均在 **UB 驻留的同一 span** 上，不产生第二次 GM 读。
> 这**消除**了同族前作曾踩过的「`vcmp(gt)` 与 `vmax(acc)` 相对次序」陷阱：本设计的第一遍不含任何 first-hit 比较，
> 第二遍的 `acc` 已定稿，无「与更新前/后累加器比较」的分歧。§3.3/§6.2 冻结该两遍结构；`verify_equiv.py` 的
> N2N3-ord 守卫证明「把 winner 比较放进值累加之前（前缀最大值形态）」会被机器检出。
> **N3 等价前提的完整清单（v1 补全，共 4 支）**：① 访问序 `iw` 随 `k_w` 严格递增（`d_W>0`）；
> ② `acc` 已定稿（两遍法）；③ **非空窗口**（≥1 in-bounds lane）——此支下 `min{iw : x[iw]==acc}` 恒有解且 = 源 first-hit；
> ④ **空窗口分支**（无 in-bounds lane）——`max_idx==BIG ∧ ¬has_nan ⇒ idx=0`，与源 `max_idx` 初值同值（§0.1 空窗口契约）。
> 支 ③ 与 ④ 互斥且覆盖全部窗口。
> **无「倒数乘 / 归一化」优化空间**（显式结论）：与同族 avg 池化不同，max 池化**无除法 / 归一化**（结果即被选元素），
> 故 avg 迁移的「除法转乘倒数」在本算子不存在。亦**无高代价算子降代换**（无 exp/rsqrt 类）、
> **无多处出现的公共子表达式**（每元素仅参与一次比较；唯一可 CSE 的是边界判据与 NaN 掩码——前者由 C1 钳制载入消解、
> 后者由 R2 块级一次掩码消解）。**候选 1/2/3 不采纳**（数学不等价，§1.6.0 R1）。

**C4/C4b 的处置（USER SUPPLEMENT S1 的显式回答，独立段落）**：

> **C4（整线归约）/ C4b（恒等拷贝）在本设计中仅为「数学等价性论证（math-equivalence argument）」，
> 本设计不采用（NOT adopted）。** 值半（整线 `reduce_max` / 恒等拷贝）数学等价，已由
> `verify_equiv.py` 的 `specialisation_math_only()` 以 **value-only** 形式机器确认（3 dtype 全 `EQUIV_PASS`）；
> 但双输出变体的 C4 需构造整线 argmax 索引，其自然实现依赖 `T.arange` 派生的**线内下标 pattern**
> 在 `vselect + reduce` 链内消费——即 `TRAP-npuir-arange-pattern-with-whole-plane-reduce` 描述的
> **已验证断裂链**（同族 `AdaptiveMaxPool2dIndicesFwdOp` 因此 values 全对、索引大面积错，最终交回一般路径）。
> **本设计无法证明索引构造不依赖该断裂链**（该链的形态本身就是「派生 pattern 供 `vselect`/`reduce` 消费」；
> 1D 线内下标仍是同一 pattern 形态），**故按 S1「无法证明 ⇒ 不采用，走一般路径」执行**：
> 全部 shape（含 manifest `textcnn-global`，`L_out==1`）走一般网格。**§3/§5/§6 无任何 C4/C4b 分支**（一致性见 §0.6 R5）。
> **v1 注（与空窗口契约的关系）**：§0.1 的空窗口契约只给索引 epilogue 加一条**全 shape 统一**的子句（A4），
> **不引入**任何几何特化分支——与本段「不采用」结论一致。

**等价性机器验证（D-1）**：采纳项与语义锚点全部由
`examples/max_pool1d_with_indices/_max_pool1d_with_indices_kernel/verify_equiv.py` 在 **torch CPU fp32 参照 + 目标 dtype 往返比对（0 ULP 判据）** 下对照验证。
脚本按 **USER SUPPLEMENT S3 采样**（不做 C4/C4b 全枚举）：`(k,s,p,d,ceil_mode)` 空间按**条件分支采样**，
每分支至少 1 个代表组——`padding>0`（cfg `p>0,s!=k`、`p>0,s!=k,Lnd`）/ `dilation>1`（`d>1,p>0`、`ragged`）/
`ceil_mode`（`ceil,p>0,s!=k`）/ `stride!=kernel`（`p>0,s!=k`、`ceil,p>0,s!=k`、`out==1,s!=k`）/
`L non-divisible`（`p>0,s!=k,Lnd` L=63、`d>1,p>0` L=63、`ceil,p>0,s!=k` L=97、`ragged` L=2049）/
**`empty window`（v1 新增：`EW-min`/`EW-oob`/`EW-l2`，`d_W≥2 ∧ p_W>0` 且窗口 lane 全落界外）**。
全脚本耗时 **≈12–43 s**（随机器负载波动；远低于「几分钟」上限；不含 C4/C4b 全枚举，不会触发前两次被杀的长跑）。
数据 kind = `rand / nan_two / tie_two_max / all_neg_inf / pm_inf / zeros`；NaN-vs-NaN 计相等，NaN-vs-finite 恒判失败；
索引 int64 精确比对。空窗口 regime 另由 `empty_window_evidence()` 逐行留证（source/design `idx=0`；torch 越界，不作 oracle）。

内嵌结果表（**按 config 聚合后的输出（数值与计数已逐项核对）**；汇总：**8 个 item 组（含 v1 新增 `EW`），
共 1074 项检查，0 违反**）：

```
check      dtype      config            kinds  worst|d|     viol    status
--------------------------------------------------------------------------------------------
E1         all        p>0,s!=k          6      0.0000e+00  0       EQUIV_PASS  (checks=18)
E1         all        p>0,s!=k,Lnd      6      0.0000e+00  0       EQUIV_PASS  (checks=18)
E1         all        d>1,p>0           6      0.0000e+00  0       EQUIV_PASS  (checks=18)
E1         all        ceil,p>0,s!=k     6      0.0000e+00  0       EQUIV_PASS  (checks=18)
E1         all        out==1,s!=k       6      0.0000e+00  0       EQUIV_PASS  (checks=18)
E1         all        ragged            6      0.0000e+00  0       EQUIV_PASS  (checks=18)
E1         all        same_s,k,p0       6      0.0000e+00  0       EQUIV_PASS  (checks=18)
E1         all        identity k=1      6      0.0000e+00  0       EQUIV_PASS  (checks=18)
E1-idx     all        (8 configs)       6      0.0000e+00  0       EQUIV_PASS  (checks=18 each)
N2N3       all        (8 configs)       6      0.0000e+00  0       EQUIV_PASS  (checks=36 each)
N2N3-neq   all        (8 configs)       1      0.0000e+00  0       EQUIV_PASS  (checks=6 each)
N2N3-ord   all        (8 configs)       3      0.0000e+00  0       EQUIV_PASS  (checks=9 each)
B2         all        (8 configs)       6      0.0000e+00  0       EQUIV_PASS  (checks=18 each)
C1         all        p>0,s!=k          6      0.0000e+00  0       EQUIV_PASS  (checks=72)
C1         all        same_s,k,p0       6      0.0000e+00  0       EQUIV_PASS  (checks=36)
C1         all        identity k=1      6      0.0000e+00  0       EQUIV_PASS  (checks=72)
EW         all        EW-min            3      0.0000e+00  0       EQUIV_PASS  (checks=18)
EW         all        EW-oob            3      0.0000e+00  0       EQUIV_PASS  (checks=18)
EW         all        EW-l2             3      0.0000e+00  0       EQUIV_PASS  (checks=18)
--------------------------------------------------------------------------------------------
criterion: exact after rounding to the target dtype (0 ULP); NaN-vs-NaN counts equal,
NaN-vs-finite fails; indices compared exactly (int64).
order guard: the fused/premature form differs from the baseline on 39/72 cases (non-empty detection required)
C1 span-clamp: 90 (config, BL) structural checks; every in-bounds lane landed inside [0, LSPAN)
empty-window contract: 27 empty windows exercised; source == design idx (0); torch differs (out-of-range), not used as an oracle
§0.1 semantics evidence: all state-machine readings confirmed
empty-window evidence: source/design idx=0; torch idx in {1,2} (out-of-range) -> not used as an oracle
hard-coded guards: all ok
specialisation_math_only: C4-math value-only EQUIV_PASS (3 dtype); C4b-math value-only EQUIV_PASS (3 dtype)
[EQUIV] ALL PASS  (total item checks=1074, failures=0)
```

> 完整逐行输出可复现：`/workspace/venv/bin/python examples/max_pool1d_with_indices/_max_pool1d_with_indices_kernel/verify_equiv.py`
> （脚本随算子目录留存，Stage 2 维度 8 会重跑复核；行内 `config` 为分支代表标签，`all` 表示 3 dtype 聚合）。

**优化结论**：采纳 **3 项**——**C1**（连续 span 钳制载入 + 访存量削减，逐位等价）、**N2**（last-NaN 集合极值改写，
逐位等价）、**N3**（first-hit 集合极值改写，逐位等价；v1 补全「非空窗口」前提 + 空窗口分支），并建立 **E1** 语义锚点；
连同空窗口分支在内均已机器验证（**1074** 项检查全 `EQUIV_PASS`）。
候选 1/2/3 因**数学不等价**不采纳；候选 7（C4/C4b）**仅数学等价性论证、不采用**；候选 8（可分离）在 1D 结构不适用；
「倒数乘」优化在本算子**不存在**（max 无归一化）。**优化后公式（供 §3.1 公式拆解使用）**：

```
一般网格 (全 shape):
  base = ow0*sW - pW ; span = (BL-1)*sW + (k_W-1)*d_W + 1      # 编译期常量
  LSPAN = min(span, L_in) ; s0 = max(0, min(base, L_in-LSPAN))  # 每单元 2 次标量钳制 (C1)
  acc[owl,c] ← -inf ; flag[owl,c] ← 1 ; max_idx[owl,c] ← BIG ; nan_idx[owl,c] ← -1   # 整 buffer 广播
  load row[0:LSPAN, c] ← x[n, c, s0:s0+LSPAN]                   # 连续 span 载入 (C1)
  eq[c] ← (row[j,c] == row[j,c]) ; eq_f32 ← cast(eq)           # 块级一次 NaN 掩码
  # 第一遍：值 + NaN + last-NaN
  for kw in [0,k_W):
    for owl in [0,BL):
      iw = base + owl*sW + kw*dW ; j = iw - s0
      if 0 <= iw < L_in:                                        # 逐 lane 守卫 (C1 等价改写)
        acc[owl]     ← max(acc[owl], row_f32[j])                # C 向量 max
        flag[owl]    ← flag[owl] * eq_f32[j]                    # eq 掩码乘性 NaN 追踪
        nan_idx[owl] ← max(nan_idx[owl], eq[j] ? -1 : iw)       # last-NaN = NaN 位置的最大 iw (N2)
  # 第二遍：first-hit（acc 已定稿）
  for kw in [0,k_W):
    for owl in [0,BL):
      iw = base + owl*sW + kw*dW ; j = iw - s0
      if 0 <= iw < L_in:
        max_idx[owl] ← min(max_idx[owl], (row_f32[j] == acc[owl]) ? iw : BIG)   # first-hit = 等于最大值的最小 iw (N3)
  has_nan = (nan_idx >= 0)
  idx_eff[owl] = has_nan ? nan_idx[owl] : (max_idx[owl]==BIG ? 0 : max_idx[owl])   # 空窗口分支 (v1, 对齐源)
  out[n,c,ow0+owl]     = has_nan ? NaN  : acc[owl]
  indices[n,c,ow0+owl] = idx_eff[owl]
```

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点实现方案（§1.4 + §3.3 伪代码 + §6）中**全部**循环与标量计算点，逐项判定能否用向量操作等价替代。
> 替代方案所用 API 均以 `docs/Tilelang.language/` 条目佐证（下表「佐证」列）。

| # | 计算点 | 原实现形态（源 GPU） | 向量替代方案 | 是否替代 | 不可替代理由 / 佐证 |
|---|--------|---------------------|-------------|---------|--------------------|
| 1 | 累加器初始化 `acc ← -inf` | 源 `max_val = T.cast(-inf)` 标量（每输出元素 1 次） | `T.vbrc(-T.infinity("float32"), acc)`（**整 buffer** 广播 $-\infty$） | ✅ | 佐证：`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2（scalar→任意 shape 广播）、`docs/Tilelang.language/创建操作/T.infinity.md` §2.2.1。**dst 必须是整 buffer**——region 形态会被静默丢弃（`TRAP-npuir-vbrc-region-dropped`，§9.2） |
| 2 | NaN 标记初始化 `flag ← 1` | 源 `has_nan = False` | `T.vbrc(T.float32(1.0), flag)`（整 buffer）；标量 operand 必须是 **typed PrimExpr** | ✅ | 同上；裸 Python 数命中前端 rank 断言（`TRAP-npuir-vbrc-region-dropped`，§9.2） |
| 3 | first-hit 累加器初始化 `max_idx ← BIG` | 源 `max_idx = T.cast(0, "int64")` | `T.vbrc(T.int32(L_in), max_idx)`（整 buffer int32） | ✅ | 佐证同 #1；typed `T.int32(...)` |
| 4 | last-NaN 累加器初始化 `nan_idx ← -1` | 源 `nan_idx = T.cast(0, "int64")` | `T.vbrc(T.int32(-1), nan_idx)`（整 buffer int32） | ✅ | 佐证同 #1 |
| 5 | 窗口取最大值 `max_val = T.max(max_val, val)` | 逐元素标量比较（每输出元素 `k_W` 条） | `T.vmax(acc[owl:owl+1,:], row_f32[j:j+1,:], acc[owl:owl+1,:])`（C 轴向量，原地） | ✅ | 佐证：`docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1（int16/int32/int64/fp16/fp32 ✓；**bf16 ✗**，须先上转） |
| 6 | **first-hit 索引更新** `elif val > max_val: max_idx = iw` | 每元素标量比较 + 条件覆写 | **第二遍**：`T.vcmp(row_f32[j:j+1,:], acc[owl:owl+1,:], eqb, "eq")` + `T.vselect(eqb, fv, bigv, cand)` + `T.vmin(max_idx[owl:owl+1,:], cand, max_idx[owl:owl+1,:])` | ✅ | 佐证：`docs/Tilelang.language/比较操作/T.vcmp.md` §2.2.2（A/B 同形；fp32 ✓）、`T.vselect.md` §2.2.1（int32 ✓）、`T.vmin.md` §2.2.1（int32 ✓）；等价性见 §1.6.1 N3 |
| 7 | **last-NaN 索引更新** `if is_nan: nan_idx = iw` | 每元素标量 isnan + 无条件覆写 | `T.vcmp(row_f32[j], row_f32[j], eq2b, "eq")` + `T.vselect(eq2b, neg1v, fv, nanc)` + `T.vmax(nan_idx[owl], nanc, nan_idx[owl])` | ✅ | 佐证同 #6；等价性见 §1.6.1 N2 |
| 8 | **NaN 追踪** `has_nan \| T.isnan(val)` | 每元素标量 isnan + 标量 bool 或 | **块级一次**：`T.vcmp(row_f32, row_f32, eq_mask, "eq")` + `T.vcast(eq_mask, eq_f32, "rint")`；窗口内逐 (kw,owl) `T.vmul(flag[owl], eq_f32[j], flag[owl])` | ✅ | 佐证：`docs/Tilelang.language/比较操作/T.vcmp.md` §2.2.1（fp32 ✓，输出 bool 片上）、`docs/Tilelang.language/数学操作/T.vmul.md` §2.3（支持原地）。**必须用 `eq`**（`ne` 为 ordered-NE，NaN 恒假，不可作 NaN 探测） |
| 9 | 值结果 `if_then_else(has_nan, NaN, max_val)` | 每元素标量分支 | `T.vcmp(flag, zeros, cfinal, "eq")` + `T.vselect(cfinal, nanbuf, acc, out32)` | ✅ | 佐证同 #8；`nanbuf = inf - inf`（`docs/Tilelang.language/数学操作/T.vsub.md` §2.2.1 fp32 ✓） |
| 10 | 索引结果 `has_nan ? nan_idx : max_idx` + **空窗口子句 `max_idx==BIG ⇒ 0`**（v1） | 每元素标量选择（含空窗口兜底） | `T.vcmp(nan_idx, zero_i32, cfinal2, "ge")` + `T.vselect(cfinal2, nan_idx, max_idx, idx32)`；再 `T.vcmp(idx32, bigv, cfinal, "eq")` + `T.vselect(cfinal, zero_i32, idx32, idx32)`（复用 `cfinal`） | ✅ | 佐证：`T.vcmp` §2.2.1（int32 ✓，`ge`/`eq` 模式）、`T.vselect` §2.2.1（int32 ✓） |
| 11 | 逐元素 `T.cast(x[...],"float")` | 每元素一次标量 cast | `T.vcast(row_dst, row_f32, round_mode="rint")`（整 tile）；**fp32 输入改用 `T.copy`** | ✅ | 佐证：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（f16/bf16→f32 `rint` ✓）；同 dtype `vcast` 误编译见 §9.2 |
| 12 | 最终值 cast + 索引 cast | 每元素一次标量 cast | 值 `T.vcast(out32, acc16, "rint")`（fp32 输入改 `T.copy`）；索引 `T.vcast(idx_t, idx64, "rint")`（i32→i64） | ✅ | 佐证：`T.vcast.md` 第 50 行（i32→i64 `rint` ✓） |
| 13 | 布局重排（C→最内） | （源无；GPU 靠 coalesced 尾轴） | `T.transpose(src, dst, [1,0])`（值/索引各一次输入侧、一次输出侧） | ✅ | 佐证：`docs/Tilelang.language/创建操作/T.transpose.md` §2.3（二轴交换；fp16/bf16/fp32/int32 ✓） |
| 14 | **iw 标量 → 索引候选向量 `fv`** | （源无；源用标量 `flat_idx`） | `T.vbrc(T.int32(iw), fv)`（**整 buffer** `(1,CH)`，dst 是整 buffer 非 region） | ✅ | 佐证：`T.vbrc.md` §2.2.2；**不得写成 region**（`TRAP-npuir-vbrc-region-dropped`）。`fv` 为 `(1,CH)` 独立小 buffer，可随 `iw` 更新（整 buffer 写入） |
| 15 | 载入范围钳制 `base/span/LSPAN/s0` | （源无；源直接逐元素算地址） | 无（保持每单元 2 次标量 `min`/`max` 钳制） | ❌ | **理由（block 级索引 / 任务映射标量）**：`base` 仅依赖单元 id，是**每单元 2 次**的标量整数运算（非逐元素热点）；无「对标量循环变量做 min/max 钳制」的向量 API（v-prefix 面向 buffer 元素）。**1D 特有** |
| 16 | **窗口 `k_w` 循环 `for kw in T.serial(k_W)`** | `T.serial` 静态界 + 逐 lane 守卫 | 无（保持 `T.serial` + `if 0<=iw<L_in`；两遍各一次） | ❌ | **理由（tile 级顺序依赖）**：`kw` 是累加链的循环携带维（跨 `kw` 累加到同一 `acc`/`flag`/`nan_idx`/`max_idx`）；`k_W` 编译期常量故上界静态。循环体内已全部向量化（#5/#6/#7/#8），故该循环是 §6 的 block/tile 级调度循环，非逐元素标量循环 |
| 17 | **输出位置 `owl` 循环 `for owl in T.serial(BL)`** | `T.serial` 静态界 + 运行时 `acc`/`flag` 行索引 | 无（保持 `T.serial` + 运行时行索引 `owl`） | ❌ | **理由（tile 级顺序依赖 + 运行时行索引）**：`owl` 是 `acc`/`flag`/`max_idx`/`nan_idx` 的运行时行索引（`BL` 编译期常量）；循环体为 C 向量 op（切片 `[owl:owl+1,:]`） |
| 18 | **逐 lane 越界守卫 `0<=iw<L_in`** | 逐元素 in-bounds 谓词 | 候选替代：把界外缓冲 lane 预填 `-inf` 后**无条件** `vmax`/`vmin`（免守卫） | ❌ | **理由（动态偏移写入陷阱 + 无收益）**：预填需对缓冲的**运行时偏移区间**写 `-inf`，属 `TRAP-UB-dynsubview-dominance` 的静态多缓冲 IR 风险域；且它把所有 lane 都做一次 `vmax`/`vmin`（op 数不减）。本设计选择「载入钳制保证读出下标恒落界 + 逐 lane 标量守卫」，守卫是**标量谓词门控向量 op**（非逐元素标量计算） |
| 19 | 单元任务解码 `(n, o_blk, c_block)` + **persistent 调度任务 id `uid = t*num_kernels + cid`** + 尾块边界 `real_c`/`BL_real` | — | 无 | ❌ | **理由（block 级标量，非逐元素）**：每单元少量整数 `%`/`//` 与 `T.min` 边界，非逐元素热点；`T.serial(num_local)` 是 persistent 调度循环（静态界 + `uid<num_logical` 守卫），亦无对应逐元素向量 API。与 §3.3/§6 一一对应 |
| 20 | 输出长度 / `CH` / `BL` / `LSPAN` / `num_logical` / `num_kernels` / `num_local` | — | 无 | ❌ | **理由（host/trace 侧元数据）**：factory 层 Python 整数算术（编译期常量），不在 kernel 运行时逐元素执行 |
| 21 | host 侧参数规范化 / kernel 缓存 | — | 无 | ❌ | **理由（host 侧元数据）**：只改 shape/stride 视图与缓存键，符合 `ascend-constraints.md` §4；不在 kernel 内 |
| 22 | `cfinal` 与 `zero`/`nanbuf` 的单元级构造 | — | `T.clear(zeros)`（f32）/ `T.clear(zero_i32)`（i32）；`T.vbrc(inf, infbuf)`+`T.vsub(infbuf,infbuf,nanbuf)`（整 buffer op） | ✅ | 佐证：`docs/Tilelang.language/内存操作/T.clear.md` §2.2；`T.vbrc` §2.2.2；`T.vsub` §2.3（原地） |

**向量化结论**：逐元素计算（累加器/标记/索引累加器初始化 / 取最大值 / first-hit 与 last-NaN 索引更新 /
NaN 探测 / NaN 判决 / 值+索引 cast / 布局重排 / iw 候选广播 / **空窗口子句** / 单元级常量构造）
**已全部向量化**，替代方案 API 均有 `docs/` 条目佐证。
保留 **8 类** 循环 / 标量点，理由类别：**#16/#17** tile 级顺序依赖（循环携带累加维 + 运行时行索引，循环体内已向量化）/
**#18** 逐 lane 标量谓词门控向量 op（动态偏移预填有陷阱风险且无收益）/ **#15** block 级索引 / 任务映射标量（1D 特有钳制）/
**#19** block 级任务解码 + persistent 调度 id + 尾块边界（`real_c`/`BL_real`）/ **#20/#21** host 侧 trace 元数据。
无「以实现简单为由保留标量循环」的情形。**§6 循环结构中不出现逐元素标量循环**（与本结论一致）。

#### 1.6.3 向量化轴与数据布局决策

> **背景**：I/O layout 是契约（3D NCL 进 / 双 NCL 出），但**核内布局与 lane 映射是自由变量**。
> 迁移任务：源的并行轴选择（GPU 逐元素 lane 扁平映射）是**输入而非结论**，对照 §0.3 独立评估。

| 评分项 | 说明 | 候选 1（NCL / L 轴） | 候选 2（核内 NLC / C 轴） | 候选 3（host permute / C 轴） |
|--------|------|----------------------|---------------------------|------------------------------|
| 整除性 | 轴长 vs 向量宽度（fp16/bf16 ×8、fp32 ×4） | `L_in`=32/128/2048/4096（多为 2 幂）但**含非 8 倍数**（如 63/97）；窗口沿 L 步进 `s_W`=2/3 ≠ 1 | **C 恒为 8 的整数倍**（8/80/128/256）✓ | 同候选 2 |
| 尾 lane 浪费率 | `(ceil(len/vw)·vw − len)/len` | 视 `L_in`（非 8 倍数时浪费）；窗口跨步系数 `s_W≠1` 使向量化退化为标量 | **0%** | 0% |
| 累加链形态 | 该轴作串行归约维（stride≠1）还是向量维（stride=1） | L 作归约维需跨步窗口最大值；`PL-1.2` 同族实测 98.3% 标量 | C 作向量维（转置后 stride=1 连续）；空间轴（`k_w`）作串行归约维 | 同候选 2 |
| repack 代价 | 布局重排代价 | 无 repack，但需跨步窗口处理 | 核内融合转置链（`T.transpose` 二轴交换，`PL-1.1` 实测 ~µs 级，输入+输出各 1 次） | **host permute 106–147µs**（`PL-1.3`，已量化证伪） |
| UB 容量影响 | 布局对驻留缓冲的影响 | 小（row `(LSPAN)` 级） | 代表档 ≈ 111.1–112.2 KiB 原始（选定档 `BL=16/17`，§4.5），×1.7 后 ≈ 188.9–190.8 KiB，对照 192 KiB | 同候选 2 |

**候选矩阵**（全部枚举，弃选必须给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益 / 代价 | 是否采纳 |
|---|---------|---------|------------|----------------|---------|
| 1 | I/O 原生 NCL | L（行内连续） | 无 | 基线：窗口步进 `s_W`=2/3≠1，跨步窗口最大值需逐元素；`L_in` 非 8 倍数时尾 lane 浪费 | ❌ 量化理由：`PL-1.2`/`PL-1.4` 同族实测「`sW≠1` 跨步系数阻碍向量化 → 98.3% 标量，换轴后 5.6x」 |
| 2 | **核内 NLC（融合转置）** | **C（恒为 8 的整数倍）** | 每次 span `T.transpose([1,0])`（输入 1 次 + 输出 2 次 = 值 1 + 索引 1） | 尾 lane 0%；转置代价 ~µs 级（远低于算子本体 3–44µs） | ✅ **选定** |
| 3 | host permute NCL↔NLC 进出 | C | 两次 host permute | **106–147µs**（`PL-1.3`，实测）> 多数 workload 本体耗时 | ❌ 量化理由：repack 代价 > 算子本体流量数倍，净亏 |
| 4 | Cube / implicit-GEMM | Cube N 轴（`L_out`） | 权重矩阵构造 | **数学不可行**（max 非线性 + 索引不可得）；`L_out` 常 < 16 违 Cube 分形下限 | ❌ 见 §1.6.0 R1/R4 |
| 5 | **L 轴分块（BL）+ 连续 span 载入**（与轴选择正交，是必需的容量维度） | C（同上） | 无（只分块不改布局） | 使 `(CH, LSPAN)` 可驻留 UB（`L_in` 达 4096 时整线不可驻留）；halo 读放大 `LSPAN/(BL·s_W)` | ✅ **必须**（与候选 2 组合；非独立布局候选） |

**布局决策结论**：选定 **方案 #2 + #5：核内布局 = span 内 NLC（C 最内连续）、向量化轴 = C、
repack 路径 = 核内 `T.transpose([1,0])`（每次 span 一次 + 输出侧值/索引各一次回转）；
L 轴按输出块 `BL` 分块、每块一次钳制连续 span 载入**。
该结论已同步至：§3.3 伪代码（`row_src/row_dst/row_f32/eq_*/acc/flag/max_idx/nan_idx` 形状与内层向量维 = C、
`owl` 为行索引）、§4 内存规划（§4.3/§4.5 的 buffer 形状与预算）、§6 循环结构（内层向量维 = C）。
弃选方案量化理由见候选矩阵 #1/#3/#4。**迁移任务注**：源 GPU 的并行轴是「输出元素扁平 lane」（§0.3），
与 NPU Vector 轴不对应，此处已独立评估并替换为 C 轴（§0.6 R1）。

**D-3 实验裁决模式（主选 + 备选 + 裁决计划）**：主选与备选的分歧落在**窄向量 op 的每-op 发射代价**的**外推区间**——
`CONST-vector-launch-overhead` 只给了 bn=512 大 op 的 0.5µs/op 实测值，本设计 op 宽度 = `CH`（8/64），未逐点实测；
且本算子每元素比同族单输出多约 1 条向量 op（索引追踪）+ 8 B/元素索引写，发射项更敏感。
**方向证据**：`layout.md` 的 `PL-1.4-tiling-heuristic` 直接实测「CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs
——更宽 C 向量 + 更少 block 胜过更高空间并行度」，与 `CONST-vector-launch-overhead` 方向一致；
主选「容量允许的最大 CH（CH_MAX=64）」落在 `PL-1.4` 已实测范围（CH16/32/64）内。

- **主选（进 Stage 3）**：`CH =` 容量允许的最大 8 的倍数（`≤ min(C,64)`，判据见 §4.5），
  `BL =` 在该 `CH` 下满足容量不等式且 `≤ min(L_out,32)` 的最大整数；单元 = `(n, o_blk, c_block)`；
  `num_logical > vector_cores` 时 persistent（见 §5.5）。
  代表值（v1 修正判据）：`smoke` CH=8/BL=16 → 1 单元（无适配）；`sincnet` CH=64/BL=16 → 5 504 单元（persistent，40 核 × 138 任务）；
  `textcnn` CH=64/BL=1 → 256 单元（×7）；`ecg` CH=64/BL=17 → 1 952 单元（×49）。
  **依据**：`PL-1.4` 实测 CH64 为该池化族最优档。
- **备选（≤1 个，结构完整，可直接实现）**：**主选 CH 的一半 + BL 尽量放大**
  （sincnet → CH=32/BL=32（`LSPAN=96`）；ecg → CH=32/BL=35（`LSPAN=77`）），即把 op 数放大 2×、核数放大 2×，
  换取更少 span 重读。这是「发射代价主导 vs 吞吐主导」的另一端；buffer 形状与主选相同仅 CH 减半，
  分核三要素按新 `num_logical` 重算（sincnet CH=32 → num_cblk=3, num_oblk=43, num_logical=4 128，persistent 40 核 × 104 任务；
  ecg CH=32 → num_cblk=4, num_oblk=30, num_logical=1 920，persistent 40 核 × 48 任务）。
- **裁决计划**：代表 shape = `sincnet-speaker-local`、`ecg-cnn-dilated`（各 1 组）；指标 = msprof op Task Duration；
  测量未知常数 = 窄向量 op 的每-op 代价（用 `perf_opt` 下两种 CH/BL 的实测反解）；
  判定阈值 = **备选 latency < 主选 × (1 − 0.10) 才翻转**（10% 稳健性门槛）；
  裁决结果回写路径 = Stage 4 实测数据经附录补记固化 §1.6.3 判定依据，若翻转则按 `perf-feedback.md` 受控路由回写 DESIGN.md。
- **shape 分派评估**：本 kernel 由 `functools.lru_cache` 按 `(n,c,l_in,k,s,p,d,ceil,dtype)` 特化，
  **支持按 shape 分派**主选/备选——裁决计划须评估「大 `L_in`（UB 受限，如 sincnet）走主选；小 `L_in` 走更大 BL」的按负载分派。
- **BL 说明**：`BL` 取容量允许的最大值（非 1）：1D 的 halo 重叠使 `BL=1` 产生 `(k_W-1)·d_W+1` 倍的 span 重读，
  大 `BL` 把重读摊薄到 `≈1.1×`。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：**Developer**（迁移 prompt 指定；`os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`）。

### 2.2 选型理由

- 算子是**纯 Vector**（无 matmul / 无 Cube），仅需逐元素向量运算（`T.vmax`/`T.vmin`/`T.vcmp`/`T.vselect`/`T.vmul`）、
  广播（`T.vbrc`）、清屏（`T.clear`）与布局重排（`T.transpose`）——最大值由 `T.vmax` 窗口累加实现、
  **不使用 `T.reduce_max`**（C4 不采用，§3.4）——正是 Developer 模式目标域
  （`examples/adaptive_avg_pool2d/_adaptive_avg_pool2d_kernel/_adaptive_avg_pool2d_kernel.py` 等均为 Developer）。
- 无跨引擎流水、无手动 L1/L0 分级需求（`T.alloc_ub` 直映 UB）；编译器管理 UB 分配与自动同步（§7）。
- 源集成包 `ascend_mode = "Developer"`（`max_pool1d_with_indices.py` 第 171/246 行）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 环境变量 | `TILELANG_ASCEND_MODE=Developer` |
| 内存分配 | `T.alloc_ub(...)`（GM→UB 路径） |
| 计算方式 | v-prefix 向量 API（`T.vmax`/`T.vmin`/`T.vcmp`/`T.vselect`/`T.vmul`/`T.vcast`/`T.vbrc`/`T.clear`）+ `T.transpose` + `T.copy` |
| 同步 | 编译器自动同步（§7）；无手动 `T.set_flag`/`T.wait_flag` |
| Kernel 维度 | 一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)` |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**；每步优先映射 v-prefix 向量 API，与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `L_out ← pool_output_dim(L_in,k_W,s_W,p_W,ceil_mode,d_W)` | trace 期 host 侧；**不得重算**（逐字引用源 helper） |
| 2 | `base ← o_blk·BL·s_W - p_W`；`span ← (BL-1)·s_W+(k_W-1)·d_W+1`；`LSPAN ← min(span,L_in)`；`s0 ← max(0,min(base, L_in−LSPAN))` | 每单元 2 次标量钳制（C1） |
| 3 | `row_src[c, j] ← x[n, c0+c, s0+j]`，`j∈[0,LSPAN)` | 连续 span 载入 `(real_c, LSPAN)` |
| 4 | `row_dst[j, c] ← row_src[c, j]` | 核内二轴转置 → C 最内 |
| 5 | `row_f32[j, c] ← cast_f32(row_dst[j, c])` | 整 tile fp32 上转（`rint`） |
| 6 | `eq_f32[j, c] ← cast_f32(row_f32[j, c] == row_f32[j, c])` | 块级一次 NaN 掩码（`x=x` ⟺ 非 NaN） |
| 7 | `acc[owl, c] ← max(acc[owl, c], row_f32[j, c])` | C 轴向量取最大值 |
| 8 | `flag[owl, c] ← flag[owl, c] · eq_f32[j, c]` | NaN 乘性累积 |
| 9 | `nan_idx[owl, c] ← max(nan_idx[owl, c], (row_f32[j,c]≠row_f32[j,c]) ? iw : −1)` | last-NaN 索引（NaN 位置的最大 iw） |
| 10 | `max_idx[owl, c] ← min(max_idx[owl, c], (row_f32[j,c]==acc[owl,c]) ? iw : BIG)` | first-hit 索引（等于最大值的最小 iw，两遍法第二遍） |
| 11 | `has_nan ← (nan_idx ≥ 0)`；`out32 ← has_nan ? NaN : acc`；`idx32 ← has_nan ? nan_idx : (max_idx==BIG ? 0 : max_idx)` | NaN 判决 + 双路取舍 + **空窗口分支（v1，对齐源 `idx=0`）** |
| 12 | `acc_t[c, owl] ← cast_dtype(out32[owl, c])`；`idx_t[c, owl] ← idx32[owl, c]` | 单次舍回 + 转置回转 |
| 13 | `out[n,c0+c,ow0+owl] ← acc_t`；`indices[n,c0+c,ow0+owl] ← cast_i64(idx_t)` | 双输出写回 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0a | `acc ← -inf` | `T.vbrc(src, dst)` | `src=-T.infinity("float32")`（typed 标量），`dst=acc`（**整 buffer**） | Developer |
| 0b | `flag ← 1` | `T.vbrc(src, dst)` | `src=T.float32(1.0)`（typed 标量），`dst=flag`（整 buffer） | Developer |
| 0c | `max_idx ← BIG` / `nan_idx ← -1` | `T.vbrc(src, dst)` | `src=T.int32(L_in)` / `T.int32(-1)`，`dst=max_idx`/`nan_idx`（整 buffer） | Developer |
| 0d | `zeros ← 0` / `nanbuf ← NaN` | `T.clear(buf)` / `T.vbrc`+`T.vsub` | `T.clear(zeros)`（f32）/ `T.clear(zero_i32)`（i32）；`T.vbrc(T.infinity("float32"), infbuf)` + `T.vsub(infbuf,infbuf,nanbuf)`（`inf-inf=NaN`） | Developer |
| 3 | span GM→UB | `T.copy(src, dst)` | `src=x[n, c0:c0+real_c, s0:s0+LSPAN]`，`dst=row_src[0:real_c, 0:LSPAN]` | Developer |
| 4 | 二轴转置 | `T.transpose(src, dst, permutation)` | `permutation=[1,0]` | Developer |
| 5 | fp32 上转 | `T.vcast(src, dst, round_mode)` | `round_mode="rint"`（同形、源为 UB）；**fp32 输入改 `T.copy`** | Developer |
| 6 | NaN 探测（IEEE eq，块级一次） | `T.vcmp(A,B,C,cmp_mod)` + `T.vcast` | `A=B=row_f32`，`C=eq_mask(bool)`，`cmp_mod="eq"`；`T.vcast(eq_mask, eq_f32, "rint")` | Developer |
| 7 | C 轴取最大值 | `T.vmax(src0, src1, dst)` | 切片 `(1,CH)`：`src0=dst=acc[owl:owl+1,:]`，`src1=row_f32[j:j+1,:]`（原地） | Developer |
| 8 | NaN 累积 | `T.vmul(src1, src2, dst)` | `src1=dst=flag[owl:owl+1,:]`，`src2=eq_f32[j:j+1,:]`（原地） | Developer |
| 9a | last-NaN 候选 | `T.vcmp` + `T.vselect` + `T.vmax` | `isnan = vcmp(row_f32[j],row_f32[j],eq2b,"eq")`；`nanc = vselect(eq2b, neg1v, fv)`；`T.vmax(nan_idx[owl], nanc, nan_idx[owl])` | Developer |
| 9b | iw 候选广播 | `T.vbrc(src, dst)` | `src=T.int32(iw)`，`dst=fv`（`(1,CH)` **整 buffer**） | Developer |
| 10a | first-hit 候选 | `T.vcmp` + `T.vselect` + `T.vmin` | `eqb=vcmp(row_f32[j], acc[owl], "eq")`；`cand=vselect(eqb, fv, bigv)`；`T.vmin(max_idx[owl], cand, max_idx[owl])` | Developer |
| 11a | has_nan 判决 | `T.vcmp` | `A=nan_idx`，`B=zero_i32`，`C=cfinal`，`cmp_mod="ge"` | Developer |
| 11b | 值取舍 | `T.vselect(Cond,A,B,Out)` | `Cond=cfinal`，`A=nanbuf`，`B=acc`，`Out=out32` | Developer |
| 11c | 索引取舍 | `T.vselect(Cond,A,B,Out)` | `Cond=cfinal`，`A=nan_idx`，`B=max_idx`，`Out=idx32` | Developer |
| 11d | **空窗口子句（v1）** | `T.vcmp(A,B,C,cmp_mod)` + `T.vselect` | `A=idx32`，`B=bigv`，`C=cfinal`（**复用**），`cmp_mod="eq"`；`T.vselect(cfinal, zero_i32, idx32, idx32)`（空窗口 `idx=0`）；须排在 11b 之后 | Developer |
| 12a | 值 dtype 回转 | `T.vcast(src,dst,round_mode)` | `round_mode="rint"`（fp32 输入改 `T.copy`） | Developer |
| 12b | 输出转置回转 | `T.transpose(src,dst,[1,0])` | 值 `(BL,CH)→(CH,BL)`；索引同 | Developer |
| 12c | 索引 i32→i64 | `T.vcast(src, dst, "rint")` | `idx_t (CH,BL) int32 → idx64 (CH,BL) int64` | Developer |
| 13 | 双输出写回 | `T.copy` | 值 `acc_t[0:real_c,0:BL_real] → out[n, c0:c0+real_c, ow0:ow0+BL_real]`；索引同 | Developer |
| 尺寸 | `L_out` | Python `pool_output_dim`（host/trace） | 逐字复用源 helper | host |
| 分块 | `CH`/`BL`/`LSPAN` | factory 层容量判据（§4.5） | `CH = max 8 的倍数 ≤ min(C,64)`；`BL = max 整数 ≤ min(L_out,32) s.t. 预算 ≤ 192KB` | host |

### 3.3 计算伪代码

> 注（迁移模板要求）：`T.Tensor` 的数据类型以**第二个位置参数**传入，不用 `dtype=` 关键字。
> 注：`block_m` 为源接口兼容参数（K9 保留），NPU tiling 由 shape 派生，其值被校验后忽略。

```python
# 迁移决策后的 NPU 算法伪代码（Developer 模式；C 轴向量化 + span 载入 + eq 掩码 NaN 追踪 + int32 向量索引）
@tilelang.jit(out_idx=[1, 2], target="npuir")   # 注：本分支 jit 不接受 compile_flags（§9.2）
def _build(block_m=None):
    @T.prim_func
    def main(
        x:       T.Tensor((N, C, L_in), dtype),        # dtype 位置参数（非 dtype=）
        out:     T.Tensor((N, C, L_out), dtype),
        indices: T.Tensor((N, C, L_out), "int64"),
    ):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            acc     = T.alloc_ub((BL, CH), "float32")   # 值累加器（向量轴 = C）
            flag    = T.alloc_ub((BL, CH), "float32")   # NaN 乘性标记（初值 1）
            zeros   = T.alloc_ub((BL, CH), "float32")
            infbuf  = T.alloc_ub((BL, CH), "float32")
            nanbuf  = T.alloc_ub((BL, CH), "float32")
            out32   = T.alloc_ub((BL, CH), "float32")
            cfinal  = T.alloc_ub((BL, CH), "bool")
            max_idx = T.alloc_ub((BL, CH), "int32")     # first-hit 累加器（初值 BIG=L_in）
            nan_idx = T.alloc_ub((BL, CH), "int32")     # last-NaN 累加器（初值 -1）
            cand    = T.alloc_ub((BL, CH), "int32")
            nanc    = T.alloc_ub((BL, CH), "int32")
            bigv    = T.alloc_ub((BL, CH), "int32")
            neg1v   = T.alloc_ub((BL, CH), "int32")
            zero_i32= T.alloc_ub((BL, CH), "int32")
            fv      = T.alloc_ub((1, CH), "int32")      # iw 广播（整 buffer）
            row_src = T.alloc_ub((CH, LSPAN), dtype)
            row_dst = T.alloc_ub((LSPAN, CH), dtype)
            row_f32 = T.alloc_ub((LSPAN, CH), "float32")
            eq_mask = T.alloc_ub((LSPAN, CH), "bool")
            eq_f32  = T.alloc_ub((LSPAN, CH), "float32")
            acc16   = T.alloc_ub((BL, CH), dtype)
            acc_t   = T.alloc_ub((CH, BL), dtype)
            idx32   = T.alloc_ub((BL, CH), "int32")
            idx_t   = T.alloc_ub((CH, BL), "int32")
            idx64   = T.alloc_ub((CH, BL), "int64")

            for t in T.serial(num_local):              # persistent（静态界）
                uid = t * num_kernels + cid
                if uid < num_logical:
                    nb   = uid // (num_oblk * num_cblk)
                    r    = uid %  (num_oblk * num_cblk)
                    oblk = r // num_cblk
                    cb   = r %  num_cblk
                    real_c = T.min(CH, C - cb * CH)
                    ow0 = oblk * BL
                    base = ow0 * s_W - p_W
                    s0 = T.max(0, T.min(base, L_in - LSPAN))         # 钳制载入起点
                    T.vbrc(-T.infinity("float32"), acc)              # acc <- -inf（整 buffer）
                    T.vbrc(T.float32(1.0), flag)                     # flag <- 1（整 buffer）
                    T.vbrc(T.int32(BIG), bigv)                       # bigv <- BIG（整 buffer）
                    T.vbrc(T.int32(-1), neg1v)                       # neg1v <- -1（整 buffer）
                    T.clear(zero_i32)                                # zero_i32 <- 0
                    T.vbrc(T.int32(BIG), max_idx)                    # max_idx <- BIG（整 buffer）
                    T.vbrc(T.int32(-1), nan_idx)                     # nan_idx <- -1（整 buffer）
                    T.clear(zeros); T.vbrc(T.infinity("float32"), infbuf)
                    T.vsub(infbuf, infbuf, nanbuf)                   # NaN = inf - inf
                    T.copy(x[nb, cb*CH:cb*CH+real_c, s0:s0+LSPAN],
                           row_src[0:real_c, 0:LSPAN])               # 连续 span 载入
                    T.transpose(row_src, row_dst, permutation=[1, 0])
                    if is_f32: T.copy(row_dst, row_f32)
                    else:      T.vcast(row_dst, row_f32, round_mode="rint")
                    T.vcmp(row_f32, row_f32, eq_mask, "eq")          # 块级一次 NaN 探针
                    T.vcast(eq_mask, eq_f32, round_mode="rint")

                    # 第一遍：值 + NaN + last-NaN
                    for kw in T.serial(k_W):
                        for owl in T.serial(BL):
                            iw = base + owl * s_W + kw * d_W
                            j  = iw - s0
                            if 0 <= iw and iw < L_in:                 # 逐 lane 守卫
                                T.vmax(acc[owl:owl+1, 0:CH],
                                       row_f32[j:j+1, 0:CH],
                                       acc[owl:owl+1, 0:CH])
                                T.vmul(flag[owl:owl+1, 0:CH],
                                       eq_f32[j:j+1, 0:CH],
                                       flag[owl:owl+1, 0:CH])
                                T.vbrc(T.int32(iw), fv)               # iw 广播（整 buffer (1,CH)）
                                T.vcmp(row_f32[j:j+1, 0:CH],
                                       row_f32[j:j+1, 0:CH], eq2b, "eq")
                                T.vselect(eq2b, neg1v[0:1, 0:CH], fv,
                                          nanc[0:1, 0:CH])            # 非 NaN -> -1, NaN -> iw
                                T.vmax(nan_idx[owl:owl+1, 0:CH],
                                       nanc[0:1, 0:CH],
                                       nan_idx[owl:owl+1, 0:CH])

                    # 第二遍：first-hit（acc 已定稿）
                    for kw in T.serial(k_W):
                        for owl in T.serial(BL):
                            iw = base + owl * s_W + kw * d_W
                            j  = iw - s0
                            if 0 <= iw and iw < L_in:
                                T.vbrc(T.int32(iw), fv)
                                T.vcmp(row_f32[j:j+1, 0:CH],
                                       acc[owl:owl+1, 0:CH], eqb, "eq")
                                T.vselect(eqb, fv, bigv[0:1, 0:CH],
                                          cand[0:1, 0:CH])            # 命中 -> iw, 否则 -> BIG
                                T.vmin(max_idx[owl:owl+1, 0:CH],
                                       cand[0:1, 0:CH],
                                       max_idx[owl:owl+1, 0:CH])

                    T.vcmp(nan_idx, zero_i32, cfinal, "ge")           # nan_idx >= 0 <=> 含 NaN
                    T.vselect(cfinal, nanbuf, acc, out32)             # has_nan ? NaN : max
                    T.vselect(cfinal, nan_idx, max_idx, idx32)        # has_nan ? nan_idx : max_idx
                    T.vcmp(idx32, bigv, cfinal, "eq")                 # 空窗口: idx32==BIG (复用 cfinal, v1)
                    T.vselect(cfinal, zero_i32, idx32, idx32)         # 空窗口 -> idx=0 (对齐源)
                    if is_f32: T.copy(out32, acc16)
                    else:      T.vcast(out32, acc16, round_mode="rint")
                    BL_real = T.min(BL, L_out - ow0)
                    T.transpose(acc16, acc_t, permutation=[1, 0])
                    T.transpose(idx32, idx_t, permutation=[1, 0])
                    T.vcast(idx_t, idx64, round_mode="rint")          # i32 -> i64
                    T.copy(acc_t[0:real_c, 0:BL_real],
                           out[nb, cb*CH:cb*CH+real_c, ow0:ow0+BL_real])
                    T.copy(idx64[0:real_c, 0:BL_real],
                           indices[nb, cb*CH:cb*CH+real_c, ow0:ow0+BL_real])
    return main
```

（伪代码为**决策示意**，Stage 3 可细化局部 buffer 复用与 `fv`/`bigv` 的操作数形态；但「向量轴 = C」「核内二轴转置」
「span 钳制载入」「块级 eq 掩码乘性 NaN 追踪」「**两遍法 first-hit（`eq`+`vmin`）**」「**last-NaN（isnan+`vmax`）**」
「`nan_idx>=0` 统一 NaN 判决」「**空窗口子句（`idx32==BIG ⇒ 0`，复用 `cfinal`，v1）**」是冻结决策。
**实现级约束（§9.2）**：① `T.vbrc` 的 dst 必须是**整 buffer**（`fv` 为 `(1,CH)` 独立整 buffer，非 region——region 形态静默丢弃，
`TRAP-npuir-vbrc-region-dropped`）；② `T.vbrc` 标量 operand 必须是 typed PrimExpr；③ `T.vcmp(eq)`+`T.vcast` 块级掩码必须在 `T.vmul` 之前；
④ 第二遍必须在第一遍**全部** `vmax`/`vmul`/`vmax(nan_idx)` 之后（`acc` 必须定稿）；⑤ `T.vcmp(nan_idx,zero_i32)` 必须在两遍之后；
⑥ **空窗口子句必须在 `out32` 计算之后**（它复用 `cfinal` 布尔缓冲；顺序颠倒会使 `out32` 的值选择读到空窗口掩码）。
`verify_equiv.py` 只验公式不验 op 次序/落地形态，故 §9.2 列入实现级检查项。）

### 3.4 API 可行性确认

| API | 来源确认 | 是否验证 |
|-----|---------|---------|
| `T.Kernel(num, is_npu=True) as (cid, _)` | `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3（NPU ≤2 维） | ✓ 现有 examples 广泛使用 |
| `T.alloc_ub` | `docs/Tilelang.language/内存操作/T.alloc_ub.md` §2.2.2（1D~2D） | ✓（同族前作同款） |
| `T.vbrc`（scalar→任意 shape 广播） | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2 | ✓ |
| `T.clear` | `docs/Tilelang.language/内存操作/T.clear.md` §2.2（fp32 ✓，1~5 维） | ✓ |
| `T.copy`（切片 GM↔UB） | `docs/Tilelang.language/内存操作/T.copy.md` §2.1/§2.2.2 | ✓ |
| `T.transpose`（二轴交换，fp16/bf16/fp32/int32） | `docs/Tilelang.language/创建操作/T.transpose.md` §2.3 | ✓ |
| `T.vcast`（f16/bf16→f32 `rint`；f32→f16/bf16 `rint`；i1→f32；i32→i64 `rint`） | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1（第 50 行 i32→i64） | ✓（**同 dtype 禁用**，§9.2） |
| `T.vmax`（int16/int32/int64/fp16/fp32；**不含 bf16**；`dst` 可原地） | `docs/Tilelang.language/比较操作/T.vmax.md` §2.2.1 | ✓ |
| `T.vmin`（int16/int32/int64/fp16/fp32；含标量广播） | `docs/Tilelang.language/比较操作/T.vmin.md` §2.2.1 | ✓ |
| `T.vcmp`（fp16/fp32/bf16/int16/int32/int64；输出 bool 片上；`eq` 模式 IEEE 正确） | `docs/Tilelang.language/比较操作/T.vcmp.md` §2.2.1/§2.3 | ✓ |
| `T.vselect`（fp16/fp32/bf16/int32/int64；bool 条件） | `docs/Tilelang.language/条件操作/T.vselect.md` §2.2.1 | ✓ |
| `T.vmul`（fp16/fp32/int16/int32/int64；支持原地） | `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.1/§2.3 | ✓ |
| `T.vsub`（fp32；标量/张量；原地） | `docs/Tilelang.language/数学操作/T.vsub.md` §2.2.1/§2.3 | ✓ |
| `T.infinity("float32")` | `docs/Tilelang.language/创建操作/T.infinity.md` §2.2.1（fp16/fp32 ✓；**bf16 ✗**） | ✓ |
| `T.min` / `T.max`（标量钳制） | `docs/Tilelang.language/比较操作/T.min.md` | ✓（前端标量 min/max） |

> **dtype 约束提醒**：`T.vmax`/`T.vmin` 的 dtype 表**不含 bf16**，`T.vsub`/`T.vmul` 亦不含 bf16；
> `T.vcmp`/`T.vselect`/`T.transpose`/`T.vcast` 支持 bf16。故 **bf16 输入必须经 `T.vcast` 上转 fp32 后计算**
> ——与「fp32 比较域」语义一致（§0.6 R2），无额外代价。
> **`bool`（i1）不可 load/store GM**：本设计的 `eq_mask`/`cfinal`/`eqb`/`eq2b` 均为**片上中间结果**，不写回 GM ✓。
> **`T.reduce_max`/`T.reduce_min` 本设计不使用**（C4 不采用）；故无「`reduce_min` 无独立文档页」的文档缺口问题。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | 否（用一维 `T.Kernel(num_kernels)`；任务维度在 kernel 内解码） | 一维网格 |
| GPU 专用 API | 否（全部 v-prefix / `T.copy` / `T.transpose` / scalar min-max，均为 NPU API） | — |
| GEMM 非整除 / L0C 溢出 | 否（无 Cube 计算，`L0C` 不适用） | — |
| 分核策略 | 涉及 | 见 §5.5（三要素齐全，persistent grid-stride） |
| host 侧禁止改输入内容 | 否（host 只做 shape/stride 元数据与缓存；无 padding / 无数据拷贝） | `ascend-constraints.md` §4 |
| 动态 `T.serial` 边界 | 否（`k_W`/`BL`/`LSPAN`/`L_out` 全为编译期常量，`T.serial` 上界天然静态） | 无需 `max_bin_extent` 类机制 |
| UB 容量 | 涉及（长 `L_in` 的 sincnet/ecg，且索引引入额外 UB buffer） | 容量驱动的 `CH`/`BL` 联合选择（§4.5 带字节数不等式） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 维度 / 并行模型 | `T.Kernel(ceildiv(total_output,block_m), threads)` + `T.Parallel(block_m)` | 一维 `T.Kernel(num_kernels, is_npu=True)` | §0.6 R1 |
| 归约单元 | 每线程标量比较链（`T.alloc_var`） | UB fp32 向量比较器（C 轴） | §0.6 R2 |
| NaN 追踪 | 标量 bool `has_nan \| isnan(val)` | 块级 C 向量 eq 掩码乘性累积（`vcmp`+`vcast`+`vmul`） | §0.6 R2 |
| 索引追踪 | 标量 `max_idx`/`nan_idx` + `first_valid` | C 向量 int32 累加器（`eq`+`vmin` / `isnan`+`vmax`） | §0.6 R3 |
| 数据搬运 | 逐元素 GM gather | UB 连续 span 载入 + 核内二轴转置 | §0.6 R4 |
| L 轴分块 | 源无 | `BL` 输出块 + `LSPAN` 连续 span（`s0` 钳制） | §0.6 R1/R4（1D 必需） |
| 越界守卫 | 逐元素 in-bounds 谓词 | 载入钳制（每单元 2 次标量）+ 逐 lane 相对守卫 | §1.6.1 C1 |
| 结构特化 | 无 | **评估后不采用**（S1；`arange` 陷阱） | §0.6 R5 / §1.6.1 |
| `threads` 参数 | factory `(block_m, threads)`；`default_config={"block_m":256}` | **移除 `threads`**（NPU 无 CUDA 线程）；callable 只留 `block_m` | 脚手架已按 K9 移除（集成包第 78–105 行） |
| 编译标记 | `compile_flags=["-O3","-DENABLE_BF16"]` | **本分支 jit 不接受 `compile_flags`**（§9.2），移除 | 同族前作 Stage 3 实测确认 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/max_pool1d/_max_pool1d_kernel/_max_pool1d_kernel.py` | **最高（同族 1D 单输出，CASE-pool-maxpool1d-migration，Stage 3 已过）** | Developer 一维 `T.Kernel(is_npu=True)`；`(n,o_blk,c_block)` 单元解码；span 钳制载入；`T.vbrc`/`T.clear`/`T.vcast`/`T.vmax`/`T.vcmp`(eq)/`T.vmul`/`T.vselect`/`T.transpose` 用法；persistent 分核与 `real_c`/`BL_real` 尾块；fp32 同 dtype 用 `T.copy` |
| `examples/adaptive_max_pool2d_with_indices/_adaptive_max_pool2d_with_indices_kernel/_adaptive_max_pool2d_with_indices_kernel.py` | **极高（双输出索引变体，前作 Stage 3 已过）** | int32 索引累加器 + `vselect`/`vmin`/`vmax`/`T.transpose`(int32)；`T.vbrc` 整 buffer/staging 形态；`nan_idx>=0` 统一 NaN 判决；`arange` pattern 陷阱的规避（退路 = 一般路径） |
| `examples/max_pool2d/_max_pool2d_kernel/_max_pool2d_kernel.py` | 高（同族窗口池化） | Developer 一维 Kernel；C 轴向量化 + eq 掩码 NaN 追踪骨架 |
| `examples/TileOPs/tileops/kernels/pool/max_pool1d_with_indices/max_pool1d_with_indices.py` | 高（本迁移的集成包 / host 层） | K5/K8/K9 适配形态；`block_m` 兼容参数；`pool_output_dim` 内联副本 |
| `examples/TileOPs/tests/ops/test_pool.py` | 高（规格/golden） | 双输出池化用例，`ref_program = F.max_pool1d(..., return_indices=True)`，值 `atol=rtol=0`，索引 int64 精确 |
| pattern-library `layout.md` 的 `PL-1.1`/`PL-1.2`/`PL-1.3`/`PL-1.4`/`PL-1.22`；`traps-compiler.md` 的两条 TRAP | 高（池化类实测模式与陷阱） | 核内融合转置链、C 轴切片累加、host permute 证伪、tiling 启发式、max 归约 NaN 追踪、`vbrc` region 丢弃、`arange` pattern 链断裂 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `x` | `(N, C, L_in)` | float16 \| bfloat16 \| float32 | **恰好 3D NCL** 连续；尾轴 `L_in` 连续 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `(N, C, L_out)` | same_as_input | NCL 连续；`out_idx=[1,2]` 第 1 路 |
| `indices` | `(N, C, L_out)` | **int64** | NCL 连续；`out_idx=[1,2]` 第 2 路 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `row_src` | `(CH, LSPAN)` | dtype | UB | span GM→UB 暂存 |
| `row_dst` | `(LSPAN, CH)` | dtype | UB | 核内二轴转置输出（C 最内） |
| `row_f32` | `(LSPAN, CH)` | float32 | UB | fp32 上转 + 向量比较源 |
| `eq_mask` | `(LSPAN, CH)` | bool | UB | 块级 NaN 掩码（**片上**，不落 GM） |
| `eq_f32` | `(LSPAN, CH)` | float32 | UB | NaN 掩码（`{0,1}`，供 `vmul`） |
| `acc` | `(BL, CH)` | float32 | UB | C 轴向量最大值累加器（初值 $-\infty$） |
| `flag` | `(BL, CH)` | float32 | UB | C 轴 NaN 乘性标记（初值 1） |
| `zeros` / `infbuf` / `nanbuf` | `(BL, CH)` | float32 | UB | 0 / +inf / NaN（`inf-inf`）常量 |
| `out32` | `(BL, CH)` | float32 | UB | 值判决结果（fp32） |
| `cfinal` | `(BL, CH)` | bool | UB | 双路共用 has_nan 判决条件（片上） |
| `max_idx` | `(BL, CH)` | int32 | UB | first-hit 累加器（初值 `BIG=L_in`） |
| `nan_idx` | `(BL, CH)` | int32 | UB | last-NaN 累加器（初值 -1） |
| `cand` / `nanc` / `bigv` / `neg1v` / `zero_i32` | `(BL, CH)` | int32 | UB | 索引候选中转 / 常量 vector |
| `fv` | `(1, CH)` | int32 | UB | `iw` 广播载体（整 buffer） |
| `acc16` / `acc_t` | `(BL, CH)` / `(CH, BL)` | dtype | UB | 值舍回 + 转置回转 |
| `idx32` / `idx_t` | `(BL, CH)` / `(CH, BL)` | int32 | UB | 索引取舍 + 转置回转 |
| `idx64` | `(CH, BL)` | int64 | UB | 索引落盘前 i64 载体 |

### 4.4 内存搬运路径

```
【一般网格（全 shape）】
GM[x] --T.copy span(real_c,LSPAN)--> UB[row_src] --T.transpose[1,0]--> UB[row_dst] --T.vcast(rint)--> UB[row_f32]
UB[row_f32] --T.vcmp(eq)+T.vcast--> UB[eq_mask/eq_f32]（块级一次）
[第一遍] UB[row_f32] --T.vmax(C 轴切片)--> UB[acc (BL,CH) f32]
          UB[eq_f32]  --T.vmul 乘性累积--> UB[flag (BL,CH) f32]
          UB[row_f32] --T.vcmp(eq)+T.vselect+T.vmax(fv)--> UB[nan_idx (BL,CH) i32]
[第二遍] UB[row_f32] --T.vcmp(eq vs acc)+T.vselect(bigv)+T.vmin--> UB[max_idx (BL,CH) i32]
cfinal = vcmp(nan_idx, zero_i32, "ge")
UB[acc]/UB[nan_idx]/UB[max_idx] --vselect--> UB[out32 f32] / UB[idx32 i32]
UB[out32] --T.vcast(rint)--> UB[acc16] --T.transpose[1,0]--> UB[acc_t (CH,BL)] --T.copy--> GM[out]
UB[idx32] --T.transpose[1,0]--> UB[idx_t] --T.vcast(rint,i32->i64)--> UB[idx64 (CH,BL)] --T.copy--> GM[indices]
（无 GM 中间往返、无 Cube / L1 / L0C 路径）
```

### 4.5 UB 内存预算

> **v1 修订**：字节系数由 v0 的 `53+2·eb` 修正为 **`69+2·eb`**（v0 漏计 `idx32`/`idx_t` 8 B + `idx64` 8 B），
> 并据修正判据重导 `BL`。逐 buffer 求和 = 闭式值（下表与判据互相印证）。

**一般路径逐 buffer 字节表（v1 **选定档**；行字节 = 元素数 × 元素宽度）**：

| Buffer | shape（sincnet，CH=64，BL=16，LSPAN=48，fp16） | Bytes | shape（ecg，CH=64，BL=17，LSPAN=41，bf16） | Bytes |
|--------|-----------------------------------------------|-------|-------------------------------------------|-------|
| `row_src` | `(64,48)` | 6 144 | `(64,41)` | 5 248 |
| `row_dst` | `(48,64)` | 6 144 | `(41,64)` | 5 248 |
| `row_f32` | `(48,64)` f32 | 12 288 | `(41,64)` f32 | 10 496 |
| `eq_mask` | `(48,64)` bool | 3 072 | `(41,64)` bool | 2 624 |
| `eq_f32` | `(48,64)` f32 | 12 288 | `(41,64)` f32 | 10 496 |
| `acc` | `(16,64)` f32 | 4 096 | `(17,64)` f32 | 4 352 |
| `flag` | `(16,64)` f32 | 4 096 | `(17,64)` f32 | 4 352 |
| `zeros`/`infbuf`/`nanbuf` | 3×`(16,64)` f32 | 12 288 | 3×`(17,64)` f32 | 13 056 |
| `out32` | `(16,64)` f32 | 4 096 | `(17,64)` f32 | 4 352 |
| `cfinal`（兼空窗口掩码，v1 复用） | `(16,64)` bool | 1 024 | `(17,64)` bool | 1 088 |
| `max_idx`/`nan_idx`/`cand`/`nanc`/`bigv`/`neg1v`/`zero_i32` | 7×`(16,64)` i32 | 28 672 | 7×`(17,64)` i32 | 30 464 |
| `fv` | `(1,64)` i32 | 256 | `(1,64)` i32 | 256 |
| `acc16` | `(16,64)` fp16 | 2 048 | `(17,64)` bf16 | 2 176 |
| `acc_t` | `(64,16)` fp16 | 2 048 | `(64,17)` bf16 | 2 176 |
| `idx32` | `(16,64)` i32 | 4 096 | `(17,64)` i32 | 4 352 |
| `idx_t` | `(64,16)` i32 | 4 096 | `(64,17)` i32 | 4 352 |
| `idx64` | `(64,16)` i64 | 8 192 | `(64,17)` i64 | 8 704 |
| **合计** | | **114 944 B ≈ 112.2 KiB** | | **113 792 B ≈ 111.1 KiB** |

> **注**：上表逐 buffer 求和 **= 容量判据闭式值**（`114 944 / 113 792 B`），二者一致、无低估。
> v0 采用的**候选档**（`BL=18/20`）修正后合计 = **`129 280 / 132 800 B`**（= 闭式值），`×1.7` 后
> `219 776 / 225 760 B = 214.6 / 220.5 KiB > 192 KiB` ⇒ **超容量被否**，故 v1 降档为 `BL=16/17`（判定行见下）。
> v0 自注「逐 buffer 略低（闭式更保守）」方向相反（逐 buffer 实高于 v0 闭式）——v1 修正为二者相等。

**容量判据（带字节数的不等式，非代表 shape 外推；v1 修正系数）**：
`G(CH, BL, LSPAN, eb) × INFLATE ≤ 196 608 B (= 192 KiB)`，其中

```
G = CH·LSPAN·(2·eb + 9)          # row_src(eb)+row_dst(eb)+row_f32(4)+eq_mask(1)+eq_f32(4)
  + BL·CH·(69 + 2·eb)            # 单输出基式 25+2·eb（6×f32=24 + cfinal bool 1 + acc16/acc_t 2·eb）
                                 #   + 7×i32(28) + idx32/idx_t(8) + idx64(8)   ← v0 漏计后两项
  + 256                          # fv (1,CH) i32
LSPAN = min((BL-1)·s_W + (k_W-1)·d_W + 1, L_in)
```

选型规则（**两维联合**）：`CH =` 满足不等式且 `≤ min(C,64)` 的**最大 8 的倍数**（`CH ≥ 8`；`C < 8` 时取 8 并以
`real_c` 收尾）；`BL =` 在该 `CH` 下满足不等式的**最大整数**，`BL ≤ min(L_out, 32)`；无解则 `CH` 折半重试。
`INFLATE` 双口径并列（`CONST-capacity-910B2C`）：**默认档（auto-multi-buffer 开）1.7**；关闭档 1.10–1.12。
**本版取路线 1**（按默认档 1.7 判据降 `BL`；不采用「改证膨胀系数 + 声明 auto-multi-buffer 关闭」的路线 2）。

- `sincnet`（fp16）：`CH=64`，`BL=16/LSPAN=48` → `G = 64·48·13 + 16·64·73 + 256 = 39 936 + 74 752 + 256 = 114 944 B`，
  `×1.7 = 195 405 B = 190.8 KiB ≤ 192 KiB` ✓（`BL=17` → `LSPAN=51` → `G = 42 432 + 79 424 + 256 = 122 112 B`，
  `×1.7 = 207 590 B ✗` ⇒ **选 `BL=16`**；v0 选的 `BL=18` 修正后 `G=129 280 B`、`×1.7=219 776 B ✗`）。
- `ecg`（bf16）：`CH=64`，`BL=17/LSPAN=41` → `G = 64·41·13 + 17·64·73 + 256 = 34 112 + 79 424 + 256 = 113 792 B`，
  `×1.7 = 193 446 B = 188.9 KiB ≤ 192 KiB` ✓（`BL=18` → `LSPAN=43` → `G = 35 776 + 84 096 + 256 = 120 128 B`，
  `×1.7 = 204 218 B ✗` ⇒ **选 `BL=17`**；v0 选的 `BL=20` 修正后 `G=132 800 B`、`×1.7=225 760 B ✗`）。
- `textcnn`（fp16，`L_out=1`，`BL=1/LSPAN=128`）：`G = 64·128·13 + 1·64·73 + 256 = 106 496 + 4 672 + 256 = 111 424 B`，
  `×1.7 = 189 420 B = 185.0 KiB ✓`（档位不变）。
- `textcnn`（fp32，`eb=4` → `CH` 折半为 32）：`G = 32·128·17 + 1·32·77 + 256 = 69 632 + 2 464 + 256 = 72 352 B`，
  `×1.7 = 123 000 B ✓`（**CH dtype 感知**：fp32 比 fp16 更早触发容量折半；档位不变）。
- `smoke`（fp16，`CH=8/BL=16/LSPAN=32`）：`G = 8·32·13 + 16·8·73 + 256 = 3 328 + 9 344 + 256 = 12 928 B`，
  `×1.7 = 21 978 B ✓`；fp32（`eb=4`）：`G = 8·32·17 + 16·8·77 + 256 = 14 464 B`，`×1.7 = 24 589 B ✓`。
- `test`（1,4,63）k3s2p1（fp16，`CH=8/BL=32/LSPAN=63`）：`G = 8·63·13 + 32·8·73 + 256 = 25 496 B`，`×1.7 = 43 343 B ✓`。
- `test`（1,4,97）k5s3p2 ceil（fp16，`CH=8/BL=32/LSPAN=97`）：`G = 8·97·13 + 32·8·73 + 256 = 29 032 B`，`×1.7 = 49 354 B ✓`。
- 逐 buffer 求和、逐编译期分支复算（同族 Stage 2 教训：「代表 shape 结论 ≠ 整条路径结论」、**漏计小 buffer 是低报高发区**——
  v0 正命中该模式，v1 以「闭式分量求和 = 系数 ∧ 逐 buffer 求和 = 闭式值」双重自洽拦截）。

**无退化路径容量核算**：本设计**不采用** C4/C4b（§0.6 R5），故**不存在**退化路径的独立容量分支——
一般路径是全 shape 的**唯一**路径（`textcnn-global` 走 `BL=1/LSPAN=128` 的一般路径，容量已在上表核实）。

### 4.6 动态轴定义

**无动态轴**：`N, C, L_in, k_W, s_W, p_W, d_W, ceil_mode, L_out` 与派生的
`CH, BL, LSPAN, num_cblk, num_oblk, num_logical, num_kernels, num_local`
全部在 factory 构建期（trace 期）确定，kernel 由 `functools.lru_cache` 按该组合特化。
（`o_w`/`iw`/`j`/`s0`/`base` 判定在 kernel 内是运行时标量，但它们是**由编译期 shape 推出的运行时值**，不是动态轴。）

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[1, 2], target="npuir")
def _build(block_m=None) -> ...      # 去掉源 GPU 的 threads 参数（K9）
```

> **注**：分支 jit **不接受 `compile_flags`**（源 GPU 的 `["-O3","-DENABLE_BF16"]` 须移除）——
> 同族前作 Stage 3 实测（`examples/max_pool2d/_max_pool2d_kernel/_max_pool2d_kernel.py` 注释）；列入 §9.2。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**纯 Vector**。

**判定依据**：算子仅含逐元素向量运算（`T.vmax`/`T.vmin`/`T.vcmp`/`T.vselect`/`T.vmul`）、广播（`T.vbrc`）、
cast（`T.vcast`）、清屏（`T.clear`）与布局重排（`T.transpose`），无 `T.gemm` / matmul
→ 不需要 L1 + L0A/L0B/L0C（§1.6.0 R1/R4 已弃 Cube 路径——数学上不可表达）。
故核数按**纯 Vector 算子翻倍**规则取物理核数 ×2（§5.5）。

### 5.2 Block 划分

本算子的分块维度是 **(通道块 CH) × (输出位置块 BL)** 的任务维度：

```python
CH = <满足 §4.5 容量判据的最大 8 的倍数，且 <= min(C, 64)；C < 8 时取 8>
BL = <在 CH 下满足 §4.5 不等式的最大整数，且 <= min(L_out, 32)>
LSPAN = min((BL - 1) * s_W + (k_W - 1) * d_W + 1, L_in)
num_cblk = ceil(C / CH)            # 通道块数
num_oblk = ceil(L_out / BL)        # 输出位置块数
# 全 shape 走一般网格（C4/C4b 不采用，§0.6 R5）——单元粒度唯一：
num_logical = N * num_oblk * num_cblk          # 单元 = (n, o_blk, c_block)
# num_kernels = min(num_logical, 40) ; num_local = ceil(num_logical / num_kernels)  （编译期常量）
```

- **`CH` 的选择理由**：① C 轴为向量轴，须为 8 的整数倍（fp16/bf16 向量宽 8、fp32 宽 4）；
  ② UB 预算上限（§4.5 判据）；③ 与 §5.5 分核方案配合使 `num_logical` 落在物理核数附近/之上（persistent）。
- **`BL` 的选择理由**：① 1D 必需（`L_in` 达 4096，整线不可驻留）；② 越大 halo 重读越小（`LSPAN/(BL·s_W)`）；
  ③ 受容量判据约束（§4.5），`BL ≤ min(L_out,32)`。
- **`num_logical` 唯一口径**：单元 = `(n, o_blk, c_block)`，通式 `N × ceil(L_out/BL) × ceil(C/CH)`
  （**无 C4b 整线口径分叉**——C4/C4b 不采用）。这也消除了同族前作「C4b 沿用通式会越界」的风险。

**代表性 shape 的分块结果**（`L_out` 由 `pool_output_dim` 给出）：

| workload | N | C | L_in | k/s/p/d/ceil | L_out | eb | CH | BL | LSPAN | num_cblk | num_oblk | num_logical |
|----------|---|---|------|--------------|-------|----|----|----|-------|----------|----------|-------------|
| `smoke-1x8x32-k3s2p1` (fp16) | 1 | 8 | 32 | 3/2/1/1/F | 16 | 2 | 8 | 16 | 32 | 1 | 1 | **1** |
| `smoke` (fp32) | 1 | 8 | 32 | 3/2/1/1/F | 16 | 4 | 8 | 16 | 32 | 1 | 1 | **1** |
| `sincnet-speaker-local` | 32 | 80 | 4096 | 3/3/0/1/F | 1365 | 2 | 64 | 16 | 48 | 2 | 86 | **5 504** |
| `textcnn-global` (fp16) | 64 | 256 | 128 | 128/1/0/1/F | 1 | 2 | 64 | 1 | 128 | 4 | 1 | **256** |
| `textcnn-global` (fp32) | 64 | 256 | 128 | 128/1/0/1/F | 1 | 4 | 32 | 1 | 128 | 8 | 1 | **512** |
| `ecg-cnn-dilated` | 16 | 128 | 2048 | 5/2/2/2/T | 1023 | 2 | 64 | 17 | 41 | 2 | 61 | **1 952** |
| test `(1,4,63)` k3s2p1 | 1 | 4 | 63 | 3/2/1/1/F | 32 | 2 | 8 | 32 | 63 | 1 | 1 | **1** |
| test `(1,4,97)` k5s3p2 ceil | 1 | 4 | 97 | 5/3/2/1/T | 33 | 2 | 8 | 32 | 97 | 1 | 2 | **2** |

> **注意**：`textcnn-global`（`L_out==1`）**走一般路径**（`BL=1`，`LSPAN=128`）——本设计**不采用**整线特化（§0.6 R5）。

### 5.3 约束分析

- **对齐约束**：向量轴 C 恒为 8 的整数倍（8 同是 fp32 向量宽 4 的倍数）；`CH` 取 8 的倍数 ✓。
  `L_in`/`LSPAN` 仅作转置后 row 长度，**不作向量轴**，不要求整除 ✓。
- **UB 容量**：代表档 ≈ 111.1 / 112.2 KiB（逐 buffer 求和），默认档 ×1.7 → ≈ 188.9 / 190.8 KiB < 192 KiB ✓（§4.5）。
- **L0 容量**：无 Cube 计算，不适用 ✓。
- **非整除**：本算子**不存在 GEMM 式整除问题**——span 长度用静态 `LSPAN`；任务维度用 persistent
  `T.serial` + `if uid < num_logical` 守卫；通道尾块用 `real_c = min(CH, C - cb*CH)`、
  输出块尾用 `BL_real = min(BL, L_out - ow0)` 动态 copy/写回范围，padded lane 永不写回 GM。
  窗口越界由 §1.6.1 C1 守卫处理。

### 5.4 注意事项

- **尾单元**：`num_logical` 非 `num_kernels` 整数倍时（如 sincnet 5504 / 40 = 137.6），
  核内 `T.serial(num_local)` 的 `if uid < num_logical` 守卫丢弃越界任务。
- **通道尾块**：`C % CH != 0`（如 C=80 → CH=64 + `real_c=16`；C=4 → CH=8 + `real_c=4`）——
  copy/写回用 `real_c` 动态范围；`row_src/row_dst/row_f32/eq_*` 的 padded lane 残留旧值，
  但 padded lane 的 `acc`/`flag`/`max_idx`/`nan_idx` 永不写回 GM（各 lane 独立），无污染。
- **输出块尾**：`L_out % BL != 0`（如 sincnet 1365 = 16×85 + 5）——`BL_real` 动态写回范围。
- **载入钳制**：`s0 = max(0, min(base, L_in - LSPAN))` 保证 `0 ≤ s0` 且 `s0+LSPAN ≤ L_in`（§1.6.1 C1）；
  `LSPAN ≤ L_in` 由定义保证。
- **窗口守卫**：`if 0 <= iw and iw < L_in`（逐 lane），`j = iw - s0` 恒落 `[0,LSPAN)`（C1 已机器断言）。
- **零输出保护**：`L_out > 0` 由 op 层保证（"output size must be greater than zero"）。
- **NaN 语义**：必须在所有输入（含 `±inf`/subnormal）下保持「NaN 胜出 + `±inf` 取 `+inf`」——
  由 `eq` 掩码 + `flag` 乘性累积 + `nan_idx>=0` 判决保证（§0.6 R2/R3），**不依赖 `T.vmax` 的 NaN 行为**。

### 5.5 分核策略（物理核数适配）⭐

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md` §1
> （依据 `docs/开发指南.md` §3.3）。

- **① 逻辑核数计算**（单元粒度 = `num_logical` 的唯一源）：
  **全 shape 走一般网格**（C4/C4b 不采用，§0.6 R5）：单元 = `(n, o_blk, c_block)` →
  `num_logical = N × ceil(L_out/BL) × ceil(C/CH)`。
  代表值见 §5.2 表：smoke 1 / sincnet 5 504 / textcnn fp16 256、fp32 512 / ecg 1 952。

- **② 物理核数依据（实查）**：
  本设计环境实际执行查询（项目 venv `/workspace/venv/bin/python`，2026-10-14，本设计会话）：
  ```python
  from tilelang.utils.npu_utils import NPUUtils
  NPUUtils.get().get_aicore_num()
  # 实查返回值 = 20   (设备 Ascend910B3，npu-smi info 显示 910B3)
  ```
  **纯 Vector 算子核数翻倍**：`vector_cores = 20 × 2 = 40`。
  （**不采用** pattern-library `CONST-aicore-910B2C` 的 24（翻倍 48）——本设备为 910B3，必须实查；
  核数常数绑定设备型号，跨型号复用会有 ~29% 级偏差。）

- **③ 规模判定与分核方案**（三选一，逐 workload 判定；`num_kernels = min(num_logical, 40)`，
  `num_local = ceil(num_logical / num_kernels)` 为 **Python 整数编译期常量**——静态界）：

  | workload | num_logical | 物理核 | 判定 | 方案 |
  |----------|-------------|--------|------|------|
  | `smoke` | 1 | 40 | **逻辑核数 ≤ 物理核数** | **无需适配**：`num_kernels = 1`，每核 1 个单元 |
  | 多数测试 shape | 1–2 | 40 | **逻辑核数 ≤ 物理核数** | **无需适配**：`num_kernels = num_logical`，每核 1 个单元 |
  | `textcnn-global`（一般路径） | 256 / 512 | 40 | **极大规模**（256 = 40×6 + 16） | **核内串行（persistent）**：`num_kernels = 40`；`num_local = 7`（fp16）/ `13`（fp32）（**静态值**）；`uid = t*num_kernels + cid`，`if uid < num_logical` 守卫 |
  | `ecg-cnn-dilated` | 1 952 | 40 | **极大规模**（1952 = 40×48 + 32） | 同款 persistent：`num_local = 49` |
  | `sincnet-speaker-local` | 5 504 | 40 | **极大规模**（5504 = 40×137 + 24） | 同款 persistent：`num_local = 138` |

  **通用规则（factory 级）**：`num_logical = N × num_oblk × num_cblk`，`num_kernels = min(num_logical, vector_cores)`，
  `num_local = ceil(num_logical / num_kernels)`（Python 整数编译期常量）。当 `num_logical ≤ vector_cores`
  → `num_local = 1`，退化为「每核 1 单元 = 无需适配」；当 `num_logical > vector_cores` → 核内串行（persistent）。
  **核内串行循环边界 `num_local` 为静态值**——满足 `core-split-strategy.md` §1 要素③硬要求。
  > **不做空核的保证**：因 `num_kernels = min(num_logical, vector_cores)`，故 `num_kernels ≤ num_logical`，
  > 每核首个任务 `uid = cid < num_logical` 恒有效。
  > **分核增益预期**：三个 full workload 若按 GPU 式超发逻辑核（5 504/256/1 952），NPU 运行时会把超出物理核数的
  > 部分串行排入 40 核 → 引入核启动开销；persistent 化后固定 40 核 × 静态任务数（`num_local` ≤ 122），摊薄核启动开销。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅指 block/tile 级调度循环。**不出现逐元素标量循环**。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 逻辑单元（task） | block 级并行 | `T.Kernel(num_kernels)` | 每核处理 1 个或多个 `(n,o_blk,c_block)` 单元 |
| 累加器/标记/索引初始化 | **整块填充/清除**（无循环） | `T.vbrc(-inf, acc)` + `T.vbrc(1.0, flag)` + `T.vbrc(BIG, max_idx)` + `T.vbrc(-1, nan_idx)` + `T.clear` | 单元入口整 buffer 初始化（§1.6.2 #1–#4/#22） |
| 核内多任务 | 迭代（persistent） | `T.serial(num_local)`（静态界）+ `if uid<num_logical` | 极大规模时摊薄核启动开销（§5.5） |
| 窗口 `k_w` 方向（第一遍/第二遍各一次） | 迭代（归约链） | `T.serial(k_W)`（**编译期常量上界**）+ `if 0<=iw<L_in` | tile 级顺序依赖（循环携带累加维）+ 逐 lane 守卫（§1.6.2 #16） |
| 输出位置 `owl` | 迭代（运行时行索引） | `T.serial(BL)`（**编译期常量上界**） | `acc`/`flag`/`max_idx`/`nan_idx` 的运行时行索引；循环体为 C 向量 op（§1.6.2 #17） |
| 元素级（C 轴） | 向量化 | `T.vmax`/`T.vmin`/`T.vcmp`/`T.vcast`/`T.vmul`/`T.vselect`/`T.vbrc`（向量轴 = C，§1.6.3 决策） | C 轴 lane 并行，无逐元素循环 |
| 载入钳制 | 标量（每单元 2 次） | `T.max`/`T.min`（block 级下标） | 非逐元素热点（§1.6.2 #15） |

### 6.2 循环伪代码

```python
# num_kernels / num_local 由 num_logical 导出（全 shape 一般网格，§5.2/§5.5①）：
#   num_logical = N * num_oblk * num_cblk
with T.Kernel(num_kernels, is_npu=True) as (cid, _):        # 一维网格（<=2 维合法）
    for t in T.serial(num_local):                            # persistent（静态界）
        uid = t * num_kernels + cid
        if uid < num_logical:                                # 尾任务守卫
            # 解码为 (n, o_blk, c_block) 后进入 §3.3 单元体
            ...
            base = ow0 * s_W - p_W
            s0 = T.max(0, T.min(base, L_in - LSPAN))         # 载入钳制（每单元 2 次标量）
            [初始化 acc/flag/max_idx/nan_idx/常量; T.copy span -> row_src; T.transpose; vcast;
             vcmp(eq)+vcast -> eq_f32]
            for kw in T.serial(k_W):                         # 第一遍（值 + NaN + last-NaN）
                for owl in T.serial(BL):
                    iw = base + owl * s_W + kw * d_W
                    j  = iw - s0
                    if 0 <= iw and iw < L_in:
                        T.vmax(acc[owl], row_f32[j], acc[owl])
                        T.vmul(flag[owl], eq_f32[j], flag[owl])
                        T.vbrc(T.int32(iw), fv)
                        T.vcmp(row_f32[j], row_f32[j], eq2b, "eq")
                        T.vselect(eq2b, neg1v, fv, nanc); T.vmax(nan_idx[owl], nanc, nan_idx[owl])
            for kw in T.serial(k_W):                         # 第二遍（first-hit，acc 已定稿）
                for owl in T.serial(BL):
                    iw = base + owl * s_W + kw * d_W
                    j  = iw - s0
                    if 0 <= iw and iw < L_in:
                        T.vbrc(T.int32(iw), fv)
                        T.vcmp(row_f32[j], acc[owl], eqb, "eq")
                        T.vselect(eqb, fv, bigv, cand); T.vmin(max_idx[owl], cand, max_idx[owl])
            T.vcmp(nan_idx, zero_i32, cfinal, "ge")
            T.vselect(cfinal, nanbuf, acc, out32); T.vselect(cfinal, nan_idx, max_idx, idx32)
            T.vcmp(idx32, bigv, cfinal, "eq")              # 空窗口子句 (v1): 复用 cfinal
            T.vselect(cfinal, zero_i32, idx32, idx32)      # 空窗口 -> idx=0
            [vcast/copy + transpose + copy -> GM(out/indices)]
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**。理由：
- 单元内只有一条 `GM→UB→计算→UB→GM` 单阶段流，无多阶段 K 循环可流水；窗口循环次数少（`k_W` ∈ 1–128，
  但 `k_W` 大时 `L_out` 小、任务数少），双缓冲收益有限。
- Developer 模式下编译器可自动 multi-buffer；若 Stage 4 实测显示 MTE2 未与 Vector 重叠，
  可在 `t` 任务循环或 `kw` 循环上探索 `T.Pipelined(num_stages=2)`（列入 Stage 4 候选，不改主结构）。

### 6.4 尾块处理

- **任务尾块**：`num_logical` 非 `num_kernels` 整数倍 → `if uid < num_logical` 丢弃越界任务。
- **通道尾块**：`real_c = min(CH, C - cb*CH)` 动态 copy/写回范围；padded lane 永不落 GM。
- **输出块尾**：`BL_real = min(BL, L_out - ow0)` 动态写回范围。
- **窗口尾块**：`iw` 守卫跳过界外位置（精确覆盖窗口，无掩码需求；窗口边界由编译期 `k/s/p/d` 与运行时 `o_w` 线性给出）。
- **空窗口（v1）**：无 in-bounds lane 的窗口由 `idx32==BIG` 子句置 `idx=0`（对齐源；§0.1/§9.3）；见 L0-15。
- **无 GEMM 式整除要求**（§5.3）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**自动同步（Developer 模式）**。

### 7.2 同步点说明

无手动同步点。Developer 模式下编译器在 `T.copy`（MTE2 搬运）与 v-prefix 向量 op（Vector 单元）之间
自动插入同步标志，`T.transpose`（VTransposeOp）亦由编译器编排。本算子无跨核通信、无 Cube/Vector 跨引擎依赖：

| 位置 | 同步 API | 理由 |
|------|----------|------|
| （无） | — | Developer 自动同步；本算子单引擎（Vector）、单阶段，无 `sync_block_set/wait` 需求 |

### 7.3 pass_configs 配置

`@tilelang.jit(out_idx=[1, 2], target="npuir")`；不设置自定义 `pass_configs`
（与 `lerp_tensor` / `logsumexp` / 同族池化先例一致）。
若 Stage 3/4 出现 UB 膨胀导致溢出（索引引入的 buffer 使预算更紧），可评估 `--enable-auto-multi-buffer=false`
（`CONST-capacity-910B2C` 关闭档膨胀 1.10–1.12）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（优先移植源仓参考实现/测试基准），
> **不得复刻 §0.6 的 NPU 算法**——保证验证独立性。

```python
import torch
import torch.nn.functional as F


def golden_max_pool1d_with_indices(x, kernel_size, stride=None, padding=0,
                                   dilation=1, ceil_mode=False):
    """PyTorch reference for max_pool1d(return_indices=True), per DESIGN.md section 0.1.

    Directly uses the source-repo test benchmark
    (``F.max_pool1d(..., return_indices=True)``).  Exactly 3D NCL; independent of
    the NPU algorithm (no transpose / no C-vectorisation / no span-load / no
    index accumulator / no specialisation).  Runs on CPU, returns
    ``(values, indices)`` with indices int64.
    """
    vals, idx = F.max_pool1d(
        x.detach().to(torch.float32),  # fp32 opmath domain for fp16/bf16 CPU ref
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )
    return vals.to(x.dtype), idx
```

（golden 以 fp32 中转后舍回输入 dtype，与 §0.1「fp32 比较域」一致；max 只选择不累加，故中转无损，
值 `atol=rtol=0` 成立；索引 int64 直接精确比对。golden 语义已由 `verify_equiv.py` 的 `semantics_evidence()`
与 E1/E1-idx 组对照本设计 NPU 公式复核。**注**：测试 harness 在 `dilation != 1` 时因 torch_npu aclnn 参数支持面较窄
而在 CPU 上求 golden 再搬回设备，与本 golden 的 CPU 实现一致。）

### 8.2 精度标准

| dtype | 值 atol | 值 rtol | 索引 |
|-------|---------|---------|------|
| float16 | 0 | 0 | int64 `torch.equal` 精确 |
| bfloat16 | 0 | 0 | int64 `torch.equal` 精确 |
| float32 | 0 | 0 | int64 `torch.equal` 精确 |

（**逐位精确**：max 是被选中元素的原值，无累加、无舍入误差；索引是整数契约。依据 §0.1 与源仓双输出池化套件。
模板通用档 1e-2/1e-4 在本算子不适用。**索引是免费 oracle**：值 `max_diff=0` 不能证明索引链正确，
int64 索引 `torch.equal` 必须独立断言。）

### 8.3 L0 门槛测试计划（`tilelang-op-develop` 展开为 L1/L2/Boundary）

| # | 用例 | 覆盖点 |
|---|------|--------|
| L0-1 | `(1,8,32)` k3s2p1 fp32 | manifest smoke（fp32 优先测）；通用路径基线 + 双输出 |
| L0-2 | `(2,8,64)` k3s2p1 fp16 / bf16 | smoke 另两 dtype |
| L0-3 | `(32,80,4096)` k3s3p0 fp16 | manifest `sincnet-speaker-local`；**容量压 BL=16** + persistent 5 504 单元 |
| L0-4 | `(64,256,128)` k128s1p0 fp16 / fp32 | manifest `textcnn-global`；**走一般路径**（`BL=1`）+ fp32 CH 折半 |
| L0-5 | `(16,128,2048)` k5s2p2 d2 ceil bf16 | manifest `ecg-cnn-dilated`；非单位 dilation + ceil + 重叠窗（BL=17，span 重读 ≈1.22×） |
| L0-6 | `(1,4,63)` k3 s=None p1 d2 fp16 | `stride=None`（= kernel）路径 + 非单位 dilation |
| L0-7 | `(1,4,97)` k5s3p2 ceil fp16/bf16 | 非重叠窗口 + ceil |
| L0-8 | **索引 first-hit 门禁**：窗口内多个相等最大值（含 `tie_two_max` 形），断言索引 = 首个命中 | §0.1 tie→first；N3 |
| L0-9 | **NaN 语义门禁**：窗口内置 NaN（首/中/尾多处、含两个 NaN）→ 值 NaN 且索引 = **last NaN** | §0.1 NaN 胜出 + last NaN；N2；不得依赖 `vmax` NaN 行为 |
| L0-10 | **±inf 语义门禁**：`{+inf,-inf}` 同窗 → `+inf`（非 NaN）；全 `-inf` 窗 → `-inf` 且索引 = 首个有效元素 | §0.1；`eq` 掩码不误伤 ±inf；N3 `first_valid` 消解 |
| L0-11 | **非连续输入** | op 层 `.contiguous()` 路径 |
| L0-12 | **通道尾块 / 输出块尾**：C=80（CH=64 + `real_c=16`）、C=4（CH=8 + `real_c=4`）；`L_out` 非 BL 整数倍（sincnet 1365） | `real_c` / `BL_real` 动态范围；padded lane 不落 GM |
| L0-13 | **前 padding 越界跳过**：`p_W>0` 使窗口起点 `iw<0`（如 `(1,8,32)` k3s2p1、`(1,4,63)` k3s2p1） | C1 钳制载入 + 逐 lane 守卫；索引不含负值 |
| L0-14 | **参数/输入校验**：非 3D / fp64 / `padding>k//2` / `stride=0` / `dilation=0` / 非正输出长度 | op 层拒绝路径 |
| L0-15 | **空窗口契约（v1 负向控制组）**：`(1,2,1)` k2s1p1 **d2**（lane `{−1,1}` 全界外）/ `(1,2,1)` k2s3p1 **d3 ceil**（lane `{−1,2}`） | `d_W≥2 ∧ p_W>0` 且窗口无 in-bounds lane；**断言设计行为 = 源契约 `(val,idx)=(−inf,0)`，NOT torch 相等**（该 regime torch 索引越界，不可作 oracle）；`verify_equiv.py` 的 `EW`/`empty_window_evidence()` 逐行留证 |
| L0-16 | **subnormal 用例（可选）**：窗口含 fp16/bf16 subnormal（如 `6e-8`），值精确且 `eq` 掩码不误伤 | `PL-1.22` 第三数据点的 subnormal 数据类；`T.vcast` 上转 fp32 后 subnormal 保持 |

---

## 9. 风险点与注意事项

### 9.1 已知约束

- 本算子为**纯 Vector**、单阶段、无跨核通信；无 Cube / L1 / L0C 路径。
- 循环上界（`k_W`/`BL`/`LSPAN`/`num_local`）全为编译期常量，**不需要** `max_bin_extent` 类机制。
- **索引引入额外 UB buffer 与更多 op**：容量判据（§4.5）与分核（§5.5）均已按新增 buffer 重算。
- `BL`/`LSPAN` 是 1D 新增的分块维度（2D 无）；`LSPAN` 由 `BL` 派生，二者必须与 §5.2 的一致口径同步。
- **C4/C4b 不采用**（§0.6 R5）——§3/§5/§6 无任何特化分支，一致性由本声明维护。
- **空窗口契约（v1）**：`d_W≥2 ∧ p_W>0` 下窗口可能无 in-bounds lane；设计对齐源（`idx=0`，§0.1），
  该 regime torch 索引越界不可作 oracle（§8.3 L0-15）。空窗口子句**不新增 UB**（复用 `cfinal` 布尔缓冲）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| **`T.vcmp(...,"ne")` 误用作 NaN 探测** | 想用 `v!=v` 判 NaN | `ne` 是 **ordered-NE**（NaN 恒假）→ 掩码恒假、静默失效 | **必须用 `eq`**（`x==x` ⟺ 非 NaN）；本设计块级 `eq` 掩码 + `flag` 判据（§0.6 R2） |
| **依赖 `T.vmax` 的 NaN 行为** | 直接 `vmax` 累加就当 NaN 会胜出 | 行为**未文档化**，跨版本不稳 | 显式 NaN 追踪（`flag`/`nan_idx>=0`），`verify_equiv.py` N2N3-neq 组证明结果与 `vmax` 的 NaN 约定无关 |
| **first-hit 次序错误**：把 winner 比较放进值累加之前 | 单遍融合（用前缀最大值判等） | 索引退化为「首个局部峰值」；与 §1.6.1 N3 前提冲突 | **两遍法**（第一遍值/NaN/last-NaN，第二遍 first-hit，`acc` 定稿）；§3.3/§6.2 冻结；`verify_equiv.py` N2N3-ord 守卫证明误形态会被检出（39/72） |
| **`T.vcast` 同 dtype（fp32→fp32）** | fp32 输入时 `vcast(row_dst, row_f32, "rint")` | 退化为 round-to-integer → **数据错乱** | 按 dtype 在 trace 期分派：fp32 走 `T.copy`，fp16/bf16 走 `T.vcast`（§3.2/§3.3 的 `is_f32` 分支） |
| **`@tilelang.jit` 传 `compile_flags`** | 照搬源 GPU 的 `["-O3","-DENABLE_BF16"]` | 本分支 jit 签名无该参数 → 类型错误 | 移除 `compile_flags` |
| **`T.vbrc` destination 写成 buffer region** | `T.vbrc(iw, fv[...])`（带运行时偏移的 UB region，且在 `T.serial` 内） | 写入被**静默丢弃**；int32 形态还会读未初始化 UB → 向量核 fault（core dump，`TRAP-npuir-vbrc-region-dropped`） | `T.vbrc` 的 dst **必须是整 buffer**（本设计 `fv` 为独立 `(1,CH)` 整 buffer，其余常量均整 buffer 广播；逐 `owl` 填不同标量时用 staging `(1,CH)` + `T.copy` 三件套） |
| **`T.vbrc` 标量 operand 用裸 Python 数** | `T.vbrc(-inf, ...)` / `T.vbrc(BIG, ...)` | 命中前端 rank 断言 | 用 **typed PrimExpr**：`-T.infinity("float32")` / `T.int32(L_in)` |
| **`arange` 派生下标 pattern 用于 argmax 索引** | 想用 `T.arange` 构造线内下标再 `vselect+reduce` | **in-chain 读回损坏**（`TRAP-npuir-arange-pattern-with-whole-plane-reduce`）→ 索引静默错、值仍精确 | **本设计不使用 arange pattern**（C4 不采用，走一般路径）；索引来自逐 lane `vbrc(iw)` + `vselect`/`vmin` |
| **把 `for k in range(n)` 当 Python 编译期展开** | 在 `@T.prim_func` 内写 `for kw in range(k_W)` | `range` 被解析为 **serial TIR 循环**；用该 var 索引 Python 列表在 trace 期失败 | 统一用 `T.serial(k_W)` / `T.serial(BL)` + 编译期常量上界 |
| **`T.vcast` 形状不一致 / 源为 GM** | 直接 `T.vcast(GM_region, ub_f32)`，或 dst 形状不同 | BiShengHIR 阶段失败 / 广播形状报错 | 先同 dtype `T.copy` GM→UB，再 `T.vcast` UB→UB（同形）；换布局用 `T.transpose` |
| **`T.transpose` 多轴 / 先 reshape** | `permutation=[2,1,0]`（3-cycle）或 reshape→transpose | verify 失败 / **mis-compile 数据错乱** | 只做二轴交换（本设计用 `[1,0]`）；禁止 reshape 后再 transpose |
| **动态偏移 UB subview** | 在内层循环对 task 级 buffer 做运行时偏移切片**拷贝** | auto-multi-buffer 非支配 IR（`operand #2 does not dominate this use`） | 本设计载入用**静态长度 + 钳制源起点**；粒度读用 vmax/vmin/vmul 的**运行时切片**（与同族前作同形，已过） |
| **UB 溢出** | 长 `L_in` + 大 `BL`/`CH`，且索引 buffer 叠加 | 编译期 UB overflow | 容量判据（§4.5，系数 `69+2·eb`）联合压 `BL`/`CH`（sincnet → BL=16）；必要时评估 `--enable-auto-multi-buffer=false` |
| **空窗口子句顺序颠倒** | 把 `vcmp(idx32,bigv)` 写在 `out32` 选择**之前** | `cfinal` 被提前覆写为空窗口掩码 → `out32` 值选择读到错误条件 | 空窗口子句必须排在 `out32` 计算之后（§3.3 实现级约束⑥）；`cfinal` 复用次序在 §6.2 冻结 |
| **`real_c` / `BL_real` 尾块漏处理** | `C % CH != 0` / `L_out % BL != 0` | 越界读写 / 落盘错值 | copy/写回用动态范围；padded lane 不落 GM |
| **`nan_idx` 初值错误** | 初值取 0（而非 -1） | `nan_idx>=0` 恒真 → 所有窗口判为含 NaN | `nan_idx` 初值必须 `-1`（哨兵）；`has_nan` 用 `nan_idx>=0` |
| **载入钳制 `s0` 越界** | `s0` 未按 `[0, L_in-LSPAN]` 钳制 | GM 越界读 | `s0 = max(0, min(base, L_in-LSPAN))`；`LSPAN ≤ L_in` 由定义保证 |

### 9.3 特殊场景处理

- **非整除分块**：无 GEMM 式要求；任务/通道/输出块/窗口四类尾块分别由 `uid` 守卫 / `real_c` / `BL_real` / `iw` 守卫处理（§6.4）。
- **极小 shape**：`C < 8`（如 C=4）→ `CH=8` + `real_c` 收尾；`L_out=1`（textcnn）走一般路径（`BL=1`）。
- **长 L**：`L_in` 达 4096 → `BL` 由容量压到 16；`L_in` 极大时若 `CH=8` 仍超容量则 `BL=1` 兜底。
- **混合精度**：三 dtype 均支持；bf16 经 `vcast` 上转 fp32 计算、末尾 `vcast` 舍回。
- **`padding>0` + `ceil_mode`**：末窗部分落在 padding，由 `iw` 守卫按 $-\infty$ 处理，与 torch 一致（**非空窗口**）。
- **空窗口（v1）**：`d_W≥2 ∧ p_W>0` 且窗口 lane 全界外 → 设计输出 `(val, idx)=(−inf, 0)`（对齐源；§0.1）；
  该 regime 与 torch 不同（torch 返回越界索引），不作 torch 等价断言（§8.3 L0-15）。
- **`stride=None`**：op 层解析为 `s_W = k_W`。
- **NaN / ±inf**：见 §0.1 / L0-9 / L0-10；实现级次序与形态检查见 §9.2。

### 9.4 待确认 / 风险登记（供 Stage 2 复核）

1. **发射项不确定性（最高优先）**：本算子每元素约 3 条向量 op（值 `vmax` + last-NaN `vmax` + first-hit `vmin`，
   另加 `vcmp`/`vselect`/`vbrc`），是同族单输出（~2 条/元素）的 **~1.5 倍**，且 `ecg`/`sincnet` 的 `L_out·k_W` 使 op 数达 0.5–0.8 M。
   `CONST-vector-launch-overhead` 0.5µs/op **不适用于窄 op 外推**（§1.6.0 估算行已标注「未文档化假设」）。→ §1.6.3 已给出主选/备选/裁决计划。
2. **设备差异**：roofline 引用的带宽/发射/容量常数均绑定 Ascend910B2C（48 向量核），本设备为 910B3（40）——
   跨型号复用属**未实测假设**，只作量级下界。
3. **toolchain stamp 差异**：本任务 stamp = `ba60e99c`（dev root build）；同族前作实证 stamp 为
   `ddf4255a`/`417dae9`（max_pool1d）与 `ba60e99c`（adaptive_max_pool2d_with_indices）。
   本设计沿用的实现级结论（**fp32 同 dtype `vcast` 误编译、`reduce_min` 判 NaN、`compile_flags` 不受支持、
   `vbrc` region 丢弃、`arange` pattern 断裂**）跨 stamp/跨设备复用属未实测假设；采用**防御性设计**：
   ① NaN 探测只用 `eq`；② `vbrc` 只写整 buffer；③ `T.serial` 用编译期常量上界；④ `vcast` 保证同形且源为 UB；
   ⑤ fp32 路径用 `T.copy`；⑥ 不使用 arange pattern。即设计本身对上述陷阱**结构免疫**——若某条已失效，不影响主结构。
4. **`fv` 广播形态的落地细节**：伪代码用 `T.vbrc(T.int32(iw), fv)`（`fv` 为独立 `(1,CH)` 整 buffer）；
   Stage 3 若遇 `vselect` 操作数 rank 不匹配，回退方案为「staging `(1,CH)` + `T.copy`」或 `fv` 提升为 `(BL,CH)` 整 buffer。
   属实现级形态，不影响算法与语义（§3.3 已冻结「整 buffer 非 region」原则）。
5. **两遍法 vs 单遍法**：本设计为确保 first-hit 等价前提（`acc` 定稿）采用两遍法（均在 UB 内，无 GM 二读）。
   若 Stage 4 显示第二遍的 `vcmp`+`vselect`+`vmin` 是发射瓶颈，可评估「单遍 last-record `gt`+`vmax`」变体
   （需正确处理全 `-inf` 窗口的 seed）——列入 Stage 4 候选，**不改语义**。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/max_pool1d_with_indices/_max_pool1d_with_indices_kernel/
├── DESIGN.md               # 本设计文档
├── verify_equiv.py         # §1.6.1 等价性机器验证脚本（D-1，采样版）
├── _max_pool1d_with_indices_kernel.py   # 算子实现 + 分层测试（Stage 3 产出）
└── RETROSPECTIVE.md        # Stage 复盘（Stage 1 起追加）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 设计文档 |
| `verify_equiv.py` | 已完成 | 等价性机器验证（**1074** 项检查全 `EQUIV_PASS`；按 S3 采样，含分支条件采样 + **空窗口 `EW` 分支（v1）** + 语义证据 + 特化 value-only 数学论证） |
| `_max_pool1d_with_indices_kernel.py` | 待实现 | 算子实现（Stage 3） |
| 分层测试（L0/L1/L2/Boundary） | 待实现 | 内嵌于实现文件或 `testing/`（Stage 3） |

### 10.3 命名规范

- 项目目录名: `max_pool1d_with_indices`（snake_case）
- 算子目录名: `_max_pool1d_with_indices_kernel`（源 factory 函数名）
- 实现文件: `_max_pool1d_with_indices_kernel.py`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）
2. ⬜ Golden 函数（§8.1，Stage 3）
3. ⬜ 算子实现（`_max_pool1d_with_indices_kernel.py`）+ 与 Golden 精度比对（值 `atol=rtol=0` + 索引 int64 `torch.equal`）
