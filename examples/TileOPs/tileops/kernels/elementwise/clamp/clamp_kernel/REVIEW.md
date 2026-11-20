# _make_clamp_tensor_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/clamp/_make_clamp_tensor_kernel/DESIGN.md
- 任务类型: 迁移（migration-harness）
- 源算子: `/home/developer/TileOPs/tileops/kernels/elementwise.py`（`_make_clamp_tensor_kernel` @ L3018–L3224、`ClampTensorFwdKernel` @ L3227–L3265；Tensor-bound 变体，非 scalar 变体 `_make_clamp_kernel`/`ClampFwdKernel`）
- 检视维度: 9 项（维度 0 源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 3（均为非阻塞 warn，见各维度说明；不构成问题列表）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: PASS（`python3 examples/clamp/_make_clamp_tensor_kernel/verify_equiv.py` → `RESULT: ALL 23 ITEMS EQUIV_PASS`）。逐项与 DESIGN.md §1.6.1 内嵌结果表核对：23 行 item 名称、样本数 262,160、`max_abs=0.000e+00`、违反率 0.0000、结论 `EQUIV_PASS` **逐项一致**。非门禁诊断亦一致（reordered fp32 max_abs 4.973e+00/viol 62,720；fp16 5.152e+00/viol 63,167；maxNum fp32 0.000e+00/viol 2——脚本另打印 maxNum fp16 viol 2，DESIGN 表未列该行，属遗漏展示非结果不符）。
- **设计期设备探针（额外证据，非门禁项）**: 独立重跑 `python3 history_version/design_probe_clamp_nan.py` → **PROBE PASS**（[1] 张量边界 `T.vclamp` == `torch.clamp` bit-exact NaN-aware；[2] NaN 置于 x/lo/hi 均传播；[3] `lo>hi→hi`；[4a/4b] 张量边界 `T.vmin`/`T.vmax` NaN 传播）。该前提是采纳项 §1.6.1 #1 的设备侧证据，本次实跑复现。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/clamp/_make_clamp_tensor_kernel/DESIGN.md`：
  - `ub_budget`: **skip**（"no per-level buffer tables parsed from the memory plan"）→ **人工已核**：§4.5 无条件全量分配手工预算 fp32 = 8×2048×4B = 65536 B (64KB)、fp16/bf16 = 3×4096×2 + 4096×2 + 3×4096×4 + 4096×4 = 98304 B (96KB)，均 ≤ UB 192KB；按 9362 未重验的 ~1.7x 膨胀上限 ≈109KB / ≈167KB 仍 ≤192KB（§4.5 已列 fallback）。
  - `l0c_budget`: **skip**（"no affirmative Cube-path API"）→ 人工已核：纯 Vector，无 `T.gemm`/`load_nd2nz`/`store_fixpipe`/`Scope("Cube")`，L0C 不适用。
  - `core_split`: **skip**（insufficient parsed dims）→ **人工已核**：§5.5 三要素齐全；物理 Vector 核数 **实查**（本会话独立复核 `NPUUtils.get().get_aicore_num()=20`、`get_aivector_core_num()=40`）；`num_kernels = min(num_logical, 40)`，`num_local_tasks = ceildiv(num_logical, num_kernels)`（编译期常量）；256M 档 65536 逻辑核 → 1639/1638 任务/核（差 1，0.06%）。
  - `r3_metrics`: **pass**（"all four R3 metrics present"）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**核对方式**：亲自 Read 源文件 `/home/developer/TileOPs/tileops/kernels/elementwise.py` L3000–L3265（含 `_make_clamp_tensor_kernel` 全部三变体与 `ClampTensorFwdKernel`、以及其边界上的 scalar 变体 L2938 起以确认未误取），并核对 Stage 0 抽取件 `_clamp_fwd_kernels.py` 与源逐字一致。

- **语义理解正确性 — pass**：§0.1 fp32 中转（源非 fp8 分支 L3150–L3152 `T.cast(...,"float32")` → L3153–L3154 `T.min(T.max(x32,lo32),hi32)` → L3160 `T.Cast(dtype,r)` 单次舍回）与源码逐语句一致；§0.1 NaN 传播/`lo>hi→hi`/I/O 一维与源注释 L3034–L3039 及三变体 `main` 签名一致；§0.1 契约 golden `torch.clamp(x.float(), lo.float(), hi.float()).to(dtype)` 与 `test_special_elementwise.py` L115–L116 `ref_program` **逐字一致**；`has_min=has_max=False` 抛 `ValueError`（源 L3041–L3044 / L3241–L3244）与 `test_clamp_both_none_rejected` 一致。§0.2 输出一维依据（源 L3139/L3173/L3203 `y: T.Tensor((N,), dtype)`）与 `ClampFwdOp._eager_forward` L136 `view` 佐证一致。
- **算法解读完整性 — pass**：§0.3 六步分解覆盖源码全部计算语句（host 预广播展平 → grid 配置 → 三段 `T.copy` → `T.Parallel` 逐元素 cast/min/max/守卫/舍回 → `T.copy` 搬出 → host `view`）；数据流图覆盖 x/lo/hi/x_reg/lo_reg/hi_reg/y 全部 buffer；host 侧（`@tilelang.jit` 工厂闭包 + `lru_cache` + `out_dtype`）已列出；分支矩阵（{is_fp8}×{both,min-only,max-only}）与 `out_idx`（both L3132 [3]、min-only L3167 [2]、max-only L3197 [2]）核对无误；L3139/L3150-3160/L3157-3159 行号引用准确。
- **优化手段覆盖完整性 — pass**：§0.4 八项（register_copy / coalesced 连续段 / thread×npt / fp32 计算域 / NaN 守卫 / has_min-has_max 特化 / 一维展平 / lru_cache）与源码显式结构逐项对应；确认源**未**使用 SMEM tiling / warp shuffle / 异步流水 / swizzle / persistent（对照源码 L3131–L3165 非 fp8 分支仅 `alloc_fragment`+`T.copy`+`T.Parallel`），未漏识别、未臆造。
- **耦合性判定合理性 — pass**：§0.5 每条具四态处置 + 依据。抽检关键项：`threads=` kwarg 判**舍弃**（npuir `T.Kernel` 不接受，`vec_add_1d.py` L27 `T.Kernel(n_num, is_npu=True)` 佐证）— 合理；NaN 守卫判**舍弃**（NPU 原生传播，非「实现简单」）— 有设备探针支撑；`alloc_fragment`→`alloc_ub` 判**等价替换** — 合理。判定统计（保留 11 / 等价替换 2 / 重设计 3 + 评估性 1 / 舍弃 2）自洽。
- **NPU 重设计可行性 — pass**：R1–R5 各含「源方案 → NPU 新算法 → 语义保持论证」。R1 逐元素映射不变、求值次序（先 max 后 min）保留；R2 fp16/fp32 承接源同域、bf16 经 `vcast` 强制中转；R3 persistent（映射任务独立、无竞写）；R5 无守卫依赖设备原生传播。均满足 NPU 约束（一维 Kernel、UB 层级、无 L0C、静态边界）。**多输入广播处理策略（重点核对）**：R4 以三候选量化表给出处置——方案 A（host 物化 + 扁平一维，采纳）、B（核内 2D+vbrc，否决）、C（核内 stride-0，否决）。核对 `docs/Tilelang.language/比较操作/T.vclamp.md` §2.2.2（张量边界需与 src/dst 逐元素对应，通常同 shape）与 `T.vmin.md` §2.2.2（自动广播仅「有且仅有一个维度不同且为 1」）确认 B 的否决前提成立；对照 `examples/elementwise/vec_add_auto_brc.py`（该例仅示单轴行广播，非一般三路广播）不构成与 R4 结论矛盾。方案 A 与冻结 Op 契约 `ClampFwdOp._expand_flat`（`ops/elementwise/clamp.py` L115–L119 `expand`→`contiguous`→`view(-1)`）及 wrapper 调用点（`clamp.py` `forward` 分别传 `(x,lo,hi)`/`(x,lo)`/`(x,hi)`）一致。
- **设计与决策一致性 — pass**：§1.4/§1.5 数据流、§3.3 伪代码、§4 内存、§6 循环均描述 §0.5/§0.6 决策后的 NPU 算法（无「§0 判重设计、§3 照抄源」的脱节）；无守卫形态贯穿 §1.4/§3.1/§3.3/§6。
- **golden 独立性 — pass**：§8.1 golden 以 §0.1 源语义为依据（`torch.clamp(x.float(),lo.float(),hi.float()).to(dtype)`，即源仓 `ref_program`），明确声明不复刻 NPU 算法（vclamp/fp32 中转/persistent/无守卫均未进入 golden）。

### 1. API 可行性 — pass

- **API 存在性 — pass**：§3.2 全部 API 均在 `docs/Tilelang.language/` 与 `examples/` 找到佐证并亲自打开核对：`T.Kernel(grid, is_npu=True)`（`vec_add_1d.py` L27 / `T.vmin.md` §2.4）、`T.alloc_ub`（`T.alloc_ub.md` §2.2.2「1D~2D」）、`T.copy`（`T.copy.md` §1 支持 GM↔UB / 切片 / 尾块，dtype 矩阵含 bf16 √）、`T.vclamp`（`T.vclamp.md` §2.1/§2.2.2/§2.4 + §3 前端展开 `npuir_max`→`npuir_min`）、`T.vmin`/`T.vmax`（§2.1/§2.4）、`T.vcast`（`T.vcast.md` §2.2.1）、`T.min`（`T.min.md` 标量）、`T.ceildiv`（`T.ceildiv.md`）、`T.serial`+grid-stride+`if` 边界（`T.vmin.md` §2.4 官方 persistent 示例）。**负向断言核对**：bf16 缺 `vclamp`/`vmin`/`vmax` 逐格回文档核对 dtype 矩阵——`T.vclamp.md` L28 / `T.vmin.md` L27 / `T.vmax.md` L26 的 Ascend 行 bf16 列**均为 ×**，断言成立。
- **API 与模式匹配 — pass**：Developer 模式，无手动同步；`T.vclamp.md` §2.4 示例即 `@tilelang.jit(target="npuir")` Developer 形态。
- **v-prefix 优先 — pass**：全部为 v-prefix（`vclamp`/`vmin`/`vmax`/`vcast`），无 `npuir_xxx`。
- **（warn，非阻塞）** `T.alloc_ub.md` §2.1 行文将 `alloc_ub` 标注为 `[Expert mode]`、`alloc_shared` 标注 `[Developer mode]`，而本设计 Developer 模式全用 `alloc_ub`。**判定不影响结论**：同族已验证先例 hardtanh（`ascend_mode="Developer"`，`_make_hardtanh_kernel.py` L144–L147）与 `vec_add_1d.py` L28–L30 均在 Developer 下用 `alloc_ub` 且 L0–Boundary 全绿，`clamp.py` wrapper `ascend_mode="Developer"`。建议 §3.4 附一句「alloc_ub 在 Developer 形态经 hardtanh/vec_add_1d 实证可用」以免 Stage 3 误改。

### 2. 内存层级规划 — pass

- **搬运路径完整 — pass**：§4.4 纯 Vector 单阶段 GM→UB→GM，无 GM→L0 越级、无 L1/L0C 参与（本算子无 Cube）；cast 全程 UB 内，无中间 GM 往返。
- **UB 预算 — pass**（机械 `ub_budget` skip，**人工已核**）：§4.5 手工预算 fp32 64KB / fp16·bf16 96KB（无条件全量分配，含 4 个空占 `*32` buffer）≤ 192KB。独立复算：fp32 = (3+1)×2048×4B（8 buffer ×4B，共 8×2048×4 = 65536 B）；fp16/bf16 = 3×4096×2 + 4096×2 + 3×4096×4 + 4096×4 = 24576+8192+49152+16384 = 98304 B。均与设计一致，且 block_size 上界反推（fp16 ≤8192、fp32 ≤6144）算术正确。
- **L0C 容量 — n/a**：无 Cube（`l0c_budget` 机械 skip，人工确认）。
- **Buffer 层级标注 — pass**：§4.3 逐 buffer 标注 UB（无 L1/L0）。

### 3. Tiling 策略 — pass

- **Block 划分明确 — pass**：§5.2 block_size = 4096（fp16/bf16）= threads(512)×npt(8)、2048（fp32）= threads(512)×npt(4)，与冻结 wrapper `clamp.py` `_DEFAULT_BLOCK_SIZE = {fp32:2048, fp16:4096, bf16:4096}` 一致，选择理由 5 条（对齐/向量宽度/UB/粒度/源语义承接）。
- **对齐约束 — pass**：block_size×elem_bytes = 8KB（32B 的 256 倍）；tile 起点 `off = block_id×block_size` 恒 8KB 对齐。
- **非整除处理 — pass**：纯 Vector 非 GEMM，§6.4 尾块 `tail = T.min(block_size, N−off)` 零起点切片（`vec_add_1d.py` L31–L38 先例）。
- **分核策略（物理核数适配）三要素 — pass**（机械 `core_split` skip，**人工已核**）：
  - ① 逻辑核数计算：§5.5 表按 manifest 4 workload 逐条给出（512/256/8192/4096/16384/65536）。核对 manifest `ClampFwdOp.workloads`（1024²/4096²/8192²/16384²）与 block_size → 一致。
  - ② 物理核数依据：**实查**，记录查询代码（`NPUUtils.get().get_aicore_num()` / `get_aivector_core_num()`）与返回值 20/40；本会话独立复核同值（20 / 40）；纯 Vector 翻倍（core-split-strategy.md §1 要素②）正确；未用跨 SoC 的 `CONST-aicore-910B2C`（24）代替。
  - ③ 规模判定与分核方案：极大规模（256–65536 >> 40，且 UB 上限使 block_size 无法把 2²⁰ 缩到 ≤40）→ `num_kernels = min(num_logical, 40)` + 核内 `T.serial(num_local_tasks)` 静态边界 + grid-stride + `if block_id < num_logical`。**核数与 block 自洽性复算**：256M fp16 num_logical=65536、num_kernels=40、num_local_tasks=ceil(65536/40)=1639；末核任务数差 1（1639 vs 1638，0.06%），无负载不均问题；静态边界由 N/block_size/40 均编译期常量保证。
- **（warn，非阻塞）** §3.3 伪代码 `num_kernels = min(num_logical, vector_cores)` 用 Python 内建 `min` 作用于 `T.ceildiv(N, block_size)`（可能是 TIR 表达式，非 Python int）。实现期若 `num_logical` 为 PrimExpr，`min()` 会因无法比较而报错。建议 Stage 3 以 Python 整数算术计算 `num_logical/num_kernels/num_local_tasks`（如 `(N + block_size - 1)//block_size`）保证编译期静态边界；设计意图（编译期常量）本身正确，hardtanh 亦用 `T.ceildiv` 作循环上界（不涉及 Python `min`）。

### 4. 技术约束检测 — pass

- **三维 Kernel — n/a**：源即一维 `ceildiv(N, block)`，NPU 一维 `T.Kernel(num_kernels)`。
- **GEMM 非整除 — n/a**：无 GEMM。
- **L0C 溢出 — n/a**：纯 Vector 无 L0C。
- **`threads=` kwarg — pass**：§0.5/§3.5.1 判舍弃，`is_npu=True` 取代。
- **bf16 Vector 算术缺失 — pass**：§0.6 R2 经 `vcast` fp32 中转（依据已逐格核对）。
- **Host 侧输入操作约束 — pass（含张力披露）**：§3.5.1 标注「部分涉及」。核对 `ascend-constraints.md` §4「host 侧仅允许 stride/shape 视图操作、约束覆盖 kernel 调用前后完整 host 路径」——`_expand_flat` 的 `.contiguous()` 会物化被广播操作数。**判定不阻塞**：该物化属**冻结的 Op 层既有契约**（`ops/elementwise/clamp.py` 模块头 O6「Op-specific flow (broadcast, `_expand_flat`, ...) preserved unchanged」；wrapper `clamp.py` 亦声明「Op layer broadcasts ... before dispatch」），不在本 kernel 重设计范围；kernel 内 host 侧零数据操作（仅 N/block_size 元数据）；manifest 4 workload 三输入同 shape → `expand` 为视图、`contiguous()` 对已连续张量无拷贝（`_expand_flat` L117 仅形状不等时才 expand）→ 基准路径零物化。§9.1 R-3 已完整披露根因/影响/升级路径（如需严格合规须改 Op/wrapper 契约，属 conductor 决策），处理透明。

### 5. 循环与同步 — pass

- **循环结构 — pass**：§6.1 明确 `T.Kernel`（persistent 核间）/ `T.serial`（核内静态边界）/ 向量化（元素级），无 `T.Parallel` 逐元素循环。
- **同步与模式匹配 — pass**：§7 Developer 自动同步，无手动 `sync_block_set/wait`；与 `clamp.py` `ascend_mode="Developer"` 一致；tile 间无跨核依赖。
- **尾块处理 — pass**：§6.4 `tail = T.min(block_size, N−off)` 零起点切片，来源 `vec_add_1d.py` L31–L38；不采用 host padding（合规）。
- **（warn，非阻塞）** 同维度 3 的 Python `min` 注记（编译期常量计算建议）。

### 6. 验证方案 — pass

- **Golden 函数 — pass**：§8.1 `torch.clamp(x.float(), lo.float(), hi.float()).to(x.dtype)`，与 `test_special_elementwise.py` `ClampTensorTest.ref_program`（L115–L116）一致，独立于 NPU 算法。
- **L0 测试计划 — pass**：§8.3 十条（smoke 全 dtype / 大规模 / NaN 逐位 / `lo>hi` / 单边界 / 广播与 0-dim / 尾块非整除 / 极小 shape / 端点极值 / persistent 边界），覆盖 manifest workload 与契约测试形态（(4,8)/(1,8)/(4,1)、(4,8)/()/()、NaN `equal_nan=True`）。
- **精度标准 — pass**：§8.2 fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2，与源仓 `_get_tolerances`（L91–L96）逐值一致；NaN 用例 atol=rtol=0 逐位。

### 7. 完整性与一致性 — pass

- **章节齐全 — pass**：模板 11 章节齐全，迁移类 §0 且 0.1–0.7 小节齐全。
- **无占位符 — pass**：全文不含模板占位标记、待办标记或未完成填充标记（grep 确认；`⬜` 为交付状态标记，非占位符）。
- **同类引用真实 — pass**：§3.5.3 引用的 5 个路径全部实存（hardtanh 实现、`vec_add_1d.py`、lerp DESIGN、`clamp.py`、`example_elementwise_add.py`）；§0/§8/§10 引用的 `_clamp_fwd_kernels.py`、`ops/elementwise/clamp.py`、`test_special_elementwise.py`、`elementwise.yaml`、`bench_independent_elementwise.py` 均实存。
- **内部一致 — pass**：§3.2 API 映射、§4 内存规划、§5 Tiling 三处相互一致（block_size、out_idx、buffer 数、核数）；§0.4→NPU 承接总表无静默丢弃。

### 8. 算法优化分析 — pass

- **§1.6 章节存在 — pass**：含 1.6.0 调研 / 1.6.1 等价优化 / 1.6.2 向量化替代 / 1.6.3 轴与布局四子节。
- **算法调研存在性与完整性 — pass**：R1 候选表**含基线**（候选 0 = 源公式）并评估 6 个候选；R2 有明确结论（"天然单遍，无需在线变体"，附结构判据）；R3 复杂度表覆盖**四口径**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值）且基线在表；R4 逐候选对照六项亲和清单。调研深度与「单步逐元素类」匹配，且明确写出「无更优替代」的调研范围（algorithm-candidates `ALG-elementwise` 条目 + 恒等变形自查 + `examples/elementwise/` + pattern-library 预注入条目 + 源结构分析）。
- **调研结论正确性（独立复核）— pass**：
  - ① 等价性：采纳项 #1（消除守卫）在「max/min 传播 NaN」语义下逐位恒等——独立推演成立（若任一操作数为 NaN，`max`/`min` 已将其传播到 r，三步 guard 谓词恒假）；前提经设备探针**独立重跑 PROBE PASS** 佐证（张量边界 `vclamp`/`vmin`/`vmax` NaN 逐位传播 + `lo>hi→hi`）。
  - ② 复杂度独立复算：基线 2N FLOPs（max+min）、4N·elem_bytes 访存（3 读+1 写，无中间 GM 往返）、1 遍、tile 级缓冲——与 manifest roofline `flops=N / bytes=4·N·elem_bytes`（口径差已注明：manifest 按 1 次融合 compare-and-select，本设计按 max+min 2 op）一致；D-2 估算复算 16M fp16 = 4·2²⁴·2B / 1.06 TB/s = 126.6 µs、16M fp32 = 253.2 µs、256M fp16 = 2026 µs，与设计数字一致；`CONST-ascend910-9362-profile` 的 BitwiseNotFwdOp 对账（64M 487.81µs→1.1006 TB/s、256M 2020.44µs→1.0629 TB/s）逐值核对无误。
  - ③ 亲和性：基线对 Vector/容量/对齐/静态边界全亲和；容量淘汰引用 `CONST-capacity-910B2C`（910B2C 24 核条目已注 9362 不适用，设计改用本机实查 20/40）；`CONST-vector-launch-overhead` 0.5µs/op 明确标注「9362 未重验、待重验假设」。
  - ④ 负向断言复核：R2「无在线变体」与 algorithm-candidates `ALG-elementwise` R2「逐元素映射天然单遍」及源码（§0.4 无 online 手段）一致，不构成矛盾；R1 未漏已知候选（`ALG-elementwise` R1 的 CSE/除法乘倒数/cast 链合并已分别由 E4/E5/E2 覆盖；clamp 族特有的顺序敏感候选 1/3/4 已逐一机器证伪）。未发现明显漏掉的已知候选。
  - ⑤ 并列候选：R3 中基线为唯一四口径并列最优，无纸面拍板风险；R4 候选 5 因边界为张量需运行期归约判定而淘汰，有量化依据。
- **调研→设计一致性 — pass**：§1.6.0 选定「基线公式单遍 Vector 算法 + 采纳候选 6」，实际驱动 §1.4（单遍逐元素三输入向量算法、无守卫）、§3.1（以优化后无守卫公式为输入）、§6（无逐元素标量循环）；与 §0.5/§0.6 处置一致。**源优化意图承接**：§0.6 末尾总表逐项承接 8 项源优化，其中 #5 NaN 守卫被替换后明确说明「源为正确性补丁、NPU 由硬件语义承接同一正确性」（非静默丢弃）。
- **数学等价优化完整性 — pass**：§1.6.1 采纳 1 项（#1 守卫消除）具四要素「原式（8 op）→ 优化后（2 op）→ 等价性论证 → 收益估算（6 op/元素消去，8→2，-75% 向量指令；带宽口径不变）」；#2 为契约域保留（非新增采纳），未采纳项 E1–E6 逐项记录理由。
- **等价性论证正确性 — pass**：独立推演 #1 恒等成立（见上）；过程无未论证的舍入/次序变更；与 §8.2 精度标准口径一致；容差内等价已在 `verify_equiv.py` 中区分「门禁（propagating 语义）」与非门禁诊断（maxNum 语义 2 点违反，证守卫仅在 CUDA 场景必要）。
- **向量化替代完整性 — pass**：§1.6.2 盘点 8 个计算点（3 类逐元素点 #1–#3 已替代/消除，5 处标量/循环 #4–#8 保留），与 §3.3 伪代码、§6 循环结构交叉核对一致——kernel 内无任何逐元素标量循环（§6.1 明示）。
- **不可替代理由充分性 — pass**：保留项理由具体到本算子（block 索引 / 依赖动态 shape 的尾块边界 / persistent tile 级调度循环 / host 元数据 / host 广播预处理），非「实现简单」；#6 引用 core-split-strategy §1 要素③ 与 `T.vmin.md` §2.4 官方结构。
- **替代 API 可行性 — pass**：`T.vclamp`/`T.vcast` 用法有 `T.vclamp.md` §2.4 / `T.vcast.md` §2.2.1 与 hardtanh 先例佐证，均为 v-prefix。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 单轴豁免（一维展平）+ 广播侧候选矩阵 4 行，逐行量化（整除性/尾 lane 浪费/repack/UB/代价）；弃选理由为量化+契约+API 文档（非「未考虑」）；对照了源码隐式轴选择（`T.Parallel(threads,npt)` 视为硬件输入非结论）。弃选论证前提与所引文档一致：`T.vclamp.md` §2.2.2（张量边界通常同 shape）、`T.vmin.md` §2.2.2（仅单维自动广播）、`T.copy.md`（无 stride-0/gather 形态，设计标注该代价论断为「未文档化假设」）。
- **布局决策与实现一致 — pass**：选定一维 `(block_size,)` UB + 展平 N 轴，§3.3/§4.3/§6 三方一致（buffer 形状、UB 预算、循环结构）。
- **与下游章节一致 — pass**：§3.1 以优化后（无守卫）公式为唯一输入；§6 无 §1.6.2 未论证的逐元素标量循环。
- **（warn，非阻塞）** §0.6 R4-B/C 的核内广播否决可补引 `examples/elementwise/vec_add_auto_brc.py`（示单轴行广播经 v-op 操作数切片实现），并以一句话澄清「单轴可承接、一般多轴（两维同时不同 / 0-dim 沿全轴）仍需逐操作数 `T.vbrc` 展开」，使否决论证更完备。该补充不改变结论（冻结契约 + manifest 零广播两点理由已足够）。

## 通过说明

所有 9 个维度均为 pass（含维度 0 与维度 8 两个迁移/算法阻塞级维度）。维度 0 的语义/算法/优化/耦合性/重设计/一致性/golden 独立性均已亲自 Read 源算子（L3000–L3265）逐项核对；维度 8 的采纳优化（消除 CUDA NaN 守卫）经独立推演 + `verify_equiv.py` 23/23 EQUIV_PASS 重跑 + 设计期设备探针独立重跑 PROBE PASS 三重佐证；bf16 缺 `vclamp`/`vmin`/`vmax` 的负向断言已逐格回文档 dtype 矩阵核对（均为 ×）；复杂度四口径已独立复算并与 manifest roofline、`CONST-ascend910-9362-profile` 对账。3 条 warn 均为实现提示/论证完备性建议，不构成阻塞。结论：**通过**，可进入 Stage 3。
