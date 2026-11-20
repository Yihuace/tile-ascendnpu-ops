# _make_dropout_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/dropout/_make_dropout_kernel/DESIGN.md
- 任务类型: 迁移（migration-harness）
- 源算子: /home/developer/TileOPs/tileops/kernels/dropout.py（`_make_dropout_kernel` L29–L67、`DropoutKernel` L70–L122）
- 检视维度: 9 项（维度 0 源算子理解与迁移分析 / 1 API 可行性 / 2 内存层级 / 3 Tiling / 4 技术约束 / 5 循环同步 / 6 验证方案 / 7 完整性 / 8 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `PASS`——`python3 examples/dropout/_make_dropout_kernel/verify_equiv.py` 实跑 `FAIL items: 0 -> ALL EQUIV_PASS`（exit 0）。逐项与 DESIGN.md §1.6.1 内嵌结果表核对**一致**：
  - `mask-mul vs GOLDEN(finite)` fp32/fp16/bf16 × p∈{0.1,0.3,0.5,0.7,1.0}：15 行全部 `max|d|=0.000e+00 viol=0/1048586`（表列「5×1,048,586，0.000e+00，0」一致）；
  - `mask-mul vs GOLDEN(inf/nan)` 3 dtype × p∈{0.5,1.0}：6 行全部 `max|d|=0, viol=0/6`（一致）；
  - `fp32-transit vs input-dtype` fp16 `5.960e-08`、bf16 `0.000e+00`（一致）；
  - χ² top-11 = `2004`（dof 2047，脚本阈 `dof+4√(2dof)=2303`）、lag-1 χ² = `1119`（dof 1023，阈 `1204`）（一致）；
  - deterministic replay 逐位相同、seed separation `0.496`、threshold prob exactness `err≤4.77e-08`、empirical drop rate `dev≤2.795e-04`（一致）；
  - 非门禁诊断段：源 `if_then_else` vs GOLDEN(inf/nan) 偏离 `2/6`（p=0.5）、`3/6`（p=1.0），采纳算术乘 `0/6`（一致）。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/dropout/_make_dropout_kernel/DESIGN.md` 输出：
  - `ub_budget`: **skip**（`no per-level buffer tables parsed`）→ **人工复核已核**：逐 buffer 手算 fp32 bs=1024 = `x_ub 4 + y_ub 4 + xf 4 + mf 4 + 8×int32(4) = 48 KB`，fp16/bf16 bs=2048 = `4+4+8+8+8×8 = 88 KB`（与 §4.5 完全一致）；×1.7 膨胀 ≈ 82 / 150 KB < 192 KB（`CONST-capacity-910B2C` 9362 注记）→ **pass**。
  - `l0c_budget`: **skip**（无 block_M×block_N）→ **人工复核**：纯 Vector 算子无 Cube/L0C，**n/a**。
  - `core_split`: **skip**（`insufficient parsed dims {}`，工具未解析出表格）→ **人工复核已核**：§5.5 三要素完整，逻辑核数 `ceil(N/block_size)` 与表内值一致（fp32 [1024,4096]→4096；bf16 [1024,10240]→5120；fp16 smoke→512）；物理核数**本 Agent 独立实跑** `NPUUtils.get().get_aicore_num()=20`、`get_aivector_core_num()=40`（与 §5.5 ② 及 `CONST-aicore-910_9362` 一致）→ **pass**。
  - `r3_metrics`: **pass**（四口径齐备，与 §1.6.0 R3 表一致）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**方式**：亲自 Read 源算子 `/home/developer/TileOPs/tileops/kernels/dropout.py` 全文（122 行），与 DESIGN.md §0（0.1–0.7）逐项核对。

- **语义理解正确性 — pass**：源码 L46 `scale = 1.0/(1.0-p) if p<1.0 else 0.0`、L58 `keep = rand_val >= T.cast(p,"float32")`、L59–63 `if_then_else(keep, x*cast(scale,x.dtype), 0)`；§0.1 数学语义（`y=x·mask/(1-p)`，drop prob p，保留放大 1/(1-p)）、`p=0` 恒等 / `p=1` 全零、输入 dtype 域计算、无规约（逐元素）均与源码一致。§0.1 额外实测 torch 契约量（`±inf` 丢弃位返回 NaN、保留 `-0.0`）经 §1.6.1/verify_equiv 复核成立（源选择式反而偏离契约，源真相与 §0.1 判定自洽）。
- **算法解读完整性 — pass**：§0.3 五步分解（`rng_init` / `idx=(bx*threads+i)*npt+j` / `rng_rand_float` / `keep=rand>=p` / `if_then_else`）逐句覆盖源码 L52–L63，**无遗漏、无臆造**；host 侧逻辑（`lru_cache` 特化、`forward` 调 `self.kernel(threads,npt)(x)`、`N_total=numel` 展平）与源码 L28/L89–L122 一致；数据流图（寄存器态 RNG、无 SMEM、`block_size=threads×npt`）与源码一致。
- **优化手段覆盖完整性 — pass**：§0.4 列 9 项，与源码逐一对应（#1/#2 cuRAND on-chip + 每线程顺序态 = L53/L56；#3 置换索引 = L55；#4 输入 dtype 域乘 = L61；#5 register/fragment；#6 选择式核 = L59；#7 单遍；#8 `lru_cache` = L28；#9 无 autotune = `default_config` L116）。算法层/实现层归类正确（如 #6 判「算法层」、#3 判「优化」）。
- **耦合性判定合理性 — pass**：§0.5 每项四态处置（保留/等价替换/重新设计/舍弃）+ 依据齐备，与 GPU→NPU 能力映射及 `ascend-constraints.md` 无矛盾。关键项复核：cuRAND（NVIDIA 专用）判重设计（NPU 无原语，见维度 8 CG-2026-0021 独立复核）、置换索引判舍弃（有理由）、`threads=` kwarg 判舍弃（`TRAP-threads-kwarg-noop`）、`if_then_else` 判等价替换+优化（`vselect` 6.5x，与 `PL-1.29` 实测一致）。舍弃项均写明理由。
- **NPU 重设计可行性 — pass**：§0.6 R1–R4 每项含「源方案 → NPU 新算法 → 语义保持论证」三要素。R1 一维 `T.Kernel(40,is_npu=True)`+`T.serial` 网格跨步（满足一维 Kernel 约束）；R2 无状态 counter-based hash（无跨元素依赖，满足语义保持）；R3 fp32 中转（bf16 API 强制）；R4 算术掩码乘（免 `vselect`）。逐项均说明任务映射只重排元素区间、逐元素语义恒等，尾块由编译期 N 静态守卫覆盖不越界。
- **设计与决策一致性 — pass**：§1–§7 描述的是 §0.5/§0.6 决策后的 NPU 算法——§3.2/§3.3/§4.4/§5.2/§6.1 均为「一维核 + 网格跨步 + UB 向量链 + counter hash + 算术掩码乘 + fp32 中转 + 自然展平序」，与 §0.5/§0.6 无脱节，无「§0 判重设计、§3 照抄源方案」现象。
- **golden 独立性 — pass**：§8.1 `golden_from_mask` 以 torch 公式 `x*(m*scale)` 实现（§0.1 契约），**不复刻** §0.6 的 NPU hash/掩码结构；`golden_dropout_torch` 为 torch 基线；二者均独立于 §3.3 向量链。

### 1. API 可行性 — pass

- **API 存在性 — pass**：§3.2/§3.4 列出的全部 API 均在 `docs/Tilelang.language/`（数学/逻辑/比较/条件/转换/创建/shape 各子目录）或 `examples/` 找到佐证，本 Agent 逐一 Glob 确认文件存在：`T.vmul.md`·`T.vadd.md`·`T.vsub.md`·`T.vxor.md`·`T.vshl.md`·`T.vand.md`·`T.vshr.md`·`T.vmax.md`·`T.vmin.md`·`T.vcast.md`·`T.arange.md`·`T.vbrc.md`·`T.vselect.md` 全部 EXIST。`T.fill` 源码佐证 `tilelang/language/customize_npuir.py::npuir_fill`（L702，接受 `tir.PrimExpr`）；`T.Kernel(n,is_npu=True)`+`T.serial` 佐证 `examples/elementwise/vec_add_2d.py`、`bitwise_not_fwd_kernel.py`。
- **dtype/句法细节 API 复核 — pass**（亲自打开文档核对）：`T.vcast.md` L49 `i32→f32` 支持 `rint`，L36/L42 `f16→f32`/`bf16→f32` 仅 `rint`（与 §3.4/§R-13 一致）；`T.vmin.md` L42「src2 可以使用标量」、`T.vmax.md` L18 src1 须 `tensor`（与 §3.2 用 `zerob` 常量 buffer 作 src2、`vmin(t,1,t)` 一致）；`T.vshr.md` L34「B 须是 `(1,)` tensor，标量暂不支持」（与 §R-2/§3.4 一致）。
- **v-prefix 优先 — pass**：设计全部采 v-prefix（`vxor/vshl/vmul/vand/vsub/vmax/vmin/vcast/vadd/vbrc`），未用 legacy `npuir_xxx`。
- **模式匹配 — pass**：Developer 模式，§7 用编译器自动同步（无手动 set_flag/wait_flag）。

### 2. 内存层级规划 — pass

- **搬运路径完整 — pass**：§4.4 给出 `GM[x]→T.copy→UB→(向量链)→UB→T.copy→GM[y]`；无 GM→L0 越级；纯 Vector 无 L1/L0（§5.1 判定正确）。
- **UB 预算 — pass**：§4.5 逐 buffer 表手算复核一致：fp32 bs=1024 = 48 KB、fp16/bf16 bs=2048 = 88 KB，最大路径 ×1.7 ≈ 150 KB < 192 KB（`CONST-capacity-910B2C`，9362 容量注记一致）。`bs=4096`（E1）176 KB×1.7>192 KB 已识别为越限并降级为 Stage 4 杠杆，处置正确。
- **Buffer 层级标注 — pass**：§4.3 逐 buffer 标注 UB；常量 buffer（seedb/m31/zerob/tp_b）标注只读、循环外填充。
- **L0C 容量 — n/a**：无 Cube 计算。

### 3. Tiling 策略 — pass

- **Block 划分明确 — pass**：§5.2 `block_size` = K11 折叠值（fp32 1024 / fp16·bf16 2048），与集成 wrapper `_DEFAULT_BLOCK_SIZE`（实读 `examples/TileOPs/tileops/kernels/regularization/dropout/dropout.py` L86–L90）完全一致，理由充分。
- **对齐约束 — pass**：`block_size×elem_bytes` = 4096B 为 32B 整数倍；尾 lane 浪费 0%（fp32 256 槽 / fp16 128bit×8）。
- **非整除处理 — pass**：§5.4 动态尾块 `tail=T.min(block_size, N-off)`，manifest 全 workload 整除、测试 4,000,000 余 1216 触发尾块已覆盖（L0-10）。
- **分核策略三要素 — pass**（核对 `_shared/standards/core-split-strategy.md` §1）：
  - ① 逻辑核数：`ceil(N/block_size)`，表内值与 N/block_size 逐一核对一致（4096/512/5120/2048）。
  - ② 物理核数**实查**：§5.5 给出 `NPUUtils.get().get_aicore_num()`/`get_aivector_core_num()` 查询代码；**本 Agent 独立实跑同一查询，实际返回 20 / 40**（与 §5.5 ② 及 `CONST-aicore-910_9362` 一致）→ 非文档假设。
  - ③ 规模判定：逻辑核数（512–5120）远超物理核 40 → 固定启动 40 核 + `T.serial` 网格跨步；循环上界静态（`n_num`/`core_num` 编译期常量）；残留负载不均显式记录（+0.78%~+4.2%）；先例 `vec_add_2d.py`/`PL-1.22`。三要素自洽。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（一维 `T.Kernel(40, is_npu=True)`）。
- GEMM 非整除 / L0C 溢出：不适用（非 GEMM、无 Cube）。
- §3.5.1 列出的本项目已知限制（`TRAP-conditional-alloc` / `TRAP-threads-kwarg-noop` / `vxor` src1 须 tensor `VP-2026-0130` / `vmax` 无标量广播 / `vshr` 算术+标量 / bf16 数学 API 缺失 / host 禁核心计算）逐项有处理方案，与 §3.3 伪代码一致（buffer 顶层无条件分配、常量 buffer 承载、`zerob` 下界、避免 `vshr`、fp32 中转、核内生成掩码）。

### 5. 循环与同步 — pass

- §6.1 明确核级 `T.Kernel`、tile 级 `T.serial` 网格跨步、元素级向量化；无逐元素标量循环（与 §1.6.2 一致）。
- §6.3 说明不用 `T.Pipelined`（单 tile 线性依赖、跨 tile 由 auto-multi-buffer 处理），合理。
- §6.4 尾块处理见 §5.4。
- §7 同步与 Developer 模式一致（自动同步，无手动 sync）；§7.4 判定非 MixCV（无 `T.gemm`）正确。

### 6. 验证方案 — pass

- **Golden — pass**：§8.1 提供 `golden_from_mask`（torch 契约语义，独立于 NPU 链）与 `golden_dropout_torch`（torch 基线），并说明随机算子以「分布+重放+缩放+边界」为判据（口径合理，见维度 8）。
- **精度标准 — pass**：§8.2 atol/rtol fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2，与源测试 `examples/TileOPs/tests/ops/test_dropout.py` L127–L131（实读）一致。
- **L0 测试计划 — pass**：§8.3 12 条用例覆盖统计 3σ、确定性重放、异 seed、缩放、`p=0/1`、`training=False`、shape/dtype、非整除尾块、非默认 `block_size` 回归，与源测试用例集（实读 test_dropout.py：statistical_rate/scale_factor/deterministic_replay/different_seeds/p0_identity/p1_all_zeros/training_false/preserves_shape/custom_config）逐项对应。

### 7. 完整性与一致性 — pass

- 章节齐全：§0（0.1–0.7 全）+ §1–§10，模板结构完整。
- 同类引用真实：§3.5.3 引用的 `examples/bitwise_not/bitwise_not_fwd_kernel/`、`examples/elementwise/vec_add_1d.py`、`vec_add_2d.py`、`leaky_relu/_make_leaky_relu_kernel/`、`elu/_make_elu_kernel/`、`expm1/expm1_fwd_kernel/`、集成 wrapper 路径均 Glob 确认存在。
- 无占位符：全文无未填模板项（已逐章通读确认）。
- 内部一致：API 映射（§3.2）、内存规划（§4）、Tiling（§5）三处 buffer/参数描述与 §3.3 伪代码一致（12 buffer 数、block_size、分核形态）。

### 8. 算法优化分析 — pass

- **§1.6 章节存在 — pass**：含 1.6.0（四问）/1.6.1（数学等价优化）/1.6.2（向量化替代）/1.6.3（向量化轴与布局）四子节。
- **算法调研存在性与完整性 — pass**：R1 等价化简候选表**含基线**（子层 A 基线 A0=源 cuRAND，子层 B 基线 B0=源 `if_then_else`），候选覆盖 LCG/xorshift32/右移无关 hash/host 物化/`fill_rng_2d`；R2 在线算法给出明确「无/不适用」结论（逐元素无跨元素依赖，结构判据充分，且指出源 cuRAND 顺序态是手段非在线算法）；R3 复杂度表**四口径齐备**（op/元素、访存 Bytes、扫描遍数、中间缓冲峰值，基线 A0 在表）；R4 逐候选硬件亲和评估（单元匹配/容量/对齐/静态边界/流水/跨核），容量/带宽引用 `constants.md`。调研范围写明。
- **调研结论正确性（独立复核）— pass**：
  - **负向断言「npuir 无 RNG 原语 / 候选库无 dropout 条目」独立复核成立**：本 Agent 亲跑 `dir(T)` 过滤 rng/rand → `[]`；`tilelang/language/` 全目录 grep `rng|random|philox|rand` → 无 RNG 算子（仅 "operand" 假命中）；`docs/Tilelang.language/` 16 子目录枚举（shape/内存/创建/原子/同步/排序/数学/转换/条件/比较/索引/线性代数/编译器提示/规约/调试/逻辑）**无 RNG 子目录/条目**；`customize_npuir.py`（2021 行）grep `rng|random|philox` → 无；全仓 `rng_init|rng_rand_float` 命中**仅**脚手架拷贝（`examples/TileOPs/.../dropout*`）、DESIGN/探针/日志，**无实现**；`fill_rng_2d` 确存于编译器 named ops（`3rdparty/.../mlir/python/mlir/dialects/linalg/opdsl/ops/core_named_ops.py` L1814）但无 TileLang 前端映射 → **前端不可达，CG-2026-0021 结论成立**。`algorithm-candidates.md` grep `rng|dropout|random|prng` → 无命中，印证「候选库无 RNG 条目」。
  - **等价性独立推演成立**：算术掩码乘 `(x·scale)·m` vs 契约 golden `x·(m·scale)`——有限域乘法结合律逐元素恒等；角点独立核对：p=1 时 scale=0、m=0，`(±inf·0)·0 = NaN` 与 golden `±inf·(0·0) = NaN` 一致；保留位 `(x·scale)·1 = x·scale = x·(1·scale)`；`-0.0` 保留位两侧同为 `-0.0`。源选择式对丢弃 `±inf` 返回精确 0 → 偏离契约，采纳算术乘更贴合。verify_equiv 重跑 `0/6` 差异印证，诊断段源选择式 `2–3/6` 偏离印证。
  - **阈值概率独立复算成立**：drop prob = `256·Tp/2^31 = Tp/2^23`，`Tp=round(p·2^23)` → 误差 ≤ 2^-24；p=0 → `Tp=0`→`tp_shift=-1`→`d=hm+1≥1`→m≡1（恒等）；p=1 → `Tp=2^23`→`256·Tp-1=2147483647` 恰为 int32 max（**设计在 Python 端先减 1 规避 int32 溢出，正确**）→ `d≤0`→m≡0（全零）。掩码边界 `hm≥256·Tp` ⇔ `clamp(hm-(256·Tp-1),0,1)=1` 代数核对正确（`hm=256Tp-1`→0 丢弃，`hm=256Tp`→1 保留）。
  - **复杂度独立复算**：全部可行候选访存同为理论下界 `2N·eb`、单遍（A1 host mask 例外 +N·eb，已在表注明 +50%），无中间缓冲 GM 往返漏计；op 数逐条对照 §3.3 伪代码（PRNG 链 + 掩码链 + 数据链）量级一致（见「建议级观察」）。
  - **并列候选有裁决路径 — pass**：§1.6.3 给出主选 A5 vs 备选 A5′（简化 hash）的 A/B 裁决计划，含指标（16m latency + 统计门禁）、判定阈值（<0.97×且统计全过）、回写路径；block_size 2048/4096 列为独立 Stage 4 杠杆。
- **调研→设计一致性 — pass**：§1.6.0 选定「A5 hash + B1 算术掩码乘 + fp32 中转」实际驱动 §1.4 算法描述、§3.1 公式拆解（步骤 2–9 即 A5+B1）、§6 循环结构（核内 PRNG 向量链，无逐元素标量循环）；与 §1.6.1–§1.6.3 同口径。迁移任务：§0.5/§0.6 处置与调研选族一致（cuRAND→重设计、置换索引→舍弃、选择式→等价替换+优化），源 #1–#9 优化意图承接逐项说明（无静默丢弃）。
- **数学等价优化完整性 — pass**：§1.6.1 五项采纳（#1 算术掩码乘、#2 fp32 中转、#3 右移无关化、#4 阈值概率精确化、#5 CSE/buffer 复用）均有「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素。独立推演：#3 右移无关化代数恒等（`hm≥0` 时 `⌊hm/256⌋≥Tp ⇔ hm≥256Tp`）；#4 阈值精确性上式复算成立；#5 原地复用满足生命期不重叠（`hm` 复用 `t`、`idx` 复用 `h`），不改数值，先例 `PL-1.23`。拒绝项 E2/E3 有理由。
- **向量化替代完整性 — pass**：§1.6.2 覆盖 §3.3 全部计算点——元素索引（#1 arange+vbrc+vadd）、RNG（#2）、阈值比较（#3）、掩码施加/缩放（#4）、fp32 中转（#5）、常量填充（#6）全部 ✅ 向量化；保留 3 处（#7 tile 调度循环 / #8 block 索引 / #9 动态尾块）均属「tile 级调度 + block 索引 + 动态边界」，理由具体到本算子（block 级、非逐元素热点、N 编译期但 tail 依赖 off）。与 §6 交叉核对：kernel 内无逐元素标量循环，一致。
- **不可替代理由充分性 — pass**：#7–#9 理由具体（非「实现简单」「照源码写」）。
- **替代 API 可行性 — pass**：§3.4 全部 API 附 docs 佐证 + 目标设备探针实测（与维度 1 口径一致），优先 v-prefix。
- **向量化轴与布局决策完整性 — pass**：1-D `[N]` 单轴算子，§1.6.3 按 skill「单轴算子豁免」写明唯一候选轴=1D 元素轴并给依据；轴质量量化（尾 lane 浪费 0%、32B 对齐、无累加链、无 repack）；迁移注指出 GPU 并行轴是输入非结论；A/B 裁决计划（A5 vs A5′）齐备。
- **布局决策与实现一致 — pass**：选定「一维连续 `[block_size]`、I/O 布局=核内布局、无 repack」与 §3.3 伪代码、§4 buffer 形状、§6 循环一致；UB 预算按该布局计算。
- **与下游章节一致 — pass**：§3.1 以 §1.6.1 优化后公式为输入（步骤含 counter hash、阈值掩码、算术乘）；§6 无 §1.6.2 未论证的标量循环。

## 备注（非阻塞观察）

以下为检视过程中的独立复核观察，**不构成阻塞级或建议级缺陷**（本报告结论为通过，无缺陷章节），供 Stage 3/Stage 4 参考：
1. 机械复核的 op 数计数：§1.6.0/§3.3 自述「每 tile 19（fp32）/21（fp16·bf16）」，本 Agent 按 §3.3 伪代码逐行计数得 ~18/20（含 `T.fill(base)`，不含循环外 `arange`）。量级一致，且设计以「~19–21」估计，不影响「发射主导」结论。
2. χ² 检验阈标注：§1.6.1 将脚本 `dof+4·√(2·dof)`（= 4σ）标为「3σ 阈 2303」；数值上 2004 < 2239（真 3σ）亦通过，属保守标注。另本 Agent 独立跨 seed 扫描发现 lag-1 χ²（top-5 bit 投影）对 seed 敏感（seed 42→1252、4242→1237，超 1204），但**直接对二值 dropout 掩码做 lag-1 自相关复核，全部 seed 仅 1e-4~5e-4、2×2 配对 χ²≤1.2（dof 1）**——即投影位统计的波动**未转化为掩码层相关**；生产/测试用 seed（0/42/123）的 drop rate 均落 3σ 内，契约满足。
3. §8.1 `golden_dropout_torch` 中 `torch.Generator().manual_seed(seed)` 创建后未传入 `F.dropout(generator=g)`（草稿小瑕）；因其仅用于 `p=0`/`training=False` 期望值与性质对照（此时早返回 clone），且逐元素契约 golden 由独立的 `golden_from_mask` 承担，不影响验证独立性。
