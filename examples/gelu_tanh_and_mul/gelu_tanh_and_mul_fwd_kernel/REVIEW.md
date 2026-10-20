# GeluTanhAndMulFwdOp 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/TileOPs/tileops/kernels/elementwise/gelu_tanh_and_mul/gelu_tanh_and_mul_kernel/gelu_tanh_and_mul_fwd_kernel_DESIGN.md
- 任务类型: 迁移（GPU→Ascend NPU，harness）
- 源算子: /mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/activations.py（`GeluTanhAndMulFwdKernel`）+ `_base.py`（`FusedGatedKernel`）+ `_builders.py`（`make_fused_gated_explicit`）+ `ops/elementwise/activations.py`（`GeluTanhAndMulFwdOp`）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 2

## 机械复核（D-1/D-5）

- verify_equiv 重跑: **PASS**。`python3 examples/gelu_tanh_and_mul/gelu_tanh_and_mul_fwd_kernel/verify_equiv.py`（torch CPU，无 NPU）执行输出 `total rows: 42   EQUIV_PASS: 42   not-pass: 0`，`OVERALL: EQUIV_PASS`，exit 0——与 DESIGN §1.6.1 内嵌结果表（采纳项 1：fp32 ≤0.0182 / fp16 ≤0.0379 / bf16 ≤0.0526；采纳项 2：fp32 ≤0.0182 / fp16 ≤0.0860 / bf16 ≤0.4310；违反率 0 / 0、NaN 失配 0）**逐项一致**，无 `EQUIV_FAIL`。
  - 全域（`RANGE`）与边界（`BOUNDARY`）自检重跑：候选 tanh 机制 `2σ(2x)−1` 在 `|x|≤50` 对 fp64 tanh **max abs err = 1.781e-7**；折叠 `gelu_tanh(g)=g·σ(2u)` 在 `|g|≤100`（`u` 涨到 ≈3.6e4）对 fp64 **max abs err = 3.784e-6**（均 < fp32 atol 1e-5）；饱和 `tanh(+inf)=+1`、`tanh(−inf)=−1`、`tanh(NaN)=NaN`、`gelu(+inf)=+inf`、`gelu(−inf)=NaN`、`gelu(NaN)=NaN`、`gelu(−0.0)` signbit 保持——`BOUNDARY` 三 dtype 源式与 golden 双路径均 `gelu(−inf)=NaN`/`gelu(+inf)=+inf`。对照组 `counter npuir-vtanh` 复现截断级数域外发散（`vtanh(2.0)=−3.308`、`vtanh(10)=−5.27e5`、`±inf→NaN`）。
- design_calc_check: 重跑 `python3 .agents/tools/design_calc_check.py --design examples/TileOPs/tileops/kernels/elementwise/gelu_tanh_and_mul/gelu_tanh_and_mul_kernel/gelu_tanh_and_mul_fwd_kernel_DESIGN.md` 的 JSON checks 摘要：
  - `ub_budget`: **skip**（"no per-level buffer tables parsed"——§4.5 表格首列非 `**UB**` 粗体锚、单元格为"4096（fp32 时 8192）"无 B 后缀，机械解析失败）→ **人工复核**：逐项 buffer 之和 fp16/bf16 全显式 = 100KB、fp32 全显式 = 112KB、复用档 80KB / 88KB，均 ≪ 192KB；DESIGN §4.5 的 4 个「总计」行数值（104/120/92/96KB）**恒高于逐项之和**（偏保守，见维度 2 建议）。
  - `l0c_budget`: skip（无 Cube 路径——纯 Vector，正确）。
  - `core_split`: **skip**（"insufficient parsed dims {M:1, N:14336, bn:2048}; declared logical cores = 2048"——§5.5 的分核用 `M×ceil(N/block_N)` 而非 `ceil(M/bm)×ceil(N/bn)` 形态，机械公式不匹配）→ **人工复核**：logical = M×n_num，block_N=2048、n_num=ceil(N/2048)，与 §5.5 表（smoke 16、prefill 14336、decode 7）自洽，物理核 20×2=40 实查，见维度 3。
  - `r3_metrics`: **pass**（"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**核对方式**：亲自 Read 源算子四文件（`activations.py` L405-423 `GeluTanhAndMulFwdKernel`、`_base.py` L419-497 `FusedGatedKernel`、`_builders.py` L685-708 `make_fused_gated_explicit`、`ops/elementwise/activations.py` L284-287），与 DESIGN §0.1–§0.7 逐项核对。

- **语义理解正确性**（§0.1/§0.2）— pass。源码 `GeluTanhAndMulFwdKernel.activation_func`（`activations.py` L415-423）计算 `x_f32 = T.Cast("float32", x)` → `inner = sqrt_2_over_pi*(x_f32 + coeff*x_f32**3)` → `tanh_val = T.Cast(x.dtype, T.tanh(inner))` → `return half * x * (one + tanh_val)`，`half`/`one`/`x` 均为 `x.dtype`；融合由 `FusedGatedKernel._get_effective_op_func`（`_base.py` L458-467）的 `fused_op(gate,value)=act(gate)*value` 构造。DESIGN §0.1 引文与 §0.3 步骤 3–8 与之**逐语句一致**（fp32 求 inner/tanh → 窄 dtype 尾部乘加）。I/O：单张量 `x[M,2N]`、输出 `[M,N]` 同 dtype，与 builder（`_builders.py` L695）及 `FusedGatedKernel` docstring（L420-423）一致；`SUPPORTED_DTYPES = FLOAT_DTYPES`（L411）。边界语义 `gelu_tanh(−inf)=NaN`（源式 `(−inf)·(1+(−1))=(−inf)·0`）经 `verify_equiv.py` 的 `BOUNDARY` 段三 dtype 双路径机器证实（源式与 `F.gelu(approximate="tanh")` 均 `[nan, inf]`）。
- **算法解读完整性**（§0.3）— pass。§0.3 步骤 1–9 覆盖源码全部计算语句（Op 校验/contiguous、列偏移取 gate/value、fp32 提升、g²/g³、inner、tanh+cast、窄 dtype 乘加、×value、返回），无遗漏、无臆造；数据流图覆盖 gate/value/tanh_val/gelu/y 全部 buffer。
- **优化手段覆盖完整性**（§0.4）— pass。7 项（`explicit_parallel`、`threads×npt`、`T.tanh`、列偏移索引、fp32 参数域、常数预计算、`@lru_cache`）均可溯源：`_base.py` L436 `DEFAULT_STRATEGY="explicit_parallel"`、`_builders.py` L692 `block_N=threads*npt`、`activations.py` L422 `T.tanh`、`_base.py` L69 `BYTES_PER_THREAD=16` 与 `activations.py` L412 `MIN_NUM_PER_THREAD=2`（推出 fp16/bf16 npt=8→2048、fp32 npt=4→1024，与 §0.4 数字一致）。未把实现层优化误判为算法层。
- **耦合性判定合理性**（§0.5）— pass。四态处置（保留 4 / 等价替换 3 / 重新设计 2 / 舍弃 0）逐条有依据：`T.tanh`→重新设计（NPU 无忠实向量 tanh，独立验证见维度 8）、`threads×npt`→重新设计（`threads=` 在 npuir 无效，TRAP-threads-kwarg-noop，`src/ir.cc` NPU 分支仅消费 grid_size）、列偏移→等价替换（`T.copy` 二维切片）、fp32 域→保留强化。无「舍弃无理由」，无与 ascend-constraints 矛盾项。
- **NPU 重设计可行性**（§0.6）— pass。R1：二维 `T.Kernel(ceildiv(N,block_N),M,threads=256)+T.Parallel` → 一维 `T.Kernel(num_cores,is_npu=True)` + `block_N` 折叠 + UB v-prefix，满足「仅支持一维 Kernel」；任务序 `task=m*n_num+n0/block_N` 行优先、逐元素映射不变（无跨元素依赖、无规约），语义保持论证成立。R2：忠实 `T.tanh` → 恒等式折叠 + 显式 fp32 sigmoid（`y=g·σ(2u)·value`），`1+tanh(u)=2σ(2u)` 为**代数精确恒等**，自饱和无 inf/inf 溢出。两项均给「源方案→NPU 新算法→语义保持论证」三要素。
- **设计与决策一致性**（§1–§7 ↔ §0.5/§0.6）— pass。§3.1 公式拆解以 §1.6.1 优化后式 `y=g·σ(2u)·value` 为输入，§3.2 为纯 v-prefix 链（无 `T.tanh`/`T.vtanh`），§3.3 伪代码、§4 内存规划、§6 循环结构均为 R1/R2 落地形态，无「§0 判重设计、§3 照抄源方案」脱节。
- **golden 独立性**（§8.1）— pass。§8.1 golden 为 CPU 显式 tanh 公式 `0.5*g*(1+torch.tanh(inner))*value`（fp32），与源仓测试 `examples/TileOPs/tests/ops/test_gelu_tanh_and_mul.py` 的 `ref_program`（L64-80）一字不差，**仅依赖 torch 原生 tanh/乘法**，不含显式 sigmoid、fp32 全链、persistent 结构——未复刻 §0.6 NPU 算法。

### 1. API 可行性 — pass

- API 存在性：§3.2/§3.4 的 `T.vcast`/`T.vmul`/`T.vadd`/`T.vexp`/`T.vdiv`/`T.vbrc`/`T.copy`/`T.alloc_ub`/`T.Kernel(一维,is_npu=True)`/`T.ceildiv`/`T.min` 均能在 `docs/Tilelang.language/`（数学操作/数据类型转换操作/shape操作/内存操作）与同族 examples（mish/hardsigmoid/add）找到佐证；`T.vtanh` 文档存在但已证伪弃用（维度 8）。
- API 与模式匹配：Developer 模式，`T.alloc_ub` 显式 UB + 自动同步，无手动 `set_flag`/`wait_flag`，与 §2/§7 一致。NPU wrapper `ascend_mode="Developer"`（`examples/TileOPs/tileops/kernels/elementwise/gelu_tanh_and_mul/gelu_tanh_and_mul.py` L91）佐证。
- v-prefix 优先：全链 v-prefix，符合 AGENTS.md 约定。
- 建议（非阻塞）：§3.3 注（"`T.div`/`vdiv` 仅允许第二操作数为标量，故被除数须为张量 buffer"）与 §1.6.2 步骤 9、§3.4（"src2 可标量、src1 必须张量"）表述不一致。核对 `docs/Tilelang.language/数学操作/T.vdiv.md`：2.1 记 `src2` 为"tensor or scalar"、2.2.2 支持 `[M,N]/[M,N]`（tensor/tensor 广播）、2.4 示例亦为 `T.vdiv(src0_ub, src1_ub, dst_ub)`；限制条款 4「仅支持 src2 为标量，不支持 src1 为标量」的准确含义是"允许的标量形态只有 src2 侧，src1 不得为标量"。故本设计 `T.vdiv(one32, den32, s32)`（双 tensor）**合法**，"被除数须为张量 buffer"的决策正确，仅 §3.3 括注措辞失准。建议统一为"src1 不得为标量，故被除数用 ones buffer"。

### 2. 内存层级规划 — warn

- 搬运路径完整：§4.4 为纯 Vector 两级路径 `GM[x]→UB[gate_ub/value_ub]→v-prefix 链→UB[out_ub]→GM[y]`，无 GM→L0 越级、无 Cube 层（§4.5 正确豁免 L0C）。buffer 层级标注齐全（§4.3 全部标 UB）。
- UB 预算（人工复核，机械 skip）：逐项 buffer 之和（`gate_ub/value_ub/out_ub` fp16=4096、11 个 fp32 buffer=8192）：fp16/bf16 全显式 = **102400 B = 100KB**；fp32 全显式 = **114688 B = 112KB**；复用档 g32←gate_ub、v32←value_ub、out_ub←s32：fp16 = **80KB**、fp32 = **88KB**。对照目标平台 UB 容量 192KB（`CONST-capacity-910B2C`，`pattern-library/constants.md` L23），即使按最悲观 ×1.7 auto-multi-buffer 膨胀（fp32 112KB×1.7 ≈ 190KB < 192KB）仍不溢出，按文档 ×1.10–1.12 上偏系数余量 ≥30%。**结论：全 dtype 路径不溢出，`block_N=2048` 安全。**
- 建议（非阻塞）：§4.5 的四个「总计」行数值（104KB/120KB/92KB/96KB）均**高于**逐项之和不一致——fp16 全显式应为 100KB（差 4KB）、fp32 全显式应为 112KB（差 8KB）、复用档分别差 12KB / 8KB（疑将 `out_ub`/某窄 buffer 按 fp32 尺寸计入或重复计一个 buffer）。偏差方向**保守**、不改变"不溢出"结论，故不阻塞；建议按逐项和修正「总计」行以消除内部算术不一致。

### 3. Tiling 策略 — pass

- Block 划分明确：§5.2 `block_N=2048`，理由（取自 fp16/bf16 GPU 折叠值 `threads(256)×npt(8)`，32B 对齐、整除向量宽、整除 manifest `N=14336=7×2048`、UB 余量充分）充分。
- 对齐约束：2048×2B=4KB、×4B=8KB 均 32B 对齐 ✓；整除向量宽（fp16/bf16 ×8、fp32 ×4）✓。
- 非整除处理：§5.4/§6.4 `tail = T.min(block_N, N−n0)`，仅 `T.copy [0:tail]` 有效列，垃圾尾部不写回（同 mish）。
- 分核策略三要素（§5.5，人工复核机械 skip）：① 逻辑核数 `num_logical_kernels = M×ceil(N/block_N)`（smoke 16 / prefill 14336 / decode 7，与 §5.5 表自洽）；② 物理核数**实查** `NPUUtils.get().get_aicore_num()` 记录返回 20、纯 Vector ×2 = 40（与 `CONST-aicore-910B2C` 记 24 不一致时一律以实查为准，处理正确）；③ 规模判定：逻辑核 ≤40 时 `num_cores=logical`（无需适配），>40 时固定 40 核 + 核内 `T.serial(tiles_per_core)` persistent（静态边界 `tiles_per_core=ceil(logical/40)`，prefill 359），内核总数 = 物理核整数倍，无负载不均。三要素齐全，核数与 block 取值自洽。

### 4. 技术约束检测 — pass

- 三维 Kernel：源为二维，§3.5.1 判定涉及并折叠为一维 `T.Kernel` ✓。
- GEMM 非整除 / L0C 溢出：无 GEMM、无 Cube，§3.5.1 判不适用 ✓。
- GPU 专用 API：`T.Parallel`/`threads=`/CUDA `T.tanh` 全部改写为 v-prefix + 显式 sigmoid ✓。
- 额外约束处置正确：`T.vcast(fp32→fp32)` 损坏数据（§9.2）→ fp32 路径改 `T.copy`；bf16 无原生 v-prefix 算术 → fp32 中转。均有先例/探针依据。

### 5. 循环与同步 — pass

- §6.1 明确核级并行 `T.Kernel(num_cores,is_npu=True)`、tile 级 `T.serial(tiles_per_core)`、元素级 v-prefix 向量化；§6.2 伪代码与 §3.3 一致。
- §7 同步为 Developer 自动同步，与 §2.1 编程模式匹配，无手动同步点、无跨核协作（逐元素独立）。
- §6.4 尾块逻辑（`T.min` + guard）与 §5.4 一致。

### 6. 验证方案 — pass

- Golden：§8.1 含 PyTorch 参考实现草案（显式 tanh，fp32），独立于 NPU 算法。
- L0 计划：§8.3 给出具体规则 shape（`2048×14336` fp16/bf16、`1×14336` decode、smoke `16×128` 三 dtype）、randn 输入、`assert_close` 与容差；并列出边界/全域覆盖提醒。
- 精度标准：§8.2 按 dtype 给 atol/rtol（fp32 1e-5 / fp16 1e-2 / bf16 1.6e-2），与源仓 `_get_tolerances`（测试 L36-43）一致；机器验证候选对 fp64 参照 0 违反，标定充分。

### 7. 完整性与一致性 — pass

- 章节齐全：迁移类 §0（0.1–0.7 全）与 §1–§11 齐全；无占位符（grep `待补充`/`TODO`/`FIXME`/`{placeholder}` 均无）。
- 同类引用真实：§3.5.3 引用的全部 `examples/`（gelu_and_mul/hardsigmoid/add DESIGN、本算子 NPU scaffold、mish/lerp_tensor、test/bench/manifest）经 Glob 核对**真实存在**；§11 的 docs/工具链/知识库路径亦真实。
- 内部一致：API 映射（§3.2）、内存规划（§4）、Tiling（§5）相互一致；唯一措辞不一致见维度 1 建议（vdiv 括注，不阻塞）。

### 8. 算法优化分析 — pass

**§1.6 章节存在性** — pass：含 1.6.0 调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代 / 1.6.3 向量化轴与布局决策四子节，均有结论行（"调研结论"/"优化结论"/"向量化结论"/"布局决策结论"）。

**算法调研存在性与完整性（§1.6.0）** — pass：四问齐全。R1 候选表**含基线候选 0**（源式 `½g(1+tanh(u))` + 源 `T.tanh`）并枚举 7 个候选；R2 明确结论"无在线/离线之分（天然单遍）"并有结构依据（逐元素无跨元素依赖）；R3 复杂度表覆盖**四口径**（FLOPs / 访存量 Bytes / 扫描遍数 / 中间缓冲峰值，基线在表）；R4 逐候选「单元匹配/容量/对齐/静态边界/流水/跨核」评估。调研深度与单步逐元素算子匹配。

**调研结论正确性（独立复核）** — pass：
- **等价性独立推演**：`1+tanh(u)=2σ(2u)` 严格成立（`(e^u+e^{−u}+e^u−e^{−u})/(e^u+e^{−u}) = 2/(1+e^{−2u})`），故 `½g(1+tanh(u))=g·σ(2u)` 代数精确；σ 用标准 `1/(1+e^{−v})` 精确实现，与 `docs/Tilelang.language/数学操作/T.sigmoid.md` §3 记载的 `T.vsigmoid` lowering（VAdd/VSub/VExp/VDiv）同源。自适应饱和、NaN 传播、−0.0 保持经机器自检。
- **复杂度复算**：基线/候选 1/2 均 FLOPs `10·M·N`（与 manifest roofline `flops=10*M*N`、`bytes=3*M*N*elem` 一致，test L161-163 佐证）、访存 `3·M·N·elem`（读 2MN+写 MN）、单遍、中间缓冲 O(block_N) 无 GM 往返——四口径无漏算，与设计数字一致。
- **负向断言复核（对照参考表/源码/案例）**：核心负向断言"`T.vtanh` 非全范围忠实、不可用"经**亲自读工具链源码**独立证实——`codegen_npuir_api.cc` L2526-2595 `VtanhCodegen`（默认/Expert）确为截断 Maclaurin `x − x³/3 + 2x⁵/15 − 17x⁷/315`（系数 `−1/3`、`2/15`、`−17/315` 逐步构造）；`codegen_npuir_dev.cc` L3365-3431 `VtanhCodegen`（Developer）确为 `(e^{2x}−1)/(e^{2x}+1)`（`exp(2x)` 溢出→inf/inf→NaN）。`T.tanh`（`tilelang/language/tir/op.py` L1885）仅 `_tvm_op.tanh(x)` 标量 TIR 表达式，无向量形态。与同族 `CG-2026-0021`（`T.verf` 截断）一致；对照 `algorithm-candidates.md` ALG-elementwise 命中行（R1 公共子表达式消除/除法乘倒数/cast 链合并；R2 天然单遍；R3 融合消除 GM 往返）无被遗漏候选，未把源算法已用的 online 手段误判为"无"。R3 无并列争议（明确候选 1 唯一正确且最短）。
- **调研→设计一致性** — pass：§1.4 算法描述、§3.1 公式拆解、§6 循环结构均与 §1.6.0 选定算法（逐元素门控 tanh 近似 GELU 单遍、折叠为 `g·σ(2u)·value`）一致；§0.4 源优化手段意图（忠实 tanh / 连续逐元素并行 / fp32 域）被 §0.6 R1/R2 逐项承接，无静默丢弃。

**数学等价优化完整性（§1.6.1）** — pass：采纳 2 项，每项含四要素。（项 1）原式 `½g(1+tanh(u))`（源 `T.tanh`）→ `g·σ(2u)`（fp32 显式 `vexp/vdiv`）→ 代数精确恒等 + 容差内等价 → 省 1 加 + 1 乘且绕开不可用的 `T.vtanh`；（项 2）窄 dtype 尾部乘加 → 全 fp32 链 + 末尾单 cast → 容差内等价（≤1 ulp，对齐 golden fp32 opmath）→ 免 bf16 无原生 v-prefix 算术、对齐 golden。无其他可采纳项（单项逐元素、无公共子表达式、无冗余 cast、无中间 GM 往返）。

**等价性论证独立推演** — pass：恒等精确成立；容差内等价经 `verify_equiv.py` 重跑（42 行全 EQUIV_PASS、0 违反、0 NaN 失配）与 `RANGE` 全域（tanh 1.78e-7、折叠 gelu 3.78e-6，均 < fp32 atol）双重确认，与 §8.2 精度标准一致。

**向量化替代覆盖完整性（§1.6.2，与 §3.3/§6 交叉核对）** — pass：§1.6.2 覆盖全部核内逐元素计算点（2 cast + g²/g³/cg³/+g/×√(2/π)/×(−2)/vexp/+1/vbrc/vdiv/×gate/×value/降 cast），逐点有替代 API 与文档佐证（vcast/vmul/vadd/vexp/vdiv/vbrc）；保留 3 处标量（任务行列解算、尾列 `T.min`、分核/host 元数据）均为**非逐元素热点**且有具体理由。§3.3 伪代码、§6 循环结构与之一致（§6 无逐元素标量循环）。

**不可替代理由充分性 / 替代 API 佐证** — pass：3 处保留理由具体到本算子（任务映射索引 / 动态尾列边界 / 编译期元数据），非泛泛；替代 API 均有 `docs/Tilelang.language/` 佐证，优先 v-prefix。

**向量化轴与布局决策完整性（§1.6.3）** — pass：多轴（行 M、列 N + 行内 gate/value 两段）算子给出 5 候选矩阵，含量化评分（整除性/尾 lane 浪费/累加链/repack 代价/UB 影响）与逐候选弃选理由。**弃选论证前提核对**：候选 #3（核内 `T.transpose` 重排）前提"核内 UB 级、二轴交换、纯增管线开销"经亲自打开 `docs/Tilelang.language/创建操作/T.transpose.md` 核对一致（底层 `hivm.hir.vtranspose`，仅支持二轴交换；示例用 `T.alloc_shared` UB），并与 `pattern-library/layout.md` `PL-1.1-transpose-chain`（实测 ~5.4µs）一致——**无与文档/实测矛盾**；候选 #2（host 切分）增流量且违反单输入契约、候选 #4/#5 无收益/同构，理由量化。迁移对照说明 GPU 轴选择是输入非结论。结论（行优先任务序 + 行内两段连续切片 + 列轴向量化）与 §3.3/§4/§6 三方一致（buffer 均 1-D 列块）。

**与下游章节一致** — pass：§3.1 以 §1.6.1 优化后公式为输入；§6 无 §1.6.2 未论证的标量循环。

## 检视总结

全部**阻塞级维度（0/1/2/3/4/6/8）均为 pass**（维度 2 含非阻塞算术一致性提醒），无阻塞级问题，结论为**通过**。两处建议级问题（§4.5「总计」行算术偏高但保守、§3.3 vdiv 括注措辞失准）不影响语义、API 可用性与 Stage 3 可行/精度，可在后续修订中顺手修正。核心迁移风险点（忠实 tanh 缺失）经独立读工具链源码 + 机器等价验证双重确认，恒等式折叠 + 显式 fp32 sigmoid 在 `|g|≤100` 全域与 `±inf/NaN/−0.0` 边界精确忠实，golden 独立，可进入 Stage 3。
