# eq_fwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/eq/eq_fwd_kernel/DESIGN.md
- 任务类型: 迁移（scenario=migration, migration_mode=harness）
- 源算子: `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（+ op 类 `/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/comparison.py`）；Stage 0 提取件 `examples/TileOPs/tileops/kernels/elementwise/eq/_eq_fwd_kernels.py`
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 4

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: **PASS**（`python examples/eq/eq_fwd_kernel/verify_equiv.py` 退出码 0，`OVERALL: EQUIV_PASS`）。全部 16 个用例行与 DESIGN.md §1.6.1 内嵌结果表**逐行一致**（E1-eq int8 vs uint8 / E1-bool view(int8 0/1) violations=0；E2 十组形状 vs 源 divmod 映射 a_mismatch=b_mismatch=oob=0；E3 int8/uint8 exact-in-fp16 与 eq-preserved violations=0）。工具链 tilelang 0.1.2+417dae9908ad / CANN 8.5.0。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/eq/eq_fwd_kernel/DESIGN.md` JSON checks 摘要：
  - `r3_metrics` → **pass**（"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"）。
  - `ub_budget` → **skip**（"no per-level buffer tables parsed from the memory plan"）→ **人工已核**：§4.5 逐 dtype 手工预算独立复算全部正确——fp16 8B/elem→32KB、fp32 12B→48KB、int64 20B→80KB、int8 10B→40KB，×1.7 膨胀后 55/82/136/68KB 均 <192KB；block_size 上限 `floor(192KB/(B_elem×1.7))` 复算 14456/9637/5782/11565 全部一致。
  - `l0c_budget` → **skip**（"no affirmative Cube-path API"）→ **人工已核**：纯 Vector 算子，无 L0/L0C，不适用。
  - `core_split` → **skip**（"insufficient parsed dims {}; declared logical cores = None"）→ **人工已核**：§5.5 逻辑核数独立复算一致（smoke-1k=1 / hidden-state=2048 / cnn-feat=4096 / row-broadcast=4096，coalesced_shape 与 verify_equiv E2 输出 `(8388608,)/(16,256,3136)/(4096,4096)` 吻合）；物理核数 40 = `NPUUtils.get().get_aicore_num()*2 = 20*2` 实查记录；核内 `T.serial(num_local)` 边界为编译期常量。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**核对方式：亲自 Read 源算子全文（`elementwise.py` L166–L1136/L1699–L1716、`comparison.py`、Stage 0 提取件 `_eq_fwd_kernels.py`）与 DESIGN.md §0.1–§0.7 逐项对照。**

- **语义理解正确（pass）**：§0.1 `y=(a==b)` 逐元素、NumPy/PyTorch 广播、输出 bool、浮点 IEEE（NaN≠NaN / ±0 相等）、整数二补码相等——与源码 `EqFwdKernel.op_func(a,b)=a==b`（elementwise.py L1707–1708）+ `OUTPUT_DTYPE=torch.bool`（L1703）+ `SUPPORTED_DTYPES=_BINARY_FULL_DTYPES`（L1702，实体 L166–L170）完全一致。§0.1「bool 特殊路径」= Op 层 `view(uint8)` + `EqBoolStorageFwdKernel.op_func=(a^b)^1`（L1714–1716）+ `_BoolOutputBinaryOp._build_kernel_instance` 以 dtype=uint8 构建 bool-storage 内核、`_eager_forward` 末 `result.view(torch.bool)`（_base.py L1099–1123）——设计与源码一致。无规约/无跨元素依赖的判定正确（逐元素映射）。
- **I/O 契约正确（pass）**：§0.2 输出 `(N_total,)` bool、`out_idx=[2]`、一维展平后由 Op 层 `reshape(out_shape)`——与提取件 `y: T.Tensor((N_total,), out_dtype)`（L56/L79）、`@tilelang.jit(out_idx=[2])`（L48/L71）、`BinaryOp._eager_forward` 的 `contiguous().view(-1)...reshape(out_shape)`（_base.py L787–789）一致。
- **算法解读完整（pass）**：§0.3 步骤表逐条对应源码计算语句——同形快路径 `T.Parallel(threads,npt)` 标量 `a[idx]==b[idx]`（提取件 L59–61）、广播路径 `_compute_broadcast_offsets` divmod 链 + 标量 gather `a[a_off]==b[b_off]`（L82–87 / L161–177）、`coalesce_broadcast_dims`（L95–158）、`_is_contiguous_same_shape` 分支（L46/L180–186）、grid `ceildiv(N_total,block_size)`。数据流图（GM→register→标量比较→GM）与循环/并行结构（grid × `T.Parallel(threads,npt)`）覆盖全部 buffer，无遗漏、无臆造。
- **优化手段识别完整（pass）**：§0.4 九项与源码显式手段一一对应（explicit_parallel L556–609、coalesced 连续段、coalesce_broadcast_dims、编译期 unrolled divmod、同形快路径、register_copy L464–499[eq 因 bool 输出不用]、lru_cache L39/L48、bool 走 uint8 存储、单条 `==`）；未把实现层优化误判为算法层。**建议级（不阻塞）**：§0.3 称「num_per_thread=8（整数族），block_size=2048（整数）」——源码实为 `bool_output_needs_scalar`（_base.py 无关；BinaryKernel.__init__ elementwise.py L967–991）对 int8/int16 输入强制 `strategy="direct"`（block_size=threads=256，无 npt），仅 int32/int64 走 explicit_parallel npt=8。该差异属源 host 侧 strategy 选择，不影响 NPU 重设计（wrapper 已把 threads×npt 折为单一 block_size），但 §0.3 的表述宜精确到「int32/int64」。
- **耦合性判定合理（pass）**：§0.5 四态处置（保留 7 / 等价替换 3 / 重新设计 4 / 舍弃 1）逐项有依据，与 migration-analysis §5.3 映射表一致——CUDA 两级并行→重新设计、coalesced→`T.copy` 等价替换、register/fragment→`T.alloc_ub` 等价替换、bool 存储→1 字节承载重设计。舍弃项（`threads=`/npt kwarg）写明理由（npuir `T.Kernel` 不接受、wrapper K11 已折叠）。
- **NPU 重设计可行且语义保持（pass）**：R1（行 base + innermost 向量化 + 显式同形物化）、R2（一维 Kernel + Vector + persistent 分核）、R3（bool→int8 / uint8→int8 承载）、R4（dtype 比较域映射 + 同形 `T.vcmp`）均有「源方案→NPU 新算法→语义保持论证」三要素，且满足 NPU 约束（一维 Kernel、UB 容量、32B 对齐、静态边界、vcmp 同形）。我独立复核了 R1 的「末维必有 `a_strides[-1]==1` 或 `b_strides[-1]==1`」论证：由 `coalesce_broadcast_dims` 的分组语义（组内某操作数 stride=0 ⇒ 其全部合并维均 size-1 广播）可推出末组至少一操作数 stride=1——论证成立（E2 机器验证 10 组形状 mismatch=0 佐证）。R3/R4 由 E1/E3 机器验证（violations=0）。
- **与 §1–§7 一致（pass）**：§1.4/§3.1/§3.3/§4/§5/§6 描述的均为 §0.6 重设计后的 NPU 算法（`T.vcmp`+`T.vcast`+`T.vbrc`+int8 承载+persistent），无「§0 判重设计、§3 照抄源方案」的脱节；§0.6 末尾「优化意图承接总表」逐项承接源 §0.4 九项手段，无静默丢弃。
- **golden 独立性（pass）**：§8.1 golden 为 `torch.eq(a,b)` 广播直算，以 §0.1 源语义为唯一依据，未复刻 §0.6 的行/内层索引、vcmp、vcast 链、persistent 结构——验证独立。

### 1. API 可行性 — pass

§3.2/§3.4 列出的每条 API 均在本仓有佐证：`T.vcmp`（`docs/.../比较操作/T.vcmp.md` §2.1/§2.2.2/§2.4，亲自打开核对 dtype 矩阵与同形约束）、`T.vcast`（`数据类型转换操作/T.vcast.md` §2.2.1 `i1→f16`/`f16→i8`/`i8→f16`/`ui8→f16` 亲核）、`T.vbrc`（`shape操作/T.vbrc.md` §2.2.1/§2.2.2）、`T.copy`/`T.alloc_ub`/`T.min`/`T.ceildiv` 均有 docs；`T.Kernel(grid, is_npu=True)`/`T.serial`+`task=i·K+cid`+`if` 有 `examples/elementwise/vec_add_1d.py` 与 `T.vcast.md` §2.4 先例。Designer 模式用自动同步、v-prefix 优先（vcmp/vcast/vbrc），未见 npuir_xxx 兼容写法。§3.4 逐 API 附探针实测状态（C1 dtype 覆盖、bool-GM 挂死、vbrc int8 挂死），与源码/capability-gaps 一致。

### 2. 内存层级规划 — pass

§4.4 给出完整纯 Vector 搬运路径（GM↔UB，无 GM→L0 越级、无 L1/L0 参与，无中间 GM 往返），§4.3 每个 buffer 标注存储层级（UB），§4.2 输出标 int8。UB 预算见「机械复核」人工核：全 dtype 路径 ×1.7 膨胀后 ≤136KB <192KB，算术独立复算一致。**建议级（不阻塞）**：§4.5 对 fp32/int64 等非 upcast 路径未计入 `a_ub`/`b_ub`，依赖「§4.3 Python 编译期分支分配（否则 DCE）」。若 Stage 3 按 §3.3 伪代码把 `a_ub/b_ub` 无条件声明，则 int64 变为 36B/elem，`4096×36×1.7≈245KB >192KB`（此时 block_size 上限降至 ~3212）。缓解：int64 的 wrapper 默认 block_size 为 2048（74KB×1.7≈125KB，安全）；建议 Stage 3 将 `a_ub/b_ub` 声明移入 `if upcast:` 分支并复核 int64@4096 的编译期 UB 报错，避免伪代码与 §4.5 口径的分歧。

### 3. Tiling 策略 — pass

§5.2 给出一维划分（`block_size=4096`）+ 6 条选择理由（末维单块、32B 对齐、向量宽度整除、UB 预算、同形 chunks、同族一致）。**分核策略三要素齐全并自洽（对照 core-split-strategy.md §1/§2.2）**：① 逻辑核数 `num_logical=R×ceildiv(L,block_size)` 给出各 workload 值（1/2048/4096/4096），我独立复算一致；② 物理核数依据为 `NPUUtils.get().get_aicore_num()` **实查记录**（2026-10-13，返回 20 ⇒ 纯 Vector 翻倍 40，含查询代码），并注明 `CONST-aicore-910B2C`（910B2C=24/48）与本机 20 不同源、以实查为准；③ 规模判定正确（smoke-1k ≤40「无需适配」；其余极大规模 → `num_kernels=min(num_logical,40)` + 核内 `T.serial(num_local)` 静态边界 grid-stride，负载不均 ≤1 task/核）。非整除尾块 `tail=T.min(...)`（§6.4）覆盖。

### 4. 技术约束检测 — pass

§3.5.1 逐条核查：不支持三维 Kernel（不涉及）、GPU 专用 API（`is_npu=True`+v-prefix 取代）、无 GEMM（非整除由 §6.4 处理）、无 L0C（纯 Vector）、物理核数限制（§5.5）、host 禁改输入（仅元数据）、uint8 不可用（→int8）、**bool 输出不可落 GM（→int8 0/1，CG-2026-0022）**、**`T.vcmp` 无自动广播（→显式物化，CG-2026-0021）**、**`T.vcmp` 不支持 int8/uint8（→升 fp16，§0.6 R4）**。三项本算子特有约束均被识别并给出处置。

### 5. 循环与同步 — pass

§6.1 明确循环结构：核间 `T.Kernel`(persistent)、核内 `T.serial`(静态边界)、元素级由 `T.vcmp`/`T.vcast`/`T.vbrc` 向量化（无 `T.Parallel` 逐元素标量循环，与源差异已注明）；§6.3 说明不用 `T.Pipelined` 的理由（单阶段无 K 维；但注明 eq 每 task op 数多于位运算族、发射项可能主导 → 交 Stage 4）；§6.4 尾块 `T.min` 模式引 `vec_add_1d.py` L31–38 先例。§7 同步策略与 Developer 模式一致（自动同步，无跨 task 依赖、无 workspace）。§1.6.2 与 §6 交叉核对：kernel 内无任何未被论证的逐元素标量循环。

### 6. 验证方案 — pass

§8.1 golden 为 `torch.eq` PyTorch 参考实现（源语义依据，非 NPU 算法）；§8.2 精度标准按 dtype：比较结果 bool 域精确（exact），附来源（`test_eq.py` `_bool_compare` + manifest）；§8.3 L0 计划 12 个用例覆盖同形/大规模(chunks>1)/广播三模式/a 侧广播/标量广播/尾块非整除/bool 存储/整数全集（含 int8→f16、uint8→int8）/已知相等边界/**浮点 IEEE 角点（L0-10，NaN/±0，风险 R6）/vcmp 同形回归（L0-12）**。计划合理、门槛明确。

### 7. 完整性与一致性 — pass

模板 11 章节齐全，迁移类 §0.1–§0.7 小节齐全。全文无未填占位符（§3.3 的 `<outer divmod …>` 为示意记号、已注明由 Python 展开链生成，非模板占位符）。§3.5.3 引用的 5 个 `examples/` 路径经 Glob 逐一确认真实存在（`bitwise_xor/.../DESIGN.md`、`bitwise_and/.../DESIGN.md`+`RETROSPECTIVE.md`、`elementwise/vec_add_1d.py`、`flash_qla/tilelang_get_warmup_chunks.py`、`.../argmax_kernel/perf_opt/_argreduce_kernel.py`）；§0.7 引用的 wrapper/Op/manifest/test/bench 路径（`examples/TileOPs/tileops/kernels/elementwise/eq/eq.py`、`ops/elementwise/eq.py`、`manifest/elementwise.yaml`、`tests/ops/test_eq.py`、`benchmarks/ops/bench_eq.py`）亦全部存在。API 映射（§3.2）/内存规划（§4）/Tiling（§5）三处对 `(block_size,)` 同形 buffer 与 int8 承载的描述相互一致。

### 8. 算法优化分析 — warn

- **§1.6 章节存在性（pass）**：§1.6.0/§1.6.1/§1.6.2/§1.6.3 四子节齐全，单步逐元素类按 algorithm-research §2 采用轻量调研但四问逐问有结论。
- **调研存在性与完整性（pass）**：R1 候选表**含基线**（候选 0 = 源算法）且列出 5 个候选（算术差零 / 位异或零 / 规范布尔位 / 幅值饱和 / 比较域重映射）；R2 明确「无在线变体」并给结构依据（逐元素映射无 running 统计量、无跨元素依赖，单遍天然在线，且源码 §0.4 无 online 手段）——**独立复核成立**；R3 复杂度表覆盖四口径（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值）且基线在表、口径写清（A/B/N 元素数、E 输入字节、输出恒 1 字节）；R4 逐候选六维评估。`r3_metrics` 机械 pass。
- **调研结论正确性独立复核（warn）**：① 等价性独立推演——候选 1（`(a−b)==0`）对浮点因 `a−b` 舍入**非严格等价**、候选 2（`(a^b)==0`）对浮点因 NaN 位模式/±0 位模式**不成立**、候选 4 op 数 ×3——设计淘汰理由**成立**；采纳候选 5（int8/uint8→fp16 升域）由 E3 验证单射（全 256 值 exact-in-fp16）且相等性保持——成立。② 复杂度算术独立复算——基线/NPU 重设计 FLOPs（N + R·O(ndim) divmod 降 ~L 倍）与 §1.6.0 roofline 四 workload 流量/发射项**逐个复算全部一致**（9.2KB→0.01µs、75.5MB→71.9µs、64.2MB→61.2µs、83.9MB→79.9µs；发射项 1.5/76.8/205/153.6µs 均复算无误），**未漏计中间缓冲 GM 往返**（本算子每 task 一次进出、无中间 GM 往返，与 §4.4 一致）。③ 负向断言「无在线变体/无更低复杂度算法」对照 `algorithm-candidates.md` 命中行 `ALG-elementwise`（R1 CSE/除法乘倒数/cast 链合并、R2 逐元素天然单遍、R3 融合消中间 GM 往返）**逐条复核未被推翻**——比较本体 `a==b` 确为 1 op/元素下界，三种分解均 op 数 ≥2 或对浮点不成立，无已知等价化简公式。④ **亲和性判断与 pattern-library 存在一处需要补证的偏差（本维度 warn 的主因，见建议）**。
- **调研→设计一致性（pass）**：§1.6.0 选定算法（基线比较语义的 NPU 行/内层重设计 + R4 比较域映射 + R3 int8 承载）实际驱动 §1.4 算法描述、§3.1 公式拆解（以「优化后公式=原式」为输入）、§6 循环结构（无逐元素标量循环）；迁移侧 §0.5/§0.6 处置与调研选族一致，§0.4 优化意图由 §0.6 末尾总表逐项承接，无静默丢弃。
- **数学等价优化完整性（pass）**：§1.6.1 对选定算法六类逐项审视（恒等变形/稳定化/CSE/算子降代/访存削减/归约），结论「无公式级优化空间」有依据（比较本体 1 op 下界、无公共子式、无除法/超越函数/归约）；§0.6 R1/R3/R4 明确标注为语义保持的等价替换（非公式级优化），由 verify_equiv E1/E2/E3 覆盖——分类恰当。
- **等价性论证独立推演（pass）**：无 §1.6.1 采纳的公式优化项，但 R1/R3/R4 三项等价替换均经我独立推演并由 verify_equiv 重跑佐证（E1 承载等价/索引映射等价/E3 升域等价，16 行全 `EQUIV_PASS`，与内嵌表一致）。
- **向量化替代覆盖完整性（pass）**：§1.6.2 盘点 8 个循环/标量点，4 类逐元素点（比较/bool→int8/1 字节升域/标量广播物化）全部替代且替代 API 有 docs+探针双佐证；4 处保留（#5 行 base divmod 每 task 一次、#6 动态 tail、#7 persistent 调度循环、#8 host 元数据）逐项给出具体到本算子的理由（如 #5 指出 `T.arange`+divmod 向量化反而引入 `CONST-arange-scalar-materialize` ~1ns/elem 物化）。与 §3.3 伪代码、§6 循环结构交叉核对无遗漏。
- **不可替代理由充分性（pass）**：保留项均为 block 级索引/tile 级调度/host 元数据，理由具体、非「照源码写」。
- **替代 API 佐证（pass）**：`T.vcmp`/`T.vcast`/`T.vbrc`/`T.min`/`T.ceildiv` 均有 docs + examples 佐证，优先 v-prefix。
- **向量化轴与布局决策完整性（pass）**：§1.6.3 候选矩阵按 elementwise/广播类枚举 6 行（IO 原生+innermost 向量轴 / 核内 N 维 auto-broadcast tile / host 预广播 / 2-D `[R_blk,L]` tile / 逐元素标量 / 位运算近似），逐候选量化取舍；弃选有量化理由（#2 由 CG-2026-0021 探针实测不可编译、#3 host 物化违反 ascend-constraints §4 且净负、#5 落 Scala 无 128bit lane）。**弃选前提亲自核对**：#2 引用的 `T.vcmp.md` §2.2.2 同形约束与 CG-2026-0021 实测一致（我打开文档确认）。判定裕度依赖未实证常数（发射摊销律/b 行重读 L2 吸收率）处给出**实验裁决三件套**：主选 #1 + 备选 #4（含 buffer 形状/UB 预算/循环结构/分核三要素重算的完整结构）+ 裁决计划（代表 shape、指标、`<0.97×` 阈值、按 shape 分派、回写路径）——满足实验裁决模式要求。源码隐式轴（`threads×npt`）已对照并注明 GPU 轴非结论。
- **布局决策与 §3.3/§4/§6 一致（pass）**：`(block_size,)` 1-D UB 布局、coalesced 末维向量化轴、无 repack——与 §3.3 伪代码 buffer 形状、§4.5 UB 预算、§6 循环结构三方一致。
- **与 §3.1/§6 一致（pass）**：§3.1 以比较本体不变为输入；§6 无 §1.6.2 未论证的标量循环。

**建议级观察（不阻塞结论，供 Stage 1 修订/Stage 4 参考）**：
1. **`T.vcmp` int16 标量化陷阱未登记（本维度主要建议）**：§0.6 R4 映射表将 int16/int32/int64 一律走「原生域 `T.vcmp`」并在 R4 亲和表标 int16 为 Vector ✅，§9.2 亦要求「int16/int32/int64 用原生 vcmp，不得降级 fp16」。但 pattern-library 已实测登记 `PL-1.11-causal-mask-scalartrap`（`traps-compiler.md` L174）：**`vcmp` 在 int16 上对所有操作数形态（含文档合法的同 shape int16 张量）一律降级为标量指令执行**（编译警告 `will execute by scalar instruction with low efficiency`，[32,256] 实测 236µs/链）——与设计「int16 落 Vector」的亲和判定矛盾。manifest 四 workload 均为 fp32、int16 仅出现在 L0-8 正确性用例（正确性不受影响），故不构成 Stage 3 编译/运行/精度阻塞；但 int16/int32/int64 属 dtype 契约，int16 路径的发射/标量行为会成为 Stage 4 的性能天花板。建议 DESIGN.md 补一条风险行（R15：int16 vcmp 标量化），并评估**数学等价的替代**——int16 → **fp32** 精确升域（fp32 24-bit 尾数可精确表示全部 int16，且 fp32 vcmp 非标量化）后 vcmp，可规避标量化且保持等价；int32/int64 因 fp32 非单射仍须原生域。
2. **§3.4 一处来源标注偏差（轻微）**：`T.vbrc` int8 dst 挂死条目标为「CG-2026-0019 记录」，但 CG-2026-0019 实为「`T.vand` 不接受标量右操作数」；正确来源是同条目同时引用的 `bitwise_and RETROSPECTIVE TRAP-vbrc-int8-vector-core-timeout`（已实测核对该 RETROSPECTIVE 存在且描述一致）。建议删去 CG-2026-0019 的错引，仅保留 RETROSPECTIVE 来源。

**本算子已登记三项特有风险的处置核对（结论：均已合理处置并列为 Stage 3 必测项）**：
- `T.vcmp` 无自动广播（CG-2026-0021）：§0.6 R1 以 `T.vbrc`/连续段显式同形物化，L0-12 强制回归；文档 §2.2.2 与探针 `probe_vcmp_bcast`（双形态 FAIL）双证。**合理**。
- bool GM 输出运行期挂死（CG-2026-0022）：§0.6 R3 改 int8 0/1 承载 + Op 层 `view(torch.bool)`，wrapper 适配列为 R11（Stage 3/5），E1 机器验证；文档「编译失败」与实测「运行期 507034」差异已登记 CG。**合理**。
- 浮点 NaN/±0 未设计期探针覆盖（R6）：设计明确登记为风险 R6 并强制 L0-10（NaN 对 NaN、+Inf、±0、次正规数），且定义不符时的回退路径。**合理**（设计期以探针预算受限推迟到 L0 门槛，符合 risk 登记 + 必测项要求）。

