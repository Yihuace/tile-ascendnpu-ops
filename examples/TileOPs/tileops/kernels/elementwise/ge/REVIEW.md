# ge_fwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/ge/ge_fwd_kernel/DESIGN.md
- 任务类型: 迁移（`scenario=migration`，`migration_mode=harness`）
- 源算子: `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（`BinaryKernel` + `GeFwdKernel`/`GeBoolStorageFwdKernel`；Op 类 `/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/comparison.py` `GeFwdOp`）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 2

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `python examples/ge/ge_fwd_kernel/verify_equiv.py` → `OVERALL: EQUIV_PASS`（exit 0）。执行输出与 §1.6.1 内嵌结果表**逐项一致**：E1（bool int8 vs uint8 = 0 / view(int8 0/1) = 0）、E2（10 组形状 a_mismatch=b_mismatch=oob=0）、E3（int8/uint8 升 fp16 精确 + `>=` 保持 = 0）、E4（int16 全 65536 值升 fp32 精确 + `>=` 保持 = 0）、E5（256×256 全枚举 + 1M 随机 + 相等位注入 = 0）全 `EQUIV_PASS`；否决证据 E-NAIVE=523878、E-NAN complement=4 / disjunction=0 亦与设计一致。**脚本存在、执行成功、与内嵌表一致、全 EQUIV_PASS。**
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design <path>` JSON 摘要——
  - `r3_metrics`: **pass**（四口径 FLOPs/访存量/扫描遍数/中间缓冲峰值齐全）。
  - `ub_budget`: **skip**（advisory checker 未解析到分级 buffer 表）→ **人工已核**（见维度 2：§4.5 逐 dtype 手工合计 ×1.7 膨胀复算 ≤136KB < 192KB，算术正确）。
  - `core_split`: **skip**（未解析出 `num_logical` 字段）→ **人工已核**（见维度 3：§5.5 逻辑核数表按 block_size=4096 独立复算全部命中；物理核数本机实查 = 20 ⇒ 40）。
  - `l0c_budget`: **skip**（设计无 Cube 路径 API）→ n/a（纯 Vector，无 L0C）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**已亲自 Read 源算子全文**（`/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py` 全文 + `ops/elementwise/comparison.py` + Stage 0 提取件 `examples/TileOPs/tileops/kernels/elementwise/ge/_ge_fwd_kernels.py` + NPU wrapper `ge.py` + `ops/elementwise/_base.py` `_BoolOutputBinaryOp`），与 §0 逐项核对：

- **语义理解正确性 — pass**：§0.1 数学语义 `y = (a >= b)` 与源码 `GeFwdKernel.op_func(a,b) = a >= b`（L1786–1788）一致；输出 bool 契约与 `OUTPUT_DTYPE = torch.bool`（L1783）一致；`_BINARY_FULL_DTYPES = bool|uint8|int8|int16|int32|int64|fp16|bf16|fp32`（源 L166–170）与 §0.1/§0.2 一致；NaN→False / ±0 相等 / ±Inf 实数序为 `torch.ge`（IEEE 754 部分全序）语义，正确。**`>=` 排序符号相关**的判定与源码一致（GPU 在原生有/无符号域做 `>=`；对比 `EqFwdKernel.op_func = a == b` L1708 符号无关）。
- **I/O 契约 — pass**：§0.2 输出 `(N_total,)`、`out_idx=[2]`，与提取件 `main` 签名 `y: T.Tensor((N_total,), out_dtype)`（L53–57 快路径 / L76–80 广播路径）、`out_dtype = output_dtype or dtype`（L45）一致；Op 层传 `"bool"`（`GeFwdKernel.__init__` L199 `output_dtype=self.output_dtype_str`，`output_dtype_str = dtype_to_str(torch.bool)` L181）一致；host 侧 `contiguous().view(-1)` + `coalesce_broadcast_dims`（`ge.py` L171）+ `reshape(out_shape)`（`_base.py` `_BoolOutputBinaryOp._eager_forward` L1119）一致。
- **算法解读完整性 — pass**：§0.3 计算步骤分解（host coalesce / block_size+grid / 同形快路径 / 广播 divmod 路径 / host reshape / bool 存储路径）覆盖源码全部计算语句，无遗漏无臆造。同形快路径 `idx=(bx*threads+i)*npt+j; y[idx]=a[idx]>=b[idx]`（提取件 L59–61）、广播路径 `flat_idx=(bx*threads+i)*npt+j; ...; y[flat_idx]=a[a_off]>=b[b_off]`（L82–87）、`_compute_broadcast_offsets` 编译期 unrolled mixed-radix divmod 链（L161–177）逐条对应。bool 路径 `bitwise_or(a, bitwise_xor(b,1))`（源 L1795–1796）与 §0.3 步骤 5 一致，且对 0/1 值域 `a|~b ≡ a>=b` 独立复核成立。
- **优化手段覆盖完整性 — pass**：§0.4 九项逐一对应源码（`_make_binary_explicit` 的 `T.Parallel(threads,npt)` / 连续段 coalesced / `coalesce_broadcast_dims` / 编译期 unrolled divmod / `_is_contiguous_same_shape` 快路径 / `register_copy`（L996–1003，且标注「仅同形且非 bool 输出」——ge 为 bool 输出，故本条实际不选，已隐含披露）/ `lru_cache(maxsize=32)` / bool 走 uint8 存储 / 单条 `>=`）；「无 SMEM tiling / warp shuffle / 异步流水 / swizzle」与源 docstring「no shared memory」一致，无遗漏。
- **耦合性判定合理性 — pass**：§0.5 每条目四态处置 + 依据；warp shuffle 等 GPU 专属被正确判「重新设计/舍弃」；`threads=`/`npt` kwarg 舍弃有理由（npuir `T.Kernel` 无此参数，wrapper K11 折叠为 `block_size`）；**`>=` 符号相关使 uint8→int8 承载从 eq 的「单向等价替换」升级为「重新设计（R5）」**——与源语义一致，是对同族先例的重要修正。
- **NPU 重设计可行性 — pass**：R1（广播索引）/R2（并行分核）/R3（1 字节承载）/R4（比较域映射）/R5（uint8 无符号修正）均有「源方案 → NPU 新算法 → 语义保持论证」三要素；新算法满足 NPU 约束（一维 Kernel、UB 层级、静态边界、vbrc 物化对 vcmp 同形约束）；语义保持论证经 `verify_equiv.py` E2/E1/E3/E4/E5 机器验证（与 §1.6.1 内嵌表一致）。
- **设计与决策一致性 — pass**：§1–§7 描述的是 §0.5/§0.6 决策后的 NPU 算法（§3.1/§3.2 比较域映射 §0.6 R4、§3.3 uint8 修正链 R5、§5.5 persistent 分核 R2、§6 无逐元素标量循环），无「§0 判重设计、§3 照抄源方案」的脱节；§1.6.1 优化意图承接总表无静默丢弃。
- **golden 独立性 — pass**：§8.1 `golden_ge` 以 §0.1 语义为依据（`torch.ge` 广播直算），显式声明不复刻 NPU 行/内层索引、vcmp/vcast 链、uint8 修正、persistent、同形物化；与源仓 `test_ge.py` 的 `ref_program = torch.ge(a,b)` 一致。

**结论：源算子解读与源码逐条一致，硬件耦合性判定合理，NPU 重设计可行且语义保持；golden 独立。**

### 1. API 可行性 — pass

§3.2/§3.3 列出的 API 均在 `docs/Tilelang.language/`、`examples/` 或 `tilelang/language/` 找到佐证，且经实查确认命名空间可用：

| API | 佐证 | 状态 |
|-----|------|------|
| `T.vcmp` | `docs/Tilelang.language/比较操作/T.vcmp.md` §2.1/§2.2.2/§2.4（示例即 `"ge"`） | ✅（`hasattr(T,'vcmp')` True） |
| `T.vcast` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1：`i8→f16`、`ui8→f16`、`i16→f32`、`f16→i8`、`i1→f16` 均在 dtype 矩阵（已逐行核对） | ✅ |
| `T.vbrc` | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2（size-1→dst）；int8 dst 文档标 √ 但实测挂死（见 R4 绕法） | ✅（绕法已述） |
| `T.vadd` | `docs/Tilelang.language/数学操作/T.vadd.md` §2.2（src1 可标量） | ✅ |
| `T.vselect` | `docs/Tilelang.language/条件操作/T.vselect.md`；`vcmp.md` §2.4 示范 | ✅（`T.vselect` 导出确认） |
| `T.alloc_ub`/`T.copy`/`T.min`/`T.ceildiv`/`T.serial`/`T.Kernel(is_npu=True)` | `vec_add_1d.py` + 各 doc | ✅ |
| `T.floordiv`/`T.floormod` | `tilelang/language/__init__.py`（`floordiv`/`floormod` 存在）+ 同族 `eq_fwd_kernel.py` L241–242 实际使用 | ✅ |

Developer 模式用自动同步 + v-prefix：与 §2 模式结论一致，且同族 `eq_fwd_kernel.py`（已实现）同构。**建议级**：§3.3/§4.3 的 uint8 修正链引用 `zero_f16` 作为 `T.vcmp(a_c, zero_f16, m, "lt")` 的第二个操作数，但 §4.3 buffer 清单未列零操作数 buffer——因 `T.vcmp` 要求严格同形（CG-2026-0021，`T.vcmp.md` §2.2.2）且不接受标量，Stage 3 必须显式物化一个 `(block_size,)` 零 buffer（见维度 7 建议）。

### 2. 内存层级规划 — pass

- **搬运路径完整**：§4.4 给出每 task 的 GM→UB（`T.copy`）→ Vector 计算 → UB→GM 单阶段路径，**无 GM→L0 越级、无 L1/L0 参与**（纯 Vector）。
- **Buffer 层级标注**：§4.3 每个 buffer 标注 UB（`a_ub/b_ub/a_c/b_c/m/sp/cond/yf/y8/a1/b1/a1c/b1c`）。
- **UB 预算（机械 ub_budget skip → 人工复核通过）**：§4.5 逐 dtype 手工合计 ×1.7 膨胀独立复算——
  fp16/bf16 8 B/elem → 4096×8=32KB → ×1.7≈55KB；fp32/int32 12 → 48KB→≈82KB；int64 20 → 80KB→≈136KB；int16 16 → 64KB→≈109KB；int8/bool 10 → 40KB→≈68KB；uint8 13 → 52KB→≈88KB。**全部 ≤136KB < 192KB**，算术与结论正确。block_size 上限（int64 ≤5782 最紧）推算亦正确。
- **L0C 容量**：n/a（无 Cube/L0C）。

### 3. Tiling 策略 — pass

- **Block 划分明确**：§5.2 给出 `block_size = 4096` 及六条量化理由（innermost 单块覆盖 L∈{3136,4096}、尾轴 32B 对齐、向量宽度整除 0% lane 浪费、UB 上限、同形 chunks 分块、同族一致性）。一维划分（无 block_M/block_N），符合逐元素/广播类特征。
- **对齐约束**：`block_size×E ∈ {4K,8K,16K,32K}` 均 32B 整数倍；末维向量宽度整除。
- **非整除处理**：§6.4 `tail = T.min(block_size, L − inner_off)` 截断（`vec_add_1d.py` 官方先例），非 GEMM 类无需 padding+crop。
- **分核策略三要素（机械 core_split skip → 人工复核通过）**：
  - ① **逻辑核数计算**：`num_logical = R × ceildiv(L, block_size)`。独立复算 §5.5 表（block_size=4096）：smoke-1k `[1024]`→ coalesced `(1024,)`，k=1,R=1,L=1024,chunks=1 ⇒ **1**；hidden-state `[2048,4096]`→coalesced `(8388608,)`，L=8388608,chunks=2048 ⇒ **2048**；cnn-feat `[16,256,56,56]^[256,1,1]`→coalesced `(16,256,3136)`，k=3,R=4096,L=3136,chunks=1 ⇒ **4096**；row-broadcast `[4096,4096]^[1,4096]`→coalesced `(4096,4096)`，k=2,R=4096,L=4096,chunks=1 ⇒ **4096**。**四行全部与设计一致**（coalesce 过程亦独立复现）。
  - ② **物理核数依据（实查）**：§5.5 记录 `NPUUtils.get().get_aicore_num()` 查询代码与实际返回值 **20**，纯 Vector 翻倍 ⇒ **40**——与 `core-split-strategy.md` §1 要素②一致。**本 Agent 独立实查本机同样返回 20**（`python -c "from tilelang.utils.npu_utils import NPUUtils; print(NPUUtils.get().get_aicore_num())"` → 20），非文档假设/经验值。
  - ③ **规模判定与分核方案**：smoke-1k（num_logical=1 ≤ 40）判「无需适配」附依据；其余三 workload（2048/4096 ≫ 40）判「极大规模」⇒ persistent：`num_kernels = min(num_logical, 40)`、核内 `for i in T.serial(num_local)`（**静态边界**，`num_local = ceildiv(num_logical, num_kernels)`）、grid-stride `task = i*num_kernels + cid` + `if task < num_logical` 屏蔽，负载不均 ≤1 task/核。与标准要素③的静态边界要求一致。

### 4. 技术约束检测 — pass

- **三维 Kernel**：不涉及（源即一维 grid，NPU 一维 `T.Kernel(num_kernels, is_npu=True)`）。
- **GEMM 非整除**：不涉及（纯 Vector，无 GEMM）。
- **L0C 溢出**：不涉及（无 L0C）。
- §3.5.1 逐条列本项目已知限制（GPU 专用 API、物理核数、host 禁改、uint8 不可用、bool GM 挂死、vcmp 无广播/不支持 int8/uint8、int16 标量化、`>=` 符号相关），处理方案与 §0.6 一一对应。

### 5. 循环与同步 — pass

- **循环结构**：§6.1 明确仅 block/task 级调度循环（`T.Kernel` persistent 并行 + `T.serial(num_local)` 串行；元素级全部向量化），**无任何逐元素标量循环**；与 §1.6.2 结论交叉一致。
- **同步与模式匹配**：§7 无手动同步点，Developer 自动同步，理由（task 完全独立、无跨核依赖、无 workspace）充分。
- **尾块处理**：§6.4 明确 `tail = T.min(...)` 截断 + 零起点 dst 切片，覆盖任意广播形状。
- 不使用 `T.Pipelined` 的理由（单阶段 load→compute→store，无 K 维累加；发射项为主 → 交 Stage 4）合理。

### 6. 验证方案 — pass

- **Golden 函数**：§8.1 含 `golden_ge` 草案（`torch.ge` 广播直算），源自 §0.1 语义，独立于 NPU 算法。
- **精度标准**：§8.2 按 dtype 给出比对方式；bool 域比较为 **exact**（无舍入）——合理，优于默认 atol/rtol 口径（并附各 dtype 来源）。
- **L0 测试计划**：§8.3 14 个代表性 case，覆盖 smoke/大规模/broadcast 三模式/a 侧广播/标量/尾块非整除/bool 存储/整数全集/相等边界/**浮点 NaN·±0·±Inf 角点（L0-10）**/**vcmp 同形回归（L0-12）**/**uint8 无符号序 ≥128（L0-13/L0-14）**。特别是 **L0-13/L0-14 针对源测试数据盲区（`test_ge_integer_dtype` uint8 仅取 [0,16]）补 ≥128 用例**——已核对 `test_ge.py` `_gen_int_inputs` 确认盲区真实存在，覆盖设计到位。

### 7. 完整性与一致性 — pass

- **章节齐全**：迁移模板 11 章节 + §0（0.1–0.7 齐全）。
- **无占位符**：全文未完成占位标记扫描为空（无占位符 token）；§3.3 的 `<outer divmod …>` 为显式说明的 Python 生成式示意，非未完成占位标记。
- **同类引用真实**：§3.5.3 全部 6 条引用路径逐一存在核对（`examples/eq/eq_fwd_kernel/DESIGN.md`+`RETROSPECTIVE.md`、`bitwise_xor`、`bitwise_and`、`examples/elementwise/vec_add_1d.py`、`examples/flash_qla/tilelang_get_warmup_chunks.py`、argmax perf_opt `_argreduce_kernel.py`）——全部 OK。
- **内部一致**：§3.2 API 映射、§4 内存规划、§5 Tiling 三处对 `(block_size,)` 同形 `a_c/b_c/cond`、int8 承载、int16→fp32 的描述相互一致。
- **建议级（不影响结论）**：
  1. §3.3/§4.3 的 uint8 修正链以 `zero_f16` 作为 `T.vcmp(..., "lt")` 的第二操作数，但未在 buffer 清单物化该零 buffer。因 `T.vcmp` 要求严格同形（CG-2026-0021）且不接受标量操作数，Stage 3 实现时须补一个 `(block_size,)` 零（或经 `T.vbrc(0.0)` 产生）的 f16 buffer，否则 uint8 路径 trace 期即失败。建议在 §3.3 注记 ⑦ / §4.3 增列 `zero`（比较域 dtype）buffer。
  2. §5.2 推荐 `block_size=4096`，而 wrapper `GeFwdKernel.default_config` 现返回 1024（浮点）/2048（整数）/4096（bool 存储）；§4.5 已如实记录「由 wrapper config 传入」。为免集成默认值与设计推荐不一致（影响发射项摊薄效率，非正确性），建议 §5.2/§10.4 明确 Stage 5 通过 config 覆盖为 4096 或确认 Stage 4 扫描基线。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性** — pass：含 §1.6.0 调研 / §1.6.1 数学等价优化 / §1.6.2 向量化替代 / §1.6.3 向量化轴与布局决策四子节，且明确标注单步逐元素类采用轻量调研深度（四问各一行但不可跳过）。
- **算法调研存在性与完整性** — pass：
  - R1 候选表**含基线（候选 0 = 源算法）**，候选 1–7 逐条给等价性初判与收益方向，「是否纳入 R3 对比」列齐备。
  - R2 在线算法：**有明确结论「无在线变体需求（天然单遍）」+ 结构依据**（`y_i` 仅依赖同位元素，无 running 统计量/无跨元素依赖）。
  - R3 复杂度表**覆盖四口径**（FLOPs / 访存量 Bytes / 扫描遍数 / 中间缓冲峰值）+ 可并行度，基线在表，口径写清（A/B/N 展平元素数、E=elem_bytes、输出恒 1 字节）。
  - R4 硬件亲和**逐候选评估**（计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水融合）。
  - 调研结论含选定算法（基线比较语义 + §0.6 R1/R4/R5/R3 域/承载重设计）与关键依据；「无更优替代」写明调研范围（① ALG-elementwise 参考表命中行 ② 恒等变形自查 ③ 本仓 elementwise examples ④ pattern-library ⑤ 源结构分析 ⑥ kb_search 命中 ⑦ E-NAN 浮点记录）。
- **调研结论正确性（独立复核）** — pass：
  - **等价性独立推演**：候选 1 补集 `¬(a<b)` 在 NaN 下 `¬(NaN<x)=True` 而 `NaN>=x=False` → 不成立；独立复算 `verify_equiv.py` E-NAN 的 6 角点得 4 处不一致（与设计一致）；析取式 `(a>b)∨(a==b)` 对 NaN 两子式均 False → 与 `ge` 一致，0 违反。候选 7 升域精确性经 E3/E4（int8/uint8 全 256 值在 fp16 精确、int16 全 65536 值在 fp32 精确）与 E5（无符号修正 256×256 全枚举）机器验证成立。
  - **复杂度算术独立复算**：R3 表中间缓冲峰值 O(block_size)、扫描遍数 1、访存量 `(A+B)·E + N·1`（与 manifest roofline `(A + B) * elem_bytes + N` 一致）；NPU 重设计路径**未漏计中间缓冲 GM 往返**（显式「全 UB 内，无额外 GM 流量」）；row-broadcast 的 b 行重读被显式列为未实证项并给上限口径（≈143.8µs），未藏匿。`design_calc_check.py` `r3_metrics=pass`。
  - **负向断言主动复核**：①「无在线变体」对照 `algorithm-candidates.md` 命中行 `ALG-elementwise`（第 98–104 行）R2「逐元素映射天然单遍」——**一致，无矛盾**；②「无化简公式」对照该行 R1（公共子表达式消除 / 除法乘倒数 / cast 链合并）——§1.6.0 结论 ① 与 §1.6.1 表逐项覆盖（`a`/`b` 各用一次无公共子式；无除法；比较本体无 cast，仅升域 + 不可省的 uint8 修正链），**无漏候选**；③ 亲和性淘汰依据（int16 原生 vcmp 标量化）对照 `pattern-library/attention.md` PL-1.11「vcmp int16 全形态标量化（236µs/链 @[32,256]）」——**数值与措辞完全一致**。
  - **并列候选裁决路径**：候选 7-2D（方案 #4）与主选 #1 的判定裕度依赖未实证常数（发射摊销律 / 2-D op 开销 / b 行重读 L2 吸收率），已指向 §1.6.3 实验裁决计划（非纸面拍板）。
  - **互联网来源**：§1.6.0 未引用互联网候选（n/a）。
- **调研 → 设计一致性** — pass：§1.4 算法描述、§3.1 公式拆解、§6 循环结构与 §1.6.0 选定算法同口径；迁移任务 §0.5/§0.6 处置（R1/R2/R3/R4/R5）与调研选族一致；源算法优化手段 1–9 经 §0.6 末尾总表逐项承接，**无静默丢弃**。
- **数学等价优化完整性** — pass：§1.6.1 对选定算法逐类别核查（恒等变形 / 稳定化 / 公共子表达式 / 算子降代换 / 访存削减 / 归约），结论「无公式级优化空间」有依据。**无采纳的公式级优化项**，故「原式 → 优化后 → 等价性论证 → 收益」四要素为空集；§0.6 R1/R3/R4/R5 被正确定性为**语义保持的等价替换（存储/索引/操作数形态/比较域）**并全部纳入 `verify_equiv.py` 机器验证——处理恰当。
- **等价性论证正确性（独立推演）** — pass：R5 修正 `u = s + 256·(s<0)` 独立推演——`s∈[-128,-1]` 时 `u=s+256∈[128,255]`、`s∈[0,127]` 时 `u=s∈[0,127]`，恰为字节真无符号值，且 fp16 对 `u∈[0,255]` 精确 ⇒ `u_a>=u_b ⟺ a_u8>=b_u8`，成立；`T.vselect(m, sp, a_c, a_c)`（Cond?A:B）语义与 `(s<0)?s+256:s` 一致。R4 升域保序在 E3/E4 全值域成立。
- **向量化替代完整性** — pass：§1.6.2 盘点 §0.6 方案**全部**循环/标量点共 10 项（与 §3.3 伪代码、§6 循环结构交叉核对，无遗漏）：#1 比较、#2 bool→int8 落盘、#3 1 字节升域、#4 int16 升域、#5 uint8 无符号修正、#6 标量广播物化已全部向量化；#7 行 base divmod、#8 尾块长度、#9 task 调度循环、#10 host 元数据保留且逐项给具体理由。§6 无 §1.6.2 未论证的标量循环，双向一致。
- **不可替代理由充分性** — pass：#7（每 task O(ndim) 标量 divmod，非逐元素热点；`T.arange` 向量化反引入 `CONST-arange-scalar-materialize` ~1ns/elem）、#8（动态边界，`T.min` 先例，padding 违反 host 禁改）、#9（persistent tile 级调度循环）、#10（host 元数据）——均具体到本算子且引证据，非泛泛而谈。
- **替代 API 可行性** — pass：替代 API 均有 docs + 机器验证双佐证（优先 v-prefix）；唯一建议级缺口见维度 1/7（uint8 修正链的零操作数 buffer 未物化）。
- **向量化轴与布局决策完整性** — pass：§1.6.3 候选矩阵（7 项）对逐元素/广播类枚举了「原生连续轴 + innermost 向量化」与「vcmp 自动广播（不可实现）」「host 预广播」「2-D 块化」等；逐候选有量化/结构理由；**弃选方案有量化依据**：候选 2 引 CG-2026-0021 + `T.vcmp.md` §2.2.2（本 Agent 已亲自打开文档核对「输入 tensor A 和 B 必须具有相同的 shape」），候选 3 引 `ascend-constraints.md` §4（host 禁改输入，已核对条款），候选 7 引额外 GM 往返。**未把 I/O 契约当内部布局依据**（显式区分 I/O layout 契约 vs 核内布局自由变量）；迁移任务对照了源 GPU 轴（threads×npt 是硬件组织输入而非结论）。轴质量评分为量化（整除性/尾 lane 浪费 0%/累加链无/repack 无/UB 容量）。
- **实验裁决模式完整性** — pass：主选 #1 与备选 #4 并列；备选 #4 结构完整（buffer 形状 `(R_blk,L)`、a 连续段载入、b 经 `T.vbrc` 物化、rank-2 vcmp/vcast/修正链、bool→int8 同 2-D、尾块 tail 截断、分核三要素按 2-D 粒度重算 `num_logical = ceildiv(R,R_blk)×ceildiv(L,block_size)`）；裁决计划含代表 shape（row-broadcast/cnn-feat fp32）、指标（msprof Task Duration + vec_ratio + mte2_active_bw）、**明确判定阈值（备选 < 主选 ×0.97 才翻转）**、回写路径、按 shape 分派可行性（lru_cache 特化天然支持）；int64 容量否决（R_blk=1 退化）已记。
- **布局决策与实现一致** — pass：§1.6.3 选定 `(block_size,)` UB buffer + coalesced 末维向量轴，与 §3.3 伪代码 buffer 形状、§4.5 UB 预算、§6 循环结构三方一致；重排路径「无」与 §3.3 一致。
- **与下游章节一致** — pass：§3.1 以「优化后公式 = 原式（比较本体不变）+ R1/R4/R5/R3 等价替换」为输入；§6 循环结构与 §1.6.2 结论一致（判定向量化的点无对应逐元素标量循环）。

**维度 8 复核小结**：调研四问齐全且深度与算子复杂度匹配；负向断言（无在线变体 / 无化简公式）对照 `algorithm-candidates.md` 命中行 `ALG-elementwise` 与源码证据逐条复核无矛盾；复杂度四口径独立复算无漏计；选定算法驱动 §1.4/§3.1/§6；等价性论证逐项独立推演 + `verify_equiv.py` 重跑一致；向量化覆盖与 §3.3/§6 交叉核对完整。**pass。**

## 检视结论说明

迁移任务 9 维度全部 pass（维度 0/2/3/8 的阻塞级结论均附源码核对证据与独立复核），2 项建议级问题已在维度 1/7 内记录（uint8 修正链零操作数 buffer 物化、block_size 集成默认值对齐），均不阻塞 Stage 3 推进。
