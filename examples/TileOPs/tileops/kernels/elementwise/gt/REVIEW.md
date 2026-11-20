# gt_fwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/gt/gt_fwd_kernel/DESIGN.md
- 任务类型: 迁移（`scenario=migration`，`migration_mode=harness`）
- 源算子: `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（`_make_binary_explicit` @ L556、`BinaryKernel` @ L890、`GtFwdKernel` @ L1739、`GtBoolStorageFwdKernel` @ L1751）；Op 类 `/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/comparison.py` `GtFwdOp` @ L48；Stage 0 提取件 `examples/TileOPs/tileops/kernels/elementwise/gt/_gt_fwd_kernels.py`
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 3（均随对应维度内联说明，不构成阻塞）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `python examples/gt/gt_fwd_kernel/verify_equiv.py` → `OVERALL: EQUIV_PASS`（exit 0）。执行输出与 §1.6.1 内嵌结果表**逐项一致**：E1（bool int8 vs uint8 = 0 / view(int8 0/1) = 0）、E2（10 组形状 a_mismatch=b_mismatch=oob=0，含 16.78M 索引的 row/a-broadcast 用例）、E3（int8/uint8 精确升 fp16 + `>` 保持 = 0）、E4（int16 全 65536 值升 fp32 精确 + `>` 保持 = 0）、E5（256×256 全枚举 + 1M 随机 + 相等位注入 = 0）全 `EQUIV_PASS`；否决/角点证据亦一致：E-NAIVE=523878、E-NAN complement=4 / conjunction=0 / gt-on-NaN=0、E-ZERO=0。**脚本存在、执行成功、与内嵌表一致、全 EQUIV_PASS。**（§1.6.1 判定「无公式级采纳项」，机器验证覆盖的是 §0.6 R1/R3/R4/R5 的等价替换，重跑结果一致。）
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/gt/gt_fwd_kernel/DESIGN.md` JSON 摘要——
  - `r3_metrics`: **pass**（四口径 FLOPs / 访存量 Bytes / 扫描遍数 / 中间缓冲峰值齐全）。
  - `ub_budget`: **skip**（advisory checker 未解析到分级 buffer 表）→ **人工已核**（见维度 2：§4.5 逐 dtype 手工合计 ×1.7 膨胀，fp16 55KB / fp32 82KB / int64 136KB / int16 109KB / int8 68KB / uint8 88KB，全部 ≤192KB；block_size 上限复算 14456/9637/5782/7228/11565/8897 全部算术正确）。
  - `core_split`: **skip**（未解析出 `num_logical` 字段）→ **人工已核**（见维度 3：§5.5 逻辑核数表按 block_size=4096 独立复算 {1, 2048, 4096, 4096} 全部命中；物理核数本机实查=20 ⇒ 40）。
  - `l0c_budget`: **skip**（设计无 Cube 路径 API）→ n/a（纯 Vector，无 L0/L0C）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**已亲自 Read 源算子全文**（`/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py` 全文 + `ops/elementwise/comparison.py` + `ops/elementwise/_base.py` `_BoolOutputBinaryOp` + Stage 0 提取件 `_gt_fwd_kernels.py` + NPU wrapper `gt.py`），与 §0 逐项核对：

- **语义理解正确性 — pass**：§0.1 数学语义 `y = (a > b)` 与源码 `GtFwdKernel.op_func(a,b) = a > b`（源 L1746–1748）一致；输出 bool 与 `GtFwdKernel.OUTPUT_DTYPE = torch.bool`（L1743）一致；`_BINARY_FULL_DTYPES = bool|uint8|int8|int16|int32|int64|fp16|bf16|fp32`（源 L166–170）与 §0.1/§0.2 一致；NaN→False / 相等（含 ±0、`a>a`）恒 False 为 IEEE 754 严格序语义，与 `torch.gt` 一致。**`>` 排序符号相关**的判定与源码一致（GPU 在原生有/无符号域做标量 `>`；对比 `EqFwdKernel.op_func = a == b` L1708 符号无关）。bool 存储路径源码 `bitwise_and(a, bitwise_xor(b,1))` = `a & ~b`（L1756）与 §0.1/§0.3 步骤 5 一致——与 `GeBoolStorageFwdKernel` 的 `bitwise_or(a, bitwise_xor(b,1))` = `a | ~b`（L1796）**互异且各自正确**（0/1 值域上 `a&~b ≡ a>b`、`a|~b ≡ a>=b`，独立复核成立）。
- **I/O 契约 — pass**：§0.2 输出 `(N_total,)`、`out_idx=[2]`，与提取件 `main` 签名 `y: T.Tensor((N_total,), out_dtype)`（快路径 L54–57 / 广播路径 L79–83，设计引作 L76–80，行号略偏但指向同一 `y` 声明，不影响语义）、`out_dtype = output_dtype or dtype`（L45）一致；Op 层传 `"bool"`（`gt.py` L181 `output_dtype_str`）一致；host 侧 `contiguous().view(-1)` + `coalesce_broadcast_dims`（`gt.py` L171）+ `reshape(out_shape)` + `view(torch.bool)`（`_base.py` `_BoolOutputBinaryOp._eager_forward` L1114–1119）一致。
- **算法解读完整性 — pass**：§0.3 计算步骤分解（host coalesce / block_size+grid / 同形快路径 / 广播 divmod 路径 / host reshape / bool 存储路径）覆盖源码全部计算语句，无遗漏无臆造。提取件同形路径 `idx=(bx*threads+i)*npt+j; y[idx]=a[idx]>b[idx]`（L64–66）、广播路径 `flat_idx=…; a_off,b_off=_compute_broadcast_offsets(...); y[flat_idx]=a[a_off]>b[b_off]`（L87–91）、`_compute_broadcast_offsets` 编译期 unrolled mixed-radix divmod 链（L161–177）逐条对应。**提取脚本丢失第二操作数索引的修正确认**（源 `_make_binary_explicit` 用 `op_func(a[a_off], b[b_off])`，提取件内联为 `a[a_off] > b[b_off]`，`b_off` 已恢复）。host 侧逻辑（lru_cache(maxsize=32) 特化、`out_dtype = output_dtype or dtype`、bool 走 `GtBoolStorageFwdKernel`）已列出。
- **优化手段覆盖完整性 — pass**：§0.4 九项逐条对应源码（`_make_binary_explicit` 的 `T.Parallel(threads,npt)` / 连续段 coalesced / `coalesce_broadcast_dims` L369 / 编译期 unrolled divmod / `_is_contiguous_same_shape` 快路径 L454 / `register_copy`（L996–1003，且明确标注「仅同形且非 bool 输出」——gt 为 bool 输出，本条实际不选，已披露）/ `lru_cache(maxsize=32)` / bool 走 uint8 存储 L1751–1756 / 单条 `>`）。「无 SMEM tiling / warp shuffle / 异步流水 / swizzle」与源 docstring「no shared memory」一致，无遗漏。默认策略核对：`GtFwdKernel.DEFAULT_STRATEGY = "explicit_parallel"`（L1744）且 bool 输出使 register_copy 分支不触发（L1000），与 §0.3 `threads=256,npt=4/8` 一致；`_default_block_size`（`gt.py`）浮点 1024 / 整数 2048 / bool 存储 4096 与 §0.3 一致。
- **耦合性判定合理性 — pass**：§0.5 每条目四态处置 + 依据；warp shuffle/SMEM 等 GPU 专属无遗漏（源本无）；`threads=`/`npt` kwarg 舍弃有理由（npuir `T.Kernel` 不接受，wrapper K11 折叠为 `block_size`）；`T.Parallel` 两级线程判「重新设计」、fragment 寄存器判「等价替换（→UB）」与 ascend-constraints 不矛盾。**关键在于 gt 的 `>` 符号相关使 uint8→int8 承载从 eq 的「单向等价替换」升级为「重新设计（R5）」**——与源语义一致（E-NAIVE 523878/1M 佐证不可省），是对同族先例的正确修正。
- **NPU 重设计可行性 — pass**：R1（广播索引：行 base + innermost 向量化 + 显式同形物化）/R2（并行与分核）/R3（1 字节承载）/R4（比较域映射 + 同形 `T.vcmp("gt")`）/R5（uint8 无符号域修正）均有「源方案 → NPU 新算法 → 语义保持论证」三要素；新算法满足 NPU 约束（一维 Kernel、UB 层级、静态边界、`T.vcmp` 同形约束下用 `T.vbrc`/连续段物化）。语义保持论证经 `verify_equiv.py` E2/E1/E3/E4/E5 机器验证（与 §1.6.1 内嵌表一致）。R1 末维 stride∈{0,1} 的闭合性论证（(0,0) 仅在 L==1 出现）与 `coalesce_broadcast_dims` 的 `_make_strides` 末维语义一致。
- **设计与决策一致性 — pass**：§1–§7 描述的是 §0.5/§0.6 决策后的 NPU 算法（§3.1/§3.2 比较域映射 §0.6 R4、§3.3 uint8 修正链 R5、§5.5 persistent 分核 R2、§6 无逐元素标量循环、§4.2 输出 int8 承载 R3），无「§0 判重设计、§3 照抄源方案」的脱节；§0.6 末尾「优化意图承接总表」9 项逐条对应 §0.4，无静默丢弃。
- **golden 独立性 — pass**：§8.1 `golden_gt` 以 §0.1 语义为依据（`torch.gt` 广播直算），显式声明不复刻 NPU 行/内层索引、vcmp/vcast 链、uint8 修正、persistent、同形物化；与源仓 `test_gt.py` 的 `ref = torch.gt(a, b)` + `_bool_compare`（精确比对）一致。

**结论：源算子解读与源码逐条一致，硬件耦合性判定合理，NPU 重设计可行且语义保持；golden 独立。**

### 1. API 可行性 — pass

§3.2/§3.3 列出的 API 均在 `docs/Tilelang.language/` 或 `examples/` 找到佐证（逐文件实查存在）：

| API | 佐证 | 状态 |
|-----|------|------|
| `T.vcmp`（`cmp_mod="gt"`） | `docs/Tilelang.language/比较操作/T.vcmp.md` §2.1（可选值含 `"gt"`）/§2.2.1（dtype 含 int16/32/64/fp16/fp32/bf16，**不含 int8/uint8**）/§2.2.2（A/B 必须同形）/§2.4（vcmp+vselect 示例） | ✅（与 CG-2026-0021/0022 一致） |
| `T.vcast` | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1：`i1→f16`、`i1→f32`、`f16→i8`、`i8→f16`、`ui8→f16`、`i16→f32` 均在 dtype 矩阵（**无 `i1→i8` 直转**，故 bool→int8 走 f16 中转为最短可用路径） | ✅ |
| `T.vbrc` | `docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2（src size-1→dst 广播）；§2.3 记 i1 尾轴 16 对齐约束 | ✅（int8 dst 实测挂死 → 先升 f16 再 vbrc，见 R4） |
| `T.vadd` | `docs/Tilelang.language/数学操作/T.vadd.md` §2.1（src1 可标量）/§2.3（dst 可原地） | ✅ |
| `T.vselect` | `docs/Tilelang.language/条件操作/T.vselect.md` §2.1（`(Cond,A,B,Out)`）/§2.2.1（fp16 √） | ✅ |
| `T.alloc_ub` / `T.copy` / `T.min` / `T.ceildiv` / `T.serial` / `T.Kernel(is_npu=True)` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`、`T.copy.md`、`比较操作/T.min.md`、`数学操作/T.ceildiv.md` + `examples/elementwise/vec_add_1d.py` | ✅ |
| `T.floordiv` / `T.floormod` | `tilelang/language/`（行 base divmod 展开链） | ✅ |

Developer 模式用自动同步 + v-prefix：与 §2 模式结论一致，且同族 `eq`/`ge` 已实现同构。**建议级（内联）**：§3.3 的 uint8 修正链 `T.vcmp(a_c, zero_f16, m, "lt")` 引用了 `zero_f16`，但 §4.3 buffer 清单未列该零操作数 buffer——因 `T.vcmp` 要求严格同形（CG-2026-0021）且不接受标量，Stage 3 必须显式物化一个 `(block_size,)` 零 buffer，建议补入 §4.3（同族 ge 检视亦有同款建议）。

### 2. 内存层级规划 — pass

- **搬运路径完整**：§4.4 给出每 task 的 GM→UB（`T.copy`）→ Vector 计算 → UB→GM 单阶段路径，**无 GM→L0 越级、无 L1/L0 参与**（纯 Vector）。
- **Buffer 层级标注**：§4.3 每个 buffer 标注 UB（`a_ub/b_ub/a_c/b_c/m/sp/cond/yf/y8/a1/b1/a1c/b1c`）。
- **UB 预算（机械跳过，人工已核）**：§4.5 逐 dtype「活跃 buffer 字节/元素 → @4096 手工合计 → ×1.7 膨胀」独立复算，全部 ≤192KB：fp16/bf16 8B→32KB→55KB ✓；fp32/int32 12B→48KB→82KB ✓；int64 20B→80KB→136KB ✓；int16 16B→64KB→109KB ✓；int8/bool 10B→40KB→68KB ✓；uint8 13B→52KB→88KB ✓。block_size 上限公式 `floor(192KB/(B/elem×1.7))` 复算 14456/9637/5782/7228/11565/8897 全部正确，推荐 4096 对全 dtype 达标（int64 最紧 136KB）。
- **L0C 容量**：n/a（纯 Vector，无 L0C）。
- **同形约束一致性**：§4.3 明确 `T.vcmp`「A、B 与 Cond 同形」故 `a_c`/`b_c`/`cond` 恒为 `(block_size,)`，与 §3.3/§5.3 一致。

### 3. Tiling 策略 — pass

- **Block 划分明确**：§5.2 给出 `block_size = 4096` 及 6 条理由（innermost 单块覆盖 L≤4096 使 chunks=1、32B 尾轴对齐、向量宽度整除、UB 预算、同形路径由 chunks 分块、与同族一致）。
- **对齐约束**：`block_size×{1,2,4,8}B` 均 32B 整数倍；末维向量宽度整除（fp16×8/fp32×4/int32×4），尾 lane 浪费 0% ✓。
- **非整除处理**：§6.4 尾块 `tail = T.min(block_size, L−inner_off)` 截断，来源 `vec_add_1d.py` L31–L38 已验证形态；不采用 host padding（符合 ascend-constraints.md §4）。
- **分核策略三要素（机械跳过，人工已核）**：按 `.agents/skills/_shared/standards/core-split-strategy.md` §1 逐要素核对——
  ① **逻辑核数**：`num_logical = R × ceildiv(L, block_size)`，§5.5 表按 workload 独立复算 = {smoke 1, hidden-state 2048, cnn-feat 4096, row-broadcast 4096}，全部命中（cnn-feat coalesced `(16,256,3136)` ⇒ R=4096,L=3136,chunks=1；row `(4096,4096)` ⇒ 4096）。
  ② **物理核数依据（实查）**：记录查询代码 `NPUUtils.get().get_aicore_num()` 与返回值 **20** ⇒ 纯 Vector 翻倍 **40**（标准 §1 要素② 需翻倍），且明示与 `CONST-aicore-910B2C`=24(48) 不同源、以本机实查为准（与 CG-2026-0021 环境注记一致）。
  ③ **规模判定与分核方案**：smoke 判「无需适配」（逻辑核 ≤40，附依据）；其余三个 workload 判「极大规模」，固定 `num_kernels = min(num_logical, 40)`，核内 `for i in T.serial(num_local)`（`num_local = ceildiv(num_logical, num_kernels)` 全编译期常量 → **静态边界**），grid-stride `task = i*num_kernels + cid` + `if task < num_logical` 屏蔽越界，负载不均 ≤1 task/核（≤2%），并给出失败信号供 Stage 3/4 对照。**三要素齐全、核数与 block 取值自洽、核内串行边界静态。**

### 4. 技术约束检测 — pass

§3.5.1 逐条核对本项目已知限制：三维 Kernel（No，一维）、GPU 专属 API（Yes→v-prefix 取代）、GEMM 非整除（No）、L0C 溢出（No）、物理核数限制（Yes→§5.5）、host 禁改输入（Yes→修正在 kernel 内、展平在 Op 层）、uint8 张量不可用（Yes→R3/R5）、**bool GM 输出不可用**（Yes→R3，CG-2026-0022）、**`T.vcmp` 无自动广播**（Yes→R1）+ 不支持 int8/uint8（Yes→R4）+ int16 标量化（Yes→R4 升 fp32）+ `>` 符号相关（Yes→R5）+ 严格序相等恒 False（Yes→vcmp 直取）。全部涉及项均有处置方案，无遗漏。

### 5. 循环与同步 — pass

- **循环结构**：§6.1 明确 task 级（核间 `T.Kernel` persistent 并行 / 核内 `T.serial` 串行）+ 元素级向量化（`T.vcmp`/`T.vcast`/`T.vbrc`/修正链），**无任何逐元素标量循环**，与 §1.6.2 结论一致。
- **流水线**：§6.3 明确不使用 `T.Pipelined`（单阶段算子，无 K 维累加可双缓冲），理由充分，引用 `CONST-copy-floor-method`；并披露 gt 每 task 向量 op 数（3–4；uint8 约 11）高于位运算族 → 发射项可能主导（驱动 §1.6.3 备选）。
- **同步与模式匹配**：§7 自动同步（Developer），无核间依赖、无 workspace、无手动 set_flag/wait_flag，与 §2 模式一致；CV 融合不适用（纯 Vector）。
- **尾块处理**：§6.4 已述（维度 3）。

### 6. 验证方案 — pass

- **Golden 函数**：§8.1 含 `golden_gt`（`torch.gt` 广播直算），以 §0.1 语义为依据、显式声明不复刻 NPU 算法。
- **L0 测试计划**：§8.3 给出 14 个代表性 case（dtype/shape/精度标准齐全），且**专门覆盖 gt 独有边界**：L0-9 严格序相等位必须 False（对应 `test_gt_edge_case` 的 `b[:n//4]=a[:n//4]` 注入模式）、L0-10 浮点 NaN/±0/±Inf 角点、L0-12 vcmp 同形约束回归、**L0-13 uint8 无符号序（强制 ≥128 取值，指出源 `_gen_int_inputs` 的 uint8 仅取 [0,16] 会「通过测试但语义错误」）**、L0-14 uint8 广播+修正组合。独立核对源测试确认：uint8 数据确为 [0,16]（`_gen_int_inputs` lo,hi=0,16），故设计对 L0-13 的补强是必要的。
- **精度标准**：比较结果 bool 域 exact（无舍入），按 dtype 分列，来源 `test_gt.py` 的 `_bool_compare`（`torch.equal` + dtype 断言）与 manifest `outputs.output.dtype: "bool"`。

### 7. 完整性与一致性 — pass

- **章节齐全**：模板 11 章节齐全，迁移类 §0 且 §0.1–§0.7 小节齐全。
- **无占位符**：全文无占位符标记（未填充模板变量、未完成标记；仅数学公式的 LaTeX `\{0,1\}` 等正常符号）。
- **同类引用真实**：§3.5.3/§0.7 引用的 `examples/ge/…`、`examples/eq/…`、`examples/bitwise_and/…`、`examples/elementwise/vec_add_1d.py`、`example_elementwise_exp2.py`、`example_elementwise_log2.py`、`vec_add_auto_brc.py`、`example_elementwise_add.py`、`examples/flash_qla/tilelang_get_warmup_chunks.py`、`examples/TileOPs/tileops/…/gt/*`、`test_gt.py`、`manifest/elementwise.yaml#GtFwdOp`（L389）均实查存在。
- **内部一致**：§3.2 API 映射 / §4 内存规划 / §5 Tiling / §6 循环四处描述相互一致（buffer 形状、比较域、承载 dtype、分核数）。
- **建议级（内联）**：① §0.2 引「提取件 L76–L80」指向广播路径 `y` 声明，实际在 L79–L83（行号轻微偏移，不影响结论）；② §0.5/§9.1 R4 将 `TRAP-vbrc-int8` 标注为「pattern-library traps」，实查该 trap 记录在 `examples/bitwise_and/…/RETROSPECTIVE.md`（`TRAP-vbrc-int8-vector-core-timeout`）而非 pattern-library，建议修正出处标注（§3.5.3/§0.7 已同时引用该 RETROSPECTIVE）。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性**：含 §1.6.0 调研、§1.6.1 数学等价优化、§1.6.2 向量化替代分析、§1.6.3 向量化轴与布局决策四子节齐全。
- **算法调研存在性与完整性**：§1.6.0 四问齐全——**R1** 候选表含**基线（候选 0）**，7 个候选（补集/合取/交换/算术/位/min-max/升域）逐条初判；**R2** 在线算法有明确结论「无在线变体需求（天然单遍）」+ 结构依据（逐元素映射无跨元素依赖）；**R3** 复杂度表覆盖**四口径**（FLOPs / 访存量 Bytes / 扫描遍数 / 中间缓冲峰值）且基线在表、口径写清；**R4** 硬件亲和逐候选评估（单元匹配/容量/对齐/静态边界/流水/跨核）。调研结论含选定算法与关键依据，「无更优替代」写入 7 项调研范围。深度与单步逐元素类匹配（四问各一表/段）。
- **调研结论正确性（独立复核）**：
  - ① **等价性**：选定算法（基线比较语义 + 行/内层重设计）与原语义等价由 `verify_equiv.py` E2（索引映射 10 组形状 mismatch/oob=0）+ E1/E3/E4/E5（承载/升域/修正）机器验证，我**重跑确认与内嵌表逐项一致、全 EQUIV_PASS**。独立推演：`>` 本体不可再化简；补集 `¬(a<=b)` 在 NaN 下 `¬(NaN<=x)=True` 而 `NaN>x=False`（E-NAN 4/6）→ 确不成立；合取 `(a>=b)∧(a!=b)` 对 NaN（左侧 False）、相等（右侧 False）、`a>b`（双真）三类逐点与 gt 一致 → 成立（E-NAN 0）。**gt 的回退式确为合取**（区别 ge 的析取），方向正确。
  - ② **复杂度算术复算**：R3 表基线/重设计/候选 1/2/候选 7-2D 的 FLOPs 与访存口径自洽——基线含逐元素 O(ndim) divmod（非比较但计入）、重设计含 2N bool→int8 vcast + 升域 + R·O(ndim) 行级 divmod + uint8 +3 op/operand；访存量 `(A+B)·E + N`、输出恒 1 字节、扫描遍数 1、中间缓冲 O(block_size)（无漏计 GM 往返——设计明确「无中间 GM 往返，每 task 一次进出」）。1.6.0 D-2 roofline 数字复算自洽（hidden-state 75.5MB/71.9µs、cnn-feat 分类 206µs、row-broadcast 155µs 与发射项公式一致）。**唯一小瑕（非阻塞）**：R3 引言「fp32 下 < 0.04 op/B」与复算值（A=B 时 1/(2·4+1)≈0.11 op/B）不符，量级差约 3×；但该句为定性说明（访存主导结论不受影响），四口径表本身正确。
  - ③ **亲和性判断**：与 ascend-constraints / pattern-library 实测一致——`T.vcmp` int16 标量化（PL-1.11「vcmp int16 全形态标量化」）、vcmp 无广播（CG-2026-0021）、bool GM 挂死（CG-2026-0022）、uint8 不可用（CG-2026-0018）、vbrc int8 挂死（TRAP-vbrc-int8）、vsadd/vsub 列广播税（PL-1.15，被 §9.1 R8 引用）。负向淘汰（int16 原生 vcmp）有 PL-1.11 佐证。
  - ④ **负向断言复核**：对照 `algorithm-candidates.md` §5 命中行 `ALG-elementwise`（R1 CSE/除法乘倒数/cast 链合并；R2 天然单遍；R3 融合消中间 GM；R4 单遍 Vector/dtype 路径）逐条复核——设计逐项评估并解释（CSE 无公共子式；无除法；cast 链无可合并冗余——`i1→i8` 直转不在 `T.vcast.md` §2.2.1 矩阵，2 步为最短；R2/R3/R4 已承接）。**未发现被漏掉的已知 elementwise 候选**；gt 属单步逐元素类，不涉及 Welford/online softmax 等族。kb_search 命中项（CG-2026-0021/0022、PL-1.11、TRAP-vbrc-int8）与设计结论一致。
  - ⑤ **并列候选裁决**：int16 原生 vcmp vs 升 fp32 存在性能不确定（PL-1.11 已实测标量化 → 明确淘汰）；广播类主选 vs 2-D tile 的判定裕度依赖未实证常数 → 已给 §1.6.3 实验裁决计划（见下），未纸面拍板。
- **调研→设计一致性**：§1.6.0 选定算法实际驱动 §1.4 算法描述（单遍行向量）、§3.1 公式拆解、§6 循环结构；§0.5/§0.6 处置与调研选族一致（同为「基线比较语义 + 重设计」）；源优化手段 1–9 的意图承接有逐项说明（§0.6 末尾总表），**无静默丢弃**。
- **数学等价优化完整性**：§1.6.1 对选定算法做 6 类系统审视（恒等变形/稳定化/CSE/算子降代换/访存削减/归约），结论「无公式级优化空间」有依据；候选 1/2/4/5/6 否决理由（op 数 ≥2 或浮点不成立）独立成立；R1/R3/R4/R5 作为「语义保持的等价替换」单列，不混入公式级采纳项，处理正确。
- **等价性论证正确性**：逐项独立推演——升域保序（int8/uint8 在 fp16、int16 在 fp32 精确且序同构）、`u = s + 256·(s<0)` 值域恒等、`(a>=b)∧(a!=b)` IEEE 恒等、`a&~b ≡ a>b`（0/1）——均成立，且与 §8.2 exact 精度标准一致。机器验证与推演互补。
- **向量化替代完整性**：§1.6.2 盘点全部循环/标量计算点（#1 比较、#2 bool→int8、#3 1 字节升域、#4 int16 升域、#5 uint8 修正、#6 标量广播物化、#7 行 base divmod、#8 尾块长度、#9 task 调度循环、#10 host 元数据），与 §3.3 伪代码、§6 循环结构交叉核对**无遗漏**；6 类逐元素点已全部向量化，4 处保留标量/循环逐项有理由（block 级索引/动态边界/tile 调度/host 元数据），§6 内无任何逐元素标量循环。
- **不可替代理由充分性**：#7（行 base divmod，仅每 task O(ndim)，`T.arange`+divmod 会引入逐元素标量物化，引用 `CONST-arange-scalar-materialize`）、#8（动态边界，采用 `vec_add_1d.py` 的 `T.min` 先例，padding 违反 host 禁改输入）、#9（persistent 调度循环，静态边界）、#10（host 元数据）——理由具体到本算子，非泛泛。
- **替代 API 可行性**：替代方案 API 均在 docs/examples 有佐证（与维度 1 同口径），优先 v-prefix。
- **向量化轴与布局决策完整性**：§1.6.3 候选矩阵 7 项，类别内必选候选齐全（广播轴×向量轴对齐 + I/O 原生布局 vs 核内布局两视角）；逐候选有量化取舍（整除性/尾 lane/累加链/repack/UB）；弃选给量化理由（#2 vcmp 无广播实不可实现（CG-2026-0021）、#3 host 物化负收益、#5 标量无 128bit lane、#6 浮点不成立、#7 GM 往返负收益），**无「未考虑」式论证，未把 I/O 契约当内部布局依据**（明确「I/O layout 是契约、核内布局是设计变量」）。轴质量评分（主选 #1）给出整除性/尾 lane 0%/累加链/repack/UB。源码轴对照（迁移要求）已完成（GPU `threads×npt` 为硬件组织输入，独立评估）。
- **弃选论证前提核对（亲自查证）**：弃选涉及的 API 前提均与实际一致——`T.vcmp.md` §2.2.2「A 和 B 必须具有相同的 shape」印证 #2 不可实现；`T.vbrc.md` 印证 (1,)→(block_size,) 合法与 i1 对齐约束；`T.vcast.md` 印证 `i1→i8` 直转不存在（故 2 步 cast 最短）；pattern-library 实测条目（PL-1.11/CG-2026-0021/0022/TRAP-vbrc-int8）与弃选理由无矛盾。
- **实验裁决模式完整性**：§1.6.3 判定裕度依赖未实证常数（向量发射摊销律、2-D vbrc/vcast 每 op 开销、b 行重读 L2 吸收率）→ 给足三件套：**主选** #1（per-row 1-D tile）、**备选** #4（2-D `[R_blk,L]` tile，结构完整：buffer 形状/UB 预算（fp32 R_blk=2,L=4096 ≈163KB <192KB）/循环结构/vbrc 链/dtype 路径/分核三要素按粒度重算）、**裁决计划**（代表 shape row-broadcast/cnn-feat、指标 msprof Task Duration + vec_ratio + mte2_active_bw、判定阈值「备选 < 主选×0.97 才翻转」、按 shape 分派、回写路径）。shape 特化工厂按 shape 分派两方案可行性已评估。达 Stage 4 可直接实现深度。
- **布局决策与实现一致 + 与下游章节一致**：§1.6.3 选定布局/轴与 §3.3 伪代码、§4.3/§4.5 内存规划、§6 循环结构三方一致（buffer 形状 `(block_size,)`、向量化轴 = coalesced 末维、累加循环内层向量维对应）；§3.1 以优化后公式（比较本体不变 + R1/R4/R5/R3 等价替换）为输入；§6 未出现 §1.6.2 未论证的标量循环。

**结论：算法调研四问齐全且经独立复核成立，调研结论驱动 §1.4/§3.1/§6；公式级无优化空间结论有依据；全部循环/标量点覆盖、向量化完整；轴与布局候选矩阵完整、弃选量化、实验裁决三件套齐备；机械验证与独立推演一致。**

---

## 检视总结

- 9 个检视维度全部 pass，**0 个阻塞级问题**。
- 迁移任务维度 0 的 7 个检查项（语义/算法/优化/耦合性/重设计/一致性/golden 独立性）均经源码逐条核对通过。
- 维度 8 的调研四问、等价性、复杂度、向量化覆盖、布局决策均经独立复核；`verify_equiv.py` 重跑 `OVERALL: EQUIV_PASS` 且与 §1.6.1 内嵌表一致。
- 3 个建议级观察（zero_f16 buffer 未入 §4.3 清单 / `TRAP-vbrc-int8` 出处标注 / §0.2 与 R3 引言的轻微行号与量级偏差）**均不构成阻塞**，已在对应维度内联记录，供 Stage 3 实现与后续修订参考。
- **结论：通过。**
