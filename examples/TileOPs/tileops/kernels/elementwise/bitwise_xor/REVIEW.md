# bitwise_xor_fwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/bitwise_xor/bitwise_xor_fwd_kernel/DESIGN.md
- 任务类型: 迁移（`scenario=migration`，`migration_mode=harness`，GPU 仓 TileOPs 同构工程）
- 源算子: `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（GPU kernel 工厂/类）与 `/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/bitwise.py`（op 类）；Stage 0 冻结件 `examples/TileOPs/tileops/kernels/elementwise/bitwise_xor/_bitwise_xor_fwd_kernels.py`、NPU wrapper `examples/TileOPs/tileops/kernels/elementwise/bitwise_xor/bitwise_xor.py`、NPU Op `examples/TileOPs/tileops/ops/elementwise/bitwise_xor.py`
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 9

## 机械复核（D-1/D-5）

- **verify_equiv 重跑（D-1，强制）**: `python3 examples/bitwise_xor/bitwise_xor_fwd_kernel/verify_equiv.py` → **`OVERALL: EQUIV_PASS`**，执行输出与 §1.6.1 内嵌表**逐行一致**：
  - E1（int8 vs uint8 存储位级等价，1M 随机字节 + 全 256 角点）: `E1-XOR / E1-OR / E1-NOT violations=0`，与 §1.6.1 表一致（§0.6 R3 int8 承载的位级恒等成立）。
  - E2（NPU 行/内层索引映射 vs 源 `_compute_broadcast_offsets` divmod 映射，numpy CPU 逐点对照）: 11 组形状（4 个 manifest workload + bias-add / row-broadcast-3d / scalar-broadcast / bias-3d + a 侧广播 3 组）全部 `a_mismatch=0, b_mismatch=0, oob=0`，与 §1.6.1 内嵌表逐行一致。
  - 结论: 全 `EQUIV_PASS`（§1.6.1 无公式级采纳项；E1/E2 为该设计两项语义保持等价替换——存储表示 int8/uint8、索引表示 §0.6 R1 重映射——的机器验证）。
- **design_calc_check 重跑（D-5，强制）**: `python3 .agents/tools/design_calc_check.py --design examples/bitwise_xor/bitwise_xor_fwd_kernel/DESIGN.md` → JSON `checks` 摘要：
  - `ub_budget`: **skip**（"no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"）→ **人工已核**：§4.5 手工表 3 buffer × 4096 × {1,2,4,8}B = 12/24/48/96 KB，×1.7 = 20.4/40.8/81.6/163.2 KB 均 < 192 KB ✓；`a1_ub`/`b1_ub` 各 1 元素可忽略。详见维度 2。
  - `l0c_budget`: **skip**（"no affirmative Cube-path API ... in this design"）→ 人工已核：纯 Vector，无 L0C，不适用。
  - `core_split`: **skip**（"insufficient parsed dims {}; declared logical cores = None"）→ **人工已核**：§5.5 逐 workload 复算 num_logical = R×ceil(L/block_size) 与设计表一致（smoke-1k=1 / hidden-state=2048 / cnn-feat=4096 / row-broadcast=4096）；物理核数 `NPUUtils.get().get_aicore_num()` 实查记录齐全。详见维度 3。
  - `r3_metrics`: **pass**（四口径 FLOPs / access bytes / scan passes / buffer peak 齐全）。
- **补充独立复核（本检视新增，非机械脚本）**: 针对 DESIGN.md §0.6 R4/§3.3 依赖的「int8 直连 `T.vbrc` 不经 float16 transit」结论——同族 `bitwise_and` 有 Stage 3 trap `TRAP-vbrc-int8-vector-core-timeout`（int8 dst vbrc 运行期挂死），本检视**亲自在本工具链复现**：
  - `python3 examples/bitwise_and/bitwise_and_fwd_kernel/repro/TRAP-vbrc-int8-vector-core-timeout.py`（bypass）→ `mode=bypass equal=True`（环境可用）；`--trap`（`vand` + 直连 int8 vbrc）→ **`timeout` 退出码 124（挂死，无输出）**，复现 trap。
  - 复刻 DESIGN.md §3.3 的 int8 路径（`T.vbrc(int8 (1,) → int8 (block_size,))` 后接 **`T.vxor`**，`/tmp/opencode/probe_vbrc_int8_vxor.py`）→ **`direct int8 vbrc -> vxor equal= True`（EXIT=0）**；同结构换成 `T.vand` → **EXIT=124 挂死**。
  - 结论: **int8-dst vbrc 挂死是 `vand`（及 `or`）特有的，`T.vxor` 直连 int8 vbrc 实测 bit-exact 通过** —— 设计 §0.6 R4「直连探针 int8 通过」的论断经本检视独立复现成立（见维度 8）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**检视方式**：亲自 Read 源算子代码全文——`/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（`_BITWISE_DTYPES` L140–147、`_strategy_npt` L181–195、`coalesce_broadcast_dims` L369–432、`_compute_broadcast_offsets` L435–451、`_is_contiguous_same_shape` L454–460、`_make_binary_register_copy` L463–499、`_make_binary_explicit` L555–609、`BinaryKernel` L890–1011、`_Uint8StorageBinaryKernel` L1329–1333、`BitwiseXorFwdKernel`/`BitwiseXorBoolStorageFwdKernel` L1905–1920）、`/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/bitwise.py`（`_BoolStorageBitwiseBinaryOp` L18–52）、Stage 0 冻结件 `_bitwise_xor_fwd_kernels.py`（全文 185 行）、NPU wrapper `bitwise_xor.py`（全文）、NPU Op `bitwise_xor.py`（全文 L120–133 `_eager_forward`），逐项核对 §0。

- **语义理解正确性 — pass**：§0.1 数学语义 `y = a ^ b`、无规约/无跨元素依赖（bit-exact，无浮点累加顺序）与源 `BitwiseXorFwdKernel.op_func = a ^ b`（elementwise.py L1910–1912）、`BitwiseXorBoolStorageFwdKernel.op_func = T.bitwise_xor(a,b)`（L1918–1920）一致；`_BITWISE_DTYPES = (bool,uint8,int8,int16,int32,int64)`（L140–147）与 §0.1 完全一致；bool 特殊路径 `view(torch.uint8)` + `*BoolStorageFwdKernel` 与 `_BoolStorageBitwiseBinaryOp._eager_forward`（`ops/elementwise/bitwise.py` L41–52）/ NPU Op `_eager_forward`（`examples/.../ops/elementwise/bitwise_xor.py` L120–133）一致。§0.2 I/O（`y: T.Tensor((N_total,), out_dtype)`、`out_idx=[2]`、一维展平输出）与冻结件 L52–56 / L75–78 一致（设计引 L55/L78 精确）；输出广播形状还原与 NPU Op `.reshape(self.out_shape)`（L131–133）一致。
- **算法解读完整性 — pass**：§0.3 步骤表覆盖冻结件全部计算语句——host `coalesce_broadcast_dims`（L94–157，wrapper 调用见 `bitwise_xor.py` L194）、同形连续快路径 `y[idx]=a[idx]^b[idx]`（L57–60）、广播路径 `y[flat]=a[a_off]^b[b_off]`（L80–86）、编译期 unrolled divmod 链（L160–176）——无遗漏、无臆造；数据流图覆盖 a/b/y；host 侧 `lru_cache` 特化（L38）已列出。设计引用的 GPU 上下文行号（`_make_binary_explicit`@L556、`BinaryKernel`@L890、`BitwiseXorFwdKernel`@L1905）经核对全部准确。
- **优化手段覆盖完整性 — pass**：§0.4 8 项逐项对应源码（`T.Parallel(threads,npt)` L58/L81；连续段 coalesced；`coalesce_broadcast_dims`；unrolled divmod L160–176；同形快路径 L45；`register_copy` fragment L463–499；工厂+lru_cache L38/L555；bool→uint8 存储）。抽检 binary 路径无 SMEM tiling / warp shuffle / 异步流水 / swizzle，§0.4 "对照清单无遗漏"成立。
- **耦合性判定合理性 — pass**：§0.5 四态处置（保留 7 / 等价替换 3 / 重新设计 4 / 舍弃 1）逐项附依据；"逐元素 divmod 广播索引 → 重新设计"（NPU 无逐元素整数 divmod 快路径 + 源形态为 GM 标量 gather）判定有据；"`T.Parallel` 逐线程标量 `^` / `T.vxor` 无自动广播 → 重新设计"（CG-2026-0020）与探针 B/C/Q1 一致；`register_copy` 寄存器→UB 等价替换、`threads=`/npt kwarg 舍弃、一维 grid 保留均合理。舍弃/重新设计项均无静默丢弃（§0.6 末尾有承接总表）。
- **NPU 重设计可行性 — pass**：R1–R4 均具「源方案 → NPU 新算法 → 语义保持论证」三要素。R1 行/内层分解（`row=flat//L, inner=flat%L` 的 mixed-radix 分解）与源 divmod 链逐点相同（E2 机器验证 11/11 mismatch=0）；"末维必有 `a_strides[-1]==1` 或 `b_strides[-1]==1`"经本检视**独立推演 `coalesce_broadcast_dims` 的 stride 合成规则**成立——合并时记录的是**最新并入维**的 raw stride（`groups[-1]=(…, a_raw[i], b_raw[i])`，注意是 `a_raw[i]` 而非组首维），故末组记录值 = `a_raw[ndim-1]`，而 `_make_strides` 末维 stride 恒为 1、仅当 `padded==1 且 out>1` 时零化 → `a_strides[-1]∈{0,1}`；`(0,0)` 仅在 `L==1`（全标量，coalesced 兜底 `(1,0,0)`）出现，此时两路 vbrc 合法（block_size>1）。R2 persistent 一维 Kernel + Vector 向量化满足一维 Kernel 约束（`NPUUtils.get().get_aicore_num()*2` 实查 20⇒40，本检视复现返回值 20）。R3 int8 承载位级恒等经 E1 验证。R4 显式同形物化经我独立探针复现（int8 直连 vbrc→vxor PASS，见机械复核）。
- **设计与决策一致性 — pass**：§1–§7 描述的是 R1–R4 决策后的 NPU 算法（行/内层向量化 + persistent 分核 + int8 承载 + 显式同形物化），无"§0 判重设计、§3 照抄源方案"的脱节；§3.3 伪代码 buffer 形状、§4.3/§4.5、§6 循环结构三方一致。
- **golden 独立性 — pass**：§8.1 golden = `torch.bitwise_xor(a,b)` 广播直算，以 §0.1 源算子语义为唯一依据，明确**不复刻** §0.6 的 NPU 行/内层索引、vbrc、persistent 结构。

### 1. API 可行性 — pass

- **API 存在性 — pass**：`T.vxor`（`docs/Tilelang.language/逻辑操作/T.vxor.md`，§2.1/§2.2/§2.4）+ 探针 A 实测 int8/16/32/64 同形 bit-correct；`T.vbrc`（`docs/Tilelang.language/shape操作/T.vbrc.md` §2.2.2 文档合法 src (1,)→dst (N,)）+ 探针 D int32 + 本检视独立复现 int8 vbrc→vxor PASS；`T.copy`/`T.alloc_ub`/`T.min`/`T.ceildiv` 文档均存在（`docs/Tilelang.language/内存操作/T.copy.md`、`内存操作/T.alloc_ub.md`、`比较操作/T.min.md`、`数学操作/T.ceildiv.md`，本检视实查）；`T.serial`+`task=i·K+cid`+`if` 有 `T.vcast.md` §2.4 persistent 分发先例；`T.floordiv`/`T.floormod` 存在于 `tilelang/language/tir/op.py` L2973/L2995（本检视实查）；`examples/elementwise/vec_add_1d.py` L27–38 提供完整 Developer 先例。设计对 `T.vxor` 的两条负向断言（无自动广播 / 不接受标量）均有探针实测（B/C/Q1/E/Q2，脚本已留存 `history_version/`）+ CG-2026-0020/0019 登记 + `T.vxor.md` §2.1/§2.2.2 条款对照。
- **API 与模式匹配 — pass**：Developer 模式全程自动同步（§7.2），无手动 `sync_block_set/wait`；§6 无逐元素 `T.Parallel` 标量循环。
- **v-prefix 优先 — pass**：主用 `T.vxor`/`T.vbrc`（v-prefix），无 npuir_xxx 回退。

### 2. 内存层级规划 — pass

- **搬运路径完整 — pass**：§4.4 给出 GM → UB → GM 路径（纯 Vector），无 GM→L0 越级路径、无 L1/L0 参与。
- **UB 预算 — pass（人工复核 design_calc_check skip 项）**: §4.5 手工表 int8/int16/int32/int64 = 3×4096×{1,2,4,8}B = ~12/24/48/96 KB，×1.7 = 20.4/40.8/81.6/163.2 KB 均 < 192KB ✓。`a1_ub`/`b1_ub` 各 1 元素可忽略。与同族 bitwise_or/fwd 复算口径一致。
- **L0C 容量 — pass/n.a.**：纯 Vector 不使用 L0C。
- **Buffer 层级标注 — pass**：§4.3 每个 buffer 标注 UB，并注明 `T.vxor` 三 buffer 必须恒同形（CG-2026-0020）。
- **建议级（§4.5 "block_size 上限"行算术偏乐观约 2%）**：设计给出 int64 ≤ 4915 / int32 ≤ 9830 / int8 ≤ 39320，标注"1.7x 膨胀口径"。严格上界 = 196608/(3×E×1.7)：int64 ≤ 4818、int32 ≤ 9637、int8 ≤ 38550——设计三个值等效因子 ≈1.667（5/3），统一偏高约 2%（同族 bitwise_and 曾出现、bitwise_or 已修正的同一偏差，本设计复现）。推荐值 block_size=4096 远低于严格上界，不影响本设计正确性与 Stage 3；仅当 Stage 4 按该行上界扫 block_size（如 int64 取 4915）会触发 UB 溢出。建议按真实 ×1.7 重算为 4818/9637/38550（或改注因子 5/3）。不阻塞。

### 3. Tiling 策略 — pass

- **Block 划分明确 — pass**：§5.2 给出 block_size=4096 及 5 条理由（innermost 单块覆盖 L≤4096、32B 尾轴对齐、向量宽度整除、UB 预算、同形分块）。
- **对齐约束 — pass**：block_size×E = 4K/8K/16K/32K（32B 整数倍）；向量宽度整除（int8×16 / int16×8 / int32×4 / int64×2）。
- **非整除处理 — pass**：§6.4 `tail = T.min(block_size, L − inner_off)` 尾块截断 + 零起点 dst 切片，来源 `vec_add_1d.py` L31–38（官方已验证形态）。
- **分核策略三要素（`_shared/standards/core-split-strategy.md` §1 核对）— pass**：
  - ① 逻辑核数计算：§5.5 `num_logical = R × ceildiv(L, block_size)`，本检视独立复算 4 个 workload（smoke-1k=1 / hidden-state=2048 / cnn-feat=4096 / row-broadcast=4096）与设计表逐值一致。**关键复算**：cnn-feat `coalesce_broadcast_dims([16,256,56,56],[256,1,1])` = `((16,256,3136),(802816,3136,1),(0,1,0))` ⇒ k=3,R=4096,L=3136,chunks=1 ⇒ 4096 ✓；row-broadcast `((4096,4096),(4096,1),(0,1))` ⇒ R=4096,L=4096 ⇒ 4096 ✓。
  - ② 物理核数依据：§5.5 记录 `NPUUtils.get().get_aicore_num()` 实查代码与返回值 **20** ⇒ 纯 Vector 40 核（本检视独立执行实查返回 20 ✓），并注明与 `CONST-aicore-910B2C`（24⇒48）不同源、按本机值。
  - ③ 规模判定与分核方案：smoke-1k `num_logical=1 ≤ 40` → "无需适配"；其余极大规模 → 固定启动 `num_kernels = min(num_logical, 40)` + 核内 `T.serial(num_local)`（`num_local=ceildiv(num_logical,num_kernels)`，静态编译期常量）+ grid-stride 映射 + `if task < num_logical` 屏蔽——三要素齐全，边界静态。
- **建议级（§5.2 推荐 block_size=4096 vs wrapper 默认 2048）**：§5.2 的 cnn-feat「chunks=1 / 发射项减半」收益与 §1.6.0 估算均基于 block_size=4096，而 wrapper 整数路径 `_DEFAULT_BLOCK_SIZE=2048`（K11）。若 Stage 3 沿用默认 2048，cnn-feat L=3136 退化为 chunks=2（仍正确，仅发射项翻倍）。设计已注明"4096 为设计推荐值，Stage 4 扫"，建议 Stage 3 显式以 4096 建 kernel 以对齐设计估算口径。不阻塞。

### 4. 技术约束检测 — pass

- **三维 Kernel — pass**：源即一维 grid（冻结件 L57/L80 `T.Kernel(ceildiv(N,block_size))`）；NPU 用一维 `T.Kernel(num_kernels, is_npu=True)`，不涉及三维 Kernel/block_metadata。
- **GEMM 非整除 — pass/n.a.**：无 GEMM；非整除由 §6.4 尾块处理。
- **L0C 溢出 — pass/n.a.**：无 Cube。
- §3.5.1 已知限制检查覆盖：GPU 专用 API（`threads=`/`T.Parallel`）舍弃、物理核数分核、host 侧禁改输入、uint8 不可用、`T.vxor` 无自动广播、`T.vxor` 不接受标量——逐项有处置。同族已知 trap（`TRAP-vbrc-int8-vector-core-timeout`：`vand`/`vor` int8 dst vbrc 挂死）经本检视复现确认为 **vand/or 特有**，不作用于本算子 `T.vxor`（见机械复核），故不构成本设计的约束缺口（建议在 §9 备注该差异，见维度 7）。

### 5. 循环与同步 — pass

- **循环结构 — pass**：§6.1/§6.2 明确：核间 persistent `T.Kernel`、核内 `T.serial(num_local)`（静态边界）、行内块级并入 `chunks`（无嵌套循环）、元素级 `T.vxor`（整 buffer 一条向量指令）、标量广播 `T.vbrc`；无逐元素 `T.Parallel` 标量循环（与 §1.6.2 一致）。
- **同步与模式匹配 — pass**：§7 Developer 自动同步，无手动同步点，依据（每 task 独立、无核间依赖、copy→compute→copy 顺序依赖由编译器保证）成立。
- **尾块处理 — pass**：§6.4 `tail=T.min(...)` 截断 + `[0:tail]` 零起点切片，UB 陈旧段不落 GM，先例明确。

### 6. 验证方案 — warn（建议级，不阻塞）

- **Golden 函数 — pass**：§8.1 含 PyTorch 参考实现草案（`torch.bitwise_xor`），以 §0.1 语义为唯一依据（§0.1 明确不复刻 NPU 算法）。
- **L0 测试计划 — pass（覆盖面广，含 10 case）**：同形各 dtype / 大规模 / 广播三模式 / a 侧广播 / 标量广播 / 尾块非整除 / bool 存储 / persistent 边界 / 二补码极值 / vxor 同形约束回归。dtype、shape、负值与极值覆盖到位。
- **精度标准 — pass**：§8.2 逐位运算 bit-exact（`torch.equal`/`exact_compare`），整数/布尔域无舍入，标准正确。
- **建议级（L0 计划缺口 1：plain uint8 直接输入未单列）**：manifest 冻结 smoke 含 `pytest.param(4_096, torch.uint8)`（`examples/TileOPs/tests/ops/test_bitwise_xor.py` L54），即存在 **plain uint8 直接输入**用例。同族 bitwise_or 复盘明确指出：uint8 缺口须按 manifest dtype 全集推广到 plain uint8（其 L0-8 专项覆盖）；本设计 §0.6 R3 与 §8.3 仅显式覆盖 bool 存储路径，未单列 plain uint8。虽 wrapper `_kernel_dtype_str`/`_as_kernel_input` 已把 uint8 映射 int8，Stage 3 大概率可过，但缺显式 L0 用例保障。建议补 L0（plain uint8 直接输入，如 `[1024]×[1024]`）。
- **建议级（L0 计划缺口 2：int8 承载 + 末维 stride=0 广播物化未单列）**：本设计广播路径末维 stride=0 的 int8 物化（bool/uint8/int8 承载）依赖直连 int8 vbrc——同族先例（and/or）对该路径一律走 f16 transit。设计 §8.3 L0-7 bool 用例（`[2048,4096]×[2048,4096]`、`[2,512,768]×[1,1,768]`）与 L0-5（int32）均**不触发**末维 stride=0 的 1 字节 vbrc（前者同形、后者末维 b 连续 stride=1）。建议补一条强制走 int8 承载 + 末维 stride=0 的广播用例（如 bool/uint8 `[16,256,56,56]×[256,1,1]` 或 `[64,128]×[1,1]`），对齐 bitwise_or L0-9 的定位，以覆盖并回归该关键路径。不阻塞。

### 7. 完整性与一致性 — warn（建议级，不阻塞）

- **章节齐全 — pass**：§0（含 0.1–0.7）+ §1–§11 齐全。
- **无占位符 — pass**：本检视实查全文无占位符标记（无未完成标记、无大写占位记号、无花括号占位变量）。
- **同类引用真实 — pass**：§3.5.3 与 §0.7 引用的路径逐一实查存在——`examples/elementwise/vec_add_1d.py`、`examples/bitwise_and/bitwise_and_fwd_kernel/DESIGN.md`、`testing/npuir/bit_ops/test_binary_bit_op_dev.py`、`examples/elementwise/vec_add_auto_brc.py`、`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/`、`examples/TileOPs/tileops/manifest/elementwise.yaml`（含 `BitwiseXorFwdOp` @L197）、`examples/TileOPs/tests/ops/test_bitwise_xor.py`、`examples/TileOPs/benchmarks/ops/bench_bitwise_xor.py` 均存在。
- **内部一致 — pass（4 处表述/证据小瑕疵，均建议级）**：
  - 建议级 a：**§4.1 输入 dtype 表遗漏 plain uint8**（仅列 `int8/int16/int32/int64（+ bool 经 int8 存储承载）`），而 §0.1/§0.5/§0.2 与 manifest 均含 uint8。建议 §4.1 补 `uint8`（经 int8 承载），与 §0.5 的 dtype 契约行一致。
  - 建议级 b：**§0.6 R4 与 Stage 1 RETROSPECTIVE 引用的 `probe_vbrc_int8.py`（直连 int8 vbrc 探针）未留存**于算子目录 `history_version/`（该目录仅 `design_probe_vxor.py`、`design_probe_vxor_followup.py`）；本检视已独立复现该论断成立（见机械复核），但证据工件缺失。建议把直连 int8 vbrc 探针补存 `history_version/`，便于下游复跑与回归（工具链升级后重验）。
  - 建议级 c：**未备注同族 int8-vbrc trap 的 op 差异**。§9.1 未提及同族 `TRAP-vbrc-int8-vector-core-timeout`（`vand`/`vor` int8 dst vbrc 挂死、需 f16 transit）。本算子 `T.vxor` 直连 int8 vbrc 经本检视复现**不挂死**，但为避免 Stage 3 照搬同族 f16 transit 或反之误判，建议在 §9.1/§9.2 加一行说明"int8 vbrc trap 为 and/or 特有，vxor 直连实测通过（附证据）"。
  - 建议级 d：**§0.3 描述的并行结构为 explicit_parallel**，而 GPU `BinaryKernel` 对同形整数输入的默认策略是 `register_copy`（`elementwise.py` L1000–1003，仅在 §0.4 #6 提及）。因本迁移的冻结源为 extract 出的 `_make_binary_explicit`（§0.7），§0.3 与标杆件一致；建议 §0.3 加一句说明（冻结源 = explicit_parallel 工厂，GPU 默认策略 register_copy 见 §0.4），消除歧义。

### 8. 算法优化分析 — pass

- **§1.6 章节存在 — pass**：含 §1.6.0 算法调研 / §1.6.1 数学等价优化 / §1.6.2 向量化替代分析 / §1.6.3 向量化轴与布局决策，四子节齐全；单步逐元素类采用轻量调研（四问各一行）符合 algorithm-research.md §3 分级。
- **算法调研存在性与完整性（§1.6.0，独立核对）— pass**：
  - R1 等价化简候选表**含基线**（源算法/输入公式 `y=a^b`）；候选 1（算术恒等 `a+b−2(a&b)`，4 op）、候选 2（布尔逻辑分解 `(a|b)&~(a&b)`，3–4 op）、候选 3（位平面重排，需 repack）逐条有等价性初判与收益方向。本检视独立推演：三条恒等式在整数域均成立但 op 数 ≥3 > 基线 1，收益为负 → 淘汰正确。
  - R2 在线算法有**明确结论**（"无在线变体需求（天然单遍）"）并附结构依据（逐元素映射无 running 统计量、无跨元素依赖），与 `algorithm-candidates.md` **ALG-elementwise** R2「逐元素映射天然单遍」一致——负向断言成立。
  - R3 复杂度对比覆盖**四口径**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值），**基线在表**，口径写清（A/B/N、E=elem_bytes；含广播路径逐元素标量 gather 的低效注记）。本检视独立复算：位运算本体 1 op/元素为下界；广播索引基线 `N·O(ndim)` vs 重设计 `R·O(ndim)`（R=N/L）降 ~L 倍成立；访存 (A+B+N)·E；扫描 1 遍；峰值 O(block_size)。无中间缓冲 GM 往返漏计（§1.5 明确"无中间 GM 往返"）。
  - R4 逐候选亲和评估（单元匹配/容量/对齐/静态边界/流水/跨核）：基线逐元素 divmod+gather 在 NPU 落错单元被淘汰（附依据）；重设计对 Vector/UB/对齐/静态边界全亲和。
  - **负向断言主动复核**：`T.vxor` 无自动广播——对照 `T.vxor.md` §2.1「src0, src1(张量时) 和 dst 应具有相同的形状」+ §2.2.2「仅支持 1-5D tensor」（无自动广播条目）+ 已留存探针 B/C（FAIL）与 Q1（xor FAIL vs vor PASS）+ CG-2026-0020 → 成立；"无更优替代"写明调研范围（algorithm-candidates.md §5/ALG-elementwise 命中行 + 恒等变形自查 + 本仓 examples + pattern-library + 源结构 + kb_search），未漏已知候选。
- **调研结论正确性（独立复核）— pass**：① 等价性——选定算法（基线位运算本体 + 行/内层索引重设计）与原语义等价，E2 逐点 0 mismatch（本检视重跑一致），位运算 bit-exact 无舍入；② 复杂度算术独立复算一致（见上）；③ 亲和性判断与 ascend-constraints、pattern-library 无矛盾；④ **负向断言复核未被推翻**（vxor 无广播有 docs+源码+探针三重证据；无在线/无化简候选与 ALG-elementwise 一致，无被漏掉的已知候选）；⑤ 无并列候选纸面拍板（主选 #1 vs 备选 #4 交 §1.6.3 实验裁决）。
- **调研→设计一致性 — pass**：§1.6.0 选定算法实际驱动 §1.4（单遍逐元素行向量算法）、§3.1（以"优化后公式 = 原式 + 索引按 R1/R4"为输入）、§6（核内无逐元素标量循环）；迁移 §0.5/§0.6 处置与选族一致；源算法优化手段 1–8 的**意图承接逐项说明（§0.6 末尾总表），无静默丢弃**。
- **数学等价优化完整性 — pass**：§1.6.1 对 6 类优化（恒等变形/稳定化/CSE/算子降代/访存削减/归约）逐类核查并给"无适用项/无需求/无公共子式/已下界/无归约"结论与依据；结论"无公式级优化空间"经独立推演成立（位运算最短形态 1 op，等价位运算恒等式 op 数 ×3–4）。§0.6 R3（int8 替换 uint8）与 R1/R4（索引重映射/同形物化）明确归为**语义保持的等价替换**而非公式优化，且均机器验证，定性准确。
- **等价性论证正确性 — pass**：R1 索引映射等价由 E2 机器验证（11 形状 mismatch=0，重跑一致）；R3 位级恒等由 E1 验证；R4 同形物化正确性由探针 D + 本检视独立复现（int8 vbrc→vxor PASS）。无空话论证。
- **向量化替代覆盖完整性 — pass**：§1.6.2 盘点 6 个循环/标量点（#1 逐位异或=#vxor、#2 标量广播=vbrc、#3 行 base divmod、#4 tail=T.min、#5 task 循环=T.serial、#6 host 元数据），与 §3.3 伪代码、§6 循环结构交叉核对无遗漏（§6 核内无逐元素标量循环）。#1/#2 已向量化（docs + testing 双佐证 + 探针）；#3–#6 不可替代理由具体到本算子（block 级索引/任务映射、动态边界、persistent 调度结构、host 侧计算），非空泛；`T.arange`+divmod 方案被 `CONST-arange-scalar-materialize`（~1ns/elem）实测否决，依据成立。
- **替代 API 可行性 — pass**：`T.vxor`（探针 A int8/16/32/64 PASS + docs）、`T.vbrc`（docs §2.2.2 + 探针 D + 本检视独立复现 int8 PASS）均有佐证，优先 v-prefix。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 按逐元素/广播类枚举候选矩阵（#1 展平一维 + innermost 向量化 + 显式同形物化；#2 核内 auto-broadcast tile；#3 host 预广播；#4 2-D `[R_blk,L]` 块化 tile），逐候选给量化评分/理由；**弃选 #2 的理由经本检视核对与文档一致**（`T.vxor.md` §2.1/§2.2.2 同形约束 + CG-2026-0020 + 探针证实 vor 可广播而 vxor 不可）；弃选 #3（host 物化）引 ascend-constraints §4「host 侧禁止改动输入/禁止 Padding」（本检视实查条款存在且相符）；源码轴对照（GPU `threads×npt` 轴是硬件组织输入而非结论）已做。
- **实验裁决模式完整性 — pass**：判定依赖未实证常数（row-broadcast b 行重读的 L2 驻留，估算 188µs vs 128µs 裕度不可忽略），已给**主选 #1 + 备选 #4（结构完整：`[R_blk,L]` buffer 形状、int32 R_blk=2/L=4096 UB≈96KB×1.7≈163KB<192KB、int64 容量否决退化 R_blk=1、转置/重排路径无、分核三要素）+ 裁决计划（代表 shape、指标 msprof Task Duration/mte2_active_bw/mte2_instructions、判定阈值 备选<主选×0.97、shape 特化分派）**，达到 Stage 4 可直接实现深度；`CONST-mte2-degradation` 已查（sets=1 L2 口径 43.1 GB/s/core）并标注 stale/跨核数外推。
- **布局决策与实现一致 — pass**：选定方案 #1（`(block_size,)` UB buffer、向量化轴=coalesced 末维）与 §3.3 伪代码（buffer 形状、`T.vxor` 操作数整 buffer）、§4.5 UB 预算、§6 循环结构三方一致。
- **与下游章节一致 — pass**：§3.1 以"本体不变、索引按 §0.6 R1/R4"为输入（与 §1.6.1 优化结论同口径）；§6 出现的是 `T.vxor`/`T.vbrc` 整 buffer 向量化，无 §1.6.2 未论证的逐元素标量循环。
- **建议级（E2 验证范围注记）**：E2 的 `_npu_offsets`（`row=flat//L, inner=flat%L` + 外层 mixed-radix）与源 `_compute_broadcast_offsets`（`divisors` mixed-radix）在 coalesced 坐标下代数同构，故 E2 机器验证的是**索引映射公式等价**；`a_strides[-1]∈{0,1}`（连续段拷贝假设）由本检视解析推演成立（见维度 0），但未纳入机器断言。建议在 `verify_equiv.py` 增补"coalesced 末维 stride∈{0,1}"的断言，使该隐含前提也受机器保护。不阻塞。

> 结论为「通过」，无阻塞级问题；上述 9 条建议级备注已分别记录于维度 2/3/6/7/8，供 Stage 1 择机订正，不阻塞 Stage 3。
