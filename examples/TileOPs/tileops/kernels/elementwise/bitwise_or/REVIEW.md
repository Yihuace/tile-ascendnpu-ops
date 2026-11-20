# bitwise_or_fwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/bitwise_or/bitwise_or_fwd_kernel/DESIGN.md
- 任务类型: 迁移（`scenario=migration`，`migration_mode=harness`，GPU 仓 TileOPs 同构工程）
- 源算子: `examples/TileOPs/tileops/kernels/elementwise/bitwise_or/_bitwise_or_fwd_kernels.py`（Stage 0 提取件）；GPU 原上下文 `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（`_make_binary_explicit` @ L556、`BinaryKernel` @ L890）、`/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/bitwise.py`、`/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/_base.py`
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 2

## 机械复核（D-1/D-5）

- **verify_equiv 重跑（D-1）**: `python3 examples/bitwise_or/bitwise_or_fwd_kernel/verify_equiv.py` → `OVERALL: EQUIV_PASS`。
  - E1（int8 vs uint8 存储位级等价，1M 随机字节 + 全 256 角点字节）: OR / XOR / NOT 三项 `violations=0`，与 §1.6.1 内嵌表一致。
  - E2（NPU 行/内层索引映射 vs 源 `_compute_broadcast_offsets` divmod 映射，numpy/torch CPU 逐点对照）: 11 组形状（含全部 4 个 manifest workload + 7 组测试/角色互换广播，coalesced shape 分别为 `(1024,)/(8388608,)/(16,256,3136)/(4096,4096)/(128,128)/(128,128)/(8192,)/(1024,768)/(4096,4096)/(2,3)/(2,3,3)`）全部 `a_mismatch=0, b_mismatch=0, oob=0`，与 §1.6.1 内嵌表逐行一致（设计写"11 组形状含全部 manifest + 测试广播模式"成立）。
  - 结论: 执行输出与 §1.6.1 内嵌表一致且全 `EQUIV_PASS`。§1.6.1 无公式级采纳项（其本体不变）；E1/E2 承载该设计的两项语义保持等价替换（存储表示 uint8→int8、索引表示 R1 行/内层重映射），机器验证有效。
- **design_calc_check 重跑（D-5）**: `python3 .agents/tools/design_calc_check.py --design examples/bitwise_or/bitwise_or_fwd_kernel/DESIGN.md` → JSON `checks` 摘要:
  - `ub_budget`: **skip**（"no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"）→ **人工已核**：§4.5 逐 dtype 复算 3 buffer × 4096 × {1,2,4,8}B = 12/24/48/96 KB；×1.7 = 20.4/40.8/81.6/163.2 KB 均 < 192 KB；int8+f16 transit 分支 ≈28 KB ×1.7 ≈48 KB。**§4.5 "block_size 上限"行独立复算 196608/(1.7×每元素系数) = int64 4818 / int32 9637 / int16 19275 / int8(3buf) 38550 / int8(3buf+4B f16 transit) 16521——与设计逐值一致（已修正同族 bitwise_and 曾出现的 5/3≈1.667 近似偏差，本设计使用真实 1.7）**。详见维度 2。
  - `l0c_budget`: **skip**（"no affirmative Cube-path API ... in this design"）→ 人工已核：纯 Vector，无 L0C，不适用。
  - `core_split`: **skip**（"insufficient parsed dims {}; declared logical cores = None"）→ **人工已核**：§5.5 逐 workload 复算 `num_logical = R × ceil(L/block_size)` 与设计表一致（smoke-1k=1 / hidden-state=2048 / cnn-feat=4096 / row-broadcast=4096）；物理核数 `NPUUtils.get().get_aicore_num()` 实查记录齐全，本检视环境独立实跑返回 **20**（⇒40 Vector 核），与设计一致。详见维度 3。
  - `r3_metrics`: **pass**（四口径 FLOPs / access bytes / scan passes / buffer peak 齐全）。
- **补充机械复核（独立重跑 NPU 能力探针 D-3）**: `python3 examples/bitwise_or/bitwise_or_fwd_kernel/history_version/design_probe_vor.py`（本机 NPU 实跑）→
  - `A-int8/int16/int32/int64: PASS`（**独立确认 `T.vor` 存在且覆盖 int8/16/32/64**）；`A-uint8: FAIL (value mismatch)`、`G-uint8: FAIL (dtype changed to torch.float32)`（确认 CG-2026-0018）；`B-col/B-row: PASS`、`C-3d: PASS`（确认 rank≥2 size-1 差维广播含交错广播）；`D-vbrc: PASS`（确认 `T.vbrc (1,)→(N,)`）；`F-rank1: FAIL (TVMError)`（确认 rank-1 `(N,)|(1,)` 不自动广播）。
  - 与 §3.4/§0.6/§9 的能力结论及 headers 一致（唯一偏差见维度 0 建议级：Probe E 脚本自身 out_idx 缺陷）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**检视方式**：亲自 Read `_bitwise_or_fwd_kernels.py` 全文（185 行）、GPU `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（L140–147 `_BITWISE_DTYPES`、L540–609 `_make_binary_explicit` 及前述 `_make_binary_*` 同形广播路径）、NPU wrapper `examples/TileOPs/tileops/kernels/elementwise/bitwise_or/bitwise_or.py`（L66–73 dtype 契约、L83–127 int8 边界适配、L182–196 coalesce、L220–234 forward）、NPU Op `examples/TileOPs/tileops/ops/elementwise/bitwise_or.py`、manifest `elementwise.yaml`（`BitwiseOrFwdOp` L149–195）、冻结测试 `tests/ops/test_bitwise_or.py`、探针 `history_version/design_probe_vor.py`，并**独立重跑探针**（见机械复核），逐项核对 §0.1–§0.7。

- **语义理解正确性 — pass**：§0.1 数学语义 `y = a | b`、逐位、无规约/无跨元素依赖、bit-exact 与源码 `_make_binary_explicit`（elementwise.py L556–609）及 `op_func=a|b` 一致；dtype 集 `bool|uint8|int8|int16|int32|int64` 与源码 `_BITWISE_DTYPES`（elementwise.py L140–147；wrapper L66–73）**逐元素一致**；"bool 走 `view(torch.uint8)` + `*BoolStorageFwdKernel`" 与 wrapper 注入的 `BitwiseOrBoolStorageFwdKernel`（L237–252，`SUPPORTED_DTYPES=(torch.uint8,)`）一致。§0.2 I/O（`y: T.Tensor((N_total,), out_dtype)`、`out_idx=[2]`、一维展平输出）与提取件 L52–56 / L75–78 一致；广播形状还原（Op 层 `reshape(out_shape)`）与 manifest `shape_rules` 一致。
- **算法解读完整性 — pass**：§0.3 步骤表覆盖源码全部计算语句——host `coalesce_broadcast_dims`（源 L94–157）、同形快路径 `y[idx]=a[idx]|b[idx]`（源 L57–60）、广播路径 `y[flat]=a[a_off]|b[b_off]`（源 L80–86）、`_compute_broadcast_offsets` 编译期 unrolled mixed-radix divmod 链（源 L160–176）——无遗漏、无臆造；数据流图覆盖 a/b/y；host 侧 `lru_cache(maxsize=32)` 特化（源 L38）、`out_dtype = output_dtype or dtype` 已列出。提取件头部所述"第二操作数内联修正"（`a[...] | b` → `a[...] | b[...]`）经核对与 GPU `_make_binary_explicit` 原样一致（elementwise.py L578/L605 `op_func(a[idx], b[idx])`）。
- **优化手段覆盖完整性 — pass**：§0.4 8 项与源码逐项对应（`T.Parallel(threads,npt)` L58/L81；连续段 coalesced；`coalesce_broadcast_dims` L94；unrolled divmod L160；同形快路径 L45/L564；`register_copy` fragment；工厂+lru_cache L38；bool→uint8 存储）。抽检 binary explicit 路径无 warp shuffle / SMEM tiling / 异步流水 / swizzle，§0.4"对照 migration-analysis.md §4.1 清单无遗漏"成立。
- **耦合性判定合理性 — pass**：§0.5 四态处置（保留 7 / 等价替换 3 / 重新设计 4〔归并 R1/R2/R3〕/ 舍弃 1）逐项附依据；"逐元素 divmod 广播索引 → 重新设计"（NPU 无逐元素整数 divmod 快路径 + 源形态为 GM 标量 gather）判定有据；`register_copy` 寄存器→UB 等价替换、`threads=`/npt 舍弃（npuir `T.Kernel` 不接受 `threads=`；wrapper K11 已折叠为 `block_size`，见 wrapper L79–80/L163–164 与 `_DEFAULT_BLOCK_SIZE`）、一维 grid 保留均合理。**关键改进（相较同族 bitwise_and Stage 2 建议）**：dtype 契约处置已把 1 字节表示重设计**显式推广到 plain uint8 直接输入路径**（§0.5 "推广到 manifest/测试声明的全部 uint8 直接输入路径，不只 bool 特例"），并配 §4.1/§8.3 L0-8 覆盖——回应了先例复盘的 dtype 覆盖盲区教训。
- **NPU 重设计可行性 — pass**：R1/R2/R3 均具「源方案 → NPU 新算法 → 语义保持论证」三要素。R1 行/内层分解与源 divmod 链逐点相同（独立推演见维度 8 + E2 机器验证 11/11 mismatch=0）；"末维必有 `a_strides[-1]==1` 或 `b_strides[-1]==1`"的证明经独立核对 `coalesce_broadcast_dims`（源 L124–132 `_make_strides` 末维 stride 恒 1、零化仅当 padded 维=1 且 out>1，且合并时末组 stride 取原末维 stride）成立，`(0,0)` 组合仅在 `L==1` 全标量出现。R2 persistent 一维 Kernel + Vector 向量化满足一维 Kernel 约束（ascend-constraints §1）。R3 int8 承载位级恒等经 E1 + 独立重跑探针（A-int8 PASS / A-uint8 FAIL / G FAIL）确认。
- **设计与决策一致性 — pass**：§1–§7 描述的是 R1/R2/R3 决策后的 NPU 算法（行/内层向量化 + persistent + int8 承载 + int8 vbrc f16 transit），无"§0 判重设计、§3 照抄源方案"的脱节；§3.3 伪代码 buffer 形状、§4.5 预算、§6 循环结构三方一致。
- **golden 独立性 — pass**：§8.1 golden = `torch.bitwise_or` 广播直算，以 §0.1 源算子语义为唯一依据，明确**不复刻** §0.6 的 NPU 行/内层索引、vbrc、persistent 结构。
- **建议级（探针脚本 Probe E 自身缺陷，不影响设计结论）**：设计头注/§3.4/§9.2 引 Probe E 佐证"`T.vor` 拒绝标量操作数"，但 `design_probe_vor.py` 的 `probe_e` 定义 `main(A, C)`（2 参数）却用 `out_idx=[2]`，本次重跑先报 `ValueError: out_idx 2 is out of bounds for kernel with 2 parameters`——**该脚本并未真正触达 vor 标量路径**（标量拒绝的实测证据实际来自位运算族共性 CG-2026-0019 中的 `T.vand` 探针与 Probe F 的 rank-1 size-1 拒绝）。设计不依赖"标量立即数"路径（行内标量广播走 `T.vbrc` size-1 张量路径，由 Probe D 与 F 覆盖），故不影响可行性；建议把 `probe_e` 的 `out_idx` 改为 `[1]` 以真实覆盖该子断言。

### 1. API 可行性 — pass

- **API 存在性 — pass**：`T.vor` 有导出佐证（`tilelang/language/__init__.py` L84–85 `npuir_or` / `npuir_or as vor`）与实现（`tilelang/language/customize_npuir.py` L200–L211 `AscendBinaryOp("or", ...)`）；`docs/Tilelang.language/逻辑操作/T.vor.md` 存在。**本检视环境独立重跑探针确认 `T.vor` 在 NPU 上 int8/16/32/64 同形 + rank≥2 广播均 bit-correct**。`T.vbrc`（`docs/Tilelang.language/shape操作/T.vbrc.md`）、`T.vcast`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`，目录实查存在）、`T.copy`/`T.alloc_ub`/`T.min`/`T.ceildiv`/`T.serial`/`T.Kernel(is_npu=True)` 均有文档；`examples/elementwise/vec_add_1d.py` 提供完整 Developer 先例。`T.floordiv`/`T.floormod`（§3.2 行 base 展开链）为项目既有形态（先例实现记录见 bitwise_and RETROSPECTIVE Stage 3）。设计对 `T.vor` 的两条负向断言（uint8 不可用 / 不接受标量或 size-1 操作数）均有 probe 复现 + CG-2026-0018/0019 登记（详见维度 8），且**本次独立重跑探针复现了 uint8 失败与 rank-1 size-1 失败**。
- **API 与模式匹配 — pass**：Developer 模式全程自动同步，无手动 `sync_block_set/wait`；§6 无逐元素 `T.Parallel` 标量循环。
- **v-prefix 优先 — pass**：主用 `T.vor`/`T.vbrc`/`T.vcast`（v-prefix），无 npuir_xxx 回退。

### 2. 内存层级规划 — pass

- **搬运路径完整 — pass**：§4.4 给出 GM → UB → GM 路径（纯 Vector），无 GM→L0 越级路径、无 L1/L0。
- **UB 预算 — pass（人工复核 design_calc_check skip 项）**: §4.5 逐 dtype 复算 3×4096×{1,2,4,8}B = 12/24/48/96 KB，×1.7 = 20.4/40.8/81.6/163.2 KB 均 < 192 KB ✓；int8 承载 + 末维 stride=0 分支含 f16 transit（a_f16/b_f16 各 block_size×2B）≈ 3×4096 + 2×8192 = 28672B ≈28 KB，×1.7 ≈48 KB ✓。**§4.5 "block_size 上限"行独立复算**：196608/(1.7×3×E) → int64 4818 / int32 9637 / int16 19275 / int8 38550；196608/(1.7×(3+4)) → int8+f16 transit 16521——与设计逐值一致（使用真实 1.7 因子）。推荐 block_size=4096 对 int64 最紧 163.2 KB < 192 KB ✓。
- **L0C 容量 / Buffer 层级标注 — pass/n.a.**：纯 Vector 无 L0C；§4.3 每个 buffer 标注 UB，并区分编译期条件分配（a1/b1 仅 stride=0 时、a_f16/b_f16 仅 int8 承载+stride=0 时）。

### 3. Tiling 策略 — pass（含 1 条建议级）

- **Block 划分明确 — pass**：§5.2 给出 block_size=4096 及 5 条理由（innermost 单块覆盖 `L≤4096`、32B 尾轴对齐、向量宽度整除、UB 预算、同形分块）。
- **对齐约束 — pass**：block_size×E = 4K/8K/16K/32K（32B 整数倍）；向量宽度整除（int8×16 / int16×8 / int32×4 / int64×2），manifest 广播末维 `L∈{128,3136,4096,768}` 对 int32×4 整除（尾 lane 0%）。
- **非整除处理 — pass**：§6.4 `tail = T.min(block_size, L − inner_off)` 尾块截断 + 零起点 dst 切片，来源 `vec_add_1d.py`；不采用 host padding（符合 ascend-constraints §4）。
- **分核策略三要素（`_shared/standards/core-split-strategy.md` §1 核对）— pass**：
  - ① 逻辑核数计算：§5.5 表逐 workload 给出 `num_logical = R × ceil(L/block_size)`。**独立复算**（含 `coalesce_broadcast_dims` 逐维核对）：smoke-1k coalesced=(1024,)→1；hidden-state coalesced=(8388608,)→chunks 2048→2048；cnn-feat coalesced=(16,256,3136)（a_strides=(802816,3136,1)，b_strides=(0,1,0)）→R=4096, chunks=1→4096；row-broadcast coalesced=(4096,4096)（a=(4096,1)，b=(0,1)）→4096。与设计表一致。
  - ② 物理核数依据（实查）：§5.5 记录 `NPUUtils.get().get_aicore_num() → 20`，纯 Vector ×2 = 40；说明与 `CONST-aicore-910B2C`(24⇒48) 不同源、以本机实查为准、工厂闭包每次特化重查无硬编码。**本检视环境独立实跑返回 20，与设计记录一致**，符合标准"禁止文档假设/经验值替代实查"。
  - ③ 规模判定与分核方案：smoke-1k（≤40）判"无需适配"并附依据；其余判"极大规模"，`num_kernels=min(num_logical,40)`、核内 `T.serial(num_local)`（静态边界，N/L/block_size/40 均编译期常量）、`task=i×num_kernels+cid` + 越界屏蔽；含失败信号。满足 persistent 标准结构。
- **建议级（§5.5 规模判定段 num_logical 集合笔误）**：正文写"其余三个 workload `num_logical ∈ {2048, 4096, 8192}`"，但同节表格与独立复算均为 {2048, 4096, 4096}（"8192" 疑为自同族 bitwise_and 设计沿用/笔误，manifest 无对应 shape）。建议更正为 {2048, 4096, 4096}。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（源即一维 grid；NPU 一维 `T.Kernel(num_kernels)`）。
- GEMM 非整除 / L0C 溢出：不适用（纯 Vector）。
- 已知约束逐项检查（§3.5.1）：`threads=`/`T.Parallel` 舍弃、uint8 不可用（CG-2026-0018，覆盖 bool + plain uint8 两路）、`T.vor` 标量/size-1 限制（CG-2026-0019）、`T.vbrc` int8 dst 运行期挂死（走 f16 transit）、host 侧输入禁改（ascend-constraints §4）均在 §0.5/§0.6/§9 有对应处置。

### 5. 循环与同步 — pass

- 循环结构（§6）：核间 persistent `T.Kernel`、核内 `T.serial(num_local)`（静态边界）、元素级 `T.vor` 整 buffer 向量化、标量广播 `T.vbrc`、int8 路径 `T.vcast` 中转；无逐元素标量循环（与 §1.6.2 结论一致）。
- 同步与模式匹配（§7）：Developer 自动同步，与模式一致；每 task 独立、无核间依赖、无 workspace、无 `sync_block_set/wait`。
- 尾块处理（§6.4）：给出 shape 不整除时的 `tail` 逻辑与零起点切片规避 UB 非零起点对齐。
- 流水（§6.3）：明确不用 `T.Pipelined`（单阶段无 K 维累加可双缓冲），理由与算子形态一致。

### 6. 验证方案 — pass

- Golden 函数（§8.1）：`torch.bitwise_or` 广播直算，独立于 NPU 算法（与冻结测试 `tests/ops/test_bitwise_or.py` 的 `ref_program`/`exact_compare` 一致）。
- L0 测试计划（§8.3）：11 个 case 覆盖同形各 dtype、大规模、广播三模式、a 侧广播、标量/全标量、尾块非整除、bool 存储路径、**uint8 直接输入路径（L0-8）**、int8 承载+广播 f16 transit（L0-9）、persistent 边界、二补码极值——**相较同族 bitwise_and 先例补齐了 uint8 直接输入与 int8 vbrc transit 专项**。
- 精度标准（§8.2）：整数/布尔 bit-exact（`exact_compare`），与源测试基准一致；输出 shape 断言列出。

### 7. 完整性与一致性 — pass

- 章节齐全：§0（0.1–0.7 全）+ §1–§11 齐全（迁移类模板要求）。
- 无占位符：全文检索未见未完成占位标记（设计文档与本节均无空占位符；唯一花括号命中为 §1.3 的 LaTeX 数学式，非占位符）。
- 同类引用真实：Glob 核对 §3.5.3 全部引用路径（`examples/elementwise/vec_add_1d.py`、`testing/npuir/bit_ops/test_binary_bit_op_dev.py`、`examples/elementwise/vec_add_auto_brc.py`、`lerp_tensor_kernel/`、`mish_kernel/`、`examples/bitwise_and/bitwise_and_fwd_kernel/`）与 §0.7/§10.1 引用（wrapper/op/manifest/test/bench、`history_version/design_probe_vor.py`、`verify_equiv.py`）均存在。
- 内部一致：API 映射（§3）/ 内存规划（§4）/ Tiling（§5）三处相互一致；§4.1 输入表与 §0.2/§8.2/§8.3 均含 uint8 + bool 承载（无先例 dtype 覆盖盲区）。

### 8. 算法优化分析 — pass

- **§1.6 章节存在 — pass**：含 §1.6.0 算法调研 / §1.6.1 数学等价优化 / §1.6.2 向量化替代分析 / §1.6.3 向量化轴与数据布局决策四子节；单步逐元素类采用轻量调研但四问齐全。
- **算法调研存在性与完整性 — pass**：R1 候选表含**基线**（候选 0 = `y=a|b` + 广播索引）并评估 3 个等价候选（算术分解 `a+b−(a&b)`、De Morgan `~(~a&~b)`、位平面重排）；R2 有明确结论"无在线变体需求（天然单遍）"并附结构依据（`y_i` 仅依赖 `(a_{off_a(i)}, b_{off_b(i)})`，无 running 统计量/跨元素依赖）；源算法亦单遍（§0.4 无 online 手段），一致；R3 表覆盖**四口径**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）且基线在表（机械 `r3_metrics=pass`）；R4 逐候选六维（单元匹配/容量/对齐/静态边界/流水融合）评估，"无更优替代"列了 6 条调研范围。
- **调研结论正确性（独立复核）— pass**：
  - ① 等价性：位运算本体 `a|b` 为 1 op/元素下界；候选 1（`a+b−(a&b)`）与候选 2（`~(~a&~b)`）独立复算 op 数均 ×3（2 加+1 与 / 2 not+1 and），候选 3 位平面重排引入 repack + 二遍扫描，均无访存收益——设计"负收益"判定成立；R1 索引重映射与原 divmod 映射等价（本人独立推演：`row=flat//L, inner=flat%L`、`a_base=Σ_{i<k−1}coord_i·a_strides[i]+inner·a_strides[-1]` 与 mixed-radix divmod 链逐点相同 + E2 机器验证 11/11 mismatch=0、oob=0）。
  - ② 复杂度复算：R3 的 `N + R·O(ndim)` divmod（较基线 `N·O(ndim)` 降 ~L 倍，cnn-feat L=3136 → 4096·3≈1.2万 vs 12.85M·3≈3855万）、访存 `(A+B+N)·E`、单遍、`O(block_size)` 中间缓冲算术自洽；纯 Vector 每 task 一次进出，**无中间缓冲 GM 往返漏计**。§1.6.0 roofline 独立复算一致：hidden-state `(3×8.39M)×4B=100.7MB ÷1.05TB/s≈95.9µs`、发射 `2048/40×(1 op)×0.5µs≈25.6µs`；cnn-feat `(2×12.845M+256)×4B≈102.8MB ÷1.05≈97.9µs`、发射 `4096/40×2 op×0.5µs≈102µs`。设计已把跨核数外推（本机 40 核 vs 常数 48 核）与 stale 常数标注为未实测假设，符合负向/不确定证据规则。
  - ③ 亲和性：与 `T.vor.md`/`T.vbrc.md`、`ascend-constraints.md` §4、pattern-library（elementwise copy-floor / loads-first）不矛盾；淘汰源基线（NPU 无逐元素 divmod 快路径 + 标量 gather 落错单元）与源形态（GM 标量 gather）一致。
  - ④ 负向断言复核：对照 `algorithm-candidates.md` **ALG-elementwise** 命中行——R1 仅公共子表达式消除 / 除法乘倒数 / cast 链合并（本算子 `a`/`b` 各引用一次、无除法、无 cast，均不适用）；R2 "逐元素映射天然单遍"——**与设计"无在线变体"结论一致，无被漏掉的已知候选**。CG-2026-0018/0019 两条负向断言均有 probe 复现（**本次独立重跑 Probe A-uint8/G-uint8 确认 uint8→float32、Probe F 确认 rank-1 size-1 verify fail**）+ docs 限制条款对照（0018 引 `T.vand.md`/`T.vbrc.md` §2.2 标 uint8 √ 与实测矛盾；0019 引 `T.vor.md` §2.1 文档声称 src1 支持标量而实测拒绝——属"文档-行为不符"），登记于 `.agents/evolution/capability-gaps.md` L580–L680 且状态 recurring（occurrences=2）。
  - ⑤ 并列候选裁决：row-broadcast 备选 #4 有主选+备选+裁决计划（见 §1.6.3）。
- **调研→设计一致性 — pass**：§1.4（单遍行/内层向量算法）与 §1.6.0 选定一致；§3.1 以"优化后公式=原式、索引路径按 R1、存储按 R3"为输入；§6 循环结构 = persistent + serial，无 §1.6.2 未论证的标量循环；迁移侧 §0.4→§0.6 承接总表逐项列出、无静默丢弃（§11 复核 8 项源优化手段全有承接）。
- **数学等价优化完整性 — pass**：§1.6.1 逐类别（恒等变形 / 稳定化 / CSE / 算子降代换 / 访存削减 / 归约结构）核查，结论"无公式级优化空间"依据充分（位运算本体 op 数下界 + 两恒等式 op 数 ×3）；R3 存储替换与 R1 索引重映射明确标注为"语义保持的等价替换（存储/索引表示），非公式级优化"。
- **等价性论证独立推演 — pass**：R1 行/内层分解与源 mixed-radix divmod 链逐点恒等（独立推演 + E2 机器验证双重确认）；R3 int8/uint8 位模式同构（E1 机器验证 + 本检视独立重跑探针 A-int8 PASS / A-uint8 FAIL 互为佐证）。§1.6.1 内嵌 E1/E2 结果表与重跑输出逐一致。
- **向量化替代覆盖完整性 — pass**：§1.6.2 盘点 7 个循环/标量点，与 §3.3 伪代码、§6 循环结构交叉核对无遗漏（#1 逐位或=vor、#2 标量广播=vbrc、#3 行 base divmod、#4 tail=T.min、#5 task 循环=T.serial、#6 host 元数据、#7 int8 vbrc f16 transit=vcast/vbrc）；逐元素 `|`（#1）与 size-1 标量广播（#2）已向量化（docs+testing 双佐证 + 探针）；#3–#6 的不可替代理由均具体到本算子（block 级索引/任务映射、动态 shape 边界、persistent 调度结构、host 侧计算），非空泛理由；`T.arange`+divmod 方案被 `CONST-arange-scalar-materialize`（~1ns/elem）实测否决，依据成立。
- **替代 API 可行性 — pass**：`T.vor`（探针实测 int8/16/32/64 PASS）、`T.vbrc`（探针 D PASS）、`T.vcast`（docs + 同工具链先例 repro bit-exact）均有佐证，优先 v-prefix。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 候选矩阵 4 项（展平一维 innermost 向量化 / 核内 N 维 auto-broadcast / host 预广播物化 / row-broadcast 2-D tile），逐候选量化取舍（跨步 MTE2 段数 ×256、UB 3.2MB 超容、host 禁改输入约束、UB `(2·R_blk+1)·L·E`）；主选 #1 有轴质量评分（整除性 / 尾 lane 浪费 / 累加链 / repack / UB）；源码 GPU 轴按"硬件组织输入非结论"独立评估（§1.6.3 源码轴对照）。**独立复算**候选 #4：`(2·2+1)×4096×8B=160KB`，×1.7 超 192KB；int32 `5×4096×4B=80KB` ✓——与设计一致。**实验裁决三件套齐全**：主选 #1 + 备选 #4（`[R_blk,L]` 2-D tile，R_blk=2/L=4096，int32/int64 路径）+ 裁决计划（代表 shape `[4096,4096]|[1,4096]` int32/int64、msprof Task Duration + mte2_active_bw、阈值备选 latency < 主选×0.97 才翻转、按 `sb=1 & b 全外层广播` 分派回写）。
- **布局决策与 §3.3/§4/§6 一致 — pass**：选定核内布局 `(block_size,)` UB buffer、向量轴 = coalesced 末维（`k=1` 即展平 N 轴），与 §3.3 伪代码 buffer 形状、§4.5 UB 预算、§6 循环结构三方一致。

---

> 结论为「通过」，无阻塞级问题。两条建议级备注（§5.5 正文 num_logical 集合笔误；探针 `probe_e` 自身 out_idx 缺陷导致标量子断言未被脚本真实覆盖）已分别记录于维度 3 / 维度 0，不阻塞 Stage 3，供 Stage 1 择机订正。维度 0/1/8 的关键能力结论（`T.vor` 存在性与 dtype 覆盖、uint8 能力缺口、rank-1 size-1 广播限制、`T.vbrc` 可用性）已由本检视**独立重跑 NPU 探针**确认。
