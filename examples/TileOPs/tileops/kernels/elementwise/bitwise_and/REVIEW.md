# bitwise_and_fwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/bitwise_and/bitwise_and_fwd_kernel/DESIGN.md
- 任务类型: 迁移（`scenario=migration`，`migration_mode=harness`，GPU 仓 TileOPs 同构工程）
- 源算子: `examples/TileOPs/tileops/kernels/elementwise/bitwise_and/_bitwise_and_fwd_kernels.py`（Stage 0 提取件）；GPU 原上下文 `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`、`/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/bitwise.py`、`/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/_base.py`
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 4

## 机械复核（D-1/D-5）

- **verify_equiv 重跑（D-1）**: `python3 examples/bitwise_and/bitwise_and_fwd_kernel/verify_equiv.py` → `OVERALL: EQUIV_PASS`。
  - E1（int8 vs uint8 存储位级等价，1M 随机字节 + 全 256 角点）: AND / XOR / NOT 三项 `violations=0`，与 §1.6.1 内嵌表一致。
  - E2（NPU 行/内层索引映射 vs 源 `_compute_broadcast_offsets` divmod 映射，torch/numpy CPU 逐点对照）: 11 组形状（含全部 4 个 manifest workload + 7 组测试/角色互换广播）全部 `a_mismatch=0, b_mismatch=0, oob=0`，与 §1.6.1 内嵌表逐行一致。
  - 结论: 执行输出与 §1.6.1 内嵌表一致且全 `EQUIV_PASS`（§1.6.1 无公式级采纳项；E1/E2 为该设计的两项语义保持等价替换——存储表示 int8/uint8、索引表示 R1 重映射——机器验证有效）。
- **design_calc_check 重跑（D-5）**: `python3 .agents/tools/design_calc_check.py --design examples/bitwise_and/bitwise_and_fwd_kernel/DESIGN.md` → JSON `checks` 摘要:
  - `ub_budget`: **skip**（"no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"）→ **人工已核**：§4.5 手工表 3 buffer × block_size，int64 @4096 = 3×4096×8 = 98304B = 96KB，×1.7 = 163.2KB < 192KB ✓；int8/16/32 = 20.4/40.8/81.6KB ✓；a1_ub/b1_ub 各 1 元素可忽略。详见维度 2。
  - `l0c_budget`: **skip**（"no affirmative Cube-path API ... in this design"）→ 人工已核：纯 Vector，无 L0C，不适用。
  - `core_split`: **skip**（"insufficient parsed dims {}; declared logical cores = None"）→ **人工已核**：§5.5 逐 workload 复算 num_logical = R×ceil(L/block_size) 与设计表一致（smoke-1k=1 / hidden-state=2048 / cnn-feat=4096 / row-broadcast=4096）；物理核数 `NPUUtils.get().get_aicore_num()` 实查记录齐全。详见维度 3。
  - `r3_metrics`: **pass**（四口径 FLOPs / access bytes / scan passes / buffer peak 齐全）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**检视方式**：亲自 Read `_bitwise_and_fwd_kernels.py` 全文（185 行）、`/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（L140–147 `_BITWISE_DTYPES`、L555–609 `_make_binary_explicit`、L890–1128 `BinaryKernel`、L1318–1338 `_Uint8StorageBinaryKernel`、L1866–1920 bitwise 子类）、`ops/elementwise/bitwise.py` 全文、`ops/elementwise/_base.py`（L684–803 `BinaryOp`、L1087–1123 `_BoolOutputBinaryOp`）、`examples/TileOPs/tileops/kernels/elementwise/bitwise_and/bitwise_and.py`（NPU wrapper）、`ops/elementwise/bitwise_and.py`（NPU Op）、`manifest/elementwise.yaml`（`BitwiseAndFwdOp`）、`tests/ops/test_bitwise_and.py`，逐项核对 §0。

- **语义理解正确性 — pass**：§0.1 数学语义 `y = a & b`、无规约/无跨元素依赖、bit-exact 与 `BitwiseAndFwdKernel.op_func = a & b`（elementwise.py L1874–1876）一致；`_BITWISE_DTYPES = (bool,uint8,int8,int16,int32,int64)`（L140–147）与 §0.1 完全一致；"整数路径在原生 dtype 域做位运算（无中间提升）"与源码一致；bool 特殊路径 `view(torch.uint8)`+`*BoolStorageFwdKernel` 与 `_BoolStorageBitwiseBinaryOp._eager_forward`（`ops/elementwise/bitwise.py` L41–52）、`_Uint8StorageBinaryKernel`（elementwise.py L1329–1333）一致。§0.2 I/O（`y: T.Tensor((N_total,), out_dtype)`、`out_idx=[2]`、一维展平输出）与源码 L52–56 / L75–78 一致；输出广播形状还原与 NPU Op `_eager_forward().reshape(self.out_shape)`（`ops/elementwise/bitwise_and.py` L120–133）一致。
- **算法解读完整性 — pass**：§0.3 步骤表覆盖源码全部计算语句——host `coalesce_broadcast_dims`（L564/L935）、同形快路径 `y[idx]=a[idx]&b[idx]`（L576–578）、广播路径 `y[flat]=a[a_off]&b[b_off]`（L600–605）、`_compute_broadcast_offsets` 编译期 unrolled divmod 链（L160–176）——无遗漏、无臆造；数据流图覆盖 a/b/y buffer；host 侧 lru_cache 特化（L555）已列出。提取件头部说明的第二操作数内联修正（`a[...] & b` → `a[...] & b[...]`）经核对与 GPU `op_func(a,b)=a&b` 一致。
- **优化手段覆盖完整性 — pass**：§0.4 8 项与源码逐项对应（`T.Parallel(threads,npt)` L576/L600；连续段 coalesced；`coalesce_broadcast_dims`；unrolled divmod；同形快路径 L564；`register_copy` fragment L464–500；工厂+lru_cache L555；bool→uint8 存储）。抽检源码 binary 路径无 warp shuffle / SMEM tiling / 异步流水 / swizzle，§0.4 "对照 migration-analysis.md §4.1 清单无遗漏"成立。
- **耦合性判定合理性 — pass（含 1 条建议级）**：§0.5 四态处置（保留 7 / 等价替换 3 / 重新设计 3 / 舍弃 1）逐项附依据；"逐元素 divmod 广播索引 → 重新设计"（NPU 无逐元素整数 divmod 快路径 + v-prefix 要求向量化，源形态为 GM 标量 gather）判定有据；`register_copy` 寄存器→UB 等价替换、`threads=`/npt kwarg 舍弃、一维 grid 保留均合理。**建议级（dtype 契约处置不完整）**：§0.5 "dtype 契约（bool/uint8/…）保留（仅 bool 存储表示需重设计）"未覆盖 manifest 明确声明且测试覆盖的**直接 uint8 输入**——CG-2026-0018 同样作用于被声明为 uint8 的输入 GM 张量（非仅 bool 存储路径），详见维度 6/7。
- **NPU 重设计可行性 — pass**：R1/R2/R3 均具「源方案 → NPU 新算法 → 语义保持论证」三要素。R1 行/内层分解（`row=flat//L, inner=flat%L` 的 mixed-radix 分解）与源 divmod 链逐点相同（独立推演 + E2 机器验证 11/11 mismatch=0）；"末维必有 `a_strides[-1]==1` 或 `b_strides[-1]==1`"的证明经独立核对 `coalesce_broadcast_dims` stride 合成规则（`_make_strides` 末维 stride 恒 1、零化仅当 padded 维=1 且 out>1）成立，且 `(0,0)` 组合仅在 `L==1` 全标量出现。R2 persistent 一维 Kernel + Vector 向量化满足一维 Kernel 约束；R3 int8 承载位级恒等经 E1 验证。
- **设计与决策一致性 — pass**：§1–§7 描述的是 R1/R2/R3 决策后的 NPU 算法（行/内层向量化 + persistent + int8 存储），无"§0 判重设计、§3 照抄源方案"的脱节；§3.3 伪代码 buffer 形状、§4.5 预算、§6 循环结构三方一致。
- **golden 独立性 — pass**：§8.1 golden = `torch.bitwise_and` 广播直算，以 §0.1 源算子语义为唯一依据，明确**不复刻** §0.6 的 NPU 行/内层索引、vbrc、persistent 结构。

### 1. API 可行性 — pass

- **API 存在性 — pass**：`T.vand` 有导出佐证（`tilelang/language/__init__.py` L98–99 `npuir_floordiv`、L87 区 `npuir_and as vand`）与官方测试（`testing/npuir/bit_ops/test_binary_bit_op_dev.py` L40 `T.npuir_and(...)`）；`T.vbrc`（`docs/Tilelang.language/shape操作/T.vbrc.md`）、`T.copy`/`T.alloc_ub`/`T.min`/`T.ceildiv`/`T.serial`/`T.Kernel(is_npu=True)` 均在 `docs/Tilelang.language/` 有文档，`examples/elementwise/vec_add_1d.py` L27–38 提供完整 Developer 先例；`T.floordiv`/`T.floormod` 存在于 `tilelang/language/tir/op.py` L2973/L2995（§3.2 行 base 展开链）。设计对 `T.vand` 的两条负向断言（uint8 不可用 / 不接受标量或 size-1）均有 probe 实测 + CG-2026-0018/0019 登记（详见维度 8）。
- **API 与模式匹配 — pass**：Developer 模式全程自动同步，无手动 `sync_block_set/wait`；§6 无逐元素 `T.Parallel` 标量循环。
- **v-prefix 优先 — pass**：主用 `T.vand`/`T.vbrc`（v-prefix），无 npuir_xxx 回退。

### 2. 内存层级规划 — pass（含 1 条建议级）

- **搬运路径完整 — pass**：§4.4 给出 GM → UB → GM 路径（纯 Vector），无 GM→L0 越级路径、无 L1/L0。
- **UB 预算 — pass（人工复核 design_calc_check skip 项）**: §4.5 手工表 int8/int16/int32/int64 = 3×4096×{1,2,4,8}B = ~12/24/48/96KB，×1.7 = 20.4/40.8/81.6/163.2KB，均 < 192KB ✓。独立复算 int64 @4096 = 98304B = 96KB → ×1.7 = 163.2KB < 192KB，与设计一致。
- **L0C 容量 — pass/n.a.**：纯 Vector 不使用 L0C。
- **Buffer 层级标注 — pass**：§4.3 每个 buffer 标注 UB。
- **建议级（§4.5 "block_size 上限"行算术偏乐观约 2%）**：设计给出 int64 ≤ 4915 / int32 ≤ 9830 / int8 ≤ 39320（标注"1.7x 膨胀口径"）。独立复算严格上界 = 196608/(3×E×1.7)：int64 ≤ 4818、int32 ≤ 9637、int8 ≤ 38550——设计的三个值等效因子 ≈1.667（5/3）而非 1.7，统一偏高约 2%。推荐值 block_size=4096 远低于严格上界，不影响本设计正确性；但若 Stage 4 按该行上界扫 block_size（如 int64 取 4915）会触发 UB 溢出。建议按 ×1.7 重算为 4818/9637/38550，或改注因子为 5/3。

### 3. Tiling 策略 — pass（含 1 条建议级）

- **Block 划分明确 — pass**：§5.2 给出 block_size=4096 及 5 条理由（innermost 单块覆盖、32B 尾轴对齐、向量宽度整除、UB 预算、同形分块）。
- **对齐约束 — pass**：block_size×E = 4K/8K/16K/32K（32B 整数倍）；向量宽度整除（int8×16 / int16×8 / int32×4 / int64×2）。
- **非整除处理 — pass**：§6.4 `tail = T.min(block_size, L − inner_off)` 尾块截断 + 零起点 dst 切片，来源 `vec_add_1d.py` L31–38。
- **分核策略三要素（`_shared/standards/core-split-strategy.md` §1 核对）— pass**：
  - ① 逻辑核数计算：§5.5 表逐 workload 给出 `num_logical = R × ceil(L/block_size)`。**独立复算**：smoke-1k coalesced=(1024,)→1；hidden-state coalesced=(8388608,)→2048；cnn-feat coalesced=(16,256,3136)（a_strides=(802816,3136,1)，b_strides=(0,1,0)）→R=4096, chunks=1→4096；row-broadcast coalesced=(4096,4096)（a=(4096,1)，b=(0,1)）→4096。与设计表一致。
  - ② 物理核数依据（实查）：§5.5 记录 `NPUUtils.get().get_aicore_num() → 20`，纯 Vector ×2 = 40；并说明与 `CONST-aicore-910B2C`(24⇒48) 不同源、以本机实查为准、工厂闭包每次特化重查无硬编码。**本检视环境实跑 `python3 -c "...get_aicore_num()"` 返回 20，与设计记录一致**；另有同仓先例 lerp_tensor 设计记 24（2026-09-07），系不同时间/环境值——设计以当前实查 20 为准且注明差异，符合标准"禁止文档假设/经验值替代实查"。
  - ③ 规模判定与分核方案：smoke-1k（≤40）判"无需适配"；其余判"极大规模"，`num_kernels=min(num_logical,40)`、核内 `T.serial(num_local)`（静态边界，N/L/block_size/40 均编译期常量）、`task=i×num_kernels+cid` + 越界屏蔽；含失败信号。满足 persistent 标准结构。
- **建议级（§5.5 规模判定段 num_logical 集合笔误）**：正文写"其余三个 workload `num_logical ∈ {2048, 4096, 8192}`"，但同节表格与独立复算均为 {2048, 4096, 4096}（8192 疑似来自非 manifest 测试形状）。建议更正为 {2048, 4096, 4096}。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（源即一维 grid；NPU 一维 `T.Kernel(num_kernels)`）。
- GEMM 非整除 / L0C 溢出：不适用（纯 Vector）。
- 已知约束逐项检查（§3.5.1）：`threads=`/`T.Parallel` 舍弃、uint8 不可用（CG-2026-0018）、`T.vand` 标量限制（CG-2026-0019）、host 侧输入禁改（ascend-constraints.md §4）均在 §0.5/§0.6/§9 有对应处置。

### 5. 循环与同步 — pass

- 循环结构（§6）：核间 persistent `T.Kernel`、核内 `T.serial(num_local)`（静态边界）、元素级 `T.vand` 整 buffer 向量化、标量广播 `T.vbrc`；无逐元素标量循环（与 §1.6.2 结论一致）。
- 同步与模式匹配（§7）：Developer 自动同步，与模式一致；每 task 独立、无核间依赖、无 workspace、无 `sync_block_set/wait`。
- 尾块处理（§6.4）：给出 shape 不整除时的 `tail` 逻辑。

### 6. 验证方案 — pass（含 1 条建议级）

- Golden 函数（§8.1）：`torch.bitwise_and` 广播直算，独立于 NPU 算法。
- L0 测试计划（§8.3）：9 个 case 覆盖同形各 dtype、大规模、广播三模式、a 侧广播、标量/全标量、尾块非整除、bool 存储、persistent 边界、二补码极值。
- 精度标准（§8.2）：整数/布尔 bit-exact（`exact_compare`），与源测试基准（`tests/ops/test_bitwise_and.py`）一致。
- **建议级（manifest 直接 uint8 输入 dtype 的验证缺口）**：manifest `BitwiseAndFwdOp` 的 signature 声明输入 dtype 含 `uint8`，冻结测试 `tests/ops/test_bitwise_and.py` L54 有 `pytest.param(4_096, torch.uint8, marks=[pytest.mark.smoke])`。§8.2 精度表与 §8.3 L0 计划均只列 int8/int16/int32/int64 + bool，**未覆盖直接 uint8 输入路径**。CG-2026-0018 表明被声明为 uint8 的 GM 张量/UB 运行时输出被静默映射 float32，故直接 uint8 输入同样需要 int8 存储重映射（与 bool 路径同机制，E1 已机器验证 int8/uint8 位级等价）。建议在 §0.5/§0.6 把 R3 由"bool 存储 dtype"扩展为"1 字节无符号语义（bool 存储 + 直接 uint8 输入）→ int8 存储重映射"，并在 §4.1/§8.2/§8.3 补入 uint8 输入 / 输出 view 回 uint8 的用例（否则 Stage 3 该 smoke 用例可能失败）。

### 7. 完整性与一致性 — pass（含 1 条建议级 + 1 条记录）

- 章节齐全：§0（0.1–0.7 全）+ §1–§10 齐全（迁移类 11 章节要求）。
- 无占位符：全文检索未见未完成占位标记（设计文档与本节均无空占位符）。
- 同类引用真实：Glob 核对 §3.5.3 全部引用路径（`examples/elementwise/vec_add_1d.py`、`testing/npuir/bit_ops/test_binary_bit_op_dev.py`、`examples/elementwise/vec_add_auto_brc.py`、`lerp_tensor_kernel/`、`mish_kernel/`）与 §0.7/§10.1 引用（wrapper/op/manifest/test/bench、`history_version/design_probe_*.py`）均存在。
- 内部一致：API 映射 / 内存规划 / Tiling 三处相互一致。
- **建议级（§0.1 引用路径笔误）**：§0.1 写 bool 特殊路径出处"（`_base.py` L18–L52、`elementwise.py` L971–L1002）"，其中 `_BoolStorageBitwiseBinaryOp`（`view(torch.uint8)`）实际位于 `ops/elementwise/bitwise.py` L18–L52（`_base.py` L18–L52 为模块文档串/import 区）。语义内容正确，仅路径笔误，建议改为 `ops/elementwise/bitwise.py`。
- **记录（非问题）**：§0.2 输入 dtype 列 uint8，而 §4.1 输入张量表仅列 int8/int16/int32/int64（+ bool 经 int8 承载）——该差异即维度 6 建议级所述的 uint8 覆盖率问题，二者同源。

### 8. 算法优化分析 — pass

- **§1.6 章节存在 — pass**：含 §1.6.0 算法调研 / §1.6.1 数学等价优化 / §1.6.2 向量化替代分析 / §1.6.3 向量化轴与数据布局决策四子节，且按单步逐元素类采用轻量调研但四问齐全。
- **算法调研存在性与完整性 — pass**：R1 候选表含**基线**（候选 0 = `y=a&b` + 广播索引）并评估 3 个等价候选（算术分解 `(a+b−(a^b))/2`、De Morgan `~(~a|~b)`、位平面重排）；R2 有明确结论"无在线变体需求（天然单遍）"并附结构依据（逐元素映射无跨元素依赖、无 running 统计量；源算法亦单遍）；R3 表覆盖**四口径**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）且基线在表（机械 `r3_metrics=pass`）；R4 逐候选六维（单元匹配/容量/对齐/静态边界/流水融合）评估。
- **调研结论正确性（独立复核）— pass**：
  - ① 等价性：位运算本体 `a&b` 为 1 op/元素下界；候选 1/2/3 独立复算 op 数与结构确实 ×3–4 且引入中间量/repack，无访存收益；R1 索引重映射与原 divmod 映射等价（本人独立推演 + E2 机器验证 11/11 逐点 mismatch=0、oob=0）。
  - ② 复杂度复算：R3 的 N + R·O(ndim) divmod（较基线 N·O(ndim) 降 ~L 倍）、访存 `(A+B+N)·E`、单遍、O(block_size) 中间缓冲算术自洽，无遗漏中间 GM 往返（纯 Vector 每 task 一次进出）。§1.6.0 roofline 独立复算一致：hidden-state 100.7MB÷1.05TB/s≈95.9µs、发射 2048/40×(1 op)×0.5µs≈25.6µs；cnn-feat a+out=102.76MB÷1.05≈97.9µs、4096/40×(vand+vbrc)×0.5µs≈102µs。b 行内标量项精确值（coalesced b_strides=(0,1,0)，每 row 读 1 个 b 元素 → 4096 元素≈16KB）与设计写的 256·4B 有微小差异，但相对 51MB 主导项可忽略，不影响下界。
  - ③ 亲和性：与 `T.vand.md`/`T.vbrc.md`、`ascend-constraints.md` §4、pattern-library（elementwise.md copy-floor / loads-first / MTE3 strided）不矛盾；淘汰源基线的理由（NPU 无逐元素 divmod 快路径 + 标量 gather 落错单元）与源形态（GM 标量 gather）一致。
  - ④ 负向断言复核：对照 `algorithm-candidates.md` ALG-elementwise 命中行——R1 仅公共子式消除 / 除法乘倒数 / cast 链合并（本算子 `a`/`b` 各引用一次、无除法、无 cast，均不适用）；R2 天然单遍——**无被漏掉的已知候选**。调研范围列了 6 条（参考表命中行、恒等变形自查、本仓 elementwise 同类、pattern-library、源结构分析、位平面外部思路），"无更优替代"结论成立。CG-2026-0018/0019 两条负向断言均有 probe 复现 + docs 限制条款对照（0019 引 `T.vadd.md` §2.3 第 4 条"仅支持 src1 标量"对照 + vand 参数表无 scalar 形态；0018 引 `T.vand.md` §2.2 标 uint8 √ 与实测矛盾），登记于 `.agents/evolution/capability-gaps.md` L580–648，并各有 `history_version/design_probe_*.py` 复现脚本。
  - ⑤ 并列候选裁决：row-broadcast 备选 #4 有主选+备选+裁决计划（见 §1.6.3）。
- **调研→设计一致性 — pass**：§1.4（单遍行/内层向量算法）与 §1.6.0 选定一致；§3.1 以"优化后公式=原式、索引路径按 R1"为输入；§6 循环结构 = persistent + serial，无 §1.6.2 未论证的标量循环；迁移侧 §0.4→NPU 承接总表逐项列出、无静默丢弃。
- **数学等价优化完整性 — pass**：§1.6.1 逐类别（恒等变形 / 稳定化 / CSE / 算子降代换 / 访存削减 / 归约结构）核查，结论"无公式级优化空间"依据充分（位运算本体下界 + 两恒等式 op 数 ×3–4）；E1 int8/uint8 存储等价与 R1 索引等价明确标注为"语义保持的等价替换（存储/索引表示），非公式级优化"。
- **等价性论证独立推演 — pass**：R1 行/内层分解 `row=flat//L, inner=flat%L`、`a_base=Σcoord_i·a_strides[i]+inner·a_strides[-1]` 与源 mixed-radix divmod 链逐点恒等（独立推演 + E2 机器验证双重确认）；R3 int8/uint8 位模式同构（E1 机器验证）。
- **向量化替代覆盖完整性 — pass**：§1.6.2 盘点 6 个循环/标量点，与 §3.3 伪代码、§6 循环结构交叉核对无遗漏；逐元素 `&`（#1）与标量广播（#2）已向量化（`T.vand`/`T.vbrc`，docs+testing 双佐证 + 探针）；#3 行 base divmod / #4 动态边界 tail / #5 tile 调度循环 / #6 host 元数据 的不可替代理由均具体到本算子（block 级索引、动态 shape 边界、persistent 结构、host 侧计算），非空泛理由；`T.arange` 向量化方案被 `CONST-arange-scalar-materialize` 实测（~1ns/elem）否决，依据成立。
- **替代 API 可行性 — pass**：`T.vand`/`T.vbrc` 有 docs + testing 佐证，探针实测通过（见维度 1）。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 候选矩阵 4 项（展平一维 innermost 向量化 / 核内 N 维 auto-broadcast / host 预广播物化 / row-broadcast 2-D tile），逐候选量化取舍（跨步 MTE2 段数、UB 3.2MB 超容、host 禁改输入约束、UB (2·R_blk+1)·L·E）；主选 #1 有轴质量评分（整除性 / 尾 lane 浪费 / 累加链 / repack / UB）；源 GPU 轴按"硬件组织输入非结论"独立评估；弃选前提与 `T.vand.md`/`ascend-constraints.md` §4 核对一致。**实验裁决三件套齐全**：主选 #1 + 备选 #4（`[R_blk,L]` 2-D tile，R_blk=2/L=4096，int32 UB 80KB，int64 按 R_blk=1 退化）+ 裁决计划（代表 shape `[4096,4096]&[1,4096]` int32/int64、msprof Task Duration + mte2_active_bw、阈值备选 latency < 主选×0.97 才翻转、按 `sb=1 & b 全外层广播` 分派回写路径）。
  - **记录（次要，不阻塞）**：① 候选 #2 主因（UB 需驻留 256×3136×4B≈3.2MB 超容）成立且量化充分；其次因"MTE2 段数 ×256、非连续"与 pattern-library `CONST-gm-to-ub-bw-dilution`（跨步列块读本身不慢，段宽≥数百字节时正常）口径略有张力——因主因独立充分，弃选结论不受影响，建议后续对齐引用口径。② 备选 #4 给出 buffer 形状/UB 预算/dtype 路径，但核内循环结构与分核三要素未按新任务粒度完整重写，略低于"Stage 4 可直接实现深度"；因仅服务 row-broadcast 单场景且阈值/分派明确，建议 Stage 4 启动时补齐。
- **布局决策与 §3.3/§4/§6 一致 — pass**：选定核内布局 `(block_size,)` UB buffer、向量轴 = coalesced 末维（`k=1` 即展平 N 轴），与 §3.3 伪代码、§4.5 UB 预算、§6 循环结构三方一致。

---

> 结论为「通过」，无阻塞级问题。四条建议级备注（§4.5 block_size 上限算术偏乐观 ~2%；§5.5 num_logical 集合笔误；§0.1 引用路径笔误；manifest 直接 uint8 输入 dtype 未在 §0.5/§0.6/§4.1/§8.2/§8.3 覆盖）已分别记录于维度 2 / 3 / 7 / 6（及维度 0）说明中，不阻塞 Stage 3，供 Stage 1 择机订正；其中 uint8 输入覆盖建议在进入 Stage 3 前优先补齐。
