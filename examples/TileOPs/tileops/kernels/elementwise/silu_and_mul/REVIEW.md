# `_make_fused_gated_explicit`（silu_and_mul / `SiluAndMulFwdOp`）设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/silu_and_mul/_make_fused_gated_explicit/DESIGN.md
- 任务类型: 迁移（harness 子模式，`developer`）
- 源算子: `/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/activations.py`（`SiluAndMulFwdOp` @ L328，GPU 工厂 `_make_fused_gated_explicit` @ `kernels/elementwise.py` L646）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 1（§1.6.1「vexp/vexp2 同价」前提与所引 doc 矛盾，见维度 8；不影响选定算法与等价性，非阻塞）

## 机械复核（D-1/D-5）

- verify_equiv 重跑: **PASS**。`python examples/silu_and_mul/_make_fused_gated_explicit/verify_equiv.py` 实跑输出 `OVERALL: EQUIV_PASS (15/15)`（15 行全 `EQUIV_PASS`，各 4120 点，违反数全 0，max_ulp 0.00），与 §1.6.1 内嵌结果表**逐行一致**（E1-cand/base/交叉 + E2-fused 各 ×{fp16,bf16,fp32}）。脚本内容核对：`silu_ref = g·sigmoid(g)`（fp64 参照）、`silu_exp = g/(1+exp(-g))`、`silu_exp2 = g/(1+exp2(-g·log2e))`，角点含 NaN/±Inf/±0/subnormal/finfo.max/tiny + 4096 随机点，判据取 manifest 容差——等价性机器验证有效且支持「撤销 exp2 降代换」结论。
- design_calc_check: `python3 .agents/tools/design_calc_check.py --design examples/silu_and_mul/_make_fused_gated_explicit/DESIGN.md` 输出：
  - `ub_budget: skip`（"no per-level buffer tables parsed from the memory plan"）→ **机械跳过，人工已核 pass**：§4.3/§4.5 的 buffer 表列于 Markdown 表格，解析器未识别；人工按 §3.3 逐 buffer 复算，即使把 doc 未计入的 `d32`（fp16 行）/`t32`+`d32`（fp32 行）全部独立计入，最坏口径 bf16 `(2+2+4+4+4+4+4)=24 B/elem × 2048 元素 = 48KB`、×1.7 膨胀 ≈ 82KB < 192KB（fp16 ≈68KB、fp32 ≈41KB），结论成立。
  - `l0c_budget: pass`（"block 2x2 x fp32 accum = 0.0 KB vs L0C 128 KB"）——纯 Vector 无 L0C，一致。
  - `core_split: skip`（"insufficient parsed dims {'bm': 2}；declared logical cores = None"）→ **机械跳过，人工已核 pass**：独立复算 §5.5 四 workload 的 `m_blocks/chunks/num_logical`（512 / 14336 / 14 / 4096）与 `num_kernels=min(num_logical,40)`、`num_local=ceildiv(num_logical,num_kernels)`（13 / 359 / 1 / 103），逐项与 §5.5 表一致（40×13=520≥512、40×359=14360≥14336、40×103=4120≥4096），算术无误。
  - `r3_metrics: pass`（"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**亲自 Read 源码核对**（`tileops/kernels/elementwise.py`、`tileops/ops/elementwise/activations.py`、`tileops/ops/elementwise/_base.py`、Stage 0 提取件 `_silu_and_mul_fwd_kernels.py`），逐项结论：

- **语义理解正确性 — pass**。§0.1 数学语义 `y[m,n]=silu(x[m,n])·x[m,N+n]` 与源码 `_get_effective_op_func`（elementwise.py L1197–L1198：`fused_op(gate,value)=act(gate)*value`）+ `SiluAndMulFwdKernel.activation_func`（L1931–L1937：`g=T.Cast("float32",x)`；`g/(one+T.exp2(-g*log2e))`，`one=1.0f32`、`log2e=1.4426950408889634f32`）一致。dtype 语义「fp32 中间域」与源码 `T.Cast("float32", x)` 逐字一致；无规约语义与源码逐线程独立（L666–L670）一致；`x.shape[1]` 为偶数契约与 `_base.py` L917–L918（`x.shape[1] % 2 != 0` raise）一致；`.contiguous()`（`_base.py` L933）一致；输出 shape `(M,N)`/`out_idx=[1]` 与 `main(x,y)` 签名 L664 一致。§0.1 边界（`silu(-inf)=NaN`、大负 `g→0⁻`）经 verify_equiv 角点复核成立。
- **算法解读完整性 — pass**。§0.3 步骤 1–5 覆盖源码全部计算语句：步骤 1= L658 `block_N=threads*num_per_thread` + L665 grid `(ceildiv(N,block_N), M)`；步骤 2= L667 `col=(bx*threads_arg+i)*npt_arg+j`；步骤 3= L668–L669 `gate=x[by,col]` / `value=x[by,N+col]`；步骤 4= L1934–L1937 标量链；步骤 5= L670 `y[by,col]=op_func(...)`。无遗漏、无臆造。host 侧逻辑亦已列（L615 `lru_cache(32)`、L629 `out_idx=[1]`、L1123–L1127 `forward`、L1181/L1155 策略默认 `explicit_parallel`）。§0.4 顶注「无 SMEM tiling / swizzle / warp shuffle / 异步流水 / online / persistent」经源码通读成立（工厂无 `alloc_shared`、无 `T.Pipelined`、无 shuffle）。
- **优化手段覆盖完整性 — pass**。§0.4 九项逐项对得上源码：`explicit_parallel= T.Parallel(threads,npt)`（L666）、coalesced 连续段（L667 线程独占 npt 连续列）、`npt=8(fp16/bf16)/4(fp32)`（L1224–L1227 硬编码 128×8；L1228–L1229 + `_strategy_npt` L190–L191 返回 fp32 npt=4，设计「128/256、8/4」表述与源码 default_config 完全一致）、`T.Cast("float32")`（L1934）、`exp2` 降代换（L1937 + 注释 L1933「exp2 lowers to one MUFU.EX2 vs expf's multi-op」）、二维 grid（L665）、寄存器级标量无 SMEM（L666–L670）、工厂+lru_cache（L615）、策略机（L1150/L1155/L1202–L1218）。**未把实现层优化误判为算法层**，也未漏项。
- **耦合性判定合理性 — pass**。§0.5 逐条四态处置（保留 4 / 等价替换 3 / 重新设计 3 / 舍弃 2），抽检无矛盾：`threads=` kwarg 判「舍弃」有据（npuir `T.Kernel` 不接受 `threads=`，源 L665 `threads=threads_arg`）；`exp2` 判「等价替换（撤销）」有 §1.6.1 承接；`T.Parallel(threads,npt)` 逐元素标量链判「重新设计」有据（NPU 无逐元素标量快路径）；fp32 计算域判「保留（必需）」有 PL-1.17 双证实测（`pattern-library/elementwise.md` PL-1.17：bf16 `vexp/vmul/vadd` dtype 矩阵无 bf16、ssd 第二证 v7 fp16 化在 bf16 workload 编译失败）；`T.alloc_fragment`→`<zero>` 空间与 `T.alloc_shared`→`cbuf` 陷阱引用 `examples/clamp_max/_make_clamp_tensor_kernel/DESIGN.md` L813 实测（已核对原文一致）。舍弃项均写明理由。源码**未**使用的 SMEM tiling / swizzle / 异步流水未臆造。
- **NPU 重设计可行性 — pass**。R1/R2/R3 三要素（源方案→NPU 新算法→语义保持论证）齐全：R1「逐元素标量链→UB 内 v-prefix 链」的等价性由「向量 op 逐 lane 独立、无累加无跨 lane 依赖」论证 + `verify_equiv.py` 15/15 机器验证支撑；R2「二维 grid→一维 `T.Kernel(num_kernels,is_npu=True)` + `(row,chunk)` raster 展平 + persistent」满足 ascend-constraints 一维 Kernel 约束（`examples/clamp_max/.../DESIGN.md` §3.5.1 同款已核）；R3「`op_func` 标量闭包→内联 v-prefix 链、工厂签名保留」与 wrapper 契约一致（`silu_and_mul.py` L121–L126 传 `(M,N,dtype,op_func)`、L140–L143 `fused_op=activation_func(gate)*value`，设计内联即该复合的向量化，风险已登记 §9.1 R7）。恒等变形式与源式逐位一致由机器验证（角点含 ±Inf/NaN）。
- **设计与决策一致性 — pass**。§1–§7 描述的确为 §0.5/§0.6 决策后的 NPU 算法：§3.3 伪代码无 `T.Parallel`/`threads=`、用 `T.alloc_ub`+v-prefix；§5.5 一维 persistent 分核；§6 核内仅 `T.serial`。未出现「§0 判重设计、§3 照抄源方案」的脱节。源手段 5（`exp2` 降代换）被撤销时在 §0.6 末尾总表**逐项登记**（非静默丢弃）。
- **golden 独立性 — pass**。§8.1 golden `(F.silu(x.float()[:, :N]) * x.float()[:, N:]).to(x.dtype)` 与源仓参考实现 `examples/TileOPs/tests/ops/test_silu_and_mul.py` L49–L53 `SiluAndMulTest.ref_program` **逐字一致**，以 §0.1 语义为唯一依据，**不复刻** §0.6 的 NPU 分核/尾块/`vexp` 链——精度对比不构成自证。

### 1. API 可行性 — pass

§3.2/§3.4 全部 API 均有 `docs/Tilelang.language/` 佐证 + 仓内先例（逐条亲自核对）：

- `T.vmul` 标量 src2（`T.vmul.md` §2.3 第 4 条「仅支持 src2 为标量」；fp16/fp32 √、bf16 ×）——仓内标量用法先例 `examples/deepseek_v4/tilelang_sparse_mla_bwd.py` L266/L270、`testing/npuir/active_ops/test_hardswish.py` L31。
- `T.vadd` 标量 src1（`T.vadd.md` §2.3 第 4 条）——先例 `examples/flash_qla/tilelang_get_warmup_chunks.py` L98、`examples/ge/ge_fwd_kernel/ge_fwd_kernel.py` L506。
- `T.vexp`（`T.vexp.md` §2.2.1 fp16/fp32 √、bf16 ×；§3 → `hivm.hir.vexp`）同形/原地用法先例 `examples/cumprod/_cumulative_kernel/_cumulative_kernel.py` L199/L268、`examples/flash_attention/flash_attn_npuir.py` L215。
- `T.vdiv`（`T.vdiv.md` §2.4 + 原地支持）——先例 `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/perf_opt/mish.py` L162。
- `T.vcast(..., round_mode="rint")`（`T.vcast.md` §2.2.1：f16/bf16→f32 `rint`；f32→f16/bf16 `rint`）——先例 `examples/ge/ge_fwd_kernel/ge_fwd_kernel.py` L486/L512、`examples/flash_qla/tilelang_correct_h0.py` L94。
- `T.copy` 切片 + runtime extents（`T.copy.md` §2.1 extent 推断规则、§2.2.2 第 2/4 条、§2.4 示例 1/3）——先例 `examples/elementwise/vec_add_1d.py` L34–L38、`testing/autotune/SiluAndMul/silu_and_mul_2d.py` L114–L120。
- `T.alloc_ub` / `T.min` / `T.ceildiv` / `T.Kernel(is_npu=True)` / `T.serial`+`task=i·K+cid`+`if`（`T.vcast.md` §2.4 示例 2 persistent 分发形态）均有文档与先例。
- **v-prefix 优先**：设计全部用 v-prefix（`vmul/vexp/vadd/vdiv/vcast`），无 `npuir_xxx` 混用，符合 AGENTS.md。
- **模式匹配**：Developer 模式 + 自动同步（§7），无手动 `sync_block_set/wait`，与 §2 选型一致。
- 弃用/未采纳 API 主张亦经核对：`vneg` 确认不在 v-prefix 导出清单（`tilelang/language/__init__.py` 无 `npuir_neg as vneg`）；`vrec` 确认无 `docs/` 条目、仅 `customize_npuir.py` L386 导出（L155–L156 `npuir_rec as vrec`）。最近邻先例 `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py`（L38–L80）与本设计结构完全同构（`block_size` 折叠、fp32 中转 `vcast(rint)`、`T.min` 尾块、Python 级 dtype 分支、v-prefix 链），实证可降低 Stage 3 风险。

### 2. 内存层级规划 — pass

- **搬运路径完整**：§4.4 仅 GM→UB→GM 两段（`T.copy` 载入 → v-prefix 链 → `T.copy` 写回），无 GM→L0 越级、无 L1/L0 参与（纯 Vector）。与 §1.5/§3.3 一致。
- **UB 预算**：§4.5 逐 dtype 给出 `block_M·block_N=2048` 元素的预算（fp16 40KB、bf16 48KB、fp32 24KB 手工；×1.7 后 68/82/41KB），目标 192KB 余量充裕。人工复核（含 doc 未单列 `d32` 的最坏口径）仍 ≤82KB < 192KB → 结论稳健。膨胀系数引用 `CONST-capacity-910B2C`（192KB/AIV、auto-multi-buffer ~1.7x），设计明确取保守上界并记录了「显式 alloc ~1.12x / 反例 0.981x」结构依赖——引用与实测条目一致（已核对 constants.md L20–L40）。
- **L0C 容量**：纯 Vector 无 L0C（机检 0 KB vs 128KB）。
- **Buffer 层级标注**：§4.3/§4.5 全部 buffer（`g_ub/v_ub/g32/v32/p32/t32/d32/y_ub`）均标 UB。
- 说明：§4.5 的 fp16 行未把 `d32`（4 B/elem）计入、fp32 行未把 `t32/d32` 计入（§4.3 注已声明 `t32/d32` 可合并、`y_ub=p32`），属口径略保守偏松的记录，但余量 2–5 倍，不影响结论（建议级，不阻塞）。

### 3. Tiling 策略 — pass

- **Block 划分明确**：§5.2 `block_N=block_size=1024`（wrapper 默认，`silu_and_mul.py` L67 `_BLOCK_SIZE=1024`、L146–L148 `default_config={"block_size":1024}`）、`block_M=2`，附 5 条理由（对齐/整除/UB/发射/契约冻结）。
- **对齐约束**：1024×{2,4}B=2KB/4KB，均为 32B 整倍数；向量宽度整除（fp16/bf16 ×8、fp32 ×4）→ 尾 lane 浪费 0%（末 chunk 垃圾不落 GM）。
- **非整除处理**：§5.4/§6.4 `m_tail=T.min(block_M, M−m0)`、`n_tail=T.min(block_N, N−n0)` 截断，不 host padding（合规 ascend-constraints §4）。显式覆盖 `M=1`（decode）、`N<block_N`、`M<block_M` 情形。
- **分核策略三要素（参照 core-split-strategy.md §1）**：
  - ① 逻辑核数计算：§5.5 表给出 `num_logical = ceildiv(M,block_M)·ceildiv(N,block_N)` 四 workload 值——**独立复算一致**（512 / 14336 / 14 / 4096）。
  - ② 物理核数依据（实查）：§5.5 记录查询代码 `NPUUtils.get().get_aicore_num()` / `get_aivector_core_num()` 与返回值 **20 / 40（2026-10-14）**。本人**实跑同一命令复核**：`python3 -c "from tilelang.utils.npu_utils import NPUUtils; ..."` 返回 `aicore 20 / aivector 40`，与设计记录**一致**；且设计正确指出与 `CONST-aicore-910B2C`（24 AICore）不同源并按实查值取 40、流量项按 40/48 折算（constants.md L158–L168 已核对）。非文档假设值。
  - ③ 规模判定与分核方案：§5.5 逐 workload 给规模判定 + `num_kernels=min(num_logical,40)` + `num_local`（编译期静态，13/359/1/103，独立复算一致）；核间负载差 ≤1；decode `num_logical=14 ≤ 40` 走「无需适配」分支说明清楚。
- **核数与 block 取值自洽**：逻辑核远超物理核时有核内 `T.serial` 串行设计（prefill/ffn/smoke）；未出现超发不处理；核内串行任务数为编译期常量、不依赖动态 shape。

### 4. 技术约束检测 — pass

§3.5.1 逐条核对本项目已知限制：不涉及三维 Kernel（源为二维 grid、NPU 一维）；`threads=`/`T.Parallel(threads,npt)` GPU 专用 API 已由 §0.5/§0.6 取代；无 GEMM（故无 GEMM 非整除、无 L0C 溢出）；物理核数限制已由 §5.5 处理；host 侧仅 `.contiguous()`（纯视图）；无 bool/uint8；bf16 无原生向量算术已由 fp32 中转处理。均与「不涉及/已处理」结论一致。

### 5. 循环与同步 — pass

- **循环结构**：§6.1/§6.2 明确「核间 `T.Kernel(num_kernels,is_npu=True)` / 核内 `T.serial(num_local)`（静态边界）/ tile 内逐元素**无循环**（v-prefix 覆盖）/ 尾块 `if task<num_logical` + `T.min`」，与 §3.3 伪代码一致，且与 §1.6.2「kernel 内无任何逐元素标量循环」交叉一致。
- **同步与模式匹配**：§7 与 Developer 模式一致（编译器自动插入 copy↔compute↔copy 依赖同步），无手动 `sync_block_set/wait`，非 MixCV。
- **尾块处理**：§6.4 说明 `m_tail`/`n_tail` 截断 + 整 buffer 计算仅截断写回（`vec_add_1d.py` 已验证同款）。
- 说明（建议级，不阻塞）：§3.3 注记④称 `t32` 承担「`-g` → `exp` → `+1` 三步原地复用」，但伪代码中 `vadd(t32,1.0,d32)` 写入 `d32`（`d32` 独立分配，§4.3 注已说明可合并）——注记与代码的表述不一致，属文字层面小瑕疵，不影响正确性（两个 buffer 均已分配，无活区间冲突）。

### 6. 验证方案 — pass

- **Golden 函数**：§8.1 给出 PyTorch 参考实现（`F.silu(gate_f32)*value_f32` 后回舍），与源仓 `ref_program`（test L49–L53）逐字一致，独立性见维度 0。
- **L0 测试计划**：§8.3 L0-1–L0-9 覆盖四个 manifest workload × dtype（含 **bf16 强制**）、非整除（`[7,2050]`/`[3,2]`）、角点（NaN/±Inf/±0/大负 g/subnormal）、Op 层 I/O 契约、roofline、奇数列报错——具体 shape/dtype 均给出。
- **精度标准**：§8.2 fp16 1e-2/1e-2、bf16 1.6e-2/1.6e-2、fp32 1e-5/1e-5、`equal_nan=True`——与 `examples/TileOPs/tests/ops/test_silu_and_mul.py` L34–L39 `_get_tolerances` **逐值一致**（已核对），符合「fp16 默认 1e-2、fp32 默认 1e-4」及算子族更严口径。

### 7. 完整性与一致性 — pass

- **章节齐全**：§0（0.1–0.7 小节全）+ §1–§11 齐全，无缺章。
- **无占位符**：全文机检 0 命中模板占位标记（花括号占位、未完成标记、待补写标记）。
- **同类引用真实**：§3.5.3/§9 引用的 11 条路径全部存在（`examples/elementwise/vec_add_1d.py`、`examples/clamp_max/.../DESIGN.md`、`examples/ge/ge_fwd_kernel/DESIGN.md`、`examples/deepseek_v4/engram/engram_gate_kernel_fwd.py`、`testing/autotune/SiluAndMul/silu_and_mul_2d.py`、`testing/autotune/Silu/silu_2d.py`、`testing/npuir/arith_ops/test_sigmoid.py`、`.../elementwise/mish/mish_kernel/`、`tests/ops/test_silu_and_mul.py`、`benchmarks/ops/bench_silu_and_mul.py`、`examples/elementwise/example_elementwise_add.py`）——逐一 Glob/ls 验证 OK。引用行号抽检亦对（如 `vec_add_1d.py` L33 `T.min`、`clamp_max` §9.2 = L813、wrapper `silu_and_mul.py` L121/L146/L159）。
- **内部一致**：§3.1 公式拆解 = §1.6.1 优化后公式；§3.2/§3.3/§4.3/§5.2/§6 的 buffer 形状与 tile 取值一致；§5.5 核数与 §6.1 循环一致；§8.2 容差与源测试一致——无相互矛盾。

### 8. 算法优化分析 — warn

> 判定说明：本章为**建议级（warn）**——§1.6 结构、调研四问、等价性机器验证与独立推演、向量化覆盖、布局决策**全部通过**；唯一扣分是 §1.6.1 一处论证前提与所引 API doc 的 lowering 一节矛盾（结论与选定算法不受影响），故非阻塞。

- **§1.6 章节存在性 — pass**：§1.6.0（算法调研）/§1.6.1（数学等价优化）/§1.6.2（向量化替代）/§1.6.3（向量化轴与布局）四子节齐全。
- **算法调研存在性与完整性 — pass**：R1 候选表 9 行**含基线（#0 源算法）**且逐行给等价性初判/收益方向/是否纳入 R3；R2 明确结论「无在线变体（天然单遍）」并给结构依据（`y[m,n]` 仅依赖 `(x[m,n],x[m,N+n])`，无 running 统计量）——与源算法亦为单遍（§0.4）不矛盾；R3 覆盖四口径（FLOPs/向量 op 数、访存 Bytes、扫描遍数、中间缓冲峰值）+ 可并行度，基线在表、口径写清（`M·N` 为输出元素数、`E=elem_bytes`、sub-fp32 的 `vcast` 另计）；R4 逐候选评估六项亲和项且负向淘汰附 `docs/`（`T.sigmoid.md` §3、`T.vexp.md` §2.2.1）与 `testing/autotune/SiluAndMul/silu_and_mul_2d.py` 先例。深度**超出**单步逐元素族「四问各一行」的最低要求。
- **调研结论正确性（独立复核）— pass**：
  - ① 等价性独立推演：`exp2(x·log2e) ≡ exp(x)` 为实数恒等（`log2e=1/ln2`）；`(g/(1+e^{-g}))·v ≡ (g·v)/(1+e^{-g})` 为乘法结合律恒等——两项推演成立；容差/舍入影响由 `verify_equiv.py` 15/15 `EQUIV_PASS`（fp64 参照，回舍后逐位一致，NaN/±Inf 模式匹配）实证，非纸面自述。
  - ② 复杂度算术独立复算：候选 1 fp32 路 = `vmul(-1)+vexp+vadd+vmul+vdiv = 5` 向量 op ✓；sub-fp32 路 +3 `vcast` = 8 ✓；访存 `3MN·E` = 读 gate（MN·E）+ 读 value（MN·E）+ 写 y（MN·E）✓，**无中间缓冲 GM 往返**（全 UB，已核对 §3.3/§4.4）✓；扫描遍数 1 ✓；缓冲峰值 O(block_M·block_N) ✓。候选 2 的 `vsigmoid` = 4 hivm op（`T.sigmoid.md` §3 逐字 `VAdd+VSub+VExp+VDiv`）+2 `vmul` = 6 ✓。
  - ③ 亲和性判断与实测条目不矛盾：bf16 无原生算术、fp16 需 fp32 golden → PL-1.17（已核对原文：bf16 中转 24B/elem、fp16 20B/elem、ssd v7 编译失败实证）一致；容量引用 `CONST-capacity-910B2C`、发射引用 `CONST-vector-launch-overhead`、重叠引用 `CONST-copy-floor-method`——均存在且引文准确。
  - ④ 负向断言主动复核：「无在线变体」「无更低复杂度算法」「无更优化简公式」逐条对照 `algorithm-candidates.md` ALG-elementwise 命中行（L98–L104：R1 CSE/除法乘倒数/cast 链合并；R2 逐元素天然单遍；R3 融合消中间 GM；R4 单遍 Vector+dtype 路径）——设计的 R1（CSE 无收益、`vrec+vmul` 2 op 劣于 `vdiv` 1 op、`exp2` 降代换撤销）、R2（单遍）、R3（无中间 GM）、R4（fp32 路径）**逐项已评估**，未见漏掉的已知候选；「op 数下界 5」经独立推演成立（最短链 1 `vexp`+1 `vadd`+1 `vdiv`+2 `vmul`；`vneg` 原语不存在已亲验 → 取负须占 1 `vmul`；`vrec` 路 6 op 劣）。恒等变形空间（`vsigmoid`/正指数形/tanh 形/倒数形/近似）已穷尽且经 op 数裁决。
  - ⑤ 并列候选裁决路径：候选 3/4 与候选 1 持平 → 以「与 golden 精度对齐更远」/「无收益」否决并给出依据；§1.6.3 主选 #1 vs 备选 #2 交 Stage 4 实验裁决（阈值 0.97）。
  - 互联网检索未执行且**如实记录原因**（`webfetch` 未验证可用 + 本地恒等变形已穷尽），未编造来源——按证据规则合规。
- **调研→设计一致性 — pass**：§1.6.0 选定候选 1（exp 直式 + 分子合并，fp32 域）实际驱动 §1.4（算法描述即单遍 v-prefix silu-and-mul 链）、§3.1（公式拆解以 `y=(g·v)/(1+exp(-g))` 为唯一输入，§3.1 顶注明写）、§6（核内无逐元素标量循环）；迁移任务 §0.5/§0.6 处置与调研选族一致（`exp2` 撤销、逐元素链重设计），源优化手段意图承接在 §0.6 末尾总表**逐项登记**（无静默丢弃）。
- **数学等价优化完整性 — pass**：§1.6.1 逐类别核查 7 类（恒等变形/稳定化/CSE/降代换/访存/归约/分子合并）并采纳 1 项，四要素齐全（原式 → 优化后公式 → 等价性论证 → 收益量化）；无优化空间的部分（CSE/归约/多轮扫描）给出显式结论与依据。
- **等价性论证正确性 — pass（附建议级观察）**：恒等式数学成立（如上）；容差内分析正确（`log2e` 舍入低于目标 dtype 精度 → 回舍后逐位一致，与 §8.2 容差一致）；机器验证独立复跑一致。
  - **建议级观察（唯一扣分点）**：§1.6.1 Item 1 / §1.6.0 R4 / §0.6 均以「NPU 上 `vexp` 与 `vexp2` **同为单向量指令**，故 `exp2` 降代换无收益」为前提，并引 `T.vexp2.md §3 → hivm.hir.vexp2`。**亲自打开该 doc 核对**：`docs/Tilelang.language/数学操作/T.vexp2.md` §1 明写「由于底层不支持硬件级别的 exp2 操作，实际上使用的是 B=exp(A·ln2)，其中 tmpBuffer 用来存储 A·ln2」；§3 逐字为「`tilelang::vexp2Op` 将被转换为 `hivm::VMulOp` 和 `hivm::VExpOp`」——即 NPU `vexp2` = **2 条向量 op + tmpBuffer**，而 `T.vexp.md` §3 = `hivm.hir.vexp`（1 条）。故前提与所引 doc 矛盾，且 R3 基线行把 `exp2` 计为 1 op（8 向量 op/tile）会低估为 9 op。**影响评估**：结论方向不变且更强——撤销降代换后选 `vexp` 直式，正是 NPU 上更短的路（少 1 `vmul`+tmpBuffer），选定算法与公式无需改动；仅 §1.6.1「无 op 数净收益」的**收益量化偏保守**、`T.vexp2.md §3` 的引用 lowering 名有误。属非阻塞（不改变算法选择、不影响等价性、不致 Stage 3 失败）。
- **向量化替代完整性 — pass**：§1.6.2 盘点 §0.6 重设计后全部 12 个循环/标量点（#1–#12），逐元素计算点 #1–#8 全部替代为向量 op，保留 4 处（#9 block 级任务索引、#10 动态尾块 `T.min`、#11 tile 级 `T.serial` 调度、#12 host 元数据）均明确为非逐元素热点。**与 §3.3 伪代码、§6 循环结构交叉核对**：伪代码中确无任何逐元素标量循环、全部走 v-prefix/`T.copy`，与结论一致（已逐行核对 §3.3 L434–L501）。
- **不可替代理由充分性 — pass**：4 处保留点理由具体到本算子（block 索引无逐元素等价 API、尾块依赖动态 shape、persistent 需 tile 级循环、host 元数据不在 kernel 内）——非「实现简单」「照源码写」式泛论。
- **替代 API 可行性 — pass**：替代方案 API（`T.copy`/`T.vcast`/`T.vmul`/`T.vexp`/`T.vadd`/`T.vdiv`）均在 `docs/` + `examples/` 有佐证（同维度 1），优先 v-prefix。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 候选矩阵 6 行（I/O 原生 packed + 2-D tile（主选）/ `block_M=1` 逐行 chunk（备选）/ host 预拆 / 核内 transpose·deinterleave / 多维展开 / 逐元素标量直译），逐候选量化评分（整除性/尾 lane 浪费/累加链/repack 代价/UB）；弃选方案有量化理由且**前提经 API 文档核对**——`T.transpose` 弃选理由（`VTransposeOp` 开销、permutation 限两轴交换、`size` 当前不可用）与 `docs/Tilelang.language/创建操作/T.transpose.md` §2.1/§2.3 一致；host 预拆弃选给出 `+2MN·E` 数量级（prefill +117MB）并引 ascend-constraints §4（host 禁改输入真实内容）；明确区分「I/O layout 是契约、核内布局是设计变量」，未把 I/O 契约当内部布局依据；**源码轴对照**（源 `col=(bx·threads+i)·npt+j` 的隐式轴向 = 输出 N 轴 + 每线程连续元素）已写明，符合迁移要求。**实验裁决模式三件套齐全**：主选 #1 结构完整；备选 #2 给出 buffer 形状 `(1,block_N)`、`T.copy` 单段载入、按 `block_M=1` 粒度重算的分核两套核数与 UB ≈34KB（达 Stage 4 可直接实现深度）；裁决计划含代表 shape、指标（Task Duration/vec_ratio/mte2_active_bw）、阈值（备选 < 主选×0.97 才翻转）、回写路径；并评估了按 shape 分派两方案的可行性。判定裕度依赖的 MTE2 跨行 stride 常数为未实证量——已按模式要求给备选+裁决，未纸面单选。pattern-library 查证：`layout.md`/`traps-*.md` 未见与该弃选论断矛盾的实测条目。
- **布局决策与实现一致 — pass**：§1.6.3 选定「输出 N 轴（段内 stride=1）+ raster 2-D tile `(block_M,block_N)`」与 §3.3 伪代码 buffer 形状 `(block_M, block_N)`、§4.3/§4.5 预算、§6 `T.serial` tile 级循环三方一致；无 repack（gate/value 本非交织、非降采样），故无重排路径需在 §3.3 落地。
- **与下游章节一致 — pass**：§3.1 以 §1.6.1 优化后公式为输入（§3.1 L397 顶注明写）；§6 循环结构与 §1.6.2 结论一致（判定向量化的点无对应逐元素标量循环）。
