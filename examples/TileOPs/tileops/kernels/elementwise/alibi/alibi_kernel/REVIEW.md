# _make_alibi_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/alibi/_make_alibi_kernel/DESIGN.md
- 任务类型: 迁移（migration-harness，生成式 zero-tensor-input 算子）
- 源算子: /home/developer/TileOPs/tileops/kernels/elementwise.py（工厂 `_make_alibi_kernel` L3557–L3591；类 `AlibiFwdKernel` L3594–L3642）
- 规格来源: /home/developer/TileOPs/tileops/manifest/elementwise_generative.yaml `AlibiFwdOp`（harness 副本 examples/TileOPs/tileops/manifest/elementwise.yaml L307–L360）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0

> 本次为 rev2（retry_count=2）独立复核。§11.1 声明「已修正上一轮唯一阻塞项 §1.6.3 候选 #2 的 `T.transpose` dtype 负向前提」——本检视**亲自打开** `docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1 逐格取值核对，确认该阻塞项**确已消除**（详见维度 8）。

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: PASS。`python3 examples/alibi/_make_alibi_kernel/verify_equiv.py` 实跑输出 **46 项 `EQUIV_PASS` / 0 项 `EQUIV_FAIL`**，与 DESIGN.md §1.6.1 内嵌结果表**逐项一致**：
  - 行组 #2 int16-ramp（fp32/fp16/bf16）max_abs 0.000e+00、0 违反；
  - 行组 #1 CSE/D-shared（0.0, 0）；
  - 行组 #3 neg-fold（0.0, 0）；
  - 行组 #4 exp2-vs-pow（0.0, 0）；
  - 行组 #5 full-cand vs golden（0.0, 0）；
  - 行组 #6 baseline-divmod vs golden（fp32 S=256/H=32 = 1.526e-05 < 容差，余 0.0；与内嵌表注一致）；
  - 行组 #7 divmod-roundtrip（exact=True, 0）；
  - 行组 #8 int16-range S=4096（0.0, 0）；
  - 行组 #5 corner diag/h-extreme（0.0, 0）；`corner diag zero: True`。
  脚本落盘、覆盖 `S∈{128,256}`×`H∈{8,32}`×`{fp32,fp16,bf16}` + S=4096 int16 值域 + 角点，判定准则 = Stage-3 门禁同口径（`|cand-base| ≤ atol + rtol·|base|`，fp32 1e-5 / fp16 1e-2 / bf16 1.6e-2）。**机器等价验证有效，与 §1.6.1 自述一致。**
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/alibi/_make_alibi_kernel/DESIGN.md` → JSON `checks` 摘要：
  - `ub_budget`: **skip** —— "no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"。**人工复核结论（机械跳过，人工已核）**：§4.3 六个 UB buffer（R/C int16 各 2 B、Rf/Cf/O fp32 各 4 B、O16 dtype 2 B）；按 BishengIR 生命期复用峰值 = 距离链 `R+C+Rf+Cf` = 12 B/elem，`bi·bj=8192` → 96 KB < 192 KB/AIV（设备报文 `1572864 bits`）；16384 elems → 192 KB 恰贴上限。**预算成立，无 VP-2026-0009 混合字节口径错误。**
  - `l0c_budget`: **skip** —— "no block_M x block_N values parsed"。**人工复核结论**：纯 Vector 生成式算子，无 Cube/L0C，**不适用**。
  - `core_split`: **skip** —— "insufficient parsed dims {}; declared logical cores = None"。**人工复核结论（机械跳过，人工已核）**：§5.5 三要素齐全；逻辑核数 test 2 / manifest 512–2048，物理 Vector 核 `NPUUtils.get().get_aivector_core_num()` **本会话实查 = 40**（`get_aicore_num()` 实查 = 20，aivector = 20×2 = 40，与 §5.5 ② 记录一致）；极大规模处置（固定 40 核 + `T.serial` 网格跨步 + 静态上界 + 显式残留负载不均 +1.3%~+1.6%）正确。
  - `r3_metrics`: **pass** —— "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"。
- 结论：机械可检项与独立人工复核互补，无冲突；r3_metrics 机检 pass，ub/l0c/core_split 三项 skip 均已完成人工核对。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

亲自 Read 源算子 `/home/developer/TileOPs/tileops/kernels/elementwise.py`（L3557–L3642）、Op 层 `ops/elementwise/alibi.py`、test `test_special_elementwise.py::test_alibi`（L225–L249）与 manifest，逐项核对 §0：

- **语义理解正确性（pass）**：§0.1 数学语义 `bias[h,i,j]=-slope_h·|i-j|`、`slope_h=2^(-8(h+1)/H)` 与源码实现**一致**——源码 L3582 `exp_val = -8.0*(h+1)/num_heads`（用 `h+1`），golden L242–L245 `torch.pow(2, -8*arange(1,H+1)/H)`（亦 `h+1`）。§0.1 **显式标注关键陷阱**：工厂 docstring（L3561）写 `2^(-8h/H)` 与实现/类 docstring（L3598 `h+1`）/golden 矛盾，设计以「实现+golden」为准。此陷阱识别**正确且必要**，是语义理解到位的核心证据。dtype 语义（`T.cast(row-col,"f32")`、fp32 exp2、末端 `T.Cast(dtype,·)` 单次舍回）与 §0.1 一致；规约语义（无规约）；边界语义（对角精确 0）；host 侧（Op.forward `reshape(H,S,S)`）。§0.2 I/O：源码 `main(out: T.Tensor((N_total,), dtype))` 1-D 唯一输出、`reshape(H,S,S)`、`lru_cache` 特化——与 §0.2 一致。输出 shape `(H,S,S)` 与 manifest `outputs.output.shape` 一致。
- **算法解读完整性（pass）**：§0.3 五步分解覆盖源码全部计算语句——L3575 `flat=(bx*threads+i)*npt+j`；L3577–3580 `h=flat//S²`、`rem=%S²`、`row=rem//S`、`col=rem%S`；L3582–3583 `exp_val`/`slope=T.exp2`；L3584 `dist=cast(row-col)`；L3586 `abs_dist=if_then_else(dist>0,dist,-dist)`；L3587 `out[flat]=T.Cast(dtype,-slope*abs_dist)`。无遗漏、无臆造。数据流图覆盖 `grid×threads×npt→GM[out]` 全 buffer（无 SMEM）；host 侧 `lru_cache`/`reshape` 已列出。flat 语义（`flat=bx·block+(i·npt+j)`，块内连续）覆盖正确。
- **优化手段覆盖完整性（pass）**：§0.4 列出源码 8 项可识别手段（一维 grid+大线程数/coalesced 写、flat div/mod、slope=f(h) exp2、fp32+单次舍回、`if flat<N` 尾守卫、`if_then_else` abs、`lru_cache` 常量特化、无 autotune），逐项标注硬件依赖与初判；未把实现层/算法层误判（#2 标为「实现层」、#3 标为「算法层」，合理）。并显式声明「识别不出来≠不存在」，指明源**未做** head 无关 `|i-j|` 复用（为 §0.6/§1.6.1 新增优化留出空间）。
- **耦合性判定合理性（pass）**：§0.5 每条目四态处置（保留/等价替换/重新设计/舍弃）+ 依据。关键条目（计算语义/算法「保留」；#2 div/mod「重新设计（性能依据，非能力缺失）」；#1 并行「重新设计」；#3/#4 保留/等价替换；#5/#6 等价替换；#7/#8 保留；`out_idx`「重新设计」；`T.Parallel` 结构「重新设计」；fp8「舍弃」）与 GPU→NPU 能力映射一致。两条 v0 错误负向断言（「NPU 无向量 div/mod」「`T.Parallel` 写 UB 不可用」）**已按目标设备实测更正**（本检视独立复核：`tilelang/language/customize_npuir.py:158 npuir_div`→`vdiv`、`:425 npuir_floordiv`→`vfloordiv`、`__init__.py:83/98/99/154/167` 导出；`testing/npuir/parallel_ops/test_floordiv_floormod_parallel.py` Developer `T.Parallel`+`int64 //,%` 存在；`docs/.../T.Parallel.md` §2.3–2.4 官方写 UB 示例）。舍弃项 fp8 理由（`SUPPORTED_DTYPES=_FLOAT_DTYPES` 仅 fp16/bf16/fp32，实测 `_FLOAT_DTYPES` L154–L158）成立。
- **NPU 重设计可行性（pass）**：§0.6 R1–R5 每项有「源方案 → NPU 新算法 → 语义保持论证」三要素。R1（arange ramp + vsub/vabs + D head 共享）、R2（一维 `T.Kernel(40,is_npu=True)` + `T.serial` 网格跨步）、R3（每 head 标量 `T.exp2`）、R4（`vcast(rint)` 单次舍回）、R5（显式输出参数）均满足 NPU 硬件约束（一维 Kernel、UB 层级、无分形/对齐问题）。语义保持论证均落到数学恒等/整数精确/单次舍回，且设备 Q1/Q2 端到端 0 违反佐证。**唯一能力性否决 R5**（`out_idx`+arange 崩溃）有 CG-2026-0016 设备实证（本检视核对 capability-gaps.md L712–L746 存在且描述一致）。
- **设计与决策一致性（pass）**：§1.4 算法描述、§3.1 公式拆解、§3.3 伪代码、§5 Tiling、§6 循环均为 §0.5/§0.6 决策后的 NPU 算法（arange ramp、D 共享、标量 exp2、显式输出、40 核网格跨步），无「§0 判重设计、§3 照抄源方案」的脱节。§3.5.2 逐项列出源/本项目差异并回指 §0.5/§0.6。
- **golden 独立性（pass）**：§8.1 golden 以 §0.1 语义为唯一依据（移植 `test_alibi` 的 `torch.pow(2,·)`+`|i-j|`+`.to(dtype)`），显式声明「不复刻 §0.6 的 arange/D-shared 结构」。golden 与 §0.6 的 NPU 数据流（arange ramp、head 维复用）结构不同，验证独立成立。

### 1. API 可行性 — pass

§3.2 API 映射逐条核对存在性佐证（本检视实读源码/文档）：
- `T.npuir_arange(dst,strides,offset)`：`tilelang/language/customize_npuir.py:1370` 定义、`__init__.py:167` 导出；`docs/.../创建操作/T.arange.md` §2.2.1（int16/int32/int64/fp16/fp32 √）；`testing/npuir/creation_ops/test_arange.py`（`T.npuir_arange(a,[bx,by],bx)` + 显式输出）。strides 语义（元素 `[i,j]=offset+i·s0+j·s1`）经 `_gqa_prefill_fwd_kernel.py` L423–424（`[0,1]`→局部 j、`[1,0]`→局部 i）核对正确。
- `T.vcast`（`__init__.py:123`）/`T.vsub`（:75）/`T.vabs`（:154）/`T.vmul`（:81）：均导出；dtype 矩阵核对见维度 8。
- 标量 `T.exp2`：`tilelang/language/tir/op.py`（§3.4 引 L1837）；源实现原式。
- `T.Kernel(n,is_npu=True)`+`T.serial`：`examples/elementwise/vec_add_2d.py` L28–L42（实读确认 `T.Kernel(BLOCK_SIZE, is_npu=True)` + `T.serial(ceildiv(...))` + `if block_id<...`）。
- `T.copy(UB→GM 区域)`：test_arange.py、vec_add_1d.py。
- **模式匹配**：Developer 模式用自动同步（§7.1），无手动 set/wait_flag，一致。**v-prefix 优先**：全部使用 v 前缀（vcast/vsub/vabs/vmul）+ `T.npuir_arange`，符合 AGENTS.md API Convention。

### 2. 内存层级规划 — pass

§4.4 搬运路径完整：GM 无读 → UB（arange/vcast/vsub/vabs/vmul）→ `T.copy` 写 GM；**无 GM→L0 直连**等越级路径。§4.3 六 buffer 全部标注存储层级（UB）。§4.5 UB 预算 = 生命期峰值 12 B/elem，`bi·bj=8192` → 96 KB < 192 KB（A2/A3 容量口径与设备报文 `1572864 bits` 实证一致），16384 → 192 KB 恰贴上限；`24576` 设备实测溢出。**机械跳过（ub_budget skip），人工已核通过**。L0C 不适用（无 Cube）。device_probe_alibi_ub_ceiling.py 落盘可复现（本检视读取确认脚本存在且 sweep 8192/12288/16384/24576，docstring 记录报文与结论一致）。

### 3. Tiling 策略 — pass

- **Block 划分明确**：§5.2 给出 `bj=S`（优先全宽）、`bi=max(2, BUDGET//bj)`、`BUDGET=8192`，并给出三 workload 推荐取值（128→bi=64；2048→bi=4；4096→bi=2）与理由（写出连续 + `bi≥2` 设备约束 + UB 预算 + 发射项权衡）。
- **对齐约束**：§5.3 `bj·elem_bytes` 为 32B 整数倍（bj≥128）、`bj%8==0` → 尾 lane 浪费 0%。
- **非整除处理**：§5.4 给出 S 编译期因子选择策略 + 通用 S 尾块守卫（`T.min` 裁剪 `T.copy`），本任务 S∈{128,2048,4096} 全整除；明确标注通用 S 尾守卫为 Stage 3 待实现。
- **分核策略三要素（核心核对，pass）**：① 逻辑核数计算 `ceil(S/bi)×ceil(S/bj)`（2/512/2048）；② 物理核数依据**实查**——§5.5 ② 记录 `NPUUtils.get().get_aicore_num()` 与 `get_aivector_core_num()` 的查询代码与返回值；**本检视独立实跑**：`aicore=20`、`aivector=40`，与设计记录**完全一致**（纯 Vector 翻倍语义正确，未用文档假设/经验值）；③ 规模判定——test（2≤40）「无需适配」，manifest（512/2048 ≫ 40）按「极大规模」固定 40 核 + `T.serial` 网格跨步 + **静态上界**（`T.ceildiv(n_tiles,40)`）+ 显式残留负载不均（+1.3%/+1.6%），符合 `core-split-strategy.md` §1 要素③（本检视核对标准文件 §1/§2.2）。核内串行上界不依赖动态 shape。**机械跳过（core_split skip），人工已核通过**。

### 4. 技术约束检测 — pass

§3.5.1 逐项检查：**不使用三维 Kernel**（一维 `T.Kernel(40,is_npu=True)`，与 T.Kernel.md「NPU 最多 2 维 block」一致）；非 GEMM（无 L0C 溢出）；物理核数限制已由 §5.5 网格跨步处理；`out_idx`+arange 崩溃已改显式输出；`arange` tile 首维 ≥2（CG-2026-0017）；`vexp2` 不用；3-D v-op 广播不支持（head 用 `T.serial`）；`vabs` 无 int16（距离在 fp32 域）；条件 `T.alloc` 非法（buffer 顶层无条件分配）；coreDim ≤ 65535（仅基线涉及）。均给出处理方案，无遗留阻塞。

### 5. 循环与同步 — pass（建议级维度）

§6.1 明确循环类型：核级 `T.Kernel(40)` 并行、tile 级 `T.serial(ceildiv(n_tiles,40))`、head 级 `T.serial(H)`、元素级向量化（无 `T.Parallel` 标量循环）。§6.2 回指 §3.3 伪代码。§6.3 说明不使用 `T.Pipelined`（write-only 单通路，跨 tile 重叠由 auto-multi-buffer 处理）。§6.4 尾块处理回指 §5.4。§7 同步策略（Developer 自动同步）与模式一致，无手动 sync，无 CV 融合（§7.4 明确非 MixCV，判定正确——无 `T.gemm` 无跨核流水）。

### 6. 验证方案 — pass

§8.1 含 PyTorch 参考实现草案 `golden_alibi`（独立于 NPU 结构）。§8.2 按 dtype 给出 atol/rtol：fp32 1e-5、fp16 1e-2（均引 `test_alibi`）、bf16 1.6e-2（引本仓 elementwise bf16 通用容差，标注 alibi test 未单列 bf16）。§8.3 L0 门槛测试计划 10 例，含具体规则 shape（含 block 整除用例）、dtype、大 S、int16 值域、对角零、非整除尾块。误差预算核对引设备实测 Q1/Q2（0 违反）。计划完整可执行。

### 7. 完整性与一致性 — pass

11 章节齐全；迁移类 §0 含 0.1–0.7 小节齐全。**无占位符**（占位符类 token 全文检索无命中）。§3.5.3 引用的 examples/testing 路径**逐一实读存在**（`test_arange.py`、`parallel_ops/`12 测试、`_gqa_prefill_fwd_kernel.py` L421–435、`_ssd_chunk_scan_fwd_kernel.py` L349–353、`vec_add_2d.py` L28–42、`leaky_relu/_make_leaky_relu_kernel/DESIGN.md`）。内部一致：API 映射（§3.2）、内存规划（§4）、Tiling（§5）三处 buffer/tile/核数描述相互一致，`bi` 默认值 §3.3 与 §5.2 推荐值显式绑定。manifest 引用（harness `elementwise.yaml` L307–360 与源 `elementwise_generative.yaml` L1–47）内容一致。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性（pass）**：含 1.6.0 算法调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代分析 / 1.6.3 向量化轴与数据布局决策四个子节。
- **算法调研存在性与完整性（四问覆盖，pass）**：R1 候选表**含基线候选 #0**（源 flat div/mod）并逐候选评估；R2 在线算法**明确「无在线化收益」**并给结构依据；R3 复杂度表覆盖**四口径**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值），基线在表、口径写明（write-only，访存下界）；R4 硬件亲和**逐候选**评估（计算单元/片上容量/对齐/静态边界/流水融合）。调研结论明确。
- **调研结论正确性（独立复核，pass）**：
  - **等价性独立推演**：D 共享 = CSE（`|i-j|` 不含 h，`-s_h·D` 与逐元素同值）；负号折入 slope（`(-a)·b == -(a·b)` IEEE 精确）；int16 ramp（`|i-j|≤4095<32767` 精确、i16→f32 无损）；每 head 标量 exp2（容差内）。逐项推演成立，且 `verify_equiv.py` 机器验证 46/46 与内嵌表一致。
  - **复杂度独立复算**：基线 ~10 scalar op/元素（4 int64 div/mod + exp2 + sub + abs + mul + neg + cast）合理；#1 `(6+2H)/(H·bi·bj) = 2/(bi·bj) + 6/(H·bi·bj)`（H=32, 8192 → ≈2.67e-4 op/元素），与「距离项摊薄 H×」一致。**中间缓冲 GM 往返核对**：生成式 write-only，全部中间在 UB，**无 GM 往返**，访存取 `H·S²·elem_bytes` 下界正确；中间缓冲峰值 `O(tile)` 非 `O(N)`，口径自洽。
  - **负向断言主动复核**：对照 `tilelang-op-design/references/algorithm-candidates.md` 的 `ALG-elementwise` 条目（L98–L104：R1「公共子表达式消除/除法乘倒数/cast 链合并」、R2「逐元素映射天然单遍」、R3「融合消除中间 GM 往返」、R4「单遍 Vector 流；dtype 路径」）——设计 R2「无在线变体」与参考表「逐元素天然单遍」一致，**无矛盾**；参考表未记 ALiBi/生成式专门条目（设计已声明调研范围与 `kb_search` 无专属条目），无「明显漏掉的已知候选」。
  - **弃选论证前提逐条核对（重点复核 rev2 修正项）**：亲自打开 `docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1 dtype 矩阵——Ascend 行 **int8 √ / int16 √ / int32 √ / uint64 √ / int64 √ / fp16 √ / fp32 √ / bf16 √**，仅 uint8/uint16/uint32/bool ×。上一轮阻塞项（设计以「bf16 ×」为弃选 #2 前提）**确已消除**：现 §1.6.3 #2 仅以「**无收益**（距离矩阵对称、bj 已可作向量轴；转置不减 op 数/访存）」弃选，并**显式注明 bf16 √、不以 dtype 为理由**——与文档一致。另核对：`T.vabs.md` §2.2.1（int16 ×、fp16/fp32/int32/int64 √）、`T.vdiv.md` §2.2.1（int64/fp16/fp32 √，int16/int32 ×）、`T.vcast.md` §2.2.1（i16→f32 rint、f32→f16/bf16 rint √）、`T.vmul.md` §2.2.1/§2.2.2（标量广播、bf16 ×、仅 2-D 规则）、`T.arange.md` §2.2.1（int16/int32/int64/fp16/fp32 √）——设计全部 dtype 断言与该等文档矩阵**逐格一致**（含 §11.1 自查 ①–⑥ 全部复核通过）。§1.6.0 R4 #7（distance profile gather）负向论断标注「未文档化假设 + 估算依据」，并核对 `docs/.../索引与元素操作/T.gather.md`（存在，§2.4 示例为常数索引 gather，无 LUT 式向量 gather 的实测），标注方式可接受；且该候选非选定，拒绝尚有不依赖此断言的独立理由（#1 已足够、profile 需驻留）。
  - **并列候选裁决路径**：设计以设备实测 Q4（基线 39.029 ms vs arange 0.660 ms，59.1×）硬裁决坐标方案，非纸面拍板；§1.6.3 无依赖未实证常数的判定裕度（选 #1 vs #2/#3/#4 由「无收益/host 契约/3-D 不支持」决定，非估算常数），**无需实验裁决三件套**。
- **调研→设计一致性（pass）**：§1.4 算法描述、§3.1 公式拆解、§6 循环结构均为 §1.6.0 选定算法（结构化分离 + D 共享 + 标量 exp2 + fp32 域单次舍回）的落地；§3.1 明确以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致（kernel 内无逐元素标量循环）。迁移任务：§0.5/§0.6 处置与调研选族一致，源优化手段（§0.4 #1–#8）的意图承接在 §0.6 逐项说明（无静默丢弃；替换项如 #2 div/mod→arange 有「性能裁决 + 实测 59×」承接说明）。
- **数学等价优化完整性（四要素，pass）**：§1.6.1 采纳 4 项（#1 D 共享、#2 负号折叠、#3 int16 ramp、#4 每 head 标量 exp2），每项含「原式 → 优化后公式 → 等价性论证 → 收益量化」，收益栏有量化（指令数 ×H→×1、省一次取负、省 UB 字节、exp2 由每元素降为每 head）。拒绝项 E1–E4 亦逐项给依据。
- **等价性论证独立推演（pass）**：逐项推演如上述成立；容差内等价项（#4 exp2 vs pow）评估了 fp16/bf16 舍入影响（末端 dtype 舍回吸收）并与 §8.2 精度标准一致；`verify_equiv.py` 机器验证与人工推演双一致。
- **向量化替代完整性（与 §3.3/§6 交叉核对，pass）**：§1.6.2 覆盖全部计算点——#1–#4（坐标生成/距离/乘积/舍回）已向量化（有 docs 佐证 + 设备 Q2）；保留 5 处（#5 每 head 标量 exp2、#6 head 调度循环、#7 核内网格跨步、#8 block 索引、#9 动态边界）。与 §3.3 伪代码（kernel 内无逐元素标量循环）与 §6 循环结构（元素级=向量化）交叉核对**无遗漏**。
- **不可替代理由充分性（pass）**：#5 理由为「API 不可靠（`vexp2` 设备 inf，CG-2026-0018；标量是 `T.vmul` 合法标量操作数）+ 每 head 1 次可忽略」；#6–#8 为 tile/block 级调度循环与索引（非逐元素热点）；#9 为编译期依赖 S 的分块整除性。理由具体到本算子，非泛泛。
- **替代 API 可行性（pass）**：替代方案（`arange/vcast/vsub/vabs/vmul`）均有 `docs/Tilelang.language/` 佐证 + 设备实测，优先 v-prefix，与维度 1 口径一致。
- **向量化轴与布局决策完整性（pass）**：§1.6.3 给出候选矩阵（#1 I/O 原生 (bi,bj) 尾轴 bj、#2 核内转置、#3 host permute、#4 3-D tile），逐候选量化评分（整除性/尾 lane 浪费/累加链/repack 代价/UB 影响/写出连续性）；弃选有量化理由（#2 无收益、#3 净亏引 PL-1.3 host permute 106–147µs〔pattern-library INDEX L35 已验证〕、#4 3-D 广播设备实测编译失败 + 文档仅 2-D）。**未把 I/O 契约当内部布局依据**（明确「I/O layout 是契约、核内布局是设计变量」）。核内重排代价机制与 `T.transpose.md`（核内 UB 级、两轴交换链）一致，**无 GM 级 transpose 矛盾**。迁移任务对照了源码隐式轴选择（源无布局优化，生成式无输入读）。
- **布局决策与实现一致（pass）**：§1.6.3 选定方案 #1（`(bi,bj)` 连续 tile、向量化轴尾轴 bj、无 repack）与 §3.3 伪代码（buffer `(bi,bj)`）、§4 内存规划（UB 预算按该布局）、§6 循环结构三方一致。
- **与 §3.1/§6 一致性（pass）**：§3.1 以 §1.6.1 优化后公式（D/vmul/vcast）为输入；§6 无 §1.6.2 未论证的标量循环。

## 备注（非阻塞性观察）

- §0.7 将 Op 层 `reshape(H,S,S)` 一并写入 Kernel `forward` 的表述链（L57「`forward()` 无参 → `self._compiled_fn().reshape(H,S,S)`」）略欠精确：源码中 Kernel `forward` 仅 `return self._compiled_fn()`（L3641–3642），`reshape` 在 Op 层 `AlibiFwdOp.forward`（`ops/elementwise/alibi.py` L71）。§0.1 对该分工的描述是准确的，故仅为行文瑕疵，不影响语义、不阻塞。
- §1.6.0 R4 #7 的「无核内向量 gather 先例/实测」措辞略宽（`T.gather.md` §2.4 有常数索引 gather 示例，但无 LUT 式变索引向量 gather 的实测）。因该候选非选定、拒绝有独立充分理由，且已标注「未文档化假设 + 估算依据」，判定为可接受的表述精度，不阻塞。

## 附：本检视对 rev2 声明的独立验证结论

| rev2 声明（§11.1） | 本检视独立验证 | 结论 |
|---|---|---|
| 删除 §1.6.3 #2 `T.transpose` dtype 负向前提 | 实读 `T.transpose.md` §2.2.1：bf16 √；现 §1.6.3 #2 仅以「无收益」弃选并注明 bf16 √ | **声明属实** |
| 落盘 UB 天花板探针 | `history_version/design_probe_alibi_ub_ceiling.py` 存在，sweep 8192/12288/16384/24576，报文与 §4.5 一致 | **声明属实** |
| bisect 改述为「会话内实测、未落盘」 | §3.4 已就 bisect 标注「会话内实测、未落盘」，不构成 §4/§5 操作界 | **声明属实** |
| 生成式算子 host permute 表述改为「输出侧一次」 | §1.6.3 #3 已改为「输出侧一次 host permute」并注明输入侧无 permute | **声明属实** |
| 全文 dtype 断言自查（①–⑥） | 逐格打开 T.transpose/T.vabs/T.vdiv/T.vcast/T.vmul/T.arange 文档矩阵核对，全部一致 | **声明属实** |
| 技术结论（div/mod 可实现性、59.1×、arange 选定、4 项优化、verify_equiv 46/46、分核三要素）不变 | verify_equiv 重跑 46/46；NPUUtils 实查 20/40；§1.6/§5.5 技术结论复核一致 | **声明属实** |
