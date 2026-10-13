结论: 通过

# add_fwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

（无阻塞级问题；2 项建议级问题，见「检视详情」维度 0 / 维度 2，供 Stage 1 顺手修订但不阻塞进入 Stage 3。）

## 检视元信息

- 文档: examples/add/add_fwd_kernel/DESIGN.md
- 任务类型: 迁移（GPU TileOPs → TileLang-NPUIR）
- 源算子: /mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/arithmetic.py（`AddFwdKernel`）、.../elementwise/_base.py、.../elementwise/_builders.py、.../elementwise/_broadcast.py、/mnt/workspace/TileOPs-main/src/tileops/ops/elementwise/arithmetic.py（`AddFwdOp`）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 2

## 机械复核（D-1/D-5）

- verify_equiv 重跑: **EQUIV_PASS**（`python3 examples/add/add_fwd_kernel/verify_equiv.py` → `total rows: 135   EQUIV_PASS: 135   EQUIV_FAIL: 0`，`OVERALL: EQUIV_PASS`）。结构结论与 §1.6.1 内嵌表一致：行数一致（135 = item1 30 + item2 105）、覆盖一致（3 dtype × 5 shape；item1 alpha∈{1,1.0}、item2 alpha∈{1,0.5,2.0,−1.0,0.0,3}）、结论一致（全 EQUIV_PASS、0 违反 / 0 NaN 失配）。注意：脚本 `_make_inputs` 用未设种子的 `torch.randn`，故 `max ulp` 数值逐次漂移（连跑三次 item2 得 0.0124 / 0.0148 / 0.0133，base_viol 1–3），内嵌表记的 0.0128、对照基线 1.093/4 违反属其中一次采样——**判定级结论（EQUIV_PASS）稳定可复现**，仅随机 ulp 数值非定值，不构成「与内嵌表不一致」。
- design_calc_check: `python3 .agents/tools/design_calc_check.py --design examples/add/add_fwd_kernel/DESIGN.md` 输出（逐项）:
  - `ub_budget`: **skip** — "no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"。人工复核已补做（见维度 2）：fp16/bf16 路径 112KB ✓、fp32 复用路径 96KB ✓；fp32 全 buffer 显式路径正文写 128KB、实算应为 160KB（建议级算术口径错误，仍 < 192KB）。
  - `l0c_budget`: **skip** — 无 Cube 路径（正确，本算子纯 Vector）。
  - `core_split`: **skip** — "insufficient parsed dims {}; declared logical cores = 8192"（工具未解析出 M/block 维度，人工复核已补做：§5.5 三要素齐全、算术自洽，见维度 3）。
  - `r3_metrics`: **pass** — "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"。

## 检视详情

### 0. 源算子理解与迁移分析 — warn

**核对方式**：已亲自 Read 源算子全链路（`arithmetic.py` `AddFwdKernel` L43-50；`_base.py` `AlphaScaledBinaryKernel._alpha_op_func` L607-655、`BinaryKernel.__init__/forward/_build_kernel` L299-416；`_builders.py` `make_binary_register_copy`/`make_binary_explicit`/`_row_broadcast_prim`；`_broadcast.py` `coalesce_broadcast_dims`/`compute_broadcast_offsets`/`row_broadcast_split`；`ops/elementwise/arithmetic.py` `AddFwdOp`/`_AlphaScaledBinaryOp`；`ops/elementwise/_base.py` `BinaryOp`），并核对 NPU scaffold `examples/TileOPs/tileops/ops/elementwise/add.py`。

- **语义理解正确性 — pass**：§0.1 数学语义 `y = a + α·b`（广播）与源码一致（`AddFwdKernel._combine` 返回 `a + scaled_b`，`arithmetic.py` L48-50）。关键精度细节正确：§0.1 指出浮点分支「`α·b` 在 fp32 域相乘、cast 回 `a.dtype`、再在 `a.dtype` 内加」——与源码 `_base.py` L647-649 `scaled_b = T.cast(T.cast(alpha,"float32")*T.cast(b,"float32"), a.dtype); return combine(a, scaled_b)` 逐字对应；`alpha==1` 恒等折叠（L619-625）亦识别正确。dtype 语义：NPU 契约浮点子集（float16/bfloat16/float32）与 NPU manifest `examples/TileOPs/tileops/manifest/elementwise.yaml` `AddFwdOp.signature`（`float16 | bfloat16 | float32`、`output: same_as(input)`、`shape_rules: broadcast_shapes`）一致；GPU 侧 `SUPPORTED_DTYPES = BINARY_FULL_DTYPES`（含整数/bool）亦如实标注。无规约语义、无跨元素依赖、边界语义（无专用 NaN 链、`0*inf=NaN` 故不折叠 `alpha==0`）均与源码一致。§0.2 I/O（输出 `same_as(input)`、无转置、广播 shape）与源码签名一致。
- **算法解读完整性 — 部分不准确（warn 来源）**：§0.3「GPU 双策略分派」段落正确识别源码级事实——`BinaryKernel` 依 `coalesce_broadcast_dims`+`is_contiguous_same_shape` 选 `register_copy`（same-shape 连续）或 `explicit_parallel`（stride 版 `make_binary_explicit` + `compute_broadcast_offsets` 逐元素解偏移），行广播走 `_row_broadcast_prim`（`row_broadcast_split`）；且 §0.6 R2 明确「GPU 在 kernel 内完成广播索引，**两操作数不物化**」——此为源码事实。**但 §0.3 步骤表第 1 行与「host 侧逻辑」段把 Op 层 `out_shape = broadcast_shapes(...)`；`expand(t,out_shape).contiguous().view(-1)` 及方法名 `AddFwdOp._eager_forward` 归于**GPU 源算子，与源码不符**：GPU `AddFwdOp` 无 `_eager_forward`（全仓 grep 无此符号），`BinaryOp.forward`（`ops/elementwise/_base.py` L118-123）仅 `input.contiguous(); other.contiguous()` 后交内核，广播 lowering 的注释明言「flattening, broadcasting and restoring the output shape are the kernel's own business」（同文件 L11-14、L87-89）；内核端 `_broadcast.flat(t)=t.reshape(-1)` 用各操作数自身 numel，非 N_total。该段描述的其实是 **NPU scaffold** `examples/TileOPs/tileops/ops/elementwise/add.py` 的 `_expand_flat`/`_eager_forward`（L103-120）。即 §0.3 步骤 1 与自身 §0.6 R2 存在**内部矛盾**（一处称源在 Op 层物化广播、一处称源在 kernel 内不物化）。此属源码来源归属/层级标注不准确，**不影响数学语义、I/O、dtype 与广播机制的正确理解，也不改变任何 §1–§7 决策**（迁移决策 R2 依据的正是正确的「GPU 内核内 stride 广播」前提），故判 warn 而非 fail；建议 Stage 1 将步骤 1/host 段改标注为「NPU scaffold/目标契约」，以消除自相矛盾。
- **优化手段覆盖完整性 — pass（一处小笔误）**：§0.4 七项优化（register_copy、coalesced 向量化、alpha 编入 op body + `alpha==1` 折叠、fp32 域标量乘、`threads×npt` 两级并行、广播 stride 化解算、lru_cache 特化）均与源码对应并逐项标硬件依赖，无把实现层误判为算法层。小笔误：§0.4 #7 引用的 `_param_key`（`_base.py` L676）实为 `ScalarParamUnaryKernel` 的方法，`AlphaScaledBinaryKernel` 走的是 `_get_effective_op_func`+`register_op_func`；`@functools.lru_cache` 在 builders 中确实存在，机制判断无误，仅方法名归属小错。
- **耦合性判定合理性 — pass**：§0.5 每条目四态处置（保留/等价替换/重新设计/舍弃）+ 依据，与 GPU→NPU 映射表及 ascend-constraints 不矛盾：语义与算法层「保留」、SMEM/寄存器→UB 与 coalesced→MTE 连续搬运为「等价替换」、`threads×npt`→一维 Kernel+UB 向量化与广播 stride 解→Op 层展平为「重新设计」、lru_cache→factory 缓存为「等价替换」；无「舍弃」项误标（广播 stride 解的「单遍、无物化」意图由 Op 层展平承接，已写明）。无 warp shuffle / mma / swizzle 类漏判。
- **NPU 重设计可行性 — pass**：§0.6 R1（并行结构：一维 `T.Kernel`+`block_size` 折叠+Vector 整块 v-prefix）与 R2（广播处理位置：Op 层 `broadcast_shapes`+`expand`+展平）均含「源方案 → NPU 新算法 → 语义保持论证」三要素；新算法满足一维 Kernel、UB 驻留、32B 对齐、静态边界约束；语义保持论证说明逐元素映射不变、每元素表达式与求值顺序与源一致。R2 的代价（广播操作数物化 ≈+21µs）如实披露并给 Stage 4 A/B 备选（`T.vbrc` 行广播）。
- **设计与决策一致性 — pass**：§1.4/§1.5/§3.1/§3.2/§4.4/§5.4/§6 描述的是 §0.5/§0.6 决策后的 NPU 算法（一维 Kernel、Op 层广播展平、UB v-prefix 链）；§6 循环结构无 §0.5 已判重设计的 `threads/npt` 残留；无「§0 判重设计、§3 照抄源方案」的脱节。
- **golden 独立性 — pass**：§8.1 golden 为 `torch.add(a.float(), b.float(), alpha).to(a.dtype)`（移植源仓测试基准），仅依赖 torch 原生广播，**未复刻** NPU 的 vmul/vadd fp32 链、persistent 结构或 Op 层展平，独立于 §0.6 算法。
- **广播处理决策（本次重点）— pass**：对照 GPU 源码，广播确由内核以 stride 计划完成（`make_binary_explicit`/`_row_broadcast_prim` + `compute_broadcast_offsets`，`_broadcast.py` L69-81），same-shape 走 `register_copy`；设计据此将 NPU 侧改为 Op 层展平为 same-shape 1-D（kernel 只见平铺输入），与 NPU scaffold `_expand_flat`、lerp_tensor 迁移先例（`.../lerp_tensor_kernel/_make_lerp_tensor_kernel_DESIGN.md` §0.3 步 1）及 `examples/TileOPs/docs/gpu_to_npu_adaptation.md` §3.4 E15 一致。负向断言「NPU 缺跨步 gather 向量指令」**附真实文档路径而非未文档化假设**：`.agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md` `TRAP-C10-strided-gather`（已打开核对：L29 id、L41-43「跨步 gather 向量指令缺失（截至 2026-08-28 build 有效）」）。决策成立。

> 结论：维度 0 语义/算法/优化/耦合/重设计/一致性/golden 独立性均核对通过；唯一瑕疵为 §0.3 步骤 1/host 段的**源码层级归属不准确**（把 NPU scaffold 的 Op 层物化写成 GPU 源行为，与 §0.6 R2 自相矛盾），判 warn。

### 1. API 可行性 — pass

§3.2/§3.4 每条 TileLang API 均有 `docs/Tilelang.language/` 佐证且已核对：`T.vadd`（`数学操作/T.vadd.md`，src1 可标量、dst 可原地、fp16/fp32 √、bf16 ×）、`T.vmul`（`数学操作/T.vmul.md`，**仅 src2 可标量**、dst 可原地、bf16 ×）、`T.vcast`（`数据类型转换操作/T.vcast.md`，f16→f32 仅 rint、bf16→f32 rint、f32→f16/bf16 含 rint、src 仅 buffer）、`T.copy`（`内存操作/T.copy.md`）、`T.alloc_ub`（`内存操作/T.alloc_ub.md`）、`T.ceildiv`/`T.min`。设计严格使用 v-prefix API（vadd/vmul/vcast），符合 AGENTS.md API 约定。bf16 无原生 v-prefix 算术的判断与两份文档的 dtype 列（bf16 ×）一致。`T.Kernel(一维, is_npu=True)` 与 mish/hardsigmoid/hardswish 同族先例一致。API 与 Developer 模式匹配（无手动 set_flag/wait_flag）。

### 2. 内存层级规划 — warn

- 搬运路径（§4.4）完整：GM→UB（`T.copy`）→ v-prefix 计算 → UB→GM，纯 Vector 两级、无 GM→L0 越级；每个 buffer 标注 UB 层级。✓
- UB 预算（§4.5）：**fp16/bf16 路径** 16KB+16KB+32KB+32KB+16KB = 114688B = 112KB ✓；**fp32 复用路径**（`a32←a_ub`、`b32←b_ub`、`y_ub←结果`）3×32KB = 96KB ✓。
- **建议级问题（warn 来源）**：**fp32「全 buffer 显式」路径**正文计为 `131072（128KB）`，但按 §4.5 表自身逐 buffer 相加应为 a_ub(32768)+b_ub(32768)+a32(32768)+b32(32768)+y_ub(32768) = **163840B（160KB）**（正文少计了 1 个 fp32 buffer，128KB = 4×32768）。按 ×1.12 上偏系数：美文写 143KB/余量 ~25%，实为 160×1.12 ≈ 179KB、余量仅 ~7%。**但 160KB 与 179KB 均 < 192KB，结论（不溢出）不变**，且主用/推荐路径（112KB / 96KB）算术正确。该行属 `design_calc_check.ub_budget` 因无法解析而 skip 的 VP-2026-0009 混合字节口径类错误（工具未拦截，本维度人工复算命中）。建议 Stage 1 修订 §4.5 该行数值与余量。因不改写可行性结论、不触发 UB 溢出，判 warn。

### 3. Tiling 策略 — pass

§5.2 block 划分明确（`block_size=8192`，承 mish/hardsigmoid/hardswish 同族实测 + 与 NPU scaffold `_DEFAULT_BLOCK_SIZE` 一致）。对齐约束：8192×2B/4B 均 32B 对齐、整除向量宽（fp16/bf16 ×8、fp32 ×4）✓。非整除：`tail = T.min(block_size, shape-offset)` + `block_id < n_num` guard，动态 N 尾块处理明确（§6.4）。**分核策略三要素齐全（§5.5）**：① 逻辑核数 `ceil(N/8192)`（四 workload 512–1568）；② 物理核数**实查** `NPUUtils.get().get_aicore_num()` → 20，纯 Vector 翻倍 = 40（含查询代码/返回值与日期，且说明与 `CONST-aicore-910B2C`(24) 不一致时以实查为准）；③ 规模判定（全部远超 40）+ persistent 方案（`num_kernels=min(ceil(N/block),40)` + 核内 `T.serial` 静态边界 + persistent 模板先例）。核数与 block 自洽，无核启动风暴、无动态边界依赖。`design_calc_check.core_split` 因无法解析维度 skip，已人工复算通过。

### 4. 技术约束检测 — pass

§3.5.1 逐项核对：三维 Kernel — 不涉及（源即一维 grid，广播经 Op 层展平后仍一维）；GPU 专用 API（`T.Parallel`/`T.vectorized`/`threads=`）— 已识别并全部改写为 NPU v-prefix + UB 路径；GEMM 非整除 / L0C 溢出 — 不适用（无 Cube）；分核策略 — 已处理（§5.5）。`design_calc_check.l0c_budget` skip 合理（无 Cube 路径）。

### 5. 循环与同步 — pass

§6.1 明确循环类型：block 级 `T.Kernel`、tile 级 `T.serial(num_local_tasks)`（核内串行、边界静态）、元素级 v-prefix 整块向量化；无逐元素标量循环。§6.4 尾块逻辑明确。§7 同步策略与 Developer 模式一致（自动同步，无手动 set_flag/wait_flag，无跨核协作）；`alpha==1` 折叠为 trace-time 单一程序、无运行时分支。§6.3 `T.Pipelined` 明确为 Stage 4 可选，不影响 Stage 3 基线正确性。

### 6. 验证方案 — pass

§8.1 golden 为 PyTorch 参考实现草案（`torch.add(a.float(), b.float(), alpha).to(a.dtype)`），独立于 NPU 算法。§8.2 精度标准按 dtype 给 atol/rtol（fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2，与源仓 `tests/ops/test_add.py::_get_tolerances` 一致）。§8.3 L0 门槛计划具体（N=2^20、shape [1024,1024]、三 dtype、含非默认 alpha 与广播路径）+ 边界用例要点（NaN/±inf 自然传播、`alpha==0` 不折叠）。完整。

### 7. 完整性与一致性 — pass

章节齐全（§0 含 0.1–0.7 全部小节，§1–§10 完整）；无 `{placeholder}`/`TODO`/`待补充`（§10.2 的「⬜ 待实现（Stage 3）」为交付状态标注，非占位符）。§3.5.3 引用的同类实现路径全部真实存在（已逐条 `ls` 核对：NPU scaffold `add.py`、hardswish DESIGN、mish、lerp_tensor 的 `_make_lerp_tensor_kernel.py`/DESIGN/perf_opt）。API 映射、内存规划、Tiling 三处描述相互一致。运行期 `shape: T.int32` 与 `N`（JIT 参数）双轨口径清楚。唯一内部矛盾（§0.3 vs §0.6 R2 的广播层级归属）已在维度 0 记录。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性**：含 §1.6.0 算法调研、§1.6.1 数学等价优化、§1.6.2 向量化替代分析、§1.6.3 向量化轴与数据布局决策四子节，齐全。
- **算法调研存在性与完整性**：§1.6.0 四问齐全——R1 候选表**含基线候选（#0 源算法）**并逐条初判；R2 在线算法明确结论「无（逐元素天然单遍）」并给结构依据（无跨元素依赖/无 running 统计量/无归约）；R3 复杂度表覆盖**四口径**（FLOPs / 访存量 Bytes / 扫描遍数 / 中间缓冲峰值，基线在表、口径写清）；R4 逐候选硬件亲和评估（单元匹配/容量/对齐/静态边界/流水融合/跨核）。「无更优替代」写明调研范围（algorithm-candidates.md elementwise 命中行、恒等变形自查、pattern-library）。深度与单步逐元素算子匹配。
- **调研结论正确性（独立复核）**：① 等价性——item1（`alpha==1` 折叠）独立推演：IEEE `1.0*x == x` 对全部可表示值精确成立（含 ±inf/NaN/−0.0/subnormal），加法逐位不变，成立；item2（完整 fp32 中转）独立推演为容差内等价（两实现均逼近 `a+αb` 真值，差 ≤1 ulp），与 `TRAP-fp16-opmath-golden` 通解一致，稳定。② **复杂度复算**：基线访存 `3·N·elem_bytes`（读 a+读 b+写 y，无中间 GM 往返，正确未漏计）；roofline 算式独立复算：cnn-feat fp16 `3×12,845,056×2B≈77MB ÷ (30.4GB/s×40)=63µs` ✓、hidden-prefill fp32 `3×8,388,608×4B≈100.7MB ÷(31.1×40)≈81µs` ✓、广播物化附加 `12,845,056×2B≈25.7MB ÷1.216TB/s≈21µs` ✓；逻辑核数/每核串行数（1024→26、1568→40、1376→35、512→13）独立复算全部吻合。③ 亲和性淘汰有佐证（见下）。④ **负向断言复核**：`TRAP-C10-strided-gather` 已亲开核对（真实、含日期条款），「无在线变体」对照 elementwise 命中行「R2 逐元素映射天然单遍」一致，未发现明显漏掉的已知候选（本算子无归约/无跨元素依赖，无 online/Welford/flash 类候选适用），不存在「无在线变体」被推翻的情形。⑤ 并列候选：R3 结论「候选间四口径基本同、基线最优、候选 5 因 R4 淘汰」无并列争议，无需实验裁决。
- **调研→设计一致性**：§1.4 算法描述、§3.1 公式拆解、§6 循环结构均以 §1.6.0 选定算法（逐元素直接映射、fp32 中转、`alpha==1` 折叠、Op 层广播展平）为准；迁移任务 §0.5/§0.6 处置与调研选族一致；源优化手段（§0.4）意图承接逐项说明（§0.6 末段），无静默丢弃。
- **数学等价优化完整性**：§1.6.1 对选定算法系统审视，采纳 2 项，每项四要素「原式 → 优化后公式 → 等价性论证 → 收益估算」齐全；无优化空间结论有依据（单次逐元素、无 CSE、无除法/超越、无 cast 链冗余、无中间 GM 往返；`alpha==0` 折叠因 `0*inf=NaN` 语义不成立被拒）。
- **等价性论证正确性**：两项独立推演成立（见①）；容差内等价与 §8.2 精度标准（fp16 1e-3/bf16 1.6e-2）口径一致；`verify_equiv.py` 重跑全 EQUIV_PASS（见机械复核段）。备注：脚本 item2 的「base_native」基线为「原生 dtype 乘加」，与 §1.6.1 所述「原式」（源：fp32 乘→cast→dtype 加）略不同名，但二者均属窄域实现的对照项，候选 vs golden/fp64 的核心机器验证成立，属表述细节。
- **向量化替代完整性**：§1.6.2 覆盖 8 个计算点，与 §3.3 伪代码、§6 循环结构交叉核对——核内 4 个逐元素点（升精度 cast / 标量乘 / 加 / 降精度 cast）全部向量化；保留 4 处标量（block 起始索引、尾块 `T.min`、分核元数据、Op 层广播）均为 block 级任务映射/动态边界/host-trace 侧，非逐元素热点，理由具体到本算子。§6 不出现 §1.6.2 未论证的逐元素标量循环。
- **替代 API 可行性**：vcast/vmul/vadd 均有 `docs/Tilelang.language/` 文档佐证（已核对），优先 v-prefix。
- **向量化轴与布局决策完整性**：§1.6.3 候选矩阵枚举 5 个候选，逐候选给量化取舍（整除性/尾 lane 浪费/累加链/repack 代价/UB 影响）；**弃选论证前提核对**：候选 #3 弃选引 `TRAP-C10-strided-gather`（已核对，真实）、候选 #4 引 `TRAP-vbrc-same-shape-empty-broadcast`（已核对 traps-compiler.md L232/244「vbrc 同形 src/dst → 空广播维 MLIR verify fail」真实）、候选 #5 引 `PL-1.3-host-permute`（已核对 INDEX/layout.md，实测 106–147µs，与设计「百 µs 级净亏」一致）。**无「未考虑」式论证，无把 I/O 契约当内部布局依据**（明确 I/O layout 为契约、核内布局为设计变量，并对照源码隐式轴选择）。
- **实验裁决模式完整性**：判定裕度未依赖未实证常数（roofline 用 pattern-library 已实测常数 `CONST-mte2-degradation`/`CONST-vector-launch-overhead`/`CONST-capacity-910B2C`/`CONST-aicore-910B2C`，均已核对条目存在）；候选 #4（`T.vbrc` 行广播免物化）作为 Stage 4 A/B 备选已给结构说明（§9.1），不属「纸面单选依赖未知常数」情形。
- **布局决策与实现一致**：§1.6.3 选定「Op 层广播展平 + 展平轴向量化」与 §3.3 伪代码（1-D buffer、展平轴）、§4 buffer 形状（均为 `(block_size,)` 1-D）、§6 循环结构三方一致。
- **与下游章节一致**：§3.1 以 §1.6.1 优化后公式为输入（含 `alpha==1` 折叠分支）；§6 循环结构与 §1.6.2 结论一致。
