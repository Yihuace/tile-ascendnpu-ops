# floor_divide_fwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/floor_divide/floor_divide_fwd_kernel/DESIGN.md
- 任务类型: 迁移（`scenario=migration`，`migration_mode=harness`）
- 源算子: `/tmp/opencode/TileOPs-ref/tileops/kernels/elementwise.py`（`FloorDivideFwdKernel` @ L1535–L1550 / `_make_binary_explicit` @ L556–L609 / `BinaryKernel` @ L890–L1128）+ `/tmp/opencode/TileOPs-ref/tileops/ops/elementwise/arithmetic.py`（`FloorDivideFwdOp` @ L127）
- 检视维度: 9 项（0 源算子理解与迁移分析 / 1 API 可行性 / 2 内存层级 / 3 Tiling / 4 技术约束 / 5 循环同步 / 6 验证方案 / 7 完整性 / 8 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `PASS`。执行 `python3 examples/floor_divide/floor_divide_fwd_kernel/verify_equiv.py` → `OVERALL: EQUIV_PASS`（exit 0），**执行输出与 DESIGN §1.6.1 内嵌结果表逐项一致**：
  - E1（行/内层映射 vs 源 `_compute_broadcast_offsets`）：9 组形状（含全部 manifest workload + 测试广播模式 + 角色互换 + 全标量）`a_mm=0, b_mm=0, a_oob=0, b_oob=0` 全 `EQUIV_PASS`；
  - E2（fp32 中转链 `vdiv`+`vcast(floor)` vs golden）：fp16/bf16/fp32 各 66,377 例（随机正数 + 角点 ±inf/NaN/±0/次正规/整数边界）`maxdiff=0, violations=0` 全 `EQUIV_PASS`；
  - E3（负向对照：免 fp32 中转的原生半精度）：fp16 66 违反/maxdiff=1、bf16 430 违反/maxdiff=4（**证明 fp32 中转为正确性必需**，非采纳项，不计入 OVERALL 判定）。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design examples/floor_divide/floor_divide_fwd_kernel/DESIGN.md`）JSON `checks` 摘要：
  - `ub_budget`: **skip** — detail「no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)」→ 人工复核：§4.5 表 fp32 16 B/elem×4096=64KB×1.7≈109KB、fp16/bf16 22 B/elem×4096=88KB×1.7≈150KB，均 ≤192KB（`CONST-capacity-910B2C`），**人工已核通过**；
  - `l0c_budget`: **skip** — 纯 Vector 无 Cube 路径，不适用，**人工已核（n/a）**；
  - `core_split`: **skip** — detail「insufficient parsed dims {}; declared logical cores = None」→ 人工复核：§5.5 逐 workload `num_logical={1,2048,4096,4096}`、`num_kernels=min(num_logical,40)`、`num_local=ceildiv(num_logical,40)={1,52,103,103}`，与 `get_aicore_num()*2` 自洽，**人工已核通过**；
  - `r3_metrics`: **pass** — 「all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)」。
- **物理核数实查复核**: 检视期独立执行 `NPUUtils.get().get_aicore_num()` → **20**（与 DESIGN §5.5 记录一致），纯 Vector 核 = 40 ✓。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

- **语义理解正确性（pass）**：亲自 Read 源 `FloorDivideFwdKernel.op_func`（elementwise.py L1546–L1550）= `a_f32 = T.cast(a,"float32"); b_f32 = T.cast(b,"float32"); return T.Cast(a.dtype, T.floor(a_f32/b_f32))` → **fp32 域除法 + fp32 floor + 单次 cast 回**。§0.1 的「先除后 floor、商为整值浮点」「fp32 域计算」与源码逐字一致；源码注释 L1538–L1541 明确 fp16 商会先舍入再 floor（`2.999…→3.0`）——§0.1 采纳。`_FLOAT_DTYPES = float16|bfloat16|float32`（L154–L158）、`FloorDivideFwdKernel.SUPPORTED_DTYPES = _FLOAT_DTYPES`（L1544）、整数/布尔拒绝（`BinaryKernel.__init__` L926–L930 抛 ValueError）均与 §0.1 一致。输出 dtype `same_as(input)`：源 `out_dtype = output_dtype or dtype`、`OUTPUT_DTYPE = None` → ✓。输出展平 `(N_total,)`：源 `main` 形参 `y: T.Tensor((N_total,), out_dtype)` + `out_idx=[2]` ✓。
- **算法解读完整性（pass）**：§0.3 计算步骤表覆盖源全部计算语句——step 1 host `coalesce_broadcast_dims`/`_is_contiguous_same_shape`（源 L935/L952）、step 2 `block_size=threads*npt` + `grid=ceildiv`（L567/L575/L599）、step 3a 同形快路径（L576–L578 / 提取件 L61–L65）、step 3b 广播路径 `_compute_broadcast_offsets`（L602–L605 / 提取件 L88–L93）、step 4 Op 层 reshape；无遗漏、无臆造。`_compute_broadcast_offsets`（elementwise.py L435–L451 = 提取件 L167–L183）的 mixed-radix divmod 链描述正确。数据流图覆盖 a/b/y 全部 buffer。host 侧 `lru_cache`/`out_dtype` 已列。
- **优化手段覆盖完整性（pass）**：§0.4 对照清单列出 8 项（`explicit_parallel` 两级并行 / coalesced 连续段 / `coalesce_broadcast_dims` / 编译期 unrolled divmod / 同形快路径 / `register_copy` fragment 驻留 / 工厂+lru_cache / fp32 域 + 单次 cast），逐项标注硬件依赖，且声明源无 SMEM tiling / mma / warp shuffle / 异步流水 / swizzle——与源码实况相符。注记（非阻塞）：源码同形默认策略为 `register_copy`（`BinaryKernel.__init__` L1000–L1003），而 Stage 0 提取件对应 `explicit_parallel`（Pattern A）；§0.4 #6 把 `register_copy` 作为族级优化列出、§0.5/§0.6 以「fragment→UB」等价替换承接，处理自洽，无正确性影响。
- **耦合性判定合理性（pass）**：§0.5 每条目四态处置齐全（保留 7 / 等价替换 3 / 重新设计 3 / 舍弃 1）：`threads=`/`npt` kwarg 舍弃有理由（npuir 不接受、`block_size` 承接）；`T.Parallel(threads,npt)` 与 GPU grid 超发判「重新设计」，逐元素 divmod/gather 判「重新设计」——与 `ascend-constraints.md`（一维 Kernel、host 禁改输入）及本项目已知限制一致；无把 warp/线程模型判「保留」的错误。
- **NPU 重设计可行性（pass）**：§0.6 R1/R2/R4 均含「源方案 → NPU 新算法 → 语义保持论证」三要素。R1 行级 base + innermost 向量化满足一维 Kernel / UB / 静态边界；R2 一维 `T.Kernel` + persistent `T.serial`（静态边界）+ 128bit lane 满足 NPU 硬件约束；R4 两段链 `T.vdiv`+`T.vcast(round_mode="floor")` + fp32 中转满足对齐/容量/静态边界。语义保持论证均落到 `verify_equiv.py` 机器证据（E1 索引逐点、E2 链 bit-exact）。注记（非阻塞）：§0.6 重设计项编号为 R1/R2/R4（无 R3），与 §0.5 判定统计「R1/R2/R4」一致，仅为编号留白，不影响结论。
- **设计与决策一致性（pass）**：§1.4 算法描述、§3.3 伪代码、§4.4 搬运路径、§6 循环结构描述的都是重设计后的 NPU 算法（一维 Kernel + 行/内层 + vdiv/vcast 链），无「§0 判重设计、§3 照抄源 `T.Parallel(threads,npt)`」的脱节；§3.5.2 逐项对照 GPU→NPU 差异。
- **golden 独立性（pass）**：§8.1 golden `torch.floor(a.detach().cpu().float()/b.detach().cpu().float()).to(a.dtype)`（广播直算）以 §0.1 源语义为唯一依据，与源仓 `test_floor_divide.py` `ref_program`（L40–L41）一致，**未**复刻 §0.6 的行/内层索引、vdiv/vcast 链或 persistent 结构；E1（索引）与 E2（链）为独立非平凡校验。

### 1. API 可行性 — pass

- 所有 API 均有源码/文档/示例三重佐证且亲自核对：`T.vdiv`（`npuir_div`，`customize_npuir.py` L158–L169；`docs/Tilelang.language/数学操作/T.vdiv.md` §2.1/§2.2/§2.4）、`T.vcast`（`docs/.../数据类型转换操作/T.vcast.md` §2.1/§2.2.1：`f32|f32|floor` 与 `f16→f32 rint`/`bf16→f32 rint`/`f32→f16`/`f32→bf16` 均列支持）、`T.vbrc`（`docs/.../shape操作/T.vbrc.md` §2.2.2 rank 一致 + size-1 广播；`examples/elementwise/vec_add_auto_brc.py` 先例）、`T.alloc_ub`/`T.copy`/`T.min`/`T.ceildiv`/`T.Kernel(...is_npu=True)`/`T.serial`（docs + `examples/elementwise/vec_add_1d.py` 1-D 先例）。
- **v-prefix 优先**：全程使用 v-prefix（vdiv/vcast/vbrc），无 legacy `npuir_xxx` 写法 ✓（`tilelang/language/__init__.py` L83/L97/L99 确认 vdiv/vfloor/vfloordiv 别名）。
- **模式匹配**：Developer 模式（`TILELANG_ASCEND_MODE=Developer`），无手动 `set_flag/wait_flag`，与 §7 自动同步一致 ✓。
- **重点核对（`T.vdiv` 自动广播）**：`T.vdiv.md` §2.2.2 明确「结论：**支持自动广播**」，并列同形/行广播 `[M,N]/[M,1]`/标量广播；探针 `design_probe_floordiv_bcast.py` F1/F2/F3 实测 `(M,N)/(1,N)`、`(M,N)/(M,1)`、`(M,N)/scalar` bit-exact。**论断与文档总述一致**。细节注记（非阻塞）：文档枚举的「列广播」示例写作 `[1,N]/[1,1]`（未逐字给出 `[M,N]/[1,N]`），与 DESIGN 的措辞存在字面差异；且主选路径 (#1) 不依赖该广播能力（广播 operand 经 `T.vbrc` 物化），自动广播仅用于 §1.6.3 备选 #2，故不构成风险。

### 2. 内存层级规划 — pass

- **搬运路径完整**：§4.4 给出 GM→UB 单阶段路径（纯 Vector，无 L1/L0），无越级直连 ✓。
- **UB 预算（machine skip → 人工已核）**：§4.5 fp32 活跃 16 B/elem（a32/b32/q/yf）×4096=64KB；fp16/bf16 22 B/elem（+a_in/b_in/y_out）×4096=88KB；×1.7 auto-multi-buffer 膨胀（`CONST-capacity-910B2C`）后 ≈109KB / ≈150KB ≤ 192KB。手工复算一致，`block_size` 上限（fp32 ≤7228、fp16/bf16 ≤5257）推导正确，推荐 4096 达标 ✓。
- **Buffer 层级标注**：§4.3 每个 buffer 标注 UB 及用途（含 `(1,)` 标量广播源）✓。
- **L0C 容量**：不适用（无 Cube）。

### 3. Tiling 策略 — pass

- **Block 划分**：§5.2 给出 `block_size=4096` 及 6 条理由（innermost 单块覆盖 L∈{3136,4096}、32B 对齐、向量宽度整除 0% 尾 lane 浪费、UB 预算、极大 L 由 chunks 分块、同族一致）✓。
- **对齐约束**：4096×{2,4}B = 8K/16K 均 32B 整除；fp32×4 / fp16×8 整除 4096 ✓。
- **非整除处理**：§6.4 `tail = T.min(block_size, L−inner_off)` 尾块截断，采用 `vec_add_1d.py` 已验证零起点 dst 切片模式 ✓。
- **分核策略三要素（core-split-strategy.md §1；machine skip → 人工已核）**：
  - ① 逻辑核数：`num_logical = R × ceildiv(L, block_size)`，逐 workload 表（smoke=1 / hidden=2048 / cnn=4096 / row=4096）——复算一致；
  - ② 物理核数依据：**实查记录**含查询代码 `NPUUtils.get().get_aicore_num()` 与返回值 20（检视期独立复跑仍为 20）⇒ 纯 Vector 核 40，未用文档假设值；并注明 `CONST-aicore-910B2C`（24）与本机不同源、以实查为准 ✓；
  - ③ 规模判定与分核：smoke `num_logical=1 ≤ 40` → 无需适配；其余 `≫40` → 极大规模，`num_kernels=min(num_logical,40)` + 核内 `T.serial(ceildiv(num_logical,40))`（**编译期静态边界**）+ grid-stride 映射 + `if task<num_logical` 屏蔽；负载不均 ≤1 task/核（≤2%）✓。
- **核数与 block 自洽**：`block_size`、`chunks`、`num_logical`、`num_kernels`、`num_local` 全编译期常量，与 §4.5 UB 上限自洽 ✓。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（源即一维 grid → NPU 一维 `T.Kernel`）；GEMM 非整除：不涉及；L0C 溢出：不涉及（纯 Vector）✓。
- §3.5.1 逐项检查本项目限制（GPU 专用 API、物理核数、host 输入约束、整数 dtype、除零角点）并给出处理方案 ✓。

### 5. 循环与同步 — pass

- 循环结构明确：§6.1 仅 block/task 级调度循环（persistent `T.Kernel` + 核内 `T.serial`），元素级全部向量化，与 §1.6.2 结论一致（kernel 内无逐元素标量循环）✓。
- §6.3 明确不用 `T.Pipelined` 并给出理由（单阶段 load→compute→store，无 K 维累加；auto multi-buffer 隐藏计算链——`CONST-copy-floor-method`）✓。
- §6.4 尾块逻辑与 §3.3/§4.4 一致 ✓。
- 同步与模式匹配：全自动同步，无跨核依赖、无 workspace、无手动 flag，与 Developer 模式一致 ✓。

### 6. 验证方案 — pass

- Golden 函数：§8.1 给出 PyTorch 参考实现（§0.1 语义直算）✓。
- 精度标准：§8.2 按 dtype 给出（fp32 `torch.equal`/atol=1e-5；fp16/bf16 atol=1.0 rtol=0.0）——与源测试 `test_floor_divide.py`（L69 `atol = 1.0 if dtype != torch.float32 else 1e-5`；L106）逐值一致；并注明本设计链 E2 实测 bit-exact，远优于容差 ✓。
- L0 测试计划：§8.3 给出 12 个代表性 case，覆盖同形/大规模/广播三模式/a 侧广播/标量/尾块非整除/整数拒绝/fp32 免恒等 cast 回归/persistent 边界/浮点角点/vdiv 广播/除数边界——shape/dtype/期望结果齐备 ✓。

### 7. 完整性与一致性 — pass

- 章节齐全：迁移类 §0（0.1–0.7 全） + §1–§11 ✓。
- 无占位符：全文检索未命中任何模板占位标记 ✓。
- 同类引用真实：§3.5.3 引用的全部路径经 `ls` 核对存在——`examples/TileOPs/tileops/kernels/elementwise/bitwise_or/bitwise_or_kernel/bitwise_or_fwd_kernel.py`、`.../lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py`、`examples/elementwise/vec_add_1d.py`、`examples/elementwise/vec_add_auto_brc.py`、`examples/bitwise_xor/bitwise_xor_fwd_kernel/DESIGN.md`、`examples/eq/eq_fwd_kernel/DESIGN.md` 均存在 ✓；`bitwise_or_fwd_kernel.py` 实读确认与 DESIGN 描述同构（persistent `T.Kernel` + `_row_base` helper + `num_kernels=min(num_logical,vector_cores)` + block_size 分解在 jit 体内 + vbrc 物化）。
- 内部一致：§3.2 API 映射、§4.3/§4.5 内存规划、§5.2/§5.5 Tiling 三处描述相互一致；§0.5 判定统计（保留 7/等价替换 3/重新设计 3/舍弃 1）与 §0.6 承接总表吻合 ✓。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性（pass）**：§1.6 含 1.6.0 调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代 / 1.6.3 向量化轴与布局决策四子节齐全 ✓。
- **算法调研存在性与完整性（pass）**：四问齐全——R1 等价化简候选表**含基线候选 0**（源算法）并评估 6 个候选（一体化 `vfloordiv`/`vfloor`、倒数乘、原生半精度、整数化、合并 cast）；R2 在线算法明确结论「无（天然单遍）」+ 结构依据（无 running 统计量/无跨元素依赖）；R3 复杂度表**四口径齐全**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）且**基线在表**（machine `r3_metrics` pass）；R4 逐候选硬件亲和评估（单元匹配/容量/对齐/静态边界/流水融合）✓；调研深度（单步逐元素 → 轻量调研）与算子复杂度匹配。「无更优替代」写明 7 层调研范围（参考表命中行 / 恒等变形自查 / 本仓 examples / pattern-library / 源结构分析 / npuir 语言层实查 / docs 全目录检索）✓。
- **调研结论正确性（独立复核，pass）**：
  - ① 等价性：选定算法（行/内层重设计 + `vdiv`+`vcast(floor)` 链）与原语义的等价性经 `verify_equiv.py` **重跑确认**（E1 索引逐点 mismatch=0、E2 三 dtype bit-exact）；半精度容差内等价风险（fp16 商先舍入）已由 E3 负向对照量化（66/430 处偏差）并据此**否决**候选 4，方向正确；
  - ② 复杂度算术复算：独立复算 R3 数字自洽——hidden-state `(8388608×3)×4B=100.7MB ÷1.05TB/s≈95.9µs`、cnn-feat `(51.4+51.4+~0)MB≈97.9µs`、row-broadcast `127.8µs`（HBM 口径）/188µs（b 重读上限口径）均一致；**中间缓冲 GM 往返已正确计为 0**（fp32 中转在 UB 内完成，无额外 GM 往返）；
  - ③ 亲和性：负向淘汰（基线逐元素 divmod/标量 gather 落错单元、候选 3/4/5/6 非等价或范围受限）依据与 `ascend-constraints` / 探测实测一致；容量/带宽淘汰引用 `CONST-*` 条目（`CONST-vector-launch-overhead` ~0.5µs/op、`CONST-copy-floor-method`、`CONST-arange-scalar-materialize` ~1ns/elem 均核对存在）；
  - ④ **负向断言主动复核**：对照 `tilelang-op-design/references/algorithm-candidates.md` 命中行 `ALG-elementwise`（R1：公共子表达式消除 / 除法乘倒数 / cast 链合并；R2：逐元素天然单遍；R3：融合消中间 GM；R4：单遍 Vector 流）——**命中行全部被 §1.6.0/§1.6.1 评估**（CSE 无公共子式；倒数乘非等价；cast 合并非等价；R2 单遍与本算子一致；R3 本体无中间 GM 往返），无被漏掉的已知候选；「无在线变体」与参考表「逐元素映射天然单遍」一致，且源算法无 online 手段（§0.4），无矛盾；
  - ⑤ 并列候选裁决：R3 中无纸面并列拍板；「一体化单指令 vs 两段链」的裕度通过探针（运行期挂死）直接定格，`vbrc 物化 vs vdiv 广播` 的裕度交 §1.6.3 备选 + Stage 4 实验裁决 ✓。
- **调研→设计一致性（pass）**：§1.4（行/内层 + vdiv/vcast 两段链）、§3.1/§3.3（以优化后公式为输入）、§6（persistent + 无逐元素标量循环）与 §1.6.0 选定算法同口径；迁移任务 §0.5/§0.6 处置与调研选族一致；§0.4 全部优化手段意图经 §0.6 末尾总表承接（**无静默丢弃**）✓。
- **数学等价优化完整性（pass）**：§1.6.1 对选定算法做 7 类系统审视（恒等变形 ×2 / 稳定化 / CSE / 算子降代换 / 访存削减 / 归约），每个采纳/否决项具「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；结论「无公式级优化空间（在可用原语约束下）」有依据（候选 1/2 非等价、候选 5 语义等价但运行期挂死、无 CSE/无归约）✓。
- **等价性论证独立推演（pass）**：逐项独立推演——(a) 倒数乘 `a·rec(b)`：`rec(b)` 舍入在整数边界邻域使 `floor` 结果偏移 ±1 → **非等价**，设计结论成立；(b) 合并 floor 与输出 cast `vcast(q→out, floor)`：大商超过 out dtype 整值精度时舍入顺序改变，与「先 fp32 floor 再 cast」不同 → **非等价**，成立；(c) 一体化 `vfloordiv`：语义等价但探针实测运行期挂死（能力缺口 CG-2026-0023 已登记，源码定义 `customize_npuir.py` L425–L436 核对存在）→ 否决成立；(d) fp32 中转链与源 `T.Cast(a.dtype, T.floor(a_f32/b_f32))` 逐位一致（floor 后为整值，回落 cast 舍入无关）→ 等价，E2 机器验证 ✓。附注（非阻塞）：§1.6.1 的 verify_equiv 表属 §0.6 R1/R4 语义保持等价替换（无 §1.6.1 采纳项），已重跑一致。
- **向量化替代覆盖完整性（与 §3.3/§6 交叉核对，pass）**：§1.6.2 盘点 9 个循环/标量点，#1–#5（除法/floor/半精度提升/输出 cast/标量广播物化）已全部向量替代且 API 有 docs+探针双佐证；#6–#9（行 base divmod / 尾块长度 / tile 调度循环 / host 元数据）为保留项，理由具体（block 级索引 / 动态边界 / tile 级调度 / host 侧），与 §3.3 伪代码、§6 循环结构交叉核对一致——kernel 内无 §1.6.2 未论证的逐元素标量循环 ✓。
- **不可替代理由充分性（pass）**：#6 引用 `CONST-arange-scalar-materialize`（逐元素物化 ~1ns/elem 反而更慢）；#7 采用 `vec_add_1d.py` 已验证 `T.min` 模式且 padding 违反 host 禁改输入；#8 为 persistent 标准结构；逐项具体到本算子，非泛泛而谈 ✓。
- **替代 API 佐证（pass）**：vdiv/vcast/vbrc 均在 `docs/Tilelang.language/` + `examples/` 找到佐证，优先 v-prefix ✓。
- **向量化轴与布局决策完整性（pass）**：§1.6.3 候选矩阵 5 项，逐元素/广播类必备的「向量化轴（innermost 广播轴）× 物化 vs vdiv 广播」两候选齐全；逐候选有量化评分（整除性 0% 尾 lane 浪费、repack=无、UB 4~7 buffer、发射项）；**弃选方案有量化理由**（#3 host 预广播负收益且违反 host 约束；#4 逐元素标量落错单元；#5 原语不可用），无「未考虑」式论证；对照了源码隐式轴（`T.Parallel(threads,npt)`）并注明「源 GPU 轴是输入非结论」✓。弃选论证前提经亲自核对：核内布局为 UB 级（无 GM 级 transpose 类矛盾）、`T.vdiv` 广播能力与文档总述一致 ✓。
- **实验裁决模式完整性（pass）**：主选 #1（进 Stage 3）+ 备选 #2（2-D `[R_blk,L]` tile，含 buffer 形状、UB 预算与 R_blk 退化条件、循环结构、dtype 路径、分核三要素按 `ceildiv(R,R_blk)×ceildiv(L,block_size)` 重算，达 Stage 4 可实现深度）+ 裁决计划（代表 shape row-broadcast/cnn-feat、指标 msprof Task Duration + vec_ratio + mte2_active_bw、判定阈值「备选 latency < 主选×0.97」、回写 §1.6.3、shape 特化工厂按 shape 分派评估）**三件套齐全** ✓。
- **布局决策与实现一致（pass）**：§1.6.3 选定 `(block_size,)` UB 布局 + coalesced 末维向量轴，与 §3.3 伪代码（buffer 形状 `(block_size,)`、vdiv/vcast 操作数即整 buffer）、§4.5 UB 预算、§6 循环结构三方一致 ✓。
- **与下游章节一致（pass）**：§3.1 以 §1.6.1 优化后公式（本体不变 = `floor(a/b)` fp32 域 + §0.6 R4 原语链 + §0.6 R1 索引表示）为唯一输入；§6 循环结构与 §1.6.2 向量化结论一致（无未论证标量循环）✓。
