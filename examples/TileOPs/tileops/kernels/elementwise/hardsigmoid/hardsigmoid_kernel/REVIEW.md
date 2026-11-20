# hardsigmoid_fwd_kernel 设计文档检视报告（Stage 2 — 第二次检视，revision v1）

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/hardsigmoid/hardsigmoid_fwd_kernel/DESIGN.md（revision v1）
- 任务类型: 迁移
- 源算子: `/mnt/workspace/TileOPs-main/src/tileops/kernels/elementwise/activations.py`（`HardsigmoidFwdKernel`）+ `_base.py` + `_builders.py`（`make_unary_regcopy`）+ `_nan.py`（`nan_min/nan_max`）+ `/mnt/workspace/TileOPs-main/src/tileops/ops/elementwise/activations.py`
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0（含 1 项已如实标注的非阻塞风险点，见维度 1/8 说明）

## 机械复核（D-1/D-5）

- verify_equiv 重跑: **PASS**。`python examples/hardsigmoid/hardsigmoid_fwd_kernel/verify_equiv.py` 输出两行 `EQUIV_PASS`，与 §1.6.1 内嵌结果表逐项一致（项 1「NaN 传播化简」max ulp 0.000 / 违反率 0；项 2「除法转乘倒数」max ulp 1.000 / 违反率 0）。执行过程无报错。
- design_calc_check: `python3 .agents/tools/design_calc_check.py --design <path>` 输出 JSON：
  - `ub_budget`: **skip**（"no per-level buffer tables parsed"）——**人工已核**：fp16/bf16 逐 buffer 求和 = 16384+32768+32768+32768+8192+0+16384 = **139264B（136KB）**；fp32 全 buffer 显式 = 32768×4+8192+32768 = **172032B（168KB）**；fp32 复用 = 32768+32768+32768+8192 = **106496B（104KB）**。三行均与 §4.5「总计」行闭合，无 v0 的混合字节口径偏差。
  - `l0c_budget`: **skip**（"no affirmative Cube-path API"）——**人工已核**：纯 Vector 算子，无 Cube/L0C，N/A。
  - `core_split`: **skip**（"insufficient parsed dims {'N': 7680}; declared logical cores = 3"）——**人工已核**：`declared logical cores = 3` 为解析伪影（§5.5 无任何「3」逻辑核数；N=7680 时 ceil(7680/8192)=1）。§5.5 分核算术独立复算自洽：逻辑核数 = ceil(N/8192)（7680→1、30720→4、1048576→128、8388608→1024）；物理核数实查 20×2=40；`num_cores = min(ceil(N/block_size), 40)`；`tiles_per_core = ceil(n_num, num_cores)`（smoke-1m→4、fc-wide→26），边界全静态。
  - `r3_metrics`: **pass**（"all four R3 metrics present"）——FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值四口径齐全。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**亲自 Read 全部 5 个源文件全文后逐项核对**：

- **语义核对**：§0.1「`y = clamp(x+3,0,6) * (1/6)`，NaN 逐层传播」与 `HardsigmoidFwdKernel.op_func`（`activations.py` L177-183）`return nan_min(nan_max(x + three, zero), six) * one_sixth` 逐字一致；常数 `three/six/zero/one_sixth` 均 `T.cast(..., "float32")` 上抛 fp32 ✓。dtype 语义「输出同输入、中间 fp32」与 `FloatUnaryKernel`（`_base.py` L500-503，`OUTPUT_DTYPE=None`）一致 ✓。边界语义「NaN→NaN、±inf 收敛到 0/6」与源 docstring（L174 "A NaN input gives NaN, as in torch"）及 `_nan.py` 实现一致 ✓。
- **算法步骤覆盖核对**：§0.3 八步（host 展平→分块装载→x+3→clamp 下界→clamp 上界→×1/6→分块写回→host 回塑）覆盖 `make_unary_regcopy`（`_builders.py` L85-121：`T.copy` fragment 装载 → `T.Parallel(threads, npt)` 逐元素 `op_func` → `T.copy` 写回）与 `UnaryKernel.forward`（`_base.py` L253-256：`flat(x)` → `_compiled_fn` → `reshape(x.shape)`）的全部计算语句，无遗漏/无臆造 ✓。host 侧逻辑（flat/reshape 视图操作）与源码一致 ✓。block_size = threads(128)×npt(4 fp32 / 8 fp16) = 512/1024 经 `_policy.py` 核对（`_DEFAULT_THREADS=128`，`register_copy_num_per_thread=4` → `_register_copy_floor` 按 elem_bytes 得 npt=4/8）成立 ✓。
- **优化手段识别核对**：§0.4 六项（register_copy / coalesced 向量化 / 除法转乘倒数 / NaN 传播 min-max / 常数 fp32 上抛 / threads×npt）逐项与源码对应（`_builders.py` L100-117、L105-110 `T.vectorized`；`op_func` `one_sixth`；`_nan.py` L56-63 fp16/bf16→`tl.max_nan/tl.min_nan`、fp32→`_propagate_nan` isnan+Select；`op_func` `T.cast fp32`；`T.Parallel(threads, npt)`）✓，硬件依赖标注正确（未把实现层优化误判为算法层）。
- **耦合性判定抽检**：§0.5 七条目四态处置（保留 4 / 等价替换 2 / 重新设计 1…实际为保留×4、等价替换×2、重新设计×2）依据充分；「重新设计」两项（threads×npt、NaN intrinsic）均有 NPU 对应方案，与 ascend-constraints 一维 Kernel / MTE / v-prefix 不矛盾 ✓。
- **重设计可行性**：§0.6 R1（一维 Kernel + block_size 折叠 + UB 向量化）与 R2（vcmp(x,x,"ne")+vselect 单点恢复）均含「源方案 → NPU 新算法 → 语义保持论证」三要素；R2 的「NaN 唯一来源是输入 x，逐层传播 ≡ 单点恢复」数学恒等成立，且如实标注「`npuir_cmp "ne"` 硬件无序语义需 Stage 3 实测标定，若证伪回退方案 (c)」——语义保持论证不虚标 ✓。
- **§1–§7 与迁移决策一致性**：§3/§5/§6 均实现 §0.5/§0.6 决策后的 NPU 形态（一维 Kernel + v-prefix 链 + vcmp+vselect），无「§0 判重设计、§3 照抄源方案」脱节 ✓。
- **golden 独立性**：§8.1 golden 以 §0.1 语义为据移植源仓参考 `F.hardsigmoid`（`x.float()`→`F.hardsigmoid`→`.to(x.dtype)`），不复刻 NPU 的 vcmp+vselect 算法，验证独立性成立 ✓。

### 1. API 可行性 — pass

**全部 API 均经 `customize_npuir.py` 源码通读 + examples 佐证，无「凭记忆猜 API」**：

- `T.vcmp(A,B,C,cmp_mod)`：`npuir_cmp`（`customize_npuir.py` L501-520）源码断言 `valid_cmp_mode = {"eq","ne","lt","gt","ge","le"}`，**`"ne"` 合法**；`__init__.py` L148 `npuir_cmp as vcmp`。examples 佐证：argmax `T.vcmp(x_work, ext_brc, cmp_eq, "eq")` / `T.vcmp(chunk_ext, running_ext, cond, "gt")`；`_gqa_prefill_fwd_kernel.py` `T.vcmp(ub_diff, K_blk, ub_cond, "le")`。A/B 参数类型 `Union[tir.Buffer, tir.Var]`，无 A==B 限制，自比较 `T.vcmp(x_f32_ub, x_f32_ub, nan_mask, "ne")` 合法（两读区无写读别名）。
- `T.vbrc(src, dst)`：`npuir_brc`（L674-699）`src` 可为标量 `PrimExpr`（`isinstance(src, tir.PrimExpr)` 分支跳过 rank 断言）；argmax `T.vbrc(T.float32(BIG), sent_v)` 标量广播先例 ✓。
- `T.vclamp(src, dst, min, max)`：`npuir_clamp`（L1128-1172）`= max(src,min) → min(·,max)` 写入 dst ✓。`T.vcast(src, dst, round_mode="rint")`：`npuir_cast`（L750）`round_mode` 含 `"rint"` ✓。`T.vselect(cond,a,b,out)`：`npuir_select`（L481）`Out = Cond ? A : B`，argmax `T.vselect(cmp_eq, idx_j, sent_v, cand)` 先例 ✓。
- `T.isnan(x, span=None)`：`tir/op.py` L2530 签名确认，入参 `PrimExpr`、返回 `PrimExpr`，**仅标量谓词**；全仓无 `visnan`/`npuir_isnan` 向量形式。§3.4 正确标「❌ 弃用」，v0 阻塞 1 已解决 ✓。
- `nan_mask` dtype `bool`：`_gqa_prefill_fwd_kernel.py` L425-426 `T.alloc_ub([half, bn_eff], "bool")` 先例佐证 ✓。`T.Kernel(一维, is_npu=True)`/`T.alloc_ub`/`T.copy` 双显式 slice：mish.py 佐证 ✓。
- v-prefix 优先：全部新 API 用 vadd/vclamp/vmul/vcast/vcmp/vselect/vbrc，npuir_xxx 仅作兼容说明 ✓。
- **非阻塞风险点（如实标注）**：`T.vcmp(x,x,"ne")` 自比较检测 NaN 依赖 IEEE 无序比较（NaN≠NaN→true），Ascend `npuir_cmp "ne"` 硬件是否按无序实现**未被 examples/文档证实**，§9.1 已如实标「需 Stage 3 NaN Boundary 探针标定，若证伪回退方案 (c)（标量 `T.isnan` 兜底）」。此为 API 存在性已证、硬件行为待实测的诚实标注，**不阻塞 Stage 2**。

### 2. 内存层级规划 — pass

- 搬运路径完整：§4.4 GM→UB→GM 两级纯 Vector，无 GM→L0 直连越级（无 Cube）✓。
- **UB 预算逐 buffer 求和闭合（独立复算）**：fp16/bf16 = **139264B(136KB)**、fp32 全 buffer = **172032B(168KB)**、fp32 复用 = **106496B(104KB)**，均与 §4.5「总计」行一致；`y_ub` fp16/fp32 尺寸（16384/32768）不再张冠李戴，v0 阻塞 2 已解决 ✓。
- 上偏系数 ×1.12 正确引用：constants.md `CONST-capacity-910B2C`「auto-multi-buffer=false 下 BishengIR 实际需求 ≈ 手工预算 ×1.10–1.12」（且显式 alloc 结构实测 ratio 0.981）；本设计 Stage 3 基线用 `T.serial`（非 `T.Pipelined`、非 auto-multi-buffer），×1.12 上偏适用。fp32 全 buffer 168KB×1.12≈188KB<192KB（余量 ~2%），复用后 ≤136KB×1.12≈152KB（余量 ~21%）→ 104KB×1.12≈116KB（余量 ~39%），均不溢出 ✓。
- 与 mish 参考一致：`perf_opt/mish.py` 以 block_size=8192 编译实测（fp16 128KB / fp32 96KB），本算子 fp16 136KB（多 8KB bool nan_mask）/fp32 复用 104KB 落在同款已验证包络内 ✓。
- Buffer 层级标注：§4.3 全 buffer 标注 UB，`nan_val` 复用 `buf2_ub`（0 新增驻留）✓。
- §4.5 与 §1.6.0 R4（~136-168KB）、§5.3、§9.1 数字一致 ✓。

### 3. Tiling 策略 — pass

- Block 划分：§5.2 block_size=8192（承接 mish 同族 Stage 4 实测，32B 对齐）✓。
- 分核策略三要素齐全（§5.5，对照 core-split-strategy.md）：① 逻辑核数 `num_logical_kernels = ceil(N/block_size)` ✓；② 物理核数实查 `NPUUtils.get().get_aicore_num()` → 20（纯 Vector ×2 = 40），查询代码与返回值已记录，与 constants.md 的 24 分歧已注明「以实查为准」✓；③ 规模判定三档 + 分核方案 `num_cores = min(ceil(N/block_size), 40)`、核内 `T.serial(tiles_per_core)`，`tiles_per_core` 为编译期常量（N/block_size/40 均编译期已知），边界静态 ✓。
- 对齐约束：8192 元素 × 2/4B 均 32B 整数倍 ✓。
- 非整除尾块：`tail_size = T.min(block_size, remaining)` + `block_id < n_num` guard，仅 copy 有效元素 ✓。

### 4. 技术约束检测 — pass

- 无三维 Kernel（一维 `T.Kernel(num_cores, is_npu=True)`）；无 GEMM 非整除；无 L0C 溢出；GPU 专用 API（`T.get_thread_binding`/`tl.max_nan`/`tl.min_nan`/`T.vectorized`/`T.unroll`）全部改写为 NPU v-prefix 路径；分核策略已在 §5.5 处理 ✓。

### 5. 循环与同步 — pass

- §6.1 循环结构明确：block 级 `T.Kernel`、tile 级 `T.serial`、元素级 v-prefix（无逐元素标量循环）✓。§7 Developer 自动同步，无手动 set_flag/wait_flag，与编程模式匹配 ✓。§6.4 尾块逻辑完整 ✓。

### 6. 验证方案 — pass

- §8.1 golden = `F.hardsigmoid`（fp32 opmath + 回 cast），移植源仓 `ref_program`，独立于 NPU 算法 ✓。§8.2 精度标准按 dtype（fp16 1e-3/1e-3、fp32 1e-5/1e-5、bf16 1.6e-2/1.6e-2，与 mish 实测一致，严于 skill 默认）✓。§8.3 L0 门槛计划（N=1048576 三 dtype vs golden assert_close）+ Boundary NaN 用例（验证 vcmp "ne" 单点恢复）✓。

### 7. 完整性与一致性 — pass

- 章节齐全：11 章 + §0（0.1–0.7 小节全）+ §0.0 修订说明 ✓。全文无占位符/未完成标记 ✓。§3.5.3 引用的 mish/lerp 路径真实存在（已 Glob 核对）✓。内部一致：§1.6.0 选定算法 ↔ §1.4 ↔ §3.1 公式拆解 ↔ §6 循环结构 ↔ §4.5 UB 预算数字一致，无矛盾 ✓。

### 8. 算法优化分析 — pass

- **§1.6 章节存在**：含 1.6.0（调研）/1.6.1（等价优化）/1.6.2（向量化替代）/1.6.3（向量化轴与布局）四子节 ✓。
- **调研存在性与完整性**：§1.6.0 四问齐全。R1 候选表含基线候选（表 #0）+ relu6 原生式 + 除法转乘倒数 + 恒等变形，对照 `algorithm-candidates.md` `ALG-elementwise` 命中行 R1（CSE/除法乘倒数/cast 链合并）均已评估（除法乘倒数=候选 2 已含基线；CSE「无公共子表达式」与 cast 链合并「无 cast 链冗余」在结论句给出依据）✓。R2「逐元素映射天然单遍」明确结论，结构判据与 `algorithm-research.md` §3 R2 判据一致 ✓。R3 四口径表（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值，基线在表）✓。R4 逐候选硬件亲和（单元匹配/容量/对齐/静态边界/流水融合）✓。
- **调研结论正确性（独立复核）**：① 等价性独立推演成立（NaN 唯一来源是输入 x，单点恢复 ≡ 逐层传播）；② 复杂度独立复算（3N FLOPs、2N·elem_bytes 访存、1 遍、O(block) 缓冲，无中间 GM 往返漏计）✓；③ 亲和判断与 `ALG-elementwise` R4「单遍 Vector 流」、`PL-1.6-copy-floor`/`PL-1.17` 不矛盾 ✓；④ **负向断言复核**：「无在线变体」与结构判据（逐元素无跨元素依赖）一致，无漏掉已知候选（hardsigmoid 无 softmax/Welford/flash 类在线变体）；「无更优化简公式」经恒等变形自查（候选 3 反而更差）成立 ✓；⑤ R3 无并列争议（四口径全同）✓。
- **调研→设计一致性**：§1.4/§3.1/§6 描述的「逐元素 `clamp(x+3,0,6)*(1/6)` + fp32 中转 + 单点 NaN 恢复」与 §1.6.0 选定算法一致；迁移任务 §0.5/§0.6 处置与调研选族一致，源优化手段意图承接（§0.6 尾部 R1/R2/寄存器/coalesced/除法倒数/fp32 上抛逐项承接说明）无静默丢弃 ✓。
- **数学等价优化完整性**：§1.6.1 采纳 2 项，均有「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素 ✓。
- **等价性论证独立推演**：项 1（NaN 逐层传播→单点恢复）数学恒等成立（x 有限两式同值，x=NaN 两式均 NaN）；项 2（`/6`→`×(1/6)`）容差内等价（≤1 ulp，与源公式一致）。verify_equiv.py 重跑结果与内嵌表一致且全 EQUIV_PASS ✓。
- **向量化替代覆盖完整性（与 §3.3/§6 交叉核对）**：§1.6.2 七点覆盖全部计算点（x+3/clamp/×1/6/cast/NaN 掩码/block 索引/尾块），§3.3 伪代码与 §6 循环结构无任何未覆盖的逐元素标量循环 ✓。
- **不可替代理由充分性**：block 起始索引（block 级任务映射）与尾块大小（动态边界）两条理由具体到本算子，非「实现简单」泛泛之词 ✓。
- **替代 API 佐证**：vadd/vclamp/vmul/vcast/vcmp/vselect/vbrc 均有 `customize_npuir.py` 源码 + examples（argmax/gqa/mish）佐证，优先 v-prefix ✓。v0「凭记忆猜 API」问题已根除。
- **向量化轴与布局决策完整性**：§1.6.3 单轴豁免说明成立（1-D `(N,)` 仅一个非 batch 维，无 ≥2 轴可选；无广播/窗口/归约轴/重排候选），符合「单轴算子 1.6.3 须有豁免说明」要求；布局决策（I/O 原生 1-D + 展平轴向量化）与 §3.3/§4/§6 三方一致（buffer 全 1-D、向量维=展平轴）✓。
- **与 §3.1/§6 一致性**：§3.1 公式拆解以 §1.6.1 优化后公式为输入（单点 NaN 恢复为 step 5）；§6 无逐元素标量循环 ✓。
- **非阻塞风险点（如实标注）**：`vcmp "ne"` 的 NaN 无序比较语义属「替代 API 已存在、硬件行为待 Stage 3 实测」的诚实风险标注（§9.1 + 回退方案 (c)），不属「等价性论证不成立」或「替代 API 不存在」，**不阻塞 Stage 2**。

---

## 检视总结

v1 已解决 v0 全部 2 项阻塞级问题（阻塞 1：`T.isnan(x,dst)` → `T.vcmp(x,x,"ne")`+`T.vselect`，`T.isnan` 标「仅标量谓词弃用」；阻塞 2：fp32 UB 预算 155648→172032，逐 buffer 求和闭合）与 1 项建议（§3.3 `nan_scalar` let 绑定 + `T.vbrc` 复用死 buffer）。全部 API 均经 `customize_npuir.py` 源码 + examples 佐证，无「凭记忆猜 API」。UB 预算三 dtype 路径闭合正确，×1.12 系数正确引用 constants.md 且与 mish 同族实测包络一致。算法调研四问齐全、负向断言经参考表复核成立、等价性机器验证通过、向量化覆盖完整。仅余 `vcmp "ne"` 的 NaN 无序比较硬件语义一项非阻塞风险，已如实标注为 Stage 3 探针标定项并给出回退方案，符合诚实风险管理规范。
