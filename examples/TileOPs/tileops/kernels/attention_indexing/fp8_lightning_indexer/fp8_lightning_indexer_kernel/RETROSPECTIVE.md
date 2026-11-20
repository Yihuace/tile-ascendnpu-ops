# RETROSPECTIVE — fp8_lightning_indexer / _fp8_lightning_indexer_kernel

> Stage 1 复盘（`first_design`，迁移 harness）。每次 DESIGN.md 产出后**追加**本节，历史章节不覆盖。

## Stage 1 — 2026-10-15（first_design）

- **task_id**: `fp8_lightning_indexer-_fp8_lightning_indexer_kernel-20261015T000100Z`
- **scenario**: migration / harness ｜ **mode**: first_design ｜ **编程模式**: Developer
- **toolchain_stamp**: tilelang `0.1.2+a2357253662babf317ada194080078f928d155b3`（commit 013dbbf, dev root build）/ CANN 9.0.0 / Ascend910B3（`get_aicore_num()=20`, `get_aivector_core_num()=40`）/ torch_npu 2.7.1.post4 / torch 2.7.1+cpu
- **产出**：`DESIGN.md`（885 行）、`verify_equiv.py`（`ALL_GATE_ITEMS EQUIV_PASS`）、capability gap **CG-2026-0022**

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|------|-------|----------|----------------------|---------|
| 迁移脚手架 / FP8 可行性 | 脚手架将 `dtype = T.float8_e4m3fn` 原样保留为「FP8 path preserved (O6)」，但 npuir 前端**根本没有 `T.float8_e4m3fn`**（`dir(T)` 无该属性），规范名为字符串 `"e4m3_float8"`；且 codegen `DTypetoMLIRType` 直接拒绝该 dtype。脚手架注释暗示 fp8「保留即可用」是误导 | `hasattr(T,'float8_e4m3fn')==False`；最小 npuir gemm `in_dtype="e4m3_float8"` → `TVMError: Cannot convert type e4m3_float8`（src/target/codegen_npuir_api.cc:593）；`tilelang/utils/tensor.py` L26 用字符串名 | 迁移 harness 在抽取期对源 `dtype=` 做**目标前端可达性预检**（对比 `dir(T)` + docs dtype 矩阵），命中不可达 dtype 时在脚手架注释里显式标注「需重设计/能力缺口」，而非「preserved」 | R（流程规则） |
| 算法调研 / 负向断言 | `T.gemm`/`T.vcast` 的 dtype 支持表分散在各自 §2.2.1，但「fp8 全链路不可用」这条跨 API + codegen + runtime 的结论没有单一索引；本次靠逐一枚举 16 个子目录 + codegen 探针才收敛 | `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.2.1、`数据类型转换操作/T.vcast.md` §2.2.1；本任务 codegen 探针 | 在 `ascend-constraints.md` 增加「dtype 支持矩阵」小节（跨 T.gemm/T.vcast/v-op/alloc 汇总 fp8/bf16/fp16/fp32 可达性），标注版本戳 | P（模式方法） |
| 分核策略 | 「无法对齐兜底」三要素判定在 `core-split-strategy.md` §1 要素③已定义，但**未点明 Developer+gemm 的核心结构会使核内串行 persistent 不可用**（该结论在 CG-2026-0010）——设计者需自行串联两份文档 | CG-2026-0010（Developer persistent 混排崩溃）；`core-split-strategy.md` §1 要素③ | 在 `core-split-strategy.md` §1 要素③「无法对齐兜底」处加一句交叉引用：`Developer+gemm` 结构下核内串行不可用（见 CG-2026-0010），须走兜底并记录残留不均 | R（流程规则） |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|-------|---------|----------|-------|-----------------|------------|
| npuir 全链路不支持 FP8（e4m3/e5m2）：前端 dtype 名缺失 + codegen `Cannot convert type e4m3_float8` + `T.gemm`/`T.vcast` dtype 矩阵无 fp8 + torch_npu 设备端 fp8 cast 失败（561103）——NPU 「fp8 GEMM」实为 fp16 载体 | D（实测） | 本任务 §0.5 条目 2 / §3.5.1 / CG-2026-0022；codegen 探针 `TVMError`；`examples/deepseek_v32/fp8_lighting_indexer.py`（`dtype="float16"`）、`examples/deepseek_v4/example_fp8_gemm_kernel.py` L35/L184（fp16 carriers） | 最小 npuir gemm `in_dtype="e4m3_float8"` 编译复现；`torch_npu` fp8 cast 探针（见 CG-2026-0022 evidence） | tilelang `0.1.2+a2357253662babf317ada194080078f928d155b3` / CANN 9.0.0 / Ascend910B3 / 2026-10-15 | CG-2026-0022（已登记）；pattern-library `traps-compiler.md`（建议新增 dtype 可达性条目） |
| **E1：scale 提出 head 归约**（`Σ(relu·w)·s` → `s·Σ(relu·w)`）——代数恒等，fp64 有理数精确验证 + fp32 契约域全过；收益 = 每次 S 元素少 1 次乘（本例 ≈2.0e9 elem-mul ≈ −1.2ms，占发射项 ≈32%）。NPU 同族先例（`deepseek_v32/fp8_lighting_indexer.py` L143/L146）即此形态 | D/P（实测+模式） | 本任务 §1.6.1 + `verify_equiv.py`（L1 200 例有理数 diff=0；L2 13/13 EQUIV_PASS）；`examples/deepseek_v32/fp8_lighting_indexer.py` L143/L146 | `python examples/fp8_lightning_indexer/_fp8_lightning_indexer_kernel/verify_equiv.py` | tilelang `0.1.2+a2357253662babf317ada194080078f928d155b3` / 2026-10-15（数值层 torch CPU，设备无关） | `algorithm-candidates.md`（ALG-attention/ALG-reduction 候选：scale/常量因子提出归约）；pattern-library `elementwise.md` / `attention.md` |
| **NPU lightning indexer 的可移植结构**：Developer 非持久 1D grid + Cube `T.gemm(b_transpose=True)` + **L0C→UB 直拷**（`T.copy(acc_frag, acc_ub)`，免 Expert `store_fixpipe` 的 GM workspace 往返）+ `T.vrelu`/`T.vmul`(行因子)/逐行 `T.reduce_sum(dim=0)` + `T.vbrc`(列因子 scale) | P（模式） | 本任务 §1.4/§3.3/§4.4；先例 `examples/indexer/indexer_fwd.py` L68–L80（同形） | 以 `indexer_fwd.py` 为最小形态 | tilelang `0.1.2+a2357253662babf317ada194080078f928d155b3` / Ascend910B3 / 2026-10-15 | pattern-library `attention.md`（新增 indexer 结构条目）/ `cases.md` |

### Transferable Lessons（可迁移教训）

1. **「保留 fp8」类脚手架注释须先做目标前端可达性预检**：`dtype=` 的可达性不能靠 GPU 直觉判断（`T.float8_e4m3fn` 在 npuir 不存在，且 codegen 会拒绝）——迁移第一步应 `python -c "import tilelang.language as T; print(hasattr(T,'<dtype>'))"` + 最小 kernel 编译探针。
2. **fp8 结论是「三层」的，缺任一层证据都不足**：前端 dtype 名 / codegen MLIR 类型 / runtime aclnn——本任务三层独立取证（避免把 codegen 限制误当 runtime 限制，或反之）。
3. **数学等价优化的收益要先算清「被优化的 op 在总量中的占比」**：本次 E1 的收益只有在把 §1.6.0 复杂度四口径（尤其发射项 ≈S 矩阵元素数）算出来后才看得出（−1.2ms）；否则容易因「少几次乘」而低估。
4. **代数恒等的机器验证用「精确有理数 + 契约域」双层**：单层 fp64 ulp 在存在抵消（cancellation）的归约里会给出失真的巨大 ulp（本例 fp64 ulp 曾报 1.3e4）；**精确 `fractions.Fraction` 层**给出 diff=0 的确定性结论，fp32 契约域层（corr/violation）给出容差结论。
5. **负向候选要留「机器诊断」证据**：折进 K（`relu(s·Sq)`）在 `s<0` 时 `corr_diff=0.997`——诊断表比散文「不成立」更有说服力（同 PL-1.27 的 verify_equiv 非门禁诊断形态）。
6. **无 manifest workload 的分支（如 `kv_group>1`）要显式标注未实测**，并在 L0 测试计划单列用例——避免「设计覆盖了但从未验过」的静默空洞。
7. **常量表的设备绑定要逐条核对**：`constants.md` 只有 `910B2C` / `910_9362` 两族，本机实为 `Ascend910B3`（核数 20 与 9362 同值）——引用任何带宽/发射常数都须标注「未实测假设 + 估算依据」并交 Stage 4 回填。

## Stage 2 (review) Retrospective — first_review — 2026-10-15

- **task_id**: `fp8_lightning_indexer-_fp8_lightning_indexer_kernel-20261015T000100Z` ｜ **结论**: 通过（阻塞级 0 / 建议级 5）
- **检视范围**: 迁移 9 维度（0–8）+ 机械复核（`verify_equiv.py` 重跑 EQUIV_PASS + `design_calc_check.py`）
- **独立复核亲手执行**: 源算子源码全读、`docs/Tilelang.language/` 全 16 子目录 fp8 枚举、最小 npuir fp8 gemm 编译探针、
  torch_npu fp8 cast 探针、`NPUUtils` 物理核实查、pattern-library 7 条 + queue 3 条引用核对

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 本设计的 `design_calc_check.py` 机械复核 **`ub_budget` 与 `core_split` 双 skip**（`no per-level buffer tables parsed` / `insufficient parsed dims {'M':128,'bn':64}; declared logical cores=32768`——解析口径产物非设计声明值），机械覆盖近乎归零，reviewer 须手工重算 UB 求和与逻辑核数 | `design_calc_check.py` JSON（本任务）；DESIGN §4.5/§5.5 | 检查器识别 DESIGN §4.5 表列名（`Buffer/Shape/dtype/大小`）与 §5.5 `num_logical_kernels` 声明式；或在设计模板里固定预算表列标题使解析稳定 | R |
| Info-source | 设计引用的带宽/发射常数（`CONST-ascend910-9362-profile` / `CONST-vector-op-cost-910_9362`）**均为 9362/910B2C 绑定**，`constants.md` **无 910B3 条目**；reviewer 无法独立核验 D-2 估算数字，只能接受「未实测假设 + 估算依据」的自述 | `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`（无 910B3）；DESIGN §1.6.0/§9.1 | constants.md 增 910B3 行（或明确标注「910B3 未实测，沿用 9362 值需回填」并在 Stage 4 强制对账） | R |
| Constraint-doc | `CONST-capacity-910B2C` 的 UB 膨胀系数语义（**默认 auto-multi-buffer 开 = ~1.7x；`auto-multi-buffer=false` = ×1.10–1.12**）在正文散述，设计 skill 的 UB 预算指引未把系数绑定「auto-multi-buffer 状态」——本设计即写成「auto-multi-buffer 膨胀 ≈1.10」 | `constants.md` CONST-capacity-910B2C 段；DESIGN §4.5 | `tilelang-op-design` UB 预算指引注明：系数取值须绑定 `--enable-auto-multi-buffer` 状态（默认 True → 1.7x） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 负向能力断言（如「dtype 全链路不可用」）的**三层实跑复核法**：①前端 `hasattr(T,'<name>')` + `tilelang/utils/tensor.py` 规范名；②最小 npuir kernel 编译探针取 codegen 报错；③设备侧 torch_npu 生成/cast 探针取 aclnn 错误码。三层独立、任一层即可证伪 | P（方法） | 本任务 REVIEW.md 维度 8「FP8 能力缺口专项复核」段；CG-2026-0022 evidence | `hasattr(T,'float8_e4m3fn')`→False；最小 gemm `in_dtype="e4m3_float8"`→`Cannot convert type e4m3_float8`；`torch.randn(...,'npu').to(fp8)`→`aclnnInplaceCopy 561103` | tilelang `0.1.2+a2357253662babf317ada194080078f928d155b3` / CANN 9.0.0 / Ascend910B3 / 2026-10-15 | `negative-claim-evidence.md` §3（reviewer 复核要求）/ pattern-library `traps-compiler.md` |
| CG-2026-0022 证据精度订正：`aclnnInplaceZero` 实测错误码为 **161002**（非条目归并书写的 561103，561103 实测对应 `aclnnInplaceCopy`）；`torch.empty(fp8,npu)` 实测可用（与条目一致） | D（实测） | 本任务 REVIEW.md 维度 8 备注；`.agents/evolution/capability-gaps.md` CG-2026-0022 evidence | 上条 repro 的 torch_npu 探针 | 同上 | `.agents/evolution/capability-gaps.md` CG-2026-0022（订正错误码） |

### Transferable Lessons（可迁移教训）

- **npui 前端无 `T.float8_e4m3fn`**（规范名 `"e4m3_float8"`），且 codegen `DTypetoMLIRType` 仅映射 float 16/32/64 → fp8 连 MLIR 类型都无法表示；任何 fp8 源算子迁移本任务后续函数（如 `clean_logits_` 若涉 fp8）都必须先换 bf16/fp16 载体。
- **`T.vrelu`/`T.vmul`/`T.reduce` 的 bf16 均不支持**（各 API §2.2.1）——载体会放 bf16 的算子，epilogue 必须落 fp32（载体仅用于 gemm 输入）。
- **Developer 模式 `T.copy` 支持 L0C fragment↔UB 直拷**（`T.copy.md` §1 方向表「UB -> UB（含 L0C fragment 与 UB 之间的局部拷贝）」）——混合算子优先走此路径，免 Expert `store_fixpipe` 的 GM workspace 往返。
- 复核负向结论时**不要只读设计文档的转述**：本任务的 FP8 结论三层实跑全部复现（dir()/codegen 编译/torch_npu），也正因实跑才发现 CG 条目的错误码精度瑕疵。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-15

- **task_id**: `fp8_lightning_indexer-_fp8_lightning_indexer_kernel-20261015T000100Z` ｜ **mode**: first_impl ｜ **编程模式**: Developer（`ASCEND_MODE="Developer"`，非持久 1D grid）｜ **attempt**: 1
- **toolchain_stamp**: tilelang `0.1.2+c144c4b2aaefb9eeca259cd438fc4cfed69c77c6`（dev root build）/ CANN 9.0.0 / Ascend910B3（`get_aicore_num()=20`）/ torch 2.7.1+cpu / torch_npu 2.7.1.post4 / 2026-10-15
- **产出**: `_fp8_lightning_indexer_kernel.py`（kernel + 内嵌 CPU golden + L0/L1/L2/Boundary + main）；判定 `[PRECISION_PASS]`（L0 3/3、L1 4/4、L2 1 warn、Boundary 3/3，corr_diff ~3e-15）
- **探针转正**: `repro/TRAP-copy-size-4d-extent-misalign.py`、`repro/TRAP-scalar-min-chain-offset.py`（均已实跑确认断言成立）

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|------|-------|----------|----------------------|---------|
| Design-flow / Develop-flow | DESIGN §3.3 伪代码在 ≥3-D 张量上用 `T.copy(dst, Logits[by, s, k0, g], size=[block_Q, block_N])`（K 装载同形 `size=[block_N, index_dim]`）——npuir 前端对索引数多于 `size` 长度的操作数**前置补 1**（`tilelang/language/copy.py:56-64`），extent 落到**尾维**，静默写错区域。设计把该形态当合法写法 | DESIGN.md §3.3 L446/L461；`repro/TRAP-copy-size-4d-extent-misalign.py`（slice 误差 0，size 形态误差 3.85） | `tilelang-op-design` 的 T.copy API 小节 + design-template 伪代码示例：≥3-D 操作数一律用**切片语法**（`A[b, s:s+M, k:k+N, g]`）表达区域；`size=` 仅用于 2-D 或 extent 恰为尾维的场合 | R（流程规则） |
| Design-flow / Develop-flow | DESIGN §3.3/§6.2 的窗口并集（`cu_k_s_min`/`cu_k_e_max`）用 `T.alloc_var` + `T.serial` 的 min/max **累加链**，再作 DMA 拷贝偏移。该形态在本工具链 **mis-compile**：偏移值错误→设备 `MPU address access is invalid`(507015) 或 NaN。与数据来源无关（GM 读、UB 读、甚至常量链均复现） | DESIGN.md §3.3 L433-435；`repro/TRAP-scalar-min-chain-offset.py`（chain 误差 9.99e1，baseline/one-shot 1.5e-5） | `tilelang-op-design` 在「动态边界/块对齐」小节加负向条目：**不要**把多轮标量 min/max 累加链的结果用作 `T.copy` 偏移；给出对齐扫描 + 单次 `T.min` clamp 的替代形态（本函数即此形态） | R（流程规则） |
| Develop-flow | TVMScript 把 Python 变量在 Python `if/else` 分支内赋值、分支外使用判为 `Undefined variable`（parser 静态作用域检查），即使条件是编译期 Python 常量。design-template 的 `if has_scale:` 形态若分支内赋值会直接报错 | 本任务探针（probe17/probe18：`k0b = ... if/else` → `error: Undefined variable: k0b`）；DESIGN §3.3 `has_scale` 分支（本函数实现改为无跨分支变量赋值的写法） | develop skill 的 kernel 模板加注：编译期开关用 Python 三元表达式或把分支放在不跨分支共享变量的位置；不要依赖 `if <py-const>:` 内赋值 | P（模式方法） |
| Constraint-doc | `CONST-capacity-910B2C` 的 UB 系数口径（默认 auto-multi-buffer ~1.7x）在本函数得到侧面印证：L0/L1 的 G=1 主选 tile 全部编译通过，但 **`kv_group>=2` 且 `Hp>=8`** 的组合在 BiShengIR 报 `ub overflow, requires 276KB while 192KB available`（即使手工 buffer 求和仅 ~50KB），说明分组路径的实际 UB 占用远超手工预算 | `_fp8_lightning_indexer_kernel.py` L2-g4 warn；本任务 `run --level all` 日志 | UB 预算指引：多分组/多拷贝形态不能按 buffer 形状手工求和，须以编译器实测占用为准（Stage 3 UB 探针 / `ub overflow, requires N bits` 反推） | R（流程规则） |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|-------|---------|----------|-------|-----------------|------------|
| **npuir `T.copy(..., size=[...])` 在 ≥3-D 操作数上前置补 1 导致 extent 落尾维 → 静默错区域**；修法 = ≥3-D 一律用切片语法 | D（实测） | `tilelang/language/copy.py:56-64`（prepend ones）；本任务 DEVELOPMENT §3.3 陷阱；`repro/TRAP-copy-size-4d-extent-misalign.py` | `python examples/fp8_lightning_indexer/_fp8_lightning_indexer_kernel/repro/TRAP-copy-size-4d-extent-misalign.py` | tilelang `0.1.2+c144c4b2aaefb9eeca259cd438fc4cfed69c77c6` / CANN 9.0.0 / Ascend910B3 / 2026-10-15 | pattern-library `traps-runtime.md`（新增 copy extent 条目）；`docs/Tilelang.language/内存操作/T.copy.md` §2.2 |
| **标量 min/max 累加链结果作 DMA 拷贝偏移 mis-compile**（设备 MPU OOB / NaN）；与数据源（GM/UB/常量）无关；修法 = 对齐扫描 + 单次 `T.min` clamp 或核外预计算 | D（实测） | `repro/TRAP-scalar-min-chain-offset.py`（chain 9.99e1 vs one-shot 1.5e-5）；DEV 期 probe19/20/21 | `python examples/fp8_lightning_indexer/_fp8_lightning_indexer_kernel/repro/TRAP-scalar-min-chain-offset.py` | tilelang `0.1.2+c144c4b2aaefb9eeca259cd438fc4cfed69c77c6` / CANN 9.0.0 / Ascend910B3 / 2026-10-15 | pattern-library `traps-runtime.md` / `traps-compiler.md`；`docs/Tilelang算子调试指南.md` |
| **Developer 非持久 1D grid 的 lightning-indexer 可用结构**（实测通过）：`T.gemm(q_ub,k_ub,acc,b_transpose=True,initC=True)`（acc `[bq*Hp,bn]`）+ `T.copy(acc,acc_ub)`（L0C→UB 直拷）+ `T.vrelu` + `T.reshape(w2_ub→w_ub)` 行因子 `T.vmul` + 逐行 `T.reduce_sum(dim=0)` + `T.vbrc(scale[1,bn]→[bq,bn])` + 列因子 `T.vmul` + 4-D **切片**写回 | D/P（实测） | `_fp8_lightning_indexer_kernel.py` L103-190；L0/L1 全过（corr_diff ~3e-15）；先例 `examples/indexer/indexer_fwd.py` L68-L80 | `python examples/fp8_lightning_indexer/_fp8_lightning_indexer_kernel/_fp8_lightning_indexer_kernel.py --level all` | tilelang `0.1.2+c144c4b2aaefb9eeca259cd438fc4cfed69c77c6` / Ascend910B3 / 2026-10-15 | pattern-library `attention.md`（indexer 结构条目）/ `cases.md` |
| **`T.reshape` 是 UB 上的零拷贝视图**，可用于把行权重 `[bq,Hp]` 展成 `[bq*Hp,1]` 以匹配 `T.vmul` 文档化行因子形 `[M,N]*[M,1]`（列因子 `[1,N]` 无直乘形，须 `T.vbrc` 展开） | D（实测） | `_fp8_lightning_indexer_kernel.py`（`T.reshape(w2_ub, w_ub)` 后 vmul 精度 3e-15）；`docs/Tilelang.language/shape操作/T.reshape.md` §2.3 | 同上一行 L0 | tilelang `0.1.2+c144c4b2aaefb9eeca259cd438fc4cfed69c77c6` / Ascend910B3 / 2026-10-15 | pattern-library `elementwise.md`；`T.vmul.md` §2.2.2 补充行列因子展开法 |
| **分组（kv_group≥2）路径的编译容量上限**：`Hp>=8` 且 `kv_group>=2` 时 BiShengIR `ub overflow`（手工预算 ~50KB 但报需 276KB），`Hp<=4` 可过；manifest 无 kv_group>1 workload，标为受限 | D（实测） | `run_case(1,1024,32,64,1024,4)` → `ub overflow, requires 2260992 bits`；`h8g2/h8g4` 通过 | `python examples/fp8_lightning_indexer/_fp8_lightning_indexer_kernel/_fp8_lightning_indexer_kernel.py --level all`（L2-g4 WARN） | tilelang `0.1.2+c144c4b2aaefb9eeca259cd438fc4cfed69c77c6` / Ascend910B3 / 2026-10-15 | `.agents/evolution/capability-gaps.md`（候选条目）/ pattern-library `traps-compiler.md` |

### Transferable Lessons（可迁移教训）

- **npuir `T.copy`：≥3-D 操作数一律用切片语法**（`A[b, s:s+M, k:k+N, g]`）。`size=[M,N]` 会被前置补 1 成 `[1,1,M,N]` 命中尾维，静默写错/读错区域——本迁移任务后续函数（如 `clean_logits_`，同样操作 4-D `Logits`）必须沿用切片写法。
- **不要把「多轮标量 min/max 累加」的结果用作 DMA 拷贝偏移**：本工具链会 mis-compile（设备 MPU OOB 或 NaN）。等价替代 = 对齐到块边界的扫描 + 单次 `T.min(nbn*block, limit)` clamp（overlap 重算同值），或把窗口端点移到 host/元数据张量；迁移带「动态窗口并集」的后续函数（`clean_logits_` 的 `[ks,ke)`）须先评估此坑。
- **TVMScript 里 Python 变量不能在 Python `if` 分支内赋值后在分支外引用**（报 `Undefined variable`）；编译期开关请用三元表达式，或让分支只含不跨分支共享的语句。
- **epilogue 一律 fp32**：`T.vrelu`/`T.vmul`/`T.reduce_sum` 的 bf16 均不支持；bf16/fp16 仅作 gemm 输入载体（本函数 L0 的 corr_diff ~3e-15 佐证 bf16 载体 + fp32 epilogue 精度充足）。
- **行因子广播用 `[M,N]*[M,1]`（可经 `T.reshape` 从 `[bq,Hp]` 得到），列因子必须先 `T.vbrc` 展开为 `[M,N]` 再 `T.vmul`**——`[M,N]*[1,N]` 无文档化形，M==N 时会静默算错。
- **UB 预算对「多分组/多拷贝」形态不可按 buffer 形状手工求和**：kv_group>1 的 `Hp>=8` 组合实际触发编译器 ub overflow（手工 ~50KB，实报 276KB）；写多分组 kernel 时用编译报错反推实测占用，勿贴限设计。

