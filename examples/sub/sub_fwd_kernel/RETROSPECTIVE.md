# RETROSPECTIVE — SubFwdOp（`sub_fwd_kernel`，migration-harness）

## Stage 1 (design) Retrospective — first_design — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 同族 `AlphaScaledBinaryKernel` 迁移直接复用 Add 设计效率最高——Sub 与 Add 仅 `_combine` 由 `+` 换 `−`，其余（广播 Op 层展平、fp32 中转、`alpha==1` 折叠、分核 persistent、UB 预算）逐字可承接；复用时须逐点核对减法特有项（次序不可交换、`0*inf` 语义、torch golden 分支） | DESIGN.md §0–§11（本体由 `examples/add/add_fwd_kernel/DESIGN.md` 结构改写）；差异见 §0.1/§0.6 R2/§1.6.1 | 迁移 skill 可在「同族已完成」时提供「同族骨架复用清单（复用字段 + 必须重核的算子特有项）」 | R |
| Design-flow | 减法算子的等价性机器验证暴露出**加法不显现**的 `opmath` 域分歧：`fp32(alpha)*fp32(b)` 中间溢出时（`a=b=±max, |alpha|≥2`），两轮 fp32 链给 `±inf`，而 torch CPU alpha 路径保留扩展精度给有限 `±max`——若照抄 Add 的 verify 脚本（仅比 golden）会出现 EQUIV_FAIL | `examples/sub/sub_fwd_kernel/verify_equiv.py`（初版 4 行 EQUIV_FAIL/135，均 `(4096,)`/`(64,128)×(1,128)` dtype∈{fp32,bf16} alpha=2）；DESIGN §1.6.1 表 | design skill 的 D-1 验证模板宜提示「中间量可能溢出目标域」的算子（减法/带 alpha 缩放类）须按「正常域 vs 中间溢出子域」分域对照，而非单一 golden 对照 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `torch.sub(a,b,alpha)` 的 CPU alpha 路径保留扩展精度中间值：`fp32(alpha)*fp32(b)` 溢出 fp32 时（`a=b=±max,|alpha|≥2`）golden 给有限 `±max`，而两轮 fp32 链（`vmul→vsub`）与源 GPU 算法同为 `±inf`；加法因中间溢出与结果溢出重合而不显现该分歧。sub 类迁移的 golden 对照须按「正常域（候选 vs golden，≤0.0148 tol-unit）+ 中间溢出子域（候选 vs 源算法，逐位）」分域验证 | C | examples/sub/sub_fwd_kernel/DESIGN.md §1.6.1（EQUIV 表 + 中间溢出子域说明）；examples/sub/sub_fwd_kernel/verify_equiv.py | `python3 examples/sub/sub_fwd_kernel/verify_equiv.py`（135 行全通过；中间溢出元素 5127，0 违反） | tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B2C（实查 20 AI Core → 40）/ torch 2.7.1+cpu / 2026-10-13 | pattern-library/traps-runtime.md（数值口径类，与 TRAP-fp16-opmath-golden 同族） |
| 同族 `AlphaScaledBinaryKernel`（add/sub/…）迁移骨架：Op 层广播展平 + 一维 Kernel + `block_size=8192` 折叠 + fp32 中转 + `alpha==1` 折叠 + `num_kernels=min(ceil(N/block), aicore×2)` persistent 分核，对 `_combine` 不同的同族算子可直接承接，仅需重核算子特有语义（减法次序、`0*inf`、golden 分支） | P | examples/sub/sub_fwd_kernel/DESIGN.md §0–§11；examples/add/add_fwd_kernel/DESIGN.md（同族基座） | none | tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B2C（实查 20） | pattern-library/cases.md 或 elementwise.md |

### Transferable Lessons（可迁移教训）

- **同族迁移先复用再重核**：Add/Sub 同属 `AlphaScaledBinaryKernel`，迁移骨架（Op 层预广播、一维 Kernel + block_size 折叠、fp32 中转、`alpha==1` 折叠、persistent 分核、UB 预算、L0 方案）可整体承接；只需重核算子特有项——减法**次序不可交换**（`T.vsub` 的 src0 必须是被减数 `a`）、`0*inf=NaN`（禁 `alpha==0` 折叠）、golden 用 `torch.sub`。
- **带 alpha 缩放的减法要防「中间溢出」**：`fp32(alpha)*fp32(b)` 可先于结果溢出 fp32，此时两轮 fp32 链给 `±inf` 而 torch CPU golden（扩展精度中间值）给有限 `±max`——等价性机器验证必须**按域对照**（正常域 vs golden、中间溢出子域 vs 源算法），否则会误报 EQUIV_FAIL。（加法因中间溢出与结果溢出重合，不暴露该问题——不可把 Add 的验证结论外推到 Sub。）
- 迁移带 numpy 广播的二元逐元素算子到 npuir：**广播在 Op 层解析**（`broadcast_shapes` → `expand` → `flatten` 为 same-shape 1-D），kernel 只写平铺逐元素链——npuir 缺跨步 gather 向量指令，把广播索引放进 kernel 会退化为逐元素标量 gather。代价是广播操作数一次设备端物化流量。
- bf16 在所有 v-prefix 算术 API（vsub/vmul/…）的官方 dtype 表均为 ×，**任何 fp16/bf16 逐元素迁移都应走 fp32 中转**（`T.vcast(rint)` 双端），这同时对齐 torch fp32 opmath golden。
- 设计散文里避免两类 gate 误伤：① 勿连写英文目录斜杠列举（如 `docs/examples`）——会被路径存在性检查当作路径；② 勿在 §1.6.1 写出 `EQUIV_FAIL` 字面量（含「无 `EQUIV_FAIL`」这类否定），改写「全部通过 / 无未通过项」。

## Stage 3 (develop) Retrospective — first_impl (attempt 1) — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 同族 binary+broadcast 迁移（add→sub）按「机械替换 `+`→`-`、`vadd`→`vsub`、保持 `out_idx=[2]` 与 factory 契约」一次通过，无需 debug 循环——骨架复用清单（DESIGN §3.5.3 已列 add）在实现层同样成立 | `examples/sub/sub_fwd_kernel/sub_fwd_kernel.py`（对照 `examples/add/add_fwd_kernel/add_fwd_kernel.py`）；`--level all` exit 0 | 迁移 skill 可在实现期复用同族 `{op}.py` 作为直接模板（含 golden/分层套件/main），仅重核算子特有项（减法次序、`0*inf`、golden 用 `torch.sub`） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| npuir fp32 逐元素 K: 5 个显式 UB buffer（`a_ub`/`b_ub`/`a32`/`b32`/`y_ub`，均 `(8192,)`）在 npuir 编译并通过——`f32` 路径 5×32768=160KB < 192KB，未触发 UB overflow；即 add 模板的「无条件全 buffer 分配」对 fp32 直接可用，DESIGN §4.5 的 96KB 复用微优化对**正确性非必需**（仅省 UB/余量） | D | examples/sub/sub_fwd_kernel/sub_fwd_kernel.py（`sub_fwd_kernel` fp32 路径）；`examples/sub/sub_fwd_kernel/RETROSPECTIVE.md` 本节 | `python sub_fwd_kernel.py --level L0`（fp32 用例全 PASS） | tilelang 0.1.2+417dae9908ad2ea5ffb74294a7075b6cb64b4f2b dev-root + CANN 8.5.2 + Ascend910B2C（实查 20 AI Core → 40）/ torch 2.7.1+cpu / 2026-10-13 | pattern-library/constants.md 或 traps-runtime.md（UB 容量实证） |
| npuir 减法次序实证：`T.vsub(a32, b32, a32)`（src0=被减数）与 `alpha==1` 折叠路径对全 dtype 精度 = 0 或 9.537e-07（fp32 内），未出现 `b-a` 符号翻转——减法算子的 src0 语义在 npuir 实现层与文档一致 | D | examples/sub/sub_fwd_kernel/sub_fwd_kernel.py:184-190；`--level all` Summary | `python sub_fwd_kernel.py --level all`（L0+L1 ALL PASS） | tilelang 0.1.2+417dae99 + CANN 8.5.2 + Ascend910B2C / torch 2.7.1+cpu / 2026-10-13 | pattern-library/traps-runtime.md（vsub 次序） |

### Transferable Lessons（可迁移教训）

- 迁移 `AlphaScaledBinaryKernel` 同族（add/sub/…）逐元素算子：直接以已通过的 `add_fwd_kernel.py` 为模板，机械替换 combine 运算（`T.vadd`→`T.vsub`、golden `torch.add`→`torch.sub`），**保留 constant**：`@tilelang.jit(out_idx=[2], target="npuir")`、factory 签名 `(N, dtype, alpha=1, output_dtype=None, block_size=8192, num_cores=None)`、persistent 分核 `min(ceil(N/block_size), aicore×2)`、`alpha==1` 折叠、fp32 中转、分层套件/main。减法类务必保持 `T.vsub(a32, b32, a32)` 的 src0=a（非交换）。
- npuir 逐元素 kernel 的 fp32 路径可直接按「输入 buffer + 两个 fp32 中转 + 输出 buffer」全显式分配（block_size=8192 → 160KB < 192KB），无需先做 buffer 复用；复用（`a32←a_ub` 等）只影响 UB 余量，不影响能否编译/运行。
- 带广播的逐元素迁移：kernel 只写平铺 1-D 链，广播在 Op 层 `broadcast_shapes`+`expand`+`flatten` 完成；测试套件的 broadcast 用例在 Op 层构造 `expand/contiguous/view(-1)` 后调用 kernel，golden 用 torch 原生广播对拍即可（本函数 fp16/bf16/fp32 广播全 PASS）。
