# _conv1d_direct_kernel — Stage 1 Retrospective

## Stage 1 (design) Retrospective — first_design — 2026-10-14

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 迁移脚手架对 depthwise/direct builder 一律 pin `ascend_mode="Expert"`（因 `DepthwiseConv1dKernel` 继承 `GroupConv1dKernel`），未区分「有通道收缩的 GEMM 类」与「无收缩的逐元素+小规约类」——设计期须自行判定计算单元（Vector vs Cube），不能照抄脚手架模式结论 | `examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d.py` L351（`ascend_mode="Expert"`）；本 DESIGN.md §2.2 判定「计算单元 = Vector，不落 Cube」 | 在 skill `migration-analysis.md` §5.3 或 decision-tree 增一行：depthwise/single-channel-per-group 的 direct conv = 逐元素+小规约 → Vector（不落 Cube），脚手架模式 pin 不可作为计算单元依据 | P |
| Info-source | algorithm-candidates 的 `ALG-conv` 命中行以 implicit GEMM / im2col / Winograd 为主，未显式覆盖「无输入通道收缩的 depthwise/direct」这一分支（设计期需自行补出 block-diagonal dense GEMM 淘汰 + 逐行加权归约选定） | 本 DESIGN.md §1.6.0 R1 候选 0–7（含候选 3 block-diagonal dense GEMM 的 groups 倍 FLOPs 淘汰论证） | 在 `references/algorithm-candidates.md` 的 `ALG-conv` 增加 depthwise/direct 子分支（逐行加权归约 + 多抽头标量乘加；im2col-GEMM 不适用） | C |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 纯 Vector kernel 若需「kernel 内构造零填充 workspace（GM 偏移目的地）」，必须用 `ascend_mode="Expert"` + 单一 `T.Scope("Vector")`；同结构在 Developer 下编译失败（`multiple entry functions detected (previous: 'main_mix_aiv', this: 'main_mix_aic')`，因 Developer 生成 AIV+AIC 双 entry 而 bshengir 只允许一个 `hivm.entry`） | D | `examples/conv1d/_conv1d_direct_kernel/history_version/design_probe_direct_vector.py`（Developer 编译失败，Expert PASS）；本 DESIGN.md §2.2 / §9.1 | `cd examples/conv1d/_conv1d_direct_kernel/history_version && python3 design_probe_direct_vector.py Developer`（失败）/ `... Expert`（PASS，max_abs_err≈7.8e-4） | tilelang dev-root build (/home/developer/tilelang-mlir-ascend/build) + CANN 8.5.2 + Ascend 910B3（get_aicore_num=20） | pattern-library/traps-compiler.md 或 traps-runtime.md |
| 无输入通道收缩的 depthwise/direct conv 是**逐元素 + 小规约（K_W）**，正确计算单元是 **Vector 而非 Cube**：把 depthwise 视作 block-diagonal dense GEMM 会使 FLOPs 与权重物化均 `groups` 倍劣、且 K 轴仅 `K_W` 致 Cube 利用率极低——设计期须据此拒绝 Cube 路径 | P | 本 DESIGN.md §0.5 / §1.6.0 R1 候选 3 / §1.6.0 调研结论 / §2.2；源 `src/tileops/kernels/convolution/conv1d.py` L1074-1080 注释「there is no contraction to tile」 | 设计期即可判定（无需运行） | tilelang dev-root build + CANN 8.5.2 + Ascend 910B3 | pattern-library / 待定 |
| depthwise 核窗加权归约的 Vector 落地形态（实测可用）：`[K_W,BN]` 移位窗 tile（per-kw 连续 `T.copy`，行=tap）× 权重 `[K_W,1]` **行广播** `vmul` → 一次 `T.reduce(dims=[0],sum)` → `[1,BN]` fp32 → bias `[1,1]` **列广播** `vadd` → `vcast` → store；相对「per-kw FMA 累加」把向量 op 数从 O(K_W) 降到 O(1)（本算子 emission-bound，op 数是主导项） | P | `examples/conv1d/_conv1d_direct_kernel/history_version/design_probe_direct_vector.py`（Expert PASS：`vmul [K,BN]·[K,1]` 行广播 + `reduce dims=[0]` + `vadd [1,N]+[1,1]` 列广播）；本 DESIGN.md §1.6.0 R1 候选 1 / §3.2 / §3.3 | `python3 design_probe_direct_vector.py Expert` | tilelang dev-root build + CANN 8.5.2 + Ascend 910B3 | pattern-library/elementwise.md（窗口/规约类） |

### Transferable Lessons（可迁移教训）

- **计算单元判定先于 API 选型**：迁移一个「看似卷积」的算子前，先判断其是否存在 tensor contraction；无输入通道收缩的 depthwise/逐通道相关是「逐元素 + 小规约」，应落 Vector，**不要**因为脚手架/同族函数是 Cube 就照抄 `T.gemm`+L1/L0C。把逐通道相关硬写成 dense GEMM 会让 FLOPs 与权重物化都 `groups` 倍劣。
- **Expert 下的单 `T.Scope("Vector")` 是纯 Vector kernel 的可用形态**（无 Cube、无跨引擎握手），但**同一 kernel 在 Developer 下可能编译失败**（AIV+AIC 双 entry 冲突）——凡 kernel 内需构造带偏移的零填充 workspace（GM 偏移目的地）或需要显式 UB 控制时，直接选 Expert + `T.Scope("Vector")`，并用一个最小探针先确认编译通过，再写完整设计。
- **GM 偏移目的地 vs UB 偏移目的地**：kernel 内构造 padding workspace 时，`T.copy` 的**UB 列偏移目的地会静默错值**（`TRAP-copy-ub-offset-stride`），必须走「同形 UB 中转 + GM 偏移目的地」两步；host 侧禁止 padding（`ascend-constraints.md` §4），padding 只能在 kernel 内完成。
- **平移窗装载（conv/pool 类）的向量化前提是「最内维连续」**：stride=1 时 per-tap 移位段是连续段，可直接 `T.copy`；`dilation` 只平移起点不破坏连续；**stride>1 的列采样是跨步读**（无跨步 gather 向量指令），须先 polyphase 抽取物化（`T.gather` + host 常量索引）再连续装载。
- **bf16 无原生 v-prefix 逐元素算术**（`vmul`/`vadd` 的 bf16 列不支持）：含 bf16 的 Vector 算子必须全链 fp32 中转（`vcast` 升/降），把 `vmul`/`vadd`/`reduce` 放 fp32 域。
- **小 shape 的 Vector 算子是「发射/延迟受限」而非「带宽受限」**：roofline 估算里 `CONST-vector-launch-overhead`（~0.5µs/op）项常主导；降低向量 op 数（如把 per-kw FMA 折叠成一次归约）与增大每任务计算量（行/块合并）是首要调优方向，不要只看 GM 流量。
- 复用同族已通过验证的设计时，**上游 Stage 3 实测修正必须逐条继承**（本设计承接同族函数 2/4 的 `TRAP-copy-ub-offset-stride`：偏移放 GM 侧；`TRAP-C10-strided-gather`：stride>1 走 polyphase）。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-14

最终产物：`examples/conv1d/_conv1d_direct_kernel/_conv1d_direct_kernel.py`（Expert / 单 `T.Scope("Vector")`，无 Cube）。测试
`python _conv1d_direct_kernel.py --level all`：L0 10/10、L1 24/24、L2 16/16、Boundary 10/10 全 PASS（真机 NPU）；
全量 max_abs = 7.81e-03（fp16/bias，L2-kW31），在 `1e-2` 容差内。verdict `[PRECISION_PASS]`。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Syntax/parser | `@T.macro` 体内**不能**使用 python `for ... in range(...)`（TVMScript 宏解析器不解析 builtin `range`），且报错信息**自相矛盾且有误导**——诊断文本把 `range` 列为合法循环，实际位置指向报错函数之后的行；`range` 直接写在 `@T.prim_func` 体内或宏内用 `T.serial` 均正常 | 本任务首次编译报 `error: Expect the for loop to be one of the following: range, T.serial, ...`（location 指向 `_run_case`/下一 prim_func 行）；最小复现 `repro/TRAP-macro-range-loop.py`（宏+range 子进程失败、宏+T.serial 与 prim_func+range 成功） | 在 `references/pattern-library/traps-compiler.md` 增条目：`@T.macro` 内循环一律用 `T.serial`/`T.unroll`，禁 python `range`；若需 python-int 循环体索引 python list，把循环内联进 `@T.prim_func` 或把索引改写成对循环变量的 TIR 运算 | D |
| Design API mapping | DESIGN §3.3 伪代码用 `r_kw[kw]`/`q_kw[kw]`（python list）配 `T.serial(kernel_l)` 循环——`T.serial` 的循环变量是 TIR var，**不能**索引 python list；伪代码形态不可直接编译。正确落地：把 `k_w*d_W` 的 polyphase 分解改写成对 TIR 循环变量的表达式 `(cpos % s_W)*Mdec + ol0 + (cpos // s_W)`（`cpos = kw*dilation`） | 本产物 `_conv1d_direct_kernel.py` 主循环；DESIGN.md §3.3 第 456-462 行 | design skill 的 §3.3 伪代码校验：禁止「`T.serial` 循环变量 + python list 索引」混用；polyphase `(q_kw, r_kw)` 表在 kernel 内应写成 TIR 算术（host 常量仅供 `poly_idx` 预建） | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `@T.macro` 体内 python `for i in range(n)` 无法编译（parser 报 `Expect the for loop to be one of the following: range, ...`，但 `range` 其实不被宏解析器支持、报错行号错位）；改用 `T.serial(n)` 可编译。同一 `range` 直接写在 `@T.prim_func` 体内则正常（T.Kernel 是否在宏内无关） | D | 本任务首次编译失败 → 重构为 `T.serial`；`repro/TRAP-macro-range-loop.py`（宏+range 在子进程失败、宏+T.serial 与 prim_func+range 成功，三条断言实测成立） | `cd examples/conv1d/_conv1d_direct_kernel && python repro/TRAP-macro-range-loop.py` | tilelang 0.1.2+417dae99 + CANN 8.5.x + Ascend 910B3（get_aicore_num=20），Expert | pattern-library/traps-compiler.md |

### Transferable Lessons（可迁移教训）

- **`@T.macro` 体内不要写 python `range` 循环**：TVMScript 宏解析器不解析 builtin `range`（尽管报错文本声称支持），一律用 `T.serial`/`T.unroll`；若循环体需要 python-int 下标去索引 python 常量表，把该循环**内联进 `@T.prim_func`**，或把下标改写成 TIR 表达式（对循环变量做算术），不要在宏里混用 `T.serial` 变量 + python list。
- **迁移 DESIGN 伪代码里的「TIR 循环变量 + python list 索引」是不可编译形态**：凡设计给出 `for kw in T.serial(K): ... lst[kw] ...` 之类的写法，落地时务必改写（本函数把 `r_kw[kw]/q_kw[kw]` 改成 `(kw*d_W % s_W)*Mdec + ol0 + (kw*d_W // s_W)`）；host 侧只预建与循环变量无关的常量表（如 `poly_idx`）。
- **本 Vector 形态（单 `T.Scope("Vector")` + GM 偏移目的地构造 `xpad` + per-kw 连续 `T.copy` + `vmul` 行广播 + `reduce(dims=[0])` + `vadd` 列广播）真机可跑、精度达标**（L0–Boundary 全 PASS，max_abs 7.81e-3 / fp16 ≤ 1e-2）；后续迁移同族 depthwise/direct 函数可直接复用该骨架。
- **polyphase `xdec` 行宽取法**：令 `Mdec = ceil16(s_W * ceil(l_pad/s_W)) // s_W`、`l_pad2 = s_W * Mdec`（16 对齐且为 stride 倍数），则 `T.gather` 索引与 tap 读索引恒在 `xpad`/`xdec` 界内（`out_l-1+max(q_kw) ≤ Mdec-1`），stride=3/4 等非 16 整除 stride 亦正确——避免 REVIEW 维度 5 记录的 `ubd[c_in,l_pad]` 与 `xdec[...,s_W*Mdec]` 形状不一致问题（本实现令二者同宽 `l_pad2`）。
