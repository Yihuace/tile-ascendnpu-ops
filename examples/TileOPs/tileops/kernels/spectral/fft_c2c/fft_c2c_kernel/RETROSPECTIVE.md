# RETROSPECTIVE — fft_c2c / _fft_c2c_kernel

## Stage 1 (design) Retrospective — first_design — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 算法调研 R1/R2 需算法族文献（Stockham 索引导出）时，`webfetch` 在本环境超时不可用；本地（docs/testing/examples/algorithm-candidates）亦无 FFT 族先例，只能以模型知识 + 自建数值验证兜底 | DESIGN.md §1.6.0「调研范围说明」/§9.3；`webfetch https://en.wikipedia.org/wiki/Stockham_FFT` 超时 | `references/algorithm-research.md` §6 增补「文献不可用时以自建数值等价验证替代引用」的许可与留痕模板 | R |
| Constraint-doc | 复数 dtype（complex64/128）与 fp64 向量元算的缺失未在 `ascend-constraints.md` 显式列出，迁移前需自行 grep 源码/全 docs 判定 | DESIGN.md §0.5/§3.5.1；evidence 见 CG-2026-0019 | `references/ascend-constraints.md` §1 已知限制表增补「无复数 dtype；向量元算无 fp64（complex128 不可实现）」行 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| npuir 无复数 dtype + 向量元算无 fp64 → complex128 算子不可实现（complex64 走实/虚 fp32 分解） | D | examples/fft_c2c/_fft_c2c_kernel/DESIGN.md §0.5/§3.5.1；`.agents/evolution/capability-gaps.md` CG-2026-0019 | `python3 -c "import tilelang.language as T; print(open('docs/Tilelang.language/数学操作/T.vcos.md').read())"`（看 fp64 ×）；`src/target/codegen_npuir.cc:303-418` | tilelang 0.1.2+d822cb9b / CANN 9.0.0 / Ascend910_9362 | pattern-library/traps-runtime.md 或 ascend-constraints.md |
| FFT 迁移首选 Stockham autosort（无 bit-reversal）：NPU 缺跨步 gather（TRAP-C10），DIT+bit-reversal 装载不可直接移植 | P | DESIGN.md §1.6.0 R3/R4；数值验证 `verify_equiv.py`（n=2..64 vs numpy/torch.fft） | `python3 examples/fft_c2c/_fft_c2c_kernel/verify_equiv.py` | tilelang 0.1.2+d822cb9b / Ascend910_9362 | pattern-library/cases.md（谱类算子无先例，可作首例索引） |
| Ascend 上 Karatsuba 复数乘为负收益（vadd/vmul 同吞吐，op 数 8>6）——降 op 数优化须先按 op 数而非 mul 数判定 | D | DESIGN.md §1.6.1 #6；`CONST-vector-launch-overhead`（op 数计时） | `python3 examples/fft_c2c/_fft_c2c_kernel/verify_equiv.py`（算法等价性）+ op 计数 | tilelang 0.1.2+d822cb9b / Ascend910_9362 | pattern-library/constants.md 或 elementwise.md |

### Transferable Lessons（可迁移教训）

- **迁移含非线性/置换数据搬移的算子（FFT/gather/scatter 类）先查 NPU 是否缺该原语**：跨步 gather 向量指令缺失（TRAP-C10-strided-gather），DIT FFT 的 bit-reversal 装载、任意 permute 均不可直接移植；改用「自排序/无重排」算法变体（如 Stockham）可整段绕开。
- **算子 dtype 能力先核实再设计**：本项目无复数 dtype（`T.Tensor` 复数 codegen 直接 LOG(FATAL)），向量元算无 fp64（T.vadd/T.vcos/… 的 fp64 列均 ×）。复数算子必须实/虚分解为 fp16/fp32 双张量；fp64 分量（complex128）无可行路径。
- **无本地先例的算法族**：Stockham 等索引格式在本仓无文档，**不要凭记忆写索引**（本次模型回忆的伪码经 n=4 手算即证伪）——用最小数值脚本（numpy 对照）验证索引后再写入 DESIGN。
- **降 op 数 ≠ 降 mul 数**：Ascend Vector 上 vadd/vmul 同为向量指令、吞吐相当，Karatsuba（少 mul 多 add）是负收益；公式优化按**总 op 数**判定（结合 CONST-vector-launch-overhead 的 op 数计时口径）。
- **不可对齐时的分核**：逻辑核数（batch 行数，2 的幂）与物理核数（40 = 2³·5）数学上无法整数倍对齐时，按标准「无法对齐兜底」取 `grid=B` 并显式记录残留不均（1.6 波次、临界路径 +60%），不必强上核内串行 persistent。

## Stage 2 (design-review) Retrospective — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Mechanical-check | `design_calc_check.py` 本算子的 ub_budget / l0c_budget / core_split **三项全 skip**（"no per-level buffer tables parsed"；"insufficient parsed dims {N:4096}"）——对「纯 Vector + ping-pong 多缓冲 + LUT 常量表」的表单形态解析失效，机械门禁退化为人工复核 | 本次实跑 JSON：`ub_budget/l0c_budget/core_split: skip`，仅 `r3_metrics: pass` | `.agents/tools/design_calc_check.py` 增补：① 解析含多个 `(N,)` 形一维 UB 表的总和；② 纯 Vector 算子的 `core_split` 支持从 §5.5 文本提取 logical/physical 核数（或标注"纯 Vector 豁免"） | R |
| Verification-gap | 本算子 UB 结论依赖 `CONST-capacity-910B2C` 的 ×1.10–1.12 膨胀系数，而该条明示"9362 未重验"；检视侧无法在本环境实测该常数，只能标注"未重验假设" | `pattern-library/constants.md` L27「9362 适用性注记」；DESIGN §4.5 | design-review 维度 2/8 增补条款：当容量结论仅由"SOC 未重验"常数支撑时，要求在 REVIEW 中显式记录"常数待重验 + 备选方案存在性"作为通过前提 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 向量元算 dtype 矩阵在 docs 间不一致：`T.vadd`/`T.vmul` 表**无 fp64 列**，`T.vcos`/`T.vsin`/`T.deinterleave` 表有 fp64 列且标 ×——能力缺口证据引用须区分「列不存在」与「列存在但 ×」 | D | 亲读 `docs/Tilelang.language/数学操作/T.vadd.md` §2.2.1（12 列无 fp64）、`T.vmul.md` §2.2.1（同）、`T.vcos.md` §2.2.1（13 列含 fp64=×）、`T.deinterleave.md` §2.2.1（fp64=×）；对照 CG-2026-0019 evidence 行 | `grep -n "fp64" docs/Tilelang.language/数学操作/T.{vadd,vmul,vcos}.md docs/Tilelang.language/shape操作/T.deinterleave.md` | tilelang 0.1.2 / Ascend910_9362 | docs/Tilelang.language/数学操作/*（补齐 fp64 列）+ tilelang-design-review 维度 8 dtype 取值核对细则 |
| 维度 8「复杂度算术误差」的分级缺明确规则：设计 R3 表 FLOPs 系数偏大 2× 且与自身 roofline/manifest 自相矛盾，但该列不参与算法选择（所有候选同比例、决策由访存主导）——现行 skill 将该类列入"fail 触发"，与"阻塞=致 Stage 3 失败"核心原则冲突 | P | DESIGN.md §1.6.0 R3（`~10·N·log2N` vs roofline `5·N·log2N`；R3 注"脊点≈4" vs R4"≈8"）；本次 REVIEW 维度 8 判 warn | 人工比对 `spectral.yaml` L47 `flops=batch*5*n*log2(n)` 与 R3 表 | — | `.agents/skills/tilelang-design-review/SKILL.md` 维度 8「复杂度量级误差」分级条款：不改变候选排序/结论的显示值误差 = 建议级（附复算证据），推翻结论的误差 = 阻塞级 |
| 迁移维度 0 对"§0.5/§0.4 列了源码不存在的优化项"（本算子 swizzle）无明确处置口径 | P | DESIGN.md §0.5 末行"swizzle 防 bank conflict 舍弃"；源 `fft.py` 无任何 swizzle/padding 语句（通读 576 行） | 对照源算子 grep `swizzle\|padding\|bank` 无命中 | — | `.agents/skills/tilelang-design-review/SKILL.md` 维度 0 表：混淆条目处置——防御性多列未影响 §0.3 步骤完整性时判 pass+建议，不判 fail |

### Transferable Lessons（可迁移教训）

- **机械核对 ≠ 语义核对，skip 必须转人工且要点名**：`design_calc_check` 对新型表单（一维 Vector 多缓冲）大面积 skip 时，REVIEW 的「机械复核」段必须逐 skip 项给出人工复算过程与数值，不得只贴 JSON（本次 ub_budget 手算 98296B、core_split 手算 logical=B/physical=40）。
- **负向断言核对要落到"被引对象的确切字段值"**：CG-2026-0019 的证据链声称多个 API "fp64 ×"，实读发现 `T.vadd/T.vmul` 表**根本没有 fp64 列**（"不存在"≠"存在但×"）——结论仍成立（无 fp64 支持），但引用措辞需精确；检视侧对"dtype 矩阵逐格取值"仍应亲自打开表格，不能以"设计已核对"采信。
- **等价性机器验证要验"脚本真的在算设计的公式"**：`verify_equiv.py` 的 full-FFT 行是否复现 §3.1 的 Stockham 迭代式（而非另写一套）——本次逐行读脚本确认其 `lo/hi/B[...]` 索引与 §0.6 R3/§3.1 逐字一致，该 `EQUIV_PASS` 才对设计公式有效。
- **迁移检视须交叉核对 host 侧引用的行号**：设计引用的 op 层 `_build_lut`/`forward` 行号与脚手架副本路径逐条打开核对（`fft_c2c.py` L81/L96/L100/L107–108；`_fft_c2c_kernels.py` 存在），host 侧引用错误会误导 Stage 3 wrapper 实现。
- **性能估算表述瑕疵与分核三要素要分开判**：分核三要素（逻辑/物理核数实查/规模方案）齐全即可判 pass；"1.6 波次 vs 先跑40再跑24=2 波次""临界路径 +60%"属估算措辞瑕疵，记建议不阻塞。

## Stage 3 (develop) Retrospective — first_impl — 2026-10-13

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | DESIGN §3.3/§6 明确写「Python `for s in range(t)` → 编译期展开（静态边界）」，但 TileLang npuir parser 把 `range` 下推为**运行时 serial 循环**（迭代变量是 TIR Var），设计所依赖的「逐级 Python 常量」机制在实现层不成立——Stage 3 必须自行发明递归 macro 链绕法；design skill 的「循环结构」判定缺少对「parser 是否真展开 range」的落地核验 | DESIGN.md §3.3/§6.1/§6.2（"Python range → 编译期展开"）；本次实测 `for s in range(3): ...coeffs[s]` → `list indices must be integers or slices, not Var` | `tilelang-op-design` 的 §6 循环结构指引增补：「声称编译期展开的 Python 循环，须在 Stage 3 前用最小探针核实 parser 行为（`range` 是否产出 TIR Var）；NPU 需要 trace-期展开时改用递归 `@T.macro` 链」 | R |
| Constraint-doc | `T.Parallel` 的 extent 若含**符号表达式**（如由运行时级循环变量导出的段长 `h`/组数 `ng`），npuir 编译器在 IR 构建期**段错误**（无诊断输出，进程 139）——是比「报错」更难定位的失败形态，`ascend-constraints.md` 未登记 | 本次探针：`T.Parallel(ng, h)`（h/ng 符号）编译期 segfault；单级常量 extent 全过 | `references/ascend-constraints.md` §1 增补「T.Parallel 要求编译期常量 extent；符号 extent 触发编译器段错误（非诊断）」+ 建议在 parser 侧加友好诊断 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| TileLang npuir 无 Python 级编译期循环展开：`for s in range()` 是运行时 serial 循环（Var），`T.Parallel` 符号 extent 段错误；绕法 = 递归 `@T.macro` 链实现 trace-期全展开 | D | `.agents/evolution/capability-gaps.md` CG-2026-0020；`tilelang/language/parser/parser.py` L174–197/L478–510；本次实测（range→Var；`T.Parallel(ng,h)` 符号 extent→segfault） | `python examples/fft_c2c/_fft_c2c_kernel/repro/TRAP-parser-no-python-unroll.py`（断言 range-form 抛错 + macro 链数值正确，exit 0） | tilelang 0.1.2 dev root build / Ascend910_9362 / CANN 9.0.0 / 2026-10-13 | pattern-library/traps-compiler.md + capability-gaps.md CG-2026-0020 |
| 逐元素 `T.Parallel(ng,h)` 内仿射索引表达式（读 `S[g*2h+i]`/写 `D[g*h+i]`/`D[g*h+i+N/2]`，段内连续）在 npuir 上**直接向量化且正确**，无需 v-prefix 全缓冲算子的分段拷贝 | D | DESIGN §3.2 列 T.vmul/vsub/vadd，§3.3 用元素级表达式（REVIEW 维度 1 已标分歧）；本次实现取元素级表达式：n=4096 单行 max_abs=4.3e-5；单级/多级/末级 h=1 全部 compile+数值正确 | `python examples/fft_c2c/_fft_c2c_kernel/_fft_c2c_kernel.py --level all` | tilelang 0.1.2 dev root build / Ascend910_9362 / CANN 9.0.0 / 2026-10-13 | pattern-library/elementwise.md（分段蝶形/仿射索引向量化）+ layout.md |
| `out_idx` 单输出返回**张量本身**、多输出（`out_idx=[i,j]`）返回**元组**——无统一约定，调用点须按输出个数分别取用（本次先按元组写单输出探针得到标量，误判"kernel 算错"，耗一轮定位） | D | `_fft_c2c_kernel.py` `out_idx=[4,5]` 取 `out[0]/out[1]`（正确）；repro 中 `out_idx=[1]` 取 `[-1][0]` 得标量 | `python examples/fft_c2c/_fft_c2c_kernel/repro/TRAP-parser-no-python-unroll.py` | tilelang 0.1.2 dev root build / Ascend910_9362 / CANN 9.0.0 / 2026-10-13 | pattern-library/traps-runtime.md TRAP-outidx-call-convention（补一条单/多输出返回值类型差异） |

### Transferable Lessons（可迁移教训）

- **`for s in range(<python常量>)` 在 npuir prim_func 里不是编译期展开**：迭代变量是 TIR `Var`，用 Python list/dict 按 `s` 取值或让 `T.Parallel` 的 extent 依赖 `s` 都会失败（前者报 "not Var"，后者编译器段错误）。需要「逐级不同编译期常量」的展开式算法（FFT 蝶形各级段长、多级 stencil）用**递归 `@T.macro` 链**：在 factory 的 Python 体里递归构造每级宏（Python 递归发生在 trace 前，不受 parser 限制），每级宏内联该级常量并调用下一级宏，prim_func 只调链头一次。
- **`T.Parallel` 的 extent 必须是编译期常量**：符号 extent（含运行时循环变量派生的量）会导致 npuir 编译器**段错误**而非报错——遇到无诊断的 139 先怀疑 parallel 循环的 extent/形状是否符号化（对照最小探针逐级二分）。
- **分段读写（非等形 buffer）的蝶形/仿射算法，用 `T.Parallel` 内的元素级索引表达式即可向量化**：读 `S[g*2h+i]`（段内 stride-1）+ 写 `D[g*h+i]` / `D[g*h+i+N/2]`（两段 stride-1）均由编译器生成向量访问；不要为了套 v-prefix 全缓冲算子（T.vadd 等，要求等形）而先做分段拷贝（额外 UB 流量）。仅末级 `h=1`（内层 extent 1）效率低但正确，perf refinement 留 Stage 4。
- **核对 `out_idx` 返回值形态**：单输出 `out_idx=[i]` 直接返回张量，多输出 `out_idx=[i,j]` 返回元组——wrapper 调用点按输出个数取用；单输出误按元组索引会拿到某一行/标量，症状酷似算子算错。
- **源 GPU 的「编译期展开」注释不可照信**：`for s in range(...)` 在 GPU 后端可用（h 符号仅进地址算术）≠ NPU 可用（h 进 T.Parallel extent 即崩）；迁移时把「循环是否真展开」当作独立于算法的实现层验证项先做最小探针。
