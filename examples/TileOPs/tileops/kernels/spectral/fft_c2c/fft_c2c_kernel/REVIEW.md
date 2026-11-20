# _fft_c2c_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/fft_c2c/_fft_c2c_kernel/DESIGN.md
- 任务类型: 迁移（migration，子模式 migration-harness）
- 源算子: /home/developer/TileOPs/tileops/kernels/fft.py（工厂 `_fft_c2c_kernel` L16；内嵌 `_fft_lut_func` L75 → `@T.prim_func _fft_lut_main` L404；辅助 macro L91/L186/L240/L304；类 `FFTC2CKernel` L476）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 8

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `python3 examples/fft_c2c/_fft_c2c_kernel/verify_equiv.py` → 退出码 0，`总采纳项: 6  失败: 0`，全部 `EQUIV_PASS`。逐项与 DESIGN.md §1.6.1 内嵌结果表**一致**（D-1 通过）：
  - `w=1  (skip mul)` max_ulp 0.000 / viol 0.00000000 ✓
  - `w=-1 (negate)` max_ulp 0.000 / viol 0.00000000 ✓
  - `w=-i (swap+negate)` max_ulp 0.000 / viol 0.00000000 ✓
  - `w=+i (swap+negate)` max_ulp 0.000 / viol 0.00000000 ✓
  - `LUT(fp32) vs in-kernel fp64 trig (all stages, n in {64,4096})` max_ulp 0.475 / viol 0.00000000 ✓
  - `full radix-2 Stockham vs torch.fft` rel_abs_err 6.830e-06 ✓（脚本输出与 §1.6.1 表列 6.83e-6 一致）
  - 独立核对脚本实现：`check_full_fft` 逐字复现 §3.1 / §0.6 R3 的 Stockham DIF 迭代式（`lo=A[g·2h+i]`、`hi=A[g·2h+h+i]`、`B[g·h+i]=lo+hi`、`B[g·h+i+N/2]=(lo-hi)·w`，`w=exp(-2πi·i/(2h))`），故该行确为「设计公式」的整链数值验证，而非另写一套等价实现；`check_special`/`check_lut_vs_trig` 逐项验证 §1.6.1 #1–#4 采纳项，验证对象成立。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design examples/fft_c2c/_fft_c2c_kernel/DESIGN.md`）JSON checks 摘要：
  - `ub_budget`: **skip**（"no per-level buffer tables parsed"）→ 人工复核（D-5）：§4.5 表 `4×4096×4B=65536B` + `2×4095×4B=32760B` = **98296B ≈ 96KB ≤ 192KB**（`CONST-capacity-910B2C`，9362 容量一致）✓，含 1.10–1.12 膨胀 ≈108–110KB、含 1.7 悲观口径 ≈167KB 均 < 192KB ✓。
  - `l0c_budget`: **skip**（无 block_M×block_N）→ 人工复核：本算子无 Cube/L0C，"不适用"成立 ✓。
  - `core_split`: **skip**（"insufficient parsed dims"）→ 人工复核：逻辑核数 = `ceil(B/1)=B`，纯 Vector 物理核数 = `get_aicore_num()*2` = 20×2 = **40**（§5.5 记录查询代码与返回值，与 `CONST-aicore-910_9362` 一致），B=64 → 1.6 波次四态方案齐全 ✓。
  - `r3_metrics`: **pass**（"all four R3 metrics present"）✓；数值自洽性由维度 8 独立复算核对（见下）。
  - 反查（维度 8/3 观察，非阻塞）：`ub_budget`/`core_split` 的 skip 与联检工具的解析局限（表头形态）一致，非设计缺陷。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**核对方式**：亲自 Read 源算子 `/home/developer/TileOPs/tileops/kernels/fft.py`（576 行）与 op 层 `examples/TileOPs/tileops/ops/spectral/fft_c2c.py`（`_build_lut` L81/L96/L100/L107–108；`forward` L123 起），逐项对照 DESIGN.md §0.1–§0.7。

- **语义理解正确性 — pass**：§0.1 的 dtype 拆分（complex64→float32 / complex128→float64，源码 L53–L58）、`accum_dtype = real_dtype`（L88）、蝶形内 cast 到 accum 再 cast 回（L158–L169）、`N` 为 2 的幂强制校验（L60–L61）、边界（`n&(n-1)!=0 or n==0 → ValueError`）均与源码一致；输出 `same_as_input`、前导维展平再 reshape（op `forward`）核对一致。**未发现语义偏差**。
- **算法解读完整性 — pass**：§0.3 步骤表 P1–P7 覆盖源码全部计算语句且行号可核：P1 位反转装载（L107–L134，`rev_idx=(rev_idx<<1)|(temp&1)`）、P2 SMEM 蝶形（L139–L171）、P3 写回（L174–L183）、P4/L5/L6 LUT radix-2/4/8（L186–L231 / L240–L301 / L304–L402）、P7 stage 调度（L404–L437，`r8_count=(lut_stage_count//3)*3`、余 2 走 radix-4、余 1 走 radix-2）。数据流图覆盖 `x_real/x_imag`、`smem_r/i`、`y_real/imag`、`lut_real/imag` 全部 buffer；host 侧 6 步逻辑（拆复/展平/建 LUT/派发/合并）齐全。**无遗漏步骤、无臆造计算步骤**。
- **优化手段覆盖完整性 — pass**：§0.4 O1–O8 全部可在源码定位：O1 LUT（op `_build_lut`）、O2 SMEM stage 融合（docstring L24–L31/L483–L487）、O3 bit-reversal 融合装载（L107–L134）、O4 radix-8/4 融合（L240/L304）、O5 per-stage kernel（L207/L264/L330）、O6 即时三角（L150–L156）、O7 二维 grid（L103）、O8 threads/block_size（L103/L208、默认 config L526–L528）。硬件依赖逐项标注，未把实现层优化误判为算法层。（非阻塞观察：§0.5 另列了源码中并不存在的「swizzle 防 bank conflict」条目并判"舍弃/不适用"——属防御性登记而非对源码的臆造，不影响 §0.3 步骤完整性口径。）
- **耦合性判定合理性 — pass**：§0.5 每条目有四态处置与依据：复杂 dtype 等价替换（codegen 无 complex 分支）、O2→UB 等价替换、O3 重新设计（TRAP-C10 跨步 gather 缺失，`pattern-library/traps-compiler.md` 实存）、O6 重新设计（T.vcos 角度域受限）、O7 一维化（ascend-constraints §1）、O8 舍弃（TRAP-threads-kwarg-noop，`src/ir.cc` NPU 分支仅消费 grid_size——已在 `pattern-library/traps-compiler.md` 核对）。与 GPU→NPU 能力映射及 ascend-constraints 不矛盾；舍弃项（O5/O8/swizzle）均写明理由。
- **NPU 重设计可行性 — pass**：§0.6 R1–R5 每项含「源方案 → NPU 新算法 → 语义保持论证」三要素。R3 的 Stockham autosort DIF 迭代式满足 NPU 硬件约束（每级读/写均为长度 h 的连续段，无 gather/scatter；`N` 为 2 的幂使 `n_groups`/`h` 整除、无尾块）；语义保持经数值验证（`verify_equiv.py` full-FFT 行 n=4096，err 6.83e-6 < 1e-3）。R2 一维 grid、R4 整行 UB 单趟、R5 全 LUT 的重设计论证成立、累加顺序变化对精度的影响已论证（容差 1e-4 内）。
- **设计与决策一致性 — pass**：§1.4（Stockham DIF + 整行 UB + 全 LUT + 实/虚 fp32）、§1.5 数据流、§3.1/§3.3（Stockham 迭代式，段索引 `g·2h+i` 等）、§5.2（1 行/block、grid=B）、§6（编译期展开 t 级 + `T.Parallel(ng,h)`）、§7（自动同步）均描述的是 §0.5/§0.6 决策后的算法，**无"§0 判重设计、§3 照抄源方案"的脱节**（§3.5.2 GPU→NPU 差异表逐项对应 §0.6）。
- **golden 独立性 — pass**：§8.1 golden = `torch.fft.fft(x, dim=-1)`，以 §0.1 语义为唯一依据（移植源仓 `tests/ops/test_fft.py` 基准），明确"不复刻 §0.6 的 NPU 算法"，验证独立性成立。

### 1. API 可行性 — pass

- **API 存在性 — pass**：§3.2/§3.4 所列 API 全部确认。`tilelang/language/__init__.py` 实查：`npuir_add as vadd`(L73)、`npuir_sub as vsub`(L75)、`npuir_mul as vmul`(L81)、`npuir_deinterleave as deinterleave`(L163–164) 均导出；`T.Kernel/T.copy/T.Parallel/T.alloc_shared/T.cast` 为全部 NPU 示例通用。docs 佐证存在：`docs/Tilelang.language/数学操作/{T.vadd,T.vsub,T.vmul,T.vcos,T.vsin}.md`、`shape操作/T.deinterleave.md`、`创建操作/T.transpose.md`、`内存操作/{T.copy,T.alloc_shared}.md`。examples 佐证 `examples/flash_attention/flash_attn_npuir_dev.py`、`examples/norm/layer_norm.py`、`testing/npuir/shape_manipulation_ops/test_slice_deinterleave_dev.py`、`testing/npuir/parallel_ops/` 均真实存在。
- **API 与模式匹配 — pass**：Developer 模式 + 自动同步；无手动 `set_flag/wait_flag`。
- **v-prefix 优先 — pass**：§1.6.2/§3.2 采用 `T.vadd/vsub/vmul/deinterleave`；`T.transpose` 仅列作弃选候选，符合 AGENTS.md「优先 v-prefix」。
- （非阻塞观察）§3.2 声称蝶形走 `T.vmul/vsub/vadd`，而 §3.3 伪代码以 `T.Parallel(ng,h)` 内元素级表达式书写——两条路径在 npuir 中是否落地为同一向量指令链需 Stage 3 核对（见维度 8 建议）；不影响 API 存在性判定。

### 2. 内存层级规划 — pass

- **搬运路径完整 — pass**：§4.4 仅 GM→UB 两级（`T.copy` 整行），无 GM→L0 越级；无 L1/L0（无 Cube）。
- **UB 预算 — pass**：§4.5 手工复算 98296B ≈ 96KB ≤ 192KB（见机械复核 D-5），含膨胀余量与保守备选（LUT per-stage 载入）；`design_calc_check.ub_budget` skip 已人工核。Buffer 层级标注齐全（§4.3：A_r/A_i/Bc_r/Bc_i 均 UB/alloc_shared；LUT GM→UB）。
- **L0C 容量 — n/a**（无 Cube）。
- （非阻塞观察）§4.3/§4.5 未单列末级 §1.6.3 主选方案里 `T.deinterleave` 的偶/奇临时缓冲（各 N/2×2 实虚 ≈32KB）；即便计入总峰值 ≈128KB，×1.10–1.12 后 ≈141KB 仍 < 192KB，故不阻塞，但建议补计入以保证余量口径完整。

### 3. Tiling 策略 — pass

- **Block 划分与理由 — pass**：§5.2 block=1 行（整行常驻 UB）、grid=B；理由（DIF 首级 stride=N/2 跨整行、分块引入跨 block 依赖）成立。
- **对齐/非整除 — pass**：`N` 为 2 的幂，段长 `h=2^k` 编译期整数、`h≥4` 时 fp32×4 满宽；无尾块；`B` 由 grid 承接。
- **分核策略三要素 — pass**（核对标准 `.agents/skills/_shared/standards/core-split-strategy.md` §1）：
  - ① 逻辑核数：`num_logical_kernels = ceil(B/1) = B`（§5.5）✓
  - ② 物理核数：纯 Vector → `get_aicore_num()*2`；记录查询代码 `NPUUtils.get().get_aicore_num()` 与返回值 **20** → **40**，与 `CONST-aicore-910_9362`（20 AIC/40 AIV）一致 ✓
  - ③ 规模判定：四态齐全——B≤40 无需适配（附 1/40 占用风险）、B=64「无法对齐兜底」（64=2⁶ 与 40=2³·5 不可整数对齐，取 grid=B、记录残差）、极大规模核内 `T.serial` 方案、B=1 提升备选（四步分解）✓
- （非阻塞观察）§5.5「运行时调度为 1.6 波次（先跑 40、再跑 24），相对理想 1 波次临界路径 +60%」措辞不自洽：先跑 40 再跑 24 实为 **2 波次**，相对 1.6 理论波次应为 +25%（利用率 64/80=80% 正确）。属性能估算表述瑕疵，不影响分核方案选择（grid=B）。

### 4. 技术约束检测 — pass

- **三维 Kernel — pass**：源二维 grid 已展平为一维 `T.Kernel(B, is_npu=True)`（§0.6 R2/§3.5.1）。
- **GEMM 非整除 / L0C 溢出 — n/a**（无 Cube）。
- **复数 dtype / fp64 缺失**：§3.5.1 如实登记「complex128 不可实现」，证据链完整（见 CG-2026-0019 独立核实）。
- **host 侧约束 — pass**：§1.6.2 #9 的 LUT 建表为常量 materialization，不改动输入张量内容，符合 ascend-constraints.md §4。

### 5. 循环与同步 — pass

- **循环结构 — pass**：§6.1 明确 block 级 `T.Kernel(B)`、级循环 Python `range(t)` 编译期展开（静态边界）、蝶形 `T.Parallel(ng,h)` 内层轴向量化、末级 `T.deinterleave`；§6.3 说明不使用 `T.Pipelined` 的理由（级间 ping-pong 顺序依赖）。
- **同步与模式匹配 — pass**：§7 全部自动同步，与 Developer 模式一致；§7.2 逐同步点列 MTE2→V、级间、V→MTE3 依赖。
- **尾块处理 — pass**：§6.4 `N` 为 2 的幂无列尾块，`B` 由 grid 承接。

### 6. 验证方案 — pass

- **Golden 函数 — pass**：§8.1 给出 `torch.fft.fft(x, dim=-1)` 参考实现，独立于 NPU 算法。
- **L0 测试计划 — pass**：§8.3 L0-1~L0-7 含具体 shape（`[64]`/`[4096]`/`[64,4096]`/`[4,64]`/`[1,4096]`/`[2,4,4096]`）、dtype、验证目标。
- **精度标准 — pass**：c64 atol=rtol=1e-4（源仓 test 口径），并如实标注 c128 不覆盖（L0-7 记 skipped）。

### 7. 完整性与一致性 — warn

- **章节齐全 — pass**：模板 11 章齐全，迁移类 §0 含 0.1–0.7 全部小节。
- **无占位符 — pass**：全篇无占位符标记（占位符关键词检索无命中；花括号仅为 LaTeX 公式记号）。
- **同类引用真实 — pass**：§3.5.3 及 §0 引用的 `examples/`、`testing/`、脚手架副本路径（含 `examples/TileOPs/tileops/kernels/spectral/fft_c2c/_fft_c2c_kernels.py`、`tests/ops/test_fft.py`、`benchmarks/ops/bench_fft.py`）经 Glob/ls 逐条确认真实存在。
- **内部一致 — warn**：主体一致（§3.1/§3.5.2/§4/§5/§6 相互自洽），但存在一处 §3.3 伪代码与 §1.6.1/§3.1 的表述冲突（非阻塞）：
  - §3.3 伪代码 `if h == 1:` 分支只赋 `dr,di`，随后仍执行通用复数乘 `dst_r[...] = dr*wr - di*wi` / `dst_i[...] = dr*wi + di*wr`，而该分支未赋 `wr,wi`——与 §1.6.1 #1（`w=1 → t=v` 直接透传，已机器验证）及 §3.1 步骤 2（"h=1/h=2 走平凡特化"）矛盾；且 §3.3 说明又把已「采纳」的平凡特化称作"可选 refinement"。建议：将 `h==1` 分支改为正确透传（`dst = dr, dst_i = di`，或直接删除该特化保持通用式），并统一 §1.6.1「采纳」与 §3.3「refinement」的口径。

### 8. 算法优化分析 — warn

- **§1.6 章节存在 — pass**：含 1.6.0 调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代 / 1.6.3 向量化轴与布局决策四子节。
- **算法调研存在性与完整性 — pass**：R1 候选表含基线候选 #0（源算法）且 #1–#9 逐项列公式/等价性初判/收益方向；R2 有明确结论「无」；R3 四口径（FLOPs/GM Bytes/扫描遍数/中间缓冲峰值）齐全且基线在表、口径写清；R4 逐候选（计算单元/容量/对齐/静态边界/流水/跨核）评估；调研结论含选定算法（Stockham #4 + LUT #5）与关键依据、调研范围说明。
- **调研结论正确性（独立复核）— warn**：
  - **等价性推演 — pass**：Stockham DIF 迭代式经独立数值复算（N=4 手算逐项匹配 `X[k]=Σx[n]e^{-2πikn/N}`；N=4096 由 `verify_equiv.py` full-FFT 行 err 6.83e-6 对照 torch.fft）；平凡 twiddle 恒等式（`v·1=v`、`v·(-1)=-v`、`v·(-i)=(v_i,-v_r)`、`v·i=(-v_i,v_r)`）逐条代数成立且机器验证 0 ulp；LUT fp64→fp32 cast 等价性 max 0.475 ulp 成立；Karatsuba 等价性成立但其否决依据（Ascend vadd/vmul 同为独立向量指令、op 数 8>6、`CONST-vector-launch-overhead` 按 op 数计时）经核对成立。
  - **负向断言主动复核 — pass**：① R2「无在线变体」——`algorithm-candidates.md` 无 FFT 族条目（无被漏候选），结构判据成立（FFT 为固定蝶形网络、每输出依赖全部输入，不可分解为 running 统计量；源算子 §0.4 亦无 online 手段，无"源已用 online"矛盾）；「流式 FFT/overlap-add」仅适配超长信号，本算子整行可驻留 UB，论证成立。② R4 淘汰项有佐证（#2 bit-reversal→TRAP-C10；#9 DFT-as-matmul 内存 `N²·8B=128MB` + L0C 128KB/L1 512KB 限制，算术复核 4096²×8B≈134MB ✓）。③ 参考表命中行：`kb_search 'fft complex'` 实跑返回 **1 条**（CG-2026-0019 能力缺口条目，非算法候选），证实候选库确无 FFT 族候选——DESIGN §1.6.0 写「kb_search … 无命中」**表述不准**（非阻塞，见建议），但"无算法候选条目"的实质结论成立。
  - **复杂度算术独立复算 — warn（非阻塞）**：GM Bytes 口径复核成立（#0 `B·(2+O(log2N))·N·elem`、#4 `B·2·N·elem`、#1 `B·2·N·elem`，中间缓冲 GM 往返未漏计）；中间缓冲峰值（#0 SMEM `2·min(N,2·threads)`、#4 `O(N)` 2N fp32）成立。**但 R3 表 FLOPs 列系数偏大约 2× 且内部自相矛盾**：#0 写 `~10·N·log2N/行（radix-2 复蝶形 8 实 FLOPs × N/2 × log2N）`——括号式算出 4·N·log2N、与标称 10·N·log2N 不符；#2/#4 写 `~10·N·log2N`、#3 写 `~8·N·log2N`。正确口径（复蝶形 10 实 FLOPs × N/2 × log2N = **5·N·log2N/行**）与 DESIGN 自身 R4 roofline 计算所用 `5·N·log2N`、manifest roofline `flops=batch·5·n·log2n`（`spectral.yaml` L47）一致，即**表格数字应约为现值的 1/2**。因该列不参与算法选择（决策由访存主导，且所有候选同比例受影响），未推翻任何结论，判为建议级。
  - **并列候选裁决路径 — pass**：R3 未出现判定裕度小的并列；#8 四步分解列作 B=1 并行度备选并指向 §5.5/§9（Stage 4 裁决）。
- **调研→设计一致性 — pass**：§1.6.0 选定算法（Stockham DIF + LUT）实际驱动 §1.4 算法描述、§3.1 公式拆解、§6 循环结构，口径一致；迁移任务 §0.5/§0.6 处置与调研选族一致，§0.4 八项优化手段的意图承接逐项说明（O1 保留/O2 强化/O3 重设计/O4 降级承接/O5 舍弃/O6 重设计/O7 重设计/O8 舍弃），**无静默丢弃**。
- **数学等价优化完整性 — pass**：§1.6.1 对选定算法做系统审视，采纳 #1–#4 四项四要素齐全（原式 → 优化后 → 等价性论证 → 收益量化）；#6 Karatsuba 评估后否决并给量化理由、#7 列不适用；无「不可优化」空洞结论。
- **等价性论证正确性 — pass**：逐项独立推演成立，且与 §8.2 精度标准（c64 1e-4）一致；`verify_equiv.py` 重跑结果与 §1.6.1 内嵌表**逐项一致且全 EQUIV_PASS**（D-1）。
- **向量化替代完整性 — pass**：§1.6.2 覆盖 §3.3/§6 中全部循环/标量计算点（bit-reversal 下标、twiddle 复数乘、加减、内核三角、行搬入搬出、stage 级调度、蝶形寻址、block 映射、host 建表），与 §3.3 伪代码、§6.1 循环表交叉核对无遗漏；能向量替代的（#1–#5）已替代。
- **不可替代理由充分性 — pass**：保留的 4 项（#6 编译期展开级调度= tile 级顺序依赖/静态边界；#7 仿射索引=任务映射级；#8 block 索引=block 级；#9 host 常量表=host 元数据且不改输入内容）逐项具体到本算子，无一为"实现简单/照源码写"。
- **替代 API 可行性 — pass**：替代方案 API（vadd/vsub/vmul/deinterleave）与维度 1 同口径确认存在，优先 v-prefix。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 含 5 维评分（整除性/尾 lane 浪费/累加链形态/repack 代价/UB 容量）与 4 候选矩阵（核内 flat+ping-pong / radix 融合 / 核内转置 `[n_groups,h]` / host permute），逐候选量化；弃选 #3 引 `T.transpose` 二轴交换链实测 ~5.4µs（PL-1.1，已核 `layout.md`）、#4 引 host permute ~百 µs（PL-1.3，已核 106–147µs）；**弃选论证前提经亲自打开文档核对**：`docs/Tilelang.language/创建操作/T.transpose.md` §2.3 证实"仅支持二轴交换、多轴需分解链"、dtype 矩阵 fp16/fp32 ✓（与设计论断一致，未把核内重排写成 GM 级）；迁移任务明确"源 GPU 轴是输入非结论"，按 NPU Vector 语义独立评估。末 2 级（h=2,1）以 `T.deinterleave`（docs 佐证，fp32 ✓，`channel_nums=2`）处理，免逐 lane 跨步。
- **实验裁决模式完整性 — pass**：§1.6.3 主选方案判定不依赖未实证常数（依据为 T.deinterleave/T.transpose 文档与 PL 实测），设计显式声明"非未实证常数裁决"；性能估算中的两条未文档化假设（每周期 1 向量指令、fp32 lane 宽 64）已如实标注，不驱动设计选择；B=1 占用风险给四步分解备选。
- **布局决策与实现一致 — pass**：§1.6.3 选定 flat 行 + ping-pong（向量轴=段内层轴 `i`，buffer `[N]`）与 §3.3 伪代码（`T.Parallel(ng,h)` 内层 i；buffer 形状 `(n,)`）、§4.3/§4.5（`(N,)` UB 预算）、§6.1（内层轴向量化）三方一致。
- **与下游章节一致 — pass**：§3.1 以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致（已判定向量化的点不再出现对应逐元素标量循环）。
- （非阻塞建议）① §1.6.0 R3 注写「脊点≈4 FLOP/B」而 R4 写「脊点≈8 FLOP/B」，两者矛盾（独立复算 = 8.4 TFLOP/s ÷ 1.06 TB/s ≈ 8，R4 正确，R3 注应改为 ≈8）；② §1.6.0 R1 未单列 split-radix（已知更低 FLOPs 变体），因算子内存受限且已含 radix-4/8 refinement 不影响结论，建议一句话说明；③ §0.5 O6 依据引 `VP-2026-0140`（实为 `T.erf` 标量循环 segfault 条目）推及 `T.cos/sin` 标量，证据略越界——结论（全 LUT 必需）不受影响，建议改引/补引 `T.vcos.md`/`T.vsin.md` 角度域限制与无标量三角文档。

## CG-2026-0019 独立核实（本次检视重点）

**结论：负向断言证据充分、无夸大/遗漏，判定成立。**

- **源码**：亲自打开 `src/target/codegen_npuir.cc::PrintType`（L303–L418）逐行核对——仅映射 f16(L323)/f32(L329)/f64(L332)/bf16(L343)/i1/i4/i8/i16/i32/i64，**无任何 complex 分支**；complex 类型既非 is_float() 也非整型，直落 L417 `LOG(FATAL) << "Cannot convert type"`。design 引用行号 L303–L418 **准确**。
- **docs dtype 矩阵逐格取值**：`T.vadd.md` §2.2.1 与 `T.vmul.md` §2.2.1 的 dtype 表**无 fp64 列**（仅 fp16/fp32/bf16 中 fp16、fp32 为 √）；`T.vcos.md` §2.2.1 **含** fp64 列且标 **×**（同时 §2.3 证实角度域 `[-1,1]` 高精度、超 `[-2,2]` 急剧劣化）；`T.vsin.md` §2.2.1 无 fp64；`T.deinterleave.md` §2.2.1 fp64 标 **×**。→ **向量元算 fp64 全线不支持**，complex128（实/虚 fp64）在 NPU 无可行向量路径。
- **能力缺口台账**：`kb_search 'fft complex'` 命中 `.agents/evolution/capability-gaps.md` CG-2026-0019（gap_id/layer/evidence 与上述源码/docs 一致），设计对 complex64 → 实/虚 fp32 分解、complex128 无路径（double-double 代价高且未验证）的处置与台账一致。
- **对迁移目标的影响披露**：DESIGN §3.5.1、§8.2、§8.3(L0-7)、§9.1 均如实披露 manifest `fft-4k-c128-b64`（`spectral.yaml` L37）不可实现，并将 complex64 作为可实现的替代/降级路径（c64 覆盖 smoke/unbatched/batched）。**披露诚实、含替代方案**，符合任务要求（不要求设计解决 c128）。未发现 npuir 实际支持复数或 fp64 的任何反证。

## 机械复核结论与判定

- D-1 等价性机器验证：重跑 6/6 `EQUIV_PASS`，与 §1.6.1 内嵌表逐项一致 → **pass**。
- D-5 算术复算：`r3_metrics` pass；`ub_budget`/`l0c_budget`/`core_split` skip 已逐项人工复核（见机械复核段）→ 人工核对通过；`r3_metrics` 数值内部一致性有 2 项非阻塞瑕疵（FLOPs 系数、脊点），不影响结论。
- 阻塞级维度（0/1/2/3/4/6/8）均无 fail；维度 7/8 的 warn 均为建议级。**综合判定：通过。**
