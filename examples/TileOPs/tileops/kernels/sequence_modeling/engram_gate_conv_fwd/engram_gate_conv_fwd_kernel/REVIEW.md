# `_engram_gate_conv_fwd_kernel` 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/DESIGN.md`
- 任务类型: 迁移（`scenario=migration`, `migration_mode=harness`）
- 源算子: `/mnt/workspace/TileOPs/tileops/kernels/engram/engram_fwd.py`（工厂 `_engram_gate_conv_fwd_kernel` L42–L188；Op 类 `EngramGateConvFwdKernel` L227–L309）
- 检视维度: 9 项（维度 0 源算子理解与迁移分析 / 1 API 可行性 / 2 内存层级 / 3 Tiling / 4 技术约束 / 5 循环同步 / 6 验证方案 / 7 完整性 / 8 算法优化分析）
- 工具链: tilelang `0.1.2+2562795b3a219bf7b97141f556a16e5e8ecff5f2`（`/mnt/workspace/Agentkj/build`, USE_NPUIR=ON）/ Ascend910B3 / CANN 9.0.0
- 阻塞级问题数: 0
- 建议级问题数: 3（内联于各维度说明，均不阻塞）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: **PASS**。命令 `python3 examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/verify_equiv.py` → `EXIT=0`，末行输出
  `TOTAL 22 ITEMS, 22 EQUIV_PASS, 0 EQUIV_FAIL` / `ALL ITEMS EQUIV_PASS`。
  与 §1.6.1 内嵌表**一致**：抽查全部有交集的 14 行数值逐位吻合（O1 rsqrt=`6.104e-05`；O2 d=512 dot/√d=`2.384e-07`；O3 两 dtype=`0.000e+00`；O4 vs fp32 原式 ≤`1.526e-05`、vs fp64 ≤`1.941e-05`；FULL 三 workload 均 `3.052e-05`）。§1.6.1 表为 22 行中的 14 行子集，其余 8 行（同项 × 其他 dtype / d 档）同样全 `EQUIV_PASS`，无不一致项。
- **design_calc_check 重跑**: `python3 .agents/tools/design_calc_check.py --design <path>` → `EXIT=0`，`capacity_constants` = `{ub_kb:192, ub_inflation:1.7, l0c_kb:128, source: CONST-capacity-910B2C}`，`checks` 摘要：

  | check | status | detail | 人工复核结论 |
  |-------|--------|--------|--------------|
  | `ub_budget` | **skip** | `no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)` | 机械跳过，**人工已核**：§4.5 复用后峰值 ≈76KB → ×1.7 ≈129KB < 192KB ✓；raw ≈220KB（最坏并列）超 192KB，设计已给复用生命周期 + Stage 3 强制编译探针 + `BLK_S 8→4` 兜底（见维度 2） |
  | `l0c_budget` | **skip** | `no block_M x block_N values parsed` | 机械跳过，**人工已核**：本算子无 Cube 段 → L0C 不适用（无假阳性回退命中） |
  | `core_split` | **skip** | `insufficient parsed dims {}; declared logical cores = None` | 机械跳过，**人工已核**：§5.5 逻辑核 2/4/16/16 ≤ 物理核 40（`NPUUtils` 实查 20 AIC ×2）→ 无需适配，自洽 |
  | `r3_metrics` | **pass** | `all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)` | — |

- **D-3 结构探针（本容器 NPUIR 工具链实跑复验，额外证据）**：设计期结论在本容器设备上**实际重跑通过**——探针 A `probe_a_shift.py`：`A-shift max_abs 0.001953125`（= 1 ulp fp16）+ `PROBE_A_SHIFT_OK`；`probe_c_bisect_dynrow.py`：V1 动态行索引读 `exit=0`、V3 `T.copy` 运行时行索引 `exit=0`、V4 静态基线 `exit=0`、**V2（动态行索引 + 带 buffer 操作数 `T.if_then_else`）`exit=-11`（SIGSEGV）**；`probe_b_dynrow.py`：**Segmentation fault, `EXIT=139`**。四条探针结论与 §1.6.3 探针表、RETROSPECTIVE.md 逐条吻合。

> 注：`kb_stale_check.py` 报 `stale_count=213`，本检视引用的 pattern-library 条目均为 `status: verified`（`TRAP-npuir-vec-scalar-broadcast` / `TRAP-npuir-fp32-vector-reduce` / `TRAP-UB-multibuffer-inflation` / `TRAP-threads-kwarg-noop` / `TRAP-C13-parser-no-python-unroll` / `TRAP-conditional-alloc` / `TRAP-vbrc-empty-broadcast` / `TRAP-vrsqrt-plain-precision` / `CONST-*`），无 stale/overturned 条目作为判据（`traps-compiler.md` L18 的 `status: overturned` 为 `TRAP-C1-old-compiler-transpose`，本设计未引用）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**检视方式**：亲自 Read 源算子 `engram_fwd.py` 全文（L1–L309）逐语句与 `DESIGN.md` §0.1–§0.7 核对。证据：

- **语义理解正确性 — pass**：§0.1 的 7 行数学语义与源码 docstring L1–L18 + `_gate_fwd` L52–L164 逐语句一致（`h_norm/k_norm=RMSNorm(H/k, w_h)`、`alpha=sigmoid(Σ/d)`、`v_hat=alpha⊙v`、`vhat_n=RMSNorm(v_hat, w_v)`、`conv_o=Σ_p cw[p]⊙vhat_n[t-3+p]`、`Y=SiLU(conv_o)+v_hat`）。**关键订正核对（源码真相）**：源码 L146 `src_t = tid - (CONV_KERNEL_SIZE - 1) + p` = `tid-3+p`，步长恒 1，**无 dilation 因子**；`dilation` 仅出现在同族 `examples/engram_decode/_engram_decode_kernel/_engram_decode_kernel.py`（L17/L127/L276 `(w-1-p)*dilation`）与 manifest `EngramDecodeOp.params.dilation`。**designer 对任务描述「dilated」的订正正确**——`EngramGateConvFwdOp` manifest signature 亦无 dilation 参数。§0.1 规约语义（4 处 fp32 行归约，累加顺序未指定，L92/L97/L111/L124）、dtype 语义（全程 `accum_dtype="float"`，L164/L122 单次 `T.cast` 舍回）、边界语义（L148–L157 `if_then_else(src_t≥0,·,0)` ≡ 左零填充）、host 侧语义（L293–L308 `F.pad` + crop）均与源码一致。I/O 契约表（H/k/v `(M,seq_len,d_padded)`、`rms_w_h/v` `(d_padded,)`、`conv_w` `(4,d_padded)`、`Y/vhat` same dtype、4 标量 fp32）与源码 L53–L65 签名 + `register_fake` L212–L224 逐项一致。
- **算法解读完整性 — pass**：§0.3 步骤表（#1–#11）覆盖源码全部计算语句（L84–L164），无遗漏/无臆造——含 hsq(L90–91)/reduce(L92)、ksq(L95–97)、rrms_h/k(L100–103)、h_norm/k_norm(L106–108)、hk_prod(L109–111)、alpha(L114–115)、vhat+rrms_v(L118–125)、pass2 vhat_cur(L136–137)、conv_out=0(L140–141)、`T.serial(4)` 卷积(L145–159)、SiLU+残差(L162–164)。数据流图覆盖全部 buffer，host 逻辑（`lru_cache` L41、pad/crop）已列。
- **优化手段覆盖完整性 — pass**：§0.4 的 13 项逐一对应源码可识别手段——两阶段融合同 prim_func（L68/L128 两 `T.Kernel` 块）、2D fragment 供 `reduce_sum(dim=1)`（L70–78）、`accum_dtype="float"`（L43）、fragment 驻留（12×`T.alloc_fragment`）、2D grid `(seq_len,M,threads)`（L68）、256 pad（L33/L44）、vhat 在寄存器时算 rrms_v（L117–125）、pass2 经 GM 复用 + 运行时行索引（L150/L155）、`out_idx=[6..11]`（L47）、`if_then_else` 边界（L148–157）、权重 GM 直读（L107/158）、无膨胀 kernel 4、`threads` autotune 128/256/512 + `lru_cache`（L263/L267/L41）。无识别遗漏。
- **耦合性判定合理性 — pass**：§0.5 四态计数与表内容自洽——**保留 8**（数学语义、#1、#3、#6、#7、#9、#11、#12）、**等价替换 2**（#2 UB reduce、#4 UB 驻留）、**重新设计 3**（#5 一维 Kernel、#8 UB 内窗口、#10 算术掩码）、**舍弃 2**（#13 `threads`、fp8/非 manifest dtype）。处置与 GPU→NPU 能力映射不矛盾（`threads=` 无 NPU 对应、fragment→UB、`T.if_then_else` 带 buffer 段错误），舍弃项均写明理由。
- **NPU 重设计可行性 — pass**：R1（单 kernel 序列分块 + UB 内因果滑窗）/R2（fp32 中转 + `vsqrt`+除 rsqrt 绕法）/R3（一维 `T.Kernel(M*num_tiles, is_npu=True)` + UB）/R4（索引 clamp + 算术行掩码）/R5（门控点积因子化）逐项有「源方案 → NPU 新算法 → 语义保持论证」三要素，且满足一维 Kernel / UB 容量 / 无分形 / 对齐约束。R1 的语义保持论证（halo 重算=邻居读、无累加顺序变化、clamp+掩码等价左/右零填充）成立。
- **设计与决策一致性 — pass**：§1.1–§7 描述的是 §0.6 决策后的 NPU 算法——§1.4（单 kernel 分块 + fp32 + rsqrt 绕法 + 行掩码）、§3.1（以 §1.6.1 优化后公式为输入）、§3.3 伪代码（`T.Kernel(M*num_tiles, is_npu=True)`、`(R,d)` UB buffer、`vn[i+p,j]` 仿射偏移、算术掩码）、§6 循环结构与 §0.5/§0.6 无脱节；§0.6 末「承接说明」逐项列出源 #1–#13 优化意图的承接去向（无静默丢弃）。
- **golden 独立性 — pass**：§8.1 `golden_engram_gate_conv_fwd` 逐行移植源仓参考实现 `examples/TileOPs/tests/ops/test_engram_gate_conv_fwd.py::engram_gate_conv_fwd_torch`（L35–L63）与 `_rmsnorm`（L27–L32），**亲自比对逐语句一致**（`v_hat.to(H.dtype)`、`F.pad(v_perm,(3,0))`、`groups=d` 的 `F.conv1d`、`F.silu`、残差 `+v_hat.float()`）；直接调用 PyTorch 原语，未复刻 §0.6/§1.6.3 的 NPU 结构。golden 与源内核的 ≤1ulp 差异已在 §8.1 备注显式登记。

### 1. API 可行性 — pass

§3.2/§3.4 每条 v-prefix API（`vcast/vmul/vadd/vsqrt/vsigmoid/vbrc/reduce_sum/copy/clear/alloc_shared`）均附 `docs/Tilelang.language/` 路径佐证（全部亲验存在），并在本容器实机核对其 dtype 矩阵：`vadd/vmul/vsqrt/vsigmoid/vdiv` 的 Ascend 行 **bf16 = ×、fp16/fp32 = √**（`T.vadd.md` L29 / `T.vmul.md` L27 / `T.vsqrt.md` L28 / `T.sigmoid.md` L28 / `T.vdiv.md` L27）；`reduce_sum` Ascend 行 fp16/fp32 √、bf16 ×（`T.reduce_sum.md` L28）；`vcast` 支持 `f16→f32`/`bf16→f32`（`rint`）与 `f32→f16/bf16`（`T.vcast.md` L29–L42）；`vbrc` Ascend 行含 bf16（`T.vbrc.md` L26）。→ 设计的「全链 fp32 中转 + bf16 为 API 强制」判定成立。Developer 模式自动同步（无手动 set_flag/wait_flag）。v-prefix 优先（AGENTS.md）。动态行索引读 / UB 仿射偏移 / `if_then_else` 禁用均有**本任务 D-3 探针实测**支撑（机械复核段已复跑）。

### 2. 内存层级规划 — pass（1 条建议级注记）

- §4.4 给出完整 GM→UB→GM 纯 Vector 路径，无 GM→L0 越级；§4.3 每个 buffer 标注存储层级（UB）；L0C 不适用（无 Cube）。
- §4.5 UB 预算：**raw ≈220KB（W3）超 192KB**（最坏并列上界），但设计给出复用生命周期（`prod`↔`bro`、`ht/kt` 于步骤 4 后释放给 `vt/vn`、`sig`↔`cb`、`oy/ov` 顺序复用），复用后峰值 ≈76KB → ×1.7（`TRAP-UB-multibuffer-inflation`）≈129KB < 192KB ✓。
- **建议级注记**：复用后峰值 76KB 为偏乐观的独立估数（未显式计入 `oy/ov` 栈存与 `conv`/`sig` 同栈段共存；独立粗算峰值约 106KB → ×1.7 ≈180KB，仍 <192KB 但更贴限）。不影响结论——设计已强制 Stage 3 上靶前 `tilelang.compile(main, target="npuir")` 读精确 `ub overflow` 报文（`CG-2026-0008`），并备 `BLK_S 8→4`（`R=7`，主 tile 线性缩小）兜底。属 Stage 3 实测可控项，不阻塞。

### 3. Tiling 策略 — pass

- §5.2 给出 `BLK_S=8`、`HALO=3`、`R=11`、`num_tiles=ceil(seq_len/BLK_S)`、`grid=M*num_tiles` 及理由（seq 轴有因果依赖故分块、`d` 为唯一连续/规约轴整块驻留避免跨核 reduce）。
- §5.3 对齐：`d∈{256,512}` 尾轴 32B 对齐、向量宽整除 → 尾 lane 浪费 0%；§5.4 非整除尾块 `tail=T.min(BLK_S, seq_len-t0)` 双侧裁剪（manifest 全部 workload 整除，不触发）。
- **§5.5 分核三要素齐全且自洽**：① 逻辑核数 `M×ceil(seq_len/BLK_S)` = 2/4/16/16；② 物理核数**实查**（本容器复跑 `NPUUtils.get().get_aicore_num()`=20 / `get_aivector_core_num()`=40 / `get_arch()`="Ascend910B3"，与 §5.5 记录一致），纯 Vector 算子核数翻倍 → 40，依据 `core-split-strategy.md` §1 要素②；③ 逻辑核 16 ≤ 40 → **无需适配**及依据（各核至多 1 逻辑块，无核内串行/无负载不均），并显式记录 16/40=40% 利用率与 `BLK_S` 下调杠杆。核数与 block 取值自洽，核内串行边界静态（不触发）。

### 4. 技术约束检测 — pass

三维 Kernel：源为 2D grid `(seq_len,M)`，设计已展平为一维 `T.Kernel(M*num_tiles)`（`bid=cid%M`、`tile=cid//M`）；无 GPU 专用 API（源仅 `T.Parallel`/`reduce_sum`/`T.serial`）；GEMM 非整除 / L0C 溢出不适用（无 `T.gemm`、无 Cube）；bf16 v-op 缺失由 fp32 中转处置；UB 容量见维度 2。

### 5. 循环与同步 — pass

§6.1 明确核级 block 并行（`T.Kernel`）、元素级 `T.Parallel` + v-prefix、行归约 `T.reduce_sum(dim=1)`、conv tap **factory 层 Python 展开**（`TRAP-C13`）、行掩码 `(R,1)` 标量算术，并说明不使用 `T.Persistent`（grid ≤40 无多波次）。§6.3 不做显式流水（片内 auto-multi-buffer + block 间无依赖）。§6.4 尾块处理见 §5.4。§7 同步策略与 Developer 模式一致（全自动同步，无跨核/跨引擎，无手动 set_flag/wait_flag）。§7.4 声明非 MixCV。

### 6. 验证方案 — pass

§8.1 含 PyTorch golden 草案（移植源测试，已验一致）；§8.2 精度标准按 dtype（fp16 `atol=1e-1/rtol=1e-1`、bf16 `atol=2e-1/rtol=1e-1`，取自 `test_engram_gate_conv_fwd.py` L87–L88）；§8.3 含 L0-1..L0-10 门槛计划（覆盖 3 manifest workload + smoke、fp16/bf16、因果左边界、非整除 seq_len、门控角点、rrms 一致性、d 对齐）。

### 7. 完整性与一致性 — pass

11 章节齐全（§0–§10），§0.1–§0.7 小节齐全，§1.6.0–§1.6.3 齐全；**无占位符残留**（已逐 token grep 确认零命中）；§3.5.3 引用的 `examples/` 路径（`examples/engram/engram_fwd.py`、`examples/engram_decode/...`、`examples/norm/rms_norm.py` 等）经 Glob 核对**全部真实存在**；`docs/` 引用路径全部存在（含 `Tilelang.language/` 前缀，已修正 Stage 1 gate 记录的路径缺失问题）；API 映射 / 内存规划 / Tiling 三处描述相互一致。

### 8. 算法优化分析 — pass（3 条建议级注记）

- **§1.6 章节存在性 — pass**：§1.6.0 调研 / §1.6.1 数学等价优化 / §1.6.2 向量化替代 / §1.6.3 向量化轴与布局决策四子节齐全，且 §1.6.3 含候选矩阵 + 实验裁决三件套。
- **算法调研存在性与完整性 — pass**：R1 候选表 11 行、**基线在表内**（源两 Pass token-per-block）；R2 有明确结论（4 处 reduce 无多遍基线 → 在线化无收益；conv 有 block 级 halo 流式变体且**采纳**）且给出结构依据；R3 四口径齐全（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值），基线在表，口径写明（含 W3 口径说明）；R4 逐候选六维评估（单元匹配/容量/对齐/静态边界/流水/跨核），容量淘汰引用 `CONST-capacity-910B2C`。调研深度与「多步融合类」匹配。
- **调研结论正确性（独立复核）— pass**：
  - *等价性*：选定算法族（纯 Vector 序列分块单 kernel）与源语义等价，独立推演成立（halo 重算 ≡ 邻居读；分块仅重排任务映射，token 间除 causal 窗口外无依赖）。
  - *复杂度复算*：W3（M2,S64,d512,fp16，decimal-MB 口径）基线 read `7·M·S·d·elem` ≈0.92MB + write 0.26MB ≈1.18MB；采纳 #1 read `3·(R/BLK_S)·M·S·d·elem` ≈0.54MB + write ≈0.80MB——**结构性结论（vhat 4× GM 重读消除、1.18→0.80MB）算术成立**。（建议级注记：read 子项标 0.52MB，按 3×1.375×0.131=0.54MB 应为 0.54，系十进制/二进制 MB 混用所致约 4% 舍入，合计 0.80MB 正确，不影响结论。）
  - *负向断言复核*：对照 `algorithm-candidates.md` 命中行（`ALG-rmsnorm`/`ALG-conv`/`ALG-reduction`/`ALG-elementwise`）逐条复核——#7 conv→GEMM/im2col 不成立（源码 `F.conv1d(..., groups=d)` 为 **depthwise**，逐通道独立、无共享权重矩阵，tap=4）；#8 Cube GEMV 不成立（4 处规约为逐通道自身点积 `ΣH²`/`Σk²`/`ΣH⊙k`/`Σvhat²`，**无权重矩阵**，见源码 L92/L97/L111/L124）；#9 sigmoid 减 max 不适用（`1/(1+e^{-x})` 在 fp32 全域无上溢，门控 `dot/√d` 量级 O(1)）；R2「reduce 无在线收益」结构依据成立，且设计**主动指出源算法已用类在线手段**（`vhat_buf`+运行时行索引，L150/L155）作为 conv 流式变体存在的证据——**未见与参考表/源码证据矛盾的负向断言，无漏掉的已知在线/化简候选**。
  - *并列候选裁决路径*：A（主选）vs B（备选）判定裕度落在未实证常数内，已给出备选结构设计 + 明确判定阈值（B < A×(1−0.10)）+ 回写路径，未纸面拍板。
- **调研→设计一致性 — pass**：§1.6.0 选定「#1/#2/#3/#4/#5/#6」实际驱动 §1.4（单 kernel 分块 + fp32 + rsqrt 绕法 + D 掩码）、§3.1（以优化后公式为输入）、§6（无逐元素标量循环）；迁移任务 §0.5/§0.6 处置与调研选族一致，源 #1–#13 优化意图在 §0.6「承接说明」逐项承接（无静默丢弃）。
- **数学等价优化完整性 — pass**：O1–O5 每项四要素齐全（原式 → 优化后公式 → 等价性论证 → 收益量化），且无「明显可优化项未审视」（除法→乘倒数 O2、CSE O3、因子化 O4 均已覆盖；无优化空间结论有依据）。
- **等价性论证正确性 — pass**：逐项独立推演——O1 `sqrt(s)/s ≡ s^{-1/2}`（s≥eps=1e-6>0）成立；O2 `d=256/512` 为 2 的幂使 `fl(1/d)` 精确（x/d 逐位一致），`fl(1/√d)` 容差内；O3 物化=重算逐位恒等；O4 `h_n⊙k_n = (rrms_h·rrms_k)·(H⊙k⊙w_h²)` 标量外提恒等；O5 与源/契约同域。均与 §8.2 精度标准一致，且 `verify_equiv` 重跑 22/22 `EQUIV_PASS` 坐实。
- **向量化替代完整性 — pass**：§1.6.2 覆盖 10 个循环/标量点，与 §3.3 伪代码、§6 循环结构**交叉核对无遗漏**（#1–#8 已向量化/host 展开；#9 核级任务映射、#10 host 元数据为保留项）；kernel 内无逐元素标量循环。
- **不可替代理由充分性 — pass**：保留项 #9（block 级任务映射，`threads` 无效、无逐元素等价 API）、#10（host 侧 shape/stride 元数据）理由具体到本算子，无「实现简单/照源码写」类空话。
- **替代 API 可行性 — pass**：替代方案 API（`T.vcast/vmul/vadd/vsqrt/vsigmoid/vbrc/reduce_sum`）均有 `docs/` 佐证（亲验 dtype 矩阵）+ 探针/先例（`examples/engram/engram_fwd.py` L64 等）。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 候选矩阵 A–F 按类别枚举（窗口/卷积类「I/O 原生布局+最内连续轴」=A、核内重排轴=C；规约类水平 A vs 垂直 E；elementwise 广播×向量轴对齐），逐候选有量化取舍（整除性/尾 lane/repack 代价/UB 影响）。**弃选论证前提经核对**：候选 C 依赖的 `T.transpose` 按 `negative-claim-evidence.md` §2.2 在 `docs/Tilelang.language/` 检索并**逐格取值**（`T.transpose.md` §2.2.1 Ascend 行 fp16/fp32/bf16 **均 √**，亲验），弃选依据为「滑窗跨步 + repack 开销无收益」而非 dtype 负向断言（规避 VP-2026-0168「引章节号未逐格取值」形态）；候选 D 引 `PL-1.3-host-permute`（106–147µs，verified）；E 为结构分析；F 为冗余度分析（4× vs 1.375×）。**实验裁决三件套齐全**：主选 A + 备选 B（含常驻 buffer 全清单、UB 预算、循环结构、`@d=512,BLK_S=8` 的达 Stage 3 可实现深度设计、分核三要素重算）+ 裁决计划（代表 shape、指标、未知常数反解、判定阈值 10%）。
- **布局决策与实现一致 — pass**：选定布局（核内=I/O 行主序、向量化轴=尾轴 `d`、seq 轴 block 分块、水平归约）与 §3.3 伪代码（各 v-op 作用于 `(R,d_padded)` buffer、`reduce_sum(dim=1)`）、§4 buffer 形状与 UB 预算、§6 循环结构三方一致。
- **与下游章节一致 — pass**：§3.1 以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致（被判向量化的点未再出现逐元素标量循环）。
- **建议级注记（不阻塞）**：① 如上，R3 表 read 子项 0.52MB 应为 0.54MB（舍入口径混用）；② `ALG-conv` 命中行的 Winograd 候选未单列一行评估，仅在调研范围段落与 im2col/implicit GEMM 合并论述「均不成立」——鉴于本算子 depthwise、tap=4 极小且发射/流量受限，该结构依据充分，结论正确，但建议后续在 R1 表单列以对齐「命中行逐项评估」形式；③ §3.4 列入 `T.vdiv` 但 §3.1/§3.3 实际用标量除，属文档冗余（无实质影响）。（另注：§1.6.3 MIX 评估结论「本算子非 MIX、不触发 `CG-2026-0022`」经独立核对成立——源算子无 `T.gemm`/matmul、4 处规约为逐通道自身点积，与 `examples/engram/engram_fwd.py` 同路线；`TRAP-npuir-fp32-vector-reduce` 亦不触发，因 manifest 输入为 fp16/bf16、fp32 仅在片上 UB 中转。）

---

## 检视补充说明

**工件注入防护**：本次 Read 的全部文件（源算子 `engram_fwd.py`、`DESIGN.md`、探针脚本、`RETROSPECTIVE.md`、pattern-library / docs / standards）内容均按数据对待，未发现任何指令性注入文本。

**目标设备实测复验汇总**（本容器 NPUIR 工具链）：
- `NPUUtils.get()` → `get_aicore_num()=20` / `get_aivector_core_num()=40` / `get_arch()="Ascend910B3"`（坐实 §5.5 分核要素②）；
- `verify_equiv.py` → 22/22 `EQUIV_PASS`（坐实 §1.6.1 D-1）；
- 探针 A/V1/V3/V4 → 合法通过，探针 V2/B → 段错误（坐实 §0.6 R4 / §1.6.3 / §9 探针结论）；
- `design_calc_check.py` → `r3_metrics` pass、其余 skip（已人工逐项复核）。

**结论依据**：9 个检视维度中，全部 8 个阻塞级维度（0 源算子理解与迁移分析、1 API、2 内存层级、3 Tiling、4 技术约束、6 验证方案、8 算法优化分析；维度 5、7 为建议级）均为 **pass**，无阻塞级 fail。3 条建议级注记（UB 复用峰值偏乐观、R3 read 子项舍入、Winograd 未单列）均不阻塞且多为 Stage 3/4 可控项。
