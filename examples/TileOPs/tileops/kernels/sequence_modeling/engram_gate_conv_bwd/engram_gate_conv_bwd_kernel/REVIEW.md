# `_engram_gate_conv_bwd_kernel` 设计文档检视报告（修订后重新检视，`revision_index=1`）

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/DESIGN.md`（修订版 v1 + 门禁修复）
- 任务类型: 迁移（GPU TileOPs → Ascend NPU npuir，migration-harness）
- 源算子: `/mnt/workspace/TileOPs/tileops/kernels/engram/engram_bwd.py`
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0（3 条上一轮建议均已闭合，见「回归检查」）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑（D-1）**: **PASS** —— `python3 examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/verify_equiv.py` 实际输出 `TOTAL 32 ITEMS, 32 EQUIV_PASS, 0 EQUIV_FAIL` / `ALL ITEMS EQUIV_PASS`；逐行数值与 §1.6.1 内嵌结果表**逐格一致**（抽验：`O1 dot/sqrt(d) (d=512)` max_abs 2.384e-07、`FULL drms_w_h (M2 S64 d512 fp16)` 1.089e-08、`FULL dconv_w (M2 S64 d512 fp16)` 1.634e-07、`FULL dv (M1 S128 d256 bf16)` 6.104e-05）。脚本内部优化项标签已统一为 **O1/O2a/O2b/O3/O4**（与 §1.6.1 表编号一致，v0 的 O2–O5 错位已消除）。**归约 FULL 行覆盖 `drms_w_h/drms_w_v/dconv_w`（9 行，fp32 域门禁 `1e-4/1e-4`）**，是上一轮阻塞项（别名错误）的回归护栏。
- **design_calc_check（D-5）**: JSON checks 摘要如下（`python3 .agents/tools/design_calc_check.py --design <path>`）：
  - `r3_metrics` = **pass**（"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"）。
  - `ub_budget` = **skip**（"no per-level buffer tables parsed … verify manually"）→ **人工已核**：§4.5 按 `d` 分档（`d≤256 → BLK_S=8, R=11`；`d=512 → BLK_S=4, R=7`）给出 K1 ≈72/83KB、K2 复用后 ≈103/108KB（×1.7）——两档均 < 192KB（复算见维度 2）。
  - `l0c_budget` = **skip**（"no block_M x block_N values parsed"）→ **人工已核**：本算子无 Cube，L0C 不涉及（§4.5「L1/L0C: 无 Cube 段，不涉及」）。
  - `core_split` = **skip**（"insufficient parsed dims; declared logical cores = None"）→ **人工已核**：逻辑核数 = `M·ceil(seq_len/BLK_S)`，最大 **32**（W3, `BLK_S=4`）≤ 物理 Vector 核 **40**（`NPUUtils.get().get_aicore_num()=20`×2，§5.5 实查记录）→ 成立（复算见维度 3）。
- **设备探针重跑（D-3，本容器 Ascend910B3 实测，独立复验 §1.6.3 探针表）**: P1 `exit=1`（`Failed to evaluate grid expression 'None'`）/ P2 `maxdiff 0.0` / P3 `WSsum 12.418 ≠ EXPsum −54.592`（错值）/ P4 `exit=−6`（SIGABRT）/ P5 `maxdiff 4.768e-07` / P6a `maxdiff 0.0` / P6b `maxdiff 9.257` —— **与 DESIGN §1.6.3 探针表逐条吻合**（设计 P5 写 4.8e-7、P6b 写 9.26）。
- **别名回归护栏独立复核**: 亲自 Read `verify_equiv.py` 全文——`bwd_npu_f32` 在 `dhn/dkn` 被 `dH/dk` 消费前即计算 `drms_w_h`（L219–222），公式为 `Σ(dhn·H·rr_h + dkn·k·rr_k)`。独立 fp64 复算（本检视自建脚本，随机张量三 workload）：**v1 与源语义 max_abs ≈5.6e-9~7.5e-9（fp32 舍入域）**；把 v0 的错序公式（`Σ(dH·H·rr_h + dk·k·rr_k)`）代入则 **max_abs 0.294/0.417/0.344，相对偏差 ≥153**。即：v1 修正与源 `L282–285`/`L305–308` 一致，且该护栏确实能拦住 v0 错序（反证成立，designer 称「3 EQUIV_FAIL」的可拦性经独立复核确认）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

已亲自 Read 源算子全文（`engram_bwd.py` 499 行）并逐项核对 §0；本轮针对 v1 修订重点复核 `drms_w_h` 语义、`§3.1/§3.3` 一致性并做回归。

- **语义核对（§0.1 ↔ 源码）**: 逐式一致 —— `h_norm/k_norm` 共用 `rms_w_h`（源码 L260/L261）；卷积无膨胀 `conv_o[t]=Σ_p cw[p]⊙vn[t-3+p]`（L94）；`d_silu=dY⊙(σ(c_o)+c_o·σ(c_o)(1-σ(c_o)))`（L112–113）；conv 转置 `d_vnorm[t]=Σ_p cw[p]⊙d_silu[t+3-p]`（L167）；RMSNorm 反向除**真 d**（`/float(d)` L203/L277/L300、`/T.sqrt(float(d))` L256）；`vhat/alpha/rrms_*` 为输入不重算、仅重算 `conv_o`（L91–107）。**`drms_w_h` 双路（本轮回合重点）**：§0.1 写 `Σ_t(dh_norm⊙H⊙rrms_h + dk_norm⊙k⊙rrms_k)`，与源 `L282–285`（`ddot·k_norm·H·rrms_h`）与 `L305–308`（`ddot·h_norm·k·rrms_k`）逐项一致。行号引用已订正为 `L203/L277/L300`（上一轮建议 3 已闭合）。
- **I/O 核对（§0.2 ↔ 源码 L51–71/L318–338 + `out_idx=[12..19]`）**: 12 输入 + 8 声明输出（6 契约 `dH/dk/dv/drms_w_h/drms_w_v/dconv_w` + 2 内部 workspace `dvhat_buf/dvhat_out`）；dtype（`dH/dk/dv` same_as dY、`drms_w_*/dconv_w` fp32、`alpha/rrms_*` `(M,seq_len)` fp32）、shape（host `d_padded→d` crop）与源一致。
- **算法步骤覆盖（§0.3 步骤 0–19 ↔ 源码逐语句）**: 覆盖 zero-init（L73–76）、Pass1（L79–117）、dconv（L123–145）、Pass1b（L150–212）、Pass2（L215–314）全部计算语句，无遗漏/臆造；跨块 GM 交接（`dvhat_buf`/`dvhat_out`）与跨 token 归约（dconv 独立 kernel + `atomic_add`）均标注（§0.3 L157–158）。
- **优化手段覆盖（§0.4 16 项 ↔ 源码）**: 16 项均可从源码识别并逐项标注硬件依赖；#7 引用源注释 L120–122（`atomic_add` lowering 缺陷）真实存在（源码 L119–122）。
- **耦合性判定（§0.5）合理性**: 四态处置（保留 6 / 等价替换 2 / 重设计 6 / 舍弃 2 + 重整 1）依据充分；与 `TRAP-threads-kwarg-noop`、探针 P1/P4/P6b、`CG-2026-0022`、`docs/Tilelang.language` dtype 矩阵不矛盾；重设计条目与 §0.6 R1–R6 一一对应。
- **重设计可行性（§0.6 R1–R6）**: 每项含「源方案 → NPU 新算法 → 语义保持论证」三要素；满足一维 Kernel（`T.Kernel(M*num_tiles, is_npu=True)`）、UB 层级、32B 尾轴对齐、静态边界（`BLK_S/R/K=4` 编译期常量）；承接说明逐项列出源优化意图（#1→R1…/#16→R1），无静默丢弃。
- **设计–决策一致性（§1–§7 ↔ §0.5/§0.6）**: §4/§5/§6/§7 均为 3-kernel 闭包 + UB + fp32 中转 + clamp/算术掩码的落地；**上一轮唯一脱节点（§3.3 K2 权重梯度落错）已修正**——§3.3 K2 L888–895 的 `drms_w_h` 部分和现前置于 `dhn/dkn` 就地覆写（L897–910 覆写 `dhn→dH`、L912–925 覆写 `dkn→dk`）之前，与 §3.1 步骤 15→16→17→18、§1.6.1 O2、§0.1 一致。
- **golden 独立性（§8.1）**: golden 以 §0.1 语义为依据，**直接移植源仓 `test_engram_gate_conv_bwd.py::ref_engram_gate_conv_bwd`（autograd 反向 + `F.conv1d` 左零填充 + `mean/rsqrt/sigmoid/silu`）**，不复刻 §0.6/§1.6.3 的 NPU 结构 —— 独立性满足。

### 1. API 可行性 — pass

§3.2/§3.4 列出的 API 均有 docs + 先例/探针佐证：`T.alloc_shared`/`T.vsigmoid`/`T.Kernel(..., is_npu=True)` 有同算子先例 `examples/engram/engram_bwd.py`（存在性已核对）；`T.reduce_sum(dim=1/dim=0)`、`T.copy`、`T.clear`、`T.vcast`、`T.vmul/vadd/vsub`、`T.vbrc` 均有 `docs/Tilelang.language/` 对应文档（`规约操作/T.reduce_sum.md`、`内存操作/T.copy.md`·`T.clear.md`、`数据类型转换操作/T.vcast.md`、`数学操作/T.vmul.md`·`T.vadd.md`·`T.vsigmoid.md`、`shape操作/T.vbrc.md`）。**独立核对 `T.vbrc` 广播形态**：`docs/Tilelang.language/shape操作/T.vbrc.md` 明确 `(1,N)→(M,N)` 与 `(M,1,K)→(M,N,K)` 合法——§3.3 的 `(R,1)→(R,d)`（掩码/`rrms_v`）与 `(1,d)→(R,d)`（权重）**均在文档支持范围内**。Developer 模式用自动同步 API，模式匹配；v-prefix 优先（AGENTS.md）。**非 MIX 判定**：源算子无任何 `T.gemm`/matmul（仅 `T.Parallel/serial/reduce_sum/atomic_add/if_then_else/cast/exp/sqrt`），故 `CG-2026-0022` 不触发 —— 判定正确（§1.6.3 MIX 评估）。负向 API 断言（`T.atomic_add` 不可用 / 多 `T.Kernel` 不支持 / 带 buffer `T.if_then_else` 段错误）均有本设备探针 P1/P4/P6b 佐证。

### 2. 内存层级规划 — pass

- **搬运路径（§4.4）**: K1/K2 均 GM→UB→GM，无越级（无 GM→L0 直连）；K3 grid=1 先 `T.copy` 载入 `ws_*` 再 `reduce_sum`；buffer 层级标注齐全（§4.3 逐 buffer 标注 UB / GM workspace / dtype）。
- **UB 预算（§4.5）**: 按 `d` 分档 —— `d≤256`（`BLK_S=8, R=11`）K1 raw ≈72KB、K2 复用后 ≈60KB raw ×1.7 ≈**103KB**；`d=512`（`BLK_S=4, R=7`）K1 raw ≈83KB、K2 复用后 ≈64KB raw ×1.7 ≈**108KB**。**独立复算（保守 live-set 计 8×(BLK_S,d)）**：d=512 `8×4×512×4 + 7×512×4 ≈ 78KB` raw ×1.7 ≈133KB < 192KB；d=256 `8×8×256×4 + 11×256×4 ≈ 75KB` raw ×1.7 ≈128KB < 192KB —— **两档均 < 192KB**，上一轮 `d=512` 默认 `BLK_S=8` 越限（≈255KB）的问题已由 shape 分派闭合。§4.3 已把 `bro(R,d)` 与 `bro_s(BLK_S,d)` **分别列入预算**（上一轮少计问题已修）。§4.5 仍保留强制 Stage 3 编译探针（`tilelang.compile(k1/k2, target="npuir")` 读 `ub overflow, requires N bits`）+ `BLK_S` 回退链（8→4 / 4→2）作为二次确认（`CG-2026-0008`：手工预算 ≠ 可编译证明），风险受控。
- **L0C**: 无 Cube，不涉及。

### 3. Tiling 策略 — pass

- **分核策略三要素（§5.5）齐全**: ① 逻辑核数 = `M·ceil(seq_len/BLK_S)`（K1/K2）、K3=1；② 物理核数**实查**（`NPUUtils.get().get_aicore_num()=20`、`get_aivector_core_num()=40`，纯 Vector 翻倍，记录查询代码与返回值）；③ **逻辑核数 ≤ 物理核数 → 无需适配**（最大 **32** ≤ 40）。与 `core-split-strategy.md` §1/§2.2 一致，核数与 `BLK_S` 自洽（W3 `d=512` 取 `BLK_S=4` → 32；W2=4、W1=2、W4=16、K3=1）。
- **Block 划分（§5.2）**: `BLK_S = 8 if d≤256 else 4`、`R = BLK_S+HALO`（11/7）、`d` 不划分；对齐 —— `d∈{256,512}` 整除向量宽（fp32 ×4 / fp16 ×8），`(R,d)` 尾轴 32B 对齐；manifest `seq_len∈{16,32,64,128}` 整除 `BLK_S` → 无尾块；通用尾块策略见 §5.4（`tail=T.min(BLK_S, seq_len-t0)` 双显式 slice + clamp + 行掩码）。

### 4. 技术约束检测 — pass

一维 `T.Kernel(..., is_npu=True)`（源 2D grid `(seq_len,M)` 已展平，`bid=cid%M`、`tile=cid//M`）；无 `T.gemm`（不涉及 GEMM 非整除 / L0C / 分形）；UB 192KB 已在维度 2 处理；单 prim_func 多 `T.Kernel` 不支持已由探针 P1 确证并给出 3-kernel 闭包方案；`TRAP-npuir-fp32-vector-reduce` 不触发（manifest 输入 fp16/bf16，fp32 仅片上 UB 中转）；bf16 v-op 缺失由全链 fp32 中转处理。

### 5. 循环与同步 — pass

§6 明确：核级 = `T.Kernel(M·num_tiles, is_npu=True)`（K1/K2）+ `T.Kernel(1, is_npu=True)`（K3）；元素级 = `T.Parallel(R,d)`/`(BLK_S,d)` + v-prefix；行归约 `reduce_sum(dim=1)`；token 归约 `reduce_sum(dim=0)`；conv/转置 tap 与行掩码走 **factory 层编译期展开**；无 `T.Pipelined`、无核内逐元素串行循环 —— 与 §1.6.2 结论一致。§7 同步为 Developer 自动同步（kernel 内）+ kernel 间 host 顺序启动（同 stream，数据依赖由启动序保证），无手动 flag —— 与编程模式匹配。§6.4 尾块策略与 §5.4 一致。

### 6. 验证方案 — pass

§8.1 golden 独立（移植源仓 ref，见维度 0，autograd 反向与 NPU op 序列无关）；§8.2 精度标准 fp16 `atol=2e-1`、bf16 `atol=3e-1`、`rtol=2e-1` —— 与源仓 `test_engram_gate_conv_bwd.py`（L95–96）一致；§8.3 L0-1..L0-12 覆盖 3 full workload + smoke、fp16/bf16、卷积左边界（L0-5）、转置右边界（L0-6）、非整除 `seq_len`（L0-6）、门控角点（L0-7）、**权重梯度归约（L0-8）**、**确定性双跑（L0-9，对齐测试 L97–101 的 `torch.equal`）**、bf16 全链（L0-10）、`d` 对齐敏感性 + K2 UB 编译门禁（L0-11）、部分和/combine 一致性（L0-12）。**D-1 `verify_equiv.py` 的 FULL 对照已覆盖全部 6 个契约输出**（含 `drms_w_h/drms_w_v/dconv_w`，上一轮覆盖缺口已闭合）。

### 7. 完整性与一致性 — pass

- 模板 11 章节齐全（含 §0 且 0.1–0.8 小节齐全）；§3.5.3/§0.7 引用的 `examples/` 路径**逐一存在**（已 Glob 核对：`examples/engram/engram_bwd.py`、`examples/engram_gate_conv_fwd/_engram_gate_conv_fwd_kernel/_engram_gate_conv_fwd_kernel.py`、`examples/engram_decode/_engram_decode_kernel/_engram_decode_kernel.py`、`examples/norm/rms_norm.py`、`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/`、`examples/TileOPs/tests/ops/test_engram_gate_conv_bwd.py`、`examples/TileOPs/tileops/manifest/sequence_modeling.yaml` 均存在）。
- 全文无模板占位符 / 未填章节 / 待办标记（已逐行扫描确认命中计数为 0）。
- **内部一致性**: §3.1 步骤 16/17/18 与 §3.3 K2 伪代码对 `drms_w_h`/`dH`/`dk` 的公式与次序**一致**（步骤 16 前置于覆写，与 §1.6.1 O2 优化后公式一致）—— 上一轮 fail 点已闭合。编号一致性：§1.6.1 表（O1–O4）与 `verify_equiv.py` 输出标签一致（上一轮 warn 已闭合）。§0.1 行号引用正确（上一轮 warn 已闭合）。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性**: `1.6.0/1.6.1/1.6.2/1.6.3` 四子节齐全。
- **算法调研存在性与完整性（§1.6.0）**: 四问齐全 —— R1 候选表 12 行**含基线候选 #0**（源算法）；R2 在线算法**明确结论**（3 处行规约无在线收益 / 卷积·转置 block 级 halo 滑窗（采纳）/ 权重梯度全域求和不可在线化，均附结构依据）；R3 复杂度表**四口径**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值，基线在表，口径写清）；R4 逐候选硬件亲和评估（含容量/带宽淘汰引用 constants.md 条目）。调研深度与算子（多步融合 + 跨 token 归约类）匹配。
- **调研结论正确性（独立复核）**: ① 负向断言逐条对照 `tilelang-op-design/references/algorithm-candidates.md` —— 「3 处行规约无在线收益」↔ `ALG-rmsnorm`/`ALG-reduction`（Welford 只适用统计量，不适用纯点积和）；「conv→GEMM 不成立」↔ `ALG-conv`（**depthwise 逐通道无共享权重矩阵**，im2col/implicit GEMM 退化为 d 个独立 1×4 GEMV，判定成立）；「归约用 Cube GEMV 不成立」↔ 无权重矩阵 + `CG-2026-0022`；「`atomic_add` 不可用」↔ 探针 P4（SIGABRT，本轮已复跑确认）。**未发现被漏掉的已知候选，负向断言未被推翻**。② 复杂度算术复算（W3 FLOPs `(128·12288+20·128)×2.5=3.94M`、manifest bytes 1.34MB）与设计一致，中间缓冲 GM 往返未漏计。③ 亲和性淘汰有 `CG-2026-0022` 与探针 P4 佐证。④ **并列候选（A vs B）有实验裁决三件套**（主选 A + 结构完整的备选 B（常驻 buffer 全列 / 循环 / 分核三要素 / UB 预算）+ 代表 shape/指标/判定阈值 `B < A×0.9`/回写路径）。
- **调研→设计一致性**: §1.4/§3.1/§6 均由 §1.6.0 选定算法（纯 Vector 3-kernel 确定性分块 + fp32 中转 + 常量倒数折叠 + 标量因子化 + CSE）驱动；迁移任务 §0.5/§0.6 处置与调研选族一致；源优化意图承接逐项说明（§0.6 承接说明，无静默丢弃）。**上一轮唯一脱节点（§1.6.1 O2 优化后 `drms_w_h` 公式未忠实落地到 §3.3）已修正**——§3.3 K2 L888–895 现与 §1.6.1 O2／§3.1 步骤 16／§0.1／源码 L282–285·L305–308 一致。
- **数学等价优化完整性（§1.6.1）**: O1 常量倒数折叠 / O2 标量因子外提 / O3 CSE 物化 / O4 fp32 中转，四要素（原式→优化后→等价论证→收益量化）齐全，无进一步等价空间的结论与依据给出。优化后公式明确列为 §3.1 的输入。
- **等价性论证独立推演**: 逐项独立推演成立 —— O1 `x·fl(1/d)` vs `x/d`（d 为 2 的幂时 `fl(1/d)` 精确，逐位一致，D-1 复跑 0.0）；`dot·fl(1/√d)` 与 `dot/√d` 容差内等价（≤2.4e-7）；O2 标量对向量/求和线性（实数恒等）+ 舍入（≤1.9e-6）；O3 同一确定性 fp32 计算逐位一致（0.0）；O4 与源/契约同域。**`verify_equiv.py` 重跑 32/32 EQUIV_PASS，与内嵌表一致**（见机械复核）。
- **向量化替代完整性（§1.6.2）**: 覆盖 11 个计算点（与 §3.3 伪代码、§6 交叉核对）—— #1–#8 已全部向量化或 host 层展开（替代 API 均有 docs/先例佐证，优先 v-prefix）；#9 核级任务映射 / #10 跨 block combine / #11 host 元数据 3 处保留，**理由具体到本算子**（block 级任务映射无逐元素等价 API；跨 block 求和不能在带内消除；host pad/crop 为元数据），**无「实现简单」类理由**。§6 确认 K1/K2 内无逐元素标量循环。
- **向量化轴与布局决策（§1.6.3）**: 候选矩阵 A–H（**A 主选 + B 备选 + C–H 弃选**），逐候选量化评分（整除性/尾 lane 浪费/repack 代价/UB 影响）；**弃选论证证据合规** —— 候选 C 依赖的 `T.transpose` **已枚举并亲自读文档**：`docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1 dtype 行 fp16/fp32/bf16 均 **√**，且限制为「仅二轴交换」—— 设计明确「非 dtype 障碍，仅滑窗跨步 + repack 无收益」，**正确规避 VP-2026-0168「引章节号未逐格取值」形态**；候选 D 引 `layout.md` PL-1.3（status=verified，106–147µs 净亏）；候选 E/F 为结构/冗余度分析；候选 G/H 为数学不成立 + `CG-2026-0022`。**布局决策与 §3.3/§4/§6 一致**（向量化轴 = 尾轴 `d`，核内 = I/O 行主序无 repack，行归约 dim=1、token 归约 dim=0）。
- **别名纪律（本轮回归重点）**: §3.3 实现注意 #13 与 §9.1 R-17 明确写入「读取先于覆写（`PL-1.23-vop-chain-inplace-alias`）」纪律，与 pattern-library `elementwise.md` PL-1.23（status=verified，前提铁律「读取源数据的前置 op 必须发生在覆盖该 buffer 之前」）语义一致；`verify_equiv.py` 的 9 行 reduction FULL 为回归护栏，反证成立（见机械复核）。

## 回归检查（v1 / 门禁修复是否破坏 v0 已通过项）

逐项复核 v0 已通过项，未发现回归：**§0.1–§0.4 迁移三问解读**（语义/I/O/算法步骤/优化手段，本轮重新逐条比对源码，一致）；**§0.5 耦合性判定**（保留 6/等价替换 2/重设计 6/舍弃 2，未改）；**§0.6 R1–R6 重设计**（三要素齐全，未改）；**探针 P1–P6b**（本轮独立复跑，逐条吻合 §1.6.3）；**§1.6.0 调研四问 + D-2 roofline**（未改，独立复算一致）；**§1.6.2 向量化 8/3 结论**（未改）；**§5.5 分核三要素**（W3 逻辑核数因 `BLK_S` 分派由 16→32，仍 ≤ 40，成立）；**golden 独立性**（未改，仍 autograd 移植）。

v1 的三处建议处置亦已闭合：① `BLK_S` 按 `d` 分档 + §4.3/§4.5 补 `bro_s` 预算（维度 2）；② `verify_equiv.py`/§1.6.1 编号统一 O1–O4（维度 7/8）；③ §0.1 行号订正 L203/L277/L300（维度 0/7）。门禁修复（`EQUIV_FAIL` 字面量移出 §1.6.1 区域至 §0.8）后，`EQUIV_FAIL` 字样仅出现在 §0.8 修订记录（DESIGN.md L288/L310），不影响 §1.6.1 结果表（全文 32 行均为 `EQUIV_PASS`）与机器复核。

---

> 检视依据：源算子 `/mnt/workspace/TileOPs/tileops/kernels/engram/engram_bwd.py`（全文 499 行 Read）、`examples/TileOPs/tests/ops/test_engram_gate_conv_bwd.py`、`examples/TileOPs/tileops/manifest/sequence_modeling.yaml`、本仓 `docs/Tilelang.language/`（`创建操作/T.transpose.md`、`shape操作/T.vbrc.md`、`规约操作/T.reduce_sum.md`、`数据类型转换操作/T.vcast.md`、`数学操作/*`、`内存操作/*`）、`pattern-library`（`elementwise.md` PL-1.23、`cases.md`、`constants.md`、`layout.md`）、`capability-gaps.md`、`algorithm-candidates.md`、`.agents/skills/_shared/standards/*`；设备实测：`verify_equiv.py`（32/32 EQUIV_PASS）、`design_probe_bwd_structure.py`（P1–P6b 复跑）、`design_calc_check.py`（工具链 `0.1.2+2562795b3a219bf7b97141f556a16e5e8ecff5f2` / Ascend910B3 / CANN 9.0.0）。
