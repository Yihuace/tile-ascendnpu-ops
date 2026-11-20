# _engram_decode_kernel 设计文档检视报告（revision_index=1，v1 重新检视）

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/engram_decode/_engram_decode_kernel/DESIGN.md（修订后 v1）
- 任务类型: 迁移（migration-harness）
- 源算子: /mnt/workspace/TileOPs/tileops/kernels/engram/engram_decode.py（Op 类 /mnt/workspace/TileOPs/tileops/ops/engram_decode.py）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0（1 处非阻塞计数残留已内联记录，不构成意见项）

> **本轮性质**：对 v0 检视「不通过」（2 阻塞 + 3 建议，集中在 §1.6.3）修订后的**重新检视**。逐项复核 2 处阻塞项是否真正闭合、回归检查 v1 是否破坏 v0 已通过项，并对全部 9 维度重新独立检视（不因「上一轮已过」而豁免）。

## 机械复核（D-1/D-5）

- **verify_equiv 重跑（D-1）**: **PASS**。`python3 examples/engram_decode/_engram_decode_kernel/verify_equiv.py` 实际输出 `TOTAL 28 ITEMS, 28 EQUIV_PASS, 0 EQUIV_FAIL / ALL ITEMS EQUIV_PASS`，与 §1.6.1 内嵌结果表**逐行一致**（抽核：O1 6.104e-05、O1 `(·)^3` 1.920e+02、O2 d=512 `dot·fl(1/√d)` 3.906e-03、O3 三 dtype 0.0、O4 各行 ≤3.052e-05、FULL y/st 三 workload 违反数全 0）。脚本存在、执行成功、结果与内嵌表一致 → 等价性机器验证未失效。
- **design_calc_check 复算（D-5）**: `python3 .agents/tools/design_calc_check.py --design examples/engram_decode/_engram_decode_kernel/DESIGN.md` JSON `checks` 摘要：
  - `ub_budget`: **skip** — "no per-level buffer tables parsed from the memory plan"（§4.5「类别×W1×W3××1.7」表头无 buffer 名列，工具未解析）。**人工复核**：§4.5 = 17×(d,)f32（W3 34KB）+ ~4×(d,)dtype（4KB）+ `st_ub`(19.5KB) + 归约 fragment(≈2KB) ≈ **59KB → ×1.7 ≈ 101KB < 192KB ✓**（结论成立，与 §1.6.3 主选 A、§9 R-8 一致）。
  - `l0c_budget`: **fail（解析器假阳性，仍在）**。根因（本 Agent 用 `design_calc_check` 内部函数定位）：`check_l0c_budget` 在 §5「Tiling」段未匹配到 `block_M/block_N`/`bm/bn`，回退正则在**全文**匹配到 **§1.6.3 候选 C 行（DESIGN.md L470）** 的写实串 `1024×512 → 多 tile 链 … fp16/fp32/bf16`，把**权重矩阵 shape `d_mem×d = 1024×512`** 误当 `block_M×block_N`，算得 2048KB。**真实 L0C = `c_frag (d_padded,1) f32` = 512×1×4 = 2KB ≪ 128KB ✓**。**可消除性**：该假阳性可由 ① 改写 L470（避免 `<num>×<num>` 与 `fp32` 同行）或 ② 在 §5 增显式 `block_M=/block_N=` 声明（本 GEMV 语义为 M=d_padded、N=1，工具首分支正则不支持该语义）消除；本质为工具回退正则脆弱（advisory）。**附更正**：上一轮 RETROSPECTIVE 归因「§5.3 的 L1 tile `128×512` 被误当 block_M×block_N」与实测不符——本 Agent 复现确认实际命中的是 **§1.6.3 候选 C 行（L470）的 `1024×512`**（`128×512` 仅出现在 §5.3/L913，非首个匹配）。维度 2 依人工复算判 pass（`l0c_budget` fail 为解析器假阳性，非设计缺陷）。
  - `core_split`: **skip** — "insufficient parsed dims {}; declared logical cores = None"。**人工复核**：§5.5 `num_logical_kernels = ceildiv(B,1) = B`（1/1/4/8），物理核数 = `NPUUtils.get().get_aicore_num()` **实查 20**（混合算子取 AI Core 返回值，非纯 Vector 的 ×2=40），B ≤ 20 → 无需适配 ✓。
  - `r3_metrics`: **pass** — "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

> 本 Agent 亲自 Read 源算子全文（`engram_decode.py` L1–L393）与提取副本，逐项与 §0 核对。

- **语义理解正确性 — pass**：§0.1 数学语义（`k=e@W_K`、`v=e@W_V`、`RMSNorm(h/k,w_h)`、`alpha=sigmoid(⟨h_n,k_n⟩/√d)`、`v_hat=alpha·v`、`v_n=RMSNorm(v_hat,w_v)`、`conv_out=Σ_{p} cw_p⊙win_p`、`win_p=state[L-(w-1-p)δ]`、`win_{w-1}=v_n`、`new_state=shift_left⊕v_n`、`y=SiLU(conv_out)+v_hat`）与源码 docstring L5–L14 及 `@T.macro _decode_fused` L82–L213 **逐语句一致**。规约语义（4 处 `T.reduce_sum` L134/142/150/160 沿 d 轴 fp32 行内归约、累加顺序未指定）与源码一致。`mean` 按**真 d**（源码 L135/L143/L161 `sumsq/float(d)+eps`）——§0.1 明确标注。dtype 语义「fp16/bf16 → fp32 计算域 → 单次舍回」（源码 `accum_dtype="float"` L72 + `T.cast(...,dtype)` 仅于 L203/L210）与源码一致。边界语义（`d_padded` 零填充不计入、越界膨胀位落左零填充区贡献 0、`max_conv_len≥δ(w-1)` 由 `__init__` L321–L326 强制）与源码 L171–L177 注释及 L321–L326 一致。I/O 契约（§0.1 表）与源码签名 L82–L94 及 wrapper L342–L393（host 左 pad + 尾维 pad + crop）一致；输出 shape `(B,d)`/`(B,max_conv_len,d)` 与 `source_output_shape` 一致。
- **算法解读完整性 — pass**：§0.3 步骤分解表（1–11）覆盖 `_decode_fused` L95–L213 的**全部计算语句**（载入 e L115–116、GEMV L119–125、载入 h L128–129、RMSNorm(h) L132–137、RMSNorm(k) L140–145、标量门控 L148–151、v_hat L153–155、RMSNorm(v_hat) L157–163（原地覆写）、conv L179–194、状态左移+append L199–203、SiLU+残差 L208–213），**无遗漏、无臆造**。数据流图覆盖全部 buffer；host 侧逻辑（`F.pad`/crop）已列出（§0.1/§0.3）。步骤 8「原地覆写 v_hat 丢失 → 步骤 11 重算 alpha·v」的解读与源码 L163/L211 一致。
- **优化手段覆盖完整性 — pass**：§0.4 列 13 项（单 kernel 融合 / fragment 驻留 / grid=(B,) / 全程 fp32 / `threads` autotune+`lru_cache` / 256 pad / 2D fragment 供 reduce_sum / 重算 alpha·v / `T.Parallel` 逐元素 / GEMV 串行 FMA 链 / 状态左移串行 / 膨胀索引编译期 / 左零填充），逐项标注硬件依赖与层级，未见把实现层误判为算法层。**v1 更正 #2 `T.alloc_fragment` 计数为 ×14**：源码 L97–L112 实为 6 向量（`e_local/k_local/v_local/h_local/vhat_local/conv_out`）+ 4 个 2D（`hsq_2d/ksq_2d/vsq_2d/hk_2d`）+ 4 标量（`sumsq_h/sumsq_k/sumsq_v/dot_hk`）= **14**，与源码一致（v0 的 ×11 已修正）。
- **耦合性判定合理性 — pass**：§0.5 每条含四态处置 + 依据，与 GPU→NPU 能力映射及 ascend-constraints 不矛盾。#5 `threads`→舍弃（依据 `TRAP-threads-kwarg-noop` + 脚手架 K3/K9 已去 `threads`）；#2 fragment→等价替换（UB/L0C）；#10 GEMV 串行链→重新设计（Cube，GEMV 占 FLOPs 99.4%）；#4 fp32→保留并强化（bf16 API 强制）；#7/#11 等价替换。**bγ16 dtype 缺失的举证**引用 `T.vadd/vmul/vdiv/vexp`、`T.vmin`、`T.vcast` §2.2.1（本 Agent 已据设计所引路径核对为真实存在），无「GPU 直觉先验否决」。
- **NPU 重设计可行性 — pass**：R1–R5 每项含「源方案 → NPU 新算法 → 语义保持论证」三要素。R1（Cube `T.gemm`，M=d_padded/K=BK/N=1）满足一维 Kernel、分形（M≥16、K=BK=128≥32）、L0C 容量（2KB）；R2（fp32 中转 + `vsqrt/vdiv` rsqrt 绕法）有实数恒等论证（`sqrt(s)/s ≡ s^{-1/2}`，`s≥eps>0`）；R3（一维 Kernel + UB + L1/L0C）、R4（CSE 物化 ≤2KB）、R5（二维区域 `T.copy`）均有语义保持论证。**累加顺序变化**（R1：串行链→K 分块树形）明确论证 fp32 同域 + fp64 差分 ≤3.1e-5 违反数 0。
- **设计与决策一致性 — pass**：§1.4/§3.1/§3.3/§6 描述的是 §0.5/§0.6 决策后的 NPU 算法（Cube GEMV、fp32 中转、`vsqrt/vdiv`、二维区域搬运、CSE 物化），无「§0 判重设计、§3 照抄源方案」的脱节。§0.5 判「舍弃 `threads`」，§3.3/§5.5/§6 确无 `threads`。
- **golden 独立性 — pass**：§8.1 golden 移植源仓参考 `test_engram_decode.py::engram_decode_step_torch`（本 Agent Read L38–L86 逐行核对），**仅用 PyTorch `@`/`mean`/`rsqrt`/`sigmoid`/`silu`/`F.pad`**，未复用 §3.3 的 NPU op 序列、未复刻 §0.6 的 Cube/`vsqrt+vdiv`/二维 copy 结构 → 验证独立性成立。

> **回归检查（v1 未破坏 v0 已通过项）**：§0.1–§0.7 与 `history_version/design_v0.md` 逐字相同（三问解读 / 耦合性 / 重设计 / 承接说明未变）；§1.6.1 采纳项 O1–O5 数学式未变（verify_equiv 无改动，重跑 28/28）；§1.6.2 向量化结论不变；§5.5 分核三要素不变。**无回归。**

### 1. API 可行性 — pass

- §3.2/§3.4 所列 API（`T.gemm`、`T.vcast`、`T.vmul`/`T.vadd`、`T.vsqrt`/`T.vdiv`、`T.vsigmoid`、`T.vbrc`、`T.reduce_sum`、`T.copy`、`T.alloc_ub`/`T.alloc_shared`/`T.alloc_fragment`、`T.Pipelined`、`T.Kernel`）均在本仓 `docs/Tilelang.language/`（本 Agent 抽核 `规约操作/T.reduce_sum.md`、`创建操作/T.transpose.md`、`数学操作/`、`线性代数操作/T.gemm.md` 存在）与 `examples/` 有佐证：`examples/gemv/example_gemv.py::naive_gemv_high_perf`（Developer `is_npu=True` + `T.gemm` GEMV，M=BLOCK_N/N=1 合同实跑——本 Agent Read 确认）、`examples/norm/rms_norm.py`（`T.reduce_sum(dim=1)` L43 + `T.alloc_shared/alloc_fragment`）、`examples/engram/engram_decode.py`（同算子 NPU 版 `vsigmoid/vsqrt/vdiv/reduce_sum`）、`examples/flash_attention/flash_attn_npuir_dev.py`（Developer + `T.gemm` + v-op 混排）。
- **v-prefix 优先**：设计全用 v-prefix（`vmul/vadd/vsqrt/vdiv/vsigmoid/vcast/vbrc`），无 legacy `npuir_xxx`。
- API 与模式匹配：Developer 模式用自动同步（§7），无手动 `set_flag/wait_flag`。
- **待设备验证点已显式标注**：`T.gemm` 的 `a_transpose=True` 未在 examples/gemv 直接实证（示例用 `(BLOCK_N,BLOCK_K)` 布局而非转置标志）→ §9 R-3 + §1.6.3 备选 A2（M=16 补零）/备选 B 兜底。**注**：本 Agent 已确认 `T.reduce_sum` doc §2.2.2 允许 `(M,N)→(M,1)` 同 rank、归约维 dst=1，故备选 B 的 `reduce_sum(dim=0)`（`(BK,d_padded)→(1,d_padded)`）形态合法；`clear` 参数在 doc §2.1 documented。

### 2. 内存层级规划 — pass

- §4.4 给出完整搬运路径：Cube `GM[W/e]→L1[w_sh/e_sh]→L0A/L0B→L0C[c_frag]→UB`；Vector `GM→UB→(vcast/向量链)→UB→GM`。无 GM→L0 直连等越级路径。
- **UB 预算（人工复算）**：§4.5 W3 ≈59KB，×1.7 ≈ **101KB < 192KB ✓**；每 buffer 标注层级（UB/L1/L0C，§4.3）。v1 已将 f32 buffer 计数统一为 **17**（§3.3 伪代码与 §4.5 表同源列出 `k_f/v_f/h_f/h_n/k_n/k2/v2/hk/vhat/vn/c_out/sig/tmp/win_f/cwf/wh_f/rrms`）。
- **L0C 容量**：`c_frag (d_padded,1) f32` = 512×1×4 = **2KB ≪ 128KB ✓**（机械 `l0c_budget: fail` 为本报告「机械复核」段所述解析器假阳性，非设计缺陷）。
- **L1 容量**：`w_sh (BK,d_padded) fp16` = 128KB × `num_stages=2` = 256KB < 512KB ✓（`e_sh` 256B 可忽略）。
- **Buffer 层级标注**：§4.3 逐 buffer 标注 UB/L1/L0C ✓。

### 3. Tiling 策略 — pass

- **Block 划分**：§5.2 `grid=batch`、`BK=128`、`d` 不划分，附理由（batch 独立；`BK|d_mem`；L1 256KB<512KB；分形 K≥32；`d_padded≤512` 单块驻留）。
- **对齐约束**：`(d_padded,)` f32 = 2048B（32B 整数倍 ✓）；`d_padded=d`（`d%256==0`）→ 尾 lane 浪费 0%（§5.3）。
- **非整除处理**：§5.4 给出 `d`/`d_mem`/`max_conv_len` 三轴的非整除策略与 `tail=T.min(BK,d_mem-k0·BK)` 守卫；manifest 全 workload 无尾块。
- **分核策略三要素（`core-split-strategy.md` §1/§2.2 核对）— 齐全且自洽**：① 逻辑核数 `num_logical_kernels=ceildiv(B,1)=B`（§5.5，表列 1/1/4/8）；② 物理核数**实查**：`NPUUtils.get().get_aicore_num()`=20（混合算子取 AI Core 返回值，非纯 Vector 的 ×2）、`get_aivector_core_num()`=40、`get_arch()="Ascend910B3"`，记录了查询代码与返回值；③ 规模判定 `B ∈ {1,4,8} ≤ 20` → **无需适配** + 依据 + 利用率显式记录（B≤8 仅部分核工作属 decode 固有）。核数与 block 取值自洽，无核内动态串行边界。

### 4. 技术约束检测 — pass

- 三维 Kernel：本算子一维 `T.Kernel(batch, is_npu=True)`，不涉及（§3.5.1）。
- GEMM 非整除：已在 §5.4 处理（`BK|d_mem`，保留 tail 守卫）。
- L0C 溢出：已在 §2 处理（2KB ≪ 128KB）。
- §3.5.1 逐条核对本项目已知限制（`threads` 移除、`T.Parallel`→v-prefix、`TRAP-C12` 显式 vcast、`TRAP-C13` tap host 展开、`TRAP-conditional-alloc` 顶层无条件 alloc、bf16 中转、`TRAP-DEVMODE-PERSIST-GEMM` 规避——非 persistent），与设计一致。

### 5. 循环与同步 — pass（附 1 处非阻塞说明）

- §6.1 明确循环类型：核级 `T.Kernel(batch)`、Cube K 分块 `T.Pipelined(...,num_stages=2)`、conv tap **编译期展开**（host 层 Python/`@T.macro` 链，规避 `TRAP-C13`）、元素级向量化（无逐元素标量循环）、状态搬运 `T.copy`（二维区域）。§6.1 显式说明**不使用 `T.Persistent`**（规避 `TRAP-DEVMODE-PERSIST-GEMM`）。
- §7 同步策略与 Developer 模式一致（自动同步 / CV auto-split），无手动 set_flag/wait_flag；§7.2 逐点位列出同步来源。
- §6.4 尾块处理与 §5.4 一致。
- **非阻塞说明**：§3.3 伪代码存在压缩表达（如 L648 `T.copy(rms_w_h, wh_f_as_dtype)`、`cw_dtype`、`x2_2d[0:1,:]` 视图），已在 §3.3 注 ①③④ 标注为「压缩表达，实现时以实际 buffer 形状为准」——属设计文档压缩惯例，不影响 Stage 3 落地。

### 6. 验证方案 — pass

- §8.1 含 PyTorch 参考实现草案（golden），移植源仓 `engram_decode_step_torch`，独立于 NPU 流水线。
- §8.2 按 dtype 给出精度标准：fp16 `atol=5e-2`/bf16 `atol=1e-1`，`rtol=5e-2`（来源 `test_engram_decode.py` L117–L118，本 Agent 核对一致）。
- §8.3 L0 门槛测试计划 10 用例（smoke/W2/W3/W4 fp16·bf16/多步递推/空态/门控角点/膨胀位置/bf16 全链/状态一致性），覆盖 manifest 全 workload + 边界；含执行方式（`torch.testing.assert_close` 对 §8.1 golden）。

### 7. 完整性与一致性 — pass（附 1 处非阻塞计数残留）

- 章节齐全：迁移类含 §0 且 0.1–0.7 小节完整，§1–§10 齐备。
- 无占位符（占位符类标记全部为空，grep 未命中任何未完成标记）。
- 同类引用真实：本 Agent Glob 确认 §3.5.3/§0.7 所列 `examples/engram/engram_decode.py`、`examples/gemv/example_gemv.py`、`examples/flash_attention/flash_attn_npuir_dev.py`、`examples/norm/rms_norm.py`、`.../ada_layer_norm_kernel/_ada_layer_norm_kernel_DESIGN.md` 均存在；`examples/TileOPs/tests/ops/test_engram_decode.py`、`benchmarks/ops/bench_engram_decode.py`、`manifest/sequence_modeling.yaml`、脚手架 `_engram_decode_kernels.py` 均存在。
- 内部一致：API 映射（§3.2）、内存规划（§4.3/§4.5）、Tiling（§5）三处描述相互一致，v1 已统一 §4.5/§3.3/§5.3/§9 R-8 的 f32 buffer 计数为 17。
- **非阻塞计数残留（不影响任何结论，记录备查）**：§1.6.3 主选 A 行（L468）仍写「UB ~10×(d,)f32 ≈ 20KB」、§1.6.0 R3 #1 行（L348）仍写「UB `(d_padded,)`×~10 f32」，与 §4.5/§3.3 的 **17** 不一致（v1 修复漏同步此两处摘要单元）。该残留**不改变容量结论**（17×512×4=34KB → §4.5 合计 59KB → ×1.7≈101KB < 192KB ✓），属 §4.5 内部同量摘要口径未闭环，建议 Stage 4/后续修订顺手同步（不构成本轮阻塞或意见项）。

### 8. 算法优化分析 — pass

> 本维度以 §1.6 为对象逐项核对，交叉核对 §3.1/§6 与 §1.6 双向一致性；等价性、复杂度、负向断言、弃选前提均**独立推演/复算/开文档核对**，不照抄设计自述。

- **§1.6 章节存在性 — pass**：含 1.6.0 调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代分析 / 1.6.3 向量化轴与数据布局决策四子节，齐备。
- **算法调研存在性与完整性 — pass**：R1 候选表 10 行**含基线（#0 源算法）**，逐候选等价性初判 + 收益方向；R2 在线算法有**明确结论**（「无在线化收益」+ 结构依据：4 reduce 作用于 h/k/v_hat 三个不同向量、单遍 `(1,d)→(1,1)`、每输入只读一次、状态递推本即在线形式）；R3 **四口径齐全**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值，基线在表，口径说明写清含什么不含什么）；R4 逐候选评估（单元匹配/容量/对齐整除/静态边界/流水融合/跨核）；「无更优替代」写明**调研范围**（候选库 + examples + pattern-library + 外部算法族 + docs 枚举）。
- **调研结论正确性（独立复核）— pass**：
  - ① **等价性**：选定算法（Cube GEMV + fp32 中转 + rsqrt 绕法 + 常数折叠 + CSE）与原语义等价——GEMV 仅累加分组变化（fp32 同域，diff ≤3.1e-5）；`sqrt(s)/s ≡ s^{-1/2}`（s>0）实数恒等（本 Agent 独立推演成立，`s=mean(x²)+eps ≥ 1e-6`）；常数倒数折叠 d∈{256,512} 为 2 的幂，`fl(1/d)` 精确（O2 逐位一致）；CSE 物化逐位恒等（O3 max_abs=0）。
  - ② **复杂度算术独立复算**：W3 FLOPs = `4·4·1024·512 + 4(16·512+2·4·512)+20·4` = 8,388,608 + 49,152 + 80 = **8.4378M ≈ 8.44M**（设计 8.44M ✓）；访存 = `(4·1024+4·512+2·4·20·512+2·1024·512+2·512+4·512+4·512)·2` = 1,141,760×2 = **2,283,520B ≈ 2.28MB**（设计 2.28MB ✓）；实现偏差 `grid=B` 每 batch 重读 W → W3 实际 ≈8.3MB（设计 ≈8.3MB ✓）；扫描遍数 1、中间缓冲峰值已列（**未漏计中间缓冲 GM 往返**：中间量全 UB 驻留，唯 conv_state 读/写属契约）。
  - ③ **亲和性判断**：与 `hivm.hir.mmadL1`（Cube）、L1/L0C 容量、`TRAP-threads-kwarg-noop` 不矛盾；容量/带宽淘汰引用 constants.md 条目（`CONST-vector-launch-overhead`、`CONST-L1-port-bw`、`CONST-capacity-910B2C`）。
  - ④ **负向断言主动复核**：R2「无在线变体」对照 `algorithm-candidates.md` 命中行（ALG-rmsnorm「平方和单遍天然在线」、ALG-reduction「分块归约+合并」仅大 N 跨块需要、ALG-layernorm）**复核成立**（d≤512 单 UB 驻留、无多遍基线、三向量不同数据）；R1「无更优替代」对照 ALG-gemv（GEMM 分形 vs Vector 规约 / split-K）、ALG-conv（im2col/implicit GEMM/Winograd——本算子为 **depthwise 膨胀卷积，w=4，无共享权重矩阵**，GEMM 化/Winograd 均不适用，弃选依据为结构而非先验）、ALG-transpose——**未见明显漏掉的已知候选**。
  - ⑤ **并列候选裁决路径**：主选 A vs 备选 B（判定裕度依赖 910B3 未实证常数）**未纸面拍板**，指向 §1.6.3 实验裁决（见下）。
- **调研→设计一致性 — pass**：§1.4（Cube GEMV / fp32 中转 / `vsqrt+vdiv` / CSE 物化）、§3.1（以 §1.6.1 优化后公式为输入：`s=sumsq·fl(1/d)+eps`、`rrms=t/s`、`α=sigmoid(dot·fl(1/√d))`）、§6（无逐元素标量循环，仅 K 分块流水 + 编译期 tap 展开）与 §1.6.0 选定算法族**同口径一致**；迁移任务 §0.5/§0.6 处置与调研选族一致，源优化意图**逐项承接**（§0.6 承接说明：源 #1→保留、#2→R3、#4→R2 强化、#5→舍弃 `threads`+保留 `lru_cache`、#8→R4、#10→R1、#11→R5、#12/#13→保留），**无静默丢弃**。
- **数学等价优化完整性 — pass**：§1.6.1 采纳 5 项（O1–O5），每项含「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；未采纳项（E1 备选、E2 归约合并、E3 稳定化）逐项有弃选理由。
- **等价性论证正确性（独立推演）— pass**：O1 实数恒等（本 Agent 独立推演 `sqrt(s)/s = s^{-1/2}` 成立，且精度优于设备 `vrsqrt` 近似 2.9e-3→1.07e-7）；O2 `x·fl(1/d)`≡`x/d`（d 为 2 的幂，逐位）、`dot·fl(1/√d)` 相对容差内（违反 0）；O3 逐位恒等；O4 与源 fp32 同域；O5 仅分组变化。**与 §8.2 精度标准一致**（余量充裕）。
- **向量化替代完整性 — pass**：§1.6.2 覆盖 10 个循环/标量点（4 处行内归约 / 逐元素链 / 状态左移 / GEMV / 标量 alpha 广播 / 标量门控 / Cube K 分块 / conv tap 展开 / 核级任务映射 / host pad-crop），与 §3.3 伪代码、§6 循环结构**交叉核对无遗漏**：kernel 内**无逐元素标量循环**，能向量替代的（#1–#6）均已替代（API 有 docs 佐证 + examples 先例，优先 v-prefix）。
- **不可替代理由充分性 — pass**：#7（Cube K 分块计算流水，非逐元素）、#8（conv tap 编译期展开，`TRAP-C13` 运行时 Var 索引致动态 slice 非法，w≤4）、#9（核级任务映射，`threads` 在 npuir 无效）、#10（host 元数据 pad/crop，`ascend-constraints.md` §4）均给出**具体到本算子**的理由，无「实现简单」类空泛理由。
- **替代 API 可行性 — pass**：替代方案 API 均有 `docs/`/`examples/` 佐证（`T.gemm`、`T.reduce_sum`、`vcast/vmul/vadd/vsqrt/vdiv/vsigmoid/vbrc`），与维度 1 核对口径一致。
- **向量化轴与布局决策完整性 — pass（v1 阻塞项 1 已闭合）**：§1.6.3 候选矩阵 A/B/C/D/E 齐全，I/O layout 作为契约（尾轴 d 连续）、核内布局为设计变量。**v1 已删除候选 C 的错误 dtype 负向前提**——本 Agent 亲自 Read `docs/Tilelang.language/创建操作/T.transpose.md` §2.2.1 Ascend 行**逐格核对**：`int8/int16/int32/int64/fp16/fp32/bf16 = √；uint8/uint16/uint32/bool = ×`（与设计 v1 表述一致；**关键更正 bf16=√ 成立**，v0 的「bf16 ×」确与文档矛盾）；§2.3 限制为「仅二轴交换」（>2 轴须分解为相邻轴交换链，与 §3.1 一致）。候选 C 弃选理由改为**结构/收益**（主选 A 由 load 侧 `a_transpose` 零 repack；核内 `T.transpose` 引入 UB 级二轴交换链 repack，W 规模 1024×512 需多 tile 链，`PL-1.1` 实测 ~5.4µs/链），与 `ALG-transpose`（VP-2026-0168 更正条目）**口径一致**，**不再依赖任何 dtype 负向断言**。候选 D 引 `PL-1.3-host-permute`（106–147µs，status=verified）；候选 E 为结构分析（水平归约优于垂直扫描）。
- **备选 B UB 预算复核（v1 阻塞项 2 已闭合）— pass（独立复算）**：@d=512、BK=8，本 Agent 逐项复算 §1.6.3 备选 B 常驻 buffer：`w_h (8,512) dtype` = 8×512×2 = **8KB**；`w_f (8,512) f32` = 8×512×4 = **16KB**；`e_brc (8,512) f32`（原地复用） = **16KB**；`acc_k/acc_v (1,512) f32` = 2×512×4 = **4KB**；共享向量链（17×(512,)f32=34KB + ~3×(512,)dtype=3KB + `st_ub` 19×512×2=19.5KB + 归约 fragment ≈2KB）≈ **58.5KB**。raw 合计 ≈ **103KB**，×1.7 ≈ **175KB < 192KB ✓**（余量 ~17KB；BK=4 回落 ≈141KB）。**结论成立**。设计并附 **Stage 4 UB 编译探针计划**（`tilelang.compile` 读 `ub overflow` 取精确实测需求，引用 VP-2026-0059「手工预算 ≠ 可编译证明」；阈值 > 手工×1.12 触发 BK 回落/生命周期回收）。结构完整达 Stage 4 可实现深度（buffer 形状/UB/循环/分核三要素）。
  - *（非阻塞观察）* 备选 B 的 `acc_k/acc_v` 与「共享向量链 17 buffer」中的 `k_f/v_f` 存在 4KB 重复计（备选 B 中 acc 即 k_f/v_f），属**偏保守的重复计**，不改变「175KB<192KB」结论；§1.6.3 (B) 段「~3×(d,)dtype=3KB」与 §4.5「~4×dtype=4KB」有 1KB 口径差，同样不影响结论。
- **实验裁决模式完整性 — pass**：主选 A vs 备选 B 判定裕度依赖未实证常数（910B3 的 L1 端口带宽/向量发射成本，均绑 910B2C/9362 且 `kb_stale_check` 标 stale）→ 设计给出**三件套**：主选 A（结构见 §3.3/§4/§6）、备选 B（结构完整，附 UB 编译探针计划）、裁决计划（代表 shape = 3 full + smoke；指标 = `msprof op` median-of-20；未知常数反解 = 用 B 实测 op 数与 A 实测时长反推 910B3 per-op 成本/L1 端口带宽；**判定阈值** = B < A×(1−0.10) 才翻转；回写 `perf_feedback.md`）。`a_transpose` vs M=16 补零形式（A1/A2）亦纳入同一裁决。**shape 特化分派可行性**已评估（`lru_cache` 按 shape 特化，支持按 workload 分派 A/B）。
- **布局决策与实现一致 — pass**：§1.6.3 选定向量化轴 = 尾轴 d、水平归约 `reduce_sum(dim=1)`；§3.3 伪代码各 v-op 作用于 `(d_padded,)` buffer；§4.5 按该布局计 UB（59KB）；§6 循环结构无逐元素标量循环——三方一致。备选 B 结构独立记录，未与主选混排。
- **与下游章节一致 — pass**：§3.1 公式拆解以 §1.6.1 优化后公式为输入（`s=sumsq·fl(1/d)+eps`、`rrms=t/s`、`α=sigmoid(dot·fl(1/√d))`）；§6 循环结构与 §1.6.2 结论一致（判定向量化的点未再出现对应逐元素标量循环）。

---

> **说明**：本轮结论为「通过」，依 skill 输出规则不设问题清单章节。上列 §7/§8 中的两处「非阻塞计数残留/重复计」为记录性观察，不构成意见项，供 Stage 4/后续修订顺手同步。
