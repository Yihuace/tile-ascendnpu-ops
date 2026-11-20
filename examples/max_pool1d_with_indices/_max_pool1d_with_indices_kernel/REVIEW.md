# _max_pool1d_with_indices_kernel 设计文档检视报告（revision v1 复审）

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/max_pool1d_with_indices/_max_pool1d_with_indices_kernel/DESIGN.md`（**revision v1**，1699 行）
- 任务类型: 迁移（TileOPs 同构仓迁移，harness 子模式，逐函数）
- 源算子: `/workspace/TileOPs-gpu/tileops/kernels/pool/max_pool1d.py`（函数 `_max_pool1d_with_indices_kernel`，第 212–288 行；同文件第 15–68 行是单输出变体，**不在本次范围**）+ `/workspace/TileOPs-gpu/tileops/ops/pool.py`（`MaxPool1dIndicesFwdOp`（第 658 行起）/ `_MaxPoolFwdOpBase`（第 451 行起）/ `_max_pool_roofline`（第 422 行起））+ `/workspace/TileOPs-gpu/tileops/kernels/pool/common.py`（`pool_output_dim` 第 80 行）
- 上一版备份: `history_version/design_v0.md`（v0，Stage 2 首检「不通过」）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 1（§1.6.2 #9/#10 布尔缓冲命名与 §3.2/§3.3/§4.3 残差不一致，非阻塞）
- 前次阻塞项闭环: **B#1 已闭环**（§4.5 系数 `69+2·eb` + 逐 buffer 表 + `BL` 重导，独立复算通过）；**B#2 已闭环**（逐条件非空窗口保证 + 空窗口契约对齐源 + N3/R3 第 4 等价分支 + §8.3 L0-15 负向控制组，经源码核对成立）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: PASS**（§1.6.1 含采纳项 C1/N2/N3 + 语义锚 E1，故强制重跑）
  - 命令: `/workspace/venv/bin/python examples/max_pool1d_with_indices/_max_pool1d_with_indices_kernel/verify_equiv.py`（exit 0）
  - 本机实测耗时 ≈**84 s**（`real 1m23.8s`，`user 40m32s`——宿主重载，多线程 numpy/torch 争用；仍远低于「几分钟」上限，且脚本确为分支采样、未做 C4/C4b 全枚举）。设计记「≈12–43 s（随机器负载波动）」，同量级、非阻塞。
  - **执行输出与 §1.6.1 内嵌结果表逐项一致**：`E1`/`E1-idx`/`N2N3`/`N2N3-neq`/`N2N3-ord`/`B2` 各 8 configs（checks=18/18/36/6/9/18）、`C1` 3 configs（checks=72/36/72）、`EW` 3 configs（checks=18 each）——**全部 worst|d|=0.0000e+00、viol=0、status=EQUIV_PASS**；汇总行一致：`order guard 39/72`、`C1 span-clamp 90 (config, BL) 结构检查全落界`、`empty-window contract: 27 empty windows; source == design idx (0); torch differs (not an oracle)`、`§0.1 semantics evidence: all state-machine readings confirmed`、`hard-coded guards: all ok`、`specialisation_math_only: C4-math / C4b-math value-only EQUIV_PASS (3 dtype)`、`[EQUIV] ALL PASS (total item checks=1074, failures=0)`。
  - **空真守卫核对（逐组）**：① 元素级比较全部经 `mismatch()`/`mismatch_idx()`，内含 `assert a.shape == b.shape and a.size > 0`（**empty shield 在位**，不存在 0 元素比较被记为 PASS 的行；`out==1,s!=k` 组 8 元素、C1-math 组非空）；② `item_ew` 内含 `assert empties, "EW config has no empty window"`（EW 配置必存在空窗口，否则脚本报错而非静默通过）——3 组共 27 个空窗口实测触发 ✓；③ `EW` 项只做 **source vs design**（`(mi, si)`/`(mv, sv)`）比对，**不拿 torch 作 oracle** ✓；④ 唯一「无元素比较」的行是 `N2N3-ord` 8 行（`candidate=None`，只做顺序守卫），其实质断言是**检出数非空（39/72 > 0）**，已核对；该行 `viol=0` 本身无信息量，本报告未据此判 pass。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/max_pool1d_with_indices/_max_pool1d_with_indices_kernel/DESIGN.md`
  ```json
  {"capacity_constants": {"ub_kb": 192.0, "ub_inflation": 1.7, "l0c_kb": 128.0,
                          "source": "pattern-library/constants.md CONST-capacity-910B2C"},
   "checks": [
     {"check": "ub_budget",   "status": "skip", "detail": "no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"},
     {"check": "l0c_budget",  "status": "skip", "detail": "no block_M x block_N values parsed"},
     {"check": "core_split",  "status": "skip", "detail": "insufficient parsed dims {}; declared logical cores = None"},
     {"check": "r3_metrics",  "status": "pass", "detail": "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"}
   ]}
  ```
  - `r3_metrics` **pass**（四口径 FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值均在 §1.6.0 R3 表，且含基线 #0）。
  - `ub_budget` **skip → 人工复核结论: PASS**（1D「逐 buffer 表 + 闭式不等式」形态，检查器无法解析）。**独立复算（B#1 复核）**：
    - **系数自洽（分量和 = 系数）**：§4.5 闭式第二项系数 `69 + 2·eb` = 单输出基式 `25+2·eb`（acc/flag/zeros/infbuf/nanbuf/out32 六个 f32 = 24 + cfinal bool 1 + acc16/acc_t `2·eb`）**+ 7×i32(28)**（max_idx/nan_idx/cand/nanc/bigv/neg1v/zero_i32）**+ idx32/idx_t(8)** + **idx64(8)** ⇒ `25+2·eb+28+8+8 = 69+2·eb` ✓（**v0 的 `53+2·eb` 漏计 idx32/idx_t 与 idx64，已修正**）。首项系数 `2·eb+9` = row_src(`eb`)+row_dst(`eb`)+row_f32(4)+eq_mask(1)+eq_f32(4) ✓；第三项 `+256` = `fv` `(1,CH)` i32 上界（CH≤64 ⇒ 4·CH≤256，保守上界，方向安全）✓。
    - **逐 buffer 求和 = 闭式值**（逐项独立相加）：`sincnet`（CH=64,BL=16,LSPAN=48,fp16）= `6144+6144+12288+3072+12288+4096+4096+12288+4096+1024+28672+256+2048+2048+4096+4096+8192` = **114 944 B** = 闭式 `64·48·13 + 16·64·73 + 256` = **114 944 B** ✓；`ecg`（CH=64,BL=17,LSPAN=41,bf16）= **113 792 B** = 闭式 `64·41·13 + 17·64·73 + 256` = **113 792 B** ✓。**二者一致，无低报**（v0 的「逐 buffer 略低/闭式更保守」自注已反转为「相等」）。
    - **容量判定**（默认 auto-multi-buffer 档 `INFLATE=1.7 ≤ 196 608 B`）：`sincnet` `114 944×1.7 = 195 405 B = 190.8 KiB ✓`（`BL=17 ⇒ LSPAN=51 ⇒ G=122 112 ⇒ ×1.7=207 590 ✗` ⇒ 取 `BL=16`）；`ecg` `113 792×1.7 = 193 446 B = 188.9 KiB ✓`（`BL=18 ⇒ LSPAN=43 ⇒ G=120 128 ⇒ ×1.7=204 218 ✗` ⇒ 取 `BL=17`）；`textcnn` fp16 (CH64/BL1/LSPAN128) `111 424×1.7=189 421 ✓`、fp32 (CH32) `72 352×1.7=122 998 ✓`；`smoke`、两个 test shape 全部 ✓。**两个 full workload 均在 192 KiB 判据内，B#1 判据错误已消除**。
    - 独立佐证：pattern-library `layout.md` `PL-1.4`（第 114 行）1D **单输出**版式 = `G = CH·LSPAN·(2·eb+9) + BL·CH·(25+2·eb)`，与本设计双输出式 `69+2·eb` 的差额恰为 `8×i32` 索引类（28+8+8=44 B/元素），逐项可对 ✓。
  - `l0c_budget` **skip → 人工复核结论: 不适用**（纯 Vector，无 Cube，`block_M×block_N` 不存在）。
  - `core_split` **skip → 人工复核结论: PASS**（1D 单元形态无 `block_M×block_N`，检查器无 dims）。**独立算术复算**（`num_logical = N × ceil(L_out/BL) × ceil(C/CH)`，`num_kernels = min(num_logical,40)`，`num_local = ceil(num_logical/40)`）：
    | workload | L_out（复算） | CH | BL | num_cblk | num_oblk | num_logical | num_kernels | num_local |
    |---|---|---|---|---|---|---|---|---|
    | `smoke`(1,8,32,k3s2p1) | 16 | 8 | 16 | 1 | 1 | **1** | 1 | 1 |
    | `sincnet`(32,80,4096,k3s3p0) | 1365 | 64 | 16 | 2 | 86 | **5 504** | 40 | **138** |
    | `textcnn` fp16(64,256,128,k128) | 1 | 64 | 1 | 4 | 1 | **256** | 40 | **7** |
    | `textcnn` fp32 | 1 | 32 | 1 | 8 | 1 | **512** | 40 | **13** |
    | `ecg`(16,128,2048,k5s2p2d2ceil) | 1023 | 64 | 17 | 2 | 61 | **1 952** | 40 | **49** |
    | test(1,4,63,k3s2p1) | 32 | 8 | 32 | 1 | 1 | **1** | 1 | 1 |
    | test(1,4,97,k5s3p2ceil) | 33 | 8 | 32 | 1 | 2 | **2** | 2 | 1 |
    逐项与 §5.2/§5.5 表一致 ✓（`ceil(5504/40)=138`、`ceil(256/40)=7`、`ceil(512/40)=13`、`ceil(1952/40)=49`）。§1.6.3 备选档亦复算一致：`sincnet` CH32/BL32/LSPAN96 ⇒ num_cblk=3, num_oblk=43, num_logical=4 128, num_local=104（`ceil(4128/40)=104`）；`ecg` CH32/BL35/LSPAN77 ⇒ num_cblk=4, num_oblk=30, num_logical=1 920, num_local=48 ✓。**B#1 的级联重算完整、无残留旧值**。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

> 检视方式：亲自 Read `max_pool1d.py` 全文（348 行）、`ops/pool.py` 的 `_MaxPoolFwdOpBase` / `MaxPool1dIndicesFwdOp` / `_max_pool_roofline`、`common.py` 的 `pool_output_dim`，并逐项与 §0 核对；`manifest/pool.yaml` 的 `MaxPool1dIndicesFwdOp` 逐字段对照。

**（0-a）语义理解正确性 — pass**
- 索引编码：§0.1「索引 = 沿 L 轴位置 `iw`（int64）」↔ 源码 `flat_idx = T.cast(iw, "int64")`（第 259 行）+ 落盘 `indices[...] = T.if_then_else(has_nan, nan_idx, max_idx)`（第 280–284 行）；`indices` dtype `"int64"`（第 234 行）✓。
- tie → first-hit：§0.1「严格 `>` ⇒ 相等不更新 ⇒ 先访问者胜出」↔ `elif val > max_val: max_val = val; max_idx = flat_idx`（第 270–272 行）✓；§0.1 实测行 `x=[9,1,9,1] k3s1p0 → idx=[0,2]` 已由重跑 `semantics_evidence()` 逐行确认 ✓。
- NaN 语义：§0.1「值 NaN + **最后一个 NaN** 的 `iw`」↔ `if is_nan: nan_idx = flat_idx; has_nan = True`（第 261–265 行，含源注释第 262–263 行 "PyTorch records the last NaN visited in a pooling window"）+ 末尾 `has_nan ? nan_idx : max_idx`（第 280–284 行）✓；实测 `x=[nan,5,nan,2] → idx=[2,2]`（last NaN）✓。
- 访问序：§0.1「`k_w` 递增」↔ `for kw in T.serial(kernel_w)`（第 255 行）+ `iw = ow*stride_w - pad_w + kw*dilation_w`（第 256 行），`d_W>0` ⇒ `iw` 严格递增 ✓（N2/N3 等价改写的前提，本检视独立确认）。
- 边界/越界：§0.1「越界 lane 跳过 ≡ $-\infty$ padding」↔ `if iw >= 0 and iw < l_in`（第 257 行）+ 初值 `T.cast(float("-inf"), accum_dtype)`（第 250 行）✓；`{+inf,-inf}` → `+inf`、`-inf` 为合法最大值 ✓（实测行）。
- dtype 语义：§0.1「`accum_dtype="float"`（第 224 行）；fp32 是**比较域**；末尾单次 `T.cast(result, dtype)`（第 279 行）」✓；`atol=rtol=0` 依据（max 只选择、被选元素原值）成立 ✓。
- 门禁 dtype：§0.1 `{float16, bfloat16, float32}` ↔ `_MaxPool1dKernelBase.__init__`（第 147–150 行）+ `_validate_pool_input_dtypes`（`ops/pool.py` 第 104 行）✓。

**（0-b）I/O 契约 — pass**：§0.2「输入恰好 3D NCL（无 2D C×L 路径）、两路输出同形 `(N,C,L_out)`、`out` 同 dtype、`indices` int64、原生 NCL、`out_idx=[1,2]`」↔ `_resolve_input` 的 `if input.ndim != nd + 2: raise`（`ops/pool.py` 第 507 行；nd=1 ⇒ 恰好 3D）✓、`register_fake` 同形双输出（`(dtype, int64)`，`max_pool1d.py` 第 336–339 行）✓、`@tilelang.jit(out_idx=[1, 2], ...)`（第 228 行）✓。

**（0-c）算法解读完整性 — pass**：§0.3 六步表覆盖源码第 224–284 行的**全部计算语句**（`out_l`/`total_output` → 第 225–226 行；grid → 第 236 行；lane 扁平解码 → 第 237–243 行；5 个 `T.alloc_var` 状态初值 → 第 245–254 行；窗口状态机三路（`is_nan` / `first_valid` / `val > max_val`）→ 第 255–272 行；`if_then_else` 判决 + 双写出 → 第 274–284 行），无遗漏、无臆造。host 侧逻辑（参数规范化/校验、`x.contiguous()`、`_kernel_cache`、`lru_cache`、`_max_pool_roofline(indices=True)`、jit 配置）逐条可核 ✓。数据流图与源码一致（无 SMEM/L1 中转、逐元素 gather、两路输出共用同一次窗口扫描）✓。

**（0-d）优化手段覆盖完整性 — pass（v0 的引文瑕疵已修，A7）**：§0.4 的 14 项覆盖源中可见的全部结构性选择；本检视确认源**未**使用 warp shuffle / SMEM swizzle / mma / 三维 Kernel / cp.async / `T.Pipelined`（通读全文，无对应语句），故 §0.5「舍弃项：无」成立 ✓。**引文已改指本算子行号**：§0.4 #4 ↔ `elif val > max_val: ...`（第 270–272 行）+ `first_valid` seed（第 266–269 行）+ 初值（第 250 行）；§1.6.0 R2 迁移佐证 ↔ 第 270–272 / 261–265 / 266–269 行（v0 曾误引单输出工厂第 57 行，A7 已修，本检视逐行核对成立）✓。

**（0-e）耦合性判定合理性 — pass**：§0.5 逐条四态处置 + 依据（migration-analysis §5.3 映射表 / `docs/Tilelang.language/` 条目 / pattern-library 实测）✓；关键判定与证据一致：`T.Parallel(block_m)` + `threads`（第 236–237 行）判「重新设计」✓（NPU 无 CUDA 线程）；逐元素 GM gather（第 258 行）判「重新设计」✓（NPU 无 L1/L2 吸收）；语义层项（first-hit / last-NaN / int64 / fp32 比较域 / 双输出融合）判「保留」并在 §0.6 给出向量化形态 ✓；int64 中间承载降为 int32 + 落盘 cast（`l_in < 2**31` 守卫）自洽 ✓。无「舍弃」项且理由成立 ✓。

**（0-f）NPU 重设计可行性 — pass**：§0.6 R1–R5 每项含「源方案 → NPU 新算法 → 语义保持论证」三要素 ✓。
- R1：一维 `T.Kernel` + `(n, o_blk, c_block)` 单元 + C 轴向量 lane；满足一维 Kernel 约束 ✓；「沿 L 分块」的 1D 特有理由（`L_in` 达 4096 不可整线驻留）成立 ✓。
- R2：`vbrc(-inf)` 单位元 + C 向量 `vmax` + **块级 `eq` 掩码乘性累积**；本检视独立确认「`eq`（`x==x`）为 IEEE 正确的非 NaN 探针、`ne` 为 ordered-NE 不可用」与 `TRAP-vcmp-ne-ordered-nan` 记载一致（traps-compiler.md）✓，「不依赖 `vmax` NaN 行为」由脚本 `N2N3-neq` 独立证实（NaN-忽略 vs NaN-传播两种 `acc` 输出完全相同）✓ —— 与 pattern-library `PL-1.22` ③/负向依赖逐字一致 ✓。
- R3：索引追踪改写（first-hit = `eq`+`vmin` 取最小 `iw`；last-NaN = `isnan`+`vmax` 取最大 `iw`）。本检视独立推演见维度 8 ✓；int32 位宽守卫合理 ✓。
- R4：连续 span 载入 + 核内二轴转置 + 载入起点钳制；`s0` 三支证明见维度 8 ✓；与 `PL-1.22` ⑥/⑦（1D 无第二轴可外提，对应优化 = 钳制 span 载入）一致 ✓。
- R5：结构特化评估（C4/C4b 不采用）；其 `d_W==1` 必要性论证与 `algorithm-candidates.md` 窗口池化域条件①**逐字同源**（含 1D 反例 `L_in=5,k=2,s=1,p=0,d=4`：窗口 `{0,4}`，整线归约 1.0 vs 正确 0.0）✓。
- §0.6 各「承接」行逐项对应 §0.4 的 14 项优化意图（含 #5/#6/#8/#9），**无静默丢弃** ✓。

**（0-g）§1–§7 与迁移决策一致性 — pass**：§1.4 / §3.1 / §3.3 / §5.2 / §5.5 / §6.1 / §6.2 描述的都是 §0.6 决策后的算法（C 轴向量 + span 载入 + 两遍法 + `nan_idx>=0` 判决 + 空窗口子句 + persistent），无「§0 判重设计、§3 照抄源方案」的脱节 ✓；**S1 一致性独立复核**：全文检索无任何几何特化分支（§3.3/§6.2 伪代码只有一般网格路径；唯一 trace 期分支是 dtype 谓词 `is_f32`），§5.2/§5.5/§9.1 明示单一 `num_logical` 口径、`textcnn-global`（`L_out==1`）走一般路径 `BL=1` ✓。

**（0-h）golden 独立性 — pass**：§8.1 golden 直接调用源仓测试基准 `F.max_pool1d(..., return_indices=True)`（fp32 中转后舍回 dtype，CPU 运行），**不含**核内转置 / C 轴向量化 / span 载入 / 索引累加器 / 空窗口子句实现 / 特化分支 ✓，与 §0.1 语义绑定而非与 §0.6 算法绑定 ✓；索引 int64 独立精确断言（§8.2）✓。

**（0-i）host/op 层引用路径真实性 — pass**：`/workspace/TileOPs-gpu/tileops/ops/pool.py` 存在，且 `_MaxPoolFwdOpBase`（第 451 行）、`MaxPool1dIndicesFwdOp`（第 658 行）、`_max_pool_roofline`（第 422 行）均在该**共享模块**内 —— 设计正确指向共享模块（未误引不存在的单算子文件）✓；`common.py` 的 `pool_output_dim`（第 80 行）存在 ✓；§3.5.3 引用的 6 条 `examples/` 路径逐条 `ls` 均存在 ✓。

**（0-j）多输出/变体函数的迁移范围 — pass**：头部与 §0 明示范围 = 双输出变体 `_max_pool1d_with_indices_kernel`（第 212–288 行），单输出兄弟（第 15–68 行）「已由同族前作迁移，不在本次范围」✓；与 manifest `MaxPool1dIndicesFwdOp`（`variant_of: MaxPool1dFwdOp`、`ref_api: torch.nn.functional.max_pool1d`）声明一致 ✓，无范围蔓延；manifest `shape_rules`（`_kW>0`、`_sW>0`、`_dW>0`、`0<=_pW<=_kW//2`）与 §0.1 引用逐字一致 ✓。

**（0-k）边界域不变量（B#2 闭环核对）— pass**
- **逐条件非空窗口保证（v1 修正）**：§0.1 现表述为「`d_W == 1 ∧ 0 ≤ p_W ≤ ⌊k_W/2⌋ ∧ s_W > 0` ⇒ 每个输出窗口至少含 1 个 in-bounds 元素；**`d_W ≥ 2` 时该保证不成立**」。本检视独立验证：`d_W==1` 时窗口 lane 集合 = `[ow·s−p, ow·s−p+k−1]` 为**连续区间**，首窗 `k−1−p ≥ 0`（因 `p ≤ ⌊k/2⌋ ≤ k−1`）且末窗 `minlane ≤ L_in−1`（非 ceil 由 `L_out` 定义给出，ceil 由 helper 末窗回退修正），故每窗交集 `[0,L_in)` 非空；`d_W≥2` 时集合退化为等差子集 `{o·s−p+k·d}`，最小反例 `L_in=1,k=2,s=1,p=1,d=2 → lane {−1,1}` 全落界外 ⇒ `d_W==1` 不可或缺 ✓（与 `algorithm-candidates.md` 窗口池化域条件①「`d==1` 必不可少」同源）。
- **空窗口契约（v1 显式定义，对齐源）**：源在空窗口输出 `(val, idx) = (−∞, 0)` —— **逐行核对**：`max_val = T.cast(float("-inf"), accum_dtype)`（第 250 行，未被消费）、`max_idx = T.cast(0, "int64")`（第 252 行，保持初值）、`if iw >= 0 and iw < l_in`（第 257 行，窗口无 lane 进入）、`result = has_nan ? NaN : max_val` 与 `indices = has_nan ? nan_idx : max_idx`（第 274–284 行，`has_nan=False` ⇒ `(−inf, 0)`）✓。设计对齐源：`max_idx==BIG ∧ ¬has_nan ⇒ idx=0`（复用 `cfinal`，不新增 UB）⇒ 空窗口 `(val,idx)=(−inf,0)`，与源**同值** ✓。本检视确认该契约使「与源等价」在**全部 manifest-legal 配置**下为真（含 `d_W≥2 ∧ p_W>0` 的空窗口 regime）；设计同时正确标注该 regime torch 返回越界索引（CPU kernel loop-artifact）**不可作 oracle** ✓（`verify_equiv.py` 的 `empty_window_evidence()` 实测 source/design idx=0、torch idx∈{1,2}）。
- **契约可见性**：§0.1（定义）↔ §3.3 伪代码（第 1076–1077 行 `vcmp(idx32,bigv,"eq")` + `vselect(cfinal,zero_i32,idx32,idx32)` + 实现级约束⑥）↔ §6.2（第 1478–1479 行）↔ §9.3（特殊场景）；三处一致 ✓。
- **N3/R3 等价前提补全**：§1.6.1 N3 现列 4 支前提（① `iw` 单调增；② `acc` 定稿＝两遍法；③ **非空窗口** ⇒ `min{iw:x[iw]==acc}` 恒有解；④ **空窗口分支** ⇒ `idx=0`），支 ③④ 互斥且覆盖全部窗口 ✓；§0.6 R3 step 4b 同步 ✓。
- **§8.3 负向控制组 L0-15**：`(1,2,1)` k2s1p1 d2（lane `{−1,1}`）/ `(1,2,1)` k2s3p1 d3 ceil（lane `{−1,2}`），断言 = **空窗口契约**（`(−inf,0)`），**不断言与 torch 相等** ✓；本检视用 `EW-min`/`EW-oob`/`EW-l2` 三配置实测确认 `empty_ow` 非空、source==design idx=0 ✓。

**维度 0 结论**：语义解读（索引编码 / tie 方向 / NaN 语义 / `first_valid`+严格 `>`+NaN 覆写状态机 / 越界跳过）、算法步骤覆盖、优化手段识别与引文、耦合性判定、NPU 重设计可行性、§1–§7 一致性、golden 独立性、I/O 契约与迁移范围**全部通过**；前次唯一 fail 项（§0.1 边界域不变量）已改写为可成立的逐条件表述并配显式空窗口契约，**B#2 闭环**。

### 1. API 可行性 — pass

- §3.2/§3.4 的每条 API 均亲自打开 `docs/Tilelang.language/` 条目核对 dtype/形态，全部成立：
  - `T.vmax`（`比较操作/T.vmax.md` §2.2.1 = `int16√ int32√ int64√ fp16√ fp32√ bf16×`）、`T.vmin`（`T.vmin.md` §2.2.1 同表，另含标量广播形态 §2.2.2）—— 与设计「bf16 须先 `vcast` 上转」一致 ✓。
  - `T.vcmp`（`T.vcmp.md` §2.2.1 = `int32√ fp32√ bf16√`；`cmp_mod ∈ {eq,ne,lt,gt,ge,le}`（第 20/36 行）—— 设计仅用 `eq`/`ge` ✓；输出 **bool 片上**、「bool 不可写回 GM（i1 无法 load/store，编译失败）」（第 38–39 行）—— 与设计「`eq_mask`/`cfinal` 仅片上中间结果」一致 ✓）。
  - `T.vselect`（`条件操作/T.vselect.md` §2.2.1 = `int32√ int64√ fp16√ fp32√ bf16√`；四操作数同形 §2.2.2）✓。
  - `T.transpose`（`创建操作/T.transpose.md` §2.3 = 仅二轴交换、禁 >2 轴、`int32√ fp16√ bf16√ fp32√`）—— 设计用 `[1,0]` 二轴交换 ✓。
  - `T.vcast`（`数据类型转换操作/T.vcast.md` 第 50 行 `i32→i64: rint`；第 36 行 `f16→f32: rint`；第 42 行 `bf16→f32: rint`）—— 设计用 `rint` ✓；同 dtype `f32→f32` 在表内但退化为 round-to-integer（`TRAP-vcast-same-dtype-rounds`）⇒ 设计按 dtype 分派 `T.copy` ✓。
  - `T.vmul`（`数学操作/T.vmul.md` §2.2.1 = `int32√ fp16√ fp32√ bf16×`；§2.3 支持原地 = dst 可与 src1 同一 buffer）✓ —— 设计只用 f32 原地。
  - `T.vsub`（`T.vsub.md` §2.2.1 = fp32√、`bf16×`；§2.3 原地 ✓）✓。
  - `T.vbrc`（`shape操作/T.vbrc.md` §2.2.2 标量→任意 shape 广播 ✓）—— 与设计/§9.2「dst 须整 buffer、标量 operand 须 typed PrimExpr」（`TRAP-npuir-vbrc-region-dropped`）一致 ✓。
  - `T.clear`（`内存操作/T.clear.md` §2.2 = fp32√、1~5 维）✓；`T.copy`（`内存操作/T.copy.md`）✓；`T.infinity("float32")`（`创建操作/T.infinity.md`）✓；`T.Kernel(num, is_npu=True)`（`编译器提示操作/T.Kernel.md` §2.3，≤2 维）✓；`T.alloc_ub` ✓；`T.min`/`T.max`（前端标量）✓。
- v-prefix 优先 ✓（计算链全为 `T.v*` + `T.copy`/`T.transpose`；无 `npuir_xxx` 形态）✓。
- 模式匹配：Developer 模式用自动同步、无手动 `set_flag`/`wait_flag` ✓（与 §7 一致）。
- **v0 的措辞不一致已修（A10）**：§2.2 现列 `T.vmax`/`T.vmin`/`T.vcmp`/`T.vselect`/`T.vmul` + 广播/清屏/重排，并明确「**不使用 `T.reduce_max`**」（与 §3.4/§9.2 一致）✓。

### 2. 内存层级规划 — pass

- 搬运路径：GM→UB 单级（`T.copy` span）/ UB 内转置（`T.transpose`）/ UB 内计算 / UB→GM 双输出写回；**无 GM→L0 越级、无 Cube/L1/L0C 路径** ✓（§4.4 与实际算法一致，§1.5 数据流图同步）。
- Buffer 层级标注：§4.3 逐 buffer 标注 UB ✓。
- L0C：不适用（纯 Vector）✓。
- **UB 预算（B#1 闭环）— pass**：判据系数修正为 **`69+2·eb`**，与同节分量注释求和一致、与逐 buffer 表逐项和一致；`sincnet` `114 944 B ×1.7 = 190.8 KiB`、`ecg` `113 792 B ×1.7 = 188.9 KiB`，**均 ≤ 192 KiB**（独立复算见「机械复核」段）；被否档 `BL=17/18` 与 v0 档 `BL=18/20` 的超容量判定亦复算一致 ✓。§4.5 自注已改为「逐 buffer 求和 = 闭式值」（v0 的反向自注已修，A9）✓。

### 3. Tiling 策略 — pass

- Block 划分：§5.2 给出 `CH`（C 轴 8 的倍数、容量判据、与 §5.5 配合）与 `BL`（1D 必需沿 L 分块、halo 摊薄、受容量约束）的具体值与理由 ✓；`LSPAN = min((BL−1)·s_W+(k_W−1)·d_W+1, L_in)` 与 §4.5 定义一致 ✓（**v0 的 `LSPAN=96` 笔误已修为 97，A8**）。
- 对齐约束：向量轴 C 恒为 8 的整数倍（8 亦为 fp32 向量宽 4 的倍数）；`L_in`/`LSPAN` 仅作转置后 row 长度、不作向量轴，故不要求整除 —— 论证成立 ✓。
- 非整除处理：`real_c` / `BL_real` 动态范围 + `uid` 守卫 + `iw` 守卫四类尾块 ✓（§5.3/§5.4/§6.4）。
- 分核策略三要素（对照 `.agents/skills/_shared/standards/core-split-strategy.md` §1/§2.2）：① `num_logical = N×ceil(L_out/BL)×ceil(C/CH)` 单一口径 ✓；② 物理核数 `NPUUtils.get().get_aicore_num()` 实查 = **20** → 纯 Vector ×2 = **40**（含查询代码与返回值；明确**不采用** pattern-library 910B2C 的 24，设备为 910B3 —— 与 `CONST-aicore-910B2C` 的设备绑定纪律一致）✓；③ 逐 workload 三选一判定 + 极大规模 persistent（`num_kernels=min(num_logical,40)`、`num_local=ceil(...)` 静态界、`uid = t*num_kernels + cid`）+「不做空核」保证 ✓。核数与 block 自洽（**算术复核见机械复核段**，含 B#1 级联后的全部新值）✓。
- **B#1 级联重算检查**：`BL` 变更后 `LSPAN→num_oblk→num_logical→num_local` 全部同步（sincnet 5504/138、ecg 1952/49），§1.6.3 备选档亦重算 ✓，无「只改一处」的残留。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（一维 `T.Kernel(num_kernels, is_npu=True)`，任务维度核内解码）✓。
- GEMM 非整除 / L0C 溢出：不适用（无 Cube）✓。
- 动态 `T.serial` 边界：`k_W`/`BL`/`LSPAN`/`num_local` 均编译期常量 ✓。
- host 侧不改输入内容：仅 shape/stride 元数据与缓存 + `.contiguous()`，与 `ascend-constraints.md` §4 一致 ✓。

### 5. 循环与同步 — pass

- §6.1 逐维度给出循环类型与 API：`T.Kernel`（block 级）/ 初始化用整 buffer `T.vbrc`/`T.clear`（无循环）/ `T.serial(num_local)`（persistent，静态界 + `uid` 守卫）/ `T.serial(k_W)` + `T.serial(BL)`（tile 级顺序依赖，循环体为 C 向量 op）/ 元素级全部向量化 ✓；§6.1 明示「不出现逐元素标量循环」✓。
- 同步与模式匹配：Developer 自动同步，无手动 `set_flag`/`wait_flag`；`T.copy`(MTE2) 与 v-op(Vector)、`T.transpose` 的同步由编译器编排 ✓（§7.2/§7.3，与 §2 模式一致）。
- 尾块处理：§6.4 覆盖任务/通道/输出块/窗口/空窗口五类 ✓。
- 流水线：§6.3 说明不入 `T.Pipelined` 的理由（单阶段流、无多阶段 K 循环），并列为 Stage 4 候选 ✓。

### 6. 验证方案 — pass

- §8.1 含可直接移植的 PyTorch golden 草案（`F.max_pool1d(..., return_indices=True)`，fp32 中转后舍回 dtype，CPU 运行、独立于 NPU 算法、无特化分支）✓。
- §8.2 精度标准按 dtype 给出：值 `atol=rtol=0`、索引 int64 `torch.equal` 精确 —— 依据（max 是纯选择、无累加舍入；索引整数契约）成立，优于模板通用档 ✓；并明确「索引是免费 oracle，必须独立断言」✓。
- §8.3 L0 门槛计划含具体 shape/dtype 与覆盖点（manifest 四 workload + `stride=None` + `ceil` + tie/NaN/±inf 三条门禁 + 非连续 + 尾块 + 前 padding + 参数校验 + **L0-15 空窗口负向控制组** + L0-16 可选 subnormal）✓，dtype 覆盖 `fp16/bf16/fp32` ✓。
- **B#2 相关**：L0-15 断言 = 空窗口契约（`(−inf,0)`）而非 torch 相等，理由（该 regime torch 索引越界不可作 oracle）成立 ✓ —— 对上一条 stage-2 建议（问题 2 建议 4）的落实到位。

### 7. 完整性与一致性 — warn（非阻塞）

- 章节齐全：§0.1–0.7 齐全、§1–§10 齐全（迁移模板 11 章节）✓。
- 无占位符：全文逐项核对未发现未填充的模板占位标记（无未填花括号槽位、无待办标记、无空章节）✓。
- 同类引用真实：§3.5.3 的 6 条路径逐条 `ls` 存在 ✓；pattern-library 引用条目（`PL-1.1/1.2/1.3/1.4/1.22`、两条 TRAP）逐条存在且内容与设计引用一致（`layout.md` 第 5/27/94/114/51 行起、`traps-compiler.md` 第 349/373 行起）✓；toolchain stamp 与 `traps-compiler.md` 第 357/381 行的 `ba60e99c…+CANN 9.0.0+910B3/2026-10-14` 完全一致 ✓。
- §4.5 闭式 ↔ 逐 buffer 表 ↔ 自注三者现已互相印证（v0 的不一致已修，A1/A9）✓；§5.2 表值（含 `LSPAN=97`）与定义一致 ✓。
- **warn 原因（建议级，非阻塞）**：§1.6.2 第 9/10 行的布尔缓冲命名/判据来源与 §3.2/§3.3/§4.3 存在残差不一致——
  - §1.6.2 #9 写 `cfinal = T.vcmp(flag, zeros, "eq")`（基于 `flag`）；#10 写 `cfinal2 = T.vcmp(nan_idx, zero_i32, "ge")`（基于 `nan_idx`）并「复用 `cfinal`」做空窗口子句 ⇒ 读作**两个** `(BL,CH)` bool 缓冲；
  - 而 §3.2（11a）、§3.3（第 1073–1077 行）、§6.2（第 1476–1479 行）、§4.3/§4.5 均为**单一** `cfinal`（=`nan_idx>=0`，值/索引/空窗口三处复用）。
  - 两种 `has_nan` 判据**数学等价**（`flag==0 ⟺ ∃NaN ⟺ nan_idx>=0`），且 §3.3 已声明并冻结（§9.2 实现级约束⑥「空窗口子句须排在 out32 之后；`cfinal` 复用次序在 §6.2 冻结」），故**不影响冻结路径的正确性与 UB 判据**（§4.5 按单缓冲计）。残留风险仅在「Stage 3 若照 §1.6.2 字面实现会多分配一个 `(BL,CH)` bool ⇒ 1 024 B，sincnet `×1.7` 至 ≈197 146 B 微超判据」——见「非阻塞改进建议」。
  - 同类：§3.3 的 `eq2b`/`eqb`（每 `(kw,owl)` 一次的 `(1,CH)` bool 临时）未列入 §4.3/§4.5（合计 ≤128 B，量级可忽略，且 §3.3 已声明「Stage 3 可细化局部 buffer 复用」）。
- 判定：该残差限于 §1.6.2 分析表的表述层，不构成阻塞（冻结决策唯一、UB 判据自洽、正确性不受影响）。

### 8. 算法优化分析 — pass

**（8-a）§1.6 章节存在性 — pass**：§1.6.0 / 1.6.1 / 1.6.2 / 1.6.3 四个子节齐全且均有实质分析 ✓。

**（8-b）算法调研存在性与完整性（四问）— pass（独立复核）**
- R1：候选表含**基线**（#0 源算法、#0′ 基线向量化 = 选定）+ 8 个候选；命中行 `algorithm-candidates.md` 的 `ALG-pooling`（第 53–59 行，family=[pooling]）**逐条评估**：R1 命中项（常数权重卷积 / sum-pool×常数）判「数学不等价」、R2「滑窗天然流式」判「基线与选定候选本身已是单遍」、R3「窗口重叠数据的重复读」判「由 span 载入削减承接」、R4「C 轴向量轴」判「选定」——本检视对照参考表确认无命中行漏评 ✓。
- R2：**明确结论「有」**，并给出结构判据（running 最大 + running NaN 标记 + running 索引，窗口内可增量更新、无跨元素依赖）与迁移佐证（源第 270–272 / 261–265 / 266–269 行的 running 更新 = 存在在线变体的直接证据）✓；收益口径（扫描遍数 = 1；索引所需「两遍」在 UB 驻留同一 span 上、非 GM 二读）自洽 ✓。
- R3：四口径表（FLOPs 比较次数 / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）**含基线 #0**，口径写清（含「不含 UB 内搬运、中间缓冲不落 GM、索引 int64 = 8 B/元素」）✓。
- R4：逐候选六维评估（计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水融合 / 结论），负向淘汰附 `docs/`、pattern-library、trap 条目佐证或标注「未文档化假设」✓。
- 调研结论 + 调研范围声明（四类覆盖：参考表命中行 / 恒等变形空间 / pattern-library 同类实现与陷阱 / 本仓同类案例）+「互联网检索未使用」声明 ✓；下游约束（§1.4/§1.6.1–§1.6.3 以 #0′ 为分析对象；§0.5/§0.6 处置与之一致；§0.4 的 14 项无静默丢弃）✓。

**（8-c）调研结论正确性（独立复核）— pass**
- ① **等价性独立推演**（本检视亲自推演，不采信自述）：
  - **N3（first-hit → 集合极值）**：源在 `iw` **严格递增**序上以「首个非 NaN 元素 seed（`first_valid`）+ 严格 `>` 更新」维护 `max_idx`；设窗口最终最大值 `m`，则 `max_idx` 只在运行最大值被严格刷新处写入，最后一次刷新 = `m` 的**首次**出现（此后等于 `m` 的 lane 不满足 `>`）⇒ 源 `max_idx = min{iw : x[iw]=m}` = 设计的 `vmin` 形态 ✓。全 `-inf` 窗口：`first_valid` seed = 首个有效 `iw`，其后无 `>` 更新 ⇒ 与 `min{iw : x[iw]=−inf}`（= 首个有效元素，源实测 P1 `idx=[0,0,1,2]`）一致 ✓（`first_valid` 分支被消解，§0.6 R3 论证成立）。**v1 补全的第 4 支（空窗口）**：无 in-bounds lane ⇒ `max_idx=BIG ∧ ¬has_nan` ⇒ `idx=0`，与源初值同值 ✓（见 0-k）。
  - **N2（last-NaN → 集合极值）**：`nan_idx` 在每个 NaN 处**无条件覆写**，`iw` 随 `kw` 单调增 ⇒ 末次覆写 = NaN 位置集合的最大 `iw` = `vmax` 形态 ✓。
  - **C1（钳制 span 载入）**：分三支证明 `∀ iw ∈ [base, base+span): (s0 ≤ iw < s0+LSPAN) ⟺ (0 ≤ iw < L_in)`，且 `0 ≤ s0`、`s0+LSPAN ≤ L_in`：① `0 ≤ base ≤ L_in−LSPAN` ⇒ `s0=base`；② `base<0` ⇒ `s0=0`；③ `base>L_in−LSPAN` ⇒ `s0=L_in−LSPAN` ✓。三支均成立 ⇒ 载入恒在界内、相对下标恒落 `[0,LSPAN)`（脚本 90 项结构断言一致）✓。
  - **E1**：`max(−inf, x) = x` ✓；`{+inf,−inf}`、全 `−inf`、NaN 窗口三种边界在 §0.6 R2 逐条核对成立（NaN 判决走独立 `eq` 掩码，与 `vmax` NaN 行为无关）✓。
  - 「两遍法」必要性：N3 要求 `acc` 定稿；单遍（以前缀最大值判等）退化为「首个局部峰值」——脚本 `N2N3-ord` 在 39/72 组检出该误形态（**非空检出**）✓，设计据此冻结两遍结构 ✓。
- ② **复杂度算术独立复算**：FLOPs `N·C·L_out·k_W` 与 manifest `roofline.flops`（`N*C*out_W*kW`）逐字一致 ✓；Bytes `(N·C·L_in + N·C·L_out)·eb + N·C·L_out·8` 与 manifest `roofline.bytes`（`(N*C*L_in + N*C*out_W)*elem_bytes + N*C*out_W*8`）一致 ✓；代表值复算：`smoke` 1 792 B ✓、`sincnet` 55 915 520 B ✓、`textcnn` 4 358 144 B ✓、`ecg` 29 339 648 B ✓（**本检视复算 = 29 339 648，与设计一致**）；**无中间缓冲 GM 往返**（§4.3 全部 UB 驻留、§4.4 明示无 GM 中间写回；与「扫描遍数 = 1」「峰值 `O((BL+LSPAN)·CH)`」自洽）✓；D-2 roofline 四项除法复算一致（1.4 ns / 44.4 µs / 3.46 µs / 23.3 µs）✓。
- ③ **负向断言主动复核**：候选 1/2/3「数学不等价」为数学事实（`max` 非线性、无逆元）——**数学不可能类**（只须数学推演）：独立复核设计反例 `x=[5,1,3]`、区间 `[1,3)`：前缀最大值 `P[3]−P[1]=0 ≠ 3` ✓；候选 8「1D 无可分离结构」✓ 且与 `ALG-pooling` 的 R3 补充实测（可分离 / 前缀和差分对两类池化均不降运算量）一致 ✓；候选 7「不采用」引 `TRAP-npuir-arange-pattern-with-whole-plane-reduce`（traps-compiler.md 第 373 行起，status **verified**，device 910B3，repro 在库；现象 = 派生 pattern in-chain 读回损坏、**values 精确而索引大面积错**，同族 `AdaptiveMaxPool2dIndicesFwdOp` 最终交回一般路径）——本检视打开该条目核对，设计与同族前作处置一致、无矛盾 ✓。**是否漏掉已知候选**：额外评估「两遍分块前缀/后缀最大值（van Herk–Gil–Werman）」——它是**两遍扫描**结构，对本带宽受限算子不降 GM 流量（口径以流量/遍数为主），且其「候选对取大」语义需额外携带下标裁决；`ALG-pooling` 的 R3 补充实测亦明示「窗口重叠重复读类结构优化对两类池化均不成立，直接 bin/窗口累加（C 轴向量化）为默认候选」⇒ 不构成被漏掉的更低复杂度候选，不判 fail ✓。
- ④ 并列候选裁决：R3/R4 无纸面拍板的并列候选；`CH`/`BL` 的判定裕度依赖未实测的窄向量 per-op 代价，已按设计 skill「实验裁决模式」给出 D-3 三件套（见 8-g）✓。
- ⑤ 容量/带宽淘汰的常数引用：`CONST-capacity-910B2C`（`constants.md` 第 8/23 行，192 KB/AIV、×1.7 膨胀、auto-multi-buffer=false 下 ×1.10–1.12）、`CONST-vector-launch-overhead`（第 46/59 行，~0.5µs/op）、`CONST-mte2-degradation`（第 82/97 行，3:1 R/W 地板 ~1.26 TB/s）均带版本戳，且设计显式标注「常数绑定 910B2C、本设备 910B3、跨型号复用属未实测假设」✓。

**（8-d）调研 → 设计一致性 — pass**：§1.6.0 选定 #0′ 实际驱动 §1.4（算法描述）、§3.1（公式拆解以 §1.6.1 优化后公式为输入）、§3.3（伪代码结构）、§5.2（C 轴向量 + BL/CH）、§6（循环结构）——本检视交叉核对无「调研选 A 设计做 B」✓；迁移侧：§0.5/§0.6 的四态处置与调研选族一致，§0.4 的 14 项优化手段意图逐项承接于 §0.6（含 #5/#6/#8/#9 的 indices 特有项），**无静默丢弃** ✓。

**（8-e）数学等价优化完整性 / 等价性论证正确性 — pass**
- §1.6.1 对选定算法做了系统审视（恒等变形、稳定化、CSE、算子降代换、访存削减、归约合并），采纳 **3 项**（C1/N2/N3）+ 语义锚 E1，四项均有「原式 → 优化后公式 → 等价性论证 → 收益估算」四要素 ✓；本检视逐项独立推演（8-c①）成立 ✓。
- 容差方面：本算子为**纯选择语义**（无累加、无舍入传播），`fp16/bf16` 的舍入面只发生在「值回写单次 `vcast`」与源一致，无 cancellation 风险 ✓；
- 显式否定结论：无除法/归一化 ⇒ 无「倒数乘」优化空间；无高代价算子降代换（无 exp/rsqrt 类）；无可 CSE 的重复计算（每元素仅一次比较；边界判据由 C1 消解、NaN 掩码由块级一次消解）✓；候选 1/2/3 不采纳理由正确 ✓。
- **v0 的措辞不实已修（A12）**：现为「torch CPU fp32 参照 + 目标 dtype 往返比对（0 ULP 判据）」，与脚本实际一致 ✓；内嵌表前言改为「按 config 聚合后的输出（数值与计数已逐项核对）」✓。
- 候选 7（C4/C4b）**仅数学等价性论证、不采用**（S1）：值半由 `specialisation_math_only()` value-only 机器确认（3 dtype 全 `EQUIV_PASS`）；不采用理由（索引半依赖已断裂的 `arange` pattern 链、无法证明独立）经本检视打开 trap 条目核对成立 ✓。

**（8-f）向量化替代完整性 / 不可替代理由 / 替代 API 佐证 — pass**
- §1.6.2 枚举 22 个计算点，覆盖 §3.3 伪代码与 §6 循环结构中的全部**循环**与逐元素计算点（初始化 / 取最大值 / first-hit / last-NaN / NaN 追踪 / 判决 / cast / 布局重排 / iw 广播 / 空窗口子句 / 单元级常量构造均已替代，替代 API 逐个有 `docs/` 条目佐证、优先 v-prefix）✓。
- 保留项 8 类理由具体：**#16/#17** tile 级顺序依赖（循环携带累加维 + 运行时行索引，循环体内已向量化）；**#18** 逐 lane 标量谓词门控向量 op（预填 `−inf` 的替代属动态偏移写入风险域，与 `PL-1.22` ⑥「守卫外提需 ≥2 空间轴，1D 唯一判据 `0≤iw<L_in` 无法外提」一致，无收益）；**#15** block 级钳制标量（1D 特有）；**#19**（v1 改写）单元任务解码 + persistent 调度 id `uid = t*num_kernels + cid` + 尾块边界（`real_c`/`BL_real`）—— block 级标量；**#20/#21** host 侧 trace 元数据 ✓（**A11 落实**：v0 的 #19 未含 persistent 调度与尾块标量，v1 已改写）。
- 无「以实现简单为由保留标量循环」✓；§6.1 结论与之一致（不出现逐元素标量循环）✓。

**（8-g）向量化轴与布局决策完整性 — pass**
- §1.6.3 按**窗口/池化类必选候选**枚举：候选 1「I/O 原生 NCL + L 轴」、候选 2「核内 NLC + C 轴」、候选 3「host permute」、候选 4「Cube/implicit-GEMM」、候选 5「L 轴分块（与轴选择正交的容量维）」；逐候选有量化评分（整除性 / 尾 lane 浪费 / 累加链形态 / repack 代价 / UB 影响）✓。
- 弃选均有量化理由，且**代价前提经本检视亲自打开 pattern-library 核对**：`PL-1.2`/`PL-1.4`「`sW≠1` 跨步系数阻碍向量化 → 98.3% 标量、换轴后 5.6x」（layout.md 第 5 行）、`PL-1.3` host permute「两次实测 **106–147 µs**」（layout.md 第 94 行）、`PL-1.1` 核内融合转置链「4 次合计 **~5.4 µs**」（layout.md 第 27 行）、`PL-1.4`「CH64 21.6 µs < CH32 26.7 µs < BH2/CH16 44.8 µs」（layout.md 第 111 行）——**逐条与设计引用一致，无矛盾** ✓；迁移注「源 GPU 的并行轴是输入而非结论」并独立改判为 C 轴 ✓；无「I/O 是 NCL 所以核内也 NCL」式论证 ✓。
- 布局决策与实现一致：§1.6.3 选定的「C 最内 + 向量轴 C + span 分块」与 §3.3 buffer 形状/内层向量维、§4.3/§4.5 buffer 形状与预算、§6 循环结构三方一致 ✓。
- **D-3 实验裁决模式（主选 + 备选 + 裁决计划）三件套齐全**：主选 = 容量允许的最大 `CH`（≤min(C,64)）；备选 = `CH` 减半 + `BL` 放大，结构到 Stage 4 可直接实现（buffer 形状同主选仅 CH 减半，**分核三要素按新 `num_logical` 重算**：sincnet CH32/BL32 → num_cblk=3, num_oblk=43, num_logical=4 128, num_local=104；ecg CH32/BL35 → num_cblk=4, num_oblk=30, num_logical=1 920, num_local=48 — **本检视复算一致**）；裁决计划含代表 shape（sincnet/ecg）、指标（msprof op Task Duration）、未知常数（窄向量 per-op 代价）的实测反解方法、**明确判定阈值**（备选 < 主选 ×(1−0.10) 才翻转）、回写路径（`perf-feedback.md` 受控路由）；shape 分派评估（`lru_cache` 按 shape 特化，支持按负载分派）✓。
- **编译期特化谓词的充分条件（该行）**：本设计**无几何特化谓词**（唯一 trace 期分支是 dtype 谓词 `is_f32`）⇒ 无「依赖变量全集 × 集合关系」需枚举，该行以 n/a 满足；§5.2/§5.5 进一步声明 `num_logical` **单一定义源**、无 C4b 口径分叉 ✓。

**（8-h）与下游章节一致 — pass**：§3.1 公式拆解以 §1.6.1 优化后公式（`base/span/LSPAN/s0` 钳制 = C1、`min{...}` = N3、`max{...}` = N2、空窗口分支 = N3 第 4 支）为输入 ✓；§6 循环结构与 §1.6.2 结论一致（被判向量化的点未再出现逐元素标量循环）✓。
- 建议级残差：§1.6.2 #9/#10 的布尔缓冲命名/判据来源与 §3.2/§3.3/§4.3 有差（**见维度 7 warn 与「非阻塞改进建议」**）；不影响冻结路径与 §3.1/§6 的算法一致性。
- C1 收益口径（v0 建议 #3）已修（A6）：现为「访存削减 = 窗口重叠复用 + 连续 span 合并访存（非纯读次数）」，读次数比 `(num_oblk·LSPAN)/(L_out·k_W)`（sincnet `86·48/(1365·3)=1.008`≈1.01、ecg `61·41/(1023·5)=0.489`≈0.48），halo 比 sincnet ≈1.01 / ecg ≈1.22 —— **本检视复算一致** ✓。

## 非阻塞改进建议

- **NB-A（§1.6.2 #9/#10 布尔缓冲口径统一，建议级）**：把 §1.6.2 #9 的 `T.vcmp(flag, zeros, cfinal, "eq")` 与 #10 的 `T.vcmp(nan_idx, zero_i32, cfinal2, "ge")` 统一为**单一** `cfinal`（沿用 §3.2 11a / §3.3 / §6.2 的 `nan_idx>=0` 判据，值/索引/空窗口三处复用），删去未在任何 buffer 表出现的 `cfinal2`；若确需保留两种等价判据，则在 §1.6.2 明注「二者数学等价」「实现采用单缓冲复用」并同步 §4.3/§4.5 的 buffer 清单。理由：与冻结路径（§3.3/§6.2 + §9.2 实现级约束⑥）保持一致，避免 Stage 3 照字面多分配 `(BL,CH)` bool（1 024 B）使 `sincnet ×1.7` 至 ≈197 146 B 微超判据。
- **NB-B（§4.3/§4.5 补齐 `(1,CH)` 临时 buffer，建议级）**：§3.3 的第一/二遍使用 `eq2b`/`eqb`（每 `(kw,owl)` 一次的 `(1,CH)` bool），未列入 §4.3 中间缓冲区表与 §4.5 逐 buffer 表。合计 ≤128 B，量级可忽略（不改变任何容量判定），但为「逐 buffer 求和 = 闭式值」的全量口径完整，建议补列或在 §3.3 注明「由 Stage 3 复用 `eq_mask` 行切片 / 单元级 scratch」。
- **NB-C（`verify_equiv.py` 耗时记数，建议级）**：本机重跑实测 ≈84 s（宿主重载：`user 40m32s`），高于 §1.6.1 记的「≈12–43 s」；脚本仍为分支采样、远低于「几分钟」上限，非阻塞。可在 §1.6.1/§A12 的耗时栏增补「重载宿主可达 ~80 s」或在 Stage 3+ 复测时回填区间。

## 检视备注（S1/S2/S3 用户硬约束的显式核对结论）

- **(S1) C4/C4b 处置 — 满足**：§0.6 R5 与 §1.6.1 独立段落**两处均显式声明**「C4/C4b 仅为**数学等价性论证**（math-equivalence argument），本设计**不采用**（NOT adopted）」；值半以 `specialisation_math_only()` 的 value-only 机器论证留档（3 dtype 全 `EQUIV_PASS`）；不采用理由（索引半依赖 `TRAP-npuir-arange-pattern-with-whole-plane-reduce` 已验证断裂链）经本检视打开该 trap 条目核对成立；**§3/§5/§6 无 C4/C4b 分支**（独立全文检索确认：§3.3 伪代码唯一分支为 dtype 谓词 `is_f32`；§5.2/§5.5/§9.1 显式声明单一 `num_logical` 口径；`textcnn-global` `L_out==1` 走一般路径 `BL=1`）✓。v1 新增的空窗口子句为**全 shape 统一**的索引 epilogue 子句，**不引入任何几何特化分支** ✓。
- **(S2) toolchain stamp — 满足**：第 24–27 行为 pattern-library 口径 `tilelang 0.1.2+ba60e99c16439d6fa6a59acdfca162c17b2bf8b2（dev root build）+ torch_npu 2.9.0.post8 + CANN 9.0.0 + Ascend910B3` —— 与 `traps-compiler.md` 第 357/381 行同日（2026-10-14）记录的 dev root build hash **完全一致**（本检视核对）；CANN 9.0.0 与设备从 `examples/TileOPs/reports/tileops/*/run.json` 的 `setup.environment`（`cann: 9.0.0`、`npu: Ascend910B3 x 1`）互证 ✓；仓库快照 `892e2c18d80c37795727236b5ebbe145626a8313` 在第 28–29 行**单独**记为「任务快照（task snapshot，非 tilelang 版本）」并显式说明「`tilelang.__version__` 后缀与 `git rev-parse HEAD` 同值」——未与 tilelang 版本混淆 ✓。
- **(S3) verify_equiv 采样口径 — 满足**：脚本头部 SCOPE 声明不枚举 C4/C4b，`(k,s,p,d,ceil)` 按条件分支采样（8 个代表 config：`p>0,s!=k` / `p>0,s!=k,Lnd` / `d>1,p>0` / `ceil,p>0,s!=k` / `out==1,s!=k` / `ragged` / `same_s,k,p0` / `identity k=1`）+ **空窗口分支 `EW-min`/`EW-oob`/`EW-l2`**；`specialisation_math_only()` 仅 1 配置/档 × 3 dtype 的 value-only 论证；无任何全 C4/C4b 枚举 ✓（本检视通读脚本确认，重跑 ≈84 s 未触发长跑）。
