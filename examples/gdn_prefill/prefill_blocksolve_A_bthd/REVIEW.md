# prefill_blocksolve_A_bthd 设计文档检视报告（revision 2 复检 / retry_count=2）

> 本文件只写 `REVIEW.md`；未改动 `DESIGN.md`、kernel、探针与任何状态文件。
> 下列所有机检数字与真机数值均为**本轮独立执行**所得（2026-10-17，Ascend910B3 / Chip 6 / CANN 9.0.0 / Expert）。

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/gdn_prefill/prefill_blocksolve_A_bthd/DESIGN.md`（**revision 2 / v2**，1525 行）
- 上一版: `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_v1.md`（v1，对应 `review_v1.md`）；
  更早 `design_v0.md`
- 任务类型: 迁移（GDN prefill 流水线 函数 2/6）
- 源算子: `/mnt/workspace/TileOPs-main/src/tileops/kernels/linear_attention/gdn/prefill_prepare.py`
  `_prefill_blocksolve_A_bthd_tl`（L158–L651）+ wrapper `prefill_blocksolve_A_bthd`（L654–L708）
  （本轮逐段 Read L158–L708，含 L280–L519、L520–L719）
- 检视维度: 迁移任务 9 项（0 源算子理解与迁移分析 / 1 API / 2 内存层级 / 3 Tiling / 4 技术约束 /
  5 循环与同步 / 6 验证方案 / 7 完整性与一致性 / 8 算法优化分析）
- 本轮性质: 设计修订循环第 2 次（`max_retry=3`），复检上轮 2 个阻塞项 + 11 条建议项
- **阻塞级问题数: 0**
- **建议级问题数: 6**（全部非阻塞，含 3 条计数/行号精度、2 条 Stage 3 观测点、1 条探针覆盖补强）
- 机检门禁: `review_lint`（结论字面量 + 9 维度 + 维度 8/0 证据）通过；`statectl gate 2` → `pass: true`

## 机械复核（D-1 / D-5：本轮独立执行，不采信设计自述）

### A. 调度/flag 双引擎仿真 `history_version/design_schedule_sim.py`：`SCHEDULE_PASS`

实跑 `python3 .../history_version/design_schedule_sim.py`（末行 `RESULT: SCHEDULE_PASS`）：

| 表 | 断言 | 实跑结果 |
|---|---|---|
| A（17 个 in-scope 形状） | v1 漏落盘数 / v2 覆盖 / 重复 / 每 block 任务数 / 尾块后缀 | v2 **exactly-once**（重复 0）；每 block 任务数 ∈ {NPER−1, NPER}；尾块后缀全 `yes` |
| B（每形状 × 每 block，两引擎**阻塞协程**） | 死锁 / 6 flag `#set==#wait` / 槽位 `W→R` 严格交替 / 落盘 exactly-once | 17 行全 `NO / yes / yes / yes` |
| C（`T=1..2048` 穷举） | 落盘多重集 == `range(T)`；每 block 任务数 ∈ {NPER−1,NPER} | **2048 例，违规 0** |
| D/E/F | NPER−1 形状分布、`T=25` 逐 block trace、`T=2` 事件全序 | 与 §7.7 表 D/E/F 逐行一致（`T=25`：15 个 block 各 1 任务，v1 全漏、v2 全中） |

**"不是硬编码"的两项证据**（我另行验证）：

1. 脚本 A 表对 **v1 也做了转写并复现了缺陷**，漏落盘数与 `review_v1.md` 的**独立**仿真逐项相同：
   `T=8192→8`、`1024→16`、`752→8`、`3072→8`、`64→16`、`25→15`、`1→0`。一个硬编码 PASS 的脚本不可能同时复现旧缺陷的数字。
2. 我**自写**一份不 import 该模块的转写（只按 §3.3 的 `if it>=1 and (t-K)<TASKS` + `last_it=(TASKS-1-bid)//K`）：
   - `T=1..20000` 的 exactly-once **0 违规**（远超设计自报的 1..2048）；
   - 每 block 任务数 ∈ {NPER−1, NPER}；
   - `scatter` 的 slot 与被落盘任务一致（循环内 `(it−1)%2`、epilogue `last_it%2`）**0 违规**。
   脚本的 `dims()`/`schedule_v2()` 以 `(T, K, NPER, head, nch)` 为参数，无形状硬编码。

### B. 等价性 `verify_equiv.py`：A1–A7 全 `EQUIV_PASS`，与 §1.6.1 内嵌表逐格一致

实跑 `python3 examples/gdn_prefill/prefill_blocksolve_A_bthd/verify_equiv.py`：

| 项 | worst rel | worst ulp | worst viol | 判定 | 与 DESIGN §1.6.1 表 |
|---|---|---|---|---|---|
| A1 | 3.0e-08 | 0.5 | 0.0e+00 | EQUIV_PASS | 一致 |
| A2 | 1.2e-07 | 1.3 | 0.0e+00 | EQUIV_PASS | 一致 |
| A3 | 3.7e-04 | 3957.5 | 0.0e+00 | EQUIV_PASS | 一致 |
| A4 | 4.0e-06 | 1.8 | 0.0e+00 | EQUIV_PASS | 一致 |
| A5 | 9.3e-05 | 1.0 | 0.0e+00 | EQUIV_PASS | 一致 |
| A6 | 4.6e-04 | 310.8 | 0.0e+00 | EQUIV_PASS | 一致 |
| A7 | 合法行 4.9e-05 | — | 0.0e+00 | EQUIV_PASS（负向对照逐条 fail） | 一致 |

- **A6 负向对照有牙齿**：`as-designed` viol 0；`v0 direction` rel 1.3e-1 / viol 3.7e-1；`v0 sign` rel 4.4e-1 / viol 4.4e-1。
- **符号裁决**：`|source − inv(I+S)| = 2.776e-17`、`|source − inv(I−S)| = 4.361e-01` ⇒ `N = −S_sys` **CONFIRMED**；
  `blocked_source(fp64)` 自检 `2.7756e-17`，与 §8.1 的 1e-10 门槛自洽。
- **A7 明细（我实跑，共 20 行，与 §1.6.1 A7 表逐行一致）**：
  - S1（block 首任务即尾 chunk，越界行注入 NaN）`valid=56/36/1` + R3-a/R3-b 均在 ⇒ 合法行 rel 4.7e-5 / 4.9e-5 / 0，**0 个 NaN/Inf**，padding 行 **== `e_i`**（`torch.equal` 精确）；
  - S1 去掉 R3-a **或** R3-b 任一条 ⇒ **整块 4096/4096 元素 NaN**（合法行也被污染）；
  - S2（尾 chunk 是后继任务，越界行是上一整块的有限值）去掉 R3-b ⇒ padding ≠ `e_i`，而 viol 仍 0（故该列只由 padding 断言拦截）——与设计描述**完全一致**。
- A7 的建模忠实度我逐行读了：`k_use/g_use[valid:]=0` 对应 R3-a、`b32[valid:]=0` 在 `vcast` 之后、`vbrc` 之前对应 R3-b，
  随后走的是 §3.3 的**同一串** `vbrc→vsub→vmin→vexp→vmul(-1)→vmul→vmul(G)→vmul(mask01)` 与 6 级 doubling，
  参考 = `block_diag(inv(I+S[:valid,:valid]), I)`（= 源 clamp+β=0 的解）。不是"另写一个能过的算法"。

### C. `design_calc_check.py`：4 项 checks（skip 项已人工复核）

```
{"design": "examples/gdn_prefill/prefill_blocksolve_A_bthd/DESIGN.md",
 "checks": [
  {"check":"ub_budget","status":"skip",
   "detail":"no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"},
  {"check":"l0c_budget","status":"pass","detail":"block 16x16 x fp32 accum = 1.0 KB vs L0C 128 KB"},
  {"check":"core_split","status":"skip",
   "detail":"insufficient parsed dims {'M': 64, 'N': 64}; declared logical cores = 64"},
  {"check":"r3_metrics","status":"pass",
   "detail":"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"}]}
```

- `ub_budget` **skip → 人工复算通过**：`7×16 384 B (114 688) + 8 192 (UB_Y) + 1 152 (5 个小 buffer) = 124 032 B = 121.1 KiB`；
  按 Expert 显式分配口径 ×1.12 ≈ **135.7 KB < 192 KB**（余量 56 KB）。不做原地复用的最坏形态 168 KB ×1.12 ≈ 188 KB 仍 < 192 KB，
  设计如实给出两档并明确"目标是 7 个独立 buffer、不要 alias"（§4.3 / R13）。
  （脚本内 `UB_INFLATION=1.7` 是 auto-multi-buffer 口径，对 Expert 逐 buffer 显式分配不适用——`constants.md` 已记两个数据点。）
- `l0c_budget` **pass 但属解析假阳性**（解析到 "16x16 x fp32 = 1.0 KB"）→ 人工复算 `l0g+l0t+l0p = 3×16 KB = 48 KB < 128 KB` ✓。
- `core_split` **skip → 人工复核通过**：`T = 752/1024/3072/8192` 复算全对（`ceil(3000/64)=47→752`、`512×16=8192`），
  `K=min(20,T)`、`NPER=ceil(T/K)` 为编译期常量，跨步映射负载差 ≤ 1（仿真 C 表 2048 例 + 我自写的 20000 例）。
- `r3_metrics` **pass**（四口径齐全）。

### D. 真机探针复跑（`_probe_env.sh`：CANN 9.0.0 + dev root + `TILELANG_ASCEND_MODE=Expert`）

| 探针 | 本轮实跑 | 与 DESIGN 记载 |
|---|---|---|
| `design_probe_gate_sign.py` | `e vs exp(g_i−g_j) = 1.412e-06`；`vs exp(g_j−g_i) = 1.067e+14`；`N vs −β·e·G = 1.118e-07`（`|ref|max 0.126`）；`vs +β = 2.519e-01`；`CHAIN fp16 2.783e-05 / fp32 5.547e-09` | 与 §3.4/§11 的 `1.4e-6 / 1.07e14 / 1.1e-7 / 2.5e-1 / 2.8e-5 / 5.5e-9` **逐项复现** |
| `design_probe_store_layout.py` | 目标行区 `2.419e-04`、相邻槽 `max|other| = 0.000e+00`、fixpipe 量化 `0.000e+00`；`contig 0.090 µs / strided 0.233 µs（2.59×）` | 正确性三项**逐项复现**；时延与设计记的 `0.084/0.194（2.29×）` 同量级（run-to-run 波动，见建议 4） |
| `design_probe_mask.py` | `MA sum=0`、`MC sum=0`、`MD sum=1008 / err=0.5`、`MB sum=2016 / err=0` | 与 §1.6.2 V9 / §3.4 / §11 的"MA/MC 静默全 0；MD 是 0.5 倍（不作依据）；只有 MB 精确"**一致** |
| `design_probe_doubling_cube.py` | `T1 2.922e-08`、`T2 2.699e-07`、`per 6-op level = 1.563 µs`（0.26 µs/op） | 与 `2.9e-8 / 2.7e-7 / 1.52 µs（0.25 µs/op）` 一致 |
| `NPUUtils.get().get_aicore_num()` | **20** | 与 §5.2/§11 一致 |

> 真机可用，未降级为"未复现"。

## 上一轮两个阻塞项的闭环判定（本轮核心）

### 阻塞项 1：一任务滞后落盘漏掉每 block 末任务 —— **已闭环**

| 证据 | 内容 | 我的独立核对 |
|---|---|---|
| E1 修法落点 | §3.3 L844–L850：`scatter` **移出** `if t < TASKS`，守卫改为 `if it >= 1 and (t-K) < TASKS`（绑定**被落盘任务**）；循环后 `last_it = (TASKS-1-bid)//K`，`if last_it == NPER-1` 才补一次 | 与上轮建议的修法逐句一致；`§6` 的"落盘位置"行、`§7.4` 的配对证明、`§13.1` 的对照表**同源**，无第三种说法 |
| E2 覆盖性证明 | 循环内 `it=1..NPER-1` 覆盖 `i=it-1`，条件 `(t-K)<TASKS` ⟺ `i ≤ last_it`，故覆盖 `i=0..min(last_it, NPER-2)`；`last_it==NPER-1` 时 epilogue 补最后一个，`last_it==NPER-2` 时 epilogue 不执行 | 我独立推演成立；关键是"任务数为 NPER-1 的 block 在 `it=NPER-1` 时 `t≥TASKS`，但 scatter 在守卫外仍执行"这一条已被覆盖 |
| E3 机检 | §7.7 A/B/C 表 | 我实跑同结果；另用**自写实现**扫 `T=1..20000`（含 `T>2048`）**0 违规** |
| E4 旧缺陷可复现 | 仿真对 v1 也转写 | 漏数与 `review_v1.md` 独立仿真**逐项相同**（8/16/8/8/16/15/0）⇒ 脚本确有判别力，不是"永远 PASS" |
| E5 flag 配对恢复 | 每任务恰好 1 次 `gram→set f_g`、1 次 `set f_s`、1 次 `set f_a`、1 次 `scatter→wait f_a` | 仿真逐 block 断言 6 个 flag `#set==#wait` 全 `yes`；`A_pad` 每任务恰好一次写 |
| E6 L0 暴露层级 | 新增 L0-i（`S=1600, head=1` ⇒ `T=25, K=20, NPER=2`，15 个 block 各 1 任务） | 与上轮建议 11 的要求吻合；该形状在 L0 即可暴露"任务数 < NPER"形态 |

**判定：已闭环。** 残留仅是"dynamic guard + 动态 flag id 能否 lower"的实现风险（§9 R5 已给两条回退），
属 Stage 3 编译期即可暴露的问题，不构成设计缺陷。

### 阻塞项 2：尾 chunk 越界行读未初始化 SRAM ⇒ NaN 经 N² 回灌合法下三角 —— **已闭环**

**机制（§0.6 R3 / §3.2 / §3.3 同源）**：R3-a 块首一次性确定性零初始化
（Cube：`T.copy(kz→l1k)`；Vector：`T.copy(zero_g→ub_gc/ub_gr)`，均为 MTE2）+ R3-b 每任务"整列 `ub_b16` 清零 → 再覆盖 `valid` 行"
+ 动态 extent 装载 + host 常量 0/1 掩码 + `A_pad` padding 视图。

**引理独立复核（我按 §3.3 的语句逐条推演，不采信散文）**：

- **L-1（尾块是 block 内后缀）**：`t = bid + it·K` 随 `it` 单调增 ⇒ `chunk = t//head` 单调不减；`valid<64` 只可能出现在最后一个 chunk
  ⇒ 同一 block 内 `valid<64` 的任务必排在所有 `valid=64` 任务之后。**成立**（仿真对 17 形状逐 block 断言 `yes`）。
- **L-2（越界任务读到的每个值都有限）**：由 L-1，越界任务之前只跑过更靠前的 chunk：`valid=64` 的任务用真实 GM 数据覆盖全部 64 行
  （输入有限是 contract）；同一尾 chunk 的其它 head 任务只覆盖 `< valid` 的行；若该越界任务就是 block 首任务，则 R3-a 给 0。三种情况都有限
  ⇒ `G`、`d`、`e=exp(min(d,0))∈(0,1]` 全有限。**成立**（注意 `vmin` 只用于限幅，不承担"把 NaN 变有限"的职责——设计没有这么用）。
- **L-3（`N` 的 padding 行精确为 0）**：R3-b 使 `bn` 的越界行**精确为 0**（先清零后覆盖，越界行不被覆盖）⇒
  `N[i,:] = 0·(有限)·(有限)·mask01 = 0`（无 `0×NaN`）；再 `P_j` 的零行在 `P²`、`P·T` 中都只产生 `0·有限`，`T` 从**精确的 `I`** 起步
  ⇒ `A` 的该行 = `e_i`。**成立**。
- **两条污染通路都真实存在且被同一机制堵住**：通路 1（同一行内 `i<valid≤j` 的 `0×NaN`）、通路 2（`T←T+P·T` 中 `P[i,k]=0 (k>i)` 但
  `T[k,j]`（`k≥valid>j`）是下三角块的非零元 ⇒ `0×NaN`）。**这条论证是对的**：只靠"行独立"或单一因果掩码确实不够。
- **与源语义等价**：源 clamp k/g 到 `end-1` + `β=0` ⇒ `S_sys` 该行整行为 0 ⇒ `A` 该行 `e_i`；源**不写**该行，本设计写入 `A_pad` 的
  padding 区，被 `A_pad[:total_tokens]` 视图裁掉。差异只有"k/g 取 0 vs 取末行"，因 `β=0` 而**不可观测**。**成立**。

**机检 A7（我实跑）**：见上文 B 节。as-designed 0 NaN + padding 精确 `e_i`；**去掉 R3-a 或 R3-b 任一条 ⇒ 4096/4096 NaN**（合法行亦被污染）；
S2 去 R3-b ⇒ padding ≠ `e_i`。负向对照 fail 的**位置与机制解释一致**，判据有牙齿。

**一致性**：`§0.6 R1 末` / `§3.1 S3` / `§3.5.2 第 1 条` / `§13.2` 现描述**同一套**机制；我 grep 全文，"留旧值"只作为 **v1 缺陷的历史描述**出现，
不再作为本设计的机制陈述（上轮的"两处互斥表述"已消除）。

**判定：已闭环。** 残留：R3-a/R3-b 的两次 MTE2 顺序是**文档级论证**（同管道程序序），§13.4 已披露，L0-d/L0-i 为首个实测点，R4 有回退
（"先 `T.clear` 再 `T.copy` + `T.pipe_barrier`"）。这是"已知且已给出观测点/回退"的实现风险，不是设计缺陷。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

- **语义理解 — pass**。逐行对照源码：`S_sys[i,j]=β_i·exp(g_i−g_j)·(k_i·k_j) (i>j)`、`A=(I+S_sys)^{-1}` 与 L429–L505 / L508–L540 一致；
  对角块 `a_s[0/2/5/9] = −G·s`（L480–L499）、非对角块 `a_s[1/3/4/6/7/8] = +G·s`（L500–L505）、行串行 `x = e + a·x`（L519–L540）
  ⇒ `X=(I−a)^{-1}=(I+S)^{-1}`，符号归属**由源码事实 + fp64 裁决（2.776e-17 vs 4.361e-1）双证**；
  `g` fp32（L216）、`accum_dtype="float32"`（L199）、`solve_dtype=dtype`（L200）、`beta_s[t]=cast(step,dtype)` / 越界置 0（L295–L297）、
  `k`/`g` 越界 clamp 到 `end-1`（L290/293、L342–L346）、输出按 `base+row<end` 逐行掩码（L622–L647）、`A=torch.empty(...,k.dtype)`（L688）
  ——全部与 §0.1/§0.2 对应。
- **算法解读完整性 — pass**。P0–P9 覆盖源码全部计算语句；P7 的 6 条块回代公式与 L543–L618 逐式对上（`A_10=−X_1S_10X_0` … `A_32=−X_3S_32X_2`）；
  我按行号点过块回代 gemm = **16**（L543/548/555/559/564/569/574/579/583/587/592/597/601/606/611/616）+ Gram 10 次/段
  ⇒ `26（dim_k=64）/ 36（dim_k=128）` ✓；host 侧 `T.min(cid,last)` 的"幂等重算、非早退"与 L263–L266 注释一致 ✓。
- **优化手段覆盖 — pass**。O1–O12 齐全且逐项标硬件依赖；O5（门表复用 + 除法复用，非"数值稳定化"）、O12（clamp 幂等重算，非早退）、
  O6/O8（行串行链的舍弃理由）均已按事实改写。
- **耦合性判定 — pass**。抽检：`store_fixpipe` 只能落 GM、`T.clear` 是 VBrcOp（`docs/Tilelang.language/内存操作/T.clear.md` §3）、
  二维 grid → 一维任务号、`threads=` 无效、`out_idx` 回喂禁用，均与 `docs/`、`ascend-constraints.md`、`EXPERT_FINDINGS.md` §12 不矛盾。
- **NPU 重设计可行性 — pass（本轮重点）**。§0.6 R1 的恒等式 `N≜−S_sys`、`A=ΣN^j=(I−N)^{-1}=Π_{i<6}(I+N^{2^i})` 独立推演成立
  （`N` 严格下三角 ⇒ 幂零指数 64；`{1,2,4,8,16,32}` 子集和唯一覆盖 `0..63`）；R3 的三条引理见上文，均成立；
  **上轮"§0.6 R1 末句与 R3 互斥"的问题已消除**（现在只有一个机制）。
- **设计与决策一致性 — pass**。§1–§7 是 §0.5/§0.6 决策后的算法：13 个 gemm 的 doubling 链、`y_blk` 三步落盘、
  `mask01` 常量 tile（而非被证伪的 `arange+vcmp`）、`−β` 唯一落点 —— 与 §1.6.0 选定 C1 同源，无"判重设计、照抄源方案"。
- **golden 独立性 — pass**。§8.1 以 §0.1 语义独立实现（`torch.linalg.solve(I+S[:valid,:valid])`，fp64，只写 `valid` 行）；
  `blocked_source` 交叉校验用的是**源**分块算法（fp64 存储）⇒ 两条参照都不经过 doubling。

### 1. API 可行性 — pass

- `T.Scope/alloc_L1/alloc_L0C/alloc_ub/rs/gemm(size,initC,b_transpose)/copy/vbrc/vsub/vmin/vexp/vmul/vcast(round_mode=)/
  sync_block_set|wait` 均在 `docs/Tilelang.language/` 与 `examples/gdn_prefill/probe/experiments/*` 有佐证；Expert 手动同步、未用 `threads=` ✓。
- **revision 2 新增的 3 类 API 语义都有落点**：
  1. `T.copy(GM 零张量 → L1)`（R3-a）：`T.copy.md` §1 方向表 GM→L1 合法；且**明确不用 `T.clear` 清 L1**
     （`T.clear` lower 成 `hivm::VBrcOp`，是 Vector 写通路）——这条负向断言我打开 `T.clear.md` §3 核对，**成立**。
  2. `T.copy(zero_b → ub_b16)` 后再 `T.copy(beta[…0:valid…] → ub_b16[0:valid])`：同 dtype、extent 由 dst 推断 ✓；顺序不可交换（设计已写明）。
  3. `last_it` 与 `last_it % 2` 作为**运行期**标量参与 `T.copy` 的 slot 下标与 flag id：与既有 `F_S + slot` 同形，
     先例 `_gqa_prefill_fwd_kernel.py:351`；回退方案（按 2 展开成常量 id）已在 §3.3 自查段给出。
- `(64,1)→(1,64)` 的转置落位（`g_j`）标注为"文档未载、探针定案"，真机 `1.4e-6 / 1.07e14` 双向对拍 ✓（本轮复现）。
- `vcast` 第 3 位置参数是 `size`（必须 `round_mode="rint"` 关键字）——探针实测位置写法编译失败，设计已在 §3.2/§11 记为陷阱 ✓。

### 2. 内存层级规划 — pass

- 搬运路径：`GM(k)→L1(MTE2, T.copy slice)→L0A/B→L0C→GM(gs_G)→UB→Vector→GM(ws_S)→L1→L0C(6 级)→fixpipe→GM(y_blk)→UB→GM(A_pad, strided)`
  + R3-a 的 3 条零填充支路，**无越级**（无 GM→L0、无 L1↔L1）✓。
- **UB**：121.1 KiB 手工（7×16 KB + 8 KB + 1.1 KB）；×1.12 ≈ 135.7 KB < 192 KB ✓。两档预算（121.1 / 168 KB）都 < 192 KB，且明确"不做 alias"。
- **L1**：`l1k(16 KB, dim_k=128) + l1e/l1p/l1t(3×16 KB) = 64 KB` < 512 KB ✓（`l1q` 已删，且本版实测 A/B 同 buffer 可行）。
- **L0C**：`l0g+l0t+l0p = 48 KB` < 128 KB ✓；三个 L0C 互不复用区域，gemm 的 dst 一律整块 buffer（`CG-2026-0011`）✓。
- 每个 buffer 均标层级；`ub_gc(64,1)/ub_gr(1,64)/ub_b16(64,1)/ub_b32(64,1)/ub_bn(64,1)` 与 `vbrc` 的 rank 断言一致 ✓。
- 机检 `ub_budget` skip / `l0c_budget` pass 的处置见"机械复核 C"。

### 3. Tiling 策略 — pass

- **分核三要素齐全**：① `T = nch_total×head` 并给 5 个形状表（1/752/1024/3072/8192，我复算全对）；
  ② 物理核数 `get_aicore_num()` **本轮实查 = 20**，并按 MixCV/Cube 口径不乘 2 的论证成立；
  ③ 规模判定 + `K=min(20,T)` + `NPER=ceil(T/K)` + `t=bid+it·K`，负载差 ≤ 1（仿真 C 表 2048 例 + 我自写 20000 例，0 违规）。
- **举例算术已更正**：`T=1024 → 4 个核 52 + 16 个核 51（208+816=1024）`；`T=752 → 12×38+8×37=752` —— 两例我都复算通过。
- **对齐**：Gram `[64,dim_k,64]`、链 64³、尾轴 64（256 B）全为分形 16 的整数倍，无 K/M/N 尾块；行方向的非整除由
  "动态 extent + 确定性归零 + 掩码 + `A_pad` padding"处理 ✓。
- 写冲突：任务边界互不重叠（各自 64 行 × 各自 head 列），无原子/加锁需求 ✓。

### 4. 技术约束检测 — pass

一维 `T.Kernel(K)` ✓；不用 `threads=` ✓；L0C 48 KB ✓；GEMM 全整除 ✓；单 Cube + 单 Vector scope ✓；不用 `out_idx` 回喂 ✓；
`T.gemm` dst 为整块 L0C 而非 region ✓；GM→L1 跨步装载用 `T.copy` **slice** 形态而非 `load_nd2nz`（`CG-2026-0004` /
`TRAP-load-nd2nz-strided`，我打开条目核对：base+size 形态同样错、slice 形态 bit-exact，引用准确）✓。

### 5. 循环与同步 — pass

- 循环结构：任务循环 `T.serial(NPER)`（静态上界）；6 级链 Python 静态展开（每级 buffer/slot 不同、末级无平方，理由充分）；
  `T.Parallel/Pipelined/Persistent` 不使用均有依据 ✓。
- **落盘覆盖（上轮 fail 点）已修**：见"阻塞项 1"。
- **flag 配对与 WAR（我逐条走查 + 仿真背书）**：
  - `gs_G`：Cube 在 `it` 重写 `gs_G[(it+1)%2]`；该 slot 上一读者是 Vector 的 `it−1`，而 Cube 在 `it−1` 已 `wait(f_s[(it−1)%2])`，
    该 set 发生在 Vector 读完 `gs_G` 之后 ⇒ 写不覆盖未消费数据 ✓。
  - `ws_S`：Vector 在 `it` 写 `ws_S[slot]`；上一读者是 Cube 的 `it−2`，它读完才在 `it−1` 发 `gram(t)`（即 Vector 本次的 wait 对象）⇒ 安全 ✓。
  - `y_blk`：Cube 在 `it` 写、Vector 在 `it+1` 读；下一次写是 `it+2`，其前 Cube 已 `wait(f_s[it%2])`，而 Vector 的 `it+1`
    **整体程序序（含 scatter 的读）** 先于它 set 该 flag ⇒ 读先于下一次写 ✓。
  - 三条都由仿真的"槽位 `W→R` 严格交替"列逐 block 背书（17 形状全 `yes`）。
- `f_a` 的 epilogue 分支：`last_it == NPER−1` 时补的正是循环未覆盖的那一个，且 `wait` 的 flag 由 Cube 同迭代 set ⇒ 不会多 wait ✓。
- **披露项（接受）**：flag id 多轮复用是"计数语义"的**未文档化假设**（§7.5 显式标注 + `14_two_loop_handshake.py` 2 轮 + L1-a 4 轮实测点 +
  回退"每任务轮换 id ≤ 12 / 每 2 轮全同步"）。按"假设 + 可回退 + 有实测点"三件套接受，不再作为阻塞。

### 6. 验证方案 — pass

- **Golden — pass**：纯 torch CPU fp64、按 §0.1 语义实现；`rows=clamp(arange(base,base+64), max=end−1)` + `bb[valid:]=0` 精确对应源 clamp/β=0；
  只写 `valid` 行；`A` 初始化为 0（padding 区不参与比较）。
- **交叉校验 — pass**：`blocked_source(store_dtype=fp64)` 实测 **2.7756e-17**（门槛 1e-10）；fp16 通道单列为"量化层（信息性）"✓。
- **L0/L1/L2 计划 — pass**：L0-a..i + L1-a/b + L2（12 组合）+ Boundary 分层与门槛清晰；
  **新增 L0-i（`S=1600,head=1`，`T=25`）填补了"每 block 任务数 < NPER"的 L0 盲区**（上轮建议 11 已闭环）；
  **L0-d / L0-i / Boundary 把"0 个 NaN/Inf + padding 行 == `e_i`"写成硬断言**，把阻塞项 2 的机制纳入回归。
- **精度标准 — pass**：fp16 `5e-3`（abs+rel）、bf16 `4e-2`，均严于 skill 默认（1e-2），并给了"4.6e-4 观测值 × ~10 余量"的推导；
  L0-h 的门/掩码单点门槛 `1e-5` 对应实测 `1.1e-7`（2 个数量级余量）✓。

### 7. 完整性与一致性 — pass

- 章节齐全：§0（0.1–0.7）+ §1–§13；占位符扫描（`待补充`/`待填写`/`TODO`/`FIXME`/`{中文}` 模板槽）**0 命中**。
- 同类引用真实：`14_two_loop_handshake.py`、`13_l0c_fp16_scratch_cann9.py`、`10_feedback_and_acc_cann9.py`、
  `_gqa_prefill_fwd_kernel.py`、`_argreduce_kernel.py`、`gdn_prefill.py`、`utils/ascend_mode.py`、`CG-2026-0004/0011`、
  `VP-2026-0087`、`VP-2026-0097`、PL-1.11/PL-1.14 与 constants 条目**均真实存在**（`design_lint` 的路径存在性检查也全过）。
- **内部一致（上轮 fail 点）已修**：边界语义全文单一化（见阻塞项 2 的一致性判定）；落盘覆盖与 flag 配对表述一致（见阻塞项 1）。
- 其余一致：API 映射 / 内存规划 / Tiling 在 tile 尺寸、slot 数、scratch 清单、`A_pad` 3D 形态上相互一致；
  `A_pad[S_pad,head,64] → A_pad[:T].unsqueeze(0)` 给出 `[1,T,head,64]` **连续**视图，`is_contiguous()` 成立 ✓。
- 仅剩 3 处**精度类**小瑕疵（行号归属、op 计数、FLOPs 口径）→ 见建议 1/2/3，不影响任何结论。

### 8. 算法优化分析 — pass

- **§1.6 章节存在 — pass**：1.6.0/1.6.1/1.6.2/1.6.3 四子节齐全且均有实质内容。
- **调研四问完整性 — pass**：R1 候选表含基线 B0 与 R1-a..h；R2 明确"无在线变体"并给 3 条结构依据；R3 表 7 候选 + **四口径**
  （FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）+ 可并行度 + 常数版本戳；R4 对 6 候选逐项 6 维打分。
- **调研结论正确性（独立复核）— pass**：
  - **等价性复核**：C1 的 `Π(I+N^{2^i})` 我独立推演成立；`T←T+P·T`、`P←P²` 与 `initC=False` 的 `dst=A@B+dst` 语义一致
    （`T.gemm.md` §2.1），且伪代码的 slot 轮转（`ws_T[lev%2]` 读 / `ws_T[(lev+1)%2]` 写、`P` 同）逐级对齐，**6 级 = 12 个链 gemm + 1 个播种**。
  - **负向断言复核**：`algorithm-candidates.md` 的 `ALG-triangular` 行 R2 = "—"，与"本算子无在线变体"一致；
    Neumann/Newton/分层/纯 Vector 候选全部在表，未发现被漏掉的已知候选；R1-g/R1-h 已落笔并各给弃选依据。
  - **复杂度复算**：`S=32768 → 512×16 = 8192 任务 / 20 核 = 410 / ×9.25 µs = 3.79 ms` ✓；B0 `16+10 = 26`（dim_k=128 为 36）✓；
    `S=3000 → 38×9.25 = 0.35 ms`、`S=4096 → 52×9.25 = 0.48 ms`、`S=12288 → 154×9.25 = 1.42 ms` ✓；
    访存 440.4 KB 相加式我逐项复算 `16+0.4+32+32+176+160+16+8 = 440.4` ✓，流量 6.1 µs < Cube 发射 9.25 µs，"Cube 主导"结论不变。
  - **affinity / 弃选论证核对**：L2（fp16 链）、L3（双 head 打包）、L4（head-major + host permute）、L5（逐行写）的弃选前提我都打开
    `docs/` 或 pattern-library 核对过：`T.transpose` 是核内 UB 级二轴交换（`T.transpose.md` §2.1/§2.3，bf16 不支持）、
    `store_fixpipe` 只落 GM（§2.1）、L4 的 128 MB 拷贝代价按 1.73 TB/s 折算 ≈74 µs 与设计一致；未发现与文档/实测矛盾之处。
  - **口径自洽**：16³ 与 64³ 统一 0.25 µs/op 已**显式标为未实测假设**；Cube 0.25（本任务实测）/ Vector 0.5（`CONST-vector-launch-overhead`）分离 ✓。
- **调研→设计一致性 — pass**：§1.4 / §3.1 / §3.3 / §1.6.2 V2 四处同源（C1、`N=−S_sys`、唯一负号落点 `vmul(β,−1)`、`mask01`）；
  §0.5 对 O6/O7/O8 的替换意图逐项写明，无静默丢弃。
- **数学等价优化完整性 — pass**：A1–A7 每项都有「原式 → 优化后 → 等价论证 → 收益量化」；A2 明标备选、A6/A7 明标"护栏而非新优化"。
- **等价性论证正确性 — pass（机检 + 独立推演）**：A1/A2 恒等 ✓；A3 仅差 1 次 fp16 舍入（3.7e-4 与 fp16 1 ulp 自洽；
  ulp 3957 因以 fp32 尺度度量 fp16 量化量，viol=0 才是判据）✓；A4/A5 见机检；**A6/A7 把"只在可执行层出现"的方向/符号/越界缺陷纳入机检，
  且负向对照 viol 3.7e-1 / 4.4e-1 / 4096-NaN 全部按预测 fail** ⇒ 判据有牙齿。
- **向量化替代完整性 — pass**：V1–V11 与 §3.3/§6 的循环/标量点一一对上（任务映射、N 构造、门表/除法、16 步行串行、16 次块回代、
  逐行写回、越界 clamp、6 级 j 循环、掩码、Gram K 规约、β 提升），无遗漏；V9 的 `arange+vcmp` 弃用有探针实测支撑。
- **不可替代理由充分性 — pass**：V1（block 索引/元数据）、V9（常量 tile，每 block 一次）理由具体。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 按 Cube/MixCV 类清单枚举 L0（原生长宽 + 行主序 + 连续中转）/L1（显式转置）/L2（fp16 链）/
  L3（双 head 块对角）/L4（head-major + host 重排）/L5（逐行写），逐候选有整除性/代价/预算量化；**L4 的弃选前提已改正**
  （"单 token 连续 ≠ 64 token 块连续，行距 `head·64`"）；落盘代价有真机实测支撑。
- **布局→实现一致 — pass**：64×64 fp32 行主序、UB 预算按该布局算、`A_pad` 3D + strided region 写与 §3.3 的 `scatter()` 一致。
- **与下游章节一致 — pass**：§3.1 以 §1.6.1 的优化后公式（含 `min(d,0)`、`−β`、`mask01`）为输入；§6 未复活已判死的逐元素标量循环。

## 上一轮 11 条建议级问题的处置核对

| 上轮建议 | 设计处置 | 我的核对 | 判定 |
|---|---|---|---|
| 3. 掩码微探针表述与实测不符 | 改为"MA/MC 静默全 0；**MD 是 0.5 倍掩码（探针自身少乘 2，不作依据）**；只有 MB 精确" | §1.6.2 V9 / §3.4 / §11 三处文字一致；我复跑得 `MA 0 / MC 0 / MD 1008(0.5) / MB 2016(0)` | **已修** |
| 4. `y_blk` 记 1.28 MB | 改 **0.32 MB**（`20×2×64×64×2 B = 327 680 B`） | §4.1 / §4.5 / R10 三处一致，GM 合计同步为 `2.56 + 0.32 + A_pad + 48 KB` | **已修** |
| 5. §5.3 分核举例算术错 | 改 `4×52 + 16×51 = 1024`，并补 `T=752` 同类推导 | 两例我复算通过；与仿真 D 列一致 | **已修** |
| 6. R3 "访存 425 KB" 无相加式 | 改 **440.4 KB** 并给全相加式；roofline 5.9 → 6.1 µs；声明"含输入读" | 相加式逐项复算吻合 | **已修** |
| 7. 源码行号漂移 | 改为 L173–L176 / L257–L271 | `P0` 的 L257–L271 ✓；**三处硬校验的归属仍错位**（详见建议 1） | **基本修，残留** |
| 8. "所有 in-scope 形状 T ≥ 752" 矛盾 | 改为"**真实业务形状** T ≥ 752；小形状由 `K=min(20,T)` 退化" | §5.3 文字已改，与 L0/L1 用例不再矛盾 | **已修** |
| 9. "alias 表"名不副实 | 正名为"UB 原地复用 / 生命周期表"，明确"7 个独立 buffer + 8 KB + 5 小 buffer = 121.1 KB，不需要 alias" | §4.3 + R13 一致；预算 121.1/168 KB 两档自洽 | **已修** |
| 10. A1 "每元素 2 exp + 1 div" 与源不符 | 原式栏改为"4 组 16 行块参考门表（4×64 次 exp，存 dtype）+ 每元素 1 次除法（2016 次）" | 与源码 L391–L425 / L429–L505 一致 | **已修** |
| 11. L0 缺"每 block 任务数 < NPER"用例 | 新增 **L0-i**（`S=1600,head=1`，`T=25,K=20,NPER=2`，15 个 block 各 1 任务） | §8.3 L0-i + §7.7 表 E 对齐；判据 = 25 个 chunk 全逐元素正确 + `last_it != NPER-1` 分支 | **已修** |
| （维度 5 扣分）flag id 计数语义未载明 | §7.5 显式标为未文档化假设 + 支撑证据 + 回退 | 与 `14_two_loop_handshake.py`、L1-a 交叉引用齐全 | **已修** |
| （维度 1 扣分）`(64,1)→(1,64)` 落位"文档未载" | §3.2 / §3.4 双处注明"文档未载、探针定案" | 真机 `1.4e-6 / 1.07e14` 我本轮复现 | **已修** |

## 建议级问题清单（非阻塞，不影响结论）

1. **§0.2 三处硬校验的源码行号归属仍错位（上轮建议 7 的残留）**。设计写 `head % key_head`（L173–L174）、
   `chunk_size/dim_k`（L173–L176）、`uniform_seq_len`（L175–L176）；实测源码为 `chunk_size/dim_k` **L173–L174**、
   `head % key_head` **L175–L176**、`uniform_seq_len * num_sequences != total_tokens` **L177–L181**。
   另 §13.3 的"L191"不对应任何校验行（L191 在 tiling 常量区）。
   **修法**：把 §0.2 三处改为 `L173–L174 / L175–L176 / L177–L181`，§13.3 删去 "L191"（或改为 L173–L181）。
2. **C1 的 Cube 侧 op 计数少 1 个（§1.6.0 R3 / roofline / §5.2 / §13.3）**。按 §3.3 逐语句数：
   13 个 gemm + 25 个搬运（`k` 1、`gs_G` 1、`ws_S` 1、`ws_T` 11、`ws_P` 10、**`l0t→y_blk` 1**）= **38**，设计记 37（24 搬运），
   故总 op 应为 **57** 而非 56。影响：Cube 发射项 9.25 → **9.50 µs**（+2.7%），四形状合计 0.35/0.49/1.46/3.89 ms；
   **"Cube 主导（9.50 vs Vector 7.50）"与 R1 的 1.2× 阈值结论不变**。
   **修法**：把 Cube 搬运改 25、总数改 57（Cube 38 + Vector 19），或显式声明"`l0t→y_blk` 的 fixpipe 计入 Vector 侧 8 个拷贝"
   并同步 §5.2 的 `37`。
3. **C1 的 FLOPs 口径差 ≤6%（§1.6.0 R3 / §0.6 R1 的 "3.47 M MAC"）**。按 13 个 gemm 复算：
   `12×64³ + 1×(64×64×dim_k)` = **3.41 M MAC（dim_k=64）/ 3.67 M（dim_k=128）** ⇒ **6.8 / 7.34 MFLOP**，设计记 6.94 M（MAC 3.47 M）。
   **修法**：在 R3 表下补一行相加式并取 `6.94 M（dim_k=64 口径，dim_k=128 时 7.34 M）`；结论（MAC 非瓶颈、op 数主导）不变。
4. **strided 落盘时延是单次采样值**。我本轮实测 `contig 0.090 / strided 0.233 µs（2.59×）`，上轮 `0.080/0.206（2.58×）`，
   设计记 `0.084/0.194（2.29×）`。三次采样比值稳定在 2.3–2.6×，绝对值波动 ±20%。
   **修法**：§1.6.3 / §11 记成区间（`strided ≈ 0.19–0.23 µs / 8 KB`）并注明 run-to-run；结论 `+0.1~0.3 µs/任务` 不变（R14 已把 `head=16` 列为待测）。
5. **`ub_gr` 的"动态 extent 转置落位"未被任何探针覆盖**。`design_probe_gate_sign.py` 只验了 `valid=64` 的
   `(64,1)→(1,64)` 静态形态；§3.3 的 `T.copy(g[0,b:b+vv,hd:hd+1], ub_gr[0:1,0:vv])` 在 `vv<64` 时是**动态 extent + 跨步转置**，
   R4 只把它笼统列为"动态 extent 风险"。**修法**：在 L0-d（`valid=36`）用已有的 L0-h dump 通道加一条精确断言
   `ub_gr[0,0:36] == g[b:b+36,hid]`（以及 `ub_gc[0:36,0]` 同断言），把该形态从"编译/数值失败"提前到"逐元素断言"。
6. **R3-a/R3-b 的同管道顺序与 flag 计数语义仍是文档级论证**（§13.4 已披露，L0-d/L0-i/L1-a 是首测点）。
   **修法（Stage 3 判据补强）**：在 L0-h 的 dump 里增加"越界行 `ub_b16[valid:64] ≡ 0`、`ub_bn[valid:64] ≡ 0`"的设备侧断言，
   使 §0.6 引理 L-3 的**前件**在真机可观测，而不是只由最终数值反推；顺序假设被推翻时直接按 R4 回退
   （`T.clear` + `T.pipe_barrier`）。

## 最终判定

- **是否仍有阻塞级问题：无**。上轮 2 个阻塞项（每 block 末任务漏落盘、尾 chunk 未初始化 SRAM 的 NaN 回灌）均已用
  "可执行修法 + 引理 + 机械复核/负向对照"闭环，且我本轮用独立实现（`T=1..20000`、自写 slot 校验）与真机探针复跑复核，
  未发现新的阻塞级缺陷。
- **建议级问题：6 条**（上表 1–6），全部为计数/行号精度、探针覆盖补强与 Stage 3 观测点，均不改变任何设计决策与结论。
- **是否建议进入 Stage 3：是**。进入前建议（非强制）顺手改掉建议 1/2/3 的三处数字（纯文档改动，<5 分钟），
  并把建议 5/6 的两条断言写进 §8.3 的 L0-d / L0-h 判据。
- Stage 3 的**硬前置**（据 §9/§13.4，非本报告新增要求）：L0-a 先验 6 级链 + 每级 store-then-reuse L0C（R2/R3）；
  L0-d/L0-i 验 R3-a/R3-b 与落盘 exactly-once；L1-a 验 4 轮 flag 复用；L0-g 验 `head=16` 的 2048 B 行距落盘；
  UB 实际字节数以编译报错文本复核（R13）。

## 门禁

`python3 .agents/tools/statectl.py gate 2 --dir examples/gdn_prefill/prefill_blocksolve_A_bthd`：

```json
{"action": "gate", "failures": [], "ok": true, "pass": true, "stage": 2, "warnings": []}
```

---

（本检视只写 `REVIEW.md`；未改动 `DESIGN.md`、kernel、探针与任何状态文件。所有机检与真机数值为本轮独立复跑。）
