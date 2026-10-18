# prefill_blocksolve_A_bthd 设计文档检视报告（修订版复检 / retry_count=1）

## 检视结论

结论: 不通过

## 检视元信息

- 文档: `examples/gdn_prefill/prefill_blocksolve_A_bthd/DESIGN.md`（revision 1 / v1，1227 行）
- 上一版: `examples/gdn_prefill/prefill_blocksolve_A_bthd/history_version/design_v0.md`（v0）
- 任务类型: 迁移（GDN prefill 流水线 函数 2/6）
- 源算子: `/mnt/workspace/TileOPs-main/src/tileops/kernels/linear_attention/gdn/prefill_prepare.py`
  `_prefill_blocksolve_A_bthd_tl`（L158–L651）+ wrapper `prefill_blocksolve_A_bthd`（L654–L708）
- 检视维度: 迁移任务 9 项（0 源算子理解与迁移分析 / 1 API / 2 内存层级 / 3 Tiling / 4 技术约束 /
  5 循环与同步 / 6 验证方案 / 7 完整性与一致性 / 8 算法优化分析）
- 本轮性质: 设计修订循环第 1 次（v0 的 4 个阻塞项逐条复检）
- 阻塞级问题数: 2
- 建议级问题数: 11

## 机械复核（D-1/D-5）

### verify_equiv 重跑：PASS（A1–A6 全 EQUIV_PASS，与 §1.6.1 内嵌表逐格一致）

实跑 `python3 examples/gdn_prefill/prefill_blocksolve_A_bthd/verify_equiv.py`（2026-10-17，torch CPU，fp64 参照），
输出与 DESIGN.md L440–L449 的 8 case × 6 项表、L451–L458 的汇总表**逐项相同**：

| 项 | worst rel | worst ulp | worst viol | 判定 |
|---|---|---|---|---|
| A1 | 3.0e-08 | 0.5 | 0.0e+00 | EQUIV_PASS |
| A2 | 1.2e-07 | 1.3 | 0.0e+00 | EQUIV_PASS |
| A3 | 3.7e-04 | 3957.5 | 0.0e+00 | EQUIV_PASS |
| A4 | 4.0e-06 | 1.8 | 0.0e+00 | EQUIV_PASS |
| A5 | 9.3e-05 | 1.0 | 0.0e+00 | EQUIV_PASS |
| A6 | 4.6e-04 | 310.8 | 0.0e+00 | EQUIV_PASS |

- **A6 反例对照（判据有牙齿）**：`as-designed (g_i-g_j, -beta)` viol 0；`v0 direction` rel 1.3e-1 / viol 3.7e-1；
  `v0 sign` rel 4.4e-1 / viol 4.4e-1。v0 的两个可执行层缺陷现已进入机检并确实被判 fail。
- **符号裁决（fp64 结构通路）**：`|source - inv(I+S)| = 2.776e-17`、`|source - inv(I-S)| = 4.361e-01`
  ⇒ `N = -S_sys` **CONFIRMED**（与 §0.1/§1.3 一致）。v0 的 fp64 通道 bug 已修：
  `blocked_source(..., store_dtype=torch.float64)` 末行 sanity 打印 **2.7756e-17**（v0 为 2.3e-8），
  与 §8.1 的 1e-10 门槛自洽。
- 信息性通道：`blocked_source` fp16 部分存储 `SC-typical` rel 7.0e-5、`SC-beta-x20` nan/viol 8.5e-3，
  与 §1.6.1 注、§8.1、§11 的"源 fp16 溢出边界"叙述一致。

### design_calc_check：4 项 checks（skip 项已人工复核）

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

- `ub_budget` **skip → 人工复核通过**：§4.5 手工峰值 = 7×16 KB（`MK/UB_G/UB_A/UB_B/UB_E/UB_F/UB_N`）
  + 8 KB（`UB_Y`）+ ≈1.15 KB（`ub_gc/ub_gr/ub_b16/ub_b32/ub_bn`）≈ **122 KB**；按 `CONST-capacity-910B2C`
  的 **Expert 显式 alloc 结构**系数 ×1.10–1.12 ⇒ 134–137 KB < 192 KB ✓。
  **注意**：机检脚本内 `UB_INFLATION = 1.7`（auto-multi-buffer 口径），按 1.7 算 122 KB → 207 KB 会"越界"；
  `constants.md` 2026-09-15/09-17 两个数据点明确"Expert 逐 buffer 显式分配 ≈ ×1.10–1.12（甚至实际低于手工）"，
  故设计的 ×1.12 口径**有据**。Stage 3 仍须以编译器报的实际 UB 字节数复核（见 §9 R13）。
- `l0c_budget` **pass 但属假阳性**（解析到 "16x16 x fp32 = 1.0 KB"）→ 人工复算 `l0g/l0t/l0p = 3×16 KB = 48 KB` < 128 KB ✓。
- `core_split` **skip → 人工复核通过**：`T = nch_total×head = 752/1024/3072/8192`（§5.1 复算
  `ceil(3000/64)=47→752`、`512×16=8192` 全对），`K=min(20,T)`、`NPER=ceil(T/K)` 为编译期常量，
  跨步映射 `t=bid+it·K` 负载差 ≤ 1 ✓（但 §5.3 的举例算术有误，见建议 3）。
- `r3_metrics` **pass**（v0 为 fail）：§1.6.0 R3 表已含 `FLOPs` / `访存 Bytes` / `扫描遍数` / `中间缓冲峰值` 四口径。

### 真机探针复跑（本检视独立执行；910B3 chip6 / CANN 9.0.0 / Expert，`_probe_env.sh` 环境）

| 探针 | 复跑结果 | 与 DESIGN 记载对比 |
|---|---|---|
| `design_probe_store_layout.py` | `PROBE-STORE-STRIDED` 目标行区 `2.419e-04`（= fp16 自身舍入）、相邻槽 `max|other| = 0.000e+00`、fixpipe 量化 `0.000e+00`；`contig 0.080 µs/8 KB`、`strided 0.206 µs/8 KB`（2.58×） | 设计记 `2.419e-04` / `0` / `0` ✓；时延记 `0.084 / 0.194（2.29×）`，**同量级、比值差 12%（run-to-run 噪声）**，结论（≈ +0.3 µs/任务）不变 |
| `design_probe_gate_sign.py` | `e vs exp(g_i-g_j) = 1.412e-06`、`vs exp(g_j-g_i) = 1.067e+14`；`N vs -β·e·G = 1.118e-07（|ref|max 0.126）`、`vs +β = 2.519e-01`；`PROBE-CHAIN fp16 rel 2.783e-05 / fp32 rel 5.547e-09` | 与 §3.4/§11 的 `1.4e-6 / 1.07e14 / 1.1e-7 / 2.5e-1 / 2.8e-5 / 5.5e-9` **逐项复现** |
| `design_probe_mask.py` | `MA 1.0e+00 sum=0`、`MC 1.0e+00 sum=0`、`MD 5.0e-01 sum=1008`、`MB 0.0e+00 sum=2016` | MA/MC/MB 与设计一致；**MD 不是"0 误差"而是 0.5 偏差**（见建议 1） |
| `design_probe_vcast_shapes.py` | M1–M5 全 `max_abs_err = 0.000e+00` | 与 §3.4/§11 一致 |
| `design_probe_doubling_cube.py` | `T1 2.922e-08`、`T2 2.699e-07`、`per 6-op level = 1.492 µs` | 设计记 `2.9e-8 / 2.7e-7 / 1.52 µs` ✓（0.249 µs/op） |
| `NPUUtils.get().get_aicore_num()` | **20** | 与 §5.2/§11 一致 ✓ |
| 独立脚本（本检视自写，非设计工件） | 按 §3.3 的任务映射仿真落盘覆盖：`T=8192 → 漏 8 个任务（8172–8179）`、`T=1024 → 漏 16 个（1004–1019）`、`T=752 → 漏 8 个（732–739）`、`T=64 → 漏 16 个（44–59）`、`T=1 → 0` | 见阻塞问题 1（本检视的判定依据） |

> 真机可用，未降级为"未复现"。上表所有数值均为本轮实跑输出。

## 检视详情

### 0. 源算子理解与迁移分析 — warn

逐项亲自对照源码（`prefill_prepare.py` L158–L708）核对 §0。

- **语义理解正确性 — pass（含数值复证）**。§0.1 的 `S_sys[i,j]=β_i·exp(g_i−g_j)·(k_i·k_j) (i>j)`、
  `A=(I+S_sys)^{-1}`、`g` fp32、`accum_dtype="float32"`、`solve_dtype=dtype`、`β` 先 cast 到 dtype
  与源码逐句一致（`A=torch.empty(...,dtype=k.dtype)` L688；`g: T.Tensor(...,"float32")` L216；
  `accum_dtype` L199；`solve_dtype=dtype` L200；`i_s=alloc_shared(...,solve_dtype)` L242；
  `beta_s[t]=T.cast(step,dtype)` L295–L297）。**符号归属**经我按源码 L429–L618 复算确认：
  对角块 `a_s=−G·s`（L480–L499）、行串行 `X = I + aX`（L519–L540）⇒ `X=(I−a)^{-1}=(I+S)^{-1}`，
  `verify_equiv` 的 fp64 结构通路给出 `2.8e-17` vs `4.4e-1`，符号唯一 ✓。
- **I/O 与签名 — pass**。`k/g/beta/A/k_rnorm` 的形状与 dtype 对应 L215–L220、L686–L691 ✓；
  `group=head//key_head`、`khid=hid//group` 对应 L186/L270 ✓；`k_rnorm` 仅 `l2norm` 写 ✓。
- **算法解读完整性 — pass**。P0–P9 与源码语句一一对应；P7 的 6 条块回代公式与 L543–L618 逐式核对一致；
  **gemm 计数已改正**为 16 次块回代（L543/548/555/559/564/569/574/579/583/587/592/597/601/606/611/616，
  我逐行点过）+ 10 次 Gram ⇒ 26（`dim_k=128` 时 36）✓；host 侧 `T.min(cid,last)` 的"幂等重算、非早退"
  与 L263–L266 注释一致 ✓。
- **优化手段覆盖完整性 — pass**。O1–O12 齐全且逐项标注硬件依赖；v0 的两处动机误判已修（O12 改为
  clamp 幂等重算；O5 改为"门表复用 4096→4×64 次 exp + 除法复用"并注明"全差式数值不更差"）✓。
- **耦合性判定合理性 — pass**。抽检 O1（`store_fixpipe` 只能落 GM）、O6（`T.get_thread_binding` 在
  `docs/Tilelang.language/` 全目录检索不到）、O7（op 数主导）、O9/O12（语义保留/等价替换）、
  二维 grid → 一维任务号，均与 `docs/`、`ascend-constraints.md`、`EXPERT_FINDINGS.md` §12 不矛盾 ✓。
- **NPU 重设计可行性 — warn**。§0.6 R1 的恒等式已改正为 `N ≜ -S_sys`、`A=ΣN^j=(I−N)^{-1}=(I+S_sys)^{-1}`
  ✓；`N^64=0` 与 `{1,2,4,8,16,32}` 覆盖 `0..63` 我独立复算成立 ✓；R3 的"行独立传播"数学成立 ✓。
  **但 R1 段末的边界语义"越界行 `β=0 ⇒ N 的该行为 0 ⇒ A 的该行为 e_i`"在本设计的可执行路径里没有落点**：
  §0.6 R3 自己写的是"动态 extent 只搬合法行，越界行留在 L1/UB 的是**上一个任务的旧值**"，
  §3.3 也没有对越界行置 `β=0`（`T.copy(beta[...vv...], ub_b16[0:vv,0:1])` 之后无任何掩码）。
  两种描述互斥，且"旧值有限"在 **block 的第一个任务**上不成立（见阻塞问题 2）。
- **设计与决策一致性 — warn**。§1–§7 确是 §0.5/§0.6 决策后的算法（无"判重设计、照抄源方案"脱节）✓；
  唯一不一致即上一条的边界语义。
- **golden 独立性 — pass**。§8.1 以 §0.1 语义独立实现（`torch.linalg.solve(I+S)`，fp64），
  未被 §0.6 的 doubling 污染；`blocked_source` 交叉校验用的是**源**分块算法，也不是本设计算法 ✓。

### 1. API 可行性 — pass

- **API 存在性与模式匹配**：`T.Scope/alloc_L1/alloc_L0C/alloc_ub/rs/gemm(size,initC,b_transpose)/copy/
  vbrc/vsub/vmin/vexp/vmul/vcast(round_mode=)/sync_block_set|wait` 均在 `docs/Tilelang.language/` 与
  `examples/gdn_prefill/probe/experiments/*` 有佐证；Expert 手动同步、未用 `threads=` ✓；v-prefix 优先 ✓。
- **本次修订涉及的 4 个"易错 API 语义"全部有真机或前端证据**：
  1. `T.copy` 的 `(64,1)→(1,64)` 落位（`g_j` 的来源）：docs `T.copy.md` §2.2.2 规则 1 只讲"静态 1 维折叠"，
     该形态实质是跨步转置写，**文档未明示**；本设计用 `design_probe_gate_sign.py` 真机对拍
     （`e` 与 `exp(g_i−g_j)` 差 1.4e-6、与反方向差 1.07e14）定案 ⇒ 可用 ✓（建议在 §3.2 注明"文档未载、探针定案"）。
  2. `T.store_fixpipe`/`T.copy`(L0C→GM) 的 region 语义：§3.1 S7 已改为"连续中转 `y_blk` + Vector 侧一次
     strided region 写"，`T.copy.md` §2.3 第 3 条给出 f32→f16/bf16 量化配对；真机 `design_probe_store_layout.py`
     用**与 §3.3 完全相同的 region 形态**（`A3[0:C, HID:HID+1, 0:C]`）验证目标行区正确、相邻槽零污染 ✓。
  3. `T.vcast` 第 3 位置参数是 `size`：一律 `round_mode="rint"` 关键字，M1/M2/M5 实测 err 0 ✓。
  4. `T.vbrc` 的 `(64,1)→(M,N)` 是"每行同值 = g_i"、`(1,64)→(M,N)` 是"每列同值 = g_j"：
     `T.vbrc.md` §2.2.2 + `VP-2026-0087`（`vmul/vsub/vadd` 无 `[M,N]*[1,N]` 形态、`M==N` 时反向静默错）
     共同支撑"两个操作数都先 vbrc 到全形再相减"的做法 ✓。
- **flag id**：`F_G/F_S/F_A + slot`（6 个 ≤ 15）以常量基址 + TIR 表达式表达，先例
  `_gqa_prefill_fwd_kernel.py:351` 的 `sync_block_set(i)`；§3.3 自查段给了"按 2 展开成常量 id"的回退 ✓。
- 未使用 `out_idx` 回喂（全部独立 scratch 输入张量），与 `EXPERT_FINDINGS.md` §12.1 K2 一致 ✓。

### 2. 内存层级规划 — warn

- **搬运路径完整**：§4.4 的 `GM(k)→L1(MTE2/NZ)→L0A/B→L0C→GM(gs_G)→UB→Vector→GM(ws_S)→L1→L0C(6 级)→
  fixpipe→GM(y_blk)→UB(MTE2)→GM(A_pad, strided)` 无越级路径（无 GM→L0、无 L1↔L1）✓。
- **UB 预算**：122 KB（7×16 KB + 8 KB + 小 buffer），×1.10–1.12 ⇒ 134–137 KB < 192 KB ✓（机检 skip，人工已核）。
- **L0C**：3×16 KB = 48 KB < 128 KB ✓（机检 pass 属假阳性，人工已复算）。
- **L1**：`l1k`(16 KB, dim_k=128) + `l1e/l1p/l1t`(3×16 KB) = **64 KB** < 512 KB ✓（`l1q` 删除后与 §3.3 一致）。
- **Buffer 层级标注**：§4.3/§3.3 每个 buffer 均标 L1/L0C/UB/GM，且 `ub_gc(64,1)/ub_gr(1,64)/ub_b16(64,1)`
  与 `vbrc` 的 rank 断言一致 ✓（v0 问题 8 已修）。
- **扣分项（建议级）**：§4.1/§4.5 把 `y_blk[K,2,64,64]` 的 fp16 工作区记为 **1.28 MB**，
  实算 `20×2×64×64×2 B = 327 680 B = 320 KB`（偏大 4×，对 64 GB HBM 无风险，但属"计数可追溯"纪律的漏项）。

### 3. Tiling 策略 — warn

- **分核三要素齐全**（`core-split-strategy.md` §1）：① 逻辑任务数 `T=nch_total×head` 并给 5 个形状表
  （1/752/1024/3072/8192，我复算全对）；② 物理核数**实查** `get_aicore_num()=20`（本轮复跑 = 20），
  并按 MixCV/Cube 口径不乘 2 的论证成立；③ 规模判定 + `K=min(20,T)` + `NPER=ceil(T/K)` + `t=bid+it·K`，
  跨步映射负载差 ≤ 1、无跨核同步、无 `A_pad` 写冲突 ✓。
- **block/对齐**：任务粒度 1 chunk×1 head（64 行）；Gram `[64,dim_k,64]`、链 `64×64×64`、K∈{64,128}
  均为分形 16 整数倍，无尾块；行方向尾块由"动态 extent + 掩码 + padding"处理 ✓。
- **扣分项（建议级，不影响分核方案）**：
  1. §5.3 例"`T=1024` → 12 个核 52 个、8 个核 51 个"算术不成立（`12×52+8×51=1032≠1024`），
     正确为 `4×52+16×51=1024`；
  2. §5.3 开头"所有 in-scope 形状 `T ≥ 752`"与 §5.1 自己的 `L0: T=1`、`L1-a: T=64` 矛盾
     （`K=min(20,T)` 已覆盖，属表述问题）。

### 4. 技术约束检测 — pass

三维 Kernel 不涉及（一维 `T.Kernel(K)`）✓；`threads=` 不使用 ✓；L0C 不溢出（48 KB）✓；
GEMM 无非整除问题（分形全对齐）✓；单 Cube scope + 单 Vector scope ✓；不使用 `out_idx` 回喂 ✓；
`T.gemm` 的 dst 一律是**整个** L0C buffer、不用 region dst ✓（`CG-2026-0011`）；GM→L1 跨步装载用
`T.copy` **slice** 形态而非 `load_nd2nz`，正确引用 `CG-2026-0004` / `TRAP-load-nd2nz-strided` ✓
（我打开该条目核对：`T.copy` base+size 形态同样错、**slice 形态 bit-exact**，设计引用准确）。

### 5. 循环与同步 — fail（阻塞）

- **循环结构**：任务循环 `T.serial(NPER)`（静态上界）✓；6 级链 Python 静态展开、理由充分 ✓；
  `T.Parallel/T.Pipelined/T.Persistent` 不使用均有依据 ✓；Gram 单次 gemm 不引 K 段循环 ✓。
- **同步与模式匹配**：6 个 flag（≤15）✓；set 侧 `PIPE_FIX`/`PIPE_MTE3`、wait 侧 `PIPE_MTE2` 与
  `14_two_loop_handshake.py` 形态一致 ✓；Cube 领先一拍取 Gram、Vector 落后一拍落盘的依赖图无环，
  我独立走查 `it`/`it-1`/`it-2` 三个视角，`gs_G`、`ws_S` 的 WAR 结论成立 ✓。
- **fail 项**：**"一任务滞后落盘"的尾部覆盖有洞**——`scatter` 只出现在 `if t < TASKS` 之内，
  循环后的 epilogue 只补"全局最后一个任务"；任何"任务数 < NPER"的 block 的**最后一个任务永远不会落盘**
  （每形状漏 8~16 个任务，`A` 对应行保持 `torch.empty` 垃圾）。详见阻塞问题 1。
- **扣分项**：§7.7 的"flag 计数严格配对"结论因此不成立；§7.6 的 `y_blk` WAR 论证依赖"flag id 复用为
  计数/FIFO 语义"，`docs/同步管道操作/*` 未载明该语义（只给单次 set/wait 示例与 `FFTS_FLAG_THRESHOLD`
  循环），目前仅由 `14` 的 2 轮实验佐证——这一条本身可接受（L1-a 有 4 轮覆盖），但应在 §7 显式标注为假设。

### 6. 验证方案 — warn

- **Golden — pass**：§8.1 纯 torch CPU fp64、按 §0.1 语义实现（`torch.linalg.solve(I+S[:valid,:valid])`），
  未被 NPU 算法污染；`rows=clamp(arange(base,base+64), max=end−1)` + `bb[valid:]=0` 精确对应源码 clamp/β=0；
  只写 `valid` 行 ✓。
- **交叉校验 — pass（v0 问题 6 已闭环）**：`blocked_source(..., store_dtype=torch.float64)` 保留 fp64，
  实测 **2.7756e-17**，与 §8.1 的 1e-10 门槛自洽；fp16 通道单列为"量化层（信息性）"✓。
- **L0/L1/L2 计划 — warn**：L0-a..h、L1-a/b、L2（12 组合）、Boundary（56/64/65）分层清晰，门槛明确；
  并新增 L0-g（head=16 行距 2048 B）与 L0-h（门/符号单点）覆盖本轮修订的两处要害 ✓。
  **但 L0 层没有任何"每 block 任务数 < NPER"的用例**：`T=1/2` 时 `NPER=1`，阻塞问题 1 在 L0 上不可见
  （要到 L1-a `T=64,NPER=4` 才暴露）。建议补一个 `S=640, head=1`（T=10, K=10）或 `S=1280, head=1`
  的低成本 L0 用例，把该形态的落盘覆盖提前到 L0。
- **精度标准 — pass**：fp16 `5e-3`（abs+rel）、bf16 `4e-2`，均严于 skill 默认且给了余量推导 ✓。

### 7. 完整性与一致性 — fail（阻塞）

- **章节齐全 — pass**：§0（0.1–0.7）+ §1–§12 齐全；我对全文做了占位符扫描（空占位、待补项、未填模板槽），0 命中。
- **同类引用真实 — pass**：`14_two_loop_handshake.py`、`13_l0c_fp16_scratch_cann9.py`、
  `10_feedback_and_acc_cann9.py`、`_gqa_prefill_fwd_kernel.py`、`_argreduce_kernel.py`、`gdn_prefill.py`、
  `utils/ascend_mode.py`、`CG-2026-0004/0011`、`VP-2026-0087` 与 pattern-library 常数条目均真实存在 ✓。
- **内部一致 — fail（阻塞）**：
  1. §0.6 R1"越界行 β=0 ⇒ 行 = e_i"与 §0.6 R3/§3.3"越界行留旧值、靠 0×有限=0 保丢弃元素为 0"互斥；
     后者在 block 首任务（恰为尾 chunk）时不成立 ⇒ 合法行可能出现 NaN（阻塞问题 2）。
  2. §3.3/§7.7 的落盘覆盖与"flag 计数严格配对"表述不成立（阻塞问题 1）。
- **其余内部一致 — pass**：API 映射/内存规划/Tiling 在 tile 尺寸、slot 数、scratch 清单
  （`gs_G/ws_S/ws_T/ws_P/y_blk`）、`A_pad` 3D 形态上相互一致 ✓；`A_pad[S_pad,head,64] → A_pad[:T].unsqueeze(0)`
  给出 `[1,T,head,64]` **连续**视图，与源 `A` 契约（L219/L688）一致，`is_contiguous()` 成立 ✓。
- **扣分项（建议级）**：
  1. `y_blk` 工作区 1.28 MB → 应 0.32 MB（见维度 2）；
  2. §5.3 的 `12×52+8×51` 算术、§5.3 的"T≥752"表述（见维度 3）；
  3. §0.2 的两处硬校验行号 `L189–L192`/`L191` 应为 **L173–L176**；§0.3 P0 的 `L256–L271` 应为 `L257–L271`；
  4. §4.3 称为"alias 表"，实为**原地复用**（7 个独立 UB buffer，未做 buffer 别名），
     §9 R13 的"必须照表落 buffer"宜明确"目标就是 7 个独立 buffer + 原地复用"，避免 Stage 3 误加 alias。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性 — pass**：1.6.0/1.6.1/1.6.2/1.6.3 四子节齐全且均有实质内容 ✓。
- **调研四问完整性 — pass**：R1 候选表含基线 B0 与 R1-a..h（本轮补 R1-g/R1-h，对应 `ALG-triangular`
  命中行"直接求解替代求逆"/"对角三角降阶"）；R2 给"无在线变体"结论 + 3 条结构依据；R3 表 7 个候选
  + **四口径**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）+ 可并行度 + 常数来源与版本戳；
  R4 对 6 候选逐项 6 维打分 ✓。
- **调研结论正确性（独立复核）— pass**：
  - **负向断言复核**：`references/algorithm-candidates.md` 的 `ALG-triangular` 行 **R2 = "—"**（无在线变体），
    与设计一致；R1 的两条命中线索均已在 R1-g/R1-h 落笔；Neumann/Newton/分层/doubling/纯 Vector 全在表，
    未发现被漏掉的已知候选 ✓。
  - **等价性复核**：C1 的 `Π(I+N^{2^i})` 我独立推演成立（`N` 严格下三角 ⇒ 幂零指数 64；子集和唯一覆盖
    `0..63`；`T+=P·T`、`P=P²` 与 `initC=False` 的 `dst=A@B+dst` 语义一致，`T.gemm.md` §2.1 佐证）✓。
  - **复杂度复算**：`S=32768` → `512×16=8192` 任务、`8192/20=410`、`410×9.25 µs=3.79 ms` ✓（v0 的 6144 已改）；
    B0 **16+10=26**（`dim_k=128` 36）✓；C1 **12 链 gemm（1 播种 + 6 T + 5 P²）+ Gram = 13** ✓；
    op 总数 **37 Cube（13 gemm + 24 搬运）+ 18 Vector（11 宽 op + 7 拷贝）= 55** ✓（我逐项相加一致）；
    `S=3000→38×9.25=0.35 ms`、`S=4096→52×9.25=0.48 ms`、`S=12288→154×9.25=1.42 ms` 全对 ✓；
    16³ 与 64³ 同价 0.25 µs/op 已显式标注**未实测假设**，`CONST-vector-launch-overhead` 已改回 **0.5 µs/op** ✓。
  - **残留（建议级）**：C1 的"访存 425 KB"按 §4.4 的流量口径逐项相加为 **≈440 KB**（差 15 KB，疑漏计
    16 KB 的 `k` 读），对应 5.9 vs 6.1 µs，不影响"Cube 主导"的结论，但应给出相加式。
- **调研→设计一致性 — pass**：§1.4/§3.1/§3.3/§1.6.2 V2 与 §1.6.0 的 C1、`N=-S_sys`、单一负号落点
  （`vcast(β)` 后的 `vmul(-1)`）**四处同源** ✓（v0 的"调研选 A、设计做 B"已消除）；§0.5/§0.6 对 O6/O7/O8
  的替换意图逐项写明、无静默丢弃 ✓。
- **数学等价优化完整性 — pass**：A1–A6 每项都有「原式 → 优化后 → 等价论证 → 收益量化」四要素；
  A2 明确标为备选形态、A6 明确定位为"A4 输入构造"的护栏而非新优化 ✓。
- **等价性论证正确性 — pass（独立推演 + 机检一致）**：A1 恒等式 ✓；A2 `1/exp(x)=exp(−x)` ✓；
  A3 仅差 1 次 fp16 舍入（3.7e-4 对应 fp16 1 ulp，ulp 3957 因以 fp32 尺度度量，解释合理）✓；
  A4/A5 见机械复核 ✓；A6 把广播方向与符号纳入机检并给出两条反例（viol 3.7e-1 / 4.4e-1）✓。
- **向量化替代覆盖完整性（与 §3.3/§6 交叉核对）— pass**：V1–V11 与我扫 §3.3/§6 得到的循环/标量点一一对上
  （任务映射、N 构造、门表/除法、4 个对角块行串行、16 次块回代、逐行写回、越界 clamp、6 级 j 循环、
  掩码构造、Gram K 规约、β 提升），无遗漏；V6 已改为 3 步落盘的准确描述 ✓。
- **不可替代理由充分性 — pass**：V1（block 索引/元数据）、V9（常量 tile，每 block 一次）理由具体 ✓。
- **替代 API 佐证 — pass**：`vbrc/vsub/vmin/vexp/vmul/vcast/arange/transpose` 逐条在 `docs/` 找到佐证，
  且本轮四个微探针把"文档未载/文档与实现不一致"的点全部实测定案 ✓。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 按 Cube/MixCV 类清单枚举 L0（原生长宽 + 行主序 + 连续中转）/
  L1（显式转置）/ L2（fp16 链）/ L3（双 head 块对角）/ L4（head-major + host 重排）/ L5（逐行写），
  逐候选有整除性/代价/预算量化；**L4 的弃选前提已改正**（"单 token 连续 ≠ 64 token 块连续，行距 head·64"），
  L0 的落盘代价由真机实测支撑 ✓；未把 I/O 契约当内部布局依据 ✓。
- **布局决策与 §3.3/§4/§6 一致性 — pass**：64×64 fp32 行主序、UB 预算按该布局计算、`A_pad` 3D 与
  strided region 写与 §3.3 `scatter()` 一致 ✓。
- **与下游章节一致 — pass**：§3.1 以 §1.6.1 的优化后公式（含 `min(d,0)`、`-β`、`mask01`）为输入；
  §6 未再出现 §1.6.2 已判死的逐元素标量循环 ✓。

## 检视问题列表

### 问题 1（v1 新增，阻塞）: "一任务滞后落盘"漏掉每 block 的最后一个任务，8~16 个任务的 A 行永不写

- 严重程度: 阻塞
- 位置: DESIGN.md §3.3（L699–L729，`scatter` 在 `if t < TASKS` 内 + 尾部 epilogue）、§6（L955）、§7.7（L991–L992）
- 问题描述: Vector 侧的落盘是"滞后一拍"：任务 `t` 的 `y_blk → A_pad` 搬运发生在**下一次迭代**
  `it+1` 的 `scatter(t-K, ...)`；而该 `scatter` 位于 `if t < TASKS:` 的**内部**。因此：
  - block 的任务数 `n = NPER` 时：最后一个任务由循环后的 epilogue
    `if bid + (NPER-1)*K < TASKS: scatter(...)` 补上 ✓；
  - block 的任务数 `n = NPER-1` 时（`NPER=ceil(T/K)`、`T` 不是 `K` 的整数倍时必然存在）：其最后一个任务
    在 `it = NPER-2`，本应由 `it = NPER-1` 的 `scatter(t-K)` 落盘，但 `it = NPER-1` 的 `t ≥ TASKS`
    ⇒ 整个 body（含 `scatter`）被跳过；epilogue 的守卫 `bid+(NPER-1)*K < TASKS` 对这些 block 也为假
    ⇒ **该任务永远不会被搬运，`A_pad` 对应 64 行保持 `torch.empty` 的未初始化值**。
  我按 §3.3 的任务映射独立仿真（K=min(20,T)）：漏 `K - T%K` 个任务/形状 ——
  `T=8192` 漏 8 个（8172–8179）、`T=1024` 漏 16 个（1004–1019）、`T=752` 漏 8 个（732–739）、
  `T=3072` 漏 8 个（3052–3059）、`T=64`（L1-a）漏 16 个（44–59）、`T=1`（L0-a）漏 0 个。
  受影响任务集中在**最后 1~2 个 chunk**（含尾 chunk），即精度验证最敏感的位置；这是 `torch.empty`
  垃圾值直接进入返回值，属静默精度失败。同时 §7.7 的"`f_a` 的 wait 合计等于该 slot 的任务数"结论不成立
  （这些 block 上 `set` 比 `wait` 多 1 次，虽不造成死锁，但恰好暴露覆盖缺口）。
- 修改建议: 按"滞后 scatter 移出当前任务守卫 + epilogue 按 block 收尾"改一处即可，同步 §6/§7.7/§8.3：
  ```
  for it in T.serial(NPER):
      t = bid + it * K
      slot = it % 2
      if t < TASKS:
          <组装 N → ws_S → set(f_s[slot])>
      if it >= 1 and (t - K) < TASKS:          # ← 守卫与被落盘任务绑定，不再依赖当前 t
          scatter(t - K, (it - 1) % 2, bid)
  last_it = (TASKS - 1 - bid) // K             # 本 block 自己的最后一个任务
  if last_it == NPER - 1:                      # 仅当循环内最多只覆盖到 last_it-1 时补一次
      scatter(bid + last_it * K, last_it % 2, bid)
  ```
  论证：`n = NPER` 时循环内在 `it = NPER-1` 覆盖任务 `NPER-2`，epilogue 覆盖 `NPER-1`；
  `n = NPER-1` 时循环内在 `it = NPER-1`（body 被跳过、但 scatter 已提出守卫）覆盖 `n-1 = NPER-2`，
  且 `last_it = NPER-2 ≠ NPER-1` ⇒ epilogue 不执行，**不会重复写、也不会重复 wait**。
  这样每个任务恰好一次 `f_a` set / 一次 `wait`，§7.7 的配对结论才成立。
  另建议把 §6 的"落盘位置"一行改为"滞后一拍 + 每 block 末任务由 epilogue 按 `last_it` 收尾"。

### 问题 2（v1 新增，阻塞）: `0 × 有限 = 0` 的前提在"block 首任务恰为尾 chunk"时不成立，NaN 会回灌合法行

- 严重程度: 阻塞
- 位置: DESIGN.md §0.6 R3（L202–L215）、§1.3（L261–L266）、§3.3（L703–L722）、§3.5.2 第 1 条（L777–L779）、§0.6 R1 末（L188）
- 问题描述: 设计放弃源算子的"越界行 clamp + β=0"，改为"动态 extent 只搬 `valid` 行，越界行留旧值 +
  `vmin` 限幅 + 0/1 常量掩码"，并以 **"0 × 有限 = 0 精确为 0"** 作为合法行不被污染的关键论证。
  该论证依赖"旧值恒有限"，而设计对"旧值"的唯一说明是"上一个任务的旧值（有限数）"——
  **对 block 的第一个任务不成立**：此时 `l1k` 的 `valid..63` 行、`ub_gc/ub_gr/ub_b16` 的 `valid..63` 行
  是未初始化的片上 SRAM（不是"上一个任务"）。fp16/bf16 的随机位里有相当比例是 NaN/Inf，于是：
  `G[i,j] = k_i·k_j`（`j ≥ valid`）与 `e[i,j] = exp(min(g_i−g_j,0))`（`g_j` 未初始化）都可能为 NaN；
  `N[i,j] = bn_i·e·G·mask01` 在 `i<valid ≤ j`（mask=0 的丢弃区）变成 `NaN × 0 = NaN`；
  随后 `(N²)[i,j] = Σ_k N[i,k]N[k,j]` 会经 `k ≥ valid` 的 NaN 项**回灌到合法下三角元素**，
  使返回值 `A[i, :]`（`i < valid`）出现 NaN。行独立性只保证"垃圾行走不到别的行"，不能保证
  "同一行内上三角的 NaN 不进下三角"。
  触发形态：`Boundary S=56`（`T=1, K=1`，首个任务即尾 chunk，`valid=56`）与 `L0-d S=100`
  （`T=2, K=2`，block 1 的首个任务即尾 chunk，`valid=36`）——**即设计自己的 L0 用例就会踩中**。
  另注：§0.6 R1 段末"越界行 `β=0 ⇒ … A 的该行为 e_i`"与这条实现路径互斥（§3.3 未对越界行置 β=0），
  §3.5.2 第 1 条"A 的 padding 行会被写入单位行"在现伪代码下也不成立（写的是垃圾行）。
- 修改建议: 在 §3.3 明确一个"越界行确定性归零"的落点，三者任选（建议 a）：
  (a) **块首清零 + 动态 extent 覆盖**：`with T.Kernel` 开头对 `l1k`（整块 64×dim_k）与
      `ub_gc/ub_gr/ub_b16` 做一次显式清零（`T.clear`；若 `T.clear` 不支持 L1，则用一次 GM 零张量
      `T.copy` 到 `l1k`），之后每任务的动态 extent 拷贝只覆盖 `valid` 行 ⇒ 被覆盖区恒有限、
      首任务时为零 ⇒ `β=0 ⇒ N 行 = 0 ⇒ A 行 = e_i`，与 §0.6/§3.5.2 的文字一致；
  (b) 尾 chunk 分支改用源算子的 clamp 装载（`k[min(base+t,end−1)]`、`g` clamp、`β=0`），
      即 §9 R4 的回退方案，代价是尾 chunk 走非异步标量装载；
  (c) 在 `N` 产出后补一次"行有效域"归零：用 `rowv[i]=clamp(valid−i,0,1)`（`vsub`+`vmin`+`vmax` 三个
      64-wide op）广播成 `[64,64]` 再 `vmul`，把 `i ≥ valid` 的整行压成 0（可同时保证 padding 行 = e_i）。
  任何一种都要把 §0.6 R1 末句 / §0.6 R3 第 1 条 / §3.5.2 第 1 条改写为**同一套机制**，并在 §8.3 的
  Boundary（`S=56`）/L0-d 明确断言"合法行无 NaN、padding 行为单位行"。

### 问题 3（建议）: 掩码微探针的表述与实测不符（"三形态全 0"、"vmin 路径 0 误差"）

- 严重程度: 建议
- 位置: §1.6.2 V9（L484）、§3.4（L754）、§11 常数表（L1158）
- 问题描述: 我复跑 `design_probe_mask.py` 得 `MA sum=0`、`MC sum=0`、**`MD sum=1008 / max_abs_err=5.0e-01`**、
  `MB sum=2016 / err=0`。即：`arange+vcmp` 两形态确为"静默全 0"（结论成立），但 **MD（`vmin`+`vsub`+`vmul`
  那条替代路径）不是"精确 0 误差"，而是恰好 0.5 倍掩码**（探针里少乘 2）；§1.6.2 V9 写
  "`vmin`+标量运算与「GM 常量 tile」路径精确（0 误差）"不成立。
- 修改建议: V9/§3.4/§11 改为"MA/MC（arange+vcmp 两形态）静默全 0；MD 的 clamp/减法形态在本探针实现下
  得 0.5 倍掩码（实现缺陷，非结论依据）；**只有 MB（host 常量 0/1 tile）精确**"，或把 MD 修正为
  `…×2` 后重跑再决定是否保留该对照。

### 问题 4（建议）: `y_blk` 工作区容量写大 4 倍

- 严重程度: 建议
- 位置: §4.1（L808）、§4.5（L871）
- 问题描述: `y_blk[K,2,64,64]` fp16、`K=20` 实为 `20×2×64×64×2 B = 320 KB`，设计记 **1.28 MB**
  （等于按 fp32 或 4 份计）。HBM 充裕，不影响容量结论，但属"计数须带推导式"的纪律漏项。
- 修改建议: 改为 0.32 MB，并把 §4.5 的 GM 合计同步为 `2.56 + 0.32 + A_pad + 32 KB`。

### 问题 5（建议）: §5.3 的分核举例算术错误

- 严重程度: 建议
- 位置: §5.3（L931）
- 问题描述: "`T=1024` → 12 个核 52 个、8 个核 51 个"合计 1032 ≠ 1024；正确为 **4 个核 52 个、16 个核 51 个**
  （`4×52+16×51=1024`）。分核公式本身正确，仅举例数字错。
- 修改建议: 更正举例，并补一句推导（`1024 = 20×51 + 4`）。

### 问题 6（建议）: R3 的"访存 425 KB"缺少可逐项相加的推导

- 严重程度: 建议
- 位置: §1.6.0 R3（L347–L360）、roofline 流量项（L402）
- 问题描述: 按 §4.4/§3.3 的流量口径逐项相加：`k 16 + g/β 0.4 + gs_G 32 + ws_S 32 + ws_T(6 store+5 load)176
  + ws_P(5+5)160 + y_blk 16 + A_pad 8 = 440.4 KB`，设计记 425 KB（差 15 KB，疑漏计 16 KB 的 `k` 读）。
  流量项 5.9 vs 6.1 µs，不改变"Cube 发射主导（9.25 µs）"的结论。
- 修改建议: 在 R3 下补一行相加式；若刻意不计输入读，显式注明"访存口径 = 中间 GM 往返 + 输出写，不含输入读"。

### 问题 7（建议）: 源码行号引用漂移

- 严重程度: 建议
- 位置: §0.2（L80–L81）、§0.3 P0（L92）
- 问题描述: `head % key_head == 0` 与 `chunk_size != 64 or dim_k not in (64,128)` 的硬校验实际在
  **L173–L176**，设计写 `L189–L192`/`L191`（v0 原本是正确的 L173–L176，本轮被改错）；
  P0 的 `L256–L271` 应为 `L257–L271`。
- 修改建议: 更正为 L173–L176 / L257–L271，并顺手复核其余行号（我抽查的 L216/L199/L200/L242/L295/L480–L503/
  L519–L540/L543–L618/L688 均正确）。

### 问题 8（建议）: §5.3 的"所有 in-scope 形状 T ≥ 752"与 §5.1 矛盾

- 严重程度: 建议
- 位置: §5.3（L915）
- 问题描述: §5.1 自己列了 `L0: T=1`、`L1-a: T=64`；`K=min(20,T)` 已覆盖小形状，
  但"所有 in-scope 形状 T ≥ 752"的表述不成立。
- 修改建议: 改为"真实业务形状 `T ≥ 752`；L0/L1 的小形状由 `K=min(20,T)` 退化为单轮/少轮"。

### 问题 9（建议）: §4.3 的"alias 表"名不副实，易误导 Stage 3

- 严重程度: 建议
- 位置: §4.3（L843–L856）、§9 R13（L1107）
- 问题描述: 表中列的是 `UB_A/UB_B/UB_F/UB_E` 的**原地复用**（`vsub/vmul` in-place），代码里仍是
  7 个独立的 `T.alloc_ub`，并没有 buffer 别名；§9 R13 的"Stage 3 必须照表落 buffer"应说明
  "目标形态 = 7 个独立 buffer + 原地复用（122 KB）"，而不是"做 alias"。
- 修改建议: 把表标题改为"UB 原地复用/生命周期表"，并在 R13 明确 7 个独立 buffer 即达标；
  若 Stage 3 真的想用 buffer 别名，需另行验证 `T.alloc_ub` 的别名语义（当前无先例）。

### 问题 10（建议）: §1.6.1 A1 的基线代价表述与源实现不一致

- 严重程度: 建议
- 位置: §1.6.1 A1（L430）
- 问题描述: A1 的"收益"栏正确写了"4 组 64-wide exp 表 + 2016 次除法"，但"原式"栏写
  "每元素 2 exp + 1 div"，与源实现（表化后每元素仅 1 次除法、exp 只 4×64 次）不符，容易被读成
  "省掉 2/3 的 exp"。
- 修改建议: 原式栏改为"块内参考 + 4×64 次 exp 建表 + 每元素 1 次除法（fp16 门表中转）"，
  收益栏保留现有量化。

### 问题 11（建议）: L0 层缺少能暴露"每 block 任务数 < NPER"的用例

- 严重程度: 建议
- 位置: §8.3（L1065–L1082）
- 问题描述: L0 的 `T=1/2` 时 `NPER=1`，问题 1 的覆盖缺口在 L0 全绿；要到 L1-a（`T=64, NPER=4`）才暴露。
- 修改建议: 增加 L0-i：`S=640, head=1, key_head=1, dim_k=64`（`T=10, K=10, NPER=1`，不触发）
  **或** `S=1280, head=1`（`T=20, K=20, NPER=1`）——需满足 `NPER≥2` 且 `T%K≠0` 才有覆盖力，
  建议直接取 `T=25`（`S=1600, head=1`，`K=20, NPER=2`，5 个 block 只有 1 个任务）；成本仅 25 个任务。

## 上一轮（v0）9 条建议级问题的处置核对

| v0 问题 | 处置 | 判定 |
|---|---|---|
| 5. fp16 退路与 V4（err=1.875）冲突 | V4 判废（理由：探针 if-chain 缺 `elif mode=="V4"` 分支，实为空执行 ⇒ 垃圾输出；我读 `design_probe_accum_form.py` L35–L55 确认成立）；改用 2 级链实测（fp16 rel 2.8e-5 / fp32 5.5e-9，我复跑一致）；L2 重定义为"fp16 链 scratch + fp16 操作数"并加"必须先过 L0 6 级链"前置 | **已修**（残留：6 级链 fp16 误差仍未实测，§9 R2 已披露） |
| 6. §8.1 fp64 通道与 1e-10 矛盾 | `blocked_source` 保留 fp64（实测 2.7756e-17）；§8.1 拆"结构层 1e-10"与"量化层 信息性" | **已修** |
| 7. §3.3 四处不可执行写法 | `F_S+slot` 常量基址、`gram(bid,0)`、删 `T.transpose`（改二次落位）、`vcast(round_mode=)` 关键字 | **已修**（均由真机探针佐证） |
| 8. `g_ub/b_ub` 一维与 `vbrc` rank 冲突 | §4.1/§4.3 统一 `(64,1)/(1,64)` | **已修** |
| 9. "gemm A/B 不能同 buffer"与事实不符 | 删该断言、删 `l1q`、`P²` 用 `T.gemm(l1p,l1p,l0p)`，实测 rel 5.5e-9；L1 预算 64 KB | **已修** |
| 10. R1 未落笔参考表命中行 | 新增 R1-g（直接求解）/R1-h（对角三角降阶） | **已修** |
| 11. O5/O12 动机误判 | O5 改"门表复用 + 除法复用"、O12 改"clamp 幂等重算（非早退）" | **已修** |
| 12. UB alias 未列表化 | §4.3 新增复用/生命周期表 + 两档预算（122/168 KB） | **已修**（表述上"alias"名不副实 → 新建议 9） |
| 13. 算子级 §6（Neumann）未裁决 | §0.6 新增裁决段（contract 不同、量化对比、不改算子级文档、建议 kernel 5/6 改 `gemm(A,rhs)`） | **已修** |

## 上一轮 4 个阻塞项闭环判定（摘要）

| v0 阻塞项 | 判定 | 关键证据 |
|---|---|---|
| 1. §3.3 广播方向反了 | **已闭环** | §3.1 S3/§3.3/§1.6.2 V2 三处统一为 `vbrc(64,1)→g_i`、`vbrc(1,64)→g_j`、`vsub(g_i,g_j)`；`T.vbrc.md` §2.2.2 语义 + `VP-2026-0087`；A6 机检 viol 0、反例 viol 3.7e-1；真机 `e vs exp(g_i−g_j)=1.4e-6`、反向 1.07e14（我复跑逐项一致） |
| 2. `N` 负号全链路丢失 / 恒等式写错 | **已闭环** | §0.6/§1.3 改为 `N≜−S_sys`、`A=(I−N)^{-1}=ΣN^j=(I+S)^{-1}=Π(I+N^{2^i})`；负号唯一落点 `vcast(β)→vmul(−1)` 在 §3.1 S3/§3.3/§3.2/§1.6.2 V2 四处同源；`verify_equiv` 的 `doubling()` 用 `N=−S`、符号裁决 2.776e-17 vs 4.361e-1；真机 `N vs −β·e·G = 1.1e-7`、`vs +β = 2.5e-1` |
| 3. `store_fixpipe` 输出 region 语义错 | **已闭环（落盘语义本身）** | 改三步落盘（`fixpipe→y_blk` 连续块 → UB → 一次 strided region 写），`A_pad` 改 3D `[S_pad,head,64]`，返回 `[1,T,head,64]` 连续视图；`T.copy.md` §2.2.2/§2.3 佐证；真机 `design_probe_store_layout.py` 用同一 region 形态验证目标行区正确、相邻槽零污染（2.419e-4 = fp16 舍入、other=0）；§1.6.3 L4 的弃选前提已改正。**但同一处修订引入了新的调度缺陷（本次阻塞问题 1）** |
| 4. 复杂度表算术/依据不可追溯 | **已闭环（4 项主诉全修）** | 8192 任务/410 每核/3.79 ms ✓；B0 16/26（dim_k=128 为 36）✓；C1 12 链 gemm + Gram = 13、op 55 ✓；16³ 与 64³ 统一 0.25 µs/op 并标注未实测；`CONST-vector-launch-overhead` 改回 0.5 µs/op 且 Cube 0.25 单列；`r3_metrics` 机检 **pass**。残留 3 处小算术（425 KB、y_blk 1.28 MB、§5.3 举例）→ 建议 4/5/6 |

---

（本检视只写 REVIEW.md；未改动 DESIGN.md、kernel、探针与任何状态文件。所有真机数值为本轮独立复跑。）
