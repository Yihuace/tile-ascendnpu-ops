# `_gqa_bwd_kernel` 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/multi_head_attention/_gqa_bwd_kernel/DESIGN.md`（**1669 行**，revision 1）
- 上一版: `examples/multi_head_attention/_gqa_bwd_kernel/history_version/design_v0.md`（1565 行）
- 任务类型: 迁移（harness 模式，`mode=revision`；源算子 `gqa_bwd.py` 的 `_gqa_bwd_kernel`）
- 源算子: `/workspace/TileOPs-gpu/tileops/kernels/attention/gqa_bwd.py`（`_gqa_bwd_kernel` 第 514–619 行 + `GQABwdKernel` 第 622–663 行；op 层 `tileops/ops/attention/gqa.py` 第 1376–1383 行）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: **0**
- 建议级问题数: 6

> 本轮为**设计修订后的重检**：上一轮 `结论: 不通过`（2 阻塞 + 5 建议）。本轮逐条核对闭环，
> 并**亲自 Read GPU 源码全文 + 亲自 Read 并重跑三枚设计期探针**（上一轮的阻塞漏检根因即探针证据链断点）。

## 机械复核（D-1/D-5）

### ① `verify_equiv.py` 重跑（D-1，2026-10-16，`/workspace/venv/bin/python`）

```
O1 mask-penalty            : cases=110  max_ulp=0      violations=0     -> EQUIV_PASS
    mask coverage          : 80/110 cases have masked>0 (61740 masked elements);
                             tiles with masks=[(1, 1), (7, 5), (16, 64), (32, 32), (64, 32), (64, 64)]
O1 non-finite domain note  : candidate propagates NaN where the source select yields 0  (observed=True; ...)
O4 vexp2 -> vexp           : cases= 60  max_ulp=16     violations=0     -> EQUIV_PASS
    vs fp64 oracle (ulps)  : baseline=13  candidate=13  -> candidate not worse by more than one tolerance budget: True
O2 in-place dS             : bit-identical=True  -> EQUIV_PASS
O3 shared cast scratch     : bit-identical=True  -> EQUIV_PASS
P1 upcast+fp32 product     : fp16 max_ulp=0  bf16 max_ulp=0  -> EQUIV_PASS
R1 scale-into-K (REJECTED) : max_ulp=2543873248  max_rel=2.782e+00  -> rejected (input-dtype operand quantization)
VERDICT: ALL ADOPTED ITEMS EQUIV_PASS
```

执行结果与 §1.6.1「D-1 等价性机器验证结果表」内嵌数字**逐项一致**：
O1 110 案例 / 最大 ulp 0 / 超容差 0 / 80 含 masked / 61 740 masked 元素 / 6 tile 全覆；
O4 60 案例 / 最大 ulp 16 / 超容差 0 / 基线 13 ulp = 候选 13 ulp；O2/O3 逐位；P1 0/0；R1 2543873248 / 2.782。
**上一轮问题 5 已闭环**：掩码分支不再集中于单一 tile（`_cases()` 扫 4 个 `d0`，6 个 tile 形状**都含 masked 位置**），
且 `check_o1` 对 **clamp 形式 / `min(−PEN·(diff−d0),0)` 形式 / 源 select 三者互比**（脚本内 `_ulp(Pc,Pb)/_ulp(Pa,Pb)/_ulp(Pa,Pc)`）。

**空真守卫核对（Phase 1 第 7 步）**：逐组核对比较元素数——O1 最小 `(1,1)` = 1 元素、O4 最小 `(1,1)` = 1、O2/O3 `(64,64)` = 4096、
P1 `2×65536`、R1 `512×128`，**无零元素比较**（`viol=0` 非空真）。O1 的每条 case 均满足其声明前置（`valid = diff ≤ d0` 由构造式确定）。
⇒ 机械复核通过。

### ② `design_calc_check.py`（D-5）

```json
{
  "design": "examples/multi_head_attention/_gqa_bwd_kernel/DESIGN.md",
  "capacity_constants": {"ub_kb": 192.0, "ub_inflation": 1.7, "l0c_kb": 128.0,
                         "source": "pattern-library/constants.md CONST-capacity-910B2C"},
  "checks": [
    {"check": "ub_budget",  "status": "skip", "detail": "no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"},
    {"check": "l0c_budget", "status": "pass", "detail": "block 64x32 x fp32 accum = 8.0 KB vs L0C 128 KB"},
    {"check": "core_split", "status": "skip", "detail": "insufficient parsed dims {'bm': 64, 'bn': 32}; declared logical cores = None; recognised workload rows = 2"},
    {"check": "r3_metrics", "status": "pass", "detail": "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"}
  ]
}
```

- `l0c_budget = pass` —— **本次为真阳性**（本算子是 MixCV，有真实 Cube 路径）。人工复核单次 GEMM 最大输出：
  `dv_acc`/`dk_acc` `[bm,dim]` fp32（`dim=128 ⇒ 64·128·4 = 32 KB`）与 `dq_blk` `[bn,dim]`（16 KB）⇒ **32 KB ≤ 128 KB** ✓。
- `ub_budget = skip` —— **机械跳过，人工已核（pass）**：按 §4.3 公式 `(2·bn·bm + (2·bm+bn)·dim)·4 B + bn·bm·2 B` 独立复算：
  `(64,32,128)` ⇒ `(4096 + 20480)·4 + 4096 = 102 400 B = 100 KB`（§4.5 声明 100 KB ✓）；
  `(64,64,64)` ⇒ `(8192 + 12288)·4 + 8192 = 90 112 B = 88 KB` ✓；`(32,32,128)` ⇒ 59 392 B = 58 KB ✓；
  上探档 `(64,64,128)` ⇒ 139 264 B = 136 KB ✓；causal 追加 `bn·bm·4` = 8/16 KB ✓。
  ×1.12 = 112 KB / ×1.7 = 170 KB **均 ≤ 196 608 B**（主选档双口径均可行）✓。
- `core_split = skip` —— **机械跳过，人工已核（pass）**：逻辑核 `B·H·⌈S/bm⌉` = 128（smoke）/ 8192（full-large）/ 4096（full-bf16）复算一致；
  `num_kernels = min(num_logical, 20) = 20`、`num_local_tasks = ⌈8192/20⌉ = 410 / ⌈4096/20⌉ = 205 / ⌈128/20⌉ = 7` 复算一致。
  物理核数**本机实查复核**：`NPUUtils.get().get_aicore_num() = 20`、`AscendArch().aivector_core_num = 40`、`ub_cap = 196608`
  （2026-10-16 本机 910B3 实跑，与 §5.5 记录一致，且与 `constants.md#CONST-aicore-910B2C` 的 910B3 数据点互证）。
- `r3_metrics = pass` —— R3 四口径（GEMM 单位 / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）齐备，且基线候选 0 在表 ✓。

### ③ 设计期探针重跑（D-3；**本轮亲自读源码 + 单独串行重跑**，避免 NPU 争用污染计时）

| 探针 | DESIGN 声称 | 本轮实测 | 判定 |
|------|-------------|----------|------|
| `design_probe_row_atomic_bshd.py`（探针三，新增） | 目标区 2048/2048 全中；区外 65536/131072 全 0（含 `h=H−1`）；20 并发同址 ⇒ 20.0；动态行数 36 行 ⇒ 2304/2304；行原子/store 比 0.99× / 0.79× | geo1 `(1,128,8,64)` rows0:32 h3 ⇒ **2048/2048 == 1.0**、区外 65536 全 0；geo2 `(1,64,16,128)` rows0:16 h5 ⇒ **2048/2048 == 1.0**、区外 131072 全 0；N=20 ⇒ **2048/2048 == 20.0**、N=8 ⇒ 8.0、**h7=H−1** N=4 ⇒ 4.0（区外全 0）；dyn `(1,100,8,64)` row0=64 bn=64 ⇒ nrow=36、**2304/2304 == 3.0**、区外 51200 全 0；计时 1.00× / 0.85×（计时抖动；地板掩盖单价，与设计诚实登记一致） | **逐项一致 ⇒ 「已验证」表述与工件相符**（上一轮阻塞 2 关闭） |
| `design_probe_persist_and_strided.py` | Q1/Q2 rank-2 双侧显式切片跨步 store/atomic 逐位正确（band 1.0、区外 0）；Q3 persistent+gemm 可用且 `max_abs_err=4.578e-05` | Q1 `strided T.copy : band[1.0,1.0] expect 1 | outside max 0.0 | exact=True`；Q2 `strided atomic : exact=True`；Q3 `n_kernels=4/n_tasks=8` 与 `20/4` **均 OK、`max_abs_err=4.578e-05`** | **逐项一致**；且 docstring 已把被测形态显式收窄为 **rank-2 显式切片**（不再外推 BSHD rank-reduced 4-D）⇒ 上一轮阻塞 2 的「形态外推」表述已改写 |
| `design_probe_atomic_vs_store.py` | ① 连续 2-D 区域 原子/store = **0.97×**；② 跨步 band 原子与 store 逐位精确（4.0 / 1.0、区外 0） | ① **0.99×**（129.28 vs 128.50 µs / 8.4 MB）；② `strided atomic_add band[128:256] [4.0,4.0] expect 4 | outside max 0.0 | exact=True`、`strided store band[128:256] [1.0,1.0] | exact=True` | **① 与 ② 均复现（0.97× ≈ 0.99×，同量级同结论）**；⚠ **遗留缺陷见「非阻塞改进建议 #4」**：**连续** case 的期望值仍写 `NBLK_N=4` 而实际每 cell 写 1 次 ⇒ 该行 `exact=False` 恒真、脚本退出码 1 |

> 三枚探针编号与 §1.6.0 探针表 / §10 交付清单**逐条对应**；探针三在**设计自身 manifest 几何**（`H ∈ {8,16}`、`dim ≠ H·D`、
> 含 `h = H−1`）上正面覆盖修正形态 ⇒ 上一轮的两条阻塞（问题 1 形态不可落地 / 问题 2 证据链断点）均已闭环。
> 独立复核：§1.6.0 的调研（R1/R2/R3/R4）、R3 四口径算术、§1.6.1 四项优化的等价性、§1.6.2 的向量化覆盖与 §1.6.3 的布局候选矩阵**均由本轮逐项推演/复算**（详见维度 8）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

逐项亲自 Read GPU 源码（`gqa_bwd.py` 第 1–103、510–663 行）核对结果：

| 检查项 | 结论 | 证据（源码语句/行 + DESIGN 章节） |
|--------|------|-----------------------------------|
| 语义理解正确性 | pass | `scale = (1/dim)**0.5 * LOG2E`（522）/ `sm_scale = (1/dim)**0.5`（521）↔ §0.1、§1.3；`P = T.exp2(qkt*scale − lse_shared[j])`（589）↔ §0.1；causal 掩码**置 P=0**（592–593，`qkt` 此时已是 P）↔ §0.1 边界语义行；P/ds 回落输入 dtype（`qkt_cast` 598 / `dst_cast` 604）↔ §0.1 dtype 行；`dv/dk` 沿 `k_idx` 跨迭代累加、循环末一次性写回（613/615）↔ §0.1 规约行；`dq` 每迭代清空后逐元素原子（608–611）↔ §0.1；`loop_st = floordiv(by*bm, bn)`（580）↔ §0.6 R6；9 参数全为入参、工厂**无 `out_idx`**（528–532）↔ §0.2；`lse` base-2 域与同包 fwd golden `logsumexp/log(2)` 自洽（交叉核对 `_gqa_prefill_fwd_kernel.py`） |
| 算法解读完整性 | pass | §0.3 的 S0–S15 逐语句归位，覆盖源码 550–615 行全部计算语句（`k_shared/v_shared` 装载 575–576、`dv/dk` 清零 577–578、5 处 `T.gemm` 586/597/599/605/609、两次 `T.Parallel` 元素链 588–593/603–604、3 处原子 611/613/615），**无遗漏、无臆造**；`make_dq_layout`（99–103）正确归类为实现层手段而非 host 逻辑 |
| 优化手段覆盖完整性 | pass | §0.4 的 12 项均可在源码中逐条指认（#1 SMEM tiling、#2 fp32 fragment 累加、#3 单趟融合、#4 mma、#5 Pipelined、#6 causal 块级跳过、#7 块内掩码、#8 P/ds 回落、#9 原子、#10 三维 grid、#11 swizzle 234–235、#12 线程级并行）；#3/#6 归算法层、#7/#9/#11 归优化层**判定正确**；§0.4 末尾明示本函数**没有** wgmma / `load_nd2nz` / `store_fixpipe` / workspace / running max（与源码一致，未误搬） |
| 耦合性判定合理性 | pass | §0.5 共 18 行，四态计数复核 = 保留 5 + 等价替换 4 + 重新设计 3 + 舍弃 5 + 不适用 1 = **18** ✓；舍弃项（`threads`、`make_swizzled_layout`、`make_dq_layout`、`pass_configs`+`compile_flags`、`supported_archs`）在源码中均确实存在（551 的 `threads=threads`、234–235、232–233、528–532、623）且均为 GPU 专属；与 `T.Kernel.md` §2.1（`is_npu` 最多 2 维）不矛盾 |
| NPU 重设计可行性 | pass | R1–R7 逐项有「源方案 → NPU 新算法 → 语义保持论证」。**R3（本轮重点）**：新算法改为逐行 1-D 行原子 `T.atomic_add(dq[bz,c0+r,bx,0:dim], dq_blk[r,0:dim],[1,dim])`（§3.3 第 1030–1032 行），由**探针三**在 BSHD 几何上正面实测（本轮重跑：目标区全中/区外 0/20 并发精确/动态行数正确）；`groups==1` 的单写者论证经复核成立（写者数 = `groups`，`bx//groups` 与 `by` 在 `groups==1` 时构成双射 + op 层预置 0）；R1 的 persistent 可行性有探针二 Q3 本机复验；R6 尾块与 §0.6 R6 同款前作形态；R4 惩罚掩码的代数等价**由本轮独立推演确认**（见维度 8） |
| 设计与决策一致性 | pass | §1.4（步骤 15/16）、§1.5（数据流末段）、§1.6.2 #9/#10、§3.1（步骤 15/16）、§3.2 B7/B8、§3.3（1030–1044）、§4.4（1201–1202）、§7.1（1441）、§7.2（1449）、§9 R1/R9（1598/1606）**全部同步为逐行 1-D 行原子**，无「§0 判重设计、§3 照抄源方案」的脱节；§0.6 末尾 12 项承接表逐项落实、**无静默丢弃** |
| golden 独立性 | pass | §8.1 golden 只按 §0.1 语义实现（fp32 物化 softmax + `exp2(qk·scale2 − lse)` + 三个矩阵乘 + `valid` 乘法掩码），**不引用**分块循环 / tile 形状 / v-prefix / 原子策略 / UB 预算；golden 的 `valid = torch.le(kj, qi)`（kv ≤ q）与源 `by·bm+i ≤ k_idx·bn+j` 同向；GQA 的 `dvf[b,g] += pᵀ·dof[b,h]` 按组归约与 §0.1 一致 |
| host/op 层引用路径真实性 | pass | `tileops/ops/attention/gqa.py` 第 1376 行 `do = do.contiguous()`、1378–1380 `torch.zeros_like(..., dtype=torch.float32)`、1381 `self.kernel(q,k,v,do,lse,delta,dq,dk,dv)` **逐行核对存在**；§0.3 已按上一轮问题 7 改写为「只对 `do` 显式 `contiguous()`」（对照 §11.1 第 7 行） |
| 多输出/变体函数的迁移范围 | pass | 本函数无 `_*_with_indices` 类变体；`GQABwdWgmmaPipelinedKernel` / `MHABwd*` 属其它 kernel，`default_kernel_map` 的 `gqa_bwd_kernel` 在非 hopper 时映射到 `GQABwdKernel`（op 层 `gqa.py` 1370–1372 核对）⇒ manifest 只迁移本函数，无范围蔓延 |

> **上一轮阻塞 1 复核结论**：新形态**在文档层面自洽**（§0.6 R3 / §0.5 / §1.4 / §1.5 / §1.6.0 / §1.6.2 / §1.6.4 / §3.1–§3.3 / §4.4 / §7 / §9 全部同步），
> 且**发射项重算算术如实**：每迭代 issued-op `8 整 tile + 2 微([bn,1]) + real_n 行原子` = **42**（dim=128/bn=32）/ **74**（dim≤64/bn=64）（§1.6.0 第 550 行）；
> `full-bwd-fp16-large` 迭代 `524 288`（每核 `26 214`）× 42 = **1 101 005 op/核**（复算 524288/20×42 = 1 101 004.8 ✓）；
> 上界档 `×{0.25,0.5} µs` = **275–550 ms** ✓；地板档 `20 544 元素/迭代 × 524 288 ÷ 20 × 17.6 ps = 9.48 ms ≈ 9.5 ms` ✓；
> 流量项（口径 A）`1.34 GB / 1.06 TB/s = 1.27 ms` **不变** ✓；两趟备选聚合 issued-op `2 × 524 288 × 10 = 10.49 M ≈ 10.5 M`（每核 524 288）vs 主选 `22.02 M` ✓。
> **上一轮阻塞 2 复核结论**：§1.6.0 探针表已逐条注明**实际被测形态**（探针一/二 = rank-2 双侧显式切片，并删除外推），
> **新增探针三**专测修正形态（四类断言 + 工具链戳），两枚旧探针的**期望值判据已修正后重跑**（探针二 `n_write=1`、探针一跨步 `expect=NBLK_N`）——
> 本轮**亲自读源码 + 重跑**确认断言真实覆盖修正形态（非形态外推）✓（唯一遗留：探针一**连续** case 的期望值仍错，见建议 #4，不影响设计任一结论）。

### 1. API 可行性 — pass

- `T.atomic_add`（**逐行 1-D 行原子**，跨步 BSHD 目标）：`docs/Tilelang.language/原子操作/T.atomic_add.md` §2.2.1（fp32 ✓）、§2.2.2「atomic_add 对形状无特殊限制」、§2.4 区域形态示例；**生产先例** `examples/Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py` 第 257 行 `T.atomic_add(dKV[by, cur_idx, :], acc_dkv[row, :])`（**动态下标的 1-D 行原子**，亲自 Read 核对）；**本任务探针三重跑逐项通过**。上一轮阻塞的 rank-reduced 4-D 二维区域形态已在 §0.6 R3/§3.2 B7 明确标注**实测不可用**并弃用 ⇒ 阻塞解除。
- `T.Kernel(一维, is_npu=True)`：`T.Kernel.md` §2.1/§2.2（「最多 2 个 block 维度」）✓。
- `T.alloc_shared`：`T.alloc_shared.md` §2.4 示例 1（matmul 场景 ⇒ L1）✓；§2.2.2「1D~2D；3D~5D 有待充分验证」⇒ 设计「全部 2D 声明」的规避纪律**成立**（§4.3）。
- `T.copy`：`T.copy.md` §2.2 规则；双显式 slice + dst 零起点规避 `TRAP-T-copy-region-semantics` 与 `TRAP-UB-dst-align` ✓。
- `T.gemm`（`b_transpose`/`a_transpose`/`initC`/`size`）：`T.gemm.md` §2.1–§2.4；先例 `examples/deepseek_v32/sparse_mla_bwd.py` 第 215 行（`b_transpose=True`）、第 258 行（`a_transpose=True, initC=True`）**亲自核对存在**；其 `acc_dq` 确为 `alloc_shared` + 循环外 `T.clear`（第 187 行）+ 循环内 `T.gemm(initC=False)` 跨迭代累加（第 255 行）✓。
- `T.vmul`/`T.vsub`/`T.vadd`：亲自打开文档 —— `T.vmul.md` §2.2.2「行广播 `[M,N]*[M,1]`」✓；`T.vsub.md` §2.1 `src1 = "tensor or scalar"`、§45「仅支持 `src1` 为标量」⇒ §3.3 的全部标量用法（`vsub(dplane, d0)`、`vmul(x, ±sm_scale/−PEN)`）**都把标量放在 src1** ✓；原位（dst 同 buffer）✓。
- `T.vmin`（scalar src2）：`T.vmin.md` §2.1 `src2 = "tensor or scalar"`、§42「src2 可以使用标量」✓；**不用 `T.vmax`** 的理由核实成立（`T.vmax.md` §2.1 `src0/src1` 均为 `tensor`）✓。
- `T.vexp`（O4 后）：`T.vexp.md`（fp32 ✓、src/dst 同 shape ✓）；`T.vexp2.md` §1「底层不支持硬件级别的 exp2，实际使用 `exp(A*ln2)`」⇒ O4/C5 论据**逐字成立** ✓。
- `T.vcast(..., round_mode="rint")`：`T.vcast.md` §2.2.1（`f32→f16` / `f32→bf16` rint ✓）、§2.2.2（shape 须一致）✓；`TRAP-vcast-same-dtype-rounds` 存在（设计两处均为真 dtype 变化）✓。
- `T.min`（标量）、`T.Cast`/`T.Parallel`（掩码平面物化，体内单一算术表达式）、`NPUUtils.get().get_aicore_num()`：均核实 ✓（后者本机实跑 = 20）。
- `T.atomic_addx4`：文档 §1 明示与 `T.atomic_add` 完全等价 ⇒ 不采用的理由成立 ✓。
- v-prefix 优先 ✓；无 `compile_flags`/`pass_configs`/`out_idx`/`threads` ✓（符合 `target="npuir"` 约束与 K9）。
- `T.transpose` 弃选前提亲自核对：`docs/Tilelang.language/创建操作/T.transpose.md` 存在 + `TRAP-transpose-ub-gm-roundtrip` 存在 ✓。

### 2. 内存层级规划 — pass

- 搬运路径完整：GM → L1（`k_s`/`v_s`/`q_s`/`do_s`）→ L0A/L0B/L0C（5 处 `T.gemm`）→ UB（v 链 + fp32 累加器）；**无 GM→L0 越级**、无中间结果落回 GM（除三输出）、无 workspace ✓。
- UB 预算：本轮独立复算 100 KB（dim=128/bn=32）/ 88 KB（dim=64/bn=64）/ 58 KB（兜底档）/ 136 KB（上探档）、causal +8/+16 KB；×1.12 / ×1.7 均已在报告「机械复核②」列出，主选档双口径均 ≤ 196 608 B ✓。
- L0C：单次最大输出 32 KB ≤ 128 KB ✓（与 D-5 的 `l0c_budget = pass` 一致，且本次该 check 是有真实 Cube 路径的真阳性）。
- Buffer 层级标注：§4.3 逐 buffer 标注 UB/L1 且**全部 2D** ✓；角色复用时序（`qkt` P→ds、`cast` P→ds、`dsum` 掩码临时→g2→ds 中间）与 §3.3 时序一致 ✓。

### 3. Tiling 策略 — pass

- Block 划分与量化依据 ✓（`bm=64` 受 `dv_acc+dk_acc = 2·bm·dim·4` 约束；`bn` 受 P 链 + `dq_blk` 约束；`dim` 不切分）。
- 对齐/整除 ✓（`dim ∈ {64,128}` 为 8 与 16 对齐；`bm=64`、`bn ∈ {32,64}` 为 32 倍数；`dq_blk`/`dv_acc` 尾轴 = `dim` 连续段）。
- 非整除 ✓（`real_m`/`real_n` 双显式 slice + `size=` fractal 裁剪 + 写出裁剪；与 `TRAP-L1-band-dst-tail-overrun` 教训一致）。
- 分核三要素齐备 ✓：① 逻辑核数公式与四档数值复算一致；② 物理核数**本机实查复核 20**（混合算子不翻倍，与 `core-split-strategy.md` §1 要素②一致）；③ 极大规模判定（128–8192 ≫ 20）⇒ 固定核数 + `T.serial(num_local_tasks)` 静态界 + `uid < num_logical` 守卫 ✓。
- **上一轮问题 4 已闭环**：兜底档改为 `(32,32)`（K=32 ✓），工厂 guard 改为 `block_m, block_n ≥ 32`（§3.3 第 929–930 行 `if block_m < 32 or block_n < 32: raise`），§4.5 阶梯表与「域上界」行同步 ✓。

### 4. 技术约束检测 — pass

- 三维 Kernel 已识别并给出 block 元数据方案（一维 + 整除/取模分解）✓。
- GEMM 非整除已在 Tiling 维度处理 ✓；L0C 溢出已处理（32 KB ≤ 128 KB）✓。
- GPU 专属 API 清单（`threads`/`T.Parallel` 元素循环/`alloc_fragment`/wgmma/swizzle/`make_dq_layout`）逐项处置 ✓。
- bf16 无 Vector 算术：整条向量链在 fp32 域、仅送 GEMM 前 `vcast` 回落 ⇒ 免疫 ✓。
- 物理核数适配 ✓（本机实查 20）。

### 5. 循环与同步 — pass

- 循环结构明确：`T.serial(num_local_tasks)`（任务）+ `T.Pipelined(loop_st, n_q, ns)`（query 块）+ `T.Parallel` 掩码平面（每任务一次，体内单一算术表达式）+ **新增** B7/B8 的 `T.serial(real_n)/(real_m)` 行步进循环；「不使用 `T.Persistent`」有本工具链 grep 依据 ✓。
- 同步与模式匹配：Developer 自动同步，全篇无手动 flag/barrier ✓；§7 依赖表覆盖了 §3.3 的全部跨引擎交接（含 `cast` 的双写 RAW/WAR、`qkt` 原位链、`dsum` 角色切换、跨迭代累加、跨核原子）✓。
- 尾块：§5.3/§6.4 覆盖 `S % bm/bn`、`heads % heads_kv`、`dim` 校验、`num_logical < num_kernels`、causal 的 runtime `loop_st` ✓。
- B7 行原子循环的行数 `real_n` 为 runtime 标量、`T.serial` 接受（探针三动态行数档实测）✓。

### 6. 验证方案 — pass

- Golden ✓（§8.1 独立，含 `golden_lse_delta` 自洽提供 base-2 `lse` 与 `delta`，与「lse 域」契约一致）。
- L0 计划 ✓：**16 条**，含 manifest 四档、causal、非整除、**GQA（`groups=4`）**、退化小档、角点、输入校验、UB 编译阶梯，以及**新增 L0-16**（`(2,300,16,4,128)`：`H=16>1` + `dim=128 ≠ H·D` + `S=300` 非整除 + `groups=4`，专测「逐元素区域写错位置」类缺陷）——上一轮问题 1 建议 4 已闭环。
- 精度标准 ✓：`atol=5e-3, rtol=1e-5` 与 `examples/TileOPs/tests/ops/test_multi_head_attention.py` 第 251 行**逐字核对一致**；tier-2/tier-3 有同包 fwd 先例依据 ✓。
- 该 L0 套件**能**暴露原子形态类缺陷（L0-1 `H=8`、L0-8/L0-9 GQA、L0-16 专测）⇒ 检测链路成立 ✓。

### 7. 完整性与一致性 — warn

- 章节齐全 ✓（§0.1–§0.7 + §1–§11，含 §1.6.0–§1.6.4 与 §11 修订说明）；无占位符 ✓。
- 同类引用真实 ✓：逐条 `ls` 核对 13 个被引用 `examples/` 路径**全部存在**（三个同族 npuir 实现、`atomic_add.py`/`_dev`、`indexer_bwd.py`、`engram_bwd.py`、测试文件、同包 wrapper/fwd/bwd 源工件、同批前作 DESIGN 与实现、`group_norm` DESIGN）。
- 被引用的 pattern-library 条目**全部存在**（逐条 `grep` 核对 35 个：`CONST-*` 12 项、`PL-1.7/1.8/1.9-*/1.11/1.12/1.13/1.15/1.16/1.18/1.21`、`TRAP-*` 21 项）；被引用 `docs/` 15 份**全部存在**。
- **上一轮问题 7 的第 2/3/4/5/6 项已闭环**（pass 计数统一为「整 tile 8/12 + 2 微 + `real_n` 行原子」、候选 2 = 1.88 GB、候选 5 = 52.1 GB、`make_dq_layout` 改指 GPU 源 99–103、§0.3 的 `contiguous` 声称、§1.6.2 #13 改引 `CONST-aiv-time-decomposition`）；第 1 项（O1 掩码单一定义源）形式已统一为 `vmin` 形式。
- **warn（残留内部一致性小项，均为建议级）**：① §1.6.0 D-2 估算表 `full-bwd-bf16` 行的迭代数写 262144（应为 524288，同行的口径 C 用 `⌈S/bm⌉=64` 反证）；② §1.6.4 的「零原子 / `groups` 无关」与 §1.6.0 R4 第 514 行（pass A `groups>1` 用逐行行原子）矛盾；③ §1.6.1 结果表残留 v0 数字（R1 `max_rel 3.25` 应为 2.782、`2.52e9` 应为 2.5439e9、O2/O3 元素数标签）；④ `§3.2 B8`/`§1.6.2 #10` 用探针二 ①② （rank-2）支撑 rank-reduced 4-D `T.copy`；⑤ §0.6 R4 仍写 `exp2(qkt·scale − lse + pen)`（O4 后应为 `exp(x')`）。逐条见「非阻塞改进建议」。

### 8. 算法优化分析 — warn（非阻塞）

**（a）§1.6 章节存在性 ✓**：1.6.0 / 1.6.1 / 1.6.2 / 1.6.3（+1.6.4 备选）齐备，四个子节均有实质分析。

**（b）算法调研存在性与完整性 ✓**：R1 候选表含**基线候选 0**（GPU 源算法）且 8 个候选逐个评估（含新增的「`PEN` 1e38→1e30」项）；R2 在线算法**有明确结论**（「有在线变体但本算子不需要」+ 结构判据：`lse`/`delta` 外部给定 ⇒ 无 running max/重缩放/两遍扫描，且本算子的跨 `kq` 累加本身已是在线式）；R3 四口径齐备（GEMM 单位 / 访存 Bytes 三口径 A/B′/C / 扫描遍数 / 中间缓冲峰值，基线在表，口径定义句含「含什么/不含什么」）；R4 逐候选六维亲和评估；调研结论含选定算法与关键依据；「未使用互联网检索」已声明并给出本地覆盖范围。对照 `algorithm-candidates.md` 的 `ALG-attention` 命中行（R1 scale 融入 Q/K + softmax 分母、R2 flash/online 结构、R3 materialize vs flash、R4 Cube+Vector 流水）——**命中行候选均已评估**（scale 融入 K 被机器证据否决；online 被结构判据正面回应），未发现被漏掉的已知候选（attention-bwd 的候选空间「单趟 vs 两趟、materialize vs 重算、原子 vs 无原子」三类均在表中，且新增了 `mb` 分块与 dq-owner 两个结构候选）。

**（c）调研结论正确性（独立复核）**：
1. **等价性独立推演 ✓**：O1 —— 本轮独立推演设计自身掩码映射：`dplane[i0,j0] = j0 − i0`（`i0` = query 轴、`j0` = kv 轴，由 §1.6.3「P tile 声明 `[q 行, kv 列]`」与 §3.3 的 `(bn,bm)` 声明确定），`pen = min(−PEN·(dplane − d0), 0)` ⇒ `pen ≡ 0 ⟺ j0 − i0 ≤ d0 ⟺ by·bm + j0 ≤ kq·bn + i0`（`d0 = kq·bn − by·bm`）⟺ **kv_pos ≤ q_pos** = 源条件 `by·bm+i ≤ k_idx·bn+j` ⇒ **逐位一致**成立（机器验证 ulp 0 覆盖）；O2 —— 逐元素无跨元素依赖、P 的最后消费点早于 ds 写回（机器 `torch.equal=True`）；O3 —— 两次 `vcast` 被 g3 隔开（机器逐位）；O4 —— `exp2(a) = exp(a·ln2)`、`scale·ln2 = D^{-1/2}·1.0000000000000002`（相对偏差 2e-16）⇒ 容差内等价，候选 vs fp64 偏差（13 ulp）**不劣于**基线（13 ulp）✓；R1 否决项有机器证据（2.782）✓。
   **⚠ 但 §1.6.1 O1 / §0.6 R4 的等价性论证句子写作有缺陷**（见建议 #3）：该句用**源的下标约定**（`i` = kv、`j` = query）书写源不等式，却用**设计的下标约定**（`i` = query、`j` = kv）给出结论「`j−i ≤ d0`」，而所写的「两边同减 `by·bm+i`」在该约定下得 `j−i ≥ −d0`（符号相反）⇒ 论证句按字面不成立（设计公式与 §3.3 伪代码经本轮推演**正确**，故不阻塞）。
2. **复杂度算术独立复算 ✓**（重点核对）：口径 A `E·(4·eb+12)`：full-fp16 `E = 16·16·2048·128 = 6.7109e7` ⇒ `×20 = 1.342 GB` ✓；口径 B′ `(⌈S/bm⌉−1)·E·4 = 31·E·4 = 8.32 GB` ✓；口径 C `1.342 + 8.32 + 31·2·E·eb = 17.98 GB` ✓；候选 2 `E·28 = 1.88 GB` / B′ `0` / C `1.88 + 8.32 + 63·2·E·eb = 27.11 GB` ✓；候选 3 B′ `31/2·E·4 = 4.16`、C `13.82` ✓；候选 5 B′ `63·2·E·4 = 33.82`、C `52.05` ✓ —— **四口径算术全部如实**（revision 0 的记账不对称已修正；`r3_metrics` 机检 pass）。**⚠ 唯一算术错误见建议 #1**（D-2 估算表 `full-bwd-bf16` 行的**发射项**迭代数，非 R3 四口径）。
3. **亲和性淘汰依据核对 ✓**：候选 3「UB 结构性超限」与 §4.5 复算一致（`mb=2` 使 `dv_acc`/`dk_acc` 翻倍，`(64,128)` 档 +64~96 KB ⇒ 声明量 ≥164 KB，已逼近/超过 196 608 B，双口径下更差）；候选 4「可免费避免的列广播开销」有 `PL-1.15`（+12~16%）与 `T.vmul.md` 行广播文档佐证；候选 5 有统一口径 C 对比；候选 6 有机器证据（2.782）；候选 7 有 fractal 对齐 + 舍入路径否决；容量/带宽类淘汰引用 `constants.md` 条目（已核存在）✓。
4. **负向断言复核 ✓**：R2「本算子不需要在线化」不是无据否定，而是给出结构判据（`lse`/`delta` 由前序 kernel 给定 ⇒ 无 running max/重缩放）；「无更优替代」写明调研范围（§1.6.0「调研范围」段）；与源码证据不矛盾（源算法确实非 online：`lse` 直接由外部消费，无 running 统计量）✓。未发现「源码已用 online 手段却断言无在线变体」类矛盾。
5. **R3 并列候选裁决路径 ✓**：候选 2（两趟原子自由）未被纸面淘汰，而是降为 §1.6.4 的结构级并列候选，给出**判定阈值**（`< 主选 × 0.95` 且精度不回归）、代表 shape、指标（`Task Duration` + `aiv_vec_time`/`aiv_mte2_time`/`aiv_mte3_time` + L2 命中计数）✓。

**（d）调研→设计一致性 ✓**：§1.4 开篇声明与 §1.6.0 选定算法一致；§3.1 以 §1.6.1 优化后公式为输入（§1.6.1 末「优化后公式」块 + §3.1 步骤 1–16 与 O1–O4 对应）；§6 循环结构与 §1.6.2 结论一致（kernel 内不存在逐元素标量循环，源 S6/S7/S10/S13 的 `T.Parallel` 元素链与逐元素原子已全部消除）；迁移侧 §0.6 末尾 12 项承接表逐项说明意图承接、无静默丢弃 ✓。

**（e）数学等价优化完整性 ✓**：O1–O4 逐项四要素齐备（原式 / 优化后公式 / 等价性论证 / 收益量化，收益栏有数字：`[bn,bm]` fp32 3 块→2 块、dtype 缓冲 2→1 块、整 tile 向量 op 9→8 ≈ 整 tile 发射项 −11%）；P1/P2/P3 三条数值等价性主张有机器/分析性证据 ✓。

**（f）等价性论证正确性 ✓**（逐项独立推演见 (c)1；O1 的 NaN 边界差异已在 §0.6 R4/§9 R11 诚实登记并提供 `vcmp`+`vselect` 精确形态）——**仅论证句的下标约定书写有缺陷（建议 #3）**，公式本身成立。

**（g）向量化替代覆盖完整性 ✓**：§1.6.2 盘点 **18 个**计算点，覆盖 §3.3 伪代码中全部循环/标量点；与 §6 交叉核对一致（`T.Parallel` 仅剩 `dplane` 物化、标量仅剩 `real_m`/`real_n`/`uid` 分解/`d0`/host 常量）；8 个逐元素/条件/归约点（#3–#10）均已给向量替代并有 docs 佐证；保留项理由均具体（#13 动态边界、#14 block 级映射、#15 每任务一次、#16 每迭代 1 标量、#17 kernel 外、**#9 的行步进 `T.serial(real_n)` = 结构性行步进，行体为向量宽 1-D op，因跨步 BSHD 目标的跨步行区域无法单条区域 op 表达**），**无「实现简单」类理由** ✓。

**（h）向量化轴与布局决策完整性 ✓**：MixCV 类别必选候选（P/O tile 轴向、fractal/NZ 路径、epilogue 归属侧、偏置广播轴、装载粒度）均枚举；7 个候选逐条量化评分与弃选理由；弃选前提亲自核对（`TRAP-transpose-ub-gm-roundtrip`、`CONST-store-fixpipe-gm-only` 存在；cube skill 的 Developer/ND 约定与 `T.alloc_shared.md` 一致）；**对 GPU 源码隐式轴选择（`qkt = (kv,q)` + `lse` 列因子）做了独立复核并判定为 CUDA 产物**（正确）；布局决策与 §3.3/§4.3/§6 三方一致（`[bn,bm]` 整 tile、`dv_acc`/`dk_acc [bm,dim]`、`dq_blk [bn,dim]`）✓。

**（i）实验裁决模式完整性 ✓**：§1.6.3 参数级 A/B（判定为参数级、不触发结构三件套——因主选档 = wrapper 默认档，UB 双口径均可行）+ §1.6.4 结构级备选（两趟原子自由）三件套：备选结构到 Stage 4 可实现深度（任务划分 / GEMM 数 / issued-op / UB 84/36 KB / 分核重算 / 闭包工厂形态）、裁决计划含代表 shape、指标、**判定阈值**、回写路径 ✓。

**（j）编译期特化谓词充分条件 ✓**：`atomic_dk = groups > 1` 的依赖变量全集 = {`heads`, `heads_kv`}（`groups = heads//heads_kv` 为工厂常量），集合关系推演成立（每元素写者数 = `groups`，`groups==1 ⇒ 单写者`），「对预置 0 的单写者原子加 ≡ 普通写」的前提在 §9 R5 登记 ✓；负向守卫用例 L0-1（`groups=1`）与 L0-8/L0-9/L0-16（`groups=4`）分别覆盖两个分支 ✓。

**（k）与下游章节一致 ✓**：§3.1 以优化后公式为输入、§6 与 §1.6.2 一致、§3.3 原子形态与 §0.6 R3/§1.6.0/§3.2 全部同步 ✓。

**（l）维度内小结**：阻断级子项（等价性、负向断言、复杂度四口径、布局决策、向量化完整性、实验裁决）**全部 pass**；`warn` 来自 (c)2 的 D-2 发射项迭代数错误与 (c)1 的论证句下标约定缺陷，二者均不改变设计结论（`full-bwd-fp16-large` 数字正确，full-bf16 修正后与 full-large 同量级）。

## 非阻塞改进建议（warn；不影响本结论，供 Stage 1 顺手修订或 Stage 3/4 参考）

### 建议 1（建议级）: §1.6.0 D-2 估算表 `full-bwd-bf16` 行的发射项迭代数少算 2×

- 位置: DESIGN.md §1.6.0「设计期估算」表第 559 行
- 问题描述: `full-bwd-bf16 (4,4096,16,128)` 的 `(bm,bn,ns) = (64,32,1)`，其迭代数应为 `num_logical × ⌈S/bn⌉ = 4096 × 128 = 524 288`
  （与同行口径 C 所用 `⌈S/bm⌉ = 64` 一致，且与 `full-bwd-fp16-large` **迭代数相同**——两档的 `num_logical × n_q` 都等于 524 288）。
  表中写 `262 144 iters → 13 107 iter/核 × 42 = 550 512 op`（= 正确值的一半，对应 `bn=64` 上探档），
  导致该行结论 `138–275 ms / 4.7 ms / ≈4.7–275 ms` 全部少算 2×（应为 `275–550 ms / 9.5 ms / ≈9.5–550 ms`）。
  revision 0 的同一行用的是正确的 524 288（见 `history_version/design_v0.md` 第 525 行），本版改写时把迭代数错换成了上探档值。
- 修改建议: 把该行改为 `524288 iters → 26214 iter/核 × 42 = 1 101 005 op → 275–550 ms（上界档）/ 9.5 ms（地板档）`、结论 `≈9.5–550 ms`；
  并把 §1.6.0 估算表**每行的 `iters` 显式写出推导式**（`num_logical × ⌈S/bn⌉`）以便跨行互校。结论（发射受限）不受影响。

### 建议 2（建议级）: §1.6.4 备选方案的「零原子 / `groups` 无关」与 §1.6.0 R4 自相矛盾

- 位置: DESIGN.md §1.6.4 表第 776/779/783 行（+ §9 R1 第 1598 行、§9 R9 第 1606 行的「零原子备选」措辞）
- 问题描述: §1.6.4 称备选 pass A「任务末**普通区域存储**（两趟 owner 维完整 ⇒ 单写者，`groups` 无关）」、原子行写「**无**」、优势行写「**零原子**」；
  但同一文档 §1.6.0 R4 第 514 行明确写「两趟的 pass A `groups>1` 分支在本设计中同样只用逐行 1-D 行原子」。
  两趟 pass A 的任务是 `(bz, bx, by)`（`bx` = query head），GQA 下同组 `groups` 个 query head 写同一
  `dk/dv[bz, by·bm…, bx//groups, :]` ⇒ **多写者，仍需原子**；「单写者」仅对 `groups == 1` 成立。
- 修改建议: 把 §1.6.4 的「单写者，`groups` 无关」改为「单写者（**仅 `groups == 1`**）；`groups > 1` 时 pass A 的 `dk/dv` 末写回用逐行 1-D 行原子（与 §1.6.0 R4 一致）」，
  并把优势行的「零原子」限定为「**manifest（MHA，`groups==1`）下零原子**」；§9 R1/R9 的「零原子备选」同步加限定词。

### 建议 3（建议级）: §1.6.1 O1 / §0.6 R4 的掩码等价性论证句下标约定不一致（公式正确、论证句按字面不成立）

- 位置: DESIGN.md §1.6.1 O1 行（第 593 行）、§0.6 R4（第 244 行）
- 问题描述: 论证句引用源不等式 `by·bm+i ≤ k_idx·bn+j`（此处 `i` = kv 行、`j` = query 列，与源码第 592 行一致），
  但设计自身的 `dplane[i0,j0] = j0 − i0`（§3.3）中 `i0` = **query** 轴、`j0` = **kv** 轴（由 §1.6.3「P tile 声明 `[q 行, kv 列]`」确定）。
  按所写的「两边同减 `by·bm+i`」，由 `by·bm+i ≤ k_idx·bn+j` 得 `j − i ≥ −d0`，与结论「`j − i ≤ d0`」**符号相反**（字面推导不成立）。
  本轮已**独立推演确认设计公式正确**：`pen ≡ 0 ⟺ dplane ≤ d0 ⟺ by·bm + j0 ≤ kq·bn + i0 ⟺ kv_pos ≤ q_pos`（= 源条件），
  §3.3 伪代码与 `verify_equiv.py` 均与该推演一致 ⇒ 不影响实现与结论；但论证句保留现状会在 Stage 3 复算时造成误导（尤其差一位符号即变成反因果）。
- 修改建议: 在论证句中显式声明两套下标约定：「设计的 `dplane` 以 `i` = query 块内位置、`j` = KV 块内位置（**与源码 `i`/`j` 的轴含义相反**），
  源条件在设计的约定下写作 `by·bm + j ≤ kq·bn + i`，两边同减 `by·bm+j` 得 `j − i ≤ d0`」；
  或直接改写成纯 kv_pos/q_pos 形式（`kv_pos = by·bm + j`、`q_pos = kq·bn + i`）以避免字母冲突。建议 §3.3 的 `dplane` 行旁加一句同样的轴声明。

### 建议 4（建议级）: `design_probe_atomic_vs_store.py` 的**连续** case 自证判据仍错（脚本退出码 1）

- 位置: `history_version/design_probe_atomic_vs_store.py`（`_check(k_store, N, float(NBLK_N), ...)` / `_check(k_atom, N, float(NBLK_N), ...)`）
- 问题描述: 连续 case 的目标是 `dst[bm*BM:(bm+1)*BM, bn*BN:(bn+1)*BN]`（每个 (bm,bn) tile 唯一），**每 cell 恰写 1 次**，
  但 `_check` 传入的期望值是 `NBLK_N = 4` ⇒ 恒 `exact=False`（本轮实测打印 `contiguous store ... exact=False`、`contiguous atomic_add ... exact=False`、
  `contiguous-region atomic numerically exact: False`，`main()` 因此 `return 1`）。这正是上一轮问题 2「探针自带的 `exact` 判据须恢复意义」的**同类遗留**——
  revision 1 只修了**跨步** case 的期望值（`expect=0.0 → NBLK_N`），未修**连续** case。设计对该探针的依赖（`原子/store = 0.97×`）经本轮复跑成立（0.99×），
  故不阻塞；但脚本「绿灯/红灯」不可读，后续复核者会被误导。
- 修改建议: 连续 case 的期望值由 `float(NBLK_N)` 改为 `1.0`（与 label「each tile added once」一致），并把跨步 case 的 `expect` 保持 `NBLK_N`；
  或在 `main()` 的返回判定中区分「本探针真正依赖的两条断言」（连续比值 + 跨步精确）与辅助打印项。

### 建议 5（建议级）: §3.2 B8 / §1.6.2 #10 用 rank-2 探针支撑 rank-reduced 4-D `T.copy`（形态引用不准），及 §1.6.1 结果表残留 v0 数字

- 位置: DESIGN.md §3.2 B8 行（第 882 行）、§1.6.2 #10（第 678 行）；§1.6.1 结果表第 614/615 行、P1 佐证表第 605 行、§1.6.0 R1 表第 454 行
- 问题描述:
  (a) B8/#10 称 rank-reduced 4-D `T.copy`（`dv[bz, r0:r0+real_m, hkv, 0:dim]`）「由探针二 ①② 支撑」，但探针二 docstring 已明确其被测形态是 **rank-2 张量双侧显式切片**（不覆盖 rank-reduced 4-D）。
      该断言**本身为真**（同包 fwd 生产实现与同批前作的装载/写回均为该形态；上一轮 Stage 2 探针 V3/V4 亦验证其正确），但**所引证据与形态不符**——与上一轮问题 2 的「形态外推」属同一类引用纪律问题，仅是方向相反（这次引用不足而非外推过度）。
  (b) §1.6.1 结果表残留 v0 数字未随重跑更新：O2 行「65536 元素」实为 `(64,64) = 4096`；O3 行「2×8192 元素」实为 `2×4096`；P1 佐证表「最大 ulp 差 2.52e9」「最大相对偏差 3.25」应为 `2543873248`（2.5439e9）/ `2.782`（§1.6.1 结果表第 620 行与 §1.6.0 R1 表第 454 行分别写着 2.782 与 3.25，彼此不一致）。
- 修改建议: (a) 把 B8/#10 的佐证改为「同包 fwd 生产实现（`_gqa_prefill_fwd_kernel.py` 与 `_flashattn_bwd_preprocess_kernel.py` 的 `T.copy(o[bz, s_slice, bx, :], frag)` 形态）+ `T.copy.md` §2.2」，或在探针三中并入一条 rank-reduced 4-D `T.copy` 断言；
  (b) 用本轮重跑的实际值统一 §1.6.1 结果表与 §1.6.0 R1 表的 R1 数字（`2543873248` / `2.782`），并把 O2/O3 的元素数标签改为脚本实际张量尺寸（`4096` / `2×4096`）。

### 建议 6（建议级）: 若干引用/表述小项

- DESIGN.md §0.6 R4（第 241 行）仍写 `P = exp2(qkt·scale − lse + pen)`——O4 采纳后应为 `P = exp(x' + pen)`（`x' = fp32(qk)·sm_scale − lse·ln2`）；建议改为引用 §3.3 的 `T.vexp` 形态，保持「单一定义源」纪律。
- DESIGN.md §0.1（第 37 行）/§0.3（第 112 行）引 op 层行号「第 1377 至 1383 行」——实际相关区间为 1376（`do.contiguous()`）至 1381（kernel 调用），建议改为「第 1376 至 1381 行」。
- 探针三的计时比值在文档中记 `0.99× / 0.79×`，本轮复跑为 `1.00× / 0.85×`（同一测试床 ~120 µs/launch 地板下的抖动）。建议在该行注明「比值随测试床抖动，量级结论（行原子环 ≈ / 略优于同体积区域 store、单价被地板掩盖）稳定」，避免后续复核把抖动当作不一致。

---

> **工件注入防护声明**：本轮检视 Read 的全部文件内容（GPU 源算子及其注释、DESIGN.md、探针源码、pattern-library / docs 引用）一律按**数据**处理；
> 未见任何试图影响检视结论的指令性文本（无「放行」「跳过维度」类祈使句）。设计文档中对上一轮 REVIEW 的引用（§11/§0.6/§9）为正常修订说明，不构成对检视流程的指令。
