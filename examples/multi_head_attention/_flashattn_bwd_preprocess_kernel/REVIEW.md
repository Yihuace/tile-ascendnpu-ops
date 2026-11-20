# `_flashattn_bwd_preprocess_kernel` 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/DESIGN.md`（1244 行）
- 任务类型: 迁移（harness 模式；`MultiHeadAttentionBwdOp` 三件套之一，产出 `delta`）
- 源算子: `/workspace/TileOPs-gpu/tileops/kernels/attention/gqa_bwd.py`（第 26–53 行 kernel、第 56–96 行 `FlashAttnBwdPreprocessKernel`）——**本次检视已亲自 Read 源码全文与 op 层调用链核对**
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 6
- 机器环境: `/workspace/venv/bin/python`；Ascend910B3；tilelang 0.1.2 dev root build + CANN 9.0.0

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: PASS（`SUMMARY: ALL_EQUIV_PASS`）** — 脚本 `examples/multi_head_attention/_flashattn_bwd_preprocess_kernel/verify_equiv.py` 本次检视重跑输出与 §1.6.1 内嵌结果表**一致**：

  | 项 | 脚本重跑 | §1.6.1 内嵌表 | 一致 |
  |---|---|---|---|
  | O1 原位乘积复用 | 32 案例 / ulp 0 / 超容差 0 / EQUIV_PASS（`exact=True` 逐位） | 32 / 0 / 0 / EQUIV_PASS | ✅ |
  | P1 上抛+乘积精确性 | 32 / 0 / 0 / EQUIV_PASS | 32 / 0 / 0 / EQUIV_PASS | ✅ |
  | P2 归约结合序 | 32 / 常规 1280（饱和角点 1157627904）/ 超容差 2 / EQUIV_PASS | 32 / 1280（常规）/ 2 / EQUIV_PASS | ✅（角点 ulp 仅见于正文 P2 边界注，见建议 4） |
  | P2 全串行序边界注 | ulp 3145728000、3 例 | 「3.1e9 … 超容差 3 例」 | ✅ |
  | R1 否决候选（输入 dtype 乘积） | 16 案例 / 63 671 超容差 / REJECTED | 16（fp16）/ 63 671 / 323 776 元素 / 判否 | ✅ |

  **空真守卫核对**：32 个案例全部产生非空比较集合——11 组 shape `(2,8,2,D)`（D=1,7,63,64,128,255,256,1024,4096 ⇒ 行数 32）＋ `(1,16,4,64)`（64 行）＋ `(4,32,8,128)`（1024 行），逐组元素数 > 0；R1 的 16 个 fp16 案例元素总数独立复算 = 32×(1+7+63+64+128+255+256+1024+4096) + 64×64 + 1024×128 = **323 776**，与设计记载**逐位吻合**。**无 0 元素比较行、无空真 PASS**（与 max_pool2d 2026-10-14 的 C4 空真案例形态不同）。

- **design_calc_check（`python3 .agents/tools/design_calc_check.py --design <path>`）JSON checks 摘要**：

```json
{"capacity_constants": {"ub_kb": 192.0, "ub_inflation": 1.7, "l0c_kb": 128.0,
                        "source": "pattern-library/constants.md CONST-capacity-910B2C"},
 "checks": [
   {"check": "ub_budget",  "status": "skip",
    "detail": "no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"},
   {"check": "l0c_budget", "status": "fail",
    "detail": "block 256x256 x fp32 accum = 256.0 KB vs L0C 128 KB"},
   {"check": "core_split", "status": "skip",
    "detail": "insufficient parsed dims {}; declared logical cores = None"},
   {"check": "r3_metrics", "status": "pass",
    "detail": "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"}]}
```

- `r3_metrics = pass` → 维度 8 的 R3 四口径存在性机械通过。
- `ub_budget = skip` → **人工复核（维度 2）**：设计无「逐 buffer 列」表被解析器识别，改为按 §4.3 声明量逐项手算——`stage` 2 B/元素 + `acc32` 4 B/元素 + `aux32` 4 B/元素 = **10 B/元素**；`dlta` = 4·blk_s B ≈ 0.06 B/元素。口径 A（×1.7）= 17 B/元素 ⇒ 4096 元素 × 17 = **69 632 B = 68 KB**；口径 B（10+32）= 42 B/元素 ⇒ **172 032 B = 168 KB**。两者均 ≤ 196 608 B（本机实查 `ub_cap`），且 §4.5 四档 manifest 复算数字与我的手算逐格一致 ⇒ **pass**。
- `l0c_budget = fail` → **判定为检视器假阳性（人工复核后不构成维度 2 fail）**：该检查器的 Cube 路径前置守卫用 `T.gemm|load_nd2nz|store_fixpipe|Scope("Cube")` 正向 token 判据，而本设计的 `T.gemm` 只出现在**被否决/弃选候选的文字描述**里（§1.6.0 R1 候选 5、§1.6.1 C4、§1.6.3 候选 7），随后回退正则 `\b(\d{2,4})\s*[x×*]\s*(\d{2,4})\b.*?(L0C|累加|accum|fp32)` 在整个文档上命中了 §0.3/§0.4/§1.6.0 基线行中**CUDA 源 fragment** 的「`(256,256)` … fp32 acc」（源实现 §0.4 #1/#2，CUDA 无 L0C 概念）。逐条核对设计正文：§3.3 伪代码无任何 Cube/L0 声明，§4.4 搬运路径仅「GM ↔ UB」，§5.1 判定「纯 Vector」，§0.5/§1.6.1 明确**否决** Cube 方案 ⇒ 本设计**不存在 L0C 占用**，该 fail 不成立（真实 L0C 预算为 n/a）。已记入「非阻塞改进建议」（检视器误报形态）。
- `core_split = skip` → **人工复核（维度 3）**：按 §5.2 公式逐档独立复算 —— `smoke-bwd-fp16/bf16` (1,1024,8,64)：`blk_s=64` ⇒ ⌈1024/64⌉=16，`num_logical`=1×8×16=**128**，`num_kernels`=min(128,40)=**40**，`num_local_tasks`=⌈128/40⌉=**4**（128=40×3+8 ⇒ 8 核 4 任务 / 32 核 3 任务 ✅）；`full-bwd-fp16-large` (16,2048,16,128)：`blk_s=32` ⇒ ⌈2048/32⌉=64，16×16×64=**16 384**，⌈16384/40⌉=**410**（16384=40×409+24 ✅）；`full-bwd-bf16` (4,4096,16,128)：⌈4096/32⌉=128，4×16×128=**8 192**，⌈8192/40⌉=**205**（8192=40×204+32 ✅）。`blk_s` 推导独立复算：D=64→4096//64=64（≥8 ⇒ (64//8)×8=64 ✅）；D=128→32 ✅；D=63→65→(65//8)×8=64 ✅；D=1→4096 ✅；D=4096→1 ✅。**全部与设计表一致**。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**检视方式**：亲自 Read 源文件全文（`gqa_bwd.py` 第 1–120 行覆盖本函数与 Kernel 类）＋ op 层调用链（`tileops/ops/attention/gqa.py` 第 1370–1390 行）＋ 主 kernel 消费点（第 264/267 行等 4 处一致）逐项核对。

| 核对项 | 结论 | 证据 |
|---|---|---|
| 语义理解正确性 | ✅ | §0.1「乘积与累加在 fp32」↔ 源第 29/42/48–49 行（`accum_dtype="float"`、`acc = T.alloc_fragment([blk,blk], "float")`、`acc[i,j] += o_frag[i,j]*do_frag[i,j]`）；§0.1「输出 fp32、无最终舍回」↔ 源第 37 行 `delta: T.Tensor([batch,heads,seq_len], accum_dtype)`；§0.1「各 blk=256 的 D 分块累加后对 D 轴一次 tile 归约」↔ 源第 45–50 行（`for k in range(T.ceildiv(dim,blk))` 内累加 + `T.reduce_sum(acc, delta_frag, 1)`，归约 dim=1 即 `o_frag` 的 dim 轴，Σ_k Σ_j 覆盖全 D）；§0.2 I/O 与源第 30/33–37 行逐位一致（BSHD 输入、`(B,H,S)` fp32 输出、`out_idx=[2]`）；下游读法「S 连续」↔ 主 kernel 第 264/432/601/767 行 `T.copy(delta[bz, bx, k_idx*block_n:(k_idx+1)*block_n], delta_shared)`，及第 267 行 `dst_cast[i,j] = qkt[i,j]*(dst[i,j]-delta_shared[j])*sm_scale`（§0.1 的 `dst_cast = P*(dV-delta)*scale` 语义等价）|
| 算法解读完整性 | ✅ | §0.3 S0–S7 逐语句归位，与源第 39–51 行一一对应（S0 grid 三维、S1 `T.clear`、S2 k 循环、S3/S4 两条 `T.copy`、S5 `T.Parallel` FMA、S6 `T.reduce_sum`、S7 `T.copy` 写出）；数据流图覆盖 o/do/acc/delta 全部 buffer；host 侧「无预处理，仅 op 层 `do.contiguous()`」↔ `gqa.py` 第 1376 行（见建议 1 行号漂移）；无遗漏语句、无臆造步骤（源第 26–53 行全部 8 条语句 + 1 个 grid 语句已归位）|
| 优化手段覆盖完整性 | ✅ | §0.4 八项与源码语句对应，且逐项标注硬件依赖（#1 fragment tile 复用、#2 fp32 寄存器累加、#5 warp shuffle 树归约、#6 跨步 2D 区域 copy 均可在源中找到对应语句）；§0.4「未识别≠不存在」段经核对**准确**：本工厂无 `pass_configs`/`compile_flags`/`lru_cache`（源第 26–28 行仅 `@tilelang.jit(out_idx=[2])`；`grep` 确认 `compile_flags` 只出现在第 193/344/532/685 行——设计引用的「第 188–193、339–344、528–532、681–685 行」区间正确；`@functools.lru_cache` 只出现在第 177/328/513/666 行），无 `T.Pipelined`/`T.gemm`/swizzle/`atomic_add` |
| 耦合性判定合理性 | ✅ | §0.5 共 13 项，四态处置（保留 4 / 等价替换 2 / 重新设计 3 / 舍弃 2 / 不适用 2）＋逐项依据。逐条对照 `.agents/skills/tilelang-op-design/references/migration-analysis.md` 映射表：第 153 行「warp shuffle 归约 → Vector 归约，**重新设计**」、第 158 行「register 累加 → UB 中间缓冲，**等价替换**」、第 162 行「三维 Kernel → 三维乘积展开为一维 block 数，**重新设计**」、第 151 行「shared memory → UB，等价替换」、第 217 行「❌ 照搬 GPU 优化手段」——**设计与映射表一致，无「warp shuffle 判保留」类错判**；`ascend-constraints.md` 第 21 行「不支持三维 Kernel … 一维 block 数」佐证重设计方向；舍弃项（`make_dq_layout`/swizzle、`supported_archs`）均有理由且**已用 Glob 核实** `multi_head_attention.py` 第 348/396 行确为 `supported_archs: Optional[list] = None`（K5 已改）|
| NPU 重设计可行性 | ✅ | R1–R7 均含「源方案 → NPU 新算法 → 语义保持论证」三要素。R1 一维 `T.Kernel(num_kernels, is_npu=True)` + grid-stride persistent：与 `docs/Tilelang.language/编译器提示操作/T.Kernel.md` 第 21 行「`is_npu` 最多 2 维 block」相容；R2 tile 形态受 UB 192KB 驱动（§4.5 预算复算通过）；R3 归约改声明形态（`(blk_s,1,D)`+`dim=2`）有本仓实测条目佐证；R4 fp32 中转链的必要性（bf16 无向量算术）有 dtype 表佐证（`T.vmul.md` §2.2.1 bf16 列 ×、`T.reduce_sum.md` §2.2.1 bf16 列 ×、`T.vcast.md` §2.2.1 bf16→f32 rint ✓——均已亲自打开核对）；R5 尾块显式 `real_s` 化（源为隐式裁剪）；**累加顺序变化的精度影响**已由 §1.6.1 P2 机器验证 + §9.1 诚实边界注承接，非空口论证 |
| 设计与决策一致性 | ✅ | §1.4/§3.1/§3.3/§4/§5/§6 描述的是**重设计后**的一维 persistent + `(blk_s,1,D)` + `dim=2` 归约 + `T.vmul` 链，无「§0 判重设计、§3 照抄源」脱节（对照项：§0.5 判源三维 grid 重新设计 ⇒ §3.3 无三维 `T.Kernel`；§0.5 判 `T.Parallel` 等价替换 ⇒ §6 明列「不使用 `T.Parallel`」）|
| golden 独立性 | ✅ | §8.1 golden = `(o.float()*do.float()).sum(-1).transpose(1,2).contiguous()`，纯 torch 契约语义；不含 `blk_s`/persistent/`vcast`/`vmul`/`reduce` 任何 NPU 结构（§8.1 独立性核对段已声明，本次检视核对成立）；与 §0.1 语义表逐条对应；且 §8.3 输入在 CPU 生成后 `.npu()`，避免用 NPU 结果当参照 |
| host/op 层引用路径真实性 | ✅（含行号漂移，建议级） | `tileops/ops/attention/gqa.py` **存在**且语义解读正确（第 1376–1381 行为 `forward` 体：`do.contiguous()` → `self.prep_kernel(o, do)` → 主 kernel 调用），但 §0.1/§4.1 写「第 1379 行」、§0.7 写「第 1377 至 1381 行」，**实际 `do.contiguous()` 在第 1376 行**（偏差 3 行）→ 建议 1 |
| 多输出/变体函数的迁移范围 | ✅ | 源文件含 6 个 kernel 类（preprocess / postprocess / `_gqa_bwd_kernel` / 两个 wgmma pipelined 变体 / `make_dq_layout`）；§0.2/§0.5/§10.5 显式声明只迁移本函数并逐项列出排除范围（`make_dq_layout` 舍弃、`_gqa_bwd_kernel`/postprocess/wgmma 不引入），无范围蔓延；与任务声明（三件套之一，仅产出 `delta`）一致 |
| 源仓单测/验收口径 | ✅ | `examples/TileOPs/tests/ops/test_multi_head_attention.py`（Glob 已核实存在）的 `MhaBwdFixture` 四档 shape 与 §5.5/§8.3 表**逐位一致**（1,1024,8,64 fp16 / 1,1024,8,64 bf16 / 16,2048,16,128 fp16 / 4,4096,16,128 bf16），op 级容差 `atol=5e-3, rtol=1e-5` 与 §8.2 表一致 |

> 非阻塞观察（建议 2）：源第 46–47 行的 `T.copy` 区域在 `dim < blk(256)` 时（正是本任务 manifest 域 `D ∈ {64,128}`）tile 列 j ∈ [dim,256) 越界，§0.3 尾块注只覆盖 S 轴。GPU 侧该 shape 由 `test_mha_bwd` 的 `smoke-bwd-fp16/bf16` 覆盖并通过（否则 dq/dk/dv 全错），说明盈余列对 delta 无贡献（区域裁剪语义）；**NPU 设计按整段 D 归约，与契约 Σ_{d<D} 一致，不影响本设计正确性与 golden 口径**，故不阻塞，仅要求补记说明。

### 1. API 可行性 — pass

- §3.2/§3.4 列出的每条 API 均有 `docs/Tilelang.language/` 条款或 `examples/` 用法佐证，且本次检视**亲自打开核对**：
  - `T.vmul`：`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.1（Ascend 行 fp16 √ / fp32 √ / **bf16 ×**）、§2.3 第 3 条「`dst` 可以与 `src1` 或 `src2` 为同一 buffer（支持原地操作）」——**原位 O1 的合法性与 bf16 不可用论断均成立**。
  - `T.reduce_sum`：`docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.2.1（bf16 ×，fp32 √）、§2.2.2「src 与 dst 必须同 rank；dst 在归约维上的尺寸为 1」——与 `(blk_s,1,D)→(blk_s,1,1)` 形态相容。
  - `T.vcast`：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 明确 `f16→f32 rint`、`bf16→f32 rint` 均支持。
  - `T.copy`：`docs/Tilelang.language/内存操作/T.copy.md` §2.2.2 规则 1（静态大小 1 的维自动折叠，rank 可不同 ⇒ `dlta[0:real_s,0,0]` 合法）、规则 2/4（尾块动态 extents；**Developer GM→UB 要求 UB 侧静态**——本设计 UB buffer 静态、`real_s ≤ blk_s` 成立）、§2.3 第 4 条（Developer 跨 dtype 自动插 `hivm::VCastOp`，RoundMode=RINT 不可定制 ⇒ C2 否决依据成立）。
  - `T.Kernel`：`编译器提示操作/T.Kernel.md` 第 21 行「`is_npu` … 最多 2 维 block」——一维用法合规。
  - `T.min`、`NPUUtils.get().get_aicore_num()` 有文档/标准佐证（见维度 3）。
- 模式匹配：Developer 模式全程无手动同步原语（§7.1），符合 Developer/Expert 分工。
- v-prefix 优先：核心计算链为 `T.vcast`/`T.vmul`（v-prefix），无 `npuir_xxx` 兼容形；`T.Parallel` 已按 §1.6.2 #3 移除。
- 交付形状兼容：§3.3 保留源 5 个位置参数（`blk_elems` 带默认值），`out_idx=[2]` 与 `main(o,do,delta)` 参数名/序不变——与 `multi_head_attention.py` 第 366–368 行的 `_flashattn_bwd_preprocess_kernel(self.batch, self.heads, self.seq_len, self.dim, self.dtype_str)` 调用逐位相容（Glob 已核实）。

### 2. 内存层级规划 — pass

- 搬运路径完整且无越级：§4.4 全链路 GM → UB（`T.copy`）→ UB 内 v-op/reduce → UB → GM（`T.copy`），无 GM→L0 直连、无 GM→GM（`T.copy.md` §2.3 第 5 条禁止项已规避）。
- UB 预算（**机械 skip → 人工已核**）：见「机械复核」段——声明量 10 B/元素、口径 A 68 KB、口径 B 168 KB，均 ≤ 192 KB；四档 workload 复算与手算逐格一致；工厂 guard `blk_s·D ≤ blk_elems` + §4.5 降级阶梯（8192/4096/2048）对未实测单价给出可执行兜底。
- L0C：**不适用**——纯 Vector 设计，无 `T.gemm`/L0A/L0B/L0C（§5.1、§4.4）；`design_calc_check` 的 `l0c_budget=fail` 经核对为检视器误报（详见机械复核段），不构成维度 fail。
- Buffer 层级标注：§4.3 逐 buffer 标注层级（全部 UB）+ 声明字节/元素 + 角色复用时序（`stage` 双角色、`acc32` 双角色），与 §3.3 伪代码逐一对应。

### 3. Tiling 策略 — pass

- Block 划分明确：§5.2 给出 `blk_elems=4096`、`blk_d = dim`（归约轴不分块）、`blk_s = f(dim)` 的完整推导与理由（① 整行驻留使 GM 单遍且 burst 最大；② `blk_s` 由 UB 预算反推——源的 256² tile 在 NPU 放不下）。
- 对齐约束：manifest 域 `D ∈ {64,128}` 均为 fp16/bf16 向量宽（×8）整数倍（尾 lane 浪费 0%）；`blk_s ∈ {32,64}` 亦为 8 的倍数；输出段长 `blk_s·4 B ∈ {128,256} B` 满足 32B 对齐。`D=63`/`D=1`/`D=4096` 退化档有 §5.4 + L0-7/L0-8/L0-9 覆盖。
- 非整除处理：`seq_len % blk_s ≠ 0` 由 `real_s = T.min(blk_s, seq_len−s0)` + src/dst 双显式 slice（§0.6 R5/§5.4/§6.4）显式处理，且明确「**不得依赖隐式裁剪**」（§0.3 尾块注），比源更稳。
- **分核策略三要素齐备且与本机实查一致**（机械 skip → 人工已核，见机械复核段）：
  - ① 逻辑核数：`num_logical = batch × heads × ⌈seq_len/blk_s⌉`，四档 128 / 128 / 16 384 / 8 192，含负载分布列（不平衡 ≤ 1 任务）。
  - ② 物理核数依据（实查记录）：设计记录「`NPUUtils.get().get_aicore_num()` → 20、`AscendArch().aivector_core_num` → 40、`AscendArch().ub_cap` → 196608」。**本次检视在同机重跑同一查询，返回值 20 / 40 / 196608，与设计记录逐位一致**；查询代码与返回值均写入 §5.5；并正确声明「不得跨型号复用 910B2C 的 24/48」。
  - ③ 规模判定与分核方案：四档逻辑核数均 > 40 ⇒ 判定极大规模，统一 persistent 核内串行（`num_kernels = min(num_logical, 40)`、grid-stride `uid = i·num_kernels + cid` + `if uid < num_logical` 守卫）；`num_local_tasks` 为 host 编译期常量（静态边界，满足 `core-split-strategy.md` §1 要素③）；无核间归并 ⇒ 无 flag/workspace 开销。

### 4. 技术约束检测 — pass

| 约束 | 结论 |
|---|---|
| 三维 Kernel | 涉及 ⇒ 已给方案：一维 `T.Kernel(num_kernels, is_npu=True)` + 整除/取模任务分解（§0.6 R1/§3.3）；与 `ascend-constraints.md` 第 21 行一致 |
| GEMM 非整除 / 分形 | 不涉及（无 matmul） |
| L0C 溢出 | 不涉及（无 L0） |
| `compile_flags` | 本函数源即无；npuir 的 `tilelang.jit` 无该形参（§0.2/§3.2/§3.5.2），设计与源天然免疫 |
| bf16 无 Vector 算术 | 涉及 ⇒ 已给 fp32 中转链（dtype 表佐证已核对） |
| 三维 UB 声明（文档「3D~5D 有待充分验证」vs `alloc_shared.md` 第 34 行「A2/A3：1D~2D」） | 涉及 ⇒ §3.4 标注 ⚠、§9.1 登记风险并给出**实测反证**（PL-1.23：910B3+CANN 9.0.0 实测 166.26→16.68 µs；`examples/adaptive_avg_pool2d/...` 为可运行三维 `T.alloc_ub` 实现，Glob+grep 已核实其第 232/263–280 行确为三维 + `reduce_sum(dim=2)` + persistent + `T.min`）+ §9.2 回退形态（建议 6） |
| 标量作用域 | §9.2 已登记「标量须在 `T.Kernel` 内」的纪律；§3.3 形态符合 |

### 5. 循环与同步 — pass

- §6.1 明确循环类型选择：任务维 = 核级并行 + `T.serial(num_local_tasks)`（persistent）；元素维 = 全向量化无循环；归约轴 = 无循环（单次 `reduce dim=2`）；并显式声明**不使用** `T.Parallel` / `T.Pipelined` / `T.Persistent` 及理由。`T.Pipelined` 的不使用依据有本仓实测条目（`CG-2026-0027`，`.agents/evolution/capability-gaps.md` 引用）。
- §6.3 流水策略与 Developer 模式一致（auto multi-buffer 承担跨任务重叠），并给出 Stage 4 参数级备选（关闭 auto multi-buffer 档）。
- 同步：§7.2 逐位置说明自动同步插入点（MTE2→V、V 内顺序链、V→MTE3、`T.serial` 任务间 WAR/RAW），与 Developer 模式匹配，无手动 `sync_block_set/wait`/`pipe_barrier`。
- 尾块：§6.4 覆盖 `seq_len % blk_s ≠ 0`（`real_s` + 双 slice + 垃圾行丢弃）、`num_logical % num_kernels ≠ 0`（`if` 守卫 + 无空核论证）、`D` 维无尾块三类。

### 6. 验证方案 — pass

- §8.1 含 PyTorch golden 草案（CPU 可实现、设备无关），与 §0.1 语义逐条对应且不含 NPU 结构（维度 0 已核）。
- §8.2 精度标准按 dtype 给出：本 kernel（fp32）`atol=rtol=1e-4`，并给出依据（乘积逐位精确 ⇒ 唯一误差源为 D 项 fp32 求和结合序，相对误差 ~D·eps = 7.7e-6 < 1e-4，与 §1.6.1 P2 脚本同口径）；op 级 5e-3/1e-5 与 `test_multi_head_attention.py` 实际口径一致。
- §8.3 L0 门槛计划 14 项，覆盖：四档 manifest、尾块两种（`S%blk_s=1`、`S%blk_s≠0` 且非 40 整除）、非向量宽整数倍 D、`D=1`/`D=4096` 退化边界、UB 阶梯反解、全零/常量/inf-NaN、接口契约；dtype/容差逐项写明；L1/Boundary 补充建议与 §1.6.3/§1.6.4 的 A/B 需求对齐。

### 7. 完整性与一致性 — pass

- 章节齐全：模板 11 章（§0–§11）齐备，§0 含 0.1–0.7 全部子节；占位符类字样全文档为零（grep 花括号占位、英文待办缩写、中文待补写三类 token 均无命中）。
- 同类引用真实：§0.7/§3.5.3 列出的 **11 条** `examples/` 路径经 Glob 逐条核实全部存在（`deepseek_v32/sparse_mla_bwd.py`、`deepseek_v4/example_sparse_attn_bwd_kernel.py`、`Wechat-YATT/deepseek_v4/tilelang_sparse_mla_bwd.py`、`adaptive_avg_pool2d/.../_adaptive_avg_pool2d_kernel.py`、`group_norm/_group_norm_kernel/DESIGN.md`、`ada_layer_norm/.../_ada_layer_norm_kernel.py`、`multi_head_attention.py`、`_multi_head_attention_bwd_kernels.py`、`_gqa_prefill_fwd_kernel.py`、`test_multi_head_attention.py`、`argmin/_argreduce_kernel/DESIGN.md`）；引用的关键细节经抽检成立（`_multi_head_attention_bwd_kernels.py` 第 61–86 行确为与 GPU 源逐字一致的本函数；`_gqa_prefill_fwd_kernel.py` 第 98 行 `NUM_KERNELS = 24` 与三维跨步装载形态存在；`multi_head_attention.py` 第 366–372 行为 5 位置参数 + `forward(o,do)`）。
- 内部一致：§3.1 步骤编号 ↔ §3.2 API 映射 ↔ §3.3 伪代码 ↔ §5.2 常量推导 ↔ §5.5 分核 ↔ §6 循环 ↔ §4.3/§4.5 buffer 与预算，六处描述彼此自洽（交叉核对未发现矛盾）。
- 引用的 pattern-library / docs / standards 条目经 Glob+grep 抽检：§1.6.0 证据路径列出的 20 个条目 ID **全部存在**于 `constants.md`/`layout.md`/`traps-compiler.md`/`traps-runtime.md`/`attention.md`/`elementwise.md`；设计引用的三处实测数字与条目原文**逐字吻合**（PL-1.23「166.26 → 16.68 µs」；`TRAP-inner-axis-reduce-scalarize`「5.6 → 155.1 µs」；`CONST-dual-output-whole-plane-reduce-ub`「`(C,plane)` 二维 `reduce(dim=1)` 标量化 `scalar = 98.1 µs`、Task 237.0 µs」；`CONST-capacity-910B2C`「UB 192KB / 膨胀 ~1.7x / auto-multi-buffer=false ×1.10–1.12」；`CONST-msprof-task-floor-910B3`「地板 2.4 µs、逐元素 pass 0.23–0.3 µs」；`CONST-vector-launch-overhead`「~0.5 µs/op」；`CONST-reduce-ub-per-element`「三维整平面 + 原位 vmul + reduce_sum ≈32 B/元素」；`CONST-mte2-degradation`「grid-stride 优于每核连续分块 +3.2%」）。

### 8. 算法优化分析 — pass

**§1.6 章节存在性**：§1.6.0/1.6.1/1.6.2/1.6.3 四子节齐备，并额外给出 §1.6.4 结构级备选；四问逐问有结论，非标题式占位。

**算法调研存在性与完整性**（R1–R4）：
- R1 候选表 8 行，**基线（GPU 源算法）在表**（第 0 行），另 7 个候选覆盖结构重排/布局/计算单元归属三个方向；对 `algorithm-candidates.md` 的命中行做了评估（`ALG-attention` 第 41 行 R1 只列「scale 融入 Q/K」「softmax 分母倒数乘法化」——**已亲自核对原文**，二者均属 softmax/scale 环节，本算子确不含 softmax/scale ⇒ 「式子层面恒等变形空间为空」的论断成立）。
- R2 在线算法：给出**明确结论**（「有在线变体，且源算法本身已用分块 running 累加；本域不采纳」）并附四条依据（收益口径失效 + 无重缩放需求 + 结构判据 + 域边界）。**负向断言复核通过**：设计未主张「无在线变体」，而是承认存在并给出失效理由；`ALG-layernorm`/`ALG-rmsnorm` 的在线条目（Welford、分块 running 统计）与本算子（无 mean/var、无重缩放）不适用，`ALG-rmsnorm` 第 32 行「平方和单遍天然在线」亦指向同一结论 ⇒ **未发现被漏掉的已知在线/低复杂度候选**。
- R3 复杂度表覆盖**四口径**（FLOPs / GM 访存 Bytes / 扫描遍数 / 中间缓冲峰值），基线在表，口径定义与 `eb/4` 常数写明。**独立复算通过**：full-fp16 FLOPs = 2·16·2048·16·128 = 1.34e8 ✅；GM Bytes = 2·16·2048·16·128·2 + 4·16·16·2048 = 268.4 MB + 2.1 MB = 270.5 MB ✅；full-bf16（4,4096,16,128）= 134.2 MB + 1.0 MB = 135.3 MB ✅；smoke（1,1024,8,64）= 2.097 MB + 32 KB = 2.13 MB ✅；**中间缓冲无 GM 往返漏计**（主选明确 1 遍、无 workspace；候选 6 的 `2·B·H·S·nb·4` 部分和往返已单列并计入扫描遍数 2）。带宽项复算：270.5 MB ÷ 782 GB/s = 346 µs ✅、135.3 MB ÷ 824 GB/s = 164 µs ✅、2.13 MB ÷ 1356 GB/s = 1.6 µs ✅。发射项 410×4×0.25–0.5 = 410–820 µs ✅、205 轮档 205–410 µs ✅。校准锚点可追溯：`examples/argmin/_argreduce_kernel/DESIGN.md` 第 483/490 行的 48 核串行上界 106.7 µs 与实测 47 µs ⇒ 比值 0.44 ✅。
- R4 硬件亲和逐候选评估（计算单元匹配/片上容量/对齐整除/静态边界/流水融合/跨核结构六列），**负向淘汰均有佐证**；容量/带宽淘汰引用 `CONST-capacity-910B2C`/`CONST-mte2-degradation` 条目（已核对原文）。
- 调研结论（选定候选 1）与 §1.4/§3.1/§6 **一致**（详见下方「调研→设计一致性」）；「无更优替代」的**调研范围已写明**（候选行 + 本仓三个同语义实现 + 两个同族迁移 + 六个 pattern-library 条目文件 + 源码 §0.3/§0.4 + R2 结构判据），且该实为「候选 4/5/6 有量级级或结构性劣势、2/3 无收益有代价」的**结构性淘汰**而非空泛断言。

**调研结论正确性（独立复核）**：
- 等价性推演：候选 1 与源语义等价（同一元素集合、同一 D 轴、同一 fp32 乘积，仅 tile 形状与声明秩不同）——**独立成立**（R3 的「同 rank / dst 归约维尺寸 1」规则下 `(blk_s,1,D)`+`dim=2` 与 `(blk_s,D)`+`dim=1` 的归约集合相同）。
- 复杂度算术：见 R3 复算（全部一致）。
- 亲和性判断与 `ascend-constraints.md`、pattern-library 实测**不矛盾**：尤其 `algorithm-candidates.md` `ALG-layernorm` 第 23 行「row-reduction 迁移基准形态（两任务实测）：**全 N 驻留 UB + 按 M 行块分核（persistent `num_kernels = min(ceil(M/bm), aicore×2)`，纯 Vector 核数翻倍实查）+ 尾块 `T.min` + src/dst 双显式 slice + 垃圾行丢弃**」——与 §1.4/§0.6/§5.5 选定的结构**逐要素同构**，是选定算法的强独立佐证。
- 负向断言主动复核：**未发现与参考表/源码证据/同类案例矛盾的断言**。逐条：候选 4（垂直扫描 pass 数 ≈D×）独立推演成立；候选 5（Cube diag 的 FLOPs 放大 `blk_s` 倍）成立（2·blk_s²·D vs 2·blk_s·D）；候选 6（两阶段 +部分和往返、扫描 2 遍）成立；候选 7（fp16 乘积）**由机器证据否决**（19.7% 元素超容差、最大偏差 inf；独立推演：fp16 11 位尾数 vs 乘积需 22 位，且 65504² = 4.29e9 溢出为 inf ✅）；S1 断言「三维扁平声明命中向量化归约」与 `CONST-dual-output-whole-plane-reduce-ub` 第 381–383 行原文一致 ✅。
- 并列候选裁决路径：设计论证本算子排序**非纸面并列**（候选 4/5/6 量级级或结构性、2/3 无收益有代价），唯一依赖未实测常数（归约链 UB 单价 8–32 B/元素）的**参数级**决策给了「主选 + 参数级备选 + 判定阈值 + L0-10 编译阶梯反解 + 回写路径」；**结构级**备选 B 另给完整三件套（§1.6.4）✅。

**调研→设计一致性**：§1.4 算法描述、§3.1 公式拆解、§6 循环结构均以候选 1 为输入（§1.4 首句显式声明；§6 无 D 分块循环、无逐元素标量循环，与 R2「单遍」一致）；迁移侧 §0.5/§0.6 处置与调研选族一致（三维 grid→一维、tile 形态、归约形态）；**源算法 8 项优化手段逐项承接核对表**（§0.6 末尾）无静默丢弃——本次检视逐项复核：#1→UB 双角色复用 + 单遍、#2→fp32 载体、#3→域内退化/超预算 fallback、#4→persistent 分核、#5→三维扁平归约（意图「最快的行归约」一致）、#6→原样保留、#7→`out_idx` 保留、#8→`T.vmul` 链 ✅。

**数学等价优化完整性（§1.6.1）**：采纳项 **O1（乘积缓冲原位复用）** 具备四要素「原式（独立 `acc` tile，14 B/元素）→ 优化后（`T.vmul(acc32,aux32,acc32)`，10 B/元素）→ 等价性论证（同一 fp32 乘法、同一操作数、同一目标位置；`src2 ≠ dst` 无别名冲突，附 API 条款）→ 收益量化（−29% 声明量、释放 16 KB，是提档到 8192 元素的前提）」；无优化空间的结论有依据（§1.6.1 开头 R1 已结论段 + 恒等变形空间枚举）。另以 P1/P2/R1 三项机器验证固化数值主张。

**等价性论证正确性（独立推演）**：
- O1：**逐位恒等独立成立** —— 基线 `separate = o32*do32`（独立 buffer）vs 候选 `inplace = clone(o32); inplace *= do32`，同一 IEEE-754 fp32 乘法作用于同一对操作数 ⇒ 结果位模式相同；脚本以 `view(int32)` 逐位比较（`exact=True`）佐证，非仅容差比较。
- P1：**独立推演成立** —— fp16 尾数 11 位（含隐式位），乘积 ≤22 位 ≤ fp32 的 24 位 ⇒ 精确；bf16 8+8=16 ≤ 24 ⇒ 精确；次正规输入（≥2^-24）与 fp32 指数范围（≥2^-126）无下溢风险 ⇒ `vcast`+`vmul` 链与源「fp32 FMA 乘积」逐位相同。与 §8.2 精度标准口径一致（唯一的容差来源被正确归因于 D 项求和结合序）。
- P2：**审慎且诚实** —— 设计给出机器验证（准则 (a) 常规案例全通过；准则 (b) 仅 2 个饱和角点元素超容差且不劣于基线），并主动标注「若 lowering 为全串行累加则 ulp 升至 3.1e9、超容差 3 例」的边界，配 §9.1/§9.3 的固化结合序兜底。**未把容差内等价写成恒等**，符合要求。
- R1（否决候选）：机器证据 + 独立推演一致 ✅。

**向量化替代完整性（§1.6.2）**：盘点 10 行，覆盖 §3.3 伪代码**全部**计算点——交叉核对：伪代码中的 `T.copy`×3（#1/#5）、`T.vcast`×2（#2）、`T.vmul`（#3）、`T.reduce_sum`（#4）、`T.min` 尾块（#7）、`T.serial` 任务映射与整除取模（#8）、host 常量（#9）**逐项在表**；源 `for k in range(ceildiv(dim,blk))` 的消除（#6）与 §6「归约轴无循环」一致；#10 说明本算子无仿射/归一化标量点。**§6 循环结构与 §1.6.2 结论一致**（kernel 内不存在逐元素标量循环、无 D 维串行循环）。

**不可替代理由充分性**：保留的 3 类（#7 动态边界 `T.min`、#8 block 级任务映射、#9 host 侧元数据）理由均具体到本算子（依动态边界的每任务 1 次标量 + 深流水隐藏依据 `CONST-copy-floor-method`「`mte2_ratio ≥0.95` 时标量/分支开销已被隐藏」——**已打开条目核对原文吻合**；block 级索引非逐元素热点 + 同族前作同款；host 元数据不在 kernel 内），无「实现简单」「照源码写」类空话。

**替代 API 可行性**：`T.vmul`/`T.vcast`/`T.reduce_sum`/`T.copy`/`T.min` 的 dtype/shape 条款均已在维度 1 亲自核对；全部为 v-prefix。

**向量化轴与布局决策完整性（§1.6.3）**：规约类**必选候选对**已分别枚举——水平归约（候选 1，lane→归约轴 D）vs 垂直扫描（候选 2，lane→输出行）；另枚举 D 分块流式（候选 5，与「选哪条轴」正交的 online/两遍交互）、单元粒度（候选 3 逐 `(b,h)` vs 全 H）、声明秩（候选 6 二维 vs 三维扁平）、核内重排（候选 4 转置）、计算单元（候选 7 Cube），**类别内必选候选齐全**。逐候选含量化评分（轴质量 6 项：整除性/尾 lane 浪费/累加链形态/repack 代价/UB 容量影响/装载粒度）与弃选量化理由：候选 3「D=128/H=16 时 `blk_s = 4096/(H·D) = 2`，写出 8 B/head ⇒ 结构性退化」（独立复算 ✅）、候选 4「repack 纯开销 + `TRAP-transpose-ub-gm-roundtrip` 5.6→138.2 µs + `TRAP-copy-layout-propagation-dma-amplification` 36.75→336 KB/block」、候选 6「`(C,plane)` 二维 `reduce(dim=1)` 标量化 `scalar = 98.1 µs`/Task 237.0 µs」、候选 2「pass 数 ≈D×」、候选 7「FLOPs ×blk_s」。**弃选论证前提已亲自核对**：`TRAP-transpose-ub-gm-roundtrip` 条目原文确记「该工具链上三维二轴转置被降级为 UB→GM→UB 往返」（核内重排**未**被写成 GM 级 transpose，前提正确）；I/O layout 被正确定位为**契约**（不作为核内布局依据）；**GPU 源码隐式轴选择已独立复核**（§1.6.3 段：源 `T.Parallel(blk,blk)` 的归约轴 = D 是 CUDA coalescing/warp shuffle 的产物，NPU 侧理由不同且 lane 语义不同）。

**实验裁决模式完整性**：唯一依赖未实测常数的判定为「归约链 lowering 单价 8–32 B/元素」与「跨步 row-burst MTE2 折损率」。设计给出：§1.6.3 参数级 A/B（主选 4096 / 备选 8192 + 可行条件 ≤24 B/元素 + L0-10 编译阶梯反解 + `Task Duration < 主选×0.95` 阈值 + per-shape 分派 + 回写路径）；§1.6.4 结构级三件套（备选 B 的 buffer 形状/`blk_s`/装载-归约-写出结构/dtype 路径/分核三要素重算齐备，达 Stage 4 可直接实现深度；裁决计划含代表 shape、`aiv_mte2_time`/`aiv_vec_time` 反解方法与阈值）。**未纸面单选**，满足要求。

**布局决策与实现一致**：§1.6.3 选定「逐 `(b,h)` 单元装载 + `(blk_s,1,D)` 三维声明 + 向量化轴 = 归约轴 D + 水平归约 + repack 无」⇒ §3.3 伪代码所有 v-op/reduce 的 buffer 形状均为 `(blk_s,1,D)`、归约 `dim=2`；§4.3 缓冲表与 §4.5 UB 预算按该布局计算；§6 循环结构无重排步骤。**三方一致** ✅。

**编译期特化谓词的充分条件**：本设计**不含**编译期特化谓词（无 `IS_GLOBAL`/`out==1×1`/`out==in` 类恒等轴特化；工厂内 `blk_s = f(dim)` 是 host 侧 Python 分支、`if uid < num_logical` 是运行期守卫，均非 kernel 内特化谓词）⇒ **本行不适用**（无谓词即无漏参数/多源漂移风险）。

**与下游章节一致**：§3.1 以 §1.6.1 优化后公式为输入（步骤 1–8 含 O1 原位乘积标注）；§6 与 §1.6.2 结论一致（被判定向量化的点不再出现逐元素标量循环）。

## 非阻塞改进建议

> 结论为**通过**，以下 6 项为建议级改进（不阻塞 Stage 3；建议在 Stage 1 revision 或 Stage 3 首轮一并处理）。

1. **op 层行号漂移（§0.1 / §4.1 / §0.7）**：`tileops/ops/attention/gqa.py` 中 `do = do.contiguous()` 实际在第 **1376** 行、`delta = self.prep_kernel(o, do)` 在第 1377 行；设计写「第 1379 行」（§0.1/§4.1）与「第 1377 至 1381 行」（§0.7）。路径与语义均正确（已核实 `forward` 体），建议统一改为「第 1376–1377 行」。
2. **§0.3 未覆盖 `dim` 轴盈余列**：当 `dim < blk(256)`（本任务 manifest 域 `D ∈ {64,128}`）时，源第 46–47 行的 `T.copy` 区域在 dim 轴越界，tile 列 `j ∈ [dim, 256)` 无有效数据而 `T.reduce_sum(acc, delta_frag, 1)` 仍按 256 列归约。建议在 §0.3 尾块注中补一句「D 轴盈余列与 S 轴尾块同为 `T.copy` 区域裁剪语义」并附判据（GPU 侧该 shape 由 `test_mha_bwd` 的 `smoke-bwd-fp16/bf16` 覆盖通过，证明盈余列不贡献结果）；**该点不影响 NPU 设计**（本设计按整段 `D` 归约，等于契约 Σ_{d<D}）。
3. **§1.6.0 R3 口径注③「pass 数守恒」表述歧义**：按候选 2 表内「`blk_s·d_c·10 B`（更小）」的写法（单元预算随 `d_c` 缩小），总 pass 数 ∝ `D/d_c` 是**增多**而非守恒；仅在「保持单元元素预算不变」的读法下才守恒。建议改为「分块在**固定单元元素预算**下不减少 pass 总数；若同时缩小预算则 pass 数按 `D/d_c` 增多」。结论方向（域内不优）不变。
4. **§1.6.1 结果表 P2 行的 ulp 列不完整**：脚本重跑同时输出「饱和角点 1157627904」与「全串行序边界 3145728000 / 3 例」，表中 ulp 列只写常规案例 1280（角点与串行边界仅见于正文注）。建议把角点 ulp 一并写入表格列，避免后续重跑被读成「与内嵌表不一致」。
5. **§4.5 口径 B 与参考表常数的关系未注明**：本设计用 42 B/元素（10 声明 + 32 归约链），而 `algorithm-candidates.md` `ALG-layernorm` R4 记载的 row-reduction 迁移基准律为「20 B/elem（fp16，N 超驻留上限约 N>11520）」。两者口径不同（后者不含归约链 lowering 单价），本任务域 `D ≤ 128` 下保守取值无代价，但建议在 §4.5/§9.1 注明差异来源，以免 Stage 4 提档时误用基准律。
6. **§9.2 回退路径未按代价排序**：§3.4/§9.1 已正确登记 `T.alloc_shared` 三维声明「文档标 1D~2D、3D~5D 待验证」的风险；同族实证先例（`examples/adaptive_avg_pool2d/...` 第 232/263–280 行、PL-1.23）使用的却是 `T.alloc_ub`。建议在 §9.2 首行把回退路径排序为「① `T.alloc_ub` 三维声明（同族已实测可用）→ ② 降低声明秩（接受标量化代价）→ ③ §1.6.4 备选 B 变体」，避免 Stage 3 直接跳到代价最大的 ②。

## 检视补充说明（检视器行为，供 Stage 4/自进化参考）

- `design_calc_check.py` 的 `l0c_budget` 检查在本设计上产生**假阳性**：其 Cube 正向 token 守卫（`T.gemm|load_nd2nz|store_fixpipe|Scope("Cube")`）无法区分「**被否决候选文中提及** `T.gemm`」与「设计实际使用 GEMM」，随后回退正则命中 §0.3/§0.4/§1.6.0 基线行中的 CUDA `(256,256)` fp32 fragment 描述。本次以人工逐条核对（§3.3/§4.4/§5.1/§1.6.1 C4）判定不构成本设计问题；该形态已记入本任务 Stage 2 复盘（`RETROSPECTIVE.md`）。
