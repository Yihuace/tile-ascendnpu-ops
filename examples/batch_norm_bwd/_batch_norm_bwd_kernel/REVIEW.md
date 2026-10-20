# _batch_norm_bwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

> 本轮为 **revision v1 复检（RE-REVIEW）**：v0 判「不通过」（2 阻塞 + 3 建议），v1 作定点修订。复检确认 **两个阻塞项均已真实修复**，且 v0 的 3 个建议项全部落实，未引入新的阻塞级问题。

## 检视元信息
- 文档: examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd_kernel/_batch_norm_bwd_kernel_DESIGN.md（revision_index 1 / revision v1；上一版 history_version/design_v0.md）
- 任务类型: 迁移（harness 模式）
- 源算子: /mnt/workspace/TileOPs-main/src/tileops/kernels/norm/batch_norm.py（`_batch_norm_bwd_kernel` L1487 + host `BatchNormBwdKernel` L2146 + 配置源 `_TiledPath` L66）；Stage 0 提取件 `examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/_batch_norm_bwd_kernels.py`（L357）；NPU wrapper `examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd.py`（L327/`forward` L383）；op `examples/TileOPs/tileops/ops/norm/batch_norm_bwd.py`；探针 `examples/batch_norm_bwd/probe/PROBE.md`
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0

## 机械复核（D-1/D-5）

### verify_equiv 重跑（D-1）
- 命令: `python examples/batch_norm_bwd/_batch_norm_bwd_kernel/verify_equiv.py`（torch CPU、fp64 参照 + 角点值），退出码 0。
- 结果（采纳项 O1+O2 联合，21 案例 = 3 dtype × 7 shape）：

```
O1+O2 (joint)  float16  |  max ulp 0.00 | 违反率 0.0000 | EQUIV_PASS
O1+O2 (joint)  bfloat16 |  max ulp 0.00 | 违反率 0.0000 | EQUIV_PASS
O1+O2 (joint)  float32  |  max ulp 0.00 | 违反率 0.0000 | EQUIV_PASS
总体结论: 采纳项 O1+O2 全部 EQUIV_PASS；raw-moment 式被弃选 — 可采纳
```

- 逐案 21 行全部 `EQUIV_PASS`（max ulp=0.00、违反率=0.0000），与 DESIGN §1.6.1 内嵌结果表（三 dtype 全 0）**逐格一致**。
- 判别段 A（raw-moment `gw` 拒绝证据）：`float32, N=4, S=2048` 违反率 = **0.1553**（`DISCRIM_OK`），fp16/bf16 = 0.0000（宽容差掩蔽）——与 §1.6.1 内嵌表 0.1553 一致，非纸面论证。

### design_calc_check 重跑（D-5）
- 命令: `python3 .agents/tools/design_calc_check.py --design examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd_kernel/_batch_norm_bwd_kernel_DESIGN.md`

```json
{
  "design": "examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd_kernel/_batch_norm_bwd_kernel_DESIGN.md",
  "capacity_constants": {
    "ub_kb": 192.0,
    "ub_inflation": 1.7,
    "l0c_kb": 128.0,
    "source": "pattern-library/constants.md CONST-capacity-910B2C"
  },
  "checks": [
    {"check": "ub_budget", "status": "skip", "detail": "no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"},
    {"check": "l0c_budget", "status": "skip", "detail": "no affirmative Cube-path API (T.gemm/load_nd2nz/store_fixpipe/Scope(Cube)) in this design"},
    {"check": "core_split", "status": "skip", "detail": "insufficient parsed dims {'N': 1}; declared logical cores = 40"},
    {"check": "r3_metrics", "status": "pass", "detail": "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"}
  ]
}
```

- **skip 项人工复核结论（机械跳过，人工已核）**：
  - `ub_budget`（人工已核）——按 §3.3/§5.2 的 `BPE·bN·bS` 公式（fp16/bf16=22、fp32=26，§4.5 同族实测膨胀值）独立复算：held 档最大 resnet-stage1 `8·1024·22 = 180,224`；tiled 档大 C `16·512·22 = 180,224`、large-spatial `2·4184·22 = 184,096`——均 ≤ `UB_SAFE_BYTES = 184,320`（180 KiB）且 ≪ 192 KiB 物理容量（192,000B 十进制 / 196,608B 二进制）。**UB 由公式结构性有界，无溢出**。
  - `l0c_budget`（人工已核）——纯 Vector，无 Cube/L0C/L1，不适用（§3.5.1/§4.4/§5.3）。
  - `core_split`（人工已核）——`num_kernels = min(C,40)`、`num_local_tasks = ceil(C/40)`（64→40/2、128→40/4、256→40/7、2048→40/52），`40 = NPUUtils.get().get_aicore_num()` 实查 20 × 2（纯 Vector 翻倍，§5.5）；核内 serial 边界为 host 编译期常量，算术自洽。
  - `r3_metrics`——pass，与 §1.6.0 R3 四口径表一致。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

检视方式：亲自 Read 源算子 `/mnt/workspace/TileOPs-main/src/tileops/kernels/norm/batch_norm.py`（`_batch_norm_bwd_kernel` L1487–1639、host `BatchNormBwdKernel` L2146–2212、`_TiledPath` L66–115）、`ops/norm/batch_norm.py`、NPU wrapper `.../batch_norm_bwd/batch_norm_bwd.py`（`BatchNormBwdKernel.forward` L383–384）、探针 `PROBE.md`，逐项与 §0 核对。

- **语义理解正确性 — pass（源码证据）**：§0.1 三式与源码逐句一致——`grad_bias[c] = Σ grad_out`（源码 L1600/L1603）、`grad_weight[c] = Σ grad_out·x_hat`（L1596/L1604）、`grad_x = w·rstd/L·(L·g − gb − x̂·gw)`（L1607 系数 + L1615–1617 写出）、`x_hat = (x−mean)·rstd`（L1564/L1574）；元素定位 `l → [l//S, c, l%S]`（源码 docstring L1505 + L1556/L1570 `j//S, bc, j%S`）与 §0.1「沿展平轴 `N·S`、通道轴为中间轴（逐通道平面 GM 行跨步 `C·S`、行内 stride=1）」一致。**dtype 语义**「fp32 opmath + 单次舍回」与 `accum_dtype = "float32"`（L1515）、`T.cast(..., dtype)`（L1620）一致。**持久/非持久路径**与源码 docstring L1507–1511 一致。**host block_l 来源** `_TiledPath.for_length`（`PERSISTENT_MAX_L=8192`、`MAX_BLOCK_L=512`、`REDUCE_THREADS=(256,128,64,32)`）核对无误。**空张量**：GPU op 有 `numel()==0` 早退，本迁移所用 **NPU op** `forward` 核对**确无**该守卫——§0.1/§9.1/§9.3 已明确由 NPU op/wrapper 集成期承接，L0-14 覆盖。§0.2 的 `out_idx=[-1]`（仅 grad_x 自动分配，源码 L1518）、host 预分配 `grad_weight/grad_bias` 后 7 参调用（源码 forward L2182–2211）与 §0.2/§4.2 逐位一致。
- **算法解读完整性 / 优化手段覆盖完整性 — pass**：§0.3 步骤表 P0–E2 覆盖源码全部计算语句（含 pass-1/pass-2、persistent/non-persistent 两路、`ragged` 尾块），无遗漏/无臆造；§0.4 十一项优化可逐项溯源，未把实现层优化误判为算法层；舍弃项在 §0.6 末尾意图承接汇总，无静默丢弃。
- **耦合性判定合理性 — pass**：§0.5 每条四态处置（保留/等价替换/重新设计/舍弃）有依据，与 GPU→NPU 映射不矛盾（shared 流式→UB、线程树→两步 `T.reduce`、flat `l//S`→仿射二维 slice、谓词→`T.min`+双 slice+reduce `size`、`threads`/`T.Parallel`/`T.Pipelined`→舍弃）。
- **NPU 重设计可行性 / 设计与决策一致性 — pass（v0 阻塞项 1 已修复，本条为复检重点）**：
  - **§0.6 R2/R7 的「跨 N-block 累加」语义在 §3.3 实现中成立**——`red`/`red2` 的 `T.clear` 已从 v0 的 `for nn` **外**（v0 design L537–538）**移入 `for nn` 循环体内**（v1 §3.3 L568–572：`n0 = nn*bN; rN = T.min(...); T.clear(red); T.clear(red2)`），而 `gb/gw` 的 `T.clear` 仍留在 `for nn` **外**（v1 L566–567）一次。因此块内 `T.reduce(g32, red, dims=[1], clear=False)` 的跨 S-tile 行和仅在**本 N-block 内**累加，块末 `T.reduce(red, gb, dims=[0], clear=False)` 得到 `gb = Σ_nn rowsum(blk_nn)`——**不再重复计入前序块**，`gb/gw/grad_x` 对 `n_tile_N ≥ 2` 的 tiled 用例（大 C `(32,2048,512)` `bN=16,n_tile_N=2`；large-spatial `(4,128,1048576)` `bN=2,n_tile_N=2`）亦正确。该写法与同族已交付拆分核 §3.3 **逐字一致**（`examples/TileOPs/tileops/kernels/norm/batch_norm_bwd/batch_norm_bwd_kernel/_batch_norm_bwd_split_kernel_DESIGN.md` L577–578 `T.clear(gb/gw)` 在 `for nn` 外、L582–583 `T.clear(red/red2)` 在 `for nn` **内**；已亲自 Read 核对）。
  - **v-op 静态 extent 语义在 §3.3 实现中成立（v0 阻塞项 2 已修复）**——§3.3 全部 up/down-cast 已从 v0 的运行时切片（v0 L549/L554/L592/L597 `T.vcast(stage[0:rN,0:rS], ...)`）改为**全 buffer** 形态：`T.vcast(stage, g32, round_mode="rint")`（v1 L578/L632）、`T.vcast(stage, xh32, round_mode="rint")`（v1 L583/L637）、`T.vcast(prod, stage, round_mode="rint")`（v1 L619/L650）；所有 v-prefix 逐元素 op 均在 `(bN,bS)` 全 buffer 上运算；运行时 extent（`rN/rS = T.min(...)`）**仅**传给 `T.copy`（I/O 双 slice）与 `T.reduce(size=[rN,rS] / [rN,1])`（归约）。尾块正确性由 **reduce `size` 截断 + dst 显式 slice** 承担，不依赖 v-op 切片——与同族宽核 Stage 3 实测修正（`_batch_norm_bwd_wide_kernel/RETROSPECTIVE.md` L75/L83/L89，全 buffer vcast）一致。
  - **跨 § 一致性 — 无自相矛盾**：§3.3（伪代码）↔ §3.2（映射表 L474/L482 全 buffer）↔ §5.4（L844 复位约束、L845 v-op 静态 extent）↔ §6.1/§6.2/§6.4（L891/L895/L903 累加器复位）↔ §9.2（L1037 N-block 复位条、L1039 v-op 运行时切片条）五处措辞与伪代码完全同口径。
- **golden 独立性 — pass**：§8.1 golden 直接调用 `torch.sum(dim=(0,2))`/`torch` 逐元素模块函数，**不含** NPU 设计的 persistent 分核、二维 `(bN,bS)` 分块、两步 `T.reduce`、`T.vbrc` 广播、fp32 中转链、分布式 epilogue 等任何算法特征；以 §0.1 语义为唯一依据（移植源仓 `workloads/norm.py`），验证独立。

**维度 0 结论: pass**（语义/算法/优化/耦合性/golden 独立性均核对通过；**v0 的两个阻塞项均已由 v1 定点修复**——T.clear 移入 `for nn` 内、v-op 全 buffer，且 §3.3/§5.4/§6/§9.2 交叉一致，重设计保持语义）。

### 1. API 可行性 — pass

§3.2/§3.4 每条 API 均有文档佐证且路径经核对存在（`T.Kernel`/`T.alloc_ub`/`T.copy`/`T.vcast`/`T.reduce`/`T.vbrc`/`T.vmul`/`T.vsub`/`T.vadd`/`T.min`/`T.clear`）；`T.reduce(size=...)` 存在（`docs/Tilelang.language/规约操作/T.reduce.md`：`size` 控制 reduce 实际参与计算的数据范围）；Expert/Vector 形态有先例（`examples/conv1d/_conv1d_direct_kernel`）；v-prefix 优先。

- **v0 阻塞项 2（v-op 运行时切片）已消除**：§3.3 所有 v-op 操作数切片均为编译期常量 `(bN,bS)` 全 buffer（L578/L583/L619/L632/L637/L650），运行时 `rN/rS` 只喂 `T.copy` 与 `T.reduce(size=…)`；§3.2 map 表（L474/L482）与 §3.4（L669）同口径。无 API 用法违反工具链静态-extent 约束。

**维度 1 结论: pass**（API 均存在且用法合规；v0 的运行时切片用法已修正为全 buffer）。

### 2. 内存层级规划 — pass

§4.4 完整给出 GM→UB 路径（无 GM→L0 直连），全部计算 UB 内闭环、无中间 GM 往返；§4.3 每 buffer 标注层级（全部 UB）。§4.5 的 BPE 采用同族宽核 Stage 3 实测的 auto-multibuffer 膨胀后 footprint（transit 22、fp32 26），规避「额定 BPE 高估 bS → UB 溢出」陷阱。独立复算见「机械复核」`ub_budget`：held ≤ 180,224、tiled ≤ 184,096，均 ≤ 184,320 预算；公式结构性有界，任意 N/S 不越物理容量。held 临界（resnet-stage1 180,224B）依赖膨胀系数不超 22，§9.1 已列「超预算即回退 tiled」+ 工厂 guard，属参数级风险。

**维度 2 结论: pass**。

### 3. Tiling 策略 — pass

§5.2 block 划分明确（向量轴 = S 尾轴、归约内层维 `dims=[1]`、N 为外层 reduce）；对齐（行内 stride=1、fp16/bf16 ×8、fp32 ×4；尾 lane 浪费 ≤0.5%）；非整除经 `T.min`+双 slice+reduce `size`。**分核策略三要素齐全**：① 逻辑核数 `num_logical = C`（§5.5 表）；② 物理核数 `NPUUtils.get().get_aicore_num()` 实查 = 20（记录查询代码 + 返回值，纯 Vector 翻倍 → 40）；③ 极大规模分支 `num_kernels = min(C,40)`、核内 `T.serial(num_local_tasks)` grid-stride、`if c < C` 守卫、边界静态。核数与 block 取值自洽，负载不均 ≤1 任务。§5.5 已标注本函数实际服务 `_holder` 判 `tiled` 的 shape（大 C）。

**维度 3 结论: pass**。

### 4. 技术约束检测 — pass

§3.5 逐项核对：无三维 Kernel（一维 `T.Kernel(num_kernels, is_npu=True)`）；GPU 专用 API（`threads=`/`T.Parallel`/`T.Pipelined`/`T.alloc_shared`/`T.alloc_fragment`/`T.reduce_sum(dim=1)` 线程树）已列舍弃清单；无 GEMM 分形/非整除；L0C 不适用（纯 Vector）；物理核数实查 20→40；bf16 无 Vector 算术经 fp32 中转链（必需，已核对 `T.vmul.md` §2.2.1 bf16 列为 ×）；中间轴归约（非尾轴）经 strided 平面 gather（探针验证）。无遗留阻塞约束。

**维度 4 结论: pass**。

### 5. 循环与同步 — pass

§6.1/§6.2 明确循环类型（任务维 `T.Kernel`+`T.serial`、N/S-子块维 `T.serial`、元素维向量化、尾块标量 `T.min`+双 slice+reduce `size`）；§7 同步与 Expert 模式匹配（单 Vector scope 内编译器按 MTE2→V→MTE3 自动 `pipe_barrier`；无跨引擎/跨核协作，无需手动 `set_flag/wait_flag`；conv1d Expert/Vector 先例佐证）；§6.3 说明不显式 `T.Pipelined` 的理由；§6.4 尾块逻辑包含 reduce `size` 截断。

- **v0 阻塞项 1（N-block 累加时序）已消除**：§3.3/§6.2 的 pass-1 累加结构现为「`gb/gw` 在 `for nn` 外清一次、`red/red2` 在 `for nn` 内每块清一次」，跨 N-block `clear=False` 累加不再重复计数（§6.1 L891、§6.2 L895、§6.4 L903 同口径）。循环边界静态，满足 Ascend 约束。

**维度 5 结论: pass**（循环边界静态、尾块与累加时序均正确；v0 阻塞项 1 已修复）。

### 6. 验证方案 — pass

§8.1 含独立 PyTorch 参考实现草案（golden 独立性见维度 0）；§8.2 精度标准取自源仓 `batch_norm_bwd_tolerance`（fp32 1e-5 / fp16 1e-2 / bf16 2e-2，含通道梯度 `sum_atol`），与探针口径一致；§8.3 L0 计划 **15 项**覆盖 fp16/bf16/fp32、held/tiled、S=1/N=1 退化、非向量宽整数倍 S=900、微型 shape、空张量 `L=0`（op 层拦截）、**L0-15 `(1,8,1)` 1 元素平面退化路由（v1 建议项 4 新增）**，三输出各自比对。**L0-9（大 C，`n_tile_N=2`）与 L0-8（large-spatial，`n_tile_N=2`）为 v0 阻塞项 1 的回归门住点**；L0-8/L0-13 命中尾 S-tile、L0-9 命中尾 N-block。

**维度 6 结论: pass**。

### 7. 完整性与一致性 — pass

§0–§11 章节齐全（迁移类含 §0，0.1–0.7 子节齐全）；全文无 `{placeholder}`/`TODO`/`待补充`（grep 已核）；§3.5.3 引用的 examples 路径经核对**全部真实存在**；API 映射（§3）/内存规划（§4）/Tiling（§5）主体一致。**v0 的 2 处措辞 warn 已订正**：§3.3/§0.2/§4.2/§10.5 的 `out_idx=[-1]` 返回表述统一为「返回 `grad_x` 张量」（`tilelang/jit/jit_npu.py` 在 `len(out_idx)==1` 时直接返回张量）；§1.6.0 R3/R4 缓冲峰值口径已与 §4.3 对齐（4 路 fp32 `(bN,bS)` + 1 路 dtype 中转 + `(bN,1)` 累加器）。

**维度 7 结论: pass**。

### 8. 算法优化分析 — pass

检视方式：以 §1.6 为对象逐项核对，并交叉核对 §3.1/§3.3/§6 与 §1.6 的双向一致性；数学等价性、调研结论由本维度**独立推演 / 独立复核**核对（不轻信文档自述），弃选前提亲自打开 API 文档 / 候选库**核对**。

- **§1.6 章节存在**：含 1.6.0 调研、1.6.1 数学等价优化、1.6.2 向量化替代、1.6.3 向量化轴与布局决策四子节，各有结论行。
- **调研存在性与完整性 — pass**：§1.6.0 四问齐全——R1 候选表**含基线 #0（源 `_batch_norm_bwd_kernel`）**并评估 5 候选；R2 在线算法**有明确结论**（「无新增收益/等价于基线」+ 结构依据）；R3 复杂度表覆盖**四口径**（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值，基线在表，口径写清）；R4 逐候选亲和评估（单元匹配/容量/对齐/静态边界/流水/跨核）；「无更优替代」写明调研范围（本地源，无互联网）。
- **调研结论正确性（独立复核）**：
  - **负向断言复核（对照候选库 + 源码证据）**：亲自 Read `.agents/skills/tilelang-op-design/references/algorithm-candidates.md`——`ALG-layernorm`（L17–23，family 含 `batch_norm`）R2 列 Welford/分块 running 统计；`ALG-reduction`（L62–68）R2 列「分块归约 + 合并（两阶段/树形）；Welford」。逐条复核设计的负向论断成立：R2#5 online/Welford——本反向的 `mean/rstd` 为**前向输入**（源码 L1564/L1574 直接读入，算子不计算 mean/var），算子只算两个**可加一阶矩** `Σ grad_out`、`Σ grad_out·x̂`，无需要 rescale 的 running 统计量（对比 online softmax 的 running max/sum）；「在线化」退化为候选库所述「分块归约 + 合并」结构（即其 held/tiled 实现），**非独立算法族**——与候选库定位不矛盾；R1#3 raw-moment（= `ALG-layernorm` R1 的 `E[x²]−E[x]²` 族变体）已被**机器证据**否决（复跑 fp32 违反率 0.1553 > 0）；R4 跨核 split 被判「C 大并行度已足、无收益」并被拆分核覆盖。**未发现与候选库/源码相矛盾的负向断言，未漏掉已知候选**。
  - **复杂度算术独立复算**：held 单扫描 Bytes = 读 g + 读 x + 写 gx = `3·L·C·eb`（+5C）✓；tiled 两扫描 = `5·L·C·eb` ✓；代表 resnet-stage1 `3·8192·64·2 ≈ 3.1MB` ✓、large-spatial `5·4194304·128·2 ≈ 5.37GB` ✓；FLOPs ≈9L/通道与 manifest roofline `flops = 9·B·C·L` 一致；中间缓冲为 UB 常驻（无 GM 往返），未漏计。设计期 roofline：3.1MB/1.26TB/s ≈ 2.5µs、5.37GB/1.26TB/s ≈ 4.26ms，复算一致。
  - **弃选论证前提核对（亲自开文档核对）**：raw-moment 以 D-1 机器证据否决 ✓；否决项「fp16/bf16 原生域计算链」引用 `数学操作/T.vmul.md` §2.2.1——亲自核对表格：bf16 列为 **×**、fp16/fp32 为 √（`T.vmul.md` L25–27），fp32 中转必需 ✓；「跨 dtype `T.copy` 作 cast 融合」引用 `内存操作/T.copy.md` §3——亲自核对该文档 L54–55/L70：**Expert 模式通用路径要求 src/dst 同 dtype**（跨 dtype 编译期报错 "T.copy does not support element type casting"），与「统一走显式 `T.vcast(rint)`」结论一致 ✓；候选 #2 flat 弃选依据「非仿射、无仿射 slice 可载」与拆分核 `RETROSPECTIVE.md` 验证一致 ✓；候选 #3 转置写成「host permute 或核内 `T.transpose` 链」，亲自打开 `创建操作/T.transpose.md` 核对——`T.transpose` 底层 `hivm.hir.vtranspose` 只支持转置两轴（L36），示例用 `T.alloc_shared`（UB 级，L56–60），**未误写成 GM 级 transpose** ✓；R4 容量/带宽淘汰引用 `constants.md` CONST-capacity-910B2C / CONST-mte2-degradation / CONST-vector-launch-overhead / CONST-aicore-910B2C（已核对四条均存在，L8/L46/L82/L158）✓。
  - **并列候选（R1#1 held vs #2 tiled）裁决路径**：判定裕度 = 流量差（held 3 vs tiled 5，+66%）远超未实证常数不确定区间 → 无须三件套；唯一未实证量（auto-multibuffer 膨胀系数）为**参数级**（bS/bN 取值）而非结构级，已有同族宽核 Stage 3 实测值，且 §1.6.3 已说明规避陷阱。判据充分，非纸面拍板。
- **调研→设计一致性 — pass**：§1.4、§3.1（输入声明为 §1.6.1 优化后公式）、§6 与 §1.6.0 选定算法族一致；迁移 §0.5/§0.6 处置与调研选族一致，源算法被替换项（shared/线程树/threads/谓词/flat map）意图承接在 §0.6 末尾逐项说明，无静默丢弃。
- **数学等价优化完整性 + 等价性独立推演 — pass**：§1.6.1 对选定算法系统审视，采纳 O1/O2/O3，逐项四要素齐全。**独立推演**：O1 `per = (w·rstd)·fl(1/L)` 对 `coef = w·rstd/L`——L 为编译期常量（`L_inv = 1.0/float(L)` 工厂期 Python 常量，§3.3 L493），`fl(1/L)` 相对误差 ≤ 2⁻²⁴，L=2^k 时精确；O2 分配律恒等 `coef·(L·g − gb − x̂·gw) = (w·rstd)·g − per·gb − per·gw·x̂`（`coef·L = scale`，`coef = per`），逐项展开成立；O3 `x̂` CSE 为同值复用（held 驻留，无浮点语义变化）。三项经 `verify_equiv.py` **复跑机器验证**（21 案例全 `EQUIV_PASS`，三 dtype max ulp=0），与 §3.1/§3.3 输入口径一致。
- **向量化替代覆盖完整性 — pass**：§1.6.2 覆盖方案全部循环/标量点（13 项，与 §3.3 伪代码、§6 循环结构交叉核对）——逐元素计算（gather/cast/x̂/累加/epilogue/舍回）与通道归约已全部向量化/归约化（替代 API 有 docs + 探针佐证，优先 v-prefix）；保留 #10 通道任务映射、#11 N/S-子块调度、#12 尾块 `T.min` 三类，理由具体（block/tile 级调度、动态边界元数据，非逐元素热点），无「实现简单」类泛理由；§6 循环结构与之一致（kernel 内无逐元素标量循环）。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 规约类必选候选齐全——水平归约 vs 垂直扫描两种 lane 映射分别评估（#1 水平、#4 垂直），另评估核内 flat 展平（#2）与转置布局（#3）；候选矩阵逐项量化（整除性/尾 lane 浪费/累加链/repack 代价/UB 影响）；弃选均给量化理由（#2 非仿射地址破坏向量化、#3 归约轴不变纯开销、#4 跨步无向量宽度）。**GPU 源码隐式轴选择独立复核**——§1.6.3 指出 GPU 把 `threads` 映射到通道内扁平 `L` 连续段（源码 L1581），NPU 独立评估后选 S 为向量轴（行内 stride=1、v-op 行内广播形态、held 单扫描流量最小），结论一致、理由不同，未把 GPU 轴当结论。
- **与下游章节一致 — pass**：§3.1 以 §1.6.1 优化后公式为输入；§6 循环结构与 §1.6.2 结论一致；布局决策（S 轴、`(bN,bS)`）与 §3.3/§4/§6 三方一致。

**维度 8 结论: pass**（调研完整性 / 调研结论正确性〔复杂度独立复算 + 负向断言对照候选库独立复核 + 判别段 A 机器证据〕/ 调研→设计一致性 / 优化完整性 / 等价性独立推演 / 弃选论证前提亲自核对 / 向量化覆盖 / 不可替代理由 / 替代 API 佐证 / 与 §3.1§6 一致性 全部通过）。

## 复检总结（供 conductor / Stage 1）

- **阻塞项 1（pass-1 跨 N-block 双重计数）**: FIXED——`T.clear(red/red2)` 已置于 `for nn` 循环内（§3.3 L568–572），`gb/gw` 仍在循环外清一次；与拆分核 §3.3 逐字一致；§5.4/§6.1/§6.2/§6.4/§9.2 同步。证据：v1 §3.3 L566–572，对照 v0 L537–540（循环外）。
- **阻塞项 2（v-op 运行时 extent 切片）**: FIXED——§3.3 全部 up/down-cast 改为全 buffer（L578/L583/L619/L632/L637/L650），运行时 slice 仅留给 `T.copy`/`T.reduce(size=…)`；§3.2/§5.4/§9.2 同口径。证据：对照 v0 L549/L554/L592/L597。
- **建议 3/4/5**: 全部落实（R3/R4 缓冲口径对齐 §4.3；`scalar_plane` 工厂守卫 + L0-15；`if single_tile:` held 单扫描分支 + 返回 `grad_x` 张量措辞订正）。
- **新引入问题**: 无。两个修复均未引入新的阻塞级或建议级问题；机械复核（D-1 全 EQUIV_PASS、D-5 r3_metrics pass + skip 项人工复核）通过。
