# _conv1d_direct_kernel 设计文档检视报告

结论: 通过

## 检视元信息

- 文档: examples/conv1d/_conv1d_direct_kernel/DESIGN.md
- 任务类型: 迁移（GPU → Ascend NPU harness）
- 源算子: /mnt/workspace/TileOPs-main/src/tileops/kernels/convolution/conv1d.py（`_conv1d_direct_kernel` builder L316-402；`DepthwiseConv1dKernel` L1074-1126）
- 检视维度: 9 项（维度 0 源算子理解与迁移分析 / 1 API 可行性 / 2 内存层级规划 / 3 Tiling 策略 / 4 技术约束检测 / 5 循环与同步 / 6 验证方案 / 7 完整性与一致性 / 8 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 6

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: **不适用**。§1.6.1 结论为「无优化空间」（无采纳的公式级优化项），按 gate `S1-EQUIV-EXEC` 豁免规则不生成 `verify_equiv.py`；设计 §10.2 文件清单已显式标注「`verify_equiv.py` —（不适用）」并给出豁免依据。已核对 §1.6.1 表内 row 1–4（Winograd / FFT / 降代换 / 加法重排）均为**否决项或「不适用」**，无「采纳项」，与「无优化空间」结论一致。
- **design_calc_check**: 执行
  `python3 .agents/tools/design_calc_check.py --design examples/conv1d/_conv1d_direct_kernel/DESIGN.md`，JSON 输出：

```json
{
  "design": "examples/conv1d/_conv1d_direct_kernel/DESIGN.md",
  "capacity_constants": {
    "ub_kb": 192.0,
    "ub_inflation": 1.7,
    "l0c_kb": 128.0,
    "source": "pattern-library/constants.md CONST-capacity-910B2C"
  },
  "checks": [
    {"check": "ub_budget", "status": "skip", "detail": "no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"},
    {"check": "l0c_budget", "status": "pass", "detail": "block 128x128 x fp32 accum = 64.0 KB vs L0C 128 KB"},
    {"check": "core_split", "status": "skip", "detail": "insufficient parsed dims {'N': 1, 'bn': 128}; declared logical cores = 31"},
    {"check": "r3_metrics", "status": "pass", "detail": "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"}
  ]
}
```

  - `ub_budget` = **skip** → 机械跳过，**人工已核**：§4.5 逐 buffer 复算代表 shape（`full-conformer-depthwise-k31-fp16`，`c_in=c_out=64, l_in=128, K_W=31, block_N=128, fp16`）= `ub2` 16384 + `xt` 7936 + `xt32` 15872 + `prod` 15872 + `wc` 62 + `w32` 124 + `acc` 512 + `b11/b11_32` 6 + `out_ub` 256 = **57024 B ≈ 55.7 KB**，×1.12 ≈ 62.4 KB < 192 KB，通过（详见维度 2；一般上界式存在算术偏差，见维度 2 建议项）。
  - `l0c_budget` = **pass**（128×128×fp32=64 KB ≤ 128 KB）——本算子无 Cube/L0C，此项不适用，仅作背景。
  - `core_split` = **skip** → 机械跳过，**人工已核**：checker 将 `K_W=31` 误判为逻辑核数。实际 §5.5 逻辑核数 `num_logical = R×GX_L = 64`（conformer）/ `32`（smoke），物理核数 `NPUUtils.get().get_aicore_num()=20`（实查）×2=40（纯 Vector），`num_local_tasks` 算术自洽（详见维度 3）。
  - `r3_metrics` = **pass**：R3 复杂度表 FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值**四口径齐全**（§1.6.0 R3 表），与维度 8 独立复算一致。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

已亲自 Read 源算子全文核对：`/mnt/workspace/TileOPs-main/src/tileops/kernels/convolution/conv1d.py` 的 `_conv1d_direct_kernel`（L316-402，`_conv1d_direct_body` L342-377）、`DepthwiseConv1dKernel`（L1074-1126）、ops 层 `src/tileops/ops/convolution.py`（`Conv1dFwdOp.forward` L112-163）。

- **语义理解正确性 — pass**：§0.1 数学语义 `out[n,c_o,l]=bias[c_o]+Σ_{k_w} W[c_o,0,k_w]·x[n,c_o,l·s_W+k_w·d_W-p_L]` 与源码 `out_local[i,j] += if_then_else(valid, cast(x[bz,oc,il],fp32)*cast(weight[oc,0,kw],fp32), 0)`（源码 L354-365，`il = ol*stride_l + kw*dilation_l - pad_left` L358）逐步一致；`L_out` 公式与源码 L331 一致。dtype 语义（fp32 累加 `accum_dtype="float"` L330、bias 在 fp32 域相加后统一 cast，源码 L372-375）核对一致；边界语义 `valid=(oc<c_out)&(ol<out_l)&(il>=0)&(il<l_in)`（L359）与 §0.1 边界行一致；**无跨输入通道收缩**（`x[bz,oc,il]` 直接以输出通道号 `oc` 索引输入通道，L362）与 §0.1/§0.2 一致。§0.2 I/O：`x (n,c_in,l_in)` / `weight (c_out,1,kernel_l)` / `out (n,c_out,out_l)` 与源码 builder 签名（L382-397，`weight: T.Tensor((c_out,1,kernel_l))`）逐项一致。host 侧：ops 层 `.contiguous()`（`convolution.py` L141-144）确为 metadata 物理连续化、depthwise dispatch、`DepthwiseConv1dKernel.forward` 直接 `launch`（`conv1d.py` L1065-1071）无 permute/reshape/padding——§0.1 host 行核对一致。
- **算法解读完整性 — pass**：§0.3 步骤分解 4 步（grid / clear / 核窗并行 MAC / epilogue）覆盖源码全部计算语句（L344-377），无遗漏、无臆造；§0.3 记录 `T.use_swizzle`（L351）、`default_config`（L1113-1122：`block_m=1, block_n=128, block_k=1, num_stages=1, threads=128`）与源码一致；host 侧逻辑（`forward` 直调、`_build_program` 仅调 builder）已在 §0.3 列出。
- **优化手段覆盖完整性 — pass**：§0.4 识别逐元素 MAC+register 累加、核窗 `T.serial`、M/N tiling、边界掩码、swizzle、threads、`num_stages`/`enable_rasterization` 七项并逐项标注硬件依赖；源码本算子无 SMEM tiling / tensor core mma / `T.Pipelined` / warp shuffle（与源码 L342-377 无 `alloc_shared`/`T.gemm`/`Pipelined` 一致），未误报。
- **耦合性判定合理性 — pass**：§0.5 每条目有四态处置+依据，与 `migration-analysis.md` §5.3 映射表一致（register 累加→等价替换 UB+Vector；三维 grid/threads→重新设计一维 persistent；掩码→重新设计 padded workspace；swizzle/threads→舍弃并写明理由）。舍弃项（swizzle、threads）均有理由（无 SMEM；`TRAP-threads-kwarg-noop` 已核对该 pattern-library 条目存在且证实 `threads=` 在 npuir 无效果）。GPU 专用 API（`T.Parallel` 掩码 MAC、`threads`）判「重新设计/舍弃」，未误判为可保留。
- **NPU 重设计可行性 — pass**：§0.6 R1–R5 每项含「源方案 → NPU 新算法 → 语义保持论证」三要素；新算法满足 NPU 约束（一维 `T.Kernel`、GM→UB→GM、无分形/L0C 依赖、编译期静态边界）；**模式判定正确且论证充分**——本算子写不构成 GEMM（无输入通道收缩），源码注释本身即明言「with a single channel per group there is no contraction to tile, so each thread accumulates one output position over the kernel window instead of running an implicit GEMM」（源码 L1077-1079），与 §2.2/§1.6.0 R1 候选 3 的淘汰论证一致；design probe（`history_version/design_probe_direct_vector.py`）实测 Expert+单 `T.Scope("Vector")` 编译运行成功（`max_abs_err≈7.8e-4`），Developer 编译失败，**该判定有直接实证**。**无残留 GEMM/Cube 依赖**：§3.3/§6 伪代码仅 `T.copy`/`T.vcast`/`T.vmul`/`T.reduce`/`T.vadd`，无 `T.gemm`/`T.alloc_L1`/`T.alloc_L0C`/`T.Scope("Cube")`；§2.2 显式声明「不声明 `T.Scope("Cube")`、不使用 `T.alloc_L1`/`T.alloc_L0C`/`T.gemm`」。**Vector 模式判定为真**。
- **设计与决策一致性 — pass**：§1–§7（§4.4 搬运路径、§5.1 计算类型「纯 Vector」、§6 循环结构、§7 同步）描述的是 §0.5/§0.6 决策后的 NPU 算法，无「判重设计却照抄源方案」的脱节。
- **golden 独立性 — pass**：§8.1 golden 以 §0.1 源算子语义为唯一依据（`torch.nn.functional.conv1d(..., groups=C_in)`，weight `(C_out,1,K_W)`，fp32 域计算后一次 cast），**未复刻** §0.6 的 `xpad`/移位装载/`reduce` NPU 数据流；与源仓测试基准 `test_conv1d_depthwise_matches_torch` 同款（`examples/TileOPs/tests/ops/test_conv1d.py` L167-178 已核）。

### 1. API 可行性 — pass

- **API 存在性 — pass**：§3.2/§3.4 每条 API 均有佐证：`T.vcast`（`docs/Tilelang.language/数据类型转换操作/T.vcast.md`，已核 f16/f32/bf16 转换与 `rint`）、`T.vmul`（`docs/Tilelang.language/数学操作/T.vmul.md` §2.2.2 已核「行广播 `[M,N]*[M,1]`」）、`T.vadd`（`.../T.vadd.md` §2.2.2 已核列广播 `[1,N]+[1,1]`）、`T.reduce`（`.../规约操作/T.reduce.md` §2.2.2 已核 `(M,N)→(1,N)` 合法、`clear=True`）、`T.gather`（`.../索引与元素操作/T.gather.md` 已核 int32 索引、沿最后维）、`T.copy`/`T.alloc_ub`/`T.ceildiv`/`T.rs` 均有文档或 repro/先例（`T.rs` 见 `_gqa_prefill_fwd_kernel.py` 多例 + repro）。所有引用的 `examples/` 路径经 Glob 核对**真实存在**（add DESIGN、mish kernel、`_conv1d_kernel.py`、repro、design probe 全部命中）。
- **API 与模式匹配 — pass**：Expert 模式用单 `T.Scope("Vector")` + `T.rs` 搬运段标注（非 Developer 自动同步）。探针实证该形态可运行。
- **v-prefix 优先 — pass**：全链使用 `vcast`/`vmul`/`vadd`/`reduce`，无 `npuir_xxx` 旧式调用。
- **建议（非阻塞，维度 5 亦有记）**：§3.3 伪代码把前置段（L417-433）与主循环段（L436-474）写成**两个** `with T.Scope("Vector")`，而 §7.2/探针为**单一** Vector Scope 内的顺序依赖。建议实现时合并为单 Scope（或补显式 GM 写→读 barrier），与 §7.2 自述一致。

### 2. 内存层级规划 — pass

- **搬运路径完整 — pass**：§4.4 给出 GM→UB→GM 完整路径（无 GM→L0 直连、无越级）；`xpad`/`xdec` 为一次性 GM workspace，非每 tile 往返。
- **UB 预算 — pass（含 1 建议项）**：代表 shape 逐 buffer 复算 57024 B ≈ 55.7 KB（明细见机械复核），×1.12（`CONST-capacity-910B2C` 上偏系数）≈ 62.4 KB < 192 KB，余量 ~68%，**不溢出**。**建议（非阻塞）**：§4.5 的一般上界式 `(K_W·block_N·4)·2 + K_W·block_N·2 ≤ 192KB/1.12 ⇒ K_W·block_N ≤ ~26K` 存在**算术偏差**——按该式 `10·K_W·block_N ≤ 196608/1.12 ≈ 175542 ⇒ K_W·block_N ≤ ~17.6K`（即 `block_N=128` 时 `K_W ≤ ~137`），而非文中的 `~26K`/`K_W ≤ ~200`；另 `ub2 = [c_in, l_in]` 的容量未纳入一般上界（大 `c_in·l_in` 时可能成为主导项），建议补 `c_in·l_in·eb` 项或对前置按 `l_in` 分块。代表 shape 不受影响，Stage 3 落地前建议顺带修正该一般式。
- **L0C 容量 — pass（不适用）**：无 Cube 计算，无 L0C。
- **Buffer 层级标注 — pass**：§4.3 每个 buffer 标注 UB/GM 层级。

### 3. Tiling 策略 — pass

- **Block 划分明确 — pass**：§5.2 `block_N=128`（承接源 `default_config` 的 `block_n=128`，源码 L1114-1122）、`block_M≡1`（每行一任务，depthwise 行间无复用），有理由。
- **对齐约束 — pass**：`block_N=128` 为 32B 整数倍且整除向量宽（fp16 ×8 / fp32 ×4）✓。
- **非整除处理 — pass**：尾块 `tn = T.min(block_N, out_l - ol0)` + 双显式 slice + `lid < num_logical` guard（§6.4）；输入越界由零填充 `xpad` 隐式处理。
- **分核策略三要素 — pass**：① 逻辑核数 `num_logical = R×GX_L`（§5.5，conformer 64 / smoke 32）；② **物理核数实查** `NPUUtils.get().get_aicore_num()=20`（§5.5 记录查询代码与实际返回值，并注明与 `CONST-aicore-910B2C` 的 24 不一致、以实查为准），纯 Vector 翻倍 `20×2=40`（与 `core-split-strategy.md` §1/`ascend-constraints.md` §1 规则一致）；③ 规模判定：conformer（64>40）中等规模 `num_cores=40`、`num_local_tasks=ceil(64/40)=2`，smoke（32≤40）`num_cores=32`、`num_local_tasks=1`——算术自洽，核内 `T.serial` 边界为编译期常量（静态）✓。
- **建议（非阻塞）**：§3.3 伪代码 `num_cores = NPUUtils.get().get_aicore_num() * 2`（=40，**无 `min`**）与 §5.5/§5.4/§9.3 的 `num_cores = min(num_logical, 40)` **不一致**；建议伪代码补 `min`（否则 smoke 会启动 40 核、8 核空转，虽被 `lid` guard 兜住不越界）。另 §5.5 「每核 ≤2 任务…无负载不均」对 conformer（64 任务/40 核）实为 24 核执行 2 任务、16 核执行 1 任务，措辞宜改为「无超发、轻微波形不均」。

### 4. 技术约束检测 — pass

- **三维 Kernel — pass**：源三维 `T.Kernel(ceildiv(out_l,BN), ceildiv(c_out,BM), n, threads=…)`（源码 L344-349）已重设计为一维 `T.Kernel(num_cores, is_npu=True)` + `lid` 解码（§0.6 R1/§5.5）。
- **GEMM 非整除 / L0C 溢出 — pass（不适用）**：无 GEMM、无 L0C。
- **GPU 专用 API — pass**：`T.Parallel` 掩码 MAC、`T.alloc_fragment`、`T.use_swizzle`、`threads` 全部改写为 NPU Vector 形态。
- **host 禁改输入 / 禁 padding — pass**：`xpad` 由 kernel 内 Vector 前置构造，host 仅 `torch.zeros` 分配 workspace（不违反 `ascend-constraints.md` §4，已核该节条款）。

### 5. 循环与同步 — warn

- **循环结构 — pass**：§6.1 明确 `T.Kernel`（核级）/`T.serial(num_local_tasks)`（tile 级 persistent）/`T.serial(kernel_l)`（per-kw 装载）/`T.reduce`、`T.vmul`（向量）选择；§6.4 尾块逻辑完整。
- **同步与模式匹配 — warn（非阻塞）**：§7.1/§7.2 采用单 `T.Scope("Vector")` + `with T.rs("PIPE_MTE2"/"PIPE_MTE3")`，并论证「本算子无跨引擎数据流，无需 `sync_block_set/wait`」——方向正确（纯 Vector，且设计源 repro/探针实测）。但：（a）§3.3 伪代码前置段与主循环被切成**两个** `T.Scope("Vector")`，与 §7.2「单 Scope 内顺序依赖」自述及探针的单 Scope 形态不符——GM `xpad` 写→读跨 Scope 的可见性需实现时确认；（b）**stride>1 polyphase 形状细节**：§3.3 中 `ubd = alloc_ub([c_in, l_pad])` 而 `xdec` 形参为 `(n,c_in,stride_l*Mdec)`、`Mdec=l_pad//stride_l`，`T.copy(ubd, xdec[..., 0:stride_l*Mdec])` 在 `l_pad % stride_l ≠ 0`（如 stride=3、l_pad=160→s_W·Mdec=159）时源/目的形状不一致（建议源侧显式切片 `ubd[0:c_in, 0:stride_l*Mdec]` 或与同族 `_conv1d_kernel.py` 一致令 `xdec` 行宽取 `l_pad`）。代表 workload 均 stride=1、L0 计划 stride=2（l_pad 恒 16 倍数、可整除）**不受影响**，故列为建议级。

### 6. 验证方案 — pass

- **Golden 函数 — pass**：§8.1 给出 `F.conv1d(groups=C_in)` 参考实现草案（fp32 域计算 + 一次 cast），独立于 NPU 算法。
- **L0 测试计划 — pass**：§8.3 代表 workload（`full-conformer-depthwise-k31-fp16` 有 bias / smoke 无 bias）+ 6 个必测点（非整除尾块 / bias 两路径 / batch>1 / padding>0 / dilation>1 / stride>1）。
- **精度标准 — pass**：§8.2 按 dtype 给 atol/rtol（fp16 1e-2、fp32 1e-4、bf16 1.6e-2），与源仓 `conv1d_tolerance` 口径一致（测试文件已核 import）。

### 7. 完整性与一致性 — pass

- **章节齐全 — pass**：模板 11 章节齐全，迁移类 §0 及 0.1–0.7 小节齐全（§0.1–0.7 均已核）。
- **无占位符 — pass**：全文无 `TODO`/`待补充`/`{placeholder}`（grep 无命中）。
- **同类引用真实 — pass**：§3.5.3 引用的 `examples/` 路径经 Glob 全部命中（add DESIGN、mish、`_conv1d_kernel.py`、repro、design probe）。
- **内部一致 — pass（含 1 建议项）**：API 映射 §3.2、内存规划 §4、Tiling §5 相互一致。**建议**：§1.6.0 两处将向量先例写作「hardshrink」（L194、L235），本仓实际为 `hardsigmoid`（`examples/TileOPs/tileops/kernels/elementwise/hardsigmoid/`）——非结构性引用，建议更正名称。

### 8. 算法优化分析 — pass

> 检视方式：对 §1.6 逐项核对，并交叉核对 §3.1/§6/§3.3 与 §1.6 的一致性；调研结论、负向断言、弃选论证前提均**独立复核/复算/核对**，不轻信自述。

- **§1.6 章节存在 — pass**：含 1.6.0 算法调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代分析 / 1.6.3 向量化轴与布局决策四子节。
- **算法调研存在性与完整性 — pass**：§1.6.0 **调研四问齐全**——R1 等价化简候选表**含基线**（候选 0，输入/源算法）并覆盖 7 个候选；R2 在线算法给出明确结论「无独立在线变体」并附结构依据（窗口类逐行相关、无 cross-tile 依赖/running 统计量）；R3 复杂度对比表覆盖**四口径**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值，基线在表、口径写明）；R4 硬件亲和逐候选评估（计算单元/容量/对齐/静态边界/流水/跨核）；「无更优替代」写明调研范围（algorithm-candidates ALG-conv 命中行、恒等变形自查、pattern-library、本仓 Vector 先例、源算子、design probe）。深度与算子复杂度匹配。
- **调研结论正确性（独立复核）— pass**：
  - **等价性复核**：选定候选 1（`[K_W,BN]` 移位 tile · 权重 `[K_W,1]` 行广播 `vmul` → `reduce(dims=[0],sum)`）与原语义独立推演成立——逐元素 `xt[kw,j]·w[kw]` 后沿 `kw` 求和，与源 `Σ_{k_w} w·x_shift` 逐项对应；`reduce` 结合序差异仅 fp32 末位舍入（≤ulp），与 §8.2 容差一致。
  - **复杂度复算**：独立复算基线 `2·n·C_out·L_out·K_W`（MAC=2）与候选 1「不增 FLOPs、只换执行结构」四口径；核对**中间缓冲 GM 往返未漏计**（§1.6.0 R3 明确计入 `xpad` 前置 `≈2·C_in·l_pad·eb` 与每 tap 移位段读；candidate 7 计入 `xdec` 物化），与设计数字一致。设计期估算（发射项主导 ≈30µs，`CONST-vector-launch-overhead` ×0.5µs/op）算术自洽。
  - **负向断言复核**：R2「无在线变体」对照 `algorithm-candidates.md` ALG-conv 命中行（R2 记「滑窗天然流式」）与源码证据（§0.3 核窗串行单遍结构）——结论一致，成立；R1「无更优替代」对照命中行 `im2col+GEMM / implicit GEMM / Winograd`：im2col+GEMM（候选 3）、Winograd（候选 4）均已评估淘汰，implicit GEMM 虽未单列候选行，但其淘汰由候选 3 的「无通道收缩 → Cube K 轴仅 `K_W`、利用率极低、FLOPs/权重 groups 倍劣」结构性覆盖（源码注释亦印证无 implicit GEMM），负向断言未被推翻。
  - **亲和性淘汰依据核对**：容量/带宽淘汰引用 `CONST-capacity-910B2C`（已核该条目存在：UB 192KB、`ub_inflation`、逐 buffer 显式 alloc 结构特性）；发射开销引 `CONST-vector-launch-overhead`（已核 ~0.5µs/op）；microDMA 退化引 `CONST-mte2-degradation`（已核条目存在）。无与 pattern-library 实测条目矛盾之处。
- **调研→设计一致性 — pass**：§1.6.0 选定算法（逐行加权归约/Vector）实际驱动 §1.4 算法描述、§3.1 公式拆解、§6 循环结构（累加归约维 = `K_W`，向量维 = L 轴）；与迁移 §0.5/§0.6 处置一致（源算法优化意图逐项承接见 §0.6 末段），无静默丢弃。
- **数学等价优化完整性 — pass**：§1.6.1 对选定算法做系统审视（Winograd/FFT/降代换/稳定化/重排），逐项给结论；**结论「无优化空间」有依据**（纯乘加、无除法/超越函数/公共子表达式/溢出），与 gate `S1-EQUIV-EXEC` 豁免规则一致；无采纳项故无四要素要求。
- **等价性论证正确性 — pass（不适用）**：无采纳的公式级优化项，无变形语义风险。
- **向量化替代完整性 — pass**：§1.6.2 覆盖 12 个计算点，与 §3.3 伪代码、§6 循环结构**交叉核对一致**（§6 无逐元素标量循环，仅保留 polyphase 抽取索引、任务解码、尾块 `tn`/guard 三类 block 级/动态边界标量点）；逐元素乘/归约/cast/bias 相加/边界补零全部向量化。
- **不可替代理由充分性 — pass**：保留项逐项具体理由（跨步 gather API 缺失 `TRAP-C10-strided-gather`、任务解码 block 级、尾块动态边界），非泛泛而谈。
- **替代 API 佐证 — pass**：替代 API 均落在 `docs/Tilelang.language/数学操作|规约操作|数据类型转换操作/` 与 design probe 佐证（与维度 1 同口径），优先 v-prefix。
- **向量化轴与布局决策完整性 — pass**：§1.6.3 按卷积类枚举候选矩阵（直接跨步窗口 / UB im2col / 行合并 2-D），逐候选给量化评分（整除性/尾 lane 浪费/累加链/repack 代价/UB 影响）；弃选 #2–#5 有量化理由，明确「I/O layout 是契约、核内布局是设计变量」，并对照了源码隐式轴（GPU 轴为输入非结论）。
- **弃选论证前提核对（亲自查证）— pass**：`T.vmul` bf16 不支持（打开 `docs/Tilelang.language/数学操作/T.vmul.md` §2.2.1 核对——bf16 列标 `×`，设计「bf16 必经 fp32 中转」成立）；`T.gather` 沿最后维、int32 索引（打开 `T.gather.md` 核对，polyphase 行宽/索引形态成立）；候选 #5「host permute/pad 违规」前提与 `ascend-constraints.md` §4（禁改输入内容、禁 padding）核对一致；未使用 `T.transpose` 弃选前提，无「核内重排写成 GM 级 transpose」类矛盾。
- **实验裁决模式完整性 — pass（含 1 建议项）**：§1.6.3 主选 #1 明确；备选 #2（2-D 行合并）给出 **Stage 4 A/B 裁决计划 + 判定阈值**（备选 latency < 主选 × 0.9 才翻转），§9.1 亦标注该判定依赖未实证常数（行合并收益）。**建议（非阻塞）**：备选 #2 仅有代价论证与结构概述，未给出 `[K_W,BM,BN]` 3-D tile 的 UB 预算/循环结构/weight 转置链位置/分核三要素重算，未达「Stage 4 可直接实现」深度；建议补齐备选结构设计（备选仅服务于大 `L_out`/`ceil(L_out/BN)>1` 的未来 shape，且代表 workload `GX_L=1` 下不影响主选正确性）。
- **布局决策与实现一致 — pass**：§1.6.3 选定「逐行展平 + L 轴向量化」与 §3.3 伪代码（累加维 `K_W`、向量维 L、buffer `[K_W,BN]`/`[1,BN]`）、§4 内存规划、§6 循环结构三方一致；`xpad`/`xdec` 前置路径在 §3.3/§4.4 有对应实现。
- **与下游章节一致 — pass**：§3.1 公式拆解以 §1.6.1「优化后公式 = 基线」为输入；§6 循环结构与 §1.6.2 结论一致。
