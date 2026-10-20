# _conv1d_pointwise_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_pointwise_kernel_DESIGN.md（revision v1；上一版 `history_version/design_v0.md`）
- 任务类型: 迁移（harness，GPU TileOPs-main → NPUIR Ascend）
- 源算子: /mnt/workspace/TileOPs-main/src/tileops/kernels/convolution/conv1d.py（`_conv1d_pointwise_kernel` builder L542-638、`Conv1dPointwiseKernel` L641-742）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0

> 本轮为 v0（`结论: 不通过`，1 阻塞级 + 2 建议级）修订后的**再审**。已逐项复核 v0 三个问题的修复，并独立复推 fix 的正确性（不轻信设计自述）。

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: 不适用（无采纳优化项）**——§1.6.1 结论为「无优化空间」（bias 折叠 / 转置 GEMM / split-K 三候选均无正收益，不采纳；见 §1.6.1 表 + §10.2），无采纳优化项，故按 gate `S1-EQUIV-EXEC` 豁免规则不生成脚本。已确认 `examples/conv1d/_conv1d_pointwise_kernel/verify_equiv.py` **不存在**（`ls` 返回 No such file），豁免合法。
- **design_calc_check**: 任务书写 `--dir`，脚本实际接口为 `--design`（`python3 .agents/tools/design_calc_check.py -h` 确认仅接受 `--design`，且不接受目录），故以脚本自身接口运行 `python3 .agents/tools/design_calc_check.py --design examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_pointwise_kernel_DESIGN.md`，输出：
```json
{
  "design": "examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_pointwise_kernel_DESIGN.md",
  "capacity_constants": {"ub_kb": 192.0, "ub_inflation": 1.7, "l0c_kb": 128.0, "source": "pattern-library/constants.md CONST-capacity-910B2C"},
  "checks": [
    {"check": "l1_budget", "status": "pass", "detail": "L1: 2 buffer entries sum = 48.0 KB; capacity = 512.0 KB", "within_inflation_budget": null},
    {"check": "ub_budget", "status": "pass", "detail": "UB: 4 buffer entries sum = 80.2 KB; capacity = 192.0 KB; x1.7 auto-multi-buffer inflation budget = 113 KB", "within_inflation_budget": true},
    {"check": "l0c_budget", "status": "pass", "detail": "block 64x64 x fp32 accum = 16.0 KB vs L0C 128 KB"},
    {"check": "core_split", "status": "skip", "detail": "insufficient parsed dims {'N': 1, 'bm': 64, 'bn': 128}; declared logical cores = 64"},
    {"check": "r3_metrics", "status": "pass", "detail": "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"}
  ]
}
```
  - `l0c_budget` pass，但机械把 `block_M×block_N` 欠解析为 `64×64`（16KB）；设计实际 `64×128×4 = 32768 B = 32KB`（§4.5）——两者均 ≤ 128KB，**人工复核通过**。
  - `core_split` **skip**（解析到的 `N/bm/bn` 不足，且把与核数无关的 `64` 误当 declared logical cores）→ **人工复核通过**：whisper-1x1 `GY=ceil(1280/64)=20`、`GX=ceil(3000/128)=24`、`n=1` → `480 = 20×24`（物理核 20 的整数倍）；encodec-deep-1x1 `4×5=20`；wide-1x1 `16`（§5.5），三档自洽。
  - 机械 skip 项由维度 2 / 维度 3 人工复核覆盖，结论一致。
- **门禁自检**（`python3 .agents/tools/statectl.py gate 2 --dir examples/conv1d/_conv1d_pointwise_kernel`，exit 0）：
```json
{"action": "gate", "failures": [], "ok": true, "pass": true, "stage": 2, "warnings": []}
```

## 检视详情

### 0. 源算子理解与迁移分析 — pass
已亲自 Read 源算子全文（`conv1d.py` L542-742 本函数 + `Conv1dPointwiseKernel`）并与 §0 逐项核对：

- **语义理解正确**：§0.1 公式 `out[n,c_o,l]=bias[c_o]+Σ_{c_i} W[c_o,c_i]·x[n,c_i,l]` 与源码 `T.gemm(weight_shared, data_shared, out_local)`（`conv1d.py:595`）一致；规约轴 = `C_in`（GEMM K 轴），累加域 `accum_dtype="float"`（`:551`）→ `T.alloc_fragment(out_local, accum_dtype)`（`:572`），bias 在 **fp32 域**相加后统一 cast（`:603` `out_local[i,j] + T.cast(bias[oc], accum_dtype)`，再 `T.cast(..., dtype)`）——与 §0.1 dtype 语义一致。`L_out=L_in`、`kW=1` 退化为纯 GEMM 判断正确（`:565` 用 `ceildiv(l_in,block_n)`）。
- **I/O 与源码签名一致**：§0.2 的 `x(n,c_in,l_in)` / `weight(c_out,c_in)` / `out(n,c_out,l_in)` / `bias(c_out)` 与 `_conv1d_pointwise_main`（`:629-632`）及 bias 变体（`:618-622`）逐字对应；host `weight[:, :, 0].contiguous()`（`:741`）已在 §0.1/§0.3 步骤 1 记录。
- **算法步骤覆盖完整、无臆造**：§0.3 八步覆盖 `T.clear`（`:576`）、`T.Pipelined(ceildiv(c_in,block_k))`（`:580`）、`T.copy(weight...)`（`:581`）、x 的 `T.copy` / 掩码 gather 分支（`:583-593`）、`T.gemm`（`:595`）、epilogue（`:597-611`）、`T.copy(out_shared,...)`（`:613`）；host 侧逻辑已列出，源本无 im2col（未遗漏、未臆造）。
- **优化手段识别完整**：§0.4 覆盖 SMEM tiling（`:570-571`）、tensor-core mma（`:595`）、`T.Pipelined` 流水（`:580`）、fragment fp32 累加（`:572`）、尾块掩码 gather（`:586-593`）、coalesced 连续 `T.copy`（`:584`）、`T.use_swizzle`（`:575`）、`threads=128`（`:568`）；未把实现层优化误判为算法层。
- **耦合性判定与重设计**：§0.5 四态处置（保留/等价替换/重新设计/舍弃）逐项有依据，舍弃项（swizzle / threads）理由写明；§0.6 R1（三维 grid+threads → 一维 `T.Kernel`+cid 解码+persistent）、R2（掩码 gather → 分形 clamp+slice 尾块+裁剪 store）、R3（fragment bias → MixCV Vector 后处理）三要素齐全，语义保持论证成立（映射双射 + 补 0 无贡献 + 裁剪不落 GM；bias fp32 加序不变）。
- **NPU 重设计可行性（v0 fail 项已修复）**：v0 的 R3 缺 Cube 侧写前 WAR 等待。v1 已改为**深度 2 双槽 + 4-flag 双向握手**（§0.6 R3 补「握手正确性条件」四步论证，§3.2/§3.3/§6.2/§7.2 单一来源），竞争消除条件完整（读前就绪 / 写前空闲 / 前导置空闲 / 严格交替），见维度 5 独立复推。
- **§1–§7 与迁移决策一致**；**golden 独立性成立**：§8.1 以 §0.1 语义为唯一依据移植 `F.conv1d`（kW=1），在 fp32 域计算后 cast，不复刻 NPU 数据流。

### 1. API 可行性 — pass
§3.2/§3.4 所列 API 全部有佐证且优先 v-prefix：
- `T.Scope("Cube")`/`T.Scope("Vector")`、`T.rs("PIPE_*")`、`T.sync_block_set/wait` — `_gqa_prefill_fwd_kernel.py`、`_ssd_chunk_scan_fwd_kernel.py`（Cube Expert + MixCV 双 Scope 先例）；`docs/Tilelang.language/同步管道操作/T.sync_block_set.md` / `T.sync_block_wait.md` 签名一致，且 §2.4 规范示例即「Cube 写前 wait / 写后 set、Vector 读前 wait / 消费后 set」的双向形态。
- `T.gemm(a,b,c,size,initC,b_transpose)` — `docs/.../线性代数操作/T.gemm.md`「size=[a,b,c]：src1 [a,b] 或 [b,a]、src2 [b,c] 或 [c,b]、dst [a,c]；fp16 输入 dst 须 fp32」，与设计 `size=[tmc,tkc,tnc]`、dst `l0_c` fp32 一致；先例 gqa `:334`、ssd `:299`。
- `T.copy` slice 形态（GM→L1 自动 ND2NZ、L0C→GM 自动 fixpipe）— `docs/.../内存操作/T.copy.md` §2.3 条目 2/6、§2.4 示例 4 吻合；探针 `examples/conv1d/probe/conv1d_geomm_probe.py` 实测。
- `T.alloc_L1`/`T.alloc_L0C`/`T.alloc_ub`、`T.vbrc`/`T.vadd`/`T.vcast` — 文档 + gqa/ssd 先例（`_ssd_chunk_scan_fwd_kernel.py` `:439` vbrc、`:443` vadd、`:421/:450` vcast）。
- Expert 通用路径 `T.copy` 不跨 dtype（须显式 `T.vcast`）— `T.copy.md` §2.3 条目 2，设计 §4.4/§9.1 已按此声明。
- `NPUUtils.get().get_aicore_num()` 实查（`tilelang/utils/npu_utils.py`）— §5.5 附查询代码与返回值。
- `T.load_nd2nz` 存在但**主动避开**（跨步误读，`TRAP-load-nd2nz-strided` / CG-2026-0004），改用 slice 形态 `T.copy`——与探针实测绕法一致。
无 API 存在性 / 模式匹配问题。

### 2. 内存层级规划 — pass
- 搬运路径完整（§4.4）：无 bias 纯 Cube `GM→L1→L0C→GM`；bias `GM→L1→L0C→GM[ws]→UB→GM[out]`，无 GM→L0 越级。
- L0C：`64×128×4 = 32768 B = 32KB ≤ 128KB`（`CONST-capacity-910B2C`）。
- L1：单槽 `(64×128 + 128×128)×2 = 49152 B = 48KB ≤ 512KB`（双缓冲 96KB）。
- UB（bias）：`ub_f32 32KB + ub_f16 16KB + bias_col 256B + bias_brc 32KB ≈ 80.2KB ≤ 192KB`；机械 `ub_budget` pass（×1.12 显式 alloc 口径 ≈ 90KB，余量 53%）。
- **跨引擎 `ws`（v1 改为双槽）**：`(num_kernels=20, 2, 64, 128) fp32 = 1.31MB`，位于 **GM**、不计入片上容量（§4.5 明确），不发散片上预算；每 buffer 标注层级（§4.3 L1/L0C/GM/UB）。
- 机械 `l1_budget`/`ub_budget`/`l0c_budget` 全 pass（见机械复核）。无阻塞。

### 3. Tiling 策略 — pass
- block 划分明确：§5.2 `block_M/N/K = 64/128/128`，承接源 `default_config`（`conv1d.py:703`），理由充分。
- 对齐：`block_N=128` fp16 尾轴 256B、`block_K=128` 256B（32B 对齐 ✓）；分形 clamp `tmc/tkc/tnc = T.max(16, ceil16(tail))`（§3.3）。
- 非整除：R2 分形 clamp + slice 尾块 + 裁剪 store（§6.4），M/N/K 三处非整除覆盖。
- 分核三要素齐全（§5.5）：① `num_logical = GY×GX×n` 计算与三档 workload 自洽（恢复见机械复核）；② 物理核数 `NPUUtils.get().get_aicore_num()` **实查 = 20**，附查询代码与返回值，并按 `core-split-strategy.md`「Cube/混合算子直取返回值、纯 Vector 才翻倍」**不翻倍**——判定正确（与 pattern-library `CONST-aicore-910B2C` 的「本机返回 20、以实查为准」注记一致）；③ 统一 persistent 结构 + 核内 `T.serial(num_local_tasks)`，边界静态。
- 机械 `core_split` skip → 人工复核通过（见机械复核段）。无阻塞。

### 4. 技术约束检测 — pass
- 三维 Kernel：源三维 `T.Kernel(bx,by,bz)` 已由 §0.6 R1 改为一维 `T.Kernel(num_kernels, is_npu=True)` + cid 解码（§3.5.1）。
- GEMM 非整除：已在维度 3 处理（§6.4）。
- L0C 溢出：已在维度 2 处理（32KB）。
- `T.load_nd2nz` 跨步误读（`TRAP-load-nd2nz-strided` / CG-2026-0004）已识别并改用 slice `T.copy`（§3.5.1、§9.1）。
- Expert 通用路径跨 dtype（`TRAP-C12-copy-dtype-cast`）已识别，output cast 由显式 `T.vcast` 承担（§4.4、§9.1、§9.2 解决表）。
- 尾块越界写（`TRAP-T-copy-region-semantics`）已识别，一律双显式 slice（§6.4、§9.2）。
无阻塞。

### 5. 循环与同步 — pass
- 循环结构清晰（§6.1：核级并行 / 核内 `T.serial(num_local_tasks)` / K 分块 `T.serial(ceil(c_in/block_K))` / Cube `T.gemm` / Vector `vbrc+vadd+vcast`），与 §1.6.2 一致，无逐元素标量循环。
- **v0 阻塞项（§7.2 与 §3.3/§6.2 自相矛盾 + 单槽 WAR 竞争）已修复，且独立复推为无竞争**：
  - **协议单一来源一致**：§3.2（步骤 5/6）、§3.3（`_body` Cube/Vector 段 + 注 3/注 4）、§6.2（循环伪代码）、§7.2（同步点表 + 协议概要）现**逐字段一致**——`slot = task_id % 2`；Cube 写 `ws` **前** `T.sync_block_wait(2*slot+1)`（WAR）、写后 `T.sync_block_set(2*slot)`（ready）；Vector 读 `ws` **前** `T.sync_block_wait(2*slot)`、消费写出后 `T.sync_block_set(2*slot+1)`（归还）；Cube 段前导 `T.sync_block_set(1)/T.sync_block_set(3)` 置两槽初始空闲。三节的 **flag 编号、set/wait 方向、放置位置（写前/读前/消费后）** 经逐字段核对无矛盾（v0 为单槽 + Cube 侧无 wait，已消除）。
  - **独立时间线复推（race-free）**：flag0=ready(slot0)、flag1=free(slot0)、flag2=ready(slot1)、flag3=free(slot1)。前导 set(1)、set(3) 使 Cube task0/task1 的写前 wait 立即通过；Cube task0 写 ws[0]→set(0)，Vector task0 wait(0) 通过后读 ws[0]→set(1)；Cube task2 的 wait(1) 直到 Vector task0 归还才通过——生产者最多领先消费者 2 个任务，**Cube task t+2 覆盖 ws[slot] 必晚于 Vector task t 读毕**（WAR 闭合），**Vector 读 ws[slot] 必晚于 Cube 同任务写毕**（RAW 闭合）。每个 flag id 严格 set/wait 交替（flag0/flag2 由 Cube set、Vector wait；flag1/flag3 由前导 + Vector set、Cube wait），计数不欠账，无死锁。
  - **对齐已验证先例**：结构即 `_ssd_chunk_scan_fwd_kernel.py`（PL-1.12）的「深度 2 双槽 + 4-flag（ready/cons × slot）+ 消费侧前导 set」形态——ssd `:253-255` 前导 `set(1)/set(3)`、`:398-399` 生产者写前 `wait(2*slot+1)`（其注释明确「placing the wait BEFORE any ws[slot] write prevents the off-by-one WAR race」）、`:456-457` 写后 `set(2*slot)`、`:266-267` 消费者读前 `wait(2*slot)`、`:341-342` 消费后 `set(2*slot+1)`。本设计的 Cube/Vector 生产者-消费者角色与 ssd 互为镜像，flag 语义（`2*slot`=ready、`2*slot+1`=free）与 ssd **同名同义**，前导置于 Cube（生产者）Scope 仅初始化、不构成竞争（先于首个 wait）。`docs/.../T.sync_block_wait.md` §2.4 规范示例同款（写前 wait）。
- §6.3 说明 Stage 4 可对 K 分块启用 `T.Pipelined`/L1 ping-pong（与 bias 交接双槽正交）；§6.4 尾块逻辑完整。无阻塞。

### 6. 验证方案 — pass
- §8.1 golden 以 §0.1 语义为依据的 `F.conv1d`（kW=1）参考实现，fp32 域计算后 cast，与源累加域一致且**不复刻** NPU 算法（独立性成立）。
- §8.3 L0 计划：代表 workload `whisper-1x1`、`encodec-deep-1x1`，覆盖 `has_bias=False/True` 两条路径；必测点含非整除（`C_out=100/L=1000/C_in=50`）与 batch>1。
- §8.2 按 dtype 给 atol/rtol（fp16 1e-2、fp32 1e-4、bf16 1.6e-2），与源仓容差口径一致。无阻塞。

### 7. 完整性与一致性 — pass
- 章节齐全（含 §0 的 0.0–0.7 子节；v1 新增 §0.0 revision 记录），全文无占位符标记（未检出模板占位符、未完成标记或空缺项标记）。
- §3.5.3 引用的 `examples/conv1d/probe/conv1d_geomm_probe.py`、`.../attention/.../_gqa_prefill_fwd_kernel.py`、`.../mamba/ssd_chunk_scan/.../_ssd_chunk_scan_fwd_kernel.py`、`examples/TileOPs/tileops/kernels/convolution/conv1d/_conv1d_fwd_kernels.py`、`examples/TileOPs/tests/ops/test_conv1d.py` 均**真实存在**（已核对）。
- **两处 v0 内部不一致已修复**：① （问题 1）§7.2 同步表与 §3.3/§6.2 伪代码现一致（见维度 5）；② （问题 2）`ws` 已在 §3.3 的 `_pointwise_bias_main` 签名显式声明 `ws: T.Tensor((num_kernels, slots, block_M, block_N), "float32")`（对齐 ssd `ws_c`/`ws_lcb` 形参），并写明 wrapper `torch.empty((num_kernels, 2, block_M, block_N), dtype=torch.float32)` 分配与 `launch` 传参（§3.3 注 4、§4.3、§4.5）。
- API 映射（§3.2）、内存规划（§4.3/§4.4/§4.5）、Tiling（§5）三处对 `ws` 双槽、UB 预算、操作数轴描述相互一致，无矛盾。无阻塞。

### 8. 算法优化分析 — pass
- **§1.6 四子节齐全**（1.6.0 调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代 / 1.6.3 向量化轴与布局）。
- **调研完整性**：R1 候选表含**基线**（`out=W@x+bias`）；R2 在线算法给出明确「无（单遍 K 分块累加即最优）」结论；R3 复杂度**四口径**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值，基线在表）；R4 逐候选硬件亲和评估。
- **v0 建议项（问题 3）已修复——`ALG-conv` 命中行候选全覆盖**：核对 `tilelang-op-design/references/algorithm-candidates.md` 的 `ALG-conv` 命中行原文为「R1：im2col + GEMM；implicit GEMM；Winograd……」。v1 §1.6.0 R1 现补入候选 5（**im2col + GEMM**：kW=1 退化为恒等 ≡ 基线）与候选 6（**Winograd**：kW=1 无窗口、不适用），并注明 **implicit GEMM 即基线候选 0**——三项命中行候选均显式评估，无漏候选。
- **调研结论正确性（独立复核/复算）**：
  - R1 负向断言「无更优等价化简」——独立复核成立：pointwise（kW=1）即 GEMM，`2·C_out·L·C_in` 已是该语义 FLOPs 下界；bias 折叠改变 `K=C_in+1`（非 16 对齐）并需物化 `x_aug`（净亏）；转置是布局问题（非公式化简，归 §1.6.3）；split-K 增跨核归并代价；im2col 在 kW=1 退化、Winograd 需 kW≥2。
  - R2 负向断言「无在线变体」——独立复核成立：输出沿 N 轴相互独立、唯一规约是 K 轴求和，K 分块 running 累加本身即单遍流式，不存在 running-max/sum 重缩放（对照 `algorithm-candidates.md` ALG-conv「滑窗天然流式」），无 Welford/online-softmax 类遗漏。
  - R3 独立复算：FLOPs `2·C_out·L·C_in`；whisper `2·1280·3000·80 = 6.144e8 ≈ 0.61 GFLOP` ✓；访存 `x 80·3000·2 = 0.48MB + W 1280·80·2 = 0.20MB + out 1280·3000·2 = 7.68MB ≈ 8.36MB` ✓；无 bias 时**无中间 GM 往返**、bias 路径额外 `2·C_out·L·4B` ws 往返（§1.6.0 已计）；扫描 1 遍、缓冲 `O(tile)` ✓。未发现漏计中间缓冲 GM 往返。
  - R4 亲和淘汰引用 `pattern-library/constants.md`（`CONST-capacity-910B2C`、`CONST-mte2-degradation`、`CONST-L1-port-bw`、`CONST-store-fixpipe-gm-only`、`CONST-vector-launch-overhead`、`CONST-aicore-910B2C`）与 `hardware-cost-model.md`；均已确认条目存在，判定与实测条目不矛盾（`CONST-store-fixpipe-gm-only` 证实跨引擎传输强制 GM 往返 → ws 中继合法；`CONST-aicore-910B2C` 注记 20 核实查）。
- **弃选论证前提核对（亲自打开文档）**：§1.6.3 候选 #2/#3 的 repack 代价前提经独立查证一致——核内 `T.transpose` 二轴交换链实测 **~5.4µs/4 次**（`layout.md` PL-1.1 L27，且 L28 明确「转置仅支持二轴交换、reshape→transpose 链 mis-compile」）；host permute 两次实测 **106–147µs**（`layout.md` PL-1.3 L63-65）；epilogue 活跃源 transpose 毒化 **2.6x**（`traps-compiler.md` TRAP-transpose-epilogue-poison L178-190）。设计把「核内重排」写为核内 UB 级、把 host permute 与核内 transpose 分开量化，**前提与文档一致，无「GM 级 transpose」类矛盾**；负向论断均有 `docs/`/pattern-library 实测佐证。
- **调研→设计一致性**：选定算法（基线 plain implicit GEMM + optional bias）实际驱动 §1.4、§3.1（以优化后=基线公式为输入）、§6（K 分块 `T.serial` + Cube `T.gemm`）；无「调研选 A 设计做 B」。迁移任务 §0.5/§0.6 处置与选族一致，源优化手段意图承接逐项说明（§0.6 末段）。
- **数学等价优化完整性**：§1.6.1 逐项评估，采纳项为空并写明「无优化空间」的充分依据（纯 GEMM+加法，无除法/超越函数/公共子表达式/稳定化需求）；无采纳项故**无等价性独立推演对象**，`verify_equiv.py` 豁免合法。
- **向量化替代完整性**：§1.6.2 覆盖 §3.3/§6 全部循环/标量点（7 项）；逐元素点（bias 相加、输出 cast）已向量化（`vbrc`/`vadd`/`vcast`，ssd 佐证），保留的 3 处标量（cid 解码 / 尾块尺寸 / cid 钳位）理由充分（block 级映射 / 动态边界，非逐元素热点）；矩阵乘落 Cube 为正确单元选择（非保留标量循环）。
- **向量化轴与布局决策完整性**：§1.6.3 候选矩阵按 Cube 类枚举（原生 A/B 轴、转置、host permute、`b_transpose` 变体），逐候选量化评分，弃选均给量化理由；明确 I/O layout 契约 ≠ 核内布局；选定 #1（A=`W[C_out,C_in]`、B=`x[C_in,L]`、dst L0C`[C_out,L]`，**无 repack**）与 §3.3 伪代码 / §4 L0C tile 形状 / §6 循环三方一致。
- **与下游章节一致**：§3.1 以优化后（=基线）公式为输入；§6 不出现 §1.6.2 未论证的逐元素标量循环；§1.6.3 选定轴与 §6 内层 `size=[M,K,N]` 一致。无阻塞。

## Review Result
- stage: 2
- project: conv1d
- operator: _conv1d_pointwise_kernel
- output: examples/conv1d/_conv1d_pointwise_kernel/REVIEW.md
- is_migration: true
- source_op_path: /mnt/workspace/TileOPs-main/src/tileops/kernels/convolution/conv1d.py
- conclusion: 通过
- validation: pass
- validation_details:
  - 结论行存在: pass
  - 结论一致: pass
  - 维度完整（迁移 9）: pass
  - 维度 0 有源码证据: pass
  - 维度 8 有推演证据: pass
  - 建议完整: n/a（通过，无阻塞级问题）
  - 无占位符: pass
- blocking_issues: 0
- migration_findings: 语义/pass 算法/pass 优化/pass 耦合性/pass 重设计/pass（v0 R3 握手缺陷已修复） 一致性/pass golden独立性/pass
- algo_opt_findings: 调研完整性/pass 调研结论正确性/pass 调研→设计一致性/pass 优化完整性/pass 等价论证/pass（无采纳项） 向量化覆盖/pass 不可替代理由/pass 替代API佐证/pass 与§3§6一致性/pass
- suggestion_count: 0
- skills_consulted: .agents/skills/tilelang-design-review/SKILL.md; .agents/skills/_shared/standards/gate-and-retry.md; .opencode/agents/tilelang-design-reviewer.md
- summary: v0 唯一阻塞项（MixCV 单槽 ws 缺 Cube 侧 WAR 等待）已改为深度 2 双槽 + 4-flag 双向握手，独立复推无竞争且与 ssd PL-1.12 先例一致，§3.3/§6.2/§7.2 三方一致；另两处建议项（ws 声明、ALG-conv 候选行）均已补齐——9 维度全 pass，结论通过。
- issues: none
