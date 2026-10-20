# _conv1d_unit_stride_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_unit_stride_kernel_DESIGN.md（v1，上一版 `history_version/design_v0.md`）
- 任务类型: 迁移（harness，GPU TileOPs-main → NPUIR Ascend，Expert Cube）
- 源算子: /mnt/workspace/TileOPs-main/src/tileops/kernels/convolution/conv1d.py（`_conv1d_unit_stride_kernel` builder L151-313；`Conv1dUnitStrideKernel` L874-926，`general=False`）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 0

> 复检对象为 v0 REVIEW「结论: 不通过」后的修订版（v1）。v0 的 1 个阻塞项（§8.3 L0 必测点 ⑤ `taps=2`：K 循环装入 `taps` 个 tap 但 `T.gemm` 的 K extent 取单 tap 宽度 `kcl`，第二 tap 被丢弃）与 2 个建议项（`w_flat` 补零列数值前提 / UB 预算论证口径）在 v1 中逐项核对为**已修复**，且未引入新问题。

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: 不适用（无采纳优化项）**——§1.6.1「优化结论」为「无优化空间」（Winograd / bias 折叠 / 转置 GEMM / split-K 四候选经 §1.6.0 R1/R3/R4 评估均无正收益，不采纳；§1.6.1 第 5 行为「不适用」），按 gate `S1-EQUIV-EXEC` 豁免规则不生成脚本。已确认 `examples/conv1d/_conv1d_unit_stride_kernel/verify_equiv.py` **不存在**（`ls` 返回 `No such file or directory`），豁免合法。v1 未新增任何采纳的公式级优化项，故本次复检仍无等价性机器验证对象。
- **design_calc_check**（脚本接口为 `--design`）：`python3 .agents/tools/design_calc_check.py --design examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_unit_stride_kernel_DESIGN.md` 输出：
```json
{
  "design": "examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_unit_stride_kernel_DESIGN.md",
  "capacity_constants": {
    "ub_kb": 192.0,
    "ub_inflation": 1.7,
    "l0c_kb": 128.0,
    "source": "pattern-library/constants.md CONST-capacity-910B2C"
  },
  "checks": [
    {
      "check": "l1_budget",
      "status": "pass",
      "detail": "L1: 2 buffer entries sum = 48.0 KB; capacity = 512.0 KB",
      "within_inflation_budget": null
    },
    {
      "check": "l0c_budget",
      "status": "pass",
      "detail": "block 64x64 x fp32 accum = 16.0 KB vs L0C 128 KB"
    },
    {
      "check": "core_split",
      "status": "skip",
      "detail": "insufficient parsed dims {'N': 1, 'bm': 64, 'bn': 128}; declared logical cores = 64"
    },
    {
      "check": "r3_metrics",
      "status": "pass",
      "detail": "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"
    }
  ]
}
```
  - `l1_budget` **pass**：单槽 `l1_a(64×128)+l1_b(128×128)` fp16 = 48KB ≤ 512KB（§4.5）。双槽 96KB、`taps=2` 单槽 96KB 未被机械合计行捕获，人工核对均 ≤ 512KB。
  - `l0c_budget` **pass**：机械把 `block_M×block_N` 欠解析为 `64×64`（16KB）；设计实为 `64×128×4 = 32768 B = 32KB`（§4.5）——两者均 ≤ 128KB，人工复核通过。
  - `core_split` **skip**（解析到的 `N/bm/bn` 不足，且把与核数无关的 `64` 误当 declared logical cores）→ **人工复核通过**：whisper-conv1 `GY=ceil(1280/64)=20`、`GX=ceil(2998/128)=24`、`n=1` → `480 = 20×24`（物理核 20 的整数倍）；encodec-deep `4×5=20`；非整除 `2×8=16`；batch>1 `3×3×3=27`（§5.5），四档自洽且 `num_local_tasks` 为编译期常量。
  - `r3_metrics` **pass**：R3 四口径（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）齐备。
- **门禁自检**（`python3 .agents/tools/statectl.py gate 2 --dir examples/conv1d/_conv1d_unit_stride_kernel`）：
```json
{"action": "gate", "failures": [], "ok": true, "pass": true, "stage": 2, "warnings": []}
```

## 检视详情

### 0. 源算子理解与迁移分析 — pass
已亲自 Read 源算子全文（`conv1d.py` L151-313 本函数 builder + `Conv1dUnitStrideKernel` L874-926）与 v1 §0 逐项核对；**重点复核 v0 阻塞项（K/taps）的修复**。

- **语义/I/O/优化/耦合/golden（v0 通过项，v1 未改动，复核仍成立）**：
  - 语义：§0.1 公式 `out[n,c_o,l]=bias[c_o]+Σ_{c_i,k_w} W[c_o,c_i,k_w]·x[n,c_i,l+k_w·d_W-p_L]`（stride=1）、`L_out` 公式与源码 `out_l = l_in + pad_left + pad_right - dilation_l*(kernel_l-1)`（源码 L169）逐字一致；builder 入口断言 `stride_l != 1 → ValueError`（源码 L165-166）、`Conv1dUnitStrideKernel.refusal` 排除 `stride≠1 | kernel_l≤1 | c_in<16`（源码 L885-889）与 §0.1/§3.5.1 一致。dtype 语义（同 dtype I/O、fp32 累加、bias 在 fp32 域相加后 cast）与 `accum_dtype="float"`（L167）+ `out_local[i,j]+T.cast(bias[oc],accum_dtype)`（L278）一致。
  - 算法步骤：§0.3 十步覆盖 `T.Pipelined(kernel_l//taps·channel_blocks)`（L240）、`_load_tap` A 装载（L193-197）、B 快/慢路径（L198-213）、`T.gemm`（L270）、epilogue（L272-286）、输出写回（L288）、`taps=2` 第二 tap（L256-269）、`kw/c0` 解码（L241-242）；host `weight.permute(0,2,1).contiguous().view(c_out,k_total)`（L870）已列出——无遗漏、无臆造步骤（源无 im2col 物化）。
  - 优化手段：§0.4 覆盖 tap-major 通道分块（源码注释 L231-233）、`taps∈{1,2}`（L256-269）、SMEM tiling（L223-224）、tensor-core mma（L270）、`T.Pipelined`（L240）、fragment fp32 累加（L225）、interior 快路径（L198-204）、慢路径掩码 gather（L205-213）、host permute（L870）、coalesced `T.copy`、`T.use_swizzle`（L228）、`threads`（L221）、`disable_tma`（L196/L203）；未把实现层优化误判为算法层。
  - 耦合性/重设计：§0.5 四态处置逐项有依据，舍弃项（swizzle / threads / disable_tma）理由写明；§0.6 R1–R5 三要素齐全，语义保持论证成立。**stride=1 特化减法正确**：源 builder 断言 `stride_l==1`，函数 2 的 polyphase/`xdec`/`T.gather` 路径在此不触发（§0.5「无 stride>1 条目」），全文无 polyphase 残留（§1.5 数据流图无 `y_poly`/`xdec`；§4.3 buffer 表无 `xdec`/`poly_idx`）。
  - golden 独立性：§8.1 以 §0.1 语义为唯一依据移植 `F.conv1d`（stride=1），fp32 域计算后 cast，**不复刻** §0.6 的 Cube/persistent 数据流。

- **v0 阻塞项修复复核（K extent / L1 layout / 对齐）—— 已修复且自洽**：
  - **根因与源语义**：v0 K 循环把一个 k tile 装入 `taps` 个相邻 tap 的 A/B 片，但 `T.gemm` 的 K（`size[1]`）取单 tap 的 `kcl`；`T.gemm` 的收缩长度即 `size[1]`（`docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1「size=[a,b,c] → src1 [a,b]、src2 [b,c]」），故 `taps=2` 时第二个 tap 被丢弃。源算子 `T.gemm(weight_shared, data_shared, out_local)`（源码 L270）在 `taps=2` 时 K = 全 `taps*block_k`（无 `size` 参数），两 tap 均参与——v1 现与之对齐。
  - **v1 修法（§0.0）**：① L1 K 轴按 tap 连续打包——第 t 个 tap 写 `l1_a[0:tm, t·kcl : t·kcl+kcl]` / `l1_b[t·kcl : t·kcl+kcl, 0:tn]`（v0 为偏移 `t·block_k`）；② `T.gemm` 的 `size=[tmc, taps·kcl, tnc]`，与装入宽度逐字相等；`taps=1` 退化为 `kcl`（基线不变）。
  - **独立推演（对齐 + 无 gap + 无遗漏）**：`c_in_pad = max(16, ceil16(c_in))` 与 `block_k` 均为 16 的倍数 ⇒ `kcl = min(block_k, c_in_pad - kc·block_k)` 为 16 的倍数 ⇒ 组内偏移 `t·kcl`（`t∈[0,taps)`）均为 16 元素对齐、L1 可寻址。装入区间 `[0, taps·kcl)` 为全部 `taps` 个 tap 的连续拼接（无 `kcl<block_k` 时的 stale gap），gemm K = `taps·kcl` 恰好读全该区间——既不污染有效输出也无遗漏。独立按 encodec-deep（`c_in=128, block_k=128, channel_blocks=1, kernel_l=10, taps=2`）与 `c_in_pad=160`（`channel_blocks=2, kcl=128/32`）两档复算：前者每步 K=256 覆盖 `kw∈{0,1},{2,3},…`；后者不同 kc 的 `taps·kcl` 分别为 256/64，逐 kb 累加总量 = `C_in·K_W`，均正确。
  - **跨章节一致性核对**：`t·kcl` 打包 + `size=[tmc, taps·kcl, tnc]` 在 §0.6 R2（L158）/R4（L173）、§1.6.2（L319）、§1.6.3 布局结论（L361）、§3.1 步骤 6（L398）、§3.2 步骤 5（L411）、**§3.3 伪代码（L508-523）与注 3（L605）**、§4.3（L704-705）、§5.2/§5.3/§5.4（L810/L818/L822）、**§6.1/§6.2/§6.4（L872/L876/L900-902/L931）**、**§8.3 必测点 ⑤（L1017）**、§9.2（L1052-1053）**逐处一致**；无任何残留 `t*block_k` 或 `size=[tmc,kcl,tnc]` 的 v0 表述（DESIGN.md 中 `kcl`/`taps·kcl` 仅出现在精确尾块与修复说明语境）。**唯一可议**：§1.6.3 候选矩阵 #1 行「预估收益/代价」列的「L1 目标偏移 0」措辞（L352）宜按同节布局结论（L361）/§3.3 读作「通道块 K 基址为 0、组内第 t 个 tap 偏移 `t·kcl`」；该行非决策行、与 §3.3 伪代码不矛盾，不影响正确性。
  - **防复发机制**：§0.0 把「L1 装入 K 宽度 == `T.gemm` 的 `size[1]`」定为单一来源约束并写入 §3.3 注 3 / §5.2 / §5.4 / §6.1 / §6.2 / §6.4；§8.3 必测点 ⑤（`taps=2` 与 `taps=1` 结果 `assert_close` 一致）直接门住该缺陷——在 v0 形态必失败、在 v1 形态通过。收敛合法。
- **§1–§7 与 §0.5/§0.6 决策一致**（并行组织→§5.5/§6，尾块→§6.4，weight-layout→§3.2/§4.4，B 装载→§3.2/§4.4，bias→§7.2），无「§0 判重设计、§3 照抄源方案」脱节。

### 1. API 可行性 — pass
§3.2/§3.4 所列 API 均有佐证、优先 v-prefix、与模式（Expert）匹配：
- `T.Scope("Cube")`/`T.Scope("Vector")`、`T.rs("PIPE_*")`、`T.sync_block_set/wait` — `_gqa_prefill_fwd_kernel.py`、`_ssd_chunk_scan_fwd_kernel.py`（Cube Expert + MixCV 双 Scope 先例）；`docs/Tilelang.language/同步管道操作/T.sync_block_set.md`、`T.sync_block_wait.md` 签名一致。
- **`T.gemm` 与 K extent** — `docs/Tilelang.language/线性代数操作/T.gemm.md` §2.1「size=[a,b,c]：src1 [a,b]、src2 [b,c]、dst [a,c]」，即收缩长度 = `size[1]`；设计 `size=[tmc, taps*kcl, tnc]` 与装入宽度逐字相等（与 v0 阻塞项修复一致）；fp16 输入 dst 须 fp32（设计 `l0_c` fp32 ✓）；`size[1]` 可小于 L1 物理 K 维、只取前 `size[1]`（函数 2 `_conv1d_kernel.py` L294-301 以 `size=[tmc,kcl,tnc]` 从 `l1_a[block_m,block_k]` 取 `kcl` 列为先例；探针同样以 `size=[BLOCK_M,K_TOTAL,BLOCK_N]` 配 per-tap B）。
- `T.copy` slice 形态（GM→L1 自动 ND2NZ；L0C→GM 自动 fixpipe；GM↔UB 同 dtype）— 探针 `examples/conv1d/probe/conv1d_geomm_probe.py` L83-101 实测（`xp[0:C_IN, bx*BLOCK_N+kw : +BLOCK_N] → l1_b[kw*C_IN:(kw+1)*C_IN, 0:BLOCK_N]` 的 **per-tap 非零 K 偏移装填**即 v1 打包形态的直接实证）。
- `T.transpose(src,dst,permutation)` — `docs/Tilelang.language/创建操作/T.transpose.md` §2.3「仅支持转置两个轴，>2 轴分解为相邻轴交换链」，其示例 `(A,B,C) -[0,2,1]-> (A,C,B) -[1,0,2]-> (C,A,B)` 与设计 weight `[C_out,C_in,K_W]→[K_W,C_out,C_in]` 的两次交换链一致；fp16 ✓。
- `T.alloc_L1`/`T.alloc_L0C`/`T.alloc_ub`、`T.vbrc`/`T.vadd`/`T.vcast` — 文档 + `_ssd_chunk_scan_fwd_kernel.py` 先例。
- Expert 通用路径 `T.copy` 不跨 dtype（须显式 `T.vcast`）— 与 `_conv1d_kernel.py` 实现一致。
- `NPUUtils.get().get_aicore_num()` 实查（§5.5 附查询代码与返回值 20）；`T.load_nd2nz` 主动避开（跨步静默误读，`TRAP-load-nd2nz-strided`/CG-2026-0004），改用 slice `T.copy`——`examples/conv1d/_conv1d_kernel/repro/TRAP-copy-ub-offset-stride.py` 与探针实测先例存在。
无 API 存在性 / 模式匹配问题。

### 2. 内存层级规划 — pass
- 搬运路径完整（§4.4）：无 bias 纯 Cube `GM→L1→L0C→GM`；bias `GM→L1→L0C→GM[ws]→UB→GM[out]`，无 GM→L0 越级；每 buffer 标注层级（§4.3）。
- L0C：`64×128×4 = 32KB ≤ 128KB`（`CONST-capacity-910B2C`）。
- L1：单槽 `(64×128 + 128×128)×2 = 48KB ≤ 512KB`；双缓冲 96KB；`taps=2` 单槽 `(64×256 + 256×128)×2 = 96KB ≤ 512KB`（§4.5），余量充分。
- UB（**v1 建议 3 修复**）：§4.5 改为「同时驻留总和」口径——`wsrc+wt+wdst` 各 `16·c_in·K_W·2B`（encodec-deep 各 40KB×3=120KB）+ `ub2 = ch_xp·l_in·2 ≤ 32KB`（encodec-deep ≈18.75KB），合计 ≈138.75KB ≤ 192KB（余量 ≈53KB；whisper-conv1 ≈45.9KB）；bias 段 ≈80KB 与前置段不同时驻留；附 `bm_w`/`ch_xp` 降档 guard 与 `TRAP-UB-multibuffer-inflation` 复算提示。删除 v0「单段峰值 120KB」的复用假设，与 §3.3 `_prelude` 的实际分配范围一致。
- 跨引擎 `ws`（`20×2×64×128` fp32 = 1.31MB）位于 GM、明确不计入片上容量（§4.3/§4.5）。
- 机械 `l1_budget`/`l0c_budget` pass（见机械复核，人工复核实际值 ≤ 容量）。

### 3. Tiling 策略 — pass（v0 阻塞项已消除）
- block 划分明确（§5.2 `block_M/N/K = 64/128/128, taps=1`，承接源 `Conv1dUnitStrideKernel.default_config`，源码 L891-893）；对齐满足（`block_N=128` fp16 尾轴 256B、`block_K=128` 256B，32B 对齐 ✓；`c_in_pad = max(16, ceil16(c_in))`）。
- **K 方向 tap 分组与 `T.gemm` K extent 已自洽**：§5.2「单一来源约束」+ §5.4 明确「K 循环每步 L1 装入宽度 = `taps*kcl`，`T.gemm` 的 `size[1]` 必须同为 `taps*kcl`」；§5.3 分形约束行同口径。v0 的「装多取少」不再存在。
- 非整除：三处（`c_out % block_M`、`out_l % block_N`、`c_in_pad % block_K`）由 §6.4/§3.3 的 `tm/tn/kcl = T.min(block, remain)` + 分形 clamp `T.max(16, ceil16(tail))` + slice 尾块 + 裁剪 store 覆盖；`K extent = taps·kcl`（精确尾块）。
- 分核三要素齐全（§5.5）：① `num_logical = GY×GX×n`，whisper-conv1 480 / encodec-deep 20 / 非整除 16 / batch>1 27 四档自洽；② 物理核数 `NPUUtils.get().get_aicore_num()` **实查 = 20** 并附查询代码，按 `core-split-strategy.md`「Cube/混合算子直取返回值、不翻倍」判定正确（并与 `CONST-aicore-910B2C` 的 24 差异明确以实查为准）；③ 统一 persistent + 核内 `T.serial(num_local_tasks)`，边界静态、`cid` 钳位防 idle 核越界。

### 4. 技术约束检测 — pass
- 三维 Kernel：源三维 `T.Kernel(bx,by,bz)` 已由 §0.6 R1 改为一维 `T.Kernel(num_kernels, is_npu=True)` + cid 解码（§3.5.1）。
- GEMM 非整除：维度 3 处理（§6.4）。L0C 溢出：维度 2 处理（32KB）。
- 其余已知陷阱均在 §3.5.1/§9.1/§9.2 识别并给出方案：`T.load_nd2nz` 跨步误读（slice `T.copy`）、Expert 通用路径跨 dtype（显式 `T.vcast`）、尾块越界写（双显式 slice）、`initC` 组合条件段错误（单比较 `initC=(kb==0)`）、idle 核越界（cid 钳位）、非 16 对齐 `c_in` 的 per-tap L1 偏移（组内 tap 连续打包 `t·kcl` + `c_in_pad`）、`x_pad` UB 侧偏移静默错值（偏移放 GM 侧）、Expert Vector 逐元素 `T.Parallel` 崩溃（stride=1 无 polyphase 天然规避）。

### 5. 循环与同步 — pass（v0 warn 已消除）
- 循环结构清晰（§6.1：核级并行 / 核内 `T.serial(num_local_tasks)` / K 分块 `T.serial(n_kb)` / tap 分组 `T.serial(taps)` / Cube `T.gemm` / Vector `vbrc+vadd+vcast`），与 §1.6.2 一致，无逐元素标量循环、无 Vector 逐元素 `T.Parallel`。**K 分块循环（含 `taps` 内层）现与 gemm K extent 自洽**（§6.1/§6.2/§6.4 均 `size=[tmc, taps*kcl, tnc]`，§6.2 伪代码 `load l1_a ... (L1 K 轴偏移 t*kcl)`），v0 的循环侧不自洽已消除。
- 跨引擎握手独立复推为 race-free：flag `4` = 前置就绪（Vector 写后 `set` / Cube 读前 `wait`）；`2*slot` = ws 就绪（Cube 写后 `set` / Vector 读前 `wait`）；`2*slot+1` = 槽空闲（Vector 消费后 `set` / Cube 写前 `wait`）。「槽空闲」前导 `set(1)/set(3)` 置于**消费者（Vector）Scope**（§3.3 L545-547、§6.2、§7.2）——每个 flag id 的 set 方与 wait 方恒为不同引擎，无自 set→自 wait（`TRAP-sync-self-set-wait`，`examples/conv1d/_conv1d_pointwise_kernel/repro/TRAP-sync-self-set-wait.py`）；深度 2 使 Cube(t) 与 Vector(t-1) 并发且 slot 不冲突，flag id 用量 ≤5 ≤ `CONST-flag-id-budget`（15/核）。
- §6.3 说明 Stage 4 可对 K 分块启用 `T.Pipelined`/L1 ping-pong（`PL-1.12-task-pipeline-depth2`），与 bias 交接双槽正交；§6.4 尾块逻辑完整。

### 6. 验证方案 — pass
- §8.1 golden 以 §0.1 语义为依据的 `F.conv1d`（stride=1）参考实现，fp32 域计算后 cast，与源累加域（`accum_dtype="float"`）一致且**不复刻** NPU 算法（独立性成立）。
- §8.2 按 dtype 给 atol/rtol（fp16 1e-2、fp32 1e-4、bf16 1.6e-2），与源仓容差口径一致。
- §8.3 L0 计划：representative workload（whisper-conv1 / encodec-deep / 非整除 / dilation>1 / batch>1 / `taps=2` 变体）覆盖 `has_bias=False/True` 两条路径；必测点 ①–⑤ 覆盖非整除、bias 两路径、batch>1、padding+dilation、`taps=2`。**必测点 ⑤（`taps=2` 与 `taps=1` 结果一致）现可门住 v0 缺陷且依 v1 形态可通过**（§3.3/§6 已装全 tap、gemm K = `taps·kcl`）。

### 7. 完整性与一致性 — pass（v0 warn 已消除）
- 章节齐全（§0.1–0.7、§1–§10，含 §1.6.0/1.6.1/1.6.2/1.6.3）；全文未见模板占位符残留（已对常见占位符标记逐一 grep 核对，均无）。
- §3.5.3 引用的 `examples/` 路径**均真实存在**（逐个 Glob 核对：`_conv1d_kernel/_conv1d_kernel.py`、`_conv1d_kernel/DESIGN.md`、`probe/conv1d_geomm_probe.py`、`_conv1d_pointwise_kernel/DESIGN.md`、`_gqa_prefill_fwd_kernel.py`、`_ssd_chunk_scan_fwd_kernel.py`、`convolution/conv1d/_conv1d_fwd_kernels.py`、`tests/ops/test_conv1d.py`）。
- **v0 的两处内部矛盾均已修复**：① §5.4/§3.3 注 3 与 §3.1/§3.2/§3.3/§6.1/§6.2/§6.4 的 K extent 描述现已统一为 `taps·kcl`（§5.4 与伪代码一致）；② §4.5 改为同时驻留总和口径，与 §3.3 `_prelude` 分配范围一致。API 映射 / 内存规划 / Tiling 三处描述相互一致。

### 8. 算法优化分析 — pass
- **§1.6 四子节齐全**（1.6.0 调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代 / 1.6.3 向量化轴与布局）。
- **算法调研存在性与完整性**：R1 候选表含**基线**（implicit GEMM，tap-major 通道分块）并覆盖 `algorithm-candidates.md` `ALG-conv`（L44-50）命中行候选——im2col+GEMM、implicit GEMM、Winograd（另评估 FFT/split-K/bias 折叠/转置/direct）；R2 对在线算法给出明确「无独立在线变体」结论并附结构依据；R3 复杂度**四口径**齐全（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值，基线在表，口径标明含/不含），机械 `r3_metrics` pass；R4 逐候选硬件亲和评估并引用 pattern-library 常数。
- **调研结论正确性（独立复核）**：
  - **负向断言复核**：R2「无独立在线变体」——conv 每个输出位置独立、唯一规约是 K 轴，滑窗天然流式、无跨位置 running 统计量；对照 `ALG-conv` R2「滑窗天然流式」及源码 §0.3（单遍 K 分块累加）**成立**，无遗漏已知候选（未出现 Welford/online-softmax 类且本算子无此类需求）。R1「无更优等价化简」——基线已是最小 FLOPs/访存形式，Winograd/FFT 对小核（`K_W≤10`）无 NPU 实测收益，与 `ALG-conv` R1 描述一致。
  - **复杂度复算**：whisper-conv1 FLOPs `2·1280·2998·80·3 ≈ 1.84 GFLOP` ✓；访存 `x 0.48MB + x_pad 写 0.48MB + x_pad 读 ≈0.48MB + weight 0.61MB + w_flat 写 0.61MB + w_flat 读 0.61MB + out 7.68MB ≈ 11.0MB` ✓；`11.0MB/0.62TB/s ≈ 17.7µs` ✓；发射项 `480·((64+128)·240)·2 ≈ 44.2MB ÷ 148GB/s ÷ 20 ≈ 14.9µs` ✓；**未漏计 `x_pad` 的 GM 往返**（相对函数 2 多 ≈1.0MB ≈ +9%，§1.6.0 已注明）。常数引用（`CONST-mte2-degradation`/`CONST-L1-port-bw`/`CONST-vector-launch-overhead`/`CONST-capacity-910B2C`）与 `constants.md` 条目一致。
  - **调研→设计一致性**：选定算法（基线 implicit GEMM）实际驱动 §1.4、§3.1（以优化后=基线公式为输入）、§6（K 分块 `T.serial` + Cube `T.gemm`）；迁移任务 §0.5/§0.6 处置与选族一致，源优化手段（tap-major / taps / SMEM / tensor-core / 流水 / fragment 累加 / 快路径 / coalesced）意图承接逐项说明（§0.6 末段），无静默丢弃。
- **弃选论证前提核对（亲自打开文档）**：§1.6.1 #1–#4 与 §1.6.3 候选 2/3/4/5/6 的代价前提逐条核对——核内 `T.transpose` 为**二轴交换链**、UB 级（`docs/Tilelang.language/创建操作/T.transpose.md` §2.3 + pattern-library `PL-1.1-transpose-chain` 实测 ~5.4µs），与设计「一次性 ~µs」一致；host permute 弃选依据 `ascend-constraints.md` §4「host 侧禁止改动输入 NPU 张量内的真实内容」**条款存在**；候选 3 引用 `TRAP-transpose-epilogue-poison` 存在；候选 2 的「`(c_i,k_w)` 序 B 为 band/Toeplitz 非矩形、需逐元素 gather」前提与卷积 im2col 结构一致。**无「把核内重排写成 GM 级 transpose」类矛盾，负向论断均有 `docs/`/pattern-library 佐证**（Winograd 的 NPU 无实测收益标注为「未文档化假设 + 估算依据」，合规）。
- **数学等价优化完整性 / 等价性独立推演**：§1.6.1 逐项评估，**采纳项为空**并写明「无优化空间」的充分依据（纯 im2col-GEMM+加法，无除法/超越函数/公共子表达式/稳定化需求；四候选均无正收益）。无采纳项故无等价性独立推演对象，`verify_equiv.py` 豁免合法（已在机械复核确认脚本不存在）。
- **向量化替代完整性（与 §3.3/§6 交叉核对）**：§1.6.2 覆盖 §3.3 伪代码与 §6 循环结构的全部计算点（12 项）；逐元素点（weight 重排、`x_pad` 前置、B 装载、bias 相加、输出 cast）已全部向量化（`T.transpose`/`T.copy`/`vbrc`/`vadd`/`vcast`，ssd/gqa 佐证）；保留 4 类标量（cid 解码 / 尾块尺寸 / cid 钳位 / 地址算术）理由充分（block/tile 级映射与动态边界，非逐元素热点）；矩阵乘落 Cube 为正确单元选择。
- **向量化轴与布局决策完整性**：§1.6.3 候选矩阵按 Cube/卷积类枚举 6 候选（implicit GEMM / `(c_i,k_w)` 序 / 转置 GEMM / host permute / per-channel 分组 / 非通道分块；direct 逐元素形态由 §1.6.0 R1 #7 与 §1.6.2 承载），逐候选量化评分；弃选均给量化理由（I/O layout 契约 ≠ 核内布局，明确）；选定 #1 的布局结论（A=`w_flat[K_W,C_out,C_in_pad]` 按 (tap,通道块) 取片、组内 tap 连续打包、**L1 K 轴偏移 `t·kcl`**）与 §3.3 伪代码 / §4.3 buffer 形状+§4.5 UB 预算 / §6 循环结构**四方一致**（布局决策与实现一致）。候选 #1 行内「L1 目标偏移 0」措辞宜按同节结论理解为通道块 K 基址 0（组内 tap 偏移另计 `t·kcl`），不构成决策层矛盾。
- **与下游章节一致**：§3.1 以优化后（=基线）公式为输入；§6 不出现 §1.6.2 未论证的逐元素标量循环；K extent 修复后 §6 与 §3.3 一致。

## Review Result
- stage: 2
- project: conv1d
- operator: _conv1d_unit_stride_kernel
- output: examples/conv1d/_conv1d_unit_stride_kernel/REVIEW.md
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
  - 建议完整: n/a（无阻塞项）
  - 无占位符: pass
- blocking_issues: 0
- migration_findings: 语义/pass 算法/pass 优化/pass 耦合性/pass 重设计/pass 一致性/pass golden独立性/pass
- algo_opt_findings: 调研完整性/pass 调研结论正确性/pass 调研→设计一致性/pass 优化完整性/pass 等价论证/pass（无采纳项） 向量化覆盖/pass 不可替代理由/pass 替代API佐证/pass 与§3§6一致性/pass
- suggestion_count: 0
- skills_consulted: .agents/skills/tilelang-design-review/SKILL.md; .agents/skills/_shared/standards/gate-and-retry.md; .opencode/agents/tilelang-design-reviewer.md; .agents/skills/tilelang-op-design/references/algorithm-candidates.md; docs/Tilelang.language/线性代数操作/T.gemm.md; docs/Tilelang.language/创建操作/T.transpose.md
- summary: v0 唯一阻塞项（`taps=2` 第二 tap 被单 tap 宽度的 gemm K 丢弃）已按「L1 K 轴按 tap 连续打包 `t·kcl` + `size=[tmc, taps*kcl, tnc]`」修复，对齐由 `kcl` 为 16 倍数独立推得、消除了 `block_k | c_in_pad` 附加前提，且在 §3.3/§6/§8.3 及全文逐处自洽；2 个建议项（`w_flat` 预零、UB 同时驻留口径）均已落实；调研结论与弃选论证前提经独立复核/文档核对通过，无采纳优化项故 verify_equiv 豁免合法——全部 9 维度 pass。
- issues: none
