# Elementwise 多输入搬运效率与带宽曲线

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。条目带 front-matter（schema 见 INDEX.md §3）；`repro: repro-missing` 表示待回填最小复现代码。

---
id: PL-1.6-copy-floor
kind: constant
family: [elementwise]
apis: [T.copy, T.vadd]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z
toolchain: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0
repro: repro-missing
---

### 1.6 copy-floor 标定与 MTE2 带宽退化曲线（elementwise 多输入）✅ 已验证

- **copy-only 探针标定法**：与目标 kernel 同结构（同 persistent/分块/映射）的 3 载入 + 2 vadd 保活（DCE-proof）+ 1 写出探针，一步判定计算链暴露量。lerp_tensor（3 输入 1 输出，7-pass fp32-transit 链）：≥16M 全 dtype delta ≤1.1us（16M 仅 0.34us）——**auto multi-buffer 将全部向量 pass 隐藏于 MTE2 窗口内**，计算链优化在该区间 ROI≈0，直接进入搬运效率维度。
- **MTE2 有效带宽随总流量退化**（4-set 轮换，fp16 bs4096，48 vector core）：1M 33.9 → 16M 30.2 → 64M 20.9 → 256M 18.4 GB/s/core；256M 的 3:1 R/W 混合全流量 ~1.26 TB/s 为该设备地板——**"~1.8 TB/s 峰值参考"对混合读写流量不可达**（不要据其估算大 N headroom）。1M 档读带宽含 L2 成分。
- **访问模式**：grid-stride（48 核聚集于 48×tile 移动窗口）实测优于每核连续分块 +3.2%（256M）——HBM 混合流下聚集窗口访问更优，"连续流 DRAM 局部性"直觉反向（详见 bottleneck-patterns BP 候选）。
- **标量削减判定式**：mte2_ratio ≥0.95（搬运饱和）时 per-tile 标量/分支开销已被隐藏——静态尾块（去 T.min）全档 tie/-0.6%，guard-free 预判 <0.3%；此类优化仅在 mte2_ratio <0.9 的流水浅区（如 1M 档 0.70–0.84）可能有收益。
- 溯源：lerp_tensor optimize 任务（2026-09-07），`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md`；tilelang 0.1.2+ed787bb / Ascend910B2C / CANN 8.5.0。

---
id: PL-1.10-loads-first-decoupling
kind: pattern
family: [elementwise, reduction, norm]
apis: [T.copy, T.vcast, T.serial, T.alloc_shared]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z（2026-09-10 蒸馏溯源归位：原回写误标 145312Z，实际 optimize task_id 以 stage_state.json 为准）
toolchain: tilelang 0.1.2+a83118285a（2026-09-10 build）/ Ascend910B2C / CANN 8.5.0
repro: repro/PL-1.10-loads-first-decoupling.py
---

### 1.10 单 staging 复用链的 MTE2/VEC 零重叠 → per-input staging + 前置装载 ✅ 已验证

**现象信号**：多输入 kernel 的 msprof PipeUtilization 中**各 pipe 忙时之和 ≈ Task Duration**（如 prefill fp16 bm=1：vec 38.8 + scalar 13.2 + mte2 42.4 + mte3 6.5 ≈ 101µs vs wall 90.5µs）——MTE2 与 VEC 交替串行而非重叠。**根因**：单一 staging 缓冲被 x→scale→shift 顺序复用，数据流强制 MTE2(x)→VEC(vcast)→MTE2(scale)→VEC→MTE2(shift)，每次 GM 装载都等前一个 v-op 释放缓冲。

**优化（慢→快关键更改，delta 形态见 repro）**：① 每个输入一张专用 staging 缓冲；② 三输入 `T.copy` 全部前置到 persistent serial body 顶部（纯 MTE2 相）；③ 向量链不间断地跑完；④ 输出 staging 复用 x-in staging（活区间不相交：x-in 死于首个 vcast，out 始于末个 vcast）——不新增缓冲。auto-multi-buffer pass 会物化第 2 实例使跨迭代重叠合法（UB 预算按 TRAP-UB-multibuffer-inflation 平台值：transit ≈20 B/elem、fp32 ≈26 B/elem，**不是** resident × 系数）。

**实测**（ada_layer_norm，2048×4096 fp16，bm=2，msprof op median-of-20）：58.27 → 39.81µs（**-31.7%**）；pipe 忙时和 100.9→68.9µs vs wall 39.8（~29µs 跨 pipe 重叠）；kernel 转为 vector-bound（vec_ratio 0.914）。次级收益：dit-xl-2（1024×1152 fp16）-12.5%、decode（1×4096 bf16）-16.1%（少 2 个 mid-chain MTE2→V 同步点）。**适用条件**：persistent serial（num_local_tasks≥2，跨迭代重叠有深度）收益最大；num_local=1 时退化为 tie~小幅回退（smoke 64×1152 A/B tie——intra-iteration 无重叠可赢，实测勿盲改）。同族先例：PL-1.6 lerp 3 载入前置 + 向量链全隐藏（≥16M 档）。

**注意**：bm 选择与该结构正交但交互——bm 上限受 multi-buffer 平台 footprint 约束（见上）；且最优 bm 非单调（ada dit 1024×1152：bm 曲线 {5:9.75, 6:9.66, 7:8.76, 8:9.55}µs，bm=7 局部最优；同 N 下 4096×1152 则 bm=8 胜——**按 shape 实测扫描，勿从单点泛化 UB 松弛律**）。

- 溯源：ada_layer_norm Stage 4（2026-09-10），`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/opt_log.md`（Iteration 2 机制确认 + A/B 裁决记录）。

---
id: PL-1.14-mte3-strided-ws
kind: pattern
family: [elementwise, mixcv, mte]
apis: [T.copy]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（Stage 4；2026-09-17 蒸馏 D2 溯源归位——原回写误标 20260917T0855Z，实际 task_id 以 .stage_state.json 为准）
toolchain: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17；2026-09-20 4515de8 重验存活（task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 重跑复刻 ws 块连续布局，全量门禁绿）
repro: repro/PL-1.13-aiv-dup-subid-split.py
---

### 跨引擎 ws 中继的 MTE3 布局铁律：块连续 ≫ band 化

- **形态**：Vector→Cube 因子中继（GM ws）的写侧布局两个候选——①块连续（每 [64,64] 块占连续 8KB，ws 形如 [..., s_blk, bl, bs]）；②band 化（每 l-tile 的因果带 [bl, l0+bs] 连续，行 stride = Q×2B=512B，换取 Cube 侧单条 band 读）。
- **实测（ssd_chunk_scan w2，msprof op）**：band 化使 **AIV MTE3 时间 3×**（106→220µs，ratio 0.17→0.35）——512B 行距的跨步写吞掉 Cube 侧全部搬运合并收益（w2 净 +2.3% 回退）。**写侧块连续 + 读侧逐块连续读入 L1 band 列偏移区**（`l1[0:bl, s0:s0+ts]`，**ts 必须尾块裁剪**——见 traps-runtime.md TRAP-L1-band-dst-tail-overrun）是两全形态：Vector 写快、Cube 仍可单 gemm（v5_xband，w2 −1.8%/w4 −4.3%）。
- 机制：MTE3 跨步写每行 128B 段 + 512B 行距 → 段级延迟不摊销（CONST-mte2-degradation 的指令/段维度）；块连续 8KB 单段流。
- 溯源：`examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/perf_opt/`（opt_log R1 v2_band 行 + R2 v5_xband 行；profiles/round1|round2 对比）。

---
id: PL-1.17-subfp32-fp32-transit
kind: pattern
family: [elementwise, vector, norm, mamba]
apis: [T.vcast, T.vadd, T.vsub, T.vmul, T.vexp, T.copy]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z（首证，VP-2026-0002）/ ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（第二证：bf16 触发编译实证 + 因子链形态）
toolchain: 首证 tilelang-mlir-dev dev root build（2026-09-07）/ 第二证 tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C（2026-09-17）
repro: repro-missing
---

### sub-fp32 逐元素算子 fp32 中转模式（bf16 支持缺失 + fp16 golden 对齐双触发）✅ 两任务实证

- **双触发条件**：① 契约 dtype 含 bf16——v-prefix 算术（vadd/vsub/vmul/vexp）dtype 矩阵不含 bf16（docs/Tilelang.language/数学操作/ 各 §2.2.1），vcast 升 fp32 是唯一计算路径（ssd 第二证：c_scaled 链 fp16 化变体 v7 在 bf16 workload **编译失败** rollback——`T.vmul` bf16 × 的运行时实证）；② fp16 需对齐 torch CPU golden——golden 经 fp32 opmath + 单次舍回，原生域逐步舍入差 2–3 ulp（TRAP-fp16-opmath-golden）。
- **链结构**：GM→UB(同 dtype) → `vcast(round_mode="rint")`→fp32 → fp32 域 v-prefix 原地链 → `vcast(rint)`→原 dtype → UB→GM（f16/bf16→f32 仅 rint，T.vcast.md §2.2.1）。ssd 因子链形态：cb/C/dt(dtype)→f32 因子域运算、末端回写（Stage 3 L0–Boundary 全过实证）。
- **UB 预算**：中转路径按 Σ(elem_bytes × buffer_count) 复算（bf16 中转 24B/elem、fp16 中转 20B/elem——VP-2026-0009 混合字节口径）。
- **实测**（lerp 首证）：max_diff ≤1 ulp fp16（抽样 binade 4.883e-04～1.953e-03）、全量 0 violations；NaN/Inf 角点 IEEE 传播与舍入路径无关（中转后 inf-corner 逐位不变）。
- repro：repro-missing（知识域最小 repro 待同族任务回填；两证任务内复现命令见 queue VP-2026-0002 证据链——provenance 允许失效）。

---
id: PL-1.19-scatter-col-write-amplification
kind: pattern
family: [reduction, elementwise]
apis: [T.copy]
dtype: [fp16, fp32]
device: 910B2C
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z
toolchain: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C
repro: repro/repro_strided_colblock_read.py
---

### (M,1) 列切片 UB→GM 写 42× 放大 → 扁平 chunk-major 连续写

慢：`T.copy(ext_out[0:M,0:1], ws[0:M, c:c+1])`（(M,1) 目标列 × 行 stride）——每散布元素拉 128B 行：tn=256/nchunk=400 实测 UB_to_GM 400KB（真实 9.6KB，42×），mte3_ratio 0.78。快：workspace 取扁平 (nchunk*M,) chunk-major，每 chunk 一次 1D 连续写 `T.copy(ext_out[0:M], ws_flat[c*M:(c+1)*M])`（同 kernel mte3 降至 0.03–0.18）；消费端按 flat-view 重解释形状（narrow-path 先例）。写侧块连续律与 PL-1.14（跨引擎 ws）同族，此处为纯 Vector UB→GM。

---
id: PL-1.20-nsplit-perrow-partial
kind: pattern
family: [reduction]
apis: [T.Kernel, T.Scope, T.alloc_ub, T.block_barrier, T.reduce_max, T.reduce_min, T.vbrc]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: argmax-_argreduce_kernel-20260924T031005Z
toolchain: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C
repro: repro/repro_nsplit_pattern.py
---

### C6 N-split：单 launch 的 per-row partial + 核内 merge

适用：小 M、大 N，单 kernel 的 bm=1 路径无法占满 V-core（lm-head 类 `(4,102400)`）。

编程模式：Expert。该形态在 `T.Scope("Vector")` 中使用 `T.alloc_ub`，
N-split fused 分支须以 `TILELANG_ASCEND_MODE=expert` 编译；其他不依赖
该结构的路径可保持其原有编程模式。

硬约束：
- partial、workspace 写入、跨 block 同步、最终 merge 在同一个
  `@T.prim_func` / 一次 device launch 中；
- `num_blocks <= aicore_num`，保证 `T.block_barrier` 单波完成；
- 使用 `(cid, subid)` 展开为 `vid = cid * 2 + subid`，避免同一 AI Core
  的两个 V-core 重复计算同一 chunk。

结构：
- 各 V-core 对其 chunk 执行 **`for r in range(M)` 展开的 `(1, tn)` 行链**；
  连续读取输入，并连续写入二维 `ws_val[nchunk, M]` / `ws_idx[nchunk, M]`。
  该形态避开 `(M, tn)` 2D fused-ite 的标量化陷阱
  `TRAP-parallel2d-bm-ge2-scalarize`；
- `T.block_barrier(0)` 在所有 producer 写完 workspace 后汇合；
- `vid == 0` 读取二维 workspace，沿 `dim=0` 规约 value，并以全局化 idx
  做首现 tie-break；
- merge 不需要 `T.transpose`，workspace 不需要重解释为转置布局。

`tn` 与 `num_blocks` 是该结构的调优参数：默认应让 `nchunk` 落在单波、且
V-core 利用率合理的区间；需针对目标形状和当前工具链以完整单 launch 路径测量。

---
id: PL-1.21-overlap-saturation-signal
kind: pattern
family: [gemv, reduction, elementwise]
apis: [T.copy, T.serial]
dtype: [fp16, fp32]
device: 910B3
status: verified
origin_task: TileOPs-_engram_decode_kernel-20261030T121905Z（首证，queue VP-2026-0145）/ TileOPs-_engram_decode_kernel-20261030T145026Z（第二证 round 3，含分界修正）
toolchain: 首证 tilelang 0.1.2+76e4550（commit 013dbbf）/ CANN 9.0.0 / Ascend910B3；第二证 tilelang 0.1.2+01805d7a / CANN 9.0.0 / Ascend910B3
repro: repro-missing
---

### 已重叠判据：`mte2_time + vec_time > Task Duration` 时降 MTE2 是无效方向（含「放大 tile vs 破坏多缓冲」分界）

**触发信号（首证，910B3 / engram_decode b4）**：`aiv_mte2_time + aiv_vec_time = 51.4 + 54.8 = 106.2 µs ≫ Task Duration 73.6 µs`——两 pipe 已在重叠，单独降低任一条 pipe 的搬运量都不改变 Task Duration。

**首证实测**（2026-10-30，tilelang 76e4550）：slab 把 `mte2_time` 55.2→31.9 µs、`mte2_active_bw` 36→63 GB/s，Task Duration 反而 78.402→107.162 µs（手动展开变体 97.502 µs）。原始归因为「放大暂存块（2×(16,d)→2×(32,d)）挤掉 `--enable-auto-multi-buffer` 的双缓冲」。

**第二证（2026-10-30 round 3，tilelang 01805d7a）给出更精确的分界**——「放大 tile」本身不是禁忌，破坏自动多缓冲的是**循环体指令排列/第二段暂存的引入**：

| 结构改动 | `mte2_time`（b4，µs） | Task Duration | 判读 |
|---|---:|---:|---|
| 基线 BK=16（b1 96 条 DMA / b4 192 条） | 51.4 | 73.59 | 基准 |
| `BK=16→32`（仅编译期 tile 常量，b1 48 条 DMA） | 14.36（b1；51.4→51.8 b4） | 26.75（b1，−5.7%）/ 73.53（b4） | **正收益**：tile 放大但排列不变 |
| `e_n` 改从常驻 `e_local` 做 UB→UB 切片（去掉 nk 条 GM DMA） | 29.5 | 102.18（+38.8%，`mte2_wait` 0.50） | 回退：流水被拆 |
| K/V 各自暂存（解 WAR） | 28.7 | 76.42（+3.8%，mte2 −45%） | 回退 |
| `T.Pipelined(num_stages=2)` | 51.8 | 73.62（±0） | 编译器已等效 |

**判据（三条，按顺序）**：① `aiv_mte2_wait_ratio < 0.05`（或 `mte2_time + vec_time > duration`）⟹ 搬运已被隐藏，凡是「只降 mte2」的改动预期收益 ≈0，须同时降 `aiv_scalar_time` 才可能动 duration；② 只改**编译期 tiling 常量**（BK/stage 元素数）到 UB 预算允许的上限，自动多缓冲不受影响——同族第二证为 +6%；③ 一旦引入**第二段暂存 / 显式分块复用 / 原地别名**，`mte2_wait` 从 0.02 跳到 0.44–0.83、duration 反升 4–39%——这类结构的验收指标是 `mte2_wait` 是否仍 <0.05，而不是 mte2 busy time 是否下降。

**关键代码更改（慢→快方向，delta 骨架）**：
```
# 慢：为「减少 DMA 条数」引入显式 slab + 内层串行复用（或 UB→UB 切片换掉 GM 读）
for ks in T.serial(2):
    T.copy(W[k0*BK + ks*BK//2 : k0*BK + (ks+1)*BK//2, :], w_slab[ks])   # 第二段暂存 → 自动多缓冲预算不足
# 快：把「每条 DMA 覆盖多少元素」交给编译期 tile 常量，循环体形状不变
BK = TUNED_STAGE_ELEMS // d_padded        # 8192 // 256 = 32（d=512 时为 16，受共享 UB 预算限）
for k0 in T.serial(d_mem // BK):
    T.copy(W[k0*BK:(k0+1)*BK, :], w_f); T.copy(e[k0*BK:(k0+1)*BK], e_n)   # 与基线同形
```
**复现条件（待回填，ED-B delta 形态）**：效应只在完整 kernel 规模显现——同一 kernel 仅差 staging 结构与循环体排列的两个分支，用 `msprof op --kernel-name=main --launch-count=20` 取 `aiv_mte2_time` / `aiv_vec_time` / `aiv_scalar_time` / Task Duration 四元组对照即可判定本条（慢形态判据：`mte2_wait` 抬升且 duration 上升；快形态判据：`mte2_wait` 保持 <0.05 且 duration 不升）。

机制归因：`CONST-mte2-degradation`（910B3 单核每 DMA 指令固定开销 0.16 µs @76e4550 → 0.20–0.21 µs @01805d7a）与 `CONST-ub-budget-vs-pattern-cap`（staging tile 与尾部工作集共享 192 KiB，可用上限 = `8192 // d_padded`）；多缓冲失效的一手信号是 `aiv_mte2_wait_ratio` 抬升（见 `TRAP-UB-multibuffer-inflation`）。
