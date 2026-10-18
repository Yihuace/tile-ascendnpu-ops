# 优化报告：Conv3dBiasFwdOp（Stage-4 optimize）

> **本文件版本 v1（2026-10-18 晚，第二轮 Stage-4）**：§0–§8 为第一轮（r3d-stem 1.99×）原文保留；
> 第二轮新增 **§9：unet 路径 6428 → 301.6 µs（21.3×）**、r3d-stem 追加到 **2588.7 µs（对 Stage-3 2.36×）**。
> 当前产物：`perf_opt/Conv3dBiasFwdOp.py`，sha256 `2e226a27…`（§9.6 给出最终表与 run_id）。

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `Conv3dBiasFwdOp`；单 kernel；dense NCDHW implicit GEMM（Cube Expert / MixCV bias 中继） |
| kernel（基准，只读） | `tileops/kernels/convolution/conv3d/conv3d_kernel/_conv3d_kernel.py` |
| 调优产物 | `.../conv3d_kernel/perf_opt/Conv3dBiasFwdOp.py`（R1：tap 打包 + 权重 L1 常驻，带**编译期形状级回退**；sha256 `1d50614b…`） |
| 口径 | `msprof op Task Duration(us)`（唯一 kernel 时延口径）；所有上机命令经 `.agents/tools/npu_lock.sh` 串行执行 |
| **工具链版本戳** | tilelang `0.1.2` dev root（`/home/developer/tilelang-mlir-ascend`）/ **CANN 9.0.0**（bishengir-compile 1.1.0 + hivmc 0.2.0）/ Ascend910B3（Chip 6，20 AIC）/ 2026-10-18 |
| 达标线 | 原报告以「**Task Duration 下降 ≥3%**」为 tune shape 达标判据（SOL 仅作参考；SOL 达标线原文未记录，队列表记 60%） |
| benchmark | `benchmarks/ops/bench_conv3d_bias.py::test_conv3d_bias_manifest_bench`（**3 tune + 2 smoke**） |
| 过程日志 | `.../perf_opt/opt_log.md`、`perf_records.jsonl`、`workload_inventory.json`、`profiles/`、`logs/`；实验分支 `perf_opt/candidates/r1_tap_pack_weight_resident.py`；探针 `perf_opt/probe_sliced_gemm.py` |
| 前置报告 | 本文件 v0 归档 `history_version/OPTIMIZE_REPORT_v0.md`（内容相同，仅结构未归一） |
| **当前产物（v1，第二轮）** | `perf_opt/Conv3dBiasFwdOp.py`，sha256 `2e226a27abaf…`：R3(NDHWC) ∥ R2(padded-N) ∥ R1 ∥ Stage-3 回退，四个**兄弟编译期分支**（详见 §9） |

## 1. 结论

- 一句话：病根是 **Cube 循环内逐 task 权重重载（48384 次）+ 27-tap im2col 输入重复读**；R1 把权重按
  co 块分核、在 task 循环前一次性载入 L1，使 `aic_mte2_time` 6050→3042µs，
  **r3d-stem-bias fp16 6161.50 → 3090.14 µs（1.99×，SOL 0.25% → ~0.51%）**。
- **达标数（双分母）**：全 shape **1/3**、可调 shape **1/3**（3 个 tune case 全部可调）、豁免 **0**
  （2 个 smoke 仅精度回归，不计入分母）。
  - `r3d-stem-bias`：1.99× ✅；
  - `unet-encoder-bias` bf16 6422.53 → 6449.49 µs（**+0.4%，噪声内**）→ **回退 baseline 等效实现**；
  - `video-down-bias` fp16 87.00 → 86.42 µs（**−0.7%，噪声内**）→ **回退 baseline**。
- 剩余 gap：implicit-GEMM 的 27-tap im2col 输入重复读 + 小通道 16-fractal 填充（`c_in=3→16` 是 5.3×
  FLOP/访存浪费），在当前工具链（无切片 gemm、L1→L1 拷贝与 MTE2 共享 L1 端口）下参数/tiling 级不可解；
  **不满足 `[DESIGN_LIMIT]` 触发条件（结构加速估计 <2x），故不产出 `perf_feedback.md`**。
- **是否建议采纳 wrapper**：✅ **建议采纳**——r3d-stem-bias 1.99× 为真实 Task Duration 下降、精度全通过；
  video-down / unet 走 baseline 等效回退（性能不劣于 baseline）；stride2-fp16 等其余 shape 经形状级回退后
  全量 17/17 通过。**wrapper 已切 perf_opt**（队列表）。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测 | 数值 | 结论 |
|---|---|---|
| `aic_mte2_ratio` | **0.99** | Cube 侧 99% 时间做 GM→L1 搬运，不是算力受限 |
| `aic_cube_ratio` | **0.016** | Cube MAC 几乎空闲 |
| `aic_scalar_mte2_stall_time` | **6061 µs** | 标量管线全程等 MTE2 完成 |
| `GM_to_L1_datas` | **72900 KB** | 每核逐 task 重载权重 + 27-tap im2col 输入 |
| 权重重载次数 | r3d-stem：1792 task × 27 tap = **48384 次** | 每次 gemm 前都从 GM 重载 A 块（权重） |
| 输入重复读 | `x_pad` 每 task 逐 tap 重读 27 次（kw 维 3 个相邻 tap 是同一行的 3 个平移视图） | 可共享但当前不可共享 |
| 与 Conv1d 的差异 | conv3d 的权重重排 / `x_pad` 在 host `_host_prep` 完成 | **病根不是 prelude 逐核冗余，而是 Cube 循环内的逐 task 重载**（优化手法不能照搬 Conv1d） |
| 其他 tune shape | video-down 87.00µs（stride-2 → 224 逻辑任务）；unet 6422.53µs（MTE2 99%） | video-down 已快；unet 与 r3d-stem 同症但参数不同 |

### 2.2 方法（做了什么、为什么这样做）

1. **tap 打包**：`for tap in serial(KV)` 折叠进 gemm 的 K 轴，每个 co 块只做一次 gemm，`K = KV*c_in_pad`
   （A/B 块沿 K 轴在 `tap*c_in_pad` 处连续打包）。
   - 为什么有效：把 27 次独立 gemm 合成 1 次，权重与输入的「每 tap 一次装载」变成「每 co 块一次装载」。
   - 正确性依据：K 轴扩展是**打包宽度的单源不变量**（同一 `c_in_pad` 决定 A/B 布局），不改变算术语义；
     L0/L1/L2/boundary 全 PASS（`max_abs ≤ 5e-1`，与基线一致）。
2. **权重 L1 常驻 + 固定 co 块分核**：`by = kernel_id % gy` 把每核固定到一个输出通道块
   （`gy <= num_kernels`），该块权重 `w_flat[:, by*block_m:..., :]` 在 task 循环**前**载入 L1 一次，
   被该核所有空间 task 复用。
   - 为什么有效：直接消除「逐 task 从 GM 重载 27 次权重」——r3d-stem 重载 48384→540 次，
     `aic_mte2_time` 6050→3042µs（2×）。
   - 正确性依据：每核只读自己那份常驻权重，无跨核同步；L0 全 PASS。
3. **B 块 N 维度 `n_max = ceil16(min(block_n, out_w))`**（否则 video-down `k_full=1728` 使
   `l1_a + l1_b = 663KB > 512KB` 触发 bishengir SIGSEGV）。
4. **编译期形状级回退（回归修复）**：`use_r1 = stride1 and n_kc==1 and gy<=num_kernels and c_in_pad==16
   and l1_total<=512KB`；不满足则走 baseline 等效实现（逐 tap gemm + 逐 task 权重重载），
   保证结果正确、性能不劣于 baseline。该分派是 **Python `if`（编译期）**，不是运行时形状分派。

**adopted vs fallback（原文表）**：

| shape | 路径 | 说明 |
|---|---|---|
| r3d-stem-bias（c_in=3→16, s=1） | **R1（保留）** | 1.99× 收益 |
| smoke-k3-s1-bias（c_in=8→16, s=1） | **R1（保留）** | 小通道常驻受益 |
| video-down-bias（c_in=64, s=2） | baseline 回退 | stride-2 多相，本就快 |
| unet-encoder-bias（c_in=32, s=1） | baseline 回退 | `c_in_pad=32` 常驻不生效 |
| stride2-fp16 / nondiv / k2-pad0 等 | baseline 回退 | 崩溃触发或 no_gain |

### 2.3 逐轮记录

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| R0 | baseline 诊断 | profile + `PipeUtilization.csv`（r3d-stem-bias/baseline） | 6161.50 / 87.00 / 6422.53 µs；`aic_mte2 0.99`、`cube 0.016`、stall 6061µs、72900 KB/核 | 定位「Cube 循环逐 task 权重重载」 |
| R1 | `r1_weight_resident` | tap 打包 + 权重 L1 常驻 + 固定 co 块分核 + `n_max` 分配 B 块 | r3d-stem 6161.50→**3096.64（−49.7%）**、unet 6422.53→6379.99（−0.7%）、video-down 87.00→90.28（+3.8%，噪声内）；3 项 L0 全 PASS | ✅ 采纳为 current best（unet/video-down 判 no_gain，后续回退） |
| R1 陷阱 | L1 超配 | 首版用 `block_n=128` 分配 B 块 | video-down `l1_a(221KB) + l1_b(442KB) = 663KB > 512KB` → bishengir **SIGSEGV**（非溢出报错） | ❌ 否决 `block_n` 分配；改 `n_max = ceil16(min(block_n, out_w))` |
| 归因 | unet 为何未生效 | ① unet-shape 以 fp16 重测；② `block_m=32`（l1_a 55KB）对照；③ 切片 gemm 探针 `T.gemm(l1_a, l1_b[0:K, 1:1+N], ...)` | ① fp16 6425.7→6384.9（同样无增益，**非 bf16 专属**）；② `block_m=32` 实测 **12756µs**（更差：gy=2 使每核 task 数翻倍、gemm M=32 效率下降）；③ 探针编译通过但**读数错位**（`max_abs=64`，gemm 忽略 N 偏移） | ❌ 判定为工具链限制（无切片 gemm、L1→L1 拷贝与 MTE2 共享端口），非参数可解 |
| R2 | 回归修复 | `use_r1` 编译期形状分派（R1 vs baseline 等效两套 `_cube`/`_bias_epilogue`/`prim_func` 并存） | `tests/ops/test_conv3d_bias.py::test_conv3d_bias[stride2-fp16]` 崩溃消除；全量 **17/17 passed**；`--level all` 全绿 | ✅ 采纳（形状级回退） |
| 收束 | final 复测 | 3 个 tune workload 同一候选 | 6161.50→3090.14；87.00→86.42；6422.53→6449.49（见 §3） | ✅ 收束 |

**回归修复记录（原文）**：wrapper 端到端验收发现 `test_conv3d_bias[stride2-fp16]` 崩溃
（`std::runtime_error` → Aborted，rc=134）：单跑通过，但全量套件（前序 6 kernel 同进程）必现，
崩溃点在设备端 `launch_npu`。A/B 确认 baseline 同 case `1 passed` ⇒ **系 R1 改动引入**；修法即上述
形状级回退（详见 `opt_log.md §5`）。

## 3. 优化结果（final，3 workload 同一候选 `r1_weight_resident`）

| workload | base µs | final µs | Δ | base SOL% | final SOL% | 是否达标 | 路径 |
|---|---:|---:|---:|---:|---:|---:|---|
| r3d-stem-bias-float16 | 6161.50 | 3090.14 | **−49.8%** | 0.25 | ~0.51 | ✅ | R1 |
| video-down-bias-float16 | 87.00 | 86.42 | −0.7% | 原文未记录 | 原文未记录 | ❌ | baseline 回退 |
| unet-encoder-bias-bfloat16 | 6422.53 | 6449.49 | +0.4% | 原文未记录 | 原文未记录 | ❌ | baseline 回退 |

> SOL = roofline 理论时长（`max(flops/峰值算力, bytes/峰值带宽)`）/ 实测 Task Duration；conv3d 小通道
> workload 的 implicit-GEMM 结构天然远离 roofline（16-fractal 通道填充 + 27-tap im2col 重复读）。
> 测量为单次 `msprof op`；原文未记录交错复测次数与噪声带数值（no_gain 项以「噪声内」定性）。

## 4. 精度

- 全量 `python -m pytest tests/ops/test_conv3d_bias.py -q` → **17 passed**。
- `python perf_opt/Conv3dBiasFwdOp.py --level all` → **All check passed!**：L0 **5 例**、L1 **10 例**、
  L2 **6 例**、boundary **3 例** 全 PASS（**`max_abs ≤ 5e-1`，与基线一致**）；2 个 `smoke-k3-s1-bias`
  fp16/bf16 仅精度回归 PASS。
- `tileops-report --op Conv3dBiasFwdOp --prof-mode msprof` → **status=passed**（17/17，r3d-stem 3090.14 µs）。
- 关键边界用例（NaN/±inf/−0/subnormal）逐项：原文未记录（仅记 boundary 3 例 PASS、`max_abs ≤ 5e-1`）。

## 5. 时间与成本

| 项 | 值 |
|---|---|
| 首轮调优（+08:00） | 2026-10-18 09:09 → 09:40 |
| 回归修复（+08:00） | 2026-10-18 09:45 → 10:15 |
| 墙钟时长 | 原文未记录（仅分段窗口） |
| token input/output/reasoning | 原文未记录 |
| cost | 原文未记录 |

## 6. 产物

| 类别 | 路径 / 状态 |
|---|---|
| final kernel | `.../conv3d_kernel/perf_opt/Conv3dBiasFwdOp.py`（`r1_weight_resident` + 形状级回退，sha256 `1d50614b…`） |
| 调优记录 | `perf_opt/opt_log.md`、`perf_records.jsonl`（append-only）、`workload_inventory.json`、`profiles/`、`logs/` |
| 实验分支 | `perf_opt/candidates/r1_tap_pack_weight_resident.py` |
| 探针 | `perf_opt/probe_sliced_gemm.py`（切片 gemm 读数错位，`max_abs=64`） |
| 测量脚本 | `perf_opt/bench_kernel.py`、`perf_opt/measure.py`、`perf_opt/quick_l0.py`、`perf_opt/env.sh` |
| gate 4 | **pass**（队列表记录） |
| 算子状态 | `phase=DONE`（队列表 #10：`it=1`，`last_improvement 49.9%`） |
| wrapper 切换 | **已切 perf_opt**（`conv3d.py` 已 import `perf_opt.Conv3dBiasFwdOp`）；队列表记「端到端 report 复核中」 |
| 端到端 report | `tileops-report --op Conv3dBiasFwdOp --prof-mode msprof` → status=passed（17/17） |
| pattern-library 回写 | 候选价值点（原文 Skill Retrospective，4 条 D/P）；实际写入状态原文未记录 |

## 7. 天花板 / 豁免 / 缺口

- **豁免项：无**（3 个 tune workload 均参与判定；2 个 smoke 仅精度回归）。
- **ceiling 证据**：
  - `aic_mte2_ratio = 0.99`、`aic_cube_ratio = 0.016`、`aic_scalar_mte2_stall_time = 6061µs`（baseline）；
    R1 后 MTE2 6050→3042µs，但 unet 的 `aic_scalar_mte2_stall_time` 6289→6227µs **几乎不变**
    （`read_main_memory_datas` 32136→21120 KB，HBM 层重载已除，编译器仍逐 task 从 L2 重载 110KB 常驻权重）。
  - unet 对照实验：fp16 重测同样无增益（6425.7→6384.9）；`block_m=32` 实测 12756µs（更差）；
    切片 gemm 探针读数错位（`max_abs=64`）⇒ 无 API 级替代方案。
- **贴线 / 未达标项**：`video-down-bias` −0.7%、`unet-encoder-bias` +0.4% 均在噪声内，判 no_gain
  并回退 baseline（不是「差一点到线」，而是无收益）。
- **`[DESIGN_LIMIT]` 判定**：**不触发**（结构加速估计 <2x 且非设计层归因）⇒ **不产出 `perf_feedback.md`**。
- **能力缺口**：L1 超配以 **SIGSEGV** 形式失败（CG-2026-0003 家族同族，原文记）；切片 gemm 忽略 N 偏移
  （原文记为 D 类实测价值点，未编 CG 号）。

## 8. 建议

1. **采纳 `perf_opt/Conv3dBiasFwdOp.py`**（已切 wrapper）：r3d-stem-bias 1.99× 为真实下降，精度全通过；
   video-down / unet 走 baseline 等效回退，性能不劣于 baseline。
2. **保留形状级回退**：`use_r1` 条件是回归修复的核心（`stride==1 and c_in_pad==16 and L1≤512KB`）；
   任何放宽都必须重跑全量 `tests/ops/test_conv3d_bias.py`（单跑通过不代表全量套件通过）。
3. 工具链能力补齐后重验剩余 gap：27-tap im2col 输入重复读需要可用的**切片 gemm**（当前读数错位）
   或 L1→L1 拷贝不占 MTE2 端口；小通道 16-fractal 填充需要更小的通道 fractal。
4. 若升级 CANN（>9.0.0），需复核两件事：`c_in_pad=32` 的权重常驻是否开始生效（unet 可能转为有收益）、
   L1 超配是否仍以 SIGSEGV 失败（影响 `n_max` 绕法是否可简化）。
5. 未解决项：unet-encoder-bias 的常驻失效属编译器对该 shape 的 L1 生命周期行为，参数/tiling 级不可解
   （原文已用 fp16 对照、`block_m=32` 对照、切片 gemm 探针三条证据排除）。


---

# 9. 第二轮优化（2026-10-18 晚）：unet 路径 21.3×

> 目标：unet-encoder（`c_in_pad=32`）此前完全走 Stage-3 回退，未获任何优化。本轮不改 `DESIGN.md` /
> `_conv3d_kernel.py` / wrapper，产物全在 `perf_opt/`。口径仍为 `msprof op Task Duration(us)`。

## 9.1 本轮起点（同工具链复测，分两候选）

| workload | Stage-3 基线 µs | perf_opt(R1) µs | 起点判定 |
|---|---:|---:|---|
| r3d-stem-bias-fp16 | 6116.38 | 3095.80 | R1 已 1.98× |
| video-down-bias-fp16 | 86.48 | 87.10 | 已快 |
| unet-encoder-bias-bf16 | 6428.05 | 6428.57 | **与 Stage-3 相同**（`c_in_pad==16` 闸门回退）|

## 9.2 病根更正（推翻「权重常驻没写进去」的旧解释）

第一轮把 unet 无收益归因为「编译器对 110 KB 常驻权重仍逐 task 从 L2 重载」。本轮用三条实测更正：

1. **IR 层**：强制 R1（放开 `c_in_pad==16`）dump npuir IR，权重 27 次 `nd2nz` 位于 task 循环**外**，
   `l1_a` 的 `memref.alloc` 也在循环外 ⇒ 常驻**已经写进 IR**。
2. **copy 拆分探针**（把一侧 copy 缩到 16×16，只测时间）：A 全量/B 缩小 → 2749 µs；
   B 全量/A 缩小 → 7160 µs；原版 6428 µs ⇒ 两侧各占约一半且部分重叠，**单改一侧最多 2.3×**。
3. **机理**：常驻把 `l1_a`(110 KB)+`l1_b`(110 KB) 顶到 220 KB，**丧失多缓冲/流水重叠**，
   逐 tap 串行化，抵消了「少一半 copy」的收益；真正的病根是 **im2col 输入 copy 的 DMA 段数**
   （`aic_mte2_instructions = 91 774`，每次 copy 的源是 `c_in_pad` 个 128 B 小段、跨 359 KB 平面；
   `GM_to_L1_bw_usage_rate` 仅 **5.63%**，即额定带宽的 1/18）。

## 9.3 结构改造（两轮）

| 轮次 | 改动（单一主要优化点） | unet µs | 相对起点 | r3d-stem µs | 关键 profile |
|---|---|---:|---:|---:|---|
| R2 | **padded-N 多行分块**：N 轴 = R 条输出行 × `w_pad` padded 坐标（`n_tile=R*w_pad`），一条连续 span 覆盖 R 行且自带 kw 平移；kw 打包进 K；尾部行组重叠；逐行 fixpipe 丢弃 padding 列 | **1534.21** | −76.1%（4.19×）| 2609.39（**−15.7%**）| MTE2 指令 91 774 → 16 039；bw 5.63% → 27.98% |
| R3 | **+ channel-last（NDHWC）暂存**：通道改最内轴，每次 tap copy 变成**单条连续段**，GEMM 用 `b_transpose=True` 读 `[N,K]` | **306.41** | −80.0%（再 5.0×）| 4961.30（**回退，见 9.4**）| MTE2 指令 → 973；cube 0.018 → 0.248 |
| 合并 | R3 闸门加 `c_in_pad>=32`，`c_in_pad<32` 走 R2（兄弟编译期分支） | 315.27 | — | 2630.27 | 全 tune + smoke 非回退 |
| 参数 | `r3_rows` 4/5/6、`r2_rows` 2/3/4 扫描 | 309.8/303.7/**302.5** | 默认即最优 | 3547.7/**2641.4**/3065.1 | 上界 = L0C 容量 ⇒ `config_no_gain` |

**新增工具链实测（两个探针，均已落地为可复现脚本）**：

- `probes/probe_cube_caps.py` P1：`T.copy(l0_c[0:M, off:off+N], gm)` 的 **L0C 列偏移被正确执行**
  （`max_abs=7.6e-6`，off=32）。这正是 R2/R3 逐行丢弃 padding 列的前提；与
  `TRAP-cube-gemm-sliced-operand`（gemm 操作数偏移被忽略）互补 —— 结论绑定 CANN 9.0.0。
- 同脚本 P2：**Expert+L1 的 `b_transpose=True` 可用且数值正确**（`max_abs=9.5e-6`）。
  CG-2026-0026 记录的 `b_transpose` 崩溃是 **Developer 模式 + shared buffer** 形态，不适用于本形态。
- **陷阱**：`@T.prim_func` 若比同级分支多嵌一层 Python `if`，tilelang 的 source line-map 错位，
  报 `InternalError: requested line: N at index: N-1 ... line_map size`（真实报错被渲染器吞掉）。
  绕法：所有 prim_func 分支放同一层 `if`。

## 9.4 r3d-stem 的反例（为什么 R3 要按 `c_in_pad` 分闸）

NDHWC 下 `c_in_pad=16` 的内层连续段只有 **32 B**：`aic_mte2_instructions` 降到 1675 条，但
`aic_mte2_time` 4712 µs = **2.8 µs/条**（unet 为 0.19 µs/条）⇒ 比「16 段 × 256 B 跨步」更差，
实测 2609 → 4961 µs（+90%）。因此 R3 闸门收紧为 `c_in_pad >= 32`，`c_in_pad==16` 保留 R2。
**「通道最内」不是普适优化，段宽 < 64 B 时反而退化**（新 D 类实测）。

## 9.5 精度与回归

- `python perf_opt/Conv3dBiasFwdOp.py --level all` → **All check passed**：L0 5 例、L1 10 例、
  L2 6 例（6/6）、boundary 3 例（3/3）全 PASS；`max_abs` 与基线一致（unet bf16 5e-1 级）。
- `python -m pytest tests/ops/test_conv3d_bias.py -q`（经 `tileops-report` correctness 阶段）→
  **17/17 通过，0 failed，无 skip**（不再需要 CG-2026-0003 豁免）。
- `tileops-report --op Conv3dBiasFwdOp --prof-mode msprof` → **status=passed**，
  run_id `20261018_185919_687302_Conv3dBiasFwdOp`。

## 9.6 最终结果（同一最终候选 `r3_merged`，sha256 `2e226a27…`）

| workload | Stage-3 µs | 起点 µs | final µs | 对 Stage-3 | 对起点 | ratio（report）| 路径 |
|---|---:|---:|---:|---:|---:|---:|---|
| r3d-stem-bias-fp16 | 6116.38 | 3095.80 | **2588.67** | −57.7%（**2.36×**）| −16.4% | 1.68% | R2 |
| video-down-bias-fp16 | 86.48 | 87.10 | 86.70 | −0.2%（噪声内）| −0.5% | 6.26% | base（stride-2）|
| unet-encoder-bias-bf16 | 6428.05 | 6428.57 | **301.59** | **−95.3%（21.3×）** | −95.3% | **20.43%** | R3 |
| smoke-k3-s1-bias（仅精度）| 174.76 | — | 74.22 / 74.64 | −57.5% | — | 不计入 | R2 |

- 报告 ratio 区间：**0.76–6.29% → 1.68–20.43%**（unet 0.76% → **20.43%**）。
- 采纳判定：**建议采纳**（三 workload 均非回退；correctness 17/17）。
- 中止原因：**plateau/budget** —— 结构方向两轮已落地、参数方向探到 L0C 上界（`config_no_gain`）。

## 9.7 剩余 gap 与 ceiling 证据

unet（301.6 µs）已从「MTE2 单侧独大」转为**三管道均衡**：
`aic_mte2_ratio 0.621` / `aic_mte1_ratio 0.338`（L1→L0 操作数端口）/ `aic_cube_ratio 0.256` /
`aic_fixpipe_ratio 0.107`。剩余成本来自 implicit-GEMM 的**固有 3× kh 重复读**
（同一输入行被 kh=0/1/2 各读一次）：继续消除需要「按 kh 预移位的主机侧布局把 K 扩到 `kh*c_in`」
（主机侧 3× 流量）或「gemm 支持操作数平移视图」（被 TRAP-cube-gemm-sliced-operand 阻断）。
据此判定为**参数/tiling 之外的结构收益递减**，不满足 `[DESIGN_LIMIT]` 触发条件
（结构加速估计 < 2×、且归因不在 DESIGN 假设），**不产出 `perf_feedback.md`**。
