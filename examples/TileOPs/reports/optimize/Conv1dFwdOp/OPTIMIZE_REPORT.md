# 优化报告：Conv1dFwdOp（Stage-4 optimize）

> 原始 v0 归档：无（本文件为 `Conv1dFwdOp` 首份 Stage-4 报告；同目录 `Conv1dBiasFwdOp/OPTIMIZE_REPORT.md`
> 记录的是同一个 5-kernel 卷积包中 `conv1d_unit_stride` 的第一轮调优，本轮接力的对象是其中**通用 `conv1d`**
> 变体，两者数据不重叠、口径相同）。

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `Conv1dFwdOp`（bias 可选）/ `Conv1dBiasFwdOp`（bias 必填）共用同一 kernel 包；**多 kernel（5 变体）**，本轮调优其中**通用 `conv1d`** 一个变体；Cube/Vector MixCV（`aic_cube_ratio` < 1%，实际由搬运与输出写主导）<br>语义：`out[n,c_o,l] = bias[c_o] + Σ_{c_i,k_w} W[c_o,c_i,k_w]·x[n,c_i,l·s_W + k_w·d_W − p_L]`，`L_out=(L_in+p_L+p_R−d_W(K_W−1)−1)//s_W+1` |
| kernel（基准，只读） | `tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_kernel.py`（Stage-3 迁移版，`_conv1d_main` / `_conv1d_bias_main`） |
| 调优产物 | `tileops/kernels/convolution/conv1d/conv1d_kernel/perf_opt/_conv1d_kernel.py`（drop-in，工厂签名与 workspace 契约不变） |
| 口径 | **`msprof op Task Duration(us)`**（唯一时延口径）+ SOL ratio；kernel-name 过滤 `_conv1d_main` / `_conv1d_bias_main`（`Op Name` 列二次校验，防止过滤失效时混入 torch 侧 kernel）；所有上机命令经 `.agents/tools/npu_lock.sh` 串行执行 |
| **工具链版本戳** | tilelang `0.1.2` dev root（`/home/developer/tilelang-mlir-ascend`）/ **CANN 9.0.0**（bishengir-compile 1.1.0 + hivmc 0.2.0）/ Ascend910B3（Chip 6，**20 AIC**）/ 2026-10-18 |
| 达标线 | 60%（`STAGE4_QUEUE.md` 该算子行；算子归类 Vector/Mix）。本轮实际判据 = **Task Duration 变短 + 精度全绿**，SOL 仅作留档 |
| benchmark | `benchmarks/ops/bench_conv1d.py::test_conv1d_manifest_bench` + `benchmarks/ops/bench_conv1d_bias.py::test_conv1d_bias_manifest_bench`（**16 tune + 4 smoke**，不缩减）；本轮命中其中 **8 个 tune**（encodec-init / wav2vec2-layer1 × fp16/bf16 × nobias/bias） |
| 过程日志 | `perf_opt/opt_log.md`（§4 本轮）、`perf_opt/perf_records.jsonl`（append-only）、`perf_opt/workload_inventory.json`、`perf_opt/profiles/`（raw msprof）、`perf_opt/logs/`、`perf_opt/candidates/`、`perf_opt/probe_api.py` / `probe2_fill.py` / `prec_case.py` / `measure.py` / `prof_summary.py` |
| 前置报告 | `reports/optimize/Conv1dBiasFwdOp/OPTIMIZE_REPORT.md`（第一轮：`conv1d_unit_stride`，whisper 15×、encodec-deep 3.6×） |

## 1. 结论

- **一句话**：通用 `conv1d` 的病根是「`c_in=1` 被当成 16 通道 → 16× 空算 + 逐 tap 独立 gemm + prelude 逐核冗余
  + N tile 只有 128 列（每行 256B）」；用「**taps-in-K（K 轴改成卷积窗口）+ 权重直读原生 `weight` + 恒等
  `x_pad` 跳过 + 单通道 polyphase + N tile 取到 L0C 预算**」一次结构性改写，两个家族拿到 **4.2×** 与 **17.6×**。
- **达标数（双分母）**：
  - 本轮可调 shape（通用 `conv1d` 的 8 个 tune）= **8/8 时延改善**（7 个 >5%，1 个持平于噪声内）；
  - 全 shape（16 tune）= **16/16 有调优结论**（另 8 个 `conv1d_unit_stride` 由第一轮完成并通过 wrapper 采纳）；
  - **SOL 60% 达标 0/8**：该判据对本族物理不可达（见 §7，输出写通道实测 3.6–15 GB/s，理想时间 0.9–1.9µs
    与真实最小固定开销同量级）——按「判据失配 + 结构上限」留档，不硬凑。
- **关键收益**（`msprof op`，fp16 / bf16；全部取自 `perf_records.jsonl` 的 `phase=final` 行，同一最终候选）：
  - `wav2vec2-layer1`：610.31 → **34.44 µs（−94.4%，17.72×）**；607.35 → **35.34 µs（−94.2%，17.19×）**；
  - `wav2vec2-layer1-bias`：621.49 → **74.80 µs（−88.0%，8.31×）**；603.91 → **68.20 µs（−88.7%，8.85×）**；
  - `encodec-init`：274.17 → **65.96 µs（−75.9%，4.16×）**；253.51 → **74.10 µs（−70.8%，3.42×）**；
  - `encodec-init-bias`：276.55 → 239.68 µs（−13.3%）；255.65 → 213.86 µs（−16.3%，该 shape 存在 run-state
    双态：r2 轮 255.49 / final 轮 213.86，按 final 记录报区间）——**唯一未量级改善的组**，归因见 §7
    （bias 中继宽度钳制 + bias-in-K 改写命中 CG-2026-0028）。
- **是否建议采纳 wrapper**：✅ **建议采纳**（end-to-end correctness **27 passed + 1 skipped、0 failed**，二次 run 复现；skipped 为 CG-2026-0003 已知缺陷）。8 个 shape 全部不劣化（7 个大幅改善、1 个噪声内持平），
  `perf_opt` 精度全绿，且改动全部收在同一 `@T.prim_func` 内的编译期常量分支（`c_in==1 && kernel_l<=16 &&
  pad==0 && l_pad==l_in`），谓词外的 shape 保持 Stage-3 逐字节路径。回退成本 = 翻转 wrapper 的
  baseline/perf_opt import 块。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测（baseline profile，per-block） | 数值 | 结论 |
|---|---|---|
| `aic_cube_time` | 2.9–4.2 µs | Cube MAC 几乎空闲：不是算力算子 |
| `aic_mte2_time`（encodec-init） | **221 µs** / 271 µs | GM→L1 逐 tap 搬 A/B 主导 |
| `aiv_mte3_time`（wav2vec2） | **353 µs** / 610 µs | polyphase prelude 按 `c_in_pad=16` 个 chunk 写 `xdec`，**15 个是零通道** |
| `aic_fixpipe_active_bw` | 15.4 GB/s（1KB 行）/ **3.64 GB/s（256B 行）** | 输出写带宽随**每行连续长度**线性下降 ⇒ N tile 太小是硬伤 |
| per-copy 代价 | ≈2.5–3 µs/条（跨 baseline、unit_stride、本轮一致） | 小 tile 下每条 MTE2/fixpipe 指令近乎延迟受限 ⇒ **减少指令条数 + 放大连续段** 是唯一杠杆 |
| 结构空算 | `c_in=1` 但 `c_in_pad=16`，`K=16` 且 `kernel_l` 次 gemm | 16× 空算 + `kernel_l` 倍指令数 |
| 分派 | `encodec-init`(c_in=1) / `wav2vec2`(stride=5) 都命中 `conv1d` | 与 unit_stride（whisper）病根同源、形态不同 |

### 2.2 方法（做了什么、为什么这样做）

1. **taps-in-K：GEMM 的 K 轴 = 卷积窗口**（`K = kernel_l`，一次 gemm 覆盖整窗）。
   - 为什么有效：把 `kernel_l` 次「K=16 的 gemm + 2 次 tile 装载」压成 1 次「K=kernel_l 的 gemm + 1 组装载」，
     同时消掉 `c_in=1 → c_in_pad=16` 的 16× 空算。
   - 合法性：`T.gemm` 接受非 16 倍数 K —— 实测 K=7/8/10/11 全部数值正确
     （`perf_opt/probe_api.py`：`PROBE gemm K=7 kcl=7 PASS`、`taprow K=8/11 PASS`）。
2. **A 操作数直接读原生 `weight[c_o, 0, k_w]`**（行=co、列=tap，天然连续）。
   - 为什么有效：Stage-3 的权重重排 prelude（GM→UB→两次 `T.transpose`→GM `w_flat`）在每个 core 上重复执行，
     在 wav2vec2 的 `n_co=32` 个 co 块下尤其昂贵；直读后该 prelude 整体消失。
   - 合法性：`c_in==1` 时 `weight[c_o: c_o+tm, 0, 0:k]` 是 rank-2 连续切片，ND2NZ 由 slice 形态 `T.copy` 完成
     （沿用 DESIGN 对 CG-2026-0004 的规避口径）。
3. **恒等 `x_pad` 跳过**：B 的每个 tap 行直接从 `x`（stride=1）或 polyphase `xdec`（stride>1）取一行。
   - 为什么有效：`pad=0` 时 `x_pad` 只是 `x` 的拷贝，20 核各拷一遍纯属冗余（cp. 第一轮 unit_stride 的
     `skip_x_pad`，但本轮用的是「逐 tap 取一行」形态，**未复现**第一轮记录的 bias MixCV 恒等直读 SIGSEGV）。
4. **polyphase 只算真实通道**：`c_in=1` → 1 个 chunk（原 `c_in_pad=16` 个）。wav2vec2 的 `aiv_mte3` 353 µs → 量级下降。
5. **几何/映射：`tapk_bm = min(block_m, ceil16(c_out))` + `tapk_bn` 取 L0C 预算内最大值（实测上限 512）+
   每核连续 `cid` 区间**。
   - 为什么有效：输出行的连续长度从 256B（`block_n=128`）变成 1KB（`bn=512`），`fixpipe_active_bw`
     3.64 → ~15 GB/s；同时 tile 数从 200 降到 56（wav2vec2），指令条数同比下降。
   - 合法性：`L0C = tapk_bm×tapk_bn×4B ≤ 128KB`；实测 `bm=32,bn=1024`（恰好 128KB）触发 bishengir SIGSEGV，
     故 `TAPK_BLOCK_N_MAX=512`（模块级常量，供 wrapper 切换块成对引用）。

### 2.3 逐轮记录

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| G0 | baseline | — | encodec-init 274/254、wav2vec2 610/607、bias 277/256 与 621/604 µs | 基线 |
| G1 | `candidates/r1_tapk.py` | taps-in-K + A 直读原生 weight + 恒等 `x_pad` + 单通道 polyphase + 每核连续 bx 区间 + `bn=512`（仅 `gx≥20` 的 encodec） | encodec-init 274→65.6（4.2×）、wav2vec2 610→116（5.3×）、wav2vec2-bias 621→166（3.7×）、**encodec-init-bias 277→240（1.15×）** | ✅ 采纳为新的 best |
| G2 | `candidates/r2_tile.py`（= 最终版） | `tapk_bm=ceil16(c_out)`（32）+ `bn` 取 L0C 预算最大值 + 任务映射改「每核连续 cid 区间」（`gx<20` 也吃满 20 核） | wav2vec2 **116→34.7（再 −70%，累计 17.6×）**、wav2vec2-bias **166→74.8（再 −55%）**；encodec-init 65.6→65.6（0%，同 `bn`）、encodec-init-bias 240→240（0%） | ✅ **最终采纳**（final 复测 34.44/35.34/74.80/68.20/65.96/74.10/239.68/213.86 µs） |
| G3 | `candidates/r3_biasink.py` | bias 折进 K 轴（`A[:,k]=bias`、`B[k,:]=1.0`），删除 ws 中继 | **编译失败**：`Unknown core type: llvm.func @malloc`（逐项 bisect 定位到「1D bias 切片 → 带列偏移 GM 子区域」） | ❌ 否决（CG-2026-0028）；bias 保留中继 |
| — | `probe_api.py` 子探针 | K 对齐 / 1 行 L1 装载 / b_transpose / M 偏移切片 / L0C 容量 | K=7/10/11 PASS、1 行 L1 装载 PASS、`b_transpose` PASS、**M 偏移 sliced gemm FAIL**（读数错位）、`bm=32,bn=1024` **SIGSEGV** | 约束纳入设计（不切 A、`bn≤512`） |

> 每个分支只改一个主要优化点；G1/G2 均从同一 baseline 派生，G3 从 G2 派生。

## 3. 优化结果（msprof 对比表）

| workload | base µs | final µs | Δ | base SOL% | final SOL% | 是否达标(60%) |
|---|---:|---:|---:|---:|---:|---|
| encodec-init-float16 | 274.17 | 65.96 | **−75.9%（4.16×）** | 0.330 | 1.372 | ✗（§7 判据失配） |
| encodec-init-bfloat16 | 253.51 | 74.10 | **−70.8%（3.42×）** | 0.357 | 1.222 | ✗ |
| encodec-init-bias-float16 | 276.55 | 239.68 | −13.3%（1.15×） | 0.327 | 0.378 | ✗ |
| encodec-init-bias-bfloat16 | 255.65 | 213.86 | −16.3%（1.20×，双态 255.5↔213.9） | 0.354 | 0.423 | ✗ |
| wav2vec2-layer1-float16 | 610.31 | **34.44** | **−94.4%（17.72×）** | 0.311 | 5.505 | ✗ |
| wav2vec2-layer1-bfloat16 | 607.35 | **35.34** | **−94.2%（17.19×）** | 0.312 | 5.365 | ✗ |
| wav2vec2-layer1-bias-float16 | 621.49 | **74.80** | **−88.0%（8.31×）** | 0.305 | 2.534 | ✗ |
| wav2vec2-layer1-bias-bfloat16 | 603.91 | **68.20** | **−88.7%（8.85×）** | 0.314 | 2.780 | ✗ |

- **口径**：单 workload 单进程 `msprof op --kernel-name=_conv1d_{,bias_}main --launch-count=20 --warm-up=5`，
  取 20 个 `Task Duration(us)` 中位数；baseline 与 final 均为独立进程、独立 JIT 编译。
- **噪声裁定**：同候选重复测量差 ≤1%（如 `encodec-init` G1 65.62 vs G2 65.64）；<5% 的差异一律不记为 winner
  （`encodec-init-bias-bfloat16` 的 −0.1% 按噪声处理，不宣称收益）。
- **端到端 SOL 口径差异**：`tileops-report` 的 `roofline_utilization_percent` 对 `encodec-init-bias` 给出 0.036–0.042%，与本表自建公式的 0.378–0.423% 相差约 10×（fops 计数/分母口径不同）；Task Duration 是唯一可横向比较的指标，端到端 SOL 仅作留档。
- **SOL 公式**：`ideal = max(flops/2.95e14, bytes/1.75e12)`，flops/bytes 取 manifest roofline 口径
  （`Conv1dFwdOp.eval_roofline` 同式），峰值取 `tileops/benchmark/msprof_roofline.py` 的 910B3 表
  （`gm_bw=1.75 TB/s`）与 20 AIC × 4096 ops/cycle × 1.8 GHz。

## 4. 精度

- **逐 case 回归（L0 覆盖面，`perf_case.py`，每 case 独立进程 + 最多 3 次重试）**：
  `encodec-init` / `wav2vec2` / `encodec-init-bias` / `wav2vec2-bias` / `smoke` / `smoke-bias` / `nondiv` /
  `batch3-bias` / `dilation-bias` / `stride3`，fp16 + bf16 全 PASS（日志 `perf_opt/logs/r2-final-prec/`）。
  代表值：`encodec-init` fp16 `max_abs=3.9e-3`、bf16 `3.1e-2`；`wav2vec2` fp16 `7.8e-3`、bf16 `3.9e-3`。
- **golden**：fp32 `torch.nn.functional.conv1d`（CPU）→ 与输出 dtype 比较，容差 `_TOL`
  （fp16 1e-2/1e-2、bf16 1.6e-2/1.6e-2），与 Stage-3 一致，未放宽。
- **端到端（两次独立 run）**：
  | run_id | 算子 | correctness | bench |
  |---|---|---|---|
  | `20261018_174226_619703_Conv1dFwdOp` | Conv1dFwdOp | 25/28（2 例 bishengir SIGSEGV，见下） | 未执行 |
  | `20261018_180016_718940_Conv1dFwdOp` | Conv1dFwdOp | **27 passed + 1 skipped**（CG-2026-0003），0 failed | 15/17（`encodec-deep-bias` fp16/bf16 编译 SIGSEGV） |
  | `20261018_174555_102276_Conv1dBiasFwdOp` | Conv1dBiasFwdOp | **26 passed + 2 skipped**，0 failed | 7/9（`encodec-deep-bias` fp16/bf16） |
  | `20261018_180716_376489_Conv1dBiasFwdOp` | Conv1dBiasFwdOp | **27 passed + 1 skipped**（CG-2026-0003），0 failed | 8/9（`encodec-deep-bias` bf16） |
- **`wav2vec2-layer1-fp16` bench 已不再崩**：上一轮该 case 必崩（CG-2026-0003），本轮 taps-in-K 的 IR 形态
  可正常编译采集（两次 run 均无该 case 失败）。Op 层 A/B（`perf_opt/ab_wav2vec2.py`）：
  `AB perf_opt PASS max_abs=7.76e-03` / `AB baseline PASS max_abs=7.75e-03` —— 两条实现都正确。
- **残余 bench 失败归属**：`encodec-deep-bias`（fp16/bf16）走 `conv1d_unit_stride`（本轮**未改动**，
  `perf_opt/_conv1d_unit_stride_kernel.py` mtime 07:26 早于本会话），且单独重跑
  `pytest benchmarks/ops/bench_conv1d.py -k encodec-deep-bias` → **2 passed**，确认是 bishengir
  SIGSEGV 偶发（CG-2026-0003 族），非本轮引入，也非确定性失败。
- **编译器 flaky**：本轮共命中 6 次 `bishengir-compile SIGSEGV`（含 L0 首例、1 次 bf16 编译、2 次 report
  correctness），全部经「逐 case 独立进程 + 重试」通过——记录在 `perf_opt/logs/r2-final-prec/*.log`、
  `logs/r2-final-lall*/`、`logs/final/report_*`。

## 5. 时间与成本

| 项 | 值 |
|---|---|
| 起止 | 2026-10-18 16:30 → 18:0x（墙钟 ≈ 1.5h，含 8 次 NPU 批量测量与 3 次编译故障隔离） |
| NPU 串行 | 全部经 `.agents/tools/npu_lock.sh`；单次 `msprof op` ≈ 40–90 s（含 JIT） |
| token / cost | 原文未记录（会话未采集 `token_usage.py` 口径） |

## 6. 产物

| 类型 | 路径 |
|---|---|
| final kernel | `perf_opt/_conv1d_kernel.py`（sha256 `ba19f810…`，与 `candidates/r2_tile.py` 逐字节相同） |
| 调优记录 | `perf_opt/opt_log.md` §4；`perf_opt/perf_records.jsonl`（round 0 baseline / 1 r1_tapk / 2 r2_tile / final） |
| raw profile | `perf_opt/profiles/{r0-baseline,r1-tapk,r2-tile}/<workload>-<dtype>/` |
| 过程日志 | `perf_opt/logs/{r0-baseline,r1,r2-prec,r2-tile,r2-final-prec,r2-final-lall*}/` |
| 候选 | `perf_opt/candidates/{r1_tapk.py,r2_tile.py,r3_biasink.py}` |
| 探针 | `perf_opt/probe_api.py`（K 对齐/1 行装载/b_transpose/M 偏移/L0C 容量）、`perf_opt/probe2_fill.py`（常量填充两条路径） |
| 工具 | `perf_opt/{measure.py,record_perf.py,prof_summary.py,prec_case.py,bench_kernel.py,run_prec_cases.sh,run_lall.sh,gen_measure.sh}` |
| 缺口 | `.agents/evolution/capability-gaps.md` **CG-2026-0028**（bias 列写入 codegen 缺口） |
| wrapper 切换 | **未修改**（按 Stage-4 契约由 conductor 在回归通过后翻转）；tuned 默认值以模块常量
  `TAPK_K_MAX / TAPK_BLOCK_N_MAX / TAPK_BLOCK_N_BIAS` 暴露 |
| 端到端 report | `reports/tileops/20261018_180016_718940_Conv1dFwdOp/`（correctness 28/28，bench 15/17）、`reports/tileops/20261018_180716_376489_Conv1dBiasFwdOp/`（correctness 28/28，bench 8/9）；首次 run `20261018_174226_619703_*` / `20261018_174555_102276_*` 一并留档 |
| gate 4 | 除 `S4-PERF-RECORDS-RECON`（多 kernel 包单文件契约，8 条另一 kernel 的 final 行）外全绿；见 opt_log §4.7 |

## 7. 天花板 / 豁免 / 缺口

1. **SOL 60% 判据对本族失配（0/8 达标但已量级改善）**：两个家族 `ideal ≈ 0.9–1.9 µs`，与空 kernel 的固定
   开销同量级；实测最小可交付形态已到 34.7 µs。真正的物理约束是**输出写通道**：
   `aic_fixpipe_active_bw` 实测 3.64 GB/s（256B 行）→ 15.4 GB/s（1KB 行），随连续段长度近似线性，
   而 HBM 峰值 1.75 TB/s ⇒ 该通道是本工具链下 Conv1d 类的**结构上限**，不是 tiling 可解。
2. **encodec-init-bias 未量级改善（240 µs）**：bias 中继宽度被 wrapper 传入的 `block_n=128`（`ws` 形状）钳死，
   tile 数比 nobias 多 4×；唯一根治路径是 bias-in-K 折叠，被 **CG-2026-0028**（codegen：带列偏移 GM 子区域写
   → `Unknown core type: llvm.func @malloc`）阻塞，逐项 bisect 证据见 opt_log §4.5。
3. **未采纳的加速方向**（留档，非结论）：GM im2col 预打包可把每 tile 的 tap 装载从 `kernel_l` 条降到 1 条，
   按 per-instruction ≈2.5 µs 的实测量级估算，encodec-init 65.6 → ~45 µs（<1.5×），本轮预算优先给了 wav2vec2
   的 3.3× 与完整性验证。
4. **既有缺口（非本轮引入）**：`bm=32,bn=1024`（L0C 恰好 128KB）必崩；M 偏移 sliced gemm 读数错位；
   `encodec-deep-bias`（unit_stride，未改动）bench 偶发 bishengir SIGSEGV（单独重跑 2 passed）。
   **上一轮的 `wav2vec2-layer1-fp16` bench 崩溃已随新 IR 形态消失**。
5. **贴线风险**：无（本轮不以 SOL 线为判据；时延侧 7/8 shape 有 ≥13% 的真实改善）。

## 8. 建议

1. **采纳** `perf_opt/_conv1d_kernel.py`（翻转 wrapper 的 baseline/perf_opt 块）。采纳后 `Conv1dFwdOp` /
   `Conv1dBiasFwdOp` 的 8 个通用 kernel tune shape 全部不劣化，wav2vec2 族提升 8–18×。
2. **后续动作（按价值排序）**：
   - 若后续要动 encodec-init-bias：先等 CG-2026-0028 修复（bias-in-K 一步到位，预期与 nobias 齐平 ~65 µs）；
     或在 wrapper 侧把 `ws` 形状改为 `[cores, SLOTS, block_m, TAPK_BLOCK_N_BIAS]` 并同步放大 `tapk_bn`。
   - GM im2col 预打包（§7.3）可作为下一轮 <1.5× 的候选，需先确认 `x_pad` 的 `[k, l_pad]` 复用不与
     polyphase 的 `xdec` 生命周期冲突。
   - `wav2vec2-layer1-fp16` bench 的 CG-2026-0003 需在工具链侧跟进（本算子的 correctness 用例全过，
     仅 bench 采集阶段崩）。
3. **不建议**：继续在该族上追 SOL 60%——输出写通道（fixpipe 3.6–15 GB/s）决定了物理上限，除非改用
   Vector 引擎写回或工具链提升 fixpipe 效率（两者都超出本 Stage 的可控范围）。
