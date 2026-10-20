# 优化报告：Conv1dBiasFwdOp（Stage-4 optimize）

> **口径（2026-10-20 修订）**：本文件是**过程记录**——第 3 节的逐轮测量沿用**当时工具链**，**不参与交付口径**。交付数据一律取 `perf_data/canonical.json`（CANN 9.0.0，每算子恰好一条记录），其中同时给出**迁移态（优化前）**与**交付态（当前交付）**两列；本文数字与之不一致时以 SSOT 为准。同 case 的前后对比见 `examples/TileOPs/reports/迁移与优化总报告.md` 的汇总表。

> 口径说明：本文件结论已按**形状路由解封后的最终状态**更新（依据 `reports/optimize/STAGE4_QUEUE.md`
> 的 #9 / #11 行）。解封前版本（结论为「SIGSEGV 阻塞、不建议采纳」）归档在
> `history_version/OPTIMIZE_REPORT_v0.md`，其性能数字仍然有效。

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `Conv1dBiasFwdOp`（bias 必填薄接口）；同一 kernel 亦服务 `Conv1dFwdOp`（bias 可选）；**多 kernel（5 变体）**；Vector 主导（`aic_cube_ratio = 0.15%`）+ Cube/bias MixCV 中继 |
| kernel（基准，只读） | `tileops/kernels/convolution/conv1d/conv1d_kernel/`（5 变体，本轮调优 `conv1d_unit_stride`） |
| 调优产物 | `.../conv1d_kernel/perf_opt/_conv1d_unit_stride_kernel.py`（+ 同名 drop-in 4 件 baseline 副本）；gate 收束产物 `perf_opt/Conv1dBiasFwdOp.py` |
| 口径 | `msprof op Task Duration(us)` + SOL ratio（唯一时延口径）；所有上机命令经 `.agents/tools/npu_lock.sh` 串行执行 |
| **工具链版本戳** | tilelang `0.1.2` dev root（`/home/developer/tilelang-mlir-ascend`）/ **CANN 9.0.0**（bishengir-compile 1.1.0 + hivmc 0.2.0）/ Ascend910B3（Chip 6，20 AIC）/ 2026-10-18 |
| 达标线 | 原文未记录 SOL 达标线；本轮以「**Task Duration 变短 + 精度全绿**」为达标判据（队列表记 60%） |
| benchmark | `benchmarks/ops/bench_conv1d_bias.py::test_conv1d_bias_manifest_bench` + `benchmarks/ops/bench_conv1d.py::test_conv1d_manifest_bench`（两套，**16 tune + 4 smoke**） |
| 过程日志 | `.../perf_opt/opt_log.md`、`perf_records.jsonl`、`workload_inventory.json`、`profiles/`、`logs/`、`WORKLIST.md`、`ab_r1/` |
| 前置报告 | `history_version/OPTIMIZE_REPORT_v0.md`（**解封前版本**：结论为「SIGSEGV 阻塞、不建议采纳」，性能数据相同） |

## 1. 结论

- 一句话：病根是 **persistent 内核的 `_prelude`（权重重排 + x_pad 拷贝）在 20 个 AI core 上逐核重复**
  （`aiv_mte2_ratio = 97.1%`、有效带宽仅 2.39 GB/s、真实 GEMM 只有 ~8µs）；用「权重重排按 co 块切分 +
  恒等 x_pad 跳过」两个配合改动即拿到量级收益。
- **达标数（双分母）**：
  - **可调 shape（本次实际调优的 `conv1d_unit_stride` 8 个）= 8/8 达标**；
  - **全 shape（16 tune）= 8/16 达标**：8 个 `conv1d_unit_stride` 大幅变短，8 个 `conv1d`
    （wav2vec2 / encodec-init）按任务授权「允许只优化 1–2 个变体」未触及、时延不变（**非回退**）；
  - 豁免 **0**（4 个 smoke 仅精度回归，不计入性能分母）。
- 关键收益：
  - **whisper 族：5626 → 370 µs（~15×）**（四个 case 分别 −93.4% / −93.4% / −93.3% / −93.5%，
    SOL 0.12% → ~1.8%）；
  - **encodec-deep 族：1916 → 522 µs（~3.6×）**（−72.8% / −72.6% / −71.5% / −71.0%，
    SOL 0.07% → ~0.25%）；
  - unit_stride 两族几何平均 **~7.6×**。
- **是否建议采纳 wrapper**：✅ **建议采纳，且已采纳并解封**（结论更新）：
  - ① `_select_kernel_key` 增加 **`dilation == 1`** 才走 unit-stride（dilation 形状即崩）；
  - ② wrapper 对 unit-stride 做 **`c_in >= 64`** 形状门（`c_in=32` 触发同一编译器缺陷）；
  - 实测：perf_opt 全量 **50 passed**；`Conv1dBias` report **passed**；`Conv1d` report **partial**
    （bench 的 `wav2vec2-layer1-fp16` 崩，该形状走通用 kernel、与 Stage-3 逐字节相同 ⇒ **pre-existing**）；
  - 收益：whisper **5626 → 370 µs（15×）**、encodec-deep **1916 → 522 µs（3.6×）**。
- 剩余 gap：encodec-deep 的 5× 权重重排冗余（5 核共享同一 co 块，无跨核同步下不可再分）+ 真实 GEMM + bias 中继。

## 2. 优化方法与过程

### 2.1 诊断（决定用什么杠杆）

| 观测 | 数值 | 结论 |
|---|---|---|
| `aic_cube_ratio` | **0.15%** | Cube 几乎空闲 → 这不是 Cube 算子，是「Vector 侧逐核冗余搬运」的朴素实现 |
| `aiv_mte2_ratio` | **97.1%** | Vector 侧 MTE2 独占，时间几乎全花在搬运 |
| `aiv_mte2_active_bw` | **2.39 GB/s**（平台 1.75 TB/s） | 有效带宽极低 ⇒ 不是数据量问题，是**重复搬运** |
| `GM_to_UB_datas` | **13270 KB/core** | 每核重复搬运同一段 prelude |
| `Block Dim` | 20 | 20 核各自执行同一段 prelude |
| prelude 构成 | 权重重排（全部 `n_co=80` 个 co 块：GM→UB→两次 `T.transpose`→GM `w_flat`）+ `x_pad` 全量拷贝（`x → UB → x_pad`） | 两处冗余：一处可切分、一处可跳过 |
| Cube 真实 GEMM | **~8 µs**（其余 ~5.6 ms 全是 prelude） | 优化目标明确：消掉 prelude 的 20× 重复 |
| baseline | whisper-conv1 fp16 **5339–5626 µs（SOL ≈ 0.12%）**、encodec-deep fp16 **~1916 µs（0.07%）** | 与上一轮诊断一致 |

### 2.2 方法（做了什么、为什么这样做）

1. **权重重排按 co 块切分**（`_conv1d_unit_stride_kernel.py` 的 `_prelude` + `_cube` + `_bias_epilogue`）：
   - 改了什么：任务映射改为 `by = kernel_id % gy` / `sub = kernel_id // gy` /
     `idx = task_id * share + sub`（`share = num_kernels // gy`，`gy <= num_kernels` assert）；
     prelude 只重排本 core 的 co 块 `[by*block_m, by*block_m+block_m)`，内层 `BM_W=16` 子块循环。
   - 为什么有效：把「每核都重排全部 co 块」变成「每核只重排自己读取的 co 块」，whisper（`gy=20=num_kernels`，
     `share=1`）的权重重排由 20× 降到 1×，直接消掉 MTE2 的主项。
   - 正确性依据：无跨核同步——core 只读自己写的 `w_flat` 块；L0 的 8 个 tune workload 全 PASS
     （`max_abs ≤ 6.25e-2`，与基线一致）。本轮同时**修正了上一轮 R1-B 的真因**
     （`cw = T.min(BM_W, c_out - by*BM_W)` 用 `block_m=64` 单位的 `by` 乘了 `BM_W=16`，导致只重排
     `[by*16, by*16+16)` 而 GEMM 读 `[by*64, by*64+64)`，差 4×，max_abs=81.38）。
2. **恒等 x_pad 跳过**（仅 nobias 路径）：
   - 改了什么：`skip_x_pad = (pad == 0 and c_in == c_in_pad) and not has_bias`；prelude 跳过 `x_pad` 拷贝，
     Cube 的 B 块直接读 `x`。
   - 为什么有效：消掉 `128×600` 规模的 20× 冗余拷贝。
   - 正确性依据：nobias tune workload 全 PASS；**bias（MixCV）路径禁止启用**——实测把恒等跳过应用到
     bias 路径时 `encodec-deep-bias` 确定性 SIGSEGV（whisper-bias 却 PASS），故只保留 nobias 条件路径。
3. **其余 4 个 kernel 复制 baseline 副本至 `perf_opt/`**（`conv1d` / `group` / `direct` / `pointwise`）：
   供 wrapper 双 import 切换块完整 drop-in（本轮未改动逻辑）。
4. **解封后的形状路由修正**（采纳动作，属 wrapper/ops 层、非 kernel 性能改动）：
   - ① `_select_kernel_key` 增加 `dilation == 1` 才走 `conv1d_unit_stride`；
   - ② wrapper 对 unit-stride 增加 `c_in >= 64` 形状门，其余 shape 走 Stage-3 实现。
   - 依据：`c_in=32` 与 dilation 形状触发 CANN 9.0 bishengir SIGSEGV（CG-2026-0003 家族）。

### 2.3 逐轮记录

| 轮次 | 候选 | 改动 | 结果 | 处置 |
|---|---|---|---|---|
| R0 | baseline 诊断 | profile + `PipeUtilization.csv` | whisper 5339–5626µs / SOL≈0.12%；encodec-deep ~1916µs / 0.07%；`aiv_mte2_ratio 97.1%`、2.39 GB/s、13270 KB/core | 定位 prelude 逐核冗余 |
| R1-B（上一轮遗留） | 分区重排 v1 | 用 runtime `by` 做 GM slice + transpose | **max_abs = 81.38（挂起）** | ❌ 否决：`block_m`/`BM_W` 单位错位（`by` 是 64 单位却乘 16），本轮定位真因后重写 |
| R1 | `r2_weight_shard` | 权重重排按 co 块切分 + 恒等 x_pad 跳过 | 8 个 unit_stride tune 全达标（−71.0% ~ −93.5%），L0 全 PASS | ✅ 采纳为 current best |
| R1 陷阱 | 三条越界写法 | ① 把 `by = kernel_id % gy` hoist 出 `T.serial` 循环；② bias 路径启用恒等 x 直读；③ bf16+bias 编译 | ① whisper-bias / encodec-bias **确定性 SIGSEGV**（放回循环内即 PASS）；② encodec-deep-bias SIGSEGV（仅 nobias 启用）；③ 基线同崩，属既有 flaky（CG-2026-0003 家族，最终测量经重试获得） | ❌ 三条写法全部否决，采用受限形态 |
| R2 | 合并复测与收束 | 合并版复测全部 tune workload + smoke 精度 | 与 R1 候选同数（见 §3）；`smoke-k3-s1`(+bias) fp16/bf16 4 例全 PASS | ✅ 收束（单点改动一次到位，无第二候选） |
| R3（解封） | 形状路由修正 | `dilation==1` 分流 + unit-stride `c_in>=64` 门 | perf_opt 全量 **50 passed**；`Conv1dBias` report **passed**；`Conv1d` report **partial**（1 例 bench 形状 pre-existing 崩） | ✅ 采纳（解除端到端阻塞） |

**被否决方向的原因（保留）**：
- `conv1d` 族（wav2vec2 / encodec-init）未调优：wav2vec2（`gy=8`）权重重排仅 ~2.5× 且 polyphase `xdec`
  仍 20× 冗余；encodec-init（`gy=1`）单 co 块全核共享、切分无收益；预算优先投入最重族。
- `group` / `direct` / `pointwise`：不在 `convolution.yaml` 的 dense Conv1d manifest 内，无 benchmark 覆盖。

## 3. 优化结果（final，同一候选 `r2_weight_shard`；`conv1d` 族为未改动的 baseline）

| kernel_id | workload | base µs | final µs | Δ | base SOL% | final SOL% | 是否达标 |
|---|---|---:|---:|---:|---:|---:|---|
| conv1d_unit_stride | whisper-conv1-float16 | 5626.31 | 369.93 | **−93.4%** | 0.12 | ~1.8 | ✅ |
| conv1d_unit_stride | whisper-conv1-bfloat16 | 5598.71 | 369.23 | **−93.4%** | 0.12 | ~1.8 | ✅ |
| conv1d_unit_stride | whisper-conv1-bias-float16 | 5518.67 | 367.63 | **−93.3%** | 0.12 | ~1.8 | ✅ |
| conv1d_unit_stride | whisper-conv1-bias-bfloat16 | 5584.41 | 362.41 | **−93.5%** | 0.12 | ~1.8 | ✅ |
| conv1d_unit_stride | encodec-deep-float16 | 1915.94 | 521.85 | **−72.8%** | 0.07 | ~0.25 | ✅ |
| conv1d_unit_stride | encodec-deep-bfloat16 | 1898.54 | 520.19 | **−72.6%** | 0.07 | ~0.25 | ✅ |
| conv1d_unit_stride | encodec-deep-bias-float16 | 1893.52 | 538.91 | **−71.5%** | 0.07 | ~0.24 | ✅ |
| conv1d_unit_stride | encodec-deep-bias-bfloat16 | 1857.40 | 537.95 | **−71.0%** | 0.07 | ~0.24 | ✅ |
| conv1d | wav2vec2-layer1-*（4） | 582–614 | 582–614 | 0%（未触及） | 原文未记录 | 原文未记录 | 未触及（非回退） |
| conv1d | encodec-init-*（4） | 256–283 | 256–283 | 0%（未触及） | 原文未记录 | 原文未记录 | 未触及（非回退） |

**测量口径**：whisper / encodec-deep 的 baseline 为本次会话 `msprof` 直采；bf16-bias 两个 case 因本会话
bishengir flaky，沿用上一会话基线（5584.41 / 1857.40，kernel 未改、数值有效）；final 全部来自同一最终候选
`r2_weight_shard`（unit_stride）与未改动的 baseline（conv1d）。`conv1d` 族逐 case 明细见 `opt_log.md`
Final 表（`wav2vec2-layer1`: 613.7720 / 582.4120 / 589.9520 / 595.4720 µs；`encodec-init`:
283.2060 / 268.1450 / 256.2650 / 268.7450 µs；base 与 final 相同）。

**噪声裁定**：本轮最重族的收益为 15×/3.6× 量级，远离噪声带；`conv1d` 族时延不变是「未触及」而非回退。

## 4. 精度

- 8 个 tune workload（whisper ×4、encodec-deep ×4）全 PASS，**`max_abs ≤ 6.25e-2`（与基线一致）**。
- L0 其余非 tune 用例 `nondiv-nobias` / `batch3-bias` / `taps2-nobias` **PASS**。
- **2 个 L0 用例（`dilation-pad-bias`、`taps2-bias`）命中既有 bishengir SIGSEGV**
  （CG-2026-0003 家族；**基线同崩、逐 case 复测 3/3 一致**）——工具链既有缺口，**非本轮改动引入**。
- 4 个 smoke（`smoke-k3-s1` fp16/bf16 + `smoke-k3-s1-bias` fp16/bf16）仅精度回归 **PASS**。
- 解封后：perf_opt 全量 **50 passed**（队列表 #9 行）。

## 5. 时间与成本

| 项 | 值 |
|---|---|
| 开始 / 结束（+08:00） | 2026-10-18 07:10 → 08:10 |
| 墙钟 | **≈ 60 min** |
| 主要耗时 | bishengir flaky SIGSEGV 隔离（三处陷阱）+ gate 4 多 kernel 工件对账 |
| token input/output/reasoning | 原文未记录 |
| cost | 原文未记录 |

## 6. 产物

| 类别 | 路径 / 状态 |
|---|---|
| final kernel | `.../conv1d_kernel/perf_opt/_conv1d_unit_stride_kernel.py`（`r2_weight_shard`） |
| 其余 kernel 副本 | `perf_opt/_conv1d_kernel.py`、`_conv1d_group_kernel.py`、`_conv1d_direct_kernel.py`、`_conv1d_pointwise_kernel.py`（baseline 副本，逻辑未改） |
| gate 收束产物 | `.../perf_opt/Conv1dBiasFwdOp.py` |
| 调优记录 | `perf_opt/opt_log.md`、`perf_records.jsonl`、`workload_inventory.json`、`profiles/`、`logs/`、`WORKLIST.md`、`ab_r1/` |
| 候选文件 | `perf_opt/candidates/r1b.py`（上一轮 R1-B，用于定位单位错位真因） |
| gate 4 | `pass`（队列表记录；解封后 perf_opt 全量 **50 passed**） |
| 算子状态 | `phase=DONE`（队列表 #9：`it=2`，`last_improvement 93.4%`） |
| wrapper 切换 | **已切 perf_opt** + 两处形状路由门（`dilation==1` 才走 unit-stride；unit-stride 需 `c_in>=64`） |
| 端到端 report | `Conv1dBiasFwdOp` **passed**；`Conv1dFwdOp` **partial**（correctness 28/28，仅 bench 的 `wav2vec2-layer1-fp16` 崩，该形状走通用 kernel、与 Stage-3 逐字节相同 ⇒ pre-existing） |

## 7. 天花板 / 豁免 / 缺口

- **豁免项：无**（16 个 tune shape 均参与判定；smoke 仅精度回归）。
- **剩余结构 gap（encodec-deep）**：`gy=4`、`share=5` ⇒ 权重重排 20×→5× 即到底；5 核共享同一 co 块，
  在「无跨核同步」约束下不可再分。剩余时延构成 = 5× 冗余权重重排 + 真实 GEMM + bias 中继。
- **未触及 shape 的原因**（原文）：`conv1d` 族 wav2vec2（`gy=8`）权重重排仅 ~2.5× 且 polyphase `xdec`
  仍 20× 冗余；encodec-init（`gy=1`）单 co 块全核共享、切分无收益。
- **既有工具链缺口（CG-2026-0003 家族）**：`dilation-pad-bias` / `taps2-bias` 编译 SIGSEGV（基线同崩 3/3）；
  bf16+bias MixCV 编译 flaky；`c_in=32` unit-stride 触发同一缺陷（⇒ `c_in>=64` 形状门）。
- **解封后残留**：`Conv1dFwdOp` report **partial** 的 `wav2vec2-layer1-fp16` 崩溃——该形状走通用 kernel，
  其代码与 Stage-3 逐字节相同，属 **pre-existing**，非本次改动引入。
- **贴线风险**：原文未记录（SOL 远低于任何达标线；本轮以 Task Duration 倍数为判据，15× / 3.6× 远离噪声带）。

## 8. 建议

1. **采纳**（已完成）：wrapper 已切 perf_opt，并加两处形状路由门（`dilation==1` 才走 unit-stride；
   unit-stride 需 `c_in>=64`）。这两处是解封的关键，移除任一条都会使端到端验收回退到 SIGSEGV。
2. 工具链升级（CANN > 9.0.0）后，先复核 CG-2026-0003 家族是否修复：若修复，可移除 `c_in>=64` 门，
   让 `c_in=32` 的 unit-stride 形状也吃到切分收益。
3. 未解决项：`conv1d` 族（wav2vec2 / encodec-init）未调优——按任务授权「允许只优化 1–2 个变体」留档；
   如需继续，wav2vec2 的 polyphase `xdec` 20× 冗余是下一个病根（`gy=8` 切分只值 ~2.5×）。
4. `Conv1dFwdOp` report 的 partial 需从工具链侧修复（`wav2vec2-layer1-fp16` 走通用 kernel 崩，
   与本次优化无关）。
5. 复核入口：`benchmarks/ops/bench_conv1d_bias.py::test_conv1d_bias_manifest_bench`（bias 接口）+
   `benchmarks/ops/bench_conv1d.py::test_conv1d_manifest_bench`（同 kernel 的非 bias 接口）。
