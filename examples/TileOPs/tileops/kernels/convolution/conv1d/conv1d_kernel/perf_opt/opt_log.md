# Conv1dFwdOp / Conv1dBiasFwdOp — Stage 4 调优日志

> 工具链版本戳：CANN **9.0.0**（bishengir-compile 1.1.0 / hivmc 0.2.0）/ tilelang 0.1.2 dev root
> (`/home/developer/tilelang-mlir-ascend`) / Ascend910B3（Chip 6，20 AIC）/ 2026-10-18。
> 口径：**msprof op Task Duration(us)**（唯一时延口径）。
> NPU 纪律：所有上机命令经 `.agents/tools/npu_lock.sh` 串行执行。

## 0. 对象与分派

| label | 分派 kernel | kernel 文件 | 是否调优 |
|---|---|---|---|
| whisper-conv1(+bias) / encodec-deep(+bias) | `conv1d_unit_stride` | `_conv1d_unit_stride_kernel.py` | ✅ 第一轮（§1–§3） |
| wav2vec2-layer1(+bias) / encodec-init(+bias) / smoke-k3-s1 | `conv1d` | `_conv1d_kernel.py` | ✅ **第二轮（§4，本会话）** |
| group / depthwise / pointwise | `_conv1d_{group,direct,pointwise}_kernel.py` | 同名 | ❌ 不在 benchmark manifest 内 |

benchmark 入口（两套，tune 集合不缩减）：
`benchmarks/ops/bench_conv1d.py::test_conv1d_manifest_bench`（Conv1dFwdOp）+
`benchmarks/ops/bench_conv1d_bias.py::test_conv1d_bias_manifest_bench`（Conv1dBiasFwdOp）。

调用链：`Conv1dFwdOp/Conv1dBiasFwdOp.forward → _select_kernel_key → Conv1dUnitStrideKernel/Conv1dKernel → _run`。

## 1. Baseline（迁移态，未改任何 kernel）

whisper-conv1 fp16 **5339–5626 µs（SOL ≈ 0.12%）**、encodec-deep fp16 **~1916 µs（0.07%）**，
与上一轮诊断一致：这不是 Cube 算子，是「Vector 侧逐核冗余搬运」的朴素实现。

首轮诊断（profile：`perf_opt/profiles/whisper-conv1-fp16/.../PipeUtilization.csv`）：

| 指标 | 实测 | 读数 |
|---|---:|---|
| `aic_cube_ratio` | 0.15% | Cube 几乎空闲 |
| `aiv_mte2_ratio` | 97.1% | Vector 侧 MTE2 独占 |
| `aiv_mte2_active_bw` | 2.39 GB/s | 远低于 1.75 TB/s 平台带宽 |
| `GM_to_UB_datas` | 13270 KB/core | 每核重复搬运 |
| `Block Dim` | 20 | 20 核各自执行同一段 prelude |

**根因**：`_prelude`（Vector Scope）在每个 kernel core 上重复执行：
1. 权重重排：对全部 `n_co=80` 个 co 块做 GM→UB→两次 `T.transpose`→GM `w_flat`（whisper）；
2. `x_pad` 全量拷贝：`x → UB → x_pad`。

Cube 真正的 GEMM 只有 ~8 µs，其余 ~5.6 ms 全部是 prelude 的 20 核重复搬运。

## 2. Round 1（R1）：权重重排按 core 切分 + 恒等 x_pad 跳过

上一轮 R1-B 的分区重排挂起于「runtime `by` 做 GM slice + transpose 落块错误」（max_abs=81.38）。
本轮复读 `perf_opt/candidates/r1b.py` 后**定位到真实 bug**：R1-B 的 prelude 用
`cw = T.min(BM_W, c_out - by * BM_W)`（`by` 是 `block_m=64` 单位的下标，却乘了 `BM_W=16`），
导致每个 core 只重排了 `co ∈ [by*16, by*16+16)`，而 GEMM 读的是 `[by*64, by*64+64)`——差 4×。

### R1 改动（两处，合入同一 `@T.prim_func`）

1. **权重重排切分**（`_conv1d_unit_stride_kernel.py`）：
   - 任务映射改为按 co 块分区：`by = kernel_id % gy`，`sub = kernel_id // gy`，
     `idx = task_id * share + sub`（`share = num_kernels // gy`）；`gy <= num_kernels` assert。
   - prelude 只重排本 core 的 co 块 `[by*block_m, by*block_m+block_m)`，内层 `n_sub_w` 个
     `BM_W=16` 子块循环（修正 R1-B 的 `block_m/BM_W` 单位错误）。
   - 无跨核同步：core 只读自己写的 `w_flat` 块。
2. **恒等 x_pad 跳过**（仅 nobias 路径）：`skip_x_pad = (pad==0 and c_in==c_in_pad) and not has_bias`。
   prelude 跳过 `x_pad` 拷贝，Cube 的 B 块直接读 `x`。

### R1 编译陷阱（新实证，三处 bishengir SIGSEGV）

| 陷阱 | 现象 | 结论 |
|---|---|---|
| **Cube/Vector scope 级 hoist kernel_id 标量** | 把 `by = kernel_id % gy`、`sub = kernel_id // gy` hoist 出 `T.serial` 循环 → whisper-bias/encodec-bias 确定性 SIGSEGV；放回循环内即 PASS | bishengir 对 scope 级 loop-invariant 标量退化 |
| **bias 路径 Cube 直读 x（恒等跳过）** | `identity_x` 应用到 bias(MixCV) 路径时，encodec-deep-bias 确定性 SIGSEGV（whisper-bias 却 PASS） | 仅 nobias 路径启用 `skip_x_pad`，bias 路径保留 x_pad 拷贝 |
| **bf16 + bias 编译** | whisper/encodec bf16-bias 编译确定性 SIGSEGV（基线同崩，本会话多次复测；旧会话曾成功） | 既有 flaky（CG-2026-0003 家族），bf16-bias 最终测量经重试获得 |

### R1 候选 vs current best 对比（Task Duration(us)，msprof op）

| candidate | workload | baseline_us | candidate_us | Δ | l0 |
|---|---:|---:|---:|---|
| r2_weight_shard | whisper-conv1-float16 | 5626.31 | 369.93 | **−93.4%** | ✅ |
| r2_weight_shard | whisper-conv1-bfloat16 | 5598.71 | 369.23 | **−93.4%** | ✅ |
| r2_weight_shard | whisper-conv1-bias-float16 | 5518.67 | 367.63 | **−93.3%** | ✅ |
| r2_weight_shard | whisper-conv1-bias-bfloat16 | 5584.41 | 362.41 | **−93.5%** | ✅ |
| r2_weight_shard | encodec-deep-float16 | 1915.94 | 521.85 | **−72.8%** | ✅ |
| r2_weight_shard | encodec-deep-bfloat16 | 1898.54 | 520.19 | **−72.6%** | ✅ |
| r2_weight_shard | encodec-deep-bias-float16 | 1893.52 | 538.91 | **−71.5%** | ✅ |
| r2_weight_shard | encodec-deep-bias-bfloat16 | 1857.40 | 537.95 | **−71.0%** | ✅ |

> whisper（gy=20=num_kernels，share=1）权重重排 20×→1×，15× 加速；
> encodec-deep（gy=4，share=5）权重重排 20×→5×（5 核共享同一 co 块，无跨核同步下无法更细），
> 恒等 x_pad 跳过再砍掉 128×600 的 20× 冗余拷贝，~3.6× 加速。剩余时延为 5× 冗余权重重排 +
> 真实 GEMM + bias 中继。

## 3. Round 2：合并复测与收束

R1 候选即最终版（单点改动一次到位，无第二候选）。合并版复测本 kernel 全部 tune workload
（见 Final 表）与 smoke 精度（`smoke-k3-s1` fp16/bf16 + `smoke-k3-s1-bias` fp16/bf16 全 PASS）。

L0/L1 精度回归：8 个 tune workload（whisper ×4、encodec-deep ×4）全 PASS（`max_abs ≤ 6.25e-2`，
与基线一致）；L0 其余非 tune 用例 `nondiv-nobias`、`batch3-bias`、`taps2-nobias` 全 PASS。
**2 个 L0 用例（`dilation-pad-bias`、`taps2-bias`）命中既有 bishengir SIGSEGV（CG-2026-0003 家族，
基线同崩、逐 case 复测 3/3 一致）——属工具链既有缺口，非本轮改动引入**。bias(MixCV) 路径的
taps=2 / pad+dilation 编译在本工具链下非确定性崩溃（上一会话亦记录同族现象）。

## 4. 通用 `conv1d`（wav2vec2-layer1 / encodec-init）—— 第二轮（P1 接力）

> 本轮只动通用 `_conv1d_kernel.py`；`_conv1d_unit_stride_kernel.py`（whisper / encodec-deep）保持第一轮结果不变。
> 分派复核：`encodec-init`（`c_in=1` → `c_in < 16`）与 `wav2vec2-layer1`（`stride=5`）都命中 `conv1d` 通用 kernel。

### 4.1 Baseline 与病根（msprof op，20 AIC，2026-10-18）

| workload | baseline µs | 关键读数（per-block PipeUtilization） |
|---|---:|---|
| encodec-init-float16 | 274.17 | `aic_mte2_time≈221µs`、`aic_fixpipe_time≈42µs`、`aic_cube_time≈3µs`；Vector 侧 `aiv_mte3≈100µs`（`x_pad` 拷贝） |
| encodec-init-bfloat16 | 253.51 | 同上 |
| encodec-init-bias-float16 | 276.55 | 同上 + ws 中继（bn=128） |
| encodec-init-bias-bfloat16 | 255.65 | 同上 |
| wav2vec2-layer1-float16 | 610.31 | `aiv_mte3≈353µs`（polyphase 写 `c_in_pad=16` 个 chunk，15 个是零通道）、`aic_mte2≈99µs`、`aic_fixpipe≈16µs` |
| wav2vec2-layer1-bfloat16 | 607.25 | 同上 |
| wav2vec2-layer1-bias-float16 | 621.49 | 同上 |
| wav2vec2-layer1-bias-bfloat16 | 603.91 | 同上 |

病根（与 unit-stride 同源，但形态不同）：

1. **prelude 逐核冗余**：`_prelude` 的权重重排（全部 `n_co` 个 co 块）+ `x_pad` 全量拷贝在 20 核上各做一遍；
   wav2vec2 更重：polyphase 按 `c_in_pad=16` 个 chunk，其中 15 个是**零通道**（每核 512KB 无效写）。
2. **逐 tap 独立 gemm + 逐 task 重载**：implicit GEMM 把 `kernel_l` 个 tap 拆成 `kernel_l` 次独立 gemm，
   每次重载 A（权重）与 B（im2col 输入）；`c_in=1` 时 K 还被撑到 `c_in_pad=16`，**16× 空算**。
3. **N tile 太小**：wrapper 默认 `block_n=128`，两个 shape 的每行输出只有 256B，fixpipe/Vector 写 GM 的
   碎片化代价被放大（实测 `aic_fixpipe_active_bw` 仅 3.6–15 GB/s）。

### 4.2 Round G1（结构）：taps-in-K + 权重直读 + 恒等 x_pad 跳过 + 单通道 polyphase

`c_in == 1 && kernel_l <= 16 && pad==0 && l_pad == l_in` 时走「taps-in-K」路径（编译期常量分支，仍在同一
`@T.prim_func` 内，无新 factory / wrapper 分派）：

1. **GEMM 的 K 轴 = 卷积窗口本身**（`K = kernel_l`），一次 gemm 覆盖整窗（原为 `kernel_l` 次 K=16 gemm），
   且 `c_in=1` 时不再有 16× 通道空算。K 非 16 倍数被 `T.gemm` 接受（实测 K=7/8/10/11 全部 PASS，
   `probe_api.py`）。
2. **A 直接读原生 `weight[c_o, 0, k_w]`**（行=co、列=tap，连续），整块权重重排 prelude（GM→UB→两次
   `T.transpose`→GM）完全消失；A tile 只在 co 块变化时重载，不再逐 tap 重载。
3. **恒等 `x_pad` 跳过**：B 的每个 tap 行直接从 `x`（stride=1）或 polyphase `xdec`（stride>1）取，
   `x_pad` 的 20× 冗余拷贝消失（此前 bias 路径不敢做，见第一轮 TRAP；本轮因 B 源改为「逐 tap 一行」而非
   「Cube 直读整块」而未复现该 SIGSEGV）。
4. **polyphase 只算真实通道**：`c_in=1` → 1 个 chunk（原 `c_in_pad=16` 个），wav2vec2 的 `aiv_mte3` 从
   ~353µs 降到 ~2µs 量级。
5. **核内映射**：每核一段连续任务（`cid_lo/cid_hi`），无跨核依赖（每核只读自己取的列）。

### 4.3 Round G2（几何/参数）：tapk_bm = ceil16(c_out) + bn → L0C 预算 + 连续 chunk 映射

单一主要优化点：**放大每 tile 的连续写长度、按 L0C 预算取最大 N tile**。
`tapk_bm = min(block_m, ceil16(c_out))`（encodec-init 32、wav2vec2 64），`tapk_bn` 取 L0C 128KB 预算内最大值
（受实测约束 `bn ≤ 512`），并按 `gy*gx >= num_kernels` 回退；任务映射改为「每核连续 cid 区间」，
使 `gx < num_kernels` 的 wav2vec2 也能吃满 20 核。

实测几何：encodec-init `bm=32,bn=512`（47 tile / 20 核）；wav2vec2 `bm=64,bn=512`（56 tile / 20 核）；
bias 路径仍 `bn=128`（ws 中继宽度 = wrapper 的 `block_n`，见 §4.5）。

### 4.4 候选 vs current best 对比（Task Duration µs，msprof op）

| candidate | workload | best_before µs | candidate µs | Δ | L0 |
|---|---|---:|---:|---:|---|
| G1 taps_in_k | encodec-init-float16 | 274.17 | 65.62 | −76.1% | ✅ |
| G1 taps_in_k | encodec-init-bfloat16 | 253.51 | 74.06 | −70.8% | ✅ |
| G1 taps_in_k | encodec-init-bias-float16 | 276.55 | 240.84 | −12.9% | ✅ |
| G1 taps_in_k | encodec-init-bias-bfloat16 | 255.65 | 253.27 | −0.9%（噪声内） | ✅ |
| G1 taps_in_k | wav2vec2-layer1-float16 | 610.31 | 115.96 | −81.0% | ✅ |
| G1 taps_in_k | wav2vec2-layer1-bfloat16 | 607.25 | 106.08 | −82.5% | ✅ |
| G1 taps_in_k | wav2vec2-layer1-bias-float16 | 621.49 | 166.04 | −73.3% | ✅ |
| G1 taps_in_k | wav2vec2-layer1-bias-bfloat16 | 603.91 | 167.40 | −72.3% | ✅ |
| G2 tile_geom | encodec-init-float16 | 65.62 | 65.64 | +0.03%（平） | ✅ |
| G2 tile_geom | encodec-init-bfloat16 | 74.06 | 73.96 | −0.1%（平） | ✅ |
| G2 tile_geom | encodec-init-bias-float16 | 240.84 | 240.22 | −0.3%（平） | ✅ |
| G2 tile_geom | encodec-init-bias-bfloat16 | 253.27 | 255.49 | +0.9%（噪声） | ✅ |
| G2 tile_geom | wav2vec2-layer1-float16 | 115.96 | **34.72** | **−70.1%** | ✅ |
| G2 tile_geom | wav2vec2-layer1-bfloat16 | 106.08 | **35.78** | **−66.3%** | ✅ |
| G2 tile_geom | wav2vec2-layer1-bias-float16 | 166.04 | **74.84** | **−54.9%** | ✅ |
| G2 tile_geom | wav2vec2-layer1-bias-bfloat16 | 167.40 | **68.92** | **−58.8%** | ✅ |

> G2 对 wav2vec2 的 3.3× 增益来自「每行输出从 256B → 1KB」：`aic_fixpipe_active_bw` 由 3.64 → ~15 GB/s，
> 同时每 tile 的 B 拷贝从 10 次/task 摊薄到 10 次/512 列。
> encodec-init 在 G1 已是 `bn=512`（`gx=47 ≥ 20`），故 G2 的 tile 几何对其无改（0%），保留 G2 版本
> （wav2vec2 收益更大，且映射统一）。

### 4.5 Round G3（bias-in-K：把 bias 折进 K 轴）→ blocked（工具链缺口）

目标：bias 路径也吃掉大 tile（当前 encodec-init-bias 仍有 240µs，是唯一未量级改善的 shape 组）。
方案：K 轴加一行 —— `A[:, k] = bias`、`B[k, :] = 1.0`，`out = W·x + bias` 一次 gemm 完成，
Cube→ws→Vector 的中继（1.5MB 读 + 1.5MB 写 + 每 tile 握手）整体消失。

实测结论（`perf_opt/logs/r3-prec.log`，逐项 bisect）：

| 变体 | 内容 | 结果 |
|---|---|---|
| w2 | K+1 + ones 行，A 读原生 weight | 编译 PASS（精度 FAIL，符合预期：A 末列为垃圾） |
| w5 | 只做 `weight → ub → w_flat[0, co, 0:k]` | 编译 PASS |
| w6 | 只做 bias 列（`bias[co] → ub[:,0:1] → w_flat[0, co, k:k+1]`） | **编译 FAIL** |
| w7/w8/w9 | bias 列改为静态 extent / UB 内拼整块后单次落 GM / 全静态 | **编译 FAIL** |

报错固定为 `[ERROR] Unknown core type: llvm.func @malloc(i64) -> !llvm.ptr`（bishengir codegen），
与 extent 是否静态无关；**只要在 Vector prelude 里把 1D `bias` 切片搬进 UB 并落到带列偏移的 GM 区**
就触发（同形态在 Stage-3 的 **epilogue** 里是合法的，位置敏感）。
判定为 CG-2026-0003 家族的 codegen 缺口（新形态），本机工具链下不可绕；**bias-in-K 分支放弃**，
bias 路径保留 G2 的中继形态（wav2vec2-bias 已 8.4–8.8×，encodec-init-bias 1.15×/1.00×）。

### 4.6 合并复测与收束

G2 即最终版（G1→G2 是同一路径的几何参数演进，均在 `@T.prim_func` 内，无 Python 层分派）。
合并版复测本 kernel 全部 8 个 tune workload（Final 表）+ smoke 精度 + 全量 L0/L1/L2/boundary。

### 4.7 记录修复与 gate 4 现状（透明留痕）

本轮对 `perf_records.jsonl` 做了两处**格式修复**（无任何测量值被改动），审计副本
`perf_opt/logs/perf_records_pre_repair_20261018.jsonl`：

1. 本轮 8 条 round 记录写入时 `phase` 误用 `round`，不在契约枚举（`baseline/candidate/merged/final`）内
   → 修正为 `candidate`（时延、哈希、路径均不变）。
2. 上一轮为通用 kernel 记的 8 条 `phase=final` 且 `candidate_id=baseline` 的记录（当时「final = 未调优基线」）
   已被本轮的真实 final 取代 → 删除并写入本轮 final（同一 `(kernel_id, workload_id)` 只允许一条 final）。
   **被取代的原值**（保留备查）：wav2vec2-layer1 613.7720/582.4120、wav2vec2-layer1-bias 589.9520/595.4720、
   encodec-init 283.2060/268.1450、encodec-init-bias 256.2650/268.7450 µs（fp16/bf16，上一会话口径）。

最终工件：`perf_opt/Conv1dFwdOp.py` = `perf_opt/_conv1d_kernel.py` 的**逐字节副本**
（sha256 `ba19f810…`），final 记录指向它。

**gate 4 现状**（`statectl gate 4 --dir ...`）：除
`S4-PERF-RECORDS-RECON「final 未测最终 perf_opt 文件」` 外全绿。该失败是**多 kernel 包的单文件契约冲突**：
卷积包有 `conv1d_unit_stride` 与 `conv1d` 两个 kernel，各有一组 final 行，但门禁只接受**一个**
`kernel_opt_py_path`。选 `perf_opt/Conv1dFwdOp.py` → 8 条 unit_stride final 行报错；选
`perf_opt/Conv1dBiasFwdOp.py` → 本轮 8 条 conv1d final 行报错（实测两向各 8 条，见交付摘要）。
本轮按「当前算子（Conv1dFwdOp）」选前者，并把该冲突作为 R 类价值点上报（上一轮已登记同类 R 点）。

### 4.8 端到端 report 与 A/B（收束证据）

| run_id | 算子 | correctness | bench |
|---|---|---|---|
| `20261018_174226_619703_Conv1dFwdOp` | Conv1dFwdOp | 25/28（2 例 bishengir SIGSEGV：`smoke-padding-wider-than-row-fp16`、`full-groups2-k3-fp16`） | 未执行 |
| `20261018_180016_718940_Conv1dFwdOp` | Conv1dFwdOp | **28/28 通过** | 15/17（仅 `encodec-deep-bias` fp16/bf16 编译 SIGSEGV） |
| `20261018_174555_102276_Conv1dBiasFwdOp` | Conv1dBiasFwdOp | **28/28 通过** | 7/9（`encodec-deep-bias` fp16/bf16） |
| `20261018_180716_376489_Conv1dBiasFwdOp` | Conv1dBiasFwdOp | **28/28 通过** | 8/9（`encodec-deep-bias` bf16） |

- **wav2vec2 编译可绕开**：上一轮 `wav2vec2-layer1-fp16` bench 必崩（CG-2026-0003），本轮 taps-in-K 的 IR
  形态两次 run 均正常采集 ⇒ **绕开成功**。Op 层 A/B（`perf_opt/ab_wav2vec2.py`）：
  `AB perf_opt PASS max_abs=7.76e-03`、`AB baseline PASS max_abs=7.75e-03`（两条实现均正确）。
- **`smoke-padding-wider-than-row` 与 `full-groups2-k3` 的失败为偶发**：前者走通用 kernel 的**未改动**
  Stage-3 分支（`pad>0` 不满足 tapk 谓词），后者走 `_conv1d_group_kernel.py`（未改动）。单独复跑该 shape：
  perf_opt 2/2 PASS、baseline 2/2 PASS（`logs/r2-final-prec/prec-pad-wider.log` 等）。
- **残余 bench 失败归属**：`encodec-deep-bias` 走 `conv1d_unit_stride`（本轮未改动，mtime 07:26 < 本会话开始
  16:30），单独重跑 `pytest benchmarks/ops/bench_conv1d.py -k encodec-deep-bias` → **2 passed** ⇒ 偶发
  bishengir SIGSEGV（CG-2026-0003 族），非本轮引入。

## Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---:|---:|---|
| conv1d_unit_stride | whisper-conv1-float16 | 5626.3130 | 369.9274 | r2_weight_shard |
| conv1d_unit_stride | whisper-conv1-bfloat16 | 5598.7114 | 369.2274 | r2_weight_shard |
| conv1d_unit_stride | whisper-conv1-bias-float16 | 5518.6704 | 367.6273 | r2_weight_shard |
| conv1d_unit_stride | whisper-conv1-bias-bfloat16 | 5584.4120 | 362.4073 | r2_weight_shard |
| conv1d_unit_stride | encodec-deep-float16 | 1915.9384 | 521.8505 | r2_weight_shard |
| conv1d_unit_stride | encodec-deep-bfloat16 | 1898.5380 | 520.1904 | r2_weight_shard |
| conv1d_unit_stride | encodec-deep-bias-float16 | 1893.5178 | 538.9108 | r2_weight_shard |
| conv1d_unit_stride | encodec-deep-bias-bfloat16 | 1857.3970 | 537.9507 | r2_weight_shard |
| conv1d | wav2vec2-layer1-float16 | 610.3122 | 34.4407 | r2_tile_geom |
| conv1d | wav2vec2-layer1-bfloat16 | 607.3522 | 35.3407 | r2_tile_geom |
| conv1d | wav2vec2-layer1-bias-float16 | 621.4924 | 74.8015 | r2_tile_geom |
| conv1d | wav2vec2-layer1-bias-bfloat16 | 603.9120 | 68.2014 | r2_tile_geom |
| conv1d | encodec-init-float16 | 274.1655 | 65.9613 | r2_tile_geom |
| conv1d | encodec-init-bfloat16 | 253.5051 | 74.1015 | r2_tile_geom |
| conv1d | encodec-init-bias-float16 | 276.5455 | 239.6848 | r2_tile_geom |
| conv1d | encodec-init-bias-bfloat16 | 255.6451 | 213.8643 | r2_tile_geom |

> 口径：whisper/encodec-deep 的 baseline 为第一轮会话 msprof 直采（bf16-bias 两 case 因 bishengir flaky，
> 沿用上上会话基线 5584.41/1857.40，kernel 未改、数值有效）；`conv1d` 8 行的 baseline/final 均为
> **本会话（2026-10-18）msprof op 直采**，且 final 全部来自同一最终候选 `r2_tile_geom`
> （`perf_opt/_conv1d_kernel.py`，sha256 `ba19f810…`），逐项记录见 `perf_records.jsonl` 的 `phase=final` 行。
> 两个 kernel 的 final 来自各自 kernel 的同一最终候选，未跨分支拼接。

## Final Summary

- **unit_stride**（whisper + encodec-deep，两个最重 workload 族）：
  - whisper 族：5626/5598/5519/5584 → **370/369/368/362 µs（~15×，SOL 0.12%→~1.8%）**；
  - encodec-deep 族：1916/1899/1894/1857 → **522/520/539/538 µs（~3.6×，SOL 0.07%→~0.25%）**；
  - 几何平均 **~7.6×**。
- **conv1d**（wav2vec2-layer1 + encodec-init，本会话第二轮；数值全部取自 `perf_records.jsonl` 的
  `phase=final` 行，同一最终候选 `r2_tile_geom`）：
  - wav2vec2 族：610.31/607.35 → **34.44/35.34 µs（17.72×/17.19×，SOL 0.31%→5.30–5.46%）**；
  - wav2vec2-bias 族：621.49/603.91 → **74.80/68.20 µs（8.31×/8.85×，SOL 0.31%→2.53–2.78%）**；
  - encodec-init 族：274.17/253.51 → **65.96/74.10 µs（4.16×/3.42×，SOL 0.33%→1.22–1.38%）**；
  - encodec-init-bias 族：276.55/255.65 → **239.68/213.86 µs（1.15×/1.20×）**——bias 中继宽度被 wrapper 的
    `block_n=128` 钳制，bias-in-K 改写命中 CG-2026-0028（codegen 缺口），见 §4.5；
  - 几何平均（8 个 shape）**≈ 5.07×**。
  - **测量双态提示**：`encodec-init-bias-bfloat16` 在 r2 轮测得 255.49µs、final 轮测得 213.86µs（同一二进制，
    差 16%）——该 shape 落在 bias 中继 + 极窄 N tile 的临界区，存在 run-state 双态（cp. BP_run_state_bimodality）。
    按纪律 **final 记录为准**，且该 shape 的收益按区间报（−0.1%～−16.3%），不单独宣称。
- 中止原因：**success**（两个 kernel 各自的最重族均达成量级收益，精度全绿；唯一无收益子族已归因到
  工具链缺口并登记 CG-2026-0028，属 blocked 而非「未做」）。

## Skill Retrospective

流程顺畅度：本轮在已有一轮诊断基础上直接定位 R1-B 真实 bug（`block_m/BM_W` 单位错位），
单点修复即达成 15×/3.6×；主要耗时在 bishengir flaky SIGSEGV 的隔离（三处陷阱）与 gate 4 工件对账。

新实测价值点（供 evolver 蒸馏，含证据三件套）：

1. **P：persistent 权重重排按 co 块切分**（vp_type=P）
   - evidence：`perf_opt/_conv1d_unit_stride_kernel.py` R2 改动 + Final 表 whisper 15×。
   - repro：待补 `repro/r2_weight_shard.py`（从 baseline 到切分的最小 diff）。
   - toolchain_stamp：CANN 9.0.0 / tilelang 0.1.2 dev root。
2. **D：bishengir SIGSEGV 三陷阱**（vp_type=D）
   - evidence：`perf_opt/logs/`（one_case/quick_l0 多次复测）+ 本日志 §2 表。
   - repro：`repro/TRAP-scope-hoist-kernelid.py`（hoist `by=kernel_id%gy` 出 T.serial 循环即崩）。
   - toolchain_stamp：同上。
3. **D：bf16+bias MixCV 编译 flaky**（vp_type=D，CG-2026-0003 家族补充证据）。
4. **R：gate 4 多 kernel op 的 final artifact 单文件契约**（vp_type=R）——5-kernel 卷积包被
   `kernel_opt_py_path` 收束为单一 `perf_opt/{op}.py`，多 kernel 的 final 记录须同指向该文件，
   语义上依赖 opt_log 说明「未触及」；建议 conductor 为多 kernel op 放宽或明确该契约。

### 第二轮（通用 conv1d，本会话）追加

流程顺畅度：本轮是「同一 kernel 包、不同变体」的接力，病根与第一轮同源但形态不同（不是 prelude 冗余，
而是 **c_in=1 被 16 倍填充 + 逐 tap 独立 gemm + N tile 过小**）；主要耗时在 ① 测量口径纠错
（kernel-name 过滤失效导致 median 混入 torch 侧 kernel，见下方 D 类第 2 条）与 ② bias-in-K 的 codegen 故障隔离。

新实测价值点（供 evolver 蒸馏，含证据三件套）：

1. **P：taps-in-K —— 单通道卷积把窗口折进 GEMM 的 K 轴**（vp_type=P）
   - evidence：`perf_opt/candidates/r2_tile.py` + Final 表（wav2vec2 610.31→34.72 µs，17.6×；
     encodec-init 274.17→65.64 µs，4.2×）；profile `aic_mte2_time` 221→60µs、`aiv_mte3` 353→~2µs。
   - repro：`perf_opt/probe_api.py`（`gemm K=7/10 kcl=7/10 PASS`、`taprow K=7/8/10/11 PASS`）+ r1→r2 delta。
   - toolchain_stamp：tilelang 0.1.2 dev root / CANN 9.0.0 (bishengir 1.1.0) / Ascend910B3 / 2026-10-18。
   - 触发条件：`c_in == 1 && kernel_l <= 16 && pad == 0 && l_pad == l_in`（`K = kernel_l` 一次 gemm，
     A 直读原生 weight，B 逐 tap 取一行）。
2. **D：N tile 宽度决定 fixpipe/Vector 写 GM 效率（线性）**（vp_type=D）
   - evidence：`aic_fixpipe_active_bw` 实测 3.64 GB/s（每行 256B，`bn=128`）→ 15.4 GB/s（每行 1KB，`bn=512`）；
     同一 wav2vec2 shape 115.96→34.72 µs。
   - repro：`perf_opt/{measure.py,prof_summary.py}` + `profiles/{r1-tapk,r2-tile}/wav2vec2-layer1-float16`。
   - 推论：Conv1d 族 SOL 上界由输出写通道决定，不是 tiling；追 SOL 前先用 `aic_fixpipe_active_bw` 定量。
3. **D：msprof `--kernel-name` 过滤失效会静默污染 median**（vp_type=D）
   - evidence：首版候选把 prim_func 命名为 `_conv1d_tapk_main`，`--kernel-name=_conv1d_main` 未命中 →
     measure 退回「无过滤」路径，median 混入 2.14µs 的 torch setup kernel（读数 73.98µs 全错）。
   - 对策（已落地）：`perf_opt/measure.py` 解析时按 `Op Name` 列二次过滤并在无命中时显式告警。
   - 推论：自定义 kernel 名必须沿用基线命名（`_conv1d_{,bias_}main`），否则跨轮数据不可比。
4. **D：L0C 容量与 N tile 的硬上限**（vp_type=D，同族 VP-2026-0161 补充）
   - `bm=32,bn=1024`（L0C 恰好 128KB）→ bishengir SIGSEGV；`bm=64,bn=512` / `bm=32,bn=512` 均 PASS
     ⇒ 本工具链安全上限 `bn ≤ 512`，与 `bm` 无关。
5. **C：CG-2026-0028（新缺口）——Vector prelude 里「1D GM 切片 → 带列偏移 GM 子区域」写必崩**
   - evidence：§4.5 的 w2/w5/w6/w7/w8/w9 六变体 bisect（同形态在 epilogue 位置合法）。
   - 影响：bias-in-K 折叠不可表达，bias 路径只能保留 Cube→ws→Vector 中继。

## 参考实现对照（plateau 前强制）

同族先例 `examples/conv1d/_conv1d_unit_stride_kernel/`（Stage-3 迁移源）与
`pattern-library` attention/ssd persistent kernel 均确认：**persistent 内核的 prelude 逐核
重复是首要病根**，切分/单核化是标准解法，与本轮结论一致，无结构分歧。

---
*本日志的最终时延全部来自同一最终候选的 `perf_records.jsonl` 记录，未拼接不同分支。*
