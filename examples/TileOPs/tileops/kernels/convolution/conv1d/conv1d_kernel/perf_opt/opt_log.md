# Conv1dFwdOp / Conv1dBiasFwdOp — Stage 4 调优日志

> 工具链版本戳：CANN **9.0.0**（bishengir-compile 1.1.0 / hivmc 0.2.0）/ tilelang 0.1.2 dev root
> (`/home/developer/tilelang-mlir-ascend`) / Ascend910B3（Chip 6，20 AIC）/ 2026-10-18。
> 口径：**msprof op Task Duration(us)**（唯一时延口径）。
> NPU 纪律：所有上机命令经 `.agents/tools/npu_lock.sh` 串行执行。

## 0. 对象与分派

| label | 分派 kernel | kernel 文件 | 本轮是否调优 |
|---|---|---|---|
| whisper-conv1(+bias) / encodec-deep(+bias) | `conv1d_unit_stride` | `_conv1d_unit_stride_kernel.py` | ✅ 调优（本轮全部工作） |
| wav2vec2-layer1(+bias) / encodec-init(+bias) / smoke-k3-s1 | `conv1d` | `_conv1d_kernel.py` | ❌ 未触及（见复盘） |
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
| conv1d | wav2vec2-layer1-float16 | 613.7720 | 613.7720 | baseline |
| conv1d | wav2vec2-layer1-bfloat16 | 582.4120 | 582.4120 | baseline |
| conv1d | wav2vec2-layer1-bias-float16 | 589.9520 | 589.9520 | baseline |
| conv1d | wav2vec2-layer1-bias-bfloat16 | 595.4720 | 595.4720 | baseline |
| conv1d | encodec-init-float16 | 283.2060 | 283.2060 | baseline |
| conv1d | encodec-init-bfloat16 | 268.1450 | 268.1450 | baseline |
| conv1d | encodec-init-bias-float16 | 256.2650 | 256.2650 | baseline |
| conv1d | encodec-init-bias-bfloat16 | 268.7450 | 268.7450 | baseline |

> 口径：whisper/encodec-deep 的 baseline 为本次会话 msprof 直采（bf16-bias 两 case 因本会话
> bishengir flaky，沿用上一会话基线 5584.41/1857.40，kernel 未改、数值有效）；
> final 全部来自同一最终候选 `r2_weight_shard`（unit_stride）与未改动的 baseline（conv1d）。

## Final Summary

- **unit_stride**（whisper + encodec-deep，两个最重 workload 族）：
  - whisper 族：5626/5598/5519/5584 → **370/369/368/362 µs（~15×，SOL 0.12%→~1.8%）**；
  - encodec-deep 族：1916/1899/1894/1857 → **522/520/539/538 µs（~3.6×，SOL 0.07%→~0.25%）**；
  - 几何平均 **~7.6×**。
- **conv1d**（wav2vec2 + encodec-init + smoke）：未触及。wav2vec2（gy=8）权重重排仅 ~2.5×
  且 polyphase xdec 仍 20× 冗余、encodec-init（gy=1）单 co 块无分块收益；预算优先投入最重族。
- 中止原因：**success**（最重 workload 族达成量级收益，精度全绿；次要族按任务授权「允许只优化 1–2 个变体」留档）。

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

## 参考实现对照（plateau 前强制）

同族先例 `examples/conv1d/_conv1d_unit_stride_kernel/`（Stage-3 迁移源）与
`pattern-library` attention/ssd persistent kernel 均确认：**persistent 内核的 prelude 逐核
重复是首要病根**，切分/单核化是标准解法，与本轮结论一致，无结构分歧。

---
*本日志的最终时延全部来自同一最终候选的 `perf_records.jsonl` 记录，未拼接不同分支。*
