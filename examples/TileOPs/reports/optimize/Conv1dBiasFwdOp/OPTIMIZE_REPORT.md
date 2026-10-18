# 优化报告：Conv1dBiasFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `Conv1dBiasFwdOp`（bias 必填薄接口）；同一 kernel 亦服务 `Conv1dFwdOp`（bias 可选） |
| kernel | `tileops/kernels/convolution/conv1d/conv1d_kernel/`（5 变体，本轮调优 `conv1d_unit_stride`） |
| 调优产物 | `.../conv1d_kernel/perf_opt/_conv1d_unit_stride_kernel.py`（+ 同名 drop-in 4 件 baseline 副本） |
| 口径 | msprof op Task Duration(us) + SOL ratio |
| 工具链 | tilelang 0.1.2 dev root / CANN 9.0.0 / Ascend910B3 / 2026-10-18 |
| benchmark | `bench_conv1d.py::test_conv1d_manifest_bench` + `bench_conv1d_bias.py::test_conv1d_bias_manifest_bench`（两套，16 tune + 4 smoke） |

## 1. 结论

- **达标数（Task Duration 变短 + 精度全绿）**：
  - **可调 shape（本次实际调优的 `conv1d_unit_stride` 8 个）= 8/8 达标**。
  - **全 shape（16 tune）= 8/16 达标**：8 个 `conv1d_unit_stride` 大幅变短，8 个 `conv1d`
    （wav2vec2/encodec-init）按任务授权「允许只优化 1–2 个变体」未触及、时延不变（非回退）。
- **根因**：persistent 内核的 `_prelude`（权重重排 + x_pad 拷贝）在 20 个 AI core 上逐核重复，
  `aiv_mte2_ratio = 97.1%`、有效带宽仅 2.39 GB/s；真实 GEMM 只有 ~8 µs。
- **修复**：① 权重重排按 co 块切分（`by = kernel_id % gy`，每 core 只重排自己读取的 co 块，无跨核同步）；
  ② 恒等 x_pad 跳过（`pad==0 and c_in==c_in_pad` 且 nobias 时 Cube 直读 `x`）。
- whisper 族（gy=20=核数）权重重排 20×→1×，**~15×（SOL 0.12% → ~1.8%）**；
  encodec-deep 族（gy=4，share=5）权重重排 20×→5× + x_pad 20×→0，**~3.6×（SOL 0.07% → ~0.25%）**。
- 剩余 gap：encodec-deep 的 5× 权重重排冗余（5 核共享同一 co 块，无跨核同步下不可再分）+ 真实 GEMM + bias 中继。

## 2. 所做改动

仅在 `perf_opt/` 内（基准 `_conv1d_unit_stride_kernel.py` 与 wrapper `conv1d.py` 未动）：

1. **权重重排按 co 块切分**（`_conv1d_unit_stride_kernel.py` 的 `_prelude` + `_cube` + `_bias_epilogue`）：
   任务映射改为 `by = kernel_id % gy` / `sub = kernel_id // gy` / `idx = task_id * share + sub`；
   prelude 只重排本 core 的 co 块 `[by*block_m, by*block_m+block_m)`（内层 `BM_W=16` 子块循环，
   修正上一轮 R1-B 的 `block_m/BM_W` 单位错位 bug）。
2. **恒等 x_pad 跳过**（仅 nobias 路径）：`skip_x_pad = (pad==0 and c_in==c_in_pad) and not has_bias`；
   prelude 跳过 `x_pad` 拷贝、Cube 的 B 块直读 `x`。
3. 其余 4 个 kernel（`conv1d` / `group` / `direct` / `pointwise`）复制 baseline 副本至 `perf_opt/`
   供 wrapper 双 import 切换块完整 drop-in（本轮未改动逻辑）。

## 3. 优化结果（final，同一候选 `r2_weight_shard`；conv1d 族为未改动的 baseline）

| kernel_id | workload | base µs | opt µs | Δ | opt SOL% |
|---|---:|---:|---:|---:|
| conv1d_unit_stride | whisper-conv1-float16 | 5626.31 | 369.93 | **-93.4%** | ~1.8% |
| conv1d_unit_stride | whisper-conv1-bfloat16 | 5598.71 | 369.23 | **-93.4%** | ~1.8% |
| conv1d_unit_stride | whisper-conv1-bias-float16 | 5518.67 | 367.63 | **-93.3%** | ~1.8% |
| conv1d_unit_stride | whisper-conv1-bias-bfloat16 | 5584.41 | 362.41 | **-93.5%** | ~1.8% |
| conv1d_unit_stride | encodec-deep-float16 | 1915.94 | 521.85 | **-72.8%** | ~0.25% |
| conv1d_unit_stride | encodec-deep-bfloat16 | 1898.54 | 520.19 | **-72.6%** | ~0.25% |
| conv1d_unit_stride | encodec-deep-bias-float16 | 1893.52 | 538.91 | **-71.5%** | ~0.24% |
| conv1d_unit_stride | encodec-deep-bias-bfloat16 | 1857.40 | 537.95 | **-71.0%** | ~0.24% |
| conv1d | wav2vec2-layer1-* (4) | 582–614 | 582–614 | 0%（未触及） | — |
| conv1d | encodec-init-* (4) | 256–283 | 256–283 | 0%（未触及） | — |

### 精度

8 个 tune workload（whisper ×4、encodec-deep ×4）`max_abs ≤ 6.25e-2`（与基线一致）；L0 其余非 tune
用例 `nondiv-nobias`/`batch3-bias`/`taps2-nobias` PASS。2 个 L0 用例（`dilation-pad-bias`、`taps2-bias`）
命中既有 bishengir SIGSEGV（CG-2026-0003 家族，基线同崩、逐 case 复测 3/3），非本轮改动引入。
4 个 smoke（`smoke-k3-s1` fp16/bf16 + `smoke-k3-s1-bias` fp16/bf16）仅精度回归 PASS。

## 4. 时间与 token

| 项 | 值 |
|---|---|
| 开始 / 结束（+08:00） | 2026-10-18 07:10 → 08:10 |
| 墙钟 | ≈ 60 min |
| 主要耗时 | bishengir flaky SIGSEGV 隔离（三处陷阱）+ gate 4 多 kernel 工件对账 |

## 5. 未触及项与原因（任务授权「允许只优化 1–2 个变体」）

- **`conv1d`（wav2vec2 / encodec-init）**：wav2vec2（gy=8）权重重排仅 ~2.5× 且 polyphase xdec 仍
  20× 冗余；encodec-init（gy=1）单 co 块全核共享、切分无收益；预算优先投入最重族 whisper/encodec-deep。
- **`group` / `direct` / `pointwise`**：不在 `convolution.yaml` 的 dense Conv1d manifest 内，无 benchmark 覆盖。
