# 优化报告：Conv3dBiasFwdOp（Stage-4 optimize）

## 0. 身份

| 项 | 值 |
|---|---|
| 算子 | `Conv3dBiasFwdOp`；dense NCDHW implicit GEMM（Cube Expert / MixCV bias 中继） |
| kernel | `tileops/kernels/convolution/conv3d/conv3d_kernel/` |
| 调优产物 | `.../conv3d_kernel/perf_opt/Conv3dBiasFwdOp.py`（R1 tap 打包 + 权重 L1 常驻，带编译期形状级回退） |
| 口径 | msprof op Task Duration(us)（唯一 kernel 时延口径） |
| 工具链 | tilelang 0.1.2 dev root / CANN 9.0.0 / Ascend910B3（Chip 6，20 AIC）/ 2026-10-18 |
| benchmark | `benchmarks/ops/bench_conv3d_bias.py::test_conv3d_bias_manifest_bench`（3 tune + 2 smoke） |

## 1. 结论

- **达标数**：可调 shape（3 tune）**1/3 达成 ≥3% Task Duration 下降**（r3d-stem 1.99×）；unet-encoder-bias、
  video-down-bias 无有效增益（见 §3 归因），并**回退 baseline 等效实现**（见 §2 回退）。
- **r3d-stem-bias fp16 6161.50 → 3090.14 µs（1.99×，SOL 0.25%→~0.51%）**：病根是 Cube 循环内逐 task
  权重重载（48384 次）+ 27-tap im2col 输入重复读；R1 把权重按 co 块分核、在 task 循环前一次性载入 L1，
  `aic_mte2_time` 6050→3042µs。
- **unet-encoder-bias bf16 6422.53 → 6449.49 µs（+0.4%，噪声内）**：weight-resident 对该 `c_in_pad=32`
  shape 未生效（编译器仍逐 task 从 L2 重载 110KB 常驻权重），剩余瓶颈是 27-tap 输入重复读（byte-bound）→ **回退 baseline**。
- **video-down-bias fp16 87.00 → 86.42 µs（−0.7%，噪声内）**：本就很快（stride-2 → 224 逻辑任务）→ **回退 baseline**。
- 剩余 gap 是 implicit-GEMM 的 27-tap im2col 输入重复读 + 小通道 16-fractal 填充（c_in=3→16 是 5.3×
  FLOP/访存浪费），在当前工具链（无切片 gemm、L1→L1 拷贝与 MTE2 共享 L1 端口）下参数/tiling 级不可解；
  不满足 `[DESIGN_LIMIT]` 触发条件（结构加速估计 <2x），故不产出 `perf_feedback.md`。

## 2. 所做改动

仅在 `perf_opt/` 内（基准与 wrapper 未动）：

1. **tap 打包**：`for tap in serial(KV)` 折叠进 gemm K 轴，每 co 块一次 gemm，`K = KV*c_in_pad`
   （A/B 块沿 K 轴在 `tap*c_in_pad` 处连续打包；K 扩展 = 打包宽度的单源不变量）。
2. **权重 L1 常驻 + 固定 co 块分核**：`by = kernel_id % gy` 固定每核到一输出通道块（`gy <= num_kernels`），
   该块权重 `w_flat[:, by*block_m:..., :]` 在 task 循环前载入 L1 一次、被该核所有空间 task 复用。
3. **B 块 N 维度 `n_max = ceil16(min(block_n, out_w))`**（否则 video-down `k_full=1728` 使
   `l1_a+l1_b=663KB > 512KB` 触发 bishengir SIGSEGV）。
4. **编译期形状级回退（回归修复）**：`use_r1 = stride1 and n_kc==1 and gy<=num_kernels and c_in_pad==16
   and l1_total<=512KB`。不满足则走 baseline 等效实现（逐 tap gemm + 逐 task 权重重载），保证结果正确、
   性能不劣于 baseline。

### adopted vs fallback

| shape | 路径 | 说明 |
|---|---|---|
| r3d-stem-bias（c_in=3→16, s=1） | **R1（保留）** | 1.99× 收益 |
| smoke-k3-s1-bias（c_in=8→16, s=1） | **R1（保留）** | 小通道常驻受益 |
| video-down-bias（c_in=64, s=2） | baseline 回退 | stride-2 多相，本就快 |
| unet-encoder-bias（c_in=32, s=1） | baseline 回退 | c_in_pad=32 常驻不生效 |
| stride2-fp16 / nondiv / k2-pad0 等 | baseline 回退 | 崩溃触发或 no_gain |

### 回归修复记录

wrapper 端到端验收发现 `tests/ops/test_conv3d_bias.py::test_conv3d_bias[stride2-fp16]` 崩溃
（`std::runtime_error` → Aborted，rc=134）：单跑通过但全量套件（前序 6 kernel 同进程）必现，设备端
`launch_npu` 阶段抛异常。A/B 确认 baseline 同 case `1 passed`，系 R1 改动引入。修法即上述形状级回退，
见 `opt_log.md §5`。

## 3. 优化结果（final，3 workload 同一候选 `r1_weight_resident`）

| workload | base µs | opt µs | Δ | 达标 | 路径 |
|---|---|---:|---:|---:|:--:|---|
| r3d-stem-bias-float16 | 6161.50 | 3090.14 | **−49.8%** | ✅ | R1 |
| video-down-bias-float16 | 87.00 | 86.42 | −0.7% | ❌ | baseline 回退 |
| unet-encoder-bias-bfloat16 | 6422.53 | 6449.49 | +0.4% | ❌ | baseline 回退 |

> SOL = roofline 理论时长（max(flops/峰值算力, bytes/峰值带宽)）/ 实测 Task Duration；conv3d 小通道
> workload 的 implicit-GEMM 结构天然远离 roofline（16-fractal 通道填充 + 27-tap im2col 重复读）。

### 精度

- 全量 `python -m pytest tests/ops/test_conv3d_bias.py -q` → **17 passed**。
- `python perf_opt/Conv3dBiasFwdOp.py --level all` → **All check passed!**：L0 5 例、L1 10 例、L2 6 例、
  boundary 3 例全 PASS（max_abs ≤ 5e-1，与基线一致）；2 个 `smoke-k3-s1-bias` fp16/bf16 仅精度回归 PASS。
- `tileops-report --op Conv3dBiasFwdOp --prof-mode msprof` → **status=passed**（17/17，r3d-stem 3090.14 µs）。

## 4. 时间

| 项 | 值 |
|---|---|
| 首轮调优（+08:00） | 2026-10-18 09:09 → 09:40 |
| 回归修复（+08:00） | 2026-10-18 09:45 → 10:15 |

## 5. 产物

- `.../conv3d_kernel/perf_opt/Conv3dBiasFwdOp.py`（final，`r1_weight_resident` + 形状级回退，
  sha256 1d50614b…）
- `perf_opt/opt_log.md`、`perf_records.jsonl`（append-only）、`workload_inventory.json`、`profiles/`、`logs/`
- 实验分支：`perf_opt/candidates/r1_tap_pack_weight_resident.py`；探针：`perf_opt/probe_sliced_gemm.py`

## 6. 建议

**建议采纳 wrapper**（把 `perf_opt/Conv3dBiasFwdOp.py` 切换为生产 baseline）：r3d-stem-bias 1.99× 为真实
Task Duration 下降，精度全通过；video-down/unet 走 baseline 等效回退（性能不劣于 baseline），stride2-fp16
等其余 shape 经形状级回退后全量 17/17 通过。剩余 gap（unet 27-tap 输入重复读、小通道 16-fractal 填充）是
工具链级限制（切片 gemm 读数错位、L1→L1 与 MTE2 共享端口），参数/tiling 级不可解，留待工具链能力补齐后重验。
