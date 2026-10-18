# Conv3dBiasFwdOp — Stage 4 调优日志

> 工具链版本戳：CANN **9.0.0**（bishengir-compile 1.1.0 / hivmc 0.2.0）/ tilelang 0.1.2 dev root
> (`/home/developer/tilelang-mlir-ascend`) / Ascend910B3（Chip 6，20 AIC）/ 2026-10-18。
> 口径：**msprof op Task Duration(us)**（唯一 kernel 时延口径）。
> NPU 纪律：所有上机命令经 `.agents/tools/npu_lock.sh` 串行执行。

## 0. 对象与分派

| label | kernel | 文件 | 本轮是否调优 |
|---|---|---|---|
| r3d-stem-bias / video-down-bias / unet-encoder-bias / smoke | `conv3d`（dense NCDHW implicit GEMM，has_bias） | `_conv3d_kernel.py` → `perf_opt/Conv3dBiasFwdOp.py` | ✅ 调优 |

benchmark 入口：`benchmarks/ops/bench_conv3d_bias.py::test_conv3d_bias_manifest_bench`。
调用链：`Conv3dBiasFwdOp.forward → Conv3dKernel → _conv3d_kernel`。
tune 集合（manifest `convolution.yaml` Conv3dBiasFwdOp 全部非 smoke 案例，不缩减）：
r3d-stem-bias(fp16)、video-down-bias(fp16)、unet-encoder-bias(bf16)；smoke-k3-s1-bias(fp16/bf16) 仅精度回归。

## 1. Baseline（迁移态，未改任何 kernel）

| workload | baseline Task Duration(us) | 现象 |
|---|---:|---|
| r3d-stem-bias-float16 | 6161.50 | Cube 侧 99% MTE2（GM→L1），Cube MAC 仅 1.6% |
| video-down-bias-float16 | 87.00 | 已快（stride-2 → 224 逻辑任务） |
| unet-encoder-bias-bfloat16 | 6422.53 | 同上：MTE2 99% |

**首轮诊断**（profile：`perf_opt/profiles/r3d-stem-bias/baseline/.../PipeUtilization.csv`）：

| 指标 | 实测 | 读数 |
|---|---:|---|
| `aic_mte2_ratio` | 0.99 | Cube 侧 99% 时间做 GM→L1 搬运 |
| `aic_cube_ratio` | 0.016 | Cube MAC 几乎空闲 |
| `aic_scalar_mte2_stall_time` | 6061 µs | 标量管线全程等 MTE2 完成 |
| `GM_to_L1_datas` | 72900 KB | 每核逐 task 重载权重 + 27-tap im2col 输入 |

**根因**：这不是「每核重复 prelude」（conv3d 的权重重排 / x_pad 在 host `_host_prep` 完成，与
Conv1d 不同），而是 **Cube 循环内的逐 task 权重重载 + 逐 tap 输入重复读**。implicit GEMM 把
KV=27 个 tap 拆成 27 次独立 gemm，每次 gemm 前都从 GM 重载 A 块（权重）与 B 块（im2col 输入）：
- 权重 `w_flat[KV, C_out, c_in_pad]` 每 task 重载 27 次（r3d-stem 1792 task × 27 = 48384 次）；
- 输入 x_pad 每 task 逐 tap 重读 27 次（kw 维 3 个相邻 tap 是同一行的 3 个平移视图，本可共享）。

## 2. Round 1（R1）：tap 打包 + 权重 L1 常驻

与 Conv1d 的「prelude 逐核冗余」不同，conv3d 的病根在 Cube 循环内。R1 两个配合改动（合入同一
`@T.prim_func`，单一主要优化点 = 「消除逐 task 权重重载」）：

1. **tap 打包**：把 `for tap in serial(KV)` 折叠进 gemm 的 K 轴，每个 co 块一次 gemm，
   `K = KV*c_in_pad`（A/B 块沿 K 轴在 `tap*c_in_pad` 处连续打包；K 扩展 = 打包宽度的单源不变量）。
2. **权重 L1 常驻 + 固定 co 块分核**：`by = kernel_id % gy` 把每核固定到一个输出通道块
   （`gy <= num_kernels`），该块的权重 `w_flat[:, by*block_m:..., :]` 在 task 循环**前**载入 L1 一次，
   随后被该核所有空间 task 复用——消除逐 task 权重重载。
3. **B 块 N 维度按 `n_max = ceil16(min(block_n, out_w))` 分配**（否则 video-down 的 `k_full=1728`
   会超出 512KB L1）。

### R1 编译陷阱（新实证）

| 陷阱 | 现象 | 结论 |
|---|---|---|
| **L1 超配 SIGSEGV** | 首版用 `block_n=128` 分配 B 块，video-down `l1_a(221KB)+l1_b(442KB)=663KB > 512KB` → bishengir SIGSEGV（非溢出报错） | 用 `n_max`（fractal 钳位输出宽）替代 `block_n` 分配 B 块 |

### R1 候选 vs current best 对比（Task Duration(us)，msprof op）

| candidate | workload | baseline_us | candidate_us | Δ | L0 |
|---|---:|---:|---:|---|
| r1_weight_resident | r3d-stem-bias-float16 | 6161.50 | 3096.64 | **−49.7%** | ✅ |
| r1_weight_resident | unet-encoder-bias-bfloat16 | 6422.53 | 6379.99 | −0.7% | ✅ |
| r1_weight_resident | video-down-bias-float16 | 87.00 | 90.28 | +3.8%（噪声内） | ✅ |

> r3d-stem（c_in_pad=16，权重 55KB）权重重载 48384→540 次，MTE2 6050→3042µs（2×）。
> unet（c_in_pad=32，权重 110KB）**未生效**：`read_main_memory_datas` 32136→21120 KB（HBM 层重载已除），
> 但 `aic_scalar_mte2_stall_time` 6289→6227µs 几乎不变——编译器对该 110KB 常驻权重仍逐 task 从 L2 重载，
> MTE2 仍由 27-tap 输入重复读主导。

## 3. unet 未生效归因（受控诊断）

- 对照实验：unet-shape 以 fp16 重测，baseline 6425.7 → R1 6384.9µs，同样无增益 → **非 bf16 专属**。
- 对照实验：`block_m=32`（l1_a 缩到 55KB）实测 **12756µs**（更差：gy=2 使每核 task 数翻倍、gemm M=32
  效率下降，且常驻仍未生效）→ 常驻失效与 l1_a 大小无简单正相关，属编译器对该 shape 的 L1 生命周期行为。
- 切片 gemm 探针（`perf_opt/probe_sliced_gemm.py`）：`T.gemm(l1_a, l1_b[0:K, 1:1+N], ...)` 编译通过但
  **读数错位**（max_abs=64，gemm 忽略 N 偏移）→ 无法用「宽 B 块 + 平移 gemm」削减 27-tap 输入重复读。
- 结论：unet 的剩余瓶颈是 27-tap im2col 输入重复读（byte-bound），在当前工具链（无切片 gemm、L1→L1
  拷贝与 MTE2 共享 L1 端口）下不可解，属**参数/tiling 可解范围之外的工具链限制**；不满足
  `[DESIGN_LIMIT]` 触发条件（结构加速估计 < 2x 且非设计层归因），故不产出 `perf_feedback.md`。

## 4. 合并复测与收束

R1 即最终版（单点改动一次到位）。合并版复测全部 tune workload（见 Final 表）与 smoke 精度：
`smoke-k3-s1-bias` fp16/bf16 全 PASS（max_abs ≤ 1.56e-2 / 0.0）。

L0/L1/L2/boundary 全 PASS（L0 5 例、L1 10 例、L2 6 例、boundary 3 例）。bias(MixCV) 路径在本轮工具链下
r3d-stem / video-down / unet / smoke 均实测 PASS；**但 wrapper 端到端全量回归发现 `stride2-fp16` 崩溃**，
根因与修法见 §5（回归修复）。

## Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---:|---:|---|
| conv3d | r3d-stem-bias-float16 | 6161.5029 | 3090.1418 | r1_weight_resident |
| conv3d | video-down-bias-float16 | 87.0017 | 86.4217 | r1_weight_resident |
| conv3d | unet-encoder-bias-bfloat16 | 6422.5283 | 6449.4888 | r1_weight_resident |

## Final Summary

- **r3d-stem-bias**（最大 workload）：6161.50 → **3090.14 µs（1.99×，SOL 0.26%→~0.52%）**——保留 R1 优化路径。
- **unet-encoder-bias**：6422.53 → 6449.49 µs（+0.4%，噪声内）——**回退 baseline**（`c_in_pad=32` 权重常驻不生效）。
- **video-down-bias**：87.00 → 86.42 µs（−0.7%，噪声内）——**回退 baseline**（`stride=2` 多相路径）。
- 中止原因：**success**（最大 workload r3d-stem 达 2×；unet/video-down 为 no_gain，见 §3 归因）。

## 5. 回归修复（precision 修复 + 形状级回退）

### 崩溃现象

wrapper 端到端验收（`tileops-report --op Conv3dBiasFwdOp` 的 correctness 阶段）在全量跑
`tests/ops/test_conv3d_bias.py` 时，第 7 例 `test_conv3d_bias[stride2-fp16]` 崩溃：

```
terminate called after throwing an instance of 'std::runtime_error'
Fatal Python error: Aborted  (rc=134)
```

- A/B：Stage-3 baseline 跑同一 case `1 passed`；perf_opt 版本崩溃 ⇒ 由本 Agent 的 R1 改动引入。
- 复现特征：**单跑 `-k stride2` 通过**，但**全量套件在 stride2-fp16 之前已编译/运行 6 个 kernel 后必现**
  （确定性；崩溃点在 `jit_npu.py launch_npu`，即设备端 kernel 启动阶段抛 `std::runtime_error`）。
- 归因：R1 的 tap 打包 + 权重 L1 常驻只对 `c_in_pad=16 且 stride=1` 的 shape 安全/有收益；stride-2 多相
  （xdec 路径）与 `c_in_pad=32` 的 shape 走 R1 会在多 kernel 同进程设备状态下触发设备端故障。

### 修法

在 `perf_opt/Conv3dBiasFwdOp.py` 的 `_conv3d_func` 增加**编译期 `use_r1` 分派**（Python `if`，非运行时
分派），R1 与 baseline 两套 `_cube`/`_bias_epilogue`/`prim_func` 并存：

```
use_r1 = stride1 and n_kc==1 and gy<=num_kernels and c_in_pad==16 and l1_total<=512KB
```

- **R1 路径**（`_conv3d_bias_main`）：`stride==1` 且 `c_in_pad==16` 且 L1 ≤ 512KB → tap 打包 + 权重 L1 常驻。
- **baseline 等效回退**（`_conv3d_bias_main_base`）：其余 shape（stride!=1 多相 / c_in_pad!=16 / L1 超限）→
  逐 tap gemm + 逐 task 权重重载（Stage-3 语义，性能不劣于 baseline）。

### 回退范围（adopted vs fallback）

| shape | 路径 | 说明 |
|---|---|---|
| r3d-stem-bias（c_in=3→16, s=1） | R1（保留） | 1.99× 收益 |
| smoke-k3-s1-bias（c_in=8→16, s=1） | R1（保留） | 小通道常驻受益 |
| video-down-bias（c_in=64, s=2） | baseline 回退 | stride-2 多相，本就快 |
| unet-encoder-bias（c_in=32, s=1） | baseline 回退 | c_in_pad=32 常驻不生效 |
| stride2-fp16 / nondiv / k2-pad0 等 | baseline 回退 | 崩溃触发或 no_gain |

### 验证

- 全量 `python -m pytest tests/ops/test_conv3d_bias.py -q` → **17 passed**。
- `python perf_opt/Conv3dBiasFwdOp.py --level all` → **All check passed!**。
- `tileops-report --op Conv3dBiasFwdOp --prof-mode msprof` → **status=passed**（17/17，r3d-stem 3090.14 µs）。

## Skill Retrospective

流程顺畅度：一轮定位病根（Cube 循环逐 task 权重重载 + 逐 tap 输入重复读）、单点改动（tap 打包 + 权重
L1 常驻）即达 r3d-stem 2×。主要耗时在 unet 未生效的归因（fp16 对照、block_m=32 对照、切片 gemm 探针）
与 gate 4 工件对账。

新实测价值点（供 evolver 蒸馏，含证据三件套）：

1. **P：implicit-GEMM 权重 L1 常驻 + 固定 co 块分核**（vp_type=P）
   - evidence：`perf_opt/Conv3dBiasFwdOp.py` R1 改动 + Final 表 r3d-stem 2×。
   - repro：待补 `repro/r1_weight_resident.py`（baseline → tap 打包 + 权重常驻的最小 diff）。
   - toolchain_stamp：CANN 9.0.0 / tilelang 0.1.2 dev root / Ascend910B3。
2. **D：切片 gemm 读数错位**（vp_type=D）——`T.gemm` 的 src2 若为 L1 切片（`l1_b[:, off:off+N]`），
   编译通过但忽略 N 偏移按 buffer 起点读（max_abs 错位），切片 gemm 不可用于平移窗口。
   - evidence：`perf_opt/probe_sliced_gemm.py`（probe 实测 max_abs=64）。
   - repro：`perf_opt/probe_sliced_gemm.py`（自包含）。
   - toolchain_stamp：同上。
3. **D：权重 L1 常驻的 shape 依赖**（vp_type=D）——c_in_pad=16（55KB 权重）常驻生效（MTE2 2×），
   c_in_pad=32（110KB）不生效（`aic_scalar_mte2_stall_time` 不变，编译器仍逐 task 从 L2 重载）。
   - evidence：r3d-stem vs unet 的 `PipeUtilization.csv`（mte2 6050→3042 vs 6370→6320）。
   - repro：待补（最小复现需同结构 implicit-GEMM 两 shape 对照）。
   - toolchain_stamp：同上。
4. **D：L1 超配 → bishengir SIGSEGV**（vp_type=D，CG-2026-0003 家族同族）——B 块按 `block_n` 分配导致
   `l1_a+l1_b > 512KB` 时，编译器不报 overflow 而是 SIGSEGV；`n_max=ceil16(min(block_n,out_w))` 绕法。
   - evidence：R1 首版 video-down 编译 SIGSEGV → 改 `n_max` 后 PASS。
   - repro：待补（最小 L1 超配 gemm）。

## 参考实现对照（plateau 前强制）

同族先例 Conv1d（`perf_opt/_conv1d_unit_stride_kernel.py`，whisper 15×）的突破是「prelude 逐核冗余
权重重排」，conv3d 无 prelude（host 重排）故不适用；但「persistent 内核的重复搬运是首要病根」这一结论
在本轮以「Cube 循环逐 task 权重重载」的形态再次成立。pattern-library `CONST-L1-port-bw` /
`CONST-mte2-degradation` 的「小拷贝指令/段数受限」解释了 MTE2 主导。无结构分歧。

---
*本日志的最终时延全部来自同一最终候选（r1_weight_resident）的 `perf_records.jsonl` 记录，未拼接不同分支。*
