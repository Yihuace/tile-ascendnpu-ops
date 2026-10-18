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
|---|---|---:|---:|---|
| conv3d | r3d-stem-bias-float16 | 6116.3818 | 2588.6714 | r3_merged |
| conv3d | video-down-bias-float16 | 86.4812 | 86.7015 | r3_merged |
| conv3d | unet-encoder-bias-bfloat16 | 6428.0469 | 301.5900 | r3_merged |

> baseline = Stage-3 `_conv3d_kernel.py` 同工具链复测（`phase=baseline`，见 §6.0）；
> final = 最终 `perf_opt/Conv3dBiasFwdOp.py`（sha256 `2e226a27…`）在同一候选上重测（`phase=final`）。
> smoke（`smoke-k3-s1-bias` fp16/bf16）只做精度回归（PASS），不计入本表。

### 第一轮 Final 表（历史，已被本轮取代）

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| conv3d | r3d-stem-bias-float16 | 6161.5029 | 3090.1418 | r1_weight_resident |
| conv3d | video-down-bias-float16 | 87.0017 | 86.4217 | r1_weight_resident |
| conv3d | unet-encoder-bias-bfloat16 | 6422.5283 | 6449.4888 | r1_weight_resident |

## Final Summary（第一轮，已被 §6 取代）

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

---

# 6. 第二轮调优（同一 9.0 工具链复测基线）：unet 路径 21×

> 本节为 2026-10-18 晚（18:20 → 19:10）的第二次 Stage-4 会话，目标：**unet-encoder（`c_in_pad=32`）的
> 权重/输入搬运**。§1–§5 的结论保留为历史；本轮不改 `DESIGN.md` / `_conv3d_kernel.py` / wrapper，
> 全部产物写在 `perf_opt/`。

## 6.0 本轮基线（先测，分两个候选）

同一会话、同一 9.0 工具链、20 次 launch 中位数（`perf_opt/run_measure.py`）：

| workload | Stage-3 `_conv3d_kernel.py` | 当前 `perf_opt/Conv3dBiasFwdOp.py`（R1） | 说明 |
|---|---:|---:|---|
| r3d-stem-bias-float16 | 6116.38 | 3095.80 | R1 生效（1.98×）|
| video-down-bias-float16 | 86.48 | 87.10 | stride-2 回退 baseline |
| unet-encoder-bias-bfloat16 | 6428.05 | 6428.57 | `c_in_pad=32` 走 baseline 回退 ⇒ **与 Stage-3 完全相同** |

即本轮起点：unet **完全没有**被优化（R1 的 `c_in_pad==16` 闸门把它送回 Stage-3 路径）。

## 6.1 诊断（现象 → 结论，逐条实测）

| 观测（unet baseline） | 数值 | 结论 |
|---|---|---|
| `aic_mte2_ratio` / `aic_cube_ratio` | **0.991 / 0.018** | 纯 MTE2（GM→L1）受限，Cube 空闲 |
| `aic_mte2_instructions` | **91 774**/核 | 2048 task × 27 tap × 2 copy = 110 592 次「切片 copy」被拆成 ~92 k 条 DMA |
| `GM_to_L1_datas` / `GM_to_L1_bw_usage_rate` | 100 116 KB / **5.63%** | 带宽利用率只有额定值的 1/18 ⇒ 不是带宽受限，是**小段 DMA 的段数/延迟受限** |
| 每 copy 源形态 | `x_pad[bz, 0:32, off:off+64]` = 32 行 × **128 B**，行间跨 359 KB（通道平面） | 一次 copy = 32 个分散小段 |
| 端口实测 | 每 MTE2 指令 ≈ 1.09 KB、69 ns | 与「每段 ~70 ns」吻合 |
| **A/B 探针**（把一侧 copy 缩到 16×16，仅测时间） | A(权重) copy 全量 + B 缩小 → **2749 µs**；B 全量 + A 缩小 → 7160 µs；原版 6428 µs | 权重侧 ≈ 2.7 ms、输入侧 ≈ 3.7 ms（两者在 MTE2 上部分重叠）⇒ **单改一侧最多 2.3×** |
| 强制 R1（放开 `c_in_pad==16`）编译 IR | `l1_a` 的 27 次 `nd2nz` 确实被提到 task 循环**外**（`memref.alloc` 在循环外） | 「常驻没写进去」不成立；不生效的原因是 **L1 占用 220 KB 后丧失双缓冲/流水重叠**，逐 tap 被串行化 |
| 副本布局探针 | `T.copy` 的 `dst_continuous` 语义 | 只有把**源/目的都变成连续段**才能真正提速 |

**结论**：unet 的病根不是「权重没常驻」，而是 **im2col 输入 copy 的段数**：每次 copy 的源是
`c_in_pad` 个 128 B 的小段（通道平面跨度 359 KB），段数 × 27 tap × 2048 task = 91 774 条 DMA；
`c_in_pad=16`（r3d-stem）段更短但 tap 数相同，所以 R1 靠**减少 copy 次数**拿到 2×，而
`c_in_pad=32` 时权重常驻把 L1 撑到 220 KB、丢掉多缓冲，抵消了收益。**要再快必须减少 DMA 段数。**

## 6.2 Round 1（结构）：R2 = padded-N 多行空间分块

改动（`perf_opt/Conv3dBiasFwdOp_round1_r2_padded_n.py`，单一主要优化点 = **减少 copy 段数**）：

- GEMM 的 N 轴从「一条输出行（64 列）」改为 **`r2_rows` 条输出行 × `w_pad` 的 padded 坐标**
  （`n_tile = r2_rows*w_pad`）。因为 `x_pad` 的行距本来就是 `w_pad`，一条连续 span
  `x_pad[bz, c, row0*w_pad + kw : + n_tile]` 正好覆盖 `r2_rows` 行、且每行自带 kw 平移 —
  一次 copy 的数据量 ×`r2_rows`，copy 次数 ÷`r2_rows`。
- `kw` 打包进 GEMM 的 K 轴（K = `kw*c_in_pad`），`(kd, kh)` 走 9 次循环。
- 尾部行组**重叠**（`oh0 = min(hg*R, out_h-R)`）替代动态尾块：重复计算的行写出相同值，幂等。
- 行内 `c >= out_w` 的列是 padding 垃圾值，由**逐行 fixpipe 的 L0C 列切片**丢弃。
  新增探针 `probes/probe_cube_caps.py` 证实：**fixpipe 的 L0C 列偏移被正确执行**
  （与 TRAP-cube-gemm-sliced-operand 描述的 gemm 操作数偏移不同一侧）。
- 主机侧：`_host_prep` 多分配 1 个全零 depth 平面（`plane_tot = (d_pad+1)*h_pad*w_pad`），
  避免最后一条 span 越过张量末尾 2 个元素（unet 的 `x_pad` 恰好结束在页边界）。

| candidate | workload | current best µs | candidate µs | Δ | L0 |
|---|---|---:|---:|---:|---|
| r2_padded_n | unet-encoder-bias-bfloat16 | 6428.57 | **1534.21** | **−76.1%（4.19×）** | ✅ |
| r2_padded_n | r3d-stem-bias-float16 | 3095.80 | **2609.39** | −15.7% | ✅ |
| r2_padded_n | video-down-bias-float16 | 87.10 | 85.90 | −1.4%（噪声内，仍走 base） | ✅ |

profile 变化（unet）：`aic_mte2_instructions` 91 774 → **16 039**（−82.5%），
`GM_to_L1_bw_usage_rate` 5.63% → **27.98%**，`aic_mte2_ratio` 0.99 → 0.94，
`aic_cube_ratio` 0.018 → 0.048。

## 6.3 Round 2（结构）：R3 = R2 + channel-last（NDHWC）输入

R2 之后段数 = 每次 copy `c_in_pad` 段（通道仍是跨平面的最外轴）。把输入改成 **NDHWC 暂存**
（`kw` 平移 = 平铺流上 `c_in_pad` 个元素），每次 tap copy 变成**一条连续段**，
GEMM 以 `b_transpose=True` 消费 `[N, K]`：

- `b_transpose` 在 Expert+L1 上**实测可用且数值正确**（探针 P2，`max_abs=9.5e-6`）——
  与 CG-2026-0026 记录的 Developer 模式 `b_transpose` 崩溃不同形态，该 CG 结论**不适用**于本形态。
- `xdec` 槽位承载 NDHWC（stride-1 时多相张量本来无用），多相 shape 仍放 decimated NCDHW；
  各路径按 `need_poly` 声明自己的 `xdec` 形状（poly 路径读 NCDHW、R3 读 NDHWC、R1/R2/base 不读）。
- **解析器陷阱（新实证）**：把 R3 的 `@T.prim_func` 再嵌一层 Python `if`（比 R1/R2 深一级）会让
  tilelang 的 source line-map 错位（`Source::GetLine` 的 `line_map size` 与绝对行号不匹配），
  现象是 `InternalError: ... requested line: N at index: N-1`，且**底层真实报错被渲染器吞掉**。
  修法：所有 `@T.prim_func` 分支必须放在**同一层** Python `if`（R2/R3 做成兄弟分支）。

| candidate | workload | current best µs | candidate µs | Δ | L0 |
|---|---|---:|---:|---:|---|
| r3_ndhwc | unet-encoder-bias-bfloat16 | 1534.21 | **306.41** | **−80.0%（5.0×；对 Stage-3 共 21.0×）** | ✅ |
| r3_ndhwc | r3d-stem-bias-float16 | 2609.39 | 4961.30 | **+90%（回退）** | ✅ |
| r3_ndhwc | video-down-bias-float16 | 85.90 | 86.20 | 噪声内 | ✅ |

r3d-stem 回退归因（实测）：`c_in_pad=16` 时 NDHWC 的内层连续段只有 **32 B**，
`aic_mte2_instructions` 1675 条但 `aic_mte2_time` 4712 µs（**2.8 µs/条**，unet 为 0.19 µs/条）⇒
32 B 段落回比「16 段 × 256 B 跨步」更差的 DMA 形态。**据此把 R3 闸门收紧到 `c_in_pad >= 32`，
`c_in_pad==16` 走 R2。**

profile（unet，R3）：`aic_mte2_ratio` 0.605、`aic_cube_ratio` **0.248**、`aic_mte1_ratio` 0.326、
`aic_fixpipe_ratio` 0.112 —— 已从「单侧 MTE2 独大」转为三管道均衡（L1→L0 操作数端口开始成为新成本）。

## 6.4 Round 3（合并）：R3(c_in_pad≥32) ∥ R2(c_in_pad<32) ∥ R1 ∥ Stage-3 回退

同一 `@T.prim_func` 内的**兄弟编译期分支**（互斥闸门，无运行时分派）：

```
use_r3 = stride1 & dil1 & n_kc==1 & gy<=K & out_w<=block_n & R>=2 & c_in_pad>=32 & L1/L0C fits
use_r2 = 同上（不含 c_in_pad>=32） & c_in_pad<32
use_r1 = stride1 & n_kc==1 & gy<=K & c_in_pad==16 & L1<=512KB
else   = Stage-3 等效回退（含 stride-2 多相）
```

| candidate | workload | current best µs | merged µs | Δ | L0 |
|---|---|---:|---:|---:|---|
| r3_merged | unet-encoder-bias-bfloat16 | 306.41 | 315.27 | +2.9%（噪声内） | ✅ |
| r3_merged | r3d-stem-bias-float16 | 2609.39 | 2630.27 | +0.8%（噪声内） | ✅ |
| r3_merged | video-down-bias-float16 | 85.90 | 86.26 | 噪声内 | ✅ |

合并版复测本 kernel 全部 tune workload + smoke 精度（smoke fp16/bf16 PASS，74.2/74.6 µs，
Stage-3 为 174.8 µs）⇒ 采纳为 current best。

## 6.5 Round 4（参数）：每 task 行数扫描（`r3_rows` / `r2_rows`）

| candidate | workload | µs | 结论 |
|---|---|---:|---|
| r3_rows=4 | unet | 309.77 | 差于默认 |
| r3_rows=5 | unet | 303.69 | 平 |
| **r3_rows=6（默认，L0C 上限）** | unet | **302.47** | ✅ 默认即最优（`config_no_gain`）|
| r2_rows=2 | r3d-stem | 3547.73 | 差 |
| **r2_rows=3（默认，L0C 上限）** | r3d-stem | **2641.37** | ✅ 默认即最优 |
| r2_rows=4 | r3d-stem | 3065.08 | 差（n_tile=512 顶满 L0C 128 KB）|

参数面已探到边界：`r*_rows` 的上界就是 L0C 容量（`block_m*n_tile*4 ≤ 120 KB`）与 `512/w_pad`，
默认配置已取到最优；本轮参数方向 `config_no_gain`。

## 6.6 中止判定

- 结构方向：R2（段数 ÷6）、R3（段数 ÷32）两轮均产出真实增益；参数方向已到边界（`config_no_gain`）。
- unet：6428 → **301.6 µs（21.3×）**；r3d-stem：3095.8 → **2588.7 µs（对 Stage-3 共 2.36×）**；
  video-down 持平（本 workload 已 86 µs，无结构空间）。
- 剩余 gap（unet，ceiling 证据）：`aic_cube_ratio 0.256`、`aic_mte1_ratio 0.338`、
  `aic_mte2_ratio 0.621`、`aic_fixpipe_ratio 0.107` —— 三管道已均衡，剩余成本是
  **27-tap im2col 的固有 3× kh 重复**（同一输入行被 kh=0/1/2 读三次）与 L1→L0 操作数端口。
  再往下需要「按 kh 预移位的主机侧布局把 K 扩到 `kh*c_in`」或「gemm 支持操作数平移视图」，
  前者是主机侧 3× 流量、后者被 TRAP-cube-gemm-sliced-operand 阻断 ⇒ **本轮止损**，
  不满足 `[DESIGN_LIMIT]`（结构加速估计 < 2× 且非设计层归因），不产出 `perf_feedback.md`。

## 6.7 Final（本轮收束，全部来自同一最终候选 `r3_merged`，sha256 `2e226a27…`）

| kernel_id | workload_id | Stage-3 基线 µs | 本轮起点（perf_opt R1）µs | final µs | 对 Stage-3 | 对起点 | 路径 |
|---|---|---:|---:|---:|---:|---:|---|
| conv3d | r3d-stem-bias-float16 | 6116.38 | 3095.80 | **2588.67** | −57.7%（2.36×）| −16.4% | R2 |
| conv3d | video-down-bias-float16 | 86.48 | 87.10 | 86.70 | −0.2%（噪声内）| −0.5% | base（stride-2）|
| conv3d | unet-encoder-bias-bfloat16 | 6428.05 | 6428.57 | **301.59** | **−95.3%（21.3×）** | −95.3% | R3 |
| conv3d | smoke-k3-s1-bias fp16/bf16 | 174.76 | — | 74.22 / 74.64 | −57.5% | — | R2（仅精度回归）|

`tileops-report --op Conv3dBiasFwdOp --prof-mode msprof`（run_id `20261018_185919_687302_Conv3dBiasFwdOp`）：
**correctness 17/17（0 failed，无 skip）、status=passed**；ratio 由 `0.76–6.29%` 提升到
**`1.68% – 20.43%`**（unet 0.76% → **20.43%**）。

## 6.8 本轮价值点（供 evolver 蒸馏，含证据三件套）

1. **P：padded-N 多行空间分块**（vp_type=P）——把 implicit-GEMM 的 N 轴从「一条输出行」扩到
   「R 条输出行 × padded 行距」，使一条连续 GM span 覆盖 R 行且自带 kw 平移；copy 次数 ÷R、
   每 copy 数据 ×R。unet `c_in_pad=32`：1534 → 306 µs 的直接前置（6428 → 1534）。
   - evidence：`perf_opt/Conv3dBiasFwdOp_round1_r2_padded_n.py`；profile 91 774 → 16 039 条 MTE2、
     bw 5.63% → 27.98%；repro 待补（结构级，最小 repro 需 conv3d 级 host prep）。
   - toolchain_stamp：tilelang 0.1.2 dev root / CANN 9.0.0 / Ascend910B3 / 2026-10-18。
2. **P：channel-last（NDHWC）暂存 + b_transpose 消除 DMA 分段**（vp_type=P）——通道改最内轴后
   每次 tap copy 是单条连续段；Expert+L1 的 `b_transpose=True` 实测可用（`max_abs=9.5e-6`）。
   注意**内层段宽必须 ≥64 B**：`c_in_pad=16`（32 B）反而比跨步版本慢 1.9×。
   - evidence：`perf_opt/Conv3dBiasFwdOp_round2_r3_ndhwc.py`、`probes/probe_cube_caps.py`（P2）；
     unet 1534 → 306 µs，r3d 2609 → 4961 µs（反例）。
3. **D：`T.copy` fixpipe 的 L0C 列偏移被正确执行**（vp_type=D）——`T.copy(l0_c[0:M, off:off+N], gm)`
   实测等于 `(A@B)[:, off:off+N]`（`max_abs=7.6e-6`，偏移 32 列）；与
   TRAP-cube-gemm-sliced-operand（gemm 操作数偏移被忽略）互补：**固定管线支持列切片，gemm 不支持**。
   - evidence：`probes/probe_cube_caps.py`（P1）。repro：该脚本自包含。
4. **D：多层 Python `if` 嵌套的 `@T.prim_func` 触发 line-map 崩溃**（vp_type=D）——
   比同级分支深一层即触发 `Source::GetLine ... line_map size` 的 InternalError，真实报错被渲染器吞掉。
   绕法：所有 prim_func 分支放在同一层 `if`。
   - evidence：本轮 R3 首版（`InternalError: requested line: 302 at index: 301`）。
5. **D：copy 段数与段宽的实测代价**（vp_type=D）——跨 359 KB 平面的 128 B 段 ≈ **70 ns/段**；
   30 KB 连续段 ≈ 2.9 µs（≈ 4 GB/s/核）；32 B 宽的内层连续段退化到 2.8 µs/条。
   - evidence：`perf_opt/run_measure.py` 各轮 profile + `probes/probe_copy_floor.py`。

## 6.9 Skill Retrospective（本轮）

| 项 | 记录 |
|---|---|
| 主要耗时 | 机理定位（copy 段数 vs 字节）≈ 全程 1/3；三轮结构改动各 1 次编译+测量；解析器陷阱排查 ≈ 20 min |
| 有效方法 | ①「删/缩一侧 copy 只测时间」的 A/B 探针（一步区分 A/B 两侧占比，避免继续按旧结论猜）；②能力探针先行（L0C 列切片、`b_transpose`）再动手写结构；③用 `parse_only2.py`（mock `NPUUtils`）把「解析/编译」与「上机」解耦，1 分钟一轮迭代 |
| 流程问题 | 第一轮的「权重 L1 常驻」结论停留在现象层（"编译器不生效"），未验证 IR/机制；本轮 dump IR 才发现常驻**已生效**、真因是多缓冲被顶掉 + 段数受限。**建议**：D 类负向结论必须附「IR/机制证据」而非仅 profile 计数 |
| 流程问题 | `@T.prim_func` 的多层 Python `if` 嵌套会以误导性 InternalError 失败（真实报错被吞），且**无法从异常文本定位**。**建议**：写进 traps-compiler（已回写 §6.8-价值点 4） |
| BP 候选 | `BP_persistent_im2col_copy_segments`：persistent implicit-GEMM 的调优顺序应为「先数 DMA 段数（`aic_mte2_instructions` 与每段字节），再谈权重常驻」——段数未降时，权重常驻的收益会被多缓冲丢失吃掉 |
| 预算 | 本会话结构实验 4 个（R2/R3/合并/参数），全部 1 次上机即判定；无重复实验 |

## 6.10 perf_records.jsonl 记账更正（透明留痕）

本轮向 `perf_records.jsonl` 追加的是**结构化测量记录**，为满足 gate 4 的 `S4-PERF-RECORDS-*` 契约
做了两处**仅元数据**的更正（duration / l0_pass / artifact_sha256 / msprof_raw_path **一律未改**）：

1. **上一会话的 3 条 `phase=final`（候选 `r1_weight_resident`，sha `1d50614b`）→ 改记 `phase=merged`**，
   并加 `note` 说明。原因：契约要求同一 kernel 的 final **恰有一条**、且哈希须等于最终文件；
   旧 final 记录描述的是已被本轮取代的第一轮最终版本，其数据完整保留为 merged 记录。
2. **本轮追加行的 `phase` 归一到枚举 `baseline/candidate/merged/final`**（原先误用了
   `r2_padded_n`/`r3_ndhwc` 等实验名作为 phase；实验名仍完整保留在 `candidate_id` 中）。

两处更正后 `gate_lint.perf_lint`（Stage-4）实测 **failures=0, warnings=0**。
