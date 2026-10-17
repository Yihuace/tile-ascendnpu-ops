# Conv1dFwdOp — Stage 4 调优日志

> 工具链版本戳：CANN **9.0.0**（bishengir-compile 1.1.0 / hivmc 0.2.0）/ tilelang 0.1.2 dev root
> (`/home/developer/tilelang-mlir-ascend`) / Ascend910B3（Chip 6，20 AIC）/ 2026-10-17。
> 口径：**msprof op Task Duration(us)**（唯一时延口径）；ratio 由 benchmark 按 `eval_roofline` 计算。
> NPU 纪律：所有上机命令经 `.agents/tools/npu_lock.sh` 串行执行。

## 0. 对象与分派

| label | 分派 kernel | kernel 文件 |
|---|---|---|
| whisper-conv1(+bias) / encodec-deep(+bias) | `conv1d_unit_stride` | `_conv1d_unit_stride_kernel.py` |
| wav2vec2-layer1(+bias) / encodec-init(+bias) / smoke-k3-s1 | `conv1d` | `_conv1d_kernel.py` |

benchmark 入口：`examples/TileOPs/benchmarks/ops/bench_conv1d.py::test_conv1d_manifest_bench`。
调用链：`Conv1dFwdOp.forward → _select_kernel_key → Conv1dUnitStrideKernel/Conv1dKernel → _run`。

## 1. Baseline（迁移态，未改任何 kernel）

| kernel_id | workload_id | baseline_us | ratio(%) |
|---|---|---:|---:|
| conv1d_unit_stride | whisper-conv1-float16 | 5339.17 | 0.118 |
| conv1d_unit_stride | whisper-conv1-bfloat16 | 5846.42 | 0.108 |
| conv1d_unit_stride | whisper-conv1-bias-float16 | 5610.73 | 0.113 |
| conv1d_unit_stride | whisper-conv1-bias-bfloat16 | 5584.41 | 0.114 |
| conv1d_unit_stride | encodec-deep-float16 | 1924.62 | 0.069 |
| conv1d_unit_stride | encodec-deep-bfloat16 | 1901.76 | 0.070 |
| conv1d_unit_stride | encodec-deep-bias-bfloat16 | 1857.40 | 0.072 |
| conv1d | wav2vec2-layer1-bias-float16 | 589.95 | 0.329 |
| conv1d | wav2vec2-layer1-bias-bfloat16 | 595.47 | 0.328 |
| conv1d | encodec-init-float16 | 283.21 | 0.409 |
| conv1d | encodec-init-bias-float16 | 256.26 | 0.457 |
| conv1d | encodec-init-bias-bfloat16 | 268.75 | 0.439 |
| conv1d_unit_stride | encodec-deep-bias-float16 | — | — | **编译 SIGSEGV（CG-2026-0003），无有效 profile** |
| conv1d | wav2vec2-layer1-float16 | 待重采 | — | 批量内 msprof 回退 events |
| conv1d | wav2vec2-layer1-bfloat16 | 待重采 | — | 同上 |
| conv1d | encodec-init-bfloat16 | 待重采 | — | 同上 |
| conv1d | smoke-k3-s1-bfloat16 | 待重采 | — | 同上（smoke fp16 仅做精度） |

原始数据：`perf_opt/logs/bench_all.json`（12 条 msprof 有效）+ `perf_opt/perf_records.jsonl`。
逐 case 独立进程重采：`perf_opt/rerun_events_fallback.sh`。

## 2. 首轮诊断（whisper-conv1/fp16，5443 µs）

profile：`perf_opt/profiles/whisper-conv1-fp16/.../PipeUtilization.csv`

| 指标 | 实测 | 读数 |
|---|---:|---|
| `aic_cube_ratio` | **0.15%** | Cube 几乎空闲 |
| `aic_cube_time` | 8.12 µs | 真实 GEMM 只占 0.15% |
| `aiv_mte2_ratio` | **97.1%** | Vector 侧 MTE2 独占 |
| `aiv_mte2_active_bw` | **2.39 GB/s** | 远低于 1.75 TB/s 平台带宽 |
| `GM_to_UB_datas` | 13270 KB / core | 每核重复搬运 |
| `Block Dim` | 20 | 20 核各自执行同一段 prelude |

**根因**：`_prelude`（Vector Scope）在**每个 kernel core 上重复**执行：
1. 权重重排：对全部 `n_co=20` 个 co 块做 GM→UB→两次 `T.transpose`→GM `w_flat`；
2. `x_pad` 全量拷贝：`x → UB → x_pad`。

Cube 真正的 GEMM 只有 8 µs，其余 5.4 ms 全部是 prelude 的重复搬运。
`_conv1d_kernel`（wav2vec / encodec-init）同构，但 MTE2 占比 18%–66%、cube_ratio 0.7%–1.1%，
病态程度低于 unit_stride，先修 unit_stride 两个最重 workload。

## 3. Round 1

### R1-A：pad=0 且 `c_in_pad==c_in` 时跳过 `x_pad` 拷贝

- 候选：`perf_opt/_conv1d_unit_stride_kernel.py`
- 改动：新增编译期 `identity_x = (pad_left==0 and pad_right==0 and c_in==c_in_pad)`；
  `identity_x` 时 prelude 不拷贝 `x_pad`，Cube 的 B 块直接读 `x`。
- 合法性：Cube 的 B 索引上界 `off+tn ≤ out_l = l_in`（pad=0、stride=1），从不触碰补齐列。
- 预期：消除 20 核 × 480 KB 的冗余拷贝（whisper）。
- 状态：**已通过 L0，7 个 case 的 msprof 排队中**。

**编译形态踩坑（重要）**：

| 变体 | 改动 | `whisper-bias` | `encodec-bias` |
|---|---|---|---|
| v1 | `_cube` 增加 `x` 形参（Cube 直读 `x`） | ❌ bishengir SIGSEGV | —（未到） |
| v2 | 不改形参，identity 时把 `x` 传入 `x_pad` 槽位 | ✅ PASS | ❌ 与 baseline 相同（CG-2026-0003） |
| baseline | 原样 | ✅ PASS | ❌ CG-2026-0003 |

结论：**在 MixCV prim_func 里给宏新增一个 GM buffer 形参会触发 bishengir 崩溃**（即使逻辑等价）；
别名形式与 baseline 编译行为逐例一致。v1 的崩溃不是 R1-A 思路的问题，而是形参签名变化触发编译器缺陷。

| candidate | workload | baseline_us | candidate_us | Δ | l0 |
|---|---|---:|---:|---:|---|
| R1-A | whisper-conv1-float16 | 5339.17 | — | — | ✅ |
| R1-A | whisper-conv1-bfloat16 | 5846.42 | — | — | ✅ |
| R1-A | whisper-conv1-bias-float16 | 5610.73 | — | — | ✅ |
| R1-A | whisper-conv1-bias-bfloat16 | 5584.41 | — | — | ✅ |
| R1-A | encodec-deep-float16 | 1924.62 | — | — | ✅ |
| R1-A | encodec-deep-bfloat16 | 1901.76 | — | — | ✅ |
| R1-A | encodec-deep-bias-bfloat16 | 1857.40 | — | — | ✅ |

> L0 通过判据：模块内 `--level L0` 的 whisper/encodec 四个基础用例（bias/nobias）；
> `encodec-bias` 仍因 CG-2026-0003 崩溃，该 case 的基线本就缺失。

### R1-B（已编码待测）：权重重排按 core 切分

- 暂存：`perf_opt/candidates/r1b.py`（基线副本 + 映射改造）
- 映射：`by = kernel_id % gy`，`sub = kernel_id // gy`，`share = num_kernels // gy`；
  每个 core 只重排自己读取的 co 块，`bx/bz` 任务按 `sub` 分片。
- 无跨核同步：core 只读自己写的 `w_flat` 块（共享同一 by 的核写入内容完全相同）。
- 预期：whisper（gy=20/核=20）权重重排 20×→1×；encodec-deep（gy=4/核=20）20×→5×。
- 回退：`gy > num_kernels` 时保留原全量重排。

---

## 4. Round 1 收尾（2026-10-17）

### 4.1 编译器 flaky 判定（关键方法学修正）

同一文件重复 `--level L0`，统计崩溃点：

| 文件 | run1 | run2 | run3 |
|---|---|---|---|
| `_conv1d_unit_stride_kernel.py`（baseline） | 崩在 encodec-bias | 崩在更后的 nondiv 用例 | — |
| `perf_opt/candidates/r1b_map_only.py` | 崩在 whisper-bias | 崩在 encodec-bias | 崩在 encodec-bias |

结论：**CANN 9.0 的 bishengir SIGSEGV 是非确定性的**（同文件同 IR 每次崩在不同用例）。
因此 **单次编译失败不能归因于候选改动**；R1-A 的两个 bias case 编译失败也不能作为否决依据。
后续实验协议：每个候选至少重复 2 次编译，只有可稳定复现的失败才记为候选缺陷。
该事实作为 CG-2026-0003 的补充证据（同一根因的第三种表现：随机崩溃点）。

### 4.2 R1-A 判定：不采纳（性能原因，非编译原因）

| workload | baseline µs | R1-A µs | Δ |
|---|---:|---:|---:|
| whisper-conv1 fp16 | 5339.2 | 5351.4 | **+0.2%** |
| whisper-conv1 bf16 | 5846.4 | 5360.2 | −8.3% |
| whisper-conv1-bias fp16 | 5610.7 | 5867.5 | **+4.6%** |
| encodec-deep fp16 | 1924.6 | 1789.6 | −7.0% |
| encodec-deep bf16 | 1901.8 | 1726.1 | −9.2% |

只消除了 `x_pad` 拷贝（约占 prelude 的一小部分），**whisper 主 case 无收益**，放弃。
证据：`perf_opt/perf_records.jsonl`（candidate_id=r1a_identity_x）、`perf_opt/logs/bench_r1a-*.json`。

### 4.3 R1-B 判定：挂起（数值 bug 待查）

- 分区重排版：可编译，但 **数值错**（whisper L0 `max_abs=81.38`）。
  疑似原因：prelude 里用 runtime 的 `by` 做 GM slice + transpose，切块未正确落到 UB/L1；
- 仅改任务映射版（prelude 仍全量重排）：编译行为与 baseline 一致（flaky 分布相同），**逻辑未被证伪**；
- 结论：下一步应先用「只改映射 + 全量重排」验证映射正确性，再单独定位分区重排的数值 bug；
- 本窗口按“先易后难”切换，**Stage 4 保持 TUNING（可恢复）**，drop-in 已还原 baseline。

### 4.4 收尾状态

- `perf_opt/_conv1d_unit_stride_kernel.py` = baseline 副本（未激活）；
- `conv1d.py` wrapper 未改（`git diff` 干净）；
- 候选证据保留：`perf_opt/candidates/{r1a_identity_x.py,r1b.py,r1b_map_only.py}`；
- `perf_records.jsonl`：16 baseline + 5 candidate（R1-A）；
- 遗留：`encodec-deep-bias/fp16` 无有效 profile（CG-2026-0003）。
