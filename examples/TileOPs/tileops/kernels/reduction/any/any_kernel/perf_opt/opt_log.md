# Stage 4 调优日志 — AnyFwdOp `_logical_reduce_kernel`（op_kind = "any"）

- project: `TileOPs`；算子单元：`tileops/kernels/reduction/any/any_kernel/`
- baseline（只读）：`_logical_reduce_kernel.py`（sha256 `be23433d6ce0060bf89541a95834c2604dd799343ab4b29ced84a88ce55a4706`）
- 交付：`perf_opt/_logical_reduce_kernel.py`（sha256 `bb552517b7fbddf0b85bdf9e915a7c8f895f91bb26a1abfbc2119231e0a1e866`）
- 回归入口：`python perf_opt/_logical_reduce_kernel.py --level all`（L0 26 例 + L1 26 例 + L2 + Boundary 全 PASS，exit 0，101 次逐位精确比较）
- 唯一时延口径：`msprof op` 的 `Task Duration(us)`（OpBasicInfo.csv，30 launch 的 mean；A/B 用交错多 run 的合并中位数裁决）
- 设备/工具链戳：**Ascend910B4**（`npu-smi`：HBM 32768 MB / AIV 40 / UB 196608 B）、CANN 9.0.0、tilelang `0.1.2+e21da325`（repo commit `e21da32`，2026-10-18）
- 预算：`max_rounds=4`、`max_experiments=16`；本任务用 **7 个候选分支**（r1×3、r2×1、r3×2、merged×1）与 **4 轮**
- wrapper `../../any.py` 的 baseline/perf_opt 切换块**未翻转**（用户约束：wrapper 只读）；perf_opt 版本为独立交付件

---

## 1. Workload inventory（从迁移后 benchmark 展开，未用 DESIGN/manifest 代填）

来源：`pytest benchmarks/ops/bench_any.py --collect-only -q` → 6 例（`workloads_to_params("AnyFwdOp", include_extra=True)`，
manifest `reduction.yaml` 的 `AnyFwdOp.workloads` + `bench_any.py` 的 `extra.setdefault("dim", -1)`）。完整清单见
`workload_inventory.json`。

| workload_id | pytest id | shape | 声明 dtype | kernel dtype | params | 2D 视图 | kind |
|---|---|---|---|---|---|---|---|
| smoke-any-bool | `test_any_bench[smoke-any-bool]` | (32,256) | bool | float32 | dim=-1 | (32,256) | smoke |
| smoke-any-float16 | `[smoke-any-float16]` | (32,256) | float16 | float16 | dim=-1 | (32,256) | smoke |
| smoke-any-float32 | `[smoke-any-float32]` | (32,256) | float32 | float32 | dim=-1 | (32,256) | smoke |
| mask-validation-4k-bool | `[mask-validation-4k-bool]` | (32,4096) | bool | float32 | dim=-1 | (32,4096) | tune |
| mask-validation-32k-bool | `[mask-validation-32k-bool]` | (32,32768) | bool | float32 | dim=-1 | (32,32768) | tune |
| 3d-multidim-reduce-bool | `[3d-multidim-reduce-bool]` | (4,128,4096) | bool | float32 | dim=[0,2] | (128,16384) | tune |

分类依据：`smoke` 来自 pytest `smoke` 标记 + 独立 label token `smoke`；其余 3 例无 smoke 标记 → `tune`。无 skipped。

**字节口径**：manifest dtype=bool，但 Op 层 `to_logical_float32` 在**进 kernel 之前**把输入提升为 float32，
所以 kernel 实际读 4 B/元素、写 1 B/元素（int8 载体）。本文件所有带宽/验收线数字按 kernel 实际字节：
4k 524 320 B、32k 4 194 336 B、3d 8 388 736 B。manifest/report 的 bool 1 B/元素仅作台账，不作判据。

---

## 2. Baseline Performance Test Data（round 0，本机实测）

命令模板（每个 workload 一条，`launch-count=30`、`warm-up=5`，全部经 `bench_driver.py --impl <file>` 走同一代码路径；
`msprof op` 串行，`flock /tmp/tilelang-msprof.lock`）：

```text
msprof op --kernel-name=main --output=perf_opt/profiles/baseline/round0/<cand>/<wl> \
  --launch-count=30 --warm-up=5 --dump=off --aic-metrics=Roofline,BasicInfo,PipeUtilization,... \
  python perf_opt/bench_driver.py --impl <kernel.py> --case <workload_id> --launches 35
```

| kernel_id | workload_id | 路径/配置 | Task Duration(us) | AICore(vec) 利用率 | MTE2 | MTE3 | GM 字节 | 达成带宽 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `_logical_reduce_kernel::main` | mask-validation-4k-bool | R, bm=2, 16 blocks, 1 tile | 2.9833 | 0.385 | 0.281 | 0.111 | 524 320 | 175.8 GB/s |
| `_logical_reduce_kernel::main` | mask-validation-32k-bool | C, bm=1, tn=8192, 4 tiles, 32 blocks | 7.7300 | 0.370 | 0.459 | 0.033 | 4 194 336 | 542.7 GB/s |
| `_logical_reduce_kernel::main` | 3d-multidim-reduce-bool | C, bm=1, tn=8192, 2 tiles, 40 blocks × 4 tasks | 14.7160 | 0.326 | 0.561 | 0.061 | 8 388 736 | 569.6 GB/s |

排序（先大后小、先结构收益大的）：**3d（14.72 µs，绝对耗时最大、双 tile + 4 串行任务）→ 32k（7.73 µs，4 tile，
仅 32/40 核）→ 4k（2.98 µs，latency-bound）**。smoke 不采性能、不排 winner，只做精度回归。

### 2.1 设计估算 vs 实测（D-2 回填，DESIGN.md §4.5 设计期 roofline）

| workload | DESIGN 估算下界 | 本机实测 baseline | 偏差 | 失准项 |
|---|---:|---:|---|---|
| 32k | 12–24 µs | 7.73 µs | 实测 **1.6–3.1× 低于**估算 | 发射项（12 op/block × 0.25–0.5 µs/op 的串行上界模型）高估；实测 mte2 主导 |
| 3d | 19–38 µs | 14.72 µs | 实测 **1.3–2.6× 低于**估算 | 同上 |
| 4k | 5–7 µs | 2.98 µs | 实测 **1.7–2.3× 低于**估算 | 同上（且 DESIGN 假设的 2.4 µs 固定地板在本机为 1.77 µs） |

结论：DESIGN 的发射项模型（0.25–0.5 µs/op）对本算子偏保守 2× 左右，实测是**读带宽主导**而非发射主导；
该偏差方向不影响设计选型（三条路径结构都正确），但 Stage 4 的判据必须以实测读地板为准（见 §3）。

### 2.2 设计层天花板核对（[DESIGN_LIMIT] 判定，未触发）

`perf-feedback.md` §1 双门槛：① 天花板归因到设计层假设；② 结构性加速 >2x 或与设计假设直接矛盾。
实测：三个 workload 的 pure-read 地板分别为 12.48 / 6.94 / 1.77 µs（§3），baseline 距地板仅 15% / 10% / 68%，
而 4k 的 68% 是**kernel 启动/写回地板**（1.77 µs 的空 kernel 实测），不是算法选择造成。没有任何一条
设计假设被实测推翻（f16 载体、指示平面、双路径分派都在预算内成立），结构性加速上限 <2x ⇒ **不产
`perf_feedback.md`，不附 `[DESIGN_LIMIT]`**。

---

## 3. 结构标定（先定标，再出候选；数据独立落 `copy_floor_calibration.jsonl` / `op_cost_calibration.jsonl`）

探针文件 `probe_floor.py`（**不是** tune workload，不进 `perf_records.jsonl`），按 baseline 的 path-C 结构
（40 核 persistent + `T.serial` 任务循环 + 列 tile）逐级裁剪链条：

| probe（mode） | shape / 配置 | Task Duration(us) mean | median | MTE2 | 说明 |
|---|---|---:|---:|---:|---|
| 3d_empty | 128×16384, tn=8192, bm=1 | 6.1693 | 6.18 | 0.002 | 只做每任务 1 B 写回（160 次小写）；MTE3 0.89 → **写回链是地板**，不与读相加 |
| 3d_read | 同上 | 12.4767 | 12.46 | 0.922 | **纯读地板**（8.39 MB → 672 GB/s，16.8 GB/s/核） |
| 3d_read（第二 session） | 同上 | 12.532 | 12.54 | 0.920 | 复现性 0.4% |
| 3d_read t=16384 | tn=16384 | 12.3513 | 12.36 | 0.918 | 更宽 tile 仅 −1.0% |
| 3d_read t=4096 | tn=4096 | 13.9433 | 13.84 | 0.926 | 更窄 tile **+11.8%** |
| 3d_read static-extent | 静态行列范围（无 `T.min`） | 14.944 | 14.92 | 0.939 | **假设被证伪**：动态 guard 不是读路径成本 |
| 3d_ind_vmin (f32) | vcmp→vcast→vmin 平面 + 1 次 reduce | 14.6493 | 14.64 | 0.615 | 链开销 = +2.17 µs |
| 3d_vmin_f16 | 同上，f16 平面 | 13.9853 | 14.00 | 0.669 | 链开销 = +1.51 µs（**−4.5%**） |
| 32k_read | 32×32768, tn=8192 | 6.9427 | 6.92 | 0.876 | 纯读地板 604 GB/s（18.9 GB/s/核 × 32 核） |
| 32k_read t=16384 | tn=16384 | 6.838 | 6.84 | 0.868 | −1.5% |
| 32k_read t=4096 | tn=4096 | 7.4173 | 7.42 | 0.884 | +6.8% |
| 4k_empty | 32×4096, bm=2, 16 blocks | 1.7713 | 1.76 | 0.004 | **启动/写回地板 1.77 µs**（= baseline 的 59%） |
| 4k_ind_vmin | 同上 | 3.1507 | 3.16 | 0.246 | vec 0.45 → 4k 是向量/延迟受限，不是带宽受限 |

**由此确定的判据**（`op_cost_calibration.jsonl` 逐条量化）：

1. 3d/32k 的 `Task ≤ 读地板 + 未重叠链`。读地板占 baseline 的 **84.8% / 89.8%** ⇒ 全部结构候选的上限只有
   15% / 10%，且任何 <5% 差异必须走 A/B 交错协议（T-2）。
2. 4k 的 `Task = 1.77 µs 地板 + 约 1.2 µs 工作`（其中 zero 平面填充量 = 读量 = 32 KB/block）⇒ 只能削工作项，
   不能削地板；削 bm（更多 block）无效（见 §4 已测）。
3. 列 tile 宽度 **≥8192 已饱和**（8192 与 16384 差 1%），**<8192 会劣化**（4096 +7~12%）⇒ tile 加宽类候选
   只能指望"减少每元素归约 pass"，不能指望 DMA。

**已证伪的假设（留痕）**：① "动态 `real_m = T.min(...)` 打断 burst 读" — 静态 extent 版本反而慢 20%
（12.48 → 14.94）；② "4k 用 bm=1 铺满 32 核" — `--block-m 1` 实测 3.001 µs vs bm=2 的 2.986 µs，无收益
（latency-bound，非并行度不足）；③ "reduce pass 是主开销" — 每 tile reduce 与整平面 vmin+单次 reduce 等价
（baseline 14.716 vs ind_vmin 14.649）。

---

## 4. Iteration 1 — 指示链载体 dtype + 容量路径（3 分支）

### 4.1 Diagnostic Context

- kernel_id: `_logical_reduce_kernel::main`；current best: `_logical_reduce_kernel.py`（baseline，§2 数字）
- Profile Facts（3d）：Task Duration 14.716 µs；AICore(vec) 0.326 / MTE2 0.561 / MTE3 0.061；读地板 12.477 µs（MTE2 0.922）
- Theory Estimate：8 388 736 B ÷ 672 GB/s（实测读天花板）= 12.48 µs；链开销 2.24 µs 未被 MTE2 隐藏
- Code Structure：单 TileLang kernel；path R 1 tile/行块，path C `bm=1` + `num_full` 静态列 tile + no-alias
  running `vmin`；persistent `num_kernels=min(num_logical,40)` + `T.serial`（3d：4 任务/核，4 vs 3 不均衡）；
  片上 buffer：`x_ub / zero_ub / mask(bool) / ind(f32)` 四平面 + `(bm,1)` 微累加器；尾块为独立 buffer 组
- Known Iteration Conclusions：无（首轮）

### 4.2 Current Phenomena

- **P1（3d）**：baseline 比纯读地板慢 2.24 µs（15%）；mte2 从 0.922 掉到 0.561 ⇒ DMA 在链计算期间空转 36%，
  即"链未被隐藏"，而 read 探针已证明搬运本身无问题。
- **P2（32k）**：baseline 比读地板慢 0.79 µs（10%）；且 M=32/bm=1 只用 32/40 核（20% 核空闲）。
- **P3（4k）**：baseline 2.983 µs，空 kernel 地板 1.771 µs（59%）；vec 0.385 / scalar 0.32 / mte2 0.28 ⇒
  latency-bound；prologue 的 `zero_ub` 填充量（bm·N·4 B = 32 KB/block）与真实读量相同。
- **P4（全）**：指示链为 `vcmp(eq) → vcast(i1→f32) → reduce_min` + 3 跳 epilogue cast；指示量只取 0/1，
  f32 载体在信息量上是浪费（每元素多 2 B 的写/读 + 多一跳 cast）。

### 4.3 Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP1 | P4（+P1/P3） | 指示链载体 f32→f16（R+C 无尾块路径） | 0/1 在 f16 精确可表示；`vcast` 表有 i1→f16 与 f16→i8；3d 探针 f16 平面 −4.5% | `IND/ACC/one/any` 全 f16，epilogue 单跳 f16→i8；尾块保持 f32 | Task Duration / 逐位精度 | improved(4k)/tie(C) |
| OP2 | P2（+P1） | OP1 + 列 tile 上限 8192→16384（f16 腾出的容量） | 每元素归约单价在 N≥8192 档饱和（8192 与 16384 同价）⇒ 加宽可摊薄 | `TILE_CAP_ELEMS = {f16:16384, f32:16384, bf16:8192}` | 同上 | config_no_gain |
| OP3 | P1 | OP1 + 延迟归约（每 tile 元素级 `vmin` 入平面，末尾 1 次 reduce） | 把 `num_full` 次 reduce 换成 1 次 | 新增 `acc/acc_new` 平面（f16） | 同上 | config_no_gain |

预筛（不进分支、只做探针/参数覆盖）：bm=1 铺核（`--block-m` 实测无收益）、静态 extent（探针证伪）、
tile 4096（读地板 +7~12%）——均不立分支。

### 4.4 Experiment Branches

| branch | base | opt_id | major_change | L0 | Task Duration 4k / 32k / 3d (us) | profile | result |
|---|---|---|---|---|---|---|---|
| `_logical_reduce_kernel_round1_op1_f16plane.py` | baseline | OP1 | 指示/累加/epilogue f16 | PASS | 2.828 / 8.0733 / 14.2833 | valid | 4k improved，C tie |
| `_logical_reduce_kernel_round1_op2_f16_tile16k.py` | OP1 | OP2 | 列 tile 16384（f16 容量） | PASS | 2.8467 / 8.0907 / 14.0113 | valid | config_no_gain |
| `_logical_reduce_kernel_round1_op3_f16_defer.py` | OP1 | OP3 | 延迟归约 vmin 平面 | PASS | 2.878 / 7.8167 / 14.9627 | valid | config_no_gain |

> OP2 需同步更新 L0/L1 的配置断言（`expect_tile_n` / `(num_full, tail)`）——这是**配置阶梯**的期望值，
> 不是精度断言；若不更新会被回归门禁误报为精度失败（见 §9 R-2）。

### 4.5 候选 vs current best 对比表（B2 结构化回流）

current best = `_logical_reduce_kernel.py`（baseline）。数据来自 `perf_records.jsonl` + 对应 raw profile。

| workload_id | candidate | Task Duration(us) | vs current best | AICore(vec) | MTE2 | MTE3 | 内存/UB 观察 | L0 |
|---|---|---:|---:|---:|---:|---:|---|---|
| mask-validation-4k-bool | **current best (baseline)** | 2.9833 | — | 0.385 | 0.281 | 0.111 | ind 平面 f32 16 KB/block | PASS |
| mask-validation-4k-bool | op1_f16plane | 2.8280 | **+5.2%** | 0.322 | 0.295 | 0.130 | ind 平面减半至 8 KB/block | PASS |
| mask-validation-4k-bool | op2_f16_tile16k | 2.8467 | +4.6% | 0.325 | 0.303 | 0.108 | 同 op1（R 路径不受 tile 上限影响） | PASS |
| mask-validation-4k-bool | op3_f16_defer | 2.8780 | +3.5% | — | — | — | 追加一个 8 KB acc 平面 | PASS |
| mask-validation-32k-bool | **current best (baseline)** | 7.7300 | — | 0.370 | 0.459 | 0.033 | 88 KB/block 工作集 | PASS |
| mask-validation-32k-bool | op1_f16plane | 8.0733 | −4.4% | 0.275 | 0.571 | 0.034 | ind 减半 → UB 余量↑，但时间未降 | PASS |
| mask-validation-32k-bool | op2_f16_tile16k | 8.0907 | −4.7% | 0.284 | 0.583 | 0.026 | tile 16384：2 tile/行 | PASS |
| mask-validation-32k-bool | op3_f16_defer | 7.8167 | −1.1% | 0.256 | 0.635 | 0.053 | acc 平面 16 KB 额外 | PASS |
| 3d-multidim-reduce-bool | **current best (baseline)** | 14.7160 | — | 0.326 | 0.561 | 0.061 | 88 KB/block | PASS |
| 3d-multidim-reduce-bool | op1_f16plane | 14.2833 | +2.9% | 0.253 | 0.634 | 0.053 | 同上 | PASS |
| 3d-multidim-reduce-bool | op2_f16_tile16k | 14.0113 | +4.8% | 0.251 | 0.656 | 0.064 | 单 tile 16384（无 tile 循环） | PASS |
| 3d-multidim-reduce-bool | op3_f16_defer | 14.9627 | −1.7% | 0.259 | 0.667 | 0.092 | acc 平面 32 KB 额外 | PASS |

**A/B 裁决（<5% 差异必经，`ab_test.py` 交错 3 对，launch-count 15）**

| 对比 | workload | A 合并中位 | B 合并中位 | 相对变化 | 配对方向 | p | verdict |
|---|---|---:|---:|---:|---|---:|---|
| baseline vs op1 | 4k | 2.90 | 2.78 | **−4.14%** | 3/3 一致 | 0.25 | **B_better** |
| baseline vs op1 | 32k | 8.18 | 7.94 | −2.93% | 2/3 | 1.0 | tie |
| baseline vs op2 | 3d | 14.88 | 15.10 | +1.48% | 1/3 | 1.0 | tie |

结论：**只有 4k 的 f16 载体是有效增益**（−4.1%，3/3 同向）；C 路径上 OP1/OP2 的表观 ±5% 全部落在 run-state
双态内（同 session 内 baseline 8.18 vs 7.86 的漂移即 >3%）。OP2/OP3 记 `config_no_gain`。

### 4.6 合并检查 / winner

本轮不做跨 workload 合并（4k 的 winner 是 OP1 的 R 部分，C 部分无 winner）。current best 记录为
"R=f16（OP1）、C=baseline"，在 round 4 落地成 `_logical_reduce_kernel.py` 并做全量复测（§7）。

- winner（4k）：`op1_f16plane` 的 resident 段；rollback：op2/op3（config_no_gain）
- smoke 精度（合并版）：见 §7.3

---

## 5. Iteration 2 — 列 tile 循环软件流水（1 分支）

### 5.1 现象与候选

P1/P2 的剩余解释是"DMA 与链不重叠"。唯一未测的显式手段是 `T.Pipelined`。

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 |
|---|---|---|---|---|
| R2-OP1 | P1/P2 链未隐藏 | 列 tile 循环 `T.Pipelined(num_full, num_stages=2)` | 显式流水与 auto-multi-buffer 是否等价未知 | 只改循环头 |

### 5.2 候选 vs current best 对比表

current best（C 路径）= baseline。

| workload_id | candidate | Task Duration(us) | vs current best | AICore(vec) | MTE2 | MTE3 | L0 |
|---|---|---:|---:|---:|---:|---:|---|
| mask-validation-32k-bool | current best (baseline, C) | 7.7300 | — | 0.370 | 0.459 | 0.033 | PASS |
| mask-validation-32k-bool | r2_op1_f16_pipe2 | 7.8540 | −1.6% | 0.289 | 0.555 | 0.030 | PASS |
| 3d-multidim-reduce-bool | current best (baseline, C) | 14.7160 | — | 0.326 | 0.561 | 0.061 | PASS |
| 3d-multidim-reduce-bool | r2_op1_f16_pipe2 | 14.2793 | +3.0% | 0.253 | 0.634 | 0.095 | PASS |

A/B（op1 vs pipe2，3d）：A 14.28 / B 14.26 / **−0.14%** / 1:2 方向 / p=1.0 → **tie**。
机制判读：`T.Pipelined` 后 `mte2`/`vec` 与 auto-multi-buffer 版本逐项相同 ⇒ 编译器已在做同等双缓冲，
瓶颈是**链与 DMA 的串行依赖**（tile 的 compare 必须等 copy 完成），不是缓冲深度。记 `config_no_gain`。

---

## 6. Iteration 3 — 静态展开 + 外层任务流水（2 分支）

### 6.1 候选 vs current best 对比表

current best（C 路径）= baseline。

| workload_id | candidate | Task Duration(us) | vs current best | AICore(vec) | MTE2 | MTE3 | L0 |
|---|---|---:|---:|---:|---:|---:|---|
| mask-validation-32k-bool | current best (baseline, C) | 7.7300 | — | 0.370 | 0.459 | 0.033 | PASS |
| mask-validation-32k-bool | r3_op1_f16_unroll（`T.unroll` tile 循环） | 8.0407 | −4.0% | — | — | — | PASS |
| 3d-multidim-reduce-bool | current best (baseline, C) | 14.7160 | — | 0.326 | 0.561 | 0.061 | PASS |
| 3d-multidim-reduce-bool | r3_op1_f16_unroll | 14.7847 | −0.5% | 0.255 | 0.632 | 0.056 | PASS |
| 3d-multidim-reduce-bool | r3_op2_f16_taskpipe（外层 persistent 任务循环 2 级流水） | 14.2900 | +2.9% | 0.254 | 0.632 | 0.086 | PASS |

两者相对 baseline 与相对 op1 的差异都 ≤3%（<噪声），且与 op1（14.28）逐位同量级 ⇒ `config_no_gain`。
**C 路径到此已覆盖四类结构杠杆**（载体 dtype、tile 宽度、归约组织、循环调度），全部落在 ±3% 噪声带内；
按 T-1 做参照结构 diff 检查：姊妹 `all` 任务在同一 910B4 上的最终形态是"resident=f16 / column-tiled=baseline"，
`count_nonzero` 的最终形态亦为"resident 驻留档 + C 路径保持"，与本轮独立得出的结论同向 ⇒ 无未试的结构差异，
允许进入收束（不是仅凭"没想法"）。

---

## 7. Iteration 4 — 合并版落地与全量复测

### 7.1 合并形态

`perf_opt/_logical_reduce_kernel.py` = **R 路径 f16 载体（OP1 的已验证部分）+ C 路径与尾块保持 Stage-3 f32**
（C 路径无有效增益，"更小改动面"胜出；尾块保 f32 亦规避窄尾块 f16 mis-reduce，VP-2026-0326）。
模块级常量 `IND_DTYPE = "float16"` 被 R 路径默认使用，取值依据写在常量注释与文件头 docstring 中。

### 7.2 候选 vs current best 对比表（合并版全量）

current best = baseline（round-0）。`merged` = 合并版（sha `bb552517…`）。

| workload_id | candidate | Task Duration(us) | vs current best | AICore(vec) | MTE2 | MTE3 | L0 |
|---|---|---:|---:|---:|---:|---:|---|
| mask-validation-4k-bool | current best (baseline) | 2.9833 | — | 0.385 | 0.281 | 0.111 | PASS |
| mask-validation-4k-bool | merged_f16res | 2.7820 | **+6.7%** | 0.323 | 0.299 | 0.122 | PASS |
| mask-validation-32k-bool | current best (baseline) | 7.7300 | — | 0.370 | 0.459 | 0.033 | PASS |
| mask-validation-32k-bool | merged_f16res | 8.2453 | −6.7% | 0.343 | 0.498 | 0.031 | PASS |
| 3d-multidim-reduce-bool | current best (baseline) | 14.7160 | — | 0.326 | 0.561 | 0.061 | PASS |
| 3d-multidim-reduce-bool | merged_f16res | 14.4840 | +1.6% | 0.334 | 0.555 | 0.056 | PASS |

**32k 的 −6.7% 是 run-state，不是回退**：合并版对这两个 shape 的**lowered MLIR 摘要与 baseline 逐字节相同**
（`codegen_compare.py`：(32,32768) sha equal / 4318 B；(128,16384) sha equal / 4976 B），设备代码不可能不同。
同 session 的 `baseline_recheck`（sha `be23433d…`）给出 2.9587 / 8.1547 / 15.0513 —— 即本 session 整机比
round 0 慢 5.5%（32k）/ 2.3%（3d），足以吞掉全部表观差异。因此最终判据用**同 session 交错的 A/B**（§7.4）。

### 7.3 精度回归（阻塞项）

| 检查 | 结果 |
|---|---|
| `python perf_opt/_logical_reduce_kernel.py --level all`（交付文件） | **exit 0**；`[L0] ALL PASS: 26`、`[L1] ALL PASS: 26`、L2 全 PASS、Boundary 全 PASS、`exact bool comparisons=101 mismatching entries=0`（日志 `logs/final/level_all_postcleanup.log`） |
| 每轮候选 L0 | 7 个候选全部 PASS（`logs/round{1,2,3,4}/L0_*.log`） |
| smoke-any-bool | PASS（`bench_driver --check`，torch.equal 逐位精确） |
| smoke-any-float16 | PASS |
| smoke-any-float32 | PASS |
| 语义红线自检 | any = `1 − min[x==0]`；`reduce_min`/`vmin` no-alias；累加器单位元 **1.0**；`vcmp` 用 `"eq"`；`op_kind != "any"` 显式 `raise ValueError` |

### 7.4 最终 A/B 裁决（交付文件 vs baseline，交错 3 对，launch-count 15）

| workload | A=baseline 合并中位 | B=交付版 合并中位 | 相对变化 | 配对方向 | p | verdict |
|---|---:|---:|---:|---|---:|---|
| mask-validation-4k-bool（交付版，4 对，`profiles/ab/ab_delivered/4k`） | 2.96 | 2.82 | **−4.73%** | **4/4 一致** | 0.125 | **B_better** |
| mask-validation-4k-bool（同码前一 session，3 对） | 2.96 | 2.78 | **−6.08%** | 3/3 一致 | 0.25 | **B_better** |
| mask-validation-32k-bool | 8.06 | 8.18 | +1.49% | 1/3 | 1.0 | tie |
| 3d-multidim-reduce-bool | 14.68 | 14.80 | +0.82% | 1/3 | 1.0 | tie |

> 两次 4k A/B 互为复现（−4.73% / −6.08%，均为 100% 同向）；交付版与 `round1_op1_f16plane` 在 (32,4096) 与
> (32,256) 上的 MLIR 摘要逐字节相同（3162 B / 3133 B），说明该增益来自 f16 载体本身而非调试残留。

---

## 8. Final Summary

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| _logical_reduce_kernel::main | mask-validation-4k-bool | 2.8300 | 2.7113 | final_cf16_r5 |
| _logical_reduce_kernel::main | mask-validation-32k-bool | 6.6450 | 6.1907 | final_cf16_r5 |
| _logical_reduce_kernel::main | 3d-multidim-reduce-bool | 10.8800 | 9.9767 | final_cf16_r5 |

**Round 5 更新（本表已按新 deliverable 重测，supersede 说明）**：交付文件在 round 5 由 `bb552517…`（C=f32）
换成 `5b11da03…`（C=f16 载体，round-5 A/B 采纳）。上表三行均为 round 5 在同一最终候选
（`final_cf16_r5`，`phase=final`，30 launch 的 mean）上重采的记录，`baseline_us` 取**同一 session** 的交错
A/B baseline 中位（`baseline_r5`，sha `be23433d…`），不再用 round-0/910B4 的历史 baseline 作分母——本机
（910B3 / `719c6da`）与 round-0（910B4 / `e21da32`）不可通约。与 `perf_records.jsonl` 的对应行逐位一致（偏差 0%）。
旧的 round-4 final 行（`final_f16res`，sha `bb552517…`，2.7820 / 8.2453 / 14.4840）已移入
`perf_records_full_history.jsonl`（只追加、不删除）。

同 session 交错 A/B（§10.5，判据用）：4k **−5.30%**（6/6，p=0.031）/ 32k **−7.91%**（4/4，round-5 C 载体增益）/
3d **−9.21%**（10/10，p=0.002；第二 session **−8.68%**，10/10）。

**final 覆盖说明（supersede，如实披露）**：第一次 `phase=final` 采集（sha `300fa9dc…`，2.8067 / 8.0633 / 15.1633）
之后，为满足"tuned 常量必须被默认路径实际使用"，删除了未被引用的 `ACC_DTYPE` 常量并同步 docstring（纯注释/常量
清理，R 路径 trace 出的 dtype 不变：与 op1 的 MLIR 摘要逐字节相同）。文件内容变化使旧 final 行失效，处置：
① 3 条旧 final 行原样保存在 `perf_records_full_history.jsonl`（该文件含全部 26 条原始测量）；
② 主 `perf_records.jsonl` 保留 baseline/candidate/recheck 行 + 每 tune workload **唯一**权威 final 行（23 行）；
③ `workload_inventory.json` 的 `final_artifact_sha256` 与本表 final 行一致。

### 8.1 Ratio 与族验收线（分母显式标注）

> **Round 5 后本表数值已被 §10.6 取代**（交付文件换为 C 路径 f16 载体，且设备/工具链从 910B4/`e21da32`
> 变为 910B3/`719c6da`）。本节保留原文作为历史记录，判据请读 §10.5 / §10.6。

字节口径 = kernel 实际（读 4 B/元素 + 写 1 B/元素）。两套分母：report 的 **1.75 TB/s**（fabric 聚合口径）
与设备自报 **0.80 TB/s**（AIV 侧单向 GM 流量口径，本机 `visualize_data.bin` GM Read+Write 实测 0.8 TB/s）。
族验收线 = `0.7 × 0.80 TB/s = 560 GB/s`。

| workload | GM 字节 | baseline 带宽 | final 带宽 | Ratio vs 1.75 TB/s（base→final） | Ratio vs 0.80 TB/s（base→final） | 560 GB/s 线 | 读地板 | 判定 |
|---|---:|---:|---:|---|---|---|---:|---|
| 4k | 524 320 | 175.8 GB/s | **188.5 GB/s** | 10.0% → 10.8% | 22.0% → 23.6% | —（latency-bound） | 1.77 µs 启动地板（59% of baseline） | **不可达 560**：空 kernel 地板 1.77 µs 对应上限 296 GB/s，结构性不可达；按 `Task Duration` 判据看，4k 距地板仅 1.0 µs（A/B −6.1%） |
| 32k | 4 194 336 | 542.7 GB/s | 508.7 GB/s（mean）/ 512.7（A/B 中位） | 31.0% → 29.1% | 67.8% → 63.6% | 96.9% → 90.8%（A/B 中位 91.6%） | 6.94 µs = 604 GB/s | **未达线但同码同性能**：A/B tie；读地板已达线的 108%，缺口是未隐藏的链（10%） |
| 3d | 8 388 736 | 569.6 GB/s | 579.0 GB/s | 32.5% → 33.1% | 71.2% → 72.4% | 101.7% → **103.4%**（A/B 中位 98.7%→97.9%） | 12.48 µs = 672 GB/s | **达线**（baseline 与 final 都在 560 以上） |

> 32k/3d 的 `final_us` 落在慢态时表观带宽低于 baseline，但其设备代码与 baseline **逐字节相同**（§7.2），
> 差异纯属 run-state；A/B（同 session 交错）判 tie，故取"无性能变化"为结论。
> Ratio 口径提醒（`profile-collection.md` §4.1）：本族是"纯 Vector + 大 N 读主导"，Ratio 反映的是比较/归约 op 的
> 记账密度，不是硬件利用率；判据用流量达成率，不用 Ratio 硬目标。

### 8.2 有效优化点汇总

| 优化点 | 作用域 | 实测 | 证据 |
|---|---|---|---|
| 指示链载体 f32 → f16（并省掉一跳 epilogue cast） | path R（`N ≤ E_CAP`） | **−4.1%（op1）/ −6.1%（交付版）**，A/B 3/3 同向 | §4.4/§7.4、`perf_records.jsonl`、`profiles/*/r1_op1_f16plane`、`profiles/final/*` |
| 列 tile 16384 | path C | tie（+1.0%，3/3 反向） | §4.5 A/B |
| 延迟归约（vmin 平面） | path C | tie | §4.5 |
| `T.Pipelined`（tile / 任务循环）、`T.unroll` | path C | tie（≤3%） | §5/§6 |
| 静态 copy extent | path C | **更差 20%**（探针） | `op_cost_calibration.jsonl` static_extent_scan |
| bm=1 铺满 32 核 | path R（4k） | 无收益（3.001 vs 2.986 µs） | `logs/round1/bm1_4k.log` |

### 8.3 已知残留（如实披露）

1. `probe_floor.py --mode reduce` / `--mode ind_red` 在本工具链上运行期 core dump（未定位；两者结果未参与任何决策，
   只用 `empty/read/ind_vmin/vmin_f16` 四条有效探针）。
2. `profiles/ab/` 下保留 A/B 的 msprof 原始输出（约 1.2 GB，非 gate 引用对象）；`profiles/` 共 1.3 GB，
   `perf_records.jsonl` 的 `msprof_raw_path` 全部指向存在的目录。
3. `perf_opt/` 顶层含辅助脚本（`bench_driver.py` / `run_bench.py` / `run_experiments.py` / `probe_floor.py` /
   `probe_run.py` / `make_candidates.py` / `codegen_compare.py`），属 skill 允许的 runner/helper。
4. 交付版 path C 与尾块未做任何性能改动（有意为之：无有效增益 + 更小改动面）。

---

## 9. Skill Retrospective

### 9.1 流程问题与改进

1. **"先定标再出候选"在本任务直接省掉了 3 个错误分支**：读地板探针（12.48/6.94/1.77 µs）一出来就排除了
   "DMA 粒度/静态 extent/tile 加宽" 三个方向（其中静态 extent 实测反而 −20%），把预算集中到载体 dtype 与
   归约组织上。建议把"read/empty 双探针"写成本族（行归约）首轮的**必做前置**，而不是可选。
2. **A/B 命令必须指向 bench entry**：本次第一次 A/B 把 `--a-cmd` 的第一个 token 写成了 kernel 文件本身，结果
   `msprof` 采到的是该文件 `--level L0` 里的一堆 kernel（两个 arm 的合并中位都是 2.58 µs、verdict=tie），
   **看起来完全合理**。这是"误配置产生貌似可信数据"的高危失败模式（详见 R-1）。
3. **配置阶梯改动会撞上 L0/L1 的配置期望**：`expect_tile_n=8192` 这类断言使"改容量策略"的分支被误报为精度失败
   （op2 第一次 L0 FAIL）。测试里的配置期望应显式标注 `CONFIG-EXPECTATION` 以便与精度断言区分（R-2）。
4. **run-state 漂移在本机达 ±6%**（同 shape 跨 session：32k 7.73→8.25），比 skill 默认的 3% 噪声阈值大一倍。
   本任务全程用 A/B 兜底是对的；建议把"跨 session 漂移量级"也记进 `baseline_recheck` 行，作为 report 的必读字段。

### 9.2 Value point proposals（vp_type + 证据三件套）

| id | vp_type | 结论（一句话） | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-ANY-1 | D | 910B4 上 any/all 类行归约的**纯读地板**：3d（8.39 MB，tn=8192）= 12.48 µs（672 GB/s；16.8 GB/s/核 × 40），32k（4.19 MB）= 6.94 µs（604 GB/s；18.9 GB/s/核 × 32）；tile 4096 分别 +11.8% / +6.8%，tile 16384 仅 −1%，⇒ **列 tile ≤4096 是劣化档，≥8192 饱和**，加宽 tile 只能从"每元素归约 pass"取收益 | 本文件 §3 表 + `copy_floor_calibration.jsonl`（30 launch mean，两次 session 复现 0.4%） | `perf_opt/probe_floor.py` + `probe_run.py --only 3d_read,32k_read,3d_read_t4k,32k_read_t16k` | tilelang `0.1.2+e21da325`（`e21da32`）/ CANN 9.0.0 / 910B4 |
| VP-ANY-2 | D | 谓词行归约的**载体 dtype** 杠杆按路径分裂：resident（`N ≤ E_CAP`）f16 指示/累加载体 **−4.1%~−6.1%**（A/B 3/3）；column-tiled（`N > E_CAP`）**无收益**（四种结构变体全部 ±3% 内）——即"容量富余度"决定该杠杆是否成立：resident 的 UB 填充量≈读量，f16 直接省掉一跳 epilogue cast 与半宽 pass；C 路径的 DMA 是地板，省下的 pass 被掩盖 | §4.5 / §7.4 的 A/B 与原值表；`perf_records.jsonl` 全 23 行 | `perf_opt/make_candidates.py --spec f16plane`（慢→快最小 diff：`ind/mn/one_v/any_v` f32→f16 且删 `any_f16` 跳） | 同上 |
| VP-ANY-3 | P | 行归约的候选生成顺序（按收益/成本排序）：①先测 read/empty 地板定上限 → ②若 `地板/baseline ≥ 0.85` 则只做单点 dtype/组织变体并全部走 A/B → ③若 `<0.65`（如本任务 4k=0.59）转向削 prologue（zero 平面填充量=读量）与 epilogue 跳数，不要调并行度 | §3 判据 1/2 与 §8.2 汇总（6 条方向中 4 条被探针提前否决） | `perf_opt/probe_floor.py`（read/empty/vmin_f16 三档）+ `run_experiments.py` | 同上 |
| VP-ANY-4 | R | `ab_test.py` 的 arm 命令必须以 bench entry 开头（`python bench_driver.py --impl ...`），不能是 kernel 文件本身：后者会静默跑 kernel 自带的 `--level L0` 全套用例，`msprof` 采到多个同名 kernel，仍能产出结构完整的 JSON（本次两 arm 都是 2.58 µs、verdict=tie）。**校验手法**：arm 的 `--launches` 应等于 `warm-up+launch-count`，且 `a_medians_us` 的量级应与该 workload 的 `perf_records` 行同量级（偏离 >2× 即命令写错） | 本次 first A/B：`a_medians=[2.58,2.58,2.54]`、`b_medians=[2.56,2.58,2.62]`（真值应为 2.9/2.78）；重跑后才得到 −4.14% | 无需 repro（协议误用）；对照 `perf_opt/ab_round1/` 与 `logs/round1/bm*_4k.log` | 同上 |
| VP-ANY-5 | R | 容量/阶梯类候选会触发 L0/L1 的**配置期望断言**（`expect_tile_n` / `(num_full, tail)`），必须先同步这些期望再跑回归，否则"精度失败"是假阳性（本次 op2 在 L0-3a/3d/3b 连续三次误报）。建议测试内把这类断言统一命名为 `expect_*` 并在日志里标 `CONFIG` 前缀 | `logs/round1/L0_*op2*` 三次 FAIL 的 traceback（均为 tile_n 期望不符，非数值不符） | `perf_opt/make_candidates.py` 的 `f16_tile16k` spec + 后续 patch 脚本 | 同上 |
| VP-ANY-6 | C | 迁移算子"父工厂四反转"审计的可复用检查单：any 的 `reduce_min` / `1−min` / `vmin` / 单位元 `1.0` 与 all 的 `reduce_max` / `1−max` / `vmax` / `0.0` 四项全对偶；本任务交付版仍以 `vcmp("eq")` + `min` 落地，L0-2 的 "NaN+zeros 只有零" 行是唯一能暴露 `"ne"` 误用的判别用例 | §7.3 精度表（101 次逐位比较 0 mismatch）+ L0-2a/b/c | `perf_opt/_logical_reduce_kernel.py --level all` | 同上 |

### 9.3 BP proposals

- **BP-candidate-1（定标前置）**：`tilelang-op-optimize` Phase 1 之后、Phase 2 之前增加"read/empty 双探针"强制步骤，
  并以 `地板/baseline` 比值分档（≥0.85 → 只做单点变体 + A/B；≤0.65 → 削 prologue/epilogue），
  本任务据此在 7 个分支内收束，避免了对 4 类已证伪方向的投入。
- **BP-candidate-2（A/B 命令自检）**：`ab_test.py` 增加 arm 命令合法性校验（可执行文件必须是 bench entry /
  `--launches` 与 launch-count 的比例校验 / 与 `perf_records` 同量级比对），把 VP-ANY-4 的失败模式机械化拦下。
- **BP-candidate-3（配置期望标记）**：Stage 3 生成的测试模板中，把"配置阶梯期望"（`expect_tile_n`、`(num_full, tail)`）
  与精度断言分离并加 `CONFIG` 标记，供 Stage 4 在改容量策略时批量更新，避免假阳性回归失败。

### 9.4 pattern-library 回写状态（受任务授权范围限制）

本轮产出的 D/P 类条目（VP-ANY-1/2/3）本应追加进
`references/pattern-library/constants.md`（读地板档位、tile 宽度饱和档）与 `layout.md`（`PL-1.25` 的
"载体 dtype 按 path 分裂"数据点：resident 有效 / column-tiled 无效，与 `all`/`count_nonzero` 的独立
结论同向，构成第三、四数据点）。**但本任务的交付授权明确限定"只写 `perf_opt/`"**，故上述条目以
本表形式留档、未直接改 pattern-library；建议由终态 `tilelang-skill-evolver` 蒸馏时按 Tier 0 合入，
repro 直接引用 `perf_opt/probe_floor.py` 与 `perf_opt/make_candidates.py`（已在本机跑通）。

---

## 10. Round 5 — 3d f16 A/B stabilization + 32k overlap probe + net-gain recheck

> 本节为承接轮（round 5）追加，不改写 §1–§9 的历史记录。**判据唯一**：`msprof op` `Task Duration(us)`，
> 30 launch、warm-up 5、同 session 交错 A/B 多 run、merged median + 配对符号检验。

### 10.1 诊断上下文与方法（设备/工具链已更换，旧数字不可通约）

| 项 | round 0–4（历史） | round 5（本机实测） |
|---|---|---|
| 设备 | Ascend910B4（AIV=40 / UB 196608） | **Ascend910B3**（AIV=40 / UB 196608，`npu-smi` 1×910B3） |
| tilelang | `0.1.2+e21da32` / `e21da32` | **`0.1.2+719c6da62061f8f7984ac9f87012ca0319b82020`** |
| CANN / torch | 9.0.0 / 2.9.0 | 9.0.0（`npu-smi 26.1.1`）/ torch 2.9.0+cpu + torch_npu 2.9.0.post8 |
| 交付文件 | `bb552517…`（R=f16，C=f32） | `bb552517…` → **`5b11da03…`**（C=f16，采纳后） |

**本轮唯一新候选**：`_logical_reduce_kernel_round5_op1_cf16.py`（sha `a2fe7d6e…`），由
`make_candidates_round5.py` 从 current best **脚本生成**（非手改），改动面 = `_build_tiled()`
（path C，`tail == 0`）一个函数体内的载体常数：`ind / mn / acc / new_acc / acc_flat / one_v / any_v`
统一为 `C_IND_DTYPE = "float16"`，A 路径与静态尾块（`_build_tiled_tail`，VP-2026-0326）不动。

**单点改动的代码级证据**（`codegen_compare.py`，lowered MLIR sha）：

| shape | path/config | current best (`bb552517`) | candidate (`a2fe7d6e`) | 结论 |
|---|---|---|---|---|
| (128,16384) 3d | C/1/8192 | 4976 B | 4836 B | **不同**（C 路径被改） |
| (32,32768) 32k | C/1/8192 | 4318 B | 4184 B | **不同**（C 路径被改） |
| (32,4096) 4k | R/2/4096 | 3162 B | 3162 B | **逐字节相同**（R 路径未动） |

采纳后交付文件的 doc 更新（叙述 + 常数注释）由同一脚本产出，并与被 A/B 实测的候选逐 shape 对账：
`codegen_compare --a <candidate> --b <deliverable>` → **三个 shape 的 MLIR sha 全部相同**
（4184 / 4836 / 3162 B），即交付文件的可执行代码与实测候选逐字节等同，只有注释不同
（`logs/round5/codegen_deliverable_vs_candidate.log`）。

**协议**：`python3 .agents/tools/ab_test.py --pairs N --launch-count 30 --warm-up 5 --kernel-name main`，
arme = `python bench_driver.py --impl <file> --case <wl> --launches 35`（VP-ANY-4 自检：launch 数 = warm-up +
launch-count）。原始 profile：`profiles/round5/ab_3d/{s1,s2,guard32k}/pair{1..10}{A,B}/`、`profiles/round5/t3/*`。

### 10.2 候选 vs current best 对比表（B2 结构化回流）

数据来自 `perf_records.jsonl` round 5 行 + 对应 raw profile（`extract_metrics.py`，30 launch 的 per-block 均值）。
3d / 32k 为 A/B 首个 pair 的原始 dir（pair 内两臂同环境、交替顺序）。

| kernel_id | workload | 候选 | Task Duration(us) | AICore(vec) 利用率 | MTE2 忙时(us) / 占比 | MTE3 忙时(us) | scalar(us) | L0 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `_logical_reduce_kernel::main` | 3d (128,16384) | current best（C=f32） | 10.790 | 0.4376（3.419 us） | 3.203 / 0.4096 | 0.906 | 1.140 | PASS |
| `_logical_reduce_kernel::main` | 3d (128,16384) | **r5_op1_cf16** | **9.840** | 0.3686（**2.587 us**） | 3.232 / 0.4598 | 0.843 | 1.144 | PASS |
| `_logical_reduce_kernel::main` | 32k (32,32768) | current best（C=f32） | 6.640 | 0.4085（2.098 us） | 1.971 / 0.3831 | 0.208 | 0.935 | PASS |
| `_logical_reduce_kernel::main` | 32k (32,32768) | **r5_op1_cf16** | **6.090** | 0.3491（**1.619 us**） | 2.011 / 0.4322 | 0.225 | 0.885 | PASS |

机制（结构归因）：MTE2 忙时**基本不变**（3d +0.03 us、32k +0.04 us），而 **vec 忙时下降 24%**
（3d 3.419→2.587 = −0.832 us；32k 2.098→1.619 = −0.479 us），与 Task 降幅（−0.95 / −0.55 us）同量级。
即：指示平面宽度减半 → `vcast` 写 + `reduce_min` 读的向量侧流量减半；因为链完全串行（§10.4），
vec 侧省下的时间几乎 1:1 落到 Task 上。`AICore(vec)` 占比下降只是分母（aiv_time）变小，不是利用率变差。

### 10.3 T1 — 3d f16 C 载体的 ≥10 对交错 A/B（主判据）

A = current best `bb552517`（C=f32），B = `r5_op1_cf16`（`a2fe7d6e`）。

**Session 1**（`profiles/round5/ab_3d/s1`，10 对，交替顺序）：

| pair | A (us) | B (us) | Δ (us) | Δ (%) | 方向 |
|---:|---:|---:|---:|---:|:--|
| 1 | 10.790 | 9.840 | −0.950 | −8.80% | B 快 |
| 2 | 10.860 | 9.930 | −0.930 | −8.56% | B 快 |
| 3 | 10.840 | 10.060 | −0.780 | −7.20% | B 快 |
| 4 | 10.870 | 9.910 | −0.960 | −8.83% | B 快 |
| 5 | 10.870 | 9.860 | −1.010 | −9.29% | B 快 |
| 6 | 10.850 | 9.800 | −1.050 | −9.68% | B 快 |
| 7 | 10.820 | 9.890 | −0.930 | −8.60% | B 快 |
| 8 | 10.860 | 9.850 | −1.010 | −9.30% | B 快 |
| 9 | 10.840 | 9.820 | −1.020 | −9.41% | B 快 |
| 10 | 10.980 | 9.850 | −1.130 | −10.29% | B 快 |
| **merged** | **10.855** | **9.855** | **−1.000** | **−9.21%** | **10/10**，p = **0.002** |

**Session 2**（独立进程、与 session 1 相隔 ~25 min，`profiles/round5/ab_3d/s2`，10 对）：

| pair | A (us) | B (us) | Δ (us) | Δ (%) | 方向 |
|---:|---:|---:|---:|---:|:--|
| 1 | 10.970 | 9.900 | −1.070 | −9.75% | B 快 |
| 2 | 10.910 | 9.930 | −0.980 | −8.98% | B 快 |
| 3 | 10.830 | 9.880 | −0.950 | −8.77% | B 快 |
| 4 | 10.830 | 9.860 | −0.970 | −8.96% | B 快 |
| 5 | 10.860 | 9.890 | −0.970 | −8.93% | B 快 |
| 6 | 10.820 | 9.860 | −0.960 | −8.87% | B 快 |
| 7 | 10.860 | 9.870 | −0.990 | −9.12% | B 快 |
| 8 | 10.800 | 10.040 | −0.760 | −7.04% | B 快 |
| 9 | 10.830 | 10.070 | −0.760 | −7.02% | B 快 |
| 10 | 10.820 | 9.890 | −0.930 | −8.60% | B 快 |
| **merged** | **10.830** | **9.890** | **−0.940** | **−8.68%** | **10/10**，p = **0.002** |

**32k 回归护栏**（C 路径共享，`profiles/round5/ab_3d/guard32k`，4 对）：

| pair | A (us) | B (us) | Δ (%) | 方向 |
|---:|---:|---:|---:|:--|
| 1 | 6.640 | 6.090 | −8.28% | B 快 |
| 2 | 6.590 | 5.980 | −9.26% | B 快 |
| 3 | 6.640 | 6.150 | −7.38% | B 快 |
| 4 | 6.660 | 6.140 | −7.81% | B 快 |
| **merged** | **6.640** | **6.115** | **−7.91%** | **4/4**，p = 0.125（n=4 精确符号检验下限 0.125） |

**采纳判定（四项全中）**：

| 条件 | 阈值 | 实测 | 结论 |
|---|---|---|---|
| (a) 同向率 | ≥ 9/10 | 10/10（s1）、10/10（s2） | ✅ |
| (b) merged Δ | ≤ −3% | −9.21%（s1）、−8.68%（s2） | ✅（远超本机零效应噪声 ±0.6%，§10.5） |
| (c) 配对符号检验 | p < 0.05 | 0.002（s1）、0.002（s2） | ✅ |
| (d) 第二独立 session 同向 | 同号 | −8.68%，同号 | ✅ |
| 32k 护栏 | B 不得比 A 慢 >3% | **快 7.91%** | ✅（顺带把 32k 也提了） |

⇒ **T1 = adopted**。注意这**推翻** round 1 在 910B4/`e21da32` 上的"C 路径 f16 载体 tie"结论：
同一改动在 910B3/`719c6da` 上是 −9%（3d）/ −8%（32k），且 3d 的两次独立 session 与同族 `all` 任务的
历史"session1 −9.99% → session2 +1.22%"形态**不同**（本任务两次都是 10/10 同向、幅度相近），
故不适用 VP-2026-0324 的 `inconclusive` 判据。

### 10.4 T2 — 32k：MTE2 能否与归约链重叠？（只做量化，不建结构候选）

**本机探针**（`probe_run.py --only 32k_empty,32k_read,32k_ind_vmin --tag round5 --phase round5_probe`，
30 launch；行写入 `copy_floor_calibration.jsonl`（**不**进 `perf_records.jsonl`））：

| probe | Task (us) | aiv_time (us) | vec 忙时 (us) | mte2 忙时 (us) | mte3 忙时 (us) | scalar (us) |
|---|---:|---:|---:|---:|---:|---:|
| `32k_empty`（栅格 + 写回） | 2.332 | 0.941 | 0.008 | 0.002 | 0.228 | 0.667 |
| `32k_read`（GM→UB 纯读地板） | **4.679** | 3.209 | 0.008 | 2.482（ratio 0.784） | 0.236 | 0.687 |
| `32k_ind_vmin`（读 + vcmp + vcast(f16) + vmin 平面 + 末尾一次 reduce_min） | 5.950 | 4.516 | 2.033 | 2.145 | 0.210 | 0.803 |

**生产版本机新 profile**（`profiles/round5/t3/32k/pair1{A,B}`，30 launch，per-block 均值）：

| 版本 | Task (us) | aiv_time (us) | vec (us) | mte2 (us) | scalar_wait (us) |
|---|---:|---:|---:|---:|---:|
| current best（C=f32） | 6.640 | 5.171 | 2.098 | 1.971 | 4.034 |
| r5_op1_cf16（C=f16） | 6.090 | 4.679 | 1.619 | 2.011 | 3.516 |

**重叠算术（显式）**：

- 纯读代价（核体口径）= `read.aiv_time − empty.aiv_time` = 3.209 − 0.941 = **2.268 µs**（Task 口径 4.679 − 2.332 = 2.347）。
- 隔离链代价 = `ind_vmin.aiv_time − read.aiv_time` = 4.516 − 3.209 = **1.307 µs**（Task 口径 1.271）。
- f16 生产版链代价 = 4.679 − 3.209 = **1.470 µs** ⇒ 重叠率 = 1 − 1.470/1.307 = **−12.5%**
  （负值 = 链不仅零重叠，还比隔离探针多 0.16 µs，因为生产链是 per-tile `reduce_min` + `vmin` 合并）。
- f32 生产版链代价 = 5.171 − 3.209 = **1.962 µs** ⇒ 重叠率 = 1 − 1.962/1.307 = **−50%**（f32 平面 + 双跳 epilogue 更贵）。
- Task 分解自洽：`2.332(empty) + 2.268(read) + 1.470(chain) = 6.07 ≈ 6.09`（f16，误差 0.02 us）；
  f32：`2.332 + 2.268 + 1.962 = 6.56 ≈ 6.64`。
- `mte2_busy` 对 Task：f16 2.011/6.090 = **33%**；纯读探针 2.482/4.679 = 53%。即生产 kernel 的 DMA 只有
  1/3 时间在搬数，其余在等归约 consumer；`scalar_wait` 3.516 µs（58%）就是这条串行链的等待。
- **若** DMA 能与链完全重叠，aiv_time 应 ≈ max(2.268, 1.470) + 0.941 ≈ 3.2 µs，Task ≈ **4.6 µs**
  ⇒ 曝光量 = **1.47 µs（当前 Task 的 24%，即 32k 还有 −24% 的空间）**。

**结论**：`Task ≈ read_floor + chain` 严格成立（f32 甚至超出 0.5 µs），**重叠不可能**——与
`CG-2026-0027` 的结论一致（本机 `719c6da` 复现）。这是编译器层（BishengIR 归约 lowering / 流水调度）
的地板，不是 tiling 参数可解；杠杆只能是"让 reduce consumer 与 GM→UB DMA 双缓冲重叠"。
按 AGENTS.md「Compiler Capability Gap Rule」登记（**只在本文件留提案文本，不改登记簿**）：

```yaml
# 拟追加/计数到 .agents/evolution/capability-gaps.md（由 conductor / evolver 执行）
gap_id: CG-2026-0027   # 已存在，occurrences 3 -> 拟 +1
layer: BishengIR（归约 consumer 与 GM→UB DMA 的流水重叠）
capability: >-
  同一 kernel 内 T.reduce_*(dim=1) 与 GM→UB T.copy 无法重叠。910B3/719c6da 实测 2-D 行归约
  （32k，tn=8192，4 tile，32 block）：Task = empty 2.332 + read 2.268 + chain 1.470 = 6.07 us
  ≈ 实测 6.09 us，链暴露率 100%；相对隔离链探针（1.307 us）还多 12%。生产 mte2_busy 仅占 Task 的
  33%（2.011/6.090），scalar_wait 3.516 us（58%）即等链完成。
blocked_algo: >-
  行归约逼近纯读地板的通路受阻：本机 32k 读地板 4.679 us（896 GB/s），f16 载体生产版 6.090 us，
  若重叠可用应 ≈4.6 us（−24%）；3d 同族同构（读地板参考 ~8.1 us / 生产 9.86 us）。
evidence:
  - perf_opt/opt_log.md §10.4（本轮三探针 + 两版生产 profile 的重叠算术）
  - perf_opt/copy_floor_calibration.jsonl（tag=round5, phase=round5_probe: 32k_empty/32k_read/32k_ind_vmin）
  - perf_opt/profiles/round5/{probe,t3,ab_3d}/（msprof op 原始数据）
workaround: >-
  只能压缩链本身（本轮 f16 载体把 f32 链 1.962 us 压到 1.470 us，Task −7.9%），无法让 DMA 与链重叠；
  T.Pipelined(2/3) / auto-multi-buffer 在前序任务已实测 config_no_gain，本轮按"已证伪方向不重建"未重试。
toolchain_stamp: tilelang 0.1.2+719c6da62061f8f7984ac9f87012ca0319b82020 + CANN 9.0.0 + Ascend910B3（AIV=40 / UB 196608）
```

### 10.5 T3 — 净收益复核（同 session，逐 workload 显式分母）

**(1) 代码同异判定**（`codegen_compare.py`，A = Stage-3 baseline，B = 采纳前交付版 `bb552517`）：
(32,32768) **4318 B 逐字节相同**；(128,16384) **4976 B 逐字节相同**；(32,4096) 3296 → 3162 **不同**（R 路径 f16）。
⇒ 32k / 3d 的 A/B 是**零效应对照（zero-effect control）**，其 Δ 只能当噪声地板读，不能当收益。

**(2) 交错 A/B（baseline vs 采纳前 perf_opt，同 session）**：

| workload | 性质 | pairs | A=baseline (us) | B=perf_opt (us) | Δ | 同向 | p | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mask-validation-4k-bool | **真实增益**（代码不同） | 6 | 2.830 | 2.680 | **−5.30%** | 6/6 | 0.031 | B_better |
| mask-validation-32k-bool | **零效应对照**（同码） | 4 | 6.645 | 6.630 | −0.23% | 3/4 | 0.625 | tie |
| 3d-multidim-reduce-bool | **零效应对照**（同码） | 4 | 10.880 | 10.845 | −0.32% | 2/4（1 平） | 1.0 | tie |
| 4k 同码 A/A（两臂同一文件） | 额外噪声标定 | 2 | 2.710 | 2.695 | −0.55% | 1/2 | 1.0 | tie |

**本机噪声地板**：零效应/同码对照的 merged Δ = **−0.23% / −0.32% / −0.55%**，单对最大偏离 ±1.20%
（32k pair3 −1.20%，pair4 +0.76%）⇒ 本 session 噪声地板 **±0.6%（merged）/ ±1.2%（单对）**，
与 VP-2026-0352 的 ±1.3% 同量级。因此 4k 的 −5.30%（6/6 同向，p=0.031）**是真实增益**，
且与 910B4 上两次独立 session 的 −4.73%（4/4）/ −6.08%（3/3）同向同量级 ⇒ **910B3 上复现**。

**(3) 净收益陈述（逐 workload，分母显式）**：

| workload | 分母（同 session baseline） | 采纳前 perf_opt | 采纳后（`final_cf16_r5`） | 采纳前 Δ | 采纳后 Δ | 增益来源 |
|---|---:|---:|---:|---:|---:|---|
| 4k | 2.830 us（`baseline_r5`，交错 A/B 中位） | 2.680（A/B） | 2.7113（final mean） | −5.30% | **−4.20%**（vs 同 session baseline）/ −5.30%（A/B） | R 路径 f16 载体（round 1） |
| 32k | 6.645 us（`baseline_r5`） | 6.630（A/B，同码 tie） | 6.1907（final mean） | −0.23%（噪声） | **−6.84%**（/ A/B −7.91%） | **round 5：C 路径 f16 载体** |
| 3d | 10.880 us（`baseline_r5`） | 10.845（A/B，同码 tie） | 9.9767（final mean） | −0.32%（噪声） | **−8.30%**（/ A/B −9.21%） | **round 5：C 路径 f16 载体** |

参考（**跨设备不可通约，非判据**）：相对 round-0 baseline（910B4，2.9833 / 7.7300 / 14.7160）的
表观变化为 −9.1% / −19.9% / −32.2%，其中设备与工具链差异占大头。

### 10.6 Round-5 最终性能（`phase=final`，同一候选 `final_cf16_r5`）

`profiles/final/round5/final_cf16_r5/<workload>/`，30 launch 的 mean；`IMPL_CHECK: PASS`（torch.equal）
对三个 workload 全部成立；交付文件 sha `5b11da03…`。

| workload | Task (us) | 有效带宽（kernel 字节） | vs 560 GB/s 族验收线 | vs 本机读地板 | 备注 |
|---|---:|---:|---:|---:|---|
| 4k (32,4096) | 2.7113 | 193.4 GB/s（524 320 B） | 不可达（空 kernel 地板 2.332 us 结构限制） | — | 距本机空 kernel 地板仅 0.38 us |
| 32k (32,32768) | 6.1907 | 677.5 GB/s（4 194 336 B） | **121%**（已过线） | 4.679 us = 896 GB/s → 76% | 剩余缺口 = 未隐藏的链 1.47 us |
| 3d (128,16384) | 9.9767 | 840.8 GB/s（8 388 736 B） | **150%** | 参考（910B3 同族）~8.1 us | A/B 实测 9.855 us |

### 10.7 工件/记录簿记（sha、supersede、一处更正披露）

| 项 | 采纳前 | 采纳后 |
|---|---|---|
| `perf_opt/_logical_reduce_kernel.py` | `bb552517b7fd…a4706`（C=f32） | **`5b11da0371874a68ad4e7336dad15003b7be8ef91b80704ff007dc2ca40258d2`**（C=f16） |
| 被 A/B 实测的候选 | — | `_logical_reduce_kernel_round5_op1_cf16.py` = `a2fe7d6e…`（与交付文件代码逐字节等同，仅注释不同） |
| 采纳前交付版备份 | — | `_logical_reduce_kernel_round4_delivered_bb552517.py`（sha 相同） |
| `perf_records.jsonl` | 23 行（含 3 条 round-4 final） | **35 行**：20 条历史（candidate/baseline）+ 12 条 round-5 A/B 行 + 3 条 round-5 final 行 |
| `perf_records_full_history.jsonl` | 23 行 | **26 行**（+3 条 round-4 final `bb552517`：2.7820 / 8.2453 / 14.4840） |

披露（如实）：

1. **final 行 supersede**：交付文件改变使 round-4 的 3 条 final 行失效；处置 = 原样追加进
   `perf_records_full_history.jsonl`，主文件移除这 3 行（gate 4 要求每个 tune workload 恰有一条 final），
   其余 20 行一字未改。新 final 行在同一最终候选上对三个 tune workload 重测（§10.6）。
2. **6 条 round-5 行的 artifact 归属更正**：`write_round5_records.py` 先于采纳执行时把 A 臂
   （T1 的 `final_f16res` 3 行）与 T3 的 B 臂（`r5_netgain_*` 3 行）误标成了**采纳后**的 sha `5b11da03`；
   这 6 行的 `artifact_path` / `artifact_sha256` 已改指它们**实际测量**的文件
   （`_logical_reduce_kernel_round4_delivered_bb552517.py`，sha `bb552517`）。**时延数值一字未改**，
   且这 6 行从未参与任何 final 对账（phase 均为 `candidate`/`baseline`）。
3. 探针（`32k_empty/32k_read/32k_ind_vmin`）写入 `copy_floor_calibration.jsonl`（`phase=round5_probe`），
   **不进** `perf_records.jsonl`，不作 tune workload。
4. `smoke` 三个 workload 仅做精度回归：`--level all` 中 L0 26 例（含 32×256 bool/f16/f32）全 PASS，
   L1 26 例全 PASS，最终 `exact bool comparisons=101 mismatching entries=0`。

### 10.8 Round-5 复盘与 value point proposals

| id | vp_type | 结论（一句话） | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-ANY-7 | D | **载体 dtype 杠杆不按"路径"分裂，而按"链是否被 DMA 掩盖"分裂**：910B4/`e21da32` 上 C 路径 f16 载体 tie，910B3/`719c6da` 上 −9.21%（3d，10/10×2 session）/ −7.91%（32k，4/4）——同一改动跨工具链/设备翻转，说明旧结论是"该工具链下 vec 链被 mte2 掩盖"的特例，不是路径属性 | §10.2/§10.3（vec 忙时 3.419→2.587 us，mte2 不变） | `perf_opt/make_candidates_round5.py --emit-deliverable`（慢→快：`ind/mn/acc/one_v/any_v` f32→f16 + 删一跳 epilogue） | 910B3 / `719c6da` / CANN 9.0.0 |
| VP-ANY-8 | D | 910B3/`719c6da` 上 32k 行归约的**三段分解常数**：empty 2.332 + read 2.268 + chain 1.470 = 6.07 ≈ 实测 6.09 us；读地板 4.679 us = 896 GB/s；链零重叠（重叠率 −12.5%） | §10.4 表 + `copy_floor_calibration.jsonl`（tag=round5） | `perf_opt/probe_run.py --only 32k_empty,32k_read,32k_ind_vmin --tag round5 --phase round5_probe` | 同上 |
| VP-ANY-9 | R | **跨工具链翻转的判决纪律**：`status: stale` 或"旧戳结论"在换设备/换工具链后必须**先复测再引用**（本条：round-1 的 C 路径 tie 被 round-5 直接推翻，若沿用会白丢 −8~−9%）；且"tie"结论必须绑定工具链戳写进 opt_log | §10.1 版本戳表 + §10.3 采纳表 | 无需 repro（流程规则）；对照 `logs/round5/codegen_T1_delivered_vs_cf16.log` | — |

BP proposal（延续 §9.3，未新增流程文档改动）：BP-candidate-4 = 在 `tilelang-op-optimize` Phase 0 的
E-5 版本失效核对里，把"设备型号变化"与"工具链变化"同等对待（本任务换了设备但旧结论仍被沿用，
直到本轮 A/B 才推翻）。
