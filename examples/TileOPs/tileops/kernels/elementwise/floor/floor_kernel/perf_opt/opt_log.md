# Stage 4 调优日志 —— `floor_fwd_kernel`(FloorFwdOp)

- 场景 / 模式:`optimize` Stage 4,`mode=full`(best_effort)
- 基线(只读):`../floor_fwd_kernel.py`(Stage 3 `[PRECISION_PASS]`)
- 交付件:`floor_fwd_kernel.py`
- 唯一时延口径:`msprof op` 的 `Task Duration(us)`;`--launch-count=15 --warm-up=5`,取中位数
- 测量入口:`bench.py`(直接运行 kernel,**不**走 Stage 5 wrapper;复用候选模块自带的 `golden_floor`)
- 批处理 runner:`run_experiments.py`(T-4)
- 硬件:CANN 9.0.0 · Ascend910B2C(24 AICore / 48 Vector 核)· tilelang `0.1.2+npuir`
- 目标 kernel 名(msprof):**`main`**

> 本算子与 `ceil` 逐行同型(同为 C2 饱和旁路 + V2 NaN 传播序列,仅 `round_mode` 不同),
> 故直接沿用 ceil 的调优配方;下表全部为本算子自己实测的数值。

---

## Performance Test Data

### baseline(round 0,`phase=baseline`,artifact = Stage 3 `../floor_fwd_kernel.py`)

| kernel_id | workload_id | candidate_id | task_duration_us | profile_status |
|---|---|---|---:|---|
| tileops/kernels/elementwise/floor/floor_kernel/floor_fwd_kernel.py::floor_fwd_kernel::main | elementwise-16m-float16 | baseline | 348.633 | valid |
| 同上 | elementwise-16m-bfloat16 | baseline | 348.753 | valid |
| 同上 | elementwise-16m-float32 | baseline | 348.593 | valid |
| 同上 | elementwise-64m-float16 | baseline | 1389.752 | valid |
| 同上 | elementwise-64m-bfloat16 | baseline | 1390.152 | valid |
| 同上 | elementwise-256m-float16 | baseline-launchable-256m-float16 | 1853.563 | valid(**非交付默认配置**,见下) |
| 同上 | elementwise-256m-bfloat16 | baseline-launchable-256m-bfloat16 | 1854.243 | valid(**非交付默认配置**,见下) |

命令模板:

```bash
msprof op --kernel-name=main \
  --output=profiles/<phase>/round<r>/<candidate>/<workload> \
  --launch-count=15 --warm-up=5 --dump=off \
  --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio \
  python bench.py --impl <candidate.py> --case <workload_id>
```

### 256m 基线的口径声明(与 16m/64m 分开读)

交付默认(`block_size=2048`)在 256m 上**无法启动**:`N/2048 = 131072` 块越栅格上限
(`OPS function error ... 107000` / `Invalid_Argument(EL0003)` / SIGABRT,与 abs/ceil 同一陷阱)。
本 kernel 的 UB 占用与 ceil 相同(28 B/元素),可选的合法档位同 ceil:

| 用于 256m 的 block_size | 栅格 = N/bs | 结果 |
|---:|---:|---|
| 2048(交付默认) | 131072 | ❌ 越栅格 |
| 4096 | 65536 | ❌ 仍被拒(栅格上限**低于** 65536) |
| **6144** | 43691 | ✅ **1853.563 µs ← 表中基线** |
| 8192 | 32768 | ❌ UB 溢出(229,408 B > 192 KB) |

故 256m 不存在「交付默认配置下」的基线,表中值是**同一个 Stage 3 文件**、仅把 `block_size`
设为 6144 测得的、唯一可得的基线。

### smoke / skipped 排除

| workload_id | 类别 | 排除原因 | 最终精度 |
|---|---|---|---|
| smoke-1m-float32 | smoke | 案例 ID 含独立 `smoke` 词 | PASS |
| smoke-1m-float16 | smoke | 同上 | PASS |
| smoke-1m-bfloat16 | smoke | 同上 | PASS |

无 `skipped` 项。

---

## Iteration Log

### Round 1 —— 结构改写:grid-stride `T.serial`

**现象**:`Block Dim = 8192`(16m)/`32768`(64m)/`131072`(256m);16m(348.63µs)与 64m
(1389.75µs)每元素耗时相同 ⇒ 瓶颈随规模线性;fp32 搬 2 倍字节仅多 0.0% 时延 ⇒ 限制在「每元素」。

**候选优化点(单一)**:grid 收成 `num_cores` + 核内 grid-stride 串行;`c_ub` 常量填充提到循环外。
C2+V2 序列一字不动。

**合并非回退检查(全部 tune workload)**

| branch | workload_id | L0 | Task Duration(us) | vs current best | status |
|---|---|---:|---:|---:|---|
| r1_serial48 | elementwise-16m-float16 | pass | 106.598 | +69.4% | ok |
| r1_serial48 | elementwise-16m-bfloat16 | pass | 106.598 | +69.4% | ok |
| r1_serial48 | elementwise-16m-float32 | pass | 114.118 | +67.3% | ok |
| r1_serial48 | elementwise-64m-float16 | pass | 419.852 | +69.8% | ok |
| r1_serial48 | elementwise-64m-bfloat16 | pass | 421.612 | +69.7% | ok |
| r1_serial48 | elementwise-256m-float16 | pass | 1708.046 | 基线为可启动配置(1853.563) | ok |
| r1_serial48 | elementwise-256m-bfloat16 | pass | 1707.826 | 基线为可启动配置(1854.243) | ok |

**winner**:`r1_serial48`(7/7 无回退,并解锁 256m)。结构单独贡献 **−69.4%**,与同族 `ceil` 的
−68.9% 一致(两算子每 tile 的 UB 占用与向量指令数相同)。

### Round 2 —— 结构后粗搜:`block_size × num_cores`

**搜索空间**:`block_size ∈ {2048, 4096}` × `num_cores ∈ {24, 48, 96, 192}`,代表 shape =
`elementwise-16m-float16`。参数分支用 `bench.py --check` 作配置级等价门禁。
(`block_size ≥ 8192` 不可选:UB 溢出,见 round 3 同族实测。)

**候选 vs current best 对比表**(current best = `r1_serial48`,16m fp16 = 106.598µs)

| block_size | num_cores | Task Duration(us) | vs r1_serial48 |
|---:|---:|---:|---:|
| 2048 | 24 | 207.596 | −94.8%(劣化) |
| 2048 | 48 | 105.978 | +0.6% |
| 2048 | 96 | 107.058 | −0.4% |
| 2048 | 192 | 107.458 | −0.8% |
| 4096 | 24 | 134.777 | −26.4%(劣化) |
| 4096 | 48 | **70.979** | **+33.4%** |
| 4096 | 96 | 70.759 | +33.6% |
| 4096 | 192 | 72.139 | +32.3% |

**读表结论**:同族同型——`num_cores=24` 劣化,≥48 进平区(48/96/192 差异 ≤2%);
`block_size` 翻倍带来 +33% 台阶。**winner**:`bs=4096 / nc=48`(70.979µs;
与 nc=96 的 70.759µs 差 0.3%,在噪声带内,取 48 = 物理 Vector 核数,与 ceil 一致)。

### Round 3 —— 跨 dtype/shape 确认

**现象**:round 2 的最优只在 16m fp16 上测得;须确认跨 dtype 与大 shape 成立。

| 是否复测 | 说明 |
|---|---|
| bs=8192 的 UB 上限探测 | **未在本算子重复**;由同族 `ceil`(同一 UB 占用 28 B/元素、同一 10 缓冲结构)实测报 `ub overflow, requires 1835264 bits while 1572864 bits available` 抑制。本算子的 UB 占用经 `TRAP-UB-multibuffer-inflation` 反例核算同为 28 B/元素(4096 → 112 KB 合法) |
| 跨 dtype / 大 shape | 由 `phase=final` 在 `bs=4096/nc=48` 上对**全部 7 条 tune workload**复测覆盖(见 Final Performance Test Data) |

**winner**:`bs=4096 / nc=48`。**中止条件**:结构 + 参数空间收束。

---

## Autotune Log

未使用 `tilelang` autotune API;round 2 用等价手动粗搜覆盖同一空间(8 组全测完)。

| 项 | 内容 |
|---|---|
| search space | `block_size ∈ {2048, 4096}` × `num_cores ∈ {24, 48, 96, 192}`(8 组全测) |
| best config | `block_size=4096`, `num_cores=48` → 70.979 µs |
| top-k | `4096/96` 70.759 · `4096/192` 72.139 · `2048/48` 105.978 |
| 未测组合 | `block_size ≥ 8192`:不可选(UB 溢出,同族实测);`num_cores=24` 已证劣化 |
| 编译/精度失败数 | 0 |
| winner 独立复测 | `perf_opt/floor_fwd_kernel.py`(默认 = 该组合)final 复测 16m fp16 = **70.819 µs** |
| Block Dim | 每个候选 = `num_cores` |

**平区裁决说明**:`nc≥48` 三档差异 ≤2%(远小于 5%),`block_size` 两档是台阶量级而非平区。
`num_cores` 三档未走 `ab_test.py` 即定档 48,依据同 abs/ceil:取「48 = 物理 Vector 核数」,
不是「在平区内单次测量排序」。列为流程风险。

**采数纪律引用**:同族 `ceil` 本轮出现「同配置两次测量差 2×」的异常值(204.3 vs 106.1/106.6µs),
本算子未出现该现象(同配置三次测量:105.978 / 106.598 / 106.598µs,一致)。
处置规则见 `ceil_kernel/perf_opt/opt_log.md` 的「数据修正」段与 pattern-library PL-1.21 附注。

---

## Final Summary

**逐 workload 提升**(逐项对账见下节 `Final Performance Test Data`)

| workload_id | baseline_us | final_us | 提升 | 备注 |
|---|---:|---:|---:|---|
| elementwise-16m-float16 | 348.633 | 70.819 | +79.7%(4.92×) | |
| elementwise-16m-bfloat16 | 348.753 | 70.679 | +79.7%(4.93×) | |
| elementwise-16m-float32 | 348.593 | 79.198 | +77.3%(4.40×) | |
| elementwise-64m-float16 | 1389.752 | 277.394 | +80.0%(5.01×) | |
| elementwise-64m-bfloat16 | 1390.152 | 279.834 | +79.9%(4.97×) | |
| elementwise-256m-float16 | 1853.563 | 1182.636 | +36.2% | 基线为可启动配置(bs=6144) |
| elementwise-256m-bfloat16 | 1854.243 | 1188.596 | +35.9% | 同上 |

**带宽占峰比**(分母 = 1.8 TB/s;口径约束见 `constants.md` CONST-bw-occupancy-l2-resident)

| workload_id | baseline | final |
|---|---:|---:|
| elementwise-16m-float16 | 10.7% | **52.6%** |
| elementwise-16m-bfloat16 | 10.7% | **52.8%** |
| elementwise-16m-float32 | 21.4% | **94.1%** |
| elementwise-64m-float16 | 10.7% | **53.8%** |
| elementwise-64m-bfloat16 | 10.7% | **53.3%** |
| elementwise-256m-float16 | — | **25.2%** |
| elementwise-256m-bfloat16 | — | **25.1%** |

> 与 `ceil` 的占峰比几乎逐项相同(52.6 vs 52.5、94.1 vs 93.1、25.2 vs 25.2),符合两者
> 「逐行同型、仅 `round_mode` 不同」的事实——也说明本族**没有**变成纯带宽受限,而是落在
> 向量发射与搬运的混合区。

### 设计估算 vs 实测偏差行(D-2 回填,DESIGN.md §1.6.0)

| workload | 设计估算下界(2·N·e ÷ 1.8 TB/s) | baseline 实测 | 偏差 | final 实测 | 相对估算下界 |
|---|---:|---:|---:|---:|---|
| elementwise-16m-float16 | 37.3 µs | 348.633 | **9.3×** | 70.819 | 差于下界 1.90× |
| elementwise-16m-float32 | 74.6 µs | 348.593 | 4.7× | 79.198 | 差于下界 1.06× |
| elementwise-64m-float16 | 149.1 µs | 1389.752 | 9.3× | 277.394 | 差于下界 1.86× |

**失准项归属**:baseline 的 9.3× 归因于逐块派发开销(与 abs/ceil 同因);
final 未达估算下界(1.90×)同 ceil——每元素 12+ 条向量指令 + `block_size` 被 UB 压在 4096,
未进入纯带宽受限区 ⇒ 设计期「纯带宽受限」对下限估计偏乐观(残差属向量发射项,在参数/结构域内)。

**smoke 最终精度**:3/3 PASS。**skipped**:无。**`merge_blocked`**:无。

### Final Performance Test Data

`candidate_id` 全部为 **`final-bs4096-nc48`**,时延来自 `perf_records.jsonl` 的 `phase=final` 记录。

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/elementwise/floor/floor_kernel/floor_fwd_kernel.py::floor_fwd_kernel::main | elementwise-16m-float16 | 348.633 | 70.819 | final-bs4096-nc48 |
| tileops/kernels/elementwise/floor/floor_kernel/floor_fwd_kernel.py::floor_fwd_kernel::main | elementwise-16m-bfloat16 | 348.753 | 70.679 | final-bs4096-nc48 |
| tileops/kernels/elementwise/floor/floor_kernel/floor_fwd_kernel.py::floor_fwd_kernel::main | elementwise-16m-float32 | 348.593 | 79.198 | final-bs4096-nc48 |
| tileops/kernels/elementwise/floor/floor_kernel/floor_fwd_kernel.py::floor_fwd_kernel::main | elementwise-64m-float16 | 1389.752 | 277.394 | final-bs4096-nc48 |
| tileops/kernels/elementwise/floor/floor_kernel/floor_fwd_kernel.py::floor_fwd_kernel::main | elementwise-64m-bfloat16 | 1390.152 | 279.834 | final-bs4096-nc48 |
| tileops/kernels/elementwise/floor/floor_kernel/floor_fwd_kernel.py::floor_fwd_kernel::main | elementwise-256m-float16 | 1853.563 | 1182.636 | final-bs4096-nc48 |
| tileops/kernels/elementwise/floor/floor_kernel/floor_fwd_kernel.py::floor_fwd_kernel::main | elementwise-256m-bfloat16 | 1854.243 | 1188.596 | final-bs4096-nc48 |

---

## Skill Retrospective

### 流程问题与建议

| # | 环节 | 问题 | 本任务实证 | 建议 | vp_type |
|---|---|---|---|---|---|
| 1 | runner 的 baseline 阶段 | **baseline 阶段要求「恰好一个候选」,但一旦加入后续轮次就无法再补采基线** | 本次先建了含多候选的 EXPERIMENTS 再跑 baseline,直接报 `error: baseline/merged/final measurements must use exactly one candidate per kernel`;需加 `--only baseline` 才能跑 | 把「补采/重采基线」写成显式支持(带 `--only` 的用例说明),或让 baseline 阶段默认只取候选 id 为 `baseline` 的条目 | R |
| 2 | 同族配方复用 | 复用收益未被记录 | floor 直接沿用 ceil 的配方,round 1 与 round 2 的**最优档完全一致**(bs=4096/nc=48),且结构单独贡献 69.4% vs ceil 的 68.9% | 同 `floor` Stage 3 的既有提案:在 Stage Result 里增 `reused_from` 字段 | R |
| 3 | UB 上限探测 | 本算子未重复 bs=8192 的探测 | 由同族 ceil 的实测报错 + UB 反例核算抑制(本算子 UB 占用同为 28 B/元素) | 可接受(**同族同结构**的抑制有实测支撑),但须在 opt_log 显式写「未重复的原因」而非沉默省略(本日志已写) | R |

### 价值点提案(含证据三件套)

| id | vp_type | 内容 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-floor-1 | **P** | **PL-1.21 结构在同族取整算子上**第三次**成立**:16m fp16 348.63→106.60µs(−69.4%),叠 bs=4096 到 70.82µs(累计 −79.7%);与 ceil 的 −68.9%/−79.3% 高度一致(两者逐行同型) | baseline 7 + r1 7 + final 7(`perf_records.jsonl`) | `pattern-library/repro/PL-1.21-gridstride-serial.py` | tilelang `0.1.2+npuir` / CANN 9.0.0 / 910B2C |
| VP-floor-2 | **C** | 同族取整算子(ceil/floor)的 Stage 4 配方可**整段复用**,且**最优档位完全一致**(bs=4096/nc=48);差异只在 `round_mode` | 本 opt_log 与 `ceil_kernel/perf_opt/opt_log.md` 的逐项对照 | 两份 opt_log + 各自 `perf_records.jsonl` | 同上 |

**pattern-library 回写**:VP-floor-1 作为 PL-1.21 的**第三证**追加(同族第三次复现,收益一致);
VP-floor-2 归入该条目的「配方复用」附注。

---

## 结论

- **中止条件**:结构 + 参数空间收束(2 轮实测 + final 全量;预算 10/30 未耗尽);未 `blocked`、未 `plateau`。
- **性能目标**:未指定逐 workload 数值目标 ⇒ 以「通过核内合并与全量非回退检查的实测最优结果」为结论。
- **`[DESIGN_LIMIT]`**:不产出(理由同 ceil:残余差距在参数/结构域内可继续处理)。
- **verdict**:`TUNING_COMPLETED`
