# Stage 4 调优日志 —— `cos_fwd_kernel`(CosFwdOp)

- 场景 / 模式:`optimize` Stage 4,`mode=full`(best_effort)
- 基线(只读):`../cos_fwd_kernel.py`(Stage 3 `[PRECISION_PASS]`)
- 交付件:`cos_fwd_kernel.py`
- 唯一时延口径:`msprof op` 的 `Task Duration(us)`;`--launch-count=15 --warm-up=5`,取中位数
- 测量入口:`bench.py`(直接运行 kernel,**不**走 Stage 5 wrapper;复用候选模块自带的 `golden_cos`)
- 批处理 runner:`run_experiments.py`(T-4)
- 硬件:CANN 9.0.0 · Ascend910B2C(24 AICore / 48 Vector 核)· tilelang `0.1.2+npuir`
- 目标 kernel 名(msprof):**`main`**

> 本算子是 elementwise 族里**每 tile 工作量最大**的一个:12 个 UB 缓冲 + 20 余条向量指令
> (四象限范围归约:减 k·π/2、取 q mod 4、vcos/vsin、按象限选符号)。这一点直接决定了本轮的结论。

---

## Performance Test Data

### baseline(round 0,`phase=baseline`,artifact = Stage 3 `../cos_fwd_kernel.py`)

| kernel_id | workload_id | candidate_id | task_duration_us | profile_status |
|---|---|---|---:|---|
| tileops/kernels/elementwise/cos/cos_kernel/cos_fwd_kernel.py::cos_fwd_kernel::main | elementwise-16m-float16 | baseline | 393.332 | valid |
| 同上 | elementwise-16m-bfloat16 | baseline | 395.372 | valid |
| 同上 | elementwise-16m-float32 | baseline | 380.512 | valid |
| 同上 | elementwise-64m-float16 | baseline | 1565.369 | valid |
| 同上 | elementwise-64m-bfloat16 | baseline | 1558.149 | valid |
| 同上 | elementwise-256m-float16 | baseline-launchable-256m-float16 | 5188.856 | valid(**非交付默认配置**,见下;有效 launch 数 11) |
| 同上 | elementwise-256m-bfloat16 | baseline-launchable-256m-bfloat16 | 5202.776 | valid(同上;**有效 launch 数 3**) |

命令模板:

```bash
msprof op --kernel-name=main \
  --output=profiles/<phase>/round<r>/<candidate>/<workload> \
  --launch-count=15 --warm-up=5 --dump=off \
  --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio \
  python bench.py --impl <candidate.py> --case <workload_id>
```

### 256m 基线的口径声明 —— 两个约束**互斥**,只有一个窄窗口可启动

Stage 3 结构是「一个 tile 一个 block」,故栅格 = `N / block_size`。本 kernel 有 12 个
block-sized UB 缓冲、实测占用 **44 B/元素**,于是 256m(`N = 268 435 456`)上:

| 约束 | 不等式 | 含义 |
|---|---|---|
| UB 容量 | `block_size × 44 ≤ 192 KB` | ⇒ `block_size ≤ 4468` |
| 发射栅格 | `N / block_size ≤ 栅格上限` | ⇒ `block_size ≥ 6144`(实测 4096 → 栅格 65536 **被拒**) |

**两个区间不相交** —— 也就是说 256m 上**不存在任何一个 `block_size ≥ 4468` 的合法 Stage 3 配置**。
唯一可启动的窗口在栅格刚好压线处:

| block_size | 栅格 = N/bs | UB | 结果 |
|---:|---:|---:|---|
| 2048(交付默认) | 131072 | 88 KB | ❌ 越栅格 |
| 4096 | 65536 | 176 KB | ❌ 栅格被拒 |
| 6144 | 43691 | 264 KB | ❌ UB 溢出 |
| **4200** | **63914** | **180.5 KB** | ✅ **5188.856 µs ← 表中基线** |
| 8192 | 32768 | 352 KB | ❌ UB 溢出 |

故 256m 的基线是**同一 Stage 3 文件**、仅把 `block_size` 设为 4200 测得的唯一可启动值。

⚠️ **采数披露**:该规模下 Stage 3 结构的 `Block Dim = 63914`,msprof 对这么大的 trace 做分析极慢,
**两次采集都超过了 runner 的 900 s 超时**(实测单次 ~15 min);OPPROF 与 `OpBasicInfo` CSV 均已产出,
故按 runner 的同一 schema 从 CSV 取中位数补录,有效 launch 数为 fp16 **11**、bf16 **3**(而非 15)。
数值可信但精度低于 15 launch 口径,如实标注。

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

**现象**:`Block Dim = 8192`(16m)/`32768`(64m)/`131072`(256m);16m(393.33µs)与 64m
(1565.37µs)每元素耗时相同 ⇒ 瓶颈随规模线性;fp32 搬 2 倍字节反而快 3.3% ⇒ 限制在「每元素」。

**候选优化点(单一)**:grid 收成 `num_cores` + 核内 grid-stride 串行;四象限归约链一字不动。

**合并非回退检查(全部 tune workload)**

| branch | workload_id | L0 | Task Duration(us) | vs current best | status |
|---|---|---:|---:|---:|---|
| r1_serial48 | elementwise-16m-float16 | pass | 288.114 | +26.7% | ok |
| r1_serial48 | elementwise-16m-bfloat16 | pass | 290.414 | +26.5% | ok |
| r1_serial48 | elementwise-16m-float32 | pass | 277.294 | +27.1% | ok |
| r1_serial48 | elementwise-64m-float16 | pass | 1140.777 | +27.1% | ok |
| r1_serial48 | elementwise-64m-bfloat16 | pass | 1149.557 | +26.2% | ok |
| r1_serial48 | elementwise-256m-float16 | pass | 4551.449 | 基线为可启动配置(5188.856) | ok |
| r1_serial48 | elementwise-256m-bfloat16 | pass | 4588.268 | 基线为可启动配置(5202.776) | ok |

**winner**:`r1_serial48`(7/7 无回退,并解锁 256m)。

**⚠️ 与同族其它四个算子的对比(本轮回写 PL-1.21 的关键数据)**:

| 算子 | 每 tile 的 UB 缓冲数 | 每 tile 的向量指令数 | 结构单独收益 |
|---|---:|---:|---:|
| abs | 2 | 1 | **−81.9%** |
| exp | 4 | 3 | −81.4% |
| ceil / floor | 10 | 12+ | −68.9% / −69.4% |
| **cos** | **12** | **20+** | **−26.7%** |

⇒ **PL-1.21 的适用边界**:该结构消除的是「逐块派发/搬运摊薄」开销,收益上限由**每 tile 的
实际工作量**决定。tile 内工作量越大,派发开销占比越小,结构改写能拿到的越少。cos 每元素要跑
20 余条向量指令,所以结构改写只值 1/3。

### Round 2 —— 结构后粗搜:`block_size × num_cores`

**搜索空间**:`block_size ∈ {2048, 4096}` × `num_cores ∈ {24, 48, 96, 192}`
(`block_size ≥ 8192` 不可选:44 B/元素 × 8192 = 352 KB > 192 KB UB)。

**候选 vs current best 对比表**(current best = `r1_serial48`,16m fp16 = 288.114µs)

| block_size | num_cores | Task Duration(us) | vs r1_serial48 |
|---:|---:|---:|---:|
| 2048 | 24 | 571.869 | −98.5%(劣化) |
| 2048 | 48 | 287.894 | +0.1% |
| 2048 | 96 | 288.534 | −0.1% |
| 2048 | 192 | 289.814 | −0.6% |
| 4096 | 24 | 494.050 | −71.5%(劣化) |
| 4096 | 48 | **250.595** | **+13.0%** |
| 4096 | 96 | 251.015 | +13.0% |
| 4096 | 192 | 252.755 | +12.3% |

**读表结论**:与前四个算子完全同型——`num_cores=24` 大幅劣化,≥48 进平区(≤0.4%);
`block_size` 翻倍是唯一有效台阶。**winner**:`bs=4096 / nc=48`(250.595µs)。

### Round 3 —— UB 上限与跨 dtype/shape 确认

| 项 | 结果 |
|---|---|
| `bs=8192` UB 探测 | **未实测**:按 44 B/元素算需 352 KB ≫ 192 KB,且同族在 28 B/元素时已实测 `bs=8192` 报 `ub overflow`;本算子系数更大,无需重复 |
| 跨 dtype / 大 shape | 由 `phase=final` 在 `bs=4096/nc=48` 上对全部 7 条 tune workload 复测覆盖 |

**winner**:`bs=4096 / nc=48`。**中止条件**:结构 + 参数空间收束。

---

## Autotune Log

未使用 `tilelang` autotune API;round 2 用等价手动粗搜覆盖同一空间(8 组全测完)。

| 项 | 内容 |
|---|---|
| search space | `block_size ∈ {2048, 4096}` × `num_cores ∈ {24, 48, 96, 192}`(8 组全测) |
| best config | `block_size=4096`, `num_cores=48` → 250.595 µs |
| top-k | `4096/96` 251.015 · `4096/192` 252.755 · `2048/48` 287.894 |
| 未测组合 | `block_size ≥ 8192`:不可选(UB 溢出,44 B/元素);`num_cores=24` 已证劣化 |
| 编译/精度失败数 | 0 |
| winner 独立复测 | `perf_opt/cos_fwd_kernel.py`(默认 = 该组合)final 复测 16m fp16 = **250.695 µs** |
| Block Dim | 每个候选 = `num_cores`;理论 tile 数 = `ceildiv(N, block_size)` |

**平区裁决说明**:`nc≥48` 三档差异 ≤0.4%(远小于 5%),定档 48 = 物理 Vector 核数。

---

## Final Summary

**逐 workload 提升**(逐项对账见下节 `Final Performance Test Data`)

| workload_id | baseline_us | final_us | 提升 | 备注 |
|---|---:|---:|---:|---|
| elementwise-16m-float16 | 393.332 | 250.695 | +36.3%(1.57×) | |
| elementwise-16m-bfloat16 | 395.372 | 252.135 | +36.2%(1.57×) | |
| elementwise-16m-float32 | 380.512 | 242.115 | +36.4%(1.57×) | |
| elementwise-64m-float16 | 1565.369 | 988.180 | +36.9%(1.58×) | |
| elementwise-64m-bfloat16 | 1558.149 | 994.440 | +36.2%(1.57×) | |
| elementwise-256m-float16 | 5188.856 | 3937.001 | +24.1% | 基线为可启动配置(bs=4200,launch 数 11) |
| elementwise-256m-bfloat16 | 5202.776 | 3963.161 | +23.8% | 基线同上(launch 数 3) |

> 本算子的提升幅度显著低于同族(`+36%` vs `abs/exp` 的 `+92%`),原因见 round 1 的边界条件表:
> 结构改写消除的是派发/搬运摊薄开销,而 cos 每元素有 20 余条向量指令,这部分开销本就不占主导。

**带宽占峰比**(分母 = 1.8 TB/s;口径约束见 `constants.md` CONST-bw-occupancy-l2-resident)

| workload_id | baseline | final |
|---|---:|---:|
| elementwise-16m-float16 | 9.5% | **14.9%** |
| elementwise-16m-bfloat16 | 9.5% | **14.8%** |
| elementwise-16m-float32 | 19.6% | **30.8%** |
| elementwise-64m-float16 | 9.5% | **15.1%** |
| elementwise-64m-bfloat16 | 9.5% | **15.0%** |
| elementwise-256m-float16 | — | **7.6%** |
| elementwise-256m-bfloat16 | — | **7.5%** |

> 与前四个算子**形成鲜明对照**:且不说 16m/64m 只有 15%(它们**没有**像 abs/exp 那样超过 100%),
> 连越过 L2 的 256m 也从 exp 的 38.4% 掉到 7.5%。原因是同一个:每元素搬运量相同,但每元素要跑
> 20 余条向量指令,时间被指令链吃掉 —— **cos 不是带宽受限的**。

### 设计估算 vs 实测偏差行(D-2 回填,DESIGN.md §1.6.0)

| workload | 设计估算下界(2·N·e ÷ 1.8 TB/s) | baseline 实测 | 偏差 | final 实测 | 相对估算下界 |
|---|---:|---:|---:|---:|---|
| elementwise-16m-float16 | 37.3 µs | 393.332 | **10.5×** | 250.695 | **仍差 6.72×** |
| elementwise-16m-float32 | 74.6 µs | 380.512 | 5.1× | 242.115 | 仍差 3.25× |
| elementwise-64m-float16 | 149.1 µs | 1565.369 | 10.5× | 988.180 | 仍差 6.63× |

**失准项归属(与前四个算子不同,须单独说明)**:abs/exp 的 final **优于**估算下界(1.12–1.54×),
ceil/floor 略差于下界(1.9×);而 cos 的 final **仍差 6.7×**。

归因:设计期把本算子判为「纯带宽受限」(`§1.6.0` 的估算取 `max(流量项, 发射项, 容量项) = 流量项`),
但**实测反驳了该判定** —— 本 kernel 的四象限归约每元素含 20 余条**串行相关**的向量指令
(减 k·π/2 → 取整 → 取 mod → vcos/vsin → 象限选符号),且多次跨 fp32/int32 类型转换,
真正的瓶颈是**指令链/发射**,不是带宽。故 final 无法接近带宽下界。

**这是本轮唯一一处「实测与设计假设直接矛盾」**。**是否构成 `[DESIGN_LIMIT]`**:依
`_shared/standards/perf-feedback.md` §1 需**同时**满足两项——① 天花板由设计/算法层决定
(✅ 归因到 §1.6.0 的算法选型与 op 数);② 结构性加速估计 > 2× 或与设计假设直接矛盾
(⚠️ 就「矛盾」一词可算命中,但 > 2× 这一支**未满足**:本轮结构改写只给 1.57×)。
**本日志不产出 `perf_feedback.md`**,理由是该标准还要求**参照锚定**章节(同族同 API 先例检索 +
morph 式最小增量对照),而本轮**未做**该对照(仅与本项目内四个同族算子做了跨算子对比)。
按「断言前强制检索仓库内同族先例」的纪律,缺该证据即**不产出**该文件,以免给出未经锚定的天花板断言。
**如实登记为「设计假设偏离」,留给后续独立处置(补做参照锚定后可由 designer 评估)。**

**smoke 最终精度**:3/3 PASS。**skipped**:无。**`merge_blocked`**:无。

### Final Performance Test Data

`candidate_id` 全部为 **`final-bs4096-nc48`**,时延来自 `perf_records.jsonl` 的 `phase=final` 记录。

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/elementwise/cos/cos_kernel/cos_fwd_kernel.py::cos_fwd_kernel::main | elementwise-16m-float16 | 393.332 | 250.695 | final-bs4096-nc48 |
| tileops/kernels/elementwise/cos/cos_kernel/cos_fwd_kernel.py::cos_fwd_kernel::main | elementwise-16m-bfloat16 | 395.372 | 252.135 | final-bs4096-nc48 |
| tileops/kernels/elementwise/cos/cos_kernel/cos_fwd_kernel.py::cos_fwd_kernel::main | elementwise-16m-float32 | 380.512 | 242.115 | final-bs4096-nc48 |
| tileops/kernels/elementwise/cos/cos_kernel/cos_fwd_kernel.py::cos_fwd_kernel::main | elementwise-64m-float16 | 1565.369 | 988.180 | final-bs4096-nc48 |
| tileops/kernels/elementwise/cos/cos_kernel/cos_fwd_kernel.py::cos_fwd_kernel::main | elementwise-64m-bfloat16 | 1558.149 | 994.440 | final-bs4096-nc48 |
| tileops/kernels/elementwise/cos/cos_kernel/cos_fwd_kernel.py::cos_fwd_kernel::main | elementwise-256m-float16 | 5188.856 | 3937.001 | final-bs4096-nc48 |
| tileops/kernels/elementwise/cos/cos_kernel/cos_fwd_kernel.py::cos_fwd_kernel::main | elementwise-256m-bfloat16 | 5202.776 | 3963.161 | final-bs4096-nc48 |

---

## Skill Retrospective

### 流程问题与建议

| # | 环节 | 问题 | 本任务实证 | 建议 | vp_type |
|---|---|---|---|---|---|
| 1 | 采数 runner 超时 | **`TIMEOUT_S=900` 在「块数极大」的基线上不够**,且超时后 OPPROF 与 CSV 其实已产出 | 256m 基线 `Block Dim=63914`,msprof 分析单次 ~15 min,两次都被超时打断;CSV 完整可读 | runner 增加「超时后回读已产出的 `OpBasicInfo` CSV」的恢复路径,或把 `TIMEOUT_S` 按 `Block Dim` 估算;并把「有效 launch 数」写进记录 | D/P |
| 2 | 基线口径 | **「可启动基线」的合法性需要判据,不能默认存在** | cos 在 256m 上 UB 与栅格两个约束**互斥**(`bs≤4468` vs `bs≥6144`),只能取窗口内的 `bs=4200` | `signal-registry.md` §5 或 profile-collection 增加规则:当交付默认配置不可启动时,基线须登记「配置 + 合法性依据 + 与默认配置的差异」 | R |
| 3 | 结构收益 | **PL-1.21 的收益随每 tile 工作量下降的规律在本轮得到最强证据** | 同族五算子的结构收益:abs −81.9% / exp −81.4% / floor −69.4% / ceil −68.9% / **cos −26.7%**,与各算子每 tile 的指令数单调对应 | 回写 PL-1.21:给出「每 tile 向量指令数 → 结构收益上限」的经验表 | D |

### 价值点提案(含证据三件套)

| id | vp_type | 内容 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-cos-1 | **D** | **PL-1.21 的适用边界(最强证据)**:结构收益与「每 tile 向量指令数」单调负相关——1 条指令(abs)得 −81.9%,20+ 条(cos)只得 −26.7% | 五个 `perf_records.jsonl` 的 baseline/merged 行 + 各自 opt_log 的 round-1 表 | `pattern-library/repro/PL-1.21-gridstride-serial.py` | tilelang `0.1.2+npuir` / CANN 9.0.0 / 910B2C |
| VP-cos-2 | **D** | **Ub × 栅格双约束可能互斥**:256m 上 cos 需 `bs ≤ 4468`(UB 44 B/元素)且 `bs ≥ 6144`(栅格),交集为空 ⇒ 交付默认配置**不可启动**,唯一合法窗口是 `bs=4200`(栅格 63914、UB 180.5 KB) | 基线尝试表(2048/4096/6144/8192 全失败)+ `bs=4200` 成功记录 | repro-missing(需 NPU) | 同上 |
| VP-cos-3 | **D** | **msprof profiling 开销随 `Block Dim` 爆炸**:63914 块的 kernel,单次采集(20 launch)的分析耗时 ~15 min,而同一算子 grid-stride 版(48 块)只需 ~40 s | 两次超时日志 + OPPROF 产出时间戳(10:01 / 10:16) | repro-missing | 同上 |
| VP-cos-4 | **C** | cos 是 elementwise 族里**唯一非带宽受限**的算子:final 仍差设计带宽下界 6.7×,占峰比 14.9%(256m 仅 7.6%) | 本 opt_log 的 D-2 行与占峰比表 | 本 opt_log | 同上 |

**pattern-library 回写**:VP-cos-1 作为 PL-1.21 的**第五证 + 适用边界表**追加(该条目的关键增量);
VP-cos-2 拟入 `constants.md`(UB × 栅格互斥判据);VP-cos-3 拟入 `traps-runtime.md`。

---

## 结论

- **中止条件**:结构 + 参数空间收束(`bs ≥ 8192` 因 UB 不可选、`nc=24` 已证劣化);未 `blocked`、未 `plateau`。
- **性能目标**:未指定逐 workload 数值目标 ⇒ 以「通过核内合并与全量非回退检查的实测最优结果」为结论。
- **`[DESIGN_LIMIT]`**:**不产出** —— 条件 ① 满足、② 的「>2×」一支不满足,且该标准还要求**参照锚定**
  章节(同族同 API 先例 + morph 最小增量对照),本轮未做。已如实登记为「设计假设偏离」并说明缺哪项证据。
- **遗留(须独立处置)**:`§1.6.0` 对 cos 的「纯带宽受限」判定被实测反驳(final 仍差带宽下界 6.7×),
  建议补做参照锚定后由 designer 评估是否改判为「指令链受限」并考虑更省 op 的归约实现。
- **verdict**:`TUNING_COMPLETED`

---

# 追加(2026-10-22):第二轮尝试 —— int 位运算象限,候选被否决

## 体检发现的新事实

`Ratio` 在 16m/64m/256m 上**平坦**(53.6% / 54.3% / 55.4%),不像 abs 那样随规模爬升
⇒ **cos 是计算受限,不是带宽受限**。每个元素有 **29 条向量指令**
(vmul 11 / vcast 8 / vsub 6 / vadd 2 / vcos 1 / vsin 1)。

其中**象限计算占 10 条**,而且审出一个此前未记录的事实:
**`q = k mod 4` 是死中间量** —— 下游只消费 `b0 = q&1` 与 `b1 = (q>>1)&1`,`q` 本身从未被使用。
所以那 10 条 fp32 算术链(`mul→floor→cast→mul→sub`,再对 q 做一遍)`理论上可压到 6 条`。

## 候选 `r3_intquad`(29 → 25 条指令)

```python
T.vand(ki_ub, one_i, ki_ub)                # ki := k & 1   (= b0),就地
T.vcast(ki_ub, b0_ub)                      # b0 fp32
T.vcast(kf_ub, ki_ub, round_mode="rint")   # 重算 k(保证 UB 中性)
T.vand(ki_ub, two_i, ki_ub)                # ki := k & 2   (= 2*b1)
T.vcast(ki_ub, half_ub)
T.vmul(half_ub, 0.5, half_ub)              # b1
```

常量用 `(1,1)` int32(只占 8 字节,保持 UB 中性)。

## 结果:**编译失败,候选否决**

```
loc(.../kernel.npuir:32:19): ...
error: failed to legalize operation 'memref.alloc' that was explicitly marked illegal
error: Failed to run BiShengHIR HIVM To LLVM pipeline
[ERROR] ... ERR99999 UNKNOWN application exception（原文该词此处有拼写错误，本例按规范拼写录入）
```

**根因**:`T.vand` 的第二操作数不接受 `(1,1)` 标量广播 —— 生成的 IR 里该分配无法 legalize。

## 为什么没有退而求其次用"整块 int 常量"

cos 的 fp32 在 `block_size=4096` 时已是 **48 B/元素 × 4096 = 192 KB,正好占满 UB**。
改成整块 int32 常量需 **+16 B/元素**,即 256 KB ✗;而减小 block_size 又会撞上
**栅格上限**(已有的死锁结论:`bs ≤ 4468` 对 UB vs `bs ≥ 6144` 对栅格,256m 时无解)。

## 结论

**cos 在当前结构下没有可行的优化余量**,两条路都被堵死:

| 路线 | 阻塞点 |
|---|---|
| 放大 block_size(像 ceil/floor 那样减缓冲换大块) | UB 已 100% 占满,且 bs 与栅格互相死锁 |
| 减少指令数(int 位运算替代象限算术) | `T.vand` 不支持 (1,1) 广播 ⇒ 编译失败;整块常量 ⇒ UB 溢出 |

前一轮 Stage 4 的 `final-bs4096-nc48` 结论成立。**若要继续挖 cos,需要先解除 UB 约束**
(例如把 `c_ub`/`s_ub` 的 cos/sin 结果改为按需重算、或用 half 精度暂存中间量),那是结构性重写,风险高于收益。

---

# 追加（2026-11-02 12:21 北京时间）：第 5 轮 —— 无三角函数 poly-R 链 + 按 dtype 分档 block_size

**用户给本轮设的性能目标：ratio 达到 60%**（进入时为单点 56.17% / 平均 54.40%）。
口径与本文档前文一致：`msprof op` 的 `Task Duration(us)` 是唯一 kernel 时延口径；ratio 取
`msprof op --aic-metrics=Roofline` 产出的 `visualize_data.bin` 中 `GM/L2 → GM Read + Write`
roofline 项（与 `ManifestBenchmark.profile` 完全同源，见 `ratio_from_bin.py`）。

## 0. 本轮结论

| 指标 | 进入时（C3, bs=4096） | 本轮交付（r5-polyr-dtier） | 变化 |
|---|---:|---:|---:|
| **单点口径 max ratio** | 56.17% | **69.22%** | **+13.05 pp** |
| **平均口径 ratio（n=7）** | 54.40% | **64.01%** | **+9.61 pp** |
| ≥60% 的 tune workload | 0 / 7 | **6 / 7** | +6 |
| 逐 workload 时延 | 147.8–2344.5 µs | 96.8–1502.0 µs | **1.27–1.60×** |

- **目标达成**：单点与平均两个口径都 ≥ 60%。
- **未达成项（如实登记）**：`elementwise-16m-float32` 单 workload 49.52%（比 C3 的 51.59% 低 2.07 pp，虽然时延 147.8 → 116.6 µs 快了 1.27×）。根因见 §6。

## 1. 现象重述：C3 已经不是 stall 问题

对 C3 用 `msprof op --aic-metrics=BasicInfo,ArithmeticUtilization,PipeUtilization,...` 直采（`profiles/r5/base16m`）：

| 指标 | 值 | 含义 |
|---|---:|---|
| `Task Duration` | 148.75 µs | — |
| `aiv_vec_ratio` | **0.967** | 向量管线 96.7% 占用 |
| `aiv_vec_fp32_ratio` | 0.945 | 其中 fp32 管线 94.5% |
| `aiv_mte2_ratio` / `aiv_mte3_ratio` | 0.191 / 0.107 | 搬运只有 19% / 11%，且已被覆盖 |
| `aiv_total_cycles` | 3071 / tile | 16m fp16 = 85.33 tile/核 |
| `aiv_vec_fops` | 106,336 / tile ≈ **26 记账单元** | 1 单元 = 4096 fops |
| 实测 fops/周期 | **34.6 / 64** | ratio 53.27% 的全部来源 |

⇒ 管线没有空转余量（理想也只能拿 +3.4%），ratio 低的真正原因是
**「每周期只能记 34.6 个 fops」，而模型分母按 64 算**。
所以本轮的唯一杠杆是：**提高「每周期记账 fops」（换贵指令 / 摊薄每 tile 固定开销）**，
而不是再省指令数——这也解释了历史结论「C3 砍 35% 指令、ratio 持平」。

## 2. 指令级成本标定（`probe_instr.py`，9 个变体）

同一外壳下测「+1 类指令」的边际成本：`1 vcast(加宽) + K×vmul + 1 <special> + 1 vcast(收窄)`。

| 变体 | Task Duration(µs) | `aiv_total_cycles` | 备注 |
|---|---:|---:|---|
| mul8（9 记账单元） | 45.90 | 78,482 | — |
| mul12（13 单元） | 62.98 | 109,111 | — |
| mul16（17 单元） | 81.38 | 142,179 | — |
| mul20（21 单元） | 99.14 | 173,674 | — |
| sin / cos / cast_rint / cast_floor | 72.38 / 56.70 / 39.40 / — | — | **数据不可用**，见下 |

**由 mul8→mul20 的斜率**：7933 cycles/单元/核 ÷ 85.33 tile/核 = **93.0 cycles 每条 4096 元素向量指令**
（≈ **44 fops/周期**，即单条普通指令的天花板是 68.8%），固定开销外推 ≈ 7081 cycles/核（≈2 条指令）。

**sin/cos/cast 变体的数据被编译器链折叠污染**：`vmul(·, 0.999999)` 的重复链在带三角函数时被折叠
（`probe_sin` 记账 fops 只有 45,008/tile，比 mul15 的 66,364 还少），因此**不能**用这一组下结论。
可靠的成本数据改用 §3 的真实链消融。

## 3. 真实链消融（两个错误版本只用于计时，`probes/probe_novsin.py`、`probes/probe_nocast.py`）

| 变体 | 改动 | Task Duration | cycles/tile | 记账单元/tile | 结论 |
|---|---|---:|---:|---:|---|
| base（=C3） | — | 148.75 µs | 3071 | 106,336 | — |
| novsin | `T.vsin(r,f)` → `T.vmul(r,1.0,f)` | **90.52 µs** | **1856** | 69,552 | **vsin 一项占 1215 cycles（40% 时间）且被记 9 个单元（30 fops/周期，比普通指令差 1.45×）** |
| nocast | 两处同 dtype 量化 `vcast` → `vmul(·,1.0)` | 149.51 µs | 3083 | 106,336 | **精度量化 cast 与 vmul 同价（Δ=+12 cycles 属噪声）** |

⇒ `T.vsin` 是唯一「贵指令」：吃掉 40% 周期、只贡献 34% 的记账 fops。
把它换成一段便宜多项式是**同时**降时延与提 ratio 的正道。

## 4. 候选 R5：无三角函数、无象限奇偶的 poly-R 链

数学（`cos_fwd_kernel_round5_polyr*.py` 头部注释同源）：

```
cos(x) = 1 - 2*sin^2(x/2)
y = x/2 ,  k = rint(x/(2*pi)) ,  r = y - k*pi ∈ [-pi/2, pi/2]
sin^2(y) = sin^2(r + k*pi) = sin^2(r)      # sin^2 周期为 pi ⇒ 符号被平方吸收
sin^2(r) = t*Q(t) ,  t = r^2 ∈ [0, (pi/2)^2]
⇒ cos(x) = 1 - 2*t*Q(t)                     # Q 为 Horner 多项式，无三角函数、无 mod-4、无符号位
```

相较 C3：去掉 `vsin`(1215 cycles) 与象限奇偶计算（4 条 `vmul/vcast/vsub`），
代价是 deg-5 Horner（11 条 `vmul/vadd`）；缓冲 5 → **4** 个（`x_ub/f_ub/r_ub/y_ub`），
足迹 12 B/元素（fp16/bf16）、16 B/元素（fp32）⇒ 为放大 `block_size` 留出空间。

多项式系数（`_POLY_C`/`_POLY_C7`，Chebyshev 最小二乘拟合）：

| 档位 | 系数个数 | t∈[0,(π/2)²] 上 max abs err | 用途 |
|---|---:|---:|---|
| deg-5 | 6 | **9.69e-8** | fp16/bf16（容差 1e-3 / 1.6e-2） |
| deg-7 | 8 | **9.44e-12** | 试过 fp32 填槽，结论见 §6（未采纳） |

## 5. `block_size` 才是 ratio 的第二杠杆（本轮最大收益点）

固定开销 ≈ 478 cycles/tile；tile 越大，这部分被摊得越薄：

| block_size | Task Duration(µs) | ratio | vs 4096 | 备注 |
|---:|---:|---:|---:|---|
| 4096 | 111.36 | 54.75% | — | 只换了算法链，ratio 仅 +1.5 pp（印证「砍指令不提 ratio」） |
| 6144 | 103.90 | 58.59% | +3.84 pp | |
| 8192 | 100.46 | **60.54%** | +5.79 pp | 单 workload 首次过 60 |
| **12288** | **96.74** | **62.85%** | **+8.10 pp** | 采纳（fp16/bf16） |
| 16384 | — | — | — | ❌ `ub overflow`（12 B/元素 × 16384 > 192 KB） |

fp32（16 B/元素）单独扫：12288 → 122.48 µs / 47.17%，**16384 → 116.54 µs / 49.53%**（采纳），
20480 → 116.68 µs / 49.50%（打平，说明已不是固定开销）。

## 6. fp32 的失败方向（如实登记，未采纳）

fp32 的病根：**向量管线只有 72% 占用**（fp16 是 97%），而向量工作量完全相同
（`aiv_vec_time` 89.5 µs vs 89.2 µs）；多出来的 ~30 µs 是搬运/依赖停顿，
`T.Pipelined` 与装载缓冲结构都关不掉：

| 分支 | 唯一改动 | Task Duration(µs) | ratio | 裁决 |
|---|---|---:|---:|---|
| polyr1 | fp32 走 `x_ub` 中转 + UB→UB 加宽 | 125.97 (bs 8192) | 45.88% | ❌ 多一趟不记账的 copy |
| **polyr2** | fp32 GM 直落计算缓冲 `f_ub`，出口 UB→GM 直存 | **116.54 (bs 16384)** | **49.53%** | ✅ 采纳（fp32 档位） |
| polyr2 @12288 / @20480 | 仅 block_size | 122.48 / 116.68 | 47.17 / 49.50% | 16384 已到平区 |
| polyr3 | fp32 换 deg-7（去填 28% 空槽） | 143.21 (12288) / 135.95 (16384) | 48.84 / 51.40% | ❌ 多 4 条依赖指令涨 17% 时间，空槽不可用 |
| polyr4 | `T.Pipelined(iters, num_stages=2)` + x_ub 装载 | 133.57 (6144) / 126.09 (8192) | 43.31 / 45.83% | ❌ 与不加流水逐位一致，纯向量核无重叠 |

fp32 的物理上限（`aiv_vec_time` 89.5 µs 完全隐藏搬运）= 64.5%，当前 49.53%；
要跨 60% 需要把 28% 停顿压到 8% 以内，本轮四条路（暂存结构 / block_size / 多项式阶数 / 流水）
全部实测排除，**属于已知遗留**。

## 7. 本轮「候选 vs current best」结构化对比表（gate 4 `S4-OPTLOG-COMPTABLE`）

current best（进入时）= C3 `final-bs4096-nc48`；工作负载 = `elementwise-16m-float16`（标定基准）。

| candidate_id | 唯一改动 | Task Duration(µs) | aiv_vec_ratio | 记账 fops/tile | ratio | L0 |
|---|---|---:|---:|---:|---:|---|
| `c3-bs4096` | 基准 | 148.75 | 0.967 | 106,336 | 53.27% | PASS |
| `r5-polyr-bs4096` | 换 poly-R 链 | 111.36 | — | 81,920 | 54.75% | PASS |
| `r5-polyr-bs6144` | +block_size | 103.90 | — | 81,920 | 58.59% | PASS |
| `r5-polyr-bs8192` | +block_size | 100.46 | — | 81,920 | 60.54% | PASS |
| **`r5-polyr-bs12288`** | **+block_size** | **96.74** | — | 81,920 | **62.85%** | **PASS** |
| `r5-polyr2-fp32` | fp32 GM 直落 | 116.54 | 0.72 | 76,800 | 49.53% | PASS |

（`Perf(TOps/s)` 与 `Computility(TOps/s)` 均取自 roofline bin：baseline 2.9385 / 5.5163，
polyr@12288 3.4730 / 5.5254；`AI` 由 13.03 → 10.01 Ops/Byte，仍在算力分支。）

## 8. 最终逐 workload（同一最终候选 `r5-polyr-dtier`，bs 分档 12288/12288/16384）

| workload | baseline µs | final µs | 加速 | baseline ratio | final ratio | Δ |
|---|---:|---:|---:|---:|---:|---:|
| elementwise-16m-float16 | 148.5 | 96.76 | 1.53× | 53.39 | **62.84** | +9.45 |
| elementwise-16m-bfloat16 | 150.7 | 97.42 | 1.55× | 54.60 | **65.56** | +10.96 |
| elementwise-16m-float32 | 147.8 | 116.56 | 1.27× | 51.59 | 49.52 | −2.07 |
| elementwise-64m-float16 | 582.4 | 363.31 | 1.60× | 54.44 | **66.96** | +12.52 |
| elementwise-64m-bfloat16 | 589.4 | 368.97 | 1.60× | 55.85 | **69.22** | +13.37 |
| elementwise-256m-float16 | 2316.0 | 1473.98 | 1.57× | 54.76 | **65.97** | +11.21 |
| elementwise-256m-bfloat16 | 2344.5 | 1502.02 | 1.56× | 56.17 | **67.97** | +11.80 |

- **单点 69.22%**（`elementwise-64m-bfloat16`）、**平均 64.01%**、6/7 ≥ 60%。
- 原始 profile：`profiles/final/r5/<workload>/OPPROF_*`；逐行记录：`perf_records.jsonl` 的 `round=5, phase=final, candidate_id=r5-polyr-dtier`。

## 9. 精度门禁（最终交付文件）

| 门禁 | 结果 | 备注 |
|---|---|---|
| L0（1M，三 dtype，随机） | **PASS 9/9** | 阻塞项 |
| L1（含 \|x\|~1e3/1e4 边界） | **PASS** | 阻塞项 |
| L2（N=63/4097、beyond-domain 记录） | PASS | \|x\|≤1e4 的 max_abs 7.4e-4 为 Cody-Waite 归约限制，与 C3 同源，NOT asserted |
| Boundary-π/2π/−2π（fp32/fp16） | PASS | max_diff 0 |
| smoke（1M×3 dtype） | PASS | 仍 < 15 µs，不计入平均口径 |

## 10. 本轮新增工具链陷阱（4 条，均为实测，建议回写 pattern-library/traps-*.md）

| # | 陷阱 | 报错/现象 | 规避 |
|---|---|---|---|
| 1 | 含 `@T.prim_func` 的模块里加 `from __future__ import annotations` | `TVMError: script.ir_builder.tir.Arg ... expected Object but got str`（注解被延迟成字符串） | kernel 文件不能带这个 future import |
| 2 | `@tilelang.jit` 工厂的**字符串**参数 | 同上 `expected Object but got str` | 变体/分支只能用 int 或 Python 闭包常量传递 |
| 3 | prim_func 体内 Python 级 `if/elif` 里**给缓冲区别名赋值** | `DiagnosticError: Undefined variable: src`（TVMScript 静态解析看不到分支内赋值） | 分支外先赋值，或把 chain 显式写死 |
| 4 | 同一个 jit 工厂的 if 分支里定义**多个** `@T.prim_func` | 同陷阱 1 的报错形态 | 每个变体一个顶层工厂（`probe_instr.py` 的 5 工厂写法） |
| 5 | 用 `vmul(·, 0.999999)` 串长链做「微基准」 | 编译器把重复乘常数折叠，`aiv_vec_fops` 记账失真（sin 变体记账比纯乘链还少） | 指令成本标定必须用**真实 kernel 消融**（§3） |

## 11. wrapper 采纳要求（conductor 翻转时需要成对改）

本 kernel 的 tuned 档位高于原默认值，`perf_opt/cos_fwd_kernel.py` 已把它暴露成模块级常量
`TUNED_DEFAULT_BLOCK_SIZE = {"float16": 12288, "bfloat16": 12288, "float32": 16384}`。
wrapper `tileops/kernels/elementwise/cos/cos.py` 的 `_DEFAULT_BLOCK_SIZE`（现为三档 4096）
必须同步改成同一组值，否则正式 benchmark 仍按 4096 跑（只能拿到 54.75% 一档）。

## 12. 遗留与建议

1. **fp32 49.52% 未达 60%**（唯一未达标 workload）：向量管线 72% 占用，四条优化路径实测排除（§6）。
   后续若要做，方向是「同一核内两 tile 交错（2 条独立链提升 ILP）」，需 2 套 fp32 缓冲（≈147 KB），
   本轮受时间与 UB 约束未实施。
2. **pattern-library 回写**：§10 的 5 条陷阱、§2 的「93 cycles/4096 元素均值指令、44 fops/周期」
   与「vsin = 1215 cycles + 9 记账单元」两个常数、以及「block_size 摊薄固定开销是 ratio 主杠杆」
   的量化曲线（54.75→62.85）。
3. `perf_records.jsonl` 中本轮的 `extra.ratio_percent` 已按 roofline bin 口径登记，便于后续 gate 对账。

---

# 追加（2026-11-02 14:39 北京时间）：第 6 轮 —— fp32 双流（两 tile 交错），7/7 全部 ≥60%

进入时（r5 交付）：单点 69.22%、平均 64.01%、6/7 ≥60%，唯一短板是
`elementwise-16m-float32` **49.52%**（比它自己的 C3 基线 51.59% 还低 2.07 pp）。

## 0. 本轮结论

| 指标 | r5（进入） | **r6（交付）** | 累计 vs C3 基线 |
|---|---:|---:|---:|
| 单点口径 max ratio | 69.22% | **69.14%**（噪声内持平） | 56.17 → **69.14（+12.97 pp）** |
| 平均口径（n=7） | 64.01% | **66.23%** | 54.40 → **66.23（+11.83 pp）** |
| ≥60% workload | 6 / 7 | **7 / 7** | 0/7 → **7/7** |
| fp32 单 workload | 49.52% | **63.91%** | 51.59 → 63.91（+12.32 pp） |

**本轮唯一改动**：给 fp32 加一条**双流（两 tile 交错）**执行路径；fp16/bf16 保持 r5 单流不动
（因此这 6 条 workload 的 ratio 与 r5 逐条在噪声内一致）。

## 1. 现象与归因：fp32 的搬运被串行化

fp32（`polyr1`, bs 8192）的流水线构成（`profiles/r5/diag_f32_polyr1_8192`）：

| 项 | fp32 | fp16（bs 12288, 参照） |
|---|---:|---:|
| `aiv_time` | 122.72 µs | 92.59 µs |
| `aiv_vec_time` | 89.50 µs | 89.17 µs（**向量工作量相同**） |
| `aiv_vec_ratio` | **0.724** | 0.967 |
| `aiv_mte2_time` / `aiv_mte3_time` | 18.92 / 11.98 µs | 12.56 / 6.29 µs |
| 判定 | `mte2+mte3 ≈ total − vec`（30.9 ≈ 33.2）⇒ **搬运完全没被遮住** | 搬运基本被遮住 |

根因：fp32 每元素搬 4 B（fp16 是 2 B），DMA 时间翻倍；而装载目标与计算缓冲是同一个
（WAR 依赖），编译器无法双缓冲，于是 load → compute → store 只能串行。
r5 试过的三条路（`x_ub` 中转、deg-7、`T.Pipelined`）全部无效，见 r5 §6。

## 2. 候选 R6：fp32 双流

一条**独立**的第二 tile 与第一条交错发射，装载槽分开（`x0/x1`），计算原地在 `x` 上做
（`r0/r1` 作 scratch），tile 映射 `pair p → tile 2p, 2p+1`：

```
for i in T.serial(iters):
    p = i*num_cores + cid
    if p < n_pairs:
        b0 = 2*p ; b1 = min(2*p+1, total_blocks-1)     # 奇数尾巴退化为「同一 tile 算两遍」
        copy(x->x0) ; copy(x->x1)
        <19 步链，x0/r0 与 x1/r1 逐条交错>              # 38 条向量指令
        copy(x0->y) ; copy(x1->y)
```

- 每流 `x(4B)+r(4B)=8 B/元素`，两流 16 B/元素；但实测编译器还要为多缓冲多要一档
  （12288 报 `ub overflow, requires 2359296 bits while 1572864 bits` ⇒ 实需 **24 B/元素**），
  故 **8192 = 192 KB 正好压线**。
- fp32 档位常量 `TUNED_DEFAULT_BLOCK_SIZE["float32"] = 8192`（唯一事实源）。

| 变体 | block_size | Task Duration(µs) | ratio | 裁决 |
|---|---:|---:|---:|---|
| r5 fp32（单流直落） | 16384 | 116.54 | 49.53% | 进入时 |
| **r6 fp32 双流** | **8192** | **89.86** | **64.31%** | ✅ 采纳 |
| r6 fp32 双流 | 6144 | 114.96 | 50.33% | ❌ 档位太小，固定开销重新占主导 |

⇒ 双流把 fp32 的 **116.5 → 89.9 µs（−23%）**、ratio **+14.8 pp**，也印证了
「fp32 的瓶颈是搬运没重叠，不是指令数」。

## 3. 本轮失败方向

| 分支 | 做法 | 结果 | 裁决 |
|---|---|---|---|
| `probe_dual` 微基准 | 两条独立纯乘链交错 vs 单链同指令数 | 两条链被编译器 DCE（只记 1 个单元，`fops/cyc`=5.3） | ❌ 微基准不可用；改用真实双流候选直接验证 |
| r7 slim（fp16/bf16） | 去掉 `y_ub`、复用 `x_ub` 出出口 ⇒ 3 缓冲，推到 16384 | 16m fp16 94.32 µs/**64.41%**（+0.59）；64m bf16 362.35/**70.41%**（+1.27）；但 **256m bf16 1704.05 µs/59.89%（−8.11 pp）** | ❌ 形状相关回退（3 缓冲在 16384 时 UB 压不住双缓冲，大 shape 才暴露），不采纳 |
| r7 slim @18432 | 继续放大档位 | 107.38 µs / 56.62%（16m fp16） | ❌ |

## 4. 最终逐 workload（同一最终候选 `r6-dual-dtier`）

fp16/bf16 = 单流 + bs 12288；fp32 = 双流 + bs 8192。

| workload | 路径 | baseline µs | final µs | 加速 | baseline ratio | final ratio | Δ vs 基线 |
|---|---|---|---:|---:|---:|---:|---:|
| elementwise-16m-float16 | 单流 12288 | 148.5 | 95.32 | 1.56× | 53.39 | **63.82** | +10.43 |
| elementwise-16m-bfloat16 | 单流 12288 | 150.7 | 97.04 | 1.55× | 54.60 | **65.82** | +11.22 |
| elementwise-16m-float32 | **双流 8192** | 147.8 | 90.42 | 1.63× | 51.59 | **63.91** | +12.32 |
| elementwise-64m-float16 | 单流 12288 | 582.4 | 363.29 | 1.60× | 54.44 | **66.96** | +12.52 |
| elementwise-64m-bfloat16 | 单流 12288 | 589.4 | 369.39 | 1.60× | 55.85 | **69.14** | +13.29 |
| elementwise-256m-float16 | 单流 12288 | 2316.0 | 1474.06 | 1.57× | 54.76 | **65.97** | +11.21 |
| elementwise-256m-bfloat16 | 单流 12288 | 2344.5 | 1501.50 | 1.56× | 56.17 | **68.00** | +11.83 |

原始 profile：`profiles/final/r6/<workload>/OPPROF_*`；逐行记录：
`perf_records.jsonl` 的 `round=6, phase=final, candidate_id=r6-dual-dtier`。

精度门禁（最终交付文件）：L0/L1/L2/Boundary/smoke **全部 PASS**（双流路径涉及奇偶 tail 映射，
L2 的 N=63/4097 与 Boundary 用例已覆盖）。

## 5. wrapper 采纳要求（更新）

`tileops/kernels/elementwise/cos/cos.py` 的 `_DEFAULT_BLOCK_SIZE` 必须改成
**`{"float16": 12288, "bfloat16": 12288, "float32": 8192}`**（与 kernel 侧
`TUNED_DEFAULT_BLOCK_SIZE` 成对引用）。注意 fp32 走的是双流路径，档位是 8192 而不是 r5 的 16384。

## 6. 遗留

1. fp32 双流的档位窗口很窄（8192 可用、12288 UB 溢出、6144 打平不了），因为编译器多缓冲要多一档 UB。
   若后续能把每流缓冲再压（例如 fp32 也用 3 缓冲复用），fp32 还有上探空间（当前 63.91%，上限 68.8%）。
2. fp16/bf16 的 3 缓冲瘦身只在 16m/64m 有 ~+1 pp 收益、在 256m 回退 8 pp，需要按 shape 分档才敢用；
   本轮不做（会引入 shape 条件路径，收益与风险不匹配）。

---

# 追加（2026-11-02 16:05 北京时间）：第 7 轮 —— 走官方 pytest 路径复核，fp32 双流收益不可复现

**触发**：用户要求「把 wrapper 档位改成 tuned 并走官方路径复测」。
**纪律说明**：Stage 4 契约里 wrapper 只读，本轮改动**由用户明确授权**（原话「要」），下文留完整 diff。

## 0. 结论先行

| 口径 | 数字 | 备注 |
|---|---|---|
| **官方 pytest 路径（最终）** | 单点 **67.98%** / 平均 **63.96%** / **6/7 ≥60%** | fp32 单 workload **53.53%** 未达 60 |
| 直跑 bench.py 口径（此前上报） | 单点 69.14% / 平均 66.23% / 7/7 | **7/7 不成立，此处更正** |
| 16m fp16 官方 | 96.98 µs / **62.70%** | 直跑 95.32 / 63.82，一致 |
| 256m bf16 官方（单点来源） | 1501.84 µs / **67.98%** | 直跑 1501.50 / 68.00，一致 |

⇒ **fp16/bf16 六条 workload 在官方路径上完全复现**；**fp32 一条在官方路径上掉到 53.5%**（直跑是 63.9%）。

## 1. wrapper 改动（用户授权）

`tileops/kernels/elementwise/cos/cos.py`：

```diff
-_DEFAULT_BLOCK_SIZE = {"float32": 4096, "float16": 4096, "bfloat16": 4096}  # C4: 4096 + 流水
+_DEFAULT_BLOCK_SIZE = {"float32": 8192, "float16": 12288, "bfloat16": 12288}
```

与 kernel 侧 `TUNED_DEFAULT_BLOCK_SIZE` 成对一致。

## 2. 官方首轮（r6 相邻映射 `2p / 2p+1`）

`python3 -m pytest benchmarks/ops/bench_cos.py -q -p no:cacheprovider` → **10 passed / 246.59 s**
（报告 `profile_run_msprof_20261102_073700.json`）

| workload | bs | latency(µs) | ratio |
|---|---:|---:|---:|
| elementwise-16m-float16 | 12288 | 96.60 | 62.94% |
| elementwise-16m-bfloat16 | 12288 | 98.48 | 64.83% |
| **elementwise-16m-float32** | 8192 | **109.24** | **52.90%** |
| elementwise-64m-float16 | 12288 | 371.45 | 65.46% |
| elementwise-64m-bfloat16 | 12288 | 378.12 | 67.51% |
| elementwise-256m-float16 | 12288 | 1474.72 | 65.94% |
| elementwise-256m-bfloat16 | 12288 | 1501.82 | 67.98% |

## 3. fp32 异常诊断（结论：停顿，不是降频、不是指令数）

同一份交付件、同一档位 8192，两条路径：

| 路径 | duration | cycles | fops | vec_ratio | mte2/mte3 |
|---|---:|---:|---:|---:|---|
| 直跑（早先两次） | 90.08 / 90.46 µs | 151 662 / 151 630 | 6 545 344 | — | — |
| 直跑（4 连跑，本次） | 108.34 / 108.58 / 108.72 / 108.84 µs | 183 602–183 994 | **6 545 344（完全相同）** | 0.787 | 17.5 / 11.9 µs |
| 官方 pytest | 109.24 / 107.96 µs | — | — | — | — |

- `Current Freq` 两次都是 **1800 MHz**（非降频）；
- fops 完全相同 ⇒ 指令数没变，**周期多 21%（151.6k → 184.0k）是停顿**；
- 说明 fp32 双流的收益对**访存布局/地址**敏感，同一二进制在不同调用路径下落在 90 µs 或 108 µs 两档。

## 4. 尝试：远距映射 r6b（`b0 = p`，`b1 = p + n_pairs`）

目的：避免相邻两流（2p/2p+1）的 32 KB 块落在同一 DRAM channel/L2 set 上。

| 采样 | duration | ratio |
|---|---:|---:|
| 直跑 run1 / run2 | **90.48 / 90.60 µs** | 63.87 / 63.78% |
| 官方 pytest（10 passed / 252.00 s） | **107.96 µs** | **53.53%** |

⇒ 远距映射在直跑下稳定（2/2 快档），但**官方路径仍然慢**。
结论修正：慢档不是相邻映射独有，**官方 wrapper 路径下的访存布局才是诱因**（两版映射在官方路径都 ~108 µs）。

## 5. 裁决

- 交付件采用**远距映射**版 `cos_fwd_kernel.py`（sha256 `a15873bf52e650e9a5b758513c661a3675c6e0fcb35e500b27e2c38faeee53cb`）：
  官方 fp32 53.53% 略优于 r5 单流的官方 52.90%，且直跑更稳（90.5 vs 116.5 µs）。
- **不上报 7/7**：官方口径为 **6/7**，fp32 53.53% 未达 60%，如实登记。

## 6. 遗留

1. fp32 要稳定 ≥60%，需要把「双流收益在官方路径下不可复现」这件事挖到底：候选方向
   （a）输出张量在 wrapper 路径下每次重新分配 → 排查 out_idx 分配对齐；
   （b）给两条流的读/写基址加固定 padding（例如 2 MB 对齐）以消除 L2 set 冲突；
   （c）放弃双流、回到单流并换更大档位（52.9% 稳定但低于 60）。
2. 官方路径的其余 6 条已稳定 ≥62%，不受影响。
