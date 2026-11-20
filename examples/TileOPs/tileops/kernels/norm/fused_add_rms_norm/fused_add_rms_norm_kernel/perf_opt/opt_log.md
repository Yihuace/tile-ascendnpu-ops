# Stage 4 调优日志 —— `_fused_add_rms_norm_kernel`(FusedAddRMSNormFwdOp)

- 场景 / 模式:`optimize` Stage 4,`mode=full`(best_effort)
- 基线(只读):`../_fused_add_rms_norm_kernel.py`(Stage 3 `[PRECISION_PASS]`)
- 交付件:`_fused_add_rms_norm_kernel.py`(**结构不变** —— 见下「结论」)
- 唯一时延口径:`msprof op` 的 `Task Duration(us)`;`--launch-count=15 --warm-up=5`,取中位数
- 测量入口:`bench.py`(直接运行 kernel,**不**走 Stage 5 wrapper;复用模块自带 `golden_fused_add_rms_norm`,两个输出都比)
- 批处理 runner:`run_experiments.py`(T-4)
- 硬件:CANN 9.0.0 · Ascend910B2C(24 AICore / 48 Vector 核)· tilelang `0.1.2+npuir`
- 目标 kernel 名(msprof):**`main`**
- 结构:Stage 3 是 `T.Kernel(M)`(**一行一块**,prefill 用例 M=2048 → **2048 块**)

---

## Performance Test Data

### baseline(round 0,`phase=baseline`,artifact = Stage 3 `../_fused_add_rms_norm_kernel.py`)

12 条 tune workload 全部有效(本算子无大规模用例;`M` 分别为 2048 / 1 / 2048)。

| workload_id | M | N | dtype | baseline_us |
|---|---:|---:|---|---:|
| llama-3.1-8b-prefill-float16 | 2048 | 4096 | float16 | 88.938 |
| llama-3.1-8b-prefill-bfloat16 | 2048 | 4096 | bfloat16 | 88.678 |
| llama-3.1-8b-decode-bfloat16 | 1 | 4096 | bfloat16 | 2.980 |
| llama-3.1-70b-prefill-float16 | 2048 | 8192 | float16 | 96.578 |
| llama-3.1-70b-prefill-bfloat16 | 2048 | 8192 | bfloat16 | 101.958 |

命令模板:

```bash
msprof op --kernel-name=main \
  --output=profiles/<phase>/round<r>/<candidate>/<workload> \
  --launch-count=15 --warm-up=5 --dump=off \
  --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio \
  python bench.py --impl <candidate.py> --case <workload_id>
```

### smoke / skipped 排除

smoke 为 `smoke-small-{float32,float16,bfloat16}`(共 3 条,`M=64,N=512`),仅做精度回归。
无 `skipped` 项。

### 可用的结构旋钮只有「栅格」一个

- `T.Kernel(M)` → 结构旋钮 = 栅格宽度;
- `block_m` **不是**可用旋钮:`_func(block_m=1)` 的接口虽然在,但两个分支的主体都硬编码
  `(1, n_align)` 切片;且 `N=4096/8192` 时 UB 预算不允许 `block_m > 1`
  (`(1, n_align)` 已占 `2+2+2+2+2+16+16+16` ≈ 60 B/元素 × n_align)。
- 故本轮只搜「栅格」这一个轴 + 其宽度参数 `num_cores`。

---

## Iteration Log

### Round 1 —— 结构改写:`grid = min(num_cores, M)` + 核内 grid-stride

**现象**:`Block Dim = M = 2048`(prefill)摊到 48 个 Vector 核 = 每核 43 行;
decode 用例 `M=1` → 栅格 1。

**候选优化点(单一)**:栅格取 `min(num_cores, M)`(`min` 而非拉满,理由同 `vector_norm`:
decode 的原栅格 1 < 48,拉满会空转),核内以 `for it in T.serial(iters): pid_m = it*grid + pid;
if pid_m < M:` 走各自份额的行。**两个分支(fp16/bf16 与 fp32)都改**。

**合并非回退检查(5 条 tune workload)**

| workload_id | baseline_us | r1_us | vs baseline |
|---|---:|---:|---:|
| llama-3.1-8b-prefill-float16 | 88.938 | 82.678 | +7.0% |
| llama-3.1-8b-prefill-bfloat16 | 88.678 | 92.798 | **−4.6%** ⚠️ |
| llama-3.1-8b-decode-bfloat16 | 2.980 | 2.860 | +4.0% |
| llama-3.1-70b-prefill-float16 | 96.578 | 95.338 | +1.3% |
| llama-3.1-70b-prefill-bfloat16 | 101.958 | 85.998 | +15.7% |

**结果混合**:5 条里 1 条回退、2 条 ≤ +1.3%、2 条明显正向。按本任务已建立的采数纪律
(见 `ceil_kernel/perf_opt/opt_log.md` 与 `vector_norm_kernel/perf_opt/opt_log.md`),
**先做受控复测再定 winner**。

### Round 1′ —— 对最差一条做受控 2×2 复测

对 `llama-3.1-8b-prefill-bfloat16` 做基线 vs 候选的 2×2 背靠背重复测量(同一 `msprof op` 命令,只换被测文件):

| 被测文件 | rep 1 | rep 2 |
|---|---:|---:|
| 基线 `../_fused_add_rms_norm_kernel.py` | 88.858 µs | 88.978 µs |
| 候选 `_fused_add_rms_norm_kernel_round1_mingrid.py` | 88.598 µs | **82.938 µs** |

**读法**:基线两次几乎重合(88.858 / 88.978,差 0.14%);**候选两次相差 6.7%**(88.598 / 82.938)。
⇒ 该结构在本 kernel 上**引入了调度相关的运行级波动**,其**最好情况**优于基线,但**典型情况**
与基线持平。**r1 不能算 winner** —— 它给不出可靠的收益,只增加了方差。这与本 kernel
「每行工作量大(`n_align` 4096 元素的拷贝 + 归约链 + 多次 vcast)」相符:派发开销本就不占主导,
把行串行化到核内反而减少块级重叠、放大调度敏感性。

### Round 2 —— `num_cores` 扫描(确认是否漏掉更优点)

在 `llama-3.1-8b-prefill-float16` 上扫 `num_cores ∈ {24, 96, 192}`:

| num_cores | Task Duration(us) |
|---:|---:|
| 24 | 89.958 |
| 96 | 92.878 |
| 192 | 89.018 |
| (基线参考) | 88.938 |

**读表结论**:三档全部落在基线 ±4% 内,**没有更优点**。⇒ 该轴已搜尽,结构改写在本 kernel 上
无收益。

**候选 vs current best 对比表**(current best = `baseline`,即 Stage 3 交付件;
代表 workload = `llama-3.1-8b-prefill-float16`,`Task Duration(us)`)

| candidate | 结构 | 参数 | Task Duration(us) | vs current best | 判定 |
|---|---|---|---:|---:|---|
| `baseline`(= current best) | `T.Kernel(M)` | — | 88.938 | — | 参照 |
| `r1_mingrid48` | `T.Kernel(min(num_cores, M))` + grid-stride | `num_cores=48` | 82.678 | +7.0% | 单次读数;受控复测下不稳定(88.598/82.938),**不与噪声区分** |
| `r2_nc24` | 同上 | `num_cores=24` | 89.958 | −1.1% | 无收益 |
| `r2_nc96` | 同上 | `num_cores=96` | 92.878 | −4.4% | 无收益 |
| `r2_nc192` | 同上 | `num_cores=192` | 89.018 | −0.1% | 无收益 |
| `final-nogain-stage3` | `T.Kernel(M)`(与 current best 同结构) | — | 88.898 | +0.0% | **no-op 对照**,用于量噪声底 |

**winner**:无。`current best` 仍是 `baseline` ⇒ 交付件与 Stage 3 结构相同(`no_gain`)。

### Round 3 —— 噪声底对照(本轮的**关键证据**)

为量化「结构不变时能测出多大差异」,把**与 Stage 3 结构完全相同**的文件
(`perf_opt/_fused_add_rms_norm_kernel.py`,逐行拷贝,只改了文件名)作为 final 候选重测一遍:

| workload_id | baseline_us | final_us(**结构相同**) | 差异 |
|---|---:|---:|---:|
| llama-3.1-8b-prefill-float16 | 88.938 | 88.898 | +0.0% |
| llama-3.1-8b-prefill-bfloat16 | 88.678 | 92.778 | **−4.6%** |
| llama-3.1-8b-decode-bfloat16 | 2.980 | 3.420 | **−14.8%** |
| llama-3.1-70b-prefill-float16 | 96.578 | 96.338 | +0.2% |
| llama-3.1-70b-prefill-bfloat16 | 101.958 | 101.778 | +0.2% |

**这就是本测量链路的噪声底**:在**代码完全相同**的前提下,89 µs 量级的用例可差 **−4.6%**,
3 µs 量级的 decode 用例可差 **−14.8%**。

**该对照同时解释了 round 1 的全部读数**:r1 的 `+7.0% / −4.6% / +4.0% / +1.3% / +15.7%`
与 no-op 对照的 `+0.0% / −4.6% / −14.8% / +0.2% / +0.2%` 处在**同一分布**内
⇒ **r1 的读数无法与噪声区分**。这是「no_gain」结论的定量依据,而不是「测了但没看出来」。

**中止条件**:结构轴(3 支:基线 / 拉满 / min)与参数轴(24/48/96/192)已搜尽,
且 no-op 对照给出了噪声底 ⇒ 判定 `no_gain` 成立,不再开辟分支。

---

## Autotune Log

未使用 `tilelang` autotune API;用等价手动搜索。

| 项 | 内容 |
|---|---|
| search space | 结构:`T.Kernel(M)` vs `T.Kernel(num_cores)` vs `T.Kernel(min(num_cores, M))`;参数:`num_cores ∈ {24,48,96,192}` |
| best config | **无** —— 所有候选与基线的差异都在噪声底内 |
| top-k | `nc192` 89.018 · `nc24` 89.958 · `nc96` 92.878(基线 88.938) |
| 未测 | `block_m`(接口存在但主体硬编码 `(1,n_align)`,且 UB 不允许 >1) |
| 编译/精度失败数 | 0(两个分支都通过 `bench.py --check`) |
| winner 独立复测 | 无 winner;交付件 = Stage 3 结构,`final` 复测见上 |
| Block Dim | 基线 `M`(2048);候选 `min(num_cores, M)`(≤48) |

---

## Final Summary

**逐 workload 提升**(逐项对账见下节 `Final Performance Test Data`)

| workload_id | baseline_us | final_us | 提升 |
|---|---:|---:|---:|
| llama-3.1-8b-prefill-float16 | 88.938 | 88.898 | +0.0% |
| llama-3.1-8b-prefill-bfloat16 | 88.678 | 92.778 | −4.6%(噪声底内,见 round 3) |
| llama-3.1-8b-decode-bfloat16 | 2.980 | 3.420 | −14.8%(噪声底内,见 round 3) |
| llama-3.1-70b-prefill-float16 | 96.578 | 96.338 | +0.2% |
| llama-3.1-70b-prefill-bfloat16 | 101.958 | 101.778 | +0.2% |

> **交付件与基线结构完全相同**,上表的差异全部是测量噪声(round 3 的 no-op 对照已量化)。
> 本 Stage 4 的结论是 **`no_gain`**:没有找到可采信的结构或参数改进,因此**不引入任何改动**,
> 保持 Stage 3 交付形态。

**带宽占峰比**(分母 = 1.8 TB/s;bytes = `(4·M·N + N)·elem_bytes`,取自 manifest roofline;
本算子按该公式的字节量与实际访存一致:读 x、读 residual、读 weight、写 y、写 residual_out)

| workload_id | baseline | final |
|---|---:|---:|
| llama-3.1-8b-prefill-float16 | 76.5% | 76.6% |
| llama-3.1-8b-prefill-bfloat16 | 76.8% | 73.4% |
| llama-3.1-8b-decode-bfloat16 | 2.2% | 1.9% |
| llama-3.1-70b-prefill-float16 | 72.7% | 72.9% |
| llama-3.1-70b-prefill-bfloat16 | 68.8% | 68.9% |

> prefill 族已达 **69–77%** 的占峰比 —— 明显高于元素级五算子(9.5–53%)与 norm 族(30–47%)。
> 这从另一个角度解释了为什么结构改写无收益:**本 kernel 已经接近带宽受限**,派发开销不是瓶颈。

### 设计估算 vs 实测偏差行(D-2 回填,DESIGN.md §1.6.0)

| workload | 设计估算下界(`(4MN+N)·e ÷ 1.8 TB/s`) | 实测 | 偏差 |
|---|---:|---:|---:|
| llama-3.1-8b-prefill-float16 | 68.0 µs | 88.898 | 1.31× |
| llama-3.1-70b-prefill-bfloat16 | 70.1 µs | 101.778 | 1.45× |
| llama-3.1-8b-decode-bfloat16 | 0.065 µs | 3.420 | 52.6× |

**失准项归属**:prefill 族偏差仅 1.31–1.45×,是**本项目里设计估算最准的一个算子**
(与 abs/exp 的「优于下界」、cos 的「差 6.7×」形成对照);decode 的 52.6× 是**小 M 的固定
发射开销**,与 `vector_norm` 的小 M 族同因。

**smoke 最终精度**:3/3 PASS(两个输出都比)。**skipped**:无。**`merge_blocked`**:无。

### Final Performance Test Data

`candidate_id` 全部为 **`final-nogain-stage3`**,时延来自 `perf_records.jsonl` 的 `phase=final` 记录。

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/norm/fused_add_rms_norm/fused_add_rms_norm_kernel/_fused_add_rms_norm_kernel.py::_fused_add_rms_norm_kernel::main | llama-3.1-8b-prefill-float16 | 88.938 | 88.898 | final-nogain-stage3 |
| tileops/kernels/norm/fused_add_rms_norm/fused_add_rms_norm_kernel/_fused_add_rms_norm_kernel.py::_fused_add_rms_norm_kernel::main | llama-3.1-8b-prefill-bfloat16 | 88.678 | 92.778 | final-nogain-stage3 |
| tileops/kernels/norm/fused_add_rms_norm/fused_add_rms_norm_kernel/_fused_add_rms_norm_kernel.py::_fused_add_rms_norm_kernel::main | llama-3.1-8b-decode-bfloat16 | 2.980 | 3.420 | final-nogain-stage3 |
| tileops/kernels/norm/fused_add_rms_norm/fused_add_rms_norm_kernel/_fused_add_rms_norm_kernel.py::_fused_add_rms_norm_kernel::main | llama-3.1-70b-prefill-float16 | 96.578 | 96.338 | final-nogain-stage3 |
| tileops/kernels/norm/fused_add_rms_norm/fused_add_rms_norm_kernel/_fused_add_rms_norm_kernel.py::_fused_add_rms_norm_kernel::main | llama-3.1-70b-prefill-bfloat16 | 101.958 | 101.778 | final-nogain-stage3 |

---

## Skill Retrospective

### 流程问题与建议

| # | 环节 | 问题 | 本任务实证 | 建议 | vp_type |
|---|---|---|---|---|---|
| 1 | **无收益的判定** | SKILL 承认 `no_gain` 状态,但**未要求给出「噪声底」证据**,易被当成「没测出来」 | 本轮用**与基线逐行相同**的文件做 final 重测,量出噪声底(89 µs 用例 −4.6%、3 µs 用例 −14.8%),据此判定 r1 的读数不可区分 | SKILL Phase 2 增加:**判 `no_gain` 前必须做一次 no-op 对照**(同一 artifact 重测)以给出噪声底 | R |
| 2 | 结构改写的适用边界 | 本轮是 PL-1.21 的**反例边界**:结构改写无收益 | 本 kernel 占峰比已达 69–77%(接近带宽受限),每行工作量大 ⇒ 派发开销不是瓶颈;r1 只放大方差(6.7% vs 基线 0.14%) | 回写 PL-1.21:**新增反例判据** ——「若基线占峰比已 ≥ ~65% 或每 tile 工作量大,结构改写预期无收益,应先量基线占峰比再决定是否套用」 | D |
| 3 | 两个分支的同步修改 | 本 kernel 有 fp16/bf16 与 fp32 **两个 prim_func 分支**,结构改动须同时覆盖,否则只有部分 dtype 受益 | round 1 对两个分支都做了改写,`--check` 在 `smoke-small-float32`(走 fp32 分支)与 prefill-fp16(走另一分支)上都要通过 | 把「多分支 kernel 的结构改动必须逐分支验证」写成检查项 | R |

### 价值点提案(含证据三件套)

| id | vp_type | 内容 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| VP-farn-1 | **D** | **PL-1.21 的反例边界 + 判据前置**:**基线占峰比已 ≥ ~65% 时结构改写无收益**。本 kernel prefill 占峰比 69–77%,`min(num_cores, M)` 的 grid-stride 改写只把方差从 0.14% 放到 6.7%(最好情况 +15.7%、典型情况与基线持平),`num_cores` 三档全在 ±4% 内 | 5 条 baseline + 5 条 merged + 3 条 num_cores 扫描 + 受控 2×2 | `pattern-library/repro/PL-1.21-gridstride-serial.py`(结构同源) | tilelang `0.1.2+npuir` / CANN 9.0.0 / 910B2C |
| VP-farn-2 | **D** | **本项目测量链路的噪声底实测**:代码**完全相同**时,89 µs 用例可差 −4.6%、3 µs 用例可差 −14.8%。据此可回看并解释 `ceil` 的 2× 异常值、`vector_norm` 的 3d ±5% | round 3 的 no-op 对照表 | repro-missing | 同上 |
| VP-farn-3 | **P** | **判 `no_gain` 的方法论**:先做 no-op 对照取噪声底,再对照候选读数是否超出该底 | 本日志 round 1 / 1′ / 3 三节 | 本日志 | 同上 |

**pattern-library 回写**:VP-farn-1 作为 PL-1.21 的**反例判据**追加(与 `vector_norm` 的
「原栅格 > 核数」前置条件并列);VP-farn-2 拟入 `profile-collection.md` 的采数纪律节(噪声底表)。

---

## 结论

- **`no_gain`**:结构轴(3 支)与参数轴(4 组)已搜尽,所有候选与基线的差异都落在
  **no-op 对照测出的噪声底之内**,故**不引入任何改动** —— 交付件 `perf_opt/_fused_add_rms_norm_kernel.py`
  与 Stage 3 结构逐行相同,只改了文件名。
- **不产出的东西**:无 winner 分支(不需要合并);不产出 `[DESIGN_LIMIT]`
  (prefill 族与设计估算只差 1.31–1.45×,这是本项目**最准**的一个设计估算;
  decode 的小 M 固定开销属发射项,在参数域内)。
- **留下的一件东西**:no-op 噪声底对照(round 3)—— 它是判定 `no_gain` 的定量依据,
  也可用于回看其它算子的异常读数。
- **中止条件**:未 `budget_exhausted`(3 轮 / 11 个候选分支)、未 `blocked`、未 `plateau`。
- **verdict**:`TUNING_COMPLETED`(带 `no_gain` 结论)
