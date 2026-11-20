# Stage 4 调优日志 —— `TopkSelectorOp`(`_topk_selector_kernel`)

- **算子**:`TopkSelectorOp`(manifest `attention_indexing.yaml`),kernel 目录
  `tileops/kernels/attention_indexing/topk_selector/_topk_selector_kernel/`
- **基线交付件**:`../_topk_selector_kernel.py`(Site 3,sha256 `2bdf44cb…`)
- **本轮交付件**:`perf_opt/_topk_selector_kernel.py`(sha256 `b83dca42…`)
- **主指标**:`msprof op` 的 `Task Duration(us)`(**唯一**时延口径;不采 NPU event、不用端到端时间)
- **Ratio 口径**(用户指定):带宽占峰比 = `manifest roofline bytes ÷ latency ÷ 1.8 TB/s`,
  `bytes = 4*M*N + 4*M*k`(`manifest/attention_indexing.yaml` 的 roofline)
- **工具链戳**:tilelang `0.1.2+npuir+7fc4a3c8` · CANN 9.0.0 · Ascend910B2C(24 Cube / 48 Vector)·
  venv `/root/venv-npuir`(`env -u PYTHONPATH`)· `TILELANG_ASCEND_MODE=Developer`
- **测量结构说明**:本算子由 **两个 kernel + host 侧 40 轮二分**组成,两个 prim_func 同名 `main`,
  `msprof op` 一次只能干净地对一个 kernel 名计数,因此 bench 分 `--mode count|select` 分别采集,
  再按算法实际 launch 次数折算:
  **`op_us = n_count_launches × count_us + 1 × select_us`**(两代实现的 count launch 次数均为 40)

---

## 1. Performance Test Data

### 1.1 workload 清单(`workload_inventory.json`)

| workload_id | shape(M,N) | K | kind | 说明 |
|---|---|---|---|---|
| `smoke-256` | 8, 256 | 8 | smoke | 仅精度,不做性能采集 |
| `topk-1024` | 16, 1024 | 16 | tune | 整行驻留 |
| `topk-4096` | 8, 4096 | 64 | tune | 整行驻留(gather 路径) |
| `topk-8192-tiled` | 8, 8192 | 64 | tune | manifest 标 tiled;gather 路径下单 tile 驻留 |

### 1.2 baseline(`phase=baseline`,round 0,sha256 `2bdf44cb`)

| workload | count(单次 launch) | count 总 | select | op_us | Ratio |
|---|---:|---:|---:|---:|---:|
| topk-1024 | 16.861 | 674.4(×40) | 494.260 | **1168.700** | 0.0032% |
| topk-4096 | 24.561 | 982.4(×40) | 981.499 | **1963.939** | 0.0038% |
| topk-8192-tiled | 35.281 | 1411.2(×40) | 1944.538 | **3355.778** | 0.0044% |

raw profile:`profiles/baseline/round0/baseline_2bdf44cb/*/{count,select}/OPPROF_*`;
逐行记录:`perf_records.jsonl`(append-only)。

**首个现象**:select kernel 是最大项(42%–58%),不是 count;这决定了后面的优化顺序。

---

## 2. Iteration Log

### Round 1:把 40 轮二分搬进核内(目标:40 次 launch → 1 次)

| 分支 | 做法 | 结果 | 证据 |
|---|---|---|---|
| `r1a_unrolled` | Python 层展开 40 份二分体 | ❌ **构建期 SIGSEGV**(exit 139) | `logs/round1_l0.log` |
| `r1b_serial` | 改为运行期 `T.serial(rounds)`(仅 1 份体) | ❌ **同样 exit 139** | `logs/round1b_l0.log` |

最小探针 `probe_inbranch_rounds.py` 把边界定死:

| rounds | 结果 |
|---:|---|
| 1 | ✅ 编译 + 运行(thr=892.5,1 轮未收敛属预期) |
| 2 | ⛔ `bishengir-compile` 崩溃(LLVM 栈回溯,exit 1) |

⇒ **与“是否展开”无关:二分体出现 ≥2 份即崩**(展开与运行期循环都崩)。
这**修正**了 `2OPS_GOAL.md` §1.2 里“无收尾 × 40 份二分体 ✅ 能构建”的记录——在**本工具链**上,
无收尾的 40 份二分体同样崩;该表当时所用的探针体与本轮不同(本轮体含 `reduce min/max` 与 (1,1) v-op 链)。

**结论**:核内二分在本工具链不可行,**40 次 host launch 是该算法在此环境下的硬成本**。

### Round 2:行并行(目标:把行映射到 block,利用 48 个 Vector 核)

行并行的可行性先验:**独立探针** `probe_multiblock_rows.py` 三种形态全 PASS

| 形态 | 结果 |
|---|---|
| `grid1`(1 block + 串行行) | PASS |
| `gridM`(`T.Kernel(M)`,一行一 block) | PASS |
| `grid2`(网格步进,2 block) | PASS |

据此写了 `_round2_rowparallel.py`(count 与 select 都按行分块)。**L0+L1 结果:7 例里只有 `k=N` 通过**,
其余 6 例 FAIL。用 `probe_round2_isolate.py`(宿主给精确阈值 + monkeypatch 网格)隔离:

| 组合(M=8,N=1024) | count | select |
|---|---|---|
| grid=1 | PASS | PASS |
| grid=M | ⛔ **FAIL**:`[16, 7.0e-44, 16, 2.1e-43, 16, 3.3e-43, 16, 16]`(未初始化/次正规垃圾) | PASS |

⇒ **多 block 的 GM 标量写是间歇性不可靠**:同一形态在独立探针里 PASS 过一次,换输入/换调用便丢行。
(dropped 行的读回值是次正规垃圾,说明该 block 根本没写。)Stage 3 的“多 block 丢行”结论**成立**,
本轮补上了“间歇性 + 与 shape/调用相关”的性质。**行并行不可作为交付基础,rollback。**

另两条被量化后否掉的方向(不再单独开分支):

| 方向 | 量化依据 | 结论 |
|---|---|---|
| 多阈值批量(k-ary,一次 launch 算 R 个阈值) | 由 baseline 解出**单次 launch 固定开销 ≈ 10.7 µs、单趟计算 ≈ 6.1 µs**(16.861 = H+W,35.281 = H+4W) | 二分是**pass 数最少**的方案(40 趟);R-ary 的 `R×log_R(2^40) ≥ 40`,只会增加总趟数 ⇒ 无法用“少 launch”换取收益 |
| 提前收敛退出(lo==hi 即停) | 宿主侧数值模拟:三个 workload 收敛所需轮数为 **60 轮**(k=1 才 25 轮) | 40 轮本身就不够收敛到 lo==hi,提前退出**省不到轮数** |

### Round 3:`select` 压实去掉标量 `cumsum + 散射`(采用)

**现象**:build 期编译器自己告警 `Op 'hivm.hir.vcumsum' will execute by scalar instruction with low effiency`;
散射 `scratch[0, sposi[0,j]] = …` 同样是逐元素标量写。

**做法**(`_round3_gather_select.py` → `perf_opt/_topk_selector_kernel.py`):利用**算子契约是集合**
(顺序无关),把“压缩”改成“逐槽取最小已选索引”,全程向量算子、**零散射**:

```text
mask = (x >= thr)                      # 每行一次 vcmp(标量操作数,省掉 diff 缓冲)
for q in T.serial(K):                  # K 个输出槽,静态上界
    cand  = mask ? iota : BIG           # vselect
    cand2 = cand <= last ? BIG : cand   # vcmp + vselect(排除已取)
    m     = reduce_min(cand2)           # vector reduce
    out[r,q] = m ; last = m             # 一次标量 store(不是 N 次)
```

过程中定位到三个真实陷阱(都有最小证据):

| 现象 | 根因 | 处置 |
|---|---|---|
| `float(j)` 报 `float() argument must be … not 'Var'` | `T.Parallel` 的循环变量是 TIR Var | iota 改成 `cumsum(ones) - 1` 构造,且只建一次 |
| 输出恒为 `[12,12,12,12]`(同一索引重复) | `mask[0, redi[0,0]] = 0`(运行期索引的标量散射写)**被静默丢弃** | 改为**单调过滤**(排除 `<= last`),不写 mask,零散射 |
| 单行探针 PASS,多行套件全 FAIL | `last` 游标只在核首初始化,**跨行未复位** | 每行开头 `lastf = -1` |

**候选 vs current best**(`perf_records.jsonl` 的 `phase=candidate`,round 3,sha256 = 交付件):

| workload | count(×40) | select | op_us | vs baseline | Ratio |
|---|---:|---:|---:|---:|---:|
| topk-1024 | 17.521 | **81.663** | 782.503 | **−33.0%** | 0.0047% |
| topk-4096 | 24.901 | **273.671** | 1269.711 | **−35.3%** | 0.0058% |
| topk-8192-tiled | 35.941 | **468.839** | 1906.479 | **−43.2%** | 0.0077% |

→ 三个 workload 同向改善,**无需按 shape 分条件路径**,直接合并进同一份 `@T.prim_func` 族(即交付件)。

**Autotune Log**:本 kernel 无可扫参数(`num_cores`/`block_size` 均不是自由旋钮:单 block 是正确性约束、
tile 由 UB 预算静态决定),故未启用 autotune。

---

## 3. Final Summary

最终候选 = `perf_opt/_topk_selector_kernel.py`(sha256 `c33e6f3b…`,`phase=final` 全部在同一文件上复测)。
该文件在首次 final 采集后补了两个**分层测试入口**(`run_L2` / `run_boundary`,kernel 体一行未改),
因此重采了一次 final;旧的三条 final 记录**原样归档**在
`perf_records.stale-final-20261018T070752Z.jsonl`(不篡改数值,仅让"当前最终版本恰有一条 final 记录")。

### Final Performance Test Data

| kernel_id | workload_id | baseline(µs) | final(µs) | candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/attention_indexing/topk_selector/_topk_selector_kernel/_topk_selector_kernel.py::count+select::main | topk-1024 | 1168.700 | 752.483 | final |
| tileops/kernels/attention_indexing/topk_selector/_topk_selector_kernel/_topk_selector_kernel.py::count+select::main | topk-4096 | 1963.939 | 1251.031 | final |
| tileops/kernels/attention_indexing/topk_selector/_topk_selector_kernel/_topk_selector_kernel.py::count+select::main | topk-8192-tiled | 3355.778 | 1879.579 | final |

提升与 Ratio(`baseline → final`):topk-1024 **−35.6%(1.55×)**、Ratio 0.0032% → **0.0049%**;
topk-4096 **−36.3%(1.57×)**、0.0038% → **0.0059%**;topk-8192-tiled **−44.0%(1.79×)**、0.0044% → **0.0078%**。

分解(final): count 项 `16.761/24.421/35.261 × 40` = `670.44 / 976.84 / 1410.44 µs`,占 final 的
**89.1% / 78.1% / 75.0%**;select 项 `82.043 / 274.191 / 469.139 µs`。count 占比越高,
说明剩余时间越是被"40 趟 host 二分"这一算法性成本锁定(见 §4)。


其中 **select kernel 单独**改善:`494.260→82.043`(**6.03×**)、`981.499→274.191`(**3.58×**)、
`1944.538→469.139`(**4.14×**);count 侧不变(仍是 40 × ~17/24/35 µs)。

**精度**:最终交付件 `--level all` **12/12 PASS**——L0、L1、L1-k1、L1-kN、L1-2048、L1-tiled、
L1-tiled-4096,加上补入的 L2(3,8192,64)与 boundary 四例(单行、k=1、k=N、K==N 短行)
(`logs/final_suite_all.log`);final 阶段另跑一次 L0+L1 亦 `pass=7 fail=0`。
smoke(`smoke-256`)按规则只用精度、不计性能。

**tuning_status**:`winner_merged`(三个 tune workload 均采同一最终候选,无回退、无需条件路径)。

**中止原因**:非“预算耗尽”,而是剩余结构方向被实测封死——核内二分(编译器崩溃)、
行并行(GM 写间歇丢行)、多阈值批量(总趟数不减)、提前收敛(60 轮才收敛)。
受限条件与证据见 §2。

**集成侧提示(留给 conductor)**:Stage 5 的 wrapper(`topk_selector.py`)当前 import 的仍是
`../_topk_selector_kernel.py`(baseline)。按 skill 约定,本阶段**不修改 wrapper**;
若要采纳 tuned 版本,需把 import 指向 `_topk_selector_kernel/perf_opt/_topk_selector_kernel.py`。
两文件接口完全一致(`topk_indices(x, k, rounds=ROUNDS, tile=None)` + 同名 kernel 工厂),可直接切换。

---

## 4. 设计层天花板(`[DESIGN_LIMIT]`)

见 `perf_feedback.md`(本轮产出)。要点:manifest 的 roofline 只计**一趟**读写(`4MN+4Mk`),
而“阈值二分求精确 top-K”按算法必须做 **40 趟比较**;即便达到满带宽,占峰比的结构上限也只有约 `1/40`。
本轮把与算法无关的两项成本压掉(标量 cumsum/散射),剩下的 40 趟是本设计族的硬成本。

---

## 5. Skill Retrospective

| # | 类型 | 发现 | 证据 | 建议 |
|---|---|---|---|---|
| 1 | **D** | 同一 kernel 体内**任意**迭代形式(展开 / `T.serial`)只要出现 **≥2 份**“含 `reduce min/max` + (1,1) v-op 链”的体,`bishengir-compile` 即崩溃;rounds=1 正常 | `probe_inbranch_rounds.py`、`logs/round1_l0.log`、`logs/round1b_l0.log` | 回写 `traps-compiler.md`,并**修正** `2OPS_GOAL.md` §1.2 “无收尾 × 40 份可构建”的旧记录(工具链戳绑定) |
| 2 | **D** | 运行期索引的**标量散射写(`buf[0, idx[0,0]] = v`)在 `T.Parallel` 之外被静默丢弃**,不报错、结果错 | `probe_gather_out.py` → `[12,12,12,12]` | 回写 traps;替代手法是“单调过滤”等无散射算法 |
| 3 | **D** | 多 block 的 GM 写**间歇性**丢行:同一形态一次 PASS、换输入即出现未初始化垃圾值 | `probe_multiblock_rows.py`(PASS)vs `probe_round2_isolate.py`(FAIL) | 回写 traps:**单次 PASS 不能证明多 block 可用**,至少 2 形状 × 2 输入 |
| 4 | **P** | **零散射紧凑化**:集合契约下用“逐槽最小提取 + 单调排除”替代 `cumsum`+散射,实测 select 提速 **3.6–6.1×** | 本轮 final 数据 | 回写 `bottleneck-patterns.md`(模式 + 适用条件:K 远小于 N 时收益最大) |
| 5 | **P** | `T.vcmp` 接受 **(1,1) UB 缓冲作标量操作数**,可省掉“先减后比”的一趟 | 本轮 gather kernel | 与既有“GM 标量先拷进 (1,1) 再作 v-op 标量操作数”并列登记 |
| 6 | **R** | 行级游标(如 `last`)必须在**每行开头复位**;`T.Parallel` 变量不能 `float()`;iota 用 `cumsum(ones)-1` 构造 | 本轮三次失败→修复 | 写入开发流程检查清单(行级状态复位、Var 不可转 Python 数值) |

---

# 追加(2026-10-20):把整轮二分搬进内核 —— 41 次下发压到 2 次


## 7.1. 剖析:钱花在哪

| 段 | 实测耗时 | 说明 |
|---|---:|---|
| H2D `x.npu()` | 0.028 ms | 不是瓶颈 |
| bisect 内核(设备) | **0.188 ms** | 计算量本身极小 |
| select 内核(设备) | **0.156 ms** | 同上 |
| D2H `o.cpu()` | 0.017 ms | 不是瓶颈 |
| **op 整条(调优前)** | **5.2 ~ 8.6 ms** | 差额 = **主机侧 41 次内核下发** |

**关键量化**:一次内核下发约 **0.11 ms**(与 kernel 做多少活无关),
而主机二分要发 `rounds + 1 = 41` 次 ⇒ **40 × 0.11 ≈ 4.4 ms 纯下发开销**,
设备侧实际只有 **0.34 ms**。也就是说调优前 **约 93% 的时间是下发开销**。

> ⚠ 测量口径提醒:`msprof` 只统计**设备内核时间**(6.5–35 µs),它**看不到**这个主机循环,
> 所以用 msprof 的 Ratio 会虚高约 100 倍。本记录一律用 `events`(含主机开销)口径。

## 7.2. 采纳的优化:把整轮二分搬进内核

**做法**:新增 `_topk_bisect_kernel(M, N, K, rounds)`,行只需载入 UB 一次,
之后 40 轮二分在**设备内**完成(循环体只做 UB 内 v-op,不碰 `T.copy`);
wrapper 在"整行能装进 UB"时走此快路径 ⇒ **下发次数 41 → 2**。

### 实现中撞到的两个硬约束(均实测)

| # | 约束 | 现象 | 解法 |
|---|---|---|---|
| 1 | **`T.if_then_else` 作用于运行期标量会段错误** | 内置二分探针 `dumped core` | 分支必须用**向量指令**构造:vcmp/vselect 导出 0/1 标志 |
| 2 | **向量 pipe 写 → 标量 pipe 读 无跨 pipe 排序** | 用标量读回标志再更新 lo/hi 时,**只有第 1 轮生效**(得 220.5 而非 385.875) | lo/hi 的更新也**全程用 vadd/vsub/vmul**,不回落标量 |

隔离探针:`repro/PROBE-topk-inkernel-isolate.py --case {plain|ifelse|vselect}`

| case | 结果 |
|---|---|
| `plain`(无分支) | ✅ 编译并运行 |
| `ifelse`(T.if_then_else) | ❌ **段错误** |
| `vselect`(全 v-op) | ✅ 运行,且 4 轮后得 **385.875**(与理论值一致) |

## 7.3. 调优结果

| workload | 调优前 | 调优后 | 加速 | Ratio(前) | Ratio(后) |
|---|---:|---:|---:|---:|---:|
| 8×256, k=8 | 5.222 ms | **0.373 ms** | **14.0×** | 0.00009% | **0.0013%** |
| 16×1024, k=16 | 5.893 ms | **0.992 ms** | **5.9×** | 0.00063% | **0.0037%** |
| 8×4096, k=64 | 7.113 ms | **1.464 ms** | **4.9×** | 0.00104% | **0.0051%** |
| 8×8192, k=64 | 8.615 ms | **2.486 ms** | **3.5×** | 0.00170% | **0.0059%** |

- 报告:`status=passed`,correctness **9/9**,benchmark **0/4 失败**
- 正确性未回退;内核级 7 项(`L0/L1/L1-k1/L1-kN/L1-2048/L1-tiled/L1-tiled-4096`)全 PASS

## 7.4. Ratio 的算法(与 harness 同口径)

```
manifest roofline:  bytes = 4 * M * N + 4 * M * k
Ratio(%) = Perf / Computility × 100% = bytes / (latency × 1.8 TB/s)
```

验算(8×256,k=8):`bytes = 4·8·256 + 4·8·8 = 8448 B`,
`0.373 ms` ⇒ `8448 / (0.373e-3 × 1.8e12) = 1.26e-5` ⇒ **0.0013%** ✓

## 7.5. 为什么加速比随规模下降(14× → 3.5×)

下发开销是**常数**(41 次 × 0.11 ms),设备计算随 `M×N×rounds` 增长。
规模越大,被摊薄的比例越小 ⇒ 加速比自然从 14× 降到 3.5×。

## 7.6. 未做的

- **tiled 路径仍走主机二分**(`tile` 显式给出时):此时行不在 UB 内,
  循环体就必须含 `T.copy`,而那正是实测会触发 `ERR99999`/"complex loop body" 的形态。
  要覆盖它需要实现 UB 内的分块累积二分,本轮未做。
- **列方向并行**:多 block 的 GM 写会丢行(见 `PROBE-rows-arch`),故仍为单 block。

---

## 8. 口径更正(重要):harness 的 Ratio 与"算子总耗时"不是一回事

### 8.1 harness 在 msprof 下测的是什么

`tileops/benchmark/msprof.py:664-666`:

```python
durations_us.sort()
median_us = durations_us[len(durations_us) // 2]      # 每个内核实例耗时的中位数
latency_ms = median_us / 1000.0
```

**它取的是"每个内核实例耗时的中位数",不是算子总耗时。**
另外 11 个算子**每次调用只发 1 个内核**,所以中位数恰好等于算子设备时间,才具可比性;
本算子优化前发 **41 次**、优化后发 **2 次**,中位数只代表其中一个子内核。

### 8.2 两个口径的实测对照

| workload | 优化前 msprof | 优化前 墙钟 | 优化后 msprof | 优化后 墙钟 |
|---|---|---|---|---|
| smoke-256 | 6.56 µs / 0.0748% | 5222 µs / 0.00009% | 94.12 µs / 0.1002% | 373 µs / 0.0013% |
| topk-1024 | 17.20 µs / 0.2167% | 5893 µs / 0.00063% | 494.0 µs / 0.2276% | 992 µs / 0.0037% |
| topk-4096 | 24.72 µs / 0.2925% | 7113 µs / 0.00104% | 981.0 µs / 0.4646% | 1464 µs / 0.0051% |
| topk-8192 | 35.46 µs / 0.4131% | 8615 µs / 0.00170% | 1944 µs / 0.6091% | 2486 µs / 0.0059% |

### 8.3 报告采用哪个口径

**采用 msprof 口径(0.07%→0.41% / 0.10%→0.61%)**,理由:它与其他 11 个算子的取法完全一致,
都直接读 harness 报告的 `Ratio(%)` 字段,不做任何手工换算。

### 8.4 必须同时声明的偏差

优化前 msprof 只覆盖了真实耗时的 **0.13%**(6.56 µs / 5222 µs),优化后覆盖 **25.2%**
(94.12 µs / 373 µs)。所以:

- **优化前的 msprof Ratio 0.0748% 是被严重虚高的**,不可作为"迁移后性能"的真实基线;
- 真实提速以墙钟为准:**3.5 – 14.0×**;
- 墙钟口径的 Ratio 为 **0.00009%–0.0017% → 0.0013%–0.0059%**。

### 8.5 顺带记录:为什么本算子不能像其余 11 个那样"一个内核搞定"

要素集落选区内核要额外的 `posf/inv/spos/sposi/krow` 等缓冲,融合后约 11 个 N 宽缓冲
× 4 B;N=8192 时需 **352 KB > UB 192 KB**,装不下。而 N ≤ 4096 可以融合,
本轮未做(harness 已认可 2 次下发,`invalid_profile_count = 0`)。
