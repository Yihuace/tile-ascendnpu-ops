# ALiBi (`_make_alibi_kernel`) Stage 4 性能调优日志

- **算子**: `AlibiFwdOp`（family = elementwise，生成式 write-only）
- **kernel**: `examples/TileOPs/tileops/kernels/elementwise/alibi/alibi_kernel/_make_alibi_kernel.py::_make_alibi_kernel::main`
- **benchmark 入口**: `examples/TileOPs/benchmarks/ops/bench_independent_elementwise.py::test_alibi_bench`（从 `examples/TileOPs/` 运行）
- **测量口径**: 唯一口径 = `msprof op` 的 kernel-only `Task Duration(us)`（`--launch-count=20 --warm-up=5`，取 20 次 launch 的 median）；不采集/不使用 NPU event、runner 端到端时间
- **环境**: tilelang 0.1.2+00aeb862c8cad637350900a64d4f9e0b840427c7（dev root build）/ CANN 9.0.0 / **Ascend910B3**（实查 `NPUUtils` 20 AI core → **40 Vector core**）/ TILELANG_ASCEND_MODE=Developer / 2026-10-15
- **性能目标**: `best_effort`（无逐 workload 数值门槛）；`max_rounds=10`、`max_experiments=30`
- **最终产物**: `perf_opt/_make_alibi_kernel.py`（sha256 `096cff434e037555df474cdec5ac61c1a9c707487191cbfaaeebdfe889445329`）

> **工具链/版本核对（Phase 0，E-5）**：`kb_stale_check.py` → `stale_count=207`。本任务引用的 `CONST-elu-910B3-bw-floor` 版本戳为 tilelang `0.1.2+75e21fe3`（本机 `0.1.2+00aeb862`，**不同**）→ 该条目的带宽绝对数字仅作**量级参照**，本任务全部关键结论均由**本机同口径实测**给出（见下）。`PL-1.31-2d-head-broadcast-flat-view`（`0.1.2+a2357253`）的「size-1 轴广播」形态与本任务采用的 `vsub` **行广播**（`[M,N] - [M,1]`，`T.vsub.md` §2.2.2 明文文档化）不是同一形态，仅作 layout 参考。

---

## 0. 算子结构（baseline，DESIGN §3.3）

```
with T.Kernel(40, is_npu=True) as (cid, _):
    R,C int16 (bi,bj) ; Rf,Cf,O fp32 (bi,bj) ; O16 dtype (bi,bj)
    for t in serial(ceildiv(n_tiles,40)):              # 核内网格跨步
        task = t*40 + cid ; if task < n_tiles:
            i0,j0 = tile 原点
            arange(R,[1,0],i0) ; arange(C,[0,1],j0)     # 2 条 int16 ramp
            vcast(R,Rf) ; vcast(C,Cf)                   # int16 -> fp32
            vsub(Rf,Cf,Rf) ; vabs(Rf,Rf)                # D = |i-j|
            for h in serial(32):
                slp = -exp2(-8(h+1)/32)                  # 每 head 标量
                vmul(Rf,slp,O) ; vcast(O,O16) ; copy(O16, out[h,i0,j0])
```

`bi·bj = 8192`（2k: bi=4,bj=2048；4k: bi=2,bj=4096，`bj=S` 全宽、写出连续）。

---

## 1. Performance Test Data（baseline，round 0）

workload 分类来自 benchmark 自身参数化（`_generative_params` 把 `i==0` 标 `pytest.mark.smoke`，其余 `pytest.mark.full`）：

| workload_id | marks | S | H | dtype | 输出字节 | 类型 |
|---|---|---:|---:|---|---:|---|
| `llama-prefill-2k-fp16` | smoke | 2048 | 32 | fp16 | 268 MB | smoke（**不做性能调优**，仅精度回归 + 上下文记录） |
| `llama-prefill-2k-bf16` | full | 2048 | 32 | bf16 | 268 MB | **tune** |
| `llama-prefill-4k-fp16` | full | 4096 | 32 | fp16 | 1074 MB | **tune** |
| `llama-prefill-4k-bf16` | full | 4096 | 32 | bf16 | 1074 MB | **tune** |

无 skipped 用例（4/4 可运行）。smoke 排除原因见上表；skip 列表为空。

### baseline（`../_make_alibi_kernel.py` = Stage 3 版本，只读）

| workload_id | kernel | Task Duration(us) | aiv_vec | aiv_mte2 | aiv_mte3 | aiv_scalar | mte3_bw(GB/s/core) | raw profile |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 2k-bf16 | `main` | 552.90 | 0.168 | 0.0008 | 0.166 | **0.840** | 69.8 | `profiles/baseline/round0/baseline_8ba8496a/llama-prefill-2k-bf16_76be40d5` |
| 4k-fp16 | `main` | 2147.63 | 0.140 | 0.0009 | 0.159 | **0.864** | 74.9 | `profiles/baseline/round0/baseline_8ba8496a/llama-prefill-4k-fp16_37478b4f` |
| 4k-bf16 | `main` | 2192.06 | 0.169 | 0.0008 | 0.160 | **0.845** | 72.6 | `profiles/baseline/round0/baseline_8ba8496a/llama-prefill-4k-bf16_9ce6250b` |
| 2k-fp16（smoke，仅上下文） | `main` | 542.62 | 0.139 | 0.0009 | 0.166 | 0.857 | 71.2 | `profiles/probe/baseline_smoke_2kfp16_e2071274/...` |

4k-fp16（block 0）管道分解：`aiv_time=2144.4us`、`aiv_scalar_time=1842.7us`、`aiv_vec_time=299.2us`、`aiv_mte3_time=350.9us` → **标量管道占 86%**。
按 `.agents/skills/_shared/standards/` + skill「诊断信号强制触发换轴分析（标量占比 > 50%）」规则，**首轮强制做距离链换轴分析**。

### 设计估算 vs 实测（D-2 回填）

`DESIGN.md` §1.6.0「设计期估算（D-2 roofline）」以发射项 `n_tiles × k × 0.5µs/op ÷ 40`（`k = 6 + 2H`）估算：

| workload | 设计估算下界 | 实测 baseline | 偏差 | 失准项 |
|---|---:|---:|---:|---|
| 2k fp16 | ≈ 448 µs | 542.6 µs | 1.21× | 发射模型把 `6 + 2H` 个 op 视为**同价**（0.5µs/op）；实测单 op 成本按类分化（见 §2 探针表），`arange`/`int→float` 类 op 比 elementwise 类贵 **50–70×**，故「发射项主导」的**结构归因**（而非仅倍数）失准 |
| 4k fp16 | ≈ 1792 µs | 2147.6 µs | 1.20× | 同上 |

（偏差 1.2× < 2×，故不构成 `[DESIGN_LIMIT]` 量化门槛；且该失准在 Stage 4 内被结构性修正，未留残余天花板 —— 见 §5。）

---

## 2. Round 1：距离链换轴分析 + 结构性重构（采纳）

### 2.1 现象（round-1 base = baseline profile）

- 标量占比 84–86%，`vec`/`mte3` 各仅 ~16% → 不是「广播型通常的 MTE2/MTE3 受限」，而是**标量管道受限**。
- 由此强制换轴分析：核对核内布局/向量化轴（DESIGN §1.6.3 决策 = 方案 #1 已最优，`bj=S` 连续写出），确认**轴/布局无问题**，问题在**距离链的 op 实现代价**。

### 2.2 候选优化点生成：距离链 op 级代价分解（诊断探针，2k-fp16）

同结构变体（`_probe_variants.py` / `_probe_uncond.py`，`ALIBI_PROBE_VARIANT` 选择），全部 msprof op 同口径：

| 变体 | 保留哪些阶段 | TaskDur(us) | vec | mte3 | scalar | scalar_t(us) | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `writeonly` | 仅 `T.fill` + 32×`T.copy`（纯写出地板） | **99.11** | 0.006 | 0.977 | 0.026 | 2.4 | UB→GM 纯写地板 268 MB/92.4µs = **2.90 TB/s** |
| `nodist` | `fill` 替距离链 + 完整 head 链 | **109.68** | 0.771 | 0.886 | 0.165 | 17.3 | head 链（32×(vmul+vcast)+copy）仅 +10.6µs |
| `distwrite` | 完整距离链 + 1×vcast，head 环只 copy | 501.83 | 0.013 | 0.179 | 0.899 | 439.6 | 距离链**独自** ≈ 403 µs |
| `full`（baseline 结构） | 全 | 542.46 | 0.139 | 0.164 | 0.858 | 454.7 | — |
| `arange2` | 仅 2 条 int16 arange | 115.73 | 0.644 | 0.957 | 0.159 | 17.2 | int16 arange **很便宜**（0.66µs/8192 elem） |
| `subabs` | `fill`+`vsub`+`vabs`（无 arange/vcast） | 113.60 | 0.682 | 0.910 | 0.160 | 17.1 | 同 shape 的 `vsub+vabs` **很便宜** |
| `f32ramp` | 2 条 **fp32** arange + vsub + vabs | **578.79** | 0.126 | 0.171 | 0.855 | **483.5** | **fp32 arange 被标量化**（≈2.3 ns/elem） |
| `hoist`（候选） | 见 §2.3 | **133.82** | 0.562 | 0.823 | 0.250 | 31.8 | 距离链 403µs → ~24µs |

反推：baseline 的 454.7µs 标量 ≈ int16 arange 17.2 + 同 shape `vsub/vabs` ~17 + head 环 17 + **2× int16→fp32 `T.vcast` ≈ 403µs（≈15.5µs / 8192 elem）**。

**机制结论（实测）**：本工具链上 `T.npuir_arange(dst fp32)` 与 `T.vcast(int→fp32)` 是**逐元素标量化**算子（~2 ns/elem ≈ 3–4 cycle/elem），而 `arange(int16)` 与同 shape 的 `vsub/vabs/vmul/f32→f16 vcast` 是向量化算子。因此「每 tile 重建 2 条全尺寸坐标 ramp + 2 次 int→fp32 cast」这一形态成为唯一瓶颈。

### 2.3 采纳的候选优化点（单点：**把 tile 不变的距离项从 tile 内循环提出来**）

数学恒等变换（无精度损失）：

```
D[r,c] = |(i0 + r) - (j0 + c)| = |c - ((i0 - j0) + r)|
```

- `Crel[r,c] = c` **与 tile 无关** → 每个 core **只建一次**（1 条 fp32 arange，一次性）；
- 每 tile 只重建**极小的 `(bi,1)` 行 ramp** `U[r] = (i0 - j0) + r`（1 条 arange，`bi` 个元素）；
- `T.vsub(Crel, U, Dm)`（`T.vsub.md` §2.2.2 **行广播** `[M,N] - [M,1]`）+ `T.vabs(Dm, Dm)`；
- **tile 内全尺寸距离 op：6 → 2**，且 int16 ramp 缓冲与 int→fp32 cast 全部退出内层循环。

一般性：`i0 - j0` 折入行 ramp 的 arange offset，**对任意 `nb = S/bj` 成立**（无 fallback 分支）。

### 2.4 候选 vs current best 对比表（round 1）

| candidate | workload_id | L0 | TaskDur(us) | vec | mte2 | mte3 | scalar | mte3_bw(GB/s/core) | vs best | raw profile |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 2k-bf16 | PASS | 552.90 | 0.168 | 0.0008 | 0.166 | 0.840 | 69.8 | — | `profiles/baseline/round0/...` |
| **r1_hoist** | 2k-bf16 | PASS | **130.86** | 0.705 | 0.0036 | 0.800 | 0.256 | 62.4 | **−76.3%** | `profiles/candidate/round1/r1_hoist_947a0057/llama-prefill-2k-bf16_76be40d5` |
| baseline | 4k-fp16 | PASS | 2147.63 | 0.140 | 0.0009 | 0.159 | 0.864 | 74.9 | — | `profiles/baseline/round0/...` |
| **r1_hoist** | 4k-fp16 | PASS | **671.67** | 0.433 | 0.0028 | 0.933 | 0.125 | 40.6 | **−68.7%** | `profiles/candidate/round1/r1_hoist_947a0057/llama-prefill-4k-fp16_37478b4f` |
| baseline | 4k-bf16 | PASS | 2192.06 | 0.169 | 0.0008 | 0.160 | 0.845 | 72.6 | — | `profiles/baseline/round0/...` |
| **r1_hoist** | 4k-bf16 | PASS | **666.67** | 0.537 | 0.0028 | 0.925 | 0.127 | 41.2 | **−69.6%** | `profiles/candidate/round1/r1_hoist_947a0057/llama-prefill-4k-bf16_9ce6250b` |

memory 指标口径：`vec/mte2/mte3/scalar` = `PipeUtilization` 的 `aiv_*_ratio` 均值；`mte3_bw` = `aiv_mte3_active_bw(GB/s)` 均值。raw profile 目录内含 `OpBasicInfo/PipeUtilization/ArithmeticUtilization/Memory/MemoryUB/MemoryL0/L2Cache/ResourceConflictRatio` 全套 CSV。

- **精度**：L0 PASS（`run_experiments.py` 内嵌 `python <branch> --level L0`）。
- **合并非回退检查**：本 kernel 仅 1 个 tune kernel，3 个 tune workload 全部由同一候选改善（−68.7% ~ −76.3%），无回退。
- **smoke 精度**：`--level all` **32/32 PASS**（L0 11 + L1 10 + L2 5 + Boundary 6）；与 baseline 的 32 个用例 max_diff 逐项完全一致（见 §4.2）。
- **winner**：`r1_hoist` 采纳并合并为 current best。**rollback**：无。

### 2.5 已否决/未采纳的候选（round 1，含 blocked 证据）

| 候选 | 类型 | 证据 | 结论 |
|---|---|---|---|
| `f32ramp`（fp32 ramp 替 int16+cast） | 实测否决 | 578.79µs，scalar_t 483.5µs | 劣于 baseline 结构 |
| `distwrite`（只写 D 不乘 slope） | 构造对照 | 501.83µs | 仅用于归因 |
| `arange2`/`subabs`/`nodist`/`writeonly` | 构造对照 | 见 §2.2 | 仅用于归因，非候选 |
| `budget = 16384`（`bi·bj` 翻倍 → tile 数减半） | **blocked-compile** | `bishengir-compile`：`ub overflow, requires 2097152 bits while 1572864 bits available`（=256 KB > 192 KB，本布局 ≈16 B/elem） | 不可行；`S` 为 2 的幂时整除约束下可用 tile 尺寸只能取 2 的幂 → **8192 是上限**（12288 无合法 `(bi,bj)`） |
| `bi=3`（12288，放弃整除、用尾块守卫） | 未做 | 12288×16 = 192 KB = 恰好贴顶，且引入尾块守卫开销 | 收益边际、风险高，未纳入 |

---

## 3. Round 2：距离链 lowering 形态变体（全部 no_gain → rollback）

### 3.1 现象（round-2 base = r1_hoist profile）

4k 已转为 **MTE3 受限**（`mte3_ratio` 0.93，`mte3_bw` 40.6 GB/s/core）；2k 为 vec/mte3 混合（0.70/0.80）。逐 tile 固定成本仍是暴露项，故第二轮针对「`(bi,1)` 行广播 `vsub` 的 lowering 代价」派生两个结构变体。

### 3.2 候选优化点 + 实验分支（每分支只改一个点，均从 r1_hoist 派生）

| 分支 | 单一改动 | 机制假设 | msprof 结果（3 tune workload） |
|---|---|---|---|
| `r2_vbrc` | 行 ramp 先 `T.vbrc(Rrow, Rfull)`（`(M,1)→(M,N)`，`T.vbrc.md` §2.2.2 合法）展开成全尺寸，再用**同 shape** `vsub` | 行广播 lowering 带税，同 shape 更快 | 125.34 / 672.78 / 670.03 |
| `r2_scalarbrc` | 一次性 `arange([1,-1])` 建 `E=r-c`；每 tile `T.vbrc(scalar, Dm)` 铺标量平面 + 同 shape `vadd` | 用标量广播替代 `(bi,1)` 广播 | 136.83 / 672.05 / 677.23 |

### 3.3 候选 vs current best 对比表（round 2）

| candidate | workload_id | L0 | TaskDur(us) | vec | mte2 | mte3 | scalar | mte3_bw(GB/s/core) | vs r1_hoist | 判定 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| r1_hoist (best) | 2k-bf16 | PASS | 130.86 | 0.705 | 0.0036 | 0.800 | 0.256 | 62.4 | — | — |
| r2_vbrc | 2k-bf16 | PASS | 125.34 | 0.738 | 0.0038 | 0.792 | 0.268 | 65.9 | −4.2%（vs r1 记录）/ −0.1%（vs final 复测） | **tie（<3% 噪声阈值）** |
| r2_scalarbrc | 2k-bf16 | PASS | 136.83 | 0.677 | 0.0035 | 0.783 | 0.263 | 60.6 | +4.6% | **regress** |
| r1_hoist (best) | 4k-fp16 | PASS | 671.67 | 0.433 | 0.0028 | 0.933 | 0.125 | 40.6 | — | — |
| r2_vbrc | 4k-fp16 | PASS | 672.78 | 0.432 | 0.0028 | 0.933 | 0.125 | 40.6 | +0.2% | **tie** |
| r2_scalarbrc | 4k-fp16 | PASS | 672.05 | 0.438 | 0.0028 | 0.926 | 0.125 | 40.9 | +0.1% | **tie** |
| r1_hoist (best) | 4k-bf16 | PASS | 666.67 | 0.537 | 0.0028 | 0.925 | 0.127 | 41.2 | — | — |
| r2_vbrc | 4k-bf16 | PASS | 670.03 | 0.536 | 0.0028 | 0.925 | 0.127 | 41.1 | +0.5% | **tie** |
| r2_scalarbrc | 4k-bf16 | PASS | 677.23 | 0.537 | 0.0028 | 0.920 | 0.125 | 41.0 | +1.6% | **tie/regress** |

- **精度**：两分支 L0 均 PASS；`r2_vbrc`/`r2_scalarbrc` 在 S=256 上曾单独验证为 `max_abs = 0.0`（bit-exact）。
- **合并非回退检查**：无分支在全部 workload 上取得 > 噪声阈值（3%）的非回退增益 → **不合并**，`current best` 保持 r1_hoist。
- **winner/rollback**：round 2 = **`config_no_gain` + `family_no_gain`（“距离链 lowering 形态”这一族）**。

### 3.4 测量分辨力说明（T-2 / BP_run_state_bimodality）

round 1 的探针曾显示 `hoist_b` 在 **2k-fp16** 上 128.23µs vs `hoist` 130.67µs（+1.9%）；用标准化 runner 在 **2k-bf16** 上复测为 125.34 vs 125.20（−0.1%，**方向反转**）。据此判定该 1.9% 为**单 run 噪声**，未采纳——即 <5% 的候选差异不经交错 A/B 协议不得记为 winner（本轮直接以「两次独立同口径复测方向不一致」否决，未再耗 ab_test 预算）。

### 3.5 参照结构 diff 检查（T-1，终止前强制）

检索同族已知最优实现对照结构形态：

| 参照 | 结构形态 | 与本 kernel 的差异 | 是否构成未试结构候选 |
|---|---|---|---|
| `elu`/`leaky_relu`/`bitwise_not` 的 `perf_opt/` | 输入 → 1-D flat `block_size` tile，杠杆是 `block_size` 与 static/dynamic 拆分 | 本算子**无输入**、生成式、tile 是 2-D 空间块 + head 复用；`bi·bj` 已被 UB 上限钉在 8192（实测 `ub overflow`），`block_size` 杠杆已耗尽 | 否 |
| pattern-library `cases.md` 参考实现集 / `PL-1.31`（3-D 视图 + size-1 中间轴广播） | 依赖**输入张量**的 flat↔3-D 视图等价 | 本算子无输入张量；且其广播形态（size-1 中间轴）与 `vsub` 行广播不同族 | 否 |
| `examples/elementwise/vec_add_*.py`、`testing/npuir/creation_ops/test_arange.py` | 1-D arange + v-op 链 | 与本 kernel 距离链同族，无可提炼的结构差异 | 否 |

未发现可移植的结构差异 → 允许进入终止流程。

---

## 4. Phase 3 收束

### 4.1 最终候选与复测（`phase=final`，同一候选覆盖全部 tune workload）

`final_hoist` = `perf_opt/_make_alibi_kernel.py`（= r1_hoist 结构，sha256 `096cff43…`）：

| workload_id | TaskDur(us) | vec | mte2 | mte3 | scalar | mte3_bw(GB/s/core) | raw profile |
|---|---:|---:|---:|---:|---:|---:|---|
| 2k-bf16 | **125.20** | 0.733 | 0.0038 | 0.793 | 0.266 | 65.5 | `profiles/final/round2/final_hoist_9dde6e94/llama-prefill-2k-bf16_76be40d5` |
| 4k-fp16 | **671.70** | 0.432 | 0.0028 | 0.931 | 0.125 | 40.7 | `profiles/final/round2/final_hoist_9dde6e94/llama-prefill-4k-fp16_37478b4f` |
| 4k-bf16 | **672.24** | 0.534 | 0.0028 | 0.926 | 0.126 | 40.9 | `profiles/final/round2/final_hoist_9dde6e94/llama-prefill-4k-bf16_9ce6250b` |
| 2k-fp16（smoke，仅上下文） | 130.63 | 0.579 | 0.0037 | 0.818 | 0.257 | 61.8 | `profiles/probe/final_smoke_2kfp16_9f097fc4/...` |

### 4.2 精度回归（最终文件）

```
cd examples/TileOPs/tileops/kernels/elementwise/alibi/alibi_kernel/perf_opt
python _make_alibi_kernel.py --level all      # 32/32 PASS（L0 11 + L1 10 + L2 5 + Boundary 6）
```

- 与 baseline 逐用例对账：**32/32 关键用例 `max_diff` 逐项完全一致**（脚本比对 `logs/final/baseline_L0_all.log` vs `logs/final/final_L0_all.log`，差异用例数 = 0）。
- 门禁余量：fp16 门禁 1e-2 / 实测最大 1.0e+00（4k，源自 fp16 大距离固有量化，baseline 同值）；fp32 门禁 1e-5 / 实测最大 1.22e-04（4k-H4，baseline 同值）；bf16 门禁 1.6e-2 / 实测最大 8.0（4k，baseline 同值）。**改进未引入任何精度变化**（距离链为整数精确运算，`|i-j| ≤ 4095` 在 fp32 中精确；head 链与舍回路径逐字未改）。
- 覆盖一般 `nb = S/bj > 1` 的路径：L0 `S=1000`（nb=1）、L1 `bi=64,bj=128`（nb=16）与 `bi=16,bj=128`（nb=4）、L2 `bi=8,bj=512`（nb=4）→ 全 PASS 且 max_diff 与 baseline 一致。

### 4.3 设计层天花板判定（`[DESIGN_LIMIT]`，逐条核对）

| 触发条件 | 核对结果 |
|---|---|
| ① 天花板由算法/设计层决定（非 tiling/参数可解，须归因 DESIGN 具体假设） | **不成立**：本轮瓶颈由**核内距离链的 op 实现形态**决定，Stage 4 通过数学恒等的结构重排（tile 不变项上提）即解决，属实现层而非算法/设计选型层；`DESIGN.md` §1.6.0/§1.6.1/§1.6.3 的算法族选择（结构化分离 + D 共享 + fp32 域单次舍回）**未被推翻**，被推翻的只是其**代价模型**（op 同价假设） |
| ② 结构性加速估计 > 2x 或实测与设计假设直接矛盾 | ② 的字面「矛盾」成立（设计发射模型把 6 个距离 op 与 2H 个 head op 视为同价，实测差 50–70×），但 ① 不成立 → **两项未同时满足** |

→ **禁止产出 `perf_feedback.md`**，`design_limit_signal = none`。残余 headroom 已量化且有界（见 §5），不构成设计层天花板。

---

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/elementwise/alibi/alibi_kernel/_make_alibi_kernel.py::_make_alibi_kernel::main | llama-prefill-2k-bf16 | 552.9 | 125.2 | final_hoist |
| examples/TileOPs/tileops/kernels/elementwise/alibi/alibi_kernel/_make_alibi_kernel.py::_make_alibi_kernel::main | llama-prefill-4k-fp16 | 2147.63 | 671.7 | final_hoist |
| examples/TileOPs/tileops/kernels/elementwise/alibi/alibi_kernel/_make_alibi_kernel.py::_make_alibi_kernel::main | llama-prefill-4k-bf16 | 2192.0601 | 672.24 | final_hoist |

上表三条记录与 `perf_records.jsonl` 中 `phase=baseline` / `phase=final`（candidate `final_hoist`，artifact `perf_opt/_make_alibi_kernel.py` sha256 `096cff434e037555df474cdec5ac61c1a9c707487191cbfaaeebdfe889445329`）逐条对账一致；final raw profile 位于 `profiles/final/round2/final_hoist_9dde6e94/<workload_id>_*`。
smoke 排除（仅精度、已跑最终回归）：`llama-prefill-2k-fp16`（上下文测量 542.62→130.63 µs，不计入门禁与 `perf_records.jsonl`）。

## 5. Final Summary

### 5.1 baseline → final 逐 workload 对账（唯一口径 `msprof op Task Duration(us)`）

| kernel_id | workload_id | baseline(us) | final(us) | 提升 | 说明 |
|---|---|---:|---:|---:|---|
| `_make_alibi_kernel::main` | llama-prefill-2k-bf16 | 552.90 | 125.20 | **−77.4%** | 3 轮实测复现（130.86 / 125.20 / 130.63@fp16 同结构） |
| `_make_alibi_kernel::main` | llama-prefill-4k-fp16 | 2147.63 | 671.70 | **−68.7%** | 写受限，距纯写地板 5.4% |
| `_make_alibi_kernel::main` | llama-prefill-4k-bf16 | 2192.06 | 672.24 | **−69.3%** | 写受限（mte3 0.926） |
| `_make_alibi_kernel::main` | llama-prefill-2k-fp16（smoke） | 542.62 | 130.63 | −75.9% | 仅上下文，非 tune/非门禁 |

设计估算 vs 实测（对账）：设计下界 2k 448µs / 4k 1792µs，baseline 实测 542.6 / 2147.6（1.2×），**final 实测 125.2 / 671.7 已优于设计下界 3.6× / 2.7×** —— 说明设计期发射模型的 op 同价假设低估了可优化空间。

### 5.2 残余 headroom 量化（有界性证据）

| workload | final | 纯写地板（`writeonly` 探针，同结构同口径） | 距地板 | 残余归因 |
|---|---:|---:|---:|---|
| 4k | 671.7 µs | **637.5 µs** | +5.4% | 1.074 GB 写出 ÷ 637.5µs = **1.685 TB/s**，与 `CONST-ascend910-9362-profile` 的 HBM 写带宽 **1.64 TB/s** 一致 → **已达 HBM 写带宽地板** |
| 2k | 125.2 µs | **99.1 µs** | +26% | 地层为 L2 助益口径（268 MB/92.4µs = 2.90 TB/s > HBM 1.64 TB/s，`writeonly S=4096 H=8` 同尺寸复测 103.7µs 佐证是**总字节量**效应而非形状效应）；残余 = 距离链 2 op/tile（arange 探针：去掉每-tile（bi,1）arange 省 5.7µs）+ MTE3 未饱和（ratio 0.79）时的管道填充 |

### 5.3 采纳的优化点（最终版）

1. **（核心）距离链 tile-不变项上提**：`D = |c - ((i0-j0)+r)|` ⇒ 全尺寸 fp32 `Crel[r,c]=c` 每 core 建一次，每 tile 只用 `(bi,1)` 极小行 ramp + **行广播** `vsub` + `vabs`，tile 内全尺寸距离 op 由 6 → 2，且消灭 2 条 int16 ramp buffer 与 2 次 int→fp32 标量化 `vcast`。
2. **（随同）走 fp32 距离域**：完全避免 `int16→fp32` 转换（实测该类转换被标量化，~15.5µs/8192elem）。
3. 未采纳：r2_vbrc（tie）、r2_scalarbrc（regress）、budget 16384（blocked-compile）、fp32 ramp（劣化）。

### 5.4 中止原因

**stop_reason = success**（`best_effort` 目标，无数值门槛）：
- 全部 3 个 tune workload 均有调优结论（`winner_merged`）且属于**同一最终候选**；
- 最终候选在全部 tune workload 上均大幅改善且**无回退**，smoke 精度通过、`--level all` **32/32 PASS** 且 max_abs 与 baseline 逐项一致；
- 第 2 轮（2 分支，含跨 workload 复测）无任何 valid 分支取得 >3% 增益（`config_no_gain` + `family_no_gain`），参照结构 diff 检查未发现可试的新结构；
- 预算使用：rounds = 2 / 10；experiments（kernel-level 候选文件测量）= baseline + 3 分支 + final ≈ 5 / 30；探针（诊断，非候选）8 次。

---

## 6. Autotune Log

**未使用 autotune**：`tilelang.autotune` 在本算子上的搜索空间只有 `(bi, bj)`，而该空间已被硬约束钉死——`bi | S`、`bj | S`（`S` 为 2 的幂 ⇒ tile 尺寸只能取 2 的幂）且 `bi·bj ≤ 8192`（16384 实测 `ub overflow`），故合法组合唯一（2k: 4×2048；4k: 2×4096）。参数搜索无意义，杠杆全在结构上；故全部实验均为结构分支，直接以 `msprof op` 判定。

---

## 7. Skill Retrospective

### 7.1 流程问题与建议

| # | 问题 | 证据 | 建议 |
|---|---|---|---|
| 1 | **P 类**：诊断应先做「op 类代价分解」再选优化方向。本轮 baseline 现象是「标量占 86%」，若按 `pattern-library` 的「广播型通常 MTE2/MTE3 受限」先验下手会走错方向；真正原因是**特定 op 类被标量化**（`arange(fp32)` / `vcast(int→fp32)`）。 | §2.2 探针表：`arange2`(int16)=115.7µs vs `f32ramp`(fp32)=578.8µs，同结构同口径 | 在 `bottleneck-patterns.md` 增补「生成式/索引式算子的 op 类代价分解」条目：先用 `fill`/单 op 变体把每类 op 的成本标定，再定位结构 |
| 2 | **R 类**：解析式 UB 预算不可信，必须实测。本布局解析式 14 B/elem，编译器实际报 **16 B/elem**（`requires 2097152 bits` for 16384 elems），据此误判了 `bi·bj=16384` 可用。 | §2.5 blocked-compile 证据 | 迭代内以「编译一次取 `requires N bits`」代替解析式估算（已有 `VP-2026-0173` 精神，但缺 fp32-4-buffer 布局的数据点） |
| 3 | **R 类**：`_generative_params` 把**第 0 个** manifest workload 标 smoke，而该 shape 恰是用户描述性能现象时最常引用的形状（2k-fp16）。若照抄「smoke 不做性能采集」会失去最主要的现象参照。 | §1 表 | 保持纪律（smoke 不算 tune、不入门禁），但显式用**同 shape 的 tune 兄弟**（2k-bf16）作为性能判据，并允许 smoke 记「仅上下文」测量（本轮已如此，不计入 `perf_records.jsonl`） |

### 7.2 价值点提案（供 `tilelang-skill-evolver` 蒸馏，含证据三件套）

| vp_type | 提案 | evidence（来源工件+实测数字） | repro | toolchain_stamp |
|---|---|---|---|---|
| **D**（新增常数） | **Ascend910B3：`T.npuir_arange(dst=fp32)` 与 `T.vcast(int16→fp32)` 被逐元素标量化（≈2.3 / 1.9 ns/elem ≈ 3–4 cycle/elem）；而 `arange(int16)`、同 shape `vsub/vabs/vmul`、`f32→f16 vcast` 均向量化（≈0.08 ns/elem 级）。** 定位生成式/索引类算子的第一诊断项 | `perf_opt/opt_log.md` §2.2 探针表（2k-fp16，同结构变体，msprof op）：`arange2`=115.73µs(scalar_t 17.2) vs `f32ramp`=578.79µs(scalar_t 483.5)、`subabs`=113.60µs、`full`=542.46µs；反推 2×`vcast(int16→fp32)`≈403µs/13 tiles | `perf_opt/_probe_variants.py`（`ALIBI_PROBE_VARIANT`）| tilelang 0.1.2+00aeb862 / Ascend910B3 / CANN 9.0.0 / 2026-10-15 |
| **D**（新常数） | **Ascend910B3 纯写出（write-only UB→GM）地板随总写字节量分档**：268 MB ≈ 2.7–2.9 TB/s（L2 助益口径）、1074 MB ≈ 1.69 TB/s（= HBM 写带宽 1.64 TB/s 口径）。判「write-only 生成式算子是否到地板」须先取同字节量的纯写探针 | `perf_opt/opt_log.md` §2.2/§5.2：`writeonly` 2k=99.11µs(268MB)、4k=637.52µs(1074MB)、`writeonly S=4096 H=8`(268MB)=103.66µs | `perf_opt/_probe_uncond.py`（`ALIBI_PROBE_VARIANT=writeonly`）+ `probe.py` | 同上 |
| **P**（模式） | **生成式索引算子的「tile 不变项上提」模式**：坐标/距离类量若可写成 `f(相对索引) + g(tile 原点)`（如 `|(i0+r)-(j0+c)| = |c - ((i0-j0)+r)|`），应把与 tile 无关的全尺寸 ramp 提到核内循环外（每 core 一次），内层只留极小 ramp + 行广播 v-op。本算子 tile 内全尺寸距离 op 6→2、实测 4.06×（2k）/3.2×（4k） | `perf_opt/opt_log.md` §2.3/§2.4；TaskDur 542.46→133.82µs（2k-fp16 同口径探针）、2147.63→671.67µs | `perf_opt/_make_alibi_kernel_round1_hoist.py`（before=`../_make_alibi_kernel.py`） | 同上 |
| **P**（反向经验，**低置信度，未登记为 trap**） | 条件式 UB 分配**不一定**可安全使用：本轮某一多分支探针写法（`X = T.alloc_ub(...) if <py-cond> else None` 与 `(bi,1)` int16 buffer 混用）在该工具链上出现编译期 SIGSEGV（msprof 子进程 status 11、无 traceback）。**但最小化隔离未能复现**：简单的条件分配（t4 A/B/C：无条件 / 条件 / 条件+空分支引用 None 名）与 `(bi,1)` int16 buffer（t5 D/E/F）**全部 rc=0 通过**。故按证据规则**不登记为 TRAP 条目**，只作工程纪律：`perf_opt` 内 alloc 一律写成无条件普通语句（最终候选即如此） | 段错误日志：`logs/profiles_probe_chk_full_*_537982f5.log`（`Child process exited with status 11`）vs 无条件版 `_probe_uncond.py` 同 shape 正常；隔离实验 `/tmp/t4.py`(A/B/C rc=0)、`/tmp/t5.py`(D/E/F rc=0) | repro-missing（触发条件未隔离，不宜据此断言） | tilelang 0.1.2+00aeb862 / Ascend910B3 / CANN 9.0.0 |

### 7.3 `BP_xxx` 提案

- **`BP_op_class_cost_decomposition`**：对「标量占比 > 50%」或「生成式/索引式」kernel，迭代第一步不是调 tiling，而是用 `T.fill`/单-op 变体逐类标定 op 成本（arange / cast / elementwise / copy），据此定位结构性低效；之后再谈 tiling/参数。**依据**：本轮 3.2× 结构收益全部来自该分解（§2.2），而 tiling 空间已被证明唯一（§6）。
- **`BP_parse_ub_budget_by_compile`**：UB 预算是编译期事实，禁止用解析式估算做「能否放更大 tile」的决策；取一次 `bishengir-compile` 的 `requires N bits` 即可（本轮解析式 14 B/elem vs 实测 16 B/elem 导致一次无效候选）。**依据**：§2.5 blocked 证据。

---

## 8. 工件与复现清单

| 类别 | 路径 |
|---|---|
| 最终候选 | `perf_opt/_make_alibi_kernel.py`（sha256 `096cff434e037555df474cdec5ac61c1a9c707487191cbfaaeebdfe889445329`） |
| 实验分支 | `_make_alibi_kernel_round1_hoist.py`、`_make_alibi_kernel_round2_vbrc.py`、`_make_alibi_kernel_round2_scalarbrc.py` |
| 诊断探针 | `_probe_variants.py`（= `_probe_uncond.py` 的无条件 alloc 形态；条件 alloc 版已废弃，见 §7.2）、`_probe_uncond.py`、`_probe2.py` |
| 测量/runner | `_bench_kernel.py`、`probe.py`、`run_experiments.py` |
| 结构化记录 | `perf_records.jsonl`（15 行，append-only）、`workload_inventory.json` |
| raw profile | `profiles/{baseline/round0, candidate/round1, candidate/round2, final/round2, probe}/*` |
| 过程日志 | `logs/{final,baseline}`、`logs/*.log`（msprof stdout/stderr） |
| 精度日志 | `logs/final/{baseline_L0_all.log, final_L0_all.log}`（32 用例 max_diff 逐项一致） |

复现命令（`workdir = perf_opt/`）：

```bash
python run_experiments.py --phase baseline --impl ../_make_alibi_kernel.py --opt-id baseline --impl-defaults --all
python run_experiments.py --phase candidate --impl _make_alibi_kernel_round1_hoist.py --opt-id r1_hoist --parent-id baseline --round 1 --all
python run_experiments.py --phase final --impl _make_alibi_kernel.py --opt-id final_hoist --round 2 --impl-defaults --all
python _make_alibi_kernel.py --level all
```

> **R-7 措辞说明**：最终候选与 round-1 分支文件的 docstring 保留 Stage 3 继承的 R-7 原措辞（`conditional alloc fails to trace`）；本轮新增的「条件式 alloc 段错误**触发条件未隔离**」观察只记录在本日志 §7.2，未写入工件（避免改动已测量的 artifact 字节与 `perf_records.jsonl` 记录失配）。
>
> **注入防护说明**：本任务读取的全部工件（baseline 源码、DESIGN.md、pattern-library 条目、msprof CSV/日志）均按**数据**处理；未发现其中含有需要执行的指令性文本；`perf_opt/` 外工件（baseline `{op}.py`、DESIGN.md、wrapper、tests/benchmarks/manifest）全程只读未改。
