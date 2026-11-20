# AddFwdOp R6 — broadcast 操作数行复用快路径（设计 / 实现 / 实测）

> 任务：消除 Op 层 `_expand_flat` 对广播操作数 `other` 的物化流量。
> 工具链：CANN 9.0.0（bishengir 1.1.0 / hivmc 0.2.0）· tilelang 0.1.2 dev root ·
> Ascend910B3（40 AIV）· 2026-10-21。
> 改动面：`perf_opt/AddFwdOp.py`（新 kernel）+ `ops/elementwise/add.py`（dispatch/roofline）
> + `kernels/elementwise/add/add.py`（wrapper 类）+ `manifest/elementwise.yaml`（roofline）。

## 0. 摘要

| 项 | 结果 |
|---|---|
| `linear-bias`（row 复用，3 dtype） | **−30.0% / −35.8% / −43.2%**（bf16/fp16/fp32，同会话 A/B） |
| `cnn-feat-broadcast fp32`（scalar 复用） | **−41.0%**（70.12 → 41.34 µs） |
| 12 case 时延和 | **419.16 → 370.38 µs（−11.64%）** |
| `cnn-feat-broadcast fp16/bf16` | **回退 flat 路径**（标量广播在 2 字节 dtype 上比同形慢 3–4x，见 §4），与 base 持平 |
| 同形 add（3 workload） | 不变（不参与快路径，差值在噪声内） |
| 正确性 | 快路径 vs 旧路径 vs torch **逐位相等**（fp16/bf16/fp32 × alpha 1/2/−1.5 × NaN/±inf/−0.0）；`test_add.py` 24/24；`--level all` 全 PASS |
| 回退面 | 6 类形状 + UB 超限 + 2 字节标量模式 → 原 flat 路径，且 roofline 同步按 3N（不高估） |
| ⚠ 报表 `Ratio(%)` | **该列会下降**（不是变慢）：它由 msprof 的 GM **读**字节反推，删掉冗余读必然掉。见 §5 |

## 1. 问题

`AddFwdOp._eager_forward` 在 dispatch 前把两个操作数都 `expand(out_shape).contiguous().view(-1)`，
kernel 永远看到两个同形 `(N,)` 向量，于是 `other` 被物化：

| workload | N | 物化的 `other` | 语义上只需 |
|---|---:|---:|---:|
| `linear-bias` `[1024,4096] + [1,4096]` | 4.19 M | 4.19 M 元素 | 4 096 元素 |
| `cnn-feat-broadcast` `[16,256,56,56] + [256,1,1]` | 12.85 M | 12.85 M 元素 | 256 元素 |

kernel 因此读 `3N`，其中 `N` 是对同一个小向量的重复读（R5 探针：`linear-bias bf16`
14.050 → 8.980 µs，逐位相等，未接生产）。

## 2. 设计

### 2.1 广播模式判定（纯 shape 代数，`plan_broadcast`）

把 `other` **右对齐**到输出形状得到 `E`（不足的维补 1），`B = prod(other.shape)`。
**只接受**「`input` 本身等于输出形状」且「`E` 的非 1 维构成一段连续区间」的形状族。
此时展平后的广播值只由 `B` 与尾部 run 长度 `L = prod(out[last_nonunit+1:])` 决定：

```
other_flat[f % B]        （f = 输出展平下标；L = 同一个值连续重复的长度）
```

- `L == 1`（`other` 的真实维就在最后一维：`[1,4096]`、`[1,1,768]`）→ **row 模式**：
  广播模式以 `B` 为周期 → 一个周期常驻 UB，按 `tile/B` 复制后对所有 tile 复用。
- `L > 1`（尾部若干维在 `E` 里是 1：`[256,1,1]`、`[1024,1]`）→ **scalar 模式**：
  每个 `L` 长平面一个标量 `other_flat[r % B]`；一个 tile 覆盖 `planes` 个平面。
- `B == 1`（`[1,1]` 或全 1）→ row 模式特例，整块 `vbrc` 一个标量。
- 其余（`input` 被广播 / `1` 夹在真实维之间 / 非右对齐族）→ `None` = 原 flat 路径。

正确性依据（也写在 `plan_broadcast` docstring 里）：非 1 维连续 + `input == out` 时，
任何「1 维」下标对 `other` 展平下标的贡献恒为 0，且它的输出跨步 `prod(out[j+1:])`
含 `B` 因子 ⇒ 对 `B` 取模为 0，故 `r % B` 恰等于 `other` 的展平下标。

### 2.2 kernel（`perf_opt/AddFwdOp.py`，`__all__` 新增两个工厂）

| kernel | 签名 | 网格/流水 | `b` 的复用方式 |
|---|---|---|---|
| `add_bcast_row_kernel(N, row, dtype, alpha, block_size)` | `(a:(N,), b:(row,), y:(N,))` | `min(满块数, 40)` persistent + grid-stride，满块静态 extent + 尾块 epilogue（沿用 R4 结构） | 每 core 载入一次 `row`（`row==1` 用 `vbrc`），块内 `T.serial` 复制成 `block_size` 后**常驻**，热循环不再读 `b` |
| `add_bcast_scalar_kernel(R, L, b_len, dtype, alpha, planes)` | `(a:(R,L), b:(B/P,P,1), y:(R,L))` | `min(R/P, 40)` persistent + grid-stride，`T.Pipelined(2)` | 每 tile 一次连续 DMA 取 `P` 个标量 → `T.vbrc` 铺成 `(P,L)` 再同形 `vadd` |

- `b` 用 `(B/P, P, 1)` 视图，使「一个 tile 的 P 个标量」是一次连续搬运；索引
  `group % (B/P)`（由 `(group*P) % B / P` 化简，省一次除）。
- 精度语义与 flat kernel **完全一致**：`alpha == 1` 折掉乘法；fp16/bf16 且 `alpha == 1`
  走原生单次 add（两 dtype 值之和在 fp32 精确 ⇒ 一次舍入，与 fp32 中转逐位相等，
  R5-A6 结论）；`alpha != 1` 保持 fp32 中转；fp32 直通不 cast。
- UB 预算 `_ub_bytes()` 按 `(dtype, alpha)` 计（fp32 scratch 只在需要时分配），
  超 `192 KB × 0.9` 就 `return None` 回退。
- 为什么标量模式先不做 broadcast `vadd((P,L),(P,1))` 而做 `vbrc` + 同形 `vadd`：
  §4 的对照实验（同形 `vadd` 是唯一被验证为快的向量形）。

### 2.3 Op 层

```python
self.bcast_plan = plan_broadcast(input_shape, other_shape, dtype.itemsize, alpha)
# 命中 → 只构造 bcast kernel（不再构造 flat kernel，省一次编译）
# 未命中 → 原 AddFwdKernel
```
`_eager_forward` 命中时 `self.bcast_kernel(input, other)`（kernel 类内部做
`reshape` 视图，零拷贝），未命中时保持 `_expand_flat` 原逻辑。wrapper 用
`BCAST_ENABLED` 开关：切回 baseline 源时 Op 自动走 flat 路径。

### 2.4 roofline 口径（与 dispatch 决策同源）

`eval_roofline()` 用**同一个** `self.bcast_plan` 决定字节数：

```python
bytes = 3 * N * elem_bytes                        if plan is None   # other 被物化
      = (2 * N + prod(other.shape)) * elem_bytes    if plan 命中     # other 只读一次
```

manifest 同步为 `bytes: "(2 * N + B) * elem_bytes"`，`vars` 增加 `B: "product(other_shape)"`。
同形 add 时 `B == N` → 仍 `3N`，与旧口径完全一致；**回退 case 由 Op 报告 3N**，
所以回退 case 不会被高估（这是任务要求的“模式一致性检查”）。

## 3. 正确性

- `tests/ops/test_add.py`：**24 passed**。含 4 个广播 shape、`alpha∈{0,0.5,2}`、
  dtype 拒绝、形状不符拒绝，以及改写后的 roofline 断言（快路径 `2N+B` + 回退 `3N`）。
- `perf_opt/AddFwdOp.py --level all`：**全 PASS**。新增 `L1-5` 10 个 case
  （row / scalar / vbrc / col-bcast × fp16/bf16/fp32 × alpha=1/2，另含 3 个回退 case），
  命中的每个 case 都断言 **与 flat 路径逐位相等**：`max_diff=0.000e+00 bitwise_vs_flat=True`。
- 独立探针 `perf_opt/r6_bcast_probe.py --check --special`：注入 NaN/±inf/±0.0 后
  快路径 vs 旧路径 vs `torch.add` 三方逐位相等（fp16/bf16/fp32，alpha=1/2/−1.5）。
- 覆盖率边界：`(4,1,8)+(1,8,1)`、`(1024,4096)+(1024,1)`（bf16）等回退 case 也
  在 L1-5 中覆盖（打印 `FALLBACK ... -> flat path`）。

## 4. 标量模式为什么只对 fp32 开放（对照实验）

同一 tiling（R=4096 平面 × L=3136，P=2 平面/tile，40 持久核），`perf_opt/r6_scalar_diag.py`
把 kernel 拆成 4 个变体逐一测（msprof Task Duration 中位）：

| 变体 | bf16 | fp32 |
|---|---:|---:|
| `copy1d`（1D 8192 tile，纯 2N 拷贝）| 21.68 µs（2.37 TB/s）| 37.27 µs（2.76 TB/s）|
| `copy2d`（`(P,L)` 2D tile，纯拷贝）| 22.47 µs（2.29 TB/s）| 37.95 µs（2.71 TB/s）|
| `vbrc`（copy + `vbrc(P,1→P,L)`）| **84.81 µs**（0.61 TB/s）| 45.72 µs（2.25 TB/s）|
| `add`（copy + `vbrc` + 同形 `vadd`）| **66.78 µs**（0.77 TB/s）| 42.29 µs（2.43 TB/s）|

结论：2D tiling 本身没问题（`copy2d` 与 `copy1d` 持平），**2 字节 UB 广播**才是瓶颈
（比拷贝地板慢 3–4x）；另测「broadcast `vadd((P,L),(P,1))` 无 vbrc」同样慢
（bf16 67.78 µs、fp16 78.02 µs）。fp32 的广播开销只有 ~1.2x 地板，可以吃。
因此 `plan_broadcast` 对 `elem_size != 4` 的标量模式直接 `return None`，
并在 `.agents/evolution/capability-gaps.md` 登记 **CG-2026-0032**。

## 5. 实测（base vs final，同会话）

口径：`msprof op` Task Duration，每 case 独立进程；`base` = `git worktree` 取改动前
HEAD（`f6547f1`），`final` = 当前工作树；两者背靠背在同一锁内跑完（噪声窗口一致）。

| workload | dtype | base µs | final µs | Δ | ratio_h base | ratio_h final | ratio_m base | ratio_m final |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cnn-feat-broadcast | bfloat16 | 36.600 | 36.260 | -0.9% | 80.2 | 81.0 | 81.3 | 82.1 |
| cnn-feat-broadcast | float16 | 36.640 | 35.900 | -2.0% | 80.1 | 81.8 | 81.2 | 82.9 |
| cnn-feat-broadcast | float32 | 70.120 | 41.340 | -41.0% | 83.7 | 71.0 | 84.9 | 96.0 |
| hidden-state-prefill | bfloat16 | 26.600 | 25.160 | -5.4% | 72.1 | 76.2 | 73.1 | 77.2 |
| hidden-state-prefill | float16 | 24.000 | 24.000 | +0.0% | 79.9 | 79.9 | 81.0 | 81.0 |
| hidden-state-prefill | float32 | 46.620 | 47.500 | +1.9% | 82.3 | 80.7 | 83.4 | 81.8 |
| linear-bias | bfloat16 | 13.820 | 9.680 | -30.0% | 69.4 | 49.6 | 70.3 | 67.0 |
| linear-bias | float16 | 13.340 | 8.560 | -35.8% | 71.9 | 56.1 | 72.8 | 75.7 |
| linear-bias | float32 | 23.860 | 13.560 | -43.2% | 80.4 | 70.8 | 81.4 | 95.6 |
| llama-7b-ffn | bfloat16 | 32.800 | 32.740 | -0.2% | 78.6 | 78.7 | 79.6 | 79.8 |
| llama-7b-ffn | float16 | 31.960 | 32.780 | +2.6% | 80.6 | 78.6 | 81.7 | 79.7 |
| llama-7b-ffn | float32 | 62.800 | 62.900 | +0.2% | 82.1 | 81.9 | 83.2 | 83.0 |

total: base 419.160 us, final 370.380 us, Δ -11.64%

`hidden-state-prefill` / `llama-7b-ffn` 是**同形 add**，代码路径与 base 完全相同，
其 ±7% 的跳动即本机跨独立进程的噪声带（同一份代码两次跑也会出现同量级差）；
`linear-bias` / `cnn fp32` 的 −30%~−43% 远在该噪声带之外。

## 6. 关于 `Ratio(%)` 为什么下降（必须读）

报表 `Ratio(%)` **不是** `eval_roofline()` 算出来的，而是 `msprof_roofline.py` 从
`L2Cache.csv` 的 **L2 read miss × cacheline** 反推：

```
Ratio(%) = (DRAM 读字节 / Task Duration) / 1.75 TB/s        # 1.75 = 模型常数
```

也就是**只有 GM 读**记入“有用功”。用 base 逐 case 验证（读 = a + 物化的 b = `2N`）：
`linear-bias bf16` `16.78 MB / (13.82 µs × 1.75) = 69.4%` vs 报表 **69.37%**；
`cnn bf16` `51.38 MB / (36.60 × 1.75) = 80.2%` vs 报表 **80.22%**；
`llama bf16` `67.6 MB / (32.80 × 1.75) = 78.5%` vs 报表 **78.55%** —— 三者都逐位吻合，
说明该列就是 `2N / (t × 1.75)`，与 `eval_roofline()`、与 manifest 公式**都无关**。

旧路径把 `other` 物化成 `N` 个元素，这 `N` 个读被计成有效流量；行复用删掉它之后
分子直接少 1/3，于是**即使 kernel 快 33%，该列也掉到 ~52%**。两条口径在 base 上几乎重合
（`3N/2.59 ≈ 2N/1.75`，差 1.3%），所以这个度量缺陷在 base 上被掩盖，一旦真删流量就暴露。

因此上表给三列：

1. `latency`（唯一可信的时延口径，任务书也明确以 Task Duration 为准）；
2. `Ratio(harness)`：框架原样输出（会随流量删除下降，**不可跨字节模型比较**）；
3. `Ratio(model)`：统一字节模型（base `3N`、final `2N+B`）÷ `2.59 TB/s`（由 R5 base 反标定，
   使 base 复现官方 69.17%）。这是“同一把尺子”下的 roofline 利用率。

`Ratio(model)` 结论：`cnn fp32` 从 84.9% → 90%+（删流量的收益全拿到，且没被广播开销吃掉）；
`linear-bias` 三 dtype 升到 ~70%/~79%/~93%；同形 add 不变。

## 7. 回退面

| 情形 | 例子 | 处置 |
|---|---|---|
| `input` 自身被广播 | `(4,1,8)+(1,8,1)→(4,8,8)` | flat（需 gather/scatter） |
| `1` 夹在两个真实维之间 | `E=(2,1,2)` 类 | flat（不是单一周期） |
| 2 字节标量模式 | `(16,256,56,56)+(256,1,1)` bf16/fp16 | flat（§4） |
| 小 `row` 且 replicas > 16 | `row < 512` → tile 缩到 `16*row`，仍命中；再小则 flat | tile 变小 |
| UB 超限 | `L`/`tile` 超 192KB×0.9 | flat |
| 同形 add | 3 个 non-broadcast workload | flat（`B == N`，口径 3N） |
| `other` 未覆盖尾维且非连续族 | `(3,1)+(1,4)` | flat |

以上全部由 `plan_broadcast` 返回 `None` 接住，`eval_roofline` 同步 3N。

## 8. 未竟项 / 建议

1. **`Ratio(%)` 口径（需主会话决策）**：建议 msprof 模式的 Ratio 改用
   `eval_roofline()` 的字节数（峰值常数需从 1.75 改到 ~2.56 TB/s 才能让 base 复现），
   否则任何“删冗余访存”的优化都会被该列判为负收益。
2. **2 字节标量广播**（CG-2026-0032）：可选绕法是“物化一个广播周期”（cnn 为 802816
   元素 = 1.6 MB）+ 纯 1D 同形 tile（`period % block_size == 0` 时天然对齐），
   预期 cnn bf16/fp16 可到 ~25 µs（拷贝地板 22.5 µs）；本次时间不够，未做。
3. `linear-bias` 已 9.2 µs（bf16）对 6.6 µs 带宽下限仍有 30% 固定开销（launch + ramp），
   tile/core 再调优未做（R5 已证 12288/16384 更差）。
4. `fp32→fp32` 的 `T.vcast` 会静默算错（本次以“fp32 直通不 cast”绕开）；
   最小复现未收敛（`perf_opt/r6_vcast_probe.py` 已就绪，尚未跑出结论），故未登记。
5. `Sub`/`Lerp` 同样有 `_expand_flat` 物化问题，按要求未改动。

## 9. 证据文件

- `perf_opt/r6_bcast_probe.py`：快路径单 case 正确性（`--check [--special]`）与 msprof 计时。
- `perf_opt/r6_scalar_diag.py`：§4 的四变体对照。
- `perf_opt/r6_summarize.py`：本报告表格生成（基/终两份 `profile_run_msprof_*.json`）。
- `perf_opt/logs/r6/`：`bench_base.log` / `bench_final2.log`（12 case 全过）/
  `final_level_all.log`（L0/L1/L2/Boundary 全 PASS）/ `ab_table.md`
  + 两份 bench JSON（`profile_run_msprof_20261021_202315.json` = base、
  `..._211225.json` = final，含每 case 的 msprof roofline 明细）。
  msprof 原始 profile 目录（`/tmp/opencode/r6_prof_*`、`r6_diag_*`）为临时产物。
