# 性能反馈（[DESIGN_LIMIT]）

## 触发判定

- 设计层归因: `examples/topk_selector/_topk_selector_kernel/DESIGN.md` 的算法选型——**host 侧 40 轮阈值二分 + 两个"无展开轮次"的简单内核**（该设计明确记录:`≥2 份展开体 + 收尾` 会在本工具链崩溃,故二分被迫搬到 host）。性能天花板由**算法趟数**决定,不是 tiling/参数可解:每个 tune workload 的 count 项都是 `40 × 单趟成本`,且这 40 趟与取值无关、无法合并。
- 结构性加速估计: **≈6.5x**（依据: 本轮 msprof 实测外推。final `topk-1024` = 752.483 µs,其中 count 项 = `40 × 16.761 = 670.44 µs`（89.1%）,select 项仅 82.043 µs。若换成"单趟类"算法（GPU 版 radix-select 形态）,count 项退化为 1 趟 + 无 40 次 host 往返,按实测单趟 ≈17 µs 外推 total ≈ 100 µs 量级;相对 baseline 1168.700 µs 为 **≈10x**。两个数量级均 > 2x。）

## 参照锚定

- 参照实现:
  1. GPU 侧 TileOPs 的同算子实现 **CUDA-thread radix-select**（结构清单见 `examples/topk_selector/_topk_selector_kernel/DESIGN.md` §3.2「源算子三问解读」的算法与优化手段段落）;
  2. 仓内参考实现集 `.agents/skills/tilelang-op-optimize/references/pattern-library/cases.md`;
  3. 同仓 Vector 族参考 kernel `examples/flash_attention/`（**不同族**,仅用于确认 v-op/归约 API 的可达形态）。
- 不可移植论证（逐条,均有实测或文档依据）:
  - GPU 参照的 5 个结构要素在 npuir **全部不存在**: `threads=1024` / `T.get_thread_binding()` / 256-bin 共享直方图 + `T.atomic_add(..., return_prev=True)` / `T.sync_threads()` / `T.dynamic()` 形状（DESIGN.md §3.2 逐条记录,且 `TILELANG_ASCEND_MODE` 的 K3/K9 移除项）;
  - 单趟直方图/radix 若要在核内迭代,需要**同一 kernel 体内出现 ≥2 份"含 `reduce min/max` + (1,1) v-op 链"的迭代体**——实测该结构在 `bishengir-compile` 直接崩溃（`probe_inbranch_rounds.py`: rounds=1 正常、rounds=2 崩;展开与 `T.serial` 两种写法同样崩,`logs/round1_l0.log` / `logs/round1b_l0.log`）;
  - 行并行是唯一能摊薄 40 趟成本的结构（把行映射到 48 个 Vector 核）,但实测**多 block 的 GM 写间歇性丢行**（`probe_round2_isolate.py`: M=8/N=1024 时 3 行读回次正规垃圾;而同一形态 `probe_multiblock_rows.py` 曾 PASS 一次）,不能作为交付基础;
  - 用"运行期索引的标量散射"做直方图累加亦不可行: 该写在 `T.Parallel` 之外**被静默丢弃**（`probe_gather_out.py` 输出 `[12,12,12,12]`）。
  - 结论限定: 上述否决均针对**实测过的结构族**与**当前工具链戳**;「单趟算法在本硬件上不可达」不作为普遍结论（见 `negative-claim-evidence.md` 的证据边界要求）。

## 假设

`DESIGN.md` 的冻结假设（引用自 `examples/topk_selector/_topk_selector_kernel/DESIGN.md`）:

1. 本算子只做**精确** top-K 索引（契约是集合,顺序无关）,因此阈值必须收敛到"count(x ≥ thr) == K"的精确值;
2. 在该目标下,host 侧二分是唯一可行的阈值求解方式（核内迭代被编译器封死）;
3. 在 manifest 的 roofline 口径里,`bytes = 4*M*N + 4*M*k` 只计**一趟**读写,即隐含"算法只需一趟"的假设。

假设 3 与本设计直接矛盾: 精确阈值需要 40 趟比较,故占峰比的**结构上限**约为 `1/40`（即便达到满带宽）。这是 `[DESIGN_LIMIT]` 的核心理由。

## 实测

口径: `msprof op` 的 `Task Duration(us)`（`--launch-count=15 --warm-up=5`,取中位数）,`op_us = n_count_launches × count_us + select_us`。逐行记录见 `perf_opt/perf_records.jsonl`（append-only）,raw profile 见 `perf_opt/profiles/`。

| workload | baseline(µs) | final(µs) | 提升 | count 项占 final | Ratio 迁移后 → 优化后 |
|---|---:|---:|---:|---:|---:|
| topk-1024 | 1168.700 | 752.483 | −35.6% | 670.44 / 752.48 = **89.1%** | 0.0032% → 0.0049% |
| topk-4096 | 1963.939 | 1251.031 | −36.3% | 976.84 / 1251.03 = **78.1%** | 0.0038% → 0.0059% |
| topk-8192-tiled | 3355.778 | 1879.579 | −44.0% | 1410.44 / 1879.58 = **75.0%** | 0.0044% → 0.0078% |

dispatch 结构（两代一致）: `count kernel × 40`（每次一个阈值）+ `select kernel × 1`。本轮把**与算法无关**的两项成本压掉（select 的标量 `cumsum` 与逐元素散射 → 零散射单调过滤,select 单项 3.58–6.06×）,剩下的 count 40 趟是设计族硬成本。

## 影响面

- 算子族: `attention_indexing`（精确 top-K 索引）,以及任何"需要精确分位/阈值"的索引类算子（top-p、阈值筛选、稀疏化 mask 生成）;
- shape 范围: 所有 workload 同构受影响（count 项占比 75%–90%）,与 M/N/K 无关;
- 改设计预期影响: 需要 **Stage 1 重设计 + Stage 3 重实现**（单趟直方图/radix 选型）,且**前提是工具链先提供**以下任一能力:① 同体内可重复迭代体（修 `bishengir-compile` 崩溃）;② 可靠多 block GM 写;③ 块内原子或向量散射。三者当前均不具备,故**不是本轮可推进的修订**。

## 反馈结论

精确 top-K 的阈值求解在本工具链上必须做 40 趟 host 侧二分,占峰比的结构上限约为 manifest 单趟 roofline 的 1/40;本轮已把算法无关成本压到最低（select 3.58–6.06×,总时延 −33.7%～−42.9%）,要突破需换单趟算法族,而该族所需的三种原语当前都不可用。

建议路由: 附录补记
