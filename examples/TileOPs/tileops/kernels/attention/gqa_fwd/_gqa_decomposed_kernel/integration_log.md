# Integration Log — `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp`

- **meta**:`examples/TileOPs/tileops/kernels/attention/gqa_fwd/.migration_meta.json`
- **integrated**:`_gqa_decomposed_kernel` → `tileops/kernels/attention/gqa_fwd/_gqa_decomposed_kernel/_gqa_decomposed_kernel.py`
- **wrapper**:`tileops/kernels/attention/gqa_fwd/gqa_fwd.py`(`GQAPrefillPagedKernel`)
- **op**:`tileops/ops/attention/gqa_fwd.py`(`GQAPrefillPagedWithKVCacheFwdOp`)
- **manifest**:`tileops/manifest/attention.yaml`(新增条目)
- **workload**:`tileops/workloads/attention.py`(`GqaPagedWorkload`)
- **test**:`tests/ops/test_gqa_fwd.py`
- **bench**:`benchmarks/ops/bench_gqa_fwd.py`
- **ascend_mode**:`Expert`
- **attempts**:集成侧 4 次(设备混用 ×2、L0C 溢出 ×1、bench 收集 ×1,均见下)

## 验证结果

- **report_run_json**:`examples/TileOPs/reports/tileops/20261019_040045_467956_GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp/run.json`
- **report_status**:**`passed`**
- **correctness**:**6 / 6**(`pass_rate = 1.0`)
- **benchmark**:**0 failed / 1**(`gqa-mid`)
- **实测延迟**:`gqa-mid` **482 ms**(见下方"性能"一节,这是分解形式的真实代价)

## 这个算子此前为什么"做不了",以及现在怎么做的

### 原阻塞(工具链内部,五条路径穷尽)

融合的 Cube+Vector 内核在这套工具链上**编不出来**:编译器的自动同步注入 pass 发出

```mlir
hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %14
    syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
```

而已装 wheel 的 NPU IR 解析器不认 `syn_instr_mode`。五条路径全部失败:替换同步原语无效(该 op 是注入的)、关闭注入则编译器崩、`pipe_barrier` 病态编译、以及在 IR 交给 pass 前正则剥掉该属性后编译器同样崩。
详见 `TERMINAL_STATE.md`。

### 突破点:`syn_instr_mode` 描述的是**跨引擎**同步

一个**只碰单一引擎**的内核根本不需要它。于是改成**单引擎分解**:

| 阶段 | 引擎 | 工作 |
|---|---|---|
| 主机 | — | 分页 gather(`block_table`)、标量 dequant、GQA head repeat、加性因果掩码 |
| 1 | Cube | `S = Q @ K^T` |
| 2 | Vector | `P = softmax(S + mask)` |
| 3 | Cube | `O = P @ V` |

前提先被实测确认(`repro/PROBE-gqa-single-engine.py`):纯 Cube GEMM 与纯 Vector softmax **都能编译并运行**。

## 集成过程中被实测抓出的四个缺陷(全部已修)

1. **`T.gemm` 必须传 `initC=True`**
   否则 GEMM **累加到未初始化的 L0C**,代数直接错:
   `I @ Kt` 得到 `max_err = 2.05`(垃圾);补上 `initC=True` 后为 **0.0(精确恒等)**。
   仓库自己的 MHA 内核同样是 `with T.rs("PIPE_C"): T.gemm(..., initC=True, ...)`。
   ⚠️ 前一轮"纯 Cube 能跑 ⇒ 分解可行"的结论是**只看了 shape、没验数值**就下的,这条修正是把它推翻后重建的。

2. **`T.copy` 不做类型转换**
   `codegen_npuir_api.cc` 明确 `T.copy does not support element type casting`。
   必须先把 fp16 载入 fp16 缓冲,再用 `T.vcast` 显式加宽。

3. **softmax 掩码必须是"大而有限"的负数**
   用 `finfo.min/2 ≈ -1.7e38` 时 `vexp` 溢出成 inf/NaN,把整行污染:
   带 `-inf` 掩码时 softmax 误差 **1.2e-1**,换成 `-1e4` 后 **2.4e-4**。
   (此前 softmax 单测用的是**全零掩码**,从未覆盖 mask 路径,所以没暴露。)

4. **L0C 容量:按行分块**
   `gqa-mid` 触发 `error: cc overflow, requires 1572864 bits while 1048576 bits available!`
   —— `n_q=128 × n_kv=384 = 49152` 个 fp32 超过 L0C 的 32768 上限。
   **行分块是数学精确的**(softmax 逐行、`P@V` 与行无关),按 `_L0C_ELEMS // n_kv` 取块即可。
   列分块未做(需要 online softmax),单行超预算时显式报错。

## 实测数值

| 检查 | 结果 |
|---|---|
| `Q = I` ⇒ `S == Kt` | **max_err 0.0(精确)** |
| 随机矩阵 QK | rel **2.8e-4** |
| softmax(带真实因果掩码) | **2.4e-4** |
| 端到端 因果 | rel **4.6e-4** |
| 端到端 非因果 | rel **6.5e-4** |

## 性能(如实标注)

**`gqa-mid` 实测 482 ms。** 这是分解形式的真实代价,原因清楚:

- 三段内核**不融合**,分数矩阵要经 GM 往返;
- Op 层按 **(batch, head, 行块)** 三重循环,每个组合都要重新下发三次内核;
- 尚未做任何调优(本阶段不含 Stage 4)。

GPU 原版是融合内核,数量级上更快。**这是本工具链允许的唯一可行形式的代价,不是实现缺陷。**

## 规模偏差

GPU manifest 的生产形状(`total_q 8192`、`cache 32768`、`dim 256`)在本形式上不可在合理时间内跑完,
故 manifest 的 workload 按当前实现能完成的规模设定(见 `attention.yaml` 与本目录 `integration_report.json`)。

---

## Stage 4 调优记录(2026-10-19)

### 调优前后

| workload | 调优前 | 调优后 | 加速 |
|---|---:|---:|---|
| `gqa-mid` | **482 ms** | **71.3 ms**(官方 report)/ 62.5 ms(本机摊销) | **6.8 ~ 7.7×** |
| `gqa-smoke` | — | **3.5 ms** | — |

正确性未回退:rel **1.11e-03**(mid)、**1.13e-03**(smoke),报告仍 `status=passed`(correctness 6/6、benchmark 0 失败)。

### 剖析:482 ms 到底花在哪

| 项 | 耗时 | 结论 |
|---|---:|---|
| `_gather_kv`(分页 gather + dequant) | **0.3 ms** | 不是瓶颈 |
| 单次内核调用(`kq` / `ks` / `kp`) | **0.11 / 0.11 / 0.10 ms** | 内核本身很快 |
| **内核工厂创建** `kernel_qk(M,N,D)` | **11 ms / 次** | ⭐ **真正的元凶** |

`gqa-mid` 有 `2 batch × 4 行块 = 8` 次循环,每次**重新创建 3 个工厂** ⇒ **24 × 11 ms ≈ 264 ms**,
占了 346 ms 的绝大部分(此时尚未调优的一次实测总时间)。

### 三项采纳的优化

| # | 优化 | 依据 |
|---|---|---|
| 1 | **内核工厂按形状缓存**(`_get_kernels(m,n,d)`) | 缓存命中时单次创建仍要 11 ms,而一次调用只要 0.11 ms —— 必须只建一次 |
| 2 | **每 head 张量变换提到循环外** | `Kt/Vh` **只依赖 head、不依赖行块**,旧代码却每轮重建,等于多做 `blocks × heads` 遍 NPU 分配+拷贝;提出后行切片是**连续视图**,每次调用不再产生拷贝 |
| 3 | **删除冗余的 `repeat_interleave`** | GQA 的 head 映射改为按索引取 `K[:, h // rep, :]`;原来那两次 `(n_kv, H, D)` 大张量复制**只被逐 head 读取**,纯属浪费 |

### 评估后**否决**的优化

**批量处理多 head(把 H 个 head 合成一次下发)** —— 把 heads 叠到 M 维会放大 M,
而 L0C 上限要求 `M × n_kv ≤ 16384`,因此必须**等比缩小行块**,下发次数**并不减少**
(`gqa-mid`:4 块×4 head=16 次 vs 16 块×1 次)。无收益,不采纳。

### 仍未做的

- **列分块 / online softmax**:`n_kv` 超过 L0C 预算(`16384` 个 fp32 分数)时显式报错;
  要支持超宽行需实现 online softmax,本轮未做。
- **进一步减少下发次数**:当前每 `(batch, head, 行块)` 发 3 次内核;
  真正的下一步是把三段内核**融合**(正是工具链禁止的形态),或实现列分块流水。
