# Integration Log — `TopkSelectorOp`

- **meta**:`examples/TileOPs/tileops/kernels/attention_indexing/topk_selector/.migration_meta.json`
- **integrated**:`_topk_selector_kernel` → `tileops/kernels/attention_indexing/topk_selector/_topk_selector_kernel/_topk_selector_kernel.py`
- **wrapper**:`tileops/kernels/attention_indexing/topk_selector/topk_selector.py`(`TopkSelectorKernel`)
- **op**:`tileops/ops/attention_indexing/topk_selector.py`(`TopkSelectorOp`)
- **manifest**:`tileops/manifest/attention_indexing.yaml`
- **workload**:`tileops/workloads/attention_indexing.py`
- **test**:`tests/ops/test_topk_selector.py`
- **bench**:`benchmarks/ops/bench_topk_selector.py`
- **ascend_mode**:`Developer`(**以环境变量形式生效**)
- **attempts**:集成侧 2 次(第 1 次:内核副本是**加分块参数之前**的旧版本 → `TypeError`;同步后通过)

## 验证结果

- **report_run_json**:`examples/TileOPs/reports/tileops/20261016_135447_124960_TopkSelectorOp/run.json`
- **report_status**:**`passed`**
- **correctness**:**9 / 9**(`pass_rate = 1.0`,`correctness_failed = 0`)
- **benchmark**:**4 / 4**(`benchmark_failed = 0`)

## 算法:重设计,不是移植

GPU 版是 CUDA-thread radix-select,依赖 `threads=1024`、`T.get_thread_binding()`、
256-bin 共享直方图 + `T.atomic_add(..., return_prev=True)`、`T.sync_threads()`、
`T.dynamic()` 形状 —— **npuir 一个都没有**(K3/K9 移除 `threads`;无块内原子;无 CUDA 语义栅栏;形状必须静态)。

**NPU 设计:两个"无展开轮次"的简单内核 + host 侧阈值二分**

| 内核 | 作用 |
|---|---|
| `_topk_count_kernel(x, thr) -> cnt` | 每行统计 `x >= thr` 的元素个数 |
| `_topk_select_kernel(x, thr) -> idx` | 每行输出被选中元素的索引(最多 K 个) |

host 用行 min/max 作区间,40 轮二分调用 count,收敛后调用一次 select。

## 三条由实测强制得出的架构约束(**都有 repro 证据**)

### 1. 不能把二分展开在核内

`repro/PROBE-topk-min-repro.py`:含收尾 **且** ≥2 份展开的二分体 → **构建期段错误**;
`rounds=1` 与全部四种写法组合都能编译并运行;`rounds>=2` 必崩。
与具体指令无关(去掉散射 / cumsum / 切片拷贝 / 循环依赖 都不改变结果)。
`TILELANG_DUMP_IR=TRUE` 显示 IR 死在 TVM bufferization 的
`materialize_in_destination`(K 长切片拷贝的步长不匹配),但那是最后打印出来的东西,不是全部原因。

### 2. 多 block 的 GM 写**不可靠**(本算子最关键的发现)

`repro/PROBE-rows-arch.py`:

| 架构 | 结果 |
|---|---|
| `T.Kernel(M)`(每行一 block) | ⛔ 4 行只落 2 行 |
| `T.Kernel(1)` + `T.serial` 遍历行 | ✅ 4 行全对 |
| `T.Kernel(min(M,核数))` + 核内串行 | ⛔ 对小 M 等价于"每行一 block",同样 FAIL |

⇒ **全文使用单 block + 核内串行遍历行**。

### 3. 行超 UB 必须分块,且**由算子自动选分块**

benchmark 一度失败于:
```
error: ub overflow, requires 1841920 bits while 1572864 bits available!
```
原因是 manifest 里 `(8, 8192)` 那条用例没给 `tile`,整行一次装进 UB 直接溢出。
修法是**让算子自选**:`_auto_tile(N)` 取 **N 的、≤2048 的最大因子**
(取因子可让分块循环精确,既无参差尾巴要守卫,也避开"对 serial 循环变量做运行时条件判断"的已文档化陷阱)。

## 其它实测原语规则(交付件依赖)

- `T.Parallel` 赋值循环的右值**不能读 `(1,1)` 缓冲**;GM 标量要先拷进 `(1,1)` UB,再作 **v-op 的标量操作数**用(`T.vadd(row, nts[0,0], diff)`)—— 否则报 `size mismatch`;
- **带 guard 的散射会静默不写** ⇒ 把掩码折进**落点算术**,未选中元素落到**槽位 K**(`scratch` 取 `(1, K+1)`,槽 K 永不被读);
- 计数必须在 **fp32 域**(1-D int32 归约结果是错的);
- `T.prim_func` 体内不能有裸 `return`,也不能有嵌套 `def`。

## 规模偏差(如实记录)

GPU manifest 的完整 workload 是 `seq_len=32768 × seq_len_kv=65536`(topk 至 1024)。
**单 block + 核内串行虽然正确,但那个规模跑不完**,故 manifest 里的 workload 按
**当前内核能跑完的规模**设定(smoke-256 / topk-1024 / topk-4096 / topk-8192 流式),
偏差已写进 `tileops/manifest/attention_indexing.yaml` 与本文件。

**性能未做调优**(迁移 harness 跳过 Stage 4),此处按"正确性优先"交付。
