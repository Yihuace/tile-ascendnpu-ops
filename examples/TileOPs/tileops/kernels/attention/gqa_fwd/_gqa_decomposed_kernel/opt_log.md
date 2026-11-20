# GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp — Stage 4 调优记录

## 1. 剖析:482 ms 里钱花在哪

| 段 | 实测耗时 | 判定 |
|---|---:|---|
| `_gather_kv`(分页 gather + dequant) | 0.3 ms | 不是瓶颈 |
| 单次内核调用 `kq`/`ks`/`kp` | 0.11 / 0.11 / 0.10 ms | 内核本身很快 |
| **内核工厂创建** `kernel_qk(M,N,D)` | **11 ms / 次** | ⭐ 元凶 |

`gqa-mid` 有 `2 batch × 4 行块 = 8` 次循环、每次重建 3 个工厂
⇒ `24 × 11 ms ≈ 264 ms`(占当时 346 ms 的绝大部分)。**数学计算只占约 3%。**

## 2. 三项优化

```python
# ① 工厂按形状缓存(绝不在循环里重建)
_KERNEL_CACHE = {}
def _get_kernels(m, n, d):
    hit = _KERNEL_CACHE.get((m, n, d))
    if hit is None:
        hit = (kernel_qk(m,n,d), kernel_softmax(m,n), kernel_pv(m,n,d))
        _KERNEL_CACHE[(m, n, d)] = hit
    return hit

# ② 每 head 张量变换提到循环外(只依赖 head,不依赖行块)
Kt_heads = [K[:, h // rep, :].t().contiguous().to(torch.float16) for h in range(H)]

# ③ 删掉冗余 repeat_interleave,改索引映射
#    K_rep = K.repeat_interleave(rep, dim=1)   ← 大张量复制 ×2,结果只被逐 head 读
```

**评估后否决**:把多 head 合并成一次下发 —— 叠到 M 维会放大 M,而 L0C 上限要求
`M × n_kv ≤ 16384`,必须等比缩小行块,下发次数**并不减少**(16 次 vs 16 次)。

## 3. 结果(msprof / 官方 report)

| 指标 | 迁移后 | 优化后 | 提升 |
|---|---:|---:|---:|
| gqa-mid 延迟 | 482.34 ms | **71.33 ms** | **6.8×** |
| gqa-mid(本机摊销) | — | 62.5 ms | 7.7× |
| gqa-smoke | — | 3.51 ms | — |
| 正确性 | rel 4.58e-04 | rel 1.11e-03 / 1.13e-03 | 未回退 |

## 4. 正确性(端到端)

| 检查 | 结果 |
|---|---|
| `Q = I` ⇒ `S == Kt` | max_err **0.0**(精确恒等) |
| 随机矩阵 QK | rel **2.8e-4** |
| softmax(带真实因果掩码) | **2.4e-4** |
| 端到端 因果 / 非因果 | rel **4.6e-4 / 6.5e-4** |

## 5. 为什么 Ratio 只有 0.49%

msprof 的延迟是**每个内核实例耗时的中位数**(`msprof.py:664-666`)。本算子一次调用发
**96 个**内核,中位数只代表其中一个子内核 ⇒ msprof 口径对优化完全不可见。
真实提速以墙钟为准:**482 → 71.3 ms(6.8×)**。

报表 `status=partial`:`captured_op_count = 2`(`main` + `main_mix_aic`),
触发 harness 的 `>1 → invalid` 规则 —— **任何 Cube 算子都会撞上这个启发式**。
