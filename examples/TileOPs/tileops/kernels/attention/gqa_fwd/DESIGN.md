# `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp` 迁移评估与设计 —— 阻塞在工具链版本落差上

- 目标算子:`GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp`
- 目标线:`tile-ai/tilelang-mlir-ascend` / npuir(`main` @ `417dae99`)
- 源:`/tmp/tops/TileOPs-main/tileops/kernels/attention/gqa_fwd.py`
- 结论:**设计可以给出,但实现被一个预先存在的工具链不兼容阻塞** —— 见 §2

---

## 1. 重要更正:仓库里已有 Cube+Vector attention(但跑不起来)

我最初的评估是"**需完整重设计**,2971 行、4 个变体、Cube 矩阵核、Expert + mixcv"。**这个判断不完整** ——
我没查仓库里是否已有可复用的实现。查了之后发现:

```
examples/TileOPs/tileops/kernels/attention/multi_head_attention/
├── multi_head_attention.py                                     284 行
└── multi_head_attention_kernel/
    ├── _gqa_prefill_fwd_kernel.py                             1269 行   ← ★ Cube+Vector 实现
    └── perf_opt/_gqa_prefill_fwd_kernel.py                    1615 行   ← ★ 调优版
```

它**已经是 Cube+Vector 协同**的写法(实测抓取):

| 机制 | 出现位置 |
|---|---|
| `T.Scope("Cube")` | 第 274 行 |
| `T.alloc_L1` / `T.alloc_L0C` | 同上 |
| `T.gemm` | 第 334、376 行 |
| `T.rs("PIPE_C")` / `T.rs("PIPE_FIX")` | 第 342、377 行 |
| `T.Scope("Vector")` | 第 388 行 |
| `T.sync_block_set` | Cube→Vector 握手 |
| `T.vbrc` / `T.vand` / `T.vadd` … | Vector 流 |

而且它已在 manifest 里有条目:`MultiHeadAttentionFwdOp`(dense:`q, k, v, is_causal`)。

**所以这不是"从零重设计",而是"在既有 Cube+Vector 实现上扩展分页 KV 寻址"。** 量级完全不同。

### 1.1 那既有实现的 docstring 本身就是一份设计说明书

它把两阶段流水、跨引擎握手、e-domain online softmax、尾块处理都写清了,还附了 **5 条实测工具链事实**:

1. **GM→L1 块加载必须用 slice 形式 `T.copy`,不能用 `T.load_nd2nz`** —— 后者会把跨步的 BSHD tile
   当作扁平连续内存静默读错(实测 max diff 49)
2. 掩码索引矩阵用 **int16**;`vcmp` 的标量操作数**不能是 `tir.Cast` 表达式**
3. `vbrc` 的标量源必须是 **let-bound 局部变量**,不能是裸字面量
4. `vtanh` 要求 **fp32** 操作数
5. `T.reduce` 的 `clear=False` 是**文档化的静默出错形式**

**这 5 条是本次最省时间的东西** —— 任何 GQA 变体的实现都要绕开这些坑。

---

## 2. 阻塞点:既有实现在当前环境上编译不过(**预先存在的失败**)

### 2.1 实测

```bash
env -u PYTHONPATH /root/venv-npuir/bin/python -m pytest \
  tests/ops/test_multi_head_attention.py -m smoke -q
```

```
FAILED test_mha_fwd[smoke-fwd-fp16] - RuntimeError: NPU IR opt failed:
       loc("kernel.npuir":50:72):
       error: Dialect `' not found for custom op 'syn_instr_mode'
FAILED test_mha_fwd[smoke-fwd-bf16] - 同上
2 failed, 2 passed, 2 deselected
```

### 2.2 这不是我改坏的

```bash
git status --short examples/TileOPs/tileops/kernels/attention/multi_head_attention/
#   (空 —— 与原始克隆完全一致,我一行都没动)
```

我本轮在 attention 目录下**只新增**了 `gqa_fwd/`(extract 脚本产出的 GPU 参考),没有修改任何既有文件。

### 2.3 根因:版本落差(开工时就标记过的风险)

| | 版本 |
|---|---|
| 既有 MHA 实现**验证时的环境** | docstring 第 49 行:"on Ascend910B2C / **CANN 8.5.0**" |
| 当前部署环境 | **CANN 9.0.0** |
| 运行用的 wheel | `0.1.2+npuir+7fc4a3c8`(2026-05,面向 CANN 8.5.0) |
| 仓库 main | `417dae99`(2026-10) |

**关键证据**:`syn_instr_mode` 这个 custom op **只出现在 wheel 的 `.so` 里**,
仓库内核源码**零命中**:

```
grep -rln "syn_instr_mode" examples/TileOPs/tileops/        -> (无)
grep -rln "syn_instr_mode" $VENV/.../tilelang/              -> lib/libtilelang.so
                                                              lib/libtilelang_module.so
                                                              lib/libtilelangir.so
```

⇒ 该 dialect 报错出在**工具链内部**(wheel 自带的 TileLangIR/BishengIR 组件),
不在 kernel 源码里。**无法通过改 kernel 源码修掉。**

### 2.4 为什么只有 attention 中招

本轮迁移的 11 个算子(逐元素 5 个 / 归约 3 个 / norm 3 个)的 Tier 2 **全部通过**。
只有 attention 报 dialect 错 —— 因为它用到了其它算子没用到的 Cube 路径
(`T.Scope("Cube")` / `T.gemm` / `T.PIPE_FIX`)与跨引擎同步原语,
而这些正是 `syn_instr_mode` 相关 lowering 覆盖的范围。

**即:纯 Vector 算子在这套环境上没问题;涉及 Cube 的算子过不去。**

---

## 3. 若要继续:可执行的路径

| # | 路径 | 说明 |
|---|---|---|
| **A** | **对齐工具链版本** | 用与 CANN 9.0.0 匹配的 tilelang(wheel 或源码编译)重试。开工时曾评估源码编译需数小时 + 数十 GB,当时为避免磁盘风险选了 wheel —— 现在看,**Cube 算子的移植必须先解决这一项** |
| **B** | 定位 `syn_instr_mode` 的 lowering | 它在 `libtilelangir.so` 里。若源码里有对应实现,可在源码编译时修补;需要先把 wheel 与仓库 `3rdparty` 的版本对齐 |
| **C** | 绕开 Cube 路径 | 用纯 Vector 重写 attention —— 但 `T.gemm` 是性能核心,纯 Vector 的 attention 在 910B 上性能会差很多,且要重新设计分块,量级反而更大 |
| **D** | 只做 dense 变体、不碰 paged | 无意义:`MultiHeadAttentionFwdOp` 已经存在且同样过不去 |

**建议:先做 A。** 在 Cube 路径能在当前 CANN 上跑通之前,写 paged 变体的任何代码都无法验证 ——
写出来也只是不可编译的骨架,与我在 TopkSelector 上拒绝留下骨架的理由相同。

---

## 4. paged 变体的差距(已量清,供 A 解决后直接开工)

GPU 侧谱系(`gqa_fwd.py`):

| 行 | 工厂 | NPU 侧状态 |
|---|---|---|
| 789 | `_gqa_prefill_fwd_kernel` | ✅ **已迁**(即 `multi_head_attention/`) |
| 1041 | `_gqa_prefill_with_kv_cache_fwd_kernel` | ❌ 待迁 ← **本算子** |
| 1356 | `_gqa_prefill_with_kv_cache_rope_append_kernel` | ❌ 待迁(变体) |
| 1452 | `_gqa_prefill_with_kv_cache_rope_fwd_kernel` | ❌ 待迁(变体) |

**最省事的发现:两者的工厂签名完全一致。**

```python
_gqa_prefill_fwd_kernel(batch, heads, heads_kv, seq_len_q, seq_len_kv, dim,
                        is_causal, sm_scale=None, softcap=0.0, dtype='float16')
_gqa_prefill_with_kv_cache_fwd_kernel(batch, heads, heads_kv, seq_len_new, seqlen_kv, dim,
                                      is_causal, sm_scale=None, softcap=0.0, dtype='float16')
```

⇒ 差别**不在参数,在主体**:K/V 的来源从"新 token"变为"分页 KV cache"。

### 4.1 增量清单(相对已迁的 dense 版)

| 增量 | 内容 |
|---|---|
| 分页寻址 | `block_table` 把逻辑页 → 物理页;`k_pages`/`v_pages` 按 `page_size` 切分读 K/V |
| 变长 | `cu_seqlens_q`(变长 Q)、`cache_seqlens`(每行已缓存长度) |
| 可选 RoPE 融合 | `fuse_rope` / `rope_base` / `max_position` / `rotary_dim`;对应另两个变体 |
| 可选 FP8 KV cache | `k_scale` / `v_scale` / `cache_dtype` |
| 参数 | `max_pages_per_req` / `page_size` / `max_seqlen_q`(静态形状实例化) |

10 个输入:`q, k_new, v_new, k_pages, v_pages, k_scale, v_scale, cu_seqlens_q,
cache_seqlens, block_table`。

### 4.2 迁移策略建议

1. **先只做 dense-paged 单变体**(不带 RoPE、不带 FP8),这与 `multi_head_attention` 的差距最小
2. 复用既有实现的两阶段 Cube/Vector 流水与跨引擎握手,**只替换 K/V 的取数路径**
3. 变体(rope_append / rope_fwd / fp8)在其后逐个加
4. 全程遵守 §1.1 的 5 条工具链事实

---

## 5. 本算子的诚实结论

- **设计层面**:差距已量清(§4),策略已给出(§4.2),前置工具链事实已收集(§1.1)
- **实现层面**:**被预先存在的工具链不兼容阻塞**,且该阻塞**不在 kernel 源码可修范围内**
- **与 TopkSelector 的区别**:TopkSelector 是我写的 kernel 撞上编译器;
  GQA 是**仓库自带的、上游已验证过的实现**在当前 CANN 版本上跑不起来 —— 性质不同,
  但都不支持"现在就写实现"

**在工具链版本对齐之前,不写 paged 变体的代码。** 写了也无法编译验证,
只会留下一个看起来像实现、实际不可用的骨架。
