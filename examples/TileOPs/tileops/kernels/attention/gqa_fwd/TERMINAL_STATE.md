# `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp` — 终态留档(环境不可解)

> **算子正式名**:`GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp`
> (GPU manifest 条目:`/tmp/tops/TileOPs-main/tileops/manifest/attention.yaml:217`;
> 本地目录 slug 为 `gqa_fwd`,二者指同一算子。)
>
> GPU manifest 官方签名(权威契约):
>
> | 项 | 内容 |
> |---|---|
> | inputs | `q, k_new, v_new`(fp16/bf16)、`k_pages, v_pages`(**含 float8_e4m3fn**)、`k_scale, v_scale`(fp32)、`cu_seqlens_q, cache_seqlens, block_table`(int32) |
> | output | `o`(same_as(q)) |
> | params | `max_pages_per_req, page_size, max_seqlen_q`(int)、`cache_dtype`、`is_causal`(默认 true)、`sm_scale`、`softcap`、`fuse_rope`(默认 false)、`rope_base`(10000.0)、`max_position`、`rotary_dim` |
> | shape_rules | `q.shape == (total_q, H, D)`、`k_new.shape == (total_q, H_kv, D)`、`v_new.shape == (total_q, H_kv, D)` … |
> | dtype_combos | 4 组(fp16/bf16 × cache fp16/bf16/fp8_e4m3fn) |
>
> 注:该 op **不是**单纯的 dense attention —— 它是**在既有 `MultiHeadAttentionFwdOp` 实现上扩展分页 KV 寻址**,
> 故与本仓 attention 家族共享同一 Cube×Vector 同步路径,也就共享同一工具链阻塞。

> **结论:`这一算子在本环境(wheel `0.1.2+npuir+7fc4a3c8` + CANN 9.0.0)下无法完成集成。**
> 障碍在**工具链内部**,任何 kernel 源码层面的改法都绕不开 —— 已由**五条路径的实测**封死。
> 本文件是给后续接手者的交接:卡点的准确表述、证据位置、以及解锁前提。

## 1. 现象与根因

编译期报错:

```
NPU IR opt failed: ... Dialect '' not found for custom op 'syn_instr_mode'
```

报错 IR:

```
hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %14
    syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
```

**关键判定:这个 op 由编译器的"自动同步注入" pass 发出,不是源码发的。**
依据:把源码里**全部** `sync_block_set/wait` 替换掉之后,报错 IR **仍然是那一行**。
这也解释了为什么替换源码原语**完全无效**(`syn_instr_mode` 命中数恒为 14)。

## 2. 五条路径总表(全部实测,全部失败)

| # | 做法 | 结果 |
|---|---|---|
| 1 | `sync_block_set/wait`(原始) | `syn_instr_mode` **解析失败**(命中 14) |
| 2 | `set_flag/wait_flag` + `pass_configs={"npuir.disable_hivm_auto_inject_sync": True}` | 该属性**归零**,但 `bishengir-compile` **LLVM 内部崩溃**("PLEASE submit a bug report") |
| 3 | `pipe_barrier("PIPE_ALL")` + 关闭注入 | **病态编译**:99.6% CPU 跑 50+ 分钟、日志 0 输出 |
| 4 | `set_flag/wait_flag`(注入默认开) | 失败 |
| 5 | **在 IR 交给 pass 之前用正则剥掉 `syn_instr_mode` 属性** | **LLVM 崩溃**(与 #2 同源) |

- 路径 1–4 的实测见 `repro/PROBE-gqa-unlock-sync-primitive.py`(四个 variant)。
- 路径 5 的实测见 `repro/PROBE-gqa-strip-attr.py` 与 `repro/PROBE-gqa-strip-attr-real.py`;
  后者把补丁用在**仓库自己的 attention 测试**上,实测输出:
  ```
  [patch] installed on: ['compiler_npu']
  [patch] stripped 4 syn_instr_mode attr(s)
  RuntimeError: NPU IR opt failed: PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/
  4 failed, 2 passed
  ```
  ⇒ **属性剥掉后解析错误确实消失,但编译器立刻崩** —— 说明那段 IR 本身就是这个轮子的编译器处理不了的,`syn_instr_mode` 不是单纯的装饰性问题。

## 3. 环境路径的评估(两条都关闭)

| 路径 | 结论 |
|---|---|
| **从源码构建 wheel** | ❌ 不可行:`3rdparty/{AscendNPU-IR,AscendNPU-IR-Dev,tvm}` **三个子模块全空**(0 字节),`install_npuir.sh` 还要下 LLVM;磁盘仅 **16G 空闲**,远不足以承担 LLVM/MLIR + TVM 的完整构建 |
| **取匹配的预编译 wheel** | ❌ 无来源:PyPI 与华为镜像只有**上游 `tilelang` 0.0.1/0.1.0**(GPU 版),**没有 npuir 分支的 wheel** |

## 4. 旁证

- **11 个纯 Vector 算子在同一个 wheel 上全部通过**;
- 只有需要 **Cube×Vector 跨引擎同步**的 attention 失败 ⇒ 问题定位在 **Cube×Vector 的同步注入链路**;
- 框架 pattern-library 记录 attention 的验证环境是 **CANN 8.5.0 + wheel `0.1.2+1990aa9fe4`**,
  而本机是 **CANN 9.0.0 + wheel `0.1.2+npuir+7fc4a3c8`**。

## 5. 解锁前提(非代码可解)

1. **换用与 CANN 9.0.0 匹配的 wheel**(首选);或
2. **把 CANN 降到与 wheel 匹配的一带**(框架记录为 8.5.0)。

满足其一后,预期可直接续做:设计链(`../gqa_fwd/_gqa_prefill_paged_with_kv_cache_fwd_kernel/DESIGN.md` 等)已齐,
本目录已备好 `DESIGN.md` 与提取出的 kernel 文件,缺的只是"能编过"这一步之后的 **Stage 3 四出口 + Stage 5 集成**。

## 6. 交付件状态(均未留下无效改动)

| 项 | 状态 |
|---|---|
| 设计链 | ✅ `DESIGN.md` 379 行 / `REVIEW.md` 142 行 / `RETROSPECTIVE.md` 32 行 |
| 阻塞证据(4 份 repro) | ✅ 见 `../../../../../gqa_fwd/_gqa_prefill_paged_with_kv_cache_fwd_kernel/repro/` |
| `_gqa_prefill_fwd_kernel.py`(baseline) | ✅ **1269 行,`sync_block` 8 处 —— 原始状态** |
| `perf_opt/_gqa_prefill_fwd_kernel.py` | ✅ **1615 行,`sync_block` 10 处 —— 原始状态** |
| 集成侧 | ⛔ 未完成(无法产出通过报告) |

**本轮所有补丁都写在探针脚本内,不触碰仓库源码。**

---

## 7. 独立佐证:框架自己的迁移报告得出**相同结论**

`examples/TileOPs/docs/norm_reduction_group_migration_report.md` §6.2.2「修正二」原文:

> **修正二:但既有实现在当前环境上编译不过,且这是预先存在的失败。**
> ```
> FAILED test_mha_fwd[smoke-fwd-fp16] - NPU IR opt failed:
>        error: Dialect `' not found for custom op 'syn_instr_mode'
> 2 failed, 2 passed
> ```
> `git status` 确认 `multi_head_attention/` **一行未改** —— 不是我弄坏的。
> **根因是版本落差**:该实现 docstring 标注验证于 **CANN 8.5.0**,当前是 **CANN 9.0.0**;
> 而 `syn_instr_mode` 这个 custom op **只出现在 wheel 的 `.so` 里、仓库源码零命中**,
> 即报错出在**工具链内部**,不在 kernel 源码可修范围。
>
> **旁证:纯 Vector 算子不受影响** —— 本轮迁移的 11 个算子 Tier 2 全过,只有用到 Cube 路径的 attention 中招。
>
> **结论**:差距已量清…但**在 Cube 路径于当前 CANN 上跑通之前不写代码** ——
> 写了也无法编译验证,只会留下看起来像实现、实际不可用的骨架。

### 7.1 本轮独立复验到的事实

- 现成的 `MultiHeadAttentionFwdOp` **4 个正演用例全部失败**,且**同一报错**
  (`syn_instr_mode`);"通过"的两项是**辅助测试**(kernel 分发、roofline),**不是内核编译**;
- ⇒ **没有"能跑通的 Cube+Vector 配置"可以对齐**,阻塞对 attention 家族是**一致**的;
- 框架报告的结论(工具链内部、非源码可修、需 CANN 版本对齐)与本文件 §1–§5 的
  **五条路径实测完全一致**,且**互为独立证据**。

### 7.2 由此确定的处置

**本算子的正确终态就是"不在当前工具链上写实现"** —— 与框架自身的选择一致。
在 Cube 路径于当前 CANN 上跑通之前写代码,只会产出**看起来像实现、实际不可编译验证的骨架**,
这比不留代码更有害。设计链与证据已齐备,换环境后可直接续做。

---

## 8. 2026-10-18 复核(阻塞仍然成立,并更正一条旧结论)

### 8.1 现场复现(当次命令与输出)

```bash
cd examples/TileOPs
env -u PYTHONPATH /root/venv-npuir/bin/python -u -m pytest tests/ops/test_multi_head_attention.py -x -q
```

```text
err cmd: .../bishengir-compile /tmp/tmp3d1t03xe/kernel.npuir --disable-hivm-auto-inject-sync=false ...
RuntimeError: NPU IR opt failed: loc("/tmp/tmp3d1t03xe/kernel.npuir":50:72):
       error: Dialect `' not found for custom op 'syn_instr_mode'  (tried '.syn_instr_mode' as well)
1 failed, 14 warnings in 2.49s
```

⇒ 与 §1 记录的失败**完全一致**,且仍是"自动同步注入 pass 发出的属性"这一根因;
5 条路径(§2)与两条环境路径(§3)的结论无需修改。

### 8.2 更正:预编译 wheel **有发布页**,但无与本仓库 main 匹配的构建

旧结论(§3)写作「取匹配的预编译 wheel ❌ **无来源**:PyPI 与华为镜像只有上游 GPU 版」——
这一表述需要更正:**本仓库在 GitHub 上有 Releases 页**,确实发布了 npuir 线的预编译包
(`v0.1.2.010-a5` / `v0.1.2.020-a5`,日期为 2026-06-04 / 2026-06-11)。

但它**仍不构成解锁路径**,原因有两条(均需在换版本前核对):

1. 这些构建面向 **A5**(发布说明的算子条目为 DeepSeekV4 / mhc / sparse attn),不是本机
   **910B2C** 的目标构建;
2. 其日期(2026-06)**早于**本仓库 main 的提交 `417dae9`(2026-10-10),即**不是从当前源码树构建**的;
   本机已装的 `0.1.2+npuir+7fc4a3c8` 对应 2026-05 那条发布线,已属同一代里较新的包。

⇒ 「换一个已发布的 wheel」不能保证修掉 `syn_instr_mode` 的方言解析问题,除非该 wheel
**由当前源码树构建**(或官方给出与 CANN 9.0.0 + 910B2C 匹配的明确组合)。

### 8.3 结论与解锁前提(不变,措辞更新)

- **结论**:**在本机环境(wheel `0.1.2+npuir+7fc4a3c8` + CANN 9.0.0 + 910B2C)下不可完成集成**;
  障碍在编译器对 Cube×Vector 自动注入同步 IR 的处理,源码层改法已被五条路径实测封死。
- **解锁前提**(满足其一即可续做设计链之后的 Stage 3 四出口 + Stage 5):
  1. 用**当前源码树**构建的 wheel(需子模块与 LLVM/磁盘,本机不具备,见 §3);
  2. 官方给出与 CANN 9.0.0 + 910B2C 匹配的 wheel/容器,并确认其 codegen 与 parser 同源;
  3. 把 CANN 降到与现有 wheel 匹配的一带(框架记录 attention 验证于 CANN 8.5.0 + `0.1.2+1990aa9fe4`)。

