# 信息收集与信息源优先级

## 目录

- [信息收集与信息源优先级](#信息收集与信息源优先级)
  - [目录](#目录)
  - [1. 强制步骤 0：搜索本项目同类实现](#1-强制步骤-0搜索本项目同类实现)
  - [2. 信息收集步骤](#2-信息收集步骤)
  - [3. 禁止行为](#3-禁止行为)
  - [4. 信息源优先级](#4-信息源优先级)

---

## 1. 强制步骤 0：搜索本项目同类实现

在生成 design.md 前，**必须**执行以下工具调用：

```bash
# 1. 搜索同类算子（根据算子名称）
glob examples/**/*{算子名称}*.py
glob examples/**/*{算子类别}*.py  # 如 gemm, softmax, reduce

# 2. 如果找到同类实现，完整阅读
read examples/{找到的同类实现路径}

# 3. 检查关键技术点
grep "T.Kernel" examples/{同类实现}     # Kernel 维度
grep "T.gemm\|T.gemm_v0" examples/{同类实现}  # GEMM API
grep "T.alloc" examples/{同类实现}      # 内存分配方式
grep "T.Scope\|T.barrier" examples/{同类实现}  # 同步方式
```

## 2. 信息收集步骤

> **算法调研（Phase R）的信息源与查证规则**单独见 [algorithm-research.md](algorithm-research.md)（§5 常见算子族参考表命中行必查、§6 信息源清单）；本文件的优先级表与禁止行为同样适用于调研阶段的候选查证——外部算法只取思路，API 存在性与代价论断必须本地佐证。

1. 查阅 `examples/` 中同类算子实现（**强制步骤 0**）
2. 查阅 **pattern-library（强制步骤 0.5）**：`tilelang-op-optimize` skill 的 `references/pattern-library/`（入口 INDEX.md：layout/elementwise/attention.md 已验证模式与实测代价、traps-*.md 编译器陷阱〔注意版本戳〕、constants.md 硬件常数、cases.md 案例索引〔同类算子正/反例参考〕；检索可用 `python3 .agents/tools/kb_search.py "<查询>"`）
3. 查阅 `docs/Tilelang.language/` 确认 API 可用性和用法（按 AGENTS.md 文档路由规则）
4. 查阅 `docs/开发指南.md` 确认编程模式和 pass_configs 配置
5. 如有参考实现，分析其计算步骤（**仅用于理解数学逻辑，不可直接使用 API**）

## 3. 禁止行为

- ❌ 在没有执行强制步骤 0 的情况下，直接使用外部参考实现的 API
- ❌ 凭记忆猜测 API 名称或参数

## 4. 信息源优先级

| 优先级 | 信息源 | 用途 | 说明 |
|--------|--------|------|------|
| **0** | **本项目 `examples/` 同类实现** | **主要参考：API、编程模式、Kernel 结构** | **最权威**，直接可用 |
| **0.5** | **pattern-library/**（`tilelang-op-optimize` skill 的 `references/pattern-library/`，入口 INDEX.md） | **实测代价数据、已验证模式（布局/轴/陷阱）、硬件常数与案例索引（cases.md）** | Stage 4 调优实测积累；代价类论断优先于 docs 规格与 examples 先例；**条目绑定工具链版本戳**（kb_stale_check 检测），重编译后旧结论待重验，勿引用 `status: overturned/stale` 条目 |
| 1 | `docs/开发指南.md` | API 完整说明 | 补充细节 |
| 2 | `docs/Tilelang.language/` | API 语义与签名速查 | 按 AGENTS.md 文档路由规则 |
| 3 | `testing/python/language/` | 边界用法和测试模式参考 | 测试参考 |
| 4 | **外部参考实现** | **仅用于理解数学逻辑** | **不可直接使用 API** |
| 4.5 | **互联网检索（webfetch，若环境可用）** | **算法调研（Phase R）的候选补充**：论文 / 开源实现 / 厂商文档 / 技术博客中的算法族变体与复杂度结论 | 只取算法思路；外部 API 不可直接采用（存在性须本地佐证）；外部性能数字仅作候选排序参考（GPU 加速比 ≠ NPU 判定依据）；引用须记录来源（URL/论文/仓库 + 访问日期），检索未命中如实记录；与本地任何信息源矛盾时以本地为准；详见 [algorithm-research.md](algorithm-research.md) §6 |
| 5 | `tilelang/language/__init__.py` + `tilelang/language/*.py` | 公开 API 导出关系与前端定义 | API 定义 |
| 6 | `src/op/` + `src/target/` | lowering 与后端实现状态 | 实现验证 |

**规则**：当信息源之间矛盾时，以 `examples/` 为准。若 `examples/` 未覆盖，以 `docs/` 为准。若 `docs/` 未覆盖，以 `tilelang/language/` 源码实际实现为准。**性能代价类论断**（某操作快/慢/阻碍向量化）矛盾时，以 pattern-library 实测条目为准（未覆盖时标注「未实证假设」并进实验裁决流程，见 SKILL.md Phase 2）。**互联网检索结果**与上述任何信息源矛盾时一律以本地为准，且不得直接作为 API 存在性与性能代价的判定依据（见优先级 4.5 行与 algorithm-research.md §6 纪律）。
