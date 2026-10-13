# NPU 性能模式库（已拆分——见 pattern-library/ 目录）

> **本文件已按渐进披露原则拆分为主题文件**（K-2，2026-09-10）——本存根仅保留入口指引与兼容重定向，**不再更新内容**；所有条目已迁移至：
>
> ```text
> .agents/skills/tilelang-op-optimize/references/pattern-library/
> ├── INDEX.md            # 唯一入口：维护规则 + 证伪协议 + 检索规则 + 条目索引（必读）
> ├── layout.md           # §1.1–1.5 向量化轴与布局模式
> ├── elementwise.md      # §1.6 copy-floor 与 MTE2 带宽曲线
> ├── attention.md        # §1.7–1.9 Expert persistent / attention 族
> ├── traps-compiler.md   # §2 编译器/解析器陷阱
> ├── traps-runtime.md    # §2 运行时/数值/语义陷阱
> ├── constants.md        # 硬件常数表（设计期 roofline 口径，D-2）
> ├── cases.md            # §4 案例索引 + 参考实现集 + 反例档案互链
> └── repro/              # 最小可复现代码（ED-B，知识回归测试）
> ```
>
> **历史引用对照**：`pattern-library.md §1` → layout/elementwise/attention.md；`§2` → traps-compiler/traps-runtime.md；`§3` → INDEX.md §2/§3；`§4` → cases.md；`§1.6 常数` → constants.md。
>
> **检索**：`python3 .agents/tools/kb_search.py "<算子族/症状/API/dtype>"`（覆盖本库全部主题 + bottleneck-patterns + algorithm-candidates + capability-gaps + queue pending）；**写入**（optimizer 回写 / evolver 合入）一律作用于主题文件，五种 delta 与预算规则不变（INDEX.md §1）。
