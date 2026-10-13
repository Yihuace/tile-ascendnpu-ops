# repro/ — 最小可复现代码（ED-B 规范）

> 位置：`.agents/skills/tilelang-op-optimize/references/pattern-library/repro/`——知识域内，随知识条目版本化管理。**知识与过程文件解耦**（ED-A）：条目的代码证据唯一合法形态是本目录内的自包含脚本；任务工作区路径（`examples/{project}/{op}/...`、`/tmp/...`）只是 provenance（允许失效）。**原则：经验如果需要代码，就用最少的代码表达这个经验**——过程文件甚至最终调优版本都不一定随 agent 合入主干，经验的可验证性不得依赖它们存活。

## 每个 repro 的要求（ED-B）

1. **可独立执行/可独立检查**：`python repro/<条目ID>.py` 直接跑或通过语法检查——输入自造（脚本内构造最小张量/shape）、不 import 任务工作区模块、不读 `examples/{project}/` 路径（kb_lint 的 `KB-REPRO-DECOUPLE` 机械校验此解耦）；
2. **自描述头**（docstring）：条目 ID 与所在文件、现象/优化点一句话、断言说明、首次验证版本戳、origin_task（provenance）、重验记录（追加式）；
3. **分级断言形态**（按条目 kind，见下节）；
4. **最小化预算**：≤100 行 / ≤8KB——超限即未完成最小化（削减输入规模、剥离与经验无关的结构、只保留慢→快的关键更改）；
5. **单一经验**：一个 repro 只表达一个条目的经验，组合现象拆多文件。

## 分级断言形态（按条目 kind）

**陷阱类（kind: trap；正向/反向经验中的"反向"）**——assert 现象存在或绕法通过：错误数据的 max_diff 阈值 / 编译失败文本匹配 / **绕法通过**（合法形态 bit-exact）。例：`TRAP-load-nd2nz-strided.py`。

**常数类（kind: constant）**——可运行部分 assert 正确性/结构自检；测量部分输出数字并 assert **量级与相对关系**（绝对值仅打印供人对照，环境漂移不锁绝对值）；需要 msprof 的测量步骤在头部写明命令（人工/工具链升级时按需执行，`repro_runner` 只验证可运行部分）。例：`CONST-copy-floor-method.py`（3 载入 + 2 vadd 保活探针，从 lerp `probe_copy.py` 抽取最小化）。

**优化点（kind: pattern；正向经验与优化点）**——**关键代码更改 + 优化见解摘要**，按效应的可复现规模二选一：

- **完整形态**（效应可在小规模独立复现时，如 C 轴切片累加、乘常数倒数）：最小 **before/after 可运行对照**——"慢形态"与"快形态"两个最小 kernel（或同 kernel 的两参数形态），断言**相对关系**（快形态正确且不慢于慢形态 / 收益方向与条目记载一致；收益绝对值仅打印）；
- **delta 形态**（效应只在完整 kernel 规模显现时，如两相位重构、块宽摊减）：**慢→快的关键代码更改**——足以让后续任务定位"改了什么"的最小 diff（慢结构骨架 vs 快结构骨架，≤100 行，`py_compile` 通过，不要求可运行）+ **优化见解摘要**（见下）写入 repro 头部与条目正文。例：`PATT-twophase-restructure.py`（两相位重构的相位分离骨架 + 机制归因）。

**优化见解摘要**（LLM 在回写/蒸馏时从任务上下文生成的一般化洞察）：为什么这个代码更改导致慢→快——机制归因到硬件常数（引用 CONST-\* 条目）或编译器行为（引用 TRAP-\* 条目），写成自包含的一段文字（后续任务不读原 opt_log 也能理解因果）；摘要必须同时落在条目正文（经验自包含）与 repro 头部（与代码同址）。条目正文已含机制说明的（如 PL-1.9 系列的块宽摊减机制解释）即达标，repro 承载代码部分。

## 命名与登记

- 文件名 = 条目 ID（`TRAP-*.py` / `CONST-*.py` / `PATT-*.py`）；
- 条目 front-matter 的 `repro:` 字段登记 `repro/<file>.py`；`repro-missing` 表示待回填（存量条目允许，新 D 类条目缺 repro 降级 Tier 1 入队，merge-policy §1）；
- **责任前移（ED-C）**：repro 的编写与首次运行发生在有环境、有上下文的任务内 Subagent（Stage 3 developer 探针转正 `examples/{op}/repro/`；Stage 4 optimizer 回写 D 类条目与优化模式条目时同步抽取关键代码并实际跑一遍）；evolver 只做**转正（机械拷贝）与校验**（头部规范 + 语法 + 登记），不编写、不裁剪代码。**抽取原则**：从任务工件（perf_opt 探针 / 实验分支 / 参考 kernel）裁出表达该经验所需的最小代码——经验不随过程文件或最终 kernel 的存亡而失效。

## 批量执行（ED-D：知识回归测试）

```bash
python3 .agents/tools/repro_runner.py                     # 全量（trap/const 可运行部分）
python3 .agents/tools/repro_runner.py --filter traps      # 按类别
python3 .agents/tools/repro_runner.py --filter TRAP-load-nd2nz-strided  # 按条目
python3 .agents/tools/repro_runner.py --filter stale      # 工具链升级后受影响条目（kb_stale_check 联动）
```

用途：① 工具链升级重验（`kb_stale_check` 列 stale → `repro_runner --filter stale` → FAIL 即条目推翻、PASS 即刷新版本戳）；② apply 预验证（E-4）与 Tier 1 第二次独立证据（另一上下文重跑通过）复用同一套件；③ delta 形态的 PATT-\* 只做语法级检查（全量 NPU 执行与 msprof 测量按需人工触发）。

## 现存 repro 索引

| repro | 条目 | 形态/断言 | 首次验证 |
|-------|------|---------|---------|
| `TRAP-load-nd2nz-strided.py` | traps-runtime.md TRAP-load-nd2nz-strided | 陷阱：绕法通过（slice 形态 bit-exact；base+size 误读形态见头部重验记录） | 2026-09-07 session（2026-09-10 转正重跑 PASS） |
| `TRAP-T-copy-region-semantics.py` | traps-runtime.md TRAP-T-copy-region-semantics | 陷阱：合法形态通过（[H,D]ub → 4D **slice**；base+size 形态当前工具链 MTE fault 待重验，见头部重验记录） | 2026-09-07 session（2026-09-10 转正重跑 PASS） |
| `TRAP-zero-input-crash.py` | traps-runtime.md TRAP-zero-input-crash | 陷阱：绕法通过（dummy input 后全过；崩溃形态见头部说明） | 2026-09-07 session（2026-09-10 转正重跑 PASS） |
| `TRAP-tvm-parser-rules.py` | traps-compiler.md TRAP-tvm-parser-rules | 陷阱：合法形态通过（两种 FLAG 均正确） | 2026-09-07 session（2026-09-10 转正重跑 PASS） |
| `TRAP-threads-kwarg-noop.py` | traps-compiler.md TRAP-threads-kwarg-noop | 陷阱：源码语义核对（src/ir.cc NPU 分支无 threadIdx 绑定） | 2026-09-07 源码通读（2026-09-10 转正重跑 PASS） |
| `CONST-copy-floor-method.py` | constants.md CONST-copy-floor-method | 常数：可运行部分探针数值精确（从 lerp `probe_copy.py` 抽取）；msprof 测量步骤见头部 | 2026-09-07 task（2026-09-10 抽取重跑 PASS） |
| `PATT-twophase-restructure.py` | attention.md PL-1.9-twophase | 优化点 **delta 形态**：两相位重构关键代码更改（相位分离骨架 + 单 L0C 复用 + flag=n-block 下标）+ 机制归因见解摘要；py_compile | 2026-09-09 task（2026-09-10 抽取，语法 PASS） |
| `PL-1.10-loads-first-decoupling.py` | elementwise.md PL-1.10-loads-first-decoupling | 优化点 **delta 形态**：单 staging 复用链（慢）vs per-input staging + 三输入前置装载（快）骨架 + 机制归因见解摘要；py_compile | 2026-09-10 ada_layer_norm Stage 4（任务内编写，语法 PASS） |
| `TRAP-vrsqrt-plain-precision.py` | traps-runtime.md TRAP-vrsqrt-plain-precision | 陷阱：现象存在（raw vrsqrt rel err > 1e-3）+ 绕法通过（vsqrt+vdiv < 1e-6）；NPU 可运行 | 2026-09-10 ada_layer_norm Stage 3（evolver 机械转正，py_compile PASS；任务内首次运行见 origin 工件） |
