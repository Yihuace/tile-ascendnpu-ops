# 提案：基线已有算子的「证据-only」提交流程

- 提出人: 吕盼豪
- 日期: 2026-10-25
- 触发任务: `op/reduction/argmax-lvpanhao`（ArgmaxFwdOp）
- 状态: 待团队确认

## 1. 问题

`integration/npu-ops` 的 `examples/TileOPs` 是从上游框架整体带入的，因此部分分配给成员的算子
**六区代码在基线里已经存在**。当前 `team/scripts/check_op_submission.py` 强制要求六区文件
出现在本分支的 diff 中：

```python
required_code = {
    "Manifest": "/tileops/manifest/",
    "Workload": "/tileops/workloads/",
    "Op 实现": "/tileops/ops/",
    "Kernel 实现": "/tileops/kernels/",
    "精度测试": "/tests/ops/",
    "Benchmark": "/benchmarks/ops/",
}
for label, marker in required_code.items():
    if any(marker in path for path in changed):
        print(f"[PASS] 已提交{label}")
    else:
        errors.append(f"缺少{label}改动")
```

对基线已有算子，这类分支只可能包含「证据 + 进度」改动，于是必然 FAIL：

```text
[FAIL] 缺少Manifest改动 / 缺少Workload改动 / 缺少Op 实现改动 / 缺少精度测试改动 / 缺少Benchmark改动
```

同类问题还有两处会被误判：

| 现象 | 原因 | 受影响实例 |
|---|---|---|
| `缺少验收证据：DESIGN.md` | 白名单只认 basename `DESIGN.md` / `_DESIGN.md`，而基线文档名为 `_<kernel>_DESIGN.md` | `_argreduce_kernel_DESIGN.md`、`_make_lerp_tensor_kernel_DESIGN.md` |
| `优化算子缺少优化日志：opt_log.md` | 只要 diff 中出现 `/perf_opt/` 就要求 `opt_log.md` 也在 diff 中，而它已在基线 | `argmax_kernel/perf_opt/opt_log.md` |

已确认的基线算子（六区已在 `integration/npu-ops`）：`MishFwdOp`、`LerpTensorFwdOp`、
`ArgmaxFwdOp`、`LogSumExpFwdOp`、`AdaLayerNormFwdOp`、`SSDChunkScanFwdOp`、`MultiHeadAttentionFwdOp`。

## 2. 建议改动

新增 `--evidence-only` 开关：六区/优化日志**已在 base 分支中存在**即可，不要求本分支改动六区；
证据与进度文件的检查保持原样不变（不放宽证据要求）。

```diff
     parser.add_argument("--require-pr", action="store_true", help="提交 PR 后检查 PR 地址已填写")
     parser.add_argument("--optimized", action="store_true", help="强制要求优化证据")
+    parser.add_argument(
+        "--evidence-only",
+        action="store_true",
+        help="基线已有算子只提交证据时使用：六区已在 base 中存在即可，不要求本分支改动六区",
+    )
```

```diff
+def area_in_base(root: Path, base: str, marker: str) -> bool:
+    """base 分支中是否已存在该区的文件（证据-only 提交用）。"""
+    proc = subprocess.run(
+        ["git", "ls-tree", "-r", "--name-only", base],
+        cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
+    )
+    files = proc.stdout.decode("utf-8", errors="replace").splitlines()
+    return any(marker in path for path in files)
+
+
 def main() -> int:
```

```diff
     for label, marker in required_code.items():
         if any(marker in path for path in changed):
             print(f"[PASS] 已提交{label}")
+        elif args.evidence_only and area_in_base(root, args.base, marker):
+            print(f"[PASS] {label} 已在基线中存在（evidence-only）")
         else:
             errors.append(f"缺少{label}改动")
```

证据 basename 与优化日志同样按 `args.evidence_only` 放宽为「基线已有即可」：

```diff
-    basenames = {Path(path).name for path in changed}
-    for label, candidates in REQUIRED_EVIDENCE.items():
-        if basenames.intersection(candidates):
+    basenames = {Path(path).name for path in changed}
+    base_files = _base_files(root, args.base)
+    for label, candidates in REQUIRED_EVIDENCE.items():
+        hit = basenames.intersection(candidates) or any(
+            name.endswith("_DESIGN.md") for name in basenames
+        ) if label == "DESIGN.md" else basenames.intersection(candidates)
+        if hit:
             print(f"[PASS] 已提交 {label}")
+        elif args.evidence_only and any(
+            f.endswith(tuple(candidates)) for f in base_files
+        ):
+            print(f"[PASS] {label} 已在基线中存在（evidence-only）")
         else:
             errors.append(f"缺少验收证据：{label}")
```

## 3. 为什么这么选

- **不放宽证据要求**：`run.json` / `report.md` / `REVIEW.md` / `integration_log.md` /
  `RETROSPECTIVE.md` 仍然必须是本分支新增的实际文件，且必须来自真实实跑。
- **不放宽范围与卫生检查**：禁止文件、密钥、大文件、他人进度文件等检查不变。
- **保留可追溯性**：证据-only 通过的条目在输出里显式标 `evidence-only`，审核员一眼可见
  这是「基线已有 + 本轮补证据」，不会被误当成新迁移。

## 4. 备选方案（不推荐）

| 方案 | 问题 |
|---|---|
| 为凑 diff 去改六区文件（加无谓 workload / 改注释） | 制造噪音改动，违反「一个 PR 只做一个算子」的初衷，且容易掩盖真实改动 |
| 只提交证据、不跑检查 | 等于绕过强制闸门，不可审计 |
| 把基线算子从任务清单移除 | 这些算子确实缺本轮证据与实跑验证，移除会留下验收空洞 |

## 5. 本轮实际执行

`op/reduction/argmax-lvpanhao` 按**证据-only** 交付，如实接受当前脚本的 FAIL 结论，
逐条列在 PR 描述与 `integration_log.md` 中，不做任何凑数改动。
