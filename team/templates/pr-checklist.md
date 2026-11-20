# Pull Request：单个算子迁移或优化

## 对应任务

- 算子名称：
- family：
- 负责人：
- GPU 源仓库：
- GPU 源 commit：

## 分支确认

- [ ] 源分支为 `op/<family>/<op>-<owner>`
- [ ] 目标分支为 `integration/npu-ops`
- [ ] 一个 PR 只包含一个算子
- [ ] 未修改其他人的进度文件

## 提交前自动检查

- [ ] 已运行检查并通过：

```powershell
python team/scripts/check_op_submission.py `
  --op-name <OpName> `
  --owner <负责人> `
  --require-pr
```

## 交付内容

- [ ] Op 实现
- [ ] Kernel 实现
- [ ] Manifest
- [ ] Workload
- [ ] 精度测试
- [ ] Benchmark
- [ ] DESIGN.md
- [ ] REVIEW.md
- [ ] integration_log.md
- [ ] run.json
- [ ] report.md
- [ ] RETROSPECTIVE.md
- [ ] 优化算子已提交 perf_records.jsonl 和 opt_log.md

## 精度结果

- 通过用例：
- 跳过用例：
- 最大误差：
- 测试环境：
  - NPU：
  - CANN：
  - PyTorch：
  - torch_npu：

## 性能结果

- baseline：
- optimized：
- 提升：
- benchmark 入口：

## 证据路径

- run.json：
- report.md：
- perf_records.jsonl：
- integration_log.md：

## 已知问题

- 无 / 具体说明：

## 安全检查

- [ ] 没有 API Key、密码或 token
- [ ] 没有 full.json、历史临时状态或原始 profiler 文件
- [ ] 没有无关算子改动
- [ ] 没有降低测试容差、增加 skip 或删除用例