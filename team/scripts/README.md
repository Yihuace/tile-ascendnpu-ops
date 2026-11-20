# 提交前检查脚本

## 使用方式

在算子开发分支上运行：

```powershell
python team/scripts/check_op_submission.py `
  --op-name 算子名 `
  --owner 名字
```

审核员-提交 PR 后，审核员或开发者可以加 `--require-pr`：

```powershell
python team/scripts/check_op_submission.py `
  --op-name BitwiseNotFwdOp `
  --owner 姓名 `
  --require-pr
```

如果算子包含优化版本：

```powershell
python team/scripts/check_op_submission.py `
  --op-name BitwiseNotFwdOp `
  --owner 姓名 `
  --optimized `
  --require-pr
```

## 检查内容

- 当前分支是否为 `op/` 算子分支
- 是否修改了允许范围之外的文件
- 是否提交临时状态、profiler 原始数据或缓存
- 是否包含疑似密钥或密码
- 是否包含大型文件
- 是否包含 Op、Kernel、Manifest、Workload、测试和 Benchmark
- 是否包含设计、复审、集成和验收证据
- 是否更新了自己的进度文件
- `dev_stage`、`branch`、`pr` 是否填写正确
- 优化算子是否包含 `perf_records.jsonl` 和 `opt_log.md`

检查失败时，脚本退出码为 1，不提交、不推送、不合并。