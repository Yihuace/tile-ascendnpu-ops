# 团队成员操作说明

这份文档只说明团队成员如何加入仓库、创建算子分支、更新进度和提交 PR。

审核、测试和合并由审核员负责，开发者不需要修改 `review_status` 和 `merge_status`。

## 基本规则

- 一个算子对应一个分支
- 一个算子对应一个 PR
- 从 `integration/npu-ops` 创建开发分支
- PR 合并到 `integration/npu-ops`
- 不要直接修改 `main`
- 不要直接修改 `integration/npu-ops`

---

## 0. 云环境代理

如果云环境无法直接访问 GitHub，先配置代理。将地址和端口替换成实际值：

```powershell
git config --global http.proxy http://<代理地址>:<端口>
git config --global https.proxy http://<代理地址>:<端口>
git config --global http.version HTTP/1.1
```

已经能直接访问 GitHub 时跳过本节。

---

## 1. 第一次加入并克隆仓库

使用自己的 GitHub 账号接受仓库邀请。

配置 Git：

```powershell
git config --global user.name "你的姓名"
git config --global user.email "你的GitHub邮箱"
```

克隆仓库：

```powershell
git clone https://github.com/Op-Development/tilelang-ascend-npu-ops.git
cd tilelang-ascend-npu-ops
```

切换到集成分支：

```powershell
git fetch origin
git switch integration/npu-ops
git pull
```

确认当前分支：

```powershell
git branch --show-current
```

应该显示：

```text
integration/npu-ops
```

---

## 2. 找到自己的任务和进度文件

进度文件位于：

```text
team/progress/
```

每个人只修改自己的文件，例如：

```text
team/progress/吕盼豪.yaml
```

找到分配给自己负责的算子：

```yaml
  - op_name: BitwiseNotFwdOp
    difficulty: 简单
    dev_stage: 未开始
    review_status: 待测试
    merge_status: 未合并
    branch: ""
    pr: ""
```

开发者只需要维护：

- `dev_stage`
- `branch`
- `pr`

`review_status` 和 `merge_status` 由审核员维护。

---

## 3. 创建算子开发分支

开始每个新算子前，先更新集成分支：

```powershell
git switch integration/npu-ops
git pull
```

创建自己的分支：

```powershell
git switch -c op/<family>/<算子名>-<姓名>
```

例如：

```powershell
git switch -c op/elementwise/bitwise-not-lvpanhao
```

不要把多个算子放在同一个分支。

---

## 4. 更新开发进度

创建分支并开始迁移后，更新自己的 YAML：

```yaml
    dev_stage: 迁移中
    branch: op/elementwise/bitwise-not-lvpanhao
    pr: ""
```

迁移完成后：

```yaml
    dev_stage: 已迁移
```

优化完成后：

```yaml
    dev_stage: 已优化
```

## 开发阶段可选值

```text
未开始
迁移中
已迁移
优化中
已优化
已提交审核
已阻塞
```

提交进度修改：

```powershell
git add team/progress/你的姓名.yaml
git commit -m "chore(progress): update BitwiseNotFwdOp status"
```

---

## 5. 提交 PR

迁移和优化完成后，将进度改为：

```yaml
    dev_stage: 已提交审核
    branch: op/elementwise/bitwise-not-lvpanhao
    pr: "https://github.com/Op-Development/tilelang-ascend-npu-ops/pull/编号"
```

提交前不需要逐个选择文件。确认当前分支只做这个算子后，执行：

```powershell
git add -A
python team/scripts/check_op_submission.py `
  --op-name BitwiseNotFwdOp `
  --owner 吕盼豪
```

检查通过后再提交并推送：

```powershell
git commit -m "feat(elementwise): migrate and optimize BitwiseNotFwdOp"
git push -u origin op/elementwise/bitwise-not-lvpanhao
```

如果检查失败，不要提交或推送，按错误提示处理。

在 GitHub 创建 Pull Request：

```text
源分支：自己的算子开发分支
目标分支：integration/npu-ops
```

不要把目标分支选成 `main`。

PR 创建后，把 PR 地址填写到自己的进度文件中，并再次提交：

```powershell
git add team/progress/你的姓名.yaml
git commit -m "chore(progress): add BitwiseNotFwdOp PR link"
git push
```

PR 创建后通知审核员测试。

---

## 分支与合并方向

```text
integration/npu-ops
    ↓ 创建个人分支
op/<family>/<算子名>-<姓名>
    ↓ 提交 PR
integration/npu-ops
    ↓ 审核员最终合并
main
```

开发者只负责开发分支和 PR，不负责审核与合并。