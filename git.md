1. 查看提交状态
查看修改了哪些文件
git status

查看具体改了哪几行
git diff

查看已加入暂存区（add之后）的差异
git diff --cached

二、提交相关
2. 修改最近一次 commit（比如忘记改消息）
git commit --amend


• 如果你只是想补充文件，但不改 commit message：

git commit --amend --no-edit



🧹 三、撤销、恢复（救命用的）

这些命令是开发者的“撤销键”。

3. 从暂存区撤销 add

你 add 错东西时：

git restore --staged 文件名

4. 把某个文件恢复到最新 commit 状态（丢弃本地改动）
git restore 文件名

5. 回到之前的 commit（但保留修改）
git reset --soft HEAD~1

6. 回到之前的 commit（清理工作区，危险）
git reset --hard HEAD~1


soft = 退回但不丢内容
hard = 全部清理，像没改过一样

🗂 四、分支管理（量化项目很需要）

单分支开发很快会乱，你迟早会用分支。

7. 新建分支
git branch dev

8. 切换分支
git checkout dev


或者用推荐的新版：

git switch dev

9. 创建并切换
git checkout -b dev

🔀 五、合并、拉取（团队必备，个人项目也很好用）
10. 合并分支

你在 dev 干完活，把它合到 main：

git checkout main
git merge dev

11. 拉取远端更新（推荐）
git pull --rebase


rebase 会让你的提交历史更干净，不会出现奇怪的多余 merge commit。

🧳 六、查看历史（排查 bug、看日志、回溯用）
12. 查看提交历史
git log


更紧凑好看的版本：

git log --oneline --graph --decorate


超级好用，像一棵树一样显示分支和合并。

七、临时存储（stash：神技能，乱七八糟时最救命）

当你写到一半必须切分支，但不想 commit：

13. 临时保存
git stash

14. 恢复
git stash pop

15. 查看 stash 列表
git stash list

 八、远端管理
16. 查看远端仓库
git remote -v

17. 添加远端仓库
git remote add origin 地址