Day 1: Git Bash 基础命令练习记录
📅 学习日期
2025-12-01

✅ 今日任务完成情况
安装并配置 Git Bash

掌握 pwd 命令

掌握 ls 命令

掌握 mkdir 命令

掌握 rm 命令

完成所有练习并截图

🧪 练习记录
练习1: 查看当前目录
bash
$ pwd
/d/01_Projects/python/LLM__Learner/28day-llm
学习要点：

pwd = print working directory

显示当前所在位置的完整路径

练习2: 查看目录内容
bash
$ ls
code  docs  models  week1  week2  week3  week4

$ ls -la
total 28
drwxr-xr-x 1 Yosemite 197609    0 Dec  1 10:30 .
drwxr-xr-x 1 Yosemite 197609    0 Dec  1 10:20 ..
drwxr-xr-x 1 Yosemite 197609    0 Dec  1 10:25 code
drwxr-xr-x 1 Yosemite 197609    0 Dec  1 10:25 docs
drwxr-xr-x 1 Yosemite 197609    0 Dec  1 10:25 models
drwxr-xr-x 1 Yosemite 197609    0 Dec  1 10:30 week1
drwxr-xr-x 1 Yosemite 197609    0 Dec  1 10:25 week2
drwxr-xr-x 1 Yosemite 197609    0 Dec  1 10:25 week3
drwxr-xr-x 1 Yosemite 197609    0 Dec  1 10:25 week4
学习要点：

ls = list directory contents

ls -la 显示详细信息：

-l：长格式显示

-a：显示所有文件（包括隐藏文件）

练习3: 创建目录和文件
bash
$ mkdir day1-practice
$ cd day1-practice
$ pwd
/d/01_Projects/python/LLM__Learner/28day-llm/day1-practice

$ touch hello.txt test.py README.md
$ ls
hello.txt  README.md  test.py
学习要点：

mkdir = make directory

cd = change directory

touch = 创建空文件或更新文件时间戳

练习4: 删除文件和目录
bash
$ rm hello.txt
$ ls
README.md  test.py

$ cd ..
$ rm -r day1-practice
$ ls
code  docs  models  week1  week2  week3  week4
学习要点：

rm = remove

-r = recursive（递归删除目录）

警告：rm 删除的文件无法恢复！

📊 命令速查表
命令	用途	常用选项	示例
pwd	显示当前目录	无	pwd
ls	列出目录内容	-l 详细, -a 全部	ls -la
mkdir	创建目录	-p 创建多级目录	mkdir new_folder
rm	删除文件/目录	-r 递归, -f 强制	rm -r folder
cd	切换目录	无	cd ..
touch	创建文件	无	touch file.txt
🎯 核心掌握
绝对路径 vs 相对路径
bash
# 绝对路径（从根目录开始）
cd /d/01_Projects/python

# 相对路径（从当前位置开始）
cd ../..          # 上两级目录
cd ./week1        # 当前目录下的week1
特殊目录符号
. 当前目录

.. 上级目录

~ 用户主目录

/ 根目录

安全使用 rm
bash
# 危险！误删可能无法恢复
rm -rf /           # 绝对不要执行！

# 安全习惯
rm -i file.txt     # 删除前询问
ls before deleting # 先查看再删除
📸 学习证明截图
截图1: 基础命令验证
<img width="1138" height="394" alt="day1-screenshot png" src="https://github.com/user-attachments/assets/9da22533-440f-45f0-a06c-803505a76d9b" />
💡 实用技巧
1. Tab 键自动补全
bash
# 输入部分命令后按 Tab
cd we[Tab]        # 自动补全为 cd week1
ls -[Tab][Tab]    # 显示所有可用选项
2. 命令历史
bash
history           # 查看所有历史命令
!10               # 执行历史中第10条命令
!!                # 执行上一条命令
3. 清除屏幕
bash
clear             # 或按 Ctrl + L
4. 查看命令帮助
bash
ls --help         # 查看命令帮助
man ls            # 查看手册页
🧠 学习心得
掌握的技能
环境熟悉：适应了 Git Bash 的命令行界面

路径理解：理解了 Windows 路径和 Linux 路径的映射关系

基本操作：能够完成文件和目录的基本管理

安全意识：认识到 rm 命令的危险性，养成先检查再操作的习惯

遇到的困难
起初对相对路径和绝对路径有些混淆

rm -r 删除目录时需要加 -r 参数

Windows 和 Linux 路径格式不同需要适应

解决方案
通过多次练习 cd 命令加深理解

记住：删除目录要加 -r，删除文件不需要

使用 pwd 随时确认当前位置

📚 扩展学习
下一步学习计划
明天：Python 虚拟环境配置

后天：Git 版本控制基础

本周内：PyTorch 基础

推荐练习
bash
# 挑战练习1：创建复杂目录结构
mkdir -p project/{src,docs,tests}
cd project
tree

# 挑战练习2：批量操作
touch file{1..10}.txt
ls *.txt
rm file*.txt

# 挑战练习3：查看命令类型
type pwd
which ls
🎉 今日总结
完成状态
✅ 所有基础命令掌握
✅ 练习任务完成
✅ 学习笔记整理
✅ 学习证明截图

明日预告
Day 2：Python 虚拟环境配置

安装 Python 和 pip

创建虚拟环境

安装 numpy 并进行基础练习

📌 提交要求

保存本文件为 day1-summary.md

将命令输出截图保存为 day1-screenshot.png

提交到 Git 仓库
