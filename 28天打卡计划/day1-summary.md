# Day 1：Git Bash 基础命令学习

## 学习日期
2025-12-01

## 今日任务完成情况
* [x] 安装并配置 Git Bash
* [x] 掌握 `pwd` 命令
* [x] 掌握 `ls` 命令
* [x] 掌握 `mkdir` 命令
* [x] 掌握 `rm` 命令
* [x] 完成所有练习并截图

---

## 练习记录

### 练习1：查看当前目录
<img width="1128" height="101" alt="image" src="https://github.com/user-attachments/assets/1afe5382-396d-4625-9006-386aa79d67fd" />

### 学习要点：
 * `pwd` = print working directory
 * 显示当前所在位置的完整路径

### 练习2：查看目录内容
<img width="1106" height="462" alt="image" src="https://github.com/user-attachments/assets/9686463b-dc14-4f62-9b30-cb8bfe658af1" />

### 学习要点：
 * `ls` = list directory contents
 * `ls -la` 显示详细信息：
   * `-l`：长格式显示
   * `-a`：显示所有文件（包括隐藏文件）`.`和`..`
  
### 练习3：创建目录和文件
<img width="1317" height="517" alt="image" src="https://github.com/user-attachments/assets/623ade81-da92-4b66-9ce9-097aa0d1b66c" />

### 学习要点：
* `mkdir` = make directory
* `cd` = change directory
* `touch` = 创建空文件或更新文件时间戳。

### 练习4：删除文件和目录
<img width="1148" height="493" alt="image" src="https://github.com/user-attachments/assets/d91834e3-f3bc-40a0-88c0-79ac2d26e230" />

### 学习要点：
* `rm` = remove
* `-r` = recursive （递归删除目录，必须用于删除目录）
* 警告：rm 删除的文件通常无法恢复！

---

## 命令速查表

| 命令 | 用途 | 常用选项 | 示例 |
| :--- | :--- | :--- | :--- |
| `pwd` | 显示当前目录 | 无 | `pwd` |
| `ls` | 列出目录内容 | `-l` 详细, `-a` 全部 | `ls -la` |
| `mkdir` | 创建目录 | `-p` 创建多级目录 | `mkdir -p new_folder/sub` |
| `rm` | 删除文件/目录 | `-r` 递归, `-f` 强制 | `rm -r folder` |
| `cd` | 切换目录 | 无 | `cd ..` |
| `touch` | 创建文件 | 无 | `touch file.txt` |

## 核心掌握

### 绝对路径 vs 相对路径
```bash
# 绝对路径（从根目录开始）
cd /d/01_Projects/python

# 相对路径（从当前位置开始）
cd ../..          # 上两级目录
cd ./week1        # 当前目录下的week1
```

### 特殊目录符号
* `.` 当前目录
* `..` 上级目录
* `~` 用户主目录
* `/` 根目录

### 安全使用rm
```bash
# 危险！误删可能无法恢复
rm -rf /           # 绝对不要执行！

# 安全习惯
rm -i file.txt     # 删除前询问
ls before deleting # 先查看再删除
```

## 实用技巧
2. 命令历史
```bash
history           # 查看所有历史命令
!10               # 执行历史中第10条命令
!!                # 执行上一条命令
```
3. 清除屏幕
```bash
clear             # 或按 Ctrl + L
```
4. 查看命令帮助
```bash
ls --help         # 查看命令帮助
man ls            # 查看手册页
```
