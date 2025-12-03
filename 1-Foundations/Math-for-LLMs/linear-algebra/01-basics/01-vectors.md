# 第1集：向量是什么

## 🎥 视频信息
- **视频标题**：向量究竟是什么？
- **3B1B原视频**：[What is a vector?](https://www.youtube.com/watch?v=fNk_zzaMoSs)
- **B站中文字幕**：[【官方双语/合集】线性代数的本质 - 01 - 向量究竟是什么？](https://www.bilibili.com/video/BV1ys411472E?p=1)

## 📚 核心观点

# 1. 向量的两种视角

两种表示本质是相同的
向量 v = [3, 2]

### 视角1：几何观点 - 空间中的箭头
从原点 (0,0) 指向点 (3,2)

### 视角2：代数观点 - 有序的数字列表
第一个数字：x方向的变化量
第二个数字：y方向的变化量

# 2. 向量的基本要素
方向：箭头指向哪里

大小（模长）：箭头的长度

位置无关性：向量可以在空间中任意平移，只要方向大小不变，就是同一个向量

## 🎨 可视化理解
代码实现：向量可视化
# 第1集完整代码示例
```python
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ========== 图1：基本向量表示 ==========
ax1.set_title('向量的几何表示', fontsize=14, pad=20)

# 定义几个向量
vectors = {
    'v₁': np.array([3, 2]),
    'v₂': np.array([-2, 2]),
    'v₃': np.array([1, -1]),
    'v₄': np.array([-1, -1.5])
}

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

# 绘制每个向量
for (label, vec), color in zip(vectors.items(), colors):
    ax1.quiver(0, 0, vec[0], vec[1], 
               angles='xy', scale_units='xy', scale=1,
               color=color, width=0.008,
               label=f'{label} = [{vec[0]}, {vec[1]}]',
               alpha=0.8)
    
    # 标记终点
    ax1.scatter(vec[0], vec[1], color=color, s=50, zorder=5)
    ax1.text(vec[0]*1.05, vec[1]*1.05, label, 
             fontsize=11, color=color, weight='bold')

# 设置坐标轴
ax1.set_xlim(-4, 4)
ax1.set_ylim(-3, 3)
ax1.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
ax1.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
ax1.grid(True, alpha=0.2)
ax1.set_aspect('equal')
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xlabel('x轴', fontsize=12)
ax1.set_ylabel('y轴', fontsize=12)

# ========== 图2：位置无关性 ==========
ax2.set_title('向量的位置无关性', fontsize=14, pad=20)

# 同一个向量在不同位置
vector = np.array([2, 1])
start_points = [
    np.array([0, 0]),
    np.array([1, 0.5]),
    np.array([-1, 1]),
    np.array([-0.5, -1])
]

colors2 = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD166']

for i, start in enumerate(start_points):
    end = start + vector
    color = colors2[i]
    
    # 绘制向量
    ax2.quiver(start[0], start[1], vector[0], vector[1],
               angles='xy', scale_units='xy', scale=1,
               color=color, width=0.008,
               label=f'v = [2, 1]' if i == 0 else "",
               alpha=0.8)
    
    # 标记起点和终点
    ax2.scatter(start[0], start[1], color=color, s=80, 
                marker='o', alpha=0.6, zorder=5)
    ax2.scatter(end[0], end[1], color=color, s=80,
                marker='s', alpha=0.6, zorder=5)
    
    # 添加连接线
    ax2.plot([start[0], end[0]], [start[1], end[1]], 
             color=color, alpha=0.3, linestyle='--')

# 设置坐标轴
ax2.set_xlim(-3, 4)
ax2.set_ylim(-2, 3)
ax2.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
ax2.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
ax2.grid(True, alpha=0.2)
ax2.set_aspect('equal')
ax2.legend(loc='upper right', fontsize=10)
ax2.set_xlabel('x轴', fontsize=12)
ax2.set_ylabel('y轴', fontsize=12)

plt.tight_layout()
plt.show()
```
运行上述代码，你会看到：
左图：不同方向和大小的向量

右图：同一个向量在不同位置 - 方向大小相同就是同一个向量

## 🔢 向量的数学运算
### 1. 向量加法
```python
def vector_addition(v1, v2):
    """向量加法：首尾相接"""
    return v1 + v2

# 示例
v = np.array([3, 2])
w = np.array([1, -1])
result = vector_addition(v, w)
print(f"v + w = {v} + {w} = {result}")
```
### 2. 向量数乘
```python
def scalar_multiplication(scalar, vector):
    """向量数乘：缩放向量"""
    return scalar * vector

# 示例
v = np.array([2, 1])
print(f"2v = 2 * {v} = {scalar_multiplication(2, v)}")
print(f"-0.5v = -0.5 * {v} = {scalar_multiplication(-0.5, v)}")
print(f"0v = 0 * {v} = {scalar_multiplication(0, v)}  # 零向量")
```
### 3. 向量模长（大小）
```python
def vector_magnitude(v):
    """计算向量的模长（长度）"""
    return np.sqrt(np.sum(v**2))

# 示例
v = np.array([3, 4])
magnitude = vector_magnitude(v)
print(f"向量 {v} 的模长 = {magnitude:.2f}")
print(f"验证：√(3² + 4²) = √{3**2 + 4**2} = {magnitude}")
```
## 🧮 与LLM的联系
### 1. 词向量（Word Embeddings）
```python
# 在LLM中，每个词被表示为高维向量
# 例如在300维空间中：
word_vectors = {
    "king": np.random.randn(300),    # 随机初始化，实际是学习得到的
    "queen": np.random.randn(300),
    "man": np.random.randn(300),
    "woman": np.random.randn(300)
}

print("在词向量空间中：")
print("- king, queen, man, woman 都是300维向量")
print("- 语义关系可以通过向量运算捕获")
print("- 例如：king - man + woman ≈ queen")
```
### 2. 向量运算示例
```python
# 模拟著名的词向量关系
def analogy(a, b, c, word_vectors):
    """解决类比问题：a is to b as c is to ?"""
    # 关键思想：向量偏移在语义空间中保持一致
    return word_vectors[b] - word_vectors[a] + word_vectors[c]

# 注意：实际中需要训练好的词向量
print("""
在训练好的词向量中：
king - man + woman ≈ queen
Paris - France + Germany ≈ Berlin
等向量运算可以揭示语义关系
""")
```
## 💡 关键要点总结
### 必须掌握的概念
* 向量是箭头也是数组：两种视角都很重要

* 向量的核心属性：方向、大小、位置无关性

* 向量运算：加法（三角形法则）、数乘（缩放）

* 向量模长：√(x² + y²) 在二维，√(x² + y² + z²) 在三维

### 3B1B的深刻见解
* 向量是数学与几何的桥梁

* 线性代数是关于向量和向量变换的学科

* 理解向量是理解高维空间的基础

## 📝 练习与思考
### 练习题
* 计算向量 [5, -12] 的模长

* 如果 v = [2, 3], w = [-1, 4]，计算 v + w 和 3v - 2w

* 画出向量 [1, 2] 和 [-2, 1]，计算它们的和并画图验证

### 思考题
* 为什么向量可以任意平移而不改变其本质？

* 在机器学习中，为什么用向量表示数据？

如何用向量表示一张图片？

### 🚀 下一步学习建议
* 运行代码：把上面的Python代码跑一遍

* 动手修改：改变向量值，观察图形变化

* 扩展思考：想想三维向量如何可视化

* 连接现实：找找生活中哪些东西可以用向量表示
