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
```python
import numpy as np

def vectors_magnitude(v):
    return np.sqrt(np.sum(v**2))

v = np.array([5,-12])
print(vectors_magnitude(v))
```
* 如果 v = [2, 3], w = [-1, 4]，计算 v + w 和 3v - 2w
```python
import numpy as np

def vectors_addition(v,w):
    return v + w

def linear_combination(a, v, b, w):
    return a * v + b * w

v = np.array([2,3])
w = np.array([-1,4])

print(vectors_addition(v,w))
print(linear_combination(3,v,(-2),w))
```
* 画出向量 [1, 2] 和 [-2, 1]，计算它们的和并画图验证
```python
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义向量
v = np.array([1, 2])    # 向量 v
w = np.array([-2, 1])   # 向量 w
u = v + w               # 向量和 u = v + w = [-1, 3]

print("=== 向量计算 ===")
print(f"向量 v = {v}")
print(f"向量 w = {w}")
print(f"v + w = {v} + {w} = {u}")

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ========== 图1：分别绘制 v 和 w ==========
ax1.set_title('向量 v 和 w', fontsize=14, pad=20)

# 绘制向量 v（红色）
ax1.quiver(0, 0, v[0], v[1],
           angles='xy', scale_units='xy', scale=1,
           color='red', width=0.01,
           label=f'v = {v.tolist()}', alpha=0.8)

# 绘制向量 w（蓝色）
ax1.quiver(0, 0, w[0], w[1],
           angles='xy', scale_units='xy', scale=1,
           color='blue', width=0.01,
           label=f'w = {w.tolist()}', alpha=0.8)

# 标记终点
ax1.scatter(v[0], v[1], color='red', s=80, zorder=5)
ax1.scatter(w[0], w[1], color='blue', s=80, zorder=5)
ax1.text(v[0]+0.1, v[1]+0.1, 'v', fontsize=12, color='red', weight='bold')
ax1.text(w[0]+0.1, w[1]+0.1, 'w', fontsize=12, color='blue', weight='bold')

# 设置坐标轴
ax1.set_xlim(-3, 2)
ax1.set_ylim(-1, 4)
ax1.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
ax1.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')
ax1.legend(loc='upper right')
ax1.set_xlabel('x轴')
ax1.set_ylabel('y轴')

# ========== 图2：验证向量加法（三角形法则）==========
ax2.set_title('向量加法验证: v + w = u', fontsize=14, pad=20)

# 方法1：首尾相接（三角形法则）
# 先画 v（从原点开始）
ax2.quiver(0, 0, v[0], v[1],
           angles='xy', scale_units='xy', scale=1,
           color='red', width=0.01,
           label=f'v = {v.tolist()}', alpha=0.8)

# 再画 w（从 v 的终点开始）
ax2.quiver(v[0], v[1], w[0], w[1],
           angles='xy', scale_units='xy', scale=1,
           color='blue', width=0.01,
           label=f'w = {w.tolist()}', alpha=0.8)

# 绘制和向量 u（从原点到终点）
ax2.quiver(0, 0, u[0], u[1],
           angles='xy', scale_units='xy', scale=1,
           color='green', width=0.015,
           label=f'u = v + w = {u.tolist()}', alpha=0.8)

# 标记所有关键点
points = {
    '原点 O': (0, 0),
    'v的终点 A': (v[0], v[1]),
    'u的终点 B': (u[0], u[1])
}

for label, (x, y) in points.items():
    ax2.scatter(x, y, s=100, zorder=5)
    ax2.text(x + 0.1, y + 0.1, label, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# 连接点形成三角形
ax2.plot([0, v[0]], [0, v[1]], 'r-', alpha=0.3)      # O到A
ax2.plot([v[0], u[0]], [v[1], u[1]], 'b-', alpha=0.3)  # A到B
ax2.plot([0, u[0]], [0, u[1]], 'g--', alpha=0.5)     # O到B（对角线）

# 添加文字说明
ax2.text(v[0]/2, v[1]/2 - 0.3, 'v', fontsize=11, color='red', ha='center')
ax2.text(v[0] + w[0]/2, v[1] + w[1]/2 + 0.3, 'w', fontsize=11, color='blue', ha='center')
ax2.text(u[0]/2 + 0.5, u[1]/2, 'u = v + w', fontsize=12, color='green', weight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

# 设置坐标轴
ax2.set_xlim(-3, 2)
ax2.set_ylim(-1, 4)
ax2.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
ax2.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')
ax2.legend(loc='upper right')
ax2.set_xlabel('x轴')
ax2.set_ylabel('y轴')

plt.tight_layout()
plt.show()

# 验证计算
print("\n=== 验证向量加法 ===")
print(f"直接计算: v + w = {v} + {w} = {u}")
print(f"几何验证: 从原点(0,0)到点{u} = {u}")
print("✓ 两种方法结果一致！")

print("\n=== 向量模长计算 ===")
print(f"|v| = √({v[0]}² + {v[1]}²) = √({v[0]**2} + {v[1]**2}) = √{np.sum(v**2)} = {np.linalg.norm(v):.2f}")
print(f"|w| = √({w[0]}² + {w[1]}²) = √({w[0]**2} + {w[1]**2}) = √{np.sum(w**2)} = {np.linalg.norm(w):.2f}")
print(f"|u| = √({u[0]}² + {u[1]}²) = √({u[0]**2} + {u[1]**2}) = √{np.sum(u**2)} = {np.linalg.norm(u):.2f}")
```
### 思考题
* 为什么向量可以任意平移而不改变其本质？
  
答案：向量只有方向和大小的属性，不包含位置信息。平移后方向和大小不变，本质就不变。

例子：
```python
# 向量 [3,2] 可以在不同位置
从(0,0)到(3,2)    # 向量 [3,2]
从(1,1)到(4,3)    # (4-1, 3-1) = [3,2]  ✓ 相同
```
* 在机器学习中，为什么用向量表示数据？
```
六大原因：
1. 数学运算方便：可加、可乘、可标准化
2. 几何直观：相似数据在空间中靠近
3. 统一接口：图片、文本、音频都能向量化
4. 工具丰富：可使用线性代数所有工具
5. 降维可视化：高维数据可投影到2D/3D
6. 神经网络支持：网络层本质是向量变换
关键思想：向量是数据的"数学语言"。
```
* 如何用向量表示一张图片？
  
四层方法：

① 基础：像素向量
```python
# 灰度图8x8 → 64维向量
pixels = img.flatten()  # [p1, p2, ..., p64]
```
② 彩色：RGB拼接
```python
# 32x32 RGB → 3072维向量  
vector = np.concatenate([R.flatten(), G.flatten(), B.flatten()])
```
③ 进阶：特征向量（HOG）
```python
# 提取边缘纹理特征（3780维）
features = hog(img)  # 比原始像素更有意义
```
④ 高级：深度学习特征
```python
# CNN提取语义特征（512维）
features = cnn_model(img)  # 鲁棒性强，适合分类搜索
```
### 🚀 下一步学习建议
* 运行代码：把上面的Python代码跑一遍

* 动手修改：改变向量值，观察图形变化

* 扩展思考：想想三维向量如何可视化

* 连接现实：找找生活中哪些东西可以用向量表示
