# 第2集：线性组合、张成的空间与基向量

## 🎥 视频信息
- **视频标题**：线性组合、张成的空间和基向量
- **3B1B原视频**：[Linear combinations, span, and basis vectors](https://www.youtube.com/watch?v=k7RM-ot2NWY)
- **B站中文字幕**：[【官方双语】线性代数的本质 - 02集](https://www.bilibili.com/video/BV1ys411472E?p=2)

## 📚 核心概念

### 1. **基向量（Basis Vectors）**
基向量是构成坐标系的"基本构件"，就像建筑的地基。

```python
# 标准基向量（我们通常使用的坐标系）
import numpy as np

i_hat = np.array([1, 0])  # x方向的单位向量
j_hat = np.array([0, 1])  # y方向的单位向量

print(f"标准基向量:")
print(f"î = {i_hat}  (指向x正方向)")
print(f"ĵ = {j_hat}  (指向y正方向)")
```
#### 关键点：

* 基向量定义了坐标系的方向
* 任何向量都可以用基向量的组合来表示
* 标准基向量是最简单、最直观的选择

### 2. 线性组合（Linear Combination）
线性组合就是用基向量"混合"出任意向量的过程。

```python
def linear_combination(scalar1, vector1, scalar2, vector2):
    """计算两个向量的线性组合"""
    return scalar1 * vector1 + scalar2 * vector2
# 示例：用基向量表示任意向量
v = np.array([3, 2])
# v = 3*î + 2*ĵ
result = linear_combination(3, i_hat, 2, j_hat)
print(f"向量 {v} = 3*î + 2*ĵ = {result}")
```
#### 数学公式：

```text
给定向量 v₁, v₂ 和标量 a, b
线性组合：a·v₁ + b·v₂
```
### 3. 张成的空间（Span）
张成的空间是所有可能的线性组合构成的集合。

```python
def generate_span(v1, v2, scalars_range=(-2, 2), num_points=20):
    """
    生成两个向量张成的空间中的点
    """
    span_points = []
    
    # 生成所有可能的线性组合
    for a in np.linspace(scalars_range[0], scalars_range[1], num_points):
        for b in np.linspace(scalars_range[0], scalars_range[1], num_points):
            point = a * v1 + b * v2
            span_points.append(point)
    
    return np.array(span_points)
```
## 🎨 可视化理解
#### 可视化1：基向量与线性组合
```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_basis_and_combinations():
    """可视化基向量和线性组合"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 标准基向量
    i_hat = np.array([1, 0])
    j_hat = np.array([0, 1])
    
    # ========== 图1：标准基向量 ==========
    ax1 = axes[0, 0]
    ax1.quiver(0, 0, i_hat[0], i_hat[1], 
               color='red', width=0.008, scale=1,
               label='î = [1, 0]')
    ax1.quiver(0, 0, j_hat[0], j_hat[1],
               color='blue', width=0.008, scale=1,
               label='ĵ = [0, 1]')
    
    ax1.text(0.5, -0.2, 'î', fontsize=14, color='red', weight='bold')
    ax1.text(-0.2, 0.5, 'ĵ', fontsize=14, color='blue', weight='bold')
    
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_title('标准基向量 î 和 ĵ')
    ax1.legend()
    
    # ========== 图2：用基向量表示向量 ==========
    ax2 = axes[0, 1]
    
    # 要表示的向量
    target_vector = np.array([3, 2])
    
    # 表示为基向量的线性组合
    components = [
        (3, i_hat, '3*î', 'red'),
        (2, j_hat, '2*ĵ', 'blue')
    ]
    
    current_point = np.array([0.0, 0.0])
    
    for scalar, basis_vec, label, color in components:
        ax2.quiver(current_point[0], current_point[1],
                  scalar * basis_vec[0], scalar * basis_vec[1],
                  color=color, width=0.008, scale=1,
                  alpha=0.7, label=label)
        
        current_point += scalar * basis_vec
        
        if scalar != 0:
            ax2.scatter(current_point[0], current_point[1], 
                       color=color, s=30, alpha=0.5)
    
    # 绘制最终向量
    ax2.quiver(0, 0, target_vector[0], target_vector[1],
              color='green', width=0.01, scale=1,
              label=f'v = {target_vector}')
    
    ax2.text(target_vector[0]/2, target_vector[1]/2 + 0.3,
            'v = 3*î + 2*ĵ', fontsize=12, color='green',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax2.set_xlim(-1, 4)
    ax2.set_ylim(-1, 3)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_title('用基向量表示向量: v = 3*î + 2*ĵ')
    ax2.legend()
    
    # ========== 图3：不同基向量 ==========
    ax3 = axes[1, 0]
    
    # 非标准基向量
    basis1 = np.array([2, 1])
    basis2 = np.array([1, 2])
    
    ax3.quiver(0, 0, basis1[0], basis1[1],
               color='red', width=0.008, scale=1,
               label=f'b₁ = {basis1}')
    ax3.quiver(0, 0, basis2[0], basis2[1],
               color='blue', width=0.008, scale=1,
               label=f'b₂ = {basis2}')
    
    # 用新基表示同一个向量 v = [3, 2]
    # 解方程：a*b₁ + b*b₂ = [3, 2]
    a = 4/3
    b = 1/3
    
    # 绘制分量
    current = np.array([0.0, 0.0])
    
    # 第一分量
    comp1 = a * basis1
    ax3.quiver(0, 0, comp1[0], comp1[1],
              color='red', width=0.008, scale=1,
              alpha=0.5, linestyle='--')
    
    # 第二分量
    comp2 = b * basis2
    ax3.quiver(comp1[0], comp1[1], comp2[0], comp2[1],
              color='blue', width=0.008, scale=1,
              alpha=0.5, linestyle='--')
    
    # 最终向量
    ax3.quiver(0, 0, target_vector[0], target_vector[1],
              color='green', width=0.01, scale=1,
              label=f'v = {target_vector}')
    
    ax3.text(1.5, 1, f'v = ({a:.2f})*b₁ + ({b:.2f})*b₂', 
            fontsize=11, color='green',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax3.set_xlim(-1, 4)
    ax3.set_ylim(-1, 3)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.set_title('不同基向量表示同一个向量')
    ax3.legend()
    
    # ========== 图4：张成的空间 ==========
    ax4 = axes[1, 1]
    
    # 生成张成空间的点
    span_points = []
    for a in np.linspace(-2, 2, 15):
        for b in np.linspace(-2, 2, 15):
            point = a * basis1 + b * basis2
            span_points.append(point)
    
    span_points = np.array(span_points)
    
    # 绘制所有点
    ax4.scatter(span_points[:, 0], span_points[:, 1], 
               s=20, alpha=0.5, color='purple', label='张成的空间')
    
    # 绘制基向量
    ax4.quiver(0, 0, basis1[0], basis1[1],
               color='red', width=0.01, scale=1,
               label=f'b₁ = {basis1}')
    ax4.quiver(0, 0, basis2[0], basis2[1],
               color='blue', width=0.01, scale=1,
               label=f'b₂ = {basis2}')
    
    # 手动绘制平行四边形边界
    corners = [
        -2*basis1 - 2*basis2,
        -2*basis1 + 2*basis2,
        2*basis1 + 2*basis2,
        2*basis1 - 2*basis2
    ]
    
    # 绘制边界线
    for i in range(4):
        ax4.plot([corners[i][0], corners[(i+1)%4][0]],
                [corners[i][1], corners[(i+1)%4][1]],
                'g--', alpha=0.5, linewidth=2)
    
    ax4.set_xlim(-6, 6)
    ax4.set_ylim(-6, 6)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    ax4.set_title('两个向量张成的空间（平行四边形区域）')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

# 运行函数
visualize_basis_and_combinations()
```
### 可视化2：线性相关与线性无关
```python
def visualize_linear_dependence():
    """可视化线性相关与线性无关"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # ========== 情况1：线性无关 ==========
    ax1 = axes[0]
    
    v1 = np.array([2, 1])
    v2 = np.array([1, 2])
    
    # 绘制向量
    ax1.quiver(0, 0, v1[0], v1[1],
               color='red', width=0.008, scale=1,
               label=f'v₁ = {v1}')
    ax1.quiver(0, 0, v2[0], v2[1],
               color='blue', width=0.008, scale=1,
               label=f'v₂ = {v2}')
    
    # 生成张成空间的点
    points = []
    for a in np.linspace(-2, 2, 10):
        for b in np.linspace(-2, 2, 10):
            points.append(a * v1 + b * v2)
    
    points = np.array(points)
    ax1.scatter(points[:, 0], points[:, 1], s=15, alpha=0.4, color='green')
    
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-6, 6)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_title('线性无关\n张成整个二维平面')
    ax1.legend()
    
    # ========== 情况2：线性相关（共线） ==========
    ax2 = axes[1]
    
    v3 = np.array([2, 1])
    v4 = np.array([4, 2])  # v4 = 2 * v3
    
    ax2.quiver(0, 0, v3[0], v3[1],
               color='red', width=0.008, scale=1,
               label=f'v₃ = {v3}')
    ax2.quiver(0, 0, v4[0], v4[1],
               color='blue', width=0.008, scale=1,
               label=f'v₄ = {v4} = 2*v₃')
    
    # 生成点（只能沿一条线）
    points_line = []
    for t in np.linspace(-3, 3, 50):
        points_line.append(t * v3)
    
    points_line = np.array(points_line)
    ax2.scatter(points_line[:, 0], points_line[:, 1], 
                s=15, alpha=0.6, color='orange')
    
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-6, 6)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_title('线性相关（共线）\n张成一条直线')
    ax2.legend()
    
    # ========== 情况3：零向量 ==========
    ax3 = axes[2]
    
    v5 = np.array([2, 1])
    v6 = np.array([0, 0])  # 零向量
    
    ax3.quiver(0, 0, v5[0], v5[1],
               color='red', width=0.008, scale=1,
               label=f'v₅ = {v5}')
    ax3.quiver(0, 0, v6[0], v6[1],
               color='blue', width=0.008, scale=1,
               label=f'v₆ = {v6} (零向量)')
    
    # 零向量和任何向量张成都只能得到直线
    points_zero = []
    for t in np.linspace(-2, 2, 30):
        points_zero.append(t * v5)  # v6贡献为0
    
    points_zero = np.array(points_zero)
    ax3.scatter(points_zero[:, 0], points_zero[:, 1], 
                s=15, alpha=0.6, color='purple')
    
    ax3.set_xlim(-6, 6)
    ax3.set_ylim(-6, 6)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.set_title('包含零向量\n张成一条直线')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

visualize_linear_dependence()
```
## 🧮 数学公式与性质
### 1. 线性组合的数学表示
```text
给定向量 v₁, v₂, ..., vₙ 和标量 a₁, a₂, ..., aₙ
线性组合：a₁v₁ + a₂v₂ + ... + aₙvₙ
```
### 2. 张成空间的定义
```text
Span{v₁, v₂, ..., vₙ} = {所有可能的线性组合}
```
### 3. 基向量的条件
* 一组向量是空间的基，当且仅当：
* 线性无关：不能互相表示
* 张成整个空间：能表示空间中所有向量
* 最小集合：去掉任何一个都不能张成整个空间

#### 4. 维度的意义
```python
# 二维空间的基有2个向量
basis_2d = [np.array([1, 0]), np.array([0, 1])]

# 三维空间的基有3个向量  
basis_3d = [
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, 0, 1])
]

print(f"二维空间维度: {len(basis_2d)}")
print(f"三维空间维度: {len(basis_3d)}")
```
## 🔗 与LLM的联系
### 1. 词向量空间的基
```python
# 在词向量空间中：
# - 每个词向量是高维空间中的点
# - 语义相似的词在空间中靠近
# - 整个词表张成一个高维空间

# 例如在300维词向量空间中：
# 理论上需要300个线性无关的词向量作为基
# 但实际上我们使用标准正交基更简单
```
### 2. 注意力机制中的线性组合
```python
# 注意力权重本质是线性组合的系数
def attention_linear_combination(values, attention_weights):
    """
    values: 值向量 [v₁, v₂, ..., vₙ]
    attention_weights: 注意力权重 [α₁, α₂, ..., αₙ]
    输出: 加权和 = α₁v₁ + α₂v₂ + ... + αₙvₙ
    """
    # 这就是线性组合！
    return np.sum(values * attention_weights[:, np.newaxis], axis=0)

print("""
在Transformer的注意力机制中：
上下文向量 = Σ(注意力权重 × 值向量)
这本质上就是值向量的线性组合！
""")
```
## 💡 关键要点总结
### 必须掌握的概念：
1. 基向量是"坐标系"：决定如何描述空间中的点
2. 线性组合是"配方"：用基向量混合得到任意向量
3. 张成空间是"可达范围"：所有可能到达的点集合
4. 线性相关是"冗余"：有的向量不提供新方向

### 3B1B的深刻见解：
1. 线性代数是在选择基向量和在不同基之间转换
2. 同一个向量在不同基下有不同坐标
3. 好的基能让问题变得简单

## 📝 练习与思考
### 练习题
1. 给定向量 u = [1, 2], v = [3, 1]，计算：
* 2u + 3v
* -u + 0.5v
* 它们能张成整个二维平面吗？
```python
import numpy as np

# 定义向量
u = np.array([1, 2])
v = np.array([3, 1])

print("=== 向量计算题 ===")
print(f"u = {u}")
print(f"v = {v}")

# 1. 计算 2u + 3v
result1 = 2*u + 3*v
print(f"\n1. 2u + 3v = 2*{u} + 3*{v}")
print(f"   = {2*u} + {3*v}")
print(f"   = {result1}")

# 2. 计算 -u + 0.5v
result2 = -u + 0.5*v
print(f"\n2. -u + 0.5v = -{u} + 0.5*{v}")
print(f"   = {-u} + {0.5*v}")
print(f"   = {result2}")

# 3. 判断它们能张成整个二维平面吗？
print(f"\n3. u和v能张成整个二维平面吗？")

# 方法：检查是否线性无关（计算矩阵的秩）
matrix = np.column_stack((u, v))
rank = np.linalg.matrix_rank(matrix)

if rank == 2:
    print(f"   ✓ 可以！因为矩阵 [{u}, {v}] 的秩为 {rank} = 2")
    print(f"   说明 u 和 v 线性无关")
else:
    print(f"   ✗ 不可以！因为矩阵 [{u}, {v}] 的秩为 {rank} < 2")
    print(f"   说明 u 和 v 线性相关")
```
2. 判断向量组是否线性无关：
* [1, 0], [0, 1]
* [1, 2], [2, 4]
* [1, 1, 0], [0, 1, 1], [1, 0, 1]
```python
import numpy as np

# 定义向量
u = np.array([1, 1, 0])
v = np.array([0, 1, 1])
w = np.array([1, 0, 1])
print("=== 向量计算题 ===")
print(f"u = {u}")
print(f"v = {v}")
print(f"v = {w}")

print(f"\nu,v和w是否线性相关？")

# 方法：检查是否线性无关（计算矩阵的秩）
matrix = np.column_stack((u, v, w))
rank = np.linalg.matrix_rank(matrix)

if rank == 3:
    print(f"   ✓ 可以！因为矩阵 [{u}, {v}, {w}] 的秩为 {rank} = 3")
    print(f"   说明 u 和 v 线性无关")
else:
    print(f"   ✗ 不可以！因为矩阵 [{u}, {v}, {w}] 的秩为 {rank} < 3")
    print(f"   说明 u 和 v 线性相关")
```
### 思考题
1. 为什么二维空间至少需要2个向量才能作为基？
```
答案：因为二维空间需要两个独立方向才能确定所有点的位置。一个向量只能确定一条直线上的点，无法覆盖整个平面。
类比：就像在平面上定位需要经纬度两个坐标，一个坐标只能确定一条线。
```
2. 如果两个向量线性相关，它们的张成空间是什么？
```
答案：一条直线（一维空间）。
解释：线性相关意味着一个向量是另一个的倍数，它们指向同一方向，所有线性组合都落在同一条直线上。
数学表达：Span{v₁, v₂} = {t·v₁ | t∈ℝ} 或 {t·v₂ | t∈ℝ}
```
3. 在机器学习中，为什么特征需要线性无关？
```
答案：三个关键原因：
1. 避免冗余信息：相关特征重复计数，浪费计算资源
2. 防止过拟合：多重共线性导致模型不稳定，预测不可靠
3. 保证可解释性：每个特征应有独立贡献，便于分析影响
实例：用"房屋面积"和"房间数"预测房价——这两个特征高度相关，只需保留一个。
```
## 🚀 下一步学习建议
1. 运行代码：修改向量值，观察张成空间的变化
2. 动手实验：尝试三维向量的线性组合
3. 连接下一集：理解线性组合如何引出矩阵变换
4. 实际应用：思考你的数据可以用什么"基向量"表示 
