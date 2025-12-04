# 🎬 第3集：矩阵与线性变换

## 📺 视频信息
- **视频标题**：矩阵与线性变换
- **3B1B原视频**：[Matrices as linear transformations](https://www.youtube.com/watch?v=kYB8IZa5AuE&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=3)
- **B站中文字幕**：[【官方双语】线性代数的本质 - 03集]()

---

## 🎯 核心概念

### 1. 线性变换（Linear Transformation）
线性变换是满足两个条件的变换：
- 📐 **直线保持直线**（不弯曲）
- 🎯 **原点保持固定**
```
一个函数 T: ℝ² → ℝ² 是线性变换，当且仅当满足：

可加性：T(u + v) = T(u) + T(v)

齐次性：T(cv) = cT(v)

等价条件：T 可以表示为 矩阵乘法 形式：T(v) = M @ v
```

```python
import numpy as np

def linear_transform(vector, matrix):
    """
    应用线性变换：y = A @ x
    
    参数：
        vector: 输入向量
        matrix: 变换矩阵
    
    返回：
        变换后的向量
    """
    return matrix @ vector  # 矩阵乘法
```
### 2. 💡 矩阵作为线性变换
核心思想：矩阵的列就是基向量变换后的位置！

```python
# 标准基向量
i_hat = np.array([1, 0])  # 变换前的î
j_hat = np.array([0, 1])  # 变换前的ĵ

# 一个线性变换矩阵
# 第一列：î变换后的位置
# 第二列：ĵ变换后的位置
A = np.array([[1, 2],   # î → [1, 1]
              [1, 1]])  # ĵ → [2, 1]

print("🔄 变换矩阵 A：")
print(f"第一列: {A[:, 0]} = î变换后的位置")
print(f"第二列: {A[:, 1]} = ĵ变换后的位置")

# 应用变换
v = np.array([2, 3])
v_transformed = A @ v
print(f"\n🎯 向量 {v} 变换后: {v_transformed}")
```
### 3. 🔧 常见线性变换类型
```python
def common_transformations():
    """常见线性变换矩阵"""
    
    transformations = {
        "恒等变换": np.array([[1, 0],  # î不变
                             [0, 1]]), # ĵ不变
        
        "缩放": np.array([[2, 0],   # x方向放大2倍
                         [0, 0.5]]), # y方向缩小一半
        
        "旋转90度": np.array([[0, -1],  # 逆时针旋转90度
                             [1,  0]]),
        
        "剪切": np.array([[1, 1],   # x方向增加y分量
                         [0, 1]]),  # y方向不变
        
        "投影到x轴": np.array([[1, 0],  # 保留x，丢弃y
                              [0, 0]]),
        
        "反射": np.array([[-1, 0],  # 关于y轴反射
                         [0,  1]]),
    }
    
    return transformations
```
## 🎨 可视化理解
```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_linear_transformation(matrix, title="线性变换"):
    """
    可视化线性变换对网格的影响
    """
    
    # 创建网格点
    x = np.linspace(-3, 3, 7)
    y = np.linspace(-3, 3, 7)
    X, Y = np.meshgrid(x, y)
    
    # 原始点
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # 变换后的点
    transformed = points @ matrix.T  # 每个点左乘矩阵
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # ========== 左图：变换前 ==========
    ax1.set_title('变换前', fontsize=14, pad=20)
    
    # 绘制网格线
    for i in range(len(x)):
        # 水平线
        ax1.plot(x, np.full_like(x, y[i]), 'b-', alpha=0.3, linewidth=1)
        # 垂直线
        ax1.plot(np.full_like(y, x[i]), y, 'b-', alpha=0.3, linewidth=1)
    
    # 绘制基向量
    i_hat = np.array([1, 0])
    j_hat = np.array([0, 1])
    
    ax1.quiver(0, 0, i_hat[0], i_hat[1], 
               color='red', width=0.015, scale=1,
               label='î = [1, 0]')
    ax1.quiver(0, 0, j_hat[0], j_hat[1],
               color='blue', width=0.015, scale=1,
               label='ĵ = [0, 1]')
    
    # 标记单位正方形
    square = patches.Rectangle((0, 0), 1, 1, 
                              linewidth=2, edgecolor='green', 
                              facecolor='green', alpha=0.2)
    ax1.add_patch(square)
    
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
    ax1.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)
    ax1.grid(True, alpha=0.2)
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # ========== 右图：变换后 ==========
    ax2.set_title(f'{title}\n矩阵: {matrix[0]}\n       {matrix[1]}', 
                  fontsize=14, pad=20)
    
    # 绘制变换后的网格线
    X_trans = transformed[:, 0].reshape(X.shape)
    Y_trans = transformed[:, 1].reshape(Y.shape)
    
    for i in range(len(x)):
        # 水平线（变换后）
        ax2.plot(X_trans[i, :], Y_trans[i, :], 'b-', alpha=0.3, linewidth=1)
        # 垂直线（变换后）
        ax2.plot(X_trans[:, i], Y_trans[:, i], 'b-', alpha=0.3, linewidth=1)
    
    # 绘制变换后的基向量
    i_hat_trans = matrix @ i_hat
    j_hat_trans = matrix @ j_hat
    
    ax2.quiver(0, 0, i_hat_trans[0], i_hat_trans[1],
               color='red', width=0.015, scale=1,
               label=f"î → {i_hat_trans}")
    ax2.quiver(0, 0, j_hat_trans[0], j_hat_trans[1],
               color='blue', width=0.015, scale=1,
               label=f"ĵ → {j_hat_trans}")
    
    # 绘制变换后的单位正方形
    square_corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    trans_corners = square_corners @ matrix.T
    
    polygon = patches.Polygon(trans_corners, 
                             linewidth=2, edgecolor='green',
                             facecolor='green', alpha=0.2)
    ax2.add_patch(polygon)
    
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
    ax2.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)
    ax2.grid(True, alpha=0.2)
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# 示例：可视化剪切变换
print("=== 剪切变换示例 ===")
shear_matrix = np.array([[1, 1.5],
                         [0, 1]])
fig = visualize_linear_transformation(shear_matrix, "剪切变换")
```
## 🧮 矩阵乘法与变换复合
```python
def demonstrate_transformation_composition():
    """
    演示变换的复合：先旋转，再剪切
    """
    
    # 定义两个变换
    rotation = np.array([[0, -1],  # 旋转90度
                         [1,  0]])
    
    shear = np.array([[1, 1],     # 剪切
                      [0, 1]])
    
    # 复合变换：先旋转，再剪切
    composite = shear @ rotation  # 注意顺序！
    
    print("=== 变换复合演示 ===")
    print(f"旋转矩阵 R:\n{rotation}")
    print(f"\n剪切矩阵 S:\n{shear}")
    print(f"\n复合变换 S @ R（先旋转，再剪切）:\n{composite}")
    print(f"\n注意：R @ S（先剪切，再旋转）:\n{rotation @ shear}")
    print("这两个结果不同！矩阵乘法不满足交换律。")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    vectors = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
    colors = ['red', 'blue', 'green']
    labels = ['î', 'ĵ', 'v=[1,1]']
    
    for idx, (matrix, title, ax_row) in enumerate(zip(
        [rotation, shear, composite, rotation @ shear],
        ['旋转', '剪切', '先旋转再剪切 (S@R)', '先剪切再旋转 (R@S)'],
        axes
    )):
        
        for vec, color, label in zip(vectors, colors, labels):
            # 原始向量
            ax_row.quiver(0, 0, vec[0], vec[1],
                         color=color, width=0.008, scale=1,
                         alpha=0.3, label=f'{label} (原始)')
            
            # 变换后的向量
            trans_vec = matrix @ vec
            ax_row.quiver(0, 0, trans_vec[0], trans_vec[1],
                         color=color, width=0.01, scale=1,
                         label=f'{label} → {trans_vec}')
        
        ax_row.set_xlim(-2, 2)
        ax_row.set_ylim(-2, 2)
        ax_row.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        ax_row.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)
        ax_row.grid(True, alpha=0.3)
        ax_row.set_aspect('equal')
        ax_row.set_title(title)
        ax_row.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()

# 运行演示
demonstrate_transformation_composition()
```
## 🔗 与LLM的联系
### 1. Transformer中的线性变换
```python
def transformer_linear_transformation():
    """
    Transformer中的线性变换示例
    在自注意力机制中，Q、K、V都是通过线性变换得到的
    """
    
    # 假设输入向量 x (d_model维)
    d_model = 512
    d_k = 64  # key的维度
    
    # 随机初始化权重矩阵（实际中是学习得到的）
    W_q = np.random.randn(d_model, d_k)  # Query变换矩阵
    W_k = np.random.randn(d_model, d_k)  # Key变换矩阵
    W_v = np.random.randn(d_model, d_k)  # Value变换矩阵
    
    # 输入向量（假设是词向量）
    x = np.random.randn(d_model)
    
    # 应用线性变换得到Q、K、V
    Q = W_q.T @ x  # Query向量
    K = W_k.T @ x  # Key向量
    V = W_v.T @ x  # Value向量
    
    print("=== Transformer中的线性变换 ===")
    print(f"输入向量 x 维度: {x.shape}")
    print(f"Query矩阵 W_q 形状: {W_q.shape}")
    print(f"Query向量 Q = W_q^T @ x: {Q.shape}")
    print(f"\n每个注意力头都有自己的变换矩阵！")
    print("这就是矩阵作为线性变换的实际应用。")

# transformer_linear_transformation()
```
### 2. 神经网络层就是线性变换
```python
class LinearLayer:
    """
    神经网络中的线性层
    本质就是：y = Wx + b
    """
    
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(output_dim, input_dim) * 0.01  # 权重矩阵
        self.b = np.zeros((output_dim, 1))  # 偏置向量
    
    def forward(self, x):
        """前向传播：线性变换 + 偏置"""
        return self.W @ x + self.b
    
    def __str__(self):
        return f"Linear Layer: W({self.W.shape}), b({self.b.shape})"

# 示例
layer = LinearLayer(input_dim=10, output_dim=5)
x = np.random.randn(10, 1)
y = layer.forward(x)

print(f"神经网络线性层：")
print(f"输入: {x.shape}")
print(f"权重矩阵 W: {layer.W.shape}")
print(f"输出: {y.shape}")
print(f"这就是 y = Wx + b 的线性变换！")
```
## 💡 关键要点总结
### 必须掌握的概念：
✅ 矩阵 = 线性变换的编码：每列是基向量变换后的位置

✅ 矩阵乘法 = 变换复合：AB表示先应用B，再应用A

✅ 线性变换性质：保持直线性和原点固定

✅ 几何直观：矩阵变换拉伸、旋转、剪切空间

### 3B1B的深刻见解：
🧠 矩阵不是一堆数字，而是空间的变换规则

🧠 理解矩阵就是理解基向量的去向

🧠 矩阵乘法从右向左读：先应用右边的变换

## 📝 练习与思考
### 🔢 练习题
1. 给定矩阵 A = [[2, 0], [0, 3]]，计算：

* A @ [1, 0] 和 A @ [0, 1]

* A @ [2, 3]

* 这个矩阵实现了什么变换？
```python
import numpy as np

A = np.array([[2, 0],
              [0, 3]])

print("=== 矩阵乘法练习 ===")
print(f"变换矩阵 A:\n{A}")

# 1. 计算 A @ [1, 0] 和 A @ [0, 1]
v1 = np.array([1, 0])
v1_transformed = A @ v1
print(f"\n1. A @ [1, 0] = {v1_transformed}")
print(f"   解释: x方向单位向量被拉伸2倍")

v2 = np.array([0, 1])
v2_transformed = A @ v2
print(f"\n2. A @ [0, 1] = {v2_transformed}")
print(f"   解释: y方向单位向量被拉伸3倍")

# 2. 计算 A @ [2, 3]
v3 = np.array([2, 3])
v3_transformed = A @ v3
print(f"\n3. A @ [2, 3] = {v3_transformed}")
print(f"   验证: [4, 9] = 2×[2, 0] + 3×[0, 3] = [4, 0] + [0, 9]")

# 3. 解释这个矩阵实现了什么变换
print(f"\n4. 矩阵 A 实现了什么变换？")
print(f"   A 是一个对角矩阵，实现了缩放变换：")
print(f"   - x方向缩放2倍")
print(f"   - y方向缩放3倍")
```
2. 给定变换：先旋转45度，再放大2倍。写出复合变换矩阵。
```python
import numpy as np
import math

# 1. 创建旋转45度矩阵
theta = math.radians(45)  # 45度转弧度
R = np.array([[math.cos(theta), -math.sin(theta)],
              [math.sin(theta),  math.cos(theta)]])

# 2. 创建放大2倍矩阵
S = np.array([[2, 0],
              [0, 2]])

# 3. 计算复合矩阵（先旋转再放大：S @ R）
M = S @ R

print("复合变换矩阵 M = S @ R:")
print(np.round(M, 4))
```
3. 判断以下哪些是线性变换：

* f([x, y]) = [2x, y+1]

* f([x, y]) = [x+y, x-y]

* f([x, y]) = [x², y]
```python
def is_linear_transform(func):
    """判断函数是否为线性变换"""
    
    # 测试1：原点是否保持
    if not np.allclose(func(np.array([0, 0])), [0, 0]):
        return False, "f(0,0) ≠ (0,0)"
    
    # 测试2：随机测试可加性和齐次性
    for _ in range(100):
        u = np.random.randn(2)
        v = np.random.randn(2)
        c = np.random.randn()
        
        # 可加性
        if not np.allclose(func(u + v), func(u) + func(v)):
            return False, "不满足可加性"
        
        # 齐次性  
        if not np.allclose(func(c * u), c * func(u)):
            return False, "不满足齐次性"
    
    return True, "是线性变换"

# 定义三个函数
def f1(v):
    x, y = v
    return np.array([2*x, y+1])

def f2(v):
    x, y = v
    return np.array([x+y, x-y])

def f3(v):
    x, y = v
    return np.array([x**2, y])

# 测试
print("自动判断结果：")
for i, (name, func) in enumerate([("f1", f1), ("f2", f2), ("f3", f3)], 1):
    is_linear, reason = is_linear_transform(func)
    symbol = "✅" if is_linear else "❌"
    print(f"{symbol} {name}: {reason}")
```
## 🤔 思考题
1. 为什么矩阵乘法不满足交换律？从几何角度解释。
```
几何解释：线性变换的顺序决定最终效果。

先旋转再缩放 ≠ 先缩放再旋转

就像先穿衬衫再穿外套 ≠ 先穿外套再穿衬衫

每个变换都在前一个变换的结果空间中进行
```
2. 单位矩阵为什么对应恒等变换？
```
定义：对角线为1，其余为0的矩阵

几何意义：每个基向量保持不变
I @ [1,0] = [1,0]，I @ [0,1] = [0,1]

作用：就像乘法中的1，任何矩阵与单位矩阵相乘都保持不变
```
3. 在机器学习中，为什么权重矩阵可以看作线性变换？
```
前向传播：输出 = 权重矩阵 @ 输入 + 偏置

权重矩阵的作用：将输入空间映射到输出空间
例如：512维词向量 → 64维注意力向量

学习过程：调整矩阵参数就是学习最佳的变换规则

多层级联：多个线性变换+激活函数 = 复杂非线性映射

核心思想：权重矩阵定义了如何转换和组合特征，是学习到的"特征变换规则"。
```
## 🚀 下一步学习建议
▶️ 运行代码：修改矩阵值，观察变换效果

🔧 动手实验：创建自己的变换矩阵

📖 连接下一集：理解矩阵乘法为什么是那样定义的

🤖 实际应用：思考神经网络中的线性变换
