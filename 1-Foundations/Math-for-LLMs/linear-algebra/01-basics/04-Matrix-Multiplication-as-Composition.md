# 🎬 **第4集：矩阵乘法与线性变换复合**

## 📺 视频信息
- **视频标题**：Matrix multiplication as composition
- **3B1B原视频**：https://www.youtube.com/watch?v=XkY2DOUCWMU
- **B站中文字幕**：https://www.bilibili.com/video/BV1ys411472E?p=4

---

## 🎯 核心概念

### 1. **矩阵乘法 = 变换复合**
矩阵乘法对应线性变换的复合（连续应用）

### 2. **从右向左执行**
`A @ B @ v` = 先应用B，再应用A

### 3. **结合律成立**
`(A @ B) @ C = A @ (B @ C)`

---

## 🧮 数学原理

### 变换复合的数学定义
如果：
- 变换1: `y = A x`
- 变换2: `z = B y`

那么复合变换：`z = B (A x) = (B A) x`

**注意**：矩阵乘法顺序与变换应用顺序**相反**！

---

## 💻 代码实现

### 1. **基础变换复合**
```python
import numpy as np

# 定义三个基本变换
def rotation(theta):
    """旋转theta弧度"""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def scale(sx, sy):
    """缩放变换"""
    return np.array([[sx, 0],
                     [0, sy]])

def shear(k):
    """剪切变换"""
    return np.array([[1, k],
                     [0, 1]])

# 复合变换：先旋转30度，再放大2倍，最后剪切
R = rotation(np.pi/6)      # 30度旋转
S = scale(2, 2)           # 放大2倍
H = shear(0.5)            # 剪切

# 复合矩阵：注意顺序！
# 先旋转，再缩放，最后剪切 = H @ S @ R
composite = H @ S @ R

print("复合变换矩阵 H @ S @ R:")
print(np.round(composite, 4))
```

### 2. **验证结合律**
```python
def verify_associative_law():
    """验证矩阵乘法的结合律"""
    
    # 随机生成三个矩阵
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 5)
    C = np.random.randn(5, 6)
    
    # 计算两种顺序
    left_assoc = (A @ B) @ C
    right_assoc = A @ (B @ C)
    
    print("矩阵维度：")
    print(f"A: {A.shape}, B: {B.shape}, C: {C.shape}")
    print(f"(A@B)@C 形状: {left_assoc.shape}")
    print(f"A@(B@C) 形状: {right_assoc.shape}")
    print(f"\n是否相等？ {np.allclose(left_assoc, right_assoc)}")
    
    # 数值差异
    diff = np.max(np.abs(left_assoc - right_assoc))
    print(f"最大差异: {diff:.10f}")

verify_associative_law()
```

### 3. **变换复合可视化**
```python
import matplotlib.pyplot as plt

def visualize_transform_composition():
    """可视化变换复合过程"""
    
    # 创建单位正方形
    square = np.array([[0,0], [1,0], [1,1], [0,1], [0,0]])
    
    # 定义变换序列
    transforms = [
        ("旋转45°", rotation(np.pi/4)),
        ("x方向放大2倍", scale(2, 1)),
        ("剪切", shear(0.5))
    ]
    
    # 逐步应用变换
    points = square.copy()
    all_points = [points.copy()]
    transform_names = ["原始"]
    
    for name, T in transforms:
        points = points @ T.T  # 应用变换
        all_points.append(points.copy())
        transform_names.append(name)
    
    # 绘制
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (ax, pts, title) in enumerate(zip(axes, all_points, transform_names)):
        ax.plot(pts[:,0], pts[:,1], 'b-o', linewidth=2)
        ax.fill(pts[:,0], pts[:,1], alpha=0.2)
        
        # 标记顶点顺序
        for j, (x, y) in enumerate(pts[:-1]):
            ax.text(x, y, str(j), fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax.set_title(f"Step {i}: {title}", fontsize=12, fontweight='bold')
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.suptitle('变换复合过程：旋转 → 缩放 → 剪切', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# visualize_transform_composition()
```

### 4. **矩阵幂与重复变换**
```python
def matrix_powers_demo():
    """演示矩阵幂对应重复应用同一个变换"""
    
    # 创建一个旋转15度的矩阵
    R = rotation(np.pi/12)  # 15度
    
    # 计算矩阵幂
    R_powers = {
        "R^1 (15°)": R,
        "R^2 (30°)": R @ R,
        "R^3 (45°)": R @ R @ R,
        "R^6 (90°)": np.linalg.matrix_power(R, 6)
    }
    
    print("矩阵幂演示：旋转矩阵的幂")
    print("=" * 50)
    
    for name, matrix in R_powers.items():
        print(f"\n{name}:")
        print(np.round(matrix, 4))
    
    # 验证：R^6 应该等于旋转90度
    R_90 = rotation(np.pi/2)
    print(f"\n验证：R^6 ≈ 旋转90度矩阵？ {np.allclose(R_powers['R^6 (90°)'], R_90, atol=1e-10)}")
    
    # 可视化重复旋转
    points = np.array([[1, 0]])  # 初始点在x轴上
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i in range(1, 25):
        # 计算 R^i @ point
        R_power = np.linalg.matrix_power(R, i)
        new_point = (R_power @ points.T).T
        
        # 绘制点
        ax.plot(new_point[:,0], new_point[:,1], 'ro', alpha=0.6, markersize=5)
        
        # 添加标签
        if i in [1, 4, 8, 12, 16, 20, 24]:
            ax.text(new_point[0,0], new_point[0,1], f' {i}×15°', 
                   fontsize=9, ha='left', va='bottom')
    
    ax.set_title('重复应用旋转变换 (每次15°)', fontsize=14)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(0, color='gray', alpha=0.3)
    ax.axvline(0, color='gray', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 添加角度指示
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'b--', alpha=0.3)
    
    plt.show()

# matrix_powers_demo()
```

---

## 🔄 **结合律的几何解释**

### 可视化验证
```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 定义三个基本变换
def rotation(theta):
    """旋转theta弧度"""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def scale(sx, sy):
    """缩放变换"""
    return np.array([[sx, 0],
                     [0, sy]])


def shear(k):
    """剪切变换"""
    return np.array([[1, k],
                     [0, 1]])


def visualize_associative_geometry_fixed():
    """从几何角度理解结合律"""

    # 定义三个不同的变换
    T1 = shear(0.5)  # 剪切
    T2 = scale(1.5, 0.8)  # 不均匀缩放
    T3 = rotation(np.pi / 6)  # 旋转30度

    # 测试点
    point = np.array([[1, 0.5]])

    # 两种计算顺序
    composite1 = T1 @ T2
    result1 = (composite1 @ T3) @ point.T

    composite2 = T2 @ T3
    result2 = T1 @ (composite2 @ point.T)

    print("结合律几何验证：")
    print(f"T1 @ T2 @ point 维度: {T1.shape} @ {T2.shape} @ {point.shape}")
    print(f"\n方法1 (先T1T2再T3): {result1.flatten()}")
    print(f"方法2 (先T2T3再T1): {result2.flatten()}")
    print(f"是否相等？ {np.allclose(result1, result2)}")

    # 使用2x3的子图布局，但要正确处理索引
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 原始三角形
    triangle = np.array([[0, 0], [1, 0], [0.5, 1], [0, 0]])

    # 方法1的步骤（第1行）
    steps1 = [triangle]
    steps1.append(triangle @ T1.T)  # Step1: 剪切
    steps1.append(steps1[-1] @ T2.T)  # Step2: 缩放
    steps1.append(steps1[-1] @ T3.T)  # Step3: 旋转

    # 方法2的步骤（第2行）
    steps2 = [triangle]
    steps2.append(triangle @ T2.T)  # Step1: 缩放
    steps2.append(steps2[-1] @ T3.T)  # Step2: 旋转
    steps2.append(steps2[-1] @ T1.T)  # Step3: 剪切

    # 只绘制前3列（0,1,2列），第3列（索引2）留给最终结果
    titles_row1 = ["原始", "剪切后", "再缩放后"]
    titles_row2 = ["原始", "缩放后", "再旋转后"]

    # 绘制第0行（方法1的前3步）
    for i in range(3):
        ax = axes[0, i]
        ax.plot(steps1[i][:, 0], steps1[i][:, 1], 'bo-', linewidth=2)
        ax.fill(steps1[i][:, 0], steps1[i][:, 1], 'blue', alpha=0.2)
        ax.set_title(f"方法1: {titles_row1[i]}", fontsize=11)
        ax.set_xlim(-1, 3)
        ax.set_ylim(-1, 3)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    # 绘制第1行（方法2的前3步）
    for i in range(3):
        ax = axes[1, i]
        ax.plot(steps2[i][:, 0], steps2[i][:, 1], 'go-', linewidth=2)
        ax.fill(steps2[i][:, 0], steps2[i][:, 1], 'green', alpha=0.2)
        ax.set_title(f"方法2: {titles_row2[i]}", fontsize=11)
        ax.set_xlim(-1, 3)
        ax.set_ylim(-1, 3)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    # 第3列（索引2）显示最终结果
    # 左上角（方法1最终结果）
    axes[0, 2].clear()  # 清空原来的内容
    axes[0, 2].plot(steps1[3][:, 0], steps1[3][:, 1], 'ro-', linewidth=3)
    axes[0, 2].fill(steps1[3][:, 0], steps1[3][:, 1], 'red', alpha=0.2)
    axes[0, 2].set_title("方法1最终结果\n(剪切→缩放→旋转)", fontsize=11)
    axes[0, 2].set_xlim(-1, 3)
    axes[0, 2].set_ylim(-1, 3)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_aspect('equal')

    # 左下角（方法2最终结果）
    axes[1, 2].clear()  # 清空原来的内容
    axes[1, 2].plot(steps2[3][:, 0], steps2[3][:, 1], 'mo-', linewidth=3)
    axes[1, 2].fill(steps2[3][:, 0], steps2[3][:, 1], 'purple', alpha=0.2)
    axes[1, 2].set_title("方法2最终结果\n(缩放→旋转→剪切)", fontsize=11)
    axes[1, 2].set_xlim(-1, 3)
    axes[1, 2].set_ylim(-1, 3)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_aspect('equal')

    plt.suptitle('矩阵乘法结合律：不同计算顺序，相同最终结果', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# 运行
visualize_associative_geometry_fixed()
```

---

## 🎯 **关键理解**

### **重要公式**
```
变换复合：C = B @ A
应用顺序：先A，后B

结合律：(A @ B) @ C = A @ (B @ C)
几何意义：最终结果相同，计算过程可重组

矩阵幂：A^n = A @ A @ ... @ A (n次)
几何意义：重复应用同一个变换n次
```

### **记忆技巧**
```
1. "从右向左"执行变换
2. "先变换的在右边"
3. 结合律允许我们重新分组，但不能改变顺序
4. 矩阵幂 = 重复变换
```

---

## 📝 **练习与思考**

### **练习题**
1. 给定 `A = [[1,2],[3,4]]`, `B = [[0,-1],[1,0]]`, 计算：
   - `A @ B` 和 `B @ A`
   - `(A @ B) @ A` 和 `A @ (B @ A)`
   - 验证结合律

```python
import numpy as np

# 定义矩阵
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[0, -1],
              [1, 0]])

print("矩阵定义：")
print(f"A = \n{A}")
print(f"\nB = \n{B}")
print("\n" + "=" * 50)

# 1. 计算 A @ B 和 B @ A
print("\n1. 计算 A @ B 和 B @ A：")
AB = A @ B
BA = B @ A

print(f"A @ B = \n{AB}")
print(f"\nB @ A = \n{BA}")
print(f"\nA @ B == B @ A ? {np.array_equal(AB, BA)}")
print("💡 矩阵乘法不满足交换律！")

# 2. 计算 (A @ B) @ A 和 A @ (B @ A)
print("\n" + "=" * 50)
print("\n2. 计算 (A @ B) @ A 和 A @ (B @ A)：")

left_assoc = (A @ B) @ A  # 先算A@B，再乘以A
right_assoc = A @ (B @ A)  # 先算B@A，再左乘A

print(f"(A @ B) @ A = \n{left_assoc}")
print(f"\nA @ (B @ A) = \n{right_assoc}")
print(f"\n是否相等？ {np.array_equal(left_assoc, right_assoc)}")

# 3. 验证结合律
print("\n" + "=" * 50)
print("\n3. 验证结合律 (A @ B) @ C = A @ (B @ C)：")

# 创建第三个矩阵C
C = np.array([[2, 0],
              [0, 0.5]])

print(f"添加矩阵 C = \n{C}")

# 计算两种顺序
left_result = (A @ B) @ C
right_result = A @ (B @ C)

print(f"\n(A @ B) @ C = \n{left_result}")
print(f"\nA @ (B @ C) = \n{right_result}")
print(f"\n是否相等？ {np.array_equal(left_result, right_result)}")

# 4. 更多验证
print("\n" + "=" * 50)
print("\n4. 更多验证：")


# 验证结合律通用性
def verify_associative(matrices):
    """验证多个矩阵的结合律"""
    n = len(matrices)

    # 从左到右结合
    left_to_right = matrices[0]
    for i in range(1, n):
        left_to_right = left_to_right @ matrices[i]

    # 从右到左结合
    right_to_left = matrices[-1]
    for i in range(n - 2, -1, -1):
        right_to_left = matrices[i] @ right_to_left

    return left_to_right, right_to_left


# 测试三个矩阵
matrices = [A, B, C]
left_result, right_result = verify_associative(matrices)

print(f"A @ B @ C (从左到右) = \n{left_result}")
print(f"\nA @ B @ C (从右到左) = \n{right_result}")
print(f"\n是否相等？ {np.array_equal(left_result, right_result)}")

# 5. 几何解释
print("\n" + "=" * 50)
print("\n5. 几何解释：")


def explain_geometrically():
    """从几何角度解释"""

    # 看看每个变换的作用
    print("A 的作用：")
    print("A @ [1,0] =", A @ np.array([1, 0]))
    print("A @ [0,1] =", A @ np.array([0, 1]))

    print("\nB 的作用（旋转90度）：")
    print("B @ [1,0] =", B @ np.array([1, 0]))
    print("B @ [0,1] =", B @ np.array([0, 1]))

    print("\n(A@B) 的作用：先旋转90度，再应用A")
    print("(A@B) @ [1,0] =", AB @ np.array([1, 0]))

    print("\n(B@A) 的作用：先应用A，再旋转90度")
    print("(B@A) @ [1,0] =", BA @ np.array([1, 0]))

    print("\n💡 几何意义：")
    print("- A@B: 先旋转90度，再拉伸剪切")
    print("- B@A: 先拉伸剪切，再旋转90度")
    print("- 顺序不同，结果不同！")
    print("- 但结合律成立：(A@B)@A = A@(B@A)")


explain_geometrically()

# 6. 数值精度验证
print("\n" + "=" * 50)
print("\n6. 数值精度验证：")

# 使用随机矩阵验证结合律
np.random.seed(42)  # 固定随机种子
test_matrices = [np.random.randn(3, 4),
                 np.random.randn(4, 5),
                 np.random.randn(5, 6)]

left_val = (test_matrices[0] @ test_matrices[1]) @ test_matrices[2]
right_val = test_matrices[0] @ (test_matrices[1] @ test_matrices[2])

print(f"随机矩阵测试：")
print(f"矩阵维度: {test_matrices[0].shape}, {test_matrices[1].shape}, {test_matrices[2].shape}")
print(f"(M1@M2)@M3 形状: {left_val.shape}")
print(f"M1@(M2@M3) 形状: {right_val.shape}")
print(f"最大差异: {np.max(np.abs(left_val - right_val)):.10e}")
print(f"是否相等（考虑浮点误差）？ {np.allclose(left_val, right_val)}")
```

2. 创建一个旋转30度的矩阵R，计算：
   - `R^3`（应用3次旋转）
   - `R^6`（应用6次旋转）
   - 验证 `R^6` 是否等于旋转180度

```python
import numpy as np

# 1. 创建旋转30度的矩阵
theta_30 = np.pi / 6  # 30度 = π/6 弧度
R = np.array([[np.cos(theta_30), -np.sin(theta_30)],
              [np.sin(theta_30),  np.cos(theta_30)]])

print("=== 旋转矩阵幂运算 ===")
print(f"旋转30度矩阵 R:")
print(np.round(R, 4))

# 2. 计算 R^3（应用3次旋转）
R_power_3 = np.linalg.matrix_power(R, 3)  # R @ R @ R

print(f"\nR^3（旋转3次，每次30度）：")
print(np.round(R_power_3, 4))
print(f"理论值：旋转{30*3}度 = 旋转90度")

# 3. 计算 R^6（应用6次旋转）
R_power_6 = np.linalg.matrix_power(R, 6)  # R @ R @ R @ R @ R @ R

print(f"\nR^6（旋转6次，每次30度）：")
print(np.round(R_power_6, 4))
print(f"理论值：旋转{30*6}度 = 旋转180度")

# 4. 验证 R^6 是否等于旋转180度
theta_180 = np.pi  # 180度 = π 弧度
R_180 = np.array([[np.cos(theta_180), -np.sin(theta_180)],
                  [np.sin(theta_180),  np.cos(theta_180)]])

print(f"\n直接计算的旋转180度矩阵：")
print(np.round(R_180, 4))

# 比较
print(f"\n验证：R^6 ≈ 旋转180度矩阵？")
print(f"数值相等？ {np.array_equal(np.round(R_power_6, 10), np.round(R_180, 10))}")
print(f"近似相等（考虑浮点误差）？ {np.allclose(R_power_6, R_180)}")

# 5. 计算差异
diff = np.abs(R_power_6 - R_180)
print(f"\n差异矩阵：")
print(np.round(diff, 10))
print(f"最大差异：{np.max(diff):.10e}")
```
3. 证明：对于任何矩阵A，有 `A @ I = I @ A = A`
```python
import numpy as np

# 你的原始代码
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

m, n = A.shape
I_left = np.eye(m)
I_right = np.eye(n)

# 计算结果
result1 = I_left @ A  # I @ A
result2 = A @ I_right # A @ I

print("=== 单位矩阵恒等性质证明 ===")
print(f"矩阵A ({m}×{n}):")
print(A)

print(f"\n1. 计算 I_{m} @ A:")
print(result1)
print(f"  是否等于A？ {np.array_equal(result1, A)}")

print(f"\n2. 计算 A @ I_{n}:")
print(result2)
print(f"  是否等于A？ {np.array_equal(result2, A)}")

print(f"\n3. 验证 I_{m} @ A == A @ I_{n} == A:")
print(f"  I@A == A？ {np.array_equal(result1, A)}")
print(f"  A@I == A？ {np.array_equal(result2, A)}")
print(f"  I@A == A@I？ {np.array_equal(result1, result2)}")

print("\n✅ 证明完成：对于矩阵A，有 A @ I = I @ A = A")
```
### **思考题**
1. 为什么矩阵乘法要定义成"行乘列"的形式？
```
核心：为了对应线性变换的复合
矩阵的列是基向量变换后的位置，A @ B 表示：
"先用B变换基向量，再用A变换结果"
行乘列的点积正好实现这个复合运算
```
2. 从几何角度看，结合律为什么成立？
```
核心：线性变换的复合顺序不影响最终结果
(A @ B) @ C = A @ (B @ C) 意味着：
"先C再B再A" = "先(B再C)再A"
几何上：最终把空间变换到同一个位置
```
3. 在什么情况下 `(A @ B)^n = A^n @ B^n` 成立？
```
核心：当 A @ B = B @ A（矩阵可交换）
常见情况：

A和B都是对角矩阵

A和B是同一个矩阵的幂

A或B是单位矩阵的倍数

A和B代表可交换的几何变换（如同方向的缩放）
```
4. 如果两个矩阵可交换（A@B = B@A），在几何上意味着什么？
```
核心：两个变换顺序不影响结果
几何解释：

变换作用于独立的维度（如x缩放和y缩放）

变换是同类型的（如两个旋转）

一个变换是恒等变换的倍数
意味着这两个变换"互不干扰"，可以按任意顺序应用
```
---

## 🚀 **下一步学习建议**

### **立即练习：**
```python
# 动手验证
def practice_exercises():
    # 练习1
    A = np.array([[1,2],[3,4]])
    B = np.array([[0,-1],[1,0]])
    
    # 计算并比较
    AB = A @ B
    BA = B @ A
    
    print(f"A@B = \n{AB}")
    print(f"\nB@A = \n{BA}")
    print(f"\n是否相等？ {np.array_equal(AB, BA)}")
    
    # 验证结合律
    C = np.array([[2,0],[0,0.5]])
    left = (A @ B) @ C
    right = A @ (B @ C)
    print(f"\n结合律验证: {np.allclose(left, right)}")

practice_exercises()
```

### **连接应用：**
- 神经网络中的层：多个线性变换的复合
- 计算机图形学：模型变换、视图变换、投影变换的复合
- 机器人运动学：多个关节变换的复合
