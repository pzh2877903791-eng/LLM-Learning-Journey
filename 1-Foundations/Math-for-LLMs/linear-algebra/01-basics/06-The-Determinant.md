# 🎬 **第6集：行列式**

## 📺 视频信息
- **视频标题**：The determinant
- **3B1B原视频**：https://www.youtube.com/watch?v=Ip3X9LOh2dk
- **B站中文字幕**：https://www.bilibili.com/video/BV1ys411472E?p=6

---

## 🎯 核心概念

### 1. **行列式的几何意义**
- 衡量线性变换对**面积（2D）**或**体积（3D）**的缩放因子
- 保持方向（正号）或翻转方向（负号）
- 值为0表示降维（面积/体积压缩为0）

### 2. **行列式的性质**
- `det(I) = 1`（单位矩阵不改变面积/体积）
- `det(AB) = det(A) × det(B)`（复合变换的缩放因子相乘）
- `det(A⁻¹) = 1/det(A)`（逆变换反向缩放）

### 3. **计算行列式**
- 2×2矩阵：`det([[a,b],[c,d]]) = ad - bc`
- 3×3矩阵：使用Sarrus法则或余子式展开
- n×n矩阵：递归定义

---

## 💻 代码实现

### 1. **计算行列式**
```python
import numpy as np

def demonstrate_determinant_2d():
    """2D行列式的几何意义"""
    
    # 几个2D变换矩阵
    matrices = {
        "单位矩阵": np.eye(2),
        "缩放2倍": np.array([[2, 0], [0, 2]]),
        "不均匀缩放": np.array([[2, 0], [0, 0.5]]),
        "旋转90度": np.array([[0, -1], [1, 0]]),
        "剪切": np.array([[1, 1], [0, 1]]),
        "投影到x轴": np.array([[1, 0], [0, 0]]),
        "翻转": np.array([[-1, 0], [0, 1]])
    }
    
    print("=== 2D矩阵的行列式 ===")
    print("矩阵名称 | 矩阵 | 行列式 | 几何解释")
    print("-" * 60)
    
    for name, M in matrices.items():
        det = np.linalg.det(M)
        area_scale = abs(det)
        direction = "保持方向" if det > 0 else "翻转方向" if det < 0 else "降维"
        
        print(f"{name:10} | {M[0]} | {det:7.2f} | 面积缩放{area_scale:.1f}倍，{direction}")
        print(f"{' ':10} | {M[1]} |")
    
    return matrices

matrices_2d = demonstrate_determinant_2d()
```
### 2. **可视化行列式的几何意义**
```python
import matplotlib.pyplot as plt

def visualize_determinant_effect():
    """可视化行列式对面积的影响"""
    
    # 创建单位正方形
    square = np.array([[0,0], [1,0], [1,1], [0,1], [0,0]])
    
    # 定义几个变换
    transformations = [
        ("原始", np.eye(2), 1.0),
        ("缩放2倍", np.array([[2,0],[0,2]]), 4.0),
        ("剪切", np.array([[1,0.5],[0,1]]), 1.0),
        ("不均匀缩放", np.array([[2,0],[0,0.5]]), 1.0),
        ("旋转45°", np.array([[0.707,-0.707],[0.707,0.707]]), 1.0),
        ("面积减半", np.array([[0.707,0.707],[-0.354,0.354]]), 0.5)
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (title, M, expected_det) in enumerate(transformations):
        ax = axes[i]
        
        # 计算变换后的正方形
        transformed = square @ M.T
        
        # 计算实际行列式
        actual_det = np.linalg.det(M)
        
        # 绘制原始和变换后的形状
        ax.plot(square[:,0], square[:,1], 'b-', alpha=0.5, linewidth=1, label='原始')
        ax.plot(transformed[:,0], transformed[:,1], 'r-', linewidth=2, label='变换后')
        
        # 填充面积
        ax.fill(square[:,0], square[:,1], 'blue', alpha=0.1)
        ax.fill(transformed[:,0], transformed[:,1], 'red', alpha=0.2)
        
        # 计算面积（用于验证）
        def polygon_area(points):
            """计算多边形面积"""
            x, y = points[:,0], points[:,1]
            return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        original_area = polygon_area(square[:-1])
        new_area = polygon_area(transformed[:-1])
        area_ratio = new_area / original_area
        
        ax.set_title(f"{title}\ndet={actual_det:.3f}, 面积比={area_ratio:.3f}", fontsize=11)
        ax.set_xlim(-1, 3)
        ax.set_ylim(-1, 3)
        ax.axhline(0, color='gray', alpha=0.3)
        ax.axvline(0, color='gray', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
    
    plt.suptitle('行列式：衡量面积缩放因子', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# visualize_determinant_effect()
```
### 3. **3D行列式与体积**
```python
def demonstrate_3d_determinant():
    """3D行列式的几何意义"""
    
    # 3D变换矩阵
    matrices_3d = {
        "单位矩阵": np.eye(3),
        "均匀缩放2倍": np.diag([2, 2, 2]),
        "不均匀缩放": np.diag([2, 1, 0.5]),
        "绕z轴旋转": np.array([[0.866, -0.5, 0],
                               [0.5, 0.866, 0],
                               [0, 0, 1]]),
        "体积减半": np.array([[0.5, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]]),
        "降维（秩2）": np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]])
    }
    
    print("=== 3D矩阵的行列式 ===")
    print("矩阵名称 | 行列式 | 几何解释")
    print("-" * 50)
    
    for name, M in matrices_3d.items():
        det = np.linalg.det(M)
        
        if abs(det) < 1e-10:
            geometry = "体积压缩为0（降维）"
        elif det > 0:
            geometry = f"体积缩放{abs(det):.2f}倍，保持方向"
        else:
            geometry = f"体积缩放{abs(det):.2f}倍，翻转方向"
        
        print(f"{name:12} | {det:7.3f} | {geometry}")
    
    return matrices_3d

matrices_3d = demonstrate_3d_determinant()
```
### 4. **行列式性质验证**
```python
def verify_determinant_properties():
    """验证行列式的性质"""
    
    print("=== 行列式性质验证 ===")
    
    # 随机生成两个矩阵
    np.random.seed(42)
    A = np.random.randn(3, 3)
    B = np.random.randn(3, 3)
    
    det_A = np.linalg.det(A)
    det_B = np.linalg.det(B)
    
    print(f"矩阵A的行列式: det(A) = {det_A:.6f}")
    print(f"矩阵B的行列式: det(B) = {det_B:.6f}")
    
    # 性质1: det(AB) = det(A) × det(B)
    det_AB = np.linalg.det(A @ B)
    product_det = det_A * det_B
    print(f"\n1. det(AB) = det(A) × det(B):")
    print(f"   det(AB) = {det_AB:.6f}")
    print(f"   det(A)×det(B) = {product_det:.6f}")
    print(f"   是否相等？ {np.isclose(det_AB, product_det)}")
    
    # 性质2: det(A⁻¹) = 1/det(A) （如果可逆）
    if abs(det_A) > 1e-10:
        A_inv = np.linalg.inv(A)
        det_A_inv = np.linalg.det(A_inv)
        print(f"\n2. det(A⁻¹) = 1/det(A):")
        print(f"   det(A⁻¹) = {det_A_inv:.6f}")
        print(f"   1/det(A) = {1/det_A:.6f}")
        print(f"   是否相等？ {np.isclose(det_A_inv, 1/det_A)}")
    
    # 性质3: det(I) = 1
    I = np.eye(3)
    det_I = np.linalg.det(I)
    print(f"\n3. det(I) = 1:")
    print(f"   det(I) = {det_I}")
    print(f"   是否正确？ {np.isclose(det_I, 1)}")
    
    # 性质4: det(cA) = cⁿ det(A)（n是维度）
    c = 2.5
    det_cA = np.linalg.det(c * A)
    expected = (c ** 3) * det_A
    print(f"\n4. det(cA) = cⁿ det(A) (n=3):")
    print(f"   det({c}A) = {det_cA:.6f}")
    print(f"   {c}³×det(A) = {expected:.6f}")
    print(f"   是否相等？ {np.isclose(det_cA, expected)}")
    
    return A, B

A, B = verify_determinant_properties()
```
### 5. **行列式与线性方程组**
```python
def determinant_and_linear_systems():
    """行列式与线性方程组解的关系"""
    
    print("=== 行列式与线性方程组 ===")
    
    # 三个不同的3×3矩阵
    matrices = {
        "满秩矩阵（有唯一解）": np.array([[2, 1, -1],
                                       [-3, -1, 2],
                                       [-2, 1, 2]]),
        "秩亏矩阵（无穷多解或无解）": np.array([[1, 2, 3],
                                          [2, 4, 6],  # 第二行是第一行的2倍
                                          [1, 1, 1]]),
        "接近奇异的矩阵": np.array([[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9.0001]])  # 稍微修改避免完全奇异
    }
    
    for name, M in matrices.items():
        det_M = np.linalg.det(M)
        cond_M = np.linalg.cond(M)  # 条件数，衡量数值稳定性
        
        print(f"\n{name}:")
        print(f"  矩阵 M = \n{M}")
        print(f"  行列式 det(M) = {det_M:.6f}")
        print(f"  条件数 cond(M) = {cond_M:.2e}")
        
        if abs(det_M) < 1e-10:
            print("  → 行列式为0，矩阵奇异，方程组可能无解或有无穷多解")
        elif cond_M > 1e10:
            print("  → 条件数很大，矩阵接近奇异，数值计算可能不稳定")
        else:
            print("  → 行列式非0，矩阵可逆，方程组有唯一解")
            
            # 解一个具体的线性方程组 Mx = b
            b = np.array([8, -11, -3])
            try:
                x = np.linalg.solve(M, b)
                print(f"  解方程 Mx = {b} 的解: x = {x}")
            except:
                print("  求解失败（数值不稳定）")
    
    return matrices

matrices_systems = determinant_and_linear_systems()
```
### 6. **手动计算行列式**
```python
def manual_determinant_calculation():
    """手动计算行列式"""
    
    print("=== 手动计算行列式 ===")
    
    # 2×2矩阵：det([[a,b],[c,d]]) = ad - bc
    def det_2x2(M):
        return M[0,0]*M[1,1] - M[0,1]*M[1,0]
    
    # 3×3矩阵：Sarrus法则
    def det_3x3_sarrus(M):
        # 复制前两列
        extended = np.hstack([M, M[:,:2]])
        
        # 正对角线乘积之和
        pos_sum = (extended[0,0]*extended[1,1]*extended[2,2] +
                   extended[0,1]*extended[1,2]*extended[2,3] +
                   extended[0,2]*extended[1,3]*extended[2,4])
        
        # 反对角线乘积之和
        neg_sum = (extended[0,2]*extended[1,1]*extended[2,0] +
                   extended[0,3]*extended[1,2]*extended[2,1] +
                   extended[0,4]*extended[1,3]*extended[2,2])
        
        return pos_sum - neg_sum
    
    # 3×3矩阵：余子式展开（第一行）
    def det_3x3_cofactor(M):
        a, b, c = M[0,0], M[0,1], M[0,2]
        
        # 2×2子矩阵的行列式
        det_M11 = M[1,1]*M[2,2] - M[1,2]*M[2,1]  # 去掉第1行第1列
        det_M12 = M[1,0]*M[2,2] - M[1,2]*M[2,0]  # 去掉第1行第2列
        det_M13 = M[1,0]*M[2,1] - M[1,1]*M[2,0]  # 去掉第1行第3列
        
        return a*det_M11 - b*det_M12 + c*det_M13
    
    # 测试
    M_2x2 = np.array([[3, 8], [4, 6]])
    M_3x3 = np.array([[6, 1, 1],
                      [4, -2, 5],
                      [2, 8, 7]])
    
    print(f"2×2矩阵 M = \n{M_2x2}")
    print(f"手动计算: det(M) = {det_2x2(M_2x2)}")
    print(f"NumPy计算: det(M) = {np.linalg.det(M_2x2)}")
    
    print(f"\n3×3矩阵 M = \n{M_3x3}")
    print(f"Sarrus法则: det(M) = {det_3x3_sarrus(M_3x3)}")
    print(f"余子式展开: det(M) = {det_3x3_cofactor(M_3x3)}")
    print(f"NumPy计算: det(M) = {np.linalg.det(M_3x3)}")
    
    # 验证
    methods = [
        ("Sarrus法则", det_3x3_sarrus(M_3x3)),
        ("余子式展开", det_3x3_cofactor(M_3x3)),
        ("NumPy", np.linalg.det(M_3x3))
    ]
    
    all_close = all(np.isclose(methods[0][1], m[1]) for m in methods[1:])
    print(f"\n所有方法结果一致？ {all_close}")
    
    return det_2x2, det_3x3_sarrus, det_3x3_cofactor

det_functions = manual_determinant_calculation()
```
## 🎯 **关键理解**

### **行列式的符号意义**
- **正行列式**：变换保持方向（右手系保持右手系）
- **负行列式**：变换翻转方向（右手系变左手系）
- **零行列式**：变换降维（面积/体积压缩为0）

### **在机器学习中的应用**
1. **模型可逆性**：行列式非零 ↔ 矩阵可逆 ↔ 变换可逆
2. **概率密度变换**：多元高斯分布中的雅可比行列式
3. **归一化流**：使用行列式计算概率密度变换
4. **主成分分析**：协方差矩阵的行列式表示总方差

### **数值注意事项**
- 行列式值可能**非常大**或**非常小**（数值溢出/下溢）
- **条件数**比行列式更能反映数值稳定性
- 对于大规模矩阵，直接计算行列式代价高昂

---

## 📝 **练习与思考**
### **练习题**

1. 计算以下矩阵的行列式，并解释几何意义：

   (1) 缩放矩阵

   $$
   \begin{bmatrix} 
   2 & 0 \\ 
   0 & 3 \\
   \end{bmatrix}
   $$

   (2) 奇异矩阵（行列式为 0）
   
   $$
   \begin{bmatrix}
   1 & 2 \\
   2 & 4 \\
   \end{bmatrix}
   $$

   (3) 90° 旋转矩阵
   
   $$
   \begin{bmatrix}
   0 & -1 \\
   1 & 0 \\
   \end{bmatrix}
   $$
   
2. 验证：对于任意2×2矩阵A和B，有det(AB) = det(A)det(B)
```python
import numpy as np
import random


def verify_det_property(n_tests=1000):
    """验证 det(AB) = det(A)det(B)"""
    for _ in range(n_tests):
        # 随机生成两个2×2矩阵
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 2)

        # 计算两边
        det_AB = np.linalg.det(A @ B)
        det_A_det_B = np.linalg.det(A) * np.linalg.det(B)

        # 验证是否相等（考虑浮点误差）
        if not np.isclose(det_AB, det_A_det_B):
            print(f"❌ 验证失败！")
            print(f"A = {A}")
            print(f"B = {B}")
            print(f"det(AB) = {det_AB}")
            print(f"det(A)det(B) = {det_A_det_B}")
            return False

    print(f"✅ 所有 {n_tests} 次验证通过！")
    return True


verify_det_property()
```
3. 创建一个3×3矩阵，使其行列式为：
   - 正值
   - 负值
   - 零
   并解释每个矩阵的几何意义

4. 编写函数计算4×4矩阵的行列式（使用余子式展开）
#### 方法1：按第一行展开（余子式法）
```python
def determinant_4x4_expansion(M):
    """
    计算4×4矩阵的行列式 - 按第一行余子式展开
    公式: det(M) = Σ (-1)^(1+j) * M[0,j] * det(M_{0j})
    其中M_{0j}是去掉第0行第j列的3×3子矩阵
    """
    
    # 先定义3×3行列式计算（使用Sarrus法则）
    def det_3x3(A):
        """计算3×3矩阵行列式（Sarrus法则）"""
        return (A[0,0]*A[1,1]*A[2,2] + 
                A[0,1]*A[1,2]*A[2,0] + 
                A[0,2]*A[1,0]*A[2,1] -
                A[0,2]*A[1,1]*A[2,0] - 
                A[0,1]*A[1,0]*A[2,2] - 
                A[0,0]*A[1,2]*A[2,1])
    
    det = 0
    for j in range(4):
        # 创建子矩阵（去掉第0行第j列）
        sub_matrix = []
        for row in range(1, 4):  # 跳过第0行
            new_row = []
            for col in range(4):
                if col != j:
                    new_row.append(M[row, col])
            sub_matrix.append(new_row)
        
        sub_matrix = np.array(sub_matrix)
        
        # 计算余子式
        cofactor = ((-1) ** j) * M[0, j] * det_3x3(sub_matrix)
        det += cofactor
    
    return det
```
#### 方法2：通用递归版本
```python
def determinant_recursive(M):
    """递归计算任意n×n矩阵的行列式"""
    n = M.shape[0]
    
    # 基本情况
    if n == 1:
        return M[0, 0]
    elif n == 2:
        return M[0, 0]*M[1, 1] - M[0, 1]*M[1, 0]
    
    det = 0
    for j in range(n):
        # 创建余子矩阵（去掉第0行第j列）
        sub_matrix = np.delete(np.delete(M, 0, axis=0), j, axis=1)
        
        # 计算余子式：(-1)^j * M[0,j] * det(余子矩阵)
        cofactor = ((-1) ** j) * M[0, j] * determinant_recursive(sub_matrix)
        det += cofactor
    
    return det

def determinant_4x4_recursive(M):
    """使用递归计算4×4矩阵行列式"""
    return determinant_recursive(M)
```
#### 方法3：拉普拉斯展开（按行列展开）
```python
def determinant_4x4_laplace(M, row=0):
    """
    拉普拉斯展开计算行列式
    可以按任意行或列展开
    """
    
    def det_3x3_quick(A):
        """快速计算3×3行列式"""
        return (A[0,0]*(A[1,1]*A[2,2] - A[1,2]*A[2,1]) -
                A[0,1]*(A[1,0]*A[2,2] - A[1,2]*A[2,0]) +
                A[0,2]*(A[1,0]*A[2,1] - A[1,1]*A[2,0]))
    
    # 按第row行展开
    det = 0
    for col in range(4):
        if M[row, col] != 0:  # 跳过0元素加速计算
            # 创建余子矩阵
            rows = [i for i in range(4) if i != row]
            cols = [j for j in range(4) if j != col]
            minor = M[np.ix_(rows, cols)]
            
            # 计算余子式
            cofactor = ((-1) ** (row + col)) * M[row, col] * det_3x3_quick(minor)
            det += cofactor
    
    return det
```
#### 测试验证
```python
# 测试矩阵
print("=== 测试4×4行列式计算 ===")

# 测试1：单位矩阵（行列式应为1）
I4 = np.eye(4)
print(f"测试1 - 4×4单位矩阵:")
print(f"  展开法: {determinant_4x4_expansion(I4):.6f}")
print(f"  递归法: {determinant_4x4_recursive(I4):.6f}")
print(f"  拉普拉斯: {determinant_4x4_laplace(I4):.6f}")
print(f"  NumPy验证: {np.linalg.det(I4):.6f}")

# 测试2：对角矩阵（行列式应为对角线乘积）
D = np.diag([2, 3, 4, 5])
print(f"\n测试2 - 对角矩阵 diag(2,3,4,5):")
print(f"  理论值: 2×3×4×5 = {2*3*4*5}")
print(f"  展开法: {determinant_4x4_expansion(D):.6f}")
print(f"  NumPy验证: {np.linalg.det(D):.6f}")

# 测试3：随机矩阵
np.random.seed(42)
M_rand = np.random.randn(4, 4)
print(f"\n测试3 - 随机矩阵:")
print(f"  展开法: {determinant_4x4_expansion(M_rand):.6f}")
print(f"  递归法: {determinant_4x4_recursive(M_rand):.6f}")
print(f"  拉普拉斯: {determinant_4x4_laplace(M_rand):.6f}")
print(f"  NumPy验证: {np.linalg.det(M_rand):.6f}")

# 测试4：奇异矩阵（行列式应为0）
M_singular = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]])
print(f"\n测试4 - 奇异矩阵（第4行是线性组合）:")
print(f"  展开法: {determinant_4x4_expansion(M_singular):.6f}")
print(f"  理论值: 应该接近0")
```
#### 可视化解释：4×4行列式的几何意义
```python
def explain_4d_determinant():
    """解释4×4行列式的几何意义"""
    
    print("\n=== 4×4行列式的几何意义 ===")
    print("在4维空间中，4×4矩阵行列式的绝对值表示：")
    print("1. 4维'超平行体'的'超体积'缩放因子")
    print("2. 线性变换对4维空间体积的缩放程度")
    print("3. 符号表示方向：+保持方向，-翻转方向")
    print("4. 0表示将4维空间压缩到更低维度")
    
    print("\n🔍 举例说明：")
    
    # 单位矩阵：体积不变
    print("1. 单位矩阵 I₄:")
    print("   det = 1，表示4维体积不变")
    
    # 缩放矩阵：体积缩放
    S = np.diag([2, 2, 2, 2])
    print(f"\n2. 均匀缩放2倍矩阵:")
    print(f"   det = {np.linalg.det(S)}")
    print(f"   几何：每个维度放大2倍，4维体积放大2⁴=16倍")
    
    # 投影矩阵：降维
    P = np.diag([1, 1, 1, 0])
    print(f"\n3. 投影到前3维的矩阵:")
    print(f"   det = {np.linalg.det(P)}")
    print(f"   几何：将4维空间投影到3维子空间，4维体积为0")

explain_4d_determinant()
```
#### 性能比较
```python
import time

def performance_comparison():
    """比较不同方法的性能"""
    
    np.random.seed(42)
    test_matrix = np.random.randn(4, 4)
    
    methods = [
        ("展开法", determinant_4x4_expansion),
        ("递归法", determinant_4x4_recursive),
        ("拉普拉斯", lambda m: determinant_4x4_laplace(m, 0)),
        ("NumPy", np.linalg.det)
    ]
    
    print("=== 性能比较（运行1000次） ===")
    
    for name, func in methods:
        start = time.time()
        for _ in range(1000):
            result = func(test_matrix.copy())
        elapsed = time.time() - start
        
        print(f"{name:10} : {elapsed:.4f}秒")
    
    print("\n💡 结论：")
    print("- NumPy最快（使用优化算法）")
    print("- 展开法和拉普拉斯法相当")
    print("- 递归法最慢（但最通用）")

performance_comparison()
```
### **思考题**
1. 为什么行列式为0意味着矩阵不可逆？从几何角度解释
```
* 几何解释：行列式=0表示线性变换降维（2D→线，3D→面/线）

* 信息丢失：多个不同输入映射到相同输出

* 不可逆：无法从输出唯一确定原始输入

* 就像把3D物体压扁成2D照片，无法还原深度信息
```
2. 在神经网络中，权重矩阵的行列式有什么意义？
```
* 模型容量：行列式绝对值大 → 变换能力强

* 梯度稳定性：行列式接近0 → 梯度消失风险

* 归一化流：用行列式计算概率密度变换（雅可比行列式）

* 初始化：确保权重矩阵行列式合理，避免训练问题
```
3. 如何快速判断一个矩阵是否接近奇异（病态）？
```
* 条件数：$\kappa(A) = |A| \cdot |A^{-1}|$，越大越病态

* 行列式接近0：但需与矩阵尺度比较

* 奇异值：最小奇异值接近0

* 直观检查：行/列几乎线性相关
```
4. 行列式在计算机图形学中有哪些应用？
```
* 变换可逆性：判断模型变换是否可逆

* 体积缩放：计算3D对象的缩放因子

* 法向量变换：使用逆转置矩阵（涉及行列式）

* 投影矩阵：透视投影矩阵行列式≠0确保可逆

* 背面剔除：判断三角形朝向（叉积行列式符号）
```
---

## 🚀 **下一步学习建议**

### **立即练习：**
```python
# 练习1：验证行列式性质
import numpy as np

def practice_determinant():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print(f"det(A) = {np.linalg.det(A):.2f}")
    print(f"det(B) = {np.linalg.det(B):.2f}")
    print(f"det(A@B) = {np.linalg.det(A@B):.2f}")
    print(f"det(A)×det(B) = {np.linalg.det(A)*np.linalg.det(B):.2f}")
    print(f"det(AB) = det(A)det(B)？ {np.isclose(np.linalg.det(A@B), np.linalg.det(A)*np.linalg.det(B))}")

practice_determinant()
```
### 连接应用：
* 计算机图形学：判断变换是否可逆，计算缩放因子

* 物理学：坐标变换的雅可比行列式

* 机器学习：归一化流、变分推断

* 工程学：系统稳定性分析
