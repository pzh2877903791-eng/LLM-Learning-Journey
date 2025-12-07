# 🎬 **第7集：逆矩阵、列空间与零空间**

## 📺 视频信息
- **视频标题**：Inverse matrices, column space and null space
- **3B1B原视频**：https://www.youtube.com/watch?v=uQhTuRlWMxw
- **B站中文字幕**：https://www.bilibili.com/video/BV1ys411472E?p=7

---

## 🎯 核心概念

### 1. **逆矩阵（Inverse Matrix）**
- $A^{-1}$：撤销变换 $A$ 的效果
- $AA^{-1} = A^{-1}A = I$（单位矩阵）
- 存在条件：行列式不为0（变换不降维）

### 2. **秩（Rank）**
- 矩阵的**列空间维度**
- 变换后空间的维度
- 最大线性无关列/行的数量

### 3. **列空间（Column Space）**
- 所有列向量的线性组合构成的子空间
- 变换后所有可能的输出向量
- 维度 = 秩

### 4. **零空间（Null Space）**
- 所有被映射到零向量的输入向量
- $A\mathbf{x} = \mathbf{0}$ 的解空间
- 满秩矩阵的零空间只有零向量

### 5. **解的存在性**
- $A\mathbf{x} = \mathbf{b}$ 有解 ⇔ $\mathbf{b}$ 在列空间中
- 唯一解 ⇔ 零空间只有零向量
- 无穷多解 ⇔ 零空间有非零向量

---

## 💻 代码实现

```python
import numpy as np
import matplotlib.pyplot as plt
```

### 1. 逆矩阵实战演示
```python
def inverse_matrix_demo():
    """逆矩阵完整演示"""
    
    print("=== 逆矩阵演示 ===\n")
    
    # 示例1：可逆矩阵
    A_invertible = np.array([[2, 1],
                             [1, 3]])
    
    print("1. 可逆矩阵示例:")
    print(f"矩阵 A:\n{A_invertible}")
    print(f"行列式 det(A) = {np.linalg.det(A_invertible):.2f}")
    
    # 计算逆矩阵
    if np.linalg.det(A_invertible) != 0:
        A_inv = np.linalg.inv(A_invertible)
        print(f"\n逆矩阵 A⁻¹:\n{A_inv}")
        print(f"\n验证 A @ A⁻¹:\n{A_invertible @ A_inv}")
        print(f"验证 A⁻¹ @ A:\n{A_inv @ A_invertible}")
    
    # 示例2：奇异矩阵（不可逆）
    print("\n" + "="*40 + "\n")
    print("2. 奇异矩阵示例:")
    
    B_singular = np.array([[1, 2],
                           [2, 4]])  # 第二行是第一行的2倍
    
    print(f"矩阵 B:\n{B_singular}")
    print(f"行列式 det(B) = {np.linalg.det(B_singular):.2f}")
    
    # 尝试计算逆矩阵（会报错）
    try:
        B_inv = np.linalg.inv(B_singular)
    except np.linalg.LinAlgError:
        print("❌ 无法计算逆矩阵：矩阵奇异")
    
    # 几何解释
    print("\n3. 几何解释:")
    print("• 可逆变换：保持维度，可以逆向恢复")
    print("• 奇异变换：降低维度，信息丢失，无法恢复")
    
    return A_invertible, B_singular

# 运行演示
A_inv, B_sing = inverse_matrix_demo()
```
### 2. 列空间可视化
```python
def column_space_demo():
    """列空间可视化演示"""
    
    print("\n" + "="*60 + "\n")
    print("=== 列空间演示 ===\n")
    
    # 创建不同秩的矩阵
    matrices = {
        "满秩矩阵 (秩=2)": np.array([[2, 1],
                                     [1, 3]]),
        "秩1矩阵": np.array([[1, 2],
                            [2, 4]]),
        "秩0矩阵 (零矩阵)": np.array([[0, 0],
                                     [0, 0]])
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (title, M) in enumerate(matrices.items()):
        ax = axes[idx]
        
        # 绘制列向量
        v1 = M[:, 0]
        v2 = M[:, 1]
        
        # 绘制原点
        ax.scatter(0, 0, color='black', s=50, zorder=5)
        
        # 绘制列向量
        ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', 
                  scale=1, color='red', width=0.01, label=f'v1={v1}')
        ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', 
                  scale=1, color='blue', width=0.01, label=f'v2={v2}')
        
        # 计算秩和列空间
        rank = np.linalg.matrix_rank(M)
        det = np.linalg.det(M) if M.shape[0] == M.shape[1] else "N/A"
        
        # 可视化列空间（张成的空间）
        if rank == 2:
            # 填充整个平面
            x = np.linspace(-3, 3, 10)
            y = np.linspace(-3, 3, 10)
            X, Y = np.meshgrid(x, y)
            ax.fill_between([-3, 3], -3, 3, alpha=0.1, color='green')
            col_space = "整个平面"
        elif rank == 1:
            # 填充一条线
            if np.any(v1):  # v1不是零向量
                line_dir = v1 / np.linalg.norm(v1)
                line_points = line_dir.reshape(-1, 1) * np.linspace(-3, 3, 10)
                ax.plot(line_points[0], line_points[1], 'g-', linewidth=3, alpha=0.3)
            col_space = "一条直线"
        else:
            col_space = "原点"
        
        # 设置图形属性
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axhline(0, color='gray', alpha=0.3)
        ax.axvline(0, color='gray', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        ax.set_title(f"{title}\n秩={rank}, 行列式={det}\n列空间: {col_space}", 
                    fontsize=11)
        ax.legend(loc='upper right')
    
    plt.suptitle('列空间可视化：矩阵的列向量张成的空间', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()
    
    # 数学定义
    print("数学定义:")
    print("列空间 = 所有列向量的线性组合")
    print("        = { c₁v₁ + c₂v₂ + ... + cₙvₙ | cᵢ ∈ ℝ }")
    print("        = 矩阵所有可能的输出 A𝐱")
    
    return matrices

# 运行演示
matrices_dict = column_space_demo()
```
### 3. 零空间计算与可视化
```python
def null_space_demo():
    """零空间计算演示"""
    
    print("\n" + "="*60 + "\n")
    print("=== 零空间演示 ===\n")
    
    # 定义几个不同矩阵
    matrices = {
        "满秩矩阵": np.array([[2, 1],
                             [1, 3]]),
        "秩亏矩阵1": np.array([[1, 2],
                              [2, 4]]),
        "秩亏矩阵2": np.array([[1, 2, 3],
                              [2, 4, 6],
                              [3, 6, 9]])
    }
    
    for name, M in matrices.items():
        print(f"\n{name}:")
        print(f"矩阵 M:\n{M}")
        print(f"形状: {M.shape}")
        
        # 计算秩
        rank = np.linalg.matrix_rank(M)
        print(f"秩: {rank}")
        
        # 计算零空间维度（零度）
        nullity = M.shape[1] - rank
        print(f"零空间维度（零度）: {nullity}")
        
        # 寻找零空间基向量
        if nullity > 0:
            print("寻找零空间向量（解 Mx = 0）:")
            
            # 方法1：使用SVD
            U, S, Vt = np.linalg.svd(M)
            
            # 零空间基向量是Vt的最后nullity行
            null_space_basis = Vt[-nullity:, :] if nullity > 0 else []
            
            print("零空间基向量:")
            for i, vec in enumerate(null_space_basis):
                print(f"  v{i+1} = {vec}")
                
                # 验证确实是零空间向量
                result = M @ vec
                print(f"  验证 M @ v{i+1} = {result} (应为零向量)")
                print(f"  是否为0? {np.allclose(result, 0)}")
        else:
            print("零空间只有零向量")
        
        # 几何解释
        if nullity == 0:
            print("几何: 变换是一对一的，没有信息丢失")
        elif nullity == 1:
            print("几何: 将一条直线压缩到原点")
        else:
            print(f"几何: 将{nullity}维空间压缩到原点")
    
    return matrices

# 运行演示
matrices_ns = null_space_demo()
```
### 4. 解的存在性分析
```python
def solution_existence_demo():
    """解的存在性分析"""
    
    print("\n" + "="*60 + "\n")
    print("=== 解的存在性分析 ===\n")
    
    # 创建测试用例
    test_cases = [
        {
            "name": "情况1：唯一解",
            "A": np.array([[2, 1],
                          [1, 3]]),
            "b": np.array([3, 4])
        },
        {
            "name": "情况2：无解（b不在列空间中）",
            "A": np.array([[1, 2],
                          [2, 4]]),
            "b": np.array([1, 0])  # 不在列空间中
        },
        {
            "name": "情况3：无穷多解",
            "A": np.array([[1, 2],
                          [2, 4]]),
            "b": np.array([2, 4])  # 在列空间中
        }
    ]
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        print(f"A = \n{case['A']}")
        print(f"b = {case['b']}")
        
        A = case['A']
        b = case['b']
        
        # 分析矩阵属性
        rank_A = np.linalg.matrix_rank(A)
        rank_Ab = np.linalg.matrix_rank(np.column_stack((A, b)))
        
        print(f"秩(A) = {rank_A}")
        print(f"秩([A|b]) = {rank_Ab}")
        
        # 判断解的情况
        if rank_A == rank_Ab:
            if rank_A == A.shape[1]:  # 列满秩
                print("→ 存在唯一解")
                try:
                    x = np.linalg.solve(A, b)
                    print(f"解 x = {x}")
                    print(f"验证 Ax = {A @ x} (应为 {b})")
                except np.linalg.LinAlgError:
                    print("数值求解失败")
            else:
                print("→ 存在无穷多解")
                print("零空间维度 =", A.shape[1] - rank_A)
        else:
            print("→ 无解")
            print(f"b不在A的列空间中")
        
        # 可视化
        if A.shape == (2, 2):
            visualize_2d_solution(A, b, case['name'])
    
    return test_cases

def visualize_2d_solution(A, b, title):
    """可视化2D线性方程组的解"""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制列向量
    v1 = A[:, 0]
    v2 = A[:, 1]
    
    # 绘制列向量
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', 
              scale=1, color='red', width=0.008, label=f'a₁={v1}')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', 
              scale=1, color='blue', width=0.008, label=f'a₂={v2}')
    
    # 绘制列空间（所有可能的Ax）
    # 生成网格点
    x_vals = np.linspace(-2, 2, 20)
    y_vals = np.linspace(-2, 2, 20)
    
    col_space_points = []
    for x in x_vals:
        for y in y_vals:
            point = A @ np.array([x, y])
            col_space_points.append(point)
    
    col_space_points = np.array(col_space_points)
    
    # 绘制列空间区域
    if np.linalg.matrix_rank(A) == 2:
        # 满秩：填充整个区域
        ax.fill_between([-5, 5], -5, 5, alpha=0.1, color='green', 
                       label='列空间（整个平面）')
    else:
        # 秩1：绘制直线
        if np.linalg.norm(v1) > 0:
            direction = v1 / np.linalg.norm(v1)
        else:
            direction = v2 / np.linalg.norm(v2)
        
        t = np.linspace(-5, 5, 100)
        line_points = np.outer(direction, t)
        ax.plot(line_points[0], line_points[1], 'g-', linewidth=3, 
               alpha=0.3, label='列空间（直线）')
    
    # 绘制目标向量b
    ax.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', 
              scale=1, color='purple', width=0.012, 
              label=f'b={b}', zorder=5)
    
    # 标记b是否在列空间中
    rank_A = np.linalg.matrix_rank(A)
    rank_Ab = np.linalg.matrix_rank(np.column_stack((A, b)))
    
    if rank_A == rank_Ab:
        ax.scatter(b[0], b[1], color='green', s=100, marker='o', 
                  edgecolor='black', zorder=6, label='b在列空间中')
    else:
        ax.scatter(b[0], b[1], color='red', s=100, marker='x', 
                  linewidth=2, zorder=6, label='b不在列空间中')
    
    # 设置图形
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axhline(0, color='gray', alpha=0.3)
    ax.axvline(0, color='gray', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title(f'{title}\n秩(A)={rank_A}, 秩([A|b])={rank_Ab}', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# 运行演示
test_cases = solution_existence_demo()
```
### 5. LLM中的实际应用
```python
def llm_applications():
    """LLM中的逆矩阵、列空间、零空间应用"""
    
    print("\n" + "="*60 + "\n")
    print("=== 在LLM中的应用 ===\n")
    
    applications = {
        "1. 注意力机制分析": {
            "概念": "列空间与秩",
            "解释": "注意力头的列空间维度决定了它能捕获的信息类型",
            "代码示例": """
# 分析注意力头的表达能力
attention_head = model.layers[0].attention.heads[0]
W_q, W_k, W_v = attention_head.query, attention_head.key, attention_head.value

# 计算秩
rank_q = np.linalg.matrix_rank(W_q.weight.detach().numpy())
print(f"Query投影矩阵秩: {rank_q}/{W_q.weight.shape[1]}")
            """
        },
        "2. 模型可解释性": {
            "概念": "逆变换",
            "解释": "如果变换可逆，可以从输出反推输入特征",
            "代码示例": """
# 检查线性层是否近似可逆
linear_layer = model.fc
W = linear_layer.weight.detach().numpy()
det = np.linalg.det(W.T @ W)  # 对于非方阵

if det > 1e-6:
    print("层近似可逆，可能具有良好解释性")
else:
    print("层不可逆，信息有丢失")
            """
        },
        "3. 参数效率分析": {
            "概念": "零空间",
            "解释": "零空间大的权重矩阵有冗余参数",
            "代码示例": """
def analyze_parameter_efficiency(weight_matrix):
    # 计算零空间维度
    rank = np.linalg.matrix_rank(weight_matrix)
    nullity = weight_matrix.shape[1] - rank
    efficiency = rank / weight_matrix.shape[1]
    
    print(f"参数效率: {efficiency:.1%}")
    print(f"零空间维度: {nullity} (冗余参数)")
    return efficiency
            """
        },
        "4. 梯度问题诊断": {
            "概念": "条件数",
            "解释": "矩阵接近奇异会导致梯度消失/爆炸",
            "代码示例": """
def check_gradient_stability(weight_matrix):
    # 计算条件数（最大奇异值/最小奇异值）
    U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
    condition_number = S[0] / S[-1]
    
    if condition_number > 1e6:
        print(f"⚠️ 条件数过大: {condition_number:.2e}")
        print("可能导致梯度问题")
    return condition_number
            """
        }
    }
    
    for title, info in applications.items():
        print(f"\n{title}:")
        print(f"  相关概念: {info['概念']}")
        print(f"  解释: {info['解释']}")
        if '代码示例' in info:
            print(f"  代码示例:{info['代码示例']}")
    
    # 实际演示
    print("\n" + "-"*40)
    print("实际演示：分析一个简单的权重矩阵")
    
    # 模拟一个LLM中的权重矩阵
    np.random.seed(42)
    n_input = 64
    n_output = 32
    
    # 创建权重矩阵
    W = np.random.randn(n_output, n_input)
    
    print(f"\n权重矩阵形状: {W.shape}")
    print(f"参数数量: {W.size:,}")
    
    # 分析秩和效率
    rank = np.linalg.matrix_rank(W)
    nullity = n_input - rank
    efficiency = rank / min(n_input, n_output)
    
    print(f"秩: {rank}/{min(n_input, n_output)}")
    print(f"零空间维度: {nullity}")
    print(f"参数效率: {efficiency:.1%}")
    
    if nullity > 0:
        print("→ 存在参数冗余，可以考虑压缩")
    else:
        print("→ 参数效率较高")
    
    # 分析条件数
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    condition_number = S[0] / S[-1]
    
    print(f"\n奇异值范围: [{S[-1]:.2e}, {S[0]:.2e}]")
    print(f"条件数: {condition_number:.2e}")
    
    if condition_number > 1e6:
        print("⚠️ 条件数过大，训练时可能出现梯度问题")
    else:
        print("✓ 条件数合理")
    
    return applications

# 运行演示
llm_apps = llm_applications()
```
## 🎯 关键理解
### 几何直观总结
```text
1. 逆矩阵：
   • 可逆变换 ↔ 保持维度，可逆向恢复
   • 行列式≠0 ↔ 面积/体积缩放≠0

2. 列空间：
   • 矩阵所有可能的输出
   • 列向量张成的空间
   • 维度 = 秩

3. 零空间：
   • 被压缩到原点的输入向量
   • 解A𝐱=𝟎的所有𝐱
   • 维度 = 列数 - 秩
解的情况总结
条件	解的情况	几何意义
𝐛在列空间中 & 零空间={𝟎}	唯一解	变换是一对一的
𝐛在列空间中 & 零空间≠{𝟎}	无穷多解	变换是多对一的
𝐛不在列空间中	无解	𝐛无法被变换得到
```
## 📝 练习与思考
### 练习题
1. 判断矩阵是否可逆

```python
def check_invertibility():
    """判断矩阵是否可逆"""
    
    matrices = [
        np.array([[1, 2], [3, 4]]),
        np.array([[1, 2], [2, 4]]),
        np.array([[2, 1, 3], [1, 2, 1], [3, 3, 6]]),
        np.eye(3)
    ]
    
    print("判断矩阵是否可逆:")
    for i, M in enumerate(matrices):
        det = np.linalg.det(M) if M.shape[0] == M.shape[1] else None
        rank = np.linalg.matrix_rank(M)
        
        print(f"\n矩阵{i+1}:")
        print(f"形状: {M.shape}, 秩: {rank}")
        if det is not None:
            print(f"行列式: {det:.4f}")
        
        if M.shape[0] == M.shape[1]:
            if abs(det) > 1e-10:
                print("✓ 可逆")
            else:
                print("✗ 不可逆（奇异）")
        else:
            print("非方阵，没有通常的逆矩阵")
    
    return matrices

matrices_check = check_invertibility()
```
2. 计算零空间基向量

```python
def find_null_space_basis():
    """计算零空间基向量"""
    
    # 定义矩阵
    A = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [3, 6, 9]])
    
    print("计算矩阵的零空间基向量:")
    print(f"矩阵 A:\n{A}")
    
    # 方法1：使用SVD
    U, S, Vt = np.linalg.svd(A)
    
    # 零空间维度
    rank = np.linalg.matrix_rank(A)
    nullity = A.shape[1] - rank
    
    print(f"\n秩(A) = {rank}")
    print(f"零空间维度 = {nullity}")
    
    if nullity > 0:
        # 零空间基向量是Vt的最后nullity行
        null_basis = Vt[-nullity:, :]
        
        print(f"\n零空间基向量:")
        for i, vec in enumerate(null_basis):
            print(f"v{i+1} = {vec}")
            
            # 验证
            result = A @ vec
            print(f"  A @ v{i+1} = {result}")
            print(f"  是否为零向量? {np.allclose(result, 0)}\n")
    else:
        print("零空间只有零向量")
    
    return A, null_basis if nullity > 0 else None

A_example, null_basis_example = find_null_space_basis()
```
3. 线性方程组求解分析

```python
def analyze_linear_system():
    """分析线性方程组的解"""
    
    # 测试用例
    systems = [
        {
            "name": "唯一解系统",
            "A": np.array([[2, 1], [1, -1]]),
            "b": np.array([3, 1])
        },
        {
            "name": "无解系统",
            "A": np.array([[1, 2], [1, 2]]),
            "b": np.array([3, 4])  # 矛盾方程
        },
        {
            "name": "无穷多解系统",
            "A": np.array([[1, 2], [2, 4]]),
            "b": np.array([3, 6])  # 第二个方程是第一个的2倍
        }
    ]
    
    print("分析线性方程组的解:")
    
    for system in systems:
        print(f"\n{system['name']}:")
        print(f"A = \n{system['A']}")
        print(f"b = {system['b']}")
        
        A = system['A']
        b = system['b']
        
        # 计算秩
        rank_A = np.linalg.matrix_rank(A)
        Ab = np.column_stack((A, b))
        rank_Ab = np.linalg.matrix_rank(Ab)
        
        print(f"秩(A) = {rank_A}")
        print(f"秩([A|b]) = {rank_Ab}")
        
        # 判断解的情况
        if rank_A == rank_Ab:
            if rank_A == A.shape[1]:
                print("→ 存在唯一解")
                try:
                    x = np.linalg.solve(A, b)
                    print(f"解: x = {x}")
                except:
                    print("数值求解失败")
            else:
                print("→ 存在无穷多解")
                # 找一个特解
                x_particular = np.linalg.lstsq(A, b, rcond=None)[0]
                print(f"一个特解: xₚ = {x_particular}")
        else:
            print("→ 无解")
            
            # 找到最小二乘解
            x_approx = np.linalg.lstsq(A, b, rcond=None)[0]
            print(f"最小二乘近似解: x ≈ {x_approx}")
            print(f"残差: ||Ax - b|| = {np.linalg.norm(A @ x_approx - b):.4f}")
    
    return systems

linear_systems = analyze_linear_system()
```
### 思考题
1. 为什么奇异矩阵没有逆矩阵？从几何和代数两个角度解释

```text
几何角度：
• 奇异矩阵将高维空间压缩到低维空间
• 信息丢失，无法从输出唯一确定输入
• 就像把3D物体拍成2D照片，无法恢复深度

代数角度：
• 行列式=0，无法定义逆矩阵
• 存在非零向量被映射到零向量
• Ax=0有非零解，导致A⁻¹无法唯一确定
```
2. 在LLM中，权重矩阵的秩对模型能力有什么影响？

```text
高秩（接近满秩）：
• 表示能力强，能学习复杂函数
• 参数效率高，但可能过拟合
• 需要更多训练数据

低秩：
• 表示能力有限，类似正则化
• 参数有冗余，可能欠拟合
• 训练更稳定，泛化可能更好

实际应用：
• LoRA：故意使用低秩适配器
• 模型压缩：用低秩近似减少参数
• 注意力头：不同头可能有不同秩
```
3. 如何判断一个线性层是否"信息瓶颈"？

```text
检查方法：
1. 计算权重矩阵的秩
   rank = np.linalg.matrix_rank(W)
   
2. 分析奇异值分布
   U, S, Vt = np.linalg.svd(W)
   如果小奇异值很多接近0 → 信息丢失
   
3. 检查条件数
   cond = S.max() / S.min()
   条件数过大 → 数值不稳定
   
4. 实际测试重构误差
   x_test = np.random.randn(batch, input_dim)
   x_recon = (W.T @ (W @ x_test.T)).T
   误差 = ||x_test - x_recon||
```
4. 在Transformer中，为什么多头注意力比单头效果好？

```text
列空间视角：
• 每个注意力头有不同的列空间
• 多头组合扩大了总的列空间
• 能捕获更多类型的信息关系

零空间视角：
• 不同头的零空间不同
• 一个头忽略的信息可能被另一个头捕获
• 减少了总体信息丢失

数学表达：
多头注意力 = concat(head₁, ..., headₙ) @ Wₒ
每个headᵢ = Attention(QWᵢ_Q, KWᵢ_K, VWᵢ_V)
扩大了表示能力，增加了模型容量
```
## 🚀 下一步学习建议
### 与本集的连接：
```python
# 第7集 → 第9集（特征向量）
# 逆矩阵和秩的概念是特征值分解的基础

# 第7集 → LLM实践
# 理解模型容量、参数效率、训练稳定性
```
### 实战项目建议：
分析BERT的注意力头

```python
from transformers import BertModel
import torch

model = BertModel.from_pretrained('bert-base-uncased')
layer = model.encoder.layer[0]

# 分析每个注意力头的权重矩阵
for i in range(12):  # 12个注意力头
    W_q = layer.attention.self.query.weight
    W_k = layer.attention.self.key.weight
    W_v = layer.attention.self.value.weight
    
    # 计算秩和效率
    rank_q = torch.matrix_rank(W_q)
    efficiency = rank_q / min(W_q.shape)
    
    print(f"头{i+1}: 秩={rank_q}, 效率={efficiency:.1%}")
```
### 实现简单的LoRA

```python
class LoRALayer(nn.Module):
    """低秩适配器"""
    def __init__(self, original_layer, rank=4):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        
        # 低秩分解：ΔW = A @ B
        self.A = nn.Parameter(torch.randn(original_layer.out_features, rank))
        self.B = nn.Parameter(torch.randn(rank, original_layer.in_features))
        
    def forward(self, x):
        # 原始权重 + 低秩适配
        delta = self.A @ self.B
        return F.linear(x, self.original.weight + delta, self.original.bias)
```
### 可视化权重矩阵的奇异值

```python
def plot_singular_values(model):
    """绘制权重矩阵的奇异值谱"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    layers_to_plot = [
        ('embedding', model.embeddings.word_embeddings.weight),
        ('attention_q', model.encoder.layer[0].attention.self.query.weight),
        ('attention_out', model.encoder.layer[0].attention.output.dense.weight),
        ('ffn1', model.encoder.layer[0].intermediate.dense.weight),
        ('ffn2', model.encoder.layer[0].output.dense.weight),
        ('classifier', model.classifier.weight if hasattr(model, 'classifier') else None)
    ]
    
    for idx, (name, weight) in enumerate(layers_to_plot):
        if weight is None:
            continue
            
        ax = axes[idx]
        W = weight.detach().numpy()
        
        # 计算奇异值
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        
        # 绘制奇异值
        ax.plot(S, 'bo-', linewidth=1, markersize=3)
        ax.set_xlabel('奇异值索引', fontsize=9)
        ax.set_ylabel('大小', fontsize=9)
        ax.set_title(f'{name}\n秩={np.linalg.matrix_rank(W)}', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加95%能量标记
        total_energy = np.sum(S**2)
        cumulative = np.cumsum(S**2) / total_energy
        idx_95 = np.argmax(cumulative >= 0.95) + 1
        ax.axvline(idx_95, color='red', linestyle='--', alpha=0.7)
        ax.text(idx_95+0.5, S[0]*0.8, f'95%', fontsize=8)
    
    plt.suptitle('LLM各层权重矩阵的奇异值分布', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
```
### 学习检查清单：
* 理解逆矩阵的几何意义和存在条件

* 能计算矩阵的秩和零空间维度

* 能判断线性方程组解的情况

* 理解列空间和零空间的关系

* 能在LLM中应用这些概念分析模型
