# 🎬 第11集：抽象向量空间

## 📺 视频信息
- **视频标题**：Abstract vector spaces
- **3B1B原视频**：https://www.youtube.com/watch?v=TgKwz5Ikpc8
- **B站中字幕**：https://www.bilibili.com/video/BV1ys411472E?p=11

## 🎯 核心概念

### 1. 什么是抽象向量空间？
**核心思想**：将向量概念从几何空间推广到更一般的数学对象

**关键洞察**：只要满足8条公理，任何东西都可以是"向量"

### 2. 向量空间的8条公理
#### 加法公理：
1. **封闭性**：u + v 仍在空间中
2. **交换律**：u + v = v + u
3. **结合律**：(u + v) + w = u + (v + w)
4. **零向量存在**：存在0使得 v + 0 = v
5. **负向量存在**：对于每个v，存在-v使得 v + (-v) = 0

#### 标量乘法公理：
6. **封闭性**：αv 仍在空间中
7. **分配律1**：α(u + v) = αu + αv
8. **分配律2**：(α + β)v = αv + βv
9. **结合律**：α(βv) = (αβ)v
10. **单位标量**：1·v = v

### 3. 常见的抽象向量空间例子
* **多项式空间**：所有多项式的集合
* **函数空间**：满足某些条件的函数集合
* **矩阵空间**：所有m×n矩阵的集合
* **数列空间**：收敛数列的集合
* **解空间**：齐次线性方程组的解集合

### 4. 在ML中的重要性
* 理解高维嵌入空间（词向量、图像特征）
* 函数空间的机器学习（核方法、高斯过程）
* 希尔伯特空间与再生核希尔伯特空间（RKHS）
* 从有限维推广到无限维

## 💻 代码实现

### 1. 向量空间公理验证
```python
import numpy as np

print("=== 向量空间公理验证 ===")
print()

class VectorSpace:
    """验证向量空间公理的演示类"""
    
    def __init__(self, vectors):
        self.vectors = np.array(vectors)
        
    def check_closure_addition(self, u, v):
        """验证加法封闭性"""
        result = u + v
        # 检查结果是否仍在空间中（简化验证）
        return True  # 对于实数向量空间总是成立
    
    def check_commutativity(self, u, v):
        """验证交换律"""
        return np.allclose(u + v, v + u)
    
    def check_associativity(self, u, v, w):
        """验证结合律"""
        return np.allclose((u + v) + w, u + (v + w))
    
    def check_zero_vector(self):
        """验证零向量存在"""
        zero_vec = np.zeros_like(self.vectors[0])
        for v in self.vectors:
            if not np.allclose(v + zero_vec, v):
                return False
        return True
    
    def check_all_axioms(self):
        """验证所有公理"""
        print("验证向量空间公理:")
        print("-" * 40)
        
        u, v, w = self.vectors[:3]
        
        axioms = {
            "加法封闭性": self.check_closure_addition(u, v),
            "加法交换律": self.check_commutativity(u, v),
            "加法结合律": self.check_associativity(u, v, w),
            "零向量存在": self.check_zero_vector(),
        }
        
        for axiom, result in axioms.items():
            print(f"{axiom}: {'✓' if result else '✗'}")
        
        return all(axioms.values())

# 创建实数向量空间示例
vectors_r3 = [
    np.array([1, 2, 3]),
    np.array([4, 5, 6]),
    np.array([7, 8, 9]),
    np.array([-1, 0, 2])
]

vs = VectorSpace(vectors_r3)
vs.check_all_axioms()
```
### 2. 多项式向量空间
```python
print("\n" + "="*60)
print("=== 多项式向量空间 ===")
print()

class Polynomial:
    """多项式类，展示多项式构成向量空间"""
    
    def __init__(self, coefficients):
        """系数从低次到高次，如 [1, 2, 3] 表示 1 + 2x + 3x²"""
        self.coeffs = np.array(coefficients, dtype=float)
        self.degree = len(coefficients) - 1 if len(coefficients) > 0 else -1
    
    def __add__(self, other):
        """多项式加法"""
        max_len = max(len(self.coeffs), len(other.coeffs))
        coeffs1 = np.pad(self.coeffs, (0, max_len - len(self.coeffs)))
        coeffs2 = np.pad(other.coeffs, (0, max_len - len(other.coeffs)))
        return Polynomial(coeffs1 + coeffs2)
    
    def __mul__(self, scalar):
        """标量乘法"""
        return Polynomial(self.coeffs * scalar)
    
    def __str__(self):
        """字符串表示"""
        terms = []
        for i, coeff in enumerate(self.coeffs):
            if abs(coeff) > 1e-10:  # 忽略接近零的系数
                if i == 0:
                    terms.append(f"{coeff:.2f}")
                elif i == 1:
                    terms.append(f"{coeff:.2f}x")
                else:
                    terms.append(f"{coeff:.2f}x^{i}")
        
        if not terms:
            return "0"
        
        return " + ".join(terms).replace("+ -", "- ")
    
    def evaluate(self, x):
        """计算多项式在x处的值"""
        return np.polyval(self.coeffs[::-1], x)  # polyval需要从高次到低次

# 创建多项式向量空间
print("多项式向量空间示例:")
print("-" * 40)

p1 = Polynomial([1, 2, 3])    # 1 + 2x + 3x²
p2 = Polynomial([0, 1, -1])   # 0 + 1x - 1x²
p3 = Polynomial([2, 0, 0, 1]) # 2 + 0x + 0x² + 1x³

print(f"p1 = {p1}")
print(f"p2 = {p2}")
print(f"p3 = {p3}")
print()

# 验证向量空间操作
print("向量空间操作验证:")
print(f"p1 + p2 = {p1 + p2}")
print(f"p1 * 3 = {p1 * 3}")
print(f"零多项式 = {Polynomial([0])}")
print(f"p1 + 零多项式 = {p1 + Polynomial([0])}")
```
### 3. 函数向量空间
```python
print("\n" + "="*60)
print("=== 函数向量空间 ===")
print()

class FunctionSpace:
    """函数空间示例"""
    
    def __init__(self, functions):
        self.functions = functions
    
    def evaluate_at_points(self, x_points):
        """在多个点处计算所有函数的值"""
        results = {}
        for name, func in self.functions.items():
            results[name] = [func(x) for x in x_points]
        return results
    
    def check_linear_combination(self, coefficients):
        """验证线性组合仍在空间中（如果空间是线性空间）"""
        # 这里简化处理，只做演示
        print(f"线性组合: {coefficients}")
        return True

# 定义一些函数
def f1(x):
    return x

def f2(x):
    return x**2

def f3(x):
    return np.sin(x)

def f4(x):
    return np.exp(-x**2)

# 创建函数空间
function_space = {
    "f1(x)=x": f1,
    "f2(x)=x²": f2,
    "f3(x)=sin(x)": f3,
    "f4(x)=exp(-x²)": f4
}

fs = FunctionSpace(function_space)

# 在多个点处评估
x_points = np.linspace(-2, 2, 5)
results = fs.evaluate_at_points(x_points)

print("函数在多个点处的值:")
print(f"x点: {x_points}")
print("-" * 40)
for name, values in results.items():
    print(f"{name}: {values}")
```
### 4. 矩阵向量空间
```python
print("\n" + "="*60)
print("=== 矩阵向量空间 ===")
print()

class MatrixSpace:
    """所有m×n矩阵构成向量空间"""
    
    def __init__(self, m, n):
        self.m = m
        self.n = n
    
    def random_matrix(self):
        """生成随机矩阵"""
        return np.random.randn(self.m, self.n)
    
    def check_axioms(self, A, B, C, alpha, beta):
        """验证向量空间公理"""
        print("验证矩阵向量空间公理:")
        print("-" * 40)
        
        # 1. 加法封闭性
        print(f"1. 加法封闭性: A+B 是 {self.m}×{self.n} 矩阵 ✓")
        
        # 2. 加法交换律
        print(f"2. 交换律: A+B = B+A? {np.allclose(A+B, B+A)}")
        
        # 3. 加法结合律
        print(f"3. 结合律: (A+B)+C = A+(B+C)? {np.allclose((A+B)+C, A+(B+C))}")
        
        # 4. 零矩阵存在
        zero_matrix = np.zeros((self.m, self.n))
        print(f"4. 零矩阵: A+0 = A? {np.allclose(A+zero_matrix, A)}")
        
        # 5. 负矩阵存在
        print(f"5. 负矩阵: A+(-A) = 0? {np.allclose(A + (-A), zero_matrix)}")
        
        # 6. 标量乘法封闭性
        print(f"6. 标量乘法封闭性: αA 是 {self.m}×{self.n} 矩阵 ✓")
        
        # 7. 标量乘法分配律
        print(f"7. 分配律1: α(A+B) = αA+αB? {np.allclose(alpha*(A+B), alpha*A + alpha*B)}")
        print(f"8. 分配律2: (α+β)A = αA+βA? {np.allclose((alpha+beta)*A, alpha*A + beta*A)}")
        
        # 8. 标量乘法结合律
        print(f"9. 结合律: α(βA) = (αβ)A? {np.allclose(alpha*(beta*A), (alpha*beta)*A)}")
        
        # 9. 单位标量
        print(f"10. 单位标量: 1·A = A? {np.allclose(1*A, A)}")

# 创建2×3矩阵空间
ms = MatrixSpace(2, 3)

A = ms.random_matrix()
B = ms.random_matrix()
C = ms.random_matrix()
alpha = 2.5
beta = -1.3

ms.check_axioms(A, B, C, alpha, beta)
```
### 5. 子空间与基
```python
print("\n" + "="*60)
print("=== 子空间与基 ===")
print()

def check_subspace(vectors, verbose=True):
    """检查向量集合是否构成子空间"""
    vectors = np.array(vectors)
    
    if verbose:
        print("检查向量集合是否构成子空间:")
        print(f"向量: {vectors.tolist()}")
    
    # 1. 包含零向量
    zero_check = any(np.allclose(v, 0) for v in vectors)
    
    # 2. 加法封闭性（简化检查）
    if len(vectors) >= 2:
        add_check = True
        for i in range(len(vectors)):
            for j in range(len(vectors)):
                result = vectors[i] + vectors[j]
                # 检查结果是否可由原向量线性表示（简化）
                # 在实际中需要更严格的检查
                pass
    else:
        add_check = True
    
    # 3. 标量乘法封闭性
    scalar_check = True
    
    is_subspace = zero_check and add_check and scalar_check
    
    if verbose:
        print(f"包含零向量: {'✓' if zero_check else '✗'}")
        print(f"加法封闭性: {'✓' if add_check else '✗'}")
        print(f"标量乘法封闭性: {'✓' if scalar_check else '✗'}")
        print(f"是否构成子空间: {'是' if is_subspace else '否'}")
        print()
    
    return is_subspace

def find_basis(vectors):
    """寻找向量组的基"""
    vectors = np.array(vectors)
    
    # 使用SVD找线性无关的列
    U, s, Vh = np.linalg.svd(vectors.T)
    
    # 非零奇异值对应的向量
    rank = np.sum(s > 1e-10)
    basis_vectors = vectors[:rank]
    
    print(f"原始向量组: {len(vectors)} 个向量")
    print(f"秩（线性无关向量数）: {rank}")
    print(f"基向量:")
    for i, v in enumerate(basis_vectors):
        print(f"  v{i+1} = {v}")
    
    return basis_vectors

# 示例
vectors = [
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([1, 1, 0]),
    np.array([2, -1, 0])
]

print("R³中的向量组（都在xy平面）:")
check_subspace(vectors)

print("寻找基:")
basis = find_basis(vectors)
```
### 6. 无限维向量空间：傅里叶基
```python
print("\n" + "="*60)
print("=== 无限维向量空间：傅里叶基 ===")
print()

def fourier_basis_demo():
    """傅里叶基演示（函数空间的基）"""
    
    import matplotlib.pyplot as plt
    
    # 定义傅里叶基函数
    def fourier_basis(k, x):
        """傅里叶基函数：sin和cos"""
        if k == 0:
            return 1 / np.sqrt(2*np.pi)  # 常数项
        elif k % 2 == 1:
            n = (k + 1) // 2
            return np.sin(n * x) / np.sqrt(np.pi)
        else:
            n = k // 2
            return np.cos(n * x) / np.sqrt(np.pi)
    
    # 生成基函数
    x = np.linspace(-np.pi, np.pi, 1000)
    basis_functions = []
    
    print("傅里叶基函数（前5个）:")
    print("-" * 40)
    
    for k in range(5):
        y = np.array([fourier_basis(k, xi) for xi in x])
        basis_functions.append(y)
        
        if k == 0:
            print(f"k={k}: 常数项")
        elif k % 2 == 1:
            n = (k + 1) // 2
            print(f"k={k}: sin({n}x)")
        else:
            n = k // 2
            print(f"k={k}: cos({n}x)")
    
    # 可视化
    plt.figure(figsize=(12, 8))
    for k in range(5):
        plt.subplot(3, 2, k+1)
        y = basis_functions[k]
        plt.plot(x, y)
        plt.title(f"傅里叶基函数 k={k}")
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 验证正交性
    print("\n验证基函数的正交性:")
    print("-" * 40)
    
    # 数值积分验证内积
    dx = x[1] - x[0]
    for i in range(3):
        for j in range(3):
            if i <= j:
                fi = basis_functions[i]
                fj = basis_functions[j]
                inner_product = np.sum(fi * fj) * dx
                
                if i == j:
                    expected = 1.0  # 正交归一基
                else:
                    expected = 0.0
                
                print(f"⟨f{i}, f{j}⟩ = {inner_product:.6f} (期望: {expected})")
    
    return basis_funcs

# 运行傅里叶基演示
basis_funcs = fourier_basis_demo()
```
## 🎯 关键理解
### 抽象向量空间在ML中的四大应用
```python
abstract_space_applications = {
    "词向量空间": {
        "空间类型": "高维欧几里得空间 (Rⁿ)",
        "维度": "通常100-1000维",
        "应用": "word2vec, GloVe, BERT嵌入",
        "特点": "语义关系编码为几何关系"
    },
    "函数空间": {
        "空间类型": "无限维希尔伯特空间",
        "维度": "无限维",
        "应用": "核方法、高斯过程、神经网络",
        "特点": "通过核函数隐式定义"
    },
    "再生核希尔伯特空间(RKHS)": {
        "空间类型": "特殊的希尔伯特空间",
        "维度": "由核函数决定",
        "应用": "支持向量机(SVM)、核PCA",
        "特点": "再生性质：f(x) = ⟨f, K(x,·)⟩"
    },
    "流形学习": {
        "空间类型": "微分流形",
        "维度": "数据内在维度",
        "应用": "t-SNE, UMAP, 自编码器",
        "特点": "局部像欧几里得空间，全局复杂"
    }
}
```
## 📝 练习与思考
### 练习题
```python
print("=== 练习题 ===")
print()

# 练习1：验证向量空间公理
print("练习1：验证向量空间公理")
print("考虑所有2×2对称矩阵的集合：")
print("{ [[a, b], [b, c]] | a, b, c ∈ R }")
print("1. 验证这个集合是否构成向量空间")
print("2. 如果是，找出它的维数和一组基")
print()

# 练习2：多项式空间
print("练习2：多项式空间")
print("考虑所有次数≤2的多项式集合：")
print("{ a + bx + cx² | a, b, c ∈ R }")
print("1. 验证这是向量空间")
print("2. 找出标准基 {1, x, x²} 下的坐标表示")
print("3. 计算多项式 p(x)=2+3x-x² 在该基下的坐标")
print()

# 练习3：函数空间
print("练习3：函数空间")
print("考虑区间[0,1]上所有连续函数的集合 C[0,1]")
print("1. 解释为什么这是一个向量空间")
print("2. 这个空间的维数是多少？")
print("3. 能否找到一组有限基？为什么？")
```
### 思考题
#### 为什么说"所有神经网络层的集合构成一个向量空间"？
```text
提示：考虑层的加法和标量乘法
```
#### 在词向量空间中，为什么"国王 - 男人 + 女人 ≈ 女王"？
```text
提示：考虑向量空间中的线性关系
```
#### 无限维向量空间如何处理实际计算？
```text
提示：考虑有限维近似、基展开
```
## 🚀 下一步学习建议
### 你已经掌握了：
* ✅ 向量空间的抽象定义和8条公理
* ✅ 多种向量空间的例子（多项式、函数、矩阵）
* ✅ 子空间和基的概念
* ✅ 无限维向量空间的初步理解

### 第12集预告：克莱姆法则
```python
第12集联系 = {
    "与第11集的关系": "在线性方程组求解中的应用",
    "在ML中的应用": [
        "1. 线性回归的解析解",
        "2. 最小二乘法",
        "3. 理论理解线性系统"
    ],
    "重要性": "⭐⭐（可快速学习，实用价值有限）"
}
```
### 学习检查清单：
* 理解向量空间的8条公理

* 能举例说明不同的向量空间

* 理解有限维和无限维向量空间的区别

* 知道基和维数的概念

* 理解向量空间公理如何推广向量概念

#### 记住：抽象向量空间是理解现代机器学习数学基础的关键！ 🧠
