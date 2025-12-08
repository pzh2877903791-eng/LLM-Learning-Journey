# 🎬 **第8集：点积与对偶性**

## 📺 视频信息
- **视频标题**：Dot products and duality
- **3B1B原视频**：https://www.youtube.com/watch?v=LyGKycYT2v0
- **B站中文字幕**：https://www.bilibili.com/video/BV1ys411472E?p=8

---

## 🎯 核心概念

### 1. **点积的两种视角**
- **代数定义**： `a·b = Σ aᵢbᵢ` (i=1到n)
- **几何定义**： `a·b = |a|·|b|·cosθ`

### 2. **对偶性（Duality）**
- 每个向量对应一个线性变换
- 点积 `a·b` = 将 `b` 投影到 `a` 方向并缩放
- 矩阵乘法的特例：`A×b` 当 `A` 是 `1×n` 矩阵时

### 3. **在LLM中的重要性**
- **注意力机制的核心**：`Q·Kᵀ` 就是点积！
- **相似度计算**：词向量相似度用点积/余弦相似度
- **对偶性**：每个词向量可以看作一个"评分函数"

---

## 💻 代码实现（无可视化）

### 1. **点积的基础计算**
```python
import numpy as np

print("=== 点积的基本计算 ===")
print()

# 定义两个向量
v1 = np.array([2, 3])
v2 = np.array([4, 1])

print(f"向量 v1 = {v1}")
print(f"向量 v2 = {v2}")
print()

# 方法1：使用numpy的dot函数
dot_np = np.dot(v1, v2)
print(f"1. np.dot(v1, v2) = {dot_np}")

# 方法2：使用@运算符（Python 3.5+）
dot_at = v1 @ v2
print(f"2. v1 @ v2 = {dot_at}")

# 方法3：手动计算
dot_manual = sum(v1[i] * v2[i] for i in range(len(v1)))
print(f"3. 手动计算 Σ(v1[i]*v2[i]) = {dot_manual}")
print()

# 验证三种方法结果相同
assert abs(dot_np - dot_at) < 1e-10
assert abs(dot_np - dot_manual) < 1e-10
print("✅ 三种方法结果一致")
```
### 2. 点积的几何意义
```python
print("\n" + "="*60)
print("=== 点积的几何意义 ===")
print()

def dot_product_geometry(v1, v2):
    """计算点积及其几何分量"""
    
    print(f"向量1: {v1}, 长度: {np.linalg.norm(v1):.4f}")
    print(f"向量2: {v2}, 长度: {np.linalg.norm(v2):.4f}")
    
    # 计算点积
    dot = v1 @ v2
    
    # 计算夹角余弦
    cos_theta = dot / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    # 计算夹角（弧度）
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta_deg = np.degrees(theta_rad)
    
    print(f"\n点积 v1·v2 = {dot:.4f}")
    print(f"cosθ = {cos_theta:.4f}")
    print(f"夹角 θ = {theta_rad:.4f} 弧度 = {theta_deg:.2f}°")
    
    # 投影长度
    proj_length = dot / np.linalg.norm(v1)
    print(f"v2 在 v1 方向上的投影长度 = {proj_length:.4f}")
    
    return dot, cos_theta, theta_deg

# 测试不同情况的向量
test_cases = [
    ("同方向", np.array([3, 0]), np.array([2, 0])),
    ("垂直", np.array([3, 0]), np.array([0, 2])),
    ("相反方向", np.array([3, 0]), np.array([-1, 0])),
    ("45度角", np.array([1, 0]), np.array([1, 1])),
]

for name, v1, v2 in test_cases:
    print(f"\n{'='*40}")
    print(f"测试: {name}")
    dot, cos_theta, theta_deg = dot_product_geometry(v1, v2)
    
    # 验证公式：||v1|| * ||v2|| * cosθ = 点积
    left_side = np.linalg.norm(v1) * np.linalg.norm(v2) * cos_theta
    print(f"验证 ||v1||*||v2||*cosθ = {left_side:.6f}, 点积 = {dot:.6f}")
    print(f"是否相等? {np.allclose(left_side, dot)}")
```
### 3. 对偶性理解
```python
print("\n" + "="*60)
print("=== 对偶性：向量 ↔ 线性变换 ===")
print()

def demonstrate_duality():
    """演示对偶性"""
    
    print("核心思想：每个向量对应一个线性变换")
    print()
    
    # 定义一个向量
    u = np.array([2, 3])
    print(f"向量 u = {u}")
    print()
    
    # 这个向量u定义了一个线性变换：点积
    # 变换：f(x) = u·x
    # 这个变换可以表示为一个矩阵：把u当作行向量
    
    # 创建变换矩阵（1×2矩阵，就是u作为行向量）
    A = u.reshape(1, -1)  # 形状: (1, 2)
    print(f"对应的线性变换矩阵 A (1×2):")
    print(A)
    print()
    
    # 测试这个变换
    test_vectors = [np.array([1, 0]), np.array([0, 1]), np.array([2, 1])]
    
    print("测试变换 f(x) = u·x = A @ x:")
    for x in test_vectors:
        # 方法1：直接点积
        result_dot = u @ x
        
        # 方法2：矩阵乘法
        result_matmul = A @ x  # 注意：x是列向量
        
        print(f"  x = {x}")
        print(f"    u·x = {result_dot}")
        print(f"    A @ x = {result_matmul[0]}")
        print(f"    是否相等? {np.allclose(result_dot, result_matmul[0])}")
        print()
    
    print("结论：")
    print("1. 向量 u 定义了一个线性变换 f(x) = u·x")
    print("2. 这个变换可以用矩阵 A = [u₁, u₂] 表示")
    print("3. 这就是对偶性：向量 ↔ 线性变换")
    
    return u, A

u_example, A_example = demonstrate_duality()
```
### 4. 点积在LLM中的应用：注意力机制
```python
print("\n" + "="*60)
print("=== 点积在Transformer注意力中的应用 ===")
print()

def attention_dot_product_example():
    """注意力机制中的点积"""
    
    print("Transformer注意力公式：")
    print("Attention(Q, K, V) = softmax(QK^T/√d) V")
    print("其中 QK^T 就是点积！")
    print()
    
    # 简化示例：3个词，每个词2维向量
    print("示例：3个词的句子，每个词用2维向量表示")
    
    # 词向量（3个词，每个2维）
    words = np.array([
        [1.0, 0.5],  # 词1
        [0.5, 1.0],  # 词2  
        [0.8, 0.8],  # 词3
    ])
    
    print("词向量矩阵（3×2）：")
    print(words)
    print()
    
    # 简化的Q、K、V投影（实际中是不同的权重矩阵）
    # 这里为了简化，假设Q=K=V=原始词向量
    Q = words  # 查询
    K = words  # 键
    V = words  # 值
    
    print("1. 计算 QK^T（点积注意力分数）")
    print("QK^T = Q @ K.T")
    print()
    
    # 计算注意力分数矩阵
    d_k = Q.shape[1]  # 向量维度
    scores = Q @ K.T  # 形状: (3, 3)
    
    print(f"注意力分数矩阵（形状 {scores.shape}）：")
    print(scores)
    print()
    
    print("解释：scores[i, j] = 词i对词j的关注度")
    print("例如：")
    for i in range(3):
        for j in range(3):
            print(f"  词{i+1}·词{j+1} = {scores[i, j]:.3f}", end="  ")
        print()
    print()
    
    # 缩放
    scaled_scores = scores / np.sqrt(d_k)
    print(f"2. 缩放：除以 √d_k = √{d_k} ≈ {np.sqrt(d_k):.3f}")
    print("缩放后的分数：")
    print(scaled_scores)
    print()
    
    # softmax
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    attention_weights = softmax(scaled_scores)
    print(f"3. Softmax后的注意力权重：")
    print(attention_weights)
    print()
    
    # 验证每行和为1
    print("验证每行和为1：")
    for i in range(3):
        row_sum = np.sum(attention_weights[i])
        print(f"  行{i}和 = {row_sum:.6f} {'✓' if np.allclose(row_sum, 1.0) else '✗'}")
    print()
    
    # 输出
    output = attention_weights @ V
    print(f"4. 最终输出 = 注意力权重 @ V：")
    print("形状:", output.shape)
    print("值:")
    print(output)
    
    return scores, attention_weights, output

# 运行示例
scores_example, weights_example, output_example = attention_dot_product_example()
```
### 5. 余弦相似度 vs 点积
```python
print("\n" + "="*60)
print("=== 余弦相似度与点积的关系 ===")
print()

def cosine_vs_dot_product():
    """比较余弦相似度和点积"""
    
    print("在LLM中，词向量相似度常用余弦相似度")
    print("但注意力机制用点积，为什么？")
    print()
    
    # 定义几个词向量
    word_vectors = {
        "king": np.array([0.8, 0.3]),
        "queen": np.array([0.7, 0.4]),
        "man": np.array([0.9, 0.1]),
        "woman": np.array([0.8, 0.2]),
        "apple": np.array([0.1, 0.9]),
    }
    
    print("词向量：")
    for word, vec in word_vectors.items():
        print(f"  {word:6s}: {vec}, 长度: {np.linalg.norm(vec):.3f}")
    print()
    
    # 计算两种相似度
    def cosine_similarity(a, b):
        """余弦相似度 = (a·b) / (||a|| * ||b||)"""
        return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    words = list(word_vectors.keys())
    
    print("相似度矩阵：")
    print("行/列: ", "  ".join(words))
    print()
    
    print("点积相似度：")
    for w1 in words:
        row = []
        for w2 in words:
            dot = word_vectors[w1] @ word_vectors[w2]
            row.append(f"{dot:.3f}")
        print(f"{w1:6s}: {'  '.join(row)}")
    print()
    
    print("余弦相似度：")
    for w1 in words:
        row = []
        for w2 in words:
            cos_sim = cosine_similarity(word_vectors[w1], word_vectors[w2])
            row.append(f"{cos_sim:.3f}")
        print(f"{w1:6s}: {'  '.join(row)}")
    print()
    
    print("关键观察：")
    print("1. 余弦相似度消除了向量长度的影响")
    print("2. 点积同时考虑了方向和长度")
    print("3. 在注意力中，长度可能包含重要信息")
    print("   （比如：重要词可能有更长的向量）")
    print("4. 这就是为什么Transformer用点积而不是余弦")
    
    # 特殊情况：长度归一化后的点积 = 余弦相似度
    print("\n验证：长度归一化后的点积 = 余弦相似度")
    w1, w2 = "king", "queen"
    v1, v2 = word_vectors[w1], word_vectors[w2]
    
    # 归一化
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    dot_norm = v1_norm @ v2_norm
    cos_sim = cosine_similarity(v1, v2)
    
    print(f"{w1}·{w2}（归一化后） = {dot_norm:.6f}")
    print(f"余弦相似度 = {cos_sim:.6f}")
    print(f"是否相等? {np.allclose(dot_norm, cos_sim)}")
    
    return word_vectors

word_vecs = cosine_vs_dot_product()
```
### 6. 对偶性的高级理解
```python
print("\n" + "="*60)
print("=== 对偶性在神经网络中的应用 ===")
print()

def duality_in_neural_networks():
    """神经网络中的对偶性"""
    
    print("神经网络中的对偶性例子：")
    print("=" * 50)
    print()
    
    # 例子1：全连接层
    print("1. 全连接层中的对偶性")
    print("   权重矩阵 W 的每一行可以看作：")
    print("   • 一个线性变换的系数")
    print("   • 或者：一个'模板向量'")
    print()
    
    # 模拟一个简单的全连接层
    input_dim = 3
    output_dim = 2
    
    # 权重矩阵
    W = np.array([
        [0.5, -0.2, 0.1],   # 第一个输出神经元的权重
        [0.1, 0.3, -0.4]    # 第二个输出神经元的权重
    ])
    
    print(f"权重矩阵 W（形状 {W.shape}）：")
    print(W)
    print()
    
    # 输入向量
    x = np.array([1.0, 0.5, -0.2])
    print(f"输入 x = {x}")
    print()
    
    # 前向传播
    output = W @ x
    print(f"输出 = W @ x = {output}")
    print()
    
    print("解释：")
    print(f"  输出[0] = {output[0]:.3f} = W[0]·x")
    print(f"         = {W[0]} · {x}")
    print(f"         = {W[0][0]}×{x[0]} + {W[0][1]}×{x[1]} + {W[0][2]}×{x[2]}")
    print()
    print(f"  输出[1] = {output[1]:.3f} = W[1]·x")
    print()
    
    print("对偶性：")
    print("  • 每个输出神经元对应一个向量（W的一行）")
    print("  • 输出 = 输入与这些向量的点积")
    print("  • 每个向量定义了一个'特征检测器'")
    print()
    
    # 例子2：词嵌入中的对偶性
    print("2. 词嵌入中的对偶性")
    print()
    
    # 假设的嵌入矩阵
    vocab_size = 5
    embedding_dim = 3
    
    # 嵌入矩阵：每行是一个词的向量
    embedding_matrix = np.array([
        [0.8, 0.2, 0.1],  # 词0
        [0.1, 0.9, 0.3],  # 词1
        [0.4, 0.5, 0.8],  # 词2
        [0.7, 0.1, 0.6],  # 词3
        [0.2, 0.3, 0.9],  # 词4
    ])
    
    print(f"嵌入矩阵（{vocab_size}个词，每个{embedding_dim}维）：")
    print(embedding_matrix)
    print()
    
    print("对偶性视角：")
    print("  每个词向量有两种理解方式：")
    print("  1. 作为一个'点'（在嵌入空间中的位置）")
    print("  2. 作为一个'函数'（用于计算相似度）")
    print()
    
    # 查询向量（比如从上下文计算得来）
    query = np.array([0.5, 0.3, 0.7])
    print(f"查询向量（上下文表示）: {query}")
    print()
    
    # 计算与所有词的相似度
    similarities = embedding_matrix @ query
    
    print("与每个词的点积相似度：")
    for i in range(vocab_size):
        sim = similarities[i]
        print(f"  词{i}: {sim:.3f} = {embedding_matrix[i]} · {query}")
    print()
    
    # 找出最相似的词
    most_similar_idx = np.argmax(similarities)
    print(f"最相似的词: 词{most_similar_idx}, 相似度: {similarities[most_similar_idx]:.3f}")
    print()
    
    print("总结：")
    print("  词向量既是'表示'也是'评分函数'")
    print("  这就是对偶性在NLP中的体现！")
    
    return W, embedding_matrix

W_fc, embedding_mat = duality_in_neural_networks()
```
### 7. 点积的性质和运算
```python
print("\n" + "="*60)
print("=== 点积的重要性质 ===")
print()

def dot_product_properties():
    """点积的数学性质"""
    
    print("点积的基本性质：")
    print("=" * 40)
    print()
    
    # 测试向量
    a = np.array([2, 3])
    b = np.array([4, 1])
    c = np.array([1, 2])
    scalar = 3
    
    print(f"测试向量：")
    print(f"  a = {a}")
    print(f"  b = {b}")
    print(f"  c = {c}")
    print(f"  标量 k = {scalar}")
    print()
    
    # 性质1：交换律
    print("1. 交换律：a·b = b·a")
    ab = a @ b
    ba = b @ a
    print(f"   a·b = {ab}")
    print(f"   b·a = {ba}")
    print(f"   是否相等? {np.allclose(ab, ba)}")
    print()
    
    # 性质2：分配律
    print("2. 分配律：a·(b + c) = a·b + a·c")
    left = a @ (b + c)
    right = (a @ b) + (a @ c)
    print(f"   a·(b + c) = {a}·{b + c} = {left}")
    print(f"   a·b + a·c = {ab} + {a @ c} = {right}")
    print(f"   是否相等? {np.allclose(left, right)}")
    print()
    
    # 性质3：标量乘法
    print("3. 标量乘法：(ka)·b = k(a·b) = a·(kb)")
    left1 = (scalar * a) @ b
    middle = scalar * (a @ b)
    right1 = a @ (scalar * b)
    print(f"   (ka)·b = ({scalar}×{a})·{b} = {left1}")
    print(f"   k(a·b) = {scalar}×({ab}) = {middle}")
    print(f"   a·(kb) = {a}·({scalar}×{b}) = {right1}")
    print(f"   是否都相等? {np.allclose([left1, middle, right1], left1)}")
    print()
    
    # 性质4：与自身点积 = 长度的平方
    print("4. a·a = ||a||²")
    dot_aa = a @ a
    norm_squared = np.linalg.norm(a) ** 2
    print(f"   a·a = {a}·{a} = {dot_aa}")
    print(f"   ||a||² = {np.linalg.norm(a):.3f}² = {norm_squared}")
    print(f"   是否相等? {np.allclose(dot_aa, norm_squared)}")
    print()
    
    # 性质5：正交向量点积为0
    print("5. 正交（垂直）向量：a·b = 0 当 a ⟂ b")
    orthogonal_a = np.array([1, 0])
    orthogonal_b = np.array([0, 2])
    orth_dot = orthogonal_a @ orthogonal_b
    print(f"   {orthogonal_a}·{orthogonal_b} = {orth_dot}")
    print(f"   是否正交? {np.allclose(orth_dot, 0)}")
    print()
    
    # 性质6：柯西-施瓦茨不等式
    print("6. 柯西-施瓦茨不等式：|a·b| ≤ ||a|| ||b||")
    abs_dot = abs(a @ b)
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    print(f"   |a·b| = |{ab}| = {abs_dot}")
    print(f"   ||a|| ||b|| = {np.linalg.norm(a):.3f} × {np.linalg.norm(b):.3f} = {norm_product:.3f}")
    print(f"   不等式成立? {abs_dot <= norm_product + 1e-10}")
    print()
    
    # 性质7：点积与夹角的关系
    print("7. 点积与夹角：a·b = ||a|| ||b|| cosθ")
    theta = np.arccos((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    left_side = a @ b
    right_side = np.linalg.norm(a) * np.linalg.norm(b) * np.cos(theta)
    print(f"   a·b = {left_side}")
    print(f"   ||a|| ||b|| cosθ = {right_side}")
    print(f"   夹角 θ = {np.degrees(theta):.2f}°")
    print(f"   是否相等? {np.allclose(left_side, right_side)}")
    
    return a, b, c

vec_a, vec_b, vec_c = dot_product_properties()
```
## 🎯 关键理解
### 点积在LLM中的三大应用
```python
llm_applications = {
    "注意力机制": {
        "公式": "Attention(Q,K,V) = softmax(QK^T/√d_k)V",
        "解释": "QK^T就是查询和键的点积矩阵",
        "重要性": "⭐⭐⭐⭐⭐ Transformer的核心"
    },
    "词向量相似度": {
        "公式": "相似度 = (v1·v2)/(||v1|| ||v2||) (余弦相似度)",
        "解释": "点积标准化后得到余弦相似度",
        "重要性": "⭐⭐⭐⭐ 词义计算的基础"
    },
    "全连接层": {
        "公式": "输出[i] = W[i]·输入",
        "解释": "每个神经元的输出是权重向量与输入的点积",
        "重要性": "⭐⭐⭐ 神经网络的基本运算"
    }
}
```
### 对偶性的核心思想
```text
对偶性：两种等价的理解方式

1. 向量观点：
   - 每个词是一个向量
   - 相似度 = 向量间的点积
   - 注意力 = 查询向量与键向量的点积

2. 线性变换观点：
   - 每个词向量定义一个线性函数
   - f(x) = 词向量·x
   - 这个函数给其他词"打分"

本质：向量 ↔ 线性变换 是对偶的
```
## 📝 练习与思考
### 练习题
```python
print("=== 练习题 ===")
print()

# 练习1：手动计算点积
print("练习1：计算点积")
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"请手动计算 v1·v2，然后用代码验证")
print()

# 练习2：理解注意力中的点积
print("练习2：注意力分数计算")
Q = np.array([[1, 0.5], [0.5, 1]])
K = np.array([[1, 0], [0, 1]])
print(f"Q = \n{Q}")
print(f"K = \n{K}")
print(f"计算 QK^T，理解每个元素的意义")
print()

# 练习3：余弦相似度
print("练习3：余弦相似度")
word1 = np.array([3, 4])   # 长度=5
word2 = np.array([6, 8])   # 长度=10
print(f"word1 = {word1}, 长度 = {np.linalg.norm(word1)}")
print(f"word2 = {word2}, 长度 = {np.linalg.norm(word2)}")
print("1. 计算点积相似度")
print("2. 计算余弦相似度")
print("3. 解释为什么两者不同")
```
### 思考题
#### 为什么Transformer用点积而不是余弦相似度计算注意力？

```text
提示：考虑向量的长度是否包含信息
```
#### 对偶性如何帮助我们理解词向量的作用？

```text
提示：词向量既是"表示"也是"函数"
```
#### 点积的交换律在注意力机制中意味着什么？

```text
提示：QK^T不是对称的，但每个元素是点积
```
## 🚀 下一步学习建议
### 你已经掌握了：
✅ 点积的两种定义（代数/几何）
✅ 对偶性的核心思想
✅ 点积在注意力机制中的应用
✅ 点积与余弦相似度的关系

### 第9集预告：特征向量与特征值
```python
第9集联系 = {
    "与第8集的关系": "特征向量是特殊的向量，点积运算会用到",
    "在LLM中的应用": [
        "1. 分析权重矩阵的稳定性",
        "2. PCA降维（基于特征值分解）", 
        "3. 理解训练中的梯度问题"
    ],
    "重要性": "⭐⭐⭐⭐⭐（实际工作中常用）"
}
```
### 学习检查清单：
* 能计算任意两个向量的点积
* 理解点积的几何意义（投影）
* 理解对偶性：向量 ↔ 线性变换
* 知道注意力机制中QK^T是点积
* 能区分点积相似度和余弦相似度

#### 记住：点积是注意力机制的核心，对偶性是理解向量作用的双重视角。这两个概念对理解LLM至关重要！
