# 🎬 **第5集：三维线性变换**

## 📺 视频信息
- **视频标题**：Three-dimensional linear transformations
- **3B1B原视频**：https://www.youtube.com/watch?v=rHLEWRxRGiM
- **B站中文字幕**：https://www.bilibili.com/video/BV1ys411472E?p=5

---

## 🎯 核心概念

### 1. **从2D到3D的扩展**
- 2D线性变换：作用在平面上的点
- 3D线性变换：作用在空间中的点
- 基本性质不变：保持直线、保持原点

### 2. **3D线性变换矩阵**
3×3矩阵的每一列代表：
- 第1列：î (x轴单位向量) 变换后的位置
- 第2列：ĵ (y轴单位向量) 变换后的位置  
- 第3列：k̂ (z轴单位向量) 变换后的位置

### 3. **常见的3D变换**
- 3D旋转（绕x、y、z轴）
- 3D缩放（均匀/不均匀）
- 3D剪切
- 3D投影

---

## 💻 代码实现

### 1. **创建3D变换矩阵**
```python
import numpy as np

# 3D单位矩阵（恒等变换）
I_3d = np.eye(3)
print("3D单位矩阵（什么都不做）:")
print(I_3d)

# 3D缩放矩阵
def scale_3d(sx, sy, sz):
    """3D缩放变换"""
    return np.array([[sx, 0, 0],
                     [0, sy, 0],
                     [0, 0, sz]])

# 3D旋转矩阵
def rotate_x(theta):
    """绕x轴旋转"""
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def rotate_y(theta):
    """绕y轴旋转"""
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def rotate_z(theta):
    """绕z轴旋转"""
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

# 3D剪切矩阵
def shear_3d(kxy, kxz, kyx, kyz, kzx, kzy):
    """3D剪切变换"""
    return np.array([[1, kxy, kxz],
                     [kyx, 1, kyz],
                     [kzx, kzy, 1]])
```

### 2. **应用3D变换**
```python
def apply_3d_transformation():
    """应用3D变换示例"""
    
    # 定义3D点（单位立方体的8个顶点）
    cube = np.array([
        [0, 0, 0],  # 顶点0
        [1, 0, 0],  # 顶点1
        [1, 1, 0],  # 顶点2
        [0, 1, 0],  # 顶点3
        [0, 0, 1],  # 顶点4
        [1, 0, 1],  # 顶点5
        [1, 1, 1],  # 顶点6
        [0, 1, 1]   # 顶点7
    ])
    
    print("单位立方体顶点:")
    print(cube)
    
    # 应用缩放变换
    S = scale_3d(2, 1.5, 0.8)
    cube_scaled = cube @ S.T
    
    print(f"\n缩放矩阵 S(2, 1.5, 0.8):")
    print(S)
    print(f"\n缩放后的立方体:")
    print(cube_scaled)
    
    # 应用旋转变换
    R = rotate_x(np.pi/4)  # 绕x轴旋转45度
    cube_rotated = cube @ R.T
    
    print(f"\n绕x轴旋转45度矩阵:")
    print(np.round(R, 4))
    print(f"\n旋转后的立方体（前4个顶点）:")
    print(np.round(cube_rotated[:4], 4))
    
    return cube, cube_scaled, cube_rotated

cube, cube_scaled, cube_rotated = apply_3d_transformation()
```

### 3. **3D可视化**
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_transformations():
    """可视化3D变换"""
    
    # 创建单位立方体
    cube = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # 底面
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # 顶面
    ])
    
    # 定义边（连接哪些顶点）
    edges = [
        [0,1], [1,2], [2,3], [3,0],  # 底面
        [4,5], [5,6], [6,7], [7,4],  # 顶面
        [0,4], [1,5], [2,6], [3,7]   # 侧面
    ]
    
    # 创建三种变换
    # 1. 原始立方体
    # 2. 缩放变换
    S = scale_3d(2, 1.5, 0.8)
    cube_scaled = cube @ S.T
    
    # 3. 旋转变换
    R = rotate_y(np.pi/6) @ rotate_x(np.pi/4)  # 复合旋转
    cube_rotated = cube @ R.T
    
    # 绘制
    fig = plt.figure(figsize=(15, 5))
    
    titles = ["原始立方体", "缩放变换 (2, 1.5, 0.8)", "旋转变换 (绕y30°+绕x45°)"]
    cubes = [cube, cube_scaled, cube_rotated]
    
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # 绘制边
        for edge in edges:
            points = cubes[i][edge]
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=2)
        
        # 绘制顶点
        ax.scatter(cubes[i][:, 0], cubes[i][:, 1], cubes[i][:, 2], 
                  c='red', s=50, alpha=0.8)
        
        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1, 2.5])
        ax.set_ylim([-1, 2.5])
        ax.set_zlim([-1, 2.5])
        
        # 设置视角
        ax.view_init(elev=20, azim=30*i)
    
    plt.suptitle('3D线性变换可视化', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()

# 运行可视化
visualize_3d_transformations()
```

### 4. **3D变换复合**
```python
def demonstrate_3d_composition():
    """演示3D变换复合"""
    
    print("=== 3D变换复合演示 ===")
    
    # 定义三个变换
    S = scale_3d(2, 1, 1)      # x方向拉伸2倍
    R_y = rotate_y(np.pi/4)    # 绕y轴旋转45度
    R_x = rotate_x(np.pi/6)    # 绕x轴旋转30度
    
    # 测试点
    point = np.array([1, 0, 0])  # x轴上的点
    
    print(f"原始点: {point}")
    
    # 不同的复合顺序
    # 顺序1: 先缩放，再绕y旋转，最后绕x旋转
    composite1 = R_x @ R_y @ S
    result1 = composite1 @ point
    
    # 顺序2: 先绕x旋转，再绕y旋转，最后缩放
    composite2 = S @ R_y @ R_x
    result2 = composite2 @ point
    
    print(f"\n顺序1 (缩放→绕y→绕x):")
    print(f"  复合矩阵: \n{np.round(composite1, 4)}")
    print(f"  变换后点: {np.round(result1, 4)}")
    
    print(f"\n顺序2 (绕x→绕y→缩放):")
    print(f"  复合矩阵: \n{np.round(composite2, 4)}")
    print(f"  变换后点: {np.round(result2, 4)}")
    
    print(f"\n两个结果相等吗？ {np.allclose(result1, result2)}")
    print("💡 3D变换一般不满足交换律！")
    
    return composite1, composite2

composite1, composite2 = demonstrate_3d_composition()
```

### 5. **3D投影变换**
```python
def projection_transforms():
    """3D投影变换"""
    
    print("=== 3D投影变换 ===")
    
    # 正交投影到xy平面（丢弃z坐标）
    P_xy = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]])
    
    # 正交投影到xz平面
    P_xz = np.array([[1, 0, 0],
                     [0, 0, 0],
                     [0, 0, 1]])
    
    # 正交投影到yz平面
    P_yz = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
    
    # 测试点
    points = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    
    print(f"原始3D点:")
    print(points)
    
    print(f"\n投影到xy平面:")
    print(points @ P_xy.T)
    
    print(f"\n投影到xz平面:")
    print(points @ P_xz.T)
    
    print(f"\n投影到yz平面:")
    print(points @ P_yz.T)
    
    # 透视投影（简化版）
    # 将3D点投影到z=1的平面
    def perspective_projection(points, focal_length=1):
        """简单的透视投影"""
        projected = np.zeros((len(points), 2))
        for i, (x, y, z) in enumerate(points):
            # 避免除以零
            if z + focal_length != 0:
                scale = focal_length / (z + focal_length)
                projected[i] = [x * scale, y * scale]
        return projected
    
    print(f"\n透视投影（焦距=1）到2D:")
    print(perspective_projection(points))

projection_transforms()
```

---

## 🧮 数学原理

### **3D线性变换的一般形式**
对于 $3 \times 3$ 矩阵 $M$ 和 3D 向量 $\mathbf{v} = [x, y, z]^T$：

$$
M\mathbf{v} = \begin{bmatrix}
m_{11} & m_{12} & m_{13} \\
m_{21} & m_{22} & m_{23} \\
m_{31} & m_{32} & m_{33}
\end{bmatrix}
\begin{bmatrix} x \\ y \\ z \end{bmatrix}
= \begin{bmatrix}
m_{11}x + m_{12}y + m_{13}z \\
m_{21}x + m_{22}y + m_{23}z \\
m_{31}x + m_{32}y + m_{33}z
\end{bmatrix}
$$

### 旋转矩阵推导

绕 $z$ 轴旋转（在 $xy$ 平面旋转）：

$$
R_z(\theta) = 
\begin{bmatrix}
\cos\theta & -\sin\theta & 0 \\
\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

绕其他轴的旋转类似，只是影响的平面不同。

---

## 🎯 **关键理解**

### **从2D到3D的类比**
| 概念 | 2D | 3D |
|------|----|----|
| 基向量 | î, ĵ | î, ĵ, k̂ |
| 矩阵形状 | 2×2 | 3×3 |
| 变换类型 | 旋转、缩放、剪切 | 旋转、缩放、剪切、投影 |
| 可视化 | 平面网格 | 空间网格/立方体 |

### **3D旋转的复合**
绕不同轴的旋转**不满足交换律**：
- 先绕x转30°，再绕y转45° ≠ 先绕y转45°，再绕x转30°
- 这对应了飞行器姿态的"万向节锁"问题

### **右手坐标系**
3D图形学通常使用**右手坐标系**：
- x轴：向右
- y轴：向上
- z轴：向屏幕外

---

## 📝 **练习与思考**

### **练习题**
1. 创建绕x轴旋转90度的矩阵，验证它交换y和z坐标
2. 计算 `scale_3d(2, 3, 4) @ rotate_x(π/2)` 和 `rotate_x(π/2) @ scale_3d(2, 3, 4)`，观察是否相等
3. 创建一个将立方体变为平行六面体的剪切变换
4. 验证3D单位矩阵的恒等性质

### **思考题**
1. 为什么3D旋转需要三个基本旋转矩阵（绕x、y、z轴）？
2. 3D投影变换的秩是多少？这意味着什么？
3. 如何判断一个3×3矩阵是否表示可逆的线性变换？
4. 在机器学习中，3D变换有哪些应用场景？

---

## 🚀 **下一步学习建议**

### **立即练习：**
```python
# 练习1：验证旋转矩阵的性质
R = rotate_x(np.pi/2)
print(f"绕x轴旋转90度:")
print(R)

# 验证：[0,1,0]应该变成[0,0,1]
v = np.array([0, 1, 0])
print(f"\n旋转[0,1,0]: {R @ v}")
print(f"应该是[0,0,1]吗？ {np.allclose(R @ v, [0,0,1])}")
```

### **连接应用：**
- **计算机图形学**：3D模型变换、相机视图
- **机器人学**：机械臂运动学、姿态控制
- **计算机视觉**：3D重建、点云处理
- **物理学**：刚体动力学、坐标变换

---
