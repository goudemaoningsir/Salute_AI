import numpy as np
import matplotlib.pyplot as plt


# 定义目标函数
def f(x1, x2):
    return x1**2 + x2**2


# 定义约束函数
def g(x1, x2):
    return x1**2 + x2**2 - 1


# 生成网格点
x1 = np.linspace(-1.5, 1.5, 400)
x2 = np.linspace(-1.5, 1.5, 400)
X1, X2 = np.meshgrid(x1, x2)

# 计算函数值
F = f(X1, X2)
G = g(X1, X2)

# 绘制等高线
plt.figure(figsize=(8, 8))
contours = plt.contour(X1, X2, F, levels=np.logspace(-2, 0, 20), cmap="viridis")
plt.clabel(contours, inline=True, fontsize=8)
plt.contour(X1, X2, G, levels=[0], colors="red", linewidths=2)

# 绘制可行域
plt.fill_between(x1, np.sqrt(1 - x1**2), -np.sqrt(1 - x1**2), color="gray", alpha=0.3)

# 添加标签和标题
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Objective Function and Constraint")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.show()
