import numpy as np
import matplotlib.pyplot as plt


# 定义目标函数
def f(x, y):
    return 8 * x**2 - 2 * y


# 定义约束函数
def g(x, y):
    return x**2 + y**2 - 1


# 定义梯度计算函数
def gradient(f, x, y, h=0.001):
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return df_dx, df_dy


# 定义几个关键点和圆周上的点
points = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (np.sqrt(2) / 2, np.sqrt(2) / 2),
    (-np.sqrt(2) / 2, -np.sqrt(2) / 2),
]

# 计算这些点的梯度
gradients_f = []
for point in points:
    df_dx, df_dy = gradient(f, point[0], point[1])
    gradients_f.append((df_dx, df_dy))

# 计算圆周上点的梯度
theta_vals = np.linspace(0, 2 * np.pi, 9)
circle_points = [(np.cos(theta), np.sin(theta)) for theta in theta_vals]

gradients_g = []
for point in circle_points:
    df_dx, df_dy = gradient(g, point[0], point[1])
    gradients_g.append((df_dx, df_dy))

# 绘制图像
plt.figure(figsize=(8, 8))

# 绘制目标函数的等值线
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)
contour_levels = [-5, -2, -1, 0, 1, 2, 5, 8, 10]
contour = plt.contour(X, Y, Z, levels=contour_levels, cmap="viridis")
plt.clabel(contour, inline=True, fontsize=8)

# 绘制约束条件 x^2 + y^2 = 1 的圆
theta = np.linspace(0, 2 * np.pi, 400)
x_circle = np.cos(theta)
y_circle = np.sin(theta)
plt.plot(x_circle, y_circle, "b", label="$x^2 + y^2 = 1$")

# 标记关键点的梯度箭头
for i, point in enumerate(points):
    plt.plot(point[0], point[1], "ro")
    plt.text(
        point[0] + 0.1, point[1] + 0.1, f"({point[0]:.1f}, {point[1]:.1f})", color="red"
    )
    dx, dy = gradients_f[i]
    plt.arrow(
        point[0],
        point[1],
        dx * 0.1,
        dy * 0.1,
        head_width=0.1,
        head_length=0.1,
        fc="green",
        ec="green",
    )

# 标记圆周上点的梯度箭头
for i, point in enumerate(circle_points):
    dx, dy = gradients_g[i]
    plt.arrow(
        point[0],
        point[1],
        dx * 0.1,
        dy * 0.1,
        head_width=0.1,
        head_length=0.1,
        fc="blue",
        ec="blue",
    )

# 图像设置
plt.title("Gradient Vectors at Key Points")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.gca().set_aspect("equal", adjustable="box")

# 显示图像
plt.show()
