import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve, diff

# 定义变量
x, y, lambda1, lambda2, lambda3, lambda4, lambda5 = symbols(
    "x y lambda1 lambda2 lambda3 lambda4 lambda5"
)

# 定义目标函数和约束条件
f = 8 * x**2 - 2 * y
g1 = x**2 + y**2 - 1
g2 = x + y - 1
g3 = x - y
g4 = x - 0.5
g5 = y - 0.5

# 定义拉格朗日函数
L1 = f + lambda1 * g1
L2 = f + lambda2 * g2
L3 = f + lambda3 * g3
L4 = f + lambda4 * g4
L5 = f + lambda5 * g5

# 求解方程组
gradients = []
for L, lambd in zip(
    [L1, L2, L3, L4, L5], [lambda1, lambda2, lambda3, lambda4, lambda5]
):
    grad_x = Eq(diff(L, x), 0)
    grad_y = Eq(diff(L, y), 0)
    grad_lambda = Eq(diff(L, lambd), 0)

    solutions = solve([grad_x, grad_y, grad_lambda], (x, y, lambd))
    gradients.append(solutions)


# 计算梯度
def compute_gradient(func, x_val, y_val):
    grad_x = diff(func, x).subs([(x, x_val), (y, y_val)])
    grad_y = diff(func, y).subs([(x, x_val), (y, y_val)])
    return np.array([float(grad_x), float(grad_y)])


# 绘制函数和约束条件
def plot_with_constraints(constraint, constraint_label, solutions):
    x_vals = np.linspace(-2, 2, 400)
    y_vals = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = 8 * X**2 - 2 * Y

    plt.figure(figsize=(10, 10))
    contour_levels = [-5, -2, -1, 0, 1, 2, 5, 8, 10]
    contour = plt.contour(X, Y, Z, levels=contour_levels, cmap="viridis")
    plt.clabel(contour, inline=True, fontsize=8)

    if constraint == g1:
        theta = np.linspace(0, 2 * np.pi, 400)
        x_circle = np.cos(theta)
        y_circle = np.sin(theta)
        plt.plot(x_circle, y_circle, "b", label=constraint_label)
    elif constraint == g2:
        y_line = 1 - x_vals
        plt.plot(x_vals, y_line, "g", label=constraint_label)
    elif constraint == g3:
        plt.plot(x_vals, x_vals, "r", label=constraint_label)
    elif constraint == g4:
        plt.axvline(0.5, color="orange", linestyle="--", label=constraint_label)
    elif constraint == g5:
        plt.axhline(0.5, color="purple", linestyle="--", label=constraint_label)

    # 标记关键点及其梯度
    for sol in solutions:
        x_val = float(sol[0])
        y_val = float(sol[1])
        plt.plot(x_val, y_val, "ro")
        plt.text(x_val + 0.1, y_val + 0.1, f"({x_val:.2f}, {y_val:.2f})", color="red")

        grad_f = compute_gradient(f, x_val, y_val)
        plt.arrow(
            x_val,
            y_val,
            grad_f[0] * 0.1,
            grad_f[1] * 0.1,
            head_width=0.05,
            head_length=0.1,
            fc="red",
            ec="red",
        )

        grad_g = compute_gradient(constraint, x_val, y_val)
        plt.arrow(
            x_val,
            y_val,
            grad_g[0] * 0.1,
            grad_g[1] * 0.1,
            head_width=0.05,
            head_length=0.1,
            fc="blue",
            ec="blue",
        )

    plt.title(f"Function Contour with Constraint: {constraint_label}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


# 绘制每个约束条件下的图
constraints = [
    (g1, "$x^2 + y^2 = 1$"),
    (g2, "$x + y = 1$"),
    (g3, "$x - y = 0$"),
    (g4, "$x = 0.5$"),
    (g5, "$y = 0.5$"),
]
for i, (constraint, label) in enumerate(constraints):
    plot_with_constraints(constraint, label, gradients[i])
