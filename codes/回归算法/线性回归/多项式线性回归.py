import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，以便结果可复现
np.random.seed(0)

# 生成数据
X = np.linspace(-5, 5, 100)
y_true = np.sin(X) + np.random.normal(0, 0.2, 100)

# 绘制原始数据
plt.figure(figsize=(10, 6))
plt.scatter(X, y_true, s=20, color="blue", label="Original Data")
plt.title("Generated Data for Polynomial Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 将X转换为二维数组，以符合scikit-learn的要求
X = X.reshape(-1, 1)


# 用于展示不同多项式次数的函数
def polynomial_regression(degree):
    # 多项式特征扩展
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # 线性回归模型
    model = LinearRegression()
    model.fit(X_poly, y_true)

    # 预测结果
    y_pred = model.predict(X_poly)

    # 计算均方误差
    mse = mean_squared_error(y_true, y_pred)

    # 绘制拟合曲线
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y_true, s=20, color="blue", label="Original Data")
    plt.plot(X, y_pred, color="red", label=f"Fitted Curve (Degree={degree})")
    plt.title(f"Polynomial Regression (Degree={degree})")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.text(
        0.8,
        0.1,
        f"MSE: {mse:.4f}",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    plt.show()


# 尝试不同次数的多项式回归
polynomial_regression(degree=1)  # 一次多项式
polynomial_regression(degree=3)  # 三次多项式
polynomial_regression(degree=10)  # 十次多项式
