import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 生成示例数据
np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 100)
y = X - 2 * (X**2) + np.random.normal(-3, 3, 100)

# 拟合线性回归模型
model = LinearRegression()
X = X[:, np.newaxis]
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 绘图
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X, y_pred, color="red", linewidth=2, label="Linear fit")
plt.legend()
plt.title("High Bias Model (Underfitting)")
plt.show()
