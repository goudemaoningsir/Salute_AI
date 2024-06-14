import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# 生成示例数据
np.random.seed(0)
X = 2 - 3 * np.random.uniform(-3.0, 3.0, 100)
y = 0.5 * X ** 2 + X + 2 + np.random.normal(0, 1, 100)

# 将数据转换为二维数组
X = X[:, np.newaxis]

# 对数据进行排序，以便于绘制
sorted_indices = np.argsort(X[:, 0])
X_sorted = X[sorted_indices]
y_sorted = y[sorted_indices]

# 定义高次多项式回归模型
poly_degree = 15
model = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())

# 拟合模型
model.fit(X_sorted, y_sorted)

# 预测
y_pred = model.predict(X_sorted)

# 绘图
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_sorted, y_pred, color='red', linewidth=2, label=f'Polynomial fit (degree={poly_degree})')
plt.legend()
plt.title('High Variance Model (Overfitting)')
plt.show()
