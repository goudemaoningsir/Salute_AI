import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 设置随机种子，确保结果可重复
np.random.seed(0)

# 生成自变量 X，包含100个数据点
X = 2 - 3 * np.random.normal(0, 1, 100)

# 生成因变量 y，具有非线性关系
y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 3, 100)

# 将 X 转换为二维数组，符合 Scikit-Learn 的输入格式
X = X[:, np.newaxis]
y = y[:, np.newaxis]

# 创建包含多项式特征生成和线性回归模型的 Pipeline
poly_degree = 3
pipeline = Pipeline([
    ('poly_features', PolynomialFeatures(degree=poly_degree)),
    ('std_scaler', StandardScaler()),
    ('linear_regression', LinearRegression())
])

# 使用 Pipeline 拟合数据
pipeline.fit(X, y)

# 使用 Pipeline 进行预测
y_pred = pipeline.predict(X)

# 计算模型评估指标
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# 输出评估结果
print(f'Polynomial Regression (degree={poly_degree}) MSE: {mse}')
print(f'Polynomial Regression (degree={poly_degree}) R^2: {r2}')

# 对 X 进行排序
sorted_X = np.sort(X, axis=0)
sorted_y_pred = pipeline.predict(sorted_X)

plt.scatter(X, y, s=10, label='Data')
plt.plot(sorted_X, sorted_y_pred, color='g', label='Polynomial Regression (degree={poly_degree})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()