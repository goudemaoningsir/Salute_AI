import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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

# 拟合线性回归模型
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)
y_pred_linear = linear_regressor.predict(X)
mse_linear = mean_squared_error(y, y_pred_linear)
r2_linear = r2_score(y, y_pred_linear)
print(f'Linear Regression MSE: {mse_linear}')
print(f'Linear Regression R^2: {r2_linear}')

# 拟合多项式回归模型
# 创建多项式特征转换器，degree 表示多项式的最高次数
poly_features = PolynomialFeatures(degree=3)
# 将原始特征 X 转换为多项式特征 X_poly
X_poly = poly_features.fit_transform(X)

# 创建线性回归模型
poly_regressor = LinearRegression()

# 使用多项式特征的线性回归模型拟合数据
poly_regressor.fit(X_poly, y)

# 使用训练好的多项式回归模型进行预测
y_pred_poly = poly_regressor.predict(X_poly)

# 计算多项式回归模型的均方误差 (Mean Squared Error, MSE)
mse_poly = mean_squared_error(y, y_pred_poly)

# 计算多项式回归模型的决定系数 (R^2 score)
r2_poly = r2_score(y, y_pred_poly)

# 输出多项式回归模型的评估结果
print(f'Polynomial Regression MSE: {mse_poly}')
print(f'Polynomial Regression R^2: {r2_poly}')


# 可视化结果
# 对 X 进行排序
sorted_X = np.sort(X, axis=0)
sorted_X_poly = poly_features.transform(sorted_X)
sorted_y_pred_linear = linear_regressor.predict(sorted_X)
sorted_y_pred_poly = poly_regressor.predict(sorted_X_poly)

plt.scatter(X, y, s=10, label='Data')
plt.plot(sorted_X, sorted_y_pred_linear, color='r', label='Linear Regression')
plt.plot(sorted_X, sorted_y_pred_poly, color='g', label='Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
