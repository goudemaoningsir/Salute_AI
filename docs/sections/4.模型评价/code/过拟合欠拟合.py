import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 设置随机种子，确保结果可重复
np.random.seed(0)

# 生成自变量 X，包含100个数据点
X = 2 - 3 * np.random.normal(0, 1, 100)

# 生成因变量 y，具有非线性关系
y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 3, 100)

# 将 X 转换为二维数组，符合 Scikit-Learn 的输入格式
X = X[:, np.newaxis]
y = y[:, np.newaxis]

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 欠拟合示例
# 创建线性回归模型
linear_regressor = LinearRegression()

# 使用线性回归模型拟合数据
linear_regressor.fit(X_train, y_train)

# 使用训练好的线性回归模型进行预测
y_train_pred = linear_regressor.predict(X_train)
y_test_pred = linear_regressor.predict(X_test)

# 计算线性回归模型的均方误差和 R^2 分数
mse_train_linear = mean_squared_error(y_train, y_train_pred)
mse_test_linear = mean_squared_error(y_test, y_test_pred)
r2_train_linear = r2_score(y_train, y_train_pred)
r2_test_linear = r2_score(y_test, y_test_pred)

# 输出线性回归模型的评估结果
print(f'Linear Regression Train MSE: {mse_train_linear}')
print(f'Linear Regression Train R^2: {r2_train_linear}')
print(f'Linear Regression Test MSE: {mse_test_linear}')
print(f'Linear Regression Test R^2: {r2_test_linear}')

# 可视化线性回归结果
sorted_X_train = np.sort(X_train, axis=0)
sorted_y_train_pred = linear_regressor.predict(sorted_X_train)
plt.scatter(X_train, y_train, s=10, label='Train Data')
plt.plot(sorted_X_train, sorted_y_train_pred, color='r', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression (Underfitting)')
plt.legend()
plt.show()

# 过拟合示例
# 创建多项式特征转换器，degree 表示多项式的最高次数
poly_features = PolynomialFeatures(degree=10)

# 将原始特征 X 转换为多项式特征
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# 创建线性回归模型
poly_regressor = LinearRegression()

# 使用多项式特征的线性回归模型拟合数据
poly_regressor.fit(X_train_poly, y_train)

# 使用训练好的多项式回归模型进行预测
y_train_pred_poly = poly_regressor.predict(X_train_poly)
y_test_pred_poly = poly_regressor.predict(X_test_poly)

# 计算多项式回归模型的均方误差和 R^2 分数
mse_train_poly = mean_squared_error(y_train, y_train_pred_poly)
mse_test_poly = mean_squared_error(y_test, y_test_pred_poly)
r2_train_poly = r2_score(y_train, y_train_pred_poly)
r2_test_poly = r2_score(y_test, y_test_pred_poly)

# 输出多项式回归模型的评估结果
print(f'Polynomial Regression Train MSE: {mse_train_poly}')
print(f'Polynomial Regression Train R^2: {r2_train_poly}')
print(f'Polynomial Regression Test MSE: {mse_test_poly}')
print(f'Polynomial Regression Test R^2: {r2_test_poly}')

# 可视化多项式回归结果
sorted_X_train_poly = poly_features.transform(np.sort(X_train, axis=0))
sorted_y_train_pred_poly = poly_regressor.predict(sorted_X_train_poly)
plt.scatter(X_train, y_train, s=10, label='Train Data')
plt.plot(np.sort(X_train, axis=0), sorted_y_train_pred_poly, color='g', label='Polynomial Regression (degree=10)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression (Overfitting)')
plt.legend()
plt.show()


# 使用正则化方法解决过拟合
from sklearn.linear_model import Ridge

# 创建多项式特征转换器，degree 表示多项式的最高次数
poly_features = PolynomialFeatures(degree=10)

# 将原始特征 X 转换为多项式特征
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# 创建 Ridge 回归模型，设置正则化参数 alpha
ridge_regressor = Ridge(alpha=1.0)

# 使用 Ridge 回归模型拟合数据
ridge_regressor.fit(X_train_poly, y_train)

# 使用训练好的 Ridge 回归模型进行预测
y_train_pred_ridge = ridge_regressor.predict(X_train_poly)
y_test_pred_ridge = ridge_regressor.predict(X_test_poly)

# 计算 Ridge 回归模型的均方误差和 R^2 分数
mse_train_ridge = mean_squared_error(y_train, y_train_pred_ridge)
mse_test_ridge = mean_squared_error(y_test, y_test_pred_ridge)
r2_train_ridge = r2_score(y_train, y_train_pred_ridge)
r2_test_ridge = r2_score(y_test, y_test_pred_ridge)

# 输出 Ridge 回归模型的评估结果
print(f'Ridge Regression Train MSE: {mse_train_ridge}')
print(f'Ridge Regression Train R^2: {r2_train_ridge}')
print(f'Ridge Regression Test MSE: {mse_test_ridge}')
print(f'Ridge Regression Test R^2: {r2_test_ridge}')

# 可视化 Ridge 回归结果
sorted_X_train_poly = poly_features.transform(np.sort(X_train, axis=0))
sorted_y_train_pred_ridge = ridge_regressor.predict(sorted_X_train_poly)
plt.scatter(X_train, y_train, s=10, label='Train Data')
plt.plot(np.sort(X_train, axis=0), sorted_y_train_pred_ridge, color='b', label='Ridge Regression (degree=10)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Ridge Regression (Regularization)')
plt.legend()
plt.show()
