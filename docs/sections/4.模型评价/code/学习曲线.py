import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 设置随机种子，确保结果可重复
np.random.seed(0)

# 生成自变量 X，包含100个数据点
x = 2 - 3 * np.random.uniform(-3.0, 3.0, 100)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)

# 将 X 转换为二维数组，符合 Scikit-Learn 的输入格式
X = x[:, np.newaxis]
y = y[:, np.newaxis]


# 将数据集分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def plot_learning_curve(model, X_train, y_train, X_val, y_val):
    train_errors, val_errors = [], []

    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Training error")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation error")
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.legend()


# 欠拟合
# 创建线性回归模型
linear_regressor = LinearRegression()

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plot_learning_curve(linear_regressor, X_train, y_train, X_val, y_val)
plt.title("Learning Curve (Underfitting) - Linear Regression")
plt.show()

# 过拟合
# 创建多项式特征转换器和线性回归模型
poly_degree_high = 15
poly_features_high = PolynomialFeatures(degree=poly_degree_high)
X_train_poly_high = poly_features_high.fit_transform(X_train)
X_val_poly_high = poly_features_high.transform(X_val)

poly_regressor_high = LinearRegression()

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plot_learning_curve(poly_regressor_high, X_train_poly_high, y_train, X_val_poly_high, y_val)
plt.title("Learning Curve (Overfitting) - Polynomial Regression (degree=15)")
plt.show()

# 正常拟合
# 创建多项式特征转换器和线性回归模型
poly_degree_optimal = 3
poly_features_optimal = PolynomialFeatures(degree=poly_degree_optimal)
X_train_poly_optimal = poly_features_optimal.fit_transform(X_train)
X_val_poly_optimal = poly_features_optimal.transform(X_val)

poly_regressor_optimal = LinearRegression()

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plot_learning_curve(poly_regressor_optimal, X_train_poly_optimal, y_train, X_val_poly_optimal, y_val)
plt.title("Learning Curve (Good Fit) - Polynomial Regression (degree=3)")
plt.show()
