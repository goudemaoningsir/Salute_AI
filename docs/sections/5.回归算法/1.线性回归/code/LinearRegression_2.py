import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. 生成数据
np.random.seed(42)
n_samples = 100

# 自变量 x1 和 x2
x1 = np.random.rand(n_samples, 1) * 10
x2 = np.random.rand(n_samples, 1) * 10

# 目标变量 y = 3x1 + 5x2 + 噪声
noise = np.random.randn(n_samples, 1)
y = 3 * x1 + 5 * x2 + noise

# 拼接自变量
X = np.hstack([x1, x2])

# 展示生成的数据
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(x1, y, color='blue', label='Data (x1 vs y)')
plt.xlabel('x1')
plt.ylabel('y')
plt.legend()
plt.title('Generated Data (x1 vs y)')

plt.subplot(1, 2, 2)
plt.scatter(x2, y, color='green', label='Data (x2 vs y)')
plt.xlabel('x2')
plt.ylabel('y')
plt.legend()
plt.title('Generated Data (x2 vs y)')

plt.show()

# 2. 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练多元线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 评估模型
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# 3. 绘制回归结果
plt.figure(figsize=(12, 6))

# 训练数据集 x1 与 y 的关系
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], y_train, color='blue', label='Training data (x1 vs y)')
plt.scatter(X_train[:, 0], y_train_pred, color='red', label='Fitted data (x1 vs y)')
# 绘制回归线
x1_line = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100).reshape(-1, 1)
x2_mean = np.full_like(x1_line, X_train[:, 1].mean())  # 使用 x2 的平均值
X_line = np.hstack([x1_line, x2_mean])
y_line = model.predict(X_line)
plt.plot(x1_line, y_line, color='black', label='Regression line (x1)')
plt.xlabel('x1')
plt.ylabel('y')
plt.legend()
plt.title('Training data and Fitted data (x1 vs y)')

# 测试数据集 x1 与 y 的关系
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], y_test, color='blue', label='Testing data (x1 vs y)')
plt.scatter(X_test[:, 0], y_test_pred, color='red', label='Predicted data (x1 vs y)')
plt.plot(x1_line, y_line, color='black', label='Regression line (x1)')
plt.xlabel('x1')
plt.ylabel('y')
plt.legend()
plt.title('Testing data and Predicted data (x1 vs y)')

plt.show()

# 训练数据集 x2 与 y 的关系
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 1], y_train, color='green', label='Training data (x2 vs y)')
plt.scatter(X_train[:, 1], y_train_pred, color='orange', label='Fitted data (x2 vs y)')
# 绘制回归线
x2_line = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100).reshape(-1, 1)
x1_mean = np.full_like(x2_line, X_train[:, 0].mean())  # 使用 x1 的平均值
X_line = np.hstack([x1_mean, x2_line])
y_line = model.predict(X_line)
plt.plot(x2_line, y_line, color='black', label='Regression line (x2)')
plt.xlabel('x2')
plt.ylabel('y')
plt.legend()
plt.title('Training data and Fitted data (x2 vs y)')

# 测试数据集 x2 与 y 的关系
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 1], y_test, color='green', label='Testing data (x2 vs y)')
plt.scatter(X_test[:, 1], y_test_pred, color='orange', label='Predicted data (x2 vs y)')
plt.plot(x2_line, y_line, color='black', label='Regression line (x2)')
plt.xlabel('x2')
plt.ylabel('y')
plt.legend()
plt.title('Testing data and Predicted data (x2 vs y)')

plt.show()
