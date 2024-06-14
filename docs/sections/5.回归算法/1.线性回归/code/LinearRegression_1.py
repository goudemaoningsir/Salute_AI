import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. 生成数据
np.random.seed(42)
n_samples = 100

# 自变量 x
x = np.random.rand(n_samples, 1) * 10

# 目标变量 y = 3x + 5 + 噪声
noise = np.random.randn(n_samples, 1)
y = 3 * x + 5 + noise

# 展示生成的数据
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Generated data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Generated Data (x vs y)')
plt.show()

# 2. 分割数据集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 训练一元线性回归模型
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
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Testing data')
plt.plot(x, model.predict(x), color='red', label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Training and Testing Data with Regression Line')
plt.show()
