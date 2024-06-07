from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 加载糖尿病数据集
diabetes = load_diabetes()
print(diabetes.DESCR)
print(diabetes.feature_names)
X = diabetes.data[:, 2]  # 只取一个特征
print(X.shape)
y = diabetes.target
print(y.shape)
plt.scatter(X, y)
plt.show()

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.2, random_state=42)

# 使用sklearn的线性回归模型进行训练
model = LinearRegression()
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 绘制数据点和回归线
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Number of Rooms')
plt.ylabel('House Price')
plt.title('Linear Regression on Boston Housing Data (Using sklearn)')
plt.legend()
plt.show()

# 计算均方误差（MSE）、均方根误差（RMSE）和平均绝对误差（MAE）
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 打印评估指标
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")
