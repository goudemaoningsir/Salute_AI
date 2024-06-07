import numpy as np
import matplotlib.pyplot as plt

# 定义数据点
x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])

# 绘制数据点
plt.scatter(x, y)
plt.axis([0, 6, 0, 6])
plt.title('数据点分布')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 计算均值
x_mean = np.mean(x)
y_mean = np.mean(y)

# 计算 Sxy 和 Sxx
# Sxy = Σ(xy) - n * x_mean * y_mean
Sxy = np.sum(x * y) - len(x) * x_mean * y_mean
# Sxx = Σ(x^2) - n * x_mean^2
Sxx = np.sum(x ** 2) - len(x) * x_mean ** 2

# 计算斜率 a 和截距 b
a = Sxy / Sxx
b = y_mean - a * x_mean

# 打印参数 a 和 b
print(f"斜率 a: {a}")
print(f"截距 b: {b}")

# 计算回归线的 y 值
y_pred = a * x + b

# 绘制数据点和回归线
plt.scatter(x, y, color='blue', label='数据点')
plt.plot(x, y_pred, color='red', label='回归线')
plt.title('简单线性回归')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.axis([0, 6, 0, 6])

# 显示图形
plt.show()
