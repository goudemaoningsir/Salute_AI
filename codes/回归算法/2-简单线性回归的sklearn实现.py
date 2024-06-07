import numpy as np
import matplotlib.pyplot as plt


class SimpleLinearRegression:
    def __init__(self):
        self.a_ = 0
        self.b_ = 0

    def fit(self, X, y):
        # 断言检查
        assert X.shape[0] == y.shape[0], "X and y should have the same length"
        assert X.size > 0 and y.size > 0, "X and y should not be empty"

        # 计算均值
        x_mean = np.mean(X)
        y_mean = np.mean(y)

        # 计算向量化后的 a 和 b
        self.a_ = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean) ** 2)
        self.b_ = y_mean - self.a_ * x_mean

    def predict(self, X):
        return self.a_ * X + self.b_

    def get_params(self):
        return self.a_, self.b_


# 定义数据点
X = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])

# 创建线性回归模型实例
model = SimpleLinearRegression()

# 拟合模型
model.fit(X, y)

# 打印参数 a 和 b
a, b = model.get_params()
print(f"斜率 a: {a}")
print(f"截距 b: {b}")

# 预测 y 值
y_pred = model.predict(X)

# 绘制数据点和回归线
plt.scatter(X, y, color='blue', label='数据点')
plt.plot(X, y_pred, color='red', label='回归线')
plt.title('简单线性回归')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.axis([0, 6, 0, 6])

# 显示图形
plt.show()
