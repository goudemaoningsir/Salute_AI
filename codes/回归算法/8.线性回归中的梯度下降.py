import numpy as np
from sklearn.metrics import r2_score


class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None  # 保存线性回归模型的系数
        self.intercept_ = None  # 保存线性回归模型的截距
        self._theta = None  # 保存线性回归模型的参数

    def fit_normal(self, X_train, y_train):
        """根据训练数据集X_train, y_train训练Linear Regression模型（使用正规方程）"""
        assert X_train.shape[0] == y_train.shape[0], \
            "训练数据X_train的样本数量必须等于y_train的样本数量"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])  # 添加偏置项
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)  # 计算参数theta

        self.intercept_ = self._theta[0]  # 截距
        self.coef_ = self._theta[1:]  # 系数

        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "训练数据X_train的样本数量必须等于y_train的样本数量"

        def J(theta, X_b, y):
            """计算代价函数"""
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            """计算代价函数的梯度"""
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            """梯度下降法求解"""
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if np.linalg.norm(theta - last_theta) < epsilon:
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])  # 添加偏置项
        initial_theta = np.zeros(X_b.shape[1])  # 初始化theta
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)  # 使用梯度下降法求解theta

        self.intercept_ = self._theta[0]  # 截距
        self.coef_ = self._theta[1:]  # 系数

        return self

    def fit_sgd(self, X_train, y_train, n_iters=5, t0=5, t1=50):
        """根据训练数据集X_train, y_train, 使用随机梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "训练数据X_train的样本数量必须等于y_train的样本数量"
        assert n_iters >= 1, "迭代次数必须大于等于1"

        def dJ_sgd(theta, X_b_i, y_i):
            """计算单个样本的梯度"""
            return X_b_i * (X_b_i.dot(theta) - y_i) * 2.

        def sgd(X_b, y, initial_theta, n_iters, t0=5, t1=50):
            """随机梯度下降法求解"""

            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)

            for cur_iter in range(n_iters):
                indexes = np.random.permutation(m)  # 打乱样本顺序
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter * m + i) * gradient

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])  # 添加偏置项
        initial_theta = np.random.randn(X_b.shape[1])  # 初始化theta
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)  # 使用随机梯度下降法求解theta

        self.intercept_ = self._theta[0]  # 截距
        self.coef_ = self._theta[1:]  # 系数

        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "必须先进行拟合再预测！"
        assert X_predict.shape[1] == len(self.coef_), \
            "X_predict的特征数量必须等于X_train的特征数量"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])  # 添加偏置项
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"


# 示例使用
if __name__ == "__main__":
    # 生成一些示例数据
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # 创建并训练模型
    lin_reg = LinearRegression()
    lin_reg.fit_gd(X, y)

    # 打印模型参数
    print(f"截距（intercept_）：{lin_reg.intercept_}")
    print(f"系数（coef_）：{lin_reg.coef_}")

    # 预测
    X_new = np.array([[0], [2]])
    y_predict = lin_reg.predict(X_new)
    print(f"预测结果：{y_predict}")

    # 计算模型的准确度
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lin_reg.fit_gd(X_train, y_train)
    score = lin_reg.score(X_test, y_test)
    print(f"模型的R^2分数：{score}")
