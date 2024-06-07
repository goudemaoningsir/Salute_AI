import numpy as np


class LinearRegression:
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y):
        # 添加截距项
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        # 计算伪逆
        pseudo_inv = np.linalg.pinv(X.T @ X) @ X.T
        # 计算回归系数
        self.coef_ = pseudo_inv @ y
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def predict(self, X):
        # 添加截距项
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ np.hstack((self.intercept_, self.coef_))


# 示例用法
if __name__ == "__main__":
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    y = np.array([1, 2, 2.5, 4, 5])

    model = LinearRegression()
    model.fit(X, y)
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)

    X_test = np.array([[6, 6], [7, 7]])
    y_pred = model.predict(X_test)
    print("Predictions:", y_pred)
