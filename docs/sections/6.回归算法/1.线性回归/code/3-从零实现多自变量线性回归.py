import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


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
    # 加载糖尿病数据集
    diabetes = load_diabetes()
    X = diabetes.data  # 所有特征
    y = diabetes.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)

    # 预测
    y_pred = model.predict(X_test)

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
