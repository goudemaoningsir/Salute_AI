from sklearn.model_selection import train_test_split
import numpy as np


if __name__ == "__main__":
    # 创建示例数据
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])

    # 使用scikit-learn的train_test_split分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # 打印分割结果
    print("X_train:\n", X_train)
    print("X_test:\n", X_test)
    print("y_train:\n", y_train)
    print("y_test:\n", y_test)
