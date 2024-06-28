import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 创建留一法交叉验证器
loo = LeaveOneOut()

# 创建逻辑回归模型
model = LogisticRegression(max_iter=200)

accuracies = []

# 进行留一法交叉验证
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 训练模型
    model.fit(X_train, y_train)

    # 预测并计算准确率
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

print("LOOCV Accuracies:", accuracies)
print("Mean Accuracy:", np.mean(accuracies))
