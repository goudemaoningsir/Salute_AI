import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 创建逻辑回归模型
model = LogisticRegression(max_iter=200)

# 使用交叉验证评估模型
scores = cross_val_score(model, X, y, cv=5)  # 5折交叉验证

print("Cross-Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())