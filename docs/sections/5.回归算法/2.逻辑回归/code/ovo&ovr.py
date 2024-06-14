import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score

# 加载 Iris 数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用 One-vs-Rest 进行多分类
ovr_classifier = OneVsRestClassifier(LogisticRegression())
ovr_classifier.fit(X_train, y_train)

# 预测
y_pred_ovr = ovr_classifier.predict(X_test)

# 评估
accuracy_ovr = accuracy_score(y_test, y_pred_ovr)
print(f'OVR Accuracy: {accuracy_ovr}')

# 使用 One-vs-One 进行多分类
ovo_classifier = OneVsOneClassifier(LogisticRegression())
ovo_classifier.fit(X_train, y_train)

# 预测
y_pred_ovo = ovo_classifier.predict(X_test)

# 评估
accuracy_ovo = accuracy_score(y_test, y_pred_ovo)
print(f'OVO Accuracy: {accuracy_ovo}')
