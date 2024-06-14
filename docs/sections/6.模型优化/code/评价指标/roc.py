import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 生成二分类数据集
X, y_true = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# 使用随机森林模型预测概率值
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y_true)
y_scores = clf.predict_proba(X)[:, 1]

# 计算TPR和FPR
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# 计算AUC
roc_auc = auc(fpr, tpr)
print(f'AUC: {roc_auc:.2f}')

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='.', label=f'ROC Curve (AUC={roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()