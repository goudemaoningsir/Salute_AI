from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 定义模型
rf = RandomForestClassifier()

# 定义超参数空间
param_grid = {"n_estimators": [50, 100, 200], "max_features": ["auto", "sqrt", "log2"]}

# 网格搜索选择最优超参数
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 输出最优超参数组合
print("Best Parameters:", grid_search.best_params_)

# 使用最优超参数训练模型
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# 在测试集上评估模型
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
