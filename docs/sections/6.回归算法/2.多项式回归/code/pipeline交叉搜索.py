from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# 设置随机种子，确保结果可重复
np.random.seed(0)

# 生成自变量 X，包含100个数据点
X = 2 - 3 * np.random.normal(0, 1, 100)

# 生成因变量 y，具有非线性关系
y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 3, 100)

# 将 X 转换为二维数组，符合 Scikit-Learn 的输入格式
X = X[:, np.newaxis]
y = y[:, np.newaxis]

# 创建包含多项式特征生成和线性回归模型的 Pipeline
pipeline = Pipeline([
    ('poly_features', PolynomialFeatures()),
    ('std_scaler', StandardScaler()),
    ('linear_regression', LinearRegression())
])

# 定义参数网格
param_grid = {
    'poly_features__degree': [2, 3, 4, 5],  # 多项式次数
    'linear_regression__fit_intercept': [True, False],  # 是否拟合截距
}

# 创建 GridSearchCV 对象
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')

# 使用 GridSearchCV 进行模型训练
grid_search.fit(X, y)

# 输出最佳参数和最佳评分
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best R^2 score: {grid_search.best_score_}')