from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# 创建示例数据
X = np.array([[2, 3],
              [3, 4],
              [4, 5]])

# 创建 PolynomialFeatures 实例，并指定 degree（多项式的最高次幂）
poly = PolynomialFeatures(degree=2)

# 使用 fit_transform 方法转换原始特征
X_poly = poly.fit_transform(X)

print("Original features:\n", X)
print("Polynomial features:\n", X_poly)