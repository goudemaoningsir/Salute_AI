import numpy as np

# 定义矩阵
X = np.random.normal(0, 1, (3, 1))
W_xh = np.random.normal(0, 1, (1, 4))
H = np.random.normal(0, 1, (3, 4))
W_hh = np.random.normal(0, 1, (4, 4))

# 分别计算矩阵乘法并相加
output1 = np.dot(X, W_xh) + np.dot(H, W_hh)

# 拼接矩阵并计算乘法
X_H = np.concatenate((X, H), axis=1)
W_xh_W_hh = np.concatenate((W_xh, W_hh), axis=0)
output2 = np.dot(X_H, W_xh_W_hh)

print(output1)
print(output2)
