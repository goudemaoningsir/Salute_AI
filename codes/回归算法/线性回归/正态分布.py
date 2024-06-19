import math
import numpy as np
import matplotlib.pyplot as plt


def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu) ** 2)


# 定义x轴数据
x = np.arange(-7, 7, 0.01)

# 均值和标准差组合
params = [(0, 1), (0, 2), (3, 1)]

# 绘制正态分布图
for mu, sigma in params:
    plt.plot(x, normal(x, mu, sigma), label=f"mean {mu}, std {sigma}")
plt.xlabel("x")
plt.ylabel("p(x)")
plt.legend()
plt.show()
