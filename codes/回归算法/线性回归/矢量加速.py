import torch
from time import time

a = torch.ones(1000)
b = torch.ones(1000)

########################### 将这两个向量按元素逐一做标量加法
start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print(time() - start)

########################### 将这两个向量直接做矢量加法
start = time()
d = a + b
print(time() - start)

########################### 加法运算使用了广播机制
a = torch.ones(3)
b = 10
print(a + b)
