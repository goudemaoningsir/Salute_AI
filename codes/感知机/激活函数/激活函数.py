import numpy as np
import matplotlib.pyplot as plt


# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


# 定义输入范围
x = np.linspace(-5, 5, 100)

# 画图
plt.figure(figsize=(6, 4))

# Sigmoid函数图像
plt.plot(x, sigmoid(x))
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))

# ReLU函数图像
plt.plot(x, relu(x))
plt.title('ReLU Function')
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))

# Tanh函数图像
plt.plot(x, tanh(x))
plt.title('Tanh Function')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))

# Leaky ReLU函数图像
plt.plot(x, leaky_relu(x))
plt.title('Leaky ReLU Function')
plt.xlabel('x')
plt.ylabel('leaky_relu(x)')
plt.grid(True)
plt.show()
