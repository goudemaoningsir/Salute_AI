# !/usr/bin/env python
# -*- coding:utf-8 -*-　
# @Time : 2023/6/8 19:53 
# @Author : sanmaomashi
# @GitHub : https://github.com/sanmaomashi
# @Summary : 数据特征预处理

print("======================== MinMax 归一化 ========================")
from sklearn.preprocessing import MinMaxScaler

# 原始数据
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

# 创建 MinMaxScaler 的实例
scaler = MinMaxScaler()

# 拟合并转换数据
print(scaler.fit_transform(data))
print("======================== MinMax 归一化 ========================")
print("======================== 单位长度归一化 ========================")

from sklearn.preprocessing import Normalizer

# 原始数据
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

# 创建 Normalizer 的实例
scaler = Normalizer()

# 拟合并转换数据
print(scaler.fit_transform(data))
print("======================== 单位长度归一化 ========================")


print("======================== 标准化 ========================")
from sklearn.preprocessing import StandardScaler

# 原始数据
data = [[0, 0], [0, 0], [1, 1], [1, 1]]

# 创建 StandardScaler 的实例
scaler = StandardScaler()

# 拟合并转换数据
print(scaler.fit_transform(data))

print("======================== 标准化========================")
print("======================== 缺失值========================")
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# 创建一个包含缺失值的 DataFrame
df = pd.DataFrame({'A':[1, 2, np.nan], 'B':[4, np.nan, np.nan], 'C':[7, 8, 9]})

print("原始数据：")
print(df)

# 使用均值填充缺失值
imp = SimpleImputer(strategy='mean')
df_filled = imp.fit_transform(df)

print("填充后的数据：")
print(df_filled)

print("======================== 缺失值========================")
print("======================== PCA ========================")
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 创建一个PCA对象，设置保留的主成分数量为2
pca = PCA(n_components=2)

# 对数据进行PCA降维
X_reduced = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[y == 0, 0], X_reduced[y == 0, 1], color='red', alpha=0.5,label='Iris-setosa')
plt.scatter(X_reduced[y == 1, 0], X_reduced[y == 1, 1], color='blue', alpha=0.5,label='Iris-versicolor')
plt.scatter(X_reduced[y == 2, 0], X_reduced[y == 2, 1], color='green', alpha=0.5,label='Iris-virginica')
plt.legend()
plt.title('PCA of IRIS dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
print("======================== PCA ========================")
print("======================== t-SNE  ========================")
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载手写数字数据集
digits = datasets.load_digits()
X = digits.data
y = digits.target

# 创建一个t-SNE对象，设置降维后的维度为2
tsne = TSNE(n_components=2)

# 对数据进行t-SNE降维
X_reduced = tsne.fit_transform(X)

plt.figure(figsize=(8, 6))
for i in range(10):
    plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1], alpha=0.5, label=str(i))
plt.legend()
plt.title('t-SNE of Digits dataset')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

print("======================== t-SNE  ========================")
