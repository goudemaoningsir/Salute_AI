# !/usr/bin/env python
# -*- coding:utf-8 -*-　
# @Time : 2023/6/8 19:34 
# @Author : sanmaomashi
# @GitHub : https://github.com/sanmaomashi
# @Summary : 文本特征提取

print("======================== one-hot ========================")
from sklearn.preprocessing import OneHotEncoder

# 原始数据
data = [
    ['red'],
    ['green'],
    ['blue'],
    ['red']
]

# 创建一个 OneHotEncoder 实例
encoder = OneHotEncoder(sparse=False)

# 用 OneHotEncoder 拟合并转化数据
one_hot_encoded_data = encoder.fit_transform(data)

print(one_hot_encoded_data)
print("======================== one-hot ========================")
print("======================== 词袋模型 (Bag of Words) ========================")
# 步骤 1: 导入所需的库
from sklearn.feature_extraction.text import CountVectorizer

# 步骤 2: 创建文档数据集
documents = [
    'Hello, how are you?',
    'I am getting started with Natural Language Processing.',
    'This is an example of bag of words.',
]

# 步骤 3: 创建 CountVectorizer 的实例
vectorizer = CountVectorizer()

# 步骤 4: 拟合并转化文档数据集
X = vectorizer.fit_transform(documents)

# 查看特征向量
print(X.toarray())

# 查看每个特征对应的单词
print(vectorizer.get_feature_names_out())
print("======================== 词袋模型 (Bag of Words) ========================")

print("======================== tf-idf ========================")

from sklearn.feature_extraction.text import TfidfVectorizer

# 我们的文档集
documents = [
    'Hello, how are you?',
    'I am getting started with Natural Language Processing.',
    'This is an example of bag of words.',
]

# 创建 TfidfVectorizer 的实例
vectorizer = TfidfVectorizer()

# 拟合并转换文档集
X = vectorizer.fit_transform(documents)

# 输出每个文档的特征向量
print(X.toarray())

# 输出每个特征的名称
print(vectorizer.get_feature_names_out())
print("======================== tf-idf ========================")
