# !/usr/bin/env python
# -*- coding:utf-8 -*-　
# @Time : 2023/6/4 16:22
# @Summary : 字典特征提取

from sklearn.feature_extraction import DictVectorizer

data = [
    {'age': 30, 'sex': 'male', 'occupation': 'engineer'},
    {'age': 25, 'sex': 'female', 'occupation': 'teacher'},
    {'age': 50, 'sex': 'female', 'occupation': 'engineer'},
    {'age': 23, 'sex': 'male', 'occupation': 'student'},
]

vec = DictVectorizer(sparse=False)

features = vec.fit_transform(data)

print(features)
print(vec.get_feature_names_out())
