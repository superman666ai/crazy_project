# -*- coding: utf-8 -*-

# @Time    : 2019-04-28 11:38
# @Author  : jian
# @File    : 1.py

"""
分析数据
"""
import seaborn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

train = pd.read_csv('data/zhengqi_train.txt', sep='\t')
test = pd.read_csv('data/zhengqi_test.txt', sep='\t')
train_x = train.drop(['target'], axis=1)
all_data = pd.concat([train_x, test])

# 查看每个特征的值的分布情况
# for col in all_data.columns:
#     seaborn.distplot(train[col])
#     seaborn.distplot(test[col])
#     plt.show()
#
# 特征'V5', 'V17', 'V28', 'V22', 'V11', 'V9'
# 训练集数据与测试集数据分布不一致，会导致模型泛化能力差，采用删除此类特征方法。

all_data.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)

# 数据归一化  缩放到0-1之间
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
data_minmax = pd.DataFrame(min_max_scaler.fit_transform(all_data), columns=all_data.columns)

# =============================================================================
# 数据观察（可视化）
# for col in data_minmax.columns:
#     seaborn.distplot(data_minmax[col])
#     seaborn.distplot(train[col])
#     seaborn.distplot(test[col])
#     plt.show()
# =============================================================================

# 针对特征['V0','V1','V6','V30']做数据变换，使得数据符合正态分布

data_minmax['V0'] = data_minmax['V0'].apply(lambda x: math.exp(x))
data_minmax['V1'] = data_minmax['V1'].apply(lambda x: math.exp(x))
data_minmax['V6'] = data_minmax['V6'].apply(lambda x: math.exp(x))
data_minmax['V30'] = np.log1p(data_minmax['V30'])

# for col in data_minmax.columns:
#     seaborn.distplot(data_minmax[col])
#     seaborn.distplot(train[col])
#     seaborn.distplot(test[col])
#     plt.show()

#

X_scaled = pd.DataFrame(preprocessing.scale(data_minmax),columns = data_minmax.columns)

# 将训练和测试分开
train_x = X_scaled.ix[0:len(train)-1]

test = X_scaled.ix[len(train):]
Y=train['target']

"""
特征选择
通过方差阈值来筛选特征，采用threshold=0.85，剔除掉方差较小，
即变化较小的特征删除，因为预测意义小；
大多数数据已经被标准化到【0，1】之间，通过分析，
方差的值域控制为 0.85*（1-0.85）之间有利于特征选择，太大容易删除过多的特征，太小容易保留无效的特征，对预测造成干扰。
"""

#特征选择
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


# 去掉小于方差
threshold = 0.85
vt = VarianceThreshold().fit(train_x)
# Find feature names

feat_var_threshold = train_x.columns[vt.variances_ > threshold * (1-threshold)]
train_x = train_x[feat_var_threshold]
test = test[feat_var_threshold]



# 单变量选择
X_scored = SelectKBest(score_func=f_regression, k='all').fit(train_x, Y)
feature_scoring = pd.DataFrame({
        'feature': train_x.columns,
        'score': X_scored.scores_
    })
# print(feature_scoring.head())
head_feature_num = 18
feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']

train_x_head = train_x[train_x.columns[train_x.columns.isin(feat_scored_headnum)]]

print("train_x_head", train_x_head.shape)

X_scaled = pd.DataFrame(preprocessing.scale(train_x),columns = train_x.columns)

print("X_scaled", X_scaled.shape)

# =============================================================================

# from sklearn.decomposition import PCA, KernelPCA
# components = 8
# pca = PCA(n_components=components).fit(train_x)
# pca_variance_explained_df = pd.DataFrame({
#      "component": np.arange(1, components+1),
#      "variance_explained": pca.explained_variance_ratio_
#     })
#
# cols = train_x.columns
# train_x = pca.transform(train_x)
# test = pca.transform(test[cols])

# =============================================================================
