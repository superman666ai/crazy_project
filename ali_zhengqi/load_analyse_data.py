# -*- coding: utf-8 -*-

# @Time    : 2019/1/10 17:42
# @Author  : jian
# @File    : read_data.py
"""
加载和分析数据
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import SelectPercentile, f_classif

# 读取数据 分析
df = pd.read_csv('data/zhengqi_train.txt', sep='\t')

# 数据处理

# 剔除认为不重要的特征
# df.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)

x = df.iloc[:, :-1]
y = df.target

""" 数据分布"""

x_data = x.iloc[:, 3:4]
y_data = y

plt.scatter(x_data, y_data)
plt.show()


"""相关性热力图"""

# # 读入数据并显示前两行
#
# # #相关性分析
# column = df.columns.tolist()
# # print(column)
# mcorr = df[column].corr()
# # print(mcorr)
# mcorr_data = np.array(mcorr.target)
# # print(mcorr_data)
# mask = np.zeros_like(mcorr, dtype=np.bool)  # 构造与mcorr同维数矩阵 为bool型
# mask[np.triu_indices_from(mask)] = True  # 角分线右侧为True
# cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 返回matplotlib colormap对象
# g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')  # 热力图（看两两相似度）
# plt.show()

"""---------"""
# # f_classif为内置的计算函数函数，有多种计算函数供选择；保持10%最有意义的特征
# selectorPer = SelectPercentile(f_classif, percentile=10)
# selectorPer = selectorPer.fit(x, y)
# print(selectorPer.scores_) #特征的得分值
# # [  119.26450218    47.3644614   1179.0343277    959.32440573]
# print(selectorPer.pvalues_) #特征的p-value值
# # [  1.66966919e-31   1.32791652e-16   3.05197580e-91   4.37695696e-85]，结果说明第2,4个特征被选择
# scores = -np.log10(selectorPer.pvalues_)
# print(scores)
# # [ 30.77736957  15.87682923  90.51541891  84.35882772]
# # scores /= scores.max()
# # print(scores)
# # [ 0.3400235   0.17540469  1.          0.93198296]
#
# e, t = x.shape
# t = np.arange(t)
#
# plt.figure()
# plt.plot(t, scores, 'r-', linewidth=2, label='scores')
# # plt.plot(t, selectorPer.pvalues_, 'g-', linewidth=2, label='p-value')
# plt.xticks(t)
# plt.grid()
# plt.show()
"""---------"""
# SelectKBest
# 作用:根据k最高分选择特征

# from sklearn.feature_selection import SelectKBest
#
# # (150, 4)
# # f_classif为内置的计算函数函数，有多种计算函数供选择；选择前k个最高得分特征
# selectKB = SelectKBest(f_classif)
# selectKB = selectKB.fit(x, y)
# print(selectKB.scores_) #特征的得分值
# # [  119.26450218    47.3644614   1179.0343277    959.32440573]
# print(selectKB.pvalues_) #特征的得分值
# # [  1.66966919e-31   1.32791652e-16   3.05197580e-91   4.37695696e-85]
# print(selectKB.get_support()) #特征的得分值
# # [False False  True  True]
# print (selectKB.get_support(True))
#
# e, t = x.shape
# t = np.arange(t)
#
# plt.figure()
# plt.plot(t, selectKB.get_support(), 'r-', linewidth=2, label='scores')
# # plt.plot(t, selectorPer.pvalues_, 'g-', linewidth=2, label='p-value')
# plt.xticks(t)
# plt.grid()
# plt.show()

"""------"""

# # xgb 分析特征相关
# model = xgb.XGBRegressor()
# model.fit(x, y)
#
# ### plot feature importance
# fig, ax = plt.subplots(figsize=(15, 15))
# xgb.plot_importance(model,
#                     height=0.5,
#                     ax=ax,
#                     max_num_features=64 )
# plt.show()


# # 画图分析
# train = pd.read_csv('data/zhengqi_train.txt', sep='\t')
# test = pd.read_csv('data/zhengqi_test.txt', sep='\t')
# train_x = train.drop(['target'], axis=1)

# all_data = pd.concat([train_x, test])
# name = 0
# for col in all_data.columns:
#     seaborn.distplot(train[col])
#     seaborn.distplot(test[col])
#     # plt.show()
#     plt.savefig("image/" + "V" + str(name) + ".png")
#     name += 1
#     plt.close()
#

# 删除特征因子
# all_data.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)
