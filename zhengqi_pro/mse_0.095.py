# -*- coding: utf-8 -*-

# @Time    : 2019-04-24 11:38
# @Author  : jian
# @File    : mse_0.095.py
import pandas as pd
import numpy as np
from sklearn.decomposition import pca
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from numpy import concatenate
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# load dataset
dataset = pd.read_table('data/zhengqi_train.txt')
test = pd.read_table('data/zhengqi_test.txt')

# import matplotlib.pyplot as plt
# import seaborn as sns
# #特征相关性热力图分析
# corrmat=dataset.corr()
# sns.heatmap(corrmat,vmax=.10)
# k=20
# cols=corrmat.nlargest(k,'target')['target'].index
# cm=np.corrcoef(dataset[cols].values.T)
# sns.set(font_scale=1.25)
# hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)
#
# plt.show()


# y=dataset['target']
# train_labels=y.values.copy
# x=np.random.randn(100)
# ax=sns.distplot(x)


selected_features = ['V0', 'V1', 'V2', 'V3', 'V4', 'V6', 'V7', 'V8', 'V10', 'V13', 'V18', 'V12', 'V16',
                     'V20', 'V23', 'V27', 'V30', 'V31', 'V36']  # 通过相关系数选出的特征
test = test[selected_features]
test_data = test.values
y = dataset['target'].values
X = dataset[selected_features].values
# PCA数据处理(做了这个处理效果反而不好)
# pca = pca.PCA(n_components=0.95)
# pca.fit(X)
# X = pca.transform(X)
# test_data = pca.transform(test_data)


train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=33)
print(train_X)
# xgboost模型
model = xgb.XGBRegressor(max_depth=5, learning_rate=0.05, n_estimators=150, silent=False, objective='reg:linear')
model.fit(train_X, train_y)
y_pred_line1 = model.predict(test_X)
mse = mean_squared_error(y_pred_line1, test_y)
print("-------", mse)
# pre_y1 = model.predict(test_data)

# # xgboost模型调参
# cv_params = {'max_depth': [4, 5]}
# other_params = {'learning_rate': 0.05, 'n_estimators': 150, 'max_depth': 5, 'min_child_weight': 1,
#                 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
# model = xgb.XGBRegressor(**other_params)
# optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5,
#                              verbose=1, n_jobs=4)
# optimized_GBM.fit(train_X, train_y)
# evalute_result = optimized_GBM.scorer_
# print('每轮迭代运行结果:{0}'.format(evalute_result))
# print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
# print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))


# 线性模型
reg = linear_model.LinearRegression()
reg.fit(train_X, train_y)
y_pred_line2 = reg.predict(test_X)
mae_line = mean_squared_error(y_pred_line2, test_y)
print("MAE line_score:", mae_line)
