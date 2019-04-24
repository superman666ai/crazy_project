# -*- coding: utf-8 -*-

# @Time    : 2019-04-24 11:20
# @Author  : jian
# @File    : lasso_gbr.py
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import pca
from sklearn import linear_model
# import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel

# 导入数据
zhengqi_train = pd.read_table('data/zhengqi_train.txt', encoding='utf-8')
zhengqi_test = pd.read_table('data/zhengqi_test.txt', encoding='utf-8')

# print(zhengqi_train.head())


# 数据分割
X = np.array(zhengqi_train.drop(['target'], axis = 1))
y = np.array(zhengqi_train.target)

print('================================')
print(X.shape)
print(y.shape)
print('================================')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(len(X_train))
print(len(X_test))

# PCA数据处理-降维
pca = pca.PCA(n_components=0.95)
pca.fit(X)
X_pca = pca.transform(X)
X1_pca = pca.transform(zhengqi_test)

X_train, X_test, Y_train, Y_test = train_test_split(X_pca, y, test_size=0.3, random_state=0)

#线性回归
clfL = linear_model.LinearRegression()

clfL.fit(X_train,Y_train)

y_true, y_pred = Y_test, clfL.predict(X_test)

print(mean_squared_error(y_true, y_pred))

ans_Liner = clfL.predict(X1_pca)
print(ans_Liner.shape)


'''GBR'''
#这里使用GBR
# 分离出训练集和测试集，并用梯度提升回归训练
X_train, X_test, Y_train, Y_test = train_test_split(X_pca, y, test_size=0.2, random_state=40)
myGBR = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                                  learning_rate=0.03, loss='huber', max_depth=15,
                                  max_features='sqrt', max_leaf_nodes=None,
                                  min_impurity_decrease=0.0, min_impurity_split=None,
                                  min_samples_leaf=10, min_samples_split=40,
                                  min_weight_fraction_leaf=0.0, n_estimators=300,
                                  presort='auto', random_state=10, subsample=0.8, verbose=0,
                                  warm_start=False)
myGBR.fit(X_train, Y_train)
Y_pred = myGBR.predict(X_test)
print(mean_squared_error(Y_test, Y_pred))
ans_GBR = myGBR.predict(X1_pca)
print(ans_GBR.shape)


'''加权融合'''
# 这里可以换个思路加
final_ans = (0.5*ans_Liner +0.5*ans_GBR)


# # 预测输出
# pd.DataFrame(final_ans).to_csv('./mergeGBR&Lasso&NN.txt',index=False, header=False)
# print('over')