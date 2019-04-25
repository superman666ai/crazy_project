# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from utils import plot_learning_curve
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('data/zhengqi_train.txt', sep='\t')

# 结果集
test_df = pd.read_csv('data/zhengqi_test.txt', sep='\t')

# 数据处理

df.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)
test_df.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)

x = df.iloc[:, :-1]
y = df.target

# # 标准化特征
# mm = MinMaxScaler()
# x = mm.fit_transform(x)

# 结果集标准
# test_df = mm.transform(test_df)

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# 线性回归
model = LinearRegression()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("score", score)
y_pred = model.predict(x_test)
print("线性回归mean_squared_error", mean_squared_error(y_test, y_pred))

# 岭回归
model = Ridge(alpha=0.1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("score", score)
y_pred = model.predict(x_test)
print("岭回归 mean_squared_error", mean_squared_error(y_test, y_pred))
# 绘制学习率曲线
# plot_learning_curve(model, title="带核函数的ridge learn_rate", X=x_train, y=y_train, cv=None)

# 带核函数的ridge
model = KernelRidge(kernel='linear', alpha=0.1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("score", score)
y_pred = model.predict(x_test)
print("带核函数的ridge mean_squared_error", mean_squared_error(y_test, y_pred))
# 绘制学习率曲线
# # plot_learning_curve(model, title="带核函数的ridge learn_rate", X=x_train, y=y_train, cv=None)


# lasso 回归
model = Lasso(alpha=0.1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("score", score)
y_pred = model.predict(x_test)
print("lasso回归 mean_squared_error", mean_squared_error(y_test, y_pred))
# 绘制学习率曲线
# plot_learning_curve(model, title="lasso回归 learn_rate", X=x_train, y=y_train, cv=None)




# model = SVR(kernel='rbf', C=1e3, gamma=0.1)
# # model = SVR(kernel='linear', C=1e3)
# # model = SVR(kernel='poly', C=1e3, degree=2)
#
# model.fit(x_train, y_train)
# score = model.score(x_test, y_test)
# print("score", score)
# y_pred = model.predict(x_test)
# print("SVR mean_squared_error", mean_squared_error(y_test, y_pred))
# # 绘制学习率曲线
# plot_learning_curve(model, title="SVR learn_rate", X=x_train, y=y_train, cv=None)




#
#     # 保存结果
#     # result = model.predict(test_df)
#     # print(result)
#     # result_df = pd.DataFrame(result, columns=['target'])
#     # result_df.to_csv("0.098.txt", index=False, header=False)


# # 贝叶斯回归
model = BayesianRidge()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("score", score)
y_pred = model.predict(x_test)
print("BayesianRidge mean_squared_error", mean_squared_error(y_test, y_pred))
# # # 绘制学习率曲线
# plot_learning_curve(model, title="SVR learn_rate", X=x_train, y=y_train, cv=10)


# # random forest

# 随机森林
# model = RandomForestRegressor(n_estimators=2,  # 学习器个数
#                               criterion='mse',  # 评价函数
#                               max_depth=None,  # 最大的树深度，防止过拟合
#                               min_samples_split=2,  # 根据属性划分节点时，每个划分最少的样本数
#                               min_samples_leaf=1,  # 最小叶子节点的样本数，防止过拟合
#                               max_features='auto',  # auto是sqrt(features)还有 log2 和 None可选
#                               max_leaf_nodes=None,  # 叶子树的最大样本数
#                               bootstrap=True,  # 有放回的采样
#                               min_weight_fraction_leaf=0,
#                               n_jobs=5)  # 同时用多少个进程训练

# 极端随机森林
# model =ExtraTreesRegressor()

# 梯度提升
# model = GradientBoostingRegressor()

# xgboost
model = xgb.XGBRegressor()
# model.fit(x_train, y_train)
# score = model.score(x_test, y_test)
# print("xgboost score", score)
# y_pred = model.predict(x_test)
# print(" xgboost mean_squared_error", mean_squared_error(y_test, y_pred))
# 绘制学习率曲线
# plot_learning_curve(model, title="learn_rate", X=x_train, y=y_train, cv=20)

# 寻找最优参数

"""----模型调参----"""
param_grid = {"n_estimators": [7, 8],  # 学习器个数
              "max_depth": [4],  # 最大深度
              "min_samples_split": [2],  # 划分节点时最少样本数
              "min_samples_leaf": [3]  # 最小叶子节点样本数
              }


def find_params(estimator, param_grid):
    gsearch = GridSearchCV(estimator, param_grid=param_grid, scoring='neg_mean_squared_error',
                           n_jobs=2, iid=False, cv=5)
    gsearch.fit(x_train, y_train)
    evalute_result = gsearch.scorer_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(gsearch.best_params_))
    print('最佳模型得分:{0}'.format(gsearch.best_score_))
    return gsearch.best_params_, gsearch.best_score_
#

find_params(model, param_grid=param_grid)




# 保存结果
# result = model.predict(test_df)
# print(result)
# result_df = pd.DataFrame(result, columns=['target'])
# result_df.to_csv("0.098.txt", index=False, header=False)
