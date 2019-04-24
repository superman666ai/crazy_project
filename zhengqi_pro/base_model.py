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
model = RandomForestRegressor(n_estimators=2,  # 学习器个数
                              criterion='mse',  # 评价函数
                              max_depth=None,  # 最大的树深度，防止过拟合
                              min_samples_split=2,  # 根据属性划分节点时，每个划分最少的样本数
                              min_samples_leaf=1,  # 最小叶子节点的样本数，防止过拟合
                              max_features='auto',  # auto是sqrt(features)还有 log2 和 None可选
                              max_leaf_nodes=None,  # 叶子树的最大样本数
                              bootstrap=True,  # 有放回的采样
                              min_weight_fraction_leaf=0,
                              n_jobs=5)  # 同时用多少个进程训练

# 极端随机森林
# model =ExtraTreesRegressor()

# 梯度提升
# model = GradientBoostingRegressor()

# xgboost
# model = xgb.XGBRegressor()

# model.fit(x_train, y_train)

# score = model.score(x_test, y_test)
# print("score", score)
# y_pred = model.predict(x_test)
# print(" random forest mean_squared_error", mean_squared_error(y_test, y_pred))
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
                           n_jobs=4, iid=False, cv=5)
    gsearch.fit(x_train, y_train)
    evalute_result = gsearch.cv_results_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(gsearch.best_params_))
    print('最佳模型得分:{0}'.format(gsearch.best_score_))
    return gsearch.best_params_, gsearch.best_score_

find_params(model, param_grid=param_grid)



# def run(x_train, y_train, i, x_predict, blending_i):
#
#     clf = RandomForestRegressor(
#             n_estimators=2,             # 学习器个数
#             criterion='mse',             # 评价函数
#             max_depth=None,              # 最大的树深度，防止过拟合
#             min_samples_split=2,         # 根据属性划分节点时，每个划分最少的样本数
#             min_samples_leaf=1,          # 最小叶子节点的样本数，防止过拟合
#             max_features='auto',         # auto是sqrt(features)还有 log2 和 None可选
#             max_leaf_nodes=None,         # 叶子树的最大样本数
#             bootstrap=True,              # 有放回的采样
#             min_weight_fraction_leaf=0,
#             n_jobs=5)                   # 同时用多少个进程训练
#
#     # 1 首先确定迭代次数
#     param_test1 = {
#         'n_estimators': [i for i in range(100, 201, 20)]
#     }
#     best_params, best_score = find_params(clf, param_test1)
#     print('model_rf', i, ':')
#     print(best_params, ':best_score:', best_score)
#     clf.set_params(n_estimators=best_params['n_estimators'])
#
#     # 2.1 对max_depth 和 min_samples_split 和 min_samples_leaf 进行粗调
#     param_test2_1 = {
#         'max_depth': [20, 25, 30],
#         'min_samples_split' : [10, 25],
#         'min_samples_leaf' : [10, 25]
#     }
#     best_params, best_score = find_params(clf, param_test2_1)
#
#     # 2.2 对max_depth 和 min_samples_split 和 min_samples_leaf 进行精调
#     max_d = best_params['max_depth']
#     min_ss = best_params['min_samples_split']
#     min_sl = best_params['min_samples_leaf']
#     param_test2_2 = {
#         'max_depth': [max_d-2, max_d, max_d+2],
#         'min_samples_split': [min_ss-5, min_ss, min_ss+5],
#         'min_samples_leaf' : [min_sl-5, min_sl, min_sl+5]
#     }
#     best_params, best_score = find_params(param_test2_2, clf, x_train, y_train)
#     clf.set_params(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'],
#                    min_samples_leaf=best_params['min_samples_leaf'])
#     print('model_rf', i, ':')
#     print(best_params, ':best_score:', best_score)
#
#     # 3.1 对 max_features 进行调参：
#     param_test3_1 = {
#         'max_features': [0.5, 0.7, 0.9]
#     }
#     best_params, best_score = find_params(param_test3_1, clf, x_train, y_train)
#
#     # 3.2 对 max_features 进行精调：
#     max_f = best_params['max_features']
#     param_test3_2 = {
#         'max_features': [max_f-0.1, max_f, max_f+0.1]
#     }
#     best_params, best_score = find_params(param_test3_2, clf, x_train, y_train)
#     clf.set_params(max_features=best_params['max_features'])
#     print('model_rf', i, ':')
#     print(best_params, ':best_score:', best_score)
#
#     clf.fit(x_train, y_train)
#     y_predict = clf.predict(x_predict)
#     blending_predict_i = clf.predict(blending_i)
#
#     return y_predict, blending_predict_i
#


# 保存结果
# result = model.predict(test_df)
# print(result)
# result_df = pd.DataFrame(result, columns=['target'])
# result_df.to_csv("0.098.txt", index=False, header=False)
