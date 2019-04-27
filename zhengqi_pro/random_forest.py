# -*- coding: utf-8 -*-

# @Time    : 2019-04-26 10:49
# @Author  : jian
# @File    : random_forest.py
import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pylab as plt

df = pd.read_csv('data/zhengqi_train.txt', sep='\t')

# 数据处理
# df.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)

x = df.iloc[:, :-1]
y = df.target

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

model = RandomForestRegressor(
    n_estimators=40,  # 学习器个数
    # criterion='mse',  # 评价函数
    max_depth=23,  # 最大的树深度，防止过拟合
    min_samples_split=20,  # 根据属性划分节点时，每个划分最少的样本数
    min_samples_leaf=5,  # 最小叶子节点的样本数，防止过拟合
    max_features='0.6',  # auto是sqrt(features)还有 log2 和 None可选
    max_leaf_nodes=None,  # 叶子树的最大样本数
    bootstrap=True,  # 有放回的采样
    min_weight_fraction_leaf=0,
    n_jobs=5)  # 同时用多少个进程训练


def find_params(para_dict, estimator, x_train, y_train):
    gsearch = GridSearchCV(estimator, param_grid=para_dict, scoring=None,
                           n_jobs=4, iid=False, cv=5)
    gsearch.fit(x_train, y_train)
    print('参数的最佳取值：{0}'.format(gsearch.best_params_))
    print('最佳模型得分:{0}'.format(gsearch.best_score_))
    return gsearch.best_params_, gsearch.best_score_


if __name__ == "__main__":

    print("----start", datetime.datetime.now())
    step = 1
    for i in range(step):
        # # 1 确定估计器
        # param_test1 = {
        #     'n_estimators': [i for i in range(200, 501, 20)]
        # }
        # best_params, best_score = find_params(param_test1, model, x_train, y_train)
        # print('model_rf', i, ':')
        # print(best_params, ':best_score:', best_score)
        # # 设置最优参数
        # model.set_params(n_estimators=best_params['n_estimators'])   # n_estimators  360 最优


        # # 2.1 对max_depth 和 min_samples_split 和 min_samples_leaf 进行粗调
        # param_test2_1 = {
        #     'max_depth': [20, 25, 30],
        #     'min_samples_split': [10, 25],
        #     'min_samples_leaf': [10, 25]
        # }
        # best_params, best_score = find_params(param_test2_1, model, x_train, y_train)

        # # 2.2 对max_depth 和 min_samples_split 和 min_samples_leaf 进行精调
        # max_d = 25
        # min_ss = 25
        # min_sl = 10
        # param_test2_2 = {
        #     'max_depth': [max_d - 2, max_d, max_d + 2],
        #     'min_samples_split': [min_ss - 5, min_ss, min_ss + 5],
        #     'min_samples_leaf': [min_sl - 5, min_sl, min_sl + 5]
        # }
        # best_params, best_score = find_params(param_test2_2, model, x_train, y_train)
        #
        # model.set_params(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'],
        #                min_samples_leaf=best_params['min_samples_leaf'])
        #
        # print(best_params, ':best_score:', best_score)
        # print(model)
        #

        # 3.1 对 max_features 进行调参：
        param_test3_1 = {
            'max_features': [0.5, 0.7, 0.9]
        }
        best_params, best_score = find_params(param_test3_1, model, x_train, y_train)

        # 3.2 对 max_features 进行精调：
        max_f = best_params['max_features']
        param_test3_2 = {
            'max_features': [max_f - 0.1, max_f, max_f + 0.1]
        }
        best_params, best_score = find_params(param_test3_2, model, x_train, y_train)
        model.set_params(max_features=best_params['max_features'])
        print('model_rf', i, ':')
        print(best_params, ':best_score:', best_score)

    print("----end", datetime.datetime.now())
    #

    #
    # clf.fit(x_train, y_train)
    # y_predict = clf.predict(x_predict)
    # blending_predict_i = clf.predict(blending_i)
    #
    # f = open('./rf_para/model' + str(i) + '.txt', 'w')
    # f.write(str(clf.get_params()))
    # f.close()
    #
    # return y_predict, blending_predict_i
