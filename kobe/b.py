# -*- coding: utf-8 -*-

# @Time    : 2019-04-29 14:13
# @Author  : jian
# @File    : b.py
from kobe.a import train_kobe
from kobe.a import train_label
from kobe.a import test_kobe
from sklearn.metrics import confusion_matrix, log_loss
import time
import numpy as np

range_m = np.logspace(0, 2, num=5).astype(int)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV

n_folds = 5


def rmsle_cv(model, train_x_head=train_kobe):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_x_head)
    accuracy = cross_val_score(model, train_x_head, train_label, scoring="accuracy", cv=kf)
    return accuracy


# 随机森林
my_random = RandomForestClassifier()

# =============================================================================
parameters = dict()
parameters["n_estimators"] = [x for x in range(10, 200, 10)]  # 学习器个数
# parameters["criterion "] = 'mse',  # 评价函数
# parameters["max_depth "] = [x for x in range(40)] # 最大的树深度，防止过拟合
# parameters["min_samples_split "] = [50, 100, 150, 200, 250, 300]  # 根据属性划分节点时，每个划分最少的样本数
# parameters["min_samples_leaf "] = [50, 100, 150, 200, 250, 300] # 最小叶子节点的样本数，防止过拟合
# parameters["max_features "] = [50, 100, 150, 200, 250, 300] # auto是sqrt(features)还有 log2 和 None可选

# parameters = {"n_estimators": [50, 100, 150, 200, 250, 300]
#               # "criterion": 'mse',  # 评价函数
#               # "max_depth": 23,  # 最大的树深度，防止过拟合
#               # "min_samples_split": 20,  # 根据属性划分节点时，每个划分最少的样本数
#               # "min_samples_leaf": 5,  # 最小叶子节点的样本数，防止过拟合
#               # "max_features": 'auto',  # auto是sqrt(features)还有 log2 和 None可选
#               # "max_leaf_nodes": None,  # 叶子树的最大样本数
#               # "bootstrap": True,  # 有放回的采样
#               # "min_weight_fraction_leaf": 0,
#               # "n_jobs": 5  # 同时用多少个进程训练
#               }

optimized_GBM = GridSearchCV(my_random, param_grid=parameters, cv=5, n_jobs=3, scoring='accuracy')
optimized_GBM.fit(train_kobe, train_label)
evalute_result = optimized_GBM.scorer_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
# =============================================================================
#
#
# score = rmsle_cv(my_random)
# print("\nRandomForestClassifier 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
