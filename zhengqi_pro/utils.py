# -*- coding: utf-8 -*-

# @Time    : 2019/1/11 14:44
# @Author  : jian
# @File    : learning_rate.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.draw()
    plt.show()


def model_gridsearch(estimator, param, x_train, x_test, y_train, y_test, cv=5):
    """

    :param estimator:
    :param param:
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param cv:
    :return:
    """

    gc = GridSearchCV(estimator, param_grid=param, cv=cv)
    gc.fit(x_train, y_train)

    # 预测准确率
    print(gc.score(x_test, y_test))

    # 交叉验证中最好的结果
    print(gc.best_score_)

    # 最好的模型
    print(gc.best_estimator_)

    # 每个k的 验证结果
    print(gc.cv_results_)


def save_result(result):
    """

    :param result: 结果集
    :return:
    """

    result_df = pd.DataFrame(result, columns=['target'])
    result_df.to_csv("result.txt", index=False, header=False)


def load_data(path, **kwargs):
    # 读取数据 分析
    df = pd.read_csv(path, sep=kwargs["sep"])
    x = df.iloc[:, :-1]
    y = df.target
    y = np.array(y)[:, np.newaxis]
    x, y = et_method(x, y)
    return x, y


# 处理特征数量
def et_method(x, y):
    clf = ExtraTreesRegressor()
    clf = clf.fit(x, y)

    # 筛选特征
    model = SelectFromModel(clf, prefit=True)
    x = model.transform(x)
    return x, y
