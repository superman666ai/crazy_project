# -*- coding: utf-8 -*-

# @Time    : 2019-01-18 10:57
# @Author  : jian
# @File    : feature_analyse.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_regression, f_classif, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

# 读取数据 分析
df = pd.read_csv('data/zhengqi_train.txt', sep='\t')
x = df.iloc[:, :-1]
y = df.target
# y = np.array(y)[:, np.newaxis]

# x_mm = MinMaxScaler()
# x = x_mm.fit_transform(x)
# # print(x)
# y_mm = MinMaxScaler()
# y = y_mm.fit_transform(y)


#
# print(y)


# print(x.shape)
# print(y.values)

def linear_feature(x, y):
    # 设置图大小
    # plt.figure(figsize=(6, 3))
    x_names = x.columns

    for i in x_names:
        xa = x[i]
        plt.scatter(y, xa)
        plt.title(i)
        # plt.show()
        plt.savefig("image2/" + i + ".png")
        plt.close()

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


def et_method(x, y):
    clf = ExtraTreesRegressor()
    clf = clf.fit(x, y)

    # # 画图
    # x = np.arange(x.shape[1])
    # y = clf.feature_importances_
    # # 画出 x 和 y 的柱状图
    # plt.bar(x, y)
    # for x, y in zip(x, y):
    #     plt.text(x, y, '%.2f' % y, ha='center', va='bottom')
    # plt.show()

    # 筛选特征
    model = SelectFromModel(clf, prefit=True)
    x = model.transform(x)

    clf = clf.fit(x, y)

    x = np.arange(x.shape[1])
    y = clf.feature_importances_
    # 画出 x 和 y 的柱状图
    plt.bar(x, y)
    for x, y in zip(x, y):
        plt.text(x, y, '%.2f' % y, ha='center', va='bottom')
    plt.show()


def plot_feature_scores(x, y, names=None):
    if not names:
        names = [x for x in range(x.shape[1])]

    # 1. 使用 sklearn.feature_selection.SelectKBest 给特征打分
    slct = SelectKBest(k="all")
    slct.fit(x, y)
    scores = slct.scores_

    # 2. 将特征按分数 从大到小 排序
    named_scores = zip(names, scores)
    sorted_named_scores = sorted(named_scores, key=lambda z: z[1], reverse=True)

    sorted_scores = [each[1] for each in sorted_named_scores]
    sorted_names = [each[0] for each in sorted_named_scores]

    y_pos = np.arange(len(names))  # 从上而下的绘图顺序

    # 3. 绘图
    fig, ax = plt.subplots()
    ax.barh(y_pos, sorted_scores, height=0.7, align='center', color='#6699CC', tick_label=sorted_names)
    # ax.set_yticklabels(sorted_names)      # 也可以在这里设置 条条 的标签~
    ax.set_yticks(y_pos)
    ax.set_xlabel('Feature Score')
    ax.set_ylabel('Feature Name')
    ax.invert_yaxis()
    ax.set_title('F_classif scores of the features.')

    # 4. 添加每个 条条 的数字标签
    for score, pos in zip(sorted_scores, y_pos):
        ax.text(score + 20, pos, '%.1f' % score, ha='center', va='bottom', fontsize=8)

    plt.show()


def feature_spread(x, y):
    # 训练数据分布情况
    # plt.figure(figsize=(18, 18))

    for column_index, column in enumerate(x.columns):
        plt.subplot(10, 4, column_index + 1)
    g = sns.kdeplot(x[column])
    g.set_xlabel(column)
    g.set_ylabel('Frequency')

    # 特征相关性
    plt.figure(figsize=(20, 16))
    colnm = x.columns.tolist()
    mcorr = x[colnm].corr(method="spearman")
    mask = np.zeros_like(mcorr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
    plt.show()


def ka_fang(x, y):
    print(y.values)
    # model1 = SelectKBest(f_regression, k=5)  # 选择k个最佳特征
    # model1 = SelectKBest(f_regression, k=5)  # 选择k个最佳特征

    # model1 = SelectPercentile(f_regression, percentile=80)  # 选择k个最佳特征

    model1 = SelectPercentile(mutual_info_regression, percentile=80)  # 选择k个最佳特征

    x = model1.fit_transform(x, y.values)
    print(x.shape)
    print(model1.scores_)


if __name__ == "__main__":
    # et_method(x, y)
    # plot_feature_scores(x, y)
    # feature_spread(x, y)
    # linear_feature(x, y)
    ka_fang(x, y)
