# -*- coding: utf-8 -*-

# @Time    : 2019-04-03 8:56
# @Author  : jian
# @File    : titannic_class.py
"""
Created on Tue Apr 10 17:21:16 2018
@author: CSH
"""
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# load data
df = pd.read_csv("../data/titanic_train.csv")

# feature processing
# 使用 中位数填充
df["Age"] = df["Age"].fillna(df["Age"].median())

# print(titanic["Sex"].unique())
# 分类
df.loc[df["Sex"] == "male", "Sex"] = 0
df.loc[df["Sex"] == "female", "Sex"] = 1

# 众数填充
# # print(titanic["Embarked"].value_counts())
df["Embarked"] = df["Embarked"].fillna("S")
df.loc[df["Embarked"] == "S", "Embarked"] = 0
df.loc[df["Embarked"] == "C", "Embarked"] = 1
df.loc[df["Embarked"] == "Q", "Embarked"] = 2


# 提取名字信息
def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

titles = df["Name"].apply(get_title)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Mlle": 7, "Major": 8, "Col": 9,
                 "Ms": 10, "Mme": 11, "Lady": 12, "Sir": 13, "Capt": 14, "Don": 15, "Jonkheer": 16, "Countess": 17}
for k, v in title_mapping.items():
    titles[titles == k] = v
# print(pd.value_counts(titles))
df["Title"] = titles
# print(df)

# 选取个别特征
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived", "Title"]
X = df[predictors]

Y = df["Survived"]

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# print(x_test)

"""
基本分类器
"""


def classifier():
    scores = []
    models = [LinearRegression(),
              LogisticRegression(solver="liblinear"),
              KNeighborsClassifier(),
              SVC(gamma='auto'),
              MLPClassifier(max_iter=1000),
              DecisionTreeClassifier(),
              RandomForestClassifier(),
              AdaBoostClassifier()
              ]
    for model in models:
        print(model)
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        print("score", score)


classifier()




# 特征选择
# # =============================================================================
# import numpy as np
# from sklearn.feature_selection import SelectKBest, f_classif
# import matplotlib.pyplot as plt
#
# predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "NameLength"]
# selector = SelectKBest(f_classif, k=5)
# selector.fit(df[predictors], df["Survived"])
# scores = -np.log10(selector.pvalues_)
#
# plt.bar(range(len(predictors)), scores)
# plt.xticks(range(len(predictors)), predictors, rotation='vertical')
# plt.show()
# # =============================================================================
#
#
# # =============================================================================
# # from sklearn import cross_validation
# # from sklearn.ensemble import RandomForestClassifier
# # predictors=["Pclass","Sex","Fare","Title","NameLength"]
# # alg=RandomForestClassifier(random_state=1,n_estimators=50,min_samples_split=12,min_samples_leaf=1)
# # kf=cross_validation.KFold(titanic.shape[0],n_folds=3,random_state=1)
# # scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=kf)
# # print(scores.mean())
# # =============================================================================
#
