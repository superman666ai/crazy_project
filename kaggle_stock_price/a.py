# -*- coding: utf-8 -*-

# @Time    : 2019-04-30 14:48
# @Author  : jian
# @File    : a.py
import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot
# from pandas import read_csv, set_option
from pandas import Series, datetime
# from pandas.tools.plotting import scatter_matrix, autocorrelation_plot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, \
    ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.arima_model import ARIMA
from xgboost import XGBClassifier
import seaborn as sns

sentence_file = "data/combined_stock_data.csv"
sentence_df = pd.read_csv(sentence_file, parse_dates=[1])  # parse_dates is used to 指定时间序列
# print(sentence_df.shape)
# print(sentence_df.dtypes)
print(sentence_df.columns)
print(sentence_df.head())

stock_prices = "data/DJIA_table.csv"
stock_data = pd.read_csv(stock_prices, parse_dates=[0])
print(stock_data.head())

# 你会发现Volumn是int型，为了保持所有数据类型的一致性
# 应该把数据转换为float
# print(stock_data.shape)
# print(stock_data.dtypes)

# merged_dataframe = sentence_df[
#     ['Date', 'Label', 'Subjectivity', 'Objectivity', 'Positive', 'Negative', 'Neutral']].merge(stock_data, how='inner',
#                                                                                                on='Date',
#                                                                                                left_index=True)
# print(merged_dataframe.shape)
# print(merged_dataframe.head())
