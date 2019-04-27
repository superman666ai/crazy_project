# -*- coding: utf-8 -*-

# @Time    : 2019-04-26 10:09
# @Author  : jian
# @File    : test1.py

import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale, PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.model_selection import validation_curve
from keras.models import load_model
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, ARDRegression, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
import lightgbm as lgb

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, PReLU
from keras.layers import Input, add, multiply, maximum
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers  # 正则化


# 构建机器学习模型
def Model_1(train_x, train_y):
    '''
    输入：训练数据的特征数据，和标签数据
    输出：一个深度神经网络模型
    '''
    # 建立一个三层的神经网络（不带输出层）
    model = Sequential()
    model.add(Dense(500, input_shape=(train_x.shape[1],)))
    model.add(Activation('sigmoid'))

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('tanh'))

    # 输出层
    model.add(Dense(1))
    model.add(Activation('linear'))

    # 三种优化器：SGD，Adam,rmsprop
    model.compile(optimizer='sgd',
                  loss='mean_squared_error')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,
                                  verbose=0, mode='auto', min_delta=0.001,
                                  cooldown=0, min_lr=0)

    epochs = 50  # 迭代次数
    model.fit(train_x, train_y, epochs=epochs,
              batch_size=20, validation_split=0.0,
              callbacks=[reduce_lr],
              verbose=0)
    return model


def kfold_loss(df_x, df_y):
    '''
    输入：特征数据，和标签数据（dataframe类型的）
    输出：利用交叉验证划分数据，得到mean_loss
    '''
    loss_list = []
    df_x = pd.DataFrame(df_x, index=None)
    df_y = pd.DataFrame(df_y, index=None)
    sfloder = KFold(n_splits=5, shuffle=False)

    for train_id, test_id in sfloder.split(df_x, df_y):
        model = Model_1(df_x.iloc[train_id], df_y.iloc[train_id])
        loss = model.evaluate(df_x.iloc[test_id], df_y.iloc[test_id], verbose=0)
        loss_list.append(loss)
    return np.array(loss_list).mean()


def HeatGrape(df):
    '''
    输入dataframe数据
    输入特征之间的热力相关图
    '''
    # 找出相关程度
    plt.figure(figsize=(20, 16))  # 指定绘图对象宽度和高度
    colnm = df.columns.tolist()  # 列表头
    mcorr = df[colnm].corr()  # 相关系数矩阵，即给出了任意两个变量之间的相关系数
    mask = np.zeros_like(mcorr, dtype=np.bool)  # 构造与mcorr同维数矩阵 为bool型
    mask[np.triu_indices_from(mask)] = True  # 角分线右侧为True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 返回matplotlib colormap对象
    g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')  # 热力图（看两两相似度）
    plt.show()


# 创建多项式模型的函数
def Polynomial_model(degree=1):
    '''
    输入：一个维度
    输出：一个对应维度的多项式机器学习模型
    '''
    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=False)
    linear_regression = LinearRegression(normalize=True)
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    return pipeline


# 画出学习曲线->判断过拟合和欠拟合
def plot_learning_curve(estimator, X, y):
    '''
    输入：model, x, y
    输出：画出学习曲线，来判断是否过拟合或者欠拟合。
    '''
    train_sizes = np.linspace(.1, 1.0, 5)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    plt.figure()
    plt.title('learning curve')

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=1, train_sizes=train_sizes)
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
    plt.show()


def Validation_curve_demo(x, y, model, param_name, param_range):
    '''
    输入x, y, model, param_name, param_range
    输出：输出一个图像，用来选取最佳参数值
    '''
    train_loss, test_loss = validation_curve(
        model, x, y, param_name=param_name,
        param_range=param_range, cv=5,
        scoring='neg_mean_squared_error')
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)

    plt.plot(param_range, train_loss_mean, 'o-', color='r',
             label='Training')
    plt.plot(param_range, test_loss_mean, 'o-', color='g',
             label='Cross-validation')

    plt.xlabel(param_name)
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()


def get_feature_std(std, feature_name, limit_std):
    '''
    输入：方差，特征名称列表，限制方差
    输出：大于限制方差值得对应特征名称列表
    '''
    feature = []
    for i, j in zip(std, feature_name):
        if i < limit_std:
            feature.append(j)

    return feature


def data_process(df_train, df_test):
    """
    输入：训练特征数据，和测试特征数据
    输出：预处理之后的，训练特征数据，和测试特征数据
    :param df_train:
    :param df_test:
    :return:
    """
    scale_column = ['V0', 'V1', 'V6', 'V30']
    # 训练数据和测试数据特征分布差异较大的featute
    drop_list_1 = ['V9', 'V17', 'V22', 'V28']

    mcorr = df_train.corr()
    # 和target相关性较小的feature
    drop_list_2 = [c for c in mcorr['target'].index if abs(mcorr['target'][c]) < 0.15]

    # 方差较小的feature
    drop_list_3 = get_feature_std(df_train.std(), df_train.columns, 0.6)

    drop_label = list(set(drop_list_3 + drop_list_2))

    print(drop_label)
    df_train = df_train.drop(drop_label, axis=1)
    df_test = df_test.drop(drop_label, axis=1)

    #    #将部分特征数据标准
    #    for column in scale_column:
    #        df_train[column] = scale(df_train[column])
    #        df_test[column] = scale(df_test[column])


    return df_train, df_test


def cross_validation(x, y, model):
    """
    :param x:
    :param y:
    :param model:
    :return: 交叉验证后的误差均值
    """
    loss_list = cross_val_score(model, x, y, cv=5,
                                scoring='neg_mean_squared_error')
    return -loss_list.mean()


def Model_stack(df_train_x, df_train_y, df_test):
    svr_ = SVR(kernel='linear', degree=3, coef0=0.0, tol=0.001,
               C=1.0, epsilon=0.1, shrinking=True, cache_size=20)

    lgb_ = lgb.LGBMModel(boosting_type='gbdt', num_leaves=35,
                         max_depth=20, max_bin=255, learning_rate=0.03, n_estimator=10,
                         subsample_for_bin=2000, objective='regression', min_split_gain=0.0,
                         min_child_weight=0.001, min_child_samples=20, subsample=1.0, verbose=0,
                         subsample_freq=1, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
                         random_state=None, n_jobs=-1, silent=True)

    RF_model = RandomForestRegressor(n_estimators=50, max_depth=25, min_samples_split=20,
                                     min_samples_leaf=10, max_features='sqrt', oob_score=True,
                                     random_state=10)

    BR_model = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
                             copy_X=True, fit_intercept=True, lambda_1=1e-06,
                             lambda_2=1e-06, n_iter=300,
                             normalize=False, tol=0.0000001, verbose=False)

    linear_model = LinearRegression()
    ls = Lasso(alpha=0.00375)

    x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y,
                                                        test_size=0.6)
    rg = RidgeCV(cv=5)

    stack = pd.DataFrame()
    stack_test = pd.DataFrame()

    ls.fit(x_train, y_train)
    lgb_.fit(x_train, y_train)
    RF_model.fit(x_train, y_train)
    svr_.fit(x_train, y_train)
    linear_model.fit(x_train, y_train)
    BR_model.fit(x_train, y_train)

    stack['rf'] = ls.predict(x_test)
    stack['adaboost'] = lgb_.predict(x_test)
    stack['gbdt'] = RF_model.predict(x_test)
    stack['lightgbm'] = svr_.predict(x_test)
    stack['linear_model'] = linear_model.predict(x_test)
    stack['BR'] = BR_model.predict(x_test)
    print('stacking_model: ', cross_validation(stack, y_test, rg))

    rg.fit(stack, y_test)
    stack_test['rf'] = ls.predict(df_test)
    stack_test['adaboost'] = lgb_.predict(df_test)
    stack_test['gbdt'] = RF_model.predict(df_test)
    stack_test['lightgbm'] = svr_.predict(df_test)
    stack_test['linear_model'] = linear_model.predict(df_test)
    stack_test['BR'] = BR_model.predict(df_test)

    # final_ans = rg.predict(stack_test)
    # pd.DataFrame(final_ans).to_csv('predict_drop2+3.txt', index=False, header=False)


if __name__ == "__main__":

    df_train = pd.read_table('data/zhengqi_train.txt')  # 获得训练数据
    df_test = pd.read_table('data/zhengqi_test.txt')  # 得到预测的数据s

    df_train, df_test = data_process(df_train, df_test)

    df_train_x = df_train.drop(['target'], axis=1)
    df_train_y = df_train['target']

    # pca = PCA(n_components = 0.95)
    pca = PCA(n_components='mle', svd_solver='full')
    pca.fit(df_train_x)
    df_train_x = pca.transform(df_train_x)
    df_test = pca.transform(df_test)

    # LinearRegression
    line_R_model = LinearRegression()

    ARD_model = ARDRegression(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
                              copy_X=True, fit_intercept=True, lambda_1=1e-06,
                              lambda_2=1e-06, n_iter=300,
                              normalize=False, threshold_lambda=10000.0,
                              tol=0.001, verbose=False)

    BR_model = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
                             copy_X=True, fit_intercept=True, lambda_1=1e-06,
                             lambda_2=1e-06, n_iter=30,
                             normalize=False, tol=0.0000001, verbose=True)

    myGBR = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                                      learning_rate=0.01, loss='huber', max_depth=14,
                                      max_features='sqrt', max_leaf_nodes=None,
                                      min_impurity_decrease=0.0, min_impurity_split=None,
                                      min_samples_leaf=10, min_samples_split=40,
                                      min_weight_fraction_leaf=0.0, n_estimators=300,
                                      presort='auto', random_state=10, subsample=0.8,
                                      verbose=0, warm_start=False)

    RF_model = RandomForestRegressor(n_estimators=50, max_depth=25, min_samples_split=20,
                                     min_samples_leaf=10, max_features='sqrt',
                                     oob_score=True, random_state=10)

    Model_stack(df_train_x, df_train_y, df_test)

    loss = kfold_loss(df_train_x, df_train_y)
    print('keras_model     : ', loss)
    print('linearRegression: ', cross_validation(df_train_x, df_train_y, line_R_model))
    print('BayesianRidge   : ', cross_validation(df_train_x, df_train_y, BR_model))
    print('GradientBosting : ', cross_validation(df_train_x, df_train_y, myGBR))
    print('ARDRegression   : ', cross_validation(df_train_x, df_train_y, ARD_model))
    print('RandomForest    : ', cross_validation(df_train_x, df_train_y, RF_model))

    # # 将最终预测结果保存到文件当中
    # BR_model.fit(df_train_x, df_train_y)
    # final_ans = BR_model.predict(df_test)
    # pd.DataFrame(final_ans).to_csv('predict_BR_drop_3.txt', index=False, header=False)
