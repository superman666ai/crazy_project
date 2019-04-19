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
from utils import save_result

df = pd.read_csv('data/zhengqi_train.txt', sep='\t')

# 结果集
test_df = pd.read_csv('data/zhengqi_test.txt', sep='\t')

# 数据处理

# 剔除认为不重要的特征
df.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)
test_df.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)

x = df.iloc[:, :-1]
y = df.target

# # 标准化特征
mm = MinMaxScaler()
x = mm.fit_transform(x)


# 结果集标准
# test_df = mm.transform(test_df)

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)



# 简单回归模型
def simple_regression(x_train, x_test, y_train, y_test):
    # 线性回归
    # model = LinearRegression()

    # 岭回归
    # model = Ridge(alpha=0.1)

    # lasso 回归
    model = Lasso(alpha=0.1)

    # kernal ridge
    # model = KernelRidge(kernel='rbf', alpha=0.1)
    # model = KernelRidge(kernel='linear', alpha=0.1)
    # model = KernelRidge(kernel='sigmoid', alpha=0.1)
    # model = KernelRidge(kernel='poly', alpha=0.1)
    # model = KernelRidge(kernel='laplacian', alpha=0.1)
    # model = KernelRidge(kernel='cosine', alpha=0.1)

    # 不知道为什么用不了
    # model = KernelRidge(kernel='chi2', alpha=0.1)
    # model = KernelRidge(kernel='additive_chi2', alpha=0.1)

    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)

    print("score", score)
    y_pred = model.predict(x_test)

    print("mean_squared_error", mean_squared_error(y_test, y_pred))

    plot_learning_curve(model, "rate", x_train, y_train)
    result = model.predict(test_df)
    save_result(result)
    print(result)


# svm
def svr_regression(x_train, x_test, y_train, y_test):
    # model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    # model = SVR(kernel='linear', C=1e3)
    model = SVR(kernel='poly', C=1e3, degree=2)

    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)

    print("score", score)
    y_pred = model.predict(x_test)

    print("mean_squared_error", mean_squared_error(y_test, y_pred))

    # 保存结果
    # result = model.predict(test_df)
    # print(result)
    # result_df = pd.DataFrame(result, columns=['target'])
    # result_df.to_csv("0.098.txt", index=False, header=False)

    # 绘制学习率曲线
    plot_learning_curve(model, title="learn_rate", X=x_train, y=y_train)


# 贝叶斯回归
def bayes_regression(x_train, x_test, y_train, y_test):
    model = BayesianRidge()

    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)

    print("score", score)
    y_pred = model.predict(x_test)

    print("mean_squared_error", mean_squared_error(y_test, y_pred))

    # 保存结果
    # result = model.predict(test_df)
    # print(result)
    # result_df = pd.DataFrame(result, columns=['target'])
    # result_df.to_csv("0.098.txt", index=False, header=False)

    # 绘制学习率曲线
    plot_learning_curve(model, title="learn_rate", X=x_train, y=y_train, cv=10)


# random forest
def random_forest_regression(x_train, x_test, y_train, y_test):
    # 随机森林
    # model = RandomForestRegressor()

    # 极端随机森林
    # model =ExtraTreesRegressor()

    # 梯度提升
    # model = GradientBoostingRegressor()

    # xgboost
    model = xgb.XGBRegressor()

    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)

    print("score", score)
    y_pred = model.predict(x_test)

    print("mean_squared_error", mean_squared_error(y_test, y_pred))

    # 保存结果
    # result = model.predict(test_df)
    # print(result)
    # result_df = pd.DataFrame(result, columns=['target'])
    # result_df.to_csv("0.098.txt", index=False, header=False)

    # 绘制学习率曲线
    plot_learning_curve(model, title="learn_rate", X=x_train, y=y_train, cv=5)


def mlp_model(x_train, x_test, y_train, y_test):

    model = MLPRegressor(solver="lbfgs")

    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)

    print("score", score)
    y_pred = model.predict(x_test)

    print("mean_squared_error", mean_squared_error(y_test, y_pred))

    # 保存结果
    result = model.predict(test_df)
    print(result)
    result_df = pd.DataFrame(result, columns=['target'])
    result_df.to_csv("0.098.txt", index=False, header=False)

    # 绘制学习率曲线
    # plot_learning_curve(model, title="learn_rate", X=x_train, y=y_train, cv=5)


if __name__ == "__main__":
    # 简单回归
    # simple_regression(x_train, x_test, y_train, y_test)

    # svm
    # svr_regression(x_train, x_test, y_train, y_test)

    # 贝叶斯回归
    # bayes_regression(x_train, x_test, y_train, y_test)

    # 随机森林
    random_forest_regression(x_train, x_test, y_train, y_test)

    # 神经
    # mlp_model(x_train, x_test, y_train, y_test)

    """
    参数调优：
    # Ridge Lasso alpha=0.1
    
    
    总结：
    Linear Rregression 效果中规中矩 88%
    Ridge Rregression 回归效果不错  准确率在88% 
    Lasso Rregression 回归效果特别差
    
    SVR kernel=rbf  准确率85% 学习率图显示如果训练数据集增加 可能会增加准确率
    SVR kernel=linear 准确率在85% 
    SVR kernel=poly 准确率在87% 增加训练样本 可能会提高准确率
    
    bayes_regression 准确率在87%
    
    random_forest_regression 准确率在80% 
    ExtraTreesRegressor  
    GradientBoostingRegressor 准确率在85% 增加样本集可以增加准确率
    
    
    """
