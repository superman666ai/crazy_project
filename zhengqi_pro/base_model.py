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
from utils import plot_learning_curve_new
from utils import plot_learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv('data/zhengqi_train.txt', sep='\t')
test_df = pd.read_csv('data/zhengqi_test.txt', sep='\t')
# 数据处理
drop_label = ['V32', 'V35', 'V17', 'V28', 'V21', 'V14', 'V22', 'V26', 'V5', 'V19', 'V27', 'V25', 'V29',
              'V33', 'V34', 'V9']
df.drop(drop_label, axis=1, inplace=True)
test_df.drop(drop_label, axis=1, inplace=True)

x = df.iloc[:, :-1]
y = df.target
# # 使用处理过的数据集
# from a import train_x, Y
# x, y = train_x, Y
# 查看相关性
# from utils import heat_grape
# heat_grape(x)

# # 标准化特征
# mm = MinMaxScaler()
# x = mm.fit_transform(x)

# 结果集标准
# test_df = mm.transform(test_df)

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)


n_folds = 10
def rmsle_cv(model, train_x_head=x_train):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_x_head)
    rmse = -cross_val_score(model, train_x_head, y_train, scoring="neg_mean_squared_error", cv=kf)
    return (rmse)



# 线性回归
# model = LinearRegression()
# model.fit(x_train, y_train)
# score = model.score(x_test, y_test)
# print("线性回归 score", score)
# y_pred = model.predict(x_test)
# print("线性回归 mean_squared_error", mean_squared_error(y_test, y_pred))

# 岭回归
# model = Ridge(alpha=0.1)
# model.fit(x_train, y_train)
# score = model.score(x_test, y_test)
# print("岭回归 score", score)
# y_pred = model.predict(x_test)
# print("岭回归 mean_squared_error", mean_squared_error(y_test, y_pred))
# 绘制学习率曲线
# plot_learning_curve(model, title="带核函数的ridge learn_rate", X=x_train, y=y_train, cv=None)

# 带核函数的ridge
# model = KernelRidge(kernel='linear', alpha=0.1)
# model.fit(x_train, y_train)
# score = model.score(x_test, y_test)
# print("带核函数的ridge score", score)
# y_pred = model.predict(x_test)
# print("带核函数的ridge mean_squared_error", mean_squared_error(y_test, y_pred))
# 绘制学习率曲线
# # plot_learning_curve(model, title="带核函数的ridge learn_rate", X=x_train, y=y_train, cv=None)


# lasso 回归
# model = Lasso(alpha=0.1)
# model.fit(x_train, y_train)
# score = model.score(x_test, y_test)
# print("score", score)
# y_pred = model.predict(x_test)
# print("lasso回归 mean_squared_error", mean_squared_error(y_test, y_pred))
# 绘制学习率曲线
# plot_learning_curve(model, title="lasso回归 learn_rate", X=x_train, y=y_train, cv=None)


svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
# model = SVR(kernel='linear', C=1e3)
# model = SVR(kernel='poly', C=1e3, degree=2)
bays = BayesianRidge()
score = rmsle_cv(bays)
print("svr: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
svr.fit(x_train, y_train)


# # 绘制学习率曲线
# plot_learning_curve(model, title="SVR learn_rate", X=x_train, y=y_train, cv=None)


#
#     # 保存结果
#     # result = model.predict(test_df)
#     # print(result)
#     # result_df = pd.DataFrame(result, columns=['target'])
#     # result_df.to_csv("0.098.txt", index=False, header=False)


# # 贝叶斯回归
bays = BayesianRidge()
score = rmsle_cv(bays)
print(" bays: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
bays.fit(x_train, y_train)


# # # 绘制学习率曲线
# plot_learning_curve(model, title="BayesianRidge learn_rate", X=x_train, y=y_train, cv=200)

# # random forest

# 随机森林
my_random = RandomForestRegressor(n_estimators=360,  # 学习器个数
                              # criterion='mse',  # 评价函数
                              max_depth=23,  # 最大的树深度，防止过拟合
                              min_samples_split=20,  # 根据属性划分节点时，每个划分最少的样本数
                              min_samples_leaf=5,  # 最小叶子节点的样本数，防止过拟合
                              max_features='auto',  # auto是sqrt(features)还有 log2 和 None可选
                              max_leaf_nodes=None,  # 叶子树的最大样本数
                              bootstrap=True,  # 有放回的采样
                              min_weight_fraction_leaf=0,
                              n_jobs=5)  # 同时用多少个进程训练


score = rmsle_cv(my_random)
print(" my_random: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
my_random.fit(x_train, y_train)

# 绘制学习率曲线

# plot_learning_curve(model, title="random learn_rate", X=x_train, y=y_train, cv=10)

# 极端随机森林
# model =ExtraTreesRegressor()

# 梯度提升
# model = GradientBoostingRegressor()

# xgboost
# model = xgb.XGBRegressor()
# model.fit(x_train, y_train)
# score = model.score(x_test, y_test)
# print("xgboost score", score)
# y_pred = model.predict(x_test)
# print(" xgboost mean_squared_error", mean_squared_error(y_test, y_pred))
# 绘制学习率曲线
# plot_learning_curve(model, title="learn_rate", X=x_train, y=y_train, cv=20)


# 保存结果
# result = model.predict(test_df)
# print(result)
# result_df = pd.DataFrame(result, columns=['target'])
# result_df.to_csv("0.098.txt", index=False, header=False)


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # 遍历所有模型，你和数据
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            model.fit(X, y)

        return self

    # 预估，并对预估结果值做average
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        # return 0.85*predictions[:,0]+0.15*predictions[:,1]
        # return 0.7*predictions[:,0]+0.15*predictions[:,1]+0.15*predictions[:,2]
        return np.mean(predictions, axis=1)
        # averaged_models = AveragingModels(models = (lasso,KRR))


print("1111")
averaged_models = AveragingModels(models=(svr, my_random, bays))

score = rmsle_cv(averaged_models)
print(" 对基模型集成后的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

print("33333")