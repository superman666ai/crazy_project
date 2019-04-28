# -*- coding: utf-8 -*-

# @Time    : 2019-04-28 15:44
# @Author  : jian
# @File    : test2.py
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})
import seaborn as sns

# modelling
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler

# load_dataset
with open("data/zhengqi_train.txt")  as fr:
    data_train = pd.read_table(fr, sep="\t")
with open("data/zhengqi_test.txt") as fr_test:
    data_test = pd.read_table(fr_test, sep="\t")

# merge train_set and test_set
data_train["oringin"] = "train"
data_test["oringin"] = "test"
data_all = pd.concat([data_train, data_test], axis=0, ignore_index=True)
# View data
print(data_all.shape)

# 删除特征"V5","V9","V11","V17","V22","V28"，训练集和测试集分布不均
# for column in ["V5", "V9", "V11", "V17", "V22", "V28"]:
#     g = sns.kdeplot(data_all[column][(data_all["oringin"] == "train")], color="Red", shade=True)
#     g = sns.kdeplot(data_all[column][(data_all["oringin"] == "test")], ax=g, color="Blue", shade=True)
#     g.set_xlabel(column)
#     g.set_ylabel("Frequency")
#     g = g.legend(["train", "test"])
#     plt.show()

data_all.drop(["V5", "V9", "V11", "V17", "V22", "V28"], axis=1, inplace=True)

print(data_all.shape)

# figure parameters
data_train1 = data_all[data_all["oringin"] == "train"].drop("oringin", axis=1)

# Threshold for removing correlated variables
threshold = 0.1
# Absolute value correlation matrix
corr_matrix = data_train1.corr().abs()
drop_col = corr_matrix[corr_matrix["target"] < threshold].index
data_all.drop(drop_col, axis=1, inplace=True)
print(data_all.shape)

# normalise numeric columns
cols_numeric = list(data_all.columns)
cols_numeric.remove("oringin")


def scale_minmax(col):
    return (col - col.min()) / (col.max() - col.min())


scale_cols = [col for col in cols_numeric if col != 'target']
data_all[scale_cols] = data_all[scale_cols].apply(scale_minmax, axis=0)

cols_transform = data_all.columns[0:-2]
for col in cols_transform:
    # transform column
    data_all.loc[:, col], _ = stats.boxcox(data_all.loc[:, col] + 1)

# print(data_all.target.describe())

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.distplot(data_all.target.dropna(), fit=stats.norm);
plt.subplot(1, 2, 2)
_ = stats.probplot(data_all.target.dropna(), plot=plt)

# Log Transform SalePrice to improve normality
sp = data_train.target
data_train.target1 = np.power(1.5, sp)


# print(data_train.target1.describe())


# function to get training samples
def get_training_data():
    # extract training samples
    from sklearn.model_selection import train_test_split
    df_train = data_all[data_all["oringin"] == "train"]
    df_train["label"] = data_train.target1
    # split SalePrice and features
    y = df_train.target
    X = df_train.drop(["oringin", "target", "label"], axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=100)
    return X_train, X_valid, y_train, y_valid


# extract test data (without SalePrice)
def get_test_data():
    df_test = data_all[data_all["oringin"] == "test"].reset_index(drop=True)
    return df_test.drop(["oringin", "target"], axis=1)


from sklearn.metrics import make_scorer


# metric for evaluation
def rmse(y_true, y_pred):
    diff = y_pred - y_true
    sum_sq = sum(diff ** 2)
    n = len(y_pred)

    return np.sqrt(sum_sq / n)


def mse(y_ture, y_pred):
    return mean_squared_error(y_ture, y_pred)


# scorer to be used in sklearn model fitting
rmse_scorer = make_scorer(rmse, greater_is_better=False)
mse_scorer = make_scorer(mse, greater_is_better=False)


# function to detect outliers based on the predictions of a model
def find_outliers(model, X, y, sigma=3):
    # predict y values using model
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    # if predicting fails, try fitting the model first
    except:
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=y.index)

    # calculate residuals between the model prediction and true y values
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # calculate z statistic, define outliers to be where |z|>sigma
    z = (resid - mean_resid) / std_resid
    outliers = z[abs(z) > sigma].index

    # print and plot the results
    print('R2=', model.score(X, y))
    print('rmse=', rmse(y, y_pred))
    print("mse=", mean_squared_error(y, y_pred))
    print('---------------------------------------')

    print('mean of residuals:', mean_resid)
    print('std of residuals:', std_resid)
    print('---------------------------------------')

    print(len(outliers), 'outliers:')
    print(outliers.tolist())

    plt.figure(figsize=(15, 5))
    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(y, y_pred, '.')
    plt.plot(y.loc[outliers], y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y_pred');

    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y - y_pred, '.')
    plt.plot(y.loc[outliers], y.loc[outliers] - y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y - y_pred');

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax=ax_133)
    z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('z')

    plt.savefig('outliers.png')

    return outliers


# get training data
from sklearn.linear_model import Ridge

X_train, X_valid, y_train, y_valid = get_training_data()
test = get_test_data()

# find and remove outliers using a Ridge model
outliers = find_outliers(Ridge(), X_train, y_train)

# permanently remove these outliers from the data
# df_train = data_all[data_all["oringin"]=="train"]
# df_train["label"]=data_train.target1
# df_train=df_train.drop(outliers)
X_outliers = X_train.loc[outliers]
y_outliers = y_train.loc[outliers]
X_t = X_train.drop(outliers)
y_t = y_train.drop(outliers)


def get_trainning_data_omitoutliers():
    y1 = y_t.copy()
    X1 = X_t.copy()
    return X1, y1


from sklearn.preprocessing import StandardScaler


def train_model(model, param_grid=[], X=[], y=[],
                splits=5, repeats=5):
    # get unmodified training data, unless data to use already specified
    if len(y) == 0:
        X, y = get_trainning_data_omitoutliers()
        # poly_trans=PolynomialFeatures(degree=2)
        # X=poly_trans.fit_transform(X)
        # X=MinMaxScaler().fit_transform(X)

    # create cross-validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)

    # perform a grid search if param_grid given
    if len(param_grid) > 0:
        # setup grid search parameters
        gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                               scoring="neg_mean_squared_error",
                               verbose=1, return_train_score=True)

        # search the grid
        gsearch.fit(X, y)

        # extract best model from the grid
        model = gsearch.best_estimator_
        best_idx = gsearch.best_index_

        # get cv-scores for best model
        grid_results = pd.DataFrame(gsearch.cv_results_)
        cv_mean = abs(grid_results.loc[best_idx, 'mean_test_score'])
        cv_std = grid_results.loc[best_idx, 'std_test_score']

    # no grid search, just cross-val score for given model
    else:
        grid_results = []
        cv_results = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=rkfold)
        cv_mean = abs(np.mean(cv_results))
        cv_std = np.std(cv_results)

    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean': cv_mean, 'std': cv_std})

    # predict y using the fitted model
    y_pred = model.predict(X)

    # print stats on model performance
    print('----------------------')
    print(model)
    print('----------------------')
    print('score=', model.score(X, y))
    print('rmse=', rmse(y, y_pred))
    print('mse=', mse(y, y_pred))
    print('cross_val: mean=', cv_mean, ', std=', cv_std)

    # residual plots
    y_pred = pd.Series(y_pred, index=y.index)
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()
    z = (resid - mean_resid) / std_resid
    n_outliers = sum(abs(z) > 3)

    plt.figure(figsize=(15, 5))
    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(y, y_pred, '.')
    plt.xlabel('y')
    plt.ylabel('y_pred')
    plt.title('corr = {:.3f}'.format(np.corrcoef(y, y_pred)[0][1]))
    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y - y_pred, '.')
    plt.xlabel('y')
    plt.ylabel('y - y_pred')
    plt.title('std resid = {:.3f}'.format(std_resid))

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax=ax_133)
    plt.xlabel('z')
    plt.title('{:.0f} samples with z>3'.format(n_outliers))

    return model, cv_score, grid_results


# places to store optimal models and scores
opt_models = dict()
score_models = pd.DataFrame(columns=['mean', 'std'])

# no. k-fold splits
splits = 5
# no. k-fold iterations
repeats = 5

"""
"""
model = 'Ridge'
opt_models[model] = Ridge()
alph_range = np.arange(0.25, 6, 0.25)
param_grid = {'alpha': alph_range}
opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid,
                                                        splits=splits, repeats=repeats)

cv_score.name = model
score_models = score_models.append(cv_score)

# plt.figure()
# plt.errorbar(alph_range, abs(grid_results['mean_test_score']),
#              abs(grid_results['std_test_score'])/np.sqrt(splits*repeats))
# plt.xlabel('alpha')
# plt.ylabel('score')
# plt.show()
#
#
# model = 'Lasso'
#
# opt_models[model] = Lasso()
# alph_range = np.arange(1e-4,1e-3,4e-5)
# param_grid = {'alpha': alph_range}
#
# opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid,
#                                               splits=splits, repeats=repeats)
#
# cv_score.name = model
# score_models = score_models.append(cv_score)
#
# plt.figure()
# plt.errorbar(alph_range, abs(grid_results['mean_test_score']),abs(grid_results['std_test_score'])/np.sqrt(splits*repeats))
# plt.xlabel('alpha')
# plt.ylabel('score')
