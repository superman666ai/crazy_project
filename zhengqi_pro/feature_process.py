# -*- coding: utf-8 -*-

# @Time    : 2019-04-24 14:59
# @Author  : jian
# @File    : feature_process.py

def drop_some(df):
    """
    剔除认为不重要的特征
    :param df:
    :return:
    """
    df.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)
    return df

