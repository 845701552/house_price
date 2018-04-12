import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
train=pd.read_csv("E:/pycharm project/compitition/train.csv")
test=pd.read_csv("E:/pycharm project/compitition/test.csv")
def train_corr(train):
    corrmat = train.corr()
    plt.subplots(figsize=(14,15))
    sns.heatmap(corrmat,vmax=1)
    plt.show()
    '''线性关系：saleprice correlation matrix
    '''
    k = 10  # number of variables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    # plt.show()
    '''分析强相关的，处理异常点，组合特征
    非线性：分批次：10 with saleprice and using pairplot,'''
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    good_corr=train[cols]
    good_corr.to_csv("C:/Users/yu/Desktop/good_corr.csv")
    # sns.pairplot(train[cols], size=2.5)
    # plt.show()
    '''缺失率，结合相关性程度填充'''
    # ratio_null=train.isnull().sum()/train.isnull().count()
    # ratio_null=ratio_null[ratio_null.values>0].sort_values(ascending=False)
    # print(ratio_null)
    # f, ax = plt.subplots(figsize=(15, 12))
    # plt.xticks(rotation='90')
    # sns.barplot(x=ratio_null.index, y=ratio_null)
    # plt.xlabel('Features', fontsize=15)
    # plt.show()
    return cols
# train_num_num=train_corr(train)
print(skew(train["GrLivArea"]))
print(skew(train["TotalBsmtSF"]))

def outer_num(y,train_num_num):
    for i in train_num_num.columns:
        plt.plot(x=i, y='SalePrice', ylim=(0, 800000),kind="scatter");
# outer_num(y,train_num_num)

def outer_object(y,train_num_object,train_object):
    for i in train_num_object:
        plt.plot(x=var, y='SalePrice', ylim=(0, 800000), kind="");




# '''查找 空格'''
# data=np.isnan(train).any()
# data=data[data==True]
# print(data)
