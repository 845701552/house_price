import pandas as pd
from scipy.stats import skew,probplot
from scipy.special import boxcox1p
import numpy as np
import seaborn as sns
from time import time
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
train=pd.read_csv("E:/pycharm project/compitition/train.csv")
test=pd.read_csv("E:/pycharm project/compitition/test.csv")
'''show:根据强相关变量，专注分析强变量，
1.的缺失值(重要的缺失值可以设置为is_bool,包括连续性的为0可以设置为没有)，
2.异常点
3.特征组合，其他正常处理
4.log'''
train.drop(train[(train["SalePrice"] <200000) & (train["TotalBsmtSF"]>5000)].index, inplace=True)
train.drop(train[(train["SalePrice"]<200000) & (train["GrLivArea"]>4000)].index,inplace=True)
def fill_na(train,test):
    ratio_null = train.isnull().sum() / train.isnull().count()
    ratio_null = ratio_null[ratio_null.values > 0].sort_values(ascending=False)
    ratio_null_over70=ratio_null[ratio_null.values > 0.7].index
    for col in ratio_null_over70:
        if train[col].dtype not in [object,"category"]:
            train[col]=train[col].fillna(train[col].mean())
            test[col] = test[col].fillna(test[col].mean())
        else:
            train[col] = train[col].fillna("None")
            test[col] = test[col].fillna("None")

    # train["FireplaceQu"]=train.groupby("Fireplaces")["FireplaceQu"].transform(
    # lambda x: x.fillna(x.mode().iloc[0]))
    # test["FireplaceQu"] = test.groupby("Fireplaces")["FireplaceQu"].transform(
    #     lambda x: x.fillna(x.mode().iloc[0]))
    train["LotFrontage"] = train["LotFrontage"].fillna(train["LotFrontage"].mean())
    test["LotFrontage"] = test["LotFrontage"].fillna(test["LotFrontage"].mean())
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars',"FireplaceQu"):
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)

    for col in ["GarageType","GarageFinish","GarageQual","GarageCond"]:
        if train[col].dtype not in [object,"category"]:
            train[col] = train[col].fillna(0)
            test[col] = test[col].fillna(0)
        else:
            test[col] = test[col].fillna("None")
            train[col] = train[col].fillna("None")

    for col in ["MasVnrType","Electrical","BsmtFinType2","BsmtFinType1","BsmtCond","BsmtQual","BsmtExposure"]:
        train[col] = train[col].fillna("None")
        test[col] = test[col].fillna("None")
    test["MasVnrArea"] = test["MasVnrArea"].fillna(test["MasVnrArea"].mean())
    train["MasVnrArea"] = train["MasVnrArea"].fillna(train["MasVnrArea"].mean())
# train.drop(["Id","PoolQC","Alley","FireplaceQu","MiscFeature","Fence"],axis=1,inplace=True)
# test.drop(["Id","PoolQC","Alley","FireplaceQu","MiscFeature","Fence"],axis=1,inplace=True)
#判断数值与类目关系
fill_na(train,test)
train.drop(["Id"],axis=1,inplace=True)
test.drop(["Id"],axis=1,inplace=True)
'''拿出应变量'''
SalePrice=train["SalePrice"]

train.drop(["SalePrice"],inplace=True,axis=1)
'''添加特征'''
train_shape=train.shape[0]
train=pd.concat([train,test])
# train_num=train_num.fillna(train_num.mean()) #解决思路：先用正则将空格匹配出来，然后全部替换为NULL，再在用pandas读取csv时候指定 read_csv（na_values='NULL'）就是将NULL认为是nan处理，接下来就可以用dropna()或者fillna()来处理了
# train_object=train_object.fillna(train_object.mode().iloc[0])

def add_feature(train):
    #train["OverallQual_rito"]=train["OverallQual"]/max(train["OverallQual"])
    train.loc[train["YearBuilt"]<1900,"YearBuilt_split"]=1
    train.loc[train["YearBuilt"] >=2000, "YearBuilt_split"] = 4
    train.loc[(train["YearBuilt"]<1950) & (train["YearBuilt"]>=1900),"YearBuilt_split"] = 2
    train.loc[(train["YearBuilt"] >= 1950) & (train["YearBuilt"] < 2000), "YearBuilt_split"] = 3

    # train["is_Gar"]=train.loc[train["GarageType"]!="NA","is_Gar"]=1
    # train["is_Gar"] = train.loc[train["GarageType"]=="NA", "is_Gar"]==0
    train["GrLivArea+TotalBsmtSF"]=train["GrLivArea"]+train["TotalBsmtSF"]
    train["is_GarageCars"]=np.where(train["GarageCars"] > 0, 1,0)
    train["YearBuilt_from _now"]=2018-train["YearBuilt"]
    train['GarageYrBlt_from _now']=2018-train['GarageYrBlt']
    train['GarageYrBlt_from _now']=2018-train["GarageYrBlt"]
    train['YearRemodAdd_from _now'] = 2018 - train['YearRemodAdd']
    train["YearRemodAdd-YearBuilt"]=train['YearRemodAdd']-train["YearBuilt"]
    train["LotArea_log"]=np.log(train["LotArea"]+1)
    #train["LotArea_2"] = train["LotArea"]**2
    #train["GarageYrBlt-YearBuilt"]=train["GarageYrBlt"]-train["YearBuilt"]
    train["BsmtFinSF2+BsmtFinSF1"]=train["BsmtFinSF2"]+train["BsmtFinSF1"]
    train["BsmtFullBath+BsmtHalfBath"]=train["BsmtFullBath"]+train["BsmtHalfBath"]*0.5
    train["is_base"]=np.where(train["TotalBsmtSF"]>0,1,0)
    train["is_BsmtFullBath+BsmtHalfBath"] = np.where(train["BsmtFullBath+BsmtHalfBath"] > 0, 1, 0)
    train["is_TotalBsmtSF"]= np.where(train["TotalBsmtSF"] > 0, 1,0)
    train["TotalBsmtSF/BsmtFullBath+BsmtHalfBath"]= train["TotalBsmtSF"]/(train["BsmtFullBath+BsmtHalfBath"]+0.5)
    train["is_FullBath"]=np.where(train["FullBath"] > 0, 1, 0)
    train["FullBath+HalfBath"]=train["FullBath"]+train["HalfBath"]*0.5
    train["is_FullBath+HalfBath"]=np.where(train["FullBath+HalfBath"]>0,1,0)
    train["is_BedroomAbvGr"]=np.where(train["BedroomAbvGr"]>0,1,0)
    train["is_KitchenAbvGr"]=np.where(train["KitchenAbvGr"]>0,1,0)
    train["YrSold-YearBuilt"]=train["YrSold"]-train["YearBuilt"]
add_feature(train)
SalePrice=np.log1p(SalePrice)
res=probplot(SalePrice,plot=plt)
sns.distplot(SalePrice)

train_num=train.select_dtypes(exclude=[object,'category'])
train_object=train.select_dtypes(include=[object,'category'])

for col in train_num.columns:
    if len(set(train_num[col]))<=25:
        train[col]=train_num[col].astype('category')

train_num_num=train_num.select_dtypes(exclude=[object,'category'])
train_num_object=train_num.select_dtypes(include=[object,'category'])#连续lable编码
#label后类型为int
def label_EN(train_num_object):
    for col in train_num_object:
        lbl=LabelEncoder()
        lbl.fit(train_num_object[col].values)
        train_object[col]=lbl.transform(train_num_object[col])
    return train_num_object
train_num_object=label_EN(train_num_object)

'''包含bool,category,int float'''
skewness = train_num_num.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print(skewness)
def box_cox(train_num_num):
    skewness_box = skewness[abs(skewness) > 0.75]
    for col in skewness_box.index:
        train_num[col]=boxcox1p(train_num_num[col],0.15)
    return train_num
train_num_num=box_cox(train_num_num)


train=pd.concat([train_num_num,train_num_object,train_object],axis=1)
for i in train.columns:
    if train[i].dtype not in [object,bool]:
        train[i]=train[i].fillna(train[i].mean())
    else:
        train[i] = train[i].fillna(train[i].mode().iloc[0])
train=pd.get_dummies(train)#只对类目转换，提前将顺序的类目进行进行label编码，变为int后再进行get_dummies



# data=np.isnan(train).any()
# data=data[data==True]
# print(data)
# y=SalePrice

X=train[:train_shape]

train_final=pd.concat([X,y],axis=1)
train_final.to_csv("E:/pycharm project/compitition/train_preprocessing_fillna_addfeat_goodcorr.csv",index=False)
test=train[train_shape:]
test.to_csv("E:/pycharm project/compitition/test_preprocessing_fillna_addfeat_goodcorr.csv",index=False)
print("ok")