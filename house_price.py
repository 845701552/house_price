import pandas as pd
from scipy.stats import skew,probplot
from scipy.special import boxcox1p
import numpy as np
import seaborn as sns
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR,libsvm
from sklearn.linear_model import LassoCV,RidgeCV,LinearRegression,ElasticNet,ElasticNetCV
from sklearn.model_selection import GridSearchCV
import pickle
import sklearn.tree.tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
train=pd.read_csv("E:/pycharm project/compitition/train.csv")
test=pd.read_csv("E:/pycharm project/compitition/test.csv")
train.drop(["Id","PoolQC","Alley","FireplaceQu","MiscFeature","Fence"],axis=1,inplace=True)
test.drop(["Id","PoolQC","Alley","FireplaceQu","MiscFeature","Fence"],axis=1,inplace=True)
#判断数值与类目关系

SalePrice=train["SalePrice"]
train.drop(["SalePrice"],inplace=True,axis=1)
train=pd.concat([train,test])

train_num=train.select_dtypes(exclude=[object,'category'])
train_object=train.select_dtypes(include=[object,'category'])
train_num=train_num.fillna(train_num.mean()) #解决思路：先用正则将空格匹配出来，然后全部替换为NULL，再在用pandas读取csv时候指定 read_csv（na_values='NULL'）就是将NULL认为是nan处理，接下来就可以用dropna()或者fillna()来处理了
train_object=train_object.fillna(train_object.mode().iloc[0])

SalePrice=np.log1p(SalePrice)
res=probplot(SalePrice,plot=plt)
sns.distplot(SalePrice)
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
skewed_feats = train_num.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame(skewed_feats)
def box_cox(train_num):
    skewness_box = skewness[abs(skewness) > 0.75]
    for col in skewness.index:
        train_num[col]=boxcox1p(train_num[col],0.15)
    return train_num
train_num_num=box_cox(train_num_num)


train=pd.concat([train_num_num,train_num_object,train_object],axis=1)

train=pd.get_dummies(train)#只对类目转换，提前将顺序的类目进行进行label编码，变为int后再进行get_dummies

y=SalePrice

X=train[:1460]

train_final=pd.concat([X,y],axis=1)
train_final.to_csv("E:/pycharm project/compitition/train_preprocessing.csv",index=False)
test=train[1460:]
test.to_csv("E:/pycharm project/compitition/test_preprocessing.csv",index=False)
train_X,test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 42)







GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=5)


ELNET=ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
def cross_val_score1(model,X_train,y_train):
    cv=KFold( 5,shuffle=True, random_state=42)
    mean_score1=np.mean(cross_val_score(model,X_train,y_train,cv=cv,scoring="neg_mean_squared_error"))
    print("train_score:"+str(mean_score1))
# cross_val_score1(GBoost,train_X,train_y)
# cross_val_score1(ELNET,train_X,train_y)
# R2方法是将预测值跟只使用均值的情况下相比，看能好多少。其区间通常在（0,1）之间。0表示还不如什么都不预测，直接取均值的情况，而1表示所有预测跟真实结果完美匹配的情况。
# ””’ 与均值相比的优秀程度，介于[0~1]。0表示不如均值。1表示完美预测. ”’
# def R2(y_test, y_true):
# return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()



def elas(train_X,train_y,test_X,test_y):
    stan=StandardScaler()
    stan.fit(train_X)
    train_X=stan.transform(train_X)
    test_X=stan.transform(test_X)
    starttime = time()
    clf = ElasticNetCV(fit_intercept=True, normalize=False)
    clf.fit(train_X, train_y)
    result = clf.predict(test_X)
    print("弹性网格均方根：%f" % np.sqrt(mean_squared_error(test_y, result)))
    print("\033[1;34m")
    print("弹性网格_r2得分：%f" % r2_score(test_y, result))
    print("\033[0m")
    print("弹性网格用时：%f" % (time() - starttime))


def linear(train_X,train_y,test_X,test_y):
    stan = StandardScaler()
    stan.fit(train_X)
    train_X = stan.transform(train_X)
    test_X = stan.transform(test_X)
    starttime=time()
    clf=LinearRegression(fit_intercept=True,normalize=False)
    clf.fit(train_X,train_y)
    result = clf.predict(test_X)
    print("linear均方根：%f" % np.sqrt(mean_squared_error(test_y, result)))
    print("********")
    print("linear_r2得分：%f" % r2_score(test_y,result))
    print("linear用时：%f" % (time()-starttime))


def lass(train_X,train_y,test_X,test_y):
    stan = StandardScaler()
    stan.fit(train_X)
    train_X = stan.transform(train_X)
    test_X = stan.transform(test_X)
    starttime = time()
    clf=LassoCV(fit_intercept=True,normalize=False,alphas=[0.01,0.1, 1.0, 10.0,100])
    clf.fit(train_X, train_y)
    result = clf.predict(test_X)
    print("lass_&: %f" % (clf.alpha_)) #得出&值后再细分0.1 np.lispace(0,1,100)
    print("lass均方根：%f" % np.sqrt(mean_squared_error(test_y, result)))
    print("********")
    print("lass_r2得分：%f" % r2_score(test_y, result))
    print("lass用时：%f" % (time()-starttime))


def ridge_cv(train_X,train_y,test_X,test_y):
    stan = StandardScaler()
    stan.fit(train_X)
    train_X = stan.transform(train_X)
    test_X = stan.transform(test_X)
    starttime = time()
    clf = RidgeCV(fit_intercept=True,alphas=[0.1, 1.0, 10.0],normalize=False)
    clf.fit(train_X, train_y)
    result = clf.predict(test_X)
    print("ridge_&: %f" % (clf.alpha_))
    print("ridge_cv均方根：%f" % np.sqrt(mean_squared_error(test_y, result)))
    print("********")
    print("ridge_cv_r2得分：%f" % r2_score(test_y, result))
    print("ridge用时：%f" % (time()-starttime))

def randomf(train_X,train_y,test_X,test_y):
    starttime = time()
    clf = RandomForestRegressor(n_estimators=300,max_features="sqrt",
                                max_depth=5,min_samples_split=15,min_samples_leaf=10)
    clf.fit(train_X, train_y)
    result = clf.predict(test_X)
    print("randomf均方根：%f" % np.sqrt(mean_squared_error(test_y, result)))
    print("********")
    print("randomf_r2得分：%f" % r2_score(test_y, result))
    print("randomf用时：%f" % (time()-starttime))

def svr(train_X,train_y,test_X,test_y):
    starttime = time()
    clf = SVR(kernel='rbf', C=1.0, epsilon=0.05)
    clf.fit(train_X, train_y)
    result = clf.predict(test_X)
    print("svr均方根：%f" % np.sqrt(mean_squared_error(test_y, result)))
    print("********")
    print("svr_r2得分：%f" % r2_score(test_y, result))
    print("svr用时：%f" % (time()-starttime))
def ada(train_X,train_y,test_X,test_y):
    starttime = time()
    clf =AdaBoostRegressor()
    clf.fit(train_X, train_y)
    result = clf.predict(test_X)
    print("ada均方根：%f" % np.sqrt(mean_squared_error(test_y, result)))
    print("********")
    print("ada_r2得分：%f" % r2_score(test_y, result))
    print("ada用时：%f" % (time()-starttime))
def decision(train_X,train_y,test_X,test_y):
    starttime = time()
    clf =DecisionTreeRegressor()
    clf.fit(train_X, train_y)
    result = clf.predict(test_X)
    print("decision均方根：%f" % np.sqrt(mean_squared_error(test_y, result)))
    print("********")
    print("decision_r2得分：%f" % r2_score(test_y, result))
    print("decision用时：%f" % (time()-starttime))


def GBDTRR(train_X,train_y,test_X,test_y):
    starttime = time()
    est = GradientBoostingRegressor(
    loss='huber',      ##默认ls损失函数'ls'是指最小二乘回归lad'（最小绝对偏差）'huber'是两者的组合
    n_estimators=1500, ##默认100 回归树个数 弱学习器个数
    learning_rate=0.1,  ##默认0.1学习速率/步长0.0-1.0的超参数  每个树学习前一个树的残差的步长
    max_depth=3,   ## 默认值为3每个回归树的深度  控制树的大小 也可用叶节点的数量max leaf nodes控制
    subsample=1,  ##用于拟合个别基础学习器的样本分数 选择子样本<1.0导致方差的减少和偏差的增加
    min_samples_split=2, ##生成子节点所需的最小样本数 如果是浮点数代表是百分比
    min_samples_leaf=1, ##叶节点所需的最小样本数  如果是浮点数代表是百分比
    max_features=None, ##在寻找最佳分割点要考虑的特征数量auto全选/sqrt开方/log2对数/None全选/int自定义几个/float百分比
    max_leaf_nodes=None, ##叶节点的数量 None不限数量
    min_impurity_decrease=1e-7, ##停止分裂叶子节点的阈值
    verbose=0,  ##打印输出 大于1打印每棵树的进度和性能
    warm_start=False, ##True在前面基础上增量训练 False默认擦除重新训练 增加树
    random_state=0  ##随机种子-方便重现
    )
    est.fit(train_X,train_y)
    result=est.predict(test_X)
    print("gbdt均方差根: %f" % np.sqrt(mean_squared_error(test_y,result)))
    print("gbdt均方差: %f" % (mean_squared_error(test_y,result)))
    print("********")
    print("gbdt_r2得分：%f" % r2_score(test_y,result))
    print("gbdt用时：%f" % (time()-starttime))
#GBDTRR(train_X,train_y,test_X,test_y)

def all_clf(train_X,train_y,test_X,test_y):
    #linear(train_X,train_y,test_X,test_y)
    lass(train_X,train_y,test_X,test_y)
    ridge_cv(train_X,train_y,test_X,test_y)
    elas(train_X, train_y, test_X, test_y)
    randomf(train_X,train_y,test_X,test_y)
    GBDTRR(train_X,train_y,test_X,test_y)
    svr(train_X, train_y, test_X, test_y)
    ada(train_X, train_y, test_X, test_y)
    decision(train_X, train_y, test_X, test_y)
all_clf(train_X,train_y,test_X,test_y)



class Ensemble(object):
    #base_models=[clf1,clf2,clf3,clf4,....]  stacking using simple model
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        folds = list(KFold(n_splits=self.n_splits,
                           shuffle=True, random_state=2016).split(X, y))
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], self.n_splits))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                print ("Fit Model %d fold %d" % (i, j))
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
                #feature_importances=clf.feature_importance
                #feature_importances_df=pd.DateFrame(feature_importances,index=X.colums).sort_values()
                #li.append()
            S_test[:, i] = S_test_i.mean(axis=1)

        # results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='r2')
        # print("Stacker score: %.4f (%.4f)" % (results.mean(), results.std()))
        # exit()

        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_test)[:]
        return res





def stacking(X,y,test_X,test_y,test):
    # stan=StandardScaler()
    # stan.fit(X)
    # X=stan.transform(X)
    # test=stan.transform(test)
    enet = ElasticNetCV(fit_intercept=True, normalize=False)
    ridge = RidgeCV(fit_intercept=True, alphas=[0.1, 1.0, 10.0], normalize=False)
    lass = LassoCV(fit_intercept=True, normalize=False, alphas=[0.01, 0.1, 1.0, 10.0, 100])
    rf=RandomForestRegressor()
    ada=AdaBoostRegressor()
    dt=DecisionTreeRegressor()
    gbdt=GradientBoostingRegressor(
        loss='huber',  ##默认ls损失函数'ls'是指最小二乘回归lad'（最小绝对偏差）'huber'是两者的组合
        n_estimators=500,  ##默认100 回归树个数 弱学习器个数
        learning_rate=0.1,  ##默认0.1学习速率/步长0.0-1.0的超参数  每个树学习前一个树的残差的步长
        max_depth=3,  ## 默认值为3每个回归树的深度  控制树的大小 也可用叶节点的数量max leaf nodes控制
        subsample=1,  ##用于拟合个别基础学习器的样本分数 选择子样本<1.0导致方差的减少和偏差的增加
        min_samples_split=2,  ##生成子节点所需的最小样本数 如果是浮点数代表是百分比
        min_samples_leaf=1,  ##叶节点所需的最小样本数  如果是浮点数代表是百分比
        max_features=None,  ##在寻找最佳分割点要考虑的特征数量auto全选/sqrt开方/log2对数/None全选/int自定义几个/float百分比
        max_leaf_nodes=None,  ##叶节点的数量 None不限数量
        min_impurity_decrease=1e-7,  ##停止分裂叶子节点的阈值
        verbose=0,  ##打印输出 大于1打印每棵树的进度和性能
        warm_start=False,  ##True在前面基础上增量训练 False默认擦除重新训练 增加树
        random_state=0  ##随机种子-方便重现
    )
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.05)
    stack = Ensemble(n_splits=5,
                     stacker=LinearRegression(),
                     #1 :1500 rf,ada,dt,svr,gbdt
                     base_models=(svr,gbdt,enet,ridge,lass))
    result=stack.fit_predict(train_X, train_y,test)
    #test_y=stack.stacker.predict(test)[:]
    # print("stacking均方根：%f" % np.sqrt(mean_squared_error(test_y, result)))
    # print("stacking_r2得分：%f" % r2_score(test_y, result))
    # print("stacking用时：%f" % (time()-starttime))
    return result
test_predict=stacking(train_X,train_y,test_X,test_y,test)
test_predict=np.exp(test_predict)-1
submission=pd.DataFrame(test_predict)
print(test_predict)
submission.to_csv("C:/Users/yu/Desktop/compitition/submission2.csv")


# General Parameters（常规参数）
# 1.booster [default=gbtree]：选择基分类器，gbtree: tree-based models/gblinear: linear models
# 2.silent [default=0]:设置成1则没有运行信息输出，最好是设置为0.
# 3.nthread [default to maximum number of threads available if not set]：线程数
#
# Booster Parameters（模型参数）
# 1.eta [default=0.3]:shrinkage参数，用于更新叶子节点权重时，乘以该系数，避免步长过大。参数值越大，越可能无法收敛。把学习率 eta 设置的小一些，小学习率可以使得后面的学习更加仔细。
# 2.min_child_weight [default=1]:这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
# 3.max_depth [default=6]: 每颗树的最大深度，树高越深，越容易过拟合。
# 4.max_leaf_nodes:最大叶结点数，与max_depth作用有点重合。
# 5.gamma [default=0]：后剪枝时，用于控制是否后剪枝的参数。
# 6.max_delta_step [default=0]：这个参数在更新步骤中起作用，如果取0表示没有约束，如果取正值则使得更新步骤更加保守。可以防止做太大的更新步子，使更新更加平缓。
# 7.subsample [default=1]：样本随机采样，较低的值使得算法更加保守，防止过拟合，但是太小的值也会造成欠拟合。
# 8.colsample_bytree [default=1]：列采样，对每棵树的生成用的特征进行列采样.一般设置为： 0.5-1
# 9.lambda [default=1]：控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
# 10.alpha [default=0]:控制模型复杂程度的权重值的 L1 正则项参数，参数值越大，模型越不容易过拟合。
# 11.scale_pos_weight [default=1]：如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
#
# Learning Task Parameters（学习任务参数）
# 1.objective [default=reg:linear]：定义最小化损失函数类型，常用参数：
# binary:logistic –logistic regression for binary classification, returns predicted probability (not class)
# multi:softmax –multiclass classification using the softmax objective, returns predicted class (not probabilities)
# you also need to set an additional num_class (number of classes) parameter defining the number of unique classes
# multi:softprob –same as softmax, but returns predicted probability of each data point belonging to each class.
# 2.eval_metric [ default according to objective ]：
# The metric to be used for validation data.
# The default values are rmse for regression and error for classification.
# Typical values are:
# rmse – root mean square error
# mae – mean absolute error
# logloss – negative log-likelihood
# error – Binary classification error rate (0.5 threshold)
# merror – Multiclass classification error rate
# mlogloss – Multiclass logloss
# auc: Area under the curve
# 3.seed [default=0]：
# The random number seed. 随机种子，用于产生可复现的结果
# Can be used for generating reproducible results and also for parameter tuning.
#
# 注意: python sklearn style参数名会有所变化
# eta –> learning_rate
# lambda –> reg_lambda
# alpha –> reg_alpha








