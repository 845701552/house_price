import pandas as pd
from scipy.stats import skew,probplot
from scipy.special import boxcox1p
from catboost import CatBoostRegressor,Pool
import numpy as np
import seaborn as sns
from time import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR,libsvm
from sklearn.linear_model import LassoCV,RidgeCV,LinearRegression,ElasticNet,ElasticNetCV,Lasso
from sklearn.model_selection import GridSearchCV
import pickle
import sklearn.tree.tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import lightgbm as lgt
import warnings

train=pd.read_csv("E:/pycharm project/compitition/train_preprocessing_fillna_addfeat_goodcorr.csv")
test=pd.read_csv("E:/pycharm project/compitition/test_preprocessing_fillna_addfeat_goodcorr.csv")
y=train["SalePrice"]
X=train.drop(["SalePrice"],axis=1)
train_X,test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 42)

'''RobustScaler() 去异常点'''
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9,random_state=1))
# KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)


model_lgt= lgt.LGBMRegressor(objective='regression',num_leaves=31,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11
                           )
def grid_model(train_X,train_y,test_X,test_y,test):
    params={"depth":[4,6,8],
            "learning_rate":[0.03,0.1,0.15],
            "iterations":[300,500,800,1000],
            "l2_leaf_reg":[1,4,9],
            }
    cat = CatBoostRegressor()
    grid=GridSearchCV(cat,param_grid=params,scoring="r2",cv=5,n_jobs=-1)
    grid.fit(X,y)
    print(grid.best_params_)
    print(grid.best_score_)
#grid_model(X,y,test_X,test_y,test)


def cat_model(train_X,train_y,test_X,test_y,test):
    starttime=time()
    train_pool = Pool(train_X,train_y,)
    test_pool = Pool(test_X)
    cat=CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            l2_leaf_reg=1,
            max_depth=3,
            loss_function='RMSE',
            depth=None,
            random_seed=2018,
            use_best_model=None,
            verbose=1,
            logging_level=None,
            metric_period=None,
            objective=None,
            eta=None,
            max_bin=55,
    )
    cat.fit(train_pool)
    result=cat.predict(test_pool)
    print("cat_rmes：%f" % np.sqrt(mean_squared_error(test_y, result)))
    print("\033[1;34m")
    print("cat_r2得分：%f" % r2_score(test_y, result))
    print("\033[0m")
    print("cat：%f" % (time() - starttime))
#cat_model(train_X,train_y,test_X,test_y,test)

# def xgb_model():
#     import pandas as pd
#     import numpy as np
#     train=pd.read_csv("E:/pycharm project/compitition/train_preprocessing_fillna_addfeat_goodcorr.csv")
#     test=pd.read_csv("E:/pycharm project/compitition/test_preprocessing_fillna_addfeat_goodcorr.csv")
#     y=train["SalePrice"]
#     X=train.drop(["SalePrice"],axis=1)
#     import xgboost as xgb
#     model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
#                                  learning_rate=0.05, max_depth=3,
#                                  min_child_weight=1.7817, n_estimators=2200,
#                                  reg_alpha=0.4640, reg_lambda=0.8571,
#                                  subsample=0.5213, silent=1,
#                                   nthread = -1)
#     model_xgb.fit(X,y)
#     final_predict=model_xgb.predict(test)
#     final_predict = np.exp(final_predict) - 1
#     submission = pd.DataFrame(final_predict)
#     submission=submission*0.2
#     submission.to_csv("C:/Users/yu/Desktop/xgboost_fillna_addfeat_goodcorr.csv")
#     print("ok")
def lgt_raw_using(train_X,train_y,test_X,test_y,test):
    starttime = time()
    lgb_train = lgt.Dataset(train_X,train_y)
    lgb_eval = lgt.Dataset(test_X,test_y, reference=lgb_train)
    params = { "objective":"regression",
               'max_depth': 6,
               'num_leaves': 64,
               #"num_leaves":5,   #num_leaves = 2^(max_depth)
               "learning_rate":0.05,
               "n_estimators":720,
               "max_bin":55,
               "bagging_fraction": 0.8,
               "bagging_freq": 5,
               "feature_fraction": 0.2319,
               "feature_fraction_seed":9,
               "bagging_seed":9,
               "min_data_in_leaf":6,
               "min_sum_hessian_in_leaf":11,
    }
    model=lgt.train(params=params,train_set=lgb_train,early_stopping_rounds=10,
                              valid_sets=lgb_eval,)

    #model.save_model("E:/pycharm project/compitition/lgt.txt")
    result = model.predict(test,num_iteration=model.best_iteration) #num_iteration=clf.best_iteration
    # print("lgt：%f" % np.sqrt(mean_squared_error(test_y, result)))
    # print("\033[1;34m")
    # print("lgt_r2得分：%f" % r2_score(test_y, result))
    # print("\033[0m")
    # print("lgt：%f" % (time() - starttime))
    return model,result
lgt_save,lgt_result=lgt_raw_using(X,y,test_X,test_y,test)
lgt_result=np.exp(lgt_result)-1


'''单个模型交叉验证'''
def rmsle_cv(model,X,y):
        kf = KFold(n_splits=5, shuffle=True, random_state=42).get_n_splits(X.values)
        rmse= np.sqrt((-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))).mean()
        f1_score = np.mean(cross_val_score(model, X, y,scoring="r2", cv = kf))
        print("rmse:"+str(rmse))
        print("f1_score:" + str(f1_score))
#rmsle_cv(model_lgt,X,y)
#rmsle_cv(GBoost,X,y)

'''特征重要性 gini or split'''
def ret_feat_impt(gbm):
    gain = gbm.feature_importance("split").reshape(-1, 1) / sum(gbm.feature_importance("gain"))
    col = np.array(gbm.feature_name()).reshape(-1, 1)
    feature_important=sorted(np.column_stack((col, gain)),key=lambda x: x[1],reverse=True)
    for i in feature_important:
        print(i)

#ret_feat_impt(lgt_save)
def light_ffm_data(gbm,train_X,test_X, train_y, test_y,test):
    y_train_ = train_y.values
    y_valid_ = test_y.values

    X_train_ = train_X
    X_valid_ = test_X
    X_test_ = test.values
    #print(X_train_.shape)#(1022, 270)
    train_leaves = gbm.predict(X_train_, num_iteration=gbm.best_iteration, pred_leaf=True)
    valid_leaves = gbm.predict(X_valid_, num_iteration=gbm.best_iteration, pred_leaf=True)
    test_leaves = gbm.predict(X_test_, num_iteration=gbm.best_iteration, pred_leaf=True)
    print(train_leaves[:2])
    #print(train_leaves.shape)#(1022, 199)199颗树
    # tree_info = dump['tree_info']
    # tree_counts = len(tree_info)
    # for i in range(tree_counts):
    #     train_leaves[:, i] = train_leaves[:, i] + tree_info[i]['num_leaves'] * i + 1
    #     valid_leaves[:, i] = valid_leaves[:, i] + tree_info[i]['num_leaves'] * i + 1
    #     test_leaves[:, i] = test_leaves[:, i] + tree_info[i]['num_leaves'] * i + 1
    # #                     print(train_leaves[:, i])
    # #                     print(tree_info[i]['num_leaves'])
    #
    # for idx in range(len(y_train_)):
    #     out_train.write((str(y_train_[idx]) + '\t'))
    #     out_train.write('\t'.join(
    #         ['{}:{}'.format(ii, val) for ii, val in enumerate(train_leaves[idx]) if float(val) != 0]) + '\n')
    #
    # for idx in range(len(y_valid_)):
    #     out_valid.write((str(y_valid_[idx]) + '\t'))
    #     out_valid.write('\t'.join(
    #         ['{}:{}'.format(ii, val) for ii, val in enumerate(valid_leaves[idx]) if float(val) != 0]) + '\n')
    #
    # for idx in range(len(X_test_)):
    #     out_test.write('\t'.join(
    #         ['{}:{}'.format(ii, val) for ii, val in enumerate(test_leaves[idx]) if float(val) != 0]) + '\n')
    #
    #
    #

#light_ffm_data(lgt_save,train_X,test_X, train_y, test_y,test)
'''

'''
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
#all_clf(train_X,train_y,test_X,test_y)



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
    starttime=time()
    enet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3,fit_intercept=True, normalize=False)
    ridge = RidgeCV(fit_intercept=True, alphas=[0.1, 1.0, 10.0], normalize=False)
    lass = Lasso(alpha =0.0005, random_state=1,fit_intercept=True, normalize=False)
    rf=RandomForestRegressor()
    ada=AdaBoostRegressor()
    dt=DecisionTreeRegressor()
    cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        l2_leaf_reg=1,
        max_depth=3,
        loss_function='RMSE',
        depth=None,
        random_seed=2018,
        use_best_model=None,
        verbose=1,
        logging_level=None,
        metric_period=None,
        objective=None,
        eta=None,
        max_bin=55,
    )

    gbdt= GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=5)
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.05)
    stack = Ensemble(n_splits=5,
                     stacker=LinearRegression(),
                     #1 :1500 rf,ada,dt,svr,gbdt
                     #2 :864  768 gbdt,enet,lass
                     #
                     base_models=(cat,gbdt,enet,lass))
    result=stack.fit_predict(X,y,test)
    # print("stacking均方根：%f" % np.sqrt(mean_squared_error(test_y, result)))
    # print("stacking_r2得分：%f" % r2_score(test_y, result))
    # print("stacking用时：%f" % (time()-starttime))
    return result

'''输出'''
stack_predict=stacking(X,y,test_X,test_y,test)
stack_predict=np.exp(stack_predict)-1
xgb_predict=pd.read_csv("C:/Users/yu/Desktop/xgboost_fillna_addfeat_goodcorr.csv")
final_predict=stack_predict*0.6+lgt_result*0.2+xgb_predict["saleprice"]
submission=pd.DataFrame(final_predict)
submission.to_csv("C:/Users/yu/Desktop/submission_goodcorr.csv")
print(submission.head())
print("ok")

