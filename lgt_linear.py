import pandas as pd
import lightgbm as lgt
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
train=pd.read_csv("E:/pycharm project/compitition/train_preprocessing_fillna_addfeat3.csv")
test=pd.read_csv("E:/pycharm project/compitition/test_preprocessing_fillna_addfeat3.csv")
y=train["SalePrice"]
X=train.drop(["SalePrice"],axis=1)
train_X,test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 42)
def lgt_raw_using(X,y,test_X,test_y,test):

    lgb_train = lgt.Dataset(X,y)
    lgb_eval = lgt.Dataset(test_X,test_y, reference=lgb_train)
    params = { "objective":"regression",
               'max_depth': 6,
               'num_leaves': 64,
               #"num_leaves":5,   #num_leaves = 2^(max_depth)
               "learning_rate":0.05,
               "n_estimators":30,
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


    result = model.predict(X,num_iteration=model.best_iteration,pred_leaf=True) #num_iteration=clf.best_iteration
    result_test = model.predict(test, num_iteration=model.best_iteration, pred_leaf=True)
    # print("lgt：%f" % np.sqrt(mean_squared_error(test_y, result)))
    # print("\033[1;34m")
    # print("lgt_r2得分: %f" % r2_score(test_y, result))
    # print("\033[0m")
    # print("lgt：%f" % (time() - starttime))
    return model,result,result_test
lgt_save,lgt_result,result_test=lgt_raw_using(X,y,test_X,test_y,test)
lgt_result=pd.DataFrame(lgt_result,columns=list(range(30)))
test_result=pd.DataFrame(result_test,columns=list(range(30)))

train_lgt=pd.concat([X,lgt_result],axis=1)
test_lgt=pd.concat([test,test_result],axis=1)

Ela=ElasticNet(alpha=0.001, l1_ratio=1.1,random_state=1)
Ela.fit(train_lgt,y)

lgt_ela_result=Ela.predict(test_lgt)
lgt_ela_result=np.exp(lgt_ela_result)-1
lgt_ela_result=pd.DataFrame(lgt_ela_result)
# lgt_ela_result.to_csv("C:/Users/yu/Desktop/submission_lgt_ela1.csv")