'''
'''
# xgb矩阵赋值
xgb_val = xgb.DMatrix(val_X, label=val_y)
xgb_train = xgb.DMatrix(X, label=y)
xgb_test = xgb.DMatrix(test_x)
# xgboost模型 #####################

params = {
    'booster': 'gbtree',
    # 'objective': 'multi:softmax',  # 多分类的问题、
    # 'objective': 'multi:softprob',   # 多分类概率
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    # 'num_class': 9,  # 类别数，与 multisoftmax 并用
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 8,  # 构建树的深度，越大越容易过拟合
    'alpha': 0,  # L1正则化系数
    'lambda': 10,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.5,  # 生成树时进行的列采样
    'min_child_weight': 3,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.03,  # 如同学习率
    'seed': 1000,
    'nthread': -1,  # cpu 线程数
    'missing': 1,
    'scale_pos_weight': (np.sum(y == 0) / np.sum(y == 1))
# 用来处理正负样本不均衡的问题,通常取：sum(negative cases) / sum(positive cases)
    # 'eval_metric': 'auc'
}
'''
'''
X_train = X
y_train = y
X_test = val_X
y_test = val_y

# create dataset for lightgbm  
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict  
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'num_leaves': 5,
    'max_depth': 6,
    'min_data_in_leaf': 450,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'lambda_l1': 1,
    'lambda_l2': 0.001,  # 越小l2正则程度越高  
    'min_gain_to_split': 0.2,
    'verbose': 5,
    'is_unbalance': True
}
'''
# train
print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_eval,
                early_stopping_rounds=500)

print('Start predicting...')

preds = gbm.predict(test_x, num_iteration=gbm.best_iteration)  # 输出的是概率结果

# 导出结果
threshold = 0.5
for pred in preds:
    result = 1 if pred > threshold else 0

# xgb矩阵赋值
xgb_val = xgb.DMatrix(val_X, label=val_y)
xgb_train = xgb.DMatrix(X, label=y)
xgb_test = xgb.DMatrix(test_x)

# xgboost模型 #####################

params = {
    'booster': 'gbtree',
    # 'objective': 'multi:softmax',  # 多分类的问题、
    # 'objective': 'multi:softprob',   # 多分类概率
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    # 'num_class': 9,  # 类别数，与 multisoftmax 并用
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 8,  # 构建树的深度，越大越容易过拟合
    'alpha': 0,  # L1正则化系数
    'lambda': 10,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.5,  # 生成树时进行的列采样
    'min_child_weight': 3,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.03,  # 如同学习率
    'seed': 1000,
    'nthread': -1,  # cpu 线程数
    'missing': 1,
    'scale_pos_weight': (np.sum(y == 0) / np.sum(y == 1))
# 用来处理正负样本不均衡的问题,通常取：sum(negative cases) / sum(positive cases)
    # 'eval_metric': 'auc'
}
plst = list(params.items())
num_rounds = 2000  # 迭代次数
watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]

# 交叉验证
result = xgb.cv(plst, xgb_train, num_boost_round=200, nfold=4, early_stopping_rounds=200, verbose_eval=True,
                folds=StratifiedKFold(n_splits=4).split(X, y))

# 训练模型并保存
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=200)
model.save_model('../data/model/xgb.model')  # 用于存储训练出的模型

preds = model.predict(xgb_test)

# 导出结果
threshold = 0.5
for pred in preds:
    result = 1 if pred > threshold else 0

********


# 导出特征重要性
importance = gbm.feature_importance()
names = gbm.feature_name()
with open('./feature_importance.txt', 'w+') as file:
    for index, im in enumerate(importance):
        string = names[index] + ', ' + str(im) + '\n'
        file.write(string)
    # create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 9,
    'metric': 'multi_error',
    'num_leaves': 300,
    'min_data_in_leaf': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.4,
    'lambda_l2': 0.5,
    'min_gain_to_split': 0.2,
    'verbose': 5,
    'is_unbalance': True
}

# train
print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_eval,
                early_stopping_rounds=500)

print('Start predicting...')

preds = gbm.predict(test_x, num_iteration=gbm.best_iteration)  # 输出的是概率结果

# 导出结果
for pred in preds:
    result = prediction = int(np.argmax(pred))

# 导出特征重要性
importance = gbm.feature_importance()
names = gbm.feature_name()
with open('./feature_importance.txt', 'w+') as file:
    for index, im in enumerate(importance):
        string = names[index] + ', ' + str(im) + '\n'
        file.write(string)
        
        
         param_test = {  
        'max_depth': range(5,15,2),  
        'num_leaves': range(10,40,5),  
    }  
    estimator = LGBMRegressor(  
        num_leaves = 50, # cv调节50是最优值  
        max_depth = 13,  
        learning_rate =0.1,   
        n_estimators = 1000,   
        objective = 'regression',   
        min_child_weight = 1,   
        subsample = 0.8,  
        colsample_bytree=0.8,  
        nthread = 7,  
    )  
    gsearch = GridSearchCV( estimator , param_grid = param_test, scoring='roc_auc', cv=5 )  
    gsearch.fit( values, labels )  
    gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_  
    print_best_score(gsearch,param_test)  
  
'''