'''
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
'''
