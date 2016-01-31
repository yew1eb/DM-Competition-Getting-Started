## importing libraries
import numpy as np
import pandas as pd

import xgboost

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score


## function for evaluation metric
def eval_metric(a, b, metric='auc'):
    if metric == 'auc':
        return roc_auc_score(a, b)


## function for logistic regression
def Logistic_Regression(train, target, test, lr, cv=5, seed=123, metric='auc'):
    """Performs k-fold logistic regression.
    Returns train and test dataframes with predictions
    """

    # imputing missing values
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    # preparing data
    X_train = pd.DataFrame.copy(train)
    X_test = pd.DataFrame.copy(test)
    target = np.array(target)

    # stratified k-fold
    kfolds = StratifiedKFold(target, n_folds=cv, random_state=seed)

    # cross-validation
    print('Logistic Regression')
        
    k = 1
    cv_score = []

    for build_index, val_index in kfolds:
        X_build, X_val, y_build, y_val = X_train.ix[build_index], X_train.ix[val_index], target[build_index], target[val_index]
        lr.fit(np.array(X_build), np.array(y_build))
                    
        X_val['pred_lr'] = lr.predict_proba(X_val)[:,1]
        X_test['pred_lr'] = lr.predict_proba(np.array(test))[:,1]

        if k == 1:
            train_lr = X_val[:]
            test_lr = X_test[:]

        if k > 1:
            train_lr = pd.concat([train_lr, X_val])
            test_lr['pred_lr'] = ((k-1) * test_lr['pred_lr'] + X_test['pred_lr']) / k

        cv_score.append(eval_metric(y_val, X_val['pred_lr'], metric=metric))

        print('Completed: %d%%' % (round(100*k/cv)))
        k += 1


    score = eval_metric(target, train_lr['pred_lr'], metric=metric)
    sd = np.std(cv_score)
            
    print("LogisticRegression %d-fold Cross Validation %s: %f + %f" % (cv, metric, score, sd))

    return train_lr, test_lr


## function for random forest
def Random_Forest(train, target, test, rf, cv=5, seed=123, metric='auc'):
    """Performs k-fold random forest.
    Returns train and test dataframes with predictions
    """

    # imputing missing values
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)

    # preparing data
    X_train = pd.DataFrame.copy(train)
    X_test = pd.DataFrame.copy(test)
    target = np.array(target)

    # stratified k-fold
    kfolds = StratifiedKFold(target, n_folds=cv, random_state=seed)

    # cross-validation
    print('Random Forest')
        
    k = 1
    cv_score = []

    for build_index, val_index in kfolds:
        X_build, X_val, y_build, y_val = X_train.ix[build_index], X_train.ix[val_index], target[build_index], target[val_index]
        rf.fit(np.array(X_build), np.array(y_build))
                    
        X_val['pred_rf'] = rf.predict_proba(X_val)[:,1]
        X_test['pred_rf'] = rf.predict_proba(np.array(test))[:,1]

        if k == 1:
            train_rf = X_val[:]
            test_rf = X_test[:]

        if k > 1:
            train_rf = pd.concat([train_rf, X_val])
            test_rf['pred_rf'] = ((k-1) * test_rf['pred_rf'] + X_test['pred_rf']) / k

        cv_score.append(eval_metric(y_val, X_val['pred_rf'], metric=metric))

        print('Completed: %d%%' % (round(100*k/cv)))
        k += 1


    score = eval_metric(target, train_rf['pred_rf'], metric=metric)
    sd = np.std(cv_score)
            
    print("RandomForest %d-fold Cross Validation %s: %f + %f" % (cv, metric, score, sd))

    return train_rf, test_rf


## function for extra trees
def Extra_Trees(train, target, test, et, cv=5, seed=123, metric='auc'):
    """Performs k-fold extra trees.
    Returns train and test dataframes with predictions
    """

    # imputing missing values
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)

    # preparing data
    X_train = pd.DataFrame.copy(train)
    X_test = pd.DataFrame.copy(test)
    target = np.array(target)

    # stratified k-fold
    kfolds = StratifiedKFold(target, n_folds=cv, random_state=seed)

    # cross-validation
    print('Extra Trees')
        
    k = 1
    cv_score = []

    for build_index, val_index in kfolds:
        X_build, X_val, y_build, y_val = X_train.ix[build_index], X_train.ix[val_index], target[build_index], target[val_index]
        et.fit(np.array(X_build), np.array(y_build))
                    
        X_val['pred_et'] = et.predict_proba(X_val)[:,1]
        X_test['pred_et'] = et.predict_proba(np.array(test))[:,1]

        if k == 1:
            train_et = X_val[:]
            test_et = X_test[:]

        if k > 1:
            train_et = pd.concat([train_et, X_val])
            test_et['pred_et'] = ((k-1) * test_et['pred_et'] + X_test['pred_et']) / k

        cv_score.append(eval_metric(y_val, X_val['pred_et'], metric=metric))

        print('Completed: %d%%' % (round(100*k/cv)))
        k += 1


    score = eval_metric(target, train_et['pred_et'], metric=metric)
    sd = np.std(cv_score)
            
    print("ExtraTrees %d-fold Cross Validation %s: %f + %f" % (cv, metric, score, sd))

    return train_et, test_et


## function for xgboost
def XGBoost(train, target, test, params, missing=-1, cv=5, seed=123, metric="auc"):
    """Performs k-fold xgboost.
    Returns train and test dataframes with predictions
    """

    # imputing missing values
    train.fillna(missing, inplace=True)
    test.fillna(missing, inplace=True)

    # preparing data
    X_train = pd.DataFrame.copy(train)
    X_test = pd.DataFrame.copy(test)
    target = np.array(target)

    xgtest = xgboost.DMatrix(X_test)

    # stratified k-fold
    kfolds = StratifiedKFold(np.zeros(len(target)), n_folds=cv, random_state=seed)

    # cross-validation
    print("XGBoost")

    k = 1
    cv_score = []

    for build_index, val_index in kfolds:
        X_build, X_val, y_build, y_val = X_train.ix[build_index], X_train.ix[val_index], target[build_index], target[val_index]
        xgbuild = xgboost.DMatrix(np.array(X_build), np.array(y_build))
        xgval = xgboost.DMatrix(np.array(X_val), np.array(y_val))

        evallist  = [(xgval,"eval"), (xgbuild,"train")]

        xgb = xgboost.train(params, xgbuild, params["nrounds"], evallist)

        X_val["pred_xgb"] = xgb.predict(xgval)
        X_test["pred_xgb"] = xgb.predict(xgtest)

        if k == 1:
            train_xgb = X_val[:]
            test_xgb = X_test[:]

        if k > 1:
            train_xgb = pd.concat([train_xgb, X_val])
            test_xgb["pred_xgb"] = ((k-1) * test_xgb["pred_xgb"] + X_test["pred_xgb"]) / k

        cv_score.append(eval_metric(y_val, X_val["pred_xgb"], metric=metric))

        print("Completed: %d%%" % (round(100*k/cv)))
        k += 1

    score = eval_metric(target, train_xgb["pred_xgb"], metric=metric)
    sd = np.std(cv_score)
            
    print("XGBoost %d-fold Cross Validation %s: %f + %f" % (cv, metric, score, sd))

    return train_xgb, test_xgb


