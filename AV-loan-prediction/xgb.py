#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb
import time
import math
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

def load_data():
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')
    train_y = (train.Loan_Status == 'Y').astype('int')
    train_x = train.drop(['Loan_ID','Loan_Status'], axis=1)
    test_uid = test.Loan_ID
    test_x = test.drop(['Loan_ID'], axis=1)

    cat_var = ['Gender','Married','Dependents','Education', 'Self_Employed', 'Property_Area']
    num_var = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']
    for var in num_var:
        train_x[var] = train_x[var].fillna(value = train_x[var].mean())
        test_x[var]  = test_x[var].fillna(value = test_x[var].mean())
    train_x['Credibility'] = train_x['ApplicantIncome'] / train_x['LoanAmount']
    test_x['Credibility'] = test_x['ApplicantIncome'] / test_x['LoanAmount']
    train_x = train_x.fillna(value = -999)
    test_x  = test_x.fillna(value = -999)

    for var in cat_var:
        lb = preprocessing.LabelEncoder()
        full_data = pd.concat((train_x[var],test_x[var]),axis=0).astype('str')
        lb.fit( full_data )
        train_x[var] = lb.transform(train_x[var].astype('str'))
        test_x[var] = lb.transform(test_x[var].astype('str'))

    return train_x, train_y, test_x, test_uid


def using_xgb(train_x, train_y, test_x, test_uid):
    scale_val = (train_y.sum() / train_y.shape[0])
    X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, train_size=0.75, random_state=0)
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_val   = xgb.DMatrix(X_val,   label=y_val)
    xgb_test  = xgb.DMatrix(test_x)

    # 设置xgboost分类器参数
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'early_stopping_rounds': 200,
        'gamma':0,
        'lambda': 1000,
        'min_child_weight': 5,
        'scale_pos_weight': scale_val,
        'subsample': 0.7,
        'max_depth':6,
        'eta': 0.01,
        #'colsample_bytree': 0.7,
        'nthread': 2
    }
    watchlist = [(xgb_val, 'val'), (xgb_train, 'train')]
    num_round = 10000
    bst = xgb.train(params, xgb_train, num_boost_round=num_round, evals=watchlist)
    scores = bst.predict(xgb_test, ntree_limit=bst.best_ntree_limit)
    pred = np.where(scores > 0.5, 'Y','N')


    print((pd.value_counts(pred)))


    return 0
    result = pd.DataFrame({"Loan_ID":test_uid, "Loan_Status":pred}, columns=['Loan_ID','Loan_Status'])
    result.to_csv('result/xgb_'+str(time.time())[-4:]+'.csv', index=False)

def main():
    train_x, train_y, test_x, test_uid = load_data()
    print("load_data() end!")
    using_xgb(train_x, train_y, test_x, test_uid)


if __name__ == '__main__':
    main()