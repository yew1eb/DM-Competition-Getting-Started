#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
@filename: xgb_gridsearch.py
@author: yew1eb
@site: http://blog.yew1eb.net
@contact: yew1eb@gmail.com
@time: 2016/01/01 下午 5:25
'''

# Note: Kaggle only runs Python 3, not Python 2

#code skeleton for feature engineering and hyperparameter search for DC-loan-rp
#surely, I don't reveal the best parameters here and it doesn't pass KS test :-)
#do you know you can 'pip install DC-loan-rp'?
import numpy as np
import pandas as pd
import xgboost as xgb
import sys

#import evaluation
exec(open("../data/evaluation.py").read())

from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV

import math

""" Implemented Scikit- style grid search to find optimal XGBoost params"""
""" Use this module to identify optimal hyperparameters for XGBoost"""

#have some feature engineering work for better rank
def add_features(df):
    #significance of flight distance
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    return df

print("Load the training/test data using pandas")
train = pd.read_csv("../data/training.csv")
test = pd.read_csv("../data/test.csv")

print("Adding features to both training and testing")
train = add_features(train)
test = add_features(test)

print("Loading check agreement for KS test evaluation")
check_agreement = pd.read_csv('../data/check_agreement.csv')
check_correlation = pd.read_csv('../data/check_correlation.csv')
check_agreement = add_features(check_agreement)
check_correlation = add_features(check_correlation)
train_eval = train[train['min_ANNmuon'] > 0.4]

print("Eliminate SPDhits, which makes the agreement check fail")
filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal','SPDhits','IP']
features = list(f for f in train.columns if f not in filter_out)
print(("features:",features))

print("Train a XGBoost model")

xgb_model = xgb.XGBClassifier()

parameters = {'nthread':[4], #when use hyperthread, DC-loan-rp may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.3], #so called `eta` value
              'max_depth': [5,6],
              'min_child_weight': [3],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [50,100], #number of trees
              'seed': [1337]}

clf = GridSearchCV(xgb_model, parameters, n_jobs=4,
                   cv=StratifiedKFold(train['signal'], n_folds=5, shuffle=True),
                   verbose=2, refit=True,scoring='roc_auc')

clf.fit(train[features], train["signal"])

best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print(('Raw AUC score:', score))
for param_name in sorted(best_parameters.keys()):
    print(("%s: %r" % (param_name, best_parameters[param_name])))

test_agreement_ = True
if test_agreement_:
    agreement_probs= (clf.predict_proba(check_agreement[features])[:,1])

    ks = compute_ks(
            agreement_probs[check_agreement['signal'].values == 0],
            agreement_probs[check_agreement['signal'].values == 1],
            check_agreement[check_agreement['signal'] == 0]['weight'].values,
            check_agreement[check_agreement['signal'] == 1]['weight'].values)
    print(('KS metric', ks, ks < 0.09))

    correlation_probs = clf.predict_proba(check_correlation[features])[:,1]
    print ('Checking correlation...')
    cvm = compute_cvm(correlation_probs, check_correlation['mass'])
    print(('CvM metric', cvm, cvm < 0.002))

    train_eval_probs = clf.predict_proba(train_eval[features])[:,1]
    print ('Calculating AUC...')
    AUC = roc_auc_truncated(train_eval['signal'], train_eval_probs)
    print(('AUC', AUC))
'''
Sample output can be:

    ('Raw AUC score:', 0.8746021642266073)
    colsample_bytree: 0.7
    learning_rate: 0.3
    max_depth: 5
    min_child_weight: 3
    n_estimators: 100
    nthread: 4
    objective: 'binary:logistic'
    seed: 1337
    silent: 1
    subsample: 0.7
    ('KS metric', 0.12431348282537757, False)
    Checking correlation...
    ('CvM metric', 0.0010158549152792396, True)
    Calculating AUC...
    ('AUC', 0.99349523223341396)
'''
test_probs = clf.predict_proba(test[features])[:,1]
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("xgboost_best_parameter_submission.csv", index=False)