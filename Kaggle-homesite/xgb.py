#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
@filename: xgb.py
@author: yew1eb
@site: http://blog.yew1eb.net
@contact: yew1eb@gmail.com
@time: 2016/01/01 下午 4:23
'''

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

random_seed = 260681

path = 'd:/dataset/homesite/'
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')

y = train.QuoteConversion_Flag.values

train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
test_uid = test.QuoteNumber
test = test.drop('QuoteNumber', axis=1)

# Lets play with some dates
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train = train.drop('Original_Quote_Date', axis=1)

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)

train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['weekday'] = train['Date'].dt.dayofweek

test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
test['weekday'] = test['Date'].dt.dayofweek

train = train.drop('Date', axis=1)
test = test.drop('Date', axis=1)

train = train.fillna(-1)
test = test.fillna(-1)

for f in train.columns:
    if train[f].dtype == 'object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))


clf = xgb.XGBClassifier(n_estimators=25,
                        nthread=-1,
                        max_depth=8,
                        learing_rate=0.025,
                        silent=True,
                        subsample=0.8,
                        colsample_bytree=0.8)

xgb_model = clf.fit(train, y, eval_metric="auc")

preds = clf.predict_proba(test)[:,1]
result = pd.DataFrame({"QuoteNumber":test_uid,
                       "QuoteConversion_Flag":preds})
result.to_csv(path+'xgb_benchmark.csv', index=False)