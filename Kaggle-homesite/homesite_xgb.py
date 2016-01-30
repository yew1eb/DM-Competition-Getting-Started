import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split


random_seed = 23145

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")


train = train.drop('QuoteNumber', axis=1)
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

train = train.fillna(-999)
test = test.fillna(-999)

features = list(train.columns[1:])  #la colonne 0 est le quote_conversionflag
print(features)


for f in train.columns:
    if train[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))



X_train, X_val, y_train, y_val = train_test_split(train[features], train['QuoteConversion_Flag'], train_size=0.75, random_state=random_seed)
xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_val   = xgb.DMatrix(X_val,   label=y_val)
xgb_test  = xgb.DMatrix(test[features])

params  =  {
    #1- General Parameters
    'booster' : "gbtree", #booster [default=gbtree]
    'silent': 0 , #silent [default=0]
    #'nthread' : -1 , #nthread [default to maximum number of threads available if not set]

    #2A-Parameters for Tree Booster
    'eta'  :0.02, # eta [default=0.3] range: [0,1]
    #'gamma':0 ,#gamma [default=0] range: [0,鈭瀅
    'max_depth'           :8, #max_depth [default=6] range: [1,鈭瀅
    #'min_child_weight':1,  #default=1]range: [0,鈭瀅
    #'max_delta_step':0, #max_delta_step [default=0] range: [0,鈭瀅
    'subsample'           :0.83, #subsample [default=1]range: (0,1]
    'colsample_bytree'    :0.77, #colsample_bytree [default=1]range: (0,1]
    #'lambda': 1,  #lambda [default=1]
    #'alpha':0.0001, #alpha [default=0]


    #2B- Parameters for Linear Booster
    #'lambda': 0,  #lambda [default=0]
    #'alpha':0, #alpha [default=0]
    #'lambda_bias':0, #default 0

    #3- earning Task Parameters
    'objective': 'binary:logistic',  #objective [ default=reg:linear ]
    #'base_score'=0.5,        #base_score [ default=0.5 ]
    'eval_metric' : 'auc', #eval_metric [ default according to objective ]
    'seed':random_seed #seed [ default=0 ]

    }

watchlist = [(xgb_val, 'val'), (xgb_train, 'train')]

num_round = 1800

bst = xgb.train(params, xgb_train, num_boost_round=num_round, evals=watchlist)


#prediction
preds= bst.predict(xgb_test)

#print to CSV
sample = pd.read_csv('./data/sample_submission.csv')
sample.QuoteConversion_Flag = preds
sample.to_csv('result_xgb_python.csv', index=False)