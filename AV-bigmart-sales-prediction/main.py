import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

#train['Item_Outlet_Sales'] = np.log(train['Item_Outlet_Sales'])

train= train.fillna(0)
test = test.fillna(0)

cat_var = ['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type']

def dummy(col,data):
    for a in col:
        dummy_train = pd.get_dummies(data[a],prefix=a)
        data = pd.concat([data,dummy_train],axis=1)
        data.drop(a,axis=1,inplace=True)
    return data

train = dummy(cat_var,train)
test = dummy(cat_var,test)

train.to_csv('data/train-cleaned.csv',index=False)
test.to_csv('data/test-cleaned.csv',index=False)

train = pd.read_csv('data/train-cleaned.csv')
test = pd.read_csv('data/test-cleaned.csv')

train['Outlet_Identifier_2'] = pd.factorize(train['Outlet_Identifier'])[0]

test['Outlet_Identifier_2'] = pd.factorize(test['Outlet_Identifier'])[0]

cat = ['Item_Identifier']

for a in cat:
    lb = preprocessing.LabelEncoder()
    full_var_data = pd.concat((train[a],test[a]),axis=0).astype('str')
    lb.fit( full_var_data )
    train[a] = lb.transform(train[a])
    test[a] = lb.transform(test[a])

def evaluation(x,y):
    sum =0
    for a,b in zip(x,y):
        sum = sum + np.square(a-b)
    return np.sqrt(sum/len(x))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error


train_y = np.array(train["Item_Outlet_Sales"])

## Creating the IDVs from the train and test dataframe ##
train_X = train.copy()
test_X = test.copy()

train_X = np.array( train_X.drop(['Item_Outlet_Sales','Outlet_Identifier'],axis=1) )

kfolds = KFold(train_X.shape[0], n_folds=6)
for dev_index, val_index in kfolds:
    dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]

    reg = RandomForestRegressor(n_estimators=1000, max_depth=6, min_samples_leaf=1, max_features="auto", n_jobs=-1, random_state=88888)
    reg.fit(dev_X, dev_y)
    pred_val_y = reg.predict(val_X)
    '''dtrain = xgb.DMatrix(dev_X,label = dev_y)
    dtest = xgb.DMatrix(val_X)
    bst = xgb.train( plst,dtrain, num_rounds)
    ypred = bst.predict(dtest,ntree_limit=bst.best_iteration)
    pred_val_y = (pred_val_y + ypred) / 2'''
    print(np.sqrt(mean_squared_error(val_y, pred_val_y)))

print("Building RF1")
reg = RandomForestRegressor(n_estimators=1000, max_depth=6, min_samples_leaf=1, max_features="auto", n_jobs=-1, random_state=88888)
reg.fit(train_X, train_y)
pred = reg.predict(test_X.drop(['Outlet_Identifier'],axis=1))

test['Item_Identifier'] = lb.inverse_transform(test['Item_Identifier'])

test['Item_Outlet_Sales'] = pred

#test['Item_Outlet_Sales'] =  np.exp(test['Item_Outlet_Sales'])

test.to_csv('submission/sumbit-8.csv',columns=['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'],index=False)