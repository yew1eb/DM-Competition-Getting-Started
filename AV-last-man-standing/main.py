# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
import xgboost as xgb

# In[2]:

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# In[3]:

train = train.fillna(-1)
test = test.fillna(-1)

# In[4]:

num_rounds = 200
parm = {'objective': "multi:softmax", 'booster': "gbtree", 'eval_metric': "merror", 'num_class': 3,
        'max_depth': 6, 'min_child_weight': 50,  'eta': 0.2, 'seed': 88888,
        #'subsample':0.7, 'colsample_bytree':0.75, 'gamma':0.05, 'scale_pos_weight':0.5,
}

plst = parm.items()

# In[5]:

train_y = np.array(train['Crop_Damage'])

## Creating the IDVs from the train and test dataframe ##
train_X = train.copy()
test_X = test.copy()

train_X = np.array(train_X.drop(['Crop_Damage', 'ID'], axis=1))

# In[6]:

a = np.array([])
kfolds = KFold(train_X.shape[0], n_folds=6)
for dev_index, val_index in kfolds:
    dev_X, val_X = train_X[dev_index, :], train_X[val_index, :]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    dtrain = xgb.DMatrix(dev_X, label=dev_y)
    dtest = xgb.DMatrix(val_X)
    bst = xgb.train(plst, dtrain, num_rounds)
    ypred_bst = bst.predict(dtest, ntree_limit=bst.best_iteration)
    score = metrics.confusion_matrix(val_y, ypred_bst)
    res = (score[0][0] + score[1][1] + score[2][2]) * 1.0 / sum(sum(score))
    a = np.append(a, [res])
    print("Accuracy = %.7f" % (res))

print("Overall Mean Accuracy = %.7f" % (np.mean(a)))

# In[15]:

print("Building XGB")
y = train['Crop_Damage'].values
dtrain = xgb.DMatrix(train.drop(['Crop_Damage', 'ID'], axis=1), label=y)
dtest = xgb.DMatrix(test.drop(['ID'], axis=1))
bst = xgb.train(plst, dtrain, num_rounds)


ypred_bst = bst.predict(dtest, ntree_limit=bst.best_iteration).astype('int')

test['Crop_Damage'] = ypred_bst
print("Overall Mean Accuracy = %.7f" % (np.mean(a)))
test.to_csv('submission/xgb_KFold=6_auc='+str(np.mean(a))+'.csv', columns=['ID', 'Crop_Damage'], index=False)

'''
lc: 0.8482298
lb: 0.849292070065
'''