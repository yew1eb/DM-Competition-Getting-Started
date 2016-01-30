#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import time

'''
ID                          UniqueID
Estimated_Insects_Count     Estimated insects count per square meter
Crop_Type                   Category of Crop(0,1)
Soil_Type                   Category of Soil (0,1)
Pesticide_Use_Category      Type of pesticides uses (1- Never, 2-Previously Used, 3-Currently Using)
Number_Doses_Week           Number of doses per week
Number_Weeks_Used           Number of weeks used
Number_Weeks_Quit           Number of weeks quit
Season                      Season Category (1,2,3)
Crop_Damage                 Crop Damage Category (0=alive, 1=Damage due to other causes, 2=Damage due to Pesticides)
'''
train = pd.read_csv("data/train.csv")
train_y = train['Crop_Damage']
train_x = train.drop(['ID','Crop_Damage'], axis=1)
test  = pd.read_csv("data/test.csv")
test_uid = test['ID']
test_x  = test.drop(['ID'], axis=1)
print(pd.isnull(train).sum())
print(pd.isnull(test).sum())

print("Filled Missing Values")
train_x = train_x.fillna(value = -999)
test_x = test_x.fillna(value = -999)
print("Sample Labels")
print(pd.value_counts(train_y).to_dict())

from sklearn.cross_validation import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, train_size=0.75, random_state=0)
xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_val   = xgb.DMatrix(X_val,   label=y_val)
xgb_test  = xgb.DMatrix(test_x)

params = {
    'booster':'gbtree',
    'objective': 'multi:softmax',
    'num_class':3, # 类数，与 multisoftmax 并用
    'eval_metric': 'merror',
    'early_stopping_rounds': 120,
    'gamma':0.05, # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守
    #'lambda': 1000,# L2 正则项权重
    'min_child_weight': 10, # 节点的最少特征数
    'subsample': 0.5, # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    'max_depth':4, # 构建树的深度
    'eta': 0.3,
    'colsample_bytree': 0.5, # 构建树树时的采样比率 (0:1]
}

watchlist = [(xgb_val, 'val'), (xgb_train, 'train')]
num_round = 200
bst = xgb.train(params, xgb_train, num_boost_round=num_round, evals=watchlist)
#bst.save_model('./model/xgb.model') # 用于存储训练出的模型

pred = bst.predict(xgb_test, ntree_limit=bst.best_ntree_limit).astype('int')

print(pd.value_counts(pred))

result = pd.DataFrame({"ID":test_uid, "Crop_Damage":pred}, columns=['ID','Crop_Damage'])
result.to_csv('submission/xgb_MultiSoftmax.csv', index=False)