#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
import xgboost as xgb
import time
from sklearn.grid_search import GridSearchCV

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

train_x = train_x.fillna(value = -1)
test_x = test_x.fillna(value = -1)

clf = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=300,
    objective="multi:softmax",
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=1,
    colsample_bytree=1,
    scale_pos_weight=1,
    seed=0
)

param_grid = {
    'min_child_weight': [48, 49,50,51,52,53, 55, 80],
    #'subsample': [0.7, 0.8],
    #'colsample_bytree': [0.7, 0.8],
    #'n_estimators': [100, 250, 500, 1000],
    #'gamma':[0.05, 0.1, 0.5],
}

print("Starting GridSearchCV...")
clf = GridSearchCV(clf, param_grid)

clf.fit(train_x, train_y)

#trust your CV!
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
