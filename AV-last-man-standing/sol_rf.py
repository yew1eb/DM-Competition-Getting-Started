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


print("Filled Missing Values")
train_x = train_x.fillna(value = -999)
test_x = test_x.fillna(value = -999)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score = True, max_features = "auto",random_state=10, min_samples_split=2, min_samples_leaf=2)
rf.fit(train_x, train_y)
print(('Training accuracy:', rf.oob_score_))

print ("Starting to predict on the dataset")
pred = rf.predict(test_x)

print ("Prediction Completed")
test['Crop_Damage'] = pred
test.to_csv('submission/rf.csv', columns=['ID','Crop_Damage'],index=False)
