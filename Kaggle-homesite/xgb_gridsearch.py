#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
@filename: xgb_gridsearch.py
@author: yew1eb
@site: http://blog.yew1eb.net
@contact: yew1eb@gmail.com
@time: 2016/01/01 下午 5:25
'''

import numpy as np
import pandas as pd
import xgboost as xgb
import sys

from sklearn.corss_validation import
from sklearn.grid_search import GridSearchCV
import math
'''
grid search + cross validation
http://scikit-learn.org/0.12/auto_examples/grid_search_digits.html#example-grid-search-digits-py
'''
""" Implemented Scikit- style grid search to find optimal XGBoost params"""
""" Use this module to identify optimal hyperparameters for XGBoost"""

#特征工程：添加特征
def  add_features(df):
    #significance of flight distance
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    #Stepan Obraztsov's magic features
    df['NEW_FD_SUMP']=df['FlightDistance']/(df['p0_p']+df['p1_p']+df['p2_p'])
    df['NEW5_lt']=df['LifeTime']*(df['p0_IP']+df['p1_IP']+df['p2_IP'])/3
    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
    #some more magic features
    df['p0p2_ip_ratio']=df['IP']/df['IP_p0p2']
    df['p1p2_ip_ratio']=df['IP']/df['IP_p1p2']
    df['DCA_MAX'] = df.loc[:, ['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)
    df['iso_bdt_min'] = df.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)
    df['iso_min'] = df.loc[:, ['isolationa', 'isolationb', 'isolationc','isolationd', 'isolatione', 'isolationf']].min(axis=1)

    return df

print("Load the training/test data using pandas")
path = './'
train = pd.read_csv(path+"training.csv")
test = pd.read_csv(path+"test.csv")

print("Adding features to both training and testing")
train = add_features(train)
test = add_features(test)

