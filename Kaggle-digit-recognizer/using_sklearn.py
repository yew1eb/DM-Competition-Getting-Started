#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Filename: handwriting.py
@Author: yew1eb
@Date: 2015/12/23 0023
"""

'''
使用sickit-learn中的分类算法预测
用户使用文档：http://scikit-learn.org/dev/user_guide.html
'''

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import  GaussianNB   #naive bayes 高斯分布的数据
from sklearn.naive_bayes import MultinomialNB #naive bayes 多项式分布的数据
from sklearn.linear_model import LinearRegression

def load_data():
    # strain.csv  3000条数据; train.csv 完整训练数据集
    train_data = np.loadtxt('d:/dataset/digits/train.csv', dtype=np.uint8,delimiter=',', skiprows=1)
    test_data = np.loadtxt('d:/dataset/digits/test.csv', dtype=np.uint8,delimiter=',', skiprows=1)
    label = train_data[:,:1]
    data  = np.where(train_data[:, 1:]!=0, 1, 0)# 数据归一化
    test  = np.where(test_data !=0, 1, 0)
    return data, label, test

def save2csv(labels, csv_name):
    np.savetxt('d:/dataset/digits/'+csv_name, np.c_[list(range(1,len(labels)+1)),labels],
               delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

def sklearn_logistic(train_data, train_label, test_data):
    model = LinearRegression()
    model.fit(train_data, train_label.ravel())
    test_label = model.predict(test_data)
    save2csv(test_label, 'sklearn_logistic_result.csv')

def sklearn_knn(train_data, train_label, test_data):
    model = KNeighborsClassifier(n_neighbors=6)
    model.fit(train_data, train_label.ravel())
    test_label = model.predict(test_data)
    save2csv(test_label, 'sklearn_knn_result.csv')

def sklearn_random_forest(train_data, train_label, test_data):
    model = RandomForestClassifier(n_estimators=1000, min_samples_split=5)
    model = model.fit(train_data, train_label.ravel() )
    test_label = model.predict(test_data)
    save2csv(test_label, 'sklearn_random_forest.csv')

def sklearn_svm(train_data, train_label, test_data):
    model = svm.SVC(C=14, kernel='rbf', gamma=0.001, cache_size=200)
    # svm.SVC(C=6.2, kernel='poly', degree=4, coef0=0.48, cache_size=200)
    model.fit(train_data, train_label.ravel() )
    test_label = model.predict(test_data)

    save2csv(test_label, 'sklearn_svm_rbf_result.csv')

def sklearn_GaussianNB(train_data, train_label, test_data):
    model = GaussianNB()
    model.fit(train_data, train_label.ravel())
    test_label = model.predict(test_data)
    save2csv(test_label, 'sklearn_GaussianNB_Result.csv')

def sklearn_MultinomialNB(train_data, train_label, test_data):
    model = MultinomialNB(alpha=0.1)
    model.fit(train_data, train_label.ravel())
    test_label = model.predict(test_data)
    save2csv(test_label, 'sklearn_MultinomialNB_Result.csv')


def main():
    train_data, train_label, test_data = load_data()

    #sklearn_logistic(train_data, train_label, test_data)

    #sklearn_knn(train_data, train_label, test_data)

    #sklearn_random_forest(train_data, train_label, test_data)

    sklearn_svm(train_data, train_label, test_data)

    # naive bayes 0.5~
    #sklearn_GaussianNB(train_data, train_label, test_data)
    #sklearn_MultinomialNB(train_data, train_label, test_data)

if __name__ == '__main__':
    main()