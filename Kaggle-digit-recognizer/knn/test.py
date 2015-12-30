#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Filename: handwriting.py
@Author: yew1eb
@Date: 2015/12/23 0023
"""


import numpy as np



from numpy import *
import csv

def load_data():
    train_data = np.loadtxt('d:\\dataset\\digits\\train.csv', dtype=np.uint8,delimiter=',', skiprows=1)
    test_data = np.loadtxt('d:\\dataset\\digits\\test.csv', dtype=np.uint8,delimiter=',', skiprows=1)
    label = train_data[:,:1]
    data  = np.where(train_data[:, 1:]!=0, 1, 0)# 数据归一化
    test  = np.where(test_data !=0, 1, 0)
    return data, label, test


#result是结果列表
#csvName是存放结果的csv文件名
def save_result(result,csvName):
    with open('d:\\dataset\\digits\\'+csvName,'wb') as myFile:
        myWriter=csv.writer(myFile)
        for i in result:
            tmp=[]
            tmp.append(i)
            myWriter.writerow(tmp)


#调用scikit的knn算法包
from sklearn.neighbors import KNeighborsClassifier
def knnClassify(train_data,train_label,test_data):
    knnClf=KNeighborsClassifier()
    knnClf.fit(train_data,ravel(train_label))
    test_label=knnClf.predict(test_data)
    save_result(test_label,'sklearn_knn_result.csv')
    return test_label

#调用scikit的SVM算法包
from sklearn import svm
def svcClassify(train_data,train_label,test_data):
    svcClf=svm.SVC(C=5.0) #default:C=1.0,kernel = 'rbf'. you can try kernel:‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    svcClf.fit(train_data,ravel(train_label))
    test_label=svcClf.predict(test_data)
    saveResult(test_label,'sklearn_SVC_C=5.0_Result.csv')
    return test_label

#调用scikit的朴素贝叶斯算法包,GaussianNB和MultinomialNB
from sklearn.naive_bayes import GaussianNB      #nb for 高斯分布的数据
def GaussianNBClassify(train_data,train_label,test_data):
    nbClf=GaussianNB()
    nbClf.fit(train_data,ravel(train_label))
    test_label=nbClf.predict(test_data)
    saveResult(test_label,'sklearn_GaussianNB_Result.csv')
    return test_label

from sklearn.naive_bayes import MultinomialNB   #nb for 多项式分布的数据
def MultinomialNBClassify(train_data,train_label,test_data):
    nbClf=MultinomialNB(alpha=0.1)      #default alpha=1.0,Setting alpha = 1 is called Laplace smoothing, while alpha < 1 is called Lidstone smoothing.
    nbClf.fit(train_data,ravel(train_label))
    test_label=nbClf.predict(test_data)
    saveResult(test_label,'sklearn_MultinomialNB_alpha=0.1_Result.csv')
    return test_label


def digitRecognition():
    train_data,train_label, test_data=load_data()

    #使用不同算法
    result1=knnClassify(train_data,train_label,test_data)
    #result2=svcClassify(train_data,train_label,test_data)
    #result3=GaussianNBClassify(train_data,train_label,test_data)
    #result4=MultinomialNBClassify(train_data,train_label,test_data)


if __name__ == '__main__':
    digitRecognition()









