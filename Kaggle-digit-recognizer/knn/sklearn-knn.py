#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Filename: handwriting.py
@Author: yew1eb
@Date: 2015/12/23 0023
"""


import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def load_data():
    #data = np.loadtxt('d:\\dataset\\digits\\test.csv', dtype=np.uint8 ,delimiter=',', skiprows=1)
    #np.savetxt('d:\\dataset\\digits\\stest.csv', data[:2, ], fmt='%d', delimiter=',')
    train_data = np.loadtxt('d:\\dataset\\digits\\strain.csv', dtype=np.uint8,delimiter=',')
    test_data = np.loadtxt('d:\\dataset\\digits\\stest.csv', dtype=np.uint8,delimiter=',')
    label = train_data[:,:1]
    data  = np.where(train_data[:, 1:]!=0, 1, 0)# 数据归一化
    test  = np.where(test_data !=0, 1, 0)
    return data, label, test

def save_result(result, file_name):
    np.savetxt('d:\\dataset\\digits\\'+file_name, zip(np.arange(1,result.shape[0]+1), result),fmt='%d',
               header=['ImageId', 'Label'], delimiter=',')


def sklearn_knn(train_data, train_label, test_data):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(train_data, np.ravel(train_label))
    test_label = model.predict(test_data)
    save_result(test_label, 'sklearn_knn_result.csv')

def main():
    train_data, train_label, test_data = load_data()
    sklearn_knn(train_data, train_label, test_data)

if __name__ == '__main__':
    main()


















