#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
@filename: knn_by_myself.py
@author: yew1eb
@site: http://blog.yew1eb.net
@contact: yew1eb@gmail.com
@time: 2015/12/24 下午 9:59

https://www.kaggle.com/c/digit-recognizer/
score: 0.96300
pred test
time cost: 12311.181267s
os:  windows 10
CPU: AMD A6-4400M APU with Radeon(tm) Graphics 2.70GHz
RAM: 6GB
'''
import numpy as np
import time


def load_data():
    train_data = np.loadtxt('d:/dataset/digits/train.csv', dtype=np.uint8, delimiter=',', skiprows=1)
    test_data = np.loadtxt('d:/dataset/digits/test.csv', dtype=np.uint8, delimiter=',', skiprows=1)
    label = np.ravel(train_data[:, :1])  # 多维转一维 扁平化
    data = np.where(train_data[:, 1:] != 0, 1, 0)  # 数据归一化
    test = np.where(test_data != 0, 1, 0)
    return data, label, test


def test_knn(train_data, train_label, test_data, test_label):
    start = time.clock()
    error = 0
    m = len(test_data)
    labels = []
    for i in range(m):
        calc_label = classify(test_data[i], train_data, train_label, 3)
        labels.append(calc_label)
        error = error + (calc_label != test_label[i])

    print(('error: ', error))
    print(('error percent: %f' % (float(error) / m)))
    print(('time cost: %f s' % (time.clock() - start)))


def save2csv(labels, csv_name):
    f = open('d:/dataset/digits/' + csv_name, 'w')
    f.write('ImageId,Label\n')
    for i in range(1, len(labels)+1):
        f.write(str(i)+','+str(labels[i]))
        f.write("\n")
    f.close()


def knn_pred(train_data, train_label, test_data):
    start = time.clock()
    m = len(test_data)
    labels = []
    for i in range(m):
        calc_label = classify(test_data[i], train_data, train_label, 3)
        labels.append(calc_label)
    save2csv(labels, 'knn_result.csv')
    print(('time cost: %f s' % (time.clock() - start)))


def classify(inx, train_data, train_label, k):
    sz = train_data.shape[0]
    inx_temp = np.tile(inx, (sz, 1)) - train_data
    sq_inx_temp = inx_temp ** 2
    sq_distance = sq_inx_temp.sum(axis=1)
    distance = sq_distance ** 0.5
    sort_dist = distance.argsort()
    class_set = {}
    for i in range(k):
        label = train_label[sort_dist[i]]
        class_set[label] = class_set.get(label, 0) + 1
    sorted_class_set = sorted(list(class_set.items()), key=lambda d: d[1], reverse=True)  # 按字典中的从大到小排序
    # python2.7 -> python3.5 : itertimes() -> items()
    return sorted_class_set[0][0]
