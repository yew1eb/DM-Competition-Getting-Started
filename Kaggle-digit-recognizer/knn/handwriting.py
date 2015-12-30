#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Filename: handwriting.py
@Author: yew1eb
@Date: 2015/12/23 0023
"""
# http://blog.csdn.net/u012162613/article/details/41978235#0-tsina-1-23162-397232819ff9a47a7b7e80a40613cfe1
# http://blog.csdn.net/u013691510/article/details/43195227  Random Forest
# https://www.kaggle.com/c/digit-recognizer/data
# https://github.com/MarcoGiancarli/DigitRecognizer/tree/master/gen
# https://github.com/dzhibas/kaggle-digit-recognizer/tree/master/py-knn
# https://github.com/Broham/DigitRecognizer/blob/master/DigitRecognizer.py


import numpy as np
import operator


def load_data():
    #data = np.loadtxt('d:\\dataset\\digits\\test.csv', dtype=np.uint8 ,delimiter=',', skiprows=1)
    #np.savetxt('d:\\dataset\\digits\\stest.csv', data[:2, ], fmt='%d', delimiter=',')
    train_data = np.loadtxt('d:\\dataset\\digits\\strain.csv', dtype=np.uint8,delimiter=',')
    test_data = np.loadtxt('d:\\dataset\\digits\\stest.csv', dtype=np.uint8,delimiter=',')
    label = train_data[:,:1]
    data  = np.where(train_data[:, 1:]!=0, 1, 0)# 数据归一化
    test  = np.where(test_data !=0, 1, 0)
    return data, label, test

def load_test_label():
    label_data = np.loadtxt('d:\\dataset\\digits\\rf_benchmark.csv', dtype=np.uint8, delimiter=',', skiprows=1)
    label = label_data[:,1]
    return label

#分类主体程序，计算欧式距离，选择距离最小的k个，返回k个中出现频率最高的类别
#inX是所要测试的向量
#dataSet是训练样本集，一行对应一个样本。dataSet对应的标签向量为labels
#k是所选的最近邻数目
def classify(inX, dataSet, labels, k):
    print(inX.shape)
    print(dataSet.shape)
    print(labels.shape)
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet   #tile(A,(m,n))将数组A作为元素构造m行n列的数组
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)  #array.sum(axis=1)按行累加，axis=0为按列累加
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()       #array.argsort()，得到每个元素的排序序号
    classCount={}                                  #sortedDistIndicies[0]表示排序后排在第一个的那个数在原来数组中的下标
    for i in range(k):
        print(sortedDistIndicies[i],0)
        print('l',labels[1,0],'l',labels[0,0])
        voteIlabel = labels[sortedDistIndicies[i],0]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #get(key,x)从字典中获取key对应的value，没有key的话返回0

    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) #sorted()函数，按照第二个元素即value的次序逆向（reverse=True）排序
    return sortedClassCount[0][0]

def main():
    train_data, train_label, test_data = load_data()
    test_label = load_test_label()
    sz = test_data.shape[0]
    error = 0
    test_result = []
    for i in range(sz):
        classifier_result = classify( test_data[i], train_data, train_label, 5)
        test_result.append(classifier_result)
        if(classifier_result != test_label[0,i]): error += 1
    print("\nthe total number of errors is: %d" % error)
    print("\nthe total error rate is: %f" % (error/float(sz)))


if __name__ == '__main__':
    main()




















