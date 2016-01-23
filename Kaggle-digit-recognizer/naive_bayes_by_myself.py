#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
@filename: naive_bayes_by_myself.py
@author: yew1eb
@site: http://blog.yew1eb.net
@contact: yew1eb@gmail.com
@time: 2015/12/25 1:51
'''

import numpy as np
import time

def csv2vector(file_name):
    pass

def savefile(labels, file_name):
    pass


def trainNB0(trainMatrix,trainclass):
    numpics = len(trainMatrix)  #record numbers
    numpix = len(trainMatrix[0])#pix numbers
    pDic={}
    for v in trainclass:
        pDic[v] = pDic.get(v,0)+1
    for k,v in list(pDic.items()):
        pDic[k]=v/float(numpics)#p of every class
    pnumdic={}
    psumdic={}
    for k in list(pDic.keys()):
        pnumdic[k]=np.ones(numpix)
    for i in range(numpics):
        pnumdic[trainclass[i]] += trainMatrix[i]
        psumdic[trainclass[i]] = psumdic.get(trainclass[i],2) + sum(trainMatrix[i])
    pvecdic={}
    for k in list(pnumdic.keys()):
        pvecdic[k]=np.log(pnumdic[k]/float(psumdic[k]))
    return pvecdic,pDic

def classifyNB(vec2class,pvecdic,pDic):
    presult={}
    for k in list(pDic.keys()):
        presult[k]=sum(vec2class*pvecdic[k])+np.log(pDic[k])
    tmp=float("-inf")
    result=""
    for k in list(presult.keys()):
        if presult[k]>tmp:
            tmp= presult[k]
            result=k
    return result

def testNB():
    print("load train data...")
    trainSet, trainlabel=csv2vector("train.csv",1)
    print("load test data...")
    testSet,testlabel = csv2vector("test.csv")
    print("start train...")
    pvecdic,pDic=trainNB0(trainSet, trainlabel)
    start = time.clock()
    print("start test...")
    result="ImageId,Label\n"
    for i in range(len(testSet)):
        tmp = classifyNB(testSet[i],pvecdic,pDic)
        result += str(i+1)+","+tmp+"\n"
        #print tmp
    savefile(result,"result_NB.csv")
    end = time.clock()
    print(("time cost: %f s" % (end - start)))
