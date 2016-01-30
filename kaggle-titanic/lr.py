#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_path = './data/'
train = pd.read_csv(base_path+'train.csv')

# 初步观察数据
#print(train.info())
'''
特征信息：
PassengerId => 乘客ID
Pclass => 乘客等级(1/2/3等舱位)
Name => 乘客姓名
Sex => 性别
Age => 年龄
SibSp => 堂兄弟/妹个数
Parch => 父母与小孩个数
Ticket => 船票信息
Fare => 票价
Cabin => 客舱
Embarked => 登船港口

Age,Cabin列有缺失
Name,Sex,Ticket,Cabin,Embarked列为分类类型
'''

#print(train.describe())
'''
查看数值类型特征的统计信息
'''

# 数据初步分析
'''
看看每个/多个 属性和最后的Survived之间有着什么样的关系
'''

def analyze_features(train):
    fig = plt.figure()

    fig.set(alpha=0.2) # 设定图表颜色alpha参数
    plt.subplot2grid((2,3), (0,0)) # 在一张大图里分列几个小图
    plt.title('显示中文')
    train.Survived.value_counts().plot(kind='bar') # 柱状图
    plt.title('获救情况 (1为获救)') # 标题
    plt.ylabel('人数')

    plt.subplot2grid((2,3),(0,1))
    train.Pclass.value_counts().plot(kind='bar')
    plt.ylabel('人数')
    plt.title('乘客等级分布')

    plt.subplot2grid((2,3),(0,2))
    plt.scatter(train.Survived, train.Age)
    plt.ylabel('年龄')
    plt.grid(b=True, which='major', axis='y')
    plt.title('按年龄看获救分布（1为获救)')

    plt.subplot2grid((2,3),(1,0), colspan=2)
    train.Age[train.Pclass == 1].plot(kind='kde')
    train.Age[train.Pclass == 2].plot(kind='kde')
    train.Age[train.Pclass == 3].plot(kind='kde')
    plt.xlabel("年龄")# plots an axis lable
    plt.ylabel("密度")
    plt.title("各等级的乘客年龄分布")
    plt.legend(('头等舱', '2等舱','3等舱'),loc='best') # sets our legend for our graph.


    plt.subplot2grid((2,3),(1,2))
    train.Embarked.value_counts().plot(kind='bar')
    plt.title("各登船口岸上船人数")
    plt.ylabel("人数")
    plt.show()

analyze_features(train)
'''
不同舱位/乘客等级可能和财富/地位有关系，最后获救概率可能会不一样
年龄对获救概率也一定是有影响的，毕竟前面说了，副船长还说『小孩和女士先走』呢
和登船港口是不是有关系呢？也许登船港口不同，人的出身地位不同？
'''
# http://blog.csdn.net/han_xiaoyang/article/details/49797143