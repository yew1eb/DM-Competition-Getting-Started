#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
@filename: sklearn-random-forest.py
@author: yew1eb
@site: http://blog.yew1eb.net
@contact: yew1eb@gmail.com
@time: 2015/12/27 下午 10:18
'''
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


def load_data():
    df = pd.read_csv('D:/dataset/titanic/train.csv', header=0)
#特征选择
#   只取出三个自变量
#   将Age（年龄）缺失的数据补全
#   将Pclass变量转变为三个哑（Summy）变量
#   将sex转为0-1变量
    subdf = df[['Pclass', 'Sex', 'Age']]
    y = df.Survived
    age = subdf['Age'].fillna(value=subdf.Age.mean())
    pclass = pd.get_dummies(subdf['Pclass'], prefix='Pclass')
    sex = (subdf['Sex']=='male').astype('int')
    X = pd.concat([pclass, age, sex], axis=1)
    #print(X.head())
    return X, y

# 分析各特征的重要性
# feature_importance = clf.feature_importances_
# 对于随机森林如何得到变量的重要性，可以看scikit-learn官方文档 http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#example-ensemble-plot-forest-importances-py
# important_features = X_train.columns.values[0::]
# analyze_feature(feature_importance, important_features)
def analyze_feature(feature_importance, important_features):
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)[::-1]
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    plt.title('Feature Importance')
    plt.barh(pos, feature_importance[sorted_idx[::-1]], color='r', align='center')
    plt.yticks(pos, important_features)
    plt.xlabel('Relativ Importance')
    plt.draw()
    plt.show()

def sklearn_decisoin_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=6)
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
    bst = clf.fit(X_train, y_train)
    # 准确率 print("accuracy rate: {:.6f}".format(bst.score(X_test, y_test)))
# 交叉验证
    scores = cross_validation.cross_val_score(clf, X, y, cv=10)
    print(scores)


def main():
    X, y = load_data()




if __name__ == '__main__':
    main()





