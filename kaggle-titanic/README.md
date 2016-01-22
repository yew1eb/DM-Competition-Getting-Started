说句题外话，网上貌似有遇难者名单，LB上好几个score 1.0的。有坊间说，score超过90%就怀疑作弊了，不知真假，不过top300绝大多数都集中在0.808-0.818。这个题目我后面没有太多的改进想法了，求指导啊~
数据包括数值和类别特征，并存在缺失值。类别特征这里我做了one-hot-encode，缺失值是采用均值/中位数/众数需要根据数据来定，我的做法是根据pandas打印出列数据分布来定。
模型我采用了DT/RF/GBDT/SVC，由于xgboost输出是概率，需要指定阈值确定0/1，可能我指定不恰当，效果不好0.78847。
效果最好的是RF，0.81340。这里经过筛选我使用的特征包括’Pclass’,’Gender’, ‘Cabin’,’Ticket’,’Embarked’,’Title’进行onehot编码，’Age’,’SibSp’,’Parch’,’Fare’,’class_age’,’Family’ 归一化。
我也尝试进行构建一些新特征和特征组合，比如title分割为Mr/Mrs/Miss/Master四类或者split提取第一个词，添加fare_per_person等，pipeline中也加入feature selection，但是效果都没有提高，求指导~



[kaggle数据挖掘竞赛初步--Titanic](http://www.cnblogs.com/north-north/tag/kaggle/)

[Kaggle Titanic Competition Part I – Intro]
(http://www.ultravioletanalytics.com/2014/10/30/kaggle-titanic-competition-part-i-intro/)


[Kaggle Competition | Titanic Machine Learning from Disaster]
(http://nbviewer.ipython.org/github/agconti/kaggle-titanic/blob/master/Titanic.ipynb)

https://github.com/agconti/kaggle-titanic

http://www.sotoseattle.com/blog/categories/kaggle/

[Titanic: Machine Learning from Disaster - Getting Started With R]
https://github.com/trevorstephens/titanic
https://github.com/wehrley/wehrley.github.io/blob/master/SOUPTONUTS.md


http://mlwave.com/tutorial-titanic-machine-learning-from-distaster/
Full Titanic Example with Random Forest
https://www.youtube.com/watch?v=0GrciaGYzV0

[Tutorial: Titanic dataset machine learning for Kaggle]
(http://corpocrat.com/2014/08/29/tutorial-titanic-dataset-machine-learning-for-kaggle/)

[Getting Started with R: Titanic Competition in Kaggle]
(http://armandruiz.com/kaggle/Titanic_Kaggle_Analysis.html)

[A complete guide to getting 0.79903 in Kaggle’s Titanic Competition with Python](https://triangleinequality.wordpress.com/2013/09/05/a-complete-guide-to-getting-0-79903-in-kaggles-titanic-competition-with-python/)
[机器学习系列(3)_逻辑回归应用之Kaggle泰坦尼克之灾](http://blog.csdn.net/han_xiaoyang/article/details/49797143)
  https://www.kaggle.com/malais/titanic/kaggle-first-ipythonnotebook/notebook