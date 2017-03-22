# Data Mining Competition Getting Started
***************
## Analytics Vidhya
### AV Loan Prediction [url](http://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction#)  
  仅作为练习的小问题, 根据用户的特征预测是否发放住房贷款，二分类问题  
  总11个特征(Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area)
  ，Loan_ID是用户ID，Loan_Status是需要预测的，特征包含数值类型和分类类型

## Data Castle
### 微额借款用户人品预测大赛 [url](http://pkbigdata.com/common/competition/148.html)
   同上，区别在与这个的特征比较多

## Kaggle
### Digit Recognizer [url](https://www.kaggle.com/c/digit-recognizer)
多分类练习题

### Titanic: Machine Learning from Disaster [url](https://www.kaggle.com/c/titanic)  
二分类问题，给出0/1即可，评价指标为accuracy。

### Bag of Words Meets Bags of Popcorn [url](https://www.kaggle.com/c/word2vec-nlp-tutorial)  
这是一个文本情感二分类问题。评价指标为AUC。
http://www.cnblogs.com/lijingpeng/p/5787549.html

### Display Advertising Challenge [url](https://www.kaggle.com/c/criteo-display-ad-challenge)  
这是一个广告CTR预估的比赛，由知名广告公司Criteo赞助举办。数据包括4千万训练样本，500万测试样本，特征包括13个数值特征，26个类别特征，评价指标为logloss。
CTR工业界做法一般都是LR，只是特征会各种组合/transform，可以到上亿维。这里我也首选LR，特征缺失值我用的众数，对于26个类别特征采用one-hot编码，
数值特征我用pandas画出来发现不符合正态分布，有很大偏移，就没有scale到[0,1]，
采用的是根据五分位点（min,25%,中位数,75%,max）切分为6个区间(负值/过大值分别分到了1和6区间作为异常值处理)，然后一并one-hot编码，最终特征100万左右，训练文件20+G。
强调下可能遇到的坑：1.one-hot最好自己实现，除非你机器内存足够大(需全load到numpy，而且非sparse);2.LR最好用SGD或者mini-batch，
而且out-of-core模式(http://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html#example-applications-plot-out-of-core-classification-py), 
除非还是你的内存足够大;3.Think twice before code.由于数据量大，中间出错重跑的话时间成品比较高。
我发现sklearn的LR和liblinear的LR有着截然不同的表现，sklearn的L2正则化结果好于L1，liblinear的L1好于L2，我理解是他们优化方法不同导致的。
最终结果liblinear的LR的L1最优，logloss=0.46601，LB为227th/718，这也正符合lasso产生sparse的直觉。
我也单独尝试了xgboost，logloss=0.46946，可能还是和GBRT对高维度sparse特征效果不好有关。Facebook有一篇论文把GBRT输出作为transformed feature喂给下游的线性分类器，
取得了不错的效果，可以参考下。（Practical Lessons from Predicting Clicks on Ads at Facebook）
我只是简单试验了LR作为baseline，后面其实还有很多搞法，可以参考forum获胜者给出的solution，
比如：1. Vowpal Wabbit工具不用区分类别和数值特征；2.libFFM工具做特征交叉组合；3.feature hash trick；4.每个特征的评价点击率作为新特征加入；5.多模型ensemble等。
