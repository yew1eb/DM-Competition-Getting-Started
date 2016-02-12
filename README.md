# Data Mining Competition Getting Started

***************

## Kaggle
### Bike Sharing Demand [url](https://www.kaggle.com/c/bike-sharing-demand)  | [Data Set Description](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
这是一个回归问题，给出一个城市的自行车租借系统的历史租借数据，要求预测自行车租借数量。

### Bag of Words Meets Bags of Popcorn [url](https://www.kaggle.com/c/word2vec-nlp-tutorial)  
这是一个文本情感二分类问题。评价指标为AUC。

### Titanic: Machine Learning from Disaster [url](https://www.kaggle.com/c/titanic)  
二分类问题，给出0/1即可，评价指标为accuracy。

### San Francisco Crime Classification [url](https://www.kaggle.com/c/sf-crime)


### Caterpillar Tube Pricing [url](https://www.kaggle.com/c/caterpillar-tube-pricing)  
这也是一个回归问题，预测tube报价。30213条训练样本，特征分散在N个文件中，需要你left join起来。评价指标为RMLSE，哈哈，是不是很熟悉？对，跟bike sharing的一样，所以怎么转换知道了吧？看，做多了套路trick也就了然了。感觉这个需要领域知识，但其实有些特征我是不知道含义的，anyway，先merge所有特征不加domain特征直接搞起。
这是我见过小样本里特征处理最麻烦的(后面的CTR大数据处理更耗时)，它特征分散在多个文件中，还好我们有神器pandas，直接left join搞定。这里有些trick需要注意，比如comp_*文件要用append不能join，这样正好是一个全集，否则就会多个weight特征了。特征存在缺失值，这里我全部采用0值，不知是否恰当？
模型我主要试了RF和xgboost，RF tree_num=1000，其他默认值，RMLSE=0.255201，主要精力放在了xgboost上，调了几次参数(depth=58,col_sample=0.75,sample=0.85,shrinkage=0.01,tree_num=2000)，最好RMLSE=0.231220，最好位置120th，目前跌倒206th了，看来需要好好搞搞特征了！

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

### Avito Context Ad Clicks [url](https://www.kaggle.com/c/avito-context-ad-clicks)  
跟上一个CTR比赛不同的是，这个数据没有脱敏，特征有明确含义，userinfo/adinfo/searchinfo等特征需要和searchstream文件 join起来构成完整的训练/测试样本。
数据包含392356948条训练样本，15961515条测试样本，特征基本都是id类别特征和query/title等raw text特征。评价指标还是logloss。
由于数据量太大，跑一组结果太过耗时，根据比赛6的参考，目前我只选择liblinear lasso LR做了一组结果。
最终目标是预测contextual ad，为了减小数据量，*searchstream都过滤了非contextual的，visitstream和phonerequeststream及params目前我都没有使用，
但其实都是很有价值的特征（比如query和title各种similarity），后面可以尝试。
对于这种大数据，在小内存机器上sklearn和pandas处理起来已经非常吃力了，这时就需要自己定制实现left join和one-hot-encoder了，采用按行这种out-of-core方式，
不过真心是慢啊。类似比赛6，price数值特征还是三分位映射成了类别特征和其他类别特征一起one-hot，最终特征大概600万左右，当然要用sparse矩阵存储了，train文件大小40G。
Libliear貌似不支持mini-batch,为了省事没办法只好找一台大内存服务器专门跑lasso LR了。由于上面过滤了不少有价值信息，
也没有类似libFM或libFFM做特征交叉组合，效果不好，logloss只有0.05028，LB排名248th/414。


## Data Castle
### 微额借款用户人品预测大赛 [url](http://pkbigdata.com/common/competition/148.html)

## Analytics Vidhya
### AV Loan Prediction [url](http://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction#)  
  仅作为练习的小问题, 借贷预测，二分类问题  
  总11个特征(Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area)
  ，Loan_ID是用户ID，Loan_Status是需要预测的，特征包含数值类型和分类类型



## DrivenData
### DD-Predict-Blood-Donations [url](http://www.drivendata.org/competitions/2/page/7/) | [Data Set Description](https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center)
  
## 链接
* [my machine learning notes](https://github.com/yew1eb/machine-learning/)  
* [Hacker's Guide to Machine Learning and Predictive Modelling](https://github.com/apeeyush/machine-learning)  
* [Kaggle 机器学习竞赛冠军及优胜者的源代码汇总](http://suanfazu.com/t/kaggle-ji-qi-xue-xi-jing-sai-guan-jun-ji-you-sheng-zhe-de-yuan-dai-ma-hui-zong/230)
* [Kaggle Competitions: How and where to begin?](http://www.analyticsvidhya.com/blog/2015/06/start-journey-kaggle/)