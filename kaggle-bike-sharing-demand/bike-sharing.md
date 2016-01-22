
## 特征信息
datetime   - hourly date + timestamp    
season     -  1 = spring, 2 = summer, 3 = fall, 4 = winter   
holiday    - whether the day is considered a holiday  
workingday - whether the day is neither a weekend nor holiday  
weather    - 1: Clear, Few clouds, Partly cloudy, Partly cloudy   
             2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist   
             3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds   
             4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog   
temp       - temperature in Celsius  
atemp      - "feels like" temperature in Celsius  
humidity   - relative humidity  
windspeed  - wind speed  
casual     - number of non-registered user rentals initiated  
registered - number of registered user rentals initiated  
count      - number of total rentals  

最后预测租车数量。这里需要注意一点，最后总数实际等于casual+registered。
原始共10个特征，
包括datetime特征，
season/holiday等类别特征，
temp/atemp等数值特征，
没有特征缺失值。
评价指标为RMSLE，其实就是RMSE原来的p和a加1取ln。
当时正在研究GBRT，所以使用了xgboost。
由于使用RMSLE，xgboost自带的loss是square loss，eval_metric是RMSE，
这时两种选择1.修改xgboost代码，派生新的优化objective，求新objective的gradient（一阶导）/hessian（二阶导），派生新的eval_metric；
2.训练数据的y做ln(y+1)转化，最后预测时再做exp(y^)-1就转回来了。当然2简单了，我也是这么实施的。
关于数据特征处理，datetime转成y/m/d/h/dayofweek，y/m等类别特征由于有连续性，这里没有做one-hot编码。
经过cv最后cut掉了日/season。
Xgboost参数其实没有怎么去调，shrinkage=0.1，tree_num=1000，depth=6，其他默认。
效果三次提升拐点分别是：1.RMSE转换为RMLSE(square loss转为square log loss)，说明预测值的范围很大，log转化后bound更tight了；
2.cut了日/season特征；3.转换为对casual和registered的分别回归问题，在加和。
最后RMLSE结果为0.36512，public LB最好为30位，最终private LB为28，还好说明没有overfit。


[jyfeather/Bike_Sharing](https://github.com/jyfeather/Bike_Sharing)  
[Using Gradient Boosted Trees to Predict Bike Sharing Demand]
(http://blog.dato.com/using-gradient-boosted-trees-to-predict-bike-sharing-demand)  
[Kaggle Bike Sharing Demand Prediction – How I got in top 5 percentile of participants?]
(http://www.analyticsvidhya.com/blog/2015/06/solution-kaggle-competition-bike-sharing-demand/)  

https://www.kaggle.com/benhamner/bike-sharing-demand/what-drives-demand-for-dc-bike-rentals

<http://blog.csdn.net/wiking__acm/article/details/44158353>  

http://nbviewer.ipython.org/gist/whbzju/ff06fce9fd738dcf8096  
<http://efavdb.com/bike-share-forecasting/>
<https://github.com/EFavDB/bike-forecast>  

http://nbviewer.jupyter.org/github/gig1/Python_Kaggle_Byke_Sharing_Demand/blob/master/Bicycle%20Tutorial.ipynb

https://github.com/dirtysalt/tomb/tree/master/kaggle/bike-sharing-demand
