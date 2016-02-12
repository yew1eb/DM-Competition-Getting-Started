
这是一个多分类问题，一般三种处理方法：one vs all, one vs one, softmax，信息损失逐渐递减。
87809条训练数据，数据包括datetime/类别/数值特征，没有缺失值，label共39种。
评价指标为logloss，这里要说下和AUC的区别，AUC更强调相对排序。
我抽取后特征包括year,m,d,h,m,dow,district,address,x,y，模型选择softmax objective的LR和xgboost。
这两个模型对特征挑食，有不同的偏好，LR喜好0/1类别或者locale到0-1的数值特征，而xgboost更喜好原始的数值特征，而且对缺失值也能很好的处理。
所以对于LR就是2个归一化的数值特征和8个待one-hot编码的特征，对于xgboost是8个原始数值特征（包括year/m/d等，具有连续性）和2个待one-hot编码的特征。
LR效果要略好于xgboost效果，logloss分别为2.28728/2.28869，最好位置为3rd，目前跌到4th，后面找时间再搞一搞。