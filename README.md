[Kaggle实战(一) | Nanjunxiao](http://nanjunxiao.github.io/2015/07/30/Kaggle%E5%AE%9E%E6%88%98%E4%B8%80/)  
[Kaggle实战(二) | Nanjunxiao](http://nanjunxiao.github.io/2015/07/31/Kaggle%E5%AE%9E%E6%88%98%E4%BA%8C/)
[Kaggle Competitions: How and where to begin?](http://www.analyticsvidhya.com/blog/2015/06/start-journey-kaggle/)  
# 1. Kaggle Digit Recognizer
此任务是在MNIST（一个带Label的数字像素集合）上训练一个数字分类器，训练集的大小为42000个training example，
每个example是28*28=784个灰度像素值和一个0~9的label。最后的排名以在测试集上的分类正确率为依据排名。
### 数据集格式
一张手写数字图片由28*28=784个像素组成，每一个像素的取值范围[0,255]。  
**训练集train.csv**  
每一行由[label,pixel0~pixel783]，label代表这张图是什么数字。  
**测试集test.csv**  
中没有label这一列，label是需要预测的。  
**提交结果文件name.csv**  
列名 [ImageId,Label]，ImageId对应测试集中的每一行。  

## 其他
* [Kaggle 机器学习竞赛冠军及优胜者的源代码汇总](http://suanfazu.com/t/kaggle-ji-qi-xue-xi-jing-sai-guan-jun-ji-you-sheng-zhe-de-yuan-dai-ma-hui-zong/230)