## Use Google's Word2Vec for movie reviews

In this tutorial competition, we dig a little "deeper" into sentiment analysis. Google's Word2Vec is a deep-learning inspired method that focuses on the meaning of words. Word2Vec attempts to understand meaning and semantic relationships among words. It works in a way that is similar to deep approaches, such as recurrent neural nets or deep neural nets, but is computationally more efficient. This tutorial focuses on Word2Vec for sentiment analysis.

Sentiment analysis is a challenging subject in machine learning. People express their emotions in language that is often obscured by sarcasm, ambiguity, and plays on words, all of which could be very misleading for both humans and computers. There's another Kaggle competition for movie review sentiment analysis. In this tutorial we explore how Word2Vec can be applied to a similar problem.

Deep learning has been in the news a lot over the past few years, even making it to the front page of the New York Times. These machine learning techniques, inspired by the architecture of the human brain and made possible by recent advances in computing power, have been making waves via breakthrough results in image recognition, speech processing, and natural language tasks. Recently, deep learning approaches won several Kaggle competitions, including a drug discovery task, and cat and dog image recognition.


## Data Set Description 
<https://www.kaggle.com/c/word2vec-nlp-tutorial/data>


# Word2Vec 简介

Word2vec 是 Google 在 2013 年年中开源的一款将词表征为实数值向量的高效工具, 其利用深度学习的思想，可以通过训练，把对文本内容的处理简化为 K 维向量空间中的向量运算，而向量空间上的相似度可以用来表示文本语义上的相似度。Word2vec输出的词向量可以被用来做很多 NLP 相关的工作，比如聚类、找同义词、词性分析等等。如果换个思路， 把词当做特征，那么Word2vec就可以把特征映射到 K 维向量空间，可以为文本数据寻求更加深层次的特征表示 。

Word2vec 使用的是 Distributed representation 的词向量表示方式。Distributed representation 最早由 Hinton在 1986 年提出[4]。其基本思想是 通过训练将每个词映射成 K 维实数向量（K 一般为模型中的超参数），通过词之间的距离（比如 cosine 相似度、欧氏距离等）来判断它们之间的语义相似度.其采用一个 三层的神经网络 ，输入层-隐层-输出层。有个核心的技术是 根据词频用Huffman编码 ，使得所有词频相似的词隐藏层激活的内容基本一致，出现频率越高的词语，他们激活的隐藏层数目越少，这样有效的降低了计算的复杂度。而Word2vec大受欢迎的一个原因正是其高效性，Mikolov 在论文[2]中指出，一个优化的单机版本一天可训练上千亿词。

这个三层神经网络本身是 对语言模型进行建模 ，但也同时 获得一种单词在向量空间上的表示 ，而这个副作用才是Word2vec的真正目标。

与潜在语义分析（Latent Semantic Index, LSI）、潜在狄立克雷分配（Latent Dirichlet Allocation，LDA）的经典过程相比，Word2vec利用了词的上下文，语义信息更加地丰富。


   
   
* [文本深度表示模型Word2Vec](http://wei-li.cnblogs.com/p/word2vec.html)    
* [深度学习word2vec笔记之基础篇](http://blog.csdn.net/mytestmy/article/details/26961315)  
* [深度学习word2vec笔记之算法篇](http://blog.csdn.net/mytestmy/article/details/26969149)  
* [基于Kaggle数据的词袋模型文本分类教程](http://www.csdn.net/article/1970-01-01/2825782)  
* [情感分析的新方法——基于Word2Vec/Doc2Vec/Python](http://datartisan.com/article/detail/48.html)  
http://nbviewer.jupyter.org/github/MatthieuBizien/Bag-popcorn/blob/master/Kaggle-Word2Vec.ipynb
***************

这是一个文本情感二分类问题。25000的labeled训练样本，只有一个raw text 特征”review“。
评价指标为AUC，所以这里提交结果需要用概率，我开始就掉坑里了，结果一直上不来。
比赛里有教程如何使用word2vec进行二分类，可以作为入门学习材料。
我没有使用word embeddinng，直接采用BOW及ngram作为特征训练，效果还凑合，后面其实可以融合embedding特征试试。
对于raw text我采用TfidfVectorizer(stop_words=’english’, ngram_range=(1,3), sublinear_tf=True, min_df=2)，
并采用卡方检验进行特征选择，经过CV，最终确定特征数为200000。
单模型我选取了GBRT/NB/LR/linear SVC。
GBRT一般对于维度较大比较稀疏效果不是很好，但对于该数据表现不是很差。
NB采用MultinomialNB效果也没有想象的那么惊艳。
几个模型按效果排序为linear SVC(0.95601)>LR(0.94823)>GBRT(0.94173)>NB(0.93693)，看来线性SVM在文本上还是很强悍的。
后续我又采用LDA生成主题特征，本来抱着很大期望，现实还是那么骨感，采用上述单模型AUC最好也只有0.93024。
既然单独使用主题特征没有提高，那和BOW融合呢？果然work了!
后面试验证实特征融合还是linear SVC效果最好，LDA主题定为500，而且不去除停用词效果更好，AUC为0.95998。
既然没有时间搞单模型了，还有最后一招，多模型融合。这里有一个原则就是模型尽量多样，不一定要求指标最好。
最终我选取5组不是很差的多模型结果进行average stacking，AUC为0.96115，63位。
最终private LB跌倒了71st，应该融合word enbedding试试，没时间细搞了。




http://cs.stanford.edu/~quocle/paragraph_vector.pdf
* https://cs224d.stanford.edu/reports/SadeghianAmir.pdf
* 用简单的TDF 作为Feature，然后用简单的M-Bayesian方法来进行分类。
http://nbviewer.ipython.org/github/jmsteinw/Notebooks/blob/master/NLP_Movies.ipynb