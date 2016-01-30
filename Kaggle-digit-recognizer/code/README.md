kaggle-digit-recognizer
=======================

Code for Python digit recognizer starter comptetition. 
http://www.kaggle.com/c/digit-recognizer

benchmark kNN k=10, cover_tree gives 0.96557 accuracy
benchmark rf 1000 trees gives 0.96829 accuracy

knnpy-knn/experiment2 gives same accuracy as benchmark,
but if we reduce dimmensions with PCA to lets say ~100 components
kNN performs better than rf benchmark.

plotted explained variance of PCA helps to choose optimal component
count to keep 90% variance

py-knn
======
For kNN experiments

py-rf
======
For random forests experiments

libraries used
=====================
* numpy
* scipy
* sklearn (scikit-learn)
* matplotlib