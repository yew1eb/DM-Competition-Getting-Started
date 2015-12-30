import csv
import numpy
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA

## http://blog.csdn.net/u012162613/article/details/43157801
input_df = pd.read_csv('train.csv', header=0)
submit_df  = pd.read_csv('test.csv',  header=0)

# merge the two DataFrames into one
df = pd.concat([input_df, submit_df])
df = df .reset_index()
df = df.drop('index', axis=1)
df = df.reindex_axis(input_df.columns, axis=1)


features = input_df.values[:, 1:]
labels = input_df.values[:,0]

pca = PCA(n_components = 64)
pca.fit(df.values[:,1:])
features = pca.transform(features)
pred_data = pca.transform(submit_df.values)

clf = KNeighborsClassifier().fit(features, labels)
#print cross_val_score(clf, features, labels)
output = clf.predict(pred_data).astype(int)
ids = range(1, 28001)
# write to csv file
predictions_file = open("KNN.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ImageId","Label"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()

print "done."