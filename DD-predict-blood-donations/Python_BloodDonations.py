# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:18:16 2015

@author: pcooman
"""
## ----------------------------------------------------------------------------
# Load libraries
import pandas as pd      # to create and manipulate data frames
import numpy as np       # because math is awesome!
import seaborn as sns    # to make pretty correlation and pairwise plots
import matplotlib.pyplot as plt   # to set up figures

## ----------------------------------------------------------------------------
# read in data and store as data frame
data_dir = "./Data/"
train = pd.read_csv(data_dir + "Training.csv")
test = pd.read_csv(data_dir + "Testing.csv")

## ----------------------------------------------------------------------------
# explore data structure

# Number of observations and number of features
print "Training data: " + str(train.shape[0]) + " rows, " + str(train.shape[1]) + " columns"  
print "Testing data: " + str(test.shape[0]) + " rows, " + str(test.shape[1]) + " columns"  

# Type of features, missing values
train.info()
test.info()

# A quick look at the first 5 rows
print train.head(n=5)
print train.tail(n=5)

# Change the column names to something more manageable
colNames = train.columns.values.tolist()
train.columns = ["Id", "MonthsLast",'NumberDonations',"Volume","MonthsFirst","Donated2007"]
test.columns = ["Id", "MonthsLast",'NumberDonations',"Volume","MonthsFirst"]

# Proportion of positive labels (% of returning donors)
float(np.sum(train["Donated2007"]))/len(train["Donated2007"])

## ----------------------------------------------------------------------------
# Add feature: consistency
train["Consistency"] = (train["MonthsFirst"] - train["MonthsLast"])/train["NumberDonations"]
train = train[["Id", "MonthsLast",'NumberDonations',"Volume","MonthsFirst","Consistency","Donated2007"]]  # change column order
# If the person only made 1 donation, repace Consistency by their MonthsLast value
train.Consistency[train.Consistency == 0] = train.MonthsLast[train.Consistency == 0] 

test["Consistency"] = (test["MonthsFirst"] - test["MonthsLast"])/test["NumberDonations"]
test.Consistency[test.Consistency == 0] = test.MonthsLast[test.Consistency == 0]

# Drop the ID column
train = train.drop("Id", axis = 1)

## ----------------------------------------------------------------------------
# Correlation matrix plot

# Compute the correlation matrix
corr = train.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
sns.set(style="white")

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.show()

## ----------------------------------------------------------------------------
# Pairwise plot

sns.set(style="ticks", color_codes=True)
sns.pairplot(train, hue="Donated2007", diag_kind = "kde",palette = dict([(0, "blue"), (1, "red")]),vars=["MonthsLast",'NumberDonations',"Volume","MonthsFirst","Consistency"])

## ----------------------------------------------------------------------------
# End of exploratory analysis

## ----------------------------------------------------------------------------
# See code for exploratory analysis in previous post

from sklearn import metrics

## ----------------------------------------------------------------------------
# Drop Volume (perfectly correlated with NumberDonations)

train = train.drop("Volume", axis = 1)
test = test.drop("Volume", axis = 1)

## ----------------------------------------------------------------------------
# pre-process data
labels = train["Donated2007"]

# Drop label column from the training data frame
train = train.drop("Donated2007", axis = 1)

# Scale training and testing data
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
trainMatrix = train.as_matrix().astype(float)
train_scaled = min_max_scaler.fit_transform(trainMatrix)
train_scaled = pd.DataFrame(train_scaled)
train_scaled.columns = train.columns.values.tolist()
testMatrix = test.as_matrix().astype(float)
test_scaled = min_max_scaler.fit_transform(testMatrix)
test_scaled = pd.DataFrame(test_scaled)
test_scaled.columns = test.columns.values.tolist()

from sklearn.cross_validation import train_test_split
TrainFeats, TestFeats, TrainLabels, TestLabels = train_test_split(train_scaled, labels, test_size=0.25)

from sklearn import cross_validation
kf = cross_validation.KFold(len(TrainLabels), n_folds=5, shuffle = False, random_state = 123)

## ----------------------------------------------------------------------------
# 1. Linear Discriminant Analysis
from sklearn.lda import LDA     # loads the library

score_train = np.array([])
score_test = np.array([])

for train_index, test_index in kf:
    CVTrainFeats, CVTestFeats = TrainFeats[train_index], TrainFeats[test_index]
    CVTrainLabels, CVTestLabels = TrainLabels[train_index], TrainLabels[test_index]

    model = LDA()
    model.fit(CVTrainFeats, CVTrainLabels)
    score_train = np.append(score_train,metrics.log_loss(CVTrainLabels, model.predict_proba(CVTrainFeats)))
    score_test = np.append(score_test,metrics.log_loss(CVTestLabels, model.predict_proba(CVTestFeats)))
    score = metrics.log_loss(TestLabels, model.predict_proba(TestFeats))

# To make sure we're not overfitting
print("Average CV Training Log loss: %.2f" % np.mean(score_train))
print("Average CV Testing Log loss: %.2f" % np.mean(score_test))
print("Testing Log loss: %.2f" % score)


print metrics.confusion_matrix(CVTestLabels,model.predict(CVTestFeats),labels = [1,0])

        
## ----------------------------------------------------------------------------
# 2. Logistic Regression
from sklearn.linear_model import LogisticRegression

score_train = np.array([])
score_test = np.array([])

for train_index, test_index in kf:
    CVTrainFeats, CVTestFeats = TrainFeats[train_index], TrainFeats[test_index]
    CVTrainLabels, CVTestLabels = TrainLabels[train_index], TrainLabels[test_index]

    model = LogisticRegression()
    model.fit(CVTrainFeats, CVTrainLabels)
    score_train = np.append(score_train,metrics.log_loss(CVTrainLabels, model.predict_proba(CVTrainFeats)))
    score_test = np.append(score_test,metrics.log_loss(CVTestLabels, model.predict_proba(CVTestFeats)))
    score = metrics.log_loss(TestLabels, model.predict_proba(TestFeats))

# To make sure we're not overfitting
print("Average CV Training Log loss: %.2f" % np.mean(score_train))
print("Average CV Testing Log loss: %.2f" % np.mean(score_test))
print("Testing Log loss: %.2f" % score)

print metrics.confusion_matrix(CVTestLabels,model.predict(CVTestFeats),labels = [1,0])

## ----------------------------------------------------------------------------
# 3. Naive Bayes

from sklearn.naive_bayes import GaussianNB

score_train = np.array([])
score_test = np.array([])

for train_index, test_index in kf:
    CVTrainFeats, CVTestFeats = TrainFeats[train_index], TrainFeats[test_index]
    CVTrainLabels, CVTestLabels = TrainLabels[train_index], TrainLabels[test_index]

    model = GaussianNB()
    model.fit(CVTrainFeats, CVTrainLabels)
    score_train = np.append(score_train,metrics.log_loss(CVTrainLabels, model.predict_proba(CVTrainFeats)))
    score_test = np.append(score_test,metrics.log_loss(CVTestLabels, model.predict_proba(CVTestFeats)))
    score = metrics.log_loss(TestLabels, model.predict_proba(TestFeats))

# To make sure we're not overfitting
print("Average CV Training Log loss: %.2f" % np.mean(score_train))
print("Average CV Testing Log loss: %.2f" % np.mean(score_test))
print("Testing Log loss: %.2f" % score)

print metrics.confusion_matrix(CVTestLabels,model.predict(CVTestFeats),labels = [1,0])

## ----------------------------------------------------------------------------            
# 4. Classification tree
from sklearn import tree

score_train = np.array([])
score_test = np.array([])

for train_index, test_index in kf:
    CVTrainFeats, CVTestFeats = TrainFeats[train_index], TrainFeats[test_index]
    CVTrainLabels, CVTestLabels = TrainLabels[train_index], TrainLabels[test_index]

    model = tree.DecisionTreeClassifier()
    model.fit(CVTrainFeats, CVTrainLabels)
    score_train = np.append(score_train,metrics.log_loss(CVTrainLabels, model.predict_proba(CVTrainFeats)))
    score_test = np.append(score_test,metrics.log_loss(CVTestLabels, model.predict_proba(CVTestFeats)))
    score = metrics.log_loss(TestLabels, model.predict_proba(TestFeats))

# To make sure we're not overfitting
print("Average CV Training Log loss: %.2f" % np.mean(score_train))
print("Average CV Testing Log loss: %.2f" % np.mean(score_test))
print("Testing Log loss: %.2f" % score)

print metrics.confusion_matrix(CVTestLabels,model.predict(CVTestFeats),labels = [1,0])

tree.export_graphviz(model,out_file='tree_tall.dot')   

## ----------------------------------------------------------------------------
score_train = np.array([])
score_test = np.array([])

for train_index, test_index in kf:
    CVTrainFeats, CVTestFeats = TrainFeats[train_index], TrainFeats[test_index]
    CVTrainLabels, CVTestLabels = TrainLabels[train_index], TrainLabels[test_index]

    model = tree.DecisionTreeClassifier(max_depth = 3)
    model.fit(CVTrainFeats, CVTrainLabels)
    score_train = np.append(score_train,metrics.log_loss(CVTrainLabels, model.predict_proba(CVTrainFeats)))
    score_test = np.append(score_test,metrics.log_loss(CVTestLabels, model.predict_proba(CVTestFeats)))
    score = metrics.log_loss(TestLabels, model.predict_proba(TestFeats))

# To make sure we're not overfitting
print("Average CV Training Log loss: %.2f" % np.mean(score_train))
print("Average CV Testing Log loss: %.2f" % np.mean(score_test))
print("Testing Log loss: %.2f" % score)

print metrics.confusion_matrix(CVTestLabels,model.predict(CVTestFeats),labels = [1,0])

tree.export_graphviz(model,out_file='tree_stump.dot')   

## ----------------------------------------------------------------------------
# 5. Support Vector Machine
from sklearn.svm import SVC

score_train = np.array([])
score_test = np.array([])

for train_index, test_index in kf:
    CVTrainFeats, CVTestFeats = TrainFeats[train_index], TrainFeats[test_index]
    CVTrainLabels, CVTestLabels = TrainLabels[train_index], TrainLabels[test_index]

    model = SVC(probability=True)
    model.fit(CVTrainFeats, CVTrainLabels)
    score_train = np.append(score_train,metrics.log_loss(CVTrainLabels, model.predict_proba(CVTrainFeats)))
    score_test = np.append(score_test,metrics.log_loss(CVTestLabels, model.predict_proba(CVTestFeats)))
    score = metrics.log_loss(TestLabels, model.predict_proba(TestFeats))

# To make sure we're not overfitting
print("Average CV Training Log loss: %.2f" % np.mean(score_train))
print("Average CV Testing Log loss: %.2f" % np.mean(score_test))
print("Testing Log loss: %.2f" % score)

print metrics.confusion_matrix(CVTestLabels,model.predict(CVTestFeats),labels = [1,0])
           
## ----------------------------------------------------------------------------
# 6. Random Forest
from sklearn.ensemble import RandomForestClassifier

score_train = np.array([])
score_test = np.array([])

for train_index, test_index in kf:
    CVTrainFeats, CVTestFeats = TrainFeats[train_index], TrainFeats[test_index]
    CVTrainLabels, CVTestLabels = TrainLabels[train_index], TrainLabels[test_index]

    model = RandomForestClassifier(n_estimators=100, max_depth=3)
    model.fit(CVTrainFeats, CVTrainLabels)
    score_train = np.append(score_train,metrics.log_loss(CVTrainLabels, model.predict_proba(CVTrainFeats)))
    score_test = np.append(score_test,metrics.log_loss(CVTestLabels, model.predict_proba(CVTestFeats)))
    score = metrics.log_loss(TestLabels, model.predict_proba(TestFeats))

# To make sure we're not overfitting
print("Average CV Training Log loss: %.2f" % np.mean(score_train))
print("Average CV Testing Log loss: %.2f" % np.mean(score_test))
print("Testing Log loss: %.2f" % score)

print metrics.confusion_matrix(CVTestLabels,model.predict(CVTestFeats),labels = [1,0])

## ----------------------------------------------------------------------------
# 7. Gradient Boosting Machine
from sklearn.ensemble import AdaBoostClassifier

score_train = np.array([])
score_test = np.array([])

for train_index, test_index in kf:
    CVTrainFeats, CVTestFeats = TrainFeats[train_index], TrainFeats[test_index]
    CVTrainLabels, CVTestLabels = TrainLabels[train_index], TrainLabels[test_index]

    model = AdaBoostClassifier()
    model.fit(CVTrainFeats, CVTrainLabels)
    score_train = np.append(score_train,metrics.log_loss(CVTrainLabels, model.predict_proba(CVTrainFeats)))
    score_test = np.append(score_test,metrics.log_loss(CVTestLabels, model.predict_proba(CVTestFeats)))
    score = metrics.log_loss(TestLabels, model.predict_proba(TestFeats))

# To make sure we're not overfitting
print("Average CV Training Log loss: %.2f" % np.mean(score_train))
print("Average CV Testing Log loss: %.2f" % np.mean(score_test))
print("Testing Log loss: %.2f" % score)

print metrics.confusion_matrix(CVTestLabels,model.predict(CVTestFeats),labels = [1,0])

## ----------------------------------------------------------------------------
# Ensemble: mean probability

# Linear Discriminant Analysis
model = LDA()
model.fit(train_scaled, labels)
X_LDA = model.predict_proba(train_scaled)

# Logistic Regression
model = LogisticRegression()
model.fit(train_scaled, labels)
X_LR = model.predict_proba(train_scaled)

# Naive Bayes
model = GaussianNB()
model.fit(train_scaled, labels)
X_NB = model.predict_proba(train_scaled)

# Classification Tree
model = tree.DecisionTreeClassifier(max_depth = 3)
model.fit(train_scaled, labels)
X_CART = model.predict_proba(train_scaled)

# Support Vector Machine
model = SVC(probability=True)
model.fit(train_scaled, labels)
X_SVC = model.predict_proba(train_scaled)

# Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=3)
model.fit(train_scaled, labels)
X_RF = model.predict_proba(train_scaled)

# Gradient Boosting Machine
model = AdaBoostClassifier()
model.fit(train_scaled, labels)
X_GBM = model.predict_proba(train_scaled)

X_train_0 = np.vstack((X_LDA[:,0],X_LR[:,0],X_NB[:,0],X_CART[:,0],X_SVC[:,0],X_RF[:,0],X_GBM[:,0]))
X_train_1 = np.vstack((X_LDA[:,1],X_LR[:,1],X_NB[:,1],X_CART[:,1],X_SVC[:,1],X_RF[:,1],X_GBM[:,1]))

X_train_0 = np.mean(X_train_0, axis = 0)
X_train_1 = np.mean(X_train_1, axis = 0)

X_train = np.vstack((X_train_0,X_train_1)).T

print("Ensemble Training Log loss: %.2f" % metrics.log_loss(labels, X_train))