from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import time
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from ggplot import *

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
train = train.drop(['Loan_ID'], axis=1)
test_uid = test['Loan_ID']
test  = test.drop(['Loan_ID'], axis=1)
#print(train.info())

#Convert Categorical data to number and making any missing value -9999
from sklearn import preprocessing
def convert(data):
    number = preprocessing.LabelEncoder()
    for i in data.columns:
        if data[i].dtype == 'object':
            data[i] = pd.factorize(data[i])[0]
    data = data.fillna(-9999)
    return data

train = convert(train)
test  = convert(test)

#Divide Train into train and Validate
train, validate = train_test_split(train, train_size=0.7)

#select Inuput & Target Feature
features = ['Gender',
            'Married',
            'Dependents',
            'Education',
            'Self_Employed',
            'ApplicantIncome',
            'CoapplicantIncome',
            'LoanAmount',
            'Loan_Amount_Term',
            'Credit_History',
            'Property_Area'
]
x_train = train[features].values
x_validate = validate[features].values
y_train = train['Loan_Status'].values
y_validate = validate['Loan_Status'].values
x_test = test[features].values

#Build Random Forest
'''
min_samples_leaf score
5,10 0.777777777778
7,15,16 0.784722222222
'''
rf = RandomForestClassifier(n_estimators=1000,min_samples_leaf=16, max_features="auto", n_jobs=2, random_state=0)
rf.fit(x_train, y_train)

#Look at Important Feature
importances = rf.feature_importances_
indices = np.argsort(importances)

ind=[]
for i in indices:
    ind.append(features[i])

def feature_importances():
    plt.figure()
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)),ind)
    plt.xlabel('Relative Importance')
    plt.show()

#Plot ROC_AUC curve and cross validate
status = rf.predict_proba(x_validate)
fpr, tpr, _ = roc_curve(y_validate, status[:, 1])
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
ggplot(df, aes(x='fpr', y='tpr')) + geom_line() + geom_abline(linetype='dashed')

roc_auc = auc(fpr, tpr)
print(roc_auc)

#Predict for test data set and export test data set

status = rf.predict_proba(x_test)
pred = np.where(status[:,0] > 0.5, 'Y', 'N')
print(pd.value_counts(pred))
result = pd.DataFrame({"Loan_ID":test_uid, "Loan_Status":pred}, columns=['Loan_ID','Loan_Status'])
result.to_csv('result/rf_'+str(time.time())[-4:]+'.csv', index=False)