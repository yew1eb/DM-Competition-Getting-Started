import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print ("Starting...")
print(('Number of training examples {0} '.format(train.shape[0])))
print((train.Loan_Status.value_counts()))
print(('Number of test examples {0} '.format(test.shape[0])))


cat_vbl = {'Gender','Married','Dependents','Self_Employed','Property_Area'}
num_vbl = {'LoanAmount','Loan_Amount_Term','Credit_History'}

for var in num_vbl:
    train[var] = train[var].fillna(value = train[var].mean())
    test[var] = test[var].fillna(value = test[var].mean())
train['Credibility'] = train['ApplicantIncome'] / train['LoanAmount']
test['Credibility'] = test['ApplicantIncome'] / test['LoanAmount']

print ("Starting Label Encode")
for var in cat_vbl:
    lb = preprocessing.LabelEncoder()
    full_data = pd.concat((train[var],test[var]),axis=0).astype('str')
    lb.fit( full_data )
    train[var] = lb.transform(train[var].astype('str'))
    test[var] = lb.transform(test[var].astype('str'))

train = train.fillna(value = -999)
test = test.fillna(value = -999)
print ("Filled Missing Values")

features = ['Credibility',
            'Gender',
            'Married',
            'Dependents',
            'Self_Employed',
            'Property_Area',
            'ApplicantIncome',
            'CoapplicantIncome',
            'LoanAmount',
            'Loan_Amount_Term',
            'Credit_History'
]

x_train = train[features].values
y_train = train['Loan_Status'].values
x_test = test[features].values

# Random Forest
rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, oob_score = True, max_features = "auto",random_state=10, min_samples_split=2, min_samples_leaf=2)
rf.fit(x_train, y_train)
print(('Training accuracy:', rf.oob_score_))


print ("Starting to predict on the dataset")
rec= rf.predict(x_test)

print ("Prediction Completed")
test['Loan_Status'] = rec
test.to_csv('result_sklearn_rf.csv',columns=['Loan_ID','Loan_Status'],index=False)
