import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

def inverse_mapping(pred):
    mapping_dict = {1: 'Y', 0: 'N'}
    return mapping_dict[pred]

def create_submissions(ids, preds, filename='baseline_submission.csv'):
    submission = pd.read_csv('./data/Sample_Submission_ZAuTl8O.csv')

    submission['Loan_ID'] = ids
    submission['Loan_Status'] = preds

    submission.to_csv('./submissions/' + filename, index=False)

def binary_from_prob(preds, threshold=0.5):
    return np.array(['Y' if pred > threshold else 'N' for pred in preds])

def get_mask(dataset, train_size=0.7):
    """
    Returns the boolean mask which can be used to split the dataset
    """

    loantrain, loantest = train_test_split(xrange(dataset.shape[0]), train_size=train_size)
    loanmask = np.ones(dataset.shape[0], dtype='int')
    loanmask[loantrain] = 1
    loanmask[loantest] = 0
    loanmask = (loanmask==1)

    return loanmask

def split_dataset(X, y):
    """
    Splits the training dataset into X_train and X_val
    """

    loanmask = get_mask(X)

    X_train = X[loanmask]
    y_train = y[loanmask]

    X_val = X[~loanmask]
    y_val = y[~loanmask]

    return (X_train, X_val, y_train, y_val)

def vectorizer(train_df, test_df):
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()

    train = train_df_copy.T.to_dict().values()
    test = test_df_copy.T.to_dict().values()

    vec = DictVectorizer()
    train = vec.fit_transform(train)
    test = vec.transform(test)

    return train, test

def encode(train_df, test_df):
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()
    all_object_cols = get_all_object_cols(train_df_copy)

    train_df, test_df = fill_nan(train_df_copy, test_df_copy, all_object_cols)
    train_df, test_df = encode_labels(train_df_copy, test_df_copy, all_object_cols)

    return train_df, test_df

def get_all_object_cols(train_df):
    return [col for col in train_df.columns if train_df[col].dtype == 'O']

def encode_labels(train_df, test_df, cols):
    for col in cols:

        lbl = LabelEncoder()
        feature = list(train_df[col].copy())
        feature.extend(test_df[col])
        lbl.fit(feature)

        train_df[col] = lbl.transform(train_df[col])
        test_df[col] = lbl.transform(test_df[col])

    return train_df, test_df

def fill_nan(train_df, test_df, cols):
    for col in cols:
        train_df[col].fillna('-999', inplace=True)
        test_df[col].fillna('-999', inplace=True)

    return train_df, test_df
