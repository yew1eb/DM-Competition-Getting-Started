import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

"""
Prepare different datasets

1. Label Encoding
2. One Hot Encoding

"""


def get_label_encoded_data(train_df, test_df, obj_cols):
    train_df_cpy = train_df.copy()
    test_df_cpy = test_df.copy()

    for col in obj_cols:
        lbl = LabelEncoder()
        data = pd.concat([train_df_cpy[col], test_df_cpy[col]])

        lbl.fit(data)
        train_df_cpy[col] = lbl.transform(train_df_cpy[col])
        test_df_cpy[col] = lbl.transform(test_df_cpy[col])

    return train_df_cpy, test_df_cpy



def get_dummy_variable_data(train_df, test_df, non_obj_cols, obj_cols):
    train_df_cpy = train_df.copy()
    test_df_cpy = test_df.copy()

    non_obj_train_df = train_df_cpy[non_obj_cols]
    non_obj_test_df = test_df_cpy[non_obj_cols]

    obj_train_df = train_df_cpy[obj_cols]
    obj_test_df = test_df_cpy[obj_cols]

    # fill missing values
    obj_train_df = obj_train_df.fillna('NA')
    obj_test_df = obj_test_df.fillna('NA')

    #transform the categorical to dict
    dict_train_data = obj_train_df.T.to_dict().values()
    dict_test_data = obj_test_df.T.to_dict().values()

    vectorizer = DictVectorizer(sparse=False)
    vec_train_data = vectorizer.fit_transform(dict_train_data)
    vec_test_data = vectorizer.transform(dict_test_data)

    #merge numerical and categorical sets
    x_train = np.column_stack((non_obj_train_df, vec_train_data))
    x_test = np.column_stack((non_obj_test_df, vec_test_data))

    feature_names = list(non_obj_cols.values) + vectorizer.get_feature_names()
    
    x_train = pd.DataFrame(x_train, columns=feature_names)
    x_test = pd.DataFrame(x_test, columns=feature_names)

    return x_train, x_test

