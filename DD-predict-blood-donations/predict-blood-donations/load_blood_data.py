def load_blood_data(train=True, SEED=97, scale  = False, 
                                         minmax = False,
                                         norm   = False,
                                         nointercept = False,
                                         engineering = False):
    """
    Load training and test datasets
    for DrivenData's Predict Blood Donations warmup contest
    
    The training data is shuffled before it's returned; test data is not
    
    Note: patsy returns float64 data; Theano requires float32 so conversion
          will be required; the y values are converted to int32, so they're OK
    
    Arguments
    ---------
        train (bool) if True
                         y_train, X_train = load_blood_data(train=True, ...
                     if False
                         X_test, IDs = load_blood_data(train=False, ...
                         
        SEED (int)   random seed
        
        scale (bool) if True, scale the data to mean zero, var 1; standard normal
        
        minmax (2-tuple) to scale the data to a specified range, provide a
                         2-tuple (min, max)
                         
        norm (bool)  if True, L2 normalize for distance and similarity measures
        
        nointercept (bool) if True, patsy will not create an intercept
                         
                         
    Usage
    -----
    from load_blood_data import load_blood_data
    """
    from sklearn.utils         import shuffle
    from patsy                 import dmatrices, dmatrix
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import Normalizer
    import numpy  as np
    import pandas as pd
    import re
    
    global scaler
    global minmaxer
    global normalizer
    
    if (scale and minmax): raise ValueError("cannot specify both scale and minmax")
    if (scale and norm):   raise ValueError("cannot specify both scale and norm")
    if (norm  and minmax): raise ValueError("cannot specify both norm and minmax")
        
    if type(train) is not bool: raise ValueError("train must be boolean")
    if type(SEED)  is not int:  raise ValueError("SEED must be int")
    if type(scale) is not bool: raise ValueError("scale must be boolean")
    if type(norm)  is not bool: raise ValueError("norm must be boolean")
    if type(nointercept) is not bool: raise ValueError("nointercept must be boolean")
    if type(engineering) is not bool: raise ValueError("engineering must be boolean")
    
    # ------------- read the file -------------
    
    file_name = '../input/train.csv' if train else '../input/test.csv'
    data = pd.read_csv(file_name)
    
    
    # ------------- shorten the column names -------------
    
    column_names = ['ID','moSinceLast','numDonations','volume','moSinceFirst','donated']
    data.columns = column_names if train else column_names[:-1]
    
    
    # ------------- create new variables -------------
    
    if engineering:
        # Ratio of moSinceLast / moSinceFirst = moRatio
        data['moRatio'] = pd.Series(data.moSinceLast / data.moSinceFirst, index=data.index)
    
        # Ratio of (volume/numDonations) / moSinceFirst = avgDonation
        data['avgDonation'] = pd.Series((data.volume/data.numDonations) / data.moSinceFirst, index=data.index)
    
        # Ratio of moSinceFirst / numDonations = avgWait
        data['avgWait'] = pd.Series(data.moSinceFirst / data.numDonations, index=data.index)

        
    # ------------- scale the data -------------

    # transform data to mean zero, unit variance
    # ==========================================
    if scale:
        if train:
            scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
            exclude = ['ID','donated']
            data.ix[:, data.columns.difference(exclude)] = scaler.fit_transform(
                data.ix[:, data.columns.difference(exclude)].values.astype(np.float32))
        else:
            exclude = ['ID','donated']
            data.ix[:, data.columns.difference(exclude)] = scaler.transform(
                data.ix[:, data.columns.difference(exclude)].values.astype(np.float32))
            
    # transform data to fit in a range
    # ================================
    if minmax:
        if len(minmax) != 2: raise ValueError("minmax must be a 2-tuple")
        if train:
            minmaxer = MinMaxScaler(feature_range = minmax)
            exclude = ['ID','donated']
            data.ix[:, data.columns.difference(exclude)] = minmaxer.fit_transform(
                data.ix[:, data.columns.difference(exclude)].values.astype(np.float32))
        else:
            exclude = ['ID','donated']
            data.ix[:, data.columns.difference(exclude)] = minmaxer.transform(
                data.ix[:, data.columns.difference(exclude)].values.astype(np.float32))
            
    # transform data to unit vector (L2 norm for distance and similarity)
    # ===================================================================
    if norm:
        if train:
            normalizer = Normalizer(norm='l2', copy=True)
            exclude = ['ID','donated']
            data.ix[:, data.columns.difference(exclude)] = normalizer.fit_transform(
                data.ix[:, data.columns.difference(exclude)].values.astype(np.float32))
        else:
            exclude = ['ID','donated']
            data.ix[:, data.columns.difference(exclude)] = normalizer.transform(
                data.ix[:, data.columns.difference(exclude)].values.astype(np.float32))
        
        
    # ------------- create the design matrix -------------
        
    # create the datasets with a patsy formula
    formula = 'donated ~ moSinceLast * moSinceFirst +  numDonations + volume'
    
    if engineering:
        formula = formula + ' + moRatio + avgDonation + avgWait'
        
    if nointercept: 
        formula = formula + ' -1'
        
    if not train:
        match = re.search(r"~\s??(.*)", formula)
        if match:
            formula = match.group(1)
        else:
            raise ValueError("Patsy formula {} does not match the expected format".format(formula))
            
            
    # ------------- return the values -------------
            
    if train:
        y_train, X_train = dmatrices(formula, data=data, return_type="dataframe")
        y_train = np.ravel(y_train).astype(np.int32)
        
        X_train, y_train = shuffle(X_train, y_train, random_state=SEED)
        return y_train, X_train
    else:
        X_test = dmatrix(formula, data=data, return_type="dataframe")
        IDs    = data.ID.values
        return X_test, IDs
