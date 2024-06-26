import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
import pathlib

def read_data(data_path, chunksize:int=None, sep=";"):
    '''
    Read data from path. 
    '''
    # read data
    if chunksize is None:
        df = pd.read_csv(data_path, sep=sep)

    else:
        reader = pd.read_csv(data_path, sep=sep, chunksize=chunksize)
        dfs = [chunk for chunk in tqdm(reader)]
        df = pd.concat(dfs, ignore_index=True)

    return df    

def split_sliding_window(X:pd.Series, gap:int=48, val_size:int=48, max_train_size:int=10):
    '''
    Split the data into small windows of train and test (sliding window approach). 
    Uses TimeSeriesSplit from sklearn.

    Args
        X: data 
        gap: gap between train and test
        val_size: size of each validation set in each fold
        max_train_size: maximum size of the training set

    Returns
        train_inds: dictionary with train indices
        test_inds: dictionary with test indices
    '''

    # identify the length of a full fold
    fold_length = gap + val_size + max_train_size

    # subtract from length of data to get the number of folds
    n_splits = (int(len(X) - fold_length) // val_size)
    print(f"Number of splits: {n_splits}")

    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap,
        max_train_size=max_train_size,
        val_size=val_size
    )
    data_generator = tscv.split(X)

    # create lists of lists for train and test indices
    train_inds = {}
    test_inds = {}

    # iterate over the generator object
    for i, (train_ind, test_ind) in enumerate(data_generator):
        train_inds[i] = train_ind
        test_inds[i] = test_ind

    return train_inds, test_inds

def split_rolling_origin(X, gap:int=24, val_size:int=36, test_size:int=36, steps:int=1, min_train_size:int=24*7):
    '''
    function to split the data into n_splits using TimeSeriesSplit

    Args
        X: data
        gap: gap between train and test
        val_size: size of each validation set in each fold
        steps: number of steps to take for rolling origin
        min_train_size: minimum size of the training set

    Returns
        train_inds, val_inds, test_inds: dictionary with train, test indices
    '''
    # get only indicies for the test set from the data
    test_inds = X.index[-test_size:].tolist()

    # define dropped inds as the test indices and the gap before it 
    dropped_inds = X.index[-(test_size + gap-1):].tolist()

    # remove test indices and the gap before it from the data
    X_train_val = X.drop(index=dropped_inds, axis=0)

    # subtract from length of data to get the number of folds1
    n_splits = (len(X_train_val) - min_train_size - gap) // val_size 

    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap,
        test_size=val_size,
    )
    # create a generator object
    data_generator = tscv.split(X_train_val)

    # create lists of lists for train and test indices
    all_train_inds = {}
    all_val_inds = {}

    # iterate over the generator object
    for i, (train_ind, val_ind) in enumerate(data_generator):
        all_train_inds[i] = train_ind
        all_val_inds[i] = val_ind

    # compute step_size for rolling origin
    step_size = steps * val_size
    print(f"[INFO:] Step size for rolling origin: {step_size}")

    # subset the key according to step_size but we always include the maximum key
    train_inds = {k: v for k, v in all_train_inds.items() if k % steps == 0 or k == len(all_train_inds) - 1}
    val_inds = {k: v for k, v in all_val_inds.items() if k % steps == 0 or k == len(all_val_inds) - 1}

    print("Number of folds for rolling origin:", len(train_inds))

    return train_inds, val_inds, test_inds


def impute_missing(df, method='rolling', window=24):
    '''
    Impute missing values in a dataframe using a specified method.
    Iteratively applies the method to handle large gaps of missing values.
    '''
    if method == 'linear':
        df['y'] = df['y'].interpolate(method='linear', limit_direction='both', limit=window)
    elif method == 'rolling':
        # We need to loop until no more NaNs are filled to handle large gaps
        last_nan_count = df['y'].isna().sum()
        while True:
            df['y'] = df['y'].fillna(df['y'].rolling(window=window, min_periods=1).mean())
            current_nan_count = df['y'].isna().sum()
            if current_nan_count == last_nan_count:  # No more NaNs are filled
                break
            last_nan_count = current_nan_count
    else:
        raise ValueError(f"Method {method} not recognized. Please use 'linear' or 'rolling'.")

    return df