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

def split_sliding_window(X:pd.Series, gap:int=48, test_size:int=48, max_train_size:int=10):
    '''
    Split the data into small windows of train and test (sliding window approach). 
    Uses TimeSeriesSplit from sklearn.

    Args
        X: data 
        gap: gap between train and test
        test_size: size of the test set
        max_train_size: maximum size of the training set

    Returns
        train_inds: dictionary with train indices
        test_inds: dictionary with test indices
    '''
    # identify the length of a full fold
    fold_length = gap + test_size + max_train_size

    # subtract from length of data to get the number of folds
    n_splits = (int(len(X) - fold_length) // test_size)
    print(f"Number of splits: {n_splits}")

    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap,
        max_train_size=max_train_size,
        test_size=test_size
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

def split_rolling_origin(X, gap:int=24, test_size:int=36, steps:int=1, min_train_size:int=24*7):
    '''
    function to split the data into n_splits using TimeSeriesSplit
    '''

    # subtract from length of data to get the number of folds1
    n_splits = (int(len(X) - min_train_size) // test_size)

    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap,
        test_size=test_size,
    )
    # create a generator object
    data_generator = tscv.split(X)

    # create lists of lists for train and test indices
    train_inds = {}
    test_inds = {}

    # iterate over the generator object
    for i, (train_ind, test_ind) in enumerate(data_generator):
        train_inds[i] = train_ind
        test_inds[i] = test_ind

    # compute step_size for rolling origin
    step_size = steps * test_size
    print(f"[INFO:] Step size for rolling origin: {step_size}")

    # subset the key according to step_size i.e., if steps = 4, keep every 4th key e..g, 0, 4, 8, 12, ...
    train_inds = {k: v for k, v in train_inds.items() if k % steps == 0}
    test_inds = {k: v for k, v in test_inds.items() if k % steps == 0}

    print("Number of folds for rolling origin:", len(train_inds))

    return train_inds, test_inds

def main(): 
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / 'data'

    # load the data
    df = pd.read_csv(data_path / 'processed_1A_norreport.csv')

    # split the data
    gap = 24
    max_train_size = 24 * 7 * 2 # 24 hours x 7 (days) x 2 (weeks)
    #generator = split_timeseries_data(df['ds'], gap=gap, test_size=48, max_train_size=max_train_size)
    train_inds, test_inds = split_rolling_origin(df['ds'], gap=gap, test_size=36, steps=4, min_train_size=24*7)

    # print the test_inds for key 0 and 552
    print(train_inds[0])
    print(test_inds[0])


if __name__ == "__main__":
    main()