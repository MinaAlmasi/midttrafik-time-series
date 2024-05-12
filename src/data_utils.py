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

def split_timeseries_data(X, gap:int=48, test_size:int=48, max_train_size:int=10):
    '''
    function to split the data into n_splits using TimeSeriesSplit
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

    return data_generator


def main(): 
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / 'data'

    # load the data
    df = pd.read_csv(data_path / 'processed_1A_norreport.csv')

    # split the data
    gap = 24
    max_train_size = 48 * 7 * 2 # 24 hours x 7 (days) x 2 (weeks)
    generator = split_timeseries_data(df['ds'], gap=gap, test_size=48, max_train_size=max_train_size)
    
    for i, (train_ind, test_ind) in enumerate(generator):
        if i == 0 or i == 1:
            print(f"Fold: {i}")
            print(f"Train: {train_ind}")
            print(f"Test: {test_ind}")

if __name__ == "__main__":
    main()