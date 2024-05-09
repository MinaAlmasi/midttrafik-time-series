import pandas as pd
from tqdm import tqdm

def read_data(data_path, chunksize:int=None, sep=";"):
    """
    Read data from path. 
    """
    # read data
    if chunksize is None:
        df = pd.read_csv(data_path, sep=sep)

    else:
        reader = pd.read_csv(data_path, sep=sep, chunksize=chunksize)
        dfs = [chunk for chunk in tqdm(reader)]
        df = pd.concat(dfs, ignore_index=True)

    return df    
