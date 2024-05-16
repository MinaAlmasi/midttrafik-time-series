import pathlib
import pandas as pd 
from pmdarima.arima import auto_arima

import sys 
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from data_utils import split_rolling_origin


def main():
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "data"

    # load data
    df = pd.read_csv(data_path / 'processed_1A_norreport.csv')

    # data setup
    gap = 24 # gap between train and val and train and test
    steps = 4 # how fast rolling origin moves
    test_size = 36 # hours 
    min_train_size = 24*7 # i.e. 1 week 


    # split the data into train and test
    train_inds, val_inds, _ = split_rolling_origin(df['ds'], gap=gap, test_size=test_size, steps=steps, min_train_size=min_train_size)

    # iterate over each fold
    for (train_ind, val_ind) in zip(train_inds.values(), val_inds.values()):
        # fit an ARIMA model
        model = auto_arima(df['y'].iloc[train_ind], seasonal=True, m=24)

        print(model.summary())
    
    
if __name__ == "__main__":
    main()