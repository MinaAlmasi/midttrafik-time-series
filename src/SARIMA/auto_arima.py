import pathlib
import pandas as pd 
from pmdarima.arima import auto_arima

import sys 
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from data_utils import split_rolling_origin, impute_missing


def main():
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "data"

    # load data
    df = pd.read_csv(data_path / 'processed_1A_norreport.csv')

    # impute missing values
    df = impute_missing(df, method='rolling', window=24)

    # data setup
    gap = 24 # gap between train and val and train and test
    steps = 4 # how fast rolling origin moves
    test_size = 36 # hours 
    min_train_size = 24*7 # i.e. 1 week 

    # split the data, use only train
    train_inds, _, _ = split_rolling_origin(df['ds'], gap=gap, test_size=test_size, steps=steps, min_train_size=min_train_size)

    # get the last value in the train_inds (last fold in rolling origin)
    last_train_inds = list(train_inds.values())[-1]
    
    # get the training data correpsonding to the last train_inds
    train_data = df.iloc[last_train_inds]
    
    # use auto_arima to find the best model (d and D are set based on values found in stationarity.py)
    model = auto_arima(train_data['y'], seasonal=True, m=24, d=1, D=0, stepwise=True, stationary=False, test="kpss", trace=3)

    print(model.summary())

    results_dir = path.parents[2] / "results"
    with open(results_dir / "auto_arima_results.txt", "w") as file:
        file.write(str(model.summary()))
    
if __name__ == "__main__":
    main()