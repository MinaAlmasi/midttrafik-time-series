import pandas as pd
import pathlib
from pmdarima.arima.utils import ndiffs, nsdiffs

import sys 
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from data_utils import split_rolling_origin, impute_missing

def plot_timeseries(df, y, figsize:tuple=(40,10), ylim:tuple=(0,75), save_path=None, save_file=None, **kwargs):
    ts_plot = df.plot(x="ds", y=y, figsize=figsize, ylim=ylim, **kwargs)

    if save_path and save_file:
        ts_plot.get_figure().savefig(save_path / save_file)
    
    return ts_plot

def main():
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / 'data' / "clean_stops"

    # load the data
    df = pd.read_csv(data_path / 'clean_1A_norreport.csv')

    # impute missing values
    df = impute_missing(df, method='rolling', window=24)

    # remove test data (last 36 datapoints) and the gap before it (last 24 datapoints)
    test_size = 36
    gap = 24
    df = df.iloc[:-(test_size+gap)]

    # check how many differences are needed to make the data stationary (d)
    d = ndiffs(df['y'], test='kpss')
    print(f"Standard differencing term (d): {d}")

    # estimate seasonal differencing term (D)
    D = nsdiffs(df['y'], m=24, test='ch')
    print(f"Seasonal differencing term (D): {D}")

    # differencing the data according to d and D
    df['y_diff'] = df['y'].diff(d)

if __name__ == '__main__':
    main()