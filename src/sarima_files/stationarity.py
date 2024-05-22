import pandas as pd
import numpy as np
import pathlib
from pmdarima.arima.utils import ndiffs, nsdiffs

def plot_timeseries(df, y, figsize:tuple=(40,10), ylim:tuple=(0,75), save_path=None, save_file=None, **kwargs):
    ts_plot = df.plot(x="ds", y=y, figsize=figsize, ylim=ylim, **kwargs)

    if save_path and save_file:
        ts_plot.get_figure().savefig(save_path / save_file)
    
    return ts_plot

def main():
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / 'data'

    # load the data
    df = pd.read_csv(data_path / 'processed_1A_norreport.csv')

    # remove test data (last 36 datapoints) and the gap before it (last 24 datapoints)
    test_size = 36
    gap = 24
    df = df.iloc[:-(test_size+gap)]

    # fill out missing values with 0
    #df['y'] = df['y'].fillna(0)

    # check how many differences are needed to make the data stationary (d)
    d = ndiffs(df['y'], test='kpss')
    print(f"Number of differences needed to make the data stationary (d): {d}")

    # estimate seasonal differencing term (D)
    D = nsdiffs(df['y'], m=24, test='ocsb')
    print(f"Seasonal differencing term (D): {D}")

    # differencing the data according to d and D
    df['y_diff'] = df['y'].diff(d)

    # create another plot that is zoomed into the last two month (24 hours x 30 days * 2 months)
    df_recent = df.tail(2*24*30)
    plot_timeseries(df_recent, y="y_diff", figsize=(25,10), ylim=(0,70), save_path=plot_dir, save_file="norreport_1A_ts_recent.png")

    # create another that is just the last two days
    df_two_days = df.tail(2*24)
    plot_timeseries(df_two_days, y="y_diff", figsize=(15,10), ylim=(0,60), save_path=plot_dir, save_file="norreport_1A_ts_two_days.png", linewidth=2)

if __name__ == '__main__':
    main()