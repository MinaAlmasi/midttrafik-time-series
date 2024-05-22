import pathlib 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

import sys 
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from data_utils import impute_missing


def plot_timeseries(df, figsize:tuple=(40,10), ylim:tuple=(0,75), label_share=10, save_path=None, save_file=None, **kwargs):
    # reset index to ensure that the x-axis is the index
    df = df.reset_index()

    # set font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # create plot
    ts_plot = df.plot(x="ds", y="y", figsize=figsize, ylim=ylim, **kwargs)

    # rm legend
    ts_plot.get_legend().remove()

    # set limits on x-axis based on the DataFrame's index size
    ts_plot.set_xlim(0, len(df) - 1)

    # add x and y labels
    ts_plot.set_xlabel("Datetime", fontsize=24, labelpad=16)
    ts_plot.set_ylabel("Passenger Count", fontsize=24, labelpad=16)

    # rotate x-axis labels 90 degrees
    ts_plot.set_xticklabels(ts_plot.get_xticklabels(), rotation=90)

    # configure ticks to show a label every 'label_share' values
    tick_positions = range(0, len(df), label_share)  # generates a range from 0 to the length of df, stepping by 'label_share'
    tick_labels = [df.loc[i, 'ds'] if i in df.index else "" for i in tick_positions]  # get labels for these positions
    ts_plot.set_xticks(tick_positions)  # set the tick positions
    ts_plot.set_xticklabels(tick_labels)  # set the tick labels

    # increase size of x and y values
    ts_plot.tick_params(axis='both', which='major', labelsize=16)

    # tighten the layout
    plt.tight_layout()

    if save_path and save_file:
        plt.savefig(save_path / save_file, dpi=200)  # ensure to use plt.savefig for correct saving

    return ts_plot


def plot_decompose(df, freq:int=24, label_share:int=10, save_path=None, save_file=None):
    # set font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # decompose the timeseries
    decomp = seasonal_decompose(df['y'], period=freq, model='additive')

    # plot the decomposed timeseries
    decomp_plot = decomp.plot()

    # set size
    decomp_plot.set_size_inches((20, 10))

    # remove "y" title
    decomp_plot.axes[0].set_title("")

    # ensure y-axis is only whole numbers
    decomp_plot.axes[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    decomp_plot.axes[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    decomp_plot.axes[2].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    decomp_plot.axes[3].yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # increase font of x and y labels
    decomp_plot.axes[0].set_ylabel("Observed (Y)", fontsize=22, labelpad=20)
    decomp_plot.axes[1].set_ylabel("Trend (T)", fontsize=22, labelpad=20)
    decomp_plot.axes[2].set_ylabel("Seasonality (S)", fontsize=22, labelpad=10)
    decomp_plot.axes[3].set_ylabel("Remainder (R)", fontsize=22, labelpad=10)

    # increase font of x and y values
    decomp_plot.axes[0].tick_params(axis='both', which='major', labelsize=17)
    decomp_plot.axes[1].tick_params(axis='both', which='major', labelsize=17)
    decomp_plot.axes[2].tick_params(axis='both', which='major', labelsize=17)
    decomp_plot.axes[3].tick_params(axis='both', which='major', labelsize=17)

    # set x-axis label
    decomp_plot.axes[3].set_xlabel("Observations", fontsize=22, labelpad=16)

    # tighten the layout
    plt.tight_layout()

    # increase ticks on x-axis
    decomp_plot.axes[0].set_xticks(range(df.index[0], df.index[-1], label_share))

    if save_path and save_file:
        decomp_plot.get_figure().savefig(save_path / save_file, dpi=300)

    return decomp_plot

def main():
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "data"
    plot_dir = path.parents[2] / "plots" / "graphical_analysis"

    # load data
    df = pd.read_csv(data_path / "processed_1A_norreport.csv")

    # plot the timeseries for Cumulative
    plot_timeseries(df, figsize=(25,10), ylim=(0,75), label_share = 500, save_path=plot_dir, save_file="norreport_1A_ts.png", linewidth=0.8)

    # create another plot that is zoomed into the last two month (24 hours x 30 days * 2 months)
    df_recent = df.tail(2*24*30)
    plot_timeseries(df_recent, figsize=(25,10), ylim=(0,65), label_share = 40, save_path=plot_dir, save_file="norreport_1A_ts_recent.png", linewidth=2)

    # create another that is just the last two days
    df_two_days = df.tail(2*24)
    plot_timeseries(df_two_days, figsize=(25,10), ylim=(0,60), label_share = 2, save_path=plot_dir, save_file="norreport_1A_ts_two_days.png", linewidth=3)

    # create a custom subset that covers the snowstorm days (from 03-01-2024 to 08-01-2024)
    df_snowstorm = df[(df['ds'] >= '2023-12-25') & (df['ds'] <= '2024-01-15')]
    plot_timeseries(df_snowstorm, figsize=(25,10), ylim=(0,60), label_share = 10, save_path=plot_dir, save_file="norreport_1A_ts_snowstorm.png", linewidth=3)

    # impute missing before decomposing
    df = impute_missing(df, method='rolling', window=24)
    
    # plot the decomposed timeseries
    plot_decompose(df, freq=24, label_share = 800, save_path=plot_dir, save_file="norreport_1A_decompose.png")

    # plot the decomposed timeseries for the last two days
    plot_decompose(df_recent, freq=24, label_share = 100, save_path=plot_dir, save_file="norreport_1A_decompose_recent.png")

if __name__ == "__main__":
    main()