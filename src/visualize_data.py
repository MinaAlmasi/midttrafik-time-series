import pathlib 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

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


def plot_decompose(df, freq:int=24, save_path=None, save_file=None):
    # decompose the timeseries
    decomp = seasonal_decompose(df['y'], period=freq)

    # plot the decomposed timeseries
    decomp_plot = decomp.plot()
    decomp_plot.set_size_inches((40, 10))

    if save_path and save_file:
        decomp_plot.get_figure().savefig(save_path / save_file)

    return decomp_plot

def main():
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / "data"
    plot_dir = path.parents[1] / "plots" / "graphical_analysis"

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
    
    # plot the decomposed timeseries
    #plot_decompose(df, freq=24, save_path=plot_dir, save_file="norreport_1A_decompose.png")

    # plot the decomposed timeseries for the last two days
    #plot_decompose(df_recent, freq=24, save_path=plot_dir, save_file="norreport_1A_decompose_recent.png")

if __name__ == "__main__":
    main()