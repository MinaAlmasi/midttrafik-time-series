import pathlib
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf
import matplotlib.pyplot as plt


def plot_autocorrelation(data, n_lags=24*7*2, file_name="norreport_1A_autocorrelation.png", save_dir=pathlib.Path(__file__).parents[1] / "plots"):
    '''
    Plot autocorrelation of time series data

    Args
        data: dataframe
        n_lags: int, number of lags to include in the plot (defaults to 2 weeks i.e., 24*7*2)
        save_dir: path to save the plot
        file_name: name of the file to save the plot as
    '''

    print("[INFO:] Creating autocorrelation plot")
    fig, ax = plt.subplots(figsize=(24, 6)) 
    autocor_plot = plot_acf(data["y"], lags=n_lags, zero=False, ax=ax) # markersize = 1, linewidth = 0.5 (could be added)

    # change axis tick values from lags to hours 
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([int(xtick)/2 for xtick in xticks])
    ax.set_xlim(0, n_lags)

    # labels
    ax.set_title('')  
    ax.set_xlabel('Hours')  
    ax.set_ylabel('Autocorrelation') 

    # rm whitespace
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / file_name, dpi=300)  


def plot_partial_autocorrelation(data, n_lags, file_name="norreport_1A_partial_autocorrelation.png", save_dir=pathlib.Path(__file__).parents[1] / "plots"):

    print("[INFO:] Creating partial autocorrelation plot")
    fig, ax = plt.subplots(figsize=(24, 6)) 
    pacf_plot = plot_pacf(data["y"], lags=n_lags, zero=False, ax=ax) # markersize = 1, linewidth = 0.5 (could be added)

    # change axis tick values from lags to hours 
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([int(xtick)/2 for xtick in xticks])
    ax.set_xlim(0, n_lags)

    # labels
    ax.set_title('')  
    ax.set_xlabel('Hours')  
    ax.set_ylabel('Partial Autocorrelation') 

    # rm whitespace
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / file_name, dpi=300)

def main():
    # set paths
    path = pathlib.Path(__file__)
    data_dir = path.parents[1] / "data"
    plot_dir = path.parents[1] / "plots"

    # load data
    data = pd.read_csv(data_dir / "processed_1A_norreport.csv")

    # replace NA values with 0 
    data.fillna(0, inplace=True)

    # plots
    plot_autocorrelation(data, n_lags=24*7*2, file_name="norreport_1A_autocorrelation.png", save_dir=plot_dir)

    plot_partial_autocorrelation(data, n_lags=24*7+1, file_name="norreport_1A_partial_autocorrelation.png", save_dir=plot_dir)


   

if __name__ == "__main__":
    main()