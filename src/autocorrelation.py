import pathlib
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf
import matplotlib.pyplot as plt


def plot_autocorrelation(data, plot_dir):
     # create autocorrelation plot
    n_lags = 48 * 7 * 2 # 24 hours x 7 (days) x 2 (weeks)

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
    fig.savefig(plot_dir / "norreport_1A_autocorrelation.png", dpi=300)  


def plot_partial_autocorrelation(data, plot_dir):
    pass

def main():
    # set paths
    path = pathlib.Path(__file__)
    data_dir = path.parents[1] / "data"
    plot_dir = path.parents[1] / "plots"

    # load data
    data = pd.read_csv(data_dir / "processed_1A_norreport.csv")

    # plot autocorrelation
    plot_autocorrelation(data, plot_dir)

   

if __name__ == "__main__":
    main()