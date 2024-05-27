import pathlib
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import sys 
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from data_utils import impute_missing



def plot_autocorrelation(data, n_lags=24*7*2, file_name="norreport_1A_autocorrelation.png", save_dir=pathlib.Path(__file__).parents[1] / "plots"):
    print("[INFO:] Creating autocorrelation plot")
    # Initialize figure
    fig, ax = plt.subplots(figsize=(18, 8)) 

    # Set font via FontProperties
    font = FontProperties(family='Times New Roman', size=25)

    # Plot autocorrelation
    autocor_plot = plot_acf(data["y"], title='', lags=n_lags, zero=True, ax=ax)

    # Set x ticks to every 12 hours
    ax.set_xticks(range(0, n_lags+1, 12))
    ax.set_xticklabels(range(0, n_lags+1, 12), fontproperties=font)  # Applying font properties to xtick labels
    ax.set_xlim(0, n_lags+1)

    # set y tricks to every 0.25
    ax.set_yticks([i/4 for i in range(-4, 5)])
    ax.set_yticklabels([i/4 for i in range(-4, 5)], fontproperties=font)  # Applying font properties to ytick labels

    # Modify tick parameters
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Setting labels with font properties
    ax.set_xlabel('Lag / Hour', fontproperties=font, labelpad=18)  
    ax.set_ylabel('Autocorrelation', fontproperties=font, labelpad=18) 

    # Remove whitespace and save the figure
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / file_name, dpi=300)


def plot_partial_autocorrelation(data, n_lags=24*7*2, file_name="norreport_1A_partial_autocorrelation.png", save_dir=pathlib.Path(__file__).parents[1] / "plots"):
    print("[INFO:] Creating partial autocorrelation plot")
    
    # initialize figure
    fig, ax = plt.subplots(figsize=(18, 8))  # Matched figure size to plot_autocorrelation

    # set font via FontProperties
    font = FontProperties(family='Times New Roman', size=25)

    # plot partial autocorrelation
    pacf_plot = plot_pacf(data["y"], lags=n_lags, zero=True, ax=ax, title='')

    # set x ticks to every 12 hours (for consistency)
    ax.set_xticks(range(0, n_lags+1, 12))
    ax.set_xticklabels(range(0, n_lags+1, 12), fontproperties=font)
    ax.set_xlim(0, n_lags)

    # set y ticks (optional customization)
    ax.set_yticks([i/4 for i in range(-4, 5)])
    ax.set_yticklabels([i/4 for i in range(-4, 5)], fontproperties=font) 

    # set y lims
    ax.set_ylim(-0.25, 1)

    # Modify tick parameters
    ax.tick_params(axis='both', which='major', labelsize=20)

    # setting labels with font properties
    ax.set_xlabel('Lag / Hour', fontproperties=font, labelpad=18)
    ax.set_ylabel('Partial Autocorrelation', fontproperties=font, labelpad=18)

    # remove whitespace and save the figure
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / file_name, dpi=300)

def main():
    # set paths
    path = pathlib.Path(__file__)
    data_dir = path.parents[2] / "data" / "clean_stops"
    plot_dir = path.parents[2] / "plots"

    # load data
    data = pd.read_csv(data_dir / "clean_1A_norreport.csv")

    # impute missing
    data = impute_missing(data, method='rolling', window=24)

    # plots
    plot_autocorrelation(data, n_lags=24*7*2, file_name="norreport_1A_autocorrelation.png", save_dir=plot_dir / "graphical_analysis")
    plot_partial_autocorrelation(data, n_lags=24*7+1, file_name="norreport_1A_partial_autocorrelation.png", save_dir=plot_dir)

if __name__ == "__main__":
    main()