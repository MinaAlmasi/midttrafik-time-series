import pathlib
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

def main():
    # set paths
    path = pathlib.Path(__file__)
    data_dir = path.parents[1] / "data"
    plot_dir = path.parents[1] / "plots"

    # load data
    data = pd.read_csv(data_dir / "processed_1A_norreport.csv")

    # create autocorrelation plot
    print("[INFO:] Creating autocorrelation plot")
    autocor_plot = plot_acf(data["y"], lags=50)

    # save plot
    autocor_plot.savefig(plot_dir / "autocorrelation.png")


if __name__ == "__main__":
    main()