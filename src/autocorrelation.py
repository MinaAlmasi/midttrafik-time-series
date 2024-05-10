import pathlib
import pandas as pd
from datetime import datetime, timedelta

def main():
    # set paths
    path = pathlib.Path(__file__)
    data_dir = path.parents[1] / "data"

    # load data
    data = pd.read_csv(data_dir / "processed_1A_norreport.csv")

    #

if __name__ == "__main__":
    main()