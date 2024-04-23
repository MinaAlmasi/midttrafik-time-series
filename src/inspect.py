import pathlib
import pandas as pd

def main(): 
    # define path
    path = pathlib.Path(__file__)

    # data path 
    data_path = path.parents[2] / "data" / "1A og 2A 2021 til nu top 1000.csv"

    # read 
    df = pd.read_csv(data_path, sep=";")

    print(df)

if __name__ == "__main__":
    main()