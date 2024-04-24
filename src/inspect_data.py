import pathlib
import pandas as pd
from tqdm import tqdm

def read_data(data_path, chunksize:int=None, sep=";"):
    """
    Read data from path. 
    """
    # read data
    if chunksize is None:
        df = pd.read_csv(data_path, sep=sep)

    else:
        reader = pd.read_csv(data_path, sep=sep, chunksize=chunksize)
        dfs = [chunk for chunk in tqdm(reader)]
        df = pd.concat(dfs, ignore_index=True)

    return df    

def main(): 
    # define path
    path = pathlib.Path(__file__)

    # data path 
    data_path = path.parents[2] / "data" / "1A og 2A 2021 til nu.csv"

    # read data
    df = read_data(data_path, chunksize=100000)

    # df 
    sorted_df = df.sort_values(by="date")

    # get only 1A
    df_1A = sorted_df[sorted_df["line"] == "1A"]

    # add year col 
    df_1A["year"] = pd.to_datetime(df_1A["date"]).dt.year

    # get only 2021
    df_1A_2023 = df_1A[df_1A["year"] == 2023]
    print(len(df_1A_2023))

    # drop duplicates
    df_1A_2023 = df_1A_2023.drop_duplicates()
    print(len(df_1A_2023))

    print(df_1A_2023)


if __name__ == "__main__":
    main()