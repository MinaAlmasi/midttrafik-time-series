import pathlib
import pandas as pd
from data_utils import read_data

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

    # get only one direction (Kolt - Vejlby / Skejbyparken)
    retningnavn = "Kolt - Vejlby / Skejbyparken"
    print(f"[INFO]: Filter on direction {retningnavn}")
    df_1A = df_1A[df_1A["retningnavn"] == retningnavn]

    # save 1A from_Kolt
    df_1A.to_csv(path.parents[1] / "data" / "1A_from_kolt.csv", index=False)

    # save only stop number = 751301201
    stopnumber = 751301201
    print(f"[INFO]: Filter on stop number {stopnumber}")
    norreport_1A = df_1A[df_1A["stopnumber"] == stopnumber]

    # save as CSV 
    norreport_1A.to_csv(path.parents[1] / "data" / "1A_norreport.csv", index=False)

if __name__ == "__main__":
    main()