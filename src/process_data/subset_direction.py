import pathlib
import pandas as pd  
from tqdm import tqdm

import sys 
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from data_utils import read_data

stops = {"norreport": 751301201, 
         "kolt_osterparken": 751473002,
         "hasselager_alle": 751422802,
         "park_alle": 751001502,
         "vejlby_centervej_grenåvej": 751100201
         }

def main(): 
    # define path
    path = pathlib.Path(__file__)

    # data path 
    print("[INFO:] Loading the raw data")
    data_path = path.parents[2] / "raw_data" / "1A og 2A 2021 til nu.csv"

    # read data
    df = read_data(data_path, chunksize=10000)
    df = df.sort_values(by="date")

    # get only 1A
    print("[INFO:] Filter on line 1A")
    df_1A = df[df["line"] == "1A"]

    # delete df from memory
    del df
    
    ## DIRECTION ## 
    # get only one direction (Kolt - Vejlby / Skejbyparken)
    retningnavn = "Kolt - Vejlby / Skejbyparken"
    print(f"[INFO]: Filter on direction {retningnavn}")
    df_1A = df_1A[df_1A["retningnavn"] == retningnavn]
    
    # remove busses where the 
    print("[INFO:] Create bus id column")
    df_1A['bus_id'] = df_1A['date'] + "_" + df_1A['ITCS_number'].astype(str) + "_" + df_1A['turstarttid']

    # loop over unique bus ids
    threshold = 0.8
    remove_bus_ids = []

    tqdm.pandas()
    
    print("[INFO:] Removing busses based on missing counts")
    # Calculate zero percentage for each bus id
    zero_percentage = df_1A.groupby('bus_id')['Cumulative'].progress_apply(lambda x: (x == 0).mean())

    # Filter out bus ids with zero percentage above threshold
    remove_bus_ids = zero_percentage[zero_percentage > threshold].index.tolist()

    # Remove bus ids from data
    df_1A_filtered = df_1A[~df_1A['bus_id'].isin(remove_bus_ids)]

    print(df_1A_filtered.head())

    # save filtered 1a kolt 
    print("[INFO:] Saving filtered 1A from Kolt to CSV")
    df_1A_filtered.to_csv(path.parents[2] / "data" / "1A_from_kolt_filtered.csv", index=False)

if __name__ == "__main__":
    main()