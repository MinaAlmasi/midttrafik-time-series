import pathlib
import pandas as pd  
from tqdm import tqdm

import sys 
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from data_utils import read_data

def main(): 
    # define path
    path = pathlib.Path(__file__)

    # data path 
    print("[INFO:] Loading the raw data")
    data_path = path.parents[2] / "data" / "1A_from_kolt_filtered.csv"

    # read data
    df = read_data(data_path, chunksize=10000, sep=",")

    # define the stops and their number
    stops = {"norreport": 751301201, 
         "kolt_osterparken": 751473002,
         "hasselager_alle": 751422802,
         "park_alle": 751001502,
         "vejlby_centervej": 751100203
         }
    
    # create the raw stops folder
    raw_stops_path = path.parents[2] / "data" / "raw_stops"
    raw_stops_path.mkdir(parents=True, exist_ok=True)

    # iterate over stops
    for stop_name, stop_number in stops.items(): 
        print(f"[INFO]: Filter on stop: {stop_name} ({stop_number})")
        
        # filter according to single stop
        df_filtered = df[df["stopnumber"] == stop_number]

        # save as CSV 
        df_filtered.to_csv(raw_stops_path / f"1A_{stop_name}.csv", index=False)


if __name__ == "__main__":
    main()