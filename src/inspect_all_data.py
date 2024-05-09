import pathlib
import pandas as pd
from tqdm import tqdm
from data_utils import read_data


def main(): 
    # define path
    path = pathlib.Path(__file__)

    # data path 
    kolt_1a_path = path.parents[1] / "data" / "1A_from_kolt.csv"

    # read data
    kolt_1a_data = read_data(kolt_1a_path, chunksize=10000, sep=",")

    # bus id col 
    print("[INFO:] Create bus id column")
    kolt_1a_data['bus_id'] = kolt_1a_data['date'] + "_" + str(kolt_1a_data['ITCS_number']) + "_" + kolt_1a_data['turstarttid']

    # loop over unique bus ids
    threshold = 0.8
    remove_bus_ids = []

    tqdm.pandas()

    # Calculate zero percentage for each bus id
    zero_percentage = kolt_1a_data.groupby('bus_id')['Cumulative'].progress_apply(lambda x: (x == 0).mean())

    # Filter out bus ids with zero percentage above threshold
    remove_bus_ids = zero_percentage[zero_percentage > threshold].index.tolist()

    # Remove bus ids from data
    kolt_1a_data_filtered = kolt_1a_data[~kolt_1a_data['bus_id'].isin(remove_bus_ids)]

    # print len of both
    print(f"Original data: {len(kolt_1a_data)}")
    print(f"Filtered data: {len(kolt_1a_data_filtered)}")

    # save to csv
    print("[INFO:] Save filtered data")
    kolt_1a_data_filtered.to_csv(path.parents[1] / "data" / "kolt_1a_data_filtered.csv", index=False)

    
    # select date (2022-07-20) and ITCS_number (299)
    #selected_bus = kolt_1a_data[(kolt_1a_data["date"] == "2022-07-20") & (kolt_1a_data["ITCS_number"] == 299)]

    # save to csv
    #selected_bus.to_csv(path.parents[2] / "data" / "selected_bus.csv", index=False)
    



    
if __name__ == "__main__":
    main()