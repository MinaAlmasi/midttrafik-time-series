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
    kolt_1a_data['bus_id'] = kolt_1a_data['date'] + "_" + str(kolt_1a_data['ITCS_number']) + "_" + kolt_1a_data['turstarttid']

    # loop over unique bus ids
    unique_bus_ids = kolt_1a_data['bus_id'].unique()
    threshold = 0.8
    remove_bus_ids = []

    for bus_id in tqdm(unique_bus_ids, total=len(unique_bus_ids)):
        # select bus id
        bus = kolt_1a_data[kolt_1a_data['bus_id'] == bus_id]

        # calculate how many percent of "Cumulative" equals zero
        zero_percentage = (bus['Cumulative'] == 0).sum() / len(bus['Cumulative'])

        # if percentage is above threshold, add id to remove list
        if zero_percentage > threshold:
            remove_bus_ids.append(bus_id)

    # remove bus ids from data
    kolt_1a_data_filtered = kolt_1a_data[~kolt_1a_data['bus_id'].isin(remove_bus_ids)]

    # print len of both
    print(f"Original data: {len(kolt_1a_data)}")
    print(f"Filtered data: {len(kolt_1a_data_filtered)}")

    # save to csv
    kolt_1a_data_filtered.to_csv(path.parents[1] / "data" / "kolt_1a_data_filtered.csv", index=False)

    
    # select date (2022-07-20) and ITCS_number (299)
    #selected_bus = kolt_1a_data[(kolt_1a_data["date"] == "2022-07-20") & (kolt_1a_data["ITCS_number"] == 299)]

    # save to csv
    #selected_bus.to_csv(path.parents[2] / "data" / "selected_bus.csv", index=False)
    



    
if __name__ == "__main__":
    main()