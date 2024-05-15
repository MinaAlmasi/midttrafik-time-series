import pathlib
import pandas as pd
from datetime import datetime, timedelta


def add_missing_time_intervals(df, freq='1h'):
    # ensure the 'ds' column is datetime
    df['ds'] = pd.to_datetime(df['ds'])

    # set 'ds' as the index
    df.set_index('ds', inplace=True)

    # create a date range with 1 hour frequency
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)

    # reindex the original DataFrame to the new index, filling missing values with 0
    df_full = df.reindex(full_range, fill_value=0)

    # reset index to make 'ds' a column again
    df_full.reset_index(inplace=True)
    df_full.rename(columns={'index': 'ds'}, inplace=True)

    return df_full

def main():
    path = pathlib.Path(__file__)
    plot_dir = path.parents[1] / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    # read the data
    norreport_1A = pd.read_csv('data/1A_norreport.csv')

    # remove all rows which has 2021 in date (formatted YYYY-MM-DD) using datetime (due to covid)
    norreport_1A['year'] = pd.to_datetime(norreport_1A['date'])
    norreport_1A = norreport_1A[norreport_1A['year'].dt.year != 2021]

    # remove rows missing in actualarrivaltime or actualdeparturetime
    norreport_1A = norreport_1A.dropna(subset=['actualarrivetime'])
    norreport_1A = norreport_1A.dropna(subset=['actualdeparturetime'])
    
    # remove duplicates 
    norreport_1A = norreport_1A.drop_duplicates()

    # keep only the 'hour' from scheduledtimearrive (HH:MM:SS)
    norreport_1A['time_interval'] = norreport_1A['scheduledarrivetime'].str.split(':').str[0]

    # subset to only relevant columns
    norreport_1A = norreport_1A[['date', 'time_interval', 'Cumulative']]

    # create ds column based on date and time_interval
    norreport_1A['ds'] = norreport_1A['date'] + ' ' + norreport_1A['time_interval']

    # drop date and time_interval columns
    norreport_1A = norreport_1A.drop(columns=['date', 'time_interval'])

    # rename cumulative to y
    norreport_1A = norreport_1A.rename(columns={'Cumulative': 'y'})

    # take the mean y for each ds
    norreport_1A = norreport_1A.groupby('ds').mean().reset_index()

    # add missing time intervals
    norreport_1A = add_missing_time_intervals(norreport_1A)

    print(f"Length of the dataset: {len(norreport_1A)}")

    # save 
    norreport_1A.to_csv('data/processed_1A_norreport.csv', index=False)

if __name__ == "__main__":
    main()