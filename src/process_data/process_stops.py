import pathlib
import pandas as pd
from datetime import datetime, timedelta


def add_missing_time_intervals(df, freq='1h'):
    '''
    Reformat data to have a row for each hour in the dataset.
    '''
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


def add_missing_dates(df, missing_dates:list):
    '''
    Identifies the missing dates in DF and replaces their values with NA
    '''
    # convert 'ds' to datetime
    df['ds'] = pd.to_datetime(df['ds'])

    # remove time, only keep the date
    df['date'] = df['ds'].dt.date

    # change date to string type
    df['date'] = df['date'].astype(str)

    # for all 'date' that are in missing_dates, replace 'y' with NA
    for date in missing_dates:
        df.loc[df['date'] == date, 'y'] = None
    
    # drop date
    df = df.drop(columns=['date'])

    return df

def process_stop(stop_df):
    # remove all rows which has 2021 in date (formatted YYYY-MM-DD) using datetime (due to covid)
    stop_df['year'] = pd.to_datetime(stop_df['date'])
    stop_df = stop_df[stop_df['year'].dt.year != 2021]

    # remove rows missing in actualarrivaltime or actualdeparturetime
    stop_df = stop_df.dropna(subset=['actualarrivetime'])
    stop_df = stop_df.dropna(subset=['actualdeparturetime'])
    
    # remove duplicates 
    stop_df = stop_df.drop_duplicates()

    # keep only the 'hour' from scheduledtimearrive (HH:MM:SS)
    stop_df['time_interval'] = stop_df['scheduledarrivetime'].str.split(':').str[0]

    # subset to only relevant columns
    stop_df = stop_df[['date', 'time_interval', 'Cumulative']]

    # create ds column based on date and time_interval
    stop_df['ds'] = stop_df['date'] + ' ' + stop_df['time_interval']

    # drop date and time_interval columns
    stop_df = stop_df.drop(columns=['date', 'time_interval'])

    # rename cumulative to y
    stop_df = stop_df.rename(columns={'Cumulative': 'y'})

    # take the mean y for each ds
    stop_df = stop_df.groupby('ds').mean().reset_index()

    # add missing time intervals
    stop_df = add_missing_time_intervals(stop_df)

    print(f"Length of the dataset: {len(stop_df)}")

    # add missing dates (these are the days where 1A did not drive due to massive snowfall)
    missing_dates = ["2024-01-03", "2024-01-04", "2024-01-06", "2024-01-07", "2024-01-08"] # note that the busses drove on 2024-01-05, but not the rest of the period
    stop_df = add_missing_dates(stop_df, missing_dates)

    return stop_df

def main():
    # set paths for directories
    path = pathlib.Path(__file__)
    plot_dir = path.parents[2] / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    data_dir = path.parents[2] / "data" / "raw_stops"

    # set path for clean stops
    clean_stops_path = path.parents[2] / "data" / "clean_stops"
    clean_stops_path.mkdir(exist_ok=True, parents=True)

    # iterate over all stops and process
    for stop in data_dir.iterdir():
        print(f"[INFO:] Processing {stop.stem}")

        # read the data
        stop_df = pd.read_csv(stop)

        # process the data
        stop_df = process_stop(stop_df)

        # save the processed data
        stop_df.to_csv(clean_stops_path / f"clean_{stop.stem}.csv", index=False)

if __name__ == "__main__":
    main()