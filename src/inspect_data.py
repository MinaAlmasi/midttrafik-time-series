import pathlib
import pandas as pd
from datetime import datetime, timedelta

def round_up_half_hour(time_str):
    time_obj = datetime.strptime(time_str, '%H:%M:%S')
    # Calculate the number of minutes to round up
    minutes = time_obj.minute
    if minutes == 0 or minutes == 30:
        rounded_minutes = minutes
    elif minutes < 30:
        rounded_minutes = 30
    else:
        time_obj += timedelta(hours=1)
        rounded_minutes = 0
    # Return the time rounded up to the nearest half-hour
    return time_obj.replace(minute=rounded_minutes, second=0).strftime('%H:%M:%S')

def add_missing_time_intervals(df):
    # ensure the 'ds' column is datetime
    df['ds'] = pd.to_datetime(df['ds'])

    # set 'ds' as the index
    df.set_index('ds', inplace=True)

    # create a date range with 30-minute intervals covering the entire range
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='30T')

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

    # add aggregate
    norreport_1A['time_interval'] = norreport_1A['scheduledarrivetime'].apply(round_up_half_hour)

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

    # save 
    norreport_1A.to_csv('data/processed_1A_norreport.csv', index=False)

    # subset to only first 1000 rows
    norreport_1A = norreport_1A.head(10000)

    # plot the timeseries for Cumulative
    ts_plot = norreport_1A.plot(x="ds", y="y", figsize=(25,10))

    # save to plot dir
    ts_plot.get_figure().savefig(plot_dir / "norreport_1A_ts.png")


if __name__ == "__main__":
    main()