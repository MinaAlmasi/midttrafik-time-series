import pathlib
import pandas as pd

def main():
    path = pathlib.Path(__file__)
    plot_dir = path.parents[1] / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    # read the data
    norreport_1A = pd.read_csv('data/1A_norreport.csv')
    print(norreport_1A.head())

    # bind date and scheduledarrivetime to a single datetime column
    norreport_1A['datetime'] = norreport_1A['date'] + " " + norreport_1A['scheduledarrivetime']

    # sort by datetime
    norreport_1A = norreport_1A.sort_values('datetime').reset_index(drop=True)

    # remove rows missing in actualarrivaltime
    norreport_1A = norreport_1A.dropna(subset=['actualarrivetime'])

    norreport_1A = norreport_1A.dropna(subset=['actualdeparturetime'])
    
    # remove duplicates 
    print(len(norreport_1A))
    norreport_1A = norreport_1A.drop_duplicates()
    print(len(norreport_1A))

    # save 
    norreport_1A.to_csv('data/processed_1A_norreport.csv', index=False)

    # subset to only first 1000 rows
    norreport_1A = norreport_1A.head(10000)

    # plot the timeseries for Cumulative
    ts_plot = norreport_1A.plot(x="datetime", y="Cumulative", figsize=(25,10))

    # save to plot dir
    ts_plot.get_figure().savefig(plot_dir / "norreport_1A_ts.png")


if __name__ == "__main__":
    main()