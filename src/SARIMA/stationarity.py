import pandas as pd
import numpy as np
import pathlib
from statsmodels.tsa.stattools import kpss

def kpss_test(timeseries):
    '''
    Perform KPSS to test for stationarity: 
        Null Hypothesis: The process is trend stationary.
        Alternate Hypothesis: The series has a unit root (series is not stationary)

    Function from https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
    '''
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)


def main():
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / 'data'

    # load the data
    df = pd.read_csv(data_path / 'processed_1A_norreport.csv')

    # remove test data (last 36 datapoints) and the gap before it (last 24 datapoints)
    test_size = 36
    gap = 24
    df = df.iloc[:-(test_size+gap)]

    # check for stationarity
    print('KPSS test for stationarity')
    output = kpss_test(df['ds'])
    
    print(output)


if __name__ == '__main__':
    main()