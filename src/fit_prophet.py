import pathlib
import pandas as pd 
from neuralprophet import NeuralProphet, set_log_level
from sklearn.model_selection import TimeSeriesSplit

def split_timeseries_data(X, n_splits:int=100, gap:int=48, test_size:int=48):
    '''
    function to split the data into n_splits using TimeSeriesSplit
    '''
    # set the time series split
    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap,
        max_train_size=None,
        test_size=test_size
    )
    
    # split the data
    data_generator = tscv.split(X)

    return data_generator
    
def cross_validate(df, data_generator, freq="30min"):
    '''
    function to cross validate the model
    '''
    mae_vals = []
    for i, (train_ind, test_ind) in enumerate(data_generator):
        print(f"Fold {i+1}\n")
        
        # split the data on the indices
        df_train = df.iloc[train_ind]
        df_test = df.iloc[test_ind]

        # init model
        model = NeuralProphet()

        # fit the model
        model.fit(df_train, freq=freq)

        # predict
        metrics = model.test(df_test)

        # extract MAE_val
        mae_val = metrics['MAE_val'].values[0]
        mae_vals.append(mae_val)
    
    # calculate mean MAE and sd
    mean_mae = sum(mae_vals) / len(mae_vals)
    sd_mae = (sum([(x - mean_mae)**2 for x in mae_vals]) / len(mae_vals))**0.5 # double check formula

    return mean_mae, sd_mae

def main(): 
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / 'data'

    # load the data
    df = pd.read_csv(data_path / 'processed_1A_norreport.csv')

    # split the data
    generator = split_timeseries_data(df['ds'], n_splits=10, gap=48, test_size=48)

    # cross validate
    mean_mae, sd_mae = cross_validate(df, generator)

    print(f"Mean MAE: {mean_mae}")
    print(f"SD MAE: {sd_mae}")

if __name__ == "__main__":
    main()