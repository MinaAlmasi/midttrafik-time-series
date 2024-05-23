import pathlib
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import sys
import multiprocessing as mp

sys.path.append(str(pathlib.Path(__file__).parents[1]))
from data_utils import split_rolling_origin, impute_missing

def evaluate_model(model_fit, df_test, save_path=None):
    '''
    Evaluate sarima model on test data
    '''
    # make predictions
    y_pred = model_fit.predict(start=df_test.index[0], end=df_test.index[-1])

    # calculate metrics
    mae_test = np.mean(np.abs(df_test['y'] - y_pred))
    rmse_test = np.sqrt(np.mean((df_test['y'] - y_pred)**2))

    if save_path is not None:
        df_forecast = pd.DataFrame({'ds': df_test['ds'], 'y': y_pred})
        df_forecast.to_csv(save_path, index=False)
    
    return mae_test, rmse_test, y_pred

def stop_pipeline(df, forecast_path, results_path, save_test_only=False):
    # impute missing values
    df = impute_missing(df, method='rolling', window=24)

    # data setup
    gap = 24 # gap between train and val and train and test
    steps = 4 # how fast rolling origin moves
    test_size = 36 # hours 
    min_train_size = 24*7 # i.e. 1 week 

    # split the data
    _, _, test_inds = split_rolling_origin(df['ds'], gap=gap, test_size=test_size, steps=steps, min_train_size=min_train_size) 

    # (best model found by auto_arima.py)
    order = (2, 1, 0) # p, d, q
    seasonal_order = (2, 1, 0, 24) # P, D, Q, m (or s)

    # add gap to the test indices
    dropped_inds = df.index[-(len(test_inds) + gap-1):].tolist()

    # get full train data (all data except test data and gap)
    df_full_train = df.copy().drop(index=dropped_inds, axis=0)
    df_test = df.iloc[test_inds]

    # refit model 
    model = SARIMAX(df_full_train['y'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()

    # evaluate model
    mae_test, rmse_test, y_pred = evaluate_model(model_fit, df_test, save_path=forecast_path / 'sarima_forecast.csv')
    print(f"MAE test: {mae_test}")
    print(f"RMSE test: {rmse_test}")

    if save_test_only:
        # save results with the test results only
        results = pd.DataFrame({'mae_test': [mae_test], 'rmse_test': [rmse_test]})
        results.to_csv(results_path / 'sarima_results.csv', index=False)
    
    else:
        # save results with the train-val results 
        train_val_df = pd.read_csv(results_path / 'sarima_train_val.csv')

        # add the test results
        train_val_df['mae_test'] = mae_test
        train_val_df['rmse_test'] = rmse_test

        train_val_df.to_csv(results_path / 'sarima_results.csv', index=False)
        

def main(): 
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "data" / "clean_stops"
    results_path = path.parents[2] / "results"

    # load the nørreport data
    df = pd.read_csv(data_path / 'clean_1A_norreport.csv')

        
    # run on norreport
    norreport_results_path = results_path / "norreport"
    norreport_forecast_path = norreport_results_path / "forecasts"
    norreport_results_path.mkdir(parents=True, exist_ok=True)
    norreport_forecast_path.mkdir(parents=True, exist_ok=True)

    stop_pipeline(df, norreport_forecast_path, norreport_results_path, save_test_only=False)

    # run on all stops
    for stop in data_path.iterdir():
        if stop.name == 'clean_1A_norreport.csv': # skip norreport as it has its seperate pipeline
            continue
        
        # load the data
        df = pd.read_csv(stop)

        # stop name (remove clean_1A_ and .csv) from name
        stop_name = stop.stem[9:]

        # results path 
        other_results_path = results_path / "other_stops" / stop_name
        other_results_path.mkdir(parents=True, exist_ok=True)

        # run the pipeline
        stop_pipeline(df, other_results_path, other_results_path, save_test_only=True)
    


    


if __name__ == "__main__":
    main()