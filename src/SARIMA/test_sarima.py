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

def main(): 
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "data"
    results_path = path.parents[2] / "results" / "forecasts"

    # load data
    df = pd.read_csv(data_path / 'processed_1A_norreport.csv')

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
    seasonal_order = (2, 0, 0, 24) # P, D, Q, m

    # add gap to the test indices
    dropped_inds = df.index[-(len(test_inds) + gap-1):].tolist()

    # get full train data (all data except test data and gap)
    df_full_train = df.copy().drop(index=dropped_inds, axis=0)
    df_test = df.iloc[test_inds]

    # refit model 
    model = SARIMAX(df_full_train['y'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()

    # evaluate model
    mae_test, rmse_test, y_pred = evaluate_model(model_fit, df_test, save_path=results_path / 'sarima_forecast.csv')
    print(f"MAE test: {mae_test}")
    print(f"RMSE test: {rmse_test}")


if __name__ == "__main__":
    main()