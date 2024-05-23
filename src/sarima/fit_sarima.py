import pathlib
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import sys
import multiprocessing as mp

sys.path.append(str(pathlib.Path(__file__).parents[1]))
from data_utils import split_rolling_origin, impute_missing


def cv_single_fold(df, train_ind, val_ind, order:tuple, seasonal_order:tuple): 
    '''
    Process a single fold of cross-validation using provided parameters.
    '''
    df_train = df.iloc[train_ind]
    df_val = df.iloc[val_ind]

    # fit the model
    model = SARIMAX(df_train['y'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()

    # make predictions
    y_pred = model_fit.predict(start=val_ind[0], end=val_ind[-1])

    # calculate metrics
    mae_train = np.mean(np.abs(df_train['y'] - model_fit.fittedvalues))
    mae_val = np.mean(np.abs(df_val['y'] - y_pred))
    
    rmse_train = np.sqrt(np.mean((df_train['y'] - model_fit.fittedvalues)**2))
    rmse_val = np.sqrt(np.mean((df_val['y'] - y_pred)**2))

    return mae_train, mae_val, rmse_train, rmse_val

def get_mean_sd(vals:list):
    '''
    calculate the mean and standard deviation of a list of values.
    '''
    mean = sum(vals) / len(vals)
    sd = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5

    return mean, sd

def cross_validate(df, train_inds, val_inds, order:tuple, seasonal_order:tuple, n_cores:int=1, save_dir=None): 
    '''
    Cross validate SARIMA model
    '''
    # prep args for mp
    processes = []
    for (train_ind, val_ind) in zip(train_inds.values(), val_inds.values()):
        args = (df, train_ind, val_ind, order, seasonal_order)
        processes.append(args)
    
    with mp.Pool(n_cores) as pool:
        mae_trains, mae_vals, rmse_trains, rmse_vals = zip(*pool.starmap(cv_single_fold, processes)) # unpack results

    # calculate mean and standard deviation of the results
    metrics = {}
    metrics["model"] = f"SARIMA_{order}_{seasonal_order}"
    metrics["mean_mae_train"], metrics["sd_mae_train"] = get_mean_sd(mae_trains)
    metrics["mean_mae_val"], metrics["sd_mae_val"] = get_mean_sd(mae_vals)
    metrics["mean_rmse_train"], metrics["sd_rmse_train"] = get_mean_sd(rmse_trains)
    metrics["mean_rmse_val"], metrics["sd_rmse_val"] = get_mean_sd(rmse_vals)

    # make into df 
    metrics = pd.DataFrame(metrics, index=[0])

    # save to csv 
    if save_dir: 
        metrics.to_csv(save_dir / f"sarima_train_val.csv", index=False)

    return metrics

def main():
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "data" / "clean_stops"

    # load data
    df = pd.read_csv(data_path / 'clean_1A_norreport.csv')

    # impute missing values
    df = impute_missing(df, method='rolling', window=24)

    # data setup
    gap = 24 # gap between train and val and train and test
    steps = 4 # how fast rolling origin moves
    test_size = 36 # hours 
    min_train_size = 24*7 # i.e. 1 week 

    # split the data
    train_inds, val_inds, _ = split_rolling_origin(df['ds'], gap=gap, test_size=test_size, steps=steps, min_train_size=min_train_size) # no need for test_inds here!

    # (best model found by auto_arima.py)
    order = (5, 1, 0) # p, d, q
    seasonal_order = (2, 1, 0, 24) # P, D, Q, m

    # cross validate
    n_cores = mp.cpu_count() - 1
    metrics = cross_validate(df, train_inds, val_inds, order, seasonal_order, n_cores=n_cores, save_dir=path.parents[2] / "results" / "norreport")

if __name__ == "__main__":
    main()