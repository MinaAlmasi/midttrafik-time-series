import pathlib
import pandas as pd
from data_utils import split_rolling_origin, impute_missing
import matplotlib.pyplot as plt

def get_splits(df, train_inds, val_inds, test_inds):
    '''
    get the train, val and test splits from df and splitted indices

    Args:
        df: data 
        train_inds, val_inds: dictionary with indices for train and val across folds
        test_inds: indices for a HOLDOUT test set
    
    Returns:
        df_trains, df_vals, df_test: dataframes with train, val and test splits
    '''
    df_trains = pd.DataFrame()
    df_vals = pd.DataFrame()
    for i, (train_ind, test_ind) in enumerate(zip(train_inds.values(), val_inds.values())):
        df_train = df.iloc[train_ind]
        df_val = df.iloc[test_ind]

        # create fold column as the first column
        df_train.insert(0, 'fold', i)
        df_val.insert(0, 'fold', i)

        # add to dataframe
        df_trains = pd.concat([df_trains, df_train], axis=0)
        df_vals = pd.concat([df_vals, df_val], axis=0)

    # create test df 
    df_test = df.iloc[test_inds]
    
    return df_trains, df_vals, df_test

def mean_model(df, df_vals, df_test):
    '''
    Mean model which always predicts the mean of the training set
    '''
    # remove the test indices from 
    df_train_val = df[~df.index.isin(df_test.index)] # df_train val WIRHOUT rolling origin 

    # calculate the mean
    mean = df_train_val['y'].mean()

    mae_values = {}
    rmse_values = {}
    for split, df in zip(["val", "test"], [df_vals, df_test]):
        df['y_pred'] = mean # add mean to pred col

        # compute mae 
        mae = abs(df['y'] - df['y_pred']).mean()
        mae = round(mae, 3)

        # compute the rmse
        rmse = (((df['y'] - df['y_pred'])**2).mean())**0.5
        rmse = round(rmse, 3)

        mae_values[split] = mae
        rmse_values[split] = rmse

    # get all forecasts for test set (ds and y_pred column in df_test)
    mm_test_forecast = df_test[['ds', 'y_pred']]

    # rename y pred to y
    mm_test_forecast.rename(columns={'y_pred': 'y'}, inplace=True)

    return mae_values, rmse_values, mm_test_forecast

def naive_model(df_trains, df_vals, df_test, gap=24):
    '''
    Naive model which always predicts last value in the training set (per fold)
    '''

    # for each fold, identify the last value in the training set
    for fold in df_trains['fold'].unique():
        # get the last value in the training set
        last_value = df_trains[df_trains['fold'] == fold].iloc[-1]['y']

        # add to the test set
        df_vals.loc[df_vals['fold'] == fold, 'y_pred'] = last_value

    # calculate MAE and MSE as columns
    df_vals['mae'] = abs(df_vals['y'] - df_vals['y_pred'])
    df_vals['mse'] = (df_vals['y'] - df_vals['y_pred'])**2

    # compute the mean MAE and RMSE
    val_mae = df_vals['mae'].mean()
    val_mae = round(val_mae, 3)

    val_rmse = (df_vals['mse'].mean())**0.5
    val_rmse = round(val_rmse, 3)

    # for the test set, we simply predict the last value in the training set overall
    last_value = df_trains.iloc[-1]['y']
    df_test['y_pred'] = last_value

    # calculate MAE
    df_test['mae'] = abs(df_test['y'] - df_test['y_pred'])
    test_mae = df_test['mae'].mean()
    test_mae = round(test_mae, 3)

    # calculate RMSE
    df_test['mse'] = (df_test['y'] - df_test['y_pred'])**2
    test_rmse = (df_test['mse'].mean())**0.5
    test_rmse = round(test_rmse, 3)

    # create a dict with the MAE values and one with RMSE values
    mae_values = {'val': val_mae, 'test': test_mae}
    rmse_values = {'val': val_rmse, 'test': test_rmse}
    
    # make forecast
    naive_test_forecast = df_test[['ds', 'y_pred']]
    naive_test_forecast.rename(columns={'y_pred': 'y'}, inplace=True) # rename for plotting

    return mae_values, rmse_values, naive_test_forecast

def weekly_naive_model(df, df_vals, df_test):
    '''
    Seasonal naive model which predicts the value from the same time one week ago
    '''
    # calculate how many indexes to go back to get the value from one week ago
    timesteps_back = 24 * 7

    for _, row in df_vals.iterrows():
        index = row.name
        index_week_ago = index - timesteps_back 
        value_week_ago = df.loc[index_week_ago, 'y']
        df_vals.loc[index, 'y_pred'] = value_week_ago

    # calculate MAE
    val_mae = abs(df_vals['y'] - df_vals['y_pred']).mean()
    val_mae = round(val_mae, 3)

    # calculate RMSE
    val_rmse = (((df_vals['y'] - df_vals['y_pred'])**2).mean())**0.5
    val_rmse = round(val_rmse, 3)

    # repeat the process for the test set
    for _, row in df_test.iterrows():
        index = row.name
        index_week_ago = index - timesteps_back
        value_week_ago = df.loc[index_week_ago, 'y']
        df_test.loc[index, 'y_pred'] = value_week_ago

    # calculate MAE & RMSE
    test_mae = abs(df_test['y'] - df_test['y_pred']).mean()
    test_mae = round(test_mae, 3)
    
    test_rmse = (((df_test['y'] - df_test['y_pred'])**2).mean())**0.5
    test_rmse = round(test_rmse, 3)

    # add to one dict
    mae_values = {'val': val_mae, 'test': test_mae}
    rmse_values = {'val': val_rmse, 'test': test_rmse}

    # identify as test forecasts
    weekly_naive_test_forecasts = df_test[['ds', 'y_pred']]

    # rename y pred to y
    weekly_naive_test_forecasts.rename(columns={'y_pred': 'y'}, inplace=True)
    

    return mae_values, rmse_values, weekly_naive_test_forecasts

def norreport_pipeline(df, forecast_path, results_path):
    '''
    Analysis pipeline for the norreport stop which is our main focus. 
    Function will perform cross validation and test the models on the test set.
    '''
    # impute missing values
    df = impute_missing(df, method='rolling', window=24)

    # split the data
    gap = 24
    train_inds, val_inds, test_inds = split_rolling_origin(df['ds'], gap=gap, test_size=36, steps=4, min_train_size=24*7)
    df_trains, df_vals, df_test = get_splits(df, train_inds, val_inds, test_inds)
    
    # fit the mean model
    mean_model_mae, mean_model_rmse, mean_model_test_forecast = mean_model(df, df_vals, df_test.copy())

    # fit the naive model
    naive_mae, naive_rmse, naive_forecast = naive_model(df_trains, df_vals, df_test.copy())
    
    # fit the weekly naive model
    weekly_naive_mae, weekly_naive_rmse, weekly_naive_forecast = weekly_naive_model(df, df_vals, df_test.copy())

    # write the test forecasts to csv
    mean_model_test_forecast.to_csv(forecast_path / 'mm_forecast.csv', index=False)
    naive_forecast.to_csv(forecast_path / 'naive_forecast.csv', index=False)
    weekly_naive_forecast.to_csv(forecast_path / 'weekly_naive_forecast.csv', index=False)
    
    # print the results
    print("\n")
    print(f'Mean model. MAE: {mean_model_mae}, RMSE: {mean_model_rmse}')
    print(f'Naive model. MAE: {naive_mae}, RMSE: {naive_rmse}')
    print(f'Weekly naive model MAE: {weekly_naive_mae}, RMSE: {weekly_naive_rmse}')

    # save the results
    results = pd.DataFrame({"model": ['Mean', 'Naive', 'Weekly Naive'],
                            'mean_mae_val': [mean_model_mae['val'], naive_mae['val'], weekly_naive_mae['val']],
                            'mae_test': [mean_model_mae['test'], naive_mae['test'], weekly_naive_mae['test']],
                            'mean_rmse_val': [mean_model_rmse['val'], naive_rmse['val'], weekly_naive_rmse['val']],
                            'rmse_test': [mean_model_rmse['test'], naive_rmse['test'], weekly_naive_rmse['test']]})
    
    results.to_csv(results_path / 'baseline_results.csv', index=False)

def main(): 
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / 'data' / "clean_stops"
    results_path = path.parents[1] / 'results'
    forecast_path = results_path / 'forecasts'

    # load the data
    df = pd.read_csv(data_path / 'clean_1A_norreport.csv')

    # run the pipeline
    norreport_pipeline(df, forecast_path, results_path)

    # run other stops 
    ## coming soon


if __name__ == "__main__":
    main()