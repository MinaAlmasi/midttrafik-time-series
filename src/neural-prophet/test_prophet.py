import pathlib
from neuralprophet import NeuralProphet
import pandas as pd

import sys 
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from data_utils import split_rolling_origin, impute_missing

def refit_model(df_full_train:pd.DataFrame, params:dict, freq="1h"):
    '''
    Refit the model with the best parameters.
    '''
    # combine init and default parameters
    default_params = {
        "yearly_seasonality": False, # set yearly seasonality also to avoid model having to check for it 
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'newer_samples_start': 0.5
    }
    all_params = {**default_params, **params}

    # init model with parameters from the initialization grid
    model = NeuralProphet(**all_params)

    # fit the model with the best parameters
    model.fit(df_full_train, freq=freq)

    return model

def evaluate_model(df_full_train, dropped_inds, df_test, model, gap=24): 
    '''
    Evaluate the model on the test set.
    '''
    # test to get the metrics
    metrics = model.test(df_test)

    # set period to forecast
    period = (len(df_test) + gap) - 1

    # make future df
    for i in range(period):
        # make a future df with the next ds
        future = model.make_future_dataframe(df_full_train, n_historic_predictions=len(df_full_train))

        # get prediction for the new ds
        preds = model.predict(future)

        # get the yhat1 of the last prediction
        pred = preds['yhat1'].iloc[-1]
        pred_date = preds['ds'].iloc[-1]

        # turn into a df
        pred_df = pd.DataFrame({'ds': [pred_date], 'y': [pred]})

        # concat with the full train
        df_full_train = pd.concat([df_full_train, pred_df], axis=0)

    # subset forecast to only include the test set
    forecast = df_full_train.iloc[-len(df_test):]

    return metrics, forecast

def stop_pipeline(stop_df, np_results_df, forecast_path, results_path, top_n_models=5, save_test_only=False):
    # impute missing values
    stop_df = impute_missing(stop_df, method='rolling', window=24)

    # identify best models
    best_models = np_results_df.sort_values(by="mean_rmse_val").iloc[:top_n_models]

    # get the n best parameters from "model" column as a dict
    top_params = best_models['model'].apply(eval).tolist()
    top_model_numbers = best_models['model_number'].tolist()

    # specify the data setup
    gap = 24 # gap between train and val and train and test
    steps = 4 # how fast rolling origin moves
    test_size = 36 # hours 
    min_train_size = 24*7 # i.e. 1 week 

    # load the test data
    _, _, test_inds = split_rolling_origin(stop_df['ds'], gap=gap, test_size=test_size, steps=steps, min_train_size=min_train_size)

    # add gap to the test indices
    dropped_inds = stop_df.index[-(len(test_inds) + gap-1):].tolist()

    # get full train data (all data except test data and gap)
    df_full_train = stop_df.copy().drop(index=dropped_inds, axis=0)
    df_test = stop_df.iloc[test_inds]

    mae_values = []
    rmse_values = []

    # iterate and refit the models
    for (params, model_number) in zip(top_params, top_model_numbers):
        # fit model
        model = refit_model(df_full_train, params, freq="1h")
       
        # evaluate
        print(f"[INFO:] Evaluating the model on the test set... on {len(df_test)} samples.")
        metrics, forecast = evaluate_model(df_full_train, dropped_inds, df_test, model, gap=gap)

        # save the forecast named after the model number
        forecast.to_csv(forecast_path / f'np{model_number}_forecast.csv', index=False)

        # get the MAE and RMSE from the results (they are called val but they are from the test set)
        mae_test = metrics['MAE_val'].values[0]
        rmse_test = metrics['RMSE_val'].values[0]

        mae_values.append(mae_test)
        rmse_values.append(rmse_test)

    # add the results to the best_models dataframe
    best_models['mae_test'] = mae_values
    best_models['rmse_test'] = rmse_values  
    
    if save_test_only: 
        # drop all cols but mae_test and rmse_test
        best_models = best_models[['model_number', 'model', 'mae_test', 'rmse_test']]

    # save the results
    best_models.to_csv(results_path / 'np_results.csv', index=False)

def main(): 
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "data" / "clean_stops"
    results_path = path.parents[2] / "results" 

    # load the results file from the grid search
    neuralprophet_path = results_path / "norreport" /  "neural-prophet-grid-search" # grid search only done on norreport
    np_results_df = pd.read_csv(neuralprophet_path / 'np_gridsearch_20240520_210018.csv')
    
    # load nørreport data
    df = pd.read_csv(data_path / 'clean_1A_norreport.csv')

    # run on norreport
    norreport_results_path = results_path / "norreport"
    norreport_forecast_path = norreport_results_path / "forecasts"
    norreport_results_path.mkdir(parents=True, exist_ok=True)
    norreport_forecast_path.mkdir(parents=True, exist_ok=True)
    stop_pipeline(df, np_results_df, norreport_forecast_path, norreport_results_path, top_n_models=5, save_test_only=False)

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
        stop_pipeline(df, np_results_df, other_results_path, other_results_path, top_n_models=5, save_test_only=True)

if __name__ == "__main__":
    main()