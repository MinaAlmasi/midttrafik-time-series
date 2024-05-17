import pathlib
from neuralprophet import NeuralProphet
import pandas as pd
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

def main(): 
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / "data"
    results_path = path.parents[1] / "results"
    neuralprophet_path = results_path / "neuralprophet"

    # load data and impute missing
    df = pd.read_csv(data_path / 'processed_1A_norreport.csv')
    df = impute_missing(df, method='rolling', window=24)

    # load the results file from the grid search
    np_results = pd.read_csv(neuralprophet_path / 'np_gridsearch_20240516_160815.csv')

    # select top n models based on validation rmse
    top_n_models = 5
    best_models = np_results.sort_values(by="mean_rmse_val").iloc[:top_n_models]

    # get the n best parameters from "model" column as a dict
    top_params = best_models['model'].apply(eval).tolist()
    top_model_numbers = best_models['model_number'].tolist()

    # specify the data setup
    gap = 24 # gap between train and val and train and test
    steps = 4 # how fast rolling origin moves
    test_size = 36 # hours 
    min_train_size = 24*7 # i.e. 1 week 

    # load the test data
    _, _, test_inds = split_rolling_origin(df['ds'], gap=gap, test_size=test_size, steps=steps, min_train_size=min_train_size)

    # add gap to the test indices
    dropped_inds = df.index[-(len(test_inds) + gap-1):].tolist()

    # get full train data (all data except test data and gap)
    df_full_train = df.copy().drop(index=dropped_inds, axis=0)
    df_test = df.iloc[test_inds]

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
        forecast.to_csv(results_path / "forecasts" / f'np{model_number}_forecast.csv', index=False)

        # get the MAE and RMSE from the results (they are called val but they are from the test set)
        mae_test = metrics['MAE_val'].values[0]
        rmse_test = metrics['RMSE_val'].values[0]

        mae_values.append(mae_test)
        rmse_values.append(rmse_test)

    # add the results to the best_models dataframe
    best_models['MAE_test'] = mae_values
    best_models['RMSE_test'] = rmse_values

    # save the results
    best_models.to_csv(results_path / 'np_results.csv', index=False)

if __name__ == "__main__":
    main()