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
        'newer_samples_start': 0.5,
    }
    all_params = {**default_params, **params}

    # init model with parameters from the initialization grid
    model = NeuralProphet(**all_params)

    # fit the model with the best parameters
    model.fit(df_full_train, freq=freq)

    return model

def evaluate_model(df_test, model): 
    '''
    Evaluate the model on the test set.
    '''
    metrics = model.test(df_test)

    return metrics

def main(): 
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / "data"
    results_path = path.parents[1] / "results"
    neuralprophet_path = results_path / "neuralprophet"

    # load data and impute missing
    df = pd.read_csv(data_path / 'processed_1A_norreport.csv')
    df = impute_missing(df, method='rolling', window=24)

    # print all na observations
    print(df['y'].isna().sum())

    # load the results file from the grid search
    np_results = pd.read_csv(neuralprophet_path / 'np_gridsearch_20240516_160815.csv')

    # select top n models based on validation rmse
    top_n_models = 1
    best_models = np_results.sort_values(by="mean_rmse_val").iloc[:top_n_models]

    # get the n best parameters from "model" column as a dict
    top_n_params = best_models['model'].apply(eval).tolist()

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
    df_full_train = df.drop(index=dropped_inds, axis=0)
    df_test = df.iloc[test_inds]

    # set new columns for best models
    best_models["MAE_test"] = None
    best_models["RMSE_test"] = None

    # iterate and refit the models
    for i, params in enumerate(top_n_params):
        print(f"[INFO:] Refitting model {i+1} of {top_n_models}...")

        # fit model
        model = refit_model(df_full_train, params, freq="1h")
       
        # evaluate
        print(f"[INFO:] Evaluating the model on the test set... on {len(df_test)} samples.")
        metrics = evaluate_model(df_test, model)

        # get the MAE and RMSE from the results (they are called val but they are from the test set)
        mae_test = metrics['MAE_val'].values[0]
        rmse_test = metrics['RMSE_val'].values[0]

        # add to best models df 
        best_models.loc[i, "mae_test"] = mae_test
        best_models.loc[i, "rmse_test"] = rmse_test        

    # save the results
    best_models.to_csv(results_path / 'np_results.csv', index=False)

if __name__ == "__main__":
    main()