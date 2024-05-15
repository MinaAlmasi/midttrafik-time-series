import pathlib
from neuralprophet import NeuralProphet
import pandas as pd
from data_utils import split_rolling_origin

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

def evaluate_model(df_test, model): 
    '''
    Evaluate the model on the test set.
    '''
    results = model.test(df_test)

    return results

def main(): 
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / "data"
    df = pd.read_csv(data_path / 'processed_1A_norreport.csv')

    # temp 
    params = {"n_lags": 2, "newer_samples_weight": 4, "newer_samples_start": 0.5, "ar_layers": [32, 16], "seasonality_reg": 0.5, "learning_rate": 0.15, "batch_size": 24, "epochs": 1}

    # data setup
    gap = 24 # gap between train and val and train and test
    steps = 4 # how fast rolling origin moves
    test_size = 36 # hours 
    min_train_size = 24*7 # i.e. 1 week 

    _, _, test_inds = split_rolling_origin(df['ds'], gap=gap, test_size=test_size, steps=steps, min_train_size=min_train_size)

    # add gap to the test indices
    dropped_inds = df.index[-(len(test_inds) + gap-1):].tolist()

    df_full_train = df.drop(index=dropped_inds, axis=0)
    df_test = df.iloc[test_inds]

    # refit 
    print(f"[INFO:] Refitting the model with the best parameters... on {len(df_full_train)} samples.")
    model = refit_model(df_full_train, params, freq="1h")

    print(df_full_train.tail())

    # evaluate
    print(df_test)
    print(f"[INFO:] Evaluating the model on the test set... on {len(df_test)} samples.")
    results = evaluate_model(df_test, model)
    print(results)


if __name__ == "__main__":
    main()