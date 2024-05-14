import pandas as pd
import pathlib
import multiprocessing as mp
from neuralprophet import NeuralProphet
from sklearn.model_selection import ParameterGrid
import random
from data_utils import split_rolling_origin

def cv_single_fold(train_ind, test_ind, df, init_params, fit_params, freq):
    '''
    Process a single fold of cross-validation using provided parameters.
    '''
    df_train = df.iloc[train_ind]
    df_test = df.iloc[test_ind]
    
    default_params = {
        "yearly_seasonality": False, # set yearly seasonality also to avoid model having to check for it 
        'weekly_seasonality': True,
        'daily_seasonality': True,
    }

    # combine default parameters with the initialization grid
    params = {**default_params, **init_params}

    # Initialize model with parameters from the initialization grid
    model = NeuralProphet(**params)
    
    # Fit and test the model with parameters for the fit method
    train_metrics = model.fit(df_train, freq=freq, **fit_params)
    val_metrics = model.test(df_test, verbose=False)

    # get the mae from both the training and validation set
    mae_train = train_metrics['MAE'].values[-1]
    mae_val = val_metrics['MAE_val'].values[0]
    
    return mae_train, mae_val

def get_mean_sd(vals:list):
    '''
    calculate the mean and standard deviation of a list of values.
    '''
    mean = sum(vals) / len(vals)
    sd = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5

    return mean, sd

def cross_validate(df, train_inds:dict, test_inds:dict, init_params:dict, fit_params:dict, freq="1h", n_cores:int=1):
    '''
    Cross validate the model with given hyperparameters.

    Args:
        df: dataframe with the data
        train_inds, test_inds: dictionary with the train, test indices for each fold
        init_params, fit_params: dictionary with the initialization, fit parameters
        freq: frequency of the data
        n_cores: number of cores to use for parallel processing

    Returns:
        mean_mae: mean MAE across all folds
        sd_mae: standard deviation of the MAE across all folds
        
    '''
    processes = []
    for (train_ind, test_ind) in zip(train_inds.values(), test_inds.values()):
        args = (train_ind, test_ind, df, init_params, fit_params, freq)
        processes.append(args)
    
    with mp.Pool(n_cores) as pool:
        mae_trains, mae_vals = zip(*pool.starmap(cv_single_fold, processes))

    mean_mae_train, sd_mae_train = get_mean_sd(mae_trains)
    mean_mae_val, sd_mae_val = get_mean_sd(mae_vals)

    return mean_mae_train, sd_mae_train, mean_mae_val, sd_mae_val

def main():
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / 'data'
    df = pd.read_csv(data_path / 'processed_1A_norreport.csv')

    # hyperparameters to explore
    init_param_grid = {
        'n_lags': [1, 2],
        'newer_samples_weight': [2, 4],
        'newer_samples_start': [0.5],
        'ar_layers': [[1], [32, 16]],
        'seasonality_reg': [0.1, 0.5]
    }

    fit_param_grid = {
        'learning_rate': [0.05, 0.15, 0.3],
        'batch_size': [24, 48],
        'epochs': [1],
    }
    
    # sample all combinations and sample randomly from them if necessary
    init_combinations = list(ParameterGrid(init_param_grid))
    fit_combinations = list(ParameterGrid(fit_param_grid))
    
    sampled_init_combinations = random.sample(init_combinations, min(len(init_combinations), 2))  
    sampled_fit_combinations = random.sample(fit_combinations, min(len(fit_combinations), 2))  

    # wplit data
    train_inds, test_inds = split_rolling_origin(df['ds'], gap=24, test_size=36, steps=4, min_train_size=24*7)

    results = []
    # cross-validation for each sampled parameter set
    for init_params in sampled_init_combinations:
        for fit_params in sampled_fit_combinations:
            mean_mae_train, sd_mae_train, mean_mae_val, sd_mae_val = cross_validate(df, train_inds, test_inds, init_params, fit_params, freq="1h", n_cores=mp.cpu_count() - 1)
            
            # add the results to the DataFrame
            results.append({
                'init_params': init_params,
                'fit_params': fit_params,
                'mean_mae_train': mean_mae_train,
                'sd_mae_train': sd_mae_train,
                'mean_mae_val': mean_mae_val,
                'sd_mae_val': sd_mae_val
            })

    results_df = pd.DataFrame(results)
    results_dir = path.parents[1] / 'results'
    results_dir.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(results_dir / 'neuralprophet_cv_gridsearch.csv', index=False)

if __name__ == "__main__":
    main()