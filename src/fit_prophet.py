import pandas as pd
import pathlib
import multiprocessing as mp
from neuralprophet import NeuralProphet
from sklearn.model_selection import ParameterGrid
import random
from data_utils import split_rolling_origin
from datetime import datetime

def cv_single_fold(train_ind, test_ind, df, params, freq):
    '''
    Process a single fold of cross-validation using provided parameters.
    '''
    df_train = df.iloc[train_ind]
    df_test = df.iloc[test_ind]
    
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

    train_metrics = model.fit(df_train, freq=freq)
    val_metrics = model.test(df_test, verbose=False)

    # get last MAE from training and (only) MAE from validation
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

def sample_parameter_combinations(n_combinations, param_grid):
    # Create a list of all possible combinations from the parameter grid
    all_combinations = list(ParameterGrid(param_grid))

    # check if n_combinations is larger than the number of possible combinations
    if n_combinations > len(all_combinations):
        print("Warning: Requested more combinations than possible. Returning all combinations.")
        sampled_combinations = all_combinations
    
    # Sample n_combinations from the list, ensuring not to exceed the number of available combinations
    if len(all_combinations) < n_combinations:
        print("Warning: Requested more combinations than possible. Returning all combinations.")
        sampled_combinations = all_combinations
    else:
        sampled_combinations = random.sample(all_combinations, n_combinations)
    
    # announce sampled combinations
    print(f"Sampled {len(sampled_combinations)} parameter combinations.")

    print("Sampled parameter combinations:")
    for i, params in enumerate(sampled_combinations):
        print(f"Combination {i+1}: {params}")

    return sampled_combinations

def cross_validate(df, train_inds:dict, test_inds:dict, params:dict, freq="1h", n_cores:int=1):
    '''
    Cross validate the model with given hyperparameters using multiprocessing.

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
        args = (train_ind, test_ind, df, params, freq)
        processes.append(args)
    
    with mp.Pool(n_cores) as pool:
        mae_trains, mae_vals = zip(*pool.starmap(cv_single_fold, processes)) # unpack results

    mean_mae_train, sd_mae_train = get_mean_sd(mae_trains)
    mean_mae_val, sd_mae_val = get_mean_sd(mae_vals)

    return mean_mae_train, sd_mae_train, mean_mae_val, sd_mae_val

def main():
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / 'data'
    df = pd.read_csv(data_path / 'processed_1A_norreport.csv')

    # hyperparameters to explore
    param_grid = {
        'n_lags': [1, 6],
        'newer_samples_weight': [2, 4],
        'ar_layers': [[1], [32, 16]],
        'seasonality_reg': [0.1, 0.5],
        'learning_rate': [0.15, 0.3],
        'batch_size': [24, 48],
        'epochs': [1],
    }


    # sample parameter combinations
    n_param_combinations = 5
    sampled_combinations = sample_parameter_combinations(n_param_combinations, param_grid)

    # data setup
    gap = 24 # gap between train and val and train and test
    steps = 4 # how fast rolling origin moves
    test_size = 36 # hours 
    min_train_size = 24*7 # i.e. 1 week 

    train_inds, val_inds, _ = split_rolling_origin(df['ds'], gap=gap, test_size=test_size, steps=steps, min_train_size=min_train_size) # no need for test_inds here!

    # Setup results path and unique filename
    results_dir = path.parents[1] / "results" / "neuralprophet"
    results_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = results_dir / f'np_gridsearch_{timestamp}.csv'

    results = []
    # cross-validation for each sampled parameter set
    for i, params in enumerate(sampled_combinations):
        mean_mae_train, sd_mae_train, mean_mae_val, sd_mae_val = cross_validate(df, train_inds, val_inds, params, freq="1h", n_cores=mp.cpu_count() - 1)
        
        # Add the results to the DataFrame
        results.append({
            'model_number': i+1,
            'parameters': params,
            'mean_mae_train': mean_mae_train,
            'sd_mae_train': sd_mae_train,
            'mean_mae_val': mean_mae_val,
            'sd_mae_val': sd_mae_val
        })
        
        # convert results to DataFrame and append to file after each iteration
        results_df = pd.DataFrame([results[-1]])  # create a DataFrame of the latest results
        if not filename.exists():  # write header only if file doesn't exist yet
            results_df.to_csv(filename, index=False, mode='a')
        else:
            results_df.to_csv(filename, index=False, mode='a', header=False)

if __name__ == "__main__":
    main()