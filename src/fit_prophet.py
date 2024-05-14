import pandas as pd
import pathlib
import multiprocessing as mp
from neuralprophet import NeuralProphet
from sklearn.model_selection import ParameterGrid
import random
from data_utils import split_timeseries_data

def cv_single_fold(train_ind, test_ind, df, init_params, fit_params, freq):
    '''
    Process a single fold of cross-validation using provided parameters.
    '''
    df_train = df.iloc[train_ind]
    df_test = df.iloc[test_ind]
    
    # Initialize model with parameters from the initialization grid
    model = NeuralProphet(**init_params)
    
    # Fit and test the model with parameters for the fit method
    model.fit(df_train, freq=freq, **fit_params)
    metrics = model.test(df_test, verbose=False)
    mae_val = metrics['MAE_val'].values[0]
    
    return mae_val

def cross_validate(df, data_generator, init_params, fit_params, freq="1h", n_cores:int=1):
    '''
    Function to cross validate the model with given hyperparameters.
    '''
    processes = []
    for train_ind, test_ind in data_generator:
        args = (train_ind, test_ind, df, init_params, fit_params, freq)
        processes.append(args)
    
    print(f"Starting {processes} processes with {n_cores} cores...")
    with mp.Pool(n_cores) as pool:
        results = pool.starmap(cv_single_fold, processes)

    mean_mae = sum(results) / len(results)
    sd_mae = (sum([(x - mean_mae) ** 2 for x in results]) / len(results)) ** 0.5
    return mean_mae, sd_mae

def main():
    # Set paths and load data
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / 'data'
    df = pd.read_csv(data_path / 'processed_1A_norreport.csv')

    # Define the grid of hyperparameters to explore
    init_param_grid = {
        'n_lags': [1, 2],
        'newer_samples_weight': [2, 4],
        'newer_samples_start': [0.5],
        #'add_country_holidays': ['DA', False],
        'ar_layers': [[1], [32, 16]],
        'seasonality_reg': [0.1, 0.5]
    }

    fit_param_grid = {
        'learning_rate': [0.05, 0.15, 0.3],
        'batch_size': [24, 48],
        'epochs': [2, 3, 4]
    }

    # Create all combinations and sample randomly from them if necessary
    init_combinations = list(ParameterGrid(init_param_grid))
    fit_combinations = list(ParameterGrid(fit_param_grid))
    
    sampled_init_combinations = random.sample(init_combinations, min(len(init_combinations), 10))  # Adjust sampling size as needed
    sampled_fit_combinations = random.sample(fit_combinations, min(len(fit_combinations), 10))  # Adjust sampling size as needed

    # Perform cross-validation for each sampled parameter set
    best_mae = float('inf')
    best_params = None
    for init_params in sampled_init_combinations:
        for fit_params in sampled_fit_combinations:
            print("[INFO] Testing parameters:", init_params, fit_params)
            
            # Split the data
            gap = 24
            max_train_size = 24 * 7 * 4
            generator = split_timeseries_data(df['ds'], gap=gap, test_size=24, max_train_size=max_train_size)
            
            mean_mae, sd_mae = cross_validate(df, generator, init_params, fit_params, freq="1h", n_cores=mp.cpu_count() - 1)
            print(f"Results for parameters {init_params} and {fit_params}: Mean MAE = {mean_mae}, SD MAE = {sd_mae}")
            if mean_mae < best_mae:
                best_mae = mean_mae
                best_params = (init_params, fit_params)

    print("Best parameters found:", best_params)
    print(f"Best Mean MAE: {best_mae}")

if __name__ == "__main__":
    main()