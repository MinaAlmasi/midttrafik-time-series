import pathlib
import pandas as pd 
from neuralprophet import NeuralProphet, set_log_level
import multiprocessing as mp
from data_utils import split_timeseries_data
    
def cv_single_fold(train_ind, test_ind, df, freq):
    '''
    process a single fold of cross-validation
    '''
    df_train = df.iloc[train_ind]
    df_test = df.iloc[test_ind]
    
    # init model
    model = NeuralProphet(
                        weekly_seasonality = True, 
                        daily_seasonality = True,
                        epochs=50,
                        batch_size=48,
                        learning_rate=0.2,
                        loss_func = "MSE"
                        )
    # fit and test
    model.fit(df_train, freq=freq)
    metrics = model.test(df_test, verbose=False)
    mae_val = metrics['MAE_val'].values[0]
    
    return mae_val

def cross_validate(df, data_generator, freq="30min", n_cores:int=1):
    '''
    function to cross validate the model
    '''
    processes = []
    for i, (train_ind, test_ind) in enumerate(data_generator):
        args = (train_ind, test_ind, df, freq)
        processes.append(args)

    # perform multiprocessing for each fold using a context manager
    with mp.Pool(n_cores) as pool:
        # use starmap to apply the process_fold function to each set of arguments in parallel
        results = pool.starmap(cv_single_fold, processes)

    # compute mean MAE and sd
    mean_mae = sum(results) / len(results)
    sd_mae = (sum([(x - mean_mae) ** 2 for x in results]) / len(results)) ** 0.5

    return mean_mae, sd_mae

def main(): 
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / 'data'

    # load the data
    df = pd.read_csv(data_path / 'processed_1A_norreport.csv')

    # split the data
    gap = 24
    max_train_size = 48 * 7 * 2 # 24 hours x 7 (days) x 2 (weeks)
    generator = split_timeseries_data(df['ds'], gap=gap, test_size=48, max_train_size=max_train_size)

    # cross validate
    print("[INFO:] Cross-validating the model...")
    n_cores = mp.cpu_count() - 1
    mean_mae, sd_mae = cross_validate(df, generator, freq="30min", n_cores=n_cores)

    print(f"Mean MAE: {mean_mae}")
    print(f"SD MAE: {sd_mae}")

if __name__ == "__main__":
    main()