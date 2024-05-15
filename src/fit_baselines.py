import pathlib
import pandas as pd
from data_utils import split_rolling_origin
import matplotlib.pyplot as plt

def impute_missing(df, method='rolling', window=20):
    '''
    Impute missing values in a dataframe using a specified method.
    '''
    if method == 'linear':
        df['y'] = df['y'].interpolate(method='linear', limit_direction='both', limit=window)
    elif method == 'rolling':
        df['y'] = df['y'].interpolate(method='linear', limit_direction='both', limit=window)
    else:
        raise ValueError(f"Method {method} not recognized. Please use 'linear' or 'rolling'.")

    return df


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
    for split, df in zip(["val", "test"], [df_vals, df_test]):
        df['y_pred'] = mean # add mean to pred col

        mae = abs(df['y'] - df['y_pred']).mean() # compute mae 
        mae = round(mae, 3)

        mae_values[split] = mae

    return mae_values

def naive_model(df_trains, df_vals, df_tests, gap=24):
    '''
    Naive model which always predicts last value in the training set (per fold)
    '''

    # for each fold, identify the last value in the training set
    for fold in df_trains['fold'].unique():
        # get the last value in the training set
        last_value = df_trains[df_trains['fold'] == fold].iloc[-1]['y']

        # add to the test set
        df_vals.loc[df_vals['fold'] == fold, 'y_pred'] = last_value

    # calculate MAE as a column
    df_vals['mae'] = abs(df_vals['y'] - df_vals['y_pred'])

    val_mae = df_vals['mae'].mean()
    val_mae = round(val_mae, 3)

    # add horizon col
    df_vals["horizon"] = df_vals.groupby('fold').cumcount() + gap

    # select mae and horizon for new df 
    naive_results = df_vals[['mae', 'horizon']]

    # for the test set, we simply predict the last value in the training set overall
    last_value = df_trains.iloc[-1]['y']

    df_tests['y_pred'] = last_value

    # calculate MAE
    df_tests['mae'] = abs(df_tests['y'] - df_tests['y_pred'])
    train_mae = df_tests['mae'].mean()
    train_mae = round(train_mae, 3)

    # create a dict with the MAE values
    mae_values = {'val': val_mae, 'test': train_mae}

    return mae_values, naive_results

def weekly_naive_model(df, df_vals, df_test):
    '''
    Seasonal naive model which predicts the value from the same time one week ago
    '''
    # calculate how many indexes to go back to get the value from one week ago
    timesteps_back = 24 * 7

    # loop over each row in the test set
    for _, row in df_vals.iterrows():
        # get the index of the row
        index = row.name

        # get the index of the value from one week ago
        index_week_ago = index - timesteps_back

        # get the value from one week ago
        value_week_ago = df.loc[index_week_ago, 'y']

        # add to the test set
        df_vals.loc[index, 'y_pred'] = value_week_ago

    # calculate MAE
    val_mae = abs(df_vals['y'] - df_vals['y_pred']).mean()
    val_mae = round(val_mae, 3)

    # repeat the process for the test set
    for _, row in df_test.iterrows():
        index = row.name
        index_week_ago = index - timesteps_back
        value_week_ago = df.loc[index_week_ago, 'y']
        df_test.loc[index, 'y_pred'] = value_week_ago
    
    test_mae = abs(df_test['y'] - df_test['y_pred']).mean()
    test_mae = round(test_mae, 3)

    # add to one dict
    mae_values = {'val': val_mae, 'test': test_mae}

    return mae_values

def plot_naive_horizon(results, save_path=None, file_name=None):
    '''
    Plot the MAE as a function of the horizon
    '''
    # group by horizon and calculate the mean and sd
    results = results.groupby('horizon').agg({'mae': ['mean', 'std']}).reset_index()

    # plot as a line plot with shaded error
    plt.figure(figsize=(10, 6))

    plt.plot(results['horizon'], results['mae']['mean'], color='blue')

    plt.fill_between(results['horizon'],
                        results['mae']['mean'] - results['mae']['std'],
                        results['mae']['mean'] + results['mae']['std'],
                        color='blue', alpha=0.2)

    plt.xlabel('Horizon')
    plt.ylabel('MAE')

    # save 
    if save_path and file_name:
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / file_name)
    
def main(): 
    # set paths
    path = pathlib.Path(__file__)
    data_path = path.parents[1] / 'data'

    # load the data
    df = pd.read_csv(data_path / 'processed_1A_norreport.csv')

    # impute missing values
    df = impute_missing(df, method='rolling', window=20)

    # split the data
    gap = 24
    train_inds, val_inds, test_inds = split_rolling_origin(df['ds'], gap=gap, test_size=36, steps=4, min_train_size=24*7)
    df_trains, df_vals, df_test = get_splits(df, train_inds, val_inds, test_inds)
    
    # fit the mean model
    mean_model_mae = mean_model(df, df_vals, df_test)

    # fit the naive model
    naive_mae, naive_results = naive_model(df_trains, df_vals, df_test)

    plot_naive_horizon(naive_results, save_path=path.parents[1] / 'plots', file_name='naive_horizon.png')

    # fit the weekly naive model
    weekly_naive_mae = weekly_naive_model(df, df_vals, df_test)
    
    # print the results
    print("\n")
    print(f'Mean model MAE: {mean_model_mae}')
    print(f'Naive model MAE: {naive_mae}')
    print(f'Weekly naive model MAE: {weekly_naive_mae}')


if __name__ == "__main__":
    main()