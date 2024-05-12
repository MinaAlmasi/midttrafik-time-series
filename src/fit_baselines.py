import pathlib
import pandas as pd
from data_utils import split_timeseries_data

def mean_model(df, df_tests):
    # remove the test indices from the dataframe
    df_train = df[~df.index.isin(df_tests.index)]

    # calculate the mean
    mean = df_train['y'].mean()

    # add as a column to the test dataframe
    df_tests['y_pred'] = mean

    # calculate MAE
    mae = abs(df_tests['y'] - df_tests['y_pred']).mean()
    mae = round(mae, 3)

    return mae

def naive_model(df_trains, df_tests):
    '''
    Naive model which always predicts last value in the training set (per fold)
    '''
    # for each fold, identify the last value in the training set
    for fold in df_trains['fold'].unique():
        # get the last value in the training set
        last_value = df_trains[df_trains['fold'] == fold].iloc[-1]['y']

        # add to the test set
        df_tests.loc[df_tests['fold'] == fold, 'y_pred'] = last_value

    # calculate MAE
    mae = abs(df_tests['y'] - df_tests['y_pred']).mean()
    mae = round(mae, 3)

    return mae

def weekly_naive_model(df, df_tests):
    '''
    Seasonal naive model which predicts the value from the same time one week ago
    '''
    # calculate how many indexes to go back to get the value from one week ago
    timesteps_back = 48 * 7

    # loop over each row in the test set
    for i, row in df_tests.iterrows():
        # get the index of the row
        index = row.name

        # get the index of the value from one week ago
        index_week_ago = index - timesteps_back

        # get the value from one week ago
        value_week_ago = df.loc[index_week_ago, 'y']

        # add to the test set
        df_tests.loc[index, 'y_pred'] = value_week_ago
    
    # calculate MAE
    mae = abs(df_tests['y'] - df_tests['y_pred']).mean()
    mae = round(mae, 3)

    return mae

def get_splits(df, data_generator):
    '''
    function to cross validate the model
    '''
    df_trains = pd.DataFrame()
    df_tests = pd.DataFrame()
    for i, (train_ind, test_ind) in enumerate(data_generator):
        df_train = df.iloc[train_ind]
        df_test = df.iloc[test_ind]

        # create fold column as the first column
        df_train.insert(0, 'fold', i)
        df_test.insert(0, 'fold', i)

        # add to dataframe
        df_trains = pd.concat([df_trains, df_train], axis=0)
        df_tests = pd.concat([df_tests, df_test], axis=0)
    
    return df_trains, df_tests
    
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
    df_trains, df_tests = get_splits(df, generator)

    # fit the mean model
    mae = mean_model(df, df_tests)
    print(f'Mean model MAE: {mae}')

    # fit the naive model
    mae = naive_model(df_trains, df_tests)
    print(f'Naive model MAE: {mae}')

    # fit the weekly naive model
    mae = weekly_naive_model(df, df_tests)
    print(f'Weekly naive model MAE: {mae}')


if __name__ == "__main__":
    main()