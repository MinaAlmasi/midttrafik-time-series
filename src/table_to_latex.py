import pathlib
import pandas as pd



def main():
    # set paths
    path = pathlib.Path(__file__)
    table_path = path.parents[1] / 'results' / 'norreport' / 'neural-prophet-grid-search'

    # read table
    table = pd.read_csv(table_path / 'np_gridsearch_20240520_210018.csv')

    # drop unnecessary columns
    table = table.drop(columns=['sd_mae_train', 'sd_mae_val', 'sd_rmse_train', 'sd_rmse_val'])

    # reorder columns to mean_mae_train, mean_mae_val, mean_rmse_train, mean_rmse_val
    table = table[['model_number', 'model', 'mean_mae_train', 'mean_rmse_train', 'mean_mae_val', 'mean_rmse_val']]

    # create replace dict
    replace_dict = {"ar_layers": "AR",
                    "batch_size": "BS",
                    "epochs": "EP",
                    "learning_rate": "LR",
                    "n_lags": "NL",
                    "newer_samples_weight": "NS",
                    "seasonality_reg": "SR"}

    # in the model column, every time a key from the replace dict is found, it is replaced by the corresponding value
    table['model'] = table['model'].replace(replace_dict, regex=True)

    # remove citation signs from the model column
    table['model'] = table['model'].str.replace("'", "")

    # in the model_number column, add "NeuralProphet" to the model number
    table['model_number'] = "NeuralProphet " + table['model_number'].astype(str)

    # round all numeric columns to 2 decimal places
    table = table.round(2)

    # sort by mean_rmse_val (ascending)
    table = table.sort_values(by='mean_rmse_val')

    # change column names
    table.columns = ['Model Name', 'Parameters', 'MAE Train', 'RMSE Train', 'MAE Test', 'RMSE Test']

    # save as a latex table with multicolumn MAE TRAIN and RMSE TRAIN as well as MAE TEST and RMSE TEST
    table.to_latex(table_path / 'np_gridsearch_20240520_210018.tex', index=False, multicolumn=True, multicolumn_format='c', escape=False)


    print(table)

if __name__ == '__main__':
    main()
