import pathlib
import pandas as pd
import matplotlib.pyplot as plt

def plot_test_forecasts(df_test, test_forecasts:dict, forecast_colors:list, save_path=None, file_name=None):
    '''
    Plot the test forecasts against the true values
    '''

    # set the figure
    plt.figure(figsize=(10, 6))

    # plot the true values
    plt.plot(df_test['ds'], df_test['y'], label='True values', color='black', linestyle='--')

    # set the colors

    # plot each of the test forecasts 
    for model_name, forecast in test_forecasts.items():
        plt.plot(df_test['ds'], forecast, label=model_name, linestyle='-', color=colors.pop(0))
    
    # add legend
    plt.legend()

    # set axis
    plt.xlabel('Date')
    plt.ylabel('Value')

    # show only every 6th tick
    plt.xticks(df_test['ds'][::12])

    # save
    if save_path and file_name:
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / file_name)

def main(): 
    # set paths
    path = pathlib.Path(__file__)
    forecasts_path = path.parents[1] / "results" / "forecasts"
    data_path = path.parents[1] / "data"

    # load the data
    df = pd.read_csv(data_path / 'processed_1A_norreport.csv')
    
    # subset to only test set (last 36 hours)
    df_test = df.iloc[-36:]

    # load the baseline forecasts
    mm_forecast = pd.read_csv(forecasts_path / "mm_forecast.csv")
    naive_forecast = pd.read_csv(forecasts_path / "naive_forecast.csv")
    weekly_naive_forecast = pd.read_csv(forecasts_path / "weekly_naive_forecast.csv")

    # load the sarima forecast
    sarima_forecast = pd.read_csv(forecasts_path / "sarima_forecast.csv")

    # load the neuralprophet forecasts
    np28_forecast = pd.read_csv(forecasts_path / "np28_forecast.csv")
    np33_forecast = pd.read_csv(forecasts_path / "np33_forecast.csv")
    np34_forecast = pd.read_csv(forecasts_path / "np34_forecast.csv")
    np1_forecast = pd.read_csv(forecasts_path / "np1_forecast.csv")
    np21_forecast = pd.read_csv(forecasts_path / "np21_forecast.csv")

    # create a dict with all the forecasts
    test_forecasts = {
        "Mean Model": mm_forecast['yhat'],
        "Naive": naive_forecast['yhat'],
        "Weekly Naive": weekly_naive_forecast['yhat'],
        "SARIMA": sarima_forecast['yhat'],
        "NeuralProphet 28": np28_forecast['yhat1'],
        "NeuralProphet 33": np33_forecast['yhat1'],
        "NeuralProphet 34": np34_forecast['yhat1'],
        "NeuralProphet 1": np1_forecast['yhat1'],
        "NeuralProphet 21": np21_forecast['yhat1']
    }

    #


    