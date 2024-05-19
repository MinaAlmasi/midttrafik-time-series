import pathlib
import pandas as pd
import matplotlib.pyplot as plt

def plot_test_forecasts(df_test, test_forecasts:dict, forecast_colors:list, save_path=None, file_name=None):
    '''
    Plot the test forecasts against the true values
    '''

    # set the figure
    plt.figure(figsize=(15, 12))

    # plot the true values
    plt.plot(df_test['ds'], df_test['y'], label='True values', color='black')

    # set the colors

    # plot each of the test forecasts 
    for i, (model_name, forecast) in enumerate(test_forecasts.items()):
        plt.plot(df_test['ds'], forecast, label=model_name, color=forecast_colors[i], linestyle='--')    
    
    # add legend
    plt.legend()

    # set axis
    plt.xlabel('Date')
    plt.ylabel('Value')

    # show only every 6th tick
    #plt.xticks(df_test['ds'][::2])

    # turn the x-axis labels
    plt.xticks(rotation=90)

    # ensure tight layout
    plt.tight_layout()

    # save
    if save_path and file_name:
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / file_name)

def load_np_forecasts(forecasts_path):
    data = {}

    for file in forecasts_path.iterdir():
        forecast = pd.read_csv(file)
        model_name = file.stem.split("_")[0]
        
        # check if it is an np model
        if model_name[:2] == "np":
            model_name = "NeuralProphet " + model_name[2:]
            
            # load 
            df = pd.read_csv(file)
            data[model_name] = df['y']

    return data 

def main(): 
    # set paths
    path = pathlib.Path(__file__)
    forecasts_path = path.parents[1] / "results" / "forecasts"
    data_path = path.parents[1] / "data"
    plot_path = path.parents[1] / "plots" / "forecasts"

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
    np_forecasts = load_np_forecasts(forecasts_path)

    # create a dict with all the other forecasts
    baseline_forecasts = {
        "Mean Model": mm_forecast['y'],
        "Naive": naive_forecast['y'],
        "Weekly Naive": weekly_naive_forecast['y'],
    }


    # create a plot for the np models
    plot_test_forecasts(df_test, np_forecasts, forecast_colors=['red', 'blue', 'green', 'orange', 'purple'], save_path=plot_path, file_name='np_forecasts.png')

    # create a plot for only SARIMA
    plot_test_forecasts(df_test, {"SARIMA": sarima_forecast['y']}, forecast_colors=['orange'], save_path=plot_path, file_name='sarima_forecast.png')

    # create a plot for only the baselines
    plot_test_forecasts(df_test, baseline_forecasts, forecast_colors=['red', 'blue', 'green'], save_path=plot_path, file_name='baseline_forecasts.png')


if __name__ == "__main__":
    main()