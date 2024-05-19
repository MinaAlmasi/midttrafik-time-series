import pathlib
import pandas as pd
import matplotlib.pyplot as plt

def plot_test_forecasts(actual_results, test_forecasts:dict, forecast_colors:list, save_path=None, file_name=None):
    '''
    Plot the test forecasts against the true values
    '''

    # set the figure
    plt.figure(figsize=(24, 12))

    # change font to times new roman
    plt.rcParams['font.family'] = 'Times New Roman'

    # plot the true values
    plt.plot(actual_results['ds'], actual_results['y'], label='Actual Passengers', color='black', linestyle='-', linewidth=4)

    # plot each of the test forecasts 
    for i, (model_name, forecast) in enumerate(test_forecasts.items()):
        plt.plot(actual_results['ds'][36:], forecast, label=model_name, color=forecast_colors[i], linestyle='--', linewidth=2)    
    
    # add legend
    plt.legend(fontsize=18, loc='upper right')

    # set axis
    plt.xlabel('Datetime', fontsize=22, labelpad=18)
    plt.ylabel('Number of Passengers', fontsize=22, labelpad=18)

    # extend y-axis to go to 60
    plt.ylim(-2, 60)

    # increase size of x and y axis titles
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # show only every 2nd x-axis label
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(2))

    # turn the x-axis labels
    plt.xticks(rotation=90)

    # set the xlim
    plt.xlim(actual_results['ds'].iloc[0], actual_results['ds'].iloc[-1])

    # put background shading for the first 12 hours
    plt.axvspan(actual_results['ds'].iloc[0], actual_results['ds'].iloc[12], color='grey', alpha=0.4)

    # put other shading for the next 24 hours
    plt.axvspan(actual_results['ds'].iloc[12], actual_results['ds'].iloc[36], color='grey', alpha=0.2)

    # put third shading for the final 36 hours
    plt.axvspan(actual_results['ds'].iloc[36], actual_results['ds'].iloc[-1], color='grey', alpha=0.1)

    # add labels for the shaded areas
    plt.text(actual_results['ds'].iloc[6], 56, 'TRAINING PHASE \n(partially shown)', fontsize=22, horizontalalignment='center', weight='bold', verticalalignment='center')
    plt.text(actual_results['ds'].iloc[24], 56, 'GAP', fontsize=22, horizontalalignment='center', weight='bold')
    plt.text(actual_results['ds'].iloc[54], 56, 'TESTING PHASE', fontsize=22, horizontalalignment='center', weight='bold')

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
    
    # set data
    gap = 24
    test_size = 36
    
    # subset to last 72 hours (36 test, 24 gap, 12 train)
    actual_results = df.iloc[-72:]

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
    plot_test_forecasts(actual_results, np_forecasts, forecast_colors=['red', 'blue', 'green', 'orange', 'purple'], save_path=plot_path, file_name='np_forecasts.png')

    # create a plot for only SARIMA
    plot_test_forecasts(actual_results, {"SARIMA": sarima_forecast['y']}, forecast_colors=['orange'], save_path=plot_path, file_name='sarima_forecast.png')

    # create a plot for only the baselines
    plot_test_forecasts(actual_results, baseline_forecasts, forecast_colors=['red', 'blue', 'green'], save_path=plot_path, file_name='baseline_forecasts.png')


if __name__ == "__main__":
    main()