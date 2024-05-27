import pathlib
import pandas as pd
import matplotlib.pyplot as plt

def plot_test_forecasts(actual_results, test_forecasts:dict, forecast_colors:list, figsize=(15,12), heading_fontsize=22, legend_outside=False, save_path=None, file_name=None):
    '''
    Plot the test forecasts against the true values
    '''

    # set the figure
    plt.figure(figsize=figsize)

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
    plt.text(actual_results['ds'].iloc[6], 56, 'TRAINING \n(partially shown)', fontsize=heading_fontsize, horizontalalignment='center', weight='bold', verticalalignment='center')
    plt.text(actual_results['ds'].iloc[24], 56, 'GAP', fontsize=heading_fontsize, horizontalalignment='center', weight='bold')
    plt.text(actual_results['ds'].iloc[54], 56, 'FORECASTING', fontsize=heading_fontsize, horizontalalignment='center', weight='bold')

    # place legend on top of the plot if legend_outside is True
    if legend_outside:
        plt.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)

    # save
    if save_path and file_name:
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / file_name, bbox_inches='tight', dpi=300)

def load_np_forecasts(forecasts_path):
    '''
    Load forecasts from the NeuralProphet models
    '''
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

def norreport_forecasts(norreport_df, forecasts_path, save_path):
   
    # subset to last 72 hours (36 test, 24 gap, 12 train)
    actual_results = norreport_df.iloc[-72:]

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
    figsize = (24, 12)
    heading_fontsize = 20
    plot_test_forecasts(actual_results, np_forecasts, forecast_colors=['red', 'blue', 'green', 'orange', 'purple'], 
                        figsize=figsize, heading_fontsize=heading_fontsize,
                        save_path=save_path, file_name='norreport_np_forecasts.png')

    # create a plot for only SARIMA
    plot_test_forecasts(actual_results, {"SARIMA": sarima_forecast['y']}, forecast_colors=['orange'],
                        figsize=figsize, heading_fontsize=heading_fontsize, 
                        save_path=save_path, file_name='norreport_sarima_forecast.png')

    # create a plot for only the baselines
    plot_test_forecasts(actual_results, baseline_forecasts, forecast_colors=['red', 'blue', 'green'], 
                        figsize=figsize, heading_fontsize=heading_fontsize,
                        save_path=save_path, file_name='norreport_baseline_forecasts.png')

def other_stops_forecast(df, stop_name, results_path, plot_path):
    '''
    Plot the forecasts for the other stops
    '''
    # subset to last 72 hours (36 test, 24 gap, 12 train)
    actual_results = df.iloc[-72:]

    # load the weekly naive forecast
    weekly_naive_forecast = pd.read_csv(results_path / "weekly_naive_forecast.csv")

    # load the sarima forecast
    sarima_forecast = pd.read_csv(results_path / "sarima_forecast.csv")
    
    # load the neuralprophet forecasts
    np_forecast = pd.read_csv(results_path / "np37_forecast.csv")

    # create a dict with all three forecasts
    forecasts = {"Weekly Naive": weekly_naive_forecast['y'], 
                 "SARIMA": sarima_forecast['y'], 
                 "NeuralProphet 37": np_forecast['y']}

    # create a plot for the three models
    figsize = (15, 12)
    heading_fontsize = 17
    plot_test_forecasts(actual_results, forecasts, forecast_colors=['green', 'orange', 'purple'], 
                        figsize=figsize, heading_fontsize=heading_fontsize,
                        save_path=plot_path, file_name=f'{stop_name}_forecasts.png', legend_outside=True)


def main(): 
    # set paths
    path = pathlib.Path(__file__)
    results_path = path.parents[2] / "results"
    data_path = path.parents[2] / "data" / "clean_stops"
    plot_path = path.parents[2] / "plots" / "forecasts"

    # set norreport paths
    norreport_forecast = results_path / "norreport" / "forecasts"

    # load the norreport data
    df = pd.read_csv(data_path / 'clean_1A_norreport.csv')

    # plot the forecasts for norreport
    norreport_forecasts(df, norreport_forecast, plot_path)

    # iterate over the other stops
    for stop in data_path.iterdir():
        if stop.name == 'clean_1A_norreport.csv': # skip norreport as it has its seperate pipeline
            continue
        
        # load the data
        df = pd.read_csv(stop)

        # stop name (remove clean_1A_ and .csv) from name
        stop_name = stop.stem[9:]

        # results path 
        other_results_path = results_path / "other_stops" / stop_name

        # run the pipeline
        other_stops_forecast(df, stop_name, other_results_path, plot_path, )



if __name__ == "__main__":
    main()