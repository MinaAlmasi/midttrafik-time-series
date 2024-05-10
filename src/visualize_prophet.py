import pathlib
import pandas as pd 
from neuralprophet import NeuralProphet, set_log_level
import matplotlib.pyplot as plt

def main():
    path = pathlib.Path(__file__)

    data_path = path.parents[1] / "data" 
    plot_dir = path.parents[1] / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    # read the data
    norreport_1A = pd.read_csv(data_path / 'processed_1A_norreport.csv')

    # init neural prophet model
    model = NeuralProphet(
                        yearly_seasonality = True,  
                        weekly_seasonality = True, 
                        daily_seasonality = True
                        )

    # fit the model with 30-minute frequency
    metrics = model.fit(norreport_1A, freq='30min')

    print(metrics)

    # forecast
    forecast = model.predict(norreport_1A)

    # plot the seasonality components
    plot = model.plot_parameters(components=["seasonality"])
    
    plot.write_image(file = str(plot_dir / "seasonality_components.png"), format = "png")


if __name__ == "__main__":
    main()