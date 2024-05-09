import pathlib
import pandas as pd 
from neuralprophet import NeuralProphet, set_log_level

def main():
    path = pathlib.Path(__file__)

    data_path = path.parents[1] / "data" 

    # read the data
    norreport_1A = pd.read_csv(data_path / 'processed_1A_norreport.csv')

    # init neural prophet model
    model = NeuralProphet()

    # set plotting backend to resampler
    model.set_plotting_backend('plotly-resampler')

    # subset data and rename for neural prophet
    model_data = norreport_1A[['datetime', 'Cumulative']]
    model_data.columns = ['ds', 'y']

    # fit the model
    model.fit(model_data)




if __name__ == "__main__":
    main()