The `src` folder contains the following:

| Folder/File               | Description |
|---------------------------|-------------|
| `plotting/`               | Scripts to plot time series vizualisations, ACF&PACF and forecasts |
| `process_data/`           | Subset raw data and process bus stops |
| `neural-prophet/`         | Scripts to run rolling-origin grid search (`fit_prophet.py`) and forecast with the five best models (`test_prophet.py`) |
| `sarima/`         | Scripts to find optimal parameters (`stationarity.py` and `auto_arima.py`), perform rolling-origin evaluation (`fit_sarima.py`) and forecast with the best model (`test_sarima.py)`)|
| `fit_baselines.py`         | Evaluate and do forecasts with baseline models |
| `data_utils.py`         | Functions for data loading, spliting and imputation used in various scripts|
| `table_to_latex.py`         | Convert grid search results of neural-prophet to a neat latex table|