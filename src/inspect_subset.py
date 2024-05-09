import pandas as pd

# read the data
norreport_1A = pd.read_csv('data/1A_norreport.csv', parse_dates=['date', 'scheduledarrivetime'])

print(len(norreport_1A))
