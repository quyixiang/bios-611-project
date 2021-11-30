
import pandas as pd
import os

df = pd.read_csv("real_data/multivariate_simulation_dl_interpolation.csv")

if not os.path.exists('split_data'):
    os.makedirs('split_data')

for i in range(int(max(df["ID"])+1)):
    temp_data = df[df["ID"] == i]
    temp_data.to_csv("split_data/ID_{}.csv".format(i))

