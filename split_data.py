# %%
import pandas as pd
import numpy as np
# %%
df = pd.read_csv("real_data/multivariate_simulation_dl_interpolation.csv")
# %%
for i in range(int(max(df["ID"])+1)):
    temp_data = df[df["ID"] == i]
    temp_data.to_csv("split_data/ID_{}.csv".format(i))
# %%
