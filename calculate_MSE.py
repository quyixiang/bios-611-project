# %%
import pandas as pd
import numpy as np
# %%
df = pd.read_csv("./real_data/all_interpolation.csv")
# %%
masked_data_df = df.loc[df["real_data_with_mask"]>0]
# %%
real_data_with_mask = np.array(masked_data_df["real_data_with_mask"])
bs_interpolation = np.array(masked_data_df["bs_interpolation"])
dl_interpolation = np.array(masked_data_df["dl_interpolation"])
# %%
np.mean(abs(real_data_with_mask - bs_interpolation))
# %%
np.mean(abs(real_data_with_mask - dl_interpolation))
# %%
