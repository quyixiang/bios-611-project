# %%
import pandas as pd
import numpy as np
# %%
df = pd.read_csv("./real_data/all_interpolation.csv")
# %%
masked_data_df = df.loc[df["real_data_with_mask"] > 0]
# %%
real_data_with_mask = np.array(masked_data_df["real_data_with_mask"])
bs_interpolation = np.array(masked_data_df["bs_interpolation"])
dl_interpolation = np.array(masked_data_df["dl_interpolation"])
# %%
bs_mask_diff = list(abs(real_data_with_mask - bs_interpolation))
# %%
dl_mask_diff = list(abs(real_data_with_mask - dl_interpolation))
# %%
no_masked_data_df = df.loc[df["real_data_without_mask"] > 0]
# %%
real_data_without_mask = np.array(no_masked_data_df["real_data_without_mask"])
bs_interpolation = np.array(no_masked_data_df["bs_interpolation"])
dl_interpolation = np.array(no_masked_data_df["dl_interpolation"])

# %%
bs_no_mask_diff = list(abs(real_data_without_mask - bs_interpolation))
# %%
dl_no_mask_diff = list(abs(real_data_without_mask - dl_interpolation))
# %%
bs_diff = bs_mask_diff + bs_no_mask_diff
dl_diff = dl_mask_diff + dl_no_mask_diff
# %%
np.mean(bs_diff)
# %%
np.mean(dl_diff)
# %%
