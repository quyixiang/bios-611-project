import numpy as np
import pandas as pd

model_name = "multivariate_simulation"
data_path = "real_data/{}.npy".format(model_name)
summary_dict_path = "real_data/{}_summarydict.npy".format(model_name)
data_dict = np.load(data_path, allow_pickle=True).item()

summary_dict = np.load(summary_dict_path, allow_pickle=True).item()
zero_rate_dict = summary_dict["model_params"]["zero_rate_dict"]
zero_rate_list = []
for key in zero_rate_dict:
    zero_rate_list.append(zero_rate_dict[key])

def truncate(data, zero_rate_list, t):
    final_data = []
    for i in range(data.shape[2]):
        temp_data = data[:, :, i]
        threshold = np.quantile(temp_data, zero_rate_list[i] * t)
        # any data smaller than threshold should be treated as 0
        temp_data_with_real_zero = temp_data * (temp_data > threshold)
        final_data.append(temp_data_with_real_zero)

    target = np.stack(final_data, axis=2)
    return target


pre_values = truncate(data_dict["pre_values_no_normalization"], zero_rate_list,
                      0.3).transpose(1, 0, 2)
time = data_dict["times"]

all_values_df = pd.read_csv(
    "simulated_data/simulation_all_data.csv"
)
column_name = [
    "Predict_Value_" + str(i + 1) for i in range(pre_values.shape[2])
]
predict_df = pd.DataFrame(columns=column_name)

for i in range(pre_values.shape[0]):
    temp_df = pd.DataFrame(pre_values[i])
    column_name = [
        "Predict_Value_" + str(i + 1) for i in range(pre_values.shape[2])
    ]
    temp_df.columns = column_name
    predict_df = pd.concat([predict_df, temp_df], axis=0)
predict_df.index = all_values_df.index

all_info_df = pd.concat([all_values_df, predict_df], axis=1)
row_column_name = [
    "data",
    "time",
    "real_data_without_mask",
    "real_data_unavailable",
    "real_data_undetectable",
    "ID",
    "species",
]


def row_function_two_points(row):
    new_row = row.to_frame().T.copy()
    mask_info = new_row.loc[:, [i.startswith("Mask") for i in new_row.columns]]
    value_info = new_row.loc[:,
                             [i.startswith("Value") for i in new_row.columns]]
    predict_value_info = new_row.loc[:, [
        i.startswith("Predict") for i in new_row.columns
    ]]
    temp_df = pd.DataFrame(columns=row_column_name)
    for i, mask in enumerate(mask_info.values[0]):
        # print(new_row.loc[0,mask])
        if mask == -1:
            temp_df = temp_df.append(
                {
                    "data": predict_value_info.iloc[0, i],
                    "time": new_row["Time"].values[0],
                    "real_data_unavailable": value_info.iloc[0, i],
                    "ID": new_row["ID"].values[0],
                    "species": i,
                },
                ignore_index=True,
            )
        elif mask == 0:
            temp_df = temp_df.append(
                {
                    "data": predict_value_info.iloc[0, i],
                    "time": new_row["Time"].values[0],
                    "real_data_undetectable": value_info.iloc[0, i],
                    "ID": new_row["ID"].values[0],
                    "species": i,
                },
                ignore_index=True,
            )
        else:
            temp_df = temp_df.append(
                {
                    "data": predict_value_info.iloc[0, i],
                    "time": new_row["Time"].values[0],
                    "real_data_without_mask": value_info.iloc[0, i],
                    "ID": new_row["ID"].values[0],
                    "species": i,
                },
                ignore_index=True,
            )
    return temp_df


final_df = pd.DataFrame(columns=row_column_name)
print("There are {} lines to be processed".format(all_info_df.shape[0]))
for i in range(all_info_df.shape[0]):
    if i % 1000 == 0:
        print("Processing the {}-th line".format(i))
    temp_df = row_function_two_points(all_info_df.loc[i])
    final_df = pd.concat([final_df, temp_df], axis=0)
result_file = "real_data/{}_dl_interpolation.csv".format(model_name)
final_df.to_csv(result_file, index=False)
